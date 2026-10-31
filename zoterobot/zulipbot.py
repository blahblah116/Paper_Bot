"""Zulip side of zoterobot: DM commands and chat through the same bot account paperbot posts with.
zoterobot의 Zulip 쪽 — paperbot이 글을 올리는 바로 그 봇 계정으로 DM을 받아 저장 명령·질문을 처리한다.

- Slack과 달리 스레드 멘션이 아니라 DM이 입구다: `저장 <컬렉션> <링크>`(또는 논문 topic 인용/메시지 링크) → 저장,
  그 밖의 DM → 챗봇. 스트림 메시지는 받지 않는다(이벤트 큐를 DM으로 좁힌다).
- ZulipBot은 SlackBot과 같은 다섯 메서드(reply/update/react/parent_text/allowed)를 제공해 핸들러가 플랫폼을 모르게 한다.
  이벤트 dict 모양도 Slack과 맞춘다: channel("zulip:dm:<ids>"), ts(메시지 id), user(이메일), user_id, text.
- 인증: ~/.paperbot_env의 ZULIP_EMAIL/ZULIP_API_KEY/ZULIP_SITE, 없으면 zuliprc(zulip.config_file) — paperbot과 공유.
  자기 자신이 보낸 메시지(paperbot 발행분 포함)는 버린다.
- 메시지 수정(update)은 realm의 편집 제한 시간을 넘기면 실패한다 → 핸들러의 예비 경로(새 메시지)가 그대로 동작.
"""
from __future__ import annotations

import logging
import threading
from typing import Callable

from paperbot.publisher import make_zulip_client, split_zulip_message, zulip_auth_error, zulip_math_spacing

from .messages import ZULIP

logger = logging.getLogger(__name__)

_CHANNEL_PREFIX = "zulip:dm:"
# Slack 이모지 이름 → Zulip 이모지 이름 (핸들러는 Slack 이름으로 부른다)
_EMOJI = {"eyes": "eyes", "white_check_mark": "check", "x": "cross_mark"}


class ZulipBot:
    fmt = ZULIP

    def __init__(self, client, allowed_users: list[str], bot_user_id: int | None = None,
                 bot_email: str = "", update_every: float = 5.0):
        self.client = client                 # zulip.Client (테스트에서는 가짜)
        self.allowed_users = [u.strip().lower() for u in allowed_users if u.strip()]
        self.bot_user_id = bot_user_id
        self.bot_email = bot_email
        self.update_every = update_every     # 스트리밍 갱신 최소 간격(초) — 편집마다 이력이 남아 Slack보다 띄엄띄엄
        self._thread: threading.Thread | None = None

    # ---------------------------------------------------------------- 이벤트 → ev dict

    def to_event(self, msg: dict) -> dict | None:
        """Zulip message 이벤트 → 핸들러용 ev dict. 봇 자신의 메시지나 DM이 아닌 것은 None."""
        if msg.get("type") != "private":
            return None
        if self.bot_user_id is not None and msg.get("sender_id") == self.bot_user_id:
            return None
        recipients = sorted({int(r["id"]) for r in (msg.get("display_recipient") or [])
                             if isinstance(r, dict) and r.get("id") != self.bot_user_id})
        if not recipients:
            recipients = [int(msg["sender_id"])]
        return {
            "platform": "zulip",
            "type": "message",
            "channel_type": "im",
            "channel": _CHANNEL_PREFIX + ",".join(str(i) for i in recipients),
            "ts": str(msg["id"]),
            "user": str(msg.get("sender_email", "")).lower(),
            "user_id": int(msg["sender_id"]),
            "recipients": recipients,
            "text": msg.get("content", "") or "",
        }

    def run_forever(self, callback: Callable[[dict], None]) -> None:
        """이벤트 큐 롱폴링(끊기면 라이브러리가 재등록). 콜백 예외는 여기서 막는다 — 안 막으면 루프가 죽는다."""
        def on_event(event: dict) -> None:
            if event.get("type") != "message":
                return
            try:
                callback(event["message"])
            except Exception:  # noqa: BLE001
                logger.exception("Zulip 이벤트 처리 중 예외")

        self.client.call_on_each_event(on_event, event_types=["message"], narrow=[["is", "private"]])

    def start(self, callback: Callable[[dict], None]) -> None:
        self._thread = threading.Thread(target=self.run_forever, args=(callback,), name="zulip-events", daemon=True)
        self._thread.start()

    # ---------------------------------------------------------------- 핸들러 인터페이스 (SlackBot과 동일)

    @staticmethod
    def _recipients_of(channel: str) -> list[int]:
        ids = channel[len(_CHANNEL_PREFIX):] if channel.startswith(_CHANNEL_PREFIX) else ""
        return [int(i) for i in ids.split(",") if i.strip().isdigit()]

    def _send(self, to: list[int], content: str) -> int:
        res = self.client.send_message({"type": "private", "to": to, "content": content})
        if res.get("result") != "success":
            raise RuntimeError(f"Zulip 전송 실패: {res.get('msg', res)}")
        return int(res["id"])

    def reply(self, ev: dict, text: str, thread: bool = True) -> dict:
        """DM 답장 (thread 인자는 Slack 호환용 — Zulip DM에는 스레드가 없다). 10,000자 초과면 나눠 보낸다."""
        to = ev.get("recipients") or self._recipients_of(ev["channel"]) or [int(ev["user_id"])]
        first: int | None = None
        for chunk in split_zulip_message(zulip_math_spacing(text)) or [""]:
            mid = self._send(to, chunk)
            first = first if first is not None else mid
        return {"ts": str(first), "channel": ev["channel"]}

    def update(self, channel: str, ts: str, text: str) -> None:
        chunks = split_zulip_message(zulip_math_spacing(text)) or [""]
        res = self.client.update_message({"message_id": int(ts), "content": chunks[0]})
        if res.get("result") != "success":
            raise RuntimeError(f"Zulip 메시지 갱신 실패: {res.get('msg', res)}")
        for extra in chunks[1:]:    # 넘치는 부분은 같은 DM에 이어서
            self._send(self._recipients_of(channel), extra)

    def react(self, ev: dict, name: str, remove: bool = False) -> None:
        try:
            fn = self.client.remove_reaction if remove else self.client.add_reaction
            res = fn({"message_id": int(ev["ts"]), "emoji_name": _EMOJI.get(name, name)})
            if res.get("result") != "success":
                logger.debug("reaction %s 실패: %s", name, res.get("msg"))
        except Exception as e:  # noqa: BLE001 — 비치명적
            logger.debug("reaction %s 실패: %s", name, e)

    def fetch_message(self, message_id: int) -> str:
        """메시지 id → 원문(마크다운). 봇이 볼 수 있는 메시지(공개 스트림·구독 스트림·자기 DM)만."""
        res = self.client.get_messages({"anchor": message_id, "num_before": 0, "num_after": 0,
                                        "narrow": [{"operator": "id", "operand": str(message_id)}],
                                        "apply_markdown": False})
        if res.get("result") != "success":
            raise RuntimeError(f"Zulip 메시지 조회 실패 (id={message_id}): {res.get('msg', res)}")
        msgs = [m for m in res.get("messages") or [] if int(m.get("id", -1)) == int(message_id)]
        if not msgs:
            raise RuntimeError(f"Zulip 메시지 {message_id}를 찾지 못했습니다 — 봇이 그 스트림을 볼 수 있는지 확인하세요.")
        return msgs[0].get("content", "") or ""

    def parent_text(self, ev: dict) -> str:
        """DM에 적힌 메시지 링크(/near/<id>)의 원문 — 리스너가 ev['thread_ts']에 그 id를 넣어 둔다."""
        mid = ev.get("thread_ts")
        return self.fetch_message(int(mid)) if mid and str(mid).isdigit() else ""

    def allowed(self, ev: dict, thread: bool) -> bool:
        user, uid = ev.get("user", ""), str(ev.get("user_id", ""))
        if user in self.allowed_users or uid in self.allowed_users:
            return True
        logger.warning("허용되지 않은 Zulip 사용자 %s(%s) 거절 (허용하려면 zoterobot/config.yaml zulip.allowed_users에 추가)",
                       user, uid)
        self.reply(ev, f"이 봇은 허용된 사용자만 쓸 수 있습니다. (사용자: `{user}`, id `{uid}`)", thread=thread)
        return False


def connect(config_file: str, allowed_users: list[str], update_every: float = 5.0) -> ZulipBot:
    """환경변수(ZULIP_EMAIL/ZULIP_API_KEY/ZULIP_SITE) 또는 zuliprc로 접속해 봇 프로필을 확인한 ZulipBot.
    인증 정보·접속 문제는 RuntimeError."""
    client = make_zulip_client(config_file)
    profile = client.get_profile()
    if profile.get("result") != "success":
        raise RuntimeError(zulip_auth_error(profile))
    bot = ZulipBot(client, allowed_users, bot_user_id=int(profile["user_id"]),
                   bot_email=str(profile.get("email", "")), update_every=update_every)
    logger.info("Zulip 봇 %s (id %s) 인증 OK — 허용 사용자 %d명", bot.bot_email, bot.bot_user_id, len(bot.allowed_users))
    return bot
