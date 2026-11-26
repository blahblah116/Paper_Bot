"""Zulip side of zoterobot: DM commands and chat through the same bot account paperbot posts with.
zoterobot의 Zulip 쪽 — paperbot이 글을 올리는 바로 그 봇 계정으로 DM을 받아 저장 명령·질문을 처리한다.

- Slack과 달리 스레드 멘션이 아니라 DM이 입구다: `저장 <컬렉션> <링크>`(또는 논문 topic 인용/메시지 링크) → 저장,
  그 밖의 DM → 챗봇. 스트림 메시지는 받지 않는다(이벤트 큐를 DM으로 좁힌다).
- ZulipBot은 SlackBot과 같은 다섯 메서드(reply/update/react/parent_text/allowed)를 제공해 핸들러가 플랫폼을 모르게 한다.
  이벤트 dict 모양도 Slack과 맞춘다: channel("zulip:dm:<ids>"), ts(메시지 id), user(이메일), user_id, text.
- 인증: ~/.paperbot_env의 ZULIP_EMAIL/ZULIP_API_KEY/ZULIP_SITE, 없으면 zuliprc(zulip.config_file) — paperbot과 공유.
  자기 자신이 보낸 메시지(paperbot 발행분 포함)는 버린다.
- 메시지 수정(update)은 realm의 편집 제한 시간을 넘기면 실패한다 → 핸들러의 예비 경로(새 메시지)가 그대로 동작.
- 'gpu …' DM은 챗봇을 타지 않고 gpu_bot의 webhook(GPU_BOT_WEBHOOK_URL)으로 넘긴다 — Zulip Outgoing webhook 모양의 JSON.
  넘기기 전에 원래 DM에 ⏳(GPU_PENDING_EMOJI)를 달고, 전달에 실패하면 지운다.
  허용 목록은 zulip.gpu_allowed_users("*"=모두, 키가 없으면 allowed_users)로 따로 본다(gpu_allowed).
"""
from __future__ import annotations

import json
import logging
import threading
import urllib.error
import urllib.request
from typing import Callable

from paperbot.publisher import make_zulip_client, split_zulip_message, zulip_auth_error, zulip_math_spacing

from .messages import ZULIP

logger = logging.getLogger(__name__)

_CHANNEL_PREFIX = "zulip:dm:"
# Slack 이모지 이름 → Zulip 이모지 이름 (핸들러는 Slack 이름으로 부른다)
_EMOJI = {"eyes": "eyes", "white_check_mark": "check", "x": "cross_mark"}
GPU_WEBHOOK_ENV = "GPU_BOT_WEBHOOK_URL"   # ~/.paperbot_env — 'gpu …' DM을 넘길 gpu_bot webhook. URL에 key가 들어 있다
# 'gpu …' DM 전달 중 원래 DM에 다는 ⏳ (Zulip 기본 emoji 이름 — emoji_codes.json에서 23f3의 canonical_name).
# gpu_bot이 답을 마치면 같은 이름으로 지우고 ✅(check)를 단다. 반응은 단 계정만 지울 수 있다.
GPU_PENDING_EMOJI = "time_ticking"


class ZulipBot:
    fmt = ZULIP

    def __init__(self, client, allowed_users: list[str], bot_user_id: int | None = None,
                 bot_email: str = "", update_every: float = 5.0, bot_full_name: str = "",
                 gpu_allowed_users: list[str] | None = None):
        self.client = client                 # zulip.Client (테스트에서는 가짜)
        self.allowed_users = [u.strip().lower() for u in allowed_users if u.strip()]
        # 'gpu …' DM 전용 허용 목록 — None(설정 키 없음)이면 allowed_users를 그대로 쓴다. "*"가 있으면 모두 허용
        self.gpu_allowed_users = (None if gpu_allowed_users is None
                                  else [u.strip().lower() for u in gpu_allowed_users if u.strip()])
        self.bot_user_id = bot_user_id
        self.bot_email = bot_email
        self.bot_full_name = bot_full_name
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

    def set_reaction(self, message_id: int, emoji_name: str, remove: bool = False) -> None:
        """메시지에 Zulip emoji 반응을 달거나(remove=True면 지우거나) 한다. emoji_name은 Zulip 이름 그대로.
        실패해도 예외를 내지 않고 warning log만 남긴다 — 반응 때문에 전달·다른 DM 처리가 멈추면 안 된다."""
        what = "제거" if remove else "추가"
        try:
            fn = self.client.remove_reaction if remove else self.client.add_reaction
            res = fn({"message_id": int(message_id), "emoji_name": emoji_name})
            if res.get("result") != "success":
                logger.warning("Zulip 반응 %s %s 실패 (메시지 %s): %s", emoji_name, what, message_id, res.get("msg", res))
        except Exception as e:  # noqa: BLE001 — 예외 문자열에 접속 정보가 섞일 수 있어 종류만 남긴다
            logger.warning("Zulip 반응 %s %s 실패 (메시지 %s): %s", emoji_name, what, message_id, type(e).__name__)

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
        return self._check(ev, self.allowed_users, "zulip.allowed_users", thread)

    def gpu_allowed(self, ev: dict) -> bool:
        """'gpu …' DM 허용 여부 — zulip.gpu_allowed_users("*"면 모두), 키가 없으면 allowed_users(이때 "*"는 특별하지 않다)."""
        if self.gpu_allowed_users is None:
            return self._check(ev, self.allowed_users, "zulip.gpu_allowed_users", thread=False)
        if "*" in self.gpu_allowed_users:
            return True
        return self._check(ev, self.gpu_allowed_users, "zulip.gpu_allowed_users", thread=False)

    def _check(self, ev: dict, users: list[str], key: str, thread: bool) -> bool:
        user, uid = ev.get("user", ""), str(ev.get("user_id", ""))
        if user in users or uid in users:
            return True
        logger.warning("허용되지 않은 Zulip 사용자 %s(%s) 거절 (허용하려면 zoterobot/config.yaml %s에 추가)",
                       user, uid, key)
        self.reply(ev, f"이 봇은 허용된 사용자만 쓸 수 있습니다. (사용자: `{user}`, id `{uid}`)", thread=thread)
        return False


# ---------------------------------------------------------------- 'gpu …' DM → gpu_bot webhook

def gpu_webhook_payload(msg: dict, bot_email: str, bot_full_name: str = "") -> dict:
    """Zulip message 이벤트 → Zulip Outgoing webhook 모양의 JSON. gpu_bot 쪽 필터는 bot_email과
    message.sender_email을 본다. token·API key 같은 인증 정보는 넣지 않는다."""
    content = msg.get("content", "") or ""
    return {
        "bot_email": bot_email,
        "bot_full_name": bot_full_name,
        "trigger": "direct_message",
        "data": content,
        "message": {
            "id": int(msg.get("id") or 0),
            "type": "private",
            "sender_id": int(msg.get("sender_id") or 0),
            "sender_email": str(msg.get("sender_email", "")),
            "sender_full_name": str(msg.get("sender_full_name", "")),
            "display_recipient": msg.get("display_recipient") or [],
            "content": content,
            "timestamp": int(msg.get("timestamp") or 0),
        },
    }


def post_gpu_webhook(url: str, payload: dict, timeout: float = 10.0) -> str | None:
    """payload를 gpu_bot webhook에 JSON POST. 2xx면 None, 아니면 사용자에게 보일 짧은 사유.
    URL에는 key가 들어 있다 — log·반환값에 URL이나 예외 문자열을 남기지 않고 상태 코드(실패면 예외 종류)만 쓴다."""
    try:
        req = urllib.request.Request(url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
    except urllib.error.HTTPError as e:
        status = e.code
        e.close()
    except Exception as e:  # noqa: BLE001 — 잘못된 URL의 ValueError처럼 예외 문자열에 URL이 섞일 수 있어 종류만 남긴다
        timed_out = isinstance(e, TimeoutError) or isinstance(getattr(e, "reason", None), TimeoutError)
        logger.warning("gpu_bot webhook 전달 실패: %s%s", type(e).__name__, " (timeout)" if timed_out else "")
        return "시간 초과" if timed_out else "연결 실패"
    logger.info("gpu_bot webhook 응답 HTTP %d", status)
    return None if 200 <= status < 300 else f"HTTP {status}"


def connect(config_file: str, allowed_users: list[str], update_every: float = 5.0,
            gpu_allowed_users: list[str] | None = None) -> ZulipBot:
    """환경변수(ZULIP_EMAIL/ZULIP_API_KEY/ZULIP_SITE) 또는 zuliprc로 접속해 봇 프로필을 확인한 ZulipBot.
    인증 정보·접속 문제는 RuntimeError."""
    client = make_zulip_client(config_file)
    profile = client.get_profile()
    if profile.get("result") != "success":
        raise RuntimeError(zulip_auth_error(profile))
    bot = ZulipBot(client, allowed_users, bot_user_id=int(profile["user_id"]),
                   bot_email=str(profile.get("email", "")), update_every=update_every,
                   bot_full_name=str(profile.get("full_name", "")), gpu_allowed_users=gpu_allowed_users)
    gpu = bot.gpu_allowed_users
    logger.info("Zulip 봇 %s (id %s) 인증 OK — 허용 사용자 %d명, gpu DM 허용 %s", bot.bot_email, bot.bot_user_id,
                len(bot.allowed_users), "allowed_users와 같음" if gpu is None else ("모두" if "*" in gpu else f"{len(gpu)}명"))
    return bot
