"""Thin Slack Web API helper shared by the handlers.
핸들러가 공유하는 Slack Web API 헬퍼 — 답글·반응·메시지 갱신·스레드 원글 조회·사용자 허용.
"""
from __future__ import annotations

import logging

from .messages import SLACK

logger = logging.getLogger(__name__)


class SlackBot:
    fmt = SLACK
    update_every = 2.5                      # DM 답변 스트리밍 시 chat.update 최소 간격(초) — Slack rate limit 여유

    def __init__(self, web, allowed_users: list[str]):
        self.web = web                      # slack_sdk.WebClient (테스트에서는 가짜)
        self.allowed_users = allowed_users

    def reply(self, ev: dict, text: str, thread: bool = True) -> dict:
        kw = {"thread_ts": ev.get("thread_ts") or ev["ts"]} if thread else {}
        return self.web.chat_postMessage(channel=ev["channel"], text=text, unfurl_links=False,
                                         unfurl_media=False, **kw)

    def update(self, channel: str, ts: str, text: str) -> None:
        self.web.chat_update(channel=channel, ts=ts, text=text)

    def react(self, ev: dict, name: str, remove: bool = False) -> None:
        from slack_sdk.errors import SlackApiError

        try:
            fn = self.web.reactions_remove if remove else self.web.reactions_add
            fn(channel=ev["channel"], timestamp=ev["ts"], name=name)
        except SlackApiError as e:   # reactions:write 스코프 없음 등 — 비치명적
            logger.debug("reaction %s 실패: %s", name, e.response.get("error"))

    def parent_text(self, ev: dict) -> str:
        """스레드 원글 본문 (channels:history / groups:history 스코프 필요)."""
        from slack_sdk.errors import SlackApiError

        try:
            resp = self.web.conversations_replies(channel=ev["channel"], ts=ev["thread_ts"],
                                                  limit=1, inclusive=True)
        except SlackApiError as e:
            err = e.response.get("error")
            if err == "missing_scope":
                raise RuntimeError("스레드 원글을 읽을 권한이 없습니다 — Slack 앱에 "
                                   "channels:history(비공개 채널은 groups:history) 스코프를 추가하세요.") from e
            raise RuntimeError(f"스레드 원글 조회 실패: {err}") from e
        msgs = resp.get("messages") or []
        return msgs[0].get("text", "") if msgs else ""

    def allowed(self, ev: dict, thread: bool) -> bool:
        """허용 목록 확인. 아니면 사용자 ID를 알려주는 거절 답글을 보내고 False."""
        user = ev.get("user")
        if user in self.allowed_users:
            return True
        logger.warning("허용되지 않은 사용자 %s 거절 (허용하려면 zoterobot/config.yaml allowed_users에 추가)", user)
        self.reply(ev, f"이 봇은 허용된 사용자만 쓸 수 있습니다. (사용자 ID: `{user}`)", thread=thread)
        return False
