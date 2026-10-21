"""Socket Mode listener: routes Slack events to the handlers and runs library sync in the background.
Slack Socket Mode 리스너 — 이벤트를 핸들러로 라우팅하고 라이브러리 동기화를 백그라운드로 돌린다.

- 채널 멘션(app_mention) → 저장 큐, DM(message.im) → 챗 큐. 큐·워커 스레드가 분리돼 있어
  DM 답변(LLM, 느림)이 저장 명령을 막지 않는다. DM 안의 멘션은 message.im으로 이미 받으므로 무시.
- 봇 자신에게 온 이벤트만 처리한다(앱에 '사용자 이벤트'가 구독돼 있으면 남의 DM까지 들어오므로 버림).
- 동기화(sync_library)는 DM이 올 때(마지막 확인 후 sync_interval_min 경과 시)와 저장 직후 백그라운드로
  한 번에 하나만 돈다. DM은 기다리지 않고 현재 캐시로 답한다(Drive 응답이 2~40초로 들쭉날쭉 — 실측).
"""
from __future__ import annotations

import logging
import queue
import threading
from collections import deque

from .agent import Agent
from .config import Config
from .drive import DriveUploader
from .handlers import ChatHandler, SaveHandler
from .messages import esc
from .slackbot import SlackBot
from .sync import sync_library
from .textcache import TextCache
from .zotero import ZoteroClient

logger = logging.getLogger("zoterobot")


class Listener:
    def __init__(self, cfg: Config, web, zot: ZoteroClient, drive: DriveUploader,
                 cache: TextCache | None = None, agent: Agent | None = None, bot_user_id: str = ""):
        self.cfg, self.zot, self.drive, self.cache = cfg, zot, drive, cache
        self.bot_user_id = bot_user_id
        self.bot = SlackBot(web, cfg.allowed_users)
        self.save = SaveHandler(cfg, self.bot, zot, drive, on_saved=self._after_save)
        self.chat = ChatHandler(cfg, self.bot, cache, agent, on_question=self.kick_sync)
        self.jobs: queue.Queue = queue.Queue()      # 저장 명령 (빠름)
        self.chats: queue.Queue = queue.Queue()     # DM (LLM — 느림, 저장을 막지 않게 분리)
        self._seen: deque[str] = deque(maxlen=500)  # Slack 재전송 이벤트 중복 방지
        self._sync_running = threading.Lock()
        self._warned_user_event = False

    # ---------------------------------------------------------------- 라우팅

    def on_request(self, client, req) -> None:
        """Socket Mode 콜백 — 3초 안에 ack하고 실제 처리는 워커 스레드로 넘긴다."""
        from slack_sdk.socket_mode.response import SocketModeResponse

        client.send_socket_mode_response(SocketModeResponse(envelope_id=req.envelope_id))
        if req.type != "events_api":
            return
        ev = req.payload.get("event") or {}
        event_id = req.payload.get("event_id") or ev.get("ts")
        auths = req.payload.get("authorizations") or []
        auth_users = [a.get("user_id") for a in auths]
        logger.info("이벤트 수신 type=%s channel_type=%s app=%s 받는봇=%s (이 봇 토큰=%s)",
                    ev.get("type"), ev.get("channel_type"), req.payload.get("api_app_id"),
                    auth_users, self.bot_user_id)
        if self.bot_user_id and auth_users and self.bot_user_id not in auth_users and any(a.get("is_bot") for a in auths):
            logger.error("앱 토큰(xapp)과 봇 토큰(xoxb)이 서로 다른 Slack 앱의 것 — 이벤트는 봇 %s로 왔지만 "
                         "답장은 봇 %s로 보내게 됨. 같은 앱의 토큰 두 개를 쓰세요.", auth_users, self.bot_user_id)
        if not self._for_bot(auths):
            return
        if ev.get("bot_id") or ev.get("subtype") or event_id in self._seen:
            return
        is_dm = ev.get("channel_type") == "im" or str(ev.get("channel", "")).startswith("D")
        if ev.get("type") == "message" and ev.get("channel_type") == "im":
            self._seen.append(event_id)
            self.chats.put(ev)
        elif ev.get("type") == "app_mention" and not is_dm:   # DM 안의 멘션은 message.im으로 이미 받음
            self._seen.append(event_id)
            self.jobs.put(ev)

    def _for_bot(self, auths: list[dict]) -> bool:
        """봇 자신에게 온 이벤트만 처리. authorizations에 봇이 없는 이벤트는 내용을 보지 않고 버린다."""
        if not auths or any(a.get("is_bot") or a.get("user_id") == self.bot_user_id for a in auths):
            return True
        if not self._warned_user_event:
            self._warned_user_event = True
            logger.warning("봇이 아닌 사용자 권한으로 온 이벤트를 무시함 — Slack 앱 Event Subscriptions에서 "
                           "'events on behalf of users'를 지우고 'bot events'에만 등록하세요.")
        return False

    # ---------------------------------------------------------------- 워커

    def _work(self, q: queue.Queue, handler) -> None:
        while True:
            ev = q.get()
            try:
                handler(ev)
            except Exception as e:  # noqa: BLE001 — 한 명령 실패가 리스너를 죽이면 안 된다
                logger.exception("처리 중 예외")
                try:
                    self.bot.reply(ev, f"⚠️ 처리 중 오류: {esc(str(e))[:300]}", thread=ev.get("type") != "message")
                except Exception:  # noqa: BLE001
                    pass

    def worker(self) -> None:
        self._work(self.jobs, self.handle)

    def chat_worker(self) -> None:
        self._work(self.chats, self.handle_dm)

    def handle(self, ev: dict) -> None:
        self.save.handle(ev)

    def handle_dm(self, ev: dict) -> None:
        self.chat.handle(ev)

    # ---------------------------------------------------------------- 동기화

    def _after_save(self) -> None:
        if self.cache:
            self.cache.mark_stale()
            self.kick_sync()   # 저장 직후 바로 백그라운드 동기화 — 이어서 DM으로 물어도 반영되게

    def kick_sync(self) -> bool:
        """라이브러리 동기화가 필요하면 백그라운드 스레드로 시작 (이미 도는 중이면 생략). 시작했으면 True."""
        if self.cache is None or not self.cache.check_due(self.cfg.chat.sync_interval_min):
            return False
        if not self._sync_running.acquire(blocking=False):
            return False

        def run() -> None:
            try:
                sync_library(self.cache, self.zot, self.drive, self.cfg.library.link_base,
                             drive_scan=self.cfg.chat.drive_scan)
            except Exception as e:  # noqa: BLE001 — 동기화 실패는 다음 기회에 재시도, 기존 캐시 유지
                logger.warning("라이브러리 동기화 실패 — 기존 캐시 유지: %s", e)
            finally:
                self._sync_running.release()

        threading.Thread(target=run, name="library-sync", daemon=True).start()
        return True
