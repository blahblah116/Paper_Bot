"""Listener: routes Slack (Socket Mode) and Zulip (event queue) messages to the handlers; background library sync.
리스너 — Slack(Socket Mode)·Zulip(이벤트 큐) 메시지를 핸들러로 라우팅하고 라이브러리 동기화를 백그라운드로 돌린다.

- Slack: 채널 멘션(app_mention) → 저장 큐, DM(message.im) → 챗 큐. 큐·워커 스레드가 분리돼 있어
  DM 답변(LLM, 느림)이 저장 명령을 막지 않는다. DM 안의 멘션은 message.im으로 이미 받으므로 무시.
- Zulip: DM만 받는다. 첫 단어가 '저장/save'면 저장 큐(링크·인용·메시지 링크에서 논문 식별), 나머지는 챗 큐.
  두 플랫폼이 같은 큐·핸들러를 쓰므로 LLM 작업은 여전히 한 번에 하나만 돈다.
- 핸들러가 받는 bot은 MultiBot — ev["platform"]으로 SlackBot/ZulipBot을 고른다.
- 봇 자신에게 온 이벤트만 처리한다(앱에 '사용자 이벤트'가 구독돼 있으면 남의 DM까지 들어오므로 버림).
- 동기화(sync_library)는 DM이 올 때(마지막 확인 후 sync_interval_min 경과 시)와 저장 직후 백그라운드로
  한 번에 하나만 돈다. DM은 기다리지 않고 현재 캐시로 답한다(Drive 응답이 2~40초로 들쭉날쭉 — 실측).
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque

from .agent import Agent
from .config import Config
from .drive import DriveUploader
from .handlers import ChatHandler, SaveHandler
from .messages import esc, find_refs, parse_zulip_dm
from .slackbot import SlackBot
from .sync import sync_library
from .textcache import TextCache
from .zotero import ZoteroClient
from .zulipbot import ZulipBot

logger = logging.getLogger("zoterobot")


class MultiBot:
    """플랫폼별 봇(SlackBot/ZulipBot)을 ev["platform"]으로 고르는 얇은 디스패처 — 핸들러는 플랫폼을 모른다.
    update(channel, ts, …)는 ev가 없으므로 채널 키("zulip:dm:…")로 가른다."""

    def __init__(self, bots: dict):
        if not bots:
            raise ValueError("봇이 하나도 없습니다 (Slack 토큰 또는 zulip.enabled 필요)")
        self.bots = bots
        self.default = "slack" if "slack" in bots else next(iter(bots))

    def _for(self, ev: dict):
        return self.bots.get(ev.get("platform", "slack")) or self.bots[self.default]

    def _for_channel(self, channel: str):
        key = "zulip" if str(channel).startswith("zulip:") else "slack"
        return self.bots.get(key) or self.bots[self.default]

    def reply(self, ev: dict, text: str, thread: bool = True) -> dict:
        return self._for(ev).reply(ev, text, thread=thread)

    def update(self, channel: str, ts: str, text: str) -> None:
        self._for_channel(channel).update(channel, ts, text)

    def react(self, ev: dict, name: str, remove: bool = False) -> None:
        self._for(ev).react(ev, name, remove=remove)

    def parent_text(self, ev: dict) -> str:
        return self._for(ev).parent_text(ev)

    def allowed(self, ev: dict, thread: bool) -> bool:
        return self._for(ev).allowed(ev, thread)

    def fmt(self, ev: dict):
        return self._for(ev).fmt

    def update_every(self, ev: dict) -> float:
        return self._for(ev).update_every


class Listener:
    def __init__(self, cfg: Config, web, zot: ZoteroClient, drive: DriveUploader,
                 cache: TextCache | None = None, agent: Agent | None = None, bot_user_id: str = "",
                 zulip: ZulipBot | None = None):
        self.cfg, self.zot, self.drive, self.cache = cfg, zot, drive, cache
        self.bot_user_id = bot_user_id
        bots: dict = {}
        if web is not None:
            bots["slack"] = SlackBot(web, cfg.allowed_users)
        if zulip is not None:
            bots["zulip"] = zulip
        self.zulip = zulip
        self.bot = MultiBot(bots)
        self.save = SaveHandler(cfg, self.bot, zot, drive, on_saved=self._after_save)
        self.chat = ChatHandler(cfg, self.bot, cache, agent, on_question=self.kick_sync)
        self.jobs: queue.Queue = queue.Queue()      # 저장 명령 (빠름)
        self.chats: queue.Queue = queue.Queue()     # DM (LLM — 느림, 저장을 막지 않게 분리)
        self._seen: deque[str] = deque(maxlen=500)  # Slack 재전송 이벤트 중복 방지
        self._sync_running = threading.Lock()
        self._warned_user_event = False
        self.draining = threading.Event()           # 종료 중: 새 요청은 거절하고 처리 중인 것만 끝낸다
        self._in_flight = 0
        self._in_lock = threading.Lock()

    # ---------------------------------------------------------------- 종료 대기

    def drain(self, timeout_s: float) -> bool:
        """새 요청을 막고, 큐와 처리 중인 작업이 비워질 때까지 기다린다. 시간 안에 끝나면 True."""
        self.draining.set()
        t0 = time.time()
        last_log = 0.0
        while True:
            with self._in_lock:
                busy = self._in_flight
            if busy == 0 and self.jobs.empty() and self.chats.empty():
                return True
            if time.time() - t0 >= timeout_s:
                return False
            if time.time() - last_log >= 30:
                logger.info("종료 대기 중 — 처리 중 %d건, 대기열 저장 %d·DM %d (%.0fs 경과)",
                            busy, self.jobs.qsize(), self.chats.qsize(), time.time() - t0)
                last_log = time.time()
            time.sleep(0.5)

    def mark_interrupted(self) -> None:
        self.chat.mark_interrupted()

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
        wanted = (ev.get("type") == "message" and ev.get("channel_type") == "im") or \
                 (ev.get("type") == "app_mention" and not is_dm)   # DM 안의 멘션은 message.im으로 이미 받음
        if not wanted:
            return
        self._seen.append(event_id)
        if self.draining.is_set():
            try:
                self.bot.reply(ev, "🔄 봇이 재시작 중입니다 — 1분쯤 뒤에 다시 보내 주세요.", thread=not is_dm)
            except Exception:  # noqa: BLE001
                pass
            return
        (self.chats if is_dm else self.jobs).put(ev)

    def on_zulip_message(self, msg: dict) -> None:
        """Zulip 이벤트 스레드 콜백 — DM만: '저장 …'은 저장 큐, 나머지는 챗 큐. 봇 자신의 메시지는 버린다."""
        if self.zulip is None:
            return
        ev = self.zulip.to_event(msg)
        if ev is None:
            return
        cmd = parse_zulip_dm(ev["text"])
        logger.info("Zulip DM 수신 user=%s kind=%s", ev.get("user"), cmd.kind)
        if self.draining.is_set():
            try:
                self.bot.reply(ev, "🔄 봇이 재시작 중입니다 — 1분쯤 뒤에 다시 보내 주세요.", thread=False)
            except Exception:  # noqa: BLE001
                pass
            return
        ev["text"] = cmd.text
        if cmd.kind == "save":
            if cmd.message_ids and not find_refs(cmd.text):
                ev["thread_ts"] = str(cmd.message_ids[0])   # 식별자가 없으면 링크된 메시지(논문 원글)에서 찾는다
            self.jobs.put(ev)
        else:
            self.chats.put(ev)

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
            with self._in_lock:
                self._in_flight += 1
            try:
                handler(ev)
            except Exception as e:  # noqa: BLE001 — 한 명령 실패가 리스너를 죽이면 안 된다
                logger.exception("처리 중 예외")
                try:
                    self.bot.reply(ev, f"⚠️ 처리 중 오류: {esc(str(e))[:300]}", thread=ev.get("type") != "message")
                except Exception:  # noqa: BLE001
                    pass
            finally:
                with self._in_lock:
                    self._in_flight -= 1

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
        if self.cache is None:
            return False
        self.cache.refresh_if_changed()   # paperbot/articlebot이 수집 문서를 넣었으면 바로 반영 (쿼리 하나)
        if not self.cache.check_due(self.cfg.chat.sync_interval_min):
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
