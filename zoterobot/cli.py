"""Command line entry for zoterobot (`python -m zoterobot`): run the listener, or one-off checks.
zoterobot 명령줄 진입점 — 리스너 실행, 또는 Slack 없이 점검·저장·동기화·질문.

  python -m zoterobot                      리스너 (scripts/zoterobot.sh가 tmux로 띄움)
  python -m zoterobot --check              Slack·Zotero·Drive 인증 + 링크 경로 형식 점검
  python -m zoterobot --save GDL 2505.13631   한 편 저장
  python -m zoterobot --sync [--force]     라이브러리 본문 캐시 동기화
  python -m zoterobot --ask "질문"         챗봇에 질문 (도구 호출 과정 출력)
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
from logging.handlers import RotatingFileHandler

from .agent import Agent, LLMError
from .config import Config, ConfigError, load_config
from .drive import DriveError, DriveUploader
from .gpu import GpuBusy, ensure_gpu
from .library import save_paper
from .listener import Listener
from .messages import format_result
from .resolver import clean_slack_text, first_ref, resolve
from .sync import sync_library
from .textcache import TextCache
from .zotero import ZoteroClient, ZoteroError

logger = logging.getLogger("zoterobot")

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.dirname(_PKG_DIR)
NEEDED_SCOPES = ("chat:write", "app_mentions:read", "channels:history", "groups:history",
                 "im:history", "reactions:write")


def setup_logging() -> None:
    log_path = os.path.join(_REPO_DIR, "logs", "zoterobot.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    fh = RotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
    fh.setFormatter(fmt)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(fh)
    root.addHandler(console)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="zoterobot — Slack thread → Zotero/Drive save + DM library chatbot")
    p.add_argument("--config", default=os.path.join(_PKG_DIR, "config.yaml"), help="zoterobot config.yaml 경로")
    p.add_argument("--check", action="store_true", help="Slack·Zotero·Drive 인증 점검 후 종료")
    p.add_argument("--save", nargs=2, metavar=("COLLECTION", "PAPER"), help="Slack 없이 한 편 저장")
    p.add_argument("--sync", action="store_true", help="라이브러리 본문 캐시 동기화 후 종료")
    p.add_argument("--force", action="store_true", help="--sync 시 전부 다시 추출")
    p.add_argument("--ask", metavar="QUESTION", help="Slack 없이 챗봇에 질문")
    return p.parse_args(argv)


def slack_tokens() -> tuple[str, str]:
    """(봇 토큰 xoxb, 앱 토큰 xapp). zoterobot은 paperbot과 별도 Slack 앱이므로 ZOTEROBOT_* 우선,
    없으면 SLACK_* (paperbot과 같은 앱을 쓸 때). 두 토큰은 반드시 같은 앱의 것이어야 한다 —
    다르면 DM은 받는데 답장이 channel_not_found로 실패한다."""
    env = os.environ
    bot = (env.get("ZOTEROBOT_BOT_TOKEN") or env.get("SLACK_BOT_TOKEN") or "").strip()
    app = (env.get("ZOTEROBOT_APP_TOKEN") or env.get("SLACK_APP_TOKEN") or "").strip()
    return bot, app


def _missing_scopes(auth) -> list[str]:
    scopes = set((auth.headers.get("x-oauth-scopes") or "").split(","))
    return sorted(set(NEEDED_SCOPES) - scopes)


def check(cfg: Config, zot: ZoteroClient, drive: DriveUploader) -> int:
    """Zotero·Drive·Slack을 각각 점검하고 하나라도 실패하면 1."""
    from slack_sdk import WebClient

    rc = 0
    try:
        logger.info("Zotero OK — 컬렉션 %d개", len(zot.collections()))
        paths = zot.linked_file_paths()
        for p in paths[:3]:
            logger.info("기존 링크 첨부 경로 예: %s", p)
        if paths and not any(p.startswith(cfg.library.link_base) for p in paths):
            logger.warning("기존 링크 첨부가 '%s'로 시작하지 않음 — Zotero '링크 첨부 파일 기본 디렉토리'가 "
                           "Attanger 폴더로 설정돼 있는지 확인하세요.", cfg.library.link_base)
    except ZoteroError as e:
        logger.error("Zotero 점검 실패: %s", e)
        rc = 1
    try:
        drive.check()
        logger.info("Drive OK — %s", drive.remote)
    except DriveError as e:
        logger.error("Drive 점검 실패: %s", e)
        rc = 1
    bot, app = slack_tokens()
    if bot and app:
        auth = WebClient(token=bot).auth_test()
        missing = _missing_scopes(auth)
        logger.info("Slack 봇 %s (%s) 인증 OK — 빠진 스코프: %s", auth.get("user"), auth.get("user_id"),
                    ", ".join(missing) or "없음")
        if missing:
            rc = 1
    else:
        logger.warning("Slack 토큰 없음 — ZOTEROBOT_BOT_TOKEN(xoxb-)과 ZOTEROBOT_APP_TOKEN(xapp-)이 필요합니다.")
        rc = 1
    return rc


def serve(cfg: Config, zot: ZoteroClient, drive: DriveUploader, cache: TextCache, agent: Agent | None) -> int:
    """Socket Mode로 Slack에 연결하고 종료 신호까지 대기."""
    from slack_sdk import WebClient
    from slack_sdk.socket_mode import SocketModeClient

    bot_token, app_token = slack_tokens()
    if not bot_token or not app_token:
        logger.error("ZOTEROBOT_BOT_TOKEN(xoxb-)과 ZOTEROBOT_APP_TOKEN(xapp-, Socket Mode) 환경변수가 필요합니다.")
        return 2
    web = WebClient(token=bot_token)
    auth = web.auth_test()
    logger.info("Slack 봇 %s (%s) 인증 OK", auth.get("user"), auth.get("user_id"))
    missing = _missing_scopes(auth)
    if missing:
        logger.warning("봇 토큰에 없는 스코프: %s — Bot Token Scopes에 추가 후 앱 재설치 필요", ", ".join(missing))
    if not cfg.allowed_users:
        logger.warning("allowed_users가 비어 있음 — 모든 요청을 거절합니다(거절 답글에 사용자 ID 표시).")
    try:
        drive.check()
    except DriveError as e:
        logger.error("Drive 점검 실패 — 리스너는 띄우되 저장은 실패할 것: %s", e)

    cache.docs()   # 캐시를 미리 메모리에 — 첫 백그라운드 동기화 중에도 DM 검색이 막히지 않게
    listener = Listener(cfg, web, zot, drive, cache, agent, bot_user_id=auth.get("user_id", ""))
    sm = SocketModeClient(app_token=app_token, web_client=web)   # 끊기면 자동 재연결
    sm.socket_mode_request_listeners.append(listener.on_request)
    threading.Thread(target=listener.worker, name="save-worker", daemon=True).start()
    threading.Thread(target=listener.chat_worker, name="chat-worker", daemon=True).start()
    sm.connect()
    logger.info("Socket Mode 연결 — 멘션·DM 대기 중 (DM 챗봇 %s)", "켜짐" if agent else "꺼짐")

    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    stop.wait()
    logger.info("종료 신호 — 연결 닫는 중")
    sm.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        cfg = load_config(args.config)
    except ConfigError as e:
        print(f"[설정 오류] {e}", file=sys.stderr)
        return 2
    setup_logging()

    try:
        zot = ZoteroClient(os.environ.get("ZOTERO_API_KEY", "").strip(),
                           os.environ.get("ZOTERO_USER_ID", "").strip())
        drive = DriveUploader(cfg.library.drive_remote, cfg.library.rclone_bin)
    except (ZoteroError, DriveError) as e:
        logger.error("%s", e)
        return 2

    if args.check:
        return check(cfg, zot, drive)

    if args.save:
        coll, text = args.save
        ref = first_ref(clean_slack_text(text))
        if ref is None:
            logger.error("논문 식별자를 찾지 못했습니다: %s", text)
            return 2
        try:
            result = save_paper(cfg, zot, drive, resolve(ref), coll)
        except (ZoteroError, DriveError, RuntimeError) as e:
            logger.error("저장 실패: %s", e)
            return 1
        print(format_result(result, drive))
        return 0

    cache = TextCache(cfg.resolve(cfg.chat.text_db))
    if args.sync:
        print(sync_library(cache, zot, drive, cfg.library.link_base, force=args.force, drive_scan=cfg.chat.drive_scan))
        return 0
    agent = Agent(cfg.chat, cache) if cfg.chat.enabled else None

    if args.ask:
        if agent is None:
            logger.error("chat.enabled가 false입니다.")
            return 2
        try:
            ensure_gpu(cfg.chat)
            if cache.check_due(cfg.chat.sync_interval_min):
                sync_library(cache, zot, drive, cfg.library.link_base, drive_scan=cfg.chat.drive_scan)
            res = agent.run(args.ask, [], on_status=lambda s: print(s, flush=True))
        except (GpuBusy, LLMError) as e:
            logger.error("%s", e)
            return 1
        print("\n" + res.answer)
        print("\n[참고]", [cache.get(k).title for k in res.papers if cache.get(k)])
        print(f"[steps={res.steps} tools={len(res.tools)} {res.duration_s:.1f}s, load {res.load_s:.1f}s]")
        return 0

    return serve(cfg, zot, drive, cache, agent)
