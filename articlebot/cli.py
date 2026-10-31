"""Command line entry for articlebot (`python -m articlebot`): args, logging, lockfile → pipeline.run.
articlebot 명령줄 진입점 — 인자 파싱, 로깅, 중복 실행 방지 lockfile. 파이프라인 자체는 pipeline.py.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from logging.handlers import RotatingFileHandler

from .config import Config, ConfigError, load_config
from .lockfile import acquire_lock, release_lock
from .pipeline import run

logger = logging.getLogger("articlebot")

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.dirname(_PKG_DIR)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="기업 연구 블로그 요약 Slack 봇")
    p.add_argument("--config", default=None, help="config.yaml 경로 (기본: articlebot/config.yaml)")
    p.add_argument("--dry-run", action="store_true",
                   help="Slack 전송 대신 stdout + data/preview 저장. DB는 메모리 사본에서만 변경 (파일 불변)")
    p.add_argument("--source", action="append", default=None,
                   help="특정 소스만 처리 (config의 name, 여러 번 지정 가능)")
    p.add_argument("--limit", type=int, default=None, help="이번 실행의 발행 수 상한 (테스트용)")
    p.add_argument("--lookback-days", type=int, default=None, help="config의 fetch.lookback_days 대신 사용")
    p.add_argument("--skip-judge", action="store_true", help="judge 없이 전부 백로그로 (디버그용)")
    p.add_argument("--list", action="store_true", help="기관별 백로그·미채점 신규 글 현황만 출력 (LLM·발행 없음)")
    p.add_argument("--ingest-only", action="store_true", help="수집·채점·백로그 적재까지만 (발행 없음)")
    p.add_argument("--force-ingest", action="store_true",
                   help="수집 주기(fetch.interval_days)와 무관하게 이번 실행에서 수집")
    p.add_argument("--rejudge", action="store_true",
                   help="채점 기준(judge.interest/threshold)을 바꾼 뒤: 피드에 남아 있는 대기·탈락 글을 다시 채점")
    p.add_argument("--mark-seen", action="store_true",
                   help="현재 피드의 미처리 글을 채점·발행 없이 '본 것'으로 기록")
    p.add_argument("--url", default=None, help="피드·백로그와 무관하게 이 URL 하나만 요약·발행")
    p.add_argument("--backfill-chat", action="store_true",
                   help="DB의 done 글 본문을 다시 받아 저장된 요약과 함께 챗봇 캐시(storage.chat_text_db)에 채움 (LLM·발행 없음, --limit 적용)")
    return p.parse_args(argv)


def setup_logging(cfg: Config) -> None:
    log_path = os.path.join(_REPO_DIR, "logs", "articlebot.log")
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
    for noisy in ("httpx", "httpx2", "openai", "urllib3", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config or os.path.join(_PKG_DIR, "config.yaml")
    try:
        cfg = load_config(config_path)
    except ConfigError as e:
        print(f"[설정 오류] {e}", file=sys.stderr)
        return 2
    setup_logging(cfg)

    lock = cfg.resolve(os.path.join(os.path.dirname(cfg.storage.db_path), ".lock"))
    if not acquire_lock(lock):
        return 0
    try:
        return run(cfg, args)
    except ConfigError as e:
        logger.error("설정 오류: %s", e)
        return 2
    finally:
        release_lock(lock)
