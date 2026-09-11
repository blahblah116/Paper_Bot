#!/usr/bin/env python3
"""Paper Summary Bot v2 — 엔트리포인트 (cron이 호출).

흐름: config 로드 → topic별 각 소스 fetch → uid dedupe + store 미처리분 필터
→ relevance 필터(임베딩 → LLM judge) → 논문별 (extract → summarize → publish
→ store 기록). 논문 하나가 실패해도 전체는 계속 진행한다.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import replace
from logging.handlers import RotatingFileHandler

from paperbot import summarizer
from paperbot.config import Config, ConfigError, load_config
from paperbot.extractor import extract
from paperbot.filter import RelevanceFilter
from paperbot.models import Paper
from paperbot.publisher import PublishConfigError, make_publisher
from paperbot.sources import ArxivSource, SemanticScholarSource
from paperbot.store import Store
from paperbot.summarizer import LLMError

logger = logging.getLogger("paperbot")


# ---------------------------------------------------------------- CLI/로깅/락

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paper Summary Bot v2")
    p.add_argument("--config", default=None,
                   help="config.yaml 경로 (기본: run.py 옆의 config.yaml)")
    p.add_argument("--dry-run", action="store_true",
                   help="config를 무시하고 Zulip 전송 대신 stdout 출력")
    p.add_argument("--limit", type=int, default=None,
                   help="처리할 논문 수 제한 (테스트용)")
    p.add_argument("--topic", default=None,
                   help="특정 topic만 처리 (config의 name)")
    p.add_argument("--backfill", action="store_true",
                   help="lookback 필터를 끄고 검색 결과 전부 처리 (첫 실행용)")
    p.add_argument("--skip-filter", action="store_true",
                   help="relevance 필터 없이 전부 요약 (디버그용)")
    return p.parse_args(argv)


def setup_logging(cfg: Config) -> None:
    log_path = cfg.resolve("logs/paperbot.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    file_handler = RotatingFileHandler(
        log_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(file_handler)
    root.addHandler(console)


def _lock_path(cfg: Config) -> str:
    data_dir = os.path.dirname(cfg.resolve(cfg.storage.db_path))
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, ".lock")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def acquire_lock(path: str) -> bool:
    """중복 실행 방지 lockfile 획득. 이미 살아있는 실행이 있으면 False."""
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                old_pid = int((f.read().strip() or "0"))
        except (ValueError, OSError):
            old_pid = 0
        if old_pid and _pid_alive(old_pid):
            logger.warning("이미 실행 중(pid=%s). 종료합니다.", old_pid)
            return False
        logger.warning("오래된 lockfile 발견(pid=%s, 미실행). 회수합니다.", old_pid)

    with open(path, "w") as f:
        f.write(str(os.getpid()))
    return True


def release_lock(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


# ---------------------------------------------------------------- 파이프라인

def fetch_all(cfg: Config, backfill: bool) -> list[Paper]:
    """모든 topic × 소스를 검색하고 uid로 dedupe한 리스트를 반환.

    S2가 주 소스이므로 topic마다 S2 → arXiv 순으로 수집한다
    (중복 시 메타데이터가 풍부한 S2 레코드가 남는다).
    소스 하나가 실패해도 나머지는 계속 진행한다.
    """
    sources = [SemanticScholarSource(cfg, backfill=backfill), ArxivSource(cfg, backfill=backfill)]
    seen: set[str] = set()
    out: list[Paper] = []

    for topic in cfg.topics:
        for source in sources:
            try:
                papers = source.fetch(topic)
            except Exception as e:  # noqa: BLE001 — 소스 단위 실패 격리
                logger.error("%s '%s' 검색 실패: %s", source.name, topic.name, e)
                continue
            for p in papers:
                if p.uid in seen:
                    continue
                seen.add(p.uid)
                out.append(p)

    logger.info("검색 완료: 고유 논문 %d편", len(out))
    return out


def apply_quota(papers: list[Paper], quota) -> list[Paper]:
    """topic(채널)×소스별 발행 상한 적용. 0이면 해당 소스 무제한.

    papers는 fetch 순서(소스별 최신순)를 유지하므로 상한 내에서 최신 논문이 선택된다.
    초과분은 DB에 기록되지 않아 다음 실행 때 자연스럽게 이월(carry-over)된다.
    """
    if quota.s2_per_topic <= 0 and quota.arxiv_per_topic <= 0:
        return papers
    counts: dict[tuple[str, str], int] = {}
    out: list[Paper] = []
    for p in papers:
        cap = quota.s2_per_topic if p.source == "semantic_scholar" else quota.arxiv_per_topic
        if cap <= 0:
            out.append(p)
            continue
        key = (p.topic_name, p.source)
        if counts.get(key, 0) < cap:
            counts[key] = counts.get(key, 0) + 1
            out.append(p)
    return out


def process_paper(cfg, store, publisher, paper, client) -> bool:
    """단일 논문: extract → summarize → publish → store. 성공 시 True."""
    try:
        extracted = extract(cfg, paper)
        # abstract도 본문도 없으면 요약할 재료가 없다 — 무의미한(환각 위험) 요약 대신 skip.
        # failed로 기록해 다음 실행 때 1회 재시도(S2가 abstract를 뒤늦게 채우는 경우 있음),
        # 그래도 없으면 영구 skip.
        if extracted.abstract_only and not paper.abstract:
            logger.warning("본문·abstract 모두 확보 실패 — 발행하지 않음: %s", paper.uid)
            store.mark_failed(paper, "본문/abstract 모두 없음")
            return False
        summary = summarizer.summarize(cfg, paper, extracted, client=client)
        publisher.publish(paper, summary, abstract_only=extracted.abstract_only,
                          figures=extracted.figures)
        store.mark_done(paper)
        return True
    except PublishConfigError:
        raise  # 설정 문제 — 논문을 failed로 소모하지 않고 상위에서 실행 중단
    except LLMError as e:
        logger.error("LLM 실패로 skip: %s (%s)", paper.uid, e)
        err = str(e)
    except Exception as e:  # noqa: BLE001 — 개별 논문 실패 격리
        logger.exception("논문 처리 실패: %s (%s)", paper.uid, e)
        err = str(e)
    store.mark_failed(paper, err)
    return False


def run(cfg: Config, args: argparse.Namespace) -> int:
    # --topic 필터: config를 축소.
    if args.topic:
        topics = [t for t in cfg.topics if t.name == args.topic]
        if not topics:
            logger.error("--topic '%s'에 해당하는 topic이 config에 없습니다.", args.topic)
            return 2
        cfg = replace(cfg, topics=topics)

    try:
        # fetch/LLM에 시간을 쓰기 전에 발행 설정(토큰 등)부터 검증 — 실패 시 즉시 종료.
        publisher = make_publisher(cfg, force_dry_run=args.dry_run)
    except PublishConfigError as e:
        logger.error("발행 설정 오류: %s", e)
        return 1
    client = summarizer.make_client(cfg)
    topics_by_name = {t.name: t for t in cfg.topics}

    fetched = fetch_all(cfg, backfill=args.backfill)

    with Store(cfg.resolve(cfg.storage.db_path)) as store:
        todo = store.filter_unprocessed(fetched)
        logger.info("미처리 논문 %d편 (검색 %d편 중)", len(todo), len(fetched))

        # relevance 필터.
        if args.skip_filter:
            logger.info("--skip-filter: relevance 필터 생략")
            passed = todo
        else:
            flt = RelevanceFilter(cfg, judge_client=client)
            passed = flt.apply(todo, topics_by_name, store)
        n_passed = len(passed)
        logger.info("필터 통과 %d편 (미처리 %d편 중)", n_passed, len(todo))

        # topic(채널)×소스별 일일 발행 쿼터 — 초과분은 다음 실행으로 이월.
        before_quota = len(passed)
        passed = apply_quota(passed, cfg.quota)
        if len(passed) < before_quota:
            logger.info("발행 쿼터 적용: %d편 → %d편 (초과 %d편은 다음 실행으로 이월)",
                        before_quota, len(passed), before_quota - len(passed))

        if args.limit is not None:
            passed = passed[: args.limit]
            logger.info("--limit 적용: %d편만 처리", len(passed))

        done = failed = 0
        config_error: PublishConfigError | None = None
        for i, paper in enumerate(passed, start=1):
            logger.info("[%d/%d] %s — %.80s", i, len(passed), paper.uid, paper.title)
            try:
                ok = process_paper(cfg, store, publisher, paper, client)
            except PublishConfigError as e:
                config_error = e
                logger.error(
                    "전송 설정 오류로 실행 중단: %s — 이 논문과 남은 논문은 "
                    "failed로 기록하지 않으며, 설정을 고치면 다음 실행 때 처리됩니다.", e,
                )
                break
            if ok:
                done += 1
            else:
                failed += 1

    publisher.close()
    if todo:  # LLM(judge/요약)을 실제로 썼을 때만 — GPU 즉시 반납
        summarizer.unload_model(cfg)
    logger.info(
        "완료: 검색 %d / 미처리 %d / 필터 통과 %d / 요약 완료 %d / 실패 %d",
        len(fetched), len(todo), n_passed, done, failed,
    )
    return 1 if config_error else 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    config_path = args.config or os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    try:
        cfg = load_config(config_path)
    except ConfigError as e:
        print(f"[설정 오류] {e}", file=sys.stderr)
        return 2

    setup_logging(cfg)

    lock = _lock_path(cfg)
    if not acquire_lock(lock):
        return 0
    try:
        return run(cfg, args)
    finally:
        release_lock(lock)


if __name__ == "__main__":
    raise SystemExit(main())
