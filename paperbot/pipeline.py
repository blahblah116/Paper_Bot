"""One daily run of paperbot: fetch → dedupe → filter → quota → extract → summarize → publish.
paperbot 하루 1회 실행 파이프라인 (cli.py가 호출).

흐름: topic별 각 소스 fetch → uid dedupe + store 미처리분 필터 → relevance 필터(임베딩 → LLM judge)
→ pending 대기열 + 쿼터 → 논문별 (extract → summarize → publish → store 기록).
논문 하나가 실패해도 전체는 계속 진행하고, 전송 '설정' 오류만 실행 전체를 중단한다.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace

from . import chatdb, summarizer
from .config import Config
from .extractor import extract
from .filter import RelevanceFilter
from .models import Paper
from .publisher import PublishConfigError, make_publisher
from . import replay as _replay   # [REPLAY] Zulip 되올리기 (한시적) — replay.py 참고
from .sources import ArxivSource, SemanticScholarSource
from .store import Store
from .summarizer import LLMError

logger = logging.getLogger("paperbot")


@dataclass(frozen=True)
class RunOptions:
    """CLI 옵션 중 파이프라인이 보는 것."""
    dry_run: bool = False       # 전송 대신 stdout
    limit: int | None = None    # 처리할 논문 수 제한 (테스트용)
    topic: str | None = None    # 특정 topic만
    backfill: bool = False      # lookback 필터 해제 (첫 실행용)
    skip_filter: bool = False   # relevance 필터 생략 (디버그)


def fetch_all(cfg: Config, backfill: bool, store: Store | None = None) -> list[Paper]:
    """모든 topic × 소스를 검색하고 uid로 dedupe한 리스트를 반환.

    S2가 주 소스이므로 topic마다 S2 → arXiv 순으로 수집한다
    (중복 시 메타데이터가 풍부한 S2 레코드가 남는다).
    소스 하나가 실패해도 나머지는 계속 진행한다.
    store를 넘기면 S2 인용순 topic이 DB 기존 논문을 건너뛰고 페이지 커서를 저장/복원한다.
    """
    sources = [SemanticScholarSource(cfg, backfill=backfill, store=store),
               ArxivSource(cfg, backfill=backfill)]
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


def apply_quota(papers: list[Paper], quota, topics_by_name: dict | None = None) -> list[Paper]:
    """topic(채널)×소스별 발행 상한 적용. 0이면 해당 소스 무제한.

    입력 순서를 유지한 채 (topic, source)별 앞에서부터 상한만큼 고른다.
    호출자는 [pending 대기열(오래된 순)] + [신규 필터 통과분(최신순)] 순서로 넘기므로
    대기열이 먼저(FIFO) 소진되고, 남은 자리를 신규 최신 논문이 채운다.
    선택되지 않은 신규분은 호출자가 store.mark_pending()으로 대기열에 넣는다.
    topics_by_name을 주면 Topic.quota_s2 / quota_arxiv(topic별 오버라이드)가 전역 quota보다 우선한다.
    """
    topics_by_name = topics_by_name or {}

    def cap_for(p: Paper) -> int:
        t = topics_by_name.get(p.topic_name)
        if p.source == "semantic_scholar":
            return t.quota_s2 if t is not None and t.quota_s2 is not None else quota.s2_per_topic
        return t.quota_arxiv if t is not None and t.quota_arxiv is not None else quota.arxiv_per_topic

    counts: dict[tuple[str, str], int] = {}
    out: list[Paper] = []
    for p in papers:
        cap = cap_for(p)
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
        chatdb.store_paper(cfg, paper, extracted, summary)   # 챗봇 캐시에 전문·요약 (실패해도 발행은 끝났으니 경고만)
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


def _select(cfg: Config, store: Store, passed: list[Paper], limit: int | None,
            topics_by_name: dict, replay_items: list = ()) -> list[Paper]:   # [REPLAY] replay_items 인자만 지운다
    """pending 대기열 + 신규 통과분에 쿼터를 적용해 이번 실행 처리분을 고르고, 밀린 신규분은 pending으로.
    replay_items([REPLAY], Zulip 되올리기)가 있으면 그 topic의 신규는 빼고 --limit도 그만큼 줄여 센다."""
    expired = store.expire_pending(cfg.quota.pending_max_age_days)
    if expired:
        logger.info("pending 만료 폐기: %d편 (%d일 초과)", expired, cfg.quota.pending_max_age_days)
    pending = store.load_all_pending(list(topics_by_name))
    if pending:
        logger.info("pending 대기열 %d편 로드 (오래된 순 우선)", len(pending))

    # topic(채널)×소스별 일일 발행 쿼터 — 대기열 먼저, 남은 자리는 신규 최신순.
    candidates = pending + passed
    selected = apply_quota(candidates, cfg.quota, topics_by_name)
    selected = _replay.hold_new(selected, list(replay_items))   # [REPLAY] 되올리는 topic의 신규는 보류(아래에서 pending으로)
    if limit is not None:
        limit = max(0, limit - len(replay_items))   # [REPLAY] --limit은 되올리기부터 센다
        selected = selected[:limit]
        logger.info("--limit 적용: %d편만 처리", len(selected))
    selected_uids = {p.uid for p in selected}
    newly_pending = [p for p in passed if p.uid not in selected_uids]
    for p in newly_pending:
        store.mark_pending(p)
    if pending or newly_pending or len(selected) < len(candidates):
        counts = store.count_pending()
        logger.info("발행 쿼터 적용: 대기열 %d + 신규 %d → %d편 선택, 신규 %d편 pending 적재 (대기열 총 %d편%s)",
                    len(pending), len(passed), len(selected), len(newly_pending), sum(counts.values()),
                    "".join(f", {t}/{s.replace('semantic_scholar', 's2')}={n}" for (t, s), n in sorted(counts.items())))
    return selected


def run(cfg: Config, opts: RunOptions) -> int:
    """파이프라인 1회 실행. 종료 코드: 0 정상, 1 전송 설정 오류로 중단, 2 옵션 오류."""
    if opts.topic:
        topics = [t for t in cfg.topics if t.name == opts.topic]
        if not topics:
            logger.error("--topic '%s'에 해당하는 topic이 config에 없습니다.", opts.topic)
            return 2
        cfg = replace(cfg, topics=topics)

    try:
        # fetch/LLM에 시간을 쓰기 전에 발행 설정(토큰 등)부터 검증 — 실패 시 즉시 종료.
        publisher = make_publisher(cfg, force_dry_run=opts.dry_run)
    except PublishConfigError as e:
        logger.error("발행 설정 오류: %s", e)
        return 1
    client = summarizer.make_client(cfg)
    topics_by_name = {t.name: t for t in cfg.topics}

    with Store(cfg.resolve(cfg.storage.db_path)) as store:
        # store를 fetch에도 넘긴다 — S2 인용순 topic의 기존 논문 스킵 + 페이지 커서용.
        fetched = fetch_all(cfg, backfill=opts.backfill, store=store)
        todo = store.filter_unprocessed(fetched)
        logger.info("미처리 논문 %d편 (검색 %d편 중)", len(todo), len(fetched))

        if opts.skip_filter:
            logger.info("--skip-filter: relevance 필터 생략")
            passed = todo
        else:
            passed = RelevanceFilter(cfg, judge_client=client).apply(todo, topics_by_name, store)
        n_passed = len(passed)
        logger.info("필터 통과 %d편 (미처리 %d편 중)", n_passed, len(todo))

        replay_items = _replay.select(cfg, store, publisher)[: opts.limit]   # [REPLAY] Zulip 되올리기 — 옛 논문 먼저
        selected = _select(cfg, store, passed, opts.limit, topics_by_name, replay_items)   # [REPLAY] 마지막 인자만 지운다

        done = failed = 0
        config_error: PublishConfigError | None = None
        for i, item in enumerate(replay_items, start=1):   # [REPLAY] 되올리기 블록 — 옛 논문을 다 올리면 통째로 지운다
            logger.info("[되올리기 %d/%d] %s — %.80s", i, len(replay_items), item.uid, item.title)
            try:
                ok = _replay.process(cfg, store, publisher, item, client)
            except PublishConfigError as e:
                config_error = e
                logger.error("전송 설정 오류로 실행 중단: %s", e)
                break
            done += ok
            failed += not ok
        for i, paper in enumerate(selected, start=1):
            if config_error:   # [REPLAY] 되올리기에서 설정 오류가 났으면 신규도 건너뛴다 (이 두 줄)
                break          # [REPLAY]
            logger.info("[%d/%d] %s — %.80s", i, len(selected), paper.uid, paper.title)
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
    if summarizer.llm_used():  # 이 실행이 LLM(judge/요약)을 실제로 불렀을 때만 — GPU 즉시 반납 (ttl 10분을 기다리지 않음)
        summarizer.unload_model(cfg)
    logger.info(
        "완료: 검색 %d / 미처리 %d / 필터 통과 %d / 요약 완료 %d / 실패 %d",
        len(fetched), len(todo), n_passed, done, failed,
    )
    return 1 if config_error else 0
