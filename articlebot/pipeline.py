"""The articlebot pipeline: ingest (feeds → judge → backlog) and report (per-org newest → extract → summarize → publish).
articlebot 파이프라인 — 수집(피드 → judge → 백로그)과 발행(기관별 최신 → 추출 → 요약 → 그림 → 발행).

흐름:
  1) 수집: 소스별 RSS/Atom → lookback·카테고리·처리 이력 필터 → 새 글마다 본문 앞부분으로 LLM judge
     → threshold 미만은 filtered_out, 통과분은 백로그(DB, status=pending)에 Article 전체를 보관
  2) 발행: 기관(org)별로 백로그에서 게시일 최신순으로 report.per_org편 (새 글이 있으면 그게 가장 최신,
     없으면 백로그의 다음 최신 글) → 본문·그림 추출 → 요약 → 그림 검증·선택 → Slack/Zulip → done
글 하나가 실패해도 전체는 계속 진행한다 (실패 글은 failed로 백로그에 남아 다음 실행에 1회 재시도).
진입점·인자·로깅·락은 cli.py.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from urllib.parse import urlsplit

from . import chatdb, llm
from .config import Config, ConfigError, SourceConfig
from .extractor import extract
from .feeds import category_allowed, fetch_feed
from .http import Http
from .images import fetch_images
from .llm import LLMError
from .lockfile import lock_held
from .models import Article, canonical_url
from .publisher import PublishConfigError, make_publisher
from .store import Store

logger = logging.getLogger("articlebot")


REJUDGE_STATUSES = ("pending", "filtered_out")   # --rejudge 대상 (발행·실패·seen 글은 건드리지 않음)


LAST_INGEST_KEY = "last_ingest_date"              # meta 테이블 — 마지막 전체 수집 날짜 (수집 주기 판단)


def _active_sources(cfg: Config, args) -> list[SourceConfig]:
    if args.source:
        names = set(args.source)
        unknown = names - {s.name for s in cfg.sources}
        if unknown:
            raise ConfigError(f"--source에 config에 없는 이름: {sorted(unknown)}")
        return [s for s in cfg.sources if s.name in names]
    return [s for s in cfg.sources if s.enabled]


def _org_groups(sources: list[SourceConfig]) -> dict[str, list[str]]:
    """기관 → 소스 이름들 (config 순서 유지)."""
    groups: dict[str, list[str]] = {}
    for s in sources:
        groups.setdefault(s.org, []).append(s.name)
    return groups


def _cutoff(cfg: Config, args) -> datetime | None:
    days = cfg.fetch.lookback_days if args.lookback_days is None else args.lookback_days
    return datetime.now(timezone.utc) - timedelta(days=days) if days > 0 else None


def _date(a: Article) -> str:
    return a.published.strftime("%Y-%m-%d") if a.published else "????-??-??"


def judge_excerpt(cfg: Config, http: Http, a: Article) -> str:
    """judge 입력용 본문 앞부분. 피드 요약문은 소스마다 부실해서(Google Research는 카테고리명뿐) 본문을 쓴다."""
    try:
        text = extract(cfg, http, a).text
    except Exception as e:  # noqa: BLE001 — 추출 실패 시 피드 요약문으로
        logger.warning("judge용 본문 추출 실패 (%s): %s", a.uid, e)
        text = ""
    if len(text) < 200:
        text = f"{a.feed_summary}\n\n{text}".strip()
    return text[: cfg.judge.excerpt_chars]


def ingest(cfg: Config, args, http: Http, store: Store, client) -> bool:
    """새 글을 채점해 백로그에 적재. 반환: LLM을 썼는지."""
    cutoff = _cutoff(cfg, args)
    rejudge = getattr(args, "rejudge", False) and not args.mark_seen
    used_llm = False
    feeds_ok = 0
    seen: set[str] = set()          # 소스 간 중복(같은 글이 두 피드에 실림) 제거

    for src in _active_sources(cfg, args):
        try:
            items = fetch_feed(http, src)
        except Exception as e:  # noqa: BLE001 — 소스 단위 실패 격리
            logger.error("피드 수집 실패 %s: %s", src.name, e)
            continue
        feeds_ok += 1

        new: list[Article] = []
        prev: dict[str, str | None] = {}
        for a in items:
            if a.uid in seen:
                continue
            seen.add(a.uid)
            if cutoff and a.published and a.published < cutoff:
                continue
            if not category_allowed(a, src):
                continue
            status = store.status(a.uid)
            if status is not None and not (rejudge and status in REJUDGE_STATUSES):
                continue
            prev[a.uid] = status
            new.append(a)

        if args.mark_seen:
            for a in new:
                store.mark_seen(a)
            logger.info("%s: 신규 %d편을 seen으로 기록", src.name, len(new))
            continue

        added = dropped = errors = 0
        revived = demoted = 0        # --rejudge: 탈락→대기, 대기→탈락
        for a in new:
            if src.judge and not args.skip_judge:
                used_llm = True
                res = llm.judge(cfg, client, a, judge_excerpt(cfg, http, a))
                if res is None:
                    errors += 1
                    logger.warning("judge 실패 — 다음 실행에 재채점: %s", a.uid)
                    continue
                a.judge_score, a.judge_reason = res
                if a.judge_score < cfg.judge.threshold:
                    store.mark_filtered(a)
                    dropped += 1
                    demoted += prev[a.uid] == "pending"
                    logger.info("judge 탈락 %.0f점 [%s] %.70s — %s", a.judge_score, src.name, a.title, a.judge_reason)
                    continue
                logger.info("judge 통과 %.0f점 [%s] %.70s — %s", a.judge_score, src.name, a.title, a.judge_reason)
            store.mark_pending(a)
            added += 1
            revived += prev[a.uid] == "filtered_out"
        logger.info("%s: 피드 %d → %s %d → 백로그 %d / 탈락 %d / 채점 실패 %d%s",
                    src.name, len(items), "재채점" if rejudge else "신규", len(new), added, dropped, errors,
                    f" (탈락→대기 {revived}, 대기→탈락 {demoted})" if rejudge else "")

    # 전체 소스 수집을 마쳤을 때만 주기 기준일 갱신 (--source 부분 수집·--mark-seen은 제외)
    if feeds_ok and not args.source and not args.mark_seen:
        store.set_meta(LAST_INGEST_KEY, _today().isoformat())
    return used_llm


def _today() -> date:
    """로컬(서버) 날짜 — 수집 주기를 '날짜' 단위로 센다 (paperbot 종료 시각이 매일 조금씩 달라도 밀리지 않게)."""
    return datetime.now().astimezone().date()


def should_ingest(cfg: Config, args, store: Store) -> tuple[bool, str]:
    """이번 실행에서 수집(피드 확인·채점)할지. 반환: (여부, 로그용 사유)."""
    if args.ingest_only or getattr(args, "rejudge", False) or args.mark_seen or getattr(args, "force_ingest", False):
        return True, "명시적 수집 요청"
    interval = cfg.fetch.interval_days
    if interval <= 1:
        return True, "매 실행 수집"
    last = store.get_meta(LAST_INGEST_KEY)
    if not last:
        return True, "수집 이력 없음"
    last_date = date.fromisoformat(last)
    days = (_today() - last_date).days
    if days >= interval:
        return True, f"마지막 수집 {last} ({days}일 전) ≥ 주기 {interval}일"
    return False, (f"마지막 수집 {last} ({days}일 전), 주기 {interval}일 → "
                   f"다음 수집 {last_date + timedelta(days=interval)}")


def process_article(cfg: Config, http: Http, store: Store, publisher, client, a: Article) -> bool:
    """글 하나: 추출 → 요약 → 그림 → 발행 → 기록. 성공 시 True."""
    try:
        ex = extract(cfg, http, a)
        if not a.title:
            a.title = ex.title or a.url
        if len(ex.text) < cfg.fetch.min_body_chars:
            raise ValueError(f"본문이 너무 짧음 ({len(ex.text)}자) — 추출 실패로 간주")
        summary = llm.summarize(cfg, client, a, ex.text)
        images = fetch_images(http, ex.images, cfg.images)
        images = llm.pick_images(cfg, client, a, images)
        publisher.publish(a, summary, ex, images)
        store.mark_done(a, summary)
        chatdb.store_article(cfg, a, ex.text, summary)   # 챗봇 캐시에 본문·요약 (실패해도 발행은 끝났으니 경고만)
        return True
    except PublishConfigError:
        raise
    except LLMError as e:
        logger.error("LLM 실패로 skip: %s (%s)", a.uid, e)
        err = str(e)
    except Exception as e:  # noqa: BLE001 — 개별 글 실패 격리
        logger.exception("글 처리 실패: %s (%s)", a.uid, e)
        err = str(e)
    store.mark_failed(a, err)
    return False


def report(cfg: Config, args, http: Http, store: Store, publisher, client) -> tuple[int, int]:
    """기관별로 백로그에서 최신순 per_org편 발행. 반환: (성공, 실패)."""
    cutoff = _cutoff(cfg, args)
    if cutoff:
        n = store.expire_older_than(cutoff)
        if n:
            logger.info("백로그 만료: %d편 (게시 %s 이전)", n, cutoff.date())

    done = failed = 0
    for org, names in _org_groups(_active_sources(cfg, args)).items():
        backlog = store.backlog(names)
        if not backlog:
            logger.info("[%s] 새 글·백로그 없음 — 오늘은 발행하지 않음", org)
            continue
        posted = attempts = 0
        for a in backlog:
            if posted >= cfg.report.per_org or attempts >= cfg.report.max_attempts:
                break
            if args.limit is not None and done + failed >= args.limit:
                return done, failed
            attempts += 1
            score = f"{a.judge_score:.0f}점" if a.judge_score is not None else "미채점"
            logger.info("[%s] 발행 대상: %s (%s, %s, 백로그 %d편 중 최신) — %.80s",
                        org, a.source, _date(a), score, len(backlog), a.title)
            ok = process_article(cfg, http, store, publisher, client, a)
            posted += ok
            done += ok
            failed += not ok
    return done, failed


def backfill_chat(cfg: Config, http: Http, store: Store, limit: int | None = None) -> dict:
    """DB의 done 글(요약 저장됨)을 챗봇 캐시에 채운다 — 본문은 다시 받아 추출(LLM·발행 없음). 본문이 이미 있는 글은 건너뛴다."""
    if not cfg.storage.chat_text_db:
        raise ConfigError("storage.chat_text_db가 비어 있습니다")
    rows = store.conn.execute(
        "SELECT uid, source, org, title, url, published, summary FROM articles WHERE status='done' "
        "ORDER BY COALESCE(published, processed_at) DESC").fetchall()
    labels = {s.name: s.label for s in cfg.sources}
    cache = chatdb.ChatCache(cfg.resolve(cfg.storage.chat_text_db))
    stats = {"done_in_db": len(rows), "skipped": 0, "stored": 0, "no_text": 0, "failed": 0}
    try:
        todo = [r for r in rows if not cache.has_text(r["uid"])]
        stats["skipped"] = len(rows) - len(todo)
        if limit is not None:
            todo = todo[:limit]
        for i, r in enumerate(todo, start=1):
            a = Article(uid=r["uid"], source=r["source"], source_label=labels.get(r["source"], r["source"]),
                        title=r["title"] or "", url=r["url"], org=r["org"] or "",
                        published=datetime.fromisoformat(r["published"]) if r["published"] else None)
            try:
                ex = extract(cfg, http, a)
                if not a.title:
                    a.title = ex.title or a.url
                text = ex.text if len(ex.text) >= cfg.fetch.min_body_chars else ""
                cache.upsert(a, text, r["summary"], cfg.llm.model)
                stats["stored"] += 1
                stats["no_text"] += not text
                logger.info("[백필 %d/%d] %s — %d자%s: %.70s", i, len(todo), a.uid, len(text),
                            "" if text else " (본문 없음)", a.title)
            except Exception as e:  # noqa: BLE001 — 한 편 실패는 다음으로
                stats["failed"] += 1
                logger.warning("[백필 %d/%d] 실패 %s: %s", i, len(todo), a.uid, e)
    finally:
        cache.close()
    logger.info("챗봇 캐시 백필 끝: %s", stats)
    return stats


def show_list(cfg: Config, args, http: Http, store: Store) -> None:
    cutoff = _cutoff(cfg, args)
    by_name = {s.name: s for s in cfg.sources}
    for org, names in _org_groups(_active_sources(cfg, args)).items():
        print(f"== {org}")
        for name in names:
            try:
                items = fetch_feed(http, by_name[name])
            except Exception as e:  # noqa: BLE001
                print(f"   {name}: 피드 수집 실패 ({e})")
                continue
            unjudged = [a for a in items if not (cutoff and a.published and a.published < cutoff)
                        and category_allowed(a, by_name[name]) and store.status(a.uid) is None]
            print(f"   {name}: 피드 {len(items)}편, 미채점 신규 {len(unjudged)}편")
        backlog = store.backlog(names)
        print(f"   백로그 {len(backlog)}편 (최신순 상위 5)")
        for a in backlog[:5]:
            score = f"{a.judge_score:.0f}" if a.judge_score is not None else "-"
            print(f"     {_date(a)}  [{score:>2}] {a.title[:80]}")


def _manual_article(cfg: Config, url: str) -> Article:
    host = (urlsplit(url).hostname or "").lower()
    src = next((s for s in cfg.sources if (urlsplit(s.feed).hostname or "").lower() == host), None)
    return Article(uid=canonical_url(url), source=src.name if src else "manual",
                   source_label=src.label if src else host, title="", url=url, published=None,
                   org=src.org if src else "manual")


def _log_backlog(store: Store) -> None:
    counts = store.counts()
    orgs = sorted({o for o, _ in counts})
    parts = [f"{o} 대기 {counts.get((o, 'pending'), 0) + counts.get((o, 'failed'), 0)}"
             f"/발행 {counts.get((o, 'done'), 0)}/탈락 {counts.get((o, 'filtered_out'), 0)}" for o in orgs]
    if parts:
        logger.info("DB 현황: %s", " | ".join(parts))


def run(cfg: Config, args) -> int:
    dry = args.dry_run or cfg.publisher.dry_run
    needs_publisher = not (args.list or args.mark_seen or args.ingest_only or getattr(args, "backfill_chat", False))
    try:
        publisher = make_publisher(cfg, force_dry_run=args.dry_run) if needs_publisher else None
    except PublishConfigError as e:
        logger.error("발행 설정 오류: %s", e)
        return 1
    http = Http(cfg.fetch)
    client = llm.make_client(cfg)
    used_llm = False
    rc = 0

    with Store(cfg.resolve(cfg.storage.db_path), dry=dry) as store:
        if dry:
            logger.info("dry-run: DB를 메모리 사본으로 사용 — 파일은 변경되지 않음")
        if args.list:
            show_list(cfg, args, http, store)
            return 0
        if getattr(args, "backfill_chat", False):
            print(backfill_chat(cfg, http, store, limit=args.limit))
            return 0
        if args.url:
            a = _manual_article(cfg, args.url)
            used_llm = True
            try:
                ok = process_article(cfg, http, store, publisher, client, a)
            except PublishConfigError as e:
                logger.error("전송 설정 오류: %s", e)
                ok = False
            rc = 0 if ok else 1
        else:
            go, why = should_ingest(cfg, args, store)
            if go:
                logger.info("수집 실행: %s", why)
                used_llm = ingest(cfg, args, http, store, client)
            else:
                logger.info("수집 건너뜀 — 백로그에서만 발행: %s", why)
            if not (args.mark_seen or args.ingest_only):
                try:
                    done, failed = report(cfg, args, http, store, publisher, client)
                except PublishConfigError as e:
                    logger.error("전송 설정 오류로 발행 중단: %s — 백로그는 그대로 남아 다음 실행 때 처리됩니다.", e)
                    done, failed, rc = 0, 0, 1
                used_llm = used_llm or done + failed > 0
                logger.info("발행 완료: 성공 %d / 실패 %d", done, failed)
        _log_backlog(store)

    if used_llm and cfg.llm.unload_after_run:
        busy = [p for p in cfg.llm.peer_lockfiles if lock_held(p)]
        if busy:
            logger.info("다른 봇이 같은 모델을 사용 중(%s) — 언로드 생략", ", ".join(busy))
        else:
            llm.unload_model(cfg)
    return rc
