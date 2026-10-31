"""Chat cache hook: put each summarized paper's full text and summary into zoterobot's text cache.
챗봇 캐시 연결 — 요약해 올린 논문의 전문(참고문헌 제거, 절단 전)과 요약을 zoterobot 본문 캐시(storage.chat_text_db)에 넣는다.

- 아침 실행(pipeline.process_paper)과 Zulip 되올리기(replay.process)가 발행 직후 store_paper()를 부른다.
  Zotero/Drive에 저장하지 않은 논문도 DM 챗봇이 search_papers/read_paper로 찾아 읽을 수 있게 하는 것이 목적 (2026-10-23 사용자 요청).
  캐시 쪽 id는 uid 해시('P…'), 컬렉션 자리에는 topic 이름이 들어간다 (zoterobot/textcache.py의 PAPERBOT_TYPE 참고).
- backfill()은 DB(papers.db)의 done 논문을 거슬러 채운다: PDF를 다시 받아 본문을 뽑고(GPU·발행 없음), 요약은 Zulip 고정 topic에
  이미 올라간 메시지에서 되읽는다(Slack 시절 것은 요약 없이 본문만 — 나중에 되올리기가 새 요약으로 덮어쓰거나 챗봇 summarize_paper가 만든다).
  `python -m paperbot --backfill-chat [--limit N] [--topic NAME]`. 본문이 이미 있는 논문은 건너뛰므로 여러 번 돌려도 된다.
- 캐시 쓰기 실패는 경고만 남긴다 — 발행은 이미 끝났고, 다음 날 같은 논문이 다시 요약되지는 않지만 백필로 채울 수 있다.
"""
from __future__ import annotations

import logging
import re
import time

from .config import Config
from .extractor import ExtractResult, extract
from .models import Paper, uid_url_key
from .publisher.zulip import (PAPER_LINK_RE, SUMMARY_TITLE, _topic_stream_map, _zulip_topic_map, make_zulip_client,
                              topic_messages)
from .sources.lookup import paper_from_uid
from .store import Store

logger = logging.getLogger(__name__)

# 발행 형식(publisher/zulip.py build_zulip_messages)의 접힌 요약 — 길면 '전체 요약 (i/n)' 조각으로 나뉜다
_SPOILER_RE = re.compile(r"````spoiler " + re.escape(SUMMARY_TITLE) + r"(?: \((\d+)/(\d+)\))?\n(.*?)\n````", re.DOTALL)
_SLEEP_S = 2.0        # 백필에서 논문 사이 쉬는 시간 — arXiv/출판사 서버 예의


def _open(cfg: Config):
    """캐시 열기 (zoterobot 패키지는 같은 레포·venv). 경로가 비면 None."""
    if not cfg.storage.chat_text_db:
        return None
    from zoterobot.textcache import TextCache
    return TextCache(cfg.resolve(cfg.storage.chat_text_db))


def paper_meta(paper: Paper) -> dict:
    return {"title": paper.title, "authors": list(paper.authors or []),
            "year": str(paper.year) if paper.year else "", "collections": [paper.topic_name],
            "abstract": paper.abstract or "", "url": paper.url}


def store_paper(cfg: Config, paper: Paper, extracted: ExtractResult, summary: str | None) -> str | None:
    """논문 전문·요약을 캐시에. 반환: 캐시 id (껐거나 실패하면 None). 예외를 밖으로 내지 않는다."""
    try:
        cache = _open(cfg)
        if cache is None:
            return None
        from zoterobot.textcache import PAPERBOT_TYPE
        text = extracted.full_text or ("" if extracted.abstract_only else extracted.text)
        try:
            key = cache.upsert_feed(PAPERBOT_TYPE, paper.uid, paper_meta(paper), text, "pdf", summary, cfg.llm.model)
        finally:
            cache.close()
        logger.info("챗봇 캐시 저장 %s ← %s (%d자%s)", key, paper.uid, len(text), ", 요약 포함" if summary else "")
        return key
    except Exception as e:  # noqa: BLE001 — 캐시는 부가 기능
        logger.warning("챗봇 캐시 저장 실패 %s: %s", paper.uid, e)
        return None


# ---------------------------------------------------------------- 백필

def zulip_summaries(cfg: Config) -> dict[str, str]:
    """Zulip 고정 topic들의 메시지에서 (논문 링크 키 → 전체 요약) — 발행 형식(publisher/zulip.py build_zulip_messages)을 되읽는다.
    메시지 하나에 머리말+접힌 요약이 있고, 길면 머리말 뒤에 '전체 요약 (i/n)' 조각 메시지가 이어진다. 실패하면 빈 dict."""
    fixed = _zulip_topic_map(cfg)
    if not fixed:
        return {}
    try:
        client = make_zulip_client(cfg.resolve(cfg.zulip.config_file))
    except Exception as e:  # noqa: BLE001
        logger.warning("Zulip 접속 실패 — 요약 없이 본문만 백필: %s", e)
        return {}
    streams = _topic_stream_map(cfg)
    out: dict[str, str] = {}
    for topic_name, ztopic in fixed.items():
        stream = streams.get(topic_name, cfg.zulip.stream)
        try:
            msgs = topic_messages(client, stream, ztopic)
        except Exception as e:  # noqa: BLE001
            logger.warning("Zulip '%s / %s' 메시지 조회 실패(요약 생략): %s", stream, ztopic, e)
            continue
        found = parse_summaries(msgs)
        out.update(found)
        logger.info("Zulip '%s / %s': 메시지 %d개에서 요약 %d편", stream, ztopic, len(msgs), len(found))
    return out


def parse_summaries(msgs: list[str]) -> dict[str, str]:
    """오래된 순 메시지 본문들 → {논문 링크 키(소문자): 전체 요약}. 링크 없는 '전체 요약 (i/n)' 조각은 직전 논문에 이어 붙인다."""
    out: dict[str, str] = {}
    last_key = None
    for content in msgs:
        m = PAPER_LINK_RE.search(content)
        parts = [(int(a) if a else 1, body) for a, _, body in _SPOILER_RE.findall(content)]
        if m:
            last_key = m.group(1).casefold()
            if parts:
                out[last_key] = "\n\n".join(b for _, b in sorted(parts))
        elif parts and last_key:   # 머리말 다음에 이어지는 요약 조각들
            out[last_key] = (out.get(last_key, "") + "\n\n" + "\n\n".join(b for _, b in sorted(parts))).strip()
    return out


def backfill(cfg: Config, store: Store, limit: int | None = None, topic: str | None = None,
             sleep_s: float = _SLEEP_S) -> dict:
    """DB done 논문 → 캐시 (최근 것부터). 본문이 이미 있는 논문은 건너뛴다. 반환: 통계 dict."""
    from zoterobot.textcache import PAPERBOT_TYPE

    cache = _open(cfg)
    if cache is None:
        raise RuntimeError("storage.chat_text_db가 비어 있습니다")
    sql = "SELECT uid, title, source, topic, judge_score FROM papers WHERE status='done'"
    params: tuple = ()
    if topic:
        sql += " AND topic=?"
        params = (topic,)
    rows = store.conn.execute(sql + " ORDER BY processed_at DESC", params).fetchall()
    summaries = zulip_summaries(cfg)
    stats = {"done_in_db": len(rows), "skipped": 0, "stored": 0, "no_text": 0, "failed": 0, "with_summary": 0}
    todo = [r for r in rows if not cache.has_feed_text(PAPERBOT_TYPE, r["uid"])]
    stats["skipped"] = len(rows) - len(todo)
    if limit is not None:
        todo = todo[:limit]
    logger.info("챗봇 캐시 백필: done %d편 중 본문 없는 %d편 처리 (Zulip 요약 %d편 확보)", len(rows), len(todo), len(summaries))
    try:
        for i, r in enumerate(todo, start=1):
            uid = r["uid"]
            try:
                # S2 단건 조회 먼저(키 있으면 1초 간격) — arXiv API는 느릴 때 편당 30~45초(2026-10-23 실측)
                paper = paper_from_uid(uid, r["title"], r["source"], r["topic"], r["judge_score"], prefer="s2")
                extracted = extract(cfg, paper, figures=False)
                text = extracted.full_text or ""
                summary = summaries.get(uid_url_key(uid).casefold())
                cache.upsert_feed(PAPERBOT_TYPE, uid, paper_meta(paper), text, "pdf", summary, cfg.llm.model)
                stats["stored"] += 1
                stats["no_text"] += not text
                stats["with_summary"] += bool(summary)
                logger.info("[백필 %d/%d] %s — %d자%s%s: %.70s", i, len(todo), uid, len(text),
                            "" if text else " (본문 없음)", ", 요약" if summary else "", paper.title)
            except Exception as e:  # noqa: BLE001 — 한 편 실패는 다음으로
                stats["failed"] += 1
                logger.warning("[백필 %d/%d] 실패 %s: %s", i, len(todo), uid, e)
            if sleep_s and i < len(todo):
                time.sleep(sleep_s)
    finally:
        cache.close()
    logger.info("챗봇 캐시 백필 끝: %s", stats)
    return stats
