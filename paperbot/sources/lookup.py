"""Single-paper metadata lookup by uid (arXiv API or Semantic Scholar), for papers already in the DB.
DB 레코드(uid·제목·출처·topic)로 논문 한 편의 메타데이터를 복원한다 — Zulip 되올리기(replay.py)와 챗봇 캐시 백필(chatdb.py)이 쓴다.

- fetch_s2(): S2 단건 조회. 검색 소스(semantic_scholar.py)와 같은 전역 rate limiter·필드를 쓴다.
- fetch_arxiv(): arXiv API 단건 조회 → S2 item과 같은 모양의 dict.
- paper_from_uid(): 둘을 prefer 순서대로 시도하고, 둘 다 없으면 DB 제목만 가진 최소 Paper. uid와 url은 DB 기준으로 고정한다.
"""
from __future__ import annotations

import dataclasses
import logging
import os
import time

import requests

from ..models import Paper, arxiv_base_id, canonical_url
from . import semantic_scholar as s2mod
from .arxiv_source import _clean

logger = logging.getLogger(__name__)

_S2_PAPER_URL = "https://api.semanticscholar.org/graph/v1/paper/{}"
_S2_RETRY_WAITS = (3.0, 5.0, 10.0)


def s2_id_for(uid: str) -> str:
    """uid → S2 단건 조회 id ('arxiv:X' → 'arXiv:X', 'doi:Y' → 'DOI:Y', 's2:Z' → 'Z')."""
    kind, _, value = uid.partition(":")
    return {"arxiv": f"arXiv:{value}", "doi": f"DOI:{value}"}.get(kind, value)


def fetch_s2(s2_id: str, session: requests.Session | None = None) -> dict | None:
    """S2 단건 조회. 없으면(404) None, 429는 세 번 기다렸다 재시도, 그래도 안 되면 RuntimeError."""
    session = session or requests.Session()
    api_key = os.environ.get("S2_API_KEY", "").strip()
    headers = {"x-api-key": api_key} if api_key else {}
    interval = s2mod._MIN_INTERVAL_KEYED if api_key else s2mod._MIN_INTERVAL_KEYLESS
    last: Exception | None = None
    for attempt in range(len(_S2_RETRY_WAITS) + 1):
        s2mod._LIMITER.wait(interval)
        try:
            r = session.get(_S2_PAPER_URL.format(s2_id), params={"fields": s2mod._FIELDS}, headers=headers, timeout=30)
            if r.status_code == 404:
                return None
            if r.status_code == 429:
                raise RuntimeError("S2 429")
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            last = e
            if attempt < len(_S2_RETRY_WAITS):
                time.sleep(_S2_RETRY_WAITS[attempt])
    raise RuntimeError(f"S2 단건 조회 실패 ({s2_id}): {last}")


def fetch_arxiv(arxiv_id: str) -> dict | None:
    """arXiv API 단건 조회 → S2 item과 같은 모양의 dict (없으면 None)."""
    import arxiv

    result = next(arxiv.Client().results(arxiv.Search(id_list=[arxiv_id])), None)
    if result is None:
        return None
    published = result.published
    base = arxiv_base_id(result.get_short_id())
    return {
        "title": _clean(result.title), "abstract": _clean(result.summary) or None, "venue": None,
        "year": published.year if published else None,
        "externalIds": {"ArXiv": base, **({"DOI": result.doi} if result.doi else {})},
        "openAccessPdf": {"url": result.pdf_url} if result.pdf_url else None, "citationCount": None,
        "authors": [{"name": a.name} for a in result.authors],
        "publicationDate": published.strftime("%Y-%m-%d") if published else None,
        "url": f"https://arxiv.org/abs/{base}",
    }


def paper_from_uid(uid: str, title: str, source: str, topic_name: str, judge_score: float | None = None,
                   session: requests.Session | None = None, prefer: str = "arxiv") -> Paper:
    """DB 레코드(uid·title·source·topic) → Paper. 둘 다 없으면 제목만.

    prefer="arxiv": arXiv uid면 arXiv API 먼저, 없거나 실패하면 S2 (되올리기 — arXiv 쪽 PDF·초록이 정확).
    prefer="s2": S2 먼저, 없거나 실패하면 arXiv uid만 arXiv API (백필 — arXiv API는 느릴 때 편당 30~45초, 2026-10-23 실측).
    uid와 url은 DB 기준으로 고정한다 — 고정 Zulip topic의 '이미 올렸는지' 판정이 첫 메시지 링크로 이뤄지므로.
    """
    kind, _, value = uid.partition(":")

    def via_arxiv():
        if kind != "arxiv":
            return None
        try:
            return fetch_arxiv(value)
        except Exception as e:  # noqa: BLE001 — 다음 수단으로
            logger.warning("arXiv 조회 실패 %s: %s", uid, e)
            return None

    def via_s2():
        try:
            return fetch_s2(s2_id_for(uid), session)
        except Exception as e:  # noqa: BLE001 — 다음 수단으로
            logger.warning("S2 조회 실패 %s: %s", uid, e)
            return None

    order = (via_s2, via_arxiv) if prefer == "s2" else (via_arxiv, via_s2)
    item = order[0]() or order[1]()
    if item is not None:
        p = s2mod.SemanticScholarSource._to_paper(item, topic_name)
        if p is not None:
            return dataclasses.replace(p, uid=uid, url=canonical_url(uid), source=source, topic_name=topic_name,
                                       judge_score=judge_score)
    arxiv_id = value if kind == "arxiv" else None
    doi = value if kind == "doi" else None
    return Paper(
        uid=uid, title=title, authors=[], abstract=None, venue=None, year=None, url=canonical_url(uid),
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else None,
        arxiv_id=arxiv_id, doi=doi, citation_count=None, source=source, topic_name=topic_name, judge_score=judge_score,
    )
