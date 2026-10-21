"""Slack 텍스트 속 논문 식별자 → Paper (listener.py용 단건 조회).

식별자 추출 (텍스트에 먼저 나온 것 우선):
  arXiv  : arxiv.org/abs|pdf|html/<id>, "arXiv:<id>", 맨 id(2505.13631)
  DOI    : doi.org/<doi>, "doi:<doi>", 맨 10.xxxx/...
  S2     : semanticscholar.org/paper/[slug/]<40hex>
조회 순서: S2 단건 API → (arXiv면) arXiv API → (arXiv면) 스레드 원글 제목으로 최소 Paper.
S2는 갓 올라온 arXiv 논문을 아직 색인하지 않았을 수 있어 arXiv 폴백이 필요하다.
"""
from __future__ import annotations

import html
import logging
import os
import re
import time
from dataclasses import dataclass

import requests

from paperbot.models import Paper, arxiv_base_id, make_uid

logger = logging.getLogger(__name__)

_S2_PAPER_URL = "https://api.semanticscholar.org/graph/v1/paper/{}"
_S2_FIELDS = ",".join([
    "title", "abstract", "venue", "year", "externalIds", "openAccessPdf", "citationCount",
    "authors", "publicationDate", "url", "publicationTypes", "journal",
])
_S2_RETRY_WAITS = (3.0, 5.0, 10.0)   # 429/5xx 재시도 대기(초) — semantic_scholar._RATE_LIMIT_WAITS와 같음
_TOPIC = "library"   # Paper.topic_name — 수동 저장 경로 표시용

# Slack mrkdwn: <https://x|label> / <https://x> / <@U123> / <#C123|name>
_SLACK_LINK_RE = re.compile(r"<((?:https?|mailto):[^|>]+)(?:\|[^>]*)?>")
_SLACK_TOKEN_RE = re.compile(r"<[@#!][^>]*>")

_ARXIV_ID = r"(\d{4}\.\d{4,5}|[a-z][a-z\-]*(?:\.[A-Z]{2})?/\d{7})(v\d+)?"
_PATTERNS = [
    ("arxiv", re.compile(r"arxiv\.org/(?:abs|pdf|html)/" + _ARXIV_ID, re.IGNORECASE)),
    ("arxiv", re.compile(r"\barxiv:\s*" + _ARXIV_ID, re.IGNORECASE)),
    ("doi", re.compile(r"(?:doi\.org/|\bdoi:\s*)(10\.\d{4,9}/[^\s<>\"|]+)", re.IGNORECASE)),
    ("s2", re.compile(r"semanticscholar\.org/paper/(?:[^/\s>|]+/)?([0-9a-f]{40})\b", re.IGNORECASE)),
    ("doi", re.compile(r"(?<![\w/.])(10\.\d{4,9}/[^\s<>\"|]+)")),
    ("arxiv", re.compile(r"(?<![\w/.])(\d{4}\.\d{4,5})(v\d+)?(?![\w.])")),
]
_DOI_TRAIL = ".,;:)]}>'\""


@dataclass(frozen=True)
class PaperRef:
    kind: str    # "arxiv" | "doi" | "s2"
    value: str   # arXiv base id / DOI / S2 paperId

    @property
    def s2_id(self) -> str:
        return {"arxiv": f"arXiv:{self.value}", "doi": f"DOI:{self.value}"}.get(self.kind, self.value)


@dataclass
class Resolved:
    paper: Paper
    pub_types: list[str]         # S2 publicationTypes (예: ["Conference"]) — Zotero itemType 결정용
    pub_date: str | None         # "YYYY-MM-DD" (없으면 year만 씀)
    journal: str | None = None   # S2 journal.name


def clean_slack_text(text: str) -> str:
    """Slack 메시지 원문 → 평문: 링크는 URL만 남기고 멘션/채널 토큰 제거, HTML 엔티티 복원."""
    text = _SLACK_LINK_RE.sub(r"\1", text or "")
    text = _SLACK_TOKEN_RE.sub(" ", text)
    return html.unescape(text)


def find_refs(text: str) -> list[tuple[int, int, PaperRef]]:
    """평문에서 논문 식별자를 (시작, 끝, ref)로 모두 찾는다. 겹치는 매치는 먼저 잡힌 패턴이 이긴다."""
    found: list[tuple[int, int, PaperRef]] = []
    for kind, pat in _PATTERNS:
        for m in pat.finditer(text):
            s, e = m.span()
            if any(s < fe and fs < e for fs, fe, _ in found):
                continue
            val = m.group(1)
            if kind == "doi":
                val = val.rstrip(_DOI_TRAIL)
                if val.lower().startswith("10.48550/arxiv."):   # arXiv 자동 DOI → arXiv id
                    kind, val = "arxiv", val.split(".", 2)[2]
                else:
                    val = val.lower()
            if kind == "arxiv":
                val = arxiv_base_id(val)
            found.append((s, e, PaperRef(kind, val)))
    found.sort(key=lambda t: t[0])
    return found


def first_ref(text: str) -> PaperRef | None:
    refs = find_refs(text)
    return refs[0][2] if refs else None


# ---------------------------------------------------------------- 조회

def _s2_lookup(ref: PaperRef) -> Resolved | None:
    from paperbot.sources.semantic_scholar import SemanticScholarSource  # 매핑 규칙 재사용

    headers = {}
    key = os.environ.get("S2_API_KEY", "").strip()
    if key:
        headers["x-api-key"] = key
    url = _S2_PAPER_URL.format(requests.utils.quote(ref.s2_id, safe=":/"))
    tries = len(_S2_RETRY_WAITS) + 1
    for attempt in range(1, tries + 1):
        wait = _S2_RETRY_WAITS[min(attempt, len(_S2_RETRY_WAITS)) - 1]
        try:
            resp = requests.get(url, params={"fields": _S2_FIELDS}, headers=headers, timeout=30)
        except requests.RequestException as e:
            logger.warning("S2 단건 조회 연결 실패 (%d/%d): %s", attempt, tries, e)
            if attempt < tries:
                time.sleep(wait)
            continue
        if resp.status_code == 404:
            logger.info("S2에 없음: %s", ref.s2_id)
            return None
        if resp.status_code == 429 or resp.status_code >= 500:
            # 키가 있어도 일시적 429가 난다(실측: 수 초~15초 안에 풀림) — 기존 S2 소스와 같은 대기.
            logger.warning("S2 단건 조회 HTTP %d (%d/%d)", resp.status_code, attempt, tries)
            if attempt < tries:
                time.sleep(wait)
            continue
        if resp.status_code != 200:
            logger.warning("S2 단건 조회 실패 HTTP %d: %s", resp.status_code, resp.text[:200])
            return None
        item = resp.json()
        paper = SemanticScholarSource._to_paper(item, _TOPIC)
        if paper is None:
            return None
        journal = (item.get("journal") or {}).get("name") or None
        return Resolved(paper=paper, pub_types=list(item.get("publicationTypes") or []),
                        pub_date=item.get("publicationDate"), journal=journal)
    return None


def _arxiv_lookup(arxiv_id: str) -> Resolved | None:
    import arxiv

    client = arxiv.Client(page_size=1, delay_seconds=3.0, num_retries=3)
    try:
        result = next(client.results(arxiv.Search(id_list=[arxiv_id])), None)
    except Exception as e:  # noqa: BLE001 — arXiv API 429가 잦다 (README 참조)
        logger.warning("arXiv 단건 조회 실패 %s: %s", arxiv_id, e)
        return None
    if result is None:
        return None
    base_id = arxiv_base_id(result.get_short_id())
    paper = Paper(
        uid=make_uid(base_id, result.doi, None),
        title=" ".join(result.title.split()),
        authors=[a.name for a in result.authors],
        abstract=" ".join((result.summary or "").split()) or None,
        venue=None,
        year=result.published.year,
        url=f"https://arxiv.org/abs/{base_id}",
        pdf_url=result.pdf_url,
        arxiv_id=base_id,
        doi=result.doi,
        citation_count=None,
        source="arxiv",
        topic_name=_TOPIC,
    )
    return Resolved(paper=paper, pub_types=[], pub_date=result.published.strftime("%Y-%m-%d"))


def resolve(ref: PaperRef, title_hint: str | None = None) -> Resolved:
    """식별자 → 메타데이터. 전부 실패하면 RuntimeError (사용자에게 그대로 보여줄 메시지)."""
    got = _s2_lookup(ref)
    if got is None and ref.kind == "arxiv":
        got = _arxiv_lookup(ref.value)
        if got is None and title_hint:
            # S2·arXiv API 둘 다 실패(arXiv 429 등) — 스레드 원글의 제목으로 최소 항목만 만든다.
            logger.warning("메타데이터 조회 실패 — 원글 제목으로 최소 항목 생성: %s", ref.value)
            got = Resolved(
                paper=Paper(
                    uid=make_uid(ref.value, None, None), title=title_hint, authors=[],
                    abstract=None, venue=None, year=None,
                    url=f"https://arxiv.org/abs/{ref.value}",
                    pdf_url=f"https://arxiv.org/pdf/{ref.value}",
                    arxiv_id=ref.value, doi=None, citation_count=None,
                    source="arxiv", topic_name=_TOPIC,
                ),
                pub_types=[], pub_date=None,
            )
    if got is None:
        raise RuntimeError(f"논문 메타데이터를 찾지 못했습니다 ({ref.kind}: {ref.value})")
    return got
