"""소스 공통 Paper 스키마.

소스(Semantic Scholar / arXiv)가 달라도 이후 파이프라인은 이 dataclass만 다룬다.

uid 규칙 (소스 간 dedupe 키):
  1) arXiv id가 있으면        arxiv:<base_id>   (버전 접미사 제거)
  2) 없고 DOI가 있으면        doi:<doi 소문자>
  3) 둘 다 없으면(S2 전용)    s2:<paperId>

주의 — 스펙은 "DOI 우선"이었으나 arXiv id 우선으로 구현했다:
S2는 같은 논문에 DOI+arXiv id를 함께 주지만 arXiv API는 arXiv id만 주므로,
DOI를 우선하면 두 소스에서 온 같은 논문의 uid가 서로 달라져 dedupe가 깨진다.
공통 분모인 arXiv id를 우선해야 소스 간 dedupe가 성립한다.
(arXiv 자동 발급 DOI "10.48550/arXiv.XXXX"도 arxiv uid로 정규화된다.)
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_VERSION_RE = re.compile(r"v\d+$")
# arXiv가 자동 발급하는 DOI 형태: 10.48550/arXiv.2401.12345
_ARXIV_DOI_RE = re.compile(r"^10\.48550/arxiv\.(.+)$", re.IGNORECASE)


def arxiv_base_id(arxiv_id: str) -> str:
    """'2401.12345v2' -> '2401.12345', 'cs/0303001v1' -> 'cs/0303001'."""
    return _VERSION_RE.sub("", arxiv_id.strip())


def make_uid(arxiv_id: str | None, doi: str | None, s2_id: str | None) -> str:
    # DOI가 arXiv 자동 발급 형태면 arXiv id로 취급.
    if not arxiv_id and doi:
        m = _ARXIV_DOI_RE.match(doi.strip())
        if m:
            arxiv_id = m.group(1)

    if arxiv_id:
        return f"arxiv:{arxiv_base_id(arxiv_id)}"
    if doi:
        return f"doi:{doi.strip().lower()}"
    if s2_id:
        return f"s2:{s2_id}"
    raise ValueError("uid를 만들 식별자가 없습니다 (arxiv_id/doi/s2_id 모두 None)")


@dataclass
class Paper:
    uid: str
    title: str
    authors: list[str]
    abstract: str | None      # S2에서 간혹 None
    venue: str | None         # arXiv 프리프린트면 None
    year: int | None
    url: str                  # 사람이 여는 링크 (abs 페이지 / doi.org / S2 페이지)
    pdf_url: str | None       # 없으면 abstract 기반 요약으로 폴백
    arxiv_id: str | None      # base id (버전 제거)
    doi: str | None
    citation_count: int | None
    source: str               # "semantic_scholar" | "arxiv"
    topic_name: str
    # 필터 단계에서 채워지는 부가 정보 (DB 기록용)
    judge_score: float | None = None

    @property
    def embed_text(self) -> str:
        """relevance 임베딩에 쓰는 텍스트 (abstract 없으면 title만)."""
        if self.abstract:
            return f"{self.title}\n\n{self.abstract}"
        return self.title
