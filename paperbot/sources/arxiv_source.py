"""arXiv 소스 (보조 — 학회 게재 전 최신 프리프린트용).

venue 필터가 없으므로 이 소스의 결과는 relevance 필터를 반드시 거친다.
SubmittedDate 내림차순 + arxiv_lookback_days 조기 중단.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone

import arxiv

from ..config import Topic
from ..models import Paper, arxiv_base_id, make_uid
from .base import PaperSource

logger = logging.getLogger(__name__)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


class ArxivSource(PaperSource):
    name = "arxiv"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = arxiv.Client(
            page_size=min(self.cfg.fetch.max_results_per_query, 100),
            delay_seconds=3.0,
            num_retries=3,
        )

    def fetch(self, topic: Topic) -> list[Paper]:
        if topic.arxiv is None or not topic.arxiv.enabled:
            return []

        cutoff = datetime.now(timezone.utc) - timedelta(days=self.cfg.fetch.arxiv_lookback_days)
        search = arxiv.Search(
            query=topic.arxiv.query,
            max_results=self.cfg.fetch.max_results_per_query,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )

        papers: list[Paper] = []
        for result in self.client.results(search):
            published = result.published
            if published.tzinfo is None:
                published = published.replace(tzinfo=timezone.utc)
            # 내림차순이므로 cutoff 이전이 나오면 그 뒤는 전부 오래된 것.
            # 단 --backfill이면 lookback을 무시하고 max_results까지 수집.
            if not self.backfill and published < cutoff:
                break

            base_id = arxiv_base_id(result.get_short_id())
            papers.append(
                Paper(
                    uid=make_uid(base_id, result.doi, None),
                    title=_clean(result.title),
                    authors=[a.name for a in result.authors],
                    abstract=_clean(result.summary) or None,
                    venue=None,  # 프리프린트
                    year=published.year,
                    url=f"https://arxiv.org/abs/{base_id}",
                    pdf_url=result.pdf_url,
                    arxiv_id=base_id,
                    doi=result.doi,
                    citation_count=None,
                    source="arxiv",
                    topic_name=topic.name,
                )
            )

        logger.info(
            "arXiv '%s': %d편 (lookback %d일%s)",
            topic.name, len(papers), self.cfg.fetch.arxiv_lookback_days,
            ", backfill" if self.backfill else "",
        )
        return papers
