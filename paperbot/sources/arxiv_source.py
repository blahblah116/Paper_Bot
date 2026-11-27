"""arXiv 소스 (보조 — 학회 게재 전 최신 프리프린트용).

venue 필터가 없으므로 이 소스의 결과는 relevance 필터를 반드시 거친다.
SubmittedDate 내림차순 + arxiv_lookback_days 조기 중단.
hf_orgs가 있으면 Hugging Face 회사 계정 모델 태그(arxiv:<id>)에서 id를 모아 arXiv에서 받는 경로를 더한다.
"""
from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timedelta, timezone

import arxiv
import requests

from ..config import Topic
from ..models import Paper, arxiv_base_id, make_uid
from .base import PaperSource

logger = logging.getLogger(__name__)

_HF_MODELS_URL = "https://huggingface.co/api/models"
_HF_PIPELINE_TAGS = ("text-generation", "image-text-to-text")
_HF_USER_AGENT = "paperbot/0.2 (paper summary bot)"
_HF_ARXIV_TAG_RE = re.compile(r"^arxiv:(\d{4}\.\d{4,5})")   # 새 형식 id만 — cs/0101001 같은 옛 형식은 버린다
_HF_MIN_LOOKBACK_DAYS = 90   # 논문이 나오고 몇 주 뒤에 모델 카드에 링크가 걸리거나 가중치가 늦게 공개되는 경우도 잡는다


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _to_paper(result: arxiv.Result, topic: Topic) -> Paper:
    """arxiv.Result -> Paper (shared by the keyword and Hugging Face paths).
    키워드 검색·HF 경로가 같이 쓰는 변환."""
    base_id = arxiv_base_id(result.get_short_id())
    return Paper(
        uid=make_uid(base_id, result.doi, None),
        title=_clean(result.title),
        authors=[a.name for a in result.authors],
        abstract=_clean(result.summary) or None,
        venue=None,  # 프리프린트
        year=result.published.year,
        url=f"https://arxiv.org/abs/{base_id}",
        pdf_url=result.pdf_url,
        arxiv_id=base_id,
        doi=result.doi,
        citation_count=None,
        source="arxiv",
        topic_name=topic.name,
    )


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
        papers = self._fetch_query(topic) if topic.arxiv.query else []
        if topic.arxiv.hf_orgs:
            try:
                seen = {p.uid for p in papers}
                papers += [p for p in self._fetch_hf(topic) if p.uid not in seen]
            except Exception as e:  # noqa: BLE001 — HF 경로가 실패해도 키워드 결과는 돌려준다
                logger.warning("arXiv '%s' HF 경로 실패: %s", topic.name, e)
        return papers

    def _fetch_query(self, topic: Topic) -> list[Paper]:
        """Keyword search (newest first). 키워드 검색 — max_results_per_query 상한은 이 경로에만."""
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

            papers.append(_to_paper(result, topic))

        logger.info(
            "arXiv '%s': %d편 (lookback %d일%s)",
            topic.name, len(papers), self.cfg.fetch.arxiv_lookback_days,
            ", backfill" if self.backfill else "",
        )
        return papers

    def _fetch_hf(self, topic: Topic) -> list[Paper]:
        """arXiv ids from the model tags of Hugging Face org accounts, then fetched from arXiv.
        회사 계정 모델의 tags에서 arxiv:<id>를 모아 arXiv에서 받는다. 새 논문인지는 id 앞 네 자리(YYMM)만 본다
        (모델 카드는 옛 논문도 인용). 기간은 평소 max(arxiv_lookback_days, 90)일, --backfill이면 topic의 hf_backfill_days(기본 1095일).
        게시일(v1 제출일)은 보지 않는다 — 심사 보류 뒤 공개된 논문은 id의 달보다 몇 달 앞설 수 있다.
        같은 논문이 매일 다시 나와도 DB(store.filter_unprocessed)가 이미 본 논문을 거른다."""
        if self.backfill:
            days = topic.arxiv.hf_backfill_days
        else:
            days = max(self.cfg.fetch.arxiv_lookback_days, _HF_MIN_LOOKBACK_DAYS)
        min_yymm = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%y%m")
        ids: set[str] = set()
        for org in topic.arxiv.hf_orgs:
            try:
                for pipeline_tag in _HF_PIPELINE_TAGS:
                    time.sleep(0.5)
                    r = requests.get(
                        _HF_MODELS_URL,
                        # createdAt이 아니라 lastModified 순 — 공개 전에 만든 저장소, 나중에 건 arXiv 링크도 잡는다
                        params={"author": org, "pipeline_tag": pipeline_tag, "sort": "lastModified",
                                "direction": -1, "limit": 1000 if self.backfill else 100},
                        headers={"User-Agent": _HF_USER_AGENT}, timeout=30,
                    )
                    r.raise_for_status()
                    ids.update(m.group(1) for model in r.json() for tag in model.get("tags") or []
                               if (m := _HF_ARXIV_TAG_RE.match(tag)))
            except Exception as e:  # noqa: BLE001 — 계정 단위 실패 격리
                logger.warning("HF 모델 목록 실패 (%s): %s", org, e)

        recent = sorted(i for i in ids if i[:4] >= min_yymm)
        papers = [_to_paper(result, topic) for k in range(0, len(recent), 40)
                  for result in self.client.results(arxiv.Search(id_list=recent[k:k + 40], max_results=None))]

        logger.info(
            "arXiv '%s' (HF 계정 %d곳): 태그 id %d개 → %s 이후 %d개 → %d편 (%d일%s)",
            topic.name, len(topic.arxiv.hf_orgs), len(ids), min_yymm, len(recent), len(papers), days,
            ", backfill" if self.backfill else "",
        )
        return papers
