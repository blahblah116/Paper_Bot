"""Semantic Scholar Graph API 소스 (주 소스).

- 엔드포인트: GET /graph/v1/paper/search/bulk (venue/연도/인용 수 필터 공식 지원)
- 페이지네이션: 응답의 `token` 사용, max_results_per_query에서 중단.
- API 키: 환경변수 S2_API_KEY (없으면 요청 간격을 길게 잡아 무키로 동작).
- rate limit 정책:
  * 모든 S2 호출(페이지네이션·topic 간 포함)에 공유되는 **전역 limiter** —
    요청 시작 간격을 키 있으면 1.2s, 없으면 8s로 보장하고, 락으로 직렬화한다
    (향후 topic 병렬화가 생겨도 S2 요청만은 순서대로 나감).
  * 429는 Retry-After 헤더를 따르고, 없으면 최소 2초 백오프.
  * 5xx/연결 오류는 지수 백오프. 재시도 총 3회, 마지막 시도 실패엔 대기 없음.
- bulk 검색은 날짜 정렬이 아니므로 lookback은 아이템 단위로 필터한다(조기 중단 없음).
"""
from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timedelta, timezone

import requests

from ..config import Topic
from ..models import Paper, arxiv_base_id, make_uid
from .base import PaperSource

logger = logging.getLogger(__name__)

_BULK_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
_FIELDS = ",".join([
    "title", "abstract", "venue", "year", "externalIds",
    "openAccessPdf", "citationCount", "authors", "publicationDate", "url",
])
_MAX_RETRIES = 3
_MIN_INTERVAL_KEYED = 1.2     # 키 있음: S2 권장(1 rps)에 여유를 둔 간격
_MIN_INTERVAL_KEYLESS = 8.0   # 키 없음: 공용 풀이라 훨씬 보수적으로
_RETRY_AFTER_CAP = 120.0      # 비정상적으로 큰 Retry-After 방어


class _GlobalRateLimiter:
    """프로세스 내 모든 S2 요청이 공유하는 시작-간격 limiter.

    락을 잡은 채로 자기 슬롯까지 대기하므로, 병렬 호출자가 있어도
    요청 시작이 직렬화되고 시작 간격 >= interval이 보장된다.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next_ok = 0.0  # 다음 요청이 허용되는 monotonic 시각

    def wait(self, interval: float) -> None:
        with self._lock:
            delay = self._next_ok - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self._next_ok = time.monotonic() + interval


_LIMITER = _GlobalRateLimiter()  # 모듈 전역 — 인스턴스가 여러 개여도 공유


class _RateLimited(Exception):
    """429 응답. retry_after는 서버가 준 값(없으면 None)."""

    def __init__(self, retry_after: float | None):
        super().__init__("HTTP 429")
        self.retry_after = retry_after


class _AuthRejected(RuntimeError):
    """401/403 — 키 문제. 재시도 없이 즉시 실패해야 한다."""


class SemanticScholarSource(PaperSource):
    name = "semantic_scholar"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.session = requests.Session()
        self.api_key = os.environ.get("S2_API_KEY", "").strip()
        if self.api_key:
            self.session.headers["x-api-key"] = self.api_key
            self.min_interval = _MIN_INTERVAL_KEYED
            # 키가 실제로 로드됐는지 로그로 확인 가능하게 (키 자체는 남기지 않음).
            logger.info("S2 API 키 사용 (…%s, 요청 시작 간격 %.1fs)",
                        self.api_key[-4:], self.min_interval)
        else:
            self.min_interval = _MIN_INTERVAL_KEYLESS
            logger.warning(
                "S2_API_KEY 미설정 — 무키 모드(공용 rate limit, 요청 시작 간격 %.0fs)",
                self.min_interval,
            )

    # ------------------------------------------------------------- HTTP

    def _get(self, params: dict) -> dict:
        last_err: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            _LIMITER.wait(self.min_interval)  # 전역 limiter: 시작 간격 보장 + 직렬화
            try:
                resp = self.session.get(_BULK_URL, params=params, timeout=30)
                if resp.status_code == 429:
                    ra = resp.headers.get("Retry-After")
                    try:
                        retry_after = min(float(ra), _RETRY_AFTER_CAP) if ra else None
                    except ValueError:
                        retry_after = None
                    raise _RateLimited(retry_after)
                if resp.status_code in (401, 403):
                    # 잘못된/비활성 키 — 재시도해도 소용없으므로 즉시 명확히 실패.
                    raise _AuthRejected(
                        f"S2 API 키가 거부되었습니다 (HTTP {resp.status_code}). "
                        "S2_API_KEY 값을 확인하세요 — 오타/만료/미활성 가능."
                    )
                if resp.status_code >= 500:
                    raise requests.HTTPError(f"HTTP {resp.status_code}", response=resp)
                resp.raise_for_status()
                return resp.json()
            except _AuthRejected:
                raise  # 키 문제는 재시도 없이 즉시 상위로 (topic 단위 실패 처리)
            except _RateLimited as e:
                last_err = e
                # Retry-After가 있으면 따르고, 없으면 최소 2초 + 지수 백오프.
                wait = e.retry_after if e.retry_after is not None else max(2.0, float(2 ** attempt))
            except Exception as e:  # noqa: BLE001 — 5xx/연결 오류
                last_err = e
                wait = (2 ** attempt) * (1 if self.api_key else 3)  # 무키면 더 길게

            if attempt < _MAX_RETRIES:
                logger.warning("S2 요청 실패 (%d/%d): %s — %.0fs 대기",
                               attempt, _MAX_RETRIES, last_err, wait)
                time.sleep(wait)
            else:
                # 마지막 시도: 더 기다리지 않으므로 '대기' 문구 없이.
                logger.warning("S2 요청 실패 (%d/%d): %s", attempt, _MAX_RETRIES, last_err)
        raise RuntimeError(f"S2 요청 최종 실패: {last_err}") from last_err

    # ------------------------------------------------------------- 매핑

    @staticmethod
    def _to_paper(item: dict, topic_name: str) -> Paper | None:
        title = (item.get("title") or "").strip()
        if not title:
            return None

        ext = item.get("externalIds") or {}
        arxiv_id_raw = ext.get("ArXiv")
        arxiv_id = arxiv_base_id(arxiv_id_raw) if arxiv_id_raw else None
        doi = ext.get("DOI")

        try:
            uid = make_uid(arxiv_id, doi, item.get("paperId"))
        except ValueError:
            return None  # 식별자가 전혀 없는 레코드는 dedupe 불가 → 스킵

        # PDF URL 우선순위: openAccessPdf → arXiv PDF → None
        pdf_url: str | None = None
        oa = item.get("openAccessPdf") or {}
        if oa.get("url"):
            pdf_url = oa["url"]
        elif arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

        # 사람이 여는 링크 우선순위: arXiv abs → doi.org → S2 페이지
        if arxiv_id:
            url = f"https://arxiv.org/abs/{arxiv_id}"
        elif doi:
            url = f"https://doi.org/{doi}"
        else:
            url = item.get("url") or f"https://www.semanticscholar.org/paper/{item.get('paperId')}"

        return Paper(
            uid=uid,
            title=" ".join(title.split()),
            authors=[a.get("name", "") for a in (item.get("authors") or []) if a.get("name")],
            abstract=(item.get("abstract") or None),
            venue=(item.get("venue") or None),
            year=item.get("year"),
            url=url,
            pdf_url=pdf_url,
            arxiv_id=arxiv_id,
            doi=doi,
            citation_count=item.get("citationCount"),
            source="semantic_scholar",
            topic_name=topic_name,
        )

    # ------------------------------------------------------------- fetch

    def fetch(self, topic: Topic) -> list[Paper]:
        if topic.s2 is None:
            return []

        s2 = topic.s2
        # 정렬 명시 — 미지정 시 bulk는 paperId 순(사실상 무작위)이라
        # 앞 max_results건만 보는 우리 로직이 원하는 논문을 놓친다.
        #   recency   → publicationDate:desc (최신순, 기본)
        #   citations → citationCount:desc  (인용순 — year 범위 내 저명작 수집용)
        sort_param = "citationCount:desc" if s2.sort == "citations" else "publicationDate:desc"
        params: dict = {"query": s2.query, "fields": _FIELDS, "sort": sort_param}
        if s2.venues:
            params["venue"] = ",".join(s2.venues)
        if s2.year:
            params["year"] = s2.year
        if s2.min_citations > 0:
            params["minCitationCount"] = str(s2.min_citations)

        # lookback: 0이면 비활성 — topic의 year 필터에 시간 범위를 위임.
        lookback_days = self.cfg.fetch.s2_lookback_days
        use_lookback = (not self.backfill) and lookback_days > 0
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        max_results = self.cfg.fetch.max_results_per_query

        papers: list[Paper] = []
        seen_pages = 0
        token: str | None = None
        reached_cutoff = False

        while not reached_cutoff and len(papers) < max_results:
            if token:
                params["token"] = token
            data = self._get(params)
            seen_pages += 1

            for item in data.get("data") or []:
                if len(papers) >= max_results:
                    break

                paper = self._to_paper(item, topic.name)
                if paper is None:
                    continue

                # lookback 필터 (--backfill 또는 s2_lookback_days=0이면 미적용).
                # 최신순 정렬일 때만 "이후는 전부 더 오래됨"이 성립해 조기 중단 가능.
                # 인용순 정렬은 날짜 순서가 아니므로 아이템 단위로만 건너뛴다.
                if use_lookback:
                    pub = item.get("publicationDate")  # "YYYY-MM-DD" 또는 None
                    if pub:
                        try:
                            pub_dt = datetime.strptime(pub, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                            if pub_dt < cutoff:
                                if s2.sort == "recency":
                                    reached_cutoff = True
                                    break
                                continue  # citations 정렬: 이 아이템만 스킵
                        except ValueError:
                            pass  # 날짜 형식 이상 → 필터하지 않고 통과

                papers.append(paper)

            token = data.get("token")
            if not token:
                break
            # 페이지 간 간격은 전역 limiter(_get 진입 시)가 보장하므로 별도 sleep 불필요.

        logger.info(
            "S2 '%s': %d편 수집 (%d페이지%s%s)",
            topic.name, len(papers), seen_pages,
            ", lookback 도달" if reached_cutoff else "",
            ", backfill" if self.backfill else "",
        )
        return papers
