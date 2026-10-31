"""파이프라인 공통 데이터 구조.

uid 규칙: 피드에 실린 글 링크를 정규화한 URL (scheme·host 소문자, fragment·utm_* 제거,
끝의 '/' 제거). 리다이렉트 후 최종 URL이 아니라 피드 링크 기준이라 매일 같은 값이 나온다.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

_OWN_FIG_NO_RE = re.compile(r"^(fig(ure)?\.?|table)\s*\d+", re.I)
_TRACKING_PREFIXES = ("utm_",)
_TRACKING_NAMES = {"fbclid", "gclid", "mc_cid", "mc_eid", "ref", "ncid"}


def _is_tracking(key: str) -> bool:
    k = key.lower()
    return k.startswith(_TRACKING_PREFIXES) or k in _TRACKING_NAMES


def canonical_url(url: str) -> str:
    """dedupe 키용 URL 정규화."""
    p = urlsplit(url.strip())
    query = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True) if not _is_tracking(k)]
    path = p.path.rstrip("/") or "/"
    return urlunsplit((p.scheme.lower(), p.netloc.lower(), path, urlencode(query), ""))


@dataclass
class Article:
    uid: str
    source: str                    # SourceConfig.name
    source_label: str              # Slack 표시용 출처 이름
    title: str
    url: str                       # 사람이 여는 링크 (피드 링크)
    published: datetime | None     # tz-aware(UTC). 피드에 날짜가 없으면 None
    categories: list[str] = field(default_factory=list)
    feed_summary: str = ""         # 피드 description/summary (평문) — judge 입력
    feed_html: str = ""            # 피드에 실린 본문 HTML (content:encoded 등) — 페이지 추출 폴백
    channel: str | None = None     # 발행 채널 (None이면 slack.channel)
    org: str = ""                  # 기관 (하루 발행 상한 단위)
    judge_score: float | None = None
    judge_reason: str | None = None

    def to_payload(self) -> str:
        """백로그 보관용 JSON (피드에서 사라진 뒤에도 발행할 수 있도록 메타데이터 전체)."""
        d = asdict(self)
        d["published"] = self.published.isoformat() if self.published else None
        return json.dumps(d, ensure_ascii=False)

    @classmethod
    def from_payload(cls, payload: str) -> "Article":
        d = json.loads(payload)
        d["published"] = datetime.fromisoformat(d["published"]) if d.get("published") else None
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class ImageRef:
    """본문에서 찾은 그림 후보 (아직 다운로드 전)."""
    url: str
    no: int = 0                    # 본문 텍스트의 [그림 N] 번호 (hero는 0)
    caption: str = ""
    alt: str = ""
    context: str = ""              # 그림 직전 문단 일부 — LLM 그림 선택용
    is_hero: bool = False          # og:image 대표 이미지


@dataclass
class FetchedImage:
    ref: ImageRef
    data: bytes
    ext: str                       # "png" | "jpg" | "gif"
    width: int
    height: int

    @property
    def label(self) -> str:
        """Slack 파일 제목 / 미리보기용 한 줄 설명."""
        text = self.ref.caption or self.ref.alt
        head = "대표 이미지" if self.ref.is_hero else f"그림 {self.ref.no}"
        if not text:
            return head
        text = " ".join(text.split())
        if len(text) > 180:
            text = text[:179].rstrip() + "…"
        if _OWN_FIG_NO_RE.match(text):            # 캡션에 원문 번호("Figure 3:")가 있으면 그대로
            return text
        return f"{head} — {text}"


@dataclass
class Extracted:
    text: str                      # 마크다운풍 본문 ([그림 N] 마커 포함)
    images: list[ImageRef]         # 본문 등장 순 (hero는 정책에 따라 맨 앞)
    links: list[tuple[str, str]]   # (라벨, URL) — 논문·코드·데이터 링크
    videos: list[str]              # YouTube 등 영상 링크
    final_url: str                 # 리다이렉트 후 최종 URL
    origin: str                    # "page" | "feed" — 본문을 어디서 얻었나
    title: str = ""                # 페이지 og:title/<title> (--url 수동 실행 시 제목으로 사용)
