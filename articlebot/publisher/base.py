"""Publisher interface and helpers shared by the Slack and Zulip publishers.
발행 인터페이스와 공용 도우미(한 줄 요약 추출, 파일명 슬러그).
"""
from __future__ import annotations

import abc
import re

from ..models import Article, Extracted, FetchedImage


class PublishConfigError(Exception):
    """전송 '설정' 문제(채널/토큰/권한). 글마다 재시도해도 소용없으므로 실행 전체를 중단한다."""


_ONE_LINER_RE = re.compile(r"###\s*1\.\s*한\s*줄\s*요약\s*\n+(.*?)(?=\n###|\Z)", re.DOTALL)


_MD_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")


_MD_HEADING_RE = re.compile(r"^#{1,6}\s*(.+)$", re.MULTILINE)


def extract_one_liner(summary: str, max_chars: int = 300) -> str:
    m = _ONE_LINER_RE.search(summary)
    text = " ".join((m.group(1) if m else summary).split())
    text = _MD_HEADING_RE.sub(r"\1", _MD_BOLD_RE.sub(r"\1", text)).strip()
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    return text


def _slug(article: Article) -> str:
    tail = article.uid.rstrip("/").rsplit("/", 1)[-1] or "article"
    return re.sub(r"[^A-Za-z0-9._-]+", "-", f"{article.source}_{tail}")[:100]


class Publisher(abc.ABC):
    @abc.abstractmethod
    def publish(self, article: Article, summary: str, ex: Extracted, images: list[FetchedImage]) -> None:
        ...
