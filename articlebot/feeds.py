"""RSS 2.0 / Atom 피드 → Article 리스트.

feedparser 없이 lxml로 직접 파싱한다 (네임스페이스 무관하게 local-name으로 접근).
외부 엔티티·네트워크 접근은 끈다 (XXE 방지).
"""
from __future__ import annotations

import email.utils
import logging
import re
from datetime import datetime, timezone
from html import unescape

import lxml.html
from lxml import etree

from .config import SourceConfig
from .http import Http
from .models import Article, canonical_url

logger = logging.getLogger(__name__)

_WS_RE = re.compile(r"\s+")


class FeedError(Exception):
    pass


def _children(el, *names: str) -> list:
    return [c for c in el if isinstance(c.tag, str) and etree.QName(c).localname in names]


def _first_text(el, *names: str) -> str:
    """names 순서대로 처음 나오는 자식의 텍스트(하위 노드 포함)."""
    for name in names:
        for c in _children(el, name):
            text = "".join(c.itertext()).strip()
            if text:
                return text
    return ""


def html_to_text(html: str) -> str:
    """피드 description 같은 짧은 HTML 조각 → 평문."""
    if not html or not html.strip():
        return ""
    if "<" not in html:
        return _WS_RE.sub(" ", unescape(html)).strip()
    try:
        frag = lxml.html.fragment_fromstring(html, create_parent="div")
    except (etree.ParserError, ValueError):
        return _WS_RE.sub(" ", unescape(re.sub(r"<[^>]+>", " ", html))).strip()
    for bad in frag.xpath(".//script|.//style"):
        bad.drop_tree()
    return _WS_RE.sub(" ", frag.text_content()).strip()


def parse_date(text: str) -> datetime | None:
    """RFC 822(RSS) / ISO 8601(Atom) → tz-aware UTC datetime."""
    text = (text or "").strip()
    if not text:
        return None
    dt: datetime | None = None
    try:
        dt = email.utils.parsedate_to_datetime(text)
    except (TypeError, ValueError, IndexError):
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _link(item) -> str:
    # Atom: <link rel="alternate" href="..."/> (rel 없으면 alternate로 간주)
    for c in _children(item, "link"):
        href = c.get("href")
        if href and c.get("rel", "alternate") == "alternate":
            return href.strip()
    # RSS: <link>URL</link>
    for c in _children(item, "link"):
        if c.text and c.text.strip():
            return c.text.strip()
    # 최후: guid가 URL인 경우
    guid = _first_text(item, "guid", "id")
    return guid if guid.startswith(("http://", "https://")) else ""


def _categories(item) -> list[str]:
    out: list[str] = []
    for c in _children(item, "category"):
        val = (c.get("term") or c.text or "").strip()   # Atom은 term 속성, RSS는 텍스트
        if val and val not in out:
            out.append(unescape(val))
    return out


def parse_feed(data: bytes, src: SourceConfig) -> list[Article]:
    parser = etree.XMLParser(recover=True, resolve_entities=False, no_network=True, huge_tree=False)
    try:
        root = etree.fromstring(data, parser)
    except etree.XMLSyntaxError as e:
        raise FeedError(f"피드 파싱 실패: {e}") from e
    if root is None:
        raise FeedError("피드가 비어 있거나 XML이 아님")

    items = root.xpath('//*[local-name()="item"] | //*[local-name()="entry"]')
    out: list[Article] = []
    for item in items:
        title = html_to_text(_first_text(item, "title"))
        link = _link(item)
        if not title or not link:
            continue
        published = parse_date(_first_text(item, "pubDate", "published", "date", "updated"))
        summary_html = _first_text(item, "description", "summary")
        content_html = _first_text(item, "encoded", "content")   # content:encoded(RSS) / content(Atom)
        out.append(Article(
            uid=canonical_url(link),
            source=src.name,
            source_label=src.label,
            title=title,
            url=link,
            published=published,
            categories=_categories(item),
            feed_summary=html_to_text(summary_html)[:3000],
            feed_html=content_html,
            channel=src.channel,
            org=src.org,
        ))
    return out


def category_allowed(article: Article, src: SourceConfig) -> bool:
    cats = {c.lower() for c in article.categories}
    if src.include_categories and not cats & {c.lower() for c in src.include_categories}:
        return False
    if src.exclude_categories and cats & {c.lower() for c in src.exclude_categories}:
        return False
    return True


def fetch_feed(http: Http, src: SourceConfig) -> list[Article]:
    resp = http.get(src.feed, max_bytes=10_000_000,
                    accept="application/rss+xml, application/atom+xml, application/xml;q=0.9, */*;q=0.8")
    articles = parse_feed(resp.content, src)
    logger.info("피드 %s: %d편", src.name, len(articles))
    return articles
