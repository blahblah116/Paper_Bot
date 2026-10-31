"""Chat cache hook: put each published article's body text and summary into zoterobot's text cache.
챗봇 캐시 연결 — 발행한 글의 본문(추출 텍스트)과 요약을 zoterobot 본문 캐시(storage.chat_text_db, sqlite)에 넣는다.

저장은 zoterobot/textcache.py의 TextCache.upsert_feed (같은 레포·venv). id는 'A' + uid 해시 7자, item_type은 'blog_article',
컬렉션 자리에 기관(org), 저자 자리에 출처 라벨. DM 챗봇이 search_papers(scope=feed)/read_paper로 찾아 읽는다.
- pipeline.process_article이 발행 직후 store_article()을 부른다. 실패는 경고만(발행은 끝났다).
- backfill_chat(pipeline)은 DB의 done 글(요약 저장됨)을 거슬러 본문을 다시 받아 채운다: `python -m articlebot --backfill-chat`.
"""
from __future__ import annotations

import logging

from zoterobot.textcache import ARTICLE_TYPE, TextCache, feed_key as _feed_key

from .config import Config
from .models import Article

logger = logging.getLogger(__name__)


def feed_key(uid: str) -> str:
    return _feed_key(ARTICLE_TYPE, uid)


def article_meta(a: Article) -> dict:
    return {"title": a.title or a.url, "authors": [a.source_label or a.source],
            "year": a.published.strftime("%Y") if a.published else "", "collections": [a.org] if a.org else [],
            "abstract": (a.feed_summary or "")[:2000], "url": a.url}


class ChatCache:
    """zoterobot 본문 캐시에 글 넣기만 하는 얇은 writer."""

    def __init__(self, db_path: str):
        self.cache = TextCache(db_path)

    def close(self) -> None:
        self.cache.close()

    def has_text(self, uid: str) -> bool:
        return self.cache.has_feed_text(ARTICLE_TYPE, uid)

    def upsert(self, a: Article, text: str, summary: str | None, model: str = "") -> str:
        return self.cache.upsert_feed(ARTICLE_TYPE, a.uid, article_meta(a), text, "web", summary, model)


def store_article(cfg: Config, a: Article, text: str, summary: str | None) -> str | None:
    """글 본문·요약을 캐시에. 반환: 캐시 id (껐거나 실패하면 None). 예외를 밖으로 내지 않는다."""
    path = cfg.storage.chat_text_db
    if not path:
        return None
    try:
        cache = ChatCache(cfg.resolve(path))
        try:
            key = cache.upsert(a, text, summary, cfg.llm.model)
        finally:
            cache.close()
        logger.info("챗봇 캐시 저장 %s ← %s (%d자%s)", key, a.uid, len(text or ""), ", 요약 포함" if summary else "")
        return key
    except Exception as e:  # noqa: BLE001 — 캐시는 부가 기능
        logger.warning("챗봇 캐시 저장 실패 %s: %s", a.uid, e)
        return None
