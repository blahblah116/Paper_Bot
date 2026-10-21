"""Parse Slack commands and format replies — pure functions, no I/O.
Slack 명령 파싱과 답글 포맷 — 순수 함수(I/O 없음).
"""
from __future__ import annotations

import re

from .drive import DriveUploader
from .library import SaveResult
from .resolver import PaperRef, clean_slack_text, find_refs
from .textcache import TextCache

TITLE_RE = re.compile(r"^\*(.+?)\*\s*$", re.MULTILINE)     # paperbot 메인 메시지 첫 줄 '*제목*'
HELP_WORDS = {"", "help", "도움말", "사용법", "list", "목록"}
_ID_PREFIX_WORDS = {"arxiv:", "doi:"}


def parse_command(raw_text: str) -> tuple[list[PaperRef], str]:
    """멘션 본문 → (논문 식별자들, 컬렉션명). 링크·식별자를 뺀 나머지 단어가 컬렉션명."""
    text = clean_slack_text(raw_text)
    refs = [r for _, _, r in find_refs(text)]
    words = [w for w in text.split()
             if not (w.lower().startswith(("http://", "https://")) or find_refs(w)
                     or w.lower() in _ID_PREFIX_WORDS)]
    return refs, " ".join(words).strip(" \"'“”‘’`")


def esc(s: str) -> str:
    """Slack mrkdwn 특수문자 이스케이프."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def format_result(r: SaveResult, drive: DriveUploader) -> str:
    """저장 완료 답글."""
    coll = f"Zotero 컬렉션 `{esc(r.collection)}`" + (" (새로 만듦)" if r.collection_created else "")
    if r.existed:
        coll += " — 이미 라이브러리에 있던 항목에 추가"
    if r.pdf == "uploaded":
        pdf = f"PDF → Drive `{esc(drive.target(r.drive_path or ''))}` (Zotero 링크 첨부)"
    elif r.pdf == "already":
        pdf = "PDF는 이미 첨부되어 있어 컬렉션만 추가했습니다"
    else:
        pdf = f"⚠️ {r.note or '공개 PDF 없음 — 메타데이터만 저장'}"
    return f"✅ *{esc(r.title)}*\n• {coll}\n• {pdf}"


def format_usage(collections: list[str] | None, remote: str, error: str | None = None) -> str:
    """사용법 답글 (+ 현재 컬렉션 목록)."""
    if error:
        colls = f"(조회 실패: {esc(error)[:200]})"
    else:
        colls = ", ".join(f"`{esc(n)}`" for n in (collections or [])[:60]) or "(없음)"
    return ("*사용법* — 논문 스레드에서 `@zoterobot <컬렉션명>`\n"
            f"→ 원글 논문을 Zotero 컬렉션과 Drive `{esc(remote)}/<컬렉션명>/`에 저장 "
            "(컬렉션이 없으면 새로 만듦)\n"
            "스레드 밖에서는 `@zoterobot <컬렉션명> <arXiv·DOI 링크>` · 질문은 DM으로\n"
            f"*현재 컬렉션*: {colls}")


def format_sources(cache: TextCache, keys: list[str], limit: int = 5) -> str:
    """DM 답변 끝에 붙는 '참고한 논문' 목록."""
    lines = []
    for k in keys[:limit]:
        d = cache.get(k)
        if not d:
            continue
        colls = f" ({', '.join(d.collections)})" if d.collections else ""
        title = f"<{d.url}|{esc(d.title)}>" if d.url else esc(d.title)
        lines.append(f"• {title}{colls}")
    return ("\n\n📚 *참고한 논문*\n" + "\n".join(lines)) if lines else ""
