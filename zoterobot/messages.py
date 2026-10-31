"""Parse commands (Slack mention / Zulip DM) and format replies — pure functions, no I/O.
명령 파싱(Slack 멘션·Zulip DM)과 답글 포맷 — 순수 함수(I/O 없음).

답글 표기는 Fmt(SLACK / ZULIP)로 플랫폼을 가른다: 볼드·링크·이스케이프·LLM 마크다운 렌더링.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from paperbot.publisher import md_to_mrkdwn

from .drive import DriveUploader
from .library import SaveResult
from .resolver import PaperRef, clean_slack_text, find_refs
from .textcache import TextCache

TITLE_RE = re.compile(r"^\*{1,2}(.+?)\*{1,2}\s*$", re.MULTILINE)   # paperbot 원글 첫 줄 '*제목*'(Slack) / '**제목**'(Zulip)
HELP_WORDS = {"", "help", "도움말", "사용법", "list", "목록"}
SAVE_WORDS = {"저장", "save", "/save"}       # Zulip DM에서 저장 명령을 여는 첫 단어 (나머지는 질문)
_ID_PREFIX_WORDS = {"arxiv:", "doi:"}

# Zulip DM 본문 정리용 — quote-and-reply 블록, 멘션(@**name** / @_**name|id**), "[said](link):", 마크다운 링크, 메시지 링크
_ZULIP_QUOTE_RE = re.compile(r"`{3,}\s*quote[^\n]*\n(.*?)\n`{3,}", re.DOTALL)
_ZULIP_MENTION_RE = re.compile(r"@_?\*\*[^*]+\*\*")
_ZULIP_SAID_RE = re.compile(r"\[[^\]]*\]\([^)]*/near/\d+[^)]*\):?")
_ZULIP_MDLINK_RE = re.compile(r"\[([^\]]*)\]\((https?://[^)\s]+)\)")
_ZULIP_NEAR_RE = re.compile(r"/near/(\d+)")
# 'Copy link to message'가 주는 다른 형식: #**스트림>topic@메시지id** (별표 없이 붙여넣은 '스트림>topic@id'도)
_ZULIP_MSGREF_RE = re.compile(r"#?\*{0,2}[^\s*#>]+>[^\s*@]*@(\d{4,})\*{0,2}")


# ---------------------------------------------------------------- 플랫폼별 표기

class Fmt:
    """Slack mrkdwn 표기 (기본)."""
    name = "slack"
    math_style = "plain"      # 답변 수식 규칙 (Slack은 수식을 렌더링하지 않는다)
    max_chars = 39000            # 한 메시지에 넣을 최종 답 상한 (Slack 40,000)

    def bold(self, s: str) -> str:
        return f"*{s}*"

    def link(self, url: str, label: str) -> str:
        return f"<{url}|{label}>"

    def esc(self, s: str) -> str:
        return esc(s)

    def render(self, markdown: str) -> str:
        """LLM 답(마크다운) → 플랫폼 표기."""
        return md_to_mrkdwn(markdown)


class ZulipFmt(Fmt):
    """Zulip 마크다운 표기 — 표준 마크다운에 가까워 LLM 답은 그대로 보낸다. 10,000자 초과는 봇이 나눠 보낸다."""
    name = "zulip"
    math_style = "latex"      # Zulip은 $$LaTeX$$ 인라인, ```math 블록을 렌더링한다
    max_chars = 60000

    def bold(self, s: str) -> str:
        return f"**{s}**"

    def link(self, url: str, label: str) -> str:
        return f"[{label}]({url})"

    def esc(self, s: str) -> str:
        return s

    def render(self, markdown: str) -> str:
        return markdown


SLACK = Fmt()
ZULIP = ZulipFmt()


# ---------------------------------------------------------------- 명령 파싱


def parse_command(raw_text: str) -> tuple[list[PaperRef], str]:
    """멘션 본문 → (논문 식별자들, 컬렉션명). 링크·식별자를 뺀 나머지 단어가 컬렉션명."""
    text = clean_slack_text(raw_text)
    refs = [r for _, _, r in find_refs(text)]
    words = [w for w in text.split()
             if not (w.lower().startswith(("http://", "https://")) or find_refs(w)
                     or w.lower() in _ID_PREFIX_WORDS)]
    return refs, " ".join(words).strip(" \"'“”‘’`")


def ref_token(r: PaperRef) -> str:
    """PaperRef → 명령 본문에 다시 넣을 수 있는 식별자 토큰 (parse_command가 그대로 알아듣는 형식)."""
    if r.kind == "arxiv":
        return f"arxiv:{r.value}"
    if r.kind == "doi":
        return f"doi:{r.value}"
    return f"https://www.semanticscholar.org/paper/{r.value}"


@dataclass(frozen=True)
class ZulipCommand:
    kind: str                       # "save" | "chat"
    text: str                       # save: 키워드를 뺀 명령 본문(+인용 속 식별자) / chat: 멘션을 뺀 질문
    message_ids: tuple[int, ...]    # 본문의 Zulip 메시지 링크(/near/<id>) — 식별자가 없을 때 그 메시지에서 찾는다


def parse_zulip_dm(content: str) -> ZulipCommand:
    """Zulip DM 본문 → 저장 명령인지 질문인지.

    첫 단어가 SAVE_WORDS면 저장: `저장 ICML/2026 <arXiv 링크>`. 논문 topic에서 'Quote and reply'로 가져온
    인용 블록(```quote … ```)이나 메시지 링크(…/near/123)를 붙여도 된다 — 인용 속 식별자는 명령에 덧붙이고,
    식별자가 전혀 없으면 message_ids로 원글을 조회한다. 그 밖의 DM은 전부 질문(chat).
    """
    quotes = _ZULIP_QUOTE_RE.findall(content)
    body = _ZULIP_QUOTE_RE.sub(" ", content)
    message_ids = tuple(dict.fromkeys(int(m) for m in [*_ZULIP_NEAR_RE.findall(content), *_ZULIP_MSGREF_RE.findall(content)]))
    body = _ZULIP_SAID_RE.sub(" ", body)
    body = _ZULIP_MSGREF_RE.sub(" ", body)     # 메시지 참조 토큰은 컬렉션명/명령 본문에서 뺀다
    body = _ZULIP_MENTION_RE.sub(" ", body)
    words = body.split()
    if words and words[0].lower() in SAVE_WORDS:
        rest = _ZULIP_MDLINK_RE.sub(r"\2", " ".join(words[1:]))
        extra = [ref_token(r) for _, _, r in find_refs("\n".join(quotes))]
        return ZulipCommand("save", " ".join([rest, *extra]).strip(), message_ids)
    return ZulipCommand("chat", body.strip(), ())


# ---------------------------------------------------------------- 답글 포맷

def esc(s: str) -> str:
    """Slack mrkdwn 특수문자 이스케이프 (Zulip은 엔티티를 그대로 보여주므로 무해)."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def format_result(r: SaveResult, drive: DriveUploader, fmt: Fmt = SLACK) -> str:
    """저장 완료 답글."""
    coll = f"Zotero 컬렉션 `{fmt.esc(r.collection)}`"
    if r.collection_created:
        made = "/".join(r.collection_created)
        coll += " (새로 만듦)" if made == r.collection else f" (하위 `{fmt.esc(made)}` 새로 만듦)"
    if r.existed:
        coll += " — 이미 라이브러리에 있던 항목에 추가"
    if r.pdf == "uploaded":
        pdf = f"PDF → Drive `{fmt.esc(drive.target(r.drive_path or ''))}` (Zotero 링크 첨부)"
    elif r.pdf == "already":
        pdf = "PDF는 이미 첨부되어 있어 컬렉션만 추가했습니다"
    else:
        pdf = f"⚠️ {r.note or '공개 PDF 없음 — 메타데이터만 저장'}"
    return f"✅ {fmt.bold(fmt.esc(r.title))}\n• {coll}\n• {pdf}"


def format_usage(collections: list[str] | None, remote: str, error: str | None = None,
                 fmt: Fmt = SLACK) -> str:
    """사용법 답글 (+ 현재 컬렉션 목록). Slack은 스레드 멘션, Zulip은 DM 저장 명령 기준."""
    if error:
        colls = f"(조회 실패: {fmt.esc(error)[:200]})"
    else:
        colls = ", ".join(f"`{fmt.esc(n)}`" for n in (collections or [])[:60]) or "(없음)"
    if fmt.name == "zulip":
        head = (f"{fmt.bold('사용법')} — DM으로 `저장 <컬렉션명> <arXiv·DOI 링크>`\n"
                f"→ 그 논문을 Zotero 컬렉션과 Drive `{fmt.esc(remote)}/<컬렉션명>/`에 저장 "
                "(컬렉션이 없으면 새로 만듦)\n"
                "논문 topic에서 'Quote and reply'(인용)나 메시지 링크를 붙여 `저장 <컬렉션명>`만 적어도 됩니다\n")
    else:
        head = (f"{fmt.bold('사용법')} — 논문 스레드에서 `@zoterobot <컬렉션명>`\n"
                f"→ 원글 논문을 Zotero 컬렉션과 Drive `{fmt.esc(remote)}/<컬렉션명>/`에 저장 "
                "(컬렉션이 없으면 새로 만듦)\n")
    return (head +
            "하위 컬렉션은 `ICML/2026`처럼 `/`로 → Zotero `ICML > 2026`, Drive `…/ICML/2026/` "
            "(이름만 적으면 최상위 컬렉션)\n"
            + ("" if fmt.name == "zulip" else "스레드 밖에서는 `@zoterobot <컬렉션명> <arXiv·DOI 링크>` · 질문은 DM으로\n")
            + f"{fmt.bold('현재 컬렉션')}: {colls}")


def format_sources(cache: TextCache, keys: list[str], limit: int = 5, fmt: Fmt = SLACK) -> str:
    """DM 답변 끝에 붙는 '참고한 논문' 목록."""
    lines = []
    for k in keys[:limit]:
        d = cache.get(k)
        if not d:
            continue
        colls = f" ({', '.join(d.collections)})" if d.collections else ""
        origin = f" [{d.origin}]" if d.feed else ""     # 수집 문서는 라이브러리 논문과 구분
        title = fmt.link(d.url, fmt.esc(d.title)) if d.url else fmt.esc(d.title)
        lines.append(f"• {title}{colls}{origin}")
    return (f"\n\n📚 {fmt.bold('참고한 논문')}\n" + "\n".join(lines)) if lines else ""
