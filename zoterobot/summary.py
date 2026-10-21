"""summarize_paper tool: run paperbot's summarizer on the paper's full text (cache first, PDF from Drive as fallback).
summarize_paper 도구 — 논문 본문 전체를 paperbot 요약기로 전문 요약. 본문은 캐시(SQLite)의 추출 텍스트를 쓰고,
캐시에 본문이 없는 논문만 PDF를 Drive(또는 Zotero 저장소)에서 받아 추출한다.

- 캐시 본문은 동기화 때 같은 PDF에서 뽑은 것이라 내용이 같다 (2026-10-21 사용자 결정: 다시 받지 않는다).
- PDF 예비 경로: Zotero 항목은 API로 첨부를 조회해 linked_file(attachments:…) → rclone으로 Drive에서, imported_file/url →
  Zotero 저장소에서 받는다(Drive 우선). Drive 전용 문서는 동기화가 기록한 경로(meta 'drive_paths').
- 본문 처리는 paperbot extractor와 같다(strip_references → 정리 → max_input_chars 절단) — 아침 요약과 같은 입력.
- 요약 LLM: paperbot.summarizer.summarize (OpenAI 호환 /v1, 같은 모델이라 VRAM 추가 없음). 실측 편당 2~3분.
- 결과는 캐시 summaries 테이블에 저장 — 같은 논문은 다시 요약하지 않고(force 제외), read_paper 결과 앞에도 붙어
  후속 질문에 쓰인다.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from typing import Callable

from paperbot.extractor import extract_text, strip_references, truncate
from paperbot.models import Paper

from .drive import DriveUploader
from .sync import _clean as clean_text
from .textcache import DRIVE_TYPE, Doc, TextCache
from .zotero import ZoteroClient

logger = logging.getLogger(__name__)

SummarizeFn = Callable[[Paper, str], str]       # (논문 메타, 본문) → 요약 마크다운


class SummaryError(Exception):
    pass


@dataclass
class SummaryResult:
    key: str
    title: str
    summary: str
    source: str          # 'cache:pdf'·'cache:zotero_fulltext' | 예비 경로면 'drive:ICML/2026/x.pdf' | 'zotero:ATTKEY'
    chars: int           # 요약기에 넣은 본문 글자 수
    reused: bool         # 저장된 요약을 돌려줬는지
    seconds: float


def doc_to_paper(d: Doc) -> Paper:
    """캐시 문서 → paperbot 요약 프롬프트용 Paper (venue는 캐시에 없어 출처 표기만)."""
    return Paper(uid=d.key, title=d.title, authors=list(d.authors), abstract=d.abstract or None,
                 venue="Zotero library", year=int(d.year) if str(d.year).isdigit() else None, url=d.url,
                 pdf_url=None, arxiv_id=None, doi=None, citation_count=None, source="zotero",
                 topic_name=d.collections[0] if d.collections else "")


class PaperSummarizer:
    def __init__(self, cache: TextCache, zot: ZoteroClient, drive: DriveUploader | None, link_base: str,
                 summarize_fn: SummarizeFn, max_input_chars: int, model: str = ""):
        self.cache, self.zot, self.drive, self.link_base = cache, zot, drive, link_base
        self.summarize_fn, self.max_input_chars, self.model = summarize_fn, max_input_chars, model

    def locate(self, d: Doc) -> list[tuple[str, str]]:
        """PDF 후보 (kind, locator) — Drive 링크 우선, 그다음 Zotero 저장소."""
        if d.item_type == DRIVE_TYPE:
            path = json.loads(self.cache.get_meta("drive_paths", "{}")).get(d.key)
            return [("drive", path)] if path else []
        out = []
        for a in self.zot.children(d.key):
            path = a.get("path") or ""
            if a.get("itemType") != "attachment" or not (
                    a.get("contentType") == "application/pdf" or path.lower().endswith(".pdf")):
                continue
            mode = a.get("linkMode")
            if mode == "linked_file" and path.startswith(self.link_base):
                out.append(("drive", path[len(self.link_base):]))
            elif mode in ("imported_file", "imported_url"):
                out.append(("zotero", a["key"]))
        out.sort(key=lambda x: x[0] != "drive")
        return out

    def fetch_text(self, d: Doc, on_status=None) -> tuple[str, str]:
        """PDF를 받아 본문 텍스트 추출 → (본문, 출처 'drive:…'/'zotero:…')."""
        cands = self.locate(d)
        if not cands:
            raise SummaryError("이 논문의 PDF 첨부를 찾지 못했습니다")
        errors = []
        for kind, loc in cands:
            raw = ""
            with tempfile.TemporaryDirectory(prefix="zoterobot-sum-") as tmp:
                local = os.path.join(tmp, "paper.pdf")
                try:
                    if kind == "drive":
                        if self.drive is None:
                            raise SummaryError("Drive 접근이 설정되지 않았습니다")
                        if on_status:
                            on_status(f"📥 Drive에서 PDF 받는 중: {self.drive.target(loc)}")
                        self.drive.download(loc, local)
                    else:
                        if on_status:
                            on_status("📥 Zotero 저장소에서 PDF 받는 중")
                        self.zot.download_file(loc, local)
                    raw = extract_text(local)
                except Exception as e:  # noqa: BLE001 — 받기·파싱 실패는 다음 후보로
                    errors.append(f"{kind}:{loc} ({e})")
                    continue
            text = clean_text(strip_references(raw)) if raw and raw.strip() else ""
            if text:
                return text, f"{kind}:{loc}"
            errors.append(f"{kind}:{loc} (텍스트 없음 — 스캔 PDF?)")
        raise SummaryError("PDF를 읽지 못했습니다: " + "; ".join(errors))

    def summarize(self, key: str, force: bool = False, on_status=None) -> SummaryResult:
        d = self.cache.get(key)
        if d is None:
            raise SummaryError(f"id '{key}'인 논문이 없습니다 — search_papers/list_papers로 id를 확인하세요")
        if not force:
            old = self.cache.get_summary(d.key)
            if old:
                return SummaryResult(d.key, d.title, old["summary"], old["source"], old["chars"], True, 0.0)
        t0 = time.time()
        if d.text.strip():
            text, source = d.text, f"cache:{d.text_source or 'text'}"
        else:   # 캐시에 본문이 없을 때만 PDF를 직접 받는다
            text, source = self.fetch_text(d, on_status)
        text = truncate(text, self.max_input_chars)
        if on_status:
            on_status(f"🧠 전문 요약 생성 중 ({len(text):,}자 — 2~3분 걸려요)")
        try:
            summary = self.summarize_fn(doc_to_paper(d), text)
        except Exception as e:  # noqa: BLE001 — paperbot LLMError 등
            raise SummaryError(f"요약 모델 호출 실패: {e}") from e
        self.cache.set_summary(d.key, summary, source, len(text), self.model)
        secs = time.time() - t0
        logger.info("전문 요약 %s (%s, %d자 → %d자) %.0fs", d.key, source, len(text), len(summary), secs)
        return SummaryResult(d.key, d.title, summary, source, len(text), False, secs)
