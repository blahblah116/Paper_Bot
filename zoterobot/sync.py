"""Fill the text cache from Zotero (and, optionally, PDFs that exist only in the Drive folder).
Zotero 라이브러리(+Drive 폴더)로 본문 캐시 채우기 — 바뀐 항목만 본문을 다시 추출한다.

본문 출처 우선순위:
  1) Zotero 본문 색인 API — 데스크톱 Zotero가 추출해 동기화한 텍스트 (실측: Drive 링크 파일은 대개 없음)
  2) PDF 직접 추출 — 링크 파일(attachments:…)은 rclone으로 Drive에서, 저장소 파일은 Zotero API로 받아 pymupdf
  3) 둘 다 없으면 제목·초록만. PDF 첨부는 있는데 파일을 못 받으면 'missing'으로 두고 _RETRY_HOURS마다 재시도.
동기화는 라이브러리 버전이 바뀌었을 때만 돌고(요청 1건으로 확인), 항목·첨부 버전 서명(sig)이 달라진 논문만 다시 추출한다.
drive_scan이면 Drive 폴더의 PDF 중 Zotero 링크가 가리키지 않는 것을 'Drive 전용' 문서로 넣는다
(제목: PDF 메타데이터 → 파일명, 컬렉션: 상위 폴더, id: 경로 해시 'D…').
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
import unicodedata
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import pymupdf

from paperbot.extractor import extract_text, strip_references

from .drive import DriveError, DriveUploader
from .textcache import DRIVE_TYPE, TEXT_MAX, TextCache
from .zotero import ZoteroClient, collection_paths

logger = logging.getLogger(__name__)

_MIN_FULLTEXT = 500          # 이보다 짧은 Zotero 본문 색인은 무시하고 PDF 추출 시도
_RETRY_HOURS = 6             # 본문을 못 받은 PDF(text_source='missing') 재시도 간격
_FNAME_RE = re.compile(r"^(?P<who>[^-]+?) - (?:(?P<year>\d{4}) - )?(?P<title>.+)$")   # Zotero 파일명 형식


# ---------------------------------------------------------------- 메타데이터 헬퍼

def _clean(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()[:TEXT_MAX]


def nfc(s: str) -> str:
    """Drive·Zotero 경로 비교용 — macOS에서 만든 한글 파일명은 NFD일 수 있다."""
    return unicodedata.normalize("NFC", s or "")


def item_meta(item: dict, coll_names: dict[str, str]) -> dict:
    """Zotero 항목 JSON → 캐시 메타데이터."""
    authors = []
    for c in item.get("creators") or []:
        name = c.get("name") or " ".join(p for p in (c.get("firstName"), c.get("lastName")) if p)
        if name.strip():
            authors.append(name.strip())
    year = (re.search(r"\d{4}", item.get("date") or "") or [""])[0]
    return {
        "title": item.get("title") or "(제목 없음)",
        "authors": authors, "year": year,
        "collections": [coll_names[k] for k in item.get("collections") or [] if k in coll_names],
        "abstract": item.get("abstractNote") or "", "url": item.get("url") or "",
        "item_type": item.get("itemType") or "",
    }


def item_sig(item: dict, kids: list[dict]) -> str:
    return f"{item.get('version')}|" + ",".join(sorted(f"{a['key']}:{a.get('version')}" for a in kids))


def drive_key(path: str) -> str:
    """Drive 전용 문서 id — Zotero 키처럼 8자 대문자 (경로 해시, 'D'로 시작)."""
    return "D" + hashlib.sha1(nfc(path).encode()).hexdigest()[:7].upper()


def pdf_metadata(local: str) -> dict:
    try:
        with pymupdf.open(local) as doc:
            return dict(doc.metadata or {})
    except Exception:  # noqa: BLE001
        return {}


def _good_title(t: str) -> bool:
    t = (t or "").strip()
    return len(t) >= 12 and " " in t and not re.search(r"(?i)^(microsoft|untitled)|\.(pdf|dvi|tex|docx?)$", t)


def drive_meta(path: str, file_id: str | None, pdfmeta: dict | None = None) -> dict:
    """Drive 전용 PDF 메타데이터 — 제목: PDF 메타데이터 → 파일명('저자 등 - 연도 - 제목' 형식이면 분해).
    컬렉션 = 상위 폴더 경로('ICML/2026' — Zotero 컬렉션 경로 표기와 같게)."""
    folder, _, fname = path.rpartition("/")
    stem = fname[:-4] if fname.lower().endswith(".pdf") else fname
    who, year, title = "", "", stem
    m = _FNAME_RE.match(stem)
    if m:
        who, year, title = m["who"].strip(), m["year"] or "", m["title"].strip()
    pdfmeta = pdfmeta or {}
    if _good_title(pdfmeta.get("title")):
        title = " ".join(pdfmeta["title"].split())
    authors = [a.strip() for a in re.split(r"[;,]| and ", pdfmeta.get("author") or "") if a.strip()] or \
        ([who] if who else [])
    return {"title": title, "authors": authors, "year": year,
            "collections": [folder] if folder else [],
            "abstract": "", "url": f"https://drive.google.com/file/d/{file_id}/view" if file_id else "",
            "item_type": DRIVE_TYPE}


# ---------------------------------------------------------------- 본문 확보

def _fetch_text(zot: ZoteroClient, drive: DriveUploader | None, link_base: str,
                kids: list[dict]) -> tuple[str, str]:
    """항목의 첨부들에서 본문 텍스트 확보 → (텍스트, 출처). 출처: zotero_fulltext | pdf | missing | none."""
    pdfs = [a for a in kids if a.get("contentType") == "application/pdf"]
    others = [a for a in kids if a not in pdfs and a.get("linkMode") != "linked_url"]
    for a in pdfs + others:
        t = zot.fulltext(a["key"])
        if t and len(t) >= _MIN_FULLTEXT:
            return _clean(strip_references(t)), "zotero_fulltext"
    for a in pdfs:
        with tempfile.TemporaryDirectory(prefix="zoterobot-") as tmp:
            local = os.path.join(tmp, "a.pdf")
            try:
                mode, path = a.get("linkMode"), a.get("path") or ""
                if mode == "linked_file" and drive and path.startswith(link_base):
                    drive.download(path[len(link_base):], local)
                elif mode in ("imported_file", "imported_url"):
                    zot.download_file(a["key"], local)
                else:
                    continue
                text = extract_text(local)
            except Exception as e:  # noqa: BLE001 — Drive/Zotero 오류, pymupdf 파싱 오류 등: 다음 첨부로
                logger.warning("PDF 본문 추출 실패 %s: %s", a["key"], e)
                continue
        if text.strip():
            return _clean(strip_references(text)), "pdf"
    return "", ("missing" if pdfs else "none")


# ---------------------------------------------------------------- 동기화

def _store_linked(cache: TextCache, atts: list[dict], link_base: str) -> None:
    """Zotero 링크 첨부가 가리키는 Drive 상대경로 목록 — Drive 전용 PDF 판별용."""
    linked = sorted({nfc(a["path"][len(link_base):]) for a in atts
                     if a.get("linkMode") == "linked_file" and (a.get("path") or "").startswith(link_base)})
    cache.set_meta("linked_paths", json.dumps(linked, ensure_ascii=False))


def _sync_zotero(cache: TextCache, zot: ZoteroClient, drive: DriveUploader | None,
                 link_base: str, force: bool) -> dict:
    since = None if force else (int(cache.get_meta("library_version", "0")) or None)
    ver = zot.library_version(since)
    cache.touch_checked()
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=_RETRY_HOURS)).isoformat()
    if ver is None and not cache.missing_retry_due(cutoff):
        return {"unchanged": True}

    items, ver = zot.top_items()
    atts = zot.attachments()
    _store_linked(cache, atts, link_base)
    by_parent: dict[str, list[dict]] = defaultdict(list)
    for a in atts:
        if a.get("parentItem"):
            by_parent[a["parentItem"]].append(a)
    coll_names = collection_paths(zot.collections())    # 'ICML/2026' — 저장 명령·Drive 폴더와 같은 표기
    known = cache.known_zotero()

    stats = {"added": 0, "updated": 0, "removed": 0, "meta_only": 0, "no_text": 0}
    for it in items:
        kids = by_parent.get(it["key"], [])
        meta = item_meta(it, coll_names)
        sig = item_sig(it, kids)
        old_sig, old_src, old_at = known.get(it["key"], (None, None, ""))
        # PDF 첨부는 있는데 파일을 못 받은 논문(Drive 미동기화 등)은 _RETRY_HOURS마다 다시 시도
        retry = old_src == "missing" and (old_at or "") < cutoff
        if not force and old_sig == sig and not retry:
            cache.update_collections(it["key"], meta["collections"])   # 본문은 그대로 — 컬렉션 변경만 반영
            stats["meta_only"] += 1
            continue
        text, source = _fetch_text(zot, drive, link_base, kids)
        if not text:
            stats["no_text"] += 1
        if retry and text:
            logger.info("본문 재시도 성공: %s", meta["title"][:60])
        stats["updated" if it["key"] in known else "added"] += 1
        cache.upsert(it["key"], meta, sig, text, source)
        logger.info("본문 캐시 %s [%s, %d자]: %s", it["key"], source, len(text), meta["title"][:60])
    gone = set(known) - {it["key"] for it in items}
    cache.delete(gone)
    stats["removed"] = len(gone)
    cache.set_meta("library_version", str(ver))
    cache.invalidate()
    logger.info("라이브러리 동기화 완료 (버전 %s): %s", ver, stats)
    return stats


def _sync_drive(cache: TextCache, drive: DriveUploader) -> dict:
    """Drive 폴더의 PDF 중 Zotero 링크가 가리키지 않는 것을 'Drive 전용' 문서로 추가·갱신·삭제."""
    linked = set(json.loads(cache.get_meta("linked_paths", "[]")))
    known = cache.known_drive()
    stats = {"added": 0, "updated": 0, "removed": 0, "files": 0}
    seen = set()
    paths: dict[str, str] = {}          # key → Drive 상대경로 (summarize_paper가 PDF를 다시 받을 때 씀)
    for f in drive.list_pdfs():
        path = nfc(f.get("Path", ""))
        if not path or path in linked:
            continue
        stats["files"] += 1
        key = drive_key(path)
        seen.add(key)
        paths[key] = path
        sig = f"drive|{f.get('Size')}|{f.get('ModTime')}"
        if known.get(key) == sig:
            continue
        text, meta = "", drive_meta(path, f.get("ID"))
        with tempfile.TemporaryDirectory(prefix="zoterobot-") as tmp:
            local = os.path.join(tmp, "a.pdf")
            try:
                drive.download(path, local)
                text = extract_text(local)
                meta = drive_meta(path, f.get("ID"), pdf_metadata(local))
            except Exception as e:  # noqa: BLE001 — 받기·파싱 실패는 다음 동기화에서 재시도
                logger.warning("Drive 전용 PDF 추출 실패 %s: %s", path, e)
                sig = "failed"
        text = _clean(strip_references(text)) if text.strip() else ""
        stats["updated" if key in known else "added"] += 1
        cache.upsert(key, meta, sig, text, "pdf" if text else "missing")
        logger.info("Drive 전용 문서 %s [%d자]: %s", key, len(text), path)
    cache.set_meta("drive_paths", json.dumps(paths, ensure_ascii=False))
    gone = set(known) - seen
    cache.delete(gone)
    stats["removed"] = len(gone)
    if stats["added"] or stats["updated"] or stats["removed"]:
        cache.invalidate()
        logger.info("Drive 폴더 동기화: %s", stats)
    return stats


def sync_library(cache: TextCache, zot: ZoteroClient, drive: DriveUploader | None, link_base: str,
                 force: bool = False, drive_scan: bool = False) -> dict:
    """Zotero 라이브러리 동기화 + (drive_scan) Drive 폴더에만 있는 PDF 반영. 통계 dict 반환."""
    with cache.lock:
        stats = _sync_zotero(cache, zot, drive, link_base, force)
        if drive_scan and drive is not None:
            if not cache.get_meta("linked_paths"):
                _store_linked(cache, zot.attachments(), link_base)
            try:
                d = _sync_drive(cache, drive)
            except DriveError as e:
                logger.warning("Drive 폴더 훑기 실패 — Drive 전용 문서는 이전 상태 유지: %s", e)
                d = {"error": str(e)[:200]}
            stats = {**stats, "drive": d}
        return stats
