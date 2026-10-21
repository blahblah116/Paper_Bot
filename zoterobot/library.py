"""논문 하나를 Zotero 컬렉션 + Google Drive 컬렉션 폴더에 저장 (스레드 '@zoterobot <컬렉션>').

순서 — 어느 단계에서 실패해도 같은 명령을 다시 보내면 이어서 처리되게(idempotent):
  1) 컬렉션 찾기/생성
  2) 라이브러리에 같은 논문이 있으면 재사용(컬렉션만 추가), 없으면 메타데이터 항목 생성
  3) 항목에 PDF가 이미 있으면 끝. 없으면 PDF 다운로드 → rclone으로 Drive에 업로드 → linked_file 첨부
"""
from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass

from paperbot.extractor import download_pdf, iter_candidate_urls

from .config import Config
from .drive import DriveUploader
from .resolver import Resolved
from .zotero import (ZoteroClient, build_item, folder_name, has_pdf, link_path,
                     pdf_filename)

logger = logging.getLogger(__name__)


@dataclass
class SaveResult:
    title: str
    collection: str
    collection_created: bool
    item_key: str
    existed: bool                 # 라이브러리에 이미 있던 항목인지
    pdf: str                      # "uploaded" | "already" | "unavailable"
    drive_path: str | None = None # 리모트 기준 경로 (uploaded일 때)
    note: str | None = None


def fetch_pdf(cfg: Config, paper, dest: str) -> bool:
    """PDF 후보 URL을 순서대로 시도. 진짜 PDF(%PDF-)를 받으면 True."""
    for url in iter_candidate_urls(paper, cfg.library.unpaywall_email):
        try:
            download_pdf(url, dest, retries=1, timeout=60)
            with open(dest, "rb") as f:
                if f.read(5) == b"%PDF-":
                    return True
            logger.warning("PDF가 아닌 응답(랜딩 페이지 등) — 다음 후보: %s", url)
        except Exception as e:  # noqa: BLE001
            logger.warning("PDF 소스 실패, 다음 후보 시도: %s (%s)", url, e)
    return False


def save_paper(cfg: Config, zot: ZoteroClient, drive: DriveUploader,
               resolved: Resolved, collection_name: str) -> SaveResult:
    paper = resolved.paper
    lib = cfg.library

    coll, created = zot.find_or_create_collection(collection_name)
    coll_key, coll_name = coll["key"], coll["name"]

    data = zot.find_existing(paper)
    existed = data is not None
    if existed:
        zot.add_to_collection(data, coll_key)
        logger.info("기존 항목 재사용 %s → 컬렉션 %s", data["key"], coll_name)
        if has_pdf(zot.children(data["key"])):
            return SaveResult(paper.title, coll_name, created, data["key"], True, "already")
    else:
        data = zot.create_item(build_item(
            paper, pub_types=resolved.pub_types, pub_date=resolved.pub_date,
            journal=resolved.journal, collection_key=coll_key, tag=lib.tag,
        ))
        logger.info("Zotero 항목 생성 %s: %s", data["key"], paper.title)

    folder = folder_name(coll_name)
    filename = pdf_filename(paper, lib.filename_lang)
    rel = f"{folder}/{filename}" if folder else filename
    with tempfile.TemporaryDirectory(prefix="paperbot-lib-") as tmp:
        local = os.path.join(tmp, "paper.pdf")
        if not fetch_pdf(cfg, paper, local):
            return SaveResult(paper.title, coll_name, created, data["key"], existed, "unavailable",
                              note="공개 PDF를 찾지 못해 메타데이터만 저장했습니다.")
        drive.upload(local, rel)
    zot.create_linked_pdf(data["key"], link_path(lib.link_base, folder, filename))
    return SaveResult(paper.title, coll_name, created, data["key"], existed, "uploaded",
                      drive_path=rel)
