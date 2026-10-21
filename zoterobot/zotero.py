"""Zotero Web API v3 클라이언트 (개인 라이브러리) + 항목 JSON 빌더.

- 인증: 환경변수 ZOTERO_API_KEY(쓰기 권한), ZOTERO_USER_ID(숫자).
- PDF는 Zotero Storage에 올리지 않는다 — Drive에 올린 파일을 가리키는 linked_file 첨부만 만든다.
  path가 'attachments:GDL/x.pdf'면 데스크톱 Zotero가 '링크 첨부 파일 기본 디렉토리' 기준으로 풀어서 연다
  (Zotero 소스의 BASE_PATH_PLACEHOLDER, '/' 구분자로 기기 간 동기화).
- 링크 파일은 그룹 라이브러리·웹·모바일에서 열리지 않으므로 개인 라이브러리 전용.
- 컬렉션은 'ICML/2026'처럼 '/'로 하위 컬렉션을 가리킨다 (Zotero: ICML > 2026, Drive 폴더: ICML/2026/).
  Attanger의 {{collection}}이 부모 이름을 포함한 전체 경로를 폴더로 쓰는 것과 같은 규칙.
"""
from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

import requests

from paperbot.models import Paper

logger = logging.getLogger(__name__)

_API = "https://api.zotero.org"
_MAX_RETRIES = 4
_TITLE_MAX = 100                                   # Zotero 7 기본 파일명 형식의 title truncate="100"
_INVALID_FILENAME_RE = re.compile(r'[\\/?*:|"<>\x00-\x1f]')   # Zotero.File.getValidFileName과 같은 문자 집합
_NON_VENUES = {"arxiv", "arxiv.org", "biorxiv", "medrxiv", "ssrn"}


class ZoteroError(Exception):
    pass


@dataclass
class Collection:
    """find_or_create_collection 결과 — key는 경로의 가장 안쪽(마지막) 컬렉션."""
    key: str
    path: str                                   # 실제 저장된 이름 기준 전체 경로, 예: 'ICML/2026'
    created: list[str] = field(default_factory=list)   # 이번에 새로 만든 단계 이름들


# ---------------------------------------------------------------- 순수 함수 (테스트 대상)

def _split_name(name: str) -> dict:
    parts = name.strip().split()
    if len(parts) < 2:
        return {"creatorType": "author", "name": name.strip()}
    return {"creatorType": "author", "firstName": " ".join(parts[:-1]), "lastName": parts[-1]}


def _last_name(name: str) -> str:
    parts = name.strip().split()
    return parts[-1] if parts else ""


# Zotero UI 언어별 firstCreator 접속어 (and, et al.) — 한국어 Zotero는 'A 및 B', 'A 등'
_CREATOR_WORDS = {"en": ("and", "et al."), "ko": ("및", "등")}


def first_creator(authors: list[str], lang: str = "en") -> str:
    """Zotero firstCreator 규칙: 1명 'A', 2명 'A and B', 3명+ 'A et al.' (lang에 따라 접속어)"""
    and_word, etal = _CREATOR_WORDS.get(lang, _CREATOR_WORDS["en"])
    names = [_last_name(a) for a in authors if a.strip()]
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} {and_word} {names[1]}"
    return f"{names[0]} {etal}"


def sanitize_filename(name: str) -> str:
    name = _INVALID_FILENAME_RE.sub("", name)
    name = " ".join(name.split()).strip().lstrip(".")
    return name


def pdf_filename(paper: Paper, lang: str = "en") -> str:
    """Zotero 7 기본 파일명 형식과 같게: '{firstCreator} - {year} - {title[:100]}.pdf'.

    Zotero처럼 제목을 100자로 먼저 자른 뒤 금지문자를 지운다 (실측: 'UrbanFusion: …' → 99자).
    """
    parts = [first_creator(paper.authors, lang), str(paper.year or ""), paper.title[:_TITLE_MAX]]
    stem = sanitize_filename(" - ".join(p for p in parts if p.strip()))
    return f"{stem or paper.uid.replace(':', '_')}.pdf"


def collection_path(name: str) -> list[str]:
    """'ICML/2026' → ['ICML', '2026']. '/'가 하위 컬렉션 구분자, 빈 단계는 무시."""
    return [s.strip() for s in name.split("/") if s.strip()]


def collection_paths(cols: list[dict]) -> dict[str, str]:
    """컬렉션 key → 부모 이름을 포함한 전체 경로 ('ICML/2026'). 사용법 목록·본문 캐시 표시용."""
    by_key = {c["key"]: c for c in cols}

    def path(key: str) -> str:
        c = by_key[key]
        parent = c.get("parentCollection") or None
        return (path(parent) + "/" if parent in by_key else "") + c["name"].strip()

    return {k: path(k) for k in by_key}


def folder_name(collection_name: str) -> str:
    """컬렉션 경로('ICML/2026') → Drive 하위 폴더 경로. Attanger처럼 '/'는 하위 폴더 구분자로 남긴다."""
    segs = [sanitize_filename(s) for s in collection_name.split("/")]
    return "/".join(s for s in segs if s)


def link_path(link_base: str, folder: str, filename: str) -> str:
    """linked_file attachment의 path. link_base='attachments:'면 기본 디렉토리 기준 상대경로."""
    rel = f"{folder}/{filename}" if folder else filename
    if link_base.endswith((":", "/", "\\")):
        return link_base + rel
    return f"{link_base}/{rel}"


def build_item(paper: Paper, *, pub_types: list[str], pub_date: str | None,
               journal: str | None, collection_key: str | None, tag: str) -> dict:
    """Paper → Zotero 항목 JSON.

    유형: S2가 Conference라 하면 conferencePaper, 실제 journal 이름이 있으면 journalArticle,
    venue만 있으면 conferencePaper, venue가 없는 arXiv 논문은 preprint.
    S2는 NeurIPS 같은 학회 논문에도 JournalArticle + journal "ArXiv"를 주므로
    publicationTypes보다 journal 이름 유무를 믿는다 (실측: arXiv 2505.13631).
    """
    venue = paper.venue if paper.venue and paper.venue.strip().lower() not in _NON_VENUES else None
    journal = journal if journal and journal.strip().lower() not in _NON_VENUES else None
    types = {t.lower() for t in pub_types}
    item: dict = {
        "title": paper.title,
        "creators": [_split_name(a) for a in paper.authors if a.strip()],
        "abstractNote": paper.abstract or "",
        "date": pub_date or (str(paper.year) if paper.year else ""),
        "url": paper.url,
        "accessDate": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tags": [{"tag": tag}] if tag else [],
        "collections": [collection_key] if collection_key else [],
    }
    extra = []
    if venue and "conference" in types:
        item.update(itemType="conferencePaper", proceedingsTitle=venue, conferenceName=venue)
    elif journal:
        item.update(itemType="journalArticle", publicationTitle=journal)
    elif venue and "journalarticle" in types and not paper.arxiv_id:
        item.update(itemType="journalArticle", publicationTitle=venue)
    elif venue:
        item.update(itemType="conferencePaper", proceedingsTitle=venue, conferenceName=venue)
    elif paper.arxiv_id:
        item.update(itemType="preprint", repository="arXiv", archiveID=f"arXiv:{paper.arxiv_id}")
    else:
        item.update(itemType="journalArticle", publicationTitle=journal or "")

    if item["itemType"] == "preprint":
        item["DOI"] = paper.doi or f"10.48550/arXiv.{paper.arxiv_id}"
    else:
        if paper.doi:
            item["DOI"] = paper.doi
        if paper.arxiv_id:
            extra.append(f"arXiv: {paper.arxiv_id}")
    if extra:
        item["extra"] = "\n".join(extra)
    return item


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (t or "").lower())


def item_matches(data: dict, paper: Paper) -> bool:
    """라이브러리의 기존 항목이 같은 논문인지: DOI → arXiv id → 정규화 제목 순으로 비교."""
    doi = (data.get("DOI") or "").strip().lower()
    if paper.doi and doi and doi == paper.doi.lower():
        return True
    if paper.arxiv_id:
        blob = " ".join(str(data.get(k) or "") for k in ("archiveID", "url", "extra", "DOI")).lower()
        if paper.arxiv_id.lower() in blob:
            return True
    return bool(_norm_title(data.get("title")) and _norm_title(data.get("title")) == _norm_title(paper.title))


def _first_success(resp_json: dict, what: str) -> dict:
    """POST 응답(successful/failed 맵)에서 첫 항목을 꺼낸다."""
    ok = resp_json.get("successful") or {}
    if "0" in ok:
        return ok["0"]
    failed = (resp_json.get("failed") or {}).get("0") or resp_json
    raise ZoteroError(f"{what} 생성 실패: {failed}")


# ---------------------------------------------------------------- API 클라이언트

class ZoteroClient:
    def __init__(self, api_key: str, user_id: str = ""):
        if not api_key:
            raise ZoteroError("ZOTERO_API_KEY 환경변수가 필요합니다.")
        self.session = requests.Session()
        self.session.headers.update({"Zotero-API-Key": api_key, "Zotero-API-Version": "3"})
        if not user_id.isdigit():
            # userID 자리에 로그인 아이디를 넣는 실수가 흔하다 — 키 소유자의 숫자 userID를 API로 조회.
            user_id = self._user_id_from_key(user_id)
        self.prefix = f"{_API}/users/{user_id}"

    def _user_id_from_key(self, given: str) -> str:
        try:
            resp = self.session.get(f"{_API}/keys/current", timeout=30)
        except requests.RequestException as e:
            raise ZoteroError(f"Zotero 키 조회 실패: {e}") from e
        if resp.status_code != 200:
            raise ZoteroError(f"Zotero API 키가 거부되었습니다 (HTTP {resp.status_code}).")
        uid = str(resp.json().get("userID", ""))
        if given:
            logger.warning("ZOTERO_USER_ID=%r는 숫자 userID가 아님 — API 키 기준 userID %s 사용", given, uid)
        return uid

    def _req(self, method: str, path: str, **kw) -> requests.Response:
        headers = kw.pop("headers", {})
        if method in ("POST", "PATCH", "PUT") and "Zotero-Write-Token" not in headers \
                and "If-Unmodified-Since-Version" not in headers:
            headers["Zotero-Write-Token"] = uuid.uuid4().hex   # 재시도해도 중복 생성되지 않게
        last = None
        for attempt in range(1, _MAX_RETRIES + 1):
            resp = self.session.request(method, self.prefix + path, headers=headers, timeout=30, **kw)
            backoff = resp.headers.get("Backoff")
            if resp.status_code in (429, 503):
                wait = float(resp.headers.get("Retry-After") or backoff or 2 ** attempt)
                logger.warning("Zotero %s %s → HTTP %d, %.0fs 대기", method, path, resp.status_code, wait)
                time.sleep(min(wait, 60))
                last = resp
                continue
            if backoff:
                time.sleep(min(float(backoff), 60))
            if resp.status_code == 403:
                raise ZoteroError("Zotero API 키가 거부되었습니다 (403) — 키의 쓰기 권한·userID를 확인하세요.")
            if resp.status_code >= 400:
                raise ZoteroError(f"Zotero {method} {path} → HTTP {resp.status_code}: {resp.text[:300]}")
            return resp
        raise ZoteroError(f"Zotero {method} {path} 재시도 초과 (HTTP {last.status_code if last else '?'})")

    # --- 컬렉션 ---
    def collections(self) -> list[dict]:
        out, start = [], 0
        while True:
            resp = self._req("GET", "/collections", params={"limit": 100, "start": start})
            page = resp.json()
            out.extend(c["data"] for c in page)
            start += len(page)
            if not page or start >= int(resp.headers.get("Total-Results", start)):
                return out

    def find_or_create_collection(self, name: str) -> Collection:
        """'ICML/2026' 경로를 단계별로 찾고(대소문자 무시, 같은 부모 아래에서만), 없는 단계만 새로 만든다.

        이름만 적으면 최상위 컬렉션을 뜻한다 — 'dataset'은 'GraphTransformer > dataset'과 매칭되지 않고
        최상위 'dataset'을 만든다. 하위 컬렉션은 반드시 'GraphTransformer/dataset'처럼 경로로 적는다.
        """
        segs = collection_path(name)
        if not segs:
            raise ZoteroError("컬렉션 이름이 비어 있습니다.")
        cols = self.collections()
        parent: str | None = None
        names, created = [], []
        for seg in segs:
            want = seg.lower()
            match = next((c for c in cols if c["name"].strip().lower() == want
                          and (c.get("parentCollection") or None) == parent), None)
            if match is None:
                resp = self._req("POST", "/collections",
                                 json=[{"name": seg, "parentCollection": parent or False}])
                match = _first_success(resp.json(), "컬렉션")["data"]
                cols.append(match)
                created.append(seg)
                logger.info("Zotero 컬렉션 생성: %s (%s, parent=%s)", "/".join(names + [seg]), match["key"], parent)
            parent = match["key"]
            names.append(match["name"].strip())
        return Collection(parent, "/".join(names), created)

    # --- 항목 ---
    def find_existing(self, paper: Paper) -> dict | None:
        resp = self._req("GET", "/items/top", params={
            "q": paper.title[:200], "qmode": "titleCreatorYear", "limit": 25,
        })
        for it in resp.json():
            if item_matches(it["data"], paper):
                return it["data"]
        return None

    def create_item(self, item: dict) -> dict:
        resp = self._req("POST", "/items", json=[item])
        return _first_success(resp.json(), "항목")["data"]

    def add_to_collection(self, data: dict, collection_key: str) -> None:
        cols = list(data.get("collections") or [])
        if collection_key in cols:
            return
        self._req("PATCH", f"/items/{data['key']}", json={"collections": cols + [collection_key]},
                  headers={"If-Unmodified-Since-Version": str(data["version"])})

    # --- 라이브러리 전체 조회 (DM 챗봇 본문 캐시용) ---
    def _get_all(self, path: str, params: dict) -> tuple[list[dict], int]:
        """페이지를 끝까지 넘겨 data 목록과 라이브러리 버전(Last-Modified-Version)을 돌려준다."""
        out, start, version = [], 0, 0
        while True:
            resp = self._req("GET", path, params={**params, "limit": 100, "start": start})
            version = int(resp.headers.get("Last-Modified-Version", version) or 0)
            page = resp.json()
            out.extend(x["data"] for x in page)
            start += len(page)
            if not page or start >= int(resp.headers.get("Total-Results", start)):
                return out, version

    def library_version(self, since: int | None = None) -> int | None:
        """현재 라이브러리 버전. since와 같으면(변경 없음) None — 요청 1건으로 끝나는 변경 감지."""
        headers = {"If-Modified-Since-Version": str(since)} if since else {}
        resp = self._req("GET", "/items/top", params={"limit": 1, "format": "keys"}, headers=headers)
        if resp.status_code == 304:
            return None
        return int(resp.headers.get("Last-Modified-Version", 0) or 0)

    def top_items(self) -> tuple[list[dict], int]:
        items, ver = self._get_all("/items/top", {})
        return [d for d in items if d.get("itemType") not in ("note", "attachment", "annotation")], ver

    def attachments(self) -> list[dict]:
        return self._get_all("/items", {"itemType": "attachment"})[0]

    def fulltext(self, att_key: str) -> str | None:
        """데스크톱 Zotero가 추출해 동기화한 본문. 없으면 None (Drive 링크 파일은 대개 없음)."""
        resp = self.session.get(f"{self.prefix}/items/{att_key}/fulltext", timeout=30)
        if resp.status_code != 200:
            return None
        return resp.json().get("content") or None

    def download_file(self, att_key: str, dest: str) -> None:
        """Zotero 저장소 파일 다운로드. S3 리다이렉트에는 API 키 헤더를 넘기지 않는다."""
        resp = self.session.get(f"{self.prefix}/items/{att_key}/file", timeout=60, allow_redirects=False)
        if resp.status_code in (301, 302, 303, 307):
            resp = requests.get(resp.headers["Location"], timeout=120)
        if resp.status_code != 200:
            raise ZoteroError(f"Zotero 파일 다운로드 실패 {att_key}: HTTP {resp.status_code}")
        with open(dest, "wb") as f:
            f.write(resp.content)

    def linked_file_paths(self, limit: int = 100) -> list[str]:
        """최근 첨부 중 linked_file의 path들 — Attanger 경로 형식(상대/절대) 점검용."""
        resp = self._req("GET", "/items", params={"itemType": "attachment", "limit": limit,
                                                  "sort": "dateAdded", "direction": "desc"})
        return [it["data"].get("path", "") for it in resp.json()
                if it["data"].get("linkMode") == "linked_file"]

    def children(self, key: str) -> list[dict]:
        return [c["data"] for c in self._req("GET", f"/items/{key}/children").json()]

    def create_linked_pdf(self, parent_key: str, path: str, title: str = "PDF") -> dict:
        return self.create_item({
            "itemType": "attachment", "linkMode": "linked_file", "parentItem": parent_key,
            "title": title, "contentType": "application/pdf", "path": path,
            "tags": [], "relations": {},
        })


def has_pdf(children: list[dict]) -> bool:
    for c in children:
        if c.get("itemType") != "attachment":
            continue
        if c.get("contentType") == "application/pdf" or str(c.get("path") or c.get("filename") or "").lower().endswith(".pdf"):
            return True
    return False
