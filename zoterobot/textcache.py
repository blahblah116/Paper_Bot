"""Library text cache (sqlite): documents, keyword search and reading for the chat tools.
라이브러리 본문 텍스트 캐시 — 문서 저장·키워드 검색·발췌 (DM 챗봇 도구용, 임베딩 모델 없음).

문서 = Zotero 최상위 항목(논문) 또는 Drive 폴더에만 있는 PDF(item_type = DRIVE_TYPE).
채우는 쪽은 sync.py(sync_library). 이 모듈은 저장소와 조회만 담당한다.
검색: 키워드별로 제목(가중 6)·초록(3)·본문(log 횟수, 상한 4) 점수 + 키워드 커버리지 보정. 하이픈·대소문자 무시.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

DRIVE_TYPE = "drive_pdf"      # Zotero에 없고 Drive 폴더에만 있는 PDF (item_type 값)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS docs (
    item_key    TEXT PRIMARY KEY,
    title       TEXT,
    authors     TEXT,      -- JSON list
    year        TEXT,
    collections TEXT,      -- JSON list (컬렉션 이름)
    abstract    TEXT,
    url         TEXT,
    item_type   TEXT,
    sig         TEXT,      -- 항목 version + 첨부 key:version — 바뀌면 본문 재추출
    text        TEXT,
    text_source TEXT,      -- zotero_fulltext | pdf | missing(PDF 첨부는 있으나 파일 못 받음 — 재시도) | none(PDF 없음)
    updated_at  TEXT
);
CREATE TABLE IF NOT EXISTS summaries (          -- summarize_paper 결과 (summary.py)
    item_key   TEXT PRIMARY KEY,
    summary    TEXT,
    source     TEXT,       -- 'drive:<경로>' | 'zotero:<첨부 key>'
    chars      INTEGER,    -- 요약기에 넣은 본문 글자 수
    model      TEXT,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""
_SNIPPET = 160               # 검색 스니펫 반경(자)


def norm(s: str) -> str:
    """검색용 정규화: 소문자, PDF 줄끝 하이픈 연결, 하이픈→공백, 공백 압축."""
    s = (s or "").lower().replace("-\n", "")
    s = re.sub(r"[-‐–—_/]", " ", s)
    return " ".join(s.split())


def short_authors(authors: list[str]) -> str:
    if not authors:
        return "저자 미상"
    return authors[0] + (" 외" if len(authors) > 1 else "")


@dataclass
class Doc:
    key: str
    title: str
    authors: list[str]
    year: str
    collections: list[str]
    abstract: str
    url: str
    item_type: str
    text: str
    text_source: str
    # 검색용 정규화 캐시
    _t: str = field(default="", repr=False)
    _a: str = field(default="", repr=False)
    _b: str = field(default="", repr=False)

    def __post_init__(self):
        self._t, self._a, self._b = norm(self.title), norm(self.abstract), norm(self.text)

    def line(self) -> str:
        """카탈로그 한 줄: id | 제목 | 첫 저자 연도 | 컬렉션 — 시스템 프롬프트·목록 도구가 쓴다."""
        colls = ", ".join(self.collections) or "-"
        who = f"{short_authors(self.authors)} {self.year}".strip()
        mark = " [Drive 전용]" if self.item_type == DRIVE_TYPE else ""
        return f"{self.key} | {self.title} | {who} | {colls}{mark}"


class TextCache:
    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()
        self.lock = threading.RLock()        # 동기화(sync_library)가 잡는다
        self._docs: list[Doc] | None = None
        self._last_check: datetime | None = None

    # --- meta / 동기화 시점 ---
    def get_meta(self, key: str, default: str = "") -> str:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute("INSERT INTO meta(key, value) VALUES(?, ?) "
                          "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))
        self.conn.commit()

    # 마지막 확인 시각은 메모리에만 둔다 — DM 스레드가 동기화 중인 DB 연결을 건드리지 않게 (재시작 직후엔 바로 확인)
    def touch_checked(self) -> None:
        self._last_check = datetime.now(timezone.utc)

    def mark_stale(self) -> None:
        """다음 check_due가 바로 True가 되게 — 스레드 저장 직후 새 논문이 바로 반영되도록."""
        self._last_check = None

    def check_due(self, interval_min: int) -> bool:
        last = self._last_check
        if last is None:
            return True
        return (datetime.now(timezone.utc) - last).total_seconds() >= interval_min * 60

    # --- 쓰기 (sync.py 전용) ---
    def upsert(self, key: str, meta: dict, sig: str, text: str, source: str) -> None:
        self.conn.execute(
            "INSERT INTO docs(item_key, title, authors, year, collections, abstract, url, item_type, "
            "sig, text, text_source, updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(item_key) DO UPDATE SET title=excluded.title, authors=excluded.authors, "
            "year=excluded.year, collections=excluded.collections, abstract=excluded.abstract, "
            "url=excluded.url, item_type=excluded.item_type, sig=excluded.sig, text=excluded.text, "
            "text_source=excluded.text_source, updated_at=excluded.updated_at",
            (key, meta["title"], json.dumps(meta["authors"], ensure_ascii=False), meta["year"],
             json.dumps(meta["collections"], ensure_ascii=False), meta["abstract"], meta["url"],
             meta["item_type"], sig, text, source, datetime.now(timezone.utc).isoformat()))
        self.conn.commit()

    def update_collections(self, key: str, collections: list[str]) -> None:
        self.conn.execute("UPDATE docs SET collections=? WHERE item_key=?",
                          (json.dumps(collections, ensure_ascii=False), key))

    def delete(self, keys) -> None:
        for k in keys:
            self.conn.execute("DELETE FROM docs WHERE item_key=?", (k,))
            self.conn.execute("DELETE FROM summaries WHERE item_key=?", (k,))
        self.conn.commit()

    # --- 전문 요약 (summary.py가 쓴다) ---
    def get_summary(self, key: str) -> dict | None:
        r = self.conn.execute("SELECT summary, source, chars, model, created_at FROM summaries WHERE item_key=?",
                              (key.strip().upper(),)).fetchone()
        return dict(r) if r else None

    def set_summary(self, key: str, summary: str, source: str, chars: int, model: str = "") -> None:
        self.conn.execute(
            "INSERT INTO summaries(item_key, summary, source, chars, model, created_at) VALUES(?,?,?,?,?,?) "
            "ON CONFLICT(item_key) DO UPDATE SET summary=excluded.summary, source=excluded.source, "
            "chars=excluded.chars, model=excluded.model, created_at=excluded.created_at",
            (key.strip().upper(), summary, source, int(chars), model,
             datetime.now(timezone.utc).isoformat(timespec="seconds")))
        self.conn.commit()

    def known_zotero(self) -> dict[str, tuple[str, str, str]]:
        """Zotero 문서 key → (sig, text_source, updated_at)."""
        return {r["item_key"]: (r["sig"], r["text_source"], r["updated_at"]) for r in self.conn.execute(
            "SELECT item_key, sig, text_source, updated_at FROM docs WHERE item_type != ?", (DRIVE_TYPE,))}

    def known_drive(self) -> dict[str, str]:
        """Drive 전용 문서 key → sig."""
        return {r["item_key"]: r["sig"] for r in self.conn.execute(
            "SELECT item_key, sig FROM docs WHERE item_type=?", (DRIVE_TYPE,))}

    def missing_retry_due(self, cutoff_iso: str) -> int:
        """본문을 못 받은(missing) Zotero 문서 중 cutoff 이전에 시도한 것의 수."""
        return self.conn.execute(
            "SELECT COUNT(*) FROM docs WHERE text_source='missing' AND updated_at < ? AND item_type != ?",
            (cutoff_iso, DRIVE_TYPE)).fetchone()[0]

    def invalidate(self) -> None:
        self._docs = None

    # --- 조회 ---
    def docs(self) -> list[Doc]:
        cached = self._docs
        if cached is not None:   # 백그라운드 동기화가 락을 오래 잡아도 DM 검색은 막히지 않게
            return cached
        with self.lock:
            if self._docs is None:
                rows = self.conn.execute("SELECT * FROM docs ORDER BY title").fetchall()
                self._docs = [Doc(key=r["item_key"], title=r["title"], authors=json.loads(r["authors"] or "[]"),
                                  year=r["year"] or "", collections=json.loads(r["collections"] or "[]"),
                                  abstract=r["abstract"] or "", url=r["url"] or "", item_type=r["item_type"] or "",
                                  text=r["text"] or "", text_source=r["text_source"] or "none")
                              for r in rows]
            return self._docs

    def get(self, key: str) -> Doc | None:
        key = key.strip().upper()
        return next((d for d in self.docs() if d.key == key), None)

    def find_by_title(self, ref: str) -> Doc | None:
        """id 또는 제목(일부)로 문서 찾기 — 모델이 id 대신 제목을 넣었을 때. 제목 단어의 60% 이상이 맞는 최선 후보."""
        ref = (ref or "").strip()
        if not ref:
            return None
        d = self.get(ref)
        if d:
            return d
        want = norm(ref)
        exact = [d for d in self.docs() if want and want in d._t]
        if exact:
            return min(exact, key=lambda d: len(d._t))
        words = [w for w in re.split(r"[^a-z0-9]+", want) if len(w) >= 3]
        if not words:
            return None
        best, best_frac = None, 0.0
        for d in self.docs():
            frac = sum(1 for w in words if w in d._t) / len(words)
            if frac > best_frac or (frac == best_frac and best and len(d._t) < len(best._t)):
                best, best_frac = d, frac
        if best_frac >= 0.6:
            return best
        # 'GOAT: Global-Local …'처럼 부제를 틀리게 적어도 콜론 앞 약칭(4자 이상)이 제목에 유일하게 들어 있으면 그 논문
        short = norm(ref.split(":")[0]) if ":" in ref else ""
        if len(short) >= 4:
            cands = [d for d in self.docs() if re.search(r"(?<![a-z0-9])" + re.escape(short) + r"(?![a-z0-9])", d._t)]
            if len(cands) == 1:
                return cands[0]
        return None

    def _filter(self, collection: str | None) -> list[Doc]:
        docs = self.docs()
        if collection:
            want = norm(collection)
            docs = [d for d in docs if any(want in norm(c) for c in d.collections)]
        return docs

    def catalog(self) -> list[str]:
        return [d.line() for d in self.docs()]

    def list_papers(self, collection: str | None = None) -> str:
        docs = self._filter(collection)
        if not docs:
            names = sorted({c for d in self.docs() for c in d.collections}, key=str.lower)
            return f"'{collection}' 컬렉션에 논문이 없습니다. 있는 컬렉션: {', '.join(names)}"
        return f"{len(docs)}편 (id | 제목 | 첫 저자 연도 | 컬렉션)\n" + "\n".join(d.line() for d in docs)

    def search(self, keywords: list[str], collection: str | None = None, limit: int = 8) -> list[dict]:
        kws = [norm(k) for k in keywords if norm(k)][:8]
        if not kws:
            return []
        scored = []
        for d in self._filter(collection):
            score, matched = 0.0, {}
            for kw in kws:
                t, a, b = d._t.count(kw), d._a.count(kw), d._b.count(kw)
                if t or a or b:
                    matched[kw] = t + a + b
                score += 6 * min(t, 1) + 3 * min(a, 1) + min(math.log1p(b), 4.0)
            if matched:
                score *= 0.5 + 0.5 * len(matched) / len(kws)
                scored.append((score, d, matched))
        scored.sort(key=lambda x: -x[0])
        out = []
        for score, d, matched in scored[:limit]:
            rare = min(matched, key=lambda k: matched[k])
            out.append({"id": d.key, "title": d.title, "authors": short_authors(d.authors), "year": d.year,
                        "collections": d.collections, "score": round(score, 2), "matches": matched,
                        "snippet": self._snippet(d, rare)})
        return out

    @staticmethod
    def _snippet(d: Doc, kw: str) -> str:
        for src in (d.abstract, d.text):
            hay = norm(src)
            i = hay.find(kw)
            if i != -1:
                s = hay[max(0, i - _SNIPPET): i + len(kw) + _SNIPPET]
                return ("…" if i > _SNIPPET else "") + s + "…"
        return ""

    def read(self, key: str, focus: str | None = None, max_chars: int = 40000) -> str:
        """논문 하나의 메타데이터·초록·본문(focus 키워드 주변 발췌 또는 앞부분+끝부분, 본문 max_chars자까지).

        max_chars는 호출자(agent)가 config read_max_chars와 질문당 남은 총량으로 정한다.
        """
        d = self.get(key)
        if d is None:
            return f"id '{key}'인 논문이 없습니다. search_library나 list_papers로 id를 확인하세요."
        head = (f"제목: {d.title}\n저자: {', '.join(d.authors[:8])}{' 외' if len(d.authors) > 8 else ''}\n"
                f"연도: {d.year or '-'} · 컬렉션: {', '.join(d.collections) or '-'} · URL: {d.url or '-'}\n"
                f"초록: {d.abstract or '(없음)'}\n\n")
        s = self.get_summary(d.key)
        if s:   # summarize_paper가 만든 전문 요약 — 후속 질문에 바로 쓰이게 본문 앞에
            head += f"[전문 요약 — {(s['created_at'] or '')[:10]}, 본문 {s['chars']:,}자 기준]\n{s['summary']}\n\n"
        if not d.text:
            return head + "(본문 텍스트 없음 — 제목·초록만 있습니다)"
        text = d.text
        if focus:
            terms = [norm(t) for t in re.split(r"[,;]", focus) if norm(t)] or [norm(focus)]
            hay = norm(text)
            spans = []
            for t in terms:
                for m in re.finditer(re.escape(t), hay):
                    spans.append((max(0, m.start() - 600), m.end() + 600))
            spans.sort()
            merged: list[list[int]] = []
            for s, e in spans:
                if merged and s <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], e)
                else:
                    merged.append([s, e])
            parts, total = [], 0
            for s, e in merged:
                if total >= max_chars:
                    break
                parts.append(hay[s:e][: max_chars - total])
                total += e - s
            if parts:
                return head + f"[본문 중 '{focus}' 관련 발췌 {len(parts)}곳]\n" + "\n…\n".join(parts)
            head += f"('{focus}'는 본문에서 찾지 못해 앞부분을 보여줍니다)\n"
        tail = min(2000, max_chars // 4) if len(text) > max_chars else 0   # 끝부분은 상한의 1/4 이하
        if tail:
            omitted = len(text) - max_chars
            body = text[: max_chars - tail] + f"\n…(중략 {omitted:,}자 — 필요하면 focus 키워드로 발췌)…\n{text[-tail:]}"
            return head + f"[본문 — 전체 {len(text):,}자 중 앞 {max_chars - tail:,}자 + 끝 {tail:,}자]\n" + body
        return head + "[본문]\n" + text
