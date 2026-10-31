"""sqlite 기반 처리 이력 + 백로그 (uid = 정규화한 글 URL).

status 값:
  pending           judge 통과, 아직 발행 안 됨 → 백로그. payload에 Article 전체(JSON) 보관
  done              발행 완료 (summary 컬럼에 요약 보관)
  filtered_out      judge 탈락 (threshold 미만) → 재채점 안 함
  failed            처리 1회 실패 → 백로그에 남아 다음 실행에 재시도
  failed_permanent  재시도도 실패 → 영구 skip
  expired           lookback_days보다 오래된 백로그 → 더 이상 발행하지 않음
  seen              --mark-seen으로 발행 없이 '본 것' 처리

발행 순서: 기관(org)별로 pending·failed 중 게시일 최신순 (게시일 없으면 수집 시각).
새 글이 있으면 가장 최신이므로 먼저 나가고, 없으면 백로그의 다음 최신 글이 나간다.

dry-run(dry=True)은 DB 파일을 메모리로 복사해 그 위에서만 동작한다 — 로직은 같고 파일은 그대로.
"""
from __future__ import annotations

import logging
import os
import sqlite3
from datetime import datetime, timezone

from .models import Article

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    uid           TEXT PRIMARY KEY,
    source        TEXT,
    org           TEXT,
    title         TEXT,
    url           TEXT,
    published     TEXT,          -- ISO8601 UTC (없으면 NULL)
    status        TEXT,
    judge_score   REAL,
    judge_reason  TEXT,
    discovered_at TEXT,          -- 처음 수집한 시각
    processed_at  TEXT,          -- 마지막 상태 변경 시각
    error         TEXT,
    payload       TEXT,          -- Article JSON (백로그 발행용)
    summary       TEXT           -- 발행한 요약 (done)
);
CREATE INDEX IF NOT EXISTS idx_articles_org_status ON articles(org, status);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,      -- 예: last_ingest_date (수집 주기 판단용, 로컬 날짜 YYYY-MM-DD)
    value TEXT
);
"""

BACKLOG_STATUSES = ("pending", "failed")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, db_path: str, dry: bool = False):
        self.dry = dry
        if dry:
            self.conn = sqlite3.connect(":memory:")
            if os.path.exists(db_path):
                src = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
                src.backup(self.conn)
                src.close()
        else:
            os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
            self.conn = sqlite3.connect(db_path, timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)

    # ------------------------------------------------------------ 조회

    def get(self, uid: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM articles WHERE uid = ?", (uid,)).fetchone()

    def status(self, uid: str) -> str | None:
        row = self.get(uid)
        return row["status"] if row else None

    def backlog(self, sources: list[str], limit: int | None = None) -> list[Article]:
        """sources의 백로그(pending·failed)를 게시일 최신순으로."""
        if not sources:
            return []
        marks = ",".join("?" * len(sources))
        sql = (f"SELECT uid, payload, status, judge_score, judge_reason FROM articles "
               f"WHERE status IN ('pending','failed') AND source IN ({marks}) "
               f"ORDER BY COALESCE(published, discovered_at) DESC")
        if limit:
            sql += f" LIMIT {int(limit)}"
        out: list[Article] = []
        for r in self.conn.execute(sql, sources):
            if not r["payload"]:
                logger.warning("백로그에 payload 없음 — 건너뜀: %s", r["uid"])
                continue
            try:
                a = Article.from_payload(r["payload"])
            except (TypeError, ValueError, KeyError) as e:
                logger.warning("백로그 payload 복원 실패 — 건너뜀: %s (%s)", r["uid"], e)
                continue
            a.judge_score, a.judge_reason = r["judge_score"], r["judge_reason"]
            out.append(a)
        return out

    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute("INSERT INTO meta (key, value) VALUES (?, ?) "
                          "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))
        self.conn.commit()

    def counts(self) -> dict[tuple[str, str], int]:
        rows = self.conn.execute("SELECT org, status, COUNT(*) n FROM articles GROUP BY org, status")
        return {(r["org"], r["status"]): r["n"] for r in rows}

    # ------------------------------------------------------------ 기록

    def _upsert(self, a: Article, status: str, *, error: str | None = None, summary: str | None = None,
                keep_payload: bool = True) -> None:
        payload = a.to_payload() if keep_payload else None
        self.conn.execute(
            """
            INSERT INTO articles (uid, source, org, title, url, published, status, judge_score, judge_reason,
                                  discovered_at, processed_at, error, payload, summary)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(uid) DO UPDATE SET
                source=excluded.source, org=excluded.org, title=excluded.title, url=excluded.url,
                published=excluded.published, status=excluded.status,
                judge_score=excluded.judge_score, judge_reason=excluded.judge_reason,
                processed_at=excluded.processed_at, error=excluded.error,
                payload=excluded.payload, summary=COALESCE(excluded.summary, articles.summary)
            """,
            (a.uid, a.source, a.org, a.title, a.url, a.published.isoformat() if a.published else None,
             status, a.judge_score, a.judge_reason, _now(), _now(),
             error[:2000] if error else None, payload, summary),
        )
        self.conn.commit()

    def mark_pending(self, a: Article) -> None:
        self._upsert(a, "pending")

    def mark_done(self, a: Article, summary: str) -> None:
        self._upsert(a, "done", summary=summary, keep_payload=False)

    def mark_filtered(self, a: Article) -> None:
        self._upsert(a, "filtered_out", keep_payload=False)

    def mark_seen(self, a: Article) -> None:
        self._upsert(a, "seen", keep_payload=False)

    def mark_failed(self, a: Article, error: str) -> None:
        """첫 실패는 failed(백로그에 남아 재시도), 이미 failed였다면 failed_permanent."""
        new = "failed_permanent" if self.status(a.uid) == "failed" else "failed"
        self._upsert(a, new, error=error, keep_payload=(new == "failed"))
        if new == "failed_permanent":
            logger.warning("영구 skip 처리: %s (재시도도 실패)", a.uid)

    def expire_older_than(self, cutoff: datetime) -> int:
        """게시일이 cutoff보다 오래된 백로그를 expired로. 반환: 처리 수."""
        cur = self.conn.execute(
            "UPDATE articles SET status='expired', payload=NULL, processed_at=? "
            "WHERE status IN ('pending','failed') AND published IS NOT NULL AND published < ?",
            (_now(), cutoff.isoformat()),
        )
        self.conn.commit()
        return cur.rowcount

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
