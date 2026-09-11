"""sqlite 기반 중복/상태 관리 (v2 — uid 키).

status 값:
  done              요약·포스팅 완료 → 재처리 안 함
  filtered_out      relevance 필터 탈락 → 재채점하지 않도록 기록
  failed            1회 실패 → 다음 실행 때 재시도 대상
  failed_permanent  재시도도 실패 → 영구 skip

should_process()는 (미등록) 또는 (status == 'failed')일 때만 True.
"""
from __future__ import annotations

import logging
import os
import sqlite3
from datetime import datetime, timezone

from .models import Paper

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    uid          TEXT PRIMARY KEY,
    topic        TEXT,
    title        TEXT,
    source       TEXT,
    status       TEXT,
    judge_score  REAL,
    processed_at TEXT,
    error        TEXT
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(_SCHEMA)
        self.conn.commit()

    # --- 조회 ---
    def _status(self, uid: str) -> str | None:
        row = self.conn.execute(
            "SELECT status FROM papers WHERE uid = ?", (uid,)
        ).fetchone()
        return row["status"] if row else None

    def should_process(self, uid: str) -> bool:
        """미등록이거나 직전에 1회 실패(failed)한 논문만 처리 대상."""
        status = self._status(uid)
        return status is None or status == "failed"

    def filter_unprocessed(self, papers: list[Paper]) -> list[Paper]:
        return [p for p in papers if self.should_process(p.uid)]

    # --- 기록 ---
    def _upsert(self, paper: Paper, status: str, error: str | None) -> None:
        self.conn.execute(
            """
            INSERT INTO papers (uid, topic, title, source, status, judge_score, processed_at, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(uid) DO UPDATE SET
                topic        = excluded.topic,
                title        = excluded.title,
                source       = excluded.source,
                status       = excluded.status,
                judge_score  = excluded.judge_score,
                processed_at = excluded.processed_at,
                error        = excluded.error
            """,
            (
                paper.uid, paper.topic_name, paper.title, paper.source,
                status, paper.judge_score, _now(),
                error[:2000] if error else None,
            ),
        )
        self.conn.commit()

    def mark_done(self, paper: Paper) -> None:
        self._upsert(paper, "done", None)

    def mark_filtered(self, paper: Paper, reason: str) -> None:
        """relevance 필터 탈락 기록 — 같은 논문을 매번 재채점하지 않기 위함."""
        self._upsert(paper, "filtered_out", reason)

    def mark_failed(self, paper: Paper, error: str) -> None:
        """첫 실패는 'failed'(재시도 가능), 이미 failed였다면 'failed_permanent'."""
        prev = self._status(paper.uid)
        new_status = "failed_permanent" if prev == "failed" else "failed"
        self._upsert(paper, new_status, error)
        if new_status == "failed_permanent":
            logger.warning("영구 skip 처리: %s (재시도도 실패)", paper.uid)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
