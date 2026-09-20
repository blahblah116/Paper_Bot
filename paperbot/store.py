"""sqlite 기반 중복/상태 관리 (v2 — uid 키).

s2_cursor 테이블: S2 인용순 topic의 페이지 커서(bulk 검색 continuation token).
  한 페이지의 항목을 전부 스캔(기존 논문 스킵)한 뒤에만 다음 토큰으로 전진하므로,
  다음 실행은 이미 소진한 페이지를 다시 요청하지 않는다. 쿼리 파라미터가 바뀌면 리셋.

status 값:
  done              요약·포스팅 완료 → 재처리 안 함
  filtered_out      relevance 필터 탈락 → 재채점하지 않도록 기록
  pending           필터는 통과했지만 발행 쿼터에 밀림 → 대기열(FIFO). payload에 Paper 전체 보존.
  expired           pending이 pending_max_age_days를 넘겨 폐기됨
  failed            1회 실패 → 다음 실행 때 재시도 대상
  failed_permanent  재시도도 실패 → 영구 skip

should_process()는 (미등록) 또는 (status == 'failed')일 때만 True.
pending은 검색 결과에 다시 나타나도 재채점하지 않고, 대기열 경로(load_pending)로만 처리된다.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta, timezone

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
    error        TEXT,
    payload      TEXT
);
CREATE TABLE IF NOT EXISTS s2_cursor (
    topic        TEXT PRIMARY KEY,
    query_key    TEXT,
    token        TEXT,
    pages_done   INTEGER NOT NULL DEFAULT 0,
    updated_at   TEXT
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self._migrate()
        self.conn.commit()

    def _migrate(self) -> None:
        """기존 DB(payload 컬럼 없음)에 컬럼 추가 — 데이터 손실 없는 additive 마이그레이션."""
        cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(papers)")}
        if "payload" not in cols:
            self.conn.execute("ALTER TABLE papers ADD COLUMN payload TEXT")
            logger.info("DB 마이그레이션: papers.payload 컬럼 추가 (pending 대기열용)")

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
    def _upsert(self, paper: Paper, status: str, error: str | None,
                payload: str | None = None) -> None:
        self.conn.execute(
            """
            INSERT INTO papers (uid, topic, title, source, status, judge_score, processed_at, error, payload)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(uid) DO UPDATE SET
                topic        = excluded.topic,
                title        = excluded.title,
                source       = excluded.source,
                status       = excluded.status,
                judge_score  = excluded.judge_score,
                processed_at = excluded.processed_at,
                error        = excluded.error,
                payload      = excluded.payload
            """,
            (
                paper.uid, paper.topic_name, paper.title, paper.source,
                status, paper.judge_score, _now(),
                error[:2000] if error else None,
                payload,
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

    # --- pending 대기열 (쿼터 초과분, FIFO) ---
    def mark_pending(self, paper: Paper) -> bool:
        """필터 통과 후 쿼터에 밀린 논문을 대기열에 넣는다. Paper 전체를 payload(JSON)로 보존.

        이미 pending이면 건드리지 않는다 — processed_at(=대기열 진입 시각)이 FIFO 순서이므로.
        반환: 새로 추가됐으면 True.
        """
        if self._status(paper.uid) == "pending":
            return False
        payload = json.dumps(dataclasses.asdict(paper), ensure_ascii=False)
        self._upsert(paper, "pending", None, payload=payload)
        return True

    def load_pending(self, topic: str, source: str) -> list[Paper]:
        """topic × source의 대기열을 오래된 순(FIFO)으로 반환."""
        rows = self.conn.execute(
            "SELECT uid, payload FROM papers WHERE status='pending' AND topic=? AND source=? "
            "ORDER BY processed_at ASC",
            (topic, source),
        ).fetchall()
        out: list[Paper] = []
        for r in rows:
            if not r["payload"]:
                logger.warning("pending 레코드에 payload 없음 — 건너뜀: %s", r["uid"])
                continue
            try:
                out.append(Paper(**json.loads(r["payload"])))
            except (TypeError, ValueError) as e:
                logger.warning("pending payload 복원 실패 — 건너뜀: %s (%s)", r["uid"], e)
        return out

    def load_all_pending(self, topics: list[str]) -> list[Paper]:
        """모든 (topic, source) 대기열을 합쳐 FIFO로 반환 — apply_quota 입력용."""
        out: list[Paper] = []
        for t in topics:
            for src in ("semantic_scholar", "arxiv"):
                out.extend(self.load_pending(t, src))
        return out

    def count_pending(self) -> dict[tuple[str, str], int]:
        rows = self.conn.execute(
            "SELECT topic, source, COUNT(*) AS n FROM papers WHERE status='pending' GROUP BY topic, source"
        ).fetchall()
        return {(r["topic"], r["source"]): r["n"] for r in rows}

    def expire_pending(self, max_age_days: int) -> int:
        """max_age_days보다 오래 대기한 pending을 'expired'로 폐기. 0이면 비활성. 반환: 폐기 수."""
        if max_age_days <= 0:
            return 0
        cutoff = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).isoformat()
        cur = self.conn.execute(
            "UPDATE papers SET status='expired', error=?, payload=NULL "
            "WHERE status='pending' AND processed_at < ?",
            (f"pending {max_age_days}일 초과", cutoff),
        )
        self.conn.commit()
        return cur.rowcount

    # --- S2 페이지 커서 (인용순 topic의 bulk 검색 continuation token) ---
    def get_s2_cursor(self, topic: str) -> dict | None:
        """topic의 저장된 커서. 없으면 None.
        반환 dict: {query_key, token, pages_done, updated_at}. token None = 1페이지부터."""
        row = self.conn.execute(
            "SELECT query_key, token, pages_done, updated_at FROM s2_cursor WHERE topic = ?", (topic,)
        ).fetchone()
        return dict(row) if row else None

    def save_s2_cursor(self, topic: str, query_key: str, token: str | None, pages_done: int) -> None:
        """커서 저장(upsert). token은 '다음에 요청할 페이지'의 토큰(None = 1페이지)."""
        self.conn.execute(
            """
            INSERT INTO s2_cursor (topic, query_key, token, pages_done, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(topic) DO UPDATE SET
                query_key  = excluded.query_key,
                token      = excluded.token,
                pages_done = excluded.pages_done,
                updated_at = excluded.updated_at
            """,
            (topic, query_key, token, pages_done, _now()),
        )
        self.conn.commit()

    def clear_s2_cursor(self, topic: str) -> None:
        """커서 삭제 — 결과 소진(마지막 페이지 도달)·쿼리 변경·토큰 무효 시 1페이지부터 다시."""
        self.conn.execute("DELETE FROM s2_cursor WHERE topic = ?", (topic,))
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
