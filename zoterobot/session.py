"""Per-DM conversation memory and the JSONL chat log.
DM별 대화 기억(최근 N쌍, 메모리에만) + 날짜별 jsonl 대화 로그.

- Conversations: DM 채널별 최근 질문·답 N쌍. 도구 호출 내역은 기억하지 않고, 리스너를 재시작하면 사라진다.
  RESET_WORDS 중 하나를 보내면 초기화.
- ChatLog: chatlog_dir/YYYY-MM-DD.jsonl 에 질문·답·도구 호출·참고 논문·소요 시간·상태를 한 줄씩 추가.
"""
from __future__ import annotations

import json
import os
import threading
from collections import deque
from datetime import datetime

RESET_WORDS = {"새 대화", "새대화", "초기화", "reset", "/new", "new chat"}


class Conversations:
    def __init__(self, turns: int):
        self.turns = turns
        self._h: dict[str, deque] = {}
        self._lock = threading.Lock()

    def history(self, channel: str) -> list[dict]:
        with self._lock:
            out = []
            for q, a in self._h.get(channel, ()):
                out += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
            return out

    def add(self, channel: str, question: str, answer: str) -> None:
        if self.turns <= 0:
            return
        with self._lock:
            self._h.setdefault(channel, deque(maxlen=self.turns)).append((question, answer))

    def reset(self, channel: str) -> None:
        with self._lock:
            self._h.pop(channel, None)


class ChatLog:
    def __init__(self, log_dir: str):
        self.dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, record: dict) -> None:
        now = datetime.now()
        rec = {"time": now.isoformat(timespec="seconds"), **record}
        path = os.path.join(self.dir, f"{now:%Y-%m-%d}.jsonl")
        with self._lock, open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
