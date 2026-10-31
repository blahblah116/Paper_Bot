"""Lockfile helpers: one articlebot at a time, and "is the peer bot (paperbot) running?" for the shared model.
중복 실행 방지 lockfile + 같은 모델을 쓰는 다른 봇(paperbot)이 실행 중인지 확인(언로드 생략 판단).
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger("articlebot")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def lock_held(path: str) -> bool:
    """lockfile이 있고 그 pid가 살아 있으면 True."""
    try:
        with open(path) as f:
            pid = int(f.read().strip() or "0")
    except (OSError, ValueError):
        return False
    return pid > 0 and _pid_alive(pid)


def acquire_lock(path: str) -> bool:
    if lock_held(path):
        logger.warning("이미 실행 중입니다 (%s). 종료합니다.", path)
        return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(str(os.getpid()))
    return True


def release_lock(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass
