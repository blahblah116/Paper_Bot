"""PaperSource 추상 클래스."""
from __future__ import annotations

import abc

from ..config import Config, Topic
from ..models import Paper


class PaperSource(abc.ABC):
    """검색 소스 인터페이스. 새 소스는 이 클래스를 구현한다."""

    name: str = "base"

    def __init__(self, cfg: Config, *, backfill: bool = False):
        self.cfg = cfg
        self.backfill = backfill  # True면 lookback 필터를 적용하지 않음 (첫 실행용)

    @abc.abstractmethod
    def fetch(self, topic: Topic) -> list[Paper]:
        """topic 하나에 대한 검색 결과를 Paper 리스트로 반환.

        해당 topic에서 이 소스가 비활성이면 빈 리스트를 반환한다.
        예외는 호출자(run.py)가 소스 단위로 격리한다.
        """
