"""Dry-run publisher: prints the Slack or Zulip layout instead of posting.
전송 없이 해당 발행처의 레이아웃을 stdout에 출력 (개발·테스트용).
"""
from __future__ import annotations

from ..figures import Figure
from ..models import Paper
from .base import Publisher
from .slack import format_slack_main, format_slack_thread
from .zulip import build_zulip_messages, format_zulip_main, zulip_topic_for


class DryRunPublisher(Publisher):
    """전송 없이 stdout으로 출력 (개발/테스트용). backend("slack"|"zulip")의 레이아웃대로 미리보기."""

    def __init__(self, default_channel: str = "?", topic_channels: dict[str, str] | None = None,
                 backend: str = "slack", zulip_topics: dict[str, str] | None = None):
        self.default_channel = default_channel
        self.topic_channels = topic_channels or {}
        self.backend = backend
        self.zulip_topics = zulip_topics or {}     # zulip: 주제별 고정 topic (없으면 제목)

    def _dest(self, paper: Paper) -> str:
        dest = self.topic_channels.get(paper.topic_name, self.default_channel)
        if self.backend == "zulip":
            return f"{dest} / {zulip_topic_for(paper, self.zulip_topics.get(paper.topic_name))}"
        return dest

    def publish(self, paper: Paper, summary: str, abstract_only: bool = False,
                figures: list[Figure] | None = None) -> None:
        fig_note = ""
        if figures:
            sizes = ", ".join(f"{f.label} {len(f.png)//1024}KB" for f in figures)
            fig_note = f" (+figure/표 {len(figures)}개 첨부 예정: {sizes})"
        tag = f"[DRY-RUN/{self.backend} → {self._dest(paper)}]"
        print("=" * 70)
        if self.backend == "zulip":
            msgs = build_zulip_messages(format_zulip_main(paper, summary, abstract_only), summary,
                                        [f"[{f.label}](<업로드 예정 {len(f.png)//1024}KB>)" for f in (figures or [])])
            for i, m in enumerate(msgs, start=1):
                print(f"{tag} 메시지 {i}/{len(msgs)}{fig_note if i == 1 else ''}")
                print("-" * 70)
                print(m)
                print("-" * 70)
        else:
            print(f"{tag} 채널 메인 메시지")
            print("-" * 70)
            print(format_slack_main(paper, summary, abstract_only))
            print("-" * 70)
            print(f"{tag} 스레드 댓글{fig_note}")
            print("-" * 70)
            print(format_slack_thread(summary))
        print("=" * 70)
