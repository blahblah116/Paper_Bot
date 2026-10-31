"""Publishing: Slack (channel message + thread) or Zulip (stream topic), chosen per topic.
발행 패키지 — topic별 platform 설정에 따라 Slack 또는 Zulip으로 보낸다.

- slack.SlackPublisher / zulip.ZulipPublisher: 실제 전송. dryrun.DryRunPublisher: stdout 미리보기.
- MultiPublisher: topic.platform에 따라 라우팅 (두 발행처를 함께 쓸 때만). make_publisher()가 조립한다.
- 외부(파이프라인·zoterobot·replay·테스트)는 `from paperbot.publisher import …`로 아래 이름들을 쓴다.
"""
from __future__ import annotations

import logging

from ..config import Config
from ..figures import Figure
from ..models import Paper
from .base import PublishConfigError, Publisher, extract_one_liner
from .dryrun import DryRunPublisher
from .slack import SlackPublisher, _topic_channel_map, format_slack_main, md_to_mrkdwn
from .zulip import (ZulipPublisher, _topic_stream_map, _zulip_topic_map, build_zulip_messages, format_zulip_main,
                    make_zulip_client, split_zulip_message, zulip_auth_error, zulip_math_spacing, zulip_topic_for,
                    zulip_topic_name)

logger = logging.getLogger(__name__)

__all__ = [
    "Publisher", "PublishConfigError", "DryRunPublisher", "SlackPublisher", "ZulipPublisher", "MultiPublisher", "make_publisher",
    "extract_one_liner", "md_to_mrkdwn", "format_slack_main",
    "zulip_topic_name", "zulip_topic_for", "format_zulip_main", "build_zulip_messages", "split_zulip_message",
    "zulip_math_spacing", "make_zulip_client", "zulip_auth_error",
]


class MultiPublisher(Publisher):
    """topic.platform에 따라 Slack/Zulip publisher로 라우팅 (두 발행처를 함께 쓸 때)."""

    def __init__(self, by_platform: dict[str, Publisher], topic_platform: dict[str, str], default: str = "slack"):
        self.by_platform = by_platform
        self.topic_platform = topic_platform
        self.default = default

    def _for_topic(self, paper_topic: str) -> Publisher:
        return self.by_platform.get(self.topic_platform.get(paper_topic, self.default)) or self.by_platform[self.default]

    def publish(self, paper: Paper, summary: str, abstract_only: bool = False,
                figures: list[Figure] | None = None) -> None:
        self._for_topic(paper.topic_name).publish(paper, summary, abstract_only=abstract_only, figures=figures)

    def close(self) -> None:
        for p in self.by_platform.values():
            p.close()


def _build_publisher(cfg: Config, platform: str, force_dry_run: bool) -> Publisher:
    if platform == "zulip":
        if cfg.zulip.dry_run or force_dry_run:
            logger.info("dry-run 모드: Zulip 전송 대신 stdout 출력")
            return DryRunPublisher(default_channel=cfg.zulip.stream, topic_channels=_topic_stream_map(cfg),
                                   backend="zulip", zulip_topics=_zulip_topic_map(cfg))
        return ZulipPublisher(cfg)
    if cfg.slack.dry_run or force_dry_run:
        logger.info("dry-run 모드: Slack 전송 대신 stdout 출력")
        return DryRunPublisher(default_channel=cfg.slack.channel, topic_channels=_topic_channel_map(cfg))
    return SlackPublisher(cfg)


def make_publisher(cfg: Config, force_dry_run: bool = False) -> Publisher:
    """topic들이 쓰는 발행처(slack/zulip)별로 publisher를 만들고, 둘 다 쓰면 MultiPublisher로 묶는다.
    각 발행처는 자기 dry_run(slack.dry_run / zulip.dry_run) 또는 --dry-run이면 stdout 출력."""
    platforms = cfg.platforms() or ["slack"]
    built = {p: _build_publisher(cfg, p, force_dry_run) for p in platforms}
    if len(built) == 1:
        return next(iter(built.values()))
    return MultiPublisher(built, {t.name: t.platform for t in cfg.topics}, default=platforms[0])
