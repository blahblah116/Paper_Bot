"""Publishing: Slack (channel + thread) or Zulip (stream topic), chosen by publisher.backend.
발행 패키지 — publisher.backend에 따라 Slack 또는 Zulip. 외부는 `from articlebot.publisher import …`로 아래 이름들을 쓴다.
"""
from __future__ import annotations

import logging

from ..config import Config
from .base import PublishConfigError, Publisher, extract_one_liner
from .dryrun import DryRunPublisher
from .slack import SlackPublisher, format_main, format_thread, md_to_mrkdwn, slack_escape
from .zulip import (ZULIP_ENV_VARS, ZulipPublisher, build_zulip_messages, format_main_zulip, format_thread_zulip,
                    make_zulip_client, split_zulip_message, zulip_auth_error, zulip_math_spacing, zulip_spoiler)

logger = logging.getLogger(__name__)

__all__ = [
    "Publisher", "PublishConfigError", "DryRunPublisher", "SlackPublisher", "ZulipPublisher", "make_publisher",
    "extract_one_liner", "slack_escape", "md_to_mrkdwn", "format_main", "format_thread",
    "format_main_zulip", "format_thread_zulip", "build_zulip_messages", "split_zulip_message", "zulip_spoiler",
    "zulip_math_spacing", "make_zulip_client", "zulip_auth_error", "ZULIP_ENV_VARS",
]


def make_publisher(cfg: Config, force_dry_run: bool = False) -> Publisher:
    backend = cfg.publisher.backend
    if cfg.publisher.dry_run or force_dry_run:
        logger.info("dry-run 모드: %s 전송 대신 stdout 출력 + %s 에 미리보기 저장", backend, cfg.storage.preview_dir)
        return DryRunPublisher(cfg)
    if backend == "zulip":
        return ZulipPublisher(cfg)
    return SlackPublisher(cfg)
