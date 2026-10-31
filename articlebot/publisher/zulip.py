"""Zulip publishing: one message per article in a fixed stream topic (helpers shared with paperbot).
Zulip 발행 — zulip.stream / zulip.topic 하나에 글마다 메시지 하나(머리말 + 접힌 전체 요약 + 그림 링크).
인증·메시지 레이아웃·수식 공백 도우미는 paperbot/publisher/zulip.py 것을 그대로 쓴다(같은 봇 계정).
"""
from __future__ import annotations

import logging
import time

from paperbot.publisher.zulip import (ZULIP_ENV_VARS, _NamedBytes, _zulip_config_error, build_zulip_messages,
                                      make_zulip_client, split_zulip_message, zulip_auth_error, zulip_math_spacing,
                                      zulip_spoiler)

from ..config import Config
from ..models import Article, Extracted, FetchedImage
from .base import PublishConfigError, Publisher, _slug, extract_one_liner

logger = logging.getLogger(__name__)

__all__ = ["ZULIP_ENV_VARS", "ZulipPublisher", "build_zulip_messages", "format_main_zulip", "format_thread_zulip",
           "make_zulip_client", "split_zulip_message", "zulip_auth_error", "zulip_math_spacing", "zulip_spoiler"]


def format_meta_zulip(article: Article) -> str:
    parts = [article.source_label]
    if article.published:
        parts.append(article.published.strftime("%Y-%m-%d"))
    if article.categories:
        parts.append(", ".join(article.categories[:3]))
    if article.judge_score is not None:
        parts.append(f"관련도 {article.judge_score:.0f}/10")
    return " · ".join(parts)


def format_main_zulip(article: Article, summary: str) -> str:
    return (f"**{' '.join(article.title.split())}**\n"
            f"{format_meta_zulip(article)}\n"
            f"🔗 {article.url}\n"
            f"➤ {extract_one_liner(summary)}")


def format_thread_zulip(summary: str, ex: Extracted) -> str:
    out = summary.strip()
    if ex.links:
        out += "\n\n**📎 관련 링크**\n" + "\n".join(f"• [{label}]({url})" for label, url in ex.links)
    if ex.videos:
        out += "\n\n**🎬 영상**\n" + "\n".join(f"• {v}" for v in ex.videos)
    return out


class ZulipPublisher(Publisher):
    """스트림의 고정 topic(zulip.stream / zulip.topic)에 글마다 첫 메시지 → 요약 → 그림 링크.
    인증·스트림은 생성 시점에 확인한다(설정 문제는 PublishConfigError → 실행 중단, 백로그 보존)."""

    _MAX_RETRIES = 3

    def __init__(self, cfg: Config, client=None):
        self.stream, self.topic = cfg.zulip.stream, cfg.zulip.topic
        self.attach = cfg.images.attach
        if client is None:
            try:
                client = make_zulip_client(cfg.resolve(cfg.zulip.config_file))
            except RuntimeError as e:
                raise PublishConfigError(f"{e} (--dry-run이면 전송 없이 테스트 가능)") from e
        self.client = client
        profile = self.client.get_profile()
        if profile.get("result") != "success":
            raise PublishConfigError(zulip_auth_error(profile))
        logger.info("Zulip 봇 %s 인증 OK", profile.get("email") or profile.get("full_name"))
        try:
            sid = self.client.get_stream_id(self.stream)
        except Exception as e:  # noqa: BLE001 — 네트워크 등: 전송 때 다시 판정
            logger.warning("Zulip 스트림 '%s' 확인 실패(전송 때 재확인): %s", self.stream, e)
            sid = {"result": "success"}
        if sid.get("result") != "success":
            raise PublishConfigError(f"Zulip 스트림 '{self.stream}'이 없거나 봇이 볼 수 없습니다 — 스트림을 만들고 봇을 구독시키세요.")

    def _send(self, content: str) -> dict:
        last_err: Exception | None = None
        for attempt in range(1, self._MAX_RETRIES + 1):
            try:
                res = self.client.send_message({"type": "stream", "to": self.stream, "topic": self.topic, "content": content})
                if res.get("result") != "success":
                    if _zulip_config_error(res):
                        raise PublishConfigError(f"Zulip 설정 오류 '{res.get('msg')}': stream={self.stream!r}. 스트림·봇 구독·API 키를 점검하세요.")
                    raise RuntimeError(f"Zulip API 오류: {res.get('msg', res)}")
                return res
            except PublishConfigError:
                raise
            except Exception as e:  # noqa: BLE001
                last_err = e
                logger.warning("Zulip 전송 실패 (%d/%d): %s", attempt, self._MAX_RETRIES, e)
                if attempt < self._MAX_RETRIES:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"Zulip 전송 최종 실패: {last_err}") from last_err

    def publish(self, article: Article, summary: str, ex: Extracted, images: list[FetchedImage]) -> None:
        links = self._upload_images(images, article) if images and self.attach else []
        first = None
        for content in build_zulip_messages(format_main_zulip(article, summary), format_thread_zulip(summary, ex), links):
            res = self._send(content)
            first = first or res
        logger.info("Zulip 전송 완료: %s / %s id=%s 그림=%d", self.stream, self.topic, (first or {}).get("id"), len(links))

    def _upload_images(self, images: list[FetchedImage], article: Article) -> list[str]:
        """그림들을 업로드해 '[캡션](uri)' 링크 목록으로. 실패는 비치명적(빈 목록)."""
        slug = _slug(article)
        links: list[str] = []
        try:
            for i, img in enumerate(images, start=1):
                res = self.client.upload_file(_NamedBytes(img.data, f"{slug}_{i:02d}.{img.ext}"))
                if res.get("result") != "success":
                    logger.warning("그림 %d 업로드 실패(건너뜀): %s", i, res.get("msg", res))
                    continue
                uri = res.get("url") or res.get("uri")
                if uri:
                    links.append(f"[{img.label}]({uri})")
        except Exception as e:  # noqa: BLE001 — 그림은 비치명적
            logger.warning("그림 업로드 실패(텍스트만 발행): %s", e)
        return links
