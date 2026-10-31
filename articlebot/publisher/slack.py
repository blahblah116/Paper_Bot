"""Slack publishing: channel message + thread summary + thread images.
Slack 발행 — 채널 메인(제목·출처·링크·한 줄 요약) + 스레드(요약·관련 링크·영상) + 스레드 그림.
토큰은 환경변수 SLACK_BOT_TOKEN (paperbot과 같은 봇). 그림 첨부에는 files:write 스코프가 필요하다.
"""
from __future__ import annotations

import logging
import os
import time

import requests

from ..config import Config
from ..models import Article, Extracted, FetchedImage
from .base import _MD_BOLD_RE, _MD_HEADING_RE, PublishConfigError, Publisher, _slug, extract_one_liner

logger = logging.getLogger(__name__)


_SLACK_CONFIG_ERRORS = {
    "channel_not_found", "not_in_channel", "is_archived", "invalid_auth", "not_authed",
    "account_inactive", "token_revoked", "token_expired", "missing_scope", "restricted_action",
}


def slack_escape(text: str) -> str:
    """Slack mrkdwn 제어 문자 이스케이프 (&, <, >)."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def md_to_mrkdwn(text: str) -> str:
    """요약 마크다운 → Slack mrkdwn (**볼드**·### 헤딩 → *볼드*)."""
    text = slack_escape(text)
    text = _MD_BOLD_RE.sub(r"*\1*", text)
    return _MD_HEADING_RE.sub(r"*\1*", text)


def format_meta(article: Article) -> str:
    parts = [article.source_label]
    if article.published:
        parts.append(article.published.strftime("%Y-%m-%d"))
    if article.categories:
        parts.append(", ".join(article.categories[:3]))
    if article.judge_score is not None:
        parts.append(f"관련도 {article.judge_score:.0f}/10")
    return " · ".join(slack_escape(p) for p in parts)


def format_main(article: Article, summary: str) -> str:
    return (f"*{slack_escape(article.title)}*\n"
            f"{format_meta(article)}\n"
            f"🔗 {article.url}\n"
            f"➤ {slack_escape(extract_one_liner(summary))}")


def format_thread(summary: str, ex: Extracted) -> str:
    out = md_to_mrkdwn(summary)
    if ex.links:
        items = "\n".join(f"• <{url}|{slack_escape(label)}>" for label, url in ex.links)
        out += f"\n\n*📎 관련 링크*\n{items}"
    if ex.videos:
        out += "\n\n*🎬 영상*\n" + "\n".join(f"• {v}" for v in ex.videos)
    return out


class SlackPublisher(Publisher):
    _API = "https://slack.com/api/"
    _MAX_RETRIES = 3

    def __init__(self, cfg: Config):
        token = os.environ.get("SLACK_BOT_TOKEN", "").strip()
        if not token:
            raise PublishConfigError(
                "환경변수 SLACK_BOT_TOKEN이 없습니다 (~/.paperbot_env를 source 하거나 --dry-run으로 테스트).")
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {token}"})
        self.default_channel = cfg.slack.channel
        self.attach = cfg.images.attach

    def _call(self, method: str, *, json_body: dict | None = None, form: dict | None = None) -> dict:
        last_err: Exception | None = None
        for attempt in range(1, self._MAX_RETRIES + 1):
            try:
                if json_body is not None:
                    resp = self.session.post(self._API + method, json=json_body, timeout=30,
                                             headers={"Content-Type": "application/json; charset=utf-8"})
                else:
                    resp = self.session.post(self._API + method, data=form, timeout=30)
                if resp.status_code == 429:
                    wait = int(resp.headers.get("Retry-After", 2 ** attempt))
                    time.sleep(min(wait, 60))
                    raise RuntimeError(f"rate limited (Retry-After={wait}s)")
                data = resp.json()
                if not data.get("ok"):
                    err = data.get("error", "unknown")
                    if err in _SLACK_CONFIG_ERRORS:
                        raise PublishConfigError(
                            f"Slack 설정 오류 '{err}' ({method}). 채널명·봇 초대(/invite)·토큰·스코프를 점검하세요.")
                    raise RuntimeError(f"Slack API 오류 {method}: {err}")
                return data
            except PublishConfigError:
                raise
            except Exception as e:  # noqa: BLE001 — 연결 오류/429 재시도
                last_err = e
                logger.warning("Slack %s 실패 (%d/%d): %s", method, attempt, self._MAX_RETRIES, e)
                if attempt < self._MAX_RETRIES:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"Slack {method} 최종 실패: {last_err}") from last_err

    def publish(self, article: Article, summary: str, ex: Extracted, images: list[FetchedImage]) -> None:
        channel = article.channel or self.default_channel
        main = self._call("chat.postMessage", json_body={
            "channel": channel,
            "text": format_main(article, summary),
            "unfurl_links": False,
            "unfurl_media": False,
        })
        channel_id, ts = main.get("channel", channel), main["ts"]
        self._call("chat.postMessage", json_body={
            "channel": channel_id,
            "thread_ts": ts,
            "text": format_thread(summary, ex),
            "unfurl_links": False,
            "unfurl_media": False,
        })
        if images and self.attach:
            self._upload_images(channel_id, ts, images, article)
        logger.info("Slack 전송 완료: channel=%s ts=%s", channel, ts)

    _scope_warned = False

    def _upload_images(self, channel_id: str, thread_ts: str, images: list[FetchedImage],
                       article: Article) -> None:
        """그림들을 스레드 댓글 하나로 묶어 업로드 (실패해도 발행은 성공으로 취급)."""
        slug = _slug(article)
        uploaded: list[dict] = []
        try:
            for i, img in enumerate(images, start=1):
                r = self.session.post(self._API + "files.getUploadURLExternal", timeout=30, data={
                    "filename": f"{slug}_{i:02d}.{img.ext}", "length": str(len(img.data)),
                }).json()
                if not r.get("ok"):
                    if r.get("error") == "missing_scope":
                        if not SlackPublisher._scope_warned:
                            SlackPublisher._scope_warned = True
                            logger.warning("그림 첨부에 files:write 스코프가 필요합니다 — 텍스트만 발행합니다.")
                        return
                    logger.warning("그림 %d 업로드 URL 발급 실패(건너뜀): %s", i, r.get("error"))
                    continue
                up = requests.post(r["upload_url"], data=img.data, timeout=120)
                up.raise_for_status()
                uploaded.append({"id": r["file_id"], "title": img.label})
            if uploaded:
                self._call("files.completeUploadExternal", json_body={
                    "channel_id": channel_id, "thread_ts": thread_ts, "files": uploaded,
                })
                logger.info("그림 %d장 첨부 완료", len(uploaded))
        except PublishConfigError:
            raise
        except Exception as e:  # noqa: BLE001 — 그림은 비치명적
            logger.warning("그림 업로드 실패(텍스트만 발행): %s", e)
