"""발행(Publisher).

추상 Publisher를 기준으로 세 구현을 제공한다:
- ZulipPublisher: 스트림에 논문당 topic 하나, 전체 요약 한 메시지
- SlackPublisher: 채널에 메인 메시지(제목·메타·링크·한 줄 요약) + 스레드에 전체 요약
- DryRunPublisher: 전송 없이 stdout (backend에 맞는 레이아웃으로 미리보기)
Notion 아카이브는 이번 범위 밖 — NotionPublisher는 인터페이스 스텁만 둔다.
"""
from __future__ import annotations

import abc
import logging
import os
import re
import time

from .config import Config
from .models import Paper

logger = logging.getLogger(__name__)

_TOPIC_MAX = 60


class PublishConfigError(Exception):
    """전송 '설정' 문제(채널/토큰/권한). 논문별 재시도가 무의미하므로
    개별 논문을 failed로 소모하지 않고 실행 전체를 중단해야 한다."""


# 재시도해도 소용없는 Slack 설정 오류들.
_SLACK_CONFIG_ERRORS = {
    "channel_not_found",   # 채널명 오타 또는 비공개 채널에 봇 미초대
    "not_in_channel",      # 공개 채널에 봇 미초대 (/invite @봇이름)
    "is_archived",         # 보관된 채널
    "invalid_auth", "not_authed", "account_inactive",
    "token_revoked", "token_expired",
    "missing_scope",       # chat:write 스코프 누락
    "restricted_action",
}

# 요약 마크다운에서 "### 1. 한 줄 요약" 섹션 본문만 추출 (Slack 메인 메시지용).
_ONE_LINER_RE = re.compile(r"###\s*1\.\s*한\s*줄\s*요약\s*\n+(.*?)(?=\n###|\Z)", re.DOTALL)
# 우리 요약 형식의 마크다운 → Slack mrkdwn 변환용.
_MD_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_MD_HEADING_RE = re.compile(r"^#{1,6}\s*(.+)$", re.MULTILINE)


def _format_authors(authors: list[str]) -> str:
    if not authors:
        return "저자 미상"
    head = authors[:3]
    s = ", ".join(head)
    if len(authors) > 3:
        s += ", et al."
    return s


def _topic_for(paper: Paper) -> str:
    """Zulip topic = 논문 제목 (60자 초과 시 절단)."""
    title = paper.title
    if len(title) > _TOPIC_MAX:
        return title[: _TOPIC_MAX - 1].rstrip() + "…"
    return title


def _format_meta(paper: Paper, abstract_only: bool) -> str:
    venue = paper.venue or "arXiv preprint"
    meta_parts = [_format_authors(paper.authors), f"{venue} {paper.year or ''}".strip()]
    if paper.citation_count is not None:
        meta_parts.append(f"인용 {paper.citation_count}")
    meta_parts.append(f"`#{paper.topic_name}`")
    meta = " · ".join(meta_parts)
    if abstract_only:
        meta += " · _(abstract 기반 요약)_"
    return meta


def format_message(paper: Paper, summary: str, abstract_only: bool) -> str:
    """Zulip용 단일 메시지 형식.

    **{title}**
    {authors 앞 3명, et al.} · {venue or "arXiv preprint"} {year} · 인용 {n} · `#{topic}`
    📄 {url}
    """
    return (
        f"**{paper.title}**\n"
        f"{_format_meta(paper, abstract_only)}\n"
        f"📄 {paper.url}\n\n"
        f"{summary}"
    )


# ---------------------------------------------------------------- Slack 헬퍼

def extract_one_liner(summary: str, max_chars: int = 300) -> str:
    """요약 마크다운에서 '한 줄 요약' 섹션 본문을 추출 (실패 시 앞부분으로 폴백)."""
    m = _ONE_LINER_RE.search(summary)
    text = m.group(1) if m else summary
    text = " ".join(text.split())  # 개행/중복 공백 정리
    # 폴백 시 남아 있을 수 있는 헤딩/볼드 장식 제거
    text = _MD_HEADING_RE.sub(r"\1", _MD_BOLD_RE.sub(r"\1", text)).strip()
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    return text


def md_to_mrkdwn(text: str) -> str:
    """우리 요약 형식의 마크다운을 Slack mrkdwn으로 변환.

    Slack은 **볼드**와 ### 헤딩을 렌더링하지 않으므로
    둘 다 *볼드* 한 줄로 바꾼다. 불릿/백틱은 그대로 동작.
    """
    text = _MD_BOLD_RE.sub(r"*\1*", text)
    text = _MD_HEADING_RE.sub(r"*\1*", text)
    return text


def format_slack_main(paper: Paper, summary: str, abstract_only: bool) -> str:
    """Slack 채널 메인 메시지 — 제목·메타·링크·한 줄 요약 (mrkdwn)."""
    return (
        f"*{paper.title}*\n"
        f"{_format_meta(paper, abstract_only)}\n"
        f"📄 {paper.url}\n"
        f"➤ {extract_one_liner(summary)}"
    )


def format_slack_thread(summary: str) -> str:
    """Slack 스레드 댓글 — 6섹션 전체 요약 (mrkdwn)."""
    return md_to_mrkdwn(summary)


def _topic_channel_map(cfg: Config) -> dict[str, str]:
    """topic name → 전용 발행 대상(채널/스트림) 매핑. channel 미지정 topic은 제외."""
    return {t.name: t.channel for t in cfg.topics if t.channel}


class Publisher(abc.ABC):
    """발행 인터페이스. 새 백엔드(Notion 등)는 이 클래스를 구현한다."""

    @abc.abstractmethod
    def publish(self, paper: Paper, summary: str, abstract_only: bool = False,
                figures: list[bytes] | None = None) -> None:
        ...

    def close(self) -> None:  # 필요 시 오버라이드
        pass


class DryRunPublisher(Publisher):
    """전송 없이 stdout으로 출력 (개발/테스트용). backend 레이아웃대로 미리보기."""

    def __init__(self, backend: str = "zulip", default_channel: str = "?",
                 topic_channels: dict[str, str] | None = None):
        self.backend = backend
        self.default_channel = default_channel
        self.topic_channels = topic_channels or {}

    def _dest(self, paper: Paper) -> str:
        return self.topic_channels.get(paper.topic_name, self.default_channel)

    def publish(self, paper: Paper, summary: str, abstract_only: bool = False,
                figures: list[bytes] | None = None) -> None:
        print("=" * 70)
        if self.backend == "slack":
            fig_note = ""
            if figures:
                sizes = ", ".join(f"{len(f)//1024}KB" for f in figures)
                fig_note = f" (+figure {len(figures)}개 첨부 예정: {sizes})"
            print(f"[DRY-RUN/slack → {self._dest(paper)}] 채널 메인 메시지")
            print("-" * 70)
            print(format_slack_main(paper, summary, abstract_only))
            print("-" * 70)
            print(f"[DRY-RUN/slack → {self._dest(paper)}] 스레드 댓글{fig_note}")
            print("-" * 70)
            print(format_slack_thread(summary))
        else:
            print(f"[DRY-RUN/zulip → {self._dest(paper)}] topic={_topic_for(paper)!r}")
            print("-" * 70)
            print(format_message(paper, summary, abstract_only))
        print("=" * 70)


class ZulipPublisher(Publisher):
    """Zulip 스트림에 논문당 topic 하나로 포스팅."""

    def __init__(self, cfg: Config):
        import zulip  # 지연 임포트 — dry-run 경로에선 필요 없음

        config_file = cfg.resolve(cfg.zulip.config_file)
        self.client = zulip.Client(config_file=config_file)
        self.stream = cfg.zulip.stream
        self.topic_channels = _topic_channel_map(cfg)  # topic.channel → 스트림 override

    def publish(self, paper: Paper, summary: str, abstract_only: bool = False,
                figures: list[bytes] | None = None) -> None:
        # figure 첨부는 현재 Slack 전용 (Zulip은 텍스트만).
        stream = self.topic_channels.get(paper.topic_name, self.stream)
        topic = _topic_for(paper)
        content = format_message(paper, summary, abstract_only)
        result = self.client.send_message(
            {
                "type": "stream",
                "to": stream,
                "topic": topic,
                "content": content,
            }
        )
        if result.get("result") != "success":
            raise RuntimeError(f"Zulip 전송 실패: {result.get('msg', result)}")
        logger.info("Zulip 전송 완료: stream=%s topic=%r", stream, topic)


class SlackPublisher(Publisher):
    """Slack 채널에 메인 메시지 + 스레드 댓글로 포스팅.

    - 토큰: 환경변수 SLACK_BOT_TOKEN (봇 토큰 xoxb-..., chat:write 스코프)
    - 채널: config의 slack.channel (채널명 "#papers" 또는 채널 ID)
    - 429(rate limit)는 Retry-After를 존중해 3회 재시도.
    """

    _API_URL = "https://slack.com/api/chat.postMessage"
    _MAX_RETRIES = 3

    def __init__(self, cfg: Config):
        import requests  # 지연 임포트 — dry-run 경로에선 필요 없음

        token = os.environ.get("SLACK_BOT_TOKEN", "").strip()
        if not token:
            raise PublishConfigError(
                "환경변수 SLACK_BOT_TOKEN이 없습니다. Slack 앱의 봇 토큰(xoxb-...)을 "
                "설정하거나 dry_run: true로 테스트하세요. (발급 방법은 README 참조)"
            )
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        })
        self.channel = cfg.slack.channel                # 기본 채널
        self.topic_channels = _topic_channel_map(cfg)   # topic별 전용 채널
        self.attach_figure = cfg.slack.attach_figure    # 대표 figure 스레드 첨부 여부

    def _channel_for(self, paper: Paper) -> str:
        return self.topic_channels.get(paper.topic_name, self.channel)

    def _post(self, payload: dict, url: str | None = None) -> dict:
        last_err: Exception | None = None
        for attempt in range(1, self._MAX_RETRIES + 1):
            try:
                resp = self.session.post(url or self._API_URL, json=payload, timeout=30)
                if resp.status_code == 429:
                    wait = int(resp.headers.get("Retry-After", 2 ** attempt))
                    raise RuntimeError(f"rate limited (Retry-After={wait}s)")
                data = resp.json()
                if not data.get("ok"):
                    err = data.get("error", "unknown")
                    if err in _SLACK_CONFIG_ERRORS:
                        # 설정 문제 — 재시도/논문 소모 없이 실행 중단용 예외.
                        raise PublishConfigError(
                            f"Slack 설정 오류 '{err}': channel={payload.get('channel')!r}. "
                            "채널명 확인, 봇 초대(/invite), 토큰·스코프를 점검하세요."
                        )
                    raise RuntimeError(f"Slack API 오류: {err}")
                return data
            except PublishConfigError:
                raise
            except Exception as e:  # noqa: BLE001 — 연결 오류/429 재시도
                last_err = e
                wait = 2 ** attempt
                logger.warning("Slack 전송 실패 (%d/%d): %s", attempt, self._MAX_RETRIES, e)
                if attempt < self._MAX_RETRIES:
                    time.sleep(wait)
        raise RuntimeError(f"Slack 전송 최종 실패: {last_err}") from last_err

    def publish(self, paper: Paper, summary: str, abstract_only: bool = False,
                figures: list[bytes] | None = None) -> None:
        channel = self._channel_for(paper)
        # 1) 채널 메인 메시지 (제목·메타·링크·한 줄 요약)
        main = self._post({
            "channel": channel,
            "text": format_slack_main(paper, summary, abstract_only),
            "unfurl_links": False,   # arXiv 링크 프리뷰로 채널이 길어지는 것 방지
            "unfurl_media": False,
        })
        channel_id = main.get("channel", channel)  # 채널명 → ID 정규화
        # 2) 같은 메시지의 스레드에 전체 요약
        self._post({
            "channel": channel_id,
            "thread_ts": main["ts"],
            "text": format_slack_thread(summary),
            "unfurl_links": False,
            "unfurl_media": False,
        })
        # 3) (옵션) 대표 figure들을 같은 스레드에 첨부 — 실패해도 발행은 성공으로 취급.
        if figures and self.attach_figure:
            self._upload_figures(channel_id, main["ts"], figures, paper)
        logger.info("Slack 전송 완료: channel=%s ts=%s", channel, main["ts"])

    _scope_warned = False  # files:write 스코프 누락 경고는 1회만

    def _upload_figures(self, channel_id: str, thread_ts: str,
                        figures: list[bytes], paper: Paper) -> None:
        """figure들을 스레드 댓글 하나에 묶어 업로드 (files:write 스코프 필요, 비치명적).

        Slack 2단계 external upload: 파일별 getUploadURLExternal → 바이트 업로드,
        마지막에 completeUploadExternal 한 번으로 스레드에 일괄 공유.
        """
        import requests

        safe_uid = paper.uid.replace(":", "_").replace("/", "_")
        uploaded: list[dict] = []
        try:
            for i, fig in enumerate(figures, start=1):
                # 1단계: 업로드 URL 발급 (form-encoded 엔드포인트)
                r = self.session.post(
                    "https://slack.com/api/files.getUploadURLExternal",
                    data={"filename": f"{safe_uid}_fig{i}.png", "length": str(len(fig))},
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    timeout=30,
                ).json()
                if not r.get("ok"):
                    if r.get("error") == "missing_scope":
                        if not SlackPublisher._scope_warned:
                            SlackPublisher._scope_warned = True
                            logger.warning(
                                "figure 첨부에 files:write 스코프가 필요합니다 — Slack 앱 설정에서 "
                                "스코프 추가 후 재설치하세요. (이후 figure 없이 텍스트만 발행)"
                            )
                        return  # 스코프 없으면 나머지도 소용없음
                    logger.warning("figure %d 업로드 URL 발급 실패(건너뜀): %s", i, r.get("error"))
                    continue
                # 2단계: 바이트 업로드 (pre-signed URL, 인증 헤더 불필요)
                up = requests.post(r["upload_url"], data=fig, timeout=60)
                up.raise_for_status()
                uploaded.append({"id": r["file_id"], "title": f"Figure {i}"})

            if not uploaded:
                return
            # 3단계: 업로드 완료 → 스레드 댓글 하나로 일괄 공유
            self._post({
                "channel_id": channel_id,
                "thread_ts": thread_ts,
                "files": uploaded,
            }, url="https://slack.com/api/files.completeUploadExternal")
            logger.info("figure %d개 첨부 완료 (%s)", len(uploaded),
                        ", ".join(f"{len(f)//1024}KB" for f in figures[:len(uploaded)]))
        except PublishConfigError:
            raise  # 토큰/권한 문제는 상위 정책대로 중단
        except Exception as e:  # noqa: BLE001 — figure는 비치명적
            logger.warning("figure 업로드 실패(텍스트만 발행): %s", e)


class NotionPublisher(Publisher):
    """[스텁] Notion 데이터베이스 아카이브 — 이번 범위 밖.

    향후 구현 시 이 클래스에 Notion API 연동을 채워 넣고,
    make_publisher가 반환하도록 연결하면 된다.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def publish(self, paper: Paper, summary: str, abstract_only: bool = False,
                figures: list[bytes] | None = None) -> None:
        raise NotImplementedError("Notion 아카이브는 아직 구현되지 않았습니다.")


def make_publisher(cfg: Config, force_dry_run: bool = False) -> Publisher:
    """config의 publisher.backend와 dry_run에 따라 적절한 Publisher를 생성."""
    backend = cfg.publisher.backend
    if cfg.publisher.dry_run or force_dry_run:
        logger.info("dry-run 모드: %s 전송 대신 stdout 출력", backend)
        default = cfg.slack.channel if backend == "slack" else cfg.zulip.stream
        return DryRunPublisher(backend=backend, default_channel=default,
                               topic_channels=_topic_channel_map(cfg))
    if backend == "slack":
        return SlackPublisher(cfg)
    return ZulipPublisher(cfg)
