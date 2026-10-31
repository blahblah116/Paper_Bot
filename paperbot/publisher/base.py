"""Publisher interface and text helpers shared by the Slack and Zulip publishers.
발행 인터페이스와 Slack·Zulip이 함께 쓰는 텍스트 도우미(메타 줄, 한 줄 요약 추출).

- Publisher: 발행 인터페이스. 새 발행처는 이 클래스를 구현하고 __init__.make_publisher에 연결한다.
- PublishConfigError: 토큰·채널·스트림 같은 '설정' 문제. 논문별 재시도가 무의미하므로 실행 전체를 중단한다.
"""
from __future__ import annotations

import abc
import re

from ..figures import Figure, label_slug
from ..models import Paper


class PublishConfigError(Exception):
    """전송 '설정' 문제(채널/토큰/권한). 논문별 재시도가 무의미하므로
    개별 논문을 failed로 소모하지 않고 실행 전체를 중단해야 한다."""


# 요약 마크다운에서 "### 1. 한 줄 요약" 섹션 본문만 추출 (메인 메시지의 한 줄 요약).
_ONE_LINER_RE = re.compile(r"###\s*1\.\s*한\s*줄\s*요약\s*\n+(.*?)(?=\n###|\Z)", re.DOTALL)
# 우리 요약 형식의 마크다운 → Slack mrkdwn 변환용.
_MD_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_MD_HEADING_RE = re.compile(r"^#{1,6}\s*(.+)$", re.MULTILINE)


def figure_filename(paper: Paper, fig: Figure) -> str:
    """첨부 파일명: uid의 ':'·'/'를 '_'로 + 라벨 ('arxiv_2401.12345_table_1.png')."""
    return f"{paper.uid.replace(':', '_').replace('/', '_')}_{label_slug(fig.label)}.png"


def _format_authors(authors: list[str]) -> str:
    if not authors:
        return "저자 미상"
    head = authors[:3]
    s = ", ".join(head)
    if len(authors) > 3:
        s += ", et al."
    return s


def _format_meta(paper: Paper, abstract_only: bool, italic: str = "_") -> str:
    """메타 줄. italic은 플랫폼의 기울임 표기 — Slack mrkdwn '_', Zulip 마크다운 '*'."""
    venue = paper.venue or "arXiv preprint"
    meta_parts = [_format_authors(paper.authors), f"{venue} {paper.year or ''}".strip()]
    if paper.citation_count is not None:
        meta_parts.append(f"인용 {paper.citation_count}")
    meta_parts.append(f"`#{paper.topic_name}`")
    meta = " · ".join(meta_parts)
    if abstract_only:
        meta += f" · {italic}(abstract 기반 요약){italic}"
    return meta


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


class Publisher(abc.ABC):
    """발행 인터페이스. 새 백엔드는 이 클래스를 구현한다."""

    @abc.abstractmethod
    def publish(self, paper: Paper, summary: str, abstract_only: bool = False,
                figures: list[Figure] | None = None) -> None:
        ...

    def close(self) -> None:  # 필요 시 오버라이드
        pass
