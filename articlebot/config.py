"""config.yaml 로드 및 검증.

소스 추가·채널·필터·이미지 정책은 전부 config.yaml에서 조절한다 (코드 수정 없이).
누락/오타는 여기서 명확히 실패시킨다.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

import yaml


class ConfigError(Exception):
    """설정 파일이 유효하지 않을 때 발생."""


# ---------------------------------------------------------------- 섹션들

@dataclass(frozen=True)
class SourceConfig:
    name: str                      # 내부 키 — DB 기록에 쓰이므로 운영 중 바꾸지 말 것
    label: str                     # Slack에 표시할 출처 이름
    feed: str                      # RSS/Atom URL
    org: str = ""                  # 기관 — 하루 발행 상한(report.per_org)의 단위. 여러 피드가 한 기관을 공유 가능
    enabled: bool = True
    judge: bool = True             # LLM relevance judge 적용 (threshold 미만은 버림)
    channel: str | None = None     # 이 소스 전용 채널 (None이면 slack.channel)
    include_categories: tuple[str, ...] = ()   # 피드 category 화이트리스트 (비면 전체 허용)
    exclude_categories: tuple[str, ...] = ()   # 블랙리스트 (대소문자 무시)


@dataclass(frozen=True)
class FetchConfig:
    interval_days: int = 3         # 수집(피드 확인·채점) 주기. 사이 날에는 백로그에서만 발행. 1 이하 = 매 실행
    lookback_days: int = 90        # 이 기간 안에 게시된 글만 수집·백로그 대상 (더 오래된 백로그는 expired)
    timeout: int = 30
    user_agent: str = "articlebot/0.1 (research blog digest bot)"
    min_interval: float = 1.0      # 같은 호스트 연속 요청 간 최소 간격(초)
    min_body_chars: int = 400      # 페이지 본문이 이보다 짧으면 피드 본문으로 폴백, 둘 다 짧으면 요약 skip


@dataclass(frozen=True)
class JudgeConfig:
    interest: str = ""             # 채점 기준 (자연어 관심사 서술)
    threshold: int = 5             # 0~10 점수 하한 — 미만은 filtered_out
    excerpt_chars: int = 3000      # judge에 넣을 본문 앞부분 길이 (피드 요약문이 부실한 소스가 많아 본문 사용)


@dataclass(frozen=True)
class ReportConfig:
    per_org: int = 1               # 기관별 하루 발행 수 — 백로그(DB)에서 최신순으로 꺼낸다
    max_attempts: int = 3          # 기관별로 처리 실패 시 다음 후보를 시도하는 최대 횟수


@dataclass(frozen=True)
class ImageConfig:
    attach: bool = True            # 스레드에 그림 첨부 (Slack 앱에 files:write 스코프 필요)
    max_images: int = 4            # 글당 첨부 상한 — 후보가 더 많으면 LLM이 캡션을 보고 고른다
    max_candidates: int = 12       # 다운로드해 검사할 후보 상한 (본문 등장 순)
    hero: str = "fallback"         # og:image 사용: "fallback"(본문 그림 없을 때만) | "always" | "never"
    min_width: int = 320           # 이보다 작은 그림(아이콘·썸네일)은 버림
    min_height: int = 160
    max_download_bytes: int = 20_000_000
    max_upload_bytes: int = 8_000_000   # 초과 시 재인코딩 (애니메이션 GIF는 첫 프레임)
    max_side: int = 2400           # 긴 변이 이보다 크면 축소


@dataclass(frozen=True)
class LLMConfig:
    base_url: str = "http://localhost:8090/v1"   # llama-swap
    model: str = "qwen3.8:27b"
    max_input_chars: int = 60000   # 본문 절단 상한 (블로그 글은 보통 5k~20k자)
    max_output_tokens: int = 4096
    temperature: float = 0.3
    language: str = "ko"
    unload_after_run: bool = True  # 실행 종료 시 모델 언로드(VRAM 즉시 반납)
    peer_lockfiles: tuple[str, ...] = ()   # 같은 LLM 서버를 쓰는 다른 봇의 lockfile — 그 봇이 실행 중이면 언로드하지 않음


@dataclass(frozen=True)
class PublisherConfig:
    backend: str = "slack"         # "slack" | "zulip" — 어디로 보낼지
    dry_run: bool = False          # true면 전송 대신 stdout 출력 + 그림을 preview_dir에 저장


@dataclass(frozen=True)
class SlackConfig:
    channel: str = "#research_blogs"   # 채널명("#name") 또는 채널 ID — 봇 초대 필수
    # 봇 토큰은 config에 넣지 않고 환경변수 SLACK_BOT_TOKEN으로 주입한다.


@dataclass(frozen=True)
class ZulipConfig:
    """publisher.backend가 zulip일 때. 인증은 환경변수 ZULIP_EMAIL/ZULIP_API_KEY/ZULIP_SITE(~/.paperbot_env), 없으면 config_file(zuliprc).
    글마다 stream의 topic에 첫 메시지(제목·출처·링크·한 줄) → 전체 요약 → 그림 순으로 이어 올린다(소스별 channel은 무시)."""
    stream: str = "papers"         # 스트림 이름 — 봇 구독 필수
    topic: str = "articles"        # 스트림 안의 고정 topic (60자 이하)
    config_file: str = "../zuliprc" # 예비용 zuliprc (레포 루트 — paperbot·zoterobot과 공용)


@dataclass(frozen=True)
class StorageConfig:
    db_path: str = "./data/articles.db"
    preview_dir: str = "./data/preview"
    chat_text_db: str = ""         # zoterobot 본문 캐시(library_text.db) — 발행한 글의 본문·요약을 넣어 DM 챗봇이 찾게 한다. 비면 끔


@dataclass(frozen=True)
class Config:
    sources: list[SourceConfig]
    fetch: FetchConfig
    judge: JudgeConfig
    report: ReportConfig
    images: ImageConfig
    llm: LLMConfig
    publisher: PublisherConfig
    slack: SlackConfig
    storage: StorageConfig
    zulip: ZulipConfig = field(default_factory=ZulipConfig)
    base_dir: str = field(default=".")     # config.yaml 위치 — 상대 경로 기준점

    def resolve(self, path: str) -> str:
        """config 내 상대 경로를 config.yaml 기준 절대 경로로 변환."""
        if os.path.isabs(path):
            return path
        return os.path.normpath(os.path.join(self.base_dir, path))


# ---------------------------------------------------------------- 파서

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_HERO_MODES = ("fallback", "always", "never")


def _as_dict(value: Any, ctx: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"'{ctx}'는 매핑이어야 합니다. (현재: {type(value).__name__})")
    return value


def _str_tuple(value: Any, ctx: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ConfigError(f"'{ctx}'는 리스트여야 합니다.")
    return tuple(str(v).strip() for v in value if str(v).strip())


def _section(cls, raw: Any, ctx: str):
    """dataclass 기본값 위에 YAML 값을 덮어쓴다. 모르는 키는 오타로 보고 실패."""
    d = _as_dict(raw, ctx)
    known = set(cls.__dataclass_fields__)
    unknown = set(d) - known
    if unknown:
        raise ConfigError(f"'{ctx}'에 알 수 없는 키: {sorted(unknown)} (가능: {sorted(known)})")
    return cls(**d)


def _parse_source(i: int, raw: Any) -> SourceConfig:
    ctx = f"sources[{i}]"
    d = dict(_as_dict(raw, ctx))
    for key in ("name", "feed"):
        if not str(d.get(key) or "").strip():
            raise ConfigError(f"{ctx}에 필수 키 '{key}'가 없습니다.")
    name = str(d["name"]).strip()
    if not _NAME_RE.match(name):
        raise ConfigError(f"{ctx}.name은 영소문자·숫자·_·-만 허용합니다: {name!r}")
    feed = str(d["feed"]).strip()
    if not feed.startswith(("http://", "https://")):
        raise ConfigError(f"{ctx}.feed는 http(s) URL이어야 합니다: {feed!r}")
    known = set(SourceConfig.__dataclass_fields__)
    unknown = set(d) - known
    if unknown:
        raise ConfigError(f"{ctx}에 알 수 없는 키: {sorted(unknown)}")
    label = str(d.get("label") or name).strip()
    return SourceConfig(
        name=name,
        label=label,
        feed=feed,
        org=str(d.get("org") or label).strip(),
        enabled=bool(d.get("enabled", True)),
        judge=bool(d.get("judge", True)),
        channel=(str(d["channel"]).strip() or None) if d.get("channel") else None,
        include_categories=_str_tuple(d.get("include_categories"), f"{ctx}.include_categories"),
        exclude_categories=_str_tuple(d.get("exclude_categories"), f"{ctx}.exclude_categories"),
    )


def load_config(path: str) -> Config:
    if not os.path.exists(path):
        raise ConfigError(f"설정 파일이 없습니다: {path} (config.example.yaml을 복사하세요)")
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    root = _as_dict(raw, "config")

    raw_sources = root.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ConfigError("'sources'에 소스가 하나 이상 있어야 합니다.")
    sources = [_parse_source(i, s) for i, s in enumerate(raw_sources)]
    names = [s.name for s in sources]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise ConfigError(f"sources.name이 중복됩니다: {dup}")

    judge = _section(JudgeConfig, root.get("judge"), "judge")
    if any(s.enabled and s.judge for s in sources) and not judge.interest.strip():
        raise ConfigError("judge: true인 소스가 있는데 judge.interest가 비어 있습니다.")

    images = _section(ImageConfig, root.get("images"), "images")
    if images.hero not in _HERO_MODES:
        raise ConfigError(f"images.hero는 {_HERO_MODES} 중 하나여야 합니다: {images.hero!r}")

    publisher = _section(PublisherConfig, root.get("publisher"), "publisher")
    if publisher.backend not in ("slack", "zulip"):
        raise ConfigError(f"publisher.backend는 'slack' 또는 'zulip'이어야 합니다: {publisher.backend!r}")
    zulip = _section(ZulipConfig, root.get("zulip"), "zulip")
    zulip = ZulipConfig(stream=zulip.stream.strip().lstrip("#"), topic=" ".join(zulip.topic.split()),
                        config_file=zulip.config_file)
    if publisher.backend == "zulip" and (not zulip.stream or not zulip.topic or len(zulip.topic) > 60):
        raise ConfigError("zulip.stream과 zulip.topic(60자 이하)이 필요합니다.")

    return Config(
        sources=sources,
        fetch=_section(FetchConfig, root.get("fetch"), "fetch"),
        judge=judge,
        report=_section(ReportConfig, root.get("report"), "report"),
        images=images,
        llm=_section(LLMConfig, root.get("llm"), "llm"),
        publisher=publisher,
        slack=_section(SlackConfig, root.get("slack"), "slack"),
        storage=_section(StorageConfig, root.get("storage"), "storage"),
        zulip=zulip,
        base_dir=os.path.dirname(os.path.abspath(path)),
    )
