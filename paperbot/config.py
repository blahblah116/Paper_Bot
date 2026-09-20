"""config.yaml 로드 및 검증 (v2).

핵심 요구: 주제·학회·연도·인용 수 등 모든 필터를 코드 수정 없이
config.yaml에서 조절할 수 있어야 한다. 누락/오타는 여기서 명확히 실패시킨다.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import yaml


class ConfigError(Exception):
    """설정 파일이 유효하지 않을 때 발생."""


# ---------------------------------------------------------------- topic/소스

@dataclass(frozen=True)
class S2SourceConfig:
    query: str
    venues: list[str] = field(default_factory=list)   # 빈 리스트 = venue 필터 없음
    year: str | None = None                           # S2 문법: "2024", "2024-2026", "2024-"
    min_citations: int = 0
    sort: str = "recency"                             # "recency"(최신순) | "citations"(인용순)


@dataclass(frozen=True)
class ArxivSourceConfig:
    enabled: bool = True
    query: str = ""


@dataclass(frozen=True)
class Topic:
    name: str
    interest: str                          # relevance 필터의 기준이 되는 자연어 서술
    s2: S2SourceConfig | None = None       # None이면 S2 소스 스킵
    arxiv: ArxivSourceConfig | None = None # None 또는 enabled=False면 arXiv 스킵
    channel: str | None = None             # 이 topic 전용 발행 대상(Slack 채널/Zulip 스트림).
                                           # None이면 backend의 기본값(slack.channel/zulip.stream) 사용.


# ---------------------------------------------------------------- 섹션들

@dataclass(frozen=True)
class FetchConfig:
    max_results_per_query: int = 50
    arxiv_lookback_days: int = 3
    s2_lookback_days: int = 180    # 0 = lookback 비활성 (topic별 year 필터에 위임)
    s2_max_pages_per_run: int = 3  # 인용순 topic: 실행당 넘길 최대 bulk 페이지 수(1페이지=1000건)
    unpaywall_email: str = ""      # Unpaywall PDF 구조용 식별 이메일 (비우면 Unpaywall 미사용)


@dataclass(frozen=True)
class QuotaConfig:
    """실행(=하루)당 topic(채널)별 발행 상한. 0 = 무제한.

    필터를 통과한 논문 중 소스별로 상한만큼만 발행한다. 초과분은 DB에 pending으로 적재되어
    다음 실행에서 오래된 순(FIFO)으로 먼저 쿼터를 채운다.
    pending_max_age_days: 이보다 오래 대기한 pending은 폐기(expired). 0 = 무제한.
    """
    s2_per_topic: int = 0
    arxiv_per_topic: int = 0
    pending_max_age_days: int = 30


@dataclass(frozen=True)
class FilterConfig:
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"          # GPU는 요약 모델 전용으로 남긴다
    embedding_threshold: float = 0.35
    llm_judge: bool = True
    judge_threshold: float = 6.0


@dataclass(frozen=True)
class LLMConfig:
    base_url: str = "http://localhost:11434/v1"
    model: str = "qwen3.5:9b"
    num_ctx: int = 32768                   # Ollama 기본값에 맡기지 않고 명시
    max_input_chars: int = 30000
    max_output_tokens: int = 8192          # 요약 출력 상한 (thinking 토큰 포함 — 폭주 방지)
    temperature: float = 0.3
    language: str = "ko"


@dataclass(frozen=True)
class PublisherConfig:
    backend: str = "zulip"       # "zulip" | "slack"
    dry_run: bool = False        # true면 전송 대신 stdout 출력


@dataclass(frozen=True)
class ZulipConfig:
    config_file: str = "./zuliprc"
    stream: str = "papers"


@dataclass(frozen=True)
class SlackConfig:
    channel: str = "#papers"     # 채널명("#papers") 또는 채널 ID("C0123...")
    attach_figure: bool = False  # 대표 figure PNG를 스레드에 첨부 (files:write 스코프 필요)
    max_figures: int = 3         # 논문당 첨부할 최대 figure 수 (앞 8페이지에서 선별)
    # 봇 토큰은 config에 넣지 않고 환경변수 SLACK_BOT_TOKEN으로 주입한다.


@dataclass(frozen=True)
class StorageConfig:
    db_path: str = "./data/papers.db"
    pdf_dir: str = "./data/pdfs"
    keep_pdfs: bool = False


@dataclass(frozen=True)
class Config:
    topics: list[Topic]
    fetch: FetchConfig
    quota: QuotaConfig
    filter: FilterConfig
    llm: LLMConfig
    publisher: PublisherConfig
    zulip: ZulipConfig
    slack: SlackConfig
    storage: StorageConfig
    base_dir: str = field(default=".")     # config.yaml 위치 — 상대 경로 기준점

    def resolve(self, path: str) -> str:
        """config 내 상대 경로를 config.yaml 기준 절대 경로로 변환."""
        if os.path.isabs(path):
            return path
        return os.path.normpath(os.path.join(self.base_dir, path))


# ---------------------------------------------------------------- 파서

def _require(d: dict[str, Any], key: str, ctx: str) -> Any:
    if key not in d:
        raise ConfigError(f"'{ctx}'에 필수 키 '{key}'가 없습니다.")
    return d[key]


def _as_dict(value: Any, ctx: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"'{ctx}'는 매핑이어야 합니다. (현재: {type(value).__name__})")
    return value


def _parse_topic(i: int, raw: Any) -> Topic:
    ctx = f"topics[{i}]"
    t = _as_dict(raw, ctx)
    name = str(_require(t, "name", ctx)).strip()
    interest = str(_require(t, "interest", ctx)).strip()
    if not name:
        raise ConfigError(f"{ctx}.name이 비어 있습니다.")
    if not interest:
        raise ConfigError(f"{ctx}.interest가 비어 있습니다 — relevance 필터의 기준이므로 필수.")

    sources = _as_dict(t.get("sources"), f"{ctx}.sources")

    s2: S2SourceConfig | None = None
    if "semantic_scholar" in sources and sources["semantic_scholar"] is not None:
        s2_d = _as_dict(sources["semantic_scholar"], f"{ctx}.sources.semantic_scholar")
        venues = s2_d.get("venues", [])
        if venues is None:
            venues = []
        if not isinstance(venues, list):
            raise ConfigError(f"{ctx}.sources.semantic_scholar.venues는 리스트여야 합니다.")
        year = s2_d.get("year")
        sort = str(s2_d.get("sort", "recency")).strip().lower()
        if sort not in ("recency", "citations"):
            raise ConfigError(
                f"{ctx}.sources.semantic_scholar.sort는 'recency' 또는 'citations'여야 합니다: {sort}"
            )
        s2 = S2SourceConfig(
            query=str(_require(s2_d, "query", f"{ctx}.sources.semantic_scholar")).strip(),
            venues=[str(v) for v in venues],
            year=str(year) if year is not None else None,
            min_citations=int(s2_d.get("min_citations", 0)),
            sort=sort,
        )

    arxiv: ArxivSourceConfig | None = None
    if "arxiv" in sources and sources["arxiv"] is not None:
        ax_d = _as_dict(sources["arxiv"], f"{ctx}.sources.arxiv")
        enabled = bool(ax_d.get("enabled", True))
        query = str(ax_d.get("query", "")).strip()
        if enabled and not query:
            raise ConfigError(f"{ctx}.sources.arxiv: enabled인데 query가 없습니다.")
        arxiv = ArxivSourceConfig(enabled=enabled, query=query)

    if s2 is None and (arxiv is None or not arxiv.enabled):
        raise ConfigError(f"{ctx}: 활성화된 소스가 하나도 없습니다.")

    channel = t.get("channel")
    channel = str(channel).strip() if channel else None

    return Topic(name=name, interest=interest, s2=s2, arxiv=arxiv, channel=channel)


def load_config(path: str) -> Config:
    """config.yaml을 로드하고 검증된 Config 객체를 반환."""
    if not os.path.exists(path):
        raise ConfigError(f"설정 파일을 찾을 수 없습니다: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ConfigError("config.yaml 최상위는 매핑이어야 합니다.")

    base_dir = os.path.dirname(os.path.abspath(path))

    # --- topics ---
    raw_topics = _require(raw, "topics", "config")
    if not isinstance(raw_topics, list) or not raw_topics:
        raise ConfigError("'topics'는 비어 있지 않은 리스트여야 합니다.")
    topics = [_parse_topic(i, t) for i, t in enumerate(raw_topics)]
    names = [t.name for t in topics]
    if len(names) != len(set(names)):
        raise ConfigError(f"중복된 topic name이 있습니다: {names}")

    # --- fetch ---
    fetch_d = _as_dict(raw.get("fetch"), "fetch")
    fetch = FetchConfig(
        max_results_per_query=int(fetch_d.get("max_results_per_query", 50)),
        arxiv_lookback_days=int(fetch_d.get("arxiv_lookback_days", 3)),
        s2_lookback_days=int(fetch_d.get("s2_lookback_days", 180)),
        s2_max_pages_per_run=int(fetch_d.get("s2_max_pages_per_run", 3)),
        unpaywall_email=str(fetch_d.get("unpaywall_email", "")).strip(),
    )
    if fetch.max_results_per_query <= 0:
        raise ConfigError("fetch.max_results_per_query는 1 이상이어야 합니다.")
    if fetch.arxiv_lookback_days <= 0:
        # arXiv는 year 같은 서버측 시간 필터가 없어 lookback이 유일한 시간 제어.
        raise ConfigError("fetch.arxiv_lookback_days는 1 이상이어야 합니다.")
    if fetch.s2_lookback_days < 0:
        raise ConfigError("fetch.s2_lookback_days는 0(비활성 — year에 위임) 이상이어야 합니다.")
    if fetch.s2_max_pages_per_run <= 0:
        raise ConfigError("fetch.s2_max_pages_per_run은 1 이상이어야 합니다.")

    # --- quota ---
    quota_d = _as_dict(raw.get("quota"), "quota")
    quota = QuotaConfig(
        s2_per_topic=int(quota_d.get("s2_per_topic", 0)),
        arxiv_per_topic=int(quota_d.get("arxiv_per_topic", 0)),
        pending_max_age_days=int(quota_d.get("pending_max_age_days", 30)),
    )
    if quota.s2_per_topic < 0 or quota.arxiv_per_topic < 0 or quota.pending_max_age_days < 0:
        raise ConfigError("quota 값은 0(무제한) 이상이어야 합니다.")

    # --- filter ---
    filter_d = _as_dict(raw.get("filter"), "filter")
    flt = FilterConfig(
        embedding_model=str(filter_d.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")),
        embedding_device=str(filter_d.get("embedding_device", "cpu")),
        embedding_threshold=float(filter_d.get("embedding_threshold", 0.35)),
        llm_judge=bool(filter_d.get("llm_judge", True)),
        judge_threshold=float(filter_d.get("judge_threshold", 6.0)),
    )
    if not (0.0 <= flt.embedding_threshold <= 1.0):
        raise ConfigError("filter.embedding_threshold는 0~1 사이여야 합니다.")
    if not (0.0 <= flt.judge_threshold <= 10.0):
        raise ConfigError("filter.judge_threshold는 0~10 사이여야 합니다.")

    # --- llm ---
    llm_d = _as_dict(raw.get("llm"), "llm")
    llm = LLMConfig(
        base_url=str(llm_d.get("base_url", "http://localhost:11434/v1")),
        model=str(llm_d.get("model", "qwen3.5:9b")),
        num_ctx=int(llm_d.get("num_ctx", 32768)),
        max_input_chars=int(llm_d.get("max_input_chars", 30000)),
        max_output_tokens=int(llm_d.get("max_output_tokens", 8192)),
        temperature=float(llm_d.get("temperature", 0.3)),
        language=str(llm_d.get("language", "ko")),
    )
    if not (256 <= llm.max_output_tokens < llm.num_ctx):
        raise ConfigError("llm.max_output_tokens는 256 이상, num_ctx 미만이어야 합니다.")
    if llm.max_input_chars < 1000:
        raise ConfigError("llm.max_input_chars가 너무 작습니다 (>=1000 권장).")
    if llm.num_ctx < 4096:
        raise ConfigError("llm.num_ctx가 너무 작습니다 (>=4096 권장).")
    # 대략적 토큰 환산(영문 ~4자/토큰)으로 입력이 컨텍스트를 넘치지 않는지 경고 수준 검증.
    if llm.max_input_chars / 3 > llm.num_ctx:
        raise ConfigError(
            f"llm.max_input_chars({llm.max_input_chars})가 num_ctx({llm.num_ctx}) 대비 "
            "너무 큽니다 — 프롬프트가 조용히 잘릴 수 있습니다."
        )

    # --- publisher / zulip / slack / storage ---
    zulip_d = _as_dict(raw.get("zulip"), "zulip")
    zulip = ZulipConfig(
        config_file=str(zulip_d.get("config_file", "./zuliprc")),
        stream=str(zulip_d.get("stream", "papers")),
    )

    slack_d = _as_dict(raw.get("slack"), "slack")
    slack = SlackConfig(
        channel=str(slack_d.get("channel", "#papers")),
        attach_figure=bool(slack_d.get("attach_figure", False)),
        max_figures=int(slack_d.get("max_figures", 3)),
    )
    if slack.max_figures < 0:
        raise ConfigError("slack.max_figures는 0 이상이어야 합니다.")

    pub_d = _as_dict(raw.get("publisher"), "publisher")
    publisher = PublisherConfig(
        backend=str(pub_d.get("backend", "zulip")).lower(),
        # 하위 호환: 구버전 config의 zulip.dry_run도 인정.
        dry_run=bool(pub_d.get("dry_run", zulip_d.get("dry_run", False))),
    )
    if publisher.backend not in ("zulip", "slack"):
        raise ConfigError(f"publisher.backend는 'zulip' 또는 'slack'이어야 합니다: {publisher.backend}")

    storage_d = _as_dict(raw.get("storage"), "storage")
    storage = StorageConfig(
        db_path=str(storage_d.get("db_path", "./data/papers.db")),
        pdf_dir=str(storage_d.get("pdf_dir", "./data/pdfs")),
        keep_pdfs=bool(storage_d.get("keep_pdfs", False)),
    )

    return Config(
        topics=topics, fetch=fetch, quota=quota, filter=flt, llm=llm,
        publisher=publisher, zulip=zulip, slack=slack,
        storage=storage, base_dir=base_dir,
    )
