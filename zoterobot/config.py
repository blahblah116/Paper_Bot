"""zoterobot/config.yaml 로드·검증.

paperbot(매일 요약 봇)과 설정을 분리한다. 공유하는 것은 Slack 앱·토큰(~/.paperbot_env),
Ollama 서버·모델, NAS 데이터 폴더뿐이다.
인증은 환경변수: ZOTEROBOT_BOT_TOKEN(xoxb-), ZOTEROBOT_APP_TOKEN(xapp-) — 둘 다 zoterobot Slack 앱의 것 (없으면 SLACK_*), ZOTERO_API_KEY, ZOTERO_USER_ID(선택), S2_API_KEY(선택).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import yaml


class ConfigError(Exception):
    pass


@dataclass(frozen=True)
class LibraryConfig:
    """스레드 '@zoterobot <컬렉션>' 저장 — PDF는 Drive <drive_remote>/<컬렉션>/에, Zotero엔 링크 첨부."""
    drive_remote: str = ""            # rclone "리모트:경로" — Attanger File Storage Directory
    link_base: str = "attachments:"   # Zotero '링크 첨부 파일 기본 디렉토리' 기준 상대경로 접두어
    rclone_bin: str = "rclone"
    tag: str = "paperbot"             # Zotero 항목 태그 (빈 문자열이면 생략)
    filename_lang: str = "en"         # 파일명 접속어 "en"(A and B / A et al.) | "ko"(A 및 B / A 등)
    unpaywall_email: str = ""         # PDF 폴백(Unpaywall) 식별 이메일 — 비우면 미사용


@dataclass(frozen=True)
class ChatConfig:
    """DM 챗봇 — qwen이 질문 유형(search·question·comparing)을 판단해 도구를 골라 쓰는 에이전트. 임베딩 모델 없음."""
    enabled: bool = True
    llm_url: str = "http://localhost:11434"   # Ollama 네이티브 API (keep_alive·think·tools 지원)
    model: str = "qwen3.8:27b"                # paperbot llm.model과 같게 — 다르면 VRAM에 두 모델이 올라간다
    keep_alive: str = "10m"                   # 마지막 DM 이후 모델 유지 시간
    temperature: float = 0.5
    max_output_tokens: int = 2048
    max_tool_steps: int = 5                   # 한 질문에 허용할 도구 호출 라운드
    ctx_tokens: int = 65536                   # Ollama 서버 컨텍스트(OLLAMA_CONTEXT_LENGTH) — 첫 응답 뒤 /api/ps로 실제 값 확인
    output_reserve_tokens: int = 3072         # 풀 계산 때 답변용으로 떼어두는 토큰 (관측 최대 답변 ~2,400의 1.3배; 생성 상한은 max_output_tokens)
    safety_tokens: int = 1024                 # 고정 여유 토큰 (+ 미측정 추가분 추정의 10%는 budget.py가 얹는다)
    history_max_tokens: int = 8000            # 대화 기억(history_turns 쌍)이 넘으면 오래된 쌍부터 뺀다 (긴 비교 답변 4쌍 분량)
    chars_per_token: float = 3.4              # 영어 본문 글자/토큰 초기값 — 실측 보정값이 캐시 meta에 저장되면 그쪽을 쓴다
    summary_enabled: bool = True              # summarize_paper 도구 — 캐시 본문(없으면 Drive PDF)을 paperbot 요약기로 전문 요약
    paperbot_config: str = "../config.yaml"   # 요약기가 쓸 paperbot config.yaml (llm.max_input_chars·프롬프트·모델)
    summaries_per_question: int = 2           # 질문 하나에서 새로 만들 전문 요약 수(편당 2~3분) — 저장된 요약은 안 셈
    history_turns: int = 4                    # DM별로 기억할 최근 질문·답 쌍 수
    vram_gpus: str = "1,2"                    # 모델이 올라갈 GPU (Ollama GGML_VK_VISIBLE_DEVICES와 같게)
    vram_min_mib: int = 13312                 # 장당 최소 여유 — 모델이 이미 올라가 있으면 체크 생략
    catalog_max: int = 0                      # 0이면 목록을 프롬프트에 넣지 않고 항상 search_papers로 찾는다; N이면 N편 이하일 때 전체 목록 포함
    text_db: str = "./data/library_text.db"   # 라이브러리 본문 텍스트 캐시 (sqlite)
    chatlog_dir: str = "./data/chatlog"       # 대화 로그 (날짜별 jsonl)
    sync_interval_min: int = 10               # DM 시 라이브러리 변경 확인 최소 간격
    drive_scan: bool = True                   # Drive(library.drive_remote)에만 있고 Zotero에 없는 PDF도 검색 대상에


@dataclass(frozen=True)
class Config:
    allowed_users: list[str]
    library: LibraryConfig
    chat: ChatConfig
    base_dir: str = "."

    def resolve(self, path: str) -> str:
        path = os.path.expanduser(path)
        if os.path.isabs(path):
            return path
        return os.path.normpath(os.path.join(self.base_dir, path))


def _as_dict(value: Any, ctx: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"'{ctx}'는 매핑이어야 합니다. (현재: {type(value).__name__})")
    return value


def load_config(path: str) -> Config:
    if not os.path.exists(path):
        raise ConfigError(f"설정 파일을 찾을 수 없습니다: {path} (config.example.yaml을 복사해 만드세요)")
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ConfigError("config.yaml 최상위는 매핑이어야 합니다.")

    allowed = raw.get("allowed_users") or []
    if not isinstance(allowed, list):
        raise ConfigError("allowed_users는 Slack 사용자 ID 리스트여야 합니다.")

    lib_d = _as_dict(raw.get("library"), "library")
    library = LibraryConfig(
        drive_remote=str(lib_d.get("drive_remote", "")).strip().rstrip("/"),
        link_base=str(lib_d.get("link_base", "attachments:")).strip(),
        rclone_bin=os.path.expanduser(str(lib_d.get("rclone_bin", "rclone")).strip()),
        tag=str(lib_d.get("tag", "paperbot") or "").strip(),
        filename_lang=str(lib_d.get("filename_lang", "en")).strip().lower(),
        unpaywall_email=str(lib_d.get("unpaywall_email", "") or "").strip(),
    )
    if library.drive_remote and ":" not in library.drive_remote:
        raise ConfigError(f"library.drive_remote는 rclone '리모트:경로' 형식이어야 합니다: {library.drive_remote!r}")
    if not library.link_base:
        raise ConfigError("library.link_base가 비어 있습니다 (기본값 'attachments:').")
    if library.filename_lang not in ("en", "ko"):
        raise ConfigError(f"library.filename_lang은 'en' 또는 'ko'여야 합니다: {library.filename_lang}")

    chat_d = _as_dict(raw.get("chat"), "chat")
    d = ChatConfig()
    chat = ChatConfig(
        enabled=bool(chat_d.get("enabled", d.enabled)),
        llm_url=str(chat_d.get("llm_url", d.llm_url)).rstrip("/").removesuffix("/v1"),
        model=str(chat_d.get("model", d.model)),
        keep_alive=str(chat_d.get("keep_alive", d.keep_alive)),
        temperature=float(chat_d.get("temperature", d.temperature)),
        max_output_tokens=int(chat_d.get("max_output_tokens", d.max_output_tokens)),
        max_tool_steps=int(chat_d.get("max_tool_steps", d.max_tool_steps)),
        ctx_tokens=int(chat_d.get("ctx_tokens", d.ctx_tokens)),
        output_reserve_tokens=int(chat_d.get("output_reserve_tokens", d.output_reserve_tokens)),
        safety_tokens=int(chat_d.get("safety_tokens", d.safety_tokens)),
        history_max_tokens=int(chat_d.get("history_max_tokens", d.history_max_tokens)),
        chars_per_token=float(chat_d.get("chars_per_token", d.chars_per_token)),
        summary_enabled=bool(chat_d.get("summary_enabled", d.summary_enabled)),
        paperbot_config=str(chat_d.get("paperbot_config", d.paperbot_config)),
        summaries_per_question=int(chat_d.get("summaries_per_question", d.summaries_per_question)),
        history_turns=int(chat_d.get("history_turns", d.history_turns)),
        vram_gpus=str(chat_d.get("vram_gpus", d.vram_gpus)),
        vram_min_mib=int(chat_d.get("vram_min_mib", d.vram_min_mib)),
        catalog_max=int(chat_d.get("catalog_max", d.catalog_max)),
        text_db=str(chat_d.get("text_db", d.text_db)),
        chatlog_dir=str(chat_d.get("chatlog_dir", d.chatlog_dir)),
        sync_interval_min=int(chat_d.get("sync_interval_min", d.sync_interval_min)),
        drive_scan=bool(chat_d.get("drive_scan", d.drive_scan)),
    )
    if chat.max_tool_steps < 1 or chat.history_turns < 0 or chat.max_output_tokens < 256:
        raise ConfigError("chat.max_tool_steps>=1, history_turns>=0, max_output_tokens>=256 이어야 합니다.")
    if chat.output_reserve_tokens < 256 or chat.ctx_tokens < chat.output_reserve_tokens + chat.safety_tokens + 8192:
        raise ConfigError("chat.output_reserve_tokens>=256, ctx_tokens >= output_reserve_tokens + safety_tokens + 8192 이어야 합니다.")
    if chat.history_max_tokens < 0 or not (2.0 <= chat.chars_per_token <= 6.0):
        raise ConfigError("chat.history_max_tokens>=0, chars_per_token은 2~6 이어야 합니다.")
    if chat.summaries_per_question < 0:
        raise ConfigError("chat.summaries_per_question>=0 이어야 합니다.")

    return Config(
        allowed_users=[str(u).strip() for u in allowed if str(u).strip()],
        library=library, chat=chat,
        base_dir=os.path.dirname(os.path.abspath(path)),
    )
