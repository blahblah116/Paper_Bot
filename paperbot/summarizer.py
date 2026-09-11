"""LLM 호출 공용 헬퍼 + 논문 요약.

Ollama의 OpenAI 호환 엔드포인트(/v1)를 openai 패키지로 호출한다.

num_ctx — 중요:
매 요청에 extra_body로 num_ctx를 명시하지만, 실측 결과 Ollama 0.34의
OpenAI 호환 엔드포인트는 이를 무시한다 (향후 버전 대비로 유지).
실효 제어는 서버 환경변수 OLLAMA_CONTEXT_LENGTH이므로,
첫 호출 성공 직후 /api/ps로 실제 로드된 컨텍스트를 확인해
config보다 작으면 경고를 남긴다 (논문이 조용히 잘리는 것 방지).

judge(filter.py)와 요약이 이 모듈의 make_client/chat을 공유한다.
"""
from __future__ import annotations

import logging
import re
import time

from openai import OpenAI

from .config import Config
from .extractor import ExtractResult
from .models import Paper
from .prompts import build_summary_system_prompt, build_summary_user_prompt

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
# 출력이 max_tokens에 잘려 </think> 없이 끝난 경우 — 미완 thinking 전체를 제거.
_THINK_OPEN_RE = re.compile(r"<think>.*\Z", re.DOTALL | re.IGNORECASE)


class LLMError(Exception):
    """재시도를 모두 소진한 뒤에도 LLM 호출에 실패한 경우."""


def make_client(cfg: Config) -> OpenAI:
    # Ollama는 API 키를 검증하지 않지만 openai 클라이언트는 값이 필요하다.
    return OpenAI(base_url=cfg.llm.base_url, api_key="ollama", timeout=600.0)


def _clean_output(text: str) -> str:
    text = _THINK_RE.sub("", text or "")
    text = _THINK_OPEN_RE.sub("", text)  # 잘린(미완) thinking → 빈 응답 처리 → 재시도 유도
    return text.strip()


_ctx_verified = False


def _verify_num_ctx_once(cfg: Config) -> None:
    """서버에 실제 로드된 컨텍스트 길이를 확인, config보다 작으면 경고 (1회만)."""
    global _ctx_verified
    if _ctx_verified:
        return
    _ctx_verified = True
    try:
        import requests

        origin = cfg.llm.base_url.rstrip("/").removesuffix("/v1")
        resp = requests.get(f"{origin}/api/ps", timeout=5)
        for m in resp.json().get("models", []):
            if m.get("name", "").startswith(cfg.llm.model.split(":")[0]):
                actual = m.get("context_length") or 0
                if 0 < actual < cfg.llm.num_ctx:
                    logger.warning(
                        "서버의 실제 컨텍스트(%d)가 config num_ctx(%d)보다 작습니다 — "
                        "긴 논문이 조용히 잘릴 수 있습니다. Ollama를 "
                        "OLLAMA_CONTEXT_LENGTH=%d 로 재시작하세요.",
                        actual, cfg.llm.num_ctx, cfg.llm.num_ctx,
                    )
                else:
                    logger.info("서버 컨텍스트 확인: %d (config num_ctx=%d)", actual, cfg.llm.num_ctx)
                return
    except Exception as e:  # noqa: BLE001 — 확인 실패는 치명적이지 않음
        logger.debug("컨텍스트 확인 실패(무시): %s", e)


def chat(
    client: OpenAI,
    cfg: Config,
    messages: list[dict],
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    label: str = "llm",
) -> str:
    """지수 백오프 재시도(3회)를 포함한 chat completion. 실패 시 LLMError.

    max_tokens 미지정 시 config의 max_output_tokens 사용 — 폭주 생성 방지.
    (Ollama OpenAI 호환은 max_tokens를 num_predict로 매핑함)
    """
    last_err: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=cfg.llm.model,
                messages=messages,
                temperature=cfg.llm.temperature if temperature is None else temperature,
                max_tokens=cfg.llm.max_output_tokens if max_tokens is None else max_tokens,
                # Ollama 전용 옵션 — 컨텍스트 길이를 요청 단위로 명시.
                extra_body={"num_ctx": cfg.llm.num_ctx},
            )
            content = _clean_output(resp.choices[0].message.content)
            if not content:
                raise ValueError("LLM이 빈 응답을 반환")
            _verify_num_ctx_once(cfg)
            return content
        except Exception as e:  # noqa: BLE001 — 연결/응답 오류 모두 재시도
            last_err = e
            wait = 2 ** attempt  # 2s, 4s, 8s
            logger.warning("%s 호출 실패 (%d/%d): %s", label, attempt, _MAX_RETRIES, e)
            if attempt < _MAX_RETRIES:
                time.sleep(wait)
    raise LLMError(f"{label} 최종 실패") from last_err


def unload_model(cfg: Config) -> bool:
    """Ollama에 로드된 모델을 즉시 언로드해 VRAM을 반납 (실행 종료 시 호출).

    기본 keep_alive(5분)를 기다리지 않고 GPU 점유를 끝낸다.
    실패해도 치명적이지 않음 — keep_alive가 지나면 어차피 내려간다.
    """
    try:
        import requests

        origin = cfg.llm.base_url.rstrip("/").removesuffix("/v1")
        resp = requests.post(
            f"{origin}/api/chat",
            json={"model": cfg.llm.model, "messages": [], "keep_alive": 0},
            timeout=10,
        )
        ok = resp.status_code == 200
        if ok:
            logger.info("모델 언로드 완료 — GPU 반납 (%s)", cfg.llm.model)
        return ok
    except Exception as e:  # noqa: BLE001
        logger.debug("모델 언로드 실패(무시): %s", e)
        return False


def summarize(cfg: Config, paper: Paper, extracted: ExtractResult, client: OpenAI | None = None) -> str:
    """논문 하나를 요약해 마크다운 문자열을 반환. 실패 시 LLMError."""
    client = client or make_client(cfg)
    messages = [
        {"role": "system", "content": build_summary_system_prompt(cfg.llm.language)},
        {"role": "user", "content": build_summary_user_prompt(paper, extracted.text, extracted.abstract_only)},
    ]
    content = chat(client, cfg, messages, label=f"요약({paper.uid})")
    logger.info("요약 완료 %s: %d자", paper.uid, len(content))
    return content
