"""LLM 호출 공용 헬퍼 + 논문 요약.

llama-swap(로컬 프록시, 기본 127.0.0.1:8090)의 OpenAI 호환 엔드포인트(/v1)를 openai 패키지로 호출한다.
llama-swap이 요청 시 llama-server(llama.cpp CUDA)를 띄운다 (관리 헬퍼: llmserver.py).

num_ctx — 중요:
컨텍스트 길이는 요청이 아니라 서버 실행 옵션(~/llama-swap/config.yaml의 llama-server -c)으로 정해진다.
첫 호출 성공 직후 llama-server /props로 실제 컨텍스트를 확인해
config보다 작으면 경고를 남긴다 (논문이 조용히 잘리는 것 방지).

judge(filter.py)와 요약이 이 모듈의 make_client/chat을 공유한다.
"""
from __future__ import annotations

import logging
import re
import time

from openai import OpenAI

from . import llmserver
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
    # llama-swap은 API 키를 검증하지 않지만(apiKeys 미설정) openai 클라이언트는 값이 필요하다.
    return OpenAI(base_url=cfg.llm.base_url, api_key="none", timeout=600.0)


def _clean_output(text: str) -> str:
    text = _THINK_RE.sub("", text or "")
    text = _THINK_OPEN_RE.sub("", text)  # 잘린(미완) thinking → 빈 응답 처리 → 재시도 유도
    return text.strip()


_ctx_verified = False
_llm_used = False   # 이 프로세스가 LLM을 한 번이라도 불렀나 — 실행 끝 언로드 판단용 (llm_used())


def _verify_num_ctx_once(cfg: Config) -> None:
    """서버에 실제 로드된 컨텍스트 길이를 확인, config보다 작으면 경고 (1회만)."""
    global _ctx_verified
    if _ctx_verified:
        return
    _ctx_verified = True
    actual = llmserver.server_ctx(llmserver.origin(cfg.llm.base_url), cfg.llm.model)
    if actual is None:
        logger.debug("서버 컨텍스트 확인 불가(무시)")
    elif actual < cfg.llm.num_ctx:
        logger.warning(
            "서버의 실제 컨텍스트(%d)가 config num_ctx(%d)보다 작습니다 — 긴 논문이 조용히 잘릴 수 있습니다. "
            "~/llama-swap/config.yaml의 llama-server -c를 %d 이상으로 올리세요.",
            actual, cfg.llm.num_ctx, cfg.llm.num_ctx,
        )
    else:
        logger.info("서버 컨텍스트 확인: %d (config num_ctx=%d)", actual, cfg.llm.num_ctx)


def chat(
    client: OpenAI,
    cfg: Config,
    messages: list[dict],
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    response_format: dict | None = None,
    label: str = "llm",
) -> str:
    """지수 백오프 재시도(3회)를 포함한 chat completion. 실패 시 LLMError.

    max_tokens 미지정 시 config의 max_output_tokens 사용 — 폭주 생성 방지.
    thinking은 항상 끈다(chat_template_kwargs enable_thinking=false — Qwen 채팅 템플릿 옵션, 서버 기본값도 끔)
    — 요약·judge 모두 답이 입력에 있어 추론 이득이 거의 없고, thinking이 max_tokens를 잡아먹어
    빈 응답이 나는 사고를 막는다.
    response_format은 OpenAI 호환 json_schema — llama-server가 문법(grammar)으로 출력 형식을 강제한다.
    """
    global _llm_used
    _llm_used = True
    last_err: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            kwargs = {}
            if response_format is not None:
                kwargs["response_format"] = response_format
            resp = client.chat.completions.create(
                model=cfg.llm.model,
                messages=messages,
                temperature=cfg.llm.temperature if temperature is None else temperature,
                max_tokens=cfg.llm.max_output_tokens if max_tokens is None else max_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                **kwargs,
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


def llm_used() -> bool:
    """이 프로세스에서 chat()(judge·요약)을 부른 적이 있나 — 없으면 모델을 올린 적도 없으니 언로드할 이유가 없다."""
    return _llm_used


def unload_model(cfg: Config) -> bool:
    """실행 종료 시 모델을 내려 VRAM을 반납 — llama-swap ttl(10분)을 기다리지 않는다.

    다른 봇(zoterobot DM 등)의 요청이 처리 중이면 끝날 때까지 기다렸다가 내린다(최대 10분, llmserver.unload).
    실패해도 치명적이지 않음 — ttl이 지나면 어차피 내려간다.
    """
    return llmserver.unload(llmserver.origin(cfg.llm.base_url), cfg.llm.model)


def summarize(cfg: Config, paper: Paper, extracted: ExtractResult, client: OpenAI | None = None) -> str:
    """논문 하나를 요약해 마크다운 문자열을 반환. 실패 시 LLMError."""
    client = client or make_client(cfg)
    math = "latex" if cfg.platform_for(paper.topic_name) == "zulip" else "plain"   # Zulip만 $$LaTeX$$ 렌더링
    messages = [
        {"role": "system", "content": build_summary_system_prompt(cfg.llm.language, math)},
        {"role": "user", "content": build_summary_user_prompt(paper, extracted.text, extracted.abstract_only)},
    ]
    content = chat(client, cfg, messages, label=f"요약({paper.uid})")
    logger.info("요약 완료 %s: %d자", paper.uid, len(content))
    return content
