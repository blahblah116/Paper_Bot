"""LLM 호출 공용 헬퍼 + 논문 요약.

llama-swap(로컬 프록시, 기본 127.0.0.1:8090)의 OpenAI 호환 엔드포인트(/v1)를 openai 패키지로 호출한다.
llama-swap이 요청 시 llama-server(llama.cpp CUDA)를 띄운다 (관리 헬퍼: llmserver.py).

num_ctx — 중요:
컨텍스트 길이는 요청이 아니라 서버 실행 옵션(~/llama-swap/config.yaml의 llama-server -c)으로 정해진다.
첫 호출 성공 직후 llama-server /props로 실제 컨텍스트를 확인해
config보다 작으면 경고를 남긴다 (논문이 조용히 잘리는 것 방지).
요약 입력은 호출 전에 llama-server /tokenize로 token 수를 세어 실제 컨텍스트 − max_output_tokens에 맞춘다
(_fit_input — 셀 수 없으면 max_input_chars 글자 수 절단).

judge(filter.py)와 요약이 이 모듈의 make_client/chat을 공유한다.
"""
from __future__ import annotations

import logging
import re
import time

from openai import OpenAI

from . import llmserver
from .config import Config
from .extractor import ExtractResult, truncate_tokens
from .models import Paper
from .prompts import build_summary_system_prompt, build_summary_user_prompt

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_TEMPLATE_SLACK_TOKENS = 256   # 요약 입력 예산에서 더 떼는 몫 — chat template의 역할 표시·특수 토큰(system·user를 이어 셀 때 빠짐)
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


def _fit_input(cfg: Config, paper: Paper, extracted: ExtractResult, system: str) -> str:
    """요약 입력 본문 — full_text(없으면 text)를 서버 컨텍스트에 token 기준으로 맞춘다.

    예산 = 서버의 실제 컨텍스트(/props, 모르면 num_ctx) − max_output_tokens − chat template 여유.
    system + user 프롬프트 전체의 token 수가 예산 안이면 그대로, 넘으면 넘는 만큼만 중간을 자른다(truncate_tokens).
    모델이 아직 안 올라와 있는 등 token을 셀 수 없으면 max_input_chars로 자른 extracted.text를 쓴다.
    """
    origin = llmserver.origin(cfg.llm.base_url)
    ctx = llmserver.server_ctx(origin, cfg.llm.model) or cfg.llm.num_ctx
    budget = ctx - cfg.llm.max_output_tokens - _TEMPLATE_SLACK_TOKENS
    full = extracted.full_text or extracted.text

    def count(body: str) -> int | None:
        user = build_summary_user_prompt(paper, body, extracted.abstract_only)
        return llmserver.count_tokens(origin, cfg.llm.model, f"{system}\n{user}")

    fit = truncate_tokens(full, budget, count)
    if fit is None:
        logger.info("요약 입력 %s: token 수로 맞추지 못함(모델이 안 올라와 있음 등) — 글자 수로 자른 %d자 (전체 %d자, max_input_chars %d)",
                    paper.uid, len(extracted.text), len(full), cfg.llm.max_input_chars)
        return extracted.text
    body, total, kept = fit
    if total <= budget:
        logger.info("요약 입력 %s: %d token (예산 %d) — 자르지 않음, %d자", paper.uid, total, budget, len(body))
    else:
        logger.info("요약 입력 %s: 전체 %d token 중 %d token 남김 (예산 %d) — %d자 중 %d자",
                    paper.uid, total, kept, budget, len(full), len(body))
    return body


def summarize(cfg: Config, paper: Paper, extracted: ExtractResult, client: OpenAI | None = None) -> str:
    """논문 하나를 요약해 마크다운 문자열을 반환. 실패 시 LLMError."""
    client = client or make_client(cfg)
    math = "latex" if cfg.platform_for(paper.topic_name) == "zulip" else "plain"   # Zulip만 $$LaTeX$$ 렌더링
    system = build_summary_system_prompt(cfg.llm.language, math)
    body = _fit_input(cfg, paper, extracted, system)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": build_summary_user_prompt(paper, body, extracted.abstract_only)},
    ]
    content = chat(client, cfg, messages, label=f"요약({paper.uid})")
    logger.info("요약 완료 %s: %d자", paper.uid, len(content))
    return content
