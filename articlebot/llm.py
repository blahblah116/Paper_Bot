"""LLM 호출 — 요약 / relevance judge / 대표 그림 선택.

paperbot과 같은 LLM 서버·모델(qwen3.8:27b)을 OpenAI 호환 엔드포인트(/v1)로 호출한다.
서버는 llama-swap(127.0.0.1:8090) — 요청이 오면 llama-server(llama.cpp CUDA)를 띄우고 ttl 10분 뒤 내린다
(설정 ~/llama-swap/config.yaml).
- thinking은 chat_template_kwargs enable_thinking=false로 끈다 (서버 기본값도 끔).
- JSON 출력은 OpenAI 호환 response_format(json_schema)으로 서버가 강제한다(grammar).
- 컨텍스트 길이는 서버 실행 옵션(llama-server -c, 64K)으로 정해진다 — 본문 상한(llm.max_input_chars)이 그 안에 들어가게 둘 것.
"""
from __future__ import annotations

import json
import logging
import re
import time

from openai import OpenAI

from paperbot import llmserver

from .config import Config
from .models import Article, FetchedImage
from .prompts import (
    JUDGE_SYSTEM_PROMPT, PICK_SYSTEM_PROMPT, build_judge_user_prompt, build_pick_user_prompt,
    build_summary_system_prompt, build_summary_user_prompt,
)

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"<think>.*\Z", re.DOTALL | re.IGNORECASE)   # 잘린(미완) thinking
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

_JUDGE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "relevance_judge",
        "schema": {
            "type": "object",
            "properties": {
                "score": {"type": "integer", "minimum": 0, "maximum": 10},
                "reason": {"type": "string"},
            },
            "required": ["score", "reason"],
            "additionalProperties": False,
        },
    },
}

_PICK_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "figure_pick",
        "schema": {
            "type": "object",
            "properties": {"selected": {"type": "array", "items": {"type": "integer"}}},
            "required": ["selected"],
            "additionalProperties": False,
        },
    },
}


class LLMError(Exception):
    """재시도를 모두 소진한 뒤에도 LLM 호출에 실패한 경우."""


def make_client(cfg: Config) -> OpenAI:
    # llama-swap은 API 키를 검증하지 않지만(apiKeys 미설정) openai 클라이언트는 값이 필요하다.
    return OpenAI(base_url=cfg.llm.base_url, api_key="none", timeout=600.0)


def _clean_output(text: str) -> str:
    text = _THINK_RE.sub("", text or "")
    text = _THINK_OPEN_RE.sub("", text)
    return text.strip()


def chat(client: OpenAI, cfg: Config, messages: list[dict], *, temperature: float | None = None,
         max_tokens: int | None = None, response_format: dict | None = None, label: str = "llm") -> str:
    """지수 백오프 재시도(3회)를 포함한 chat completion. 실패 시 LLMError."""
    last_err: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            kwargs = {"response_format": response_format} if response_format else {}
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
            return content
        except Exception as e:  # noqa: BLE001 — 연결/응답 오류 모두 재시도
            last_err = e
            logger.warning("%s 호출 실패 (%d/%d): %s", label, attempt, _MAX_RETRIES, e)
            if attempt < _MAX_RETRIES:
                time.sleep(2 ** attempt)
    raise LLMError(f"{label} 최종 실패") from last_err


def unload_model(cfg: Config) -> bool:
    """실행 종료 시 모델을 내려 VRAM 반납 (llama-swap ttl 10분을 기다리지 않음). 실패는 무시.
    다른 요청(zoterobot DM 등)이 처리 중이면 끝날 때까지 기다린다 — paperbot/llmserver.py unload."""
    return llmserver.unload(llmserver.origin(cfg.llm.base_url), cfg.llm.model)


def _parse_json(text: str) -> dict | None:
    m = _JSON_RE.search(text or "")
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


# ---------------------------------------------------------------- 요약

def summarize(cfg: Config, client: OpenAI, article: Article, body: str) -> str:
    messages = [
        {"role": "system", "content": build_summary_system_prompt(
            cfg.llm.language, "latex" if cfg.publisher.backend == "zulip" else "plain")},   # Zulip만 $$LaTeX$$ 렌더링
        {"role": "user", "content": build_summary_user_prompt(article, body)},
    ]
    t0 = time.monotonic()
    content = chat(client, cfg, messages, label=f"요약({article.uid})")
    logger.info("요약 완료 %s: %d자 (%.0fs)", article.uid, len(content), time.monotonic() - t0)
    return content


# ---------------------------------------------------------------- judge

def judge(cfg: Config, client: OpenAI, article: Article, excerpt: str) -> tuple[float, str] | None:
    """0~10 점수와 근거. 호출·파싱이 끝내 실패하면 None (호출자는 다음 실행에 재채점)."""
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": build_judge_user_prompt(cfg.judge.interest, article, excerpt)},
    ]
    for attempt in (1, 2):
        try:
            out = chat(client, cfg, messages, temperature=0.0, max_tokens=512,
                       response_format=_JUDGE_FORMAT, label=f"judge({article.uid})")
        except LLMError:
            return None
        obj = _parse_json(out)
        try:
            score = float(obj["score"]) if obj else None
        except (KeyError, TypeError, ValueError):
            score = None
        if score is not None and 0 <= score <= 10:
            return score, str(obj.get("reason", ""))
        logger.warning("judge JSON 파싱 실패 (%d/2) %s: %.120s", attempt, article.uid, out)
        messages = messages + [
            {"role": "assistant", "content": out[:500]},
            {"role": "user", "content": '형식이 잘못됐다. 다른 텍스트 없이 정확히 '
                                        '{"score": <0-10 정수>, "reason": "<한 문장>"} JSON 하나만 다시 출력하라.'},
        ]
    return None


# ---------------------------------------------------------------- 대표 그림 선택

def _fallback_pick(images: list[FetchedImage], k: int) -> list[FetchedImage]:
    """LLM 선택 실패 시: 캡션 있는 그림 우선, 본문 순서 유지."""
    ranked = sorted(range(len(images)), key=lambda i: (not (images[i].ref.caption or images[i].ref.alt), i))
    keep = sorted(ranked[:k])
    return [images[i] for i in keep]


def pick_images(cfg: Config, client: OpenAI | None, article: Article,
                images: list[FetchedImage]) -> list[FetchedImage]:
    """그림이 max_images보다 많으면 LLM이 캡션을 보고 대표 그림을 고른다 (본문 순서 유지)."""
    k = cfg.images.max_images
    if len(images) <= k:
        return images
    if client is None:
        return _fallback_pick(images, k)
    messages = [
        {"role": "system", "content": PICK_SYSTEM_PROMPT},
        {"role": "user", "content": build_pick_user_prompt(article, images, k)},
    ]
    try:
        out = chat(client, cfg, messages, temperature=0.0, max_tokens=256,
                   response_format=_PICK_FORMAT, label=f"그림선택({article.uid})")
    except LLMError:
        return _fallback_pick(images, k)
    obj = _parse_json(out) or {}
    picked: list[int] = []
    for v in obj.get("selected") or []:
        try:
            i = int(v)
        except (TypeError, ValueError):
            continue
        if 1 <= i <= len(images) and i - 1 not in picked:
            picked.append(i - 1)
    if not picked:
        logger.warning("그림 선택 결과 없음/파싱 실패 — 휴리스틱으로 대체: %.120s", out)
        return _fallback_pick(images, k)
    picked = sorted(picked[:k])
    logger.info("대표 그림 선택 %s: %d개 중 %s", article.uid, len(images), [p + 1 for p in picked])
    return [images[i] for i in picked]
