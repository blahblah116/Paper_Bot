"""DM chat agent: qwen classifies the question (search / question / comparing), picks the tool and answers.
DM 챗봇 에이전트 — qwen이 질문 유형(search·question·comparing)을 판단해 도구를 골라 쓰고 답한다.

- search_papers: 주제 검색 → 편당 초록·발췌·저장된 요약 앞부분 → 모델이 3~5줄 요약
- read_paper: 특정 논문 질문 → 본문 발췌(focus) → 본문 근거로만 답 (id 대신 제목도 받음)
- compare_papers: 여러 논문 → 편당 자료(요약+발췌)를 남은 예산을 나눠 배분 → 공통점·차이점
- summarize_paper(summary.py): 캐시 원문 전체를 paperbot 요약기로 요약(편당 2~3분) — 질문당 summaries_per_question회,
  저장된 요약은 횟수에 세지 않는다.

- llama-swap의 OpenAI 호환 /v1/chat/completions SSE 스트리밍 (tools, enable_thinking=false). 임베딩 모델 없이 도구 호출로
  근거를 찾는다. 모델 유지 시간은 llama-swap ttl(10분). 스트림이 끝 표시 없이 끊기면(언로드·재시작) 한 번 다시 요청한다.
  도구 호출 기록은 OpenAI 형식: 인자는 JSON 문자열, tool_call_id로 결과 연결.
- 도구 라운드는 max_tool_steps까지, 마지막 라운드는 도구 없이 답하게 강제한다.
- 입력 길이는 글자 상한이 아니라 토큰 예산(budget.py)으로 제어한다:
  풀 = ctx − 출력 예약(output_reserve_tokens) − 안전 여유 − 지금까지 메시지. read_paper/compare_papers는 남은 풀만큼
  본문을 받고(본문이 풀에 들어가면 전문, 아니면 발췌/절단 + 안내문), 풀이 바닥 읽기 분량(_READ_FLOOR) 아래로 내려가면
  다음 라운드는 도구 없이 답하게 한다. 라운드마다 서버가 잰 실제 입력 토큰(usage.prompt_tokens)으로 추정 비율을 보정하고
  (보정값은 캐시 meta 'chars_per_token'에 저장해 재시작 후에도 이어쓴다), 최종 생성 토큰도 남은 컨텍스트에 맞춘다.
  대화 기억은 history_max_tokens를 넘으면 오래된 쌍부터 뺀다.
- 답에 제목이 언급된 논문도 근거(papers)로 표시한다 — 검색 결과만 보고 답한 경우.
"""
from __future__ import annotations

import json
import logging
import math
import re
import time
from dataclasses import dataclass, field
from datetime import datetime

import requests

from paperbot import llmserver

from .budget import TokenBudget
from .config import ChatConfig
from .prompts import SYSTEM_PROMPT, TOOLS, MATH_RULES
from .summary import PaperSummarizer, SummaryError
from .textcache import SCOPE_LABELS, TextCache, norm

logger = logging.getLogger(__name__)

_READ_FLOOR = 4000          # 예산이 바닥나도 편당 이만큼은 준다 (초록+서론)
_SEARCH_DEFAULT, _SEARCH_MAX = 5, 8   # search_papers 결과 수
_ABSTRACT_MAX = 1500        # 검색 결과에 붙이는 초록 길이(자)
_SUMMARY_PEEK = 600         # 검색 결과에 붙이는 저장된 요약 앞부분(자)
_COMPARE_MAX = 5            # compare_papers 논문 수 상한


class LLMError(Exception):
    pass


class _Truncated(Exception):
    """스트림이 끝 표시(finish_reason·[DONE]) 없이 닫힘 — 모델 언로드·서버 재시작 등. 한 번 다시 요청한다."""


@dataclass
class AgentResult:
    answer: str
    tools: list[dict] = field(default_factory=list)   # [{name, args, result_chars}]
    papers: list[str] = field(default_factory=list)   # 근거로 열어본/언급된 논문 id
    steps: int = 0
    load_s: float = 0.0
    duration_s: float = 0.0
    prompt_tokens: int = 0                             # 마지막 라운드의 실제 입력 토큰(서버 측정, 캐시 재사용분 포함)
    ratio_en: float = 0.0                              # 보정된 영어 글자/토큰 비율


def describe_tool(name: str, args: dict, cache: TextCache) -> str:
    """진행 상황 표시용 한 줄 (Slack·Zulip DM)."""
    if name == "search_papers":
        label = SCOPE_LABELS.get(TextCache.norm_scope(args.get("scope")))
        return f"🔎 논문 검색{f' ({label})' if label else ''}: {', '.join(args.get('keywords') or [])}"
    if name == "read_paper":
        d = cache.find_by_title(str(args.get("paper_id", "")))
        return f"📖 읽는 중: {d.title if d else args.get('paper_id')}"
    if name == "compare_papers":
        refs = args.get("paper_ids") or []
        names = [(cache.find_by_title(str(r)).title if cache.find_by_title(str(r)) else str(r))[:40] for r in refs[:_COMPARE_MAX]]
        return f"⚖️ 비교 자료 수집: {' vs '.join(names)}"
    if name == "list_papers":
        where = SCOPE_LABELS.get(TextCache.norm_scope(args.get("scope")), "전체")
        return f"📚 목록 조회: {where}{' / ' + str(args['collection']) if args.get('collection') else ''}"
    if name == "summarize_paper":
        d = cache.find_by_title(str(args.get("paper_id", "")))
        return f"📝 전문 요약: {d.title if d else args.get('paper_id')}"
    return f"🔧 {name}"


class Agent:
    def __init__(self, cfg: ChatConfig, cache: TextCache, summarizer: PaperSummarizer | None = None):
        self.cfg = cfg
        self.cache = cache
        self.summarizer = summarizer          # None이면 summarize_paper 도구는 안내만 돌려준다
        saved = cache.get_meta("chars_per_token", "")     # 지난 실행에서 보정된 비율 (없으면 config 초기값)
        self.ratio_en = float(saved) if saved else float(cfg.chars_per_token)
        self._server_ctx: int | None = None   # llama-server의 실제 컨텍스트(/props) — 설정과 다르면 그쪽을 쓴다
        # 채팅 템플릿이 프롬프트에 렌더링하는 도구 정의 몫 — 첫 측정 전 추정에만 쓰인다
        self._tools_overhead = TokenBudget(1, 0, 0, self.ratio_en).est(json.dumps(TOOLS, ensure_ascii=False)) + 60

    # --- 예산 ---
    def ctx_tokens(self) -> int:
        return self._server_ctx or self.cfg.ctx_tokens

    def new_budget(self) -> TokenBudget:
        return TokenBudget(self.ctx_tokens(), self.cfg.output_reserve_tokens, self.cfg.safety_tokens, self.ratio_en,
                           overhead=self._tools_overhead)

    def new_state(self, budget: TokenBudget | None = None, messages: list[dict] | None = None) -> dict:
        """질문 하나 동안의 한도 — 토큰 예산(메시지 누적 기준), 전문 요약 횟수."""
        return {"budget": budget or self.new_budget(), "messages": messages if messages is not None else [],
                "summaries_left": self.cfg.summaries_per_question}

    def _check_server_ctx(self) -> None:
        """첫 응답 뒤 한 번, 떠 있는 llama-server의 실제 컨텍스트를 확인한다 (설정 ctx_tokens과 다르면 경고 후 사용)."""
        if self._server_ctx is not None:
            return
        self._server_ctx = self.cfg.ctx_tokens
        actual = llmserver.server_ctx(self.cfg.llm_url, self.cfg.model)   # 확인 실패면 None → 설정값
        if actual and actual != self.cfg.ctx_tokens:
            logger.warning("서버 실제 컨텍스트 %d ≠ config ctx_tokens %d — 실제 값으로 예산을 잡습니다",
                           actual, self.cfg.ctx_tokens)
            self._server_ctx = actual

    def system_prompt(self, math_style: str = "plain") -> str:
        """math_style: "plain"(Slack) | "latex"(Zulip DM — $$…$$ 인라인, ```math 블록)."""
        docs = self.cache.docs()
        if not docs:
            catalog = "[라이브러리 목록] (아직 비어 있음 — 동기화 전)"
        elif len(docs) <= self.cfg.catalog_max:
            catalog = (f"[라이브러리 논문 목록 — {len(docs)}편, 형식: id | 제목 | 첫 저자 연도 | 컬렉션]\n"
                       + "\n".join(self.cache.catalog()))
        else:
            catalog = (f"[라이브러리: {len(docs)}편 — 목록은 프롬프트에 넣지 않음. 논문은 search_papers로 찾고, "
                       "제목을 알면 read_paper·compare_papers·summarize_paper에 제목을 바로 넣어도 됨. 컬렉션별 목록은 list_papers]")
        return SYSTEM_PROMPT.format(today=datetime.now().strftime("%Y-%m-%d"), catalog=catalog, math_rule=MATH_RULES[math_style])

    def _read_limit(self, state: dict, n: int = 1) -> tuple[int, int]:
        """지금 남은 풀(토큰)과 그에 맞는 편당 본문 글자 수(바닥 _READ_FLOOR)."""
        pool = state["budget"].pool(state["messages"])
        return pool, max(_READ_FLOOR, state["budget"].chars_for(pool, n))

    def exec_tool(self, name: str, args: dict, state: dict | None = None,
                  on_status=None) -> tuple[str, list[str]]:
        """도구 실행 → (모델에 줄 결과 텍스트, 관련 논문 id들). state: new_state()의 질문당 예산·한도."""
        state = state if state is not None else self.new_state()
        if name == "search_papers":
            kws = args.get("keywords") or []
            if isinstance(kws, str):
                kws = [k for k in re.split(r"[,;]", kws) if k.strip()]
            try:
                limit = max(1, min(_SEARCH_MAX, int(args.get("limit") or _SEARCH_DEFAULT)))
            except (TypeError, ValueError):
                limit = _SEARCH_DEFAULT
            scope = TextCache.norm_scope(args.get("scope"))
            hits = self.cache.search(kws, args.get("collection") or None, limit=limit, scope=scope)
            if not hits:
                where = SCOPE_LABELS.get(scope, "라이브러리·수집분")
                return (f"검색 결과 없음 ({where}). 다른 영어 키워드(동의어·약어·상위 개념)로 다시 검색해 보거나 "
                        + ("scope=all로 범위를 넓히거나 " if scope != "all" else "") + "목록을 확인하라."), []
            blocks = []
            for h in hits:
                d = self.cache.get(h["id"])
                s = self.cache.get_summary(h["id"])
                b = (f"{h['id']} | {h['title']} | {h['authors']} {h['year']} | {', '.join(h['collections']) or '-'}"
                     + (f" [{h['origin']}]" if h.get("origin") else "")
                     + f" | 매칭 {h['matches']}\n  초록: {(d.abstract if d and d.abstract else '(없음)')[:_ABSTRACT_MAX]}")
                if s:
                    b += f"\n  저장된 요약(앞부분): {s['summary'][:_SUMMARY_PEEK]}"
                if h["snippet"]:
                    b += f"\n  발췌: {h['snippet']}"
                blocks.append(b)
            return (f"[검색 결과 {len(hits)}편 — 관련도순. 요약은 아래 초록·발췌·저장된 요약에만 근거할 것]\n"
                    + "\n\n".join(blocks)), [h["id"] for h in hits]
        if name == "read_paper":
            d = self.cache.find_by_title(str(args.get("paper_id", "")))
            if d is None:
                return (f"'{args.get('paper_id')}'에 해당하는 논문을 찾지 못했습니다. "
                        "목록의 id를 쓰거나 search_papers로 먼저 찾으세요."), []
            pool, limit = self._read_limit(state)
            focus = args.get("focus") or None
            if len(d.text) <= limit:
                # 전문이 예산에 들어가면 focus와 무관하게 통째로 — 발췌보다 항상 정보가 많다 (2026-10-21: 세부 질문 답이 얕았던 원인)
                text = self.cache.read(d.key, None, max_chars=limit)
                if focus:
                    text = f"[본문 {len(d.text):,}자 전체 — 예산에 들어가므로 focus('{focus}')로 자르지 않고 전문을 줍니다]\n" + text
            else:
                text = self.cache.read(d.key, focus, max_chars=limit)
                text = (f"[남은 컨텍스트에 맞춰 이 논문은 본문 {len(d.text):,}자 중 {limit:,}자까지만 보여줍니다"
                        + (" — 꼭 필요한 부분만 focus 키워드로 다시 읽으세요" if pool > 0 else
                           " — 예산이 바닥났으니 지금까지 읽은 내용으로 답하세요") + "]\n" + text)
            return text, [d.key]
        if name == "compare_papers":
            refs = args.get("paper_ids") or []
            if isinstance(refs, str):
                refs = [r for r in re.split(r"[,;\n]", refs) if r.strip()]
            found, missing = [], []
            for r in refs[:_COMPARE_MAX]:
                d = self.cache.find_by_title(str(r))
                (found.append(d) if d and d not in found else missing.append(str(r)))
            if len(found) < 2:
                return (f"비교할 논문을 2편 이상 찾지 못했습니다 (찾음: {[d.title for d in found]}, 못 찾음: {missing}). "
                        "목록의 id나 정확한 제목을 쓰거나 search_papers로 먼저 찾으세요."), [d.key for d in found]
            pool, per = self._read_limit(state, len(found))
            aspects = args.get("aspects") or None
            parts = [f"[비교 자료 {len(found)}편 — 편당 최대 {per:,}자"
                     + (f", 비교 축: {aspects}" if aspects else ", 각 논문 앞부분(초록·서론)과 끝부분")
                     + "; 전문이 들어가는 논문은 전문]"]
            if missing:
                parts.append(f"(못 찾은 항목: {missing} — 비교에서 제외)")
            for i, d in enumerate(found, 1):
                whole = len(d.text) <= per            # 예산에 들어가면 비교 축과 무관하게 전문
                parts.append(f"=== 논문 {i}/{len(found)}: {d.title}" + (" (전문)" if whole else " (발췌)") + " ===\n"
                             + self.cache.read(d.key, None if whole else aspects, max_chars=per))
            return "\n\n".join(parts), [d.key for d in found]
        if name == "summarize_paper":
            d = self.cache.find_by_title(str(args.get("paper_id", "")))
            pid = d.key if d else str(args.get("paper_id", "")).strip().upper()
            if self.summarizer is None:
                return "전문 요약 기능이 꺼져 있습니다(chat.summary_enabled). read_paper로 본문을 읽어 답하세요.", []
            force = str(args.get("force", "")).lower() in ("1", "true", "yes")
            if state["summaries_left"] <= 0 and (force or not self.cache.get_summary(pid)):
                return ("이번 질문에서 새로 만들 수 있는 전문 요약 수를 다 썼습니다. "
                        "이미 받은 요약과 read_paper(focus)로 답하세요."), []
            try:
                sr = self.summarizer.summarize(pid, force=force, on_status=on_status)
            except SummaryError as e:
                return f"전문 요약 실패: {e}. read_paper로 본문을 읽어 답하세요.", [pid]
            if not sr.reused:
                state["summaries_left"] -= 1
            head = (f"[전문 요약 — {sr.title} | 출처 {sr.source} | 본문 {sr.chars:,}자"
                    + (" | 저장된 요약 재사용" if sr.reused else f" | {sr.seconds:.0f}초") + "]\n")
            return (head + sr.summary
                    + "\n\n(이 요약을 바탕으로 답한다. 세부 내용이 더 필요하면 read_paper에 focus 키워드를 준다)"), [pid]
        if name == "list_papers":
            return self.cache.list_papers(args.get("collection") or None, scope=args.get("scope")), []
        return f"알 수 없는 도구: {name}", []

    def _chat(self, messages: list[dict], use_tools: bool, on_partial=None,
              max_tokens: int | None = None) -> tuple[str, list[dict], dict]:
        """llama-swap /v1/chat/completions 1회 호출(SSE 스트리밍) → (본문, tool_calls, 통계).

        tool_calls는 OpenAI 형식 [{id, type: function, function: {name, arguments: JSON 문자열}}].
        통계: prompt_tokens(캐시 재사용분 포함 전체 입력 토큰), cached_tokens, prompt_s, load_s(콜드 스타트였으면
        첫 청크까지 걸린 시간 − 프롬프트 처리 시간 ≈ 모델 로드 시간, 아니면 0).
        스트림이 끝 표시 없이 끊기면(다른 봇의 언로드·서버 재시작) 2초 뒤 한 번 다시 요청한다 — 모델은 자동으로 다시 올라간다.
        """
        payload = {"model": self.cfg.model, "messages": messages, "stream": True,
                   "stream_options": {"include_usage": True},
                   "temperature": self.cfg.temperature, "max_tokens": max_tokens or self.cfg.max_output_tokens,
                   "chat_template_kwargs": {"enable_thinking": False}}
        if use_tools:
            payload["tools"] = TOOLS
        last: Exception | None = None
        for attempt in (1, 2):
            cold = not llmserver.is_loaded(self.cfg.llm_url, self.cfg.model)
            try:
                return self._stream_once(payload, on_partial, cold)
            except _Truncated as e:
                last = e
                logger.warning("LLM 스트림이 중간에 끊김 (%d/2): %s", attempt, e)
                if attempt == 1:
                    time.sleep(2)
        raise LLMError(f"LLM 응답이 중간에 끊겼습니다: {last}")

    def _stream_once(self, payload: dict, on_partial, cold: bool) -> tuple[str, list[dict], dict]:
        t0 = time.monotonic()
        ttfb: float | None = None
        content, slots, usage, timings, finished = "", {}, {}, {}, False
        try:
            with requests.post(f"{self.cfg.llm_url}/v1/chat/completions", json=payload, stream=True,
                               timeout=(10, 600)) as r:
                if r.status_code != 200:
                    raise LLMError(f"LLM 서버 HTTP {r.status_code}: {r.text[:300]}")
                for line in r.iter_lines(decode_unicode=False):
                    if not line or not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        finished = True
                        break
                    chunk = json.loads(data)
                    if chunk.get("error"):
                        raise LLMError(f"LLM 서버 오류: {chunk['error']}")
                    if ttfb is None:
                        ttfb = time.monotonic() - t0
                    for ch in chunk.get("choices") or []:
                        delta = ch.get("delta") or {}
                        if delta.get("content"):
                            content += delta["content"]
                            if on_partial:
                                on_partial(content)
                        for tc in delta.get("tool_calls") or []:   # 이름·인자가 여러 청크에 나뉘어 온다 (index로 이어 붙임)
                            slot = slots.setdefault(int(tc.get("index") or 0), {"id": "", "name": "", "arguments": ""})
                            slot["id"] = tc.get("id") or slot["id"]
                            fn = tc.get("function") or {}
                            slot["name"] += fn.get("name") or ""
                            slot["arguments"] += fn.get("arguments") or ""
                        if ch.get("finish_reason"):
                            finished = True
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    if chunk.get("timings"):
                        timings = chunk["timings"]
        except requests.exceptions.ChunkedEncodingError as e:   # 응답 도중 연결이 끊김
            raise _Truncated(str(e)) from e
        except requests.RequestException as e:
            raise LLMError(f"LLM 서버 연결 실패 ({self.cfg.llm_url}): {e}") from e
        if not finished:
            raise _Truncated("끝 표시(finish_reason·[DONE]) 없이 스트림이 닫힘")
        calls = [{"id": s["id"] or f"call_{i}", "type": "function",
                  "function": {"name": s["name"], "arguments": s["arguments"] or "{}"}}
                 for i, s in sorted(slots.items())]
        prompt_s = float(timings.get("prompt_ms") or 0) / 1000
        stats = {"prompt_tokens": int(usage.get("prompt_tokens") or 0),
                 "cached_tokens": int((usage.get("prompt_tokens_details") or {}).get("cached_tokens")
                                      or timings.get("cache_n") or 0),
                 "prompt_s": prompt_s,
                 "load_s": max(0.0, (ttfb or 0.0) - prompt_s) if cold else 0.0}
        return content, calls, stats

    def run(self, question: str, history: list[dict], on_status=None, on_partial=None, math_style: str = "plain") -> AgentResult:
        t0 = time.time()
        res = AgentResult(answer="")
        budget = self.new_budget()
        history = budget.trim_history(history, self.cfg.history_max_tokens)
        messages = [{"role": "system", "content": self.system_prompt(math_style)}, *history,
                    {"role": "user", "content": question}]
        state = self.new_state(budget, messages)
        overflow = False
        for step in range(self.cfg.max_tool_steps + 1):
            use_tools = step < self.cfg.max_tool_steps and not overflow   # 마지막 라운드·예산 초과면 도구 없이 답
            est = budget.used(messages)
            content, calls, stats = self._chat(messages, use_tools, on_partial,
                                               budget.output_tokens(messages, self.cfg.max_output_tokens))
            res.steps = step + 1
            res.load_s += float(stats.get("load_s") or 0)
            measured = int(stats.get("prompt_tokens") or 0)
            if measured:
                self._check_server_ctx()
                budget.ctx = self.ctx_tokens()
                if budget.calibrate(messages, measured):
                    res.prompt_tokens = measured
                    if budget.ratio_en != self.ratio_en:
                        self.ratio_en = budget.ratio_en
                        try:
                            self.cache.set_meta("chars_per_token", f"{self.ratio_en:.3f}")
                        except Exception as e:  # noqa: BLE001 — 저장 실패는 다음 보정 때 다시
                            logger.debug("비율 저장 실패: %s", e)
                logger.info("라운드 %d: 입력 토큰 측정 %d (추정 %d, 캐시 재사용 %d) · 영어 %.2f자/토큰 · 남은 풀 %d",
                            step + 1, measured, est, int(stats.get("cached_tokens") or 0), budget.ratio_en,
                            budget.pool(messages))
            if not (calls and use_tools):
                res.answer = content.strip()
                break
            # _chat이 돌려준 OpenAI 형식 그대로 기록: 인자는 JSON 문자열, 결과는 tool_call_id로 연결
            messages.append({"role": "assistant", "content": content, "tool_calls": calls})
            for tc in calls:
                fn = tc["function"]
                name = fn.get("name", "")
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                if not isinstance(args, dict):
                    args = {}
                if on_status:
                    on_status(describe_tool(name, args, self.cache))
                result, ids = self.exec_tool(name, args, state, on_status)
                res.tools.append({"name": name, "args": args, "result_chars": len(result)})
                if name in ("read_paper", "compare_papers", "summarize_paper"):
                    res.papers += [i for i in ids if i not in res.papers]
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
                logger.info("도구 %s %s → %d자", name, args, len(result))
            if budget.pool(messages) < math.ceil(_READ_FLOOR / budget.ratio_en):   # 바닥 읽기도 못 넣으면 끝
                overflow = True
                logger.warning("토큰 예산 소진 (사용 추정 %d / ctx %d) — 다음 라운드는 도구 없이 답하게 함",
                               budget.used(messages), budget.ctx)
        # 답에 제목이 언급된 논문도 근거로 표시 (검색 결과만 보고 답한 경우)
        ans = norm(res.answer)
        for d in self.cache.docs():
            if d.key not in res.papers and len(d._t) > 15 and d._t in ans:
                res.papers.append(d.key)
        res.ratio_en = budget.ratio_en
        res.duration_s = time.time() - t0
        return res
