"""DM chat agent: qwen picks tools (search / read / list) over the library and answers.
DM 챗봇 에이전트 — qwen이 도구(라이브러리 검색·논문 읽기·목록)를 골라 쓰고 답한다.

- Ollama 네이티브 /api/chat 스트리밍 (tools, think=false, keep_alive). 임베딩 모델 없이 도구 호출로 근거를 찾는다.
  실측(2026-10-21): 한국어 질문 → 영어 키워드로 search_library 호출 → 정답 논문 제목 (첫 단계 15.9s 중 로딩 12.6s).
- 도구 라운드는 max_tool_steps까지, 마지막 라운드는 도구 없이 답하게 강제한다.
- 답에 제목이 언급된 논문도 근거(papers)로 표시한다 — 검색 결과만 보고 답한 경우.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime

import requests

from .config import ChatConfig
from .prompts import SYSTEM_PROMPT, TOOLS
from .textcache import TextCache, norm

logger = logging.getLogger(__name__)


class LLMError(Exception):
    pass


@dataclass
class AgentResult:
    answer: str
    tools: list[dict] = field(default_factory=list)   # [{name, args, result_chars}]
    papers: list[str] = field(default_factory=list)   # 근거로 열어본/언급된 논문 id
    steps: int = 0
    load_s: float = 0.0
    duration_s: float = 0.0


def describe_tool(name: str, args: dict, cache: TextCache) -> str:
    """Slack 진행 상황 표시용 한 줄."""
    if name == "search_library":
        return f"🔎 라이브러리 검색: {', '.join(args.get('keywords') or [])}"
    if name == "read_paper":
        d = cache.get(str(args.get("paper_id", "")))
        return f"📖 읽는 중: {d.title if d else args.get('paper_id')}"
    if name == "list_papers":
        return f"📚 목록 조회: {args.get('collection') or '전체'}"
    return f"🔧 {name}"


class Agent:
    def __init__(self, cfg: ChatConfig, cache: TextCache):
        self.cfg = cfg
        self.cache = cache

    def system_prompt(self) -> str:
        docs = self.cache.docs()
        if not docs:
            catalog = "[라이브러리 목록] (아직 비어 있음 — 동기화 전)"
        elif len(docs) <= self.cfg.catalog_max:
            catalog = (f"[라이브러리 논문 목록 — {len(docs)}편, 형식: id | 제목 | 첫 저자 연도 | 컬렉션]\n"
                       + "\n".join(self.cache.catalog()))
        else:
            catalog = f"[라이브러리: {len(docs)}편 — 목록이 길어 생략. list_papers/search_library로 조회]"
        return SYSTEM_PROMPT.format(today=datetime.now().strftime("%Y-%m-%d"), catalog=catalog)

    def exec_tool(self, name: str, args: dict) -> tuple[str, list[str]]:
        """도구 실행 → (모델에 줄 결과 텍스트, 관련 논문 id들)."""
        if name == "search_library":
            kws = args.get("keywords") or []
            if isinstance(kws, str):
                kws = [k for k in re.split(r"[,;]", kws) if k.strip()]
            hits = self.cache.search(kws, args.get("collection") or None)
            if not hits:
                return "검색 결과 없음. 다른 영어 키워드(동의어·약어·상위 개념)로 다시 검색해 보거나 목록을 확인하라.", []
            lines = [f"{h['id']} | {h['title']} | {h['authors']} {h['year']} | {', '.join(h['collections']) or '-'} "
                     f"| 매칭 {h['matches']}\n  …{h['snippet']}" for h in hits]
            return "\n".join(lines), [h["id"] for h in hits]
        if name == "read_paper":
            pid = str(args.get("paper_id", ""))
            return self.cache.read(pid, args.get("focus") or None), [pid.strip().upper()]
        if name == "list_papers":
            return self.cache.list_papers(args.get("collection") or None), []
        return f"알 수 없는 도구: {name}", []

    def _chat(self, messages: list[dict], use_tools: bool, on_partial=None) -> tuple[str, list[dict], dict]:
        """Ollama /api/chat 1회 호출(스트리밍) → (본문, tool_calls, 마지막 청크 통계)."""
        payload = {"model": self.cfg.model, "messages": messages, "stream": True, "think": False,
                   "keep_alive": self.cfg.keep_alive,
                   "options": {"temperature": self.cfg.temperature, "num_predict": self.cfg.max_output_tokens}}
        if use_tools:
            payload["tools"] = TOOLS
        content, calls, stats = "", [], {}
        try:
            with requests.post(f"{self.cfg.llm_url}/api/chat", json=payload, stream=True,
                               timeout=(10, 600)) as r:
                if r.status_code != 200:
                    raise LLMError(f"Ollama HTTP {r.status_code}: {r.text[:300]}")
                for line in r.iter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    if chunk.get("error"):
                        raise LLMError(f"Ollama 오류: {chunk['error']}")
                    msg = chunk.get("message") or {}
                    if msg.get("content"):
                        content += msg["content"]
                        if on_partial:
                            on_partial(content)
                    if msg.get("tool_calls"):
                        calls.extend(msg["tool_calls"])
                    if chunk.get("done"):
                        stats = chunk
        except requests.RequestException as e:
            raise LLMError(f"Ollama 연결 실패: {e}") from e
        return content, calls, stats

    def run(self, question: str, history: list[dict], on_status=None, on_partial=None) -> AgentResult:
        t0 = time.time()
        res = AgentResult(answer="")
        messages = [{"role": "system", "content": self.system_prompt()}, *history,
                    {"role": "user", "content": question}]
        for step in range(self.cfg.max_tool_steps + 1):
            use_tools = step < self.cfg.max_tool_steps   # 마지막 라운드는 도구 없이 답하게 강제
            content, calls, stats = self._chat(messages, use_tools, on_partial)
            res.steps = step + 1
            res.load_s += stats.get("load_duration", 0) / 1e9
            if not (calls and use_tools):
                res.answer = content.strip()
                break
            messages.append({"role": "assistant", "content": content, "tool_calls": calls})
            for tc in calls:
                fn = tc.get("function") or {}
                name, args = fn.get("name", ""), fn.get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                if on_status:
                    on_status(describe_tool(name, args, self.cache))
                result, ids = self.exec_tool(name, args)
                res.tools.append({"name": name, "args": args, "result_chars": len(result)})
                if name == "read_paper":
                    res.papers += [i for i in ids if i not in res.papers]
                messages.append({"role": "tool", "content": result, "tool_name": name})
                logger.info("도구 %s %s → %d자", name, args, len(result))
        # 답에 제목이 언급된 논문도 근거로 표시 (검색 결과만 보고 답한 경우)
        ans = norm(res.answer)
        for d in self.cache.docs():
            if d.key not in res.papers and len(d._t) > 15 and d._t in ans:
                res.papers.append(d.key)
        res.duration_s = time.time() - t0
        return res
