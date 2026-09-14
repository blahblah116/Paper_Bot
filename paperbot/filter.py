"""2단계 relevance 필터.

1단계 — 임베딩 프리필터:
  topic의 interest 문단과 논문의 title+abstract를 임베딩해 cosine similarity가
  embedding_threshold 미만이면 컷. 모델은 프로세스당 1회만 로드(CPU 기본).
  abstract가 없는 논문은 title만으로 점수를 내되 하드컷하지 않고 judge로 넘긴다
  (title-only 유사도는 체계적으로 낮게 나와 같은 threshold로 자르면 과잉 컷).

2단계 — LLM judge (llm_judge: true일 때):
  임베딩 통과분을 Ollama로 0~10 채점. JSON 파싱 실패 시 1회 재요청,
  그래도 실패하면 보수적으로 통과(놓치는 것보다 나음).
  judge_threshold 미만이면 filtered_out으로 기록해 재채점을 방지.
"""
from __future__ import annotations

import json
import logging
import re

from openai import OpenAI

from .config import Config, Topic
from .models import Paper
from .prompts import JUDGE_SYSTEM_PROMPT, build_judge_user_prompt
from .store import Store
from .summarizer import LLMError, chat

logger = logging.getLogger(__name__)

_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)

# judge 출력 스키마 — 서버(Ollama)가 이 형식을 강제하므로 파싱 실패가 구조적으로 사라진다.
_JUDGE_RESPONSE_FORMAT = {
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


def _parse_judge_json(text: str) -> tuple[float, str] | None:
    """LLM 출력에서 {"score": n, "reason": "..."} 파싱. 실패 시 None."""
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        score = float(obj["score"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    if not (0.0 <= score <= 10.0):
        return None
    return score, str(obj.get("reason", ""))


class RelevanceFilter:
    def __init__(self, cfg: Config, judge_client: OpenAI):
        self.cfg = cfg
        self.judge_client = judge_client
        self._st_model = None            # sentence-transformers 모델 (lazy)
        self._interest_emb: dict[str, "object"] = {}  # topic name -> 임베딩 캐시

    # ------------------------------------------------------------ 임베딩

    def _model(self):
        if self._st_model is None:
            # 무거운 import라 필요할 때만 (--skip-filter 실행이 torch를 로드하지 않도록).
            from sentence_transformers import SentenceTransformer

            fc = self.cfg.filter
            logger.info("임베딩 모델 로드: %s (device=%s)", fc.embedding_model, fc.embedding_device)
            self._st_model = SentenceTransformer(fc.embedding_model, device=fc.embedding_device)
        return self._st_model

    def _interest_embedding(self, topic: Topic):
        if topic.name not in self._interest_emb:
            self._interest_emb[topic.name] = self._model().encode(
                topic.interest, normalize_embeddings=True
            )
        return self._interest_emb[topic.name]

    def _embedding_scores(self, topic: Topic, papers: list[Paper]) -> list[float]:
        interest = self._interest_embedding(topic)
        texts = [p.embed_text for p in papers]
        embs = self._model().encode(texts, normalize_embeddings=True, batch_size=32)
        return [float(interest @ e) for e in embs]  # 정규화됐으므로 내적 = cosine

    # ------------------------------------------------------------ judge

    def _judge(self, topic: Topic, paper: Paper) -> tuple[float, str] | None:
        """0~10 점수와 근거. 파싱 실패 1회 재요청, 그래도 실패면 None(=통과)."""
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": build_judge_user_prompt(topic.interest, paper)},
        ]
        for attempt in (1, 2):
            try:
                out = chat(
                    self.judge_client, self.cfg, messages,
                    temperature=0.0, max_tokens=2560,  # thinking 꺼짐 + 스키마 강제 → 실제론 수십 토큰
                    response_format=_JUDGE_RESPONSE_FORMAT,
                    label=f"judge({paper.uid})",
                )
            except LLMError:
                return None  # 호출 자체가 계속 실패 → 보수적으로 통과
            parsed = _parse_judge_json(out)
            if parsed is not None:
                return parsed
            logger.warning("judge JSON 파싱 실패 (%d/2) %s: %.120s", attempt, paper.uid, out)
            # temperature 0으로 같은 프롬프트를 다시 보내면 같은 오류가 반복되므로,
            # 재요청에는 잘못된 출력과 교정 지시를 덧붙인다.
            messages = messages + [
                {"role": "assistant", "content": out[:500]},
                {"role": "user", "content": '형식이 잘못됐다. 다른 텍스트 없이 정확히 '
                                            '{"score": <0-10 정수>, "reason": "<한 문장>"} '
                                            'JSON 객체 하나만 다시 출력하라.'},
            ]
        return None

    # ------------------------------------------------------------ 메인

    def apply(self, papers: list[Paper], topics: dict[str, Topic], store: Store) -> list[Paper]:
        """필터를 통과한 논문 리스트를 반환. 탈락분은 store에 filtered_out으로 기록."""
        fc = self.cfg.filter
        passed: list[Paper] = []

        # topic별로 묶어 임베딩을 배치 처리.
        by_topic: dict[str, list[Paper]] = {}
        for p in papers:
            by_topic.setdefault(p.topic_name, []).append(p)

        for topic_name, group in by_topic.items():
            topic = topics[topic_name]
            scores = self._embedding_scores(topic, group)

            for paper, sim in zip(group, scores):
                # 1단계: 임베딩 컷 (abstract 있는 논문만 하드컷).
                if paper.abstract and sim < fc.embedding_threshold:
                    logger.info("임베딩 컷 %s: sim=%.3f < %.2f — %.60s",
                                paper.uid, sim, fc.embedding_threshold, paper.title)
                    store.mark_filtered(paper, f"embedding_sim={sim:.3f}")
                    continue

                # 2단계: LLM judge.
                if fc.llm_judge:
                    result = self._judge(topic, paper)
                    if result is not None:
                        score, reason = result
                        paper.judge_score = score
                        if score < fc.judge_threshold:
                            logger.info("judge 컷 %s: %.1f < %.1f — %s",
                                        paper.uid, score, fc.judge_threshold, reason[:80])
                            store.mark_filtered(paper, f"judge: {reason}")
                            continue
                        logger.info("judge 통과 %s: %.1f (sim=%.3f)", paper.uid, score, sim)
                    else:
                        logger.warning("judge 실패 %s — 보수적으로 통과", paper.uid)

                passed.append(paper)

        return passed
