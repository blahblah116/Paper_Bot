"""Token budget for the DM agent: how much of the server context a question may spend on tool results.
DM 에이전트의 토큰 예산 — 질문 하나가 서버 컨텍스트 중 도구 결과에 쓸 수 있는 몫을 계산한다.

- 고정 글자 상한 대신: 풀 = ctx − 출력 예약(max_output_tokens) − 안전 여유 − (지금까지의 메시지 토큰).
- 토큰은 글자 수로 추정한다: 한글은 RATIO_KO(1.7자/토큰) 고정, 그 외(영어 본문·기호)는 ratio_en.
  ratio_en은 Ollama가 돌려주는 실제 입력 토큰 수(prompt_eval_count)로 라운드마다 보정한다(EMA, 2~6 범위).
- 측정값이 있으면 그 시점까지의 메시지는 측정 토큰을 쓰고, 그 뒤에 붙은 메시지만 추정한다.
- 대화 기억은 max_tokens를 넘으면 오래된 쌍부터 버린다.
"""
from __future__ import annotations

import json
import math
import re

_HANGUL = re.compile(r"[가-힣]")
_MSG_OVERHEAD = 6            # 메시지당 역할·구분 토큰
RATIO_KO = 1.7               # 한글 글자/토큰 (Qwen 토크나이저 실측 범위 1.5~2.0)
RATIO_MIN, RATIO_MAX = 2.0, 6.0


def split_chars(text: str) -> tuple[int, int]:
    """(한글 글자 수, 그 외 글자 수)"""
    ko = len(_HANGUL.findall(text))
    return ko, len(text) - ko


def _message_text(m: dict) -> str:
    t = str(m.get("content") or "")
    if m.get("tool_calls"):
        t += json.dumps(m["tool_calls"], ensure_ascii=False)
    return t


class TokenBudget:
    def __init__(self, ctx: int, reserve_output: int, safety: int, ratio_en: float = 3.5):
        self.ctx, self.reserve_output, self.safety = int(ctx), int(reserve_output), int(safety)
        self.ratio_en = float(ratio_en)
        self.measured: tuple[int, int] | None = None   # (측정 시점의 메시지 수, 측정된 입력 토큰)

    # --- 추정 ---
    def est(self, text: str) -> int:
        ko, other = split_chars(text or "")
        return math.ceil(ko / RATIO_KO + other / self.ratio_en)

    def est_message(self, m: dict) -> int:
        return self.est(_message_text(m)) + _MSG_OVERHEAD

    def est_messages(self, messages: list[dict], start: int = 0) -> int:
        return sum(self.est_message(m) for m in messages[start:])

    def used(self, messages: list[dict]) -> int:
        """지금 메시지들의 입력 토큰 — 측정값이 있으면 그것 + 이후 추가분 추정."""
        if self.measured and self.measured[0] <= len(messages):
            n, tokens = self.measured
            return tokens + self.est_messages(messages, n)
        return self.est_messages(messages)

    # --- 배분 ---
    def pool(self, messages: list[dict]) -> int:
        """도구 결과에 더 쓸 수 있는 토큰 (음수면 이미 초과)."""
        return self.ctx - self.reserve_output - self.safety - self.used(messages)

    def chars_for(self, tokens: int, n: int = 1) -> int:
        """영어 본문 기준으로 토큰 → 글자 수 (n편이면 n으로 나눔)."""
        return int(max(0, tokens) * self.ratio_en / max(1, n))

    def output_tokens(self, messages: list[dict], max_output: int) -> int:
        """최종 생성에 줄 토큰 — 남은 컨텍스트를 넘지 않게 (최소 256)."""
        left = self.ctx - self.used(messages) - self.safety
        return max(256, min(int(max_output), left))

    # --- 보정 ---
    def calibrate(self, messages: list[dict], measured_tokens: int) -> bool:
        """실측 입력 토큰으로 ratio_en 보정. 측정값이 추정의 절반 미만이면(캐시 제외 집계 등) 무시 → False."""
        if not measured_tokens or measured_tokens <= 0:
            return False
        est = self.est_messages(messages)
        if measured_tokens < 0.5 * est:
            return False
        ko = other = 0
        for m in messages:
            a, b = split_chars(_message_text(m))
            ko += a
            other += b
        rest = measured_tokens - ko / RATIO_KO - _MSG_OVERHEAD * len(messages)
        # 영어가 주가 아닐 때(프롬프트·한국어 답변만 있을 때)는 한글 몫의 오차가 영어 비율로 튀므로 측정값만 기억한다
        if other > 0 and rest > 0 and other >= 2 * ko:
            ratio = other / rest
            if RATIO_MIN <= ratio <= RATIO_MAX:
                self.ratio_en = round(0.7 * self.ratio_en + 0.3 * ratio, 3)
        self.measured = (len(messages), int(measured_tokens))
        return True

    def trim_history(self, history: list[dict], max_tokens: int) -> list[dict]:
        """오래된 질문·답 쌍부터 버려 기억을 max_tokens 안으로."""
        h = list(history)
        while h and self.est_messages(h) > max_tokens:
            h = h[2:] if len(h) >= 2 else []
        return h
