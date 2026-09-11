"""LLM 프롬프트 템플릿 — relevance judge용 + 요약용.

요약은 항상 6개 섹션을 고정된 마크다운 헤딩으로 출력한다.
출력 언어는 config의 language를 따르되, 전문 용어는 영어 원문을 유지한다.
"""
from __future__ import annotations

from .models import Paper

# ---------------------------------------------------------------- 요약

# (heading, 작성 가이드) — 순서 고정.
SECTIONS: list[tuple[str, str]] = [
    ("한 줄 요약", "이 논문이 푸는 문제와 해법을 한 문장으로."),
    ("문제 정의", "어떤 문제를 왜 푸는가. 배경과 동기."),
    ("핵심 방법", "접근법을 구체적으로 — 모델 구조, 학습 방식, 핵심 수식/아이디어."),
    ("기존 연구 대비 novelty", "선행 연구와 무엇이 다른가. 무엇이 새로운 기여인가."),
    ("실험 결과", "데이터셋, 베이스라인, 주요 정량 수치. 없으면 '명시되지 않음'."),
    ("한계점", "저자가 밝힌/드러난 한계와 향후 과제."),
]

SUMMARY_SYSTEM_PROMPT = """\
당신은 머신러닝/컴퓨터과학 논문을 정확하게 요약하는 연구 보조자다.
주어진 논문 본문만 근거로 삼고, 본문에 없는 내용을 지어내지 마라.
수치는 본문에 등장한 값만 사용하고, 확실하지 않으면 '명시되지 않음'이라고 적어라.

출력 규칙:
- 출력 언어: {language}. 단, 전문 용어(예: attention, B-rep, contrastive learning)는 영어 원문을 유지한다.
- 아래 6개 섹션을 정확히 이 순서와 헤딩으로 출력한다. 헤딩 텍스트를 바꾸지 마라.
- 각 섹션은 '###' 마크다운 헤딩을 사용한다.
- 서론/맺음말/메타 발언(예: '요약하겠습니다') 없이 첫 헤딩부터 바로 시작한다.
- <think> 같은 사고 과정을 출력에 포함하지 마라.

출력 형식:
{skeleton}
"""


def _skeleton() -> str:
    lines: list[str] = []
    for i, (title, guide) in enumerate(SECTIONS, start=1):
        lines.append(f"### {i}. {title}")
        lines.append(f"({guide})")
        lines.append("")
    return "\n".join(lines).rstrip()


def build_summary_system_prompt(language: str) -> str:
    return SUMMARY_SYSTEM_PROMPT.format(language=language, skeleton=_skeleton())


def build_summary_user_prompt(paper: Paper, body_text: str, abstract_only: bool) -> str:
    authors = ", ".join(paper.authors[:8])
    if len(paper.authors) > 8:
        authors += " et al."

    source_label = (
        "아래는 이 논문의 abstract이다 (PDF 본문 확보 실패로 abstract만 제공)."
        if abstract_only
        else "아래는 이 논문의 본문 텍스트다 (길면 중간이 절단되어 있을 수 있다)."
    )

    venue = paper.venue or "arXiv preprint"
    return f"""\
# 논문 메타데이터
- 제목: {paper.title}
- 저자: {authors}
- 출처: {venue} {paper.year or ""}

{source_label}

---
{body_text}
---

위 내용을 근거로 지정된 6개 섹션 형식에 맞춰 요약하라."""


# ---------------------------------------------------------------- relevance judge

JUDGE_SYSTEM_PROMPT = """\
당신은 연구자의 관심사와 논문의 관련성을 채점하는 심사자다.
관심사 서술과 논문 정보를 보고 0~10 정수 점수를 매긴다.

기준:
- 9~10: 관심사의 핵심 주제를 정면으로 다룸
- 6~8: 관심사와 상당히 관련됨 (방법/문제가 겹침)
- 3~5: 주변부 관련 (같은 분야지만 관심사 서술과 거리가 있음)
- 0~2: 무관하거나 관심사에서 명시적으로 제외한 부류

출력 규칙: 다음 JSON 객체 하나만 출력한다. 다른 텍스트/설명/코드블록 금지.
{"score": <0-10 정수>, "reason": "<한 문장 근거>"}
"""


def build_judge_user_prompt(interest: str, paper: Paper) -> str:
    abstract = paper.abstract or "(abstract 없음 — 제목만으로 판단)"
    venue = paper.venue or "arXiv preprint"
    return f"""\
# 연구자의 관심사
{interest.strip()}

# 논문
- 제목: {paper.title}
- 출처: {venue} {paper.year or ""}
- Abstract: {abstract}

이 논문이 관심사에 얼마나 관련되는지 JSON으로 채점하라."""
