"""LLM 프롬프트 — 요약 / relevance judge / 대표 그림 선택.

요약은 고정된 5개 섹션을 마크다운 헤딩으로 출력한다 (Slack 메인 메시지에 '한 줄 요약'을 뽑아 쓴다).
출력 언어는 config의 language를 따르되 전문 용어는 영어 원문을 유지한다.
"""
from __future__ import annotations

from .models import Article, FetchedImage

# ---------------------------------------------------------------- 요약

SECTIONS: list[tuple[str, str]] = [
    ("한 줄 요약", "이 글이 발표·설명하는 핵심을 한 문장으로."),
    ("배경과 동기", "어떤 문제를 왜 다루는가. 기존 접근의 한계."),
    ("핵심 내용", "모델·방법·시스템을 구체적으로 — 구조, 데이터, 학습/추론 방식, 핵심 아이디어. 불릿 3~6개."),
    ("결과", "평가 방법, 비교 대상, 주요 수치, 적용 사례. 본문에 없으면 '명시되지 않음'."),
    ("공개 범위와 한계", "논문·코드·모델·데이터·API 공개 여부, 글이 밝힌 한계와 향후 계획."),
]

SUMMARY_SYSTEM_PROMPT = """\
당신은 AI 연구 기관·기업의 기술 블로그 글을 연구자에게 정확하게 요약해 주는 연구 보조자다.
주어진 글 본문만 근거로 삼고, 본문에 없는 내용을 지어내지 마라.
수치는 본문에 등장한 값만 사용하고, 확실하지 않으면 '명시되지 않음'이라고 적어라.
본문의 "[그림 N] ..." 줄은 그 위치에 있는 그림의 캡션이다. 내용 파악에 활용하되 요약에 그림 번호를 쓰지는 마라.
홍보 문구(예: '혁신적인', '획기적인')는 빼고 사실 위주로 쓴다.

출력 규칙:
- 출력 언어: {language}. 단, 전문 용어·모델명·제품명(예: diffusion, state-space model, Gemini)은 영어 원문을 유지한다.
- 영어 용어를 한글로 음차하거나 한글·영어를 한 단어 안에 섞지 마라 (나쁜 예: '에임보디', '스펙ulative'). 원문 철자 그대로 쓴다 (좋은 예: embodied, speculative decoding).
- 아래 5개 섹션을 정확히 이 순서와 헤딩으로 출력한다. 헤딩 텍스트를 바꾸지 마라.
- 각 섹션은 '###' 마크다운 헤딩을 사용한다.
- 서론/맺음말/메타 발언(예: '요약하겠습니다') 없이 첫 헤딩부터 바로 시작한다.
{math_rule}
출력 형식:
{skeleton}
"""

# 수식 표기 — 발행처별. Slack은 LaTeX를 렌더링하지 못하므로 평문, Zulip은 $$…$$(인라인)과 ```math 블록을 렌더링한다.
MATH_RULES = {
    "plain": "- 수식은 평문으로 적는다(예: O(n^2), x_i). LaTeX 명령은 쓰지 마라.\n",
    "latex": ("- 수식은 LaTeX로 적는다: 인라인 수식은 $$...$$ 로 감싸되 **$$ 앞뒤에 반드시 공백을 하나씩 둔다** "
              "(좋은 예: '합은 $$a+b+c$$ 이다', '복잡도는 $$O(n^2)$$ 이다' — 나쁜 예: '합은$$a+b+c$$이다', '$$O(n^2)$$이다'). "
              "조사가 바로 붙어도 공백을 둔다. 여러 줄 수식은 ```math 로 시작해 ``` 로 끝나는 블록에 넣는다. 수식 밖에서는 LaTeX 명령을 쓰지 마라.\n"),
}


def _skeleton() -> str:
    lines: list[str] = []
    for i, (title, guide) in enumerate(SECTIONS, start=1):
        lines.append(f"### {i}. {title}")
        lines.append(f"({guide})")
        lines.append("")
    return "\n".join(lines).rstrip()


def build_summary_system_prompt(language: str, math: str = "plain") -> str:
    """math: "plain"(Slack) | "latex"(Zulip — $$…$$ 인라인, ```math 블록)."""
    return SUMMARY_SYSTEM_PROMPT.format(language=language, skeleton=_skeleton(), math_rule=MATH_RULES[math])


def _meta_lines(article: Article) -> str:
    date = article.published.strftime("%Y-%m-%d") if article.published else "미상"
    cats = ", ".join(article.categories[:6]) or "없음"
    return (f"- 제목: {article.title}\n"
            f"- 출처: {article.source_label}\n"
            f"- 게시일: {date}\n"
            f"- 카테고리: {cats}")


def build_summary_user_prompt(article: Article, body: str) -> str:
    return f"""\
# 글 메타데이터
{_meta_lines(article)}

아래는 이 글의 본문이다 (길면 중간이 절단되어 있을 수 있다).

---
{body}
---

위 내용을 근거로 지정된 5개 섹션 형식에 맞춰 요약하라."""


# ---------------------------------------------------------------- relevance judge

JUDGE_SYSTEM_PROMPT = """\
당신은 연구자의 관심사와 기술 블로그 글의 관련성을 채점하는 심사자다.
관심사 서술과 글 정보(제목·카테고리·본문 앞부분)를 보고 0~10 정수 점수를 매긴다.

기준:
- 9~10: 관심사의 핵심을 정면으로 다루는 연구·기술 글
- 6~8: 관심사와 상당히 관련된 연구·기술 글
- 3~5: 같은 분야지만 관심사와 거리가 있거나, 기술 내용이 얕은 글
- 0~2: 무관하거나, 관심사에서 명시적으로 제외한 부류(제품 홍보·행사·투자·파트너십 소식 등)

출력 규칙: 다음 JSON 객체 하나만 출력한다. 다른 텍스트/설명/코드블록 금지.
{"score": <0-10 정수>, "reason": "<한 문장 근거>"}
"""


def build_judge_user_prompt(interest: str, article: Article, excerpt: str) -> str:
    excerpt = excerpt.strip() or "(본문 없음 — 제목과 카테고리만으로 판단)"
    return f"""\
# 연구자의 관심사
{interest.strip()}

# 글
{_meta_lines(article)}

## 본문 앞부분
{excerpt}

이 글이 관심사에 얼마나 관련되는지 JSON으로 채점하라."""


# ---------------------------------------------------------------- 대표 그림 선택

PICK_SYSTEM_PROMPT = """\
당신은 기술 블로그 글의 그림 중 Slack 요약에 첨부할 대표 그림을 고르는 편집자다.
각 그림의 캡션·대체 텍스트·주변 문맥을 보고, 글의 핵심을 이해하는 데 가장 도움이 되는 그림을 고른다.

우선순위:
1. 방법·시스템 구조도, 파이프라인 개요도
2. 핵심 결과 그래프·표, 정성적 비교
3. 대표 예시·데모
피할 것: 장식용 배너, 비슷한 UI 스크린샷의 반복, 로고·인물 사진.

출력 규칙: {"selected": [그림 번호, ...]} JSON 객체 하나만 출력한다. 번호는 목록에 있는 것만 쓴다.
"""


def build_pick_user_prompt(article: Article, images: list[FetchedImage], k: int) -> str:
    lines = []
    for i, img in enumerate(images, start=1):
        r = img.ref
        desc = r.caption or r.alt or "(캡션 없음)"
        ctx = r.context[-200:] if r.context else ""
        kind = "대표 이미지(og:image)" if r.is_hero else f"본문 그림 {r.no}"
        lines.append(f"[{i}] {kind}, {img.width}x{img.height}\n    캡션: {desc[:400]}"
                     + (f"\n    직전 문단: {ctx}" if ctx else ""))
    listing = "\n".join(lines)
    return f"""\
# 글
- 제목: {article.title}
- 출처: {article.source_label}

# 그림 후보
{listing}

이 중 최대 {k}개를 골라 JSON으로 답하라."""
