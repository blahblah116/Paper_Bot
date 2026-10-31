"""Tool definitions and the system prompt for the DM chat agent.
DM 챗봇의 도구 정의와 시스템 프롬프트 — 모델이 받는 지시는 전부 여기 있다.

질문은 세 유형으로 나눈다 (2026-10-21 사용자 설계):
  1) search    — "~에 대한 논문 찾아줘": search_papers → 논문마다 제목 + 3~5줄 요약
  2) question  — 특정 논문에 대한 질문: read_paper(focus) → 본문에 근거해 추론, 없으면 모른다고
  3) comparing — 여러 논문 비교: compare_papers → 공통점·차이점
보조: summarize_paper(전문 요약), list_papers(컬렉션 목록).

검색 범위(scope, 2026-10-23): 라이브러리(Zotero/Drive — 사용자가 직접 저장) + 수집분(paperbot이 매일 요약해 올린 논문,
articlebot이 올린 연구 블로그 글). 기본은 전체(all)이고, 사용자가 "내 라이브러리에서" / "봇이 올린 것 중" 하면 좁힌다.

행동 방식(어떤 유형에 어떤 도구를 쓰는지, 지어내지 말 것, Slack 서식)만 프롬프트로 지시한다.
대화 기억 길이·도구 호출 횟수·읽기 분량 같은 한도는 코드/config가 강제한다(agent.py, config.py).
"""

TOOLS = [
    {"type": "function", "function": {
        "name": "search_papers",
        "description": "[search] 어떤 내용·주제에 대한 논문·글을 찾는다. 라이브러리(Zotero/Drive)와 수집분(paperbot이 요약해 올린 논문, "
                       "articlebot이 올린 연구 블로그 글)의 제목·초록·본문을 영어 키워드로 검색해 관련 문서마다 id, 제목, 저자·연도, "
                       "컬렉션/주제, 출처, 초록, 매칭 발췌, (있으면) 저장된 요약 앞부분을 돌려준다. 이 자료로 문서마다 3~5줄 요약을 써서 보여준다.",
        "parameters": {"type": "object", "properties": {
            "keywords": {"type": "array", "items": {"type": "string"},
                         "description": "영어 키워드 1~6개 (예: [\"graph transformer\", \"scalability\", \"linear attention\"])"},
            "collection": {"type": "string", "description": "특정 컬렉션(라이브러리) 또는 topic·기관명(수집분)으로 한정할 때만 (선택)"},
            "scope": {"type": "string", "enum": ["all", "library", "feed"],
                      "description": "검색 범위: all(기본, 전체) | library(사용자 Zotero/Drive 라이브러리만) | feed(봇이 수집해 올린 논문·블로그 글만)"},
            "limit": {"type": "integer", "description": "돌려줄 문서 수 (기본 5, 최대 8)"},
        }, "required": ["keywords"]}}},
    {"type": "function", "function": {
        "name": "read_paper",
        "description": "[question] 특정 논문에 대해 질문받았을 때 그 논문의 본문을 읽는다. 답은 반드시 이 본문에 근거한다. "
                       "본문이 예산에 들어가면 focus와 무관하게 전문을 주고, 긴 논문만 focus(질문의 영어 핵심어, 쉼표 구분) 주변을 발췌한다. "
                       "저장된 전문 요약이 있으면 앞에 붙는다. paper_id에는 id 대신 논문 제목을 넣어도 된다. "
                       "같은 논문에 대한 추가·세부 질문이면 다시 호출해서 읽은 뒤 답한다.",
        "parameters": {"type": "object", "properties": {
            "paper_id": {"type": "string", "description": "논문 id (목록·검색 결과) 또는 논문 제목"},
            "focus": {"type": "string", "description": "질문의 영어 핵심어, 쉼표 구분 (예: \"attention complexity, linear, Theorem\")"},
        }, "required": ["paper_id"]}}},
    {"type": "function", "function": {
        "name": "compare_papers",
        "description": "[comparing] 여러 논문을 비교할 때 각 논문의 자료(메타데이터, 저장된 요약, 본문 발췌)를 한 번에 받는다. "
                       "aspects에 비교 축 영어 키워드(쉼표 구분)를 주면 그 부분 위주로 발췌하고, 없으면 각 논문의 앞부분·끝부분을 준다. "
                       "받은 자료로 공통점과 차이점을 비교 축별로 밝힌다. 항목에는 id 또는 제목을 넣는다.",
        "parameters": {"type": "object", "properties": {
            "paper_ids": {"type": "array", "items": {"type": "string"}, "description": "비교할 논문 2~5편의 id 또는 제목"},
            "aspects": {"type": "string", "description": "비교 축 영어 키워드, 쉼표 구분 (예: \"positional encoding, complexity, datasets, accuracy\")"},
        }, "required": ["paper_ids"]}}},
    {"type": "function", "function": {
        "name": "summarize_paper",
        "description": "논문 한 편의 전체 내용 요약을 요청받았을 때 쓴다. 라이브러리에 저장된 원문 전체를 paperbot 형식(6개 섹션)으로 요약한다. "
                       "새로 만들면 2~3분 걸리고, 이미 만든 요약이 있으면 바로 돌려준다. 특정 부분만 묻는 질문에는 read_paper가 맞다.",
        "parameters": {"type": "object", "properties": {
            "paper_id": {"type": "string", "description": "논문 id 또는 제목"},
            "force": {"type": "boolean", "description": "저장된 요약이 있어도 다시 만들 때만 true (선택)"},
        }, "required": ["paper_id"]}}},
    {"type": "function", "function": {
        "name": "list_papers",
        "description": "문서 목록. collection을 주면 그 컬렉션(라이브러리) 또는 topic·기관(수집분)만. scope로 라이브러리/수집분을 고른다. "
                       "'어떤 컬렉션에 뭐가 있어?', '최근 봇이 올린 gdl 논문 뭐 있어?' 같은 질문에 쓴다. 너무 많으면 일부만 온다.",
        "parameters": {"type": "object", "properties": {
            "collection": {"type": "string", "description": "컬렉션 이름 또는 topic·기관명 (선택)"},
            "scope": {"type": "string", "enum": ["all", "library", "feed"], "description": "all(기본) | library | feed"},
        }}}},
]

SYSTEM_PROMPT = """너는 'paperbot'이다. 연구실 메신저(Slack/Zulip)에서 사용자의 논문 라이브러리(Zotero)와 봇이 수집한 논문·글을 아는 연구 도우미 챗봇이고, 한국어로 답한다. 오늘은 {today}이다.

문서 출처는 두 가지다. (1) 라이브러리: 사용자가 Zotero/Google Drive에 직접 저장한 논문(컬렉션으로 분류). (2) 수집분: paperbot이 매일 주제(topic)별로 골라 요약해 올린 논문 [paperbot 수집]과 articlebot이 올린 연구 블로그 글 [블로그 글] — 컬렉션 자리에 topic 이름·기관명이 들어간다.
도구는 기본으로 둘 다 검색한다(scope=all). 사용자가 "내 라이브러리에서", "저장한 논문 중" 하면 scope=library, "봇이 올린/요약한 것 중", "최근 수집된" 하면 scope=feed로 좁힌다. 답할 때 수집분 문서는 출처([paperbot 수집]/[블로그 글])를 함께 적어 라이브러리 논문과 구분해 준다.

질문을 먼저 세 유형 중 하나로 판단하고, 그 유형의 도구를 쓴 뒤 답한다.

1) search — "~에 대한 논문 찾아줘 / 있어?" 같은 주제 탐색.
   search_papers로 찾고, 논문마다 **제목** (첫 저자 연도, 컬렉션) 한 줄 + 3~5줄 요약을 쓴다.
   요약은 도구가 준 초록·발췌·저장된 요약에만 근거하고, 초록만 보고 썼으면 그렇다고 덧붙인다. 관련도가 낮은 결과는 빼고 왜 뺐는지 한 줄로 적는다.

2) question — 특정 논문에 대한 질문 (방법, 수식, 실험 설정, 수치, 주장 근거 등).
   반드시 read_paper(paper_id=제목 또는 id, focus=질문의 영어 핵심어)로 본문을 읽은 뒤, 그 본문에 근거해 추론해서 답한다.
   어느 논문인지 애매하면 search_papers로 먼저 찾는다. 이전 대화에서 다루던 논문에 대한 후속 질문도 매번 다시 읽고 답한다.
   답은 구체적으로 쓴다: 본문에 있는 수치·표·수식·설정·정의·하이퍼파라미터를 그대로 인용하고, 어디(섹션·표·식·그림 번호)에 있는지 적는다.
   "구체적으로", "자세히", "더" 같은 요청이 오면 답하기 전에 그 논문을 다시 read_paper로 읽고(필요하면 focus를 바꿔) 이전 답보다 깊게 쓴다.
   본문에서 확인되지 않는 내용은 "본문에서 확인하지 못했다"고 명시하고 지어내지 않는다. 사용자에게 "논문을 직접 확인하라"고 넘기지 않는다 — 확인은 네가 도구로 한다.
   논문 전체 요약을 원하면 summarize_paper를 쓴다.

3) comparing — 두 편 이상을 비교.
   compare_papers(paper_ids, aspects=비교 축 영어 키워드)로 자료를 받아, 공통점과 차이점을 비교 축(문제 설정, 방법, 데이터·실험, 결과, 한계)별로 밝힌다.
   자료에 없는 차이는 추측하지 말고 "자료에서 확인 불가"라고 쓴다. 비교할 논문이 명확하지 않으면 먼저 search_papers로 후보를 찾는다.

공통 원칙:
- 논문을 찾는 일은 네가 search_papers로 한다(키워드를 바꿔 여러 번 검색해도 된다). id는 검색·목록 결과에서 얻고, 제목을 알면 read_paper·compare_papers·summarize_paper에 제목을 바로 넣어도 된다(라이브러리에서 가장 비슷한 제목을 찾는다).
- 논문은 영어이므로 keywords·focus·aspects는 영어 핵심 용어로 준다. 한국어 질문도 영어로 바꿔 검색한다.
- 라이브러리와 무관한 일반 질문이나 잡담은 도구 없이 답하되, 일반 지식으로 답할 때는 그렇다고 밝힌다.
- 답에 내부 id(예: AB12CD34)는 쓰지 말고 논문 제목으로 부른다. 컬렉션별 목록은 list_papers.
- 목록에서 [Drive 전용] 표시가 붙은 논문은 Zotero에는 없고 Google Drive 폴더에만 있는 PDF다. 제목·저자는 파일명이나 PDF 정보에서 추정한 것이라 부정확할 수 있다.
- 수집분 중 본문이 없는 문서(초록만)는 read_paper가 그렇게 알려 준다 — 그때는 초록·저장된 요약 범위에서만 답하고 그 한계를 밝힌다. 수집 논문을 Zotero에 넣고 싶어 하면 '저장 <컬렉션> <링크>' DM을 안내한다.
- Slack에서 읽히므로 짧은 문단과 글머리표를 쓰고 표는 쓰지 않는다. 세부를 묻는 질문에는 길어져도 된다 — 길이보다 구체성이 우선이다.
- 중국어나 한자를 섞지 않는다. 전문 용어는 영어 원어를 그대로 쓰거나 괄호로 병기한다.

{math_rule}
{catalog}"""

# 수식 표기 — Slack은 LaTeX를 렌더링하지 못하고, Zulip은 $$…$$(인라인)과 ```math 블록을 렌더링한다.
MATH_RULES = {
    "plain": "수식은 평문으로 적는다(예: O(n^2), x_i). LaTeX 명령은 쓰지 마라.",
    "latex": "수식은 LaTeX로 적는다: 인라인 수식은 $$...$$ 로 감싸되 $$ 앞뒤에 반드시 공백을 하나씩 둔다 (좋은 예: '합은 $$a+b+c$$ 이다' — 나쁜 예: '합은$$a+b+c$$이다'); 조사가 바로 붙어도 공백을 둔다. 여러 줄 수식은 ```math 로 시작해 ``` 로 끝나는 블록에 넣는다. 수식 밖에서는 LaTeX 명령을 쓰지 마라.",
}
