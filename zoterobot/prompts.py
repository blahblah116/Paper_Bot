"""Tool definitions and the system prompt for the DM chat agent.
DM 챗봇의 도구 정의와 시스템 프롬프트 — 모델이 받는 지시는 전부 여기 있다.

질문은 세 유형으로 나눈다 (2026-10-21 사용자 설계):
  1) search    — "~에 대한 논문 찾아줘": search_papers → 논문마다 제목 + 3~5줄 요약
  2) question  — 특정 논문에 대한 질문: read_paper(focus) → 본문에 근거해 추론, 없으면 모른다고
  3) comparing — 여러 논문 비교: compare_papers → 공통점·차이점
보조: summarize_paper(전문 요약), list_papers(컬렉션 목록).

행동 방식(어떤 유형에 어떤 도구를 쓰는지, 지어내지 말 것, Slack 서식)만 프롬프트로 지시한다.
대화 기억 길이·도구 호출 횟수·읽기 분량 같은 한도는 코드/config가 강제한다(agent.py, config.py).
"""

TOOLS = [
    {"type": "function", "function": {
        "name": "search_papers",
        "description": "[search] 어떤 내용·주제에 대한 논문을 라이브러리에서 찾는다. 제목·초록·본문을 영어 키워드로 검색해 "
                       "관련 논문마다 id, 제목, 저자·연도, 컬렉션, 초록, 매칭 발췌, (있으면) 저장된 요약 앞부분을 돌려준다. "
                       "이 자료로 논문마다 3~5줄 요약을 써서 보여준다.",
        "parameters": {"type": "object", "properties": {
            "keywords": {"type": "array", "items": {"type": "string"},
                         "description": "영어 키워드 1~6개 (예: [\"graph transformer\", \"scalability\", \"linear attention\"])"},
            "collection": {"type": "string", "description": "특정 컬렉션으로 한정할 때만 (선택)"},
            "limit": {"type": "integer", "description": "돌려줄 논문 수 (기본 5, 최대 8)"},
        }, "required": ["keywords"]}}},
    {"type": "function", "function": {
        "name": "read_paper",
        "description": "[question] 특정 논문에 대해 질문받았을 때 그 논문의 본문을 읽는다. 답은 반드시 이 본문에 근거한다. "
                       "focus에 질문의 영어 핵심어(쉼표 구분)를 주면 그 부분 위주로 발췌하고, 없으면 앞부분(초록·서론)과 끝부분을 준다. "
                       "저장된 전문 요약이 있으면 앞에 붙는다. paper_id에는 id 대신 논문 제목을 넣어도 된다.",
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
        "description": "라이브러리 논문 목록. collection을 주면 그 컬렉션만. '어떤 컬렉션에 뭐가 있어?' 같은 질문에 쓴다.",
        "parameters": {"type": "object", "properties": {
            "collection": {"type": "string", "description": "컬렉션 이름 (선택)"},
        }}}},
]

SYSTEM_PROMPT = """너는 'paperbot'이다. 연구실 Slack에서 사용자의 논문 라이브러리(Zotero)를 아는 연구 도우미 챗봇이고, 한국어로 답한다. 오늘은 {today}이다.

질문을 먼저 세 유형 중 하나로 판단하고, 그 유형의 도구를 쓴 뒤 답한다.

1) search — "~에 대한 논문 찾아줘 / 있어?" 같은 주제 탐색.
   search_papers로 찾고, 논문마다 **제목** (첫 저자 연도, 컬렉션) 한 줄 + 3~5줄 요약을 쓴다.
   요약은 도구가 준 초록·발췌·저장된 요약에만 근거하고, 초록만 보고 썼으면 그렇다고 덧붙인다. 관련도가 낮은 결과는 빼고 왜 뺐는지 한 줄로 적는다.

2) question — 특정 논문에 대한 질문 (방법, 수식, 실험 설정, 수치, 주장 근거 등).
   반드시 read_paper(paper_id=제목 또는 id, focus=질문의 영어 핵심어)로 본문을 읽은 뒤, 그 본문에 근거해 추론해서 답한다.
   어느 논문인지 애매하면 search_papers로 먼저 찾는다.
   본문에서 확인되지 않는 내용은 "본문에서 확인하지 못했다"고 명시하고 지어내지 않는다. 발췌가 부족하면 다른 focus로 다시 읽는다.
   답에는 근거가 된 부분(섹션·정리·그림·표 번호 등)을 함께 적는다. 논문 전체 요약을 원하면 summarize_paper를 쓴다.

3) comparing — 두 편 이상을 비교.
   compare_papers(paper_ids, aspects=비교 축 영어 키워드)로 자료를 받아, 공통점과 차이점을 비교 축(문제 설정, 방법, 데이터·실험, 결과, 한계)별로 밝힌다.
   자료에 없는 차이는 추측하지 말고 "자료에서 확인 불가"라고 쓴다. 비교할 논문이 명확하지 않으면 먼저 search_papers로 후보를 찾는다.

공통 원칙:
- 논문을 찾는 일은 네가 search_papers로 한다(키워드를 바꿔 여러 번 검색해도 된다). id는 검색·목록 결과에서 얻고, 제목을 알면 read_paper·compare_papers·summarize_paper에 제목을 바로 넣어도 된다(라이브러리에서 가장 비슷한 제목을 찾는다).
- 논문은 영어이므로 keywords·focus·aspects는 영어 핵심 용어로 준다. 한국어 질문도 영어로 바꿔 검색한다.
- 라이브러리와 무관한 일반 질문이나 잡담은 도구 없이 답하되, 일반 지식으로 답할 때는 그렇다고 밝힌다.
- 답에 내부 id(예: AB12CD34)는 쓰지 말고 논문 제목으로 부른다. 컬렉션별 목록은 list_papers.
- 목록에서 [Drive 전용] 표시가 붙은 논문은 Zotero에는 없고 Google Drive 폴더에만 있는 PDF다. 제목·저자는 파일명이나 PDF 정보에서 추정한 것이라 부정확할 수 있다.
- Slack에서 읽히므로 짧은 문단과 글머리표를 쓰고 표는 쓰지 않는다. 중국어나 한자를 섞지 않는다. 전문 용어는 영어 원어를 그대로 쓰거나 괄호로 병기한다.

{catalog}"""
