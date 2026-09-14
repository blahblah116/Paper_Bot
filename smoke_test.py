"""v2 오프라인 스모크 테스트 — 네트워크/Ollama/Zulip 불필요."""
import io
import os
import sys
import tempfile
from contextlib import redirect_stdout

from paperbot.config import load_config, ConfigError
from paperbot.models import Paper, make_uid, arxiv_base_id
from paperbot.store import Store
from paperbot.extractor import strip_references, truncate, _TRUNC_MARK, extract
from paperbot.publisher import (
    format_message, _topic_for, DryRunPublisher, SlackPublisher,
    extract_one_liner, md_to_mrkdwn, format_slack_main, make_publisher,
)
from paperbot.prompts import build_summary_system_prompt, build_judge_user_prompt, JUDGE_SYSTEM_PROMPT, SECTIONS
from paperbot.filter import _parse_judge_json
from paperbot.sources.semantic_scholar import SemanticScholarSource

ok = 0
fail = 0
def check(name, cond):
    global ok, fail
    if cond: ok += 1; print(f"  PASS  {name}")
    else:    fail += 1; print(f"  FAIL  {name}")

def paper(**kw):
    base = dict(
        uid="arxiv:2401.12345",
        title="A Very Long Title That Definitely Exceeds Sixty Characters For Topic Truncation Testing",
        authors=["Alice A", "Bob B", "Carol C", "Dan D"],
        abstract="This is the abstract about graph transformers.",
        venue="ICLR", year=2026,
        url="https://arxiv.org/abs/2401.12345",
        pdf_url="https://arxiv.org/pdf/2401.12345",
        arxiv_id="2401.12345", doi=None, citation_count=42,
        source="semantic_scholar", topic_name="graph-transformer",
    )
    base.update(kw)
    return Paper(**base)

print("== models: uid 규칙 ==")
check("arxiv 우선", make_uid("2401.12345v3", "10.1234/real.doi", "S2ID") == "arxiv:2401.12345")
check("doi 차선", make_uid(None, "10.1234/Real.DOI", "S2ID") == "doi:10.1234/real.doi")
check("s2 최후", make_uid(None, None, "abc123") == "s2:abc123")
check("arXiv 자동 DOI 정규화", make_uid(None, "10.48550/arXiv.2401.12345", "S2ID") == "arxiv:2401.12345")
check("base_id 구형", arxiv_base_id("cs/0303001v2") == "cs/0303001")
check("embed_text abstract 포함", "abstract" in paper().embed_text)
check("embed_text title만(무abstract)", paper(abstract=None).embed_text == paper().title)

print("== config v2 ==")
cfg = load_config(os.path.join(os.path.dirname(__file__), "config.yaml"))
check("topics 1개 이상 로드(개수는 사용자 튜닝 영역)", len(cfg.topics) >= 1)
check("interest 로드", "Graph Transformer" in cfg.topics[0].interest)
check("s2 venues", cfg.topics[0].s2.venues == ["ICLR", "NeurIPS", "ICML", "KDD"])
check("s2 year 로드(값은 사용자 튜닝 영역)", isinstance(cfg.topics[0].s2.year, str) and cfg.topics[0].s2.year)
check("s2 sort 기본값", all(t.s2.sort in ("recency", "citations") for t in cfg.topics if t.s2))
# 잘못된 sort → ConfigError
bad_sort = """
topics:
  - name: "t"
    interest: "x"
    sources:
      semantic_scholar: {query: "q", sort: "random"}
"""
try:
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(bad_sort); badp2 = f.name
    load_config(badp2); check("잘못된 sort → ConfigError", False)
except ConfigError:
    check("잘못된 sort → ConfigError", True)
finally:
    os.unlink(badp2)
check("arxiv enabled", cfg.topics[0].arxiv.enabled is True)
check("filter 로드(값은 사용자 튜닝 영역)",
      0.0 <= cfg.filter.embedding_threshold <= 1.0 and 0.0 <= cfg.filter.judge_threshold <= 10.0)
check("llm num_ctx 로드(값은 사용자 튜닝 영역)", cfg.llm.num_ctx >= 4096)
check("llm 출력 상한 로드", 256 <= cfg.llm.max_output_tokens < cfg.llm.num_ctx)
from paperbot.summarizer import _clean_output
check("완결 think 제거", _clean_output("<think>추론</think>답변") == "답변")
check("잘린 think 제거(미완)", _clean_output("<think>추론이 잘렸") == "")
check("think 없는 출력 유지", _clean_output("그냥 답변") == "그냥 답변")

# interest 누락 → ConfigError
bad_yaml = """
topics:
  - name: "t"
    sources:
      arxiv: {enabled: true, query: "cat:cs.LG"}
"""
try:
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(bad_yaml); badp = f.name
    load_config(badp); check("interest 누락 → ConfigError", False)
except ConfigError:
    check("interest 누락 → ConfigError", True)
finally:
    os.unlink(badp)

print("== S2 전역 rate limiter ==")
import time as _time
from paperbot.sources.semantic_scholar import _GlobalRateLimiter, _RateLimited
_rl = _GlobalRateLimiter()
_t0 = _time.monotonic()
_rl.wait(0.15); _rl.wait(0.15); _rl.wait(0.15)   # 3회 → 최소 0.3s 간격 누적
_elapsed = _time.monotonic() - _t0
check("시작 간격 보장(>=0.3s)", _elapsed >= 0.3)
check("과도 대기 없음(<1s)", _elapsed < 1.0)
check("_RateLimited retry_after 보존", _RateLimited(7.5).retry_after == 7.5)
check("_RateLimited 없음 표현", _RateLimited(None).retry_after is None)

print("== S2 매핑 ==")
item = {
    "paperId": "abc", "title": "  Test   Paper  ", "abstract": "An abstract.",
    "venue": "ICLR", "year": 2025,
    "externalIds": {"ArXiv": "2405.00001v2", "DOI": "10.48550/arXiv.2405.00001"},
    "openAccessPdf": {"url": "https://openreview.net/pdf?id=x"},
    "citationCount": 7, "authors": [{"name": "A"}, {"name": "B"}],
    "publicationDate": "2025-05-01", "url": "https://semanticscholar.org/paper/abc",
}
p = SemanticScholarSource._to_paper(item, "graph-transformer")
check("uid=arxiv 정규화", p.uid == "arxiv:2405.00001")
check("제목 공백 정리", p.title == "Test Paper")
check("pdf: openAccess 우선", p.pdf_url == "https://openreview.net/pdf?id=x")
check("url: arXiv abs 우선", p.url == "https://arxiv.org/abs/2405.00001")
item2 = {**item, "externalIds": {}, "openAccessPdf": None}
p2 = SemanticScholarSource._to_paper(item2, "t")
check("uid=s2 폴백", p2.uid == "s2:abc")
check("pdf 없음 → None", p2.pdf_url is None)
check("빈 title → None", SemanticScholarSource._to_paper({"paperId": "x", "title": " "}, "t") is None)

print("== store v2 상태 전이 ==")
with tempfile.TemporaryDirectory() as d:
    st = Store(os.path.join(d, "t.db"))
    a = paper()
    check("초기 → 처리대상", st.should_process(a.uid) is True)
    st.mark_done(a)
    check("done → skip", st.should_process(a.uid) is False)
    b = paper(uid="arxiv:9999.00001")
    b.judge_score = 3.5
    st.mark_filtered(b, "judge: 관심사와 무관")
    check("filtered_out → skip (재채점 방지)", st.should_process(b.uid) is False)
    row = st.conn.execute("SELECT judge_score, status, source FROM papers WHERE uid=?", (b.uid,)).fetchone()
    check("judge_score 기록", row["judge_score"] == 3.5 and row["status"] == "filtered_out")
    c = paper(uid="doi:10.1/x")
    st.mark_failed(c, "e1")
    check("failed 1회 → 재시도", st.should_process(c.uid) is True)
    st.mark_failed(c, "e2")
    check("failed 2회 → 영구 skip", st.should_process(c.uid) is False)
    st.close()

print("== pending 대기열 (쿼터 초과분 FIFO) ==")
import time as _t
with tempfile.TemporaryDirectory() as d:
    st = Store(os.path.join(d, "p.db"))
    p1 = paper(uid="arxiv:1111.00001", source="arxiv", topic_name="t1"); p1.judge_score = 8.0
    p2 = paper(uid="arxiv:1111.00002", source="arxiv", topic_name="t1")
    p3 = paper(uid="s2:abc", source="semantic_scholar", topic_name="t1")
    check("mark_pending 신규 → True", st.mark_pending(p1) is True)
    _t.sleep(0.01); st.mark_pending(p2); st.mark_pending(p3)
    check("pending → should_process False (재채점 방지)", st.should_process(p1.uid) is False)
    check("mark_pending 중복 → False (FIFO 시각 보존)", st.mark_pending(p1) is False)
    q = st.load_pending("t1", "arxiv")
    check("load_pending FIFO 순서·소스 분리", [p.uid for p in q] == [p1.uid, p2.uid])
    check("payload 복원(judge_score·pdf_url)", q[0].judge_score == 8.0 and q[0].pdf_url == p1.pdf_url and q[0].topic_name == "t1")
    check("load_all_pending 합산", len(st.load_all_pending(["t1"])) == 3)
    check("count_pending", st.count_pending() == {("t1", "arxiv"): 2, ("t1", "semantic_scholar"): 1})
    st.mark_done(q[0])
    check("pending → done 전이", st._status(p1.uid) == "done" and len(st.load_pending("t1", "arxiv")) == 1)
    check("expire 0 = 비활성", st.expire_pending(0) == 0)
    st.conn.execute("UPDATE papers SET processed_at='2000-01-01T00:00:00+00:00' WHERE uid=?", (p2.uid,)); st.conn.commit()
    check("expire: 오래된 것만 폐기", st.expire_pending(30) == 1 and st._status(p2.uid) == "expired" and st._status(p3.uid) == "pending")
    st.close()
    # 구 스키마(payload 없음) DB 마이그레이션
    import sqlite3 as _sq
    old = os.path.join(d, "old.db"); c = _sq.connect(old)
    c.execute("CREATE TABLE papers (uid TEXT PRIMARY KEY, topic TEXT, title TEXT, source TEXT, status TEXT, judge_score REAL, processed_at TEXT, error TEXT)")
    c.execute("INSERT INTO papers VALUES ('arxiv:1', 't', 'x', 'arxiv', 'done', NULL, '2026-01-01', NULL)"); c.commit(); c.close()
    st2 = Store(old)
    check("구 DB 마이그레이션(payload 추가, 기존 행 유지)", st2.should_process("arxiv:1") is False and st2.mark_pending(paper(uid="arxiv:2")) is True)
    st2.close()

print("== 발행 쿼터 ==")
from run import apply_quota
check("quota 로드", cfg.quota.s2_per_topic == 3 and cfg.quota.arxiv_per_topic == 2)
def mkq(i, src, topic="t1"):
    return paper(uid=f"q:{topic}:{src}:{i}", source=src, topic_name=topic)
mix = ([mkq(i, "semantic_scholar") for i in range(5)] + [mkq(i, "arxiv") for i in range(4)]
       + [mkq(i, "semantic_scholar", "t2") for i in range(2)])
q = cfg.quota
out = apply_quota(mix, q)
s2_t1 = [p for p in out if p.source == "semantic_scholar" and p.topic_name == "t1"]
ax_t1 = [p for p in out if p.source == "arxiv" and p.topic_name == "t1"]
s2_t2 = [p for p in out if p.topic_name == "t2"]
check("t1 S2 상한 3", len(s2_t1) == 3)
check("t1 arXiv 상한 2", len(ax_t1) == 2)
check("t2는 독립 카운트(2편 전부)", len(s2_t2) == 2)
check("최신순(입력 순서) 유지", [p.uid for p in s2_t1] == [f"q:t1:semantic_scholar:{i}" for i in range(3)])
from paperbot.config import QuotaConfig
check("0 = 무제한", len(apply_quota(mix, QuotaConfig(0, 0))) == len(mix))

print("== filter: judge JSON 파싱 ==")
check("정상", _parse_judge_json('{"score": 8, "reason": "관련"}') == (8.0, "관련"))
check("코드블록/잡음 섞임", _parse_judge_json('설명...\n```json\n{"score": 3, "reason": "x"}\n```') == (3.0, "x"))
check("범위 밖 → None", _parse_judge_json('{"score": 99, "reason": "x"}') is None)
check("깨진 JSON → None", _parse_judge_json('score: 5') is None)

print("== extractor ==")
txt = "Intro\nMethod\n" * 30 + "References\n[1] foo\n"
check("References 제거", "[1] foo" not in strip_references(txt))
tr = truncate("A"*1000 + "B"*1000, 500)
check("truncate", len(tr) <= 500 + len(_TRUNC_MARK) and "[... truncated ...]" in tr)
# PDF 후보가 전혀 없는 경우(pdf_url·arxiv_id·doi 모두 없음) → abstract 폴백 (조회형 후보도 안 감).
no_pdf = paper(pdf_url=None, arxiv_id=None, doi=None, uid="s2:nopdf")
r = extract(cfg, no_pdf)
check("PDF 후보 없음 → abstract 폴백", r.abstract_only is True and r.text == no_pdf.abstract)
# 구조 체인 게이팅: doi 없으면 Unpaywall/제목검색 후보가 생성되지 않아야 함 (네트워크 0회)
from paperbot.extractor import _iter_candidate_urls, _norm_title
check("doi 없음 → 조회형 후보 없음", list(_iter_candidate_urls(cfg, no_pdf)) == [])
direct = paper(uid="arxiv:1", pdf_url="http://x/pdf", arxiv_id="2401.00001", doi=None)
check("직접 후보 순서(pdf→arXiv)", list(_iter_candidate_urls(cfg, direct))
      == ["http://x/pdf", "https://arxiv.org/pdf/2401.00001"])
check("제목 정규화 매칭", _norm_title("Multi-Agent CAD: Code-Generation!") == _norm_title("multi agent cad code generation"))

print("== extractor: 다중 figure ==")
import pymupdf as _pm
from paperbot.extractor import extract_figures
with tempfile.TemporaryDirectory() as d:
    # 노이즈 이미지 2장(진짜 figure 흉내: 고색상·고밀도) + 텍스트만 있는 페이지
    p1 = os.path.join(d, "t.pdf")
    doc = _pm.open()
    page = doc.new_page()
    noisy = _pm.Pixmap(_pm.csRGB, 500, 400, os.urandom(500 * 400 * 3), False)
    page.insert_image(_pm.Rect(50, 50, 350, 290), pixmap=noisy)
    page2 = doc.new_page()
    noisy2 = _pm.Pixmap(_pm.csRGB, 600, 450, os.urandom(600 * 450 * 3), False)
    page2.insert_image(_pm.Rect(40, 60, 400, 330), pixmap=noisy2)
    doc.save(p1); doc.close()
    figs = extract_figures(p1, max_figures=3)
    check("래스터 figure 추출(2장)", len(figs) == 2 and all(f.startswith(b"\x89PNG") for f in figs))
    check("max_figures=1 제한", len(extract_figures(p1, max_figures=1)) == 1)
    check("max_figures=0 → 빈 리스트", extract_figures(p1, max_figures=0) == [])
    # figure가 전혀 없는 PDF → 1페이지 렌더 폴백 1장
    p2 = os.path.join(d, "blank.pdf")
    doc = _pm.open(); pg = doc.new_page()
    pg.insert_text((72, 72), "text only paper")
    doc.save(p2); doc.close()
    figs2 = extract_figures(p2)
    check("figure 없음 → 1페이지 렌더 폴백", len(figs2) == 1)

print("== 재료 없음(본문·abstract 모두 실패) 가드 ==")
from run import process_paper
with tempfile.TemporaryDirectory() as d:
    st2 = Store(os.path.join(d, "g.db"))
    empty = paper(uid="s2:empty", abstract=None, pdf_url=None, arxiv_id=None)
    sent = []
    class _RecPub:
        def publish(self, *a, **k): sent.append(1)
    ok_flag = process_paper(cfg, st2, _RecPub(), empty, client=None)
    check("발행 안 함 + False 반환", ok_flag is False and not sent)
    check("failed 기록(1회 재시도 여지)", st2.should_process(empty.uid) is True)
    st2.close()

print("== publisher v2 포맷 ==")
pp = paper()
msg = format_message(pp, "### 요약", abstract_only=False)
check("venue+year", "ICLR 2026" in msg)
check("인용 수", "인용 42" in msg)
check("topic 태그", "`#graph-transformer`" in msg)
check("url", "📄 https://arxiv.org/abs/2401.12345" in msg)
pre = paper(venue=None, citation_count=None)
msg2 = format_message(pre, "요약", abstract_only=True)
check("프리프린트 표기", "arXiv preprint 2026" in msg2)
check("인용 없으면 생략", "인용" not in msg2)
check("abstract 표기", "abstract 기반 요약" in msg2)
check("topic 60자 절단", len(_topic_for(pp)) <= 60)
buf = io.StringIO()
with redirect_stdout(buf):
    DryRunPublisher().publish(pp, "요약본문")
check("dry-run stdout", "DRY-RUN" in buf.getvalue())

print("== publisher: slack ==")
summary_md = "### 1. 한 줄 요약\n동적 그래프에서 **SAM**을 개선한 TIDFormer를 제안한다.\n\n### 2. 문제 정의\n어쩌고"
check("한 줄 요약 추출", extract_one_liner(summary_md) == "동적 그래프에서 SAM을 개선한 TIDFormer를 제안한다.")
check("한 줄 요약 폴백(섹션 없음)", extract_one_liner("그냥 텍스트 요약") == "그냥 텍스트 요약")
long_sum = "### 1. 한 줄 요약\n" + "가" * 400
check("한 줄 요약 길이 제한", len(extract_one_liner(long_sum)) <= 300)
mr = md_to_mrkdwn("### 1. 한 줄 요약\n**볼드** 텍스트와 `code`")
check("mrkdwn: 헤딩 → 볼드", mr.startswith("*1. 한 줄 요약*"))
check("mrkdwn: **볼드** → *볼드*", "*볼드* 텍스트" in mr and "**" not in mr)
main_msg = format_slack_main(pp, summary_md, abstract_only=False)
check("slack 메인: 제목 볼드", main_msg.startswith(f"*{pp.title}*"))
check("slack 메인: 한 줄 요약 포함", "➤ 동적 그래프에서" in main_msg)
check("slack 메인: 전체 요약 미포함", "### 2" not in main_msg and "문제 정의" not in main_msg)
# dry-run slack 레이아웃
buf2 = io.StringIO()
with redirect_stdout(buf2):
    DryRunPublisher(backend="slack").publish(pp, summary_md)
out2 = buf2.getvalue()
check("dry-run slack: 메인+스레드 구분", "채널 메인 메시지" in out2 and "스레드 댓글" in out2)
# 토큰 없이 SlackPublisher 생성 → 설정 오류(실행 중단용 예외)
from paperbot.publisher import PublishConfigError
os.environ.pop("SLACK_BOT_TOKEN", None)
try:
    SlackPublisher(cfg); check("토큰 없음 → PublishConfigError", False)
except PublishConfigError as e:
    check("토큰 없음 → PublishConfigError", "SLACK_BOT_TOKEN" in str(e))

# 모델 언로드: 서버가 없어도 조용히 False (비치명적)
from dataclasses import replace as _rp
from paperbot.summarizer import unload_model
dead_cfg = _rp(cfg, llm=_rp(cfg.llm, base_url="http://localhost:59999/v1"))
check("unload_model 서버 없음 → False(무해)", unload_model(dead_cfg) is False)
# make_publisher 디스패치 (dry_run 우선)
check("make_publisher dry-run backend 반영",
      isinstance(make_publisher(cfg, force_dry_run=True), DryRunPublisher)
      and make_publisher(cfg, force_dry_run=True).backend == cfg.publisher.backend)

print("== prompts ==")
sysp = build_summary_system_prompt("ko")
check("6섹션", all(t in sysp for t, _ in SECTIONS))
jp = build_judge_user_prompt("그래프 트랜스포머", paper(abstract=None))
check("judge: abstract 없음 안내", "abstract 없음" in jp)
check("judge system: JSON 강제", '"score"' in JUDGE_SYSTEM_PROMPT)

print(f"\n결과: {ok} PASS, {fail} FAIL")
sys.exit(1 if fail else 0)
