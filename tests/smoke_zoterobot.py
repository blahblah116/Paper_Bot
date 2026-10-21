"""zoterobot offline smoke test — no network, Slack, Zotero or Ollama needed.
zoterobot 오프라인 스모크 테스트 — 리포 루트에서 `python -m tests.smoke_zoterobot` (전부 PASS가 정상).
"""
import json
import os
import sys
import tempfile
from dataclasses import replace

import pymupdf

from paperbot.models import Paper
from paperbot.publisher import format_slack_main

from zoterobot import agent as _ag
from zoterobot import gpu as _gpu
from zoterobot import handlers as _h
from zoterobot import library as _lib
from zoterobot import listener as _lis
from zoterobot import messages as _msg
from zoterobot.config import ChatConfig, Config, ConfigError, LibraryConfig, load_config
from zoterobot.resolver import PaperRef, Resolved, clean_slack_text, find_refs, first_ref
from zoterobot.session import Conversations
from zoterobot.sync import drive_meta, sync_library
from zoterobot.textcache import DRIVE_TYPE, TextCache, norm

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from zoterobot.zotero import (build_item, first_creator, folder_name, has_pdf, item_matches,
                              link_path, pdf_filename)

ok = 0
fail = 0


def check(name, cond):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {name}")
    else:
        fail += 1
        print(f"  FAIL  {name}")


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


TMP = tempfile.mkdtemp(prefix="zoterobot-test-")
CFG = Config(allowed_users=["UME"],
             library=LibraryConfig(drive_remote="gdrive:paper", link_base="attachments:", tag="paperbot",
                                   filename_lang="ko"),
             chat=ChatConfig(text_db=os.path.join(TMP, "t.db"), chatlog_dir=os.path.join(TMP, "chatlog"),
                             max_tool_steps=3),
             base_dir=TMP)

# ================================================================ 식별자·명령 파싱
print("== resolver / 명령 파싱 ==")


def refs_of(s):
    return [r for _, _, r in find_refs(clean_slack_text(s))]


check("slack 링크 언랩+멘션 제거",
      clean_slack_text("<@U1> GDL <https://arxiv.org/abs/2505.13631v2|arxiv.org/abs/2505.13631v2> &amp;")
      == "  GDL https://arxiv.org/abs/2505.13631v2 &")
check("arXiv abs URL(버전 제거)", refs_of("<https://arxiv.org/abs/2505.13631v2>") == [PaperRef("arxiv", "2505.13631")])
check("arXiv pdf URL", refs_of("https://arxiv.org/pdf/2505.13631.pdf") == [PaperRef("arxiv", "2505.13631")])
check("arXiv 구형 id", refs_of("https://arxiv.org/abs/cs/0303001v1") == [PaperRef("arxiv", "cs/0303001")])
check("arXiv: 접두어", refs_of("see arXiv: 2401.12345") == [PaperRef("arxiv", "2401.12345")])
check("doi.org URL(소문자·꼬리 괄호 제거)",
      refs_of("(https://doi.org/10.1145/ABC.123)") == [PaperRef("doi", "10.1145/abc.123")])
check("arXiv 자동 DOI → arxiv", refs_of("https://doi.org/10.48550/arXiv.2401.12345") == [PaperRef("arxiv", "2401.12345")])
check("S2 URL", refs_of("https://www.semanticscholar.org/paper/Some-Title/" + "a" * 40) == [PaperRef("s2", "a" * 40)])
check("맨 id", refs_of("2505.13631 저장") == [PaperRef("arxiv", "2505.13631")])
check("식별자 없음", refs_of("그래프 트랜스포머 논문") == [])
check("S2 조회 id 형식", PaperRef("arxiv", "2505.13631").s2_id == "arXiv:2505.13631"
      and PaperRef("doi", "10.1/x").s2_id == "DOI:10.1/x")
check("명령: 컬렉션만", _msg.parse_command("<@UBOT> GDL") == ([], "GDL"))
check("명령: 띄어쓴 컬렉션 + 링크",
      _msg.parse_command("<@UBOT> Graph Transformer <https://arxiv.org/abs/2505.13631>")
      == ([PaperRef("arxiv", "2505.13631")], "Graph Transformer"))
check("명령: 'arXiv: id'는 컬렉션명에서 제외",
      _msg.parse_command("<@UBOT> GDL arXiv: 2505.13631") == ([PaperRef("arxiv", "2505.13631")], "GDL"))
check("명령: 멘션만 → 빈 컬렉션(사용법)", _msg.parse_command("<@UBOT>")[1] == "")
_main = format_slack_main(paper(), "### 1. 한 줄 요약\n요약", False)
check("원글 → arXiv 식별자", first_ref(clean_slack_text(_main)) == PaperRef("arxiv", "2401.12345"))
check("원글 → 제목 힌트", _msg.TITLE_RE.search(_main).group(1) == paper().title)

# ================================================================ Zotero 항목·파일명
print("== zotero: 파일명·경로·항목 ==")
check("firstCreator 1/2/3+",
      first_creator(["Alice Kim"]) == "Kim" and first_creator(["A Kim", "B Lee"]) == "Kim and Lee"
      and first_creator(["A Kim", "B Lee", "C Park"]) == "Kim et al.")
check("파일명: Zotero 기본 형식 + 금지문자 제거",
      pdf_filename(paper(title="GraphGPS: General, Powerful, Scalable Graph Transformers"))
      == "A et al. - 2026 - GraphGPS General, Powerful, Scalable Graph Transformers.pdf")
check("파일명: 한국어 Zotero 형식(등/및), 100자 절단 후 금지문자 제거",
      pdf_filename(paper(title="UrbanFusion: Stochastic Multimodal Fusion for Contrastive Learning of Robust Spatial Representations"), "ko")
      == "A 등 - 2026 - UrbanFusion Stochastic Multimodal Fusion for Contrastive Learning of Robust Spatial Representations.pdf"
      and first_creator(["Pol Rubio-Madrigal", "Rebekka Burkholz"], "ko") == "Rubio-Madrigal 및 Burkholz")
check("파일명: 제목 100자 절단", len(pdf_filename(paper(title="x" * 300, authors=["A B"]))) == len("B - 2026 - ") + 100 + 4)
check("링크 경로: 상대(attachments:)", link_path("attachments:", "GDL", "a.pdf") == "attachments:GDL/a.pdf")
check("링크 경로: 절대", link_path("G:/내 드라이브/Zotero", "GDL", "a.pdf") == "G:/내 드라이브/Zotero/GDL/a.pdf")
check("폴더명: '/'는 하위 폴더, 금지문자 제거", folder_name("Research/G:DL") == "Research/GDL")
it = build_item(paper(venue=None), pub_types=[], pub_date="2026-05-01", journal=None, collection_key="C1", tag="paperbot")
check("항목: venue 없는 arXiv → preprint",
      it["itemType"] == "preprint" and it["archiveID"] == "arXiv:2401.12345"
      and it["DOI"] == "10.48550/arXiv.2401.12345" and it["date"] == "2026-05-01")
check("항목: 저자 분리·태그·컬렉션",
      it["creators"][0] == {"creatorType": "author", "firstName": "Alice", "lastName": "A"}
      and it["tags"] == [{"tag": "paperbot"}] and it["collections"] == ["C1"])
it = build_item(paper(venue="arXiv.org"), pub_types=[], pub_date=None, journal=None, collection_key=None, tag="")
check("항목: venue 'arXiv.org'는 venue 아님 → preprint", it["itemType"] == "preprint" and it["date"] == "2026")
it = build_item(paper(), pub_types=["Conference"], pub_date=None, journal=None, collection_key=None, tag="")
check("항목: Conference → conferencePaper + extra arXiv",
      it["itemType"] == "conferencePaper" and it["proceedingsTitle"] == "ICLR" and it["extra"] == "arXiv: 2401.12345")
it = build_item(paper(venue="Nature", arxiv_id=None, uid="doi:10.1/x", doi="10.1/x"),
                pub_types=["JournalArticle"], pub_date=None, journal="Nature", collection_key=None, tag="")
check("항목: JournalArticle → journalArticle", it["itemType"] == "journalArticle" and it["publicationTitle"] == "Nature")
it = build_item(paper(venue="Neural Information Processing Systems"), pub_types=["JournalArticle"],
                pub_date=None, journal="ArXiv", collection_key=None, tag="")
check("항목: S2 학회 논문 오분류(JournalArticle+journal ArXiv) → conferencePaper", it["itemType"] == "conferencePaper")
check("중복 판정: DOI", item_matches({"DOI": "10.1/X", "title": "other"}, paper(doi="10.1/x")))
check("중복 판정: arXiv id", item_matches({"archiveID": "arXiv:2401.12345", "title": "other"}, paper()))
check("중복 판정: 정규화 제목", item_matches({"title": paper().title.upper() + "!"}, paper(arxiv_id=None)))
check("중복 판정: 다른 논문", not item_matches({"title": "Different"}, paper(arxiv_id=None)))
check("PDF 첨부 판정", has_pdf([{"itemType": "attachment", "contentType": "application/pdf"}])
      and not has_pdf([{"itemType": "note"}, {"itemType": "attachment", "contentType": "text/html"}]))

# ================================================================ 저장 흐름
print("== library: 저장 흐름 ==")


class FakeZot:
    def __init__(self, existing=None, kids=()):
        self.existing, self.kids, self.calls = existing, list(kids), []

    def collections(self):
        return [{"key": "C1", "name": "GDL"}]

    def find_or_create_collection(self, name):
        self.calls.append(("coll", name))
        return ({"key": "C1", "name": "GDL"}, False) if name.lower() == "gdl" else ({"key": "C9", "name": name}, True)

    def find_existing(self, p):
        return self.existing

    def add_to_collection(self, data, key):
        self.calls.append(("add", data["key"], key))

    def children(self, key):
        return self.kids

    def create_item(self, item):
        self.calls.append(("item", item))
        return {"key": "I1", **item}

    def create_linked_pdf(self, key, path):
        self.calls.append(("link", key, path))


class FakeDrive:
    remote = "gdrive:paper"

    def __init__(self, pdf_text=None):
        self.up, self.down, self.pdf_text = [], [], pdf_text

    def upload(self, local, rel):
        self.up.append(rel)

    def download(self, rel, local):
        self.down.append(rel)
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((72, 72), self.pdf_text or "empty")
        doc.save(local)

    def target(self, rel):
        return f"{self.remote}/{rel}"


_res = Resolved(paper=paper(venue=None), pub_types=[], pub_date=None)
_orig_fetch = _lib.fetch_pdf
_lib.fetch_pdf = lambda c, p, dest: True
try:
    z, d = FakeZot(), FakeDrive()
    r = _lib.save_paper(CFG, z, d, _res, "gdl")
    _fname = pdf_filename(_res.paper, "ko")
    check("저장: 새 항목 → 컬렉션 지정·업로드·링크",
          r.pdf == "uploaded" and not r.existed and r.collection == "GDL" and d.up == [f"GDL/{_fname}"]
          and ("link", "I1", f"attachments:GDL/{_fname}") in z.calls
          and [c for c in z.calls if c[0] == "item"][0][1]["collections"] == ["C1"])
    check("저장: 결과 메시지", "Drive `gdrive:paper/GDL/" in _msg.format_result(r, d))
    r = _lib.save_paper(CFG, FakeZot(), FakeDrive(), _res, "New Coll")
    check("저장: 없는 컬렉션 → 새로 만듦", r.collection_created and "(새로 만듦)" in _msg.format_result(r, FakeDrive()))
    z, d = FakeZot(existing={"key": "OLD", "version": 3, "collections": []},
                   kids=[{"itemType": "attachment", "contentType": "application/pdf"}]), FakeDrive()
    r = _lib.save_paper(CFG, z, d, _res, "GDL")
    check("저장: 기존 항목+PDF 있음 → 컬렉션만 추가", r.pdf == "already" and d.up == [] and ("add", "OLD", "C1") in z.calls)
    z, d = FakeZot(existing={"key": "OLD", "version": 3, "collections": []}), FakeDrive()
    r = _lib.save_paper(CFG, z, d, _res, "GDL")
    check("저장: 기존 항목+PDF 없음 → 업로드 후 기존 항목에 링크",
          r.pdf == "uploaded" and any(c[:2] == ("link", "OLD") for c in z.calls) and not any(c[0] == "item" for c in z.calls))
    _lib.fetch_pdf = lambda c, p, dest: False
    z, d = FakeZot(), FakeDrive()
    r = _lib.save_paper(CFG, z, d, _res, "GDL")
    check("저장: PDF 없음 → 메타데이터만", r.pdf == "unavailable" and d.up == [] and not any(c[0] == "link" for c in z.calls))
finally:
    _lib.fetch_pdf = _orig_fetch

# ================================================================ 본문 캐시
print("== textcache: 동기화·검색·읽기 ==")


class LibZot:
    """라이브러리 조회용 가짜 Zotero: 항목 3개 (저장소 본문 색인 / Drive 링크 PDF / 첨부 없음)."""

    def __init__(self):
        self.version = 10
        self.items = [
            {"key": "AAAA1111", "version": 5, "itemType": "conferencePaper",
             "title": "Rethinking Graph Transformers with Spectral Attention",
             "creators": [{"firstName": "Devin", "lastName": "Kreuzer"}, {"firstName": "D", "lastName": "Beaini"}],
             "date": "2021", "abstractNote": "We propose SAN, a learned positional encoding.",
             "collections": ["C1"], "url": "https://arxiv.org/abs/2106.03893"},
            {"key": "BBBB2222", "version": 7, "itemType": "preprint", "title": "How Powerful are Graph Neural Networks?",
             "creators": [{"name": "Keyulu Xu"}], "date": "2019-02-22", "abstractNote": "GIN.", "collections": ["C2"]},
            {"key": "CCCC3333", "version": 2, "itemType": "journalArticle", "title": "Metadata Only Paper",
             "creators": [], "date": "", "abstractNote": "", "collections": []},
        ]
        self.atts = [
            {"key": "ATT1", "version": 1, "parentItem": "AAAA1111", "linkMode": "imported_url",
             "contentType": "application/pdf"},
            {"key": "ATT2", "version": 1, "parentItem": "BBBB2222", "linkMode": "linked_file",
             "contentType": "application/pdf", "path": "attachments:WhyGNN/Xu 등 - 2019 - How Powerful.pdf"},
        ]
        self.fulltext_calls = []

    def library_version(self, since=None):
        return None if since == self.version else self.version

    def top_items(self):
        return list(self.items), self.version

    def attachments(self):
        return list(self.atts)

    def collections(self):
        return [{"key": "C1", "name": "GraphTransformer"}, {"key": "C2", "name": "WhyGNN"}]

    def fulltext(self, key):
        self.fulltext_calls.append(key)
        if key == "ATT1":
            return ("We use the full Laplacian spectrum. Laplacian eigenvectors serve as positional "
                    "encoding in a graph transformer. " * 20)
        return None

    def download_file(self, key, dest):
        raise AssertionError("저장소 파일은 본문 색인이 있으면 받지 않아야 함")


lz, ld = LibZot(), FakeDrive(pdf_text="Graph Isomorphism Network message passing Weisfeiler-Lehman test power")
cache = TextCache(CFG.chat.text_db)
st = sync_library(cache, lz, ld, "attachments:")
check("동기화: 3편 추가", st.get("added") == 3 and len(cache.docs()) == 3)
d1, d2, d3 = cache.get("AAAA1111"), cache.get("bbbb2222"), cache.get("CCCC3333")
check("동기화: 본문 색인 우선(저장소 파일은 다운로드 안 함)", d1.text_source == "zotero_fulltext" and "Laplacian" in d1.text)
check("동기화: Drive 링크 PDF는 rclone 경로로 받아 추출",
      d2.text_source == "pdf" and "Isomorphism" in d2.text and ld.down == ["WhyGNN/Xu 등 - 2019 - How Powerful.pdf"])
check("동기화: 첨부 없음 → 메타데이터만", d3.text_source == "none" and d3.text == "")
check("동기화: 메타데이터(저자·연도·컬렉션)",
      d1.authors == ["Devin Kreuzer", "D Beaini"] and d1.year == "2021" and d1.collections == ["GraphTransformer"]
      and d2.authors == ["Keyulu Xu"] and d2.year == "2019")
check("동기화: 버전 같으면 요청 1건으로 끝", sync_library(cache, lz, ld, "attachments:") == {"unchanged": True})
lz.version = 11
lz.items[0] = {**lz.items[0], "collections": ["C1", "C2"]}   # 버전 그대로 → 본문 재추출 없이 메타만
lz.items = lz.items[:2]                                         # 항목 삭제
lz.atts[1] = {**lz.atts[1], "version": 2}                       # 첨부 변경 → 재추출
ld.down.clear()
st = sync_library(cache, lz, ld, "attachments:")
check("증분: 변경 첨부만 재추출·삭제 반영",
      st["updated"] == 1 and st["removed"] == 1 and st["meta_only"] == 1 and ld.down and cache.get("CCCC3333") is None)
check("증분: 컬렉션 메타 갱신", cache.get("AAAA1111").collections == ["GraphTransformer", "WhyGNN"])
from datetime import datetime as _dt, timezone as _tz
cache._last_check = _dt(2999, 1, 1, tzinfo=_tz.utc)
_due_before = cache.check_due(10)
cache.mark_stale()
check("mark_stale: 저장 직후 다음 DM에서 바로 동기화", not _due_before and cache.check_due(10))
check("정규화: 하이픈·대소문자", norm("Weisfeiler-Lehman\nTest") == "weisfeiler lehman test")

hits = cache.search(["Laplacian eigenvectors", "positional encoding"])
check("검색: 본문 키워드 → 정답 1위", hits and hits[0]["id"] == "AAAA1111" and hits[0]["matches"])
check("검색: 하이픈 무시(weisfeiler lehman)", [h["id"] for h in cache.search(["Weisfeiler Lehman"])] == ["BBBB2222"])
check("검색: 컬렉션 한정",
      {h["id"] for h in cache.search(["graph"])} == {"AAAA1111", "BBBB2222"}
      and [h["id"] for h in cache.search(["graph"], collection="graphtransformer")] == ["AAAA1111"])
check("검색: 결과 없음", cache.search(["diffusion model"]) == [])
check("검색: 스니펫", "laplacian" in hits[0]["snippet"])
r = cache.read("AAAA1111", focus="eigenvectors")
check("읽기: 헤더 + focus 발췌", "제목: Rethinking" in r and "eigenvectors" in r and "발췌" in r)
check("읽기: 없는 id 안내", "없습니다" in cache.read("ZZZZ9999"))
check("읽기: 본문 없음 안내", "본문 텍스트 없음" in cache.read("CCCC3333") or cache.get("CCCC3333") is None)
check("목록: 컬렉션별", "How Powerful" in cache.list_papers("WhyGNN") and "없습니다" in cache.list_papers("CAD"))
check("카탈로그 줄 형식", cache.get("BBBB2222").line() == "BBBB2222 | How Powerful are Graph Neural Networks? | Keyulu Xu 2019 | WhyGNN")

# ================================================================ 에이전트
print("== agent: 도구 호출 루프 ==")
agent = _ag.Agent(CFG.chat, cache)
sp = agent.system_prompt()
check("시스템 프롬프트: 카탈로그 포함", "AAAA1111 | Rethinking Graph Transformers" in sp and "2편" in sp)
check("시스템 프롬프트: 큰 라이브러리는 목록 생략",
      "생략" in _ag.Agent(replace(CFG.chat, catalog_max=1), cache).system_prompt())

script = [
    ("", [{"function": {"name": "search_library", "arguments": {"keywords": ["Laplacian eigenvectors"]}}}]),
    ("", [{"function": {"name": "read_paper", "arguments": json.dumps({"paper_id": "AAAA1111", "focus": "spectrum"})}}]),
    ("찾으시는 논문은 **Rethinking Graph Transformers with Spectral Attention** 입니다.", []),
]
seen_msgs = []


def fake_chat(self, messages, use_tools, on_partial=None):
    seen_msgs.append((len(messages), use_tools))
    content, calls = script.pop(0)
    if content and on_partial:
        on_partial(content)
    return content, calls, {"load_duration": 1e9 if len(seen_msgs) == 1 else 0}


statuses, partials = [], []
_orig_chat = _ag.Agent._chat
_ag.Agent._chat = fake_chat
try:
    res = agent.run("라플라시안 고유벡터 쓰는 논문?", [], statuses.append, partials.append)
    check("에이전트: 검색→읽기→답 (3단계)", res.steps == 3 and [t["name"] for t in res.tools] == ["search_library", "read_paper"])
    check("에이전트: 문자열 arguments 파싱", res.tools[1]["args"] == {"paper_id": "AAAA1111", "focus": "spectrum"})
    check("에이전트: 상태 표시", statuses[0].startswith("🔎") and "Rethinking" in statuses[1])
    check("에이전트: 근거 논문 = 읽은 논문", res.papers == ["AAAA1111"])
    check("에이전트: 로딩 시간 합산·스트리밍", res.load_s == 1.0 and partials and "Rethinking" in res.answer)
    check("에이전트: 도구 결과가 메시지에 누적", seen_msgs[-1][0] == 2 + 1 + 1 + 1 + 1)  # sys,user,(asst,tool)x2
    # 도구만 계속 부르는 모델 → 마지막 라운드는 도구 없이 강제
    seen_msgs.clear()
    script[:] = [("", [{"function": {"name": "list_papers", "arguments": {}}}])] * 3 + [("목록을 정리하면…", [])]
    res = agent.run("목록", [], None, None)
    check("에이전트: max_tool_steps 후 도구 없이 답 강제",
          res.steps == 4 and seen_msgs[-1][1] is False and all(u for _, u in seen_msgs[:-1]))
    # 제목만 언급한 답 → 근거 표시
    script[:] = [("Graph Isomorphism 관련은 How Powerful are Graph Neural Networks? 입니다.", [])]
    check("에이전트: 답에 언급된 제목도 근거로", agent.run("GIN?", [], None, None).papers == ["BBBB2222"])
finally:
    _ag.Agent._chat = _orig_chat

convs = Conversations(2)
for i in range(3):
    convs.add("D1", f"q{i}", f"a{i}")
check("대화 기억: 최근 N쌍만", [m["content"] for m in convs.history("D1")] == ["q1", "a1", "q2", "a2"])
convs.reset("D1")
check("대화 기억: 초기화", convs.history("D1") == [])
_orig_ml, _orig_free = _gpu.model_loaded, _gpu.gpu_free_mib
try:
    _gpu.model_loaded = lambda c: False
    _gpu.gpu_free_mib = lambda g: [9000, 16000]
    try:
        _gpu.ensure_gpu(CFG.chat)
        check("GPU: 여유 부족 → GpuBusy", False)
    except _gpu.GpuBusy as e:
        check("GPU: 여유 부족 → GpuBusy", "8.8" in str(e))
    _gpu.model_loaded = lambda c: True
    _gpu.ensure_gpu(CFG.chat)
    check("GPU: 모델이 이미 올라가 있으면 통과", True)
finally:
    _gpu.model_loaded, _gpu.gpu_free_mib = _orig_ml, _orig_free

# PDF 첨부는 있는데 Drive에 파일이 없으면 'missing' → 6시간 뒤 라이브러리 변경 없어도 재시도
from zoterobot.drive import DriveError as _DriveError


class FailDrive(FakeDrive):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.fail = True

    def download(self, rel, local):
        if self.fail:
            self.down.append(rel)
            raise _DriveError("directory not found")
        super().download(rel, local)


lz.version = 12
lz.items.append({"key": "DDDD4444", "version": 1, "itemType": "preprint",
                 "title": "Semi-Supervised Classification with Graph Convolutional Networks",
                 "creators": [{"firstName": "Thomas", "lastName": "Kipf"}], "date": "2017",
                 "abstractNote": "GCN.", "collections": []})
lz.atts.append({"key": "ATT4", "version": 1, "parentItem": "DDDD4444", "linkMode": "linked_file",
                "contentType": "application/pdf", "path": "attachments:AISMLLM/Kipf.pdf"})
fd = FailDrive(pdf_text="spectral graph convolution renormalization trick")
sync_library(cache, lz, fd, "attachments:")
check("본문 못 받은 PDF → missing (첨부 없는 논문은 none)", cache.get("DDDD4444").text_source == "missing")
fd.down.clear()
check("재시도: 6시간 전엔 안 함", sync_library(cache, lz, fd, "attachments:") == {"unchanged": True} and fd.down == [])
cache.conn.execute("UPDATE docs SET updated_at='2000-01-01' WHERE item_key='DDDD4444'")
cache.conn.commit()
fd.fail = False
sync_library(cache, lz, fd, "attachments:")
check("재시도: 라이브러리 변경 없어도 6시간 지나면 다시 받아 성공",
      cache.get("DDDD4444").text_source == "pdf" and "renormalization" in cache.get("DDDD4444").text)

# Drive 폴더에만 있는 PDF → 'Drive 전용' 문서 (Zotero 링크가 가리키는 파일은 제외)
import unicodedata as _ud


class ScanDrive(FakeDrive):
    def __init__(self, files, texts):
        super().__init__()
        self.files, self.texts = files, texts

    def list_pdfs(self):
        return list(self.files)

    def download(self, rel, local):
        self.down.append(rel)
        doc = pymupdf.open()
        doc.new_page().insert_text((72, 72), self.texts.get(rel, "empty"))
        if rel == "loose.pdf":
            doc.set_metadata({"title": "Learning on Loose Graph Structures", "author": "Ann Lee, Bo Kim"})
        doc.save(local)


m = drive_meta("WhyGNN/Xu 등 - 2019 - How Powerful are GNNs.pdf", "FID")
check("Drive 파일명 분해(저자·연도·제목·폴더=컬렉션)",
      (m["authors"], m["year"], m["title"], m["collections"]) == (["Xu 등"], "2019", "How Powerful are GNNs", ["WhyGNN"])
      and m["url"] == "https://drive.google.com/file/d/FID/view" and m["item_type"] == DRIVE_TYPE)
goat = "GeoAI/Kong 등 - GOAT A Global Transformer on Large-scale Graphs.pdf"
sd = ScanDrive(
    files=[{"Path": _ud.normalize("NFD", "WhyGNN/Xu 등 - 2019 - How Powerful.pdf"), "Size": 1, "ModTime": "t", "ID": "X"},
           {"Path": goat, "Size": 100, "ModTime": "t1", "ID": "G1"},
           {"Path": "loose.pdf", "Size": 5, "ModTime": "t1", "ID": "L1"}],
    texts={goat: "virtual global nodes linear attention large scale", "loose.pdf": "loose structure"})
st = sync_library(cache, lz, sd, "attachments:", drive_scan=True)
dd = [d for d in cache.docs() if d.item_type == DRIVE_TYPE]
check("Drive 전용: Zotero 링크 파일(NFD 한글 경로 포함)은 제외, 나머지 2개 추가",
      st["drive"]["added"] == 2 and len(dd) == 2 and sorted(sd.down) == sorted([goat, "loose.pdf"]))
g = next(d for d in dd if "GOAT" in d.title)
lo = next(d for d in dd if d.key != g.key)
check("Drive 전용: 파일명에서 제목·저자, 폴더에서 컬렉션, 목록에 [Drive 전용]",
      g.title == "GOAT A Global Transformer on Large-scale Graphs" and g.authors == ["Kong 등"]
      and g.collections == ["GeoAI"] and g.line().endswith("GeoAI [Drive 전용]") and g.key.startswith("D"))
check("Drive 전용: PDF 메타데이터 제목·저자 우선", lo.title == "Learning on Loose Graph Structures"
      and lo.authors == ["Ann Lee", "Bo Kim"] and lo.collections == [])
check("Drive 전용: 본문 검색됨", [h["id"] for h in cache.search(["virtual global nodes"])] == [g.key])
sd.down.clear()
st = sync_library(cache, lz, sd, "attachments:", drive_scan=True)
check("Drive 전용: 변경 없으면 다시 안 받음", sd.down == [] and st["drive"]["added"] == st["drive"]["updated"] == 0)
lz.version = 13
lz.items[0] = {**lz.items[0], "version": 6}
sync_library(cache, lz, sd, "attachments:", drive_scan=True)
check("Drive 전용: Zotero 동기화가 Drive 전용 문서를 지우지 않음", len([d for d in cache.docs() if d.item_type == DRIVE_TYPE]) == 2)
sd.files = [f for f in sd.files if f["Path"] != "loose.pdf"]
lz.version = 14
lz.atts.append({"key": "ATT5", "version": 1, "parentItem": "AAAA1111", "linkMode": "linked_file",
                "contentType": "application/pdf", "path": "attachments:" + goat})
st = sync_library(cache, lz, sd, "attachments:", drive_scan=True)
check("Drive 전용: 파일 삭제·Zotero 등록되면 Drive 전용에서 빠짐",
      st["drive"]["removed"] == 2 and not [d for d in cache.docs() if d.item_type == DRIVE_TYPE])

# ================================================================ Slack 리스너
print("== listener: 라우팅·저장·DM ==")


class FakeWeb:
    def __init__(self, parent=""):
        self.posts, self.updates, self.parent = [], [], parent

    def chat_postMessage(self, **kw):
        self.posts.append(kw)
        return {"ts": f"9.{len(self.posts)}"}

    def chat_update(self, **kw):
        self.updates.append(kw)

    def reactions_add(self, **kw):
        pass

    def reactions_remove(self, **kw):
        pass

    def conversations_replies(self, **kw):
        return {"messages": [{"text": self.parent}]}


class Req:
    def __init__(self, event, eid):
        self.type, self.envelope_id, self.payload = "events_api", "env", {"event": event, "event_id": eid}


class SM:
    def send_socket_mode_response(self, r):
        pass


L = _lis.Listener(CFG, FakeWeb(), FakeZot(), FakeDrive(), cache, agent)
L.on_request(SM(), Req({"type": "message", "channel_type": "im", "channel": "D1", "user": "UME", "text": "hi", "ts": "1"}, "E1"))
L.on_request(SM(), Req({"type": "app_mention", "channel": "C1", "user": "UME", "text": "<@B> GDL", "ts": "2"}, "E2"))
L.on_request(SM(), Req({"type": "app_mention", "channel": "D1", "user": "UME", "text": "<@B> hi", "ts": "3"}, "E3"))
L.on_request(SM(), Req({"type": "message", "channel_type": "im", "channel": "D1", "bot_id": "B1", "text": "x", "ts": "4"}, "E4"))
L.on_request(SM(), Req({"type": "message", "channel_type": "im", "channel": "D1", "user": "UME", "text": "hi", "ts": "1"}, "E1"))
check("라우팅: DM→챗, 채널 멘션→저장, DM 멘션·봇 메시지·재전송 무시", L.chats.qsize() == 1 and L.jobs.qsize() == 1)
L = _lis.Listener(CFG, FakeWeb(), FakeZot(), FakeDrive(), cache, agent, bot_user_id="UBOT")
r_user = Req({"type": "message", "channel_type": "im", "channel": "D9", "user": "UME", "text": "남과의 DM", "ts": "5"}, "E5")
r_user.payload["authorizations"] = [{"user_id": "UME", "is_bot": False}]
r_bot = Req({"type": "message", "channel_type": "im", "channel": "D1", "user": "UME", "text": "봇에게", "ts": "6"}, "E6")
r_bot.payload["authorizations"] = [{"user_id": "UBOT", "is_bot": True}]
L.on_request(SM(), r_user)
L.on_request(SM(), r_bot)
check("라우팅: 사용자 이벤트(남과의 DM)는 버리고 봇 이벤트만", L.chats.qsize() == 1 and L.chats.get()["text"] == "봇에게")

w = FakeWeb()
_lis.Listener(CFG, w, FakeZot(), FakeDrive()).handle(
    {"user": "UOTHER", "text": "<@UBOT> GDL", "channel": "C", "ts": "2.0", "thread_ts": "1.0"})
check("저장: 미허용 사용자 거절(ID 표시, 스레드 답글)", "UOTHER" in w.posts[0]["text"] and w.posts[0]["thread_ts"] == "1.0")
w = FakeWeb()
_lis.Listener(CFG, w, FakeZot(), FakeDrive()).handle({"user": "UME", "text": "<@UBOT>", "channel": "C", "ts": "2.0"})
check("저장: 컬렉션 없으면 사용법+목록", "사용법" in w.posts[0]["text"] and "`GDL`" in w.posts[0]["text"])
w = FakeWeb()
_lis.Listener(CFG, w, FakeZot(), FakeDrive()).handle({"user": "UME", "text": "<@UBOT> GDL", "channel": "C", "ts": "2.0"})
check("저장: 스레드 밖+링크 없음 → 안내", "논문을 찾지 못했습니다" in w.posts[0]["text"])
_seen = {}
_orig = (_h.resolve, _h.save_paper)
_h.resolve = lambda ref, hint=None: _seen.update(ref=ref, hint=hint) or _res
_h.save_paper = lambda c, z, d, res, name: _seen.update(name=name) or _lib.SaveResult(
    res.paper.title, "GDL", False, "I1", False, "uploaded", drive_path="GDL/x.pdf")
try:
    w = FakeWeb(parent=_main)
    _lis.Listener(CFG, w, FakeZot(), FakeDrive()).handle(
        {"user": "UME", "text": "<@UBOT> GDL", "channel": "C", "ts": "2.0", "thread_ts": "1.0"})
    check("저장: 스레드 원글 논문 → 저장 → ✅ 답글",
          _seen.get("ref") == PaperRef("arxiv", "2401.12345") and _seen.get("hint") == paper().title
          and _seen.get("name") == "GDL" and w.posts[0]["text"].startswith("✅"))
finally:
    _h.resolve, _h.save_paper = _orig


def chatlog_records():
    out = []
    for f in sorted(os.listdir(CFG.chat.chatlog_dir)):
        with open(os.path.join(CFG.chat.chatlog_dir, f), encoding="utf-8") as fh:
            out += [json.loads(line) for line in fh]
    return out


class FakeAgent:
    def __init__(self):
        self.calls = []

    def run(self, q, history, on_status=None, on_partial=None):
        self.calls.append((q, history))
        on_status("🔎 라이브러리 검색: x")
        on_partial("**굵게** 답")
        return _ag.AgentResult(answer="**굵게** 답", papers=["AAAA1111"], steps=2, tools=[{"name": "search_library"}])


_orig_gpu = _h.ensure_gpu
try:
    _h.ensure_gpu = lambda c: (_ for _ in ()).throw(_gpu.GpuBusy("GPU 1,2 여유 2.0/2.0GB"))
    w = FakeWeb()
    L = _lis.Listener(CFG, w, FakeZot(), FakeDrive(), cache, FakeAgent())
    L.handle_dm({"user": "UME", "text": "안녕", "channel": "D1", "ts": "5", "channel_type": "im"})
    check("DM: GPU 부족 → 바로 안내(스레드 아님) + 로그 busy",
          "GPU를 다른 작업이" in w.posts[0]["text"] and "thread_ts" not in w.posts[0]
          and chatlog_records()[-1]["status"] == "busy")
    _h.ensure_gpu = lambda c: None
    fa = FakeAgent()
    w = FakeWeb()
    L = _lis.Listener(CFG, w, FakeZot(), FakeDrive(), cache, fa)
    L.handle_dm({"user": "UME", "text": "라플라시안 논문?", "channel": "D1", "ts": "6", "channel_type": "im"})
    final = w.updates[-1]["text"]
    check("DM: 자리표시 → 최종 답 갱신(mrkdwn 변환 + 참고 논문 링크)",
          w.posts[0]["text"].startswith("⏳") and final.startswith("*굵게* 답")
          and "<https://arxiv.org/abs/2106.03893|Rethinking Graph Transformers" in final)
    rec = chatlog_records()[-1]
    check("DM: 채팅 로그 기록", rec["status"] == "ok" and rec["question"] == "라플라시안 논문?"
          and rec["papers"][0]["key"] == "AAAA1111" and rec["user"] == "UME")
    L.handle_dm({"user": "UME", "text": "그 논문 방법은?", "channel": "D1", "ts": "7", "channel_type": "im"})
    check("DM: 후속 질문에 이전 대화 전달", [m["content"] for m in fa.calls[-1][1]] == ["라플라시안 논문?", "**굵게** 답"])
    L.handle_dm({"user": "UME", "text": "새 대화", "channel": "D1", "ts": "8", "channel_type": "im"})
    L.handle_dm({"user": "UME", "text": "다음", "channel": "D1", "ts": "9", "channel_type": "im"})
    check("DM: '새 대화' → 기억 초기화", fa.calls[-1][1] == [] and chatlog_records()[-2]["status"] == "reset")
    class BrokenWeb(FakeWeb):
        def chat_postMessage(self, **kw):
            raise RuntimeError("channel_not_found")

    try:
        _lis.Listener(CFG, BrokenWeb(), FakeZot(), FakeDrive(), cache, FakeAgent()).handle_dm(
            {"user": "UME", "text": "전송 실패 질문", "channel": "D3", "ts": "10", "channel_type": "im"})
    except RuntimeError:
        pass
    rec = chatlog_records()[-1]
    check("DM: Slack 전송 실패도 대화 로그에 error로 남김",
          rec["status"] == "error" and rec["question"] == "전송 실패 질문" and "channel_not_found" in rec["detail"])
    w = FakeWeb()
    _lis.Listener(CFG, w, FakeZot(), FakeDrive(), cache, FakeAgent()).handle_dm(
        {"user": "UX", "text": "hi", "channel": "D2", "ts": "1", "channel_type": "im"})
    check("DM: 미허용 사용자 거절", "UX" in w.posts[0]["text"])
finally:
    _h.ensure_gpu = _orig_gpu

import threading as _th
import time as _time


class SlowCache:
    def __init__(self):
        self.calls, self.ev = 0, _th.Event()

    def check_due(self, m):
        return True

    def sync(self, *a, **k):
        self.calls += 1
        self.ev.wait(3)


sc = SlowCache()
_orig_sync = _lis.sync_library
_lis.sync_library = lambda cache, *a, **k: cache.sync(*a, **k)
L = _lis.Listener(CFG, FakeWeb(), FakeZot(), FakeDrive(), sc, None)
_t0 = _time.time()
_a, _b = L.kick_sync(), L.kick_sync()
_dt = _time.time() - _t0
sc.ev.set()
_time.sleep(0.2)
check("백그라운드 동기화: DM을 기다리게 하지 않고, 도는 중엔 중복 시작 안 함",
      _a and not _b and _dt < 0.5 and sc.calls == 1 and L.kick_sync())
sc.ev.set()
_time.sleep(0.2)
_lis.sync_library = _orig_sync

# ================================================================ config
print("== config ==")
ex = load_config(os.path.join(REPO, "zoterobot", "config.example.yaml"))
check("config.example 로드", ex.chat.model and ex.library.link_base == "attachments:" and ex.allowed_users == [])
bad = os.path.join(TMP, "bad.yaml")
with open(bad, "w", encoding="utf-8") as f:
    f.write('library:\n  drive_remote: "no-colon-path"\n')
try:
    load_config(bad)
    check("drive_remote 형식 오류 → ConfigError", False)
except ConfigError:
    check("drive_remote 형식 오류 → ConfigError", True)
okp = os.path.join(TMP, "ok.yaml")
with open(okp, "w", encoding="utf-8") as f:
    f.write('chat:\n  llm_url: "http://localhost:11434/v1"\n')
check("llm_url의 /v1 접미사 제거 (네이티브 API 사용)", load_config(okp).chat.llm_url == "http://localhost:11434")

print(f"\n결과: {ok} PASS, {fail} FAIL")
sys.exit(1 if fail else 0)
