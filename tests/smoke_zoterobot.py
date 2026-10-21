"""zoterobot offline smoke test — no network, Slack, Zotero or Ollama needed.
zoterobot 오프라인 스모크 테스트 — 리포 루트에서 `python -m tests.smoke_zoterobot` (전부 PASS가 정상).
"""
import json
import math
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
from zoterobot import summary as _sm
from zoterobot.budget import TokenBudget, split_chars
from zoterobot.sync import drive_meta, sync_library
from zoterobot.textcache import DRIVE_TYPE, TextCache, norm

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from zoterobot.zotero import (Collection, ZoteroClient, build_item, collection_path, collection_paths,
                              first_creator, folder_name, has_pdf, item_matches, link_path, pdf_filename)

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
check("컬렉션 경로 분해", collection_path(" ICML / 2026/ ") == ["ICML", "2026"] and collection_path("GDL") == ["GDL"])
_cols = [{"key": "A", "name": "ICML", "parentCollection": False}, {"key": "B", "name": "2026", "parentCollection": "A"},
         {"key": "C", "name": "project", "parentCollection": False}, {"key": "D", "name": "B-rep", "parentCollection": "C"},
         {"key": "E", "name": "CAD_agent", "parentCollection": "D"}, {"key": "F", "name": "dataset", "parentCollection": "A"}]
check("컬렉션 전체 경로", collection_paths(_cols) == {"A": "ICML", "B": "ICML/2026", "C": "project", "D": "project/B-rep",
                                                   "E": "project/B-rep/CAD_agent", "F": "ICML/dataset"})


class _StubZot(ZoteroClient):
    """네트워크 없이 find_or_create_collection 로직만 검사 — collections()/POST를 메모리로 대체."""

    def __init__(self, cols):
        super().__init__("dummy-key", "1")          # user_id가 숫자면 키 조회(네트워크) 없음
        self.cols, self.posted, self._n = [dict(c) for c in cols], [], 0

    def collections(self):
        return [dict(c) for c in self.cols]

    def _req(self, method, path, **kw):
        assert (method, path) == ("POST", "/collections"), (method, path)
        body = kw["json"][0]
        self._n += 1
        data = {"key": f"N{self._n}", "name": body["name"], "parentCollection": body["parentCollection"]}
        self.cols.append(data)
        self.posted.append((body["name"], body["parentCollection"]))

        class R:
            def json(self_inner):
                return {"successful": {"0": {"key": data["key"], "data": data}}}
        return R()


_z = _StubZot(_cols)
_c = _z.find_or_create_collection("icml/2026")
check("컬렉션 경로: 기존 하위 컬렉션 매칭(대소문자 무시, 생성 없음)",
      (_c.key, _c.path, _c.created, _z.posted) == ("B", "ICML/2026", [], []))
_c = _z.find_or_create_collection("ICML/2027")
check("컬렉션 경로: 없는 단계만 부모 아래 생성",
      (_c.key, _c.path, _c.created, _z.posted) == ("N1", "ICML/2027", ["2027"], [("2027", "A")]))
_c = _z.find_or_create_collection("New/Sub")
check("컬렉션 경로: 두 단계 모두 생성(최상위 → 하위)",
      (_c.path, _c.created, _z.posted[1:]) == ("New/Sub", ["New", "Sub"], [("New", False), ("Sub", "N2")]))
_c = _z.find_or_create_collection("2026")
check("컬렉션 경로: 이름만 적으면 최상위(하위 '2026'과 매칭 안 함)",
      (_c.path, _c.created, _z.posted[-1]) == ("2026", ["2026"], ("2026", False)))
_c = _z.find_or_create_collection("project/b-rep/CAD_agent")
check("컬렉션 경로: 3단계 기존 매칭 → 실제 저장 이름으로 경로", (_c.key, _c.path, _c.created) == ("E", "project/B-rep/CAD_agent", []))
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
        return Collection("C1", "GDL") if name.lower() == "gdl" else Collection("C9", name, collection_path(name))

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
    r = _lib.save_paper(CFG, FakeZot(), FakeDrive(), _res, "ICML/2026")
    check("저장: 하위 컬렉션 경로 → Drive 하위 폴더·링크 경로 동일",
          r.collection == "ICML/2026" and r.drive_path == f"ICML/2026/{_fname}"
          and "Drive `gdrive:paper/ICML/2026/" in _msg.format_result(r, FakeDrive()))
    r.collection_created = ["2026"]
    check("저장: 일부 단계만 새로 만든 메시지", "(하위 `2026` 새로 만듦)" in _msg.format_result(r, FakeDrive()))
    check("사용법: 컬렉션 경로 표시", "`ICML/2026`" in _msg.format_usage(["ICML", "ICML/2026"], "gdrive:paper"))
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
_full = cache.get("AAAA1111").text
r = cache.read("AAAA1111", max_chars=1000)
check("읽기: max_chars 초과 → 앞부분+끝부분, 중략 자수 표시",
      len(_full) > 1000 and "(중략" in r and f"전체 {len(_full):,}자" in r and r.endswith(_full[-200:]))
check("읽기: 기본 상한(40000) 안이면 전문 그대로", "(중략" not in cache.read("AAAA1111") and _full in cache.read("AAAA1111"))
cache.set_summary("BBBB2222", "## 요약\n- 핵심 주장", "drive:WhyGNN/x.pdf", 1234, "qwen")
_s = cache.get_summary("bbbb2222")
check("요약 저장·조회(대소문자 무시)", _s and _s["source"] == "drive:WhyGNN/x.pdf" and _s["chars"] == 1234 and _s["created_at"])
check("읽기: 저장된 요약이 본문 앞에", "[전문 요약" in cache.read("BBBB2222") and "- 핵심 주장" in cache.read("BBBB2222"))
check("읽기: 요약 없는 문서는 그대로", "[전문 요약" not in cache.read("AAAA1111"))
cache.set_summary("ZZZZ0000", "x", "zotero:A", 1); cache.delete(["ZZZZ0000"])
check("문서 삭제 시 요약도 삭제", cache.get_summary("ZZZZ0000") is None and cache.get_summary("BBBB2222"))
check("제목으로 찾기: id·정확한 제목·단어 60%·무관",
      cache.find_by_title("bbbb2222").key == "BBBB2222"
      and cache.find_by_title("how powerful are graph neural networks").key == "BBBB2222"
      and cache.find_by_title("Rethinking Graph Transformers Spectral Attention").key == "AAAA1111"
      and cache.find_by_title("completely unrelated title words") is None and cache.find_by_title("") is None)

# ================================================================ summarize_paper
print("== summary: 전문 요약 도구 ==")


class SumZot:
    def __init__(self):
        self.downloads = []

    def children(self, key):
        if key == "BBBB2222":
            return [{"itemType": "attachment", "key": "ATT2", "linkMode": "linked_file", "contentType": "application/pdf",
                     "path": "attachments:WhyGNN/Xu 등 - 2019 - How Powerful.pdf"}]
        if key == "AAAA1111":
            return [{"itemType": "note"}, {"itemType": "attachment", "key": "ATT1", "linkMode": "imported_url",
                                           "contentType": "application/pdf"}]
        return []

    def download_file(self, key, dest):
        self.downloads.append(key)
        doc = pymupdf.open()
        doc.new_page().insert_text((72, 72), "Spectral attention body text. " * 30)
        doc.save(dest)


sum_calls = []


def fake_summarize(paper, text):
    sum_calls.append((paper.uid, len(text)))
    return f"## 요약\n{paper.title} — {len(text)}자 기반 ({paper.year})"


sum_drive = FakeDrive(pdf_text="GIN body text. " * 50)
ps = _sm.PaperSummarizer(cache, SumZot(), sum_drive, "attachments:", fake_summarize, max_input_chars=400, model="fake")
sum_status = []
sr = ps.summarize("BBBB2222", force=True, on_status=sum_status.append)
check("요약: 캐시 본문으로 요약·저장 (PDF 다시 받지 않음)",
      not sr.reused and sr.source == "cache:pdf" and sum_drive.down == [] and 0 < sr.chars <= 400
      and sum_calls[-1] == ("BBBB2222", sr.chars) and cache.get_summary("BBBB2222")["summary"] == sr.summary
      and cache.get_summary("BBBB2222")["model"] == "fake" and not any(s.startswith("📥") for s in sum_status)
      and any(s.startswith("🧠") for s in sum_status))
check("요약: 저장된 요약 재사용(모델 호출 없음)", ps.summarize("BBBB2222").reused and len(sum_calls) == 1)
check("요약: force → 다시 생성", not ps.summarize("BBBB2222", force=True).reused and len(sum_calls) == 2)
# 캐시에 본문이 없는 논문 → 예비 경로: Drive 링크 PDF를 rclone으로, 저장소 PDF는 Zotero API로
_meta = {"title": "No Text Yet", "authors": ["A B"], "year": "2024", "collections": ["WhyGNN"], "abstract": "", "url": "",
         "item_type": "journalArticle"}
cache.upsert("DDDD4444", _meta, "v1", "", "missing")
cache.upsert("EEEE5555", {**_meta, "title": "No Text Either"}, "v1", "", "missing")
cache.upsert("FFFF6666", {**_meta, "title": "No Attachment"}, "v1", "", "none")
cache.invalidate()
ps.zot.children = lambda key: {
    "DDDD4444": [{"itemType": "attachment", "key": "ATT4", "linkMode": "linked_file", "contentType": "application/pdf",
                  "path": "attachments:WhyGNN/Xu 등 - 2019 - How Powerful.pdf"}],
    "EEEE5555": [{"itemType": "attachment", "key": "ATT5", "linkMode": "imported_url", "contentType": "application/pdf"}],
}.get(key, [])
sum_status.clear()
sr = ps.summarize("DDDD4444", on_status=sum_status.append)
check("요약(예비): 본문 없는 논문은 Drive 링크 PDF를 rclone으로 받아 추출",
      sr.source == "drive:WhyGNN/Xu 등 - 2019 - How Powerful.pdf" and sum_drive.down == ["WhyGNN/Xu 등 - 2019 - How Powerful.pdf"]
      and 0 < sr.chars <= 400 and any(s.startswith("📥 Drive") for s in sum_status))
sr = ps.summarize("EEEE5555")
check("요약(예비): 저장소 PDF는 Zotero API로", sr.source == "zotero:ATT5" and ps.zot.downloads == ["ATT5"])
sr = ps.summarize("AAAA1111")
check("요약: Zotero 본문 색인(zotero_fulltext)도 캐시로", sr.source.startswith("cache:") and "2021" in sr.summary)
check("요약: Paper 변환(연도·저자·컬렉션)", _sm.doc_to_paper(cache.get("AAAA1111")).year == 2021
      and _sm.doc_to_paper(cache.get("AAAA1111")).topic_name == "GraphTransformer")
try:
    ps.summarize("ZZZZ9999")
    check("요약: 없는 id → SummaryError", False)
except _sm.SummaryError:
    check("요약: 없는 id → SummaryError", True)
try:
    ps.summarize("FFFF6666")   # 본문도 PDF 첨부도 없음
    check("요약: 본문·PDF 없는 항목 → SummaryError", False)
except _sm.SummaryError as e:
    check("요약: 본문·PDF 없는 항목 → SummaryError", "찾지 못했" in str(e))
cache.delete(["DDDD4444", "EEEE5555", "FFFF6666"]); cache.invalidate()
check("읽기: 없는 id 안내", "없습니다" in cache.read("ZZZZ9999"))
check("읽기: 본문 없음 안내", "본문 텍스트 없음" in cache.read("CCCC3333") or cache.get("CCCC3333") is None)
check("목록: 컬렉션별", "How Powerful" in cache.list_papers("WhyGNN") and "없습니다" in cache.list_papers("CAD"))
check("카탈로그 줄 형식", cache.get("BBBB2222").line() == "BBBB2222 | How Powerful are Graph Neural Networks? | Keyulu Xu 2019 | WhyGNN")

# ================================================================ 토큰 예산
print("== budget: 토큰 예산 ==")
_b = TokenBudget(ctx=10000, reserve_output=2000, safety=500, ratio_en=4.0)
check("추정: 영어 4자/토큰, 한글 1.7자/토큰", _b.est("a" * 400) == 100 and _b.est("가" * 170) in (100, 101)
      and split_chars("가a") == (1, 1))
_m = [{"role": "system", "content": "x" * 4000}, {"role": "user", "content": "y" * 400}]
check("풀 = ctx − 출력 − 여유 − 메시지", _b.pool(_m) == 10000 - 2000 - 500 - (1000 + 6) - (100 + 6))
check("글자 배분: 토큰×비율/편수", _b.chars_for(1000, 2) == 2000 and _b.chars_for(-5) == 0)
check("출력 토큰: 남은 컨텍스트에 맞춤(최소 256)", _b.output_tokens(_m, 2000) == 2000
      and TokenBudget(1500, 2000, 100, 4.0).output_tokens(_m, 2000) == 1500 - 1112 - 100
      and TokenBudget(1200, 2000, 100, 4.0).output_tokens(_m, 2000) == 256)
check("보정: 실측으로 비율 하락·측정값 기억", _b.calibrate(_m, 1400) and _b.ratio_en < 4.0
      and _b.measured == (2, 1400) and _b.used(_m) == 1400)
check("보정: 비현실적 측정(추정의 절반 미만)은 무시", not TokenBudget(10000, 0, 0, 4.0).calibrate(_m, 100))
_bk = TokenBudget(10000, 0, 0, 4.0)
_mk = [{"role": "system", "content": "가" * 3000}, {"role": "user", "content": "x" * 400}]
check("보정: 한글이 주인 메시지는 측정값만 기억하고 비율은 안 건드림",
      _bk.calibrate(_mk, int(_bk.est_messages(_mk) * 1.3)) and _bk.ratio_en == 4.0 and _bk.measured[0] == 2)
_m2 = _m + [{"role": "tool", "content": "z" * 800}]
check("측정 뒤 추가 메시지만 추정해 더함", _b.used(_m2) == 1400 + math.ceil(800 / _b.ratio_en) + 6)
_hist = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a" * 4000},
         {"role": "user", "content": "q2"}, {"role": "assistant", "content": "b" * 400}]
check("기억 절단: 오래된 쌍부터", [m["content"] for m in _b.trim_history(_hist, 400)] == ["q2", "b" * 400]
      and _b.trim_history(_hist, 10) == [] and len(_b.trim_history(_hist, 10**6)) == 4)

# ================================================================ 에이전트
print("== agent: 도구 호출 루프 ==")
agent = _ag.Agent(CFG.chat, cache)
sp = agent.system_prompt()
check("시스템 프롬프트: 카탈로그 포함", "AAAA1111 | Rethinking Graph Transformers" in sp and "2편" in sp)
check("시스템 프롬프트: catalog_max 이하가 아니면 목록 대신 안내(검색 위임)",
      "목록은 프롬프트에 넣지 않음" in _ag.Agent(replace(CFG.chat, catalog_max=0), cache).system_prompt()
      and "AAAA1111" not in _ag.Agent(replace(CFG.chat, catalog_max=0), cache).system_prompt())

script = [
    ("", [{"function": {"name": "search_papers", "arguments": {"keywords": ["Laplacian eigenvectors"]}}}]),
    ("", [{"function": {"name": "read_paper", "arguments": json.dumps({"paper_id": "AAAA1111", "focus": "spectrum"})}}]),
    ("찾으시는 논문은 **Rethinking Graph Transformers with Spectral Attention** 입니다.", []),
]
seen_msgs = []


def fake_chat(self, messages, use_tools, on_partial=None, num_predict=None):
    seen_msgs.append((len(messages), use_tools))
    content, calls = script.pop(0)
    if content and on_partial:
        on_partial(content)
    return content, calls, {"load_duration": 1e9 if len(seen_msgs) == 1 else 0}


statuses, partials = [], []
_orig_chat = _ag.Agent._chat
_orig_check_ctx = _ag.Agent._check_server_ctx
_ag.Agent._chat = fake_chat
_ag.Agent._check_server_ctx = lambda self: None      # 테스트는 Ollama에 묻지 않는다
try:
    res = agent.run("라플라시안 고유벡터 쓰는 논문?", [], statuses.append, partials.append)
    check("에이전트: 검색→읽기→답 (3단계)", res.steps == 3 and [t["name"] for t in res.tools] == ["search_papers", "read_paper"])
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
    # 실측 토큰 보정: prompt_eval_count가 오면 비율이 바뀌고 prompt_tokens에 기록
    def fake_chat_measured(self, messages, use_tools, on_partial=None, num_predict=None):
        c, calls, st = fake_chat(self, messages, use_tools, on_partial, num_predict)
        est = TokenBudget(10**6, 0, 0, 3.5).est_messages(messages)
        return c, calls, {**st, "prompt_eval_count": int(est * 1.3)}
    _ag.Agent._chat = fake_chat_measured
    script[:] = [("답", [])]
    a2 = agent.__class__(CFG.chat, cache)
    r2 = a2.run("GIN? " + "graph isomorphism network expressive power " * 120, [], None, None)   # 영어가 주인 입력
    check("에이전트: 실측 토큰으로 비율 보정·기록", r2.prompt_tokens > 0 and 2.0 <= a2.ratio_en < 3.5 and r2.ratio_en == a2.ratio_en)
    _ag.Agent._chat = fake_chat
    # 토큰 예산: ctx를 작게 → 첫 읽기는 전문, 둘째는 남은 풀만큼 절단+안내, 그 뒤 예산 소진 → 도구 없이 답
    _orig_floor = _ag._READ_FLOOR
    _ag._READ_FLOOR = 300
    tool_msgs, use_flags = [], []

    def fake_chat_capture(self, messages, use_tools, on_partial=None, num_predict=None):
        tool_msgs[:] = [m["content"] for m in messages if m.get("role") == "tool"]
        use_flags.append((use_tools, num_predict))
        return fake_chat(self, messages, use_tools, on_partial, num_predict)

    _ag.Agent._chat = fake_chat_capture
    try:
        _rd = [{"function": {"name": "read_paper", "arguments": {"paper_id": "AAAA1111"}}}]
        script[:] = [("", _rd), ("", _rd), ("", _rd), ("정리", [])]
        _base = TokenBudget(10**6, 0, 0, 3.5).est_messages(
            [{"role": "system", "content": agent.system_prompt()}, {"role": "user", "content": "정리"}])
        _cfg = replace(CFG.chat, ctx_tokens=_base + 256 + 50 + 1150, max_output_tokens=256, safety_tokens=50,
                       chars_per_token=3.5, max_tool_steps=5)
        res = agent.__class__(_cfg, cache).run("정리", [], None, None)
        first, second = tool_msgs[0], tool_msgs[1]
        check("에이전트: 예산 안이면 전문(절단 없음)", "(중략" not in first and "남은 컨텍스트" not in first and _full in first)
        check("에이전트: 남은 풀만큼 절단 + 안내문",
              "남은 컨텍스트" in second and "(중략" in second and res.tools[1]["result_chars"] < res.tools[0]["result_chars"])
        check("에이전트: 예산 소진 → 다음 라운드는 도구 없이(2회 읽고 끝)",
              len(tool_msgs) == 2 and res.steps == 3 and use_flags[-1][0] is False and use_flags[-1][1] >= 256)
        check("에이전트: 예산과 무관하게 근거 논문 기록", res.papers == ["AAAA1111"])
        script[:] = []
        # summarize_paper: 요약 결과 → 근거 논문, 질문당 새 요약 1회 제한(저장된 요약은 제한에 안 걸림)
        _sq = [{"function": {"name": "summarize_paper", "arguments": {"paper_id": "BBBB2222", "force": True}}}]
        _sq_old = [{"function": {"name": "summarize_paper", "arguments": {"paper_id": "BBBB2222"}}}]
        script[:] = [("", _sq), ("", _sq), ("", _sq_old), ("요약을 바탕으로…", [])]
        n0, st2 = len(sum_calls), []
        res = agent.__class__(replace(CFG.chat, summaries_per_question=1, max_tool_steps=5), cache, ps).run(
            "GIN 논문 전체 요약해줘", [], st2.append, None)
        check("에이전트: summarize_paper → 요약 결과·근거·상태 표시",
              res.papers == ["BBBB2222"] and any(s.startswith("📝") for s in st2) and any(s.startswith("🧠") for s in st2)
              and tool_msgs[0].startswith("[전문 요약 — ") and "cache:pdf" in tool_msgs[0] and len(sum_calls) == n0 + 1)
        check("에이전트: 질문당 새 요약 횟수 초과 → 안내", "요약 수를 다 썼습니다" in tool_msgs[1])
        check("에이전트: 저장된 요약은 횟수와 무관하게 재사용", "저장된 요약 재사용" in tool_msgs[2] and len(sum_calls) == n0 + 1)
        script[:] = [("", _sq_old), ("답", [])]
        agent.__class__(CFG.chat, cache).run("요약", [], None, None)
        check("에이전트: 요약 도구 없으면 안내", "꺼져 있습니다" in tool_msgs[0])
        # 3유형 도구: search_papers 자료, read_paper 제목 지정, compare_papers
        script[:] = [("", [{"function": {"name": "search_papers", "arguments": {"keywords": ["graph"], "limit": 2}}}]), ("답", [])]
        agent.run("graph 논문 찾아줘", [], None, None)
        check("에이전트: search_papers → 초록·발췌·저장된 요약 앞부분",
              tool_msgs[0].startswith("[검색 결과 2편") and "초록:" in tool_msgs[0] and "저장된 요약(앞부분)" in tool_msgs[0]
              and "발췌:" in tool_msgs[0])
        script[:] = [("", [{"function": {"name": "read_paper", "arguments": {"paper_id": "how powerful are graph neural networks",
                                                                             "focus": "isomorphism"}}}]), ("답", [])]
        res = agent.run("GIN 질문", [], None, None)
        check("에이전트: read_paper에 제목 → id 해석·근거", res.papers == ["BBBB2222"] and tool_msgs[0].startswith("제목: How Powerful"))
        script[:] = [("", [{"function": {"name": "read_paper", "arguments": {"paper_id": "no such paper at all"}}}]), ("답", [])]
        res = agent.run("?", [], None, None)
        check("에이전트: read_paper 못 찾으면 안내", "찾지 못했습니다" in tool_msgs[0] and res.papers == [])
        st3 = []
        script[:] = [("", [{"function": {"name": "compare_papers", "arguments": {
            "paper_ids": ["AAAA1111", "How Powerful are Graph Neural Networks?", "Nonexistent Paper XYZ"], "aspects": "graph"}}}]),
                     ("비교…", [])]
        res = agent.__class__(CFG.chat, cache).run("비교", [], st3.append, None)
        check("에이전트: compare_papers → 편당 자료 분배·못 찾은 항목·근거·상태",
              "=== 논문 1/2: Rethinking" in tool_msgs[0] and "=== 논문 2/2: How Powerful" in tool_msgs[0]
              and "Nonexistent Paper XYZ" in tool_msgs[0] and "편당 최대" in tool_msgs[0] and "비교 축: graph" in tool_msgs[0]
              and res.papers == ["AAAA1111", "BBBB2222"] and st3[0].startswith("⚖️") and "Rethinking" in st3[0])
        script[:] = [("", [{"function": {"name": "compare_papers", "arguments": {"paper_ids": ["AAAA1111", "zzz qqq www"]}}}]), ("답", [])]
        agent.run("비교", [], None, None)
        check("에이전트: compare_papers 2편 미만 → 안내", "2편 이상 찾지 못했습니다" in tool_msgs[0])
    finally:
        _ag._READ_FLOOR = _orig_floor
        _ag.Agent._chat = fake_chat
finally:
    _ag.Agent._chat = _orig_chat
    _ag.Agent._check_server_ctx = _orig_check_ctx

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
check("Drive 하위 폴더 → 컬렉션 경로", drive_meta("ICML/2026/a.pdf", "F")["collections"] == ["ICML/2026"])
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
check("제목으로 찾기: 부제가 틀려도 콜론 앞 약칭이 유일하면 매칭",
      cache.find_by_title("GOAT: Global-Local Attention with Adaptive Topology").key == g.key
      and cache.find_by_title("Graph: Something Else Entirely") is None)
check("Drive 전용: 경로 기록(summarize_paper용)", json.loads(cache.get_meta("drive_paths", "{}")).get(g.key, "").endswith(".pdf"))
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
with open(bad, "w", encoding="utf-8") as f:
    f.write('library:\n  drive_remote: "gdrive:paper"\nchat:\n  ctx_tokens: 8192\n  max_output_tokens: 4096\n')
try:
    load_config(bad)
    check("ctx_tokens가 출력+여유+8192 미만 → ConfigError", False)
except ConfigError:
    check("ctx_tokens가 출력+여유+8192 미만 → ConfigError", True)
check("config.example: 토큰 예산 키", ex.chat.ctx_tokens == 65536 and ex.chat.safety_tokens == 1024
      and ex.chat.history_max_tokens == 6000 and ex.chat.chars_per_token == 3.5)
check("config.example: 요약 도구 키", ex.chat.summary_enabled and ex.chat.summaries_per_question == 2
      and ex.resolve(ex.chat.paperbot_config).endswith("/config.yaml"))
okp = os.path.join(TMP, "ok.yaml")
with open(okp, "w", encoding="utf-8") as f:
    f.write('chat:\n  llm_url: "http://localhost:11434/v1"\n')
check("llm_url의 /v1 접미사 제거 (네이티브 API 사용)", load_config(okp).chat.llm_url == "http://localhost:11434")

print(f"\n결과: {ok} PASS, {fail} FAIL")
sys.exit(1 if fail else 0)
