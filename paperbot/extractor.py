"""PDF download + text extraction (figure rendering lives in figures.py).
PDF 다운로드 + 텍스트 추출 (figure 렌더링은 figures.py).

- 다운로드: 후보 URL 폴백 체인(iter_candidate_urls), 재시도 2회, 타임아웃 60s.
- 추출: pymupdf. References 섹션(참고문헌 목록) 제거 — 그 뒤에 부록이 있으면 부록은 남긴다(휴리스틱, 본문 → 부록 순).
- 요약 입력은 summarizer가 full_text를 서버 컨텍스트에 token 기준으로 맞춘다(truncate_tokens).
  max_input_chars(초과 시 앞 70% + 뒤 30%를 남기고 중간 절단)는 token을 셀 수 없을 때의 대비책(text).
- pdf_url이 없거나 실패 시 abstract만 반환(abstract_only=True) → 상위에서 표기.
"""
from __future__ import annotations

import dataclasses
import logging
import os
import re
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass

import pymupdf

from .config import Config
from .figures import Figure, extract_figures
from .models import Paper

logger = logging.getLogger(__name__)

_TRUNC_MARK = "\n\n[... truncated ...]\n\n"
_REFERENCE_HEADINGS = {"references", "reference", "bibliography"}
# 참고문헌 뒤 부록 시작 — 줄 전체가 'Appendix'·'APPENDIX'·'Appendices'·'Supplementary Material' 등.
# 앞에 'A'·'A.'·'10.', 뒤에 'B'·': 제목'·'. 제목' 정도만 허용 ('Appendix D for details …' 같은 본문 줄 제외).
_APPENDIX_RE = re.compile(
    r"(?:[A-Z0-9]{1,2}\.?\s+)?"
    r"(?i:appendix|appendices|supplementary (?:materials?|information)|supplemental materials?)"
    r"(?:\s+[A-Z0-9]{1,2})?(?:\s*[.:–—-]\s+(?P<title>.+))?")
_USER_AGENT = "paperbot/0.2 (paper summary bot)"
_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass
class ExtractResult:
    text: str                                        # token을 셀 수 없을 때 요약기에 넣는 본문 (max_input_chars로 절단)
    abstract_only: bool                              # True면 PDF 확보 실패로 abstract만 사용
    figures: list[Figure] = dataclasses.field(default_factory=list)  # 대표 figure·표 크롭 (kind/label/png) — Slack/Zulip 첨부용
    full_text: str = ""                              # 절단 전 전체 본문 (참고문헌 목록 제거, 부록 포함) — 요약 입력의 출발점, 챗봇 캐시(chatdb)에 저장


def _pdf_path(cfg: Config, paper: Paper) -> str:
    pdf_dir = cfg.resolve(cfg.storage.pdf_dir)
    os.makedirs(pdf_dir, exist_ok=True)
    safe_id = _SAFE_RE.sub("_", paper.uid)
    return os.path.join(pdf_dir, f"{safe_id}.pdf")


def download_pdf(url: str, dest: str, *, retries: int = 2, timeout: int = 60) -> str:
    """PDF를 dest로 다운로드. 실패 시 예외를 던진다."""
    last_err: Exception | None = None
    for attempt in range(1, retries + 2):  # 최초 1회 + retries회
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            if not data:
                raise ValueError("빈 응답")
            with open(dest, "wb") as f:
                f.write(data)
            return dest
        except Exception as e:  # noqa: BLE001
            last_err = e
            logger.warning("PDF 다운로드 실패 (%d/%d): %s", attempt, retries + 1, e)
            if attempt <= retries:
                time.sleep(2 * attempt)
    raise RuntimeError(f"PDF 다운로드 최종 실패: {url}") from last_err


def extract_text(pdf_path: str) -> str:
    """pymupdf로 전체 텍스트 추출."""
    parts: list[str] = []
    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            parts.append(page.get_text())
    return "\n".join(parts)


def _nonblank(lines: list[str], i: int, step: int) -> int | None:
    """i에서 step(±1) 방향으로 가장 가까운 비어 있지 않은 줄의 위치 (없으면 None)."""
    i += step
    while 0 <= i < len(lines):
        if lines[i].strip():
            return i
        i += step
    return None


def _title_like(t: str) -> bool:
    """부록 절 제목처럼 보이나 — 대문자로 시작하는 짧은 한 줄이고, 문장·저자 목록(', '·'. '·et al)·URL이 아님."""
    t = t.strip()
    return (0 < len(t) <= 100 and t[0].isupper() and not t.endswith((".", ",", "-", ":", ";"))
            and not re.search(r"[.,]\s|\bet al\b|https?:|arXiv", t))


def _letter_heading(lines: list[str], i: int, letter: str) -> str | None:
    """lines[i]가 부록 절 헤딩이면 그 형식 — 'two'('A' 줄 + 제목 줄) / 'one'('A. 제목' 한 줄), 아니면 None.

    참고문헌 항목을 헤딩으로 보지 않도록 (2026-11-27 실측 오탐: 'Zisserman,' 다음 줄의 'A. Deep structured …',
    제목이 한 단어씩 끊긴 'In / A / Haystack'): 앞 줄이 ','·'-'로 끝나면(항목이 이어짐) 제외하고,
    'A. 제목'은 두 단어 이상이어야 하며, 제목 다음 줄이 대문자·숫자로 시작해야 한다(제목이 다음 줄로 이어지면 제외).
    """
    s = lines[i].strip()
    if s in (letter, letter + "."):
        t = _nonblank(lines, i, 1)
        if t is None:
            return None
        form, title = "two", lines[t]
    else:
        m = re.fullmatch(rf"{letter}\.\s+(.+)", s)
        if not m or len(m.group(1).split()) < 2:
            return None
        t, form, title = i, "one", m.group(1)
    prev, after = _nonblank(lines, i, -1), _nonblank(lines, t, 1)
    if prev is not None and lines[prev].rstrip().endswith((",", "-")):
        return None
    first = lines[after].lstrip()[:1] if after is not None else ""
    if not _title_like(title) or not (first.isupper() or first.isdigit()):
        return None
    return form


def _appendix_start(lines: list[str], ref_idx: int) -> int | None:
    """참고문헌 헤딩(ref_idx) 뒤에서 부록이 시작하는 줄. 못 찾으면 None.

    참고문헌 항목을 부록으로 잘못 보지 않도록 두 경우만 인정한다 (2026-11-27, arXiv PDF 104편으로 확인):
    - 줄 전체가 'Appendix'·'Supplementary Material' 등인 줄 (_APPENDIX_RE)
    - 'A' 절 헤딩이 있고, 그 뒤에 같은 형식의 'B' 절 헤딩도 있을 때 — 그 'A' 줄 (_letter_heading)
    """
    for i in range(ref_idx + 1, len(lines)):
        s = lines[i].strip()
        m = _APPENDIX_RE.fullmatch(s) if len(s) <= 100 else None
        if m and (m.group("title") is None or _title_like(m.group("title"))):
            return i
        form = _letter_heading(lines, i, "A")
        if form and any(_letter_heading(lines, j, "B") == form for j in range(i + 1, len(lines))):
            return i
    return None


def split_references(text: str) -> tuple[str, str, str]:
    """(본문, 참고문헌 목록, 부록)으로 나눈다. 참고문헌 헤딩이 없으면 (text, "", "").

    References/Bibliography 헤딩은 본문 중간의 'references' 단어 언급을 오탐하지 않도록,
    라인 전체가 헤딩인 경우만 대상으로 하고 문서의 30% 지점 이후만 인정한다.
    참고문헌 목록은 그 헤딩부터 부록 시작 직전까지 — 부록 시작(_appendix_start)을 못 찾으면 문서 끝까지.
    """
    lines = text.splitlines()
    cut_idx: int | None = None
    for i, ln in enumerate(lines):
        # 앞의 번호/점/공백 장식을 제거해 순수 헤딩만 비교.
        s = ln.strip().lower().strip("0123456789.)( \t").strip()
        if s in _REFERENCE_HEADINGS:
            cut_idx = i  # 마지막 등장 위치를 사용

    if cut_idx is None or cut_idx <= len(lines) * 0.3:
        return text, "", ""
    app_idx = _appendix_start(lines, cut_idx)
    end = len(lines) if app_idx is None else app_idx
    return "\n".join(lines[:cut_idx]), "\n".join(lines[cut_idx:end]), "\n".join(lines[end:])


def strip_references(text: str) -> str:
    """참고문헌 목록을 빼고 본문(+ 뒤에 부록이 있으면 부록)을 돌려준다 — split_references 참고."""
    body, _, appendix = split_references(text)
    return f"{body}\n{appendix}" if appendix else body


def truncate(text: str, max_chars: int) -> str:
    """max_chars 초과 시 앞 70% + 뒤 30%만 남기고 중간을 절단."""
    if len(text) <= max_chars:
        return text
    budget = max_chars - len(_TRUNC_MARK)
    head_len = int(budget * 0.7)
    tail_len = budget - head_len
    return text[:head_len] + _TRUNC_MARK + text[-tail_len:]


def truncate_tokens(text: str, max_tokens: int,
                    count: Callable[[str], int | None]) -> tuple[str, int, int] | None:
    """token 수가 max_tokens를 넘으면 넘는 만큼만 줄인다 (truncate와 같은 앞 70% + 뒤 30%, 절단 표시).

    count는 문자열의 token 수(셀 수 없으면 None). 반환 (결과, 원래 token 수, 결과 token 수) — 셀 수 없으면 None.
    글자당 token 비율로 남길 글자 수를 정해 자른 뒤 다시 세고, 아직 넘으면(남긴 부분이 평균보다 촘촘함) 그 비율로 더 줄인다.
    """
    total = count(text)
    if total is None:
        return None
    if total <= max_tokens:
        return text, total, total
    chars = int(len(text) * max_tokens / total * 0.98)   # 2%는 절단 표시·비율 오차 몫
    for _ in range(5):
        if chars <= len(_TRUNC_MARK):
            return None
        cut = truncate(text, chars)
        n = count(cut)
        if n is None:
            return None
        if n <= max_tokens:
            return cut, total, n
        chars = int(chars * max_tokens / n * 0.98)
    return None


def _normalize(text: str) -> str:
    # 과도한 빈 줄 축소 (토큰 낭비 방지).
    lines = [ln.rstrip() for ln in text.splitlines()]
    out: list[str] = []
    blank = 0
    for ln in lines:
        if ln.strip():
            blank = 0
            out.append(ln)
        else:
            blank += 1
            if blank <= 1:
                out.append("")
    return "\n".join(out).strip()


def _abstract_fallback(paper: Paper) -> ExtractResult:
    return ExtractResult(text=paper.abstract or paper.title, abstract_only=True)


def _norm_title(t: str) -> str:
    """제목 매칭용 정규화 — 소문자 + 영숫자만."""
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())


def _unpaywall_pdf(doi: str, email: str) -> str | None:
    """Unpaywall에서 합법 OA PDF URL 조회 (없으면 None)."""
    try:
        import requests

        resp = requests.get(
            f"https://api.unpaywall.org/v2/{doi}",
            params={"email": email}, timeout=20,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        locations = [data.get("best_oa_location") or {}] + (data.get("oa_locations") or [])
        for loc in locations:
            if loc.get("url_for_pdf"):
                logger.info("Unpaywall에서 OA 사본 발견 (%s): %s", loc.get("host_type"), doi)
                return loc["url_for_pdf"]
    except Exception as e:  # noqa: BLE001
        logger.debug("Unpaywall 조회 실패(무시): %s", e)
    return None


def _arxiv_pdf_by_title(title: str, authors: list[str]) -> str | None:
    """S2가 arXiv 링크를 누락한 경우 — 제목 검색으로 프리프린트를 찾는다.

    오매칭 방지: 정규화 제목 완전 일치 + (저자 정보가 있으면) 1저자 성 포함 확인.
    """
    try:
        import arxiv

        client = arxiv.Client(page_size=5, delay_seconds=3.0, num_retries=2)
        search = arxiv.Search(query=f'ti:"{title}"', max_results=5)
        want = _norm_title(title)
        first_author_surname = ""
        if authors:
            first_author_surname = (authors[0].split()[-1] if authors[0].split() else "").lower()

        for r in client.results(search):
            if _norm_title(r.title) != want:
                continue
            if first_author_surname:
                result_authors = " ".join(a.name for a in r.authors).lower()
                if first_author_surname not in result_authors:
                    continue
            logger.info("arXiv 제목 검색으로 프리프린트 발견: %s", r.get_short_id())
            return r.pdf_url
    except Exception as e:  # noqa: BLE001
        logger.debug("arXiv 제목 검색 실패(무시): %s", e)
    return None


def iter_candidate_urls(paper: Paper, unpaywall_email: str = ""):
    """PDF 후보 URL 생성기 — 순서대로 시도, 뒤의 조회형 후보는 필요할 때만 실행.
    (zoterobot도 이 체인을 쓰므로 Config 없이 호출 가능하게 둔다)

    1) S2/arXiv가 준 직접 링크 (출판사 링크는 봇 차단으로 403이 잦음)
    2) arXiv 미러 (id가 있을 때)
    3) Unpaywall 합법 OA 사본 (DOI + 이메일 설정 시 — 실패 시에만 조회)
    4) arXiv 제목 검색 (DOI는 있는데 arXiv 링크가 누락된 출판사 논문 구조용)
    """
    if paper.pdf_url:
        yield paper.pdf_url
    if paper.arxiv_id:
        ax = f"https://arxiv.org/pdf/{paper.arxiv_id}"
        if ax != paper.pdf_url:
            yield ax
    if paper.doi:  # 조회형 후보는 출판사 논문(DOI 보유)에만 의미가 있다
        if unpaywall_email:
            url = _unpaywall_pdf(paper.doi, unpaywall_email)
            if url and url != paper.pdf_url:
                yield url
        if not paper.arxiv_id:
            url = _arxiv_pdf_by_title(paper.title, paper.authors)
            if url:
                yield url


def _download_any(urls, dest: str) -> None:
    """후보 URL을 순서대로 시도, 첫 성공에서 반환. 전부 실패 시 예외."""
    last_err: Exception | None = None
    for url in urls:
        try:
            download_pdf(url, dest, retries=2, timeout=60)
            return
        except Exception as e:  # noqa: BLE001
            last_err = e
            logger.warning("PDF 소스 실패, 다음 후보 시도: %s", url)
    raise RuntimeError("모든 PDF 후보 실패") from last_err


def extract(cfg: Config, paper: Paper, figures: bool = True) -> ExtractResult:
    """논문 하나에 대한 추출 파이프라인. 실패해도 예외 대신 abstract fallback. figures=False면 figure 렌더링 생략(백필용)."""
    dest = _pdf_path(cfg, paper)
    try:
        if not (os.path.exists(dest) and os.path.getsize(dest) > 0):
            _download_any(iter_candidate_urls(paper, cfg.fetch.unpaywall_email), dest)

        raw = extract_text(dest)
        if not raw or not raw.strip():
            raise ValueError("추출된 텍스트가 비어 있음")

        body, _, appendix = split_references(raw)
        full = _normalize(f"{body}\n{appendix}")
        text = truncate(full, cfg.llm.max_input_chars)

        # figure는 이 topic의 발행처(slack/zulip)에서 첨부가 켜져 있을 때만 추출 (불필요한 렌더링 비용 회피).
        attach, max_figs, max_tabs = cfg.figures_for(paper.topic_name)
        figs: list[Figure] = []
        if attach and figures:
            figs = extract_figures(dest, max_figures=max_figs, max_tables=max_tabs)
        logger.info("추출 완료 %s: %d자(전체 %d자, 부록 %d자), figure %d·표 %d개", paper.uid, len(text), len(full),
                    len(appendix), sum(1 for f in figs if f.kind != "table"), sum(1 for f in figs if f.kind == "table"))
        return ExtractResult(text=text, abstract_only=False, figures=figs, full_text=full)

    except Exception as e:  # noqa: BLE001 — abstract로 폴백
        logger.error("PDF 처리 실패 %s (abstract로 폴백): %s", paper.uid, e)
        return _abstract_fallback(paper)
    finally:
        if not cfg.storage.keep_pdfs and os.path.exists(dest):
            try:
                os.remove(dest)
            except OSError:
                pass
