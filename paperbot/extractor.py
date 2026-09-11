"""PDF 다운로드 + 텍스트 추출.

- 다운로드: 재시도 2회, 타임아웃 60s.
- 추출: pymupdf.
- References 섹션 이후 제거(휴리스틱).
- max_input_chars 초과 시 앞 70% + 뒤 30%를 남기고 중간 절단.
- pdf_url이 없거나 실패 시 abstract만 반환(abstract_only=True) → 상위에서 표기.
"""
from __future__ import annotations

import dataclasses
import hashlib
import logging
import os
import re
import time
import urllib.request
from dataclasses import dataclass

import pymupdf

from .config import Config
from .models import Paper

logger = logging.getLogger(__name__)

_TRUNC_MARK = "\n\n[... truncated ...]\n\n"
_REFERENCE_HEADINGS = {"references", "reference", "bibliography"}
_USER_AGENT = "paperbot/0.2 (paper summary bot)"
_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass
class ExtractResult:
    text: str
    abstract_only: bool                              # True면 PDF 확보 실패로 abstract만 사용
    figures: list[bytes] = dataclasses.field(default_factory=list)  # 대표 figure PNG들 (Slack 첨부용)


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


def strip_references(text: str) -> str:
    """References/Bibliography 헤딩 이후를 잘라낸다(문서 후반부에서만).

    본문 중간의 'references' 단어 언급을 오탐하지 않도록,
    라인 전체가 헤딩인 경우만 대상으로 하고 문서의 30% 지점 이후만 자른다.
    """
    lines = text.splitlines()
    if not lines:
        return text

    cut_idx: int | None = None
    for i, ln in enumerate(lines):
        # 앞의 번호/점/공백 장식을 제거해 순수 헤딩만 비교.
        s = ln.strip().lower().strip("0123456789.)( \t").strip()
        if s in _REFERENCE_HEADINGS:
            cut_idx = i  # 마지막 등장 위치를 사용

    if cut_idx is not None and cut_idx > len(lines) * 0.3:
        return "\n".join(lines[:cut_idx])
    return text


def truncate(text: str, max_chars: int) -> str:
    """max_chars 초과 시 앞 70% + 뒤 30%만 남기고 중간을 절단."""
    if len(text) <= max_chars:
        return text
    budget = max_chars - len(_TRUNC_MARK)
    head_len = int(budget * 0.7)
    tail_len = budget - head_len
    return text[:head_len] + _TRUNC_MARK + text[-tail_len:]


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


_FIG_MAX_PAGES = 8             # 앞 N페이지에서 figure 후보 탐색
_FIG_MAX_BYTES = 4_000_000     # figure 하나당 PNG 크기 제한
_FIG_MIN_AREA = 120_000        # 이보다 작은 임베디드 이미지(로고 등)는 무시
_FIG_MAX_ASPECT = 6.0          # 가로세로비가 극단적인 것(배너/구분선) 제외
_FIG_MIN_COLORS = 1000         # 색 다양성 하한 — 실측: 진짜 figure 1.2만+, 장식 조각 50~200
_FIG_MIN_DENSITY = 0.2         # 래스터 PNG bytes/px 하한 — 그라디언트 장식(0.13)과 실물(0.57+) 분리
_ZOOM = pymupdf.Matrix(2, 2)   # 렌더 해상도 (2x ≈ 144dpi)


def _raster_candidates(doc, npages: int) -> list[tuple[int, int, float, str, object]]:
    """임베디드 래스터 이미지 후보: (픽셀면적, 페이지, y위치, 'raster', (xref, smask)).

    로고·배너·다이어그램 장식 조각을 걸러내기 위해 크기/가로세로비에 더해
    색 다양성(color_count)을 본다 — 단색 계열 장식은 수십~수백 색에 그친다.
    """
    out = []
    seen_xref: set[int] = set()
    for pno in range(npages):
        page = doc[pno]
        for img in page.get_images(full=True):
            xref, smask = img[0], img[1]
            if xref in seen_xref:
                continue
            seen_xref.add(xref)
            try:
                pix = pymupdf.Pixmap(doc, xref)
                w, h = pix.width, pix.height
            except Exception:  # noqa: BLE001 — 깨진 이미지 스킵
                continue
            if w * h < _FIG_MIN_AREA or max(w, h) / max(1, min(w, h)) > _FIG_MAX_ASPECT:
                continue
            try:
                if pix.color_count() < _FIG_MIN_COLORS:  # 장식 요소 컷
                    continue
            except Exception:  # noqa: BLE001 — 색 카운트 실패 시 필터 생략
                pass
            rects = page.get_image_rects(xref)
            y0 = rects[0].y0 if rects else 0.0
            out.append((w * h, pno, y0, "raster", (xref, smask)))
    return out


def _vector_candidates(doc, npages: int) -> list[tuple[int, int, float, str, object]]:
    """벡터 다이어그램 후보 — ML 논문의 method/아키텍처 그림은 벡터인 경우가 많다.

    drawing cluster의 bbox가 페이지의 12~85%를 차지하면 figure로 간주하고
    그 영역을 렌더링한다. (후보: (환산면적, 페이지, y위치, 'vector', rect))
    """
    if not hasattr(pymupdf.Page, "cluster_drawings"):
        return []
    out = []
    for pno in range(npages):
        page = doc[pno]
        parea = page.rect.width * page.rect.height
        try:
            clusters = page.cluster_drawings()
        except Exception:  # noqa: BLE001 — 복잡한 페이지에서 실패 가능
            continue
        big = [r for r in clusters if 0.12 * parea <= r.width * r.height <= 0.85 * parea]
        big.sort(key=lambda r: r.width * r.height, reverse=True)
        for r in big[:2]:  # 페이지당 최대 2개
            # 2x 렌더 기준 픽셀 면적으로 환산해 래스터 후보와 같은 축으로 비교.
            out.append((int(r.width * 2 * r.height * 2), pno, r.y0, "vector", r))
    return out


def _render_candidate(doc, cand) -> bytes | None:
    _, pno, _, kind, payload = cand
    try:
        if kind == "raster":
            xref, smask = payload
            pix = pymupdf.Pixmap(doc, xref)
            if pix.colorspace is None or pix.colorspace.n > 3:
                pix = pymupdf.Pixmap(pymupdf.csRGB, pix)  # CMYK 등 → RGB
            if smask:  # 투명도(SMask) 적용 — 미적용 시 검은 배경 아티팩트
                try:
                    pix = pymupdf.Pixmap(pix, pymupdf.Pixmap(doc, smask))
                except Exception:  # noqa: BLE001 — 마스크 불일치 시 원본 사용
                    pass
            data = pix.tobytes("png")
            # 그라디언트/단순 도형 장식은 압축 밀도가 극단적으로 낮다 (실측 0.13 vs 0.57+).
            if len(data) / max(1, pix.width * pix.height) < _FIG_MIN_DENSITY:
                return None
        else:
            rect = payload + (-4, -4, 4, 4)  # 캡션 경계 살짝 포함
            pix = doc[pno].get_pixmap(matrix=_ZOOM, clip=rect)
            data = pix.tobytes("png")
        return data if 0 < len(data) <= _FIG_MAX_BYTES else None
    except Exception:  # noqa: BLE001
        return None


def extract_figures(pdf_path: str, max_pages: int = _FIG_MAX_PAGES, max_figures: int = 3) -> list[bytes]:
    """대표 figure PNG 최대 max_figures개 추출 (휴리스틱).

    앞 max_pages 페이지에서 래스터 이미지 + 벡터 다이어그램 후보를 모아
    면적 상위 max_figures개를 고른 뒤 읽기 순서(페이지→세로 위치)로 반환.
    아무것도 못 찾으면 1페이지 렌더 1장으로 폴백. 실패는 치명적이지 않음.
    """
    if max_figures <= 0:
        return []
    try:
        with pymupdf.open(pdf_path) as doc:
            npages = min(max_pages, len(doc))
            cands = _raster_candidates(doc, npages) + _vector_candidates(doc, npages)
            cands.sort(key=lambda c: c[0], reverse=True)   # 면적 큰 순으로 선별
            picked = cands[: max_figures * 2]              # 렌더 실패/중복 대비 여유 선별
            picked.sort(key=lambda c: (c[1], c[2]))        # 읽기 순서로 정렬

            figures: list[bytes] = []
            seen_digest: set[str] = set()
            for cand in picked:
                if len(figures) >= max_figures:
                    break
                data = _render_candidate(doc, cand)
                if data is None:
                    continue
                digest = hashlib.md5(data).hexdigest()
                if digest in seen_digest:                  # 반복 삽입 이미지 dedup
                    continue
                seen_digest.add(digest)
                figures.append(data)

            if figures:
                return figures

            # 폴백: 1페이지 렌더 (논문 카드 역할)
            data = doc[0].get_pixmap(matrix=_ZOOM).tobytes("png")
            return [data] if 0 < len(data) <= _FIG_MAX_BYTES else []
    except Exception as e:  # noqa: BLE001
        logger.warning("figure 추출 실패(텍스트만 발행): %s", e)
        return []


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


def _iter_candidate_urls(cfg: Config, paper: Paper):
    """PDF 후보 URL 생성기 — 순서대로 시도, 뒤의 조회형 후보는 필요할 때만 실행.

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
        if cfg.fetch.unpaywall_email:
            url = _unpaywall_pdf(paper.doi, cfg.fetch.unpaywall_email)
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


def extract(cfg: Config, paper: Paper) -> ExtractResult:
    """논문 하나에 대한 추출 파이프라인. 실패해도 예외 대신 abstract fallback."""
    dest = _pdf_path(cfg, paper)
    try:
        if not (os.path.exists(dest) and os.path.getsize(dest) > 0):
            _download_any(_iter_candidate_urls(cfg, paper), dest)

        raw = extract_text(dest)
        if not raw or not raw.strip():
            raise ValueError("추출된 텍스트가 비어 있음")

        text = _normalize(strip_references(raw))
        text = truncate(text, cfg.llm.max_input_chars)

        # figure는 Slack 첨부가 켜져 있을 때만 추출 (불필요한 렌더링 비용 회피).
        figures: list[bytes] = []
        if cfg.publisher.backend == "slack" and cfg.slack.attach_figure:
            figures = extract_figures(dest, max_figures=cfg.slack.max_figures)
        logger.info("추출 완료 %s: %d자, figure %d개", paper.uid, len(text), len(figures))
        return ExtractResult(text=text, abstract_only=False, figures=figures)

    except Exception as e:  # noqa: BLE001 — abstract로 폴백
        logger.error("PDF 처리 실패 %s (abstract로 폴백): %s", paper.uid, e)
        return _abstract_fallback(paper)
    finally:
        if not cfg.storage.keep_pdfs and os.path.exists(dest):
            try:
                os.remove(dest)
            except OSError:
                pass
