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
_FIG_HEADER_H = 50.0           # 페이지 상단 이 높이(pt) 안은 헤더로 보고 figure에서 제외
_FIG_ELEM_GAP = 18.0           # figure 요소(서브플롯 등) 사이 허용 세로 간격(pt) — 표·헤더와 분리
_FIG_CLUSTER_TOL = 6.0         # 드로잉 클러스터 병합 허용 오차(pt) — 크면 옆 단 figure까지 붙는다
_FIG_TEXT_MARGIN = 12.0        # 이 거리(pt) 안의 텍스트 블록(축 레이블·범례)은 figure 영역에 합침
_FIG_TEXT_MAX_H = 45.0         # 이보다 높은 텍스트 블록은 본문 단락으로 보고 합치지 않음
_CAPTION_FIG_RE = re.compile(r"^\s*(?:Figure|Fig\.?)\s*\d+", re.IGNORECASE)
_CAPTION_TAB_RE = re.compile(r"^\s*Table\s*\d+", re.IGNORECASE)


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


def _raster_inside(doc, cand, rect: pymupdf.Rect) -> bool:
    """래스터 후보의 배치 영역이 rect 안에 (거의) 들어가는가."""
    _, pno, _, _, (xref, _) = cand
    for ir in doc[pno].get_image_rects(xref):
        inter = ir & rect
        if not inter.is_empty and inter.width * inter.height >= 0.8 * ir.width * ir.height:
            return True
    return False


def _text_blocks(page) -> list[tuple[pymupdf.Rect, str]]:
    """페이지의 텍스트 블록 (bbox, 텍스트). 이미지 블록은 제외."""
    out = []
    try:
        for b in page.get_text("blocks"):
            if len(b) >= 7 and b[6] != 0:  # block_type 1 = image
                continue
            r = pymupdf.Rect(b[:4])
            if not r.is_empty:
                out.append((r, b[4] if len(b) > 4 else ""))
    except Exception:  # noqa: BLE001
        pass
    return out


def _is_paragraph(r: pymupdf.Rect, txt: str, page_w: float) -> bool:
    """본문 단락처럼 보이는 텍스트 블록인가 (figure 안의 짧은 레이블과 구분)."""
    lines = txt.count("\n") + 1
    # 폭이 단의 절반도 안 되는 블록은 아무리 높아도 단락이 아니다 (y축 눈금 레이블 열 등).
    # 여러 줄 넓은 블록이라도 높이가 세 줄 미만이면 한 줄에 나열된 레이블 묶음이다.
    if r.width <= 0.3 * page_w:
        return False
    return r.height > _FIG_TEXT_MAX_H or (lines >= 4 and r.height >= 28.0)


def _caption_candidates(doc, npages: int) -> list[tuple[int, int, float, str, object]]:
    """'Figure N' 캡션을 기준점으로 figure 영역을 잡는다 (주 경로).

    캡션 바로 위, 같은 단(캡션의 가로 범위와 겹치는 영역)에서 본문 단락이나
    다른 캡션을 만나기 전까지를 figure 본체로 보고, 그 안의 드로잉·이미지·짧은
    텍스트(레이블)의 합집합을 tight bbox로 삼은 뒤 캡션을 붙인다.
    - 텍스트가 드로잉 bbox 밖으로 밀려 잘리는 문제 → 영역 안 텍스트를 합쳐 해결
    - 표·정리 박스가 figure로 잡히는 문제 → Figure 캡션이 있는 것만 후보
    - 옆 단 figure와 병합되는 문제 → 캡션의 가로 범위로 단을 구분
    후보: (점수, 페이지, y위치, 'region', rect)
    """
    out = []
    for pno in range(npages):
        page = doc[pno]
        prect = page.rect
        pw, parea = prect.width, prect.width * prect.height
        blocks = _text_blocks(page)
        captions = [(r, t) for r, t in blocks if _CAPTION_FIG_RE.match(t)]
        if not captions:
            continue
        try:
            drawings = page.cluster_drawings(x_tolerance=_FIG_CLUSTER_TOL, y_tolerance=_FIG_CLUSTER_TOL)
        except Exception:  # noqa: BLE001
            drawings = []
        images = []
        for img in page.get_images(full=True):
            for ir in page.get_image_rects(img[0]):
                images.append(ir)
        stoppers = [(r, t) for r, t in blocks
                    if _is_paragraph(r, t, pw) or _CAPTION_FIG_RE.match(t) or _CAPTION_TAB_RE.match(t)]

        for cap, _ in captions:
            # 캡션 위쪽으로 올라가며 같은 단의 첫 '정지 블록'(단락/다른 캡션) 아래를 상단으로.
            top = prect.y0 + _FIG_HEADER_H  # 페이지 헤더(러닝 타이틀) 영역은 제외
            for r, _ in stoppers:
                if r.y1 <= cap.y0 + 2.0 and r.x1 > cap.x0 and r.x0 < cap.x1 and r is not cap:
                    top = max(top, r.y1)
            region = pymupdf.Rect(cap.x0, top, cap.x1, cap.y0)
            if region.is_empty or region.height < 30.0:
                continue
            # 영역과 겹치는 드로잉·이미지·짧은 텍스트로 tight bbox 구성.
            # (figure가 캡션보다 넓은 경우를 위해 가로는 겹침만 요구)
            # 캡션이 단 폭(페이지의 55% 미만)이고 한쪽 단에 치우쳐 있으면 그 단 안으로
            # 요소를 잘라 옆 단 figure와 분리. 짧지만 페이지 중앙에 놓인 캡션은 전폭 figure.
            mid = prect.x0 + pw / 2.0
            if cap.width < 0.55 * pw and abs((cap.x0 + cap.x1) / 2.0 - mid) > 0.08 * pw:
                colbox = pymupdf.Rect(prect.x0, top, mid, cap.y0 + 2.0) if (cap.x0 + cap.x1) / 2.0 < mid \
                    else pymupdf.Rect(mid, top, prect.x1, cap.y0 + 2.0)
            else:
                colbox = pymupdf.Rect(prect.x0, top, prect.x1, cap.y0 + 2.0)
            # 텍스트 병합용 상자: wrapfigure처럼 옆 단락이 figure 제목과 세로로 겹치는 경우를
            # 위해 위로 12pt 여유를 둔다 (단락 자체는 stopper라 병합되지 않음).
            textbox = pymupdf.Rect(colbox.x0, max(top - 12.0, prect.y0 + _FIG_HEADER_H), colbox.x1, colbox.y1)
            elems = []
            for r in list(drawings) + images:
                if min(r.width, r.height) < 3.0:
                    continue  # 가로줄/구분선(페이지 헤더 밑줄 등)
                r = pymupdf.Rect(r) & colbox
                if not r.is_empty and r.intersects(region):
                    elems.append(r)
            if not elems:
                continue  # 캡션만 있고 그림 요소가 없음 (그림이 다음 페이지 등)
            # 캡션에 가장 가까운 요소에서 시작해 세로 간격이 좁은 요소만 이어 붙인다.
            # 같은 단 위쪽의 표(가로줄 드로잉)·페이지 헤더는 간격이 넓어 떨어져 나간다.
            elems.sort(key=lambda r: r.y1, reverse=True)
            tight = elems[0]
            rest = elems[1:]
            while rest:
                near = [r for r in rest if r.intersects(tight) or 0 <= tight.y0 - r.y1 <= _FIG_ELEM_GAP]
                if not near:
                    break
                for r in near:
                    tight |= r
                rest = [r for r in rest if r not in near]
            for r, t in blocks:
                if _is_paragraph(r, t, pw) or _CAPTION_FIG_RE.match(t) or _CAPTION_TAB_RE.match(t):
                    continue
                probe = tight + (-_FIG_TEXT_MARGIN, -_FIG_TEXT_MARGIN, _FIG_TEXT_MARGIN, _FIG_TEXT_MARGIN)
                inside = r & textbox
                # 단락 바로 아래 붙은 플롯 제목은 블록 bbox가 단락과 살짝 겹치므로 60% 포함이면 허용.
                # 단, 헤더 영역에 걸친 블록(러닝 타이틀)은 제외.
                if r.y0 < prect.y0 + _FIG_HEADER_H:
                    continue
                if not (probe.intersects(r) and not inside.is_empty and inside.height >= 0.6 * r.height):
                    continue
                if r.y1 <= tight.y0 + 1.0:
                    # figure 위쪽 텍스트는 (1) 바싹 붙은 것(행 레이블), (2) figure 중앙 정렬(플롯 제목),
                    # (3) 어느 한 요소(서브패널) 바로 위 그 요소 범위 안에 놓인 것((a)(b) 라벨)만.
                    # 단 왼쪽 끝에 붙어 있고 간격이 있는 짧은 블록은 앞 단락의 마지막 줄이다.
                    cx = (r.x0 + r.x1) / 2.0
                    centered = abs(cx - (tight.x0 + tight.x1) / 2.0) <= 0.15 * tight.width
                    over_elem = any(r.x1 > e.x0 and r.x0 < e.x1 and 0 <= e.y0 - r.y1 <= 12.0 for e in elems)
                    if tight.y0 - r.y1 > 6.0 and not centered and not over_elem:
                        continue
                tight |= r
            rect = (tight | cap) + (-4.0, -2.0, 4.0, 2.0)
            # 위로는 stopper(단락) 하단, 아래로는 캡션 다음 단락 상단을 넘지 않게 — 본문이 비치지 않도록.
            rect.y0 = max(rect.y0, min(top, tight.y0))
            below = [r.y0 for r, _ in stoppers if r.y0 >= cap.y1 - 2.0 and r.x1 > cap.x0 and r.x0 < cap.x1]
            if below:
                rect.y1 = min(rect.y1, max(min(below), cap.y1))
            rect &= prect
            if rect.width * rect.height < 0.03 * parea:
                continue
            out.append((int(rect.width * 2 * rect.height * 2), pno, rect.y0, "region", rect))
    return out


def _vector_candidates(doc, npages: int) -> list[tuple[int, int, float, str, object]]:
    """캡션 없는 PDF용 폴백 — 큰 드로잉 클러스터를 figure로 간주.

    표·정리 박스도 잡히기 때문에 _caption_candidates가 하나도 없을 때만 쓴다.
    """
    if not hasattr(pymupdf.Page, "cluster_drawings"):
        return []
    out = []
    for pno in range(npages):
        page = doc[pno]
        parea = page.rect.width * page.rect.height
        try:
            clusters = page.cluster_drawings(x_tolerance=_FIG_CLUSTER_TOL, y_tolerance=_FIG_CLUSTER_TOL)
        except Exception:  # noqa: BLE001 — 복잡한 페이지에서 실패 가능
            continue
        big = [r for r in clusters if 0.12 * parea <= r.width * r.height <= 0.85 * parea]
        big.sort(key=lambda r: r.width * r.height, reverse=True)
        for r in big[:2]:  # 페이지당 최대 2개
            out.append((int(r.width * 2 * r.height * 2), pno, r.y0, "vector", pymupdf.Rect(r) & page.rect))
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
        elif kind == "region":  # 캡션 기준 영역 — 여백·경계 처리가 이미 끝난 rect
            pix = doc[pno].get_pixmap(matrix=_ZOOM, clip=payload)
            data = pix.tobytes("png")
        else:
            rect = (payload + (-4, -2, 4, 2)) & doc[pno].rect  # 여백 살짝
            pix = doc[pno].get_pixmap(matrix=_ZOOM, clip=rect)
            data = pix.tobytes("png")
        return data if 0 < len(data) <= _FIG_MAX_BYTES else None
    except Exception:  # noqa: BLE001
        return None


def extract_figures(pdf_path: str, max_pages: int = _FIG_MAX_PAGES, max_figures: int = 3) -> list[bytes]:
    """대표 figure PNG 최대 max_figures개 추출 (휴리스틱).

    앞 max_pages 페이지에서 'Figure N' 캡션 기준 영역(_caption_candidates)을 주 후보로,
    캡션 영역에 포함되지 않은 임베디드 래스터 이미지를 보조 후보로 모아
    면적 상위 max_figures개를 고른 뒤 읽기 순서(페이지→세로 위치)로 반환.
    캡션을 하나도 못 찾은 PDF는 큰 드로잉 클러스터(_vector_candidates)로 폴백.
    아무것도 못 찾으면 1페이지 렌더 1장으로 폴백. 실패는 치명적이지 않음.
    """
    if max_figures <= 0:
        return []
    try:
        with pymupdf.open(pdf_path) as doc:
            npages = min(max_pages, len(doc))
            cap_cands = _caption_candidates(doc, npages)
            rasters = _raster_candidates(doc, npages)
            if cap_cands:
                # 캡션 영역 렌더에 이미 포함되는 임베디드 이미지는 중복이므로 제외.
                covered = [(c[1], c[4]) for c in cap_cands]
                rasters = [r for r in rasters
                           if not any(pno == r[1] and _raster_inside(doc, r, rect) for pno, rect in covered)]
                cands = cap_cands + rasters
            else:
                cands = rasters + _vector_candidates(doc, npages)
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
