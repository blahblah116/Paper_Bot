"""Figure extraction for Slack attachments (caption-anchored).
Slack 첨부용 대표 figure 추출 — 'Figure N' 캡션을 기준점으로 같은 단의 드로잉·이미지·레이블을 묶어 렌더한다.

앞 N페이지에서 캡션을 찾고, 캡션 위 같은 단(column)의 벡터 드로잉·임베디드 이미지·짧은 텍스트(축 레이블·범례)를
tight bbox로 묶은 뒤 캡션까지 포함해 2x로 렌더한다. 표·정리 박스·옆 단 병합·페이지 헤더는 캡션 기준으로 배제한다.
캡션이 없는 PDF는 큰 드로잉 클러스터를 figure로 간주하는 폴백을 쓴다.
"""
from __future__ import annotations

import hashlib
import logging
import re

import pymupdf

logger = logging.getLogger(__name__)

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
