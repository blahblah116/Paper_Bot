"""Figure/table extraction for Slack/Zulip attachments — DocLayout-YOLO + PyMuPDF.
Slack/Zulip 첨부용 figure·표 추출 — 레이아웃 검출 모델(DocLayout-YOLO)로 박스를 찾고 PyMuPDF로 크롭 렌더한다.

앞 N페이지를 렌더해 DocLayout-YOLO(DocStructBench 가중치, CPU)로 figure / figure_caption / table /
table_caption / table_footnote 박스를 검출하고, 캡션마다 가장 가까운 같은 종류의 본체를 붙여
(본체 ∪ 캡션 ∪ 표 각주) 영역을 만든 뒤 2x로 크롭한다. 라벨은 캡션 텍스트의 'Figure N'/'Table N'(로마 숫자 포함)을 읽는다.
figure와 표는 각각 면적 큰 순으로 max_figures / max_tables장을 고르고 캡션 번호 순으로 돌려준다.
"""
from __future__ import annotations

import dataclasses
import hashlib
import logging
import re
import threading
import time
from typing import NamedTuple

import pymupdf

logger = logging.getLogger(__name__)

_MAX_PAGES = 10                 # 앞 N페이지 탐색 (결과표가 6~9페이지에 몰리는 일이 많다)
_MAX_BYTES = 4_000_000          # 크롭 PNG 하나당 크기 상한
_ZOOM = pymupdf.Matrix(2, 2)    # 크롭 렌더 해상도 (2x ≈ 144dpi)
_DET_LONG_SIDE = 1024           # 검출용 렌더: 긴 변을 모델 입력 크기(imgsz)에 맞춘다 (내부 리사이즈 손실 없음)
_CONF = 0.25                    # 검출 신뢰도 하한
_DEVICE = "cpu"                 # GPU는 요약 모델 전용 — 임베딩 필터와 같은 원칙 (페이지당 ≈0.5s)
_CAPTION_GAP = 60.0             # 본체-캡션 세로 간격 상한(pt)
_FOOTNOTE_GAP = 20.0            # 표 본체-각주 세로 간격 상한(pt)
_MIN_H_OVERLAP = 0.3            # 본체-캡션 가로 겹침 비율 하한 (좁은 쪽 폭 기준)
_UNCAPTIONED_MIN_AREA = 0.08    # 캡션 없는 figure는 페이지 면적의 8% 이상일 때만 (장식·로고 배제). 캡션 없는 표는 버림
_PAD = (-4.0, -3.0, 4.0, 3.0)   # 크롭 여백(pt)

_HF_REPO = "juliozhao/DocLayout-YOLO-DocStructBench"
_HF_FILE = "doclayout_yolo_docstructbench_imgsz1024.pt"   # 약 41MB, ~/.cache/huggingface 에 캐시

_BODY_KIND = {"figure": "figure", "table": "table"}
_CAPTION_KIND = {"figure_caption": "figure", "table_caption": "table"}
# 캡션 번호: 아라비아 숫자(Figure 3, Fig. 3a) 또는 로마 숫자(IEEE 형식 'TABLE IV' — 2026-10-23 실측 arXiv 2606.10166).
# 로마 숫자는 대문자만, 뒤가 단어 경계일 때만 ('Table In …' 같은 문장은 제외).
_NUM = r"(\d+|[IVXLC]{1,7}\b)"
_LABEL_RE = re.compile(r"\b(?i:(Figure|Fig\.?|Table))\s*" + _NUM)
_CAPTION_START_RE = re.compile(r"^\s*(?i:(Figure|Fig\.?|Table))\s*" + _NUM)   # 텍스트 블록 폴백용
_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
_LABEL_WORD = {"figure": "Figure", "table": "Table"}


def _roman_to_int(s: str) -> int:
    total = 0
    for i, ch in enumerate(s):
        v = _ROMAN[ch]
        total += -v if i + 1 < len(s) and _ROMAN[s[i + 1]] > v else v
    return total


def _label_num(label: str | None) -> int:
    """'Table 3' → 3, 'Table IV' → 4, 번호 없음 → 아주 큰 수 (정렬 키)."""
    parts = (label or "").split()
    num = parts[1] if len(parts) >= 2 else ""
    if num.isdigit():
        return int(num)
    if num and all(ch in _ROMAN for ch in num):
        return _roman_to_int(num)
    m = re.search(r"(\d+)", label or "")
    return int(m.group(1)) if m else 1_000_000


def label_slug(label: str) -> str:
    """첨부 파일명용 라벨: 'Figure 3' → 'figure_3', 'Table 1 (2)' → 'table_1_2'."""
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


class _Cand(NamedTuple):
    """크롭 후보 하나 (선별 전)."""
    area: int
    page: int
    y0: float
    kind: str              # 'figure' | 'table'
    label: str | None      # 캡션에서 읽은 'Figure 3' / 'Table IV', 못 읽으면 None
    rect: pymupdf.Rect


@dataclasses.dataclass(frozen=True)
class Figure:
    """첨부 한 장: kind는 'figure' | 'table' | 'page'(폴백: 1페이지 렌더), label은 'Figure 3' / 'Table 1' / 'Page 1'."""
    kind: str
    label: str
    png: bytes


# ---------------------------------------------------------------- 모델

_model = None
_model_lock = threading.Lock()


def _get_model():
    """DocLayout-YOLO 모델 지연 로드(프로세스당 1회, 약 6초). 가중치는 Hugging Face Hub 캐시."""
    global _model
    with _model_lock:
        if _model is None:
            from doclayout_yolo import YOLOv10
            from huggingface_hub import hf_hub_download
            t0 = time.monotonic()
            try:   # 캐시에 있으면 Hub에 다시 묻지 않는다 (실행마다 0.3~1s, Hub 장애 땐 10s 타임아웃 절약)
                path = hf_hub_download(repo_id=_HF_REPO, filename=_HF_FILE, local_files_only=True)
            except Exception:  # noqa: BLE001 — 캐시 없음(첫 실행) → 내려받기
                path = hf_hub_download(repo_id=_HF_REPO, filename=_HF_FILE)
            _model = YOLOv10(path)
            logger.info("DocLayout-YOLO 로드 완료 (%s, %.1fs)", _DEVICE, time.monotonic() - t0)
        return _model


def _detect(model, page) -> tuple[list[tuple[str, pymupdf.Rect]], list[tuple[str, pymupdf.Rect]], list[pymupdf.Rect]]:
    """페이지 하나를 렌더해 검출. (본체[(kind, rect)], 캡션[(kind, rect)], 표 각주[rect]) — 좌표는 PDF pt."""
    from PIL import Image
    pr = page.rect
    z = _DET_LONG_SIDE / max(pr.width, pr.height)
    pix = page.get_pixmap(matrix=pymupdf.Matrix(z, z))
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    with _model_lock:  # predict는 스레드 안전이 보장되지 않음
        res = model.predict(img, imgsz=_DET_LONG_SIDE, conf=_CONF, device=_DEVICE, verbose=False)[0]
    bodies, captions, footnotes = [], [], []
    for xyxy, cls in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist()):
        name = res.names[int(cls)]
        x0, y0, x1, y1 = (v / z for v in xyxy)
        r = pymupdf.Rect(x0, y0, x1, y1) & pr
        if r.is_empty:
            continue
        if name in _BODY_KIND:
            bodies.append((_BODY_KIND[name], r))
        elif name in _CAPTION_KIND:
            captions.append((_CAPTION_KIND[name], r))
        elif name == "table_footnote":
            footnotes.append(r)
    return bodies, captions, footnotes


# ---------------------------------------------------------------- 묶기

def _h_overlap(a: pymupdf.Rect, b: pymupdf.Rect) -> float:
    w = min(a.x1, b.x1) - max(a.x0, b.x0)
    return w / max(1.0, min(a.width, b.width)) if w > 0 else 0.0


def _v_gap(a: pymupdf.Rect, b: pymupdf.Rect) -> float:
    """세로 간격(pt). 겹치면 0."""
    return max(0.0, a.y0 - b.y1, b.y0 - a.y1)


def _side(body: pymupdf.Rect, cap: pymupdf.Rect) -> int:
    """본체가 캡션의 위(-1)/아래(+1) 어느 쪽인가."""
    return -1 if (body.y0 + body.y1) / 2.0 < (cap.y0 + cap.y1) / 2.0 else 1


def _kind_of(word: str) -> str:
    """캡션 첫 단어('Figure'/'Fig.'/'TABLE') → 'figure' | 'table'."""
    return "figure" if word.lower().startswith("fig") else "table"


def _pair_gap(cap: pymupdf.Rect, body: pymupdf.Rect) -> float | None:
    """캡션과 본체가 짝이 될 수 있으면 세로 간격(pt), 아니면 None — 가로로 충분히 겹치고 세로로 가까워야 한다."""
    if _h_overlap(cap, body) < _MIN_H_OVERLAP:
        return None
    gap = _v_gap(cap, body)
    return gap if gap <= _CAPTION_GAP else None


def _caption_label(page, cap: pymupdf.Rect, kind: str) -> str | None:
    """캡션 영역의 텍스트에서 'Figure N' / 'Table N'을 읽는다. 못 읽으면 None."""
    try:
        txt = page.get_text("text", clip=cap + (-2.0, -2.0, 2.0, 2.0)) or ""
    except Exception:  # noqa: BLE001
        return None
    for m in _LABEL_RE.finditer(txt):
        if _kind_of(m.group(1)) == kind:
            return f"{_LABEL_WORD[kind]} {m.group(2)}"
    return None


def _text_caption_blocks(page) -> list[tuple[str, pymupdf.Rect]]:
    """'Figure N' / 'Table N'으로 시작하는 텍스트 블록 → (kind, rect). 모델이 캡션을 놓쳤을 때의 폴백."""
    out = []
    try:
        for b in page.get_text("blocks"):
            if len(b) >= 7 and b[6] != 0:  # 이미지 블록 제외
                continue
            m = _CAPTION_START_RE.match(b[4] if len(b) > 4 else "")
            if not m:
                continue
            r = pymupdf.Rect(b[:4])
            if not r.is_empty:
                out.append((_kind_of(m.group(1)), r))
    except Exception:  # noqa: BLE001
        pass
    return out


def _group_page(page, bodies, captions, footnotes) -> list[tuple[str, str | None, pymupdf.Rect]]:
    """캡션마다 가장 가까운 같은 종류의 본체를 붙인다 (간격 오름차순 그리디).

    - 본체는 캡션 하나에만 속한다. 한 캡션에 여러 본체가 붙으면(서브패널) 합치되, 첫 본체와
      같은 쪽(캡션 위/아래)에 있는 것만 — 캡션 건너편의 다음 figure가 딸려 오지 않게.
    - 표에는 바로 아래 각주(table_footnote)를 붙인다.
    - 모델이 캡션을 놓친 본체는 'Figure N'/'Table N' 텍스트 블록이 옆에 있으면 그것을 캡션으로 쓴다.
    - 그래도 캡션 없는 figure는 충분히 클 때만 단독 후보, 캡션 없는 표는 버린다(알고리즘 박스 오검출 방지).
    반환: (kind, label|None, rect)
    """
    pairs = []  # (gap, caption_idx, body_idx)
    for ci, (ck, cr) in enumerate(captions):
        for bi, (bk, br) in enumerate(bodies):
            gap = _pair_gap(cr, br) if bk == ck else None
            if gap is not None:
                pairs.append((gap, ci, bi))
    pairs.sort()
    cap_bodies: dict[int, list[int]] = {}
    taken: set[int] = set()
    for _, ci, bi in pairs:
        if bi in taken:
            continue
        have = cap_bodies.get(ci)
        if have and _side(bodies[bi][1], captions[ci][1]) != _side(bodies[have[0]][1], captions[ci][1]):
            continue
        taken.add(bi)
        cap_bodies.setdefault(ci, []).append(bi)

    # 모델이 캡션을 놓친 본체: 같은 종류의 'Figure N'/'Table N'으로 시작하는 텍스트 블록이 바로 옆에 있으면 캡션으로 삼는다
    # (실측: 한 페이지에 표 3개면 캡션을 1개만 잡는 일이 있음).
    text_caps = _text_caption_blocks(page) if len(taken) < len(bodies) else []   # 다 짝지었으면 텍스트 파싱 생략
    for bi, (bk, br) in enumerate(bodies):
        if bi in taken:
            continue
        best = None
        for tk, tr in text_caps:
            gap = _pair_gap(tr, br) if tk == bk else None
            if gap is not None and (best is None or gap < best[0]):
                best = (gap, tr)
        if best is not None:
            captions.append((bk, best[1]))
            cap_bodies[len(captions) - 1] = [bi]
            taken.add(bi)

    pr = page.rect
    out: list[tuple[str, str | None, pymupdf.Rect]] = []
    for ci, idxs in cap_bodies.items():
        kind, cap = captions[ci]
        rect = pymupdf.Rect(cap)
        for bi in idxs:
            rect |= bodies[bi][1]
        if kind == "table":
            body = pymupdf.Rect(bodies[idxs[0]][1])
            for bi in idxs[1:]:
                body |= bodies[bi][1]
            for fr in footnotes:
                if fr.y0 >= body.y1 - 2.0 and _v_gap(fr, body) <= _FOOTNOTE_GAP and _h_overlap(fr, body) >= _MIN_H_OVERLAP:
                    rect |= fr
        out.append((kind, _caption_label(page, cap, kind), rect))
    parea = pr.width * pr.height
    for bi, (kind, br) in enumerate(bodies):
        if bi in taken or kind != "figure":
            continue
        if br.width * br.height >= _UNCAPTIONED_MIN_AREA * parea:
            out.append((kind, None, pymupdf.Rect(br)))
    return out


# ---------------------------------------------------------------- 추출

def _render(page, rect: pymupdf.Rect) -> bytes | None:
    rect = (rect + _PAD) & page.rect
    if rect.is_empty:
        return None
    data = page.get_pixmap(matrix=_ZOOM, clip=rect).tobytes("png")
    return data if 0 < len(data) <= _MAX_BYTES else None


def _choose(cands: list[_Cand], limit: int, render, seen_digest: set[str]) -> list[tuple[_Cand, bytes]]:
    """후보 중 면적 큰 순으로 limit장을 고르고, 붙이는 순서는 캡션 번호 순으로 돌려준다.

    render(cand) → PNG bytes | None. 렌더 실패(None)나 이미 고른 것과 같은 그림(PNG md5가 seen_digest에 있음)이면
    그다음으로 큰 후보를 쓴다. 번호를 못 읽은 후보는 번호 있는 것들 뒤에 페이지·세로 위치 순으로 둔다.
    """
    chosen: list[tuple[_Cand, bytes]] = []
    if limit <= 0:
        return chosen
    for c in sorted(cands, key=lambda c: c.area, reverse=True):
        if len(chosen) >= limit:
            break
        data = render(c)
        if data is None:
            continue
        digest = hashlib.md5(data).hexdigest()
        if digest in seen_digest:
            continue
        seen_digest.add(digest)
        chosen.append((c, data))
    # 캡션 번호 순(Figure 1, 2, …) — 2단 레이아웃에선 페이지 위치 순이 번호와 어긋난다(실측: Table 2→1→3).
    chosen.sort(key=lambda cd: (_label_num(cd[0].label), cd[0].page, cd[0].y0))
    return chosen


def extract_figures(pdf_path: str, max_pages: int = _MAX_PAGES,
                    max_figures: int = 3, max_tables: int = 3) -> list[Figure]:
    """대표 figure 최대 max_figures장 + 표 최대 max_tables장 (캡션 포함 크롭).

    앞 max_pages 페이지에서 레이아웃 검출 → 캡션 묶기 → 종류별로 면적 큰 순 선별(_choose) → 2x 크롭.
    figure들이 먼저, 표들이 뒤에 온다(각각 캡션 번호 순). 같은 라벨이 겹치면 ' (2)'를 붙인다.
    아무것도 못 찾으면 1페이지 렌더 1장(kind='page')으로 폴백. 모델 로드·검출 실패는 경고만 남기고 빈 목록.
    """
    if max_figures <= 0 and max_tables <= 0:
        return []
    try:
        model = _get_model()
    except Exception as e:  # noqa: BLE001 — 모델 없이도 발행은 계속 (텍스트만)
        logger.warning("레이아웃 모델 로드 실패(figure 없이 발행): %s", e)
        return []
    try:
        t0 = time.monotonic()
        with pymupdf.open(pdf_path) as doc:
            npages = min(max_pages, len(doc))
            cands: list[_Cand] = []
            for pno in range(npages):
                page = doc[pno]
                bodies, captions, footnotes = _detect(model, page)
                for kind, label, rect in _group_page(page, bodies, captions, footnotes):
                    cands.append(_Cand(int(rect.width * rect.height), pno, rect.y0, kind, label, rect))

            figures: list[Figure] = []
            seen_digest: set[str] = set()
            seen_label: dict[str, int] = {}
            for kind, limit in (("figure", max_figures), ("table", max_tables)):
                group = [c for c in cands if c.kind == kind]
                for c, data in _choose(group, limit, lambda c: _render(doc[c.page], c.rect), seen_digest):
                    label = c.label or _LABEL_WORD[kind]
                    seen_label[label] = seen_label.get(label, 0) + 1
                    if seen_label[label] > 1:
                        label = f"{label} ({seen_label[label]})"
                    figures.append(Figure(kind, label, data))
            logger.debug("레이아웃 추출 %s: 후보 figure %d·table %d → %d장, %.1fs", pdf_path,
                         sum(1 for c in cands if c.kind == "figure"), sum(1 for c in cands if c.kind == "table"),
                         len(figures), time.monotonic() - t0)
            if figures:
                return figures

            # 폴백: 1페이지 렌더 (논문 카드 역할)
            data = doc[0].get_pixmap(matrix=_ZOOM).tobytes("png")
            return [Figure("page", "Page 1", data)] if 0 < len(data) <= _MAX_BYTES else []
    except Exception as e:  # noqa: BLE001
        logger.warning("figure 추출 실패(텍스트만 발행): %s", e)
        return []
