"""그림 후보 다운로드 → 검증 → Slack 업로드용 변환.

- 본문 등장 순으로 max_candidates개까지 받아 크기·비율을 검사한다
  (아이콘·작은 썸네일·가로로 긴 띠 배너 제거).
- Slack 미리보기가 확실한 PNG/JPEG/GIF로 맞춘다: WebP·AVIF 등은 PNG로 변환하고,
  PNG가 크면(사진류) JPEG로. 긴 변이 max_side를 넘으면 축소.
- 애니메이션 GIF는 max_upload_bytes 이하면 그대로(Slack에서 재생), 넘으면 첫 프레임만.
- hero(og:image) 정책: "fallback"은 본문 그림이 하나도 살아남지 못했을 때만 hero를 쓴다.
"""
from __future__ import annotations

import hashlib
import logging
import warnings
from io import BytesIO

from PIL import Image, ImageOps, UnidentifiedImageError

from .config import ImageConfig
from .http import FetchError, Http
from .models import FetchedImage, ImageRef

logger = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = 80_000_000          # 디컴프레션 폭탄 방지 (초과 시 예외)
_PNG_MAX_BEFORE_JPEG = 2_500_000             # 변환한 PNG가 이보다 크면 사진류로 보고 JPEG


class _Reject(Exception):
    pass


def _encode(img: Image.Image, cfg: ImageConfig) -> tuple[bytes, str]:
    """정지 이미지를 PNG(도표·다이어그램) 또는 JPEG(사진류)로 인코딩."""
    if max(img.size) > cfg.max_side:
        img = img.copy()
        img.thumbnail((cfg.max_side, cfg.max_side), Image.LANCZOS)
    has_alpha = img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)
    png_src = img.convert("RGBA" if has_alpha else "RGB")
    buf = BytesIO()
    png_src.save(buf, "PNG", optimize=True)
    if buf.tell() <= min(_PNG_MAX_BEFORE_JPEG, cfg.max_upload_bytes):
        return buf.getvalue(), "png"
    rgb = png_src
    if has_alpha:                             # 투명 배경은 흰색으로 평탄화
        bg = Image.new("RGB", png_src.size, (255, 255, 255))
        bg.paste(png_src, mask=png_src.getchannel("A"))
        rgb = bg
    for quality in (90, 82, 72):
        buf = BytesIO()
        rgb.save(buf, "JPEG", quality=quality, optimize=True, progressive=True)
        if buf.tell() <= cfg.max_upload_bytes:
            break
    return buf.getvalue(), "jpg"


def prepare(data: bytes, cfg: ImageConfig) -> tuple[bytes, str, int, int]:
    """원본 바이트 → (업로드 바이트, 확장자, 폭, 높이). 부적합하면 _Reject."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        try:
            img = Image.open(BytesIO(data))
            img.load()
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError,
                Image.DecompressionBombWarning, ValueError) as e:
            raise _Reject(f"이미지 아님/손상 ({e.__class__.__name__})") from e
    fmt = (img.format or "").upper()
    w, h = img.size
    if w < cfg.min_width or h < cfg.min_height:
        raise _Reject(f"너무 작음 {w}x{h}")
    if max(w / h, h / w) > 6:
        raise _Reject(f"비율 극단 {w}x{h}")

    animated = getattr(img, "is_animated", False)
    if fmt == "GIF" and animated and len(data) <= cfg.max_upload_bytes:
        return data, "gif", w, h                       # 애니메이션 그대로 (Slack에서 재생)
    if animated:
        img.seek(0)                                    # 큰 GIF·애니메이션 WebP → 첫 프레임
    else:
        img = ImageOps.exif_transpose(img)
    if fmt in ("PNG", "JPEG") and not animated and len(data) <= cfg.max_upload_bytes and max(w, h) <= cfg.max_side:
        return data, ("jpg" if fmt == "JPEG" else "png"), w, h
    out, ext = _encode(img, cfg)
    with Image.open(BytesIO(out)) as check:
        w2, h2 = check.size
    return out, ext, w2, h2


def _fetch_one(http: Http, ref: ImageRef, cfg: ImageConfig) -> FetchedImage:
    resp = http.get(ref.url, max_bytes=cfg.max_download_bytes, accept="image/*,*/*;q=0.5")
    ctype = (resp.content_type or "").lower()
    if ctype.startswith(("text/", "application/json")):
        raise _Reject(f"이미지가 아닌 응답 ({ctype})")
    data, ext, w, h = prepare(resp.content, cfg)
    return FetchedImage(ref=ref, data=data, ext=ext, width=w, height=h)


def fetch_images(http: Http, refs: list[ImageRef], cfg: ImageConfig) -> list[FetchedImage]:
    """후보를 받아 검증된 그림 리스트를 본문 순서로 반환 (hero 정책 적용)."""
    if not cfg.attach or not refs:
        return []
    hero = [r for r in refs if r.is_hero]
    body = [r for r in refs if not r.is_hero]

    def run(cands: list[ImageRef]) -> list[FetchedImage]:
        out: list[FetchedImage] = []
        seen_hash: set[str] = set()
        for ref in cands:
            try:
                img = _fetch_one(http, ref, cfg)
            except (FetchError, _Reject) as e:
                logger.info("그림 제외 [%s] %s: %s", "hero" if ref.is_hero else ref.no, ref.url[-80:], e)
                continue
            digest = hashlib.sha1(img.data).hexdigest()
            if digest in seen_hash:                    # URL만 다른 같은 그림
                continue
            seen_hash.add(digest)
            out.append(img)
        return out

    if cfg.hero == "never":
        return run(body[: cfg.max_candidates])
    if cfg.hero == "always":
        return run((hero + body)[: cfg.max_candidates])
    got = run(body[: cfg.max_candidates])            # fallback
    return got if got else run(hero)
