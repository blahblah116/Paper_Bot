"""글 페이지 → 본문 텍스트 + 그림 후보 + 관련 링크.

1) 페이지를 받아 사이트별 규칙(SITE_RULES)으로 본문 컨테이너를 찾는다.
   규칙이 없는 사이트는 entry-content/article/main → 문단 밀도 휴리스틱 순으로 찾는다.
2) 컨테이너에서 보일러플레이트(공유 버튼·뉴스레터·관련 글 등)를 제거한 뒤
   문서 순서대로 걸으며 마크다운풍 텍스트를 만든다. 그림 자리에는 "[그림 N] 캡션"을 남겨
   LLM이 그림 내용을 문맥과 함께 보게 한다. 수식은 MathML의 TeX annotation으로 치환.
3) 그림은 <figure>/<picture>/<img> 단위로 1장만 고른다:
   - 라이트/다크 모드 이중 그림(DeepMind)은 다크 쪽을 버림
   - lazy-load(data-src·noscript)와 srcset(가장 큰 해상도) 처리
   - 아이콘·로고·SVG·placeholder 제외, 같은 그림의 크기 변형(-768x432, .width-1250, =w1440) 중복 제거
   - 캡션: figcaption → 근처의 caption 클래스/slot 요소(Google Research, blog.google) → 의미 있는 alt
4) 페이지 추출이 실패하거나 너무 짧으면 피드에 실린 본문(content:encoded)으로 폴백.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit

import lxml.html
from lxml import etree

from .config import Config
from .http import FetchError, Http
from .models import Article, Extracted, ImageRef

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- 사이트 규칙

@dataclass(frozen=True)
class SiteRule:
    hosts: tuple[str, ...]          # 호스트 접미사 매칭 ("blog.google" ↔ "blog.google", "x.blog.google")
    body: tuple[str, ...]           # 본문 컨테이너 XPath (앞에서부터 시도, 여러 개 매칭 시 텍스트 최다)
    drop: tuple[str, ...] = ()      # 컨테이너 기준 상대 XPath — 걷기 전에 제거


SITE_RULES: tuple[SiteRule, ...] = (
    # 본문 블록(rich-text)과 그림 블록(media__image)이 섹션 단위로 번갈아 나온다.
    SiteRule(("research.google",), ('//div[contains(@class,"blog-detail-wrapper")]',)),
    # deepmind.google 자체 템플릿. 일부 글은 blog.google로 리다이렉트된다 (아래 규칙).
    SiteRule(("deepmind.google",), ('//main[@id="page-content"]', "//main")),
    SiteRule(("blog.google",),
             ('//div[contains(@class,"uni-blog-article-container")]',),
             drop=(".//uni-portal", './/*[contains(@class,"uni-blog-article-tags")]')),
    # WordPress 계열 (NVIDIA Technical Blog / NVIDIA Blog / Engineering at Meta / Meta Newsroom)
    SiteRule(("developer.nvidia.com", "blogs.nvidia.com", "engineering.fb.com", "about.fb.com"),
             ('//div[contains(concat(" ",normalize-space(@class)," ")," entry-content ")]',)),
)

_GENERIC_BODY = (
    '//*[@itemprop="articleBody"]',
    '//article//*[contains(concat(" ",normalize-space(@class)," ")," entry-content ")]',
    '//*[contains(@class,"post-content") or contains(@class,"article-content") or contains(@class,"article-body")]',
    "//article",
    "//main",
    '//*[@role="main"]',
)

# ---------------------------------------------------------------- 태그 분류

_SKIP_TAGS = {
    "script", "style", "template", "svg", "button", "form", "input", "select", "textarea",
    "nav", "dialog", "object", "embed", "canvas", "audio", "head", "link", "meta", "title",
    "iframe",  # 영상 링크만 따로 수집
}
_BLOCK_TAGS = {
    "p", "div", "section", "article", "main", "header", "footer", "aside", "blockquote",
    "ul", "ol", "li", "dl", "dt", "dd", "pre", "figure", "figcaption", "details", "summary",
    "center", "address", "h1", "h2", "h3", "h4", "h5", "h6",
}
_HEADING_PREFIX = {"h1": "## ", "h2": "## ", "h3": "### ", "h4": "#### ", "h5": "#### ", "h6": "#### "}

# class/id 토큰을 -,_ 로 쪼갠 조각에 이 단어가 있으면 보일러플레이트로 보고 제거.
_BOILER_WORDS = {
    "share", "sharing", "social", "newsletter", "subscribe", "subscription", "related",
    "recommended", "breadcrumb", "breadcrumbs", "comments", "cookie", "promo", "advert",
    "sidebar", "popup", "modal", "sharedaddy", "jp", "pmz", "taxonomy", "byline",
    "toc", "skip", "signup",
}
_BOILER_TOKENS_EXACT = {"tags", "entry-tags", "post-tags", "article-tags", "author-bio", "author-box"}
_RELATED_HEADING_RE = re.compile(
    r"^(related (posts?|stories|articles|content|reading|resources)|more (from|stories|to explore)\b.*|"
    r"you (may|might) also like|read (more|next)|recommended( for you)?|latest (news|posts|stories))$",
    re.I,
)

# ---------------------------------------------------------------- 그림 판정

# 경로 조각 단위(/ _ - . 경계)로 매칭 — "sharepoint"·"logout" 같은 단어 일부는 걸리지 않음.
_JUNK_IMG_RE = re.compile(
    r"(?:^|[/_\-.])(logo|logos|icon|icons|favicon|avatar|avatars|sprite|emoji|badge|spinner|"
    r"placeholder|fallback|pixel|spacer|blank|gravatar|headshot|social|arrow|arrowimg|"
    r"downloadicon|loader|loading)(?:[/_\-.]|$)",
    re.I,
)
_JUNK_HOSTS = ("gravatar.com", "doubleclick.net", "facebook.com", "googletagmanager.com")
_LAZY_ATTRS = ("data-src", "data-lazy-src", "data-original", "data-full-src", "data-orig-file")
_SRCSET_ATTRS = ("data-srcset", "data-lazy-srcset", "srcset")

# ---------------------------------------------------------------- 링크 판정

_LINK_HOST_RE = re.compile(
    r"(^|\.)(arxiv\.org|openreview\.net|github\.com|huggingface\.co|doi\.org|dl\.acm\.org|"
    r"aclanthology\.org|openaccess\.thecvf\.com|proceedings\.mlr\.press|proceedings\.neurips\.cc|"
    r"papers\.nips\.cc|biorxiv\.org|medrxiv\.org|nature\.com|science\.org|kaggle\.com|"
    r"colab\.research\.google\.com|zenodo\.org|ieeexplore\.ieee\.org|research\.nvidia\.com|"
    r"storage\.googleapis\.com)$",
    re.I,
)
_YT_ID_RE = re.compile(r"(?:youtube(?:-nocookie)?\.com/(?:embed/|watch\?v=|shorts/)|youtu\.be/)([\w-]{11})")
_MAX_LINKS = 8
_MAX_VIDEOS = 3

_XML_DECL_RE = re.compile(r"^\s*<\?xml[^>]*\?>")
_WS_RE = re.compile(r"[ \t\r\f\v ​]+")
_ALL_WS_RE = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _ALL_WS_RE.sub(" ", text or "").strip()


def _tag(el) -> str:
    """lxml 태그명 (네임스페이스 제거, 소문자). 주석/PI면 ''."""
    if not isinstance(el.tag, str):
        return ""
    return etree.QName(el).localname.lower() if "}" in el.tag else el.tag.lower()


def _class_tokens(el) -> list[str]:
    return [t.lower() for t in (el.get("class") or "").split()]


def _is_boilerplate(el) -> bool:
    for tok in _class_tokens(el) + ([el.get("id").lower()] if el.get("id") else []):
        if tok in _BOILER_TOKENS_EXACT:
            return True
        if set(re.split(r"[-_]+", tok)) & _BOILER_WORDS:
            return True
    return el.get("aria-hidden") == "true" and not el.xpath(".//img")


def _is_dark_variant(el, stop) -> bool:
    """라이트/다크 이중 그림 중 다크 쪽인가 (el부터 stop 직전 조상까지 검사).
    'has-dark-mode-alternative', 'no-dark-mode-mobile' 같은 has-/no- 토큰은 무시."""
    node = el
    while node is not None and node is not stop:
        media = (node.get("media") or "").lower()
        if "prefers-color-scheme" in media and "dark" in media:
            return True
        for tok in _class_tokens(node):
            if tok.startswith(("has-", "no-")):
                continue
            if "dark" in re.split(r"[-_]+", tok):
                return True
        node = node.getparent()
    return False


def parse_srcset(srcset: str, max_side: int = 3000) -> str | None:
    """srcset에서 가장 큰 후보 URL (max_side 이하 중 최대, 없으면 가장 작은 것)."""
    cands: list[tuple[float, str]] = []
    for part in re.split(r",\s+", (srcset or "").strip()):
        bits = part.strip().split()
        if not bits:
            continue
        url = bits[0].rstrip(",")
        size = 1.0
        if len(bits) > 1:
            m = re.match(r"([\d.]+)([wx])", bits[1])
            if m:
                size = float(m.group(1)) * (1000 if m.group(2) == "x" else 1)
        cands.append((size, url))
    if not cands:
        return None
    under = [c for c in cands if c[0] <= max_side]
    return max(under)[1] if under else min(cands)[1]


def image_key(url: str) -> str:
    """같은 그림의 크기·포맷 변형을 같은 키로 (dedupe용)."""
    p = urlsplit(url)
    path = p.path
    path = re.sub(r"=[swh]\d[^/]*$", "", path)                         # Google 이미지 서비스 (=w1440-h810-n-nu)
    path = re.sub(r"\.\w{3,4}$", "", path)                              # 확장자
    path = re.sub(r"(\.(?:width|height|fill|max|min|format|scale|original)-[^./]+)+$", "", path)  # wagtail rendition
    path = re.sub(r"\.[0-9a-f]{8}$", "", path)                          # wagtail focal hash
    path = re.sub(r"-\d{2,5}x\d{2,5}$", "", path)                       # WordPress 크기 변형
    path = re.sub(r"-scaled$", "", path)
    return (p.netloc + path).lower()


def is_junk_image(url: str) -> bool:
    if not url or url.startswith("data:"):
        return True
    p = urlsplit(url)
    if p.scheme not in ("http", "https"):
        return True
    host = (p.hostname or "").lower()
    if any(host == h or host.endswith("." + h) for h in _JUNK_HOSTS):
        return True
    path = p.path.lower()
    if path.endswith(".svg") or "format-svg" in path or path.endswith((".ico", ".mp4", ".webm")):
        return True
    return bool(_JUNK_IMG_RE.search(path))


def _meaningful_alt(alt: str) -> str:
    """파일명 같은 alt('EarthAITechReport1_Overview')는 버리고 설명문만."""
    alt = _norm(alt)
    if len(alt) < 15 or " " not in alt:
        return ""
    return alt


# ---------------------------------------------------------------- 본문 걷기

class _Walker:
    """컨테이너를 문서 순서로 걸으며 블록 텍스트·그림·링크를 모은다."""

    def __init__(self, container, base_url: str, max_side: int):
        self.base_url = base_url
        self.max_side = max_side
        self.blocks: list[tuple[str, str]] = []     # (kind, text)
        self.buf: list[str] = []
        self.images: list[ImageRef] = []
        self.links: list[tuple[str, str]] = []
        self.videos: list[str] = []
        self._unit_ref: dict = {}                   # 그림 단위 요소 → ImageRef (None이면 버린 그림)
        self._consumed: set = set()                 # 그림 마커에 흡수된 캡션 요소
        self._seen_keys: set[str] = set()
        self._seen_links: set[str] = set()
        self._style: list[tuple[str, str]] = []     # 블록 스타일 스택 (kind, 접두어)
        self._find_media_units(container)

    # ------------------------------------------------------------ 그림 단위

    def _abs(self, url: str) -> str:
        return urljoin(self.base_url, url.strip())

    def _url_of(self, el) -> tuple[str, str] | None:
        """img/source/video 요소 하나에서 최선의 URL과 alt."""
        tag = _tag(el)
        if tag == "video":
            poster = el.get("poster")
            return (self._abs(poster), "") if poster else None
        for attr in _SRCSET_ATTRS:
            if el.get(attr):
                best = parse_srcset(el.get(attr), self.max_side)
                if best and not best.startswith("data:"):
                    return self._abs(best), el.get("alt") or ""
        if tag == "img":
            for attr in _LAZY_ATTRS + ("src",):
                v = el.get(attr)
                if v and not v.strip().startswith("data:"):
                    return self._abs(v), el.get("alt") or ""
        return None

    def _unit_url(self, unit) -> tuple[str, str] | None:
        """그림 단위(figure/picture/img/video)에서 대표 URL 하나. img를 source보다 우선."""
        els = [unit] if _tag(unit) in ("img", "video") else []
        els += [e for e in unit.iterdescendants() if _tag(e) in ("img", "video")]
        els += [e for e in unit.iter() if _tag(e) == "source"]
        light = [e for e in els if not _is_dark_variant(e, unit.getparent())]
        alt_fallback = ""
        for el in light + [e for e in els if e not in light]:   # 다크 변형뿐이면 그것이라도
            got = self._url_of(el)
            if not got:
                continue
            url, alt = got
            alt_fallback = alt_fallback or alt
            if not is_junk_image(url):
                return url, alt or alt_fallback
        return None

    def _caption_for(self, unit) -> tuple[str, object | None]:
        """캡션 텍스트와 그 요소. figcaption → 근처 caption 클래스/slot 요소."""
        if _tag(unit) == "figure":
            fc = unit.xpath(".//figcaption")
            if fc and _norm(fc[0].text_content()):
                return _norm(fc[0].text_content()), fc[0]
        node = unit
        for _ in range(4):                       # 조상 4단계까지 올라가며 주변 캡션 탐색
            parent = node.getparent()
            if parent is None:
                break
            if len(_norm(parent.text_content())) > 3000:   # 본문 전체 수준까지 올라왔으면 중단
                break
            for el in parent.iter():
                if el is unit or not isinstance(el.tag, str):
                    continue
                cls = (el.get("class") or "").lower()
                slot = (el.get("slot") or "").lower()
                if _tag(el) == "figcaption" or "caption" in slot or re.search(r"(^|[\s_-])caption", cls):
                    if unit in el.iterancestors() or el in unit.iterancestors():
                        continue
                    text = _norm(el.text_content())
                    if 0 < len(text) <= 1500:
                        return text, el
            node = parent
        return "", None

    def _find_media_units(self, container) -> None:
        claimed: list = []
        for el in container.iter():
            tag = _tag(el)
            if tag not in ("figure", "picture", "img", "video"):
                continue
            if any(anc in self._unit_ref for anc in el.iterancestors()):
                continue                         # 이미 상위 figure/picture에 속함
            if tag == "figure" and not el.xpath('.//*[local-name()="img" or local-name()="source" or (local-name()="video" and @poster)]'):
                continue                         # 표 등 그림 없는 figure는 일반 블록으로
            if tag == "video" and not el.get("poster"):
                continue
            got = self._unit_url(el)
            if not got:
                self._unit_ref[el] = None        # 정크 그림 — 마커 없이 건너뜀
                continue
            url, alt = got
            key = image_key(url)
            if key in self._seen_keys:
                self._unit_ref[el] = None
                continue
            self._seen_keys.add(key)
            caption, cap_el = self._caption_for(el)
            if cap_el is not None:
                self._consumed.add(cap_el)
            ref = ImageRef(url=url, no=len(self.images) + 1, caption=caption, alt=_meaningful_alt(alt))
            self.images.append(ref)
            self._unit_ref[el] = ref
            claimed.append(el)

    # ------------------------------------------------------------ 텍스트

    def _flush(self) -> None:
        """버퍼를 블록 하나로 확정. 접두어는 현재 블록 스타일(스택 top)을 따른다."""
        text = _WS_RE.sub(" ", "".join(self.buf))
        text = re.sub(r" *\n *", "\n", text).strip()
        self.buf = []
        if text:
            kind, prefix = self._style[-1] if self._style else ("p", "")
            self.blocks.append((kind, prefix + text))

    def _block_style(self, tag: str) -> tuple[str, str]:
        if tag in _HEADING_PREFIX:
            return "h", _HEADING_PREFIX[tag]
        if tag in ("li", "dt", "dd"):
            return "li", "- "
        if tag == "blockquote":
            return "quote", "> "
        # <li><p>..</p></li> 처럼 목록/인용 안의 일반 블록은 바깥 스타일을 물려받는다.
        if self._style and self._style[-1][0] in ("li", "quote"):
            return self._style[-1]
        return "p", ""

    def _add_link(self, href: str, label: str) -> None:
        url = self._abs(href)
        p = urlsplit(url)
        if p.scheme not in ("http", "https"):
            return
        m = _YT_ID_RE.search(url)
        if m:
            self._add_video(m.group(1))
            return
        host = (p.hostname or "").lower()
        is_pdf = p.path.lower().endswith(".pdf")
        if not (_LINK_HOST_RE.search(host) or is_pdf):
            return
        if host.endswith("github.com") and len([s for s in p.path.split("/") if s]) < 2:
            return                               # 조직/사용자 페이지는 제외, 저장소만
        if host.endswith("storage.googleapis.com") and not is_pdf:
            return                               # 그림 호스트 — PDF(기술 보고서)만
        key = url.split("#")[0].rstrip("/")
        if key in self._seen_links:
            return
        self._seen_links.add(key)
        label = _norm(label)[:80] or host
        self.links.append((label, url))

    def _add_video(self, yt_id: str) -> None:
        url = f"https://www.youtube.com/watch?v={yt_id}"
        if url not in self.videos:
            self.videos.append(url)

    def _math(self, el) -> str:
        tex = el.xpath('.//*[local-name()="annotation"][contains(@encoding,"tex")]/text()')
        if tex:
            t = _norm(tex[0])
            return f" $${t}$$ " if el.get("display") == "block" else f" ${t}$ "
        return " " + _norm(el.text_content()) + " "

    def _table(self, el) -> None:
        rows = []
        for tr in el.xpath(".//tr")[:60]:
            cells = [_norm(c.text_content()) for c in tr if _tag(c) in ("td", "th")]
            if any(cells):
                rows.append(" | ".join(cells))
        if rows:
            self.blocks.append(("table", "\n".join(rows)))

    def _emit_image_marker(self, ref: ImageRef) -> None:
        self._flush()
        prev = next((t for k, t in reversed(self.blocks) if k not in ("img", "table")), "")
        ref.context = prev[-300:]
        label = ref.caption or ref.alt
        self.blocks.append(("img", f"[그림 {ref.no}] {label}".rstrip()))

    def walk(self, el) -> None:
        tag = _tag(el)
        if not tag:                              # 주석/PI — tail만
            if el.tail:
                self.buf.append(el.tail)
            return
        if el in self._consumed:
            pass                                 # 캡션은 그림 마커에 흡수됨
        elif el in self._unit_ref:
            ref = self._unit_ref[el]
            if ref is not None:
                self._emit_image_marker(ref)
            elif tag == "figure":                # 그림이 정크면 figcaption 등 텍스트만 살림
                for c in el:
                    if _tag(c) not in ("img", "picture", "video"):
                        self.walk(c)
                self._flush()
        elif tag == "iframe":
            m = _YT_ID_RE.search(el.get("src") or el.get("data-src") or "")
            if m:
                self._add_video(m.group(1))
        elif tag == "noscript":                  # lazy-load 원본 <img>만 (그림 단위로 잡혔다면 마커)
            for d in el.iterdescendants():
                if self._unit_ref.get(d) is not None:
                    self._emit_image_marker(self._unit_ref[d])
        elif tag in _SKIP_TAGS or tag in ("img", "picture", "video", "source"):
            pass
        elif "youtube" in tag or "youtube" in (el.get("class") or "").lower():
            vid = el.get("video-id") or el.get("data-video-id") or el.get("videoid")
            if vid and re.fullmatch(r"[\w-]{11}", vid):
                self._add_video(vid)
        elif tag == "math":
            self.buf.append(self._math(el))
        elif tag.startswith("mjx-"):
            math = el.xpath('.//*[local-name()="math"]')
            if math:
                self.buf.append(self._math(math[0]))
        elif "katex-html" in _class_tokens(el):
            pass                                 # KaTeX 시각 렌더 (MathML 쪽만 사용)
        elif tag == "br":
            self.buf.append("\n")
        elif tag == "table":
            self._flush()
            self._table(el)
        elif tag == "pre":
            self._flush()
            code = (el.text_content() or "").strip("\n")
            if code.strip():
                if len(code) > 3000:
                    code = code[:3000] + "\n…"
                self.blocks.append(("pre", f"```\n{code}\n```"))
        elif tag in _BLOCK_TAGS:
            self._flush()
            self._style.append(self._block_style(tag))
            if el.text:
                self.buf.append(el.text)
            for c in el:
                self.walk(c)
            self._flush()
            self._style.pop()
        else:                                    # 인라인 (a, span, strong, em, code ...)
            if tag == "a" and el.get("href"):
                self._add_link(el.get("href"), el.text_content())
            if tag == "code" and el.text:
                self.buf.append("`")
            if el.text:
                self.buf.append(el.text)
            for c in el:
                self.walk(c)
            if tag == "code" and el.text:
                self.buf.append("`")
        if el.tail:
            self.buf.append(el.tail)

    def text(self) -> str:
        self._flush()
        out: list[str] = []
        prev_kind = ""
        for kind, text in self.blocks:
            if out:
                out.append("\n" if kind == prev_kind == "li" else "\n\n")
            out.append(text)
            prev_kind = kind
        return "".join(out).strip()


# ---------------------------------------------------------------- 컨테이너 탐색

def _rule_for(url: str) -> SiteRule | None:
    host = (urlsplit(url).hostname or "").lower()
    for rule in SITE_RULES:
        if any(host == h or host.endswith("." + h) for h in rule.hosts):
            return rule
    return None


def _p_text_len(el) -> int:
    return sum(len(_norm(p.text_content())) for p in el.iter("p"))


def _pick(doc, xpaths) -> object | None:
    for xp in xpaths:
        found = [e for e in doc.xpath(xp) if isinstance(getattr(e, "tag", None), str)]
        if found:
            return max(found, key=lambda e: len(_norm(e.text_content())))
    return None


def _density_container(doc):
    """문단 텍스트가 가장 많이 모이는 조상 (readability 방식의 간이 버전)."""
    scores: dict = {}
    for p in doc.iter("p"):
        n = len(_norm(p.text_content()))
        if n < 40:
            continue
        anc = p.getparent()
        for depth in (1, 2, 3):
            if anc is None:
                break
            scores[anc] = scores.get(anc, 0.0) + n / depth
            anc = anc.getparent()
    return max(scores, key=scores.get) if scores else None


def find_container(doc, url: str):
    rule = _rule_for(url)
    if rule:
        el = _pick(doc, rule.body)
        if el is not None:
            return el, rule
        logger.warning("사이트 규칙의 본문 선택자가 매칭되지 않음 — 휴리스틱으로 폴백: %s", url)
    for xp in _GENERIC_BODY:
        el = _pick(doc, (xp,))
        if el is not None and _p_text_len(el) >= 500:
            return el, None
    return _density_container(doc), None


def _drop_related_sections(container) -> None:
    """'Related posts' 같은 제목이 붙은 섹션 제거 — 클래스로는 구분이 안 되는 경우 (DeepMind)."""
    for h in container.xpath(".//h1|.//h2|.//h3|.//h4|.//h5|.//p[strong or b]"):
        if h.getparent() is None or not _RELATED_HEADING_RE.match(_norm(h.text_content())):
            continue
        target = next((a for a in h.iterancestors() if _tag(a) == "section"), None)
        if target is None or target is container or container not in target.iterancestors():
            target = h.getparent()               # 섹션이 없으면 제목의 부모 블록
        if target is not None and target is not container and target.getparent() is not None:
            target.drop_tree()


def _clean(container, rule: SiteRule | None) -> None:
    if rule:
        for xp in rule.drop:
            for el in container.xpath(xp):
                if el is not container:
                    el.drop_tree()
    _drop_related_sections(container)
    for el in list(container.iterdescendants()):
        if not isinstance(el.tag, str) or el.getparent() is None:
            continue
        if _is_boilerplate(el):
            el.drop_tree()


def _og_image(doc, base_url: str) -> str | None:
    for xp in ('//meta[@property="og:image"]/@content', '//meta[@name="twitter:image"]/@content',
               '//meta[@property="og:image:url"]/@content'):
        vals = doc.xpath(xp)
        if vals and vals[0].strip():
            return urljoin(base_url, vals[0].strip())
    return None


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = int(max_chars * 0.85)
    tail = max_chars - head
    return text[:head] + "\n\n…(중략)…\n\n" + text[-tail:]


def extract_html(html: str, base_url: str, *, max_side: int = 2400, fragment: bool = False) -> Extracted:
    """HTML 문서(또는 피드 본문 조각)에서 본문·그림·링크 추출.
    og:image는 is_hero=True로 맨 앞에 둔다 (쓸지는 images 단계의 hero 정책이 정함)."""
    html = _XML_DECL_RE.sub("", html, count=1)   # lxml은 encoding 선언이 붙은 str을 거부
    if fragment:
        container = lxml.html.fragment_fromstring(html, create_parent="div", base_url=base_url)
        doc, rule = container, None
    else:
        doc = lxml.html.document_fromstring(html, base_url=base_url)
        container, rule = find_container(doc, base_url)
        if container is None:
            return Extracted("", [], [], [], base_url, "page")
    _clean(container, rule)
    walker = _Walker(container, base_url, max_side)
    walker.walk(container)
    images = walker.images
    hero = None if fragment else _og_image(doc, base_url)
    if hero and not is_junk_image(hero) and image_key(hero) not in {image_key(r.url) for r in images}:
        images = [ImageRef(url=hero, no=0, is_hero=True)] + images
    title = ""
    if not fragment:
        vals = doc.xpath('//meta[@property="og:title"]/@content') or doc.xpath("//title/text()")
        title = _norm(vals[0]) if vals else ""
    return Extracted(
        text=walker.text(),
        images=images,
        links=walker.links[:_MAX_LINKS],
        videos=walker.videos[:_MAX_VIDEOS],
        final_url=base_url,
        origin="feed" if fragment else "page",
        title=title,
    )


def extract(cfg: Config, http: Http, article: Article) -> Extracted:
    """글 하나의 본문 추출 — 페이지 우선, 짧거나 실패하면 피드 본문으로 폴백."""
    min_chars = cfg.fetch.min_body_chars
    page: Extracted | None = None
    try:
        resp = http.get(article.url, max_bytes=15_000_000)
        if "html" not in (resp.content_type or "html").lower():
            raise FetchError(f"HTML이 아님 ({resp.content_type})")
        page = extract_html(resp.text, resp.final_url, max_side=cfg.images.max_side)
        if resp.final_url != article.url:
            logger.info("리다이렉트: %s → %s", article.url, resp.final_url)
    except (FetchError, etree.ParserError, ValueError) as e:
        logger.warning("페이지 추출 실패 (%s): %s", article.uid, e)

    feed: Extracted | None = None
    if article.feed_html and (page is None or len(page.text) < min_chars):
        try:
            feed = extract_html(article.feed_html, article.url, max_side=cfg.images.max_side, fragment=True)
        except (etree.ParserError, ValueError) as e:
            logger.warning("피드 본문 파싱 실패 (%s): %s", article.uid, e)

    if page is not None and (feed is None or len(page.text) >= len(feed.text)):
        result = page
    elif feed is not None:
        result = feed
        if page is not None:
            result.final_url = page.final_url
            if not result.images:
                result.images = page.images      # 그림은 페이지 쪽이라도 살림
        logger.info("피드 본문 사용 (%s): 페이지 %d자 < 피드 %d자",
                    article.uid, len(page.text) if page else 0, len(feed.text))
    else:
        result = Extracted("", [], [], [], article.url, "page")

    result.text = _truncate(result.text, cfg.llm.max_input_chars)
    logger.info("본문 추출 %s: %d자, 그림 후보 %d, 링크 %d, 영상 %d (%s)", article.uid, len(result.text),
                len(result.images), len(result.links), len(result.videos), result.origin)
    return result
