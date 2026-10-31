"""HTTP 공용 클라이언트 — 정직한 User-Agent, 호스트별 요청 간격, 재시도, 크기 상한.

- User-Agent는 브라우저를 흉내 내지 않고 봇임을 밝힌다 (config fetch.user_agent).
  실측: ai.meta.com은 Chrome UA에 400을 주고 일반 UA는 통과시켰다.
- 429/5xx는 Retry-After(최대 60초)를 존중해 최대 3회 시도.
- 응답 본문은 스트리밍으로 받으며 max_bytes를 넘으면 중단 (거대한 GIF/영상 방지).
- gzip/brotli 해제는 requests가 처리 (DeepMind는 압축 응답을 준다).
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from urllib.parse import urlsplit

import requests

from .config import FetchConfig

logger = logging.getLogger(__name__)

_RETRY_STATUS = {429, 500, 502, 503, 504}
_MAX_TRIES = 3
_CHARSET_RE = re.compile(r"charset=([\w.-]+)", re.I)


class FetchError(Exception):
    """재시도 후에도 받지 못한 경우 (HTTP 오류·크기 초과·연결 실패)."""


@dataclass
class Response:
    url: str              # 요청 URL
    final_url: str        # 리다이렉트 후 URL
    status: int
    content: bytes
    content_type: str

    @property
    def text(self) -> str:
        """HTML/XML 디코딩 — 헤더 charset 우선, 없으면 UTF-8.
        (requests 기본값은 text/html에 ISO-8859-1을 가정해 한글·특수문자가 깨진다)"""
        m = _CHARSET_RE.search(self.content_type or "")
        enc = m.group(1) if m else "utf-8"
        try:
            return self.content.decode(enc, errors="replace")
        except LookupError:
            return self.content.decode("utf-8", errors="replace")


class Http:
    def __init__(self, cfg: FetchConfig):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": cfg.user_agent,
            "Accept-Language": "en-US,en;q=0.9",
        })
        self.timeout = cfg.timeout
        self.min_interval = cfg.min_interval
        self._last: dict[str, float] = {}

    def _throttle(self, url: str) -> None:
        host = urlsplit(url).hostname or ""
        wait = self._last.get(host, 0.0) + self.min_interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last[host] = time.monotonic()

    def get(self, url: str, *, max_bytes: int | None = None, accept: str | None = None) -> Response:
        headers = {"Accept": accept} if accept else None
        last_err: Exception | None = None
        for attempt in range(1, _MAX_TRIES + 1):
            self._throttle(url)
            try:
                with self.session.get(url, timeout=self.timeout, stream=True,
                                      headers=headers, allow_redirects=True) as resp:
                    if resp.status_code in _RETRY_STATUS:
                        retry_after = resp.headers.get("Retry-After", "")
                        wait = min(int(retry_after), 60) if retry_after.isdigit() else 2 * attempt + 1
                        raise _Retryable(f"HTTP {resp.status_code}", wait)
                    if resp.status_code >= 400:
                        raise FetchError(f"HTTP {resp.status_code}: {url}")
                    declared = resp.headers.get("Content-Length", "")
                    if max_bytes and declared.isdigit() and int(declared) > max_bytes:
                        raise FetchError(f"크기 초과 ({int(declared) // 1024}KB > {max_bytes // 1024}KB): {url}")
                    chunks: list[bytes] = []
                    total = 0
                    for chunk in resp.iter_content(chunk_size=65536):
                        total += len(chunk)
                        if max_bytes and total > max_bytes:
                            raise FetchError(f"크기 초과 (>{max_bytes // 1024}KB): {url}")
                        chunks.append(chunk)
                    return Response(
                        url=url,
                        final_url=resp.url,
                        status=resp.status_code,
                        content=b"".join(chunks),
                        content_type=resp.headers.get("Content-Type", ""),
                    )
            except FetchError:
                raise
            except _Retryable as e:
                last_err = e
                wait = e.wait
            except requests.RequestException as e:
                last_err = e
                wait = 2 * attempt + 1
            if attempt < _MAX_TRIES:
                logger.debug("GET 재시도 %d/%d (%s초 후): %s — %s", attempt, _MAX_TRIES, wait, url, last_err)
                time.sleep(wait)
        raise FetchError(f"{url}: {last_err}")


class _Retryable(Exception):
    def __init__(self, msg: str, wait: int):
        super().__init__(msg)
        self.wait = wait
