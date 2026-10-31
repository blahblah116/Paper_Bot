"""llama-swap helpers: running-model lookup, server context check, and a safe unload.
llama-swap(로컬 LLM 프록시) 관리 헬퍼 — 실행 중 모델 조회·실제 컨텍스트 확인·안전한 언로드.

봇들(paperbot·articlebot·zoterobot)은 llama-swap(기본 http://127.0.0.1:8090)의 OpenAI 호환 /v1로 모델을 부른다.
llama-swap은 요청이 오면 llama-server(llama.cpp CUDA 빌드)를 띄우고 마지막 요청 후 ttl이 지나면 내린다
(설정: ~/llama-swap/config.yaml).

- 상태 확인은 llama-swap의 /upstream/<model>/… 대신 /running이 알려주는 llama-server 주소로 직접 보낸다.
  /upstream은 모델이 내려가 있으면 16GB를 다시 올리기 때문이다.
- llama-swap의 언로드(/api/models/unload/<model>)는 진행 중인 요청을 기다리지 않고 끊는다(2026-10-23 실측: 스트림이
  오류 없이 중간에 닫힘). 그래서 unload()는 처리 중·대기 요청이 0인 것을 두 번 연속 확인한 뒤에만 내린다
  — 진행 중인 요청은 끝까지 받은 뒤 언로드.
"""
from __future__ import annotations

import logging
import re
import time
from urllib.parse import quote

import requests

logger = logging.getLogger(__name__)

_METRIC_RE = re.compile(r"^llamacpp:(requests_processing|requests_deferred)\s+([0-9.eE+-]+)\s*$", re.MULTILINE)


def origin(base_url: str) -> str:
    """'http://127.0.0.1:8090/v1' → 'http://127.0.0.1:8090' (llama-swap 관리 API의 기준 주소)."""
    return base_url.rstrip("/").removesuffix("/v1")


def running(origin_url: str, timeout: float = 5.0) -> list[dict]:
    """llama-swap /running → [{'model', 'state', 'proxy', 'ttl', …}]. 연결 실패는 예외 그대로."""
    r = requests.get(f"{origin_url}/running", timeout=timeout)
    r.raise_for_status()
    return r.json().get("running") or []


def is_loaded(origin_url: str, model: str) -> bool:
    """model이 올라가 있거나 올라가는 중인가. llama-swap에 닿지 않으면 False."""
    try:
        return any(m.get("model") == model for m in running(origin_url))
    except Exception as e:  # noqa: BLE001
        logger.debug("llama-swap /running 조회 실패: %s", e)
        return False


def upstream_url(origin_url: str, model: str) -> str | None:
    """model이 준비된(ready) 상태면 그 llama-server 주소, 아니면 None (조회 실패도 None)."""
    try:
        for m in running(origin_url):
            if m.get("model") == model and m.get("state") == "ready" and m.get("proxy"):
                return str(m["proxy"]).rstrip("/")
    except Exception as e:  # noqa: BLE001
        logger.debug("llama-swap /running 조회 실패: %s", e)
    return None


def server_ctx(origin_url: str, model: str) -> int | None:
    """떠 있는 llama-server의 슬롯당 컨텍스트(n_ctx). 모델이 내려가 있거나 모르면 None."""
    up = upstream_url(origin_url, model)
    if not up:
        return None
    try:
        props = requests.get(f"{up}/props", timeout=5).json()
        n = int((props.get("default_generation_settings") or {}).get("n_ctx") or 0)
        return n or None
    except Exception as e:  # noqa: BLE001
        logger.debug("llama-server /props 조회 실패: %s", e)
        return None


def parse_busy_metrics(text: str) -> int | None:
    """llama-server /metrics(Prometheus 텍스트)에서 처리 중 + 대기 요청 수. 항목이 없으면 None."""
    vals = [float(v) for _, v in _METRIC_RE.findall(text or "")]
    return int(round(sum(vals))) if vals else None


def busy_requests(upstream: str) -> int | None:
    """llama-server가 처리 중이거나 줄 세워 둔 요청 수. 서버에 닿지 않으면(이미 종료) None.

    --metrics가 켜져 있으면 대기열까지 보고, 없으면 /slots의 is_processing만 센다.
    """
    try:
        r = requests.get(f"{upstream}/metrics", timeout=5)
        if r.status_code == 200:
            n = parse_busy_metrics(r.text)
            if n is not None:
                return n
        slots = requests.get(f"{upstream}/slots", timeout=5).json()
        return sum(1 for s in slots if isinstance(s, dict) and s.get("is_processing"))
    except requests.RequestException:
        return None
    except ValueError:  # JSON 아님 — 모르면 바쁘지 않다고 보지 않는다
        return 1


def unload(origin_url: str, model: str, *, wait_s: float = 600.0, poll_s: float = 5.0) -> bool:
    """처리 중·대기 요청이 없을 때 model을 내린다. 이미 내려가 있으면 True.

    다른 봇의 요청(zoterobot DM 등)이 진행 중이면 끝날 때까지 poll_s 간격으로 기다리고, wait_s가 지나도 바쁘면
    내리지 않고 False — 그 경우 llama-swap ttl이 마지막 요청 뒤에 내린다. llama-swap에 닿지 않아도 False.
    """
    deadline = time.monotonic() + wait_s
    told = False
    idle = 0
    try:
        while True:
            entry = next((m for m in running(origin_url) if m.get("model") == model), None)
            if entry is None:
                logger.info("모델이 이미 내려가 있음 (%s)", model)
                return True
            if entry.get("state") == "ready" and entry.get("proxy"):
                n = busy_requests(str(entry["proxy"]).rstrip("/"))
                n = 0 if n is None else n          # 서버가 방금 끝남 → 다음 바퀴에 /running에서 사라진다
            else:
                n = 1                              # starting/stopping — 상태가 정해질 때까지 기다림
            if n == 0:
                idle += 1
                if idle >= 2:                      # 두 번 연속 한가할 때만 (요청 사이 틈에 끊지 않게)
                    break
                time.sleep(1.0)
                continue
            idle = 0
            if time.monotonic() >= deadline:
                logger.warning("모델 %s: 다른 요청이 계속 처리 중이라 언로드 건너뜀 — llama-swap ttl 후 자동으로 내려감", model)
                return False
            if not told:
                logger.info("모델 %s: 다른 요청(zoterobot DM 등) 처리 중 — 끝나면 언로드", model)
                told = True
            time.sleep(poll_s)
        r = requests.post(f"{origin_url}/api/models/unload/{quote(model, safe=':')}", timeout=60)
        if r.status_code == 200:
            logger.info("모델 언로드 완료 — GPU 반납 (%s)", model)
            return True
        logger.warning("모델 언로드 실패 (%s): HTTP %d %s", model, r.status_code, r.text[:200])
        return False
    except Exception as e:  # noqa: BLE001 — 언로드 실패는 치명적이지 않음 (ttl이 내린다)
        logger.debug("모델 언로드 실패(무시): %s", e)
        return False
