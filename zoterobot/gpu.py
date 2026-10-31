"""GPU availability check before loading the chat model on the shared server.
공용 GPU 서버에서 챗봇 모델을 올리기 전 여유 VRAM 확인.

모델이 이미 llama-swap에 올라가 있으면(올라가는 중 포함) 통과. 아니면 GPU별 여유 VRAM이 기준 미만일 때 GpuBusy를 던져
DM에 "바쁨"으로 바로 안내한다(대기하지 않음 — 사용자 결정).
"""
from __future__ import annotations

import logging
import subprocess

from paperbot import llmserver

from .config import ChatConfig

logger = logging.getLogger(__name__)


class GpuBusy(Exception):
    pass


def model_loaded(cfg: ChatConfig) -> bool:
    return llmserver.is_loaded(cfg.llm_url, cfg.model)


def gpu_free_mib(gpus: str) -> list[int] | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits",
                              f"--id={gpus}"], capture_output=True, text=True, timeout=15)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return [int(x) for x in out.stdout.split() if x.strip().isdigit()]


def ensure_gpu(cfg: ChatConfig) -> None:
    """모델이 이미 올라가 있으면 통과. 아니면 GPU별 여유 VRAM 확인 — 부족하면 GpuBusy."""
    if model_loaded(cfg):
        return
    free = gpu_free_mib(cfg.vram_gpus)
    if free is None:
        logger.warning("nvidia-smi 실패 — VRAM 체크 생략")
        return
    if any(f < cfg.vram_min_mib for f in free):
        shown = "/".join(f"{f / 1024:.1f}" for f in free)
        raise GpuBusy(f"GPU {cfg.vram_gpus} 여유 {shown}GB, 필요 장당 {cfg.vram_min_mib / 1024:.0f}GB")
