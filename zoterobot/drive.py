"""rclone 래퍼 — remote는 'gdrive:paper'처럼 Attanger 저장 폴더(= Zotero 링크 기본 디렉토리)를 가리킨다."""
from __future__ import annotations

import json
import logging
import subprocess

logger = logging.getLogger(__name__)


class DriveError(Exception):
    pass


class DriveUploader:
    def __init__(self, remote: str, rclone_bin: str = "rclone"):
        if not remote:
            raise DriveError("config library.drive_remote가 비어 있습니다.")
        self.remote = remote
        self.bin = rclone_bin

    def target(self, rel_path: str) -> str:
        return f"{self.remote}{'' if self.remote.endswith(':') else '/'}{rel_path}"

    def _run(self, args: list[str], what: str, timeout: int = 600) -> subprocess.CompletedProcess:
        try:
            proc = subprocess.run([self.bin, *args], capture_output=True, text=True, timeout=timeout)
        except FileNotFoundError as e:
            raise DriveError(f"rclone 실행 파일을 찾을 수 없습니다: {self.bin}") from e
        except subprocess.TimeoutExpired as e:
            raise DriveError(f"rclone {what} 시간 초과({timeout}s)") from e
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
            raise DriveError(f"rclone {what} 실패 (exit {proc.returncode}): {' / '.join(tail)}")
        return proc

    def upload(self, local_path: str, rel_path: str) -> str:
        dest = self.target(rel_path)
        self._run(["copyto", local_path, dest], "업로드")
        logger.info("Drive 업로드 완료: %s", dest)
        return dest

    def download(self, rel_path: str, local_path: str) -> None:
        self._run(["copyto", self.target(rel_path), local_path], "다운로드")

    def list_pdfs(self) -> list[dict]:
        """리모트 아래 모든 PDF (하위 폴더 포함) — rclone lsjson 항목(Path, Size, ModTime, ID …)."""
        proc = self._run(["lsjson", "-R", "--files-only", "--fast-list", "--include", "*.{pdf,PDF}", self.remote],
                         "목록 조회", timeout=300)
        return json.loads(proc.stdout or "[]")

    def check(self) -> None:
        """시작 시 점검 — 리모트 접근 가능 여부(토큰 만료 등)."""
        self._run(["lsf", "--max-depth", "1", "--dirs-only", self.remote], f"리모트 접근({self.remote})", timeout=60)
