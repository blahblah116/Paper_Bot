#!/usr/bin/env bash
# zoterobot tmux 러너 — Slack 저장 명령 + DM 챗봇 상주 프로세스.
# 리스너 자체는 GPU를 쓰지 않는다(DM이 올 때만 llama-swap이 모델을 올리고 ttl 10분 뒤 내림).
# 프로세스가 죽으면 30초 뒤 재시작, 설정·인증 오류(exit 2)면 10분 뒤 재시도.
#
# 사용:
#   tmux new -d -s zoterobot '~/paperbot/scripts/zoterobot.sh'
#   tail -f ~/paperbot/logs/zoterobot.log
set -u
cd "$(dirname "$0")/.." || exit 1   # 리포 루트로 이동

while true; do
    # 토큰 환경변수 로드 — 매번 다시 읽으므로 ~/.paperbot_env 수정 후 재시작만 기다리면 된다
    [ -f "$HOME/.paperbot_env" ] && . "$HOME/.paperbot_env"
    echo "[zoterobot] 시작: $(date '+%F %T')"
    ./venv/bin/python -m zoterobot
    rc=$?
    wait_sec=30
    [ "$rc" -eq 2 ] && wait_sec=600
    echo "[zoterobot] 종료: $(date '+%F %T') (exit ${rc}) — ${wait_sec}초 후 재시작"
    sleep "$wait_sec"
done
