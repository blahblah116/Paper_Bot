#!/usr/bin/env bash
# articlebot 1회 실행 — scripts/loop.sh가 매일 paperbot 실행 직후 부른다 (VRAM 체크·llama-swap 자가 치유는 loop.sh가 끝낸 상태).
# 수동 실행도 가능: ~/paperbot/scripts/articlebot.sh   (인자는 그대로 전달, 예: --dry-run --limit 1)
set -u
cd "$(dirname "$0")/.." || exit 1   # 리포 루트로 이동
mkdir -p logs

# 토큰 — paperbot과 같은 봇
[ -f "$HOME/.paperbot_env" ] && . "$HOME/.paperbot_env"

echo "$(date '+%F %T') [articlebot.sh] articlebot 시작" >> logs/articlebot_cron.log
./venv/bin/python -m articlebot "$@" >> logs/articlebot_cron.log 2>&1
rc=$?
echo "$(date '+%F %T') [articlebot.sh] articlebot 종료 (exit ${rc})" >> logs/articlebot_cron.log
exit "$rc"
