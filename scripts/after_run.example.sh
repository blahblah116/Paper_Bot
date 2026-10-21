#!/usr/bin/env bash
# Optional hook: scripts/loop.sh runs scripts/after_run.sh (gitignored) right after the daily paperbot run,
# with a 3-hour timeout. Use it for follow-up jobs that share the same GPU/model.
# 선택 훅: loop.sh가 매일 paperbot 실행 직후 scripts/after_run.sh(gitignore)를 3시간 제한으로 호출한다.
# 같은 GPU·모델을 쓰는 후속 작업(예: 다른 봇)을 여기서 돌린다.
#
#   cp scripts/after_run.example.sh scripts/after_run.sh && chmod +x scripts/after_run.sh
set -u
exec "$HOME/articlebot/scripts/run_once.sh"
