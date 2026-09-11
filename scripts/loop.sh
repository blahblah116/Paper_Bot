#!/usr/bin/env bash
# paperbot tmux 러너 — 매일 지정 시각(기본 07:00)에 한 번 실행하고,
# 실행이 끝나면 봇 프로세스는 종료(GPU 즉시 반납), 세션은 다음 시각까지 잠든다.
#
# 사용:
#   tmux new -d -s paperbot '~/paperbot/scripts/loop.sh 07:00'
#   tmux attach -t paperbot   # 상태 확인 (Ctrl+b d 로 분리)
set -u

AT="${1:-07:00}"
cd "$(dirname "$0")/.." || exit 1   # 리포 루트로 이동

echo "[loop] paperbot 러너 시작 — 매일 ${AT} 실행"

while true; do
    now=$(date +%s)
    target=$(date -d "today ${AT}" +%s)
    if [ "$target" -le "$now" ]; then
        target=$(date -d "tomorrow ${AT}" +%s)
    fi
    echo "[loop] 다음 실행: $(date -d "@${target}" '+%F %T') ($(( (target - now) / 60 ))분 대기)"
    sleep $(( target - now ))

    # 토큰 등 환경변수 로드 (없으면 봇이 설정 오류로 깨끗하게 중단됨)
    [ -f "$HOME/.paperbot_env" ] && . "$HOME/.paperbot_env"

    # Ollama가 죽어 있으면 자동으로 살린다 (자가 치유 — ollama 세션이 필수가 아니게 됨)
    if ! curl -s --max-time 5 http://localhost:11434/api/version >/dev/null; then
        echo "[loop] Ollama 무응답 — 재시작 시도"
        tmux kill-session -t ollama 2>/dev/null
        tmux new-session -d -s ollama \
            'CUDA_VISIBLE_DEVICES=1,2 GGML_VK_VISIBLE_DEVICES=1,2 OLLAMA_CONTEXT_LENGTH=65536 $HOME/ollama/bin/ollama serve 2>&1 | tee -a $HOME/ollama/serve.log'
        for _ in $(seq 1 30); do
            curl -s --max-time 2 http://localhost:11434/api/version >/dev/null && break
            sleep 1
        done
        if curl -s --max-time 5 http://localhost:11434/api/version >/dev/null; then
            echo "[loop] Ollama 재시작 완료"
        else
            echo "[loop] Ollama 재시작 실패 — 이번 실행은 LLM 없이 진행되어 실패 기록될 수 있음"
        fi
    fi

    echo "[loop] 실행 시작: $(date '+%F %T')"
    ./venv/bin/python run.py >> logs/cron.log 2>&1
    rc=$?
    echo "[loop] 실행 종료: $(date '+%F %T') (exit ${rc}) — 상세는 logs/cron.log"
done
