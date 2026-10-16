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

# 공용 서버 VRAM 체크 — 27b(64K ctx)가 GPU 1,2에 ~11GB씩 올라가므로, 장당 13GB 미만이면
# 타 사용자 잡을 OOM으로 죽일 수 있다. 부족하면 30분 간격으로 최대 2시간 기다리고, 그래도 안 되면 그날은 건너뜀.
VRAM_GPUS="1,2"
VRAM_MIN_MIB=13312          # 장당 최소 여유 (13GB)
VRAM_WAIT_SEC=1800          # 재확인 간격 (30분)
VRAM_MAX_WAIT_SEC=7200      # 최대 대기 (2시간)

vram_ok() {
    local free
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id="$VRAM_GPUS" 2>/dev/null) || {
        echo "[loop] nvidia-smi 실패 — VRAM 체크 생략하고 실행"
        return 0
    }
    echo "[loop] GPU ${VRAM_GPUS} 여유 VRAM(MiB): $(echo $free | tr ' ' '/') (기준 ${VRAM_MIN_MIB})"
    for f in $free; do
        [ "$f" -ge "$VRAM_MIN_MIB" ] || return 1
    done
    return 0
}

echo "[loop] paperbot 러너 시작 — 매일 ${AT} 실행"

while true; do
    now=$(date +%s)
    target=$(date -d "today ${AT}" +%s)
    if [ "$target" -le "$now" ]; then
        target=$(date -d "tomorrow ${AT}" +%s)
    fi
    echo "[loop] 다음 실행: $(date -d "@${target}" '+%F %T') ($(( (target - now) / 60 ))분 대기)"
    sleep $(( target - now ))

    waited=0
    ready=0
    while true; do
        if vram_ok; then ready=1; break; fi
        [ "$waited" -ge "$VRAM_MAX_WAIT_SEC" ] && break
        echo "[loop] VRAM 부족 — $(( VRAM_WAIT_SEC / 60 ))분 후 재확인 (누적 대기 $(( waited / 60 ))분)"
        sleep "$VRAM_WAIT_SEC"
        waited=$(( waited + VRAM_WAIT_SEC ))
    done
    if [ "$ready" -ne 1 ]; then
        msg="$(date '+%F %T') [loop] VRAM 부족으로 $(( VRAM_MAX_WAIT_SEC / 60 ))분 대기 후에도 실행 불가 — 오늘 실행 건너뜀"
        echo "$msg"
        echo "$msg" >> logs/cron.log
        continue
    fi

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

    # 후속 작업 훅 — paperbot이 끝난 직후 같은 GPU·모델로 돌릴 작업 (예: articlebot).
    # scripts/after_run.sh는 로컬 전용(gitignore). paperbot 결과(rc)와 무관하게 실행하되,
    # 멈춰도 다음 날 paperbot이 밀리지 않게 3시간 제한. VRAM 부족으로 건너뛴 날은 훅도 실행 안 됨.
    if [ -x scripts/after_run.sh ]; then
        echo "[loop] 후속 작업 시작: $(date '+%F %T')"
        timeout 3h scripts/after_run.sh
        echo "[loop] 후속 작업 종료: $(date '+%F %T') (exit $?)"
    fi
done
