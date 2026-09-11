#!/usr/bin/env bash
# 채널별 1편 실전송 테스트.
#
#   scripts/test_send.sh --check   # 사전 점검만 (토큰/스코프/Ollama/채널 — 채널엔 아무것도 안 보임)
#   scripts/test_send.sh           # 사전 점검 통과 시 topic마다 1편씩 실제 전송
#
# 채널 점검은 chat.scheduleMessage(+1h) 생성 후 즉시 삭제하는 방식이라 흔적이 남지 않는다.
set -u
cd "$(dirname "$0")/.." || exit 1
[ -f "$HOME/.paperbot_env" ] && . "$HOME/.paperbot_env"

PY=./venv/bin/python

# ── 1) 사전 점검 ─────────────────────────────────────────────
$PY - <<'EOF'
import os, sys, time
import requests, yaml

ok = True

# Ollama 서버
try:
    v = requests.get("http://localhost:11434/api/version", timeout=5).json()
    print(f"✅ Ollama 서버 응답 (v{v.get('version')})")
except Exception:
    print("❌ Ollama 서버 무응답 — tmux 세션 확인: tmux attach -t ollama")
    ok = False

# S2 키 (선택 사항)
s2 = os.environ.get("S2_API_KEY", "").strip()
if s2 and len(s2) >= 20 and "발급" not in s2:
    print(f"✅ S2_API_KEY 로드됨 (…{s2[-4:]}) — 요청 간격 1.2s")
else:
    print("ℹ️  S2_API_KEY 없음 — 무키 모드로 동작 (느리지만 정상)")

# Slack 토큰
tok = os.environ.get("SLACK_BOT_TOKEN", "").strip()
if not tok.startswith("xoxb-") or "여기에" in tok or "토큰" in tok:
    print("❌ SLACK_BOT_TOKEN이 비었거나 플레이스홀더 — ~/.paperbot_env를 채우세요.")
    sys.exit(1)

H = {"Authorization": f"Bearer {tok}"}
J = {**H, "Content-Type": "application/json; charset=utf-8"}

r = requests.post("https://slack.com/api/auth.test", headers=H, timeout=15).json()
if not r.get("ok"):
    print(f"❌ 토큰 무효: {r.get('error')}")
    sys.exit(1)
print(f"✅ 토큰 유효 — 봇 @{r.get('user')} / 워크스페이스 {r.get('team')}")

# files:write (figure 첨부) — 업로드 URL만 발급해 보고 사용하지 않음
r = requests.post("https://slack.com/api/files.getUploadURLExternal", headers=H,
                  data={"filename": "probe.png", "length": "10"}, timeout=15).json()
if r.get("ok"):
    print("✅ files:write OK — figure 첨부 가능")
else:
    print(f"⚠️  files:write 없음({r.get('error')}) — figure 생략, 텍스트만 발행됨")

# 채널별 전송 가능 여부 — 예약 메시지 생성 후 즉시 삭제 (채널에 안 보임)
cfg = yaml.safe_load(open("config.yaml"))
default_ch = (cfg.get("slack") or {}).get("channel")
hints = {
    "channel_not_found": "→ 채널명 오타이거나, 비공개 채널에 봇 미초대",
    "not_in_channel":    "→ 채널에서 /invite @봇이름 실행 필요",
    "is_archived":       "→ 보관된 채널",
}
for t in cfg["topics"]:
    name, ch = t["name"], t.get("channel") or default_ch
    r = requests.post("https://slack.com/api/chat.scheduleMessage", headers=J,
                      json={"channel": ch, "text": "connectivity probe",
                            "post_at": int(time.time()) + 3600}, timeout=15).json()
    if r.get("ok"):
        requests.post("https://slack.com/api/chat.deleteScheduledMessage", headers=J,
                      json={"channel": r["channel"],
                            "scheduled_message_id": r["scheduled_message_id"]}, timeout=15)
        print(f"✅ {name:28s} → {ch} 전송 가능")
    else:
        print(f"❌ {name:28s} → {ch}: {r.get('error')} {hints.get(r.get('error'), '')}")
        ok = False

sys.exit(0 if ok else 1)
EOF
rc=$?

if [ "${1:-}" = "--check" ]; then
    exit $rc
fi
if [ $rc -ne 0 ]; then
    echo
    echo "사전 점검 실패 — 위 ❌ 항목을 고친 뒤 다시 실행하세요."
    exit 1
fi

# ── 2) topic별 1편 실전송 ────────────────────────────────────
echo
echo "사전 점검 통과 — topic별 1편씩 전송을 시작합니다 (topic당 3~6분 소요)."
for t in $($PY -c "import yaml; print(' '.join(x['name'] for x in yaml.safe_load(open('config.yaml'))['topics']))"); do
    echo
    echo "═══ ${t}: 검색 → 필터 → 요약 → 전송 (1편) ═══"
    $PY run.py --topic "$t" --limit 1
done
echo
echo "완료 — 각 채널에서 메시지·스레드·figure를 확인하세요."
