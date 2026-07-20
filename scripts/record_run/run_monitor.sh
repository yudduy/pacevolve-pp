#!/bin/bash
# Record-run watchdog v2 (reaper-aware). Emits: new per-step reward lines,
# chain failures/completions, stall warnings, pod EXITED/GONE detection, and
# an hourly balance heartbeat. Runs on the Mac while it is awake; the pod is
# autonomous regardless.
set -u
SP="$(cd "$(dirname "$0")" && pwd)"
. "$SP/record_run.state"
KEY=$HOME/.runpod/ssh/runpodctl-ssh-key
ENVF=/Users/c-dnguyen/Documents/project/ttt-discover/.env
PAT='mean_reward|CHAIN-FAIL|DRIVER START|DRIVER END|SELF-REMOVED|Background engine crashed|CHAIN COMPLETE|artifact uploaded|imports OK|healthy-stable'
CNT=0; LASTPROG_T=$SECONDS; FAILS=0; LASTBEAT=0
SSHC() { ssh -o ConnectTimeout=12 -o StrictHostKeyChecking=no -i "$KEY" -p "$SSH_PORT" "root@$SSH_IP" "$1" 2>/dev/null; }
APIQ() {
  # shellcheck disable=SC1090
  . "$ENVF" 2>/dev/null
  curl -sm 20 -X POST "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" \
    -H 'Content-Type: application/json' \
    -d '{"query":"query { myself { clientBalance currentSpendPerHr pods { id desiredStatus costPerHr } } }"}' 2>/dev/null
}
PODSTATE() { APIQ | POD="$POD" python3 -c "
import sys, json, os
try:
    d = json.load(sys.stdin)['data']['myself']
except Exception:
    print('API-ERR'); raise SystemExit
for p in d['pods']:
    if p['id'] == os.environ['POD']:
        print(p['desiredStatus']); break
else:
    print('GONE')" 2>/dev/null; }
while true; do
  TOTAL=$(SSHC "grep -cE '$PAT' /workspace/chain.log 2>/dev/null" | tr -dc 0-9)
  if [ -n "$TOTAL" ]; then
    FAILS=0
    if [ "$TOTAL" -gt "$CNT" ]; then
      SSHC "grep -E '$PAT' /workspace/chain.log | awk -v n=$CNT 'NR>n'" | tail -30
      CNT=$TOTAL
      LASTPROG_T=$SECONDS
    elif [ $((SECONDS - LASTPROG_T)) -gt 2700 ]; then
      echo "WARN: no new progress line in $(( (SECONDS - LASTPROG_T) / 60 )) min — driver may be stalled (pod $POD rid $RID)"
      LASTPROG_T=$SECONDS
    fi
  else
    FAILS=$((FAILS + 1))
    if [ "$FAILS" -ge 3 ]; then
      ST=$(PODSTATE)
      case "$ST" in
        EXITED)
          echo "POD STOPPED (reaper fired) — run over. Harvest W&B artifact rfg-$RID-results, then remove pod $POD shell."
          exit 0;;
        GONE)
          echo "POD GONE: $POD no longer exists (terminate-after or external delete). Harvest W&B artifact rfg-$RID-results."
          exit 0;;
        RUNNING)
          echo "WARN: pod $POD RUNNING but ssh unreachable (x$FAILS) — network blip or sshd died";;
        *)
          echo "WARN: pod $POD state=$ST, ssh unreachable (x$FAILS)";;
      esac
      FAILS=0
    fi
  fi
  if [ $((SECONDS - LASTBEAT)) -ge 3600 ]; then
    BAL=$(APIQ | python3 -c "import sys,json;d=json.load(sys.stdin)['data']['myself'];print('balance=%.2f spend/hr=%.2f pods=%d' % (d['clientBalance'], d['currentSpendPerHr'], len(d['pods'])))" 2>/dev/null)
    echo "HEARTBEAT rid=$RID pod=$POD $BAL"
    LASTBEAT=$SECONDS
  fi
  sleep 300
done
