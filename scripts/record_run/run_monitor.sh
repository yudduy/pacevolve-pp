#!/bin/bash
# Record-run watchdog v3 — watches the CONTROLLER log (where step/reward/skip
# lines actually land), not chain.log. Emits: each new trained step with its
# max-score (reward*0.1334), OOM bursts, skip-storm alarm, chain failures,
# pod EXITED/GONE, hourly balance. Runs on the Mac; the pod is autonomous.
set -u
SP="$(cd "$(dirname "$0")" && pwd)"
. "$SP/record_run.state"
KEY=$HOME/.runpod/ssh/runpodctl-ssh-key
ENVF=/Users/c-dnguyen/Documents/project/ttt-discover/.env
CLOG="/workspace/pp/tasks/rectangle_free_grid/results/job_${RID}/logs"
STEPCNT=0; LASTPROG_T=$SECONDS; FAILS=0; LASTBEAT=0; OOMCNT=0
SSHC() { ssh -o ConnectTimeout=30 -o ServerAliveInterval=10 -o StrictHostKeyChecking=no -i "$KEY" -p "$SSH_PORT" "root@$SSH_IP" "$1" 2>/dev/null; }
APIQ() {
  # shellcheck disable=SC1090
  . "$ENVF" 2>/dev/null
  curl -sm 20 -X POST "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" \
    -H 'Content-Type: application/json' \
    -d '{"query":"query { myself { clientBalance currentSpendPerHr pods { id desiredStatus } } }"}' 2>/dev/null
}
PODSTATE() { APIQ | POD="$POD" python3 -c "
import sys, json, os
try: d = json.load(sys.stdin)['data']['myself']
except Exception: print('API-ERR'); raise SystemExit
for p in d['pods']:
    if p['id']==os.environ['POD']: print(p['desiredStatus']); break
else: print('GONE')" 2>/dev/null; }
while true; do
  # terminal chain states first
  END=$(SSHC "grep -hoE 'CHAIN COMPLETE rc=[0-9]+|CHAIN-FAIL [a-z-]+' /workspace/chain.log 2>/dev/null | tail -1")
  [ -n "$END" ] && { echo "CHAIN END: $END — harvest W&B rfg-$RID-results"; exit 0; }
  # step lines from controller log (source of truth)
  STEPS=$(SSHC "grep -hE 'Step [0-9]+: mean_reward' \$(ls -t $CLOG/*.log 2>/dev/null | head -1) 2>/dev/null")
  if [ -n "$STEPS" ]; then
    FAILS=0
    NOW=$(echo "$STEPS" | grep -c "Step ")
    if [ "$NOW" -gt "$STEPCNT" ]; then
      echo "$STEPS" | awk -v n=$STEPCNT 'NR>n' | python3 -c "
import sys, re
for ln in sys.stdin:
    m=re.search(r'Step (\d+): mean_reward=([0-9.]+) max_reward=([0-9.]+) skipped=(\w+).*?trained_samples.: (\d+)?', ln)
    if not m:
        m2=re.search(r'Step (\d+):.*max_reward=([0-9.]+) skipped=(\w+)', ln)
        if m2: print(f'  step {m2.group(1)} SKIPPED (max~{float(m2.group(2))*0.1334:.4f})')
        continue
    s,mn,mx,sk,ts=m.groups()
    print(f'  step {s}: max_score={float(mx)*0.1334:.4f} mean={float(mn)*0.1334:.4f} trained={ts or \"?\"} skipped={sk}')"
      # skip-storm check over last 5
      RECENT=$(echo "$STEPS" | tail -5 | grep -c "skipped=True")
      [ "${RECENT:-0}" -ge 3 ] && echo "ALARM: $RECENT of last 5 steps skipped — advisor stalling (likely OOM storm). Consider harvest-early + stop pod $POD."
      STEPCNT=$NOW; LASTPROG_T=$SECONDS
    elif [ $((SECONDS - LASTPROG_T)) -gt 2400 ]; then
      echo "WARN: no new STEP in $(((SECONDS-LASTPROG_T)/60)) min (last step count=$STEPCNT) — long rollout or true stall on pod $POD"
      LASTPROG_T=$SECONDS
    fi
  else
    FAILS=$((FAILS+1))
    if [ "$FAILS" -ge 5 ]; then
      ST=$(PODSTATE)
      case "$ST" in
        EXITED) echo "POD STOPPED — harvest W&B rfg-$RID-results, then delete pod $POD shell."; exit 0;;
        GONE)   echo "POD GONE $POD (terminate-after or delete) — harvest W&B rfg-$RID-results."; exit 0;;
        *)      echo "WARN: pod $POD state=$ST, ssh unreachable x$FAILS";;
      esac
      FAILS=0
    fi
  fi
  # OOM burst tracking
  NEWOOM=$(SSHC "grep -c RESOURCE_EXHAUSTED /workspace/chain.log 2>/dev/null" | tr -dc 0-9)
  if [ -n "$NEWOOM" ] && [ "$NEWOOM" -gt "$OOMCNT" ]; then
    echo "  (sampling OOMs: $NEWOOM total, +$((NEWOOM-OOMCNT)) — intermittent long-prompt prefill; driver skips + continues)"
    OOMCNT=$NEWOOM
  fi
  if [ $((SECONDS - LASTBEAT)) -ge 3600 ]; then
    BAL=$(APIQ | python3 -c "import sys,json;d=json.load(sys.stdin)['data']['myself'];print('balance=%.2f spend/hr=%.2f pods=%d' % (d['clientBalance'],d['currentSpendPerHr'],len(d['pods'])))" 2>/dev/null)
    echo "HEARTBEAT rid=$RID pod=$POD steps=$STEPCNT ooms=$OOMCNT $BAL"
    LASTBEAT=$SECONDS
  fi
  sleep 300
done
