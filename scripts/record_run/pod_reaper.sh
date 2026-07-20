#!/bin/bash
# Kill GPU billing from inside the pod once the run is over (no runpodctl on
# secure images). Container exit => pod EXITED => only volume disk bills, and
# this pod has none. Two triggers:
#   1. chain.log contains "CHAIN COMPLETE"  (final W&B upload already done)
#   2. pod_chain.sh process gone for 10+ min without the marker (crashed) —
#      emergency-upload whatever exists to W&B first, then stop.
set -u
RID="${1:?rid}"
PP=/workspace/pp
PY=/workspace/pvenv/bin/python
LOG=/workspace/reaper.log
C=0

stop_container() {
  echo "$(date -u +%FT%TZ) reaper: $1 -> stopping container" >> "$LOG"
  sync
  pkill -f 'sleep infinity' 2>/dev/null
  pkill -x sshd 2>/dev/null
  kill -TERM 1 2>/dev/null
  sleep 25
  kill -KILL 1 2>/dev/null
  pkill -KILL -f 'sleep' 2>/dev/null
  echo "$(date -u +%FT%TZ) reaper: still alive after kill attempts" >> "$LOG"
}

emergency_upload() {
  ST=/workspace/stage_emergency
  TARP="/workspace/rfg_${RID}_emergency.tgz"
  rm -rf "$ST"; mkdir -p "$ST"
  cp -r "$PP/tasks/rectangle_free_grid/results/job_${RID}" "$ST/" 2>/dev/null
  cp "$PP/tasks/rectangle_free_grid/config/config_${RID}.yaml" "$ST/" 2>/dev/null
  tail -c 2000000 /workspace/server.log > "$ST/server_tail.log" 2>/dev/null
  tail -c 2000000 /workspace/chain.log  > "$ST/chain_tail.log"  2>/dev/null
  tail -c 200000  /workspace/bootstrap.log > "$ST/bootstrap_tail.log" 2>/dev/null
  tar -czf "$TARP" -C "$ST" . 2>/dev/null
  ( set -a; . "$PP/.env" 2>/dev/null; set +a
    RID="$RID" TAG="emergency" TARPATH="$TARP" "$PY" /workspace/upload_artifact.py ) \
    >> "$LOG" 2>&1 && echo "$(date -u +%FT%TZ) reaper: emergency artifact uploaded" >> "$LOG"
}

echo "$(date -u +%FT%TZ) reaper: armed rid=$RID" >> "$LOG"
while true; do
  if grep -q "CHAIN COMPLETE" /workspace/chain.log 2>/dev/null; then
    sleep 60
    stop_container "chain-complete"
    sleep 300
  fi
  if pgrep -f pod_chain.sh >/dev/null 2>&1; then
    C=0
  else
    C=$((C + 1))
    if [ "$C" -ge 60 ]; then
      emergency_upload
      stop_container "chain-process-gone-10m"
      sleep 300
    fi
  fi
  sleep 10
done
