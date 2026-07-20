#!/bin/bash
# Pod-side candidate archiver. The driver overwrites each worker's solution.cpp
# every rollout and saves no champion file. Per-step "max_reward" is a max over
# ONE resampled seed, so scores across steps are NOT comparable and the true
# champion can only be found by re-ranking a POOL on a common seed panel at
# harvest. So: whenever the controller log advances to a new step, snapshot all
# four workers' current solution.cpp into archive/ (they hold the just-finished
# step's candidates; the next step's advisor generation takes minutes, so the
# files are not yet overwritten). archive/ rides pod_chain's 2-hourly tar to W&B.
set -u
RID="${1:?rid}"
D="/workspace/pp/tasks/rectangle_free_grid/results/job_${RID}"
AR="$D/archive"; mkdir -p "$AR"
LASTSTEP=-1
echo "$(date -u +%FT%TZ) archiver armed rid=$RID" >> "$AR/archiver.log"
snap() {
  local K="$1"
  for w in 0 1 2 3; do
    local SRC="$D/results/rollout_workers/w${w}/src/solution.cpp"
    [ -f "$SRC" ] && cp "$SRC" "$AR/step$(printf '%03d' "$K")_w${w}.cpp" 2>/dev/null
  done
  # record this step's per-seed candidate scores for later cross-check
  grep -hoE "Candidate: \{[^}]*\}" "$LOG" 2>/dev/null | tail -4 >> "$AR/step_candidates.log"
  echo "$(date -u +%FT%TZ) archived step $K" >> "$AR/archiver.log"
}
while true; do
  LOG=$(ls -t "$D"/logs/*.log 2>/dev/null | head -1)
  [ -z "$LOG" ] && { sleep 20; continue; }
  K=$(grep -hoE "Step [0-9]+:" "$LOG" 2>/dev/null | grep -oE "[0-9]+" | sort -n | tail -1)
  [ -z "$K" ] && { sleep 30; continue; }
  if [ "$K" -gt "$LASTSTEP" ]; then
    # snapshot every step from LASTSTEP+1..K we may have skipped (usually just K)
    snap "$K"
    LASTSTEP="$K"
  fi
  sleep 25
done
