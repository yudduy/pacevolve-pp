#!/bin/bash
# Pod-side candidate archiver v3 (hash-dedup). The driver overwrites each
# worker's solution.cpp every rollout AND resets it to the seed between steps,
# so step-boundary snapshots capture the SEED, not the high-scoring candidates.
# Fix: poll all four workers every 15s and save every DISTINCT solution.cpp
# (dedup by content hash). Rollouts take minutes, so a 15s poll cannot miss a
# candidate that ever lands on disk. archive/ rides pod_chain's 2-hourly tar to
# W&B; harvest re-evaluates every distinct candidate across a common seed panel.
set -u
RID="${1:?rid}"
D="/workspace/pp/tasks/rectangle_free_grid/results/job_${RID}"
AR="$D/archive"; mkdir -p "$AR"
SEEN="$AR/.hashes"; : > "$SEEN"
echo "$(date -u +%FT%TZ) archiver v3 (hash-dedup) armed rid=$RID" >> "$AR/archiver.log"
N=0
while true; do
  STEP=$(grep -hoE "Step [0-9]+:" "$D"/logs/*.log 2>/dev/null | grep -oE "[0-9]+" | sort -n | tail -1)
  STEP="${STEP:-x}"
  for w in 0 1 2 3; do
    SRC="$D/results/rollout_workers/w${w}/src/solution.cpp"
    [ -f "$SRC" ] || continue
    H=$(md5sum "$SRC" 2>/dev/null | cut -d' ' -f1)
    [ -z "$H" ] && continue
    if ! grep -q "$H" "$SEEN" 2>/dev/null; then
      echo "$H" >> "$SEEN"
      N=$((N+1))
      cp "$SRC" "$AR/cand_$(printf '%04d' "$N")_step${STEP}_w${w}_${H:0:8}.cpp" 2>/dev/null
      echo "$(date -u +%FT%TZ) saved cand $N step=$STEP w=$w $H" >> "$AR/archiver.log"
    fi
  done
  sleep 15
done
