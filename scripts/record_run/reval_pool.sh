#!/bin/bash
# Robust re-ranking of a candidate pool: compile+eval every *.cpp in DIR across
# a fixed seed panel, rank by MEAN score (cross-seed = the real-judge proxy),
# print the robust champion + how many seeds it clears the record. This is the
# ONLY valid champion selection — per-step max scores use different seeds and
# are not comparable.
# Usage: reval_pool.sh <dir-of-cpp> [seed_lo] [seed_hi] [record]
set -u
PP=/Users/c-dnguyen/Documents/project/pacevolve-pp
DIR="${1:?dir of cpp}"; LO="${2:-0}"; HI="${3:-19}"; REC="${4:-0.583286}"
PY="$PP/.venv/bin/python"
EVAL="$PP/tasks/rectangle_free_grid/eval/eval_rfg.py"
OUT="$DIR/reval"; mkdir -p "$OUT"
JOBS="$OUT/jobs.txt"; : > "$JOBS"
for f in "$DIR"/*.cpp; do
  [ -f "$f" ] || continue
  b=$(basename "$f" .cpp)
  for s in $(seq "$LO" "$HI"); do
    echo "$f|$b|$s" >> "$JOBS"
  done
done
echo "re-evaluating $(ls "$DIR"/*.cpp 2>/dev/null | wc -l | tr -d ' ') candidates x $((HI-LO+1)) seeds ..."
run_one() {
  IFS='|' read -r f b s <<< "$1"
  sc=$("$PY" "$EVAL" --solution "$f" --seed "$s" \
        --build_dir "$OUT/build_${b}_s${s}" --tl 1.0 2>/dev/null \
        | grep -oE "'score': -?[0-9.]+" | head -1 | grep -oE "\-?[0-9.]+")
  echo "$b $s ${sc:-NA}"
}
export -f run_one; export PY EVAL OUT
cat "$JOBS" | xargs -P 8 -I{} bash -c 'run_one "$@"' _ {} > "$OUT/raw.txt" 2>/dev/null
"$PY" - "$OUT/raw.txt" "$REC" <<'PY'
import sys, statistics as st
raw, rec = sys.argv[1], float(sys.argv[2])
from collections import defaultdict
d = defaultdict(list)
for ln in open(raw):
    p = ln.split()
    if len(p) != 3: continue
    b, s, sc = p
    try: sc = float(sc)
    except: continue
    if sc < 0: sc = 0.0   # compile/invalid -> 0
    d[b].append(sc)
rows = []
for b, v in d.items():
    rows.append((st.mean(v), st.median(v), min(v), max(v), sum(1 for x in v if x >= rec), len(v), b))
rows.sort(reverse=True)
print(f"\n{'candidate':<16} {'mean':>7} {'med':>7} {'min':>7} {'max':>7} {'>=rec':>6}")
for mean, med, mn, mx, ge, n, b in rows[:15]:
    print(f"{b:<16} {mean:7.4f} {med:7.4f} {mn:7.4f} {mx:7.4f} {ge:3d}/{n}")
mean, med, mn, mx, ge, n, b = rows[0]
print(f"\nROBUST CHAMPION: {b}  mean={mean:.4f} median={med:.4f}  clears record({rec:.4f}) on {ge}/{n} seeds")
verdict = "SUBMIT-WORTHY" if mean >= rec else "BELOW-RECORD (hold)"
print(f"VERDICT: {verdict}  (mean {mean:.4f} vs record {rec:.4f})")
PY
