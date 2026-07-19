#!/bin/bash
# Local (macOS) driver: run the advisor-RL loop on this machine against an
# EXTERNAL skyrl-tx server (RunPod pod). Same shape as farmshare_e2_driver.sh
# but with the repo venv, the sibling ttt-discover checkout on PYTHONPATH, and
# no Slurm. Implementer (OpenRouter) + compile/eval run locally for free.
#
# Usage: local_driver.sh --url http://<pod-ip>:<port> --rid <int>
#        [--model Qwen/Qwen3-8B] [--n 4] [--steps 128]
set -euo pipefail

PP="$(cd "$(dirname "$0")/.." && pwd)"
TTT="$PP/../ttt-discover"
PYBIN="$PP/.venv/bin/python"

URL=""; RID=""; MODEL="Qwen/Qwen3-8B"; NS=4; MS=128
while [ $# -gt 0 ]; do
  case "$1" in
    --url)   URL="$2"; shift 2;;
    --rid)   RID="$2"; shift 2;;
    --model) MODEL="$2"; shift 2;;
    --n)     NS="$2"; shift 2;;
    --steps) MS="$2"; shift 2;;
    *) echo "unknown arg: $1"; exit 2;;
  esac
done
[ -n "$URL" ] || { echo "--url required (http://pod-ip:port)"; exit 2; }
[ -n "$RID" ] || { echo "--rid required (unique int, names the job_<rid> workspace)"; exit 2; }

set -a
. "$PP/.env"
set +a
export TINKER_BASE_URL="$URL"
export TINKER_API_KEY="tml-dummy"
export PYTHONPATH="$TTT:${PYTHONPATH:-}"

echo "=== external server health: $URL ==="
DEADLINE=$((SECONDS + 900))
until curl -sf "$URL/docs" >/dev/null 2>&1 || curl -sf "$URL/health" >/dev/null 2>&1; do
  [ $SECONDS -ge $DEADLINE ] && { echo "server unreachable"; exit 1; }
  sleep 15
done
echo "server reachable after ${SECONDS}s"

cd "$PP"
MODEL="$MODEL" RID="$RID" "$PYBIN" - <<'EOF'
import os, shutil, yaml
rid = os.environ["RID"]
with open("tasks/rectangle_free_grid/config/config_1.yaml") as f:
    cfg = yaml.safe_load(f)
cfg["rl"]["advisor_model"] = os.environ["MODEL"]
cfg["rl"]["tinker_call_timeout"] = 1800.0
ws = f"tasks/rectangle_free_grid/results/job_{rid}"
paths = cfg.get("paths") or {}
canon_src = paths.get("src_path", "tasks/rectangle_free_grid/src")
job_src = f"{ws}/src"
os.makedirs(ws, exist_ok=True)
if not os.path.isdir(job_src):
    shutil.copytree(canon_src, job_src)
paths["src_path"] = job_src
for key, sub in (("build_dir", "build"), ("results_path", "results"),
                 ("log_dir", "logs"), ("transcript_dir", "transcripts")):
    if key in paths:
        paths[key] = f"{ws}/{sub}"
cfg["paths"] = paths
dst = f"tasks/rectangle_free_grid/config/config_{rid}.yaml"
with open(dst, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print("wrote", dst, "advisor_model=", cfg["rl"]["advisor_model"], "workspace=", ws)
EOF

echo "=== local driver: model=$MODEL n=$NS steps=$MS -> $URL ==="
RC=0
"$PYBIN" -u workflows/run_advisor_rl.py -t rectangle_free_grid -r "$RID" --backend tinker \
  --n_samples "$NS" --max_steps "$MS" || RC=$?
echo "=== local driver END rc=$RC ==="
exit $RC
