#!/bin/bash
# Pod-side autonomous chain: driver venv -> wait for server -> run advisor-RL
# -> upload results to W&B -> self-remove pod. Designed to survive the Mac
# (driver + monitoring) going offline entirely: everything after launch happens
# on the pod, and every outcome path ends with results in W&B and the pod gone.
# Usage: pod_chain.sh <RID> <HF_MODEL> <N_SAMPLES> <MAX_STEPS>
set -u
RID="${1:?rid}"; HF="${2:?hf model}"; NS="${3:-4}"; MS="${4:-128}"
PP=/workspace/pp
export HF_HOME=/workspace/hf
export PYTHONPATH="/workspace/ttt:${PYTHONPATH:-}"

if command -v runpodctl >/dev/null 2>&1 && [ -n "${RUNPOD_POD_ID:-}" ]; then
  echo "[chain] self-remove available pod=$RUNPOD_POD_ID"
else
  echo "[chain] WARNING self-remove unavailable (terminate-after backstop only)"
fi

echo "=== [chain] driver venv ==="
command -v g++ >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq g++; }
python3 -m venv --system-site-packages /workspace/pvenv
PIP=/workspace/pvenv/bin/pip
PY=/workspace/pvenv/bin/python
$PIP -q install --upgrade pip 2>&1 | tail -1
$PIP -q install tinker chz termcolor ray numpy pyyaml wandb openai transformers requests || { echo "CHAIN-FAIL pip"; exit 1; }
$PY - <<'EOF' || { echo "CHAIN-FAIL imports"; exit 1; }
import tinker, transformers, wandb, openai, numpy, yaml
from ttt_discover.rl import data_processing
from ttt_discover.rl.types import Trajectory, Transition
from ttt_discover.tinker_utils.completers import TokensWithLogprobs
print("[chain] imports OK")
EOF

set -a; . "$PP/.env"; set +a
export TINKER_BASE_URL="http://127.0.0.1:8021"
export TINKER_API_KEY="tml-dummy"

echo "=== [chain] wait for server ==="
DEADLINE=$((SECONDS + 3600))
until curl -sf "$TINKER_BASE_URL/docs" >/dev/null 2>&1; do
  [ $SECONDS -ge $DEADLINE ] && { echo "CHAIN-FAIL server-timeout"; exit 1; }
  sleep 20
done
sleep 180  # engine can crash 1-2 min after /docs turns 200; require a stable window
CR=$(grep -c "Background engine crashed" /workspace/server.log 2>/dev/null | tr -dc 0-9)
[ -z "$CR" ] && CR=0
if [ "$CR" != "0" ]; then echo "CHAIN-FAIL engine-crashed"; tail -40 /workspace/server.log; exit 1; fi
curl -sf "$TINKER_BASE_URL/docs" >/dev/null 2>&1 || { echo "CHAIN-FAIL server-died-in-window"; exit 1; }
echo "[chain] server healthy-stable"

cd "$PP"
echo "=== [chain] materialize config_${RID}.yaml ==="
MODEL="$HF" RID="$RID" $PY - <<'EOF' || { echo "CHAIN-FAIL config"; exit 1; }
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

tar_and_upload() {
  local TAG="$1"
  local ST="/workspace/stage_${TAG}"
  local TARP="/workspace/rfg_${RID}_${TAG}.tgz"
  rm -rf "$ST"; mkdir -p "$ST"
  cp -r "$PP/tasks/rectangle_free_grid/results/job_${RID}" "$ST/" 2>/dev/null
  cp "$PP/tasks/rectangle_free_grid/config/config_${RID}.yaml" "$ST/" 2>/dev/null
  tail -c 2000000 /workspace/server.log > "$ST/server_tail.log" 2>/dev/null
  tail -c 2000000 /workspace/chain.log  > "$ST/chain_tail.log"  2>/dev/null
  tar -czf "$TARP" -C "$ST" . 2>/dev/null
  RID="$RID" TAG="$TAG" TARPATH="$TARP" $PY /workspace/upload_artifact.py \
    && echo "[chain] artifact uploaded tag=$TAG" \
    || echo "[chain] WARNING artifact upload failed tag=$TAG"
}

( while true; do sleep 7200; tar_and_upload "partial-$(cat /workspace/upcount 2>/dev/null || echo 0)"; \
    echo $(( $(cat /workspace/upcount 2>/dev/null || echo 0) + 1 )) > /workspace/upcount; done ) &
UPLOOP=$!

echo "=== [chain] DRIVER START rid=$RID model=$HF n=$NS steps=$MS ==="
RC=0
$PY -u workflows/run_advisor_rl.py -t rectangle_free_grid -r "$RID" --backend tinker \
  --n_samples "$NS" --max_steps "$MS" || RC=$?
echo "=== [chain] DRIVER END rc=$RC ==="

kill "$UPLOOP" 2>/dev/null
tar_and_upload final
echo "[chain] final artifact done; self-removing pod"
if command -v runpodctl >/dev/null 2>&1 && [ -n "${RUNPOD_POD_ID:-}" ]; then
  runpodctl remove pod "$RUNPOD_POD_ID" && echo "[chain] SELF-REMOVED"
fi
echo "[chain] CHAIN COMPLETE rc=$RC"
exit $RC
