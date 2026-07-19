#!/bin/bash
# Self-hosted skyrl-tx smoke: launch the Tinker-API server + run the advisor-RL driver
# against it on the SAME node. FarmShare layout (everything under /scratch/users/duynguy).
#
# Usage: farmshare_skyrl_smoke.sh [--model <hf-id>] [--gpu] [--n <samples>] [--steps <k>]
# Defaults: --model Qwen/Qwen3-0.6B, CPU, n=2, steps=1.
#
# CPU smoke (proves client<->server contract):   bash farmshare_skyrl_smoke.sh
# GPU smoke (via sbatch farmshare_skyrl_gpu_smoke.sbatch): adds --gpu [--model Qwen/Qwen3-8B]
set -euo pipefail

SCRATCH=/scratch/users/duynguy
PP="$SCRATCH/pacevolve-pp"
SKYRL_DIR="$SCRATCH/skyrl"
UV="$HOME/.local/bin/uv"
# Job-unique port: two jobs packed onto one 4-GPU node must not collide.
PORT=$((8000 + ${SLURM_JOB_ID:-0} % 1000))

MODEL="Qwen/Qwen3-0.6B"; GPU=0; NS=2; MS=1
while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODEL="$2"; shift 2;;
    --gpu)   GPU=1; shift;;
    --n)     NS="$2"; shift 2;;
    --steps) MS="$2"; shift 2;;
    *) echo "unknown arg: $1"; exit 2;;
  esac
done

# --- env: source real keys first, THEN override tinker to point at self-hosted ---
set -a
. "$SCRATCH/ttt-discover/.env" 2>/dev/null || true   # HF_TOKEN (TINKER_API_KEY overridden below)
. "$PP/.env" 2>/dev/null || true                     # OPENROUTER_*, WANDB_*
set +a
export TINKER_BASE_URL="http://127.0.0.1:$PORT"
export TINKER_API_KEY="tml-dummy"                    # self-hosted server accepts dummy auth
# skyrl-tx futures DB defaults to a SQLite file inside the repo = NFS scratch;
# SQLite on NFS throws intermittent "disk I/O error". Keep it node-local.
export SKYRL_DATABASE_URL="sqlite:////tmp/skyrl-tinker-${SLURM_JOB_ID:-manual}.db"
export HF_HOME="$SCRATCH/hf-cache"
export UV_CACHE_DIR="$SCRATCH/uv-cache"
export XDG_CACHE_HOME="$SCRATCH/.cache"
export PYTHONPATH="$SCRATCH/pp-extra:${PYTHONPATH:-}"

# Outbound network check (compute nodes need OpenRouter for the implementer)
curl -sm 10 https://openrouter.ai/api/v1/models -o /dev/null \
  || echo "WARN: no outbound network from this node — implementer calls will fail"

EXTRAS=(--extra tinker --extra jax)
if [ "$GPU" = 1 ]; then
  EXTRAS+=(--extra gpu)
  echo "=== GPU preflight ==="
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || { echo "no GPU visible"; exit 1; }
fi

RUNTAG="$(date -u +%Y%m%dT%H%M%SZ)"
SERVER_LOG="$SCRATCH/logs/skyrl-server-$RUNTAG.log"
PIDFILE="$SCRATCH/logs/skyrl-server-$RUNTAG.pid"

GPULOG="$SCRATCH/logs/gpu-util-$RUNTAG.log"
cleanup() {
  if [ -f "$PIDFILE" ]; then
    kill "$(cat "$PIDFILE")" 2>/dev/null || true
    rm -f "$PIDFILE"
  fi
  [ -n "${DMON_PID:-}" ] && kill "$DMON_PID" 2>/dev/null || true
}
trap cleanup EXIT

# GPU utilization evidence (sm%/mem% every 30s) — verify, don't assume.
if [ "$GPU" = 1 ]; then
  nohup nvidia-smi dmon -s um -d 30 -o T > "$GPULOG" 2>&1 &
  DMON_PID=$!
  echo "gpu utilization log: $GPULOG"
fi

echo "=== launch skyrl-tx server: $MODEL (gpu=$GPU) -> $SERVER_LOG ==="
cd "$SKYRL_DIR"
# SKYRL_BACKEND_CONFIG: optional JSON for --backend-config, e.g.
#   '{"tensor_parallel_size": 2}' or '{"sample_max_num_sequences": 2, "train_micro_batch_size": 1}'
# (8B in bf16 OOMs a single 48GB L40S with default JAX buffer sizing.)
nohup "$UV" run "${EXTRAS[@]}" -m skyrl.tinker.api \
  --base-model "$MODEL" --port "$PORT" \
  ${SKYRL_BACKEND_CONFIG:+--backend-config "$SKYRL_BACKEND_CONFIG"} > "$SERVER_LOG" 2>&1 &
echo $! > "$PIDFILE"

# Health poll: generous timeout — first JAX compile/weight load is slow (esp. 8B).
echo "=== waiting for server (up to 30 min) ==="
DEADLINE=$((SECONDS + 1800))
until curl -sf "http://127.0.0.1:$PORT/docs" > /dev/null 2>&1 \
   || curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; do
  if [ $SECONDS -ge $DEADLINE ]; then
    echo "server failed to come up; last log lines:"; tail -30 "$SERVER_LOG"; exit 1
  fi
  if ! kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "server process died; last log lines:"; tail -30 "$SERVER_LOG"; exit 1
  fi
  sleep 10
done
echo "server up after ${SECONDS}s"

# Materialize a PER-JOB config: config_1.yaml with rl.advisor_model = $MODEL and
# all working paths rewritten into a per-job workspace. Without this, concurrent
# Slurm jobs share src/solution.cpp + results over NFS and cross-contaminate.
RID="${SLURM_JOB_ID:-9}"
echo "=== materialize per-job config (run_id=$RID advisor_model=$MODEL) ==="
cd "$PP"
PYBIN=/scratch/users/duynguy/ttt-discover/.venv/bin/python
# CPU tier: short generations + long call timeout (JAX-CPU sampling is slow).
# GPU tier: long timeout too — 8B weight load + JIT can exceed the 600s default.
if [ "$GPU" = 1 ]; then SMOKE_MAX_TOKENS=""; SMOKE_TIMEOUT=1800; else SMOKE_MAX_TOKENS=256; SMOKE_TIMEOUT=1800; fi
MODEL="$MODEL" RID="$RID" SMOKE_MAX_TOKENS="$SMOKE_MAX_TOKENS" SMOKE_TIMEOUT="$SMOKE_TIMEOUT" "$PYBIN" - <<'EOF'
import os, shutil, yaml
rid = os.environ["RID"]
src = "tasks/rectangle_free_grid/config/config_1.yaml"
dst = f"tasks/rectangle_free_grid/config/config_{rid}.yaml"
with open(src) as f:
    cfg = yaml.safe_load(f)
cfg["rl"]["advisor_model"] = os.environ["MODEL"]
if os.environ.get("SMOKE_IMPLEMENTER"):
    cfg["implementer_llm"]["name"] = os.environ["SMOKE_IMPLEMENTER"]
if os.environ.get("SMOKE_MAX_TOKENS"):
    cfg["rl"]["advisor_max_tokens"] = int(os.environ["SMOKE_MAX_TOKENS"])
if os.environ.get("SMOKE_TIMEOUT"):
    cfg["rl"]["tinker_call_timeout"] = float(os.environ["SMOKE_TIMEOUT"])
# Per-job workspace: concurrent jobs must not share src/build/results over NFS.
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
with open(dst, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print("wrote", dst, "advisor_model=", cfg["rl"]["advisor_model"], "workspace=", ws)
EOF

echo "=== driver: run_advisor_rl --backend tinker (model=$MODEL n=$NS steps=$MS) ==="
RC=0
"$PYBIN" -u workflows/run_advisor_rl.py -t rectangle_free_grid -r "$RID" --backend tinker \
  --n_samples "$NS" --max_steps "$MS" || RC=$?
echo "=== smoke END rc=$RC (server log: $SERVER_LOG) ==="
exit $RC
