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
PORT=8000

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

cleanup() {
  if [ -f "$PIDFILE" ]; then
    kill "$(cat "$PIDFILE")" 2>/dev/null || true
    rm -f "$PIDFILE"
  fi
}
trap cleanup EXIT

echo "=== launch skyrl-tx server: $MODEL (gpu=$GPU) -> $SERVER_LOG ==="
cd "$SKYRL_DIR"
nohup "$UV" run "${EXTRAS[@]}" -m skyrl.tinker.api \
  --base-model "$MODEL" --port "$PORT" > "$SERVER_LOG" 2>&1 &
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

# Materialize smoke config (run_id 9): config_1.yaml with rl.advisor_model = $MODEL.
# Client model_name must match the server's --base-model; no repo code change needed.
echo "=== materialize smoke config (advisor_model=$MODEL) ==="
cd "$PP"
PYBIN=/scratch/users/duynguy/ttt-discover/.venv/bin/python
# CPU tier: short generations + long call timeout (JAX-CPU sampling is slow).
if [ "$GPU" = 1 ]; then SMOKE_MAX_TOKENS=""; SMOKE_TIMEOUT=""; else SMOKE_MAX_TOKENS=256; SMOKE_TIMEOUT=1800; fi
MODEL="$MODEL" SMOKE_MAX_TOKENS="$SMOKE_MAX_TOKENS" SMOKE_TIMEOUT="$SMOKE_TIMEOUT" "$PYBIN" - <<'EOF'
import os, yaml
src = "tasks/rectangle_free_grid/config/config_1.yaml"
dst = "tasks/rectangle_free_grid/config/config_9.yaml"
with open(src) as f:
    cfg = yaml.safe_load(f)
cfg["rl"]["advisor_model"] = os.environ["MODEL"]
if os.environ.get("SMOKE_MAX_TOKENS"):
    cfg["rl"]["advisor_max_tokens"] = int(os.environ["SMOKE_MAX_TOKENS"])
if os.environ.get("SMOKE_TIMEOUT"):
    cfg["rl"]["tinker_call_timeout"] = float(os.environ["SMOKE_TIMEOUT"])
with open(dst, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print("wrote", dst, "advisor_model=", cfg["rl"]["advisor_model"])
EOF

echo "=== driver: run_advisor_rl --backend tinker (model=$MODEL n=$NS steps=$MS) ==="
RC=0
"$PYBIN" -u workflows/run_advisor_rl.py -t rectangle_free_grid -r 9 --backend tinker \
  --n_samples "$NS" --max_steps "$MS" || RC=$?
echo "=== smoke END rc=$RC (server log: $SERVER_LOG) ==="
exit $RC
