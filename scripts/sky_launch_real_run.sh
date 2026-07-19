#!/bin/bash
# USER-GATED launcher for the rented-GPU real run (spends real money).
# Reads secrets from the local gitignored .env and passes them to sky launch
# as launch-time env values (never committed, never printed).
#
#   bash scripts/sky_launch_real_run.sh --confirm
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "${1:-}" != "--confirm" ]; then
  cat <<'TABLE'
This launches a PAID cloud GPU run. Cost guards: auto-teardown when the run
finishes (--down) and after 60 idle minutes (-i 60). Manual kill: sky down rfg-tinker

  Estimated cost (128 steps x 4 samples; wall-clock dominated by implementer+eval):
    A40/A6000 48GB  $0.44-0.49/hr x 10-21h  ->  ~$5-10
    A100-80GB       $1.39/hr      x 10-21h  ->  ~$14-29
  Plus OpenRouter implementer spend (~$30-50 at RUN 1 per-call rates).

Re-run with:  bash scripts/sky_launch_real_run.sh --confirm
TABLE
  exit 1
fi

SKY="$(dirname "$0")/../.venv/bin/sky"
[ -x "$SKY" ] || SKY=sky
command -v "$SKY" >/dev/null || { echo "skypilot not installed: .venv/bin/python -m pip install 'skypilot[runpod]'"; exit 1; }
[ -f .env ] || { echo "missing .env (OPENROUTER_API_KEY etc.)"; exit 1; }

# Load secrets without echoing them.
set -a; . ./.env; set +a
: "${OPENROUTER_API_KEY:?not set in .env}"

# --down: tear down when the run job finishes; -i 60: teardown after 60 idle min.
# Per-second billing on RunPod means teardown == spend stops.
exec "$SKY" launch -c rfg-tinker scripts/skypilot_rfg_tinker.yaml \
  --env OPENROUTER_API_KEY \
  --env WANDB_API_KEY \
  --env HF_TOKEN \
  --idle-minutes-to-autostop 60 --down \
  --yes
