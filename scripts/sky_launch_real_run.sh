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
This launches a PAID cloud GPU run (on-demand, auto-terminates only via `sky down`).

  Estimated cost (128 steps x 4 samples; wall-clock dominated by implementer+eval):
    H100     $2-3/hr   x 10-21h  ->  ~$20-63
    A100-80  $1.3-2.2  x 10-21h  ->  ~$13-46
  Plus OpenRouter implementer spend (same as existing scaffold runs).

Re-run with:  bash scripts/sky_launch_real_run.sh --confirm
TABLE
  exit 1
fi

command -v sky >/dev/null || { echo "skypilot not installed: pip install 'skypilot[all]'"; exit 1; }
[ -f .env ] || { echo "missing .env (OPENROUTER_API_KEY etc.)"; exit 1; }

# Load secrets without echoing them.
set -a; . ./.env; set +a
: "${OPENROUTER_API_KEY:?not set in .env}"

exec sky launch -c rfg-tinker scripts/skypilot_rfg_tinker.yaml \
  --env OPENROUTER_API_KEY \
  --env WANDB_API_KEY \
  --env HF_TOKEN \
  --yes
