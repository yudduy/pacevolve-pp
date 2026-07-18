#!/bin/bash
# One-time setup of the self-hosted skyrl-tx Tinker-API server on FarmShare.
# Run ON THE LOGIN NODE (network access for clone + HF downloads):
#   bash /scratch/users/duynguy/pacevolve-pp/scripts/farmshare_skyrl_setup.sh
#
# Everything lives under /scratch/users/duynguy (HOME is over quota).
# Does NOT touch the ttt-discover venv or its site-packages.
set -euo pipefail

SCRATCH=/scratch/users/duynguy
SKYRL_DIR="$SCRATCH/skyrl"
# Pinned SkyRL commit (main @ 2026-07-18, verified: skyrl.tinker.api module, jax/tinker extras).
SKYRL_COMMIT=fdfcb435235ce6564e43af5e83bd85bb1f6af032
UV="$HOME/.local/bin/uv"

# Caches -> scratch (home quota protection)
export UV_CACHE_DIR="$SCRATCH/uv-cache"
export XDG_CACHE_HOME="$SCRATCH/.cache"
export HF_HOME="$SCRATCH/hf-cache"
mkdir -p "$UV_CACHE_DIR" "$XDG_CACHE_HOME" "$HF_HOME" "$SCRATCH/logs"

echo "=== [1/5] preflight: toolchain ==="
"$UV" --version
g++ --version | head -1

echo "=== [2/5] clone SkyRL @ pinned commit ==="
if [ ! -d "$SKYRL_DIR/.git" ]; then
  git clone https://github.com/NovaSky-AI/SkyRL "$SKYRL_DIR"
fi
git -C "$SKYRL_DIR" fetch --all --quiet
git -C "$SKYRL_DIR" checkout --quiet "$SKYRL_COMMIT"
echo "SkyRL at: $(git -C "$SKYRL_DIR" rev-parse --short HEAD)"

echo "=== [3/5] verify extras exist at pinned commit (repo reorg in flight) ==="
grep -nE '^(tinker|jax|gpu|cpu) *=' "$SKYRL_DIR/pyproject.toml" \
  || grep -n 'optional-dependencies' -A 20 "$SKYRL_DIR/pyproject.toml" | head -30 \
  || { echo "WARN: inspect $SKYRL_DIR/pyproject.toml extras manually"; }

echo "=== [4/5] uv sync (tinker + jax; add gpu extra only on GPU nodes at run time) ==="
cd "$SKYRL_DIR"
"$UV" venv --python 3.12 --allow-existing
"$UV" sync --extra tinker --extra jax

echo "=== [5/5] preflights: HF model pre-download + tinker SDK base_url support ==="
# Pre-download to scratch hf-cache so compute nodes need no network for weights.
"$UV" run python - <<'EOF'
from huggingface_hub import snapshot_download
for m in ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-8B"):
    print("prefetch:", m)
    snapshot_download(m)
EOF
/scratch/users/duynguy/ttt-discover/.venv/bin/python - <<'EOF'
import inspect, tinker
params = inspect.signature(tinker.ServiceClient.__init__).parameters
assert "base_url" in params or "kwargs" in params, f"tinker SDK lacks base_url: {list(params)}"
print("tinker SDK ok:", getattr(tinker, "__version__", "?"), list(params))
EOF

echo "=== setup complete ==="
