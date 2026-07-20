#!/bin/bash
# Qwen3-8B single-A100-80 bootstrap. Carries every measured fix from E2:
# - XLA fraction 0.85 (batch-4 ~15k-prompt prefill buffer needs the bigger pool)
# - XLA command buffers disabled (CUDA-graph capture fails on host driver 555)
# - gradient_checkpointing (29GiB activation stash at ~15k seq otherwise)
# - stacked.py sharded-zeros patch (harmless at 8B, guards the same load path)
set -euo pipefail
export HF_HOME=/workspace/hf UV_CACHE_DIR=/workspace/uv XDG_CACHE_HOME=/workspace/.cache
mkdir -p "$HF_HOME" "$UV_CACHE_DIR" /workspace/ckpts
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
echo "=== clone skyrl ==="
[ -d /workspace/skyrl/.git ] || git clone https://github.com/NovaSky-AI/SkyRL /workspace/skyrl
git -C /workspace/skyrl checkout fdfcb435235ce6564e43af5e83bd85bb1f6af032
echo "=== patch stacked.py (sharded zeros) ==="
python3 - <<'PY'
path = "/workspace/skyrl/skyrl/tx/layers/stacked.py"
src = open(path).read()
old = """            if hasattr(original_sharding, "spec"):
                new_spec = PartitionSpec(None, *original_sharding.spec)
                stacked = jax.device_put(jnp.zeros(stacked_shape, arr.dtype), NamedSharding(mesh, new_spec))"""
new = """            if hasattr(original_sharding, "spec"):
                new_spec = PartitionSpec(None, *original_sharding.spec)
                # Allocate directly into the target sharding: jnp.zeros +
                # device_put first materializes the FULL stacked group on one
                # device, which OOMs during load for 32B-class stacks.
                _sharding = NamedSharding(mesh, new_spec)
                stacked = jax.jit(
                    lambda shape=stacked_shape, dtype=arr.dtype: jnp.zeros(shape, dtype),
                    out_shardings=_sharding,
                )()"""
if new in src:
    print("already patched")
elif old in src:
    open(path, "w").write(src.replace(old, new, 1))
    print("patched stacked.py")
else:
    raise SystemExit("PATTERN NOT FOUND — stacked.py drifted")
PY
cd /workspace/skyrl
echo "=== uv sync ==="
uv venv --python 3.12 --allow-existing
uv sync --extra tinker --extra jax --extra gpu
echo "=== prefetch Qwen3-8B ==="
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-8B')" 2>&1 | tail -1
echo "=== launch server (8B) ==="
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export XLA_FLAGS="--xla_gpu_enable_command_buffer="
nohup uv run --extra tinker --extra jax --extra gpu -m skyrl.tinker.api \
  --base-model Qwen/Qwen3-8B --port 8021 \
  --checkpoints-base /workspace/ckpts \
  --database-url "sqlite:////tmp/skyrl-tinker.db" \
  --backend-config '{"train_micro_batch_size": 1, "sample_max_num_sequences": 4, "max_lora_adapters": 4, "gradient_checkpointing": true}' \
  > /workspace/server.log 2>&1 &
echo $! > /workspace/server.pid
echo "BOOTSTRAP-DONE pid=$(cat /workspace/server.pid)"
