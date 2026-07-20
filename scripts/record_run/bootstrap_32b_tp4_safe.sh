#!/bin/bash
# Qwen3-32B TP4 bootstrap on 4xA100-80. Carries every measured fix:
# - stacked.py sharded-zeros patch (load path OOMs materializing full stacks)
# - XLA fraction 0.85 exported (default pool too small for in-pool transients)
# - XLA command buffers disabled (CUDA-graph capture fails on host driver 555)
# - gradient_checkpointing (fwd_bwd activation stash), micro-batch 1
# - TP4: per-GPU resident weights ~33GB (train+sampler copies), ~35GB transient
#   headroom -> even a full unsharded 15.6GiB stack copy fits.
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
echo "=== prefetch Qwen3-32B ==="
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-32B')" 2>&1 | tail -1
echo "=== launch server (TP4) ==="
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export XLA_FLAGS="--xla_gpu_enable_command_buffer="
nohup uv run --extra tinker --extra jax --extra gpu -m skyrl.tinker.api \
  --base-model Qwen/Qwen3-32B --port 8021 \
  --checkpoints-base /workspace/ckpts \
  --database-url "sqlite:////tmp/skyrl-tinker.db" \
  --backend-config '{"tensor_parallel_size": 4, "train_micro_batch_size": 1, "sample_max_num_sequences": 1, "max_lora_adapters": 4, "gradient_checkpointing": true}' \
  > /workspace/server.log 2>&1 &
echo $! > /workspace/server.pid
echo "BOOTSTRAP-DONE pid=$(cat /workspace/server.pid)"
