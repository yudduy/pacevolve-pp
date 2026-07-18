#!/bin/bash
#SBATCH --job-name=skyrl-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
# (CPU-tier smoke at 32G OOM-killed during JAX forward_backward on a 0.6B model —
#  host RAM for JAX staging + driver + eval subprocesses needs headroom)
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/users/duynguy/logs/skyrl-smoke-%j.out
#
# L40S GPU smoke: server + driver in ONE allocation (localhost, lifecycle-coupled).
#   sbatch farmshare_skyrl_gpu_smoke.sbatch                     # Qwen3-0.6B quick GPU rung
#   sbatch farmshare_skyrl_gpu_smoke.sbatch Qwen/Qwen3-8B       # real-advisor-size rung
MODEL="${1:-Qwen/Qwen3-0.6B}"
exec bash /scratch/users/duynguy/pacevolve-pp/scripts/farmshare_skyrl_smoke.sh \
  --model "$MODEL" --gpu --n 2 --steps 1
