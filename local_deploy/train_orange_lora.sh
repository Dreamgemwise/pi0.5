#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONPATH="$(pwd)/src:$(pwd)/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_LEROBOT_HOME="$(pwd)/datasets"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_HOME="$(pwd)/tmp/hf-home"
export HF_DATASETS_CACHE="$(pwd)/tmp/hf-datasets"
export XDG_CACHE_HOME="$(pwd)/tmp"
export JAX_COMPILATION_CACHE_DIR="$(pwd)/tmp/jax-cache"
export MPLCONFIGDIR="$(pwd)/tmp/matplotlib"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"

EXP_NAME="${EXP_NAME:-orange_lora_18eps_v1}"
STEPS="${STEPS:-3000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"

exec conda run --no-capture-output -n openpi-dev \
  python scripts/train.py pi05_fr3_lora \
  --exp-name "${EXP_NAME}" \
  --data.repo-id fr3_pick_up_orange \
  --num-train-steps "${STEPS}" \
  --save-interval "${SAVE_INTERVAL}" \
  --keep-period "${SAVE_INTERVAL}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --overwrite
