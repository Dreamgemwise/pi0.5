#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-.70}"

POLICY_PORT="${POLICY_PORT:-8010}"
POLICY_CONFIG="${POLICY_CONFIG:-pi05_droid}"
POLICY_DIR="${POLICY_DIR:-models/pi05_droid}"

exec conda run --no-capture-output -n openpi-dev \
  python scripts/serve_policy.py \
  --port "${POLICY_PORT}" \
  policy:checkpoint \
  --policy.config="${POLICY_CONFIG}" \
  --policy.dir="${POLICY_DIR}"
