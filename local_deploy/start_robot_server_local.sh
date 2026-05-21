#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

ROBOT_FCI_IP="${ROBOT_FCI_IP:-172.16.0.2}"
BIND="${BIND:-0.0.0.0}"
DYNAMICS_FACTOR="${DYNAMICS_FACTOR:-0.05}"
EMA_ALPHA="${EMA_ALPHA:-0.35}"

args=(
  ./examples/fr3_realtime_libfranka/build/robot_server
  --robot-ip "${ROBOT_FCI_IP}"
  --bind "${BIND}"
  --dynamics-factor "${DYNAMICS_FACTOR}"
  --ema-alpha "${EMA_ALPHA}"
)

if [[ "${NO_GRIPPER:-0}" == "1" ]]; then
  args+=(--no-gripper)
fi

exec conda run --no-capture-output -n fr3-realtime-libfranka "${args[@]}"
