#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

ROBOT_FCI_IP="${ROBOT_FCI_IP:-172.16.0.2}"
TIME_TO_GO="${TIME_TO_GO:-8.0}"

args=(
  ./examples/fr3_realtime_libfranka/build/go_home
  --robot-ip "${ROBOT_FCI_IP}"
  --time-to-go "${TIME_TO_GO}"
)

if [[ "${NO_GRIPPER:-0}" == "1" ]]; then
  args+=(--no-gripper)
fi

exec conda run --no-capture-output -n fr3-realtime-libfranka "${args[@]}"
