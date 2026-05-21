#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

ROBOT_HOST="${ROBOT_HOST:-127.0.0.1}"
POLICY_HOST="${POLICY_HOST:-127.0.0.1}"
POLICY_PORT="${POLICY_PORT:-8010}"
PROMPT="${PROMPT:-pick up the cup}"
FRONT_CAMERA="${FRONT_CAMERA:-/dev/video0}"
WRIST_CAMERA="${WRIST_CAMERA:-/dev/video2}"
CAMERA_MODE="${CAMERA_MODE:-HD720}"
CAMERA_EYE="${CAMERA_EYE:-LEFT}"
CAMERA_FOURCC="${CAMERA_FOURCC:-YUYV}"
REQUEST_HZ="${REQUEST_HZ:-3.0}"
FRONT_TRANSFORM="${FRONT_TRANSFORM:-none}"
WRIST_TRANSFORM="${WRIST_TRANSFORM:-none}"

args=(
  python examples/fr3_realtime/inference_client.py
  --robot-ip "${ROBOT_HOST}"
  --policy-host "${POLICY_HOST}"
  --policy-port "${POLICY_PORT}"
  --prompt "${PROMPT}"
  --front-camera "${FRONT_CAMERA}"
  --wrist-camera "${WRIST_CAMERA}"
  --camera-mode "${CAMERA_MODE}"
  --camera-eye "${CAMERA_EYE}"
  --camera-fourcc "${CAMERA_FOURCC}"
  --request-hz "${REQUEST_HZ}"
  --front-transform "${FRONT_TRANSFORM}"
  --wrist-transform "${WRIST_TRANSFORM}"
  --display
)

if [[ "${FLIP_FRONT:-0}" == "1" ]]; then
  args+=(--flip-front)
fi

if [[ "${FLIP_WRIST:-0}" == "1" ]]; then
  args+=(--flip-wrist)
fi

if [[ -n "${RECORD_DIR:-}" ]]; then
  args+=(--record-dir "${RECORD_DIR}")
fi

exec conda run --no-capture-output -n openpi-dev "${args[@]}"
