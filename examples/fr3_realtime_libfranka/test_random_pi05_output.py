#!/usr/bin/env python3
"""Generate or send a small random pi0.5/DROID action chunk.

The C++ robot_server expects msgpack with:
  actions: float32 bytes, shape (N, 7) or (N, 8)
  actions_shape: [N, 7] or [N, 8]
  capture_ts: wall-clock timestamp
  policy_dt: seconds between policy steps

Action convention:
  actions[:, :7] = joint velocity in rad/s at 15 Hz
  actions[:, 7] = optional gripper command, 0=open, 1=close
"""

from __future__ import annotations

import argparse
import random
import struct
import time


ACTION_PULL_PORT = 5556
POLICY_HZ = 15.0
ACTION_HORIZON = 16
ACTION_DIM = 8


def make_actions(seed: int, joint_vel_scale: float, gripper: float | None) -> list[list[float]]:
    rng = random.Random(seed)
    actions: list[list[float]] = []
    for _ in range(ACTION_HORIZON):
        row = [rng.uniform(-joint_vel_scale, joint_vel_scale) for _ in range(7)]
        row.append(rng.uniform(0.0, 1.0) if gripper is None else float(gripper))
        actions.append(row)
    return actions


def pack_float32(actions: list[list[float]]) -> bytes:
    flat = [value for row in actions for value in row]
    return struct.pack(f"<{len(flat)}f", *flat)


def pack_chunk(actions: list[list[float]]) -> bytes:
    try:
        import msgpack
    except ImportError as exc:
        raise SystemExit("sending requires Python package: msgpack") from exc

    msg = {
        "actions": pack_float32(actions),
        "actions_shape": [ACTION_HORIZON, ACTION_DIM],
        "capture_ts": time.time(),
        "policy_dt": 1.0 / POLICY_HZ,
    }
    return msgpack.packb(msg, use_bin_type=True)


def make_and_pack_chunk(seed: int, joint_vel_scale: float, gripper: float | None) -> bytes:
    return pack_chunk(make_actions(seed, joint_vel_scale, gripper))


def print_actions(actions: list[list[float]]) -> None:
    print(f"actions shape=({ACTION_HORIZON}, {ACTION_DIM}), dtype=float32")
    print("[")
    for row in actions:
        values = ", ".join(f"{value: .6f}" for value in row)
        print(f"  [{values}],")
    print("]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-ip", default="127.0.0.1", help="robot_server host/IP")
    parser.add_argument("--port", type=int, default=ACTION_PULL_PORT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--joint-vel-scale",
        type=float,
        default=0.0,
        help="random joint velocity range in rad/s; default 0.0 is no arm motion",
    )
    parser.add_argument(
        "--gripper",
        type=float,
        default=None,
        help="fixed gripper command 0=open, 1=close; omitted means random",
    )
    parser.add_argument(
        "--send",
        action="store_true",
        help="actually send to tcp://<robot-ip>:<port>; otherwise only print",
    )
    parser.add_argument("--count", type=int, default=1, help="number of chunks to send")
    parser.add_argument("--interval", type=float, default=0.2, help="seconds between chunks")
    parser.add_argument(
        "--connect-delay",
        type=float,
        default=0.5,
        help="seconds to wait after ZMQ connect before first send",
    )
    args = parser.parse_args()

    if args.joint_vel_scale < 0.0:
        raise ValueError("--joint-vel-scale must be non-negative")
    if args.gripper is not None and not 0.0 <= args.gripper <= 1.0:
        raise ValueError("--gripper must be between 0 and 1")
    if args.count < 1:
        raise ValueError("--count must be at least 1")

    actions = make_actions(args.seed, args.joint_vel_scale, args.gripper)
    print_actions(actions)

    if not args.send:
        print("dry run only; add --send to push this chunk to robot_server")
        return

    try:
        import zmq
    except ImportError as exc:
        raise SystemExit("sending requires Python package: pyzmq") from exc

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUSH)
    sock.setsockopt(zmq.LINGER, 1000)
    endpoint = f"tcp://{args.robot_ip}:{args.port}"
    sock.connect(endpoint)
    time.sleep(args.connect_delay)

    for i in range(args.count):
        payload = pack_chunk(actions) if i == 0 else make_and_pack_chunk(
            args.seed + i, args.joint_vel_scale, args.gripper
        )
        sock.send(payload)
        print(f"sent {i + 1}/{args.count} to {endpoint}")
        if i + 1 < args.count:
            time.sleep(args.interval)

    sock.close()


if __name__ == "__main__":
    main()
