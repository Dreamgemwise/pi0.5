"""Read-only FR3 state publisher for data collection via Polymetis.

This script only reads Polymetis gRPC state and publishes the RobotState
message expected by data_collector.py on tcp://<bind>:5555.

It deliberately does not import polymetis.RobotInterface, start impedance
control, bind the action port, or send commands to the robot. The eef fields
are placeholders because the dataset recorder only consumes joint_position and
gripper_width.
"""
from __future__ import annotations

import argparse
import logging
import time

import grpc
import msgpack
import numpy as np
import zmq
from polymetis_pb2 import Empty
from polymetis_pb2_grpc import GripperServerStub, PolymetisControllerServerStub

from common import GRIPPER_MAX_WIDTH, STATE_PUB_PORT, RobotState, now


def wait_for_channel(channel: grpc.Channel, name: str, timeout: float) -> None:
    try:
        grpc.channel_ready_future(channel).result(timeout=timeout)
    except grpc.FutureTimeoutError as exc:
        raise RuntimeError(f"Timed out connecting to {name}") from exc


def run(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [readonly_state] %(message)s",
    )

    robot_addr = f"{args.polymetis_host}:{args.polymetis_port}"
    logging.info("Connecting to Polymetis robot state: %s", robot_addr)
    robot_channel = grpc.insecure_channel(robot_addr)
    wait_for_channel(robot_channel, robot_addr, args.connect_timeout)
    robot_stub = PolymetisControllerServerStub(robot_channel)

    gripper_stub = None
    gripper_width = float(args.default_gripper_width)
    if args.gripper_port > 0:
        gripper_addr = f"{args.polymetis_host}:{args.gripper_port}"
        logging.info("Connecting to Polymetis gripper state: %s", gripper_addr)
        gripper_channel = grpc.insecure_channel(gripper_addr)
        try:
            wait_for_channel(gripper_channel, gripper_addr, args.connect_timeout)
            gripper_stub = GripperServerStub(gripper_channel)
        except RuntimeError as exc:
            logging.warning("%s; using default gripper_width=%.3f", exc, gripper_width)

    ctx = zmq.Context.instance()
    pub = ctx.socket(zmq.PUB)
    pub.bind(f"tcp://{args.bind}:{STATE_PUB_PORT}")
    logging.info(
        "Publishing read-only RobotState on tcp://%s:%d at %.1f Hz",
        args.bind,
        STATE_PUB_PORT,
        args.hz,
    )
    logging.info("No action socket is opened; this process never sends robot commands.")

    empty = Empty()
    dt = 1.0 / args.hz
    eef_pos = np.zeros(3, dtype=np.float32)
    eef_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    while True:
        t0 = time.perf_counter()
        try:
            robot_state = robot_stub.GetRobotState(empty)
            joint_position = np.asarray(robot_state.joint_positions, dtype=np.float32)
            if joint_position.shape != (7,):
                logging.warning("Unexpected joint_position shape: %s", joint_position.shape)
                time.sleep(dt)
                continue

            if gripper_stub is not None:
                try:
                    gripper_width = float(gripper_stub.GetState(empty).width)
                except grpc.RpcError as exc:
                    logging.debug("gripper state read failed: %s", exc)

            state = RobotState(
                eef_pos=eef_pos,
                eef_quat_xyzw=eef_quat,
                gripper_width=float(np.clip(gripper_width, 0.0, GRIPPER_MAX_WIDTH)),
                joint_position=joint_position,
                capture_ts=now(),
            )
            pub.send(msgpack.packb(state.to_msg(), use_bin_type=True))
        except grpc.RpcError as exc:
            logging.warning("robot state read failed: %s", exc)
            time.sleep(min(1.0, dt))

        elapsed = time.perf_counter() - t0
        if elapsed < dt:
            time.sleep(dt - elapsed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--polymetis-host", default="127.0.0.1")
    ap.add_argument("--polymetis-port", type=int, default=50051)
    ap.add_argument("--gripper-port", type=int, default=50053, help="0 disables gripper reads")
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--hz", type=float, default=100.0)
    ap.add_argument("--connect-timeout", type=float, default=3.0)
    ap.add_argument(
        "--default-gripper-width",
        type=float,
        default=GRIPPER_MAX_WIDTH,
        help="Used when --gripper-port 0 or gripper server is unavailable",
    )
    run(ap.parse_args())


if __name__ == "__main__":
    main()
