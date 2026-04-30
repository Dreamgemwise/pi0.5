"""把 FR3 用关节空间移动到一个"安全 home 位姿"。

**跑在 RT PC 上**（依赖 franky-control 直连 172.16.0.2）。

目的：把机器人从任何位形（包括奇异位形）通过关节插值移动到一个
      "远离奇异、末端朝下、肘关节 J4 弯曲 ~100°" 的构型。
      这样后续的 Cartesian 运动（test_move / inference_client）不会
      再触发 "cannot start at singular pose"。

用法（RT PC）：
  python go_home.py --robot-ip 172.16.0.2 --dynamics-factor 0.05

!!! 运行前手动确认末端周围 30cm 内无障碍，按好急停 !!!
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
from franky import JointMotion, RelativeDynamicsFactor, Robot


# FR3 一个"教科书式安全位姿"（单位：弧度，7 关节）
# 末端大致在基座正前方 40cm、上方 40cm，朝下，肘部明显弯曲
HOME_JOINTS = np.array([
    0.0,          # J1: 绕 Z 旋转，0 = 正前
    -0.785,       # J2: -45°（大臂后仰）
    0.0,          # J3
    -2.356,       # J4: -135° → 肘部弯曲，远离奇异
    0.0,          # J5
    1.571,        # J6: +90°
    0.785,        # J7: +45° (末端绕 Z)
], dtype=np.float64)


def run(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.info("Connecting to Franka %s ...", args.robot_ip)
    robot = Robot(args.robot_ip)
    robot.recover_from_errors()
    robot.relative_dynamics_factor = RelativeDynamicsFactor(
        velocity=args.dynamics_factor,
        acceleration=args.dynamics_factor,
        jerk=args.dynamics_factor,
    )

    q_now = robot.current_joint_state.position
    logging.info("Current joints (rad): %s", np.array2string(np.asarray(q_now), precision=3))
    logging.info("Target  joints (rad): %s", np.array2string(HOME_JOINTS, precision=3))

    logging.info("Moving to home (joint space) ... (阻塞直到到位)")
    robot.move(JointMotion(HOME_JOINTS))

    q_end = robot.current_joint_state.position
    logging.info("Arrived. joints (rad): %s", np.array2string(np.asarray(q_end), precision=3))

    pose = robot.current_pose.end_effector_pose
    logging.info("End-effector pos (m): %s",
                 np.array2string(np.asarray(pose.translation), precision=4))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot-ip", default="172.16.0.2")
    ap.add_argument("--dynamics-factor", type=float, default=0.05,
                    help="整体速度系数（0.05 = 慢慢挪，安全）")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
