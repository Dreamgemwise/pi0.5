"""FR3 控制端：跑在 RT kernel 电脑上，基于 franky-control（libfranka 0.16）。

**DROID 版**（pi05_droid checkpoint）：

  1) 100 Hz 广播 RobotState（eef pose + gripper width + joint_position）
  2) 接收来自推理端的 ActionChunk (N, 8)
     - action[:, :7]: 关节速度（rad/s）
     - action[:,  7]: gripper_position（0=open, 1=close）
  3) 收到新 chunk 时：以当前实测关节角为起点，按 policy_dt 积分出 N 个目标关节角，
     用 franky.JointWaypointMotion（或降级为 JointMotion）异步下发；franky 内部以
     1kHz RT 做轨迹规划与限幅。后续 chunk 会 preempt 掉正在执行的运动，实现平滑接管。
  4) 夹爪状态机：仅在 open↔close 翻转时异步调用（阈值 0.5）

依赖：
  pip install franky-control pyzmq msgpack "numpy<2"
  另需：把老版本 libfranka 从 /usr/local/lib 移走，避免被 ld 先找到。
"""
from __future__ import annotations

import argparse
import logging
import threading
import time

import msgpack
import numpy as np
import zmq

# franky-control: bundle libfranka 0.16（兼容 FR3 FW 5.7+）
from franky import (
    Gripper,
    JointMotion,
    RelativeDynamicsFactor,
    Robot,
)

# 并非所有 franky 版本都有 JointWaypointMotion；有就用，没有降级到 JointMotion(终点)
try:
    from franky import JointWaypointMotion, JointWaypoint  # type: ignore
    _HAS_JOINT_WAYPOINT = True
except ImportError:
    _HAS_JOINT_WAYPOINT = False

from common import (
    ACTION_CHUNK_STALE_SEC,
    ACTION_PULL_PORT,
    GRIPPER_CLOSE_THRESHOLD,
    GRIPPER_GRASP_FORCE,
    GRIPPER_GRASP_SPEED,
    GRIPPER_GRASP_WIDTH,
    STATE_PUB_PORT,
    ActionChunk,
    RobotState,
    clamp_action_droid,
    clamp_joint_position,
    now,
)


# ------------------------------------------------------------------
# 夹爪状态机：仅在 open↔close 翻转时触发，避免慢 IO 堆积
# DROID 约定：gripper_position ∈ [0,1]，1 = close, 0 = open
# ------------------------------------------------------------------
class GripperFSM:
    def __init__(self, gripper: Gripper) -> None:
        self._gripper = gripper
        self._state: str = "open"
        self._pending: bool = False
        self._lock = threading.Lock()

    def update(self, gripper_cmd: float) -> None:
        want = "close" if gripper_cmd > GRIPPER_CLOSE_THRESHOLD else "open"
        with self._lock:
            if want == self._state or self._pending:
                return
            self._state = want
            self._pending = True
        threading.Thread(target=self._do, args=(want,), daemon=True).start()

    def _do(self, target: str) -> None:
        try:
            if target == "open":
                self._gripper.open(GRIPPER_GRASP_SPEED)
            else:
                self._gripper.grasp(
                    GRIPPER_GRASP_WIDTH,
                    GRIPPER_GRASP_SPEED,
                    GRIPPER_GRASP_FORCE,
                    epsilon_inner=0.08,
                    epsilon_outer=0.08,
                )
        except Exception as e:  # noqa: BLE001
            logging.warning("Gripper %s failed: %s", target, e)
        finally:
            with self._lock:
                self._pending = False


# ------------------------------------------------------------------
# 把 action chunk（关节速度）积分成关节空间 waypoint
# ------------------------------------------------------------------
def integrate_joint_velocity(
    actions: np.ndarray,       # (N, 8)
    q_start: np.ndarray,       # (7,)
    policy_dt: float,
):
    """返回 (N, 7) 的目标关节角序列，已限幅到 FR3 软限位。"""
    q = q_start.astype(np.float64).copy()
    qs: list[np.ndarray] = []
    for a in actions:
        a = clamp_action_droid(a)
        q = q + a[:7].astype(np.float64) * policy_dt
        q = clamp_joint_position(q)
        qs.append(q.copy())
    return np.stack(qs, axis=0)   # (N, 7)


def make_joint_motion(q_waypoints: np.ndarray):
    """如果 franky 有 JointWaypointMotion，用 N 个 waypoint；否则只送终点。"""
    if _HAS_JOINT_WAYPOINT:
        return JointWaypointMotion([JointWaypoint(q.tolist()) for q in q_waypoints])
    # fallback：只送最后一个终点，franky 做关节空间平滑插值
    return JointMotion(q_waypoints[-1].tolist())


# ------------------------------------------------------------------
# 主服务
# ------------------------------------------------------------------
class RobotServer:
    def __init__(self, hostname: str, bind_addr: str, dynamics_factor: float) -> None:
        logging.info("Connecting to Franka %s ...", hostname)
        self._robot = Robot(hostname)
        self._robot.recover_from_errors()
        self._robot.relative_dynamics_factor = RelativeDynamicsFactor(
            velocity=dynamics_factor,
            acceleration=dynamics_factor,
            jerk=dynamics_factor,
        )
        self._gripper = Gripper(hostname)
        self._gripper_fsm = GripperFSM(self._gripper)

        self._ctx = zmq.Context.instance()
        self._state_pub = self._ctx.socket(zmq.PUB)
        self._state_pub.bind(f"tcp://{bind_addr}:{STATE_PUB_PORT}")
        self._action_pull = self._ctx.socket(zmq.PULL)
        self._action_pull.bind(f"tcp://{bind_addr}:{ACTION_PULL_PORT}")
        logging.info("ZMQ bound: PUB state tcp://%s:%d, PULL action tcp://%s:%d",
                     bind_addr, STATE_PUB_PORT, bind_addr, ACTION_PULL_PORT)
        logging.info("JointWaypointMotion available: %s", _HAS_JOINT_WAYPOINT)

        self._stop = threading.Event()

    def _state_pub_loop(self) -> None:
        dt = 0.01  # 100 Hz
        while not self._stop.is_set():
            t0 = time.perf_counter()
            try:
                cs = self._robot.current_cartesian_state
                pose = cs.pose.end_effector_pose
                pos = np.asarray(pose.translation, dtype=np.float32)
                quat = np.asarray(pose.quaternion, dtype=np.float32)  # xyzw
                q = np.asarray(self._robot.current_joint_state.position, dtype=np.float32)
                width = float(self._gripper.width)
            except Exception as e:  # noqa: BLE001
                logging.debug("state read failed: %s", e)
                time.sleep(dt)
                continue
            state = RobotState(
                eef_pos=pos, eef_quat_xyzw=quat,
                gripper_width=width,
                joint_position=q,
                capture_ts=now(),
            )
            self._state_pub.send(msgpack.packb(state.to_msg(), use_bin_type=True))
            elapsed = time.perf_counter() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

    def _action_pull_loop(self) -> None:
        poller = zmq.Poller()
        poller.register(self._action_pull, zmq.POLLIN)
        while not self._stop.is_set():
            if not dict(poller.poll(timeout=50)):
                continue
            try:
                raw = self._action_pull.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                continue
            msg = msgpack.unpackb(raw, raw=False)
            chunk = ActionChunk.from_msg(msg)
            age = now() - chunk.capture_ts
            if age > ACTION_CHUNK_STALE_SEC:
                logging.warning("Drop stale chunk: age=%.3fs", age)
                continue

            if chunk.actions.shape[-1] != 8:
                logging.warning("Unexpected action dim %d (expected 8), drop.",
                                chunk.actions.shape[-1])
                continue

            # 以当前实测 joint 为起点
            try:
                q_now = np.asarray(
                    self._robot.current_joint_state.position, dtype=np.float64)
            except Exception as e:  # noqa: BLE001
                logging.warning("read joint state failed, skip chunk: %s", e)
                continue

            q_waypoints = integrate_joint_velocity(
                chunk.actions, q_now, chunk.policy_dt)

            # gripper：看 chunk 最后一个 action 的第 8 维
            self._gripper_fsm.update(float(clamp_action_droid(chunk.actions[-1])[7]))

            try:
                motion = make_joint_motion(q_waypoints)
                self._robot.move(motion, asynchronous=True)
                q_target = q_waypoints[-1]
                dq = q_target - q_now
                logging.info(
                    "chunk accepted: N=%d, age=%.1fms, |dq|=%.3frad, q_target=%s",
                    len(chunk.actions), age * 1000, float(np.linalg.norm(dq)),
                    np.array2string(q_target, precision=3),
                )
            except Exception as e:  # noqa: BLE001
                logging.warning("robot.move failed: %s", e)
                try:
                    self._robot.recover_from_errors()
                except Exception:
                    pass

    def run(self) -> None:
        threads = [
            threading.Thread(target=self._state_pub_loop, daemon=True, name="state_pub"),
            threading.Thread(target=self._action_pull_loop, daemon=True, name="action_pull"),
        ]
        for t in threads:
            t.start()
        try:
            while not self._stop.is_set():
                time.sleep(0.5)
        except KeyboardInterrupt:
            logging.info("Ctrl+C received, stopping.")
        finally:
            self._stop.set()
            try:
                self._robot.join_motion(timeout=1.0)
            except Exception:
                pass
            for t in threads:
                t.join(timeout=1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot-ip", required=True, help="FR3 FCI IP，例如 172.16.0.2")
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--dynamics-factor", type=float, default=0.05,
                    help="动力学上限比例（0~1），DROID 控制首次上机建议 ≤0.05")
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(threadName)s] %(message)s",
    )
    RobotServer(args.robot_ip, args.bind, args.dynamics_factor).run()


if __name__ == "__main__":
    main()
