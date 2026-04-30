"""FR3 实时部署：两台机器共享的协议与工具函数。

消息都用 msgpack 打包，ZMQ 传输。三条数据流：
  1) 机器人状态  Robot -> Inference   (PUB/SUB tcp://<robot>:5555)
  2) 动作块      Inference -> Robot   (PUSH/PULL tcp://<robot>:5556)
  3) 图像观测 + 策略                  (已有 websocket_client_policy)
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

STATE_PUB_PORT = 5555
ACTION_PULL_PORT = 5556

CONTROL_HZ = 1000.0

# ------------- DROID / pi05_droid 约定 -------------
# pi05_droid: chunk_size=15, action_dim=8 (7 joint velocity + 1 gripper_position)
POLICY_HZ = 15.0
ACTION_HORIZON = 15
ACTION_DIM = 8
STATE_DIM = 8                # DROID: [joint_position(7), gripper_position(1)]

# 关节速度限幅（单个 policy step 的 vel_cmd 绝对值不超过这个）
MAX_JOINT_VEL = 1.5          # rad/s
ACTION_CHUNK_STALE_SEC = 0.5 # 超过这个延迟的 chunk 直接丢弃；DROID 15Hz，给的余量比 Libero 大

# DROID gripper 约定：0 = open, 1 = close
GRIPPER_CLOSE_THRESHOLD = 0.5  # action[7] > 0.5 => close，否则 open
GRIPPER_MAX_WIDTH = 0.08        # Franka Hand, 米
GRIPPER_GRASP_WIDTH = 0.0
GRIPPER_GRASP_SPEED = 0.1
GRIPPER_GRASP_FORCE = 20.0

# FR3 / Franka 关节限位（软限位，比硬限位内缩）
JOINT_LIMITS_LOW = np.array([-2.85, -1.76, -2.85, -3.04, -2.85, 0.40, -2.85], dtype=np.float64)
JOINT_LIMITS_HIGH = np.array([ 2.85,  1.76,  2.85, -0.15,  2.85, 3.65,  2.85], dtype=np.float64)


@dataclass
class RobotState:
    """Robot PC 发送给 Inference PC 的实时状态。

    既有笛卡尔（eef）也有关节空间（joint），同一条消息里都带上，
    下游用哪个由 policy format 决定。
    """

    eef_pos: np.ndarray          # (3,) base frame, meter
    eef_quat_xyzw: np.ndarray    # (4,) base frame
    gripper_width: float          # Franka Hand 宽度，米 (0 ~ 0.08)
    joint_position: np.ndarray   # (7,) rad
    capture_ts: float             # Unix epoch 秒（time.time()），跨机器可比较

    def to_msg(self) -> dict[str, Any]:
        return {
            "eef_pos": np.asarray(self.eef_pos, dtype=np.float32).tobytes(),
            "eef_quat_xyzw": np.asarray(self.eef_quat_xyzw, dtype=np.float32).tobytes(),
            "gripper_width": float(self.gripper_width),
            "joint_position": np.asarray(self.joint_position, dtype=np.float32).tobytes(),
            "capture_ts": float(self.capture_ts),
        }

    @classmethod
    def from_msg(cls, msg: dict[str, Any]) -> "RobotState":
        return cls(
            eef_pos=np.frombuffer(msg["eef_pos"], dtype=np.float32).copy(),
            eef_quat_xyzw=np.frombuffer(msg["eef_quat_xyzw"], dtype=np.float32).copy(),
            gripper_width=float(msg["gripper_width"]),
            joint_position=np.frombuffer(msg["joint_position"], dtype=np.float32).copy(),
            capture_ts=float(msg["capture_ts"]),
        )

    @property
    def gripper_position_normalized(self) -> float:
        """DROID 约定：0 = fully open, 1 = fully close。"""
        return float(np.clip(1.0 - self.gripper_width / GRIPPER_MAX_WIDTH, 0.0, 1.0))


@dataclass
class ActionChunk:
    """Inference PC 发送给 Robot PC 的动作块。"""

    actions: np.ndarray          # (N, 7)
    origin_pos: np.ndarray       # (3,) 捕获观测时的 eef_pos（delta 的起点参考）
    origin_quat_xyzw: np.ndarray # (4,) 同上
    capture_ts: float             # 观测捕获时间戳（Robot PC 自己的时钟域）
    policy_dt: float = 1.0 / POLICY_HZ  # 相邻 action 的时间间隔（秒）

    def to_msg(self) -> dict[str, Any]:
        return {
            "actions": np.asarray(self.actions, dtype=np.float32).tobytes(),
            "actions_shape": list(self.actions.shape),
            "origin_pos": np.asarray(self.origin_pos, dtype=np.float32).tobytes(),
            "origin_quat_xyzw": np.asarray(self.origin_quat_xyzw, dtype=np.float32).tobytes(),
            "capture_ts": float(self.capture_ts),
            "policy_dt": float(self.policy_dt),
        }

    @classmethod
    def from_msg(cls, msg: dict[str, Any]) -> "ActionChunk":
        shape = tuple(msg["actions_shape"])
        return cls(
            actions=np.frombuffer(msg["actions"], dtype=np.float32).reshape(shape).copy(),
            origin_pos=np.frombuffer(msg["origin_pos"], dtype=np.float32).copy(),
            origin_quat_xyzw=np.frombuffer(msg["origin_quat_xyzw"], dtype=np.float32).copy(),
            capture_ts=float(msg["capture_ts"]),
            policy_dt=float(msg["policy_dt"]),
        )


# ----------------- 四元数 / 轴角 工具 -----------------
# 约定：quat 都是 [x, y, z, w]（xyzw）

def quat_to_axisangle(quat_xyzw: np.ndarray) -> np.ndarray:
    """xyzw quat -> 轴角向量 (3,)，与 Libero 中 _quat2axisangle 等价。"""
    q = np.asarray(quat_xyzw, dtype=np.float64).copy()
    q[3] = max(-1.0, min(1.0, q[3]))
    den = math.sqrt(max(0.0, 1.0 - q[3] * q[3]))
    if den < 1e-8:
        return np.zeros(3, dtype=np.float32)
    return (q[:3] * 2.0 * math.acos(q[3]) / den).astype(np.float32)


def axisangle_to_quat(aa: np.ndarray) -> np.ndarray:
    """轴角向量 -> xyzw quat"""
    aa = np.asarray(aa, dtype=np.float64)
    angle = float(np.linalg.norm(aa))
    if angle < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    axis = aa / angle
    s = math.sin(angle / 2.0)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, math.cos(angle / 2.0)], dtype=np.float32)


def quat_mul(q1_xyzw: np.ndarray, q2_xyzw: np.ndarray) -> np.ndarray:
    """xyzw q1 * q2（Hamilton 乘法，对应旋转先施加 q2 再 q1）"""
    x1, y1, z1, w1 = q1_xyzw
    x2, y2, z2, w2 = q2_xyzw
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ], dtype=np.float32)


def quat_slerp(q0_xyzw: np.ndarray, q1_xyzw: np.ndarray, t: float) -> np.ndarray:
    q0 = np.asarray(q0_xyzw, dtype=np.float64)
    q1 = np.asarray(q1_xyzw, dtype=np.float64)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        out = q0 + t * (q1 - q0)
        return (out / np.linalg.norm(out)).astype(np.float32)
    theta_0 = math.acos(dot)
    theta = theta_0 * t
    sin_theta_0 = math.sin(theta_0)
    s0 = math.cos(theta) - dot * math.sin(theta) / sin_theta_0
    s1 = math.sin(theta) / sin_theta_0
    return (s0 * q0 + s1 * q1).astype(np.float32)


# ----------------- 限幅 -----------------
def clamp_action_droid(action: np.ndarray) -> np.ndarray:
    """DROID 8 维：前 7 维 joint velocity（rad/s），第 8 维 gripper_position（0~1）。"""
    out = np.asarray(action, dtype=np.float32).copy()
    out[:7] = np.clip(out[:7], -MAX_JOINT_VEL, MAX_JOINT_VEL)
    out[7] = float(np.clip(out[7], 0.0, 1.0))
    return out


def clamp_joint_position(q: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(q, dtype=np.float64), JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH)


def now() -> float:
    """Unix epoch 秒（跨机器可比较）。两台机器应通过 NTP 同步时钟。"""
    return time.time()


def mono() -> float:
    """单机单调时钟，仅用于同机内部计时（如测推理耗时）。"""
    return time.perf_counter()
