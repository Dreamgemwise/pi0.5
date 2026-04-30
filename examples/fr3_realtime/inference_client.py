"""FR3 实时部署：推理端（跑在你这台有 RealSense + GPU 的机器）。

职责：
  1) ZMQ SUB 订阅 Robot PC 发来的最新 RobotState
  2) pyrealsense2 读两路图像（front 作为 agentview，wrist 作为 left_wrist）
  3) resize_with_pad -> 224×224 uint8，按 Libero 约定左右+上下翻转（如果相机视角需要）
  4) 通过 websocket 调 pi05 策略服务
  5) ZMQ PUSH 把 action chunk 发回 Robot PC（带 origin_pose 和 capture_ts）
  6) 可选：开启 --record-dir 后按 LeRobot v2.1 格式采集 pi05_droid fine-tune 数据
     （字段对齐 DROID，详见 dataset_recorder.py）

依赖：
  conda env create -f environment.yml
  conda activate openpi-dev

用法（仅推理）：
  python examples/fr3_realtime/inference_client.py \
    --robot-ip 192.168.1.10 \
    --policy-host 127.0.0.1 --policy-port 8010 \
    --prompt "pick up the red block and put it in the bowl" \
    --front-serial 000123456789 --wrist-serial 000987654321 \
    --display

用法（推理 + 数据采集）：
  同上，再加：
    --record-dir ./datasets/fr3_pick_block
  终端热键：<Enter>=开始一集  s=停止并保存  r=丢弃  q=优雅退出
"""
from __future__ import annotations

import argparse
import logging
import threading
import time
from dataclasses import dataclass

import cv2
import msgpack
import numpy as np
import zmq
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _wsclient

from common import ACTION_PULL_PORT
from common import POLICY_HZ
from common import STATE_PUB_PORT
from common import ActionChunk
from common import RobotState
from common import now
from dataset_recorder import (
    DROID_FPS,
    HOTKEY_HELP,
    DatasetRecorder,
    HotkeyListener,
)
from streams import RealSenseStream, RobotStateSubscriber


# -------------------- Action pusher --------------------
class ActionPusher:
    def __init__(self, robot_ip: str) -> None:
        ctx = zmq.Context.instance()
        self._sock = ctx.socket(zmq.PUSH)
        self._sock.connect(f"tcp://{robot_ip}:{ACTION_PULL_PORT}")
        self._sock.setsockopt(zmq.SNDHWM, 2)  # 队列小一点，避免堆积

    def send(self, chunk: ActionChunk) -> None:
        self._sock.send(msgpack.packb(chunk.to_msg(), use_bin_type=True))


# -------------------- Observation builder --------------------
@dataclass
class CaptureArgs:
    resize_size: int = 224
    flip_front: bool = False          # 如果 front 相机视角上下颠倒，设为 True
    flip_wrist: bool = False


def build_element(
    front_rgb: np.ndarray,
    wrist_rgb: np.ndarray,
    state: RobotState,
    prompt: str,
    cap: CaptureArgs,
) -> dict:
    def _prep(img: np.ndarray, flip: bool) -> np.ndarray:
        if flip:
            img = img[::-1, ::-1]
        img = image_tools.resize_with_pad(img, cap.resize_size, cap.resize_size)
        return image_tools.convert_to_uint8(img)

    return {
        "observation/exterior_image_1_left": _prep(front_rgb, cap.flip_front),
        "observation/wrist_image_left": _prep(wrist_rgb, cap.flip_wrist),
        "observation/joint_position": np.asarray(state.joint_position, dtype=np.float32),
        "observation/gripper_position": np.asarray([state.gripper_position_normalized], dtype=np.float32),
        "prompt": str(prompt),
    }


# -------------------- 主循环 --------------------
def run(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.info("connecting robot state @ %s:%d", args.robot_ip, STATE_PUB_PORT)
    sub = RobotStateSubscriber(args.robot_ip)
    pusher = ActionPusher(args.robot_ip)

    logging.info("opening realsense: front=%s, wrist=%s", args.front_serial, args.wrist_serial)
    cam_front = RealSenseStream(args.front_serial)
    cam_wrist = RealSenseStream(args.wrist_serial)

    logging.info("connecting policy server @ %s:%d", args.policy_host, args.policy_port)
    policy = _wsclient.WebsocketClientPolicy(args.policy_host, args.policy_port)

    cap = CaptureArgs(
        resize_size=args.resize_size,
        flip_front=args.flip_front,
        flip_wrist=args.flip_wrist,
    )

    # 等 state 和两路图都 ready
    for _ in range(100):
        if sub.latest() is not None and cam_front.latest() is not None and cam_wrist.latest() is not None:
            break
        time.sleep(0.1)
    else:
        raise RuntimeError("Timeout waiting for state / cameras")

    # 数据采集（opt-in）：开启 --record-dir 才启用
    recorder: DatasetRecorder | None = None
    listener: HotkeyListener | None = None
    quit_event = threading.Event()
    if args.record_dir:
        recorder = DatasetRecorder(
            root_dir=args.record_dir,
            task_prompt=args.prompt,
            img_size=args.resize_size,
            fps=args.record_fps,
        )
        recorder.attach(sub, cam_front, cam_wrist)
        recorder.start()
        listener = HotkeyListener(
            on_start=recorder.start_episode,
            on_stop=recorder.stop_episode,
            on_discard=recorder.discard_episode,
            on_quit=lambda: quit_event.set(),
        )
        listener.start()
        logging.info("recording -> %s", args.record_dir)
        logging.info(HOTKEY_HELP)

    period = 1.0 / args.request_hz
    try:
        while not quit_event.is_set():
            t0 = now()
            state = sub.latest()
            front = cam_front.latest()
            wrist = cam_wrist.latest()
            if state is None or front is None or wrist is None:
                time.sleep(period)
                continue

            element = build_element(front, wrist, state, args.prompt, cap)
            if args.display:
                cv2.imshow("front", cv2.cvtColor(element["observation/exterior_image_1_left"], cv2.COLOR_RGB2BGR))
                cv2.imshow("wrist", cv2.cvtColor(element["observation/wrist_image_left"], cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            capture_ts = state.capture_ts  # 使用 Robot PC 的时间戳
            t_req = now()
            out = policy.infer(element)
            t_inf = now() - t_req
            actions = np.asarray(out["actions"], dtype=np.float32)  # (N, 7)

            chunk = ActionChunk(
                actions=actions,
                origin_pos=state.eef_pos,
                origin_quat_xyzw=state.eef_quat_xyzw,
                capture_ts=capture_ts,
                policy_dt=1.0 / POLICY_HZ,
            )
            pusher.send(chunk)
            if recorder is not None:
                recorder.note_policy_chunk(actions, capture_ts)
            logging.info(
                "chunk sent: N=%d, infer=%.1fms, state_age=%.1fms",
                len(actions), t_inf * 1000, (now() - capture_ts) * 1000,
            )

            elapsed = now() - t0
            if elapsed < period:
                # 用 wait 而不是 sleep，这样 q 键能立刻唤醒循环退出
                quit_event.wait(timeout=period - elapsed)
    finally:
        # 先把正在录的一集存下来，避免丢数据
        if recorder is not None:
            try:
                recorder.stop_episode()
            except Exception as e:  # noqa: BLE001
                logging.warning("stop_episode failed: %s", e)
            recorder.stop()
        if listener is not None:
            listener.stop()
        cam_front.close()
        cam_wrist.close()
        sub.close()
        if args.display:
            cv2.destroyAllWindows()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot-ip", required=True, help="Robot PC 的 IP")
    ap.add_argument("--policy-host", default="127.0.0.1")
    ap.add_argument("--policy-port", type=int, default=8010)
    ap.add_argument("--prompt", required=True, help="任务描述文本")
    ap.add_argument("--front-serial", required=True, help="front RealSense 序列号")
    ap.add_argument("--wrist-serial", required=True, help="wrist RealSense 序列号")
    ap.add_argument("--resize-size", type=int, default=224)
    ap.add_argument("--flip-front", action="store_true", help="如果 front 视角需要 180 翻转")
    ap.add_argument("--flip-wrist", action="store_true")
    ap.add_argument("--request-hz", type=float, default=3.0,
                    help="向策略服务请求的频率，推理耗时 ~0.3s 时 3Hz 刚好无空档")
    ap.add_argument("--display", action="store_true")
    ap.add_argument("--record-dir", default=None,
                    help="开启数据采集：落盘到这个目录（LeRobot v2.1 格式，"
                         "字段对齐 DROID）。不设就不采集。")
    ap.add_argument("--record-fps", type=int, default=DROID_FPS,
                    help=f"采集频率，默认 {DROID_FPS}Hz（DROID 官方）")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
