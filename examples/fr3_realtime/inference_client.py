opened"""FR3 实时部署：推理端（跑在你这台有 ZED + GPU 的机器）。

职责：
  1) ZMQ SUB 订阅 Robot PC 发来的最新 RobotState
  2) OpenCV/UVC 读两路 ZED 单目 RGB 图像（front 作为 agentview，wrist 作为 left_wrist）
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
    --front-camera /dev/video0 --wrist-camera /dev/video2 \
    --flip-wrist \
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
from streams import ZedStream, RobotStateSubscriber


ZED_UVC_MODES = {
    "SVGA": (1344, 376, 60),
    "HD720": (2560, 720, 30),
    "HD1080": (3840, 1080, 30),
    "HD2K": (4416, 1242, 15),
}

JOINT7_MODEL_INPUT_OFFSET_RAD = np.float32(np.pi / 4.0)

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
    front_transform: str = "none"
    wrist_transform: str = "none"


def build_element(
    front_rgb: np.ndarray,
    wrist_rgb: np.ndarray,
    state: RobotState,
    prompt: str,
    cap: CaptureArgs,
) -> dict:
    def _apply_transform(img: np.ndarray, transform: str) -> np.ndarray:
        if transform == "none":
            return img
        if transform == "flip180":
            img = img[::-1, ::-1]
        elif transform == "hflip":
            img = img[:, ::-1]
        elif transform == "vflip":
            img = img[::-1, :]
        else:
            raise ValueError(f"unsupported image transform: {transform}")
        return img

    def _prep(img: np.ndarray, flip: bool, transform: str) -> np.ndarray:
        if flip and transform == "none":
            transform = "flip180"
        img = _apply_transform(img, transform)
        img = image_tools.resize_with_pad(img, cap.resize_size, cap.resize_size)
        return image_tools.convert_to_uint8(img)

    joint_position = np.asarray(state.joint_position, dtype=np.float32).copy()
    joint_position[6] -= JOINT7_MODEL_INPUT_OFFSET_RAD

    return {
        "observation/exterior_image_1_left": _prep(front_rgb, cap.flip_front, cap.front_transform),
        "observation/wrist_image_left": _prep(wrist_rgb, cap.flip_wrist, cap.wrist_transform),
        "observation/joint_position": joint_position,
        "observation/gripper_position": np.asarray([state.gripper_position_normalized], dtype=np.float32),
        "prompt": str(prompt),
    }


# -------------------- 主循环 --------------------
def run(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.info("connecting robot state @ %s:%d", args.robot_ip, STATE_PUB_PORT)
    sub = RobotStateSubscriber(args.robot_ip)
    pusher = ActionPusher(args.robot_ip)

    logging.info(
        "opening zed uvc: front=%s, wrist=%s, size=%dx%d, fps=%d, eye=%s",
        args.front_camera,
        args.wrist_camera,
        args.camera_width,
        args.camera_height,
        args.camera_fps,
        args.camera_eye,
    )
    cam_front = ZedStream(
        args.front_camera,
        width=args.camera_width,
        height=args.camera_height,
        fps=args.camera_fps,
        eye=args.camera_eye,
        fourcc=args.camera_fourcc,
    )
    cam_wrist = ZedStream(
        args.wrist_camera,
        width=args.camera_width,
        height=args.camera_height,
        fps=args.camera_fps,
        eye=args.camera_eye,
        fourcc=args.camera_fourcc,
    )

    logging.info("connecting policy server @ %s:%d", args.policy_host, args.policy_port)
    policy = _wsclient.WebsocketClientPolicy(args.policy_host, args.policy_port)

    cap = CaptureArgs(
        resize_size=args.resize_size,
        flip_front=args.flip_front,
        flip_wrist=args.flip_wrist,
        front_transform=args.front_transform,
        wrist_transform=args.wrist_transform,
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
            flip_front=args.flip_front,
            flip_wrist=args.flip_wrist,
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
    ap.add_argument(
        "--front-camera",
        required=True,
        help="front ZED UVC 源，比如 0 或 /dev/video0（exterior_image_1_left）",
    )
    ap.add_argument(
        "--wrist-camera",
        required=True,
        help="wrist ZED UVC 源，比如 2 或 /dev/video2（wrist_image_left）",
    )
    ap.add_argument(
        "--camera-mode",
        choices=sorted(ZED_UVC_MODES),
        default=None,
        help="ZED UVC 预设模式；HD2K 为 4416x1242@15fps",
    )
    ap.add_argument("--camera-width", type=int, default=2560, help="ZED UVC 拼接帧宽度，HD720 通常为 2560")
    ap.add_argument("--camera-height", type=int, default=720, help="ZED UVC 拼接帧高度，HD720 通常为 720")
    ap.add_argument("--camera-fps", type=int, default=30, help="ZED 采集帧率")
    ap.add_argument("--camera-fourcc", default="YUYV", help="OpenCV 请求的 UVC fourcc；ZED 2i UVC 通常为 YUYV")
    ap.add_argument(
        "--camera-eye",
        default="LEFT",
        type=str.upper,
        choices=["LEFT", "RIGHT", "FULL"],
        help="从 ZED 左右拼接帧里取左目/右目；FULL 表示不切半",
    )
    ap.add_argument("--resize-size", type=int, default=224)
    ap.add_argument("--flip-front", action="store_true", help="如果 front 视角需要 180 翻转")
    ap.add_argument("--flip-wrist", action="store_true")
    ap.add_argument("--front-transform", choices=("none", "flip180", "hflip", "vflip"), default="none")
    ap.add_argument("--wrist-transform", choices=("none", "flip180", "hflip", "vflip"), default="none")
    ap.add_argument("--request-hz", type=float, default=3.0,
                    help="向策略服务请求的频率，推理耗时 ~0.3s 时 3Hz 刚好无空档")
    ap.add_argument("--display", action="store_true")
    ap.add_argument("--record-dir", default=None,
                    help="开启数据采集：落盘到这个目录（LeRobot v2.1 格式，"
                         "字段对齐 DROID）。不设就不采集。")
    ap.add_argument("--record-fps", type=int, default=DROID_FPS,
                    help=f"采集频率，默认 {DROID_FPS}Hz（DROID 官方）")
    args = ap.parse_args()
    if args.camera_mode is not None:
        args.camera_width, args.camera_height, args.camera_fps = ZED_UVC_MODES[args.camera_mode]
    run(args)


if __name__ == "__main__":
    main()
