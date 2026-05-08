"""FR3 纯数据采集程序（不涉及 pi0.5 推理）。

架构：
    x86 RT PC ─[readonly_state_publisher.py, Polymetis 只读 state]─► 推理/采集 PC
    推理/采集 PC 上本程序：
      - RobotStateSubscriber  订阅 100Hz state
      - ZedStream × 2         两路 ZED UVC 单目 RGB 相机
      - DatasetRecorder       15Hz 落盘 LeRobot v2.1（字段对齐 DROID）
      - HotkeyListener        终端 <Enter>/s/r/q 控制 episode 边界

action 来源（不关心谁在驱动机器人）：
    从 joint_position 相邻两帧差分得到关节速度。只要手上的操作者是
    "专家"（物理示教按钮 / GELLO leader / 其他遥操作 …），差分出来的
    就是专家 action，可直接用于 pi05_droid fine-tune。

依赖：
    pip install opencv-python pyzmq msgpack numpy pyarrow
    （本程序不依赖 openpi_client）

用法示例：
    # 1) x86 RT PC 上使用 FrankaTeleop 创建的 Polymetis 环境，只读发布 state
    conda activate polymetis
    python readonly_state_publisher.py \\
        --polymetis-host 127.0.0.1 \\
        --polymetis-port 50051 \\
        --gripper-port 50053 \\
        --bind 0.0.0.0

    # 2) 进入 Franka 示教：按住末端 guiding 按钮拖动（或接 GELLO 遥操作）

    # 3) 本程序（采集 PC 上）
    python data_collector.py \\
        --robot-ip <RT-PC-IP> \\
        --front-camera /dev/video0 --wrist-camera /dev/video2 \\
        --prompt "pick up the red block and put it in the bowl" \\
        --record-dir ./datasets/fr3_pick_block \\
        --flip-wrist \\
        --display

    # 终端热键：
    #   <Enter> = 开始新的一集
    #   s       = 结束并保存
    #   r       = 丢弃当前集
    #   q       = 优雅退出
"""
from __future__ import annotations

import argparse
import logging
import threading
import time

import cv2

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


def _maybe_flip(img, flip: bool):
    return img[::-1, ::-1] if flip else img


def run(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    logging.info("subscribing robot state @ %s", args.robot_ip)
    sub = RobotStateSubscriber(args.robot_ip)

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

    # 等 state + 两路图像都 ready（最多 10s）
    deadline = time.time() + 10.0
    state_ok = front_ok = wrist_ok = False
    while time.time() < deadline:
        state_ok = sub.latest() is not None
        front_ok = cam_front.latest() is not None
        wrist_ok = cam_wrist.latest() is not None
        if state_ok and front_ok and wrist_ok:
            break
        time.sleep(0.1)
    if not (state_ok and front_ok and wrist_ok):
        missing = []
        if not state_ok:
            missing.append(f"state@{args.robot_ip}:5555")
        if not front_ok:
            missing.append(f"front_cam({args.front_camera})")
        if not wrist_ok:
            missing.append(f"wrist_cam({args.wrist_camera})")
        raise RuntimeError(
            "Timeout waiting for: " + ", ".join(missing)
            + "  （state 没来：检查 x86 RT PC 上 readonly_state_publisher.py 是否在 polymetis 环境里运行、"
            "--robot-ip 是否指向这台 x86 RT PC、防火墙/5555 端口是否放行；"
            "相机没来：检查 ZED USB3 连接、/dev/video* 编号、`v4l2-ctl --list-devices` 或 `ls /dev/video*`）"
        )

    quit_event = threading.Event()
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

    logging.info("recording -> %s  (fps=%d, img=%d)",
                 args.record_dir, args.record_fps, args.resize_size)
    logging.info(HOTKEY_HELP)

    try:
        while not quit_event.is_set():
            if args.display:
                front = cam_front.latest()
                wrist = cam_wrist.latest()
                if front is not None:
                    front = _maybe_flip(front, args.flip_front)
                    cv2.imshow("front", cv2.cvtColor(front, cv2.COLOR_RGB2BGR))
                if wrist is not None:
                    wrist = _maybe_flip(wrist, args.flip_wrist)
                    cv2.imshow("wrist", cv2.cvtColor(wrist, cv2.COLOR_RGB2BGR))
                # cv2 窗口焦点下按 q 也能退
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    quit_event.set()
            else:
                quit_event.wait(timeout=0.1)
    except KeyboardInterrupt:
        logging.info("Ctrl+C received, exiting")
    finally:
        # 先把正在录的一集落盘，避免丢数据
        try:
            recorder.stop_episode()
        except Exception as e:  # noqa: BLE001
            logging.warning("stop_episode failed: %s", e)
        recorder.stop()
        listener.stop()
        cam_front.close()
        cam_wrist.close()
        sub.close()
        if args.display:
            cv2.destroyAllWindows()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot-ip", required=True,
                    help="x86 RT PC 的 IP（readonly_state_publisher.py 所在机器）")
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
    ap.add_argument("--camera-width", type=int, default=2560,
                    help="ZED UVC 拼接帧宽度，HD720 通常为 2560")
    ap.add_argument("--camera-height", type=int, default=720,
                    help="ZED UVC 拼接帧高度，HD720 通常为 720")
    ap.add_argument("--camera-fps", type=int, default=30, help="ZED 采集帧率")
    ap.add_argument("--camera-fourcc", default="YUYV",
                    help="OpenCV 请求的 UVC fourcc；ZED 2i UVC 通常为 YUYV")
    ap.add_argument(
        "--camera-eye",
        default="LEFT",
        type=str.upper,
        choices=["LEFT", "RIGHT", "FULL"],
        help="从 ZED 左右拼接帧里取左目/右目；FULL 表示不切半",
    )
    ap.add_argument("--prompt", required=True,
                    help="任务描述文本，写入 meta/tasks.jsonl")
    ap.add_argument("--record-dir", required=True,
                    help="落盘目录（LeRobot v2.1，字段对齐 DROID）")
    ap.add_argument("--resize-size", type=int, default=224,
                    help="图像 resize + pad 到 (size, size)，对齐 pi05 输入")
    ap.add_argument("--flip-front", action="store_true", help="如果 front 视角需要 180 翻转")
    ap.add_argument("--flip-wrist", action="store_true", help="如果 wrist 视角需要 180 翻转")
    ap.add_argument("--record-fps", type=int, default=DROID_FPS,
                    help=f"采集频率，默认 {DROID_FPS}Hz（DROID 官方）")
    ap.add_argument("--display", action="store_true",
                    help="实时显示两路相机画面")
    args = ap.parse_args()
    if args.camera_mode is not None:
        args.camera_width, args.camera_height, args.camera_fps = ZED_UVC_MODES[args.camera_mode]
    run(args)


if __name__ == "__main__":
    main()
