"""FR3 纯数据采集程序（不涉及 pi0.5 推理）。

架构：
    RT PC ─[robot_server.py, 只用 state PUB]─► 推理/采集 PC
    推理/采集 PC 上本程序：
      - RobotStateSubscriber  订阅 100Hz state
      - RealSenseStream × 2   两路相机
      - DatasetRecorder       15Hz 落盘 LeRobot v2.1（字段对齐 DROID）
      - HotkeyListener        终端 <Enter>/s/r/q 控制 episode 边界

action 来源（不关心谁在驱动机器人）：
    从 joint_position 相邻两帧差分得到关节速度。只要手上的操作者是
    "专家"（物理示教按钮 / GELLO leader / 其他遥操作 …），差分出来的
    就是专家 action，可直接用于 pi05_droid fine-tune。

依赖：
    pip install pyrealsense2 pyzmq msgpack numpy opencv-python pyarrow
    （本程序不依赖 openpi_client）

用法示例：
    # 1) RT PC 上照常起 robot_server（它只需要发 state；execution mode 即可）
    python robot_server.py --robot-ip 172.16.0.2

    # 2) 进入 Franka 示教：按住末端 guiding 按钮拖动（或接 GELLO 遥操作）

    # 3) 本程序（采集 PC 上）
    python data_collector.py \\
        --robot-ip <RT-PC-IP> \\
        --front-serial 000123456789 --wrist-serial 000987654321 \\
        --prompt "pick up the red block and put it in the bowl" \\
        --record-dir ./datasets/fr3_pick_block \\
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
from streams import RealSenseStream, RobotStateSubscriber


def run(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    logging.info("subscribing robot state @ %s", args.robot_ip)
    sub = RobotStateSubscriber(args.robot_ip)

    logging.info(
        "opening realsense: front=%s, wrist=%s",
        args.front_serial, args.wrist_serial,
    )
    cam_front = RealSenseStream(args.front_serial)
    cam_wrist = RealSenseStream(args.wrist_serial)

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
            missing.append(f"front_cam({args.front_serial})")
        if not wrist_ok:
            missing.append(f"wrist_cam({args.wrist_serial})")
        raise RuntimeError(
            "Timeout waiting for: " + ", ".join(missing)
            + "  （state 没来：检查 RT PC 上 robot_server.py 是否在跑、--robot-ip 是否指向 RT PC、防火墙/5555 端口是否放行；"
            "相机没来：检查 RealSense 序列号、USB3 口、`rs-enumerate-devices` 能否列出）"
        )

    quit_event = threading.Event()
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

    logging.info("recording -> %s  (fps=%d, img=%d)",
                 args.record_dir, args.record_fps, args.resize_size)
    logging.info(HOTKEY_HELP)

    try:
        while not quit_event.is_set():
            if args.display:
                front = cam_front.latest()
                wrist = cam_wrist.latest()
                if front is not None:
                    cv2.imshow("front", cv2.cvtColor(front, cv2.COLOR_RGB2BGR))
                if wrist is not None:
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
                    help="RT PC 的 IP（robot_server.py 所在机器）")
    ap.add_argument("--front-serial", required=True,
                    help="front RealSense 序列号（exterior_image_1_left）")
    ap.add_argument("--wrist-serial", required=True,
                    help="wrist RealSense 序列号（wrist_image_left）")
    ap.add_argument("--prompt", required=True,
                    help="任务描述文本，写入 meta/tasks.jsonl")
    ap.add_argument("--record-dir", required=True,
                    help="落盘目录（LeRobot v2.1，字段对齐 DROID）")
    ap.add_argument("--resize-size", type=int, default=224,
                    help="图像 resize + pad 到 (size, size)，对齐 pi05 输入")
    ap.add_argument("--record-fps", type=int, default=DROID_FPS,
                    help=f"采集频率，默认 {DROID_FPS}Hz（DROID 官方）")
    ap.add_argument("--display", action="store_true",
                    help="实时显示两路相机画面")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
