#!/usr/bin/env python3
"""实时预览当前接入的 RealSense 彩色画面。"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


class RealSensePreview:
    def __init__(self, serial: str, width: int, height: int, fps: int) -> None:
        self.serial = serial
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.pipeline.start(config)

    def read(self, timeout_ms: int) -> np.ndarray | None:
        frames = self.pipeline.wait_for_frames(timeout_ms=timeout_ms)
        color = frames.get_color_frame()
        if color is None:
            return None
        return np.asanyarray(color.get_data()).copy()

    def close(self) -> None:
        try:
            self.pipeline.stop()
        except Exception:
            pass


def discover_serials() -> list[str]:
    context = rs.context()
    devices = list(context.query_devices())
    return [dev.get_info(rs.camera_info.serial_number) for dev in devices]


def add_label(image: np.ndarray, label: str) -> np.ndarray:
    annotated = image.copy()
    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(
        annotated,
        label,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return annotated


def make_mosaic(images: list[np.ndarray], tile_width: int, tile_height: int) -> np.ndarray:
    cols = 1 if len(images) == 1 else 2
    rows = math.ceil(len(images) / cols)
    canvas = np.zeros((rows * tile_height, cols * tile_width, 3), dtype=np.uint8)

    for idx, image in enumerate(images):
        row = idx // cols
        col = idx % cols
        y0 = row * tile_height
        x0 = col * tile_width
        canvas[y0 : y0 + tile_height, x0 : x0 + tile_width] = image
    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="实时预览当前接入的 RealSense 彩色画面。")
    parser.add_argument("--width", type=int, default=640, help="彩色流宽度")
    parser.add_argument("--height", type=int, default=480, help="彩色流高度")
    parser.add_argument("--fps", type=int, default=30, help="彩色流帧率")
    parser.add_argument(
        "--serial",
        action="append",
        default=None,
        help="指定要打开的序列号，可重复传入；默认自动打开所有已连接设备",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("tmp/realsense_preview"),
        help="按 s 保存截图时的输出目录",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=1000,
        help="单次等帧超时时间",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    serials = args.serial or discover_serials()
    if not serials:
        print("No RealSense devices detected.")
        return 1

    print("Opening RealSense devices:")
    for serial in serials:
        print(f"  - {serial}")

    previews: list[RealSensePreview] = []
    try:
        for serial in serials:
            previews.append(RealSensePreview(serial, args.width, args.height, args.fps))

        for _ in range(30):
            for preview in previews:
                preview.read(timeout_ms=args.timeout_ms)

        print("Preview started. Press 'q' to quit, 's' to save current frames.")
        while True:
            tiles: list[np.ndarray] = []
            saved_frames: list[tuple[str, np.ndarray]] = []
            for preview in previews:
                frame = preview.read(timeout_ms=args.timeout_ms)
                if frame is None:
                    continue
                saved_frames.append((preview.serial, frame))
                tiles.append(add_label(frame, f"serial: {preview.serial}"))

            if not tiles:
                print("No frames available from any camera.")
                time.sleep(0.1)
                continue

            mosaic = make_mosaic(tiles, args.width, args.height)
            cv2.imshow("RealSense Preview", mosaic)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                args.save_dir.mkdir(parents=True, exist_ok=True)
                ts = time.strftime("%Y%m%d_%H%M%S")
                for serial, frame in saved_frames:
                    path = args.save_dir / f"{serial}_{ts}.png"
                    cv2.imwrite(str(path), frame)
                    print(f"Saved {path}")
    finally:
        for preview in previews:
            preview.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
