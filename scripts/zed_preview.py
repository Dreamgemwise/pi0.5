#!/usr/bin/env python3
"""实时预览 ZED UVC 单目 RGB 画面。"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np


ZED_UVC_MODES = {
    "SVGA": (1344, 376, 60),
    "HD720": (2560, 720, 30),
    "HD1080": (3840, 1080, 30),
    "HD2K": (4416, 1242, 15),
}


def _camera_source(source: str) -> int | str:
    return int(source) if source.isdecimal() else source


def _video_device_path(source: str) -> str | None:
    if source.isdecimal():
        return f"/dev/video{source}"
    if source.startswith("/dev/video"):
        return source
    return None


def _set_v4l2_format(source: str, width: int, height: int, fps: int, fourcc: str) -> None:
    device = _video_device_path(str(source))
    if device is None or not shutil.which("v4l2-ctl"):
        return
    if fourcc:
        fmt = f"width={int(width)},height={int(height)},pixelformat={fourcc.upper()}"
    else:
        fmt = f"width={int(width)},height={int(height)}"
    result = subprocess.run(
        [
            "v4l2-ctl",
            "-d",
            device,
            f"--set-fmt-video={fmt}",
            f"--set-parm={int(fps)}",
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        msg = (result.stderr or result.stdout).strip()
        print(f"Warning: failed to set V4L2 format on {device}: {msg}")


class ZedPreview:
    def __init__(
        self,
        source: str,
        width: int,
        height: int,
        fps: int,
        eye: str,
        fourcc: str,
    ) -> None:
        self.source = str(source)
        self.eye = eye.upper()
        if self.eye not in {"LEFT", "RIGHT", "FULL"}:
            raise ValueError("eye must be one of: LEFT, RIGHT, FULL")

        _set_v4l2_format(self.source, width, height, fps, fourcc)
        self.cap = cv2.VideoCapture(_camera_source(self.source), cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(_camera_source(self.source))
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera source {self.source!r}")

        if fourcc:
            if len(fourcc) != 4:
                raise ValueError("camera fourcc must be exactly 4 characters, e.g. MJPG")
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc.upper()))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        self.cap.set(cv2.CAP_PROP_FPS, int(fps))
        self.requested_width = int(width)
        self.requested_height = int(height)
        self.requested_fps = int(fps)

    def describe(self) -> str:
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        fourcc_value = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join(chr((fourcc_value >> (8 * i)) & 0xFF) for i in range(4))
        requested = f"requested={self.requested_width}x{self.requested_height}@{self.requested_fps}"
        return f"{self.source}: {requested} capture={width}x{height}@{fps:.1f} fourcc={fourcc!r} eye={self.eye}"

    def format_matches_request(self) -> bool:
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return width == self.requested_width and height == self.requested_height

    def read(self) -> np.ndarray | None:
        ok, frame = self.cap.read()
        if not ok or frame is None or frame.ndim != 3 or frame.shape[2] < 3:
            return None
        if self.eye == "FULL":
            return frame[:, :, :3].copy()

        half_width = frame.shape[1] // 2
        # ZED UVC raw frames are side-by-side as RIGHT | LEFT.
        if self.eye == "LEFT":
            return frame[:, half_width:, :3].copy()
        return frame[:, :half_width, :3].copy()

    def close(self) -> None:
        try:
            self.cap.release()
        except Exception:
            pass


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


def fit_tile(image: np.ndarray, tile_width: int, tile_height: int, resize: bool) -> np.ndarray:
    if not resize:
        return image

    h, w = image.shape[:2]
    scale = min(tile_width / w, tile_height / h)
    resized_w = max(1, int(round(w * scale)))
    resized_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    tile = np.zeros((tile_height, tile_width, 3), dtype=np.uint8)
    y0 = (tile_height - resized_h) // 2
    x0 = (tile_width - resized_w) // 2
    tile[y0 : y0 + resized_h, x0 : x0 + resized_w] = resized
    return tile


def make_mosaic(images: list[np.ndarray], tile_width: int, tile_height: int, resize: bool) -> np.ndarray:
    if not resize:
        if len(images) == 1:
            return images[0]
        tile_height = max(image.shape[0] for image in images)
        tile_width = max(image.shape[1] for image in images)

    cols = 1 if len(images) == 1 else 2
    rows = math.ceil(len(images) / cols)
    canvas = np.zeros((rows * tile_height, cols * tile_width, 3), dtype=np.uint8)

    for idx, image in enumerate(images):
        row = idx // cols
        col = idx % cols
        y0 = row * tile_height
        x0 = col * tile_width
        tile = fit_tile(image, tile_width, tile_height, resize)
        h, w = tile.shape[:2]
        canvas[y0 : y0 + h, x0 : x0 + w] = tile
    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="实时预览 ZED UVC 单目 RGB 画面。")
    parser.add_argument(
        "--camera",
        action="append",
        default=None,
        help="要打开的 UVC 源，比如 0 或 /dev/video0；可重复传入。默认打开 0",
    )
    parser.add_argument(
        "--camera-mode",
        choices=sorted(ZED_UVC_MODES),
        default=None,
        help="ZED UVC 预设模式；HD2K 为 4416x1242@15fps",
    )
    parser.add_argument("--camera-width", type=int, default=2560, help="ZED UVC 拼接帧宽度，HD720 通常为 2560")
    parser.add_argument("--camera-height", type=int, default=720, help="ZED UVC 拼接帧高度，HD720 通常为 720")
    parser.add_argument("--camera-fps", type=int, default=30, help="采集帧率")
    parser.add_argument("--camera-fourcc", default="YUYV", help="OpenCV 请求的 UVC fourcc；ZED 2i UVC 通常为 YUYV")
    parser.add_argument(
        "--camera-eye",
        default="LEFT",
        type=str.upper,
        choices=["LEFT", "RIGHT", "FULL"],
        help="从 ZED 左右拼接帧里取左目/右目；FULL 表示不切半",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("tmp/zed_preview"),
        help="按 s 保存截图时的输出目录",
    )
    parser.add_argument("--tile-width", type=int, default=960, help="预览拼图单格宽度；仅影响显示窗口")
    parser.add_argument("--tile-height", type=int, default=540, help="预览拼图单格高度；仅影响显示窗口")
    parser.add_argument("--no-resize", action="store_true", help="按真实帧尺寸显示，不缩放预览画面")
    args = parser.parse_args()

    if args.camera_mode is not None:
        args.camera_width, args.camera_height, args.camera_fps = ZED_UVC_MODES[args.camera_mode]
    if args.tile_width <= 0:
        args.tile_width = args.camera_width if args.camera_eye == "FULL" else args.camera_width // 2
    if args.tile_height <= 0:
        args.tile_height = args.camera_height
    return args


def main() -> int:
    args = parse_args()
    sources = args.camera or ["0"]

    print("Opening camera sources:")
    for source in sources:
        print(f"  - {source}")
    print("Tip: use `v4l2-ctl --list-devices` or `ls /dev/video*` to find ZED video nodes.")

    previews: list[ZedPreview] = []
    try:
        for source in sources:
            preview = ZedPreview(
                source,
                args.camera_width,
                args.camera_height,
                args.camera_fps,
                args.camera_eye,
                args.camera_fourcc,
            )
            previews.append(preview)
            print(f"Opened {preview.describe()}")
            if not preview.format_matches_request():
                print(
                    "Warning: capture size does not match the requested mode. "
                    "Run `v4l2-ctl --get-fmt-video -d /dev/video0` to inspect the active format."
                )

        for _ in range(30):
            for preview in previews:
                preview.read()

        print("Preview started. Press 'q' to quit, 's' to save current frames.")
        last_no_frame_log = 0.0
        while True:
            tiles: list[np.ndarray] = []
            saved_frames: list[tuple[str, np.ndarray]] = []
            for preview in previews:
                frame = preview.read()
                if frame is None:
                    continue
                saved_frames.append((preview.source, frame))
                h, w = frame.shape[:2]
                tiles.append(add_label(frame, f"source: {preview.source}  eye: {preview.eye}  {w}x{h}"))

            if not tiles:
                now = time.monotonic()
                if now - last_no_frame_log > 2.0:
                    print(
                        "Waiting for frames. If this persists, try another node such as /dev/video1, "
                        "or inspect formats with `v4l2-ctl --list-formats-ext -d /dev/video0`."
                    )
                    last_no_frame_log = now
                time.sleep(0.1)
                continue

            mosaic = make_mosaic(tiles, args.tile_width, args.tile_height, not args.no_resize)
            cv2.imshow("ZED UVC Preview", mosaic)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                args.save_dir.mkdir(parents=True, exist_ok=True)
                ts = time.strftime("%Y%m%d_%H%M%S")
                for source, frame in saved_frames:
                    safe_source = source.replace("/", "_")
                    path = args.save_dir / f"{safe_source}_{ts}.png"
                    cv2.imwrite(str(path), frame)
                    print(f"Saved {path}")
    finally:
        for preview in previews:
            preview.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
