"""推理端 / 数据采集端共享的 IO 类。

抽离自 inference_client.py，使 data_collector.py 能复用同一套数据源而不
拖入 pi0.5 相关依赖（openpi_client 等）。

* ZedStream           —— 单台 ZED 的后台抓帧线程
* RobotStateSubscriber —— 订阅 robot_server 的 ZMQ PUB 状态流
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import time

import cv2
import msgpack
import numpy as np
import zmq

from common import STATE_PUB_PORT, RobotState


def _camera_source(source: str) -> int | str:
    return int(source) if source.isdecimal() else source


def _video_device_path(source: str) -> str | None:
    if source.isdecimal():
        return f"/dev/video{source}"
    if source.startswith("/dev/video"):
        return source
    return None


def _set_v4l2_format(source: str, width: int, height: int, fps: int, fourcc: str) -> None:
    device = _video_device_path(source)
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
        logging.warning("failed to set V4L2 format on %s: %s", device, msg)


class ZedStream:
    """单台 ZED UVC 彩色流。ZED UVC 帧为左右目横向拼接，这里取单目 RGB。"""

    def __init__(
        self,
        source: str,
        width: int = 2560,
        height: int = 720,
        fps: int = 30,
        eye: str = "LEFT",
        fourcc: str = "YUYV",
    ) -> None:
        self._source = str(source)
        self._eye = eye.upper()
        if self._eye not in {"LEFT", "RIGHT", "FULL"}:
            raise ValueError("eye must be one of: LEFT, RIGHT, FULL")

        _set_v4l2_format(self._source, width, height, fps, fourcc)
        self._cap = cv2.VideoCapture(_camera_source(self._source), cv2.CAP_V4L2)
        if not self._cap.isOpened():
            self._cap = cv2.VideoCapture(_camera_source(self._source))
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open ZED UVC camera source {self._source!r}")

        if fourcc:
            if len(fourcc) != 4:
                raise ValueError("camera fourcc must be exactly 4 characters, e.g. MJPG")
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc.upper()))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        self._cap.set(cv2.CAP_PROP_FPS, int(fps))

        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._stop = threading.Event()
        self._thr = threading.Thread(target=self._loop, daemon=True)
        self._thr.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                ok, frame = self._cap.read()
                if not ok or frame is None:
                    logging.debug("zed uvc frame read failed (%s)", self._source)
                    time.sleep(0.01)
                    continue

                if frame.ndim != 3 or frame.shape[2] < 3:
                    logging.debug("zed uvc frame has unexpected shape: %s", frame.shape)
                    continue

                if self._eye == "FULL":
                    mono = frame
                else:
                    half_width = frame.shape[1] // 2
                    # ZED UVC raw frames are side-by-side as RIGHT | LEFT.
                    if self._eye == "LEFT":
                        mono = frame[:, half_width:]
                    else:
                        mono = frame[:, :half_width]

                img = mono[:, :, :3][:, :, ::-1].copy()
                with self._lock:
                    self._frame = img
            except Exception as e:  # noqa: BLE001
                logging.debug("zed uvc frame read failed (%s): %s", self._source, e)

    def latest(self) -> np.ndarray | None:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def close(self) -> None:
        self._stop.set()
        self._thr.join(timeout=1.0)
        try:
            self._cap.release()
        except Exception:
            pass


class RobotStateSubscriber:
    """ZMQ SUB 订阅 robot_server 的 100Hz RobotState 流，只保留最新一条。"""

    def __init__(self, robot_ip: str) -> None:
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.connect(f"tcp://{robot_ip}:{STATE_PUB_PORT}")
        self._sock.setsockopt(zmq.SUBSCRIBE, b"")
        self._sock.setsockopt(zmq.CONFLATE, 1)
        self._lock = threading.Lock()
        self._latest: RobotState | None = None
        self._stop = threading.Event()
        self._thr = threading.Thread(target=self._loop, daemon=True)
        self._thr.start()

    def _loop(self) -> None:
        poller = zmq.Poller()
        poller.register(self._sock, zmq.POLLIN)
        while not self._stop.is_set():
            socks = dict(poller.poll(timeout=100))
            if self._sock in socks:
                raw = self._sock.recv()
                msg = msgpack.unpackb(raw, raw=False)
                state = RobotState.from_msg(msg)
                with self._lock:
                    self._latest = state

    def latest(self) -> RobotState | None:
        with self._lock:
            return self._latest

    def close(self) -> None:
        self._stop.set()
        self._thr.join(timeout=1.0)
