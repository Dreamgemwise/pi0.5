"""推理端 / 数据采集端共享的 IO 类。

抽离自 inference_client.py，使 data_collector.py 能复用同一套数据源而不
拖入 pi0.5 相关依赖（openpi_client 等）。

* RealSenseStream     —— 单台 RealSense 的后台抓帧线程
* RobotStateSubscriber —— 订阅 robot_server 的 ZMQ PUB 状态流
"""
from __future__ import annotations

import logging
import threading

import msgpack
import numpy as np
import zmq

try:
    import pyrealsense2 as rs
except ImportError as e:  # pragma: no cover
    raise RuntimeError(
        "FR3 realtime scripts require `pyrealsense2`. "
        "Install it in the active conda env with `python -m pip install pyrealsense2 pyzmq pyarrow`."
    ) from e

from common import STATE_PUB_PORT, RobotState


class RealSenseStream:
    """单台 RealSense 颜色流。启动后后台抓帧，主循环用 latest() 拿最新帧。"""

    def __init__(
        self,
        serial: str,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ) -> None:
        self._pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self._pipe.start(cfg)
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._stop = threading.Event()
        self._thr = threading.Thread(target=self._loop, daemon=True)
        self._thr.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                frames = self._pipe.wait_for_frames(timeout_ms=200)
                color = frames.get_color_frame()
                if color is None:
                    continue
                # BGR -> RGB
                img = np.asanyarray(color.get_data())[:, :, ::-1].copy()
                with self._lock:
                    self._frame = img
            except Exception as e:  # noqa: BLE001
                logging.debug("realsense frame timeout: %s", e)

    def latest(self) -> np.ndarray | None:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def close(self) -> None:
        self._stop.set()
        self._thr.join(timeout=1.0)
        try:
            self._pipe.stop()
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
