"""pi05_droid fine-tune 数据采集。

和 inference_client.py 跑在同一进程，借用已有的 RobotStateSubscriber 和
RealSense 流。按 LeRobot v2.1 格式（parquet + mp4）落盘，字段名对齐
DROID（这样数据能直接喂给 openpi 训练流程）：

    observation.state                          (n, 8) float32
        = [joint_position(7), gripper_position(1)]
    action                                     (n, 8) float32
        = [joint_velocity(7), gripper_position(1)]
    observation.images.exterior_image_1_left   (n, H, W, 3) uint8 (mp4)
    observation.images.wrist_image_left        (n, H, W, 3) uint8 (mp4)

action 的来源：相邻两帧 joint_position 差分 / dt（= 机器人 **实际执行到**
的关节速度）。pi0.5 原始输出的 action chunk 只作为 sidecar 辅助信息，
**不用作** 训练 target —— 这是典型 DAgger-style 数据。

采集频率：15 Hz（DROID 官方）。
键盘热键（在 inference_client 的终端窗口里）：
    Enter = 开始新的一集
    s     = 停止并保存当前集
    r     = 丢弃当前集
    q     = 优雅退出整个程序

依赖：
    pip install pyarrow opencv-python
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from common import GRIPPER_MAX_WIDTH, RobotState, mono


def _lazy_pyarrow():
    """延迟 import pyarrow，只有真正要写 parquet 时才要求装。"""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "数据采集需要 pyarrow，请先 pip install pyarrow"
        ) from e
    return pa, pq

# --------- LeRobot v2.1 字段名（对齐 DROID LeRobot 数据集）----------
KEY_STATE = "observation.state"
KEY_ACTION = "action"
KEY_IMG_EXT = "observation.images.exterior_image_1_left"
KEY_IMG_WRIST = "observation.images.wrist_image_left"

# DROID 官方训练频率
DROID_FPS = 15


# =====================================================================
# 单帧 / 单集 内存缓冲
# =====================================================================
@dataclass
class _Step:
    ts_mono: float                  # 本机单调时钟（仅 debug/诊断用）
    joint_position: np.ndarray      # (7,)   float32
    gripper_width: float            # meter
    img_ext: np.ndarray             # (H, W, 3) uint8 RGB
    img_wrist: np.ndarray           # (H, W, 3) uint8 RGB


@dataclass
class _Episode:
    steps: list[_Step] = field(default_factory=list)
    task_prompt: str = ""
    # pi0.5 每次推理输出的 chunk，仅作为 sidecar 参考，不落进 parquet 的 action 列
    policy_chunks: list[dict] = field(default_factory=list)


# =====================================================================
# meta/episodes_stats.jsonl（LeRobot v2.1 格式）
# =====================================================================
def _feature_stats_1d(arr: np.ndarray) -> dict:
    """Per-feature stats for an (n, d) or (n,) array (state/action)."""
    arr = np.asarray(arr)
    keepdims = arr.ndim == 1
    return {
        "min": np.min(arr, axis=0, keepdims=keepdims).tolist(),
        "max": np.max(arr, axis=0, keepdims=keepdims).tolist(),
        "mean": np.mean(arr, axis=0, keepdims=keepdims).tolist(),
        "std": np.std(arr, axis=0, keepdims=keepdims).tolist(),
        "count": [int(len(arr))],
    }


def _placeholder_image_stats(count: int) -> dict:
    """LeRobot 期望 shape=(3,1,1) 的图像 stats；训练时实际归一化用 AssetsConfig 的
    官方 DROID norm_stats，这里占位即可。"""
    return {
        "min": [[[0.0]], [[0.0]], [[0.0]]],
        "max": [[[1.0]], [[1.0]], [[1.0]]],
        "mean": [[[0.5]], [[0.5]], [[0.5]]],
        "std": [[[0.25]], [[0.25]], [[0.25]]],
        "count": [int(count)],
    }


def _append_episode_stats(
    root: Path,
    episode_index: int,
    ep_length: int,
    state: np.ndarray,
    action: np.ndarray,
) -> None:
    stats = {
        KEY_STATE: _feature_stats_1d(state),
        KEY_ACTION: _feature_stats_1d(action),
        KEY_IMG_EXT: _placeholder_image_stats(ep_length),
        KEY_IMG_WRIST: _placeholder_image_stats(ep_length),
    }
    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    with (meta / "episodes_stats.jsonl").open("a", encoding="utf-8") as f:
        f.write(
            json.dumps({"episode_index": int(episode_index), "stats": stats}) + "\n"
        )


# =====================================================================
# 把一集写入 LeRobot v2.1 目录结构
# =====================================================================
def _write_episode(
    root: Path,
    episode_index: int,
    episode: _Episode,
    task_index: int,
    global_index_start: int,
    img_size: int,
    fps: int,
) -> int:
    n = len(episode.steps)
    if n < 2:
        logging.warning("episode %d 只有 %d 帧，跳过写盘", episode_index, n)
        return 0

    pa, pq = _lazy_pyarrow()
    dt = 1.0 / fps

    # ---- observation.state ----
    q = np.stack(
        [s.joint_position.astype(np.float32) for s in episode.steps], axis=0
    )  # (n, 7)
    widths = np.array([s.gripper_width for s in episode.steps], dtype=np.float32)
    gripper_norm = np.clip(
        1.0 - widths / GRIPPER_MAX_WIDTH, 0.0, 1.0
    ).astype(np.float32)
    state = np.concatenate([q, gripper_norm[:, None]], axis=1)  # (n, 8)

    # ---- action ----
    # 关节速度 = 相邻两帧 joint_position 差分 / dt
    joint_vel = np.zeros((n, 7), dtype=np.float32)
    joint_vel[:-1] = (q[1:] - q[:-1]) / dt
    # 最后一帧复制倒数第二帧（或 0），反正训练时通常 mask 掉末尾
    joint_vel[-1] = joint_vel[-2] if n >= 2 else 0.0

    # gripper action：取下一帧的 gripper_position 作为"命令目标"
    gripper_act = np.zeros(n, dtype=np.float32)
    gripper_act[:-1] = gripper_norm[1:]
    gripper_act[-1] = gripper_norm[-1]

    action = np.concatenate([joint_vel, gripper_act[:, None]], axis=1)  # (n, 8)

    # ---- videos/chunk-000/<key>/episode_xxxxxx.mp4 ----
    video_root = root / "videos" / "chunk-000"
    for key, attr in [(KEY_IMG_EXT, "img_ext"), (KEY_IMG_WRIST, "img_wrist")]:
        vdir = video_root / key
        vdir.mkdir(parents=True, exist_ok=True)
        vpath = vdir / f"episode_{episode_index:06d}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(vpath), fourcc, fps, (img_size, img_size))
        if not writer.isOpened():
            logging.error("打开 VideoWriter 失败: %s", vpath)
            return 0
        for step in episode.steps:
            img = getattr(step, attr)  # RGB uint8
            writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        writer.release()

    # ---- data/chunk-000/episode_xxxxxx.parquet ----
    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)

    timestamp = (np.arange(n, dtype=np.float32) * dt).tolist()
    frame_index = np.arange(n, dtype=np.int64).tolist()
    global_index = np.arange(
        global_index_start, global_index_start + n, dtype=np.int64
    ).tolist()
    ep_idx = [int(episode_index)] * n
    t_idx = [int(task_index)] * n

    table = pa.table(
        {
            KEY_STATE: [row.tolist() for row in state],
            KEY_ACTION: [row.tolist() for row in action],
            "timestamp": timestamp,
            "frame_index": frame_index,
            "episode_index": ep_idx,
            "index": global_index,
            "task_index": t_idx,
        }
    )
    pq.write_table(table, data_dir / f"episode_{episode_index:06d}.parquet")

    # ---- meta/episodes_stats.jsonl：LeRobot v2.1 要求的 per-episode stats ----
    _append_episode_stats(
        root=root,
        episode_index=episode_index,
        ep_length=n,
        state=state,
        action=action,
    )

    # ---- sidecar：pi0.5 输出的 chunk（可选，JSON 便于肉眼看）----
    if episode.policy_chunks:
        sidecar_dir = root / "policy_chunks"
        sidecar_dir.mkdir(parents=True, exist_ok=True)
        with (sidecar_dir / f"episode_{episode_index:06d}.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(episode.policy_chunks, f)

    return n


# =====================================================================
# DatasetRecorder：对外的主接口
# =====================================================================
class DatasetRecorder:
    """后台 15 Hz 抓帧，热键触发 start / stop / discard。"""

    def __init__(
        self,
        root_dir: str,
        task_prompt: str,
        img_size: int = 224,
        fps: int = DROID_FPS,
    ) -> None:
        self._root = Path(root_dir)
        (self._root / "meta").mkdir(parents=True, exist_ok=True)
        (self._root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
        (self._root / "videos" / "chunk-000").mkdir(parents=True, exist_ok=True)

        self._task_prompt = task_prompt
        self._img_size = img_size
        self._fps = fps
        self._dt = 1.0 / fps

        # 被外部注入的数据源
        self._sub = None           # RobotStateSubscriber
        self._cam_ext = None       # RealSenseStream (front / exterior)
        self._cam_wrist = None     # RealSenseStream (wrist)

        self._ep_lock = threading.Lock()
        self._episode: _Episode | None = None
        self._recording = threading.Event()
        self._stop = threading.Event()

        self._ep_counter = self._discover_next_episode_idx()
        self._global_index = self._discover_next_global_index()
        self._task_index = 0  # 目前只支持一个 task prompt

        self._thr = threading.Thread(
            target=self._capture_loop, daemon=True, name="recorder"
        )

    # ---- 资源注入 & 生命周期 ----
    def attach(self, sub: Any, cam_ext: Any, cam_wrist: Any) -> None:
        self._sub = sub
        self._cam_ext = cam_ext
        self._cam_wrist = cam_wrist

    def start(self) -> None:
        self._thr.start()

    def stop(self) -> None:
        self._stop.set()
        self._thr.join(timeout=2.0)

    # ---- Episode 控制（由 HotkeyListener 调用）----
    def start_episode(self) -> None:
        with self._ep_lock:
            if self._recording.is_set():
                logging.warning("[recorder] 当前已在录制中，忽略 start")
                return
            self._episode = _Episode(task_prompt=self._task_prompt)
            self._recording.set()
            logging.info(
                "[recorder] >>> start episode %d (prompt=%r)",
                self._ep_counter, self._task_prompt,
            )

    def discard_episode(self) -> None:
        with self._ep_lock:
            if not self._recording.is_set():
                return
            n = len(self._episode.steps) if self._episode else 0
            self._episode = None
            self._recording.clear()
            logging.info("[recorder] !!! discard episode (frames=%d)", n)

    def stop_episode(self) -> None:
        with self._ep_lock:
            if not self._recording.is_set():
                return
            ep = self._episode
            self._episode = None
            self._recording.clear()
        if ep is None:
            return
        n = _write_episode(
            root=self._root,
            episode_index=self._ep_counter,
            episode=ep,
            task_index=self._task_index,
            global_index_start=self._global_index,
            img_size=self._img_size,
            fps=self._fps,
        )
        if n > 0:
            self._global_index += n
            self._update_meta(ep_length=n)
            self._ep_counter += 1
        logging.info("[recorder] <<< episode saved (frames=%d)", n)

    # ---- pi0.5 sidecar ----
    def note_policy_chunk(
        self, actions: np.ndarray, capture_ts: float
    ) -> None:
        """每次 pi0.5 推理完调用，附加到当前 episode 的 sidecar。"""
        with self._ep_lock:
            if self._episode is None:
                return
            self._episode.policy_chunks.append(
                {
                    "ts": float(capture_ts),
                    "actions": np.asarray(actions, dtype=np.float32).tolist(),
                }
            )

    # ---- 内部 ----
    def _discover_next_episode_idx(self) -> int:
        max_idx = -1
        for f in (self._root / "data" / "chunk-000").glob("episode_*.parquet"):
            try:
                max_idx = max(max_idx, int(f.stem.split("_")[-1]))
            except ValueError:
                pass
        return max_idx + 1

    def _discover_next_global_index(self) -> int:
        idx = 0
        files = list((self._root / "data" / "chunk-000").glob("episode_*.parquet"))
        if not files:
            return 0
        _, pq = _lazy_pyarrow()
        for f in files:
            try:
                tbl = pq.read_table(str(f), columns=["index"])
                vals = tbl.column("index").to_numpy()
                if len(vals) > 0:
                    idx = max(idx, int(vals.max()) + 1)
            except Exception:  # noqa: BLE001
                pass
        return idx

    def _update_meta(self, ep_length: int) -> None:
        meta = self._root / "meta"

        # tasks.jsonl：首次写入
        tasks_path = meta / "tasks.jsonl"
        if not tasks_path.exists():
            with tasks_path.open("w", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {"task_index": self._task_index, "task": self._task_prompt}
                    )
                    + "\n"
                )

        # episodes.jsonl：append
        with (meta / "episodes.jsonl").open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "episode_index": self._ep_counter,
                        "tasks": [self._task_prompt],
                        "length": ep_length,
                    }
                )
                + "\n"
            )

        # info.json：整体覆写
        total_eps = self._ep_counter + 1
        info = {
            "codebase_version": "v2.1",
            "robot_type": "franka_fr3",
            "total_episodes": total_eps,
            "total_frames": self._global_index,
            "total_tasks": 1,
            "total_videos": total_eps * 2,
            "total_chunks": 1,
            "chunks_size": 1000,
            "fps": self._fps,
            "splits": {"train": f"0:{total_eps}"},
            "data_path": (
                "data/chunk-{episode_chunk:03d}/"
                "episode_{episode_index:06d}.parquet"
            ),
            "video_path": (
                "videos/chunk-{episode_chunk:03d}/"
                "{video_key}/episode_{episode_index:06d}.mp4"
            ),
            "features": {
                KEY_STATE: {
                    "dtype": "float32",
                    "shape": [8],
                    "names": [
                        "joint_0", "joint_1", "joint_2", "joint_3",
                        "joint_4", "joint_5", "joint_6", "gripper_position",
                    ],
                },
                KEY_ACTION: {
                    "dtype": "float32",
                    "shape": [8],
                    "names": [
                        "joint_velocity_0", "joint_velocity_1",
                        "joint_velocity_2", "joint_velocity_3",
                        "joint_velocity_4", "joint_velocity_5",
                        "joint_velocity_6", "gripper_position",
                    ],
                },
                KEY_IMG_EXT: {
                    "dtype": "video",
                    "shape": [self._img_size, self._img_size, 3],
                    "names": ["height", "width", "channels"],
                    "info": {
                        "video.fps": float(self._fps),
                        "video.channels": 3,
                        "video.codec": "mp4v",
                    },
                },
                KEY_IMG_WRIST: {
                    "dtype": "video",
                    "shape": [self._img_size, self._img_size, 3],
                    "names": ["height", "width", "channels"],
                    "info": {
                        "video.fps": float(self._fps),
                        "video.channels": 3,
                        "video.codec": "mp4v",
                    },
                },
                "timestamp": {"dtype": "float32", "shape": [1]},
                "frame_index": {"dtype": "int64", "shape": [1]},
                "episode_index": {"dtype": "int64", "shape": [1]},
                "index": {"dtype": "int64", "shape": [1]},
                "task_index": {"dtype": "int64", "shape": [1]},
            },
        }
        with (meta / "info.json").open("w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)

    def _capture_loop(self) -> None:
        while not self._stop.is_set():
            t0 = mono()
            if self._recording.is_set() and self._sub is not None:
                state: RobotState | None = self._sub.latest()
                ext = self._cam_ext.latest() if self._cam_ext else None
                wrist = self._cam_wrist.latest() if self._cam_wrist else None
                if state is not None and ext is not None and wrist is not None:
                    step = _Step(
                        ts_mono=t0,
                        joint_position=state.joint_position.astype(np.float32).copy(),
                        gripper_width=float(state.gripper_width),
                        img_ext=_resize_pad(ext, self._img_size),
                        img_wrist=_resize_pad(wrist, self._img_size),
                    )
                    with self._ep_lock:
                        if self._episode is not None:
                            self._episode.steps.append(step)
            elapsed = mono() - t0
            if elapsed < self._dt:
                time.sleep(self._dt - elapsed)


def _resize_pad(img: np.ndarray, size: int) -> np.ndarray:
    """保长宽比 resize + pad 到 (size, size)，输出 RGB uint8。"""
    try:
        from openpi_client import image_tools  # lazy: inference_client 已装
        out = image_tools.resize_with_pad(img, size, size)
        return image_tools.convert_to_uint8(out)
    except ImportError:
        h, w = img.shape[:2]
        scale = size / max(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        pad = np.zeros((size, size, 3), dtype=np.uint8)
        y0, x0 = (size - nh) // 2, (size - nw) // 2
        pad[y0 : y0 + nh, x0 : x0 + nw] = resized
        return pad


# =====================================================================
# 终端非阻塞键盘监听
# =====================================================================
class HotkeyListener:
    """监听 stdin，把按键分发到回调。必须 stdin 是 tty。

    热键：
        Enter (\\r / \\n)  -> on_start
        s                  -> on_stop
        r                  -> on_discard
        q                  -> on_quit（会停止监听线程）
    """

    def __init__(
        self,
        on_start: Callable[[], None],
        on_stop: Callable[[], None],
        on_discard: Callable[[], None],
        on_quit: Callable[[], None],
    ) -> None:
        self._on_start = on_start
        self._on_stop = on_stop
        self._on_discard = on_discard
        self._on_quit = on_quit
        self._stop = threading.Event()
        self._thr = threading.Thread(target=self._loop, daemon=True, name="hotkey")

    def start(self) -> None:
        self._thr.start()

    def stop(self) -> None:
        self._stop.set()
        self._thr.join(timeout=1.0)

    def _loop(self) -> None:
        if not sys.stdin.isatty():
            logging.warning(
                "[hotkey] stdin 不是 tty（nohup/pipe 运行？），热键监听已禁用"
            )
            return
        # 仅 Unix：用 termios + select 非阻塞读
        import select
        import termios
        import tty

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not self._stop.is_set():
                r, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not r:
                    continue
                ch = os.read(fd, 1).decode(errors="ignore")
                if ch in ("\r", "\n"):
                    self._on_start()
                elif ch == "s":
                    self._on_stop()
                elif ch == "r":
                    self._on_discard()
                elif ch == "q":
                    self._on_quit()
                    break
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


HOTKEY_HELP = (
    "Hotkeys:  <Enter>=start episode   s=stop & save   "
    "r=discard current   q=quit"
)
