"""
OpenPI 推理：本地基准 或 WebSocket 远程推理（msgpack，与 houmo/run_pro_rtc_print 协议兼容）。

远程：客户端发送 LIBERO 格式观测（observation/image、wrist_image、state、prompt），
服务端返回 actions 及 server_timing。

依赖（仅 --serve）：  pip install msgpack websockets
.venv/bin/python src/infer_web.py --serve --host 0.0.0.0 --port 8000
aarch64 若报 libstdc++ CXXABI_1.3.15（OpenVINO/PyAV 与系统 lib 冲突），可先：
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
"""
from __future__ import annotations

import argparse
import asyncio
import functools
import logging
import os
from pathlib import Path
import time

import msgpack
import numpy as np
import torch
from openpi.policies import policy_config
from openpi.training import config as _config

# -----------------------------
# 基础环境配置
# -----------------------------
torch._dynamo.config.suppress_errors = True

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".85")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
checkpoint_dir = str(Path("models") / "pi05_libero")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _format_infer_stats_panel(
    *,
    title: str,
    index: int,
    rtc_on: bool,
    infer_ms: float,
    total_ms: float,
    avg_infer_ms: float,
    fps: float,
    period_ms: float,
    interval_ms: float,
    interval_known: bool,
) -> str:
    """与 houmo/run_pro_rtc_print.py 一致的盒线统计（仅打印时调用）。"""
    w = 58
    bar = "─" * w
    thin = "├" + "─" * 24 + "┼" + "─" * (w - 25) + "┤"
    bot = "╰" + bar + "╯"

    def row(left: str, right: str) -> str:
        lw = 23
        rw = w - lw - 5
        return "│ " + left[:lw].ljust(lw) + " │ " + right[:rw].rjust(rw) + " │"

    rtc_s = "ON" if rtc_on else "OFF"
    head = f" {title}  #{index}    RTC {rtc_s} "
    head = head[:w].ljust(w)
    interval_s = f"{interval_ms:.2f} ms" if interval_known else "—"
    lines = [
        "╭" + bar + "╮",
        "│" + head + "│",
        thin,
        row("infer (本次)", f"{infer_ms:.2f} ms"),
        row("total (全链路)", f"{total_ms:.2f} ms"),
        row("avg infer (均值)", f"{avg_infer_ms:.2f} ms"),
        row("距上次请求间隔", interval_s),
        thin,
        row("吞吐 / 等效周期", f"{fps:.2f} Hz  ·  {period_ms:.1f} ms"),
        bot,
    ]
    return "\n".join(lines)


def _summarize_obs_line(obs: dict) -> str:
    """单行摘要，便于终端对照请求参数。"""
    parts: list[str] = []
    if "prompt" in obs:
        p = str(obs["prompt"])
        parts.append(f'prompt="{p[:40]}{"…" if len(p) > 40 else ""}"')
    for key, short in (
        ("observation/image", "image"),
        ("observation/wrist_image", "wrist"),
        ("observation/state", "state"),
    ):
        if key in obs:
            x = np.asarray(obs[key])
            parts.append(f"{short} {tuple(x.shape)} {x.dtype}")
    return "  │ " + " · ".join(parts) if parts else "  │ (empty obs)"


# msgpack numpy（与 run_pro_rtc_print / openpi_client 一致）
def _pack_array(obj):
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_array(obj):
    if not isinstance(obj, dict):
        return obj
    if b"__ndarray__" in obj or "__ndarray__" in obj:
        d = obj.get(b"data") or obj.get("data")
        dt = obj.get(b"dtype") or obj.get("dtype")
        sh = obj.get(b"shape") or obj.get("shape")
        return np.ndarray(buffer=d, dtype=np.dtype(dt), shape=tuple(sh))
    if b"__npgeneric__" in obj or "__npgeneric__" in obj:
        dt = obj.get(b"dtype") or obj.get("dtype")
        d = obj.get(b"data") or obj.get("data")
        return np.dtype(dt).type(d)
    return obj


_packb = functools.partial(msgpack.packb, default=_pack_array)
_unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


def make_fixed_observation():
    return {
        "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/state": np.zeros((8,), dtype=np.float32),
        "prompt": "do something",
    }


def _load_policy(model_name: str, checkpoint_dir: str):
    logger.info("Config [%s]...", model_name)
    config = _config.get_config(model_name)
    logger.info("Creating trained policy from %s", checkpoint_dir)
    return policy_config.create_trained_policy(config, checkpoint_dir)


def run_local_benchmark(policy, inference_count: int, log_every: int) -> None:
    example = make_fixed_observation()
    print("Warming up...")
    _ = policy.infer(example)["actions"]
    _ = policy.infer(example)["actions"]
    print("-" * 50)
    print("Inference (local)...")
    times: list[float] = []
    first_t0 = None
    last_end: float | None = None
    for i in range(inference_count):
        t0 = time.perf_counter()
        out = policy.infer(example)
        t1 = time.perf_counter()
        _ = out["actions"]
        cost = t1 - t0
        times.append(cost)
        if first_t0 is None:
            first_t0 = t0
        elapsed = t1 - first_t0
        fps = (i + 1) / elapsed if elapsed > 0 else 0
        period_ms = 1000.0 / fps if fps > 0 else 0
        avg_ms = sum(times) * 1000 / len(times)
        interval_known = last_end is not None
        interval_ms = (t0 - last_end) * 1000 if interval_known else 0.0
        last_end = t1
        n = i + 1
        if log_every > 0 and n % log_every == 0:
            print(_summarize_obs_line(example), flush=True)
            print(
                _format_infer_stats_panel(
                    title="本地 openpi",
                    index=n,
                    rtc_on=False,
                    infer_ms=cost * 1000,
                    total_ms=cost * 1000,
                    avg_infer_ms=avg_ms,
                    fps=fps,
                    period_ms=period_ms,
                    interval_ms=interval_ms,
                    interval_known=interval_known,
                ),
                flush=True,
            )
    avg_latency_s = sum(times) / len(times)
    print("-" * 50)
    print(
        _format_infer_stats_panel(
            title=f"本地汇总 ({inference_count} iters)",
            index=inference_count,
            rtc_on=False,
            infer_ms=times[-1] * 1000,
            total_ms=times[-1] * 1000,
            avg_infer_ms=avg_latency_s * 1000,
            fps=1.0 / avg_latency_s if avg_latency_s > 0 else 0.0,
            period_ms=avg_latency_s * 1000,
            interval_ms=0.0,
            interval_known=False,
        ),
        flush=True,
    )


async def _websocket_handler(websocket, policy, log_every_n: int) -> None:
    logger.info("Connection from %s opened", websocket.remote_address)
    md = policy.metadata
    await websocket.send(_packb(md if isinstance(md, dict) else {}))

    request_count = 0
    first_t = None
    prev_total_t: float | None = None
    prev_end_t: float | None = None
    sum_infer_ms = 0.0

    while True:
        try:
            data = await websocket.recv()
            obs = _unpackb(bytes(data) if isinstance(data, (bytearray, memoryview)) else data)

            t0 = time.monotonic()
            out = policy.infer(obs)
            infer_t = time.monotonic() - t0

            request_count += 1
            infer_ms = infer_t * 1000
            sum_infer_ms += infer_ms
            if first_t is None:
                first_t = t0
            total_elapsed = time.monotonic() - first_t
            fps = request_count / total_elapsed if total_elapsed > 0 else 0
            period_ms = 1000.0 / fps if fps > 0 else 0
            avg_infer_ms = sum_infer_ms / request_count
            interval_known = prev_end_t is not None
            interval_ms = (t0 - prev_end_t) * 1000 if interval_known else 0.0
            prev_end_t = time.monotonic()

            out = dict(out)
            out["server_timing"] = {"infer_ms": infer_ms}
            if prev_total_t is not None:
                out["server_timing"]["prev_total_ms"] = prev_total_t * 1000
            prev_total_t = time.monotonic() - t0

            if log_every_n > 0 and request_count % log_every_n == 0:
                print(_summarize_obs_line(obs), flush=True)
                print(
                    _format_infer_stats_panel(
                        title="WebSocket openpi",
                        index=request_count,
                        rtc_on=False,
                        infer_ms=infer_ms,
                        total_ms=prev_total_t * 1000,
                        avg_infer_ms=avg_infer_ms,
                        fps=fps,
                        period_ms=period_ms,
                        interval_ms=interval_ms,
                        interval_known=interval_known,
                    ),
                    flush=True,
                )

            await websocket.send(_packb(out))

        except Exception as e:
            if "ConnectionClosed" in type(e).__name__:
                logger.info("Connection from %s closed", websocket.remote_address)
                break
            raise


def run_server(policy, host: str, port: int, log_every: int) -> None:
    try:
        import websockets.asyncio.server as ws_server
    except ImportError as e:
        raise SystemExit(
            "远程推理需要安装: pip install websockets msgpack\n" + str(e)
        ) from e

    async def _run():
        async with ws_server.serve(
            lambda ws: _websocket_handler(ws, policy, log_every),
            host,
            port,
            compression=None,
            max_size=None,
        ) as server:
            await server.serve_forever()

    asyncio.run(_run())


def main() -> None:
    p = argparse.ArgumentParser(description="OpenPI 推理：本地基准或 WebSocket 服务")
    p.add_argument("--model-name", default="pi05_libero", help="训练配置名")
    p.add_argument("--checkpoint-dir", default=str(Path("models") / "pi05_libero"), help="checkpoint 目录")
    p.add_argument("--serve", action="store_true", help="启动 WebSocket 服务，允许远程推理")
    p.add_argument("--host", default="0.0.0.0", help="监听地址")
    p.add_argument("--port", type=int, default=8000, help="监听端口")
    p.add_argument(
        "--log-every",
        type=int,
        default=3,
        help="终端面板：每 N 次请求/迭代打印（0=不打印过程，本地仍会打最终汇总）",
    )
    p.add_argument("--inference-count", type=int, default=5, help="本地模式迭代次数（--serve 时忽略）")
    args = p.parse_args()

    policy = _load_policy(args.model_name, args.checkpoint_dir)

    if args.serve:
        # 与客户端握手前做一次 warm-up，避免首包过慢
        _ = policy.infer(make_fixed_observation())["actions"]
        logger.info("OpenPI WebSocket 监听 %s:%d（msgpack 与 run_pro_rtc_print 兼容）", args.host, args.port)
        run_server(policy, args.host, args.port, args.log_every)
        return

    run_local_benchmark(policy, args.inference_count, args.log_every)


if __name__ == "__main__":
    main()
