"""Generate meta/episodes_stats.jsonl for a LeRobot v2.1 dataset from its parquet files.

For non-video features we compute real min/max/mean/std/count from the parquet rows.
For `dtype=="video"` features we fill reasonable placeholder stats with the shape
LeRobot expects (3, 1, 1). These placeholders don't affect openpi training because
openpi uses the norm stats from AssetsConfig (pointed at the pretrained DROID assets),
not the dataset-local stats.

Usage:
    python scripts/gen_episodes_stats.py --root datasets/fr3_pick_block
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _feature_stats(arr: np.ndarray) -> dict:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        keepdims = True
        axis: tuple[int, ...] | int = 0
    else:
        keepdims = False
        axis = 0
    return {
        "min": np.min(arr, axis=axis, keepdims=keepdims).tolist(),
        "max": np.max(arr, axis=axis, keepdims=keepdims).tolist(),
        "mean": np.mean(arr, axis=axis, keepdims=keepdims).tolist(),
        "std": np.std(arr, axis=axis, keepdims=keepdims).tolist(),
        "count": [int(len(arr))],
    }


def _placeholder_image_stats(count: int) -> dict:
    # Shape (3, 1, 1), values in [0, 1]. Safe defaults, not used by openpi training.
    return {
        "min": [[[0.0]], [[0.0]], [[0.0]]],
        "max": [[[1.0]], [[1.0]], [[1.0]]],
        "mean": [[[0.5]], [[0.5]], [[0.5]]],
        "std": [[[0.25]], [[0.25]], [[0.25]]],
        "count": [int(count)],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Dataset root, e.g. datasets/fr3_pick_block")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    info_path = root / "meta" / "info.json"
    episodes_path = root / "meta" / "episodes.jsonl"
    out_path = root / "meta" / "episodes_stats.jsonl"

    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"{out_path} already exists, use --overwrite to regenerate")

    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)
    features: dict = info["features"]
    data_path_fmt: str = info["data_path"]

    video_keys = [k for k, v in features.items() if v.get("dtype") == "video"]
    nonvideo_keys = [k for k, v in features.items() if v.get("dtype") != "video"]

    lines: list[str] = []
    with episodes_path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            ep = json.loads(raw)
            ep_idx: int = int(ep["episode_index"])
            ep_len: int = int(ep["length"])

            parquet_path = root / data_path_fmt.format(episode_chunk=0, episode_index=ep_idx)
            df = pd.read_parquet(parquet_path)

            ep_stats: dict = {}
            for key in nonvideo_keys:
                if key not in df.columns:
                    continue
                col = df[key].to_numpy()
                # object columns that hold np arrays per row need stacking
                if col.dtype == object:
                    col = np.stack([np.asarray(x) for x in col])
                ep_stats[key] = _feature_stats(col)

            for key in video_keys:
                ep_stats[key] = _placeholder_image_stats(ep_len)

            lines.append(json.dumps({"episode_index": ep_idx, "stats": ep_stats}))

    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out_path} with {len(lines)} episodes")


if __name__ == "__main__":
    main()
