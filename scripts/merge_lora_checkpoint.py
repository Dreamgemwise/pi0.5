"""Merge a local base checkpoint with an adapter-only LoRA checkpoint.

The output directory is a full OpenPI checkpoint root that can be passed to
scripts/serve_policy.py with --policy.dir.

Example:
    python scripts/merge_lora_checkpoint.py \
        --base-dir models/pi05_droid \
        --lora-dir checkpoints_lora/fr3_lora_199999 \
        --out-dir checkpoints/fr3_lora_merged/199999 \
        --overwrite
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
from typing import Any

from flax import traverse_util
import numpy as np
import orbax.checkpoint as ocp

from openpi.models import model as _model


def _resolve_params_dir(path: pathlib.Path) -> pathlib.Path:
    if path.name == "params":
        return path
    if (path / "params").is_dir():
        return path / "params"
    return path


def _checkpoint_root(path: pathlib.Path) -> pathlib.Path:
    return path.parent if path.name == "params" else path


def _copy_assets(
    base_root: pathlib.Path,
    lora_root: pathlib.Path,
    out_dir: pathlib.Path,
    assets_dir: pathlib.Path | None,
) -> str | None:
    candidates = []
    if assets_dir is not None:
        candidates.append(assets_dir)
    candidates.extend([lora_root / "assets", base_root / "assets"])

    for candidate in candidates:
        if candidate.is_dir():
            shutil.copytree(candidate, out_dir / "assets")
            return str(candidate)
    return None


def _save_params(params_dir: pathlib.Path, params: dict) -> None:
    params_dir = params_dir.expanduser().resolve()
    params_dir.parent.mkdir(parents=True, exist_ok=True)
    with ocp.PyTreeCheckpointer() as ckptr:
        ckptr.save(params_dir, args=ocp.args.PyTreeSave({"params": params}))


def _count_leaves(tree: Any) -> tuple[int, int]:
    leaves = traverse_util.flatten_dict(tree)
    total_values = 0
    for value in leaves.values():
        if hasattr(value, "size"):
            total_values += int(value.size)
    return len(leaves), total_values


def merge_lora(
    base_dir: pathlib.Path,
    lora_dir: pathlib.Path,
    out_dir: pathlib.Path,
    *,
    assets_dir: pathlib.Path | None,
    overwrite: bool,
) -> None:
    base_dir = base_dir.expanduser()
    lora_dir = lora_dir.expanduser()
    out_dir = out_dir.expanduser().resolve()
    assets_dir = None if assets_dir is None else assets_dir.expanduser().resolve()

    base_params_dir = _resolve_params_dir(base_dir).resolve()
    lora_params_dir = _resolve_params_dir(lora_dir).resolve()
    base_root = _checkpoint_root(base_dir).resolve()
    lora_root = _checkpoint_root(lora_dir).resolve()

    if not base_params_dir.exists():
        raise FileNotFoundError(f"Base params directory not found: {base_params_dir}")
    if not lora_params_dir.exists():
        raise FileNotFoundError(f"LoRA params directory not found: {lora_params_dir}")
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{out_dir} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(out_dir)

    base_params = _model.restore_params(base_params_dir, restore_type=np.ndarray)
    lora_params = _model.restore_params(lora_params_dir, restore_type=np.ndarray)

    flat_base = traverse_util.flatten_dict(base_params, sep="/")
    flat_lora = traverse_util.flatten_dict(lora_params, sep="/")
    non_lora = [key for key in flat_lora if "lora" not in key.lower()]
    if non_lora:
        preview = "\n  ".join(non_lora[:10])
        raise ValueError(
            "LoRA checkpoint contains non-LoRA keys. Refusing to merge.\n"
            f"First keys:\n  {preview}"
        )
    if not flat_lora:
        raise ValueError(f"No LoRA parameters found in {lora_params_dir}")

    merged_flat = dict(flat_base)
    merged_flat.update(flat_lora)
    merged_params = traverse_util.unflatten_dict(merged_flat, sep="/")

    out_params_dir = out_dir / "params"
    _save_params(out_params_dir, merged_params)
    copied_assets_from = _copy_assets(base_root, lora_root, out_dir, assets_dir)

    lora_leaf_count, lora_value_count = _count_leaves(lora_params)
    merged_leaf_count, merged_value_count = _count_leaves(merged_params)
    metadata = {
        "format": "openpi_full_params_from_base_plus_lora",
        "base_params": str(base_params_dir),
        "lora_params": str(lora_params_dir),
        "assets_from": copied_assets_from,
        "lora_leaf_count": lora_leaf_count,
        "lora_value_count": lora_value_count,
        "merged_leaf_count": merged_leaf_count,
        "merged_value_count": merged_value_count,
    }
    (out_dir / "merge_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        f"Merged {lora_leaf_count} LoRA leaves ({lora_value_count:,} values) "
        f"into {merged_leaf_count} total leaves -> {out_dir}"
    )
    if copied_assets_from is None:
        print("Warning: no assets directory was copied. serve_policy may fail to load norm stats.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        required=True,
        type=pathlib.Path,
        help="Base checkpoint root or params dir, e.g. models/pi05_droid or models/pi05_droid/params.",
    )
    parser.add_argument(
        "--lora-dir",
        required=True,
        type=pathlib.Path,
        help="Adapter-only LoRA root or params dir produced by export_lora_checkpoint.py.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=pathlib.Path,
        help="Output full checkpoint root. serve_policy should use this path.",
    )
    parser.add_argument(
        "--assets-dir",
        type=pathlib.Path,
        default=None,
        help="Optional explicit assets directory to copy into out-dir/assets.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace out-dir if it already exists.")
    args = parser.parse_args()

    merge_lora(
        args.base_dir,
        args.lora_dir,
        args.out_dir,
        assets_dir=args.assets_dir,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
