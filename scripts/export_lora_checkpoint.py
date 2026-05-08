"""Export only LoRA parameters from a full OpenPI JAX checkpoint.

Example:
    python scripts/export_lora_checkpoint.py \
        --checkpoint-dir checkpoints/pi05_fr3_lora/test_train_2/199999 \
        --out-dir checkpoints_lora/fr3_lora_199999
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


def _count_leaves(tree: Any) -> tuple[int, int]:
    leaves = traverse_util.flatten_dict(tree)
    total_values = 0
    for value in leaves.values():
        if hasattr(value, "size"):
            total_values += int(value.size)
    return len(leaves), total_values


def _save_params(params_dir: pathlib.Path, params: dict) -> None:
    params_dir = params_dir.expanduser().resolve()
    params_dir.parent.mkdir(parents=True, exist_ok=True)
    with ocp.PyTreeCheckpointer() as ckptr:
        ckptr.save(params_dir, args=ocp.args.PyTreeSave({"params": params}))


def export_lora(checkpoint_dir: pathlib.Path, out_dir: pathlib.Path, *, overwrite: bool, copy_assets: bool) -> None:
    checkpoint_dir = checkpoint_dir.expanduser()
    out_dir = out_dir.expanduser().resolve()
    params_dir = _resolve_params_dir(checkpoint_dir).resolve()
    root_dir = _checkpoint_root(checkpoint_dir).resolve()

    if not params_dir.exists():
        raise FileNotFoundError(f"Params directory not found: {params_dir}")
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{out_dir} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(out_dir)

    params = _model.restore_params(params_dir, restore_type=np.ndarray)
    flat = traverse_util.flatten_dict(params, sep="/")
    lora_flat = {key: value for key, value in flat.items() if "lora" in key.lower()}
    if not lora_flat:
        raise ValueError(f"No LoRA parameters found in {params_dir}")

    lora_params = traverse_util.unflatten_dict(lora_flat, sep="/")
    out_params_dir = out_dir / "params"
    _save_params(out_params_dir, lora_params)

    if copy_assets and (root_dir / "assets").is_dir():
        shutil.copytree(root_dir / "assets", out_dir / "assets")

    leaf_count, value_count = _count_leaves(lora_params)
    metadata = {
        "format": "openpi_lora_params",
        "source_checkpoint": str(root_dir),
        "source_params": str(params_dir),
        "filter": "*lora*",
        "leaf_count": leaf_count,
        "value_count": value_count,
    }
    (out_dir / "lora_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Exported {leaf_count} LoRA leaves ({value_count:,} values) -> {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        type=pathlib.Path,
        help="Full checkpoint root, e.g. checkpoints/pi05_fr3_lora/exp/199999. A params/ dir is also accepted.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=pathlib.Path,
        help="Output adapter directory. The script writes out-dir/params and optional out-dir/assets.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace out-dir if it already exists.")
    parser.add_argument(
        "--no-copy-assets",
        action="store_true",
        help="Do not copy checkpoint assets/norm stats into the LoRA adapter directory.",
    )
    args = parser.parse_args()

    export_lora(
        args.checkpoint_dir,
        args.out_dir,
        overwrite=args.overwrite,
        copy_assets=not args.no_copy_assets,
    )


if __name__ == "__main__":
    main()
