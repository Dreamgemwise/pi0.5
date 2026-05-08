# LoRA Merge and Export

This project provides two scripts for handling LoRA parameters in OpenPI/JAX checkpoints:

- `scripts/export_lora_checkpoint.py`: exports an adapter-only LoRA checkpoint from a full checkpoint.
- `scripts/merge_lora_checkpoint.py`: merges an adapter-only LoRA checkpoint back into a base checkpoint and produces a full checkpoint that can be deployed directly.

In this document, "export" means saving only parameter leaves whose names contain `lora`; "merge" means inserting those LoRA parameters into the base checkpoint parameter tree.

## 1. When to export

Training usually produces a full checkpoint that can be served directly:

```bash
python scripts/serve_policy.py \
  --port 8010 \
  policy:checkpoint \
  --policy.config=pi05_fr3_lora \
  --policy.dir=checkpoints/pi05_fr3_lora/<run_name>/<step>
```

If you want to save, transfer, or archive only the LoRA adapter, export the full checkpoint into an adapter-only checkpoint:

```bash
python scripts/export_lora_checkpoint.py \
  --checkpoint-dir checkpoints/pi05_fr3_lora/test_train_2/199999 \
  --out-dir checkpoints_lora/fr3_lora_199999 \
  --overwrite
```

`--checkpoint-dir` may point to the checkpoint root or directly to its `params/` directory.

Output layout:

```text
checkpoints_lora/fr3_lora_199999/
  params/
  assets/                # copied by default if the source checkpoint has assets
  lora_metadata.json
```

To export only LoRA parameters and skip assets:

```bash
python scripts/export_lora_checkpoint.py \
  --checkpoint-dir checkpoints/pi05_fr3_lora/test_train_2/199999 \
  --out-dir checkpoints_lora/fr3_lora_199999 \
  --no-copy-assets \
  --overwrite
```

## 2. When to merge

An adapter-only LoRA checkpoint cannot be deployed by itself as a full OpenPI checkpoint. Before deployment, merge it with the same base checkpoint used during training. The FR3 LoRA config in this workspace uses:

```text
models/pi05_droid/
  params/
  assets/
```

Merge command:

```bash
python scripts/merge_lora_checkpoint.py \
  --base-dir models/pi05_droid \
  --lora-dir checkpoints_lora/fr3_lora_199999 \
  --out-dir checkpoints/pi05_fr3_lora_merged/199999 \
  --overwrite
```

Both `--base-dir` and `--lora-dir` may point to checkpoint roots or directly to `params/` directories.

Output layout:

```text
checkpoints/pi05_fr3_lora_merged/199999/
  params/
  assets/
  merge_metadata.json
```

To specify the assets directory explicitly:

```bash
python scripts/merge_lora_checkpoint.py \
  --base-dir models/pi05_droid \
  --lora-dir checkpoints_lora/fr3_lora_199999 \
  --out-dir checkpoints/pi05_fr3_lora_merged/199999 \
  --assets-dir models/pi05_droid/assets \
  --overwrite
```

The script tries assets in this order: `--assets-dir`, then `assets/` in the LoRA directory, then `assets/` in the base directory.

## 3. Serve the merged checkpoint

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.70

python scripts/serve_policy.py \
  --port 8010 \
  policy:checkpoint \
  --policy.config=pi05_fr3_lora \
  --policy.dir=checkpoints/pi05_fr3_lora_merged/199999
```

Then start `inference_client.py` as described in [real_robot_deployment.md](real_robot_deployment.md).

## 4. Notes

The base checkpoint must match the base used for LoRA training. The current `pi05_fr3_lora` config uses:

```text
models/pi05_droid/params
```

The model config should also match, especially:

- `pi05=True`
- `action_dim=32`
- `action_horizon=16`
- `paligemma_variant="gemma_2b_lora"`
- `action_expert_variant="gemma_300m_lora"`

`merge_lora_checkpoint.py` refuses adapter checkpoints that contain non-LoRA parameters. If you see:

```text
LoRA checkpoint contains non-LoRA keys
```

then `--lora-dir` is not an adapter-only directory. Run `export_lora_checkpoint.py` first.

If the merged checkpoint fails to start because of norm stats or assets, check that the output contains:

```text
checkpoints/pi05_fr3_lora_merged/199999/assets/
```

If it does not, merge again with:

```bash
--assets-dir models/pi05_droid/assets
```

## 5. Recommended directory convention

```text
checkpoints/pi05_fr3_lora/<run_name>/<step>/     # full checkpoint produced by training
checkpoints_lora/fr3_lora_<step>/                # exported adapter-only LoRA
checkpoints/pi05_fr3_lora_merged/<step>/         # full checkpoint for deployment
```

This keeps training outputs, lightweight LoRA adapters, and deployment checkpoints separate.
