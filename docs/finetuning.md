# FR3 LoRA Fine-Tuning

This document corresponds to the `pi05_fr3_lora` config in `src/openpi/training/config.py`. It uses the local `models/pi05_droid` checkpoint as the base checkpoint and fine-tunes LoRA weights on an FR3 LeRobot v2.1 dataset.

## 1. Required directories

Prepare these directories from the repo root:

```text
models/pi05_droid/
  params/
  assets/

datasets/fr3_pick_up_cup/
  data/
  videos/
  meta/
```

The DROID normalization statistics in `models/pi05_droid/assets` are reused. The current FR3 data config repacks FR3 fields into the DROID/pi0.5 fields expected by the policy.

The dataset directory name must match the `repo_id` in the training config. The current config uses:

```python
repo_id="fr3_pick_up_cup"
```

So the local path should be:

```text
datasets/fr3_pick_up_cup
```

If you use another task name, such as `datasets/fr3_pick_block`, either update `repo_id` in `src/openpi/training/config.py` or rename the dataset directory.

## 2. Start training

```bash
cd /path/to/openpi
conda activate openpi-dev

export HF_LEROBOT_HOME="$(pwd)/datasets"
export HF_HUB_OFFLINE=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

python scripts/train.py pi05_fr3_lora --exp-name <run_name> --overwrite
```

Training outputs are written to:

```text
checkpoints/pi05_fr3_lora/<run_name>/<step>/
```

For example:

```text
checkpoints/pi05_fr3_lora/test_train_2/199999/
```

## 3. Current config highlights

`pi05_fr3_lora` currently uses:

- `pi05=True`
- `action_dim=32`
- `action_horizon=16`
- `paligemma_variant="gemma_2b_lora"`
- `action_expert_variant="gemma_300m_lora"`
- `weight_loader="models/pi05_droid/params"`
- `assets_dir="models/pi05_droid/assets"`
- `ema_decay=None`
- `wandb_enabled=False`

LoRA training must keep the matching `freeze_filter`; otherwise training may become full-parameter training or the parameter set may not match. The current config already uses `get_freeze_filter()`.

The FR3 joint 7 offset is applied in both the data transform and the inference client:

```text
joint_position[6] -= pi / 4
```

Training and real-robot inference should use the same code path. Do not change only one side.

## 4. Common tuning points

Edit the `pi05_fr3_lora` config in `src/openpi/training/config.py`:

- `num_train_steps`: total training steps.
- `batch_size`: lower this first when running out of GPU memory.
- `save_interval`: checkpoint save interval.
- `keep_period`: periodic checkpoint retention interval.
- `wandb_enabled`: enable only after logging is configured.

For small datasets, pay attention to real-robot behavior and validation signals, not only loss. If the model overfits, reduce training steps, add more diverse demonstrations, or cover more initial object poses for the same task.

## 5. Resume training

Continue the same experiment:

```bash
python scripts/train.py pi05_fr3_lora --exp-name <run_name> --resume
```

Do not pass `--resume` and `--overwrite` at the same time.

## 6. Serve the fine-tuned policy

The full checkpoint produced by training can be passed directly as `--policy.dir`:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.70

python scripts/serve_policy.py \
  --port 8010 \
  policy:checkpoint \
  --policy.config=pi05_fr3_lora \
  --policy.dir=checkpoints/pi05_fr3_lora/<run_name>/<step>
```

Then start `inference_client.py` as described in [real_robot_deployment.md](real_robot_deployment.md).

If you want to save or share only the LoRA adapter, or merge an adapter back into the base checkpoint, see [lora_merge_and_export.md](lora_merge_and_export.md).

## 7. Pre-training checks

```bash
python -c "import openpi, cv2, pyarrow; print('env ok')"
test -d models/pi05_droid/params
test -d models/pi05_droid/assets
test -d datasets/fr3_pick_up_cup/meta
```

If LeRobot cannot find the dataset, `HF_LEROBOT_HOME` probably does not point to the repo-local `datasets` directory, or `repo_id` does not match the dataset directory name.
