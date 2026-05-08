# LoRA 合并与拆分

本项目提供两个脚本处理 OpenPI/JAX checkpoint 里的 LoRA 参数：

- `scripts/export_lora_checkpoint.py`：从完整 checkpoint 中拆出 LoRA adapter-only checkpoint。
- `scripts/merge_lora_checkpoint.py`：把 adapter-only LoRA 合回 base checkpoint，生成可直接部署的完整 checkpoint。

这里的“拆分”指只导出参数名里包含 `lora` 的叶子节点；“合并”指用这些 LoRA 参数覆盖/补入 base checkpoint 的参数树。

## 1. 什么时候需要拆分

训练产物通常是完整 checkpoint，可以直接用于：

```bash
python scripts/serve_policy.py \
  --port 8010 \
  policy:checkpoint \
  --policy.config=pi05_fr3_lora \
  --policy.dir=checkpoints/pi05_fr3_lora/<run_name>/<step>
```

如果你想单独保存、传输或归档 LoRA adapter，就把完整 checkpoint 拆成 adapter-only：

```bash
python scripts/export_lora_checkpoint.py \
  --checkpoint-dir checkpoints/pi05_fr3_lora/test_train_2/199999 \
  --out-dir checkpoints_lora/fr3_lora_199999 \
  --overwrite
```

`--checkpoint-dir` 可以传 checkpoint 根目录，也可以直接传 `params/` 目录。

输出目录：

```text
checkpoints_lora/fr3_lora_199999/
  params/
  assets/                # 默认会复制，如果源 checkpoint 有 assets
  lora_metadata.json
```

如果只想导出 LoRA 参数，不复制 assets：

```bash
python scripts/export_lora_checkpoint.py \
  --checkpoint-dir checkpoints/pi05_fr3_lora/test_train_2/199999 \
  --out-dir checkpoints_lora/fr3_lora_199999 \
  --no-copy-assets \
  --overwrite
```

## 2. 什么时候需要合并

adapter-only LoRA 不能单独作为完整 OpenPI checkpoint 部署。部署前需要和训练时使用的 base checkpoint 合并，例如本项目的 FR3 LoRA 使用：

```text
models/pi05_droid/
  params/
  assets/
```

合并命令：

```bash
python scripts/merge_lora_checkpoint.py \
  --base-dir models/pi05_droid \
  --lora-dir checkpoints_lora/fr3_lora_199999 \
  --out-dir checkpoints/pi05_fr3_lora_merged/199999 \
  --overwrite
```

`--base-dir` 和 `--lora-dir` 都可以传 checkpoint 根目录，也可以传各自的 `params/` 目录。

输出目录：

```text
checkpoints/pi05_fr3_lora_merged/199999/
  params/
  assets/
  merge_metadata.json
```

如果需要明确指定 assets 来源：

```bash
python scripts/merge_lora_checkpoint.py \
  --base-dir models/pi05_droid \
  --lora-dir checkpoints_lora/fr3_lora_199999 \
  --out-dir checkpoints/pi05_fr3_lora_merged/199999 \
  --assets-dir models/pi05_droid/assets \
  --overwrite
```

脚本会优先使用 `--assets-dir`，其次使用 LoRA 目录里的 `assets/`，最后使用 base 目录里的 `assets/`。

## 3. 部署合并后的 checkpoint

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.70

python scripts/serve_policy.py \
  --port 8010 \
  policy:checkpoint \
  --policy.config=pi05_fr3_lora \
  --policy.dir=checkpoints/pi05_fr3_lora_merged/199999
```

然后按 `docs/真机部署.md` 启动 `inference_client.py`。

## 4. 注意事项

base checkpoint 必须和 LoRA 训练时加载的 base 对齐。当前 `pi05_fr3_lora` 配置使用的是：

```text
models/pi05_droid/params
```

模型配置也要一致，尤其是：

- `pi05=True`
- `action_dim=32`
- `action_horizon=16`
- `paligemma_variant="gemma_2b_lora"`
- `action_expert_variant="gemma_300m_lora"`

`merge_lora_checkpoint.py` 会拒绝包含非 LoRA 参数的 adapter checkpoint。如果报：

```text
LoRA checkpoint contains non-LoRA keys
```

说明传入的 `--lora-dir` 不是 adapter-only 目录，先用 `export_lora_checkpoint.py` 拆一次。

如果合并后启动策略服务时报 norm stats 或 assets 相关错误，检查输出目录是否有：

```text
checkpoints/pi05_fr3_lora_merged/199999/assets/
```

没有的话重新合并并显式传：

```bash
--assets-dir models/pi05_droid/assets
```

## 5. 推荐目录习惯

```text
checkpoints/pi05_fr3_lora/<run_name>/<step>/     # 训练产出的完整 checkpoint
checkpoints_lora/fr3_lora_<step>/                # 拆出来的 adapter-only LoRA
checkpoints/pi05_fr3_lora_merged/<step>/         # 合并后用于部署的完整 checkpoint
```

这样训练产物、轻量 LoRA 包和部署产物互不覆盖，回滚也更清楚。
