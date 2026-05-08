# pi0.5 on Franka Research 3

This repository is a local OpenPI workspace for running and fine-tuning pi0.5-style policies on a Franka Research 3 robot. It includes FR3 real-time control bridges, ZED UVC camera capture, LeRobot-format data collection, LoRA fine-tuning utilities, and checkpoint export/merge scripts.

The English documentation lives in [docs/](docs/README.md). Chinese documentation lives in [docs_Chinese/](docs_Chinese/).

## Real-Robot Demo

Replace the placeholder path below with the final demo video when it is ready.

*TODO: add FR3 real-robot demo video.*

If the platform does not render embedded videos, use a plain link:

[FR3 real-robot demo video](assets/fr3_real_robot_demo.mp4)

## Hardware Configuration

- Robot arm: Franka Research 3.
- End effector: Franka Hand with a UMI gripper.
- Wrist camera: ZED Mini.
- Third-person camera: ZED 2i.

In the data collection and inference scripts, the wrist camera is passed as `--wrist-camera` and saved as `observation.images.wrist_image_left`. The third-person ZED 2i is passed as `--front-camera` and saved as `observation.images.exterior_image_1_left`.

## What Is Included

- FR3 real-time runtime through either Python `franky-control` or C++ `libfranka`.
- ZED camera capture through UVC/OpenCV, without requiring ZED SDK or `pyzed`.
- LeRobot v2.1 data collection aligned with DROID/pi0.5 field conventions.
- LoRA fine-tuning config for `pi05_fr3_lora`.
- Scripts to export adapter-only LoRA checkpoints and merge them back into a full deployable checkpoint.
- Remote policy serving and robot-side inference client.

## Repository Layout

```text
docs/                              English documentation
docs_Chinese/                      Chinese documentation
examples/fr3_realtime/             x86_64 Python/franky-control FR3 runtime
examples/fr3_realtime_libfranka/   ARM or C++ libfranka FR3 runtime
scripts/serve_policy.py            OpenPI policy server
scripts/zed_preview.py             ZED UVC preview helper
scripts/export_lora_checkpoint.py  Export adapter-only LoRA checkpoint
scripts/merge_lora_checkpoint.py   Merge LoRA adapter into a base checkpoint
src/openpi/training/config.py      Training and data configs
```

## Quick Start

Create the main OpenPI environment from the repository root:

```bash
conda env create -f environment.yml
conda activate openpi-dev
```

Check the basic imports:

```bash
python -c "import openpi, openpi_client, cv2, zmq, pyarrow; print('openpi env ok')"
```

For ZED cameras, this project uses UVC/OpenCV and reads one RGB eye from the side-by-side stereo frame. No ZED SDK is required.

```bash
sudo apt install -y v4l-utils
v4l2-ctl --list-devices
python scripts/zed_preview.py --camera /dev/video0 --camera-eye LEFT
```

## FR3 Runtime

Choose one runtime path for the robot control machine:

- x86_64 RT machine: [examples/fr3_realtime/rt_machine.md](examples/fr3_realtime/rt_machine.md)
- ARM or C++ libfranka RT machine: [examples/fr3_realtime_libfranka/rt_machine.md](examples/fr3_realtime_libfranka/rt_machine.md)

The Python `franky-control` path does not implement action smoothing and is not recommended for real-robot policy execution. Prefer the C++ `libfranka` path for deployment, where velocity EMA and libfranka-side motion limits are available.

The runtime publishes robot state over ZMQ on port `5555` and receives action chunks on port `5556`.

## Data Collection

Pure data collection uses an x86 RT host running FrankaTeleop/Polymetis and the read-only state bridge. Do not use `robot_server.py` for this workflow.

On the x86 RT host:

```bash
conda activate polymetis
cd examples/fr3_realtime
python readonly_state_publisher.py \
  --polymetis-host 127.0.0.1 \
  --polymetis-port 50051 \
  --gripper-port 50053 \
  --bind 0.0.0.0
```

Then run data collection on the collection PC:

```bash
python examples/fr3_realtime/data_collector.py \
  --robot-ip <RT-PC-IP> \
  --front-camera /dev/video0 \
  --wrist-camera /dev/video2 \
  --prompt "pick up the cup" \
  --record-dir ./datasets/fr3_pick_up_cup \
  --flip-wrist \
  --display
```

Hotkeys:

```text
Enter = start an episode
s     = stop and save
r     = discard
q     = quit
```

See [docs/data_collection.md](docs/data_collection.md) for the full collection workflow.

## Fine-Tuning

The default FR3 LoRA config is `pi05_fr3_lora`. It expects a local pi0.5 DROID base checkpoint:

```text
models/pi05_droid/
  params/
  assets/
```

Train with:

```bash
export HF_LEROBOT_HOME="$(pwd)/datasets"
export HF_HUB_OFFLINE=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

python scripts/train.py pi05_fr3_lora --exp-name <run_name> --overwrite
```

See [docs/finetuning.md](docs/finetuning.md) for config details, resume behavior, and deployment notes.

## LoRA Export and Merge

Export only LoRA parameters from a full checkpoint:

```bash
python scripts/export_lora_checkpoint.py \
  --checkpoint-dir checkpoints/pi05_fr3_lora/<run_name>/<step> \
  --out-dir checkpoints_lora/fr3_lora_<step> \
  --overwrite
```

Merge an adapter-only LoRA checkpoint back into the base checkpoint:

```bash
python scripts/merge_lora_checkpoint.py \
  --base-dir models/pi05_droid \
  --lora-dir checkpoints_lora/fr3_lora_<step> \
  --out-dir checkpoints/pi05_fr3_lora_merged/<step> \
  --overwrite
```

See [docs/lora_merge_and_export.md](docs/lora_merge_and_export.md) for the full workflow.

## Real-Robot Deployment

Start the policy server on the GPU PC:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.70

python scripts/serve_policy.py \
  --port 8010 \
  policy:checkpoint \
  --policy.config=pi05_fr3_lora \
  --policy.dir=checkpoints/pi05_fr3_lora/<run_name>/<step>
```

Then start the inference client:

```bash
python examples/fr3_realtime/inference_client.py \
  --robot-ip <RT-PC-IP> \
  --policy-host 127.0.0.1 \
  --policy-port 8010 \
  --prompt "pick up the cup" \
  --front-camera /dev/video0 \
  --wrist-camera /dev/video2 \
  --flip-wrist \
  --request-hz 3.0 \
  --display
```

See [docs/real_robot_deployment.md](docs/real_robot_deployment.md) before running on hardware.

## Documentation

- [docs/environment_setup.md](docs/environment_setup.md)
- [docs/data_collection.md](docs/data_collection.md)
- [docs/finetuning.md](docs/finetuning.md)
- [docs/real_robot_deployment.md](docs/real_robot_deployment.md)
- [docs/lora_merge_and_export.md](docs/lora_merge_and_export.md)
- [docs/remote_inference.md](docs/remote_inference.md)
- [docs/norm_stats.md](docs/norm_stats.md)
- [docs/docker.md](docs/docker.md)

## Safety

Real-robot deployment can move hardware quickly and unexpectedly. Before testing a new checkpoint, clear the workspace, verify emergency stop access, confirm camera views and prompts, start with conservative robot-side limits, and keep the first runs short.
