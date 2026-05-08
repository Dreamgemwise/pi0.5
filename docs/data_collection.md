# Data Collection

This document describes data collection on FR3 for `pi05_fr3_lora` fine-tuning. The data is saved in LeRobot v2.1 format with DROID-aligned field names, so it can go through the OpenPI DROID/pi0.5 input and output transforms directly.

## 1. Data flow

The collection RT machine reads robot state through Polymetis and publishes 100 Hz robot state:

```text
Franka/FrankaTeleop/Polymetis
  50051: robot state gRPC
  50053: gripper state gRPC
        |
        v
x86 collection RT machine readonly_state_publisher.py
  5555: RobotState PUB  --->  collection PC
```

The collection PC:

- Subscribes to the latest robot state.
- Reads two ZED UVC single-eye RGB streams.
- Saves a LeRobot v2.1 dataset at 15 Hz.
- Uses terminal hotkeys to start, save, and discard episodes.

Pure data collection does not require a policy server or OpenPI inference. Do not start `robot_server.py` on the collection RT machine for this workflow, because collection only needs read-only state and should not open the action port or send robot commands.

Important: this workflow must use `examples/fr3_realtime/readonly_state_publisher.py`. It depends on Polymetis gRPC services, with the relevant ports provided by the FrankaTeleop/Polymetis environment. Polymetis only runs on x86, so the data-collection RT machine must be an x86 host. The ARM/C++ libfranka path is not used for this data-collection state bridge.

## 2. Start the x86 collection RT machine

First, start Polymetis/FrankaTeleop on the x86 RT host so the robot state and gripper state services are available. Use the `polymetis` environment created for FrankaTeleop:

```bash
conda activate polymetis
```

Then, on the same x86 RT host, start the read-only state publisher from `examples/fr3_realtime`:

```bash
cd examples/fr3_realtime

python readonly_state_publisher.py \
  --polymetis-host 127.0.0.1 \
  --polymetis-port 50051 \
  --gripper-port 50053 \
  --bind 0.0.0.0
```

If no gripper state service is available, disable gripper reads:

```bash
python readonly_state_publisher.py \
  --polymetis-host 127.0.0.1 \
  --polymetis-port 50051 \
  --gripper-port 0 \
  --bind 0.0.0.0
```

`readonly_state_publisher.py` only reads Polymetis gRPC state and publishes `RobotState` on `tcp://<bind>:5555`. It does not import `polymetis.RobotInterface`, does not enter impedance control, and does not open an action socket.

## 3. Check cameras and network

On the collection PC:

```bash
conda activate openpi-dev

ping <RT-PC-IP>
v4l2-ctl --list-devices
ls /dev/video*
```

`<RT-PC-IP>` is the IP of the x86 RT host running `readonly_state_publisher.py`.

Preview the front or wrist camera:

```bash
python scripts/zed_preview.py --camera /dev/video0 --camera-eye LEFT --camera-fourcc YUYV
python scripts/zed_preview.py --camera /dev/video2 --camera-eye LEFT --camera-fourcc YUYV
```

If the wrist camera is upside down, add `--flip-wrist` to the collection command. If the front camera is upside down, add `--flip-front`.

## 4. Pure data collection

Run this from the repo root on the collection PC:

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
s     = stop and save the current episode
r     = discard the current episode
q     = quit
```

During collection, hold the FR3 guiding button at the end effector and demonstrate the task, or drive the robot with another expert controller. The script computes joint velocity from adjacent joint positions and uses that as the training action.

Common camera options:

```bash
# Default HD720: 2560x720@30fps, roughly 1280x720 per eye.
python examples/fr3_realtime/data_collector.py ... --camera-mode HD720

# Higher resolution, but only 15fps: roughly 2208x1242 per eye.
python examples/fr3_realtime/data_collector.py ... --camera-mode HD2K

# Use the right eye.
python examples/fr3_realtime/data_collector.py ... --camera-eye RIGHT
```

## 5. Collection-path limitations

The FrankaTeleop/Polymetis collection path in this document uses only `readonly_state_publisher.py`. Do not replace it with `robot_server.py` on the collection RT machine, and do not use the ARM/C++ libfranka path as this collection state bridge.

`inference_client.py --record-dir` is a deployment-time logging path that requires an execution-side `robot_server` to receive action chunks. It is not the Polymetis pure data-collection workflow described here.

## 6. Dataset layout

After saving, the dataset looks like:

```text
datasets/fr3_pick_up_cup/
  data/chunk-000/episode_000000.parquet
  videos/chunk-000/observation.images.exterior_image_1_left/episode_000000.mp4
  videos/chunk-000/observation.images.wrist_image_left/episode_000000.mp4
  meta/info.json
  meta/tasks.jsonl
  meta/episodes.jsonl
  meta/episodes_stats.jsonl
```

Important fields:

- `observation.state`: 8 dimensions, `joint_position(7) + gripper_position(1)`.
- `action`: 8 dimensions, `joint_velocity(7) + gripper_position(1)`.
- `observation.images.exterior_image_1_left`: external ZED single-eye RGB.
- `observation.images.wrist_image_left`: wrist ZED single-eye RGB.

The default collection frequency is `15Hz`, matching DROID/pi0.5.

If `episodes_stats.jsonl` is missing or needs to be regenerated:

```bash
python scripts/gen_episodes_stats.py --root datasets/fr3_pick_up_cup --overwrite
```

## 7. Quality checks

Each episode should contain one clear task. Avoid mixing failures, long idle waits, or scene resets into the same episode. Failed demonstrations can be saved separately, but decide explicitly whether they should be included before training.

Check:

- The prompt matches the demonstrated behavior.
- The front and wrist images use the intended eye and are not upside down.
- Robot motion is reasonably continuous, with minimal idle pauses.
- The dataset directory matches the `repo_id` in `src/openpi/training/config.py`. For example, with `HF_LEROBOT_HOME="$(pwd)/datasets"`, `repo_id="fr3_pick_up_cup"` reads `datasets/fr3_pick_up_cup`.
