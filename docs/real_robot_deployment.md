# Real-Robot Deployment

Deployment has two sides:

- Execution side / RT control machine: connects directly to the Franka robot, moves home, publishes robot state, and executes action chunks.
- Inference side / GPU PC: starts the policy server, reads ZED images and robot state, and sends policy outputs back to the RT control machine.

There are two execution-side paths:

- x86_64/franky-control: see `examples/fr3_realtime/rt_machine.md`.
- ARM/C++ libfranka: see `examples/fr3_realtime_libfranka/rt_machine.md`.

Commands below are run from the repo root unless stated otherwise.

## 1. Pre-flight checks

Confirm:

- Emergency stop, Franka Desk, collision thresholds, and workspace are checked.
- The RT control machine can ping the robot.
- The inference PC can ping the RT control machine.
- Both ZED cameras can be opened by OpenCV.
- `--prompt` matches the task semantics used during training.
- When testing a new checkpoint for the first time, keep `--request-hz` at `3.0`, clear the table, and start with conservative robot-side motion limits.

## 2. Execution side: move home

x86_64/franky-control:

```bash
cd examples/fr3_realtime
conda activate fr3-realtime

python go_home.py --robot-ip 172.16.0.2 --time-to-go 8.0
```

ARM/C++ libfranka:

```bash
cd examples/fr3_realtime_libfranka
conda activate fr3-realtime-libfranka

./build/go_home --robot-ip 172.16.0.2 --time-to-go 8.0
```

## 3. Execution side: start robot server

x86_64/franky-control:

```bash
cd examples/fr3_realtime
conda activate fr3-realtime

python robot_server.py --robot-ip 172.16.0.2 --bind 0.0.0.0
```

ARM/C++ libfranka:

```bash
cd examples/fr3_realtime_libfranka
conda activate fr3-realtime-libfranka

./build/robot_server --robot-ip 172.16.0.2 --bind 0.0.0.0
```

If no gripper is connected, the C++ path can use:

```bash
--no-gripper
```

The inference-side `--robot-ip` should be the LAN IP of the RT control machine, not the Franka robot IP `172.16.0.2`.

## 4. Inference side: start the policy server

On the GPU PC:

```bash
conda activate openpi-dev

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.70

python ./scripts/serve_policy.py \
  --port 8010 \
  policy:checkpoint \
  --policy.config=pi05_fr3_lora \
  --policy.dir=checkpoints/pi05_fr3_lora/test_train_2/199999
```

If you already merged a LoRA adapter into a full checkpoint, point `--policy.dir` to the merged checkpoint:

```bash
--policy.dir=checkpoints/pi05_fr3_lora_merged/199999
```

For LoRA export and merge, see [lora_merge_and_export.md](lora_merge_and_export.md).

## 5. Inference side: start inference client

Open another terminal:

```bash
conda activate openpi-dev

python ./examples/fr3_realtime/inference_client.py \
  --robot-ip 192.168.1.19 \
  --policy-host 127.0.0.1 \
  --policy-port 8010 \
  --prompt "pick up the cup" \
  --front-camera /dev/video0 \
  --wrist-camera /dev/video2 \
  --flip-wrist \
  --request-hz 3.0 \
  --display
```

Arguments:

- `--robot-ip`: RT control machine IP.
- `--policy-host/--policy-port`: policy server address.
- `--front-camera`: external ZED UVC device.
- `--wrist-camera`: wrist ZED UVC device.
- `--flip-front/--flip-wrist`: use when an image needs 180-degree rotation.
- `--request-hz`: policy action chunk request frequency, default `3Hz`.

To collect data while running inference, add:

```bash
--record-dir ./datasets/fr3_pick_up_cup_dagger
```

Hotkeys:

```text
Enter = start an episode
s     = stop and save the current episode
r     = discard the current episode
q     = quit
```

## 6. Common issues

Policy server cannot be reached: make sure `serve_policy.py` is still running and `--policy-port` matches. If the policy server runs on another machine, set `--policy-host` to that machine's IP and make sure the port is reachable.

Waiting forever for state or camera: make sure `robot_server` is running on the RT control machine, and that inference-side `--robot-ip` is the RT control machine IP. Also check whether `/dev/video*` numbering changed.

Image orientation is wrong: preview with `scripts/zed_preview.py` to check `LEFT/RIGHT` and orientation, then adjust `--camera-eye`, `--flip-front`, and `--flip-wrist` in `inference_client.py`.

Actions are delayed: lowering `--request-hz` does not make inference faster; it only sends fewer requests. Check GPU inference time, network latency, and camera capture stalls first. The log fields `infer=...ms` and `state_age=...ms` are useful diagnostics.

Actions are too aggressive: stop the robot and check the checkpoint, prompt, camera views, and joint 7 offset. On the C++ libfranka path, you can also lower `--dynamics-factor` or reduce `--ema-alpha`.
