# Environment Setup

This FR3 workflow usually uses two kinds of machines:

- Inference/training/collection PC: has a GPU and runs OpenPI, the policy server, camera capture, and data collection.
- RT machine: connects directly to the Franka robot. For data collection it only publishes read-only state; for real-robot deployment it runs `robot_server`, `go_home`, and action execution.

You can run both sides on one machine, but for real-robot deployment it is better to split them. This keeps GPU inference and camera IO from interfering with real-time robot control.

Distinguish the data-collection RT side from the deployment RT side:

- The data-collection RT side must be an x86 host. It uses the `polymetis` environment created for FrankaTeleop and runs `examples/fr3_realtime/readonly_state_publisher.py`. This path only reads Polymetis state and does not open an action port.
- The real-robot deployment RT side runs `robot_server`, and can use either the x86/franky-control path or the ARM/C++ libfranka path.

## 1. Inference/training PC: OpenPI environment

Create the main environment from the repo root:

```bash
conda env create -f environment.yml
conda activate openpi-dev
```

`environment.yml` installs OpenPI, `openpi-client`, JAX training dependencies, OpenCV, ZMQ, msgpack, pyarrow, and related tools. Training, policy serving, ZED capture, and LoRA checkpoint tools are expected to run in this environment.

Quick check:

```bash
python -c "import openpi, openpi_client, cv2, zmq, pyarrow; print('openpi env ok')"
```

This workspace expects the local `models/pi05_droid` checkpoint to be the base checkpoint for FR3 LoRA fine-tuning. It should contain at least:

```text
models/pi05_droid/
  params/
  assets/
```

`assets/` contains the DROID normalization statistics reused by FR3 fine-tuning and deployment.

## 2. x86_64 RT control machine: franky-control path

Use this path if the RT control machine is x86_64 and you are using the Python/franky-control control stack in `examples/fr3_realtime`:

```bash
cd examples/fr3_realtime
conda env create -f environment.yml
conda activate fr3-realtime
```

Check the machine architecture:

```bash
uname -m
```

This should print `x86_64`. If the machine is ARM/aarch64, do not use this environment. Use the C++ libfranka path below.

If this x86 machine is used for data collection, do not use the `fr3-realtime` environment and do not start `robot_server.py`. Use the Polymetis environment created by FrankaTeleop:

```bash
conda activate polymetis
cd examples/fr3_realtime
python readonly_state_publisher.py \
  --polymetis-host 127.0.0.1 \
  --polymetis-port 50051 \
  --gripper-port 50053 \
  --bind 0.0.0.0
```

Polymetis only runs on x86, so the data-collection RT host cannot be ARM.

If this x86 machine is used for real-robot deployment, see:

```text
examples/fr3_realtime/rt_machine.md
```

## 3. ARM RT control machine: C++ libfranka path

Use this path if the RT control machine is ARM/aarch64, or if you want to control the FR3 directly through C++ `libfranka`:

```bash
cd examples/fr3_realtime_libfranka
mamba env create -f environment.yml
conda activate fr3-realtime-libfranka
```

Then follow the robot-side instructions to build C++ `libfranka` and the bridge:

```text
examples/fr3_realtime_libfranka/rt_machine.md
```

This path uses the same ZMQ/msgpack protocol: the RT machine publishes state on `5555` and receives action chunks on `5556`. It is used for real-robot deployment, not for the Polymetis/FrankaTeleop data-collection bridge.

## 4. ZED cameras: UVC/OpenCV path

The current code uses single-eye RGB images from ZED cameras through UVC/OpenCV. It does not require the ZED SDK, `pyzed`, or system CUDA. After plugging in a ZED camera, inspect the video devices:

```bash
sudo apt update
sudo apt install -y v4l-utils

v4l2-ctl --list-devices
ls /dev/video*
v4l2-ctl --list-formats-ext -d /dev/video0
```

ZED UVC frames are side-by-side stereo frames. The code assumes `RIGHT | LEFT`. By default it uses the left eye and writes DROID-style fields:

- `observation.images.exterior_image_1_left`
- `observation.images.wrist_image_left`

Preview one ZED camera:

```bash
python scripts/zed_preview.py --camera /dev/video0 --camera-eye LEFT --camera-fourcc YUYV
```

Preview at the highest resolution:

```bash
python scripts/zed_preview.py --camera /dev/video0 --camera-mode HD2K --camera-eye LEFT
```

`HD2K` is `4416x1242@15fps`, roughly `2208x1242` per eye. For real-time inference, start with the default `HD720` mode and increase resolution only after the pipeline is stable.

If the wrong eye is selected, try the right eye:

```bash
python scripts/zed_preview.py --camera /dev/video0 --camera-eye RIGHT --camera-fourcc YUYV
```

If the image is upside down, add these flags to the data collection or inference command as needed:

```bash
--flip-front
--flip-wrist
```

## 5. Network and ports

The inference/collection PC must be able to reach the RT control machine:

```bash
ping <RT-PC-IP>
```

The RT control machine needs these ports reachable:

- `5555/tcp`: RobotState PUB, subscribed to by the inference/collection PC.
- `5556/tcp`: ActionChunk PULL, used by the inference PC to send policy outputs back to the RT machine.
- `8010/tcp`: policy server port, usually local to the inference/training PC.

If the policy server runs on another machine, set `--policy-host` to that machine's IP.

## 6. Common issues

`franky-control==1.1.3` cannot be found: this is usually an architecture mismatch. Use `examples/fr3_realtime` on x86_64 and `examples/fr3_realtime_libfranka` on ARM/aarch64.

ZED camera cannot be opened: use `v4l2-ctl --list-devices` to confirm the device node, then check USB3 connection and `/dev/video*` numbering. With two ZED cameras connected, device numbers may change after replugging.

Policy server runs out of GPU memory: before starting the server, set:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.70
```

Training runs out of GPU memory: first lower the `batch_size` in the `pi05_fr3_lora` config in `src/openpi/training/config.py`. Then consider lowering `XLA_PYTHON_CLIENT_MEM_FRACTION` or reducing other GPU processes.
