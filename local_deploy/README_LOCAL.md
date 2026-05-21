# Local single-machine deployment

This checkout is configured for running both inference and FR3 deployment on this computer.

## Paths

- Repo: `/home/mscape/pick/pi0.5-on-Franka-Research3`
- Local DROID base checkpoint: `models/pi05_droid` -> `/home/mscape/pi_05/openpi/models/pi05_droid`
- Policy server: `127.0.0.1:8010`
- Robot state/action host for same-machine deployment: `127.0.0.1`
- Franka FCI IP default: `172.16.0.2`

## 1. Move robot home

```bash
./local_deploy/go_home_local.sh
```

Override the robot IP if needed:

```bash
ROBOT_FCI_IP=172.16.0.2 ./local_deploy/go_home_local.sh
```

## 2. Start robot server

Open a terminal:

```bash
./local_deploy/start_robot_server_local.sh
```

This uses the C++ libfranka bridge and binds the action/state bridge on this machine. The inference client should connect to `127.0.0.1`.

If you do not have a Franka Hand/gripper connected:

```bash
NO_GRIPPER=1 ./local_deploy/start_robot_server_local.sh
```

## 3. Start policy server

Open a second terminal:

```bash
./local_deploy/start_policy_local.sh
```

By default this serves the local DROID base checkpoint with config `pi05_droid`. For a trained FR3 checkpoint, use:

```bash
POLICY_CONFIG=pi05_fr3_lora \
POLICY_DIR=checkpoints/pi05_fr3_lora/<run_name>/<step> \
./local_deploy/start_policy_local.sh
```

For the orange LoRA checkpoint trained from `datasets/fr3_pick_up_orange`, use:

```bash
POLICY_CONFIG=pi05_fr3_lora \
POLICY_DIR=checkpoints/pi05_fr3_lora/orange_lora_18eps_v1/2999 \
./local_deploy/start_policy_local.sh
```

## 4. Start inference client

Open a third terminal and set the task prompt and camera nodes:

```bash
PROMPT="pick up the cup" \
FRONT_CAMERA=/dev/video0 \
WRIST_CAMERA=/dev/video2 \
./local_deploy/start_inference_local.sh
```

Optional overrides:

```bash
FLIP_WRIST=1 REQUEST_HZ=3.0 ./local_deploy/start_inference_local.sh
```

## Safety note

The base DROID checkpoint is useful for verifying local serving, but real FR3 task execution should use a checkpoint trained or validated for your robot, camera views, gripper convention, and prompt.
