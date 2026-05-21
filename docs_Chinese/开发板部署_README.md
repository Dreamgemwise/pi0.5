# 开发板部署 README

本文档记录当前推荐部署方式：

```text
开发板：
  从 GitHub 拉取代码仓库
  单独放置 base model / 微调 checkpoint
  运行 pi0.5 policy server
  监听 8010 端口

本机电脑：
  运行 robot_server，连接 Franka
  运行 inference_client，读取本机相机
  将图像、机器人状态、prompt 发给开发板
  接收开发板返回的 action chunk 并下发给 Franka
```

也就是说：开发板负责模型推理，本机仍然负责机械臂控制和相机采集。

## 1. 整体架构

假设：

```text
开发板 IP：192.168.1.50
本机电脑 IP：192.168.1.10
Franka FCI IP：172.16.0.2
policy server 端口：8010
robot state 端口：5555
robot action 端口：5556
```

数据流：

```text
本机相机 /dev/video0, /dev/video2
        |
        v
本机 inference_client
        |
        | 图像 + joint_position + gripper_position + prompt
        v
开发板 policy_server :8010
        |
        | action chunk
        v
本机 inference_client
        |
        | ZMQ action chunk
        v
本机 robot_server :5556
        |
        v
Franka
```

## 2. 开发板拉取代码

开发板上执行：

```bash
cd /home/user

git clone https://github.com/Dreamgemwise/pi0.5.git pi0.5-on-Franka-Research3
cd /home/user/pi0.5-on-Franka-Research3
```

如果后续本机更新了 GitHub 仓库，开发板同步代码：

```bash
cd /home/user/pi0.5-on-Franka-Research3
git pull
```

注意：GitHub 仓库只保存代码和文档，不保存大模型、checkpoint、数据集。下面的模型文件需要单独拷贝到开发板。

## 3. 开发板需要的模型目录

当前 cube + strawberry 多任务模型：

```bash
checkpoints/pi05_fr3_lora/multitask_cube_strawberry_rawbase_223eps_v1/9999
```

该 checkpoint 是完整 checkpoint，目录中应包含：

```text
9999/
  _CHECKPOINT_METADATA
  assets/
  params/
  train_state/
```

开发板上最终建议目录：

```text
/home/user/pi0.5-on-Franka-Research3/
  models/
    pi05_droid/
      assets/
      params/

  checkpoints/
    pi05_fr3_lora/
      multitask_cube_strawberry_rawbase_223eps_v1/
        9999/
          _CHECKPOINT_METADATA
          assets/
          params/
          train_state/
```

说明：

- `checkpoints/.../9999` 是你训练好的模型。
- `models/pi05_droid/assets` 建议保留，用于 norm stats 等配置。
- 最稳妥做法是先完整拷贝 `models/pi05_droid`，跑通后再考虑精简。

## 4. 从本机拷贝模型到开发板

在本机执行。

假设开发板 IP 是 `192.168.1.50`，用户名是 `user`。

先创建开发板目录：

```bash
ssh user@192.168.1.50 "mkdir -p /home/user/pi0.5-on-Franka-Research3/models /home/user/pi0.5-on-Franka-Research3/checkpoints/pi05_fr3_lora/multitask_cube_strawberry_rawbase_223eps_v1"
```

拷贝 base model：

```bash
rsync -avL \
  /home/mscape/pi_05/openpi/models/pi05_droid \
  user@192.168.1.50:/home/user/pi0.5-on-Franka-Research3/models/
```

拷贝微调 checkpoint：

```bash
rsync -avL \
  /home/mscape/pick/pi0.5-on-Franka-Research3/checkpoints/pi05_fr3_lora/multitask_cube_strawberry_rawbase_223eps_v1/9999 \
  user@192.168.1.50:/home/user/pi0.5-on-Franka-Research3/checkpoints/pi05_fr3_lora/multitask_cube_strawberry_rawbase_223eps_v1/
```

注意：本机仓库中的 `models/pi05_droid` 可能是软链接，所以这里使用 `rsync -L`，确保拷贝真实内容。

## 5. 开发板安装环境

开发板上执行：

```bash
cd /home/user/pi0.5-on-Franka-Research3

conda create -n openpi-dev python=3.11 -y
conda activate openpi-dev

pip install -e .
pip install -e packages/openpi-client
```

检查 JAX 是否使用 GPU：

```bash
python -c "import jax; print(jax.devices())"
```

理想输出应包含 GPU，例如：

```text
CudaDevice(id=0)
```

如果只看到 `CpuDevice`，说明开发板没有使用 GPU，pi0.5 实时推理会很慢，不建议直接真机运行。

## 6. 开发板启动 policy server

开发板终端执行：

```bash
cd /home/user/pi0.5-on-Franka-Research3
conda activate openpi-dev

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.70

python scripts/serve_policy.py \
  --port 8010 \
  policy:checkpoint \
  --policy.config=pi05_fr3_lora \
  --policy.dir=checkpoints/pi05_fr3_lora/multitask_cube_strawberry_rawbase_223eps_v1/9999
```

启动成功后应看到类似日志：

```text
Loading model...
Finished restoring checkpoint...
Creating server...
```

开发板保持该终端运行。

## 7. 本机检查开发板连接

本机执行：

```bash
ping 192.168.1.50
nc -vz 192.168.1.50 8010
```

如果 `8010` 不通，检查：

- 开发板 policy server 是否已经启动。
- 开发板 IP 是否正确。
- 开发板防火墙是否阻止 8010。
- 本机和开发板是否在同一网络。

## 8. 本机启动 robot server

本机终端 1：

```bash
cd /home/mscape/pick/pi0.5-on-Franka-Research3

DYNAMICS_FACTOR=0.08 EMA_ALPHA=0.45 \
./local_deploy/start_robot_server_local.sh
```

如果提示 `Address already in use`：

```bash
fuser -k 5555/tcp
fuser -k 5556/tcp
```

然后重新启动。

## 9. 本机启动 inference client

本机终端 2。

抓草莓：

```bash
cd /home/mscape/pick/pi0.5-on-Franka-Research3

PROMPT="pick up the strawberry" \
FRONT_CAMERA=/dev/video0 \
WRIST_CAMERA=/dev/video2 \
WRIST_TRANSFORM=flip180 \
POLICY_HOST=192.168.1.50 \
POLICY_PORT=8010 \
./local_deploy/start_inference_local.sh
```

抓绿色方块：

```bash
cd /home/mscape/pick/pi0.5-on-Franka-Research3

PROMPT="pick up the green cube" \
FRONT_CAMERA=/dev/video0 \
WRIST_CAMERA=/dev/video2 \
WRIST_TRANSFORM=flip180 \
POLICY_HOST=192.168.1.50 \
POLICY_PORT=8010 \
./local_deploy/start_inference_local.sh
```

注意：不要同时启动两个 inference client。它们会同时占用相机并同时给机器人发送 action。

## 10. 总共需要几个终端

```text
开发板：1 个终端
  policy_server

本机：2 个终端
  robot_server
  inference_client

总共：3 个终端
```

## 11. 本机与开发板传输内容

本机传给开发板 policy server：

```text
observation/exterior_image_1_left：外部相机 RGB，224x224
observation/wrist_image_left：腕部相机 RGB，224x224，当前使用 flip180
observation/joint_position：7 维关节角
observation/gripper_position：1 维夹爪状态，0=open，1=close
prompt：任务文本
```

开发板返回给本机：

```text
actions：action chunk
```

每个 action 通常为：

```text
action[0:7]：7 维关节速度
action[7]：夹爪开合信号
```

本机 inference client 收到 action chunk 后，会通过 ZMQ 发给本机 robot server 执行。

## 12. 速度关系

当前链路是异步的：

```text
robot_server 控制频率：约 1000Hz
policy action 频率：15Hz
inference 请求频率：默认 3Hz
```

本机 inference 日志中会显示：

```text
chunk sent: N=16, infer=xx.xms, state_age=xx.xms
```

重点关注 `infer=xx.xms`：

```text
infer < 300ms：较理想，可以 REQUEST_HZ=3.0
infer 300ms - 500ms：建议 REQUEST_HZ=2.0
infer > 500ms：动作会卡顿，不建议直接真机高速运行
```

如果开发板推理较慢，可以降低本机请求频率：

```bash
PROMPT="pick up the strawberry" \
FRONT_CAMERA=/dev/video0 \
WRIST_CAMERA=/dev/video2 \
WRIST_TRANSFORM=flip180 \
POLICY_HOST=192.168.1.50 \
POLICY_PORT=8010 \
REQUEST_HZ=2.0 \
./local_deploy/start_inference_local.sh
```

## 13. 常见问题

### 13.1 开发板只看到 CPU

```bash
python -c "import jax; print(jax.devices())"
```

如果只看到 `CpuDevice`，先解决开发板 JAX GPU 环境，否则实时推理速度不够。

### 13.2 本机连不上开发板 8010

```bash
nc -vz 192.168.1.50 8010
```

不通时检查开发板 IP、防火墙、policy server 是否启动。

### 13.3 本机相机被占用

```bash
fuser -v /dev/video0 /dev/video2
fuser -k /dev/video0
fuser -k /dev/video2
```

### 13.4 robot server 端口被占用

```bash
fuser -k 5555/tcp
fuser -k 5556/tcp
```

### 13.5 prompt 不要写成长任务

当前模型训练的是两个单步抓取任务。推荐 prompt：

```text
pick up the strawberry
pick up the green cube
```

不要直接使用：

```text
First pick up the strawberry and put it into the box, then pick up the cube...
```

这类多步放置任务需要额外采集对应数据并重新训练。
