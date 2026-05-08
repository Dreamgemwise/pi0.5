# ARM RT 机器运行说明：C++ libfranka 版

这个目录用 C++ `libfranka` 直接控制 FR3，不使用 `pylibfranka`、pybind 或 Polymetis。
网络协议仍然是原来的 ZMQ/msgpack：状态发布 5555，动作接收 5556。

动作约定是 pi0.5/DROID 风格的 action chunk：`actions[:, :7]` 为 15Hz 关节速度，
shape 可以是 `(16, 7)`，如果带夹爪则是 `(16, 8)` 且最后一维为 `0=open, 1=close`。
`robot_server` 收到第一条 chunk 后立即开始执行；新 chunk 到达时会在下一次 libfranka
控制周期接管。关节速度会跨 step 做 EMA，EMA 的前态是上一条 chunk 实际已经执行到的
最后一个 step 速度；每个 step 的 `q_target` 用当前实测关节角 `state.q` 加上速度积分量得到，
避免用旧目标累积造成持续误差。

## 1. 建环境

先确认当前目录是 `fr3_realtime_libfranka`，不是旧的 `fr3_realtime_arm`：

```bash
pwd
ls CMakeLists.txt environment.yml src
```

```bash
mamba env create -f environment.yml
conda activate fr3-realtime-libfranka
```

## 2. 获取 C++ libfranka 源码

```bash
git clone --branch 0.21.1 --depth 1 --recurse-submodules \
  https://github.com/frankarobotics/libfranka.git libfranka-0.21.1
```

如果 CMake 报：

```text
cannot find a tag in git
```

说明 shallow clone 里没有 tag。优先补 tag：

```bash
cd libfranka-0.21.1
git fetch --tags --force
cd ..
```

如果 ARM 机器网络不稳定，也可以让 CMake 不走 git 版本脚本，直接用
`CMakeLists.txt` 里的 `0.21.1`：

```bash
mv libfranka-0.21.1/.git libfranka-0.21.1/.git.disabled
```

## 3. 编译这个 bridge

在 `fr3_realtime_libfranka` 目录里：

```bash
cmake -S . -B build -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX"
cmake --build build
```

如果 CMake 找不到已安装的 `FrankaConfig.cmake`，会自动使用当前目录下的
`libfranka-0.21.1` 源码一起编。

也可以先手动安装 C++ libfranka 到 conda 环境：

```bash
cmake -S libfranka-0.21.1 -B libfranka-build -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" \
  -DBUILD_TESTS=OFF \
  -DBUILD_EXAMPLES=OFF \
  -DGENERATE_PYLIBFRANKA=OFF \
  -DEIGEN3_INCLUDE_DIRS="$CONDA_PREFIX/include/eigen3"

cmake --build libfranka-build
cmake --install libfranka-build
```

安装成功后应该能看到 CMake package 文件：

```bash
ls "$CONDA_PREFIX/lib/cmake/Franka/FrankaConfig.cmake"
```

如果这里没有这个文件，说明 `cmake --install libfranka-build` 没成功，或者安装到了别的 prefix。
先回头修第 2 步，不要继续编 bridge。

然后重新配置 bridge：

```bash
rm -rf build
cmake -S . -B build -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
  -DFranka_DIR="$CONDA_PREFIX/lib/cmake/Franka"
cmake --build build
```

## 4. 先测只读状态

这一步不会发运动命令：

```bash
./build/readonly_state_publisher --robot-ip 172.16.0.2 --no-gripper
```

有 Franka Hand 时：

```bash
./build/readonly_state_publisher --robot-ip 172.16.0.2
```

## 5. 回 home

确认急停和机器人周围安全后：

```bash
./build/go_home --robot-ip 172.16.0.2 --time-to-go 8.0
```

## 6. 启动控制 server

```bash
./build/robot_server --robot-ip 172.16.0.2 --bind 0.0.0.0
```

如果夹爪没接：

```bash
./build/robot_server --robot-ip 172.16.0.2 --bind 0.0.0.0 --no-gripper
```

可以用 `--ema-alpha` 调速度平滑程度，默认 `0.35`。值越小越平滑、响应越慢；值越接近
`1.0` 越贴近 pi0.5 原始输出。`--dynamics-factor` 仍然控制 libfranka 侧的速度、加速度、
jerk 限幅比例，第一次上机建议保持 `0.05` 或更低。

推理/采集机继续把 `--robot-ip` 指向这台 ARM RT 机器的局域网 IP。
