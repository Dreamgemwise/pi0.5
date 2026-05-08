# x86_64 RT 机器运行说明

这个目录是原始 x86_64 版本，控制链路走 `franky-control` 直连 Franka。
ARM/aarch64 控制机请用 `../fr3_realtime_arm`，那里走 `pylibfranka`。

## 1. 建环境

```bash
conda env create -f environment.yml
conda activate fr3-realtime
```

如果 `franky-control==1.1.3` 找不到，先确认这台机器是 x86_64：

```bash
uname -m
```

这里应输出 `x86_64`。ARM 机器不要用这个环境文件。

## 2. 回 home

确认急停和机器人周围安全后：

```bash
python go_home.py --robot-ip 172.16.0.2 --time-to-go 8.0
```

## 3. 启动控制 server

```bash
python robot_server.py --robot-ip 172.16.0.2 --bind 0.0.0.0
```

推理/采集机继续把 `--robot-ip` 指向这台 RT 机器的局域网 IP。
