# Astribot LeRobot 数据转换器

将 Astribot S1 人形机器人的 ROS1 bag 数据转换为 LeRobot 3.0 格式。

## 🤖 机器人配置概述

| 部件 | 关节数 | 说明 |
|------|--------|------|
| 左臂 (arm_left) | 7 | 7 自由度机械臂 |
| 右臂 (arm_right) | 7 | 7 自由度机械臂 |
| 左夹爪 (gripper_left) | 1 | 夹爪开合 |
| 右夹爪 (gripper_right) | 1 | 夹爪开合 |
| 头部 (head) | 2 | pan/tilt 云台 |
| 腰部 (torso) | 4 | 躯干关节 |
| 底盘 (chassis) | 3 | 移动底盘 |
| **总计** | **25** | 状态/动作向量维度 |

## 🚀 快速开始

### 目录转换 (ROS Bag 目录)

```bash
# 合并所有 episodes 到单个数据集 (推荐，自动读取任务描述)
python convert.py /root/astribot_raw_datasets -o ./astribot_dataset --repo-id astribot/demo

# 指定全局任务描述
python convert.py /root/astribot_raw_datasets -o ./output --task "Pick up the cup and place it on the table"

# 每个 episode 独立保存
python convert.py /root/astribot_raw_datasets -o ./output --separate --workers 4
```

### Tar 文件转换

```bash
# 转换单个 tar 文件
python convert_tar.py /path/to/astribot_data.tar -o ./output

# 转换目录下所有 tar 文件 (合并为单个数据集)
python convert.py /root/datasets/astribot_rawdata -o ./output --repo-id astribot/dataset

# 使用 convert.py 自动检测并转换 tar 文件
python convert.py /root/datasets/astribot_rawdata -o ./output
```

## 📁 项目结构

```
astribot_lerobot_converter/
├── convert.py              # 主入口脚本 (支持目录和 tar 文件)
├── convert_tar.py          # Tar 文件专用转换器
├── train_astribot.py       # 训练脚本示例
├── README.md               # 本文档
├── scripts/
│   ├── batch_convert.py    # 批量转换脚本 (独立保存模式)
│   ├── convert_single.py   # 单个 episode 转换模块
│   ├── convert_merged.py   # 合并转换模块 (核心逻辑)
│   ├── extract_bag.py      # ROS bag 提取工具
│   └── visualize.py        # 数据可视化工具
├── configs/                # 配置文件 (可选)
└── docs/                   # 详细文档
```

## 📋 输入数据格式

### 目录模式

输入数据应按以下结构组织：

```
astribot_raw_datasets/
├── episode_001/
│   ├── __loongdata_metadata.json
│   ├── meta_info.json
│   └── record/
│       └── raw_data.bag
├── episode_002/
│   └── record/
│       └── raw_data.bag
└── ...
```

### Tar 文件模式

Tar 文件内部结构：

```
astribot_data.tar
├── __loongdata_metadata.json   # 可选，包含 taskName 等
├── meta_info.json              # 可选
└── record/
    └── raw_data.bag            # 必需
```

## 📤 输出格式

### 默认模式：单个数据集 (多 episodes)

```
astribot_dataset/
├── conversion_report.json     # 转换报告
├── meta/
│   ├── info.json              # 数据集元信息
│   ├── stats.json             # 统计信息
│   ├── tasks.parquet          # 任务定义
│   └── episodes/              # Episode 索引
│       └── chunk-000/
│           └── file-000.parquet
├── data/
│   └── chunk-000/
│       └── file-000.parquet   # 所有 episodes 的数据
└── videos/
    ├── observation.images.head/
    │   └── chunk-000/
    │       └── file-000.mp4   # 多个 episodes 合并的视频
    ├── observation.images.torso/
    ├── observation.images.wrist_left/
    └── observation.images.wrist_right/
```

### --separate 模式：独立数据集

```
output/
├── conversion_report.json
├── episode_001/               # 独立数据集
│   ├── meta/
│   ├── data/
│   └── videos/
├── episode_002/
└── ...
```

## 🔧 命令行参数

### convert.py (主入口)

```
python convert.py <input_path> [选项]

位置参数:
  input_path            包含 rosbag 数据的根目录或 tar 文件路径

选项:
  -o, --output-dir DIR  输出目录
  --repo-id TEXT        数据集 ID (默认: astribot/dataset)
  --task TEXT           全局任务描述 (强制所有 episode 使用相同描述)
  --no-episode-tasks    禁用从元数据自动读取任务描述
  --separate            每个 episode 独立保存为单独的数据集
  --workers N           并行进程数 (仅用于 --separate 模式)
```

### convert_tar.py (Tar 专用)

```
python convert_tar.py <input_path> [选项]

位置参数:
  input_path            tar 文件路径或包含 tar 文件的目录

选项:
  -o, --output-dir DIR  输出目录 (必需)
  --repo-id TEXT        数据集 ID (默认: astribot/dataset)
  --task TEXT           任务描述 (如果未指定，将从元数据中读取)
  --episode-id TEXT     Episode ID (如果未指定，将从元数据中读取)
  --verbose             显示详细输出
```

## 🗣️ 语言描述 (Language Instruction) 支持

转换器支持为每个 episode 设置独立的语言描述，用于条件生成训练。

### 任务描述来源 (优先级从高到低)

1. **自定义文件**: episode 目录下的 `task_description.txt`
2. **元数据文件**: `__loongdata_metadata.json` 中的 `taskName` 字段
3. **命令行参数**: `--task` 指定的全局描述
4. **默认值**: "Astribot manipulation task"

### 使用方式

```bash
# 方式 1: 自动从元数据读取 (默认)
python convert.py /root/astribot_raw_datasets -o ./output

# 方式 2: 强制使用全局任务描述
python convert.py /root/astribot_raw_datasets -o ./output \
    --task "Pick up the red cube and place it in the box"

# 方式 3: 为每个 episode 创建自定义描述文件
echo "Grasp the bottle with left hand" > /root/astribot_raw_datasets/episode_001/task_description.txt
echo "Pour water into the cup" > /root/astribot_raw_datasets/episode_002/task_description.txt
python convert.py /root/astribot_raw_datasets -o ./output
```

### 元数据文件示例

`__loongdata_metadata.json`:
```json
{
  "taskName": "astribot_test2",
  "scene": "kitchen",
  "operator": "user1",
  "equipmentModel": "S1",
  "duration": 45000
}
```

生成的任务描述: `"astribot_test2 - 场景: kitchen"`

## 📊 数据特征 (Features)

### 观测状态

| 特征名 | 维度 | 说明 |
|--------|------|------|
| `observation.state` | (25,) | 合并状态向量 |

**手臂关节 (每侧 7 轴)**

| 特征名 | 维度 | 说明 |
|--------|------|------|
| `observation.state.arm_left.position` | (7,) | 左臂关节位置 |
| `observation.state.arm_left.velocity` | (7,) | 左臂关节速度 |
| `observation.state.arm_left.torque` | (7,) | 左臂关节力矩 |
| `observation.state.arm_right.position` | (7,) | 右臂关节位置 |
| `observation.state.arm_right.velocity` | (7,) | 右臂关节速度 |
| `observation.state.arm_right.torque` | (7,) | 右臂关节力矩 |

**夹爪 (每侧 1 轴)**

| 特征名 | 维度 | 说明 |
|--------|------|------|
| `observation.state.gripper_left.position` | (1,) | 左夹爪位置 |
| `observation.state.gripper_right.position` | (1,) | 右夹爪位置 |

**头部 (2 轴: pan/tilt)**

| 特征名 | 维度 | 说明 |
|--------|------|------|
| `observation.state.head.position` | (2,) | 头部关节位置 |
| `observation.state.head.velocity` | (2,) | 头部关节速度 |
| `observation.state.head.torque` | (2,) | 头部关节力矩 |

**腰部 (4 轴)**

| 特征名 | 维度 | 说明 |
|--------|------|------|
| `observation.state.torso.position` | (4,) | 腰部关节位置 |
| `observation.state.torso.velocity` | (4,) | 腰部关节速度 |
| `observation.state.torso.torque` | (4,) | 腰部关节力矩 |

**底盘 (3 轴)**

| 特征名 | 维度 | 说明 |
|--------|------|------|
| `observation.state.chassis.position` | (3,) | 底盘关节位置 |
| `observation.state.chassis.velocity` | (3,) | 底盘关节速度 |
| `observation.state.chassis.torque` | (3,) | 底盘关节力矩 |

### 图像观测

| 特征名 | 分辨率 | 说明 |
|--------|--------|------|
| `observation.images.head` | 720×1280 | 头部相机 |
| `observation.images.torso` | 720×1280 | 躯干相机 |
| `observation.images.wrist_left` | 360×640 | 左腕部相机 |
| `observation.images.wrist_right` | 360×640 | 右腕部相机 |

### 动作

| 特征名 | 维度 | 说明 |
|--------|------|------|
| `action` | (25,) | 合并动作指令向量 |
| `action.arm_left` | (7,) | 左臂控制指令 |
| `action.arm_right` | (7,) | 右臂控制指令 |
| `action.gripper_left` | (1,) | 左夹爪指令 |
| `action.gripper_right` | (1,) | 右夹爪指令 |
| `action.head` | (2,) | 头部控制指令 |
| `action.torso` | (4,) | 腰部控制指令 |
| `action.chassis` | (3,) | 底盘控制指令 |

### 状态/动作向量结构

`observation.state` 和 `action` 向量维度为 25，结构如下：

```
索引 0-6:   arm_left     (7个关节)
索引 7-13:  arm_right    (7个关节)
索引 14:    gripper_left (1个关节)
索引 15:    gripper_right(1个关节)
索引 16-17: head         (2个关节)
索引 18-21: torso        (4个关节)
索引 22-24: chassis      (3个关节)
```

## 📖 使用转换后的数据

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 加载数据集
dataset = LeRobotDataset(
    repo_id="astribot/demo",
    root="./astribot_dataset"
)

print(f"Episodes: {dataset.num_episodes}")  # 3
print(f"总帧数: {dataset.num_frames}")       # 2459
print(f"FPS: {dataset.fps}")                # 30

# 获取样本
sample = dataset[0]
state = sample["observation.state"]           # torch.Size([25])
action = sample["action"]                     # torch.Size([25])
head_image = sample["observation.images.head"]  # torch.Size([3, 720, 1280])
episode_idx = sample["episode_index"]         # 当前帧所属的 episode

# 分解状态向量
arm_left = state[0:7]      # 左臂 7 关节
arm_right = state[7:14]    # 右臂 7 关节
gripper = state[14:16]     # 左右夹爪
head = state[16:18]        # 头部 2 关节
torso = state[18:22]       # 腰部 4 关节
chassis = state[22:25]     # 底盘 3 关节

# 按 episode 索引数据
for i in range(dataset.num_episodes):
    ep_info = dataset.meta.episodes[i]
    print(f"Episode {i}: {ep_info['length']} 帧")
```

## 🔍 数据可视化

提供多种可视化方式查看转换后的数据：

```bash
# 显示单帧图像 (默认)
python scripts/visualize.py /root/astribot_dataset --repo-id astribot/demo --episode 0

# 使用 Rerun 可视化 (推荐，交互式)
python scripts/visualize.py /root/astribot_dataset --repo-id astribot/demo --episode 0 --rerun

# 导出为视频
python scripts/visualize.py /root/astribot_dataset --repo-id astribot/demo --episode 0 --export-video -o ./videos

# 绘制关节数据图表
python scripts/visualize.py /root/astribot_dataset --repo-id astribot/demo --episode 0 --plot -o ./plots

# 显示指定帧
python scripts/visualize.py /root/astribot_dataset --repo-id astribot/demo --episode 0 --show-frame --frame 100
```

### 可视化选项

| 选项 | 说明 |
|------|------|
| `--rerun` | 使用 Rerun 进行交互式可视化 |
| `--plot` | 生成关节位置 matplotlib 图表 |
| `--export-video` | 导出 episode 为 MP4 视频 |
| `--show-frame` | 显示单帧所有相机图像 |
| `--camera NAME` | 指定导出视频使用的相机 |
| `--episode N` | 指定可视化的 episode 索引 |
| `--frame N` | 指定显示的帧偏移 |
| `-o DIR` | 输出目录 |

## 🛠️ 其他工具

### 提取 ROS Bag 数据 (不转换为 LeRobot)

```bash
python scripts/extract_bag.py /path/to/raw_data.bag -o ./extracted
```

### 转换单个 Episode

```bash
python scripts/convert_single.py /path/to/episode_dir -o ./output
```

## 🎓 训练模型

### 方式 1: 使用自定义训练脚本

```bash
# ACT 策略 (推荐用于精细操作任务)
python train_astribot.py --policy act --steps 50000

# Diffusion 策略 (适合复杂动作序列)
python train_astribot.py --policy diffusion --steps 100000

# 自定义参数
python train_astribot.py \
    --policy act \
    --dataset-path /root/astribot_lerobot_dataset_v2 \
    --repo-id astribot/demo_v2 \
    --steps 50000 \
    --batch-size 8 \
    --output-dir ./outputs/my_model
```

### 方式 2: 使用官方 lerobot-train 命令

```bash
# ACT 策略
lerobot-train \
    --dataset.repo_id=astribot/demo_v2 \
    --dataset.root=/root/astribot_lerobot_dataset_v2 \
    --policy.type=act \
    --steps=50000 \
    --batch_size=8 \
    --wandb.enable=true \
    --wandb.project=astribot

# Diffusion 策略
lerobot-train \
    --dataset.repo_id=astribot/demo_v2 \
    --dataset.root=/root/astribot_lerobot_dataset_v2 \
    --policy.type=diffusion \
    --steps=100000

# 多 GPU 训练
accelerate launch --num_processes=4 \
    $(which lerobot-train) \
    --dataset.repo_id=astribot/demo_v2 \
    --dataset.root=/root/astribot_lerobot_dataset_v2 \
    --policy.type=act \
    --steps=50000
```

### 支持的策略类型

| 策略 | 类型 | 说明 |
|------|------|------|
| ACT | `--policy.type=act` | Action Chunking Transformer，适合精细操作 |
| Diffusion | `--policy.type=diffusion` | Diffusion Policy，适合复杂动作序列 |
| VQ-BeT | `--policy.type=vqbet` | Vector Quantized Behavior Transformer |
| TDMPC | `--policy.type=tdmpc` | Temporal Difference MPC |
| Pi0 | `--policy.type=pi0` | Physical Intelligence Pi0 |
| SmolVLA | `--policy.type=smolvla` | Small Vision-Language-Action |

## ⚙️ 环境依赖

```bash
# 基础依赖
pip install rosbags tqdm opencv-python-headless numpy

# 可视化依赖 (可选)
pip install matplotlib rerun-sdk

# LeRobot v3.0
cd /root/lerobot && pip install -e .
```

## 📝 转换报告

转换完成后会生成 `conversion_report.json`：

```json
{
  "timestamp": "2025-12-08T03:30:00",
  "repo_id": "astribot/demo",
  "total_episodes": 3,
  "total_frames": 2459,
  "total_tasks": 2,
  "tasks": ["astribot_test2 - 场景: kitchen", "Pick up cup"],
  "fps": 30,
  "robot_type": "astribot_s1",
  "episodes": [
    {
      "episode_index": 0,
      "source": "ep1",
      "task": "astribot_test2 - 场景: kitchen",
      "frames": 945
    }
  ],
  "sync_config": {
    "base_topic": "/astribot_camera/head_rgbd/color_compress/compressed",
    "joint_tolerance_ms": 50,
    "image_tolerance_ms": 100
  }
}
```

## 🔄 帧同步说明

原始数据频率:
- 关节数据: 250 Hz (手臂、夹爪、头部、腰部、底盘)
- 图像数据: 30 Hz

### 组帧逻辑

1. **基准选择**: 以 head 相机时间戳为基准 (30 Hz)
2. **关节同步**: 对每个基准时间戳 t，查找 t ± 50ms 内最近的关节数据
3. **图像同步**: 查找 t ± 100ms 内最近的其他相机图像
4. **有效帧条件**: 必须有 arm_left/arm_right 的状态和命令数据

### 支持的 ROS Topics

| Topic | 类型 | 说明 |
|-------|------|------|
| `/astribot_arm_left/joint_space_states` | JointState | 左臂状态 |
| `/astribot_arm_right/joint_space_states` | JointState | 右臂状态 |
| `/astribot_gripper_left/joint_space_states` | JointState | 左夹爪状态 |
| `/astribot_gripper_right/joint_space_states` | JointState | 右夹爪状态 |
| `/astribot_head/joint_space_states` | JointState | 头部状态 |
| `/astribot_torso/joint_space_states` | JointState | 腰部状态 |
| `/astribot_chassis/joint_space_states` | JointState | 底盘状态 |
| `/astribot_arm_left/joint_space_command` | JointController | 左臂命令 |
| `/astribot_arm_right/joint_space_command` | JointController | 右臂命令 |
| `/astribot_gripper_left/joint_space_command` | JointController | 左夹爪命令 |
| `/astribot_gripper_right/joint_space_command` | JointController | 右夹爪命令 |
| `/astribot_head/joint_space_command` | JointController | 头部命令 |
| `/astribot_torso/joint_space_command` | JointController | 腰部命令 |
| `/astribot_chassis/joint_space_command` | JointController | 底盘命令 |
| `/astribot_camera/head_rgbd/color_compress/compressed` | Image | 头部相机 |
| `/astribot_camera/torso_rgbd/color_compress/compressed` | Image | 躯干相机 |
| `/astribot_camera/left_wrist_rgbd/color_compress/compressed` | Image | 左腕相机 |
| `/astribot_camera/right_wrist_rgbd/color_compress/compressed` | Image | 右腕相机 |

## ❓ 常见问题

**Q: 并行转换时内存不足？**
A: 减少 `--workers` 数量，或增加系统内存。

**Q: 某个 episode 转换失败？**
A: 查看 `conversion_report.json` 中的错误信息，单独重试该 episode。

**Q: 如何处理 tar.gz 文件？**
A: 目前仅支持 `.tar` 文件，需先解压 `.tar.gz`:
```bash
gunzip your_file.tar.gz
python convert_tar.py your_file.tar -o ./output
```

**Q: 如何添加新的传感器数据？**
A: 修改 `scripts/convert_merged.py` 中的以下部分：
1. 添加关节数常量 (如 `NEW_JOINTS = 3`)
2. 在 `ASTRIBOT_FEATURES` 中添加特征定义
3. 更新 `observation.state` 和 `action` 的维度
4. 在 `extract_bag_data()` 中添加 topic mapping
5. 在 `synchronize_data()` 中添加数据索引
6. 在 `convert_frame_to_lerobot()` 中添加数据处理逻辑

**Q: Rerun 可视化窗口无法打开？**
A: 确保安装了 rerun-sdk: `pip install rerun-sdk`，并且在支持 GUI 的环境中运行。

## 📄 许可证

内部使用
