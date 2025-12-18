# Astribot ROS1 Bag 数据转换为 LeRobot 3.0 格式

本项目提供了将 Astribot S1 人形机器人的 ROS1 bag 数据转换为 LeRobot 3.0 数据格式的工具。

## 📋 目录

- [概述](#概述)
- [环境依赖](#环境依赖)
- [脚本说明](#脚本说明)
- [使用方法](#使用方法)
- [数据结构](#数据结构)
- [LeRobot 数据集使用](#lerobot-数据集使用)

---

## 概述

### 原始数据格式 (ROS1 Bag)

Astribot 原始数据存储在 ROS1 bag 文件中，包含以下话题：

| 话题类型 | 话题名称 | 消息类型 | 频率 |
|---------|---------|---------|------|
| 关节状态 | `/astribot_arm_left/joint_space_states` | `RobotJointState` | 250Hz |
| 关节状态 | `/astribot_arm_right/joint_space_states` | `RobotJointState` | 250Hz |
| 关节状态 | `/astribot_gripper_left/joint_space_states` | `RobotJointState` | 250Hz |
| 关节状态 | `/astribot_gripper_right/joint_space_states` | `RobotJointState` | 250Hz |
| 关节控制 | `/astribot_arm_left/joint_space_command` | `RobotJointController` | 250Hz |
| 关节控制 | `/astribot_arm_right/joint_space_command` | `RobotJointController` | 250Hz |
| 图像 | `/astribot_camera/head_rgbd/color_compress/compressed` | `CompressedImage` | 30Hz |
| 图像 | `/astribot_camera/left_wrist_rgbd/color_compress/compressed` | `CompressedImage` | 30Hz |
| 图像 | `/astribot_camera/right_wrist_rgbd/color_compress/compressed` | `CompressedImage` | 30Hz |
| 图像 | `/astribot_camera/torso_rgbd/color_compress/compressed` | `CompressedImage` | 30Hz |

### 目标格式 (LeRobot 3.0)

LeRobot 3.0 是 Hugging Face 推出的机器人学习数据标准格式，支持：
- 统一的数据访问接口
- 视频流存储 (MP4)
- 高效的 Parquet 表格存储
- Hub 直接流式加载

---

## 环境依赖

```bash
# 安装依赖
pip install rosbags tqdm opencv-python-headless numpy

# LeRobot (需要 v3.0 支持)
cd /root/lerobot
pip install -e .
```

---

## 脚本说明

### 1. ROS Bag 提取脚本

**文件**: `/root/extract_rosbag.py`

从 ROS1 bag 文件中提取原始数据到对应文件夹。

```bash
# 提取单个 bag 文件
python extract_rosbag.py /path/to/raw_data.bag

# 指定输出目录
python extract_rosbag.py /path/to/raw_data.bag --output_dir ./output
```

**输出结构**:
```
raw_data_extracted/
├── extraction_summary.json
├── astribot_arm_left_joint_space_states/
│   ├── data.json
│   └── data.csv
├── astribot_camera_head_rgbd_color_compress_compressed/
│   ├── 000000_timestamp.jpg
│   ├── 000001_timestamp.jpg
│   └── ...
└── ...
```

### 2. LeRobot 转换脚本

**文件**: `/root/convert_astribot_to_lerobot.py`

将 Astribot ROS bag 数据转换为 LeRobot 3.0 格式，包含：
- 数据提取
- 帧同步（250Hz → 30Hz）
- 格式转换
- 视频编码

```bash
# 基本用法
python convert_astribot_to_lerobot.py /path/to/bag_root --repo-id user/dataset_name

# 指定任务描述
python convert_astribot_to_lerobot.py /path/to/bag_root \
    --repo-id user/dataset_name \
    --task "抓取红色方块"

# 上传到 Hugging Face Hub
python convert_astribot_to_lerobot.py /path/to/bag_root \
    --repo-id user/dataset_name \
    --push-to-hub
```

---

## 使用方法

### 完整转换流程

```bash
# 1. 设置数据路径
export BAG_ROOT=/root/astribot_raw_datasets
export REPO_ID=astribot/demo_dataset

# 2. 运行转换
python /root/convert_astribot_to_lerobot.py $BAG_ROOT --repo-id $REPO_ID

# 3. 验证数据集
python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('$REPO_ID')
print(ds)
"
```

### 数据目录要求

输入数据应按以下结构组织（每个子目录是一个 episode）：

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
└── episode_003/
    └── record/
        └── raw_data.bag
```

---

## 数据结构

### LeRobot Features 定义

| 特征名称 | 类型 | 维度 | 说明 |
|---------|------|------|------|
| `observation.state` | float32 | (16,) | 合并状态：左臂7 + 右臂7 + 左夹爪1 + 右夹爪1 |
| `observation.state.arm_left.position` | float32 | (7,) | 左臂关节位置 |
| `observation.state.arm_left.velocity` | float32 | (7,) | 左臂关节速度 |
| `observation.state.arm_left.torque` | float32 | (7,) | 左臂关节力矩 |
| `observation.state.arm_right.position` | float32 | (7,) | 右臂关节位置 |
| `observation.state.arm_right.velocity` | float32 | (7,) | 右臂关节速度 |
| `observation.state.arm_right.torque` | float32 | (7,) | 右臂关节力矩 |
| `observation.state.gripper_left.position` | float32 | (1,) | 左夹爪位置 |
| `observation.state.gripper_right.position` | float32 | (1,) | 右夹爪位置 |
| `observation.images.head` | video | (720, 1280, 3) | 头部相机 RGB |
| `observation.images.torso` | video | (720, 1280, 3) | 躯干相机 RGB |
| `observation.images.wrist_left` | video | (360, 640, 3) | 左腕部相机 RGB |
| `observation.images.wrist_right` | video | (360, 640, 3) | 右腕部相机 RGB |
| `action` | float32 | (16,) | 合并动作指令 |
| `action.arm_left` | float32 | (7,) | 左臂控制指令 |
| `action.arm_right` | float32 | (7,) | 右臂控制指令 |
| `action.gripper_left` | float32 | (1,) | 左夹爪控制指令 |
| `action.gripper_right` | float32 | (1,) | 右夹爪控制指令 |

### 帧同步说明

原始数据中关节数据为 250Hz，图像数据为 30Hz。转换时以图像帧率为基准进行同步：

1. 以头部相机时间戳作为参考帧
2. 使用二分查找找到最近的关节数据（容差 50ms）
3. 丢弃没有匹配关节数据的图像帧

### 输出目录结构

```
~/.cache/huggingface/lerobot/astribot/test_dataset/
├── meta/
│   ├── info.json              # 数据集元信息
│   ├── stats.json             # 特征统计信息
│   ├── tasks.parquet          # 任务定义
│   └── episodes/
│       └── chunk-000/
│           └── file-000.parquet   # Episode 元数据
├── data/
│   └── chunk-000/
│       └── file-000.parquet   # 状态和动作数据
└── videos/
    ├── observation.images.head/
    │   └── chunk-000/
    │       └── file-000.mp4
    ├── observation.images.torso/
    │   └── chunk-000/
    │       └── file-000.mp4
    ├── observation.images.wrist_left/
    │   └── chunk-000/
    │       └── file-000.mp4
    └── observation.images.wrist_right/
        └── chunk-000/
            └── file-000.mp4
```

---

## LeRobot 数据集使用

### 加载数据集

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 加载本地数据集
dataset = LeRobotDataset(
    repo_id="astribot/test_dataset",
    root="~/.cache/huggingface/lerobot/astribot/test_dataset"
)

# 或从 Hub 加载
dataset = LeRobotDataset("username/dataset_name")

print(f"Episodes: {dataset.num_episodes}")
print(f"Frames: {dataset.num_frames}")
print(f"FPS: {dataset.fps}")
```

### 获取样本

```python
# 随机访问
sample = dataset[100]

# 获取观测
state = sample["observation.state"]           # torch.Size([16])
arm_left_pos = sample["observation.state.arm_left.position"]  # torch.Size([7])
head_image = sample["observation.images.head"]  # torch.Size([3, 720, 1280])

# 获取动作
action = sample["action"]                     # torch.Size([16])

# 元信息
episode_idx = sample["episode_index"]
frame_idx = sample["frame_index"]
task = sample["task"]
```

### 使用 DataLoader 训练

```python
import torch
from torch.utils.data import DataLoader

# 创建 DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
)

# 训练循环
for batch in dataloader:
    observations = batch["observation.state"]
    actions = batch["action"]
    images = batch["observation.images.head"]
    
    # 模型前向传播
    # predicted_actions = model(observations, images)
    # loss = criterion(predicted_actions, actions)
    # ...
```

### 使用时间窗口

```python
# 获取历史帧
delta_timestamps = {
    "observation.state": [-0.1, 0.0],  # 当前帧和 0.1 秒前的帧
    "observation.images.head": [-0.1, 0.0],
}

dataset = LeRobotDataset(
    repo_id="astribot/test_dataset",
    delta_timestamps=delta_timestamps,
)

sample = dataset[100]
# sample["observation.state"] 现在是 torch.Size([2, 16])
```

---

## 自定义消息定义

Astribot 使用自定义 ROS 消息类型，定义位于 `/root/astribot_msgs/src/msg/`：

### RobotJointState.msg
```
Header header
int8 mode
string[] name
float64[] position
float64[] velocity
float64[] acceleration
float64[] torque
```

### RobotJointController.msg
```
Header header
int8 mode
string[] name
float64[] command
```

---

## 常见问题

### Q: 转换速度慢？
A: 视频编码是主要耗时步骤。可以尝试：
- 减少相机数量
- 降低视频分辨率
- 使用更快的编码预设

### Q: 内存不足？
A: 每次只处理一个 episode，已优化内存使用。如果仍然不足，可以减少 `image_writer_threads`。

### Q: 如何添加新的特征？
A: 修改 `convert_astribot_to_lerobot.py` 中的 `ASTRIBOT_FEATURES` 字典，添加新的特征定义。

---

## 许可证

本项目仅供内部使用。

## 联系方式

如有问题，请联系项目维护者。

