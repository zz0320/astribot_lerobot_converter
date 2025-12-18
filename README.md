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

```bash
# 合并所有 episodes 到单个数据集 (推荐，自动读取任务描述)
python convert.py /root/astribot_raw_datasets -o ./astribot_dataset --repo-id astribot/demo

# 指定全局任务描述
python convert.py /root/astribot_raw_datasets -o ./output --task "Pick up the cup and place it on the table"

# 每个 episode 独立保存
python convert.py /root/astribot_raw_datasets -o ./output --separate --workers 4
```

## 📁 项目结构

```
astribot_lerobot_converter/
├── convert.py              # 主入口脚本
├── README.md               # 本文档
├── scripts/
│   ├── batch_convert.py    # 批量转换脚本
│   ├── convert_single.py   # 单个转换模块
│   └── extract_bag.py      # ROS bag 提取工具
├── configs/                # 配置文件 (可选)
└── docs/                   # 详细文档
```

## 📋 输入数据格式

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

```
python convert.py <bag_root> [选项]

位置参数:
  bag_root              包含 rosbag 数据的根目录

选项:
  -o, --output-dir DIR  输出目录
  --repo-id TEXT        数据集 ID (默认: astribot/dataset)
  --task TEXT           全局任务描述 (强制所有 episode 使用相同描述)
  --no-episode-tasks    禁用从元数据自动读取任务描述
  --separate            每个 episode 独立保存为单独的数据集
  --workers N           并行进程数 (仅用于 --separate 模式)
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
  "operator": "user1"
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

## 🔮 推理 (Inference)

### 方式 1: 使用自定义推理脚本

```bash
# 单帧测试推理
python inference_astribot.py --checkpoint ./outputs/astribot_act --mode single

# 数据集批量推理 (比较预测与真实动作)
python inference_astribot.py \
    --checkpoint ./outputs/astribot_act \
    --mode dataset \
    --num-samples 100 \
    --output ./inference_results

# 查看真实机器人推理示例代码
python inference_astribot.py --mode robot-example
```

### 方式 2: 使用官方 lerobot-eval (仿真环境)

```bash
# 在 PushT 仿真环境中评估
lerobot-eval \
    --policy.path=./outputs/astribot_act \
    --env.type=pusht \
    --eval.n_episodes=10

# 注意: 真实机器人评估需要配置 gym_dora 环境
```

### 方式 3: Python API 推理

```python
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor import PolicyProcessorPipeline
import torch

# 加载模型
policy = ACTPolicy.from_pretrained("./outputs/astribot_act")
policy.eval()
policy.to("cuda")

# 加载预处理器
preprocessor = PolicyProcessorPipeline.from_pretrained(
    "./outputs/astribot_act", "preprocessor"
)
postprocessor = PolicyProcessorPipeline.from_pretrained(
    "./outputs/astribot_act", "postprocessor"
)

# 准备观测 (示例)
observation = {
    "observation.images.head": torch.randn(1, 1, 3, 720, 1280).cuda(),
    "observation.images.wrist_left": torch.randn(1, 1, 3, 360, 640).cuda(),
    "observation.images.wrist_right": torch.randn(1, 1, 3, 360, 640).cuda(),
    "observation.images.torso": torch.randn(1, 1, 3, 720, 1280).cuda(),
    "observation.state": torch.randn(1, 1, 25).cuda(),  # 25维状态向量
}

# 推理
policy.reset()
with torch.no_grad():
    batch = preprocessor(observation)
    action = policy.select_action(batch)
    action = postprocessor(action)

print(f"Predicted action shape: {action['action'].shape}")
# 输出: torch.Size([1, 25]) - 25维动作向量
# 包含: arm_left(7) + arm_right(7) + gripper(2) + head(2) + torso(4) + chassis(3)
```

### 真实机器人推理示例

```python
import time
import torch
# from your_robot_driver import AstribotS1  # 替换为实际驱动

# 加载模型
policy = ACTPolicy.from_pretrained("./outputs/astribot_act")
policy.eval().cuda()
preprocessor = PolicyProcessorPipeline.from_pretrained("./outputs/astribot_act", "preprocessor")
postprocessor = PolicyProcessorPipeline.from_pretrained("./outputs/astribot_act", "postprocessor")

# 连接机器人
# robot = AstribotS1()
# robot.connect()

# 推理循环
policy.reset()
fps = 30

while True:
    # 1. 获取观测
    # images = robot.get_camera_images()
    # joint_states = robot.get_joint_states()
    
    # 2. 构建观测字典 (需要转换为正确格式)
    observation = {
        "observation.images.head": head_image_tensor,      # (1, 1, 3, 720, 1280)
        "observation.images.wrist_left": wrist_l_tensor,   # (1, 1, 3, 360, 640)
        "observation.images.wrist_right": wrist_r_tensor,  # (1, 1, 3, 360, 640)
        "observation.images.torso": torso_tensor,          # (1, 1, 3, 720, 1280)
        "observation.state": state_tensor,                 # (1, 1, 25) - 包含全身关节状态
    }
    
    # 3. 推理
    with torch.no_grad():
        batch = preprocessor(observation)
        action = policy.select_action(batch)
        action = postprocessor(action)
    
    # 4. 执行动作
    action_np = action["action"].cpu().numpy().squeeze()
    # robot.set_joint_commands(action_np)
    
    time.sleep(1/fps)
```

## ⚙️ 环境依赖

```bash
pip install rosbags tqdm opencv-python-headless numpy

# LeRobot v3.0
cd /root/lerobot && pip install -e .
```

## 📝 转换报告

转换完成后会生成 `conversion_report.json`：

```json
{
  "timestamp": "2025-12-08T03:30:00",
  "total_episodes": 3,
  "successful": 3,
  "total_frames": 2459,
  "total_duration": 120.5,
  "results": [
    {
      "bag_dir": "/root/astribot_raw_datasets/ep1",
      "success": true,
      "frames": 945,
      "duration": 45.2
    },
    ...
  ]
}
```

## 🔄 帧同步说明

原始数据频率:
- 关节数据: 250 Hz (手臂、夹爪、头部、腰部、底盘)
- 图像数据: 30 Hz

转换时以 head 相机帧率 (30 Hz) 为基准，使用最近邻插值同步关节数据。

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

**Q: 如何添加新的传感器数据？**
A: 修改 `scripts/convert_merged.py` 中的以下部分：
1. 添加关节数常量 (如 `NEW_JOINTS = 3`)
2. 在 `ASTRIBOT_FEATURES` 中添加特征定义
3. 更新 `observation.state` 和 `action` 的维度
4. 在 `extract_bag_data()` 中添加 topic mapping
5. 在 `synchronize_data()` 中添加数据索引
6. 在 `convert_frame_to_lerobot()` 中添加数据处理逻辑

## 📄 许可证

内部使用

