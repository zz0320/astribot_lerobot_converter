# Astribot LeRobot 数据转换器

将 Astribot S1 人形机器人的 ROS1 bag 数据转换为 LeRobot 3.0 格式。

**统一使用 25 维完整特征。**

## 🤖 机器人配置

| 部件 | 关节数 | 索引 |
|------|--------|------|
| 左臂 (arm_left) | 7 | 0-6 |
| 右臂 (arm_right) | 7 | 7-13 |
| 左夹爪 (gripper_left) | 1 | 14 |
| 右夹爪 (gripper_right) | 1 | 15 |
| 头部 (head) | 2 | 16-17 |
| 腰部 (torso) | 4 | 18-21 |
| 底盘 (chassis) | 3 | 22-24 |
| **总计** | **25** | 0-24 |

## 🚀 快速开始

```bash
cd /root/astribot_lerobot_converter

# 转换 (自动检测 rosbag/tar)
python convert.py /root/datasets -o ./output --repo-id astribot/demo

# 转换 tar 文件
python convert.py /path/to/data.tar -o ./output

# 独立保存每个 episode
python convert.py /root/datasets -o ./output --separate --workers 4
```

## 📁 项目结构

```
astribot_lerobot_converter/
├── convert.py              # 主入口 (自动路由)
├── train_astribot.py       # 训练脚本
├── README.md
└── scripts/
    ├── core.py             # 核心模块 (25维特征定义)
    ├── tar_converter.py    # Tar 文件转换
    ├── batch.py            # 批量独立转换
    ├── extract_bag.py      # ROS bag 提取
    └── visualize.py        # 数据可视化
```

## 📊 模块关系

```
scripts/core.py (核心)
    │
    ├── ASTRIBOT_FEATURES (25维)
    ├── extract_bag_data()
    ├── synchronize_data()
    ├── convert_frame_to_lerobot()
    └── ParallelImageDecoder
           │
    ┌──────┴──────┐
    │             │
tar_converter.py  batch.py
    │             │
    └──────┬──────┘
           │
     全部 25 维特征
```

## 📋 输入格式

**目录模式:**
```
datasets/
├── episode_001/
│   ├── __loongdata_metadata.json
│   └── record/raw_data.bag
├── episode_002/
│   └── record/raw_data.bag
```

**Tar 模式:**
```
data.tar
├── __loongdata_metadata.json
└── record/raw_data.bag
```

## 📤 输出格式

```
output/
├── conversion_report.json
├── meta/
│   ├── info.json
│   ├── stats.json
│   └── tasks.parquet
├── data/
│   └── chunk-000/file-000.parquet
└── videos/
    ├── observation.images.head/
    ├── observation.images.torso/
    ├── observation.images.wrist_left/
    └── observation.images.wrist_right/
```

## 🔧 命令行参数

```
python convert.py <input_path> [选项]

参数:
  input_path            rosbag 目录或 tar 文件
  -o, --output-dir      输出目录
  --repo-id             数据集 ID (默认: astribot/dataset)
  --task                全局任务描述
  --no-episode-tasks    禁用从元数据读取任务
  --separate            独立保存每个 episode
  --workers N           并行数 (仅 --separate)
```

## 📊 25 维特征

```
observation.state / action 结构:

索引 0-6:   arm_left      (7)
索引 7-13:  arm_right     (7)
索引 14:    gripper_left  (1)
索引 15:    gripper_right (1)
索引 16-17: head          (2)
索引 18-21: torso         (4)
索引 22-24: chassis       (3)
```

## 📖 使用数据集

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(repo_id="astribot/demo", root="./output")

sample = dataset[0]
state = sample["observation.state"]    # (25,)
action = sample["action"]              # (25,)

# 分解
arm_left = state[0:7]
arm_right = state[7:14]
gripper = state[14:16]
head = state[16:18]
torso = state[18:22]
chassis = state[22:25]
```

## 🎓 训练

```bash
# 自定义脚本
python train_astribot.py --policy act --steps 50000

# 官方命令
lerobot-train \
    --dataset.repo_id=astribot/demo \
    --dataset.root=./output \
    --policy.type=act \
    --steps=50000
```

## 🔍 可视化

```bash
python scripts/visualize.py ./output --repo-id astribot/demo --rerun
python scripts/visualize.py ./output --repo-id astribot/demo --export-video -o ./videos
```

## 🔄 帧同步

- 基准: head 相机 (30Hz)
- 关节: ±50ms
- 图像: ±100ms

## ⚙️ 依赖

```bash
pip install rosbags tqdm opencv-python-headless numpy
pip install matplotlib rerun-sdk  # 可视化
cd /root/lerobot && pip install -e .
```

## 📄 许可证

内部使用
