#!/usr/bin/env python3
"""
Astribot LeRobot 训练脚本示例

使用转换后的 LeRobot 数据集训练策略模型

用法:
    # 使用 ACT 策略训练
    python train_astribot.py --policy act --dataset-path /root/astribot_lerobot_dataset_v2

    # 使用 Diffusion 策略训练
    python train_astribot.py --policy diffusion --dataset-path /root/astribot_lerobot_dataset_v2

    # 或者使用官方 lerobot-train 命令
    lerobot-train \
        --dataset.repo_id=astribot/demo_v2 \
        --dataset.root=/root/astribot_lerobot_dataset_v2 \
        --policy.type=act \
        --steps=50000
"""

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

# 添加 LeRobot 路径
LEROBOT_PATH = '/root/lerobot/src'
if LEROBOT_PATH not in sys.path:
    sys.path.insert(0, LEROBOT_PATH)

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.factory import make_pre_post_processors


def make_delta_timestamps(delta_indices: list[int] | None, fps: int) -> list[float]:
    """将 delta indices 转换为时间戳"""
    if delta_indices is None:
        return [0.0]
    return [i / fps for i in delta_indices]


def train_act(
    dataset_path: Path,
    repo_id: str,
    output_dir: Path,
    training_steps: int = 50000,
    batch_size: int = 8,
    device: str = "cuda",
    log_freq: int = 100,
    save_freq: int = 10000,
):
    """使用 ACT 策略训练"""
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    
    print(f"\n{'='*70}")
    print("ACT Policy Training")
    print(f"{'='*70}")
    print(f"Dataset: {repo_id} @ {dataset_path}")
    print(f"Output: {output_dir}")
    print(f"Steps: {training_steps}, Batch: {batch_size}")
    print(f"{'='*70}\n")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    
    # 加载数据集元数据
    print("Loading dataset metadata...")
    dataset_metadata = LeRobotDatasetMetadata(repo_id, root=dataset_path)
    
    # 配置特征
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    
    print(f"Input features: {list(input_features.keys())}")
    print(f"Output features: {list(output_features.keys())}")
    
    # 创建策略
    print("Creating ACT policy...")
    cfg = ACTConfig(input_features=input_features, output_features=output_features)
    policy = ACTPolicy(cfg)
    policy.train()
    policy.to(device)
    
    # 创建预处理器
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)
    
    # 配置时间戳
    delta_timestamps = {
        "action": make_delta_timestamps(cfg.action_delta_indices, dataset_metadata.fps),
    }
    # 添加图像特征
    for k in cfg.image_features:
        delta_timestamps[k] = make_delta_timestamps(cfg.observation_delta_indices, dataset_metadata.fps)
    
    # 加载数据集
    print("Loading dataset...")
    dataset = LeRobotDataset(repo_id, root=dataset_path, delta_timestamps=delta_timestamps)
    print(f"Dataset size: {len(dataset)} frames, {dataset.num_episodes} episodes")
    
    # 创建数据加载器
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    
    # 优化器
    optimizer = cfg.get_optimizer_preset().build(policy.parameters())
    
    # 训练循环
    print(f"\nStarting training for {training_steps} steps...")
    step = 0
    done = False
    
    while not done:
        for batch in tqdm(dataloader, desc=f"Epoch", leave=False):
            batch = preprocessor(batch)
            loss, _ = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if step % log_freq == 0:
                print(f"Step {step:6d} | Loss: {loss.item():.4f}")
            
            if step > 0 and step % save_freq == 0:
                checkpoint_dir = output_dir / f"checkpoint-{step}"
                print(f"Saving checkpoint to {checkpoint_dir}...")
                policy.save_pretrained(checkpoint_dir)
                preprocessor.save_pretrained(checkpoint_dir)
                postprocessor.save_pretrained(checkpoint_dir)
            
            step += 1
            if step >= training_steps:
                done = True
                break
    
    # 保存最终模型
    print(f"\nSaving final model to {output_dir}...")
    policy.save_pretrained(output_dir)
    preprocessor.save_pretrained(output_dir)
    postprocessor.save_pretrained(output_dir)
    
    print("Training complete!")
    return policy


def train_diffusion(
    dataset_path: Path,
    repo_id: str,
    output_dir: Path,
    training_steps: int = 50000,
    batch_size: int = 8,
    device: str = "cuda",
    log_freq: int = 100,
    save_freq: int = 10000,
):
    """使用 Diffusion 策略训练"""
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    
    print(f"\n{'='*70}")
    print("Diffusion Policy Training")
    print(f"{'='*70}")
    print(f"Dataset: {repo_id} @ {dataset_path}")
    print(f"Output: {output_dir}")
    print(f"Steps: {training_steps}, Batch: {batch_size}")
    print(f"{'='*70}\n")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    
    # 加载数据集元数据
    print("Loading dataset metadata...")
    dataset_metadata = LeRobotDatasetMetadata(repo_id, root=dataset_path)
    
    # 配置特征
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    
    print(f"Input features: {list(input_features.keys())}")
    print(f"Output features: {list(output_features.keys())}")
    
    # 创建策略
    print("Creating Diffusion policy...")
    cfg = DiffusionConfig(input_features=input_features, output_features=output_features)
    policy = DiffusionPolicy(cfg)
    policy.train()
    policy.to(device)
    
    # 创建预处理器
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)
    
    # 配置时间戳
    delta_timestamps = {
        "observation.state": make_delta_timestamps(cfg.observation_delta_indices, dataset_metadata.fps),
        "action": make_delta_timestamps(cfg.action_delta_indices, dataset_metadata.fps),
    }
    # 添加图像特征
    for k in cfg.image_features:
        delta_timestamps[k] = make_delta_timestamps(cfg.observation_delta_indices, dataset_metadata.fps)
    
    # 加载数据集
    print("Loading dataset...")
    dataset = LeRobotDataset(repo_id, root=dataset_path, delta_timestamps=delta_timestamps)
    print(f"Dataset size: {len(dataset)} frames, {dataset.num_episodes} episodes")
    
    # 创建数据加载器
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    
    # 优化器
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    
    # 训练循环
    print(f"\nStarting training for {training_steps} steps...")
    step = 0
    done = False
    
    while not done:
        for batch in tqdm(dataloader, desc=f"Epoch", leave=False):
            batch = preprocessor(batch)
            loss, _ = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if step % log_freq == 0:
                print(f"Step {step:6d} | Loss: {loss.item():.4f}")
            
            if step > 0 and step % save_freq == 0:
                checkpoint_dir = output_dir / f"checkpoint-{step}"
                print(f"Saving checkpoint to {checkpoint_dir}...")
                policy.save_pretrained(checkpoint_dir)
                preprocessor.save_pretrained(checkpoint_dir)
                postprocessor.save_pretrained(checkpoint_dir)
            
            step += 1
            if step >= training_steps:
                done = True
                break
    
    # 保存最终模型
    print(f"\nSaving final model to {output_dir}...")
    policy.save_pretrained(output_dir)
    preprocessor.save_pretrained(output_dir)
    postprocessor.save_pretrained(output_dir)
    
    print("Training complete!")
    return policy


def main():
    parser = argparse.ArgumentParser(
        description='Train LeRobot policy on Astribot dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # ACT 策略 (推荐用于精细操作)
    python train_astribot.py --policy act --steps 50000

    # Diffusion 策略 (适合复杂动作)
    python train_astribot.py --policy diffusion --steps 100000

    # 使用自定义数据集路径
    python train_astribot.py --policy act \\
        --dataset-path /path/to/dataset \\
        --repo-id my_org/my_dataset

支持的策略:
    - act: Action Chunking Transformer (默认)
    - diffusion: Diffusion Policy
        """
    )
    
    parser.add_argument(
        '--policy', '-p',
        type=str,
        choices=['act', 'diffusion'],
        default='act',
        help='策略类型 (默认: act)'
    )
    parser.add_argument(
        '--dataset-path', '-d',
        type=Path,
        default=Path('/root/astribot_lerobot_dataset_v2'),
        help='数据集路径'
    )
    parser.add_argument(
        '--repo-id',
        type=str,
        default='astribot/demo_v2',
        help='数据集 repo_id'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=None,
        help='输出目录'
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=50000,
        help='训练步数 (默认: 50000)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='批次大小 (默认: 8)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='设备 (默认: cuda)'
    )
    parser.add_argument(
        '--log-freq',
        type=int,
        default=100,
        help='日志频率 (默认: 100)'
    )
    parser.add_argument(
        '--save-freq',
        type=int,
        default=10000,
        help='保存检查点频率 (默认: 10000)'
    )
    
    args = parser.parse_args()
    
    # 设置默认输出目录
    if args.output_dir is None:
        args.output_dir = Path(f'./outputs/astribot_{args.policy}')
    
    # 检查数据集
    if not args.dataset_path.exists():
        print(f"错误: 数据集路径不存在: {args.dataset_path}")
        print("请先运行转换脚本生成数据集:")
        print("  cd /root/astribot_lerobot_converter")
        print("  python convert.py /root/astribot_raw_datasets -o /root/astribot_lerobot_dataset_v2")
        sys.exit(1)
    
    # 开始训练
    if args.policy == 'act':
        train_act(
            dataset_path=args.dataset_path,
            repo_id=args.repo_id,
            output_dir=args.output_dir,
            training_steps=args.steps,
            batch_size=args.batch_size,
            device=args.device,
            log_freq=args.log_freq,
            save_freq=args.save_freq,
        )
    elif args.policy == 'diffusion':
        train_diffusion(
            dataset_path=args.dataset_path,
            repo_id=args.repo_id,
            output_dir=args.output_dir,
            training_steps=args.steps,
            batch_size=args.batch_size,
            device=args.device,
            log_freq=args.log_freq,
            save_freq=args.save_freq,
        )


if __name__ == '__main__':
    main()

