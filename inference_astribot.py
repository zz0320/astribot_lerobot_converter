#!/usr/bin/env python3
"""
Astribot LeRobot 推理脚本

从训练好的模型进行推理，支持:
1. 单帧推理 - 给定观测，预测动作
2. 数据集回放推理 - 在数据集上运行模型，比较预测与真实动作
3. 实时推理 - 连接真实机器人进行控制 (需要额外配置)

用法:
    # 单帧推理测试
    python inference_astribot.py --checkpoint ./outputs/astribot_act --mode single

    # 数据集回放推理
    python inference_astribot.py --checkpoint ./outputs/astribot_act --mode dataset

    # 批量推理并保存结果
    python inference_astribot.py --checkpoint ./outputs/astribot_act --mode batch --output ./predictions
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from tqdm import tqdm

# 添加 LeRobot 路径
LEROBOT_PATH = '/root/lerobot/src'
if LEROBOT_PATH not in sys.path:
    sys.path.insert(0, LEROBOT_PATH)


def load_policy(checkpoint_path: Path, device: str = "cuda"):
    """加载训练好的策略模型"""
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.policies.factory import make_pre_post_processors
    
    checkpoint_path = Path(checkpoint_path)
    
    # 读取配置判断策略类型
    config_path = checkpoint_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    policy_type = config.get('type', 'act')
    print(f"Loading {policy_type} policy from {checkpoint_path}")
    
    # 加载策略
    if policy_type == 'act':
        policy = ACTPolicy.from_pretrained(str(checkpoint_path))
    elif policy_type == 'diffusion':
        policy = DiffusionPolicy.from_pretrained(str(checkpoint_path))
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")
    
    policy.eval()
    policy.to(device)
    
    # 加载预处理器
    from lerobot.processor import PolicyProcessorPipeline
    preprocessor = PolicyProcessorPipeline.from_pretrained(str(checkpoint_path), "preprocessor")
    postprocessor = PolicyProcessorPipeline.from_pretrained(str(checkpoint_path), "postprocessor")
    
    return policy, preprocessor, postprocessor, policy_type


def create_dummy_observation(features: Dict[str, Any], device: str = "cuda") -> Dict[str, torch.Tensor]:
    """创建一个虚拟观测用于测试"""
    observation = {}
    
    for key, feat in features.items():
        if 'images' in key:
            # 图像特征
            shape = feat.shape  # (C, H, W) or (H, W, C)
            if len(shape) == 3:
                if shape[0] in [1, 3]:  # (C, H, W)
                    obs_shape = (1, 1, *shape)  # (batch, seq, C, H, W)
                else:  # (H, W, C)
                    obs_shape = (1, 1, shape[2], shape[0], shape[1])
            observation[key] = torch.randn(*obs_shape, device=device)
        elif 'state' in key:
            # 状态特征
            shape = feat.shape
            observation[key] = torch.randn(1, 1, *shape, device=device)
    
    return observation


def single_frame_inference(
    policy,
    preprocessor,
    postprocessor,
    observation: Dict[str, torch.Tensor],
) -> Dict[str, np.ndarray]:
    """
    单帧推理
    
    Args:
        policy: 策略模型
        preprocessor: 预处理器
        postprocessor: 后处理器
        observation: 观测字典
    
    Returns:
        动作字典
    """
    with torch.no_grad():
        # 预处理
        batch = preprocessor(observation)
        
        # 推理
        action = policy.select_action(batch)
        
        # 后处理
        action = postprocessor(action)
    
    # 转换为 numpy
    result = {}
    for key, value in action.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.cpu().numpy()
        else:
            result[key] = value
    
    return result


def dataset_inference(
    checkpoint_path: Path,
    dataset_path: Path,
    repo_id: str,
    device: str = "cuda",
    num_samples: int = 100,
    output_dir: Path = None,
):
    """
    在数据集上进行推理，比较预测动作与真实动作
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.datasets.utils import dataset_to_policy_features
    from lerobot.configs.types import FeatureType
    
    print(f"\n{'='*70}")
    print("Dataset Inference")
    print(f"{'='*70}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset: {repo_id} @ {dataset_path}")
    print(f"{'='*70}\n")
    
    # 加载策略
    policy, preprocessor, postprocessor, policy_type = load_policy(checkpoint_path, device)
    
    # 加载数据集元数据
    dataset_metadata = LeRobotDatasetMetadata(repo_id, root=dataset_path)
    
    # 获取 delta_timestamps
    if policy_type == 'act':
        from lerobot.policies.act.configuration_act import ACTConfig
        cfg = policy.config
    else:
        from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
        cfg = policy.config
    
    def make_delta_timestamps(delta_indices, fps):
        if delta_indices is None:
            return [0.0]
        return [i / fps for i in delta_indices]
    
    delta_timestamps = {
        "action": make_delta_timestamps(
            getattr(cfg, 'action_delta_indices', None), 
            dataset_metadata.fps
        ),
    }
    
    # 添加图像特征
    for k in getattr(cfg, 'image_features', []):
        delta_timestamps[k] = make_delta_timestamps(
            getattr(cfg, 'observation_delta_indices', None),
            dataset_metadata.fps
        )
    
    # 加载数据集
    print("Loading dataset...")
    dataset = LeRobotDataset(repo_id, root=dataset_path, delta_timestamps=delta_timestamps)
    
    # 限制样本数
    num_samples = min(num_samples, len(dataset))
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    print(f"Running inference on {num_samples} samples...")
    
    results = {
        'predicted_actions': [],
        'true_actions': [],
        'errors': [],
    }
    
    policy.reset()
    
    for idx in tqdm(indices, desc="Inference"):
        sample = dataset[idx]
        
        # 准备输入
        batch = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v 
                 for k, v in sample.items()}
        
        with torch.no_grad():
            batch = preprocessor(batch)
            predicted_action = policy.select_action(batch)
            predicted_action = postprocessor(predicted_action)
        
        # 获取真实动作
        true_action = sample.get('action', None)
        
        if true_action is not None:
            pred = predicted_action['action'].cpu().numpy() if isinstance(predicted_action['action'], torch.Tensor) else predicted_action['action']
            true = true_action.cpu().numpy() if isinstance(true_action, torch.Tensor) else true_action
            
            # 确保形状匹配
            if pred.shape != true.shape:
                pred = pred.flatten()[:true.flatten().shape[0]]
                true = true.flatten()
            
            error = np.mean(np.abs(pred - true))
            
            results['predicted_actions'].append(pred.tolist())
            results['true_actions'].append(true.tolist())
            results['errors'].append(float(error))
    
    # 计算统计
    mean_error = np.mean(results['errors'])
    std_error = np.std(results['errors'])
    
    print(f"\n{'='*50}")
    print(f"Results:")
    print(f"  Mean Absolute Error: {mean_error:.6f}")
    print(f"  Std Error: {std_error:.6f}")
    print(f"{'='*50}")
    
    # 保存结果
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        result_file = output_dir / "inference_results.json"
        with open(result_file, 'w') as f:
            json.dump({
                'mean_error': float(mean_error),
                'std_error': float(std_error),
                'num_samples': num_samples,
                'errors': results['errors'],
            }, f, indent=2)
        print(f"Results saved to {result_file}")
    
    return results


def real_robot_inference_example():
    """
    真实机器人推理示例 (伪代码)
    """
    example_code = '''
# 真实机器人推理示例 (需要根据实际机器人接口修改)

from your_robot_driver import AstribotS1  # 替换为实际驱动

# 加载策略
policy, preprocessor, postprocessor, _ = load_policy("./outputs/astribot_act")

# 连接机器人
robot = AstribotS1()
robot.connect()

# 推理循环
policy.reset()
try:
    while True:
        # 1. 获取观测
        images = robot.get_camera_images()  # Dict[str, np.ndarray]
        joint_states = robot.get_joint_states()  # np.ndarray
        
        # 2. 构建观测字典
        observation = {
            "observation.images.head": torch.from_numpy(images["head"]).permute(2, 0, 1).unsqueeze(0).unsqueeze(0),
            "observation.images.wrist_left": torch.from_numpy(images["wrist_left"]).permute(2, 0, 1).unsqueeze(0).unsqueeze(0),
            "observation.images.wrist_right": torch.from_numpy(images["wrist_right"]).permute(2, 0, 1).unsqueeze(0).unsqueeze(0),
            "observation.images.torso": torch.from_numpy(images["torso"]).permute(2, 0, 1).unsqueeze(0).unsqueeze(0),
            "observation.state": torch.from_numpy(joint_states).unsqueeze(0).unsqueeze(0),
        }
        
        # 3. 推理
        with torch.no_grad():
            batch = preprocessor(observation)
            action = policy.select_action(batch)
            action = postprocessor(action)
        
        # 4. 执行动作
        action_np = action["action"].cpu().numpy().squeeze()
        robot.set_joint_commands(action_np)
        
        # 5. 等待下一帧
        time.sleep(1/30)  # 30 FPS
        
except KeyboardInterrupt:
    pass
finally:
    robot.disconnect()
'''
    return example_code


def main():
    parser = argparse.ArgumentParser(
        description='Astribot LeRobot Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 单帧测试推理
    python inference_astribot.py --checkpoint ./outputs/astribot_act --mode single

    # 数据集批量推理
    python inference_astribot.py --checkpoint ./outputs/astribot_act --mode dataset

    # 查看真实机器人推理示例代码
    python inference_astribot.py --mode robot-example
        """
    )
    
    parser.add_argument(
        '--checkpoint', '-c',
        type=Path,
        default=Path('./outputs/astribot_act'),
        help='模型检查点路径'
    )
    parser.add_argument(
        '--mode', '-m',
        type=str,
        choices=['single', 'dataset', 'robot-example'],
        default='single',
        help='推理模式'
    )
    parser.add_argument(
        '--dataset-path', '-d',
        type=Path,
        default=Path('/root/astribot_lerobot_dataset_v2'),
        help='数据集路径 (用于 dataset 模式)'
    )
    parser.add_argument(
        '--repo-id',
        type=str,
        default='astribot/demo_v2',
        help='数据集 repo_id'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=None,
        help='输出目录'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=100,
        help='推理样本数 (用于 dataset 模式)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='设备'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'robot-example':
        print("\n" + "="*70)
        print("真实机器人推理示例代码")
        print("="*70)
        print(real_robot_inference_example())
        return
    
    if args.mode == 'single':
        print("\n" + "="*70)
        print("Single Frame Inference Test")
        print("="*70)
        
        # 检查检查点
        if not args.checkpoint.exists():
            print(f"错误: 检查点不存在: {args.checkpoint}")
            print("请先训练模型:")
            print("  python train_astribot.py --policy act --steps 50000")
            sys.exit(1)
        
        # 加载策略
        policy, preprocessor, postprocessor, policy_type = load_policy(args.checkpoint, args.device)
        
        print(f"\nPolicy type: {policy_type}")
        print(f"Input features: {list(policy.config.input_features.keys())}")
        print(f"Output features: {list(policy.config.output_features.keys())}")
        
        # 创建虚拟输入并推理
        print("\nCreating dummy observation...")
        dummy_obs = create_dummy_observation(policy.config.input_features, args.device)
        
        print("Running inference...")
        policy.reset()
        
        with torch.no_grad():
            batch = preprocessor(dummy_obs)
            action = policy.select_action(batch)
            action = postprocessor(action)
        
        print("\nPredicted action:")
        for key, value in action.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: shape={value.shape}, range=[{value.min():.4f}, {value.max():.4f}]")
            else:
                print(f"  {key}: {value}")
        
        print("\n✓ Single frame inference successful!")
    
    elif args.mode == 'dataset':
        # 检查检查点和数据集
        if not args.checkpoint.exists():
            print(f"错误: 检查点不存在: {args.checkpoint}")
            sys.exit(1)
        
        if not args.dataset_path.exists():
            print(f"错误: 数据集不存在: {args.dataset_path}")
            sys.exit(1)
        
        dataset_inference(
            checkpoint_path=args.checkpoint,
            dataset_path=args.dataset_path,
            repo_id=args.repo_id,
            device=args.device,
            num_samples=args.num_samples,
            output_dir=args.output,
        )


if __name__ == '__main__':
    main()

