#!/usr/bin/env python3
"""
Astribot LeRobot 数据可视化脚本

用法:
    # 方法1: 使用 Rerun 可视化 (推荐)
    python visualize.py /root/astribot_lerobot_dataset_v2 --repo-id astribot/demo_v2 --episode 0
    
    # 方法2: 导出为视频
    python visualize.py /root/astribot_lerobot_dataset_v2 --repo-id astribot/demo_v2 --episode 0 --export-video
    
    # 方法3: 生成 matplotlib 图表
    python visualize.py /root/astribot_lerobot_dataset_v2 --repo-id astribot/demo_v2 --episode 0 --plot
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# LeRobot 路径
LEROBOT_PATH = '/root/lerobot/src'
if LEROBOT_PATH not in sys.path:
    sys.path.insert(0, LEROBOT_PATH)

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def visualize_with_rerun(dataset: LeRobotDataset, episode_index: int):
    """使用 Rerun 可视化数据"""
    import rerun as rr
    
    # 初始化 Rerun
    rr.init(f"{dataset.repo_id}/episode_{episode_index}", spawn=True)
    
    # 获取 episode 的帧范围
    ep_info = dataset.meta.episodes[episode_index]
    start_idx = sum(dataset.meta.episodes[i]['length'] for i in range(episode_index))
    end_idx = start_idx + ep_info['length']
    
    print(f"可视化 Episode {episode_index}: 帧 {start_idx} - {end_idx} (共 {ep_info['length']} 帧)")
    
    for frame_idx in range(start_idx, end_idx):
        sample = dataset[frame_idx]
        
        # 设置时间轴
        rr.set_time("frame", sequence=frame_idx - start_idx)
        rr.set_time("timestamp", timestamp=sample['timestamp'].item())
        
        # 记录图像
        for key in dataset.meta.camera_keys:
            if key in sample:
                img = sample[key].numpy()
                # CHW -> HWC
                if img.ndim == 3 and img.shape[0] in (1, 3, 4):
                    img = np.transpose(img, (1, 2, 0))
                # float -> uint8
                if img.dtype == np.float32:
                    img = (img * 255).astype(np.uint8)
                rr.log(key, rr.Image(img))
        
        # 记录状态
        if 'observation.state' in sample:
            state = sample['observation.state'].numpy()
            for i, val in enumerate(state):
                rr.log(f"state/joint_{i}", rr.Scalars(float(val)))
        
        # 记录动作
        if 'action' in sample:
            action = sample['action'].numpy()
            for i, val in enumerate(action):
                rr.log(f"action/joint_{i}", rr.Scalars(float(val)))
    
    print("可视化完成! Rerun 窗口已打开。")


def plot_episode_data(dataset: LeRobotDataset, episode_index: int, output_dir: Path = None):
    """使用 matplotlib 绘制数据图表"""
    
    ep_info = dataset.meta.episodes[episode_index]
    start_idx = sum(dataset.meta.episodes[i]['length'] for i in range(episode_index))
    end_idx = start_idx + ep_info['length']
    num_frames = ep_info['length']
    
    print(f"绘制 Episode {episode_index}: {num_frames} 帧")
    
    # 收集数据
    timestamps = []
    states = []
    actions = []
    
    for frame_idx in range(start_idx, end_idx):
        sample = dataset[frame_idx]
        timestamps.append(sample['timestamp'].item())
        
        if 'observation.state' in sample:
            states.append(sample['observation.state'].numpy())
        if 'action' in sample:
            actions.append(sample['action'].numpy())
    
    timestamps = np.array(timestamps)
    timestamps = timestamps - timestamps[0]  # 相对时间
    
    # 创建图表
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # 图1: 左臂关节位置
    if states:
        states = np.array(states)
        ax = axes[0]
        for i in range(7):
            ax.plot(timestamps, states[:, i], label=f'left_joint_{i}', alpha=0.8)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position')
        ax.set_title('Left Arm Joint Positions')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # 图2: 右臂关节位置
    if states:
        ax = axes[1]
        for i in range(7):
            ax.plot(timestamps, states[:, 7+i], label=f'right_joint_{i}', alpha=0.8)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position')
        ax.set_title('Right Arm Joint Positions')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # 图3: 夹爪位置
    if states:
        ax = axes[2]
        ax.plot(timestamps, states[:, 14], label='gripper_left', linewidth=2)
        ax.plot(timestamps, states[:, 15], label='gripper_right', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position')
        ax.set_title('Gripper Positions')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        output_path = output_dir / f"episode_{episode_index}_plot.png"
        plt.savefig(output_path, dpi=150)
        print(f"图表已保存: {output_path}")
    else:
        plt.show()
    
    plt.close()


def export_video(dataset: LeRobotDataset, episode_index: int, output_dir: Path, camera_key: str = None):
    """导出 episode 为视频"""
    import cv2
    
    ep_info = dataset.meta.episodes[episode_index]
    start_idx = sum(dataset.meta.episodes[i]['length'] for i in range(episode_index))
    end_idx = start_idx + ep_info['length']
    
    # 选择相机
    camera_keys = dataset.meta.camera_keys
    if camera_key is None:
        camera_key = camera_keys[0] if camera_keys else None
    
    if camera_key is None:
        print("没有找到相机数据!")
        return
    
    print(f"导出视频: {camera_key}")
    
    # 获取第一帧确定尺寸
    first_sample = dataset[start_idx]
    img = first_sample[camera_key].numpy()
    if img.ndim == 3 and img.shape[0] in (1, 3, 4):
        img = np.transpose(img, (1, 2, 0))
    h, w = img.shape[:2]
    
    # 创建视频写入器
    output_path = output_dir / f"episode_{episode_index}_{camera_key.replace('.', '_')}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, dataset.fps, (w, h))
    
    for frame_idx in range(start_idx, end_idx):
        sample = dataset[frame_idx]
        img = sample[camera_key].numpy()
        
        # CHW -> HWC
        if img.ndim == 3 and img.shape[0] in (1, 3, 4):
            img = np.transpose(img, (1, 2, 0))
        
        # float -> uint8
        if img.dtype == np.float32:
            img = (img * 255).astype(np.uint8)
        
        # RGB -> BGR (OpenCV)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        out.write(img)
    
    out.release()
    print(f"视频已保存: {output_path}")


def show_sample_images(dataset: LeRobotDataset, episode_index: int, frame_offset: int = 0):
    """显示单帧的所有相机图像"""
    ep_info = dataset.meta.episodes[episode_index]
    start_idx = sum(dataset.meta.episodes[i]['length'] for i in range(episode_index))
    frame_idx = start_idx + frame_offset
    
    sample = dataset[frame_idx]
    camera_keys = dataset.meta.camera_keys
    
    if not camera_keys:
        print("没有相机数据!")
        return
    
    n_cameras = len(camera_keys)
    fig, axes = plt.subplots(1, n_cameras, figsize=(5 * n_cameras, 5))
    
    if n_cameras == 1:
        axes = [axes]
    
    for ax, key in zip(axes, camera_keys):
        img = sample[key].numpy()
        # CHW -> HWC
        if img.ndim == 3 and img.shape[0] in (1, 3, 4):
            img = np.transpose(img, (1, 2, 0))
        # float -> uint8 for display
        if img.dtype == np.float32:
            img = (img * 255).astype(np.uint8)
        
        ax.imshow(img)
        ax.set_title(key.replace('observation.images.', ''))
        ax.axis('off')
    
    plt.suptitle(f"Episode {episode_index}, Frame {frame_offset}")
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Astribot LeRobot 数据可视化')
    parser.add_argument('root', type=Path, help='数据集根目录')
    parser.add_argument('--repo-id', type=str, required=True, help='数据集 repo_id')
    parser.add_argument('--episode', type=int, default=0, help='要可视化的 episode 索引')
    parser.add_argument('--frame', type=int, default=0, help='显示单帧时的帧偏移')
    
    parser.add_argument('--rerun', action='store_true', help='使用 Rerun 可视化')
    parser.add_argument('--plot', action='store_true', help='绘制数据图表')
    parser.add_argument('--export-video', action='store_true', help='导出视频')
    parser.add_argument('--show-frame', action='store_true', help='显示单帧图像')
    
    parser.add_argument('--camera', type=str, default=None, help='指定相机 (用于导出视频)')
    parser.add_argument('-o', '--output-dir', type=Path, default=None, help='输出目录')
    
    args = parser.parse_args()
    
    # 加载数据集
    print(f"加载数据集: {args.repo_id}")
    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=args.root,
        episodes=[args.episode],
    )
    
    print(f"数据集信息:")
    print(f"  - Episodes: {dataset.num_episodes}")
    print(f"  - Frames: {dataset.num_frames}")
    print(f"  - FPS: {dataset.fps}")
    print(f"  - Cameras: {dataset.meta.camera_keys}")
    
    output_dir = args.output_dir or Path('.')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.rerun:
        visualize_with_rerun(dataset, args.episode)
    elif args.plot:
        plot_episode_data(dataset, args.episode, output_dir)
    elif args.export_video:
        export_video(dataset, args.episode, output_dir, args.camera)
    elif args.show_frame:
        show_sample_images(dataset, args.episode, args.frame)
    else:
        # 默认显示单帧
        show_sample_images(dataset, args.episode, args.frame)


if __name__ == '__main__':
    main()

