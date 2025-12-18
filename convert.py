#!/usr/bin/env python3
"""
Astribot ROS Bag 转换器 - 主入口

将 Astribot ROS1 bag 数据转换为 LeRobot 3.0 格式

用法:
    # 合并所有 episodes 到单个数据集 (推荐)
    python convert.py /root/astribot_raw_datasets -o ./output --repo-id astribot/dataset
    
    # 每个 episode 独立保存
    python convert.py /root/astribot_raw_datasets -o ./output --separate
"""

import sys
import argparse
from pathlib import Path

# 添加脚本目录到路径
SCRIPT_DIR = Path(__file__).parent / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))


def main():
    parser = argparse.ArgumentParser(
        description='将 Astribot ROS1 bag 数据转换为 LeRobot 3.0 格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 合并所有 episodes 到单个数据集 (推荐)
    python convert.py /root/astribot_raw_datasets -o ./output --repo-id astribot/dataset
    
    # 每个 episode 独立保存
    python convert.py /root/astribot_raw_datasets -o ./output --separate --workers 4
        """
    )
    
    parser.add_argument(
        'bag_root',
        type=Path,
        help='包含 rosbag 数据的根目录'
    )
    parser.add_argument(
        '-o', '--output-dir',
        type=Path,
        default=None,
        help='输出目录'
    )
    parser.add_argument(
        '--repo-id',
        type=str,
        default='astribot/dataset',
        help='LeRobot 数据集 repo_id (默认: astribot/dataset)'
    )
    parser.add_argument(
        '--task',
        type=str,
        default=None,
        help='全局任务描述 (强制所有 episode 使用相同描述)'
    )
    parser.add_argument(
        '--no-episode-tasks',
        action='store_true',
        help='禁用从元数据自动读取任务描述'
    )
    parser.add_argument(
        '--separate',
        action='store_true',
        help='每个 episode 独立保存为单独的数据集'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='并行进程数 (仅用于 --separate 模式)'
    )
    
    args = parser.parse_args()
    
    if args.separate:
        # 分开保存模式
        from batch_convert import batch_convert
        
        output_dir = args.output_dir
        if output_dir is None:
            output_dir = args.bag_root.parent / f"{args.bag_root.name}_lerobot_separate"
        
        batch_convert(
            bag_root=args.bag_root,
            output_dir=output_dir,
            task_description=args.task,
            workers=args.workers,
        )
    else:
        # 合并为单个数据集模式 (默认)
        from convert_merged import convert_all_to_single_dataset
        
        output_dir = args.output_dir
        if output_dir is None:
            output_dir = Path('./astribot_lerobot_dataset')
        
        convert_all_to_single_dataset(
            bag_root=args.bag_root,
            output_dir=output_dir,
            repo_id=args.repo_id,
            task_description=args.task,
            use_episode_tasks=not args.no_episode_tasks,
        )


if __name__ == '__main__':
    main()

