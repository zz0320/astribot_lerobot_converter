#!/usr/bin/env python3
"""
Astribot ROS Bag 转换器 - 主入口

将 Astribot ROS1 bag 数据转换为 LeRobot 3.0 格式 (25维完整特征)
自动检测输入类型 (rosbag 目录 / tar 文件)

用法:
    python convert.py /root/astribot_raw_datasets -o ./output --repo-id astribot/dataset
    python convert.py /path/to/data.tar -o ./output
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
        description='Astribot -> LeRobot 转换 (25维特征)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
特征维度: 25 (arm[7+7] + gripper[1+1] + head[2] + torso[4] + chassis[3])

示例:
    python convert.py /root/datasets -o ./output --repo-id astribot/demo
    python convert.py /path/to/data.tar -o ./output
    python convert.py /root/datasets -o ./output --separate --workers 4
        """
    )
    
    parser.add_argument('input_path', type=Path, help='rosbag 目录或 tar 文件')
    parser.add_argument('-o', '--output-dir', type=Path, default=None)
    parser.add_argument('--repo-id', type=str, default='astribot/dataset')
    parser.add_argument('--task', type=str, default=None)
    parser.add_argument('--no-episode-tasks', action='store_true')
    parser.add_argument('--separate', action='store_true')
    parser.add_argument('--workers', type=int, default=1)
    
    args = parser.parse_args()
    input_path = Path(args.input_path)
    
    def get_default_output(separate=False):
        if separate:
            return input_path.parent / f"{input_path.name}_lerobot_separate"
        return Path('./astribot_lerobot_dataset')
    
    # ========== 单个 tar 文件 ==========
    if input_path.is_file() and input_path.suffix == '.tar':
        from tar_converter import convert_tar_file
        
        output_dir = args.output_dir or get_default_output()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[模式] 单个 tar 文件 (25维)")
        frames = convert_tar_file(
            tar_path=input_path,
            output_dir=output_dir,
            repo_id=args.repo_id,
            task_description=args.task,
        )
        print(f"\n✓ 完成: {frames} 帧")
        return
    
    # ========== 目录 ==========
    if not input_path.exists():
        print(f"错误: 路径不存在: {input_path}")
        return
    
    if not input_path.is_dir():
        print(f"错误: 必须是目录或 tar 文件: {input_path}")
        return
    
    # 检查 tar 文件
    from tar_converter import find_tar_files
    tar_files = find_tar_files(input_path)
    
    if tar_files:
        # ========== 目录下有 tar ==========
        print(f"[模式] {len(tar_files)} 个 tar 文件 (25维)")
        output_dir = args.output_dir or get_default_output(args.separate)
        
        if args.separate:
            output_dir.mkdir(parents=True, exist_ok=True)
            from tar_converter import convert_tar_file
            total = 0
            for i, tar_file in enumerate(tar_files, 1):
                print(f"\n[{i}/{len(tar_files)}] {tar_file.name}")
                try:
                    frames = convert_tar_file(tar_path=tar_file, output_dir=output_dir, repo_id=args.repo_id, task_description=args.task)
                    total += frames
                    print(f"  ✓ {frames} 帧")
                except Exception as e:
                    print(f"  ✗ {e}")
            print(f"\n完成: {len(tar_files)} 文件, {total} 帧")
        else:
            from tar_converter import convert_tars_to_single_dataset
            convert_tars_to_single_dataset(
                tar_files=tar_files,
                output_dir=output_dir,
                repo_id=args.repo_id,
                task_description=args.task,
                use_episode_tasks=not args.no_episode_tasks,
            )
    else:
        # ========== 目录下是 rosbag ==========
        print(f"[模式] rosbag 目录 (25维)")
        output_dir = args.output_dir or get_default_output(args.separate)
        
        if args.separate:
            from batch import batch_convert
            batch_convert(
                bag_root=input_path,
                output_dir=output_dir,
                repo_id=args.repo_id,
                task_description=args.task,
                workers=args.workers,
            )
        else:
            from core import convert_all_to_single_dataset
            convert_all_to_single_dataset(
                bag_root=input_path,
                output_dir=output_dir,
                repo_id=args.repo_id,
                task_description=args.task,
                use_episode_tasks=not args.no_episode_tasks,
            )


if __name__ == '__main__':
    main()
