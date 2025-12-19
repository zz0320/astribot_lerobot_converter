#!/usr/bin/env python3
"""
Astribot ROS Bag 转换器 - 主入口

将 Astribot ROS1 bag 数据转换为 LeRobot 3.0 格式
支持目录和 tar 文件输入

用法:
    # 合并所有 episodes 到单个数据集 (推荐)
    python convert.py /root/astribot_raw_datasets -o ./output --repo-id astribot/dataset
    
    # 每个 episode 独立保存
    python convert.py /root/astribot_raw_datasets -o ./output --separate
    
    # 转换 tar 文件
    python convert.py /root/datasets/astribot_rawdata/s1a12e3edd74401abdbb6af11241b2a6/astribot_test2_s1a12e3edd74401abdbb6af11241b2a6.tar -o ./output
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
    
    # 转换 tar 文件
    python convert.py /root/datasets/astribot_rawdata/s1a12e3edd74401abdbb6af11241b2a6/astribot_test2_s1a12e3edd74401abdbb6af11241b2a6.tar -o ./output
        """
    )
    
    parser.add_argument(
        'input_path',
        type=Path,
        help='包含 rosbag 数据的根目录或 tar 文件路径'
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
    
    input_path = Path(args.input_path)
    
    # 检查是否是 tar 文件
    if input_path.is_file() and input_path.suffix == '.tar':
        # 使用 tar 文件转换器
        from convert_tar import convert_tar_file
        
        output_dir = args.output_dir
        if output_dir is None:
            output_dir = Path('./astribot_lerobot_dataset')
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"检测到 tar 文件，使用 tar 转换器")
        frames = convert_tar_file(
            tar_path=input_path,
            output_dir=output_dir,
            task_description=args.task,
            episode_id=None,
            verbose=True,
        )
        print(f"转换完成: {frames} 帧")
        return
    
    # 原有的目录处理逻辑
    bag_root = input_path
    
    if not bag_root.exists():
        print(f"错误: 输入路径不存在: {bag_root}")
        return
    
    if not bag_root.is_dir():
        print(f"错误: 输入路径必须是目录或 tar 文件: {bag_root}")
        return
    
    # 检查目录下是否有 tar 文件
    from convert_tar import find_tar_files
    tar_files = find_tar_files(bag_root)
    
    if tar_files:
        # 如果找到 tar 文件，使用 tar 转换逻辑
        print(f"检测到 {len(tar_files)} 个 tar 文件，使用 tar 转换器")
        
        if args.separate:
            # 分开保存模式 - 每个 tar 文件独立保存
            from convert_tar import convert_tar_file
            
            output_dir = args.output_dir
            if output_dir is None:
                output_dir = bag_root.parent / f"{bag_root.name}_lerobot_separate"
            else:
                output_dir = Path(output_dir)
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            total_frames = 0
            for i, tar_file in enumerate(tar_files, 1):
                print(f"\n处理文件 {i}/{len(tar_files)}: {tar_file.name}")
                try:
                    frames = convert_tar_file(
                        tar_path=tar_file,
                        output_dir=output_dir,
                        task_description=args.task,
                        episode_id=None,
                        verbose=True,
                    )
                    total_frames += frames
                    print(f"✓ 转换完成: {frames} 帧")
                except Exception as e:
                    print(f"✗ 转换失败: {e}")
                    import traceback
                    traceback.print_exc()
            
            print(f"\n全部完成: 共处理 {len(tar_files)} 个文件, {total_frames} 帧")
        else:
            # 合并为单个数据集模式 - 需要创建一个新的合并函数
            from convert_tar import convert_tars_to_single_dataset
            
            output_dir = args.output_dir
            if output_dir is None:
                output_dir = Path('./astribot_lerobot_dataset')
            else:
                output_dir = Path(output_dir)
            
            convert_tars_to_single_dataset(
                tar_files=tar_files,
                output_dir=output_dir,
                repo_id=args.repo_id,
                task_description=args.task,
                use_episode_tasks=not args.no_episode_tasks,
            )
    else:
        # 没有找到 tar 文件，使用原有的目录转换逻辑
        if args.separate:
            # 分开保存模式
            from batch_convert import batch_convert
            
            output_dir = args.output_dir
            if output_dir is None:
                output_dir = bag_root.parent / f"{bag_root.name}_lerobot_separate"
            
            batch_convert(
                bag_root=bag_root,
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
                bag_root=bag_root,
                output_dir=output_dir,
                repo_id=args.repo_id,
                task_description=args.task,
                use_episode_tasks=not args.no_episode_tasks,
            )


if __name__ == '__main__':
    main()
