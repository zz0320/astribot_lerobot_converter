#!/usr/bin/env python3
"""
Astribot ROS1 Bag 批量转换脚本
支持并行转换多个 bag 文件到 LeRobot 3.0 格式

用法:
    python batch_convert.py /root/astribot_raw_datasets --repo-id astribot/dataset --workers 4
"""

import os
import sys
import json
import argparse
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

# 添加脚本目录到路径
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from convert_single import convert_single_bag, ASTRIBOT_FPS, ASTRIBOT_FEATURES, ASTRIBOT_ROBOT_TYPE


def find_bag_directories(root_dir: Path) -> list:
    """查找所有包含 rosbag 的目录"""
    bag_dirs = []
    for item in sorted(root_dir.iterdir()):
        if item.is_dir():
            bag_path = item / "record" / "raw_data.bag"
            if bag_path.exists():
                bag_dirs.append(item)
    return bag_dirs


def convert_worker(args: tuple) -> dict:
    """单个转换任务的 worker 函数"""
    bag_dir, output_dir, task_description, worker_id = args
    
    result = {
        'bag_dir': str(bag_dir),
        'success': False,
        'error': None,
        'frames': 0,
        'duration': 0,
    }
    
    try:
        start_time = datetime.now()
        
        # 转换单个 bag
        frames = convert_single_bag(
            bag_dir=bag_dir,
            output_dir=output_dir,
            task_description=task_description,
            episode_id=bag_dir.name,
        )
        
        end_time = datetime.now()
        
        result['success'] = True
        result['frames'] = frames
        result['duration'] = (end_time - start_time).total_seconds()
        
    except Exception as e:
        result['error'] = str(e)
        result['traceback'] = traceback.format_exc()
    
    return result


def batch_convert(
    bag_root: Path,
    output_dir: Path,
    task_description: str = "Astribot manipulation task",
    workers: int = 1,
    episodes: list = None,
):
    """批量转换多个 bag 文件"""
    
    print(f"\n{'='*70}")
    print(f"Astribot ROS Bag 批量转换")
    print(f"{'='*70}")
    print(f"输入目录: {bag_root}")
    print(f"输出目录: {output_dir}")
    print(f"并行数: {workers}")
    print(f"{'='*70}\n")
    
    # 查找所有 bag 目录
    bag_dirs = find_bag_directories(bag_root)
    
    if episodes:
        # 只处理指定的 episodes
        bag_dirs = [d for d in bag_dirs if d.name in episodes]
    
    if not bag_dirs:
        print("错误: 没有找到任何包含 rosbag 的目录")
        return
    
    print(f"找到 {len(bag_dirs)} 个 episode:")
    for d in bag_dirs:
        print(f"  - {d.name}")
    print()
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 准备任务参数
    tasks = [
        (bag_dir, output_dir, task_description, i)
        for i, bag_dir in enumerate(bag_dirs)
    ]
    
    results = []
    
    if workers == 1:
        # 串行处理
        for i, task in enumerate(tasks):
            print(f"\n[{i+1}/{len(tasks)}] 正在处理: {task[0].name}")
            result = convert_worker(task)
            results.append(result)
            
            if result['success']:
                print(f"  ✓ 完成: {result['frames']} 帧, 耗时 {result['duration']:.1f}s")
            else:
                print(f"  ✗ 失败: {result['error']}")
    else:
        # 并行处理
        print(f"启动 {workers} 个并行进程...")
        
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_task = {
                executor.submit(convert_worker, task): task
                for task in tasks
            }
            
            completed = 0
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                completed += 1
                
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result['success']:
                        print(f"[{completed}/{len(tasks)}] ✓ {task[0].name}: {result['frames']} 帧, {result['duration']:.1f}s")
                    else:
                        print(f"[{completed}/{len(tasks)}] ✗ {task[0].name}: {result['error']}")
                
                except Exception as e:
                    print(f"[{completed}/{len(tasks)}] ✗ {task[0].name}: 执行异常 - {e}")
                    results.append({
                        'bag_dir': str(task[0]),
                        'success': False,
                        'error': str(e),
                    })
    
    # 生成汇总报告
    print(f"\n{'='*70}")
    print("转换完成汇总")
    print(f"{'='*70}")
    
    success_count = sum(1 for r in results if r['success'])
    total_frames = sum(r.get('frames', 0) for r in results)
    total_duration = sum(r.get('duration', 0) for r in results)
    
    print(f"成功: {success_count}/{len(results)}")
    print(f"总帧数: {total_frames}")
    print(f"总耗时: {total_duration:.1f}s")
    
    if success_count < len(results):
        print(f"\n失败的转换:")
        for r in results:
            if not r['success']:
                print(f"  - {r['bag_dir']}: {r['error']}")
    
    # 保存转换报告
    report_path = output_dir / "conversion_report.json"
    report = {
        'timestamp': datetime.now().isoformat(),
        'input_dir': str(bag_root),
        'output_dir': str(output_dir),
        'total_episodes': len(results),
        'successful': success_count,
        'total_frames': total_frames,
        'total_duration': total_duration,
        'results': results,
    }
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n报告已保存: {report_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='批量转换 Astribot ROS1 bag 数据到独立的 LeRobot 格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 转换所有 episodes (串行)
    python batch_convert.py /root/astribot_raw_datasets -o ./output
    
    # 并行转换 (4 个进程)
    python batch_convert.py /root/astribot_raw_datasets -o ./output --workers 4
    
    # 只转换指定的 episodes
    python batch_convert.py /root/astribot_raw_datasets -o ./output --episodes ep1 ep2 ep3
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
        help='输出目录 (默认: bag_root_lerobot)'
    )
    parser.add_argument(
        '--task',
        type=str,
        default="Astribot manipulation task",
        help='任务描述'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='并行进程数 (默认: 1, 串行处理)'
    )
    parser.add_argument(
        '--episodes',
        nargs='+',
        default=None,
        help='只转换指定的 episode 名称'
    )
    
    args = parser.parse_args()
    
    # 设置输出目录
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = args.bag_root.parent / f"{args.bag_root.name}_lerobot"
    
    batch_convert(
        bag_root=args.bag_root,
        output_dir=output_dir,
        task_description=args.task,
        workers=args.workers,
        episodes=args.episodes,
    )


if __name__ == '__main__':
    main()

