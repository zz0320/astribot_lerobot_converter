#!/usr/bin/env python3
"""
Astribot 批量转换脚本

并行转换多个 bag 文件到独立的 LeRobot 数据集 (25维特征)

用法:
    python batch.py /root/astribot_raw_datasets -o ./output --workers 4
"""

import sys
import json
import argparse
import traceback
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# 添加脚本目录到路径
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

# 从 core 导入
from core import (
    ASTRIBOT_FPS, ASTRIBOT_FEATURES, ASTRIBOT_ROBOT_TYPE,
    extract_bag_data, synchronize_data, convert_frame_to_lerobot,
    ParallelImageDecoder, load_episode_metadata,
)

# LeRobot
LEROBOT_PATH = '/root/lerobot/src'
if LEROBOT_PATH not in sys.path:
    sys.path.insert(0, LEROBOT_PATH)

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm


def find_bag_directories(root_dir: Path) -> list:
    """查找所有 rosbag 目录"""
    bag_dirs = []
    for item in sorted(root_dir.iterdir()):
        if item.is_dir():
            bag_path = item / "record" / "raw_data.bag"
            if bag_path.exists():
                bag_dirs.append(item)
    return bag_dirs


def convert_single_episode(
    bag_dir: Path,
    output_dir: Path,
    repo_id: str,
    task_description: str,
) -> int:
    """转换单个 episode"""
    bag_path = bag_dir / "record" / "raw_data.bag"
    episode_output = output_dir / bag_dir.name
    
    raw_data = extract_bag_data(bag_path, verbose=False)
    sync_frames = synchronize_data(raw_data, verbose=False)
    
    if not sync_frames:
        raise ValueError("无有效帧")
    
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=episode_output,
        robot_type=ASTRIBOT_ROBOT_TYPE,
        fps=ASTRIBOT_FPS,
        features=ASTRIBOT_FEATURES,
        image_writer_threads=4,
    )
    
    image_decoder = ParallelImageDecoder(max_workers=4)
    
    try:
        for frame in sync_frames:
            lerobot_frame = convert_frame_to_lerobot(frame, task_description, image_decoder)
            dataset.add_frame(lerobot_frame)
    finally:
        image_decoder.shutdown()
    
    dataset.save_episode()
    dataset.finalize()
    
    return len(sync_frames)


def convert_worker(args: tuple) -> dict:
    """Worker 函数"""
    bag_dir, output_dir, repo_id, task_description = args
    
    result = {
        'bag_dir': str(bag_dir),
        'episode_id': bag_dir.name,
        'success': False,
        'error': None,
        'frames': 0,
        'duration': 0,
    }
    
    try:
        start = datetime.now()
        frames = convert_single_episode(bag_dir, output_dir, repo_id, task_description)
        result['success'] = True
        result['frames'] = frames
        result['duration'] = (datetime.now() - start).total_seconds()
    except Exception as e:
        result['error'] = str(e)
        result['traceback'] = traceback.format_exc()
    
    return result


def batch_convert(
    bag_root: Path,
    output_dir: Path,
    repo_id: str = "astribot/batch",
    task_description: str = "Astribot manipulation task",
    workers: int = 1,
    episodes: list = None,
):
    """批量转换"""
    
    print(f"\n{'='*70}")
    print(f"批量转换 (25维特征)")
    print(f"{'='*70}")
    print(f"输入: {bag_root}")
    print(f"输出: {output_dir}")
    print(f"并行: {workers}")
    print(f"{'='*70}\n")
    
    bag_dirs = find_bag_directories(bag_root)
    
    if episodes:
        bag_dirs = [d for d in bag_dirs if d.name in episodes]
    
    if not bag_dirs:
        print("错误: 未找到 rosbag 目录")
        return
    
    print(f"Episodes: {len(bag_dirs)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tasks = [(bag_dir, output_dir, repo_id, task_description) for bag_dir in bag_dirs]
    results = []
    
    if workers == 1:
        for i, task in enumerate(tasks):
            print(f"\n[{i+1}/{len(tasks)}] {task[0].name}")
            result = convert_worker(task)
            results.append(result)
            if result['success']:
                print(f"  ✓ {result['frames']} 帧, {result['duration']:.1f}s")
            else:
                print(f"  ✗ {result['error']}")
    else:
        print(f"启动 {workers} 进程...")
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_task = {executor.submit(convert_worker, t): t for t in tasks}
            completed = 0
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                completed += 1
                try:
                    result = future.result()
                    results.append(result)
                    status = f"✓ {result['frames']} 帧" if result['success'] else f"✗ {result['error']}"
                    print(f"[{completed}/{len(tasks)}] {task[0].name}: {status}")
                except Exception as e:
                    results.append({'bag_dir': str(task[0]), 'success': False, 'error': str(e)})
    
    # 汇总
    success = sum(1 for r in results if r['success'])
    total_frames = sum(r.get('frames', 0) for r in results)
    
    print(f"\n{'='*70}")
    print(f"完成: {success}/{len(results)}, {total_frames} 帧")
    
    # 保存报告
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_episodes': len(results),
        'successful': success,
        'total_frames': total_frames,
        'feature_dim': 25,
        'results': results,
    }
    
    report_path = output_dir / "conversion_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"报告: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='批量转换 (25维)')
    parser.add_argument('bag_root', type=Path)
    parser.add_argument('-o', '--output-dir', type=Path, default=None)
    parser.add_argument('--repo-id', type=str, default='astribot/batch')
    parser.add_argument('--task', type=str, default="Astribot manipulation task")
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--episodes', nargs='+', default=None)
    
    args = parser.parse_args()
    
    output_dir = args.output_dir or args.bag_root.parent / f"{args.bag_root.name}_lerobot_separate"
    
    batch_convert(
        bag_root=args.bag_root,
        output_dir=output_dir,
        repo_id=args.repo_id,
        task_description=args.task,
        workers=args.workers,
        episodes=args.episodes,
    )


if __name__ == '__main__':
    main()

