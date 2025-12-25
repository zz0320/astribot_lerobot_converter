#!/usr/bin/env python3
"""
Astribot Tar 文件转换器

从 tar 文件中读取数据并转换为 LeRobot 格式 (25维特征)

用法:
    python tar_converter.py /path/to/data.tar -o ./output
    python tar_converter.py /path/to/tar_dir -o ./output --repo-id astribot/dataset
"""

import sys
import argparse
import json
import tarfile
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, List

from tqdm import tqdm

# 添加脚本目录到路径
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

# 从 core 导入
from core import (
    ASTRIBOT_FPS, ASTRIBOT_ROBOT_TYPE, ASTRIBOT_FEATURES,
    ARM_JOINTS, GRIPPER_JOINTS, HEAD_JOINTS, TORSO_JOINTS, CHASSIS_JOINTS,
    extract_bag_data, synchronize_data, convert_frame_to_lerobot,
    ParallelImageDecoder,
)

# LeRobot
LEROBOT_PATH = '/root/lerobot/src'
if LEROBOT_PATH not in sys.path:
    sys.path.insert(0, LEROBOT_PATH)

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def extract_tar_to_temp(tar_path: Path) -> Path:
    """解压 tar 到临时目录"""
    temp_dir = Path(tempfile.mkdtemp(prefix="astribot_tar_"))
    print(f"解压: {tar_path.name}")
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(temp_dir)
    return temp_dir


def read_metadata_from_tar(tar_path: Path) -> dict:
    """从 tar 读取元数据"""
    metadata = {}
    try:
        with tarfile.open(tar_path, 'r') as tar:
            try:
                f = tar.extractfile('meta_info.json')
                if f:
                    metadata['meta_info'] = json.load(f)
            except KeyError:
                pass
            try:
                f = tar.extractfile('__loongdata_metadata.json')
                if f:
                    metadata['loong_metadata'] = json.load(f)
            except KeyError:
                pass
    except Exception as e:
        print(f"警告: 读取元数据失败: {e}")
    return metadata


def get_task_from_metadata(metadata: dict) -> Optional[str]:
    """从元数据提取任务描述"""
    if 'loong_metadata' in metadata:
        loong = metadata['loong_metadata']
        task_name = loong.get('taskName', '')
        scene = loong.get('scene', '')
        if task_name:
            return f"{task_name} - 场景: {scene}" if scene else task_name
    return None


def find_tar_files(directory: Path) -> List[Path]:
    """查找目录下所有 tar 文件"""
    return sorted(directory.rglob("*.tar"))


def convert_tar_file(
    tar_path: Path,
    output_dir: Path,
    repo_id: str = "astribot/dataset",
    task_description: Optional[str] = None,
    episode_id: Optional[str] = None,
    verbose: bool = True,
) -> int:
    """转换单个 tar 文件"""
    tar_path = Path(tar_path)
    
    if not tar_path.exists():
        raise FileNotFoundError(f"文件不存在: {tar_path}")
    if tar_path.suffix != '.tar':
        raise ValueError(f"不是 tar 文件: {tar_path}")
    
    metadata = read_metadata_from_tar(tar_path)
    
    if episode_id is None:
        if 'loong_metadata' in metadata:
            episode_id = metadata['loong_metadata'].get('episode', tar_path.stem)
        else:
            episode_id = tar_path.stem
    
    if task_description is None:
        task_description = get_task_from_metadata(metadata) or "Astribot manipulation task"
    
    temp_dir = None
    try:
        temp_dir = extract_tar_to_temp(tar_path)
        bag_path = temp_dir / "record" / "raw_data.bag"
        
        if not bag_path.exists():
            raise FileNotFoundError("Bag 不存在: record/raw_data.bag")
        
        if verbose:
            print(f"Episode: {episode_id}")
            print(f"任务: {task_description}")
        
        raw_data = extract_bag_data(bag_path, verbose=verbose)
        sync_frames = synchronize_data(raw_data, verbose=verbose)
        
        if not sync_frames:
            raise ValueError("无有效帧")
        
        episode_output = output_dir / episode_id
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
            for frame in (tqdm(sync_frames, desc="转换") if verbose else sync_frames):
                lerobot_frame = convert_frame_to_lerobot(frame, task_description, image_decoder)
                dataset.add_frame(lerobot_frame)
        finally:
            image_decoder.shutdown()
        
        dataset.save_episode()
        dataset.finalize()
        
        return len(sync_frames)
    
    finally:
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


def convert_tars_to_single_dataset(
    tar_files: List[Path],
    output_dir: Path,
    repo_id: str,
    task_description: Optional[str] = None,
    use_episode_tasks: bool = True,
):
    """将多个 tar 文件转换为单个数据集"""
    
    print(f"\n{'='*70}")
    print(f"Tar -> LeRobot 转换 (25维特征)")
    print(f"{'='*70}")
    print(f"输入: {len(tar_files)} 个 tar 文件")
    print(f"输出: {output_dir}")
    print(f"Repo: {repo_id}")
    print(f"{'='*70}\n")
    
    # 收集元数据
    tar_metadata = []
    for tar_file in tar_files:
        metadata = read_metadata_from_tar(tar_file)
        episode_id = metadata.get('loong_metadata', {}).get('episode', tar_file.stem)
        task = get_task_from_metadata(metadata) or "Astribot manipulation task"
        tar_metadata.append({
            'tar_path': tar_file,
            'episode_id': episode_id,
            'task': task,
            'metadata': metadata,
        })
    
    print(f"Episodes: {len(tar_metadata)}")
    
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=output_dir,
        robot_type=ASTRIBOT_ROBOT_TYPE,
        fps=ASTRIBOT_FPS,
        features=ASTRIBOT_FEATURES,
        image_writer_threads=4,
    )
    
    image_decoder = ParallelImageDecoder(max_workers=4)
    total_frames = 0
    episode_info = []
    temp_dirs = []
    
    try:
        for idx, item in enumerate(tar_metadata):
            tar_path = item['tar_path']
            episode_id = item['episode_id']
            current_task = task_description if task_description else (item['task'] if use_episode_tasks else "Astribot manipulation task")
            
            print(f"\n[{idx}] {episode_id}: {current_task}")
            
            temp_dir = extract_tar_to_temp(tar_path)
            temp_dirs.append(temp_dir)
            
            bag_path = temp_dir / "record" / "raw_data.bag"
            if not bag_path.exists():
                print(f"  跳过 (无 bag)")
                continue
            
            raw_data = extract_bag_data(bag_path, verbose=True)
            sync_frames = synchronize_data(raw_data, verbose=True)
            
            if not sync_frames:
                print(f"  跳过 (无有效帧)")
                continue
            
            for frame in tqdm(sync_frames, desc="  转换"):
                lerobot_frame = convert_frame_to_lerobot(frame, current_task, image_decoder)
                dataset.add_frame(lerobot_frame)
            
            dataset.save_episode()
            
            ep_frames = len(sync_frames)
            total_frames += ep_frames
            episode_info.append({
                'episode_index': idx,
                'source': episode_id,
                'task': current_task,
                'frames': ep_frames,
            })
            
            print(f"  ✓ {ep_frames} 帧")
            del raw_data, sync_frames
    
    finally:
        image_decoder.shutdown()
        for temp_dir in temp_dirs:
            if temp_dir and temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
    
    dataset.finalize()
    
    # 保存报告
    report = {
        'timestamp': datetime.now().isoformat(),
        'repo_id': repo_id,
        'total_episodes': len(episode_info),
        'total_frames': total_frames,
        'feature_dim': 25,
        'fps': ASTRIBOT_FPS,
        'episodes': episode_info,
    }
    
    report_path = output_dir / "conversion_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 完成: {len(episode_info)} episodes, {total_frames} 帧")
    print(f"  输出: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Tar -> LeRobot 转换 (25维)')
    parser.add_argument('input_path', type=Path, help='tar 文件或目录')
    parser.add_argument('-o', '--output-dir', type=Path, required=True)
    parser.add_argument('--repo-id', type=str, default='astribot/dataset')
    parser.add_argument('--task', type=str, default=None)
    parser.add_argument('--no-episode-tasks', action='store_true')
    
    args = parser.parse_args()
    input_path = Path(args.input_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    if input_path.is_file() and input_path.suffix == '.tar':
        frames = convert_tar_file(
            tar_path=input_path,
            output_dir=args.output_dir,
            repo_id=args.repo_id,
            task_description=args.task,
        )
        print(f"\n✓ 完成: {frames} 帧")
    elif input_path.is_dir():
        tar_files = find_tar_files(input_path)
        if not tar_files:
            print(f"错误: 未找到 tar 文件")
            return
        convert_tars_to_single_dataset(
            tar_files=tar_files,
            output_dir=args.output_dir,
            repo_id=args.repo_id,
            task_description=args.task,
            use_episode_tasks=not args.no_episode_tasks,
        )
    else:
        print(f"错误: 无效路径: {input_path}")


if __name__ == '__main__':
    main()

