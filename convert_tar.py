#!/usr/bin/env python3
"""
Astribot Tar 文件转换器

从 tar 文件中读取数据并转换为 LeRobot 格式
支持的数据结构：
    - meta_info.json
    - record/raw_data.bag
    - __loongdata_metadata.json

用法:
    # 转换单个 tar 文件
    python convert_tar.py /root/datasets/astribot_rawdata/s1a12e3edd74401abdbb6af11241b2a6/astribot_test2_s1a12e3edd74401abdbb6af11241b2a6.tar -o ./output
    
    # 转换目录下所有 tar 文件
    python convert_tar.py /root/datasets/astribot_rawdata -o ./output --repo-id astribot/dataset
"""

import sys
import argparse
import json
import tarfile
import tempfile
from pathlib import Path
from typing import Optional

# 添加脚本目录到路径
SCRIPT_DIR = Path(__file__).parent / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from convert_single import convert_single_bag


def extract_tar_to_temp(tar_path: Path) -> Path:
    """将 tar 文件解压到临时目录"""
    temp_dir = Path(tempfile.mkdtemp(prefix="astribot_tar_"))
    
    print(f"解压 tar 文件: {tar_path.name}")
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(temp_dir)
    
    return temp_dir


def read_metadata_from_tar(tar_path: Path) -> dict:
    """从 tar 文件中读取元数据"""
    metadata = {}
    
    try:
        with tarfile.open(tar_path, 'r') as tar:
            # 读取 meta_info.json
            try:
                meta_info_file = tar.extractfile('meta_info.json')
                if meta_info_file:
                    metadata['meta_info'] = json.load(meta_info_file)
            except KeyError:
                pass
            
            # 读取 __loongdata_metadata.json
            try:
                loong_meta_file = tar.extractfile('__loongdata_metadata.json')
                if loong_meta_file:
                    metadata['loong_metadata'] = json.load(loong_meta_file)
            except KeyError:
                pass
    except Exception as e:
        print(f"警告: 读取元数据失败: {e}")
    
    return metadata


def get_task_description_from_metadata(metadata: dict) -> Optional[str]:
    """从元数据中提取任务描述"""
    # 优先从 __loongdata_metadata.json 获取
    if 'loong_metadata' in metadata:
        loong_meta = metadata['loong_metadata']
        task_name = loong_meta.get('taskName', '')
        if task_name:
            return task_name
    
    # 从 meta_info.json 获取
    if 'meta_info' in metadata:
        meta_info = metadata['meta_info']
        # 可以根据实际结构提取任务描述
        pass
    
    return None


def convert_tar_file(
    tar_path: Path,
    output_dir: Path,
    task_description: Optional[str] = None,
    episode_id: Optional[str] = None,
    verbose: bool = True,
) -> int:
    """转换单个 tar 文件到 LeRobot 格式"""
    tar_path = Path(tar_path)
    
    if not tar_path.exists():
        raise FileNotFoundError(f"Tar 文件不存在: {tar_path}")
    
    if not tar_path.suffix == '.tar':
        raise ValueError(f"不是 tar 文件: {tar_path}")
    
    # 读取元数据
    metadata = read_metadata_from_tar(tar_path)
    
    # 确定 episode_id
    if episode_id is None:
        if 'loong_metadata' in metadata:
            episode_id = metadata['loong_metadata'].get('episode', tar_path.stem)
        else:
            episode_id = tar_path.stem
    
    # 确定任务描述
    if task_description is None:
        task_description = get_task_description_from_metadata(metadata)
        if task_description is None:
            task_description = "Astribot manipulation task"
    
    # 解压到临时目录
    temp_dir = None
    try:
        temp_dir = extract_tar_to_temp(tar_path)
        bag_dir = temp_dir
        
        # 检查 bag 文件是否存在
        bag_path = bag_dir / "record" / "raw_data.bag"
        if not bag_path.exists():
            raise FileNotFoundError(f"Bag 文件不存在于 tar 中: record/raw_data.bag")
        
        if verbose:
            print(f"Episode ID: {episode_id}")
            print(f"任务描述: {task_description}")
            if 'loong_metadata' in metadata:
                loong_meta = metadata['loong_metadata']
                print(f"设备: {loong_meta.get('equipmentModel', 'N/A')}")
                print(f"持续时间: {loong_meta.get('duration', 0) / 1000:.2f} 秒")
        
        # 使用现有的转换函数
        frames = convert_single_bag(
            bag_dir=bag_dir,
            output_dir=output_dir,
            task_description=task_description,
            episode_id=episode_id,
            verbose=verbose,
        )
        
        return frames
    
    finally:
        # 清理临时目录
        if temp_dir and temp_dir.exists():
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


def find_tar_files(directory: Path) -> list:
    """在目录中查找所有 tar 文件"""
    tar_files = []
    
    # 递归查找所有 .tar 文件
    for tar_file in directory.rglob("*.tar"):
        tar_files.append(tar_file)
    
    return sorted(tar_files)


def convert_tars_to_single_dataset(
    tar_files: list,
    output_dir: Path,
    repo_id: str,
    task_description: Optional[str] = None,
    use_episode_tasks: bool = True,
):
    """
    将多个 tar 文件转换为单个 LeRobot 数据集
    
    Args:
        tar_files: tar 文件路径列表
        output_dir: 输出目录
        repo_id: LeRobot 数据集 ID
        task_description: 全局任务描述 (如果设置，所有 episode 使用相同描述)
        use_episode_tasks: 是否为每个 episode 使用独立的任务描述
    """
    import sys
    from datetime import datetime
    import json
    
    # LeRobot 路径
    LEROBOT_PATH = '/root/lerobot/src'
    if LEROBOT_PATH not in sys.path:
        sys.path.insert(0, LEROBOT_PATH)
    
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    
    # 从 convert_single 导入必要的函数和配置
    from convert_single import (
        ASTRIBOT_FPS, ASTRIBOT_ROBOT_TYPE, ASTRIBOT_FEATURES,
        extract_bag_data, synchronize_data, ParallelImageDecoder,
        ARM_JOINTS, GRIPPER_JOINTS
    )
    import numpy as np
    from tqdm import tqdm
    
    print(f"\n{'='*70}")
    print(f"Astribot Tar -> LeRobot 单数据集转换")
    print(f"{'='*70}")
    print(f"输入: {len(tar_files)} 个 tar 文件")
    print(f"输出目录: {output_dir}")
    print(f"Repo ID: {repo_id}")
    print(f"任务模式: {'每个 episode 独立任务' if use_episode_tasks and not task_description else '全局统一任务'}")
    print(f"{'='*70}\n")
    
    # 读取所有 tar 文件的元数据
    tar_metadata = []
    for tar_file in tar_files:
        metadata = read_metadata_from_tar(tar_file)
        episode_id = None
        if 'loong_metadata' in metadata:
            episode_id = metadata['loong_metadata'].get('episode', tar_file.stem)
        else:
            episode_id = tar_file.stem
        
        task = get_task_description_from_metadata(metadata)
        if task is None:
            task = "Astribot manipulation task"
        
        tar_metadata.append({
            'tar_path': tar_file,
            'episode_id': episode_id,
            'task': task,
            'metadata': metadata,
        })
    
    print(f"找到 {len(tar_metadata)} 个 episode:")
    for item in tar_metadata:
        print(f"  - {item['episode_id']}")
        print(f"    任务描述: {item['task']}")
    print()
    
    # 创建 LeRobot 数据集
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=output_dir,
        robot_type=ASTRIBOT_ROBOT_TYPE,
        fps=ASTRIBOT_FPS,
        features=ASTRIBOT_FEATURES,
        image_writer_threads=4,
    )
    
    # 创建并行图像解码器
    image_decoder = ParallelImageDecoder(max_workers=4)
    
    total_frames = 0
    episode_info = []
    temp_dirs = []
    
    try:
        # 处理每个 tar 文件
        for episode_idx, item in enumerate(tar_metadata):
            tar_path = item['tar_path']
            episode_id = item['episode_id']
            
            # 确定当前 episode 的任务描述
            if task_description:
                current_task = task_description
            elif use_episode_tasks:
                current_task = item['task']
            else:
                current_task = task_description or "Astribot manipulation task"
            
            print(f"\n[Episode {episode_idx}] 处理: {episode_id}")
            print(f"  任务描述: {current_task}")
            
            # 解压到临时目录
            temp_dir = extract_tar_to_temp(tar_path)
            temp_dirs.append(temp_dir)
            bag_dir = temp_dir
            
            # 检查 bag 文件是否存在
            bag_path = bag_dir / "record" / "raw_data.bag"
            if not bag_path.exists():
                print(f"  警告: Bag 文件不存在，跳过")
                continue
            
            # 1. 提取数据
            raw_data = extract_bag_data(bag_path, verbose=True)
            
            # 2. 同步数据 (组帧)
            sync_frames = synchronize_data(raw_data, verbose=True)
            
            if not sync_frames:
                print(f"  警告: 没有有效帧，跳过")
                continue
            
            # 3. 转换帧并添加到数据集
            print(f"  转换 {len(sync_frames)} 帧...")
            
            for frame in tqdm(sync_frames, desc="  转换帧"):
                lerobot_frame = {'task': current_task}
                
                als = frame['arm_left_state']
                ars = frame['arm_right_state']
                gls = frame['gripper_left_state']
                grs = frame['gripper_right_state']
                
                lerobot_frame['observation.state.arm_left.position'] = als['position'][:ARM_JOINTS]
                lerobot_frame['observation.state.arm_left.velocity'] = als['velocity'][:ARM_JOINTS]
                lerobot_frame['observation.state.arm_left.torque'] = als['torque'][:ARM_JOINTS]
                lerobot_frame['observation.state.arm_right.position'] = ars['position'][:ARM_JOINTS]
                lerobot_frame['observation.state.arm_right.velocity'] = ars['velocity'][:ARM_JOINTS]
                lerobot_frame['observation.state.arm_right.torque'] = ars['torque'][:ARM_JOINTS]
                
                gl_pos = np.array([gls['position'][0] if gls and len(gls['position']) > 0 else 0.0], dtype=np.float32)
                gr_pos = np.array([grs['position'][0] if grs and len(grs['position']) > 0 else 0.0], dtype=np.float32)
                lerobot_frame['observation.state.gripper_left.position'] = gl_pos
                lerobot_frame['observation.state.gripper_right.position'] = gr_pos
                
                lerobot_frame['observation.state'] = np.concatenate([
                    als['position'][:ARM_JOINTS], ars['position'][:ARM_JOINTS], gl_pos, gr_pos
                ])
                
                alc = frame['arm_left_cmd']
                arc = frame['arm_right_cmd']
                glc = frame['gripper_left_cmd']
                grc = frame['gripper_right_cmd']
                
                lerobot_frame['action.arm_left'] = alc['command'][:ARM_JOINTS] if len(alc['command']) >= ARM_JOINTS else np.zeros(ARM_JOINTS, dtype=np.float32)
                lerobot_frame['action.arm_right'] = arc['command'][:ARM_JOINTS] if len(arc['command']) >= ARM_JOINTS else np.zeros(ARM_JOINTS, dtype=np.float32)
                
                gl_cmd = np.array([glc['command'][0] if glc and len(glc['command']) > 0 else 0.0], dtype=np.float32)
                gr_cmd = np.array([grc['command'][0] if grc and len(grc['command']) > 0 else 0.0], dtype=np.float32)
                lerobot_frame['action.gripper_left'] = gl_cmd
                lerobot_frame['action.gripper_right'] = gr_cmd
                
                lerobot_frame['action'] = np.concatenate([
                    lerobot_frame['action.arm_left'], lerobot_frame['action.arm_right'], gl_cmd, gr_cmd
                ])
                
                # 并行解码图像
                decoded_images = image_decoder.decode_batch(frame.get('images', {}))
                for img_key, img_data in decoded_images.items():
                    lerobot_frame[f'observation.images.{img_key}'] = img_data
                
                dataset.add_frame(lerobot_frame)
            
            # 4. 保存当前 episode
            dataset.save_episode()
            
            # 记录帧数
            ep_frames = len(sync_frames)
            total_frames += ep_frames
            
            episode_info.append({
                'episode_index': episode_idx,
                'source': episode_id,
                'task': current_task,
                'frames': ep_frames,
            })
            
            print(f"  ✓ Episode {episode_idx} 完成: {ep_frames} 帧")
            
            # 及时释放内存
            del raw_data
            del sync_frames
    
    finally:
        # 关闭图像解码器
        image_decoder.shutdown()
        
        # 清理临时目录
        import shutil
        for temp_dir in temp_dirs:
            if temp_dir and temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
    
    # 完成数据集
    dataset.finalize()
    
    # 收集所有唯一的任务描述
    unique_tasks = list(set(ep['task'] for ep in episode_info))
    
    # 生成转换报告
    report = {
        'timestamp': datetime.now().isoformat(),
        'repo_id': repo_id,
        'input_files': [str(tar_path) for tar_path in tar_files],
        'output_dir': str(output_dir),
        'total_episodes': len(episode_info),
        'total_frames': total_frames,
        'total_tasks': len(unique_tasks),
        'tasks': unique_tasks,
        'fps': ASTRIBOT_FPS,
        'robot_type': ASTRIBOT_ROBOT_TYPE,
        'episodes': episode_info,
    }
    
    report_path = output_dir / "conversion_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*70}")
    print(f"转换完成!")
    print(f"  总 episodes: {len(episode_info)}")
    print(f"  总帧数: {total_frames}")
    print(f"  任务类型: {len(unique_tasks)}")
    print(f"  输出目录: {output_dir}")
    print(f"  报告文件: {report_path}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='将 Astribot tar 文件转换为 LeRobot 3.0 格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 转换单个 tar 文件
    python convert_tar.py /root/datasets/astribot_rawdata/s1a12e3edd74401abdbb6af11241b2a6/astribot_test2_s1a12e3edd74401abdbb6af11241b2a6.tar -o ./output
    
    # 转换目录下所有 tar 文件
    python convert_tar.py /root/datasets/astribot_rawdata -o ./output --repo-id astribot/dataset
        """
    )
    
    parser.add_argument(
        'input_path',
        type=Path,
        help='tar 文件路径或包含 tar 文件的目录'
    )
    parser.add_argument(
        '-o', '--output-dir',
        type=Path,
        required=True,
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
        help='任务描述 (如果未指定，将从元数据中读取)'
    )
    parser.add_argument(
        '--episode-id',
        type=str,
        default=None,
        help='Episode ID (如果未指定，将从元数据中读取)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='显示详细输出'
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 确定要处理的 tar 文件
    if input_path.is_file() and input_path.suffix == '.tar':
        tar_files = [input_path]
    elif input_path.is_dir():
        tar_files = find_tar_files(input_path)
        if not tar_files:
            print(f"错误: 在目录 {input_path} 中未找到 tar 文件")
            return
        print(f"找到 {len(tar_files)} 个 tar 文件")
    else:
        print(f"错误: 输入路径必须是 tar 文件或目录: {input_path}")
        return
    
    # 转换每个 tar 文件
    total_frames = 0
    for i, tar_file in enumerate(tar_files, 1):
        print(f"\n{'='*60}")
        print(f"处理文件 {i}/{len(tar_files)}: {tar_file.name}")
        print(f"{'='*60}")
        
        try:
            frames = convert_tar_file(
                tar_path=tar_file,
                output_dir=output_dir,
                task_description=args.task,
                episode_id=args.episode_id,
                verbose=args.verbose,
            )
            total_frames += frames
            print(f"✓ 转换完成: {frames} 帧")
        except Exception as e:
            print(f"✗ 转换失败: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"全部完成: 共处理 {len(tar_files)} 个文件, {total_frames} 帧")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
