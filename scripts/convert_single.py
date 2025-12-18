#!/usr/bin/env python3
"""
单个 Astribot ROS1 Bag 转换模块 - 优化版
"""

import os
import sys
import struct
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore
from tqdm import tqdm
import cv2

# LeRobot 路径
LEROBOT_PATH = '/root/lerobot/src'
if LEROBOT_PATH not in sys.path:
    sys.path.insert(0, LEROBOT_PATH)

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ============================================================================
# 配置
# ============================================================================

ASTRIBOT_FPS = 30
ASTRIBOT_ROBOT_TYPE = "astribot_s1"
ARM_JOINTS = 7
GRIPPER_JOINTS = 1

JOINT_TOLERANCE_NS = 50_000_000
IMAGE_TOLERANCE_NS = 100_000_000

ASTRIBOT_FEATURES = {
    "observation.state.arm_left.position": {
        "dtype": "float32", "shape": (ARM_JOINTS,),
        "names": {"axes": [f"joint_{i}" for i in range(ARM_JOINTS)]},
    },
    "observation.state.arm_left.velocity": {
        "dtype": "float32", "shape": (ARM_JOINTS,),
        "names": {"axes": [f"joint_{i}" for i in range(ARM_JOINTS)]},
    },
    "observation.state.arm_left.torque": {
        "dtype": "float32", "shape": (ARM_JOINTS,),
        "names": {"axes": [f"joint_{i}" for i in range(ARM_JOINTS)]},
    },
    "observation.state.arm_right.position": {
        "dtype": "float32", "shape": (ARM_JOINTS,),
        "names": {"axes": [f"joint_{i}" for i in range(ARM_JOINTS)]},
    },
    "observation.state.arm_right.velocity": {
        "dtype": "float32", "shape": (ARM_JOINTS,),
        "names": {"axes": [f"joint_{i}" for i in range(ARM_JOINTS)]},
    },
    "observation.state.arm_right.torque": {
        "dtype": "float32", "shape": (ARM_JOINTS,),
        "names": {"axes": [f"joint_{i}" for i in range(ARM_JOINTS)]},
    },
    "observation.state.gripper_left.position": {
        "dtype": "float32", "shape": (GRIPPER_JOINTS,),
        "names": {"axes": ["gripper"]},
    },
    "observation.state.gripper_right.position": {
        "dtype": "float32", "shape": (GRIPPER_JOINTS,),
        "names": {"axes": ["gripper"]},
    },
    "observation.state": {
        "dtype": "float32", "shape": (16,),
        "names": {"axes": [f"arm_left_joint_{i}" for i in range(ARM_JOINTS)] + 
                         [f"arm_right_joint_{i}" for i in range(ARM_JOINTS)] +
                         ["gripper_left", "gripper_right"]},
    },
    "observation.images.head": {
        "dtype": "video", "shape": (720, 1280, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.images.wrist_left": {
        "dtype": "video", "shape": (360, 640, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.images.wrist_right": {
        "dtype": "video", "shape": (360, 640, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.images.torso": {
        "dtype": "video", "shape": (720, 1280, 3),
        "names": ["height", "width", "channels"],
    },
    "action.arm_left": {
        "dtype": "float32", "shape": (ARM_JOINTS,),
        "names": {"axes": [f"joint_{i}" for i in range(ARM_JOINTS)]},
    },
    "action.arm_right": {
        "dtype": "float32", "shape": (ARM_JOINTS,),
        "names": {"axes": [f"joint_{i}" for i in range(ARM_JOINTS)]},
    },
    "action.gripper_left": {
        "dtype": "float32", "shape": (GRIPPER_JOINTS,),
        "names": {"axes": ["gripper"]},
    },
    "action.gripper_right": {
        "dtype": "float32", "shape": (GRIPPER_JOINTS,),
        "names": {"axes": ["gripper"]},
    },
    "action": {
        "dtype": "float32", "shape": (16,),
        "names": {"axes": [f"arm_left_joint_{i}" for i in range(ARM_JOINTS)] + 
                         [f"arm_right_joint_{i}" for i in range(ARM_JOINTS)] +
                         ["gripper_left", "gripper_right"]},
    },
}

IMAGE_SHAPES = {
    'head': (720, 1280, 3),
    'wrist_left': (360, 640, 3),
    'wrist_right': (360, 640, 3),
    'torso': (720, 1280, 3),
}


# ============================================================================
# 高性能消息解析
# ============================================================================

class FastMessageParser:
    __slots__ = ()
    
    @staticmethod
    def read_header_fast(data: bytes, offset: int = 0) -> Tuple[int, int]:
        seq, secs, nsecs = struct.unpack_from('<III', data, offset)
        offset += 12
        str_len = struct.unpack_from('<I', data, offset)[0]
        offset += 4 + str_len
        return secs * 1_000_000_000 + nsecs, offset
    
    @staticmethod
    def read_float64_array_zero_copy(data: bytes, offset: int) -> Tuple[np.ndarray, int]:
        length = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        if length > 0:
            arr = np.frombuffer(data, dtype='<f8', count=length, offset=offset)
            return arr, offset + length * 8
        return np.empty(0, dtype=np.float64), offset
    
    @staticmethod
    def skip_string_array(data: bytes, offset: int) -> int:
        count = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        for _ in range(count):
            str_len = struct.unpack_from('<I', data, offset)[0]
            offset += 4 + str_len
        return offset


def parse_joint_state_fast(raw_data: bytes) -> Optional[Dict[str, np.ndarray]]:
    try:
        offset = 0
        _, offset = FastMessageParser.read_header_fast(raw_data, offset)
        offset += 1
        offset = FastMessageParser.skip_string_array(raw_data, offset)
        position, offset = FastMessageParser.read_float64_array_zero_copy(raw_data, offset)
        velocity, offset = FastMessageParser.read_float64_array_zero_copy(raw_data, offset)
        _, offset = FastMessageParser.read_float64_array_zero_copy(raw_data, offset)
        torque, offset = FastMessageParser.read_float64_array_zero_copy(raw_data, offset)
        
        return {
            'position': position.astype(np.float32, copy=False),
            'velocity': velocity.astype(np.float32, copy=False),
            'torque': torque.astype(np.float32, copy=False),
        }
    except:
        return None


def parse_joint_controller_fast(raw_data: bytes) -> Optional[Dict[str, np.ndarray]]:
    try:
        offset = 0
        _, offset = FastMessageParser.read_header_fast(raw_data, offset)
        offset += 1
        offset = FastMessageParser.skip_string_array(raw_data, offset)
        command, _ = FastMessageParser.read_float64_array_zero_copy(raw_data, offset)
        return {'command': command.astype(np.float32, copy=False)}
    except:
        return None


# ============================================================================
# 时间戳索引
# ============================================================================

class TimestampIndex:
    __slots__ = ('data', 'timestamps', 'size')
    
    def __init__(self, data_list: List[Dict]):
        self.data = data_list
        self.size = len(data_list)
        if self.size > 0:
            self.timestamps = np.array([d['timestamp'] for d in data_list], dtype=np.int64)
        else:
            self.timestamps = np.empty(0, dtype=np.int64)
    
    def find_nearest(self, target_ts: int, tolerance_ns: int) -> Optional[Any]:
        if self.size == 0:
            return None
        
        idx = np.searchsorted(self.timestamps, target_ts)
        
        best_diff = tolerance_ns + 1
        best_idx = -1
        
        if idx > 0:
            diff = abs(self.timestamps[idx - 1] - target_ts)
            if diff < best_diff:
                best_diff = diff
                best_idx = idx - 1
        
        if idx < self.size:
            diff = abs(self.timestamps[idx] - target_ts)
            if diff < best_diff:
                best_diff = diff
                best_idx = idx
        
        if best_diff <= tolerance_ns:
            return self.data[best_idx]['data']
        return None


class ImageIndex:
    __slots__ = ('data_list', 'ts_to_idx', 'sorted_ts', 'size')
    
    def __init__(self, data_list: List[Dict]):
        self.data_list = data_list
        self.size = len(data_list)
        if self.size > 0:
            self.ts_to_idx = {d['timestamp']: i for i, d in enumerate(data_list)}
            self.sorted_ts = np.array(sorted(self.ts_to_idx.keys()), dtype=np.int64)
        else:
            self.ts_to_idx = {}
            self.sorted_ts = np.empty(0, dtype=np.int64)
    
    def get_nearest(self, target_ts: int, tolerance_ns: int) -> Optional[Dict]:
        if self.size == 0:
            return None
        
        idx = np.searchsorted(self.sorted_ts, target_ts)
        
        best_ts = None
        best_diff = tolerance_ns + 1
        
        if idx > 0:
            diff = abs(self.sorted_ts[idx - 1] - target_ts)
            if diff < best_diff:
                best_diff = diff
                best_ts = self.sorted_ts[idx - 1]
        
        if idx < len(self.sorted_ts):
            diff = abs(self.sorted_ts[idx] - target_ts)
            if diff < best_diff:
                best_diff = diff
                best_ts = self.sorted_ts[idx]
        
        if best_ts is not None and best_diff <= tolerance_ns:
            return self.data_list[self.ts_to_idx[best_ts]]
        return None


# ============================================================================
# 图像处理
# ============================================================================

def decode_image_rgb(img_bytes: bytes, target_shape: Tuple[int, int, int]) -> np.ndarray:
    try:
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img.shape[:2] != target_shape[:2]:
                img = cv2.resize(img, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_AREA)
            return img
    except:
        pass
    return np.zeros(target_shape, dtype=np.uint8)


class ParallelImageDecoder:
    def __init__(self, max_workers: int = 4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    def decode_batch(self, images: Dict[str, bytes]) -> Dict[str, np.ndarray]:
        futures = {}
        for img_key, img_bytes in images.items():
            if img_bytes is not None:
                shape = IMAGE_SHAPES.get(img_key, (720, 1280, 3))
                futures[img_key] = self.executor.submit(decode_image_rgb, img_bytes, shape)
        
        results = {}
        for img_key in IMAGE_SHAPES:
            if img_key in futures:
                results[img_key] = futures[img_key].result()
            else:
                results[img_key] = np.zeros(IMAGE_SHAPES[img_key], dtype=np.uint8)
        return results
    
    def shutdown(self):
        self.executor.shutdown(wait=True)


# ============================================================================
# 数据提取和同步
# ============================================================================

def extract_bag_data(bag_path: Path, verbose: bool = True) -> Dict:
    typestore = get_typestore(Stores.ROS1_NOETIC)
    
    data = {
        'arm_left_states': [], 'arm_right_states': [],
        'gripper_left_states': [], 'gripper_right_states': [],
        'arm_left_commands': [], 'arm_right_commands': [],
        'gripper_left_commands': [], 'gripper_right_commands': [],
        'images_head': [], 'images_wrist_left': [],
        'images_wrist_right': [], 'images_torso': [],
    }
    
    topic_mapping = {
        '/astribot_arm_left/joint_space_states': ('arm_left_states', parse_joint_state_fast),
        '/astribot_arm_right/joint_space_states': ('arm_right_states', parse_joint_state_fast),
        '/astribot_gripper_left/joint_space_states': ('gripper_left_states', parse_joint_state_fast),
        '/astribot_gripper_right/joint_space_states': ('gripper_right_states', parse_joint_state_fast),
        '/astribot_arm_left/joint_space_command': ('arm_left_commands', parse_joint_controller_fast),
        '/astribot_arm_right/joint_space_command': ('arm_right_commands', parse_joint_controller_fast),
        '/astribot_gripper_left/joint_space_command': ('gripper_left_commands', parse_joint_controller_fast),
        '/astribot_gripper_right/joint_space_command': ('gripper_right_commands', parse_joint_controller_fast),
    }
    
    image_topic_mapping = {
        '/astribot_camera/head_rgbd/color_compress/compressed': 'images_head',
        '/astribot_camera/left_wrist_rgbd/color_compress/compressed': 'images_wrist_left',
        '/astribot_camera/right_wrist_rgbd/color_compress/compressed': 'images_wrist_right',
        '/astribot_camera/torso_rgbd/color_compress/compressed': 'images_torso',
    }
    
    with Reader(bag_path) as reader:
        iterator = reader.messages()
        if verbose:
            try:
                iterator = tqdm(iterator, total=reader.message_count, desc="提取数据")
            except:
                pass
        
        for connection, timestamp, raw_data in iterator:
            topic = connection.topic
            
            if topic in topic_mapping:
                data_key, parser = topic_mapping[topic]
                parsed = parser(raw_data)
                if parsed is not None:
                    data[data_key].append({'timestamp': timestamp, 'data': parsed})
            
            elif topic in image_topic_mapping:
                data_key = image_topic_mapping[topic]
                try:
                    msg = typestore.deserialize_ros1(raw_data, 'sensor_msgs/msg/CompressedImage')
                    data[data_key].append({
                        'timestamp': timestamp,
                        'data': bytes(msg.data),
                    })
                except:
                    pass
    
    return data


def synchronize_data(raw_data: Dict, verbose: bool = True) -> List[Dict]:
    """同步数据到图像帧率"""
    if not raw_data['images_head']:
        return []
    
    indices = {
        'arm_left_states': TimestampIndex(raw_data['arm_left_states']),
        'arm_right_states': TimestampIndex(raw_data['arm_right_states']),
        'gripper_left_states': TimestampIndex(raw_data['gripper_left_states']),
        'gripper_right_states': TimestampIndex(raw_data['gripper_right_states']),
        'arm_left_commands': TimestampIndex(raw_data['arm_left_commands']),
        'arm_right_commands': TimestampIndex(raw_data['arm_right_commands']),
        'gripper_left_commands': TimestampIndex(raw_data['gripper_left_commands']),
        'gripper_right_commands': TimestampIndex(raw_data['gripper_right_commands']),
    }
    
    image_indices = {
        'head': ImageIndex(raw_data['images_head']),
        'wrist_left': ImageIndex(raw_data['images_wrist_left']),
        'wrist_right': ImageIndex(raw_data['images_wrist_right']),
        'torso': ImageIndex(raw_data['images_torso']),
    }
    
    base_timestamps = sorted([d['timestamp'] for d in raw_data['images_head']])
    
    frames = []
    iterator = base_timestamps
    if verbose:
        iterator = tqdm(iterator, desc="同步帧")
    
    for ts in iterator:
        arm_left = indices['arm_left_states'].find_nearest(ts, JOINT_TOLERANCE_NS)
        arm_right = indices['arm_right_states'].find_nearest(ts, JOINT_TOLERANCE_NS)
        arm_left_cmd = indices['arm_left_commands'].find_nearest(ts, JOINT_TOLERANCE_NS)
        arm_right_cmd = indices['arm_right_commands'].find_nearest(ts, JOINT_TOLERANCE_NS)
        
        if not all([arm_left, arm_right, arm_left_cmd, arm_right_cmd]):
            continue
        
        gripper_left = indices['gripper_left_states'].find_nearest(ts, JOINT_TOLERANCE_NS)
        gripper_right = indices['gripper_right_states'].find_nearest(ts, JOINT_TOLERANCE_NS)
        gripper_left_cmd = indices['gripper_left_commands'].find_nearest(ts, JOINT_TOLERANCE_NS)
        gripper_right_cmd = indices['gripper_right_commands'].find_nearest(ts, JOINT_TOLERANCE_NS)
        
        images = {}
        for img_key, img_idx in image_indices.items():
            img_data = img_idx.get_nearest(ts, IMAGE_TOLERANCE_NS)
            if img_data:
                images[img_key] = img_data['data']
        
        frame = {
            'timestamp': ts,
            'arm_left_state': arm_left,
            'arm_right_state': arm_right,
            'gripper_left_state': gripper_left,
            'gripper_right_state': gripper_right,
            'arm_left_cmd': arm_left_cmd,
            'arm_right_cmd': arm_right_cmd,
            'gripper_left_cmd': gripper_left_cmd,
            'gripper_right_cmd': gripper_right_cmd,
            'images': images,
        }
        
        frames.append(frame)
    
    return frames


# ============================================================================
# 转换函数
# ============================================================================

def convert_single_bag(
    bag_dir: Path,
    output_dir: Path,
    task_description: str = "Astribot manipulation task",
    episode_id: str = None,
    verbose: bool = True,
) -> int:
    """转换单个 bag 目录到 LeRobot 格式"""
    bag_path = Path(bag_dir) / "record" / "raw_data.bag"
    if not bag_path.exists():
        raise FileNotFoundError(f"Bag 文件不存在: {bag_path}")
    
    if episode_id is None:
        episode_id = bag_dir.name
    
    episode_output = output_dir / episode_id
    
    if verbose:
        print(f"转换: {bag_dir.name}")
    
    # 1. 提取数据
    raw_data = extract_bag_data(bag_path, verbose=verbose)
    
    # 2. 同步数据
    sync_frames = synchronize_data(raw_data, verbose=verbose)
    
    if not sync_frames:
        raise ValueError("没有有效的同步帧")
    
    # 3. 创建 LeRobot 数据集
    dataset = LeRobotDataset.create(
        repo_id=f"astribot/{episode_id}",
        root=episode_output,
        robot_type=ASTRIBOT_ROBOT_TYPE,
        fps=ASTRIBOT_FPS,
        features=ASTRIBOT_FEATURES,
        image_writer_threads=4,
    )
    
    # 4. 创建并行图像解码器
    image_decoder = ParallelImageDecoder(max_workers=4)
    
    try:
        iterator = sync_frames
        if verbose:
            iterator = tqdm(sync_frames, desc="转换帧")
        
        for frame in iterator:
            lerobot_frame = {'task': task_description}
            
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
    
    finally:
        image_decoder.shutdown()
    
    dataset.save_episode()
    dataset.finalize()
    
    return len(sync_frames)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='转换单个 Astribot bag 到 LeRobot')
    parser.add_argument('bag_dir', type=Path, help='Bag 目录')
    parser.add_argument('-o', '--output', type=Path, required=True, help='输出目录')
    parser.add_argument('--task', type=str, default="Astribot manipulation task")
    
    args = parser.parse_args()
    
    frames = convert_single_bag(args.bag_dir, args.output, args.task)
    print(f"转换完成: {frames} 帧")
