#!/usr/bin/env python3
"""
Astribot LeRobot 转换器 - 核心模块

包含:
- 25维特征定义 (ASTRIBOT_FEATURES)
- 高性能 ROS 消息解析器
- 数据提取和帧同步
- LeRobot 格式转换

所有其他转换模块都导入此核心模块。

用法:
    # 作为模块导入
    from core import (
        ASTRIBOT_FEATURES, ASTRIBOT_FPS,
        extract_bag_data, synchronize_data,
        convert_frame_to_lerobot, ParallelImageDecoder,
    )
    
    # 直接运行 (合并转换)
    python core.py /root/astribot_raw_datasets -o ./output --repo-id astribot/dataset
"""

import os
import sys
import struct
import json
import argparse
from pathlib import Path
from datetime import datetime
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
# 配置常量
# ============================================================================

ASTRIBOT_FPS = 30
ASTRIBOT_ROBOT_TYPE = "astribot_s1"

# 关节数量
ARM_JOINTS = 7
GRIPPER_JOINTS = 1
HEAD_JOINTS = 2
TORSO_JOINTS = 4
CHASSIS_JOINTS = 3

# 时间同步容忍度
JOINT_TOLERANCE_NS = 50_000_000   # 50ms
IMAGE_TOLERANCE_NS = 100_000_000  # 100ms

# ============================================================================
# 25维特征定义
# ============================================================================

ASTRIBOT_FEATURES = {
    # 观测状态 - 手臂
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
    # 观测状态 - 夹爪
    "observation.state.gripper_left.position": {
        "dtype": "float32", "shape": (GRIPPER_JOINTS,),
        "names": {"axes": ["gripper"]},
    },
    "observation.state.gripper_right.position": {
        "dtype": "float32", "shape": (GRIPPER_JOINTS,),
        "names": {"axes": ["gripper"]},
    },
    # 观测状态 - 头部
    "observation.state.head.position": {
        "dtype": "float32", "shape": (HEAD_JOINTS,),
        "names": {"axes": [f"joint_{i}" for i in range(HEAD_JOINTS)]},
    },
    "observation.state.head.velocity": {
        "dtype": "float32", "shape": (HEAD_JOINTS,),
        "names": {"axes": [f"joint_{i}" for i in range(HEAD_JOINTS)]},
    },
    "observation.state.head.torque": {
        "dtype": "float32", "shape": (HEAD_JOINTS,),
        "names": {"axes": [f"joint_{i}" for i in range(HEAD_JOINTS)]},
    },
    # 观测状态 - 腰部
    "observation.state.torso.position": {
        "dtype": "float32", "shape": (TORSO_JOINTS,),
        "names": {"axes": [f"joint_{i}" for i in range(TORSO_JOINTS)]},
    },
    "observation.state.torso.velocity": {
        "dtype": "float32", "shape": (TORSO_JOINTS,),
        "names": {"axes": [f"joint_{i}" for i in range(TORSO_JOINTS)]},
    },
    "observation.state.torso.torque": {
        "dtype": "float32", "shape": (TORSO_JOINTS,),
        "names": {"axes": [f"joint_{i}" for i in range(TORSO_JOINTS)]},
    },
    # 观测状态 - 底盘
    "observation.state.chassis.position": {
        "dtype": "float32", "shape": (CHASSIS_JOINTS,),
        "names": {"axes": [f"joint_{i}" for i in range(CHASSIS_JOINTS)]},
    },
    "observation.state.chassis.velocity": {
        "dtype": "float32", "shape": (CHASSIS_JOINTS,),
        "names": {"axes": [f"joint_{i}" for i in range(CHASSIS_JOINTS)]},
    },
    "observation.state.chassis.torque": {
        "dtype": "float32", "shape": (CHASSIS_JOINTS,),
        "names": {"axes": [f"joint_{i}" for i in range(CHASSIS_JOINTS)]},
    },
    # 组合观测状态向量 (7+7+1+1+2+4+3 = 25)
    "observation.state": {
        "dtype": "float32", "shape": (25,),
        "names": {"axes": [f"arm_left_joint_{i}" for i in range(ARM_JOINTS)] + 
                         [f"arm_right_joint_{i}" for i in range(ARM_JOINTS)] +
                         ["gripper_left", "gripper_right"] +
                         [f"head_joint_{i}" for i in range(HEAD_JOINTS)] +
                         [f"torso_joint_{i}" for i in range(TORSO_JOINTS)] +
                         [f"chassis_joint_{i}" for i in range(CHASSIS_JOINTS)]},
    },
    # 图像
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
    # 动作 - 手臂
    "action.arm_left": {
        "dtype": "float32", "shape": (ARM_JOINTS,),
        "names": {"axes": [f"joint_{i}" for i in range(ARM_JOINTS)]},
    },
    "action.arm_right": {
        "dtype": "float32", "shape": (ARM_JOINTS,),
        "names": {"axes": [f"joint_{i}" for i in range(ARM_JOINTS)]},
    },
    # 动作 - 夹爪
    "action.gripper_left": {
        "dtype": "float32", "shape": (GRIPPER_JOINTS,),
        "names": {"axes": ["gripper"]},
    },
    "action.gripper_right": {
        "dtype": "float32", "shape": (GRIPPER_JOINTS,),
        "names": {"axes": ["gripper"]},
    },
    # 动作 - 头部
    "action.head": {
        "dtype": "float32", "shape": (HEAD_JOINTS,),
        "names": {"axes": [f"joint_{i}" for i in range(HEAD_JOINTS)]},
    },
    # 动作 - 腰部
    "action.torso": {
        "dtype": "float32", "shape": (TORSO_JOINTS,),
        "names": {"axes": [f"joint_{i}" for i in range(TORSO_JOINTS)]},
    },
    # 动作 - 底盘
    "action.chassis": {
        "dtype": "float32", "shape": (CHASSIS_JOINTS,),
        "names": {"axes": [f"joint_{i}" for i in range(CHASSIS_JOINTS)]},
    },
    # 组合动作向量 (7+7+1+1+2+4+3 = 25)
    "action": {
        "dtype": "float32", "shape": (25,),
        "names": {"axes": [f"arm_left_joint_{i}" for i in range(ARM_JOINTS)] + 
                         [f"arm_right_joint_{i}" for i in range(ARM_JOINTS)] +
                         ["gripper_left", "gripper_right"] +
                         [f"head_joint_{i}" for i in range(HEAD_JOINTS)] +
                         [f"torso_joint_{i}" for i in range(TORSO_JOINTS)] +
                         [f"chassis_joint_{i}" for i in range(CHASSIS_JOINTS)]},
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
    """高性能 ROS1 消息解析器"""
    
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
    """解析关节状态消息"""
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
    except Exception:
        return None


def parse_joint_controller_fast(raw_data: bytes) -> Optional[Dict[str, np.ndarray]]:
    """解析关节命令消息"""
    try:
        offset = 0
        _, offset = FastMessageParser.read_header_fast(raw_data, offset)
        offset += 1
        offset = FastMessageParser.skip_string_array(raw_data, offset)
        command, _ = FastMessageParser.read_float64_array_zero_copy(raw_data, offset)
        return {'command': command.astype(np.float32, copy=False)}
    except Exception:
        return None


# ============================================================================
# 时间戳索引
# ============================================================================

class TimestampIndex:
    """时间戳索引，支持快速二分查找"""
    
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
    """图像数据索引"""
    
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
    """解码图像为 RGB"""
    try:
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img.shape[:2] != target_shape[:2]:
                img = cv2.resize(img, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_AREA)
            return img
    except Exception:
        pass
    return np.zeros(target_shape, dtype=np.uint8)


class ParallelImageDecoder:
    """并行图像解码器"""
    
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
# 数据提取
# ============================================================================

def extract_bag_data(bag_path: Path, verbose: bool = True) -> Dict:
    """从 rosbag 提取数据"""
    typestore = get_typestore(Stores.ROS1_NOETIC)
    
    data = {
        'arm_left_states': [], 'arm_right_states': [],
        'gripper_left_states': [], 'gripper_right_states': [],
        'head_states': [], 'torso_states': [], 'chassis_states': [],
        'arm_left_commands': [], 'arm_right_commands': [],
        'gripper_left_commands': [], 'gripper_right_commands': [],
        'head_commands': [], 'torso_commands': [], 'chassis_commands': [],
        'images_head': [], 'images_wrist_left': [],
        'images_wrist_right': [], 'images_torso': [],
    }
    
    topic_mapping = {
        '/astribot_arm_left/joint_space_states': ('arm_left_states', parse_joint_state_fast),
        '/astribot_arm_right/joint_space_states': ('arm_right_states', parse_joint_state_fast),
        '/astribot_gripper_left/joint_space_states': ('gripper_left_states', parse_joint_state_fast),
        '/astribot_gripper_right/joint_space_states': ('gripper_right_states', parse_joint_state_fast),
        '/astribot_head/joint_space_states': ('head_states', parse_joint_state_fast),
        '/astribot_torso/joint_space_states': ('torso_states', parse_joint_state_fast),
        '/astribot_chassis/joint_space_states': ('chassis_states', parse_joint_state_fast),
        '/astribot_arm_left/joint_space_command': ('arm_left_commands', parse_joint_controller_fast),
        '/astribot_arm_right/joint_space_command': ('arm_right_commands', parse_joint_controller_fast),
        '/astribot_gripper_left/joint_space_command': ('gripper_left_commands', parse_joint_controller_fast),
        '/astribot_gripper_right/joint_space_command': ('gripper_right_commands', parse_joint_controller_fast),
        '/astribot_head/joint_space_command': ('head_commands', parse_joint_controller_fast),
        '/astribot_torso/joint_space_command': ('torso_commands', parse_joint_controller_fast),
        '/astribot_chassis/joint_space_command': ('chassis_commands', parse_joint_controller_fast),
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
                iterator = tqdm(iterator, total=reader.message_count, desc="  提取数据")
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


# ============================================================================
# 帧同步
# ============================================================================

def synchronize_data(raw_data: Dict, verbose: bool = True) -> List[Dict]:
    """将多模态数据同步到统一时间轴"""
    if not raw_data['images_head']:
        return []
    
    indices = {
        'arm_left_states': TimestampIndex(raw_data['arm_left_states']),
        'arm_right_states': TimestampIndex(raw_data['arm_right_states']),
        'gripper_left_states': TimestampIndex(raw_data['gripper_left_states']),
        'gripper_right_states': TimestampIndex(raw_data['gripper_right_states']),
        'head_states': TimestampIndex(raw_data['head_states']),
        'torso_states': TimestampIndex(raw_data['torso_states']),
        'chassis_states': TimestampIndex(raw_data['chassis_states']),
        'arm_left_commands': TimestampIndex(raw_data['arm_left_commands']),
        'arm_right_commands': TimestampIndex(raw_data['arm_right_commands']),
        'gripper_left_commands': TimestampIndex(raw_data['gripper_left_commands']),
        'gripper_right_commands': TimestampIndex(raw_data['gripper_right_commands']),
        'head_commands': TimestampIndex(raw_data['head_commands']),
        'torso_commands': TimestampIndex(raw_data['torso_commands']),
        'chassis_commands': TimestampIndex(raw_data['chassis_commands']),
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
        iterator = tqdm(iterator, desc="  同步帧")
    
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
        
        head_state = indices['head_states'].find_nearest(ts, JOINT_TOLERANCE_NS)
        torso_state = indices['torso_states'].find_nearest(ts, JOINT_TOLERANCE_NS)
        chassis_state = indices['chassis_states'].find_nearest(ts, JOINT_TOLERANCE_NS)
        head_cmd = indices['head_commands'].find_nearest(ts, JOINT_TOLERANCE_NS)
        torso_cmd = indices['torso_commands'].find_nearest(ts, JOINT_TOLERANCE_NS)
        chassis_cmd = indices['chassis_commands'].find_nearest(ts, JOINT_TOLERANCE_NS)
        
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
            'head_state': head_state,
            'torso_state': torso_state,
            'chassis_state': chassis_state,
            'arm_left_cmd': arm_left_cmd,
            'arm_right_cmd': arm_right_cmd,
            'gripper_left_cmd': gripper_left_cmd,
            'gripper_right_cmd': gripper_right_cmd,
            'head_cmd': head_cmd,
            'torso_cmd': torso_cmd,
            'chassis_cmd': chassis_cmd,
            'images': images,
        }
        frames.append(frame)
    
    return frames


# ============================================================================
# 帧转换
# ============================================================================

def convert_frame_to_lerobot(
    frame: Dict,
    task_description: str,
    image_decoder: ParallelImageDecoder
) -> Dict:
    """将同步帧转换为 LeRobot 格式"""
    lerobot_frame = {'task': task_description}
    
    als = frame['arm_left_state']
    ars = frame['arm_right_state']
    gls = frame['gripper_left_state']
    grs = frame['gripper_right_state']
    hs = frame['head_state']
    ts = frame['torso_state']
    cs = frame['chassis_state']
    
    # 手臂
    lerobot_frame['observation.state.arm_left.position'] = als['position'][:ARM_JOINTS]
    lerobot_frame['observation.state.arm_left.velocity'] = als['velocity'][:ARM_JOINTS]
    lerobot_frame['observation.state.arm_left.torque'] = als['torque'][:ARM_JOINTS]
    lerobot_frame['observation.state.arm_right.position'] = ars['position'][:ARM_JOINTS]
    lerobot_frame['observation.state.arm_right.velocity'] = ars['velocity'][:ARM_JOINTS]
    lerobot_frame['observation.state.arm_right.torque'] = ars['torque'][:ARM_JOINTS]
    
    # 夹爪
    gl_pos = np.array([gls['position'][0] if gls and len(gls['position']) > 0 else 0.0], dtype=np.float32)
    gr_pos = np.array([grs['position'][0] if grs and len(grs['position']) > 0 else 0.0], dtype=np.float32)
    lerobot_frame['observation.state.gripper_left.position'] = gl_pos
    lerobot_frame['observation.state.gripper_right.position'] = gr_pos
    
    # 头部
    if hs and len(hs['position']) >= HEAD_JOINTS:
        head_pos = hs['position'][:HEAD_JOINTS].astype(np.float32)
        head_vel = hs['velocity'][:HEAD_JOINTS].astype(np.float32)
        head_tor = hs['torque'][:HEAD_JOINTS].astype(np.float32)
    else:
        head_pos = np.zeros(HEAD_JOINTS, dtype=np.float32)
        head_vel = np.zeros(HEAD_JOINTS, dtype=np.float32)
        head_tor = np.zeros(HEAD_JOINTS, dtype=np.float32)
    lerobot_frame['observation.state.head.position'] = head_pos
    lerobot_frame['observation.state.head.velocity'] = head_vel
    lerobot_frame['observation.state.head.torque'] = head_tor
    
    # 腰部
    if ts and len(ts['position']) >= TORSO_JOINTS:
        torso_pos = ts['position'][:TORSO_JOINTS].astype(np.float32)
        torso_vel = ts['velocity'][:TORSO_JOINTS].astype(np.float32)
        torso_tor = ts['torque'][:TORSO_JOINTS].astype(np.float32)
    else:
        torso_pos = np.zeros(TORSO_JOINTS, dtype=np.float32)
        torso_vel = np.zeros(TORSO_JOINTS, dtype=np.float32)
        torso_tor = np.zeros(TORSO_JOINTS, dtype=np.float32)
    lerobot_frame['observation.state.torso.position'] = torso_pos
    lerobot_frame['observation.state.torso.velocity'] = torso_vel
    lerobot_frame['observation.state.torso.torque'] = torso_tor
    
    # 底盘
    if cs and len(cs['position']) >= CHASSIS_JOINTS:
        chassis_pos = cs['position'][:CHASSIS_JOINTS].astype(np.float32)
        chassis_vel = cs['velocity'][:CHASSIS_JOINTS].astype(np.float32)
        chassis_tor = cs['torque'][:CHASSIS_JOINTS].astype(np.float32)
    else:
        chassis_pos = np.zeros(CHASSIS_JOINTS, dtype=np.float32)
        chassis_vel = np.zeros(CHASSIS_JOINTS, dtype=np.float32)
        chassis_tor = np.zeros(CHASSIS_JOINTS, dtype=np.float32)
    lerobot_frame['observation.state.chassis.position'] = chassis_pos
    lerobot_frame['observation.state.chassis.velocity'] = chassis_vel
    lerobot_frame['observation.state.chassis.torque'] = chassis_tor
    
    # 组合状态向量 (25维)
    lerobot_frame['observation.state'] = np.concatenate([
        als['position'][:ARM_JOINTS], ars['position'][:ARM_JOINTS], 
        gl_pos, gr_pos, head_pos, torso_pos, chassis_pos
    ])
    
    # 动作
    alc = frame['arm_left_cmd']
    arc = frame['arm_right_cmd']
    glc = frame['gripper_left_cmd']
    grc = frame['gripper_right_cmd']
    hc = frame['head_cmd']
    tc = frame['torso_cmd']
    cc = frame['chassis_cmd']
    
    lerobot_frame['action.arm_left'] = alc['command'][:ARM_JOINTS] if len(alc['command']) >= ARM_JOINTS else np.zeros(ARM_JOINTS, dtype=np.float32)
    lerobot_frame['action.arm_right'] = arc['command'][:ARM_JOINTS] if len(arc['command']) >= ARM_JOINTS else np.zeros(ARM_JOINTS, dtype=np.float32)
    
    gl_cmd = np.array([glc['command'][0] if glc and len(glc['command']) > 0 else 0.0], dtype=np.float32)
    gr_cmd = np.array([grc['command'][0] if grc and len(grc['command']) > 0 else 0.0], dtype=np.float32)
    lerobot_frame['action.gripper_left'] = gl_cmd
    lerobot_frame['action.gripper_right'] = gr_cmd
    
    head_cmd = hc['command'][:HEAD_JOINTS].astype(np.float32) if hc and len(hc['command']) >= HEAD_JOINTS else np.zeros(HEAD_JOINTS, dtype=np.float32)
    torso_cmd = tc['command'][:TORSO_JOINTS].astype(np.float32) if tc and len(tc['command']) >= TORSO_JOINTS else np.zeros(TORSO_JOINTS, dtype=np.float32)
    chassis_cmd = cc['command'][:CHASSIS_JOINTS].astype(np.float32) if cc and len(cc['command']) >= CHASSIS_JOINTS else np.zeros(CHASSIS_JOINTS, dtype=np.float32)
    
    lerobot_frame['action.head'] = head_cmd
    lerobot_frame['action.torso'] = torso_cmd
    lerobot_frame['action.chassis'] = chassis_cmd
    
    # 组合动作向量 (25维)
    lerobot_frame['action'] = np.concatenate([
        lerobot_frame['action.arm_left'], lerobot_frame['action.arm_right'], 
        gl_cmd, gr_cmd, head_cmd, torso_cmd, chassis_cmd
    ])
    
    # 图像
    decoded_images = image_decoder.decode_batch(frame.get('images', {}))
    for img_key, img_data in decoded_images.items():
        lerobot_frame[f'observation.images.{img_key}'] = img_data
    
    return lerobot_frame


# ============================================================================
# 元数据
# ============================================================================

def load_episode_metadata(bag_dir: Path) -> Dict[str, Any]:
    """加载 episode 元数据"""
    metadata = {
        'task_name': None,
        'scene': None,
        'operator': None,
        'language_instruction': None,
    }
    
    custom_desc_file = bag_dir / "task_description.txt"
    if custom_desc_file.exists():
        with open(custom_desc_file, 'r', encoding='utf-8') as f:
            metadata['language_instruction'] = f.read().strip()
        return metadata
    
    loong_meta = bag_dir / "__loongdata_metadata.json"
    if loong_meta.exists():
        try:
            with open(loong_meta, 'r', encoding='utf-8') as f:
                data = json.load(f)
                metadata['task_name'] = data.get('taskName', '')
                metadata['scene'] = data.get('scene', '')
                metadata['operator'] = data.get('operator', '')
        except Exception:
            pass
    
    if metadata['task_name']:
        parts = [metadata['task_name']]
        if metadata['scene']:
            parts.append(f"场景: {metadata['scene']}")
        metadata['language_instruction'] = ' - '.join(parts)
    
    return metadata


def find_bag_directories(root_dir: Path) -> List[Tuple[Path, Dict]]:
    """查找所有 rosbag 目录"""
    bag_dirs = []
    for item in sorted(root_dir.iterdir()):
        if item.is_dir():
            bag_path = item / "record" / "raw_data.bag"
            if bag_path.exists():
                metadata = load_episode_metadata(item)
                bag_dirs.append((item, metadata))
    return bag_dirs


# ============================================================================
# 合并转换
# ============================================================================

def convert_all_to_single_dataset(
    bag_root: Path,
    output_dir: Path,
    repo_id: str,
    task_description: str = None,
    use_episode_tasks: bool = True,
):
    """将所有 bag 文件转换为单个 LeRobot 数据集"""
    
    print(f"\n{'='*70}")
    print(f"Astribot -> LeRobot 转换 (25维特征)")
    print(f"{'='*70}")
    print(f"输入: {bag_root}")
    print(f"输出: {output_dir}")
    print(f"Repo: {repo_id}")
    print(f"{'='*70}\n")
    
    bag_dirs_with_meta = find_bag_directories(bag_root)
    
    if not bag_dirs_with_meta:
        print("错误: 没有找到 rosbag 目录")
        return
    
    print(f"找到 {len(bag_dirs_with_meta)} 个 episode")
    
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
    
    try:
        for episode_idx, (bag_dir, metadata) in enumerate(bag_dirs_with_meta):
            bag_path = bag_dir / "record" / "raw_data.bag"
            
            if task_description:
                current_task = task_description
            elif use_episode_tasks and metadata.get('language_instruction'):
                current_task = metadata['language_instruction']
            else:
                current_task = "Astribot manipulation task"
            
            print(f"\n[{episode_idx}] {bag_dir.name}: {current_task}")
            
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
                'episode_index': episode_idx,
                'source': bag_dir.name,
                'task': current_task,
                'frames': ep_frames,
            })
            
            print(f"  ✓ {ep_frames} 帧")
            del raw_data, sync_frames
    
    finally:
        image_decoder.shutdown()
    
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
    parser = argparse.ArgumentParser(description='Astribot -> LeRobot 转换 (25维)')
    parser.add_argument('bag_root', type=Path, help='rosbag 目录')
    parser.add_argument('--repo-id', type=str, default='astribot/dataset')
    parser.add_argument('-o', '--output-dir', type=Path, default=None)
    parser.add_argument('--task', type=str, default=None)
    parser.add_argument('--no-episode-tasks', action='store_true')
    
    args = parser.parse_args()
    
    output_dir = args.output_dir or Path('./astribot_lerobot_dataset')
    
    convert_all_to_single_dataset(
        bag_root=args.bag_root,
        output_dir=output_dir,
        repo_id=args.repo_id,
        task_description=args.task,
        use_episode_tasks=not args.no_episode_tasks,
    )


if __name__ == '__main__':
    main()

