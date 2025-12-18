#!/usr/bin/env python3
"""
Astribot ROS1 Bag 转换脚本 - 合并多个 episode 到单个数据集
性能优化版本

用法:
    python convert_merged.py /root/astribot_raw_datasets --repo-id astribot/dataset -o ./output

组帧逻辑:
    1. 以 head 相机时间戳为基准 (30Hz)
    2. 对每个图像时间戳，用二分查找找到 ±50ms 内最近的关节数据
    3. 同时找到其他相机最近的图像 (±100ms)
    4. 必要数据齐全时生成一帧
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
# 配置
# ============================================================================

ASTRIBOT_FPS = 30
ASTRIBOT_ROBOT_TYPE = "astribot_s1"
ARM_JOINTS = 7
GRIPPER_JOINTS = 1
HEAD_JOINTS = 2      # 头部关节数 (pan/tilt)
TORSO_JOINTS = 4     # 腰部关节数
CHASSIS_JOINTS = 3   # 底盘关节数

# 时间同步容忍度
JOINT_TOLERANCE_NS = 50_000_000   # 50ms - 关节数据
IMAGE_TOLERANCE_NS = 100_000_000  # 100ms - 图像数据

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
# 高性能消息解析 - 零拷贝优化
# ============================================================================

class FastMessageParser:
    """高性能 ROS1 消息解析器，使用批量解包和零拷贝"""
    
    __slots__ = ()  # 禁用实例字典，节省内存
    
    @staticmethod
    def read_header_fast(data: bytes, offset: int = 0) -> Tuple[int, int]:
        """快速读取 header，返回 (stamp_ns, new_offset)"""
        # 合并解包: seq(4) + secs(4) + nsecs(4) = 12 bytes
        seq, secs, nsecs = struct.unpack_from('<III', data, offset)
        offset += 12
        # 跳过 frame_id 字符串
        str_len = struct.unpack_from('<I', data, offset)[0]
        offset += 4 + str_len
        # 返回纳秒时间戳
        return secs * 1_000_000_000 + nsecs, offset
    
    @staticmethod
    def read_float64_array_zero_copy(data: bytes, offset: int) -> Tuple[np.ndarray, int]:
        """零拷贝读取 float64 数组"""
        length = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        if length > 0:
            # 使用 frombuffer 创建视图，避免复制
            arr = np.frombuffer(data, dtype='<f8', count=length, offset=offset)
            return arr, offset + length * 8
        return np.empty(0, dtype=np.float64), offset
    
    @staticmethod
    def skip_string_array(data: bytes, offset: int) -> int:
        """跳过字符串数组，只移动偏移量"""
        count = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        for _ in range(count):
            str_len = struct.unpack_from('<I', data, offset)[0]
            offset += 4 + str_len
        return offset


def parse_joint_state_fast(raw_data: bytes) -> Optional[Dict[str, np.ndarray]]:
    """高性能关节状态解析"""
    try:
        offset = 0
        # 跳过 header
        _, offset = FastMessageParser.read_header_fast(raw_data, offset)
        # 跳过 mode (1 byte)
        offset += 1
        # 跳过 names 字符串数组
        offset = FastMessageParser.skip_string_array(raw_data, offset)
        # 读取数值数组 - 零拷贝
        position, offset = FastMessageParser.read_float64_array_zero_copy(raw_data, offset)
        velocity, offset = FastMessageParser.read_float64_array_zero_copy(raw_data, offset)
        _, offset = FastMessageParser.read_float64_array_zero_copy(raw_data, offset)  # skip acceleration
        torque, offset = FastMessageParser.read_float64_array_zero_copy(raw_data, offset)
        
        # 转换为 float32 (必要的复制，但更节省内存)
        return {
            'position': position.astype(np.float32, copy=False),
            'velocity': velocity.astype(np.float32, copy=False),
            'torque': torque.astype(np.float32, copy=False),
        }
    except Exception:
        return None


def parse_joint_controller_fast(raw_data: bytes) -> Optional[Dict[str, np.ndarray]]:
    """高性能关节控制命令解析"""
    try:
        offset = 0
        _, offset = FastMessageParser.read_header_fast(raw_data, offset)
        offset += 1  # skip mode
        offset = FastMessageParser.skip_string_array(raw_data, offset)
        command, _ = FastMessageParser.read_float64_array_zero_copy(raw_data, offset)
        
        return {'command': command.astype(np.float32, copy=False)}
    except Exception:
        return None


# ============================================================================
# 时间戳索引 - 预计算优化
# ============================================================================

class TimestampIndex:
    """预计算时间戳索引，加速二分查找"""
    
    __slots__ = ('data', 'timestamps', 'size')
    
    def __init__(self, data_list: List[Dict]):
        self.data = data_list
        self.size = len(data_list)
        if self.size > 0:
            # 预先提取所有时间戳到 NumPy 数组
            self.timestamps = np.array([d['timestamp'] for d in data_list], dtype=np.int64)
        else:
            self.timestamps = np.empty(0, dtype=np.int64)
    
    def find_nearest(self, target_ts: int, tolerance_ns: int) -> Optional[Any]:
        """使用 NumPy searchsorted 进行快速查找"""
        if self.size == 0:
            return None
        
        # NumPy 二分查找比 bisect 更快
        idx = np.searchsorted(self.timestamps, target_ts)
        
        best_diff = tolerance_ns + 1
        best_idx = -1
        
        # 检查左侧候选
        if idx > 0:
            diff = abs(self.timestamps[idx - 1] - target_ts)
            if diff < best_diff:
                best_diff = diff
                best_idx = idx - 1
        
        # 检查右侧候选
        if idx < self.size:
            diff = abs(self.timestamps[idx] - target_ts)
            if diff < best_diff:
                best_diff = diff
                best_idx = idx
        
        if best_diff <= tolerance_ns:
            return self.data[best_idx]['data']
        return None


class ImageIndex:
    """图像数据索引，支持按时间戳查找"""
    
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
        """查找最近的图像数据"""
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
# 图像处理 - 并行解码
# ============================================================================

def decode_image_rgb(img_bytes: bytes, target_shape: Tuple[int, int, int]) -> np.ndarray:
    """解码图像并转换为 RGB，处理尺寸不匹配"""
    try:
        # 使用 frombuffer 避免复制
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            # BGR -> RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # 检查尺寸
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
        """并行解码多张图像"""
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
    """从 rosbag 提取数据 - 优化版"""
    typestore = get_typestore(Stores.ROS1_NOETIC)
    
    data = {
        'arm_left_states': [], 'arm_right_states': [],
        'gripper_left_states': [], 'gripper_right_states': [],
        'head_states': [], 'torso_states': [], 'chassis_states': [],  # 头部、腰部、底盘状态
        'arm_left_commands': [], 'arm_right_commands': [],
        'gripper_left_commands': [], 'gripper_right_commands': [],
        'head_commands': [], 'torso_commands': [], 'chassis_commands': [],  # 头部、腰部、底盘命令
        'images_head': [], 'images_wrist_left': [],
        'images_wrist_right': [], 'images_torso': [],
    }
    
    topic_mapping = {
        '/astribot_arm_left/joint_space_states': ('arm_left_states', parse_joint_state_fast),
        '/astribot_arm_right/joint_space_states': ('arm_right_states', parse_joint_state_fast),
        '/astribot_gripper_left/joint_space_states': ('gripper_left_states', parse_joint_state_fast),
        '/astribot_gripper_right/joint_space_states': ('gripper_right_states', parse_joint_state_fast),
        '/astribot_head/joint_space_states': ('head_states', parse_joint_state_fast),  # 头部状态
        '/astribot_torso/joint_space_states': ('torso_states', parse_joint_state_fast),  # 腰部状态
        '/astribot_chassis/joint_space_states': ('chassis_states', parse_joint_state_fast),  # 底盘状态
        '/astribot_arm_left/joint_space_command': ('arm_left_commands', parse_joint_controller_fast),
        '/astribot_arm_right/joint_space_command': ('arm_right_commands', parse_joint_controller_fast),
        '/astribot_gripper_left/joint_space_command': ('gripper_left_commands', parse_joint_controller_fast),
        '/astribot_gripper_right/joint_space_command': ('gripper_right_commands', parse_joint_controller_fast),
        '/astribot_head/joint_space_command': ('head_commands', parse_joint_controller_fast),  # 头部命令
        '/astribot_torso/joint_space_command': ('torso_commands', parse_joint_controller_fast),  # 腰部命令
        '/astribot_chassis/joint_space_command': ('chassis_commands', parse_joint_controller_fast),  # 底盘命令
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
                    # 保存原始压缩数据，延迟解码
                    data[data_key].append({
                        'timestamp': timestamp,
                        'data': bytes(msg.data),
                    })
                except:
                    pass
    
    return data


# ============================================================================
# 帧同步 - 核心组帧逻辑
# ============================================================================

def synchronize_data(raw_data: Dict, verbose: bool = True) -> List[Dict]:
    """
    帧同步 - 将多模态数据对齐到统一时间轴
    
    组帧逻辑:
    =========
    1. 基准选择: 以 head 相机图像时间戳为基准 (30Hz)
       - 原因: 图像帧率最稳定，且是训练的主要输入
    
    2. 对齐策略: 对每个基准时间戳 t:
       a) 关节状态: 找到 t ± 50ms 内最近的数据
          - arm_left_states, arm_right_states (必需)
          - gripper_left_states, gripper_right_states (可选)
       
       b) 关节命令: 找到 t ± 50ms 内最近的数据
          - arm_left_commands, arm_right_commands (必需)
          - gripper_left_commands, gripper_right_commands (可选)
       
       c) 其他相机: 找到 t ± 100ms 内最近的图像
          - wrist_left, wrist_right, torso
    
    3. 有效帧条件:
       - 必须有 arm_left_state, arm_right_state
       - 必须有 arm_left_cmd, arm_right_cmd
       - 其他数据缺失时使用默认值
    
    Returns:
        List[Dict]: 同步后的帧列表
    """
    if not raw_data['images_head']:
        return []
    
    # 构建预计算索引 - 避免重复创建时间戳列表
    indices = {
        'arm_left_states': TimestampIndex(raw_data['arm_left_states']),
        'arm_right_states': TimestampIndex(raw_data['arm_right_states']),
        'gripper_left_states': TimestampIndex(raw_data['gripper_left_states']),
        'gripper_right_states': TimestampIndex(raw_data['gripper_right_states']),
        'head_states': TimestampIndex(raw_data['head_states']),  # 头部状态
        'torso_states': TimestampIndex(raw_data['torso_states']),  # 腰部状态
        'chassis_states': TimestampIndex(raw_data['chassis_states']),  # 底盘状态
        'arm_left_commands': TimestampIndex(raw_data['arm_left_commands']),
        'arm_right_commands': TimestampIndex(raw_data['arm_right_commands']),
        'gripper_left_commands': TimestampIndex(raw_data['gripper_left_commands']),
        'gripper_right_commands': TimestampIndex(raw_data['gripper_right_commands']),
        'head_commands': TimestampIndex(raw_data['head_commands']),  # 头部命令
        'torso_commands': TimestampIndex(raw_data['torso_commands']),  # 腰部命令
        'chassis_commands': TimestampIndex(raw_data['chassis_commands']),  # 底盘命令
    }
    
    image_indices = {
        'head': ImageIndex(raw_data['images_head']),
        'wrist_left': ImageIndex(raw_data['images_wrist_left']),
        'wrist_right': ImageIndex(raw_data['images_wrist_right']),
        'torso': ImageIndex(raw_data['images_torso']),
    }
    
    # 获取基准时间戳序列
    base_timestamps = sorted([d['timestamp'] for d in raw_data['images_head']])
    
    frames = []
    iterator = base_timestamps
    if verbose:
        iterator = tqdm(iterator, desc="  同步帧")
    
    for ts in iterator:
        # 查找关节状态 (必需)
        arm_left = indices['arm_left_states'].find_nearest(ts, JOINT_TOLERANCE_NS)
        arm_right = indices['arm_right_states'].find_nearest(ts, JOINT_TOLERANCE_NS)
        
        # 查找关节命令 (必需)
        arm_left_cmd = indices['arm_left_commands'].find_nearest(ts, JOINT_TOLERANCE_NS)
        arm_right_cmd = indices['arm_right_commands'].find_nearest(ts, JOINT_TOLERANCE_NS)
        
        # 必需数据检查
        if not all([arm_left, arm_right, arm_left_cmd, arm_right_cmd]):
            continue
        
        # 查找可选数据
        gripper_left = indices['gripper_left_states'].find_nearest(ts, JOINT_TOLERANCE_NS)
        gripper_right = indices['gripper_right_states'].find_nearest(ts, JOINT_TOLERANCE_NS)
        gripper_left_cmd = indices['gripper_left_commands'].find_nearest(ts, JOINT_TOLERANCE_NS)
        gripper_right_cmd = indices['gripper_right_commands'].find_nearest(ts, JOINT_TOLERANCE_NS)
        
        # 查找头部、腰部和底盘数据 (可选)
        head_state = indices['head_states'].find_nearest(ts, JOINT_TOLERANCE_NS)
        torso_state = indices['torso_states'].find_nearest(ts, JOINT_TOLERANCE_NS)
        chassis_state = indices['chassis_states'].find_nearest(ts, JOINT_TOLERANCE_NS)
        head_cmd = indices['head_commands'].find_nearest(ts, JOINT_TOLERANCE_NS)
        torso_cmd = indices['torso_commands'].find_nearest(ts, JOINT_TOLERANCE_NS)
        chassis_cmd = indices['chassis_commands'].find_nearest(ts, JOINT_TOLERANCE_NS)
        
        # 查找所有相机图像
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
            'head_state': head_state,  # 头部状态
            'torso_state': torso_state,  # 腰部状态
            'chassis_state': chassis_state,  # 底盘状态
            'arm_left_cmd': arm_left_cmd,
            'arm_right_cmd': arm_right_cmd,
            'gripper_left_cmd': gripper_left_cmd,
            'gripper_right_cmd': gripper_right_cmd,
            'head_cmd': head_cmd,  # 头部命令
            'torso_cmd': torso_cmd,  # 腰部命令
            'chassis_cmd': chassis_cmd,  # 底盘命令
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
    """将同步后的帧转换为 LeRobot 格式"""
    lerobot_frame = {'task': task_description}
    
    # 获取关节数据
    als = frame['arm_left_state']
    ars = frame['arm_right_state']
    gls = frame['gripper_left_state']
    grs = frame['gripper_right_state']
    hs = frame['head_state']  # 头部状态
    ts = frame['torso_state']  # 腰部状态
    cs = frame['chassis_state']  # 底盘状态
    
    # 观测状态 - 手臂 (直接使用切片视图)
    lerobot_frame['observation.state.arm_left.position'] = als['position'][:ARM_JOINTS]
    lerobot_frame['observation.state.arm_left.velocity'] = als['velocity'][:ARM_JOINTS]
    lerobot_frame['observation.state.arm_left.torque'] = als['torque'][:ARM_JOINTS]
    lerobot_frame['observation.state.arm_right.position'] = ars['position'][:ARM_JOINTS]
    lerobot_frame['observation.state.arm_right.velocity'] = ars['velocity'][:ARM_JOINTS]
    lerobot_frame['observation.state.arm_right.torque'] = ars['torque'][:ARM_JOINTS]
    
    # 观测状态 - 夹爪
    gl_pos = np.array([gls['position'][0] if gls and len(gls['position']) > 0 else 0.0], dtype=np.float32)
    gr_pos = np.array([grs['position'][0] if grs and len(grs['position']) > 0 else 0.0], dtype=np.float32)
    lerobot_frame['observation.state.gripper_left.position'] = gl_pos
    lerobot_frame['observation.state.gripper_right.position'] = gr_pos
    
    # 观测状态 - 头部
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
    
    # 观测状态 - 腰部
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
    
    # 观测状态 - 底盘
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
    
    # 组合状态向量 (arm_left[7] + arm_right[7] + gripper[2] + head[2] + torso[4] + chassis[3] = 25)
    lerobot_frame['observation.state'] = np.concatenate([
        als['position'][:ARM_JOINTS], ars['position'][:ARM_JOINTS], 
        gl_pos, gr_pos,
        head_pos, torso_pos, chassis_pos
    ])
    
    # 动作
    alc = frame['arm_left_cmd']
    arc = frame['arm_right_cmd']
    glc = frame['gripper_left_cmd']
    grc = frame['gripper_right_cmd']
    hc = frame['head_cmd']  # 头部命令
    tc = frame['torso_cmd']  # 腰部命令
    cc = frame['chassis_cmd']  # 底盘命令
    
    # 动作 - 手臂
    lerobot_frame['action.arm_left'] = alc['command'][:ARM_JOINTS] if len(alc['command']) >= ARM_JOINTS else np.zeros(ARM_JOINTS, dtype=np.float32)
    lerobot_frame['action.arm_right'] = arc['command'][:ARM_JOINTS] if len(arc['command']) >= ARM_JOINTS else np.zeros(ARM_JOINTS, dtype=np.float32)
    
    # 动作 - 夹爪
    gl_cmd = np.array([glc['command'][0] if glc and len(glc['command']) > 0 else 0.0], dtype=np.float32)
    gr_cmd = np.array([grc['command'][0] if grc and len(grc['command']) > 0 else 0.0], dtype=np.float32)
    lerobot_frame['action.gripper_left'] = gl_cmd
    lerobot_frame['action.gripper_right'] = gr_cmd
    
    # 动作 - 头部
    if hc and len(hc['command']) >= HEAD_JOINTS:
        head_cmd = hc['command'][:HEAD_JOINTS].astype(np.float32)
    else:
        head_cmd = np.zeros(HEAD_JOINTS, dtype=np.float32)
    lerobot_frame['action.head'] = head_cmd
    
    # 动作 - 腰部
    if tc and len(tc['command']) >= TORSO_JOINTS:
        torso_cmd = tc['command'][:TORSO_JOINTS].astype(np.float32)
    else:
        torso_cmd = np.zeros(TORSO_JOINTS, dtype=np.float32)
    lerobot_frame['action.torso'] = torso_cmd
    
    # 动作 - 底盘
    if cc and len(cc['command']) >= CHASSIS_JOINTS:
        chassis_cmd = cc['command'][:CHASSIS_JOINTS].astype(np.float32)
    else:
        chassis_cmd = np.zeros(CHASSIS_JOINTS, dtype=np.float32)
    lerobot_frame['action.chassis'] = chassis_cmd
    
    # 组合动作向量 (arm_left[7] + arm_right[7] + gripper[2] + head[2] + torso[4] + chassis[3] = 25)
    lerobot_frame['action'] = np.concatenate([
        lerobot_frame['action.arm_left'], lerobot_frame['action.arm_right'], 
        gl_cmd, gr_cmd,
        head_cmd, torso_cmd, chassis_cmd
    ])
    
    # 并行解码图像
    decoded_images = image_decoder.decode_batch(frame.get('images', {}))
    for img_key, img_data in decoded_images.items():
        lerobot_frame[f'observation.images.{img_key}'] = img_data
    
    return lerobot_frame


# ============================================================================
# 元数据与语言描述
# ============================================================================

def load_episode_metadata(bag_dir: Path) -> Dict[str, Any]:
    """
    从 episode 目录加载元数据，提取语言描述
    
    支持的元数据文件:
    - __loongdata_metadata.json: taskName, scene, operator 等
    - meta_info.json: robot_type, duration 等
    - task_description.txt: 自定义任务描述 (优先级最高)
    """
    metadata = {
        'task_name': None,
        'scene': None,
        'operator': None,
        'language_instruction': None,  # 最终使用的语言描述
    }
    
    # 1. 优先读取自定义描述文件
    custom_desc_file = bag_dir / "task_description.txt"
    if custom_desc_file.exists():
        with open(custom_desc_file, 'r', encoding='utf-8') as f:
            metadata['language_instruction'] = f.read().strip()
        return metadata
    
    # 2. 读取 __loongdata_metadata.json
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
    
    # 3. 构建语言描述
    if metadata['task_name']:
        parts = [metadata['task_name']]
        if metadata['scene']:
            parts.append(f"场景: {metadata['scene']}")
        metadata['language_instruction'] = ' - '.join(parts)
    
    return metadata


def find_bag_directories(root_dir: Path) -> List[Tuple[Path, Dict]]:
    """查找所有包含 rosbag 的目录，并加载元数据"""
    bag_dirs = []
    for item in sorted(root_dir.iterdir()):
        if item.is_dir():
            bag_path = item / "record" / "raw_data.bag"
            if bag_path.exists():
                metadata = load_episode_metadata(item)
                bag_dirs.append((item, metadata))
    return bag_dirs


# ============================================================================
# 主转换逻辑
# ============================================================================

def convert_all_to_single_dataset(
    bag_root: Path,
    output_dir: Path,
    repo_id: str,
    task_description: str = None,
    use_episode_tasks: bool = True,
):
    """
    将所有 bag 文件转换为单个 LeRobot 数据集
    
    Args:
        bag_root: 包含 rosbag 数据的根目录
        output_dir: 输出目录
        repo_id: LeRobot 数据集 ID
        task_description: 全局任务描述 (如果设置，所有 episode 使用相同描述)
        use_episode_tasks: 是否为每个 episode 使用独立的任务描述
                          从元数据文件读取，或使用自定义 task_description.txt
    """
    
    print(f"\n{'='*70}")
    print(f"Astribot ROS Bag -> LeRobot 单数据集转换 (优化版)")
    print(f"{'='*70}")
    print(f"输入目录: {bag_root}")
    print(f"输出目录: {output_dir}")
    print(f"Repo ID: {repo_id}")
    print(f"任务模式: {'每个 episode 独立任务' if use_episode_tasks and not task_description else '全局统一任务'}")
    print(f"{'='*70}\n")
    
    # 查找所有 bag 目录并加载元数据
    bag_dirs_with_meta = find_bag_directories(bag_root)
    
    if not bag_dirs_with_meta:
        print("错误: 没有找到任何包含 rosbag 的目录")
        return
    
    print(f"找到 {len(bag_dirs_with_meta)} 个 episode:")
    for bag_dir, metadata in bag_dirs_with_meta:
        task = metadata.get('language_instruction') or task_description or "Astribot manipulation task"
        print(f"  - {bag_dir.name}")
        print(f"    任务描述: {task}")
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
    
    try:
        # 处理每个 episode
        for episode_idx, (bag_dir, metadata) in enumerate(bag_dirs_with_meta):
            bag_path = bag_dir / "record" / "raw_data.bag"
            
            # 确定当前 episode 的任务描述
            if task_description:
                # 使用全局任务描述
                current_task = task_description
            elif use_episode_tasks and metadata.get('language_instruction'):
                # 使用 episode 独立的任务描述
                current_task = metadata['language_instruction']
            else:
                # 默认任务描述
                current_task = "Astribot manipulation task"
            
            print(f"\n[Episode {episode_idx}] 处理: {bag_dir.name}")
            print(f"  任务描述: {current_task}")
            
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
                lerobot_frame = convert_frame_to_lerobot(frame, current_task, image_decoder)
                dataset.add_frame(lerobot_frame)
            
            # 4. 保存当前 episode
            dataset.save_episode()
            
            # 记录帧数
            ep_frames = len(sync_frames)
            total_frames += ep_frames
            
            episode_info.append({
                'episode_index': episode_idx,
                'source': bag_dir.name,
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
    
    # 完成数据集
    dataset.finalize()
    
    # 收集所有唯一的任务描述
    unique_tasks = list(set(ep['task'] for ep in episode_info))
    
    # 生成转换报告
    report = {
        'timestamp': datetime.now().isoformat(),
        'repo_id': repo_id,
        'input_dir': str(bag_root),
        'output_dir': str(output_dir),
        'total_episodes': len(episode_info),
        'total_frames': total_frames,
        'total_tasks': len(unique_tasks),
        'tasks': unique_tasks,
        'fps': ASTRIBOT_FPS,
        'robot_type': ASTRIBOT_ROBOT_TYPE,
        'episodes': episode_info,
        'sync_config': {
            'base_topic': '/astribot_camera/head_rgbd/color_compress/compressed',
            'joint_tolerance_ms': JOINT_TOLERANCE_NS / 1_000_000,
            'image_tolerance_ms': IMAGE_TOLERANCE_NS / 1_000_000,
        }
    }
    
    report_path = output_dir / "conversion_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*70}")
    print(f"转换完成!")
    print(f"{'='*70}")
    print(f"数据集位置: {output_dir}")
    print(f"总 Episodes: {len(episode_info)}")
    print(f"总帧数: {total_frames}")
    print(f"报告: {report_path}")
    
    return dataset


def main():
    parser = argparse.ArgumentParser(
        description='将多个 Astribot ROS1 bag 转换为单个 LeRobot 数据集',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
组帧逻辑说明:
  1. 以 head 相机时间戳为基准 (30Hz)
  2. 对每个时间戳，查找 ±50ms 内最近的关节数据
  3. 同时查找 ±100ms 内最近的其他相机图像
  4. 必要数据齐全时生成一帧

语言描述 (Language Instruction) 支持:
  默认自动从元数据读取 (优先级从高到低):
    1. episode 目录下的 task_description.txt 文件
    2. __loongdata_metadata.json 中的 taskName 字段
    3. 命令行 --task 参数
    4. 默认值 "Astribot manipulation task"
  
  使用 --task 可强制所有 episode 使用相同描述
  使用 --no-episode-tasks 禁用从元数据读取
        """
    )
    parser.add_argument(
        'bag_root',
        type=Path,
        help='包含 rosbag 数据的根目录'
    )
    parser.add_argument(
        '--repo-id',
        type=str,
        default='astribot/dataset',
        help='LeRobot 数据集 repo_id (默认: astribot/dataset)'
    )
    parser.add_argument(
        '-o', '--output-dir',
        type=Path,
        default=None,
        help='输出目录 (默认: ./astribot_lerobot_dataset)'
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
        help='禁用从元数据读取 episode 独立任务描述'
    )
    
    args = parser.parse_args()
    
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
