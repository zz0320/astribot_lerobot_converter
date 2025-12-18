#!/usr/bin/env python3
"""
ROS1 Bag 数据提取脚本
从 rosbag 文件中提取各话题数据并保存到对应文件夹
"""

import os
import sys
import json
import argparse
import struct
import csv
from pathlib import Path
from datetime import datetime
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore
from tqdm import tqdm
import numpy as np


def sanitize_topic_name(topic_name: str) -> str:
    """将话题名转换为合法的文件夹名"""
    name = topic_name.lstrip('/')
    name = name.replace('/', '_')
    return name


class ROS1MessageParser:
    @staticmethod
    def read_string(data: bytes, offset: int) -> tuple:
        length = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        string_val = data[offset:offset+length].decode('utf-8', errors='replace')
        return string_val, offset + length
    
    @staticmethod
    def read_string_array(data: bytes, offset: int) -> tuple:
        count = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        strings = []
        for _ in range(count):
            s, offset = ROS1MessageParser.read_string(data, offset)
            strings.append(s)
        return strings, offset
    
    @staticmethod
    def read_float64_array(data: bytes, offset: int) -> tuple:
        length = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        if length > 0:
            values = list(struct.unpack_from(f'<{length}d', data, offset))
            return values, offset + length * 8
        return [], offset
    
    @staticmethod
    def read_header(data: bytes, offset: int = 0) -> tuple:
        seq = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        secs, nsecs = struct.unpack_from('<II', data, offset)
        offset += 8
        frame_id, offset = ROS1MessageParser.read_string(data, offset)
        return {'seq': seq, 'stamp': {'secs': secs, 'nsecs': nsecs}, 'frame_id': frame_id}, offset


def parse_joint_state_msg(raw_data: bytes) -> dict:
    try:
        offset = 0
        header, offset = ROS1MessageParser.read_header(raw_data, offset)
        mode = struct.unpack_from('<b', raw_data, offset)[0]
        offset += 1
        names, offset = ROS1MessageParser.read_string_array(raw_data, offset)
        position, offset = ROS1MessageParser.read_float64_array(raw_data, offset)
        velocity, offset = ROS1MessageParser.read_float64_array(raw_data, offset)
        acceleration, offset = ROS1MessageParser.read_float64_array(raw_data, offset)
        torque, offset = ROS1MessageParser.read_float64_array(raw_data, offset)
        
        return {
            'header': header, 'mode': mode, 'name': names,
            'position': position, 'velocity': velocity,
            'acceleration': acceleration, 'torque': torque,
        }
    except Exception as e:
        return {'parse_error': str(e)}


def parse_joint_controller_msg(raw_data: bytes) -> dict:
    try:
        offset = 0
        header, offset = ROS1MessageParser.read_header(raw_data, offset)
        mode = struct.unpack_from('<b', raw_data, offset)[0]
        offset += 1
        names, offset = ROS1MessageParser.read_string_array(raw_data, offset)
        command, offset = ROS1MessageParser.read_float64_array(raw_data, offset)
        
        return {'header': header, 'mode': mode, 'name': names, 'command': command}
    except Exception as e:
        return {'parse_error': str(e)}


def save_compressed_image(raw_data: bytes, output_path: Path, typestore) -> bool:
    try:
        msg = typestore.deserialize_ros1(raw_data, 'sensor_msgs/msg/CompressedImage')
        image_data = bytes(msg.data)
        fmt = msg.format.lower() if hasattr(msg, 'format') else 'jpeg'
        ext = '.png' if 'png' in fmt else '.jpg'
        output_path = output_path.with_suffix(ext)
        
        with open(output_path, 'wb') as f:
            f.write(image_data)
        return True
    except:
        return False


def save_as_csv(data_list: list, output_path: Path, msg_type: str):
    if not data_list:
        return
    
    valid_items = [item for item in data_list if 'parse_error' not in item]
    if not valid_items:
        return
    
    first_item = valid_items[0]
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        if 'RobotJointController' in msg_type:
            if 'command' not in first_item or not first_item['command']:
                return
            num_joints = len(first_item['command'])
            headers = ['index', 'timestamp', 'mode'] + [f'cmd_{i}' for i in range(num_joints)]
            writer.writerow(headers)
            for item in valid_items:
                row = [item.get('_index', ''), item.get('_timestamp', ''), item.get('mode', 0)]
                row.extend(item.get('command', [0]*num_joints))
                writer.writerow(row)
        
        elif 'RobotJointState' in msg_type:
            has_pos = 'position' in first_item and first_item['position']
            has_vel = 'velocity' in first_item and first_item['velocity']
            has_tor = 'torque' in first_item and first_item['torque']
            
            if has_pos:
                num = len(first_item['position'])
            elif has_vel:
                num = len(first_item['velocity'])
            else:
                return
            
            headers = ['index', 'timestamp', 'mode']
            if has_pos:
                headers.extend([f'pos_{i}' for i in range(num)])
            if has_vel:
                headers.extend([f'vel_{i}' for i in range(num)])
            if has_tor:
                headers.extend([f'torque_{i}' for i in range(num)])
            
            writer.writerow(headers)
            
            for item in valid_items:
                row = [item.get('_index', ''), item.get('_timestamp', ''), item.get('mode', 0)]
                if has_pos:
                    row.extend(item.get('position', [0]*num))
                if has_vel:
                    row.extend(item.get('velocity', [0]*num))
                if has_tor:
                    row.extend(item.get('torque', [0]*num))
                writer.writerow(row)


def extract_bag(bag_path: str, output_dir: str = None):
    """提取 bag 文件中的所有话题数据"""
    bag_path = Path(bag_path)
    
    if not bag_path.exists():
        print(f"错误: 文件不存在 - {bag_path}")
        return
    
    if output_dir is None:
        output_dir = bag_path.parent / f"{bag_path.stem}_extracted"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"提取: {bag_path}")
    print(f"输出: {output_dir}")
    print(f"{'='*60}\n")
    
    typestore = get_typestore(Stores.ROS1_NOETIC)
    
    with Reader(bag_path) as reader:
        topics_info = reader.topics
        total_messages = reader.message_count
        
        print(f"话题数量: {len(topics_info)}")
        print(f"消息总数: {total_messages}\n")
        
        # 创建文件夹
        topic_dirs = {}
        topic_counters = {}
        topic_data = {}
        
        for topic_name in topics_info.keys():
            folder_name = sanitize_topic_name(topic_name)
            topic_dir = output_dir / folder_name
            topic_dir.mkdir(parents=True, exist_ok=True)
            topic_dirs[topic_name] = topic_dir
            topic_counters[topic_name] = 0
            topic_data[topic_name] = []
        
        # 提取消息
        for connection, timestamp, raw_data in tqdm(reader.messages(), total=total_messages, desc="提取"):
            topic_name = connection.topic
            msg_type = connection.msgtype
            topic_dir = topic_dirs[topic_name]
            idx = topic_counters[topic_name]
            
            if 'CompressedImage' in msg_type:
                output_path = topic_dir / f"{idx:06d}_{timestamp}"
                save_compressed_image(raw_data, output_path, typestore)
            
            elif 'RobotJointState' in msg_type:
                data = parse_joint_state_msg(raw_data)
                data['_timestamp'] = timestamp
                data['_index'] = idx
                topic_data[topic_name].append(data)
            
            elif 'RobotJointController' in msg_type:
                data = parse_joint_controller_msg(raw_data)
                data['_timestamp'] = timestamp
                data['_index'] = idx
                topic_data[topic_name].append(data)
            
            else:
                output_path = topic_dir / f"{idx:06d}_{timestamp}.bin"
                with open(output_path, 'wb') as f:
                    f.write(raw_data)
            
            topic_counters[topic_name] += 1
        
        # 保存数据文件
        print("\n保存数据文件...")
        for topic_name, data_list in topic_data.items():
            if data_list:
                topic_dir = topic_dirs[topic_name]
                
                with open(topic_dir / "data.json", 'w') as f:
                    json.dump(data_list, f, indent=2, default=str)
                
                msg_type = topics_info[topic_name].msgtype
                if 'RobotJoint' in msg_type:
                    try:
                        save_as_csv(data_list, topic_dir / "data.csv", msg_type)
                    except Exception as e:
                        print(f"CSV 保存失败 ({topic_name}): {e}")
        
        # 保存摘要
        summary = {
            'bag_file': str(bag_path),
            'extraction_time': datetime.now().isoformat(),
            'total_messages': total_messages,
            'topics': {
                topic_name: {
                    'folder': sanitize_topic_name(topic_name),
                    'msg_type': topics_info[topic_name].msgtype,
                    'msg_count': topics_info[topic_name].msgcount,
                    'extracted_count': topic_counters[topic_name]
                }
                for topic_name in topics_info
            }
        }
        
        with open(output_dir / "extraction_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n提取完成: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='从 ROS1 bag 文件提取数据')
    parser.add_argument('bag_files', nargs='+', help='Bag 文件路径')
    parser.add_argument('-o', '--output_dir', default=None, help='输出目录')
    
    args = parser.parse_args()
    
    for bag_file in args.bag_files:
        extract_bag(bag_file, args.output_dir)


if __name__ == '__main__':
    main()

