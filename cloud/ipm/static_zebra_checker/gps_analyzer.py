"""
GPS数据分析模块
处理ROS1 bag文件中的GPS数据
"""

from pathlib import Path
from typing import List, Tuple
from rosbags.highlevel import AnyReader

from .utils import format_timestamp, calculate_displacement_value


def read_gps_from_bag(bag_path: str) -> List[Tuple[float, dict]]:
    """
    从bag文件中读取GPS数据
    
    Args:
        bag_path: bag文件路径
    
    Returns:
        List[(timestamp, gps_data)]: GPS数据列表
    """
    gps_data = []
    
    try:
        with AnyReader([Path(bag_path)]) as reader:
            for connection, timestamp, rawdata in reader.messages():
                if connection.topic == '/rs/gps':
                    msg = reader.deserialize(rawdata, connection.msgtype)
                    
                    # 提取GPS数据
                    data = {
                        'latitude': msg.latitude,
                        'longitude': msg.longitude,
                        'altitude': msg.altitude,
                        'status': msg.status.status,
                    }
                    
                    # 转换时间戳为秒
                    timestamp_s = timestamp / 1e9
                    gps_data.append((timestamp_s, data))
                    
    except Exception as e:
        print(f"读取bag文件GPS数据出错: {e}")
        import traceback
        traceback.print_exc()
        return []
    
    return gps_data


def calculate_displacement(gps_data: List[Tuple[float, dict]]) -> List[Tuple[float, float]]:
    """
    根据GPS数据计算相邻帧的绝对位移
    
    Args:
        gps_data: GPS数据列表
    
    Returns:
        List[(timestamp, displacement)]: 位移数据列表 (米)
    """
    displacements = []
    
    if len(gps_data) < 2:
        return displacements
    
    for i in range(1, len(gps_data)):
        t1, gps1 = gps_data[i-1]
        t2, gps2 = gps_data[i]
        
        # 计算位移
        displacement = calculate_displacement_value(gps1, gps2)
        displacements.append((t2, displacement))
    
    return displacements


def get_gps_statistics(gps_data: List[Tuple[float, dict]]) -> dict:
    """
    获取GPS数据的统计信息
    
    Args:
        gps_data: GPS数据列表
    
    Returns:
        dict: 统计信息
    """
    if not gps_data:
        return {}
    
    displacements = calculate_displacement(gps_data)
    
    if not displacements:
        return {
            'count': len(gps_data),
            'start_time': format_timestamp(gps_data[0][0]),
            'end_time': format_timestamp(gps_data[-1][0]),
        }
    
    displacement_values = [v for _, v in displacements]
    
    return {
        'count': len(gps_data),
        'start_time': format_timestamp(gps_data[0][0]),
        'end_time': format_timestamp(gps_data[-1][0]),
        'avg_displacement': sum(displacement_values) / len(displacement_values),
        'max_displacement': max(displacement_values),
        'min_displacement': min(displacement_values),
    }