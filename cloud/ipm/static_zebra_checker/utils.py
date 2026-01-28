"""
工具函数模块
提供通用的工具函数
"""

import os
from pathlib import Path
from typing import List, Tuple


def format_timestamp(timestamp: float) -> str:
    """
    格式化时间戳为原始纳秒格式
    
    Args:
        timestamp: Unix时间戳（秒）
    
    Returns:
        str: 原始纳秒时间戳字符串
    """
    timestamp_ns = int(timestamp * 1e9)
    return str(timestamp_ns)


def ensure_dir(directory: str) -> Path:
    """
    确保目录存在，如果不存在则创建
    
    Args:
        directory: 目录路径
    
    Returns:
        Path: 目录的Path对象
    """
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def calculate_displacement_value(gps1: dict, gps2: dict) -> float:
    """
    计算两个GPS点之间的绝对位移（米）
    
    Args:
        gps1: 第一个GPS点，包含latitude和longitude
        gps2: 第二个GPS点，包含latitude和longitude
    
    Returns:
        float: 位移距离（米）
    """
    # 计算经纬度差
    dlat = gps2['latitude'] - gps1['latitude']
    dlon = gps2['longitude'] - gps1['longitude']
    
    # 简化计算：假设局部区域，1度约等于111km
    lat_avg = (gps1['latitude'] + gps2['latitude']) / 2
    lat_to_m = 111320.0  # 1度纬度约等于111320米
    lon_to_m = 111320.0 * (3.14159265359 / 180.0) * abs(lat_avg)
    
    # 计算绝对位移 (米)
    displacement = ((dlat * lat_to_m) ** 2 + (dlon * lon_to_m) ** 2) ** 0.5
    return displacement


def calculate_cumulative_displacement(displacements: List[Tuple[float, float]], 
                                     window_size: int) -> List[float]:
    """
    计算滑动窗口的累计位移
    
    Args:
        displacements: 位移数据列表 [(timestamp, displacement), ...]
        window_size: 窗口大小
    
    Returns:
        List[float]: 累计位移列表
    """
    cumulative = []
    for i in range(len(displacements)):
        if i < window_size - 1:
            # 前面不足window_size帧，检查从0到i的所有帧
            window_displacements = [displacements[j][1] for j in range(0, i + 1)]
            cumulative.append(sum(window_displacements))
        else:
            # 检查连续window_size帧的累计位移
            window_displacements = [displacements[j][1] 
                                  for j in range(i - window_size + 1, i + 1)]
            cumulative.append(sum(window_displacements))
    return cumulative