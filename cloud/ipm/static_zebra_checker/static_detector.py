"""
静止检测模块
根据GPS位移数据检测车辆静止时间段
"""

from typing import List, Tuple

from .utils import calculate_cumulative_displacement


def is_static_frame(cumulative_displacement: float, threshold: float) -> bool:
    """
    判断单帧是否满足静止条件
    
    Args:
        cumulative_displacement: 累计位移
        threshold: 静止位移阈值
    
    Returns:
        bool: 是否静止
    """
    return cumulative_displacement <= threshold


def is_moving_frame(cumulative_displacement: float, threshold: float) -> bool:
    """
    判断单帧是否满足移动条件
    
    Args:
        cumulative_displacement: 累计位移
        threshold: 移动位移阈值
    
    Returns:
        bool: 是否移动
    """
    return cumulative_displacement > threshold


def compute_frame_states(displacements: List[Tuple[float, float]],
                        displacement_threshold: float,
                        window_size: int) -> Tuple[List[bool], List[bool]]:
    """
    计算每帧的静止和移动状态
    
    Args:
        displacements: 位移数据列表
        displacement_threshold: 位移阈值
        window_size: 滑动窗口大小
    
    Returns:
        Tuple[List[bool], List[bool]]: (静止状态列表, 移动状态列表)
    """
    cumulative = calculate_cumulative_displacement(displacements, window_size)
    
    is_static_frame_list = []
    is_moving_frame_list = []
    
    for i in range(len(displacements)):
        if i < window_size - 1:
            # 前面不足window_size帧
            is_static_frame_list.append(
                is_static_frame(cumulative[i], displacement_threshold)
            )
            is_moving_frame_list.append(False)
        else:
            is_static_frame_list.append(
                is_static_frame(cumulative[i], displacement_threshold)
            )
            is_moving_frame_list.append(
                is_moving_frame(cumulative[i], displacement_threshold)
            )
    
    return is_static_frame_list, is_moving_frame_list


def extract_static_periods_from_states(displacements: List[Tuple[float, float]],
                                      is_static_frame_list: List[bool],
                                      is_moving_frame_list: List[bool],
                                      min_duration: float,
                                      window_size: int) -> List[Tuple[float, float]]:
    """
    从帧状态中提取静止时间段
    
    Args:
        displacements: 位移数据列表
        is_static_frame_list: 静止状态列表
        is_moving_frame_list: 移动状态列表
        min_duration: 最小静止持续时间
        window_size: 滑动窗口大小
    
    Returns:
        List[Tuple[float, float]]: 静止时间段列表
    """
    static_periods = []
    in_static_period = False
    period_start = None
    
    for i in range(len(displacements)):
        timestamp, _ = displacements[i]
        
        if not in_static_period:
            # 当前不在静止时段，检查是否开始静止
            if is_static_frame_list[i]:
                in_static_period = True
                period_start = timestamp
        else:
            # 当前在静止时段，检查是否结束静止
            if is_moving_frame_list[i]:
                period_end = displacements[i - window_size + 1][0]
                duration = period_end - period_start
                
                if duration >= min_duration:
                    static_periods.append((period_start, period_end))
                
                in_static_period = False
    
    # 处理最后一个时段
    if in_static_period and period_start is not None:
        period_end = displacements[-1][0]
        duration = period_end - period_start
        
        if duration >= min_duration:
            static_periods.append((period_start, period_end))
    
    return static_periods


def find_static_periods(displacements: List[Tuple[float, float]], 
                        displacement_threshold: float = 0.01,
                        min_duration: float = 2.0,
                        window_size: int = 3) -> List[Tuple[float, float]]:
    """
    查找车辆静止的时间段（基于连续3帧累计运动距离判断）
    
    静止开始：连续window_size帧的累计位移小于阈值
    静止结束：连续window_size帧的累计位移大于阈值
    
    Args:
        displacements: 位移数据列表 [(timestamp, displacement), ...]
        displacement_threshold: 静止累计位移阈值 (米)，默认0.01米（1厘米）
        min_duration: 最小静止持续时间 (秒)，默认2秒
        window_size: 滑动窗口大小，默认3帧
    
    Returns:
        List[(start_time, end_time)]: 静止时间段列表
    """
    if len(displacements) < window_size:
        return []
    
    # 计算每帧的静止和移动状态
    is_static_frame_list, is_moving_frame_list = compute_frame_states(
        displacements, displacement_threshold, window_size
    )
    
    # 从帧状态中提取静止时间段
    static_periods = extract_static_periods_from_states(
        displacements, is_static_frame_list, is_moving_frame_list,
        min_duration, window_size
    )
    
    return static_periods