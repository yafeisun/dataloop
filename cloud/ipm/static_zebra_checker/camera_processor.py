"""
相机图片处理模块
处理相机图片时间戳、匹配静止时段、拷贝图片等
"""

import shutil
from pathlib import Path
from typing import List, Tuple

from .utils import format_timestamp, ensure_dir


def get_camera_timestamps(camera_dir: str) -> List[float]:
    """
    获取相机文件夹中所有图片的时间戳
    
    Args:
        camera_dir: 相机文件夹路径
    
    Returns:
        List[float]: 排序后的时间戳列表
    """
    timestamps = []
    
    try:
        camera_path = Path(camera_dir)
        if not camera_path.exists():
            print(f"相机文件夹不存在: {camera_dir}")
            return []
        
        # 遍历文件夹中的所有.jpg文件
        for file in camera_path.glob('*.jpg'):
            # 从文件名中提取时间戳（纳秒）
            timestamp_ns = int(file.stem)
            # 转换为秒
            timestamp_s = timestamp_ns / 1e9
            timestamps.append(timestamp_s)
        
        # 按时间戳排序
        timestamps.sort()
        
    except Exception as e:
        print(f"读取相机时间戳出错: {e}")
        import traceback
        traceback.print_exc()
        return []
    
    return timestamps


def match_camera_to_static_periods(static_periods: List[Tuple[float, float]],
                                   camera_timestamps: List[float],
                                   camera_dir: str) -> List[Tuple[float, float, int, List[str]]]:
    """
    将静止时间段与相机图片匹配
    
    Args:
        static_periods: 静止时间段列表
        camera_timestamps: 相机图片时间戳列表
        camera_dir: 相机文件夹路径
    
    Returns:
        List[(start_time, end_time, frame_count, image_files)]: 带有帧数和图片文件名的静止时间段
    """
    matched_periods = []
    camera_path = Path(camera_dir)
    
    # 创建时间戳到文件名的映射
    timestamp_to_file = {}
    for file in camera_path.glob('*.jpg'):
        timestamp_ns = int(file.stem)
        timestamp_s = timestamp_ns / 1e9
        timestamp_to_file[timestamp_s] = file.name
    
    for start, end in static_periods:
        # 找出该时间段内的相机图片
        frame_files = []
        for ts in camera_timestamps:
            if start <= ts <= end:
                if ts in timestamp_to_file:
                    frame_files.append(timestamp_to_file[ts])
        
        # 去掉前2个和后2个图片
        if len(frame_files) > 4:
            frame_files = frame_files[2:-2]
        
        frame_count = len(frame_files)
        matched_periods.append((start, end, frame_count, frame_files))
    
    return matched_periods


def copy_images_to_directory(image_files: List[str],
                            camera_dir: str,
                            output_dir: str,
                            period_index: int) -> List[str]:
    """
    拷贝图片到指定目录
    
    Args:
        image_files: 图片文件名列表
        camera_dir: 相机文件夹路径
        output_dir: 输出目录路径
        period_index: 时段索引（用于命名子文件夹）
    
    Returns:
        List[str]: 拷贝的图片文件路径列表
    """
    camera_path = Path(camera_dir)
    output_path = ensure_dir(output_dir)
    
    # 创建子文件夹
    subfolder_name = f"period_{period_index}"
    subfolder_path = output_path / subfolder_name
    ensure_dir(str(subfolder_path))
    
    copied_files = []
    for filename in image_files:
        src_file = camera_path / filename
        if src_file.exists():
            dst_file = subfolder_path / filename
            shutil.copy2(src_file, dst_file)
            copied_files.append(str(dst_file))
    
    return copied_files


def copy_static_images(matched_periods: List[Tuple[float, float, int, List[str], bool, dict]],
                      camera_dir: str,
                      bag_dir: str) -> List[str]:
    """
    拷贝静止时间段的图片到static_periods目录
    
    Args:
        matched_periods: 匹配的静止时间段列表（包含斑马线标记和一致性信息）
        camera_dir: 相机文件夹路径
        bag_dir: bag文件所在目录
    
    Returns:
        List[str]: 创建的文件夹路径列表
    """
    created_dirs = []
    
    for i, (start, end, frame_count, frame_files, has_zebra, consistency_info) in enumerate(matched_periods, 1):
        output_dir = Path(bag_dir) / "static_periods"
        copied_files = copy_images_to_directory(frame_files, camera_dir, str(output_dir), i)
        
        if copied_files:
            period_dir = output_dir / f"period_{i}"
            created_dirs.append(str(period_dir))
            print(f"    已拷贝 {frame_count} 张图片到: {period_dir}")
    
    return created_dirs


def copy_zebra_periods(matched_periods: List[Tuple[float, float, int, List[str], bool, dict]],
                       camera_dir: str,
                       bag_dir: str) -> List[str]:
    """
    拷贝有斑马线的静止时段图片到zebra_periods目录
    
    Args:
        matched_periods: 匹配的静止时间段列表（包含斑马线标记和一致性信息）
        camera_dir: 相机文件夹路径
        bag_dir: bag文件所在目录
    
    Returns:
        List[str]: 创建的文件夹路径列表
    """
    created_dirs = []
    zebra_index = 1
    
    for start, end, frame_count, frame_files, has_zebra, consistency_info in matched_periods:
        if has_zebra:
            output_dir = Path(bag_dir) / "zebra_periods"
            copied_files = copy_images_to_directory(frame_files, camera_dir, str(output_dir), zebra_index)
            
            if copied_files:
                period_dir = output_dir / f"period_{zebra_index}"
                created_dirs.append(str(period_dir))
                print(f"    已拷贝 {frame_count} 张斑马线图片到: {period_dir}")
                zebra_index += 1
    
    return created_dirs