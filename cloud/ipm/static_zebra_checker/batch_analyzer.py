#!/usr/bin/env python3
"""
批量分析车辆静止时间段
根据bag文件中的GPS信息，遍历cam_front_right文件夹，
时间戳排序后，给出车辆静止的时间段，并检测斑马线
"""
import os
import sys
import yaml
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

# 导入模块
from .gps_analyzer import read_gps_from_bag, calculate_displacement, get_gps_statistics
from .camera_processor import (
    get_camera_timestamps,
    match_camera_to_static_periods,
    copy_static_images,
    copy_zebra_periods
)
from .static_detector import find_static_periods
from .zebra_detector import detect_zebra_crossing
from .utils import format_timestamp


def load_config(config_path: str) -> Dict:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        Dict: 配置字典
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        return {}


def check_zebra_crossing_consistency(
    frame_files: List[str],
    camera_path: Path,
    zebra_config: Dict,
    consistency_threshold: float = 0.7,
    sampling_config: Dict = None
) -> Dict:
    """
    检查静止时间段内斑马线检测的一致性
    
    Args:
        frame_files: 帧文件列表
        camera_path: 相机文件夹路径
        zebra_config: 斑马线检测配置
        consistency_threshold: 一致性阈值（检测到斑马线的帧数比例）
        sampling_config: 采样策略配置，如果为None则使用默认值
    
    Returns:
        Dict: 一致性检查结果
            {
                'has_zebra': bool,              # 是否检测到斑马线
                'sample_count': int,            # 采样帧数
                'detected_count': int,          # 检测到的帧数
                'consistency': float,           # 一致性比例
                'detections': List[bool]        # 每帧检测结果列表
            }
    """
    # 默认采样配置
    if sampling_config is None:
        sampling_config = {
            'small_frame_threshold': 5,
            'small_frame_sample_all': True,
            'medium_frame_threshold': 30,
            'medium_frame_sample_ratio': 0.33,
            'medium_frame_min_samples': 5,
            'large_frame_max_samples': 15,
            'large_frame_sample_ratio': 0.5
        }
    
    total_frames = len(frame_files)
    
    # 智能采样策略
    if total_frames <= sampling_config['small_frame_threshold']:
        # 小片段：全部采样
        sample_indices = list(range(total_frames))
    elif total_frames <= sampling_config['medium_frame_threshold']:
        # 中等片段：按比例采样，但不低于最小采样帧数
        sample_count = max(
            sampling_config['medium_frame_min_samples'],
            int(total_frames * sampling_config['medium_frame_sample_ratio'])
        )
        step = total_frames // sample_count
        sample_indices = list(range(0, total_frames, step))[:sample_count]
    else:
        # 大片段：按比例采样，但不超过最大采样帧数
        sample_count = min(
            sampling_config['large_frame_max_samples'],
            int(total_frames * sampling_config['large_frame_sample_ratio'])
        )
        step = total_frames // sample_count
        sample_indices = list(range(0, total_frames, step))[:sample_count]
    
    # 检测每一帧
    detections = []
    for idx in sample_indices:
        image_path = camera_path / frame_files[idx]
        result = detect_zebra_crossing(str(image_path), **zebra_config)
        detections.append(result)
    
    # 统计结果
    detected_count = sum(detections)
    consistency = detected_count / len(detections) if detections else 0
    has_zebra = consistency >= consistency_threshold
    
    return {
        'has_zebra': has_zebra,
        'sample_count': len(detections),
        'detected_count': detected_count,
        'consistency': consistency,
        'detections': detections
    }


def check_zebra_crossing_in_periods(
    matched_periods: List[Tuple[float, float, int, List[str]]],
    camera_dir: str,
    config: Dict
) -> List[Tuple[float, float, int, List[str], bool, Dict]]:
    """
    检查每个静止时间段是否包含斑马线（基于一致性检查）
    
    Args:
        matched_periods: 匹配的静止时间段列表
        camera_dir: 相机文件夹路径
        config: 配置字典
    
    Returns:
        List[(start, end, frame_count, frame_files, has_zebra_crossing, consistency_info)]: 
            带有斑马线标记和一致性信息的静止时间段
    """
    camera_path = Path(camera_dir)
    zebra_config = config.get('zebra_detection', {})
    consistency_config = config.get('consistency_check', {})
    sampling_config = consistency_config.get('sampling', {})
    
    for i, (start, end, frame_count, frame_files) in enumerate(matched_periods):
        print(f"  检查第 {i+1} 个静止时段的斑马线一致性...")
        
        # 执行一致性检查
        result = check_zebra_crossing_consistency(
            frame_files=frame_files,
            camera_path=camera_path,
            zebra_config=zebra_config,
            consistency_threshold=consistency_config.get('consistency_threshold', 0.7),
            sampling_config=sampling_config if sampling_config else None
        )
        
        # 输出详细结果
        print(f"    采样帧数: {result['sample_count']}")
        print(f"    检测到: {result['detected_count']} 帧")
        print(f"    一致性: {result['consistency']:.1%}")
        print(f"    判定结果: {'✓ 有' if result['has_zebra'] else '✗ 无'}斑马线")
        
        # 更新元组
        matched_periods[i] = (
            start, 
            end, 
            frame_count, 
            frame_files, 
            result['has_zebra'],
            result
        )
    
    return matched_periods


def compute_statistics(matched_periods: List[Tuple[float, float, int, List[str], bool, Dict]],
                      gps_data: List[Tuple[float, dict]]) -> Dict:
    """
    计算统计信息
    
    Args:
        matched_periods: 匹配的静止时间段列表
        gps_data: GPS数据列表
    
    Returns:
        Dict: 统计信息
    """
    total_static_time = 0
    total_static_frames = 0
    zebra_count = 0
    
    for start, end, frame_count, frame_files, has_zebra, consistency_info in matched_periods:
        duration = end - start
        total_static_time += duration
        total_static_frames += frame_count
        if has_zebra:
            zebra_count += 1
    
    # 计算静止时间占比
    total_time = gps_data[-1][0] - gps_data[0][0]
    static_ratio = (total_static_time / total_time) * 100 if total_time > 0 else 0
    
    return {
        'total_static_time': total_static_time,
        'total_static_frames': total_static_frames,
        'zebra_count': zebra_count,
        'static_ratio': static_ratio
    }


def read_gps_and_calculate_displacement(bag_file: str) -> Tuple[List[Tuple[float, dict]], List[Tuple[float, float]], Dict]:
    """
    读取GPS数据并计算位移
    
    Args:
        bag_file: bag文件路径
    
    Returns:
        Tuple: (GPS数据, 位移数据, GPS统计信息)
    """
    print(f"  读取bag文件中的GPS数据...")
    gps_data = read_gps_from_bag(bag_file)
    
    if not gps_data:
        return [], {}, {}
    
    print(f"    成功读取 {len(gps_data)} 条GPS数据")
    print(f"    时间范围: {format_timestamp(gps_data[0][0])} - {format_timestamp(gps_data[-1][0])}")
    
    print(f"  计算相邻帧的绝对位移...")
    displacements = calculate_displacement(gps_data)
    
    if not displacements:
        return gps_data, [], {}
    
    print(f"    成功计算 {len(displacements)} 个位移数据点")
    
    # 获取GPS统计信息
    stats = get_gps_statistics(gps_data)
    
    if stats:
        print(f"    平均位移: {stats.get('avg_displacement', 0):.3f} 米")
        print(f"    最大位移: {stats.get('max_displacement', 0):.3f} 米")
        print(f"    最小位移: {stats.get('min_displacement', 0):.3f} 米")
    
    return gps_data, displacements, stats


def find_and_match_static_periods(displacements: List[Tuple[float, float]],
                                  camera_timestamps: List[float],
                                  camera_dir: str,
                                  config: Dict) -> List[Tuple[float, float, int, List[str]]]:
    """
    查找静止时间段并匹配相机帧
    
    Args:
        displacements: 位移数据列表
        camera_timestamps: 相机时间戳列表
        camera_dir: 相机文件夹路径
        config: 配置字典
    
    Returns:
        List: 匹配的静止时间段列表
    """
    static_config = config.get('static_detection', {})
    
    print(f"  查找静止时间段...")
    static_periods = find_static_periods(
        displacements,
        displacement_threshold=static_config.get('displacement_threshold', 0.01),
        min_duration=static_config.get('min_duration', 2.0),
        window_size=static_config.get('window_size', 3)
    )
    
    if not static_periods:
        print("    未检测到静止时间段")
        return []
    
    print(f"    检测到 {len(static_periods)} 个静止时间段")
    
    print(f"  匹配静止时间段与相机帧...")
    matched_periods = match_camera_to_static_periods(
        static_periods, camera_timestamps, camera_dir
    )
    
    return matched_periods


def analyze_single_bag(bag_dir: str, config: Dict) -> Dict:
    """
    分析单个bag文件
    
    Args:
        bag_dir: bag文件所在目录
        config: 配置字典
    
    Returns:
        Dict: 分析结果
    """
    bag_path = Path(bag_dir)
    bag_name = bag_path.name
    bag_file = bag_path / f"{bag_name}.bag"
    camera_dir = bag_path / config.get('camera', {}).get('camera_dir', 'cam_front_right')
    
    result = {
        'bag_name': bag_name,
        'bag_dir': str(bag_path),
        'gps_count': 0,
        'camera_count': 0,
        'static_periods': [],
        'total_static_time': 0,
        'total_static_frames': 0,
        'zebra_count': 0,
        'static_ratio': 0,
        'success': False,
        'error': None
    }
    
    print(f"\n{'='*80}")
    print(f"分析数据包: {bag_name}")
    print(f"{'='*80}")
    
    # 1. 读取GPS数据并计算位移
    gps_data, displacements, gps_stats = read_gps_and_calculate_displacement(str(bag_file))
    
    if not gps_data:
        result['error'] = "未能读取到GPS数据"
        print(f"错误: {result['error']}")
        return result
    
    result['gps_count'] = len(gps_data)
    
    if not displacements:
        result['error'] = "未能计算位移数据"
        print(f"错误: {result['error']}")
        return result
    
    # 2. 查找静止时间段
    matched_periods = find_and_match_static_periods(
        displacements, [], str(camera_dir), config
    )
    
    if not matched_periods:
        result['success'] = True
        return result
    
    # 3. 获取相机时间戳
    print(f"  读取相机图片时间戳...")
    camera_timestamps = get_camera_timestamps(str(camera_dir))
    
    if not camera_timestamps:
        result['error'] = "未能读取相机图片时间戳"
        print(f"错误: {result['error']}")
        return result
    
    result['camera_count'] = len(camera_timestamps)
    print(f"    成功读取 {len(camera_timestamps)} 张图片")
    print(f"    时间范围: {format_timestamp(camera_timestamps[0])} - {format_timestamp(camera_timestamps[-1])}")
    
    # 4. 重新匹配静止时间段（使用实际的相机时间戳）
    matched_periods = find_and_match_static_periods(
        displacements, camera_timestamps, str(camera_dir), config
    )
    
    if not matched_periods:
        result['success'] = True
        return result
    
    # 5. 检测斑马线
    print(f"  检测斑马线...")
    matched_periods = check_zebra_crossing_in_periods(matched_periods, str(camera_dir), config)
    
    result['static_periods'] = matched_periods
    
    # 6. 计算统计信息
    statistics = compute_statistics(matched_periods, gps_data)
    result.update(statistics)
    
    # 7. 拷贝静止图片
    print(f"\n  拷贝静止图片...")
    copy_static_images(matched_periods, str(camera_dir), str(bag_path))
    
    # 8. 拷贝斑马线图片
    print(f"\n  拷贝斑马线图片...")
    copy_zebra_periods(matched_periods, str(camera_dir), str(bag_path))
    
    result['success'] = True
    
    return result


def print_result_summary(result: Dict):
    """
    打印单个数据包的分析结果摘要
    
    Args:
        result: 分析结果字典
    """
    print(f"\n{'='*80}")
    print(f"分析结果: {result['bag_name']}")
    print(f"{'='*80}")
    
    if not result['success']:
        print(f"\n分析失败: {result['error']}")
        return
    
    if not result['static_periods']:
        print(f"\n未检测到车辆静止时间段")
        return
    
    print(f"\n共检测到 {len(result['static_periods'])} 个静止时间段:\n")
    
    for i, (start, end, frame_count, frame_files, has_zebra, consistency_info) in enumerate(result['static_periods'], 1):
        duration = end - start
        
        print(f"序号: {i}")
        print(f"  开始时间: {format_timestamp(start)}")
        print(f"  结束时间: {format_timestamp(end)}")
        print(f"  持续时长: {duration:.2f} 秒")
        print(f"  帧数: {frame_count}")
        print(f"  斑马线: {'有' if has_zebra else '无'}")
        if consistency_info:
            print(f"  一致性: {consistency_info['consistency']:.1%} "
                  f"({consistency_info['detected_count']}/{consistency_info['sample_count']})")
        print()
    
    print(f"{'-'*80}")
    print(f"总计: 静止时长 {result['total_static_time']:.2f} 秒, 共 {result['total_static_frames']} 帧")
    print(f"斑马线时段: {result['zebra_count']} 个")
    print(f"静止时间占比: {result['static_ratio']:.2f}%")
    print(f"{'='*80}")


def print_all_summary(results: List[Dict]):
    """
    打印所有数据包的汇总结果
    
    Args:
        results: 所有数据包的分析结果列表
    """
    print(f"\n\n{'='*80}")
    print(f"所有数据包汇总")
    print(f"{'='*80}")
    
    total_bags = len(results)
    total_static_time = 0
    total_static_frames = 0
    total_zebra_periods = 0
    
    for result in results:
        if result['success']:
            total_static_time += result['total_static_time']
            total_static_frames += result['total_static_frames']
            total_zebra_periods += result['zebra_count']
    
    print(f"\n数据包总数: {total_bags}")
    print(f"总静止时长: {total_static_time:.2f} 秒")
    print(f"总静止帧数: {total_static_frames} 帧")
    print(f"总斑马线时段: {total_zebra_periods} 个")
    print(f"\n{'='*80}")
    
    # 按数据包列出有斑马线的静止时段
    print(f"\n有斑马线的静止时段详情:")
    print(f"{'='*80}")
    
    for result in results:
        if result['success'] and result['static_periods']:
            zebra_periods = [(i, p) for i, p in enumerate(result['static_periods'], 1) if p[4]]
            
            if zebra_periods:
                print(f"\n{result['bag_name']}:")
                for i, (start, end, frame_count, frame_files, has_zebra, consistency_info) in zebra_periods:
                    duration = end - start
                    consistency_str = f" ({consistency_info['consistency']:.1%})" if consistency_info else ""
                    print(f"  时段 {i}: {format_timestamp(start)} - {format_timestamp(end)} ({duration:.2f}秒, {frame_count}帧){consistency_str}")
    
    print(f"\n{'='*80}")


def extract_zebra_frame_numbers(result: Dict, camera_dir: str) -> List[int]:
    """
    从分析结果中提取有斑马线的静止时间段的图片序号
    
    Args:
        result: analyze_single_bag 的返回结果
        camera_dir: 相机图片目录路径
    
    Returns:
        List[int]: 图片序号列表（从1开始）
    """
    frame_numbers = []
    
    if not result.get('success'):
        print(f"分析失败: {result.get('error')}")
        return frame_numbers
    
    static_periods = result.get('static_periods', [])
    camera_path = Path(camera_dir)
    
    # 获取所有图片的时间戳并排序
    all_images = sorted([int(f.stem) for f in camera_path.glob('*.jpg')])
    
    if not all_images:
        print(f"未找到相机图片: {camera_dir}")
        return frame_numbers
    
    # 创建时间戳到序号的映射（序号从1开始）
    timestamp_to_index = {ts: idx + 1 for idx, ts in enumerate(all_images)}
    
    # 遍历所有静止时间段，筛选有斑马线的时段
    zebra_period_count = 0
    for i, period_data in enumerate(static_periods):
        # 解构时段数据
        if len(period_data) >= 6:
            start, end, frame_count, frame_files, has_zebra, consistency_info = period_data[:6]
        else:
            # 兼容旧版本数据结构
            start, end, frame_count, frame_files, has_zebra = period_data
            consistency_info = {}
        
        if not has_zebra:
            continue
        
        zebra_period_count += 1
        
        print(f"\n斑马线时段 {zebra_period_count}:")
        print(f"  帧数: {frame_count}")
        print(f"  一致性: {consistency_info.get('consistency', 0)*100:.1f}%")
        
        # 选择2张图片：按时间刻度均匀选择（1/4和3/4位置）
        if len(frame_files) >= 4:
            # 选择1/4和3/4位置的图片
            idx1 = len(frame_files) // 4
            idx2 = (len(frame_files) * 3) // 4
            selected_files = [frame_files[idx1], frame_files[idx2]]
        elif len(frame_files) >= 2:
            # 如果帧数较少，选择第一个和最后一个
            selected_files = [frame_files[0], frame_files[-1]]
        else:
            # 只有1帧，选择它
            selected_files = frame_files[:1]
        
        # 提取时间戳并转换为序号
        for filename in selected_files:
            timestamp_ns = int(filename.split('.')[0])
            if timestamp_ns in timestamp_to_index:
                frame_idx = timestamp_to_index[timestamp_ns]
                frame_numbers.append(frame_idx)
                print(f"  选择图片: {filename} -> 序号: {frame_idx}")
            else:
                print(f"  警告: 未找到时间戳 {timestamp_ns} 对应的图片")
    
    return frame_numbers


def main():
    parser = argparse.ArgumentParser(description='批量分析车辆静止时间段和斑马线')
    # parser.add_argument('bag_dir', type=str, help='包含bag文件的目录')
    parser.add_argument('--config', type=str, 
                       default=str(Path(__file__).parent / 'config.yaml'),
                       help='配置文件路径')
    
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)
    
    # 查找所有包含.bag文件的子目录
    # bag_dirs = get_bag_directories(args.bag_dir)
    bag_path = "/home/geely/Downloads/S39_0117"
    bags = os.listdir(bag_path)
    bag_dirs = [os.path.join(bag_path, bag) for bag in bags]

    if not bag_dirs:
        print(f"错误: 在 {args.bag_dir} 中未找到bag文件")
        return
    
    print(f"找到 {len(bag_dirs)} 个数据包:")
    
    # 分析所有数据包
    results = []
    for bag_dir in bag_dirs:
        result = analyze_single_bag(str(bag_dir), config)
        results.append(result)
        print_result_summary(result)
    
    # 打印汇总
    print_all_summary(results)


if __name__ == "__main__":
    main()