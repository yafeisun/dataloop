"""
相机尺寸检查和修正工具
检查并修正 car_*.yaml 文件中的相机 ImageSize 配置，使其与实际图片尺寸一致
"""

import os
import yaml
import cv2
from pathlib import Path
from typing import Dict, List, Tuple


def get_camera_actual_size(camera_dir: str) -> Tuple[int, int]:
    """
    获取相机文件夹中图片的实际尺寸
    
    Args:
        camera_dir: 相机文件夹路径
    
    Returns:
        Tuple[int, int]: (width, height)
    """
    camera_path = Path(camera_dir)
    
    # 获取第一张图片
    images = sorted([f for f in camera_path.glob('*.jpg')])
    
    if not images:
        raise ValueError(f"未找到图片文件: {camera_dir}")
    
    # 读取第一张图片获取尺寸
    img = cv2.imread(str(camera_path / images[0]))
    if img is None:
        raise ValueError(f"无法读取图片: {camera_path / images[0]}")
    
    h, w = img.shape[:2]
    return (w, h)


def get_camera_folder_to_topic_mapping() -> Dict[str, str]:
    """
    获取相机文件夹到 topic 的映射关系
    
    Returns:
        Dict[str, str]: 文件夹名 -> topic 映射
    """
    return {
        'cam_around_back': '/cam_around_back/compressed',
        'cam_around_front': '/cam_around_front/compressed',
        'cam_around_left': '/cam_around_left/compressed',
        'cam_around_right': '/cam_around_right/compressed',
        'cam_back': '/cam_back/compressed',
        'cam_front_left': '/cam_front_left/compressed',
        'cam_front_right': '/cam_front_right/compressed',
        'cam_side_left_back': '/cam_side_left_back/compressed',
        'cam_side_left_front': '/cam_side_left_front/compressed',
        'cam_side_right_back': '/cam_side_right_back/compressed',
        'cam_side_right_front': '/cam_side_right_front/compressed',
    }


def check_and_fix_camera_sizes(bag_dir: str, dry_run: bool = False) -> Dict:
    """
    检查并修正相机尺寸配置
    
    Args:
        bag_dir: bag 文件所在目录
        dry_run: 是否只检查不修改（默认 False）
    
    Returns:
        Dict: 检查结果
    """
    result = {
        'bag_dir': bag_dir,
        'yaml_file': None,
        'cameras': [],
        'mismatched': [],
        'fixed': [],
        'success': False
    }
    
    bag_path = Path(bag_dir)
    
    # 查找 car_*.yaml 文件
    yaml_files = list(bag_path.glob('car_*.yaml'))
    
    if not yaml_files:
        result['error'] = f"未找到 car_*.yaml 配置文件"
        return result
    
    yaml_file = yaml_files[0]
    result['yaml_file'] = str(yaml_file)
    
    # 读取 yaml 文件
    try:
        with open(yaml_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        result['error'] = f"读取配置文件失败: {e}"
        return result
    
    # 获取相机文件夹到 topic 的映射
    folder_to_topic = get_camera_folder_to_topic_mapping()
    
    # 检查每个相机
    cameras = config.get('sensors', {}).get('camera', [])
    
    for camera_config in cameras:
        topic = camera_config.get('topic', '')
        if not topic:
            continue
        
        # 找到对应的相机文件夹
        camera_folder = None
        for folder, cam_topic in folder_to_topic.items():
            if cam_topic == topic:
                camera_folder = folder
                break
        
        if not camera_folder:
            continue
        
        camera_dir = bag_path / camera_folder
        
        try:
            # 获取实际尺寸
            actual_width, actual_height = get_camera_actual_size(str(camera_dir))
            
            # 获取配置中的尺寸
            configured_size = camera_config.get('calibration', {}).get('ImageSize', [])
            configured_width, configured_height = configured_size if len(configured_size) >= 2 else (0, 0)
            
            camera_info = {
                'topic': topic,
                'folder': camera_folder,
                'actual_size': [actual_width, actual_height],
                'configured_size': [configured_width, configured_height],
                'match': (actual_width == configured_width and actual_height == configured_height)
            }
            
            result['cameras'].append(camera_info)
            
            # 如果不匹配
            if not camera_info['match']:
                result['mismatched'].append(camera_info)
                
                if not dry_run:
                    # 修改配置
                    camera_config['calibration']['ImageSize'] = [actual_width, actual_height]
                    result['fixed'].append(camera_info)
                    
        except Exception as e:
            camera_info = {
                'topic': topic,
                'folder': camera_folder,
                'error': str(e)
            }
            result['cameras'].append(camera_info)
    
    # 如果有修改，保存 yaml 文件
    if result['fixed'] and not dry_run:
        try:
            with open(yaml_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            result['success'] = True
        except Exception as e:
            result['error'] = f"保存配置文件失败: {e}"
    else:
        result['success'] = True
    
    return result


def print_check_result(result: Dict):
    """
    打印检查结果
    
    Args:
        result: check_and_fix_camera_sizes 的返回结果
    """
    print(f"\n{'='*80}")
    print(f"相机尺寸检查结果: {Path(result['bag_dir']).name}")
    print(f"{'='*80}")
    
    if 'error' in result:
        print(f"\n错误: {result['error']}")
        return
    
    print(f"\n配置文件: {result['yaml_file']}")
    print(f"\n检查了 {len(result['cameras'])} 个相机")
    
    if result['mismatched']:
        print(f"\n发现 {len(result['mismatched'])} 个尺寸不匹配的相机:")
        print(f"{'-'*80}")
        
        for cam in result['mismatched']:
            print(f"\n相机: {cam['folder']}")
            print(f"  Topic: {cam['topic']}")
            print(f"  实际尺寸: {cam['actual_size'][0]}x{cam['actual_size'][1]}")
            print(f"  配置尺寸: {cam['configured_size'][0]}x{cam['configured_size'][1]}")
            
            if cam in result['fixed']:
                print(f"  状态: ✓ 已修正")
            else:
                print(f"  状态: ✗ 未修正 (dry_run)")
    else:
        print(f"\n✓ 所有相机尺寸配置正确！")
    
    print(f"\n{'='*80}")


def check_all_bags(base_dir: str, dry_run: bool = False):
    """
    检查所有数据包的相机尺寸配置
    
    Args:
        base_dir: 包含所有数据包的目录
        dry_run: 是否只检查不修改
    """
    base_path = Path(base_dir)
    
    # 查找所有包含 car_*.yaml 的目录
    all_yaml_files = list(base_path.rglob('car_*.yaml'))
    
    if not all_yaml_files:
        print(f"未找到任何 car_*.yaml 配置文件")
        return
    
    print(f"找到 {len(all_yaml_files)} 个数据包配置文件")
    
    total_mismatched = 0
    total_fixed = 0
    
    for yaml_file in sorted(all_yaml_files):
        bag_dir = yaml_file.parent
        result = check_and_fix_camera_sizes(str(bag_dir), dry_run)
        
        if result.get('mismatched'):
            total_mismatched += len(result['mismatched'])
        
        if result.get('fixed'):
            total_fixed += len(result['fixed'])
        
        print_check_result(result)
    
    print(f"\n{'='*80}")
    print(f"汇总: 共检查 {len(all_yaml_files)} 个数据包")
    print(f"  发现尺寸不匹配: {total_mismatched} 个相机")
    print(f"  已修正: {total_fixed} 个相机")
    print(f"{'='*80}")


if __name__ == '__main__':
    import sys
    
    # 测试单个数据包
    bag_dir = '/home/geely/Downloads/S39_0117/2026-01-17-16-37-58'
    
    print("相机尺寸检查工具")
    print("=" * 80)
    
    # 先只检查不修改
    print("\n1. 检查模式（不修改配置）")
    print("-" * 80)
    result = check_and_fix_camera_sizes(bag_dir, dry_run=True)
    print_check_result(result)
    
    # 如果有不匹配的，询问是否要修正
    if result.get('mismatched'):
        print(f"\n2. 发现 {len(result['mismatched'])} 个尺寸不匹配的相机")
        print("-" * 80)
        
        # 自动修正
        print("\n正在修正配置...")
        result = check_and_fix_camera_sizes(bag_dir, dry_run=False)
        print_check_result(result)
    else:
        print("\n所有相机尺寸配置正确，无需修正")
