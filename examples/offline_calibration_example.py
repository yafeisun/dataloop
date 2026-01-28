"""
离线数据标定示例
演示如何使用三层漏斗筛选机制挖掘"黄金标定片段"
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any

from cloud.calibration.offline_calibration import (
    OfflineCalibrationMiner,
    VehicleDynamicsFilter,
    VisualPerceptionFilter,
    ObservabilityFilter,
    CalibrationClipType
)


def generate_mock_vehicle_data(duration: float = 20.0, fps: int = 10) -> Dict[str, np.ndarray]:
    """
    生成模拟车辆动力学数据
    
    Args:
        duration: 持续时间（秒）
        fps: 帧率
    
    Returns:
        车辆数据字典
    """
    num_frames = int(duration * fps)
    
    # 生成直道行驶数据（高质量）
    speed = np.random.normal(50.0, 2.0, num_frames)  # 50 km/h
    steering_angle = np.random.normal(0.0, 0.2, num_frames)  # 接近0度
    longitudinal_acc = np.random.normal(0.0, 0.1, num_frames)  # 接近0 m/s²
    yaw_rate = np.random.normal(0.0, 0.3, num_frames)  # 接近0 deg/s
    imu_acc_z = np.random.normal(9.8, 0.05, num_frames)  # 接近重力加速度
    
    return {
        "speed": speed,
        "steering_angle": steering_angle,
        "longitudinal_acc": longitudinal_acc,
        "yaw_rate": yaw_rate,
        "imu_acc_z": imu_acc_z
    }


def generate_mock_visual_data(num_frames: int = 200) -> Dict[str, Any]:
    """
    生成模拟视觉/感知数据
    
    Args:
        num_frames: 帧数
    
    Returns:
        视觉数据字典
    """
    import cv2
    
    # 生成模拟图像（创建一个带有丰富纹理的场景）
    images = []
    for _ in range(num_frames):
        # 创建一个灰度图像
        img = np.zeros((480, 640), dtype=np.uint8)
        
        # 添加一些纹理（模拟建筑物、树木、路灯等）
        # 添加垂直线条（模拟建筑物边缘）
        for x in range(50, 600, 60):
            cv2.line(img, (x, 100), (x, 400), np.random.randint(100, 200), 2)
        
        # 添加水平线条（模拟车道线）
        cv2.line(img, (100, 400), (200, 200), 255, 3)
        cv2.line(img, (500, 400), (600, 200), 255, 3)
        
        # 添加一些随机纹理（模拟树木、路灯等）
        for _ in range(50):
            x = np.random.randint(0, 640)
            y = np.random.randint(100, 400)
            radius = np.random.randint(3, 10)
            cv2.circle(img, (x, y), radius, np.random.randint(150, 250), -1)
        
        # 增加整体亮度，避免过暗
        img = cv2.add(img, np.ones(img.shape, dtype=np.uint8) * 50)
        
        # 转换为BGR图像
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        images.append(img_bgr)
    
    # 生成模拟感知结果
    perception_results = []
    for _ in range(num_frames):
        # 大部分是静态物体，少量动态物体（1/10）
        objects = [
            {"id": 1, "type": "car", "is_dynamic": True},
            {"id": 2, "type": "building", "is_dynamic": False},
            {"id": 3, "type": "building", "is_dynamic": False},
            {"id": 4, "type": "building", "is_dynamic": False},
            {"id": 5, "type": "building", "is_dynamic": False},
            {"id": 6, "type": "tree", "is_dynamic": False},
            {"id": 7, "type": "tree", "is_dynamic": False},
            {"id": 8, "type": "tree", "is_dynamic": False},
            {"id": 9, "type": "tree", "is_dynamic": False},
            {"id": 10, "type": "road_sign", "is_dynamic": False},
            {"id": 11, "type": "road_sign", "is_dynamic": False},
            {"id": 12, "type": "road_sign", "is_dynamic": False},
            {"id": 13, "type": "road_sign", "is_dynamic": False},
            {"id": 14, "type": "lane_line", "is_dynamic": False},
            {"id": 15, "type": "lane_line", "is_dynamic": False},
            {"id": 16, "type": "lane_line", "is_dynamic": False},
            {"id": 17, "type": "lane_line", "is_dynamic": False},
            {"id": 18, "type": "lane_line", "is_dynamic": False},
            {"id": 19, "type": "lane_line", "is_dynamic": False},
            {"id": 20, "type": "lane_line", "is_dynamic": False},
        ]
        perception_results.append({"objects": objects})
    
    # 生成模拟车道线检测结果
    lane_detection = []
    for _ in range(num_frames):
        lane_detection.append({
            "confidence": np.random.uniform(0.95, 0.99),
            "left_lane": {"points": [(100, 400), (150, 300), (200, 200)]},
            "right_lane": {"points": [(500, 400), (550, 300), (600, 200)]}
        })
    
    return {
        "images": images,
        "perception_results": perception_results,
        "lane_detection": lane_detection
    }


def generate_mock_observability_data(num_frames: int = 200, has_loop: bool = False) -> Dict[str, Any]:
    """
    生成模拟可观测性数据
    
    Args:
        num_frames: 帧数
        has_loop: 是否有闭环
    
    Returns:
        可观测性数据字典
    """
    # 生成稳定的消失点
    vp_center = (320, 240)
    vanishing_points = [
        (
            vp_center[0] + np.random.normal(0, 2),
            vp_center[1] + np.random.normal(0, 2)
        )
        for _ in range(num_frames)
    ]
    
    # 生成Fisher Information Matrix（模拟）
    fim = np.eye(6) * 1e-2  # 6自由度
    
    return {
        "vanishing_points": vanishing_points,
        "fim": fim,
        "has_loop_closure": has_loop
    }


def create_mock_data_segment(
    vehicle_id: str,
    duration: float = 20.0,
    fps: int = 10,
    has_loop: bool = False
) -> Dict[str, Any]:
    """
    创建模拟数据段
    
    Args:
        vehicle_id: 车辆ID
        duration: 持续时间（秒）
        fps: 帧率
        has_loop: 是否有闭环
    
    Returns:
        数据段字典
    """
    num_frames = int(duration * fps)
    start_timestamp = datetime.now()
    end_timestamp = start_timestamp + timedelta(seconds=duration)
    
    return {
        "start_timestamp": start_timestamp,
        "end_timestamp": end_timestamp,
        "vehicle_data": generate_mock_vehicle_data(duration, fps),
        "visual_data": generate_mock_visual_data(num_frames),
        "observability_data": generate_mock_observability_data(num_frames, has_loop),
        "data_paths": {
            "camera_front": f"/data/{vehicle_id}/camera_front_{start_timestamp.strftime('%Y%m%d_%H%M%S')}.bag",
            "imu": f"/data/{vehicle_id}/imu_{start_timestamp.strftime('%Y%m%d_%H%M%S')}.bag",
            "can": f"/data/{vehicle_id}/can_{start_timestamp.strftime('%Y%m%d_%H%M%S')}.log"
        }
    }


def main():
    """主函数"""
    print("=" * 60)
    print("离线数据标定示例")
    print("=" * 60)
    
    # 1. 创建筛选器（使用默认参数）
    print("\n1. 创建筛选器...")
    dynamics_filter = VehicleDynamicsFilter()
    visual_filter = VisualPerceptionFilter()
    observability_filter = ObservabilityFilter()
    
    print(f"   - 车辆动力学筛选器: 速度范围 {dynamics_filter.min_speed}-{dynamics_filter.max_speed} km/h")
    print(f"   - 视觉/感知筛选器: 最少特征点 {visual_filter.min_feature_points} 个")
    print(f"   - 可观测性筛选器: 消失点最大方差 {observability_filter.max_vp_variance} 像素²")
    
    # 2. 创建离线标定数据挖掘器
    print("\n2. 创建离线标定数据挖掘器...")
    miner = OfflineCalibrationMiner(
        dynamics_filter=dynamics_filter,
        visual_filter=visual_filter,
        observability_filter=observability_filter
    )
    
    # 3. 生成模拟数据段
    print("\n3. 生成模拟数据段...")
    vehicle_id = "vehicle_001"
    
    # 创建10个数据段（部分高质量，部分低质量）
    data_segments = []
    for i in range(10):
        # 前5个是高质量数据段
        if i < 5:
            segment = create_mock_data_segment(
                vehicle_id=vehicle_id,
                duration=20.0,
                fps=10,
                has_loop=(i == 0)  # 第0段有闭环
            )
        else:
            # 后5个是低质量数据段（故意制造一些问题）
            segment = create_mock_data_segment(
                vehicle_id=vehicle_id,
                duration=20.0,
                fps=10,
                has_loop=False
            )
            # 故意破坏部分数据
            if i == 5:
                # 速度过低
                segment["vehicle_data"]["speed"] = np.random.normal(20.0, 2.0, len(segment["vehicle_data"]["speed"]))
            elif i == 6:
                # 方向盘转角过大
                segment["vehicle_data"]["steering_angle"] = np.random.normal(2.0, 0.5, len(segment["vehicle_data"]["steering_angle"]))
            elif i == 7:
                # 特征点不足（通过减少图像数量模拟）
                segment["visual_data"]["images"] = segment["visual_data"]["images"][:50]
            elif i == 8:
                # 动态物体过多
                for result in segment["visual_data"]["perception_results"]:
                    result["objects"].extend([
                        {"id": 6, "type": "car", "is_dynamic": True},
                        {"id": 7, "type": "car", "is_dynamic": True},
                        {"id": 8, "type": "pedestrian", "is_dynamic": True}
                    ])
            elif i == 9:
                # 消失点不稳定
                segment["observability_data"]["vanishing_points"] = [
                    (np.random.uniform(200, 400), np.random.uniform(150, 300))
                    for _ in range(len(segment["observability_data"]["vanishing_points"]))
                ]
        
        data_segments.append(segment)
        print(f"   - 数据段 {i+1}: {segment['start_timestamp'].strftime('%H:%M:%S')} - {segment['end_timestamp'].strftime('%H:%M:%S')}")
    
    # 4. 挖掘黄金标定片段
    print("\n4. 挖掘黄金标定片段...")
    golden_clips = miner.mine_golden_clips(vehicle_id, data_segments)
    
    print(f"   - 总数据段数: {len(data_segments)}")
    print(f"   - 黄金片段数: {len(golden_clips)}")
    
    # 5. 显示筛选统计
    print("\n5. 筛选统计...")
    stats = miner.get_filter_statistics()
    print(f"   - 总数据段数: {stats['total_segments']}")
    print(f"   - 通过筛选: {stats['passed_segments']}")
    print(f"   - 未通过筛选: {stats['failed_segments']}")
    print(f"   - 通过率: {stats['pass_rate']*100:.1f}%")
    print(f"\n   各层级筛选统计:")
    for layer, layer_stats in stats['layer_statistics'].items():
        print(f"     - {layer}:")
        print(f"       总数: {layer_stats['total']}, 通过: {layer_stats['passed']}, 失败: {layer_stats['failed']}")
    
    # 6. 显示黄金片段详情
    print("\n6. 黄金片段详情...")
    for i, clip in enumerate(golden_clips):
        print(f"\n   片段 {i+1}: {clip.clip_id}")
        print(f"   - 类型: {clip.clip_type.value}")
        print(f"   - 质量分数: {clip.quality_score:.3f}")
        print(f"   - 时长: {clip.duration:.1f} 秒")
        print(f"   - 平均速度: {clip.avg_speed:.1f} km/h")
        print(f"   - 平均方向盘转角: {clip.avg_steering_angle:.2f}°")
        print(f"   - 平均特征点数: {clip.avg_feature_points:.0f}")
        print(f"   - 动态物体占比: {clip.dynamic_object_ratio*100:.1f}%")
        print(f"   - 车道线置信度: {clip.lane_confidence:.2f}")
        print(f"   - 光照质量: {clip.lighting_quality}")
        print(f"   - 模糊分数: {clip.blur_score:.2f}")
        print(f"   - 消失点方差: {clip.vanishing_point_variance:.1f} 像素²")
        print(f"   - FIM最小奇异值: {clip.fim_min_singular_value:.2e}")
        print(f"   - 闭环: {'是' if clip.has_loop_closure else '否'}")
        print(f"   - 场景描述: {clip.scene_description}")
        print(f"   - 数据路径:")
        for sensor, path in clip.data_paths.items():
            print(f"     {sensor}: {path}")
    
    # 7. 按类型获取片段
    print("\n7. 按类型获取片段...")
    for clip_type in CalibrationClipType:
        clips = miner.get_golden_clips_by_type(clip_type, min_quality_score=0.7)
        print(f"   - {clip_type.value}: {len(clips)} 个片段")
    
    # 8. 获取质量最高的片段
    print("\n8. 质量最高的片段...")
    top_clips = miner.get_top_clips(top_n=3)
    for i, clip in enumerate(top_clips):
        print(f"   Top {i+1}: {clip.clip_id} (质量分数: {clip.quality_score:.3f}, 类型: {clip.clip_type.value})")
    
    # 9. 导出黄金片段信息
    print("\n9. 导出黄金片段信息...")
    output_path = "/tmp/golden_clips.json"
    miner.export_golden_clips(output_path, min_quality_score=0.7)
    print(f"   - 已导出到: {output_path}")
    
    print("\n" + "=" * 60)
    print("示例完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()