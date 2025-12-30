"""
标定系统使用示例
演示如何使用端云协同的动态标定机制
"""

import numpy as np
from datetime import datetime

from common.models.calibration import (
    FactorySpec,
    LearnedSpec,
    CalibrationConfig,
    CalibrationStatus,
    CameraIntrinsics,
    CameraExtrinsics,
    SensorType,
    VehicleCalibrationState
)
from common.utils.transform_tree import TransformTree, Transform, FrameType
from onboard.calibration import (
    CalibrationManager,
    AutoCalibrationEngine,
    VanishingPointDetector,
    VirtualCameraManager,
    VirtualCameraConfig,
    AnomalyHandler,
    AnomalyType,
    ResetType
)
from cloud.calibration import (
    CalibrationMonitor,
    CalibrationDiagnosis,
    BatchAnalysis
)


def create_factory_specs():
    """创建出厂标称参数"""
    factory_specs = {}
    
    # 前视相机
    factory_specs["front_camera"] = FactorySpec(
        sensor_id="front_camera",
        sensor_type=SensorType.CAMERA,
        intrinsics=CameraIntrinsics(
            fx=1000.0,
            fy=1000.0,
            cx=960.0,
            cy=540.0
        ),
        extrinsics=CameraExtrinsics(
            translation=(2.0, 0.0, 1.5),  # 车前方2米，高度1.5米
            rotation=(0.0, 0.0, 0.0)     # 无旋转
        )
    )
    
    # 左侧相机
    factory_specs["left_camera"] = FactorySpec(
        sensor_id="left_camera",
        sensor_type=SensorType.CAMERA,
        intrinsics=CameraIntrinsics(
            fx=1000.0,
            fy=1000.0,
            cx=960.0,
            cy=540.0
        ),
        extrinsics=CameraExtrinsics(
            translation=(0.0, -1.0, 1.5),  # 车左侧1米，高度1.5米
            rotation=(0.0, 0.0, np.deg2rad(90))  # 向左转90度
        )
    )
    
    # 右侧相机
    factory_specs["right_camera"] = FactorySpec(
        sensor_id="right_camera",
        sensor_type=SensorType.CAMERA,
        intrinsics=CameraIntrinsics(
            fx=1000.0,
            fy=1000.0,
            cx=960.0,
            cy=540.0
        ),
        extrinsics=CameraExtrinsics(
            translation=(0.0, 1.0, 1.5),   # 车右侧1米，高度1.5米
            rotation=(0.0, 0.0, np.deg2rad(-90))  # 向右转90度
        )
    )
    
    return factory_specs


def example_onboard_calibration():
    """车端标定示例"""
    print("=" * 60)
    print("车端标定示例")
    print("=" * 60)
    
    # 1. 创建出厂标称参数
    factory_specs = create_factory_specs()
    
    # 2. 创建标定配置
    config = CalibrationConfig(
        sensor_id="default",
        sensor_type=SensorType.CAMERA,
        convergence_threshold=0.9,
        min_driving_distance=100.0
    )
    
    # 3. 创建标定管理器
    calibration_manager = CalibrationManager(
        vehicle_id="test_vehicle_001",
        factory_specs=factory_specs,
        config=config
    )
    
    # 4. 设置上传回调
    def upload_callback(metadata):
        print(f"[UploadCallback] Uploading calibration data...")
        print(f"  Vehicle ID: {metadata['vehicle_id']}")
        print(f"  Overall Status: {metadata['overall_status']}")
        print(f"  Overall Convergence: {metadata['overall_convergence']:.2%}")
        print(f"  Health Score: {metadata['health_score']:.2%}")
        return True
    
    calibration_manager.set_upload_callback(upload_callback)
    
    # 5. 启动标定
    calibration_manager.boot_up()
    calibration_manager.start_calibration()
    
    # 6. 模拟传感器数据更新
    print("\n模拟传感器数据更新...")
    
    for i in range(10):
        # 模拟传感器数据
        sensor_data = {
            "front_camera": {
                "lane_lines": [
                    (100, 540, 960, 540),
                    (960, 540, 1820, 540)
                ],
                "image": np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8),
                "keypoints": np.random.rand(100, 2) * 1920
            },
            "left_camera": {
                "image": np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8),
                "keypoints": np.random.rand(100, 2) * 1920
            },
            "right_camera": {
                "image": np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8),
                "keypoints": np.random.rand(100, 2) * 1920
            }
        }
        
        # 更新标定
        results = calibration_manager.update(
            sensor_data=sensor_data,
            wheel_speed=10.0,  # 10 m/s
            imu_data={
                "acceleration": np.array([0.1, 0.0, 9.8]),
                "angular_velocity": np.array([0.0, 0.0, 0.0])
            },
            driving_distance=100.0  # 100米
        )
        
        print(f"\n更新 {i+1}/10:")
        for sensor_id, result in results.items():
            if result.success:
                print(f"  {sensor_id}: 状态={result.status.value}, 置信度={result.confidence:.2%}, 进度={result.convergence_progress:.2%}")
    
    # 7. 获取标定进度
    print("\n标定进度:")
    progress_list = calibration_manager.get_calibration_progress()
    for progress in progress_list:
        print(f"  {progress.sensor_id}: 进度={progress.progress:.2%}, 状态={progress.status.value}")
    
    # 8. 获取整车标定状态
    state = calibration_manager.get_calibration_state()
    print(f"\n整车标定状态:")
    print(f"  总体状态: {state.overall_status.value}")
    print(f"  总体收敛度: {state.overall_convergence:.2%}")
    print(f"  健康度: {state.health_score:.2%}")
    print(f"  累计行驶距离: {state.total_driving_distance:.1f}米")
    
    # 9. 停止标定
    calibration_manager.stop_calibration()
    
    print("\n车端标定示例完成")


def example_cloud_monitoring():
    """云端监控示例"""
    print("\n" + "=" * 60)
    print("云端监控示例")
    print("=" * 60)
    
    # 1. 创建监控器
    monitor = CalibrationMonitor()
    
    # 2. 注册车辆批次
    monitor.register_vehicle_batch("test_vehicle_001", "batch_2024_001")
    monitor.register_vehicle_batch("test_vehicle_002", "batch_2024_001")
    monitor.register_vehicle_batch("test_vehicle_003", "batch_2024_002")
    
    # 3. 接收标定数据
    for vehicle_id in ["test_vehicle_001", "test_vehicle_002", "test_vehicle_003"]:
        metadata = {
            "vehicle_id": vehicle_id,
            "overall_status": "converged",
            "overall_convergence": 0.95,
            "health_score": 0.92,
            "anomaly_detected": False,
            "last_calibration_time": datetime.now().isoformat(),
            "total_driving_distance": 1000.0,
            "sensor_specs": {
                "front_camera": {
                    "status": "converged",
                    "confidence": 0.95,
                    "convergence_progress": 1.0,
                    "deviation": [0.01, 0.02, 0.01, 0.005, 0.003, 0.002]
                }
            }
        }
        monitor.receive_calibration_data(metadata)
    
    # 4. 计算统计信息
    statistics = monitor.compute_statistics()
    print(f"\n全队标定统计:")
    print(f"  车辆总数: {statistics.vehicle_count}")
    print(f"  已收敛: {statistics.converged_count}")
    print(f"  收敛中: {statistics.converging_count}")
    print(f"  失败: {statistics.failed_count}")
    print(f"  平均收敛度: {statistics.avg_convergence:.2%}")
    print(f"  平均健康度: {statistics.avg_health_score:.2%}")
    print(f"  异常数量: {statistics.anomaly_count}")
    
    # 5. 检测批次异常
    anomalies = monitor.detect_batch_anomalies()
    if anomalies:
        print(f"\n检测到 {len(anomalies)} 个批次异常:")
        for anomaly in anomalies:
            print(f"  批次: {anomaly.batch_id}, 传感器: {anomaly.sensor_id}")
            print(f"  类型: {anomaly.anomaly_type}, 严重程度: {anomaly.severity}")
    else:
        print("\n未检测到批次异常")
    
    # 6. 获取批次统计
    batch_stats = monitor.get_batch_statistics("batch_2024_001")
    if batch_stats:
        print(f"\n批次 batch_2024_001 统计:")
        print(f"  车辆数量: {batch_stats['vehicle_count']}")
        print(f"  平均收敛度: {batch_stats['avg_convergence']:.2%}")
        print(f"  平均健康度: {batch_stats['avg_health_score']:.2%}")
        print(f"  异常率: {batch_stats['anomaly_rate']:.2%}")
    
    print("\n云端监控示例完成")


def example_anomaly_handling():
    """异常处理示例"""
    print("\n" + "=" * 60)
    print("异常处理示例")
    print("=" * 60)
    
    # 1. 创建异常处理器
    anomaly_handler = AnomalyHandler(
        soft_reset_threshold=0.05,
        hard_reset_threshold=0.1
    )
    
    # 2. 设置重置回调
    def reset_callback(sensor_id, reset_type, details):
        print(f"[ResetCallback] 触发重置: sensor_id={sensor_id}, type={reset_type.value}")
        print(f"  原因: {details}")
        return True
    
    anomaly_handler.set_reset_callback(reset_callback)
    
    # 3. 创建Learned Spec（模拟偏差过大的情况）
    factory_specs = create_factory_specs()
    learned_specs = {}
    
    for sensor_id, factory_spec in factory_specs.items():
        learned_specs[sensor_id] = LearnedSpec(
            sensor_id=sensor_id,
            sensor_type=factory_spec.sensor_type,
            intrinsics=factory_spec.intrinsics,
            extrinsics=factory_spec.extrinsics,
            factory_spec=factory_spec.extrinsics
        )
        
        # 模拟偏差过大
        learned_specs[sensor_id].extrinsics = CameraExtrinsics(
            translation=(
                factory_spec.extrinsics.translation[0] + 0.15,  # 偏差15cm
                factory_spec.extrinsics.translation[1],
                factory_spec.extrinsics.translation[2]
            ),
            rotation=(
                factory_spec.extrinsics.rotation[0],
                factory_spec.extrinsics.rotation[1] + np.deg2rad(8),  # 偏差8度
                factory_spec.extrinsics.rotation[2]
            )
        )
    
    # 4. 检测异常
    print("\n检测异常...")
    anomalies = anomaly_handler.detect_anomalies(learned_specs)
    
    if anomalies:
        print(f"检测到 {len(anomalies)} 个异常:")
        for anomaly in anomalies:
            print(f"  传感器: {anomaly.sensor_id}")
            print(f"  类型: {anomaly.anomaly_type.value}")
            print(f"  严重程度: {anomaly.severity}")
            print(f"  详情: {anomaly.details}")
    else:
        print("未检测到异常")
    
    # 5. 处理异常
    if anomalies:
        print("\n处理异常...")
        reset_events = anomaly_handler.handle_anomalies(anomalies)
        
        print(f"触发了 {len(reset_events)} 个重置:")
        for reset in reset_events:
            print(f"  传感器: {reset.sensor_id}")
            print(f"  类型: {reset.reset_type.value}")
            print(f"  原因: {reset.reason}")
            print(f"  成功: {reset.success}")
    
    # 6. 获取统计信息
    stats = anomaly_handler.get_statistics()
    print(f"\n异常处理统计:")
    print(f"  总异常数: {stats['total_anomalies']}")
    print(f"  总重置数: {stats['total_resets']}")
    
    print("\n异常处理示例完成")


if __name__ == "__main__":
    # 运行所有示例
    example_onboard_calibration()
    example_cloud_monitoring()
    example_anomaly_handling()
    
    print("\n" + "=" * 60)
    print("所有示例运行完成")
    print("=" * 60)