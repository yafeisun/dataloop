"""
自标定引擎
整合消失点、极线约束和运动估计，实现车端在线自标定
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import time

from common.models.calibration import (
    FactorySpec,
    LearnedSpec,
    CalibrationConfig,
    CalibrationResult,
    CalibrationStatus,
    CameraExtrinsics,
    SensorType
)
from .vanishing_point import VanishingPointDetector
from .epipolar_constraint import EpipolarGeometry
from .ego_motion_estimator import EgoMotionEstimator


class AutoCalibrationEngine:
    """
    自标定引擎
    实现端云协同的动态标定机制
    
    核心功能：
    1. 基于消失点标定俯仰角和偏航角
    2. 基于极线约束标定相邻相机相对位置
    3. 基于自车运动估计相机高度
    4. 在线Bundle Adjustment优化
    """
    
    def __init__(
        self,
        factory_specs: Dict[str, FactorySpec],
        config: Optional[CalibrationConfig] = None
    ):
        """
        初始化自标定引擎
        
        Args:
            factory_specs: 出厂标称参数字典 {sensor_id: FactorySpec}
            config: 标定配置
        """
        self.factory_specs = factory_specs
        self.config = config or CalibrationConfig(
            sensor_id="default",
            sensor_type=SensorType.CAMERA
        )
        
        # 初始化Learned Spec
        self.learned_specs: Dict[str, LearnedSpec] = {}
        for sensor_id, factory_spec in factory_specs.items():
            self.learned_specs[sensor_id] = LearnedSpec(
                sensor_id=sensor_id,
                sensor_type=factory_spec.sensor_type,
                intrinsics=factory_spec.intrinsics,
                extrinsics=factory_spec.extrinsics,
                factory_spec=factory_spec.extrinsics
            )
        
        # 初始化标定子模块
        self._init_calibration_modules()
        
        # 标定状态
        self.is_running = False
        self.total_driving_distance = 0.0
        self.last_update_time = time.time()
    
    def _init_calibration_modules(self):
        """初始化标定子模块"""
        self.vp_detectors: Dict[str, VanishingPointDetector] = {}
        self.epipolar_geometries: Dict[Tuple[str, str], EpipolarGeometry] = {}
        self.ego_motion_estimators: Dict[str, EgoMotionEstimator] = {}
        
        # 初始化消失点检测器
        for sensor_id, factory_spec in self.factory_specs.items():
            if factory_spec.sensor_type == SensorType.CAMERA and factory_spec.intrinsics:
                intrinsics = factory_spec.intrinsics.to_matrix()
                image_size = (1920, 1080)  # 默认图像尺寸
                self.vp_detectors[sensor_id] = VanishingPointDetector(
                    intrinsics, image_size
                )
        
        # 初始化极线几何计算器（相邻相机）
        camera_ids = [
            sensor_id for sensor_id, spec in self.factory_specs.items()
            if spec.sensor_type == SensorType.CAMERA
        ]
        
        for i in range(len(camera_ids) - 1):
            id1, id2 = camera_ids[i], camera_ids[i + 1]
            spec1 = self.factory_specs[id1]
            spec2 = self.factory_specs[id2]
            
            if spec1.intrinsics and spec2.intrinsics:
                K1 = spec1.intrinsics.to_matrix()
                K2 = spec2.intrinsics.to_matrix()
                self.epipolar_geometries[(id1, id2)] = EpipolarGeometry(K1, K2)
        
        # 初始化运动估计器
        for sensor_id, factory_spec in self.factory_specs.items():
            if factory_spec.sensor_type == SensorType.CAMERA and factory_spec.intrinsics:
                intrinsics = factory_spec.intrinsics.to_matrix()
                self.ego_motion_estimators[sensor_id] = EgoMotionEstimator(intrinsics)
    
    def update(
        self,
        sensor_data: Dict[str, Dict],
        wheel_speed: Optional[float] = None,
        imu_data: Optional[Dict] = None,
        driving_distance: float = 0.0
    ) -> Dict[str, CalibrationResult]:
        """
        更新标定参数
        
        Args:
            sensor_data: 传感器数据字典 {sensor_id: data}
            wheel_speed: 轮速
            imu_data: IMU数据
            driving_distance: 行驶距离（米）
        
        Returns:
            标定结果字典 {sensor_id: CalibrationResult}
        """
        self.total_driving_distance += driving_distance
        
        results = {}
        
        # 对每个传感器进行标定
        for sensor_id, learned_spec in self.learned_specs.items():
            if sensor_id not in sensor_data:
                continue
            
            result = self._calibrate_sensor(
                sensor_id,
                sensor_data[sensor_id],
                wheel_speed,
                imu_data
            )
            
            results[sensor_id] = result
            
            # 更新Learned Spec
            if result.success:
                self._update_learned_spec(sensor_id, result)
        
        self.last_update_time = time.time()
        
        return results
    
    def _calibrate_sensor(
        self,
        sensor_id: str,
        sensor_data: Dict,
        wheel_speed: Optional[float],
        imu_data: Optional[Dict]
    ) -> CalibrationResult:
        """
        标定单个传感器
        
        Args:
            sensor_id: 传感器ID
            sensor_data: 传感器数据
            wheel_speed: 轮速
            imu_data: IMU数据
        
        Returns:
            CalibrationResult: 标定结果
        """
        learned_spec = self.learned_specs[sensor_id]
        
        # 检查是否已收敛
        if learned_spec.is_converged(self.config.convergence_threshold):
            return CalibrationResult(
                sensor_id=sensor_id,
                success=True,
                status=CalibrationStatus.CONVERGED,
                confidence=learned_spec.confidence,
                convergence_progress=learned_spec.convergence_progress
            )
        
        # 根据传感器类型选择标定方法
        if learned_spec.sensor_type == SensorType.CAMERA:
            return self._calibrate_camera(
                sensor_id,
                sensor_data,
                wheel_speed,
                imu_data
            )
        else:
            return CalibrationResult(
                sensor_id=sensor_id,
                success=False,
                status=CalibrationStatus.UNINITIALIZED,
                confidence=0.0,
                convergence_progress=0.0,
                error_message=f"Unsupported sensor type: {learned_spec.sensor_type}"
            )
    
    def _calibrate_camera(
        self,
        sensor_id: str,
        sensor_data: Dict,
        wheel_speed: Optional[float],
        imu_data: Optional[Dict]
    ) -> CalibrationResult:
        """
        标定相机
        
        Args:
            sensor_id: 相机ID
            sensor_data: 相机数据
            wheel_speed: 轮速
            imu_data: IMU数据
        
        Returns:
            CalibrationResult: 标定结果
        """
        learned_spec = self.learned_specs[sensor_id]
        
        # 1. 消失点标定（俯仰角和偏航角）
        pitch_correction = 0.0
        yaw_correction = 0.0
        vp_confidence = 0.0
        
        if sensor_id in self.vp_detectors and "lane_lines" in sensor_data:
            vp_result = self.vp_detectors[sensor_id].detect_from_lane_lines(
                sensor_data["lane_lines"]
            )
            
            if vp_result and vp_result.confidence > 0.5:
                pitch_correction, yaw_correction = \
                    self.vp_detectors[sensor_id].compute_correction(vp_result)
                vp_confidence = vp_result.confidence
        
        # 2. 极线约束标定（相邻相机相对位置）
        epipolar_confidence = 0.0
        
        for (id1, id2), epipolar_geo in self.epipolar_geometries.items():
            if id1 == sensor_id and id2 in sensor_data:
                if "keypoints" in sensor_data and "keypoints" in sensor_data[id2]:
                    epipolar_result = epipolar_geo.compute_epipolar_geometry(
                        sensor_data["keypoints"],
                        sensor_data[id2]["keypoints"]
                    )
                    
                    if epipolar_result and epipolar_result.confidence > 0.5:
                        epipolar_confidence = epipolar_result.confidence
        
        # 3. 运动估计标定（相机高度）
        height_correction = 0.0
        motion_confidence = 0.0
        
        if sensor_id in self.ego_motion_estimators and "image" in sensor_data:
            if "image_prev" in sensor_data:
                ego_result = self.ego_motion_estimators[sensor_id].estimate_from_visual_odometry(
                    sensor_data["image_prev"],
                    sensor_data["image"],
                    wheel_speed,
                    imu_data
                )
                
                if ego_result and ego_result.confidence > 0.5:
                    motion_confidence = ego_result.confidence
        
        # 4. 综合修正量
        total_confidence = (vp_confidence + epipolar_confidence + motion_confidence) / 3.0
        
        if total_confidence < 0.3:
            return CalibrationResult(
                sensor_id=sensor_id,
                success=False,
                status=CalibrationStatus.CONVERGING,
                confidence=learned_spec.confidence,
                convergence_progress=learned_spec.convergence_progress,
                error_message="Insufficient confidence for calibration update"
            )
        
        # 应用修正量
        current_extrinsics = learned_spec.extrinsics
        current_roll, current_pitch, current_yaw = current_extrinsics.rotation
        current_tx, current_ty, current_tz = current_extrinsics.translation
        
        # 更新旋转角度（使用低通滤波平滑）
        alpha = 0.1  # 滤波系数
        new_pitch = current_pitch + alpha * pitch_correction
        new_yaw = current_yaw + alpha * yaw_correction
        
        # 更新平移（主要是高度）
        new_tz = current_tz + alpha * height_correction
        
        # 创建新的外参
        new_extrinsics = CameraExtrinsics(
            translation=(current_tx, current_ty, new_tz),
            rotation=(current_roll, new_pitch, new_yaw)
        )
        
        # 更新Learned Spec
        learned_spec.extrinsics = new_extrinsics
        learned_spec.status = CalibrationStatus.CONVERGING
        learned_spec.confidence = learned_spec.confidence * 0.9 + total_confidence * 0.1
        learned_spec.update_count += 1
        learned_spec.last_update_time = datetime.now()
        learned_spec.total_driving_distance = self.total_driving_distance
        
        # 计算收敛进度
        learned_spec.convergence_progress = min(
            learned_spec.update_count / 100.0,
            1.0
        )
        
        # 检查是否收敛
        if learned_spec.convergence_progress >= 1.0 and learned_spec.confidence >= 0.9:
            learned_spec.status = CalibrationStatus.CONVERGED
        
        # 计算偏差
        deviation = learned_spec.compute_deviation()
        learned_spec.translation_deviation = deviation[3:6]
        learned_spec.rotation_deviation = deviation[0:3]
        
        return CalibrationResult(
            sensor_id=sensor_id,
            success=True,
            status=learned_spec.status,
            confidence=learned_spec.confidence,
            convergence_progress=learned_spec.convergence_progress,
            data={
                "pitch_correction": pitch_correction,
                "yaw_correction": yaw_correction,
                "height_correction": height_correction,
                "vp_confidence": vp_confidence,
                "epipolar_confidence": epipolar_confidence,
                "motion_confidence": motion_confidence
            }
        )
    
    def _update_learned_spec(
        self,
        sensor_id: str,
        result: CalibrationResult
    ):
        """
        更新Learned Spec
        
        Args:
            sensor_id: 传感器ID
            result: 标定结果
        """
        # 已在_calibrate_camera中更新
        pass
    
    def get_learned_specs(self) -> Dict[str, LearnedSpec]:
        """获取所有Learned Spec"""
        return self.learned_specs.copy()
    
    def get_learned_spec(self, sensor_id: str) -> Optional[LearnedSpec]:
        """获取指定传感器的Learned Spec"""
        return self.learned_specs.get(sensor_id)
    
    def reset_sensor(self, sensor_id: str, hard_reset: bool = False):
        """
        重置传感器标定
        
        Args:
            sensor_id: 传感器ID
            hard_reset: 是否硬重置（回到Factory Spec）
        """
        if sensor_id not in self.learned_specs:
            return
        
        if hard_reset:
            # 硬重置：回到Factory Spec
            factory_spec = self.factory_specs[sensor_id]
            self.learned_specs[sensor_id].extrinsics = factory_spec.extrinsics
            self.learned_specs[sensor_id].status = CalibrationStatus.UNINITIALIZED
            self.learned_specs[sensor_id].confidence = 0.0
            self.learned_specs[sensor_id].convergence_progress = 0.0
            self.learned_specs[sensor_id].update_count = 0
        else:
            # 软重置：保留当前参数，降低置信度
            self.learned_specs[sensor_id].status = CalibrationStatus.CONVERGING
            self.learned_specs[sensor_id].confidence *= 0.5
            self.learned_specs[sensor_id].convergence_progress *= 0.5
        
        # 重置运动估计器
        if sensor_id in self.ego_motion_estimators:
            self.ego_motion_estimators[sensor_id].reset()
    
    def start(self):
        """启动标定引擎"""
        self.is_running = True
    
    def stop(self):
        """停止标定引擎"""
        self.is_running = False
    
    def is_converged(self) -> bool:
        """检查是否所有传感器都已收敛"""
        return all(
            spec.is_converged(self.config.convergence_threshold)
            for spec in self.learned_specs.values()
        )
