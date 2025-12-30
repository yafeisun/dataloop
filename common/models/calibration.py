"""
标定参数数据模型
支持端云协同的动态标定机制
"""

from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
from enum import Enum
import numpy as np
from datetime import datetime


class CalibrationStatus(str, Enum):
    """标定状态"""
    UNINITIALIZED = "uninitialized"  # 未初始化
    CONVERGING = "converging"        # 收敛中
    CONVERGED = "converged"          # 已收敛
    FAILED = "failed"                # 失败
    RESET_REQUIRED = "reset_required"  # 需要重置


class SensorType(str, Enum):
    """传感器类型"""
    CAMERA = "camera"
    IMU = "imu"
    LIDAR = "lidar"
    RADAR = "radar"


class CameraIntrinsics(BaseModel):
    """相机内参"""
    fx: float  # 焦距x
    fy: float  # 焦距y
    cx: float  # 主点x
    cy: float  # 主点y
    k1: float = 0.0  # 径向畸变系数k1
    k2: float = 0.0  # 径向畸变系数k2
    k3: float = 0.0  # 径向畸变系数k3
    p1: float = 0.0  # 切向畸变系数p1
    p2: float = 0.0  # 切向畸变系数p2
    
    def to_matrix(self) -> np.ndarray:
        """转换为内参矩阵"""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])
    
    def to_distortion_coeffs(self) -> np.ndarray:
        """转换为畸变系数数组"""
        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3])


class CameraExtrinsics(BaseModel):
    """相机外参（相对于Body坐标系）"""
    translation: Tuple[float, float, float]  # 平移 (x, y, z) in meters
    rotation: Tuple[float, float, float]     # 旋转 (roll, pitch, yaw) in radians
    
    def to_rotation_matrix(self) -> np.ndarray:
        """转换为旋转矩阵（ZYX顺序）"""
        roll, pitch, yaw = self.rotation
        
        # Roll rotation
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        # Pitch rotation
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        # Yaw rotation
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        return Rz @ Ry @ Rx
    
    def to_transform_matrix(self) -> np.ndarray:
        """转换为4x4齐次变换矩阵"""
        R = self.to_rotation_matrix()
        t = np.array(self.translation)
        
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        
        return T


class FactorySpec(BaseModel):
    """出厂标称参数（Factory Spec）
    车辆设计时的理论位置或EOL测量的基准值
    作为算法启动的初始猜测值
    """
    sensor_id: str
    sensor_type: SensorType
    intrinsics: Optional[CameraIntrinsics] = None  # 相机内参
    extrinsics: CameraExtrinsics  # 相机外参（相对于Body）
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LearnedSpec(BaseModel):
    """动态在线参数（Learned Spec）
    车辆在行驶过程中实时计算的真实位置
    实际用于模型推理和坐标变换
    """
    sensor_id: str
    sensor_type: SensorType
    intrinsics: Optional[CameraIntrinsics] = None  # 相机内参（通常不变）
    extrinsics: CameraExtrinsics  # 相机外参（动态更新）
    factory_spec: CameraExtrinsics  # 原始出厂参数（用于对比偏差）
    
    # 标定状态信息
    status: CalibrationStatus = CalibrationStatus.UNINITIALIZED
    confidence: float = 0.0  # 标定置信度 [0, 1]
    convergence_progress: float = 0.0  # 收敛进度 [0, 1]
    
    # 统计信息
    update_count: int = 0  # 更新次数
    last_update_time: datetime = Field(default_factory=datetime.now)
    total_driving_distance: float = 0.0  # 累计行驶距离（米）
    
    # 偏差信息
    translation_deviation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_deviation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def compute_deviation(self) -> Tuple[float, float, float, float, float, float]:
        """计算与Factory Spec的偏差"""
        factory_R = self.factory_spec.to_rotation_matrix()
        learned_R = self.extrinsics.to_rotation_matrix()
        
        # 旋转偏差（欧拉角）
        R_diff = learned_R @ factory_R.T
        yaw_diff = np.arctan2(R_diff[1, 0], R_diff[0, 0])
        pitch_diff = np.arcsin(-R_diff[2, 0])
        roll_diff = np.arctan2(R_diff[2, 1], R_diff[2, 2])
        
        # 平移偏差
        t_diff = (
            self.extrinsics.translation[0] - self.factory_spec.translation[0],
            self.extrinsics.translation[1] - self.factory_spec.translation[1],
            self.extrinsics.translation[2] - self.factory_spec.translation[2]
        )
        
        return (roll_diff, pitch_diff, yaw_diff, t_diff[0], t_diff[1], t_diff[2])
    
    def is_converged(self, threshold: float = 0.01) -> bool:
        """检查是否已收敛"""
        return (
            self.status == CalibrationStatus.CONVERGED and
            self.confidence >= threshold
        )


class CalibrationConfig(BaseModel):
    """标定配置"""
    sensor_id: str
    sensor_type: SensorType
    
    # 收敛阈值
    convergence_threshold: float = 0.01  # 收敛置信度阈值
    max_iterations: int = 1000  # 最大迭代次数
    min_driving_distance: float = 100.0  # 最小行驶距离（米）
    
    # 重置策略
    soft_reset_threshold: float = 0.05  # 软重置阈值（偏差超过此值）
    hard_reset_threshold: float = 0.1   # 硬重置阈值（偏差超过此值）
    
    # 上传配置
    upload_interval: float = 3600.0  # 上传间隔（秒）
    upload_on_convergence: bool = True  # 收敛时是否立即上传
    
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CalibrationResult(BaseModel):
    """标定结果"""
    sensor_id: str
    success: bool
    status: CalibrationStatus
    confidence: float
    convergence_progress: float
    timestamp: datetime = Field(default_factory=datetime.now)
    error_message: str = ""
    data: Dict[str, Any] = Field(default_factory=dict)


class VehicleCalibrationState(BaseModel):
    """整车标定状态"""
    vehicle_id: str
    
    # 所有传感器的标定参数
    factory_specs: Dict[str, FactorySpec] = Field(default_factory=dict)
    learned_specs: Dict[str, LearnedSpec] = Field(default_factory=dict)
    
    # 全局标定状态
    overall_status: CalibrationStatus = CalibrationStatus.UNINITIALIZED
    overall_convergence: float = 0.0  # 整体收敛进度 [0, 1]
    
    # 统计信息
    last_calibration_time: Optional[datetime] = None
    total_driving_distance: float = 0.0
    
    # 健康度指标
    health_score: float = 1.0  # 健康度分数 [0, 1]
    anomaly_detected: bool = False
    
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def get_sensor_spec(self, sensor_id: str) -> Optional[LearnedSpec]:
        """获取指定传感器的Learned Spec"""
        return self.learned_specs.get(sensor_id)
    
    def update_overall_status(self):
        """更新整车标定状态"""
        if not self.learned_specs:
            self.overall_status = CalibrationStatus.UNINITIALIZED
            return
        
        converged_count = sum(
            1 for spec in self.learned_specs.values()
            if spec.is_converged()
        )
        
        self.overall_convergence = converged_count / len(self.learned_specs)
        
        if self.overall_convergence >= 0.9:
            self.overall_status = CalibrationStatus.CONVERGED
        elif self.overall_convergence >= 0.5:
            self.overall_status = CalibrationStatus.CONVERGING
        elif self.overall_convergence > 0:
            self.overall_status = CalibrationStatus.CONVERGING
        else:
            self.overall_status = CalibrationStatus.UNINITIALIZED
    
    def to_cloud_metadata(self) -> Dict[str, Any]:
        """转换为云端上传的元数据格式"""
        return {
            "vehicle_id": self.vehicle_id,
            "overall_status": self.overall_status.value,
            "overall_convergence": self.overall_convergence,
            "health_score": self.health_score,
            "anomaly_detected": self.anomaly_detected,
            "last_calibration_time": self.last_calibration_time.isoformat() if self.last_calibration_time else None,
            "total_driving_distance": self.total_driving_distance,
            "sensor_specs": {
                sensor_id: {
                    "status": spec.status.value,
                    "confidence": spec.confidence,
                    "convergence_progress": spec.convergence_progress,
                    "extrinsics": {
                        "translation": spec.extrinsics.translation,
                        "rotation": spec.extrinsics.rotation
                    },
                    "deviation": spec.compute_deviation()
                }
                for sensor_id, spec in self.learned_specs.items()
            }
        }