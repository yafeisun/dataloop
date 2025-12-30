"""
异常处理与重置机制
实现标定异常检测和自动重置策略
"""

import numpy as np
from typing import Dict, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from common.models.calibration import CalibrationStatus, LearnedSpec


class AnomalyType(str, Enum):
    """异常类型"""
    GROUND_PLANE_TILT = "ground_plane_tilt"  # 地平面严重倾斜
    OBJECT_JUMPING = "object_jumping"  # 静止物体在3D空间中跳动
    LARGE_DEVIATION = "large_deviation"  # 偏差过大
    CONVERGENCE_FAILURE = "convergence_failure"  # 收敛失败
    SENSOR_DISCONNECTED = "sensor_disconnected"  # 传感器断开


class ResetType(str, Enum):
    """重置类型"""
    SOFT_RESET = "soft_reset"  # 软重置：扩大搜索范围，重新收敛
    HARD_RESET = "hard_reset"  # 硬重置：回到Factory Spec，要求用户重新行驶


@dataclass
class AnomalyEvent:
    """异常事件"""
    anomaly_type: AnomalyType
    sensor_id: str
    severity: str  # "low", "medium", "high", "critical"
    timestamp: datetime
    details: Dict
    reset_triggered: bool = False
    reset_type: Optional[ResetType] = None


@dataclass
class ResetEvent:
    """重置事件"""
    sensor_id: str
    reset_type: ResetType
    reason: str
    timestamp: datetime
    success: bool


class AnomalyHandler:
    """
    异常处理器
    
    这套机制必须有容错能力（Fail-safe）：
    
    动态检测：
    - 如果算法发现路边的静止物体（如灯杆）在3D空间中"跳动"，或者地平面估算严重倾斜，系统会判定标定失效。
    
    重置策略：
    - 软重置：在后台默默扩大搜索范围，重新收敛。
    - 硬重置：如果偏差过大（例如更换了前挡风玻璃），系统会强制要求用户在平直道路上行驶几公里，并在屏幕上显示进度圈（此时Autopilot不可用）。
    """
    
    def __init__(
        self,
        soft_reset_threshold: float = 0.05,
        hard_reset_threshold: float = 0.1
    ):
        """
        初始化异常处理器
        
        Args:
            soft_reset_threshold: 软重置阈值（偏差超过此值）
            hard_reset_threshold: 硬重置阈值（偏差超过此值）
        """
        self.soft_reset_threshold = soft_reset_threshold
        self.hard_reset_threshold = hard_reset_threshold
        
        # 异常事件历史
        self.anomaly_events: List[AnomalyEvent] = []
        
        # 重置事件历史
        self.reset_events: List[ResetEvent] = []
        
        # 重置回调
        self.reset_callback: Optional[Callable] = None
    
    def detect_anomalies(
        self,
        learned_specs: Dict[str, LearnedSpec],
        sensor_data: Optional[Dict] = None
    ) -> List[AnomalyEvent]:
        """
        检测异常
        
        Args:
            learned_specs: Learn Spec字典
            sensor_data: 传感器数据（可选）
        
        Returns:
            异常事件列表
        """
        anomalies = []
        
        for sensor_id, learned_spec in learned_specs.items():
            # 1. 检测偏差过大
            deviation = learned_spec.compute_deviation()
            max_rotation_dev = max(abs(deviation[0]), abs(deviation[1]), abs(deviation[2]))
            max_translation_dev = max(abs(deviation[3]), abs(deviation[4]), abs(deviation[5]))
            max_deviation = max(max_rotation_dev, max_translation_dev)
            
            if max_deviation > self.hard_reset_threshold:
                anomalies.append(AnomalyEvent(
                    anomaly_type=AnomalyType.LARGE_DEVIATION,
                    sensor_id=sensor_id,
                    severity="critical",
                    timestamp=datetime.now(),
                    details={
                        "max_deviation": max_deviation,
                        "deviation": deviation
                    }
                ))
            elif max_deviation > self.soft_reset_threshold:
                anomalies.append(AnomalyEvent(
                    anomaly_type=AnomalyType.LARGE_DEVIATION,
                    sensor_id=sensor_id,
                    severity="medium",
                    timestamp=datetime.now(),
                    details={
                        "max_deviation": max_deviation,
                        "deviation": deviation
                    }
                ))
            
            # 2. 检测收敛失败
            if learned_spec.status == CalibrationStatus.FAILED:
                anomalies.append(AnomalyEvent(
                    anomaly_type=AnomalyType.CONVERGENCE_FAILURE,
                    sensor_id=sensor_id,
                    severity="high",
                    timestamp=datetime.now(),
                    details={
                        "update_count": learned_spec.update_count,
                        "total_driving_distance": learned_spec.total_driving_distance
                    }
                ))
            
            # 3. 检测传感器断开（需要sensor_data）
            if sensor_data and sensor_id not in sensor_data:
                anomalies.append(AnomalyEvent(
                    anomaly_type=AnomalyType.SENSOR_DISCONNECTED,
                    sensor_id=sensor_id,
                    severity="critical",
                    timestamp=datetime.now(),
                    details={}
                ))
        
        # 存储异常事件
        self.anomaly_events.extend(anomalies)
        
        # 只保留最近100个异常事件
        if len(self.anomaly_events) > 100:
            self.anomaly_events = self.anomaly_events[-100:]
        
        return anomalies
    
    def detect_ground_plane_anomaly(
        self,
        ground_plane_normal: np.ndarray,
        expected_normal: np.ndarray = np.array([0, 0, 1])
    ) -> Optional[AnomalyEvent]:
        """
        检测地平面异常
        
        Args:
            ground_plane_normal: 估算的地面法向量
            expected_normal: 期望的地面法向量（默认为垂直向上）
        
        Returns:
            异常事件（如果检测到异常）
        """
        # 计算法向量夹角
        angle = np.arccos(np.clip(
            np.dot(ground_plane_normal, expected_normal) /
            (np.linalg.norm(ground_plane_normal) * np.linalg.norm(expected_normal)),
            -1.0, 1.0
        ))
        
        # 如果夹角过大（超过10度）
        if angle > np.deg2rad(10):
            return AnomalyEvent(
                anomaly_type=AnomalyType.GROUND_PLANE_TILT,
                sensor_id="all",  # 全局异常
                severity="high",
                timestamp=datetime.now(),
                details={
                    "angle_deg": np.rad2deg(angle),
                    "ground_plane_normal": ground_plane_normal.tolist(),
                    "expected_normal": expected_normal.tolist()
                }
            )
        
        return None
    
    def detect_object_jumping(
        self,
        object_positions_history: List[Dict[str, np.ndarray]]
    ) -> Optional[AnomalyEvent]:
        """
        检测物体跳动
        
        Args:
            object_positions_history: 物体位置历史 [{object_id: position}]
        
        Returns:
            异常事件（如果检测到异常）
        """
        if len(object_positions_history) < 3:
            return None
        
        # 对每个物体检测位置跳动
        jumping_objects = []
        
        for i in range(1, len(object_positions_history)):
            prev_positions = object_positions_history[i - 1]
            curr_positions = object_positions_history[i]
            
            for object_id, curr_pos in curr_positions.items():
                if object_id in prev_positions:
                    prev_pos = prev_positions[object_id]
                    
                    # 计算位置变化
                    position_change = np.linalg.norm(curr_pos - prev_pos)
                    
                    # 如果位置变化过大（超过1米）
                    if position_change > 1.0:
                        jumping_objects.append({
                            "object_id": object_id,
                            "position_change": position_change,
                            "prev_position": prev_pos.tolist(),
                            "curr_position": curr_pos.tolist()
                        })
        
        if jumping_objects:
            return AnomalyEvent(
                anomaly_type=AnomalyType.OBJECT_JUMPING,
                sensor_id="all",
                severity="high",
                timestamp=datetime.now(),
                details={
                    "jumping_objects": jumping_objects,
                    "count": len(jumping_objects)
                }
            )
        
        return None
    
    def handle_anomalies(
        self,
        anomalies: List[AnomalyEvent]
    ) -> List[ResetEvent]:
        """
        处理异常
        
        Args:
            anomalies: 异常事件列表
        
        Returns:
            重置事件列表
        """
        reset_events = []
        
        for anomaly in anomalies:
            # 根据异常严重程度决定重置策略
            if anomaly.severity == "critical":
                # 关键异常：硬重置
                reset_type = ResetType.HARD_RESET
            elif anomaly.severity == "high":
                # 高危异常：硬重置
                reset_type = ResetType.HARD_RESET
            elif anomaly.severity == "medium":
                # 中等异常：软重置
                reset_type = ResetType.SOFT_RESET
            else:
                # 低危异常：不重置
                continue
            
            # 执行重置
            success = self._execute_reset(
                anomaly.sensor_id,
                reset_type,
                anomaly.details
            )
            
            reset_event = ResetEvent(
                sensor_id=anomaly.sensor_id,
                reset_type=reset_type,
                reason=f"{anomaly.anomaly_type.value}: {anomaly.details}",
                timestamp=datetime.now(),
                success=success
            )
            
            reset_events.append(reset_event)
            
            # 更新异常事件的重置状态
            anomaly.reset_triggered = True
            anomaly.reset_type = reset_type
        
        # 存储重置事件
        self.reset_events.extend(reset_events)
        
        # 只保留最近100个重置事件
        if len(self.reset_events) > 100:
            self.reset_events = self.reset_events[-100:]
        
        return reset_events
    
    def _execute_reset(
        self,
        sensor_id: str,
        reset_type: ResetType,
        details: Dict
    ) -> bool:
        """
        执行重置
        
        Args:
            sensor_id: 传感器ID
            reset_type: 重置类型
            details: 异常详情
        
        Returns:
            是否成功
        """
        if self.reset_callback is None:
            print(f"[AnomalyHandler] No reset callback set, skipping reset")
            return False
        
        try:
            success = self.reset_callback(sensor_id, reset_type, details)
            return success
        except Exception as e:
            print(f"[AnomalyHandler] Error executing reset: {e}")
            return False
    
    def set_reset_callback(self, callback: Callable):
        """
        设置重置回调函数
        
        Args:
            callback: 重置回调函数，接收(sensor_id, reset_type, details)参数，返回bool
        """
        self.reset_callback = callback
    
    def get_anomaly_history(
        self,
        sensor_id: Optional[str] = None,
        hours: int = 24
    ) -> List[AnomalyEvent]:
        """
        获取异常历史
        
        Args:
            sensor_id: 传感器ID（None表示所有传感器）
            hours: 查询小时数
        
        Returns:
            异常事件列表
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        history = [
            event for event in self.anomaly_events
            if event.timestamp >= cutoff_time and
            (sensor_id is None or event.sensor_id == sensor_id)
        ]
        
        return history
    
    def get_reset_history(
        self,
        sensor_id: Optional[str] = None,
        hours: int = 24
    ) -> List[ResetEvent]:
        """
        获取重置历史
        
        Args:
            sensor_id: 传感器ID（None表示所有传感器）
            hours: 查询小时数
        
        Returns:
            重置事件列表
        """
        from datetime import timedelta
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        history = [
            event for event in self.reset_events
            if event.timestamp >= cutoff_time and
            (sensor_id is None or event.sensor_id == sensor_id)
        ]
        
        return history
    
    def get_statistics(self) -> Dict:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        # 异常统计
        anomaly_counts = {}
        for anomaly in self.anomaly_events:
            key = (anomaly.anomaly_type, anomaly.severity)
            anomaly_counts[key] = anomaly_counts.get(key, 0) + 1
        
        # 重置统计
        reset_counts = {}
        for reset in self.reset_events:
            key = reset.reset_type
            reset_counts[key] = reset_counts.get(key, 0) + 1
        
        return {
            "total_anomalies": len(self.anomaly_events),
            "total_resets": len(self.reset_events),
            "anomaly_distribution": anomaly_counts,
            "reset_distribution": reset_counts
        }