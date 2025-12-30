'''
云端标定监控模块
监控全队车辆的标定状态，识别硬件质量问题
'''

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict

from common.models.calibration import CalibrationStatus, SensorType


@dataclass
class VehicleCalibrationSnapshot:
    """车辆标定快照"""
    vehicle_id: str
    timestamp: datetime
    overall_status: CalibrationStatus
    overall_convergence: float
    health_score: float
    total_driving_distance: float
    sensor_specs: Dict[str, Dict]  # {sensor_id: spec_data}
    anomaly_detected: bool


@dataclass
class CalibrationStatistics:
    """标定统计信息"""
    vehicle_count: int
    converged_count: int
    converging_count: int
    failed_count: int
    avg_convergence: float
    avg_health_score: float
    anomaly_count: int
    timestamp: datetime


@dataclass
class BatchAnomaly:
    """批次异常"""
    batch_id: str
    sensor_id: str
    anomaly_type: str  # "systematic_bias", "high_variance", "convergence_failure"
    severity: str  # "low", "medium", "high", "critical"
    affected_vehicles: List[str]
    statistics: Dict[str, float]
    timestamp: datetime


class CalibrationMonitor:
    """
    云端标定监控器
    
    云端作用：
    1. 离线大模型训练：云端Teacher Model重建场景时，必须使用车端当时时刻的标定参数，才能正确还原3D世界。
    2. 全队监控：比如监控某一批次的Model Y是否存在普遍的摄像头安装角度偏差（硬件质量问题）。
    """
    
    def __init__(self):
        """初始化标定监控器"""
        # 存储所有车辆的标定快照
        self.vehicle_snapshots: Dict[str, List[VehicleCalibrationSnapshot]] = defaultdict(list)
        
        # 存储批次信息
        self.vehicle_batches: Dict[str, str] = {}  # {vehicle_id: batch_id}
        
        # 异常记录
        self.anomalies: List[BatchAnomaly] = []
        
        # 统计信息
        self.statistics_history: List[CalibrationStatistics] = []
    
    def receive_calibration_data(self, metadata: Dict):
        """
        接收车端上传的标定数据
        
        Args:
            metadata: 标定元数据
        """
        vehicle_id = metadata["vehicle_id"]
        timestamp = datetime.fromisoformat(metadata["last_calibration_time"]) if metadata["last_calibration_time"] else datetime.now()
        
        # 创建标定快照
        snapshot = VehicleCalibrationSnapshot(
            vehicle_id=vehicle_id,
            timestamp=timestamp,
            overall_status=CalibrationStatus(metadata["overall_status"]),
            overall_convergence=metadata["overall_convergence"],
            health_score=metadata["health_score"],
            anomaly_detected=metadata["anomaly_detected"],
            total_driving_distance=metadata["total_driving_distance"],
            sensor_specs=metadata["sensor_specs"]
        )
        
        # 存储快照
        self.vehicle_snapshots[vehicle_id].append(snapshot)
        
        # 只保留最近100个快照
        if len(self.vehicle_snapshots[vehicle_id]) > 100:
            self.vehicle_snapshots[vehicle_id] = self.vehicle_snapshots[vehicle_id][-100:]
        
        print(f"[CalibrationMonitor] Received calibration data from vehicle {vehicle_id}")
    
    def register_vehicle_batch(self, vehicle_id: str, batch_id: str):
        """
        注册车辆批次信息
        
        Args:
            vehicle_id: 车辆ID
            batch_id: 批次ID
        """
        self.vehicle_batches[vehicle_id] = batch_id
    
    def compute_statistics(self) -> CalibrationStatistics:
        """
        计算全队标定统计信息
        
        Returns:
            CalibrationStatistics: 标定统计信息
        """
        vehicle_count = len(self.vehicle_snapshots)
        
        if vehicle_count == 0:
            return CalibrationStatistics(
                vehicle_count=0,
                converged_count=0,
                converging_count=0,
                failed_count=0,
                avg_convergence=0.0,
                avg_health_score=0.0,
                anomaly_count=0,
                timestamp=datetime.now()
            )
        
        # 获取所有车辆的最新快照
        latest_snapshots = []
        for snapshots in self.vehicle_snapshots.values():
            if snapshots:
                latest_snapshots.append(snapshots[-1])
        
        # 统计状态
        converged_count = sum(
            1 for s in latest_snapshots
            if s.overall_status == CalibrationStatus.CONVERGED
        )
        converging_count = sum(
            1 for s in latest_snapshots
            if s.overall_status == CalibrationStatus.CONVERGING
        )
        failed_count = sum(
            1 for s in latest_snapshots
            if s.overall_status == CalibrationStatus.FAILED
        )
        
        # 计算平均值
        avg_convergence = np.mean([s.overall_convergence for s in latest_snapshots])
        avg_health_score = np.mean([s.health_score for s in latest_snapshots])
        anomaly_count = sum(1 for s in latest_snapshots if s.anomaly_detected)
        
        statistics = CalibrationStatistics(
            vehicle_count=vehicle_count,
            converged_count=converged_count,
            converging_count=converging_count,
            failed_count=failed_count,
            avg_convergence=float(avg_convergence),
            avg_health_score=float(avg_health_score),
            anomaly_count=anomaly_count,
            timestamp=datetime.now()
        )
        
        self.statistics_history.append(statistics)
        
        return statistics
    
    def detect_batch_anomalies(self) -> List[BatchAnomaly]:
        """
        检测批次异常
        
        Returns:
            批次异常列表
        """
        anomalies = []
        
        # 按批次分组
        batch_vehicles: Dict[str, List[str]] = defaultdict(list)
        for vehicle_id, batch_id in self.vehicle_batches.items():
            batch_vehicles[batch_id].append(vehicle_id)
        
        # 检测每个批次的异常
        for batch_id, vehicle_ids in batch_vehicles.items():
            batch_anomalies = self._detect_batch_anomalies_for_batch(batch_id, vehicle_ids)
            anomalies.extend(batch_anomalies)
        
        self.anomalies.extend(anomalies)
        
        return anomalies
    
    def _detect_batch_anomalies_for_batch(
        self,
        batch_id: str,
        vehicle_ids: List[str]
    ) -> List[BatchAnomaly]:
        """
        检测指定批次的异常
        
        Args:
            batch_id: 批次ID
            vehicle_ids: 车辆ID列表
        
        Returns:
            批次异常列表
        """
        anomalies = []
        
        # 获取该批次所有车辆的最新快照
        batch_snapshots = []
        for vehicle_id in vehicle_ids:
            if vehicle_id in self.vehicle_snapshots and self.vehicle_snapshots[vehicle_id]:
                batch_snapshots.append(self.vehicle_snapshots[vehicle_id][-1])
        
        if len(batch_snapshots) < 3:
            return anomalies  # 样本太少，不做分析
        
        # 按传感器分组分析
        sensor_ids = set()
        for snapshot in batch_snapshots:
            sensor_ids.update(snapshot.sensor_specs.keys())
        
        for sensor_id in sensor_ids:
            # 收集该传感器的偏差数据
            deviations = []
            for snapshot in batch_snapshots:
                if sensor_id in snapshot.sensor_specs:
                    spec_data = snapshot.sensor_specs[sensor_id]
                    deviation = spec_data.get("deviation", [0, 0, 0, 0, 0, 0])
                    deviations.append(deviation)
            
            if len(deviations) < 3:
                continue
            
            deviations = np.array(deviations)
            
            # 1. 检测系统性偏差（均值偏离0太远）
            mean_deviation = np.mean(deviations, axis=0)
            max_mean_dev = np.max(np.abs(mean_deviation))
            
            if max_mean_dev > 0.05:  # 5度或5厘米
                anomalies.append(BatchAnomaly(
                    batch_id=batch_id,
                    sensor_id=sensor_id,
                    anomaly_type="systematic_bias",
                    severity="high" if max_mean_dev > 0.1 else "medium",
                    affected_vehicles=[s.vehicle_id for s in batch_snapshots],
                    statistics={
                        "mean_deviation": mean_deviation.tolist(),
                        "max_mean_deviation": float(max_mean_dev)
                    },
                    timestamp=datetime.now()
                ))
            
            # 2. 检测高方差（一致性差）
            std_deviation = np.std(deviations, axis=0)
            max_std_dev = np.max(std_deviation)
            
            if max_std_dev > 0.02:  # 2度或2厘米
                anomalies.append(BatchAnomaly(
                    batch_id=batch_id,
                    sensor_id=sensor_id,
                    anomaly_type="high_variance",
                    severity="medium" if max_std_dev > 0.05 else "low",
                    affected_vehicles=[s.vehicle_id for s in batch_snapshots],
                    statistics={
                        "std_deviation": std_deviation.tolist(),
                        "max_std_deviation": float(max_std_dev)
                    },
                    timestamp=datetime.now()
                ))
            
            # 3. 检测收敛失败
            failed_count = sum(
                1 for s in batch_snapshots
                if sensor_id in s.sensor_specs and
                s.sensor_specs[sensor_id].get("status") == CalibrationStatus.FAILED.value
            )
            
            if failed_count > len(batch_snapshots) * 0.3:  # 超过30%失败
                anomalies.append(BatchAnomaly(
                    batch_id=batch_id,
                    sensor_id=sensor_id,
                    anomaly_type="convergence_failure",
                    severity="critical",
                    affected_vehicles=[
                        s.vehicle_id for s in batch_snapshots
                        if sensor_id in s.sensor_specs and
                        s.sensor_specs[sensor_id].get("status") == CalibrationStatus.FAILED.value
                    ],
                    statistics={
                        "failed_count": failed_count,
                        "total_count": len(batch_snapshots),
                        "failure_rate": failed_count / len(batch_snapshots)
                    },
                    timestamp=datetime.now()
                ))
        
        return anomalies
    
    def get_vehicle_history(
        self,
        vehicle_id: str,
        days: int = 7
    ) -> List[VehicleCalibrationSnapshot]:
        """
        获取车辆标定历史
        
        Args:
            vehicle_id: 车辆ID
            days: 查询天数
        
        Returns:
            标定快照列表
        """
        if vehicle_id not in self.vehicle_snapshots:
            return []
        
        cutoff_time = datetime.now() - timedelta(days=days)
        
        history = [
            snapshot for snapshot in self.vehicle_snapshots[vehicle_id]
            if snapshot.timestamp >= cutoff_time
        ]
        
        return history
    
    def get_batch_statistics(
        self,
        batch_id: str
    ) -> Optional[Dict]:
        """
        获取批次统计信息
        
        Args:
            batch_id: 批次ID
        
        Returns:
            批次统计信息
        """
        vehicle_ids = [
            vid for vid, bid in self.vehicle_batches.items()
            if bid == batch_id
        ]
        
        if not vehicle_ids:
            return None
        
        # 获取该批次所有车辆的最新快照
        batch_snapshots = []
        for vehicle_id in vehicle_ids:
            if vehicle_id in self.vehicle_snapshots and self.vehicle_snapshots[vehicle_id]:
                batch_snapshots.append(self.vehicle_snapshots[vehicle_id][-1])
        
        if not batch_snapshots:
            return None
        
        # 计算统计信息
        avg_convergence = np.mean([s.overall_convergence for s in batch_snapshots])
        avg_health_score = np.mean([s.health_score for s in batch_snapshots])
        anomaly_count = sum(1 for s in batch_snapshots if s.anomaly_detected)
        
        return {
            "batch_id": batch_id,
            "vehicle_count": len(batch_snapshots),
            "avg_convergence": float(avg_convergence),
            "avg_health_score": float(avg_health_score),
            "anomaly_count": anomaly_count,
            "anomaly_rate": anomaly_count / len(batch_snapshots)
        }
    
    def get_anomaly_report(self) -> List[Dict]:
        """
        获取异常报告
        
        Returns:
            异常报告列表
        """
        return [
            {
                "batch_id": anomaly.batch_id,
                "sensor_id": anomaly.sensor_id,
                "anomaly_type": anomaly.anomaly_type,
                "severity": anomaly.severity,
                "affected_vehicles": anomaly.affected_vehicles,
                "statistics": anomaly.statistics,
                "timestamp": anomaly.timestamp.isoformat()
            }
            for anomaly in self.anomalies
        ]
    
    def generate_alert(self) -> Optional[str]:
        """
        生成告警信息
        
        Returns:
            告警信息（如果没有异常则返回None）
        """
        critical_anomalies = [
            a for a in self.anomalies
            if a.severity == "critical"
        ]
        
        if not critical_anomalies:
            return None
        
        # 生成告警信息
        alert_lines = [
            "CRITICAL CALIBRATION ANOMALY DETECTED",
            "=" * 50
        ]
        
        for anomaly in critical_anomalies:
            alert_lines.append(f"\nBatch: {anomaly.batch_id}")
            alert_lines.append(f"Sensor: {anomaly.sensor_id}")
            alert_lines.append(f"Type: {anomaly.anomaly_type}")
            alert_lines.append(f"Affected Vehicles: {len(anomaly.affected_vehicles)}")
            alert_lines.append(f"Statistics: {anomaly.statistics}")
        
        return "\n".join(alert_lines)
