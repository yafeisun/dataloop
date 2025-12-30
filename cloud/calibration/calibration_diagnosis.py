"""
云端标定诊断模块
基于LLM进行问题诊断
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict

from common.models.calibration import CalibrationStatus


@dataclass
class DiagnosisResult:
    """诊断结果"""
    vehicle_id: str
    issue_type: str  # "hardware_failure", "installation_error", "calibration_failure", "normal"
    severity: str  # "low", "medium", "high", "critical"
    description: str
    recommendations: List[str]
    confidence: float  # 置信度 [0, 1]
    timestamp: datetime


class CalibrationDiagnosis:
    """
    云端标定诊断器
    
    功能：
    - 时序事件序列建模
    - 弱监督在线学习
    - 批量诊断、统计跟踪
    """
    
    def __init__(self):
        """初始化标定诊断器"""
        # 存储诊断历史
        self.diagnosis_history: Dict[str, List[DiagnosisResult]] = defaultdict(list)
        
        # 存储车辆标定历史（用于时序分析）
        self.vehicle_calibration_history: Dict[str, List[Dict]] = defaultdict(list)
    
    def diagnose_vehicle(
        self,
        vehicle_id: str,
        calibration_metadata: Dict,
        historical_data: Optional[List[Dict]] = None
    ) -> DiagnosisResult:
        """
        诊断车辆标定问题
        
        Args:
            vehicle_id: 车辆ID
            calibration_metadata: 标定元数据
            historical_data: 历史数据（可选）
        
        Returns:
            DiagnosisResult: 诊断结果
        """
        # 更新历史数据
        self.vehicle_calibration_history[vehicle_id].append(calibration_metadata)
        
        # 只保留最近30天的数据
        cutoff_time = datetime.now() - timedelta(days=30)
        self.vehicle_calibration_history[vehicle_id] = [
            data for data in self.vehicle_calibration_history[vehicle_id]
            if datetime.fromisoformat(data.get("last_calibration_time", datetime.now().isoformat())) >= cutoff_time
        ]
        
        # 执行诊断
        diagnosis = self._perform_diagnosis(
            vehicle_id,
            calibration_metadata,
            historical_data or self.vehicle_calibration_history[vehicle_id]
        )
        
        # 存储诊断结果
        self.diagnosis_history[vehicle_id].append(diagnosis)
        
        return diagnosis
    
    def _perform_diagnosis(
        self,
        vehicle_id: str,
        current_metadata: Dict,
        historical_data: List[Dict]
    ) -> DiagnosisResult:
        """
        执行诊断逻辑
        
        Args:
            vehicle_id: 车辆ID
            current_metadata: 当前标定元数据
            historical_data: 历史数据
        
        Returns:
            DiagnosisResult: 诊断结果
        """
        # 提取关键指标
        overall_status = CalibrationStatus(current_metadata["overall_status"])
        overall_convergence = current_metadata["overall_convergence"]
        health_score = current_metadata["health_score"]
        anomaly_detected = current_metadata["anomaly_detected"]
        total_driving_distance = current_metadata["total_driving_distance"]
        
        # 分析传感器偏差
        sensor_issues = self._analyze_sensor_deviations(current_metadata["sensor_specs"])
        
        # 分析时序趋势
        trend_analysis = self._analyze_trends(historical_data)
        
        # 综合诊断
        if health_score < 0.5:
            # 健康度低，可能是硬件故障
            return DiagnosisResult(
                vehicle_id=vehicle_id,
                issue_type="hardware_failure",
                severity="critical",
                description="车辆标定健康度严重偏低，可能存在传感器硬件故障",
                recommendations=[
                    "立即安排车辆进店检修",
                    "检查所有传感器连接和供电",
                    "更换疑似故障的传感器"
                ],
                confidence=0.9,
                timestamp=datetime.now()
            )
        
        elif overall_status == CalibrationStatus.FAILED:
            # 标定失败
            if total_driving_distance < 100:
                # 行驶距离太短
                return DiagnosisResult(
                    vehicle_id=vehicle_id,
                    issue_type="calibration_failure",
                    severity="low",
                    description="标定未完成，行驶距离不足",
                    recommendations=[
                        "继续行驶至少100公里以完成标定",
                        "在平直道路上行驶以提高标定精度"
                    ],
                    confidence=0.8,
                    timestamp=datetime.now()
                )
            else:
                # 行驶距离足够但仍失败
                return DiagnosisResult(
                    vehicle_id=vehicle_id,
                    issue_type="installation_error",
                    severity="high",
                    description="标定失败，可能存在传感器安装错误或严重形变",
                    recommendations=[
                        "检查传感器安装是否牢固",
                        "检查传感器是否发生物理形变",
                        "重新安装或更换传感器"
                    ],
                    confidence=0.85,
                    timestamp=datetime.now()
                )
        
        elif anomaly_detected:
            # 检测到异常
            if sensor_issues["has_large_deviation"]:
                return DiagnosisResult(
                    vehicle_id=vehicle_id,
                    issue_type="installation_error",
                    severity="medium",
                    description=f"检测到传感器偏差过大：{sensor_issues['deviation_details']}",
                    recommendations=[
                        "检查传感器安装角度",
                        "重新标定传感器",
                        "如果偏差持续存在，建议进店检修"
                    ],
                    confidence=0.75,
                    timestamp=datetime.now()
                )
            else:
                return DiagnosisResult(
                    vehicle_id=vehicle_id,
                    issue_type="calibration_failure",
                    severity="low",
                    description="检测到轻微标定异常",
                    recommendations=[
                        "继续行驶以完成标定",
                        "观察后续标定结果"
                    ],
                    confidence=0.6,
                    timestamp=datetime.now()
                )
        
        elif trend_analysis["degrading"]:
            # 标定质量在下降
            return DiagnosisResult(
                vehicle_id=vehicle_id,
                issue_type="calibration_failure",
                severity="medium",
                description="标定质量呈下降趋势，可能存在传感器老化或松动",
                recommendations=[
                    "检查传感器安装是否松动",
                    "定期进行标定检查",
                    "考虑更换老化传感器"
                ],
                confidence=0.7,
                timestamp=datetime.now()
            )
        
        else:
            # 正常
            return DiagnosisResult(
                vehicle_id=vehicle_id,
                issue_type="normal",
                severity="low",
                description="标定状态正常",
                recommendations=[
                    "继续正常使用",
                    "定期检查标定状态"
                ],
                confidence=0.95,
                timestamp=datetime.now()
            )
    
    def _analyze_sensor_deviations(
        self,
        sensor_specs: Dict[str, Dict]
    ) -> Dict:
        """
        分析传感器偏差
        
        Args:
            sensor_specs: 传感器规格字典
        
        Returns:
            分析结果字典
        """
        large_deviation_sensors = []
        deviation_details = []
        
        for sensor_id, spec_data in sensor_specs.items():
            deviation = spec_data.get("deviation", [0, 0, 0, 0, 0, 0])
            status = spec_data.get("status", CalibrationStatus.UNINITIALIZED.value)
            
            # 检查偏差是否过大
            max_rotation_dev = max(abs(deviation[0]), abs(deviation[1]), abs(deviation[2]))
            max_translation_dev = max(abs(deviation[3]), abs(deviation[4]), abs(deviation[5]))
            
            if max_rotation_dev > 0.1 or max_translation_dev > 0.1:  # 10度或10厘米
                large_deviation_sensors.append(sensor_id)
                deviation_details.append(
                    f"{sensor_id}: rotation={max_rotation_dev:.3f}rad, translation={max_translation_dev:.3f}m"
                )
        
        return {
            "has_large_deviation": len(large_deviation_sensors) > 0,
            "large_deviation_sensors": large_deviation_sensors,
            "deviation_details": "; ".join(deviation_details)
        }
    
    def _analyze_trends(self, historical_data: List[Dict]) -> Dict:
        """
        分析时序趋势
        
        Args:
            historical_data: 历史数据列表
        
        Returns:
            趋势分析结果
        """
        if len(historical_data) < 5:
            return {"degrading": False, "trend": "insufficient_data"}
        
        # 提取健康度和收敛度的时间序列
        health_scores = [
            data.get("health_score", 1.0)
            for data in historical_data
        ]
        convergence_scores = [
            data.get("overall_convergence", 0.0)
            for data in historical_data
        ]
        
        # 计算趋势（线性回归斜率）
        def compute_slope(values):
            x = np.arange(len(values))
            y = np.array(values)
            if len(x) < 2:
                return 0.0
            slope = np.polyfit(x, y, 1)[0]
            return slope
        
        health_slope = compute_slope(health_scores)
        convergence_slope = compute_slope(convergence_scores)
        
        # 判断是否下降
        degrading = health_slope < -0.01 or convergence_slope < -0.01
        
        return {
            "degrading": degrading,
            "health_slope": float(health_slope),
            "convergence_slope": float(convergence_slope),
            "trend": "degrading" if degrading else "stable"
        }
    
    def batch_diagnose(
        self,
        vehicle_ids: List[str],
        calibration_data: Dict[str, Dict]
    ) -> Dict[str, DiagnosisResult]:
        """
        批量诊断
        
        Args:
            vehicle_ids: 车辆ID列表
            calibration_data: 标定数据字典 {vehicle_id: metadata}
        
        Returns:
            诊断结果字典
        """
        results = {}
        
        for vehicle_id in vehicle_ids:
            if vehicle_id in calibration_data:
                historical_data = self.vehicle_calibration_history.get(vehicle_id, [])
                result = self.diagnose_vehicle(
                    vehicle_id,
                    calibration_data[vehicle_id],
                    historical_data
                )
                results[vehicle_id] = result
        
        return results
    
    def get_diagnosis_summary(self) -> Dict:
        """
        获取诊断摘要
        
        Returns:
            诊断摘要
        """
        total_diagnoses = sum(len(history) for history in self.diagnosis_history.values())
        
        # 统计问题类型
        issue_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        
        for history in self.diagnosis_history.values():
            for diagnosis in history:
                issue_counts[diagnosis.issue_type] += 1
                severity_counts[diagnosis.severity] += 1
        
        return {
            "total_diagnoses": total_diagnoses,
            "issue_distribution": dict(issue_counts),
            "severity_distribution": dict(severity_counts),
            "vehicles_diagnosed": len(self.diagnosis_history)
        }
    
    def get_vehicle_diagnosis_history(
        self,
        vehicle_id: str,
        days: int = 30
    ) -> List[DiagnosisResult]:
        """
        获取车辆诊断历史
        
        Args:
            vehicle_id: 车辆ID
            days: 查询天数
        
        Returns:
            诊断结果列表
        """
        if vehicle_id not in self.diagnosis_history:
            return []
        
        cutoff_time = datetime.now() - timedelta(days=days)
        
        history = [
            diagnosis for diagnosis in self.diagnosis_history[vehicle_id]
            if diagnosis.timestamp >= cutoff_time
        ]
        
        return history