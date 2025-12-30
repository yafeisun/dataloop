"""
批量分析模块
对全队车辆进行批量分析和统计
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from collections import defaultdict

from common.models.calibration import CalibrationStatus


@dataclass
class BatchAnalysisResult:
    """批量分析结果"""
    batch_id: str
    sensor_id: str
    vehicle_count: int
    mean_convergence: float
    std_convergence: float
    mean_health_score: float
    std_health_score: float
    mean_deviation: Tuple[float, float, float, float, float, float]  # (roll, pitch, yaw, x, y, z)
    std_deviation: Tuple[float, float, float, float, float, float]
    outlier_count: int
    outlier_rate: float
    timestamp: datetime


@dataclass
class OutlierVehicle:
    """异常车辆"""
    vehicle_id: str
    sensor_id: str
    deviation: Tuple[float, float, float, float, float, float]
    z_score: float
    severity: str


class BatchAnalysis:
    """
    批量分析器
    
    功能：
    - 对全队车辆进行统计分析
    - 识别异常车辆
    - 生成分析报告
    """
    
    def __init__(self):
        """初始化批量分析器"""
        self.results: List[BatchAnalysisResult] = []
        self.outliers: List[OutlierVehicle] = []
    
    def analyze_batch(
        self,
        batch_id: str,
        vehicle_data: Dict[str, Dict]  # {vehicle_id: calibration_metadata}
    ) -> Dict[str, BatchAnalysisResult]:
        """
        分析批次
        
        Args:
            batch_id: 批次ID
            vehicle_data: 车辆数据字典
        
        Returns:
            分析结果字典 {sensor_id: BatchAnalysisResult}
        """
        if not vehicle_data:
            return {}
        
        # 提取所有传感器ID
        sensor_ids = set()
        for metadata in vehicle_data.values():
            sensor_ids.update(metadata["sensor_specs"].keys())
        
        results = {}
        
        for sensor_id in sensor_ids:
            # 分析单个传感器
            result = self._analyze_sensor(batch_id, sensor_id, vehicle_data)
            if result:
                results[sensor_id] = result
                self.results.append(result)
        
        return results
    
    def _analyze_sensor(
        self,
        batch_id: str,
        sensor_id: str,
        vehicle_data: Dict[str, Dict]
    ) -> Optional[BatchAnalysisResult]:
        """
        分析单个传感器
        
        Args:
            batch_id: 批次ID
            sensor_id: 传感器ID
            vehicle_data: 车辆数据字典
        
        Returns:
            BatchAnalysisResult: 分析结果
        """
        # 收集该传感器的数据
        sensor_data = []
        vehicle_ids = []
        
        for vehicle_id, metadata in vehicle_data.items():
            if sensor_id in metadata["sensor_specs"]:
                spec_data = metadata["sensor_specs"][sensor_id]
                sensor_data.append({
                    "vehicle_id": vehicle_id,
                    "convergence": spec_data.get("convergence_progress", 0.0),
                    "health_score": metadata.get("health_score", 1.0),
                    "deviation": spec_data.get("deviation", [0, 0, 0, 0, 0, 0])
                })
                vehicle_ids.append(vehicle_id)
        
        if len(sensor_data) < 3:
            return None  # 样本太少
        
        # 提取数据
        convergences = [d["convergence"] for d in sensor_data]
        health_scores = [d["health_score"] for d in sensor_data]
        deviations = np.array([d["deviation"] for d in sensor_data])
        
        # 计算统计量
        mean_convergence = np.mean(convergences)
        std_convergence = np.std(convergences)
        mean_health_score = np.mean(health_scores)
        std_health_score = np.std(health_scores)
        mean_deviation = tuple(np.mean(deviations, axis=0))
        std_deviation = tuple(np.std(deviations, axis=0))
        
        # 识别异常值（使用Z-score）
        z_scores = np.abs((deviations - np.mean(deviations, axis=0)) / (np.std(deviations, axis=0) + 1e-6))
        max_z_scores = np.max(z_scores, axis=1)
        outlier_threshold = 2.5  # Z-score阈值
        
        outlier_indices = np.where(max_z_scores > outlier_threshold)[0]
        outlier_count = len(outlier_indices)
        outlier_rate = outlier_count / len(sensor_data)
        
        # 记录异常车辆
        for idx in outlier_indices:
            vehicle_id = sensor_data[idx]["vehicle_id"]
            deviation = tuple(sensor_data[idx]["deviation"])
            z_score = float(max_z_scores[idx])
            
            severity = "high" if z_score > 3.0 else "medium"
            
            self.outliers.append(OutlierVehicle(
                vehicle_id=vehicle_id,
                sensor_id=sensor_id,
                deviation=deviation,
                z_score=z_score,
                severity=severity
            ))
        
        return BatchAnalysisResult(
            batch_id=batch_id,
            sensor_id=sensor_id,
            vehicle_count=len(sensor_data),
            mean_convergence=float(mean_convergence),
            std_convergence=float(std_convergence),
            mean_health_score=float(mean_health_score),
            std_health_score=float(std_health_score),
            mean_deviation=mean_deviation,
            std_deviation=std_deviation,
            outlier_count=outlier_count,
            outlier_rate=float(outlier_rate),
            timestamp=datetime.now()
        )
    
    def compare_batches(
        self,
        batch_results: Dict[str, Dict[str, BatchAnalysisResult]]
    ) -> Dict:
        """
        比较多个批次
        
        Args:
            batch_results: 批次结果字典 {batch_id: {sensor_id: BatchAnalysisResult}}
        
        Returns:
            比较结果
        """
        if len(batch_results) < 2:
            return {"error": "Need at least 2 batches for comparison"}
        
        # 提取所有传感器ID
        sensor_ids = set()
        for results in batch_results.values():
            sensor_ids.update(results.keys())
        
        comparison = {}
        
        for sensor_id in sensor_ids:
            # 收集该传感器在各个批次的数据
            batch_data = []
            for batch_id, results in batch_results.items():
                if sensor_id in results:
                    result = results[sensor_id]
                    batch_data.append({
                        "batch_id": batch_id,
                        "mean_convergence": result.mean_convergence,
                        "mean_health_score": result.mean_health_score,
                        "mean_deviation": result.mean_deviation,
                        "outlier_rate": result.outlier_rate
                    })
            
            if len(batch_data) < 2:
                continue
            
            # 计算批次间差异
            convergences = [d["mean_convergence"] for d in batch_data]
            health_scores = [d["mean_health_score"] for d in batch_data]
            outlier_rates = [d["outlier_rate"] for d in batch_data]
            
            comparison[sensor_id] = {
                "convergence_range": (min(convergences), max(convergences)),
                "convergence_std": float(np.std(convergences)),
                "health_score_range": (min(health_scores), max(health_scores)),
                "health_score_std": float(np.std(health_scores)),
                "outlier_rate_range": (min(outlier_rates), max(outlier_rates)),
                "batch_details": batch_data
            }
        
        return comparison
    
    def get_outlier_report(self) -> List[Dict]:
        """
        获取异常车辆报告
        
        Returns:
            异常车辆报告列表
        """
        return [
            {
                "vehicle_id": outlier.vehicle_id,
                "sensor_id": outlier.sensor_id,
                "deviation": outlier.deviation,
                "z_score": outlier.z_score,
                "severity": outlier.severity
            }
            for outlier in self.outliers
        ]
    
    def get_summary_report(self) -> Dict:
        """
        获取摘要报告
        
        Returns:
            摘要报告
        """
        if not self.results:
            return {"message": "No analysis results available"}
        
        # 统计信息
        total_analyses = len(self.results)
        total_vehicles = sum(r.vehicle_count for r in self.results)
        total_outliers = len(self.outliers)
        
        # 按传感器统计
        sensor_stats = defaultdict(lambda: {"count": 0, "total_vehicles": 0, "total_outliers": 0})
        for result in self.results:
            sensor_stats[result.sensor_id]["count"] += 1
            sensor_stats[result.sensor_id]["total_vehicles"] += result.vehicle_count
            sensor_stats[result.sensor_id]["total_outliers"] += result.outlier_count
        
        return {
            "total_analyses": total_analyses,
            "total_vehicles": total_vehicles,
            "total_outliers": total_outliers,
            "overall_outlier_rate": total_outliers / total_vehicles if total_vehicles > 0 else 0.0,
            "sensor_statistics": dict(sensor_stats),
            "timestamp": datetime.now().isoformat()
        }
    
    def clear_results(self):
        """清除结果"""
        self.results.clear()
        self.outliers.clear()