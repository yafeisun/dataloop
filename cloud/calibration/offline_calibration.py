"""
云端离线数据标定模块
从海量数据中挖掘"黄金标定片段" (Golden Clips)
通过三层漏斗筛选机制筛选高质量标定数据
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import cv2


class FilterLayer(Enum):
    """筛选层级"""
    VEHICLE_DYNAMICS = "vehicle_dynamics"  # 第一层：车辆动力学筛选
    VISUAL_PERCEPTION = "visual_perception"  # 第二层：视觉/感知质量筛选
    OBSERVABILITY = "observability"  # 第三层：可观测性与几何校验


class CalibrationClipType(Enum):
    """标定片段类型"""
    STRAIGHT_ROAD = "straight_road"  # 直道（适合标定Yaw/Pitch）
    CURVED_ROAD = "curved_road"  # 弯道（适合标定Translation）
    RICH_TEXTURE = "rich_texture"  # 丰富纹理（适合标定所有参数）
    LOOP_CLOSURE = "loop_closure"  # 闭环（最高质量）


@dataclass
class GoldenClip:
    """黄金标定片段"""
    clip_id: str
    vehicle_id: str
    start_timestamp: datetime
    end_timestamp: datetime
    duration: float  # 秒
    clip_type: CalibrationClipType
    quality_score: float  # 质量分数 [0, 1]
    
    # 第一层：车辆动力学指标
    avg_speed: float  # km/h
    std_speed: float
    avg_steering_angle: float  # 度
    max_steering_angle: float
    avg_longitudinal_acc: float  # m/s²
    max_longitudinal_acc: float
    avg_yaw_rate: float  # deg/s
    max_yaw_rate: float
    imu_acc_z_variance: float
    
    # 第二层：视觉/感知指标
    avg_feature_points: int
    min_feature_points: int
    dynamic_object_ratio: float
    lane_confidence: float
    lighting_quality: str  # "good", "overexposed", "underexposed", "uneven"
    blur_score: float  # [0, 1], 1表示清晰
    
    # 第三层：可观测性指标
    vanishing_point_variance: float
    fim_min_singular_value: float
    has_loop_closure: bool
    
    # 场景描述
    scene_description: str
    
    # 文件路径
    data_paths: Dict[str, str]  # {"camera_front": "/path/to/data", "imu": "/path/to/data", ...}


@dataclass
class FilterResult:
    """筛选结果"""
    layer: FilterLayer
    passed: bool
    reason: str
    metrics: Dict[str, float]
    timestamp: datetime


class VehicleDynamicsFilter:
    """
    第一层：车辆动力学筛选
    
    目的：确保车辆姿态稳定，避免因急加减速导致悬挂形变，影响外参（特别是 Pitch 角）。
    数据源：CAN 总线、IMU
    """
    
    def __init__(
        self,
        min_speed: float = 30.0,  # km/h
        max_speed: float = 80.0,  # km/h
        max_steering_angle: float = 1.0,  # 度
        max_longitudinal_acc: float = 0.5,  # m/s²
        max_yaw_rate: float = 2.0,  # deg/s
        max_imu_acc_z_variance: float = 0.1  # m²/s⁴
    ):
        """
        初始化车辆动力学筛选器
        
        Args:
            min_speed: 最小速度
            max_speed: 最大速度
            max_steering_angle: 最大方向盘转角
            max_longitudinal_acc: 最大纵向加速度
            max_yaw_rate: 最大横摆角速度
            max_imu_acc_z_variance: IMU Z轴加速度最大方差
        """
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.max_steering_angle = max_steering_angle
        self.max_longitudinal_acc = max_longitudinal_acc
        self.max_yaw_rate = max_yaw_rate
        self.max_imu_acc_z_variance = max_imu_acc_z_variance
    
    def filter(
        self,
        vehicle_data: Dict[str, np.ndarray]
    ) -> FilterResult:
        """
        筛选车辆动力学数据
        
        Args:
            vehicle_data: 车辆数据字典
                - "speed": 速度序列 (km/h)
                - "steering_angle": 方向盘转角序列 (度)
                - "longitudinal_acc": 纵向加速度序列 (m/s²)
                - "yaw_rate": 横摆角速度序列 (deg/s)
                - "imu_acc_z": IMU Z轴加速度序列 (m/s²)
        
        Returns:
            FilterResult: 筛选结果
        """
        metrics = {}
        
        # 提取数据
        speed = vehicle_data.get("speed", np.array([]))
        steering_angle = vehicle_data.get("steering_angle", np.array([]))
        longitudinal_acc = vehicle_data.get("longitudinal_acc", np.array([]))
        yaw_rate = vehicle_data.get("yaw_rate", np.array([]))
        imu_acc_z = vehicle_data.get("imu_acc_z", np.array([]))
        
        # 计算统计指标
        metrics["avg_speed"] = float(np.mean(speed)) if len(speed) > 0 else 0.0
        metrics["std_speed"] = float(np.std(speed)) if len(speed) > 0 else 0.0
        metrics["avg_steering_angle"] = float(np.mean(np.abs(steering_angle))) if len(steering_angle) > 0 else 0.0
        metrics["max_steering_angle"] = float(np.max(np.abs(steering_angle))) if len(steering_angle) > 0 else 0.0
        metrics["avg_longitudinal_acc"] = float(np.mean(np.abs(longitudinal_acc))) if len(longitudinal_acc) > 0 else 0.0
        metrics["max_longitudinal_acc"] = float(np.max(np.abs(longitudinal_acc))) if len(longitudinal_acc) > 0 else 0.0
        metrics["avg_yaw_rate"] = float(np.mean(np.abs(yaw_rate))) if len(yaw_rate) > 0 else 0.0
        metrics["max_yaw_rate"] = float(np.max(np.abs(yaw_rate))) if len(yaw_rate) > 0 else 0.0
        metrics["imu_acc_z_variance"] = float(np.var(imu_acc_z)) if len(imu_acc_z) > 0 else 0.0
        
        # 筛选条件
        reasons = []
        
        # 1. 速度适中且稳定
        if not (self.min_speed <= metrics["avg_speed"] <= self.max_speed):
            reasons.append(f"速度不满足条件（{metrics['avg_speed']:.1f} km/h，要求 {self.min_speed}-{self.max_speed} km/h）")
        
        if metrics["std_speed"] > 10.0:  # 速度波动太大
            reasons.append(f"速度不稳定（标准差 {metrics['std_speed']:.1f} km/h）")
        
        # 2. 直线行驶
        if metrics["max_steering_angle"] > self.max_steering_angle:
            reasons.append(f"方向盘转角过大（{metrics['max_steering_angle']:.2f}°，要求 < {self.max_steering_angle}°）")
        
        if metrics["max_yaw_rate"] > self.max_yaw_rate:
            reasons.append(f"横摆角速度过大（{metrics['max_yaw_rate']:.2f}°/s，要求 < {self.max_yaw_rate}°/s）")
        
        # 3. 路面平整度
        if metrics["imu_acc_z_variance"] > self.max_imu_acc_z_variance:
            reasons.append(f"路面不平整（IMU Z轴加速度方差 {metrics['imu_acc_z_variance']:.4f} m²/s⁴）")
        
        # 4. 纵向加速度
        if metrics["max_longitudinal_acc"] > self.max_longitudinal_acc:
            reasons.append(f"纵向加速度过大（{metrics['max_longitudinal_acc']:.2f} m/s²，要求 < {self.max_longitudinal_acc} m/s²）")
        
        passed = len(reasons) == 0
        
        return FilterResult(
            layer=FilterLayer.VEHICLE_DYNAMICS,
            passed=passed,
            reason="; ".join(reasons) if reasons else "满足所有车辆动力学条件",
            metrics=metrics,
            timestamp=datetime.now()
        )


class VisualPerceptionFilter:
    """
    第二层：视觉/感知质量筛选
    
    目的：确保环境特征丰富且清晰，满足几何计算的约束条件。
    数据源：图像、点云、基础感知结果
    """
    
    def __init__(
        self,
        min_feature_points: int = 500,
        max_dynamic_object_ratio: float = 0.15,
        min_lane_confidence: float = 0.9,
        min_blur_score: float = 0.5
    ):
        """
        初始化视觉/感知筛选器
        
        Args:
            min_feature_points: 最小特征点数量
            max_dynamic_object_ratio: 最大动态物体占比
            min_lane_confidence: 最小车道线置信度
            min_blur_score: 最小模糊分数
        """
        self.min_feature_points = min_feature_points
        self.max_dynamic_object_ratio = max_dynamic_object_ratio
        self.min_lane_confidence = min_lane_confidence
        self.min_blur_score = min_blur_score
        
        # 初始化ORB特征检测器
        self.orb = cv2.ORB_create(nfeatures=2000)
    
    def filter(
        self,
        visual_data: Dict[str, Any]
    ) -> FilterResult:
        """
        筛选视觉/感知数据
        
        Args:
            visual_data: 视觉数据字典
                - "images": 图像列表 (List[np.ndarray])
                - "perception_results": 感知结果列表 (List[Dict])
                - "lane_detection": 车道线检测结果 (List[Dict])
        
        Returns:
            FilterResult: 筛选结果
        """
        metrics = {}
        reasons = []
        
        images = visual_data.get("images", [])
        perception_results = visual_data.get("perception_results", [])
        lane_detection = visual_data.get("lane_detection", [])
        
        if len(images) == 0:
            return FilterResult(
                layer=FilterLayer.VISUAL_PERCEPTION,
                passed=False,
                reason="没有图像数据",
                metrics={},
                timestamp=datetime.now()
            )
        
        # 1. 纹理丰富度
        feature_counts = []
        for img in images:
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            
            # 检测ORB特征点
            keypoints = self.orb.detect(gray, None)
            feature_counts.append(len(keypoints))
        
        metrics["avg_feature_points"] = float(np.mean(feature_counts))
        metrics["min_feature_points"] = int(np.min(feature_counts))
        
        if metrics["min_feature_points"] < self.min_feature_points:
            reasons.append(f"特征点数量不足（最小 {metrics['min_feature_points']}，要求 > {self.min_feature_points}）")
        
        # 2. 静态环境占比
        if len(perception_results) > 0:
            dynamic_ratios = []
            for result in perception_results:
                objects = result.get("objects", [])
                dynamic_objects = [obj for obj in objects if obj.get("is_dynamic", False)]
                ratio = len(dynamic_objects) / len(objects) if len(objects) > 0 else 0.0
                dynamic_ratios.append(ratio)
            
            metrics["avg_dynamic_object_ratio"] = float(np.mean(dynamic_ratios))
            
            if metrics["avg_dynamic_object_ratio"] > self.max_dynamic_object_ratio:
                reasons.append(f"动态物体占比过高（{metrics['avg_dynamic_object_ratio']*100:.1f}%，要求 < {self.max_dynamic_object_ratio*100:.1f}%）")
        else:
            metrics["avg_dynamic_object_ratio"] = 0.0
        
        # 3. 车道线清晰度
        if len(lane_detection) > 0:
            confidences = [ld.get("confidence", 0.0) for ld in lane_detection]
            metrics["avg_lane_confidence"] = float(np.mean(confidences))
            
            if metrics["avg_lane_confidence"] < self.min_lane_confidence:
                reasons.append(f"车道线置信度不足（{metrics['avg_lane_confidence']:.2f}，要求 > {self.min_lane_confidence}）")
        else:
            metrics["avg_lane_confidence"] = 0.0
            reasons.append("未检测到车道线")
        
        # 4. 光照与模糊检测
        blur_scores = []
        lighting_qualities = []
        
        for img in images:
            # 模糊检测（拉普拉斯算子）
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_score = min(laplacian_var / 100.0, 1.0)  # 归一化到 [0, 1]
            blur_scores.append(blur_score)
            
            # 光照检测
            brightness = np.mean(gray) / 255.0
            if brightness < 0.1:
                lighting_qualities.append("underexposed")
            elif brightness > 0.9:
                lighting_qualities.append("overexposed")
            elif np.std(gray) / 255.0 < 0.1:
                lighting_qualities.append("uneven")
            else:
                lighting_qualities.append("good")
        
        metrics["avg_blur_score"] = float(np.mean(blur_scores))
        
        # 统计光照质量
        lighting_counts = {}
        for lq in lighting_qualities:
            lighting_counts[lq] = lighting_counts.get(lq, 0) + 1
        metrics["lighting_quality"] = max(lighting_counts, key=lighting_counts.get)
        
        if metrics["avg_blur_score"] < self.min_blur_score:
            reasons.append(f"图像模糊（模糊分数 {metrics['avg_blur_score']:.2f}，要求 > {self.min_blur_score}）")
        
        if metrics["lighting_quality"] != "good":
            reasons.append(f"光照质量不佳（{metrics['lighting_quality']}）")
        
        passed = len(reasons) == 0
        
        return FilterResult(
            layer=FilterLayer.VISUAL_PERCEPTION,
            passed=passed,
            reason="; ".join(reasons) if reasons else "满足所有视觉/感知条件",
            metrics=metrics,
            timestamp=datetime.now()
        )


class ObservabilityFilter:
    """
    第三层：可观测性与几何校验
    
    目的：从数学角度确认这数据能不能算出唯一解。这是最高阶的筛选。
    数据源：SLAM 前端、标定预处理算法
    """
    
    def __init__(
        self,
        max_vp_variance: float = 50.0,  # 像素²
        min_fim_singular_value: float = 1e-3
    ):
        """
        初始化可观测性筛选器
        
        Args:
            max_vp_variance: 消失点最大方差
            min_fim_singular_value: FIM最小奇异值阈值
        """
        self.max_vp_variance = max_vp_variance
        self.min_fim_singular_value = min_fim_singular_value
    
    def filter(
        self,
        observability_data: Dict[str, Any]
    ) -> FilterResult:
        """
        筛选可观测性数据
        
        Args:
            observability_data: 可观测性数据字典
                - "vanishing_points": 消失点列表 (List[Tuple[float, float]])
                - "fim": Fisher Information Matrix (np.ndarray)
                - "has_loop_closure": 是否有闭环 (bool)
        
        Returns:
            FilterResult: 筛选结果
        """
        metrics = {}
        reasons = []
        
        vanishing_points = observability_data.get("vanishing_points", [])
        fim = observability_data.get("fim", None)
        has_loop_closure = observability_data.get("has_loop_closure", False)
        
        # 1. 消失点一致性
        if len(vanishing_points) > 0:
            vp_array = np.array(vanishing_points)
            vp_center = np.mean(vp_array, axis=0)
            vp_variance = np.mean(np.sum((vp_array - vp_center) ** 2, axis=1))
            
            metrics["vanishing_point_variance"] = float(vp_variance)
            
            if vp_variance > self.max_vp_variance:
                reasons.append(f"消失点不稳定（方差 {vp_variance:.1f} 像素²，要求 < {self.max_vp_variance} 像素²）")
        else:
            metrics["vanishing_point_variance"] = 0.0
            reasons.append("未检测到消失点")
        
        # 2. 激励充分性（Fisher Information Matrix）
        if fim is not None and isinstance(fim, np.ndarray):
            singular_values = np.linalg.svd(fim, compute_uv=False)
            min_singular_value = np.min(singular_values)
            
            metrics["fim_min_singular_value"] = float(min_singular_value)
            
            if min_singular_value < self.min_fim_singular_value:
                reasons.append(f"激励不充分（FIM最小奇异值 {min_singular_value:.2e}，要求 > {self.min_fim_singular_value:.2e}）")
        else:
            metrics["fim_min_singular_value"] = 0.0
            reasons.append("未提供Fisher Information Matrix")
        
        # 3. 闭环检测（可选，加分项）
        metrics["has_loop_closure"] = has_loop_closure
        
        passed = len(reasons) == 0
        
        return FilterResult(
            layer=FilterLayer.OBSERVABILITY,
            passed=passed,
            reason="; ".join(reasons) if reasons else "满足所有可观测性条件",
            metrics=metrics,
            timestamp=datetime.now()
        )


class OfflineCalibrationMiner:
    """
    离线标定数据挖掘器
    
    功能：
    - 从海量数据中挖掘"黄金标定片段"
    - 三层漏斗筛选机制
    - 自动分类片段类型
    - 生成质量评分
    """
    
    def __init__(
        self,
        dynamics_filter: Optional[VehicleDynamicsFilter] = None,
        visual_filter: Optional[VisualPerceptionFilter] = None,
        observability_filter: Optional[ObservabilityFilter] = None
    ):
        """
        初始化离线标定数据挖掘器
        
        Args:
            dynamics_filter: 车辆动力学筛选器
            visual_filter: 视觉/感知筛选器
            observability_filter: 可观测性筛选器
        """
        self.dynamics_filter = dynamics_filter or VehicleDynamicsFilter()
        self.visual_filter = visual_filter or VisualPerceptionFilter()
        self.observability_filter = observability_filter or ObservabilityFilter()
        
        # 存储挖掘结果
        self.golden_clips: List[GoldenClip] = []
        self.filter_history: List[FilterResult] = []
    
    def mine_golden_clips(
        self,
        vehicle_id: str,
        data_segments: List[Dict[str, Any]]
    ) -> List[GoldenClip]:
        """
        挖掘黄金标定片段
        
        Args:
            vehicle_id: 车辆ID
            data_segments: 数据段列表，每个段包含：
                - "start_timestamp": 开始时间
                - "end_timestamp": 结束时间
                - "vehicle_data": 车辆动力学数据
                - "visual_data": 视觉/感知数据
                - "observability_data": 可观测性数据
                - "data_paths": 数据文件路径
        
        Returns:
            黄金标定片段列表
        """
        golden_clips = []
        
        for i, segment in enumerate(data_segments):
            # 三层漏斗筛选
            filter_results = []
            
            # 第一层：车辆动力学筛选
            dynamics_result = self.dynamics_filter.filter(segment["vehicle_data"])
            filter_results.append(dynamics_result)
            self.filter_history.append(dynamics_result)
            
            if not dynamics_result.passed:
                continue  # 未通过第一层，跳过
            
            # 第二层：视觉/感知筛选
            visual_result = self.visual_filter.filter(segment["visual_data"])
            filter_results.append(visual_result)
            self.filter_history.append(visual_result)
            
            if not visual_result.passed:
                continue  # 未通过第二层，跳过
            
            # 第三层：可观测性筛选
            observability_result = self.observability_filter.filter(segment["observability_data"])
            filter_results.append(observability_result)
            self.filter_history.append(observability_result)
            
            if not observability_result.passed:
                continue  # 未通过第三层，跳过
            
            # 通过所有筛选，创建黄金片段
            golden_clip = self._create_golden_clip(
                vehicle_id,
                segment,
                filter_results
            )
            golden_clips.append(golden_clip)
        
        self.golden_clips.extend(golden_clips)
        
        return golden_clips
    
    def _create_golden_clip(
        self,
        vehicle_id: str,
        segment: Dict[str, Any],
        filter_results: List[FilterResult]
    ) -> GoldenClip:
        """
        创建黄金标定片段
        
        Args:
            vehicle_id: 车辆ID
            segment: 数据段
            filter_results: 筛选结果列表
        
        Returns:
            GoldenClip: 黄金标定片段
        """
        start_timestamp = segment["start_timestamp"]
        end_timestamp = segment["end_timestamp"]
        duration = (end_timestamp - start_timestamp).total_seconds()
        
        # 提取各层指标
        dynamics_metrics = filter_results[0].metrics
        visual_metrics = filter_results[1].metrics
        observability_metrics = filter_results[2].metrics
        
        # 计算质量分数
        quality_score = self._compute_quality_score(
            dynamics_metrics,
            visual_metrics,
            observability_metrics
        )
        
        # 确定片段类型
        clip_type = self._determine_clip_type(
            dynamics_metrics,
            visual_metrics,
            observability_metrics
        )
        
        # 生成场景描述
        scene_description = self._generate_scene_description(
            dynamics_metrics,
            visual_metrics,
            observability_metrics
        )
        
        clip_id = f"{vehicle_id}_{start_timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        return GoldenClip(
            clip_id=clip_id,
            vehicle_id=vehicle_id,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            duration=duration,
            clip_type=clip_type,
            quality_score=quality_score,
            
            # 第一层指标
            avg_speed=dynamics_metrics["avg_speed"],
            std_speed=dynamics_metrics["std_speed"],
            avg_steering_angle=dynamics_metrics["avg_steering_angle"],
            max_steering_angle=dynamics_metrics["max_steering_angle"],
            avg_longitudinal_acc=dynamics_metrics["avg_longitudinal_acc"],
            max_longitudinal_acc=dynamics_metrics["max_longitudinal_acc"],
            avg_yaw_rate=dynamics_metrics["avg_yaw_rate"],
            max_yaw_rate=dynamics_metrics["max_yaw_rate"],
            imu_acc_z_variance=dynamics_metrics["imu_acc_z_variance"],
            
            # 第二层指标
            avg_feature_points=visual_metrics["avg_feature_points"],
            min_feature_points=visual_metrics["min_feature_points"],
            dynamic_object_ratio=visual_metrics["avg_dynamic_object_ratio"],
            lane_confidence=visual_metrics["avg_lane_confidence"],
            lighting_quality=visual_metrics["lighting_quality"],
            blur_score=visual_metrics["avg_blur_score"],
            
            # 第三层指标
            vanishing_point_variance=observability_metrics["vanishing_point_variance"],
            fim_min_singular_value=observability_metrics["fim_min_singular_value"],
            has_loop_closure=observability_metrics["has_loop_closure"],
            
            scene_description=scene_description,
            data_paths=segment["data_paths"]
        )
    
    def _compute_quality_score(
        self,
        dynamics_metrics: Dict[str, float],
        visual_metrics: Dict[str, float],
        observability_metrics: Dict[str, float]
    ) -> float:
        """
        计算质量分数
        
        Args:
            dynamics_metrics: 车辆动力学指标
            visual_metrics: 视觉/感知指标
            observability_metrics: 可观测性指标
        
        Returns:
            质量分数 [0, 1]
        """
        score = 0.0
        
        # 第一层：车辆动力学（30%）
        dynamics_score = 0.0
        
        # 速度稳定性
        if 30 <= dynamics_metrics["avg_speed"] <= 80:
            dynamics_score += 0.4
        if dynamics_metrics["std_speed"] < 5.0:
            dynamics_score += 0.2
        
        # 直线行驶
        if dynamics_metrics["max_steering_angle"] < 1.0:
            dynamics_score += 0.2
        if dynamics_metrics["max_yaw_rate"] < 2.0:
            dynamics_score += 0.2
        
        score += dynamics_score * 0.3
        
        # 第二层：视觉/感知（40%）
        visual_score = 0.0
        
        # 特征点数量
        feature_score = min(visual_metrics["min_feature_points"] / 1000.0, 1.0)
        visual_score += feature_score * 0.3
        
        # 静态环境占比
        static_ratio = 1.0 - visual_metrics["avg_dynamic_object_ratio"]
        visual_score += static_ratio * 0.3
        
        # 车道线置信度
        visual_score += visual_metrics["avg_lane_confidence"] * 0.2
        
        # 图像质量
        visual_score += visual_metrics["avg_blur_score"] * 0.2
        
        score += visual_score * 0.4
        
        # 第三层：可观测性（30%）
        observability_score = 0.0
        
        # 消失点一致性
        vp_score = max(0, 1.0 - observability_metrics["vanishing_point_variance"] / 100.0)
        observability_score += vp_score * 0.5
        
        # 激励充分性
        fim_score = min(observability_metrics["fim_min_singular_value"] / 1e-2, 1.0)
        observability_score += fim_score * 0.3
        
        # 闭环检测（加分项）
        if observability_metrics["has_loop_closure"]:
            observability_score += 0.2
        
        score += observability_score * 0.3
        
        return min(score, 1.0)
    
    def _determine_clip_type(
        self,
        dynamics_metrics: Dict[str, float],
        visual_metrics: Dict[str, float],
        observability_metrics: Dict[str, float]
    ) -> CalibrationClipType:
        """
        确定片段类型
        
        Args:
            dynamics_metrics: 车辆动力学指标
            visual_metrics: 视觉/感知指标
            observability_metrics: 可观测性指标
        
        Returns:
            CalibrationClipType: 片段类型
        """
        # 闭环检测（最高优先级）
        if observability_metrics["has_loop_closure"]:
            return CalibrationClipType.LOOP_CLOSURE
        
        # 判断是否为直道
        is_straight = (
            dynamics_metrics["max_steering_angle"] < 1.0 and
            dynamics_metrics["max_yaw_rate"] < 2.0
        )
        
        # 判断是否为弯道
        is_curved = (
            dynamics_metrics["max_steering_angle"] >= 1.0 or
            dynamics_metrics["max_yaw_rate"] >= 2.0
        )
        
        # 判断纹理是否丰富
        is_rich_texture = visual_metrics["min_feature_points"] > 800
        
        if is_straight and is_rich_texture:
            return CalibrationClipType.RICH_TEXTURE
        elif is_straight:
            return CalibrationClipType.STRAIGHT_ROAD
        elif is_curved:
            return CalibrationClipType.CURVED_ROAD
        else:
            return CalibrationClipType.RICH_TEXTURE
    
    def _generate_scene_description(
        self,
        dynamics_metrics: Dict[str, float],
        visual_metrics: Dict[str, float],
        observability_metrics: Dict[str, float]
    ) -> str:
        """
        生成场景描述
        
        Args:
            dynamics_metrics: 车辆动力学指标
            visual_metrics: 视觉/感知指标
            observability_metrics: 可观测性指标
        
        Returns:
            场景描述字符串
        """
        description_parts = []
        
        # 速度
        avg_speed = dynamics_metrics["avg_speed"]
        if avg_speed < 40:
            description_parts.append("低速行驶")
        elif avg_speed < 70:
            description_parts.append("中速行驶")
        else:
            description_parts.append("高速行驶")
        
        # 路况
        if dynamics_metrics["max_steering_angle"] < 0.5:
            description_parts.append("直道")
        elif dynamics_metrics["max_steering_angle"] < 1.0:
            description_parts.append("轻微转弯")
        else:
            description_parts.append("弯道")
        
        # 环境
        if visual_metrics["lighting_quality"] == "good":
            description_parts.append("光照良好")
        else:
            description_parts.append(f"光照{visual_metrics['lighting_quality']}")
        
        # 静态环境
        if visual_metrics["avg_dynamic_object_ratio"] < 0.1:
            description_parts.append("静态环境")
        elif visual_metrics["avg_dynamic_object_ratio"] < 0.2:
            description_parts.append("少量动态物体")
        else:
            description_parts.append("动态环境")
        
        # 特征丰富度
        if visual_metrics["min_feature_points"] > 800:
            description_parts.append("特征丰富")
        elif visual_metrics["min_feature_points"] > 500:
            description_parts.append("特征适中")
        else:
            description_parts.append("特征稀疏")
        
        # 闭环
        if observability_metrics["has_loop_closure"]:
            description_parts.append("闭环路径")
        
        return "，".join(description_parts)
    
    def get_golden_clips_by_type(
        self,
        clip_type: CalibrationClipType,
        min_quality_score: float = 0.7
    ) -> List[GoldenClip]:
        """
        按类型获取黄金片段
        
        Args:
            clip_type: 片段类型
            min_quality_score: 最小质量分数
        
        Returns:
            黄金片段列表
        """
        return [
            clip for clip in self.golden_clips
            if clip.clip_type == clip_type and clip.quality_score >= min_quality_score
        ]
    
    def get_top_clips(
        self,
        top_n: int = 10,
        clip_type: Optional[CalibrationClipType] = None
    ) -> List[GoldenClip]:
        """
        获取质量最高的片段
        
        Args:
            top_n: 返回数量
            clip_type: 片段类型（可选）
        
        Returns:
            黄金片段列表
        """
        clips = self.golden_clips
        
        if clip_type is not None:
            clips = [clip for clip in clips if clip.clip_type == clip_type]
        
        # 按质量分数排序
        clips.sort(key=lambda x: x.quality_score, reverse=True)
        
        return clips[:top_n]
    
    def get_filter_statistics(self) -> Dict[str, Any]:
        """
        获取筛选统计信息
        
        Returns:
            统计信息字典
        """
        total_segments = len(self.filter_history)
        passed_segments = len([r for r in self.filter_history if r.passed])
        
        # 按层级统计
        layer_stats = {}
        for layer in FilterLayer:
            layer_results = [r for r in self.filter_history if r.layer == layer]
            layer_stats[layer.value] = {
                "total": len(layer_results),
                "passed": len([r for r in layer_results if r.passed]),
                "failed": len([r for r in layer_results if not r.passed])
            }
        
        return {
            "total_segments": total_segments,
            "passed_segments": passed_segments,
            "failed_segments": total_segments - passed_segments,
            "pass_rate": passed_segments / total_segments if total_segments > 0 else 0.0,
            "layer_statistics": layer_stats,
            "golden_clips_count": len(self.golden_clips)
        }
    
    def export_golden_clips(
        self,
        output_path: str,
        clip_type: Optional[CalibrationClipType] = None,
        min_quality_score: float = 0.7
    ):
        """
        导出黄金片段信息
        
        Args:
            output_path: 输出路径
            clip_type: 片段类型（可选）
            min_quality_score: 最小质量分数
        """
        import json
        
        clips = self.golden_clips
        
        if clip_type is not None:
            clips = [clip for clip in clips if clip.clip_type == clip_type]
        
        clips = [clip for clip in clips if clip.quality_score >= min_quality_score]
        
        # 转换为可序列化的格式
        clips_data = []
        for clip in clips:
            clip_dict = {
                "clip_id": clip.clip_id,
                "vehicle_id": clip.vehicle_id,
                "start_timestamp": clip.start_timestamp.isoformat(),
                "end_timestamp": clip.end_timestamp.isoformat(),
                "duration": clip.duration,
                "clip_type": clip.clip_type.value,
                "quality_score": clip.quality_score,
                "avg_speed": clip.avg_speed,
                "avg_steering_angle": clip.avg_steering_angle,
                "avg_feature_points": clip.avg_feature_points,
                "dynamic_object_ratio": clip.dynamic_object_ratio,
                "lane_confidence": clip.lane_confidence,
                "lighting_quality": clip.lighting_quality,
                "blur_score": clip.blur_score,
                "vanishing_point_variance": clip.vanishing_point_variance,
                "fim_min_singular_value": clip.fim_min_singular_value,
                "has_loop_closure": clip.has_loop_closure,
                "scene_description": clip.scene_description,
                "data_paths": clip.data_paths
            }
            clips_data.append(clip_dict)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(clips_data, f, indent=2, ensure_ascii=False)
        
        print(f"[OfflineCalibrationMiner] Exported {len(clips_data)} golden clips to {output_path}")