"""
数据质量检测模块
实现自动质量检测和隐私脱敏
"""

from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field
from enum import Enum
import cv2
import numpy as np


class QualityIssueType(str, Enum):
    """质量问题类型"""
    BLUR = "blur"                     # 模糊
    OVEREXPOSED = "overexposed"       # 过曝
    UNDEREXPOSED = "underexposed"     # 过暗
    OCCLUSION = "occlusion"           # 遮挡
    NOISE = "noise"                   # 噪声
    DISTORTION = "distortion"         # 畸变
    SENSOR_DIRT = "sensor_dirt"       # 传感器脏污
    MISSING_DATA = "missing_data"     # 数据缺失
    CORRUPTION = "corruption"         # 数据损坏


class QualityLevel(str, Enum):
    """质量等级"""
    EXCELLENT = "excellent"  # 优秀
    GOOD = "good"            # 良好
    ACCEPTABLE = "acceptable"  # 可接受
    POOR = "poor"            # 差
    UNUSABLE = "unusable"    # 不可用


class QualityCheckResult(BaseModel):
    """质量检测结果"""
    data_id: str
    overall_quality: QualityLevel
    issues: List[Dict[str, Any]] = Field(default_factory=list)
    scores: Dict[str, float] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QualityChecker:
    """
    质量检测器
    自动检测数据质量问题
    """

    def __init__(self):
        self.thresholds = {
            "blur": 100.0,          # 拉普拉斯方差阈值
            "brightness_min": 30.0,  # 最小亮度
            "brightness_max": 220.0, # 最大亮度
            "noise": 50.0,          # 噪声阈值
            "occlusion_ratio": 0.3, # 遮挡比例阈值
        }

    def check_image_quality(
        self,
        image: np.ndarray,
        data_id: str
    ) -> QualityCheckResult:
        """
        检测图像质量

        Args:
            image: 图像数据（BGR格式）
            data_id: 数据ID

        Returns:
            QualityCheckResult: 质量检测结果
        """
        issues = []
        scores = {}

        # 检测模糊
        blur_score, blur_issue = self._check_blur(image)
        scores["blur"] = blur_score
        if blur_issue:
            issues.append(blur_issue)

        # 检测亮度
        brightness_score, brightness_issue = self._check_brightness(image)
        scores["brightness"] = brightness_score
        if brightness_issue:
            issues.append(brightness_issue)

        # 检测噪声
        noise_score, noise_issue = self._check_noise(image)
        scores["noise"] = noise_score
        if noise_issue:
            issues.append(noise_issue)

        # 检测遮挡
        occlusion_score, occlusion_issue = self._check_occlusion(image)
        scores["occlusion"] = occlusion_score
        if occlusion_issue:
            issues.append(occlusion_issue)

        # 计算总体质量
        overall_quality = self._calculate_overall_quality(scores)

        return QualityCheckResult(
            data_id=data_id,
            overall_quality=overall_quality,
            issues=issues,
            scores=scores
        )

    def check_sensor_data_quality(
        self,
        sensor_data: Dict[str, Any],
        data_id: str
    ) -> QualityCheckResult:
        """
        检测传感器数据质量

        Args:
            sensor_data: 传感器数据
            data_id: 数据ID

        Returns:
            QualityCheckResult: 质量检测结果
        """
        issues = []
        scores = {}

        # 检查数据完整性
        completeness_score, completeness_issue = self._check_completeness(sensor_data)
        scores["completeness"] = completeness_score
        if completeness_issue:
            issues.append(completeness_issue)

        # 检查数据一致性
        consistency_score, consistency_issue = self._check_consistency(sensor_data)
        scores["consistency"] = consistency_score
        if consistency_issue:
            issues.append(consistency_issue)

        # 检查数据范围
        range_score, range_issue = self._check_data_range(sensor_data)
        scores["range"] = range_score
        if range_issue:
            issues.append(range_issue)

        # 计算总体质量
        overall_quality = self._calculate_overall_quality(scores)

        return QualityCheckResult(
            data_id=data_id,
            overall_quality=overall_quality,
            issues=issues,
            scores=scores
        )

    def _check_blur(self, image: np.ndarray) -> Tuple[float, Optional[Dict[str, Any]]]:
        """
        检测模糊

        Args:
            image: 图像数据

        Returns:
            Tuple[float, Optional[Dict]]: (分数, 问题)
        """
        # 转灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 计算拉普拉斯方差
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # 分数归一化（0-100）
        score = min(100.0, laplacian_var / 10.0)

        # 判断是否模糊
        if laplacian_var < self.thresholds["blur"]:
            issue = {
                "type": QualityIssueType.BLUR.value,
                "severity": "high" if laplacian_var < self.thresholds["blur"] * 0.5 else "medium",
                "value": laplacian_var,
                "threshold": self.thresholds["blur"],
                "description": f"图像模糊，拉普拉斯方差={laplacian_var:.2f}"
            }
            return score, issue

        return score, None

    def _check_brightness(self, image: np.ndarray) -> Tuple[float, Optional[Dict[str, Any]]]:
        """
        检测亮度

        Args:
            image: 图像数据

        Returns:
            Tuple[float, Optional[Dict]]: (分数, 问题)
        """
        # 转灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 计算平均亮度
        avg_brightness = np.mean(gray)

        # 计算分数
        if avg_brightness < self.thresholds["brightness_min"]:
            score = (avg_brightness / self.thresholds["brightness_min"]) * 100
        elif avg_brightness > self.thresholds["brightness_max"]:
            score = ((255 - avg_brightness) / (255 - self.thresholds["brightness_max"])) * 100
        else:
            score = 100.0

        # 判断是否过曝或过暗
        if avg_brightness < self.thresholds["brightness_min"]:
            issue = {
                "type": QualityIssueType.UNDEREXPOSED.value,
                "severity": "high" if avg_brightness < self.thresholds["brightness_min"] * 0.7 else "medium",
                "value": avg_brightness,
                "threshold": self.thresholds["brightness_min"],
                "description": f"图像过暗，平均亮度={avg_brightness:.2f}"
            }
            return score, issue
        elif avg_brightness > self.thresholds["brightness_max"]:
            issue = {
                "type": QualityIssueType.OVEREXPOSED.value,
                "severity": "high" if avg_brightness > self.thresholds["brightness_max"] * 1.3 else "medium",
                "value": avg_brightness,
                "threshold": self.thresholds["brightness_max"],
                "description": f"图像过曝，平均亮度={avg_brightness:.2f}"
            }
            return score, issue

        return score, None

    def _check_noise(self, image: np.ndarray) -> Tuple[float, Optional[Dict[str, Any]]]:
        """
        检测噪声

        Args:
            image: 图像数据

        Returns:
            Tuple[float, Optional[Dict]]: (分数, 问题)
        """
        # 转灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 使用高斯滤波计算噪声
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = gray.astype(np.float32) - blurred.astype(np.float32)
        noise_level = np.std(noise)

        # 分数归一化（0-100）
        score = max(0.0, 100.0 - noise_level)

        # 判断是否有噪声
        if noise_level > self.thresholds["noise"]:
            issue = {
                "type": QualityIssueType.NOISE.value,
                "severity": "medium" if noise_level < self.thresholds["noise"] * 1.5 else "high",
                "value": noise_level,
                "threshold": self.thresholds["noise"],
                "description": f"图像存在噪声，噪声水平={noise_level:.2f}"
            }
            return score, issue

        return score, None

    def _check_occlusion(self, image: np.ndarray) -> Tuple[float, Optional[Dict[str, Any]]]:
        """
        检测遮挡（简化实现）

        Args:
            image: 图像数据

        Returns:
            Tuple[float, Optional[Dict]]: (分数, 问题)
        """
        # 简化实现：检测图像边缘密度
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (image.shape[0] * image.shape[1])

        # 分数（边缘密度适中为好）
        score = 100.0 - abs(edge_density - 0.1) * 500
        score = max(0.0, min(100.0, score))

        # 判断是否有遮挡（基于边缘密度异常）
        if edge_density < 0.02 or edge_density > 0.3:
            issue = {
                "type": QualityIssueType.OCCLUSION.value,
                "severity": "medium",
                "value": edge_density,
                "description": f"图像可能存在遮挡，边缘密度={edge_density:.4f}"
            }
            return score, issue

        return score, None

    def _check_completeness(self, sensor_data: Dict[str, Any]) -> Tuple[float, Optional[Dict[str, Any]]]:
        """
        检查数据完整性

        Args:
            sensor_data: 传感器数据

        Returns:
            Tuple[float, Optional[Dict]]: (分数, 问题)
        """
        required_fields = ["timestamp", "location", "velocity"]
        missing_fields = []

        for field in required_fields:
            if field not in sensor_data or sensor_data[field] is None:
                missing_fields.append(field)

        # 计算分数
        score = ((len(required_fields) - len(missing_fields)) / len(required_fields)) * 100

        # 判断是否有缺失
        if missing_fields:
            issue = {
                "type": QualityIssueType.MISSING_DATA.value,
                "severity": "high",
                "missing_fields": missing_fields,
                "description": f"数据缺失，缺失字段：{missing_fields}"
            }
            return score, issue

        return score, None

    def _check_consistency(self, sensor_data: Dict[str, Any]) -> Tuple[float, Optional[Dict[str, Any]]]:
        """
        检查数据一致性

        Args:
            sensor_data: 传感器数据

        Returns:
            Tuple[float, Optional[Dict]]: (分数, 问题)
        """
        issues = []

        # 检查时间戳一致性
        if "timestamp" in sensor_data:
            timestamp = sensor_data["timestamp"]
            if not isinstance(timestamp, (int, float)) or timestamp <= 0:
                issues.append("timestamp无效")

        # 检查位置一致性
        if "location" in sensor_data:
            location = sensor_data["location"]
            if not isinstance(location, dict):
                issues.append("location格式错误")
            else:
                lat = location.get("lat")
                lon = location.get("lon")
                if lat is None or lon is None:
                    issues.append("经纬度缺失")
                elif not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                    issues.append("经纬度范围错误")

        # 计算分数
        score = 100.0 if not issues else 50.0

        # 判断是否有不一致
        if issues:
            issue = {
                "type": QualityIssueType.CORRUPTION.value,
                "severity": "high",
                "issues": issues,
                "description": f"数据不一致：{issues}"
            }
            return score, issue

        return score, None

    def _check_data_range(self, sensor_data: Dict[str, Any]) -> Tuple[float, Optional[Dict[str, Any]]]:
        """
        检查数据范围

        Args:
            sensor_data: 传感器数据

        Returns:
            Tuple[float, Optional[Dict]]: (分数, 问题)
        """
        issues = []

        # 检查速度范围
        if "velocity" in sensor_data:
            velocity = sensor_data["velocity"]
            if not isinstance(velocity, (int, float)) or velocity < 0 or velocity > 200:
                issues.append(f"velocity范围异常：{velocity}")

        # 检查加速度范围
        if "acceleration" in sensor_data:
            acceleration = sensor_data["acceleration"]
            if not isinstance(acceleration, (int, float)) or abs(acceleration) > 20:
                issues.append(f"acceleration范围异常：{acceleration}")

        # 计算分数
        score = 100.0 if not issues else 50.0

        # 判断是否有范围异常
        if issues:
            issue = {
                "type": QualityIssueType.CORRUPTION.value,
                "severity": "medium",
                "issues": issues,
                "description": f"数据范围异常：{issues}"
            }
            return score, issue

        return score, None

    def _calculate_overall_quality(self, scores: Dict[str, float]) -> QualityLevel:
        """
        计算总体质量

        Args:
            scores: 各项分数

        Returns:
            QualityLevel: 质量等级
        """
        if not scores:
            return QualityLevel.UNUSABLE

        avg_score = sum(scores.values()) / len(scores)

        if avg_score >= 80:
            return QualityLevel.EXCELLENT
        elif avg_score >= 60:
            return QualityLevel.GOOD
        elif avg_score >= 40:
            return QualityLevel.ACCEPTABLE
        elif avg_score >= 20:
            return QualityLevel.POOR
        else:
            return QualityLevel.UNUSABLE

    def set_threshold(self, key: str, value: float):
        """
        设置阈值

        Args:
            key: 阈值键
            value: 阈值
        """
        self.thresholds[key] = value

    def get_thresholds(self) -> Dict[str, float]:
        """获取所有阈值"""
        return self.thresholds.copy()