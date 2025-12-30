"""
消失点检测模块
基于消失点原理标定相机俯仰角和偏航角
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class VanishingPointResult:
    """消失点检测结果"""
    vanishing_point: Tuple[float, float]  # 消失点坐标 (u, v)
    confidence: float  # 置信度 [0, 1]
    lines_used: int  # 使用的车道线数量
    pitch_error: float  # 俯仰角误差（弧度）
    yaw_error: float  # 偏航角误差（弧度）


class VanishingPointDetector:
    """
    消失点检测器
    原理：在直道行驶时，车道线在远处的交点（消失点）应该对应相机的光心方向。
    如果消失点偏了，说明相机俯仰角或偏航角变了。
    """
    
    def __init__(self, camera_intrinsics: np.ndarray, image_size: Tuple[int, int]):
        """
        初始化消失点检测器
        
        Args:
            camera_intrinsics: 3x3相机内参矩阵
            image_size: 图像尺寸 (width, height)
        """
        self.K = camera_intrinsics
        self.image_width, self.image_height = image_size
        
        # 理论消失点（图像中心）
        self.ideal_vanishing_point = (
            camera_intrinsics[0, 2],  # cx
            camera_intrinsics[1, 2]   # cy
        )
    
    def detect_from_lane_lines(
        self,
        lane_lines: List[Tuple[float, float, float, float]]
    ) -> Optional[VanishingPointResult]:
        """
        从车道线检测消失点
        
        Args:
            lane_lines: 车道线列表，每条线为 (x1, y1, x2, y2)
        
        Returns:
            VanishingPointResult: 消失点检测结果
        """
        if len(lane_lines) < 2:
            return None
        
        # 将车道线转换为参数空间 (rho, theta)
        lines_params = []
        for x1, y1, x2, y2 in lane_lines:
            # 计算直线的极坐标参数
            dx = x2 - x1
            dy = y2 - y1
            
            if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                continue
            
            # 直线方程: ax + by + c = 0
            a = dy
            b = -dx
            c = dx * y1 - dy * x1
            
            # 归一化
            norm = np.sqrt(a**2 + b**2)
            if norm < 1e-6:
                continue
            
            a, b, c = a / norm, b / norm, c / norm
            lines_params.append((a, b, c))
        
        if len(lines_params) < 2:
            return None
        
        # 使用RANSAC拟合消失点
        vanishing_point, inliers = self._ransac_vanishing_point(lines_params)
        
        if vanishing_point is None:
            return None
        
        # 计算置信度
        confidence = len(inliers) / len(lines_params)
        
        # 计算角度误差
        pitch_error, yaw_error = self._compute_angle_errors(vanishing_point)
        
        return VanishingPointResult(
            vanishing_point=vanishing_point,
            confidence=confidence,
            lines_used=len(inliers),
            pitch_error=pitch_error,
            yaw_error=yaw_error
        )
    
    def _ransac_vanishing_point(
        self,
        lines_params: List[Tuple[float, float, float]],
        iterations: int = 100,
        threshold: float = 2.0
    ) -> Tuple[Optional[Tuple[float, float]], List[int]]:
        """
        使用RANSAC拟合消失点
        
        Args:
            lines_params: 车道线参数列表
            iterations: RANSAC迭代次数
            threshold: 内点阈值（像素）
        
        Returns:
            (消失点坐标, 内点索引列表)
        """
        best_vanishing_point = None
        best_inliers = []
        best_score = 0
        
        for _ in range(iterations):
            # 随机选择两条直线
            if len(lines_params) < 2:
                break
            
            idx1, idx2 = np.random.choice(len(lines_params), 2, replace=False)
            a1, b1, c1 = lines_params[idx1]
            a2, b2, c2 = lines_params[idx2]
            
            # 计算两条直线的交点（消失点）
            det = a1 * b2 - a2 * b1
            if abs(det) < 1e-6:
                continue
            
            x = (b1 * c2 - b2 * c1) / det
            y = (c1 * a2 - c2 * a1) / det
            
            # 检查消失点是否在图像范围内（允许一定超出）
            if not self._is_valid_vanishing_point(x, y):
                continue
            
            # 计算内点
            inliers = []
            for i, (a, b, c) in enumerate(lines_params):
                distance = abs(a * x + b * y + c)
                if distance < threshold:
                    inliers.append(i)
            
            # 更新最佳结果
            if len(inliers) > best_score:
                best_score = len(inliers)
                best_inliers = inliers
                best_vanishing_point = (x, y)
        
        return best_vanishing_point, best_inliers
    
    def _is_valid_vanishing_point(self, x: float, y: float) -> bool:
        """检查消失点是否有效"""
        # 消失点应该在图像中心附近或稍远处
        margin = 1.5  # 允许超出图像范围1.5倍
        
        valid_x = -self.image_width * margin <= x <= self.image_width * (1 + margin)
        valid_y = -self.image_height * margin <= y <= self.image_height * (1 + margin)
        
        return valid_x and valid_y
    
    def _compute_angle_errors(
        self,
        vanishing_point: Tuple[float, float]
    ) -> Tuple[float, float]:
        """
        计算角度误差
        
        Args:
            vanishing_point: 消失点坐标 (u, v)
        
        Returns:
            (pitch_error, yaw_error): 俯仰角误差和偏航角误差（弧度）
        """
        vp_u, vp_v = vanishing_point
        ideal_u, ideal_v = self.ideal_vanishing_point
        
        # 计算消失点偏差（像素）
        du = vp_u - ideal_u
        dv = vp_v - ideal_v
        
        # 转换为角度误差
        # yaw_error = du / fx
        # pitch_error = dv / fy
        fx = self.K[0, 0]
        fy = self.K[1, 1]
        
        yaw_error = np.arctan(du / fx)
        pitch_error = np.arctan(dv / fy)
        
        return pitch_error, yaw_error
    
    def compute_correction(
        self,
        vanishing_point_result: VanishingPointResult
    ) -> Tuple[float, float]:
        """
        计算标定修正量
        
        Args:
            vanishing_point_result: 消失点检测结果
        
        Returns:
            (pitch_correction, yaw_correction): 俯仰角和偏航角的修正量（弧度）
        """
        # 修正量 = -误差量
        pitch_correction = -vanishing_point_result.pitch_error
        yaw_correction = -vanishing_point_result.yaw_error
        
        return pitch_correction, yaw_correction
    
    def visualize(
        self,
        image: np.ndarray,
        vanishing_point_result: VanishingPointResult,
        lane_lines: List[Tuple[float, float, float, float]]
    ) -> np.ndarray:
        """
        可视化消失点检测结果
        
        Args:
            image: 输入图像
            vanishing_point_result: 消失点检测结果
            lane_lines: 车道线列表
        
        Returns:
            可视化图像
        """
        import cv2
        
        vis_image = image.copy()
        
        # 绘制车道线
        for x1, y1, x2, y2 in lane_lines:
            cv2.line(vis_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        # 绘制消失点
        vp_u, vp_v = vanishing_point_result.vanishing_point
        cv2.circle(vis_image, (int(vp_u), int(vp_v)), 10, (0, 0, 255), -1)
        
        # 绘制理想消失点
        ideal_u, ideal_v = self.ideal_vanishing_point
        cv2.circle(vis_image, (int(ideal_u), int(ideal_v)), 10, (255, 0, 0), 2)
        
        # 绘制连接线
        cv2.line(vis_image, (int(vp_u), int(vp_v)), (int(ideal_u), int(ideal_v)), (255, 255, 0), 2)
        
        # 添加文本信息
        text = f"Confidence: {vanishing_point_result.confidence:.2f}"
        cv2.putText(vis_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return vis_image