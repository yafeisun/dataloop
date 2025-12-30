"""
自车运动估计模块
基于车辆运动估计相机高度和角度
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
from collections import deque


@dataclass
class EgoMotionResult:
    """自车运动估计结果"""
    translation: np.ndarray  # 平移向量 (x, y, z)
    rotation: np.ndarray  # 旋转矩阵
    velocity: np.ndarray  # 速度向量 (vx, vy, vz)
    angular_velocity: np.ndarray  # 角速度向量 (wx, wy, wz)
    confidence: float  # 置信度 [0, 1]
    timestamp: float  # 时间戳


@dataclass
class FeatureTrackingResult:
    """特征跟踪结果"""
    keypoints_prev: np.ndarray  # 上一帧关键点 (N, 2)
    keypoints_curr: np.ndarray  # 当前帧关键点 (N, 2)
    optical_flow: np.ndarray  # 光流 (N, 2)
    valid_mask: np.ndarray  # 有效点掩码 (N,)


class EgoMotionEstimator:
    """
    自车运动估计器
    原理：结合轮速计和IMU，车知道自己走了多少距离。
    对比图像中特征点的移动距离，反推相机的高度和角度。
    """
    
    def __init__(
        self,
        camera_intrinsics: np.ndarray,
        imu_to_body_transform: Optional[np.ndarray] = None
    ):
        """
        初始化自车运动估计器
        
        Args:
            camera_intrinsics: 相机内参矩阵 3x3
            imu_to_body_transform: IMU到Body的变换矩阵 4x4（可选）
        """
        self.K = camera_intrinsics
        self.imu_to_body = imu_to_body_transform
        
        # 历史数据（用于滑动窗口优化）
        self.history_window = deque(maxlen=10)
        
        # 上一帧数据
        self.prev_keypoints = None
        self.prev_timestamp = None
    
    def estimate_from_visual_odometry(
        self,
        image_prev: np.ndarray,
        image_curr: np.ndarray,
        wheel_speed: Optional[float] = None,
        imu_data: Optional[Dict[str, np.ndarray]] = None
    ) -> Optional[EgoMotionResult]:
        """
        基于视觉里程计估计自车运动
        
        Args:
            image_prev: 上一帧图像
            image_curr: 当前帧图像
            wheel_speed: 轮速（可选）
            imu_data: IMU数据，包含加速度和角速度（可选）
        
        Returns:
            EgoMotionResult: 自车运动估计结果
        """
        import cv2
        
        # 特征提取
        keypoints_curr = self._extract_features(image_curr)
        
        if self.prev_keypoints is None:
            self.prev_keypoints = keypoints_curr
            self.prev_timestamp = 0.0
            return None
        
        # 特征跟踪
        tracking_result = self._track_features(
            image_prev, image_curr, self.prev_keypoints
        )
        
        if tracking_result is None:
            self.prev_keypoints = keypoints_curr
            return None
        
        # 计算本质矩阵
        R, t, inlier_ratio = self._compute_motion_from_features(
            tracking_result
        )
        
        if R is None:
            self.prev_keypoints = keypoints_curr
            return None
        
        # 估计速度
        dt = 1.0  # 假设帧间隔为1秒（实际应该传入真实时间戳）
        velocity = t / dt if dt > 0 else np.zeros(3)
        
        # 估计角速度
        angular_velocity = self._rotation_to_angular_velocity(R, dt)
        
        # 融合轮速和IMU数据
        if wheel_speed is not None:
            velocity = self._fuse_wheel_speed(velocity, wheel_speed, R)
        
        if imu_data is not None:
            angular_velocity = self._fuse_imu_data(
                angular_velocity, imu_data
            )
        
        # 计算置信度
        confidence = self._compute_confidence(inlier_ratio, tracking_result)
        
        result = EgoMotionResult(
            translation=t,
            rotation=R,
            velocity=velocity,
            angular_velocity=angular_velocity,
            confidence=confidence,
            timestamp=self.prev_timestamp + dt
        )
        
        # 更新历史数据
        self.history_window.append(result)
        
        # 更新上一帧数据
        self.prev_keypoints = keypoints_curr
        self.prev_timestamp = result.timestamp
        
        return result
    
    def _extract_features(
        self,
        image: np.ndarray,
        max_corners: int = 500,
        quality_level: float = 0.01,
        min_distance: float = 10.0
    ) -> np.ndarray:
        """
        提取图像特征点
        
        Args:
            image: 输入图像
            max_corners: 最大角点数
            quality_level: 质量等级
            min_distance: 最小距离
        
        Returns:
            特征点坐标 (N, 2)
        """
        import cv2
        
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 使用Shi-Tomasi算法提取角点
        keypoints = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=max_corners,
            qualityLevel=quality_level,
            minDistance=min_distance,
            blockSize=3
        )
        
        if keypoints is None:
            return np.array([])
        
        return keypoints.reshape(-1, 2)
    
    def _track_features(
        self,
        image_prev: np.ndarray,
        image_curr: np.ndarray,
        keypoints_prev: np.ndarray
    ) -> Optional[FeatureTrackingResult]:
        """
        跟踪特征点
        
        Args:
            image_prev: 上一帧图像
            image_curr: 当前帧图像
            keypoints_prev: 上一帧特征点 (N, 2)
        
        Returns:
            FeatureTrackingResult: 特征跟踪结果
        """
        import cv2
        
        if len(keypoints_prev) < 8:
            return None
        
        # 转换为灰度图
        if len(image_prev.shape) == 3:
            gray_prev = cv2.cvtColor(image_prev, cv2.COLOR_BGR2GRAY)
            gray_curr = cv2.cvtColor(image_curr, cv2.COLOR_BGR2GRAY)
        else:
            gray_prev = image_prev
            gray_curr = image_curr
        
        # 使用Lucas-Kanade光流法跟踪特征点
        keypoints_curr, status, error = cv2.calcOpticalFlowPyrLK(
            gray_prev, gray_curr,
            keypoints_prev.astype(np.float32),
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        # 过滤无效点
        valid_mask = (status.flatten() == 1) & (error.flatten() < 10.0)
        
        if np.sum(valid_mask) < 8:
            return None
        
        keypoints_prev_valid = keypoints_prev[valid_mask]
        keypoints_curr_valid = keypoints_curr[valid_mask]
        
        # 计算光流
        optical_flow = keypoints_curr_valid - keypoints_prev_valid
        
        return FeatureTrackingResult(
            keypoints_prev=keypoints_prev_valid,
            keypoints_curr=keypoints_curr_valid,
            optical_flow=optical_flow,
            valid_mask=valid_mask
        )
    
    def _compute_motion_from_features(
        self,
        tracking_result: FeatureTrackingResult
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        """
        从特征点计算运动
        
        Args:
            tracking_result: 特征跟踪结果
        
        Returns:
            (旋转矩阵, 平移向量, 内点比例)
        """
        import cv2
        
        # 计算本质矩阵
        E, mask = cv2.findEssentialMat(
            tracking_result.keypoints_prev,
            tracking_result.keypoints_curr,
            self.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )
        
        if E is None:
            return None, None, 0.0
        
        # 分解本质矩阵
        _, R, t, mask = cv2.recoverPose(
            E,
            tracking_result.keypoints_prev,
            tracking_result.keypoints_curr,
            self.K
        )
        
        # 归一化平移向量（假设平移尺度未知）
        if np.linalg.norm(t) > 1e-6:
            t = t / np.linalg.norm(t)
        
        # 计算内点比例
        inlier_ratio = np.sum(mask.astype(bool)) / len(tracking_result.keypoints_prev)
        
        return R, t, inlier_ratio
    
    def _rotation_to_angular_velocity(
        self,
        R: np.ndarray,
        dt: float
    ) -> np.ndarray:
        """
        从旋转矩阵计算角速度
        
        Args:
            R: 旋转矩阵
            dt: 时间间隔
        
        Returns:
            角速度向量 (wx, wy, wz)
        """
        # 使用Rodrigues公式将旋转矩阵转换为旋转向量
        import cv2
        
        rotation_vector, _ = cv2.Rodrigues(R)
        
        # 角速度 = 旋转向量 / 时间间隔
        if dt > 0:
            angular_velocity = rotation_vector.flatten() / dt
        else:
            angular_velocity = np.zeros(3)
        
        return angular_velocity
    
    def _fuse_wheel_speed(
        self,
        velocity: np.ndarray,
        wheel_speed: float,
        R: np.ndarray
    ) -> np.ndarray:
        """
        融合轮速数据
        
        Args:
            velocity: 视觉估计的速度
            wheel_speed: 轮速
            R: 旋转矩阵
        
        Returns:
            融合后的速度
        """
        # 假设车辆主要沿前进方向运动
        forward_direction = R @ np.array([1, 0, 0])
        
        # 使用轮速修正速度大小
        visual_speed = np.linalg.norm(velocity)
        if visual_speed > 1e-6:
            scale_factor = wheel_speed / visual_speed
            velocity = velocity * scale_factor
        else:
            velocity = forward_direction * wheel_speed
        
        return velocity
    
    def _fuse_imu_data(
        self,
        angular_velocity: np.ndarray,
        imu_data: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        融合IMU数据
        
        Args:
            angular_velocity: 视觉估计的角速度
            imu_data: IMU数据，包含angular_velocity
        
        Returns:
            融合后的角速度
        """
        if "angular_velocity" not in imu_data:
            return angular_velocity
        
        imu_angular_velocity = imu_data["angular_velocity"]
        
        # 简单加权平均
        fused_angular_velocity = 0.7 * angular_velocity + 0.3 * imu_angular_velocity
        
        return fused_angular_velocity
    
    def _compute_confidence(
        self,
        inlier_ratio: float,
        tracking_result: FeatureTrackingResult
    ) -> float:
        """
        计算置信度
        
        Args:
            inlier_ratio: 内点比例
            tracking_result: 特征跟踪结果
        
        Returns:
            置信度 [0, 1]
        """
        # 内点比例权重
        inlier_score = inlier_ratio
        
        # 特征点数量权重
        num_features = len(tracking_result.keypoints_prev)
        feature_score = min(num_features / 100.0, 1.0)
        
        # 光流一致性权重
        flow_magnitude = np.linalg.norm(tracking_result.optical_flow, axis=1)
        flow_std = np.std(flow_magnitude) if len(flow_magnitude) > 0 else 0
        flow_score = np.exp(-flow_std / 10.0)
        
        # 综合置信度
        confidence = 0.5 * inlier_score + 0.3 * feature_score + 0.2 * flow_score
        
        return np.clip(confidence, 0.0, 1.0)
    
    def estimate_camera_height(
        self,
        ego_motion: EgoMotionResult,
        ground_plane_points: Optional[np.ndarray] = None
    ) -> Optional[float]:
        """
        估计相机高度
        
        Args:
            ego_motion: 自车运动估计结果
            ground_plane_points: 地面点（可选）
        
        Returns:
            相机高度（米）
        """
        if ground_plane_points is not None:
            # 基于地面点估计高度
            return self._estimate_height_from_ground(ground_plane_points)
        else:
            # 基于运动估计高度
            return self._estimate_height_from_motion(ego_motion)
    
    def _estimate_height_from_ground(
        self,
        ground_plane_points: np.ndarray
    ) -> float:
        """
        基于地面点估计相机高度
        
        Args:
            ground_plane_points: 地面点 (N, 3) in camera coordinates
        
        Returns:
            相机高度（米）
        """
        # 地面点应该在z轴方向（假设相机坐标系z轴向下）
        heights = ground_plane_points[:, 2]
        
        # 使用中位数作为高度估计
        camera_height = np.median(heights)
        
        return abs(camera_height)
    
    def _estimate_height_from_motion(
        self,
        ego_motion: EgoMotionResult
    ) -> Optional[float]:
        """
        基于运动估计相机高度
        
        Args:
            ego_motion: 自车运动估计结果
        
        Returns:
            相机高度（米）
        """
        # 基于历史数据估计高度
        if len(self.history_window) < 2:
            return None
        
        # 计算累积位移
        total_displacement = np.zeros(3)
        for motion in list(self.history_window):
            total_displacement += motion.translation
        
        # 假设车辆在水平面上运动，高度变化应该很小
        height_variation = np.abs(total_displacement[2])
        
        # 如果高度变化过大，说明标定可能有问题
        if height_variation > 0.5:  # 50cm
            return None
        
        # 返回平均高度（需要先验知识）
        return 1.5  # 假设典型相机高度为1.5米
    
    def reset(self):
        """重置估计器状态"""
        self.history_window.clear()
        self.prev_keypoints = None
        self.prev_timestamp = None