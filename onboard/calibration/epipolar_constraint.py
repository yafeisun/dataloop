"""
极线约束模块
基于极线几何标定相邻相机的相对位置
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass


@dataclass
class EpipolarResult:
    """极线约束检测结果"""
    fundamental_matrix: np.ndarray  # 基础矩阵
    essential_matrix: np.ndarray  # 本质矩阵
    rotation: np.ndarray  # 旋转矩阵
    translation: np.ndarray  # 平移向量
    inlier_ratio: float  # 内点比例
    reprojection_error: float  # 重投影误差（像素）
    confidence: float  # 置信度 [0, 1]


class EpipolarGeometry:
    """
    极线几何计算
    原理：同一个物体在两个摄像头中的投影关系必须满足"基础矩阵"。
    如果不满足，说明相对位置变了。
    """
    
    def __init__(
        self,
        camera1_intrinsics: np.ndarray,
        camera2_intrinsics: np.ndarray
    ):
        """
        初始化极线几何计算器
        
        Args:
            camera1_intrinsics: 相机1的内参矩阵 3x3
            camera2_intrinsics: 相机2的内参矩阵 3x3
        """
        self.K1 = camera1_intrinsics
        self.K2 = camera2_intrinsics
    
    def compute_epipolar_geometry(
        self,
        keypoints1: np.ndarray,
        keypoints2: np.ndarray,
        method: str = "RANSAC"
    ) -> Optional[EpipolarResult]:
        """
        计算极线几何
        
        Args:
            keypoints1: 相机1的关键点 (N, 2)
            keypoints2: 相机2的关键点 (N, 2)
            method: 计算方法 ("RANSAC" 或 "8POINT")
        
        Returns:
            EpipolarResult: 极线几何结果
        """
        if len(keypoints1) != len(keypoints2) or len(keypoints1) < 8:
            return None
        
        # 计算基础矩阵
        F, mask = self._compute_fundamental_matrix(
            keypoints1, keypoints2, method
        )
        
        if F is None:
            return None
        
        # 计算本质矩阵
        E = self.K2.T @ F @ self.K1
        
        # 从本质矩阵分解出旋转和平移
        R, t, inlier_ratio = self._decompose_essential_matrix(
            E, keypoints1, keypoints2, mask
        )
        
        # 计算重投影误差
        reprojection_error = self._compute_reprojection_error(
            F, keypoints1, keypoints2, mask
        )
        
        # 计算置信度
        confidence = self._compute_confidence(inlier_ratio, reprojection_error)
        
        return EpipolarResult(
            fundamental_matrix=F,
            essential_matrix=E,
            rotation=R,
            translation=t,
            inlier_ratio=inlier_ratio,
            reprojection_error=reprojection_error,
            confidence=confidence
        )
    
    def _compute_fundamental_matrix(
        self,
        keypoints1: np.ndarray,
        keypoints2: np.ndarray,
        method: str = "RANSAC"
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        计算基础矩阵
        
        Args:
            keypoints1: 相机1的关键点 (N, 2)
            keypoints2: 相机2的关键点 (N, 2)
            method: 计算方法
        
        Returns:
            (基础矩阵, 内点掩码)
        """
        try:
            import cv2
            
            if method == "RANSAC":
                F, mask = cv2.findFundamentalMat(
                    keypoints1,
                    keypoints2,
                    cv2.FM_RANSAC,
                    ransacReprojThreshold=3.0,
                    confidence=0.99
                )
            else:  # 8POINT
                F, mask = cv2.findFundamentalMat(
                    keypoints1,
                    keypoints2,
                    cv2.FM_8POINT
                )
            
            if F is None or np.linalg.norm(F) < 1e-6:
                return None, None
            
            # 归一化基础矩阵（最后一个元素为1）
            F = F / F[2, 2]
            
            return F, mask.astype(bool).flatten()
        
        except Exception as e:
            print(f"Error computing fundamental matrix: {e}")
            return None, None
    
    def _decompose_essential_matrix(
        self,
        E: np.ndarray,
        keypoints1: np.ndarray,
        keypoints2: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        从本质矩阵分解出旋转和平移
        
        Args:
            E: 本质矩阵 3x3
            keypoints1: 相机1的关键点 (N, 2)
            keypoints2: 相机2的关键点 (N, 2)
            mask: 内点掩码
        
        Returns:
            (旋转矩阵, 平移向量, 内点比例)
        """
        try:
            import cv2
            
            # SVD分解
            _, R1, R2, t = cv2.decomposeEssentialMat(E)
            
            # 尝试四种可能的组合，选择使最多点在相机前方的组合
            best_R = None
            best_t = None
            best_inliers = 0
            
            for R in [R1, R2]:
                for t_sign in [t, -t]:
                    # 三角化点
                    points_4d = cv2.triangulatePoints(
                        np.eye(3), np.hstack((R, t_sign)),
                        self.K1 @ np.eye(3), self.K2 @ np.hstack((R, t_sign)),
                        keypoints1.T, keypoints2.T
                    )
                    
                    # 转换为齐次坐标
                    points_3d = points_4d[:3] / points_4d[3]
                    
                    # 检查点是否在相机前方
                    points_3d_cam1 = points_3d
                    points_3d_cam2 = R @ points_3d + t_sign.reshape(-1, 1)
                    
                    inliers_cam1 = points_3d_cam1[2, :] > 0
                    inliers_cam2 = points_3d_cam2[2, :] > 0
                    inliers = inliers_cam1 & inliers_cam2
                    
                    if mask is not None:
                        inliers = inliers & mask
                    
                    inlier_count = np.sum(inliers)
                    
                    if inlier_count > best_inliers:
                        best_inliers = inlier_count
                        best_R = R
                        best_t = t_sign
            
            # 归一化平移向量
            if np.linalg.norm(best_t) > 1e-6:
                best_t = best_t / np.linalg.norm(best_t)
            
            inlier_ratio = best_inliers / len(keypoints1)
            
            return best_R, best_t, inlier_ratio
        
        except Exception as e:
            print(f"Error decomposing essential matrix: {e}")
            return np.eye(3), np.zeros((3, 1)), 0.0
    
    def _compute_reprojection_error(
        self,
        F: np.ndarray,
        keypoints1: np.ndarray,
        keypoints2: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> float:
        """
        计算重投影误差
        
        Args:
            F: 基础矩阵
            keypoints1: 相机1的关键点 (N, 2)
            keypoints2: 相机2的关键点 (N, 2)
            mask: 内点掩码
        
        Returns:
            平均重投影误差（像素）
        """
        if mask is not None:
            keypoints1 = keypoints1[mask]
            keypoints2 = keypoints2[mask]
        
        # 转换为齐次坐标
        points1_hom = np.column_stack([keypoints1, np.ones(len(keypoints1))])
        points2_hom = np.column_stack([keypoints2, np.ones(len(keypoints2))])
        
        # 计算极线
        epilines1 = F.T @ points2_hom.T  # 图像1上的极线
        epilines2 = F @ points1_hom.T    # 图像2上的极线
        
        # 计算点到极线的距离
        errors1 = np.abs(
            epilines1[0] * keypoints1[:, 0] +
            epilines1[1] * keypoints1[:, 1] +
            epilines1[2]
        ) / np.sqrt(epilines1[0]**2 + epilines1[1]**2 + 1e-6)
        
        errors2 = np.abs(
            epilines2[0] * keypoints2[:, 0] +
            epilines2[1] * keypoints2[:, 1] +
            epilines2[2]
        ) / np.sqrt(epilines2[0]**2 + epilines2[1]**2 + 1e-6)
        
        # 对称重投影误差
        symmetric_error = np.mean(errors1 + errors2)
        
        return symmetric_error
    
    def _compute_confidence(
        self,
        inlier_ratio: float,
        reprojection_error: float
    ) -> float:
        """
        计算置信度
        
        Args:
            inlier_ratio: 内点比例
            reprojection_error: 重投影误差
        
        Returns:
            置信度 [0, 1]
        """
        # 内点比例权重
        inlier_score = inlier_ratio
        
        # 重投影误差权重（误差越小，置信度越高）
        error_score = np.exp(-reprojection_error / 2.0)
        
        # 综合置信度
        confidence = 0.6 * inlier_score + 0.4 * error_score
        
        return np.clip(confidence, 0.0, 1.0)
    
    def compute_relative_pose_correction(
        self,
        epipolar_result: EpipolarResult,
        expected_rotation: Optional[np.ndarray] = None,
        expected_translation: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算相对位姿修正量
        
        Args:
            epipolar_result: 极线几何结果
            expected_rotation: 期望的旋转矩阵（可选）
            expected_translation: 期望的平移向量（可选）
        
        Returns:
            (旋转修正, 平移修正)
        """
        R = epipolar_result.rotation
        t = epipolar_result.translation
        
        if expected_rotation is not None:
            # 旋转修正 = 期望旋转 - 实际旋转
            R_correction = expected_rotation @ R.T
        else:
            R_correction = np.eye(3)
        
        if expected_translation is not None:
            # 平移修正 = 期望平移 - 实际平移
            t_correction = expected_translation - t
        else:
            t_correction = np.zeros((3, 1))
        
        return R_correction, t_correction
    
    def visualize_epipolar_lines(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        keypoints1: np.ndarray,
        keypoints2: np.ndarray,
        F: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        可视化极线
        
        Args:
            image1: 相机1图像
            image2: 相机2图像
            keypoints1: 相机1的关键点 (N, 2)
            keypoints2: 相机2的关键点 (N, 2)
            F: 基础矩阵
            mask: 内点掩码
        
        Returns:
            (可视化图像1, 可视化图像2)
        """
        import cv2
        
        vis1 = image1.copy()
        vis2 = image2.copy()
        
        # 绘制关键点
        for i, (x, y) in enumerate(keypoints1):
            color = (0, 255, 0) if mask is None or mask[i] else (0, 0, 255)
            cv2.circle(vis1, (int(x), int(y)), 5, color, -1)
        
        for i, (x, y) in enumerate(keypoints2):
            color = (0, 255, 0) if mask is None or mask[i] else (0, 0, 255)
            cv2.circle(vis2, (int(x), int(y)), 5, color, -1)
        
        # 绘制极线
        if mask is not None:
            keypoints1 = keypoints1[mask]
            keypoints2 = keypoints2[mask]
        
        # 在图像2上绘制图像1关键点的极线
        points1_hom = np.column_stack([keypoints1, np.ones(len(keypoints1))])
        epilines2 = F @ points1_hom.T
        
        h, w = image2.shape[:2]
        for i in range(len(keypoints1)):
            a, b, c = epilines2[:, i]
            
            # 计算极线与图像边界的交点
            x0, y0 = 0, int(-c / b) if abs(b) > 1e-6 else 0
            x1, y1 = w, int(-(a * w + c) / b) if abs(b) > 1e-6 else 0
            
            cv2.line(vis2, (x0, y0), (x1, y1), (255, 0, 0), 1)
        
        # 在图像1上绘制图像2关键点的极线
        points2_hom = np.column_stack([keypoints2, np.ones(len(keypoints2))])
        epilines1 = F.T @ points2_hom.T
        
        for i in range(len(keypoints2)):
            a, b, c = epilines1[:, i]
            
            x0, y0 = 0, int(-c / b) if abs(b) > 1e-6 else 0
            x1, y1 = w, int(-(a * w + c) / b) if abs(b) > 1e-6 else 0
            
            cv2.line(vis1, (x0, y0), (x1, y1), (255, 0, 0), 1)
        
        return vis1, vis2