"""
虚拟相机与去畸变模块
将原始图像转换为标准虚拟相机视角
"""

import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass

from common.models.calibration import CameraIntrinsics, CameraExtrinsics


@dataclass
class VirtualCameraConfig:
    """虚拟相机配置"""
    image_size: Tuple[int, int]  # 图像尺寸 (width, height)
    fov: float  # 视场角（度）
    distortion_free: bool = True  # 是否无畸变


class VirtualCamera:
    """
    虚拟相机
    将原始图像转换为标准虚拟相机视角
    
    原理：
    原始图像 → 去畸变 + 旋转平移 → 虚拟相机图像（标准View）
    
    模型看到的永远是一个"完美的、标准的"视角。
    标定的本质：就是实时计算从"原始扭曲图像"到"标准虚拟相机"的那个变换矩阵。
    """
    
    def __init__(
        self,
        learned_intrinsics: CameraIntrinsics,
        learned_extrinsics: CameraExtrinsics,
        virtual_config: VirtualCameraConfig
    ):
        """
        初始化虚拟相机
        
        Args:
            learned_intrinsics: Learn的相机内参
            learned_extrinsics: Learn的相机外参
            virtual_config: 虚拟相机配置
        """
        self.learned_intrinsics = learned_intrinsics
        self.learned_extrinsics = learned_extrinsics
        self.virtual_config = virtual_config
        
        # 计算变换矩阵
        self._compute_transform_matrices()
    
    def _compute_transform_matrices(self):
        """计算变换矩阵"""
        # 获取Learned内参矩阵
        K_learned = self.learned_intrinsics.to_matrix()
        
        # 获取Learned外参变换矩阵
        T_learned = self.learned_extrinsics.to_transform_matrix()
        
        # 计算虚拟相机内参（标准内参）
        self.K_virtual = self._compute_virtual_intrinsics()
        
        # 计算从Learned到Virtual的变换
        # 假设虚拟相机的外参为标准值（相对于Body）
        T_virtual = np.eye(4)  # 虚拟相机在Body坐标系下的标准位置
        
        # 变换链：Learned → Body → Virtual
        # T_virtual_from_learned = T_virtual @ inv(T_learned)
        self.T_virtual_from_learned = T_virtual @ np.linalg.inv(T_learned)
        
        # 计算单应性矩阵（H = K_virtual @ R @ inv(K_learned) + ...）
        self.H = self._compute_homography_matrix()
    
    def _compute_virtual_intrinsics(self) -> np.ndarray:
        """
        计算虚拟相机内参
        
        Returns:
            3x3内参矩阵
        """
        width, height = self.virtual_config.image_size
        fov_rad = np.deg2rad(self.virtual_config.fov)
        
        # 计算焦距
        fx = width / (2 * np.tan(fov_rad / 2))
        fy = fx  # 假设像素为正方形
        
        # 主点在图像中心
        cx = width / 2.0
        cy = height / 2.0
        
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        
        return K
    
    def _compute_homography_matrix(self) -> np.ndarray:
        """
        计算单应性矩阵
        
        Returns:
            3x3单应性矩阵
        """
        # 获取旋转矩阵和平移向量
        R = self.T_virtual_from_learned[:3, :3]
        t = self.T_virtual_from_learned[:3, 3]
        
        # 获取内参矩阵
        K_learned = self.learned_intrinsics.to_matrix()
        
        # 无畸变情况下的单应性矩阵
        # H = K_virtual @ R @ inv(K_learned)
        H = self.K_virtual @ R @ np.linalg.inv(K_learned)
        
        return H
    
    def rectify(self, image: np.ndarray) -> np.ndarray:
        """
        去畸变和校正图像
        
        Args:
            image: 原始图像
        
        Returns:
            校正后的图像
        """
        import cv2
        
        # 获取图像尺寸
        h, w = image.shape[:2]
        
        # 1. 去畸变
        K_learned = self.learned_intrinsics.to_matrix()
        D = self.learned_intrinsics.to_distortion_coeffs()
        
        # 计算去畸变映射
        map1, map2 = cv2.initUndistortRectifyMap(
            K_learned, D, None, K_learned, (w, h), cv2.CV_32FC1
        )
        
        # 应用去畸变
        undistorted = cv2.remap(image, map1, map2, cv2.INTER_LINEAR)
        
        # 2. 旋转和平移（使用单应性变换）
        virtual_width, virtual_height = self.virtual_config.image_size
        
        # 使用单应性矩阵进行透视变换
        rectified = cv2.warpPerspective(
            undistorted,
            self.H,
            (virtual_width, virtual_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        
        return rectified
    
    def rectify_batch(self, images: np.ndarray) -> np.ndarray:
        """
        批量去畸变和校正图像
        
        Args:
            images: 原始图像批次 (N, H, W, C)
        
        Returns:
            校正后的图像批次 (N, H', W', C)
        """
        rectified_images = []
        
        for image in images:
            rectified = self.rectify(image)
            rectified_images.append(rectified)
        
        return np.stack(rectified_images, axis=0)
    
    def project_3d_to_2d(
        self,
        points_3d: np.ndarray,
        frame: str = "virtual"
    ) -> np.ndarray:
        """
        将3D点投影到2D图像平面
        
        Args:
            points_3d: 3D点 (N, 3) in Body coordinates
            frame: 坐标系 ("virtual" 或 "learned")
        
        Returns:
            2D点 (N, 2)
        """
        # 转换为齐次坐标
        points_3d_hom = np.column_stack([points_3d, np.ones(len(points_3d))])
        
        if frame == "virtual":
            # Virtual坐标系
            K = self.K_virtual
            T = np.eye(4)  # 虚拟相机在Body坐标系下的标准位置
        else:  # "learned"
            # Learned坐标系
            K = self.learned_intrinsics.to_matrix()
            T = self.learned_extrinsics.to_transform_matrix()
        
        # 变换到相机坐标系
        points_cam = (np.linalg.inv(T) @ points_3d_hom.T).T
        
        # 投影到图像平面
        points_2d_hom = (K @ points_cam[:, :3].T).T
        
        # 归一化
        points_2d = points_2d_hom[:, :2] / points_2d_hom[:, 2:3]
        
        return points_2d
    
    def unproject_2d_to_3d(
        self,
        points_2d: np.ndarray,
        depth: float,
        frame: str = "virtual"
    ) -> np.ndarray:
        """
        将2D图像点反投影到3D空间
        
        Args:
            points_2d: 2D点 (N, 2)
            depth: 深度值（米）
            frame: 坐标系 ("virtual" 或 "learned")
        
        Returns:
            3D点 (N, 3) in Body coordinates
        """
        # 转换为齐次坐标
        points_2d_hom = np.column_stack([points_2d, np.ones(len(points_2d))])
        
        if frame == "virtual":
            # Virtual坐标系
            K = self.K_virtual
            T = np.eye(4)
        else:  # "learned"
            # Learned坐标系
            K = self.learned_intrinsics.to_matrix()
            T = self.learned_extrinsics.to_transform_matrix()
        
        # 反投影到相机坐标系
        points_cam_hom = (np.linalg.inv(K) @ points_2d_hom.T).T
        points_cam = points_cam_hom * depth
        
        # 转换为齐次坐标
        points_cam_hom = np.column_stack([points_cam, np.ones(len(points_cam))])
        
        # 变换到Body坐标系
        points_body_hom = (T @ points_cam_hom.T).T
        points_body = points_body_hom[:, :3]
        
        return points_body
    
    def get_transform_matrix(self) -> np.ndarray:
        """获取从Learned到Virtual的变换矩阵"""
        return self.T_virtual_from_learned.copy()
    
    def get_homography_matrix(self) -> np.ndarray:
        """获取单应性矩阵"""
        return self.H.copy()
    
    def update_extrinsics(self, new_extrinsics: CameraExtrinsics):
        """
        更新外参（标定更新后调用）
        
        Args:
            new_extrinsics: 新的外参
        """
        self.learned_extrinsics = new_extrinsics
        self._compute_transform_matrices()
    
    def visualize_comparison(
        self,
        original_image: np.ndarray
    ) -> np.ndarray:
        """
        可视化原始图像和虚拟相机图像的对比
        
        Args:
            original_image: 原始图像
        
        Returns:
            对比图像
        """
        import cv2
        
        # 去畸变和校正
        rectified = self.rectify(original_image)
        
        # 调整大小以便并排显示
        h1, w1 = original_image.shape[:2]
        h2, w2 = rectified.shape[:2]
        
        # 统一高度
        target_height = min(h1, h2)
        original_resized = cv2.resize(original_image, (int(w1 * target_height / h1), target_height))
        rectified_resized = cv2.resize(rectified, (int(w2 * target_height / h2), target_height))
        
        # 水平拼接
        comparison = np.hstack([original_resized, rectified_resized])
        
        # 添加标签
        cv2.putText(
            comparison, "Original", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2
        )
        cv2.putText(
            comparison, "Virtual Camera", (w1 + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2
        )
        
        return comparison


class VirtualCameraManager:
    """
    虚拟相机管理器
    管理多个相机的虚拟相机转换
    """
    
    def __init__(self):
        """初始化虚拟相机管理器"""
        self.virtual_cameras: Dict[str, VirtualCamera] = {}
    
    def add_virtual_camera(
        self,
        sensor_id: str,
        learned_intrinsics: CameraIntrinsics,
        learned_extrinsics: CameraExtrinsics,
        virtual_config: VirtualCameraConfig
    ):
        """
        添加虚拟相机
        
        Args:
            sensor_id: 传感器ID
            learned_intrinsics: Learn的相机内参
            learned_extrinsics: Learn的相机外参
            virtual_config: 虚拟相机配置
        """
        self.virtual_cameras[sensor_id] = VirtualCamera(
            learned_intrinsics,
            learned_extrinsics,
            virtual_config
        )
    
    def rectify_image(
        self,
        sensor_id: str,
        image: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        去畸变和校正图像
        
        Args:
            sensor_id: 传感器ID
            image: 原始图像
        
        Returns:
            校正后的图像
        """
        if sensor_id not in self.virtual_cameras:
            return None
        
        return self.virtual_cameras[sensor_id].rectify(image)
    
    def rectify_multiview(
        self,
        images: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        批量去畸变和校正多视角图像
        
        Args:
            images: 图像字典 {sensor_id: image}
        
        Returns:
            校正后的图像字典
        """
        rectified_images = {}
        
        for sensor_id, image in images.items():
            rectified = self.rectify_image(sensor_id, image)
            if rectified is not None:
                rectified_images[sensor_id] = rectified
        
        return rectified_images
    
    def update_sensor_extrinsics(
        self,
        sensor_id: str,
        new_extrinsics: CameraExtrinsics
    ):
        """
        更新传感器外参
        
        Args:
            sensor_id: 传感器ID
            new_extrinsics: 新的外参
        """
        if sensor_id in self.virtual_cameras:
            self.virtual_cameras[sensor_id].update_extrinsics(new_extrinsics)
    
    def get_virtual_camera(self, sensor_id: str) -> Optional[VirtualCamera]:
        """获取虚拟相机"""
        return self.virtual_cameras.get(sensor_id)