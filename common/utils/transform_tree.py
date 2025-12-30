"""
刚体变换链（Transform Tree）
维护车辆坐标系树，支持任意坐标系之间的变换
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum


class FrameType(str, Enum):
    """坐标系类型"""
    BODY = "body"           # 车身坐标系（以后轴中心为原点）
    CAMERA = "camera"       # 相机坐标系
    IMU = "imu"             # IMU坐标系
    LIDAR = "lidar"         # 激光雷达坐标系
    RADAR = "radar"         # 毫米波雷达坐标系
    WORLD = "world"         # 世界坐标系
    MAP = "map"             # 地图坐标系


@dataclass
class Transform:
    """刚体变换"""
    frame_id: str           # 坐标系ID
    parent_frame_id: str    # 父坐标系ID
    translation: np.ndarray # 平移向量 (3,)
    rotation: np.ndarray    # 旋转矩阵 (3, 3)
    timestamp: float        # 时间戳
    static: bool = True     # 是否静态变换
    
    def to_matrix(self) -> np.ndarray:
        """转换为4x4齐次变换矩阵"""
        T = np.eye(4)
        T[:3, :3] = self.rotation
        T[:3, 3] = self.translation
        return T
    
    @classmethod
    def from_matrix(
        cls,
        frame_id: str,
        parent_frame_id: str,
        transform_matrix: np.ndarray,
        timestamp: float = 0.0,
        static: bool = True
    ) -> 'Transform':
        """
        从变换矩阵创建Transform
        
        Args:
            frame_id: 坐标系ID
            parent_frame_id: 父坐标系ID
            transform_matrix: 4x4齐次变换矩阵
            timestamp: 时间戳
            static: 是否静态变换
        
        Returns:
            Transform对象
        """
        translation = transform_matrix[:3, 3]
        rotation = transform_matrix[:3, :3]
        
        return cls(
            frame_id=frame_id,
            parent_frame_id=parent_frame_id,
            translation=translation,
            rotation=rotation,
            timestamp=timestamp,
            static=static
        )


class TransformTree:
    """
    刚体变换链
    
    特斯拉非常严格地维护一套坐标系树：
    - Body Frame (车身坐标系): 以后轴中心为原点（通常），这是绝对基准。
    - Camera Frame (相机坐标系): 每个相机相对于Body的6-DoF (x,y,z, roll, pitch, yaw)。
    
    标定不仅是标相机，还要标IMU到车身的安装误差。
    因为所有的运动预测都依赖IMU，如果IMU歪了，整个世界的预测都会歪。
    """
    
    def __init__(self, root_frame: str = "body"):
        """
        初始化变换树
        
        Args:
            root_frame: 根坐标系ID（通常是body）
        """
        self.root_frame = root_frame
        self.transforms: Dict[str, Transform] = {}  # {frame_id: Transform}
        self.children: Dict[str, List[str]] = {}     # {parent_frame_id: [child_frame_ids]}
        self.children[root_frame] = []
    
    def add_transform(self, transform: Transform):
        """
        添加变换
        
        Args:
            transform: Transform对象
        """
        frame_id = transform.frame_id
        parent_id = transform.parent_frame_id
        
        # 检查父坐标系是否存在
        if parent_id not in self.transforms and parent_id != self.root_frame:
            raise ValueError(f"Parent frame '{parent_id}' does not exist")
        
        # 添加变换
        self.transforms[frame_id] = transform
        
        # 更新子节点列表
        if parent_id not in self.children:
            self.children[parent_id] = []
        
        if frame_id not in self.children[parent_id]:
            self.children[parent_id].append(frame_id)
    
    def get_transform(
        self,
        target_frame: str,
        source_frame: str
    ) -> Optional[np.ndarray]:
        """
        获取从source_frame到target_frame的变换矩阵
        
        Args:
            target_frame: 目标坐标系
            source_frame: 源坐标系
        
        Returns:
            4x4变换矩阵
        """
        if target_frame == source_frame:
            return np.eye(4)
        
        # 检查坐标系是否存在
        if target_frame not in self.transforms and target_frame != self.root_frame:
            return None
        
        if source_frame not in self.transforms and source_frame != self.root_frame:
            return None
        
        # 查找从source到root的路径
        path_to_root_source = self._find_path_to_root(source_frame)
        if path_to_root_source is None:
            return None
        
        # 查找从target到root的路径
        path_to_root_target = self._find_path_to_root(target_frame)
        if path_to_root_target is None:
            return None
        
        # 计算从source到root的变换
        T_source_to_root = np.eye(4)
        for frame_id in reversed(path_to_root_source):
            if frame_id == self.root_frame:
                continue
            transform = self.transforms[frame_id]
            T_source_to_root = transform.to_matrix() @ T_source_to_root
        
        # 计算从target到root的变换
        T_target_to_root = np.eye(4)
        for frame_id in reversed(path_to_root_target):
            if frame_id == self.root_frame:
                continue
            transform = self.transforms[frame_id]
            T_target_to_root = transform.to_matrix() @ T_target_to_root
        
        # 计算从source到target的变换
        # T_source_to_target = inv(T_target_to_root) @ T_source_to_root
        T_source_to_target = np.linalg.inv(T_target_to_root) @ T_source_to_root
        
        return T_source_to_target
    
    def transform_point(
        self,
        point: np.ndarray,
        target_frame: str,
        source_frame: str
    ) -> Optional[np.ndarray]:
        """
        变换点坐标
        
        Args:
            point: 3D点 (3,)
            target_frame: 目标坐标系
            source_frame: 源坐标系
        
        Returns:
            变换后的3D点 (3,)
        """
        T = self.get_transform(target_frame, source_frame)
        if T is None:
            return None
        
        # 转换为齐次坐标
        point_hom = np.append(point, 1.0)
        
        # 应用变换
        point_transformed_hom = T @ point_hom
        
        # 返回3D坐标
        return point_transformed_hom[:3]
    
    def transform_points(
        self,
        points: np.ndarray,
        target_frame: str,
        source_frame: str
    ) -> Optional[np.ndarray]:
        """
        批量变换点坐标
        
        Args:
            points: 3D点 (N, 3)
            target_frame: 目标坐标系
            source_frame: 源坐标系
        
        Returns:
            变换后的3D点 (N, 3)
        """
        T = self.get_transform(target_frame, source_frame)
        if T is None:
            return None
        
        # 转换为齐次坐标
        points_hom = np.column_stack([points, np.ones(len(points))])
        
        # 应用变换
        points_transformed_hom = (T @ points_hom.T).T
        
        # 返回3D坐标
        return points_transformed_hom[:, :3]
    
    def _find_path_to_root(self, frame_id: str) -> Optional[List[str]]:
        """
        查找从frame_id到root的路径
        
        Args:
            frame_id: 坐标系ID
        
        Returns:
            路径列表 [frame_id, parent_id, ..., root_frame]
        """
        if frame_id == self.root_frame:
            return [self.root_frame]
        
        if frame_id not in self.transforms:
            return None
        
        path = [frame_id]
        current_frame = frame_id
        
        while current_frame != self.root_frame:
            transform = self.transforms.get(current_frame)
            if transform is None:
                return None
            
            parent_id = transform.parent_frame_id
            path.append(parent_id)
            current_frame = parent_id
            
            # 防止循环
            if len(path) > len(self.transforms) + 1:
                return None
        
        return path
    
    def update_transform(self, frame_id: str, new_transform: Transform):
        """
        更新变换
        
        Args:
            frame_id: 坐标系ID
            new_transform: 新的Transform对象
        """
        if frame_id not in self.transforms:
            raise ValueError(f"Frame '{frame_id}' does not exist")
        
        # 检查父坐标系是否一致
        if new_transform.parent_frame_id != self.transforms[frame_id].parent_frame_id:
            raise ValueError(f"Parent frame mismatch for frame '{frame_id}'")
        
        self.transforms[frame_id] = new_transform
    
    def get_transform_info(self, frame_id: str) -> Optional[Transform]:
        """获取指定坐标系的变换信息"""
        return self.transforms.get(frame_id)
    
    def get_all_frames(self) -> List[str]:
        """获取所有坐标系ID"""
        return list(self.transforms.keys()) + [self.root_frame]
    
    def get_children(self, frame_id: str) -> List[str]:
        """获取指定坐标系的所有子坐标系"""
        return self.children.get(frame_id, [])
    
    def is_valid(self) -> bool:
        """检查变换树是否有效"""
        # 检查是否有循环
        visited = set()
        
        def dfs(frame_id: str) -> bool:
            if frame_id in visited:
                return False
            
            visited.add(frame_id)
            
            for child_id in self.get_children(frame_id):
                if not dfs(child_id):
                    return False
            
            return True
        
        return dfs(self.root_frame)
    
    def visualize(self) -> str:
        """
        可视化变换树结构
        
        Returns:
            树结构的字符串表示
        """
        lines = []
        
        def dfs(frame_id: str, depth: int = 0):
            indent = "  " * depth
            lines.append(f"{indent}{frame_id}")
            
            for child_id in self.get_children(frame_id):
                dfs(child_id, depth + 1)
        
        dfs(self.root_frame)
        
        return "\n".join(lines)
    
    def save_to_dict(self) -> Dict:
        """
        保存变换树到字典
        
        Returns:
            字典表示
        """
        transforms_dict = {}
        
        for frame_id, transform in self.transforms.items():
            transforms_dict[frame_id] = {
                "parent_frame_id": transform.parent_frame_id,
                "translation": transform.translation.tolist(),
                "rotation": transform.rotation.tolist(),
                "timestamp": transform.timestamp,
                "static": transform.static
            }
        
        return {
            "root_frame": self.root_frame,
            "transforms": transforms_dict
        }
    
    @classmethod
    def load_from_dict(cls, data: Dict) -> 'TransformTree':
        """
        从字典加载变换树
        
        Args:
            data: 字典数据
        
        Returns:
            TransformTree对象
        """
        tree = cls(root_frame=data["root_frame"])
        
        for frame_id, transform_data in data["transforms"].items():
            transform = Transform(
                frame_id=frame_id,
                parent_frame_id=transform_data["parent_frame_id"],
                translation=np.array(transform_data["translation"]),
                rotation=np.array(transform_data["rotation"]),
                timestamp=transform_data["timestamp"],
                static=transform_data["static"]
            )
            tree.add_transform(transform)
        
        return tree
    
    def compute_relative_pose(
        self,
        frame1: str,
        frame2: str
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        计算两个坐标系之间的相对位姿
        
        Args:
            frame1: 坐标系1
            frame2: 坐标系2
        
        Returns:
            (旋转矩阵, 平移向量)
        """
        T = self.get_transform(frame1, frame2)
        if T is None:
            return None
        
        R = T[:3, :3]
        t = T[:3, 3]
        
        return R, t
    
    def compute_distance(
        self,
        frame1: str,
        frame2: str
    ) -> Optional[float]:
        """
        计算两个坐标系原点之间的距离
        
        Args:
            frame1: 坐标系1
            frame2: 坐标系2
        
        Returns:
            距离（米）
        """
        R, t = self.compute_relative_pose(frame1, frame2)
        if t is None:
            return None
        
        return np.linalg.norm(t)