"""
车端标定管理器
实现标定流程闭环：启动→收敛→更新→上传
"""

import numpy as np
import time
import threading
from typing import Dict, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass

from common.models.calibration import (
    FactorySpec,
    LearnedSpec,
    CalibrationConfig,
    CalibrationResult,
    CalibrationStatus,
    VehicleCalibrationState,
    CameraExtrinsics,
    SensorType
)
from common.utils.transform_tree import TransformTree
from .auto_calibration import AutoCalibrationEngine
from .virtual_camera import VirtualCameraManager, VirtualCameraConfig


@dataclass
class CalibrationProgress:
    """标定进度"""
    sensor_id: str
    progress: float  # 进度 [0, 1]
    status: CalibrationStatus
    confidence: float
    driving_distance: float  # 累计行驶距离（米）


class CalibrationManager:
    """
    车端标定管理器
    
    标定流程闭环：
    1. 启动阶段 (Boot-up): 加载上次行驶保存的 Learned Spec。如果丢失或被重置，则回退到 Factory Spec。
    2. 行驶阶段 (Run-time): 自标定 (Autocalib): 只要车轮在转，后台的标定进程就在跑。
    3. 收敛 (Convergence): 当偏差小于阈值时，认为标定完成。
    4. 更新 (Update): 实时修正摄像头的 外参（旋转 Rotation / 平移 Translation）。
    5. 上传机制 (Upload to Cloud): 车端会将当前的 Learned Spec 作为元数据包含在 Snapshot 或 Log 中上传。
    """
    
    def __init__(
        self,
        vehicle_id: str,
        factory_specs: Dict[str, FactorySpec],
        config: Optional[CalibrationConfig] = None
    ):
        """
        初始化标定管理器
        
        Args:
            vehicle_id: 车辆ID
            factory_specs: 出厂标称参数字典
            config: 标定配置
        """
        self.vehicle_id = vehicle_id
        self.factory_specs = factory_specs
        self.config = config or CalibrationConfig(
            sensor_id="default",
            sensor_type=SensorType.CAMERA
        )
        
        # 初始化整车标定状态
        self.calibration_state = VehicleCalibrationState(
            vehicle_id=vehicle_id,
            factory_specs=factory_specs.copy()
        )
        
        # 初始化自标定引擎
        self.auto_calibration_engine = AutoCalibrationEngine(
            factory_specs,
            self.config
        )
        
        # 初始化虚拟相机管理器
        self.virtual_camera_manager = VirtualCameraManager()
        self._init_virtual_cameras()
        
        # 初始化变换树
        self.transform_tree = TransformTree(root_frame="body")
        self._init_transform_tree()
        
        # 标定状态
        self.is_running = False
        self.is_calibrating = False
        self.last_upload_time = 0.0
        
        # 上传回调
        self.upload_callback: Optional[Callable] = None
        
        # 工作线程
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
    
    def _init_virtual_cameras(self):
        """初始化虚拟相机"""
        for sensor_id, factory_spec in self.factory_specs.items():
            if factory_spec.sensor_type == SensorType.CAMERA and factory_spec.intrinsics:
                # 创建虚拟相机配置
                virtual_config = VirtualCameraConfig(
                    image_size=(1920, 1080),  # 默认图像尺寸
                    fov=60.0  # 默认视场角
                )
                
                # 添加虚拟相机
                self.virtual_camera_manager.add_virtual_camera(
                    sensor_id,
                    factory_spec.intrinsics,
                    factory_spec.extrinsics,
                    virtual_config
                )
    
    def _init_transform_tree(self):
        """初始化变换树"""
        # Body坐标系是根坐标系
        # 添加各个传感器相对于Body的变换
        for sensor_id, factory_spec in self.factory_specs.items():
            T = factory_spec.extrinsics.to_transform_matrix()
            
            from common.utils.transform_tree import Transform
            transform = Transform.from_matrix(
                frame_id=sensor_id,
                parent_frame_id="body",
                transform_matrix=T,
                timestamp=0.0,
                static=False  # 标定参数会动态更新
            )
            
            self.transform_tree.add_transform(transform)
    
    def boot_up(self):
        """
        启动阶段
        加载上次行驶保存的 Learned Spec。如果丢失或被重置，则回退到 Factory Spec。
        """
        # TODO: 从持久化存储加载Learned Spec
        # 这里简化为使用Factory Spec初始化
        print(f"[CalibrationManager] Boot up for vehicle {self.vehicle_id}")
        print(f"[CalibrationManager] Loaded {len(self.factory_specs)} factory specs")
    
    def start_calibration(self):
        """启动标定"""
        if self.is_running:
            return
        
        self.is_running = True
        self.is_calibrating = True
        self._stop_event.clear()
        
        # 启动工作线程
        self._worker_thread = threading.Thread(target=self._calibration_loop)
        self._worker_thread.daemon = True
        self._worker_thread.start()
        
        print("[CalibrationManager] Calibration started")
    
    def stop_calibration(self):
        """停止标定"""
        if not self.is_running:
            return
        
        self.is_running = False
        self.is_calibrating = False
        self._stop_event.set()
        
        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)
        
        print("[CalibrationManager] Calibration stopped")
    
    def _calibration_loop(self):
        """标定循环（在后台线程中运行）"""
        while not self._stop_event.is_set():
            try:
                # 检查是否需要上传
                current_time = time.time()
                if current_time - self.last_upload_time > self.config.upload_interval:
                    self._upload_calibration_data()
                    self.last_upload_time = current_time
                
                # 等待下一次检查
                time.sleep(1.0)
            
            except Exception as e:
                print(f"[CalibrationManager] Error in calibration loop: {e}")
    
    def update(
        self,
        sensor_data: Dict[str, Dict],
        wheel_speed: Optional[float] = None,
        imu_data: Optional[Dict] = None,
        driving_distance: float = 0.0
    ) -> Dict[str, CalibrationResult]:
        """
        更新标定参数
        
        Args:
            sensor_data: 传感器数据字典
            wheel_speed: 轮速
            imu_data: IMU数据
            driving_distance: 行驶距离（米）
        
        Returns:
            标定结果字典
        """
        if not self.is_calibrating:
            return {}
        
        # 调用自标定引擎更新
        results = self.auto_calibration_engine.update(
            sensor_data,
            wheel_speed,
            imu_data,
            driving_distance
        )
        
        # 更新整车标定状态
        for sensor_id, result in results.items():
            if result.success:
                learned_spec = self.auto_calibration_engine.get_learned_spec(sensor_id)
                if learned_spec:
                    self.calibration_state.learned_specs[sensor_id] = learned_spec
                    
                    # 更新虚拟相机
                    self.virtual_camera_manager.update_sensor_extrinsics(
                        sensor_id,
                        learned_spec.extrinsics
                    )
                    
                    # 更新变换树
                    self._update_transform_tree(sensor_id, learned_spec.extrinsics)
        
        # 更新整车状态
        self.calibration_state.total_driving_distance += driving_distance
        self.calibration_state.update_overall_status()
        
        # 检查是否需要上传
        if self.config.upload_on_convergence:
            if self.calibration_state.overall_status == CalibrationStatus.CONVERGED:
                self._upload_calibration_data()
        
        return results
    
    def _update_transform_tree(self, sensor_id: str, extrinsics: CameraExtrinsics):
        """更新变换树"""
        T = extrinsics.to_transform_matrix()
        
        from common.utils.transform_tree import Transform
        transform = Transform.from_matrix(
            frame_id=sensor_id,
            parent_frame_id="body",
            transform_matrix=T,
            timestamp=time.time(),
            static=False
        )
        
        self.transform_tree.update_transform(sensor_id, transform)
    
    def _upload_calibration_data(self):
        """上传标定数据到云端"""
        if self.upload_callback is None:
            return
        
        # 生成云端元数据
        metadata = self.calibration_state.to_cloud_metadata()
        
        # 调用上传回调
        try:
            success = self.upload_callback(metadata)
            if success:
                self.calibration_state.last_calibration_time = datetime.now()
                print(f"[CalibrationManager] Calibration data uploaded successfully")
            else:
                print(f"[CalibrationManager] Failed to upload calibration data")
        except Exception as e:
            print(f"[CalibrationManager] Error uploading calibration data: {e}")
    
    def set_upload_callback(self, callback: Callable):
        """
        设置上传回调函数
        
        Args:
            callback: 上传回调函数，接收metadata参数，返回bool
        """
        self.upload_callback = callback
    
    def get_calibration_progress(self) -> List[CalibrationProgress]:
        """
        获取标定进度
        
        Returns:
            标定进度列表
        """
        progress_list = []
        
        for sensor_id, learned_spec in self.calibration_state.learned_specs.items():
            progress = CalibrationProgress(
                sensor_id=sensor_id,
                progress=learned_spec.convergence_progress,
                status=learned_spec.status,
                confidence=learned_spec.confidence,
                driving_distance=learned_spec.total_driving_distance
            )
            progress_list.append(progress)
        
        return progress_list
    
    def get_calibration_state(self) -> VehicleCalibrationState:
        """获取整车标定状态"""
        return self.calibration_state
    
    def rectify_image(self, sensor_id: str, image: np.ndarray) -> Optional[np.ndarray]:
        """
        去畸变和校正图像
        
        Args:
            sensor_id: 传感器ID
            image: 原始图像
        
        Returns:
            校正后的图像
        """
        return self.virtual_camera_manager.rectify_image(sensor_id, image)
    
    def rectify_multiview(self, images: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        批量去畸变和校正多视角图像
        
        Args:
            images: 图像字典
        
        Returns:
            校正后的图像字典
        """
        return self.virtual_camera_manager.rectify_multiview(images)
    
    def reset_calibration(self, sensor_id: Optional[str] = None, hard_reset: bool = False):
        """
        重置标定
        
        Args:
            sensor_id: 传感器ID（None表示重置所有）
            hard_reset: 是否硬重置
        """
        if sensor_id is None:
            # 重置所有传感器
            for sid in self.calibration_state.learned_specs.keys():
                self.auto_calibration_engine.reset_sensor(sid, hard_reset)
        else:
            # 重置指定传感器
            self.auto_calibration_engine.reset_sensor(sensor_id, hard_reset)
        
        print(f"[CalibrationManager] Calibration reset: {sensor_id or 'all'}, hard={hard_reset}")
    
    def is_converged(self) -> bool:
        """检查是否所有传感器都已收敛"""
        return self.auto_calibration_engine.is_converged()
    
    def get_transform_tree(self) -> TransformTree:
        """获取变换树"""
        return self.transform_tree
    
    def save_calibration_state(self, filepath: str):
        """
        保存标定状态到文件
        
        Args:
            filepath: 文件路径
        """
        import json
        
        # 保存Learned Spec
        learned_specs_dict = {}
        for sensor_id, learned_spec in self.calibration_state.learned_specs.items():
            learned_specs_dict[sensor_id] = {
                "sensor_id": learned_spec.sensor_id,
                "sensor_type": learned_spec.sensor_type.value,
                "extrinsics": {
                    "translation": learned_spec.extrinsics.translation,
                    "rotation": learned_spec.extrinsics.rotation
                },
                "status": learned_spec.status.value,
                "confidence": learned_spec.confidence,
                "convergence_progress": learned_spec.convergence_progress,
                "update_count": learned_spec.update_count,
                "total_driving_distance": learned_spec.total_driving_distance
            }
        
        # 保存整车状态
        calibration_state_dict = {
            "vehicle_id": self.calibration_state.vehicle_id,
            "overall_status": self.calibration_state.overall_status.value,
            "overall_convergence": self.calibration_state.overall_convergence,
            "total_driving_distance": self.calibration_state.total_driving_distance,
            "health_score": self.calibration_state.health_score,
            "learned_specs": learned_specs_dict
        }
        
        with open(filepath, 'w') as f:
            json.dump(calibration_state_dict, f, indent=2)
        
        print(f"[CalibrationManager] Calibration state saved to {filepath}")
    
    def load_calibration_state(self, filepath: str):
        """
        从文件加载标定状态
        
        Args:
            filepath: 文件路径
        """
        import json
        
        with open(filepath, 'r') as f:
            calibration_state_dict = json.load(f)
        
        # 恢复Learned Spec
        for sensor_id, learned_spec_dict in calibration_state_dict["learned_specs"].items():
            if sensor_id in self.calibration_state.learned_specs:
                learned_spec = self.calibration_state.learned_specs[sensor_id]
                
                # 更新外参
                learned_spec.extrinsics = CameraExtrinsics(
                    translation=tuple(learned_spec_dict["extrinsics"]["translation"]),
                    rotation=tuple(learned_spec_dict["extrinsics"]["rotation"])
                )
                
                # 更新状态
                learned_spec.status = CalibrationStatus(learned_spec_dict["status"])
                learned_spec.confidence = learned_spec_dict["confidence"]
                learned_spec.convergence_progress = learned_spec_dict["convergence_progress"]
                learned_spec.update_count = learned_spec_dict["update_count"]
                learned_spec.total_driving_distance = learned_spec_dict["total_driving_distance"]
                
                # 更新虚拟相机
                self.virtual_camera_manager.update_sensor_extrinsics(
                    sensor_id,
                    learned_spec.extrinsics
                )
                
                # 更新变换树
                self._update_transform_tree(sensor_id, learned_spec.extrinsics)
        
        # 恢复整车状态
        self.calibration_state.overall_status = CalibrationStatus(
            calibration_state_dict["overall_status"]
        )
        self.calibration_state.overall_convergence = calibration_state_dict["overall_convergence"]
        self.calibration_state.total_driving_distance = calibration_state_dict["total_driving_distance"]
        self.calibration_state.health_score = calibration_state_dict["health_score"]
        
        print(f"[CalibrationManager] Calibration state loaded from {filepath}")