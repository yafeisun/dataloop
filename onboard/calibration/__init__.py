"""
车端标定模块
实现端云协同的动态标定机制
"""

from .auto_calibration import AutoCalibrationEngine
from .vanishing_point import VanishingPointDetector
from .epipolar_constraint import EpipolarGeometry
from .ego_motion_estimator import EgoMotionEstimator
from .virtual_camera import VirtualCamera, VirtualCameraManager, VirtualCameraConfig
from .calibration_manager import CalibrationManager, CalibrationProgress
from .anomaly_handler import AnomalyHandler, AnomalyType, ResetType, AnomalyEvent, ResetEvent

__all__ = [
    "AutoCalibrationEngine",
    "VanishingPointDetector",
    "EpipolarGeometry",
    "EgoMotionEstimator",
    "VirtualCamera",
    "VirtualCameraManager",
    "VirtualCameraConfig",
    "CalibrationManager",
    "CalibrationProgress",
    "AnomalyHandler",
    "AnomalyType",
    "ResetType",
    "AnomalyEvent",
    "ResetEvent"
]