"""
隐私脱敏模块
实现人脸、车牌等隐私信息的自动模糊处理
"""

from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field
from enum import Enum
import cv2
import numpy as np


class PrivacyType(str, Enum):
    """隐私类型"""
    FACE = "face"           # 人脸
    LICENSE_PLATE = "license_plate"  # 车牌
    PERSON = "person"       # 人体
    VEHICLE = "vehicle"     # 车辆
    DOCUMENT = "document"   # 文档


class MaskMethod(str, Enum):
    """脱敏方法"""
    BLUR = "blur"           # 模糊
    PIXELATE = "pixelate"   # 像素化
    BLACKOUT = "blackout"   # 黑屏
    OVERLAY = "overlay"     # 遮盖


class DetectionResult(BaseModel):
    """检测结果"""
    type: PrivacyType
    bbox: List[int] = Field(description="边界框 [x, y, w, h]")
    confidence: float = Field(description="置信度")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MaskingConfig(BaseModel):
    """脱敏配置"""
    privacy_types: List[PrivacyType] = Field(default_factory=list)
    mask_method: MaskMethod = Field(default=MaskMethod.BLUR)
    blur_kernel: int = Field(default=31, description="模糊核大小")
    pixelate_size: int = Field(default=10, description="像素化块大小")
    overlay_color: Tuple[int, int, int] = Field(default=(0, 0, 0), description="覆盖颜色")
    min_confidence: float = Field(default=0.5, description="最小置信度")


class MaskingResult(BaseModel):
    """脱敏结果"""
    original_image: Optional[bytes] = None
    masked_image: Optional[bytes] = None
    detections: List[DetectionResult] = Field(default_factory=list)
    masked_count: Dict[str, int] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PrivacyMasker:
    """
    隐私脱敏器
    自动检测和脱敏隐私信息
    """

    def __init__(self):
        self.face_cascade = None
        self._load_models()

    def _load_models(self):
        """加载检测模型"""
        try:
            # 加载人脸检测模型
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        except Exception as e:
            print(f"Failed to load models: {e}")

    def detect_privacy(
        self,
        image: np.ndarray,
        config: MaskingConfig
    ) -> List[DetectionResult]:
        """
        检测隐私信息

        Args:
            image: 图像数据（BGR格式）
            config: 脱敏配置

        Returns:
            List[DetectionResult]: 检测结果列表
        """
        detections = []

        # 检测人脸
        if PrivacyType.FACE in config.privacy_types:
            face_detections = self._detect_faces(image, config.min_confidence)
            detections.extend(face_detections)

        # 检测车牌（简化实现）
        if PrivacyType.LICENSE_PLATE in config.privacy_types:
            plate_detections = self._detect_license_plates(image, config.min_confidence)
            detections.extend(plate_detections)

        # 检测人体（简化实现）
        if PrivacyType.PERSON in config.privacy_types:
            person_detections = self._detect_persons(image, config.min_confidence)
            detections.extend(person_detections)

        return detections

    def mask_image(
        self,
        image: np.ndarray,
        config: MaskingConfig
    ) -> MaskingResult:
        """
        脱敏图像

        Args:
            image: 图像数据（BGR格式）
            config: 脱敏配置

        Returns:
            MaskingResult: 脱敏结果
        """
        # 检测隐私信息
        detections = self.detect_privacy(image, config)

        # 复制图像
        masked_image = image.copy()

        # 统计脱敏数量
        masked_count = {}

        # 脱敏处理
        for detection in detections:
            ptype = detection.type.value
            masked_count[ptype] = masked_count.get(ptype, 0) + 1

            bbox = detection.bbox
            x, y, w, h = bbox

            # 应用脱敏方法
            if config.mask_method == MaskMethod.BLUR:
                self._apply_blur(masked_image, x, y, w, h, config.blur_kernel)
            elif config.mask_method == MaskMethod.PIXELATE:
                self._apply_pixelate(masked_image, x, y, w, h, config.pixelate_size)
            elif config.mask_method == MaskMethod.BLACKOUT:
                self._apply_blackout(masked_image, x, y, w, h, config.overlay_color)
            elif config.mask_method == MaskMethod.OVERLAY:
                self._apply_overlay(masked_image, x, y, w, h, config.overlay_color)

        # 编码图像
        _, original_encoded = cv2.imencode('.jpg', image)
        _, masked_encoded = cv2.imencode('.jpg', masked_image)

        return MaskingResult(
            original_image=original_encoded.tobytes(),
            masked_image=masked_encoded.tobytes(),
            detections=detections,
            masked_count=masked_count
        )

    def _detect_faces(self, image: np.ndarray, min_confidence: float) -> List[DetectionResult]:
        """
        检测人脸

        Args:
            image: 图像数据
            min_confidence: 最小置信度

        Returns:
            List[DetectionResult]: 人脸检测结果
        """
        detections = []

        if self.face_cascade is None:
            return detections

        # 转灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 检测人脸
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        for (x, y, w, h) in faces:
            # 简化：使用固定置信度
            confidence = 0.9

            if confidence >= min_confidence:
                detections.append(DetectionResult(
                    type=PrivacyType.FACE,
                    bbox=[x, y, w, h],
                    confidence=confidence,
                    metadata={"method": "haar_cascade"}
                ))

        return detections

    def _detect_license_plates(self, image: np.ndarray, min_confidence: float) -> List[DetectionResult]:
        """
        检测车牌（简化实现）

        Args:
            image: 图像数据
            min_confidence: 最小置信度

        Returns:
            List[DetectionResult]: 车牌检测结果
        """
        # 简化实现：使用边缘检测和轮廓分析
        # 实际应用中应使用专门的车牌检测模型

        detections = []

        # 转灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 高斯模糊
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 边缘检测
        edges = cv2.Canny(blurred, 50, 150)

        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # 获取边界框
            x, y, w, h = cv2.boundingRect(contour)

            # 过滤：宽高比和尺寸
            aspect_ratio = w / h if h > 0 else 0
            if 2.0 < aspect_ratio < 5.0 and 50 < w < 300 and 20 < h < 100:
                # 简化：使用固定置信度
                confidence = 0.6

                if confidence >= min_confidence:
                    detections.append(DetectionResult(
                        type=PrivacyType.LICENSE_PLATE,
                        bbox=[x, y, w, h],
                        confidence=confidence,
                        metadata={"method": "contour_analysis"}
                    ))

        return detections

    def _detect_persons(self, image: np.ndarray, min_confidence: float) -> List[DetectionResult]:
        """
        检测人体（简化实现）

        Args:
            image: 图像数据
            min_confidence: 最小置信度

        Returns:
            List[DetectionResult]: 人体检测结果
        """
        # 简化实现：使用HOG检测器
        # 实际应用中应使用专门的行人检测模型

        detections = []

        try:
            # 初始化HOG检测器
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

            # 转灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 检测行人
            boxes, weights = hog.detectMultiScale(gray, winStride=(8, 8), padding=(8, 8), scale=1.05)

            for (x, y, w, h), weight in zip(boxes, weights):
                confidence = float(weight)

                if confidence >= min_confidence:
                    detections.append(DetectionResult(
                        type=PrivacyType.PERSON,
                        bbox=[x, y, w, h],
                        confidence=confidence,
                        metadata={"method": "hog_detector"}
                    ))

        except Exception as e:
            print(f"Person detection failed: {e}")

        return detections

    def _apply_blur(self, image: np.ndarray, x: int, y: int, w: int, h: int, kernel_size: int):
        """
        应用模糊

        Args:
            image: 图像
            x, y, w, h: 边界框
            kernel_size: 模糊核大小
        """
        # 确保核大小为奇数
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

        # 提取区域
        roi = image[y:y+h, x:x+w]

        # 应用模糊
        blurred = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)

        # 放回原图
        image[y:y+h, x:x+w] = blurred

    def _apply_pixelate(self, image: np.ndarray, x: int, y: int, w: int, h: int, block_size: int):
        """
        应用像素化

        Args:
            image: 图像
            x, y, w, h: 边界框
            block_size: 像素化块大小
        """
        # 提取区域
        roi = image[y:y+h, x:x+w]

        # 缩小再放大
        small = cv2.resize(roi, (max(1, w // block_size), max(1, h // block_size)), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

        # 放回原图
        image[y:y+h, x:x+w] = pixelated

    def _apply_blackout(self, image: np.ndarray, x: int, y: int, w: int, h: int, color: Tuple[int, int, int]):
        """
        应用黑屏

        Args:
            image: 图像
            x, y, w, h: 边界框
            color: 颜色
        """
        # 填充黑色
        cv2.rectangle(image, (x, y), (x+w, y+h), color, -1)

    def _apply_overlay(self, image: np.ndarray, x: int, y: int, w: int, h: int, color: Tuple[int, int, int]):
        """
        应用覆盖

        Args:
            image: 图像
            x, y, w, h: 边界框
            color: 颜色
        """
        # 半透明覆盖
        overlay = image.copy()
        cv2.rectangle(overlay, (x, y), (x+w, y+h), color, -1)
        cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)

    def mask_video_frame_by_frame(
        self,
        video_path: str,
        output_path: str,
        config: MaskingConfig,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        逐帧脱敏视频

        Args:
            video_path: 输入视频路径
            output_path: 输出视频路径
            config: 脱敏配置
            progress_callback: 进度回调

        Returns:
            Dict: 处理结果
        """
        # 打开视频
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return {"success": False, "error": "无法打开视频"}

        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        total_masked = 0
        frame_idx = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # 脱敏帧
            result = self.mask_image(frame, config)
            masked_frame = cv2.imdecode(np.frombuffer(result.masked_image, np.uint8), cv2.IMREAD_COLOR)

            # 写入帧
            out.write(masked_frame)

            total_masked += len(result.detections)
            frame_idx += 1

            # 进度回调
            if progress_callback:
                progress = frame_idx / frame_count * 100
                progress_callback(progress, frame_idx, frame_count)

        # 释放资源
        cap.release()
        out.release()

        return {
            "success": True,
            "total_frames": frame_count,
            "total_masked": total_masked,
            "output_path": output_path
        }