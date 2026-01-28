"""
斑马线检测模块（优化版）
基于 Sobel 梯度 + 行投影 + 峰值检测的方法
适合前视相机 + 斑马线场景，速度快、准确率高
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional


def extract_roi(
    img: np.ndarray, roi_ratio: Tuple[float, float, float, float]
) -> np.ndarray:
    """
    提取感兴趣区域（ROI）

    Args:
        img: 输入图像
        roi_ratio: ROI比例 (x_start_ratio, x_end_ratio, y_start_ratio, y_end_ratio)

    Returns:
        np.ndarray: ROI区域图像
    """
    height, width = img.shape[:2]
    x_start, x_end, y_start, y_end = roi_ratio

    roi_x_start = int(width * x_start)
    roi_x_end = int(width * x_end)
    roi_y_start = int(height * y_start)
    roi_y_end = int(height * y_end)

    return img[roi_y_start:roi_y_end, roi_x_start:roi_x_end]


def preprocess_with_sobel(roi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    图像预处理：灰度化 + 高斯模糊 + Sobel(y) 梯度

    Args:
        roi: ROI区域图像

    Returns:
        Tuple: (sobel_y, binary) - Sobel梯度图和二值化图
    """
    # 转换为灰度图
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 高斯模糊降噪
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Sobel 算子，只计算 y 方向梯度（斑马线是横条）
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel_y = np.abs(sobel_y)

    # 归一化到 0-255
    sobel_y_norm = np.uint8(255 * sobel_y / (sobel_y.max() + 1e-6))

    # 二值化（OTSU 自适应阈值）
    _, binary = cv2.threshold(sobel_y_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return sobel_y_norm, binary


def calculate_row_projection(binary: np.ndarray) -> np.ndarray:
    """
    计算行投影（按行求和）

    Args:
        binary: 二值图像

    Returns:
        np.ndarray: 归一化的行投影信号
    """
    # 按行求和
    row_proj = np.sum(binary > 0, axis=1)

    # 归一化（除以宽度，得到每行的白色像素比例）
    row_proj_norm = row_proj / (binary.shape[1] + 1e-6)

    return row_proj_norm


def detect_peaks(
    signal: np.ndarray,
    min_peak_height: float,
    min_distance: int = 5,
) -> List[int]:
    """
    检测信号中的峰值（寻找局部最大值）

    Args:
        signal: 输入信号
        min_peak_height: 最小峰值高度
        min_distance: 峰值之间的最小距离（像素）

    Returns:
        List[int]: 峰值中心位置列表
    """
    if len(signal) < 3:
        return []

    # 找到所有高于阈值的点，且是局部最大值
    peaks = []
    for i in range(1, len(signal) - 1):
        if signal[i] > min_peak_height:
            if signal[i] >= signal[i - 1] and signal[i] >= signal[i + 1]:
                peaks.append(i)

    if not peaks:
        return []

    # 过滤距离太近的峰（保留较高的那个）
    filtered_peaks = []
    # 按信号强度排序
    sorted_indices = sorted(peaks, key=lambda x: signal[x], reverse=True)

    for p in sorted_indices:
        keep = True
        for fp in filtered_peaks:
            if abs(p - fp) < min_distance:
                keep = False
                break
        if keep:
            filtered_peaks.append(p)

    return sorted(filtered_peaks)


def calculate_horizontal_features(
    binary: np.ndarray, peak_centers: List[int]
) -> List[dict]:
    """
    计算每个峰值行的水平特征（跨度、连续性、占用度）

    Args:
        binary: 二值图像
        peak_centers: 峰值行索引列表

    Returns:
        List[dict]: 每个峰值的特征列表
    """
    features = []
    width = binary.shape[1]
    num_segments = 10
    segment_width = width / num_segments

    for row_idx in peak_centers:
        row = binary[row_idx, :]
        white_positions = np.where(row > 0)[0]

        if len(white_positions) == 0:
            features.append({"span_ratio": 0, "continuity": 0, "occupancy": 0})
            continue

        # 计算跨度
        span = white_positions[-1] - white_positions[0] + 1
        span_ratio = span / width

        # 计算连续性（白色像素占跨度的比例）
        continuity = len(white_positions) / span

        # 计算分段占用度（10个分段中有多少个包含白色像素）
        occupied_segments = 0
        for s in range(num_segments):
            seg_start = s * segment_width
            seg_end = (s + 1) * segment_width
            if np.any((white_positions >= seg_start) & (white_positions < seg_end)):
                occupied_segments += 1

        features.append(
            {
                "span_ratio": span_ratio,
                "continuity": continuity,
                "occupancy": occupied_segments / num_segments,
            }
        )

    return features


def validate_zebra_pattern(
    peak_centers: List[int],
    roi_height: int,
    horizontal_features: List[dict],
    min_peaks: int,
    min_spacing: float,
    max_spacing: float,
    min_coverage: float,
    min_span: float = 0.6,
    min_occupancy: float = 0.5,
    max_continuity: float = 0.4,
    min_regularity: float = 0.6,
    max_spacing_ratio: float = 2.0,
) -> Tuple[bool, dict]:
    """
    验证斑马线模式（基于“优质峰值”计数和鲁棒规律性）

    优质峰值定义：
    1. 水平跨度足够大 (span_ratio > min_span)
    2. 水平分布足够广 (occupancy > min_occupancy)
    3. 连续性不过高 (continuity < max_continuity)

    Args:
        peak_centers: 峰值中心位置列表
        roi_height: ROI 高度
        horizontal_features: 水平特征列表
        min_peaks: 最小“优质峰值”数量
        min_spacing: 最小平均间距（像素）
        max_spacing: 最大平均间距（像素）
        min_coverage: 最小覆盖率
        min_regularity: 最小间距规律性比例（多少比例的间距接近中位数）
        max_spacing_ratio: 最大平均间距/中值间距比例（防止存在巨大空隙）

    Returns:
        Tuple: (is_zebra, details)
    """
    details = {
        "peak_count": len(peak_centers),
        "good_peak_count": 0,
        "mean_spacing": 0,
        "median_spacing": 0,
        "spacing_ratio": 0,
        "regularity": 0,
        "coverage": 0,
        "good_peaks": [],
    }

    if len(peak_centers) < 2:
        return False, details

    # 筛选优质峰值
    good_peaks_idx = []
    for i, f in enumerate(horizontal_features):
        if (
            f["span_ratio"] >= min_span
            and f["occupancy"] >= min_occupancy
            and f["continuity"] <= max_continuity
        ):
            good_peaks_idx.append(i)

    good_peak_centers = [peak_centers[i] for i in good_peaks_idx]
    details["good_peak_count"] = len(good_peak_centers)
    details["good_peaks"] = good_peak_centers

    if len(good_peak_centers) < min_peaks:
        return False, details

    # 计算间距（基于优质峰值）
    spacings = np.diff(good_peak_centers)
    if len(spacings) == 0:
        return False, details

    mean_spacing = np.mean(spacings)
    median_spacing = np.median(spacings)

    # 计算规律性：有多少比例的间距在 [0.5, 2.0] * 中位数 范围内
    regular_count = sum(
        1 for s in spacings if 0.5 * median_spacing <= s <= 2.0 * median_spacing
    )
    regularity = regular_count / len(spacings)

    # 计算平均值与中值的比例，用于识别是否有巨大空隙
    spacing_ratio = mean_spacing / median_spacing if median_spacing > 0 else 0

    peak_span = good_peak_centers[-1] - good_peak_centers[0]
    coverage = peak_span / roi_height

    details["mean_spacing"] = float(mean_spacing)
    details["median_spacing"] = float(median_spacing)
    details["spacing_ratio"] = float(spacing_ratio)
    details["regularity"] = float(regularity)
    details["coverage"] = float(coverage)

    # 判定逻辑
    is_zebra = (
        len(good_peak_centers) >= min_peaks
        and coverage >= min_coverage
        and min_spacing <= mean_spacing <= max_spacing
        and regularity >= min_regularity
        and spacing_ratio <= max_spacing_ratio
    )

    return is_zebra, details


def detect_zebra_crossing(
    image_path: str,
    roi_ratio: Tuple[float, float, float, float] = (0.25, 0.75, 0.60, 0.88),
    min_peaks: int = 4,
    min_spacing: float = 5,
    max_spacing: float = 120,
    min_coverage: float = 0.05,
    min_span: float = 0.6,
    min_occupancy: float = 0.5,
    max_continuity: float = 0.4,
    min_regularity: float = 0.6,
    max_spacing_ratio: float = 2.0,
    peak_threshold_ratio: float = 0.3,
    min_peak_width: int = 5,
    debug: bool = False,
    **kwargs,
) -> bool:
    """
    检测图片中是否包含斑马线（增强版：Sobel + 行投影 + 水平特征分析）

    检测策略：
    1. 提取 ROI 区域
    2. 灰度化 + 高斯模糊 + Sobel(y) 梯度提取横向边缘
    3. OTSU 二值化
    4. 行投影并进行峰值检测 (局部最大值)
    5. 对每个峰值行进行水平特征分析（跨度、占用度、连续性）
    6. 统计满足特定条件的“优质峰值”数量，并校验其垂直分布规律（鲁棒规律性）

    Args:
        image_path: 图片文件路径
        roi_ratio: ROI 比例 (x_start, x_end, y_start, y_end)
        min_peaks: 最少“优质峰值”数量
        min_spacing: 最小平均间距（像素）
        max_spacing: 最大平均间距（像素）
        min_coverage: 最小垂直覆盖率
        min_span: 最小水平跨度比例
        min_occupancy: 最小水平占用率
        max_continuity: 最大水平连续性
        min_regularity: 最小间距规律性比例
        max_spacing_ratio: 最大平均/中值间距比例
        peak_threshold_ratio: 峰值阈值比例
        min_peak_width: 最小峰值间隔
        debug: 是否输出调试信息

    Returns:
        bool: 是否检测到斑马线
    """
    try:
        # 读取图片
        img = cv2.imread(image_path)
        if img is None:
            return False

        # 提取 ROI 区域
        roi = extract_roi(img, roi_ratio)

        # 图像预处理（Sobel + 二值化）
        sobel_y, binary = preprocess_with_sobel(roi)

        # 行投影
        row_proj = calculate_row_projection(binary)

        # 峰值检测
        max_val = np.max(row_proj)
        if max_val == 0:
            return False

        threshold = peak_threshold_ratio * max_val
        peak_centers = detect_peaks(row_proj, threshold, min_distance=min_peak_width)

        # 计算水平特征
        horizontal_features = calculate_horizontal_features(binary, peak_centers)

        # 验证斑马线模式
        is_zebra, details = validate_zebra_pattern(
            peak_centers,
            roi.shape[0],
            horizontal_features,
            min_peaks,
            min_spacing,
            max_spacing,
            min_coverage,
            min_span,
            min_occupancy,
            max_continuity,
            min_regularity,
            max_spacing_ratio,
        )

        # 调试输出
        debug = False
        if debug:
            print(f"    图片: {image_path.split('/')[-1]}")
            print(
                f"    总峰值: {details['peak_count']}, 优质峰值: {details['good_peak_count']}"
            )
            if details["good_peak_count"] >= 2:
                print(f"    垂直覆盖: {details['coverage']:.2f} (阈值: {min_coverage})")
                print(
                    f"    平均间距: {details['mean_spacing']:.1f}, 规律性: {details['regularity']:.2f}, 间距比: {details['spacing_ratio']:.2f}"
                )
            print(f"    判定结果: {'✓ 有' if is_zebra else '✗ 无'}斑马线")

        return is_zebra

    except Exception as e:
        print(f"    检测斑马线时出错 {image_path}: {e}")
        return False

        # 提取 ROI 区域
        roi = extract_roi(img, roi_ratio)

        # 图像预处理（Sobel + 二值化）
        sobel_y, binary = preprocess_with_sobel(roi)

        # 行投影
        row_proj = calculate_row_projection(binary)

        # 峰值检测
        max_val = np.max(row_proj)
        if max_val == 0:
            return False

        threshold = peak_threshold_ratio * max_val
        peak_centers = detect_peaks(row_proj, threshold, min_distance=min_peak_width)

        # 计算水平特征
        horizontal_features = calculate_horizontal_features(binary, peak_centers)

        # 验证斑马线模式
        is_zebra, details = validate_zebra_pattern(
            peak_centers,
            roi.shape[0],
            horizontal_features,
            min_peaks,
            min_spacing,
            max_spacing,
            min_coverage,
            min_span,
            min_occupancy,
            max_continuity,
            min_regularity,
        )

        # 调试输出
        if debug:
            print(f"    图片: {image_path.split('/')[-1]}")
            print(
                f"    总峰值: {details['peak_count']}, 优质峰值: {details['good_peak_count']}"
            )
            print(f"    垂直覆盖: {details['coverage']:.2f} (阈值: {min_coverage})")
            print(
                f"    平均间距: {details['mean_spacing']:.1f}, 规律性: {details['regularity']:.2f}"
            )
            print(f"    判定结果: {'✓ 有' if is_zebra else '✗ 无'}斑马线")

        return is_zebra

    except Exception as e:
        print(f"    检测斑马线时出错 {image_path}: {e}")
        return False

        # 提取 ROI 区域
        roi = extract_roi(img, roi_ratio)

        # 图像预处理（Sobel + 二值化）
        sobel_y, binary = preprocess_with_sobel(roi)

        # 行投影
        row_proj = calculate_row_projection(binary)

        # 峰值检测
        max_val = np.max(row_proj)
        if max_val == 0:
            return False

        threshold = peak_threshold_ratio * max_val
        peak_centers = detect_peaks(row_proj, threshold, min_distance=min_peak_width)

        # 计算水平特征
        horizontal_features = calculate_horizontal_features(binary, peak_centers)

        # 验证斑马线模式
        is_zebra, details = validate_zebra_pattern(
            peak_centers,
            roi.shape[0],
            horizontal_features,
            min_peaks,
            min_spacing,
            max_spacing,
            min_coverage,
            min_span,
            min_occupancy,
            max_continuity,
            max_cv,
        )

        # 调试输出
        if debug:
            print(f"    图片: {image_path.split('/')[-1]}")
            print(
                f"    总峰值: {details['peak_count']}, 优质峰值: {details['good_peak_count']}"
            )
            print(f"    优质峰值索引: {details['good_peaks']}")
            print(f"    垂直覆盖: {details['coverage']:.2f} (阈值: {min_coverage})")
            print(
                f"    平均间距: {details['mean_spacing']:.1f}, CV: {details['spacing_cv']:.2f}"
            )
            print(f"    判定结果: {'✓ 有' if is_zebra else '✗ 无'}斑马线")

        return is_zebra

    except Exception as e:
        print(f"    检测斑马线时出错 {image_path}: {e}")
        return False

        # 提取 ROI 区域
        roi = extract_roi(img, roi_ratio)

        # 图像预处理（Sobel + 二值化）
        sobel_y, binary = preprocess_with_sobel(roi)

        # 行投影
        row_proj = calculate_row_projection(binary)

        # 峰值检测
        max_val = np.max(row_proj)
        if max_val == 0:
            return False

        threshold = peak_threshold_ratio * max_val
        peak_centers = detect_peaks(row_proj, threshold, min_distance=min_peak_width)

        # 计算水平特征
        horizontal_features = calculate_horizontal_features(binary, peak_centers)

        # 验证斑马线模式
        is_zebra, details = validate_zebra_pattern(
            peak_centers,
            roi.shape[0],
            horizontal_features,
            min_peaks,
            min_spacing,
            max_spacing,
            min_coverage,
            min_span,
            min_occupancy,
            max_continuity,
        )

        # 调试输出
        if debug:
            print(f"    图片: {image_path.split('/')[-1]}")
            print(
                f"    总峰值: {details['peak_count']}, 优质峰值: {details['good_peak_count']}"
            )
            print(f"    优质峰值索引: {details['good_peaks']}")
            print(f"    垂直覆盖: {details['coverage']:.2f} (阈值: {min_coverage})")
            print(f"    平均间距: {details['mean_spacing']:.1f}")
            print(f"    判定结果: {'✓ 有' if is_zebra else '✗ 无'}斑马线")

        return is_zebra

    except Exception as e:
        print(f"    检测斑马线时出错 {image_path}: {e}")
        return False
