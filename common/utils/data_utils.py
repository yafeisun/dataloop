"""
数据工具函数
提供数据处理、转换、验证等通用工具函数
"""

from typing import Dict, List, Optional, Any, Tuple, Union
import json
import time
import numpy as np
from datetime import datetime
import hashlib


def load_json_file(file_path: str) -> Optional[Dict[str, Any]]:
    """
    加载JSON文件

    Args:
        file_path: 文件路径

    Returns:
        Dict: JSON数据
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to load JSON file {file_path}: {e}")
        return None


def save_json_file(data: Dict[str, Any], file_path: str) -> bool:
    """
    保存JSON文件

    Args:
        data: 数据
        file_path: 文件路径

    Returns:
        bool: 是否成功
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Failed to save JSON file {file_path}: {e}")
        return False


def generate_id(prefix: str = "id") -> str:
    """
    生成唯一ID

    Args:
        prefix: ID前缀

    Returns:
        str: 唯一ID
    """
    timestamp = int(time.time() * 1000)
    random_str = hashlib.md5(str(timestamp).encode()).hexdigest()[:8]
    return f"{prefix}_{timestamp}_{random_str}"


def calculate_distance(pos1: Dict[str, float], pos2: Dict[str, float]) -> float:
    """
    计算两点之间的距离

    Args:
        pos1: 位置1 (x, y, z)
        pos2: 位置2 (x, y, z)

    Returns:
        float: 距离
    """
    x1, y1, z1 = pos1.get("x", 0.0), pos1.get("y", 0.0), pos1.get("z", 0.0)
    x2, y2, z2 = pos2.get("x", 0.0), pos2.get("y", 0.0), pos2.get("z", 0.0)

    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


def calculate_velocity(pos1: Dict[str, float], pos2: Dict[str, float], time_delta: float) -> Dict[str, float]:
    """
    计算速度

    Args:
        pos1: 位置1
        pos2: 位置2
        time_delta: 时间差（秒）

    Returns:
        Dict: 速度
    """
    if time_delta <= 0:
        return {"x": 0.0, "y": 0.0, "z": 0.0}

    x1, y1, z1 = pos1.get("x", 0.0), pos1.get("y", 0.0), pos1.get("z", 0.0)
    x2, y2, z2 = pos2.get("x", 0.0), pos2.get("y", 0.0), pos2.get("z", 0.0)

    return {
        "x": (x2 - x1) / time_delta,
        "y": (y2 - y1) / time_delta,
        "z": (z2 - z1) / time_delta
    }


def calculate_acceleration(vel1: Dict[str, float], vel2: Dict[str, float], time_delta: float) -> Dict[str, float]:
    """
    计算加速度

    Args:
        vel1: 速度1
        vel2: 速度2
        time_delta: 时间差（秒）

    Returns:
        Dict: 加速度
    """
    if time_delta <= 0:
        return {"x": 0.0, "y": 0.0, "z": 0.0}

    x1, y1, z1 = vel1.get("x", 0.0), vel1.get("y", 0.0), vel1.get("z", 0.0)
    x2, y2, z2 = vel2.get("x", 0.0), vel2.get("y", 0.0), vel2.get("z", 0.0)

    return {
        "x": (x2 - x1) / time_delta,
        "y": (y2 - y1) / time_delta,
        "z": (z2 - z1) / time_delta
    }


def smooth_trajectory(trajectory: List[Dict[str, float]], window_size: int = 5) -> List[Dict[str, float]]:
    """
    平滑轨迹

    Args:
        trajectory: 轨迹点列表
        window_size: 窗口大小

    Returns:
        List: 平滑后的轨迹
    """
    if len(trajectory) < window_size:
        return trajectory

    smoothed = []

    for i in range(len(trajectory)):
        start = max(0, i - window_size // 2)
        end = min(len(trajectory), i + window_size // 2 + 1)

        window = trajectory[start:end]

        # 计算平均值
        avg_x = np.mean([p["x"] for p in window])
        avg_y = np.mean([p["y"] for p in window])
        avg_z = np.mean([p["z"] for p in window])

        smoothed.append({"x": avg_x, "y": avg_y, "z": avg_z})

    return smoothed


def interpolate_trajectory(
    trajectory: List[Dict[str, float]],
    num_points: int
) -> List[Dict[str, float]]:
    """
    插值轨迹

    Args:
        trajectory: 轨迹点列表
        num_points: 插值点数

    Returns:
        List: 插值后的轨迹
    """
    if len(trajectory) < 2:
        return trajectory

    # 提取坐标
    x_coords = [p["x"] for p in trajectory]
    y_coords = [p["y"] for p in trajectory]
    z_coords = [p["z"] for p in trajectory]

    # 生成插值点
    t_original = np.linspace(0, 1, len(trajectory))
    t_new = np.linspace(0, 1, num_points)

    x_interp = np.interp(t_new, t_original, x_coords)
    y_interp = np.interp(t_new, t_original, y_coords)
    z_interp = np.interp(t_new, t_original, z_coords)

    # 构建插值轨迹
    interpolated = [
        {"x": float(x), "y": float(y), "z": float(z)}
        for x, y, z in zip(x_interp, y_interp, z_interp)
    ]

    return interpolated


def validate_bbox(bbox: Dict[str, Any]) -> bool:
    """
    验证边界框

    Args:
        bbox: 边界框

    Returns:
        bool: 是否有效
    """
    required_keys = ["x", "y", "width", "height"]

    for key in required_keys:
        if key not in bbox:
            return False

    if bbox["width"] <= 0 or bbox["height"] <= 0:
        return False

    return True


def calculate_iou(bbox1: Dict[str, Any], bbox2: Dict[str, Any]) -> float:
    """
    计算交并比（IoU）

    Args:
        bbox1: 边界框1
        bbox2: 边界框2

    Returns:
        float: IoU
    """
    if not validate_bbox(bbox1) or not validate_bbox(bbox2):
        return 0.0

    # 计算交集
    x1 = max(bbox1["x"], bbox2["x"])
    y1 = max(bbox1["y"], bbox2["y"])
    x2 = min(bbox1["x"] + bbox1["width"], bbox2["x"] + bbox2["width"])
    y2 = min(bbox1["y"] + bbox1["height"], bbox2["y"] + bbox2["height"])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    # 计算并集
    area1 = bbox1["width"] * bbox1["height"]
    area2 = bbox2["width"] * bbox2["height"]
    union_area = area1 + area2 - intersection_area

    if union_area == 0:
        return 0.0

    return intersection_area / union_area


def normalize_vector(vector: Dict[str, float]) -> Dict[str, float]:
    """
    归一化向量

    Args:
        vector: 向量

    Returns:
        Dict: 归一化后的向量
    """
    x, y, z = vector.get("x", 0.0), vector.get("y", 0.0), vector.get("z", 0.0)
    magnitude = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    if magnitude == 0:
        return {"x": 0.0, "y": 0.0, "z": 0.0}

    return {
        "x": x / magnitude,
        "y": y / magnitude,
        "z": z / magnitude
    }


def calculate_angle_between_vectors(
    vec1: Dict[str, float],
    vec2: Dict[str, float]
) -> float:
    """
    计算两个向量之间的角度

    Args:
        vec1: 向量1
        vec2: 向量2

    Returns:
        float: 角度（弧度）
    """
    norm1 = normalize_vector(vec1)
    norm2 = normalize_vector(vec2)

    dot_product = (
        norm1["x"] * norm2["x"] +
        norm1["y"] * norm2["y"] +
        norm1["z"] * norm2["z"]
    )

    # 限制范围以避免数值误差
    dot_product = max(-1.0, min(1.0, dot_product))

    return np.arccos(dot_product)


def timestamp_to_datetime(timestamp: float) -> datetime:
    """
    时间戳转日期时间

    Args:
        timestamp: 时间戳

    Returns:
        datetime: 日期时间
    """
    return datetime.fromtimestamp(timestamp)


def datetime_to_timestamp(dt: datetime) -> float:
    """
    日期时间转时间戳

    Args:
        dt: 日期时间

    Returns:
        float: 时间戳
    """
    return dt.timestamp()


def format_timestamp(timestamp: float, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    格式化时间戳

    Args:
        timestamp: 时间戳
        format_str: 格式字符串

    Returns:
        str: 格式化后的字符串
    """
    dt = timestamp_to_datetime(timestamp)
    return dt.strftime(format_str)


def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """
    计算统计信息

    Args:
        values: 数值列表

    Returns:
        Dict: 统计信息
    """
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0
        }

    values_array = np.array(values)

    return {
        "count": len(values),
        "mean": float(np.mean(values_array)),
        "std": float(np.std(values_array)),
        "min": float(np.min(values_array)),
        "max": float(np.max(values_array)),
        "median": float(np.median(values_array))
    }


def filter_data_by_time_range(
    data: List[Dict[str, Any]],
    start_time: float,
    end_time: float,
    timestamp_key: str = "timestamp"
) -> List[Dict[str, Any]]:
    """
    按时间范围过滤数据

    Args:
        data: 数据列表
        start_time: 起始时间
        end_time: 结束时间
        timestamp_key: 时间戳键名

    Returns:
        List: 过滤后的数据
    """
    return [
        item for item in data
        if start_time <= item.get(timestamp_key, 0) <= end_time
    ]


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    合并多个字典

    Args:
        *dicts: 字典列表

    Returns:
        Dict: 合并后的字典
    """
    result = {}
    for d in dicts:
        result.update(d)
    return result


def deep_copy_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    深拷贝字典

    Args:
        data: 字典数据

    Returns:
        Dict: 拷贝后的字典
    """
    return json.loads(json.dumps(data))


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    将列表分块

    Args:
        lst: 列表
        chunk_size: 块大小

    Returns:
        List: 分块后的列表
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def flatten_dict(data: Dict[str, Any], separator: str = ".") -> Dict[str, Any]:
    """
    展平嵌套字典

    Args:
        data: 嵌套字典
        separator: 分隔符

    Returns:
        Dict: 展平后的字典
    """
    result = {}

    def _flatten(obj: Any, parent_key: str = ""):
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_key = f"{parent_key}{separator}{key}" if parent_key else key
                _flatten(value, new_key)
        else:
            result[parent_key] = obj

    _flatten(data)
    return result


def unflatten_dict(data: Dict[str, Any], separator: str = ".") -> Dict[str, Any]:
    """
    反展平字典

    Args:
        data: 展平字典
        separator: 分隔符

    Returns:
        Dict: 嵌套字典
    """
    result = {}

    for key, value in data.items():
        keys = key.split(separator)
        current = result

        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        current[keys[-1]] = value

    return result


def safe_get(data: Dict[str, Any], key: str, default: Any = None) -> Any:
    """
    安全获取字典值

    Args:
        data: 字典
        key: 键
        default: 默认值

    Returns:
        Any: 值
    """
    return data.get(key, default)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    安全除法

    Args:
        numerator: 分子
        denominator: 分母
        default: 默认值

    Returns:
        float: 结果
    """
    if denominator == 0:
        return default
    return numerator / denominator