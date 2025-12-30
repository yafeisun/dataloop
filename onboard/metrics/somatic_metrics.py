"""
体感指标监控模块
实现一级、二级、三级体感指标的监控和统计
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum
import time
from collections import deque
import numpy as np


class MetricLevel(str, Enum):
    """指标级别"""
    LEVEL_1 = "level_1"  # 一级指标（严重体感）
    LEVEL_2 = "level_2"  # 二级指标（轻微体感）
    LEVEL_3 = "level_3"  # 三级指标（技术指标）


class MetricType(str, Enum):
    """指标类型"""
    ACCELERATION = "acceleration"  # 加速度相关
    VELOCITY = "velocity"  # 速度相关
    ANGULAR_VELOCITY = "angular_velocity"  # 角速度相关
    POSITION = "position"  # 位置相关
    TIME = "time"  # 时间相关
    PERCEPTION = "perception"  # 感知相关
    PLANNING = "planning"  # 规划相关
    CONTROL = "control"  # 控制相关


class MetricConfig(BaseModel):
    """指标配置"""
    metric_id: str
    name: str
    level: MetricLevel
    metric_type: MetricType
    description: str = ""
    unit: str = ""
    threshold: Optional[float] = None  # 阈值
    time_window: float = 1.0  # 时间窗口（秒）
    enabled: bool = True


class MetricValue(BaseModel):
    """指标值"""
    metric_id: str
    timestamp: float
    value: float
    threshold: Optional[float] = None
    exceeded: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MetricStatistics(BaseModel):
    """指标统计"""
    metric_id: str
    count: int = 0
    sum: float = 0.0
    min: float = float('inf')
    max: float = float('-inf')
    mean: float = 0.0
    std: float = 0.0
    exceed_count: int = 0  # 超阈值次数
    exceed_rate: float = 0.0  # 超阈值率

    def update(self, value: float, threshold: Optional[float] = None):
        """更新统计"""
        self.count += 1
        self.sum += value
        self.min = min(self.min, value)
        self.max = max(self.max, value)
        self.mean = self.sum / self.count

        if threshold is not None:
            if value > threshold:
                self.exceed_count += 1
                self.exceed_rate = self.exceed_count / self.count

        # 计算标准差（简化版）
        if self.count > 1:
            self.std = np.sqrt(np.sum((value - self.mean) ** 2) / self.count)

    def reset(self):
        """重置统计"""
        self.count = 0
        self.sum = 0.0
        self.min = float('inf')
        self.max = float('-inf')
        self.mean = 0.0
        self.std = 0.0
        self.exceed_count = 0
        self.exceed_rate = 0.0


class SomaticMetricsMonitor:
    """
    体感指标监控器
    监控和统计各种体感指标
    """

    def __init__(self, history_size: int = 1000):
        self.metrics: Dict[str, MetricConfig] = {}
        self.values: Dict[str, deque] = {}  # metric_id -> deque of MetricValue
        self.statistics: Dict[str, MetricStatistics] = {}
        self.history_size = history_size

    def register_metric(self, config: MetricConfig) -> bool:
        """
        注册指标

        Args:
            config: 指标配置

        Returns:
            bool: 注册是否成功
        """
        if config.metric_id in self.metrics:
            return False

        self.metrics[config.metric_id] = config
        self.values[config.metric_id] = deque(maxlen=self.history_size)
        self.statistics[config.metric_id] = MetricStatistics(metric_id=config.metric_id)
        return True

    def unregister_metric(self, metric_id: str) -> bool:
        """
        注销指标

        Args:
            metric_id: 指标ID

        Returns:
            bool: 注销是否成功
        """
        if metric_id not in self.metrics:
            return False

        del self.metrics[metric_id]
        del self.values[metric_id]
        del self.statistics[metric_id]
        return True

    def update_metric(
        self,
        metric_id: str,
        value: float,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[MetricValue]:
        """
        更新指标值

        Args:
            metric_id: 指标ID
            value: 指标值
            timestamp: 时间戳（默认为当前时间）
            metadata: 元数据

        Returns:
            MetricValue: 指标值
        """
        if metric_id not in self.metrics:
            return None

        config = self.metrics[metric_id]
        if not config.enabled:
            return None

        if timestamp is None:
            timestamp = time.time()

        # 检查是否超阈值
        threshold = config.threshold
        exceeded = False
        if threshold is not None:
            exceeded = abs(value) > threshold

        # 创建指标值
        metric_value = MetricValue(
            metric_id=metric_id,
            timestamp=timestamp,
            value=value,
            threshold=threshold,
            exceeded=exceeded,
            metadata=metadata or {}
        )

        # 保存历史值
        self.values[metric_id].append(metric_value)

        # 更新统计
        self.statistics[metric_id].update(value, threshold)

        return metric_value

    def get_metric_value(self, metric_id: str, index: int = -1) -> Optional[MetricValue]:
        """
        获取指标值

        Args:
            metric_id: 指标ID
            index: 索引（-1表示最新值）

        Returns:
            MetricValue: 指标值
        """
        if metric_id not in self.values:
            return None

        values = self.values[metric_id]
        if not values:
            return None

        return values[index]

    def get_metric_values(
        self,
        metric_id: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> List[MetricValue]:
        """
        获取指标值列表

        Args:
            metric_id: 指标ID
            start_time: 起始时间
            end_time: 结束时间

        Returns:
            List[MetricValue]: 指标值列表
        """
        if metric_id not in self.values:
            return []

        values = list(self.values[metric_id])

        if start_time is not None:
            values = [v for v in values if v.timestamp >= start_time]

        if end_time is not None:
            values = [v for v in values if v.timestamp <= end_time]

        return values

    def get_metric_statistics(self, metric_id: str) -> Optional[MetricStatistics]:
        """获取指标统计"""
        return self.statistics.get(metric_id)

    def get_exceeded_metrics(self, level: Optional[MetricLevel] = None) -> List[str]:
        """
        获取超阈值的指标

        Args:
            level: 指标级别（None表示所有级别）

        Returns:
            List[str]: 指标ID列表
        """
        exceeded = []

        for metric_id, config in self.metrics.items():
            if not config.enabled:
                continue

            if level is not None and config.level != level:
                continue

            latest_value = self.get_metric_value(metric_id)
            if latest_value and latest_value.exceeded:
                exceeded.append(metric_id)

        return exceeded

    def reset_statistics(self, metric_id: Optional[str] = None):
        """
        重置统计

        Args:
            metric_id: 指标ID（None表示重置所有）
        """
        if metric_id is None:
            for stats in self.statistics.values():
                stats.reset()
        elif metric_id in self.statistics:
            self.statistics[metric_id].reset()

    def get_all_metrics(self) -> Dict[str, MetricConfig]:
        """获取所有指标配置"""
        return self.metrics.copy()

    def get_metrics_by_level(self, level: MetricLevel) -> List[MetricConfig]:
        """按级别获取指标"""
        return [config for config in self.metrics.values() if config.level == level]

    def enable_metric(self, metric_id: str) -> bool:
        """启用指标"""
        if metric_id in self.metrics:
            self.metrics[metric_id].enabled = True
            return True
        return False

    def disable_metric(self, metric_id: str) -> bool:
        """禁用指标"""
        if metric_id in self.metrics:
            self.metrics[metric_id].enabled = False
            return True
        return False

    def get_summary(self) -> Dict[str, Any]:
        """获取摘要"""
        return {
            "total_metrics": len(self.metrics),
            "enabled_metrics": sum(1 for m in self.metrics.values() if m.enabled),
            "level_1_metrics": len(self.get_metrics_by_level(MetricLevel.LEVEL_1)),
            "level_2_metrics": len(self.get_metrics_by_level(MetricLevel.LEVEL_2)),
            "level_3_metrics": len(self.get_metrics_by_level(MetricLevel.LEVEL_3)),
            "exceeded_metrics": self.get_exceeded_metrics()
        }


# 预定义体感指标
def create_default_metrics() -> SomaticMetricsMonitor:
    """创建默认体感指标监控器"""
    monitor = SomaticMetricsMonitor()

    # 一级指标（严重体感）
    level_1_metrics = [
        MetricConfig(
            metric_id="emergency_brake",
            name="急刹车",
            level=MetricLevel.LEVEL_1,
            metric_type=MetricType.ACCELERATION,
            description="减速度超过3m/s²",
            unit="m/s²",
            threshold=3.0,
            time_window=1.0
        ),
        MetricConfig(
            metric_id="sharp_turn",
            name="急转弯",
            level=MetricLevel.LEVEL_1,
            metric_type=MetricType.ANGULAR_VELOCITY,
            description="横摆角速度超过15°/s",
            unit="°/s",
            threshold=15.0,
            time_window=1.0
        ),
        MetricConfig(
            metric_id="snake_driving",
            name="蛇形行驶",
            level=MetricLevel.LEVEL_1,
            metric_type=MetricType.ACCELERATION,
            description="横向加速度波动超过0.5m/s²",
            unit="m/s²",
            threshold=0.5,
            time_window=2.0
        ),
        MetricConfig(
            metric_id="frequent_stop",
            name="频繁启停",
            level=MetricLevel.LEVEL_1,
            metric_type=MetricType.VELOCITY,
            description="1分钟内停止次数超过3次",
            unit="count/min",
            threshold=3.0,
            time_window=60.0
        ),
        MetricConfig(
            metric_id="mysterious_slow",
            name="莫名慢行",
            level=MetricLevel.LEVEL_1,
            metric_type=MetricType.VELOCITY,
            description="速度低于5km/h且非拥堵",
            unit="km/h",
            threshold=5.0,
            time_window=10.0
        )
    ]

    # 二级指标（轻微体感）
    level_2_metrics = [
        MetricConfig(
            metric_id="late_arrival",
            name="晚点",
            level=MetricLevel.LEVEL_2,
            metric_type=MetricType.TIME,
            description="偏离计划时间超过30s",
            unit="s",
            threshold=30.0,
            time_window=60.0
        ),
        MetricConfig(
            metric_id="path_deviation",
            name="路径偏离",
            level=MetricLevel.LEVEL_2,
            metric_type=MetricType.POSITION,
            description="横向偏移超过0.5m",
            unit="m",
            threshold=0.5,
            time_window=1.0
        ),
        MetricConfig(
            metric_id="hesitant_lane_change",
            name="换道犹豫",
            level=MetricLevel.LEVEL_2,
            metric_type=MetricType.TIME,
            description="换道准备时间超过5s",
            unit="s",
            threshold=5.0,
            time_window=10.0
        )
    ]

    # 三级指标（技术指标）
    level_3_metrics = [
        MetricConfig(
            metric_id="perception_recall",
            name="感知召回率",
            level=MetricLevel.LEVEL_3,
            metric_type=MetricType.PERCEPTION,
            description="感知召回率",
            unit="%",
            threshold=90.0,
            time_window=10.0
        ),
        MetricConfig(
            metric_id="prediction_accuracy",
            name="预测准确率",
            level=MetricLevel.LEVEL_3,
            metric_type=MetricType.PLANNING,
            description="预测准确率",
            unit="%",
            threshold=85.0,
            time_window=10.0
        ),
        MetricConfig(
            metric_id="planning_success_rate",
            name="规划成功率",
            level=MetricLevel.LEVEL_3,
            metric_type=MetricType.PLANNING,
            description="规划成功率",
            unit="%",
            threshold=95.0,
            time_window=10.0
        ),
        MetricConfig(
            metric_id="control_tracking_error",
            name="控制跟踪误差",
            level=MetricLevel.LEVEL_3,
            metric_type=MetricType.CONTROL,
            description="控制跟踪误差",
            unit="m",
            threshold=0.2,
            time_window=1.0
        )
    ]

    # 注册所有指标
    for metric in level_1_metrics + level_2_metrics + level_3_metrics:
        monitor.register_metric(metric)

    return monitor


# 全局监控器实例
_global_monitor = None


def get_global_monitor() -> SomaticMetricsMonitor:
    """获取全局监控器实例"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = create_default_metrics()
    return _global_monitor