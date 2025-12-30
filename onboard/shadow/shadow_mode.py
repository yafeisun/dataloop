"""
Shadow Mode（影子模式）模块
实现在线模型与影子模型的并行运行和分歧检测
"""

from typing import Dict, List, Optional, Any, Callable
from pydantic import BaseModel, Field
from enum import Enum
import time
import threading


class ShadowModelType(str, Enum):
    """影子模型类型"""
    NEW_VERSION = "new_version"  # 新版本模型
    ABLATION = "ablation"       # 消融实验模型
    ALTERNATIVE = "alternative"  # 替代方案模型


class DivergenceType(str, Enum):
    """分歧类型"""
    TRAJECTORY = "trajectory"       # 轨迹分歧
    CLASSIFICATION = "classification"  # 分类分歧
    CONFIDENCE = "confidence"       # 置信度分歧
    LANE_CHANGE = "lane_change"     # 换道分歧
    BRAKING = "braking"             # 刹车分歧


class DivergenceEvent(BaseModel):
    """分歧事件"""
    divergence_id: str
    timestamp: float
    divergence_type: DivergenceType
    online_model_output: Dict[str, Any]
    shadow_model_output: Dict[str, Any]
    divergence_value: float
    threshold: float
    severity: str  # low, medium, high, critical
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ShadowModelConfig(BaseModel):
    """影子模型配置"""
    model_id: str
    model_type: ShadowModelType
    model_version: str
    divergence_threshold: float = Field(default=0.3, ge=0, le=1, description="分歧阈值")
    enabled: bool = Field(default=True, description="是否启用")
    priority: int = Field(default=1, ge=1, le=10, description="优先级")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ShadowMode:
    """
    影子模式管理器
    管理在线模型和影子模型的并行运行
    """

    def __init__(self):
        self.shadow_models: Dict[str, ShadowModelConfig] = {}
        self.divergence_history: List[DivergenceEvent] = []
        self.max_history_size = 10000
        self._lock = threading.Lock()
        self._divergence_callback: Optional[Callable] = None

    def register_shadow_model(self, config: ShadowModelConfig) -> bool:
        """
        注册影子模型

        Args:
            config: 影子模型配置

        Returns:
            bool: 注册是否成功
        """
        with self._lock:
            if config.model_id in self.shadow_models:
                return False

            self.shadow_models[config.model_id] = config
            return True

    def unregister_shadow_model(self, model_id: str) -> bool:
        """
        注销影子模型

        Args:
            model_id: 模型ID

        Returns:
            bool: 注销是否成功
        """
        with self._lock:
            if model_id not in self.shadow_models:
                return False

            del self.shadow_models[model_id]
            return True

    def enable_shadow_model(self, model_id: str) -> bool:
        """
        启用影子模型

        Args:
            model_id: 模型ID

        Returns:
            bool: 是否成功
        """
        with self._lock:
            if model_id not in self.shadow_models:
                return False

            self.shadow_models[model_id].enabled = True
            return True

    def disable_shadow_model(self, model_id: str) -> bool:
        """
        禁用影子模型

        Args:
            model_id: 模型ID

        Returns:
            bool: 是否成功
        """
        with self._lock:
            if model_id not in self.shadow_models:
                return False

            self.shadow_models[model_id].enabled = False
            return True

    def evaluate_models(
        self,
        timestamp: float,
        online_model_output: Dict[str, Any],
        shadow_model_outputs: Dict[str, Dict[str, Any]]
    ) -> List[DivergenceEvent]:
        """
        评估在线模型和影子模型的输出，检测分歧

        Args:
            timestamp: 时间戳
            online_model_output: 在线模型输出
            shadow_model_outputs: 影子模型输出（key为model_id）

        Returns:
            List[DivergenceEvent]: 分歧事件列表
        """
        divergence_events = []

        with self._lock:
            for model_id, shadow_output in shadow_model_outputs.items():
                # 检查影子模型是否启用
                config = self.shadow_models.get(model_id)
                if not config or not config.enabled:
                    continue

                # 检测轨迹分歧
                trajectory_divergence = self._check_trajectory_divergence(
                    online_model_output,
                    shadow_output,
                    config.divergence_threshold
                )

                if trajectory_divergence:
                    divergence_events.append(trajectory_divergence)

                # 检测分类分歧
                classification_divergence = self._check_classification_divergence(
                    online_model_output,
                    shadow_output,
                    config.divergence_threshold
                )

                if classification_divergence:
                    divergence_events.append(classification_divergence)

                # 检测换道分歧
                lane_change_divergence = self._check_lane_change_divergence(
                    online_model_output,
                    shadow_output
                )

                if lane_change_divergence:
                    divergence_events.append(lane_change_divergence)

                # 检测刹车分歧
                braking_divergence = self._check_braking_divergence(
                    online_model_output,
                    shadow_output
                )

                if braking_divergence:
                    divergence_events.append(braking_divergence)

            # 记录分歧历史
            self.divergence_history.extend(divergence_events)

            # 限制历史大小
            if len(self.divergence_history) > self.max_history_size:
                self.divergence_history = self.divergence_history[-self.max_history_size:]

        # 触发回调
        if divergence_events and self._divergence_callback:
            self._divergence_callback(divergence_events)

        return divergence_events

    def _check_trajectory_divergence(
        self,
        online_output: Dict[str, Any],
        shadow_output: Dict[str, Any],
        threshold: float
    ) -> Optional[DivergenceEvent]:
        """
        检测轨迹分歧

        Args:
            online_output: 在线模型输出
            shadow_output: 影子模型输出
            threshold: 阈值

        Returns:
            Optional[DivergenceEvent]: 分歧事件
        """
        online_traj = online_output.get("trajectory", [])
        shadow_traj = shadow_output.get("trajectory", [])

        if not online_traj or not shadow_traj:
            return None

        # 计算轨迹差异
        divergence = self._calculate_trajectory_divergence(online_traj, shadow_traj)

        if divergence > threshold:
            # 判断严重程度
            severity = self._calculate_severity(divergence, threshold)

            return DivergenceEvent(
                divergence_id=f"traj_{int(time.time())}",
                timestamp=time.time(),
                divergence_type=DivergenceType.TRAJECTORY,
                online_model_output=online_output,
                shadow_model_output=shadow_output,
                divergence_value=divergence,
                threshold=threshold,
                severity=severity
            )

        return None

    def _check_classification_divergence(
        self,
        online_output: Dict[str, Any],
        shadow_output: Dict[str, Any],
        threshold: float
    ) -> Optional[DivergenceEvent]:
        """
        检测分类分歧

        Args:
            online_output: 在线模型输出
            shadow_output: 影子模型输出
            threshold: 阈值

        Returns:
            Optional[DivergenceEvent]: 分歧事件
        """
        online_objects = online_output.get("objects", [])
        shadow_objects = shadow_output.get("objects", [])

        if not online_objects or not shadow_objects:
            return None

        # 计算分类差异
        divergence = self._calculate_classification_divergence(online_objects, shadow_objects)

        if divergence > threshold:
            # 判断严重程度
            severity = self._calculate_severity(divergence, threshold)

            return DivergenceEvent(
                divergence_id=f"class_{int(time.time())}",
                timestamp=time.time(),
                divergence_type=DivergenceType.CLASSIFICATION,
                online_model_output=online_output,
                shadow_model_output=shadow_output,
                divergence_value=divergence,
                threshold=threshold,
                severity=severity
            )

        return None

    def _check_lane_change_divergence(
        self,
        online_output: Dict[str, Any],
        shadow_output: Dict[str, Any]
    ) -> Optional[DivergenceEvent]:
        """
        检测换道分歧

        Args:
            online_output: 在线模型输出
            shadow_output: 影子模型输出

        Returns:
            Optional[DivergenceEvent]: 分歧事件
        """
        online_lane_change = online_output.get("lane_change_decision")
        shadow_lane_change = shadow_output.get("lane_change_decision")

        # 如果两者都为None，无分歧
        if online_lane_change is None and shadow_lane_change is None:
            return None

        # 如果其中一个为None，有分歧
        if online_lane_change is None or shadow_lane_change is None:
            return DivergenceEvent(
                divergence_id=f"lc_{int(time.time())}",
                timestamp=time.time(),
                divergence_type=DivergenceType.LANE_CHANGE,
                online_model_output=online_output,
                shadow_model_output=shadow_output,
                divergence_value=1.0,
                threshold=0.0,
                severity="high"
            )

        # 如果决策不同，有分歧
        if online_lane_change != shadow_lane_change:
            return DivergenceEvent(
                divergence_id=f"lc_{int(time.time())}",
                timestamp=time.time(),
                divergence_type=DivergenceType.LANE_CHANGE,
                online_model_output=online_output,
                shadow_model_output=shadow_output,
                divergence_value=1.0,
                threshold=0.0,
                severity="high"
            )

        return None

    def _check_braking_divergence(
        self,
        online_output: Dict[str, Any],
        shadow_output: Dict[str, Any]
    ) -> Optional[DivergenceEvent]:
        """
        检测刹车分歧

        Args:
            online_output: 在线模型输出
            shadow_output: 影子模型输出

        Returns:
            Optional[DivergenceEvent]: 分歧事件
        """
        online_braking = online_output.get("braking_decision")
        shadow_braking = shadow_output.get("braking_decision")

        # 如果两者都为False，无分歧
        if not online_braking and not shadow_braking:
            return None

        # 如果一个刹车一个不刹车，有分歧
        if online_braking != shadow_braking:
            severity = "critical" if (online_braking and not shadow_braking) else "high"

            return DivergenceEvent(
                divergence_id=f"brake_{int(time.time())}",
                timestamp=time.time(),
                divergence_type=DivergenceType.BRAKING,
                online_model_output=online_output,
                shadow_model_output=shadow_output,
                divergence_value=1.0,
                threshold=0.0,
                severity=severity
            )

        return None

    def _calculate_trajectory_divergence(
        self,
        traj1: List[Dict[str, float]],
        traj2: List[Dict[str, float]]
    ) -> float:
        """
        计算两条轨迹的差异

        Args:
            traj1: 轨迹1
            traj2: 轨迹2

        Returns:
            float: 差异值
        """
        max_divergence = 0.0

        for p1, p2 in zip(traj1, traj2):
            x_diff = abs(p1.get("x", 0) - p2.get("x", 0))
            y_diff = abs(p1.get("y", 0) - p2.get("y", 0))
            divergence = (x_diff ** 2 + y_diff ** 2) ** 0.5
            max_divergence = max(max_divergence, divergence)

        return max_divergence

    def _calculate_classification_divergence(
        self,
        objects1: List[Dict[str, Any]],
        objects2: List[Dict[str, Any]]
    ) -> float:
        """
        计算分类差异

        Args:
            objects1: 对象列表1
            objects2: 对象列表2

        Returns:
            float: 差异值
        """
        # 简单实现：计算类别不同的比例
        min_len = min(len(objects1), len(objects2))

        if min_len == 0:
            return 0.0

        diff_count = 0

        for obj1, obj2 in zip(objects1[:min_len], objects2[:min_len]):
            if obj1.get("type") != obj2.get("type"):
                diff_count += 1

        return diff_count / min_len

    def _calculate_severity(self, divergence: float, threshold: float) -> str:
        """
        计算严重程度

        Args:
            divergence: 分歧值
            threshold: 阈值

        Returns:
            str: 严重程度
        """
        ratio = divergence / threshold if threshold > 0 else 0

        if ratio >= 3.0:
            return "critical"
        elif ratio >= 2.0:
            return "high"
        elif ratio >= 1.5:
            return "medium"
        else:
            return "low"

    def set_divergence_callback(self, callback: Callable):
        """
        设置分歧回调函数

        Args:
            callback: 回调函数
        """
        self._divergence_callback = callback

    def get_divergence_history(
        self,
        divergence_type: Optional[DivergenceType] = None,
        severity: Optional[str] = None,
        limit: int = 100
    ) -> List[DivergenceEvent]:
        """
        获取分歧历史

        Args:
            divergence_type: 分歧类型过滤
            severity: 严重程度过滤
            limit: 返回数量限制

        Returns:
            List[DivergenceEvent]: 分歧事件列表
        """
        with self._lock:
            history = self.divergence_history.copy()

            if divergence_type is not None:
                history = [h for h in history if h.divergence_type == divergence_type]

            if severity is not None:
                history = [h for h in history if h.severity == severity]

            return history[-limit:]

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            total_divergences = len(self.divergence_history)

            # 按类型统计
            type_stats = {}
            for event in self.divergence_history:
                dtype = event.divergence_type.value
                type_stats[dtype] = type_stats.get(dtype, 0) + 1

            # 按严重程度统计
            severity_stats = {}
            for event in self.divergence_history:
                severity = event.severity
                severity_stats[severity] = severity_stats.get(severity, 0) + 1

            return {
                "total_divergences": total_divergences,
                "active_shadow_models": sum(1 for m in self.shadow_models.values() if m.enabled),
                "type_stats": type_stats,
                "severity_stats": severity_stats
            }


# 全局影子模式实例
_global_shadow_mode = None


def get_global_shadow_mode() -> ShadowMode:
    """获取全局影子模式实例"""
    global _global_shadow_mode
    if _global_shadow_mode is None:
        _global_shadow_mode = ShadowMode()
    return _global_shadow_mode