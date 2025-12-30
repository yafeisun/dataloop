"""
组合Trigger实现
支持多个Trigger的组合逻辑
"""

from typing import Any, Dict, List
from common.models.trigger_base import (
    BaseTrigger, TriggerConfig, TriggerResult, TriggerType, TriggerPriority
)
import time


class CompositeTriggerConfig(TriggerConfig):
    """组合Trigger配置"""
    trigger_type: TriggerType = TriggerType.COMPOSITE
    triggers: List[str] = Field(default_factory=list)  # 子Trigger ID列表
    logic: str = "AND"  # AND, OR, NOT


class CompositeTrigger(BaseTrigger):
    """
    组合Trigger
    支持多个Trigger的组合逻辑
    """

    def __init__(self, config: CompositeTriggerConfig, trigger_manager=None):
        super().__init__(config)
        self.trigger_ids = config.triggers
        self.logic = config.logic.upper()
        self.trigger_manager = trigger_manager  # Trigger管理器引用

    def evaluate(self, data: Dict[str, Any]) -> TriggerResult:
        """
        评估数据是否触发组合Trigger

        Args:
            data: 输入数据

        Returns:
            TriggerResult: 触发结果
        """
        if not self.is_enabled():
            return TriggerResult(
                triggered=False,
                trigger_id=self.config.trigger_id,
                timestamp=time.time(),
                reason="Composite trigger is disabled"
            )

        # 获取所有子Trigger
        child_triggers = self._get_child_triggers()
        if not child_triggers:
            return TriggerResult(
                triggered=False,
                trigger_id=self.config.trigger_id,
                timestamp=time.time(),
                reason="No child triggers found"
            )

        # 评估所有子Trigger
        child_results = []
        for trigger in child_triggers:
            result = trigger.evaluate(data)
            child_results.append(result)

        # 根据逻辑判断最终结果
        triggered = self._evaluate_logic(child_results)
        reason = self._generate_reason(child_results)
        confidence = self._calculate_confidence(child_results)

        return TriggerResult(
            triggered=triggered,
            trigger_id=self.config.trigger_id,
            timestamp=time.time(),
            confidence=confidence,
            reason=reason,
            data={
                "child_results": child_results,
                "logic": self.logic
            }
        )

    def _get_child_triggers(self) -> List[BaseTrigger]:
        """获取子Trigger列表"""
        if self.trigger_manager is None:
            return []

        triggers = []
        for trigger_id in self.trigger_ids:
            trigger = self.trigger_manager.get_trigger(trigger_id)
            if trigger is not None:
                triggers.append(trigger)

        return triggers

    def _evaluate_logic(self, child_results: List[TriggerResult]) -> bool:
        """根据逻辑判断结果"""
        triggered_results = [r for r in child_results if r.triggered]

        if self.logic == "AND":
            # 所有子Trigger都触发
            return len(triggered_results) == len(child_results)
        elif self.logic == "OR":
            # 至少一个子Trigger触发
            return len(triggered_results) > 0
        elif self.logic == "NOT":
            # 所有子Trigger都不触发
            return len(triggered_results) == 0
        else:
            raise ValueError(f"Unsupported logic: {self.logic}")

    def _generate_reason(self, child_results: List[TriggerResult]) -> str:
        """生成触发原因"""
        triggered_results = [r for r in child_results if r.triggered]

        if not triggered_results:
            return "No child triggers triggered"

        reasons = []
        for result in triggered_results:
            reasons.append(f"Trigger {result.trigger_id}: {result.reason}")

        return "; ".join(reasons)

    def _calculate_confidence(self, child_results: List[TriggerResult]) -> float:
        """计算置信度"""
        if not child_results:
            return 0.0

        # 取所有子Trigger置信度的平均值
        confidences = [r.confidence for r in child_results]
        return sum(confidences) / len(confidences)

    def add_child_trigger(self, trigger_id: str):
        """添加子Trigger"""
        if trigger_id not in self.trigger_ids:
            self.trigger_ids.append(trigger_id)

    def remove_child_trigger(self, trigger_id: str):
        """移除子Trigger"""
        if trigger_id in self.trigger_ids:
            self.trigger_ids.remove(trigger_id)

    def clear_child_triggers(self):
        """清空子Trigger"""
        self.trigger_ids.clear()

    @classmethod
    def create_and_trigger(
        cls,
        name: str,
        trigger_ids: List[str],
        trigger_manager=None
    ) -> "CompositeTrigger":
        """创建AND逻辑的组合Trigger"""
        config = CompositeTriggerConfig(
            name=name,
            trigger_type=TriggerType.COMPOSITE,
            priority=TriggerPriority.HIGH,
            description=f"AND composite trigger: {', '.join(trigger_ids)}",
            triggers=trigger_ids,
            logic="AND"
        )
        return cls(config, trigger_manager)

    @classmethod
    def create_or_trigger(
        cls,
        name: str,
        trigger_ids: List[str],
        trigger_manager=None
    ) -> "CompositeTrigger":
        """创建OR逻辑的组合Trigger"""
        config = CompositeTriggerConfig(
            name=name,
            trigger_type=TriggerType.COMPOSITE,
            priority=TriggerPriority.HIGH,
            description=f"OR composite trigger: {', '.join(trigger_ids)}",
            triggers=trigger_ids,
            logic="OR"
        )
        return cls(config, trigger_manager)


class SequentialTrigger(BaseTrigger):
    """
    时序Trigger
    检测事件序列
    """

    def __init__(self, config: TriggerConfig, sequence: List[str], max_duration: float = 10.0):
        super().__init__(config)
        self.sequence = sequence  # 期望的Trigger序列
        self.max_duration = max_duration  # 最大持续时间（秒）
        self.current_sequence = []  # 当前检测到的序列
        self.start_time = None

    def evaluate(self, data: Dict[str, Any]) -> TriggerResult:
        """
        评估数据是否触发时序Trigger

        Args:
            data: 输入数据，包含triggered_triggers列表

        Returns:
            TriggerResult: 触发结果
        """
        triggered_triggers = data.get("triggered_triggers", [])

        for trigger_id in triggered_triggers:
            self._add_to_sequence(trigger_id)

        # 检查是否匹配期望序列
        triggered = self._check_sequence()
        reason = self._generate_reason()

        if triggered:
            self._reset_sequence()

        return TriggerResult(
            triggered=triggered,
            trigger_id=self.config.trigger_id,
            timestamp=time.time(),
            confidence=1.0 if triggered else 0.0,
            reason=reason,
            data={
                "current_sequence": self.current_sequence,
                "expected_sequence": self.sequence
            }
        )

    def _add_to_sequence(self, trigger_id: str):
        """添加到当前序列"""
        current_time = time.time()

        if self.start_time is None:
            self.start_time = current_time
        elif current_time - self.start_time > self.max_duration:
            # 超时，重置序列
            self._reset_sequence()
            self.start_time = current_time

        self.current_sequence.append(trigger_id)

    def _check_sequence(self) -> bool:
        """检查序列是否匹配"""
        if len(self.current_sequence) < len(self.sequence):
            return False

        # 检查最后N个元素是否匹配
        start_idx = len(self.current_sequence) - len(self.sequence)
        actual_sequence = self.current_sequence[start_idx:]

        return actual_sequence == self.sequence

    def _generate_reason(self) -> str:
        """生成触发原因"""
        if not self.current_sequence:
            return "No sequence detected"

        return f"Current sequence: {self.current_sequence}, Expected: {self.sequence}"

    def _reset_sequence(self):
        """重置序列"""
        self.current_sequence = []
        self.start_time = None


# 便捷创建函数
def create_cut_in_emergency_brake_trigger(trigger_manager=None) -> CompositeTrigger:
    """创建变道后急刹车组合Trigger"""
    return CompositeTrigger.create_or_trigger(
        name="cut_in_emergency_brake",
        trigger_ids=["cut_in", "emergency_brake"],
        trigger_manager=trigger_manager
    )


def create_lane_change_failure_trigger(trigger_manager=None) -> SequentialTrigger:
    """创建换道失败时序Trigger"""
    config = TriggerConfig(
        name="lane_change_failure",
        trigger_type=TriggerType.COMPOSITE,
        priority=TriggerPriority.HIGH,
        description="检测换道失败事件序列"
    )
    return SequentialTrigger(
        config=config,
        sequence=["lane_change_start", "lane_change_abort"],
        max_duration=5.0
    )