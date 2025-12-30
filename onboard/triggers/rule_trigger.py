"""
规则Trigger实现
基于阈值和逻辑规则的Trigger
"""

from typing import Any, Dict, List, Callable, Optional
from common.models.trigger_base import BaseTrigger, TriggerConfig, TriggerResult, TriggerType
import operator
import time


class RuleTriggerConfig(TriggerConfig):
    """规则Trigger配置"""
    trigger_type: TriggerType = TriggerType.RULE
    rules: List[Dict[str, Any]] = Field(default_factory=list)
    logic: str = "AND"  # AND, OR


class RuleTrigger(BaseTrigger):
    """
    规则Trigger
    基于阈值和逻辑规则判断是否触发
    """

    # 支持的操作符映射
    OPERATORS = {
        ">": operator.gt,
        ">=": operator.ge,
        "<": operator.lt,
        "<=": operator.le,
        "==": operator.eq,
        "!=": operator.ne,
        "in": lambda x, y: x in y,
        "not_in": lambda x, y: x not in y,
        "contains": lambda x, y: y in x if isinstance(x, (str, list)) else False,
    }

    def __init__(self, config: RuleTriggerConfig):
        super().__init__(config)
        self.rules = config.rules
        self.logic = config.logic.upper()

    def evaluate(self, data: Dict[str, Any]) -> TriggerResult:
        """
        评估数据是否触发规则

        Args:
            data: 输入数据，例如 {"acceleration": 3.5, "speed": 10}

        Returns:
            TriggerResult: 触发结果
        """
        if not self.is_enabled():
            return TriggerResult(
                triggered=False,
                trigger_id=self.config.trigger_id,
                timestamp=time.time(),
                reason="Trigger is disabled"
            )

        triggered_rules = []
        all_results = []

        for rule in self.rules:
            result = self._evaluate_rule(data, rule)
            all_results.append(result)
            if result["triggered"]:
                triggered_rules.append(result)

        # 根据逻辑判断最终结果
        if self.logic == "AND":
            triggered = len(triggered_rules) == len(self.rules)
        else:  # OR
            triggered = len(triggered_rules) > 0

        reason = self._generate_reason(triggered_rules, all_results)
        confidence = self._calculate_confidence(triggered_rules, all_results)

        return TriggerResult(
            triggered=triggered,
            trigger_id=self.config.trigger_id,
            timestamp=time.time(),
            confidence=confidence,
            reason=reason,
            data={
                "triggered_rules": triggered_rules,
                "all_results": all_results
            }
        )

    def _evaluate_rule(self, data: Dict[str, Any], rule: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估单个规则

        Args:
            data: 输入数据
            rule: 规则定义，例如 {"field": "acceleration", "op": ">", "value": 3.0}

        Returns:
            Dict: 规则评估结果
        """
        field = rule.get("field")
        op = rule.get("op", "==")
        value = rule.get("value")
        tolerance = rule.get("tolerance", 0.0)

        # 支持嵌套字段访问，如 "vehicle.acceleration"
        field_value = self._get_nested_value(data, field)

        # 应用容差
        if tolerance > 0 and op in [">", "<", ">=", "<="]:
            value = self._apply_tolerance(value, op, tolerance)

        # 执行比较
        triggered = False
        if op in self.OPERATORS:
            triggered = self.OPERATORS[op](field_value, value)
        else:
            raise ValueError(f"Unsupported operator: {op}")

        return {
            "triggered": triggered,
            "field": field,
            "op": op,
            "actual_value": field_value,
            "expected_value": value,
            "tolerance": tolerance
        }

    def _get_nested_value(self, data: Dict[str, Any], field: str) -> Any:
        """获取嵌套字段值"""
        keys = field.split(".")
        value = data
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        return value

    def _apply_tolerance(self, value: float, op: str, tolerance: float) -> float:
        """应用容差"""
        if op == ">":
            return value - tolerance
        elif op == ">=":
            return value - tolerance
        elif op == "<":
            return value + tolerance
        elif op == "<=":
            return value + tolerance
        return value

    def _generate_reason(self, triggered_rules: List[Dict], all_results: List[Dict]) -> str:
        """生成触发原因"""
        if not triggered_rules:
            return "No rules triggered"

        reasons = []
        for rule_result in triggered_rules:
            reason = f"{rule_result['field']} {rule_result['op']} {rule_result['expected_value']} (actual: {rule_result['actual_value']})"
            reasons.append(reason)

        return "; ".join(reasons)

    def _calculate_confidence(self, triggered_rules: List[Dict], all_results: List[Dict]) -> float:
        """计算置信度"""
        if not all_results:
            return 0.0

        if self.logic == "AND":
            # 所有规则都触发才置信度高
            return len(triggered_rules) / len(all_results)
        else:  # OR
            # 至少一个规则触发就置信度高
            return 1.0 if len(triggered_rules) > 0 else 0.0

    def add_rule(self, field: str, op: str, value: Any, tolerance: float = 0.0):
        """添加规则"""
        rule = {
            "field": field,
            "op": op,
            "value": value,
            "tolerance": tolerance
        }
        self.rules.append(rule)

    def remove_rule(self, index: int):
        """移除规则"""
        if 0 <= index < len(self.rules):
            self.rules.pop(index)

    def clear_rules(self):
        """清空规则"""
        self.rules.clear()

    @classmethod
    def from_yaml(cls, yaml_config: Dict[str, Any]) -> "RuleTrigger":
        """从YAML配置创建"""
        config = RuleTriggerConfig(**yaml_config)
        return cls(config)

    @classmethod
    def from_json(cls, json_config: Dict[str, Any]) -> "RuleTrigger":
        """从JSON配置创建"""
        config = RuleTriggerConfig(**json_config)
        return cls(config)


# 便捷创建函数
def create_emergency_brake_trigger(threshold: float = 3.0) -> RuleTrigger:
    """创建急刹车Trigger"""
    config = RuleTriggerConfig(
        name="emergency_brake",
        trigger_type=TriggerType.RULE,
        priority="critical",
        description="检测急刹车事件（减速度超过阈值）",
        rules=[
            {
                "field": "acceleration",
                "op": "<",
                "value": -threshold
            }
        ]
    )
    return RuleTrigger(config)


def create_sharp_turn_trigger(threshold: float = 15.0) -> RuleTrigger:
    """创建急转弯Trigger"""
    config = RuleTriggerConfig(
        name="sharp_turn",
        trigger_type=TriggerType.RULE,
        priority="critical",
        description="检测急转弯事件（横摆角速度超过阈值）",
        rules=[
            {
                "field": "yaw_rate",
                "op": ">",
                "value": threshold
            }
        ]
    )
    return RuleTrigger(config)


def create_frequent_stop_trigger(
    count_threshold: int = 3,
    time_window: float = 60.0
) -> RuleTrigger:
    """创建频繁启停Trigger"""
    config = RuleTriggerConfig(
        name="frequent_stop",
        trigger_type=TriggerType.RULE,
        priority="critical",
        description="检测频繁启停事件（指定时间内停止次数超过阈值）",
        rules=[
            {
                "field": "stop_count",
                "op": ">",
                "value": count_threshold
            },
            {
                "field": "time_window",
                "op": "<=",
                "value": time_window
            }
        ],
        logic="AND"
    )
    return RuleTrigger(config)