"""
Trigger管理器
统一管理所有Trigger，支持注册、评估、配置更新
"""

from typing import Dict, List, Optional, Any
from common.models.trigger_base import BaseTrigger, TriggerConfig, TriggerResult
from common.models.trigger_base import TriggerMetadata
import time
import json


class TriggerManager:
    """
    Trigger管理器
    负责Trigger的注册、评估、配置管理
    """

    def __init__(self):
        self.triggers: Dict[str, BaseTrigger] = {}  # trigger_id -> Trigger
        self.trigger_groups: Dict[str, List[str]] = {}  # group_name -> [trigger_ids]
        self.evaluation_history: List[Dict[str, Any]] = []
        self.max_history_size = 1000

    def register_trigger(self, trigger: BaseTrigger) -> bool:
        """
        注册Trigger

        Args:
            trigger: Trigger实例

        Returns:
            bool: 注册是否成功
        """
        trigger_id = trigger.config.trigger_id

        if trigger_id in self.triggers:
            return False

        self.triggers[trigger_id] = trigger
        return True

    def unregister_trigger(self, trigger_id: str) -> bool:
        """
        注销Trigger

        Args:
            trigger_id: Trigger ID

        Returns:
            bool: 注销是否成功
        """
        if trigger_id not in self.triggers:
            return False

        # 从所有组中移除
        for group_name in self.trigger_groups:
            if trigger_id in self.trigger_groups[group_name]:
                self.trigger_groups[group_name].remove(trigger_id)

        del self.triggers[trigger_id]
        return True

    def get_trigger(self, trigger_id: str) -> Optional[BaseTrigger]:
        """获取Trigger"""
        return self.triggers.get(trigger_id)

    def get_all_triggers(self) -> Dict[str, BaseTrigger]:
        """获取所有Trigger"""
        return self.triggers.copy()

    def enable_trigger(self, trigger_id: str) -> bool:
        """启用Trigger"""
        trigger = self.get_trigger(trigger_id)
        if trigger:
            trigger.enable()
            return True
        return False

    def disable_trigger(self, trigger_id: str) -> bool:
        """禁用Trigger"""
        trigger = self.get_trigger(trigger_id)
        if trigger:
            trigger.disable()
            return True
        return False

    def evaluate_trigger(self, trigger_id: str, data: Dict[str, Any]) -> Optional[TriggerResult]:
        """
        评估单个Trigger

        Args:
            trigger_id: Trigger ID
            data: 输入数据

        Returns:
            TriggerResult: 触发结果
        """
        trigger = self.get_trigger(trigger_id)
        if trigger is None:
            return None

        result = trigger.evaluate(data)

        # 记录历史
        self._record_evaluation(trigger_id, result)

        return result

    def evaluate_all(self, data: Dict[str, Any]) -> List[TriggerResult]:
        """
        评估所有Trigger

        Args:
            data: 输入数据

        Returns:
            List[TriggerResult]: 所有Trigger的触发结果
        """
        results = []

        for trigger_id, trigger in self.triggers.items():
            if trigger.is_enabled():
                result = trigger.evaluate(data)
                results.append(result)
                self._record_evaluation(trigger_id, result)

        return results

    def evaluate_group(self, group_name: str, data: Dict[str, Any]) -> List[TriggerResult]:
        """
        评估指定组的Trigger

        Args:
            group_name: 组名
            data: 输入数据

        Returns:
            List[TriggerResult]: 指定组Trigger的触发结果
        """
        results = []

        if group_name not in self.trigger_groups:
            return results

        for trigger_id in self.trigger_groups[group_name]:
            result = self.evaluate_trigger(trigger_id, data)
            if result:
                results.append(result)

        return results

    def create_group(self, group_name: str, trigger_ids: List[str]) -> bool:
        """
        创建Trigger组

        Args:
            group_name: 组名
            trigger_ids: Trigger ID列表

        Returns:
            bool: 创建是否成功
        """
        # 验证所有Trigger都存在
        for trigger_id in trigger_ids:
            if trigger_id not in self.triggers:
                return False

        self.trigger_groups[group_name] = trigger_ids
        return True

    def add_to_group(self, group_name: str, trigger_id: str) -> bool:
        """添加Trigger到组"""
        if trigger_id not in self.triggers:
            return False

        if group_name not in self.trigger_groups:
            self.trigger_groups[group_name] = []

        if trigger_id not in self.trigger_groups[group_name]:
            self.trigger_groups[group_name].append(trigger_id)

        return True

    def get_group(self, group_name: str) -> Optional[List[str]]:
        """获取Trigger组"""
        return self.trigger_groups.get(group_name)

    def update_trigger_config(self, trigger_id: str, **kwargs) -> bool:
        """
        更新Trigger配置

        Args:
            trigger_id: Trigger ID
            **kwargs: 配置参数

        Returns:
            bool: 更新是否成功
        """
        trigger = self.get_trigger(trigger_id)
        if trigger is None:
            return False

        trigger.update_config(**kwargs)
        return True

    def get_trigger_metadata(self, trigger_id: str) -> Optional[Dict[str, Any]]:
        """
        获取Trigger元数据（用于LLM理解）

        Args:
            trigger_id: Trigger ID

        Returns:
            Dict: 元数据
        """
        trigger = self.get_trigger(trigger_id)
        if trigger is None:
            return None

        metadata = TriggerMetadata(trigger)
        return metadata.to_dict()

    def export_triggers_to_json(self, file_path: str) -> bool:
        """
        导出所有Trigger配置到JSON文件

        Args:
            file_path: 文件路径

        Returns:
            bool: 导出是否成功
        """
        try:
            trigger_configs = []
            for trigger in self.triggers.values():
                config_dict = trigger.config.dict()
                trigger_configs.append(config_dict)

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "triggers": trigger_configs,
                    "groups": self.trigger_groups
                }, f, indent=2, ensure_ascii=False)

            return True
        except Exception as e:
            print(f"Export failed: {e}")
            return False

    def import_triggers_from_json(self, file_path: str) -> bool:
        """
        从JSON文件导入Trigger配置

        Args:
            file_path: 文件路径

        Returns:
            bool: 导入是否成功
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 导入组
            if "groups" in data:
                self.trigger_groups = data["groups"]

            return True
        except Exception as e:
            print(f"Import failed: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_triggers = len(self.triggers)
        enabled_triggers = sum(1 for t in self.triggers.values() if t.is_enabled())

        # 按优先级统计
        priority_stats = {}
        for trigger in self.triggers.values():
            priority = trigger.config.priority.value
            priority_stats[priority] = priority_stats.get(priority, 0) + 1

        # 按类型统计
        type_stats = {}
        for trigger in self.triggers.values():
            trigger_type = trigger.config.trigger_type.value
            type_stats[trigger_type] = type_stats.get(trigger_type, 0) + 1

        return {
            "total_triggers": total_triggers,
            "enabled_triggers": enabled_triggers,
            "disabled_triggers": total_triggers - enabled_triggers,
            "priority_stats": priority_stats,
            "type_stats": type_stats,
            "groups": list(self.trigger_groups.keys()),
            "evaluation_history_size": len(self.evaluation_history)
        }

    def _record_evaluation(self, trigger_id: str, result: TriggerResult):
        """记录评估历史"""
        record = {
            "trigger_id": trigger_id,
            "timestamp": result.timestamp,
            "triggered": result.triggered,
            "confidence": result.confidence,
            "reason": result.reason
        }

        self.evaluation_history.append(record)

        # 限制历史大小
        if len(self.evaluation_history) > self.max_history_size:
            self.evaluation_history = self.evaluation_history[-self.max_history_size:]

    def clear_history(self):
        """清空评估历史"""
        self.evaluation_history.clear()

    def get_triggered_triggers(self, data: Dict[str, Any]) -> List[str]:
        """
        获取所有触发的Trigger ID

        Args:
            data: 输入数据

        Returns:
            List[str]: 触发的Trigger ID列表
        """
        results = self.evaluate_all(data)
        return [r.trigger_id for r in results if r.triggered]

    def __len__(self) -> int:
        return len(self.triggers)

    def __repr__(self) -> str:
        return f"TriggerManager(triggers={len(self.triggers)}, groups={len(self.trigger_groups)})"


# 全局Trigger管理器实例
_global_trigger_manager = None


def get_global_trigger_manager() -> TriggerManager:
    """获取全局Trigger管理器实例"""
    global _global_trigger_manager
    if _global_trigger_manager is None:
        _global_trigger_manager = TriggerManager()
    return _global_trigger_manager