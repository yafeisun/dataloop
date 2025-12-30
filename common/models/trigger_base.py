"""
Trigger基类定义
支持车端/云端/仿真三端代码复用
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from enum import Enum
import uuid


class TriggerType(str, Enum):
    """Trigger类型枚举"""
    RULE = "rule"           # 规则Trigger
    MODEL = "model"         # 模型Trigger
    COMPOSITE = "composite" # 组合Trigger


class TriggerPriority(str, Enum):
    """Trigger优先级"""
    CRITICAL = "critical"   # 一级指标（严重体感）
    HIGH = "high"           # 二级指标（轻微体感）
    MEDIUM = "medium"       # 三级指标（技术指标）
    LOW = "low"             # 其他


class TriggerConfig(BaseModel):
    """Trigger配置基类"""
    trigger_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    trigger_type: TriggerType
    priority: TriggerPriority
    enabled: bool = True
    description: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TriggerResult(BaseModel):
    """Trigger执行结果"""
    triggered: bool
    trigger_id: str
    timestamp: float
    confidence: float = 1.0
    reason: str = ""
    data: Dict[str, Any] = Field(default_factory=dict)


class BaseTrigger(ABC):
    """
    Trigger抽象基类
    所有Trigger必须实现evaluate方法
    """

    def __init__(self, config: TriggerConfig):
        self.config = config
        self._validate_config()

    def _validate_config(self):
        """验证配置"""
        if not self.config.name:
            raise ValueError("Trigger name cannot be empty")

    @abstractmethod
    def evaluate(self, data: Dict[str, Any]) -> TriggerResult:
        """
        评估数据是否触发Trigger

        Args:
            data: 输入数据，包含传感器数据、状态机信息等

        Returns:
            TriggerResult: 触发结果
        """
        pass

    def get_config(self) -> TriggerConfig:
        """获取配置"""
        return self.config

    def update_config(self, **kwargs):
        """更新配置"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

    def is_enabled(self) -> bool:
        """检查是否启用"""
        return self.config.enabled

    def enable(self):
        """启用Trigger"""
        self.config.enabled = True

    def disable(self):
        """禁用Trigger"""
        self.config.enabled = False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.config.name}, type={self.config.trigger_type})"


class TriggerMetadata:
    """
    Trigger元数据
    用于生成文档，供LLM理解
    """

    def __init__(self, trigger: BaseTrigger):
        self.trigger = trigger

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "trigger_id": self.trigger.config.trigger_id,
            "name": self.trigger.config.name,
            "type": self.trigger.config.trigger_type.value,
            "priority": self.trigger.config.priority.value,
            "description": self.trigger.config.description,
            "input_schema": self._get_input_schema(),
            "output_schema": self._get_output_schema(),
            "metadata": self.trigger.config.metadata
        }

    def _get_input_schema(self) -> Dict[str, Any]:
        """获取输入数据模式（子类可重写）"""
        return {"type": "object", "description": "通用输入数据"}

    def _get_output_schema(self) -> Dict[str, Any]:
        """获取输出数据模式"""
        return {
            "type": "object",
            "properties": {
                "triggered": {"type": "boolean"},
                "confidence": {"type": "number"},
                "reason": {"type": "string"}
            }
        }

    def to_llm_prompt(self) -> str:
        """转换为LLM可理解的提示词"""
        metadata = self.to_dict()
        prompt = f"""
Trigger: {metadata['name']}
Type: {metadata['type']}
Priority: {metadata['priority']}
Description: {metadata['description']}

Input Schema:
{metadata['input_schema']}

Output Schema:
{metadata['output_schema']}
"""
        return prompt.strip()