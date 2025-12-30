"""
LLM问题诊断模块
基于时序事件序列建模，实现语义对齐和弱监督学习
"""

from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field
from enum import Enum
import time
import json
from dataclasses import dataclass


class DiagnosisStatus(str, Enum):
    """诊断状态"""
    PENDING = "pending"     # 待处理
    RUNNING = "running"     # 运行中
    COMPLETED = "completed" # 已完成
    FAILED = "failed"       # 失败


class ProblemType(str, Enum):
    """问题类型"""
    PERCEPTION = "perception"       # 感知问题
    PREDICTION = "prediction"       # 预测问题
    PLANNING = "planning"           # 规划问题
    CONTROL = "control"             # 控制问题
    SENSOR = "sensor"               # 传感器问题
    ALGORITHM = "algorithm"         # 算法问题
    UNKNOWN = "unknown"             # 未知问题


class ProblemSeverity(str, Enum):
    """问题严重程度"""
    CRITICAL = "critical"   # 严重
    HIGH = "high"           # 高
    MEDIUM = "medium"       # 中等
    LOW = "low"             # 低


class DiagnosisConfig(BaseModel):
    """诊断配置"""
    diagnosis_id: str = Field(description="诊断任务ID")
    name: str = Field(description="诊断任务名称")
    description: str = Field(default="", description="描述")
    llm_model: str = Field(default="gpt-4", description="LLM模型")
    llm_api_key: str = Field(default="", description="LLM API密钥")
    llm_temperature: float = Field(default=0.7, description="LLM温度参数")
    llm_max_tokens: int = Field(default=2000, description="LLM最大token数")
    event_sequence: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="时序事件序列"
    )
    trigger_metadata: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Trigger元数据"
    )
    enable_weak_supervision: bool = Field(default=True, description="是否启用弱监督学习")
    enable_online_learning: bool = Field(default=True, description="是否启用在线学习")


class DiagnosisResult(BaseModel):
    """诊断结果"""
    diagnosis_id: str
    timestamp: float
    problem_type: ProblemType
    problem_severity: ProblemSeverity
    root_cause: str = Field(description="根本原因")
    description: str = Field(description="问题描述")
    confidence: float = Field(description="置信度")
    suggested_actions: List[str] = Field(default_factory=list, description="建议行动")
    related_events: List[str] = Field(default_factory=list, description="相关事件")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EventSequence(BaseModel):
    """时序事件序列"""
    events: List[Dict[str, Any]] = Field(default_factory=list)
    start_time: float = Field(description="起始时间")
    end_time: float = Field(description="结束时间")
    duration: float = Field(description="持续时间")


class LLMDiagnosisEngine:
    """
    LLM诊断引擎
    基于时序事件序列进行问题诊断
    """

    def __init__(self, config: DiagnosisConfig):
        self.config = config
        self.llm_client = None
        self._init_llm_client()

    def _init_llm_client(self):
        """初始化LLM客户端"""
        try:
            from openai import OpenAI

            if self.config.llm_api_key:
                self.llm_client = OpenAI(api_key=self.config.llm_api_key)
            else:
                # 使用环境变量中的API密钥
                self.llm_client = OpenAI()
        except ImportError:
            print("OpenAI library not installed")
            self.llm_client = None

    def diagnose(self, event_sequence: EventSequence) -> Optional[DiagnosisResult]:
        """
        诊断问题

        Args:
            event_sequence: 时序事件序列

        Returns:
            DiagnosisResult: 诊断结果
        """
        if self.llm_client is None:
            return None

        # 构建提示词
        prompt = self._build_diagnosis_prompt(event_sequence)

        # 调用LLM
        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个自动驾驶系统问题诊断专家。请根据时序事件序列分析问题根源，给出诊断结果。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens
            )

            # 解析响应
            result_text = response.choices[0].message.content
            return self._parse_diagnosis_result(result_text)

        except Exception as e:
            print(f"LLM diagnosis error: {e}")
            return None

    def _build_diagnosis_prompt(self, event_sequence: EventSequence) -> str:
        """
        构建诊断提示词

        Args:
            event_sequence: 时序事件序列

        Returns:
            str: 提示词
        """
        # 将事件序列转换为文本描述
        events_text = self._events_to_text(event_sequence.events)

        # 添加Trigger元数据
        trigger_metadata_text = self._trigger_metadata_to_text(self.config.trigger_metadata)

        prompt = f"""
请分析以下自动驾驶系统的时序事件序列，诊断问题根源：

## 事件序列
时间范围: {event_sequence.start_time} - {event_sequence.end_time}
持续时间: {event_sequence.duration:.2f}秒

事件详情:
{events_text}

## Trigger元数据
{trigger_metadata_text}

## 诊断要求
请以JSON格式返回诊断结果，包含以下字段：
{{
  "problem_type": "问题类型(perception/prediction/planning/control/sensor/algorithm/unknown)",
  "problem_severity": "问题严重程度(critical/high/medium/low)",
  "root_cause": "根本原因描述",
  "description": "问题描述",
  "confidence": "置信度(0-1之间的浮点数)",
  "suggested_actions": ["建议行动1", "建议行动2", ...],
  "related_events": ["相关事件ID1", "相关事件ID2", ...]
}}

请基于事件序列的时序关系和Trigger触发情况，准确诊断问题根源。
"""

        return prompt

    def _events_to_text(self, events: List[Dict[str, Any]]) -> str:
        """
        将事件转换为文本描述

        Args:
            events: 事件列表

        Returns:
            str: 文本描述
        """
        text_lines = []

        for event in events:
            timestamp = event.get("timestamp", 0)
            event_type = event.get("event_type", "unknown")
            object_id = event.get("object_id", "")
            attributes = event.get("attributes", {})

            # 构建事件描述
            event_desc = f"[{timestamp:.3f}s] {event_type}"
            if object_id:
                event_desc += f" (对象: {object_id})"

            # 添加属性
            if attributes:
                attr_list = []
                for key, value in attributes.items():
                    attr_list.append(f"{key}={value}")
                event_desc += f" - {', '.join(attr_list)}"

            text_lines.append(event_desc)

        return "\n".join(text_lines)

    def _trigger_metadata_to_text(self, trigger_metadata: Dict[str, Dict[str, Any]]) -> str:
        """
        将Trigger元数据转换为文本

        Args:
            trigger_metadata: Trigger元数据

        Returns:
            str: 文本描述
        """
        text_lines = []

        for trigger_id, metadata in trigger_metadata.items():
            text_lines.append(f"Trigger: {metadata.get('name', trigger_id)}")
            text_lines.append(f"  类型: {metadata.get('type', 'unknown')}")
            text_lines.append(f"  优先级: {metadata.get('priority', 'unknown')}")
            text_lines.append(f"  描述: {metadata.get('description', '')}")
            text_lines.append("")

        return "\n".join(text_lines)

    def _parse_diagnosis_result(self, result_text: str) -> Optional[DiagnosisResult]:
        """
        解析诊断结果

        Args:
            result_text: LLM返回的文本

        Returns:
            DiagnosisResult: 诊断结果
        """
        try:
            # 尝试提取JSON
            import re

            # 查找JSON块
            json_match = re.search(r'\{[\s\S]*\}', result_text)
            if json_match:
                json_str = json_match.group()
                result_dict = json.loads(json_str)

                # 创建诊断结果
                return DiagnosisResult(
                    diagnosis_id=self.config.diagnosis_id,
                    timestamp=time.time(),
                    problem_type=ProblemType(result_dict.get("problem_type", "unknown")),
                    problem_severity=ProblemSeverity(result_dict.get("problem_severity", "medium")),
                    root_cause=result_dict.get("root_cause", ""),
                    description=result_dict.get("description", ""),
                    confidence=float(result_dict.get("confidence", 0.5)),
                    suggested_actions=result_dict.get("suggested_actions", []),
                    related_events=result_dict.get("related_events", []),
                    metadata={"raw_response": result_text}
                )

        except Exception as e:
            print(f"Failed to parse diagnosis result: {e}")

        return None


class WeakSupervisionLearner:
    """
    弱监督学习器
    基于LLM诊断结果进行在线学习
    """

    def __init__(self):
        self.diagnosis_history: List[DiagnosisResult] = []
        self.problem_patterns: Dict[str, List[DiagnosisResult]] = {}
        self.confidence_threshold = 0.7

    def add_diagnosis(self, result: DiagnosisResult):
        """
        添加诊断结果

        Args:
            result: 诊断结果
        """
        self.diagnosis_history.append(result)

        # 按问题类型分组
        problem_type = result.problem_type.value
        if problem_type not in self.problem_patterns:
            self.problem_patterns[problem_type] = []
        self.problem_patterns[problem_type].append(result)

    def learn_from_diagnosis(self, result: DiagnosisResult) -> Dict[str, Any]:
        """
        从诊断结果中学习

        Args:
            result: 诊断结果

        Returns:
            Dict: 学习结果
        """
        if result.confidence < self.confidence_threshold:
            return {"status": "low_confidence", "message": "置信度过低，不进行学习"}

        # 统计问题模式
        problem_type = result.problem_type.value
        similar_cases = self.problem_patterns.get(problem_type, [])

        learning_result = {
            "problem_type": problem_type,
            "total_cases": len(similar_cases),
            "learning_patterns": []
        }

        # 提取学习模式
        if len(similar_cases) >= 3:
            # 统计常见根本原因
            root_causes = [case.root_cause for case in similar_cases]
            from collections import Counter
            common_causes = Counter(root_causes).most_common(3)

            learning_result["learning_patterns"] = [
                {"root_cause": cause, "count": count}
                for cause, count in common_causes
            ]

        return learning_result

    def get_problem_statistics(self) -> Dict[str, Any]:
        """
        获取问题统计

        Returns:
            Dict: 问题统计
        """
        total_diagnoses = len(self.diagnosis_history)

        # 按类型统计
        type_stats = {}
        for result in self.diagnosis_history:
            problem_type = result.problem_type.value
            type_stats[problem_type] = type_stats.get(problem_type, 0) + 1

        # 按严重程度统计
        severity_stats = {}
        for result in self.diagnosis_history:
            severity = result.problem_severity.value
            severity_stats[severity] = severity_stats.get(severity, 0) + 1

        return {
            "total_diagnoses": total_diagnoses,
            "type_stats": type_stats,
            "severity_stats": severity_stats,
            "problem_patterns": {
                problem_type: len(cases)
                for problem_type, cases in self.problem_patterns.items()
            }
        }


class DiagnosisManager:
    """
    诊断管理器
    管理诊断任务和结果
    """

    def __init__(self):
        self.diagnosis_configs: Dict[str, DiagnosisConfig] = {}
        self.diagnosis_status: Dict[str, DiagnosisStatus] = {}
        self.diagnosis_results: Dict[str, DiagnosisResult] = {}
        self.weak_supervision_learner = WeakSupervisionLearner()

    def create_diagnosis_task(self, config: DiagnosisConfig) -> bool:
        """
        创建诊断任务

        Args:
            config: 诊断配置

        Returns:
            bool: 创建是否成功
        """
        if config.diagnosis_id in self.diagnosis_configs:
            return False

        self.diagnosis_configs[config.diagnosis_id] = config
        self.diagnosis_status[config.diagnosis_id] = DiagnosisStatus.PENDING

        return True

    def run_diagnosis(self, diagnosis_id: str, event_sequence: EventSequence) -> bool:
        """
        运行诊断任务

        Args:
            diagnosis_id: 诊断任务ID
            event_sequence: 时序事件序列

        Returns:
            bool: 运行是否成功
        """
        if diagnosis_id not in self.diagnosis_configs:
            return False

        config = self.diagnosis_configs[diagnosis_id]

        # 更新状态
        self.diagnosis_status[diagnosis_id] = DiagnosisStatus.RUNNING

        try:
            # 创建诊断引擎
            engine = LLMDiagnosisEngine(config)

            # 运行诊断
            result = engine.diagnose(event_sequence)

            if result:
                self.diagnosis_results[diagnosis_id] = result
                self.diagnosis_status[diagnosis_id] = DiagnosisStatus.COMPLETED

                # 弱监督学习
                if config.enable_weak_supervision:
                    self.weak_supervision_learner.add_diagnosis(result)

                return True
            else:
                self.diagnosis_status[diagnosis_id] = DiagnosisStatus.FAILED
                return False

        except Exception as e:
            print(f"Diagnosis error: {e}")
            self.diagnosis_status[diagnosis_id] = DiagnosisStatus.FAILED
            return False

    def get_diagnosis_result(self, diagnosis_id: str) -> Optional[DiagnosisResult]:
        """
        获取诊断结果

        Args:
            diagnosis_id: 诊断任务ID

        Returns:
            DiagnosisResult: 诊断结果
        """
        return self.diagnosis_results.get(diagnosis_id)

    def get_diagnosis_status(self, diagnosis_id: str) -> Optional[DiagnosisStatus]:
        """
        获取诊断状态

        Args:
            diagnosis_id: 诊断任务ID

        Returns:
            DiagnosisStatus: 诊断状态
        """
        return self.diagnosis_status.get(diagnosis_id)

    def get_problem_statistics(self) -> Dict[str, Any]:
        """获取问题统计"""
        return self.weak_supervision_learner.get_problem_statistics()

    def delete_diagnosis_task(self, diagnosis_id: str) -> bool:
        """
        删除诊断任务

        Args:
            diagnosis_id: 诊断任务ID

        Returns:
            bool: 删除是否成功
        """
        if diagnosis_id in self.diagnosis_configs:
            del self.diagnosis_configs[diagnosis_id]
        if diagnosis_id in self.diagnosis_status:
            del self.diagnosis_status[diagnosis_id]
        if diagnosis_id in self.diagnosis_results:
            del self.diagnosis_results[diagnosis_id]

        return True

    def get_all_diagnosis_tasks(self) -> Dict[str, DiagnosisConfig]:
        """获取所有诊断任务"""
        return self.diagnosis_configs.copy()

    def get_summary(self) -> Dict[str, Any]:
        """获取摘要"""
        total_tasks = len(self.diagnosis_configs)
        running_tasks = sum(
            1 for status in self.diagnosis_status.values()
            if status == DiagnosisStatus.RUNNING
        )
        completed_tasks = sum(
            1 for status in self.diagnosis_status.values()
            if status == DiagnosisStatus.COMPLETED
        )
        failed_tasks = sum(
            1 for status in self.diagnosis_status.values()
            if status == DiagnosisStatus.FAILED
        )

        return {
            "total_tasks": total_tasks,
            "running_tasks": running_tasks,
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "pending_tasks": total_tasks - running_tasks - completed_tasks - failed_tasks
        }


# 全局诊断管理器实例
_global_diagnosis_manager = None


def get_global_diagnosis_manager() -> DiagnosisManager:
    """获取全局诊断管理器实例"""
    global _global_diagnosis_manager
    if _global_diagnosis_manager is None:
        _global_diagnosis_manager = DiagnosisManager()
    return _global_diagnosis_manager