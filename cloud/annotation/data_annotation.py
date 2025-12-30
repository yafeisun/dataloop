"""
数据标注模块
管理挖掘标签和真值标签，支持标签隔离和版本管理
"""

from typing import Dict, List, Optional, Any, Set
from pydantic import BaseModel, Field
from enum import Enum
import time
import json
from dataclasses import dataclass
from datetime import datetime


class LabelType(str, Enum):
    """标签类型"""
    MINING = "mining"     # 挖掘标签
    GROUND_TRUTH = "ground_truth"  # 真值标签


class LabelStatus(str, Enum):
    """标签状态"""
    PENDING = "pending"       # 待标注
    IN_PROGRESS = "in_progress"  # 标注中
    REVIEW = "review"         # 审核中
    APPROVED = "approved"     # 已批准
    REJECTED = "rejected"     # 已拒绝


class AnnotationTask(BaseModel):
    """标注任务"""
    task_id: str = Field(description="任务ID")
    data_id: str = Field(description="数据ID")
    label_type: LabelType = Field(description="标签类型")
    status: LabelStatus = Field(default=LabelStatus.PENDING, description="标签状态")
    assigned_to: Optional[str] = Field(default=None, description="分配给谁")
    created_at: float = Field(default_factory=time.time, description="创建时间")
    updated_at: float = Field(default_factory=time.time, description="更新时间")
    completed_at: Optional[float] = Field(default=None, description="完成时间")
    labels: Dict[str, Any] = Field(default_factory=dict, description="标签内容")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    version: int = Field(default=1, description="版本号")
    parent_task_id: Optional[str] = Field(default=None, description="父任务ID")


class Label(BaseModel):
    """标签"""
    label_id: str = Field(description="标签ID")
    data_id: str = Field(description="数据ID")
    label_type: LabelType = Field(description="标签类型")
    category: str = Field(description="类别")
    attributes: Dict[str, Any] = Field(default_factory=dict, description="属性")
    confidence: float = Field(default=1.0, description="置信度")
    created_at: float = Field(default_factory=time.time, description="创建时间")
    created_by: str = Field(description="创建者")
    approved_by: Optional[str] = Field(default=None, description="批准者")
    approved_at: Optional[float] = Field(default=None, description="批准时间")
    version: int = Field(default=1, description="版本号")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


class AnnotationConfig(BaseModel):
    """标注配置"""
    config_id: str = Field(description="配置ID")
    label_type: LabelType = Field(description="标签类型")
    categories: List[str] = Field(default_factory=list, description="类别列表")
    required_attributes: List[str] = Field(default_factory=list, description="必需属性")
    optional_attributes: List[str] = Field(default_factory=list, description="可选属性")
    validation_rules: Dict[str, Any] = Field(default_factory=dict, description="验证规则")
    description: str = Field(default="", description="描述")


class DataAnnotationManager:
    """
    数据标注管理器
    管理挖掘标签和真值标签
    """

    def __init__(self):
        self.annotation_tasks: Dict[str, AnnotationTask] = {}
        self.labels: Dict[str, Label] = {}
        self.annotation_configs: Dict[str, AnnotationConfig] = {}
        self.task_counter = 0
        self.label_counter = 0

    def register_annotation_config(self, config: AnnotationConfig) -> bool:
        """
        注册标注配置

        Args:
            config: 标注配置

        Returns:
            bool: 注册是否成功
        """
        if config.config_id in self.annotation_configs:
            return False

        self.annotation_configs[config.config_id] = config
        return True

    def create_annotation_task(
        self,
        data_id: str,
        label_type: LabelType,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AnnotationTask:
        """
        创建标注任务

        Args:
            data_id: 数据ID
            label_type: 标签类型
            metadata: 元数据

        Returns:
            AnnotationTask: 标注任务
        """
        self.task_counter += 1
        task_id = f"task_{self.task_counter}"

        task = AnnotationTask(
            task_id=task_id,
            data_id=data_id,
            label_type=label_type,
            metadata=metadata or {}
        )

        self.annotation_tasks[task_id] = task
        return task

    def assign_task(self, task_id: str, assigned_to: str) -> bool:
        """
        分配任务

        Args:
            task_id: 任务ID
            assigned_to: 分配给谁

        Returns:
            bool: 分配是否成功
        """
        if task_id not in self.annotation_tasks:
            return False

        task = self.annotation_tasks[task_id]
        task.assigned_to = assigned_to
        task.status = LabelStatus.IN_PROGRESS
        task.updated_at = time.time()

        return True

    def submit_annotation(
        self,
        task_id: str,
        labels: Dict[str, Any],
        created_by: str
    ) -> bool:
        """
        提交标注

        Args:
            task_id: 任务ID
            labels: 标签内容
            created_by: 创建者

        Returns:
            bool: 提交是否成功
        """
        if task_id not in self.annotation_tasks:
            return False

        task = self.annotation_tasks[task_id]

        # 验证标签
        if not self._validate_labels(task, labels):
            return False

        # 创建标签
        self.label_counter += 1
        label_id = f"label_{self.label_counter}"

        label = Label(
            label_id=label_id,
            data_id=task.data_id,
            label_type=task.label_type,
            category=labels.get("category", ""),
            attributes=labels.get("attributes", {}),
            confidence=labels.get("confidence", 1.0),
            created_by=created_by,
            version=task.version,
            metadata=labels.get("metadata", {})
        )

        self.labels[label_id] = label

        # 更新任务状态
        task.labels = labels
        task.status = LabelStatus.REVIEW
        task.updated_at = time.time()

        return True

    def _validate_labels(self, task: AnnotationTask, labels: Dict[str, Any]) -> bool:
        """
        验证标签

        Args:
            task: 标注任务
            labels: 标签内容

        Returns:
            bool: 是否有效
        """
        # 查找对应的标注配置
        config = self._get_annotation_config(task.label_type)
        if config is None:
            return True  # 没有配置，跳过验证

        # 验证类别
        if config.categories and labels.get("category") not in config.categories:
            return False

        # 验证必需属性
        attributes = labels.get("attributes", {})
        for required_attr in config.required_attributes:
            if required_attr not in attributes:
                return False

        # 应用验证规则
        if config.validation_rules:
            return self._apply_validation_rules(attributes, config.validation_rules)

        return True

    def _get_annotation_config(self, label_type: LabelType) -> Optional[AnnotationConfig]:
        """
        获取标注配置

        Args:
            label_type: 标签类型

        Returns:
            AnnotationConfig: 标注配置
        """
        for config in self.annotation_configs.values():
            if config.label_type == label_type:
                return config
        return None

    def _apply_validation_rules(self, attributes: Dict[str, Any], rules: Dict[str, Any]) -> bool:
        """
        应用验证规则

        Args:
            attributes: 属性
            rules: 验证规则

        Returns:
            bool: 是否通过验证
        """
        for attr, rule in rules.items():
            if attr not in attributes:
                continue

            value = attributes[attr]

            # 范围验证
            if "range" in rule:
                min_val, max_val = rule["range"]
                if not (min_val <= value <= max_val):
                    return False

            # 枚举验证
            if "enum" in rule:
                if value not in rule["enum"]:
                    return False

            # 类型验证
            if "type" in rule:
                expected_type = rule["type"]
                if expected_type == "int" and not isinstance(value, int):
                    return False
                elif expected_type == "float" and not isinstance(value, (int, float)):
                    return False
                elif expected_type == "str" and not isinstance(value, str):
                    return False
                elif expected_type == "bool" and not isinstance(value, bool):
                    return False

        return True

    def approve_annotation(self, task_id: str, approved_by: str) -> bool:
        """
        批准标注

        Args:
            task_id: 任务ID
            approved_by: 批准者

        Returns:
            bool: 批准是否成功
        """
        if task_id not in self.annotation_tasks:
            return False

        task = self.annotation_tasks[task_id]
        task.status = LabelStatus.APPROVED
        task.completed_at = time.time()
        task.updated_at = time.time()

        # 更新标签的批准信息
        for label in self.labels.values():
            if label.data_id == task.data_id and label.label_type == task.label_type:
                label.approved_by = approved_by
                label.approved_at = time.time()
                break

        return True

    def reject_annotation(self, task_id: str, reason: str) -> bool:
        """
        拒绝标注

        Args:
            task_id: 任务ID
            reason: 拒绝原因

        Returns:
            bool: 拒绝是否成功
        """
        if task_id not in self.annotation_tasks:
            return False

        task = self.annotation_tasks[task_id]
        task.status = LabelStatus.REJECTED
        task.updated_at = time.time()
        task.metadata["rejection_reason"] = reason

        return True

    def create_ground_truth_from_mining(
        self,
        mining_task_id: str,
        data_id: str,
        verified_by: str
    ) -> Optional[AnnotationTask]:
        """
        从挖掘标签创建真值标签

        Args:
            mining_task_id: 挖掘任务ID
            data_id: 数据ID
            verified_by: 验证者

        Returns:
            AnnotationTask: 真值标注任务
        """
        # 查找挖掘标签
        mining_labels = [
            label for label in self.labels.values()
            if label.data_id == data_id and label.label_type == LabelType.MINING
        ]

        if not mining_labels:
            return None

        # 创建真值标注任务
        task = self.create_annotation_task(
            data_id=data_id,
            label_type=LabelType.GROUND_TRUTH,
            metadata={"mining_task_id": mining_task_id}
        )

        # 复制挖掘标签的内容
        mining_label = mining_labels[0]
        task.labels = {
            "category": mining_label.category,
            "attributes": mining_label.attributes.copy(),
            "confidence": 1.0,  # 真值标签置信度为1
            "metadata": {
                "verified_by": verified_by,
                "source": "mining_label"
            }
        }

        # 创建真值标签
        self.label_counter += 1
        label_id = f"label_{self.label_counter}"

        ground_truth_label = Label(
            label_id=label_id,
            data_id=data_id,
            label_type=LabelType.GROUND_TRUTH,
            category=mining_label.category,
            attributes=mining_label.attributes.copy(),
            confidence=1.0,
            created_by=verified_by,
            approved_by=verified_by,
            approved_at=time.time(),
            metadata={"source": "mining_label"}
        )

        self.labels[label_id] = ground_truth_label

        # 更新任务状态
        task.status = LabelStatus.APPROVED
        task.completed_at = time.time()

        return task

    def get_labels_by_data_id(self, data_id: str, label_type: Optional[LabelType] = None) -> List[Label]:
        """
        根据数据ID获取标签

        Args:
            data_id: 数据ID
            label_type: 标签类型过滤

        Returns:
            List[Label]: 标签列表
        """
        labels = [
            label for label in self.labels.values()
            if label.data_id == data_id
        ]

        if label_type:
            labels = [label for label in labels if label.label_type == label_type]

        return labels

    def get_annotation_task(self, task_id: str) -> Optional[AnnotationTask]:
        """
        获取标注任务

        Args:
            task_id: 任务ID

        Returns:
            AnnotationTask: 标注任务
        """
        return self.annotation_tasks.get(task_id)

    def get_tasks_by_status(self, status: LabelStatus) -> List[AnnotationTask]:
        """
        根据状态获取任务

        Args:
            status: 任务状态

        Returns:
            List[AnnotationTask]: 任务列表
        """
        return [
            task for task in self.annotation_tasks.values()
            if task.status == status
        ]

    def get_annotation_statistics(self) -> Dict[str, Any]:
        """
        获取标注统计

        Returns:
            Dict: 统计信息
        """
        total_tasks = len(self.annotation_tasks)
        total_labels = len(self.labels)

        # 按状态统计任务
        status_stats = {}
        for task in self.annotation_tasks.values():
            status = task.status.value
            status_stats[status] = status_stats.get(status, 0) + 1

        # 按类型统计标签
        type_stats = {}
        for label in self.labels.values():
            label_type = label.label_type.value
            type_stats[label_type] = type_stats.get(label_type, 0) + 1

        # 按类别统计标签
        category_stats = {}
        for label in self.labels.values():
            category = label.category
            category_stats[category] = category_stats.get(category, 0) + 1

        return {
            "total_tasks": total_tasks,
            "total_labels": total_labels,
            "status_stats": status_stats,
            "type_stats": type_stats,
            "category_stats": category_stats
        }

    def delete_task(self, task_id: str) -> bool:
        """
        删除任务

        Args:
            task_id: 任务ID

        Returns:
            bool: 删除是否成功
        """
        if task_id not in self.annotation_tasks:
            return False

        del self.annotation_tasks[task_id]
        return True

    def delete_label(self, label_id: str) -> bool:
        """
        删除标签

        Args:
            label_id: 标签ID

        Returns:
            bool: 删除是否成功
        """
        if label_id not in self.labels:
            return False

        del self.labels[label_id]
        return True

    def export_labels(self, output_path: str, label_type: Optional[LabelType] = None) -> bool:
        """
        导出标签

        Args:
            output_path: 输出路径
            label_type: 标签类型过滤

        Returns:
            bool: 导出是否成功
        """
        labels = list(self.labels.values())

        if label_type:
            labels = [label for label in labels if label.label_type == label_type]

        try:
            labels_dict = [label.dict() for label in labels]

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "export_time": time.time(),
                    "total_labels": len(labels_dict),
                    "labels": labels_dict
                }, f, indent=2, ensure_ascii=False)

            return True
        except Exception as e:
            print(f"Failed to export labels: {e}")
            return False

    def import_labels(self, input_path: str) -> bool:
        """
        导入标签

        Args:
            input_path: 输入路径

        Returns:
            bool: 导入是否成功
        """
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for label_dict in data.get("labels", []):
                self.label_counter += 1
                label_id = f"label_{self.label_counter}"

                label = Label(**label_dict)
                label.label_id = label_id

                self.labels[label_id] = label

            return True
        except Exception as e:
            print(f"Failed to import labels: {e}")
            return False

    def get_all_tasks(self) -> Dict[str, AnnotationTask]:
        """获取所有任务"""
        return self.annotation_tasks.copy()

    def get_all_labels(self) -> Dict[str, Label]:
        """获取所有标签"""
        return self.labels.copy()

    def get_summary(self) -> Dict[str, Any]:
        """获取摘要"""
        return self.get_annotation_statistics()


# 全局数据标注管理器实例
_global_annotation_manager = None


def get_global_annotation_manager() -> DataAnnotationManager:
    """获取全局数据标注管理器实例"""
    global _global_annotation_manager
    if _global_annotation_manager is None:
        _global_annotation_manager = DataAnnotationManager()
    return _global_annotation_manager