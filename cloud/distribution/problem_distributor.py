"""
问题分发模块
实现智能映射团队、优先级排序、任务创建
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum
import time
from dataclasses import dataclass
from datetime import datetime, timedelta


class TaskStatus(str, Enum):
    """任务状态"""
    PENDING = "pending"       # 待分配
    ASSIGNED = "assigned"     # 已分配
    IN_PROGRESS = "in_progress"  # 进行中
    REVIEW = "review"         # 审核中
    COMPLETED = "completed"   # 已完成
    FAILED = "failed"         # 失败
    CANCELLED = "cancelled"   # 已取消


class TaskPriority(str, Enum):
    """任务优先级"""
    CRITICAL = "critical"     # 紧急
    HIGH = "high"             # 高
    MEDIUM = "medium"         # 中等
    LOW = "low"               # 低


class TeamType(str, Enum):
    """团队类型"""
    PERCEPTION = "perception"     # 感知团队
    PREDICTION = "prediction"     # 预测团队
    PLANNING = "planning"         # 规划团队
    CONTROL = "control"           # 控制团队
    SENSOR = "sensor"             # 传感器团队
    ALGORITHM = "algorithm"       # 算法团队
    GENERAL = "general"           # 通用团队


class Team(BaseModel):
    """团队信息"""
    team_id: str = Field(description="团队ID")
    name: str = Field(description="团队名称")
    team_type: TeamType = Field(description="团队类型")
    members: List[str] = Field(default_factory=list, description="成员列表")
    max_concurrent_tasks: int = Field(default=5, description="最大并发任务数")
    current_tasks: int = Field(default=0, description="当前任务数")
    contact: str = Field(default="", description="联系方式")
    description: str = Field(default="", description="描述")


class ProblemTask(BaseModel):
    """问题任务"""
    task_id: str = Field(description="任务ID")
    problem_id: str = Field(description="问题ID")
    title: str = Field(description="任务标题")
    description: str = Field(description="问题描述")
    problem_type: str = Field(description="问题类型")
    problem_severity: str = Field(description="问题严重程度")
    root_cause: str = Field(description="根本原因")
    suggested_actions: List[str] = Field(default_factory=list, description="建议行动")
    assigned_team_id: Optional[str] = Field(default=None, description="分配的团队ID")
    assigned_to: Optional[str] = Field(default=None, description="分配给谁")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="任务状态")
    priority: TaskPriority = Field(default=TaskPriority.MEDIUM, description="任务优先级")
    created_at: float = Field(default_factory=time.time, description="创建时间")
    assigned_at: Optional[float] = Field(default=None, description="分配时间")
    started_at: Optional[float] = Field(default=None, description="开始时间")
    completed_at: Optional[float] = Field(default=None, description="完成时间")
    estimated_hours: float = Field(default=8.0, description="预估工时")
    actual_hours: float = Field(default=0.0, description="实际工时")
    related_data_ids: List[str] = Field(default_factory=list, description="相关数据ID")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


class DistributionRule(BaseModel):
    """分发规则"""
    rule_id: str = Field(description="规则ID")
    name: str = Field(description="规则名称")
    problem_type: str = Field(description="问题类型")
    problem_severity: str = Field(description="问题严重程度")
    team_type: TeamType = Field(description="团队类型")
    priority_mapping: Dict[str, TaskPriority] = Field(
        default_factory=dict,
        description="优先级映射"
    )
    enabled: bool = Field(default=True, description="是否启用")


class ProblemDistributor:
    """
    问题分发器
    负责问题的智能分发
    """

    def __init__(self):
        self.teams: Dict[str, Team] = {}
        self.tasks: Dict[str, ProblemTask] = {}
        self.distribution_rules: List[DistributionRule] = []
        self.problem_counter = 0
        self.task_counter = 0

    def register_team(self, team: Team) -> bool:
        """
        注册团队

        Args:
            team: 团队信息

        Returns:
            bool: 注册是否成功
        """
        if team.team_id in self.teams:
            return False

        self.teams[team.team_id] = team
        return True

    def unregister_team(self, team_id: str) -> bool:
        """
        注销团队

        Args:
            team_id: 团队ID

        Returns:
            bool: 注销是否成功
        """
        if team_id not in self.teams:
            return False

        del self.teams[team_id]
        return True

    def add_distribution_rule(self, rule: DistributionRule):
        """
        添加分发规则

        Args:
            rule: 分发规则
        """
        self.distribution_rules.append(rule)

    def create_problem_task(
        self,
        title: str,
        description: str,
        problem_type: str,
        problem_severity: str,
        root_cause: str,
        suggested_actions: List[str],
        related_data_ids: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> ProblemTask:
        """
        创建问题任务

        Args:
            title: 任务标题
            description: 问题描述
            problem_type: 问题类型
            problem_severity: 问题严重程度
            root_cause: 根本原因
            suggested_actions: 建议行动
            related_data_ids: 相关数据ID
            metadata: 元数据

        Returns:
            ProblemTask: 问题任务
        """
        self.problem_counter += 1
        self.task_counter += 1

        problem_id = f"problem_{self.problem_counter}"
        task_id = f"task_{self.task_counter}"

        # 自动确定优先级
        priority = self._determine_priority(problem_type, problem_severity)

        task = ProblemTask(
            task_id=task_id,
            problem_id=problem_id,
            title=title,
            description=description,
            problem_type=problem_type,
            problem_severity=problem_severity,
            root_cause=root_cause,
            suggested_actions=suggested_actions,
            priority=priority,
            related_data_ids=related_data_ids,
            metadata=metadata or {}
        )

        self.tasks[task_id] = task
        return task

    def distribute_task(self, task_id: str) -> bool:
        """
        分发任务

        Args:
            task_id: 任务ID

        Returns:
            bool: 分发是否成功
        """
        if task_id not in self.tasks:
            return False

        task = self.tasks[task_id]

        if task.status != TaskStatus.PENDING:
            return False

        # 查找合适的团队
        team = self._find_suitable_team(task)

        if team is None:
            return False

        # 分配任务
        task.assigned_team_id = team.team_id
        task.assigned_at = time.time()
        task.status = TaskStatus.ASSIGNED

        # 更新团队任务数
        team.current_tasks += 1

        return True

    def _find_suitable_team(self, task: ProblemTask) -> Optional[Team]:
        """
        查找合适的团队

        Args:
            task: 任务

        Returns:
            Team: 合适的团队
        """
        # 根据分发规则查找团队类型
        team_type = self._determine_team_type(task)

        # 查找该类型的团队
        candidate_teams = [
            team for team in self.teams.values()
            if team.team_type == team_type
        ]

        if not candidate_teams:
            return None

        # 选择负载最低的团队
        suitable_team = min(
            candidate_teams,
            key=lambda t: t.current_tasks / t.max_concurrent_tasks
        )

        # 检查是否达到最大并发数
        if suitable_team.current_tasks >= suitable_team.max_concurrent_tasks:
            return None

        return suitable_team

    def _determine_team_type(self, task: ProblemTask) -> TeamType:
        """
        确定团队类型

        Args:
            task: 任务

        Returns:
            TeamType: 团队类型
        """
        # 查找匹配的分发规则
        for rule in self.distribution_rules:
            if not rule.enabled:
                continue

            if (rule.problem_type == task.problem_type and
                rule.problem_severity == task.problem_severity):
                return rule.team_type

        # 默认映射
        problem_type_mapping = {
            "perception": TeamType.PERCEPTION,
            "prediction": TeamType.PREDICTION,
            "planning": TeamType.PLANNING,
            "control": TeamType.CONTROL,
            "sensor": TeamType.SENSOR,
            "algorithm": TeamType.ALGORITHM
        }

        return problem_type_mapping.get(task.problem_type, TeamType.GENERAL)

    def _determine_priority(self, problem_type: str, problem_severity: str) -> TaskPriority:
        """
        确定任务优先级

        Args:
            problem_type: 问题类型
            problem_severity: 问题严重程度

        Returns:
            TaskPriority: 任务优先级
        """
        # 根据严重程度确定优先级
        severity_mapping = {
            "critical": TaskPriority.CRITICAL,
            "high": TaskPriority.HIGH,
            "medium": TaskPriority.MEDIUM,
            "low": TaskPriority.LOW
        }

        return severity_mapping.get(problem_severity, TaskPriority.MEDIUM)

    def update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        assigned_to: Optional[str] = None
    ) -> bool:
        """
        更新任务状态

        Args:
            task_id: 任务ID
            status: 任务状态
            assigned_to: 分配给谁

        Returns:
            bool: 更新是否成功
        """
        if task_id not in self.tasks:
            return False

        task = self.tasks[task_id]
        task.status = status

        if assigned_to:
            task.assigned_to = assigned_to

        # 更新时间戳
        if status == TaskStatus.IN_PROGRESS and task.started_at is None:
            task.started_at = time.time()
        elif status == TaskStatus.COMPLETED and task.completed_at is None:
            task.completed_at = time.time()

        # 如果任务完成，减少团队任务数
        if status == TaskStatus.COMPLETED and task.assigned_team_id:
            team = self.teams.get(task.assigned_team_id)
            if team:
                team.current_tasks = max(0, team.current_tasks - 1)

        return True

    def get_team_tasks(self, team_id: str) -> List[ProblemTask]:
        """
        获取团队的任务

        Args:
            team_id: 团队ID

        Returns:
            List[ProblemTask]: 任务列表
        """
        return [
            task for task in self.tasks.values()
            if task.assigned_team_id == team_id
        ]

    def get_pending_tasks(self, priority: Optional[TaskPriority] = None) -> List[ProblemTask]:
        """
        获取待分配任务

        Args:
            priority: 优先级过滤

        Returns:
            List[ProblemTask]: 任务列表
        """
        tasks = [
            task for task in self.tasks.values()
            if task.status == TaskStatus.PENDING
        ]

        if priority:
            tasks = [task for task in tasks if task.priority == priority]

        # 按优先级排序
        priority_order = {
            TaskPriority.CRITICAL: 0,
            TaskPriority.HIGH: 1,
            TaskPriority.MEDIUM: 2,
            TaskPriority.LOW: 3
        }

        tasks.sort(key=lambda t: priority_order.get(t.priority, 4))

        return tasks

    def get_task_statistics(self, team_id: Optional[str] = None) -> Dict[str, Any]:
        """
        获取任务统计

        Args:
            team_id: 团队ID（None表示所有团队）

        Returns:
            Dict: 统计信息
        """
        tasks = self.tasks.values()

        if team_id:
            tasks = [task for task in tasks if task.assigned_team_id == team_id]

        total_tasks = len(tasks)

        # 按状态统计
        status_stats = {}
        for task in tasks:
            status = task.status.value
            status_stats[status] = status_stats.get(status, 0) + 1

        # 按优先级统计
        priority_stats = {}
        for task in tasks:
            priority = task.priority.value
            priority_stats[priority] = priority_stats.get(priority, 0) + 1

        # 按问题类型统计
        type_stats = {}
        for task in tasks:
            problem_type = task.problem_type
            type_stats[problem_type] = type_stats.get(problem_type, 0) + 1

        # 计算平均工时
        completed_tasks = [task for task in tasks if task.status == TaskStatus.COMPLETED]
        avg_actual_hours = 0.0
        if completed_tasks:
            avg_actual_hours = sum(task.actual_hours for task in completed_tasks) / len(completed_tasks)

        return {
            "total_tasks": total_tasks,
            "status_stats": status_stats,
            "priority_stats": priority_stats,
            "type_stats": type_stats,
            "avg_actual_hours": avg_actual_hours
        }

    def get_overdue_tasks(self) -> List[ProblemTask]:
        """
        获取逾期任务

        Returns:
            List[ProblemTask]: 逾期任务列表
        """
        overdue_tasks = []
        current_time = time.time()

        for task in self.tasks.values():
            if task.status in [TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS]:
                # 假设预估工时转换为秒
                deadline = task.assigned_at + task.estimated_hours * 3600 if task.assigned_at else 0

                if deadline > 0 and current_time > deadline:
                    overdue_tasks.append(task)

        return overdue_tasks

    def delete_task(self, task_id: str) -> bool:
        """
        删除任务

        Args:
            task_id: 任务ID

        Returns:
            bool: 删除是否成功
        """
        if task_id not in self.tasks:
            return False

        task = self.tasks[task_id]

        # 减少团队任务数
        if task.assigned_team_id:
            team = self.teams.get(task.assigned_team_id)
            if team:
                team.current_tasks = max(0, team.current_tasks - 1)

        del self.tasks[task_id]
        return True

    def get_all_teams(self) -> Dict[str, Team]:
        """获取所有团队"""
        return self.teams.copy()

    def get_all_tasks(self) -> Dict[str, ProblemTask]:
        """获取所有任务"""
        return self.tasks.copy()

    def get_summary(self) -> Dict[str, Any]:
        """获取摘要"""
        total_teams = len(self.teams)
        total_tasks = len(self.tasks)

        # 统计各状态任务数
        status_counts = {}
        for task in self.tasks.values():
            status = task.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

        # 统计团队负载
        team_loads = {}
        for team in self.teams.values():
            team_loads[team.team_id] = {
                "current_tasks": team.current_tasks,
                "max_concurrent_tasks": team.max_concurrent_tasks,
                "utilization": team.current_tasks / team.max_concurrent_tasks if team.max_concurrent_tasks > 0 else 0
            }

        return {
            "total_teams": total_teams,
            "total_tasks": total_tasks,
            "status_counts": status_counts,
            "team_loads": team_loads
        }


# 全局问题分发器实例
_global_problem_distributor = None


def get_global_problem_distributor() -> ProblemDistributor:
    """获取全局问题分发器实例"""
    global _global_problem_distributor
    if _global_problem_distributor is None:
        _global_problem_distributor = ProblemDistributor()
    return _global_problem_distributor