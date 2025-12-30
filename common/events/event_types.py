"""
事件类型定义
定义自动驾驶系统中的各种事件类型
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum
import time
import uuid


class EventType(str, Enum):
    """事件类型"""
    # 感知事件
    PERCEPTION_DETECTED = "perception_detected"           # 感知检测
    PERCEPTION_LOST = "perception_lost"                   # 感知丢失
    PERCEPTION_TRACKING = "perception_tracking"           # 感知跟踪

    # 预测事件
    PREDICTION_STARTED = "prediction_started"             # 预测开始
    PREDICTION_UPDATED = "prediction_updated"             # 预测更新
    PREDICTION_FAILED = "prediction_failed"               # 预测失败

    # 规划事件
    PLANNING_STARTED = "planning_started"                 # 规划开始
    PLANNING_COMPLETED = "planning_completed"             # 规划完成
    PLANNING_CHANGED = "planning_changed"                 # 规划改变
    PLANNING_ABORTED = "planning_aborted"                 # 规划中止

    # 控制事件
    CONTROL_STARTED = "control_started"                   # 控制开始
    CONTROL_UPDATED = "control_updated"                   # 控制更新
    CONTROL_FAILED = "control_failed"                     # 控制失败

    # 车辆事件
    VEHICLE_STARTED = "vehicle_started"                   # 车辆启动
    VEHICLE_STOPPED = "vehicle_stopped"                   # 车辆停止
    VEHICLE_EMERGENCY = "vehicle_emergency"               # 车辆紧急

    # 传感器事件
    SENSOR_DATA_RECEIVED = "sensor_data_received"         # 传感器数据接收
    SENSOR_ERROR = "sensor_error"                         # 传感器错误
    SENSOR_CALIBRATION = "sensor_calibration"             # 传感器校准

    # 系统事件
    SYSTEM_INITIALIZED = "system_initialized"             # 系统初始化
    SYSTEM_ERROR = "system_error"                         # 系统错误
    SYSTEM_SHUTDOWN = "system_shutdown"                   # 系统关闭

    # 自定义事件
    CUSTOM = "custom"                                     # 自定义


class EventPriority(str, Enum):
    """事件优先级"""
    CRITICAL = "critical"     # 紧急
    HIGH = "high"             # 高
    MEDIUM = "medium"         # 中等
    LOW = "low"               # 低


class Event(BaseModel):
    """事件基类"""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="事件ID")
    event_type: EventType = Field(description="事件类型")
    timestamp: float = Field(default_factory=time.time, description="时间戳")
    priority: EventPriority = Field(default=EventPriority.MEDIUM, description="优先级")
    source: str = Field(default="", description="事件源")
    data: Dict[str, Any] = Field(default_factory=dict, description="事件数据")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


class PerceptionEvent(Event):
    """感知事件"""
    event_type: EventType = Field(default=EventType.PERCEPTION_DETECTED)
    object_id: str = Field(description="对象ID")
    object_type: str = Field(description="对象类型")
    position: Dict[str, float] = Field(description="位置")
    velocity: Dict[str, float] = Field(default_factory=lambda: {"x": 0.0, "y": 0.0, "z": 0.0}, description="速度")
    confidence: float = Field(default=1.0, description="置信度")
    attributes: Dict[str, Any] = Field(default_factory=dict, description="属性")


class PredictionEvent(Event):
    """预测事件"""
    event_type: EventType = Field(default=EventType.PREDICTION_UPDATED)
    object_id: str = Field(description="对象ID")
    prediction_horizon: float = Field(default=5.0, description="预测时长（秒）")
    predicted_trajectory: List[Dict[str, float]] = Field(default_factory=list, description="预测轨迹")
    confidence: float = Field(default=1.0, description="置信度")


class PlanningEvent(Event):
    """规划事件"""
    event_type: EventType = Field(default=EventType.PLANNING_COMPLETED)
    plan_id: str = Field(description="规划ID")
    planned_trajectory: List[Dict[str, float]] = Field(default_factory=list, description="规划轨迹")
    planned_speed: float = Field(default=0.0, description="规划速度")
    planning_time: float = Field(default=0.0, description="规划耗时")


class ControlEvent(Event):
    """控制事件"""
    event_type: EventType = Field(default=EventType.CONTROL_UPDATED)
    control_id: str = Field(description="控制ID")
    throttle: float = Field(default=0.0, description="油门")
    brake: float = Field(default=0.0, description="刹车")
    steering: float = Field(default=0.0, description="转向")
    control_time: float = Field(default=0.0, description="控制耗时")


class VehicleEvent(Event):
    """车辆事件"""
    event_type: EventType = Field(default=EventType.VEEHICLE_STARTED)
    vehicle_id: str = Field(description="车辆ID")
    position: Dict[str, float] = Field(description="位置")
    velocity: Dict[str, float] = Field(default_factory=lambda: {"x": 0.0, "y": 0.0, "z": 0.0}, description="速度")
    acceleration: Dict[str, float] = Field(default_factory=lambda: {"x": 0.0, "y": 0.0, "z": 0.0}, description="加速度")
    heading: float = Field(default=0.0, description="航向角")


class SensorEvent(Event):
    """传感器事件"""
    event_type: EventType = Field(default=EventType.SENSOR_DATA_RECEIVED)
    sensor_id: str = Field(description="传感器ID")
    sensor_type: str = Field(description="传感器类型")
    data: Dict[str, Any] = Field(default_factory=dict, description="传感器数据")
    timestamp: float = Field(default_factory=time.time, description="数据时间戳")


class SystemEvent(Event):
    """系统事件"""
    event_type: EventType = Field(default=EventType.SYSTEM_INITIALIZED)
    component: str = Field(description="组件")
    status: str = Field(description="状态")
    message: str = Field(default="", description="消息")


class EventSequence(BaseModel):
    """事件序列"""
    sequence_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="序列ID")
    events: List[Event] = Field(default_factory=list, description="事件列表")
    start_time: float = Field(description="起始时间")
    end_time: float = Field(description="结束时间")
    duration: float = Field(description="持续时间")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")

    def add_event(self, event: Event):
        """添加事件"""
        self.events.append(event)
        self.events.sort(key=lambda e: e.timestamp)

    def get_events_by_type(self, event_type: EventType) -> List[Event]:
        """根据类型获取事件"""
        return [event for event in self.events if event.event_type == event_type]

    def get_events_by_priority(self, priority: EventPriority) -> List[Event]:
        """根据优先级获取事件"""
        return [event for event in self.events if event.priority == priority]

    def get_events_in_time_range(self, start_time: float, end_time: float) -> List[Event]:
        """获取时间范围内的事件"""
        return [
            event for event in self.events
            if start_time <= event.timestamp <= end_time
        ]

    def get_event_statistics(self) -> Dict[str, Any]:
        """获取事件统计"""
        total_events = len(self.events)

        # 按类型统计
        type_stats = {}
        for event in self.events:
            event_type = event.event_type.value
            type_stats[event_type] = type_stats.get(event_type, 0) + 1

        # 按优先级统计
        priority_stats = {}
        for event in self.events:
            priority = event.priority.value
            priority_stats[priority] = priority_stats.get(priority, 0) + 1

        # 按源统计
        source_stats = {}
        for event in self.events:
            source = event.source
            source_stats[source] = source_stats.get(source, 0) + 1

        return {
            "total_events": total_events,
            "type_stats": type_stats,
            "priority_stats": priority_stats,
            "source_stats": source_stats,
            "duration": self.duration
        }


class EventBuilder:
    """事件构建器"""

    @staticmethod
    def create_perception_event(
        object_id: str,
        object_type: str,
        position: Dict[str, float],
        confidence: float = 1.0,
        **kwargs
    ) -> PerceptionEvent:
        """创建感知事件"""
        return PerceptionEvent(
            object_id=object_id,
            object_type=object_type,
            position=position,
            confidence=confidence,
            **kwargs
        )

    @staticmethod
    def create_prediction_event(
        object_id: str,
        predicted_trajectory: List[Dict[str, float]],
        confidence: float = 1.0,
        **kwargs
    ) -> PredictionEvent:
        """创建预测事件"""
        return PredictionEvent(
            object_id=object_id,
            predicted_trajectory=predicted_trajectory,
            confidence=confidence,
            **kwargs
        )

    @staticmethod
    def create_planning_event(
        plan_id: str,
        planned_trajectory: List[Dict[str, float]],
        planned_speed: float = 0.0,
        **kwargs
    ) -> PlanningEvent:
        """创建规划事件"""
        return PlanningEvent(
            plan_id=plan_id,
            planned_trajectory=planned_trajectory,
            planned_speed=planned_speed,
            **kwargs
        )

    @staticmethod
    def create_control_event(
        control_id: str,
        throttle: float = 0.0,
        brake: float = 0.0,
        steering: float = 0.0,
        **kwargs
    ) -> ControlEvent:
        """创建控制事件"""
        return ControlEvent(
            control_id=control_id,
            throttle=throttle,
            brake=brake,
            steering=steering,
            **kwargs
        )

    @staticmethod
    def create_vehicle_event(
        vehicle_id: str,
        position: Dict[str, float],
        velocity: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> VehicleEvent:
        """创建车辆事件"""
        return VehicleEvent(
            vehicle_id=vehicle_id,
            position=position,
            velocity=velocity or {"x": 0.0, "y": 0.0, "z": 0.0},
            **kwargs
        )

    @staticmethod
    def create_sensor_event(
        sensor_id: str,
        sensor_type: str,
        data: Dict[str, Any],
        **kwargs
    ) -> SensorEvent:
        """创建传感器事件"""
        return SensorEvent(
            sensor_id=sensor_id,
            sensor_type=sensor_type,
            data=data,
            **kwargs
        )

    @staticmethod
    def create_system_event(
        component: str,
        status: str,
        message: str = "",
        **kwargs
    ) -> SystemEvent:
        """创建系统事件"""
        return SystemEvent(
            component=component,
            status=status,
            message=message,
            **kwargs
        )


# 事件类型映射
EVENT_TYPE_MAPPING = {
    "perception": PerceptionEvent,
    "prediction": PredictionEvent,
    "planning": PlanningEvent,
    "control": ControlEvent,
    "vehicle": VehicleEvent,
    "sensor": SensorEvent,
    "system": SystemEvent
}


def create_event_from_dict(event_dict: Dict[str, Any]) -> Event:
    """
    从字典创建事件

    Args:
        event_dict: 事件字典

    Returns:
        Event: 事件
    """
    event_type_str = event_dict.get("event_type", "custom")
    event_type = EventType(event_type_str)

    # 根据事件类型选择对应的类
    event_class = Event
    for type_str, cls in EVENT_TYPE_MAPPING.items():
        if type_str in event_type_str:
            event_class = cls
            break

    return event_class(**event_dict)


def serialize_event(event: Event) -> Dict[str, Any]:
    """
    序列化事件

    Args:
        event: 事件

    Returns:
        Dict: 序列化后的字典
    """
    return event.dict()