"""
数据采集与日志策略模块
实现特斯拉机制（Telemetry/Snapshot/Clip/Engineering）
"""

from typing import Dict, List, Optional, Any, Callable
from pydantic import BaseModel, Field
from enum import Enum
import time
import json
import threading
from queue import PriorityQueue, Empty, Queue
from dataclasses import dataclass, order


class LogLevel(str, Enum):
    """日志级别 - 特斯拉机制"""
    TELEMETRY = "telemetry"  # 微日志：定频或状态变更
    SNAPSHOT = "snapshot"    # 快照：影子模式分歧、AEB触发、Campaign命中
    CLIP = "clip"            # 短视频：复杂场景、高风险接管
    ENGINEERING = "engineering"  # 全量：白名单车辆、重大事故


class LogPriority(int, Enum):
    """日志优先级"""
    CRITICAL = 0  # 碰撞/AEB触发 - 4G优先队列
    HIGH = 1      # 影子模式分歧、Campaign命中 - 4G优先队列
    MEDIUM = 2    # 普通触发 - Wi-Fi待传队列
    LOW = 3       # Telemetry - 4G实时


@dataclass(order=True)
class LogTask:
    """日志任务"""
    priority: int
    timestamp: float
    log_level: LogLevel
    trigger_id: str
    data: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.priority, LogPriority):
            self.priority = self.priority.value


class LogConfig(BaseModel):
    """日志配置"""
    log_id: str
    log_level: LogLevel
    trigger_id: str
    data_window: float = 10.0  # 数据窗口（秒）
    before_window: float = 5.0  # 前置窗口（秒）
    after_window: float = 5.0   # 后置窗口（秒）
    include_sensors: List[str] = Field(default_factory=list)
    include_state_machine: bool = True
    include_control_commands: bool = True
    include_fusion_results: bool = False
    upload_policy: str = "wifi_only"  # wifi_only, 4g_priority, 4g_realtime, manual_only
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LogEntry(BaseModel):
    """日志条目"""
    log_id: str
    log_level: LogLevel
    trigger_id: str
    timestamp: float
    start_time: float
    end_time: float
    data: Dict[str, Any]
    size: int = 0  # 数据大小（字节）
    uploaded: bool = False
    upload_time: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DataLogger:
    """
    数据日志器
    实现特斯拉机制（Telemetry/Snapshot/Clip/Engineering）
    """

    def __init__(self, max_queue_size: int = 1000):
        self.log_configs: Dict[str, LogConfig] = {}
        # 4G优先队列（碰撞/AEB、影子模式分歧）
        self.priority_queue: PriorityQueue = PriorityQueue(maxsize=max_queue_size)
        # Wi-Fi待传队列（普通触发）
        self.wifi_queue: Queue = Queue(maxsize=max_queue_size * 10)
        # Telemetry队列（实时上传）
        self.telemetry_queue: Queue = Queue(maxsize=1000)
        self.upload_history: List[LogEntry] = []
        self.max_history_size = 10000
        self._running = False
        self._upload_thread: Optional[threading.Thread] = None
        self._telemetry_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._campaigns: Dict[str, Dict[str, Any]] = {}  # 存储云端下发的Campaign

    def register_log_config(self, config: LogConfig) -> bool:
        """
        注册日志配置

        Args:
            config: 日志配置

        Returns:
            bool: 注册是否成功
        """
        if config.log_id in self.log_configs:
            return False

        self.log_configs[config.log_id] = config
        return True

    def unregister_log_config(self, log_id: str) -> bool:
        """
        注销日志配置

        Args:
            log_id: 日志ID

        Returns:
            bool: 注销是否成功
        """
        if log_id not in self.log_configs:
            return False

        del self.log_configs[log_id]
        return True

    def create_telemetry(
        self,
        timestamp: float,
        location: Dict[str, float],
        vehicle_state: Dict[str, Any],
        autopilot_stats: Dict[str, Any]
    ) -> Optional[LogEntry]:
        """
        创建Telemetry（微日志）- 定频或状态变更

        Args:
            timestamp: 时间戳
            location: 位置信息（经纬度）
            vehicle_state: 车辆状态（速度、档位等）
            autopilot_stats: Autopilot统计（接管次数、急刹次数、开启时长）

        Returns:
            LogEntry: 日志条目
        """
        data = {
            "location": location,
            "vehicle_state": vehicle_state,
            "autopilot_stats": autopilot_stats,
            "timestamp": timestamp
        }

        log_entry = LogEntry(
            log_id=f"telemetry_{int(timestamp)}",
            log_level=LogLevel.TELEMETRY,
            trigger_id="telemetry",
            timestamp=timestamp,
            start_time=timestamp,
            end_time=timestamp,
            data=data,
            size=self._calculate_size(data),
            metadata={"upload_policy": "4g_realtime"}
        )

        # 添加到Telemetry队列
        try:
            self.telemetry_queue.put(log_entry, block=False)
        except:
            pass  # 队列满，丢弃

        return log_entry

    def create_snapshot(
        self,
        trigger_id: str,
        timestamp: float,
        trigger_type: str,  # shadow_divergence, aeb_trigger, user_feedback, campaign_hit
        sensor_data: Dict[str, Any],
        key_frame_image: Optional[bytes] = None,
        object_list: Optional[List[Dict[str, Any]]] = None,
        ego_trajectory: Optional[List[Dict[str, float]]] = None
    ) -> Optional[LogEntry]:
        """
        创建Snapshot（快照）- 影子模式分歧、AEB触发、Campaign命中

        Args:
            trigger_id: 触发Trigger ID
            timestamp: 触发时间戳
            trigger_type: 触发类型
            sensor_data: 传感器数据
            key_frame_image: 关键帧图片（非视频）
            object_list: 目标列表
            ego_trajectory: 自车轨迹（前后10s）

        Returns:
            LogEntry: 日志条目
        """
        config = self._get_log_config(trigger_id, LogLevel.SNAPSHOT)
        if config is None:
            return None

        # 提取数据窗口
        start_time = timestamp - 10.0
        end_time = timestamp + 0.0

        # 过滤数据
        data = self._extract_data_window(
            sensor_data,
            start_time,
            end_time,
            config.include_sensors
        )

        # 添加关键帧图片
        if key_frame_image:
            data["key_frame_image"] = key_frame_image

        # 添加目标列表
        if object_list:
            data["object_list"] = object_list

        # 添加自车轨迹
        if ego_trajectory:
            data["ego_trajectory"] = ego_trajectory

        # 添加触发类型
        data["trigger_type"] = trigger_type

        # 创建日志条目
        log_entry = LogEntry(
            log_id=f"snapshot_{trigger_id}_{int(timestamp)}",
            log_level=LogLevel.SNAPSHOT,
            trigger_id=trigger_id,
            timestamp=timestamp,
            start_time=start_time,
            end_time=end_time,
            data=data,
            size=self._calculate_size(data),
            metadata={"upload_policy": "4g_priority", "trigger_type": trigger_type}
        )

        # 添加到4G优先队列
        self._add_to_priority_queue(log_entry, LogPriority.HIGH)

        return log_entry

    def check_shadow_divergence(
        self,
        timestamp: float,
        online_model_output: Dict[str, Any],
        shadow_model_output: Dict[str, Any],
        threshold: float = 0.3
    ) -> bool:
        """
        检测影子模式分歧

        Args:
            timestamp: 时间戳
            online_model_output: 在线模型输出
            shadow_model_output: 影子模型输出
            threshold: 分歧阈值

        Returns:
            bool: 是否产生分歧
        """
        # 比较两个模型的输出
        # 示例：比较轨迹预测
        online_traj = online_model_output.get("trajectory", [])
        shadow_traj = shadow_model_output.get("trajectory", [])

        if not online_traj or not shadow_traj:
            return False

        # 计算轨迹差异
        divergence = self._calculate_trajectory_divergence(online_traj, shadow_traj)

        if divergence > threshold:
            # 产生分歧，自动触发Snapshot
            sensor_data = online_model_output.get("sensor_data", {})
            key_frame_image = online_model_output.get("key_frame_image")
            object_list = online_model_output.get("object_list")
            ego_trajectory = online_model_output.get("ego_trajectory")

            self.create_snapshot(
                trigger_id="shadow_divergence",
                timestamp=timestamp,
                trigger_type="shadow_divergence",
                sensor_data=sensor_data,
                key_frame_image=key_frame_image,
                object_list=object_list,
                ego_trajectory=ego_trajectory
            )
            return True

        return False

    def check_campaign_hit(
        self,
        timestamp: float,
        sensor_data: Dict[str, Any],
        vehicle_state: Dict[str, Any]
    ) -> Optional[str]:
        """
        检查是否命中云端Campaign

        Args:
            timestamp: 时间戳
            sensor_data: 传感器数据
            vehicle_state: 车辆状态

        Returns:
            Optional[str]: 命中的Campaign ID，未命中返回None
        """
        for campaign_id, campaign_config in self._campaigns.items():
            if self._evaluate_campaign_query(campaign_config["query"], sensor_data, vehicle_state):
                # 命中Campaign，触发Snapshot
                self.create_snapshot(
                    trigger_id=campaign_id,
                    timestamp=timestamp,
                    trigger_type="campaign_hit",
                    sensor_data=sensor_data
                )
                return campaign_id

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
        # 简单实现：计算位置差异
        max_divergence = 0.0

        for p1, p2 in zip(traj1, traj2):
            x_diff = abs(p1.get("x", 0) - p2.get("x", 0))
            y_diff = abs(p1.get("y", 0) - p2.get("y", 0))
            divergence = (x_diff ** 2 + y_diff ** 2) ** 0.5
            max_divergence = max(max_divergence, divergence)

        return max_divergence

    def _evaluate_campaign_query(
        self,
        query: Dict[str, Any],
        sensor_data: Dict[str, Any],
        vehicle_state: Dict[str, Any]
    ) -> bool:
        """
        评估Campaign查询条件

        Args:
            query: 查询条件
            sensor_data: 传感器数据
            vehicle_state: 车辆状态

        Returns:
            bool: 是否命中
        """
        # 解析查询条件
        weather_condition = query.get("weather")
        speed_min = query.get("speed_min")
        speed_max = query.get("speed_max")
        object_types = query.get("object_types", [])
        road_type = query.get("road_type")

        # 检查天气条件
        if weather_condition:
            current_weather = sensor_data.get("weather", "unknown")
            if current_weather != weather_condition:
                return False

        # 检查速度范围
        current_speed = vehicle_state.get("speed", 0)
        if speed_min is not None and current_speed < speed_min:
            return False
        if speed_max is not None and current_speed > speed_max:
            return False

        # 检查对象类型
        if object_types:
            detected_objects = sensor_data.get("objects", [])
            detected_types = set(obj.get("type") for obj in detected_objects)
            if not any(obj_type in detected_types for obj_type in object_types):
                return False

        # 检查道路类型
        if road_type:
            current_road_type = sensor_data.get("road_type", "unknown")
            if current_road_type != road_type:
                return False

        return True

    def create_clip(
        self,
        trigger_id: str,
        timestamp: float,
        trigger_type: str,  # complex_scenario, high_risk_takeover
        sensor_data: Dict[str, Any],
        video_stream: Optional[bytes] = None,
        lidar_data: Optional[bytes] = None,
        radar_data: Optional[bytes] = None
    ) -> Optional[LogEntry]:
        """
        创建Clip（短视频）- 复杂场景、高风险接管

        Args:
            trigger_id: 触发Trigger ID
            timestamp: 触发时间戳
            trigger_type: 触发类型
            sensor_data: 传感器数据
            video_stream: 前后30s原始视频流（H.265）
            lidar_data: 完整Lidar点云
            radar_data: 完整Radar数据

        Returns:
            LogEntry: 日志条目
        """
        config = self._get_log_config(trigger_id, LogLevel.CLIP)
        if config is None:
            return None

        # 提取数据窗口（前后30s）
        start_time = timestamp - 30.0
        end_time = timestamp + 30.0

        # 过滤数据
        data = self._extract_data_window(
            sensor_data,
            start_time,
            end_time,
            config.include_sensors
        )

        # 添加视频流
        if video_stream:
            data["video_stream"] = video_stream

        # 添加Lidar数据
        if lidar_data:
            data["lidar_data"] = lidar_data

        # 添加Radar数据
        if radar_data:
            data["radar_data"] = radar_data

        # 添加触发类型
        data["trigger_type"] = trigger_type

        # 创建日志条目
        log_entry = LogEntry(
            log_id=f"clip_{trigger_id}_{int(timestamp)}",
            log_level=LogLevel.CLIP,
            trigger_id=trigger_id,
            timestamp=timestamp,
            start_time=start_time,
            end_time=end_time,
            data=data,
            size=self._calculate_size(data),
            metadata={"upload_policy": "wifi_only", "trigger_type": trigger_type}
        )

        # 添加到Wi-Fi待传队列
        try:
            self.wifi_queue.put(log_entry, block=False)
        except:
            pass  # 队列满，丢弃

        return log_entry

    def create_engineering(
        self,
        trigger_id: str,
        timestamp: float,
        trigger_type: str,  # whitelist_vehicle, major_accident
        full_data: Dict[str, Any],
        feature_maps: Optional[Dict[str, Any]] = None,
        debug_logs: Optional[str] = None
    ) -> Optional[LogEntry]:
        """
        创建Engineering（全量）- 白名单车辆、重大事故

        Args:
            trigger_id: 触发Trigger ID
            timestamp: 触发时间戳
            trigger_type: 触发类型
            full_data: 完整数据
            feature_maps: 中间层特征图
            debug_logs: 调试日志

        Returns:
            LogEntry: 日志条目
        """
        data = full_data.copy()

        # 添加特征图
        if feature_maps:
            data["feature_maps"] = feature_maps

        # 添加调试日志
        if debug_logs:
            data["debug_logs"] = debug_logs

        # 添加触发类型
        data["trigger_type"] = trigger_type

        log_entry = LogEntry(
            log_id=f"engineering_{trigger_id}_{int(timestamp)}",
            log_level=LogLevel.ENGINEERING,
            trigger_id=trigger_id,
            timestamp=timestamp,
            start_time=full_data.get("start_time", timestamp - 3600),
            end_time=full_data.get("end_time", timestamp + 3600),
            data=data,
            size=self._calculate_size(data),
            metadata={"upload_policy": "manual_only", "trigger_type": trigger_type}
        )

        # 不自动上传，需要手动触发或专线传输
        return log_entry

    def _get_log_config(self, trigger_id: str, log_level: LogLevel) -> Optional[LogConfig]:
        """获取日志配置"""
        log_id = f"{log_level.value}_{trigger_id}"
        return self.log_configs.get(log_id)

    def _extract_data_window(
        self,
        sensor_data: Dict[str, Any],
        start_time: float,
        end_time: float,
        include_sensors: List[str]
    ) -> Dict[str, Any]:
        """
        提取数据窗口

        Args:
            sensor_data: 传感器数据
            start_time: 起始时间
            end_time: 结束时间
            include_sensors: 包含的传感器列表

        Returns:
            Dict: 过滤后的数据
        """
        filtered_data = {}

        # 如果没有指定传感器，包含所有
        sensors_to_include = include_sensors if include_sensors else sensor_data.keys()

        for sensor in sensors_to_include:
            if sensor not in sensor_data:
                continue

            data = sensor_data[sensor]

            # 如果是时序数据，按时间窗口过滤
            if isinstance(data, list) and len(data) > 0:
                filtered = [
                    item for item in data
                    if start_time <= item.get("timestamp", 0) <= end_time
                ]
                filtered_data[sensor] = filtered
            else:
                filtered_data[sensor] = data

        return filtered_data

    def _calculate_size(self, data: Dict[str, Any]) -> int:
        """计算数据大小（字节）"""
        return len(json.dumps(data).encode('utf-8'))

    def _add_to_priority_queue(self, log_entry: LogEntry, priority: LogPriority):
        """添加到4G优先队列"""
        task = LogTask(
            priority=priority,
            timestamp=log_entry.timestamp,
            log_level=log_entry.log_level,
            trigger_id=log_entry.trigger_id,
            data=log_entry.dict(),
            metadata=log_entry.metadata
        )

        try:
            self.priority_queue.put(task, block=False)
        except:
            # 队列满，丢弃最旧的低优先级任务
            pass

    def add_campaign(self, campaign_id: str, query: Dict[str, Any], priority: int = 1):
        """
        添加云端Campaign

        Args:
            campaign_id: Campaign ID
            query: 查询条件
            priority: 优先级
        """
        self._campaigns[campaign_id] = {
            "query": query,
            "priority": priority,
            "created_at": time.time()
        }

    def remove_campaign(self, campaign_id: str):
        """
        移除云端Campaign

        Args:
            campaign_id: Campaign ID
        """
        if campaign_id in self._campaigns:
            del self._campaigns[campaign_id]

    def get_campaigns(self) -> Dict[str, Dict[str, Any]]:
        """获取所有Campaign"""
        return self._campaigns.copy()

    def start_upload_worker(self, upload_callback: Callable, telemetry_callback: Optional[Callable] = None):
        """
        启动上传工作线程

        Args:
            upload_callback: 上传回调函数（Snapshot/Clip）
            telemetry_callback: Telemetry上传回调函数
        """
        if self._running:
            return

        self._running = True

        # 启动优先队列上传线程（4G优先）
        self._upload_thread = threading.Thread(
            target=self._priority_upload_worker,
            args=(upload_callback,),
            daemon=True
        )
        self._upload_thread.start()

        # 启动Telemetry上传线程（实时）
        if telemetry_callback:
            self._telemetry_thread = threading.Thread(
                target=self._telemetry_upload_worker,
                args=(telemetry_callback,),
                daemon=True
            )
            self._telemetry_thread.start()

    def stop_upload_worker(self):
        """停止上传工作线程"""
        self._running = False
        if self._upload_thread:
            self._upload_thread.join(timeout=5.0)
        if self._telemetry_thread:
            self._telemetry_thread.join(timeout=5.0)

    def _priority_upload_worker(self, upload_callback: Callable):
        """4G优先队列上传线程"""
        while self._running:
            try:
                # 从优先队列获取任务（超时1秒）
                task = self.priority_queue.get(timeout=1.0)

                # 调用上传回调
                success = upload_callback(task)

                if success:
                    # 记录上传历史
                    with self._lock:
                        log_entry = LogEntry(**task.data)
                        log_entry.uploaded = True
                        log_entry.upload_time = time.time()
                        self.upload_history.append(log_entry)

                        # 限制历史大小
                        if len(self.upload_history) > self.max_history_size:
                            self.upload_history = self.upload_history[-self.max_history_size:]

                self.priority_queue.task_done()

            except Empty:
                continue
            except Exception as e:
                print(f"Priority upload error: {e}")

    def _telemetry_upload_worker(self, telemetry_callback: Callable):
        """Telemetry实时上传线程"""
        while self._running:
            try:
                # 从Telemetry队列获取任务（超时0.1秒）
                log_entry = self.telemetry_queue.get(timeout=0.1)

                # 调用上传回调
                success = telemetry_callback(log_entry)

                if success:
                    # 记录上传历史
                    with self._lock:
                        log_entry.uploaded = True
                        log_entry.upload_time = time.time()
                        self.upload_history.append(log_entry)

                self.telemetry_queue.task_done()

            except Empty:
                continue
            except Exception as e:
                print(f"Telemetry upload error: {e}")

    def upload_wifi_queue(self, upload_callback: Callable) -> int:
        """
        手动触发Wi-Fi队列上传

        Args:
            upload_callback: 上传回调函数

        Returns:
            int: 上传成功的数量
        """
        success_count = 0

        while not self.wifi_queue.empty():
            try:
                log_entry = self.wifi_queue.get_nowait()

                # 调用上传回调
                success = upload_callback(log_entry)

                if success:
                    # 记录上传历史
                    with self._lock:
                        log_entry.uploaded = True
                        log_entry.upload_time = time.time()
                        self.upload_history.append(log_entry)

                        # 限制历史大小
                        if len(self.upload_history) > self.max_history_size:
                            self.upload_history = self.upload_history[-self.max_history_size:]

                    success_count += 1

            except Empty:
                break
            except Exception as e:
                print(f"Wi-Fi upload error: {e}")

        return success_count

    def get_upload_queue_size(self) -> Dict[str, int]:
        """获取各上传队列大小"""
        return {
            "priority_queue": self.priority_queue.qsize(),
            "wifi_queue": self.wifi_queue.qsize(),
            "telemetry_queue": self.telemetry_queue.qsize()
        }

    def get_upload_history(
        self,
        trigger_id: Optional[str] = None,
        log_level: Optional[LogLevel] = None
    ) -> List[LogEntry]:
        """
        获取上传历史

        Args:
            trigger_id: Trigger ID过滤
            log_level: 日志级别过滤

        Returns:
            List[LogEntry]: 上传历史
        """
        history = self.upload_history.copy()

        if trigger_id is not None:
            history = [h for h in history if h.trigger_id == trigger_id]

        if log_level is not None:
            history = [h for h in history if h.log_level == log_level]

        return history

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_logs = len(self.upload_history)
        uploaded_logs = sum(1 for h in self.upload_history if h.uploaded)

        # 按级别统计
        level_stats = {}
        for log in self.upload_history:
            level = log.log_level.value
            level_stats[level] = level_stats.get(level, 0) + 1

        # 按Trigger统计
        trigger_stats = {}
        for log in self.upload_history:
            trigger = log.trigger_id
            trigger_stats[trigger] = trigger_stats.get(trigger, 0) + 1

        return {
            "total_logs": total_logs,
            "uploaded_logs": uploaded_logs,
            "pending_logs": total_logs - uploaded_logs,
            "queue_size": self.get_upload_queue_size(),
            "level_stats": level_stats,
            "trigger_stats": trigger_stats
        }


# 全局日志器实例
_global_logger = None


def get_global_logger() -> DataLogger:
    """获取全局日志器实例"""
    global _global_logger
    if _global_logger is None:
        _global_logger = DataLogger()
    return _global_logger