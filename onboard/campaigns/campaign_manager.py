"""
Campaign（任务）管理模块
实现云端任务下发和车端动态触发
"""

from typing import Dict, List, Optional, Any, Callable
from pydantic import BaseModel, Field
from enum import Enum
import time
import json
import threading


class CampaignStatus(str, Enum):
    """Campaign状态"""
    ACTIVE = "active"      # 活跃
    PAUSED = "paused"      # 暂停
    COMPLETED = "completed"  # 已完成
    EXPIRED = "expired"    # 已过期


class CampaignQuery(BaseModel):
    """Campaign查询条件"""
    weather: Optional[str] = Field(default=None, description="天气条件")
    speed_min: Optional[float] = Field(default=None, description="最小速度")
    speed_max: Optional[float] = Field(default=None, description="最大速度")
    object_types: Optional[List[str]] = Field(default=None, description="对象类型列表")
    road_type: Optional[str] = Field(default=None, description="道路类型")
    time_range: Optional[Dict[str, float]] = Field(default=None, description="时间范围")
    location_range: Optional[Dict[str, Any]] = Field(default=None, description="位置范围")
    custom_conditions: Optional[Dict[str, Any]] = Field(default=None, description="自定义条件")


class CampaignConfig(BaseModel):
    """Campaign配置"""
    campaign_id: str
    name: str
    description: str
    query: CampaignQuery
    priority: int = Field(default=1, ge=1, le=10, description="优先级（1-10）")
    max_samples: int = Field(default=1000, description="最大样本数")
    collected_samples: int = Field(default=0, description="已收集样本数")
    status: CampaignStatus = Field(default=CampaignStatus.ACTIVE)
    created_at: float = Field(default_factory=time.time)
    expires_at: Optional[float] = Field(default=None, description="过期时间")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CampaignManager:
    """
    Campaign管理器
    管理云端下发的任务和车端触发逻辑
    """

    def __init__(self):
        self.campaigns: Dict[str, CampaignConfig] = {}
        self._lock = threading.Lock()
        self._evaluation_callback: Optional[Callable] = None

    def register_campaign(self, config: CampaignConfig) -> bool:
        """
        注册Campaign

        Args:
            config: Campaign配置

        Returns:
            bool: 注册是否成功
        """
        with self._lock:
            if config.campaign_id in self.campaigns:
                return False

            self.campaigns[config.campaign_id] = config
            return True

    def unregister_campaign(self, campaign_id: str) -> bool:
        """
        注销Campaign

        Args:
            campaign_id: Campaign ID

        Returns:
            bool: 注销是否成功
        """
        with self._lock:
            if campaign_id not in self.campaigns:
                return False

            del self.campaigns[campaign_id]
            return True

    def update_campaign(self, campaign_id: str, **kwargs) -> bool:
        """
        更新Campaign

        Args:
            campaign_id: Campaign ID
            **kwargs: 更新的字段

        Returns:
            bool: 更新是否成功
        """
        with self._lock:
            if campaign_id not in self.campaigns:
                return False

            config = self.campaigns[campaign_id]

            # 更新字段
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)

            return True

    def get_campaign(self, campaign_id: str) -> Optional[CampaignConfig]:
        """
        获取Campaign

        Args:
            campaign_id: Campaign ID

        Returns:
            Optional[CampaignConfig]: Campaign配置
        """
        with self._lock:
            return self.campaigns.get(campaign_id)

    def get_all_campaigns(self, status: Optional[CampaignStatus] = None) -> List[CampaignConfig]:
        """
        获取所有Campaign

        Args:
            status: 状态过滤

        Returns:
            List[CampaignConfig]: Campaign列表
        """
        with self._lock:
            campaigns = list(self.campaigns.values())

            if status is not None:
                campaigns = [c for c in campaigns if c.status == status]

            return campaigns

    def evaluate_campaigns(
        self,
        sensor_data: Dict[str, Any],
        vehicle_state: Dict[str, Any]
    ) -> List[str]:
        """
        评估所有Campaign，返回命中的Campaign ID列表

        Args:
            sensor_data: 传感器数据
            vehicle_state: 车辆状态

        Returns:
            List[str]: 命中的Campaign ID列表
        """
        hit_campaigns = []

        with self._lock:
            for campaign_id, config in self.campaigns.items():
                # 检查Campaign状态
                if config.status != CampaignStatus.ACTIVE:
                    continue

                # 检查是否过期
                if config.expires_at and time.time() > config.expires_at:
                    config.status = CampaignStatus.EXPIRED
                    continue

                # 检查样本数是否达到上限
                if config.collected_samples >= config.max_samples:
                    config.status = CampaignStatus.COMPLETED
                    continue

                # 评估查询条件
                if self._evaluate_query(config.query, sensor_data, vehicle_state):
                    hit_campaigns.append(campaign_id)

                    # 增加已收集样本数
                    config.collected_samples += 1

        return hit_campaigns

    def _evaluate_query(
        self,
        query: CampaignQuery,
        sensor_data: Dict[str, Any],
        vehicle_state: Dict[str, Any]
    ) -> bool:
        """
        评估查询条件

        Args:
            query: 查询条件
            sensor_data: 传感器数据
            vehicle_state: 车辆状态

        Returns:
            bool: 是否命中
        """
        # 检查天气条件
        if query.weather:
            current_weather = sensor_data.get("weather", "unknown")
            if current_weather != query.weather:
                return False

        # 检查速度范围
        current_speed = vehicle_state.get("speed", 0)
        if query.speed_min is not None and current_speed < query.speed_min:
            return False
        if query.speed_max is not None and current_speed > query.speed_max:
            return False

        # 检查对象类型
        if query.object_types:
            detected_objects = sensor_data.get("objects", [])
            detected_types = set(obj.get("type") for obj in detected_objects)
            if not any(obj_type in detected_types for obj_type in query.object_types):
                return False

        # 检查道路类型
        if query.road_type:
            current_road_type = sensor_data.get("road_type", "unknown")
            if current_road_type != query.road_type:
                return False

        # 检查时间范围
        if query.time_range:
            current_time = time.time()
            start_time = query.time_range.get("start")
            end_time = query.time_range.get("end")
            if start_time and current_time < start_time:
                return False
            if end_time and current_time > end_time:
                return False

        # 检查位置范围
        if query.location_range:
            location = vehicle_state.get("location", {})
            lat = location.get("lat")
            lon = location.get("lon")

            if lat is None or lon is None:
                return False

            lat_range = query.location_range.get("lat")
            lon_range = query.location_range.get("lon")

            if lat_range:
                if lat < lat_range.get("min", -90) or lat > lat_range.get("max", 90):
                    return False

            if lon_range:
                if lon < lon_range.get("min", -180) or lon > lon_range.get("max", 180):
                    return False

        # 检查自定义条件
        if query.custom_conditions:
            for key, expected_value in query.custom_conditions.items():
                actual_value = sensor_data.get(key, vehicle_state.get(key))
                if actual_value != expected_value:
                    return False

        return True

    def import_campaigns_from_json(self, json_str: str) -> int:
        """
        从JSON导入Campaign

        Args:
            json_str: JSON字符串

        Returns:
            int: 导入成功的数量
        """
        try:
            campaigns_data = json.loads(json_str)
            success_count = 0

            for campaign_data in campaigns_data:
                query_data = campaign_data.pop("query", {})
                query = CampaignQuery(**query_data)

                config = CampaignConfig(
                    query=query,
                    **campaign_data
                )

                if self.register_campaign(config):
                    success_count += 1

            return success_count

        except Exception as e:
            print(f"Import campaigns error: {e}")
            return 0

    def export_campaigns_to_json(self, status: Optional[CampaignStatus] = None) -> str:
        """
        导出Campaign为JSON

        Args:
            status: 状态过滤

        Returns:
            str: JSON字符串
        """
        campaigns = self.get_all_campaigns(status)
        campaigns_data = [c.dict() for c in campaigns]
        return json.dumps(campaigns_data, indent=2)

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            total_campaigns = len(self.campaigns)
            active_campaigns = sum(1 for c in self.campaigns.values() if c.status == CampaignStatus.ACTIVE)
            completed_campaigns = sum(1 for c in self.campaigns.values() if c.status == CampaignStatus.COMPLETED)
            expired_campaigns = sum(1 for c in self.campaigns.values() if c.status == CampaignStatus.EXPIRED)

            total_collected = sum(c.collected_samples for c in self.campaigns.values())
            total_capacity = sum(c.max_samples for c in self.campaigns.values())

            return {
                "total_campaigns": total_campaigns,
                "active_campaigns": active_campaigns,
                "completed_campaigns": completed_campaigns,
                "expired_campaigns": expired_campaigns,
                "total_collected": total_collected,
                "total_capacity": total_capacity,
                "collection_rate": total_collected / total_capacity if total_capacity > 0 else 0
            }


# 全局Campaign管理器实例
_global_campaign_manager = None


def get_global_campaign_manager() -> CampaignManager:
    """获取全局Campaign管理器实例"""
    global _global_campaign_manager
    if _global_campaign_manager is None:
        _global_campaign_manager = CampaignManager()
    return _global_campaign_manager