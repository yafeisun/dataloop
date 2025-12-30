"""
场景生成模块
基于真实数据生成仿真场景，支持多样化场景生成
"""

from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field
from enum import Enum
import time
import json
import numpy as np
from dataclasses import dataclass


class SceneType(str, Enum):
    """场景类型"""
    INTERSECTION = "intersection"       # 十字路口
    LANE_CHANGE = "lane_change"         # 换道
    CUT_IN = "cut_in"                   # 变道切入
    U_TURN = "u_turn"                   # 掉头
    PEDESTRIAN = "pedestrian"           # 行人
    EMERGENCY_BRAKE = "emergency_brake" # 急刹车
    SHARP_TURN = "sharp_turn"           # 急转弯
    WEATHER = "weather"                 # 天气变化
    TUNNEL = "tunnel"                   # 隧道
    BRIDGE = "bridge"                   # 桥梁
    CUSTOM = "custom"                   # 自定义


class WeatherCondition(str, Enum):
    """天气条件"""
    SUNNY = "sunny"         # 晴天
    CLOUDY = "cloudy"       # 多云
    RAINY = "rainy"         # 雨天
    SNOWY = "snowy"         # 雪天
    FOGGY = "foggy"         # 雾天
    NIGHT = "night"         # 夜间


class SceneConfig(BaseModel):
    """场景配置"""
    scene_id: str = Field(description="场景ID")
    name: str = Field(description="场景名称")
    scene_type: SceneType = Field(description="场景类型")
    description: str = Field(default="", description="描述")
    duration: float = Field(default=30.0, description="场景时长（秒）")
    weather: WeatherCondition = Field(default=WeatherCondition.SUNNY, description="天气条件")
    time_of_day: str = Field(default="day", description="时间段(day/night)")
    traffic_density: str = Field(default="medium", description="交通密度(low/medium/high)")
    num_vehicles: int = Field(default=5, description="车辆数量")
    num_pedestrians: int = Field(default=2, description="行人数量")
    road_type: str = Field(default="urban", description="道路类型(urban/highway/rural)")
    ego_vehicle_start_pos: Dict[str, float] = Field(
        default_factory=lambda: {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
        description="主车起始位置"
    )
    ego_vehicle_target_pos: Dict[str, float] = Field(
        default_factory=lambda: {"x": 100.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
        description="主车目标位置"
    )
    obstacles: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="障碍物列表"
    )
    traffic_lights: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="交通灯列表"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


class SceneObject(BaseModel):
    """场景对象"""
    object_id: str = Field(description="对象ID")
    type: str = Field(description="对象类型(vehicle/pedestrian/obstacle)")
    position: Dict[str, float] = Field(description="位置(x, y, z)")
    rotation: Dict[str, float] = Field(description="旋转(pitch, yaw, roll)")
    velocity: Dict[str, float] = Field(default_factory=lambda: {"x": 0.0, "y": 0.0, "z": 0.0}, description="速度")
    acceleration: Dict[str, float] = Field(default_factory=lambda: {"x": 0.0, "y": 0.0, "z": 0.0}, description="加速度")
    dimensions: Dict[str, float] = Field(default_factory=lambda: {"length": 4.5, "width": 2.0, "height": 1.5}, description="尺寸")
    attributes: Dict[str, Any] = Field(default_factory=dict, description="属性")
    trajectory: List[Dict[str, float]] = Field(default_factory=list, description="轨迹")


class Scene(BaseModel):
    """场景"""
    scene_id: str = Field(description="场景ID")
    config: SceneConfig = Field(description="场景配置")
    objects: List[SceneObject] = Field(default_factory=list, description="场景对象")
    created_at: float = Field(default_factory=time.time, description="创建时间")
    source_data_id: Optional[str] = Field(default=None, description="源数据ID")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


class SceneGenerator:
    """
    场景生成器
    基于真实数据生成仿真场景
    """

    def __init__(self):
        self.scenes: Dict[str, Scene] = {}
        self.scene_counter = 0

    def generate_scene_from_real_data(
        self,
        real_data: Dict[str, Any],
        scene_type: SceneType,
        modifications: Optional[Dict[str, Any]] = None
    ) -> Scene:
        """
        从真实数据生成场景

        Args:
            real_data: 真实数据
            scene_type: 场景类型
            modifications: 修改参数

        Returns:
            Scene: 生成的场景
        """
        self.scene_counter += 1
        scene_id = f"scene_{self.scene_counter}"

        # 创建基础配置
        config = SceneConfig(
            scene_id=scene_id,
            name=f"{scene_type.value}_scene",
            scene_type=scene_type
        )

        # 应用修改
        if modifications:
            for key, value in modifications.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        # 从真实数据提取对象
        objects = self._extract_objects_from_real_data(real_data)

        # 创建场景
        scene = Scene(
            scene_id=scene_id,
            config=config,
            objects=objects,
            source_data_id=real_data.get("data_id"),
            metadata={"source": "real_data"}
        )

        self.scenes[scene_id] = scene
        return scene

    def generate_scene_from_template(
        self,
        template: SceneConfig,
        variations: Optional[Dict[str, Any]] = None
    ) -> Scene:
        """
        从模板生成场景

        Args:
            template: 场景模板
            variations: 变化参数

        Returns:
            Scene: 生成的场景
        """
        self.scene_counter += 1
        scene_id = f"scene_{self.scene_counter}"

        # 复制模板配置
        config_dict = template.dict()
        config_dict["scene_id"] = scene_id

        # 应用变化
        if variations:
            self._apply_variations(config_dict, variations)

        config = SceneConfig(**config_dict)

        # 生成场景对象
        objects = self._generate_objects_from_config(config)

        # 创建场景
        scene = Scene(
            scene_id=scene_id,
            config=config,
            objects=objects,
            metadata={"source": "template"}
        )

        self.scenes[scene_id] = scene
        return scene

    def _extract_objects_from_real_data(self, real_data: Dict[str, Any]) -> List[SceneObject]:
        """
        从真实数据提取对象

        Args:
            real_data: 真实数据

        Returns:
            List[SceneObject]: 场景对象列表
        """
        objects = []

        # 提取车辆
        vehicles = real_data.get("vehicles", [])
        for i, vehicle in enumerate(vehicles):
            object_id = f"vehicle_{i}"
            obj = SceneObject(
                object_id=object_id,
                type="vehicle",
                position=vehicle.get("position", {"x": 0.0, "y": 0.0, "z": 0.0}),
                rotation=vehicle.get("rotation", {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}),
                velocity=vehicle.get("velocity", {"x": 0.0, "y": 0.0, "z": 0.0}),
                acceleration=vehicle.get("acceleration", {"x": 0.0, "y": 0.0, "z": 0.0}),
                dimensions=vehicle.get("dimensions", {"length": 4.5, "width": 2.0, "height": 1.5}),
                attributes=vehicle.get("attributes", {}),
                trajectory=vehicle.get("trajectory", [])
            )
            objects.append(obj)

        # 提取行人
        pedestrians = real_data.get("pedestrians", [])
        for i, pedestrian in enumerate(pedestrians):
            object_id = f"pedestrian_{i}"
            obj = SceneObject(
                object_id=object_id,
                type="pedestrian",
                position=pedestrian.get("position", {"x": 0.0, "y": 0.0, "z": 0.0}),
                rotation=pedestrian.get("rotation", {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}),
                velocity=pedestrian.get("velocity", {"x": 0.0, "y": 0.0, "z": 0.0}),
                acceleration=pedestrian.get("acceleration", {"x": 0.0, "y": 0.0, "z": 0.0}),
                dimensions=pedestrian.get("dimensions", {"length": 0.5, "width": 0.5, "height": 1.7}),
                attributes=pedestrian.get("attributes", {}),
                trajectory=pedestrian.get("trajectory", [])
            )
            objects.append(obj)

        return objects

    def _generate_objects_from_config(self, config: SceneConfig) -> List[SceneObject]:
        """
        从配置生成场景对象

        Args:
            config: 场景配置

        Returns:
            List[SceneObject]: 场景对象列表
        """
        objects = []

        # 生成车辆
        for i in range(config.num_vehicles):
            object_id = f"vehicle_{i}"
            position = self._generate_random_position(config)

            obj = SceneObject(
                object_id=object_id,
                type="vehicle",
                position=position,
                rotation={"pitch": 0.0, "yaw": np.random.uniform(-np.pi, np.pi), "roll": 0.0},
                velocity={"x": np.random.uniform(-10, 10), "y": np.random.uniform(-5, 5), "z": 0.0},
                dimensions={"length": 4.5, "width": 2.0, "height": 1.5},
                attributes={"category": "car"}
            )
            objects.append(obj)

        # 生成行人
        for i in range(config.num_pedestrians):
            object_id = f"pedestrian_{i}"
            position = self._generate_random_position(config)

            obj = SceneObject(
                object_id=object_id,
                type="pedestrian",
                position=position,
                rotation={"pitch": 0.0, "yaw": np.random.uniform(-np.pi, np.pi), "roll": 0.0},
                velocity={"x": np.random.uniform(-2, 2), "y": np.random.uniform(-2, 2), "z": 0.0},
                dimensions={"length": 0.5, "width": 0.5, "height": 1.7},
                attributes={"category": "person"}
            )
            objects.append(obj)

        # 生成障碍物
        for i, obstacle in enumerate(config.obstacles):
            object_id = f"obstacle_{i}"
            obj = SceneObject(
                object_id=object_id,
                type="obstacle",
                position=obstacle.get("position", {"x": 0.0, "y": 0.0, "z": 0.0}),
                rotation=obstacle.get("rotation", {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}),
                dimensions=obstacle.get("dimensions", {"length": 1.0, "width": 1.0, "height": 1.0}),
                attributes=obstacle.get("attributes", {})
            )
            objects.append(obj)

        return objects

    def _generate_random_position(self, config: SceneConfig) -> Dict[str, float]:
        """
        生成随机位置

        Args:
            config: 场景配置

        Returns:
            Dict: 位置
        """
        # 根据道路类型生成位置
        if config.road_type == "highway":
            x = np.random.uniform(-100, 100)
            y = np.random.uniform(-10, 10)
        elif config.road_type == "urban":
            x = np.random.uniform(-50, 50)
            y = np.random.uniform(-20, 20)
        else:  # rural
            x = np.random.uniform(-200, 200)
            y = np.random.uniform(-30, 30)

        return {"x": x, "y": y, "z": 0.0}

    def _apply_variations(self, config_dict: Dict[str, Any], variations: Dict[str, Any]):
        """
        应用变化参数

        Args:
            config_dict: 配置字典
            variations: 变化参数
        """
        for key, value in variations.items():
            if key == "weather_variation":
                # 天气变化
                weather_list = [w.value for w in WeatherCondition]
                current_weather = config_dict.get("weather", WeatherCondition.SUNNY.value)
                weather_idx = weather_list.index(current_weather)
                new_idx = (weather_idx + value) % len(weather_list)
                config_dict["weather"] = weather_list[new_idx]

            elif key == "density_variation":
                # 交通密度变化
                current_density = config_dict.get("traffic_density", "medium")
                density_list = ["low", "medium", "high"]
                density_idx = density_list.index(current_density)
                new_idx = max(0, min(len(density_list) - 1, density_idx + value))
                config_dict["traffic_density"] = density_list[new_idx]

            elif key == "vehicle_count_variation":
                # 车辆数量变化
                config_dict["num_vehicles"] = max(0, config_dict.get("num_vehicles", 5) + value)

            elif key == "pedestrian_count_variation":
                # 行人数量变化
                config_dict["num_pedestrians"] = max(0, config_dict.get("num_pedestrians", 2) + value)

            else:
                # 直接设置
                config_dict[key] = value

    def generate_scene_variations(
        self,
        base_scene: Scene,
        num_variations: int,
        variation_params: Dict[str, Any]
    ) -> List[Scene]:
        """
        生成场景变体

        Args:
            base_scene: 基础场景
            num_variations: 变体数量
            variation_params: 变化参数

        Returns:
            List[Scene]: 场景变体列表
        """
        variations = []

        for i in range(num_variations):
            # 创建新的场景配置
            config_dict = base_scene.config.dict()
            config_dict["scene_id"] = f"scene_{self.scene_counter + i + 1}"
            config_dict["name"] = f"{base_scene.config.name}_variation_{i}"

            # 应用随机变化
            for key, range_value in variation_params.items():
                if isinstance(range_value, (list, tuple)):
                    value = np.random.uniform(range_value[0], range_value[1])
                else:
                    value = np.random.uniform(-range_value, range_value)

                self._apply_variations(config_dict, {key: value})

            config = SceneConfig(**config_dict)

            # 生成对象
            objects = self._generate_objects_from_config(config)

            # 创建场景
            scene = Scene(
                scene_id=config.scene_id,
                config=config,
                objects=objects,
                metadata={"source": "variation", "base_scene_id": base_scene.scene_id}
            )

            variations.append(scene)

        # 更新计数器
        self.scene_counter += num_variations

        # 保存场景
        for scene in variations:
            self.scenes[scene.scene_id] = scene

        return variations

    def get_scene(self, scene_id: str) -> Optional[Scene]:
        """
        获取场景

        Args:
            scene_id: 场景ID

        Returns:
            Scene: 场景
        """
        return self.scenes.get(scene_id)

    def get_scenes_by_type(self, scene_type: SceneType) -> List[Scene]:
        """
        根据类型获取场景

        Args:
            scene_type: 场景类型

        Returns:
            List[Scene]: 场景列表
        """
        return [
            scene for scene in self.scenes.values()
            if scene.config.scene_type == scene_type
        ]

    def delete_scene(self, scene_id: str) -> bool:
        """
        删除场景

        Args:
            scene_id: 场景ID

        Returns:
            bool: 删除是否成功
        """
        if scene_id not in self.scenes:
            return False

        del self.scenes[scene_id]
        return True

    def export_scene(self, scene_id: str, output_path: str) -> bool:
        """
        导出场景

        Args:
            scene_id: 场景ID
            output_path: 输出路径

        Returns:
            bool: 导出是否成功
        """
        if scene_id not in self.scenes:
            return False

        try:
            scene = self.scenes[scene_id]
            scene_dict = scene.dict()

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(scene_dict, f, indent=2, ensure_ascii=False)

            return True
        except Exception as e:
            print(f"Failed to export scene: {e}")
            return False

    def import_scene(self, input_path: str) -> Optional[Scene]:
        """
        导入场景

        Args:
            input_path: 输入路径

        Returns:
            Scene: 场景
        """
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            scene = Scene(**data)
            self.scenes[scene.scene_id] = scene

            return scene
        except Exception as e:
            print(f"Failed to import scene: {e}")
            return None

    def get_all_scenes(self) -> Dict[str, Scene]:
        """获取所有场景"""
        return self.scenes.copy()

    def get_summary(self) -> Dict[str, Any]:
        """获取摘要"""
        total_scenes = len(self.scenes)

        # 按类型统计
        type_stats = {}
        for scene in self.scenes.values():
            scene_type = scene.config.scene_type.value
            type_stats[scene_type] = type_stats.get(scene_type, 0) + 1

        # 按天气统计
        weather_stats = {}
        for scene in self.scenes.values():
            weather = scene.config.weather.value
            weather_stats[weather] = weather_stats.get(weather, 0) + 1

        return {
            "total_scenes": total_scenes,
            "type_stats": type_stats,
            "weather_stats": weather_stats
        }


# 全局场景生成器实例
_global_scene_generator = None


def get_global_scene_generator() -> SceneGenerator:
    """获取全局场景生成器实例"""
    global _global_scene_generator
    if _global_scene_generator is None:
        _global_scene_generator = SceneGenerator()
    return _global_scene_generator