"""
自动化结构化提取模块 (Auto-Labeling)
使用离线大模型（Teacher Model）对视频进行高精度标注，生成几何元数据
支持细粒度分类和属性标注
"""

from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field
from enum import Enum
import time
import json
import numpy as np
from dataclasses import dataclass

# 导入细粒度Schema
from .fine_grained_schema import (
    VehicleType, VRUType, ObstacleType, AnimalType,
    LightState, DoorState, PedestrianPose, OcclusionLevel,
    RoadMarkingType, TrafficLightShape, RoadSurfaceType,
    WeatherType, LightingType, RoadType, CameraDirtType,
    EgoRelationType, TrafficFlowType, ViolationType,
    VoxelState, OccupancyFlow,
    FineGrainedObject, StaticRoadElement, SceneTag, BehaviorTag,
    FineGrainedAnnotation
)


class ObjectType(str, Enum):
    """对象类型（基础）"""
    VEHICLE = "vehicle"
    PEDESTRIAN = "pedestrian"
    CYCLIST = "cyclist"
    TRAFFIC_LIGHT = "traffic_light"
    TRAFFIC_SIGN = "traffic_sign"
    LANE_MARKING = "lane_marking"
    ROAD_EDGE = "road_edge"
    BARRIER = "barrier"
    OBSTACLE = "obstacle"
    ANIMAL = "animal"
    UNKNOWN = "unknown"


class TrafficLightState(str, Enum):
    """红绿灯状态"""
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"
    UNKNOWN = "unknown"


class BoundingBox3D(BaseModel):
    """3D边界框"""
    center: Dict[str, float] = Field(description="中心点坐标 (x, y, z)")
    size: Dict[str, float] = Field(description="尺寸 (width, length, height)")
    rotation: Dict[str, float] = Field(default_factory=lambda: {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}, description="旋转角度")
    confidence: float = Field(default=1.0, description="置信度")


class ObjectTrack(BaseModel):
    """对象轨迹"""
    object_id: str = Field(description="对象ID")
    object_type: ObjectType = Field(description="对象类型")
    sub_type: Optional[str] = Field(default=None, description="细分类型")
    track_id: str = Field(description="跟踪ID")
    frames: List[int] = Field(description="帧索引列表")
    bboxes: List[BoundingBox3D] = Field(description="3D边界框列表")
    velocities: List[Dict[str, float]] = Field(default_factory=list, description="速度列表")
    attributes: Dict[str, Any] = Field(default_factory=dict, description="属性")
    
    # 细粒度属性
    vehicle_lights: Optional[Dict[str, LightState]] = Field(default=None, description="车灯状态")
    door_state: Optional[DoorState] = Field(default=None, description="车门状态")
    pedestrian_pose: Optional[PedestrianPose] = Field(default=None, description="行人姿态")
    occlusion_level: OcclusionLevel = Field(default=OcclusionLevel.NONE, description="遮挡等级")
    
    avg_confidence: float = Field(default=1.0, description="平均置信度")


class LaneInfo(BaseModel):
    """车道信息"""
    lane_id: str = Field(description="车道ID")
    lane_type: str = Field(description="车道类型 (driving, parking, shoulder)")
    center_line: List[Dict[str, float]] = Field(description="中心线点集")
    left_boundary: List[Dict[str, float]] = Field(description="左边界")
    right_boundary: List[Dict[str, float]] = Field(description="右边界")
    speed_limit: Optional[float] = Field(default=None, description="限速")
    direction: str = Field(default="forward", description="方向 (forward, backward)")
    
    # 细粒度属性
    road_marking_type: Optional[RoadMarkingType] = Field(default=None, description="地面标识类型")
    road_surface_type: Optional[RoadSurfaceType] = Field(default=None, description="路面类型")


class TrafficLightInfo(BaseModel):
    """红绿灯信息"""
    light_id: str = Field(description="红绿灯ID")
    position: Dict[str, float] = Field(description="位置")
    state: TrafficLightState = Field(description="状态")
    shape: Optional[TrafficLightShape] = Field(default=None, description="形状")
    countdown: Optional[int] = Field(default=None, description="倒计时读秒")
    direction: str = Field(description="方向")
    confidence: float = Field(default=1.0, description="置信度")


class FrameAnnotation(BaseModel):
    """单帧标注"""
    frame_id: int = Field(description="帧ID")
    timestamp: float = Field(description="时间戳")
    objects: List[ObjectTrack] = Field(default_factory=list, description="对象列表")
    lanes: List[LaneInfo] = Field(default_factory=list, description="车道列表")
    traffic_lights: List[TrafficLightInfo] = Field(default_factory=list, description="红绿灯列表")
    ego_vehicle: Dict[str, Any] = Field(default_factory=dict, description="自车状态")
    
    # 静态道路要素
    static_elements: List[StaticRoadElement] = Field(default_factory=list, description="静态道路要素")
    
    # 场景标签
    scene_tag: Optional[SceneTag] = Field(default=None, description="场景标签")
    
    # 行为标签
    behavior_tags: List[BehaviorTag] = Field(default_factory=list, description="行为标签")
    
    # 占用流（可选）
    occupancy_flow: Optional[OccupancyFlow] = Field(default=None, description="占用流数据")


class VideoAnnotation(BaseModel):
    """视频标注结果"""
    video_id: str = Field(description="视频ID")
    clip_id: str = Field(description="片段ID")
    start_time: float = Field(description="起始时间")
    end_time: float = Field(description="结束时间")
    duration: float = Field(description="持续时间")
    fps: float = Field(default=30.0, description="帧率")
    total_frames: int = Field(description="总帧数")
    frames: List[FrameAnnotation] = Field(description="帧标注列表")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")

    def get_object_tracks(self) -> Dict[str, ObjectTrack]:
        """获取所有对象轨迹"""
        tracks = {}

        for frame in self.frames:
            for obj in frame.objects:
                if obj.track_id not in tracks:
                    tracks[obj.track_id] = obj

        return tracks

    def get_objects_by_type(self, obj_type: ObjectType) -> List[ObjectTrack]:
        """根据类型获取对象"""
        objects = []
        for frame in self.frames:
            for obj in frame.objects:
                if obj.object_type == obj_type and obj.track_id not in [o.track_id for o in objects]:
                    objects.append(obj)
        return objects

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_objects = len(self.get_object_tracks())
        total_lanes = len(self.frames[0].lanes) if self.frames else 0

        # 按类型统计
        type_stats = {}
        for track in self.get_object_tracks().values():
            obj_type = track.object_type.value
            type_stats[obj_type] = type_stats.get(obj_type, 0) + 1

        return {
            "total_objects": total_objects,
            "total_lanes": total_lanes,
            "type_stats": type_stats,
            "total_frames": self.total_frames,
            "duration": self.duration
        }


class AutoLabelingConfig(BaseModel):
    """自动标注配置"""
    model_name: str = Field(default="teacher_model_v2", description="模型名称")
    model_path: str = Field(description="模型路径")
    device: str = Field(default="cuda:0", description="设备")
    batch_size: int = Field(default=8, description="批处理大小")
    confidence_threshold: float = Field(default=0.5, description="置信度阈值")
    max_objects: int = Field(default=100, description="最大对象数")
    
    # 细粒度标注配置
    enable_fine_grained: bool = Field(default=True, description="是否启用细粒度标注")
    enable_vehicle_lights: bool = Field(default=True, description="是否检测车灯")
    enable_door_state: bool = Field(default=True, description="是否检测车门状态")
    enable_pedestrian_pose: bool = Field(default=True, description="是否检测行人姿态")
    enable_occlusion: bool = Field(default=True, description="是否检测遮挡")
    
    # 静态要素配置
    enable_lanes: bool = Field(default=True, description="是否检测车道")
    enable_traffic_lights: bool = Field(default=True, description="是否检测红绿灯")
    enable_road_markings: bool = Field(default=True, description="是否检测地面标识")
    enable_road_surface: bool = Field(default=True, description="是否检测路面特征")
    
    # 场景标签配置
    enable_scene_tagging: bool = Field(default=True, description="是否启用场景标签")
    enable_behavior_tagging: bool = Field(default=True, description="是否启用行为标签")
    
    # 占用流配置
    enable_occupancy_flow: bool = Field(default=False, description="是否启用占用流")
    occupancy_resolution: float = Field(default=0.1, description="占用流分辨率（米）")


class AutoLabelingEngine:
    """
    自动标注引擎
    使用离线大模型对视频进行高精度标注，支持细粒度分类和属性标注
    """

    def __init__(self, config: AutoLabelingConfig):
        self.config = config
        self.model = None
        self._load_model()

    def _load_model(self):
        """加载模型"""
        # TODO: 实际使用时加载真实的离线大模型
        # 这里使用模拟实现
        print(f"Loading teacher model from {self.config.model_path}")
        self.model = {
            "name": self.config.model_name,
            "loaded": True,
            "fine_grained": self.config.enable_fine_grained
        }

    def process_video(
        self,
        video_path: str,
        clip_id: str,
        start_time: float = 0.0,
        end_time: Optional[float] = None
    ) -> VideoAnnotation:
        """
        处理视频，生成标注

        Args:
            video_path: 视频路径
            clip_id: 片段ID
            start_time: 起始时间
            end_time: 结束时间

        Returns:
            VideoAnnotation: 视频标注结果
        """
        print(f"Processing video: {video_path}")

        # 模拟视频处理
        duration = end_time - start_time if end_time else 10.0
        fps = self.config.fps if hasattr(self.config, 'fps') else 30.0
        total_frames = int(duration * fps)

        # 生成模拟标注
        frames = []
        for frame_idx in range(total_frames):
            frame_annotation = self._process_frame(frame_idx, frame_idx / fps)
            frames.append(frame_annotation)

        annotation = VideoAnnotation(
            video_id=video_path,
            clip_id=clip_id,
            start_time=start_time,
            end_time=end_time or start_time + duration,
            duration=duration,
            fps=fps,
            total_frames=total_frames,
            frames=frames,
            metadata={
                "model": self.config.model_name,
                "processing_time": time.time()
            }
        )

        return annotation

    def _process_frame(self, frame_id: int, timestamp: float) -> FrameAnnotation:
        """
        处理单帧

        Args:
            frame_id: 帧ID
            timestamp: 时间戳

        Returns:
            FrameAnnotation: 帧标注
        """
        # 模拟对象检测
        objects = self._detect_objects(frame_id, timestamp)

        # 模拟车道检测
        lanes = self._detect_lanes(frame_id) if self.config.enable_lanes else []

        # 模拟红绿灯检测
        traffic_lights = self._detect_traffic_lights(frame_id) if self.config.enable_traffic_lights else []

        # 模拟静态道路要素检测
        static_elements = self._detect_static_elements(frame_id) if self.config.enable_road_markings or self.config.enable_road_surface else []

        # 模拟自车状态
        ego_vehicle = self._get_ego_vehicle_state(timestamp)

        # 生成场景标签
        scene_tag = self._generate_scene_tag(frame_id, timestamp) if self.config.enable_scene_tagging else None

        # 生成行为标签
        behavior_tags = self._generate_behavior_tags(frame_id, objects) if self.config.enable_behavior_tagging else []

        # 生成占用流
        occupancy_flow = self._generate_occupancy_flow(frame_id) if self.config.enable_occupancy_flow else None

        return FrameAnnotation(
            frame_id=frame_id,
            timestamp=timestamp,
            objects=objects,
            lanes=lanes,
            traffic_lights=traffic_lights,
            ego_vehicle=ego_vehicle,
            static_elements=static_elements,
            scene_tag=scene_tag,
            behavior_tags=behavior_tags,
            occupancy_flow=occupancy_flow
        )

    def _detect_objects(self, frame_id: int, timestamp: float) -> List[ObjectTrack]:
        """检测对象（模拟）"""
        objects = []

        # 模拟检测到的对象
        if frame_id % 30 == 0:  # 每30帧检测到一个新对象
            # 细粒度车辆类型
            vehicle_types = [
                VehicleType.SEDAN,
                VehicleType.SUV,
                VehicleType.TRUCK,
                VehicleType.POLICE_CAR,
                VehicleType.AMBULANCE,
                VehicleType.TRAILER
            ]
            import random
            vehicle_type = random.choice(vehicle_types)
            
            obj = ObjectTrack(
                object_id=f"obj_{frame_id}",
                object_type=ObjectType.VEHICLE,
                sub_type=vehicle_type.value,
                track_id=f"track_{frame_id // 30}",
                frames=[frame_id],
                bboxes=[BoundingBox3D(
                    center={"x": 20.0, "y": 0.0, "z": 0.0},
                    size={"width": 2.0, "length": 4.5, "height": 1.5},
                    confidence=0.95
                )],
                velocities=[{"x": 10.0, "y": 0.0, "z": 0.0}],
                attributes={"color": "white", "brand": "unknown"},
                
                # 细粒度属性
                vehicle_lights={
                    "brake": LightState.BRAKE,
                    "turn_left": LightState.OFF,
                    "turn_right": LightState.OFF
                } if self.config.enable_vehicle_lights else None,
                door_state=DoorState.CLOSED if self.config.enable_door_state else None,
                occlusion_level=OcclusionLevel.NONE if self.config.enable_occlusion else OcclusionLevel.UNKNOWN,
                avg_confidence=0.95
            )
            objects.append(obj)
        
        # 模拟检测行人
        if frame_id % 45 == 0:
            vru_types = [
                VRUType.PEDESTRIAN,
                VRUType.CHILD,
                VRUType.WHEELCHAIR,
                VRUType.CYCLIST
            ]
            import random
            vru_type = random.choice(vru_types)
            
            obj = ObjectTrack(
                object_id=f"obj_{frame_id}_ped",
                object_type=ObjectType.PEDESTRIAN,
                sub_type=vru_type.value,
                track_id=f"track_ped_{frame_id // 45}",
                frames=[frame_id],
                bboxes=[BoundingBox3D(
                    center={"x": 15.0, "y": 2.0, "z": 0.0},
                    size={"width": 0.6, "length": 0.3, "height": 1.7},
                    confidence=0.90
                )],
                velocities=[{"x": 2.0, "y": 0.0, "z": 0.0}],
                attributes={"clothing_color": "blue"},
                
                # 细粒度属性
                pedestrian_pose=PedestrianPose.WALKING if self.config.enable_pedestrian_pose else None,
                occlusion_level=OcclusionLevel.LIGHT if self.config.enable_occlusion else OcclusionLevel.UNKNOWN,
                avg_confidence=0.90
            )
            objects.append(obj)
        
        # 模拟检测障碍物
        if frame_id % 60 == 0:
            obstacle_types = [
                ObstacleType.TRAFFIC_CONE,
                ObstacleType.WATER_BARRIER,
                ObstacleType.BOX
            ]
            import random
            obstacle_type = random.choice(obstacle_types)
            
            obj = ObjectTrack(
                object_id=f"obj_{frame_id}_obs",
                object_type=ObjectType.OBSTACLE,
                sub_type=obstacle_type.value,
                track_id=f"track_obs_{frame_id // 60}",
                frames=[frame_id],
                bboxes=[BoundingBox3D(
                    center={"x": 25.0, "y": -1.0, "z": 0.5},
                    size={"width": 0.3, "length": 0.3, "height": 0.8},
                    confidence=0.85
                )],
                velocities=[{"x": 0.0, "y": 0.0, "z": 0.0}],
                attributes={"material": "plastic"},
                
                # 细粒度属性
                occlusion_level=OcclusionLevel.NONE if self.config.enable_occlusion else OcclusionLevel.UNKNOWN,
                avg_confidence=0.85
            )
            objects.append(obj)

        return objects

    def _detect_lanes(self, frame_id: int) -> List[LaneInfo]:
        """检测车道（模拟）"""
        lanes = []

        # 模拟检测到的车道
        lane = LaneInfo(
            lane_id="lane_0",
            lane_type="driving",
            center_line=[
                {"x": 0.0, "y": -1.75, "z": 0.0},
                {"x": 100.0, "y": -1.75, "z": 0.0}
            ],
            left_boundary=[
                {"x": 0.0, "y": -3.5, "z": 0.0},
                {"x": 100.0, "y": -3.5, "z": 0.0}
            ],
            right_boundary=[
                {"x": 0.0, "y": 0.0, "z": 0.0},
                {"x": 100.0, "y": 0.0, "z": 0.0}
            ],
            speed_limit=60.0,
            direction="forward"
        )
        lanes.append(lane)

        return lanes

    def _detect_traffic_lights(self, frame_id: int) -> List[TrafficLightInfo]:
        """检测红绿灯（模拟）"""
        lights = []

        # 模拟检测到的红绿灯
        if frame_id % 100 == 0:
            # 细粒度红绿灯类型
            light_states = [TrafficLightState.RED, TrafficLightState.YELLOW, TrafficLightState.GREEN]
            light_shapes = [TrafficLightShape.CIRCULAR, TrafficLightShape.ARROW, TrafficLightShape.COUNTDOWN]
            
            import random
            light_state = random.choice(light_states)
            light_shape = random.choice(light_shapes)
            countdown = random.randint(0, 30) if light_shape == TrafficLightShape.COUNTDOWN else None
            
            light = TrafficLightInfo(
                light_id="light_0",
                position={"x": 50.0, "y": 5.0, "z": 3.0},
                state=light_state,
                shape=light_shape,
                countdown=countdown,
                direction="forward",
                confidence=0.98
            )
            lights.append(light)

        return lights

    def _detect_static_elements(self, frame_id: int) -> List[StaticRoadElement]:
        """检测静态道路要素（模拟）"""
        elements = []

        # 模拟检测地面标识
        if frame_id % 50 == 0:
            marking_types = [
                RoadMarkingType.CROSSWALK,
                RoadMarkingType.STOP_LINE,
                RoadMarkingType.TURN_ARROW,
                RoadMarkingType.TEXT
            ]
            import random
            marking_type = random.choice(marking_types)
            
            element = StaticRoadElement(
                element_id=f"marking_{frame_id}",
                element_type="road_marking",
                geometry={
                    "type": "line",
                    "points": [{"x": 0.0, "y": 0.0, "z": 0.0}, {"x": 10.0, "y": 0.0, "z": 0.0}]
                },
                position={"x": 5.0, "y": 0.0, "z": 0.0},
                road_marking_type=marking_type,
                confidence=0.95
            )
            elements.append(element)

        # 模拟检测路面特征
        if frame_id % 80 == 0:
            surface_types = [
                RoadSurfaceType.SPEED_BUMP,
                RoadSurfaceType.POTHOLE,
                RoadSurfaceType.MANHOLE
            ]
            import random
            surface_type = random.choice(surface_types)
            
            element = StaticRoadElement(
                element_id=f"surface_{frame_id}",
                element_type="road_surface",
                geometry={
                    "type": "region",
                    "center": {"x": 30.0, "y": -1.75, "z": 0.0},
                    "size": {"width": 2.0, "length": 2.0, "height": 0.1}
                },
                position={"x": 30.0, "y": -1.75, "z": 0.0},
                road_surface_type=surface_type,
                confidence=0.90
            )
            elements.append(element)

        return elements

    def _generate_scene_tag(self, frame_id: int, timestamp: float) -> SceneTag:
        """生成场景标签（模拟）"""
        import random
        
        weather_types = [
            WeatherType.CLEAR,
            WeatherType.RAIN_LIGHT,
            WeatherType.RAIN_HEAVY,
            WeatherType.FOG_LIGHT
        ]
        lighting_types = [
            LightingType.DAY,
            LightingType.NIGHT,
            LightingType.DUSK,
            LightingType.GLARE
        ]
        road_types = [
            RoadType.HIGHWAY,
            RoadType.URBAN_ROAD,
            RoadType.INTERSECTION,
            RoadType.CONSTRUCTION_ZONE
        ]
        dirt_types = [
            CameraDirtType.CLEAN,
            CameraDirtType.MUD,
            CameraDirtType.WATER,
            CameraDirtType.FLARE
        ]
        flow_types = [
            TrafficFlowType.FREE_FLOW,
            TrafficFlowType.MODERATE,
            TrafficFlowType.HEAVY
        ]
        
        return SceneTag(
            tag_id=f"scene_tag_{frame_id}",
            weather=random.choice(weather_types),
            lighting=random.choice(lighting_types),
            road_type=random.choice(road_types),
            camera_dirt=random.choice(dirt_types),
            traffic_flow=random.choice(flow_types),
            tags={"timestamp": timestamp}
        )

    def _generate_behavior_tags(self, frame_id: int, objects: List[ObjectTrack]) -> List[BehaviorTag]:
        """生成行为标签（模拟）"""
        behavior_tags = []
        
        import random
        
        for obj in objects:
            # 为车辆生成行为标签
            if obj.object_type == ObjectType.VEHICLE:
                relation_types = [
                    EgoRelationType.CIPV,
                    EgoRelationType.CUT_IN,
                    EgoRelationType.ONCOMING,
                    EgoRelationType.FOLLOWING
                ]
                relation = random.choice(relation_types)
                
                tag = BehaviorTag(
                    tag_id=f"behavior_{obj.track_id}_{frame_id}",
                    object_id=obj.object_id,
                    ego_relation=relation,
                    behavior_description=f"车辆正在{relation.value}",
                    confidence=random.uniform(0.7, 0.95),
                    timestamp=frame_id / 30.0
                )
                behavior_tags.append(tag)
            
            # 为行人生成行为标签
            elif obj.object_type == ObjectType.PEDESTRIAN:
                violation_types = [
                    ViolationType.CROSSING_ILLEGALLY,
                    ViolationType.NONE
                ]
                violation = random.choice(violation_types)
                
                if violation != ViolationType.NONE:
                    tag = BehaviorTag(
                        tag_id=f"behavior_{obj.track_id}_{frame_id}",
                        object_id=obj.object_id,
                        ego_relation=EgoRelationType.INTERACTING,
                        violation_type=violation,
                        behavior_description=f"行人正在{violation.value}",
                        confidence=random.uniform(0.7, 0.95),
                        timestamp=frame_id / 30.0
                    )
                    behavior_tags.append(tag)

        return behavior_tags

    def _generate_occupancy_flow(self, frame_id: int) -> Optional[OccupancyFlow]:
        """生成占用流数据（模拟）"""
        if not self.config.enable_occupancy_flow:
            return None
        
        # 模拟3D占用网格
        grid_size = {"x": 100, "y": 20, "z": 5}
        resolution = self.config.occupancy_resolution
        
        # 初始化网格
        voxel_grid = [[[0 for _ in range(grid_size["z"])] 
                       for _ in range(grid_size["y"])] 
                       for _ in range(grid_size["x"])]
        
        motion_vectors = [[[{"x": 0.0, "y": 0.0, "z": 0.0} 
                           for _ in range(grid_size["z"])] 
                           for _ in range(grid_size["y"])] 
                           for _ in range(grid_size["x"])]
        
        confidence = [[[0.0 for _ in range(grid_size["z"])] 
                       for _ in range(grid_size["y"])] 
                       for _ in range(grid_size["x"])]
        
        # 模拟一些被占据的体素
        import random
        for _ in range(50):  # 随机占据50个体素
            x = random.randint(0, grid_size["x"] - 1)
            y = random.randint(0, grid_size["y"] - 1)
            z = random.randint(0, grid_size["z"] - 1)
            
            voxel_grid[x][y][z] = 1  # 占据
            confidence[x][y][z] = random.uniform(0.7, 0.95)
            
            # 随机运动向量
            motion_vectors[x][y][z] = {
                "x": random.uniform(-1.0, 1.0),
                "y": random.uniform(-0.5, 0.5),
                "z": 0.0
            }
        
        return OccupancyFlow(
            voxel_grid=voxel_grid,
            motion_vectors=motion_vectors,
            confidence=confidence,
            resolution=resolution,
            grid_size=grid_size,
            origin={"x": 0.0, "y": -10.0, "z": 0.0},
            timestamp=frame_id / 30.0
        )

    def _get_ego_vehicle_state(self, timestamp: float) -> Dict[str, Any]:
        """获取自车状态（模拟）"""
        return {
            "position": {"x": 0.0, "y": -1.75, "z": 0.0},
            "velocity": {"x": 15.0, "y": 0.0, "z": 0.0},
            "acceleration": {"x": 0.0, "y": 0.0, "z": 0.0},
            "heading": 0.0,
            "yaw_rate": 0.0
        }

    def save_annotation(self, annotation: VideoAnnotation, output_path: str) -> bool:
        """
        保存标注结果

        Args:
            annotation: 标注结果
            output_path: 输出路径

        Returns:
            bool: 是否成功
        """
        try:
            # 保存为JSON格式
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(annotation.dict(), f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Failed to save annotation: {e}")
            return False

    def export_to_parquet(self, annotation: VideoAnnotation, output_path: str) -> bool:
        """
        导出为Parquet格式（用于高效查询）

        Args:
            annotation: 标注结果
            output_path: 输出路径

        Returns:
            bool: 是否成功
        """
        try:
            import pandas as pd

            # 将标注转换为DataFrame
            data = []

            for frame in annotation.frames:
                for obj in frame.objects:
                    for i, bbox in enumerate(obj.bboxes):
                        data.append({
                            "clip_id": annotation.clip_id,
                            "frame_id": frame.frame_id,
                            "timestamp": frame.timestamp,
                            "object_id": obj.object_id,
                            "track_id": obj.track_id,
                            "object_type": obj.object_type.value,
                            "bbox_center_x": bbox.center["x"],
                            "bbox_center_y": bbox.center["y"],
                            "bbox_center_z": bbox.center["z"],
                            "bbox_width": bbox.size["width"],
                            "bbox_length": bbox.size["length"],
                            "bbox_height": bbox.size["height"],
                            "confidence": bbox.confidence,
                            "ego_speed": frame.ego_vehicle.get("velocity", {}).get("x", 0.0)
                        })

            df = pd.DataFrame(data)

            # 保存为Parquet
            df.to_parquet(output_path, index=False)

            return True
        except Exception as e:
            print(f"Failed to export to parquet: {e}")
            return False

    def query_by_rules(
        self,
        annotations: List[VideoAnnotation],
        rules: Dict[str, Any]
    ) -> List[str]:
        """
        根据规则查询标注数据

        Args:
            annotations: 标注列表
            rules: 查询规则，例如：
                {
                    "ego_speed_min": 60.0,
                    "time_to_collision_max": 2.0,
                    "object_types": ["vehicle", "pedestrian"],
                    "vehicle_sub_types": ["police_car", "ambulance"],
                    "vehicle_lights": ["brake"],
                    "pedestrian_poses": ["child", "wheelchair"],
                    "weather": ["rain_heavy"],
                    "road_type": ["construction_zone"]
                }

        Returns:
            List[str]: 匹配的clip_id列表
        """
        matched_clips = []

        for annotation in annotations:
            match = True

            # 检查自车速度
            if "ego_speed_min" in rules:
                for frame in annotation.frames:
                    ego_speed = frame.ego_vehicle.get("velocity", {}).get("x", 0.0)
                    if ego_speed < rules["ego_speed_min"]:
                        match = False
                        break

            # 检查对象类型
            if match and "object_types" in rules:
                found_types = set()
                for frame in annotation.frames:
                    for obj in frame.objects:
                        found_types.add(obj.object_type.value)

                required_types = set(rules["object_types"])
                if not required_types.issubset(found_types):
                    match = False

            # 检查车辆细分类型
            if match and "vehicle_sub_types" in rules:
                found_sub_types = set()
                for frame in annotation.frames:
                    for obj in frame.objects:
                        if obj.sub_type:
                            found_sub_types.add(obj.sub_type)

                required_sub_types = set(rules["vehicle_sub_types"])
                if not required_sub_types.issubset(found_sub_types):
                    match = False

            # 检查车灯状态
            if match and "vehicle_lights" in rules:
                found_lights = set()
                for frame in annotation.frames:
                    for obj in frame.objects:
                        if obj.vehicle_lights:
                            for light, state in obj.vehicle_lights.items():
                                found_lights.add(state.value)

                required_lights = set(rules["vehicle_lights"])
                if not required_lights.issubset(found_lights):
                    match = False

            # 检查行人姿态
            if match and "pedestrian_poses" in rules:
                found_poses = set()
                for frame in annotation.frames:
                    for obj in frame.objects:
                        if obj.pedestrian_pose:
                            found_poses.add(obj.pedestrian_pose.value)

                required_poses = set(rules["pedestrian_poses"])
                if not required_poses.issubset(found_poses):
                    match = False

            # 检查天气
            if match and "weather" in rules:
                found_weather = set()
                for frame in annotation.frames:
                    if frame.scene_tag:
                        found_weather.add(frame.scene_tag.weather.value)

                required_weather = set(rules["weather"])
                if not required_weather.issubset(found_weather):
                    match = False

            # 检查道路类型
            if match and "road_type" in rules:
                found_road_types = set()
                for frame in annotation.frames:
                    if frame.scene_tag:
                        found_road_types.add(frame.scene_tag.road_type.value)

                required_road_types = set(rules["road_type"])
                if not required_road_types.issubset(found_road_types):
                    match = False

            # 检查碰撞时间（模拟）
            if match and "time_to_collision_max" in rules:
                # TODO: 实际计算TTC
                pass

            if match:
                matched_clips.append(annotation.clip_id)

        return matched_clips

    def query_by_fine_grained(
        self,
        annotations: List[VideoAnnotation],
        query: Dict[str, Any]
    ) -> List[str]:
        """
        根据细粒度条件查询

        Args:
            annotations: 标注列表
            query: 查询条件，例如：
                {
                    "special_vehicles": True,  # 特种车辆
                    "large_vehicles": True,     # 异形大车
                    "vru_sub_types": ["child", "wheelchair"],  # VRU细分
                    "small_obstacles": True,     # 小型障碍物
                    "brake_lights": True,       # 刹车灯
                    "distracted_pedestrians": True,  # 分心行人
                    "construction_zone": True    # 施工区域
                }

        Returns:
            List[str]: 匹配的clip_id列表
        """
        matched_clips = []

        for annotation in annotations:
            match = True

            for frame in annotation.frames:
                # 检查特种车辆
                if query.get("special_vehicles"):
                    special_types = ["police_car", "ambulance", "fire_truck", "school_bus"]
                    found = any(
                        obj.sub_type in special_types
                        for obj in frame.objects
                        if obj.sub_type
                    )
                    if not found:
                        match = False
                        break

                # 检查异形大车
                if match and query.get("large_vehicles"):
                    large_types = ["trailer", "concrete_mixer", "crane", "water_truck"]
                    found = any(
                        obj.sub_type in large_types
                        for obj in frame.objects
                        if obj.sub_type
                    )
                    if not found:
                        match = False
                        break

                # 检查VRU细分
                if match and "vru_sub_types" in query:
                    found = any(
                        obj.sub_type in query["vru_sub_types"]
                        for obj in frame.objects
                        if obj.sub_type
                    )
                    if not found:
                        match = False
                        break

                # 检查小型障碍物
                if match and query.get("small_obstacles"):
                    obstacle_types = ["traffic_cone", "water_barrier", "crash_barrier"]
                    found = any(
                        obj.sub_type in obstacle_types
                        for obj in frame.objects
                        if obj.sub_type
                    )
                    if not found:
                        match = False
                        break

                # 检查刹车灯
                if match and query.get("brake_lights"):
                    found = any(
                        obj.vehicle_lights and obj.vehicle_lights.get("brake") == LightState.BRAKE
                        for obj in frame.objects
                    )
                    if not found:
                        match = False
                        break

                # 检查分心行人
                if match and query.get("distracted_pedestrians"):
                    found = any(
                        obj.pedestrian_pose == PedestrianPose.DISTRACTED
                        for obj in frame.objects
                    )
                    if not found:
                        match = False
                        break

                # 检查施工区域
                if match and query.get("construction_zone"):
                    if frame.scene_tag and frame.scene_tag.road_type != RoadType.CONSTRUCTION_ZONE:
                        match = False
                        break

            if match:
                matched_clips.append(annotation.clip_id)

        return matched_clips


# 便捷函数
def create_auto_labeling_engine(model_path: str = "./models/teacher_model") -> AutoLabelingEngine:
    """创建自动标注引擎"""
    config = AutoLabelingConfig(
        model_path=model_path,
        confidence_threshold=0.5
    )
    return AutoLabelingEngine(config)