"""
细粒度对象分类和属性定义
支持高阶自动驾驶（Urban NOA、L3/L4）的自动标注需求
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum


# ========== 细粒度物体分类 ==========

class VehicleType(str, Enum):
    """车辆类型细分"""
    # 普通车辆
    SEDAN = "sedan"                    # 轿车
    SUV = "suv"                        # SUV
    HATCHBACK = "hatchback"            # 掀背车
    COUPE = "coupe"                    # 跑车
    VAN = "van"                        # 面包车
    MINIBUS = "minibus"                # 小巴
    BUS = "bus"                        # 公交车
    TRUCK = "truck"                    # 卡车
    
    # 特种车辆（需要特殊避让策略）
    POLICE_CAR = "police_car"          # 警车
    AMBULANCE = "ambulance"            # 救护车
    FIRE_TRUCK = "fire_truck"          # 消防车
    SCHOOL_BUS = "school_bus"          # 校车
    
    # 异形大车
    TRAILER = "trailer"                # 挂车
    CONCRETE_MIXER = "concrete_mixer"  # 混凝土搅拌车
    CRANE = "crane"                    # 起重机/工程车
    WATER_TRUCK = "water_truck"        # 洒水车
    GARBAGE_TRUCK = "garbage_truck"    # 垃圾车
    TANKER = "tanker"                  # 罐车
    
    # 特殊用途
    TAXI = "taxi"                      # 出租车
    RIDESHARE = "rideshare"            # 网约车
    DELIVERY_VEHICLE = "delivery"      # 配送车
    CONSTRUCTION_VEHICLE = "construction"  # 工程车
    
    UNKNOWN = "unknown"


class VRUType(str, Enum):
    """弱势交通参与者（VRU）细分"""
    # 行人
    PEDESTRIAN = "pedestrian"          # 普通行人
    CHILD = "child"                    # 儿童（行为不可测）
    ELDERLY = "elderly"                # 老人
    WHEELCHAIR = "wheelchair"          # 轮椅使用者
    STROLLER = "stroller"              # 推婴儿车的人
    CYCLIST = "cyclist"                # 骑自行车的人
    MOTORCYCLIST = "motorcyclist"      # 骑摩托车的人
    TRICYCLIST = "tricyclist"          # 骑三轮车的人
    E_SCOOTER = "e_scooter"            # 电动滑板车
    SKATEBOARDER = "skateboarder"      # 滑板使用者
    
    UNKNOWN = "unknown"


class ObstacleType(str, Enum):
    """障碍物类型"""
    # 交通设施
    TRAFFIC_CONE = "traffic_cone"      # 锥桶
    WATER_BARRIER = "water_barrier"    # 水马
    CRASH_BARRIER = "crash_barrier"    # 防撞桶
    BOLLARD = "bollard"                # 护柱/地锁
    GATE = "gate"                      # 闸门
    
    # 道路障碍
    SPEED_BUMP = "speed_bump"          # 减速带
    MANHOLE_COVER = "manhole_cover"    # 井盖
    POTHOLE = "pothole"                # 路面坑洼
    PATCH = "patch"                    # 补丁
    
    # 散落物体
    TIRE = "tire"                      # 轮胎
    BOX = "box"                        # 纸箱
    BAG = "bag"                        # 袋子
    DEBRIS = "debris"                  # 碎片/杂物
    ANIMAL_CARCASS = "animal_carcass"  # 动物尸体
    
    UNKNOWN = "unknown"


class AnimalType(str, Enum):
    """动物类型"""
    DOG = "dog"                        # 狗
    CAT = "cat"                        # 猫
    COW = "cow"                        # 牛
    HORSE = "horse"                    # 马
    SHEEP = "sheep"                    # 羊
    WILD_ANIMAL = "wild_animal"        # 野生动物
    
    UNKNOWN = "unknown"


# ========== 关键物体属性 ==========

class LightState(str, Enum):
    """车灯状态"""
    OFF = "off"
    ON = "on"
    BRAKE = "brake"                    # 刹车灯
    TURN_LEFT = "turn_left"            # 左转向灯
    TURN_RIGHT = "turn_right"          # 右转向灯
    HAZARD = "hazard"                  # 双闪灯
    REVERSE = "reverse"                # 倒车灯
    HEADLIGHT_HIGH = "headlight_high"  # 远光灯
    HEADLIGHT_LOW = "headlight_low"    # 近光灯
    FOG_LIGHT = "fog_light"            # 雾灯
    
    UNKNOWN = "unknown"


class DoorState(str, Enum):
    """车门状态"""
    CLOSED = "closed"
    OPEN_LEFT = "open_left"            # 左门开启
    OPEN_RIGHT = "open_right"          # 右门开启
    OPEN_ALL = "open_all"              # 所有门开启
    
    UNKNOWN = "unknown"


class PedestrianPose(str, Enum):
    """行人姿态"""
    STANDING = "standing"              # 站立
    WALKING = "walking"                # 行走
    RUNNING = "running"                # 奔跑
    SITTING = "sitting"                # 坐着
    LAYING = "laying"                  # 躺着
    CROSSING = "crossing"              # 正在过马路
    WAITING = "waiting"                # 等待
    PUSHING = "pushing"                # 推车
    WITH_UMBRELLA = "with_umbrella"    # 打伞
    DISTRACTED = "distracted"          # 分心（看手机等）
    CARRYING_LOAD = "carrying_load"    # 搬运物品
    
    UNKNOWN = "unknown"


class OcclusionLevel(str, Enum):
    """遮挡等级"""
    NONE = "none"                      # 无遮挡
    LIGHT = "light"                    # 轻度遮挡
    MODERATE = "moderate"              # 中度遮挡
    SEVERE = "severe"                  # 严重遮挡
    TRUNCATED = "truncated"            # 截断（部分在画面外）
    
    UNKNOWN = "unknown"


# ========== 静态道路要素 ==========

class RoadMarkingType(str, Enum):
    """地面标识类型"""
    LANE_MARKING = "lane_marking"      # 车道线
    CROSSWALK = "crosswalk"            # 斑马线
    STOP_LINE = "stop_line"            # 停止线
    YIELD_LINE = "yield_line"          # 让行线
    TURN_ARROW = "turn_arrow"          # 转向箭头
    STRAIGHT_ARROW = "straight_arrow"  # 直行箭头
    U_TURN_ARROW = "u_turn_arrow"      # 掉头箭头
    TEXT = "text"                      # 地面文字（如"慢"、"公交专用"）
    DIAGONAL_CROSSING = "diagonal_crossing"  # 斑马线（对角线）
    ZEBRA = "zebra"                    # 斑马线（斑马纹）
    
    UNKNOWN = "unknown"


class TrafficLightShape(str, Enum):
    """红绿灯形状"""
    CIRCULAR = "circular"              # 圆饼灯
    ARROW = "arrow"                    # 箭头灯
    COUNTDOWN = "countdown"            # 倒计时灯
    PEDESTRIAN = "pedestrian"          # 行人灯
    
    UNKNOWN = "unknown"


class RoadSurfaceType(str, Enum):
    """路面特征"""
    ASPHALT = "asphalt"                # 沥青路面
    CONCRETE = "concrete"              # 混凝土路面
    GRAVEL = "gravel"                  # 碎石路面
    DIRT = "dirt"                      # 泥土路面
    BRICK = "brick"                    # 砖路面
    COBBLESTONE = "cobblestone"        # 鹅卵石路面
    SPEED_BUMP = "speed_bump"          # 减速带
    MANHOLE = "manhole"                # 井盖
    POTHOLE = "pothole"                # 坑洼
    PUDDLE = "puddle"                  # 积水
    ICE = "ice"                        # 冰面
    SNOW = "snow"                      # 积雪
    
    UNKNOWN = "unknown"


# ========== 场景与环境标签 ==========

class WeatherType(str, Enum):
    """天气类型"""
    CLEAR = "clear"                    # 晴天
    CLOUDY = "cloudy"                  # 多云
    OVERCAST = "overcast"              # 阴天
    RAIN_LIGHT = "rain_light"          # 小雨
    RAIN_MODERATE = "rain_moderate"    # 中雨
    RAIN_HEAVY = "rain_heavy"          # 大雨
    SNOW_LIGHT = "snow_light"          # 小雪
    SNOW_HEAVY = "snow_heavy"          # 大雪
    FOG_LIGHT = "fog_light"            # 轻雾
    FOG_HEAVY = "fog_heavy"            # 浓雾
    MIST = "mist"                      # 薄雾
    HAZE = "haze"                      # 霾
    SANDSTORM = "sandstorm"            # 沙尘暴
    
    UNKNOWN = "unknown"


class LightingType(str, Enum):
    """光照类型"""
    DAY = "day"                        # 白天
    NIGHT = "night"                    # 夜晚
    DUSK = "dusk"                      # 黄昏
    DAWN = "dawn"                      # 黎明
    TUNNEL_ENTRANCE = "tunnel_entrance"  # 隧道入口
    TUNNEL_EXIT = "tunnel_exit"        # 隧道出口
    TUNNEL_INSIDE = "tunnel_inside"    # 隧道内
    BACKLIGHT = "backlight"            # 逆光
    GLARE = "glare"                    # 眩光
    SHADOW = "shadow"                  # 阴影
    
    UNKNOWN = "unknown"


class RoadType(str, Enum):
    """道路类型"""
    HIGHWAY = "highway"                # 高速公路
    EXPRESSWAY = "expressway"          # 快速路
    URBAN_ROAD = "urban_road"          # 城市道路
    RESIDENTIAL = "residential"        # 居住区道路
    RURAL_ROAD = "rural_road"          # 乡村道路
    BRIDGE = "bridge"                  # 桥梁
    TUNNEL = "tunnel"                  # 隧道
    ROUNDABOUT = "roundabout"          # 环岛
    INTERSECTION = "intersection"      # 十字路口
    T_JUNCTION = "t_junction"          # T型路口
    RAMP = "ramp"                      # 匝道
    TOLL_BOOTH = "toll_booth"          # 收费站
    CONSTRUCTION_ZONE = "construction"  # 施工区域
    PARKING_LOT = "parking_lot"        # 停车场
    
    UNKNOWN = "unknown"


class CameraDirtType(str, Enum):
    """传感器脏污类型"""
    CLEAN = "clean"                    # 清洁
    MUD = "mud"                        # 泥点
    WATER = "water"                    # 水珠
    DUST = "dust"                      # 灰尘
    RAIN_DROPS = "rain_drops"          # 雨滴
    SPIDER_WEB = "spider_web"          # 蜘蛛网
    SCRATCH = "scratch"                # 划痕
    FINGERPRINT = "fingerprint"        # 指纹
    FLARE = "flare"                    # 耀斑
    
    UNKNOWN = "unknown"


# ========== 逻辑与行为标签 ==========

class EgoRelationType(str, Enum):
    """自车关系类型"""
    CIPV = "cipv"                      # Closest In-Path Vehicle（关键前车）
    CUT_IN = "cut_in"                  # 正在切入
    CUT_OUT = "cut_out"                # 正在切出
    ONCOMING = "oncoming"              # 对向来车
    FOLLOWING = "following"            # 跟随
    OVERTAKING = "overtaking"          # 超车
    BEING_OVERTAKEN = "being_overtaken"  # 被超车
    MERGING = "merging"                # 正在汇入
    YIELDING = "yielding"              # 正在让行
    BLOCKING = "blocking"              # 阻挡
    INTERACTING = "interacting"        # 交互中
    
    NONE = "none"


class TrafficFlowType(str, Enum):
    """交通流类型"""
    FREE_FLOW = "free_flow"            # 自由流
    LIGHT = "light"                    # 轻度拥堵
    MODERATE = "moderate"              # 中度拥堵
    HEAVY = "heavy"                    # 重度拥堵
    STANDSTILL = "standstill"          # 完全堵死
    CONGESTED = "congested"            # 拥堵
    
    UNKNOWN = "unknown"


class ViolationType(str, Enum):
    """违规行为类型"""
    WRONG_WAY = "wrong_way"            # 逆行
    RUNNING_RED_LIGHT = "running_red"  # 闯红灯
    CROSSING_ILLEGALLY = "crossing_illegally"  # 违规过马路
    LANE_VIOLATION = "lane_violation"  # 违规变道/压实线
    SPEEDING = "speeding"              # 超速
    STOPPING_ILLEGALLY = "stopping_illegally"  # 违规停车
    U_TURN_PROHIBITED = "u_turn_prohibited"  # 禁止掉头
    
    NONE = "none"


# ========== 通用占用网络 ==========

class VoxelState(str, Enum):
    """体素状态"""
    FREE = "free"                      # 空闲（可行驶）
    OCCUPIED = "occupied"              # 占据（不可行驶）
    UNKNOWN = "unknown"                # 未知


class OccupancyFlow(BaseModel):
    """占用流数据"""
    voxel_grid: List[List[List[int]]] = Field(description="体素网格")
    motion_vectors: List[List[List[Dict[str, float]]]] = Field(description="运动向量场")
    confidence: List[List[List[float]]] = Field(description="置信度")
    resolution: float = Field(default=0.1, description="分辨率（米）")
    grid_size: Dict[str, int] = Field(description="网格尺寸 (x, y, z)")
    origin: Dict[str, float] = Field(description="原点坐标")
    timestamp: float = Field(description="时间戳")


# ========== 细粒度对象定义 ==========

class FineGrainedObject(BaseModel):
    """细粒度对象"""
    object_id: str = Field(description="对象ID")
    track_id: str = Field(description="跟踪ID")
    
    # 基础分类
    base_type: str = Field(description="基础类型（vehicle, pedestrian, obstacle等）")
    sub_type: str = Field(description="细分类型")
    
    # 几何信息
    bbox_3d: Dict[str, Any] = Field(description="3D边界框")
    position: Dict[str, float] = Field(description="位置")
    velocity: Dict[str, float] = Field(description="速度")
    acceleration: Dict[str, float] = Field(default_factory=lambda: {"x": 0.0, "y": 0.0, "z": 0.0}, description="加速度")
    
    # 关键属性
    vehicle_lights: Optional[Dict[str, LightState]] = Field(default=None, description="车灯状态")
    door_state: Optional[DoorState] = Field(default=None, description="车门状态")
    pedestrian_pose: Optional[PedestrianPose] = Field(default=None, description="行人姿态")
    occlusion_level: OcclusionLevel = Field(default=OcclusionLevel.NONE, description="遮挡等级")
    
    # 扩展属性
    attributes: Dict[str, Any] = Field(default_factory=dict, description="扩展属性")
    
    # 置信度
    confidence: float = Field(default=1.0, description="置信度")
    
    # 时间戳
    timestamp: float = Field(description="时间戳")


class StaticRoadElement(BaseModel):
    """静态道路要素"""
    element_id: str = Field(description="要素ID")
    element_type: str = Field(description="要素类型")
    
    # 几何信息
    geometry: Dict[str, Any] = Field(description="几何信息")
    position: Dict[str, float] = Field(description="位置")
    
    # 特定属性
    road_marking_type: Optional[RoadMarkingType] = Field(default=None, description="地面标识类型")
    traffic_light_state: Optional[str] = Field(default=None, description="红绿灯状态")
    traffic_light_shape: Optional[TrafficLightShape] = Field(default=None, description="红绿灯形状")
    traffic_light_countdown: Optional[int] = Field(default=None, description="倒计时读秒")
    road_surface_type: Optional[RoadSurfaceType] = Field(default=None, description="路面类型")
    
    # 属性
    attributes: Dict[str, Any] = Field(default_factory=dict, description="扩展属性")
    
    # 置信度
    confidence: float = Field(default=1.0, description="置信度")


class SceneTag(BaseModel):
    """场景标签"""
    tag_id: str = Field(description="标签ID")
    
    # 环境标签
    weather: WeatherType = Field(description="天气")
    lighting: LightingType = Field(description="光照")
    road_type: RoadType = Field(description="道路类型")
    camera_dirt: CameraDirtType = Field(description="传感器脏污")
    
    # 交通流标签
    traffic_flow: TrafficFlowType = Field(default=TrafficFlowType.FREE_FLOW, description="交通流")
    
    # 扩展标签
    tags: Dict[str, Any] = Field(default_factory=dict, description="扩展标签")
    
    # 时间戳
    timestamp: float = Field(description="时间戳")


class BehaviorTag(BaseModel):
    """行为标签"""
    tag_id: str = Field(description="标签ID")
    object_id: str = Field(description="对象ID")
    
    # 行为类型
    ego_relation: EgoRelationType = Field(description="自车关系")
    violation_type: Optional[ViolationType] = Field(default=None, description="违规行为")
    
    # 行为描述
    behavior_description: str = Field(description="行为描述")
    
    # 置信度
    confidence: float = Field(default=1.0, description="置信度")
    
    # 时间戳
    timestamp: float = Field(description="时间戳")


class FineGrainedAnnotation(BaseModel):
    """细粒度标注结果"""
    clip_id: str = Field(description="片段ID")
    frame_id: int = Field(description="帧ID")
    timestamp: float = Field(description="时间戳")
    
    # 细粒度对象
    objects: List[FineGrainedObject] = Field(description="对象列表")
    
    # 静态道路要素
    static_elements: List[StaticRoadElement] = Field(description="静态道路要素列表")
    
    # 场景标签
    scene_tag: SceneTag = Field(description="场景标签")
    
    # 行为标签
    behavior_tags: List[BehaviorTag] = Field(description="行为标签列表")
    
    # 占用流（可选）
    occupancy_flow: Optional[OccupancyFlow] = Field(default=None, description="占用流数据")
    
    # 自车状态
    ego_state: Dict[str, Any] = Field(default_factory=dict, description="自车状态")
    
    # 元数据
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    
    def get_objects_by_sub_type(self, sub_type: str) -> List[FineGrainedObject]:
        """根据细分类型获取对象"""
        return [obj for obj in self.objects if obj.sub_type == sub_type]
    
    def get_objects_with_lights(self, light_state: LightState) -> List[FineGrainedObject]:
        """获取特定车灯状态的对象"""
        return [
            obj for obj in self.objects
            if obj.vehicle_lights and light_state in obj.vehicle_lights.values()
        ]
    
    def get_objects_by_occlusion(self, occlusion_level: OcclusionLevel) -> List[FineGrainedObject]:
        """根据遮挡等级获取对象"""
        return [obj for obj in self.objects if obj.occlusion_level == occlusion_level]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        # 按基础类型统计
        base_type_stats = {}
        for obj in self.objects:
            base_type = obj.base_type
            base_type_stats[base_type] = base_type_stats.get(base_type, 0) + 1
        
        # 按细分类型统计
        sub_type_stats = {}
        for obj in self.objects:
            sub_type = obj.sub_type
            sub_type_stats[sub_type] = sub_type_stats.get(sub_type, 0) + 1
        
        # 按车灯状态统计
        light_stats = {}
        for obj in self.objects:
            if obj.vehicle_lights:
                for light, state in obj.vehicle_lights.items():
                    light_stats[state.value] = light_stats.get(state.value, 0) + 1
        
        # 按遮挡等级统计
        occlusion_stats = {}
        for obj in self.objects:
            occlusion = obj.occlusion_level.value
            occlusion_stats[occlusion] = occlusion_stats.get(occlusion, 0) + 1
        
        # 按行人姿态统计
        pose_stats = {}
        for obj in self.objects:
            if obj.pedestrian_pose:
                pose = obj.pedestrian_pose.value
                pose_stats[pose] = pose_stats.get(pose, 0) + 1
        
        return {
            "total_objects": len(self.objects),
            "total_static_elements": len(self.static_elements),
            "base_type_stats": base_type_stats,
            "sub_type_stats": sub_type_stats,
            "light_stats": light_stats,
            "occlusion_stats": occlusion_stats,
            "pose_stats": pose_stats,
            "weather": self.scene_tag.weather.value,
            "lighting": self.scene_tag.lighting.value,
            "road_type": self.scene_tag.road_type.value,
            "traffic_flow": self.scene_tag.traffic_flow.value
        }