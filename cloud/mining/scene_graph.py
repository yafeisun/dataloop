"""
场景图构建模块 (Scene Graph)
构建场景的拓扑结构，描述对象之间的关系
"""

from typing import Dict, List, Optional, Any, Set, Tuple
from pydantic import BaseModel, Field
from enum import Enum
import time
import json
import networkx as nx


class RelationType(str, Enum):
    """关系类型"""
    OCCLUDED_BY = "occluded_by"          # 被遮挡
    NEAR = "near"                        # 靠近
    CUT_IN = "cut_in"                    # 切入
    CUT_OUT = "cut_out"                  # 切出
    FOLLOWING = "following"              # 跟随
    OVERTAKING = "overtaking"            # 超车
    CROSSING = "crossing"                # 横穿
    WAITING_FOR = "waiting_for"          # 等待
    YIELDING_TO = "yielding_to"          # 让行
    INTERACTING_WITH = "interacting_with"  # 交互
    SAME_LANE = "same_lane"              # 同车道
    ADJACENT_LANE = "adjacent_lane"      # 相邻车道
    OPPOSITE_LANE = "opposite_lane"      # 对向车道
    UNKNOWN = "unknown"


class SceneGraphNode(BaseModel):
    """场景图节点"""
    node_id: str = Field(description="节点ID")
    object_id: str = Field(description="对象ID")
    object_type: str = Field(description="对象类型")
    position: Dict[str, float] = Field(description="位置")
    attributes: Dict[str, Any] = Field(default_factory=dict, description="属性")
    timestamp: float = Field(description="时间戳")


class SceneGraphEdge(BaseModel):
    """场景图边"""
    edge_id: str = Field(description="边ID")
    source_id: str = Field(description="源节点ID")
    target_id: str = Field(description="目标节点ID")
    relation: RelationType = Field(description="关系类型")
    confidence: float = Field(default=1.0, description="置信度")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


class SceneGraph(BaseModel):
    """场景图"""
    graph_id: str = Field(description="图ID")
    clip_id: str = Field(description="片段ID")
    frame_id: int = Field(description="帧ID")
    timestamp: float = Field(description="时间戳")
    nodes: List[SceneGraphNode] = Field(description="节点列表")
    edges: List[SceneGraphEdge] = Field(description="边列表")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")

    def get_node(self, node_id: str) -> Optional[SceneGraphNode]:
        """获取节点"""
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None

    def get_edges_by_node(self, node_id: str) -> List[SceneGraphEdge]:
        """获取节点的所有边"""
        return [
            edge for edge in self.edges
            if edge.source_id == node_id or edge.target_id == node_id
        ]

    def get_neighbors(self, node_id: str) -> List[SceneGraphNode]:
        """获取邻居节点"""
        neighbors = []

        for edge in self.edges:
            if edge.source_id == node_id:
                neighbor = self.get_node(edge.target_id)
                if neighbor:
                    neighbors.append(neighbor)
            elif edge.target_id == node_id:
                neighbor = self.get_node(edge.source_id)
                if neighbor:
                    neighbors.append(neighbor)

        return neighbors

    def find_subgraph_by_pattern(
        self,
        pattern: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        根据模式查找子图

        Args:
            pattern: 查询模式，例如：
                {
                    "nodes": [{"object_type": "pedestrian"}],
                    "edges": [{"relation": "crossing"}]
                }

        Returns:
            List[Dict]: 匹配的子图
        """
        # TODO: 实现子图匹配算法
        return []

    def to_networkx(self) -> nx.DiGraph:
        """转换为NetworkX图"""
        G = nx.DiGraph()

        # 添加节点
        for node in self.nodes:
            G.add_node(
                node.node_id,
                object_id=node.object_id,
                object_type=node.object_type,
                position=node.position,
                attributes=node.attributes
            )

        # 添加边
        for edge in self.edges:
            G.add_edge(
                edge.source_id,
                edge.target_id,
                relation=edge.relation,
                confidence=edge.confidence,
                edge_id=edge.edge_id
            )

        return G

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        # 按节点类型统计
        node_type_stats = {}
        for node in self.nodes:
            obj_type = node.object_type
            node_type_stats[obj_type] = node_type_stats.get(obj_type, 0) + 1

        # 按关系类型统计
        relation_stats = {}
        for edge in self.edges:
            relation = edge.relation.value
            relation_stats[relation] = relation_stats.get(relation, 0) + 1

        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "node_type_stats": node_type_stats,
            "relation_stats": relation_stats
        }


class SceneGraphBuilderConfig(BaseModel):
    """场景图构建配置"""
    max_distance: float = Field(default=50.0, description="最大距离（米）")
    occlusion_threshold: float = Field(default=0.5, description="遮挡阈值")
    relation_confidence_threshold: float = Field(default=0.7, description="关系置信度阈值")
    enable_spatial_relations: bool = Field(default=True, description="是否启用空间关系")
    enable_temporal_relations: bool = Field(default=True, description="是否启用时序关系")


class SceneGraphBuilder:
    """
    场景图构建器
    从标注数据构建场景图
    """

    def __init__(self, config: SceneGraphBuilderConfig):
        self.config = config

    def build_from_annotation(
        self,
        annotation: Any,
        frame_id: int
    ) -> SceneGraph:
        """
        从标注数据构建场景图

        Args:
            annotation: 标注数据
            frame_id: 帧ID

        Returns:
            SceneGraph: 场景图
        """
        # 获取帧标注
        frame_annotation = None
        for frame in annotation.frames:
            if frame.frame_id == frame_id:
                frame_annotation = frame
                break

        if not frame_annotation:
            return SceneGraph(
                graph_id=f"graph_{frame_id}",
                clip_id=annotation.clip_id,
                frame_id=frame_id,
                timestamp=frame_id / annotation.fps,
                nodes=[],
                edges=[]
            )

        # 构建节点
        nodes = self._build_nodes(frame_annotation)

        # 构建边
        edges = self._build_edges(nodes, frame_annotation)

        graph = SceneGraph(
            graph_id=f"graph_{frame_id}",
            clip_id=annotation.clip_id,
            frame_id=frame_id,
            timestamp=frame_annotation.timestamp,
            nodes=nodes,
            edges=edges,
            metadata={
                "ego_vehicle": frame_annotation.ego_vehicle
            }
        )

        return graph

    def _build_nodes(self, frame_annotation: Any) -> List[SceneGraphNode]:
        """构建节点"""
        nodes = []

        # 添加对象节点
        for obj in frame_annotation.objects:
            node = SceneGraphNode(
                node_id=f"node_{obj.track_id}",
                object_id=obj.object_id,
                object_type=obj.object_type.value,
                position=obj.bboxes[0].center if obj.bboxes else {"x": 0.0, "y": 0.0, "z": 0.0},
                attributes=obj.attributes,
                timestamp=frame_annotation.timestamp
            )
            nodes.append(node)

        # 添加自车节点
        ego_vehicle = frame_annotation.ego_vehicle
        if ego_vehicle:
            ego_node = SceneGraphNode(
                node_id="node_ego",
                object_id="ego_vehicle",
                object_type="ego_vehicle",
                position=ego_vehicle.get("position", {"x": 0.0, "y": 0.0, "z": 0.0}),
                attributes=ego_vehicle,
                timestamp=frame_annotation.timestamp
            )
            nodes.append(ego_node)

        return nodes

    def _build_edges(
        self,
        nodes: List[SceneGraphNode],
        frame_annotation: Any
    ) -> List[SceneGraphEdge]:
        """构建边"""
        edges = []

        # 构建对象之间的关系
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes):
                if i >= j:
                    continue

                # 计算距离
                distance = self._calculate_distance(node1.position, node2.position)

                # 距离太远，不考虑关系
                if distance > self.config.max_distance:
                    continue

                # 分析关系
                relations = self._analyze_relations(node1, node2, distance, frame_annotation)

                # 添加边
                for relation, confidence in relations:
                    if confidence >= self.config.relation_confidence_threshold:
                        edge = SceneGraphEdge(
                            edge_id=f"edge_{node1.node_id}_{node2.node_id}_{relation.value}",
                            source_id=node1.node_id,
                            target_id=node2.node_id,
                            relation=relation,
                            confidence=confidence,
                            metadata={"distance": distance}
                        )
                        edges.append(edge)

        return edges

    def _calculate_distance(
        self,
        pos1: Dict[str, float],
        pos2: Dict[str, float]
    ) -> float:
        """计算两点距离"""
        x1, y1, z1 = pos1.get("x", 0.0), pos1.get("y", 0.0), pos1.get("z", 0.0)
        x2, y2, z2 = pos2.get("x", 0.0), pos2.get("y", 0.0), pos2.get("z", 0.0)

        return ((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2) ** 0.5

    def _analyze_relations(
        self,
        node1: SceneGraphNode,
        node2: SceneGraphNode,
        distance: float,
        frame_annotation: Any
    ) -> List[Tuple[RelationType, float]]:
        """
        分析两个节点之间的关系

        Args:
            node1: 节点1
            node2: 节点2
            distance: 距离
            frame_annotation: 帧标注

        Returns:
            List[Tuple]: 关系列表（关系类型，置信度）
        """
        relations = []

        # 靠近关系
        if distance < 10.0:
            relations.append((RelationType.NEAR, 0.9))

        # 车道关系
        if self.config.enable_spatial_relations:
            lane_relation = self._analyze_lane_relation(node1, node2, frame_annotation)
            if lane_relation:
                relations.append(lane_relation)

        # 切入/切出关系
        if self.config.enable_temporal_relations:
            cut_relation = self._analyze_cut_relation(node1, node2)
            if cut_relation:
                relations.append(cut_relation)

        # 遮挡关系（简化判断）
        occlusion_relation = self._analyze_occlusion(node1, node2)
        if occlusion_relation:
            relations.append(occlusion_relation)

        return relations

    def _analyze_lane_relation(
        self,
        node1: SceneGraphNode,
        node2: SceneGraphNode,
        frame_annotation: Any
    ) -> Optional[Tuple[RelationType, float]]:
        """分析车道关系"""
        # 简化判断：根据y坐标判断车道
        y1, y2 = node1.position.get("y", 0.0), node2.position.get("y", 0.0)
        lane_width = 3.5  # 标准车道宽度

        lane_diff = abs(y1 - y2)

        if lane_diff < lane_width / 2:
            return (RelationType.SAME_LANE, 0.8)
        elif lane_diff < lane_width * 1.5:
            return (RelationType.ADJACENT_LANE, 0.8)
        else:
            return (RelationType.OPPOSITE_LANE, 0.7)

    def _analyze_cut_relation(
        self,
        node1: SceneGraphNode,
        node2: SceneGraphNode
    ) -> Optional[Tuple[RelationType, float]]:
        """分析切入/切出关系"""
        # 简化判断：根据速度方向判断
        # TODO: 实际应该分析轨迹
        return None

    def _analyze_occlusion(
        self,
        node1: SceneGraphNode,
        node2: SceneGraphNode
    ) -> Optional[Tuple[RelationType, float]]:
        """分析遮挡关系"""
        # 简化判断：根据x坐标和距离判断
        x1, x2 = node1.position.get("x", 0.0), node2.position.get("x", 0.0)
        y1, y2 = node1.position.get("y", 0.0), node2.position.get("y", 0.0)

        # 如果node2在node1前面且距离很近，可能被遮挡
        if x2 > x1 and abs(y2 - y1) < 2.0:
            distance = self._calculate_distance(node1.position, node2.position)
            if distance < 5.0:
                return (RelationType.OCCLUDED_BY, 0.7)

        return None

    def query_by_relation(
        self,
        graphs: List[SceneGraph],
        relation_type: RelationType,
        object_types: Optional[List[str]] = None
    ) -> List[str]:
        """
        根据关系查询

        Args:
            graphs: 场景图列表
            relation_type: 关系类型
            object_types: 对象类型过滤

        Returns:
            List[str]: 匹配的clip_id列表
        """
        matched_clips = []

        for graph in graphs:
            match = False

            for edge in graph.edges:
                if edge.relation != relation_type:
                    continue

                # 检查对象类型
                if object_types:
                    source_node = graph.get_node(edge.source_id)
                    target_node = graph.get_node(edge.target_id)

                    if not source_node or not target_node:
                        continue

                    source_type = source_node.object_type
                    target_type = target_node.object_type

                    if source_type not in object_types and target_type not in object_types:
                        continue

                match = True
                break

            if match:
                matched_clips.append(graph.clip_id)

        return matched_clips

    def save_graph(self, graph: SceneGraph, output_path: str) -> bool:
        """保存场景图"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(graph.dict(), f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Failed to save graph: {e}")
            return False


# 便捷函数
def create_scene_graph_builder(
    max_distance: float = 50.0
) -> SceneGraphBuilder:
    """创建场景图构建器"""
    config = SceneGraphBuilderConfig(max_distance=max_distance)
    return SceneGraphBuilder(config)