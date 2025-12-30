"""
云端数据挖掘模块 - 多模态索引版本
集成自动化结构化提取、向量化检索、场景图构建和4D重建
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from pydantic import BaseModel, Field
from enum import Enum
import time
import json
from dataclasses import dataclass

# 导入多模态索引模块
from .auto_labeling import AutoLabelingEngine, VideoAnnotation
from .vector_indexing import VectorIndexingEngine, VideoEmbedding, VectorSearchResult
from .scene_graph import SceneGraphBuilder, SceneGraph


class MiningStatus(str, Enum):
    """挖掘状态"""
    PENDING = "pending"     # 待处理
    RUNNING = "running"     # 运行中
    COMPLETED = "completed" # 已完成
    FAILED = "failed"       # 失败


class MiningStrategy(str, Enum):
    """挖掘策略"""
    TRIGGER_BASED = "trigger_based"        # 基于Trigger
    VECTOR_SEARCH = "vector_search"        # 向量检索
    RULE_FILTER = "rule_filter"            # 规则过滤
    EVENT_SEQUENCE = "event_sequence"      # 事件序列
    HYBRID = "hybrid"                      # 混合策略（推荐）


class MiningConfig(BaseModel):
    """挖掘配置"""
    mining_id: str = Field(description="挖掘任务ID")
    name: str = Field(description="挖掘任务名称")
    description: str = Field(default="", description="描述")
    strategy: MiningStrategy = Field(default=MiningStrategy.HYBRID, description="挖掘策略")
    data_source: str = Field(description="数据源路径")
    output_path: str = Field(description="输出路径")
    max_results: int = Field(default=10000, description="最大结果数")
    
    # 多模态处理配置
    enable_auto_labeling: bool = Field(default=True, description="是否启用自动标注")
    enable_vector_indexing: bool = Field(default=True, description="是否启用向量化索引")
    enable_scene_graph: bool = Field(default=False, description="是否启用场景图")
    enable_4d_reconstruction: bool = Field(default=False, description="是否启用4D重建")
    
    # 查询配置
    text_query: Optional[str] = Field(default=None, description="文本查询")
    image_query: Optional[str] = Field(default=None, description="图像查询路径")
    trigger_rule: Optional[Dict[str, Any]] = Field(default=None, description="Trigger规则")
    filter_rules: Optional[Dict[str, Any]] = Field(default=None, description="过滤规则")
    event_sequence: Optional[List[str]] = Field(default=None, description="事件序列")
    
    # 向量检索配置
    similarity_threshold: float = Field(default=0.7, description="相似度阈值")
    top_k: int = Field(default=100, description="向量检索Top-K")
    
    enabled: bool = Field(default=True, description="是否启用")


class MiningResult(BaseModel):
    """挖掘结果"""
    mining_id: str
    clip_id: str
    timestamp: float
    similarity: Optional[float] = None
    caption: Optional[str] = None
    annotation: Optional[VideoAnnotation] = None
    embedding: Optional[VideoEmbedding] = None
    scene_graph: Optional[SceneGraph] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MiningStatistics(BaseModel):
    """挖掘统计"""
    mining_id: str
    total_processed: int = 0
    total_matched: int = 0
    total_output: int = 0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration: float = 0.0
    auto_labeling_time: float = 0.0
    vector_indexing_time: float = 0.0
    scene_graph_time: float = 0.0
    reconstruction_time: float = 0.0


class DataMiner:
    """
    数据挖掘器 - 多模态索引版本
    集成自动化结构化提取、向量化检索、场景图构建和4D重建
    """

    def __init__(self):
        from onboard.triggers.trigger_manager import TriggerManager
        from .auto_labeling import AutoLabelingConfig
        from .vector_indexing import VectorIndexingConfig
        from .scene_graph import SceneGraphBuilderConfig
        
        self.trigger_manager = TriggerManager()
        
        # 初始化多模态处理引擎
        self.auto_labeling_engine = AutoLabelingEngine(AutoLabelingConfig())
        self.vector_indexing_engine = VectorIndexingEngine(VectorIndexingConfig())
        self.scene_graph_builder = SceneGraphBuilder(SceneGraphBuilderConfig())
        
        self.mining_configs: Dict[str, MiningConfig] = {}
        self.mining_status: Dict[str, MiningStatus] = {}
        self.mining_statistics: Dict[str, MiningStatistics] = {}
        self.mining_results: Dict[str, List[MiningResult]] = {}
        
        # 存储处理后的数据
        self.annotations: Dict[str, VideoAnnotation] = {}
        self.embeddings: Dict[str, VideoEmbedding] = {}
        self.scene_graphs: Dict[str, SceneGraph] = {}

    def create_mining_task(self, config: MiningConfig) -> bool:
        """
        创建挖掘任务

        Args:
            config: 挖掘配置

        Returns:
            bool: 创建是否成功
        """
        if config.mining_id in self.mining_configs:
            return False

        self.mining_configs[config.mining_id] = config
        self.mining_status[config.mining_id] = MiningStatus.PENDING
        self.mining_statistics[config.mining_id] = MiningStatistics(mining_id=config.mining_id)
        self.mining_results[config.mining_id] = []

        return True

    def process_video_clip(
        self,
        video_path: str,
        clip_id: str,
        start_time: float = 0.0,
        end_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        处理视频片段（多模态索引流水线）

        Args:
            video_path: 视频路径
            clip_id: 片段ID
            start_time: 起始时间
            end_time: 结束时间

        Returns:
            Dict: 处理结果
        """
        result = {
            "clip_id": clip_id,
            "video_path": video_path,
            "start_time": start_time,
            "end_time": end_time,
            "annotation": None,
            "embedding": None,
            "scene_graph": None,
            "processing_time": 0.0
        }

        start = time.time()

        # 1. 自动化结构化提取
        annotation = self.auto_labeling_engine.process_video(
            video_path, clip_id, start_time, end_time
        )
        result["annotation"] = annotation
        self.annotations[clip_id] = annotation

        # 2. 向量化索引
        embedding = self.vector_indexing_engine.process_video(
            video_path, clip_id, start_time, end_time
        )
        result["embedding"] = embedding
        self.embeddings[clip_id] = embedding

        # 3. 场景图构建
        if annotation.frames:
            scene_graph = self.scene_graph_builder.build_from_annotation(
                annotation, frame_id=0
            )
            result["scene_graph"] = scene_graph
            self.scene_graphs[clip_id] = scene_graph

        result["processing_time"] = time.time() - start

        return result

    def run_mining(self, mining_id: str, video_clips: List[Dict[str, Any]]) -> bool:
        """
        运行挖掘任务（多模态索引）

        Args:
            mining_id: 挖掘任务ID
            video_clips: 视频片段列表，每个元素包含：
                {
                    "video_path": "路径",
                    "clip_id": "ID",
                    "start_time": 0.0,
                    "end_time": 10.0
                }

        Returns:
            bool: 运行是否成功
        """
        if mining_id not in self.mining_configs:
            return False

        config = self.mining_configs[mining_id]

        if not config.enabled:
            return False

        # 更新状态
        self.mining_status[mining_id] = MiningStatus.RUNNING
        stats = self.mining_statistics[mining_id]
        stats.start_time = time.time()

        try:
            # 处理所有视频片段
            for clip_info in video_clips:
                # 检查是否达到最大结果数
                if len(self.mining_results[mining_id]) >= config.max_results:
                    break

                video_path = clip_info["video_path"]
                clip_id = clip_info["clip_id"]
                start_time = clip_info.get("start_time", 0.0)
                end_time = clip_info.get("end_time")

                # 处理视频片段
                process_result = self.process_video_clip(
                    video_path, clip_id, start_time, end_time
                )

                # 更新统计
                stats.auto_labeling_time += 0.5  # 模拟
                stats.vector_indexing_time += 0.3  # 模拟
                stats.scene_graph_time += 0.1  # 模拟

                # 根据策略筛选
                if self._match_criteria(process_result, config):
                    mining_result = MiningResult(
                        mining_id=mining_id,
                        clip_id=clip_id,
                        timestamp=time.time(),
                        similarity=process_result["embedding"].global_embedding if process_result["embedding"] else None,
                        caption=process_result["embedding"].caption if process_result["embedding"] else None,
                        annotation=process_result["annotation"],
                        embedding=process_result["embedding"],
                        scene_graph=process_result.get("scene_graph"),
                        metadata={
                            "video_path": video_path,
                            "processing_time": process_result["processing_time"]
                        }
                    )
                    self.mining_results[mining_id].append(mining_result)
                    stats.total_matched += 1

                stats.total_processed += 1

            # 保存结果
            self._save_results(mining_id, config)

            # 更新状态
            self.mining_status[mining_id] = MiningStatus.COMPLETED
            stats.end_time = time.time()
            stats.duration = stats.end_time - stats.start_time
            stats.total_output = len(self.mining_results[mining_id])

            return True

        except Exception as e:
            print(f"Mining error: {e}")
            self.mining_status[mining_id] = MiningStatus.FAILED
            return False

    def _match_criteria(self, process_result: Dict[str, Any], config: MiningConfig) -> bool:
        """
        根据配置匹配筛选条件

        Args:
            process_result: 处理结果
            config: 挖掘配置

        Returns:
            bool: 是否匹配
        """
        annotation = process_result["annotation"]
        embedding = process_result["embedding"]

        # 文本查询匹配
        if config.text_query and embedding:
            query_results = self.vector_indexing_engine.search_by_text(
                config.text_query,
                top_k=1,
                similarity_threshold=config.similarity_threshold
            )
            if not query_results or query_results[0].clip_id != embedding.clip_id:
                return False

        # 图像查询匹配
        if config.image_query and embedding:
            query_results = self.vector_indexing_engine.search_by_image(
                config.image_query,
                top_k=1,
                similarity_threshold=config.similarity_threshold
            )
            if not query_results or query_results[0].clip_id != embedding.clip_id:
                return False

        # Trigger规则匹配
        if config.trigger_rule and annotation:
            # 使用标注数据进行Trigger匹配
            matched_clips = self.auto_labeling_engine.query_by_rules(
                [annotation],
                config.trigger_rule
            )
            if annotation.clip_id not in matched_clips:
                return False

        # 过滤规则匹配
        if config.filter_rules and annotation:
            # 应用过滤规则
            # TODO: 实现过滤规则匹配逻辑
            pass

        return True

    def _save_results(self, mining_id: str, config: MiningConfig):
        """
        保存挖掘结果

        Args:
            mining_id: 挖掘任务ID
            config: 挖掘配置
        """
        results = self.mining_results[mining_id]

        # 转换为字典
        results_dict = []
        for result in results:
            result_dict = {
                "mining_id": result.mining_id,
                "clip_id": result.clip_id,
                "timestamp": result.timestamp,
                "similarity": result.similarity,
                "caption": result.caption,
                "metadata": result.metadata
            }
            
            # 添加标注统计
            if result.annotation:
                result_dict["annotation_stats"] = result.annotation.get_statistics()
            
            results_dict.append(result_dict)

        # 保存到文件
        try:
            with open(config.output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "mining_id": mining_id,
                    "config": config.dict(),
                    "results": results_dict,
                    "statistics": self.mining_statistics[mining_id].dict()
                }, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Failed to save results: {e}")

    def get_mining_results(self, mining_id: str) -> List[MiningResult]:
        """获取挖掘结果"""
        return self.mining_results.get(mining_id, [])

    def get_mining_statistics(self, mining_id: str) -> Optional[MiningStatistics]:
        """获取挖掘统计"""
        return self.mining_statistics.get(mining_id)

    def get_mining_status(self, mining_id: str) -> Optional[MiningStatus]:
        """获取挖掘状态"""
        return self.mining_status.get(mining_id)

    def delete_mining_task(self, mining_id: str) -> bool:
        """删除挖掘任务"""
        if mining_id in self.mining_configs:
            del self.mining_configs[mining_id]
        if mining_id in self.mining_status:
            del self.mining_status[mining_id]
        if mining_id in self.mining_statistics:
            del self.mining_statistics[mining_id]
        if mining_id in self.mining_results:
            del self.mining_results[mining_id]

        return True

    def get_all_mining_tasks(self) -> Dict[str, MiningConfig]:
        """获取所有挖掘任务"""
        return self.mining_configs.copy()

    def get_summary(self) -> Dict[str, Any]:
        """获取摘要"""
        total_tasks = len(self.mining_configs)
        running_tasks = sum(
            1 for status in self.mining_status.values()
            if status == MiningStatus.RUNNING
        )
        completed_tasks = sum(
            1 for status in self.mining_status.values()
            if status == MiningStatus.COMPLETED
        )
        failed_tasks = sum(
            1 for status in self.mining_status.values()
            if status == MiningStatus.FAILED
        )

        return {
            "total_tasks": total_tasks,
            "running_tasks": running_tasks,
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "pending_tasks": total_tasks - running_tasks - completed_tasks - failed_tasks
        }


# 全局数据挖掘器实例
_global_data_miner = None


def get_global_data_miner() -> DataMiner:
    """获取全局数据挖掘器实例"""
    global _global_data_miner
    if _global_data_miner is None:
        _global_data_miner = DataMiner()
    return _global_data_miner