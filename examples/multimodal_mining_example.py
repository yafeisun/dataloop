"""
多模态数据挖掘示例
演示如何使用多模态索引系统进行数据挖掘
"""

import sys
import os
import time
from typing import Dict, Any, List

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cloud.mining.data_miner import (
    DataMiner,
    MiningConfig,
    MiningStrategy,
    MiningResult
)
from cloud.mining.auto_labeling import AutoLabelingEngine, VideoAnnotation
from cloud.mining.vector_indexing import VectorIndexingEngine, VideoEmbedding
from cloud.mining.scene_graph import SceneGraphBuilder, SceneGraph


def example_multimodal_pipeline():
    """示例：完整的多模态处理流水线"""
    print("\n=== 多模态处理流水线示例 ===")

    # 创建数据挖掘器
    miner = DataMiner()

    # 模拟视频片段
    video_clips = [
        {
            "video_path": "/data/videos/clip_001.mp4",
            "clip_id": "clip_001",
            "start_time": 0.0,
            "end_time": 10.0
        },
        {
            "video_path": "/data/videos/clip_002.mp4",
            "clip_id": "clip_002",
            "start_time": 0.0,
            "end_time": 10.0
        },
        {
            "video_path": "/data/videos/clip_003.mp4",
            "clip_id": "clip_003",
            "start_time": 0.0,
            "end_time": 10.0
        }
    ]

    print(f"处理 {len(video_clips)} 个视频片段...")

    # 处理每个视频片段
    for clip_info in video_clips:
        print(f"\n处理片段: {clip_info['clip_id']}")
        
        result = miner.process_video_clip(
            clip_info["video_path"],
            clip_info["clip_id"],
            clip_info["start_time"],
            clip_info["end_time"]
        )

        print(f"  自动标注: {len(result['annotation'].frames)} 帧")
        print(f"  关键帧: {len(result['embedding'].key_frames)} 个")
        print(f"  场景图: {len(result['scene_graph'].nodes)} 个节点, {len(result['scene_graph'].edges)} 条边")
        print(f"  处理时间: {result['processing_time']:.2f}s")


def example_text_search():
    """示例：基于文本的自然语言搜索"""
    print("\n=== 文本搜索示例 ===")

    # 创建数据挖掘器
    miner = DataMiner()

    # 预处理一些视频片段
    video_clips = [
        {
            "video_path": "/data/videos/clip_001.mp4",
            "clip_id": "clip_001",
            "start_time": 0.0,
            "end_time": 10.0
        },
        {
            "video_path": "/data/videos/clip_002.mp4",
            "clip_id": "clip_002",
            "start_time": 0.0,
            "end_time": 10.0
        }
    ]

    # 处理视频片段
    for clip_info in video_clips:
        miner.process_video_clip(
            clip_info["video_path"],
            clip_info["clip_id"],
            clip_info["start_time"],
            clip_info["end_time"]
        )

    # 执行文本搜索
    query = "雨天、前方有侧翻车辆、且有警察指挥交通"
    print(f"搜索查询: {query}")

    results = miner.vector_indexing_engine.search_by_text(
        query,
        top_k=5,
        similarity_threshold=0.7
    )

    print(f"找到 {len(results)} 个匹配结果")

    for i, result in enumerate(results):
        print(f"  {i + 1}. {result.clip_id}: 相似度={result.similarity:.2f}")
        if result.caption:
            print(f"     描述: {result.caption}")


def example_hybrid_search():
    """示例：混合搜索（文本 + 规则）"""
    print("\n=== 混合搜索示例 ===")

    # 创建数据挖掘器
    miner = DataMiner()

    # 创建挖掘任务
    config = MiningConfig(
        mining_id="task_hybrid_001",
        name="混合搜索任务",
        description="使用文本查询和规则过滤进行混合搜索",
        strategy=MiningStrategy.HYBRID,
        data_source="/data/videos",
        output_path="/output/hybrid_search_results.json",
        max_results=100,
        enable_auto_labeling=True,
        enable_vector_indexing=True,
        enable_scene_graph=False,
        text_query="车辆突然变道",
        filter_rules={
            "ego_speed_min": 40.0,  # 自车速度 > 40 km/h
            "weather": ["sunny", "cloudy"]  # 晴天或多云
        },
        similarity_threshold=0.7,
        top_k=50
    )

    miner.create_mining_task(config)

    # 模拟视频片段
    video_clips = [
        {
            "video_path": "/data/videos/clip_001.mp4",
            "clip_id": "clip_001",
            "start_time": 0.0,
            "end_time": 10.0
        },
        {
            "video_path": "/data/videos/clip_002.mp4",
            "clip_id": "clip_002",
            "start_time": 0.0,
            "end_time": 10.0
        }
    ]

    # 运行挖掘任务
    print("运行混合搜索任务...")
    success = miner.run_mining(config.mining_id, video_clips)

    if success:
        results = miner.get_mining_results(config.mining_id)
        stats = miner.get_mining_statistics(config.mining_id)

        print(f"\n挖掘完成!")
        print(f"  处理片段数: {stats.total_processed}")
        print(f"  匹配片段数: {stats.total_matched}")
        print(f"  输出结果数: {stats.total_output}")
        print(f"  耗时: {stats.duration:.2f}s")

        print(f"\n匹配结果:")
        for i, result in enumerate(results[:5]):
            print(f"  {i + 1}. {result.clip_id}")
            if result.similarity:
                print(f"     相似度: {result.similarity:.2f}")
            if result.caption:
                print(f"     描述: {result.caption}")


def example_rule_based_mining():
    """示例：基于规则的挖掘（几何精度）"""
    print("\n=== 基于规则的挖掘示例 ===")

    # 创建数据挖掘器
    miner = DataMiner()

    # 创建挖掘任务
    config = MiningConfig(
        mining_id="task_rule_001",
        name="基于规则的挖掘",
        description="使用几何规则筛选急刹车场景",
        strategy=MiningStrategy.TRIGGER_BASED,
        data_source="/data/videos",
        output_path="/output/rule_mining_results.json",
        max_results=100,
        enable_auto_labeling=True,
        enable_vector_indexing=False,
        trigger_rule={
            "ego_speed_min": 60.0,  # 自车速度 > 60 km/h
            "time_to_collision_max": 2.0  # 碰撞时间 < 2s
        }
    )

    miner.create_mining_task(config)

    # 模拟视频片段
    video_clips = [
        {
            "video_path": "/data/videos/clip_001.mp4",
            "clip_id": "clip_001",
            "start_time": 0.0,
            "end_time": 10.0
        },
        {
            "video_path": "/data/videos/clip_002.mp4",
            "clip_id": "clip_002",
            "start_time": 0.0,
            "end_time": 10.0
        }
    ]

    # 运行挖掘任务
    print("运行基于规则的挖掘任务...")
    success = miner.run_mining(config.mining_id, video_clips)

    if success:
        results = miner.get_mining_results(config.mining_id)
        stats = miner.get_mining_statistics(config.mining_id)

        print(f"\n挖掘完成!")
        print(f"  处理片段数: {stats.total_processed}")
        print(f"  匹配片段数: {stats.total_matched}")

        print(f"\n匹配结果:")
        for i, result in enumerate(results):
            print(f"  {i + 1}. {result.clip_id}")
            if result.annotation:
                annotation_stats = result.annotation.get_statistics()
                print(f"     对象数: {annotation_stats['total_objects']}")


def example_scene_graph_query():
    """示例：场景图查询"""
    print("\n=== 场景图查询示例 ===")

    # 创建数据挖掘器
    miner = DataMiner()

    # 处理视频片段
    video_clips = [
        {
            "video_path": "/data/videos/clip_001.mp4",
            "clip_id": "clip_001",
            "start_time": 0.0,
            "end_time": 10.0
        }
    ]

    for clip_info in video_clips:
        miner.process_video_clip(
            clip_info["video_path"],
            clip_info["clip_id"],
            clip_info["start_time"],
            clip_info["end_time"]
        )

    # 查询场景图
    from cloud.mining.scene_graph import RelationType

    matched_clips = miner.scene_graph_builder.query_by_relation(
        miner.scene_graphs.values(),
        relation_type=RelationType.CUT_IN,
        object_types=["vehicle", "pedestrian"]
    )

    print(f"找到 {len(matched_clips)} 个包含切入关系的片段")

    for clip_id in matched_clips:
        print(f"  - {clip_id}")


def show_mining_statistics():
    """显示挖掘统计"""
    print("\n=== 挖掘统计 ===")

    # 创建数据挖掘器
    miner = DataMiner()

    summary = miner.get_summary()

    print(f"总任务数: {summary['total_tasks']}")
    print(f"运行中: {summary['running_tasks']}")
    print(f"已完成: {summary['completed_tasks']}")
    print(f"失败: {summary['failed_tasks']}")
    print(f"待处理: {summary['pending_tasks']}")


def main():
    """主函数"""
    print("多模态数据挖掘示例")
    print("=" * 50)

    # 运行示例
    example_multimodal_pipeline()
    example_text_search()
    example_hybrid_search()
    example_rule_based_mining()
    example_scene_graph_query()
    show_mining_statistics()

    print("\n示例运行完成！")
    print("\n多模态索引系统优势：")
    print("1. 基础层：自动化结构化提取 - 提供几何精度的3D标注")
    print("2. 进阶层：向量化检索 - 支持自然语言语义搜索")
    print("3. 高级层：场景图构建 - 描述对象之间的复杂关系")
    print("4. 终极层：4D重建 - 支持场景编辑和数据增强")
    print("\n推荐使用混合策略（HYBRID）以获得最佳效果！")


if __name__ == "__main__":
    main()