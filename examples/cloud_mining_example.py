"""
云端数据挖掘示例
演示如何使用数据挖掘模块从海量数据中筛选目标场景
"""

import sys
import os
import time
import json
from typing import Dict, Any, List

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cloud.mining.data_miner import (
    DataMiner,
    MiningConfig,
    MiningTask,
    MiningResult,
    MiningStrategy
)
from common.events.event_types import (
    Event,
    EventSequence,
    EventType,
    EventBuilder
)
from common.models.trigger_base import TriggerType, TriggerPriority


def create_sample_events() -> List[Event]:
    """创建示例事件序列"""
    events = []

    base_time = time.time()

    # 感知事件
    events.append(EventBuilder.create_perception_event(
        object_id="vehicle_001",
        object_type="vehicle",
        position={"x": 10.0, "y": 0.0, "z": 0.0},
        confidence=0.95,
        timestamp=base_time
    ))

    # 预测事件
    events.append(EventBuilder.create_prediction_event(
        object_id="vehicle_001",
        predicted_trajectory=[
            {"x": 10.0, "y": 0.0, "z": 0.0},
            {"x": 15.0, "y": 0.5, "z": 0.0},
            {"x": 20.0, "y": 1.0, "z": 0.0}
        ],
        confidence=0.85,
        timestamp=base_time + 0.1
    ))

    # 规划事件
    events.append(EventBuilder.create_planning_event(
        plan_id="plan_001",
        planned_trajectory=[
            {"x": 0.0, "y": 0.0, "z": 0.0},
            {"x": 5.0, "y": 0.0, "z": 0.0},
            {"x": 10.0, "y": 0.0, "z": 0.0}
        ],
        planned_speed=10.0,
        timestamp=base_time + 0.2
    ))

    # 控制事件
    events.append(EventBuilder.create_control_event(
        control_id="control_001",
        throttle=0.3,
        brake=0.0,
        steering=0.0,
        timestamp=base_time + 0.3
    ))

    # 车辆事件
    events.append(EventBuilder.create_vehicle_event(
        vehicle_id="ego_vehicle",
        position={"x": 0.0, "y": 0.0, "z": 0.0},
        velocity={"x": 10.0, "y": 0.0, "z": 0.0},
        timestamp=base_time + 0.4
    ))

    return events


def create_sample_data() -> List[Dict[str, Any]]:
    """创建示例数据"""
    data = []

    for i in range(100):
        item = {
            "data_id": f"data_{i:04d}",
            "timestamp": time.time() - (100 - i) * 60,
            "vehicle_id": f"vehicle_{i % 10}",
            "location": {
                "x": i * 10.0,
                "y": i * 5.0,
                "z": 0.0
            },
            "weather": "sunny",
            "time_of_day": "day",
            "events": create_sample_events() if i % 10 == 0 else []
        }
        data.append(item)

    return data


def example_trigger_based_mining():
    """示例：基于Trigger的数据挖掘"""
    print("\n=== 基于Trigger的数据挖掘 ===")

    # 创建数据挖掘器
    config = MiningConfig(
        max_tasks=50,
        default_max_results=1000
    )
    miner = DataMiner(config)

    # 创建Trigger规则
    trigger_rule = {
        "trigger_id": "mining_trigger_001",
        "name": "急刹车挖掘",
        "trigger_type": TriggerType.RULE,
        "priority": TriggerPriority.HIGH,
        "rules": [
            {
                "field": "acceleration",
                "op": "<",
                "value": -3.0
            }
        ],
        "logic": "AND"
    }

    # 创建挖掘任务
    task = MiningTask(
        task_id="task_001",
        strategy=MiningStrategy.TRIGGER_BASED,
        trigger_rule=trigger_rule,
        max_results=100,
        description="挖掘急刹车场景"
    )

    # 添加任务
    task_id = miner.add_mining_task(task)
    print(f"添加挖掘任务: {task_id}")

    # 执行挖掘
    sample_data = create_sample_data()
    results = miner.execute_mining(task_id, sample_data)

    print(f"挖掘结果: {len(results)} 条")

    for result in results[:5]:  # 显示前5条结果
        print(f"  - {result.data_id}: {result.reason}")


def example_vector_search_mining():
    """示例：基于向量检索的数据挖掘"""
    print("\n=== 基于向量检索的数据挖掘 ===")

    # 创建数据挖掘器
    config = MiningConfig(
        max_tasks=50,
        default_max_results=1000
    )
    miner = DataMiner(config)

    # 创建目标场景
    target_scenario = {
        "description": "前方车辆突然变道",
        "events": [
            {"type": "perception_detected", "object_type": "vehicle"},
            {"type": "planning_changed", "reason": "cut_in"}
        ]
    }

    # 创建挖掘任务
    task = MiningTask(
        task_id="task_002",
        strategy=MiningStrategy.VECTOR_SEARCH,
        vector_query=target_scenario,
        max_results=10,
        description="挖掘车辆变道场景"
    )

    # 添加任务
    task_id = miner.add_mining_task(task)
    print(f"添加挖掘任务: {task_id}")

    # 执行挖掘
    sample_data = create_sample_data()

    # 注意：实际使用时需要配置嵌入模型和向量数据库
    # 这里仅演示接口调用
    results = []

    try:
        results = miner.execute_mining(task_id, sample_data)
        print(f"挖掘结果: {len(results)} 条")

        for result in results:
            print(f"  - {result.data_id}: 相似度={result.similarity:.2f}")
    except Exception as e:
        print(f"向量检索需要配置嵌入模型: {e}")


def example_rule_filter_mining():
    """示例：基于规则过滤的数据挖掘"""
    print("\n=== 基于规则过滤的数据挖掘 ===")

    # 创建数据挖掘器
    config = MiningConfig(
        max_tasks=50,
        default_max_results=1000
    )
    miner = DataMiner(config)

    # 创建过滤规则
    filter_rules = {
        "time_range": {
            "start": time.time() - 3600 * 24,  # 最近24小时
            "end": time.time()
        },
        "weather": ["sunny", "cloudy"],
        "time_of_day": ["day"],
        "location": {
            "x_min": 0.0,
            "x_max": 500.0,
            "y_min": 0.0,
            "y_max": 500.0
        }
    }

    # 创建挖掘任务
    task = MiningTask(
        task_id="task_003",
        strategy=MiningStrategy.RULE_FILTER,
        filter_rules=filter_rules,
        max_results=50,
        description="挖掘晴天白天场景"
    )

    # 添加任务
    task_id = miner.add_mining_task(task)
    print(f"添加挖掘任务: {task_id}")

    # 执行挖掘
    sample_data = create_sample_data()
    results = miner.execute_mining(task_id, sample_data)

    print(f"挖掘结果: {len(results)} 条")

    for result in results[:5]:
        print(f"  - {result.data_id}: {result.reason}")


def example_event_sequence_mining():
    """示例：基于事件序列的数据挖掘"""
    print("\n=== 基于事件序列的数据挖掘 ===")

    # 创建数据挖掘器
    config = MiningConfig(
        max_tasks=50,
        default_max_results=1000
    )
    miner = DataMiner(config)

    # 创建目标事件序列
    target_sequence = [
        EventType.PERCEPTION_DETECTED,
        EventType.PREDICTION_UPDATED,
        EventType.PLANNING_CHANGED,
        EventType.CONTROL_UPDATED
    ]

    # 创建挖掘任务
    task = MiningTask(
        task_id="task_004",
        strategy=MiningStrategy.EVENT_SEQUENCE,
        event_sequence=target_sequence,
        max_results=20,
        description="挖掘完整感知-预测-规划-控制序列"
    )

    # 添加任务
    task_id = miner.add_mining_task(task)
    print(f"添加挖掘任务: {task_id}")

    # 执行挖掘
    sample_data = create_sample_data()
    results = miner.execute_mining(task_id, sample_data)

    print(f"挖掘结果: {len(results)} 条")

    for result in results[:5]:
        print(f"  - {result.data_id}: {result.reason}")


def show_mining_statistics():
    """显示挖掘统计"""
    print("\n=== 挖掘统计 ===")

    # 创建数据挖掘器
    config = MiningConfig(
        max_tasks=50,
        default_max_results=1000
    )
    miner = DataMiner(config)

    stats = miner.get_statistics()

    print(f"总任务数: {stats['total_tasks']}")
    print(f"活跃任务数: {stats['active_tasks']}")
    print(f"已完成任务数: {stats['completed_tasks']}")
    print(f"失败任务数: {stats['failed_tasks']}")

    print("\n按策略统计:")
    for strategy, count in stats['strategy_stats'].items():
        print(f"  {strategy}: {count}")


def main():
    """主函数"""
    print("云端数据挖掘示例")
    print("=" * 50)

    # 运行示例
    example_trigger_based_mining()
    example_vector_search_mining()
    example_rule_filter_mining()
    example_event_sequence_mining()

    # 显示统计信息
    show_mining_statistics()

    print("\n示例运行完成！")


if __name__ == "__main__":
    main()
