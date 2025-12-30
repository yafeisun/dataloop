"""
车端Trigger示例
演示如何使用Trigger框架监控体感指标
"""

import sys
import os
import time
import json
from typing import Dict, Any

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from onboard.triggers.trigger_manager import get_global_trigger_manager
from onboard.triggers.rule_trigger import (
    RuleTriggerConfig,
    create_emergency_brake_trigger,
    create_sharp_turn_trigger,
    create_frequent_stop_trigger
)
from onboard.metrics.somatic_metrics import (
    get_global_monitor,
    MetricLevel,
    MetricConfig,
    MetricType
)
from onboard.logger.data_logger import get_global_logger, LogLevel


def create_sample_data() -> Dict[str, Any]:
    """创建示例传感器数据"""
    return {
        "timestamp": time.time(),
        "vehicle": {
            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
            "velocity": {"x": 10.0, "y": 0.0, "z": 0.0},
            "acceleration": {"x": 0.0, "y": 0.0, "z": 0.0},
            "heading": 0.0,
            "yaw_rate": 0.0
        },
        "sensors": {
            "camera": [],
            "lidar": [],
            "radar": []
        },
        "state_machine": {
            "state": "driving",
            "mode": "autonomous"
        },
        "control": {
            "throttle": 0.3,
            "brake": 0.0,
            "steering": 0.0
        }
    }


def simulate_emergency_brake():
    """模拟急刹车场景"""
    print("\n=== 模拟急刹车场景 ===")

    # 创建急刹车Trigger
    brake_trigger = create_emergency_brake_trigger(threshold=3.0)
    trigger_manager = get_global_trigger_manager()
    trigger_manager.register_trigger(brake_trigger)

    # 创建数据
    data = create_sample_data()

    # 模拟急刹车
    print("触发急刹车...")
    data["vehicle"]["acceleration"] = {"x": -4.0, "y": 0.0, "z": 0.0}
    data["vehicle"]["velocity"] = {"x": 5.0, "y": 0.0, "z": 0.0}

    # 评估Trigger
    result = trigger_manager.evaluate_trigger(brake_trigger.config.trigger_id, data)

    print(f"Trigger结果: {result.triggered}")
    print(f"置信度: {result.confidence}")
    print(f"原因: {result.reason}")

    # 更新体感指标
    monitor = get_global_monitor()
    monitor.update_metric("emergency_brake", 4.0, time.time())

    # 创建日志
    logger = get_global_logger()
    logger.register_log_config(
        log_id=f"micro_{brake_trigger.config.trigger_id}",
        log_level=LogLevel.MICRO,
        trigger_id=brake_trigger.config.trigger_id,
        data_window=10.0,
        before_window=5.0,
        after_window=5.0,
        include_sensors=["camera", "lidar"],
        include_state_machine=True,
        include_control_commands=True
    )

    log_entry = logger.create_micro_log(
        trigger_id=brake_trigger.config.trigger_id,
        timestamp=time.time(),
        sensor_data=data["sensors"],
        state_machine=data["state_machine"],
        control_commands=data["control"]
    )

    if log_entry:
        print(f"创建日志: {log_entry.log_id}")
        print(f"日志大小: {log_entry.size} bytes")


def simulate_sharp_turn():
    """模拟急转弯场景"""
    print("\n=== 模拟急转弯场景 ===")

    # 创建急转弯Trigger
    turn_trigger = create_sharp_turn_trigger(threshold=15.0)
    trigger_manager = get_global_trigger_manager()
    trigger_manager.register_trigger(turn_trigger)

    # 创建数据
    data = create_sample_data()

    # 模拟急转弯
    print("触发急转弯...")
    data["vehicle"]["yaw_rate"] = 20.0
    data["vehicle"]["acceleration"] = {"x": 0.0, "y": 3.0, "z": 0.0}

    # 评估Trigger
    result = trigger_manager.evaluate_trigger(turn_trigger.config.trigger_id, data)

    print(f"Trigger结果: {result.triggered}")
    print(f"置信度: {result.confidence}")
    print(f"原因: {result.reason}")

    # 更新体感指标
    monitor = get_global_monitor()
    monitor.update_metric("sharp_turn", 20.0, time.time())


def simulate_frequent_stop():
    """模拟频繁启停场景"""
    print("\n=== 模拟频繁启停场景 ===")

    # 创建频繁启停Trigger
    stop_trigger = create_frequent_stop_trigger(
        count_threshold=3,
        time_window=60.0
    )
    trigger_manager = get_global_trigger_manager()
    trigger_manager.register_trigger(stop_trigger)

    # 创建数据
    data = create_sample_data()

    # 模拟频繁启停
    print("触发频繁启停...")
    data["stop_count"] = 4
    data["time_window"] = 45.0

    # 评估Trigger
    result = trigger_manager.evaluate_trigger(stop_trigger.config.trigger_id, data)

    print(f"Trigger结果: {result.triggered}")
    print(f"置信度: {result.confidence}")
    print(f"原因: {result.reason}")

    # 更新体感指标
    monitor = get_global_monitor()
    monitor.update_metric("frequent_stop", 4, time.time())


def show_metrics_summary():
    """显示指标摘要"""
    print("\n=== 体感指标摘要 ===")

    monitor = get_global_monitor()
    summary = monitor.get_summary()

    print(f"总指标数: {summary['total_metrics']}")
    print(f"启用指标数: {summary['enabled_metrics']}")
    print(f"一级指标: {summary['level_1_metrics']}")
    print(f"二级指标: {summary['level_2_metrics']}")
    print(f"三级指标: {summary['level_3_metrics']}")

    if summary['exceeded_metrics']:
        print(f"\n超阈值指标: {summary['exceeded_metrics']}")
    else:
        print("\n无超阈值指标")


def show_trigger_statistics():
    """显示Trigger统计"""
    print("\n=== Trigger统计 ===")

    trigger_manager = get_global_trigger_manager()
    stats = trigger_manager.get_statistics()

    print(f"总Trigger数: {stats['total_triggers']}")
    print(f"启用Trigger数: {stats['enabled_triggers']}")
    print(f"禁用Trigger数: {stats['disabled_triggers']}")

    print("\n按优先级统计:")
    for priority, count in stats['priority_stats'].items():
        print(f"  {priority}: {count}")

    print("\n按类型统计:")
    for trigger_type, count in stats['type_stats'].items():
        print(f"  {trigger_type}: {count}")


def show_logger_statistics():
    """显示日志统计"""
    print("\n=== 日志统计 ===")

    logger = get_global_logger()
    stats = logger.get_statistics()

    print(f"总日志数: {stats['total_logs']}")
    print(f"已上传日志数: {stats['uploaded_logs']}")
    print(f"待上传日志数: {stats['pending_logs']}")
    print(f"队列大小: {stats['queue_size']}")

    print("\n按级别统计:")
    for level, count in stats['level_stats'].items():
        print(f"  {level}: {count}")


def main():
    """主函数"""
    print("车端Trigger示例")
    print("=" * 50)

    # 初始化全局组件
    trigger_manager = get_global_trigger_manager()
    monitor = get_global_monitor()
    logger = get_global_logger()

    # 运行示例
    simulate_emergency_brake()
    simulate_sharp_turn()
    simulate_frequent_stop()

    # 显示统计信息
    show_metrics_summary()
    show_trigger_statistics()
    show_logger_statistics()

    print("\n示例运行完成！")


if __name__ == "__main__":
    main()