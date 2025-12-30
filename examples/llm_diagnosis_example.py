"""
LLM问题诊断示例
演示如何使用LLM进行问题诊断和根因分析
"""

import sys
import os
import time
from typing import Dict, Any, List

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cloud.diagnosis.llm_diagnosis import (
    LLMDiagnosisEngine,
    DiagnosisConfig,
    DiagnosisRequest,
    DiagnosisResult,
    DiagnosisSeverity
)
from common.events.event_types import (
    Event,
    EventSequence,
    EventType,
    EventBuilder
)


def create_sample_event_sequence() -> EventSequence:
    """创建示例事件序列"""
    events = []

    base_time = time.time()

    # 感知事件：检测到车辆
    events.append(EventBuilder.create_perception_event(
        object_id="vehicle_001",
        object_type="vehicle",
        position={"x": 20.0, "y": 0.0, "z": 0.0},
        confidence=0.95,
        timestamp=base_time
    ))

    # 预测事件：预测车辆轨迹
    events.append(EventBuilder.create_prediction_event(
        object_id="vehicle_001",
        predicted_trajectory=[
            {"x": 20.0, "y": 0.0, "z": 0.0},
            {"x": 25.0, "y": 0.5, "z": 0.0},
            {"x": 30.0, "y": 1.0, "z": 0.0}
        ],
        confidence=0.85,
        timestamp=base_time + 0.1
    ))

    # 规划事件：规划变道
    events.append(EventBuilder.create_planning_event(
        plan_id="plan_001",
        planned_trajectory=[
            {"x": 0.0, "y": 0.0, "z": 0.0},
            {"x": 5.0, "y": 0.5, "z": 0.0},
            {"x": 10.0, "y": 1.0, "z": 0.0}
        ],
        planned_speed=10.0,
        timestamp=base_time + 0.2
    ))

    # 控制事件：执行控制
    events.append(EventBuilder.create_control_event(
        control_id="control_001",
        throttle=0.3,
        brake=0.0,
        steering=0.1,
        timestamp=base_time + 0.3
    ))

    # 车辆事件：急刹车
    events.append(EventBuilder.create_vehicle_event(
        vehicle_id="ego_vehicle",
        position={"x": 5.0, "y": 0.0, "z": 0.0},
        velocity={"x": 2.0, "y": 0.0, "z": 0.0},
        acceleration={"x": -4.0, "y": 0.0, "z": 0.0},
        timestamp=base_time + 0.4
    ))

    # 创建事件序列
    sequence = EventSequence(
        sequence_id="seq_001",
        events=events,
        start_time=base_time,
        end_time=base_time + 0.5,
        duration=0.5,
        metadata={
            "scenario": "cut_in_brake",
            "location": "intersection",
            "weather": "sunny"
        }
    )

    return sequence


def example_diagnosis():
    """示例：LLM诊断"""
    print("\n=== LLM诊断示例 ===")

    # 创建诊断引擎
    config = DiagnosisConfig(
        llm_model="gpt-4",
        temperature=0.7,
        max_tokens=2000,
        weak_supervision_confidence_threshold=0.7
    )
    engine = LLMDiagnosisEngine(config)

    # 创建事件序列
    event_sequence = create_sample_event_sequence()

    # 创建诊断请求
    request = DiagnosisRequest(
        request_id="req_001",
        event_sequence=event_sequence,
        problem_description="车辆在变道过程中突然急刹车，体感不佳",
        additional_context={
            "trigger_id": "emergency_brake",
            "severity": "critical",
            "location": "intersection",
            "weather": "sunny"
        }
    )

    # 执行诊断
    print("执行LLM诊断...")
    result = engine.diagnose(request)

    print(f"\n诊断结果:")
    print(f"问题类型: {result.problem_type}")
    print(f"严重程度: {result.severity}")
    print(f"根因分析: {result.root_cause}")
    print(f"影响范围: {result.impact_scope}")
    print(f"建议措施: {result.recommendations}")

    print(f"\n置信度: {result.confidence:.2f}")
    print(f"诊断时间: {result.diagnosis_time:.2f}s")


def example_batch_diagnosis():
    """示例：批量诊断"""
    print("\n=== 批量诊断示例 ===")

    # 创建诊断引擎
    config = DiagnosisConfig(
        llm_model="gpt-4",
        temperature=0.7,
        max_tokens=2000
    )
    engine = LLMDiagnosisEngine(config)

    # 创建多个诊断请求
    requests = []

    for i in range(3):
        event_sequence = create_sample_event_sequence()
        request = DiagnosisRequest(
            request_id=f"req_{i:03d}",
            event_sequence=event_sequence,
            problem_description=f"问题场景 {i + 1}",
            additional_context={"index": i}
        )
        requests.append(request)

    # 执行批量诊断
    print("执行批量诊断...")
    results = engine.batch_diagnose(requests)

    print(f"\n批量诊断结果: {len(results)} 条")

    for i, result in enumerate(results):
        print(f"\n结果 {i + 1}:")
        print(f"  问题类型: {result.problem_type}")
        print(f"  严重程度: {result.severity}")
        print(f"  置信度: {result.confidence:.2f}")


def example_weak_supervision():
    """示例：弱监督学习"""
    print("\n=== 弱监督学习示例 ===")

    # 创建诊断引擎
    config = DiagnosisConfig(
        llm_model="gpt-4",
        temperature=0.7,
        max_tokens=2000,
        weak_supervision_confidence_threshold=0.7
    )
    engine = LLMDiagnosisEngine(config)

    # 创建事件序列
    event_sequence = create_sample_event_sequence()

    # 创建诊断请求
    request = DiagnosisRequest(
        request_id="req_weak_001",
        event_sequence=event_sequence,
        problem_description="车辆突然急刹车",
        additional_context={}
    )

    # 执行诊断
    print("执行弱监督诊断...")
    result = engine.diagnose(request)

    print(f"\n诊断结果:")
    print(f"问题类型: {result.problem_type}")
    print(f"置信度: {result.confidence:.2f}")

    # 如果置信度足够高，添加到训练集
    if result.confidence >= config.weak_supervision_confidence_threshold:
        print(f"\n置信度足够高，添加到弱监督训练集")
        engine.add_to_training_set(result)
        print(f"训练集大小: {len(engine.training_set)}")


def example_diagnosis_statistics():
    """示例：诊断统计"""
    print("\n=== 诊断统计 ===")

    # 创建诊断引擎
    config = DiagnosisConfig(
        llm_model="gpt-4",
        temperature=0.7,
        max_tokens=2000
    )
    engine = LLMDiagnosisEngine(config)

    # 执行一些诊断
    for i in range(5):
        event_sequence = create_sample_event_sequence()
        request = DiagnosisRequest(
            request_id=f"req_{i:03d}",
            event_sequence=event_sequence,
            problem_description=f"问题 {i + 1}",
            additional_context={}
        )
        engine.diagnose(request)

    # 获取统计信息
    stats = engine.get_statistics()

    print(f"总诊断数: {stats['total_diagnoses']}")
    print(f"平均诊断时间: {stats['avg_diagnosis_time']:.2f}s")

    print("\n按问题类型统计:")
    for problem_type, count in stats['problem_type_stats'].items():
        print(f"  {problem_type}: {count}")

    print("\n按严重程度统计:")
    for severity, count in stats['severity_stats'].items():
        print(f"  {severity}: {count}")

    print(f"\n训练集大小: {len(engine.training_set)}")


def example_export_diagnosis_results():
    """示例：导出诊断结果"""
    print("\n=== 导出诊断结果 ===")

    # 创建诊断引擎
    config = DiagnosisConfig(
        llm_model="gpt-4",
        temperature=0.7,
        max_tokens=2000
    )
    engine = LLMDiagnosisEngine(config)

    # 执行一些诊断
    results = []
    for i in range(3):
        event_sequence = create_sample_event_sequence()
        request = DiagnosisRequest(
            request_id=f"req_{i:03d}",
            event_sequence=event_sequence,
            problem_description=f"问题 {i + 1}",
            additional_context={}
        )
        result = engine.diagnose(request)
        results.append(result)

    # 导出诊断结果
    output_file = "/tmp/diagnosis_results.json"
    engine.export_diagnosis_results(results, output_file)

    print(f"诊断结果已导出到: {output_file}")


def main():
    """主函数"""
    print("LLM问题诊断示例")
    print("=" * 50)

    # 运行示例
    example_diagnosis()
    example_batch_diagnosis()
    example_weak_supervision()
    example_diagnosis_statistics()
    example_export_diagnosis_results()

    print("\n示例运行完成！")
    print("\n注意：实际使用时需要配置LLM API密钥")
    print("请设置环境变量 OPENAI_API_KEY")


if __name__ == "__main__":
    main()