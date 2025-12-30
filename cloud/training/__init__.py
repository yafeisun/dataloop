"""
训练闭环模块
实现自动评测和训练闭环
"""

from .auto_evaluator import (
    AutoEvaluator,
    EvaluationReport,
    TestResult,
    EvaluationMetric,
    EvaluationType
)
from .training_loop import (
    TrainingLoop,
    TrainingConfig,
    TrainingResult,
    TrainingProgress,
    TrainingStatus,
    OTAConfig,
    OTAResult
)

__all__ = [
    "AutoEvaluator",
    "EvaluationReport",
    "TestResult",
    "EvaluationMetric",
    "EvaluationType",
    "TrainingLoop",
    "TrainingConfig",
    "TrainingResult",
    "TrainingProgress",
    "TrainingStatus",
    "OTAConfig",
    "OTAResult"
]