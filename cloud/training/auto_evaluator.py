"""
自动评测模块
实现自动评测和模型性能对比
"""

from typing import Dict, List, Optional, Any, Callable
from pydantic import BaseModel, Field
from enum import Enum
import time


class EvaluationMetric(str, Enum):
    """评测指标"""
    ACCURACY = "accuracy"           # 准确率
    PRECISION = "precision"         # 精确率
    RECALL = "recall"               # 召回率
    F1_SCORE = "f1_score"          # F1分数
    IOU = "iou"                     # IoU
    AVERAGE_PRECISION = "average_precision"  # 平均精度
    MAE = "mae"                     # 平均绝对误差
    RMSE = "rmse"                   # 均方根误差
    LATENCY = "latency"             # 延迟
    THROUGHPUT = "throughput"       # 吞吐量


class EvaluationType(str, Enum):
    """评测类型"""
    SINGLE_MODEL = "single_model"           # 单模型评测
    MODEL_COMPARISON = "model_comparison"   # 模型对比
    REGRESSION_TEST = "regression_test"     # 回归测试
    ABLATION_STUDY = "ablation_study"       # 消融实验


class TestResult(BaseModel):
    """测试结果"""
    test_id: str
    model_id: str
    model_version: str
    timestamp: float
    metrics: Dict[str, float] = Field(default_factory=dict)
    predictions: List[Dict[str, Any]] = Field(default_factory=list)
    ground_truth: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvaluationReport(BaseModel):
    """评测报告"""
    evaluation_id: str
    evaluation_type: EvaluationType
    models: List[str] = Field(default_factory=list)
    test_results: List[TestResult] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AutoEvaluator:
    """
    自动评测器
    自动评测模型性能
    """

    def __init__(self):
        self.evaluation_history: List[EvaluationReport] = []
        self._model_registry: Dict[str, Dict[str, Any]] = {}

    def register_model(
        self,
        model_id: str,
        model_version: str,
        model_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        注册模型

        Args:
            model_id: 模型ID
            model_version: 模型版本
            model_path: 模型路径
            metadata: 元数据

        Returns:
            bool: 注册是否成功
        """
        key = f"{model_id}_{model_version}"

        if key in self._model_registry:
            return False

        self._model_registry[key] = {
            "model_id": model_id,
            "model_version": model_version,
            "model_path": model_path,
            "metadata": metadata or {}
        }

        return True

    def evaluate_model(
        self,
        model_id: str,
        model_version: str,
        test_data: List[Dict[str, Any]],
        metrics: List[EvaluationMetric],
        prediction_callback: Callable
    ) -> TestResult:
        """
        评测单个模型

        Args:
            model_id: 模型ID
            model_version: 模型版本
            test_data: 测试数据
            metrics: 评测指标列表
            prediction_callback: 预测回调函数

        Returns:
            TestResult: 测试结果
        """
        # 获取模型信息
        key = f"{model_id}_{model_version}"
        model_info = self._model_registry.get(key)

        if not model_info:
            raise ValueError(f"Model not found: {key}")

        # 运行预测
        predictions = []
        ground_truth = []

        for data_point in test_data:
            # 调用预测回调
            prediction = prediction_callback(model_info, data_point)
            predictions.append(prediction)

            # 收集真值
            if "ground_truth" in data_point:
                ground_truth.append(data_point["ground_truth"])

        # 计算指标
        metric_values = {}
        for metric in metrics:
            value = self._calculate_metric(metric, predictions, ground_truth)
            metric_values[metric.value] = value

        # 创建测试结果
        test_result = TestResult(
            test_id=f"test_{int(time.time())}",
            model_id=model_id,
            model_version=model_version,
            timestamp=time.time(),
            metrics=metric_values,
            predictions=predictions,
            ground_truth=ground_truth
        )

        return test_result

    def compare_models(
        self,
        model_configs: List[Dict[str, str]],
        test_data: List[Dict[str, Any]],
        metrics: List[EvaluationMetric],
        prediction_callback: Callable
    ) -> EvaluationReport:
        """
        对比多个模型

        Args:
            model_configs: 模型配置列表 [{"model_id": "xxx", "model_version": "xxx"}]
            test_data: 测试数据
            metrics: 评测指标列表
            prediction_callback: 预测回调函数

        Returns:
            EvaluationReport: 评测报告
        """
        evaluation_id = f"eval_{int(time.time())}"
        test_results = []

        # 评测每个模型
        for config in model_configs:
            test_result = self.evaluate_model(
                model_id=config["model_id"],
                model_version=config["model_version"],
                test_data=test_data,
                metrics=metrics,
                prediction_callback=prediction_callback
            )
            test_results.append(test_result)

        # 生成摘要
        summary = self._generate_comparison_summary(test_results, metrics)

        # 创建评测报告
        report = EvaluationReport(
            evaluation_id=evaluation_id,
            evaluation_type=EvaluationType.MODEL_COMPARISON,
            models=[f"{c['model_id']}_{c['model_version']}" for c in model_configs],
            test_results=test_results,
            summary=summary
        )

        # 记录历史
        self.evaluation_history.append(report)

        return report

    def run_regression_test(
        self,
        model_id: str,
        model_version: str,
        test_cases: List[Dict[str, Any]],
        expected_results: List[Dict[str, Any]],
        prediction_callback: Callable,
        tolerance: float = 0.01
    ) -> EvaluationReport:
        """
        运行回归测试

        Args:
            model_id: 模型ID
            model_version: 模型版本
            test_cases: 测试用例
            expected_results: 期望结果
            prediction_callback: 预测回调函数
            tolerance: 容差

        Returns:
            EvaluationReport: 评测报告
        """
        evaluation_id = f"regression_{int(time.time())}"
        test_results = []

        # 获取模型信息
        key = f"{model_id}_{model_version}"
        model_info = self._model_registry.get(key)

        if not model_info:
            raise ValueError(f"Model not found: {key}")

        # 运行测试
        passed_count = 0
        failed_count = 0

        for test_case, expected in zip(test_cases, expected_results):
            # 调用预测回调
            prediction = prediction_callback(model_info, test_case)

            # 检查是否通过
            passed = self._check_prediction(prediction, expected, tolerance)

            if passed:
                passed_count += 1
            else:
                failed_count += 1

            # 记录结果
            test_result = TestResult(
                test_id=f"regression_{len(test_results)}",
                model_id=model_id,
                model_version=model_version,
                timestamp=time.time(),
                metrics={"passed": 1.0 if passed else 0.0},
                predictions=[prediction],
                ground_truth=[expected],
                metadata={"test_case": test_case, "expected": expected}
            )

            test_results.append(test_result)

        # 生成摘要
        summary = {
            "total_tests": len(test_cases),
            "passed": passed_count,
            "failed": failed_count,
            "pass_rate": passed_count / len(test_cases) if test_cases else 0.0
        }

        # 创建评测报告
        report = EvaluationReport(
            evaluation_id=evaluation_id,
            evaluation_type=EvaluationType.REGRESSION_TEST,
            models=[f"{model_id}_{model_version}"],
            test_results=test_results,
            summary=summary
        )

        # 记录历史
        self.evaluation_history.append(report)

        return report

    def _calculate_metric(
        self,
        metric: EvaluationMetric,
        predictions: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]]
    ) -> float:
        """
        计算指标

        Args:
            metric: 指标类型
            predictions: 预测结果
            ground_truth: 真值

        Returns:
            float: 指标值
        """
        if metric == EvaluationMetric.ACCURACY:
            return self._calculate_accuracy(predictions, ground_truth)
        elif metric == EvaluationMetric.PRECISION:
            return self._calculate_precision(predictions, ground_truth)
        elif metric == EvaluationMetric.RECALL:
            return self._calculate_recall(predictions, ground_truth)
        elif metric == EvaluationMetric.F1_SCORE:
            return self._calculate_f1_score(predictions, ground_truth)
        elif metric == EvaluationMetric.IOU:
            return self._calculate_iou(predictions, ground_truth)
        elif metric == EvaluationMetric.MAE:
            return self._calculate_mae(predictions, ground_truth)
        elif metric == EvaluationMetric.RMSE:
            return self._calculate_rmse(predictions, ground_truth)
        else:
            return 0.0

    def _calculate_accuracy(
        self,
        predictions: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]]
    ) -> float:
        """计算准确率"""
        correct = 0
        total = len(predictions)

        for pred, gt in zip(predictions, ground_truth):
            if pred.get("label") == gt.get("label"):
                correct += 1

        return correct / total if total > 0 else 0.0

    def _calculate_precision(
        self,
        predictions: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]]
    ) -> float:
        """计算精确率"""
        true_positive = 0
        false_positive = 0

        for pred, gt in zip(predictions, ground_truth):
            if pred.get("label") == 1 and gt.get("label") == 1:
                true_positive += 1
            elif pred.get("label") == 1 and gt.get("label") == 0:
                false_positive += 1

        return true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0

    def _calculate_recall(
        self,
        predictions: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]]
    ) -> float:
        """计算召回率"""
        true_positive = 0
        false_negative = 0

        for pred, gt in zip(predictions, ground_truth):
            if pred.get("label") == 1 and gt.get("label") == 1:
                true_positive += 1
            elif pred.get("label") == 0 and gt.get("label") == 1:
                false_negative += 1

        return true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0

    def _calculate_f1_score(
        self,
        predictions: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]]
    ) -> float:
        """计算F1分数"""
        precision = self._calculate_precision(predictions, ground_truth)
        recall = self._calculate_recall(predictions, ground_truth)

        return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    def _calculate_iou(
        self,
        predictions: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]]
    ) -> float:
        """计算IoU"""
        total_iou = 0
        count = 0

        for pred, gt in zip(predictions, ground_truth):
            pred_bbox = pred.get("bbox")
            gt_bbox = gt.get("bbox")

            if pred_bbox and gt_bbox:
                iou = self._calculate_bbox_iou(pred_bbox, gt_bbox)
                total_iou += iou
                count += 1

        return total_iou / count if count > 0 else 0.0

    def _calculate_bbox_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """计算边界框IoU"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # 计算交集
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)

        # 计算并集
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _calculate_mae(
        self,
        predictions: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]]
    ) -> float:
        """计算平均绝对误差"""
        errors = []

        for pred, gt in zip(predictions, ground_truth):
            pred_value = pred.get("value", 0)
            gt_value = gt.get("value", 0)
            errors.append(abs(pred_value - gt_value))

        return sum(errors) / len(errors) if errors else 0.0

    def _calculate_rmse(
        self,
        predictions: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]]
    ) -> float:
        """计算均方根误差"""
        errors = []

        for pred, gt in zip(predictions, ground_truth):
            pred_value = pred.get("value", 0)
            gt_value = gt.get("value", 0)
            errors.append((pred_value - gt_value) ** 2)

        mse = sum(errors) / len(errors) if errors else 0.0
        return mse ** 0.5

    def _check_prediction(
        self,
        prediction: Dict[str, Any],
        expected: Dict[str, Any],
        tolerance: float
    ) -> bool:
        """
        检查预测是否符合期望

        Args:
            prediction: 预测结果
            expected: 期望结果
            tolerance: 容差

        Returns:
            bool: 是否通过
        """
        for key, expected_value in expected.items():
            pred_value = prediction.get(key)

            if isinstance(expected_value, (int, float)):
                if abs(pred_value - expected_value) > tolerance:
                    return False
            elif pred_value != expected_value:
                return False

        return True

    def _generate_comparison_summary(
        self,
        test_results: List[TestResult],
        metrics: List[EvaluationMetric]
    ) -> Dict[str, Any]:
        """
        生成对比摘要

        Args:
            test_results: 测试结果列表
            metrics: 指标列表

        Returns:
            Dict: 摘要
        """
        summary = {
            "num_models": len(test_results),
            "best_models": {},
            "metric_comparison": {}
        }

        # 找出每个指标的最佳模型
        for metric in metrics:
            metric_key = metric.value
            best_model = None
            best_score = -float('inf')

            for result in test_results:
                score = result.metrics.get(metric_key, 0)
                if score > best_score:
                    best_score = score
                    best_model = f"{result.model_id}_{result.model_version}"

            summary["best_models"][metric_key] = {
                "model": best_model,
                "score": best_score
            }

        # 指标对比
        for metric in metrics:
            metric_key = metric.value
            summary["metric_comparison"][metric_key] = [
                {
                    "model": f"{r.model_id}_{r.model_version}",
                    "score": r.metrics.get(metric_key, 0)
                }
                for r in test_results
            ]

        return summary

    def get_evaluation_history(
        self,
        evaluation_type: Optional[EvaluationType] = None,
        limit: int = 100
    ) -> List[EvaluationReport]:
        """
        获取评测历史

        Args:
            evaluation_type: 评测类型过滤
            limit: 返回数量限制

        Returns:
            List[EvaluationReport]: 评测报告列表
        """
        history = self.evaluation_history.copy()

        if evaluation_type is not None:
            history = [h for h in history if h.evaluation_type == evaluation_type]

        return history[-limit:]

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_evaluations = len(self.evaluation_history)

        # 按类型统计
        type_stats = {}
        for report in self.evaluation_history:
            etype = report.evaluation_type.value
            type_stats[etype] = type_stats.get(etype, 0) + 1

        return {
            "total_evaluations": total_evaluations,
            "registered_models": len(self._model_registry),
            "type_stats": type_stats
        }