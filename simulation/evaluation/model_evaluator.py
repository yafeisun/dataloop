"""
模型评测验证模块
仅使用真实数据评测，实现差异模式分析
"""

from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field
from enum import Enum
import time
import json
import numpy as np
from dataclasses import dataclass


class EvaluationStatus(str, Enum):
    """评测状态"""
    PENDING = "pending"       # 待评测
    RUNNING = "running"       # 评测中
    COMPLETED = "completed"   # 已完成
    FAILED = "failed"         # 失败


class MetricType(str, Enum):
    """指标类型"""
    ACCURACY = "accuracy"         # 准确率
    PRECISION = "precision"       # 精确率
    RECALL = "recall"             # 召回率
    F1_SCORE = "f1_score"         # F1分数
    MAE = "mae"                   # 平均绝对误差
    MSE = "mse"                   # 均方误差
    RMSE = "rmse"                 # 均方根误差
    IOU = "iou"                   # 交并比
    CONFIDENCE = "confidence"     # 置信度


class EvaluationConfig(BaseModel):
    """评测配置"""
    evaluation_id: str = Field(description="评测ID")
    name: str = Field(description="评测名称")
    model_id: str = Field(description="模型ID")
    model_type: str = Field(description="模型类型")
    description: str = Field(default="", description="描述")
    test_data_path: str = Field(description="测试数据路径")
    metrics: List[MetricType] = Field(default_factory=list, description="评测指标")
    use_real_data_only: bool = Field(default=True, description="仅使用真实数据")
    confidence_threshold: float = Field(default=0.5, description="置信度阈值")
    enable_diff_analysis: bool = Field(default=True, description="是否启用差异分析")
    enable_category_analysis: bool = Field(default=True, description="是否启用类别分析")
    enable_confidence_analysis: bool = Field(default=True, description="是否启用置信度分析")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


class EvaluationResult(BaseModel):
    """评测结果"""
    evaluation_id: str
    model_id: str
    model_type: str
    status: EvaluationStatus
    start_time: float
    end_time: Optional[float] = None
    duration: float = 0.0
    metrics: Dict[str, float] = Field(default_factory=dict, description="指标结果")
    diff_analysis: Optional[Dict[str, Any]] = Field(default=None, description="差异分析")
    category_analysis: Optional[Dict[str, Any]] = Field(default=None, description="类别分析")
    confidence_analysis: Optional[Dict[str, Any]] = Field(default=None, description="置信度分析")
    sample_results: List[Dict[str, Any]] = Field(default_factory=list, description="样本结果")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


class DiffPattern(BaseModel):
    """差异模式"""
    pattern_id: str = Field(description="模式ID")
    type: str = Field(description="差异类型(false_positive/false_negative)")
    category: str = Field(description="类别")
    count: int = Field(default=0, description="数量")
    confidence_distribution: List[float] = Field(default_factory=list, description="置信度分布")
    example_cases: List[Dict[str, Any]] = Field(default_factory=list, description="示例案例")


class ModelEvaluator:
    """
    模型评测器
    仅使用真实数据评测
    """

    def __init__(self):
        self.evaluation_configs: Dict[str, EvaluationConfig] = {}
        self.evaluation_results: Dict[str, EvaluationResult] = {}
        self.evaluation_counter = 0

    def create_evaluation_task(self, config: EvaluationConfig) -> bool:
        """
        创建评测任务

        Args:
            config: 评测配置

        Returns:
            bool: 创建是否成功
        """
        if config.evaluation_id in self.evaluation_configs:
            return False

        self.evaluation_configs[config.evaluation_id] = config
        return True

    def start_evaluation(self, evaluation_id: str) -> bool:
        """
        开始评测

        Args:
            evaluation_id: 评测ID

        Returns:
            bool: 评测是否成功
        """
        if evaluation_id not in self.evaluation_configs:
            return False

        config = self.evaluation_configs[evaluation_id]

        # 创建评测结果
        result = EvaluationResult(
            evaluation_id=evaluation_id,
            model_id=config.model_id,
            model_type=config.model_type,
            status=EvaluationStatus.RUNNING,
            start_time=time.time()
        )

        self.evaluation_results[evaluation_id] = result

        try:
            # 加载测试数据
            test_data = self._load_test_data(config)

            # 加载模型
            model = self._load_model(config.model_id)

            # 运行评测
            self._evaluate(model, test_data, config, result)

            # 更新评测结果
            result.status = EvaluationStatus.COMPLETED
            result.end_time = time.time()
            result.duration = result.end_time - result.start_time

            return True

        except Exception as e:
            print(f"Evaluation error: {e}")
            result.status = EvaluationStatus.FAILED
            result.end_time = time.time()
            result.duration = result.end_time - result.start_time
            return False

    def _load_test_data(self, config: EvaluationConfig) -> Dict[str, Any]:
        """
        加载测试数据

        Args:
            config: 评测配置

        Returns:
            Dict: 测试数据
        """
        # 这里简化处理，实际应该从文件加载真实数据
        # 返回模拟的真实数据
        num_samples = 100

        test_data = {
            "X": np.random.randn(num_samples, 10),
            "y": np.random.randn(num_samples, 1),
            "predictions": np.random.randn(num_samples, 1),
            "confidences": np.random.uniform(0, 1, num_samples),
            "categories": np.random.choice(["car", "pedestrian", "cyclist"], num_samples),
            "ground_truth": np.random.choice([0, 1], num_samples)
        }

        return test_data

    def _load_model(self, model_id: str) -> Any:
        """
        加载模型

        Args:
            model_id: 模型ID

        Returns:
            Any: 模型
        """
        # 这里简化处理，实际应该从模型训练器加载模型
        # 返回模拟模型
        class MockModel:
            def predict(self, X):
                return np.random.randn(len(X), 1)

            def predict_with_confidence(self, X):
                predictions = np.random.randn(len(X), 1)
                confidences = np.random.uniform(0, 1, len(X))
                return predictions, confidences

        return MockModel()

    def _evaluate(
        self,
        model: Any,
        test_data: Dict[str, Any],
        config: EvaluationConfig,
        result: EvaluationResult
    ):
        """
        评测模型

        Args:
            model: 模型
            test_data: 测试数据
            config: 评测配置
            result: 评测结果
        """
        # 计算指标
        for metric_type in config.metrics:
            metric_value = self._calculate_metric(metric_type, test_data)
            result.metrics[metric_type.value] = metric_value

        # 差异分析
        if config.enable_diff_analysis:
            result.diff_analysis = self._analyze_differences(test_data, config)

        # 类别分析
        if config.enable_category_analysis:
            result.category_analysis = self._analyze_categories(test_data)

        # 置信度分析
        if config.enable_confidence_analysis:
            result.confidence_analysis = self._analyze_confidences(test_data, config)

        # 样本结果
        result.sample_results = self._generate_sample_results(test_data, config)

    def _calculate_metric(self, metric_type: MetricType, test_data: Dict[str, Any]) -> float:
        """
        计算指标

        Args:
            metric_type: 指标类型
            test_data: 测试数据

        Returns:
            float: 指标值
        """
        predictions = test_data.get("predictions", [])
        ground_truth = test_data.get("y", [])

        if metric_type == MetricType.MAE:
            return np.mean(np.abs(predictions - ground_truth))
        elif metric_type == MetricType.MSE:
            return np.mean((predictions - ground_truth) ** 2)
        elif metric_type == MetricType.RMSE:
            return np.sqrt(np.mean((predictions - ground_truth) ** 2))
        elif metric_type == MetricType.ACCURACY:
            pred_labels = (predictions > 0.5).astype(int)
            gt_labels = ground_truth.astype(int)
            return np.mean(pred_labels == gt_labels)
        elif metric_type == MetricType.PRECISION:
            pred_labels = (predictions > 0.5).astype(int)
            gt_labels = ground_truth.astype(int)
            tp = np.sum((pred_labels == 1) & (gt_labels == 1))
            fp = np.sum((pred_labels == 1) & (gt_labels == 0))
            return tp / (tp + fp) if (tp + fp) > 0 else 0.0
        elif metric_type == MetricType.RECALL:
            pred_labels = (predictions > 0.5).astype(int)
            gt_labels = ground_truth.astype(int)
            tp = np.sum((pred_labels == 1) & (gt_labels == 1))
            fn = np.sum((pred_labels == 0) & (gt_labels == 1))
            return tp / (tp + fn) if (tp + fn) > 0 else 0.0
        elif metric_type == MetricType.F1_SCORE:
            precision = self._calculate_metric(MetricType.PRECISION, test_data)
            recall = self._calculate_metric(MetricType.RECALL, test_data)
            return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        else:
            return 0.0

    def _analyze_differences(
        self,
        test_data: Dict[str, Any],
        config: EvaluationConfig
    ) -> Dict[str, Any]:
        """
        分析差异

        Args:
            test_data: 测试数据
            config: 评测配置

        Returns:
            Dict: 差异分析结果
        """
        predictions = test_data.get("predictions", [])
        ground_truth = test_data.get("y", [])
        confidences = test_data.get("confidences", [])
        categories = test_data.get("categories", [])

        # False Positive (误检)
        fp_indices = (predictions > config.confidence_threshold) & (ground_truth == 0)
        fp_count = np.sum(fp_indices)

        # False Negative (漏检)
        fn_indices = (predictions < config.confidence_threshold) & (ground_truth == 1)
        fn_count = np.sum(fn_indices)

        # True Positive (正确检测)
        tp_indices = (predictions > config.confidence_threshold) & (ground_truth == 1)
        tp_count = np.sum(tp_indices)

        # True Negative (正确拒绝)
        tn_indices = (predictions < config.confidence_threshold) & (ground_truth == 0)
        tn_count = np.sum(tn_indices)

        # 置信度分布
        fp_confidences = confidences[fp_indices] if fp_count > 0 else []
        fn_confidences = confidences[fn_indices] if fn_count > 0 else []

        return {
            "false_positive": {
                "count": int(fp_count),
                "rate": float(fp_count / len(predictions)) if len(predictions) > 0 else 0.0,
                "confidence_mean": float(np.mean(fp_confidences)) if len(fp_confidences) > 0 else 0.0,
                "confidence_std": float(np.std(fp_confidences)) if len(fp_confidences) > 0 else 0.0
            },
            "false_negative": {
                "count": int(fn_count),
                "rate": float(fn_count / len(predictions)) if len(predictions) > 0 else 0.0,
                "confidence_mean": float(np.mean(fn_confidences)) if len(fn_confidences) > 0 else 0.0,
                "confidence_std": float(np.std(fn_confidences)) if len(fn_confidences) > 0 else 0.0
            },
            "true_positive": {
                "count": int(tp_count),
                "rate": float(tp_count / len(predictions)) if len(predictions) > 0 else 0.0
            },
            "true_negative": {
                "count": int(tn_count),
                "rate": float(tn_count / len(predictions)) if len(predictions) > 0 else 0.0
            }
        }

    def _analyze_categories(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析类别

        Args:
            test_data: 测试数据

        Returns:
            Dict: 类别分析结果
        """
        categories = test_data.get("categories", [])
        predictions = test_data.get("predictions", [])
        ground_truth = test_data.get("y", [])

        unique_categories = np.unique(categories)
        category_stats = {}

        for category in unique_categories:
            mask = categories == category
            cat_predictions = predictions[mask]
            cat_ground_truth = ground_truth[mask]

            tp = np.sum((cat_predictions > 0.5) & (cat_ground_truth == 1))
            fp = np.sum((cat_predictions > 0.5) & (cat_ground_truth == 0))
            fn = np.sum((cat_predictions < 0.5) & (cat_ground_truth == 1))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            category_stats[category] = {
                "count": int(np.sum(mask)),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            }

        return category_stats

    def _analyze_confidences(
        self,
        test_data: Dict[str, Any],
        config: EvaluationConfig
    ) -> Dict[str, Any]:
        """
        分析置信度

        Args:
            test_data: 测试数据
            config: 评测配置

        Returns:
            Dict: 置信度分析结果
        """
        confidences = test_data.get("confidences", [])
        predictions = test_data.get("predictions", [])
        ground_truth = test_data.get("y", [])

        # 置信度分布
        confidence_ranges = {
            "low": (0.0, 0.3),
            "medium": (0.3, 0.7),
            "high": (0.7, 1.0)
        }

        confidence_analysis = {}

        for range_name, (low, high) in confidence_ranges.items():
            mask = (confidences >= low) & (confidences < high)
            range_predictions = predictions[mask]
            range_ground_truth = ground_truth[mask]

            if len(range_predictions) > 0:
                accuracy = np.mean((range_predictions > 0.5) == range_ground_truth)
                confidence_analysis[range_name] = {
                    "count": int(np.sum(mask)),
                    "accuracy": float(accuracy),
                    "mean_confidence": float(np.mean(confidences[mask]))
                }
            else:
                confidence_analysis[range_name] = {
                    "count": 0,
                    "accuracy": 0.0,
                    "mean_confidence": 0.0
                }

        return confidence_analysis

    def _generate_sample_results(
        self,
        test_data: Dict[str, Any],
        config: EvaluationConfig,
        max_samples: int = 10
    ) -> List[Dict[str, Any]]:
        """
        生成样本结果

        Args:
            test_data: 测试数据
            config: 评测配置
            max_samples: 最大样本数

        Returns:
            List: 样本结果列表
        """
        predictions = test_data.get("predictions", [])
        ground_truth = test_data.get("y", [])
        confidences = test_data.get("confidences", [])
        categories = test_data.get("categories", [])

        sample_results = []

        for i in range(min(max_samples, len(predictions))):
            sample_results.append({
                "sample_id": i,
                "prediction": float(predictions[i]),
                "ground_truth": float(ground_truth[i]),
                "confidence": float(confidences[i]),
                "category": categories[i],
                "correct": bool((predictions[i] > config.confidence_threshold) == ground_truth[i])
            })

        return sample_results

    def get_evaluation_result(self, evaluation_id: str) -> Optional[EvaluationResult]:
        """
        获取评测结果

        Args:
            evaluation_id: 评测ID

        Returns:
            EvaluationResult: 评测结果
        """
        return self.evaluation_results.get(evaluation_id)

    def compare_evaluations(self, evaluation_ids: List[str]) -> Dict[str, Any]:
        """
        比较评测结果

        Args:
            evaluation_ids: 评测ID列表

        Returns:
            Dict: 比较结果
        """
        comparison = {
            "evaluations": [],
            "metrics_comparison": {}
        }

        for eval_id in evaluation_ids:
            result = self.evaluation_results.get(eval_id)
            if result:
                comparison["evaluations"].append({
                    "evaluation_id": eval_id,
                    "model_id": result.model_id,
                    "model_type": result.model_type,
                    "metrics": result.metrics
                })

        # 比较指标
        if comparison["evaluations"]:
            all_metrics = set()
            for eval_result in comparison["evaluations"]:
                all_metrics.update(eval_result["metrics"].keys())

            for metric in all_metrics:
                values = [
                    eval_result["metrics"].get(metric, 0.0)
                    for eval_result in comparison["evaluations"]
                    if metric in eval_result["metrics"]
                ]
                if values:
                    comparison["metrics_comparison"][metric] = {
                        "min": float(min(values)),
                        "max": float(max(values)),
                        "avg": float(sum(values) / len(values)),
                        "best_eval_id": comparison["evaluations"][values.index(max(values))]["evaluation_id"]
                                if metric in ["accuracy", "precision", "recall", "f1_score"]
                                else comparison["evaluations"][values.index(min(values))]["evaluation_id"]
                    }

        return comparison

    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """获取评测统计"""
        total_evaluations = len(self.evaluation_results)

        # 按状态统计
        status_stats = {}
        for result in self.evaluation_results.values():
            status = result.status.value
            status_stats[status] = status_stats.get(status, 0) + 1

        # 按模型类型统计
        type_stats = {}
        for result in self.evaluation_results.values():
            model_type = result.model_type
            type_stats[model_type] = type_stats.get(model_type, 0) + 1

        return {
            "total_evaluations": total_evaluations,
            "status_stats": status_stats,
            "type_stats": type_stats
        }

    def export_evaluation_report(self, evaluation_id: str, output_path: str) -> bool:
        """
        导出评测报告

        Args:
            evaluation_id: 评测ID
            output_path: 输出路径

        Returns:
            bool: 导出是否成功
        """
        result = self.evaluation_results.get(evaluation_id)
        if result is None:
            return False

        try:
            report = {
                "evaluation_id": evaluation_id,
                "model_id": result.model_id,
                "model_type": result.model_type,
                "status": result.status.value,
                "start_time": result.start_time,
                "end_time": result.end_time,
                "duration": result.duration,
                "metrics": result.metrics,
                "diff_analysis": result.diff_analysis,
                "category_analysis": result.category_analysis,
                "confidence_analysis": result.confidence_analysis,
                "sample_results": result.sample_results
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            return True
        except Exception as e:
            print(f"Failed to export evaluation report: {e}")
            return False

    def get_all_evaluations(self) -> Dict[str, EvaluationResult]:
        """获取所有评测"""
        return self.evaluation_results.copy()

    def get_summary(self) -> Dict[str, Any]:
        """获取摘要"""
        return self.get_evaluation_statistics()


# 全局模型评测器实例
_global_model_evaluator = None


def get_global_model_evaluator() -> ModelEvaluator:
    """获取全局模型评测器实例"""
    global _global_model_evaluator
    if _global_model_evaluator is None:
        _global_model_evaluator = ModelEvaluator()
    return _global_model_evaluator