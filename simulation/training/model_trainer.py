"""
模型训练模块
支持真实数据+生成式数据训练，实现模型管理和版本控制
"""

from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field
from enum import Enum
import time
import json
import os
from dataclasses import dataclass
from datetime import datetime


class TrainingStatus(str, Enum):
    """训练状态"""
    PENDING = "pending"       # 待训练
    RUNNING = "running"       # 训练中
    COMPLETED = "completed"   # 已完成
    FAILED = "failed"         # 失败
    CANCELLED = "cancelled"   # 已取消


class ModelType(str, Enum):
    """模型类型"""
    PERCEPTION = "perception"     # 感知模型
    PREDICTION = "prediction"     # 预测模型
    PLANNING = "planning"         # 规划模型
    CONTROL = "control"           # 控制模型
    FUSION = "fusion"             # 融合模型


class TrainingConfig(BaseModel):
    """训练配置"""
    training_id: str = Field(description="训练ID")
    name: str = Field(description="训练名称")
    model_type: ModelType = Field(description="模型类型")
    description: str = Field(default="", description="描述")
    epochs: int = Field(default=100, description="训练轮数")
    batch_size: int = Field(default=32, description="批次大小")
    learning_rate: float = Field(default=0.001, description="学习率")
    optimizer: str = Field(default="adam", description="优化器")
    loss_function: str = Field(default="mse", description="损失函数")
    use_real_data: bool = Field(default=True, description="是否使用真实数据")
    use_generated_data: bool = Field(default=False, description="是否使用生成数据")
    real_data_path: str = Field(default="", description="真实数据路径")
    generated_data_path: str = Field(default="", description="生成数据路径")
    validation_split: float = Field(default=0.2, description="验证集比例")
    early_stopping: bool = Field(default=True, description="是否早停")
    early_stopping_patience: int = Field(default=10, description="早停耐心值")
    save_best_model: bool = Field(default=True, description="是否保存最佳模型")
    checkpoint_interval: int = Field(default=10, description="检查点间隔")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


class TrainingResult(BaseModel):
    """训练结果"""
    training_id: str
    model_type: ModelType
    status: TrainingStatus
    start_time: float
    end_time: Optional[float] = None
    duration: float = 0.0
    epochs_completed: int = 0
    best_epoch: int = 0
    best_loss: float = float('inf')
    best_metric: float = 0.0
    final_loss: float = 0.0
    final_metric: float = 0.0
    model_path: str = ""
    checkpoint_paths: List[str] = Field(default_factory=list)
    training_history: Dict[str, List[float]] = Field(default_factory=dict)
    validation_history: Dict[str, List[float]] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ModelInfo(BaseModel):
    """模型信息"""
    model_id: str = Field(description="模型ID")
    name: str = Field(description="模型名称")
    model_type: ModelType = Field(description="模型类型")
    version: str = Field(description="版本号")
    created_at: float = Field(default_factory=time.time, description="创建时间")
    training_id: str = Field(description="训练ID")
    model_path: str = Field(description="模型路径")
    config_path: str = Field(description="配置路径")
    performance_metrics: Dict[str, float] = Field(default_factory=dict, description="性能指标")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


class ModelTrainer:
    """
    模型训练器
    支持真实数据+生成式数据训练
    """

    def __init__(self, output_dir: str = "./models"):
        self.training_configs: Dict[str, TrainingConfig] = {}
        self.training_results: Dict[str, TrainingResult] = {}
        self.models: Dict[str, ModelInfo] = {}
        self.output_dir = output_dir
        self.training_counter = 0
        self.model_counter = 0

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)

    def create_training_task(self, config: TrainingConfig) -> bool:
        """
        创建训练任务

        Args:
            config: 训练配置

        Returns:
            bool: 创建是否成功
        """
        if config.training_id in self.training_configs:
            return False

        self.training_configs[config.training_id] = config
        return True

    def start_training(self, training_id: str) -> bool:
        """
        开始训练

        Args:
            training_id: 训练ID

        Returns:
            bool: 训练是否成功
        """
        if training_id not in self.training_configs:
            return False

        config = self.training_configs[training_id]

        # 创建训练结果
        result = TrainingResult(
            training_id=training_id,
            model_type=config.model_type,
            status=TrainingStatus.RUNNING,
            start_time=time.time()
        )

        self.training_results[training_id] = result

        try:
            # 加载数据
            train_data, val_data = self._load_data(config)

            # 创建模型
            model = self._create_model(config)

            # 训练模型
            self._train_model(model, train_data, val_data, config, result)

            # 保存模型
            model_path = self._save_model(model, config, result)
            result.model_path = model_path

            # 创建模型信息
            self.model_counter += 1
            model_id = f"model_{self.model_counter}"
            model_info = ModelInfo(
                model_id=model_id,
                name=config.name,
                model_type=config.model_type,
                version=f"v1.0.{self.model_counter}",
                training_id=training_id,
                model_path=model_path,
                config_path=self._save_config(config),
                performance_metrics={
                    "best_loss": result.best_loss,
                    "best_metric": result.best_metric,
                    "final_loss": result.final_loss,
                    "final_metric": result.final_metric
                }
            )

            self.models[model_id] = model_info

            # 更新训练结果
            result.status = TrainingStatus.COMPLETED
            result.end_time = time.time()
            result.duration = result.end_time - result.start_time

            return True

        except Exception as e:
            print(f"Training error: {e}")
            result.status = TrainingStatus.FAILED
            result.end_time = time.time()
            result.duration = result.end_time - result.start_time
            return False

    def _load_data(self, config: TrainingConfig) -> Tuple[Any, Any]:
        """
        加载数据

        Args:
            config: 训练配置

        Returns:
            Tuple: 训练数据和验证数据
        """
        # 这里简化处理，实际应该从文件加载数据
        # 返回模拟数据
        import numpy as np

        train_size = 1000
        val_size = int(train_size * config.validation_split)

        train_data = {
            "X": np.random.randn(train_size, 10),
            "y": np.random.randn(train_size, 1)
        }

        val_data = {
            "X": np.random.randn(val_size, 10),
            "y": np.random.randn(val_size, 1)
        }

        return train_data, val_data

    def _create_model(self, config: TrainingConfig) -> Any:
        """
        创建模型

        Args:
            config: 训练配置

        Returns:
            Any: 模型
        """
        # 这里简化处理，实际应该根据模型类型创建不同的模型
        import torch
        import torch.nn as nn

        class SimpleModel(nn.Module):
            def __init__(self, input_size=10, hidden_size=64, output_size=1):
                super(SimpleModel, self).__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.fc2 = nn.Linear(hidden_size, hidden_size)
                self.fc3 = nn.Linear(hidden_size, output_size)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                x = self.fc3(x)
                return x

        model = SimpleModel()
        return model

    def _train_model(
        self,
        model: Any,
        train_data: Dict[str, Any],
        val_data: Dict[str, Any],
        config: TrainingConfig,
        result: TrainingResult
    ):
        """
        训练模型

        Args:
            model: 模型
            train_data: 训练数据
            val_data: 验证数据
            config: 训练配置
            result: 训练结果
        """
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset

        # 转换数据
        X_train = torch.FloatTensor(train_data["X"])
        y_train = torch.FloatTensor(train_data["y"])
        X_val = torch.FloatTensor(val_data["X"])
        y_val = torch.FloatTensor(val_data["y"])

        # 创建数据加载器
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

        # 定义损失函数和优化器
        criterion = nn.MSELoss()

        if config.optimizer == "adam":
            optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        elif config.optimizer == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=config.learning_rate)
        else:
            optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

        # 训练循环
        best_loss = float('inf')
        patience_counter = 0

        result.training_history = {"loss": []}
        result.validation_history = {"loss": []}

        for epoch in range(config.epochs):
            # 训练
            model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            result.training_history["loss"].append(train_loss)

            # 验证
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            result.validation_history["loss"].append(val_loss)

            # 更新结果
            result.epochs_completed = epoch + 1
            result.final_loss = val_loss

            # 保存最佳模型
            if val_loss < best_loss:
                best_loss = val_loss
                result.best_loss = best_loss
                result.best_epoch = epoch + 1
                patience_counter = 0
            else:
                patience_counter += 1

            # 早停
            if config.early_stopping and patience_counter >= config.early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

            # 保存检查点
            if (epoch + 1) % config.checkpoint_interval == 0:
                checkpoint_path = self._save_checkpoint(model, config, epoch + 1)
                result.checkpoint_paths.append(checkpoint_path)

    def _save_model(self, model: Any, config: TrainingConfig, result: TrainingResult) -> str:
        """
        保存模型

        Args:
            model: 模型
            config: 训练配置
            result: 训练结果

        Returns:
            str: 模型路径
        """
        model_dir = os.path.join(self.output_dir, config.model_type.value)
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, f"{config.name}_{config.training_id}.pt")

        import torch
        torch.save(model.state_dict(), model_path)

        return model_path

    def _save_checkpoint(self, model: Any, config: TrainingConfig, epoch: int) -> str:
        """
        保存检查点

        Args:
            model: 模型
            config: 训练配置
            epoch: 轮数

        Returns:
            str: 检查点路径
        """
        checkpoint_dir = os.path.join(self.output_dir, config.model_type.value, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_path = os.path.join(checkpoint_dir, f"{config.name}_epoch_{epoch}.pt")

        import torch
        torch.save(model.state_dict(), checkpoint_path)

        return checkpoint_path

    def _save_config(self, config: TrainingConfig) -> str:
        """
        保存配置

        Args:
            config: 训练配置

        Returns:
            str: 配置路径
        """
        config_dir = os.path.join(self.output_dir, config.model_type.value, "configs")
        os.makedirs(config_dir, exist_ok=True)

        config_path = os.path.join(config_dir, f"{config.name}_{config.training_id}.json")

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config.dict(), f, indent=2, ensure_ascii=False)

        return config_path

    def get_training_result(self, training_id: str) -> Optional[TrainingResult]:
        """
        获取训练结果

        Args:
            training_id: 训练ID

        Returns:
            TrainingResult: 训练结果
        """
        return self.training_results.get(training_id)

    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """
        获取模型

        Args:
            model_id: 模型ID

        Returns:
            ModelInfo: 模型信息
        """
        return self.models.get(model_id)

    def get_models_by_type(self, model_type: ModelType) -> List[ModelInfo]:
        """
        根据类型获取模型

        Args:
            model_type: 模型类型

        Returns:
            List[ModelInfo]: 模型列表
        """
        return [
            model for model in self.models.values()
            if model.model_type == model_type
        ]

    def load_model(self, model_id: str) -> Optional[Any]:
        """
        加载模型

        Args:
            model_id: 模型ID

        Returns:
            Any: 模型
        """
        model_info = self.models.get(model_id)
        if model_info is None:
            return None

        try:
            import torch
            from simulation.training.model_trainer import ModelTrainer

            trainer = ModelTrainer()
            model = trainer._create_model(
                TrainingConfig(
                    training_id="temp",
                    name="temp",
                    model_type=model_info.model_type
                )
            )

            model.load_state_dict(torch.load(model_info.model_path))
            model.eval()

            return model
        except Exception as e:
            print(f"Failed to load model: {e}")
            return None

    def delete_model(self, model_id: str) -> bool:
        """
        删除模型

        Args:
            model_id: 模型ID

        Returns:
            bool: 删除是否成功
        """
        if model_id not in self.models:
            return False

        model_info = self.models[model_id]

        # 删除模型文件
        try:
            if os.path.exists(model_info.model_path):
                os.remove(model_info.model_path)
            if os.path.exists(model_info.config_path):
                os.remove(model_info.config_path)
        except Exception as e:
            print(f"Failed to delete model files: {e}")

        del self.models[model_id]
        return True

    def compare_models(self, model_ids: List[str]) -> Dict[str, Any]:
        """
        比较模型

        Args:
            model_ids: 模型ID列表

        Returns:
            Dict: 比较结果
        """
        comparison = {
            "models": [],
            "metrics": {}
        }

        for model_id in model_ids:
            model_info = self.models.get(model_id)
            if model_info:
                comparison["models"].append({
                    "model_id": model_id,
                    "name": model_info.name,
                    "version": model_info.version,
                    "performance": model_info.performance_metrics
                })

        # 计算指标
        if comparison["models"]:
            for metric in ["best_loss", "best_metric", "final_loss", "final_metric"]:
                values = [
                    m["performance"].get(metric, 0.0)
                    for m in comparison["models"]
                    if metric in m["performance"]
                ]
                if values:
                    comparison["metrics"][metric] = {
                        "min": min(values),
                        "max": max(values),
                        "avg": sum(values) / len(values)
                    }

        return comparison

    def get_training_statistics(self) -> Dict[str, Any]:
        """获取训练统计"""
        total_trainings = len(self.training_results)
        total_models = len(self.models)

        # 按状态统计训练
        status_stats = {}
        for result in self.training_results.values():
            status = result.status.value
            status_stats[status] = status_stats.get(status, 0) + 1

        # 按类型统计模型
        type_stats = {}
        for model in self.models.values():
            model_type = model.model_type.value
            type_stats[model_type] = type_stats.get(model_type, 0) + 1

        return {
            "total_trainings": total_trainings,
            "total_models": total_models,
            "status_stats": status_stats,
            "type_stats": type_stats
        }

    def get_all_models(self) -> Dict[str, ModelInfo]:
        """获取所有模型"""
        return self.models.copy()

    def get_summary(self) -> Dict[str, Any]:
        """获取摘要"""
        return self.get_training_statistics()


# 全局模型训练器实例
_global_model_trainer = None


def get_global_model_trainer(output_dir: str = "./models") -> ModelTrainer:
    """获取全局模型训练器实例"""
    global _global_model_trainer
    if _global_model_trainer is None:
        _global_model_trainer = ModelTrainer(output_dir)
    return _global_model_trainer