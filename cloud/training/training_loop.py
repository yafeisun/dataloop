"""
训练闭环模块
实现自动训练和OTA闭环
"""

from typing import Dict, List, Optional, Any, Callable
from pydantic import BaseModel, Field
from enum import Enum
import time
import threading


class TrainingStatus(str, Enum):
    """训练状态"""
    PENDING = "pending"       # 等待中
    RUNNING = "running"       # 运行中
    COMPLETED = "completed"   # 已完成
    FAILED = "failed"         # 失败
    CANCELLED = "cancelled"   # 已取消


class TrainingConfig(BaseModel):
    """训练配置"""
    training_id: str
    model_id: str
    base_model_version: str
    training_data: List[str] = Field(default_factory=list, description="训练数据ID列表")
    validation_data: List[str] = Field(default_factory=list, description="验证数据ID列表")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="超参数")
    max_epochs: int = Field(default=100, description="最大轮数")
    early_stopping_patience: int = Field(default=10, description="早停耐心值")
    checkpoint_interval: int = Field(default=5, description="检查点间隔")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TrainingProgress(BaseModel):
    """训练进度"""
    training_id: str
    current_epoch: int
    total_epochs: int
    train_loss: float
    val_loss: float
    metrics: Dict[str, float] = Field(default_factory=dict)
    elapsed_time: float
    estimated_remaining_time: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TrainingResult(BaseModel):
    """训练结果"""
    training_id: str
    model_id: str
    model_version: str
    status: TrainingStatus
    start_time: float
    end_time: Optional[float] = None
    final_metrics: Dict[str, float] = Field(default_factory=dict)
    best_checkpoint: Optional[str] = None
    logs: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class OTAConfig(BaseModel):
    """OTA配置"""
    ota_id: str
    model_id: str
    model_version: str
    target_fleet: List[str] = Field(default_factory=list, description="目标车队ID列表")
    rollout_percentage: float = Field(default=0.1, ge=0, le=1, description="灰度发布比例")
    validation_period: int = Field(default=7, description="验证周期（天）")
    rollback_threshold: float = Field(default=0.05, description="回滚阈值")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class OTAResult(BaseModel):
    """OTA结果"""
    ota_id: str
    model_id: str
    model_version: str
    status: str
    start_time: float
    end_time: Optional[float] = None
    deployed_vehicles: int
    success_rate: float
    issues: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TrainingLoop:
    """
    训练闭环管理器
    管理自动训练和OTA闭环
    """

    def __init__(self):
        self.training_history: List[TrainingResult] = []
        self.ota_history: List[OTAResult] = []
        self._active_trainings: Dict[str, TrainingConfig] = {}
        self._progress_callbacks: Dict[str, Callable] = {}
        self._lock = threading.Lock()

    def start_training(
        self,
        config: TrainingConfig,
        training_callback: Callable,
        progress_callback: Optional[Callable] = None
    ) -> str:
        """
        启动训练

        Args:
            config: 训练配置
            training_callback: 训练回调函数
            progress_callback: 进度回调函数

        Returns:
            str: 训练ID
        """
        with self._lock:
            # 检查是否已有活跃训练
            if config.training_id in self._active_trainings:
                raise ValueError(f"Training already active: {config.training_id}")

            # 注册训练
            self._active_trainings[config.training_id] = config

            # 注册进度回调
            if progress_callback:
                self._progress_callbacks[config.training_id] = progress_callback

        # 在后台线程中运行训练
        thread = threading.Thread(
            target=self._run_training,
            args=(config, training_callback),
            daemon=True
        )
        thread.start()

        return config.training_id

    def _run_training(
        self,
        config: TrainingConfig,
        training_callback: Callable
    ):
        """
        运行训练（后台线程）

        Args:
            config: 训练配置
            training_callback: 训练回调函数
        """
        training_id = config.training_id
        start_time = time.time()

        # 创建训练结果
        result = TrainingResult(
            training_id=training_id,
            model_id=config.model_id,
            model_version=f"{config.base_model_version}_train_{int(start_time)}",
            status=TrainingStatus.RUNNING,
            start_time=start_time,
            logs=[]
        )

        try:
            # 调用训练回调
            for epoch in range(config.max_epochs):
                # 检查是否取消
                with self._lock:
                    if training_id not in self._active_trainings:
                        result.status = TrainingStatus.CANCELLED
                        break

                # 训练一个epoch
                epoch_result = training_callback(config, epoch)

                # 更新进度
                if training_id in self._progress_callbacks:
                    progress = TrainingProgress(
                        training_id=training_id,
                        current_epoch=epoch + 1,
                        total_epochs=config.max_epochs,
                        train_loss=epoch_result.get("train_loss", 0),
                        val_loss=epoch_result.get("val_loss", 0),
                        metrics=epoch_result.get("metrics", {}),
                        elapsed_time=time.time() - start_time,
                        estimated_remaining_time=0
                    )
                    self._progress_callbacks[training_id](progress)

                # 记录日志
                result.logs.append(f"Epoch {epoch + 1}: train_loss={epoch_result.get('train_loss', 0):.4f}, val_loss={epoch_result.get('val_loss', 0):.4f}")

                # 早停检查
                if self._check_early_stopping(result.logs, config.early_stopping_patience):
                    result.logs.append(f"Early stopping at epoch {epoch + 1}")
                    break

            # 训练完成
            if result.status == TrainingStatus.RUNNING:
                result.status = TrainingStatus.COMPLETED
                result.final_metrics = epoch_result.get("metrics", {})
                result.best_checkpoint = f"checkpoint_{training_id}.pth"

        except Exception as e:
            result.status = TrainingStatus.FAILED
            result.logs.append(f"Training failed: {str(e)}")

        finally:
            # 清理
            with self._lock:
                if training_id in self._active_trainings:
                    del self._active_trainings[training_id]
                if training_id in self._progress_callbacks:
                    del self._progress_callbacks[training_id]

            result.end_time = time.time()

            # 记录历史
            with self._lock:
                self.training_history.append(result)

    def _check_early_stopping(self, logs: List[str], patience: int) -> bool:
        """
        检查是否应该早停

        Args:
            logs: 训练日志
            patience: 耐心值

        Returns:
            bool: 是否应该早停
        """
        if len(logs) < patience + 1:
            return False

        # 提取最近的验证损失
        recent_losses = []
        for log in logs[-patience - 1:]:
            if "val_loss=" in log:
                try:
                    loss_str = log.split("val_loss=")[1].split(",")[0]
                    loss = float(loss_str)
                    recent_losses.append(loss)
                except:
                    pass

        if len(recent_losses) < patience + 1:
            return False

        # 检查验证损失是否连续patience次没有改善
        best_loss = min(recent_losses[:-patience])
        recent_losses_without_best = recent_losses[-patience:]

        return all(loss >= best_loss for loss in recent_losses_without_best)

    def cancel_training(self, training_id: str) -> bool:
        """
        取消训练

        Args:
            training_id: 训练ID

        Returns:
            bool: 是否成功
        """
        with self._lock:
            if training_id in self._active_trainings:
                del self._active_trainings[training_id]
                return True

        return False

    def get_training_status(self, training_id: str) -> Optional[TrainingResult]:
        """
        获取训练状态

        Args:
            training_id: 训练ID

        Returns:
            Optional[TrainingResult]: 训练结果
        """
        with self._lock:
            # 检查活跃训练
            if training_id in self._active_trainings:
                config = self._active_trainings[training_id]
                return TrainingResult(
                    training_id=training_id,
                    model_id=config.model_id,
                    model_version=config.base_model_version,
                    status=TrainingStatus.RUNNING,
                    start_time=time.time()
                )

            # 检查历史训练
            for result in reversed(self.training_history):
                if result.training_id == training_id:
                    return result

        return None

    def deploy_model_via_ota(
        self,
        config: OTAConfig,
        deployment_callback: Callable
    ) -> OTAResult:
        """
        通过OTA部署模型

        Args:
            config: OTA配置
            deployment_callback: 部署回调函数

        Returns:
            OTAResult: OTA结果
        """
        start_time = time.time()

        # 调用部署回调
        deployment_result = deployment_callback(config)

        # 创建OTA结果
        result = OTAResult(
            ota_id=config.ota_id,
            model_id=config.model_id,
            model_version=config.model_version,
            status=deployment_result.get("status", "success"),
            start_time=start_time,
            end_time=time.time(),
            deployed_vehicles=deployment_result.get("deployed_vehicles", 0),
            success_rate=deployment_result.get("success_rate", 0),
            issues=deployment_result.get("issues", [])
        )

        # 记录历史
        with self._lock:
            self.ota_history.append(result)

        return result

    def monitor_deployment(
        self,
        ota_id: str,
        telemetry_callback: Callable,
        duration: int = 7
    ) -> Dict[str, Any]:
        """
        监控部署效果

        Args:
            ota_id: OTA ID
            telemetry_callback: 遥测数据回调函数
            duration: 监控时长（天）

        Returns:
            Dict: 监控结果
        """
        # 获取OTA结果
        ota_result = None
        for result in reversed(self.ota_history):
            if result.ota_id == ota_id:
                ota_result = result
                break

        if not ota_result:
            raise ValueError(f"OTA not found: {ota_id}")

        # 调用遥测回调
        telemetry_data = telemetry_callback(ota_id, duration)

        # 分析结果
        analysis = self._analyze_telemetry(telemetry_data, ota_result.rollback_threshold)

        return {
            "ota_id": ota_id,
            "model_id": ota_result.model_id,
            "model_version": ota_result.model_version,
            "telemetry_data": telemetry_data,
            "analysis": analysis,
            "recommendation": "rollback" if analysis["should_rollback"] else "continue"
        }

    def _analyze_telemetry(
        self,
        telemetry_data: Dict[str, Any],
        rollback_threshold: float
    ) -> Dict[str, Any]:
        """
        分析遥测数据

        Args:
            telemetry_data: 遥测数据
            rollback_threshold: 回滚阈值

        Returns:
            Dict: 分析结果
        """
        # 提取关键指标
        error_rate = telemetry_data.get("error_rate", 0)
        disengagement_rate = telemetry_data.get("disengagement_rate", 0)
        user_satisfaction = telemetry_data.get("user_satisfaction", 1.0)

        # 判断是否应该回滚
        should_rollback = (
            error_rate > rollback_threshold or
            disengagement_rate > rollback_threshold or
            user_satisfaction < (1 - rollback_threshold)
        )

        return {
            "error_rate": error_rate,
            "disengagement_rate": disengagement_rate,
            "user_satisfaction": user_satisfaction,
            "should_rollback": should_rollback,
            "rollback_threshold": rollback_threshold
        }

    def get_training_history(
        self,
        model_id: Optional[str] = None,
        status: Optional[TrainingStatus] = None,
        limit: int = 100
    ) -> List[TrainingResult]:
        """
        获取训练历史

        Args:
            model_id: 模型ID过滤
            status: 状态过滤
            limit: 返回数量限制

        Returns:
            List[TrainingResult]: 训练结果列表
        """
        with self._lock:
            history = self.training_history.copy()

            if model_id is not None:
                history = [h for h in history if h.model_id == model_id]

            if status is not None:
                history = [h for h in history if h.status == status]

            return history[-limit:]

    def get_ota_history(
        self,
        model_id: Optional[str] = None,
        limit: int = 100
    ) -> List[OTAResult]:
        """
        获取OTA历史

        Args:
            model_id: 模型ID过滤
            limit: 返回数量限制

        Returns:
            List[OTAResult]: OTA结果列表
        """
        with self._lock:
            history = self.ota_history.copy()

            if model_id is not None:
                history = [h for h in history if h.model_id == model_id]

            return history[-limit:]

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            total_trainings = len(self.training_history)
            active_trainings = len(self._active_trainings)

            # 按状态统计
            status_stats = {}
            for result in self.training_history:
                status = result.status.value
                status_stats[status] = status_stats.get(status, 0) + 1

            # OTA统计
            total_otas = len(self.ota_history)
            successful_otas = sum(1 for o in self.ota_history if o.status == "success")

            return {
                "total_trainings": total_trainings,
                "active_trainings": active_trainings,
                "status_stats": status_stats,
                "total_otas": total_otas,
                "successful_otas": successful_otas,
                "ota_success_rate": successful_otas / total_otas if total_otas > 0 else 0
            }