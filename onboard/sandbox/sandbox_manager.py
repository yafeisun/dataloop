"""
沙箱挖数管理器
实现配置下发、动态启停、独立进程挖数机制
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum
import json
import time
import threading
import multiprocessing
import queue
from dataclasses import dataclass
import traceback


class SandboxStatus(str, Enum):
    """沙箱状态"""
    IDLE = "idle"           # 空闲
    RUNNING = "running"     # 运行中
    PAUSED = "paused"       # 暂停
    ERROR = "error"         # 错误
    STOPPED = "stopped"     # 已停止


class SandboxConfig(BaseModel):
    """沙箱配置"""
    sandbox_id: str = Field(description="沙箱ID")
    name: str = Field(description="沙箱名称")
    description: str = Field(default="", description="描述")
    trigger_configs: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="挖数Trigger配置列表"
    )
    max_data_size: int = Field(default=1024 * 1024 * 100, description="最大数据大小（字节）")
    max_duration: int = Field(default=3600, description="最大运行时长（秒）")
    allowed_time_windows: List[Dict[str, str]] = Field(
        default_factory=list,
        description="允许运行的时间窗口"
    )
    enabled: bool = Field(default=True, description="是否启用")
    priority: int = Field(default=0, description="优先级（数字越大优先级越高）")


class SandboxResult(BaseModel):
    """沙箱挖数结果"""
    sandbox_id: str
    timestamp: float
    triggered: bool
    trigger_id: str
    data: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class SandboxTask:
    """沙箱任务"""
    config: SandboxConfig
    data_queue: multiprocessing.Queue
    result_queue: multiprocessing.Queue
    control_queue: multiprocessing.Queue


class SandboxManager:
    """
    沙箱挖数管理器
    负责沙箱的创建、启停、配置更新
    """

    def __init__(self, max_sandboxes: int = 10):
        self.sandboxes: Dict[str, SandboxConfig] = {}
        self.processes: Dict[str, multiprocessing.Process] = {}
        self.data_queues: Dict[str, multiprocessing.Queue] = {}
        self.result_queues: Dict[str, multiprocessing.Queue] = {}
        self.control_queues: Dict[str, multiprocessing.Queue] = {}
        self.status: Dict[str, SandboxStatus] = {}
        self.statistics: Dict[str, Dict[str, Any]] = {}
        self.max_sandboxes = max_sandboxes
        self._lock = threading.Lock()

    def create_sandbox(self, config: SandboxConfig) -> bool:
        """
        创建沙箱

        Args:
            config: 沙箱配置

        Returns:
            bool: 创建是否成功
        """
        with self._lock:
            if config.sandbox_id in self.sandboxes:
                return False

            if len(self.sandboxes) >= self.max_sandboxes:
                return False

            # 创建队列
            data_queue = multiprocessing.Queue(maxsize=1000)
            result_queue = multiprocessing.Queue(maxsize=1000)
            control_queue = multiprocessing.Queue(maxsize=100)

            # 保存配置和队列
            self.sandboxes[config.sandbox_id] = config
            self.data_queues[config.sandbox_id] = data_queue
            self.result_queues[config.sandbox_id] = result_queue
            self.control_queues[config.sandbox_id] = control_queue
            self.status[config.sandbox_id] = SandboxStatus.IDLE
            self.statistics[config.sandbox_id] = {
                "total_triggered": 0,
                "total_data_size": 0,
                "start_time": None,
                "last_trigger_time": None
            }

            return True

    def start_sandbox(self, sandbox_id: str) -> bool:
        """
        启动沙箱

        Args:
            sandbox_id: 沙箱ID

        Returns:
            bool: 启动是否成功
        """
        with self._lock:
            if sandbox_id not in self.sandboxes:
                return False

            if sandbox_id in self.processes and self.processes[sandbox_id].is_alive():
                return False

            config = self.sandboxes[sandbox_id]

            # 检查是否在允许的时间窗口内
            if not self._is_allowed_time_window(config):
                print(f"Sandbox {sandbox_id} is not in allowed time window")
                return False

            # 创建进程
            process = multiprocessing.Process(
                target=self._sandbox_worker,
                args=(
                    config,
                    self.data_queues[sandbox_id],
                    self.result_queues[sandbox_id],
                    self.control_queues[sandbox_id]
                ),
                daemon=True
            )

            process.start()
            self.processes[sandbox_id] = process
            self.status[sandbox_id] = SandboxStatus.RUNNING
            self.statistics[sandbox_id]["start_time"] = time.time()

            return True

    def stop_sandbox(self, sandbox_id: str) -> bool:
        """
        停止沙箱

        Args:
            sandbox_id: 沙箱ID

        Returns:
            bool: 停止是否成功
        """
        with self._lock:
            if sandbox_id not in self.processes:
                return False

            # 发送停止命令
            try:
                self.control_queues[sandbox_id].put({"command": "stop"}, timeout=1.0)
            except:
                pass

            # 等待进程结束
            process = self.processes[sandbox_id]
            if process.is_alive():
                process.join(timeout=5.0)
                if process.is_alive():
                    process.terminate()

            del self.processes[sandbox_id]
            self.status[sandbox_id] = SandboxStatus.STOPPED

            return True

    def pause_sandbox(self, sandbox_id: str) -> bool:
        """暂停沙箱"""
        with self._lock:
            if sandbox_id not in self.sandboxes:
                return False

            if self.status[sandbox_id] != SandboxStatus.RUNNING:
                return False

            # 发送暂停命令
            try:
                self.control_queues[sandbox_id].put({"command": "pause"}, timeout=1.0)
                self.status[sandbox_id] = SandboxStatus.PAUSED
                return True
            except:
                return False

    def resume_sandbox(self, sandbox_id: str) -> bool:
        """恢复沙箱"""
        with self._lock:
            if sandbox_id not in self.sandboxes:
                return False

            if self.status[sandbox_id] != SandboxStatus.PAUSED:
                return False

            # 发送恢复命令
            try:
                self.control_queues[sandbox_id].put({"command": "resume"}, timeout=1.0)
                self.status[sandbox_id] = SandboxStatus.RUNNING
                return True
            except:
                return False

    def update_sandbox_config(self, sandbox_id: str, **kwargs) -> bool:
        """
        更新沙箱配置

        Args:
            sandbox_id: 沙箱ID
            **kwargs: 配置参数

        Returns:
            bool: 更新是否成功
        """
        with self._lock:
            if sandbox_id not in self.sandboxes:
                return False

            config = self.sandboxes[sandbox_id]
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)

            # 如果沙箱正在运行，发送配置更新命令
            if sandbox_id in self.processes and self.processes[sandbox_id].is_alive():
                try:
                    self.control_queues[sandbox_id].put({
                        "command": "update_config",
                        "config": config.dict()
                    }, timeout=1.0)
                except:
                    pass

            return True

    def feed_data(self, sandbox_id: str, data: Dict[str, Any]) -> bool:
        """
        向沙箱投喂数据

        Args:
            sandbox_id: 沙箱ID
            data: 输入数据

        Returns:
            bool: 投喂是否成功
        """
        if sandbox_id not in self.data_queues:
            return False

        try:
            self.data_queues[sandbox_id].put(data, timeout=0.1)
            return True
        except queue.Full:
            return False

    def get_results(self, sandbox_id: str, timeout: float = 0.1) -> List[SandboxResult]:
        """
        获取沙箱结果

        Args:
            sandbox_id: 沙箱ID
            timeout: 超时时间

        Returns:
            List[SandboxResult]: 结果列表
        """
        results = []
        if sandbox_id not in self.result_queues:
            return results

        while True:
            try:
                result = self.result_queues[sandbox_id].get(timeout=timeout)
                results.append(SandboxResult(**result))
            except queue.Empty:
                break

        # 更新统计
        if results:
            stats = self.statistics[sandbox_id]
            stats["total_triggered"] += len(results)
            stats["last_trigger_time"] = time.time()

        return results

    def get_sandbox_status(self, sandbox_id: str) -> Optional[SandboxStatus]:
        """获取沙箱状态"""
        return self.status.get(sandbox_id)

    def get_sandbox_statistics(self, sandbox_id: str) -> Optional[Dict[str, Any]]:
        """获取沙箱统计信息"""
        return self.statistics.get(sandbox_id)

    def get_all_sandboxes(self) -> Dict[str, SandboxConfig]:
        """获取所有沙箱配置"""
        return self.sandboxes.copy()

    def delete_sandbox(self, sandbox_id: str) -> bool:
        """
        删除沙箱

        Args:
            sandbox_id: 沙箱ID

        Returns:
            bool: 删除是否成功
        """
        with self._lock:
            # 先停止沙箱
            if sandbox_id in self.processes:
                self.stop_sandbox(sandbox_id)

            # 删除配置和队列
            if sandbox_id in self.sandboxes:
                del self.sandboxes[sandbox_id]
            if sandbox_id in self.data_queues:
                del self.data_queues[sandbox_id]
            if sandbox_id in self.result_queues:
                del self.result_queues[sandbox_id]
            if sandbox_id in self.control_queues:
                del self.control_queues[sandbox_id]
            if sandbox_id in self.status:
                del self.status[sandbox_id]
            if sandbox_id in self.statistics:
                del self.statistics[sandbox_id]

            return True

    def _is_allowed_time_window(self, config: SandboxConfig) -> bool:
        """检查是否在允许的时间窗口内"""
        if not config.allowed_time_windows:
            return True

        current_time = time.localtime()
        current_hour = current_time.tm_hour
        current_minute = current_time.tm_min
        current_time_str = f"{current_hour:02d}:{current_minute:02d}"

        for window in config.allowed_time_windows:
            start = window.get("start", "00:00")
            end = window.get("end", "23:59")

            if start <= current_time_str <= end:
                return True

        return False

    def _sandbox_worker(
        self,
        config: SandboxConfig,
        data_queue: multiprocessing.Queue,
        result_queue: multiprocessing.Queue,
        control_queue: multiprocessing.Queue
    ):
        """
        沙箱工作进程

        Args:
            config: 沙箱配置
            data_queue: 数据队列
            result_queue: 结果队列
            control_queue: 控制队列
        """
        from onboard.triggers.trigger_manager import TriggerManager
        from onboard.triggers.rule_trigger import RuleTrigger, RuleTriggerConfig

        # 创建Trigger管理器
        trigger_manager = TriggerManager()

        # 加载Trigger配置
        for trigger_config_dict in config.trigger_configs:
            try:
                trigger_config = RuleTriggerConfig(**trigger_config_dict)
                trigger = RuleTrigger(trigger_config)
                trigger_manager.register_trigger(trigger)
            except Exception as e:
                print(f"Failed to load trigger: {e}")

        paused = False
        start_time = time.time()

        while True:
            # 检查控制命令
            try:
                control = control_queue.get(timeout=0.1)
                command = control.get("command")

                if command == "stop":
                    break
                elif command == "pause":
                    paused = True
                elif command == "resume":
                    paused = False
                elif command == "update_config":
                    # 重新加载Trigger配置
                    new_config = control.get("config")
                    if new_config:
                        config = SandboxConfig(**new_config)
                        # 重新加载Trigger
                        trigger_manager = TriggerManager()
                        for trigger_config_dict in config.trigger_configs:
                            try:
                                trigger_config = RuleTriggerConfig(**trigger_config_dict)
                                trigger = RuleTrigger(trigger_config)
                                trigger_manager.register_trigger(trigger)
                            except Exception as e:
                                print(f"Failed to reload trigger: {e}")

            except queue.Empty:
                pass

            # 检查是否暂停
            if paused:
                continue

            # 检查是否超时
            if time.time() - start_time > config.max_duration:
                print(f"Sandbox {config.sandbox_id} reached max duration")
                break

            # 获取数据
            try:
                data = data_queue.get(timeout=0.1)

                # 评估Trigger
                results = trigger_manager.evaluate_all(data)

                # 发送触发结果
                for result in results:
                    if result.triggered:
                        result_data = {
                            "sandbox_id": config.sandbox_id,
                            "timestamp": result.timestamp,
                            "triggered": True,
                            "trigger_id": result.trigger_id,
                            "data": result.data,
                            "metadata": {
                                "reason": result.reason,
                                "confidence": result.confidence
                            }
                        }
                        try:
                            result_queue.put(result_data, timeout=0.1)
                        except queue.Full:
                            pass

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Sandbox worker error: {e}")
                traceback.print_exc()

    def get_summary(self) -> Dict[str, Any]:
        """获取摘要"""
        total_sandboxes = len(self.sandboxes)
        running_sandboxes = sum(
            1 for status in self.status.values()
            if status == SandboxStatus.RUNNING
        )
        paused_sandboxes = sum(
            1 for status in self.status.values()
            if status == SandboxStatus.PAUSED
        )

        return {
            "total_sandboxes": total_sandboxes,
            "running_sandboxes": running_sandboxes,
            "paused_sandboxes": paused_sandboxes,
            "stopped_sandboxes": total_sandboxes - running_sandboxes - paused_sandboxes,
            "max_sandboxes": self.max_sandboxes
        }


# 全局沙箱管理器实例
_global_sandbox_manager = None


def get_global_sandbox_manager() -> SandboxManager:
    """获取全局沙箱管理器实例"""
    global _global_sandbox_manager
    if _global_sandbox_manager is None:
        _global_sandbox_manager = SandboxManager()
    return _global_sandbox_manager