# IFLOW.md - 自动驾驶数据闭环系统

## 项目概述

这是一个基于智驾行业7年实战经验构建的**真数据闭环架构**系统，专注于L4自动驾驶（物流无人车/载人）、Urban NOA、L3/L4场景。

**核心理念**："指标当第一公民，数据当产品，问题当Bug追"

**多模态索引理念**："不要人工打标签，也不要只存AI的文本总结"

**项目类型**：Python 软件项目

**主要技术栈**：
- Python 3.x
- Pydantic（数据验证）
- NumPy / Pandas（数据处理）
- PyTorch / Transformers（机器学习）
- FAISS（向量检索）
- CLIP / BLIP-2（VLM模型）
- NetworkX（场景图）
- NeRF / 3D Gaussian Splatting（4D重建）
- OpenAI / LangChain（LLM集成）
- CARLA（仿真）

## 项目结构

```
auto_calibration/
├── onboard/              # 车端模块
│   ├── triggers/        # Trigger框架（规则/组合/管理器）
│   ├── metrics/         # 体感指标监控（一级/二级/三级）
│   ├── logger/          # 数据采集与日志（micro/mini/full）
│   ├── sandbox/         # 沙箱挖数机制
│   └── calibration/     # 标定模块（端云协同动态标定）
├── cloud/               # 云端模块
│   ├── mining/          # 数据挖掘（多模态索引系统）
│   │   ├── auto_labeling.py      # 自动化结构化提取
│   │   ├── vector_indexing.py    # 向量化检索
│   │   ├── scene_graph.py         # 场景图构建
│   │   ├── fine_grained_schema.py # 细粒度标注Schema
│   │   └── data_miner.py          # 数据挖掘器（集成版）
│   ├── calibration/     # 标定监控与诊断
│   ├── diagnosis/       # LLM问题诊断
│   ├── distribution/    # 问题分发
│   └── annotation/      # 数据标注
├── simulation/          # 仿真模块
│   ├── generation/      # 场景生成
│   │   ├── scene_generator.py    # 场景生成
│   │   └── scene_reconstruction.py # 4D重建
│   ├── training/        # 模型训练
│   └── evaluation/      # 评测验证
├── common/              # 公共模块
│   ├── utils/           # 工具函数
│   ├── models/          # 数据模型（Trigger基类）
│   └── events/          # 事件定义
├── config/              # 配置文件
│   └── default_config.yaml  # 默认配置
├── tests/               # 测试文件
├── docs/                # 文档
├── examples/            # 示例代码
│   ├── onboard_trigger_example.py
│   ├── cloud_mining_example.py
│   ├── llm_diagnosis_example.py
│   └── multimodal_mining_example.py
├── ARCHITECTURE.md       # 架构设计文档
├── README.md             # 项目说明
├── IFLOW.md              # iFlow上下文文档
└── requirements.txt      # 依赖清单
```

## 核心模块详解

### 1. 车端模块 (onboard/)

#### 1.1 Trigger框架 (`onboard/triggers/`)

**核心文件**：
- `trigger_manager.py` - Trigger管理器，统一管理所有Trigger
- `rule_trigger.py` - 规则Trigger，基于阈值和逻辑规则
- `composite_trigger.py` - 组合Trigger，支持多Trigger组合逻辑

**设计特点**：
- 车端/云端/仿真三端代码复用
- Python DSL，易于编写和维护
- 支持动态配置更新
- 自动生成文档（供LLM理解）

**Trigger类型**：
- `RULE` - 规则Trigger（基于阈值和逻辑）
- `MODEL` - 模型Trigger（基于机器学习模型）
- `COMPOSITE` - 组合Trigger（支持AND/OR/NOT逻辑）

**Trigger优先级**：
- `CRITICAL` - 一级指标（严重体感）
- `HIGH` - 二级指标（轻微体感）
- `MEDIUM` - 三级指标（技术指标）
- `LOW` - 其他

**使用示例**：
```python
from onboard.triggers.trigger_manager import get_global_trigger_manager
from onboard.triggers.rule_trigger import create_emergency_brake_trigger

# 获取全局管理器
manager = get_global_trigger_manager()

# 创建急刹车Trigger
trigger = create_emergency_brake_trigger(threshold=3.0)

# 注册Trigger
manager.register_trigger(trigger)

# 评估Trigger
result = manager.evaluate_trigger(trigger_id, {"acceleration": -3.5})
```

#### 1.2 体感指标监控 (`onboard/metrics/`)

**核心文件**：
- `somatic_metrics.py` - 体感指标监控器

**指标体系**：

**一级指标（严重体感）**：
- `emergency_brake` - 急刹车（减速度 > 3m/s²）
- `sharp_turn` - 急转弯（横摆角速度 > 15°/s）
- `snake_driving` - 蛇形行驶（横向加速度波动 > 0.5m/s²）
- `frequent_stop` - 频繁启停（1分钟内 > 3次）
- `mysterious_slow` - 莫名慢行（速度 < 5km/h，非拥堵）

**二级指标（轻微体感）**：
- `late_arrival` - 晚点（偏离计划时间 > 30s）
- `path_deviation` - 路径偏离（横向偏移 > 0.5m）
- `hesitant_lane_change` - 换道犹豫（换道准备时间 > 5s）

**三级指标（技术指标）**：
- `perception_recall` - 感知召回率
- `prediction_accuracy` - 预测准确率
- `planning_success_rate` - 规划成功率
- `control_tracking_error` - 控制跟踪误差

**使用示例**：
```python
from onboard.metrics.somatic_metrics import get_global_monitor

# 获取全局监控器
monitor = get_global_monitor()

# 更新指标值
monitor.update_metric("emergency_brake", -3.5, timestamp=time.time())

# 获取超阈值指标
exceeded = monitor.get_exceeded_metrics()

# 获取统计信息
stats = monitor.get_metric_statistics("emergency_brake")
```

#### 1.3 数据采集与日志 (`onboard/logger/`)

**核心文件**：
- `data_logger.py` - 数据日志器

**三级日志策略**：

| 日志类型 | 触发条件 | 数据内容 | 上传时机 |
|---------|---------|---------|---------|
| **micro log** | 一级指标触发 | 前5s+后5s传感器数据、状态机、控制指令 | 实时上传 |
| **mini log** | 云端确认后 | 前30s+后30s完整数据、多传感器融合结果 | 按需上传 |
| **full log** | 人工标注需求 | 全程数据、调试信息、中间结果 | 离线拷贝 |

**使用示例**：
```python
from onboard.logger.data_logger import get_global_logger

# 获取全局日志器
logger = get_global_logger()

# 创建micro log（一级指标触发）
log_entry = logger.create_micro_log(
    trigger_id="emergency_brake",
    timestamp=time.time(),
    sensor_data=sensor_data,
    state_machine=state_machine,
    control_commands=control_commands
)

# 启动上传工作线程
def upload_callback(task):
    # 实现上传逻辑
    return True

logger.start_upload_worker(upload_callback)
```

#### 1.4 沙箱挖数 (`onboard/sandbox/`)

**核心设计**：
- 解耦原则：挖数Trigger ≠ 线上算法版本
- 配置下发：云端动态下发挖数配置（JSON格式）
- 沙箱执行：独立进程，不干扰主算法
- 场景限定：仅在非自动驾驶任务时段执行

### 2. 云端模块 (cloud/)

#### 2.1 多模态索引系统 (`cloud/mining/`)

**核心文件**：
- `auto_labeling.py` - 自动化结构化提取（基础层）
- `vector_indexing.py` - 向量化检索（进阶层）
- `scene_graph.py` - 场景图构建（高级层）
- `scene_reconstruction.py` - 4D重建（终极层）
- `fine_grained_schema.py` - 细粒度标注Schema
- `data_miner.py` - 数据挖掘器（集成版）

**四层索引架构**：

**基础层 - 自动化结构化提取**：
- 离线教师模型推理，生成高精度标注
- 关键帧提取、对象检测与跟踪
- 车道线/交通灯识别
- 场景标签生成（天气、光照、道路类型）

**进阶层 - 向量化检索**：
- VLM模型（CLIP/BLIP-2）编码
- 文本检索、图像检索、关键帧检索
- 混合检索（向量+规则）

**高级层 - 场景图构建**：
- 对象关系拓扑结构
- 12种交互关系（遮挡、跟随、超车、切入等）
- 关系模式匹配、子图检索

**终极层 - 4D重建**：
- NeRF / 3D Gaussian Splatting
- 场景编辑（添加/删除/修改对象）
- 天气修改、数据增强

**细粒度标注支持**：
- **50+物体类型**：车辆（15+）、VRU（10+）、障碍物（10+）、动物（6+）
- **关键物体属性**：车灯状态（9种）、车门状态（4种）、行人姿态（12种）、遮挡等级（5种）
- **静态道路要素**：地面标识（10种）、红绿灯（4种）、路面特征（12种）
- **场景与环境标签**：天气（13种）、光照（10种）、道路类型（15种）、传感器脏污（9种）
- **逻辑与行为标签**：自车关系（12种）、交通流（6种）、违规行为（8种）
- **Occupancy Flow**：体素网格 + 运动矢量

**混合挖掘策略**：
- `TRIGGER_BASED` - 基于Trigger规则挖掘
- `VECTOR_SEARCH` - 基于向量语义检索
- `RULE_FILTER` - 基于规则过滤
- `EVENT_SEQUENCE` - 基于事件序列模式
- `HYBRID` - 混合策略（推荐）

**使用示例**：
```python
from cloud.mining.data_miner import DataMiner, MiningConfig, MiningStrategy

# 创建数据挖掘器
miner = DataMiner(config=MiningConfig())

# 完整多模态Pipeline
result = miner.process_video_clip(
    video_path="/data/video.mp4",
    enable_auto_labeling=True,
    enable_vector_indexing=True,
    enable_scene_graph=True
)

# 文本语义检索
results = miner.search_by_text("救护车在雨天闯红灯")

# 混合检索
results = miner.hybrid_search(
    text_query="CIPV车辆切入",
    rules={"weather": "rain", "road_type": "urban_road"}
)
```

#### 2.2 LLM问题诊断 (`cloud/diagnosis/`)

**核心文件**：
- `llm_diagnosis.py` - LLM问题诊断器

**功能特性**：
- 时序事件序列建模
- 弱监督在线学习
- 批量诊断、统计跟踪

#### 2.3 问题分发 (`cloud/distribution/`)

**核心文件**：
- `problem_distributor.py` - 问题分发器

**功能特性**：
- 自动分发机制
- 团队映射、优先级排序
- 质量评估体系

#### 2.4 数据标注 (`cloud/annotation/`)

**核心文件**：
- `data_annotation.py` - 数据标注器

**两类标签体系**：
- **挖掘标签** - 数据筛选、场景分类（高频更新，天级）
- **真值标签** - 模型训练、评测验证（低频更新，周级）

### 3. 仿真模块 (simulation/)

**功能模块**：
- `generation/` - 场景生成，基于真实数据生成仿真场景
- `training/` - 模型训练，支持真实数据+生成式数据
- `evaluation/` - 评测验证，仅使用真实数据评测

**生成式数据应用策略**：
- **训练侧**：真实数据 + 生成式数据 → 提升长尾场景召回
- **评测侧**：仅使用真实数据 → 真实世界兜底，不争真值

### 4. 公共模块 (common/)

- `models/` - 数据模型，包含Trigger基类定义
- `utils/` - 工具函数
- `events/` - 事件定义

## 构建和运行

### 安装依赖

```bash
pip install -r requirements.txt
```

**主要依赖**：
- 核心依赖：numpy>=1.21.0, pandas>=1.3.0, pydantic>=2.0.0
- 数据处理：pyarrow>=12.0.0, parquet>=1.3.0
- 机器学习：scikit-learn>=1.3.0, torch>=2.0.0, transformers>=4.30.0
- 向量检索：faiss-cpu>=1.7.4, sentence-transformers>=2.2.0
- LLM相关：openai>=1.0.0, langchain>=0.0.300
- 仿真相关：carla>=0.9.15, gymnasium>=0.29.0

### 运行示例

```bash
# 车端Trigger示例
python examples/onboard_trigger_example.py

# 云端数据挖掘示例
python examples/cloud_mining_example.py

# LLM问题诊断示例
python examples/llm_diagnosis_example.py

# 多模态数据挖掘示例
python examples/multimodal_mining_example.py
```

**注意**：目前 `examples/` 目录为空，示例代码需要自行创建。

### 测试

```bash
# 运行所有测试
pytest

# 运行特定测试文件
pytest tests/test_trigger_manager.py

# 运行测试并生成覆盖率报告
pytest --cov=onboard --cov=cloud --cov-report=html
```

### 代码质量检查

```bash
# 代码格式化
black onboard/ cloud/ common/ simulation/

# 代码风格检查
flake8 onboard/ cloud/ common/ simulation/

# 类型检查
mypy onboard/ cloud/ common/ simulation/
```

## 开发规范

### 代码风格

- 使用 Black 进行代码格式化
- 使用 Flake8 进行代码风格检查
- 使用 MyPy 进行类型检查

### Trigger开发规范

1. **继承BaseTrigger**：所有Trigger必须继承 `BaseTrigger` 类
2. **实现evaluate方法**：必须实现 `evaluate` 方法，返回 `TriggerResult`
3. **配置管理**：使用 `TriggerConfig` 管理配置
4. **元数据文档**：实现 `TriggerMetadata` 生成文档，供LLM理解

**Trigger开发模板**：
```python
from common.models.trigger_base import BaseTrigger, TriggerConfig, TriggerResult, TriggerType
import time

class CustomTriggerConfig(TriggerConfig):
    """自定义Trigger配置"""
    trigger_type: TriggerType = TriggerType.MODEL  # 或 RULE, COMPOSITE

class CustomTrigger(BaseTrigger):
    """自定义Trigger"""
    
    def __init__(self, config: CustomTriggerConfig):
        super().__init__(config)
        # 初始化自定义逻辑
    
    def evaluate(self, data: Dict[str, Any]) -> TriggerResult:
        """
        评估数据是否触发Trigger
        
        Args:
            data: 输入数据
            
        Returns:
            TriggerResult: 触发结果
        """
        # 实现评估逻辑
        triggered = False
        reason = ""
        confidence = 1.0
        
        return TriggerResult(
            triggered=triggered,
            trigger_id=self.config.trigger_id,
            timestamp=time.time(),
            confidence=confidence,
            reason=reason,
            data={}
        )
```

### 指标开发规范

1. **继承MetricConfig**：使用 `MetricConfig` 定义指标配置
2. **级别分类**：明确指标级别（LEVEL_1/LEVEL_2/LEVEL_3）
3. **阈值设定**：为指标设置合理的阈值
4. **统计更新**：使用 `MetricStatistics` 跟踪指标统计

### 日志开发规范

1. **选择日志级别**：根据触发条件选择合适的日志级别（MICRO/MINI/FULL）
2. **数据窗口**：合理设置数据窗口大小
3. **上传策略**：micro log实时上传，mini log按需上传，full log离线拷贝

### 测试规范

1. **单元测试**：每个模块都应有对应的单元测试
2. **集成测试**：测试模块间的交互
3. **覆盖率要求**：核心模块覆盖率应达到80%以上

### Git提交规范

使用语义化提交信息：
- `feat:` - 新功能
- `fix:` - Bug修复
- `docs:` - 文档更新
- `refactor:` - 代码重构
- `test:` - 测试相关
- `chore:` - 构建/工具相关

## 核心原则

1. **指标第一公民**：体感指标高于技术指标
2. **数据当产品**：数据质量、成本、效率三管齐下
3. **问题当Bug追**：每个问题都有归属、有追踪、有验证
4. **三端代码复用**：车端、云端、仿真统一Trigger
5. **真值标签隔离**：挖掘标签≠真值标签
6. **生成数据辅助**：训练可用，评测必用真实数据
7. **组织结构适配**：打破部门墙，建立闭环委员会
8. **持续在线学习**：LLM分类弱监督，持续优化
9. **多模态索引**：不要人工打标签，也不要只存AI的文本总结
10. **细粒度标注**：支持Urban NOA/L3/L4场景的高精度自动标注

## 实施路线

- **阶段1**：基础建设（1-3个月）✅
  - 搭建车端Trigger框架
  - 建立体感指标体系
  - 实现三级日志策略
  - 搭建基础数据平台

- **阶段2**：多模态索引系统（3-6个月）✅
  - 实现基础层：自动化结构化提取
  - 实现进阶层：向量化检索
  - 实现高级层：场景图构建
  - 实现终极层：4D重建
  - 实现细粒度标注系统
  - 集成混合挖掘策略

- **阶段3**：闭环打通（6-9个月）✅
  - 实现云端数据挖掘（集成多模态索引）
  - 部署LLM问题诊断
  - 建立自动分发机制
  - 搭建仿真训练平台

- **阶段4**：优化迭代（9-12个月）
  - 完善质量评估体系
  - 优化数据成本控制
  - 建立组织协作机制
  - 持续优化Trigger库
  - 优化多模态索引性能

- **阶段5**：规模化复制（12个月+）
  - 平台产品化
  - 跨场景复用
  - 生态建设
  - 持续演进
  - 跨场景复用
  - 生态建设
  - 持续演进

## 关键文件索引

### 车端模块
- `onboard/triggers/trigger_manager.py` - Trigger管理器
- `onboard/triggers/rule_trigger.py` - 规则Trigger
- `onboard/triggers/composite_trigger.py` - 组合Trigger
- `onboard/metrics/somatic_metrics.py` - 体感指标监控
- `onboard/logger/data_logger.py` - 数据采集与日志
- `onboard/sandbox/sandbox_manager.py` - 沙箱挖数机制
- `onboard/calibration/auto_calibration.py` - 自标定引擎
- `onboard/calibration/vanishing_point.py` - 消失点检测
- `onboard/calibration/epipolar_constraint.py` - 极线约束
- `onboard/calibration/ego_motion_estimator.py` - 运动估计
- `onboard/calibration/virtual_camera.py` - 虚拟相机
- `onboard/calibration/calibration_manager.py` - 标定管理器
- `onboard/calibration/anomaly_handler.py` - 异常处理

### 云端模块
- `cloud/mining/data_miner.py` - 数据挖掘器（集成版）
- `cloud/mining/auto_labeling.py` - 自动化结构化提取
- `cloud/mining/vector_indexing.py` - 向量化检索
- `cloud/mining/scene_graph.py` - 场景图构建
- `cloud/mining/scene_reconstruction.py` - 4D重建
- `cloud/mining/fine_grained_schema.py` - 细粒度标注Schema
- `cloud/calibration/calibration_monitor.py` - 标定监控
- `cloud/calibration/calibration_diagnosis.py` - 标定诊断
- `cloud/calibration/batch_analysis.py` - 批量分析
- `cloud/diagnosis/llm_diagnosis.py` - LLM问题诊断
- `cloud/distribution/problem_distributor.py` - 问题分发
- `cloud/annotation/data_annotation.py` - 数据标注

### 仿真模块
- `simulation/generation/scene_generator.py` - 场景生成
- `simulation/generation/scene_reconstruction.py` - 4D重建
- `simulation/training/model_trainer.py` - 模型训练
- `simulation/evaluation/model_evaluator.py` - 评测验证

### 公共模块
- `common/models/trigger_base.py` - Trigger基类定义
- `common/models/calibration.py` - 标定参数数据模型
- `common/events/event_types.py` - 事件定义
- `common/utils/data_utils.py` - 工具函数
- `common/utils/transform_tree.py` - 刚体变换链

### 配置文件
- `config/default_config.yaml` - 默认配置

### 示例代码
- `examples/onboard_trigger_example.py` - 车端Trigger示例
- `examples/cloud_mining_example.py` - 云端数据挖掘示例
- `examples/llm_diagnosis_example.py` - LLM问题诊断示例
- `examples/multimodal_mining_example.py` - 多模态数据挖掘示例
- `examples/calibration_example.py` - 标定系统示例

### 文档
- `README.md` - 项目概述
- `ARCHITECTURE.md` - 详细架构设计文档
- `IFLOW.md` - iFlow上下文文档
- `docs/CALIBRATION.md` - 标定系统文档
- `requirements.txt` - Python依赖

## 常见问题

### Q1: 如何创建新的Trigger？

A: 继承 `BaseTrigger` 类，实现 `evaluate` 方法，使用 `TriggerConfig` 管理配置。参考 `rule_trigger.py` 或 `composite_trigger.py`。

### Q2: 如何添加新的体感指标？

A: 使用 `MetricConfig` 定义指标配置，通过 `SomaticMetricsMonitor.register_metric()` 注册指标。

### Q3: 三级日志策略如何选择？

A: 
- 一级指标触发 → micro log（实时上传）
- 云端确认后 → mini log（按需上传）
- 人工标注需求 → full log（离线拷贝）

### Q4: 如何实现云端数据挖掘？

A: 使用统一的Trigger框架，在云端部署Trigger，筛选符合条件的数据。现已集成多模态索引系统，支持混合挖掘策略。

### Q5: LLM问题诊断如何工作？

A: 将时序事件序列转换为LLM可理解的格式，通过语义对齐和弱监督学习进行问题诊断。

### Q6: 多模态索引系统如何使用？

A: 通过 `DataMiner` 类使用，支持四种索引层级：
- 基础层：`auto_labeling.py` - 自动化结构化提取
- 进阶层：`vector_indexing.py` - 向量化检索
- 高级层：`scene_graph.py` - 场景图构建
- 终极层：`scene_reconstruction.py` - 4D重建

支持混合挖掘策略（HYBRID），结合Trigger规则、向量检索、规则过滤和事件序列。

### Q7: 细粒度标注支持哪些类型？

A: 支持50+物体类型：
- 车辆（15+类型）：轿车、SUV、警车、救护车、消防车等
- VRU（10+类型）：行人、儿童、老人、轮椅、骑行者等
- 障碍物（10+类型）：交通锥、水马、护栏等
- 动物（6种）：狗、猫、牛、马、羊、野生动物

支持关键物体属性：车灯状态（9种）、车门状态（4种）、行人姿态（12种）、遮挡等级（5种）。

支持静态道路要素：地面标识（10种）、红绿灯（4种）、路面特征（12种）。

支持场景与环境标签：天气（13种）、光照（10种）、道路类型（15种）、传感器脏污（9种）。

支持逻辑与行为标签：自车关系（12种）、交通流（6种）、违规行为（8种）。

支持Occupancy Flow：体素网格 + 运动矢量。

### Q8: 如何使用标定系统？

A: 参考 `examples/calibration_example.py` 和 `docs/CALIBRATION.md`。基本流程：
1. 创建Factory Spec（出厂标称参数）
2. 创建CalibrationManager并启动标定
3. 持续更新传感器数据
4. 标定完成后自动上传到云端

### Q9: 标定系统支持哪些传感器？

A: 目前主要支持相机（Camera）标定，包括：
- 前视相机
- 侧视相机（左/右）
- 后视相机

未来可扩展支持IMU、激光雷达、毫米波雷达等传感器。

### Q10: 如何处理标定异常？

A: 系统会自动检测异常并触发重置：
- 软重置：扩大搜索范围，重新收敛
- 硬重置：回到Factory Spec，要求用户重新行驶

可通过AnomalyHandler自定义异常检测和重置策略。

### Q11: 云端如何监控标定状态？

A: 使用CalibrationMonitor接收车端上传的标定数据，支持：
- 全队标定统计
- 批次异常检测
- 异常车辆识别
- 告警生成

### Q12: 标定参数如何存储？

A: 标定参数分为两类：
- Factory Spec：出厂标称参数，存储在配置文件中
- Learned Spec：动态在线参数，可持久化到文件

可通过CalibrationManager的save_calibration_state()和load_calibration_state()方法保存和加载。

## 联系方式

- 项目地址：/home/geely/Documents/Github/auto_calibration
- 文档：README.md, ARCHITECTURE.md
- 贡献指南：欢迎提交Issue和Pull Request

## 许可证

MIT License