# 自动驾驶数据闭环系统

基于智驾行业7年实战经验构建的真数据闭环架构，实现"指标当第一公民，数据当产品，问题当Bug追"的完整闭环体系。

## 项目结构

```
auto_calibration/
├── onboard/              # 车端模块
│   ├── triggers/        # Trigger框架（规则/组合/管理器）
│   ├── metrics/         # 体感指标监控（一级/二级/三级）
│   ├── logger/          # 数据采集与日志（micro/mini/full）
│   └── sandbox/         # 沙箱挖数机制
├── cloud/               # 云端模块
│   ├── mining/          # 数据挖掘（多模态索引系统）
│   │   ├── auto_labeling.py      # 自动化结构化提取
│   │   ├── vector_indexing.py    # 向量化检索
│   │   ├── scene_graph.py         # 场景图构建
│   │   ├── fine_grained_schema.py # 细粒度标注Schema
│   │   └── data_miner.py          # 数据挖掘器（集成版）
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

## 核心特性

### 基础特性
1. **统一Trigger框架**：车端/云端/仿真三端代码复用
2. **体感指标监控**：用户体感指标第一公民
3. **三级日志策略**：micro/mini/full分级采集
4. **沙箱挖数机制**：配置下发，动态启停
5. **LLM问题诊断**：时序事件序列建模
6. **自动问题分发**：智能映射团队，优先级排序

### 多模态索引系统（行业领先方案）
7. **基础层 - 自动化结构化提取**：离线大模型生成高精度几何元数据
8. **进阶层 - 向量化检索**：VLM模型支持自然语言语义搜索
9. **高级层 - 场景图构建**：描述对象之间的复杂交互关系
10. **终极层 - 4D重建**：支持场景编辑和数据增强

### 细粒度标注（支持Urban NOA/L3/L4）
11. **细粒度物体分类**：特种车辆、异形大车、VRU细分、小型障碍物、动物
12. **关键物体属性**：车灯状态、车门状态、行人姿态、遮挡等级
13. **静态道路要素**：地面标识、路面特征、红绿灯倒计时
14. **场景与环境标签**：天气、光照、道路类型、传感器脏污
15. **逻辑与行为标签**：自车关系、交通流、违规行为
16. **通用占用网络**：Occupancy Flow支持异形障碍物

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行示例

```bash
# 车端Trigger示例
python examples/onboard_trigger_example.py

# 云端数据挖掘示例
python examples/cloud_mining_example.py

# LLM问题诊断示例
python examples/llm_diagnosis_example.py

# 多模态数据挖掘示例（推荐）
python examples/multimodal_mining_example.py
```

## 架构设计

详细架构设计请参考 [ARCHITECTURE.md](ARCHITECTURE.md)

## 实施路线

- **阶段1**：基础建设（1-3个月）
- **阶段2**：闭环打通（3-6个月）
- **阶段3**：优化迭代（6-12个月）
- **阶段4**：规模化复制（12个月+）

## 核心原则

1. 指标第一公民：体感指标高于技术指标
2. 数据当产品：数据质量、成本、效率三管齐下
3. 问题当Bug追：每个问题都有归属、有追踪、有验证
4. 三端代码复用：车端、云端、仿真统一Trigger
5. 真值标签隔离：挖掘标签≠真值标签
6. 生成数据辅助：训练可用，评测必用真实数据
7. 组织结构适配：打破部门墙，建立闭环委员会
8. 持续在线学习：LLM分类弱监督，持续优化

## 贡献指南

欢迎提交Issue和Pull Request！

## 许可证

MIT License