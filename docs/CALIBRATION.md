# 标定系统文档

## 概述

本系统实现了特斯拉风格的端云协同动态标定机制，核心设计哲学是："弱硬件依赖，强软件补偿"。

### 核心特性

1. **端云分工**
   - 车端：实时计算和修正
   - 云端：统计监控和异常诊断

2. **参数分层**
   - Factory Spec（出厂/标称参数）：算法启动的初始猜测值
   - Learned Spec（动态/在线参数）：实际用于模型推理和坐标变换

3. **标定流程闭环**
   - 启动阶段：加载Learned Spec，回退到Factory Spec
   - 行驶阶段：自标定持续运行
   - 收敛：偏差小于阈值时完成标定
   - 更新：实时修正外参
   - 上传：将Learned Spec作为元数据上传到云端

## 架构设计

### 车端模块 (`onboard/calibration/`)

#### 1. 自标定引擎 (`auto_calibration.py`)
整合消失点、极线约束和运动估计，实现车端在线自标定。

#### 2. 消失点检测 (`vanishing_point.py`)
基于消失点原理标定相机俯仰角和偏航角。

#### 3. 极线约束 (`epipolar_constraint.py`)
基于极线几何标定相邻相机的相对位置。

#### 4. 运动估计 (`ego_motion_estimator.py`)
基于自车运动估计相机高度和角度。

#### 5. 虚拟相机 (`virtual_camera.py`)
将原始图像转换为标准虚拟相机视角。

#### 6. 标定管理器 (`calibration_manager.py`)
实现标定流程闭环。

#### 7. 异常处理 (`anomaly_handler.py`)
实现异常检测和自动重置策略。

### 云端模块 (`cloud/calibration/`)

#### 1. 标定监控 (`calibration_monitor.py`)
监控全队车辆的标定状态，识别硬件质量问题。

#### 2. 标定诊断 (`calibration_diagnosis.py`)
基于LLM进行问题诊断。

#### 3. 批量分析 (`batch_analysis.py`)
对全队车辆进行批量分析和统计。

### 公共模块 (`common/`)

#### 1. 标定数据模型 (`models/calibration.py`)
定义标定参数的数据模型。

#### 2. 变换树 (`utils/transform_tree.py`)
维护车辆坐标系树，支持任意坐标系之间的变换。

## 使用示例

### 车端标定

```python
from onboard.calibration import CalibrationManager
from common.models.calibration import FactorySpec, CameraIntrinsics, CameraExtrinsics, SensorType

# 1. 创建出厂标称参数
factory_specs = {
    "front_camera": FactorySpec(
        sensor_id="front_camera",
        sensor_type=SensorType.CAMERA,
        intrinsics=CameraIntrinsics(fx=1000.0, fy=1000.0, cx=960.0, cy=540.0),
        extrinsics=CameraExtrinsics(
            translation=(2.0, 0.0, 1.5),
            rotation=(0.0, 0.0, 0.0)
        )
    )
}

# 2. 创建标定管理器
calibration_manager = CalibrationManager(
    vehicle_id="test_vehicle_001",
    factory_specs=factory_specs
)

# 3. 设置上传回调
def upload_callback(metadata):
    print(f"Uploading calibration data: {metadata}")
    return True

calibration_manager.set_upload_callback(upload_callback)

# 4. 启动标定
calibration_manager.boot_up()
calibration_manager.start_calibration()

# 5. 更新标定
sensor_data = {
    "front_camera": {
        "lane_lines": [(100, 540, 960, 540), (960, 540, 1820, 540)],
        "image": image,
        "keypoints": keypoints
    }
}

results = calibration_manager.update(
    sensor_data=sensor_data,
    wheel_speed=10.0,
    driving_distance=100.0
)

# 6. 获取标定进度
progress_list = calibration_manager.get_calibration_progress()
for progress in progress_list:
    print(f"{progress.sensor_id}: {progress.progress:.2%}")
```

### 云端监控

```python
from cloud.calibration import CalibrationMonitor

# 1. 创建监控器
monitor = CalibrationMonitor()

# 2. 注册车辆批次
monitor.register_vehicle_batch("test_vehicle_001", "batch_2024_001")

# 3. 接收标定数据
metadata = {
    "vehicle_id": "test_vehicle_001",
    "overall_status": "converged",
    "overall_convergence": 0.95,
    "health_score": 0.92,
    "sensor_specs": {...}
}

monitor.receive_calibration_data(metadata)

# 4. 计算统计信息
statistics = monitor.compute_statistics()

# 5. 检测批次异常
anomalies = monitor.detect_batch_anomalies()
```

### 异常处理

```python
from onboard.calibration import AnomalyHandler

# 1. 创建异常处理器
anomaly_handler = AnomalyHandler(
    soft_reset_threshold=0.05,
    hard_reset_threshold=0.1
)

# 2. 设置重置回调
def reset_callback(sensor_id, reset_type, details):
    print(f"Reset: {sensor_id}, {reset_type}")
    return True

anomaly_handler.set_reset_callback(reset_callback)

# 3. 检测异常
anomalies = anomaly_handler.detect_anomalies(learned_specs)

# 4. 处理异常
reset_events = anomaly_handler.handle_anomalies(anomalies)
```

## 运行示例

```bash
# 运行标定示例
python examples/calibration_example.py
```

## 关键技术点

### 1. 消失点检测
- 原理：车道线在远处的交点（消失点）应该对应相机的光心方向
- 应用：标定俯仰角和偏航角

### 2. 极线约束
- 原理：同一个物体在两个摄像头中的投影关系必须满足"基础矩阵"
- 应用：标定相邻相机的相对位置

### 3. 运动估计
- 原理：结合轮速计和IMU，对比图像中特征点的移动距离
- 应用：估计相机高度

### 4. 虚拟相机
- 原理：原始图像 → 去畸变 + 旋转平移 → 虚拟相机图像
- 应用：模型永远看到"完美的、标准的"视角

### 5. 刚体变换链
- 原理：维护一套坐标系树（Body → Camera/IMU/LIDAR）
- 应用：支持任意坐标系之间的变换

## 异常处理策略

### 动态检测
- 地平面严重倾斜
- 静止物体在3D空间中跳动
- 偏差过大
- 收敛失败
- 传感器断开

### 重置策略
- **软重置**：扩大搜索范围，重新收敛
- **硬重置**：回到Factory Spec，要求用户重新行驶

## 云端作用

### 1. 离线大模型训练
云端Teacher Model重建场景时，必须使用车端当时时刻的标定参数。

### 2. 全队监控
监控某一批次的车辆是否存在普遍的摄像头安装角度偏差（硬件质量问题）。

### 3. 异常诊断
基于LLM进行问题诊断，提供修复建议。

## 数据流

```
车端：
传感器数据 → 自标定引擎 → Learned Spec → 虚拟相机 → AI模型
                                           ↓
                                        上传到云端

云端：
接收数据 → 标定监控 → 批次分析 → 异常诊断 → 告警/修复建议
```

## 配置参数

### 标定配置 (`CalibrationConfig`)
- `convergence_threshold`: 收敛阈值（默认0.9）
- `max_iterations`: 最大迭代次数（默认1000）
- `min_driving_distance`: 最小行驶距离（默认100米）
- `soft_reset_threshold`: 软重置阈值（默认0.05）
- `hard_reset_threshold`: 硬重置阈值（默认0.1）
- `upload_interval`: 上传间隔（默认3600秒）

### 虚拟相机配置 (`VirtualCameraConfig`)
- `image_size`: 图像尺寸（默认1920x1080）
- `fov`: 视场角（默认60度）
- `distortion_free`: 是否无畸变（默认True）

## 扩展开发

### 添加新的传感器类型

1. 在`SensorType`枚举中添加新类型
2. 在`FactorySpec`和`LearnedSpec`中添加相应的内参定义
3. 在`AutoCalibrationEngine`中实现新的标定方法

### 添加新的异常检测方法

1. 在`AnomalyType`枚举中添加新类型
2. 在`AnomalyHandler`中实现检测逻辑
3. 在`handle_anomalies`中添加处理策略

## 参考资料

- 特斯拉自标定技术文档
- 多视图几何（Hartley & Zisserman）
- 计算机视觉：算法与应用（Szeliski）

## 许可证

MIT License