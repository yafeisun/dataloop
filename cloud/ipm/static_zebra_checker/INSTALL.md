# 斑马线检测和 IPM 图像生成

## 项目简介

本项目提供静止时段斑马线检测和 IPM（逆透视映射）图像生成功能，支持 ROS1 和 ROS2 bag 文件。

## 功能特性

- ✅ 静止时段检测（基于 GPS 位移）
- ✅ 斑马线检测（Sobel 边缘检测）
- ✅ 一致性检查（智能采样策略）
- ✅ IPM 图像生成
- ✅ 相机尺寸检查和修正
- ✅ ROS1/ROS2 兼容

## 安装依赖

### 方法1：使用 pip 安装（推荐）

```bash
pip install -r requirements.txt
```

### 方法2：使用虚拟环境

```bash
# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 依赖说明

| 包名 | 版本要求 | 用途 |
|------|---------|------|
| rosbags | >=0.9.20 | 读取 ROS bag 文件 |
| opencv-python | >=4.8.0 | 图像处理和斑马线检测 |
| numpy | >=1.24.0 | 数值计算 |
| Pillow | >=10.0.0 | 图像处理 |
| scipy | >=1.10.0 | 空间变换和旋转计算 |
| PyYAML | >=6.0 | 配置文件解析 |
| click | >=8.0.0 | 命令行界面 |
| prettytable | >=3.8.0 | 表格显示 |

### 可选依赖

- **tqdm** (>=4.65.0): 显示进度条

```bash
pip install tqdm>=4.65.0
```

## 验证安装

运行环境检查脚本：

```bash
python3 static_zebra_checker/check_env.py
```

或手动验证：

```bash
python3 -c "import rosbags, cv2, numpy, PIL, scipy, yaml, click, prettytable; print('所有依赖安装成功')"
```

## 使用方法

### 1. 斑马线检测

```python
from static_zebra_checker import analyze_single_bag, load_config, extract_zebra_frame_numbers

# 加载配置
config = load_config('static_zebra_checker/config.yaml')

# 分析 bag 文件
result = analyze_single_bag('path/to/bag_dir', config)

# 提取斑马线时段的图片序号
frame_numbers = extract_zebra_frame_numbers(result, 'path/to/camera_dir')
```

### 2. IPM 图像生成

```python
from src.cli import run_ipm

# 生成 IPM 图像
bag_path = 'path/to/bag.bag'
frame_numbers = (1, 2, 3)  # 要处理的帧序号
output_dir = 'path/to/output'
run_ipm(bag_path, frame_numbers, output_dir, only_shape=False)
```

### 3. 相机尺寸检查和修正

```python
from static_zebra_checker.camera_size_checker import check_and_fix_camera_sizes

# 检查单个数据包
bag_dir = '/path/to/bag_dir'

# 只检查不修改（dry_run=True）
result = check_and_fix_camera_sizes(bag_dir, dry_run=True)

# 如果有不匹配的，修正配置
if result.get('mismatched'):
    result = check_and_fix_camera_sizes(bag_dir, dry_run=False)
```

## ROS1/ROS2 兼容性

### 兼容性说明

本代码使用 `rosbags` 库，已实现 ROS1 和 ROS2 完全兼容：

- ✅ **纯 Python 库**：无需 ROS 运行时
- ✅ **自动格式识别**：支持 ROS1 (`.bag`) 和 ROS2 (`.db3`)
- ✅ **代码无需修改**：同一套代码在两种环境运行

### 部署步骤

#### ROS1 机器上部署

```bash
# 1. 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 运行代码
python3 test.py
```

#### ROS2 机器上部署

```bash
# 1. 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 运行代码
python3 test.py
```

### 环境变量配置

```bash
# .bashrc 或 .zshrc
export PYTHONPATH=/path/to/project:$PYTHONPATH
```

## 配置说明

### config.yaml

```yaml
# 静止检测配置
static_detection:
  max_displacement: 0.5      # 最大位移（米）
  window_size: 3             # 窗口大小（帧）

# 斑马线检测配置
zebra_detection:
  sobel_threshold: 50        # Sobel 阈值
  min_lines: 3               # 最少线条数
  line_spacing: (20, 80)     # 线条间距范围（像素）
  aspect_ratio: (0.1, 0.5)   # 宽高比范围

# 一致性检查配置
consistency_check:
  consistency_threshold: 0.7  # 一致性阈值（70%）

  sampling:
    # 小片段（<= 5帧）
    small_frame_threshold: 5
    small_frame_sample_all: true

    # 中等片段（6-30帧）
    medium_frame_threshold: 30
    medium_frame_sample_ratio: 0.33
    medium_frame_min_samples: 5

    # 大片段（> 30帧）
    large_frame_max_samples: 15
    large_frame_sample_ratio: 0.5

# 相机配置
camera:
  camera_dir: 'cam_front_right'  # 相机文件夹名
```

## 相机尺寸检查工具

### 功能说明

`camera_size_checker.py` 工具用于检查和修正 `car_*.yaml` 配置文件中的相机 `ImageSize` 配置，使其与实际图片尺寸一致。

### 主要功能

1. **自动检测**：扫描所有相机文件夹，获取图片的实际尺寸
2. **比对配置**：读取 `car_*.yaml` 文件中的 `ImageSize` 配置
3. **自动修正**：发现不匹配时，自动更新配置文件
4. **批量处理**：支持检查和修正多个数据包

### 使用方法

```bash
# 直接运行脚本
python3 static_zebra_checker/camera_size_checker.py

# 或在代码中调用
from static_zebra_checker.camera_size_checker import check_and_fix_camera_sizes

result = check_and_fix_camera_sizes('/path/to/bag_dir', dry_run=False)
```

### 相机文件夹映射

工具会自动识别以下11个相机文件夹：

| 文件夹名 | Topic |
|---------|-------|
| cam_around_back | /cam_around_back/compressed |
| cam_around_front | /cam_around_front/compressed |
| cam_around_left | /cam_around_left/compressed |
| cam_around_right | /cam_around_right/compressed |
| cam_back | /cam_back/compressed |
| cam_front_left | /cam_front_left/compressed |
| cam_front_right | /cam_front_right/compressed |
| cam_side_left_back | /cam_side_left_back/compressed |
| cam_side_left_front | /cam_side_left_front/compressed |
| cam_side_right_back | /cam_side_right_back/compressed |
| cam_side_right_front | /cam_side_right_front/compressed |

## 故障排查

### 问题1：依赖安装失败

```bash
# 升级 pip
pip install --upgrade pip

# 使用国内镜像源
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 问题2：rosbags 导入错误

```bash
# 升级 rosbags
pip install --upgrade rosbags

# 检查版本
pip show rosbags
```

### 问题3：bag 文件读取失败

```bash
# 检查 bag 文件格式
# ROS1: .bag
# ROS2: .db3

# 运行兼容性检查
python3 static_zebra_checker/check_env.py
```

### 问题4：相机尺寸不匹配

```bash
# 运行相机尺寸检查
python3 static_zebra_checker/camera_size_checker.py

# 或在代码中调用
from static_zebra_checker.camera_size_checker import check_and_fix_camera_sizes
result = check_and_fix_camera_sizes('/path/to/bag_dir', dry_run=False)
```

## 项目结构

```
static_zebra_checker/
├── __init__.py
├── INSTALL.md
├── config.yaml
├── batch_analyzer.py          # 批量分析
├── camera_size_checker.py     # 相机尺寸检查
├── check_env.py               # 环境检查
├── frame_extractor.py         # 帧提取器
├── camera_processor.py        # 相机处理
├── gps_analyzer.py            # GPS 分析
├── static_detector.py         # 静止检测
├── utils.py                   # 工具函数
└── zebra_detector.py          # 斑马线检测
```

## 注意事项

1. **虚拟环境**：建议使用虚拟环境隔离依赖
2. **备份配置**：修改前建议备份原始的 `car_*.yaml` 文件
3. **图片格式**：工具只支持 `.jpg` 格式的图片
4. **YAML 保留**：修改后会保留 YAML 文件的格式和注释
5. **权限要求**：需要读写 `car_*.yaml` 文件的权限

## 更多信息

- 查看 `config.yaml` 了解详细配置
- 运行 `check_env.py` 检查环境
- 查看 `camera_size_checker.py` 了解相机尺寸检查