#!/usr/bin/env python3
"""
环境检查脚本
检查Python环境、依赖包和模块导入是否满足运行要求
"""

import sys
from pathlib import Path


def check_dependencies():
    """检查所有依赖是否已安装"""
    dependencies = {
        'rosbags': '>=0.9.20',
        'cv2': '>=4.8.0',
        'numpy': '>=1.24.0',
        'PIL': '>=10.0.0',
        'scipy': '>=1.10.0',
        'yaml': '>=6.0',
        'click': '>=8.0.0',
        'prettytable': '>=3.8.0',
    }

    missing = []
    installed = []

    print("\n检查依赖包...")
    print("-" * 70)

    for package, version in dependencies.items():
        try:
            module = __import__(package)
            installed.append(package)
            version_str = getattr(module, '__version__', 'N/A')
            print(f"✓ {package:<20} {version_str:<15} (要求: {version})")
        except ImportError:
            missing.append(package)
            print(f"✗ {package:<20} {'未安装':<15} (要求: {version})")

    print("-" * 70)
    print(f"结果: 已安装 {len(installed)}/{len(dependencies)} 个依赖包")

    if missing:
        print(f"\n缺少 {len(missing)} 个依赖包:")
        for pkg in missing:
            print(f"  - {pkg}")
        print(f"\n请运行: pip install -r requirements.txt")
        return False

    return True


def check_bag_files():
    """检查 bag 文件"""
    test_bag_dir = Path("/home/geely/Downloads/S39_0117")

    if not test_bag_dir.exists():
        print("\n⚠ 警告: 测试路径不存在")
        return True

    bag_files = list(test_bag_dir.rglob("*.bag"))
    db3_files = list(test_bag_dir.rglob("*.db3"))

    print("\n检查 bag 文件...")
    print("-" * 70)

    if bag_files:
        print(f"✓ 找到 {len(bag_files)} 个 ROS1 bag 文件 (.bag)")
        for bag_file in sorted(bag_files)[:3]:
            print(f"  - {bag_file.name}")

    if db3_files:
        print(f"✓ 找到 {len(db3_files)} 个 ROS2 bag 文件 (.db3)")
        for db3_file in sorted(db3_files)[:3]:
            print(f"  - {db3_file.name}")

    if not bag_files and not db3_files:
        print("⚠ 警告: 未找到 bag 文件")

    return True


def check_module_imports():
    """检查关键模块导入"""
    print("\n检查模块导入...")
    print("-" * 70)

    # 检查第三方库
    test_imports = [
        ('rosbags.highlevel', 'AnyReader'),
    ]

    import_failed = []
    for module_path, module_name in test_imports:
        try:
            parts = module_path.split('.')
            __import__(module_path)
            print(f"✓ {module_name:<40}")
        except ImportError as e:
            print(f"✗ {module_name:<40} - {e}")
            import_failed.append(module_name)

    # 检查本地模块
    local_modules = [
        'batch_analyzer',
        'camera_processor',
        'gps_analyzer',
        'static_detector',
        'zebra_detector',
        'utils',
    ]

    # 添加父目录到 Python 路径
    parent_dir = Path(__file__).parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

    for module_name in local_modules:
        try:
            # 使用包导入
            module = __import__(f'static_zebra_checker.{module_name}', fromlist=[''])
            print(f"✓ {module_name:<40}")
        except ImportError as e:
            print(f"✗ {module_name:<40} - {e}")
            import_failed.append(module_name)

    return len(import_failed) == 0


def check_compatibility():
    """检查ROS环境兼容性"""
    print("=" * 70)
    print("环境检查")
    print("=" * 70)

    print(f"\nPython版本: {sys.version}")
    print(f"Python路径: {sys.executable}")

    # 检查依赖
    deps_ok = check_dependencies()
    if not deps_ok:
        return False

    # 检查 bag 文件
    check_bag_files()

    # 检查模块导入
    imports_ok = check_module_imports()

    print("\n" + "=" * 70)

    if not imports_ok:
        print("✗ 模块导入失败")
        return False

    print("✓ 环境检查通过！")
    print("=" * 70)
    print("\n提示:")
    print("  - 本代码使用 rosbags 库，无需 ROS 运行时")
    print("  - 自动支持 ROS1 (.bag) 和 ROS2 (.db3) 格式")
    print("  - 建议使用虚拟环境隔离依赖")
    print("=" * 70)

    return True


if __name__ == '__main__':
    success = check_compatibility()
    sys.exit(0 if success else 1)