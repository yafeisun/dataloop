import os
from pathlib import Path
from src.cli import run_ipm
from static_zebra_checker.batch_analyzer import analyze_single_bag, load_config, extract_zebra_frame_numbers
from static_zebra_checker.camera_size_checker import check_and_fix_camera_sizes

def gen_static_ipm(bag_dir):
    """
    生成静止斑马线时段的 IPM 图像
    
    Args:
        bag_dir: bag 文件所在目录
    """
    # 加载配置
    config_path = Path(__file__).parent / 'static_zebra_checker' / 'config.yaml'
    config = load_config(str(config_path))
    
    # 分析 bag 文件
    print(f"\n{'='*80}")
    print(f"开始分析: {bag_dir}")
    print(f"{'='*80}")
    result = analyze_single_bag(bag_dir, config)
    
    # 提取有斑马线的静止时间段的图片序号
    camera_dir = Path(bag_dir) / config.get('camera', {}).get('camera_dir', 'cam_front_right')
    frm_nums = extract_zebra_frame_numbers(result, str(camera_dir))
    
    if not frm_nums:
        print("\n未找到有斑马线的静止时段")
        return
    
    print(f"\n总共选择了 {len(frm_nums)} 张图片用于生成 IPM")
    print(f"图片序号: {frm_nums}")
    
    result = check_and_fix_camera_sizes(bag_dir, dry_run=False)

    # 生成 IPM 图像
    bag_path = os.path.join(bag_dir, f"{Path(bag_dir).name}.bag")
    output_dir = os.path.join(bag_dir, 'ipm')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    only_shape = False  # 处理形状和加速度
    run_ipm(bag_dir, tuple(frm_nums), output_dir, only_shape)
    
    print(f"\nIPM 图像已生成到: {output_dir}")


if __name__ == '__main__':
    # 测试数据包路径
    bag_path = '/home/geely/Downloads/S39_0117/2026-01-17-16-37-58'
    gen_static_ipm(bag_path)
