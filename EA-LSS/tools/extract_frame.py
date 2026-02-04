import argparse
import os
import multiprocessing as mp
from datautil.CollectDataToPkl import CollectionDataToPkl_bag
from datautil.convert_and_save import process_convert_bag


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('--InputPath', nargs="+", 
                       default="/home/robosense/Documents/yafeisun/0115_test/2025-01-15-17-11-06/clips/2025-01-15-17-11-06", 
                       help='Input path for bag files')
    parser.add_argument('--OutputPath', 
                       default="/home/robosense/Documents/yafeisun/0115_OD/2025-01-15-17-11-06", 
                       help='Output path for processed data')

    args = parser.parse_args()
    return args


def process_bag(NusPath, bagPath):
    """处理单个bag文件"""
    print("processing:", bagPath)
    data_nuscence = CollectionDataToPkl_bag(NusPath, bagPath)
    if data_nuscence is None:
        return
    
    output_bag_path = bagPath + "-clip-20df"
    if not os.path.exists(output_bag_path):
        os.makedirs(output_bag_path)
    
    process_convert_bag(data_nuscence, output_bag_path)


def main():
    """主函数：多进程处理bag文件"""
    args = parse_args()
    if args.InputPath is None:
        return

    NusPath = "./data/mydata/val_14cls_10v_1118_m.pkl"

    # 多进程处理
    with mp.Pool(processes=20) as pool:
        pool.starmap_async(process_bag, [(NusPath, bag_path) for bag_path in args.InputPath])
        pool.close()
        pool.join()


if __name__ == '__main__':
    main()