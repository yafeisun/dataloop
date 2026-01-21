from src.cli import run_ipm
from src.zebra_detect import check_for_zebra_crossing
import os

if __name__ == '__main__':
    bag_path = 'd:\\S23_0114_104\\sta_dyna_sta\\2026-01-14-13-01-52'
    output_dir = os.path.join(bag_path, 'ipm')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    frm_nums = [1, 2, 3] # frame numbers to process
    only_shape = False # only process shape, not acc
    run_ipm(bag_path, tuple(frm_nums), output_dir, only_shape)
