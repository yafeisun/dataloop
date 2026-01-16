#!/bin/env python3
import glob
import os
import sys
from typing import Tuple

import click

from src import __app_name__, __version__, StateCode, RunState
from src.utils import log
from src.proj2bev_ import gen_bev
from src import IPMException
from src import CarYamlMissingError
from src.find_images import ImagesTimeAligner, cam_names_dict

from src.bev_plot import BevPlot

CURRENT_SRC_DIR = os.path.dirname(__file__)


class MetaData:
    def __init__(self, path=None):
        if path is None:
            path = os.path.join(CURRENT_SRC_DIR, "metadata.txt")
        info = {}
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines: 
                key, value = line.strip().split("=")
                info[key] = value
        self.app_name = info.get("APP_NAME", "alabel_vis.py")
        self.app_version = info.get("APP_VERSION", "0.0.0")
        self.git_revision = info.get("GIT_REVISION", "")
        self.build_date = info.get("BUILD_DATE", "")
    
    def print_log_header(self):
        click.echo( 
            f'{self.app_name}, version={self.app_version}, commitID={self.git_revision}, buildTime={self.build_date}')
    
    def print_log_tailer(self, code: int, msg: str = "ok"):
        click.echo('==========================================')
        click.echo(f'code:{code}')
        click.echo(f'msg:{msg}')
        click.echo(f'alg_version:{self.app_version}')


def gen_one_group(aligner, cam_param_file, proj_type, output_dir):
    multi_group = aligner.get_multi_group_frames(cam_names_dict[proj_type])
    gen_bev(cam_param_file, multi_group, proj_type, output_dir)


def gen_one_group_new(aligner, cam_param_file, proj_type, output_dir, only_shape):
    bev_plot = BevPlot(cam_param_file, proj_type)
    multi_group = aligner.get_multi_group_frames(cam_names_dict[proj_type])
    for i, grp in enumerate(multi_group):
        bev_img = bev_plot.project_multi_img(grp, only_shape)
        out_path = os.path.join(output_dir, f'bev_{proj_type}_{i + 1}_{grp[0]}.png')
        bev_plot.save_image(bev_img, out_path)

def run_ipm(bag_dir, frm_nums: Tuple[int], output_dir, only_shape, out_view_types=('long', 'around')):
    cam_param_files = glob.glob(os.path.join(bag_dir, 'car_*.yaml'))
    if not cam_param_files:
        raise CarYamlMissingError(f'Not found cam param(car_*.yaml) files found in {bag_dir}')
    cam_param_file = cam_param_files[0]
    aligner = ImagesTimeAligner(bag_dir, frm_nums, out_view_types)
    for t in out_view_types:
        # gen_one_group(aligner, cam_param_file, t, output_dir)
        gen_one_group_new(aligner, cam_param_file, t, output_dir, only_shape)
    return RunState

def parse_value_or_range(input_str):
    if not input_str:
        return []

    try:
        if '-' in input_str:
            # ~D~P~F~L~C~[| ~O~H~B7-9~I
            start, end = map(int, input_str.split('-', 1))
            if start > end:
                raise click.BadParameter(f"range should be from small to big: {input_str}")
            return list(range(start, end + 1))
        else:
            return [int(input_str)]
    except ValueError:
        raise click.BadParameter(f"frm_nums invalid format: {input_str}")

@click.command()
@click.version_option(version=__version__, prog_name=__app_name__)
@click.option('-b', '--bag_dir', type=click.Path(exists=True), required=True, help='bag dir')
@click.option('-f', '--frm_nums', type=click.STRING, multiple=True, help='set frame numbers to generate bev, format:1 or 2-5')
@click.option('-o', '--output_dir', type=click.Path(), required=True, help='output dir')
@click.option('-s', '--only_shape', is_flag=True, default=False, help='only check image shape')
@click.option('-t', '--view-types', multiple=True, type=click.Choice(('long', 'around')), default=('long', 'around'), help='set out view types')
@click.help_option('-h', '--help')
def run(bag_dir, frm_nums, output_dir, only_shape, view_types):
    meta = MetaData()
    meta.print_log_header()
    os.makedirs(output_dir, exist_ok=True)
    status = RunState()
    frm_num_list = []
    for val in frm_nums:
        frm_num_list.extend(parse_value_or_range(val))
    try:
        status = run_ipm(bag_dir, frm_num_list, output_dir, only_shape, out_view_types=view_types)
    except IPMException as e:
        status.code = e.code
        status.msg = str(e)
    except Exception as e:
        status.code = StateCode.error_unknown
        status.msg = e
        import traceback
        log.exception(traceback.format_exc())

    meta.print_log_tailer(status.code.value, status.msg)
    sys.exit(status.code.value)

if __name__ == '__main__':
    run()
  
