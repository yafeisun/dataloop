import os
import random
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from typing import Tuple, List

import numpy as np
from src import FrameAlignError, ViewMissingError
from src.acc_reader import CanCsvReader
from src.utils import log

cam_names_dict = {
    'around': ['cam_around_back', 'cam_around_front', 'cam_around_left', 'cam_around_right',
               'cam_front_left'],
    'side': ['cam_side_left_front', 'cam_side_left_back', 'cam_side_right_front',
             'cam_side_right_back', 'cam_back', 'cam_front_right'],
    'single': ['cam_side_left_front', 'cam_side_left_back', 'cam_side_right_front',
               'cam_side_right_back', 'cam_back', 'cam_front_right', 'cam_front_left',
               'cam_around_back', 'cam_around_front', 'cam_around_left', 'cam_around_right'],
    'long': ['cam_side_left_front', 'cam_side_left_back', 'cam_side_right_front',
             'cam_side_right_back', 'cam_back', 'cam_front_right', 'cam_front_left']
}


class ImagesTimeAligner:
    max_time_diff = 50 * 1E6  # 50 ms
    cam_tss = {}  # {cam: [ts]}
    max_acc = (0.2, 0.5)

    def __init__(self, bag_dir, frm_nums: Tuple[int], view_types: Tuple[str], suffix='.jpg'):
        self.bag_dir = bag_dir
        self.suffix = suffix
        self.total_frames = 0
        self.groups_cnt = 3  # ~T~_~H~P3~D
        need_cam_names = []
        self.ref_tss = None
        for t in view_types:
            need_cam_names.extend(cam_names_dict[t])
        need_cam_names = set(need_cam_names)
        log.info(f'expected existing {len(need_cam_names)} cams: {need_cam_names}...')
        for cam_name in need_cam_names:
            tss = self.load_sorted_images_list(cam_name)
            self.total_frames = len(tss) if self.total_frames == 0 else min(self.total_frames, len(tss))
            if not self.ref_tss:
                self.ref_tss = tss

        frm_cnt = self.total_frames // self.groups_cnt

        can_path = os.path.join(bag_dir, 'can2csvOutput.csv')
        self.acc_reader = CanCsvReader(can_path)

        self.selected_frame_num_acc = [self.get_right_num((num, num + 1), False, False) for num in frm_nums]

        if frm_nums:
            log.info(
                f'input frames: {self.selected_frame_num_acc} to gen {len(self.selected_frame_num_acc)} bev images')
        else:
            self.selected_frame_num_acc = []
            for i in range(self.groups_cnt - 1):
                selected_num_acc = self.get_right_num((frm_cnt * i, frm_cnt * (i + 1)))
                self.selected_frame_num_acc.append(selected_num_acc)
            selected_num_acc = self.get_right_num(
                (frm_cnt * (self.groups_cnt - 1), self.total_frames - 1))
            self.selected_frame_num_acc.append(selected_num_acc)
            log.info(
                f'random selected frames: {self.selected_frame_num_acc} to gen {len(self.selected_frame_num_acc)} bev images')

    def get_right_num(self, frm_range, is_random=True, check_acc=True):
        if is_random:
            num = random.randint(frm_range[0], frm_range[1])
            ts = self.ref_tss[num]
        else:
            num = frm_range[0]
            ts = self.ref_tss[num]
        acc_ts, (acc_lat, acc_lgt) = self.acc_reader.get_acc(ts)
        if acc_lat is None or acc_lgt is None:
            return num, None, None
        if check_acc and abs(acc_lat) > self.max_acc[0] or abs(acc_lgt) > self.max_acc[1]:
            for num in range(frm_range[0], frm_range[1]):
                ts = self.ref_tss[num]
                acc_ts, (acc_lat, acc_lgt) = self.acc_reader.get_acc(ts)
                if abs(acc_lat) <= self.max_acc[0] and abs(acc_lgt) <= self.max_acc[1]:
                    log.debug(f'acc= [{acc_ts}, {acc_lat}, {acc_lgt}]')
                    return num, acc_lat, acc_lgt
            log.error(f'not found frame of acc less than {self.max_acc}, the current acc is [{acc_lat, acc_lgt}]')
        log.debug(f'acc= [{acc_ts}, {acc_lat}, {acc_lgt}]')
        return num, acc_lat, acc_lgt

    def load_sorted_images_list(self, cam_name) -> List[int]:
        if cam_name not in self.cam_tss:
            cam_dir = os.path.join(self.bag_dir, cam_name)
            if not os.path.exists(cam_dir):
                raise ViewMissingError(f'{cam_name} missing')
            images_ts = [int(f.split('.')[0]) for f in os.listdir(cam_dir) if f.endswith(self.suffix)]
            self.cam_tss[cam_name] = sorted(images_ts)

        return self.cam_tss[cam_name]

    def get_image_path(self, cam_name, ts):
        return os.path.join(self.bag_dir, cam_name, f'{ts}{self.suffix}')


    def find_closest_ix(self, tss, ts, idx_range):
        min_diff = np.inf
        best_ix = None
        for ix in idx_range:
            diff = abs(tss[ix] - ts)
            if diff < self.max_time_diff:
                if diff < min_diff:
                    if best_ix is not None:
                        log.warning(f'more than 1 frame of diff less than {self.max_time_diff // 1E6} ms')
                    min_diff = diff
                    best_ix = ix

        return best_ix

    def get_closest_frame(self, tss: List[int], ts: int, frame_nu: int) -> Tuple[int, int]:
        s_ts = tss[frame_nu]
        if abs(s_ts - ts) < self.max_time_diff / 100:  # diff <0.5ms
            return frame_nu, s_ts
        if s_ts < ts:
            ix = self.find_closest_ix(tss, ts, range(frame_nu, len(tss), 1))
            if ix is not None:
                return ix, tss[ix]

        else:
            ix = self.find_closest_ix(tss, ts, range(frame_nu, 0, -1))
            if ix is not None:
                return ix, tss[ix]

        return None, None


    def get_frame(self, cam_names: Tuple[str], expected_frame_num: int) -> Tuple[int, List[str], List[int]]:
        '''

        @param cam_names: ()
        @param expected_frame_num:
        @return:
        '''
        ref_cam = cam_names[-1]
        ref_cam_tss = self.load_sorted_images_list(ref_cam)
        assert expected_frame_num < len(ref_cam_tss)
        ref_cam_ts = ref_cam_tss[expected_frame_num]
        align_frms, align_frms_diff = [], []
        for cam_name in cam_names:
            tss = self.load_sorted_images_list(cam_name)
            frm_num, ts = self.get_closest_frame(tss, ref_cam_ts, expected_frame_num)
            if frm_num is None:
                raise FrameAlignError(f'not found closest frame {expected_frame_num} from {cam_name}')
            assert tss[frm_num] == ts
            diff = ts - ref_cam_ts
            assert abs(diff) < self.max_time_diff
            if frm_num != expected_frame_num:
                log.warning(f'{cam_name} {frm_num}({ts}) != {expected_frame_num} ({ref_cam_ts})')
            selected_image = self.get_image_path(cam_name, ts)
            log.info(
                f'{cam_name}: frame num {frm_num} {os.path.basename(selected_image)} time diff {diff / 1E6} ms')
            align_frms.append(selected_image)
            align_frms_diff.append(diff)

        assert len(align_frms) == len(cam_names) == len(align_frms_diff)
        # log.info(f'load {expected_frame_num} frame')
        return expected_frame_num, align_frms, align_frms_diff


    def get_multi_group_frames(self, cam_names) -> List[Tuple[int, List[int], List[float]]]:
        '''

        @param cam_names:
        @return: [(frm_num, [ts.jpg,ts2.jog...], [diff1,diff2...], acc_lat, acc_lgt)]
        '''
        multi_group = []
        for f, acc_lat, acc_lgt in self.selected_frame_num_acc:
            try:
                self.get_frame(cam_names, f)
                multi_group.append((*self.get_frame(cam_names, f), acc_lat, acc_lgt))
            except FrameAlignError as e:
                log.warning(e)
        # multi_group = [self.get_frame(cam_names, f) for f in self.selected_frame_num_acc]
        return multi_group


if __name__ == '__main__':
    # bag_dir = '/data2/ipmchecker/2025-05-31-09-18-57/2025-05-31-09-18-57'
    bag_dir = '/data2/ipmchecker/2025-06-19-13-13-55'
    sync_txt = os.path.join(bag_dir, 'sync_sensors.txt')
    with open(sync_txt, 'r') as f:
        lines = f.readlines()
    img_name_dict = {}
    keys = lines[0].split(' ')[1:]
    for i, k in enumerate(keys):
        k = k.strip('/\n')
        keys[i] = k
        img_name_dict[k] = []
    for line in lines[1:]:
        splits = line.split(' ')[1:]
        for i, ts in enumerate(splits):
            v = ts.strip('/\n')
            if v and v != 'null':
                img_name_dict[keys[i]].append(int(v))
    frm_nums = np.arange(0, len(img_name_dict['cam_front_left']))
    aligner = ImagesTimeAligner(bag_dir, frm_nums.tolist(), ('long', 'around'))
    groups = aligner.get_multi_group_frames(cam_names_dict['long'])
    notfound_cnt = 0
    ix = -1
    cnt = 0
    for k, v in img_name_dict.items():
        nv = np.array(v)
        if abs(np.diff(nv).max()) > 50E6:
            print(f'{k}  diff {np.diff(nv).max() / 1E6} {v} ')
    for i in range(len(groups)):
        ts = int(os.path.basename(groups[i][1][-1]).split('.')[0])
        # ts = aligner.cam_tss['cam_front_left'][i]
        cnt += 1
        try:
            ix = img_name_dict['cam_front_left'].index(ts)
        except Exception as e:
            notfound_cnt += 1
            continue
        assert ix >= 0
        try:
            assert img_name_dict['cam_front_left'][ix] == ts
            for cam_ix, cam in enumerate(cam_names_dict['long']):
                tts = int(os.path.basename(groups[i][1][cam_ix]).split('.')[0])
                assert abs(tts - ts) < 50E6
                diff = tts - img_name_dict[cam][ix]
                if abs(diff) > 50E6:
                    print(f'diff {cam} {i} {ix} {diff / 1E6}')

        except IndexError as e:
            pass
            # print(e)

    print(f'not found {notfound_cnt}', cnt)
