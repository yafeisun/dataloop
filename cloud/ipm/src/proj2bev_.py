import argparse
import json
import math
import os
from typing import List, Tuple
import cv2
import numpy as np
from src.utils import log
from src.find_images import cam_names_dict
from src.utils import load_cam_param, load_image, AntiDistortion, PointsTrans


def get_txt_coor(cam_name, bev_img, cx_ratio=0.5, cy_ratio=0.5):
    if cam_name == 'cam_around_back':
        return int(bev_img.shape[1] * cx_ratio - 400), int(bev_img.shape[0] * cy_ratio)
    elif cam_name == 'cam_around_front':
        return int(bev_img.shape[1] * cx_ratio + 200), int(bev_img.shape[0] * cy_ratio)
    elif cam_name == 'cam_around_left':
        return int(bev_img.shape[1] * cx_ratio), 50
    elif cam_name == 'cam_around_right':
        return int(bev_img.shape[1] * cx_ratio), bev_img.shape[0] - 100
    elif cam_name == 'cam_front_left':
        return bev_img.shape[1] - 200, int(bev_img.shape[0] * cy_ratio)
    elif cam_name == 'cam_side_left_front':
        return int(bev_img.shape[1] * cx_ratio + 150), 50
    elif cam_name == 'cam_side_left_back':
        return int(bev_img.shape[1] * cx_ratio - 200), 50
    elif cam_name == 'cam_side_right_front':
        return int(bev_img.shape[1] * cx_ratio + 150), bev_img.shape[0] - 100
    elif cam_name == 'cam_side_right_back':
        return int(bev_img.shape[1] * cx_ratio - 200), bev_img.shape[0] - 100
    elif cam_name == 'cam_back':
        return 0, int(bev_img.shape[0] * cy_ratio)
    elif cam_name == 'cam_front_right':
        return bev_img.shape[1] - 200, 50
    else:
        print('cam_name error')
        return None


class BevProjector:
    min_x, max_x = -20, 30
    min_y, max_y = -10, 10
    res = 0.05  # ~H~F__~N~G~H__/~C~O| ~I
    width = math.ceil((max_x - min_x) / res) + 1
    height = math.ceil((max_y - min_y) / res) + 1
    color_list = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (128, 128, 128)]

    def __init__(self, cam_param_path, cam_type='around'):
        self.cam_param_path = cam_param_path
        self.cam_names = cam_names_dict[cam_type]

        self.cam_param = {}
        self.trans_dict = {}
        self.distortions = {}
        self.init()

    def init(self):
        self.cam_param = load_cam_param(self.cam_param_path, v6=False)
        for name in self.cam_names:
            dist = AntiDistortion(self.cam_param[name]['intrinsic'], self.cam_param[name]['distCoeff'],
                                  self.cam_param[name]['image_shape'])
            self.cam_param[name]['intrinsic'] = dist.new_K  # only run once, update intrinsic
            intrinsic, extrinsic, image_shape, dist_coeff = self.cam_param[name]['intrinsic'], \
                self.cam_param[name]['extrinsic'], \
                self.cam_param[name]['image_shape'], \
                self.cam_param[name]['distCoeff']
            self.distortions[name] = dist
            trans = PointsTrans(intrinsic, extrinsic, image_shape)
            self.trans_dict[name] = trans


    def run(self, img_list, cam_name_list, det_results):
        bev_img = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        for img_path, cam_name in zip(img_list, cam_name_list):
            rgb_img = load_image(img_path)
            if rgb_img is None:
                continue
            rgb_img = self.distortions[cam_name].anti(rgb_img)

            y_coors, x_coors = np.mgrid[0:rgb_img.shape[0], 0:rgb_img.shape[1]]
            coords = np.dstack((x_coors, y_coors)).reshape(-1, 2)
            bev_coords, coords = self.trans_dict[cam_name].pixel_to_bev(coords)
            bev_img = self.trans_bev_to_img(bev_coords, coords, rgb_img, cam_name, bev_img)
        for i in range(len(det_results)):
            bev_img = self.trans_bev_to_img_(det_results[i], (255, 0, 0), bev_img=bev_img)
            results = det_results[i]
            # draw label
            for pts in results:
                bev_coords, _ = self.trans_dict[cam_name_list[i]].pixel_to_bev(np.array(pts))
                self.trans_bev_to_img_(bev_coords, alpha=1.0, bev_img=bev_img, color=np.array(self.color_list[i]))
        # bev_img = self.draw_coor_sys(bev_img)
        bev_img = self.draw_grid(bev_img)
        return bev_img
    
    def run(self, img_list, cam_name_list, det_results):
        bev_img = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        for img_path, cam_name in zip(img_list, cam_name_list):
            rgb_img = load_image(img_path)
            if rgb_img is None:
                continue
            rgb_img = self.distortions[cam_name].anti(rgb_img)

            y_coors, x_coors = np.mgrid[0:rgb_img.shape[0], 0:rgb_img.shape[1]]
            coords = np.dstack((x_coors, y_coors)).reshape(-1, 2)
            bev_coords, coords = self.trans_dict[cam_name].pixel_to_bev(coords)
            bev_img = self.trans_bev_to_img(bev_coords, coords, rgb_img, cam_name, bev_img)
        for i in range(len(det_results)):
            bev_img = self.trans_bev_to_img_(det_results[i], (255, 0, 0), bev_img=bev_img)
            results = det_results[i]
            # draw label
            for pts in results:
                bev_coords, _ = self.trans_dict[cam_name_list[i]].pixel_to_bev(np.array(pts))
                self.trans_bev_to_img_(bev_coords, alpha=1.0, bev_img=bev_img, color=np.array(self.color_list[i]))
        # bev_img = self.draw_coor_sys(bev_img)
        bev_img = self.draw_grid(bev_img)
        return bev_img


    def draw_coor_sys(self, bev_img):
        # ~L~E~[~F~Z~D~[~[____~B
        pts = [[15, 5], [15, -5], [-15, -5], [-15, 5]]
        for i in range(len(pts)):
            pts[i] = self.bev_to_pixel(*pts[i])
        pts = np.array(pts, np.int32)
        # for i in range(len(pts) - 1):
        #     cv2.line(bev_img, pts[i], pts[i+1], (255, 0, 0), 2)
        cv2.polylines(bev_img, [pts], isClosed=True, color=(255, 0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

        # ~]~P| ~G__~Z~D~N~_~B~Lx__~R~Ly____~B~R~L~S~]~_~B
        x_start = self.bev_to_pixel(-15, 0)
        x_end = self.bev_to_pixel(15, 0)
        y_start = self.bev_to_pixel(0, -5)
        y_end = self.bev_to_pixel(0, 5)
        cv2.arrowedLine(bev_img, x_start, x_end, (0, 255, 0, 255), 2, tipLength=0.02)
        cv2.arrowedLine(bev_img, y_start, y_end, (0, 255, 0, 255), 2, tipLength=0.06)
        origin = self.bev_to_pixel(0, 0)
        cv2.circle(bev_img, origin, 8, (0, 0, 255, 255), -1)

        # ~X~H~]~P| ~G__~Z~D~V~G~W
        x_c = self.bev_to_pixel(10, 0.2)
        y_c = self.bev_to_pixel(0.2, 2.5)
        cv2.putText(bev_img, 'x', x_c, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0, 255), 2)
        cv2.putText(bev_img, 'y', y_c, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0, 255), 2)

        return bev_img

    def draw_grid(self, bev_img):
        up_sample = 4
        bev_img = cv2.resize(
            bev_img, (bev_img.shape[1] * up_sample, bev_img.shape[0] * up_sample),
            interpolation=cv2.INTER_LINEAR
        )
        left, right, top, bottom = -10, 15, 7.5, -7.5
        # ~L~E~[~F~Z~D~[~[____~B
        pts = [[right, top], [right, bottom], [left, bottom], [left, top]]
        for i in range(len(pts)):
            pts[i] = self.bev_to_pixel(*pts[i], np.float64) * up_sample
        pts = np.array(pts, np.int32)
        # for i in range(len(pts) - 1):
        #     cv2.line(bev_img, pts[i], pts[i+1], (255, 0, 0), 2)
        cv2.polylines(bev_img, [pts], isClosed=True, color=(255, 152, 16, 255), thickness=2, lineType=cv2.LINE_AA)

        # ~Q| __~Lx__~R~Ly____~B~R~L~S~]~_~B 30*20
        grid_w, grid_h = 0.2, 0.075
        for x0 in np.arange(left, right, grid_w, dtype=np.float64):
            p1 = self.bev_to_pixel(x0, bottom, np.float64)
            p1 *= up_sample
            p2 = self.bev_to_pixel(x0, top, np.float64)
            p2 *= up_sample

            cv2.line(bev_img, p1.round().astype(np.int32), p2.round().astype(np.int32), (255, 152, 16, 250), 1)
        for y0 in np.arange(bottom, top, grid_h, dtype=np.float64):
            p1 = self.bev_to_pixel(left, y0, dtype=np.float64)
            p1 *= up_sample
            p2 = self.bev_to_pixel(right, y0, dtype=np.float64)
            p2 *= up_sample
            cv2.line(bev_img, p1.round().astype(np.int32), p2.round().astype(np.int32), (255, 152, 16, 250), 1)
        origin = self.bev_to_pixel(0, 0) * up_sample
        cv2.circle(bev_img, origin, 16, (0, 0, 255, 255), -1)

        # ~X~H~]~P| ~G__~Z~D~V~G~W
        # x_c = self.bev_to_pixel(10, 0.2)
        # y_c = self.bev_to_pixel(0.2, 2.5)
        # cv2.putText(bev_img, 'x', x_c, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0, 255), 2)
        # cv2.putText(bev_img, 'y', y_c, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0, 255), 2)

        return bev_imgz


    def bev_to_pixel(self, x, y, dtype=np.int32):
        # return np.array([(x - self.min_x - self.res / 2) / self.res, (self.max_y - y + self.res / 2) / self.res], dtype)
        return np.array([(x - self.min_x) / self.res, (self.max_y - y) / self.res], dtype)

    def pixel_to_bev(self, x, y, dtype=np.int32):
        return np.array([self.res * x + self.min_x, self.max_y - y * self.res], dtype)

    def get_pix_rgb(self, img, x, y):
        """
            ____~B~]~P| ~G(x,y)~[~L~O~L__~@~O~R~@
            ~O~B~U~Z
                img: ~S~E~[~C~O(numpy~U~D~LH~WW~WC)
                x: __~Bx~]~P| ~G(____~V~P~Q)
                y: __~By~]~P| ~G(~^~B~[~V~P~Q)
            ~T~[~^~Z
                interpolated: ~O~R~@~P~N~Z~DRGB~@(3~E~C| ~U~D)
            """
        # ~N~O~V~[~[__~B~Q~U~U~]~P| ~G
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        x1, y1 = min(x0 + 1, img.shape[1] - 1), min(y0 + 1, img.shape[0] - 1)


        # __~W~A~O__~G~O
        dx, dy = x - x0, y - y0

        # __~U~L~@~_
        x0 = max(0, x0)
        y0 = max(0, y0)

        # ~N~O~V~[~[__~R~B~Z~DRGB~@
        top_left = img[y0, x0]
        top_right = img[y0, x1]
        bottom_left = img[y1, x0]
        bottom_right = img[y1, x1]

        # ~O~L__~@~O~R~@__~W
        interpolated = (
                (1 - dx) * (1 - dy) * top_left +
                dx * (1 - dy) * top_right +
                (1 - dx) * dy * bottom_left +
                dx * dy * bottom_right
        )

        return np.clip(interpolated, 0, 255).astype(int)


    def trans_bev_to_img(self, bev_coords, img_coords, rgb_img, cam_name, bev_img=None):
        r"""
        Transform BEV coordinates to image coordinates and map RGB values.

        Args:
            bev_coords (np.ndarray): BEV coordinates, shape (N, 2), [x, y]
            img_coords (np.ndarray): Image coordinates, shape (N, 2), [u, v]
            rgb_img (np.ndarray): RGB image, shape (height, width, 3), RGB values
            cam_name (str): Camera name, e.g., 'cam_front_right'
            bev_img (np.ndarray, optional): Existing BEV image. Defaults to None.

        Returns:
            np.ndarray: BEV image, shape (height, width, 3), RGB values
        """

        if bev_img is None:
            bev_img = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        # ~I~T~Y~E~G~L~C~[~Z~D~B
        cond = (bev_coords[:, 0] >= self.min_x) & (bev_coords[:, 0] <= self.max_x) & (
                bev_coords[:, 1] >= self.min_y) & (
                       bev_coords[:, 1] <= self.max_y)
        bev_coords = bev_coords[cond]
        img_coords = img_coords[cond]
        # ~I~T~Y~I~M~F__~A~L~Z~D~L~_~_
        if cam_name == 'cam_front_right':
            cond = img_coords[:, 1] <= 1620
            bev_coords = bev_coords[cond]
            img_coords = img_coords[cond]
        # bev~]~P| ~G~B____bev_img~C~O| ~]~P| ~G
        pixel_coords = bev_coords.copy()
        pixel_coords[:, 0] = (bev_coords[:, 0] - self.min_x) / self.res
        pixel_coords[:, 1] = (self.max_y - bev_coords[:, 1]) / self.res
        # ~I~T~Y~G~M~M~B~L~]~U~Y__~@__~B~L~E~Y~B__~U~L~X__~G~M~M~X~H~F~[~V~C~O| ~B
        pixel_coords, idxs = np.unique(pixel_coords.round(), return_index=True, axis=0)
        pixel_coords = pixel_coords.astype(int)
        img_coords = img_coords[idxs]
        # __bev_img~K~Jrbg~@
        for i in range(img_coords.shape[0]):
            bev_img[pixel_coords[i][1], pixel_coords[i][0], :3] = rgb_img[img_coords[i][1], img_coords[i][0]]
            bev_img[pixel_coords[i][1], pixel_coords[i][0], 3] = 255
        # __~T__~]~Mbev ~Z~D~C~O| ~B~\img __~O~Vrgb ~@
        # bev_coords2= np.mgrid[0:bev_img.shape[0], 0:bev_img.shape[1]]
        # bev_coords2 = np.dstack(bev_coords2).reshape(-1, 2)
        # bx, by = self.pixel_to_bev(bev_coords2[:,0], bev_coords2[:,1])
        # bxy = np.vstack([bx, by])
        # pix = self.trans_dict[cam_name].bev_to_pixel(bxy)
        # for y in range(bev_img.shape[0]):
        #     for x in range(bev_img.shape[1]):
        #         bx, by = self.pixel_to_bev(x, y)
        #         pix = self.trans_dict[cam_name].bev_to_pixel(np.array([(bx, by)]))
        #         if not pix.size:
        #             continue
        #         # px, py = self.bev_to_pixel(x, y)
        #         rgb = self.get_pix_rgb(rgb_img, pix[0,0], pix[0,1])
        #         bev_img[y, x, :3] = rgb
        #         bev_img[y, x, 3] = 255
        bev_img = np.array(bev_img, dtype=np.uint8)
        return bev_img

    def trans_bev_to_img_(self, bev_coords, color, alpha=0.5, bev_img=None):
        if bev_img is None:
            bev_img = np.zeros((self.height, self.width, 3))
        if bev_coords.shape[0] == 0:
            return bev_img
        # ~I~T~Y~E~G~L~C~[~Z~D~B
        cond = (bev_coords[:, 0] >= self.min_x) & (bev_coords[:, 0] <= self.max_x) & (
                bev_coords[:, 1] >= self.min_y) & (
                       bev_coords[:, 1] <= self.max_y)
        bev_coords = bev_coords[cond]
        if bev_coords.shape[0] == 0:
            return bev_img
        # bev~]~P| ~G~B____bev_img~C~O| ~]~P| ~G
        pixel_coords = bev_coords.copy()
        pixel_coords[:, 0] = (bev_coords[:, 0] - self.min_x) / self.res
        pixel_coords[:, 1] = (self.max_y - bev_coords[:, 1]) / self.res
        # ~I~T~Y~G~M~M~B~L~]~U~Y__~@__~B~L~E~Y~B__~U~L~X__~G~M~M~X~H~F~[~V~C~O| ~B
        pixel_coords, idxs = np.unique(pixel_coords.round(), return_index=True, axis=0)
        pixel_coords = pixel_coords.astype(int)
        # __bev_img~K~Jrbg~@
        beta = 1 - alpha
        for i in range(pixel_coords.shape[0]):
            bev_img[pixel_coords[i][1], pixel_coords[i][0], :3] = alpha * color + beta * bev_img[pixel_coords[i][1],
            pixel_coords[i][0]]
            bev_img[pixel_coords[i][1], pixel_coords[i][0], 3] = 255
        bev_img = np.array(bev_img, dtype=np.uint8)
        return bev_img

def load_label(label_path):
    with open(label_path, 'r') as f:
        data = json.load(f)
    front_right_label = data['cam_front_right']
    back_label = data['cam_back']
    labels = {'cam_front_right': {}, 'cam_back': {}}
    for k, v in front_right_label.items():
        labels['cam_front_right'][k] = [line['points'] for line in v['lane_lines']]
    for k, v in back_label.items():
        labels['cam_back'][k] = [line['points'] for line in v['lane_lines']]
    return labels


def project_imgs_to_bev(projector: BevProjector, multi_group: List[Tuple[int, List[str], List[int]]],
                        out_path: str, cam_names: list, video_gen: bool = True):
    '''

    @param projector:
    @param multi_group: [(frmnum,[iamge_path1, iamge_path1...]), (frmnum,[iamge_path1, iamge_path1...]),(frmnum,[iamge_path1, iamge_path1...])]
    @param out_path:
    @param cam_names:
    @param video_gen:
    @return:
    '''

    bev_width, bev_height = projector.width * 2, projector.height * 2
    cx_ratio = abs(projector.min_x) / float(abs(projector.min_x) + abs(projector.max_x))
    cy_ratio = abs(projector.min_y) / float(abs(projector.min_y) + abs(projector.max_y))
    fps = 10
    video_writer = None
    if video_gen:
        print('gen picture and video...')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(os.path.join(out_path, "bev.avi"), fourcc, fps,
                                       (bev_width, bev_height))
        video_writer.set(cv2.CAP_PROP_BITRATE, 5000)
    else:
        print('not gen video...')
    for i in range(len(multi_group)):
        print(f'processing group {i + 1}...')
        frm_num, grp_img, grp_time_diff = multi_group[i]
        bev_img = projector.run(grp_img, projector.cam_names, det_results=[])
        bev_img = cv2.resize(bev_img, (bev_width, bev_height))

        # draw cam names
        for ix, name in enumerate(cam_names):
            coor = get_txt_coor(name, bev_img, cx_ratio, cy_ratio)
            cv2.putText(bev_img, f'{name.strip("cam_")}', coor, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255, 255), 2)
            diff_coor = np.array(coor)+ np.array([0, 30])
            cv2.putText(bev_img, f'{int(grp_time_diff[ix] // 1E6)} ms', diff_coor, cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255, 255), 2)
        img_path = f'{out_path}_{i + 1}_{frm_num}.png'
        cv2.imwrite(img_path, bev_img)
        log.info(f'output {img_path}')
        if video_writer:
            video_writer.write(bev_img[:, :, :3])

    if video_writer:
        video_writer.release()


def project_imgs_to_bev_single(projector: BevProjector, multi_group: List[List],
                               out_dir: str, cam_names: list):
    '''

    @param projector:
    @param multi_group: [[iamge_path1, iamge_path1...], [iamge_path1, iamge_path1...],[iamge_path1, iamge_path1...]]
    @param out_dir:
    @param cam_names:
    @return:
    '''
    log.error('not support')
    # bev_width, bev_height = projector.width * 2, projector.height * 2
    # for i in range(len(img_name_dict[cam_names[0]])):
    #     print(f'processing {i}')
    #     out_dir_n = os.path.join(out_dir, f'{i}')
    #     os.makedirs(out_dir_n, exist_ok=True)
    #     skip_flag = False
    #     ts_fr = int(img_name_dict["cam_front_right"][i])
    #     for k, name in enumerate(cam_names):
    #         # img_path = os.path.join(img_dir, name, img_name_dict[name][i] + '.jpg')
    #         img_path = img_name_dict[name][i]
    #         if not os.path.exists(img_path):
    #             print(f'img {img_path} not found, skip this frame')
    #             break
    #         ts = int(img_name_dict[name][i])
    #         if abs(ts_fr - ts) > 10000000:
    #             print(f'frame {i}, {name} ts {ts}, cam front right ts: {ts_fr}!')
    #         bev_img = projector.run([img_path], [name], det_results=[])
    #         bev_img = cv2.resize(bev_img, (bev_width, bev_height))
    #         # draw cam names
    #         coor = get_txt_coor(name, bev_img)
    #         cv2.putText(bev_img, f'{name.strip("cam_")}', coor, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255, 255), 2)
    #         output_path = os.path.join(out_dir_n, f'{k + 1:02d}_{name}_{i}.png')
    #         cv2.imwrite(output_path, bev_img)

def gen_bev(car_yaml, multi_group, proj_type, output_dir):
    # def gen_bev(bag_dir, output_dir, proj_type='side'):

    # sync_txt = os.path.join(input_dir, 'sync_sensors.txt')
    out_path = os.path.join(output_dir, f'bev_{proj_type}')
    assert os.path.exists(car_yaml)
    projector = BevProjector(car_yaml, cam_type=proj_type)
    # with open(sync_txt, 'r') as f:
    #     lines = f.readlines()
    # img_name_dict = {}
    # aligner = ImagesTimeAligner(bag_dir)
    # # ~I~M~]~Z~D__
    # frms = aligner.get_front_frames(cam_names_dict[proj_type])
    # multi_group = [frms]
    # # __~W~Z~D__
    # frms = aligner.get_middle_frames(cam_names_dict[proj_type])
    # multi_group.append(frms)
    # # ~P~N~]~Z~D__
    # frms = aligner.get_back_frames(cam_names_dict[proj_type])
    # multi_group.append(frms)

    # keys = lines[0].split(' ')[1:]
    # for i, k in enumerate(keys):
    #     k = k.strip('/\n')
    #     keys[i] = k
    #     img_name_dict[k] = []
    # for line in lines[1:]:
    #     splits = line.split(' ')[1:]
    #     for i, ts in enumerate(splits):
    #         img_name_dict[keys[i]].append(ts.strip('/\n'))
    if proj_type in ['around', 'side', 'long']:
        project_imgs_to_bev(projector, multi_group, out_path, cam_names_dict[proj_type], False)
    elif proj_type == 'single':
        project_imgs_to_bev_single(projector, multi_group, out_path, cam_names_dict[proj_type])

# if __name__ == '__main__':
# parser = argparse.ArgumentParser()
# parser.add_argument('--version', action='version', version='IMPChecker 0.5')
# parser.add_argument('--input_dir', type=str,
#                     default='/data2/ipmchecker/2025-05-31-09-18-57/2025-05-31-09-18-57')
# parser.add_argument('--img_dir', type=str,
#                     default='/data2/ipmchecker/2025-05-31-09-18-57/2025-05-31-09-18-57')
# parser.add_argument('--type', '-t', type=str, choices=['around', 'side', 'single', 'long'], default='long')
# args = parser.parse_args()
#
# run(args.input_dir, args.img_dir, args.type)
