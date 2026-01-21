import glob
import os

import cv2
import sys
from scipy.spatial.transform import Rotation as R

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.find_images import ImagesTimeAligner
import numpy as np
from src.utils import AntiDistortion, log, load_cam_param, PointsTrans
from src import ParamShapeError
from src.find_images import cam_names_dict


class BevPlot:
    min_x, max_x = -20, 30  # m
    min_y, max_y = -10, 10  # m
    resolution = 0.01  # ~H~F__~N~G~H__/~C~O| ~I

    channels = 4
    up_sample = 2

    def __init__(self, cam_param_path, cam_type):
        self.scale = 1 / self.resolution
        r = R.from_euler('z', 90, degrees=True)
        T = np.eye(3)
        T[:2, :2] = r.as_matrix()[:2, :2]
        T[:2, 2] = [-self.min_y, -self.min_x]

        self.bev2plot = self.scale * T
        self.bev2plot[2, 2] = 1
        self.pixel_rows = int(np.ceil((self.max_x - self.min_x) / self.resolution))
        self.pixel_cols = int(np.ceil((self.max_y - self.min_y) / self.resolution))

        self.grid_points = self.gen_grid_points(self.pixel_rows, self.pixel_cols)
        self.cam_param = {}
        self.distortions = {}
        self.trans_dict = {}
        self.cam_names = cam_names_dict[cam_type]
        self.init_param(cam_param_path)

    def init_param(self, cam_param_path):
        self.cam_param = load_cam_param(cam_param_path, v6=False)
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

    def get_txt_coor(self, cam_name):
        if cam_name == 'cam_around_back':
            return self.min_x + 1, 0
        elif cam_name == 'cam_around_front':
            return 5, 0
        elif cam_name == 'cam_around_left':
            return 0, self.min_y + 2
        elif cam_name == 'cam_around_right':
            return 0, self.max_y - 2
        elif cam_name == 'cam_front_left':
            return self.max_x - 4, 0
        elif cam_name == 'cam_side_left_front':
            return 3, self.min_y + 1
        elif cam_name == 'cam_side_left_back':
            return self.min_x / 2, self.min_y + 1
        elif cam_name == 'cam_side_right_front':
            return 3, self.max_y - 1
        elif cam_name == 'cam_side_right_back':
            return self.min_x / 2, self.max_y - 1
        elif cam_name == 'cam_back':
            return self.min_x + 1, 0
        elif cam_name == 'cam_front_right':
            return self.max_x * 2 / 3, self.max_y - 1
        else:
            print('cam_name error')
            return None
        
    def init_param(self, cam_param_path):
        self.cam_param = load_cam_param(cam_param_path, v6=False)
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

    def get_txt_coor(self, cam_name):
        if cam_name == 'cam_around_back':
            return self.min_x + 1, 0
        elif cam_name == 'cam_around_front':
            return 5, 0
        elif cam_name == 'cam_around_left':
            return 0, self.min_y + 2
        elif cam_name == 'cam_around_right':
            return 0, self.max_y - 2
        elif cam_name == 'cam_front_left':
            return self.max_x - 4, 0
        elif cam_name == 'cam_side_left_front':
            return 3, self.min_y + 1
        elif cam_name == 'cam_side_left_back':
            return self.min_x / 2, self.min_y + 1
        elif cam_name == 'cam_side_right_front':
            return 3, self.max_y - 1
        elif cam_name == 'cam_side_right_back':
            return self.min_x / 2, self.max_y - 1
        elif cam_name == 'cam_back':
            return self.min_x + 1, 0
        elif cam_name == 'cam_front_right':
            return self.max_x * 2 / 3, self.max_y - 1
        else:
            print('cam_name error')
            return None


    def gen_grid_points(self, rows, cols):
        x = np.linspace(self.min_x, self.max_x, rows)
        y = np.linspace(self.min_y, self.max_y, cols)
        X, Y = np.meshgrid(x, y, indexing='ij')
        return np.stack([X.ravel(), Y.ravel()], axis=1)

    def project(self, bev_img, cam_name, bgr_img):
        pixel, mask = self.trans_dict[cam_name].bev_to_pixel_mask(self.grid_points)
        pixel *= self.up_sample
        valid_pixel = pixel[:, mask]
        uv = valid_pixel.round().astype(np.int32).T

        img = self.distortions[cam_name].anti(bgr_img)

        if self.up_sample != 1:
            resize_shape = img.shape * np.array([self.up_sample, self.up_sample, 1])
            img = cv2.resize(img, resize_shape[:2][::-1], interpolation=cv2.INTER_LINEAR)
        pt_color = img[uv[:, 1], uv[:, 0]]
        if self.channels > 3:
            pt_color = np.hstack((pt_color, np.full((pt_color.shape[0], 1), 255)))
        mask = mask.reshape([self.pixel_rows, self.pixel_cols])
        bev_img[mask] = pt_color
        return bev_img

    def check_shape(self, cam_name, img):
        if np.array_equal(self.cam_param[cam_name]['image_shape'], np.array(img.shape[:2][::-1])):
            return True
        raise ParamShapeError(
            f'param {cam_name} shape {self.cam_param[cam_name]["image_shape"]} != {img.shape[:2][::-1]} of real image')

    def to_pixel(self, pts, r: bool = True):
        '''

        @param pts: n*2
        @return:  n*2
        '''
        if r:
            hom_pts = np.hstack((pts, np.ones((pts.shape[0], 1)))).T  # 3*n
            return (self.bev2plot @ hom_pts)[:2, :].T.round().astype(np.int32)  # n*2
        else:
            trans = pts + [-self.min_x, -self.min_y]
            return (trans * self.scale).round().astype(np.int32)

    def draw_grid(self, bev_img):
        '''

        @param bev_img: ~_~Nego ~]~P| ~G__
        @return:
        '''
        min_x, max_x = -10, 15
        min_y, max_y = -7.5, 7.5
        # left, right, top, bottom = -10, 15, 5, -5
        # ~L~E~[~F~Z~D~[~[____~B
        pts = np.array([[min_x, min_y], [min_x, max_y], [max_x, max_y], [max_x, min_y]])
        pts = self.to_pixel(pts)
        # pts = np.stack([pts_x, pts_y], axis=1).round().astype(np.int32)
        cv2.polylines(bev_img, [pts], isClosed=True, color=(255, 152, 16, 255), thickness=2, lineType=cv2.LINE_AA)

        # ~Q| __~Lx__~R~Ly____~B~R~L~S~]~_~B 30*20
        grid_w, grid_h = 0.2, 0.075

        x0 = np.arange(min_x, max_x, grid_w)
        p1_x, p1_y = x0, np.full_like(x0, min_y)
        p2_x, p2_y = x0, np.full_like(x0, max_y)
        pts1 = np.stack([p1_x, p1_y], axis=1)
        pts2 = np.stack([p2_x, p2_y], axis=1)
        pts1 = self.to_pixel(pts1)
        pts2 = self.to_pixel(pts2)
        for p1, p2 in zip(pts1, pts2):
            cv2.line(bev_img, p1, p2, (255, 152, 16, 250), 1)
        y0 = np.arange(min_y, max_y, grid_h)
        p1_x, p1_y = np.full_like(y0, min_x), y0
        p2_x, p2_y = np.full_like(y0, max_x), y0
        pts1 = np.stack([p1_x, p1_y], axis=1)
        pts2 = np.stack([p2_x, p2_y], axis=1)
        pts1 = self.to_pixel(pts1)
        pts2 = self.to_pixel(pts2)
        for p1, p2 in zip(pts1, pts2):
            cv2.line(bev_img, p1, p2, (255, 152, 16, 250), 1)
        #
        origin_x, origin_y = self.to_pixel(np.array([[0, 0]]))[0]
        cv2.circle(bev_img, [int(round(origin_x)), int(round(origin_y))], 16,
                   (0, 0, 255, 255), -1)

        return bev_img

    def put_text(self, bev_img, name, time_diff):
        coor = self.get_txt_coor(name)
        coor = self.to_pixel(np.array([coor]), False)[0]
        cv2.putText(bev_img, f'{name.strip("cam_")}', coor, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255, 255), 4)
        diff_coor = np.array(coor) + np.array([0, 60])
        str_time_diff = f'{int(time_diff // 1E6)}'
        cv2.putText(bev_img, str_time_diff, diff_coor, cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                    (0, 0, 255, 255), 2)

    def put_acc_text(self, bev_img, acc_lat, acc_lgt):
        coor = self.to_pixel(np.array([[-1, -0.4]]), False)[0]
        if acc_lat is not None and acc_lgt is not None:
            str_acc = f'[{acc_lat:.2f}, {acc_lgt:.2f}]'
            cv2.putText(bev_img, str_acc, coor, cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                        (0, 0, 255, 255), 3)


    def project_multi_img(self, grp, only_check_shape=False):
        bev_img = np.zeros((self.pixel_rows, self.pixel_cols, self.channels), dtype=np.uint8)
        bev_img[:, :, 3] = 0
        for ix in range(len(grp[1])):
            img_path = grp[1][ix]
            bgr_img = cv2.imread(img_path)
            assert bgr_img is not None, f'failed to load image {img_path}'

            name = self.cam_names[ix]
            self.check_shape(name, bgr_img)
            if only_check_shape:
                continue
            bev_img = self.project(bev_img, name, bgr_img)

        bev_img = self.draw_grid(bev_img)
        bev_img = cv2.rotate(bev_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # text
        for ix in range(len(grp[1])):
            self.put_text(bev_img, self.cam_names[ix], grp[2][ix])
        self.put_acc_text(bev_img, grp[3], grp[4])
        return bev_img

    def save_image(self, bev_img, output_path):
        cv2.imwrite(output_path, bev_img)
        log.info(f'dumped {output_path} ')

if __name__ == '__main__':

    # X, Y = np.meshgrid([1, 2], [4, 5])
    # # print(X, Y)
    # print(np.stack([X.ravel(), Y.ravel()], axis=1))
    # plt.scatter(X, Y)
    # plt.show()
    bag_dir = '/data2/ipmchecker/2025-06-19-13-13-55'
    cam_param_files = glob.glob(os.path.join(bag_dir, 'car_*.yaml'))
    if not cam_param_files:
        raise Exception(f'Not found cam param(car_*.yaml) files found in {bag_dir}')
    cam_param_file = cam_param_files[0]
    img_path = '/data2/ipmchecker/2025-06-19-13-13-55/cam_front_left/1750339002908410000.jpg'
    # bev_plot = BevPlot(cam_param_file, 'long')
    # bev_plot.project('cam_front_left', img_path)
    # bev_plot.save_image(f'/tmp/ipm/test_1.jpg')

    out_view_types = ('long', 'around')
    aligner = ImagesTimeAligner(bag_dir, [1, 100], out_view_types)
    for t in out_view_types:
        bev_plot = BevPlot(cam_param_file, t)
        multi_group = aligner.get_multi_group_frames(cam_names_dict[t])
        for grp in multi_group:
            bev_img = bev_plot.project_multi_img(grp)
            bev_plot.save_image(bev_img, f'/tmp/ipm/{t}_{grp[0]}.png')
import glob
import os

import cv2
import sys
from scipy.spatial.transform import Rotation as R

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.find_images import ImagesTimeAligner
import numpy as np
from src.utils import AntiDistortion, log, load_cam_param, PointsTrans
from src import ParamShapeError
from src.find_images import cam_names_dict


class BevPlot:
    min_x, max_x = -20, 30  # m
    min_y, max_y = -10, 10  # m
    resolution = 0.01  # ~H~F__~N~G~H__/~C~O| ~I

    channels = 4
    up_sample = 2

    def __init__(self, cam_param_path, cam_type):
        self.scale = 1 / self.resolution
        r = R.from_euler('z', 90, degrees=True)
        T = np.eye(3)
        T[:2, :2] = r.as_matrix()[:2, :2]
        T[:2, 2] = [-self.min_y, -self.min_x]

        self.bev2plot = self.scale * T
        self.bev2plot[2, 2] = 1
        self.pixel_rows = int(np.ceil((self.max_x - self.min_x) / self.resolution))
        self.pixel_cols = int(np.ceil((self.max_y - self.min_y) / self.resolution))

        self.grid_points = self.gen_grid_points(self.pixel_rows, self.pixel_cols)
        self.cam_param = {}
        self.distortions = {}
        self.trans_dict = {}
        self.cam_names = cam_names_dict[cam_type]
        self.init_param(cam_param_path)

    def init_param(self, cam_param_path):
        self.cam_param = load_cam_param(cam_param_path, v6=False)
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

    def get_txt_coor(self, cam_name):
        if cam_name == 'cam_around_back':
            return self.min_x + 1, 0
        elif cam_name == 'cam_around_front':
            return 5, 0
        elif cam_name == 'cam_around_left':
            return 0, self.min_y + 2
        elif cam_name == 'cam_around_right':
            return 0, self.max_y - 2
        elif cam_name == 'cam_front_left':
            return self.max_x - 4, 0
        elif cam_name == 'cam_side_left_front':
            return 3, self.min_y + 1
        elif cam_name == 'cam_side_left_back':
            return self.min_x / 2, self.min_y + 1
        elif cam_name == 'cam_side_right_front':
            return 3, self.max_y - 1
        elif cam_name == 'cam_side_right_back':
            return self.min_x / 2, self.max_y - 1
        elif cam_name == 'cam_back':
            return self.min_x + 1, 0
        elif cam_name == 'cam_front_right':
            return self.max_x * 2 / 3, self.max_y - 1
        else:
            print('cam_name error')
            return None
        
    def init_param(self, cam_param_path):
        self.cam_param = load_cam_param(cam_param_path, v6=False)
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

    def get_txt_coor(self, cam_name):
        if cam_name == 'cam_around_back':
            return self.min_x + 1, 0
        elif cam_name == 'cam_around_front':
            return 5, 0
        elif cam_name == 'cam_around_left':
            return 0, self.min_y + 2
        elif cam_name == 'cam_around_right':
            return 0, self.max_y - 2
        elif cam_name == 'cam_front_left':
            return self.max_x - 4, 0
        elif cam_name == 'cam_side_left_front':
            return 3, self.min_y + 1
        elif cam_name == 'cam_side_left_back':
            return self.min_x / 2, self.min_y + 1
        elif cam_name == 'cam_side_right_front':
            return 3, self.max_y - 1
        elif cam_name == 'cam_side_right_back':
            return self.min_x / 2, self.max_y - 1
        elif cam_name == 'cam_back':
            return self.min_x + 1, 0
        elif cam_name == 'cam_front_right':
            return self.max_x * 2 / 3, self.max_y - 1
        else:
            print('cam_name error')
            return None


    def gen_grid_points(self, rows, cols):
        x = np.linspace(self.min_x, self.max_x, rows)
        y = np.linspace(self.min_y, self.max_y, cols)
        X, Y = np.meshgrid(x, y, indexing='ij')
        return np.stack([X.ravel(), Y.ravel()], axis=1)

    def project(self, bev_img, cam_name, bgr_img):
        pixel, mask = self.trans_dict[cam_name].bev_to_pixel_mask(self.grid_points)
        pixel *= self.up_sample
        valid_pixel = pixel[:, mask]
        uv = valid_pixel.round().astype(np.int32).T

        img = self.distortions[cam_name].anti(bgr_img)

        if self.up_sample != 1:
            resize_shape = img.shape * np.array([self.up_sample, self.up_sample, 1])
            img = cv2.resize(img, resize_shape[:2][::-1], interpolation=cv2.INTER_LINEAR)
        pt_color = img[uv[:, 1], uv[:, 0]]
        if self.channels > 3:
            pt_color = np.hstack((pt_color, np.full((pt_color.shape[0], 1), 255)))
        mask = mask.reshape([self.pixel_rows, self.pixel_cols])
        bev_img[mask] = pt_color
        return bev_img

    def check_shape(self, cam_name, img):
        if np.array_equal(self.cam_param[cam_name]['image_shape'], np.array(img.shape[:2][::-1])):
            return True
        raise ParamShapeError(
            f'param {cam_name} shape {self.cam_param[cam_name]["image_shape"]} != {img.shape[:2][::-1]} of real image')

    def to_pixel(self, pts, r: bool = True):
        '''

        @param pts: n*2
        @return:  n*2
        '''
        if r:
            hom_pts = np.hstack((pts, np.ones((pts.shape[0], 1)))).T  # 3*n
            return (self.bev2plot @ hom_pts)[:2, :].T.round().astype(np.int32)  # n*2
        else:
            trans = pts + [-self.min_x, -self.min_y]
            return (trans * self.scale).round().astype(np.int32)

    def draw_grid(self, bev_img):
        '''

        @param bev_img: ~_~Nego ~]~P| ~G__
        @return:
        '''
        min_x, max_x = -10, 15
        min_y, max_y = -7.5, 7.5
        # left, right, top, bottom = -10, 15, 5, -5
        # ~L~E~[~F~Z~D~[~[____~B
        pts = np.array([[min_x, min_y], [min_x, max_y], [max_x, max_y], [max_x, min_y]])
        pts = self.to_pixel(pts)
        # pts = np.stack([pts_x, pts_y], axis=1).round().astype(np.int32)
        cv2.polylines(bev_img, [pts], isClosed=True, color=(255, 152, 16, 255), thickness=2, lineType=cv2.LINE_AA)

        # ~Q| __~Lx__~R~Ly____~B~R~L~S~]~_~B 30*20
        grid_w, grid_h = 0.2, 0.075

        x0 = np.arange(min_x, max_x, grid_w)
        p1_x, p1_y = x0, np.full_like(x0, min_y)
        p2_x, p2_y = x0, np.full_like(x0, max_y)
        pts1 = np.stack([p1_x, p1_y], axis=1)
        pts2 = np.stack([p2_x, p2_y], axis=1)
        pts1 = self.to_pixel(pts1)
        pts2 = self.to_pixel(pts2)
        for p1, p2 in zip(pts1, pts2):
            cv2.line(bev_img, p1, p2, (255, 152, 16, 250), 1)
        y0 = np.arange(min_y, max_y, grid_h)
        p1_x, p1_y = np.full_like(y0, min_x), y0
        p2_x, p2_y = np.full_like(y0, max_x), y0
        pts1 = np.stack([p1_x, p1_y], axis=1)
        pts2 = np.stack([p2_x, p2_y], axis=1)
        pts1 = self.to_pixel(pts1)
        pts2 = self.to_pixel(pts2)
        for p1, p2 in zip(pts1, pts2):
            cv2.line(bev_img, p1, p2, (255, 152, 16, 250), 1)
        #
        origin_x, origin_y = self.to_pixel(np.array([[0, 0]]))[0]
        cv2.circle(bev_img, [int(round(origin_x)), int(round(origin_y))], 16,
                   (0, 0, 255, 255), -1)

        return bev_img

    def put_text(self, bev_img, name, time_diff):
        coor = self.get_txt_coor(name)
        coor = self.to_pixel(np.array([coor]), False)[0]
        cv2.putText(bev_img, f'{name.strip("cam_")}', coor, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255, 255), 4)
        diff_coor = np.array(coor) + np.array([0, 60])
        str_time_diff = f'{int(time_diff // 1E6)}'
        cv2.putText(bev_img, str_time_diff, diff_coor, cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                    (0, 0, 255, 255), 2)

    def put_acc_text(self, bev_img, acc_lat, acc_lgt):
        coor = self.to_pixel(np.array([[-1, -0.4]]), False)[0]
        if acc_lat is not None and acc_lgt is not None:
            str_acc = f'[{acc_lat:.2f}, {acc_lgt:.2f}]'
            cv2.putText(bev_img, str_acc, coor, cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                        (0, 0, 255, 255), 3)


    def project_multi_img(self, grp, only_check_shape=False):
        bev_img = np.zeros((self.pixel_rows, self.pixel_cols, self.channels), dtype=np.uint8)
        bev_img[:, :, 3] = 0
        for ix in range(len(grp[1])):
            img_path = grp[1][ix]
            bgr_img = cv2.imread(img_path)
            assert bgr_img is not None, f'failed to load image {img_path}'

            name = self.cam_names[ix]
            self.check_shape(name, bgr_img)
            if only_check_shape:
                continue
            bev_img = self.project(bev_img, name, bgr_img)

        bev_img = self.draw_grid(bev_img)
        bev_img = cv2.rotate(bev_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # text
        for ix in range(len(grp[1])):
            self.put_text(bev_img, self.cam_names[ix], grp[2][ix])
        self.put_acc_text(bev_img, grp[3], grp[4])
        return bev_img

    def save_image(self, bev_img, output_path):
        cv2.imwrite(output_path, bev_img)
        log.info(f'dumped {output_path} ')

if __name__ == '__main__':

    # X, Y = np.meshgrid([1, 2], [4, 5])
    # # print(X, Y)
    # print(np.stack([X.ravel(), Y.ravel()], axis=1))
    # plt.scatter(X, Y)
    # plt.show()
    bag_dir = '/data2/ipmchecker/2025-06-19-13-13-55'
    cam_param_files = glob.glob(os.path.join(bag_dir, 'car_*.yaml'))
    if not cam_param_files:
        raise Exception(f'Not found cam param(car_*.yaml) files found in {bag_dir}')
    cam_param_file = cam_param_files[0]
    img_path = '/data2/ipmchecker/2025-06-19-13-13-55/cam_front_left/1750339002908410000.jpg'
    # bev_plot = BevPlot(cam_param_file, 'long')
    # bev_plot.project('cam_front_left', img_path)
    # bev_plot.save_image(f'/tmp/ipm/test_1.jpg')

    out_view_types = ('long', 'around')
    aligner = ImagesTimeAligner(bag_dir, [1, 100], out_view_types)
    for t in out_view_types:
        bev_plot = BevPlot(cam_param_file, t)
        multi_group = aligner.get_multi_group_frames(cam_names_dict[t])
        for grp in multi_group:
            bev_img = bev_plot.project_multi_img(grp)
            bev_plot.save_image(bev_img, f'/tmp/ipm/{t}_{grp[0]}.png')
