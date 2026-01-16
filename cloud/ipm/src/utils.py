import csv
import functools
import hashlib
import logging
import time

from prettytable import PrettyTable
import numpy as np
import cv2
import yaml

from scipy.spatial.transform import Rotation as R
from src import CarYamlMissingError

logging.basicConfig(format=
                    '[%(asctime)s]  %(pathname)-15s[%(lineno)-4d] %(levelname)-7s:%(message)s',
                    level=logging.INFO)

log = logging.getLogger('d2d')
log.setLevel(logging.INFO)


class TimeLog:
    _instance = None
    display = True
    module_time = {}  # {name:[ct,time]}

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def add(self, name, t):
        if not self.display:
            return
        m = self.module_time.get(name)
        if not m:
            self.module_time[name] = [1, t, t]
            # print(self.module_time)
        else:
            self.module_time[name][0] += 1
            self.module_time[name][1] += t
            self.module_time[name][2] = max(self.module_time[name][2], t)

    def logout(self, log_path=None):
        if self.display:
            tb = PrettyTable()
            tb.field_names = ['No.', 'Function', 'Call times', 'Duration(s)', 'Average Duration(ms)',
                              'Max Duration(ms)']
            rows = []
            idx = 1
            sorted_dict = dict(sorted(self.module_time.items(), key=lambda d: d[1][1], reverse=True))
            for k, v in sorted_dict.items():
                rows.append(
                    [idx, f'{k:22}', f'{v[0]:8}', f'{(v[1] / 1000):9.3f}', f'{v[1] / v[0]:14.3f}', f'{v[2]:14.3f}'])
                idx += 1
            if log_path:
                with open(log_path, 'w') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow(tb.field_names)
                    csv_writer.writerows(rows)
            tb.add_rows(rows)
            print(tb)

    # def __del__(self):
    #     self.logout()


def runtime(func=None, **kwargs):
    tml = TimeLog()
    if func is None:
        return functools.partial(runtime, name=kwargs.get('name'))
    func_name = kwargs.get('name', func.__name__)
    @functools.wraps(func)
    def _wrap(*args, **kwgs):
        s = time.time()
        r = func(*args, **kwgs)
        d = (time.time() - s) * 1000
        tml.add(func_name, d)
        # print(f'{func.__name__} duration {d :.3f} ms')
        return r

    return _wrap


v6_param = {
    'cam_names': [
        'cam_front_group_socb',
        'cam_side_left_front_group',
        'cam_side_right_front_group',
        'cam_back_group',
        'cam_side_left_back_group',
        'cam_side_right_back_group'
    ],
    'need_detection': ['cam_front_group_socb', 'cam_back_group']
}

v11_param = {
    'cam_names': [
        'cam_back', 'cam_front_right', 'cam_front_left', 'cam_side_left_front',
        'cam_side_right_front', 'cam_side_left_back', 'cam_side_right_back'
    ],
    'need_detection': ['cam_front_right', 'cam_back']
}


def get_cam_names(v6: bool):
    return v6_param['cam_names'] if v6 else v11_param['cam_names']


def get_need_detection(v6: bool):
    return v6_param['need_detection'] if v6 else v11_param['need_detection']


def get_front_cam_name(v6: bool):
    return get_need_detection(v6)[0]


def get_back_cam_name(v6: bool):
    return get_need_detection(v6)[1]


def load_cam_param(yaml_path, v6: bool = False):
    cam_params = {}
    with open(yaml_path, 'rb') as f:
        f_data = f.read()
        log.info(f'yaml md5=\'{hashlib.md5(f_data).hexdigest()}\', path: {yaml_path}')
        data = yaml.safe_load(f_data)
    if v6:
        for sensor in data["sensors"]["camera"]:
            cam_name = sensor["topic"].split("/")[1]
            try:
                ext = sensor['calibration']["CameraExt"]
                x, y, z = ext['x'], ext['y'], ext['z']
                rpy = np.array([ext['roll'], ext['pitch'], ext['yaw']])
                rotation = R.from_euler('xyz', rpy, degrees=False)
                transformation = np.eye(4)
                transformation[:3, :3] = rotation.as_matrix()
                transformation[:3, 3] = np.array([x, y, z])
                cam_params[cam_name] = {
                    "intrinsic": np.array(sensor['calibration']["CameraIntMat"], dtype=np.float64).reshape(3, 3),
                    "extrinsic": transformation,
                    "distCoeff": np.array(sensor['calibration']["DistCoeff"], dtype=np.float64).reshape(1, 8),
                    "image_shape": np.array(sensor['calibration']["ImageSize"], dtype=np.int32),
                }
            except ValueError as e:
                raise CarYamlMissingError(f'failed to load 6v camera param from {yaml_path} {e}')
    else:
        for sensor in data["sensors"]["camera"]:
            cam_name = sensor["topic"].split("/")[1]
            try:
                ext = sensor['calibration']["CameraExt"]
                x, y, z = ext['x'], ext['y'], ext['z']
                rpy = np.array([ext['roll'], ext['pitch'], ext['yaw']])
                rotation = R.from_euler('xyz', rpy, degrees=False)
                transformation = np.eye(4)
                transformation[:3, :3] = rotation.as_matrix()
                transformation[:3, 3] = np.array([x, y, z])
                cam_params[cam_name] = {
                    "intrinsic": np.array(sensor['calibration']["CameraIntMat"], dtype=np.float64).reshape(3, 3),
                    "extrinsic": transformation,
                    "distCoeff": np.array(sensor['calibration']["DistCoeff"], dtype=np.float64).reshape(1, 4),
                    "image_shape": np.array(sensor['calibration']["ImageSize"], dtype=np.int32),
                }
            except ValueError as e:
                raise CarYamlMissingError(f'failed to load 11v camera param from {yaml_path} {e}')
    return cam_params

@runtime(name='image loading')
def load_image(image_path):
    return cv2.imread(image_path)

class AntiDistortion:
    @runtime(name='image anti distortion init')
    def __init__(self, intrinsic, dist_coeff, shape):
        K, D = intrinsic, dist_coeff
        w, h = shape
        # new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w, h), None)
        alpha = 0.3
        self.new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha)
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), self.new_K, (w, h), cv2.CV_16SC2)

    @runtime(name='image anti distortion')
    def anti(self, img: np.dtype):
        return cv2.remap(img, self.map1, self.map2, interpolation=cv2.INTER_LINEAR, \
                         borderMode=cv2.BORDER_CONSTANT)
    
def get_vanishing_point(intrinsic, extrinsic):
    vp = np.array([[1e32, 0, 0, 1]])
    np.linalg.inv(extrinsic)
    extrinsic_inv = np.linalg.inv(extrinsic)
    pix_line = np.dot(extrinsic_inv, vp.T)
    # print(hom_pc)
    pix_line = np.dot(intrinsic, pix_line[:3])
    pix_line[0, :] = pix_line[0, :] / pix_line[2, :]
    pix_line[1, :] = pix_line[1, :] / pix_line[2, :]
    return pix_line.T[0][:2].astype(int).tolist()


def calc_iou(box1, box2):
    '''
    box1: xyxy
    box2: xyxy
    '''

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1)
    area2 = box_area(box2)
    lt_x = max(box1[0], box2[0])
    lt_y = max(box1[1], box2[1])
    rb_x = min(box1[2], box2[2])
    rb_y = min(box1[3], box2[3])
    inter = max(rb_x - lt_x, 0) * max(rb_y - lt_y, 0)
    return inter / (area1 + area2 - inter)

class TransMatrix:
    def __init__(self):
        self.T = np.eye(4)

    def rotate(self, quat):
        rotation = R.from_quat(quat).as_matrix()
        self.T[:3, :3] = rotation
        return self

    def rotate_matrix(self, rotation):
        self.T[:3, :3] = rotation
        return self

    def trans(self, xyz):
        self.T[:3, 3] = xyz
        return self

    def inv(self):
        return np.linalg.inv(self.T)


class PointsTrans:
    def __init__(self, intrin, extrin, shape):
        self.intrin = intrin
        self.extrin = extrin
        self.extrin_inv = np.linalg.inv(extrin)
        self.intrin_inv = np.linalg.inv(intrin)
        self.shape = shape
        # @runtime(name='pixel to bev')
    # def pixel_to_bev1(self, pix_points):
    #     '''
    #     pix_points: [pt1, pt2, pt3..] np.array nx2
    #     front_length: ~G__x~L~C~[  -front_length<x< front_length
    #     '''
    #     if not pix_points.size:
    #         return np.array([])
    #     # fx, fy = self.intrin[0, 0], self.intrin[1, 1]
    #     # cx, cy = self.intrin[0, 2], self.intrin[1, 2]
    #     # x1 = (pix_points[:, 0] - cx) / fx
    #     # y1 = (pix_points[:, 1] - cy) / fy
    #     hom_points = np.hstack((pix_points[:, :2], np.ones((pix_points.shape[0], 1))))
    #     pc = np.dot(self.intrin_inv, hom_points.T)
    #     x1, y1 = pc[:2]
    #     z = -self.extrin[2, 3] / (self.extrin[2, 0] * x1 + self.extrin[2, 1] * y1 + self.extrin[2, 2])
    #     pc = np.array([x1 * z, y1 * z, z])
    #     hom_pc = np.vstack((pc, np.ones((1, pc.shape[1]))))
    #     bev_points = np.dot(self.extrin, hom_pc).T[:, :2]
    #     return bev_points

    # def multi_pixel_to_bev(self, pix_points):
    #     pts = []
    #     for z in np.linspace(0, 3, 13):
    #         pts.extend(self.pixel_to_bev(pix_points, z))
    #     return np.array(pts)
    @runtime(name='pixel to bev')
    def pixel_to_bev(self, pix_points, Z=0):
        '''
        pix_points: [pt1, pt2, pt3..] np.array nx2
        front_length: ~G__x~L~C~[  -front_length<x< front_length
        '''
        if not pix_points.size:
            return np.array([]), pix_points
        hom_points = np.hstack((pix_points[:, :2], np.ones((pix_points.shape[0], 1))))
        pc = np.dot(self.intrin_inv, hom_points.T)  # pc/z

        z = np.array([Z]) - self.extrin[2, 3] / np.dot(self.extrin[2, :3], pc)
        pc *= z
        idxs = pc[2, :] > 0
        pc = pc[:, idxs]
        pix_points = pix_points[idxs]
        hom_pc = np.vstack((pc, np.ones((1, pc.shape[1]))))
        bev_points = np.dot(self.extrin, hom_pc)
        # bev_points = bev_points[:, bev_points[0, :] > 0]

        return bev_points[:2].T, pix_points

    @runtime(name='bev to pixel')
    def bev_to_pixel(self, points):
        if not len(points):
            return np.array([])
        hom_points = np.hstack((points[:, :2], np.ones((points.shape[0], 2))))
        hom_points[:, 2] = 0  # nx4
        hom_points = hom_points.T  # 4xn
        points_pc = np.dot(self.extrin_inv, hom_points)
        pts = np.dot(self.intrin, points_pc[:3, :])
        pts = pts[:, pts[2, :] > 0]
        pts = pts[:, pts[0, :] > 0]
        pts = pts[:, pts[1, :] > 0]
        pts[0, :] = pts[0, :] / pts[2, :]
        pts[1, :] = pts[1, :] / pts[2, :]
        pts = pts[:, pts[0, :] < self.shape[0]]
        pts = pts[:, pts[1, :] < self.shape[1]]
        return pts.T[:, :2]
    def bev_to_pixel_mask(self, points):
        if not len(points):
            return np.array([])
        hom_points = np.hstack((points[:, :2], np.ones((points.shape[0], 2))))
        hom_points[:, 2] = 0  # nx4
        hom_points = hom_points.T  # 4xn
        points_pc = np.dot(self.extrin_inv, hom_points)
        pts = np.dot(self.intrin, points_pc[:3, :])

        pts[0, :] = pts[0, :] / pts[2, :]
        pts[1, :] = pts[1, :] / pts[2, :]

        mask = np.ones(pts.shape[1], dtype=bool)
        mask = np.logical_and(mask, pts[2, :] > 0)
        mask = np.logical_and(mask, pts[1, :] > 0)
        mask = np.logical_and(mask, pts[0, :] > 0)
        mask = np.logical_and(mask, pts[0, :] < self.shape[0] - 1)
        mask = np.logical_and(mask, pts[1, :] < self.shape[1] - 1)
        uv = pts[:2, :]
        # uv = np.round(uv).astype(np.int32)
        return uv, mask
    @classmethod
    def filter_in_range(cls, points, x_range: tuple = (-20, 20), y_range: tuple = (-6, 6)):
        '''
        points: np.array nx2
        x_range: (-30, 30)
        y_range: (-10, 10)
        return np.array nx2
        '''
        cut_points = points
        if x_range:
            mask = (cut_points[:, 0] < x_range[1]) & (cut_points[:, 0] > x_range[0])
            cut_points = cut_points[mask]
        if y_range:
            mask = (cut_points[:, 1] < y_range[1]) & (cut_points[:, 1] > y_range[0])
            cut_points = cut_points[mask]
        return cut_points
