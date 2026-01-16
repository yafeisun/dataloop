import yaml
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R


def load_cam_param(yaml_path, v6: bool = False):
    cam_params = {}
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    if v6:
        for sensor in data["sensors"]["camera"]:
            cam_name = sensor["topic"].split("/")[1]
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
            # except ValueError as e:
            #     raise CarYamlMissingError(f'failed to load 6v camera param from {yaml_path} {e}')

    else:
        for sensor in data["sensors"]["camera"]:
            cam_name = sensor["topic"].split("/")[1]
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

    return cam_params


def image_to_bev(img_path, K, D, R, T, bev_size=(8000, 8000), scale=0.1):
    # __~O~V__~N~U~O~X
    img = cv2.imread(img_path)
    img_undist = img
    # cv2.undistort(img, K, D)

    # ~Z~IBEV__~]~[~[__~R~B~H~V~U~L~]~P| ~G__~I
    corners_world = np.array([[-2, 5, 0], [2, 5, 0], [-2, 10, 0], [2, 10, 0]])

    # __~M~H~[~\~]~P| ~G__
    # corners_cam = (R @ corners_world.T + T).T
    corners_cam = (T @ np.vstack([corners_world.T, np.ones(corners_world.shape[0])]))[:3].T

    # ~J~U__~H~[~C~O__~]
    corners_img, _ = cv2.projectPoints(corners_cam, np.eye(3), np.zeros(3), K, None)
    src_pts = corners_img.reshape(4, 2).astype(np.float32)

    # __~WBEV~X| ~D
    dst_pts = np.array([[0, 0], [bev_size[0], 0], [0, bev_size[1]], [bev_size[0], bev_size[1]]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src_pts, dst_pts)
    bev_img = cv2.warpPerspective(img_undist, H, bev_size)

    return bev_img


if __name__ == '__main__':
   img_path = '/data2/dag-l-o/2025-07-15-12-28-43/cam_side_left_front/1752553798095459000.jpg'
   cam_param_path = '/data2/dag-l-o/2025-07-15-12-28-43/car_E371_L6T79ZEE1RD023484__2025-01-12.yaml'
   cam_param = load_cam_param(cam_param_path, v6=False)
   name = 'cam_side_left_front'
   K, D =  cam_param[name]['intrinsic'], cam_param[name]['distCoeff'],
   shape= cam_param[name]['image_shape']
   extrinsic = cam_param[name]['extrinsic']
   R = extrinsic[:3, :3]
   T = extrinsic[:3, 3]
   img = image_to_bev(img_path, K, D, R, extrinsic)
   cv2.imwrite('/tmp/xx.jpg', img)
