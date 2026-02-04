import numpy as np
import pickle
import json
import os
import shutil
import copy
import cv2

# ==================== 配置常量 ====================

cam_convert_dict = {
    'CAM_FRONT': 'CAM__front', 
    'CAM_FRONT_RIGHT': 'CAM__front_right', 
    'CAM_FRONT_LEFT': 'CAM__front_left', 
    'CAM_BACK': 'CAM__back', 
    'CAM_BACK_LEFT': 'CAM__back_left', 
    'CAM_BACK_RIGHT': 'CAM__back_right', 
    'CAM_A_FRONT': 'CAM__a_front', 
    'CAM_A_BACK': 'CAM__a_back', 
    'CAM_A_LEFT': 'CAM__a_left', 
    'CAM_A_RIGHT': 'CAM__a_right',
    'CAM_FRONT_FAR': 'CAM__front_far'
}

# ==================== 标定数据处理函数 ====================

def get_calib_data(frame_data):
    """获取标定数据"""
    calib_data = {}

    for cam in frame_data["cams"].keys():
        if not "CAM" in cam:
            continue
        
        cam_value = cam_convert_dict[cam]
        temp = {}
        temp["type"] = "fisheye Equidistant"
        
        intr = frame_data["cams"][cam]["cam_intrinsic"].tolist()
        extr = frame_data["cams"][cam]["lidar2cam"].tolist()
        distort = frame_data["cams"][cam]["distortion_params"]
        
        D = np.zeros(8)
        D[:4] = distort
        D = D.tolist()
        
        imgSize = frame_data["cams"]["imgSize"][cam].tolist()
        
        temp["intr"] = intr
        temp["D"] = D
        temp["extr"] = extr
        temp["img_height"] = imgSize[0]
        temp["img_width"] = imgSize[1]
        
        calib_data[cam_value] = temp

    lidar2ego_rotation = frame_data["lidar2ego_rotation"]
    lidar2ego_translation = frame_data["lidar2ego_translation"]
    
    lidar2ego_T = np.eye(4, 4)
    lidar2ego_T[:3, :3] = lidar2ego_rotation
    lidar2ego_T[:3, 3] = lidar2ego_translation
    
    calib_data["trans_lidar2ann"] = lidar2ego_T.tolist()
    
    return calib_data


def save_calib_data(calib_data, json_path):
    """保存标定数据到JSON文件"""
    if not os.path.exists(json_path):
        with open(json_path, "w") as f:
            json.dump(calib_data, f, indent=4)

# ==================== 图像处理函数 ====================

def undistort_and_save_img(img_path, cam_value, cali_param, img_save_path):
    """去畸变并保存图像"""
    intrinsic = np.array(copy.deepcopy(cali_param['intr']))
    D = np.array(cali_param['D'])
    
    img_height = cali_param["img_height"]
    img_width = cali_param["img_width"]
    img = cv2.imread(img_path)
    
    if "_a_" in cam_value:
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            intrinsic, D[:4], None, intrinsic, (img_height, img_width), cv2.CV_16SC2
        )
        new_intrinsic = intrinsic
    else:
        alpha = 0.3
        new_intrinsic, validPixROI = cv2.getOptimalNewCameraMatrix(
            intrinsic, D[:4], (img_height, img_width), alpha
        )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            intrinsic, D[:4], None, new_intrinsic, (img_height, img_width), cv2.CV_16SC2
        )

    img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(img_save_path, img)

    cali_param['origin_intr'] = cali_param['intr']
    cali_param['intr'] = new_intrinsic.tolist()

    return cali_param

# ==================== 文件保存函数 ====================

def save_frame_desc_json(frame_path, frame_name, ego2global_rotation, ego2global_translation):
    """保存帧描述JSON文件"""
    save_loc_json = os.path.join(frame_path, frame_name + "__desc" + ".json")
    
    ego2global = np.eye(4)
    ego2global[:3, :3] = ego2global_rotation
    ego2global[:3, 3] = ego2global_translation
    ego2global = ego2global.tolist()
    
    desc = {"desc": {"ego2global": ego2global}}
    
    with open(save_loc_json, 'w') as f:
        json.dump(desc, f, indent=4)


def save_lidar_pcd(frame_path, frame_name, lidar_path):
    """保存激光雷达点云文件"""
    print("Copying PCD:", lidar_path)
    shutil.copy(lidar_path, os.path.join(frame_path, "".join([frame_name, ".pcd"])))


def save_camera_images(frame_path, frame_name, frame_data, calib_data):
    """保存相机图像"""
    for cam in frame_data["cams"].keys():
        if not "CAM_" in cam:
            continue
        
        cam_value = cam_convert_dict[cam]
        cam_path = frame_data["cams"][cam]["data_path"]
        
        img_save_path = os.path.join(
            frame_path, 
            "".join([frame_name, "__", cam_value.replace("CAM__", ""), ".jpg"])
        )
        
        if "_a_" in cam_value:
            shutil.copy(cam_path, img_save_path)
        else:
            calib_data[cam_value] = undistort_and_save_img(
                cam_path, cam_value, calib_data[cam_value], img_save_path
            )
    
    return calib_data

# ==================== 主处理函数 ====================

def process_single_frame(frame_data, save_path, frame_index, clip_name=None):
    """处理单帧数据"""
    boxes = frame_data["gt_boxes"]
    labels = frame_data["gt_names"]
    
    timestamp = frame_data["cams"]["timestamp"]
    frame_name = str(int(timestamp) // 10 ** 6)
    
    # 确定保存路径
    if clip_name:
        clip_index = frame_index // 200
        clip_path = os.path.join(save_path, clip_name + "_" + str(int(clip_index)))
    else:
        clip_path = save_path
    
    calib_json_path = os.path.join(clip_path, "calib.json")
    
    if not os.path.exists(clip_path):
        os.makedirs(clip_path)
    
    frame_path = os.path.join(clip_path, frame_name)
    if not os.path.exists(frame_path):
        os.makedirs(frame_path)
    
    # 保存描述文件
    save_frame_desc_json(
        frame_path, frame_name,
        np.array(frame_data["ego2global_rotation"]),
        np.array(frame_data["ego2global_translation"])
    )
    
    # 保存点云
    save_lidar_pcd(frame_path, frame_name, frame_data["lidar_path"])
    
    # 保存相机图像
    calib_data = get_calib_data(frame_data)
    calib_data = save_camera_images(frame_path, frame_name, frame_data, calib_data)
    save_calib_data(calib_data, calib_json_path)


def process_convert_bag(data_nuscenes, save_path):
    """处理bag数据转换"""
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    data_infos = data_nuscenes["infos"]
    
    for index, frame_data in enumerate(data_infos):
        clip = frame_data["clip"]
        process_single_frame(frame_data, save_path, index, clip)
        print(f"Processed frame {index}/{len(data_infos)}")