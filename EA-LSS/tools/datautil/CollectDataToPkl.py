import numpy as np
import os
import json
import pickle
import copy
import yaml
import open3d as o3d

# ==================== 配置常量 ====================

colorMap = [
    (0, 0, 255), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), 
    (0, 255, 255), (125, 0, 0), (0, 125, 0), (0, 0, 125), (125, 125, 0), 
    (0, 0, 0),  (127, 0, 255), (255, 127, 0), (128, 255, 0)
]

class_names = {
   'car':0, 'truck':1, 'bus':2, 'engineering_vehicle':3, 'person':4, 'pillar':5, 
   'cyclist':6, 'traffic_cone':7, 'noparking_pillar':8,'water_horse':9 ,'gate_on':10 ,
   'gate_off':11, 'anti_collision_bucket':12 ,'noparking_board':13
}

camMap = {
    "middle":"LIDAR",
    "cam_front_right":'CAM_FRONT',
    "cam_back":'CAM_BACK',
    "cam_side_left_front":"CAM_FRONT_LEFT",
    "cam_side_right_front":"CAM_FRONT_RIGHT",
    "cam_side_left_back":'CAM_BACK_LEFT',
    "cam_side_right_back":'CAM_BACK_RIGHT',
    "cam_front_left":'CAM_FRONT_FAR',
    "cam_around_front":'CAM_A_FRONT',
    "cam_around_left":'CAM_A_LEFT',
    "cam_around_right":'CAM_A_RIGHT',
    "cam_around_back":'CAM_A_BACK'
}

# ==================== 几何变换函数 ====================

def rotation_3d_in_axis(corners, yaw):
    """绕Z轴旋转3D角点"""
    rot_sin = np.sin(yaw)
    rot_cos = np.cos(yaw)
    ones    = np.ones_like(rot_sin)
    zeros   = np.zeros_like(rot_sin)

    rot_mat_T = np.stack([
        np.stack([rot_cos, rot_sin, zeros]),
        np.stack([-rot_sin, rot_cos,  zeros]),
        np.stack([zeros,   zeros,    ones])
    ])

    return np.einsum('aij, jka->aik', corners, rot_mat_T)


def get_corners(boxes):
    """从3D边界框获取8个角点"""
    if boxes is None:
        return [[]]
    corners_norm = np.stack(np.unravel_index(np.arange(8), [2]*3), axis=1)
    corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array([0.5, 0.5, 0.5])
    corners = boxes[:, 3:6].reshape([-1, 1, 3]) * corners_norm.reshape([1, 8, 3])
    corners = rotation_3d_in_axis(corners, boxes[:, -1])
    corners += boxes[:, :3].reshape(-1, 1, 3)
    return corners


def euler_to_rotation_matrix(rot):
    """欧拉角转旋转矩阵"""
    phi, theta, psi = rot
    
    def Rx(alpha):
        return np.array([[1, 0, 0],
                         [0, np.cos(alpha), -np.sin(alpha)],
                         [0, np.sin(alpha), np.cos(alpha)]])
    
    def Ry(beta):
        return np.array([[np.cos(beta), 0, np.sin(beta)],
                         [0, 1, 0],
                         [-np.sin(beta), 0, np.cos(beta)]])
    
    def Rz(gamma):
        return np.array([[np.cos(gamma), -np.sin(gamma), 0],
                         [np.sin(gamma), np.cos(gamma), 0],
                         [0, 0, 1]])

    R = Rz(psi).dot(Ry(theta)).dot(Rx(phi))
    return R


def quaternion_to_rotation_matrix(q):
    """四元数转旋转矩阵"""
    norm = np.linalg.norm(q)
    if np.abs(norm - 1.0) > 1e-6:
        q = q / norm
    
    x, y, z, w = q
    R = np.array([
        [1 - 2*y**2 - 2*z**2,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [    2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2,     2*y*z - 2*x*w],
        [    2*x*z - 2*y*w,     2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])
    
    return R


def rotation_matrix_to_euler_angles(R):
    """旋转矩阵转欧拉角"""
    if R.shape != (3, 3):
        raise ValueError("Input must be a 3x3 rotation matrix.")

    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0.0

    return x, y, z


def points_to_3d_bounding_box(points):
    """从点云计算3D边界框"""
    if points.shape != (8, 3):
        raise ValueError("Input points must be an array of shape (8, 3).")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    oriented_bounding_box = pcd.get_oriented_bounding_box()

    center = oriented_bounding_box.center
    extent = oriented_bounding_box.extent
    R = oriented_bounding_box.R

    return center, extent, R

# ==================== PCD文件操作函数 ====================

def read_pcd_binary(pcd_file):
    """读取二进制PCD文件"""
    with open(pcd_file, 'rb') as f:
        header = ""
        while True:
            line = f.readline().decode('utf-8')
            header += line
            if line.startswith("DATA"):
                break
        points = np.fromfile(f, dtype=np.float32).reshape(-1, 4)
    return points, header


def write_binary_pcd(filename, point_cloud, header_lines):
    """写入二进制PCD文件"""
    with open(filename, 'wb') as f:
        f.write((header_lines).encode())
        point_cloud.tofile(f)

# ==================== 传感器配置解析函数 ====================

def getSensorIntri(sensorA):
    """获取相机内参"""
    Intri = dict()
    for sensor in sensorA:
        sensorName = camMap[sensor['topic'].split('/')[1]]
        Intri[sensorName] = np.array(sensor['calibration']['CameraIntMat']).reshape(3, 3)
    return Intri


def getSensorDistort(sensorA):
    """获取相机畸变参数"""
    CamDistort = dict()
    for sensor in sensorA:
        sensorName = camMap[sensor['topic'].split('/')[1]]
        CamDistort[sensorName] = np.array(sensor['calibration']['DistCoeff'])
    return CamDistort


def getSensorSize(sensorA):
    """获取相机图像尺寸"""
    CamSize = dict()
    for sensor in sensorA:
        sensorName = camMap[sensor['topic'].split('/')[1]]
        CamSize[sensorName] = np.array(sensor['calibration']['ImageSize']).reshape(2)
    return CamSize


def getSensorEx(sensorA):
    """获取传感器外参"""
    RTA = dict()
    for sensor in sensorA:
        sensorN = sensor['topic'].split('/')[1]
        if sensorN not in camMap:
            continue
        sensorName = camMap[sensor['topic'].split('/')[1]]
        if 'CameraExt' in sensor['calibration']:
            rot = [sensor['calibration']['CameraExt']['roll'],
                   sensor['calibration']['CameraExt']['pitch'],
                   sensor['calibration']['CameraExt']['yaw']]
            T = [sensor['calibration']['CameraExt']['x'],
                 sensor['calibration']['CameraExt']['y'],
                 sensor['calibration']['CameraExt']['z']]
            R = euler_to_rotation_matrix(rot)
            RT = np.eye(4)
            RT[0:3, 0:3] = R
            RT[0:3, 3] = T
            RTA[sensorName] = RT
        else:
            rot = [sensor['calibration']['roll'],
                   sensor['calibration']['pitch'],
                   sensor['calibration']['yaw']]
            T = [sensor['calibration']['x'],
                 sensor['calibration']['y'],
                 sensor['calibration']['z']]
            R = euler_to_rotation_matrix(rot)
            RT = np.eye(4)
            RT[0:3, 0:3] = R
            RT[0:3, 3] = T
            RTA[sensorName] = RT
    return RTA

# ==================== 可视化函数 ====================

def visualize_box_in_pcd(boxes, labels, points, pointA, Pcd_color=None):
    """在点云中可视化3D边界框"""
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="box in pcd")
    opt = vis.get_render_option()
    opt.point_size = 2
    opt.background_color = np.asarray([1, 1, 1])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    corners = get_corners(boxes)
    for corner, label in zip(corners, labels):
        color = np.array(colorMap[class_names[label]]) / 255.0
        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7],
            [4, 6], [5, 7]
        ]
        line_colors = np.tile(color, len(lines)).reshape(-1, 3)
        line_set = o3d.geometry.LineSet()
        line_set.lines = o3d.utility.Vector2iVector(lines) 
        line_set.colors = o3d.utility.Vector3dVector(line_colors)
        line_set.points = o3d.utility.Vector3dVector(corner)
        vis.add_geometry(line_set)

    FOR1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[0, 0, 0])

    vis.add_geometry(pcd)
    vis.add_geometry(FOR1)

    vis.run()
    vis.destroy_window()

# ==================== 配置加载函数 ====================

def load_sensor_config(bagPath):
    """加载传感器配置"""
    inputPathList = os.listdir(bagPath)
    CarYaml = None
    for ind in inputPathList:
        if('car' in ind) and ('.yaml' in ind):
            CarYaml = ind
    
    if CarYaml is None:
        print("car.yaml not found!")
        return None
    
    with open(os.path.join(bagPath, CarYaml)) as file:
        configSensor = yaml.safe_load(file)['sensors']
        configSensor = dict(configSensor)
    
    return configSensor


def get_sensor_params(configSensor):
    """获取传感器参数"""
    SensorToVcs = getSensorEx(configSensor['lidar'])
    SensorToVcs.update(getSensorEx(configSensor['camera']))
    CamIntri = getSensorIntri(configSensor['camera'])
    CamDistort = getSensorDistort(configSensor['camera'])
    CamSize = getSensorSize(configSensor['camera'])
    
    return SensorToVcs, CamIntri, CamDistort, CamSize


# ==================== 同步数据加载函数 ====================

def load_sync_sensors(BasePath):
    """加载传感器同步数据"""
    try:
        with open(os.path.join(BasePath, "sync_sensors.txt"), 'r', encoding='utf-8') as file:
            lines = file.readlines()
    except:
        return None
    
    if len(lines) == 0:
        return None
    
    Sensor = lines[0].strip().split(' ')
    SenSyncAll = dict()
    
    for index in range(1, len(Sensor)):
        name = Sensor[index].split('/')[1]
        SensorName = camMap[name]
        pathSync = os.path.join(BasePath, name, "sync_sensors.txt")
        
        if not os.path.exists(pathSync):
            continue
        
        with open(pathSync, 'r', encoding='utf-8') as fileS:
            linesync = fileS.readlines()
            SenSync = dict()
            for ll in range(1, len(linesync)):
                timeSync = linesync[ll].strip().split(' ')
                if len(timeSync) != 9:
                    continue
                SenSync[int(timeSync[0])] = [
                    timeSync[1],
                    [float(timeSync[2]), float(timeSync[3]), float(timeSync[4]),
                     float(timeSync[5]), float(timeSync[6]), float(timeSync[7]), 
                     float(timeSync[8])]
                ]
            SenSyncAll[SensorName] = SenSync
    
    if len(SenSyncAll) != 12:
        return None
    
    return lines, SenSyncAll, Sensor


# ==================== 边界框处理函数 ====================

def parse_bbox_from_json(gtBoxPath):
    """从JSON文件解析边界框"""
    with open(gtBoxPath, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    BoxA = []
    pointA = []
    BoxLabel = []
    
    for indBox in range(data['num_boxes']):
        BoxT = data['boxes'][indBox]
        point = np.ones([8, 3])
        point[0, :] = [BoxT['p_back_left_bottom']['x'], BoxT['p_back_left_bottom']['y'], 
                        BoxT['p_back_left_bottom']['z']]
        point[1, :] = [BoxT['p_back_right_bottom']['x'], BoxT['p_back_right_bottom']['y'], 
                        BoxT['p_back_right_bottom']['z']]
        point[2, :] = [BoxT['p_front_right_bottom']['x'], BoxT['p_front_right_bottom']['y'], 
                        BoxT['p_front_right_bottom']['z']]
        point[3, :] = [BoxT['p_front_left_bottom']['x'], BoxT['p_front_left_bottom']['y'], 
                        BoxT['p_front_left_bottom']['z']]
        point[4, :] = [BoxT['p_back_left_top']['x'], BoxT['p_back_left_top']['y'], 
                        BoxT['p_back_left_top']['z']]
        point[5, :] = [BoxT['p_back_right_top']['x'], BoxT['p_back_right_top']['y'], 
                        BoxT['p_back_right_top']['z']]
        point[6, :] = [BoxT['p_front_right_top']['x'], BoxT['p_front_right_top']['y'], 
                        BoxT['p_front_right_top']['z']]
        point[7, :] = [BoxT['p_front_left_top']['x'], BoxT['p_front_left_top']['y'], 
                        BoxT['p_front_left_top']['z']]
        
        center, extent, R = points_to_3d_bounding_box(point)
        pointA.append(point)
        x, y, z = rotation_matrix_to_euler_angles(R)
        Box = [center[0], center[1], center[2], extent[0], extent[1], extent[2], z]
        BoxA.append(Box)
        BoxLabel.append('car')
    
    BoxTime = data['timestamp']
    BoxA = np.array(BoxA)
    BoxLabel = np.array(BoxLabel)
    
    return BoxA, BoxLabel, BoxTime

# ==================== 帧数据处理函数 ====================

def process_lidar_data(SensorName, BasePath, name, SensorToVcs, SenSyncAll, frame):
    """处理激光雷达数据"""
    Lidar_path = os.path.join(BasePath, name, str(frame) + '.pcd')
    
    if not os.path.exists(Lidar_path):
        print("file not exists:", Lidar_path)
        return None
    
    Lidar_path_new = os.path.join(BasePath, name, str(frame)+'_new.pcd')
    points, header = read_pcd_binary(Lidar_path)
    
    tmp = points[:, 3].copy()
    points[:, 3] = 1.
    points[:, 3] = tmp
    
    part = {}
    part['lidar_path'] = Lidar_path
    part['lidar2ego_rotation'] = SensorToVcs[SensorName][0:3, 0:3]
    part['lidar2ego_translation'] = SensorToVcs[SensorName][0:3, 3]
    part['ego2global_rotation'] = quaternion_to_rotation_matrix(SenSyncAll[SensorName][frame][1][3:7])
    part['ego2global_translation'] = SenSyncAll[SensorName][frame][1][0:3]
    part['timestamp'] = int(SenSyncAll[SensorName][frame][0].split('.')[0])
    
    return part


def process_camera_data(SensorName, name, InputPathClip, SenSyncAll, frame, SensorToVcs, CamIntri, CamDistort):
    """处理相机数据"""
    part = {}
    
    if SenSyncAll[SensorName][frame][0] == "null":
        return None
    
    part['cams'] = {}
    part['cams'][SensorName]['data_path'] = os.path.join(
        InputPathClip, name, 
        SenSyncAll[SensorName][frame][0].split('.')[0] + '.jpg'
    )
    part['cams'][SensorName]['data_path'] = part['cams'][SensorName]['data_path'].replace("/clips/" + name, "")
    
    if not os.path.exists(part['cams'][SensorName]['data_path']):
        print("file not exists:", part['cams'][SensorName]['data_path'])
        return None
    
    part['cams'][SensorName]['sensor2ego_rotation'] = SensorToVcs[SensorName][0:3, 0:3]
    part['cams'][SensorName]['sensor2ego_translation'] = SensorToVcs[SensorName][0:3, 3]
    part['cams'][SensorName]['lidar2cam'] = np.linalg.inv(SensorToVcs[SensorName])
    part['cams'][SensorName]['ego2global_rotation'] = quaternion_to_rotation_matrix(SenSyncAll[SensorName][frame][1][3:7])
    part['cams'][SensorName]['ego2global_translation'] = SenSyncAll[SensorName][frame][1][0:3]
    part['cams'][SensorName]['cam_intrinsic'] = CamIntri[SensorName][0:3, 0:3]
    part['cams'][SensorName]['distortion_params'] = CamDistort[SensorName]
    part['cams'][SensorName]['timestamp'] = int(SenSyncAll[SensorName][frame][0].split('.')[0])
    
    return part


def process_single_frame(lineT, Sensor, SenSyncAll, SensorToVcs, CamIntri, CamDistort, CamSize, BasePath, InputPathClip, bbox_dir, ind, part):
    """处理单帧数据"""
    frame = int(lineT)
    flag = 1
    
    # 解析边界框
    gtBoxPath = os.path.join(BasePath, "bbox", str(frame) + '.json')
    BoxA, BoxLabel, BoxTime = parse_bbox_from_json(gtBoxPath)
    
    # 处理每个传感器
    for index in range(1, len(Sensor)):
        name = Sensor[index].split('/')[1]
        SensorName = camMap[name]
        
        if frame not in SenSyncAll[SensorName]:
            flag = 0
            break
        
        if SensorName == "LIDAR":
            lidar_part = process_lidar_data(SensorName, BasePath, name, SensorToVcs, SenSyncAll, frame)
            if lidar_part is None:
                flag = 0
                break
        else:
            camera_part = process_camera_data(SensorName, name, InputPathClip, SenSyncAll, frame, SensorToVcs, CamIntri, CamDistort)
            if camera_part is None:
                flag = 0
                break
    
    if flag == 1:
        part['cams']['imgSize'] = CamSize
        part['cams']['timestamp'] = int(SenSyncAll["CAM_FRONT"][frame][0].split('.')[0])
        part['gt_boxes_st'] = BoxA
        part['gt_Label_st'] = BoxLabel
        part['timestamp_box'] = BoxTime
        part['Occ_ratio'] = 0
        part["bbox_path"] = os.path.join(bbox_dir, f"{ind}.json")
        return part
    
    return None

# ==================== 主要数据收集函数 ====================

def CollectionDataToPkl_bag(baseNus, bagPath):
    """收集bag文件数据并转换为pkl格式"""
    with open(baseNus, 'rb') as f:
        data_nuscence = pickle.load(f)
    
    partBf = copy.deepcopy(data_nuscence['infos'][0]) 
    clip_name = os.path.basename(bagPath)
    
    data_nuscence['infos'] = []
    clip_infos = []
    
    partBf['clip'] = clip_name
    partBf['cams']['CAM_FRONT_FAR'] = copy.deepcopy(partBf['cams']['CAM_FRONT'])
    
    resultPath = os.path.join(bagPath, "result")
    if not os.path.exists(resultPath):
        return None
    
    # 加载传感器配置
    configSensor = load_sensor_config(bagPath)
    if configSensor is None:
        return None
    
    SensorToVcs, CamIntri, CamDistort, CamSize = get_sensor_params(configSensor)
    
    BasePath = os.path.join(resultPath, "test_calibration")
    bbox_dir = os.path.join(BasePath, "bbox")
    
    # 加载同步数据
    sync_result = load_sync_sensors(BasePath)
    if sync_result is None:
        return None
    
    lines, SenSyncAll, Sensor = sync_result
    
    # 处理每一帧
    for ind in range(1, len(lines)):
        part = copy.deepcopy(partBf)
        lineT = lines[ind].strip().split(' ')
        frame_data = process_single_frame(
            lineT, Sensor, SenSyncAll, SensorToVcs, CamIntri, CamDistort, 
            CamSize, BasePath, bagPath, bbox_dir, ind, part
        )
        
        if frame_data is not None:
            clip_infos.append(frame_data)
    
    print("len:", len(clip_infos))
    
    if len(clip_infos) >= 38:
        data_nuscence["infos"].extend(clip_infos)
    
    print("len x:", len(data_nuscence['infos']))
    
    if len(data_nuscence['infos']) == 0:
        return None
    
    return data_nuscence