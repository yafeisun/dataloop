import os, sys, io
import rospy
import rosbag
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from foxglove_msgs.msg import CompressedVideo
import av
import cv2
from tqdm import tqdm
import multiprocessing as mp
import gc
import json
from scipy.spatial.transform import Rotation as R
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

image_quality = 85
cam_topics = [
    "/CompressedVideo_0",
    "/CompressedVideo_1",
    "/CompressedVideo_2",
    "/CompressedVideo_3",
    "/CompressedVideo_4",
    "/CompressedVideo_5",
    "/CompressedVideo_6",
    "/CompressedVideo_7",
    "/CompressedVideo_8",
    "/CompressedVideo_9",
    "/CompressedVideo_10"
]

cam_topics_2_perspective = {
    "/CompressedVideo_0": "cam_front_right",
    "/CompressedVideo_1": "cam_front_left",
    "/CompressedVideo_2": "cam_back",
    "/CompressedVideo_3": "cam_side_left_front",
    "/CompressedVideo_4": "cam_side_right_front",
    "/CompressedVideo_5": "cam_side_left_back",
    "/CompressedVideo_6": "cam_side_right_back",
    "/CompressedVideo_7": "cam_around_front",
    "/CompressedVideo_8": "cam_around_back",
    "/CompressedVideo_9": "cam_around_left",
    "/CompressedVideo_10": "cam_around_right"
}




class H265Decoder:
    def __init__(self):
        self.container = None

    def decode_h265_frame(self, h265_data):
        """
        解码单个H.265编码的帧。
        :param h265_data: 包含H.265编码数据的字节串
        :return: 解码后的OpenCV格式图像
        """

        video_data = io.BytesIO()

        for frame_data in h265_data:
            video_data.write(frame_data)
        del h265_data
        gc.collect()

        print("Creating container...")
        self.container = av.open(video_data, format="hevc")
        del video_data
        gc.collect()
        stream = self.container.streams.video[0]
        fps = stream.base_rate
        frame_width = stream.width
        frame_height = stream.height

        frames = []
        print("Decoding frames...")
        for packet in self.container.demux():
            for frame in packet.decode():
                if isinstance(frame, av.VideoFrame):
                    img = frame.to_ndarray(format='bgr24')
                    frames.append(img)
        return frames


def create_save_paths(save_path, topics):
    return {topic: create_save_path(save_path, topic) for topic in topics}


def create_save_path(save_path, topic):
    path = os.path.join(save_path, cam_topics_2_perspective[topic])
    if os.path.exists(path):
        print(f"已处理过bag: {save_path} (输出目录已存在: {path})")
        exit(1)

    if not os.path.exists(path):
        os.makedirs(path)
    return path


def decode_H265_and_save(data, frame_name):
    container = av.open(io.BytesIO(data), format="hevc")

    for packet in container.demux():
        for frame in packet.decode():
            img = frame.to_ndarray(format="bgr24")
            cv2.imwrite(frame_name, img)


def decode_H265_images(bag_path, cam_topic, cam_save_path):
    decoder = H265Decoder()
    h265_data_all = []
    timestamps = []

    with rosbag.Bag(bag_path, "r") as bag:
        for topic, msg, t in bag.read_messages(topics=[cam_topic]):
            timestamp = msg.viTimestamp
            h265_data_all.append(msg.data)
            timestamps.append(timestamp)

            if len(h265_data_all) >= 200:
                print(cam_topic)
                frames = decoder.decode_h265_frame(h265_data_all)
                print("saving images...")
                print(
                    f"len(h265_data_all): {len(h265_data_all)}, len(timestamps): {len(timestamps)}, len(frames): {len(frames)}")
                for frame, timestamp in zip(frames, timestamps):
                    formatted_timestamp = f"{timestamp:0<19}"
                    cv2.imwrite(os.path.join(cam_save_path, f"{formatted_timestamp}.jpeg"), frame,
                                [cv2.IMWRITE_JPEG_QUALITY, image_quality])

                # Explicitly free memory
                del h265_data_all, timestamps, frames
                h265_data_all = []
                timestamps = []
                gc.collect()

        if len(h265_data_all) > 0:
            print(cam_topic)
            frames = decoder.decode_h265_frame(h265_data_all)
            print("saving images...")
            print(
                f"len(h265_data_all): {len(h265_data_all)}, len(timestamps): {len(timestamps)}, len(frames): {len(frames)}")
            for frame, timestamp in zip(frames, timestamps):
                formatted_timestamp = f"{timestamp:0<19}"
                cv2.imwrite(os.path.join(cam_save_path, f"{formatted_timestamp}.jpeg"), frame,
                            [cv2.IMWRITE_JPEG_QUALITY, image_quality])

            # Explicitly free memory
            del h265_data_all, timestamps, frames
            gc.collect()


def process(bag_path, save_path, cam_topics):
    cam_save_paths = {}
    with rosbag.Bag(bag_path, "r") as bag:
        for cam_topic in cam_topics:
            info = bag.get_type_and_topic_info(topic_filters=[cam_topic])
            if not info.topics:
                continue
            cam_save_path = create_save_path(save_path, cam_topic)
            cam_save_paths[cam_topic] = cam_save_path
            print("available: ", info.topics.keys())
            topic_count = info.topics[cam_topic].message_count
            print(f"Extracting {cam_topic}...", "message count:", topic_count, "...")

    if len(cam_save_paths) <= 0:
        print(f"[Info] task num is zero")
        return

    # 线程池并行处理
    max_workers = min(len(cam_save_paths), 4)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for cam_topic, cam_save_path in cam_save_paths.items():
            future = executor.submit(decode_H265_images, bag_path, cam_topic, cam_save_path)
            futures.append((cam_topic, future))
        for cam_topic, future in futures:
            try:
                future.result()
                print(f"{cam_topic} task exec finished")
            except Exception as e:
                print(f"{cam_topic} task exec failed: {str(e)}")


def process_bag_files(bag_file, output_image_dir, cam_topics):
    """
    处理input_dir中的所有bag文件，将结果输出到output_image_dir中对应的文件夹
    """
    if not os.path.exists(bag_file):
        print(f"bag 不存在: {bag_file}")
        return

    print(f"正在处理: {bag_file}")
    try:
        process(str(bag_file), output_image_dir, cam_topics)
        print(f"成功处理: {bag_file} -> {output_image_dir}")
    except Exception as e:
        print(f"处理文件 {bag_file} 时出错: {str(e)}")


if __name__ == "__main__":
    """
    用法:
    python3 process_bag.py <rosbag文件> xx
    
    示例:
    python3 process_bag.py xxx.bag xx
    """
    if len(sys.argv) < 2:
        print("Usage: python3 process_bag.py <xxx.bag> 7v")
        sys.exit(1)

    input_bag_file = sys.argv[1]
    bag_root_dir = os.path.dirname(os.path.abspath(input_bag_file))
    output_image_dir = os.path.join(bag_root_dir)
   

    # parsing_type = sys.argv[2]
    # if parsing_type == "7v":
    #     cam_topics = cam_topics_7v
    # elif parsing_type == "5v":
    #     cam_topics = cam_topics_5v
    # else:
    #     print("input camera config incorrect")
    #     exit(1)

    process_bag_files(input_bag_file, output_image_dir, cam_topics)
    
