# mycode/utils/io.py
import os
from PIL import Image
import numpy as np

def load_rgbd_sequence(folder):
    rgb_list, depth_list = [], []
    all_files = sorted(os.listdir(folder))
    color_files = [f for f in all_files if f.endswith(".color.png")]

    for f in color_files:
        base = f.replace(".color.png", "")
        rgb_path = os.path.join(folder, f)
        depth_path = os.path.join(folder, base + ".depth.png")

        rgb = Image.open(rgb_path).convert("RGB")
        depth = Image.open(depth_path)
        depth_np = np.array(depth).astype(np.float32) / 1000.0  # mm -> meters

        rgb_list.append(rgb)
        depth_list.append(depth_np)

    return rgb_list, depth_list


def load_color_files(folder):
    all_files = sorted(os.listdir(folder))
    color_files = [os.path.join(folder, f) for f in all_files if f.endswith(".color.png")]

    return color_files


def load_pose(pose_file):
    with open(pose_file, 'r') as f:
        lines = f.readlines()
        pose = np.array([[float(val) for val in line.strip().split()] for line in lines])  # 4x4
    return pose
