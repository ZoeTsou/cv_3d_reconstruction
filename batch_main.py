# batch_main.py
import os
import torch
import gc
from mycode.fast3r_wrapper import Fast3RPredictor
from mycode.utils.io import load_pose
from mycode.utils.pointcloud import transform_to_world, save_point_cloud


def main():
    ROOT_DIR = "7scenes"
    OUTPUT_DIR = "test"

    predictor = Fast3RPredictor(device="cuda")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for scene in sorted(os.listdir(ROOT_DIR)):
        scene_path = os.path.join(ROOT_DIR, scene, "test")
        if not os.path.isdir(scene_path):
            continue

        for seq in sorted(os.listdir(scene_path)):
            seq_path = os.path.join(scene_path, seq)
            if not os.path.isdir(seq_path):
                continue

            print(f"[INFO] Processing {scene}/{seq}...")

            # 收集所有圖片路徑
            image_paths = [
                os.path.join(seq_path, f)
                for f in sorted(os.listdir(seq_path))
                if f.endswith(".color.png")
            ]

            if not image_paths:
                print(f"[WARN] No color images in {seq_path}, skipping.")
                continue

            # 推論 3D 點
            # 預設取樣20萬點
            points = predictor.predict(image_paths, max_images=128, filter_confidence=True, sample_points=100000)

            # 載入第一張 pose
            pose_file = os.path.join(seq_path, "frame-000000.pose.txt")
            if not os.path.exists(pose_file):
                print(f"[WARN] Missing pose file: {pose_file}, skipping.")
                continue
            pose0 = load_pose(pose_file)
            points_world = transform_to_world(points, pose0)

            # 儲存 ply
            ply_name = f"{scene}-{seq}.ply"
            output_path = os.path.join(OUTPUT_DIR, ply_name)
            save_point_cloud(points_world, output_path)
            print(f"[✓] Saved to {output_path}\n")

            # 強制清理 CUDA 記憶體
            del points, points_world, image_paths
            torch.cuda.empty_cache()
            gc.collect()


if __name__ == "__main__":
    main()