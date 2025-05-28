# main.py
import os
from mycode.fast3r_wrapper import Fast3RPredictor
from mycode.utils.io import load_color_files, load_pose
from mycode.utils.pointcloud import transform_to_world, save_point_cloud, visualize_points_plotly

def main():
    # === 修改這個路徑以指定場景資料夾 ===
    scene_path = "7scenes/office/test/seq-02"
    output_ply = "test/office-test.ply"

    # === 1. 載入 RGB + Depth 影像序列 ===
    img_list = load_color_files(scene_path)

    # === 2. 初始化模型 ===
    predictor = Fast3RPredictor("jedyang97/Fast3R_ViT_Large_512", device="cuda")

    # === 3. 執行推論 ===
    points = predictor.predict(img_list)  # shape: (N, 3)

    # === 4. 載入第一張相機姿勢（轉換到世界座標） ===
    pose0 = load_pose(os.path.join(scene_path, "frame-000000.pose.txt"))
    points_world = transform_to_world(points, pose0)

    # === 5. 儲存為 .ply ===
    save_point_cloud(points_world, output_ply)
    visualize_points_plotly(points_world, save_path="results/office-test.html")

    print(f"[✓] Point cloud saved to {output_ply}")

if __name__ == "__main__":
    main()
