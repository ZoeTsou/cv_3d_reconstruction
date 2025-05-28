# mycode/utils/pointcloud.py
import numpy as np
import open3d as o3d
import plotly.graph_objects as go
import os


def transform_to_world(points, pose):
    """
    points: (N, 3) in camera coordinate
    pose: (4, 4) transformation matrix
    return: (N, 3) points in world coordinate
    """
    ones = np.ones((points.shape[0], 1))
    homo_points = np.concatenate([points, ones], axis=1).T  # 4 x N
    world_points = pose @ homo_points  # 4 x N
    return world_points[:3, :].T  # N x 3


def save_point_cloud(points, filename):
    """
    points: (N, 3) numpy array
    filename: output .ply path
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, pcd)
    print(f"Saved {len(points)} points to {filename}")


def visualize_points_plotly(points: np.ndarray, save_path="./results/point_cloud_plot.html", max_points=300000):
    """
    points: np.ndarray of shape (N, 3) in XYZ
    save_path: where to save the .html visualization
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if points.shape[0] > max_points:
        indices = np.random.choice(points.shape[0], size=max_points, replace=False)
        points = points[indices]

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=1.5,
            color='blue',  # 如果你之後有 RGB，可以放進來
            opacity=0.8
        )
    )])

    fig.update_layout(scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    ),
    title='3D Point Cloud',
    margin=dict(l=0, r=0, b=0, t=30))

    fig.write_html(save_path)
    print(f'[✓] Point cloud visualization saved to {save_path}')

