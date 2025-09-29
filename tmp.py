# Open and read npy file
import os
import numpy as np

import open3d as o3d

data_path = "assets/welstory/362/pc/840_1_5.npy"
data_path = "assets/welstory_eval/9/class_210_group_0.npy"

def get_npy_shape(path):
    data = np.load(path)
    print(data.shape)

def merge_point_clouds(pcs_path):
    pcs = [np.load(os.path.join(pcs_path, f)) for f in os.listdir(pcs_path) if f.endswith('.npy')]
    merged_pc = np.vstack(pcs)
    print(f"Merged point cloud shape: {merged_pc.shape}")
    np.save(os.path.join(pcs_path, "pcs.npy"), merged_pc)

def visualize_npy_pc(path, voxel=0.0):
    pc = np.load(path)  # shape: [N, 3] or [N, 6/7]
    if pc.ndim != 2 or pc.shape[1] < 3:
        raise ValueError(f"Expected [N, >=3], got {pc.shape}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc[:, :3])

    # If colors are present, map to [0,1]
    if pc.shape[1] >= 6:
        cols = pc[:, 3:6]
        if cols.max() > 1.0:  # assume 0-255, convert
            cols = cols / 255.0
        pcd.colors = o3d.utility.Vector3dVector(cols)

    if voxel and voxel > 0:
        pcd = pcd.voxel_down_sample(voxel)

    # Optional: estimate normals for nicer shading (skip if you only want points)
    pcd.estimate_normals()

    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    # merge_point_clouds("assets/welstory/362/pc")
    visualize_npy_pc("assets/welstory/362/pc/pcs.npy", voxel=0.01)
    visualize_npy_pc(data_path, voxel=0.01)