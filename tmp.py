# Open and read npy file
import os
import numpy as np

import open3d as o3d

data_path = "assets/welstory/362/pc/840_1_5.npy"
data_path = "assets/welstory_eval/9/pcds/welstory_1st/id_0_class_210_group_0.npy"
# data_path = "assets/h3do/point/plys/008.ply"

def get_npy_shape(path):
    data = np.load(path)
    print(data.shape)

def merge_point_clouds(pcs_path):
    pcs = [np.load(os.path.join(pcs_path, f)) for f in os.listdir(pcs_path) if f.endswith('.npy')]
    merged_pc = np.vstack(pcs)
    print(f"Merged point cloud shape: {merged_pc.shape}")
    np.save(os.path.join(pcs_path, "pcs.npy"), merged_pc)

def visualize_pc(path, voxel=0.0):
    # pc = np.load(path)  # shape: [N, 3] or [N, 6/7]
    pc = load_pc(path)
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

def load_pc(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path)

    if ext == ".ply":
        # Open3D reads both ASCII and binary PLY.
        # If the PLY has rgb, Open3D gives colors in [0,1].
        pcd = o3d.io.read_point_cloud(
            path,
            remove_nan_points=True,
            remove_infinite_points=True,
        )
        xyz = np.asarray(pcd.points)  # [N,3]
        if len(pcd.colors) == len(pcd.points) and len(pcd.colors) > 0:
            rgb = np.asarray(pcd.colors)  # already float in [0,1]
            return np.concatenate([xyz, rgb], axis=1)  # [N,6]
        return xyz

    raise ValueError(f"Unsupported extension: {ext}")

if __name__ == "__main__":
    # merge_point_clouds("assets/welstory/362/pc")
    # visualize_npy_pc("assets/welstory/362/pc/pcs.npy", voxel=0.01)
    visualize_pc(data_path, voxel=0.01)