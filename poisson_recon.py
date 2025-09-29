import numpy as np
import open3d as o3d
from plyfile import PlyData
import argparse
import os

def load_gaussian_ply(path):
    ply = PlyData.read(path)
    v = ply['vertex']
    pts = np.vstack([v['x'], v['y'], v['z']]).T.astype(np.float64)
    scales = np.exp(np.vstack([v['scale_0'], v['scale_1'], v['scale_2']]).T.astype(np.float64))
    opacity = None
    if 'opacity' in v.data.dtype.names:
        opacity = np.asarray(v['opacity']).astype(np.float64)
    return pts, scales, opacity

def to_o3d(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def largest_component(mesh):
    labels, counts, _ = mesh.cluster_connected_triangles()
    labels = np.asarray(labels)
    if len(counts)==0: return mesh
    keep = labels == int(np.argmax(np.asarray(counts)))
    mesh.remove_triangles_by_mask(~keep)
    mesh.remove_unreferenced_vertices()
    return mesh

def reconstruct_watertight_compact(points, scales, opacity=None,
                                   voxel_frac=350, nb_neighbors=30, std_ratio=2.0,
                                   psr_depth=10, trim_q=0.55, pad=1.05,
                                   smooth_iter=15):
    # --- 1) Filter junk ---
    keep = np.ones(len(points), dtype=bool)
    # cull low-opacity splats if available
    if opacity is not None:
        thr = np.quantile(opacity, 0.30)  # keep top 70% by opacity
        keep &= (opacity >= thr)
    # drop very large splats (often floaters)
    smin = scales.min(axis=1)
    keep &= (smin < np.quantile(smin, 0.85))
    pts = points[keep]
    if len(pts) < 1000:
        raise RuntimeError("Too few points after filtering.")

    # --- 2) Downsample + denoise ---
    diag = np.linalg.norm(pts.max(0) - pts.min(0))
    voxel = max(diag / voxel_frac, 1e-6)
    pcd = to_o3d(pts).voxel_down_sample(voxel)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)

    # --- 3) Normals: consistent + outward ---
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=3.0*voxel, max_nn=60))
    pcd.orient_normals_consistent_tangent_plane(k=80)
    P = np.asarray(pcd.points)
    N = np.asarray(pcd.normals)
    c = P.mean(axis=0)
    # make normals point outward from centroid
    sign = np.sign((N * (P - c)).sum(1)).mean()
    if sign < 0:  # mostly inward -> flip
        pcd.normals = o3d.utility.Vector3dVector(-N)

    # --- 4) Poisson ---
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=psr_depth, scale=1.0, linear_fit=False)

    # --- 5) Trim low-density foam ---
    dens = np.asarray(dens)
    thr = np.quantile(dens, trim_q)
    mesh.remove_vertices_by_mask(dens < thr)

    # --- 6) Crop to tight bbox of the points (kills far sails/planes) ---
    bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
        o3d.utility.Vector3dVector(P))
    bbox = bbox.scale(pad, bbox.get_center())
    mesh = mesh.crop(bbox)

    # --- 7) Keep main body + clean + light smooth ---
    mesh = largest_component(mesh)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    if smooth_iter > 0:
        mesh = mesh.filter_smooth_taubin(number_of_iterations=smooth_iter)
    mesh.compute_vertex_normals()
    return mesh

def poisson_from_points(P, depth=10, trim_q=0.55, out_path="mesh.obj"):
    # 0) (optional) light voxel downsample to stabilize normals
    diag = np.linalg.norm(P.max(0) - P.min(0))
    voxel = max(diag / 400.0, 1e-6)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P.astype(np.float64))
    if len(P) > 2000:  # tiny clouds usually don't need downsampling
        pcd = pcd.voxel_down_sample(voxel)

    # 1) estimate + orient normals
    # radius ~ a few voxels; fallback if diag is tiny
    radius = max(diag / 50.0, voxel * 3.0)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=60)
    )
    # make normals consistent (helps PSR a lot)
    pcd.orient_normals_consistent_tangent_plane(k=80)

    # 2) Poisson reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, scale=1.0, linear_fit=False
    )

    # 3) trim low-density “foam”
    densities = np.asarray(densities)
    thr = np.quantile(densities, trim_q)
    mesh.remove_vertices_by_mask(densities < thr)

    # 4) crop to tight bbox of input points (removes far-off sheets)
    # bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
    #     o3d.utility.Vector3dVector(P)
    # ).scale(1.05, center=o3d.utility.Vector3dVector(P.mean(0)))
    # mesh = mesh.crop(bbox)

    bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
        o3d.utility.Vector3dVector(P)
    )
    bbox.scale(1.05, bbox.get_center())   # <- don't wrap center in Vector3dVector
    mesh = mesh.crop(bbox)

    # 5) keep largest component + clean up
    labels, counts, _ = mesh.cluster_connected_triangles()
    if len(counts):
        keep = (np.asarray(labels) == int(np.argmax(np.asarray(counts))))
        mesh.remove_triangles_by_mask(~keep)
        mesh.remove_unreferenced_vertices()

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()

    # 6) save
    o3d.io.write_triangle_mesh(out_path, mesh, write_triangle_uvs=False)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply_path", type=str)
    ap.add_argument("--pc_path", type=str, help="Point cloud (npy) path")
    ap.add_argument("--out", default="output")
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--trim_q", type=float, default=0.55)  # 0.50–0.65 usually good
    args = ap.parse_args()

    input_name = "mesh"

    if args.ply_path is not None:
        input_name = os.path.splitext(os.path.basename(args.ply_path))[0]
        args.out = os.path.join(args.out, f"{input_name}_psr_depth{args.depth}_trim{args.trim_q:.2f}.obj")
        pts, scales, opacity = load_gaussian_ply(args.ply_path)
        mesh = reconstruct_watertight_compact(
            pts, scales, opacity=opacity, psr_depth=args.depth, trim_q=args.trim_q)
        o3d.io.write_triangle_mesh(args.out, mesh, write_triangle_uvs=False)
        print("Saved:", args.out)
    if args.pc_path is not None:
        input_name = os.path.splitext(os.path.basename(args.pc_path))[0]
        args.out = os.path.join(args.out, f"{input_name}_psr_depth{args.depth}_trim{args.trim_q:.2f}.obj")
        pts = np.load(args.pc_path).astype(np.float64)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        poisson_from_points(pts, depth=args.depth, trim_q=args.trim_q, out_path=args.out)
