import argparse, os, numpy as np, open3d as o3d
from plyfile import PlyData

# --- I/O ---
def load_gaussian_ply(path):
    ply = PlyData.read(path)
    v = ply["vertex"]
    pts = np.stack([v['x'], v['y'], v['z']], axis=1).astype(np.float64)
    # 3DGS stores log-scales; exp to get radii-ish
    scales = np.exp(np.stack([v['scale_0'], v['scale_1'], v['scale_2']], axis=1)).astype(np.float64)
    return pts, scales

def save_mesh(mesh, out):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    o3d.io.write_triangle_mesh(out, mesh, write_triangle_uvs=False)

# --- helpers ---
def to_o3d_pcd(points):
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(points)
    return p

def keep_largest_component(mesh):
    lbls, counts, _ = mesh.cluster_connected_triangles()
    lbls = np.asarray(lbls)
    if len(counts)==0: return mesh
    keep = int(np.argmax(np.asarray(counts)))
    mesh.remove_triangles_by_mask(lbls != keep)
    mesh.remove_unreferenced_vertices()
    return mesh

def make_watertight_poisson(points, voxel_frac=300, depth=10, smooth_iter=15):
    # mild cleanup/downsample just for speed; no aggressive pruning
    pcd = to_o3d_pcd(points)
    diag = np.linalg.norm(points.max(0)-points.min(0))
    voxel = max(diag/voxel_frac, 1e-6)
    pcd = pcd.voxel_down_sample(voxel)
    # normals + orientation are important for Poisson
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=3*voxel, max_nn=60))
    pcd.orient_normals_consistent_tangent_plane(50)

    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, scale=1.0, linear_fit=False
    )
    # DO NOT trim by density (keeps it closed)
    mesh = keep_largest_component(mesh)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    # optional: fill tiny holes (Open3D 0.18+)
    if hasattr(mesh, "fill_holes"):
        mesh = mesh.fill_holes()  # keeps it watertight if small gaps exist

    if smooth_iter > 0:
        mesh = mesh.filter_smooth_taubin(number_of_iterations=smooth_iter)
        mesh.compute_vertex_normals()
    return mesh

# --- voxel marching-cubes option (robustly closed) ---
def make_watertight_voxels(points, scales, grid=256, sigma_k=2.0, close_iters=1, smooth_iter=10):
    try:
        from skimage.measure import marching_cubes
        from scipy.ndimage import gaussian_filter, binary_closing
    except Exception as e:
        raise RuntimeError("voxels method needs scikit-image and scipy installed") from e

    # bounds with small pad
    mn, mx = points.min(0), points.max(0)
    pad = 0.02 * np.linalg.norm(mx-mn)
    mn, mx = mn - pad, mx + pad

    # grid
    gx = np.linspace(mn[0], mx[0], grid)
    gy = np.linspace(mn[1], mx[1], grid)
    gz = np.linspace(mn[2], mx[2], grid)
    # voxel size
    vs = ((mx-mn)/(grid-1))

    vol = np.zeros((grid, grid, grid), dtype=np.float32)

    # splat gaussians into volume (fast-ish, coarse)
    radii = sigma_k * np.min(scales, axis=1)  # use smallest axis as radius
    idx = ((points - mn) / vs).astype(np.int32)
    idx = np.clip(idx, 0, grid-1)
    for (ix, iy, iz), r in zip(idx, radii):
        # write a small blob
        rad_vox = max(1, int(max(r / vs).round()))
        x0,x1 = max(0, ix-rad_vox), min(grid, ix+rad_vox+1)
        y0,y1 = max(0, iy-rad_vox), min(grid, iy+rad_vox+1)
        z0,z1 = max(0, iz-rad_vox), min(grid, iz+rad_vox+1)
        vol[x0:x1, y0:y1, z0:z1] += 1.0

    vol = gaussian_filter(vol, sigma=1.0)
    mask = vol > np.percentile(vol, 60)     # occupancy threshold
    if close_iters > 0:
        from scipy.ndimage import generate_binary_structure
        st = generate_binary_structure(3, 1)
        for _ in range(close_iters):
            mask = binary_closing(mask, structure=st)

    # marching cubes -> closed surface
    verts, faces, _, _ = marching_cubes(mask.astype(np.float32), level=0.5, spacing=vs)
    verts += mn  # back to world coords

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts),
        o3d.utility.Vector3iVector(faces.astype(np.int32))
    )
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh = keep_largest_component(mesh)

    if smooth_iter > 0:
        mesh = mesh.filter_smooth_taubin(number_of_iterations=smooth_iter)
        mesh.compute_vertex_normals()
    return mesh

def main():
    ap = argparse.ArgumentParser("Watertight mesh from 3DGS PLY")
    ap.add_argument("--ply", required=True, help="Path to 3DGS .ply")
    ap.add_argument("--out", default="output/watertight.obj", help="Output mesh (.obj/.ply/.stl/.glb)")
    ap.add_argument("--method", choices=["poisson","voxels"], default="poisson")
    # poisson params
    ap.add_argument("--depth", type=int, default=10)
    # voxel params
    ap.add_argument("--grid", type=int, default=256)
    ap.add_argument("--sigma_k", type=float, default=2.0)
    ap.add_argument("--close_iters", type=int, default=1)
    args = ap.parse_args()

    pts, scales = load_gaussian_ply(args.ply)

    if args.method == "poisson":
        mesh = make_watertight_poisson(pts, depth=args.depth)
    else:
        mesh = make_watertight_voxels(pts, scales, grid=args.grid, sigma_k=args.sigma_k, close_iters=args.close_iters)

    print("watertight:", mesh.is_watertight())
    save_mesh(mesh, args.out)
    print("✅ saved", args.out)

if __name__ == "__main__":
    main()
