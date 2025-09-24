import numpy as np
import open3d as o3d
from plyfile import PlyData
import argparse

def load_gaussian_ply(path):
    """Loads points and scales from a 3D Gaussian Splat PLY file."""
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    
    # Extract positions (centers of Gaussians)
    points = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    
    # Extract scales (used for filtering)
    scales = np.exp(np.vstack([
        vertices['scale_0'], vertices['scale_1'], vertices['scale_2']
    ]).T)
    
    print(f"Loaded {len(points)} Gaussians from {path}")
    return points, scales

def to_o3d_pcd(points):
    """Converts a numpy point array to an Open3D PointCloud."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    return pcd

def preprocess_points(points, scales, voxel_frac=400, nb_neighbors=30, std_ratio=2.0):
    """Filters, downsamples, and removes outliers from the point cloud."""
    print("Preprocessing point cloud...")
    # 1) Filter out very large Gaussians which are often floaters or background
    smin = scales.min(axis=1)
    keep_mask = smin < np.quantile(smin, 0.85)  # Keep the tightest 85%
    pts = points[keep_mask]
    print(f"  - Filtered large Gaussians: {len(points)} -> {len(pts)} points")

    # 2) Voxel downsample to create a more uniform point density
    scene_diag = np.linalg.norm(pts.max(0) - pts.min(0))
    voxel_size = scene_diag / voxel_frac
    pcd = to_o3d_pcd(pts).voxel_down_sample(voxel_size=voxel_size)
    print(f"  - Voxel downsampled: {len(pts)} -> {len(pcd.points)} points with voxel size {voxel_size:.4f}")

    # 3) Remove statistical outliers
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    print(f"  - Removed statistical outliers: {len(pcd.points)} points remain")
    
    return pcd, voxel_size

def estimate_and_orient_normals(pcd, voxel_size):
    """Estimates normals and orients them consistently, which is critical for reconstruction."""
    print("Estimating and orienting normals...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=3.0 * voxel_size, max_nn=60))
    pcd.orient_normals_consistent_tangent_plane(k=50)
    print("  - Normals estimated and oriented.")
    return pcd

def keep_largest_mesh_component(mesh):
    """Keeps only the largest connected component of the mesh to remove floating artifacts."""
    print("Clustering mesh components...")
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    
    if len(cluster_n_triangles) > 0:
        largest_cluster_idx = cluster_n_triangles.argmax()
        triangles_to_remove = triangle_clusters != largest_cluster_idx
        mesh.remove_triangles_by_mask(triangles_to_remove)
        mesh.remove_unreferenced_vertices()
        print(f"  - Kept largest component with {cluster_n_triangles[largest_cluster_idx]} triangles.")
    return mesh

def reconstruct_surface(points, scales, depth=9, alpha=0.02, smooth_iter=15):
    """
    Main function to perform robust surface reconstruction using Screened Poisson.
    """
    # 1. Preprocess the raw points from Gaussians
    pcd, voxel_size = preprocess_points(points, scales)
    
    # 2. Estimate normals
    pcd = estimate_and_orient_normals(pcd, voxel_size)

    # 3. Perform Screened Poisson Surface Reconstruction
    print(f"Starting Screened Poisson Reconstruction (Depth={depth}, Alpha={alpha})...")
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_screened_poisson(
            pcd, depth=depth, alpha=alpha, linear_fit=False)[0]
    print("  - Initial mesh reconstructed.")

    # 4. Clean up the mesh
    mesh = keep_largest_mesh_component(mesh)

    # 5. Final cleaning and smoothing
    print("Performing final mesh cleanup and smoothing...")
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    # Taubin smoothing is volume-preserving and good for noise reduction
    mesh = mesh.filter_smooth_taubin(number_of_iterations=smooth_iter)
    mesh.compute_vertex_normals() # Update normals after smoothing
    print("  - Mesh smoothed and cleaned.")
    
    return mesh

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstruct a mesh from a 3D Gaussian Splatting PLY file using Screened Poisson.")
    parser.add_argument("--ply_path", type=str, required=True, help="Path to the input .ply file.")
    parser.add_argument("--output_mesh", type=str, default="sandwich_mesh.obj", help="Path to save the output mesh file.")
    parser.add_argument("--depth", type=int, default=9, help="Octree depth for reconstruction. 8-10 is a good range. Higher is more detailed but noisier.")
    parser.add_argument("--alpha", type=float, default=0.02, help="SPSR alpha value. Controls smoothness. Higher values are smoother. 0.01-0.05 is a good range.")
    parser.add_argument("--smooth_iter", type=int, default=20, help="Number of Taubin smoothing iterations.")
    
    args = parser.parse_args()
    
    # Load the raw data
    points, scales = load_gaussian_ply(args.ply_path)

    # Run the reconstruction pipeline
    final_mesh = reconstruct_surface(
        points, scales,
        depth=args.depth,
        alpha=args.alpha,
        smooth_iter=args.smooth_iter
    )

    # Save the result
    o3d.io.write_triangle_mesh(args.output_mesh, final_mesh, write_vertex_normals=True)
    print(f"\n✅ Mesh successfully saved to {args.output_mesh}")
    print("   You can visualize it by running: open3d draw", args.output_mesh)
