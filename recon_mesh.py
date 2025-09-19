import numpy as np
import open3d as o3d
from plyfile import PlyData
import argparse

def load_gaussian_ply(path):
    """Loads necessary attributes from a 3D Gaussian Splat PLY file."""
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    
    # Extract positions
    points = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    
    # Extract scales and convert from log-space
    scales = np.exp(np.vstack([
        vertices['scale_0'], vertices['scale_1'], vertices['scale_2']
    ]).T)
    
    # Extract rotations (quaternions)
    rotations = np.vstack([
        vertices['rot_0'], vertices['rot_1'], vertices['rot_2'], vertices['rot_3']
    ]).T
    
    print(f"Loaded {len(points)} Gaussians from {path}")
    return points, scales, rotations

def quaternions_to_rotation_matrices(quats):
    """Convert quaternions (w, x, y, z) to 3x3 rotation matrices."""
    # Normalize quaternions
    norm = np.linalg.norm(quats, axis=1, keepdims=True)
    quats = quats / norm
    
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    
    # Pre-calculate reused terms
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    
    # Initialize rotation matrices
    matrices = np.zeros((len(quats), 3, 3))
    
    matrices[:, 0, 0] = 1 - 2 * (yy + zz)
    matrices[:, 0, 1] = 2 * (xy - wz)
    matrices[:, 0, 2] = 2 * (xz + wy)
    
    matrices[:, 1, 0] = 2 * (xy + wz)
    matrices[:, 1, 1] = 1 - 2 * (xx + zz)
    matrices[:, 1, 2] = 2 * (yz - wx)
    
    matrices[:, 2, 0] = 2 * (xz - wy)
    matrices[:, 2, 1] = 2 * (yz + wx)
    matrices[:, 2, 2] = 1 - 2 * (xx + yy)
    
    return matrices

def predict_normals(scales, rotations):
    """
    Predicts normals by finding the smallest eigenvector of the covariance matrix.
    This corresponds to the flattest direction of the Gaussian ellipsoid.
    """
    print("Predicting normals from Gaussian attributes...")
    
    # Get rotation matrices from quaternions
    rot_matrices = quaternions_to_rotation_matrices(rotations)
    
    # Create diagonal scaling matrices
    scaling_matrices = np.zeros((len(scales), 3, 3))
    scaling_matrices[:, 0, 0] = scales[:, 0]
    scaling_matrices[:, 1, 1] = scales[:, 1]
    scaling_matrices[:, 2, 2] = scales[:, 2]
    
    # Calculate covariance matrices: Σ = R * S * S^T * R^T
    # Since S is diagonal, S*S^T = S^2 (a diagonal matrix with squared scales)
    S_sq = scaling_matrices ** 2
    covariances = rot_matrices @ S_sq @ rot_matrices.transpose(0, 2, 1)

    # Eigendecomposition of the covariance matrices
    # np.linalg.eigh returns eigenvalues in ascending order for symmetric matrices
    eigenvalues, eigenvectors = np.linalg.eigh(covariances)
    
    # The normal is the eigenvector corresponding to the smallest eigenvalue
    # For eigh, this is the first eigenvector column
    normals = eigenvectors[:, :, 0]
    
    print("Normals prediction complete.")
    return normals

def reconstruct_mesh(points, normals, depth=9):
    """
    Performs Poisson Surface Reconstruction using Open3D.
    """
    print(f"Starting Poisson Surface Reconstruction with octree depth {depth}...")
    
    # Create an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)

    # Optional: Orient normals consistently
    # pcd.orient_normals_consistent_tangent_plane(100)
    
    # Perform Poisson reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, linear_fit=True
    )
    
    # Optional: Clean up the mesh by removing low-density vertices
    print("Filtering low-density vertices from the mesh...")
    vertices_to_remove = densities < np.quantile(densities, 0.05)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    
    print("Reconstruction complete.")
    return mesh

def texture_mapping():
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstruct a mesh from a 3D Gaussian Splatting PLY file.")
    parser.add_argument("ply_path", type=str, help="Path to the input .ply file.")
    parser.add_argument("output_mesh", type=str, default="", help="Path to save the output mesh file (e.g., mesh.obj).")
    parser.add_argument("--depth", type=int, default=9, help="Octree depth for Poisson reconstruction. Higher is more detailed.")
    
    args = parser.parse_args()
    
    # 1. Load the Gaussian data
    points, scales, rotations = load_gaussian_ply(args.input_ply)
    
    # 2. Predict the normals
    normals = predict_normals(scales, rotations)
    
    # 3. Reconstruct the mesh
    reconstructed_mesh = reconstruct_mesh(points, normals, depth=args.depth)
    
    # 4. Save the final mesh
    o3d.io.write_triangle_mesh(args.output_mesh, reconstructed_mesh)
    print(f"✅ Mesh successfully saved to {args.output_mesh}")