#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import json
from os import makedirs
from time import time, perf_counter
from argparse import ArgumentParser
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "models"))

import torch
import torchvision
from tqdm import tqdm
import imageio
import numpy as np
from pathlib import Path
import torch.nn.functional as F
import cv2
from scipy.spatial import ConvexHull

from scene import Scene
from scene.dataset_readers import loadCameras
from gaussian_renderer import render, GaussianModel
from utils.general_utils import safe_state
from utils.pose_utils import get_tensor_from_camera, get_camera_from_tensor
from utils.loss_utils import l1_loss, ssim, l1_loss_mask, ssim_loss_mask
from utils.sfm_utils import save_time
from utils.camera_utils import generate_interpolated_path
from utils.camera_utils import visualizer
from arguments import ModelParams, PipelineParams, get_combined_args

# VCO
from config import make_vco_args, init_cam_configs
from modules.segmentation import SegmentationMap
from modules.objectDetect import ObjectDetector
from interface.common import CameraView
from modules.outputData import OutputData

def save_interpolate_pose(model_path, iter, n_views):

    org_pose = np.load(model_path / f"pose/ours_{iter}/pose_optimized.npy")
    visualizer(org_pose, ["green" for _ in org_pose], model_path / f"pose/ours_{iter}/poses_optimized.png")
    n_interp = int(10 * 30 / n_views)  # 10second, fps=30
    all_inter_pose = []
    for i in range(n_views-1):
        tmp_inter_pose = generate_interpolated_path(poses=org_pose[i:i+2], n_interp=n_interp)
        all_inter_pose.append(tmp_inter_pose)
    all_inter_pose = np.concatenate(all_inter_pose, axis=0)
    all_inter_pose = np.concatenate([all_inter_pose, org_pose[-1][:3, :].reshape(1, 3, 4)], axis=0)

    inter_pose_list = []
    for p in all_inter_pose:
        tmp_view = np.eye(4)
        tmp_view[:3, :3] = p[:3, :3]
        tmp_view[:3, 3] = p[:3, 3]
        inter_pose_list.append(tmp_view)
    inter_pose = np.stack(inter_pose_list, 0)
    visualizer(inter_pose, ["blue" for _ in inter_pose], model_path / f"pose/ours_{iter}/poses_interpolated.png")
    np.save(model_path / f"pose/ours_{iter}/pose_interpolated.npy", inter_pose)


def images_to_video(image_folder, output_video_path, fps=30):
    """
    Convert images in a folder to a video.

    Args:
    - image_folder (str): The path to the folder containing the images.
    - output_video_path (str): The path where the output video will be saved.
    - fps (int): Frames per second for the output video.
    """
    images = []

    for filename in sorted(os.listdir(image_folder)):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.JPG', '.PNG')):
            image_path = os.path.join(image_folder, filename)
            image = imageio.imread(image_path)
            images.append(image)

    imageio.mimwrite(output_video_path, images, fps=fps)

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, num_sample_renderings=None, view=None):
    if view is None:
        render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
        gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    else:
        render_path = os.path.join(model_path, name, "ours_{}".format(iteration), str(view), "renders")
        gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), str(view), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    if num_sample_renderings:
        sample_renderings = []
        render_dicts = []
        indices = np.linspace(0, len(views)-1, num_sample_renderings, dtype=int)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        camera_pose = get_tensor_from_camera(view.world_view_transform.transpose(0, 1))
        get_uv = num_sample_renderings and idx in indices
        render_dict = render(
            view.float(), gaussians, pipeline, background, camera_pose=camera_pose.float(),
            get_uv=get_uv
        )
        rendering = render_dict["render"]
        
        gt = view.original_image[0:3, :, :]
        image_path = os.path.join(render_path, "{0:05d}".format(idx) + ".png")
        torchvision.utils.save_image(
            rendering, image_path
        )
        if name != "interp":
            torchvision.utils.save_image(   
                gt, os.path.join(gts_path, "{0:05d}".format(idx) + ".png")
            )
        if num_sample_renderings and idx in indices:
            image = np.array(cv2.imread(image_path))
            sample_renderings.append(image)

            render_dict["view"] = view
            render_dict["idx"] = idx
            render_dicts.append(render_dict)

            rerenddering = render_dict.get("rerender", None)
            rerender_image_path = image_path.replace("renders", "rerenders")
            if rerenddering is not None:
                makedirs(os.path.dirname(rerender_image_path), exist_ok=True)
                torchvision.utils.save_image(
                    rerenddering, rerender_image_path
                )
            # sample_renderings.append(rendering.detach().cpu().numpy())

    if num_sample_renderings:
        return sample_renderings, render_dicts

def render_set_optimize(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    gaussians._xyz.requires_grad_(False)
    gaussians._features_dc.requires_grad_(False)
    gaussians._features_rest.requires_grad_(False)
    gaussians._opacity.requires_grad_(False)
    gaussians._scaling.requires_grad_(False)
    gaussians._rotation.requires_grad_(False)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        num_iter = args.optim_test_pose_iter
        camera_pose = get_tensor_from_camera(view.world_view_transform.transpose(0, 1))

        camera_tensor_T = camera_pose[-3:].requires_grad_()
        camera_tensor_q = camera_pose[:4].requires_grad_()
        pose_optimizer = torch.optim.Adam([
            {"params": [camera_tensor_T], "lr": 0.003},
            {"params": [camera_tensor_q], "lr": 0.001}
        ],
        betas=(0.9, 0.999),
        weight_decay=1e-4
        )

        # Add a learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(pose_optimizer, T_max=num_iter, eta_min=0.0001)
        with tqdm(total=num_iter, desc=f"Tracking Time Step: {idx+1}", leave=True) as progress_bar:
            candidate_q = camera_tensor_q.clone().detach()
            candidate_T = camera_tensor_T.clone().detach()
            current_min_loss = float(1e20)
            gt = view.original_image[0:3, :, :]
            initial_loss = None

            for iteration in range(num_iter):
                rendering = render(view, gaussians, pipeline, background, camera_pose=torch.cat([camera_tensor_q, camera_tensor_T]))["render"]
                black_hole_threshold = 0.0
                mask = (rendering > black_hole_threshold).float()
                loss = l1_loss_mask(rendering, gt, mask)
                loss.backward()
                with torch.no_grad():
                    pose_optimizer.step()
                    pose_optimizer.zero_grad(set_to_none=True)

                    if iteration == 0:
                        initial_loss = loss.item()  # Capture initial loss

                    if loss < current_min_loss:
                        current_min_loss = loss
                        candidate_q = camera_tensor_q.clone().detach()
                        candidate_T = camera_tensor_T.clone().detach()

                    progress_bar.update(1)
                    progress_bar.set_postfix(loss=loss.item(), initial_loss=initial_loss)
                scheduler.step()

            camera_tensor_q = candidate_q
            camera_tensor_T = candidate_T

        optimal_pose = torch.cat([camera_tensor_q, camera_tensor_T])
        # print("optimal_pose-camera_pose: ", optimal_pose-camera_pose)
        rendering_opt = render(view, gaussians, pipeline, background, camera_pose=optimal_pose)["render"]
            
        torchvision.utils.save_image(
            rendering_opt, os.path.join(render_path, view.image_name + ".png")
        )
        torchvision.utils.save_image(
            gt, os.path.join(gts_path, view.image_name + ".png")
        )

    if args.test_fps:
        print(">>> Calculate FPS: ")
        fps_list = []
        for _ in range(1000):
            start = perf_counter()
            _ = render(view, gaussians, pipeline, background, camera_pose=optimal_pose)
            end = perf_counter()
            fps_list.append(end - start)        
        fps_list.sort()
        fps_list = fps_list[100:900]
        fps = 1 / (sum(fps_list) / len(fps_list))
        print(">>> FPS = ", fps)
        with open(f"{model_path}/total_fps.json", 'a') as fp:
            json.dump(f'{fps}', fp, indent=True)
            fp.write('\n')


def project_gaussians_to_pixels(pc, view):
    device = pc._xyz.device
    dtype  = torch.float32  # keep things consistent
    H, W = int(view.image_height), int(view.image_width)

    camera_pose = get_tensor_from_camera(view.world_view_transform.transpose(0, 1)).float()

    rel_w2c = get_camera_from_tensor(camera_pose).to(device=device, dtype=dtype)
    N = pc._xyz.shape[0]
    xyz_h = torch.cat([pc._xyz.to(dtype), torch.ones(N,1,device=device,dtype=dtype)], dim=1)
    cam = (rel_w2c @ xyz_h.T).T

    x, y, z = cam[:,0], cam[:,1], cam[:,2]
    z_safe = torch.where(z == 0, torch.full_like(z, 1e-8), z)

    FoVx = torch.as_tensor(getattr(view, "FoVx"), device=device, dtype=dtype)
    FoVy = torch.as_tensor(getattr(view, "FoVy"), device=device, dtype=dtype)
    if float(FoVx) > 3.2 or float(FoVy) > 3.2:
        FoVx = torch.deg2rad(FoVx); FoVy = torch.deg2rad(FoVy)

    fx = 0.5 * W / torch.tan(FoVx / 2)
    fy = 0.5 * H / torch.tan(FoVy / 2)
    cx = 0.5 * W
    cy = 0.5 * H
    
    zf = (-z_safe)  # NOTE: minus sign
    u = fx * (x / z_safe) + cx
    v = cy - fy * (y / z_safe)
    uv = torch.stack([u, v], dim=1)

    in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    vis_neg = (z < 0) & in_img
    vis_pos = (z > 0) & in_img
    vis = vis_neg if vis_neg.sum() >= vis_pos.sum() else vis_pos

    s = pc.get_scaling.to(device=device, dtype=dtype).max(dim=1).values
    f_mean = 0.5 * (fx + fy)
    px_radius = (f_mean * s / z.abs().clamp(min=1e-6)).clamp(max=64.0)

    print("proj:", {
        "in_img": int(in_img.sum().item()),
        "z<0": int((z<0).sum().item()),
        "z>0": int((z>0).sum().item()),
        "vis": int(vis.sum().item())}
    )
    return uv, z, vis, px_radius



def project_gaussians_to_pixels_dep(pc, view, camera_pose):
    """
    Returns:
      uv: (N, 2) pixel coords
      z_cam: (N,) depth in camera frame
      vis: (N,) bool, inside image with inferred front sign
      px_radius: (N,) approximate screen radius
    """
    device = pc._xyz.device
    dtype  = pc._xyz.dtype
    H, W = int(view.image_height), int(view.image_width)

    # --- world -> camera (homogeneous) ---
    rel_w2c = get_camera_from_tensor(camera_pose).to(device=device, dtype=dtype)   # (4,4)
    N = pc._xyz.shape[0]
    ones = torch.ones(N, 1, device=device, dtype=dtype)
    xyz_h = torch.cat([pc._xyz, ones], dim=1)                                      # (N,4)
    cam = (rel_w2c @ xyz_h.T).T                                                    # (N,4)

    x = cam[:, 0]
    y = cam[:, 1]
    z = cam[:, 2]
    z_safe = torch.where(z == 0, torch.full_like(z, 1e-8), z)

    # --- intrinsics from FoV (handle deg/rad) ---
    FoVx = torch.as_tensor(getattr(view, "FoVx"), device=device, dtype=dtype)
    FoVy = torch.as_tensor(getattr(view, "FoVy"), device=device, dtype=dtype)
    # If someone passed degrees, convert
    if float(FoVx) > 3.2 or float(FoVy) > 3.2:
        FoVx = torch.deg2rad(FoVx)
        FoVy = torch.deg2rad(FoVy)

    fx = 0.5 * W / torch.tan(FoVx / 2)
    fy = 0.5 * H / torch.tan(FoVy / 2)
    cx = 0.5 * W
    cy = 0.5 * H

    # --- camera -> pixel ---
    # image y points down, so subtract fy*(y/z)
    u = fx * (x / z_safe) + cx
    v = cy - fy * (y / z_safe)
    uv = torch.stack([u, v], dim=1)

    # --- visibility (NO NDC; just pixel bounds + z sign) ---
    in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    vis_neg = (z < 0) & in_img    # OpenGL-style: camera looks down -Z
    vis_pos = (z > 0) & in_img    # Some pipelines use +Z forward
    vis = vis_neg if vis_neg.sum() >= vis_pos.sum() else vis_pos

    # --- approximate pixel radius: f * s / |z| ---
    s = pc.get_scaling.max(dim=1).values  # assume (N,3) -> (N,)
    f_mean = (fx + fy) * 0.5
    px_radius = (f_mean * s / z.abs().clamp(min=1e-6)).clamp(max=64.0)

    print("proj:", {
        "in_img": int(in_img.sum().item()),
        "z<0": int((z<0).sum().item()),
        "z>0": int((z>0).sum().item()),
        "vis": int(vis.sum().item())}
    )

    return uv, z, vis, px_radius



def gaussians_inside_mask(uv, vis, px_radius, mask_tensor, thresh=0.5):
    """
    mask_tensor: (H,W) float/bool on same device
    Returns keep: (N,) bool
    """
    mask_tensor = torch.as_tensor(mask_tensor, dtype=torch.float32, device=uv.device)

    device = mask_tensor.device
    H, W = mask_tensor.shape
    N = uv.shape[0]

    # sample mask at subpixel coords via bilinear (treat as [1,1,H,W])
    grid_u = (uv[:,0] / max(W-1,1))*2 - 1
    grid_v = (uv[:,1] / max(H-1,1))*2 - 1
    grid = torch.stack([grid_u, grid_v], dim=1).view(1,1,N,2)  # N points
    m = F.grid_sample(mask_tensor.view(1,1,H,W).float(), grid, align_corners=True, mode="bilinear") # (1,1,1,N)
    center_inside = (m.view(N) > thresh)

    print("mask:", {
        "center_inside": int(center_inside.sum().item()),
        "vis": int(vis.sum().item()),
        "keep": int((vis & center_inside).sum().item())
    })

    # inflate by morphological max via a cheap trick: resample a downscaled mask
    # (for speed) OR just accept center test + visibility for a first pass.
    keep = vis & center_inside
    return keep


def gaussians_inside_mask_ring(uv, vis, px_radius, mask_tensor, thresh=0.5):
    """
    mask_tensor: (H,W) float/bool/uint8; returns keep: (N,) bool
    Uses a tiny disk around each uv with radius ~ px_radius to be robust.
    """
    # normalize mask
    m = torch.as_tensor(mask_tensor, dtype=torch.float32, device=uv.device)
    if m.ndim == 3:  # HWC or CHW -> squeeze
        m = m[..., 0] if m.shape[-1] in (3,4) else m.squeeze(0)
    H, W = m.shape

    N = uv.shape[0]
    # clamp to reasonable minimum/maximum screen footprint
    r = px_radius.clamp(min=0.75, max=8.0)

    # build 8-point ring samples per gaussian (cheap “dilation”)
    angles = torch.linspace(0, 2*np.pi, steps=8, device=uv.device, dtype=uv.dtype)[:-1]
    du = (r[:, None] * torch.cos(angles)[None, :])
    dv = (r[:, None] * torch.sin(angles)[None, :])

    U = uv[:, 0:1] + du
    V = uv[:, 1:2] + dv

    # also include center sample
    U = torch.cat([uv[:, 0:1], U], dim=1)  # (N,9)
    V = torch.cat([uv[:, 1:2], V], dim=1)  # (N,9)

    # grid_sample expects [-1,1]
    grid_u = (U / max(W-1, 1)) * 2 - 1
    grid_v = (V / max(H-1, 1)) * 2 - 1
    grid = torch.stack([grid_u, grid_v], dim=-1).view(1, 1, -1, 2)

    sampled = F.grid_sample(m.view(1,1,H,W), grid, mode="bilinear", align_corners=True)
    sampled = sampled.view(N, -1)  # (N,9)

    frac_inside = (sampled > thresh).float().mean(dim=1)
    keep = vis & (frac_inside > 0.25)  # need at least ~2/8 neighbors or center

    # debug counts
    print("mask:", {
        "any_inside": int((frac_inside > 0).sum().item()),
        "vis": int(vis.sum().item()),
        "keep": int(keep.sum().item())
    })
    return keep


def gaussians_inside_mask_simple(uv, z, vis, px_radius, mask_tensor, thresh=0.5):
    # assumes: vis already guarantees in-bounds & correct facing
    m = torch.as_tensor(mask_tensor, device=uv.device, dtype=torch.float32)
    
    if m.ndim == 3:                     # (H,W,3/4) -> (H,W)
        m = m[..., 0]

    i = uv[:, 1].round().long()         # row (y)
    j = uv[:, 0].round().long()         # col (x)

    keep = torch.zeros(uv.size(0), dtype=torch.bool, device=uv.device)
    v = vis.nonzero(as_tuple=True)[0]   # only index masked-in points
    if v.numel():
        keep[v] = m[i[v], j[v]] > thresh
    return keep

def gaussians_inside_mask(uv, z, vis, px_radius, mask_tensor, thresh=0.9, keep_first_hit=False):
    m = torch.as_tensor(mask_tensor, dtype=torch.float32, device=uv.device)
    if m.ndim == 3:
        m = m[...,0] if m.shape[-1] in (3,4) else m.squeeze(0)
    H, W = m.shape

    # center-only sample
    grid_u = (uv[:,0] / max(W-1,1))*2 - 1
    grid_v = (uv[:,1] / max(H-1,1))*2 - 1
    grid = torch.stack([grid_u, grid_v], dim=1).view(1,1,-1,2)
    s = F.grid_sample(m.view(1,1,H,W), grid, mode="bilinear", align_corners=True).view(-1)

    keep = vis & (s > thresh)
    
    # Only keep first-hit gaussians at pixel centers
    if keep_first_hit:
        fh = first_hit_mask_center(uv, z, vis, H, W)
        keep = keep & fh

    print("mask_center:", {"inside": int((s>thresh).sum().item()), "vis": int(vis.sum().item()), "keep": int(keep.sum().item())})
    return keep


def first_hit_mask_center(uv, z, vis, H, W, tol=1e-6):
    """
    Keep gaussians that are the 'closest' at their rounded pixel center.
    Uses |z| (since you've already enforced a consistent forward sign via `vis`).
    """
    if vis.dtype != torch.bool:
        vis = vis.bool()

    # Round to pixel centers (consistent with center-only sampling)
    u_idx = uv[:, 0].round().clamp(0, W-1).long()
    v_idx = uv[:, 1].round().clamp(0, H-1).long()
    pix   = v_idx * W + u_idx

    # Only consider visible gaussians
    pix_vis = pix[vis]
    z_eff   = z.abs()
    z_vis   = z_eff[vis]

    # Compute per-pixel min depth via scatter_reduce
    big = torch.full((H*W,), float('inf'), device=z.device, dtype=z_vis.dtype)
    depth_min = big.scatter_reduce(0, pix_vis, z_vis, reduce='amin', include_self=True)

    # Mark those that match the min for their pixel (within tol)
    is_min_vis = z_vis <= (depth_min[pix_vis] + tol)

    keep = torch.zeros_like(vis)
    keep[vis] = is_min_vis
    return keep


def gaussians_inside_mask_uv(uv, vis, m, size):
    H, W = size

    if isinstance(m, np.ndarray):
        m = torch.from_numpy(m)          # convert to torch
    m = (m > 0.5).to(torch.bool).to(uv.device)

    assert m.shape[-2:] == (H, W), f"Mask {m.shape[-2:]} != {(H,W)}"

    u = uv[:,0].round().clamp(0, W-1).long()
    v = uv[:,1].round().clamp(0, H-1).long()
    in_mask = m[v, u]

    keep = vis & in_mask
    return keep


def select_object_gaussians(pc, render_dicts, masks, vote_thresh=None):
    """
    views: list of view objects (same as your render loop)
    masks: list of (H,W) tensors (0/1) on CUDA
    camera_poses: list of (q,T) tensors (same format you pass render)
    """
    device = pc._xyz.device

    K = len(render_dicts)
    N = pc._xyz.shape[0]
    votes = torch.zeros(N, device=device, dtype=torch.int32)

    gaussians_per_view = {}

    idx = 0
    for out, mask in zip(render_dicts, masks):
        # keep = vis

        # uv, z, vis, r = project_gaussians_to_pixels(pc, view)
        # keep = gaussians_inside_mask_simple(uv, z, vis, r, mask[0])

        view = out["view"]
        size = (int(view.image_height), int(view.image_width))
        keep = gaussians_inside_mask_uv(out["uv"], out["vis"], mask[0], size)
        
        votes += keep.int()
        gaussians_per_view[idx] = keep.nonzero(as_tuple=True)[0]

        idx += 1

    if vote_thresh is None or vote_thresh < 0:
        vote_thresh = (K+1)//2
    keep_idx = (votes >= vote_thresh).nonzero(as_tuple=True)[0]
    return keep_idx, gaussians_per_view


def bbox_from_points(xyz):
    # AABB
    aabb_min = xyz.min(dim=0).values
    aabb_max = xyz.max(dim=0).values

    # OBB via PCA
    X = xyz - xyz.mean(dim=0, keepdim=True)
    C = X.T @ X / max(1, X.shape[0]-1)
    evals, evecs = torch.linalg.eigh(C.float())      # columns = eigenvectors
    # project onto OBB axes:
    proj = X @ evecs                          # (N,3)
    obb_min = proj.min(dim=0).values
    obb_max = proj.max(dim=0).values
    center = xyz.mean(dim=0)
    R = evecs                                 # 3×3 rotation
    extents = 0.5*(obb_max - obb_min)         # half-lengths
    return (aabb_min, aabb_max), (center, R, extents)


def extract_submodel(pc, keep_idx):
    sub = {}
    for name in ["_xyz","_rotation","_scaling","_opacity","_features_dc","_features_rest"]:
        sub[name] = getattr(pc, name)[keep_idx].clone()
    return sub  # or build a new GaussianModel and assign tensors

def inside_aabb(xyz, aabb_min, aabb_max, margin=0.0):
    return ((xyz >= (aabb_min - margin)).all(dim=1) &
            (xyz <= (aabb_max + margin)).all(dim=1))

def inside_obb(xyz, center, R, extents, margin=0.0):
    # xyz: (N,3); center: (3,); R: (3,3) with columns = axes; extents: (3,) half-lengths
    X = xyz - center[None, :]
    # bring points to box frame
    local = X @ R        # (N,3)
    e = extents + margin
    return (local.abs() <= e[None, :]).all(dim=1)


@torch.no_grad()
def build_submodel(pc, keep_idx, sh_degree=None):
    deg = (sh_degree if sh_degree is not None
           else getattr(pc, "max_sh_degree", getattr(pc, "active_sh_degree", 3)))
    obj_pc = GaussianModel(deg)
    def P(x): return torch.nn.Parameter(x.clone().contiguous(), requires_grad=False)
    obj_pc._xyz           = P(pc._xyz[keep_idx])
    obj_pc._features_dc   = P(pc._features_dc[keep_idx])
    obj_pc._features_rest = P(pc._features_rest[keep_idx])
    obj_pc._opacity       = P(pc._opacity[keep_idx])
    obj_pc._scaling       = P(pc._scaling[keep_idx])
    obj_pc._rotation      = P(pc._rotation[keep_idx])
    obj_pc.max_sh_degree = deg
    obj_pc.active_sh_degree = deg
    return obj_pc

@torch.no_grad()
def save_object_subcloud(pc, keep_idx, out_dir, sh_degree=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    obj_pc = build_submodel(pc, keep_idx, sh_degree=sh_degree)
    ply_path = out_dir / "point_cloud.ply"
    obj_pc.save_ply(str(ply_path))
    return ply_path, obj_pc


def get_hull(xyz_obj: torch.Tensor):
    """
    Build a 3D convex hull from selected object points.
    xyz_obj: (M,3) torch tensor
    Returns: scipy ConvexHull
    """
    pts = xyz_obj.detach().cpu().numpy()
    hull = ConvexHull(pts)
    return hull

def include_pcs_in_hull(hull, all_xyz: torch.Tensor, margin: float = 0.0):
    """
    Select all points inside or on the convex hull.
    all_xyz: (N,3) torch tensor
    margin: outward grow in world units (default 0.0)
    Returns: keep mask (N,) torch.bool
    """
    P = all_xyz.detach().cpu().numpy()
    eq = hull.equations  # (F,4), each row: [nx, ny, nz, c]
    A = eq[:, :3]
    b = eq[:, 3]
    if margin > 0:
        n_norm = np.linalg.norm(A, axis=1, keepdims=True) + 1e-12
        b = b - margin * n_norm.squeeze(1)
    inside = np.all(P @ A.T + b <= 1e-9, axis=1)
    return torch.from_numpy(inside).to(all_xyz.device)


def get_object_gaussians_and_save(gaussians, scene, dataset, iteration, pipeline, background, args, masks, render_dicts):
    device = gaussians._xyz.device

    views = [r["view"] for r in render_dicts]

    # 1) select by multi-view voting
    keep_idx, gaussians_per_view = select_object_gaussians(
        gaussians, render_dicts, masks,
        vote_thresh=args.vote_thresh
    )

    for view, idx in gaussians_per_view.items():
        obj_pc = build_submodel(gaussians, idx, sh_degree=dataset.sh_degree)
        render_set(dataset.model_path, f"{args.out_tag}_renders/", iteration, views, obj_pc, pipeline, background, view=view)

    if keep_idx.numel() == 0:
        print("⚠️ No gaussians selected. Check masks & view indices.")
        return

    xyz_obj = gaussians._xyz[keep_idx]
    keep_final = keep_idx

    print(f"Selected {keep_idx.shape[0]} gaussians after voting.")

    ## Convex Hull
    hull = get_hull(xyz_obj)
    mask_sel = include_pcs_in_hull(hull, gaussians._xyz, margin=getattr(args, "hull_margin", 0.0))

    ## Boundng Box
    # (aabb_min, aabb_max), (center, R, extents) = bbox_from_points(xyz_obj)

    # # 3) optional crop by (A) AABB or (B) OBB + margin
    # if args.use_obb:
    #     mask_sel = inside_obb(gaussians._xyz, center, R, extents, margin=args.bbox_margin)
    # else:
    #     mask_sel = inside_aabb(gaussians._xyz, aabb_min, aabb_max, margin=args.bbox_margin)

    keep_final = torch.nonzero(mask_sel, as_tuple=True)[0]
    print(f"Kept {keep_final.shape[0]} gaussians after hull cropping.")
    out_dir = Path(dataset.model_path) / f"objects/ours_{iteration}/{args.out_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 4) save cropped splats to PLY and (optional) render them
    ply_path, obj_pc = save_object_subcloud(gaussians, keep_final, out_dir, sh_degree=dataset.sh_degree)
    print(f"✅ Saved object splats: {ply_path}")

    if args.render_object:
        # render cropped object with the same views you passed
        render_set(dataset.model_path, f"{args.out_tag}_renders", iteration, views, obj_pc, pipeline, background)
        print(f"✅ Rendered object views under {dataset.model_path}/{args.out_tag}_renders/ours_{iteration}/renders")


def render_sets(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
    args,
    vco_args,
    roi_mask_coords
):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, opt=args, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # if not skip_train:
    if not skip_train and not args.infer_video and not dataset.eval:        
        optimized_pose = np.load(Path(args.model_path) / 'pose' / f'ours_{iteration}' / 'pose_optimized.npy')
        viewpoint_stack = loadCameras(optimized_pose, scene.getTrainCameras())
        render_set(
            dataset.model_path,
            "train",
            scene.loaded_iter,
            viewpoint_stack,
            gaussians,
            pipeline,
            background,
        )

    else:
        start_time = time()
        if not skip_test:
            render_set_optimize(
                dataset.model_path,
                "test",
                scene.loaded_iter,
                scene.getTestCameras(),
                gaussians,
                pipeline,
                background,
            )
        end_time = time()
        save_time(dataset.model_path, '[4] render', end_time - start_time)

    if args.infer_video and not dataset.eval:
        save_interpolate_pose(Path(args.model_path), iteration, args.n_views)
        interp_pose = np.load(Path(args.model_path) / 'pose' / f'ours_{iteration}' / 'pose_interpolated.npy')
        viewpoint_stack = loadCameras(interp_pose, scene.getTrainCameras())
        renderings, render_dicts = render_set(
            dataset.model_path,
            "interp",
            scene.loaded_iter,
            viewpoint_stack,
            gaussians,
            pipeline,
            background,
            num_sample_renderings=args.num_render_views
        )
        image_folder = os.path.join(dataset.model_path, f'interp/ours_{iteration}/renders')
        output_video_file = os.path.join(dataset.model_path, f'interp/ours_{iteration}/interp_{args.n_views}_view.mp4')
        images_to_video(image_folder, output_video_file, fps=30)

    # Task: Extract 3D Gaussian model
    # Use 3 rendered images saved above (renderings)
    config = vco_args
    # Run object detection
    object_detector = ObjectDetector(
        config, cam_coords=roi_mask_coords
    )
    valid_camera_views = np.linspace(0, len(viewpoint_stack)-1, args.num_render_views, dtype=int)
    valid_vco_camera_views = [
        'LW',
        'TC',
        'RW',
    ]
    output_data = OutputData(config)
    yolo_masks = dict()
    output_data.cls = defaultdict(list)
    output_data.conf = defaultdict(list)
    output_data.visualize_object_detector = defaultdict(list)
    bg_id = (0, 42, 43, 44, 45, 46, 47)
    is_remove_bg = True
    render_path = os.path.join(dataset.model_path, "interp", "ours_{}".format(1000), "renders")
    intermediate_output_path = os.path.join(dataset.model_path, "interp", "ours_{}".format(1000), "intermediate_output")
    os.makedirs(intermediate_output_path, exist_ok=True)

    clean_segments_output_path = os.path.join(intermediate_output_path, "clean_segments")
    os.makedirs(clean_segments_output_path, exist_ok=True)

    for i, cam_view in enumerate(valid_camera_views):
        occupancy_check = False
        if renderings[i] is None:
            print(f"no image comes : {cam_view}")

        vco_cam_view = valid_vco_camera_views[i]
        (
            output_data.objectbox[vco_cam_view],
            output_data.numobject[vco_cam_view],
            output_data.occupancy,
            yolo_masks[vco_cam_view],
        ) = object_detector(
            renderings[i],
            # output_data,
            vco_cam_view,
            mode="capture",
            occupancy_check=occupancy_check,
            bg_id=bg_id,
            is_remove_bg=is_remove_bg,
        )

        bboxes = output_data.objectbox[vco_cam_view]
        image_path = os.path.join(render_path, "{0:05d}".format(cam_view) + ".png")
        image = np.array(cv2.imread(image_path))
        # Draw bounding boxes on the image
        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Save the image with bounding boxes
        output_image_path = os.path.join(intermediate_output_path, "{0:05d}_bbox.png".format(cam_view))
        cv2.imwrite(output_image_path, image)
        print(f"Saved image with bounding boxes: {output_image_path}")

    # Run semantic segmentation
    segmentationmap = SegmentationMap(
        window=None,
        config=config,
        main_cam_view=config.main_cam,
        cam_coords=roi_mask_coords,
    )
    
    poses = []
    segments = []
    views = []
    views_all = viewpoint_stack
    device = gaussians._xyz.device

    for i, cam_view in enumerate(valid_camera_views):
        vco_cam_view = valid_vco_camera_views[i]
        output_data = segmentationmap(
            renderings[i],
            output_data,
            vco_cam_view,
            mode="capture",
            occupancy_check=False,
            yolo_masks=yolo_masks[vco_cam_view],
            use_detection_segmentation=config.use_detection_segmentation,
        )

        image_path = os.path.join(render_path, "{0:05d}".format(cam_view) + ".png")
        image = np.array(cv2.imread(image_path))
        segmentmap = output_data.multisegmentlist[vco_cam_view][0][0]
        # Draw segmentation map on the image
        colored_mask = (segmentmap * 255).astype(np.uint8)
        colored_mask = cv2.applyColorMap(colored_mask, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)
        # Save the image with segmentation map
        output_image_path = os.path.join(intermediate_output_path, "{0:05d}_segmentation.png".format(cam_view))
        cv2.imwrite(output_image_path, overlay)
        print(f"Saved image with segmentation map: {output_image_path}")

        # Save rgba segmentation image
        output_image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        background_mask = segmentmap == 0
        output_image[background_mask, 3] = 0 # Index 3 is the alpha channel

        # 5. Save the final image with a transparent background
        # The file MUST be saved as a .png to preserve transparency
        output_image_path = os.path.join(clean_segments_output_path, "{0:05d}.png".format(cam_view))
        cv2.imwrite(output_image_path, output_image)

        v = views_all[cam_view]
        cam_pose = get_tensor_from_camera(v.world_view_transform.transpose(0,1)).to(device)
        poses.append(cam_pose)
        segments.append(output_data.multisegmentlist[vco_cam_view][0])
        views.append(v)

    
    get_object_gaussians_and_save(gaussians, scene, dataset, iteration, pipeline, background, args, segments, render_dicts)


dataset_type="welstory_1st"

mannamil_dataset_path = "dataset/0000_mannamil_20250707_061547/raw"
vco_2nd_dataset_path = "dataset/vco_2nd_eval/data/vco_7/log/capture"
welstory_dataset_path = "dataset/vco_welstory_pre_eval/data/vco_vol_3/log/capture"
welstory_poc_dataset_path = "dataset/vco_eval_welstory_1st_poc/data/vco_vol_4/log/capture"

if dataset_type == "mannamil":
    data_id = "937_135"
    dataset_path = f"{mannamil_dataset_path}/{data_id}"
elif dataset_type == "vco_2nd":
    data_id = "0"
    dataset_path = f"{vco_2nd_dataset_path}/{data_id}"
elif dataset_type == "welstory_1st":
    data_id = "9"
    dataset_path = f"{welstory_poc_dataset_path}/{data_id}"

def vco_setup():
    vco_args = make_vco_args(overrides={
        "main_cam": "TB",
        "use_detection_segmentation": "true",
        "prompt_type": "box",
        "sam2_type": "imagepred",
        "image_height": 480,
        "image_width": 640,
        "object_detector": "yolo",
        "store_cd": "welstory_1st",
        "num_top_k": 2,
        "depth_threshold": 30.0,
    })
    if dataset_type == "mannamil":
        vco_args.stereo_path = f"{dataset_path}/mask/stereo_config_online.json"
        mask_path = f"{dataset_path}/mask"
        vco_args.object_detector = "dfine"
        vco_args.project = "manna"
    elif dataset_type == "vco_2nd":
        mask_path = dataset_path[:dataset_path.find("/log")] + "/mask"
        vco_args.stereo_path = dataset_path[:dataset_path.find("/data/vco")] + "/stereo_config_online.json"
        vco_args.object_detector = "yolo"
        vco_args.project = "phase3"
    elif dataset_type == "welstory_1st":
        vco_args.stereo_path = dataset_path[:dataset_path.find("/log")] + "/mask/stereo_config_online.json"
        mask_path = dataset_path[:dataset_path.find("/log")] + "/mask"
        vco_args.object_detector = "yolo"
        vco_args.store_cd = "welstory_1st"

    valid_camera_views: list[CameraView] = [
        CameraView.TOP_BACK,
        CameraView.TOP_FRONT,
        CameraView.TOP_LEFT,
        CameraView.TOP_RIGHT,
        CameraView.TOP_CENTER,
        CameraView.LEFT_WING,
        CameraView.RIGHT_WING,
    ]

    roi_mask_coords = init_cam_configs(
        valid_camera_views, cam_coords_root=mask_path
    )

    return vco_args, roi_mask_coords

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=False)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iterations", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")    
    parser.add_argument("--optim_test_pose_iter", default=500, type=int)
    parser.add_argument("--infer_video", action="store_true")
    parser.add_argument("--test_fps", action="store_true")
    
    # NEW: object extraction args
    parser.add_argument("--num_render_views", type=int, default=3, help="Number of rendering views --num_render_views (e.g., 1,4,9).")
    parser.add_argument("--vote_thresh", type=int, default=1, help="1 by default.")
    parser.add_argument("--out_tag", type=str, default="obj", help="Subfolder tag under objects/ours_<iter>/")
    parser.add_argument("--render_object", default=True, type=bool, help="Render the cropped splats with selected views.")
    parser.add_argument("--use_obb", action="store_true", help="Use OBB crop instead of AABB.")
    parser.add_argument("--bbox_margin", type=float, default=0.0, help="Expand bbox by this world distance.")
    parser.add_argument("--hull_margin", type=float, default=0.0, help="Expand convex hull by this world distance.")

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    vco_args, roi_mask_coords = vco_setup()

    

    render_sets(model.extract(args), args.iterations, pipeline.extract(args), args.skip_train, args.skip_test, args, vco_args, roi_mask_coords)
