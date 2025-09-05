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

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, num_sample_renderings=None):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    if num_sample_renderings:
        sample_renderings = []
        indices = np.linspace(0, len(views)-1, num_sample_renderings, dtype=int)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        camera_pose = get_tensor_from_camera(view.world_view_transform.transpose(0, 1))
        rendering = render(
            view, gaussians, pipeline, background, camera_pose=camera_pose
        )["render"]
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
            # sample_renderings.append(rendering.detach().cpu().numpy())

    if num_sample_renderings:
        return sample_renderings

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


def project_gaussians_to_pixels(pc, view, camera_pose):
    """
    Returns:
      uv: (N, 2) pixel coords
      z_cam: (N,) depth in camera frame
      vis: (N,) bool, inside frustum (image bounds + NDC) with inferred front sign
      px_radius: (N,) approximate screen radius
    """
    device = pc._xyz.device
    rel_w2c = get_camera_from_tensor(camera_pose).to(device)         # (4,4)
    P = view.projection_matrix.to(device)                            # (4,4)
    H, W = int(view.image_height), int(view.image_width)

    # 3D → camera
    N = pc._xyz.shape[0]
    ones = torch.ones(N, 1, device=device, dtype=pc._xyz.dtype)
    xyz_h = torch.cat([pc._xyz, ones], dim=1)                        # (N,4)
    cam = (rel_w2c @ xyz_h.T).T                                      # (N,4)
    z_cam = cam[:, 2]

    # camera → clip → NDC → pixels
    clip = (P @ cam.T).T                                             # (N,4)
    w = clip[:, 3].clamp(min=1e-8)
    ndc = clip[:, :3] / w.unsqueeze(1)                               # [-1,1]
    u = (ndc[:, 0] * 0.5 + 0.5) * W
    v = (-(ndc[:, 1]) * 0.5 + 0.5) * H                               # Y flip to match your render()
    uv = torch.stack([u, v], dim=1)

    # bounds checks
    in_ndc = (ndc[:,0].abs()<=1) & (ndc[:,1].abs()<=1) & (ndc[:,2].abs()<=1)
    in_img = (u>=0) & (u<W) & (v>=0) & (v<H)

    # infer front-facing sign (OpenGL-style cameras use z<0; others z>0)
    vis_neg = (z_cam < 0) & in_ndc & in_img
    vis_pos = (z_cam > 0) & in_ndc & in_img
    # pick the one that yields more visible points
    vis = vis_neg if vis_neg.sum() >= vis_pos.sum() else vis_pos

    # pixel radius ~ f * s / |z|
    fx = 0.5*W/torch.tan(torch.tensor(view.FoVx, device=device)/2)
    fy = 0.5*H/torch.tan(torch.tensor(view.FoVy, device=device)/2)
    s = pc.get_scaling.max(dim=1).values
    f = (fx+fy)*0.5
    z_abs = z_cam.abs().clamp(min=1e-6)
    px_radius = (f * s / z_abs).clamp(max=64.0)

    return uv, z_cam, vis, px_radius



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

    # inflate by morphological max via a cheap trick: resample a downscaled mask
    # (for speed) OR just accept center test + visibility for a first pass.
    keep = vis & center_inside
    return keep


def select_object_gaussians(pc, views, masks, camera_poses, vote_thresh=None):
    """
    views: list of view objects (same as your render loop)
    masks: list of (H,W) tensors (0/1) on CUDA
    camera_poses: list of (q,T) tensors (same format you pass render)
    """
    device = pc._xyz.device
    K = len(views)
    N = pc._xyz.shape[0]
    votes = torch.zeros(N, device=device, dtype=torch.int32)

    for view, mask, cam_pose in zip(views, masks, camera_poses):
        uv, z, vis, r = project_gaussians_to_pixels(pc, view, cam_pose)
        keep = gaussians_inside_mask(uv, vis, r, mask[0])
        votes += keep.int()

    if vote_thresh is None or vote_thresh < 0:
        vote_thresh = (K+1)//2
    keep_idx = (votes >= vote_thresh).nonzero(as_tuple=True)[0]
    return keep_idx


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


def _read_mask(path, H, W, device):
    arr = imageio.imread(path)
    if arr.ndim == 3:
        arr = arr[...,0]  # take one channel if RGB
    t = torch.from_numpy((arr > 0).astype(np.float32)).to(device)
    # resize if needed (nearest)
    if t.shape[0] != H or t.shape[1] != W:
        t = t.view(1,1,*t.shape)
        t = F.interpolate(t, size=(H,W), mode="nearest").view(H,W)
    return t


def _resolve_view_list(scene, which="test"):
    return scene.getTestCameras() if which == "test" else scene.getTrainCameras()


def _load_masks_and_views(scene, args, iteration, device):
    """
    Returns: views[list], masks[list of HxW CUDA tensors], camera_poses[list of (q,T)]
    """
    views_all = _resolve_view_list(scene, "test")
    masks, views, poses = [], [], []

    if args.seg_paths:
        paths = [p.strip() for p in args.seg_paths.split(",") if p.strip()]
        assert len(paths) > 0, "seg_paths empty"
        for i, p in enumerate(paths):
            v = views_all[i]
            H, W = int(v.image_height), int(v.image_width)
            m = _read_mask(p, H, W, device)
            cam_pose = get_tensor_from_camera(v.world_view_transform.transpose(0,1)).to(device)
            masks.append(m); views.append(v); poses.append(cam_pose)
        return views, masks, poses

    if args.seg_dir and args.seg_views:
        idxs = [int(x) for x in args.seg_views.split(",")]
        seg_dir = Path(args.seg_dir)
        for i in idxs:
            v = views_all[i]
            H, W = int(v.image_height), int(v.image_width)
            # try image_name.png first, then 00000.png
            cand1 = seg_dir / f"{v.image_name}.png"
            cand2 = seg_dir / f"{i:05d}.png"
            if cand1.exists():
                mp = cand1
            elif cand2.exists():
                mp = cand2
            else:
                raise FileNotFoundError(f"No mask for view idx {i}: tried {cand1} and {cand2}")
            m = _read_mask(mp, H, W, device)
            cam_pose = get_tensor_from_camera(v.world_view_transform.transpose(0,1)).to(device)
            masks.append(m); views.append(v); poses.append(cam_pose)
        return views, masks, poses

    raise ValueError("Provide either --seg_paths or (--seg_dir and --seg_views).")


def get_object_gaussians_and_save(gaussians, scene, dataset, iteration, pipeline, background, args, masks, camera_poses, views):
    device = gaussians._xyz.device
    # views, masks, camera_poses = _load_masks_and_views(scene, args, iteration, device)

    # 1) select by multi-view voting
    keep_idx = select_object_gaussians(
        gaussians, views, masks, camera_poses,
        vote_thresh=args.vote_thresh
    )

    if keep_idx.numel() == 0:
        print("⚠️ No gaussians selected. Check masks & view indices.")
        return

    # 2) compute bbox (AABB+OBB)
    xyz_obj = gaussians._xyz[keep_idx]
    (aabb_min, aabb_max), (center, R, extents) = bbox_from_points(xyz_obj)

    # 3) optional crop by (A) AABB or (B) OBB + margin
    if args.use_obb:
        mask_box = inside_obb(gaussians._xyz, center, R, extents, margin=args.bbox_margin)
    else:
        mask_box = inside_aabb(gaussians._xyz, aabb_min, aabb_max, margin=args.bbox_margin)

    keep_final = torch.nonzero(mask_box, as_tuple=True)[0]
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
        renderings = render_set(
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

    # Run semantic segmentation
    segmentationmap = SegmentationMap(
        window=None,
        config=config,
        main_cam_view=config.main_cam,
        cam_coords=roi_mask_coords,
    )
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

    poses = []
    segments = []
    views = []
    views_all = viewpoint_stack
    device = gaussians._xyz.device
    for i, cam_view in enumerate(valid_camera_views):
        vco_cam_view = valid_vco_camera_views[i]
        v = views_all[cam_view]
        cam_pose = get_tensor_from_camera(v.world_view_transform.transpose(0,1)).to(device)
        poses.append(cam_pose)
        segments.append(output_data.multisegmentlist[vco_cam_view][0])
        views.append(v)

    # Get 3D Bounding Boxes
    # Get object gasusian splatting
    
    # Save GSplat
    # Save rendering
    get_object_gaussians_and_save(gaussians, scene, dataset, iteration, pipeline, background, args, segments, poses, views)


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

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    vco_args, roi_mask_coords = vco_setup()

    

    render_sets(model.extract(args), args.iterations, pipeline.extract(args), args.skip_train, args.skip_test, args, vco_args, roi_mask_coords)
