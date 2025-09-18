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

import torch
import math
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.pose_utils import get_camera_from_tensor, quadmultiply


def rerender_uv(render_dict, rasterizer, raster_settings, means3D, cov3D_precomp, shs, colors_precomp,
    opacity, scales, rotations):
    keep = render_dict["vis"]
    idx  = keep.nonzero(as_tuple=False).squeeze(1)

    if idx.numel() == 0:
        subimg  = torch.ones_like(render_dict["render"]) * raster_settings.bg.view(1, -1, 1, 1)
        subrads = torch.zeros_like(render_dict["radii"])
        return None
    else:
        # Subselect EVERYTHING you passed originally
        f_means3D = means3D[idx].contiguous()
        f_means2D = (torch.zeros_like(f_means3D)
                        .requires_grad_(True))  # placeholder for screen-space grads
        try: f_means2D.retain_grad()
        except: pass

        f_opacity = opacity[idx]

        if cov3D_precomp is not None:
            f_cov3D     = cov3D_precomp[idx]
            f_scales    = None
            f_rotations = None
        else:
            f_cov3D     = None
            f_scales    = scales[idx]
            f_rotations = rotations[idx]           # <- your gaussians_rot_trans subset

        if shs is not None and shs.numel() > 0:
            f_shs          = shs[idx]
            f_colors_prec  = None
        else:
            f_shs          = None
            f_colors_prec  = colors_precomp[idx] if colors_precomp is not None else None

        # Re-render with the SAME rasterizer (same raster_settings)
        subimg, subrads = rasterizer(
            means3D=f_means3D, means2D=f_means2D,
            shs=f_shs, colors_precomp=f_colors_prec,
            opacities=f_opacity, scales=f_scales, rotations=f_rotations,
            cov3D_precomp=f_cov3D
        )
        return subimg


def render(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    camera_pose=None,
    get_uv=False,
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = (
        torch.zeros_like(
            pc.get_xyz.float(), dtype=pc.get_xyz.float().dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    # Set camera pose as identity. Then, we will transform the Gaussians around camera_pose
    w2c = torch.eye(4).cuda()
    projmatrix = (
        w2c.unsqueeze(0).bmm(viewpoint_camera.projection_matrix.unsqueeze(0))
    ).squeeze(0)
    camera_pos = w2c.inverse()[3, :3]
    projmatrix=projmatrix.float()
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        # viewmatrix=viewpoint_camera.world_view_transform,
        # projmatrix=viewpoint_camera.full_proj_transform,
        viewmatrix=w2c,
        projmatrix=projmatrix,
        sh_degree=pc.active_sh_degree,
        # campos=viewpoint_camera.camera_center,
        campos=camera_pos,
        prefiltered=False,
        debug=pipe.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # means3D = pc.get_xyz
    rel_w2c = get_camera_from_tensor(camera_pose)
    # Transform mean and rot of Gaussians to camera frame
    gaussians_xyz = pc._xyz.clone().float()
    gaussians_rot = pc._rotation.clone().float()

    xyz_ones = torch.ones(gaussians_xyz.shape[0], 1).cuda().float()
    xyz_homo = torch.cat((gaussians_xyz, xyz_ones), dim=1)
    gaussians_xyz_trans = (rel_w2c @ xyz_homo.T).T[:, :3]
    gaussians_rot_trans = quadmultiply(camera_pose[:4], gaussians_rot)
    means3D = gaussians_xyz_trans
    means2D = screenspace_points
    opacity = pc.get_opacity.float()

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier).float()
    else:
        scales = pc.get_scaling.float()
        rotations = gaussians_rot_trans  # pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.float().transpose(1, 2).view(
                -1, 3, (pc.max_sh_degree.float() + 1) ** 2
            )
            dir_pp = pc.get_xyz.float() - viewpoint_camera.camera_center.repeat(
                pc.get_features.shape[0].float(), 1
            )
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features.float()
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    means3D = means3D.float()
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    render_dict = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "radii": radii,
        "visibility_filter": radii > 0,
    }

    if get_uv:
        H = int(viewpoint_camera.image_height)
        W = int(viewpoint_camera.image_width)

        x = means3D[:, 0]
        y = means3D[:, 1]
        z = means3D[:, 2]

        eps = 1e-8
        z_safe = torch.where(z.abs() < eps, torch.where(z >= 0, torch.full_like(z, eps),
            torch.full_like(z, -eps)), z)

        ndc_x = -x / (-z_safe * tanfovx)
        ndc_y = y / (-z_safe * tanfovy)

        u = (ndc_x * 0.5 + 0.5) * W - 0.5
        v = (1.0 - (ndc_y * 0.5 + 0.5)) * H - 0.5

        # in-bounds with half-pixel centers
        in_img = (u >= 0.0) & (u <= (W - 1)) & (v >= 0.0) & (v <= (H - 1))

        # trust the rasterizer for vis, just gate by in-bounds
        vis_ras   = (radii > 0)
        vis_final = vis_ras & in_img

        render_dict["uv"]  = torch.stack([u, v], dim=1)
        render_dict["vis"] = vis_final

        print(f"In image: {int(in_img.sum())}/{pc.get_xyz.shape[0]}, "
            f"ras_vis={int(vis_ras.sum())}, vis_final={int(vis_final.sum())}")

        rerendered_image = rerender_uv(render_dict, rasterizer, raster_settings, means3D, 
            cov3D_precomp, shs, colors_precomp, opacity, scales, rotations) if vis_final.any() else None
        if rerendered_image is not None:
            render_dict["rerender"] = rerendered_image

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return render_dict
