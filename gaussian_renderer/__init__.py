# #
# # Copyright (C) 2023, Inria
# # GRAPHDECO research group, https://team.inria.fr/graphdeco
# # All rights reserved.
# #
# # This software is free for non-commercial, research and evaluation use 
# # under the terms of the LICENSE.md file.
# #
# # For inquiries contact  george.drettakis@inria.fr
# #

# import torch
# import math
# from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
# from scene.gaussian_model import GaussianModel
# from utils.sh_utils import eval_sh

# def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, separate_sh = False, override_color = None, use_trained_exp=False):
#     """
#     Render the scene. 
    
#     Background tensor (bg_color) must be on GPU!
#     """
 
#     # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
#     screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
#     try:
#         screenspace_points.retain_grad()
#     except:
#         pass

#     # Set up rasterization configuration
#     tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
#     tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

#     raster_settings = GaussianRasterizationSettings(
#         image_height=int(viewpoint_camera.image_height),
#         image_width=int(viewpoint_camera.image_width),
#         tanfovx=tanfovx,
#         tanfovy=tanfovy,
#         bg=bg_color,
#         scale_modifier=scaling_modifier,
#         viewmatrix=viewpoint_camera.world_view_transform,
#         projmatrix=viewpoint_camera.full_proj_transform,
#         sh_degree=pc.active_sh_degree,
#         campos=viewpoint_camera.camera_center,
#         prefiltered=False,
#         debug=pipe.debug,
#         antialiasing=pipe.antialiasing
#     )

#     rasterizer = GaussianRasterizer(raster_settings=raster_settings)

#     means3D = pc.get_xyz
#     means2D = screenspace_points
#     opacity = pc.get_opacity

#     # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
#     # scaling / rotation by the rasterizer.
#     scales = None
#     rotations = None
#     cov3D_precomp = None

#     if pipe.compute_cov3D_python:
#         cov3D_precomp = pc.get_covariance(scaling_modifier)
#     else:
#         scales = pc.get_scaling
#         rotations = pc.get_rotation

#     # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
#     # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
#     shs = None
#     colors_precomp = None
#     if override_color is None:
#         if pipe.convert_SHs_python:
#             shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
#             dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
#             dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
#             sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
#             colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
#         else:
#             if separate_sh:
#                 dc, shs = pc.get_features_dc, pc.get_features_rest
#             else:
#                 shs = pc.get_features
#     else:
#         colors_precomp = override_color

#     # Rasterize visible Gaussians to image, obtain their radii (on screen). 
#     if separate_sh:
#         rendered_image, radii, depth_image = rasterizer(
#             means3D = means3D,
#             means2D = means2D,
#             dc = dc,
#             shs = shs,
#             colors_precomp = colors_precomp,
#             opacities = opacity,
#             scales = scales,
#             rotations = rotations,
#             cov3D_precomp = cov3D_precomp)
#     else:
#         rendered_image, radii, depth_image = rasterizer(
#             means3D = means3D,
#             means2D = means2D,
#             shs = shs,
#             colors_precomp = colors_precomp,
#             opacities = opacity,
#             scales = scales,
#             rotations = rotations,
#             cov3D_precomp = cov3D_precomp)
        
#     # Apply exposure to rendered image (training only)
#     if use_trained_exp:
#         exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
#         rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]

#     # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
#     # They will be excluded from value updates used in the splitting criteria.
#     rendered_image = rendered_image.clamp(0, 1)
#     out = {
#         "render": rendered_image,
#         "viewspace_points": screenspace_points,
#         "visibility_filter" : (radii > 0).nonzero(),
#         "radii": radii,
#         "depth" : depth_image
#         }
    
#     return out

import torch
from utils.graphics_utils import fov2focal
from scene.gaussian_model import GaussianModel

try:
    from gsplat import rasterization
    G_SPLAT_AVAILABLE = True
except ImportError:
    try:
        from gsplat.rendering import rasterization
        G_SPLAT_AVAILABLE = True
    except ImportError:
        G_SPLAT_AVAILABLE = False

def render_gsplat(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    use_trained_exp=False,
):
    device = pc.get_xyz.device
    dtype = pc.get_xyz.dtype

    xyz = pc.get_xyz  # [N,3]
    W = int(viewpoint_camera.image_width)
    H = int(viewpoint_camera.image_height)

    if xyz.shape[0] == 0:
        rgb = bg_color.to(device=device, dtype=dtype)[:, None, None].expand(3, H, W)
        zeros = torch.zeros(1, H, W, device=device, dtype=dtype)
        return {
            "render": rgb,
            "rgb": rgb,
            "acc": zeros,
            "viewspace_points": torch.zeros(1, 0, 2, device=device, dtype=dtype),
            "visibility_filter": torch.zeros(0, dtype=torch.bool, device=device),
            "radii": torch.zeros(0, device=device, dtype=dtype),
            "depth": zeros,
        }

    # ==== Gaussian params ====
    scales = pc.get_scaling * scaling_modifier     # [N,3]
    quats  = pc.get_rotation                       # [N,4] (wxyz)
    opacities = pc.get_opacity[..., 0]             # [N]

    # ==== camera extrinsic: world->cam ====
    viewmats = viewpoint_camera.world_view_transform.mT.to(device=device, dtype=dtype)[None]  # [1,4,4]

    # ==== camera intrinsics ====
    fx = fov2focal(viewpoint_camera.FoVx, W)
    fy = fov2focal(viewpoint_camera.FoVy, H)

    if hasattr(viewpoint_camera, "primx") and hasattr(viewpoint_camera, "primy") \
    and viewpoint_camera.primx is not None and viewpoint_camera.primy is not None:
        cx = float(viewpoint_camera.primx) * W
        cy = float(viewpoint_camera.primy) * H
    else:
        cx = 0.5 * W
        cy = 0.5 * H
    Ks = torch.tensor(
        [[fx, 0.0, cx],
         [0.0, fy, cy],
         [0.0, 0.0, 1.0]],
        device=device, dtype=dtype
    )[None]  # [1,3,3]

    # ==== backgrounds ====
    backgrounds = bg_color.to(device=device, dtype=dtype).view(1, 3)  # [1,3]

    # ==== render options ====
    use_depth = getattr(pipe, "gsplat_use_depth", True)
    render_mode = "RGB+ED" if use_depth else "RGB"

    tile_size = getattr(pipe, "gsplat_tile_size", 16)

    rasterize_mode = "antialiased" if getattr(pipe, "antialiasing", False) else "classic"

    absgrad = getattr(pipe, "gsplat_absgrad", False)

    # ==== colors / SH ====
    if override_color is not None:
        # override_color: 期望 [N,3]，并且是 post-activation RGB
        colors = override_color.to(device=device, dtype=dtype)
        sh_degree = None
    else:
        # SH coeffs: [N, K, 3]
        colors = pc.get_features
        sh_degree = pc.max_sh_degree

    # ==== call high-level rasterization ====
    render_colors, render_alphas, meta = rasterization(
        means=xyz,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=W,
        height=H,
        near_plane=float(viewpoint_camera.znear),
        far_plane=float(viewpoint_camera.zfar),
        sh_degree=sh_degree,
        packed=False,
        tile_size=tile_size,
        backgrounds=backgrounds,
        render_mode=render_mode,      # "RGB+ED" 
        absgrad=absgrad,
        rasterize_mode=rasterize_mode,
    )

    if meta.get("means2d", None) is not None and meta["means2d"].requires_grad:
        meta["means2d"].retain_grad()

    # ==== unpack outputs ====
    # render_colors: [1,H,W,3] or [1,H,W,4]
    # render_alphas: [1,H,W,1]
    if use_depth:
        rgb_hw3 = render_colors[0, ..., :3]
        depth_hw = render_colors[0, ..., 3]
    else:
        rgb_hw3 = render_colors[0, ..., :3]
        depth_hw = render_alphas[0, ..., 0]

    acc_hw = render_alphas[0, ..., 0]

    rgb = rgb_hw3.permute(2, 0, 1).contiguous()
    depth = depth_hw[None, ...].contiguous()
    acc = acc_hw[None, ...].contiguous()

    # ==== exposure (keep original behavior) ====
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rgb = torch.matmul(rgb.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) \
              + exposure[:3, 3, None, None]

    rgb = rgb.clamp(0.0, 1.0)

    # ==== radii & visibility ====
    radii_meta = meta["radii"]
    radii0 = radii_meta[0]

    if radii0.ndim == 2 and radii0.shape[-1] == 2:
        visibility_filter = (radii0 > 0).all(dim=-1)
        radii_scalar = radii0.amax(dim=-1)     
    else:
        visibility_filter = radii0 > 0
        radii_scalar = radii0

    radii_scalar = radii_scalar.to(dtype=pc.max_radii2D.dtype)  # 通常 float32

    return {
        "render": rgb,
        "rgb": rgb,
        "acc": acc,
        "depth": depth,
        "viewspace_points": meta["means2d"],   # [1,N,2]
        "visibility_filter": visibility_filter, # [N]
        "radii": radii_scalar,                  # [N]
    }

def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor,
           scaling_modifier=1.0, separate_sh=False, override_color=None,
           use_trained_exp=False):

    if getattr(pipe, "use_gsplat", False):
        assert G_SPLAT_AVAILABLE, "gsplat 没装好：请先 pip install gsplat"
        return render_gsplat(
            viewpoint_camera, pc, pipe, bg_color,
            scaling_modifier=scaling_modifier,
            override_color=override_color,
            use_trained_exp=use_trained_exp,
        )

