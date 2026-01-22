from __future__ import annotations

import math
from typing import TYPE_CHECKING
from typing import Literal

import numpy as np
import torch
from rich.progress import Progress
from rich.progress import SpinnerColumn
from rich.progress import TimeElapsedColumn
from rich.progress import track

from splaty.rasterizer_utils import approximate_2d_covariances
from splaty.rasterizer_utils import assign_gaussians_to_tiles
from splaty.rasterizer_utils import compute_axis_aligned_bounding_box
from splaty.rasterizer_utils import compute_conics
from splaty.rasterizer_utils import frustum_culling
from splaty.rasterizer_utils import get_color_at_means
from splaty.rasterizer_utils import sort_gaussians_by_depth_of_means
from splaty.spatial_math import invert_batched_2x2_matrices
from splaty.spherical_harmonics import eval_spherical_harmonics_color
from splaty.visualize import save_gif


if TYPE_CHECKING:
    from pathlib import Path

    from jaxtyping import Float32
    from jaxtyping import UInt8
    from numpy import ndarray
    from torch import Tensor

    from splaty.camera import Camera
    from splaty.gaussian import GaussianSet


def render_splats_tiled(
    gs: GaussianSet,
    camera: Camera,
    sigma_multiplier: float = 3.0,
    tile_size: int = 32,
    save_frames: bool = False,
    output_dir: Path | None = None,
) -> UInt8[Tensor, "3 H W"]:
    """Render Gaussians using tiled rendering with front-to-back ordering and early stopping.

    This implements a tile-based rendering approach where the image is divided into tiles
    and Gaussians are assigned to tiles based on their bounding boxes. Within each tile,
    Gaussians are processed in front-to-back order with early ray termination when pixels
    become opaque. This is the technique used in the original 3DGS paper, though it's
    optimized for GPU parallelism and is actually slower on CPU.

    Args:
        gs: The Gaussian set to render.
        camera: Camera defining the viewpoint and projection.
        sigma_multiplier: Multiplier for Gaussian extent (typically 3.0 for 3-sigma).
            Controls how far from the mean the Gaussian is rendered.
        tile_size: Size of each tile in pixels (default 32x32).
        save_frames: Whether to save intermediate frames as a GIF.
        output_dir: Directory to save the GIF frames if save_frames is True.

    Returns:
        Rendered RGB image as uint8 tensor of shape [3, H, W].

    Note:
        This approach is designed for GPU parallelism and can be slower than the
        non-tiled approach on CPU due to overhead from tile management.
    """
    H = camera.height
    W = camera.width

    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32) / 255.0
    bg: Float32[Tensor, "3 H W"] = bg_color[:, None, None].expand(3, H, W).clone()

    means_cam = camera.transform_points_world_to_cam(gs.means)
    means_image = camera.project_points_cam_to_image_plane(means_cam)

    gs, means_cam, means_image = frustum_culling(
        gs,
        camera,
        means_cam,
        means_image,
        cull_image_bounds=False,
        cull_near_far_planes=True,
        fov_margin_ratio=0.0,
    )

    gs, means_cam, means_image = sort_gaussians_by_depth_of_means(
        gs,
        means_cam,
        means_image,
        back_to_front=False,
    )

    covariances_cam = camera.rotate_covariances_world_to_cam(gs.covariances)
    covariances_image = approximate_2d_covariances(
        points_cam=means_cam,
        covariances_cam=covariances_cam,
        camera=camera,
        eps_2d=0.3,
    )

    conics = compute_conics(covariances_image)

    axis_aligined_bounding_boxes = compute_axis_aligned_bounding_box(
        gaussian_means_image=means_image,
        camera=camera,
        covariances_image=covariances_image,
        sigma_multiplier=sigma_multiplier,
        use_simple_radius=True,
    )

    tile_dict = assign_gaussians_to_tiles(
        axis_aligned_bounding_boxes=axis_aligined_bounding_boxes,
        tile_size=tile_size,
        image_width=W,
        image_height=H,
    )

    directions = camera.view_direction_cam_to_mean_world(points_world=gs.means)
    sh_colors = eval_spherical_harmonics_color(
        sh_coefficients=gs.sh_coefficients,
        directions=directions,
        degree=gs.sh_degree,
    )

    color: Float32[Tensor, "3 H W"] = torch.zeros(3, H, W)
    alpha: Float32[Tensor, "H W"] = torch.zeros((H, W), dtype=torch.float32)

    frames: list[UInt8[ndarray, "H W 3"]] = []

    with Progress(SpinnerColumn(), *Progress.get_default_columns(), TimeElapsedColumn()) as tile_progress:
        # Iterate over each tile to be rendered
        for tile_idx_u, tile_idx_v in tile_progress.track(
            tile_dict, description=f"[red]Rendering {len(tile_dict)} tiles"
        ):
            idx_gaussians_in_tile = tile_dict[(tile_idx_u, tile_idx_v)]

            if not idx_gaussians_in_tile:
                continue  # no gaussians in this tile

            opacities_tile = gs.opacities[idx_gaussians_in_tile].contiguous()
            sh_colors_tile = sh_colors[idx_gaussians_in_tile, :].contiguous()
            means_image_tile = means_image[idx_gaussians_in_tile].contiguous()
            conics_tile = conics[idx_gaussians_in_tile].contiguous()

            # iterate over pixels in the tile
            u_start = tile_idx_u * tile_size
            v_start = tile_idx_v * tile_size
            u_end = min(u_start + tile_size, W)
            v_end = min(v_start + tile_size, H)

            with Progress(SpinnerColumn(), *Progress.get_default_columns(), TimeElapsedColumn()) as pixel_progress:
                # progress bar for pixels in tile
                pixel_task = pixel_progress.add_task(
                    f"[cyan]Rendering {len(idx_gaussians_in_tile)} gaussians in tile ({tile_idx_u}, {tile_idx_v})",
                    total=(u_end - u_start) * (v_end - v_start),
                )

                for u_pix in range(u_start, u_end):
                    for v_pix in range(v_start, v_end):
                        pixel_progress.update(pixel_task, advance=1)

                        # prcoss gaussians in this tile
                        for i in range(len(idx_gaussians_in_tile)):
                            alpha_pixel = alpha[v_pix, u_pix]

                            if alpha_pixel >= 0.99:
                                continue  # pixel already opaque, skip

                            # Retrieve values only once per gaussian and not in the inner loops
                            opacity_gaussian = opacities_tile[i]
                            sh_color = sh_colors_tile[i, :]
                            a = conics_tile[i, 0]
                            b = conics_tile[i, 1]
                            c = conics_tile[i, 2]
                            u_mean, v_mean = means_image_tile[i].tolist()

                            du = u_pix - u_mean
                            dv = v_pix - v_mean

                            dist_sq = a * du * du + 2 * b * du * dv + c * dv * dv

                            # Only draw pixel if it's within the elipsis
                            # squared distance, no need to compute sqrt.
                            if dist_sq <= sigma_multiplier * sigma_multiplier:
                                # Blending gaussian into pixel
                                weight = math.exp(-0.5 * dist_sq)
                                alpha_gaussian = opacity_gaussian * weight
                                color_gaussian = sh_color * alpha_gaussian

                                transmittance_pixel = 1 - alpha_pixel

                                # compute new alpha and color
                                alpha[v_pix, u_pix] += transmittance_pixel * alpha_gaussian
                                color[:, v_pix, u_pix] += transmittance_pixel * color_gaussian

            # save frame after each tile
            if save_frames:
                frames.append((color.clone().permute(1, 2, 0).numpy() * 255).astype(np.uint8))

    if save_frames:
        save_gif(frames, output_dir)

    color += bg * (1 - alpha)[None, :, :]
    color = torch.clamp(color, min=0.0, max=1.0)
    rendered_image = (color * 255).to(torch.uint8)

    return rendered_image


def render_splats_fov_cull_transmittance(
    gs: GaussianSet,
    camera: Camera,
    sigma_multiplier: float = 3.0,
    save_frames: bool = False,
    output_dir: Path | None = None,
) -> UInt8[Tensor, "3 H W"]:
    """Render Gaussians as elliptical splats with optimized culling and transmittance tracking.

    This rendering approach includes several optimizations over the basic splat renderer:
    - Frustum culling with a field-of-view margin to reduce Gaussians processed
    - Front-to-back rendering order for better early stopping
    - Transmittance-based early ray termination (stops when pixel alpha >= 0.99)
    - Precomputed conics (inverse covariance) for faster Mahalanobis distance computation
    - Axis-aligned bounding boxes to limit pixel iteration

    Args:
        gs: The Gaussian set to render.
        camera: Camera defining the viewpoint and projection.
        sigma_multiplier: Multiplier for Gaussian extent (typically 3.0 for 3-sigma).
            Defines the cutoff distance for rendering each Gaussian.
        save_frames: Whether to save intermediate frames as a GIF.
        output_dir: Directory to save the GIF frames if save_frames is True.

    Returns:
        Rendered RGB image as uint8 tensor of shape [3, H, W].

    Note:
        This is significantly faster than render_splats() due to early stopping and
        better culling, while producing visually identical results in most cases.
    """
    H = camera.height
    W = camera.width
    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32) / 255.0
    bg: Float32[Tensor, "3 H W"] = bg_color[:, None, None].expand(3, H, W).clone()

    means_cam = camera.transform_points_world_to_cam(gs.means)

    means_image = camera.project_points_cam_to_image_plane(means_cam)

    gs, means_cam, means_image = frustum_culling(
        gs,
        camera,
        means_cam,
        means_image,
        cull_image_bounds=False,
        cull_near_far_planes=True,
        fov_margin_ratio=0.3,
    )

    gs, means_cam, means_image = sort_gaussians_by_depth_of_means(
        gs,
        means_cam,
        means_image,
        back_to_front=False,
    )

    covariances_cam = camera.rotate_covariances_world_to_cam(gs.covariances)
    covariances_image = approximate_2d_covariances(
        points_cam=means_cam,
        covariances_cam=covariances_cam,
        camera=camera,
        eps_2d=0.3,
    )

    conics = compute_conics(covariances_image)

    directions = camera.view_direction_cam_to_mean_world(points_world=gs.means)
    sh_colors = eval_spherical_harmonics_color(
        sh_coefficients=gs.sh_coefficients,
        directions=directions,
        degree=gs.sh_degree,
    )

    axis_aligined_bounding_boxes = compute_axis_aligned_bounding_box(
        gaussian_means_image=means_image,
        camera=camera,
        covariances_image=covariances_image,
        sigma_multiplier=sigma_multiplier,
        use_simple_radius=True,
    )

    color: Float32[Tensor, "3 H W"] = torch.zeros(3, H, W)
    alpha: Float32[Tensor, "H W"] = torch.zeros((H, W), dtype=torch.float32)

    frames: list[UInt8[ndarray, "H W 3"]] = []

    with Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
    ) as progress:
        for i in progress.track(range(len(gs)), description="Rendering Gaussians..."):
            # Retrieve values only once per gaussian and not in the inner loops
            u_mean, v_mean = means_image[i]
            opacity_gaussian = gs.opacities[i]
            sh_color = sh_colors[i, :]
            u_min, v_min, u_max, v_max = axis_aligined_bounding_boxes[i].tolist()

            a = conics[i, 0]
            b = conics[i, 1]
            c = conics[i, 2]

            if opacity_gaussian < 0.01:
                continue  # skip nearly transparent gaussians

            # iterate over a square region around the projected mean
            for u_pix in range(u_min, u_max):
                for v_pix in range(v_min, v_max):
                    # Only write pixel if it's within image bounds
                    if 0 <= u_pix < W and 0 <= v_pix < H:
                        du = u_pix + 0.5 - u_mean
                        dv = v_pix + 0.5 - v_mean

                        dist_sq = a * du * du + 2 * b * du * dv + c * dv * dv

                        # Only draw pixel if it's within the elipsis
                        if dist_sq <= sigma_multiplier * sigma_multiplier:  # squared distance, no need to compute sqrt.
                            # Blending gaussian into pixel
                            weight = math.exp(-0.5 * dist_sq)
                            alpha_gaussian = opacity_gaussian * weight
                            color_gaussian = sh_color * alpha_gaussian

                            # read current alpha and color
                            alpha_pixel = alpha[v_pix, u_pix]

                            if alpha_pixel >= 0.99:
                                # 0.95: fast preview
                                # 0.99: default
                                # 0.995: high quality
                                # 0.999: overkill
                                # You could also skip based on transmittance_pixel < 0.01, meaning how much light is
                                # still getting through
                                continue  # pixel already opaque, skip

                            transmittance_pixel = 1 - alpha_pixel

                            # compute new alpha and color
                            alpha[v_pix, u_pix] += transmittance_pixel * alpha_gaussian
                            color[:, v_pix, u_pix] += transmittance_pixel * color_gaussian

            if save_frames and (i % 2000 == 0 or i == len(gs) - 1):
                frames.append((color.clone().permute(1, 2, 0).numpy() * 255).astype(np.uint8))

    if save_frames:
        save_gif(frames, output_dir)

    color += bg * (1 - alpha)[None, :, :]
    color = torch.clamp(color, min=0.0, max=1.0)
    rendered_image = (color * 255).to(torch.uint8)

    return rendered_image


def render_splats(
    gs: GaussianSet,
    camera: Camera,
    sigma_multiplier: float = 3.0,
    save_frames: bool = False,
    output_dir: Path | None = None,
) -> UInt8[Tensor, "3 H W"]:
    """Render Gaussians as elliptical splats with full 2D covariance and alpha blending.

    This is the core 3D Gaussian Splatting renderer that projects 3D Gaussians onto the
    2D image plane as ellipses defined by their 2D covariance matrices. Each Gaussian's
    contribution is computed using the Mahalanobis distance and blended using the standard
    "over" operator in back-to-front order.

    The rendering pipeline:
    1. Transform Gaussian means to camera space and project to image plane
    2. Frustum culling to remove off-screen Gaussians
    3. Depth sorting (back-to-front) for correct alpha blending
    4. Project 3D covariances to 2D using the EWA splatting approximation
    5. Rasterize each Gaussian as an ellipse using Mahalanobis distance
    6. Alpha blend with spherical harmonics colors

    Args:
        gs: The Gaussian set to render.
        camera: Camera defining the viewpoint and projection.
        sigma_multiplier: Multiplier for Gaussian extent (typically 3.0 for 3-sigma).
            Determines how many standard deviations from the mean to render.
        save_frames: Whether to save intermediate frames as a GIF.
        output_dir: Directory to save the GIF frames if save_frames is True.

    Returns:
        Rendered RGB image as uint8 tensor of shape [3, H, W].

    Note:
        This is the basic, unoptimized implementation that clearly shows all steps.
        For faster rendering, see render_splats_fov_cull_transmittance() or
        render_splats_tiled().
    """
    H = camera.height
    W = camera.width

    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32) / 255.0
    bg: Float32[Tensor, "3 H W"] = bg_color[:, None, None].expand(3, H, W).clone()

    means_cam = camera.transform_points_world_to_cam(gs.means)
    means_image = camera.project_points_cam_to_image_plane(means_cam)

    gs, means_cam, means_image = frustum_culling(
        gs,
        camera,
        means_cam,
        means_image,
        cull_image_bounds=True,
        cull_near_far_planes=True,
        fov_margin_ratio=0.3,
    )

    gs, means_cam, means_image = sort_gaussians_by_depth_of_means(
        gs,
        means_cam,
        means_image,
        back_to_front=True,
    )

    covariances_cam = camera.rotate_covariances_world_to_cam(gs.covariances)
    covariances_image = approximate_2d_covariances(
        points_cam=means_cam,
        covariances_cam=covariances_cam,
        camera=camera,
        eps_2d=0.3,
    )

    inv_covariances_image = invert_batched_2x2_matrices(covariances_image)

    directions = camera.view_direction_cam_to_mean_world(points_world=gs.means)
    sh_colors = eval_spherical_harmonics_color(
        sh_coefficients=gs.sh_coefficients,
        directions=directions,
        degree=gs.sh_degree,
    )

    axis_aligined_bounding_boxes = compute_axis_aligned_bounding_box(
        gaussian_means_image=means_image,
        camera=camera,
        covariances_image=covariances_image,
        sigma_multiplier=sigma_multiplier,
        use_simple_radius=True,
    )

    color: Float32[Tensor, "3 H W"] = torch.zeros(3, H, W)
    alpha: Float32[Tensor, "H W"] = torch.zeros((H, W), dtype=torch.float32)

    frames: list[UInt8[ndarray, "H W 3"]] = []

    with Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
    ) as progress:
        for i in progress.track(range(len(gs)), description="Rendering Gaussians..."):
            # Retrieve values only once per gaussian and not in the inner loops
            u_mean, v_mean = means_image[i]
            opacity_gaussian = gs.opacities[i]
            sh_color = sh_colors[i, :]
            u_min, v_min, u_max, v_max = axis_aligined_bounding_boxes[i].tolist()
            inv_covariance_image = inv_covariances_image[i]

            if opacity_gaussian < 0.01:
                continue  # skip nearly transparent gaussians

            for u_pix in range(u_min, u_max):
                # iterate over a square region around the projected mean
                for v_pix in range(v_min, v_max):
                    # Only write pixel if it's within image bounds
                    if 0 <= u_pix < W and 0 <= v_pix < H:
                        # compute squared distance from mean. + 0.5 to center within pixel
                        du = u_pix + 0.5 - u_mean
                        dv = v_pix + 0.5 - v_mean

                        offset_from_mean = torch.tensor([du, dv], dtype=torch.float32)

                        # Mahalanobis distance, no transpose needed since offset_from_mean is a 1D tensor, not column vector
                        # How many standard deviations (sigmas) away from the mean
                        # squared distance, no need to compute sqrt.
                        dist_sq = offset_from_mean @ inv_covariance_image @ offset_from_mean

                        # Only draw pixel if it's within the elipsis
                        if dist_sq <= sigma_multiplier * sigma_multiplier:  # squared distance, no need to compute sqrt.
                            # Perform the alpha blending with with over operator
                            weight = math.exp(-0.5 * dist_sq)
                            alpha_gaussian = opacity_gaussian * weight
                            color_gaussian = sh_color * alpha_gaussian

                            # read current alpha and color
                            alpha_pixel = alpha[v_pix, u_pix]
                            color_pixel = color[:, v_pix, u_pix]

                            # compute new alpha and color
                            alpha[v_pix, u_pix] = alpha_gaussian + alpha_pixel * (1 - alpha_gaussian)
                            color[:, v_pix, u_pix] = color_gaussian + color_pixel * (1 - alpha_gaussian)

            if save_frames and (i % 2000 == 0 or i == len(gs) - 1):
                frames.append((color.clone().permute(1, 2, 0).numpy() * 255).astype(np.uint8))

    if save_frames:
        save_gif(frames, output_dir)

    color += bg * (1 - alpha)[None, :, :]
    color = torch.clamp(color, min=0.0, max=1.0)
    rendered_image = (color * 255).to(torch.uint8)

    return rendered_image


def render_circles_alpha_blending(
    gs: GaussianSet,
    camera: Camera,
    radius_px: float = 10.0,
    save_frames: bool = False,
    output_dir: Path | None = None,
) -> UInt8[Tensor, "3 H W"]:
    """Render Gaussians as circles with Gaussian-weighted opacity and alpha blending.

    This is a simplified rendering approach that treats all Gaussians as circles instead
    of ellipses, making it easier to understand the core concepts of depth sorting and
    alpha blending without the complexity of covariance projection. Circle radii are
    inversely proportional to depth, and opacity falls off with a Gaussian weight.

    Features:
    - Circles instead of ellipses (no covariance computation needed)
    - Depth-dependent radius (closer = larger)
    - Gaussian falloff for smooth edges (sigma = radius/2)
    - Back-to-front depth sorting for correct alpha blending
    - Spherical harmonics for view-dependent color

    Args:
        gs: The Gaussian set to render.
        camera: Camera defining the viewpoint and projection.
        radius_px: Base radius in pixels for Gaussians at depth=1.
            Actual radius scales inversely with depth.
        save_frames: Whether to save intermediate frames as a GIF.
        output_dir: Directory to save the GIF frames if save_frames is True.

    Returns:
        Rendered RGB image as uint8 tensor of shape [3, H, W].

    Note:
        This simplified approach demonstrates alpha blending concepts before moving
        to the full covariance-based splatting in render_splats().
    """
    H = camera.height
    W = camera.width

    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32) / 255.0
    bg: Float32[Tensor, "3 H W"] = bg_color[:, None, None].expand(3, H, W).clone()

    means_cam = camera.transform_points_world_to_cam(gs.means)
    means_image = camera.project_points_cam_to_image_plane(means_cam)

    gs, means_cam, means_image = frustum_culling(
        gs,
        camera,
        means_cam,
        means_image,
        cull_image_bounds=True,
        cull_near_far_planes=True,
    )

    gs, means_cam, means_image = sort_gaussians_by_depth_of_means(
        gs,
        means_cam,
        means_image,
        back_to_front=True,
    )

    directions = camera.view_direction_cam_to_mean_world(points_world=gs.means)
    sh_colors = eval_spherical_harmonics_color(
        sh_coefficients=gs.sh_coefficients,
        directions=directions,
        degree=gs.sh_degree,
    )

    color: Float32[Tensor, "3 H W"] = torch.zeros(3, H, W)
    alpha: Float32[Tensor, "H W"] = torch.zeros((H, W), dtype=torch.float32)

    frames: list[UInt8[ndarray, "H W 3"]] = []

    # assign color of each gaussian to the pixel corresponding to its projected mean
    for i in track(range(len(gs)), description="Rendering Gaussians as circles..."):
        opacity_gaussian = gs.opacities[i]
        sh_color = sh_colors[i, :]

        if opacity_gaussian < 0.05:
            continue  # skip nearly transparent gaussians

        u_mean, v_mean = means_image[i]
        depth = means_cam[i, 2]

        # radius inversely proportional to depth
        r = torch.clamp(radius_px / depth, min=1.0, max=radius_px).item()

        # determine bounding box of circle
        u_min = math.floor(u_mean - r)
        u_max = math.ceil(u_mean + r)
        v_min = math.floor(v_mean - r)
        v_max = math.ceil(v_mean + r)

        # iterate over a square region around the projected mean
        for u_pix in range(u_min, u_max):
            for v_pix in range(v_min, v_max):
                # Only write pixel if it's within image bounds
                if 0 <= u_pix < W and 0 <= v_pix < H:
                    # compute squared distance from mean. + 0.5 to center within pixel
                    du = u_pix + 0.5 - u_mean
                    dv = v_pix + 0.5 - v_mean

                    dist_sq = du * du + dv * dv

                    # Only draw pixel if it's within the circle
                    if dist_sq <= r * r:  # squared distance, no need to compute sqrts
                        # Perform the alpha blending with with over operator

                        # choose sigma based on radius
                        r05 = r * 0.25
                        sigma2 = r05 * r05
                        # Gaussian weight for fall off opacity
                        weight = math.exp(-0.5 * dist_sq / sigma2)

                        # Get source alpha and color
                        alpha_circle = opacity_gaussian * weight
                        color_circle = sh_color * alpha_circle

                        # read current alpha and color
                        alpha_pixel = alpha[v_pix, u_pix]
                        color_pixel = color[:, v_pix, u_pix]

                        # compute new alpha and color
                        alpha[v_pix, u_pix] = alpha_circle + alpha_pixel * (1 - alpha_circle)
                        color[:, v_pix, u_pix] = color_circle + color_pixel * (1 - alpha_circle)

        if save_frames and (i % 2000 == 0 or i == len(gs) - 1):
            frames.append((color.clone().permute(1, 2, 0).numpy() * 255).astype(np.uint8))

    if save_frames:
        save_gif(frames, output_dir)

    color += bg * (1 - alpha)[None, :, :]
    rendered_image = (color * 255).to(torch.uint8)

    return rendered_image


def render_circles_fixed_color(
    gs: GaussianSet,
    camera: Camera,
    color_mode: Literal["depth", "palette", "sh"],
    radius_px: float = 10.0,
    save_frames: bool = False,
    output_dir: Path | None = None,
) -> UInt8[Tensor, "3 H W"]:
    """Render Gaussians as opaque circles with fixed colors (no alpha blending).

    This is the simplest rendering approach beyond just dots. Each Gaussian is drawn as
    a circle with a fixed color determined by the color_mode. Circles simply overwrite
    pixels in back-to-front order without any transparency or blending. This makes it
    easy to see the structure of the scene and understand depth sorting.

    Args:
        gs: The Gaussian set to render.
        camera: Camera defining the viewpoint and projection.
        color_mode: How to determine the color of each Gaussian.
        radius_px: Base radius in pixels for Gaussians at depth=1.
            Actual radius scales inversely with depth.
        save_frames: Whether to save intermediate frames as a GIF.
        output_dir: Directory to save the GIF frames if save_frames is True.

    Returns:
        Rendered RGB image as uint8 tensor of shape [3, H, W].

    Note:
        This demonstrates basic rasterization and depth sorting before introducing
        the complexity of alpha blending in render_circles_alpha_blending().
    """
    H = camera.height
    W = camera.width

    bg_color = torch.tensor([0, 0, 0], dtype=torch.uint8)

    means_cam = camera.transform_points_world_to_cam(gs.means)
    means_image = camera.project_points_cam_to_image_plane(means_cam)

    gs, means_cam, means_image = frustum_culling(
        gs,
        camera,
        means_cam,
        means_image,
        cull_image_bounds=True,
        cull_near_far_planes=True,
    )

    gs, means_cam, means_image = sort_gaussians_by_depth_of_means(
        gs,
        means_cam,
        means_image,
        back_to_front=True,
    )

    # initialize rendered image with background color
    rendered_image = bg_color[:, None, None].expand(3, H, W).clone()
    colors_at_means = get_color_at_means(gs, means_cam, camera, mode=color_mode)

    frames: list[UInt8[ndarray, "H W 3"]] = []

    # assign color of each gaussian to the pixel corresponding to its projected mean
    for i in track(range(len(gs)), description="Rendering Gaussians as circles..."):
        if gs.opacities[i] < 0.05:
            continue  # skip nearly transparent gaussians

        u_mean, v_mean = means_image[i]
        depth = means_cam[i, 2]

        # radius inversely proportional to depth
        r = torch.clamp(radius_px / depth, min=1.0, max=radius_px).item()

        # determine bounding box of circle
        u_min = math.floor(u_mean - r)
        u_max = math.ceil(u_mean + r)
        v_min = math.floor(v_mean - r)
        v_max = math.ceil(v_mean + r)

        # iterate over a square region around the projected mean
        for u_pix in range(u_min, u_max):
            for v_pix in range(v_min, v_max):
                # Only write pixel if it's within image bounds
                if 0 <= u_pix < W and 0 <= v_pix < H:
                    # compute squared distance from mean. + 0.5 to center within pixel
                    du = u_pix + 0.5 - u_mean
                    dv = v_pix + 0.5 - v_mean

                    dist_sq = du * du + dv * dv

                    # Only draw pixel if it's within the circle
                    if dist_sq <= r * r:  # squared distance, no need to compute sqrt.
                        rendered_image[:, v_pix, u_pix] = colors_at_means[i, :]

        if save_frames and (i % 2000 == 0 or i == len(gs) - 1):
            frames.append(rendered_image.clone().permute(1, 2, 0).numpy())

    if save_frames:
        save_gif(frames, output_dir)

    return rendered_image


def render_means(
    gs: GaussianSet,
    camera: Camera,
    color_mode: Literal["depth", "palette", "sh"],
    save_frames: bool = False,
    output_dir: Path | None = None,
) -> UInt8[Tensor, "3 H W"]:
    """Render Gaussians as single-pixel dots at their projected mean positions.

    This is the simplest possible rendering: each Gaussian is represented by a single
    pixel at the location where its 3D mean projects onto the 2D image plane. This
    demonstrates the basic camera transformation pipeline (world → camera → image)
    and frustum culling without any rasterization complexity.

    Args:
        gs: The Gaussian set to render.
        camera: Camera defining the viewpoint and projection.
        color_mode: How to determine the color of each Gaussian.
        save_frames: Whether to save intermediate frames as a GIF.
        output_dir: Directory to save the GIF frames if save_frames is True.

    Returns:
        Rendered RGB image as uint8 tensor of shape [3, H, W].

    Note:
        This is the first stage of the renderer, showing only projected points.
        It establishes the coordinate transformation pipeline used in all later stages.
    """
    H = camera.height
    W = camera.width

    bg_color = torch.tensor([0, 0, 0], dtype=torch.uint8)

    # project means of gaussians to image plane
    means_cam = camera.transform_points_world_to_cam(gs.means)
    means_image = camera.project_points_cam_to_image_plane(means_cam)

    # Filter out points that are behind the camera
    gs, means_cam, means_image = frustum_culling(
        gs,
        camera,
        means_cam,
        means_image,
    )

    # initialize rendered image with background color
    rendered_image = bg_color[:, None, None].expand(3, H, W).clone()
    colors_at_means = get_color_at_means(gs, means_cam, camera, mode=color_mode)

    frames: list[UInt8[ndarray, "H W 3"]] = []

    # assign color of each gaussian to the pixel corresponding to its projected mean
    for i in track(range(len(gs)), description="Rendering Gaussians at means..."):
        if gs.opacities[i] < 0.05:
            continue  # skip nearly transparent gaussians

        u, v = means_image[i]
        u_int, v_int = int(u.item()), int(v.item())

        # only draw if the projected pixel is within image bounds
        if 0 <= u_int < W and 0 <= v_int < H:
            rendered_image[:, v_int, u_int] = colors_at_means[i, :]

            if save_frames and (i % 2000 == 0 or i == len(gs) - 1):
                frames.append(rendered_image.clone().permute(1, 2, 0).numpy())

    if save_frames:
        save_gif(frames, output_dir)

    return rendered_image
