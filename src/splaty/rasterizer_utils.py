from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING
from typing import Literal

import rich
import torch
from matplotlib import cm
from torch import Tensor

from splaty.gaussian import GaussianSet
from splaty.spatial_math import compute_batched_eigenvalues_2x2
from splaty.spatial_math import construct_jacobian_of_perspective_projection
from splaty.spherical_harmonics import eval_spherical_harmonics_color


if TYPE_CHECKING:
    from jaxtyping import Float32
    from jaxtyping import UInt8
    from jaxtyping import UInt16
    from torch import Tensor

    from splaty.camera import Camera


def compute_conics(
    covariances_image: Float32[Tensor, "N 2 2"],
) -> Float32[Tensor, "N 3"]:
    """Compute conic representation from 2D covariance matrices.

    Args:
        covariances_image: Covariance matrices of the Gaussians in image space.

    Returns:
        Float32[Tensor, "N 3"]: Conic representation for each Gaussian.
    """
    a: Float32[Tensor, "N"] = covariances_image[:, 0, 0]
    b: Float32[Tensor, "N"] = (covariances_image[:, 0, 1] + covariances_image[:, 1, 0]) / 2
    c: Float32[Tensor, "N"] = covariances_image[:, 1, 1]

    det: Float32[Tensor, "N"] = a * c - b * b

    if neg_dets := (det < 0).nonzero().tolist():
        rich.console.Console().log(
            f"Warning: Found {len(neg_dets)} Gaussians with non-invertible 2D covariance matrices (determinant < 0). "
            f"Affected indices: {neg_dets}. Rendering is likely to fail or produce artifacts.",
            style="bold yellow",
        )

    det = det.clamp(min=1e-10)

    # Note that this is basically the inverse of the covariance matrix elements, arranged in conic form
    conics: Float32[Tensor, "N 4"] = torch.stack(
        [
            c / det,
            -b / det,
            a / det,
        ],
        dim=1,
    )
    return conics


def compute_axis_aligned_bounding_box(
    gaussian_means_image: Float32[Tensor, "N 2"],
    camera: Camera,
    covariances_image: Float32[Tensor, "N 2 2"],
    sigma_multiplier: float,
    use_simple_radius: bool = False,
) -> UInt16[Tensor, "N 4"]:
    """Compute axis-aligned bounding boxes for Gaussians in image space.

    Args:
        gaussian_means_image: Means of the Gaussians projected onto the image plane.
        camera: Camera object.
        covariances_image: Covariance matrices of the Gaussians in image space.
        sigma_multiplier: Multiplier for the standard deviation to define the bounding box size.
        use_simple_radius: Whether to use a simplified radius computation based on covariance diagonal elements.

    Returns:
        UInt16[Tensor, "N 4"]: Axis-aligned bounding boxes for each Gaussian in the format (u_min, v_min, u_max, v_max).
    """
    if use_simple_radius:
        # compute radii along u and v axes using the covariance diagonal elements. Those elements represent the variance
        # along the image axes. It's also an overestimation, but faster to compute than eigenvalues.
        # Also individual radii along u and v allows for tighter bounding boxes.
        radii_u = sigma_multiplier * torch.sqrt(covariances_image[:, 0, 0])
        radii_v = sigma_multiplier * torch.sqrt(covariances_image[:, 1, 1])

        # compute corners axis aligned bounding box using the radii and the mean in image space
        u_min = torch.floor(gaussian_means_image[:, 0] - radii_u)
        v_min = torch.floor(gaussian_means_image[:, 1] - radii_v)
        u_max = torch.ceil(gaussian_means_image[:, 0] + radii_u)
        v_max = torch.ceil(gaussian_means_image[:, 1] + radii_v)

        aabb = torch.stack([u_min, v_min, u_max, v_max], dim=1)  # (N, 4) with (u_min, v_min, u_max, v_max)
    else:
        # The eigenvalues give us the variance along the principal axes of the Gaussian in image space
        eigenvalues = compute_batched_eigenvalues_2x2(covariances_image)

        # this is an simplified estimation of the radius based on the covariance.
        # A better approach would be to compute the eigenvalues of the 2D covariance matrix and use them to define
        # the ellipse axes. Here, we just use the maximum variance direction to define a circular radius, because it
        # saves computing eigenvalues and eigenvectors.
        # we can also pre-compute in a batched fashion those now and not in the inner loop
        radii = sigma_multiplier * torch.sqrt(torch.max(eigenvalues, dim=1).values)

        # compute corners axis aligned bounding box using the radii and the mean in image space
        uv_min = torch.floor(gaussian_means_image - radii.unsqueeze(-1))
        uv_max = torch.ceil(gaussian_means_image + radii.unsqueeze(-1))

        aabb = torch.cat([uv_min, uv_max], dim=1)  # (N, 4) with (u_min, v_min, u_max, v_max)

    # clamp to image size, in case of very large splats.
    # e.g. depth and perspective projection, and the jacobian dividing by z, so gaussians close to camera,
    # have huge jacobians, and hence result in large covariances in image space.
    aabb[:, 0] = torch.clamp(aabb[:, 0], min=0, max=camera.width - 1)  # u_min
    aabb[:, 1] = torch.clamp(aabb[:, 1], min=0, max=camera.height - 1)  # v_min
    aabb[:, 2] = torch.clamp(aabb[:, 2], min=0, max=camera.width - 1)  # u_max
    aabb[:, 3] = torch.clamp(aabb[:, 3], min=0, max=camera.height - 1)  # v_max

    return aabb.to(torch.uint16)


def frustum_culling(
    gs: GaussianSet,
    camera: Camera,
    means_cam: Float32[Tensor, "N 3"],
    means_image: Float32[Tensor, "N 2"],
    *,
    cull_image_bounds: bool = False,
    cull_near_far_planes: bool = False,
    z_buffer_margin: float = 0.0,
    fov_margin_ratio: float = 0.0,
) -> tuple[GaussianSet, Float32[Tensor, "M 3"], Float32[Tensor, "M 2"]]:
    """Perform frustum culling to remove Gaussians that are outside the camera's view frustum.

    Args:
        gs: GaussianSet to be culled.
        camera: Camera object defining the view frustum.
        means_cam: Means of the Gaussians in camera coordinates.
        means_image: Means of the Gaussians projected onto the image plane.
        cull_image_bounds: Whether to cull Gaussians with their means outside the image bounds.
        cull_near_far_planes: Whether to cull Gaussians with their means outside the near and far planes.
        z_buffer_margin: Margin to add to near and far plane culling.
        fov_margin_ratio: Margin ratio to add to the field of view culling.

    Returns:
        A tuple containing the culled GaussianSet and the corresponding transformed means in camera space and the
        projected means on the image plane.
    """
    depth = means_cam[:, 2]

    mask = torch.ones(means_cam.shape[0], dtype=torch.bool)

    if cull_near_far_planes:
        mask &= depth >= camera.near_plane - z_buffer_margin
        mask &= depth <= camera.far_plane + z_buffer_margin
    else:
        mask &= depth > 0.0  # only keep points in front of the camera

    if fov_margin_ratio > 0.0:
        x = means_cam[:, 0]
        y = means_cam[:, 1]

        margin_x = camera.fov_x * fov_margin_ratio
        margin_y = camera.fov_y * fov_margin_ratio

        mask &= torch.abs(x / depth) <= torch.tan(torch.tensor((camera.fov_x / 2) + margin_x))
        mask &= torch.abs(y / depth) <= torch.tan(torch.tensor((camera.fov_y / 2) + margin_y))

    if cull_image_bounds:
        u = means_image[:, 0]
        v = means_image[:, 1]
        mask &= (u >= 0) & (u < camera.width)
        mask &= (v >= 0) & (v < camera.height)

    # Apply masks and ensure data is contiguous

    gs_culled = GaussianSet(
        means=gs.means[mask].contiguous(),
        quats=gs.quats[mask].contiguous(),
        scales=gs.scales[mask].contiguous(),
        opacities=gs.opacities[mask].contiguous(),
        sh_coefficients=gs.sh_coefficients[mask].contiguous(),
    )

    means_cam_culled = means_cam[mask].contiguous()

    means_image_culled = means_image[mask].contiguous()

    rich.console.Console().log(
        f"Culled {len(gs) - len(gs_culled)} Gaussians during frustum culling. Gaussians remaining: {len(gs_culled)}"
    )

    return (gs_culled, means_cam_culled, means_image_culled)


def sort_gaussians_by_depth_of_means(
    gs: GaussianSet,
    means_cam: Float32[Tensor, "N 3"],
    means_image: Float32[Tensor, "N 2"],
    *,
    back_to_front: bool = True,
) -> tuple[GaussianSet, Float32[Tensor, "M 3"], Float32[Tensor, "M 2"]]:
    """Sort Gaussians by depth of their means in camera space.

    Args:
        gs: GaussianSet to be sorted.
        means_cam: Means of the Gaussians in camera coordinates.
        means_image: Means of the Gaussians projected onto the image plane.
        back_to_front: Whether to sort from back to front (furthest to nearest). If False, sorts from front to back.

    Returns:
        A tuple containing the sorted GaussianSet and the corresponding transformed means in camera space and the
        projected means on the image plane.
    """
    depths = means_cam[:, 2]
    sorted_indices = torch.argsort(depths, descending=back_to_front)

    gs_sorted = GaussianSet(
        means=gs.means[sorted_indices].contiguous(),
        quats=gs.quats[sorted_indices].contiguous(),
        scales=gs.scales[sorted_indices].contiguous(),
        opacities=gs.opacities[sorted_indices].contiguous(),
        sh_coefficients=gs.sh_coefficients[sorted_indices].contiguous(),
    )

    means_cam_sorted = means_cam[sorted_indices].contiguous()

    means_image_sorted = means_image[sorted_indices].contiguous()

    return (gs_sorted, means_cam_sorted, means_image_sorted)


def get_color_at_means(
    gs: GaussianSet,
    means_cam_roi: Float32[Tensor, "N 3"],
    camera: Camera,
    mode: Literal["depth", "opacity", "palette", "sh"] = "depth",
) -> UInt8[Tensor, "N 3"]:
    """Get the color of Gaussians at their means in camera space.

    Args:
        gs: GaussianSet containing the Gaussians.
        means_cam_roi: Means of the Gaussians in camera coordinates.
        camera: Camera object.
        mode: Mode for color assignment. Options are "depth", "opacity", "palette", or "sh".

    Returns:
        UInt8[Tensor, "N 3"]: Colors of the Gaussians at their means.
    """
    if mode == "depth":
        depths = means_cam_roi[:, 2]
        return (
            torch.clamp(255 - (depths / torch.max(depths)) * 255, min=0, max=255)
            .unsqueeze(-1)
            .repeat(1, 3)
            .to(torch.uint8)
        )
    if mode == "opacity":
        opacities = gs.opacities
        norm_opacities = (opacities - torch.min(opacities)) / (torch.max(opacities) - torch.min(opacities) + 1e-8)
        colormap = cm.get_cmap("hsv")  # E.g. Spectral, RdYlGn, hsv
        colors = colormap(norm_opacities.numpy())[:, :3]  # Get RGB values
        return torch.from_numpy(colors * 255).to(torch.uint8)

    if mode == "palette":
        depths = means_cam_roi[:, 2]
        norm_depths = (depths - torch.min(depths)) / (torch.max(depths) - torch.min(depths) + 1e-8)
        colormap = cm.get_cmap("hsv")  # E.g. Spectral, RdYlGn, hsv
        colors = colormap(norm_depths.numpy())[:, :3]  # Get RGB values
        return torch.from_numpy(colors * 255).to(torch.uint8)

    if mode == "sh":
        directions = camera.view_direction_cam_to_mean_world(points_world=gs.means)
        colors = eval_spherical_harmonics_color(
            sh_coefficients=gs.sh_coefficients,
            directions=directions,
            degree=gs.sh_degree,
        )
        return (colors * 255).to(torch.uint8)

    return torch.ones((means_cam_roi.shape[0], 3), dtype=torch.uint8) * 255  # default to white if unknown mode


def approximate_2d_covariances(
    points_cam: Float32[Tensor, "N 3"],
    covariances_cam: Float32[Tensor, "N 3 3"],
    camera: Camera,
    eps_2d: float = 0.3,
) -> Float32[Tensor, "N 2 2"]:
    """Approximate the 2D covariances of the projected Gaussians on the image plane.

    Note:
        We use the Jacobian of the perspective projection to approximate the 2D covariance matrices on the image plane
        at the projected mean of each Gaussian.

    Args:
        points_cam: Points in camera coordinates.
        covariances_cam: Covariance matrices in camera coordinates.
        camera: Camera object.
        eps_2d: Small value to add to the diagonal of the 2D covariance matrices to ensure they are invertible.
            Default is 0.3.

    Returns:
        Float32[Tensor, "N 2 2"]: Approximated 2D covariance matrices on the image plane.
    """
    jacobians: Float32[Tensor, "N 2 3"] = construct_jacobian_of_perspective_projection(
        fx=camera.fx,
        fy=camera.fy,
        points_cam=points_cam,
    )

    # batched version to compute 2d covariances
    covariances_2d = torch.einsum("nij,njk,nlk->nil", jacobians, covariances_cam, jacobians)  # J @ cov @ J.T

    if eps_2d > 0.0:
        # To avoid sub-pixel gaussians with non-invertible covariance matrices, add a small value to the diagonal.
        # 0.3 as in gsplat, because sqrt(0.3)[variance] ~= 0.5477 pixels [sigma], which is about half a pixel.
        # This is not for numerical stability, but to ensure a minimum size of the splat on the image plane.
        covariances_2d += eps_2d * torch.eye(2, dtype=covariances_2d.dtype, device=covariances_2d.device)[None, :, :]

    # N = points_cam.shape[0]
    # covariances_2d = torch.zeros((N, 2, 2), dtype=points_cam.dtype, device=points_cam.device)
    # for i in range(N):
    #     J = jacobians[i]  # (2, 3)
    #     cov_cam = covariances_cam[i]  # (3, 3)
    #     covariances_2d[i] = J @ cov_cam @ J.T  # (2, 2)

    return covariances_2d


def assign_gaussians_to_tiles(
    axis_aligned_bounding_boxes: Float32[Tensor, "N 4"],
    tile_size: int,
    image_width: int,
    image_height: int,
) -> dict[tuple[int, int], list[int]]:
    """ Assign Gaussians to image tiles based on their axis-aligned bounding boxes.

    Important: This is not how it is done in 3DGS, which needs to deal with GPU constraints and performance optimizations.
    Here we do a simple CPU-based assignment for clarity and simplicity.

    Args:
        axis_aligned_bounding_boxes: Axis-aligned bounding boxes for each Gaussian in the format (u_min, v_min, u_max, v_max).
        tile_size: Size of each tile (assumed square).
        image_width: Width of the image.
        image_height: Height of the image.

    Returns:
        dict[tuple[int, int], list[int]]: A dictionary mapping tile coordinates (tile_u, tile_v) to lists of Gaussian
            indices that overlap with each tile.
    """
    N = axis_aligned_bounding_boxes.shape[0]

    tiles_u = math.ceil(image_width / tile_size)
    tiles_v = math.ceil(image_height / tile_size)

    tile_dict: dict[tuple[int, int], list[int]] = defaultdict(list)

    for i in range(N):
        u_min, v_min, u_max, v_max = axis_aligned_bounding_boxes[i].tolist()

        # determine which tiles the bounding box overlaps with
        tile_min_u = int(u_min) // tile_size
        tile_min_v = int(v_min) // tile_size
        tile_max_u = int(u_max) // tile_size
        tile_max_v = int(v_max) // tile_size

        # iterate over all tiles that the bounding box overlaps with
        for tu in range(tile_min_u, tile_max_u + 1):
            for tv in range(tile_min_v, tile_max_v + 1):
                # only add if tile is within image bounds represented as tiles
                if 0 <= tu < tiles_u and 0 <= tv < tiles_v:
                    tile_dict[(tu, tv)].append(i)

    return tile_dict
