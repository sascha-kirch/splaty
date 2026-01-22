from __future__ import annotations

from typing import TYPE_CHECKING

import rich
import torch
from scipy.spatial.transform import Rotation


if TYPE_CHECKING:
    from jaxtyping import Float32
    from torch import Tensor


def quaternion_to_rotation_matrix(
    q: Float32[Tensor, "4"] | Float32[Tensor, "N 4"],
) -> Float32[Tensor, "3 3"] | Float32[Tensor, "N 3 3"]:
    """Convert a normalized quaternion to a rotation matrix.

    Note:
        Supports both single quaternion (4,) and batch of quaternions (N, 4).

    Args:
        q (Float32[Tensor, "4"] | Float32[Tensor, "N 4"]): Quaternion as (qw, qx, qy, qz), already normalized!

    Returns:
        Float32[Tensor, "3 3"] | Float32[Tensor, "N 3 3"]: Corresponding rotation matrix.
    """
    return torch.tensor(Rotation.from_quat(q.numpy(), scalar_first=True).as_matrix(), dtype=q.dtype)


def construct_jacobian_of_perspective_projection(
    fx: float,
    fy: float,
    points_cam: Float32[Tensor, "N 3"],
) -> Float32[Tensor, "N 2 3"]:
    """Constructs the Jacobian of the perspective projection for a set of points in camera coordinates.

    Args:
        fx (float): Focal length in x direction.
        fy (float): Focal length in y direction.
        points_cam (Float32[Tensor, "N 3"]): Points in camera coordinates.

    Returns:
        Float32[Tensor, "N 2 3"]: Jacobian matrices for each point.
    """
    x = points_cam[:, 0]
    y = points_cam[:, 1]
    z = points_cam[:, 2]

    N = points_cam.shape[0]
    jacobian = torch.zeros((N, 2, 3), dtype=points_cam.dtype, device=points_cam.device)

    z_squared = z**2

    jacobian[:, 0, 0] = fx / z
    jacobian[:, 0, 2] = -fx * x / z_squared
    jacobian[:, 1, 1] = fy / z
    jacobian[:, 1, 2] = -fy * y / z_squared

    return jacobian


def renormalize_extrinsics(
    R_world_to_cam: Float32[Tensor, "3 3"],
    t_world_to_cam: Float32[Tensor, "3"],
    normalization_transform: Float32[Tensor, "4 4"],
) -> tuple[Float32[Tensor, "3 3"], Float32[Tensor, "3"]]:
    """Renormalizes the extrinsic parameters using the provided normalization transform.

    Note:
        In gsplat, a scene is normalized by applying a transformation to the camera-to-world matrices. Hence, the
        resulting gaussian and cameras are in a normalized space. If we use cameras in non-normalized space (e.g.
        because we obtained them from colmap's images.txt/images.bin, we first need to normalize the cameras so they
        are in the same reference frame as the trained gaussians, so we need to apply the same transformation that
        gsplat did during training. To obtain the correct camera parameters for rendering, we need to apply the
        normalization transform to the camera extrinsic.

    Args:
        R_world_to_cam (Float32[Tensor, "3 3"]): Rotation matrix from world to camera coordinates.
        t_world_to_cam (Float32[Tensor, "3"]): Translation vector from world to camera coordinates.
        normalization_transform (Float32[Tensor, "4 4"]): Normalization transformation matrix.
    """
    # 1. Convert COLMAP w2c to c2w (original space)
    w2c_original = torch.eye(4)
    w2c_original[:3, :3] = R_world_to_cam
    w2c_original[:3, 3] = t_world_to_cam

    c2w_original = torch.linalg.inv(w2c_original)

    # 2. Apply the normalization transform (same way gsplat did)
    c2w_normalized = normalization_transform @ c2w_original

    # 3. Normalize the rotation part (same way transform_cameras in gsplat does)
    scaling = torch.linalg.norm(c2w_normalized[0, :3])
    c2w_normalized[:3, :3] = c2w_normalized[:3, :3] / scaling

    # 4. Convert back to w2c format for your Camera class
    w2c_normalized = torch.linalg.inv(c2w_normalized)
    R_world_to_cam = w2c_normalized[:3, :3]
    t_world_to_cam = w2c_normalized[:3, 3]

    return R_world_to_cam, t_world_to_cam


def invert_batched_2x2_matrices(
    matrices: Float32[Tensor, "N 2 2"],
) -> Float32[Tensor, "N 2 2"]:
    """Inverts a batch of 2x2 matrices.

    Args:
        matrices (Float32[Tensor, "N 2 2"]): Batch of 2x2 matrices to invert.

    Returns:
        Float32[Tensor, "N 2 2"]: Inverted matrices.
    """
    det = matrices[:, 0, 0] * matrices[:, 1, 1] - matrices[:, 0, 1] * matrices[:, 1, 0]

    if neg_dets := (det < 0).nonzero().tolist():
        rich.console.Console().log(
            f"Warning: Found {len(neg_dets)} Gaussians with non-invertible 2D covariance matrices (determinant < 0). "
            f"Affected indices: {neg_dets}. Rendering is likely to fail or produce artifacts.",
            style="bold yellow",
        )

    det = torch.clamp(det, min=1e-10)  # avoid division by zero
    inv_det = 1.0 / det

    inv_matrices = torch.zeros_like(matrices)
    inv_matrices[:, 0, 0] = matrices[:, 1, 1] * inv_det
    inv_matrices[:, 0, 1] = -matrices[:, 0, 1] * inv_det
    inv_matrices[:, 1, 0] = -matrices[:, 1, 0] * inv_det
    inv_matrices[:, 1, 1] = matrices[:, 0, 0] * inv_det

    return inv_matrices


def compute_batched_eigenvalues_2x2(
    matrices: Float32[Tensor, "N 2 2"],
) -> Float32[Tensor, "N 2"]:
    """Computes the eigenvalues of a batch of 2x2 matrices.

    Args:
        matrices (Float32[Tensor, "N 2 2"]): Batch of 2x2 matrices.

    Returns:
        Float32[Tensor, "N 2"]: Eigenvalues for each matrix.
    """
    """ a = matrices[:, 0, 0]
    b = matrices[:, 0, 1]
    c = matrices[:, 1, 0]
    d = matrices[:, 1, 1]

    trace = a + d
    determinant = torch.clamp(a * d - b * c, min=1e-12)  # avoid negative values due to numerical issues
    discriminant = torch.clamp(trace**2 - 4 * determinant, min=0.0)
    sqrt_discriminant = torch.sqrt(discriminant)

    lambda1 = (trace + sqrt_discriminant) / 2
    lambda2 = (trace - sqrt_discriminant) / 2
    eigenvalues = torch.stack([lambda1, lambda2], dim=-1)
    return eigenvalues """
    eigvals = torch.linalg.eigvalsh(matrices)
    eigvals = torch.clamp(eigvals, min=1e-6)
    return eigvals
