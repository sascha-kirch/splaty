from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from splaty.spatial_math import quaternion_to_rotation_matrix
from splaty.spherical_harmonics import compute_sh_degree


if TYPE_CHECKING:
    from jaxtyping import Float32
    from torch import Tensor


class GaussianSet:
    """Set of 3D Gaussians representing a scene."""

    def __init__(
        self,
        means: Float32[Tensor, "N 3"],
        quats: Float32[Tensor, "N 4"],
        scales: Float32[Tensor, "N 3"],
        opacities: Float32[Tensor, "N"],
        sh_coefficients: Float32[Tensor, "N S 3"],
    ):
        """Initializes the GaussianSet.

        Args:
            means (Float32[Tensor, "N 3"]): Mean positions of the Gaussians.
            quats (Float32[Tensor, "N 4"]): Quaternions representing orientations of the Gaussians.
            scales (Float32[Tensor, "N 3"]): Scales along each axis for the Gaussians.
            opacities (Float32[Tensor, "N"]): Opacity values for the Gaussians.
            sh_coefficients (Float32[Tensor, "N S 3"]): Spherical harmonics coefficients for color representation.
        """
        self.means = means
        self.quats = quats
        self.scales = scales
        self.opacities = opacities
        self.sh_coefficients = sh_coefficients

        self.sh_degree = compute_sh_degree(sh_coefficients)
        self.covariances: Float32[Tensor, "N 3 3"] = self.compute_covariances(quats, scales)

    @classmethod
    def from_gsplat_ckpt(
        cls,
        ckpt_path: Path,
        device: torch.device = "cpu",
    ) -> GaussianSet:
        """Generate GaussianSet from a Gsplat checkpoint.

        Args:
            ckpt_path (Path): Path to the Gsplat checkpoint.
            device (torch.device, optional): Device to load the checkpoint onto. Defaults to "cpu".

        Returns:
            Self: An instance of GaussianSet initialized with data from the checkpoint.
        """
        ckpt = torch.load(ckpt_path, map_location=device)["splats"]

        return cls(
            means=ckpt["means"],
            quats=F.normalize(ckpt["quats"], p=2, dim=-1),
            scales=torch.exp(ckpt["scales"]),
            opacities=torch.sigmoid(ckpt["opacities"]),
            sh_coefficients=torch.cat([ckpt["sh0"], ckpt["shN"]], dim=-2),
        )

    @staticmethod
    def compute_covariances(
        quats: Float32[Tensor, "N 4"],
        scales: Float32[Tensor, "N 3"],
    ) -> Float32[Tensor, "N 3 3"]:
        """Compute covariance matrices from quaternions and scales.

        Note:
            The covariance matrix in world coordinates is given by:
                Cov = R * S * R^T
            where R is the rotation matrix from the quaternion, and S is the covariance matrix in local coordinates.

            In local coordinates, the covariance matrix S is diagonal with the variances along each axis:
                S = diag(sx^2, sy^2, sz^2)

            So, the scales represent the standard deviations along the local axes in 3d of the Gaussian. Those then need to
            be squared to get the variances for the covariance matrix.
            The original equation from the 3DGS paper is:
            Sigma = R * S * S^T * R^T
            But since S is real and diagonal, S * S^T = S^2 = diag(sx^2, sy^2, sz^2).


        Args:
            quats (Float32[Tensor, "N 4"]): Quaternions representing orientations.
            scales (Float32[Tensor, "N 3"]): Scales along each axis.

        Returns:
            Float32[Tensor, "N 3 3"]: Covariance matrices for each Gaussian.
        """
        N = quats.shape[0]

        # batched version to compute the covariances
        R = quaternion_to_rotation_matrix(quats)  # (N, 3, 3)
        S = torch.zeros((N, 3, 3), dtype=scales.dtype, device=scales.device)
        S[:, 0, 0] = scales[:, 0] ** 2
        S[:, 1, 1] = scales[:, 1] ** 2
        S[:, 2, 2] = scales[:, 2] ** 2
        covariances = torch.einsum("nij,njk,nlk->nil", R, S, R)  # R @ S @ R^T

        return covariances

    def __len__(self) -> int:
        return self.means.shape[0]


if __name__ == "__main__":
    gs = GaussianSet.from_gsplat_ckpt(
        ckpt_path=Path("~/git/splaty/data/bonsai/gsplat/ckpt_29999.pt").expanduser(),
        device="cpu",
    )
    print(f"{gs.means.shape =}")
    print(f"{gs.quats.shape =}")
    print(f"{gs.scales.shape =}")
    print(f"{gs.opacities.shape =}")
    print(f"{gs.sh_coefficients.shape =}")
    print(f"{gs.sh_degree =}")
