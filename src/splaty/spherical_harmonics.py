from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from jaxtyping import Float32
    from torch import Tensor


# Normalization constants N_l^m for real spherical harmonics basis functions.

# Nomenclature:
# Nlm = N_l^m, e.g. N2m1 = N_2^{-1}

# l=0, m=0
N00 = 0.28209479177387814  # sqrt(1/(4*pi))

# l=1, m=[-1,0,1]
N1m1 = 0.4886025119029199  # sqrt(3/(4*pi))
N10 = 0.4886025119029199  # sqrt(3/(4*pi))
N11 = 0.4886025119029199  # sqrt(3/(4*pi))

# l=2, m=[-2,-1,0,1,2]
N2m2 = 1.0925484305920792  # sqrt(15/(4*pi))
N2m1 = -1.0925484305920792  # sqrt(15/(4*pi))
N20 = 0.31539156525252005  # sqrt(5/(16*pi))
N21 = -1.0925484305920792  # sqrt(15/(4*pi))
N22 = 0.5462742152960396  # sqrt(15/(16*pi))

# l=3, m=[-3,-2,-1,0,1,2,3]
N3m3 = -0.5900435899266435  # sqrt(35/(32*pi))
N3m2 = 2.890611442640554  # sqrt(105/(4*pi))
N3m1 = -0.4570457994644658  # sqrt(21/(64*pi))
N30 = 0.3731763325901154  # sqrt(7/(16*pi))
N31 = -0.4570457994644658  # sqrt(21/(64*pi))
N32 = 1.445305721320277  # sqrt(105/(16*pi))
N33 = -0.5900435899266435  # sqrt(35/(32*pi))


def eval_spherical_harmonics_color(
    sh_coefficients: Float32[Tensor, "N S 3"],
    directions: Float32[Tensor, "N 3"],
    degree: int,
) -> Float32[Tensor, "N 3"]:
    """Evaluate spherical harmonics at given directions to obtain view-dependent color.

    Computes the color C from SH coefficients c and direction d using:
        C(d) = sum_{l=0}^{L} sum_{m=-l}^{l} c_{l,m} * Y_l^m(d)
    where Y_l^m(d) are the real spherical harmonics basis functions.

    For real SH basis functions: Y_l^m(d) = N_l^m * P_l^m(d), where P_l^m(d) are associated Legendre polynomials
    (e.g. x², xy, xz for l=2).

    Note:
        Higher degrees allow more detailed color variation but are more expensive to compute.
        This implementation uses real SH basis functions with x,y,z formulation (normalized
        direction vector on unit sphere), which is common in graphics applications.
        Since SH values can be negative, the result is shifted by +0.5 and clamped to [0, inf).

    Args:
        sh_coefficients: SH coefficients of shape (N, S, 3).
        directions: Directions to evaluate SH at, shape (N, 3). Expected to be normalized.
        degree: Degree of the spherical harmonics.

    Returns:
        Float32[Tensor, "N 3"]: Evaluated colors at the given directions.
    """
    color: Float32[Tensor, "N 3"] = N00 * sh_coefficients[:, 0, :]

    if degree >= 1:
        # Plolynomials for l=1
        x, y, z = directions[:, 0:1], directions[:, 1:2], directions[:, 2:3]  # x,y,z shape (N, 1)

        Y1m1 = N1m1 * y
        Y10 = N10 * z
        Y11 = N11 * x

        color += sh_coefficients[:, 1, :] * Y1m1
        color += sh_coefficients[:, 2, :] * Y10
        color += sh_coefficients[:, 3, :] * Y11

        if degree >= 2:
            # Polynomials for l=2
            xx, yy, zz = x * x, y * y, z * z
            xy, xz, yz = x * y, x * z, y * z

            Y2m2 = N2m2 * xy
            Y2m1 = N2m1 * yz
            Y20 = N20 * (2.0 * zz - xx - yy)
            Y21 = N21 * xz
            Y22 = N22 * (xx - yy)

            color += sh_coefficients[:, 4, :] * Y2m2
            color += sh_coefficients[:, 5, :] * Y2m1
            color += sh_coefficients[:, 6, :] * Y20
            color += sh_coefficients[:, 7, :] * Y21
            color += sh_coefficients[:, 8, :] * Y22

            if degree >= 3:
                # reusing polynomials from l=2, and l=1
                Y3m3 = N3m3 * y * (3 * xx - yy)
                Y3m2 = N3m2 * xy * z
                Y3m1 = N3m1 * y * (4 * zz - xx - yy)
                Y30 = N30 * z * (2 * zz - 3 * xx - 3 * yy)
                Y31 = N31 * x * (4 * zz - xx - yy)
                Y32 = N32 * z * (xx - yy)
                Y33 = N33 * x * (xx - 3 * yy)

                color += sh_coefficients[:, 9, :] * Y3m3
                color += sh_coefficients[:, 10, :] * Y3m2
                color += sh_coefficients[:, 11, :] * Y3m1
                color += sh_coefficients[:, 12, :] * Y30
                color += sh_coefficients[:, 13, :] * Y31
                color += sh_coefficients[:, 14, :] * Y32
                color += sh_coefficients[:, 15, :] * Y33
    return torch.clamp_min(color + 0.5, 0.0)  # shift to positive range


def compute_sh_degree(
    sh_coefficients: Float32[Tensor, "N S 3"],
) -> int:
    """Compute the spherical harmonics degree from SH coefficients.

    Note:
        The degree L of the spherical harmonics determines the number of coefficients S
        by the formula S = (L + 1)^2. For example: L=0 -> S=1, L=1 -> S=4, L=2 -> S=9, L=3 -> S=16.
        To find L, we rearrange the formula to L = sqrt(S) - 1.

    Args:
        sh_coefficients: SH coefficients of shape (N, S, 3).

    Returns:
        int: The spherical harmonics degree.
    """
    return int(math.sqrt(sh_coefficients.shape[1]) - 1)
