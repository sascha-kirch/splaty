from __future__ import annotations

import math
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import torch


if TYPE_CHECKING:
    from jaxtyping import Float32
    from jaxtyping import UInt8
    from torch import Tensor


def _opacity_falloff_circles(
    base_opacity: float,
    radius: float,
    falloff_factor: float = 0.25,
):
    # Matplotlib plot, where on the x axis is the distance from center in pixels, and on the y axis is the opacity contribution.
    # I also want to have the fall-off
    fig = plt.figure()
    ax = fig.add_subplot(111)
    distances = list(range(0, int(radius) + 1))
    alphas = []
    for d in distances:
        dist_sq = d * d
        sigma2 = (radius * falloff_factor) ** 2
        weight = math.exp(-0.5 * dist_sq / sigma2)
        opacity = base_opacity * weight
        alphas.append(opacity)
    ax.plot(distances, alphas)
    ax.set_xlabel("Distance from center (pixels)")
    ax.set_ylabel("Opacity contribution")
    ax.set_title(f"Opacity fall-off for circle with radius {radius} and base opacity {base_opacity}")
    plt.grid()
    plt.savefig(f"opacity_falloff_radius_{radius}_baseopacity_{base_opacity}_falloff_factor_{falloff_factor}.png")


def _opacity_falloff_gaussians(
    base_opacity: float,
    k: float,
):
    # Matplotlib plot, where on the x axis is the distance from center in pixels, and on the y axis is the opacity contribution.
    # I also want to have the fall-off
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sigmas = [0.1 * i for i in range(0, int(k**2 * 10) + 1)]
    alphas = []
    for s in sigmas:
        if s <= k**2:
            weight = math.exp(-0.5 * s)
            opacity = base_opacity * weight
            alphas.append(opacity)
    ax.plot(sigmas, alphas)
    ax.set_xlabel("Distance from center (sigma squared units)")
    ax.set_ylabel("Opacity contribution")
    ax.set_title(f"Opacity fall-off for gaussian with k {k} and base opacity {base_opacity}")
    plt.grid()
    plt.savefig(f"opacity_falloff_gaussian_k_{k}_baseopacity_{base_opacity}.png")


def _debug_3_circles_alpha_blending(
    with_fall_off: bool = False,
    with_checkerboard: bool = True,
    base_opacity: float = 1.0,
) -> UInt8[Tensor, "3 H W"]:
    H, W = 600, 1200
    # H, W = 600, 600
    # create checkerboard background
    bg_color = torch.tensor([50, 50, 50], dtype=torch.float32) / 255.0
    bg: Float32[Tensor, "3 H W"] = bg_color[:, None, None].expand(3, H, W).clone()

    if with_checkerboard:
        bg_color_light = torch.tensor([65, 65, 65], dtype=torch.float32) / 255.0
        for u in range(H):
            for v in range(W):
                if ((u // 100) + (v // 100)) % 2 == 0:
                    bg[:, u, v] = bg_color_light

    circles: dict[str, dict] = {
        "circle1": {
            "center": (300, 500),
            "radius": 150,
            "color": torch.tensor([240, 140, 0]) / 255,
            "opacity": base_opacity,
        },
        "circle2": {
            "center": (200, 700),
            "radius": 150,
            "color": torch.tensor([47, 158, 68]) / 255,
            "opacity": base_opacity,
        },
        "circle3": {
            "center": (400, 700),
            "radius": 150,
            "color": torch.tensor([25, 113, 194]) / 255,
            "opacity": base_opacity,
        },
    }
    # circles: dict[str, dict] = {
    #     "circle1": {"center": (300, 200), "radius": 150, "color": torch.tensor([1.0, 0.0, 0.0]), "opacity": 1.0},
    #     "circle2": {"center": (200, 400), "radius": 150, "color": torch.tensor([0.0, 1.0, 0.0]), "opacity": 1.0},
    #     "circle3": {"center": (400, 400), "radius": 150, "color": torch.tensor([0.0, 0.0, 1.0]), "opacity": 1.0},
    # }

    color: Float32[Tensor, "3 H W"] = torch.zeros((3, H, W), dtype=torch.float32)
    alpha: Float32[Tensor, "H W"] = torch.zeros((H, W), dtype=torch.float32)

    for circle in circles.values():
        radius = circle["radius"]
        color_circle = circle["color"]
        opacity_circle = torch.tensor(circle["opacity"])

        v_center, u_center = circle["center"]

        u_min = math.floor(u_center - radius)
        u_max = math.ceil(u_center + radius)
        v_min = math.floor(v_center - radius)
        v_max = math.ceil(v_center + radius)

        # iterate over a square region around the projected mean
        for u_pix in range(u_min, u_max + 1):
            for v_pix in range(v_min, v_max + 1):
                # Only write pixel if it's within image bounds
                if 0 <= u_pix < W and 0 <= v_pix < H:
                    du = u_pix + 0.5 - u_center
                    dv = v_pix + 0.5 - v_center

                    dist_sq = (du) ** 2 + (dv) ** 2

                    # Only draw pixel if it's within the circle
                    if dist_sq <= radius**2:  # squared distance, no need to compute sqrt.
                        if with_fall_off:
                            # choose sigma based on radius
                            sigma2 = (radius * 0.35) ** 2
                            # Gaussian weight for fall off opacity
                            weight = math.exp(-0.5 * dist_sq / sigma2)

                            # Get source alpha and color
                            densitiy = opacity_circle * weight

                            transmittance = torch.exp(-densitiy)
                            alpha_circle = 1 - transmittance
                        else:
                            alpha_circle = opacity_circle

                        color_src = color_circle * alpha_circle

                        # read old alpha and color
                        alpha_pixel = alpha[v_pix, u_pix]
                        color_pixel = color[:, v_pix, u_pix]

                        # compute new alpha and color
                        alpha[v_pix, u_pix] = alpha_circle + alpha_pixel * (1 - alpha_circle)
                        color[:, v_pix, u_pix] = color_src + color_pixel * (1 - alpha_circle)

    color += bg * (1 - alpha)[None, :, :]
    rendered_image = (color * 255).to(torch.uint8)
    return rendered_image


def _debug_circle() -> UInt8[Tensor, "3 H W"]:
    H, W = 10, 12
    # create checkerboard background
    bg_color = torch.tensor([50, 50, 50], dtype=torch.float32) / 255.0
    bg: Float32[Tensor, "3 H W"] = bg_color[:, None, None].expand(3, H, W).clone()
    bg_color_light = torch.tensor([200, 200, 200], dtype=torch.float32) / 255.0
    bbox_color = torch.tensor([0, 0, 200], dtype=torch.float32) / 255.0
    for u in range(H):
        for v in range(W):
            if (u + v) % 2 == 0:
                bg[:, u, v] = bg_color_light

    circle = {"center": (4.8, 6.8), "radius": 3, "color": torch.tensor([1.0, 0.0, 0.0]), "opacity": 0.7}

    color: Float32[Tensor, "3 H W"] = torch.zeros((3, H, W), dtype=torch.float32)
    alpha: Float32[Tensor, "H W"] = torch.zeros((H, W), dtype=torch.float32)

    radius = circle["radius"]
    color_circle = circle["color"]
    opacity_circle = torch.tensor(circle["opacity"])

    v, u = circle["center"]

    u_min = math.floor(u - radius)
    u_max = math.ceil(u + radius)
    v_min = math.floor(v - radius)
    v_max = math.ceil(v + radius)

    # iterate over a square region around the projected mean
    for u_pix in range(u_min, u_max + 1):
        for v_pix in range(v_min, v_max + 1):
            # Only write pixel if it's within image bounds
            if 0 <= u_pix < W and 0 <= v_pix < H:
                du = u_pix + 0.5 - u
                dv = v_pix + 0.5 - v

                dist_sq = (du) ** 2 + (dv) ** 2

                # Only draw pixel if it's within the circle
                if dist_sq <= radius**2:  # squared distance, no need to compute sqrt.
                    alpha_circle = torch.clamp(opacity_circle, min=0.0, max=1.0)
                    color_src = color_circle * alpha_circle

                    # read old alpha and color
                    alpha_pixel = alpha[v_pix, u_pix]
                    color_pixel = color[:, v_pix, u_pix]

                    # compute new alpha and color
                    alpha[v_pix, u_pix] = alpha_circle + alpha_pixel * (1 - alpha_circle)
                    color[:, v_pix, u_pix] = color_src + color_pixel * (1 - alpha_circle)
                else:
                    color[:, v_pix, u_pix] = bbox_color

    color += bg * (1 - alpha)[None, :, :]
    rendered_image = (color * 255).to(torch.uint8)
    return rendered_image


def _debug_gaussian(
    with_fall_off: bool = True,
    with_checkerboard: bool = False,
    draw_bbox: bool = True,
    radius_simple: bool = True,
    use_conics: bool = True,
) -> UInt8[Tensor, "3 H W"]:
    H, W = 600, 600

    # create checkerboard background
    bg_color = torch.tensor([50, 50, 50], dtype=torch.float32) / 255.0
    bg: Float32[Tensor, "3 H W"] = bg_color[:, None, None].expand(3, H, W).clone()

    bbox_color = torch.tensor([0, 0, 200], dtype=torch.float32) / 255.0

    if with_checkerboard:
        bg_color_light = torch.tensor([200, 200, 200], dtype=torch.float32) / 255.0
        for u in range(H):
            for v in range(W):
                if ((u // 100) + (v // 100)) % 2 == 0:
                    bg[:, u, v] = bg_color_light

    gaussians = {
        # "problematic": {
        #     "center": (434.6239, 248.2738),
        #     "sigma": 3.0,
        #     "color": torch.tensor([1.0, 0.0, 0.0]),
        #     "opacity": 0.99,
        #     "covariance2d": torch.tensor([[32.2982, 1.9657], [1.9657, 0.1196]], dtype=torch.float64),
        # },
        "gaussian1": {
            "center": (300.0, 300.0),
            "sigma": 3.33,
            "color": torch.tensor([1.0, 0.0, 0.0]),
            "opacity": 0.99,
            "covariance2d": torch.tensor([[1830.0, 800.0], [800.0, 1430.0]]),
        },
        # "gaussian2": {
        #     "center": (200.0, 200.0),
        #     "sigma": 3.0,
        #     "color": torch.tensor([0.0, 1.0, 1.0]),
        #     "opacity": 0.99,
        #     "covariance2d": torch.tensor([[1830.0, -800.0], [-800.0, 1430.0]]),
        # },
        # "gaussian3": {
        #     "center": (400.0, 200.0),
        #     "sigma": 3.0,
        #     "color": torch.tensor([1.0, 1.0, 0.0]),
        #     "opacity": 0.99,
        #     "covariance2d": torch.tensor([[1830.0, 800.0], [800.0, 1430.0]]),
        # },
        # "gaussian4": {
        #     "center": (200.0, 400.0),
        #     "sigma": 3.0,
        #     "color": torch.tensor([1.0, 0.0, 1.0]),
        #     "opacity": 0.99,
        #     "covariance2d": torch.tensor([[1830.0, 800.0], [800.0, 1430.0]]),
        # },
        # "gaussian5": {
        #     "center": (400.0, 400.0),
        #     "sigma": 3.0,
        #     "color": torch.tensor([0.0, 0.0, 1.0]),
        #     "opacity": 0.99,
        #     "covariance2d": torch.tensor([[1830.0, -800.0], [-800.0, 1430.0]]),
        # },
        # "gaussian6": {
        #     "center": (300.0, 300.0),
        #     "sigma": 3.0,
        #     "color": torch.tensor([0.0, 1.0, 0.0]),
        #     "opacity": 0.7,
        #     "covariance2d": torch.tensor([[1830.0, -800.0], [-800.0, 430.0]]),
        # },
    }

    color: Float32[Tensor, "3 H W"] = torch.zeros((3, H, W), dtype=torch.float32)
    alpha: Float32[Tensor, "H W"] = torch.zeros((H, W), dtype=torch.float32)

    for gaussian in gaussians.values():
        sigma = gaussian["sigma"]
        base_color_gaussian = gaussian["color"]
        opacity_gaussian = torch.tensor(gaussian["opacity"])
        v, u = gaussian["center"]
        covariance2d: Tensor = gaussian["covariance2d"]

        if use_conics:
            # To avoid sub-pixel gaussians with non-invertible covariance matrices, add a small value to the diagonal.
            # 0.3 as in gsplat, because sqrt(0.3)[variance] ~= 0.5477 pixels [sigma], which is about half a pixel.
            covariance2d = covariance2d + 0.3 * torch.eye(2)
            a = covariance2d[0, 0]
            b = (covariance2d[0, 1] + covariance2d[1, 0]) / 2
            c = covariance2d[1, 1]

            det = a * c - b * b
            if det <= 0:
                print(f"Warning: non-invertible covariance matrix detected: {covariance2d}, det: {det:.6f}")
                continue
            det = det.clamp(min=1e-10)
            conics = torch.stack(
                [
                    c / det,
                    -b / det,
                    a / det,
                ]
            )
        else:
            covariance2d_inv = torch.linalg.inv(covariance2d)

        if radius_simple:
            # Simple radius based on axis-aligned bounding box of covariance ellipse. Its over estimate, but faster
            # than computing eigenvalues, and also non-square box which means fewer pixels to check.
            radius_u = sigma * math.sqrt(covariance2d[0, 0])
            radius_v = sigma * math.sqrt(covariance2d[1, 1])

            u_min = math.floor(u - radius_u)
            u_max = math.ceil(u + radius_u)
            v_min = math.floor(v - radius_v)
            v_max = math.ceil(v + radius_v)
        else:
            eigenvalues, _ = torch.linalg.eigh(covariance2d)

            radius = sigma * torch.sqrt(torch.max(eigenvalues))

            u_min = math.floor(u - radius)
            u_max = math.ceil(u + radius)
            v_min = math.floor(v - radius)
            v_max = math.ceil(v + radius)

        # iterate over a square region around the projected mean
        for u_pix in range(u_min, u_max):
            for v_pix in range(v_min, v_max):
                # Only write pixel if it's within image bounds
                if 0 <= u_pix < W and 0 <= v_pix < H:
                    du = u_pix + 0.5 - u
                    dv = v_pix + 0.5 - v

                    if use_conics:
                        dist_sq = conics[0] * du * du + 2 * conics[1] * du * dv + conics[2] * dv * dv
                    else:
                        offset_from_mean = torch.tensor([du, dv], dtype=torch.float32)

                        # Mahalanobis distance, no transpose needed since offset_from_mean is a 1D tensor, not column vector
                        dist_sq = offset_from_mean @ covariance2d_inv @ offset_from_mean

                    # Only draw pixel if it's within the circle
                    if dist_sq <= sigma * sigma:  # squared distance, no need to compute sqrt.
                        if with_fall_off:
                            weight = math.exp(-0.5 * dist_sq)
                            alpha_gaussian = opacity_gaussian * weight
                        else:
                            alpha_gaussian = opacity_gaussian

                        color_gaussian = base_color_gaussian * alpha_gaussian

                        # read current alpha and color
                        alpha_pixel = alpha[v_pix, u_pix]
                        color_pixel = color[:, v_pix, u_pix]

                        # compute new alpha and color
                        alpha[v_pix, u_pix] = alpha_gaussian + alpha_pixel * (1 - alpha_gaussian)
                        color[:, v_pix, u_pix] = color_gaussian + color_pixel * (1 - alpha_gaussian)
                    elif draw_bbox:
                        color[:, v_pix, u_pix] = bbox_color

    color += bg * (1 - alpha)[None, :, :]
    rendered_image = (color * 255).to(torch.uint8)
    return rendered_image
