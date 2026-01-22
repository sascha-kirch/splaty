from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from torchvision.io import write_png

from splaty.playground.rasterizer import _debug_3_circles_alpha_blending
from splaty.playground.rasterizer import _debug_circle
from splaty.playground.rasterizer import _debug_gaussian
from splaty.playground.rasterizer import _opacity_falloff_circles
from splaty.playground.rasterizer import _opacity_falloff_gaussians


if TYPE_CHECKING:
    from pathlib import Path


def main() -> None:
    save_path = Path("outputs/debug_images")
    save_path.mkdir(parents=True, exist_ok=True)

    console = Console()
    console.rule("Splaty-Debug Images")

    console.log(f"Output directory: {save_path}")

    _opacity_falloff_circles(
        base_opacity=1.0,
        radius=50.0,
        falloff_factor=0.5,
    )

    _opacity_falloff_gaussians(
        base_opacity=0.7,
        k=1.0,
    )

    console.log("Rendering 3 circles with alpha blending...")
    rendered_image = _debug_3_circles_alpha_blending(with_checkerboard=True, with_fall_off=False, base_opacity=1.0)
    write_png(rendered_image, save_path / "3_circles_opaque.png")
    rendered_image = _debug_3_circles_alpha_blending(with_checkerboard=True, with_fall_off=False, base_opacity=0.7)
    write_png(rendered_image, save_path / "3_circles_alpha_blending.png")
    rendered_image = _debug_3_circles_alpha_blending(with_checkerboard=True, with_fall_off=True, base_opacity=1.0)
    write_png(rendered_image, save_path / "3_circles_alpha_blending_falloff.png")

    console.log("Rendering single circle...")
    rendered_image = _debug_circle()
    write_png(rendered_image, save_path / "circle.png")

    console.log("Rendering gaussians...")
    rendered_image = _debug_gaussian()
    write_png(rendered_image, save_path / "gaussians.png")


if __name__ == "__main__":
    main()
