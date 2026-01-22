from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image
from torchvision.io import write_png


if TYPE_CHECKING:
    from jaxtyping import UInt8
    from numpy import ndarray
    from torch import Tensor


def save_image(
    rendered_image: UInt8[Tensor, "3 H W"],
    save_path: Path,
) -> None:
    """Save rendered image.

    Args:
        rendered_image: The rendered image as a tensor of shape (3, H, W).
        save_path: Path to save the images.
    """
    save_path.mkdir(parents=True, exist_ok=True)
    write_png(rendered_image, save_path / "rendered.png")


def save_gif(
    frames: list[UInt8[ndarray, "H W 3"]],
    output_dir: Path | None = None,
) -> None:
    """Save frames as a GIF.

    Args:
        frames: List of image frames as numpy arrays of shape (H, W, 3).
        output_dir: Path to save the GIF.
    """
    file_path: Path = output_dir / "rendering.gif" if output_dir else Path("rendering.gif")
    file_path.parent.mkdir(parents=True, exist_ok=True)

    pil_frames = [Image.fromarray(frame) for frame in frames]

    print(f"Saving means rendering GIF to '{file_path}' with {len(pil_frames)} frames.")

    # https://pillow.readthedocs.io/en/latest/handbook/image-file-formats.html#gif-saving
    pil_frames[0].save(
        file_path,
        save_all=True,
        append_images=pil_frames[1:] + 10 * pil_frames[-1:],
        duration=100,  # 100ms per frame, most viewers limit to 10fps, even if lower duration is set...
        disposal=2,
        optimize=False,
        loop=0,
    )
