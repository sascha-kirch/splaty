from pathlib import Path

import hydra
from omegaconf import DictConfig
from rich.console import Console
from torchvision.io import ImageReadMode
from torchvision.io import decode_image

from splaty.camera import Camera
from splaty.gaussian import GaussianSet
from splaty.hydra_configs.splaty_cfg import SplatyConfig
from splaty.hydra_configs.splaty_cfg import Stage
from splaty.rasterizer import render_circles_alpha_blending
from splaty.rasterizer import render_circles_fixed_color
from splaty.rasterizer import render_means
from splaty.rasterizer import render_splats
from splaty.rasterizer import render_splats_fov_cull_transmittance
from splaty.rasterizer import render_splats_tiled
from splaty.visualize import save_image


@hydra.main(version_base="1.3", config_path="./config", config_name="bonsai")
def main(hydra_cfg: DictConfig) -> None:
    console = Console()
    console.rule("Splaty")

    cfg = SplatyConfig.from_omegaconf(hydra_cfg)

    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    console.log(f"Output directory: {output_dir}")

    camera = Camera.from_cfg(cfg.camera)
    console.log(f"Using camera: {camera}")

    console.log("Loading data...")
    gs = GaussianSet.from_gsplat_ckpt(ckpt_path=cfg.dataset.path_gsplat_ckpt, device="cpu")
    console.log(f"Loaded {len(gs)} Gaussians.")
    gt_image = decode_image(cfg.dataset.path_gt_image, mode=ImageReadMode.RGB)

    console.log("Rendering...")
    match cfg.stage:
        case Stage.MEANS:
            rendered_image = render_means(
                gs=gs,
                camera=camera,
                color_mode="palette",
                save_frames=cfg.save_gif,
                output_dir=output_dir,
            )
        case Stage.CIRCLES:
            rendered_image = render_circles_fixed_color(
                gs=gs,
                radius_px=10.0,
                camera=camera,
                color_mode="palette",
                save_frames=cfg.save_gif,
                output_dir=output_dir,
            )
        case Stage.CIRCLES_SH:
            rendered_image = render_circles_fixed_color(
                gs=gs,
                radius_px=10.0,
                camera=camera,
                color_mode="sh",
                save_frames=cfg.save_gif,
                output_dir=output_dir,
            )
        case Stage.CIRCLES_ALPHA_BLENDING:
            rendered_image = render_circles_alpha_blending(
                gs=gs,
                camera=camera,
                radius_px=10.0,
                save_frames=cfg.save_gif,
                output_dir=output_dir,
            )
        case Stage.SPLATS_ALPHA_BLENDING:
            rendered_image = render_splats(
                gs=gs,
                camera=camera,
                sigma_multiplier=3.33,
                save_frames=cfg.save_gif,
                output_dir=output_dir,
            )
        case Stage.SPLATS_TRANSMITTANCE:
            rendered_image = render_splats_fov_cull_transmittance(
                gs=gs,
                camera=camera,
                sigma_multiplier=3.33,
                save_frames=cfg.save_gif,
                output_dir=output_dir,
            )
        case Stage.SPLATS_TILED:
            rendered_image = render_splats_tiled(
                gs=gs,
                camera=camera,
                sigma_multiplier=3.33,
                tile_size=32,
                save_frames=cfg.save_gif,
                output_dir=output_dir,
            )

    console.log("Write visualizations...")
    save_image(
        rendered_image=rendered_image,
        save_path=output_dir / "visualization",
    )


if __name__ == "__main__":
    main()
