from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from enum import auto
from typing import TYPE_CHECKING

from splaty.hydra_configs.camera_cfg import CameraConfig
from splaty.hydra_configs.dataset_cfg import DatasetConfig


if TYPE_CHECKING:
    from omegaconf import DictConfig


class Stage(StrEnum):
    """Rendering stages."""

    MEANS = auto()
    CIRCLES = auto()
    CIRCLES_SH = auto()
    CIRCLES_ALPHA_BLENDING = auto()
    SPLATS_ALPHA_BLENDING = auto()
    SPLATS_TRANSMITTANCE = auto()
    SPLATS_TILED = auto()


@dataclass
class SplatyConfig:
    """Configuration for Splaty rendering."""

    dataset: DatasetConfig
    camera: CameraConfig
    stage: Stage

    save_gif: bool

    @staticmethod
    def from_omegaconf(cfg: DictConfig) -> SplatyConfig:
        """Create SplatyConfig from OmegaConf DictConfig."""
        return SplatyConfig(
            dataset=DatasetConfig.from_omegaconf(cfg.dataset),
            camera=CameraConfig.from_omegaconf(cfg.camera),
            stage=Stage(cfg.stage),
            save_gif=cfg.save_gif,
        )
