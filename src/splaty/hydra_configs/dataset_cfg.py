from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from omegaconf import DictConfig


@dataclass
class DatasetConfig:
    """Configuration for the Dataset."""

    name: str
    path_gsplat_ckpt: Path
    path_gt_image: Path

    @staticmethod
    def from_omegaconf(cfg: DictConfig) -> DatasetConfig:
        """Create DatasetConfig from OmegaConf DictConfig."""
        return DatasetConfig(
            name=cfg.name,
            path_gsplat_ckpt=Path(cfg.path_gsplat_ckpt).expanduser(),
            path_gt_image=Path(cfg.path_gt_image).expanduser(),
        )
