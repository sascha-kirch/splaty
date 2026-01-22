from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from jaxtyping import Float32
    from omegaconf import DictConfig
    from torch import Tensor


@dataclass
class CameraConfig:
    """Configuration for the pinhole camera."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    scale: int
    near_plane: float
    far_plane: float
    q_world_to_cam: Float32[Tensor, "4"]
    t_world_to_cam: Float32[Tensor, "3"]
    normalization_transform: Float32[Tensor, "4 4"]

    @staticmethod
    def from_omegaconf(cfg: DictConfig) -> CameraConfig:
        """Create CameraConfig from OmegaConf DictConfig."""
        return CameraConfig(
            fx=cfg.fx,
            fy=cfg.fy,
            cx=cfg.cx,
            cy=cfg.cy,
            width=cfg.width,
            height=cfg.height,
            scale=cfg.scale,
            near_plane=cfg.near_plane,
            far_plane=cfg.far_plane,
            q_world_to_cam=torch.tensor(cfg.q_world_to_cam),
            t_world_to_cam=torch.tensor(cfg.t_world_to_cam),
            normalization_transform=torch.tensor(cfg.normalization_transform),
        )
