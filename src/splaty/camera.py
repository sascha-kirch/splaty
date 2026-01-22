from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from splaty.spatial_math import quaternion_to_rotation_matrix
from splaty.spatial_math import renormalize_extrinsics


if TYPE_CHECKING:
    from jaxtyping import Float32
    from torch import Tensor

    from splaty.hydra_configs.camera_cfg import CameraConfig


class Camera:
    """Pinhole camera model with geometric transformations.

    Intentionally no clipping, rounding or bound checking due to separation of concerns. All handled in rasterization
    stage.
    """

    def __init__(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        width: int,
        height: int,
        q_world_to_cam: Float32[Tensor, "4"],
        t_world_to_cam: Float32[Tensor, "3"],
        scale: int = 1,
        normalization_transform: Float32[Tensor, "4 4"] | None = None,
        near_plane: float = 0.0,
        far_plane: float = float("inf"),
    ) -> None:
        """Initializes the pinhole camera.

        Args:
            fx (float): Focal length in x direction.
            fy (float): Focal length in y direction.
            cx (float): Principal point x coordinate.
            cy (float): Principal point y coordinate.
            width (int): Image width.
            height (int): Image height.
            q_world_to_cam (Float32[Tensor, "4"]): Quaternion representing rotation from world to camera
                coordinates. Convention: (qw, qx, qy, qz) and normalized.
            t_world_to_cam (Float32[Tensor, "3"]): Translation vector from world to camera coordinates.
            scale (int, optional): Scale factor for image resolution. Defaults to 1.
            normalization_transform (Float32[Tensor, "4 4"] | None, optional): Normalization transform to
                renormalize the extrinsics. If None, no renormalization is applied. Defaults to None.
            near_plane (float, optional): Near plane distance. Defaults to 0.0.
            far_plane (float, optional): Far plane distance. Defaults to float("inf").
        """
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height
        self.q_world_to_cam = q_world_to_cam
        self.t_world_to_cam = t_world_to_cam
        self.scale = scale
        self.near_plane = near_plane
        self.far_plane = far_plane

        if scale > 1:
            self.width = math.ceil(self.width / scale)
            self.height = math.ceil(self.height / scale)
            scale_x = width / self.width
            scale_y = height / self.height
            self.fx /= scale_x
            self.fy /= scale_y
            self.cx /= scale_x
            self.cy /= scale_y

        self.fov_x, self.fov_y = self.compute_fov(self.fx, self.fy, self.width, self.height)

        self.R_world_to_cam = quaternion_to_rotation_matrix(q_world_to_cam)

        if normalization_transform is not None:
            self.R_world_to_cam, self.t_world_to_cam = renormalize_extrinsics(
                self.R_world_to_cam,
                self.t_world_to_cam,
                normalization_transform,
            )

        # Transpose of rotation matrix is its inverse, because rotation matrices are orthogonal
        self.R_cam_to_world = self.R_world_to_cam.T

        # precompute camera center in world coordinates
        self.camera_center_world = -self.R_world_to_cam.T @ self.t_world_to_cam

    @classmethod
    def from_cfg(cls, cfg_camera: CameraConfig) -> Camera:
        """Creates a Camera instance from a CameraConfig.

        Args:
            cfg_camera (CameraConfig): Camera configuration.

        Returns:
            Self: Camera instance.
        """
        return cls(
            fx=cfg_camera.fx,
            fy=cfg_camera.fy,
            cx=cfg_camera.cx,
            cy=cfg_camera.cy,
            width=cfg_camera.width,
            height=cfg_camera.height,
            scale=cfg_camera.scale,
            q_world_to_cam=cfg_camera.q_world_to_cam,
            t_world_to_cam=cfg_camera.t_world_to_cam,
            normalization_transform=cfg_camera.normalization_transform,
            near_plane=cfg_camera.near_plane,
            far_plane=float(cfg_camera.far_plane),
        )

    def transform_points_world_to_cam(self, points_world: Float32[Tensor, "N 3"]) -> Float32[Tensor, "N 3"]:
        """Transforms points from world coordinates to camera coordinates.

        Note:
            Explicitly no clipping to image bounds and removal of points behind the camera.
            This is handled in the rasterization stage.

        Args:
            points_world (Float32[Tensor, "N 3"]): Points in world coordinates.

        Returns:
            Float32[Tensor, "N 3"]: Points in camera coordinates.
        """
        # for the blogpost explain why vector @ R.T and not R @ vector => because we have not a single vector[3] but N vectors stored row-wise[N,3]!!!
        return (points_world - self.camera_center_world) @ self.R_world_to_cam.T

    def rotate_covariances_world_to_cam(self, covariances_world: Float32[Tensor, "N 3 3"]) -> Float32[Tensor, "N 3 3"]:
        """Rotates covariances of 3d gaussians from world coordinates to camera coordinates.

        Args:
            covariances_world (Float32[Tensor, "N 3 3"]): Covariance matrices in world coordinates.

        Returns:
            Float32[Tensor, "N 3 3"]: Covariance matrices in camera coordinates.
        """
        return self.R_world_to_cam @ covariances_world @ self.R_world_to_cam.T

    def project_points_cam_to_image_plane(self, points_cam: Float32[Tensor, "N 3"]) -> Float32[Tensor, "N 2"]:
        """Projects points from camera coordinates to the image plane.

        Note:
            Explicitly no clipping to image bounds and rounding to pixel coordinates. The reason is that gaussians might
            have their center outside the image, but still contribute to pixels inside the image. Similarly, the mean
            of the projected gaussian might be between pixel coordinates, but still contribute to multiple pixels.

        Args:
            points_cam (Float32[Tensor, "N 3"]): Points in camera coordinates.

        Returns:
            Float32[Tensor, "N 2"]: Points in image plane coordinates (pixel coordinates).
        """
        x, y, z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
        u = self.fx * (x / z) + self.cx
        v = self.fy * (y / z) + self.cy
        return torch.stack([u, v], dim=-1)

    def view_direction_cam_to_mean_world(self, points_world: Float32[Tensor, "N 3"]) -> Float32[Tensor, "N 3"]:
        """Computes the normalized view direction in world coordinates from the camera to each point.

        Args:
            points_world (Float32[Tensor, "N 3"]): Points in world coordinates.

        Returns:
            Float32[Tensor, "N 3"]: Normalized view directions in world coordinates.
        """
        directions_world = points_world - self.camera_center_world
        return directions_world / torch.linalg.norm(directions_world, dim=-1, keepdim=True)

    @staticmethod
    def compute_fov(fx: float, fy: float, width: int, height: int) -> tuple[float, float]:
        """Computes the horizontal and vertical field of view (FOV) in radians.

        Args:
            fx (float): Focal length in x direction.
            fy (float): Focal length in y direction.
            width (int): Image width.
            height (int): Image height.

        Returns:
            tuple[float, float]: Horizontal and vertical FOV in radians.
        """
        fov_x = 2 * math.atan(width / (2 * fx))
        fov_y = 2 * math.atan(height / (2 * fy))
        return fov_x, fov_y

    def __str__(self) -> str:
        return (
            f"Camera(fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}, width={self.width}, height={self.height})"
        )
