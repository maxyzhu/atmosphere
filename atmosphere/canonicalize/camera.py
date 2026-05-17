"""
Camera abstractions for M2 (depth rendering) and M3 (pose estimation).

A Camera represents a single rigid pose plus pinhole intrinsics in the
local ENU frame. M2 consumes Camera to render synthetic depth from
OSM-derived geometry; M3 produces Camera from Mapillary API metadata.

Convention
----------
- World frame: local ENU (East-North-Up), meters, z=0 at the ground
  plane (Phase 0 assumption; Phase 1 will use DEM).
- Camera frame: standard CV convention — x right, y down, z forward.
  This matches OpenCV, COLMAP, OpenSfM, and (by inspection of
  load_prior_camera) WorldMirror 2.0's prior_cam JSON schema.
- Extrinsics: c2w (camera-to-world). Translation is the camera position
  in world coords; rotation maps a camera-frame ray to its world dir.
- Intrinsics: pinhole K with no skew. Distortion (k1, k2) is carried
  alongside but Phase 0 renders ignore it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


CameraType = Literal["perspective", "fisheye", "spherical"]


@dataclass(frozen=True)
class Camera:
    """
    A pinhole camera with pose in the local ENU frame.

    Attributes:
        position_enu: (e, n, z) camera center in meters, local ENU.
        rotation: (3, 3) c2w rotation matrix. R[:, 2] is the world
            direction the camera is looking along (camera +Z axis).
        fx, fy: Focal lengths in pixels.
        cx, cy: Principal point in pixels.
        width, height: Image resolution in pixels.
        distortion: (k1, k2) radial distortion. Ignored by Phase 0
            rendering; recorded for traceability and Phase 1 use.
        camera_type: Projection model. Phase 0 only handles
            "perspective"; fisheye and spherical raise on render.
    """

    position_enu: tuple[float, float, float]
    rotation: np.ndarray
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    distortion: tuple[float, float] = (0.0, 0.0)
    camera_type: CameraType = "perspective"

    def __post_init__(self) -> None:
        if self.rotation.shape != (3, 3):
            raise ValueError(
                f"rotation must be 3x3, got {self.rotation.shape}"
            )
        if self.rotation.dtype != np.float64:
            object.__setattr__(
                self, "rotation", self.rotation.astype(np.float64)
            )
        det = np.linalg.det(self.rotation)
        if not np.isclose(det, 1.0, atol=1e-3):
            raise ValueError(
                f"rotation determinant must be 1, got {det:.6f}"
            )
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                f"resolution must be positive, got {self.width}x{self.height}"
            )
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError(f"focal must be positive, got fx={self.fx}")

    @property
    def intrinsic_matrix(self) -> np.ndarray:
        """3x3 K matrix, the standard pinhole intrinsic."""
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    @property
    def c2w(self) -> np.ndarray:
        """4x4 camera-to-world matrix (homogeneous)."""
        m = np.eye(4, dtype=np.float64)
        m[:3, :3] = self.rotation
        m[:3, 3] = np.asarray(self.position_enu, dtype=np.float64)
        return m

    @property
    def w2c(self) -> np.ndarray:
        """4x4 world-to-camera matrix. Inverse of c2w."""
        rt = self.rotation.T
        m = np.eye(4, dtype=np.float64)
        m[:3, :3] = rt
        m[:3, 3] = -rt @ np.asarray(self.position_enu, dtype=np.float64)
        return m

    @property
    def forward_world(self) -> np.ndarray:
        """Unit vector in world coords pointing along camera +Z."""
        return self.rotation @ np.array([0.0, 0.0, 1.0])

    @classmethod
    def from_focal_ratio(
        cls,
        position_enu: tuple[float, float, float],
        rotation: np.ndarray,
        focal_ratio: float,
        width: int,
        height: int,
        distortion: tuple[float, float] = (0.0, 0.0),
        camera_type: CameraType = "perspective",
    ) -> "Camera":
        """
        Build a Camera from Mapillary's `camera_parameters[0]`.

        Mapillary/OpenSfM store focal as a ratio to image's longest edge:
            focal_pixels = focal_ratio * max(width, height)
        Principal point is assumed at image center (OpenSfM default).
        """
        focal_pixels = focal_ratio * max(width, height)
        return cls(
            position_enu=position_enu,
            rotation=rotation,
            fx=focal_pixels,
            fy=focal_pixels,
            cx=width / 2.0,
            cy=height / 2.0,
            width=width,
            height=height,
            distortion=distortion,
            camera_type=camera_type,
        )

    @classmethod
    def from_heading(
        cls,
        position_enu: tuple[float, float, float],
        compass_angle_deg: float,
        focal_ratio: float,
        width: int,
        height: int,
        pitch_deg: float = 0.0,
        roll_deg: float = 0.0,
    ) -> "Camera":
        """
        Build a Camera from compass heading (Mapillary convention).

        Compass: 0=north, 90=east, 180=south, 270=west. The resulting
        c2w puts:
            - camera +Z (forward) along heading direction in ENU
            - camera +X (right) rotates with heading
            - camera +Y (down) along -Z_enu modulo pitch/roll

        Use for Phase 0 baseline M3 when only compass is available.
        Phase 1 M3 will use `computed_rotation` via Rotation.from_rotvec.
        """
        # Base orientation when heading=0, pitch=0, roll=0:
        #   X_cam = +east  (X_enu)
        #   Y_cam = -up    (-Z_enu)
        #   Z_cam = +north (Y_enu)
        # Columns of this matrix are the camera axes expressed in ENU,
        # which is the standard form of a c2w rotation.
        base = np.array(
            [
                # X_cam_enu, Y_cam_enu, Z_cam_enu
                [1.0,  0.0,  0.0],
                [0.0,  0.0,  1.0],
                [0.0, -1.0,  0.0],
            ],
            dtype=np.float64,
        )

        # Heading: rotate around world up (+Z_enu). Compass is CW from
        # north viewed from above; in right-hand ENU that's a negative
        # yaw.
        yaw = -np.radians(compass_angle_deg)
        cy_, sy_ = np.cos(yaw), np.sin(yaw)
        r_yaw = np.array(
            [
                [cy_, -sy_, 0.0],
                [sy_,  cy_, 0.0],
                [0.0,  0.0, 1.0],
            ],
            dtype=np.float64,
        )

        pitch = np.radians(pitch_deg)
        roll = np.radians(roll_deg)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)
        r_pitch = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0,  cp, -sp],
                [0.0,  sp,  cp],
            ],
            dtype=np.float64,
        )
        r_roll = np.array(
            [
                [ cr, 0.0,  sr],
                [0.0, 1.0, 0.0],
                [-sr, 0.0,  cr],
            ],
            dtype=np.float64,
        )

        rotation = r_yaw @ base @ r_pitch @ r_roll

        return cls.from_focal_ratio(
            position_enu=position_enu,
            rotation=rotation,
            focal_ratio=focal_ratio,
            width=width,
            height=height,
        )