"""
M3: Cross-modal alignment from Mapillary metadata to Camera dataclass.

This module is a thin adapter: it reads the fields Mapillary's Graph
API provides on each image (computed_rotation, focal_ratio, width,
height) and the camera_z_m we computed from the DEM, and returns a
Camera in our local ENU frame ready for M2 depth rendering and M4
SCP bundle assembly.

There is no novel algorithm here. The complexity is in getting the
coordinate-frame conversions right, which was the job of the
literature review and source-code probes recorded in changelog Day 5
Part 1. By the time this code runs, we already know:

- Mapillary's `computed_rotation` is an axis-angle 3-vector in the
  OpenSfM topocentric frame, which equals ENU when GPS is present
  (always the case for Mapillary uploads).
- That rotation is world-to-camera (w2c) per OpenSfM's `Pose` class.
  Our Camera dataclass stores camera-to-world (c2w), so M3 transposes
  once.
- Mapillary `camera_parameters[0]` is the focal ratio in OpenSfM's
  `focal_pixels = ratio * max(W, H)` convention. Camera.from_focal_ratio
  already encapsulates that.

Fail-fast policy
----------------
WorldMirror 2.0's prior bundle is all-or-nothing: any image missing
a prior silently drops the entire batch's prior. M3 therefore returns
None for any image that can't produce a complete Camera, and the
caller filters those out before bundling. The dropped count is logged
so the user knows how many Mapillary candidates survived.

Sanity check (test-only)
------------------------
`verify_yaw_against_compass` cross-checks the yaw extracted from the
M3 Camera against the SfM-corrected `computed_compass_angle` from the
same Mapillary record. They should agree within a few degrees if our
frame interpretation is right. Used in tests and during interactive
debugging; not invoked in the production path.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation

from atmosphere.canonicalize.camera import Camera

if TYPE_CHECKING:
    from atmosphere.retrieval.mapillary import MapillaryImage

logger = logging.getLogger(__name__)


def mapillary_to_camera(img: "MapillaryImage") -> Camera | None:
    """
    Convert a fully-populated Mapillary image into an ENU Camera.

    Required fields on `img`:
        - computed_rotation (3-tuple, radians, w2c in OpenSfM/ENU frame)
        - focal_ratio (OpenSfM convention)
        - width, height (image native resolution in pixels)
        - camera_z_m (DEM elevation + camera-height prior)
        - is_pano must be False

    Returns None for any image missing any of those, or whose
    rotation matrix fails the Camera dataclass's determinant check.

    The returned Camera is in the same ENU frame as `img.position_enu`
    — the caller is responsible for keeping all frames consistent.
    """
    if img.is_pano:
        return None
    if not img.has_camera_metadata:
        return None
    if img.camera_z_m is None:
        return None

    # SfM/OpenSfM w2c rotation, axis-angle (rad) -> 3x3, then transpose
    # to get c2w as expected by Camera. See changelog Day 5 Part 1 §2
    # for the OpenSfM convention reference.
    rotvec = np.asarray(img.computed_rotation, dtype=np.float64)
    R_w2c = Rotation.from_rotvec(rotvec).as_matrix()
    R_c2w = R_w2c.T

    east, north = img.position_enu
    pos = (float(east), float(north), float(img.camera_z_m))

    try:
        return Camera.from_focal_ratio(
            position_enu=pos,
            rotation=R_c2w,
            focal_ratio=img.focal_ratio,
            width=img.width,
            height=img.height,
        )
    except ValueError as exc:
        # Camera.__post_init__ raises if the rotation determinant is
        # too far from 1 (degenerate or malformed rotation). This is
        # rare but real: at least one Mapillary image in our DLR
        # sample is expected to fail this for OpenSfM-internal reasons.
        logger.debug(
            "Dropping Mapillary image %s: Camera rejected (%s)",
            img.mapillary_id, exc,
        )
        return None


def batch_mapillary_to_cameras(
    images: list["MapillaryImage"],
) -> tuple[list[Camera], list["MapillaryImage"]]:
    """
    Map a list of MapillaryImages through M3 and return (cameras,
    surviving_images), preserving order and dropping failures.

    The two lists are positionally aligned: cameras[i] is built from
    surviving_images[i]. M4 needs both sides paired (the image stem
    indexes both the thumbnail and the camera record in the prior
    bundle).
    """
    cameras: list[Camera] = []
    surviving: list["MapillaryImage"] = []
    for img in images:
        cam = mapillary_to_camera(img)
        if cam is None:
            continue
        cameras.append(cam)
        surviving.append(img)

    dropped = len(images) - len(cameras)
    if dropped:
        logger.info(
            "M3: produced %d cameras, dropped %d / %d images "
            "(missing metadata, missing DEM, pano, or degenerate rotation)",
            len(cameras), dropped, len(images),
        )
    else:
        logger.info("M3: produced %d cameras from %d images", len(cameras), len(images))

    return cameras, surviving


def camera_yaw_deg(camera: Camera) -> float:
    """
    Extract heading (compass yaw, degrees CW from north) from a Camera.

    The camera's forward direction in ENU is `R_c2w @ [0, 0, 1]`
    (camera +Z mapped to world). Project that onto the horizontal
    plane and measure the bearing from ENU +Y (north), clockwise.
    """
    fwd = camera.forward_world  # 3-vector in ENU
    east_comp = float(fwd[0])
    north_comp = float(fwd[1])
    # atan2 with (east, north) gives angle CW from north,
    # which is exactly the compass convention.
    yaw = math.degrees(math.atan2(east_comp, north_comp))
    if yaw < 0:
        yaw += 360.0
    return yaw


def verify_yaw_against_compass(
    camera: Camera, compass_angle_deg: float,
) -> float:
    """
    Sanity-check helper: returns the absolute circular difference, in
    degrees, between the camera's yaw and the supplied compass angle.

    A small value (< 5° or so) means our frame interpretation
    (OpenSfM topocentric = ENU, computed_rotation is w2c) is consistent
    with Mapillary's separately-published SfM-corrected compass. A
    large value (≈ 90°, 180°) means an axis is flipped somewhere and
    M3 needs investigation.

    Intended for tests and ad-hoc debugging. Not called in the
    production M3 path.
    """
    yaw = camera_yaw_deg(camera)
    diff = abs(yaw - compass_angle_deg) % 360.0
    if diff > 180.0:
        diff = 360.0 - diff
    return diff
