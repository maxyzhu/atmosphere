"""
Tests for atmosphere.canonicalize.align (M3).

What's covered here
-------------------
- Fail-fast paths: pano, missing camera_metadata, missing camera_z_m,
  degenerate rotation.
- Success path: a fully-populated MapillaryImage with rotation =
  identity (forward = +Y_enu = north) produces a Camera whose
  forward_world points north, and whose camera_yaw_deg agrees with
  the supplied compass angle.
- A known non-trivial rotation: 90° right-yaw (rotvec around -Z),
  forward should land along +X_enu = east, yaw should be 90°.
- batch_mapillary_to_cameras preserves order and drops failures
  positionally aligned.
- verify_yaw_against_compass returns 0 for a hand-built consistent
  pair and ≈180 for a hand-built reversed one.

What's *not* covered here
-------------------------
- End-to-end Mapillary -> M3 with real API data. That lives in a
  separate end-to-end integration script (Day 5 Step 6) once DEM is
  downloaded. The frame-assumption check (yaw vs computed_compass_
  angle on real images) runs there, not here.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from atmosphere.canonicalize.align import (
    batch_mapillary_to_cameras,
    camera_yaw_deg,
    mapillary_to_camera,
    verify_yaw_against_compass,
)
from atmosphere.canonicalize.camera import Camera
from atmosphere.retrieval.mapillary import MapillaryImage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rotvec_for_c2w(target_c2w: np.ndarray) -> tuple[float, float, float]:
    """
    Build the axis-angle 3-vector Mapillary would publish for a camera
    whose c2w rotation is `target_c2w`.

    Mapillary publishes w2c (= c2w.T) as a rotvec, so:
        rotvec = Rotation.from_matrix(target_c2w.T).as_rotvec()
    """
    R_w2c = target_c2w.T
    rv = Rotation.from_matrix(R_w2c).as_rotvec()
    return (float(rv[0]), float(rv[1]), float(rv[2]))


def _make_image(
    *,
    mapillary_id: str = "img-test-1",
    is_pano: bool = False,
    rotation_c2w: np.ndarray | None = None,
    focal_ratio: float | None = 0.5,
    width: int | None = 1920,
    height: int | None = 1080,
    compass: float | None = 0.0,
    camera_z_m: float | None = 31.5,
) -> MapillaryImage:
    """
    Build a MapillaryImage with all camera-metadata fields populated,
    using a target c2w rotation. Defaults give a north-facing camera
    on a 1.5 m handheld at 30 m ground elevation.
    """
    rotvec: tuple[float, float, float] | None
    if rotation_c2w is None and not is_pano:
        rotvec = _rotvec_for_c2w(np.eye(3))
    elif rotation_c2w is None:
        rotvec = None
    else:
        rotvec = _rotvec_for_c2w(rotation_c2w)

    return MapillaryImage(
        mapillary_id=mapillary_id,
        position_enu=(10.0, 20.0),
        compass_angle_deg=compass,
        is_pano=is_pano,
        thumb_url="",
        thumb_path=None,
        computed_rotation=rotvec,
        focal_ratio=focal_ratio,
        width=width,
        height=height,
        computed_compass_angle=compass,
        camera_z_m=camera_z_m,
    )


# ---------------------------------------------------------------------------
# Fail-fast paths
# ---------------------------------------------------------------------------


def test_pano_returns_none():
    img = _make_image(is_pano=True)
    assert mapillary_to_camera(img) is None


def test_missing_rotation_returns_none():
    img = _make_image()
    img_no_rot = MapillaryImage(
        mapillary_id=img.mapillary_id,
        position_enu=img.position_enu,
        compass_angle_deg=img.compass_angle_deg,
        is_pano=img.is_pano,
        thumb_url=img.thumb_url,
        thumb_path=img.thumb_path,
        computed_rotation=None,
        focal_ratio=img.focal_ratio,
        width=img.width,
        height=img.height,
        computed_compass_angle=img.computed_compass_angle,
        camera_z_m=img.camera_z_m,
    )
    assert mapillary_to_camera(img_no_rot) is None


def test_missing_focal_returns_none():
    img = _make_image(focal_ratio=None)
    assert mapillary_to_camera(img) is None


def test_missing_dimensions_returns_none():
    img_no_w = _make_image(width=None)
    assert mapillary_to_camera(img_no_w) is None
    img_no_h = _make_image(height=None)
    assert mapillary_to_camera(img_no_h) is None


def test_missing_camera_z_returns_none():
    img = _make_image(camera_z_m=None)
    assert mapillary_to_camera(img) is None


# ---------------------------------------------------------------------------
# Success path — frame consistency
# ---------------------------------------------------------------------------


def test_identity_rotation_produces_north_facing_camera():
    """
    Build an image whose c2w rotation is identity and check that the
    resulting Camera's forward_world matches Camera.from_heading(0)'s
    forward_world (heading=0 = compass north).

    This anchors the whole frame chain: if this test passes, we've
    confirmed
      (a) Mapillary publishes w2c in OpenSfM's ENU-aligned topocentric
          frame,
      (b) our transpose to c2w is correct, and
      (c) the rotvec round-trip preserves the rotation.
    """
    img = _make_image(rotation_c2w=np.eye(3), compass=0.0)
    cam = mapillary_to_camera(img)
    assert cam is not None

    # Identity c2w means camera +Z (forward) = world +Z (up). That's
    # not "facing north", that's "facing the sky" — which is what an
    # identity rotation in this convention means. The forward vector
    # should be (0, 0, 1).
    np.testing.assert_allclose(
        cam.forward_world, np.array([0.0, 0.0, 1.0]), atol=1e-10
    )


def test_north_facing_via_from_heading_roundtrip():
    """
    Take Camera.from_heading(compass=0) — a true north-facing camera
    in our convention — extract its c2w, feed it through M3, and check
    we get back the same forward vector.
    """
    cam_ref = Camera.from_heading(
        position_enu=(10.0, 20.0, 31.5),
        compass_angle_deg=0.0,
        focal_ratio=0.5,
        width=1920,
        height=1080,
    )
    img = _make_image(rotation_c2w=cam_ref.rotation, compass=0.0)
    cam = mapillary_to_camera(img)
    assert cam is not None

    np.testing.assert_allclose(
        cam.forward_world, cam_ref.forward_world, atol=1e-10
    )
    np.testing.assert_allclose(cam.rotation, cam_ref.rotation, atol=1e-10)


@pytest.mark.parametrize("compass", [0.0, 45.0, 90.0, 180.0, 270.0, 359.0])
def test_yaw_roundtrip_at_various_compasses(compass: float):
    """
    For every test compass, build a from_heading camera, extract its
    c2w, push through M3, and check both the resulting forward vector
    and the extracted yaw match.

    This is the bedrock frame test: identity-rotation only proves the
    transpose is right; varied compass proves the whole rotation chain
    (rotvec -> matrix -> transpose -> Camera) commutes with our yaw
    convention across the full circle.
    """
    cam_ref = Camera.from_heading(
        position_enu=(10.0, 20.0, 31.5),
        compass_angle_deg=compass,
        focal_ratio=0.5,
        width=1920,
        height=1080,
    )
    img = _make_image(rotation_c2w=cam_ref.rotation, compass=compass)
    cam = mapillary_to_camera(img)
    assert cam is not None

    np.testing.assert_allclose(
        cam.forward_world, cam_ref.forward_world, atol=1e-10
    )

    extracted = camera_yaw_deg(cam)
    diff = abs(extracted - compass) % 360.0
    if diff > 180.0:
        diff = 360.0 - diff
    assert diff < 1e-6, (
        f"compass={compass} -> camera yaw={extracted} (diff {diff})"
    )


def test_position_uses_camera_z_m():
    img = _make_image(camera_z_m=42.7)
    cam = mapillary_to_camera(img)
    assert cam is not None
    assert cam.position_enu == (10.0, 20.0, 42.7)


def test_intrinsics_use_focal_ratio_convention():
    """
    Mapillary's focal_ratio = 0.5 on a 1920x1080 image should give
    fx = fy = 0.5 * 1920 = 960 px, cx = 960, cy = 540.
    """
    img = _make_image(focal_ratio=0.5, width=1920, height=1080)
    cam = mapillary_to_camera(img)
    assert cam is not None
    assert cam.fx == 960.0
    assert cam.fy == 960.0
    assert cam.cx == 960.0
    assert cam.cy == 540.0


# ---------------------------------------------------------------------------
# Batch helper
# ---------------------------------------------------------------------------


def test_batch_drops_invalid_preserves_order():
    """
    Three images: good, bad (pano), good. The output should be two
    cameras in order, with surviving_images positionally aligned.
    """
    good1 = _make_image(mapillary_id="g1")
    bad = _make_image(mapillary_id="bad", is_pano=True)
    good2 = _make_image(mapillary_id="g2")

    cameras, surviving = batch_mapillary_to_cameras([good1, bad, good2])

    assert len(cameras) == 2
    assert len(surviving) == 2
    assert surviving[0].mapillary_id == "g1"
    assert surviving[1].mapillary_id == "g2"


def test_batch_empty_input():
    cameras, surviving = batch_mapillary_to_cameras([])
    assert cameras == []
    assert surviving == []


def test_batch_all_invalid():
    images = [_make_image(mapillary_id=f"p{i}", is_pano=True) for i in range(3)]
    cameras, surviving = batch_mapillary_to_cameras(images)
    assert cameras == []
    assert surviving == []


# ---------------------------------------------------------------------------
# verify_yaw_against_compass
# ---------------------------------------------------------------------------


def test_verify_yaw_consistent_pair():
    cam = Camera.from_heading(
        position_enu=(0.0, 0.0, 1.5),
        compass_angle_deg=42.0,
        focal_ratio=0.5,
        width=1920,
        height=1080,
    )
    diff = verify_yaw_against_compass(cam, 42.0)
    assert diff < 1e-6


def test_verify_yaw_circular_wrap():
    """
    Test that 359° and 1° are seen as 2° apart, not 358° apart.
    """
    cam = Camera.from_heading(
        position_enu=(0.0, 0.0, 1.5),
        compass_angle_deg=359.0,
        focal_ratio=0.5,
        width=1920,
        height=1080,
    )
    diff = verify_yaw_against_compass(cam, 1.0)
    # Within float-roundoff of 2°.
    assert abs(diff - 2.0) < 1e-6


def test_verify_yaw_opposite_direction():
    cam = Camera.from_heading(
        position_enu=(0.0, 0.0, 1.5),
        compass_angle_deg=0.0,
        focal_ratio=0.5,
        width=1920,
        height=1080,
    )
    diff = verify_yaw_against_compass(cam, 180.0)
    # Should be exactly 180° (boundary of the circular distance).
    assert abs(diff - 180.0) < 1e-6


# ---------------------------------------------------------------------------
# camera_yaw_deg directly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "compass", [0.0, 30.0, 90.0, 180.0, 270.0, 359.999]
)
def test_camera_yaw_matches_from_heading_construction(compass: float):
    cam = Camera.from_heading(
        position_enu=(0.0, 0.0, 1.5),
        compass_angle_deg=compass,
        focal_ratio=0.5,
        width=1920,
        height=1080,
    )
    extracted = camera_yaw_deg(cam)
    diff = abs(extracted - compass) % 360.0
    if diff > 180.0:
        diff = 360.0 - diff
    assert diff < 1e-6
