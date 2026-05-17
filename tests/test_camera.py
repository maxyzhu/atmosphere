"""
Tests for Camera dataclass and constructors.

These tests are pure math — no rendering, no I/O. The point is to lock
down the coordinate conventions so render.py can trust them.
"""

from __future__ import annotations

import numpy as np
import pytest

from atmosphere.canonicalize.camera import Camera


def _r_identity() -> np.ndarray:
    """A valid (3, 3) identity for use in tests where rotation doesn't
    matter — Camera requires a valid rotation matrix to construct."""
    return np.eye(3, dtype=np.float64)


class TestCameraValidation:
    def test_accepts_identity(self):
        cam = Camera(
            position_enu=(0, 0, 0),
            rotation=_r_identity(),
            fx=100, fy=100, cx=50, cy=50,
            width=100, height=100,
        )
        assert cam.fx == 100

    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError, match="must be 3x3"):
            Camera(
                position_enu=(0, 0, 0),
                rotation=np.eye(4),
                fx=100, fy=100, cx=50, cy=50,
                width=100, height=100,
            )

    def test_rejects_non_rotation(self):
        """A matrix with det != 1 (e.g., scaling) must be rejected."""
        bad = np.diag([2.0, 1.0, 1.0])
        with pytest.raises(ValueError, match="determinant"):
            Camera(
                position_enu=(0, 0, 0),
                rotation=bad,
                fx=100, fy=100, cx=50, cy=50,
                width=100, height=100,
            )

    def test_rejects_zero_resolution(self):
        with pytest.raises(ValueError, match="resolution"):
            Camera(
                position_enu=(0, 0, 0), rotation=_r_identity(),
                fx=100, fy=100, cx=50, cy=50,
                width=0, height=100,
            )

    def test_rejects_negative_focal(self):
        with pytest.raises(ValueError, match="focal"):
            Camera(
                position_enu=(0, 0, 0), rotation=_r_identity(),
                fx=-100, fy=100, cx=50, cy=50,
                width=100, height=100,
            )

    def test_coerces_rotation_to_float64(self):
        r32 = np.eye(3, dtype=np.float32)
        cam = Camera(
            position_enu=(0, 0, 0), rotation=r32,
            fx=100, fy=100, cx=50, cy=50,
            width=100, height=100,
        )
        assert cam.rotation.dtype == np.float64


class TestMatrixAccessors:
    def test_intrinsic_matrix_shape_and_values(self):
        cam = Camera(
            position_enu=(0, 0, 0), rotation=_r_identity(),
            fx=500, fy=500, cx=320, cy=240,
            width=640, height=480,
        )
        K = cam.intrinsic_matrix
        assert K.shape == (3, 3)
        assert K[0, 0] == 500 and K[1, 1] == 500
        assert K[0, 2] == 320 and K[1, 2] == 240
        assert K[2, 2] == 1

    def test_c2w_and_w2c_inverse(self):
        # Build a non-identity camera and verify c2w @ w2c == I.
        cam = Camera.from_heading(
            position_enu=(10, 20, 1.5),
            compass_angle_deg=45.0,
            focal_ratio=0.5, width=640, height=480,
        )
        c2w = cam.c2w
        w2c = cam.w2c
        product = c2w @ w2c
        np.testing.assert_allclose(product, np.eye(4), atol=1e-10)

    def test_forward_world_is_unit(self):
        cam = Camera.from_heading(
            position_enu=(0, 0, 0),
            compass_angle_deg=0.0,
            focal_ratio=0.5, width=640, height=480,
        )
        fwd = cam.forward_world
        assert np.isclose(np.linalg.norm(fwd), 1.0)


class TestFromFocalRatio:
    def test_focal_pixels_uses_longest_edge(self):
        """Mapillary convention: focal_pixels = ratio * max(W, H)."""
        cam = Camera.from_focal_ratio(
            position_enu=(0, 0, 0), rotation=_r_identity(),
            focal_ratio=0.5,
            width=1920, height=1080,
        )
        assert cam.fx == 0.5 * 1920
        assert cam.fy == cam.fx

    def test_principal_point_at_center(self):
        cam = Camera.from_focal_ratio(
            position_enu=(0, 0, 0), rotation=_r_identity(),
            focal_ratio=0.5, width=640, height=480,
        )
        assert cam.cx == 320
        assert cam.cy == 240


class TestFromHeading:
    """Verify the compass→c2w conversion lands the camera looking the
    right way in ENU. These are the most critical tests: getting the
    convention wrong here flips every depth map by 180° silently."""

    def test_heading_zero_looks_north(self):
        """Compass 0 means facing north (+Y_enu). The camera forward
        in world frame should be approximately (0, 1, 0)."""
        cam = Camera.from_heading(
            position_enu=(0, 0, 0),
            compass_angle_deg=0.0,
            focal_ratio=0.5, width=640, height=480,
        )
        fwd = cam.forward_world
        np.testing.assert_allclose(fwd, [0, 1, 0], atol=1e-10)

    def test_heading_90_looks_east(self):
        cam = Camera.from_heading(
            position_enu=(0, 0, 0),
            compass_angle_deg=90.0,
            focal_ratio=0.5, width=640, height=480,
        )
        fwd = cam.forward_world
        np.testing.assert_allclose(fwd, [1, 0, 0], atol=1e-10)

    def test_heading_180_looks_south(self):
        cam = Camera.from_heading(
            position_enu=(0, 0, 0),
            compass_angle_deg=180.0,
            focal_ratio=0.5, width=640, height=480,
        )
        fwd = cam.forward_world
        np.testing.assert_allclose(fwd, [0, -1, 0], atol=1e-10)

    def test_heading_270_looks_west(self):
        cam = Camera.from_heading(
            position_enu=(0, 0, 0),
            compass_angle_deg=270.0,
            focal_ratio=0.5, width=640, height=480,
        )
        fwd = cam.forward_world
        np.testing.assert_allclose(fwd, [-1, 0, 0], atol=1e-10)

    def test_position_preserved(self):
        cam = Camera.from_heading(
            position_enu=(123.45, -67.89, 1.5),
            compass_angle_deg=42.0,
            focal_ratio=0.5, width=640, height=480,
        )
        np.testing.assert_allclose(cam.position_enu, (123.45, -67.89, 1.5))