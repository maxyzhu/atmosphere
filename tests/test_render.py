"""
Tests for M2 depth rendering.

Strategy: build synthetic Buildings (small, predictable shapes) at
known positions, render with known cameras, verify depth values match
analytical expectations within mm tolerance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from atmosphere.canonicalize.camera import Camera
from atmosphere.canonicalize.render import (
    buildings_to_mesh,
    render_depth,
    render_depth_batch,
)
from atmosphere.retrieval.buildings import Building, HeightSource
from atmosphere.scp import SKY_DEPTH_M


def _square_building(
    east: float, north: float, size: float = 10.0,
    height: float = 20.0, osm_id: int = 1,
) -> Building:
    """A square building centered at (east, north) with given size + height."""
    half = size / 2
    footprint = np.array([
        [east - half, north - half],
        [east + half, north - half],
        [east + half, north + half],
        [east - half, north + half],
        [east - half, north - half],   # closed
    ], dtype=np.float64)
    return Building(
        footprint_enu=footprint,
        height_m=height,
        height_source=HeightSource.TAG,
        osm_id=osm_id,
        building_type="commercial",
    )


def _north_facing_camera(
    east: float = 0.0, north: float = 0.0, z: float = 1.5,
    width: int = 320, height: int = 240,
) -> Camera:
    """A camera at ground level looking north (+Y_enu)."""
    return Camera.from_heading(
        position_enu=(east, north, z),
        compass_angle_deg=0.0,           # north
        focal_ratio=0.5,                  # fx = 0.5 * 320 = 160
        width=width, height=height,
    )


class TestBuildingsToMesh:
    def test_empty_input_returns_empty_mesh(self):
        mesh = buildings_to_mesh([])
        assert len(mesh.vertices) == 0

    def test_single_box_has_expected_vertex_count(self):
        b = _square_building(0, 0)
        mesh = buildings_to_mesh([b])
        # A box from a square footprint has at least 8 vertices.
        # Trimesh may add duplicates depending on cap triangulation;
        # we just check it's non-empty and box-shaped (24 = 8 unique;
        # may be reported as more if triangulation duplicates).
        assert len(mesh.vertices) >= 8
        assert len(mesh.faces) >= 12  # 6 sides × 2 triangles

    def test_height_default_for_none(self):
        """Building with height_m=None should still extrude (using default)."""
        b = Building(
            footprint_enu=_square_building(0, 0).footprint_enu,
            height_m=None,
            height_source=HeightSource.NONE,
            osm_id=1, building_type="yes",
        )
        mesh = buildings_to_mesh([b])
        assert len(mesh.vertices) >= 8
        # Mesh max Z should equal the default height (10.0 m).
        assert mesh.vertices[:, 2].max() == pytest.approx(10.0)

    def test_invalid_footprint_skipped_not_raised(self):
        """Self-intersecting footprint should not crash; M2 fails soft."""
        # Bow-tie polygon (self-intersecting)
        bowtie = np.array([
            [0, 0], [10, 10], [10, 0], [0, 10], [0, 0],
        ], dtype=np.float64)
        bad = Building(
            footprint_enu=bowtie, height_m=20.0,
            height_source=HeightSource.TAG,
            osm_id=99, building_type="commercial",
        )
        good = _square_building(50, 50)
        mesh = buildings_to_mesh([bad, good])
        # Should have rendered at least the good one.
        assert len(mesh.vertices) >= 8


class TestRenderDepthBasic:
    def test_empty_buildings_returns_all_sky(self):
        cam = _north_facing_camera()
        depth = render_depth([], cam)
        assert depth.shape == (cam.height, cam.width)
        # All pixels are sky — should equal the SCP sky constant, not
        # np.inf (M2 now writes finite depth so WorldMirror's
        # nan_to_num won't silently coerce sky to 0).
        assert np.all(depth == SKY_DEPTH_M)
        assert np.all(np.isfinite(depth))

    def test_output_shape_and_dtype(self):
        cam = _north_facing_camera(width=320, height=240)
        b = _square_building(0, 50)   # 50 m north of camera
        depth = render_depth([b], cam)
        assert depth.shape == (240, 320)
        assert depth.dtype == np.float32

    def test_building_appears_in_image(self):
        """A building directly in front should produce some non-sky hits."""
        cam = _north_facing_camera()
        b = _square_building(0, 50, size=20, height=30)
        depth = render_depth([b], cam)
        # Hit pixels are anything closer than SKY_DEPTH_M.
        hit_pixels = depth < SKY_DEPTH_M
        assert hit_pixels.sum() > 100, (
            f"Expected building hits, got {hit_pixels.sum()} non-sky pixels"
        )

    def test_no_building_means_sky(self):
        """Building behind the camera shouldn't appear in the depth map."""
        cam = _north_facing_camera()
        b = _square_building(0, -50, size=20, height=30)   # south
        depth = render_depth([b], cam)
        assert np.all(depth == SKY_DEPTH_M)


class TestRenderDepthCorrectness:
    """The geometric accuracy tests. These nail down conventions; if
    they fail, M3/M4/M5 will all silently produce nonsense."""

    def test_center_pixel_depth_matches_distance(self):
        """A camera at origin facing north sees a wall 50m north.
        The center pixel's depth (camera Z) should be ~50m."""
        cam = _north_facing_camera(width=320, height=240)
        # A wide, tall building 80m north — guaranteed to fill center.
        b = _square_building(0, 50, size=40, height=30)
        depth = render_depth([b], cam)

        # Center pixel of a 320x240 image with pixel-center convention
        # at (cx=160, cy=120) maps to pixel (159 or 160, 119 or 120).
        # The center ray's camera-frame Z hits the wall's near face at
        # north = 50 - 20 = 30  (footprint is 40m wide centered at y=50,
        # so the south face is at y=30).
        # Camera is at y=0, so depth = 30 m.
        center_depth = depth[120, 160]
        assert center_depth == pytest.approx(30.0, abs=0.1), (
            f"Center pixel depth {center_depth} not ~30m"
        )

    def test_distant_building_further_than_near(self):
        """Two buildings at different distances. Near building should
        report smaller depth in pixels where it occludes."""
        cam = _north_facing_camera(width=320, height=240)
        near = _square_building(0, 30, size=10, height=20)
        far = _square_building(0, 80, size=40, height=30)
        depth = render_depth([near, far], cam)

        # The near building's south face is at north=25, the far building's
        # south face is at north=60. Center pixel should see the near
        # building at depth 25m. (Camera height 1.5m, building height 20m,
        # so the building covers the center.)
        center_depth = depth[120, 160]
        assert center_depth == pytest.approx(25.0, abs=0.5)

    def test_lateral_pixel_depth_equals_center_for_flat_wall(self):
        """Pixels at the image edge view the same wall at oblique angles,
        so camera-frame Z is still ~30m (Z is perpendicular distance to
        the wall, not Euclidean ray length). For a *long* north-south
        wall this property holds; for a finite wall the edge pixels see
        sky beyond it. Use a wide wall to test."""
        cam = _north_facing_camera(width=320, height=240)
        # Wall 100m wide, 30m tall, 50m north. South face at y=30.
        b = _square_building(0, 80, size=100, height=30)
        depth = render_depth([b], cam)

        center = depth[120, 160]
        # Lateral pixel that still hits the wall (just off-center)
        lateral = depth[120, 200]
        # Both should be ~30m (depth_z = perpendicular to image plane).
        assert center == pytest.approx(30.0, abs=0.1)
        assert lateral == pytest.approx(30.0, abs=0.1)


class TestRenderDepthBatch:
    def test_writes_npy_files_with_correct_stems(self, tmp_path):
        cams = [_north_facing_camera()] * 3
        stems = ["img_001", "img_002", "img_003"]
        b = _square_building(0, 30)

        paths = render_depth_batch([b], cams, tmp_path, stems)
        assert len(paths) == 3
        for p, stem in zip(paths, stems):
            assert p.exists()
            assert p.name == f"{stem}.npy"
            arr = np.load(p)
            assert arr.shape == (240, 320)
            assert arr.dtype == np.float32

    def test_empty_buildings_writes_all_sky(self, tmp_path):
        cams = [_north_facing_camera()]
        stems = ["only"]
        paths = render_depth_batch([], cams, tmp_path, stems)
        arr = np.load(paths[0])
        assert np.all(arr == SKY_DEPTH_M)
        assert np.all(np.isfinite(arr))

    def test_length_mismatch_raises(self, tmp_path):
        cams = [_north_facing_camera()]
        stems = ["a", "b"]
        with pytest.raises(ValueError, match="equal length"):
            render_depth_batch([], cams, tmp_path, stems)