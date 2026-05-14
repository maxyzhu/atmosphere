"""
Tests for the Mapillary retrieval module (vector-tile pipeline).

Strategy: we do NOT hit Mapillary's tile server in unit tests — too slow,
requires a real token, and tile contents are non-deterministic. For
end-to-end coverage we patch `_fetch_tile_bytes` and `_decode_image_features`
to return crafted features, then exercise the FPS + Graph-API + download
plumbing on top.

Pure helpers (bbox math, density formula, FPS) are tested directly.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from atmosphere.retrieval.mapillary import (
    BUFFER_M,
    IMAGES_PER_M2,
    MapillaryImage,
    _density_target,
    _farthest_point_sample,
    _square_bbox_with_buffer,
    fetch_mapillary_images,
)


DLR_LAT = 47.6059
DLR_LON = -122.3392


# -----------------------------------------------------------------------------
# Bbox math
# -----------------------------------------------------------------------------


class TestSquareBboxWithBuffer:
    def test_order_is_west_south_east_north(self):
        w, s, e, n = _square_bbox_with_buffer(DLR_LAT, DLR_LON, 100)
        assert w < DLR_LON < e
        assert s < DLR_LAT < n

    def test_symmetric_around_center(self):
        w, s, e, n = _square_bbox_with_buffer(DLR_LAT, DLR_LON, 100)
        assert abs((DLR_LON - w) - (e - DLR_LON)) < 1e-10
        assert abs((DLR_LAT - s) - (n - DLR_LAT)) < 1e-10

    def test_half_side_includes_buffer(self):
        """Half-side in meters = half_side_m + BUFFER_M."""
        half_side = 100.0
        w, s, e, n = _square_bbox_with_buffer(DLR_LAT, DLR_LON, half_side)

        # Convert ns extent back to meters via the lat-deg/m factor
        # used internally. Should equal 2 * (half_side + BUFFER_M).
        meters_ns = (n - s) * 111_320.0
        expected = 2 * (half_side + BUFFER_M)
        assert meters_ns == pytest.approx(expected, rel=1e-6)

    def test_size_scales_linearly_with_half_side(self):
        """Doubling the half-side roughly doubles the bbox extent (with
        buffer offset). Sanity check that the formula isn't quadratic."""
        w1, _, e1, _ = _square_bbox_with_buffer(DLR_LAT, DLR_LON, 100)
        w2, _, e2, _ = _square_bbox_with_buffer(DLR_LAT, DLR_LON, 1000)
        # (1000+20) / (100+20) ≈ 8.5
        ratio = (e2 - w2) / (e1 - w1)
        assert 8.0 < ratio < 9.0


# -----------------------------------------------------------------------------
# Density-based target count
# -----------------------------------------------------------------------------


class TestDensityTarget:
    def test_formula(self):
        """target = side^2 * IMAGES_PER_M2, where side = 2*(r + buffer)."""
        half_side = 150.0
        side = 2 * (half_side + BUFFER_M)
        expected = round(side * side * IMAGES_PER_M2)
        assert _density_target(half_side) == expected

    def test_min_one(self):
        """A tiny radius mustn't yield zero — we always want at least one."""
        assert _density_target(0.0) >= 1

    def test_grows_with_radius(self):
        assert _density_target(50.0) < _density_target(150.0) < _density_target(500.0)


# -----------------------------------------------------------------------------
# Farthest-point sampling (no seed param; deterministic center-seeded)
# -----------------------------------------------------------------------------


def _make_image(mid: str, east: float, north: float,
                compass: float | None = 0.0,
                is_pano: bool = False) -> MapillaryImage:
    return MapillaryImage(
        mapillary_id=mid,
        position_enu=(east, north),
        compass_angle_deg=compass,
        is_pano=is_pano,
        thumb_url="",
        thumb_path=None,
    )


class TestFarthestPointSample:
    def test_below_target_returns_all(self):
        images = [_make_image(str(i), i * 10, 0) for i in range(5)]
        result = _farthest_point_sample(images, target_count=10)
        assert len(result) == 5

    def test_exact_target_returns_all(self):
        images = [_make_image(str(i), i * 10, 0) for i in range(10)]
        result = _farthest_point_sample(images, target_count=10)
        assert len(result) == 10

    def test_seeds_from_center(self):
        """The first selected image must be the one closest to (0, 0)."""
        images = [
            _make_image("center", 0.5, 0.5),       # closest to origin
            _make_image("ne", 100, 100),
            _make_image("sw", -100, -100),
            _make_image("nw", -100, 100),
        ]
        result = _farthest_point_sample(images, target_count=3)
        assert result[0].mapillary_id == "center"

    def test_downsampling_covers_extremes(self):
        """Sampling 5 from 100 points along a line should hit both ends."""
        images = [_make_image(str(i), i, 0) for i in range(100)]
        result = _farthest_point_sample(images, target_count=5)
        xs = sorted(r.position_enu[0] for r in result)
        # Center seed is index 0 (closest to origin). Greedy should
        # immediately jump to the far end, then bisect.
        assert xs[0] < 5     # near start (center seed)
        assert xs[-1] > 90   # opposite end picked

    def test_fully_deterministic(self):
        """Same input → identical output, no randomness anywhere."""
        images = [_make_image(str(i), i, 0) for i in range(50)]
        r1 = _farthest_point_sample(images, target_count=10)
        r2 = _farthest_point_sample(images, target_count=10)
        assert [r.mapillary_id for r in r1] == [r.mapillary_id for r in r2]

    def test_missing_compass_does_not_crash(self):
        images = [
            _make_image("a", 0, 0, compass=None),
            _make_image("b", 50, 0, compass=90),
            _make_image("c", 100, 0, compass=None),
            _make_image("d", 0, 50, compass=180),
        ]
        result = _farthest_point_sample(images, target_count=3)
        assert len(result) == 3


# -----------------------------------------------------------------------------
# fetch_mapillary_images: end-to-end with the tile layer mocked out
# -----------------------------------------------------------------------------


@pytest.fixture
def fake_token(monkeypatch):
    """Set a dummy token so config passes validation."""
    monkeypatch.setenv("MAPILLARY_ACCESS_TOKEN", "MLY|fake|token")
    from atmosphere.config import get_mapillary_token
    get_mapillary_token.cache_clear()


def _fake_features(n: int, center_lat: float, center_lon: float) -> list[dict]:
    """Build n synthetic feature dicts in the shape `_decode_image_features`
    returns: {id, lon, lat, compass_angle, is_pano}."""
    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        dlat = rng.uniform(-0.0008, 0.0008)
        dlon = rng.uniform(-0.0012, 0.0012)
        out.append({
            "id": f"img_{i:04d}",
            "lon": center_lon + dlon,
            "lat": center_lat + dlat,
            "compass_angle": float(rng.uniform(0, 360)) if i % 5 != 0 else None,
            "is_pano": (i % 7 == 0),
        })
    return out


class TestFetchMapillaryImages:
    """Patch `_fetch_tile_bytes` (returns a dummy non-None payload) and
    `_decode_image_features` (returns crafted features), so the public
    function exercises bbox math + parsing + FPS without any HTTP."""

    def test_basic_fetch_and_parse(self, tmp_path, fake_token):
        features = _fake_features(30, DLR_LAT, DLR_LON)

        with patch(
            "atmosphere.retrieval.mapillary._fetch_tile_bytes",
            return_value=b"dummy_pbf",
        ), patch(
            "atmosphere.retrieval.mapillary._decode_image_features",
            return_value=features,
        ):
            images = fetch_mapillary_images(
                lat=DLR_LAT, lon=DLR_LON, radius_m=100,
                target_count=10,
                download_thumbnails=False,
                cache_dir=tmp_path,
                use_cache=False,
            )

        assert len(images) == 10
        assert all(isinstance(img, MapillaryImage) for img in images)
        for img in images:
            e, n = img.position_enu
            # Bbox half-side is 100 + BUFFER_M = 120 m
            assert abs(e) < 200
            assert abs(n) < 200

    def test_is_pano_propagated_from_features(self, tmp_path, fake_token):
        """The is_pano flag from tile properties must reach MapillaryImage."""
        features = _fake_features(30, DLR_LAT, DLR_LON)

        with patch(
            "atmosphere.retrieval.mapillary._fetch_tile_bytes",
            return_value=b"dummy_pbf",
        ), patch(
            "atmosphere.retrieval.mapillary._decode_image_features",
            return_value=features,
        ):
            images = fetch_mapillary_images(
                lat=DLR_LAT, lon=DLR_LON, radius_m=100,
                target_count=30,
                download_thumbnails=False,
                cache_dir=tmp_path,
                use_cache=False,
            )

        # _fake_features marks every 7th item as a pano. After filtering
        # to bbox + FPS, at least one pano should survive.
        panos = [img for img in images if img.is_pano]
        assert len(panos) >= 1

    def test_missing_compass_preserved(self, tmp_path, fake_token):
        features = _fake_features(30, DLR_LAT, DLR_LON)

        with patch(
            "atmosphere.retrieval.mapillary._fetch_tile_bytes",
            return_value=b"dummy_pbf",
        ), patch(
            "atmosphere.retrieval.mapillary._decode_image_features",
            return_value=features,
        ):
            images = fetch_mapillary_images(
                lat=DLR_LAT, lon=DLR_LON, radius_m=100,
                target_count=30,
                download_thumbnails=False,
                cache_dir=tmp_path,
                use_cache=False,
            )

        # Every 5th item in _fake_features has compass=None.
        no_compass = [img for img in images if not img.has_compass]
        assert len(no_compass) >= 1

    def test_empty_tiles_return_empty_list(self, tmp_path, fake_token):
        """All tiles in the bbox return empty / None → no images, no crash."""
        with patch(
            "atmosphere.retrieval.mapillary._fetch_tile_bytes",
            return_value=None,
        ):
            images = fetch_mapillary_images(
                lat=DLR_LAT, lon=DLR_LON, radius_m=100,
                target_count=10,
                download_thumbnails=False,
                cache_dir=tmp_path,
                use_cache=False,
            )
        assert images == []

    def test_features_outside_bbox_are_filtered(self, tmp_path, fake_token):
        """Tiles overlap the bbox; features outside it must be dropped."""
        features = [
            {
                "id": "inside",
                "lon": DLR_LON,
                "lat": DLR_LAT,
                "compass_angle": 90.0,
                "is_pano": False,
            },
            {
                "id": "outside",
                "lon": DLR_LON + 1.0,    # ~80 km east — far beyond bbox
                "lat": DLR_LAT,
                "compass_angle": 90.0,
                "is_pano": False,
            },
        ]

        with patch(
            "atmosphere.retrieval.mapillary._fetch_tile_bytes",
            return_value=b"dummy_pbf",
        ), patch(
            "atmosphere.retrieval.mapillary._decode_image_features",
            return_value=features,
        ):
            images = fetch_mapillary_images(
                lat=DLR_LAT, lon=DLR_LON, radius_m=100,
                target_count=10,
                download_thumbnails=False,
                cache_dir=tmp_path,
                use_cache=False,
            )

        ids = [img.mapillary_id for img in images]
        assert ids == ["inside"]

    def test_duplicate_ids_deduped(self, tmp_path, fake_token):
        """A feature appearing in two overlapping tiles must be kept once."""
        features = [
            {"id": "dup", "lon": DLR_LON, "lat": DLR_LAT,
             "compass_angle": 0.0, "is_pano": False},
            {"id": "dup", "lon": DLR_LON, "lat": DLR_LAT,
             "compass_angle": 0.0, "is_pano": False},
            {"id": "uniq", "lon": DLR_LON, "lat": DLR_LAT,
             "compass_angle": 0.0, "is_pano": False},
        ]

        with patch(
            "atmosphere.retrieval.mapillary._fetch_tile_bytes",
            return_value=b"dummy_pbf",
        ), patch(
            "atmosphere.retrieval.mapillary._decode_image_features",
            return_value=features,
        ):
            images = fetch_mapillary_images(
                lat=DLR_LAT, lon=DLR_LON, radius_m=100,
                target_count=10,
                download_thumbnails=False,
                cache_dir=tmp_path,
                use_cache=False,
            )

        assert sorted(img.mapillary_id for img in images) == ["dup", "uniq"]

    def test_target_count_none_uses_density(self, tmp_path, fake_token):
        """target_count=None → density auto-compute. Verify it doesn't
        crash and respects the pool ceiling when pool is small."""
        features = _fake_features(5, DLR_LAT, DLR_LON)

        with patch(
            "atmosphere.retrieval.mapillary._fetch_tile_bytes",
            return_value=b"dummy_pbf",
        ), patch(
            "atmosphere.retrieval.mapillary._decode_image_features",
            return_value=features,
        ):
            images = fetch_mapillary_images(
                lat=DLR_LAT, lon=DLR_LON, radius_m=100,
                target_count=None,  # density-driven
                download_thumbnails=False,
                cache_dir=tmp_path,
                use_cache=False,
            )

        # Density target for r=100 is ~67, pool is 5 → all kept.
        assert len(images) == 5
