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
    ASPECT_TOLERANCE,
    BUFFER_M,
    IMAGES_PER_M2,
    TARGET_ASPECT_RATIO,
    MapillaryImage,
    _density_target,
    _farthest_point_sample,
    _matches_aspect,
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


# -----------------------------------------------------------------------------
# Aspect filter
# -----------------------------------------------------------------------------


class TestMatchesAspect:
    """
    `_matches_aspect` is the protocol-level aspect predicate. It must
    reject metadata-incomplete images (None width/height), accept exact
    matches, and respect tolerance for near-matches.
    """

    def test_exact_match(self):
        # 1920×1080 = 1.778 exactly
        assert _matches_aspect(1920, 1080, 16.0 / 9.0, 0.02)

    def test_near_match_within_tolerance(self):
        # 2048×1152 = 1.7778, also 16:9
        assert _matches_aspect(2048, 1152, 16.0 / 9.0, 0.02)

    def test_just_outside_tolerance(self):
        # 2048×1200 = 1.707 — outside 16:9 ± 0.02
        assert not _matches_aspect(2048, 1200, 16.0 / 9.0, 0.02)

    def test_4_to_3_rejected_when_target_is_16_to_9(self):
        # 2048×1536 = 4:3 — should be filtered out when targeting 16:9
        assert not _matches_aspect(2048, 1536, 16.0 / 9.0, 0.02)

    def test_square_rejected_when_target_is_16_to_9(self):
        assert not _matches_aspect(1920, 1920, 16.0 / 9.0, 0.02)

    def test_missing_width_rejected(self):
        assert not _matches_aspect(None, 1080, 16.0 / 9.0, 0.02)

    def test_missing_height_rejected(self):
        assert not _matches_aspect(1920, None, 16.0 / 9.0, 0.02)

    def test_zero_height_rejected(self):
        # Defensive: 0 height would divide-by-zero; we treat it as invalid.
        assert not _matches_aspect(1920, 0, 16.0 / 9.0, 0.02)

    @pytest.mark.parametrize("tolerance", [0.001, 0.01, 0.05, 0.1])
    def test_tolerance_widens_acceptance(self, tolerance):
        # 2048×1200 = 1.707 (diff to 1.778 ≈ 0.071).
        # Should be rejected at ≤0.05 tol, accepted at 0.1.
        result = _matches_aspect(2048, 1200, 16.0 / 9.0, tolerance)
        assert result == (tolerance >= 0.071)

    def test_target_aspect_ratio_constant_is_16_to_9(self):
        assert TARGET_ASPECT_RATIO == pytest.approx(16.0 / 9.0)

    def test_aspect_tolerance_constant_in_reasonable_range(self):
        # Sanity: tolerance should be tight enough to actually filter,
        # but loose enough to accept the 1920×1080 / 2048×1152 family.
        assert 0.001 <= ASPECT_TOLERANCE <= 0.05


class TestFetchMapillaryAspectFilter:
    """
    End-to-end aspect filtering at the `fetch_mapillary_images` level.

    We mock the tile-fetch and Graph-API metadata calls so the test runs
    offline. The pool is hand-crafted to have a known mix of aspect
    ratios, so we can verify that:

    1. The target_count is met when the pool has enough 16:9 images.
    2. Non-matching aspects are filtered out.
    3. fetch_camera_metadata is auto-forced True when aspect filtering.
    4. When the pool is too small even for escalation, we get a warning
       and return fewer than target_count images (no exception).
    """

    def _make_features(self, n_169: int, n_43: int) -> list[dict]:
        """
        Build a list of fake tile features. Positions are scattered
        across a 100 m circle so FPS has something to chew on.
        Even-indexed ids will be tagged 16:9 in the metadata mock,
        odd-indexed will be 4:3.

        First n_169 features are intended to be 16:9, next n_43 are 4:3.
        """
        feats = []
        total = n_169 + n_43
        for i in range(total):
            # Distribute around a circle so FPS picks a spread.
            theta = 2 * np.pi * i / max(total, 1)
            d_lat = 0.0005 * np.cos(theta)  # ~50 m
            d_lon = 0.0005 * np.sin(theta)
            feats.append({
                "id": str(1_000_000 + i),
                "lon": DLR_LON + d_lon,
                "lat": DLR_LAT + d_lat,
                "compass_angle": float((i * 37) % 360),
                "is_pano": False,
            })
        return feats

    def _make_metadata_mock(self, ids_169: set[str]):
        """
        Return a fake `_fetch_camera_metadata` that gives 1920×1080 for
        ids in `ids_169` and 2048×1536 otherwise.
        """
        def _mock(image_id, cache_dir, use_cache=True, timeout_s=15.0):
            if image_id in ids_169:
                w, h = 1920, 1080
            else:
                w, h = 2048, 1536
            return {
                "focal_ratio": 0.5,
                "computed_rotation": (0.1, 0.2, 0.3),
                "width": w,
                "height": h,
                "computed_compass_angle": 0.0,
            }
        return _mock

    def test_filters_to_target_aspect(self, tmp_path):
        # Pool: 30 16:9 + 20 4:3. Target 10. Should keep 10 16:9.
        feats = self._make_features(30, 20)
        ids_169 = {f["id"] for f in feats[:30]}

        with patch(
            "atmosphere.retrieval.mapillary._fetch_tile_bytes",
            return_value=b"fake_pbf",
        ), patch(
            "atmosphere.retrieval.mapillary._decode_image_features",
            return_value=feats,
        ), patch(
            "atmosphere.retrieval.mapillary._fetch_thumb_url",
            return_value="http://fake/thumb.jpg",
        ), patch(
            "atmosphere.retrieval.mapillary._download_thumbnail",
            return_value=True,
        ), patch(
            "atmosphere.retrieval.mapillary._fetch_camera_metadata",
            side_effect=self._make_metadata_mock(ids_169),
        ):
            images = fetch_mapillary_images(
                DLR_LAT, DLR_LON,
                radius_m=100.0,
                target_count=10,
                download_thumbnails=False,
                fetch_camera_metadata=False,  # should be auto-forced
                target_aspect_ratio=16.0 / 9.0,
                cache_dir=tmp_path / "cache",
            )

        # All survivors must be 16:9.
        assert len(images) == 10
        for img in images:
            assert img.width == 1920 and img.height == 1080
            assert img.mapillary_id in ids_169

    def test_underfill_yields_what_we_have(self, tmp_path):
        # Pool: 5 16:9 + 5 4:3. Target 10. Can't escalate to find more;
        # should return 5 with a warning logged.
        feats = self._make_features(5, 5)
        ids_169 = {f["id"] for f in feats[:5]}

        with patch(
            "atmosphere.retrieval.mapillary._fetch_tile_bytes",
            return_value=b"fake_pbf",
        ), patch(
            "atmosphere.retrieval.mapillary._decode_image_features",
            return_value=feats,
        ), patch(
            "atmosphere.retrieval.mapillary._fetch_thumb_url",
            return_value="http://fake/thumb.jpg",
        ), patch(
            "atmosphere.retrieval.mapillary._download_thumbnail",
            return_value=True,
        ), patch(
            "atmosphere.retrieval.mapillary._fetch_camera_metadata",
            side_effect=self._make_metadata_mock(ids_169),
        ):
            images = fetch_mapillary_images(
                DLR_LAT, DLR_LON,
                radius_m=100.0,
                target_count=10,
                download_thumbnails=False,
                target_aspect_ratio=16.0 / 9.0,
                cache_dir=tmp_path / "cache",
            )

        # Got the 5 we could find, no exception
        assert len(images) == 5
        for img in images:
            assert img.mapillary_id in ids_169
            assert img.width == 1920

    def test_no_aspect_filter_keeps_all_aspects(self, tmp_path):
        # When target_aspect_ratio is None, all aspects pass through.
        feats = self._make_features(5, 5)
        ids_169 = {f["id"] for f in feats[:5]}

        with patch(
            "atmosphere.retrieval.mapillary._fetch_tile_bytes",
            return_value=b"fake_pbf",
        ), patch(
            "atmosphere.retrieval.mapillary._decode_image_features",
            return_value=feats,
        ), patch(
            "atmosphere.retrieval.mapillary._fetch_thumb_url",
            return_value="http://fake/thumb.jpg",
        ), patch(
            "atmosphere.retrieval.mapillary._download_thumbnail",
            return_value=True,
        ), patch(
            "atmosphere.retrieval.mapillary._fetch_camera_metadata",
            side_effect=self._make_metadata_mock(ids_169),
        ):
            images = fetch_mapillary_images(
                DLR_LAT, DLR_LON,
                radius_m=100.0,
                target_count=10,
                download_thumbnails=False,
                fetch_camera_metadata=True,
                target_aspect_ratio=None,
                cache_dir=tmp_path / "cache",
            )

        # Mixed aspects returned because filter is off.
        widths = {img.width for img in images}
        assert widths == {1920, 2048}
        assert len(images) == 10

    def test_escalation_kicks_in_when_initial_pass_underfills(self, tmp_path):
        # Pool: 24 16:9 (sparse), 26 4:3. Target 24.
        # Initial FPS at 2.5×=60 over 50 → takes all 50, 24 are 16:9 →
        # actually returns 24 without escalation. To force escalation
        # path, weight pool more heavily 4:3: 30 16:9 + 70 4:3 = 100.
        # FPS 2.5× = 60 → expected ~18 16:9 (30/100 * 60) → underfilled,
        # should escalate to 4.5×=108 capped to 100 → all 30 16:9 found.
        feats = self._make_features(30, 70)
        ids_169 = {f["id"] for f in feats[:30]}

        with patch(
            "atmosphere.retrieval.mapillary._fetch_tile_bytes",
            return_value=b"fake_pbf",
        ), patch(
            "atmosphere.retrieval.mapillary._decode_image_features",
            return_value=feats,
        ), patch(
            "atmosphere.retrieval.mapillary._fetch_thumb_url",
            return_value="http://fake/thumb.jpg",
        ), patch(
            "atmosphere.retrieval.mapillary._download_thumbnail",
            return_value=True,
        ), patch(
            "atmosphere.retrieval.mapillary._fetch_camera_metadata",
            side_effect=self._make_metadata_mock(ids_169),
        ):
            images = fetch_mapillary_images(
                DLR_LAT, DLR_LON,
                radius_m=100.0,
                target_count=24,
                download_thumbnails=False,
                target_aspect_ratio=16.0 / 9.0,
                cache_dir=tmp_path / "cache",
            )

        # Should hit the 24 target after escalating.
        assert len(images) == 24
        for img in images:
            assert img.mapillary_id in ids_169
            assert img.width == 1920
