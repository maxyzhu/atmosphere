"""
Tests for the Mapillary retrieval module.

Strategy: we do NOT hit the real Mapillary API in unit tests — too slow,
requires a real token, and results are non-deterministic. Instead, we
mock `requests.get` to return crafted payloads.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from atmosphere.geo import LocalFrame
from atmosphere.retrieval.mapillary import (
    MapillaryImage,
    _farthest_point_sample,
    _radius_to_bbox,
    fetch_mapillary_images,
)


DLR_LAT = 47.6059
DLR_LON = -122.3392


# -----------------------------------------------------------------------------
# Bbox math
# -----------------------------------------------------------------------------


class TestRadiusToBbox:
    def test_order_is_west_south_east_north(self):
        w, s, e, n = _radius_to_bbox(DLR_LAT, DLR_LON, 100)
        assert w < DLR_LON < e
        assert s < DLR_LAT < n

    def test_symmetric_around_center(self):
        w, s, e, n = _radius_to_bbox(DLR_LAT, DLR_LON, 100)
        # west-to-center ≈ center-to-east
        assert abs((DLR_LON - w) - (e - DLR_LON)) < 1e-10
        # south-to-center ≈ center-to-north
        assert abs((DLR_LAT - s) - (n - DLR_LAT)) < 1e-10

    def test_size_scales_with_radius(self):
        w1, _, e1, _ = _radius_to_bbox(DLR_LAT, DLR_LON, 100)
        w2, _, e2, _ = _radius_to_bbox(DLR_LAT, DLR_LON, 200)
        assert (e2 - w2) == pytest.approx(2 * (e1 - w1), rel=1e-6)


# -----------------------------------------------------------------------------
# Farthest-point sampling
# -----------------------------------------------------------------------------


def _make_image(mid: str, east: float, north: float,
                compass: float | None = 0.0) -> MapillaryImage:
    return MapillaryImage(
        mapillary_id=mid,
        position_enu=(east, north),
        compass_angle_deg=compass,
        captured_at=datetime(2025, 1, 1),
        thumb_url="",
        thumb_path=None,
    )


class TestFarthestPointSample:
    def test_below_target_returns_all(self):
        images = [_make_image(str(i), i * 10, 0) for i in range(5)]
        result = _farthest_point_sample(images, target_count=10)
        assert len(result) == 5

    def test_exact_target(self):
        images = [_make_image(str(i), i * 10, 0) for i in range(10)]
        result = _farthest_point_sample(images, target_count=10)
        assert len(result) == 10

    def test_downsampling_favors_extremes(self):
        # 100 images along a line from (0,0) to (100,0).
        # Sampling 5 should hit both ends and 3 interior.
        images = [_make_image(str(i), i, 0) for i in range(100)]
        result = _farthest_point_sample(images, target_count=5, seed=0)
        xs = sorted(r.position_enu[0] for r in result)
        # The spread of selected points should be close to the full range.
        # Without farthest-point this would cluster randomly.
        assert xs[0] < 20  # at least one near start
        assert xs[-1] > 80  # at least one near end

    def test_reproducibility(self):
        images = [_make_image(str(i), i, 0) for i in range(50)]
        r1 = _farthest_point_sample(images, target_count=10, seed=42)
        r2 = _farthest_point_sample(images, target_count=10, seed=42)
        ids1 = [r.mapillary_id for r in r1]
        ids2 = [r.mapillary_id for r in r2]
        assert ids1 == ids2

    def test_missing_compass_does_not_crash(self):
        images = [
            _make_image("a", 0, 0, compass=None),
            _make_image("b", 50, 0, compass=90),
            _make_image("c", 100, 0, compass=None),
            _make_image("d", 0, 50, compass=180),
        ]
        result = _farthest_point_sample(images, target_count=3, seed=0)
        assert len(result) == 3


# -----------------------------------------------------------------------------
# fetch_mapillary_images with mocked HTTP
# -----------------------------------------------------------------------------


def _mock_mapillary_response(n_items: int, lat: float, lon: float) -> dict:
    """Construct a payload shaped like Mapillary v4's /images response."""
    rng = np.random.default_rng(0)
    data = []
    for i in range(n_items):
        # Small random offset in degrees (a few tens of meters)
        dlat = rng.uniform(-0.0008, 0.0008)
        dlon = rng.uniform(-0.0012, 0.0012)
        data.append({
            "id": f"img_{i:04d}",
            "geometry": {
                "type": "Point",
                "coordinates": [lon + dlon, lat + dlat],
            },
            "compass_angle": float(rng.uniform(0, 360)) if i % 5 != 0 else None,
            "captured_at": 1700000000000 + i * 1000,
            "thumb_256_url": f"https://example.com/thumb_{i}.jpg",
        })
    return {"data": data}


@pytest.fixture
def fake_token(monkeypatch):
    """Set a dummy token so config passes validation."""
    monkeypatch.setenv("MAPILLARY_ACCESS_TOKEN", "MLY|fake|token")
    # Clear the lru_cache on get_mapillary_token
    from atmosphere.config import get_mapillary_token
    get_mapillary_token.cache_clear()


class TestFetchMapillaryImages:
    def test_basic_fetch_and_parse(self, tmp_path, fake_token):
        payload = _mock_mapillary_response(20, DLR_LAT, DLR_LON)

        mock_response = MagicMock()
        mock_response.json.return_value = payload
        mock_response.raise_for_status.return_value = None

        with patch(
            "atmosphere.retrieval.mapillary.requests.get",
            return_value=mock_response,
        ):
            images = fetch_mapillary_images(
                lat=DLR_LAT, lon=DLR_LON, radius_m=100,
                target_count=10,
                download_thumbnails=False,  # avoid real HTTP
                cache_dir=tmp_path,
                use_cache=False,
            )

        assert len(images) == 10
        assert all(isinstance(img, MapillaryImage) for img in images)
        # ENU positions should be near origin (within the radius)
        for img in images:
            e, n = img.position_enu
            assert abs(e) < 200  # bbox size
            assert abs(n) < 200

    def test_missing_compass_preserved(self, tmp_path, fake_token):
        payload = _mock_mapillary_response(20, DLR_LAT, DLR_LON)

        mock_response = MagicMock()
        mock_response.json.return_value = payload
        mock_response.raise_for_status.return_value = None

        with patch(
            "atmosphere.retrieval.mapillary.requests.get",
            return_value=mock_response,
        ):
            images = fetch_mapillary_images(
                lat=DLR_LAT, lon=DLR_LON, radius_m=100,
                target_count=20,
                download_thumbnails=False,
                cache_dir=tmp_path,
                use_cache=False,
            )

        # Original payload has None compass every 5th item (i % 5 == 0)
        # After sampling, some should have no compass
        no_compass = [img for img in images if not img.has_compass]
        assert len(no_compass) >= 1

    def test_empty_response_returns_empty_list(self, tmp_path, fake_token):
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": []}
        mock_response.raise_for_status.return_value = None

        with patch(
            "atmosphere.retrieval.mapillary.requests.get",
            return_value=mock_response,
        ):
            images = fetch_mapillary_images(
                lat=DLR_LAT, lon=DLR_LON, radius_m=100,
                target_count=10,
                download_thumbnails=False,
                cache_dir=tmp_path,
                use_cache=False,
            )
        assert images == []

    def test_cache_reused(self, tmp_path, fake_token):
        payload = _mock_mapillary_response(20, DLR_LAT, DLR_LON)

        mock_response = MagicMock()
        mock_response.json.return_value = payload
        mock_response.raise_for_status.return_value = None

        with patch(
            "atmosphere.retrieval.mapillary.requests.get",
            return_value=mock_response,
        ) as mocked:
            fetch_mapillary_images(
                lat=DLR_LAT, lon=DLR_LON, radius_m=100,
                target_count=5, download_thumbnails=False,
                cache_dir=tmp_path, use_cache=True,
            )
            assert mocked.call_count == 1

        # Second call should not hit the API
        with patch(
            "atmosphere.retrieval.mapillary.requests.get",
            return_value=mock_response,
        ) as mocked2:
            fetch_mapillary_images(
                lat=DLR_LAT, lon=DLR_LON, radius_m=100,
                target_count=5, download_thumbnails=False,
                cache_dir=tmp_path, use_cache=True,
            )
            assert mocked2.call_count == 0

    def test_malformed_items_skipped_gracefully(self, tmp_path, fake_token):
        """Items missing required fields should be skipped, not crash."""
        payload = {
            "data": [
                # Valid
                {
                    "id": "good1",
                    "geometry": {"coordinates": [DLR_LON, DLR_LAT]},
                    "compass_angle": 90.0,
                    "captured_at": 1700000000000,
                    "thumb_256_url": "https://example.com/1.jpg",
                },
                # Missing geometry
                {
                    "id": "bad1",
                    "compass_angle": 90.0,
                    "captured_at": 1700000000000,
                },
                # Missing captured_at
                {
                    "id": "bad2",
                    "geometry": {"coordinates": [DLR_LON, DLR_LAT]},
                    "compass_angle": 90.0,
                },
            ]
        }

        mock_response = MagicMock()
        mock_response.json.return_value = payload
        mock_response.raise_for_status.return_value = None

        with patch(
            "atmosphere.retrieval.mapillary.requests.get",
            return_value=mock_response,
        ):
            images = fetch_mapillary_images(
                lat=DLR_LAT, lon=DLR_LON, radius_m=100,
                target_count=10, download_thumbnails=False,
                cache_dir=tmp_path, use_cache=False,
            )

        # Only the good item survives parsing
        assert len(images) == 1
        assert images[0].mapillary_id == "good1"