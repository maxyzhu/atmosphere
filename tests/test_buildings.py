"""
Tests for the building retrieval module.

Strategy: we do NOT hit the real Overpass API in unit tests — it would
be slow, flaky, and subject to rate limits. Instead, we mock
`osmnx.features_from_point` to return a crafted GeoDataFrame, and verify
that our conversion + filtering + provenance logic handles it correctly.

A separate integration test (not in this file) would exercise the real
API; that one should run manually, not in CI.
"""

from __future__ import annotations

from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point, Polygon

from atmosphere.geo import LocalFrame
from atmosphere.retrieval.buildings import (
    Building,
    HeightSource,
    _extract_height,
    _parse_osm_height,
    _parse_osm_levels,
    fetch_buildings,
)


# Seattle DLR office, our canonical test point.
DLR_LAT = 47.6097
DLR_LON = -122.3331


# -----------------------------------------------------------------------------
# Unit tests for the height parsers — these are pure functions, easy to test
# -----------------------------------------------------------------------------


class TestParseOsmHeight:
    @pytest.mark.parametrize("raw,expected", [
        ("12", 12.0),
        ("12.5", 12.5),
        ("12 m", 12.0),
        ("12m", 12.0),
        ("12 meters", 12.0),
        (" 12 ", 12.0),
        (12, 12.0),
        (12.5, 12.5),
    ])
    def test_valid_heights(self, raw, expected):
        assert _parse_osm_height(raw) == pytest.approx(expected)

    @pytest.mark.parametrize("raw", [
        None, "", "nan", "tall", "~12", float("nan"),
    ])
    def test_invalid_heights_return_none(self, raw):
        assert _parse_osm_height(raw) is None

    def test_feet_converted_to_meters(self):
        # 40 feet = 12.192 meters
        assert _parse_osm_height("40'") == pytest.approx(12.192, abs=0.01)
        assert _parse_osm_height("40 ft") == pytest.approx(12.192, abs=0.01)

    def test_unreasonable_heights_rejected(self):
        # Negative, zero, and absurd values all return None.
        assert _parse_osm_height("-5") is None
        assert _parse_osm_height("0") is None
        assert _parse_osm_height("5000") is None


class TestParseOsmLevels:
    @pytest.mark.parametrize("raw,expected", [
        ("4", 4),
        ("4.5", 4),  # rounded
        (4, 4),
        ("1", 1),
    ])
    def test_valid_levels(self, raw, expected):
        assert _parse_osm_levels(raw) == expected

    @pytest.mark.parametrize("raw", [None, "", "nan", "tall", float("nan")])
    def test_invalid_levels_return_none(self, raw):
        assert _parse_osm_levels(raw) is None


class TestExtractHeight:
    def test_height_tag_preferred_over_levels(self):
        row = {"height": "15", "building:levels": "3"}
        h, src = _extract_height(row)
        assert h == 15.0
        assert src == HeightSource.TAG

    def test_levels_fallback_when_no_height(self):
        row = {"height": None, "building:levels": "4"}
        h, src = _extract_height(row)
        assert h == pytest.approx(14.0)  # 4 * 3.5
        assert src == HeightSource.LEVELS

    def test_none_when_both_missing(self):
        row = {}
        h, src = _extract_height(row)
        assert h is None
        assert src == HeightSource.NONE


# -----------------------------------------------------------------------------
# Integration-style test of fetch_buildings with a mocked OSM response
# -----------------------------------------------------------------------------


def _fake_gdf_near(lat: float, lon: float) -> gpd.GeoDataFrame:
    """
    Construct a small mock GeoDataFrame imitating what osmnx returns.

    Three buildings:
        1. A proper building with explicit height tag
        2. A building with only floor count
        3. A tiny object (< 20 m²) that should be filtered out
    """
    # Offsets in degrees ≈ meters / 111000 (latitude) / (111000 * cos(lat))
    # These are hand-crafted so the resulting ENU polygons have known sizes.
    # Small lon deltas: 0.0001 ≈ 7.5 m at latitude 47.6.

    # Building 1: ~14 m × 14 m square, height tagged
    b1 = Polygon([
        (lon, lat),
        (lon + 0.00019, lat),
        (lon + 0.00019, lat + 0.000126),
        (lon, lat + 0.000126),
        (lon, lat),
    ])

    # Building 2: ~20 m × 20 m square, only building:levels
    b2 = Polygon([
        (lon - 0.0003, lat - 0.0003),
        (lon - 0.0003 + 0.00027, lat - 0.0003),
        (lon - 0.0003 + 0.00027, lat - 0.0003 + 0.00018),
        (lon - 0.0003, lat - 0.0003 + 0.00018),
        (lon - 0.0003, lat - 0.0003),
    ])

    # Building 3: tiny ~2 m × 2 m, should get filtered
    b3 = Polygon([
        (lon + 0.0005, lat + 0.0005),
        (lon + 0.0005 + 0.00003, lat + 0.0005),
        (lon + 0.0005 + 0.00003, lat + 0.0005 + 0.00002),
        (lon + 0.0005, lat + 0.0005 + 0.00002),
        (lon + 0.0005, lat + 0.0005),
    ])

    # A Point element — osmnx returns these sometimes, we should skip.
    node_point = Point(lon + 0.001, lat + 0.001)

    # osmnx returns a GeoDataFrame with a MultiIndex (element_type, osmid).
    gdf = gpd.GeoDataFrame(
        {
            "building": ["yes", "residential", "yes", "yes"],
            "height": ["12", None, None, None],
            "building:levels": [None, "3", None, None],
            "geometry": [b1, b2, b3, node_point],
        },
        index=[
            ("way", 1001),
            ("way", 1002),
            ("way", 1003),
            ("node", 9999),
        ],
        crs="EPSG:4326",
    )
    return gdf


class TestFetchBuildings:
    def test_basic_fetch_and_convert(self, tmp_path):
        """With mocked OSM, fetch_buildings returns correctly parsed Buildings."""
        fake = _fake_gdf_near(DLR_LAT, DLR_LON)
        with patch(
            "atmosphere.retrieval.buildings.ox.features_from_point",
            return_value=fake,
        ):
            bs = fetch_buildings(
                lat=DLR_LAT, lon=DLR_LON, radius_m=100,
                cache_dir=tmp_path, use_cache=False,
            )

        # Node filtered + tiny building filtered = 2 buildings left
        assert len(bs) == 2

        # Each should be a Building instance
        assert all(isinstance(b, Building) for b in bs)

        # No GeoPandas leakage — this is the key architectural check
        for b in bs:
            assert isinstance(b.footprint_enu, np.ndarray)
            assert b.footprint_enu.shape[1] == 2
            assert b.footprint_enu.dtype == np.float64

    def test_height_provenance_tracked(self, tmp_path):
        fake = _fake_gdf_near(DLR_LAT, DLR_LON)
        with patch(
            "atmosphere.retrieval.buildings.ox.features_from_point",
            return_value=fake,
        ):
            bs = fetch_buildings(
                lat=DLR_LAT, lon=DLR_LON, radius_m=100,
                cache_dir=tmp_path, use_cache=False,
            )

        sources = {b.height_source for b in bs}
        assert HeightSource.TAG in sources
        assert HeightSource.LEVELS in sources

        tagged = next(b for b in bs if b.height_source == HeightSource.TAG)
        assert tagged.height_m == pytest.approx(12.0)

        leveled = next(b for b in bs if b.height_source == HeightSource.LEVELS)
        assert leveled.height_m == pytest.approx(10.5)  # 3 levels × 3.5 m

    def test_cache_is_created(self, tmp_path):
        """First call writes cache; second call reads from it without API."""
        fake = _fake_gdf_near(DLR_LAT, DLR_LON)

        # First call — API is hit
        with patch(
            "atmosphere.retrieval.buildings.ox.features_from_point",
            return_value=fake,
        ) as mocked:
            fetch_buildings(
                lat=DLR_LAT, lon=DLR_LON, radius_m=100,
                cache_dir=tmp_path, use_cache=True,
            )
            assert mocked.call_count == 1

        # Check a cache file was written
        cached = list(tmp_path.glob("*.geojson"))
        assert len(cached) == 1

        # Second call — API must NOT be hit
        with patch(
            "atmosphere.retrieval.buildings.ox.features_from_point",
            return_value=fake,
        ) as mocked2:
            fetch_buildings(
                lat=DLR_LAT, lon=DLR_LON, radius_m=100,
                cache_dir=tmp_path, use_cache=True,
            )
            assert mocked2.call_count == 0

    def test_empty_osm_result_returns_empty_list(self, tmp_path):
        """If OSM returns nothing (remote area), we get an empty list, not a crash."""
        empty = gpd.GeoDataFrame(
            {"building": [], "height": [], "building:levels": [], "geometry": []},
            crs="EPSG:4326",
        )
        with patch(
            "atmosphere.retrieval.buildings.ox.features_from_point",
            return_value=empty,
        ):
            bs = fetch_buildings(
                lat=DLR_LAT, lon=DLR_LON, radius_m=100,
                cache_dir=tmp_path, use_cache=False,
            )
        assert bs == []


class TestBuildingProperties:
    def test_has_height(self):
        b_with = Building(
            footprint_enu=np.array([[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]),
            height_m=12.0, height_source=HeightSource.TAG,
            osm_id=1, building_type="yes",
        )
        b_without = Building(
            footprint_enu=np.array([[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]),
            height_m=None, height_source=HeightSource.NONE,
            osm_id=2, building_type="yes",
        )
        assert b_with.has_height is True
        assert b_without.has_height is False

    def test_footprint_area(self):
        # 10m × 10m square = 100 m²
        b = Building(
            footprint_enu=np.array([[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]),
            height_m=None, height_source=HeightSource.NONE,
            osm_id=1, building_type="yes",
        )
        assert b.footprint_area_m2 == pytest.approx(100.0)

    def test_centroid(self):
        # 10m × 10m square, true geometric centroid is (5.0, 5.0)
        b = Building(
            footprint_enu=np.array([[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]),
            height_m=None, height_source=HeightSource.NONE,
            osm_id=1, building_type="yes",
        )
        cx, cy = b.centroid_enu
        assert cx == pytest.approx(5.0)
        assert cy == pytest.approx(5.0)
    
    def test_centroid_l_shape(self):
        # L-shaped polygon: centroid should NOT be at the bounding box center.
        # Shape:
        #   (0,10)─────(5,10)
        #   │            │
        #   │            │
        #   │   (5,5)────(10,5)
        #   │               │
        #   (0,0)──────────(10,0)
        b = Building(
            footprint_enu=np.array([
                [0, 0], [10, 0], [10, 5], [5, 5], [5, 10], [0, 10], [0, 0]
            ]),
            height_m=None, height_source=HeightSource.NONE,
            osm_id=1, building_type="yes",
        )
        cx, cy = b.centroid_enu
        # Hand calculation for this L: total area = 75,
        # centroid ≈ (4.17, 4.17) by symmetry across the diagonal.
        assert cx == pytest.approx(4.17, abs=0.1)
        assert cy == pytest.approx(4.17, abs=0.1)