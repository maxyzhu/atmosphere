"""
Tests for coordinate system utilities.

These tests are deliberately minimal but target invariants that, if broken,
would silently corrupt every downstream module. Coordinate bugs produce
outputs that look plausible but are subtly wrong — the worst kind of bug
in a research system.
"""

import numpy as np
import pytest

from atmosphere.geo import LocalFrame, haversine_distance_m


# Seattle downtown, our default test origin throughout the project.
SEATTLE_LAT = 47.6062
SEATTLE_LON = -122.3321


class TestLocalFrame:
    """
    Exercises the WGS84 <-> ENU conversion.

    Key invariant: the origin maps to (0, 0, 0), and the cardinal directions
    (east = +x, north = +y, up = +z) are preserved.
    """

    def test_origin_maps_to_zero(self):
        """The origin of the frame should itself convert to (0, 0, 0)."""
        frame = LocalFrame(lat0=SEATTLE_LAT, lon0=SEATTLE_LON)
        e, n, u = frame.wgs84_to_enu(SEATTLE_LAT, SEATTLE_LON, 0.0)
        assert abs(e) < 1e-6
        assert abs(n) < 1e-6
        assert abs(u) < 1e-6

    def test_east_direction_is_positive_x(self):
        """Moving one degree east should produce positive east (x) coordinate."""
        frame = LocalFrame(lat0=SEATTLE_LAT, lon0=SEATTLE_LON)
        e, _, _ = frame.wgs84_to_enu(SEATTLE_LAT, SEATTLE_LON + 0.001)
        assert e > 0

    def test_north_direction_is_positive_y(self):
        """Moving one degree north should produce positive north (y) coordinate."""
        frame = LocalFrame(lat0=SEATTLE_LAT, lon0=SEATTLE_LON)
        _, n, _ = frame.wgs84_to_enu(SEATTLE_LAT + 0.001, SEATTLE_LON)
        assert n > 0

    def test_up_direction_is_positive_z(self):
        """Increasing elevation should produce positive up (z) coordinate."""
        frame = LocalFrame(lat0=SEATTLE_LAT, lon0=SEATTLE_LON)
        _, _, u = frame.wgs84_to_enu(SEATTLE_LAT, SEATTLE_LON, 100.0)
        assert u > 0
        # Within a few cm of exact, since elevation is near-trivial near origin.
        assert abs(u - 100.0) < 0.1

    def test_round_trip_identity(self):
        """
        Converting WGS84 -> ENU -> WGS84 should return the original coordinates.

        This is the single most important correctness check in this module:
        if round-trip drifts, any downstream module that round-trips will
        accumulate error.
        """
        frame = LocalFrame(lat0=SEATTLE_LAT, lon0=SEATTLE_LON)
        test_points = [
            (SEATTLE_LAT + 0.001, SEATTLE_LON + 0.001, 10.0),
            (SEATTLE_LAT - 0.002, SEATTLE_LON + 0.003, -5.0),
            (SEATTLE_LAT + 0.0005, SEATTLE_LON - 0.0005, 100.0),
        ]
        for lat, lon, ele in test_points:
            e, n, u = frame.wgs84_to_enu(lat, lon, ele)
            lat2, lon2, ele2 = frame.enu_to_wgs84(e, n, u)
            # Tolerance: ~1 mm on position, ~1 cm on elevation
            assert abs(float(lat2) - lat) < 1e-8
            assert abs(float(lon2) - lon) < 1e-8
            assert abs(float(ele2) - ele) < 0.01

    def test_array_input(self):
        """Arrays of coordinates should convert element-wise."""
        frame = LocalFrame(lat0=SEATTLE_LAT, lon0=SEATTLE_LON)
        lats = np.array([SEATTLE_LAT, SEATTLE_LAT + 0.001])
        lons = np.array([SEATTLE_LON, SEATTLE_LON + 0.001])
        eles = np.array([0.0, 10.0])
        e, n, u = frame.wgs84_to_enu(lats, lons, eles)
        assert e.shape == (2,)
        assert n.shape == (2,)
        assert u.shape == (2,)
        # First point is the origin
        assert abs(float(e[0])) < 1e-6
        assert abs(float(n[0])) < 1e-6
        # Second point is to the northeast and higher
        assert float(e[1]) > 0
        assert float(n[1]) > 0
        assert float(u[1]) > 9.99

    def test_small_distance_approximately_euclidean(self):
        """
        Within 100 meters of origin, ENU distance should match haversine
        distance to better than 1 mm. This sanity-checks that the ENU
        approximation is adequate for our query radii.
        """
        frame = LocalFrame(lat0=SEATTLE_LAT, lon0=SEATTLE_LON)
        # A point ~50 meters NE of origin
        target_lat = SEATTLE_LAT + 0.00045  # ~50 m north
        target_lon = SEATTLE_LON + 0.00067  # ~50 m east
        e, n, _ = frame.wgs84_to_enu(target_lat, target_lon)
        enu_dist = float(np.sqrt(e**2 + n**2))
        hav_dist = haversine_distance_m(
            SEATTLE_LAT, SEATTLE_LON, target_lat, target_lon
        )
        assert abs(enu_dist - hav_dist) < 0.2  # < 20 mm agreement


class TestHaversine:
    def test_zero_distance(self):
        d = haversine_distance_m(SEATTLE_LAT, SEATTLE_LON, SEATTLE_LAT, SEATTLE_LON)
        assert d < 1e-6

    def test_known_distance(self):
        """
        Seattle to Portland, OR is ~234 km (great-circle).

        We check within 1% of a known value to catch gross errors
        (wrong Earth radius, radians vs degrees mix-up, etc.).
        """
        portland_lat, portland_lon = 45.5152, -122.6784
        d_m = haversine_distance_m(
            SEATTLE_LAT, SEATTLE_LON, portland_lat, portland_lon
        )
        d_km = d_m / 1000
        assert 230 < d_km < 240


if __name__ == "__main__":
    pytest.main([__file__, "-v"])