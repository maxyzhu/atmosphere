"""
Geographic coordinate system utilities.

This module is the foundation of everything downstream. Every other module
assumes that geometry lives in a local ENU (East-North-Up) frame, measured
in meters, with the origin at a query point. This module defines the
transformation between that frame and the WGS84 (lat, lon, elevation) frame
in which all external data (OSM, GPS, Mapillary) arrives.

Design notes:
    - We use pymap3d for the actual math. The WGS84 ellipsoid parameters
      are baked into it, and implementing this ourselves would be a
      source of subtle bugs.
    - Valid range for ENU approximation is roughly 10 km from the origin.
      At Phase 0 query radii (80-200 m), errors are sub-millimeter.
    - All ENU coordinates are float64. Float32 is fine for geometry stored
      long-term, but loses precision during the WGS84 <-> ENU conversion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pymap3d as pm


@dataclass(frozen=True)
class LocalFrame:
    """
    A local East-North-Up coordinate frame anchored at a WGS84 point.

    Within roughly 10 km of the origin, the ENU frame is a good Euclidean
    approximation of the curved Earth surface. All downstream modules
    (rendering, alignment, evaluation) operate in this frame, in meters.

    Attributes:
        lat0: Origin latitude in degrees (WGS84)
        lon0: Origin longitude in degrees (WGS84)
        ele0: Origin ellipsoidal height in meters (default 0)

    Example:
        >>> frame = LocalFrame(lat0=47.6062, lon0=-122.3321)  # Seattle
        >>> x, y, z = frame.wgs84_to_enu(47.6070, -122.3330, 10.0)
        >>> # x is meters east of origin, y meters north, z meters up
    """

    lat0: float
    lon0: float
    ele0: float = 0.0

    def wgs84_to_enu(
        self,
        lat: float | np.ndarray,
        lon: float | np.ndarray,
        ele: float | np.ndarray = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert WGS84 geographic coordinates to local ENU meters.

        Accepts scalars or arrays (arrays must broadcast together).

        Args:
            lat: Target latitude in degrees
            lon: Target longitude in degrees
            ele: Target ellipsoidal height in meters

        Returns:
            (east, north, up) tuple, each in meters relative to origin.
        """
        east, north, up = pm.geodetic2enu(
            lat, lon, ele,
            self.lat0, self.lon0, self.ele0,
        )
        return np.asarray(east), np.asarray(north), np.asarray(up)

    def enu_to_wgs84(
        self,
        east: float | np.ndarray,
        north: float | np.ndarray,
        up: float | np.ndarray = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert local ENU meters back to WGS84 geographic coordinates.

        The inverse of wgs84_to_enu. Used when we need to hand a refined
        ENU position back to something that speaks lat/lon (e.g., writing
        a refined Mapillary pose).

        Args:
            east: East offset in meters
            north: North offset in meters
            up: Up offset in meters

        Returns:
            (lat, lon, ele) tuple.
        """
        lat, lon, ele = pm.enu2geodetic(
            east, north, up,
            self.lat0, self.lon0, self.ele0,
        )
        return np.asarray(lat), np.asarray(lon), np.asarray(ele)


def haversine_distance_m(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
) -> float:
    """
    Great-circle distance between two WGS84 points, in meters.

    This is a spherical-Earth approximation (~0.5% error worst case).
    For ENU-frame distances within a query area, prefer plain Euclidean
    distance on the ENU coordinates. This function exists for the
    coarser "is this Mapillary image within R meters of my query"
    pre-filter, where ENU conversion hasn't happened yet.

    Args:
        lat1, lon1: First point in degrees
        lat2, lon2: Second point in degrees

    Returns:
        Distance in meters.
    """
    r_earth_m = 6371008.8  # mean Earth radius, meters
    lat1_r = np.radians(lat1)
    lat2_r = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    return float(r_earth_m * c)