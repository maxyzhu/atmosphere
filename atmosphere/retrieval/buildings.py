"""
Building geometry retrieval from OpenStreetMap.

This module is the first "data source connector" in the Atmosphere pipeline.
It fetches building footprints from OSM, converts them into our own
coordinate frame (local ENU, in meters), and packages them as a list of
Building dataclass instances.

Design notes:
    - GeoPandas is used internally (osmnx returns it) but does NOT appear in
      any public interface. Downstream modules see only list[Building].
    - Height is handled with provenance: we track whether a height value
      came from an explicit tag, was estimated from floor count, or is
      missing entirely. Downstream modules can decide how to weight each.
    - The query region is a square (in ENU) centered on (lat, lon) with
      half-side radius_m + BUFFER_M. This matches the Mapillary retrieval
      region exactly, so building polygons and street-level images live
      in the same square.
    - A local filesystem cache avoids hitting the Overpass API repeatedly
      during development.
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import geopandas as gpd
import numpy as np
import osmnx as ox

from atmosphere.geo import LocalFrame
from atmosphere.retrieval.mapillary import BUFFER_M

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Public types
# -----------------------------------------------------------------------------


class HeightSource(str, Enum):
    """
    Where a building's height value came from.

    Lets downstream modules make informed decisions: e.g., a renderer
    might show TAG buildings at their tagged height but randomize LEVELS
    buildings within ±2 m. An evaluator might penalize errors more strongly
    on TAG buildings than on NONE buildings.
    """

    TAG = "tag"            # explicit `height=12` or `height=12 m` in OSM
    LEVELS = "levels"      # derived from `building:levels=4` at 3.5 m/level
    NONE = "none"          # no height info available; height is None


@dataclass(frozen=True)
class Building:
    """
    A single building in the local ENU frame.

    Attributes:
        footprint_enu: (N, 2) float64 array of (east, north) points in
            meters, forming a closed polygon. The last point equals the
            first.
        height_m: Height in meters above ground, or None if unknown.
        height_source: Where the height value came from.
        osm_id: OpenStreetMap element ID, for traceability.
        building_type: OSM tag value (e.g., "residential", "commercial").
    """

    footprint_enu: np.ndarray
    height_m: float | None
    height_source: HeightSource
    osm_id: int
    building_type: str

    @property
    def has_height(self) -> bool:
        return self.height_m is not None

    @property
    def centroid_enu(self) -> tuple[float, float]:
        """True area-weighted centroid (handles non-convex shapes)."""
        x = self.footprint_enu[:, 0]
        y = self.footprint_enu[:, 1]
        cross = x * np.roll(y, -1) - np.roll(x, -1) * y
        area = 0.5 * np.sum(cross)
        if abs(area) < 1e-10:
            return float(np.mean(x)), float(np.mean(y))
        cx = np.sum((x + np.roll(x, -1)) * cross) / (6 * area)
        cy = np.sum((y + np.roll(y, -1)) * cross) / (6 * area)
        return float(cx), float(cy)

    @property
    def footprint_area_m2(self) -> float:
        """Polygon area in square meters using the shoelace formula."""
        x = self.footprint_enu[:, 0]
        y = self.footprint_enu[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


# -----------------------------------------------------------------------------
# Height parsing
# -----------------------------------------------------------------------------


def _parse_osm_height(height_tag: object) -> float | None:
    """Parse OSM's free-form `height` tag into a float in meters."""
    if height_tag is None:
        return None
    try:
        if isinstance(height_tag, float) and np.isnan(height_tag):
            return None
    except (TypeError, ValueError):
        pass

    s = str(height_tag).strip().lower()
    if not s or s == "nan":
        return None

    for suffix in (" m", "m", " meters", " metres"):
        if s.endswith(suffix):
            s = s[: -len(suffix)].strip()
            break

    feet_mode = False
    if s.endswith("'") or s.endswith(" ft") or s.endswith("ft"):
        feet_mode = True
        s = s.rstrip("'").rstrip("ft").rstrip().rstrip(" ")

    try:
        value = float(s)
    except ValueError:
        return None

    if feet_mode:
        value *= 0.3048

    if value <= 0 or value > 1000:
        return None

    return value


def _parse_osm_levels(levels_tag: object) -> int | None:
    """Parse OSM's `building:levels` tag into an integer floor count."""
    if levels_tag is None:
        return None
    try:
        if isinstance(levels_tag, float) and np.isnan(levels_tag):
            return None
    except (TypeError, ValueError):
        pass

    s = str(levels_tag).strip()
    if not s or s.lower() == "nan":
        return None
    try:
        return int(round(float(s)))
    except ValueError:
        return None


DEFAULT_METERS_PER_LEVEL = 3.5


def _extract_height(row: dict) -> tuple[float | None, HeightSource]:
    """Apply the height-provenance policy to an OSM feature row."""
    h = _parse_osm_height(row.get("height"))
    if h is not None:
        return h, HeightSource.TAG

    levels = _parse_osm_levels(row.get("building:levels"))
    if levels is not None and levels > 0:
        return levels * DEFAULT_METERS_PER_LEVEL, HeightSource.LEVELS

    return None, HeightSource.NONE


# -----------------------------------------------------------------------------
# Bbox helpers (matches mapillary.py's square + buffer)
# -----------------------------------------------------------------------------


def _square_bbox_with_buffer(
    lat: float, lon: float, half_side_m: float
) -> tuple[float, float, float, float]:
    """
    Same square bbox as the Mapillary retrieval module: ENU-centered on
    (lat, lon) with effective half-side half_side_m + BUFFER_M.

    Returns (west, south, east, north).
    """
    effective_half = half_side_m + BUFFER_M
    lat_deg_per_m = 1.0 / 111_320.0
    lon_deg_per_m = 1.0 / (111_320.0 * math.cos(math.radians(lat)))

    dlat = effective_half * lat_deg_per_m
    dlon = effective_half * lon_deg_per_m

    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


# -----------------------------------------------------------------------------
# Filesystem cache
# -----------------------------------------------------------------------------


def _cache_path(lat: float, lon: float, radius_m: float, cache_dir: Path) -> Path:
    """
    Stable cache filename for a bbox query. The key includes the buffer
    so older radius-based caches won't be reused after the bbox change.
    """
    key = (
        f"osm_buildings_bbox_{lat:.4f}_{lon:.4f}"
        f"_r{int(radius_m)}_b{int(BUFFER_M)}"
    )
    h = hashlib.md5(key.encode()).hexdigest()[:8]
    return cache_dir / f"{key}_{h}.geojson"


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------


def fetch_buildings(
    lat: float,
    lon: float,
    radius_m: float = 150.0,
    *,
    frame: LocalFrame | None = None,
    min_area_m2: float = 20.0,
    cache_dir: Path | str = "data/osm_cache",
    use_cache: bool = True,
) -> list[Building]:
    """
    Fetch building footprints in a square WGS84 bbox, in a local ENU frame.

    The query region is a square centered on (lat, lon) with half-side
    radius_m + BUFFER_M, matching the Mapillary retrieval region.

    Args:
        lat, lon: Query center in WGS84 degrees.
        radius_m: Half-side of the unbuffered square, in meters.
        frame: ENU frame for output geometry. If None, created at
            (lat, lon).
        min_area_m2: Drop footprints smaller than this (filters trash
            bin enclosures, electrical boxes, etc. tagged as buildings).
        cache_dir: Where to store raw OSM GeoJSON between runs.
        use_cache: If False, always re-fetch from Overpass.

    Returns:
        list[Building] with footprints in the ENU frame. Order arbitrary.
    """
    if frame is None:
        frame = LocalFrame(lat0=lat, lon0=lon)

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = _cache_path(lat, lon, radius_m, cache_dir)

    bbox = _square_bbox_with_buffer(lat, lon, radius_m)
    west, south, east, north = bbox

    # --- 1. Fetch (or load from cache) ---
    if use_cache and cache_file.exists():
        logger.info("Loading OSM buildings from cache: %s", cache_file.name)
        gdf = gpd.read_file(cache_file)
    else:
        logger.info(
            "Fetching OSM buildings from Overpass API: bbox=%s "
            "(lat=%.4f, lon=%.4f, r=%d m, buffer=%d m)",
            bbox, lat, lon, int(radius_m), int(BUFFER_M),
        )
        # osmnx 2.x: bbox=(west, south, east, north)
        # osmnx 1.x: features_from_bbox(north, south, east, west, ...)
        try:
            gdf = ox.features_from_bbox(
                bbox=(west, south, east, north),
                tags={"building": True},
            )
        except TypeError:
            # Fall back to legacy positional signature for osmnx 1.x.
            gdf = ox.features_from_bbox(
                north, south, east, west, tags={"building": True},
            )

        if len(gdf) == 0:
            logger.warning(
                "OSM returned no buildings for bbox %s. Sparse coverage?",
                bbox,
            )
        else:
            gdf.to_file(cache_file, driver="GeoJSON")
            logger.info("Cached %d features to %s", len(gdf), cache_file.name)

    # --- 2. Filter to polygon geometries ---
    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()

    # --- 3. Convert each row to a Building in the ENU frame ---
    buildings: list[Building] = []
    for idx, row in gdf.iterrows():
        geom = row.geometry
        # MultiPolygon: take the largest piece (rare; happens at messy
        # edges of complex buildings).
        if geom.geom_type == "MultiPolygon":
            geom = max(geom.geoms, key=lambda p: p.area)

        # Shapely's Polygon.exterior.coords are (lon, lat).
        exterior_lonlat = np.asarray(geom.exterior.coords)
        lons = exterior_lonlat[:, 0]
        lats = exterior_lonlat[:, 1]

        east_arr, north_arr, _ = frame.wgs84_to_enu(lats, lons)
        footprint_enu = np.stack([east_arr, north_arr], axis=-1).astype(np.float64)

        # Cheap shoelace area check before committing to a Building.
        area = 0.5 * abs(
            np.dot(footprint_enu[:, 0], np.roll(footprint_enu[:, 1], -1))
            - np.dot(footprint_enu[:, 1], np.roll(footprint_enu[:, 0], -1))
        )
        if area < min_area_m2:
            continue

        row_dict = row.to_dict()
        height_m, height_source = _extract_height(row_dict)

        # osmnx indexes features by a MultiIndex (element_type, osmid).
        osm_id = int(idx[1]) if isinstance(idx, tuple) else int(idx)

        building_type = str(row_dict.get("building", "yes"))

        buildings.append(Building(
            footprint_enu=footprint_enu,
            height_m=height_m,
            height_source=height_source,
            osm_id=osm_id,
            building_type=building_type,
        ))

    logger.info(
        "Returned %d buildings (%d with height, %d without)",
        len(buildings),
        sum(1 for b in buildings if b.has_height),
        sum(1 for b in buildings if not b.has_height),
    )

    return buildings