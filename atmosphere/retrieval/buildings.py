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
    - A local filesystem cache avoids hitting the Overpass API repeatedly
      during development. The Overpass API is free but rate-limited, and
      we will hit those limits fast during iteration.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

import geopandas as gpd
import numpy as np
import osmnx as ox
from shapely.geometry import Polygon

from atmosphere.geo import LocalFrame

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Public types (these, and only these, are what downstream modules import)
# -----------------------------------------------------------------------------


class HeightSource(str, Enum):
    """
    Where a building's height value came from.

    This lets downstream modules make informed decisions: e.g., a renderer
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
        footprint_enu: (N, 2) float64 array of (east, north) points in meters,
            forming a closed polygon. The last point equals the first.
        height_m: Height in meters above ground, or None if unknown.
        height_source: Where the height value came from (see HeightSource).
        osm_id: The OpenStreetMap element ID, for traceability.
        building_type: OSM tag value (e.g., "residential", "commercial",
            "yes"). Useful for styling / filtering.
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
        """
        True geometric centroid (area-weighted) in ENU meters.

        Uses the shoelace formula, correct for any simple polygon including
        non-convex shapes (L-shaped, U-shaped buildings). For a rectangle,
        equals the center; for irregular shapes, equals the center of mass
        if the polygon were a uniform plate.
        """
        x = self.footprint_enu[:, 0]
        y = self.footprint_enu[:, 1]
        cross = x * np.roll(y, -1) - np.roll(x, -1) * y
        area = 0.5 * np.sum(cross)
        if abs(area) < 1e-10:
            # Degenerate polygon — fall back to vertex mean
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
# Height parsing (OSM's tags are messy free-form strings)
# -----------------------------------------------------------------------------


def _parse_osm_height(height_tag: object) -> float | None:
    """
    Parse OSM's `height` tag into a float in meters.

    OSM is crowd-sourced, so the tag shows up in many forms:
        "12", "12.5", "12 m", "12m", "40'" (feet!), "~12", ""
    We handle the common cases and return None on anything weird, rather
    than guessing.

    Returns None for NaN, None, empty string, or unparseable values.
    """
    if height_tag is None:
        return None
    # GeoPandas returns NaN for missing cells; check before str conversion.
    try:
        if isinstance(height_tag, float) and np.isnan(height_tag):
            return None
    except (TypeError, ValueError):
        pass

    s = str(height_tag).strip().lower()
    if not s or s == "nan":
        return None

    # Strip common suffixes
    for suffix in (" m", "m", " meters", " metres"):
        if s.endswith(suffix):
            s = s[: -len(suffix)].strip()
            break

    # Feet are rare but real in US OSM data; convert if we see a quote mark.
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

    # Sanity bound: OSM has buildings up to Burj Khalifa (828 m).
    # Anything outside 0-1000 m is almost certainly a data error.
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
        # Some entries say "4.5" for mezzanines; round.
        return int(round(float(s)))
    except ValueError:
        return None


# Meters per floor. US buildings: ~3.0-4.0 m depending on era and type.
# 3.5 is a common middle estimate used in urban-science literature.
DEFAULT_METERS_PER_LEVEL = 3.5


def _extract_height(row: dict) -> tuple[float | None, HeightSource]:
    """
    Apply the height-provenance policy to an OSM feature row.

    Priority:
        1. Explicit `height` tag wins if parseable
        2. Fall back to `building:levels` × 3.5 m
        3. Otherwise None (caller may supply a default downstream)
    """
    h = _parse_osm_height(row.get("height"))
    if h is not None:
        return h, HeightSource.TAG

    levels = _parse_osm_levels(row.get("building:levels"))
    if levels is not None and levels > 0:
        return levels * DEFAULT_METERS_PER_LEVEL, HeightSource.LEVELS

    return None, HeightSource.NONE


# -----------------------------------------------------------------------------
# Filesystem cache
# -----------------------------------------------------------------------------


def _cache_path(lat: float, lon: float, radius_m: float, cache_dir: Path) -> Path:
    """
    Build a stable cache filename for a given query.

    Coordinate precision of 4 decimals ≈ 11 m — we collapse cache keys at
    this granularity on purpose so near-identical queries reuse results.
    """
    key = f"osm_buildings_{lat:.4f}_{lon:.4f}_r{int(radius_m)}"
    # Hash is paranoia against weird chars; redundant here but harmless.
    h = hashlib.md5(key.encode()).hexdigest()[:8]
    return cache_dir / f"{key}_{h}.geojson"


# -----------------------------------------------------------------------------
# The public entry point
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
    Fetch building footprints near a WGS84 point, in a local ENU frame.

    Args:
        lat: Query center latitude in degrees.
        lon: Query center longitude in degrees.
        radius_m: Search radius in meters. Default 150 m — roughly two
            Seattle city blocks, a good balance between visual richness
            and manageable data size.
        frame: The ENU frame to express building geometry in. If None, a
            frame is created at the query point (most common case).
        min_area_m2: Drop footprints smaller than this. OSM often tags
            trash bin enclosures, electrical boxes, etc. as small
            buildings; 20 m² filters these without losing real structures.
        cache_dir: Where to store raw OSM GeoJSON between runs.
        use_cache: If False, always re-fetch from Overpass.

    Returns:
        A list of Building objects with footprints in the ENU frame.
        Order is arbitrary (OSM id order); do not rely on it.
    """
    if frame is None:
        frame = LocalFrame(lat0=lat, lon0=lon)

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = _cache_path(lat, lon, radius_m, cache_dir)

    # --- 1. Fetch (or load from cache) ---
    if use_cache and cache_file.exists():
        logger.info("Loading OSM buildings from cache: %s", cache_file.name)
        gdf = gpd.read_file(cache_file)
    else:
        logger.info(
            "Fetching OSM buildings from Overpass API: "
            "(lat=%.4f, lon=%.4f, radius=%d m)",
            lat, lon, int(radius_m),
        )
        gdf = ox.features_from_point(
            center_point=(lat, lon),
            tags={"building": True},
            dist=radius_m,
        )
        if len(gdf) == 0:
            logger.warning(
                "OSM returned no buildings at (%.4f, %.4f) within %d m. "
                "This area may have sparse OSM coverage.",
                lat, lon, int(radius_m),
            )
        else:
            # Cache the raw result for next time.
            gdf.to_file(cache_file, driver="GeoJSON")
            logger.info("Cached %d features to %s", len(gdf), cache_file.name)

    # --- 2. Filter: only polygon geometries, only actual buildings ---
    # osmnx sometimes returns nodes (Points) for small buildings tagged as
    # single points — skip those, we can only render polygons.
    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()

    # --- 3. Convert each row to a Building in the ENU frame ---
    buildings: list[Building] = []
    for idx, row in gdf.iterrows():
        geom = row.geometry
        # MultiPolygon? Take the largest piece. This is rare but happens
        # for buildings straddling complicated edges.
        if geom.geom_type == "MultiPolygon":
            geom = max(geom.geoms, key=lambda p: p.area)

        # Shapely Polygon has an `.exterior` ring of (lon, lat) coords.
        # (osmnx convention: x=lon, y=lat. Not ENU. Careful.)
        exterior_lonlat = np.asarray(geom.exterior.coords)
        lons = exterior_lonlat[:, 0]
        lats = exterior_lonlat[:, 1]

        # Project to ENU using the Day 1 frame.
        east, north, _ = frame.wgs84_to_enu(lats, lons)
        footprint_enu = np.stack([east, north], axis=-1).astype(np.float64)

        # Skip too-small footprints (noise filter).
        # Cheap shoelace area check before committing to a Building.
        area = 0.5 * abs(
            np.dot(footprint_enu[:, 0], np.roll(footprint_enu[:, 1], -1))
            - np.dot(footprint_enu[:, 1], np.roll(footprint_enu[:, 0], -1))
        )
        if area < min_area_m2:
            continue

        # Extract height with provenance.
        row_dict = row.to_dict()
        height_m, height_source = _extract_height(row_dict)

        # OSM id: osmnx indexes features by a MultiIndex (element_type, osmid).
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