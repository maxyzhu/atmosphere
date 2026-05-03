"""
Street network retrieval from OpenStreetMap.

Parallels atmosphere.retrieval.buildings: fetches road geometry from OSM
via osmnx, projects it into the local ENU frame, and packages each
intersection-to-intersection segment as a flat StreetSegment instance.

Design notes:
    - osmnx returns a networkx MultiDiGraph; that is an internal
      implementation detail. Downstream sees only list[StreetSegment].
    - Topology (which segment connects to which) is deliberately discarded
      at this layer. Phase 0 does not need it; if Phase 1 segment-aware
      sampling needs it, we promote to a StreetNetwork dataclass that
      keeps node IDs at segment endpoints. Until then, simpler is better.
    - The query region is the same square+buffer used by buildings.py
      and mapillary.py, so all three layers live in identical coordinates.
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import osmnx as ox

from atmosphere.geo import LocalFrame
from atmosphere.retrieval.mapillary import BUFFER_M

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Public types
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class StreetSegment:
    """
    A single street segment, intersection to intersection, in ENU.

    Attributes:
        polyline_enu: (N, 2) float64 array of (east, north) points in
            meters. N >= 2; intermediate points represent curve geometry
            between two intersections, not extra intersections.
        osm_id: OSM way ID. Multiple StreetSegments can share an osm_id
            when osmnx splits a long way at intersections.
        highway_type: OSM `highway` tag value. Common values:
            "primary", "secondary", "tertiary", "residential",
            "service", "footway", "cycleway", "path".
        name: Street name from OSM `name` tag, or None if unnamed.
        oneway: True if traffic flows in one direction only.
    """

    polyline_enu: np.ndarray
    osm_id: int
    highway_type: str
    name: str | None
    oneway: bool

    @property
    def length_m(self) -> float:
        """Polyline length in meters (sum of segment edge lengths)."""
        diffs = np.diff(self.polyline_enu, axis=0)
        return float(np.sum(np.sqrt((diffs ** 2).sum(axis=1))))

    @property
    def midpoint_enu(self) -> tuple[float, float]:
        """Geometric midpoint of the polyline (vertex closest to half-length)."""
        # Cheap approximation: arithmetic mean of vertices.
        # Good enough for legend / labeling; a true arc-length midpoint
        # is overkill for Phase 0.
        return (
            float(np.mean(self.polyline_enu[:, 0])),
            float(np.mean(self.polyline_enu[:, 1])),
        )


# -----------------------------------------------------------------------------
# Bbox + cache helpers (mirror buildings.py)
# -----------------------------------------------------------------------------


def _square_bbox_with_buffer(
    lat: float, lon: float, half_side_m: float
) -> tuple[float, float, float, float]:
    """Same square bbox used by buildings and mapillary retrieval."""
    effective_half = half_side_m + BUFFER_M
    lat_deg_per_m = 1.0 / 111_320.0
    lon_deg_per_m = 1.0 / (111_320.0 * math.cos(math.radians(lat)))
    dlat = effective_half * lat_deg_per_m
    dlon = effective_half * lon_deg_per_m
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def _cache_path(
    lat: float, lon: float, radius_m: float,
    network_type: str, cache_dir: Path,
) -> Path:
    """Cache key includes network_type so 'drive' and 'walk' don't collide."""
    key = (
        f"osm_streets_bbox_{lat:.4f}_{lon:.4f}"
        f"_r{int(radius_m)}_b{int(BUFFER_M)}_{network_type}"
    )
    h = hashlib.md5(key.encode()).hexdigest()[:8]
    return cache_dir / f"{key}_{h}.graphml"


# -----------------------------------------------------------------------------
# Tag parsing helpers
# -----------------------------------------------------------------------------


def _coerce_oneway(val: object) -> bool:
    """OSM oneway tag is often "yes"/"no"/True/False/None/"-1"."""
    if val is None:
        return False
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    return s in ("yes", "true", "1", "-1")  # -1 = reverse oneway, still oneway


def _coerce_str(val: object, default: str = "") -> str:
    """OSM tags occasionally come as lists when a way has multiple values."""
    if val is None:
        return default
    if isinstance(val, list):
        # Take the first non-empty value
        for item in val:
            if item:
                return str(item)
        return default
    return str(val)


def _coerce_optional_str(val: object) -> str | None:
    """Like _coerce_str but returns None instead of empty string."""
    s = _coerce_str(val)
    return s if s else None


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------


def fetch_streets(
    lat: float,
    lon: float,
    radius_m: float = 150.0,
    *,
    frame: LocalFrame | None = None,
    network_type: str = "drive",
    cache_dir: Path | str = "data/osm_cache",
    use_cache: bool = True,
) -> list[StreetSegment]:
    """
    Fetch street segments in the same square+buffer region as buildings
    and Mapillary images.

    Args:
        lat, lon: Query center in WGS84 degrees.
        radius_m: Half-side of the unbuffered square, in meters.
        frame: ENU frame for output geometry. If None, created at
            (lat, lon).
        network_type: osmnx network filter. "drive" matches Mapillary's
            primarily-vehicular capture distribution; "all" includes
            footpaths and service roads if you need fuller coverage.
        cache_dir: Where to store the cached OSM graph between runs.
        use_cache: If False, always re-fetch from Overpass.

    Returns:
        list[StreetSegment] in arbitrary order.
    """
    if frame is None:
        frame = LocalFrame(lat0=lat, lon0=lon)

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = _cache_path(lat, lon, radius_m, network_type, cache_dir)

    bbox = _square_bbox_with_buffer(lat, lon, radius_m)
    west, south, east, north = bbox

    # --- 1. Fetch graph (or load from cache) ---
    if use_cache and cache_file.exists():
        logger.info("Loading OSM streets from cache: %s", cache_file.name)
        G = ox.load_graphml(cache_file)
    else:
        logger.info(
            "Fetching OSM streets from Overpass API: bbox=%s, network=%s",
            bbox, network_type,
        )
        # osmnx 2.x: bbox=(west, south, east, north)
        # osmnx 1.x: graph_from_bbox(north, south, east, west, ...)
        try:
            G = ox.graph.graph_from_bbox(
                bbox=(west, south, east, north),
                network_type=network_type,
                simplify=True,
            )
        except TypeError:
            G = ox.graph.graph_from_bbox(
                north, south, east, west,
                network_type=network_type,
                simplify=True,
            )

        if G.number_of_edges() == 0:
            logger.warning(
                "OSM returned no streets for bbox %s. Sparse coverage?",
                bbox,
            )
        else:
            ox.save_graphml(G, cache_file)
            logger.info(
                "Cached %d edges to %s",
                G.number_of_edges(), cache_file.name,
            )

    # --- 2. Convert each edge into a StreetSegment in ENU ---
    segments: list[StreetSegment] = []
    seen_keys: set[tuple[int, int, int]] = set()  # (u, v, key) dedup

    for u, v, key, data in G.edges(keys=True, data=True):
        # MultiDiGraph: each undirected street appears as two directed
        # edges (u→v and v→u) sharing osmid + geometry. Dedupe by
        # canonicalizing (u, v) order.
        canon = (min(u, v), max(u, v), key)
        if canon in seen_keys:
            continue
        seen_keys.add(canon)

        # Geometry: when simplify=True, osmnx attaches a shapely LineString
        # to edges that span multiple OSM nodes. Edges between adjacent
        # intersections may have no geometry attribute — we synthesize one
        # from the endpoint node coords.
        geom = data.get("geometry")
        if geom is not None:
            lonlat = np.asarray(geom.coords)  # (N, 2), (lon, lat)
        else:
            u_data = G.nodes[u]
            v_data = G.nodes[v]
            lonlat = np.array([
                [u_data["x"], u_data["y"]],  # osmnx: x=lon, y=lat
                [v_data["x"], v_data["y"]],
            ])

        if len(lonlat) < 2:
            continue

        lons = lonlat[:, 0]
        lats = lonlat[:, 1]
        east_arr, north_arr, _ = frame.wgs84_to_enu(lats, lons)
        polyline_enu = np.stack([east_arr, north_arr], axis=-1).astype(np.float64)

        # OSM way ID. Some edges have a list of osmids (when osmnx merges
        # consecutive ways during simplification); take the first.
        osmid_raw = data.get("osmid", 0)
        if isinstance(osmid_raw, list):
            osm_id = int(osmid_raw[0]) if osmid_raw else 0
        else:
            osm_id = int(osmid_raw)

        highway_type = _coerce_str(data.get("highway"), default="unclassified")
        name = _coerce_optional_str(data.get("name"))
        oneway = _coerce_oneway(data.get("oneway"))

        segments.append(StreetSegment(
            polyline_enu=polyline_enu,
            osm_id=osm_id,
            highway_type=highway_type,
            name=name,
            oneway=oneway,
        ))

    logger.info(
        "Returned %d street segments (%d named, total length %.0f m)",
        len(segments),
        sum(1 for s in segments if s.name is not None),
        sum(s.length_m for s in segments),
    )

    return segments