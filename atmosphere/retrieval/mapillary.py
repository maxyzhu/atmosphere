"""
Mapillary street-level imagery retrieval (vector tile path).

Pipeline (parallels atmosphere.retrieval.buildings):

    fetch_mapillary_images(lat, lon, radius_m)
        1. Build a square bbox around (lat, lon) with side 2*(radius_m + BUFFER_M)
        2. Determine which mly1_computed_public zoom-14 tiles cover that bbox
        3. Fetch each tile (cached on disk by (z, x, y))
        4. Decode the protobuf, extract the "image" layer features
        5. Parse into MapillaryImage instances in the local ENU frame
        6. Apply greedy farthest-point sampling, seeded from the image
           closest to the bbox center, optimizing joint (position, compass)
           diversity, until target_count is reached
        7. Optionally call Graph API per selected image for thumb URL,
           then download the thumbnail to disk cache

Design notes:
    - Vector tiles are heavily cacheable: the unit of caching is a single
      tile (z, x, y), not the full query. Two queries in the same area
      will reuse the same tiles entirely.
    - mly1_computed_public uses SfM-corrected geometry, so positions are
      more accurate than mly1_public. Whether `compass_angle` is also
      SfM-corrected in this tileset is not explicit in the docs;
      empirically should be verified by diffing same-id images across
      both tilesets. For now we treat the tile-supplied angle as
      authoritative for diversity sampling.
    - The Graph API is only touched when `download_thumbnails=True`,
      because thumb URLs are signed/expiring and cannot be cached.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path

import mapbox_vector_tile
import mercantile
import numpy as np
import requests

from atmosphere.config import get_mapillary_token
from atmosphere.geo import LocalFrame

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Public types
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class MapillaryImage:
    """
    A single street-level image from Mapillary.

    Attributes:
        mapillary_id: Mapillary's globally unique image ID.
        position_enu: (east, north) in meters, in the caller's local frame.
        compass_angle_deg: Camera heading, degrees clockwise from north
                           (0 = N, 90 = E). None if missing.
        is_pano: True if this is a 360° equirectangular panorama,
                 False if a regular perspective image. Read from the
                 vector tile's `is_pano` property. Downstream world
                 model conditioning (WorldMirror i2s mode) prefers
                 panos, so this is the priority signal for selection.
        thumb_url: Mapillary CDN URL (signed, may expire). Empty string
                   when thumbnail fetching was skipped.
        thumb_path: Local filesystem path to the cached thumbnail, or None
                    if download was skipped or failed.
        ground_elevation_m: Terrain elevation at this image's position,
                            in meters above the local ENU z=0 plane.
                            None in Phase 0 (DEM is Phase 1); reserved
                            so adding terrain in Phase 1 doesn't require
                            an architectural rewrite.
    """

    mapillary_id: str
    position_enu: tuple[float, float]
    compass_angle_deg: float | None
    is_pano: bool
    thumb_url: str
    thumb_path: Path | None
    ground_elevation_m: float | None = None

    @property
    def has_compass(self) -> bool:
        return self.compass_angle_deg is not None


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

# Buffer applied beyond the requested radius. The output square is
# slightly larger than the user asked for, so generated imagery has a bit
# of margin and edge artifacts don't bite into the region of interest.
BUFFER_M: float = 20.0

# Density target: 20 images per 100m × 100m = 0.002 / m².
IMAGES_PER_M2: float = 20.0 / (100.0 * 100.0)

# Mapillary vector tiles are only served at zoom 14 for the image layer.
TILE_ZOOM: int = 14

TILE_URL_TEMPLATE: str = (
    "https://tiles.mapillary.com/maps/vtp/mly1_computed_public/2/"
    "{z}/{x}/{y}?access_token={token}"
)

# Vector tile internal coordinate extent (Mapbox spec default).
TILE_EXTENT: int = 4096

# FPS metric weighting. 20 m spatial ≈ 45° compass, so β/α ≈ 0.44.
FPS_SPATIAL_WEIGHT: float = 1.0
FPS_COMPASS_WEIGHT: float = 0.44


# -----------------------------------------------------------------------------
# Bbox math
# -----------------------------------------------------------------------------


def _square_bbox_with_buffer(
    lat: float, lon: float, half_side_m: float
) -> tuple[float, float, float, float]:
    """
    Build a square WGS84 bbox centered on (lat, lon) with half-side
    half_side_m + BUFFER_M.

    Returns (west, south, east, north). The square is "square in ENU";
    in lon/lat it is slightly stretched (longer in lon at high latitude),
    but the discrepancy is below 0.5% in the latitudes we care about
    and the result is converted back to ENU before any geometric work.
    """
    effective_half = half_side_m + BUFFER_M
    lat_deg_per_m = 1.0 / 111_320.0
    lon_deg_per_m = 1.0 / (111_320.0 * math.cos(math.radians(lat)))

    dlat = effective_half * lat_deg_per_m
    dlon = effective_half * lon_deg_per_m

    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def _density_target(half_side_m: float) -> int:
    """Target image count from the area-density formula (post-buffer)."""
    side = 2.0 * (half_side_m + BUFFER_M)
    return max(1, int(round(side * side * IMAGES_PER_M2)))


# -----------------------------------------------------------------------------
# Vector tile fetch + decode
# -----------------------------------------------------------------------------


def _tiles_covering_bbox(
    bbox: tuple[float, float, float, float],
    zoom: int = TILE_ZOOM,
) -> list[mercantile.Tile]:
    """All zoom-z tiles that intersect the given (W, S, E, N) bbox."""
    west, south, east, north = bbox
    return list(mercantile.tiles(west, south, east, north, zooms=zoom))


def _fetch_tile_bytes(
    tile: mercantile.Tile,
    cache_dir: Path,
    use_cache: bool,
    timeout_s: float = 15.0,
) -> bytes | None:
    """
    Fetch a single Mapillary vector tile, with on-disk caching by (z, x, y).

    Empty tiles (no Mapillary coverage) get a small empty placeholder
    cached so we don't re-hit the API for known-empty regions.
    Returns None if the tile is empty, otherwise the raw pbf bytes.
    """
    cache_path = cache_dir / f"{tile.z}_{tile.x}_{tile.y}.pbf"
    if use_cache and cache_path.exists():
        data = cache_path.read_bytes()
        return data if data else None

    url = TILE_URL_TEMPLATE.format(
        z=tile.z, x=tile.x, y=tile.y, token=get_mapillary_token(),
    )
    try:
        resp = requests.get(url, timeout=timeout_s)
        if resp.status_code == 404:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(b"")  # cache the miss
            return None
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("Tile fetch failed (%d/%d/%d): %s",
                       tile.z, tile.x, tile.y, exc)
        return None

    data = resp.content
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(data)
    return data if data else None


def _decode_image_features(
    pbf_bytes: bytes,
    tile: mercantile.Tile,
) -> list[dict]:
    """
    Decode a Mapillary vector tile and return the image layer's features
    with WGS84-projected coordinates.

    Vector tile coordinates are tile-local (0..TILE_EXTENT). We project
    them back to lon/lat using the tile's WGS84 bounds.
    """
    try:
        decoded = mapbox_vector_tile.decode(pbf_bytes)
    except Exception as exc:
        logger.warning("Tile decode failed (%d/%d/%d): %s",
                       tile.z, tile.x, tile.y, exc)
        return []

    image_layer = decoded.get("image")
    if not image_layer:
        return []

    bounds = mercantile.bounds(tile)  # west, south, east, north
    extent = image_layer.get("extent", TILE_EXTENT)
    lon_per_unit = (bounds.east - bounds.west) / extent
    lat_per_unit = (bounds.north - bounds.south) / extent

    features = []
    for feat in image_layer.get("features", []):
        geom = feat.get("geometry", {})
        if geom.get("type") != "Point":
            continue
        coords = geom.get("coordinates")
        if not coords or len(coords) < 2:
            continue

        x_local, y_local = coords[0], coords[1]
        # mapbox_vector_tile returns y in the standard math direction
        # (origin at SW), matching mercantile.bounds(). No flip needed.
        lon = bounds.west + x_local * lon_per_unit
        lat = bounds.south + y_local * lat_per_unit

        props = feat.get("properties", {})
        features.append({
            "id": str(props.get("id", "")),
            "lon": lon,
            "lat": lat,
            "compass_angle": props.get("compass_angle"),
            "is_pano": bool(props.get("is_pano", False)),
        })

    return features


# -----------------------------------------------------------------------------
# Greedy FPS, seeded from bbox center
# -----------------------------------------------------------------------------


def _farthest_point_sample(
    items: list[MapillaryImage],
    target_count: int,
    *,
    spatial_weight: float = FPS_SPATIAL_WEIGHT,
    compass_weight: float = FPS_COMPASS_WEIGHT,
) -> list[MapillaryImage]:
    """
    Greedily select target_count items maximizing diversity in
    (position, compass).

    Distance metric: spatial_weight * euclidean(east, north)
                   + compass_weight * circular_diff(compass)

    Items without compass angle pay zero compass cost (they remain
    selectable, scored on spatial distance alone).

    Seed: the item closest to the ENU origin (which is the bbox center,
    by construction in fetch_mapillary_images). This makes the sampling
    fully deterministic and visually anchored on the user's query point.
    """
    if len(items) <= target_count:
        return list(items)

    positions = np.array([img.position_enu for img in items])  # (N, 2)
    compasses = np.array([
        img.compass_angle_deg if img.compass_angle_deg is not None else np.nan
        for img in items
    ])

    n = len(items)
    # Seed: image closest to (0, 0) in ENU = closest to bbox center.
    dist_to_center = np.sqrt(positions[:, 0] ** 2 + positions[:, 1] ** 2)
    selected_idx: list[int] = [int(np.argmin(dist_to_center))]
    min_dists = np.full(n, np.inf)

    for _ in range(target_count - 1):
        last = selected_idx[-1]

        dx = positions[:, 0] - positions[last, 0]
        dy = positions[:, 1] - positions[last, 1]
        spatial_d = np.sqrt(dx * dx + dy * dy)

        if np.isnan(compasses[last]):
            compass_d = np.zeros(n)
        else:
            diff = np.abs(compasses - compasses[last])
            compass_d = np.where(
                np.isnan(compasses),
                0.0,
                np.minimum(diff, 360.0 - diff),
            )

        combined = spatial_weight * spatial_d + compass_weight * compass_d
        min_dists = np.minimum(min_dists, combined)

        for idx in selected_idx:
            min_dists[idx] = -np.inf

        next_idx = int(np.argmax(min_dists))
        selected_idx.append(next_idx)

    return [items[i] for i in selected_idx]


# -----------------------------------------------------------------------------
# Optional Graph API: per-image thumb URL, then download
# -----------------------------------------------------------------------------


# WorldGen / WorldMirror conditioning needs >256 px input; 2048 px is
# Mapillary's largest publicly served thumbnail. The Graph API field name
# must match the response key, so both are kept consistent below.
_FIELDS_THUMB = "thumb_2048_url"


def _fetch_thumb_url(image_id: str, timeout_s: float = 15.0) -> str | None:
    """Fetch the (signed, short-lived) thumbnail URL for one image."""
    url = f"https://graph.mapillary.com/{image_id}"
    params = {"fields": _FIELDS_THUMB}
    headers = {"Authorization": f"OAuth {get_mapillary_token()}"}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=timeout_s)
        r.raise_for_status()
        return r.json().get(_FIELDS_THUMB)
    except Exception as exc:
        logger.warning("Failed to fetch thumb URL for %s: %s", image_id, exc)
        return None


def _download_thumbnail(
    url: str,
    dest_path: Path,
    timeout_s: float = 15.0,
) -> bool:
    """Download to dest_path. Returns True on success."""
    if dest_path.exists():
        return True

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        response = requests.get(url, timeout=timeout_s, stream=True)
        response.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as exc:
        logger.warning("Thumbnail download failed for %s: %s", url, exc)
        if dest_path.exists():
            dest_path.unlink()
        return False


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------


def fetch_mapillary_images(
    lat: float,
    lon: float,
    radius_m: float = 150.0,
    *,
    frame: LocalFrame | None = None,
    target_count: int | None = None,
    download_thumbnails: bool = True,
    cache_dir: Path | str = "data/mapillary_cache",
    use_cache: bool = True,
) -> list[MapillaryImage]:
    """
    Fetch well-distributed street-level images near a WGS84 point.

    The query region is a square in ENU centered on (lat, lon) with
    half-side equal to radius_m + BUFFER_M. The buffer (default 20 m)
    gives generated imagery a small margin beyond the region of interest.

    Args:
        lat, lon: Query center in WGS84 degrees.
        radius_m: Half-side of the unbuffered square, in meters. The
            actual fetch region is (radius_m + BUFFER_M) on each side.
        frame: ENU frame for output positions. If None, created at
            (lat, lon), so the bbox center sits at ENU (0, 0).
        target_count: After FPS, return at most this many images.
            If None, computed from area density (20 imgs / 10 000 m²).
            If the candidate pool is smaller than target, all are kept.
        download_thumbnails: If True, call Graph API for signed thumb
            URLs and download each to disk. If False, no Graph API is
            touched and thumb_url / thumb_path stay empty / None.
        cache_dir: Root directory for tile cache and (if downloading)
            thumbnail cache.
        use_cache: If False, ignore the on-disk tile cache.

    Returns:
        list[MapillaryImage], length <= target_count, in FPS selection
        order (centermost image first).
    """
    if frame is None:
        frame = LocalFrame(lat0=lat, lon0=lon)

    cache_dir = Path(cache_dir)
    tile_cache_dir = cache_dir / "tiles"
    thumb_dir = cache_dir / "thumbnails"
    tile_cache_dir.mkdir(parents=True, exist_ok=True)

    if target_count is None:
        target_count = _density_target(radius_m)
        logger.info(
            "target_count auto-computed from density: %d "
            "(radius=%s m, buffer=%s m, density=%s/m²)",
            target_count, radius_m, BUFFER_M, IMAGES_PER_M2,
        )

    # --- 1. Compute bbox and covering tiles ---
    bbox = _square_bbox_with_buffer(lat, lon, radius_m)
    tiles = _tiles_covering_bbox(bbox)
    logger.info(
        "Bbox %s covered by %d zoom-%d tile(s)",
        bbox, len(tiles), TILE_ZOOM,
    )

    # --- 2. Fetch + decode each tile, accumulate features ---
    raw_features: list[dict] = []
    for tile in tiles:
        pbf = _fetch_tile_bytes(tile, tile_cache_dir, use_cache=use_cache)
        if pbf is None:
            continue
        raw_features.extend(_decode_image_features(pbf, tile))
    logger.info("Decoded %d raw image features from tiles", len(raw_features))

    # --- 3. Filter to bbox (tiles overhang) and parse into MapillaryImage ---
    west, south, east, north = bbox
    parsed: list[MapillaryImage] = []
    seen: set[str] = set()
    for feat in raw_features:
        mid = feat["id"]
        if not mid or mid in seen:
            continue
        img_lon, img_lat = feat["lon"], feat["lat"]
        if not (west <= img_lon <= east and south <= img_lat <= north):
            continue
        seen.add(mid)

        e, n, _ = frame.wgs84_to_enu(img_lat, img_lon)

        compass = feat.get("compass_angle")
        compass = float(compass) if compass is not None else None

        parsed.append(MapillaryImage(
            mapillary_id=mid,
            position_enu=(float(e), float(n)),
            compass_angle_deg=compass,
            is_pano=bool(feat.get("is_pano", False)),
            thumb_url="",
            thumb_path=None,
        ))

    logger.info(
        "Parsed %d unique images inside bbox (from %d raw features)",
        len(parsed), len(raw_features),
    )
    if not parsed:
        return []

    # --- 4. Farthest-point sampling, seeded from bbox center ---
    sampled = _farthest_point_sample(parsed, target_count=target_count)
    logger.info(
        "FPS: selected %d (target %d, pool %d)",
        len(sampled), target_count, len(parsed),
    )

    if not download_thumbnails:
        return sampled

    # --- 5. Per-image Graph API for thumb URL, then download ---
    final: list[MapillaryImage] = []
    for img in sampled:
        thumb_url = _fetch_thumb_url(img.mapillary_id) or ""
        thumb_path: Path | None = None
        if thumb_url:
            candidate_path = thumb_dir / f"{img.mapillary_id}.jpg"
            if _download_thumbnail(thumb_url, candidate_path):
                thumb_path = candidate_path
        final.append(MapillaryImage(
            mapillary_id=img.mapillary_id,
            position_enu=img.position_enu,
            compass_angle_deg=img.compass_angle_deg,
            is_pano=img.is_pano,
            thumb_url=thumb_url,
            thumb_path=thumb_path,
            ground_elevation_m=img.ground_elevation_m,
        ))
        time.sleep(0.05)  # politeness between Graph API hits

    return final