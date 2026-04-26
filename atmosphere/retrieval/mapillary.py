"""
Mapillary street-level imagery retrieval.

Pipeline (parallels atmosphere.retrieval.buildings):

    fetch_mapillary_images(lat, lon, radius_m)
        1. Convert radius to lat/lon bbox
        2. Query Mapillary API v4 /images endpoint
        3. Cache raw JSON response to disk
        4. Parse into MapillaryImage instances in local ENU frame
        5. Apply greedy farthest-point sampling to get ~K well-distributed images
        6. Download thumbnails to disk cache
        7. Return list[MapillaryImage] with local thumbnail paths

Design notes:
    - Like buildings.py, no third-party data types leak to callers.
      The raw JSON is parsed into our MapillaryImage dataclass.
    - The farthest-point sampling is the key quality step: raw Mapillary
      density is wildly uneven (car dashcam uploaders cluster on major
      streets), and downstream modules need view diversity more than raw
      coverage.
    - Thumbnails are downloaded (not just URL-stored) because Mapillary's
      CDN URLs are signed and expire after a few hours.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

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
                      Upward axis is not meaningful for street-level imagery.
        compass_angle_deg: Camera heading, degrees clockwise from north
                           (0 = N, 90 = E). None if missing from Mapillary.
        captured_at: When the image was taken (timezone-aware UTC).
        thumb_url: Original Mapillary CDN URL (may expire; for reference only).
        thumb_path: Local filesystem path to the cached thumbnail, or None
                    if the download failed or was skipped.
    """

    mapillary_id: str
    position_enu: tuple[float, float]
    compass_angle_deg: float | None
    captured_at: datetime
    thumb_url: str
    thumb_path: Path | None

    @property
    def has_compass(self) -> bool:
        return self.compass_angle_deg is not None


# -----------------------------------------------------------------------------
# Bbox math
# -----------------------------------------------------------------------------


def _radius_to_bbox(
    lat: float, lon: float, radius_m: float
) -> tuple[float, float, float, float]:
    """
    Approximate a WGS84 square bbox centered on (lat, lon) with side 2*radius_m.

    Returns (west_lon, south_lat, east_lon, north_lat) — the order Mapillary
    expects. The approximation is good to ~0.1% at city scales.
    """
    lat_deg_per_m = 1.0 / 111_000.0
    lon_deg_per_m = 1.0 / (111_000.0 * np.cos(np.radians(lat)))

    dlat = radius_m * lat_deg_per_m
    dlon = radius_m * lon_deg_per_m

    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


# -----------------------------------------------------------------------------
# API wrapper
# -----------------------------------------------------------------------------


MAPILLARY_GRAPH_URL = "https://graph.mapillary.com/images"

# Lightweight: used for the bbox search (cheap, can request many)
_FIELDS_LIST = ",".join([
    "id",
    "geometry",
    "compass_angle",
    "captured_at",
])

# Per-image: only requested for the final selected K images.
# thumb_256_url is expensive (signed CDN URL) — one image at a time.
_FIELDS_THUMB = "thumb_256_url"


def _split_bbox(
    bbox: tuple[float, float, float, float],
    n: int=3,
) -> list[tuple[float, float, float, float]]:
    """
    Split a bbox into n x n smaller bboxes.

    Mapillary's Graph API times out for densely captured areas (downtown
    Seattle, central Copenhagen) even with relatively small bboxes. The
    fix is to split a single query into a grid of smaller queries and
    union the results.

    Args:
        bbox: (west, south, east, north) in degrees.
        n: Grid size. Returns n² sub-bboxes.

    Returns:
        List of n² sub-bboxes, in row-major (row=south-to-north,
        col=west-to-east) order.
    """
    west, south, east, north = bbox
    dx = (east - west) / n
    dy = (north - south) / n
    sub_bboxes = []

    for i in range(n):
        for j in range(n):
            sub_bboxes.append((
                west+dx*j,
                south+dy*i,
                west+dx*(j+1),
                south+dy*(i+1),
            ))
    
    return sub_bboxes

def _raw_fetch(
    bbox: tuple[float, float, float, float],
    limit: int,
) -> list[dict]:
    """
    Single Mapillary /images query. No pagination, no caching.
    Returns the raw list of feature dicts from the JSON response.
    """
    west, south, east, north = bbox
    params = {
        "bbox": f"{west},{south},{east},{north}",
        "fields": _FIELDS_LIST,
        "limit": limit,
    }
    headers = {"Authorization": f"OAuth {get_mapillary_token()}"}

    response = requests.get(
        MAPILLARY_GRAPH_URL,
        params=params,
        headers=headers,
        timeout=30,
    )
    response.raise_for_status()

    payload = response.json()
    return payload.get("data", [])

def _fetch_recursive(
    bbox: tuple[float, float, float, float],
    limit: int,
    *,
    depth: int = 0,
    max_depth: int = 4,
    indent: str = "",
) -> list[dict]:
    """
    Fetch images for a bbox; if Mapillary times out (500), recursively
    split the bbox into 2x2 quarters and retry each.

    This adapts to local density: sparse bboxes return immediately,
    dense ones (downtown cores) are subdivided automatically until
    they fit under the API timeout.

    Args:
        bbox: (west, south, east, north).
        limit: per-bbox limit. Note: total returned can exceed this if
            recursion happens, since each sub-bbox gets its own limit.
        depth: current recursion depth (0 = root).
        max_depth: stop recursing past this depth. At depth 4 with the
            initial 300m bbox, the smallest sub-bbox is ~19m × 19m.
        indent: prefix for log output (visual depth indicator).

    Returns:
        Combined list of items from this bbox or its descendants.
    """
    try:
        items = _raw_fetch(bbox, limit=limit)
        logger.info(
            "%s[depth %d] %d items returned (limit %d)",
            indent, depth, len(items), limit,
        )
        return items
    except requests.exceptions.HTTPError as exc:
        if exc.response is None or exc.response.status_code != 500:
            raise  # only retry on 500 (timeout); re-raise others
        if depth >= max_depth:
            logger.warning(
                "%s[depth %d] giving up after max_depth, bbox=%s",
                indent, depth, bbox,
            )
            return []

        logger.info(
            "%s[depth %d] timeout, splitting 2×2 and recursing",
            indent, depth,
        )
        sub_bboxes = _split_bbox(bbox, n=2)
        all_items: list[dict] = []
        for sub in sub_bboxes:
            sub_items = _fetch_recursive(
                sub, limit=limit,
                depth=depth + 1, max_depth=max_depth,
                indent=indent + "  ",
            )
            all_items.extend(sub_items)
            time.sleep(0.05)
        return all_items

def _fetch_thumb_url(image_id: str, timeout_s: float = 15.0) -> str | None:
    """
    Fetch the (short-lived signed) thumbnail URL for a single image.

    Mapillary's batch endpoint refuses to return thumb URLs in bulk because
    they're individually signed CDN URLs. So we request them one image at
    a time, only for the final sampled set.
    """
    url = f"https://graph.mapillary.com/{image_id}"
    params = {"fields": _FIELDS_THUMB}
    headers = {"Authorization": f"OAuth {get_mapillary_token()}"}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=timeout_s)
        r.raise_for_status()
        return r.json().get("thumb_256_url")
    except Exception as exc:
        logger.warning("Failed to fetch thumb URL for %s: %s", image_id, exc)
        return None


# -----------------------------------------------------------------------------
# Farthest-point sampling: the quality filter
# -----------------------------------------------------------------------------


def _farthest_point_sample(
    items: list[MapillaryImage],
    target_count: int,
    *,
    # Weighting: 20 m spatial distance ≈ 45° orientation difference.
    # Beta/alpha ≈ 20/45 ≈ 0.44
    spatial_weight: float = 1.0,
    compass_weight: float = 0.44,
    seed: int | None = 0,
) -> list[MapillaryImage]:
    """
    Greedily select `target_count` items that maximize mutual diversity
    in (spatial position, compass angle).

    For each candidate, the "distance to already-selected" is:
        dist = spatial_weight * euclidean(east, north)
             + compass_weight * circular_angle_diff(compass_deg)

    Items without compass angle contribute no compass penalty (they are
    still selectable, just measured on spatial distance alone).

    This is the CV-standard farthest-point sampling algorithm. It is O(N*K)
    rather than O(N log N), acceptable here because N <= 500 and K <= 100.
    """
    if len(items) <= target_count:
        return list(items)

    rng = np.random.default_rng(seed)

    # Pre-extract arrays for speed
    positions = np.array([img.position_enu for img in items])  # (N, 2)
    compasses = np.array([
        img.compass_angle_deg if img.compass_angle_deg is not None else np.nan
        for img in items
    ])

    # Seed: pick a random starting point for reproducibility via seed
    n = len(items)
    selected_idx = [int(rng.integers(n))]
    # Track min distance from each candidate to the selected set
    min_dists = np.full(n, np.inf)

    for _ in range(target_count - 1):
        last = selected_idx[-1]

        # Spatial distance to the latest selected point
        dx = positions[:, 0] - positions[last, 0]
        dy = positions[:, 1] - positions[last, 1]
        spatial_d = np.sqrt(dx * dx + dy * dy)

        # Compass distance (circular, 0-180 deg)
        if np.isnan(compasses[last]):
            compass_d = np.zeros(n)
        else:
            diff = np.abs(compasses - compasses[last])
            compass_d = np.where(
                np.isnan(compasses),
                0.0,  # missing compass → zero penalty
                np.minimum(diff, 360.0 - diff),
            )

        combined = spatial_weight * spatial_d + compass_weight * compass_d
        min_dists = np.minimum(min_dists, combined)

        # Exclude already-selected by setting their min_dist to -inf
        for idx in selected_idx:
            min_dists[idx] = -np.inf

        # Pick the one farthest from the selected set
        next_idx = int(np.argmax(min_dists))
        selected_idx.append(next_idx)

    return [items[i] for i in selected_idx]


# -----------------------------------------------------------------------------
# Thumbnail download
# -----------------------------------------------------------------------------


def _download_thumbnail(
    url: str,
    dest_path: Path,
    timeout_s: float = 15.0,
) -> bool:
    """
    Download a thumbnail to `dest_path`. Returns True on success, False
    on any failure (including HTTP errors, timeouts, disk errors).

    Failures are logged but do not raise — a missing thumbnail is
    degraded but not fatal for downstream modules.
    """
    if dest_path.exists():
        return True  # already cached

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
            dest_path.unlink()  # remove partial file
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
    api_limit: int = 500,
    target_count: int = 100,
    download_thumbnails: bool = True,
    cache_dir: Path | str = "data/mapillary_cache",
    use_cache: bool = True,
) -> list[MapillaryImage]:
    """
    Fetch well-distributed street-level images near a WGS84 point.

    Args:
        lat: Query center latitude in degrees.
        lon: Query center longitude in degrees.
        radius_m: Search radius in meters. Returned images' positions
            are in the corresponding bounding box (not clipped to a circle).
        frame: Local ENU frame for output positions. If None, created at
            the query center.
        api_limit: Max raw images to fetch from Mapillary (before sampling).
            500 covers most Seattle neighborhoods; hitting this triggers a
            warning suggesting the user shrink radius or add pagination.
        target_count: After farthest-point sampling, return this many.
        download_thumbnails: If True, download each thumbnail to local cache
            and populate `MapillaryImage.thumb_path`. If False, thumb_path=None.
        cache_dir: Directory for raw API JSON + thumbnails.
        use_cache: If False, always re-fetch from Mapillary.

    Returns:
        A list of MapillaryImage, length <= target_count. Order reflects
        sampling order (farthest-point greedy order), not position.
    """
    if frame is None:
        frame = LocalFrame(lat0=lat, lon0=lon)

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    raw_cache = cache_dir / f"raw_{lat:.4f}_{lon:.4f}_r{int(radius_m)}.json"
    thumb_dir = cache_dir / "thumbnails"

    # --- 1. Fetch raw JSON (or load from cache) ---
    if use_cache and raw_cache.exists():
        logger.info("Loading Mapillary response from cache: %s", raw_cache.name)
        with open(raw_cache) as f:
            raw_items = json.load(f)
    else:
        bbox = _radius_to_bbox(lat, lon, radius_m)
        sub_bboxes = _split_bbox(bbox, n=3)
        per_sub_limit = max(20, api_limit//len(sub_bboxes))
        logger.info(
            "Fetching Mapillary images: bbox=%s, split into %d sub-bboxes, "
            "limit per sub-bbox=%d",
            bbox, len(sub_bboxes), per_sub_limit,
        )
        raw_items = []
        seen_ids: set[str] = set()
        for i, sub_bbox in enumerate(sub_bboxes):
            try:
                sub_items = _raw_fetch(sub_bbox, limit=per_sub_limit)
            except Exception as e:
                logger.warning(
                    "Sub-bbox %d/%d failed (%s); skipping",
                    i+1, len(sub_bboxes), e,
                )
                continue
            new_count = 0
            for item in sub_items:
                item_id = str(item.get("id", ""))
                if item_id and item_id not in seen_ids:
                    seen_ids.add(item_id)
                    raw_items.append(item)
                    new_count += 1
            logger.info(
                "Sub-bbox %d/%d: %d returned, %d new (dedup)",
                i+1, len(sub_bboxes), len(sub_items), new_count,
            )
            time.sleep(0.1)

        if len(raw_items) >= api_limit:
            logger.warning(
                "Mapillary returned %d items, matching the api_limit. "
                "This bbox likely contains more images; results are a "
                "partial sample. Consider shrinking radius or adding pagination.",
                len(raw_items),
            )
        with open(raw_cache, "w") as f:
            json.dump(raw_items, f)

    logger.info("Raw Mapillary items: %d", len(raw_items))
    if not raw_items:
        return []

    # --- 2. Parse into MapillaryImage (still in WGS84; convert to ENU) ---
    parsed: list[MapillaryImage] = []
    for item in raw_items:
        try:
            mid = str(item["id"])
            coords = item["geometry"]["coordinates"]  # [lon, lat]
            img_lon, img_lat = float(coords[0]), float(coords[1])
            e, n, _ = frame.wgs84_to_enu(img_lat, img_lon)

            compass = item.get("compass_angle")
            compass = float(compass) if compass is not None else None

            captured_ms = item.get("captured_at")
            if captured_ms is None:
                # Skip images without a timestamp; they're usually corrupt.
                continue
            captured_at = datetime.fromtimestamp(float(captured_ms) / 1000.0)

            thumb_url = ""  # filled in after sampling

            parsed.append(MapillaryImage(
                mapillary_id=mid,
                position_enu=(float(e), float(n)),
                compass_angle_deg=compass,
                captured_at=captured_at,
                thumb_url=thumb_url,
                thumb_path=None,  # filled in after download step
            ))
        except (KeyError, TypeError, ValueError) as exc:
            logger.debug("Skipping malformed Mapillary item: %s (%s)", item, exc)
            continue

    logger.info("Successfully parsed %d of %d", len(parsed), len(raw_items))

    # --- 3. Farthest-point sampling ---
    sampled = _farthest_point_sample(parsed, target_count=target_count)
    logger.info(
        "Farthest-point sampled to %d (target %d)",
        len(sampled), target_count,
    )

    # --- 3.5 Fetch thumb URLs only for the K sampled images ---
    sampled_with_urls: list[MapillaryImage] = []
    for img in sampled:
        thumb_url = _fetch_thumb_url(img.mapillary_id) or ""
        sampled_with_urls.append(MapillaryImage(
            mapillary_id=img.mapillary_id,
            position_enu=img.position_enu,
            compass_angle_deg=img.compass_angle_deg,
            captured_at=img.captured_at,
            thumb_url=thumb_url,
            thumb_path=None,
        ))
        time.sleep(0.05)  # politeness
    sampled = sampled_with_urls

    # --- 4. Download thumbnails if requested ---
    if download_thumbnails:
        final: list[MapillaryImage] = []
        for img in sampled:
            thumb_path = thumb_dir / f"{img.mapillary_id}.jpg" if img.thumb_url else None
            if thumb_path and _download_thumbnail(img.thumb_url, thumb_path):
                final.append(MapillaryImage(
                    mapillary_id=img.mapillary_id,
                    position_enu=img.position_enu,
                    compass_angle_deg=img.compass_angle_deg,
                    captured_at=img.captured_at,
                    thumb_url=img.thumb_url,
                    thumb_path=thumb_path,
                ))
            else:
                # Keep the image but without a thumb path.
                final.append(img)
            # Tiny politeness delay between downloads
            time.sleep(0.05)
        return final
    else:
        return sampled