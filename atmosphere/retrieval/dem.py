"""
Ground elevation retrieval from DEM (Digital Elevation Model) sources.

This module is the third "data source connector" in the Atmosphere
pipeline, alongside buildings.py (OSM footprints) and streets.py (OSM
street network). It exposes a function that takes WGS84 (lat, lon)
and returns ground elevation in meters above the WGS84 ellipsoid.

Phase 0 source: King County 2016-2017 PSLC LiDAR DEM, distributed by
NOAA on AWS S3 as a set of 5 GeoTIFFs covering ~8.9 GB total. The
data is bare-earth (DTM, not DSM), so trees and buildings are
already removed and we get the actual ground surface — which is what
SCP's relative-z prior requires.

Design notes:
    - Coordinate handling: source data is EPSG:2926 (Washington State
      Plane North, US Survey Feet). pyproj handles the WGS84 -> 2926
      transform; we multiply by 1200/3937 to convert US Survey Feet
      to meters at the end. Both transforms are explicit so a future
      reviewer can audit them.
    - On-demand TIFF download: the VRT index is small (~5 KB) and
      cached on first use; individual TIFFs are downloaded only when
      a query falls inside one. For DLR (47.6059, -122.3392), only
      delivery1 (~2.3 GB) is needed and other deliveries never
      download.
    - Fail-soft: a coordinate outside Seattle (e.g. a Boston test
      coordinate during generalization) returns (None, None) rather
      than raising. Caller (M3) treats this as a fail-fast condition
      for the image (drops it from the WorldMirror bundle).
    - Provenance: each elevation result is paired with a DEMSource
      tag so SFB benchmark can stratify error by source resolution
      (1 m vs 10 m vs estimated).

The module surface intentionally hides rasterio and pyproj: callers
see only sample_elevation_m / sample_elevations_batch returning
floats, never raster objects or CRS strings.
"""

from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import requests

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Constants — King County PSLC 2016-2017 DEM
# -----------------------------------------------------------------------------

# NOAA AWS S3 mirror of the PSLC dataset. The VRT is a small XML index
# listing the 5 sub-TIFFs and their spatial extents in EPSG:2926.
_KC_VRT_URL = (
    "https://noaa-nos-coastal-lidar-pds.s3.amazonaws.com"
    "/dem/WA_King_DEM_2016_8589/WA_King_DEM_2016_m8589_EPSG-2926.vrt"
)

# Each sub-TIFF lives next to the VRT.
_KC_TIFF_BASE = (
    "https://noaa-nos-coastal-lidar-pds.s3.amazonaws.com"
    "/dem/WA_King_DEM_2016_8589"
)

# 1 US Survey Foot in meters (exact rational, defined by US Survey Foot
# definition: 1200/3937 m).
_US_SURVEY_FOOT_TO_M = 1200.0 / 3937.0

# EPSG codes used here.
_EPSG_WGS84 = 4326
_EPSG_WA_STATE_PLANE_N_FEET = 2926


# -----------------------------------------------------------------------------
# Public types
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class DEMSource:
    """
    Provenance for a single elevation sample.

    Carrying source info on every sample lets SFB benchmark stratify
    reconstruction error by DEM quality (1 m LiDAR vs 10 m USGS vs
    coarser fallback) without needing a separate per-image source
    column.

    Attributes:
        name: Human-readable dataset name (e.g. "King County PSLC 2016").
        resolution_m: Nominal horizontal pixel size in meters.
        epsg: Source CRS EPSG code (before our reprojection to WGS84).
    """

    name: str
    resolution_m: float
    epsg: int


KING_COUNTY_PSLC_2016 = DEMSource(
    name="King County PSLC 2016",
    resolution_m=1.0,
    epsg=_EPSG_WA_STATE_PLANE_N_FEET,
)


@dataclass(frozen=True)
class _TiffExtent:
    """
    Internal: one sub-TIFF's identity and 2D bbox in EPSG:2926.

    Stored in the local index.json so we can route a query to the
    right TIFF without re-parsing the VRT every call.
    """

    filename: str
    minx: float
    miny: float
    maxx: float
    maxy: float

    def contains(self, x: float, y: float) -> bool:
        return self.minx <= x <= self.maxx and self.miny <= y <= self.maxy


# -----------------------------------------------------------------------------
# VRT parsing — runs once, caches the result
# -----------------------------------------------------------------------------


def _parse_vrt_extents(vrt_bytes: bytes) -> list[_TiffExtent]:
    """
    Extract sub-TIFF extents from a GDAL VRT XML document.

    A VRT looks like:
      <VRTDataset rasterXSize="..." rasterYSize="...">
        <GeoTransform>...</GeoTransform>
        <VRTRasterBand>
          <ComplexSource> or <SimpleSource>
            <SourceFilename>kingcounty_delivery1_be.tif</SourceFilename>
            <DstRect xOff="..." yOff="..." xSize="..." ySize="..."/>
          </ComplexSource>
        </VRTRasterBand>
      </VRTDataset>

    DstRect coordinates are in *pixel space relative to the VRT's own
    raster origin*, so we combine them with the top-level GeoTransform
    to recover each TIFF's bbox in the VRT CRS.
    """
    root = ET.fromstring(vrt_bytes)

    geo = root.findtext("GeoTransform")
    if geo is None:
        raise ValueError("VRT missing GeoTransform")
    # Format: "origin_x, pixel_w, 0, origin_y, 0, -pixel_h"
    gt = [float(s) for s in geo.split(",")]
    origin_x, pixel_w, _, origin_y, _, pixel_h_neg = gt
    pixel_h = -pixel_h_neg  # canonical positive

    extents: list[_TiffExtent] = []
    # Sources can be tagged Simple or Complex; both have the same
    # children we need.
    for src in root.iter():
        tag = src.tag.split("}")[-1]
        if tag not in ("SimpleSource", "ComplexSource"):
            continue

        fname_el = src.find("SourceFilename")
        dst_el = src.find("DstRect")
        if fname_el is None or dst_el is None:
            continue

        fname = (fname_el.text or "").strip()
        if not fname:
            continue

        x_off = float(dst_el.attrib["xOff"])
        y_off = float(dst_el.attrib["yOff"])
        x_sz = float(dst_el.attrib["xSize"])
        y_sz = float(dst_el.attrib["ySize"])

        minx = origin_x + x_off * pixel_w
        maxx = origin_x + (x_off + x_sz) * pixel_w
        # In GDAL convention origin_y is the top; pixels grow southward.
        maxy = origin_y - y_off * pixel_h
        miny = origin_y - (y_off + y_sz) * pixel_h

        extents.append(_TiffExtent(
            filename=fname,
            minx=minx, miny=miny, maxx=maxx, maxy=maxy,
        ))

    return extents


def _ensure_index(cache_dir: Path, *, use_cache: bool = True) -> list[_TiffExtent]:
    """
    Return the list of sub-TIFF extents, downloading + parsing the VRT
    if no cached index exists yet.

    The index.json file is small (<1 KB) and persists indefinitely; it
    only needs refresh if NOAA republishes the dataset. We don't check
    for that — manual `rm data/dem_cache/index.json` triggers redownload.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_path = cache_dir / "index.json"

    if use_cache and index_path.exists():
        raw = json.loads(index_path.read_text())
        return [_TiffExtent(**entry) for entry in raw]

    logger.info("Fetching DEM VRT index: %s", _KC_VRT_URL)
    r = requests.get(_KC_VRT_URL, timeout=30.0)
    r.raise_for_status()
    extents = _parse_vrt_extents(r.content)

    index_path.write_text(json.dumps([asdict(e) for e in extents], indent=2))
    logger.info("Cached DEM index with %d sub-TIFFs at %s", len(extents), index_path)
    return extents


# -----------------------------------------------------------------------------
# TIFF download — on demand, with streaming + progress logging
# -----------------------------------------------------------------------------


def _ensure_tiff(filename: str, cache_dir: Path) -> Path:
    """
    Download a sub-TIFF if not already cached locally; return its path.

    These files are large (1.3–2.3 GB each); we stream so memory stays
    bounded and log periodic progress so the user knows it's working.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = cache_dir / filename

    if local_path.exists() and local_path.stat().st_size > 0:
        return local_path

    url = f"{_KC_TIFF_BASE}/{filename}"
    logger.info("Downloading DEM tile: %s", filename)
    logger.info("  source: %s", url)
    logger.info("  this is a one-time download (~1-2 GB), please wait")

    tmp_path = local_path.with_suffix(local_path.suffix + ".part")
    with requests.get(url, stream=True, timeout=60.0) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        chunk = 1024 * 1024  # 1 MB
        downloaded = 0
        last_log = 0
        with open(tmp_path, "wb") as f:
            for block in r.iter_content(chunk_size=chunk):
                if not block:
                    continue
                f.write(block)
                downloaded += len(block)
                # Log every 100 MB so we don't spam.
                if downloaded - last_log >= 100 * 1024 * 1024:
                    pct = (downloaded / total * 100) if total else 0
                    logger.info(
                        "  ... %.0f%% (%d MB / %d MB)",
                        pct, downloaded // (1024 * 1024),
                        total // (1024 * 1024) if total else 0,
                    )
                    last_log = downloaded

    tmp_path.rename(local_path)
    logger.info("DEM tile cached: %s", local_path)
    return local_path


# -----------------------------------------------------------------------------
# Public API — sample one or many points
# -----------------------------------------------------------------------------


def sample_elevation_m(
    lat: float,
    lon: float,
    *,
    cache_dir: Path | str = "data/dem_cache",
) -> tuple[float | None, DEMSource | None]:
    """
    Sample ground elevation at a single WGS84 point.

    Returns:
        (elevation_m, source) on success — elevation is meters above
        the WGS84 ellipsoid, source carries provenance.
        (None, None) if the point falls outside our DEM coverage
        (caller treats this as fail-fast for the image).

    The first call for a given Seattle-area coordinate downloads one
    ~2 GB sub-TIFF (a one-time cost); subsequent calls hit the local
    cache.
    """
    results = sample_elevations_batch([(lat, lon)], cache_dir=cache_dir)
    return results[0]


def sample_elevations_batch(
    points: Iterable[tuple[float, float]],
    *,
    cache_dir: Path | str = "data/dem_cache",
) -> list[tuple[float | None, DEMSource | None]]:
    """
    Batched version: groups points by which sub-TIFF they fall in and
    opens each TIFF at most once.

    For N=24 Mapillary images all clustered in a ~150 m radius (the
    Day 5 case), every point lands in the same TIFF, so we open that
    TIFF once and sample 24 values from it — much faster than 24
    sequential single-point calls.

    Order of results matches input order.
    """
    # Lazy imports so the module is importable on a machine without
    # rasterio installed (unit tests can mock the sampling path).
    import rasterio
    from pyproj import Transformer

    cache_dir = Path(cache_dir)
    points = list(points)
    if not points:
        return []

    extents = _ensure_index(cache_dir)
    tiff_dir = cache_dir / "tiles"

    # Project all WGS84 points to EPSG:2926 (units: US Survey Feet).
    # always_xy=True returns (x, y) i.e. (lon-equivalent, lat-equivalent).
    transformer = Transformer.from_crs(
        _EPSG_WGS84, _EPSG_WA_STATE_PLANE_N_FEET, always_xy=True,
    )

    # Pre-allocate results so we can write at original indices after
    # the per-TIFF group loop.
    results: list[tuple[float | None, DEMSource | None]] = [
        (None, None)] * len(points)

    # Bucket points by which TIFF contains them.
    by_tiff: dict[str, list[tuple[int, float, float]]] = {}
    outside_count = 0
    for i, (lat, lon) in enumerate(points):
        x_ft, y_ft = transformer.transform(lon, lat)
        # Find the first (and assumed only) TIFF containing this point.
        matched = None
        for ext in extents:
            if ext.contains(x_ft, y_ft):
                matched = ext.filename
                break
        if matched is None:
            outside_count += 1
            continue
        by_tiff.setdefault(matched, []).append((i, x_ft, y_ft))

    if outside_count:
        logger.warning(
            "%d / %d points fell outside King County DEM coverage; "
            "those return (None, None).",
            outside_count, len(points),
        )

    # Sample each TIFF group with one open() call.
    for filename, group in by_tiff.items():
        tiff_path = _ensure_tiff(filename, tiff_dir)
        with rasterio.open(tiff_path) as src:
            # rasterio.sample takes an iterable of (x, y) in the
            # raster's own CRS (= EPSG:2926 here). It yields one
            # array per point with one value per band; we only have
            # one band.
            coords = [(x, y) for (_i, x, y) in group]
            for (orig_i, _x, _y), value_array in zip(
                group, src.sample(coords), strict=True,
            ):
                raw_ft = float(value_array[0])
                # Rasterio nodata sentinel for this dataset is usually
                # a large negative number; treat anything implausible
                # for Earth elevation as no-data.
                if raw_ft < -5000.0 or raw_ft > 50000.0:
                    results[orig_i] = (None, None)
                    continue
                elevation_m = raw_ft * _US_SURVEY_FOOT_TO_M
                results[orig_i] = (elevation_m, KING_COUNTY_PSLC_2016)

    return results