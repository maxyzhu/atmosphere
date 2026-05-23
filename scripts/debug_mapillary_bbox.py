"""
Diagnose the "27696 images inside r=150m bbox" anomaly.

Validates each stage of the Mapillary tile pipeline:
1. Tile coverage matches expectation
2. Raw decoded features have plausible (lon, lat)
3. Bbox filter actually filters
4. ENU positions match the geographic spread of the filtered images

Writes nothing; just prints diagnostics.
"""
from __future__ import annotations

import logging

import mercantile

from atmosphere.geo import LocalFrame
from atmosphere.retrieval.mapillary import (
    BUFFER_M,
    _decode_image_features,
    _fetch_tile_bytes,
    _square_bbox_with_buffer,
    _tiles_covering_bbox,
)


DLR_LAT = 47.6059
DLR_LON = -122.3392
RADIUS_M = 150.0


def main() -> None:
    logging.basicConfig(level=logging.WARNING)

    bbox = _square_bbox_with_buffer(DLR_LAT, DLR_LON, RADIUS_M)
    west, south, east, north = bbox
    print(f"Query bbox (W, S, E, N):")
    print(f"  W = {west:.6f}")
    print(f"  S = {south:.6f}")
    print(f"  E = {east:.6f}")
    print(f"  N = {north:.6f}")
    print(f"  bbox lon width  = {east - west:.6f} deg "
          f"({(east - west) * 111320 * 0.674:.0f} m at this lat)")
    print(f"  bbox lat height = {north - south:.6f} deg "
          f"({(north - south) * 111320:.0f} m)")

    tiles = _tiles_covering_bbox(bbox)
    print(f"\nCovering tiles ({len(tiles)}):")
    for t in tiles:
        b = mercantile.bounds(t)
        print(f"  z={t.z} x={t.x} y={t.y}  bbox={b.west:.4f},{b.south:.4f}"
              f" -> {b.east:.4f},{b.north:.4f}")
        print(f"    span: {b.east - b.west:.4f} deg lon, "
              f"{b.north - b.south:.4f} deg lat")

    print("\n--- Loading tile 0, peeking at raw decoded coords ---")
    from pathlib import Path
    cache_dir = Path("data/mapillary_cache/tiles")
    pbf_path = cache_dir / f"{tiles[0].z}_{tiles[0].x}_{tiles[0].y}.pbf"
    pbf = pbf_path.read_bytes()
    feats = _decode_image_features(pbf, tiles[0])
    print(f"  {len(feats)} raw features in tile 0")
    if feats:
        lons = [f["lon"] for f in feats]
        lats = [f["lat"] for f in feats]
        print(f"  lon range: [{min(lons):.6f}, {max(lons):.6f}]")
        print(f"  lat range: [{min(lats):.6f}, {max(lats):.6f}]")
        print(f"  expected lon: tile bounds [{mercantile.bounds(tiles[0]).west:.6f},"
              f" {mercantile.bounds(tiles[0]).east:.6f}]")
        print(f"  expected lat: tile bounds [{mercantile.bounds(tiles[0]).south:.6f},"
              f" {mercantile.bounds(tiles[0]).north:.6f}]")

    # How many features per tile actually fall inside the user bbox?
    in_bbox = sum(
        1 for f in feats
        if west <= f["lon"] <= east and south <= f["lat"] <= north
    )
    print(f"  features inside query bbox (tile 0): {in_bbox}")

    # Check the first 5 features for plausibility
    print("\n  first 5 raw features:")
    for f in feats[:5]:
        in_bb = (west <= f["lon"] <= east and south <= f["lat"] <= north)
        d_lon = f["lon"] - DLR_LON
        d_lat = f["lat"] - DLR_LAT
        print(f"    id={f['id'][:12]}... lon={f['lon']:.6f} lat={f['lat']:.6f}"
              f" d_lon={d_lon:+.4f}deg d_lat={d_lat:+.4f}deg in_bbox={in_bb}")

    # And ENU spread of the SAMPLED 24 images that the validation script saw
    print("\n--- ENU spread of FPS-sampled images ---")
    from atmosphere.retrieval.mapillary import fetch_mapillary_images
    images = fetch_mapillary_images(
        DLR_LAT, DLR_LON, RADIUS_M, target_count=24,
        download_thumbnails=False, fetch_camera_metadata=False,
        fetch_camera_z=False,
    )
    print(f"  {len(images)} sampled images")
    if images:
        es = [i.position_enu[0] for i in images]
        ns = [i.position_enu[1] for i in images]
        print(f"  ENU east  range: [{min(es):.1f}, {max(es):.1f}] m")
        print(f"  ENU north range: [{min(ns):.1f}, {max(ns):.1f}] m")
        # If spread > 200m, bbox filter is broken
        spread_e = max(es) - min(es)
        spread_n = max(ns) - min(ns)
        print(f"  spread: {spread_e:.0f} m east, {spread_n:.0f} m north")
        if spread_e > 400 or spread_n > 400:
            print(f"  *** ANOMALY: spread > 400m for r={RADIUS_M}m query")
            print("      bbox filter is broken or tile decode produces wrong coords")


if __name__ == "__main__":
    main()
