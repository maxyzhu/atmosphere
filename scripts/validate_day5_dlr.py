"""
End-to-end validation of Day 5 modules on real DLR data.

Runs the full retrieval -> M3 chain at the canonical DLR coordinate
and prints a structured report that answers three questions:

1. Does DEM sampling give a plausible ground elevation for DLR?
2. Does fetch_mapillary_images successfully populate camera_z_m + all
   four Graph-API metadata fields for most sampled images?
3. Does our Mapillary -> M3 -> Camera path produce yaw values that
   agree with Mapillary's separately-published computed_compass_angle?

The third is the real test. INSIGHTS Day 5 Part 1 §2 claimed (via
OpenSfM docs) that:
  - computed_rotation is w2c in the OpenSfM topocentric frame
  - topocentric == ENU when GPS is present
If both are true, the c2w yaw extracted from our Camera should match
computed_compass_angle within a few degrees per image.

A median yaw error < 5 deg means the claim holds. Anything > 30 deg
on most images means there's a frame mismatch we missed and M3 needs
a fix before WorldMirror inference will produce anything sensible.

Run from repo root:
    uv run python scripts/validate_day5_dlr.py
"""

from __future__ import annotations

import logging
import statistics

from atmosphere.canonicalize.align import (
    camera_yaw_deg,
    mapillary_to_camera,
)
from atmosphere.retrieval.dem import sample_elevation_m
from atmosphere.retrieval.mapillary import fetch_mapillary_images


DLR_LAT = 47.6059
DLR_LON = -122.3392


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=" * 72)
    print(f"  Day 5 validation @ DLR ({DLR_LAT}, {DLR_LON})")
    print("=" * 72)

    # -------- Q1. DEM sanity --------
    print("\n[1/3] DEM elevation sample at DLR ...")
    elev_m, source = sample_elevation_m(DLR_LAT, DLR_LON)
    if elev_m is None:
        print("  FAIL: DEM returned None — coordinate outside coverage")
        return
    print(f"  Ground elevation: {elev_m:.2f} m")
    print(f"  Source: {source.name} (resolution {source.resolution_m} m)")
    # Seattle downtown spans ~0 m (waterfront) to ~140 m (Capitol Hill),
    # and within a 150 m query radius elevation can vary 20-30 m due to
    # the steep east-west grade. DLR specifically is on 2nd Ave (one
    # block from waterfront), so its ground is around 5 m — the SCP
    # relative-z signal comes from the spread across the queried area,
    # not the absolute value at the center.
    if 0.0 < elev_m < 150.0:
        print("  OK: elevation is in the plausible Seattle downtown range")
    else:
        print(f"  WARN: elevation {elev_m:.2f} m is outside typical range")

    # -------- Q2. Full retrieval --------
    print("\n[2/3] Fetching Mapillary images with full metadata + camera_z_m ...")
    print("       (using radius_m=150, target_count=24 for first run)")
    images = fetch_mapillary_images(
        DLR_LAT, DLR_LON,
        radius_m=150.0,
        target_count=24,
        download_thumbnails=True,
        fetch_camera_metadata=True,
        fetch_camera_z=True,
    )
    print(f"  Retrieved: {len(images)} images")
    with_meta = sum(1 for i in images if i.has_camera_metadata)
    with_z = sum(1 for i in images if i.camera_z_m is not None)
    with_thumb = sum(1 for i in images if i.thumb_path is not None)
    n_pano = sum(1 for i in images if i.is_pano)
    print(f"  Field coverage:")
    print(f"    has_camera_metadata: {with_meta}/{len(images)}")
    print(f"    camera_z_m present : {with_z}/{len(images)}")
    print(f"    thumb on disk      : {with_thumb}/{len(images)}")
    print(f"    is_pano (skipped by M3): {n_pano}/{len(images)}")
    if with_meta < len(images) // 2:
        print("  WARN: fewer than half have full metadata — investigate Graph API")
    if with_z < len(images) // 2:
        print("  WARN: fewer than half have camera_z_m — DEM coverage issue?")

    # Show z distribution to confirm DEM is producing varying values
    z_values = [i.camera_z_m for i in images if i.camera_z_m is not None]
    if z_values:
        z_min, z_max = min(z_values), max(z_values)
        z_mean = statistics.mean(z_values)
        print(f"  camera_z_m range : [{z_min:.2f}, {z_max:.2f}] m, mean {z_mean:.2f}")
        print(f"  z spread         : {z_max - z_min:.2f} m"
              " (should be > 0 if DEM is per-image)")
        if z_max - z_min < 0.01:
            print("  WARN: all camera_z_m identical — DEM may be misconfigured")

    # -------- Q3. M3 yaw vs Mapillary compass --------
    print("\n[3/3] M3 -> Camera, comparing yaw to Mapillary's"
          " computed_compass_angle ...")
    yaw_errors_deg: list[float] = []
    n_m3_ok = 0
    n_m3_dropped = 0
    for img in images:
        cam = mapillary_to_camera(img)
        if cam is None:
            n_m3_dropped += 1
            continue
        n_m3_ok += 1
        if img.computed_compass_angle is None:
            continue
        derived_yaw = camera_yaw_deg(cam)
        diff = abs(derived_yaw - img.computed_compass_angle) % 360.0
        if diff > 180.0:
            diff = 360.0 - diff
        yaw_errors_deg.append(diff)

    print(f"  M3 produced {n_m3_ok} cameras, dropped {n_m3_dropped} images")
    if not yaw_errors_deg:
        print("  FAIL: no images had both a Camera and a compass — can't check")
        return
    yaw_errors_deg.sort()
    n = len(yaw_errors_deg)
    print(f"  Yaw error vs computed_compass_angle on {n} cameras:")
    print(f"    min    : {yaw_errors_deg[0]:.2f} deg")
    print(f"    median : {yaw_errors_deg[n // 2]:.2f} deg")
    print(f"    mean   : {statistics.mean(yaw_errors_deg):.2f} deg")
    print(f"    max    : {yaw_errors_deg[-1]:.2f} deg")
    if n >= 4:
        p25 = yaw_errors_deg[n // 4]
        p75 = yaw_errors_deg[(3 * n) // 4]
        print(f"    p25/p75: {p25:.2f} / {p75:.2f} deg")

    median = yaw_errors_deg[n // 2]
    print()
    if median < 5.0:
        print(f"  PASS: median yaw error {median:.2f} deg < 5 deg")
        print("        OpenSfM = ENU + w2c assumption confirmed on real data")
    elif median < 15.0:
        print(f"  MARGINAL: median {median:.2f} deg — frame is mostly right but"
              " investigate before SFB")
    else:
        print(f"  FAIL: median yaw error {median:.2f} deg — frame assumption"
              " broken, M3 needs a fix")
        print("        likely candidates: c2w/w2c reversed, axis swap, or"
              " sign flip")

    print()
    print("=" * 72)
    print("  Done.")
    print("=" * 72)


if __name__ == "__main__":
    main()
