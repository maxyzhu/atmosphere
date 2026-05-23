"""
Diagnose why some M2 depth renders have ~0% building hits.

Looks at each surviving camera's pose and asks:
- where is the camera in ENU?
- what direction is it facing?
- are there OSM buildings in that direction within ~80 m?

A 0%-hit camera with buildings all around it indicates one of:
- camera rotation wrong (looking at sky / into ground)
- focal_ratio wrong (FOV too narrow, missing visible buildings)
- camera_z_m wrong (way above buildings)
- buildings filter dropped some
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from atmosphere.canonicalize.align import batch_mapillary_to_cameras, camera_yaw_deg
from atmosphere.canonicalize.render import render_depth
from atmosphere.retrieval.buildings import fetch_buildings
from atmosphere.retrieval.mapillary import fetch_mapillary_images


DLR_LAT = 47.6059
DLR_LON = -122.3392
RADIUS_M = 150.0


def main() -> None:
    logging.basicConfig(level=logging.WARNING)

    images = fetch_mapillary_images(
        DLR_LAT, DLR_LON, radius_m=RADIUS_M, target_count=24,
        download_thumbnails=False, fetch_camera_metadata=True,
        fetch_camera_z=True,
    )
    buildings = fetch_buildings(DLR_LAT, DLR_LON, radius_m=RADIUS_M)
    cameras, surviving = batch_mapillary_to_cameras(images)

    print(f"19 cameras, 38 buildings\n")

    # Build a simple building-centroid catalog for distance queries.
    bldg_centroids: list[tuple[float, float, float]] = []
    bldg_heights: list[float | None] = []
    for b in buildings:
        ec = float(np.mean(b.footprint_enu[:, 0]))
        nc = float(np.mean(b.footprint_enu[:, 1]))
        # Use max extent as a coarse radius for the next-building heuristic
        ext = float(np.max(np.linalg.norm(b.footprint_enu - [ec, nc], axis=1)))
        bldg_centroids.append((ec, nc, ext))
        bldg_heights.append(b.height_m)

    # Pre-render to get hit counts
    print(f"{'idx':>3}  {'id':<14}  {'east':>7}  {'north':>7}  {'z':>6}  "
          f"{'yaw':>6}  {'pitch':>6}  {'fov_h':>6}  {'hits%':>6}  "
          f"{'nearest_b':>10}")
    print("-" * 100)

    for i, (cam, img) in enumerate(zip(cameras, surviving)):
        # Render depth, count hits.
        depth = render_depth(buildings, cam)
        n_hits = int((depth < 1000.0).sum())
        hit_pct = 100.0 * n_hits / depth.size

        # Camera pose summary.
        e, n, z = cam.position_enu
        yaw = camera_yaw_deg(cam)
        # Camera forward in ENU (3-vector). Pitch from horizontal:
        fwd = cam.forward_world
        horiz = np.sqrt(fwd[0]**2 + fwd[1]**2)
        pitch = float(np.degrees(np.arctan2(fwd[2], horiz)))

        # Horizontal FOV (degrees).
        fov_h_deg = float(np.degrees(2 * np.arctan2(cam.width / 2, cam.fx)))

        # Nearest building in front (within ±60° of yaw direction).
        nearest_d = float("inf")
        for (be, bn, bext) in bldg_centroids:
            de = be - e
            dn = bn - n
            d = (de * de + dn * dn) ** 0.5
            if d < 1e-6:
                continue
            # Angle from camera position to building, in same compass
            # convention (deg CW from north).
            ang = (np.degrees(np.arctan2(de, dn))) % 360
            ang_diff = min(abs(ang - yaw), 360 - abs(ang - yaw))
            if ang_diff <= 60:  # roughly in front
                # Approximate edge distance
                d_edge = max(0.0, d - bext)
                if d_edge < nearest_d:
                    nearest_d = d_edge

        nearest_str = f"{nearest_d:.1f}m" if nearest_d != float("inf") else "n/a"

        marker = "  <-- LOW" if hit_pct < 10.0 else ""
        print(f"{i:>3}  {img.mapillary_id[:12]:<14}  "
              f"{e:>7.1f}  {n:>7.1f}  {z:>6.2f}  "
              f"{yaw:>6.1f}  {pitch:>+6.1f}  {fov_h_deg:>6.1f}  "
              f"{hit_pct:>5.1f}%  {nearest_str:>10}{marker}")


if __name__ == "__main__":
    main()
