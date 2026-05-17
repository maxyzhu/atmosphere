"""
M2 demo — render an OSM-derived depth map for one virtual camera.

Usage:
    uv run python scripts/render_demo.py
    uv run python scripts/render_demo.py --lat 47.6090 --lon -122.3416 --heading 90

Defaults to the DLR Group Seattle test coordinate, with a camera at
street level (z=1.5m) looking north. Output goes to output_image/.

This is a Phase 0 sanity check: produces a single depth map from a
synthetic camera, not from a real Mapillary image. M3 (Day 5) will
swap the synthetic camera for real Mapillary metadata.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from atmosphere.canonicalize.camera import Camera
from atmosphere.canonicalize.render import buildings_to_mesh, render_depth
from atmosphere.retrieval.buildings import fetch_buildings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render an OSM depth map for one virtual camera.",
    )
    parser.add_argument("--lat", type=float, default=47.6059,
                        help="Latitude of camera position (default: DLR Group)")
    parser.add_argument("--lon", type=float, default=-122.33880,
                        help="Longitude of camera position (default: DLR Group)")
    parser.add_argument("--heading", type=float, default=180.0,
                        help="Compass heading in degrees (0=N, 90=E, 180=S, 270=W)")
    parser.add_argument("--radius", type=float, default=200.0,
                        help="Building fetch radius in meters")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--focal-ratio", type=float, default=0.5,
                        help="Focal length / max(W, H). 0.5 ≈ 53° FOV.")
    parser.add_argument("--cam-z", type=float, default=1.5,
                        help="Camera height above ground in meters")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("output_image"),
                        help="Directory to write depth.npy + viz.png")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Fetch buildings around the camera location.
    # fetch_buildings places the ENU origin at (lat, lon), so the camera
    # sits at ENU position (0, 0, cam_z).
    print(f"Fetching buildings within {args.radius} m of ({args.lat}, {args.lon})...")
    buildings = fetch_buildings(
        lat=args.lat, lon=args.lon, radius_m=args.radius, use_cache=True,
    )
    print(f"  → {len(buildings)} buildings retrieved")
    if not buildings:
        print("No buildings — depth map will be all sky. Aborting.")
        return

    # Report a height-source breakdown for traceability.
    from collections import Counter
    src_counts = Counter(b.height_source.value for b in buildings)
    print(f"  → height sources: {dict(src_counts)}")

    # 2. Build the virtual camera.
    camera = Camera.from_heading(
        position_enu=(0.0, 0.0, args.cam_z),
        compass_angle_deg=args.heading,
        focal_ratio=args.focal_ratio,
        width=args.width,
        height=args.height,
    )
    print(
        f"Camera: pos={camera.position_enu}, "
        f"heading={args.heading}°, "
        f"fx={camera.fx:.1f}px, "
        f"FOV={np.degrees(2 * np.arctan(args.width / (2 * camera.fx))):.1f}°"
    )

    # 3. Build the mesh (purely for a stat report — render_depth rebuilds it
    # internally, which is wasteful but fine for a one-shot demo).
    mesh = buildings_to_mesh(buildings)
    print(f"Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} triangles")

    # 4. Render.
    print("Rendering...")
    depth = render_depth(buildings, camera)
    finite = np.isfinite(depth)
    hit_pct = 100.0 * finite.sum() / depth.size
    if finite.any():
        d_min, d_max = depth[finite].min(), depth[finite].max()
        d_med = np.median(depth[finite])
        print(
            f"  → {hit_pct:.1f}% pixels hit a building"
            f"   (depth range: {d_min:.1f} – {d_max:.1f} m, median {d_med:.1f} m)"
        )
    else:
        print(f"  → 0% pixels hit a building — try a different heading?")

    # 5. Save .npy (raw, what WorldMirror would consume).
    tag = f"{args.lat:.4f}_{args.lon:.4f}_h{int(args.heading):03d}".replace(".", "_")
    npy_path = args.out_dir / f"depth_{tag}.npy"
    np.save(npy_path, depth)
    print(f"Wrote {npy_path}")

    # 6. Visualize. Replace inf (sky) with NaN so matplotlib treats it
    # as transparent / masked, and the colormap uses only real depths.
    viz = depth.copy()
    viz[~finite] = np.nan

    fig, ax = plt.subplots(figsize=(args.width / 200, args.height / 200), dpi=200)
    im = ax.imshow(viz, cmap="viridis_r")
    ax.set_title(
        f"OSM depth at ({args.lat:.4f}, {args.lon:.4f}), "
        f"heading={args.heading:.0f}°  ({len(buildings)} buildings)"
    )
    ax.set_xlabel("pixel x")
    ax.set_ylabel("pixel y")
    cbar = fig.colorbar(im, ax=ax, label="depth (m)")
    cbar.ax.invert_yaxis()   # so 'near' is at the top of the bar
    fig.tight_layout()

    viz_path = args.out_dir / f"depth_{tag}.png"
    fig.savefig(viz_path, dpi=200)
    plt.close(fig)
    print(f"Wrote {viz_path}")


if __name__ == "__main__":
    main()