"""
Pack a real DLR bundle end-to-end and stop right before WorldMirror
inference. Produces data/bundle_dlr_v0/ matching the WorldMirror
prior_cam_path + prior_depth_path + input_path contract.

Run from repo root:
    uv run python scripts/pack_dlr_bundle.py

Inspect the result:
    ls -la data/bundle_dlr_v0/
    cat data/bundle_dlr_v0/manifest.json
    python -c "import json; j=json.load(open('data/bundle_dlr_v0/prior_cam.json')); \
        print(len(j['extrinsics']), 'cameras'); \
        print('first cam id:', j['extrinsics'][0]['camera_id'])"
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from atmosphere.canonicalize.align import batch_mapillary_to_cameras
from atmosphere.canonicalize.render import render_depth
from atmosphere.retrieval.buildings import fetch_buildings
from atmosphere.retrieval.mapillary import fetch_mapillary_images
from atmosphere.scp.bundle import assemble_bundle


DLR_LAT = 47.6059
DLR_LON = -122.3392
RADIUS_M = 150.0
TARGET_IMAGES = 18                    # RTX 4090 24GB VRAM ceiling — 24/20 OOM in GS rendering, 18 should leave ~1 GB buffer
CAM_HEIGHT_ABOVE_GROUND_M = 1.5
BUNDLE_DIR = Path("data/bundle_dlr_v0")

# Day 5 RunPod-attempt 2: WorldMirror rejects mixed aspect ratios.
# Filter to 16:9 (the dominant aspect in modern Mapillary captures).
TARGET_ASPECT = 16.0 / 9.0


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Clean slate so we don't get stale files from prior runs.
    if BUNDLE_DIR.exists():
        print(f"Removing existing {BUNDLE_DIR} ...")
        shutil.rmtree(BUNDLE_DIR)

    print("\n=== 1. Retrieval (Mapillary + DEM + camera metadata + aspect filter) ===")
    images = fetch_mapillary_images(
        DLR_LAT, DLR_LON,
        radius_m=RADIUS_M,
        target_count=TARGET_IMAGES,
        download_thumbnails=True,
        fetch_camera_metadata=True,
        fetch_camera_z=True,
        cam_height_above_ground_m=CAM_HEIGHT_ABOVE_GROUND_M,
        target_aspect_ratio=TARGET_ASPECT,
    )
    print(f"  -> {len(images)} images retrieved (all aspect {TARGET_ASPECT:.3f})")

    print("\n=== 2. Building footprints (OSM) ===")
    buildings = fetch_buildings(DLR_LAT, DLR_LON, radius_m=RADIUS_M)
    print(f"  -> {len(buildings)} buildings")

    print("\n=== 3. M3: Mapillary -> Camera ===")
    cameras, surviving = batch_mapillary_to_cameras(images)
    n_dropped_m3 = len(images) - len(cameras)
    print(f"  -> {len(cameras)} cameras "
          f"(dropped {n_dropped_m3} of {len(images)})")
    if not cameras:
        print("FATAL: no cameras survived M3; cannot proceed")
        return

    print("\n=== 4. M2: render depth per camera ===")
    depths = []
    for i, (cam, img) in enumerate(zip(cameras, surviving), 1):
        depth = render_depth(buildings, cam)
        depths.append(depth)
        if i % 5 == 0 or i == len(cameras):
            n_hit = int((depth < 1000.0).sum())
            n_total = depth.size
            print(f"  [{i}/{len(cameras)}] {img.mapillary_id[:12]}... "
                  f"hits {n_hit}/{n_total} pixels ({100 * n_hit / n_total:.1f}%)")

    print("\n=== 5. M4: assemble SCP bundle ===")
    bundle = assemble_bundle(
        bundle_dir=BUNDLE_DIR,
        images=surviving,
        cameras=cameras,
        depths=depths,
        query_lat=DLR_LAT,
        query_lon=DLR_LON,
        radius_m=RADIUS_M,
        n_dropped_m3=n_dropped_m3,
        cam_height_above_ground_m=CAM_HEIGHT_ABOVE_GROUND_M,
        notes="DLR Group Seattle, Phase 0 first bundle, Day 5",
    )

    print()
    print("=" * 72)
    print(f"  BUNDLE READY: {bundle.bundle_dir}")
    print("=" * 72)
    print(f"  n_images       : {bundle.n_images}")
    print(f"  prior_cam.json : {bundle.prior_cam_path}")
    print(f"  manifest.json  : {bundle.manifest_path}")
    print(f"  images dir     : {bundle.bundle_dir / 'images'}")
    print(f"  depth dir      : {bundle.bundle_dir / 'prior_depth'}")
    print()
    print("  WorldMirror invocation (run on RunPod):")
    print("  python -m hyworld2.worldrecon.pipeline \\")
    print(f"      --input_path {bundle.bundle_dir / 'images'} \\")
    print(f"      --prior_cam_path {bundle.prior_cam_path} \\")
    print(f"      --prior_depth_path {bundle.bundle_dir / 'prior_depth'} \\")
    print(f"      --output_path output/dlr_v0")
    print()
    print("  A/B baseline (no SCP prior, same images):")
    print("  python -m hyworld2.worldrecon.pipeline \\")
    print(f"      --input_path {bundle.bundle_dir / 'images'} \\")
    print(f"      --output_path output/dlr_v0_no_prior")


if __name__ == "__main__":
    main()
