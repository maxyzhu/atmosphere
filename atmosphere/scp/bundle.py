"""
M4: SCP bundle assembly.

This module is a thin adapter that packs M3-produced Cameras + their
source MapillaryImages + M2-rendered depth into the on-disk layout
WorldMirror's `load_prior_camera` / `load_prior_depth` functions
expect.

Schema (verified against
hyworld2/worldrecon/hyworldmirror/utils/inference_utils.py:286-356)
------
bundle_dir/
├── images/
│   ├── {stem}.jpg          ← copied from MapillaryImage.thumb_path
│   └── ...
├── prior_depth/
│   ├── {stem}.npy          ← float32 [H, W], M2-rendered, sky = 1000.0
│   └── ...
├── prior_cam.json
│   {
│     "extrinsics": [{"camera_id": stem, "matrix": <4x4 c2w>}, ...],
│     "intrinsics": [{"camera_id": stem, "matrix": <3x3 K>}, ...]
│   }
└── manifest.json
    Provenance for SFB benchmark stratification; not consumed by
    WorldMirror, only by Atmosphere's own evaluation code.

Notes on schema correctness
---------------------------
- `matrix` for extrinsics is c2w (4x4). Confirmed by inspection of
  pipeline.py:195, which anchors the first camera to origin via
  `extr = torch.linalg.inv(first) @ extr` — this only makes sense if
  `extr` is c2w.
- `camera_id` matches the image stem (filename without extension).
  WorldMirror falls back to int(camera_id) as an index if it doesn't
  find the stem; we always provide stems matching the image filenames.
- Depth files are float32 .npy (other formats accepted by WorldMirror
  but .npy is the canonical, lossless option and matches M2's output).
- WorldMirror's `nan_to_num(depth, nan=0, posinf=0, neginf=0)` would
  silently coerce our M2 sky=inf pixels to 0 (= depth at camera
  origin) — a geometry contradiction. M4 explicitly replaces non-
  finite depths with `SKY_DEPTH_M` before writing. See changelog Day
  5 Part 1 §1 for the source-code observation that motivated this.
- All-or-nothing prior contract: if any image lacks a complete prior
  set, M4 *drops* that image entirely rather than writing a partial
  record. Logging the drop count is the caller's responsibility
  (batch_mapillary_to_cameras already does it for M3 dropouts).
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from atmosphere.canonicalize.camera import Camera
from atmosphere.retrieval.mapillary import MapillaryImage
from atmosphere.scp import SKY_DEPTH_M

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SCPBundle:
    """
    Handle to an assembled SCP bundle on disk.

    Attributes:
        bundle_dir: Root directory of the bundle.
        image_paths: List of paths to source images (one per surviving
            observation). Order matches camera/depth order.
        depth_paths: Paths to prior_depth .npy files, parallel to images.
        prior_cam_path: Path to prior_cam.json.
        manifest_path: Path to manifest.json (Atmosphere-internal).
        stems: Image stems used as camera_id in prior_cam.json.

    SCPBundle is a frozen view over the disk layout; M5 consumes it by
    pointing WorldMirror's pipeline at bundle_dir.
    """

    bundle_dir: Path
    image_paths: list[Path]
    depth_paths: list[Path]
    prior_cam_path: Path
    manifest_path: Path
    stems: list[str]

    @property
    def n_images(self) -> int:
        return len(self.image_paths)


@dataclass(frozen=True)
class BundleManifest:
    """
    Provenance metadata recorded alongside the WorldMirror-facing files.

    Not consumed by WorldMirror. Used by SFB benchmark (Phase 1) and
    paper write-up to know exactly which inputs produced which output.

    Schema is intentionally loose: this is internal Atmosphere data,
    so adding fields later won't break WorldMirror inference.
    """

    bundle_version: str
    created_utc: str
    query_lat: float
    query_lon: float
    radius_m: float
    n_images: int
    n_dropped_m3: int
    n_dropped_m4: int
    sky_depth_m: float
    cam_height_above_ground_m: float | None = None
    notes: str = ""
    mapillary_ids: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stem_for_image(img: MapillaryImage) -> str:
    """Use the Mapillary ID as the stem — globally unique, no collisions."""
    return img.mapillary_id


def _sanitize_depth_for_worldmirror(depth: np.ndarray) -> np.ndarray:
    """
    Replace non-finite depth values (inf, -inf, NaN) with SKY_DEPTH_M.

    M2's `render_depth` writes `np.inf` for sky and may write `NaN` for
    rays that hit no geometry. Both of these become 0 inside
    WorldMirror (`nan_to_num`), which the model would interpret as
    "depth at the camera origin" — a geometric contradiction. We
    replace them with a finite faraway value so the prior is
    well-defined everywhere.
    """
    if depth.dtype != np.float32:
        depth = depth.astype(np.float32)
    mask = ~np.isfinite(depth)
    if mask.any():
        depth = depth.copy()
        depth[mask] = SKY_DEPTH_M
    return depth


def _serialize_camera_extrinsic(camera: Camera) -> list[list[float]]:
    """c2w 4x4 matrix as a JSON-serializable nested list of floats."""
    return camera.c2w.tolist()


def _serialize_camera_intrinsic(camera: Camera) -> list[list[float]]:
    """K 3x3 matrix as a JSON-serializable nested list of floats."""
    return camera.intrinsic_matrix.tolist()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def assemble_bundle(
    *,
    bundle_dir: Path | str,
    images: list[MapillaryImage],
    cameras: list[Camera],
    depths: list[np.ndarray],
    query_lat: float,
    query_lon: float,
    radius_m: float,
    n_dropped_m3: int = 0,
    cam_height_above_ground_m: float | None = None,
    notes: str = "",
    bundle_version: str = "scp/v0.1",
) -> SCPBundle:
    """
    Write an SCP bundle to disk.

    Inputs are three positionally-aligned lists: images[i], cameras[i],
    depths[i] all refer to the same observation. The caller is
    responsible for that alignment — typically obtained by chaining
    `batch_mapillary_to_cameras` (M3) and `render_depth_batch` (M2),
    both of which preserve order.

    Drop policy
    -----------
    An image is silently dropped from the bundle if:
        - it has no usable thumb_path (thumb download failed or wasn't
          requested), OR
        - its depth is None, OR
        - its depth array shape doesn't match (H, W).
    M4 logs the drop count and reports it in manifest.json. The caller
    can detect 0 surviving images by checking bundle.n_images.

    Args:
        bundle_dir: Destination directory (created if absent). Existing
            files inside are *overwritten*; use a clean dir per run.
        images, cameras, depths: positionally aligned observations.
        query_lat, query_lon, radius_m: query parameters recorded in
            the manifest for SFB reproducibility.
        n_dropped_m3: Count of images dropped *before* this call (by
            M3). Recorded in manifest for accounting; M4 adds its own
            drop count separately.
        cam_height_above_ground_m: Prior used when computing camera_z_m
            upstream; recorded for SFB stratification.
        notes: Free-text field copied into manifest.
        bundle_version: Protocol version; bump on schema changes.

    Returns:
        SCPBundle pointing to the written files.
    """
    if not (len(images) == len(cameras) == len(depths)):
        raise ValueError(
            f"images/cameras/depths must align: "
            f"{len(images)}/{len(cameras)}/{len(depths)}"
        )

    bundle_dir = Path(bundle_dir)
    images_dir = bundle_dir / "images"
    depth_dir = bundle_dir / "prior_depth"
    images_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    out_image_paths: list[Path] = []
    out_depth_paths: list[Path] = []
    out_stems: list[str] = []
    out_extr: list[dict] = []
    out_intr: list[dict] = []
    surviving_ids: list[str] = []

    dropped = 0
    for img, cam, depth in zip(images, cameras, depths, strict=True):
        # Image must be on disk.
        if img.thumb_path is None or not img.thumb_path.exists():
            logger.debug(
                "Bundle: dropping %s (no thumb on disk)", img.mapillary_id,
            )
            dropped += 1
            continue
        # Depth must be present and non-empty.
        if depth is None or depth.size == 0:
            logger.debug(
                "Bundle: dropping %s (empty depth)", img.mapillary_id,
            )
            dropped += 1
            continue
        # Depth shape should match the image's native resolution; if it
        # doesn't, that's an upstream pipeline bug, not an M4 concern.
        # Still log and drop so we don't poison WorldMirror with
        # mismatched priors.
        if img.width is not None and img.height is not None:
            if depth.shape != (img.height, img.width):
                logger.warning(
                    "Bundle: dropping %s (depth shape %s != image %dx%d)",
                    img.mapillary_id, depth.shape, img.width, img.height,
                )
                dropped += 1
                continue

        stem = _stem_for_image(img)

        # Image: copy into the bundle, preserving the source extension.
        # WorldMirror's prepare_input accepts jpeg/jpg/png/webp; thumb
        # downloads are .jpg.
        dst_img = images_dir / f"{stem}{img.thumb_path.suffix}"
        if dst_img.resolve() != img.thumb_path.resolve():
            shutil.copy2(img.thumb_path, dst_img)
        out_image_paths.append(dst_img)

        # Depth: sanitize and write as .npy.
        clean_depth = _sanitize_depth_for_worldmirror(depth)
        dst_depth = depth_dir / f"{stem}.npy"
        np.save(dst_depth, clean_depth)
        out_depth_paths.append(dst_depth)

        out_stems.append(stem)
        out_extr.append({
            "camera_id": stem,
            "matrix": _serialize_camera_extrinsic(cam),
        })
        out_intr.append({
            "camera_id": stem,
            "matrix": _serialize_camera_intrinsic(cam),
        })
        surviving_ids.append(img.mapillary_id)

    # prior_cam.json — the WorldMirror-facing schema, exact field names.
    prior_cam_path = bundle_dir / "prior_cam.json"
    prior_cam_path.write_text(json.dumps(
        {"extrinsics": out_extr, "intrinsics": out_intr},
        indent=2,
    ))

    # manifest.json — Atmosphere-internal provenance.
    manifest = BundleManifest(
        bundle_version=bundle_version,
        created_utc=datetime.now(timezone.utc).isoformat(),
        query_lat=query_lat,
        query_lon=query_lon,
        radius_m=radius_m,
        n_images=len(out_image_paths),
        n_dropped_m3=n_dropped_m3,
        n_dropped_m4=dropped,
        sky_depth_m=SKY_DEPTH_M,
        cam_height_above_ground_m=cam_height_above_ground_m,
        notes=notes,
        mapillary_ids=surviving_ids,
    )
    manifest_path = bundle_dir / "manifest.json"
    manifest_path.write_text(json.dumps(asdict(manifest), indent=2))

    logger.info(
        "SCP bundle written to %s: %d images, %d dropped at M4",
        bundle_dir, len(out_image_paths), dropped,
    )

    return SCPBundle(
        bundle_dir=bundle_dir,
        image_paths=out_image_paths,
        depth_paths=out_depth_paths,
        prior_cam_path=prior_cam_path,
        manifest_path=manifest_path,
        stems=out_stems,
    )
