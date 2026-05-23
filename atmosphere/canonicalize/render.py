"""
M2 — Render synthetic depth maps from building geometry.

Pipeline: list[Building] → extruded mesh → Open3D RaycastingScene →
float32 depth map [H, W], meters, SKY_DEPTH_M for sky/miss.

Output format matches WorldMirror 2.0's `prior_depth_path` expectations
(see hyworld2/worldrecon/hyworldmirror/utils/inference_utils.py:285).
Sky pixels are written as the SCP-level SKY_DEPTH_M constant
(currently 1000 m) — a large finite value, far beyond any plausible
street-level scene scale. WorldMirror's load_prior_depth applies
`np.nan_to_num(depth, nan=0, posinf=0, neginf=0)` and would silently
coerce np.inf to 0 (= depth at camera origin), so M2 must emit a
finite value at the source. See changelog Day 5 Part 1 §1.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from shapely.geometry import Polygon

from atmosphere.canonicalize.camera import Camera
from atmosphere.retrieval.buildings import Building
from atmosphere.scp import SKY_DEPTH_M


# Default height for buildings with height_m=None. Phase 0 only — once
# DEM + LoD2 data is wired in (Phase 1), missing heights should be
# inferred per-region rather than constant.
_DEFAULT_HEIGHT_M = 10.0


def buildings_to_mesh(buildings: list[Building]) -> trimesh.Trimesh:
    """
    Extrude each Building footprint into a 3D box and concatenate.

    Footprints are 2D polygons in ENU (east, north); extrusion is
    along +z (up) from z=0 to z=height_m. Buildings with unknown
    height get _DEFAULT_HEIGHT_M; this is conservative for downtown
    test areas.

    Returns a single trimesh.Trimesh combining all buildings; an empty
    mesh if the input list is empty.
    """
    if not buildings:
        return trimesh.Trimesh()

    meshes: list[trimesh.Trimesh] = []
    for b in buildings:
        height = b.height_m if b.height_m is not None else _DEFAULT_HEIGHT_M

        # Trimesh's extrude_polygon requires a shapely.Polygon. The
        # footprint_enu array is closed (last point == first), which
        # shapely accepts but doesn't require.
        try:
            poly = Polygon(b.footprint_enu[:, :2])
            if not poly.is_valid:
                # Self-intersecting footprint — try to fix with buffer(0).
                poly = poly.buffer(0)
                if poly.is_empty:
                    continue
            mesh = trimesh.creation.extrude_polygon(poly, height)
            meshes.append(mesh)
        except Exception:
            # Pathological footprint — skip rather than abort the whole
            # render. M2 fails soft; provenance is in the Building.
            continue

    if not meshes:
        return trimesh.Trimesh()

    return trimesh.util.concatenate(meshes)


def _trimesh_to_open3d(mesh: trimesh.Trimesh) -> o3d.t.geometry.TriangleMesh:
    """Convert a trimesh to an Open3D tensor TriangleMesh."""
    if len(mesh.vertices) == 0:
        # Empty placeholder — Open3D won't accept zero verts, so give
        # it a degenerate but valid mesh.
        verts = np.zeros((3, 3), dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.uint32)
    else:
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.uint32)

    o3d_mesh = o3d.t.geometry.TriangleMesh()
    o3d_mesh.vertex.positions = o3d.core.Tensor(verts)
    o3d_mesh.triangle.indices = o3d.core.Tensor(faces)
    return o3d_mesh


def _camera_to_rays(camera: Camera) -> o3d.core.Tensor:
    """
    Build a (H*W, 6) ray tensor: [origin_xyz, direction_xyz] per pixel.

    Each pixel (u, v) corresponds to a ray from the camera center in
    the direction of the back-projected pixel through the pinhole.
    Directions are in the world (ENU) frame.
    """
    if camera.camera_type != "perspective":
        raise NotImplementedError(
            f"camera_type='{camera.camera_type}' not supported in Phase 0"
        )

    H, W = camera.height, camera.width

    # Pixel grid: (u, v) where u is column (x_image), v is row (y_image).
    # We use the pixel-center convention (+0.5), which matches OpenCV.
    u = np.arange(W, dtype=np.float64) + 0.5
    v = np.arange(H, dtype=np.float64) + 0.5
    uu, vv = np.meshgrid(u, v)  # both (H, W)

    # Back-project to camera-frame rays. Camera convention: +X right,
    # +Y down, +Z forward. The ray for pixel (u, v) in camera frame:
    #   x_cam = (u - cx) / fx
    #   y_cam = (v - cy) / fy
    #   z_cam = 1
    x_cam = (uu - camera.cx) / camera.fx
    y_cam = (vv - camera.cy) / camera.fy
    z_cam = np.ones_like(x_cam)

    # Stack to (H, W, 3) then rotate to world frame.
    dirs_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)  # (H, W, 3)
    dirs_world = dirs_cam @ camera.rotation.T  # (H, W, 3)
    # Normalize. Open3D raycasting accepts unnormalized, but normalizing
    # makes the returned t_hit equal to true Euclidean distance, which
    # we then convert to camera-frame depth (Z) below.
    norms = np.linalg.norm(dirs_world, axis=-1, keepdims=True)
    dirs_world = dirs_world / norms

    # Ray origins: camera center, repeated for every pixel.
    origin = np.asarray(camera.position_enu, dtype=np.float64)
    origins = np.broadcast_to(origin, (H, W, 3)).copy()

    # Flatten to (H*W, 6) and convert to Open3D tensor.
    rays = np.concatenate([origins, dirs_world], axis=-1).reshape(-1, 6)
    return o3d.core.Tensor(rays.astype(np.float32))


def render_depth(
    buildings: list[Building],
    camera: Camera,
) -> np.ndarray:
    """
    Render a depth map of `buildings` from `camera`'s viewpoint.

    Returns a float32 array of shape (H, W) where each pixel value is
    the camera-frame Z depth in meters; pixels with no building hit
    are set to np.inf (sky / miss).

    Empty `buildings` returns an all-inf array — a valid prior_depth
    map saying "no building constraints anywhere", which WorldMirror
    will accept and effectively ignore.
    """
    H, W = camera.height, camera.width

    # Edge case: no buildings → all sky.
    if not buildings:
        return np.full((H, W), SKY_DEPTH_M, dtype=np.float32)

    mesh = buildings_to_mesh(buildings)
    if len(mesh.vertices) == 0:
        return np.full((H, W), SKY_DEPTH_M, dtype=np.float32)

    # Build the raycasting scene and cast.
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(_trimesh_to_open3d(mesh))
    rays = _camera_to_rays(camera)
    hits = scene.cast_rays(rays)

    # t_hit is Euclidean distance along the (normalized) ray. We want
    # camera-frame Z depth. Since rays are normalized and we know each
    # ray's camera-frame Z component, we recover depth as:
    #   depth_z = t_hit * |dir_world . forward_world|
    # But there's a simpler formulation: for a normalized ray d, the
    # Z-component in camera frame is the cosine between d_world and the
    # camera forward axis. We computed that implicitly above; recompute
    # cleanly here for clarity.
    t_hit = hits["t_hit"].numpy().reshape(H, W)  # Euclidean distance

    # Recover the cos(angle from optical axis) per pixel for the
    # Euclidean-to-Z conversion. This was 1 / |dir_cam_before_norm|.
    u = np.arange(W, dtype=np.float64) + 0.5
    v = np.arange(H, dtype=np.float64) + 0.5
    uu, vv = np.meshgrid(u, v)
    x_cam = (uu - camera.cx) / camera.fx
    y_cam = (vv - camera.cy) / camera.fy
    inv_norm = 1.0 / np.sqrt(x_cam**2 + y_cam**2 + 1.0)
    # If r is normalized world dir derived from (x_cam, y_cam, 1) of
    # norm 1/inv_norm, then the camera-frame Z of r is just inv_norm.
    depth_z = t_hit * inv_norm

    # No-hit pixels: Open3D returns +inf for t_hit. Replace with the
    # finite SCP sky constant so WorldMirror's nan_to_num doesn't
    # silently coerce them to 0 (= depth at camera origin).
    depth_z = np.where(np.isfinite(depth_z), depth_z, SKY_DEPTH_M)

    return depth_z.astype(np.float32)


def render_depth_batch(
    buildings: list[Building],
    cameras: list[Camera],
    output_dir: Path,
    image_stems: list[str],
) -> list[Path]:
    """
    Render depth maps for many cameras and save them as .npy files.

    Output file naming matches WorldMirror's expectation: each .npy is
    named `{stem}.npy` where stem is the corresponding image filename
    without extension (see load_prior_depth in inference_utils.py).

    Returns the list of written file paths, in the same order as cameras.
    """
    if len(cameras) != len(image_stems):
        raise ValueError(
            f"cameras ({len(cameras)}) and image_stems ({len(image_stems)}) "
            "must have equal length"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build the scene once — buildings don't change between cameras.
    mesh = buildings_to_mesh(buildings)
    if len(mesh.vertices) == 0:
        # All-sky output for each camera.
        written: list[Path] = []
        for cam, stem in zip(cameras, image_stems):
            depth = np.full((cam.height, cam.width), SKY_DEPTH_M, dtype=np.float32)
            path = output_dir / f"{stem}.npy"
            np.save(path, depth)
            written.append(path)
        return written

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(_trimesh_to_open3d(mesh))

    written = []
    for cam, stem in zip(cameras, image_stems):
        rays = _camera_to_rays(cam)
        hits = scene.cast_rays(rays)
        t_hit = hits["t_hit"].numpy().reshape(cam.height, cam.width)

        u = np.arange(cam.width, dtype=np.float64) + 0.5
        v = np.arange(cam.height, dtype=np.float64) + 0.5
        uu, vv = np.meshgrid(u, v)
        x_cam = (uu - cam.cx) / cam.fx
        y_cam = (vv - cam.cy) / cam.fy
        inv_norm = 1.0 / np.sqrt(x_cam**2 + y_cam**2 + 1.0)
        depth_z = t_hit * inv_norm
        depth_z = np.where(
            np.isfinite(depth_z), depth_z, SKY_DEPTH_M,
        )
        depth_z = depth_z.astype(np.float32)

        path = output_dir / f"{stem}.npy"
        np.save(path, depth_z)
        written.append(path)

    return written