"""
Tests for atmosphere.scp.bundle (M4).

What's covered here
-------------------
- assemble_bundle writes the right files in the right places:
  images/{stem}.jpg, prior_depth/{stem}.npy, prior_cam.json, manifest.json
- prior_cam.json round-trips through WorldMirror's expected schema:
  extrinsics[].camera_id matches stem, .matrix is 4x4 c2w, intrinsics[]
  .matrix is 3x3 K.
- Sky depth sanitization: np.inf, -np.inf, np.nan are all replaced
  with SKY_DEPTH_M before writing.
- Drop policy:
  - missing thumb_path -> drop, n_dropped_m4 incremented in manifest
  - depth shape mismatch -> drop and warn
- Positional alignment guard: image/camera/depth lists of different
  lengths raise ValueError.
- Manifest carries provenance fields (query coords, drop counts,
  sky_depth_m, etc) for SFB benchmark stratification.

What's *not* covered
--------------------
- Real WorldMirror inference round-trip (M5). Verified separately
  during the end-to-end RunPod run.
- Performance under large batches (24+ images). The implementation
  is straight O(N) and shouldn't degrade.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from atmosphere.canonicalize.camera import Camera
from atmosphere.retrieval.mapillary import MapillaryImage
from atmosphere.scp.bundle import (
    SKY_DEPTH_M,
    SCPBundle,
    assemble_bundle,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_thumb_dir(tmp_path: Path) -> Path:
    """A directory where fixture images can be written."""
    d = tmp_path / "thumbs"
    d.mkdir()
    return d


def _make_camera(
    *, position: tuple[float, float, float] = (0.0, 0.0, 1.5),
    compass: float = 0.0,
    width: int = 1920,
    height: int = 1080,
) -> Camera:
    return Camera.from_heading(
        position_enu=position,
        compass_angle_deg=compass,
        focal_ratio=0.5,
        width=width,
        height=height,
    )


def _make_image_with_thumb(
    thumb_dir: Path,
    *,
    mapillary_id: str = "img1",
    width: int = 1920,
    height: int = 1080,
) -> MapillaryImage:
    """Create an image whose thumb_path points to a tiny real JPG."""
    thumb_path = thumb_dir / f"{mapillary_id}.jpg"
    # Make a 1-byte placeholder; shutil.copy2 doesn't care about format.
    thumb_path.write_bytes(b"\xff")
    return MapillaryImage(
        mapillary_id=mapillary_id,
        position_enu=(0.0, 0.0),
        compass_angle_deg=0.0,
        is_pano=False,
        thumb_url="",
        thumb_path=thumb_path,
        computed_rotation=(0.0, 0.0, 0.0),
        focal_ratio=0.5,
        width=width,
        height=height,
        computed_compass_angle=0.0,
        camera_z_m=1.5,
    )


def _make_depth(width: int = 1920, height: int = 1080) -> np.ndarray:
    """A trivial depth array with mixed finite/non-finite values."""
    d = np.full((height, width), 5.0, dtype=np.float32)
    # Inject sky at the top quarter (inf), one NaN pixel, one -inf pixel
    d[: height // 4, :] = np.inf
    d[0, 0] = np.nan
    d[1, 0] = -np.inf
    return d


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_assemble_basic_bundle_writes_all_expected_files(
    tmp_path: Path, sample_thumb_dir: Path,
):
    img = _make_image_with_thumb(sample_thumb_dir)
    cam = _make_camera()
    depth = _make_depth()

    bundle = assemble_bundle(
        bundle_dir=tmp_path / "bundle",
        images=[img],
        cameras=[cam],
        depths=[depth],
        query_lat=47.6059,
        query_lon=-122.3392,
        radius_m=150.0,
    )

    assert isinstance(bundle, SCPBundle)
    assert bundle.n_images == 1
    assert bundle.bundle_dir == tmp_path / "bundle"

    # Expected files exist
    assert (bundle.bundle_dir / "images" / "img1.jpg").exists()
    assert (bundle.bundle_dir / "prior_depth" / "img1.npy").exists()
    assert bundle.prior_cam_path.exists()
    assert bundle.manifest_path.exists()
    assert bundle.prior_cam_path.name == "prior_cam.json"


def test_prior_cam_json_matches_worldmirror_schema(
    tmp_path: Path, sample_thumb_dir: Path,
):
    img = _make_image_with_thumb(sample_thumb_dir)
    cam = _make_camera()
    depth = _make_depth()

    bundle = assemble_bundle(
        bundle_dir=tmp_path / "bundle",
        images=[img],
        cameras=[cam],
        depths=[depth],
        query_lat=0, query_lon=0, radius_m=150.0,
    )

    payload = json.loads(bundle.prior_cam_path.read_text())
    assert "extrinsics" in payload
    assert "intrinsics" in payload
    assert len(payload["extrinsics"]) == 1
    assert len(payload["intrinsics"]) == 1

    extr = payload["extrinsics"][0]
    assert set(extr.keys()) == {"camera_id", "matrix"}
    assert extr["camera_id"] == "img1"
    mat = np.array(extr["matrix"])
    assert mat.shape == (4, 4)
    # Bottom row of a homogeneous c2w
    np.testing.assert_allclose(mat[3], [0.0, 0.0, 0.0, 1.0])
    # The rotation block should match cam.rotation; the translation
    # block should equal cam.position_enu.
    np.testing.assert_allclose(mat[:3, :3], cam.rotation, atol=1e-10)
    np.testing.assert_allclose(mat[:3, 3], cam.position_enu, atol=1e-10)

    intr = payload["intrinsics"][0]
    assert intr["camera_id"] == "img1"
    K = np.array(intr["matrix"])
    assert K.shape == (3, 3)
    np.testing.assert_allclose(K, cam.intrinsic_matrix, atol=1e-10)


def test_depth_sky_pixels_replaced_with_finite_far_value(
    tmp_path: Path, sample_thumb_dir: Path,
):
    img = _make_image_with_thumb(sample_thumb_dir)
    cam = _make_camera()
    depth = _make_depth()

    bundle = assemble_bundle(
        bundle_dir=tmp_path / "bundle",
        images=[img],
        cameras=[cam],
        depths=[depth],
        query_lat=0, query_lon=0, radius_m=150.0,
    )

    written = np.load(bundle.depth_paths[0])
    # No non-finite values survive to disk
    assert np.all(np.isfinite(written))
    # The pixels we made inf/-inf/nan should now equal SKY_DEPTH_M
    assert written[0, 0] == SKY_DEPTH_M  # was nan
    assert written[1, 0] == SKY_DEPTH_M  # was -inf
    assert written[0, 100] == SKY_DEPTH_M  # was inf (top quarter)
    # Mid-image pixel keeps its original finite value
    assert written[500, 500] == 5.0


def test_manifest_records_provenance(
    tmp_path: Path, sample_thumb_dir: Path,
):
    img = _make_image_with_thumb(sample_thumb_dir)
    cam = _make_camera()
    depth = _make_depth()

    assemble_bundle(
        bundle_dir=tmp_path / "bundle",
        images=[img],
        cameras=[cam],
        depths=[depth],
        query_lat=47.6059,
        query_lon=-122.3392,
        radius_m=150.0,
        n_dropped_m3=7,
        cam_height_above_ground_m=1.5,
        notes="test run",
    )

    manifest = json.loads((tmp_path / "bundle" / "manifest.json").read_text())
    assert manifest["bundle_version"] == "scp/v0.1"
    assert manifest["query_lat"] == 47.6059
    assert manifest["query_lon"] == -122.3392
    assert manifest["radius_m"] == 150.0
    assert manifest["n_images"] == 1
    assert manifest["n_dropped_m3"] == 7
    assert manifest["n_dropped_m4"] == 0
    assert manifest["sky_depth_m"] == SKY_DEPTH_M
    assert manifest["cam_height_above_ground_m"] == 1.5
    assert manifest["mapillary_ids"] == ["img1"]
    assert "created_utc" in manifest


def test_multiple_images_preserve_order_and_stems(
    tmp_path: Path, sample_thumb_dir: Path,
):
    imgs = [
        _make_image_with_thumb(sample_thumb_dir, mapillary_id="alpha"),
        _make_image_with_thumb(sample_thumb_dir, mapillary_id="beta"),
        _make_image_with_thumb(sample_thumb_dir, mapillary_id="gamma"),
    ]
    cams = [_make_camera(compass=h) for h in (0.0, 90.0, 180.0)]
    depths = [_make_depth() for _ in range(3)]

    bundle = assemble_bundle(
        bundle_dir=tmp_path / "bundle",
        images=imgs,
        cameras=cams,
        depths=depths,
        query_lat=0, query_lon=0, radius_m=150.0,
    )

    assert bundle.stems == ["alpha", "beta", "gamma"]
    assert [p.stem for p in bundle.image_paths] == ["alpha", "beta", "gamma"]
    assert [p.stem for p in bundle.depth_paths] == ["alpha", "beta", "gamma"]

    payload = json.loads(bundle.prior_cam_path.read_text())
    assert [e["camera_id"] for e in payload["extrinsics"]] == [
        "alpha", "beta", "gamma",
    ]


# ---------------------------------------------------------------------------
# Drop policy
# ---------------------------------------------------------------------------


def test_drops_image_with_no_thumb(
    tmp_path: Path, sample_thumb_dir: Path,
):
    good = _make_image_with_thumb(sample_thumb_dir, mapillary_id="good")
    # No thumb_path: simulate a failed download.
    bad = MapillaryImage(
        mapillary_id="bad",
        position_enu=(1.0, 1.0),
        compass_angle_deg=0.0,
        is_pano=False,
        thumb_url="",
        thumb_path=None,
        computed_rotation=(0.0, 0.0, 0.0),
        focal_ratio=0.5,
        width=1920,
        height=1080,
        computed_compass_angle=0.0,
        camera_z_m=1.5,
    )
    cam = _make_camera()
    depth = _make_depth()

    bundle = assemble_bundle(
        bundle_dir=tmp_path / "bundle",
        images=[good, bad],
        cameras=[cam, cam],
        depths=[depth, depth],
        query_lat=0, query_lon=0, radius_m=150.0,
    )

    assert bundle.n_images == 1
    assert bundle.stems == ["good"]
    manifest = json.loads(bundle.manifest_path.read_text())
    assert manifest["n_dropped_m4"] == 1


def test_drops_image_with_depth_shape_mismatch(
    tmp_path: Path, sample_thumb_dir: Path,
):
    img = _make_image_with_thumb(sample_thumb_dir, width=1920, height=1080)
    cam = _make_camera()
    # Wrong shape: depth shape (100, 100) won't match image (1080, 1920)
    bad_depth = np.full((100, 100), 5.0, dtype=np.float32)

    bundle = assemble_bundle(
        bundle_dir=tmp_path / "bundle",
        images=[img],
        cameras=[cam],
        depths=[bad_depth],
        query_lat=0, query_lon=0, radius_m=150.0,
    )

    assert bundle.n_images == 0
    manifest = json.loads(bundle.manifest_path.read_text())
    assert manifest["n_dropped_m4"] == 1


def test_drops_empty_depth(
    tmp_path: Path, sample_thumb_dir: Path,
):
    img = _make_image_with_thumb(sample_thumb_dir)
    cam = _make_camera()
    empty_depth = np.array([], dtype=np.float32)

    bundle = assemble_bundle(
        bundle_dir=tmp_path / "bundle",
        images=[img],
        cameras=[cam],
        depths=[empty_depth],
        query_lat=0, query_lon=0, radius_m=150.0,
    )

    assert bundle.n_images == 0
    manifest = json.loads(bundle.manifest_path.read_text())
    assert manifest["n_dropped_m4"] == 1


def test_all_dropped_produces_empty_bundle(
    tmp_path: Path, sample_thumb_dir: Path,
):
    bad_img = MapillaryImage(
        mapillary_id="bad",
        position_enu=(0.0, 0.0),
        compass_angle_deg=0.0,
        is_pano=False,
        thumb_url="",
        thumb_path=None,
        computed_rotation=(0.0, 0.0, 0.0),
        focal_ratio=0.5,
        width=1920, height=1080,
        computed_compass_angle=0.0,
        camera_z_m=1.5,
    )
    cam = _make_camera()
    depth = _make_depth()

    bundle = assemble_bundle(
        bundle_dir=tmp_path / "bundle",
        images=[bad_img],
        cameras=[cam],
        depths=[depth],
        query_lat=0, query_lon=0, radius_m=150.0,
    )

    assert bundle.n_images == 0
    # The prior_cam.json should still exist, but be empty
    payload = json.loads(bundle.prior_cam_path.read_text())
    assert payload["extrinsics"] == []
    assert payload["intrinsics"] == []


# ---------------------------------------------------------------------------
# Argument-alignment guard
# ---------------------------------------------------------------------------


def test_misaligned_input_lengths_raise(
    tmp_path: Path, sample_thumb_dir: Path,
):
    img = _make_image_with_thumb(sample_thumb_dir)
    cam = _make_camera()
    depth = _make_depth()

    with pytest.raises(ValueError, match="align"):
        assemble_bundle(
            bundle_dir=tmp_path / "bundle",
            images=[img],
            cameras=[cam, cam],  # length 2 vs 1
            depths=[depth],
            query_lat=0, query_lon=0, radius_m=150.0,
        )
