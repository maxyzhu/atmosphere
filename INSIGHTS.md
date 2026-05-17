# Records of Observation

## 2026-04-23 Day 2: OSM Building Retrieval and Visualization
All tests passed.
The shape of the 2D map is pretty accurate, but the height data is not thorough as expected. In "temp_file/47_6059_-122_3392_dlr_group.png", the upper right right sides doesn't have height data, but in Google Map, they are all 3D buildings, including some high-rise. 54% data precision is not good. But it can be a more powerful statement that imperfect data can also create good scene with the power of world model and Mapillary iamges.

## 2026-04-25 Day 3: Fetch Mapillary Data Points
The algorithm works and return a workable graphic.
Recursively divide the sample area if request returns Bad 500. But in dense area, recursion is too deep and hard to scale.
sample count of the greedy algorithm (farthest_point_sample) is not ideal. We need sample variety instead of an expected number.
TODO or Need to fix:
1. **Runtime cost**: 15 minutes per query is unsustainable for Phase 1
    SFB (20 coords = 5 hours). Optimizations available:
    - Reduce per-request timeout from 30s to 10s
    - Parallelize sub-bbox requests via asyncio/threading (3-9x speedup)
    - Cache "giving up" bboxes to skip on rerun
    - Density-aware initial split (start 4×4 in known dense areas)
2. **Hardcoded target_count=100**: arbitrary. Better future API:
    `max_count=N, min_separation_m=M` so sampling adapts to true density.
3. **Mapillary vector tile API**: alternative source (tiles.mapillary.com)
    has no timeout, but requires mapbox-vector-tile parsing. Worth
    investigating as a hybrid (tile=ID list, bbox=metadata) pattern.
    **Two-API hybrid strategy:**
    1. Vector Tile API (`tiles.mapillary.com/maps/vtp/mly1_public/{z}/{x}/{y}`):
      pre-cached, no timeout, returns sparse data — image IDs + positions only
    2. Graph API by-ID (`graph.mapillary.com/{image_id}`):
      single-ID queries don't timeout. Use only on the K sampled IDs.

    **Sampling consequence:**
    - Vector tiles lack compass_angle, captured_at, etc.
    - Stage 1 farthest-point samples must use spatial-only distance
    - Spatial-only FPS is acceptable because Mapillary's dashcam data has
      spatially-correlated compass directions (cars going down a street
      have similar headings)
    - After Stage 2 fetches per-image metadata, can optionally re-FPS with
      full spatial+compass distance to refine to a smaller K'

    **Required deps:** mercantile, mapbox-vector-tile (or vt2geojson)

    **Estimated improvement:** 15 min → ~30-60 sec per query

## 2026-04-26 Day 3.1 · Sampling design must adapt to neighborhood facade-exposure

Earlier insight ("uniform facade coverage is impossible") was overstated.
Refined understanding:

Facade exposure to street imagery varies by neighborhood typology:
- Perimeter-block commercial (Pike Place, SoHo, Back Bay): ~95% facades
  street-visible. Full coverage achievable in principle.
- Mixed-use mid-density (Belltown around DLR): ~70-80%.
- Residential / suburban: 30-50%.

Implication for SCP design:
- Cannot use a single sampling strategy for all neighborhoods
- The system must adapt: dense areas pursue facade-level coverage,
  sparse areas accept partial coverage with style fallback
- Hardcoded constants (target_count=100, square bbox) work against this:
  they produce the same output regardless of density

Phase 1 algorithmic priorities:
1. Replace square bbox with circular post-filter (semantic correctness)
2. Replace fixed target_count with density-aware termination
   (e.g., "stop when minimum pairwise distance drops below D meters")
3. (Phase 2) Replace position-based FPS with visibility-based set cover:
   ray-cast from each candidate camera against LoD2 building footprints,
   maximize minimum per-facade coverage

The research contribution is precisely this adaptivity, not raw coverage.

## 2026-04-29 Day 3.2 · Use Mapbox Vector Tile to Fetch Mapillary Data Point
1. Data Source Switch: Graph API search → Vector Tiles
Old: queried graph.mapillary.com/images?bbox=... (entity search endpoint)
New: fetch tiles.mapillary.com/maps/vtp/mly1_computed_public/2/{z}/{x}/{y} at zoom 14, decode locally with mapbox-vector-tile
Why: tiles are pre-sharded so they don't time out in dense areas; cacheable by (z, x, y) so neighboring queries reuse data automatically; mly1_computed_public provides SfM-corrected positions (more accurate than raw GPS).
2. Removed
  a. _raw_fetch: bbox search call
  b. _fetch_recursive + _split_bbox: unnecessary for tiles
  c. api_limit parameter: tiles return a fixed payload per (z, x, y), no global cap
  d. MapillaryImage fields: captured_at, sequence_id, is_pano — not used downstream yet
  e. The "skip image if captured_at missing" filter — we don't read the field anymore
3. Query Region Redefinition: circular radius search → square bbox with buffer
Effective fetch region: a square of side 2 * (radius_m + 20m) — the 20m buffer gives generated imagery edge margin so artifacts don't intrude into the ROI
OSM fetch (features_from_bbox) uses the same square, so buildings and image candidates live in identical coordinates
4. Target Count: From hardcoded 100 → area-density formula
5. Density: 20 images per 100m × 100m = 0.002 images/m²
target_count = side² × density, computed automatically when not specified
Example: radius_m=150 → side = 340m → target ≈ 1156
Falls back to "take all" when the candidate pool is smaller than target (sparse areas degrade gracefully)
FPS Sampling
Metric unchanged: α · spatial_distance + β · compass_distance with α=1, β=0.44
6. Seed changed: random (with seed param) → deterministically the image closest to ENU origin (= bbox center). 
Fully reproducible, anchored on the user's query point.
7. Caching: per-query JSON file (raw_{lat}_{lon}_r{R}.json) → per-tile pbf (tiles/{z}_{x}_{y}.pbf)
Tiles are immutable infrastructure data; cache them once, reuse forever across all overlapping queries
8. Graph API Now Optional
Vector tile alone provides everything needed for sampling: id, geometry, compass_angle
Graph API only invoked when download_thumbnails=True, solely to retrieve signed thumb_*_url (which can't be cached because URLs expire)
9. Setting download_thumbnails=False → zero Graph API calls, pure tile pipeline
10. visualize_neighborhoods options:
Required: --lat --lon
Optional: 
--radius (150m)
--stage (mapillary)
--out (None)
--mapillary-limit (None)
--no-download (False)
--no-cache (False)
--title (autogen)
--verbose/-v (False)
--list-stage
--help/-h

## 2026-05-11 Day 4 (planning) · Research framing pivot: generation → reconstruction

Today's central insight, recorded before any Day 4 code: **Atmosphere's
research contribution is reconstruction with GIS priors, not generation
from GIS conditioning.** The distinction matters and reshapes Phase 0.

### What changed

The spec (atmosphere_spec.md v0.1) framed M5 as "WorldGen wrapper that
consumes SCPBundle" — implicitly assuming a generation backbone that
takes geometric conditioning and synthesizes a scene. Two questions
broke that frame:

1. **Does WorldGen actually exist?** No specific open-source project
   matched the name. "WorldGen" was a placeholder for "some
   FLUX-based 3DGS generator" inherited from earlier conversations.
2. **Do any open world models accept multi-image input?** Yes: HY-World 2.0
   (Tencent, 2026-04-15) does, but only its **WorldMirror 2.0** component
   is open-sourced — the multi-view *reconstruction* module, not the
   generation modules (HY-Pano 2.0, WorldStereo 2.0, WorldNav are
   referenced in the paper, weights not yet released).

Lyra 2.0 (NVIDIA, 2025-09) was the obvious alternative — Apache 2.0,
3DGS output, well-documented. But Lyra is a **generation** model: it
takes a single image + a synthetic camera trajectory, hallucinates a
fly-through video, and reconstructs 3DGS from its own hallucinations.
Multi-image input isn't supported; the multi-view inside Lyra is
generated, not observed.

This forced the question: which paradigm fits Atmosphere?

### Reconstruction vs generation

| Paradigm | Input | Strength | Atmosphere fit |
|---|---|---|---|
| Generation (Lyra, HY-Pano) | Single image / text | Imagination from sparse priors | Poor — we have rich observations, not a single hint |
| Reconstruction (WorldMirror) | Multi-view real images + optional camera/depth priors | Fidelity to observed geometry | Strong — Mapillary *is* the multi-view; OSM *is* the prior |

**Mapillary gives us a real observation path with structured gaps.**
The images are sparse (street-bound), but the gaps are predictable
(building backs, courtyards, roofs), and the gaps are exactly where
OSM LoD1 footprints + heights constrain the geometry. Reconstruction
with GIS priors is the natural fit; pure generation throws away the
observations we paid to retrieve.

### WorldMirror 2.0's prior interface

WorldMirror's Python API turns out to be designed for exactly our case:

```
pipeline(
    input_path='images/',          # directory of Mapillary thumbnails
    prior_cam_path='camera.json',  # our M3 output
    prior_depth_path='depth/',     # our M2 output
)
```

The `prior_cam_path` JSON schema accepts a list of 4×4 extrinsics and
3×3 intrinsics matrices; missing priors are passed as empty lists and
the model falls back to internal estimation. This is exactly the A/B
slot Atmosphere needs: same Mapillary images, toggle priors on/off,
measure reconstruction quality difference. The SFB benchmark gets a
natural design from this: three conditions (no prior / pose only /
pose+depth) on a fixed coordinate set, measure delta on building-back
region fidelity.

VRAM: ~12–24 GB for WorldMirror 2.0 inference (BF16 + FSDP offload
options for tighter budgets). RTX 4090 sufficient — don't need A100.

### What this means for M1–M6 in the spec

- **M1 (Retrieval)**: unchanged. Already complete.
- **M2 (Canonicalization)**: refocused. M2's depth render now feeds
  `prior_depth_path` directly, not a hand-rolled WorldGen conditioning
  bundle. trimesh + pyrender suffices for Phase 0 (nvdiffrast deferred;
  Mac has no NVIDIA path anyway). Output: float32 .npy per camera view,
  one per Mapillary image.
- **M3 (Cross-Modal Alignment)**: refocused. M3 outputs the 4×4
  extrinsics that go into `prior_cam_path`. Phase 0 "baseline M3" can
  just compute c2w from Mapillary GPS + compass_angle directly (no
  optimization); Phase 1 M3 uses LoD1 silhouette matching to refine
  poses from 5–10m to sub-meter.
- **M4 (SCP Bundle)**: simplified to a thin schema mapping our internal
  types to WorldMirror's expected JSON / file layout. Not a novel data
  structure — just an adapter.
- **M5 (Generation → Reconstruction)**: a wrapper around
  `WorldMirrorPipeline`, not a custom diffusion harness. The complexity
  budget for M5 collapses by an order of magnitude.
- **M6 (Evaluation)**: now has a clean three-condition design built
  into the backbone's API, not bolted on.

### Risk surfaced today

WorldMirror is *only* reconstruction — it won't fill structurally
unobserved regions (building backs that no Mapillary camera ever
faced). The output will have holes. Phase 0 accepts this as a known
limitation; Phase 2 — once WorldStereo 2.0 or an equivalent
open-source generator with multi-view conditioning is available — can
hybridize "reconstruct where observed, generate where occluded." This
is genuinely a research gap with no current open-source solution.

### Reframe for narrative / NIW

Not "AEC tool that generates worlds from GIS." Instead: **"Spatial RAG
for 3D world reconstruction: given sparse, noisy real-world
observations (street imagery) and an authoritative geometric prior
(GIS), how should they be combined to maximize the fidelity of a
reconstructed 3D scene?"** The retrieval/augmentation/generation
three-layer framing survives; the "generation" backend is just one
model type and currently the reconstruction variant is the one that
works.

## 2026-05-11 Day 4 (impl, part 1) · Code cleanup before M2

Three concrete code changes today, all aimed at unblocking M2 and the
WorldMirror integration that follows.

### Phase A patches

1. **`is_pano` restored to `MapillaryImage`.** Removed during Day 3.2 as
   unused; reinstated because WorldMirror's i2s-style modes prefer
   panorama input over perspective. The field is free from the vector
   tile properties, so cost is zero. The original deletion was
   shortsighted — "unused right now" is not the same as "will stay
   unused."
2. **`_FIELDS_THUMB` bumped from `thumb_256_url` to `thumb_2048_url`.**
   256 px is too low for meaningful visual conditioning into a world
   model. Mapillary serves up to 2048 px publicly. Verify on first
   download that the larger URLs actually work — some older images may
   only have 256/1024.
3. **`ground_elevation_m: float | None = None`** added as optional field
   on both `Building` and `MapillaryImage`. Phase 0 sets it to None
   (flat-earth z=0 assumption acceptable for PoC). The field exists now
   so that when Phase 1's DEM integration arrives, dataclass shape
   doesn't change and the codebase doesn't fork. Pure architectural
   hygiene.

### Test suite repair

`test_mapillary.py` was three commits stale: it imported `_radius_to_bbox`
(replaced by `_square_bbox_with_buffer`), constructed `MapillaryImage`
with a `captured_at` field that no longer exists, and called
`_farthest_point_sample(..., seed=...)` with a parameter that was removed
when FPS became deterministically center-seeded. Rewrote against the
current vector-tile pipeline: 17 tests covering bbox math, density
formula, FPS edge cases (center seeding, missing compass, full
determinism), and end-to-end retrieval with `_fetch_tile_bytes` /
`_decode_image_features` patched.

`test_stages.py` lacked any coverage of `StreetStage`. Added 4 tests
(population, network_type option pass-through, default network_type="drive",
earlier-stage preservation) plus a `STAGE_ORDER` invariant test ensuring
streets sit between buildings and Mapillary in the layer stack.

Day 3 leftover: `test_buildings.py` patched `ox.features_from_point`,
but Day 3.2 had migrated `buildings.py` to `ox.features_from_bbox`. The
mocks silently no-op'd and tests hit the real Overpass API, returning
18 real Pioneer Square buildings instead of the 2 fake ones the
assertions expected. Repointed the patch target; tests pass.

Final state: 84 / 84 passing in 0.39 s.

### Lesson

Three-tests-failing-on-a-clean-clone is a smell, not a passing grade.
The failures were all stale-mock issues, not logic bugs, but they were
hiding because nobody re-ran the suite after Day 3's refactor. **Phase
0 going forward: every retrieval-layer refactor must end with
`uv run pytest -v` green, not "green for the new module."**

## 2026-05-14 Day 4 (planning, part 2) · WorldMirror priors are real; Mapillary's API gives us more than we thought

Two source verifications done before writing M2:

**1. WorldMirror's `prior_cam_path` and `prior_depth_path` are actually consumed.**
Read pipeline.py + inference_utils.py. Both loaders return real tensors
(`[1, N, H, W]` depth, `[1, N, 4, 4]` extrinsics). Format details that
matter: `.npy` float32 in meters; sky as `np.inf` (loader coerces to 0
via `nan_to_num`); filename match by image stem; all-or-nothing — any
missing prior drops the whole batch.

**2. Mapillary Graph API returns more than GPS + compass.** Probed two
real images. Each carries `camera_parameters` ([focal_ratio, k1, k2]),
`camera_type`, `computed_geometry`, `computed_compass_angle`,
`computed_rotation` (axis-angle, full 3-DoF pose), and `atomic_scale`.
Per-image focal varies ~2× between the two probed samples, so M2 must
render with per-image K, not a hardcoded FOV. Caveats: raw
`compass_angle` is unreliable EXIF fallback (always use computed);
`computed_altitude` is SfM-internal, not real elevation (Phase 0 fixes
z = 1.5 m); `computed_rotation` lives in SfM frame not ENU, requiring
calibration on Day 5.

**Framing consequence:** Atmosphere is not "a better SfM" — Mapillary
already shipped a production SfM. The contribution is anchoring world
model output to absolute, independently-sourced LoD geometry, which SfM
alone cannot do (locally consistent, globally arbitrary). M3 reduces to
a thin adapter (read API → rotvec → JSON, ≤ 100 LOC). SFB's A/B becomes:
Mapillary-only baseline vs Mapillary + OSM LoD anchor.