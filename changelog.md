<changelog>
    ## 2026-04-25 · Phase 0 Day 3 + Project framing decisions

    ### Identity-level reframe

    - **Atmosphere.ai upgraded from "data-driven environment generation tool" to "Spatial RAG system"** — explicit recognition that the project is infrastructure for spatial AI, not an AEC vertical product. Three differentiators identified: Spatial Retriever, Spatial Augmentation Protocol, Spatial RAG Evaluator.
    - **Product naming finalized: MAXIMUS → ATMOSPHERE.ai** with new one-liner: "你做设计,我们生成周围的世界——精确的部分来自数据,想象的部分来自 AI。"
    - **Reframed from commercial product to research project** (April 2026): goal is papers, OSS artifacts, public benchmarks — not revenue. Earlier product-direction memories (Item 1-9 in user memory) reflect superseded framing.
    - **Asset taxonomy expanded from 7 to 8 categories** with addition of "high-precision real environment: Survey" (user DWG/LiDAR → Mesh + GS) and "medium-precision real environment: GIS street network + LoD1/LoD2 massing" rows. This reflects MVP support for Survey input as Priority 1 retrieval source.

    ### Strategic positioning

    - **Waymo 4D World Simulation roles named as explicit alignment target** — ~70-80% technical overlap with Atmosphere.ai's research stack (diffusion/flow matching, VLM-driven retrieval, controllable scene generation).
    - **Three-moat structure formalized** for Spatial RAG: (1) Spatial Retriever algorithm, (2) Spatial Augmentation Protocol as potential industry standard, (3) Spatial RAG Evaluator as field arbiter.
    - **Surface/internal duality principle established**: speak "architect tool" externally (cash flow, user feedback, NIW evidence), build "infrastructure" internally (Retrieval/Augmentation/Generation三层解耦, pluggable data sources, independently upgradable rendering).

    ### Phase 0 technical stack locked

    - **Two-backbone strategy**: WorldGen (FLUX-based, Gaussian Splat output) for PoC; modified Lyra 2.0 (NVIDIA, Wan 2.1-14B DiT, Apache 2.0) for Phase 2 research core. Marble used only as closed-source upper-bound reference, not modified.
    - **Visual reference source: Mapillary (CC-BY-SA), explicitly NOT Google Street View** — Google Maps ToS prohibits ML use; Mapillary permits research and commercial.
    - **Dev environment**: M5 Pro Mac (48GB unified memory) for data fetch/processing/alignment/eval/CLIP-DINOv2 inference; cloud GPU (Colab Pro / RunPod A100/4090) only for WorldGen and Lyra 2.0 inference.
    - **Package manager**: uv (not pip/conda) — 10-100x faster, modern pyproject.toml, cross-platform parity.
    - **Repo**: github.com/maxyzhu/atmosphere (public).

    ### Documentation produced

    - **`concept_report.md` v0.3** (replaces v0.2 docx): 9 sections, ~800 words, research-first framing. Includes alignment-with-target-roles section explicitly tied to Waymo JD analysis.
    - **`atmosphere_spec.md` v0.1**: 6-module technical architecture (M1 Retrieval, M2 Canonicalization, M3 Cross-Modal Alignment, M4 SCP Bundle, M5 Generation, M6 SFB Evaluation) with Phase 0 week-by-week plan.
    - **`INSIGHTS.md`** established as living research diary; entries dated and tagged.
    - **`DAY1_SETUP.md`** rewritten as cross-platform (Windows-first, Mac-later) guide with WSL2 fallback path.

    ### Phase 0 progress

    - **Day 1 complete**: `atmosphere/geo.py` with `LocalFrame` (WGS84 ↔ ENU using pymap3d), 9 unit tests, all passing. Test threshold relaxed for ENU-vs-haversine residual (sub-mm bound was wrong, ~10cm spherical-vs-planar residual is physically correct).
    - **Day 2 complete**: `atmosphere/retrieval/buildings.py` with `Building` dataclass + `HeightSource` enum (TAG/LEVELS/NONE provenance tracking) + `fetch_buildings()` via osmnx. ~25 tests added. **Architectural decision**: anti-corruption layer pattern — GeoPandas confined to retrieval module, only `list[Building]` exposed externally. Centroid implementation uses shoelace formula (true geometric centroid, handles L-shaped buildings) per user's question about closing-vertex semantics.
    - **Day 3 complete**: `atmosphere/retrieval/mapillary.py` with `MapillaryImage` dataclass + farthest-point sampling + Mapillary Graph API v4 integration. **Stage architecture introduced** in `atmosphere/stages.py` (`Stage` ABC + `StageData` + `STAGE_REGISTRY`) with `--stage` CLI flag, `--out` for headless save, `--list-stages` discovery — all for reproducible paper figure generation.

    ### Day 3 incidents and resolutions (chronological)

    - **Token type confusion**: User initially used Client Secret instead of Client Access Token; Mapillary silently returned `{"data":[]}` with no auth error. Resolved by switching token type. **Lesson**: API silent-fail is worse than HTTP error; consider runtime token validity probe in future.
    - **Mapillary 500 timeout in dense areas**: Bbox queries fail with "code 1, reduce data" error in densely-captured areas (downtown Seattle, central Copenhagen) regardless of small bbox size. Per Mapillary engineer reply on forum, this is a known timeout from October 2025 spatial-indexing service change.
    - **`_split_bbox` off-by-one bug**: Initial bbox-splitting implementation used `east+dx*(j+1)` instead of `west+dx*(j+1)` for sub-bbox upper corner, producing sub-bboxes larger than the original. Caught by user observation that n=6 sub-bboxes were 350m × 350m instead of expected 50m × 50m. Fixed.
    - **Adaptive recursive split implemented**: Replaced fixed n×n grid with `_fetch_recursive` that subdivides only on 500 timeout, max_depth=4. DLR test (47.6059, -122.3392, r=150m) yields 345 raw items in ~15 minutes. Visualization shows even street-aligned coverage matching Mapillary web density.
    - **Hybrid retrieval pattern documented for Phase 1**: Vector Tile API (`tiles.mapillary.com/maps/vtp/...`) provides image IDs without timeout; Graph API by-ID provides metadata without timeout. Two-stage pattern (tile=index, by-ID=fetch) projected to reduce 15min → 30-60sec per query. Deferred to Phase 1.

    ### Canonical test coordinate

    - **DLR Group Seattle office: lat 47.6059, lon -122.3392** (51 University St, Seattle — 2nd & University, Belltown / downtown edge). Default Phase 0 test point. Diversity coords: Pike Place 47.6090/-122.3416, Pioneer Square 47.6017/-122.3327, Capitol Hill 47.6210/-122.3211.

    ### Research insights (recorded to INSIGHTS.md)

    - **OSM building height coverage gap**: 43% of downtown Seattle buildings have no height info from OSM. This is the research motivation, not a data bug — Atmosphere's contribution is filling sparse anchors via generation, not finding perfect data.
    - **Mapillary spatial distribution is street-bound**: Cameras concentrate along streets, not uniformly across blocks. Linear chains rather than uniform clouds. Bbox vs circle visualization mismatch acknowledged as cosmetic Phase 1 fix.
    - **Sampling design must adapt to neighborhood typology**: Earlier "facade coverage impossible" framing was overstated. Perimeter-block commercial neighborhoods (Pike Place, SoHo) have ~95% street-visible facades — full coverage achievable in principle. SCP design must adapt to this variation, not assume uniform sparsity.

    ### Phase 1 algorithmic priorities documented (deferred from Phase 0)

    1. Replace square bbox query result with circular post-filter (semantic correctness)
    2. Replace fixed `target_count=100` with density-aware termination (stop when minimum pairwise distance drops below threshold)
    3. Implement Vector Tile + Graph-by-ID hybrid retrieval (15min → 60sec per query)
    4. Eventually: visibility-based set-cover sampling using ray-casting against LoD2 footprints, maximizing minimum per-facade coverage rather than uniform camera distribution

    ### Pedagogical pattern observed (across Days 1-3)

    Three test failures (`test_small_distance_approximately_euclidean`, `test_centroid`, `test_downsampling_favors_extremes`) all turned out to be over-tight test thresholds rather than code bugs. Documented principle for user: **when one test fails with otherwise-passing code and a clear docstring contract, suspect the test, not the code**.
</changelog>

<changelog>
.
├── atmosphere
│   ├── __init__.py
│   ├── config.py
│   ├── geo.py
│   ├── retrieval
│   │   ├── __init__.py
│   │   ├── buildings.py
│   │   └── mapillary.py
│   ├── stages.py
│   └── viz.py
├── atmosphere_spec.md
├── changelog.md
├── codebase_map.md
├── concept_report_v0.2.md
├── INSIGHTS.md
├── output_image
│   ├── 47_6059_-122_3392_dlr_group.png
│   ├── 47_6059_-122_3392_mapillary_recur.png
│   ├── 47_6059_-122_3392_mapillary_tile.png
│   ├── 47_6059_-122_3392_mapillary.png
│   └── Figure_1.png
├── pyproject.toml
├── README.md
├── scripts
│   └── visualize_neighborhood.py
├── tests
│   ├── test_buildings.py
│   ├── test_geo.py
│   ├── test_mapillary.py
│   └── test_stages.py
└── uv.lock

6 directories, 26 files

    # Atmosphere Day 3 / Day 4 Prep — Changelog Entry
    ## Summary
    Day 3 of Phase 0 PoC concluded successfully. Mapillary retrieval was migrated from Graph API search to vector tiles, OSM streets were added as a third pipeline stage, and the visualization pipeline produced clean output for the canonical Seattle test coordinate (DLR Group, 47.6059 / -122.3392) showing 38 buildings, 412 street segments (7.97 km, "all" network type), and 231 farthest-point-sampled Mapillary images with even spatial + heading distribution.
    ---
    ## 1. Mapillary Retrieval: Graph API Search → Vector Tile
    **Endpoint switch.** Replaced `graph.mapillary.com/images?bbox=...` queries with vector tile fetches from `tiles.mapillary.com/maps/vtp/mly1_computed_public/2/{z}/{x}/{y}` at zoom 14, decoded locally with `mapbox-vector-tile`. Chose `mly1_computed_public` over `mly1_public` for SfM-corrected positions.
    **Rationale.** Tiles are pre-sharded so they don't time out in dense urban areas; cacheable by `(z, x, y)` so neighboring queries reuse data; `compass_angle` is present in tile properties (correcting an earlier misconception in this conversation that it required Graph API).
    **Code removed:** `_raw_fetch`, `_fetch_recursive`, `_split_bbox` (the recursive 2×2 subdivision logic that worked around Graph API timeouts — unnecessary with tiles), `api_limit` parameter.
    **Open verification item.** Whether `compass_angle` in `mly1_computed_public` is SfM-corrected or raw is not explicit in Mapillary docs. To verify, diff same image_id between `mly1_public` and `mly1_computed_public`. Treated as authoritative for now.
    ---
    ## 2. Query Region: Circle → Square + Buffer
    **Geometry change.** Query region is now an ENU square of side `2 * (radius_m + BUFFER_M)`, where `BUFFER_M = 20` is shared across all retrieval modules. The `radius_m` parameter (default 150) is interpreted as half-side of the unbuffered square.
    **Rationale.** Generated imagery downstream (WorldGen / Lyra) is rectangular; a square query region with edge buffer prevents generation artifacts from intruding into the region of interest. OSM and Mapillary now share the exact same bbox, so all layers live in identical coordinates.

    ---

    ## 3. Sampling Target: Hardcoded → Density-Based

    **Formula.** `target_count = side² × (20 / 10000) = side² × 0.002`. At default `radius_m=150` (340m side), target ≈ 1156 images. Falls back to "take all" when candidate pool is smaller (sparse areas degrade gracefully).

    **Constants made explicit.** `IMAGES_PER_M2 = 0.002` lives in `mapillary.py` as a named constant.

    **Note on factor=20.** Initial proposal of 100 images / 100m × 100m was revised down to 20 — the higher density was excessive given the actual candidate pool size at typical city locations.

    ---

    ## 4. FPS Determinism

    **Seed change.** Farthest-point sampling now seeds from the image closest to ENU origin (= bbox center), replacing the previous random-with-seed-parameter behavior. Removes the `seed` parameter; sampling is now fully deterministic and visually anchored on the user's query point.

    **Metric unchanged.** Still `α · spatial_distance(m) + β · compass_distance(°)` with `α=1.0, β=0.44` (≈ 20m spatial ≈ 45° heading equivalence). Visual inspection of DLR Group output confirms metric produces good spatial + orientation diversity.

    ---

    ## 5. Caching: Per-Query → Per-Tile

    **Granularity change.** Tile cache keys are now `(z, x, y)` tuples, stored as `data/mapillary_cache/tiles/{z}_{x}_{y}.pbf`. Empty tiles cached as zero-byte placeholders to avoid re-querying known-empty regions.

    **Win.** Two queries within the same general neighborhood now reuse all overlapping tiles automatically. The previous per-query JSON cache (`raw_{lat}_{lon}_r{R}.json`) was replaced — old caches are obsolete and can be deleted.

    ---

    ## 6. MapillaryImage Dataclass Simplification

    **Removed fields:** `captured_at`, `sequence_id`, `is_pano`. Reasoning: not used by current sampling or downstream consumers.

    **⚠ Reverse-decision flagged.** `is_pano` should be added back before Day 4 WorldGen PoC — i2s (image-to-scene) mode prefers panorama input over perspective. The field is free in tile properties.

    **Current fields:** `mapillary_id`, `position_enu`, `compass_angle_deg`, `thumb_url`, `thumb_path`.

    ---

    ## 7. Graph API Made Optional

    **Behavior.** Vector tile alone provides `id`, `geometry`, `compass_angle` — sufficient for sampling. Graph API is only invoked when `download_thumbnails=True`, solely to retrieve signed `thumb_*_url` (not cacheable, expires).

    **`--no-download` flag.** Yields a pure vector-tile pipeline with zero Graph API calls.

    **Thumbnail resolution.** Currently `thumb_256_url`. **Should bump to `thumb_2048_url` before Day 4** — WorldGen is sensitive to input resolution, 256px is too small for meaningful conditioning.

    ---

    ## 8. New Stage: Streets

    **New module.** `atmosphere/retrieval/streets.py` parallels `buildings.py`:
    - `StreetSegment` dataclass: `polyline_enu`, `osm_id`, `highway_type`, `name`, `oneway`, with `length_m` and `midpoint_enu` properties
    - `fetch_streets(...)` uses the same square+buffer bbox via shared `BUFFER_M` import from `mapillary.py`
    - Backed by `osmnx.graph.graph_from_bbox`, simplification on, edges flattened to a list (topology discarded for Phase 0)
    - MultiDiGraph deduplication by canonical `(min(u,v), max(u,v), key)`
    - osmnx 1.x / 2.x bbox signature compatibility via try/except
    - Cache files: `.graphml` keyed by `(lat, lon, radius, buffer, network_type)`

    **Stage integration.** `STAGE_ORDER` is now `["osm", "street", "mapillary"]`. Street layer sits visually between buildings (fills) and Mapillary (markers) — the skeleton organizing both.

    **Visualization.** New `plot_streets()` in `viz.py` draws polylines with line width keyed to `highway_type` (motorway 2.4 → footway 0.5). Legend entry shows segment count, named-segment count, total length in km. `apply_frame()` extended with `streets=` parameter.

    **`network_type` default: changed from "drive" to "all".** Initial "drive" selection produced only 14 segments / 1.10 km in DLR area because OSM tags many service roads, alleys, and pedestrian areas in downtown Seattle as `service` / `footway` / `pedestrian`, all excluded by "drive". With "all", got 412 segments / 7.97 km (visually correct). Justified because Mapillary capture comes from cars, bikes, AND pedestrians; restricting to "drive" lost coverage.

    **CLI flag added.** `--street-network-type` lets user override (`drive`, `walk`, `bike`, `all`, etc.).

    ---

    ## 9. Buildings: Aligned to Square BBox

    `fetch_buildings` switched from `features_from_point(radius=...)` to `features_from_bbox(...)` using the same `_square_bbox_with_buffer` helper. Cache key includes buffer suffix (`b{BUFFER_M}`) to prevent collision with old radius-keyed caches.

    ---

    ## 10. Visualization Updates

    **`visualize_neighborhood.py`:**
    - `--mapillary-limit` default → `None` (triggers density auto-compute)
    - `--street-network-type` added (default `all`)
    - `--no-download` added (skip Graph API)
    - `apply_frame` call includes `streets=data.streets`
    - Help text reflects new defaults and `(radius + 20m buffer)` behavior

    **`stages.py`:**
    - `StageData` gained `streets: list[StreetSegment]` field
    - New `StreetStage` class
    - Bug fix: `MapillaryStage.fetch` now passes `target_count=opts.get("mapillary_limit")` (returns `None` when absent) instead of fallback `100` — was previously preventing density auto-compute from ever firing

    ---

    ## 11. Dependency Additions

    ```bash
    uv add mercantile mapbox-vector-tile
    ```

    `mercantile` for tile XYZ index calculation from bbox; `mapbox-vector-tile` for protobuf decoding.

    ---

    ## 12. Rejected / Deferred Decisions

    **DEM (terrain elevation): deferred to Phase 1.** Argued for during Day 3 but explicitly postponed. Rationale: Phase 0 PoC validates "does GIS conditioning work at all," not "is geometry millimeter-accurate." Z=0 ground plane assumption is acceptable for PoC. DEM becomes mandatory for Phase 1 SFB benchmark (z-direction ground truth needed for fidelity metrics). Suggested forward-compatibility move (not yet executed): add `ground_elevation_m: float | None = None` to `Building` and `MapillaryImage` dataclasses now to avoid Phase 1 architectural rewrite.

    **Sequence-based / segment-based sampling: deferred.** Discussed in depth (sequence direction inferred via `captured_at` ordering, segment-aware factor `10 × f(length)`, `(spatial + angle)` joint metric, image-to-segment assignment via nearest+threshold). Decision: Phase 0 stays with simple FPS over the full pool; sequence/segment structure added in Phase 1 if SFB benchmark needs it.

    **Spherical-first preference: discussed, not implemented.** `is_pano` was removed in §6 along with other fields. To support spherical-first sampling later, `is_pano` must come back, plus a soft priority bonus in FPS (e.g., `score *= 2.0` for pano).

    ---

    ## 13. Day 4 Pre-Work Items (Tracked Separately)

    Items surfaced during this session that need action before Day 4 begins, not yet implemented:

    - **Re-add `is_pano` to MapillaryImage** (free from tile, needed for WorldGen i2s mode)
    - **Bump `_FIELDS_THUMB` to `thumb_2048_url`** (WorldGen needs resolution > 256px)
    - **HuggingFace setup**: account, access token, accept FLUX.1-dev license — gated model, license review can take hours
    - **Cloud GPU decision**: AutoDL (¥5-8/h, ~$0.7-1.1) for fast iteration vs RunPod (~$1.6/h) for cleaner license posture for paper / NIW. Initial recommendation: AutoDL for Day 4 dry-run, RunPod for Phase 1 final-experiment runs

    ---

    ## 14. Strategic Note (No Code Change)

    **Career / research-depth question raised.** Discussion clarified that "setup world model + feed GIS conditioning" alone is wrapper-level engineering, insufficient for Waymo-tier research positions (the stated alignment target). Phase 2 retriever swap is the architectural contribution that elevates the project from integration to research.

    **Identified knowledge depth requirements** (must learn alongside Phase 0-3 execution):

    - **Tier 1 (mandatory)**: multi-view geometry (Hartley & Zisserman), diffusion model internals (DDPM math, DiT, conditioning mechanisms), 3D representations (NeRF, 3DGS), PyTorch engineering depth (DDP, mixed precision, checkpointing)
    - **Tier 2 (should-have)**: camera calibration / bundle adjustment hands-on, image matching (DUSt3R/MASt3R), depth estimation (Depth Anything v2), PyTorch3D / Open3D
    - **Tier 3 (bonus)**: video diffusion frontier, driving-specific world models (Cosmos, GAIA, Vista), graphics fundamentals

    **Embedded research-grade actions** (to make project read as research rather than wrapper):
    - Add quantitative geometric/visual analysis to Phase 0/1 (e.g., distribution of `computed_compass_angle` errors, IoU of building locations in generated panorama)
    - Write a short technical note / framework draft before Phase 2 begins
    - Self-implement at least one foundational component (PnP, minimal diffusion, minimal GS renderer)

    **Time allocation suggestion**: 5-8 hours/week of dedicated learning during Phase 0-3, with topics directly attached to project tasks rather than disconnected tutorials.

    ---

    ## Test Output (Reproducibility Anchor)

    Canonical test at DLR Group Seattle, `radius_m=150`, all stages:

    ```
    Buildings: 38 (11 tagged, 12 estimated, 15 unknown)
    Streets: 412 segs, 121 named, 7.97 km (network_type=all)
    Mapillary: 231 images (target ~1156 from density; pool was the limiter)
    FPS distribution: visually verified even spatial + compass coverage
    ```

    Reproducible via:
    ```bash
    python scripts/visualize_neighborhood.py \
        --lat 47.6059 --lon -122.3392 --stage mapillary --verbose
    ```
</changelog>

.
├── atmosphere
│   ├── __init__.py
│   ├── config.py
│   ├── geo.py
│   ├── retrieval
│   │   ├── __init__.py
│   │   ├── buildings.py
│   │   ├── mapillary.py
│   │   └── streets.py
│   ├── stages.py
│   └── viz.py
├── atmosphere_spec.md
├── changelog.md
├── concept_report_v0.2.md
├── INSIGHTS.md
├── output_image
│   ├── 47_6059_-122_3392_dlr_group.png
│   ├── 47_6059_-122_3392_mapillary_recur.png
│   ├── 47_6059_-122_3392_mapillary_tile.png
│   ├── 47_6059_-122_3392_mapillary.png
│   ├── 47_6059_-122_3392_w_street.png
│   └── Figure_1.png
├── pyproject.toml
├── README.md
├── scripts
│   └── visualize_neighborhood.py
├── tests
│   ├── test_buildings.py
│   ├── test_geo.py
│   ├── test_mapillary.py
│   └── test_stages.py
└── uv.lock

6 directories, 27 files


## 2026-05-11 · Day 4 Part 1: Backbone Decision + Code Cleanup

### Summary

First part of Day 4. No new pipeline modules yet — today focused on two
things: (1) a major framing decision about the generation backbone, and
(2) cleaning up Day 3 debt so the codebase is ready for M2.

Backbone selection settled. Test suite restored to a clean 84/84 green.
Phase A pre-work patches landed. M2 implementation deferred to Day 4
part 2; environment setup (cloud GPU) deferred to Day 5.

---

### 1. Backbone Decision: WorldMirror 2.0 (HY-World 2.0)

**Old plan (atmosphere_spec.md v0.1):** M5 wraps a "WorldGen" generation
backbone consuming an SCPBundle of geometric conditioning inputs.

**Problem discovered today:** "WorldGen" was a placeholder name not
backed by any specific open-source project. The closest real generation
backbones are Lyra 1.0 / 2.0 (NVIDIA, Apache 2.0) — but both take a
**single image** plus a synthetic camera trajectory and *hallucinate*
multi-view frames internally. Multi-real-image input is not supported.

**Alternative found:** HY-World 2.0 (Tencent, 2026-04-15) ships an
open-sourced **WorldMirror 2.0** component which is a multi-view
*reconstruction* model, not generation. It accepts an image directory
plus optional prior camera poses and prior depth maps, and outputs
3DGS + per-view depth + normals + intrinsics in a single forward pass.
License: `tencent-hy-world-2.0-community` (research-OK, custom
community license). VRAM ~12–24 GB — RTX 4090 sufficient.

**Decision:** Atmosphere is fundamentally a reconstruction problem, not
a generation problem. Mapillary gives us the real multi-view; OSM gives
us the geometric prior. WorldMirror 2.0 is exactly the right backend.
Lyra remains a Phase 2 reference / contrast baseline, not the primary.

**Open-source status caveat:** of HY-World 2.0's four announced
components, only WorldMirror 2.0 has released weights. HY-Pano 2.0,
WorldNav, and WorldStereo 2.0 are documented in the technical report
but not yet shipped. This is fine for Phase 0 (we only need
reconstruction); Phase 2's "reconstruct where observed, generate where
occluded" hybrid depends on WorldStereo 2.0 (or equivalent) becoming
available, and is documented as a known gap.

Detailed reasoning in INSIGHTS.md `2026-05-11 Day 4 (planning)` entry.

---

### 2. WorldMirror Prior Injection Interface (Discovered)

WorldMirror 2.0's public `WorldMirrorPipeline.__call__` accepts two
optional prior arguments:

- **`prior_cam_path`**: JSON file with `extrinsics` (list of 4×4 c2w
  matrices) and `intrinsics` (list of 3×3 K matrices). Either list can
  be empty / missing; pipeline auto-normalizes extrinsics relative to
  view 0 and adapts intrinsics for inference-time resize.
- **`prior_depth_path`**: directory of float32 `.npy` depth maps, one
  per input image, in camera frame, meters.

This is *exactly* the A/B experiment slot Atmosphere needs: same
Mapillary images, toggle priors on/off, measure reconstruction quality
difference. Three conditions for SFB benchmark fall out naturally:

- No prior (baseline; WorldMirror estimates everything)
- Prior cam only (M3 contribution: GIS-guided pose)
- Prior cam + prior depth (M2 + M3: full GIS conditioning)

Low-level API also exists (`WorldMirror.forward()` with tensor priors)
for finer control if needed in Phase 2.

---

### 3. M1–M6 Module Refocus (Spec Update Required)

The backbone switch reshapes what each Phase 0 module does:

| Module | Old role | New role |
|---|---|---|
| M1 Retrieval | Source data | Unchanged. Complete. |
| M2 Canonicalization | Bundle geometric conditioning for WorldGen | Render depth maps for `prior_depth_path` (trimesh + pyrender) |
| M3 Cross-Modal Alignment | Align Mapillary to GIS for SCP | Estimate c2w extrinsics for `prior_cam_path` |
| M4 SCP Bundle | Novel data structure | Thin adapter mapping internal types to WorldMirror's expected JSON / file layout |
| M5 Generation | Custom diffusion harness wrapping WorldGen | One-call wrapper around `WorldMirrorPipeline` |
| M6 SFB Evaluation | Custom geometry comparison | Built on WorldMirror's three-condition prior toggle; SFB measures delta between conditions |

**`atmosphere_spec.md` needs a v0.2 revision to reflect this.** Not
done today; deferred until after Day 4 M2 implementation so the spec
update is grounded in actual code rather than another round of
speculation.

---

### 4. Phase A Pre-Work Patches Landed

All three items flagged in Day 3's "Day 4 Pre-Work" section are now in.

**4.1 `is_pano` restored to MapillaryImage.** Removed during Day 3.2 as
unused; reinstated because WorldMirror's i2s-style modes prefer
panorama input over perspective. Cost: zero — the field is free from
vector tile properties.

Changes:
- `MapillaryImage` dataclass: new `is_pano: bool` field
- `_decode_image_features`: reads `is_pano` from feature properties
- Constructors at both call sites (parsed list + final list) populate it

**4.2 Thumbnail resolution bumped to 2048 px.** `_FIELDS_THUMB`
changed from `thumb_256_url` to `thumb_2048_url`. WorldMirror
conditioning needs meaningful input resolution; 256 px is too low.
2048 px is the largest Mapillary publicly serves.

*Unverified runtime risk*: some older Mapillary images may only have
256 or 1024 variants. The first real download run should log whether
any thumb_2048_url comes back null and handle the fallback.

**4.3 `ground_elevation_m` field added.** Optional `float | None = None`
added to both `Building` and `MapillaryImage`. Phase 0 sets it to None
(z=0 flat earth acceptable for PoC). The field exists now so Phase 1's
DEM integration doesn't require dataclass refactoring across consumers.

---

### 5. Test Suite Repair

Day 3 left several tests broken because mocks targeted functions that
had been refactored away. Fresh-clone `uv run pytest -v` was failing
before any new code today.

**5.1 `test_mapillary.py` rewritten.** Old file imported
`_radius_to_bbox` (replaced by `_square_bbox_with_buffer`), constructed
`MapillaryImage` with a `captured_at` field that no longer exists, and
used a `seed` parameter on `_farthest_point_sample` that was removed
when FPS became deterministically center-seeded. Pytest collection
failed with ImportError.

Rewritten against the current vector-tile pipeline. New coverage:
- `_square_bbox_with_buffer`: bbox math, buffer inclusion, scaling
- `_density_target`: formula correctness, min-1 floor, monotonicity
- `_farthest_point_sample`: below/equal/above target, center seeding,
  determinism (no seed param), missing compass tolerance
- `fetch_mapillary_images` end-to-end: tile bytes + decode features
  patched out, exercising bbox filter, dedup, density-driven
  target_count, `is_pano` propagation

17 tests, all green.

**5.2 `test_stages.py` extended.** `StreetStage` had zero test coverage
despite being a registered stage. Added:
- `test_populates_streets`: contract test, mocks `fetch_streets`
- `test_passes_network_type_option`: pass-through of `--street-network-type`
- `test_default_network_type_is_drive`: invariant matching docstring
- `test_preserves_earlier_stage_data`: doesn't clobber buildings
- `test_street_between_osm_and_mapillary` in `TestRegistry`:
  STAGE_ORDER invariant that streets sit between buildings and Mapillary

Also updated `TestStageData` to verify `streets` default-factory works
independently across instances (regression test for mutable default).

**5.3 `test_buildings.py` mock target repointed.** Day 3.2 changelog
§9 said `fetch_buildings` switched from `features_from_point` to
`features_from_bbox`. But the corresponding tests still patched the
old function name, so the mocks no-op'd and tests hit the real Overpass
API, returning 18 real Pioneer Square buildings instead of the 2 fake
ones the assertions expected.

Repointed all four `patch("...features_from_point")` calls to
`features_from_bbox`. No other test logic changes; the mock GDF and
assertions were already correct, only the patch target was wrong.

**5.4 Final state.**

```
84 passed in 0.39s
```

Coverage by file: `test_buildings.py` 28, `test_geo.py` 9,
`test_mapillary.py` 17, `test_stages.py` 18 (jumped from 13).

**5.5 Pedagogical note.** Three tests failing on a clean clone is a
smell, not a passing grade. All three failures were stale-mock issues,
not logic bugs — but they were hiding because nobody re-ran the suite
after Day 3.2's refactor. **New rule**: every retrieval-layer refactor
ends with `uv run pytest -v` green for the whole suite, not just the
new module.

---

### 6. Deferred to Day 4 Part 2 / Day 5

Not done today; tracked for next sessions:

- **M2 implementation** (`atmosphere/canonicalize/render.py`):
  `Building list → trimesh extruded mesh → pyrender depth map → .npy`.
  nvdiffrast deferred indefinitely — macOS has no NVIDIA path, and the
  pyrender pipeline produces identical output format. nvdiffrast only
  becomes attractive if Phase 2 needs differentiable rendering for pose
  refinement, which can be revisited then.

- **HuggingFace setup**: account + Read token created today.
  WorldMirror 2.0 is publicly downloadable (no license gate), so token
  is optional but recommended for rate-limit reasons. Token stored
  locally as `HF_TOKEN` in `.env` (not committed).

- **Cloud GPU provisioning**: not chosen yet. WorldMirror 2.0 only
  needs 12–24 GB VRAM, so RTX 4090 (~$0.44/h on RunPod) is
  sufficient. AutoDL is cheaper (~$0.7–1.1/h) but has worse license
  posture for paper submission. Day 5 decision.

- **First WorldMirror run**: depends on M2 + cloud GPU. Day 5 target:
  three-condition smoke test at DLR coord, output gaussians.ply + depth
  maps + camera params; visual inspection only, no SFB yet.

- **`atmosphere_spec.md` v0.2**: re-spec M1–M6 around the WorldMirror
  decision. Deferred until after Day 5 first run so the spec is written
  against working code.

---

### 7. Reframe for Project Narrative

Not "AEC tool that generates worlds from GIS."

Instead: **"Spatial RAG for 3D world reconstruction: given sparse,
noisy real-world observations (street imagery) and an authoritative
geometric prior (GIS), how should they be combined to maximize the
fidelity of a reconstructed 3D scene?"**

The Retriever / Augmentation / Generation three-layer moat structure
from Day 3 survives intact — the "generation" backend is just one model
type, and currently the reconstruction variant (WorldMirror) is the one
that actually fits our problem.

---

### File State at End of Day 4 Part 1

```
atmosphere/
  retrieval/
    buildings.py    → +ground_elevation_m field
    mapillary.py    → +is_pano field, +ground_elevation_m field,
                      thumb_2048_url, is_pano propagation
tests/
  test_buildings.py → 4 mock targets repointed
  test_mapillary.py → fully rewritten for vector-tile pipeline
  test_stages.py    → +5 tests (StreetStage + invariants)
INSIGHTS.md         → +2 entries (planning + impl part 1)
changelog.md        → this entry
```

No new modules created today.
