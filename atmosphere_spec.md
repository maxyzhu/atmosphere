**Atmosphere.ai — Technical Architecture Specification**
Version 0.1 · April 2026 · Maxime Zhu

---

## 0. Purpose of This Document

This is a *working* technical spec. It defines the modules, interfaces, and key technical choices for the Atmosphere.ai research system. It is **not** a research proposal (see `concept_report_v0.2.md` for that) and **not** a paper.

This document exists so that when Phase 0 begins, implementation questions have already been answered. It is expected to be revised throughout Phase 0 as reality disagrees with the plan.

**Scope of v0.1 (this version):** covers Phase 0 PoC and enough of Phase 1 to make module boundaries stable. Phase 2 (modified Lyra 2.0) and Phase 3 (generalization) will be specified in later revisions.

---

## 1. System Thesis (one paragraph)

Atmosphere.ai is a research system for **anchored world model generation**: taking two complementary but individually incomplete sources of real-world information — coarse 3D geometry (LoD2 + DEM + street network) and sparse street-level imagery (Mapillary) — aligning them in a common coordinate frame, and using them jointly as conditioning signals for an open-source generative world model. The world model's role is to interpolate across the gaps that neither source covers (facade detail, lighting, vegetation, incidental objects, unobserved viewpoints), while the anchoring data prevents the generation from drifting away from the real location being simulated.

The system is a pipeline of six modules. Each is specified below.

---

## 2. High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  Query: (lat, lon, bearing, elevation, radius, vibe_prompt)     │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  M1. Spatial Retrieval Layer                                    │
│      ├── M1a. Geometry retrieval (LoD2 + DEM + OSM streets)     │
│      └── M1b. Image retrieval (Mapillary panoramas/frames)      │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  M2. Geometric Canonicalization                                 │
│      Render retrieved geometry from query pose into:            │
│      depth map · normal map · segmentation mask · silhouette    │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  M3. Cross-Modal Alignment                                      │
│      Refine Mapillary GPS (5–10m) → sub-meter via visual        │
│      localization against M2 renderings                         │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  M4. Conditioning Bundle Assembly (SCP)                         │
│      Pack aligned geometry + reference imagery + vibe prompt    │
│      into a canonical Spatial Conditioning Protocol bundle      │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  M5. Grounded Generation                                        │
│      Backbone A (PoC): WorldGen (FLUX + GS)                     │
│      Backbone B (research): modified Lyra 2.0 (Phase 2+)        │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  M6. Spatial Fidelity Evaluation (SFB)                          │
│      Silhouette IoU · skyline alignment · landmark error · A/B  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
                     Output: (scene, metrics)
```

---

## 3. Module Specifications

Each module is specified with five fields:

- **Inputs** — what it consumes
- **Outputs** — what it produces
- **Phase 0 implementation** — the minimum version to ship in 4 weeks
- **Phase 1+ extensions** — what gets added later
- **Key technical choices** — libraries, methods, and non-obvious decisions

---

### M1. Spatial Retrieval Layer

**Purpose:** Given a geographic query, fetch all relevant real-world spatial data within a neighborhood radius.

This module is split into two parallel sub-retrievals because the data sources have different access patterns.

#### M1a. Geometry Retrieval

**Inputs**
- `lat, lon` — query coordinate (WGS84)
- `radius_m` — search radius in meters (default: 80m for street-level queries, 200m for elevated views)

**Outputs**
- `buildings`: list of building polygons with per-building height (LoD1) or roof geometry (LoD2 where available)
- `terrain`: DEM raster tile covering the query area
- `streets`: OSM street network (LineString features with width/type attributes)
- All outputs in a common local ENU (East-North-Up) coordinate frame centered at query point

**Phase 0 implementation**
- Seattle only. King County open data portal provides LoD2 for downtown; fall back to OSM LoD1 elsewhere
- USGS 3DEP 1-meter DEM, pre-downloaded for King County and tiled locally
- OSM streets via `osmnx` Python package
- A simple file-system cache keyed by `(lat_rounded, lon_rounded, radius)` to avoid re-fetching

**Phase 1+ extensions**
- Add second city (Nanjing). Each city is a `DataConnector` subclass.
- Support LiDAR point clouds where available (as optional high-priority geometry source)
- Support user-uploaded survey/DWG (highest priority, overrides all others in its coverage area)

**Key technical choices**
- **Local ENU frame, not Web Mercator**: all downstream math (rendering, alignment) is easier in a metric local frame than in a projected one. Reproject at the boundary.
- **Fail soft on missing LoD2**: always have an LoD1 fallback. Record provenance (`source: "lod2" | "lod1" | "lidar"`) per building so downstream modules can reason about precision.
- **Data licensing is a first-class concern**: record license per source (OSM = ODbL, USGS = public domain, Mapillary = CC-BY-SA). This matters for eventual dataset release.

#### M1b. Image Retrieval

**Inputs**
- Query location (same as M1a)
- `max_images`: default 20
- `max_distance_m`: default 100

**Outputs**
- `reference_images`: list of `(image_url, raw_gps, raw_heading, timestamp, sequence_id)` tuples
- Raw here means "as reported by Mapillary"; refinement happens in M3

**Phase 0 implementation**
- Mapillary API v4 (`/images` endpoint with bbox filter)
- Filter: only `is_pano=false` (use regular frames, not 360 panoramas, for Phase 0 simplicity)
- Download full-resolution image + camera intrinsics when available

**Phase 1+ extensions**
- Add pano support (for skyline metrics)
- Consider adding historical photo sources (e.g., USGS aerial imagery, archive.org street-level)
- Explicit negative-list for training signal (imagery with people close to camera, for privacy)

**Key technical choices**
- **Mapillary, not Google Street View**: Mapillary is CC-BY-SA, permits research and ML training. GSV ToS explicitly prohibits both. This is a non-negotiable legal constraint.
- **Don't trust the GPS**: Mapillary GPS is 5–10m in the open, worse in urban canyons. Treat raw GPS as a coarse filter only. True alignment is M3's job.
- **Cap image count, not area**: density varies wildly between neighborhoods. Always return at most N images, even if the bbox is large.

---

### M2. Geometric Canonicalization

**Purpose:** Convert the retrieved 3D geometry into a canonical 2D representation aligned to the query camera pose. This is the *geometric conditioning signal* that the world model will consume.

**Inputs**
- Output of M1a (buildings + terrain + streets in local ENU frame)
- `camera_pose`: derived from query `(bearing, elevation, focal_length)`
- `image_size`: default `(1024, 1024)` for Phase 0

**Outputs**
A `CanonicalRender` object containing:
- `depth_map`: float32 array, per-pixel depth in meters
- `normal_map`: float32 array, per-pixel surface normal (world frame)
- `semantic_mask`: uint8 array with labels {sky, ground, building, road, other}
- `silhouette`: binary mask of building outlines
- `camera_params`: intrinsics + extrinsics used for rendering
- `provenance_mask`: per-pixel tag indicating which LoD each region came from

**Phase 0 implementation**
- Use `pytorch3d` or `nvdiffrast` for differentiable rasterization (useful later for M3 alignment)
- Sky mask: anywhere ray doesn't hit geometry within max distance → sky
- Semantic mask from geometry labels (buildings are buildings, DEM ground is ground, OSM road polygons give road)

**Phase 1+ extensions**
- Multi-view rendering (render 6 cubemap faces or a set of perturbed camera poses) to support cubemap-conditioned generation (WorldGen uses cubemaps natively)
- Render additional channels: UV coordinates, instance IDs per building (for landmark-level evaluation)

**Key technical choices**
- **Differentiable rendering from day 1**: even in Phase 0 where we don't differentiate through it, `nvdiffrast` is comparable in speed to non-differentiable rasterizers, and Phase 3 may need gradients through the renderer.
- **World-frame normals, not camera-frame**: world frame is invariant to camera rotation, which means later modules can reason about "which way is up" independent of where the camera points.
- **Provenance mask is not optional**: downstream evaluation needs to know "this pixel came from LoD1, so fidelity expectations are low." Without this, metrics become uninterpretable.

---

### M3. Cross-Modal Alignment

**Purpose:** Refine the imprecise GPS of Mapillary images by visually re-localizing them against the M2 geometric renderings. This is the module that solves the "5–10m GPS error" problem.

**Inputs**
- Output of M1b (raw Mapillary images with coarse GPS)
- Output of M2 (`CanonicalRender`)

**Outputs**
- For each Mapillary image, a refined `(pose_6dof, confidence)` tuple
- Images with confidence below threshold are dropped
- Surviving images are re-tagged with sub-meter pose

**Phase 0 implementation**
This is the single most research-heavy module in Phase 0. Two candidate approaches:

1. **Classical visual localization (ACE / HLoc style)**:
   - Extract SIFT/SuperPoint features from Mapillary image
   - Extract features from multi-view renderings of M2 geometry (render from ~12 poses around the raw GPS position)
   - Match features across modalities → PnP → refined pose
   - Problem: LoD2 has no texture. Classical features rely on texture. This will fail on most buildings.

2. **Silhouette-based alignment (recommended)**:
   - Segment buildings out of the Mapillary image (use `Mask2Former` or `SAM2`)
   - Compare segmented silhouette to M2's silhouette mask rendered from candidate poses
   - Optimize camera pose to minimize silhouette IoU loss
   - Works with LoD1/LoD2 because it ignores texture
   - This is the Phase 0 default.

**Phase 1+ extensions**
- Combine silhouette alignment with learned geometric features (something like `MASt3R` fine-tuned on LoD2 pairs)
- For sequences (video), exploit temporal consistency: if frame N is aligned, frame N+1 is constrained to be nearby

**Key technical choices**
- **This module is the core of CV Research Point #2**: camera pose estimation from GIS priors is an open problem. Expect this to take more than a week of Phase 0 time.
- **Cache aggressively**: alignment is expensive; once computed for a `(mapillary_id, pose)` pair, store it in a local DB.
- **Design for failure**: many images will fail to align (occlusions, wildly wrong GPS, urban canyons). The pipeline must handle "zero usable references" gracefully.

---

### M4. Conditioning Bundle Assembly (SCP)

**Purpose:** Package everything into the canonical Spatial Conditioning Protocol bundle that downstream generation backends consume. This defines the interface between retrieval and generation.

**Inputs**
- `CanonicalRender` from M2
- Aligned reference images from M3
- `vibe_prompt`: free-form text ("golden hour, light rain, autumn")

**Outputs**
An `SCPBundle` with the following schema (stable across Phase 0 and Phase 1):

```python
@dataclass
class SCPBundle:
    # --- Geometric anchor ---
    depth_map: np.ndarray              # (H, W) float32, meters
    normal_map: np.ndarray             # (H, W, 3) float32, world frame
    semantic_mask: np.ndarray          # (H, W) uint8, class labels
    silhouette: np.ndarray             # (H, W) bool
    provenance_mask: np.ndarray        # (H, W) uint8, source tag

    # --- Visual anchor ---
    reference_images: List[RefImage]   # aligned Mapillary refs
                                       # each has: RGB, refined_pose, confidence

    # --- Semantic prompt ---
    vibe_prompt: str
    vibe_embedding: np.ndarray         # (768,) CLIP text embedding

    # --- Camera + provenance ---
    camera_params: CameraParams
    query: QuerySpec                   # original (lat, lon, bearing, ...)
    bundle_version: str                # "scp/v0.1"
    generated_at: datetime
```

**Phase 0 implementation**
- Serialize as a single `.npz` (arrays) + `.json` (metadata) pair
- Validation: write a `validate_bundle(path)` function that checks shape/dtype/schema conformance before any generation attempt

**Phase 1+ extensions**
- Add `lidar_depth` channel when LiDAR is available
- Add `agent_layer` (pedestrians, vehicles) for Phase 3 if pursuing Waymo-style dynamic scenes
- Add `provenance_license` field per data layer for downstream compliance tracking

**Key technical choices**
- **SCP is versioned from day 1**: `bundle_version: "scp/v0.1"`. Breaking changes bump the version. Generation backends declare which versions they support.
- **The schema is the contract**: when this schema is published (artifact A3 in concept report), it becomes the only thing other researchers need to consume to integrate their own generation backends.
- **Embeddings are stored, not recomputed**: if the same vibe_prompt appears twice, its CLIP embedding is identical. Cache.

---

### M5. Grounded Generation

**Purpose:** Consume an SCP bundle and produce a generated scene. Two backbones, different purposes.

#### M5a. Backbone A — WorldGen (Phase 0 / 1)

**Inputs**
- `SCPBundle`

**Outputs**
- `scene.ply`: Gaussian Splat output
- `preview.jpg`: rendered preview from query camera pose
- `generation_log.json`: parameters, seed, runtime

**Phase 0 implementation**
- Modify WorldGen's `i2s` mode: instead of a single input image, feed it the top-1 aligned Mapillary image as `image`, and use `depth_map` to replace WorldGen's own depth estimation step
- Use the vibe prompt as-is as `prompt`
- Low-vram mode enabled for 4090-class GPUs

**Phase 1+ extensions**
- Multi-reference conditioning (multiple aligned Mapillary refs feeding into a cross-attention layer)
- Explicit silhouette constraint: penalize GS splats that fall outside the LoD2 silhouette

**Key technical choices**
- **Phase 0 does not train anything.** WorldGen is used in inference mode. The only "new" code is the conditioning pre-processor that converts SCP bundles into WorldGen's native input format.
- **Reproducibility is non-negotiable**: seed everything, log everything, commit the exact SCP bundle alongside the generated output.

#### M5b. Backbone B — Modified Lyra 2.0 (Phase 2+)

Not specified in v0.1. See Phase 2 design doc (to be written after Phase 0 completion).

---

### M6. Spatial Fidelity Evaluation (SFB)

**Purpose:** Quantify how well a generated scene matches the real location it claims to represent.

**Inputs**
- Generated scene (from M5)
- A held-out ground truth reference: a Mapillary image at the same query coordinate that was **never used** in M1b/M3/M5 for this query

**Outputs**
- `metrics.json` with four scores plus diagnostics

**Phase 0 implementation — four metrics**

1. **Building silhouette IoU**
   - Rasterize LoD2 from query pose → ground-truth silhouette
   - Segment buildings from generated scene → generated silhouette
   - Compute IoU
   - Sanity check: should be high by construction (scene was conditioned on this silhouette), used as a regression test

2. **Skyline alignment error**
   - Extract skyline curve (top of building silhouette) from both generated and held-out Mapillary image
   - Compute mean vertical pixel distance between curves after horizontal alignment
   - This is the *real* test — the held-out image was never seen by generation

3. **Landmark relative position error**
   - If the query area contains known landmarks (e.g., Space Needle for Seattle), mark their expected pixel positions based on LoD2 geometry
   - Check whether generated scene places landmarks in the correct pixel region
   - Binary per-landmark, aggregated as accuracy

4. **Human pairwise preference (the only subjective metric, included anyway)**
   - Show a human rater: held-out Mapillary image + two generated scenes (with and without SCP conditioning)
   - Ask: "Which of these two looks more like the real location shown in the reference?"
   - Aggregate win rates. Small-N OK for Phase 0 (N=20 raters × 20 scenes is acceptable).

**Phase 1+ extensions**
- Depth consistency metric: compare Depth Anything output on generated scene vs. LoD2 depth render
- Learned metric: train a small CNN to predict human preference from image pairs, releasing as a reusable scorer
- Temporal metrics for video generation (Phase 2+): e.g., consistency of generated buildings across a walkthrough

**Key technical choices**
- **Held-out discipline is sacred**: if a Mapillary image is used as ground truth for evaluation, it cannot be used anywhere in M1–M5 for that query. Implement as a hard database split, not a gentleman's agreement.
- **Metrics are computed but not averaged away**: per-query scores are reported alongside aggregates. Outliers are often more informative than means.
- **A/B uses identical seeds**: when comparing "with SCP" vs "without SCP," all randomness (model seed, camera pose, prompt) must be identical except for the presence/absence of conditioning.

---

## 4. Interface Contracts Between Modules

The modules communicate through four stable contracts. These are the things that must not change without a version bump.

| Contract | Producer | Consumer | Format |
|----------|----------|----------|--------|
| `RawSpatialData` | M1 | M2, M3 | Dict of (GeoJSON buildings, DEM raster, image URL list) |
| `CanonicalRender` | M2 | M3, M4 | Dataclass with depth/normal/semantic/silhouette arrays |
| `SCPBundle` | M4 | M5, M6 | `.npz` + `.json` pair, schema in §M4 |
| `GeneratedScene` | M5 | M6 | `.ply` or `.mp4` + metadata JSON |

**Rule:** any module can be reimplemented (different library, different algorithm, different language) as long as it honors its contract. This is what makes the "two backbones" strategy feasible — M5a and M5b consume the same SCPBundle.

---

## 5. Repository Layout (Phase 0)

```
atmosphere/
├── atmosphere_spec.md           # this doc
├── concept_report_v0.2.docx     # why we're doing this
├── README.md                    # how to run a query end-to-end
│
├── atmosphere/                  # installable package
│   ├── __init__.py
│   ├── retrieval/               # M1
│   │   ├── geometry.py          # M1a
│   │   └── imagery.py           # M1b
│   ├── canonicalize/            # M2
│   │   └── render.py
│   ├── alignment/               # M3
│   │   └── silhouette_align.py
│   ├── scp/                     # M4
│   │   ├── schema.py            # SCPBundle dataclass
│   │   └── validate.py
│   ├── generation/              # M5
│   │   └── worldgen_backend.py
│   └── evaluation/              # M6
│       ├── metrics.py
│       └── sfb_benchmark.py
│
├── configs/                     # YAML configs per experiment
├── data/                        # cached GIS + Mapillary (gitignored)
├── notebooks/                   # exploratory work
├── scripts/
│   ├── run_query.py             # end-to-end CLI entry point
│   ├── build_sfb_dataset.py     # curate the 20-coord Seattle benchmark
│   └── evaluate_baseline.py
└── tests/
    ├── test_scp_schema.py
    └── test_alignment.py
```

---

## 6. Phase 0 Week-by-Week Plan

Each week ends with a specific, verifiable deliverable. If a week's deliverable slips, the phase slips — no silent buffering.

### Week 1 — Data foundation
- [ ] Set up `atmosphere` package skeleton, repo, linting, CI
- [ ] Implement M1a for Seattle: fetch LoD2/LoD1 + DEM + streets for any lat/lon
- [ ] Implement M1b: Mapillary API client with bbox query
- [ ] Manually curate a list of 5 Seattle test coordinates for development (downtown, Capitol Hill, Ballard, Fremont, waterfront)
- **Deliverable:** `scripts/fetch_neighborhood.py <lat> <lon>` produces a folder of raw geometry + imagery for any Seattle coord

### Week 2 — Canonicalization + first end-to-end stub
- [ ] Implement M2: render `CanonicalRender` using nvdiffrast
- [ ] Implement M4 schema and serialization (M3 stubbed as identity pass: raw GPS = refined GPS)
- [ ] Implement M5a: WorldGen wrapper that consumes SCPBundle
- [ ] End-to-end test: `run_query.py` produces a `.ply` for one test coord
- **Deliverable:** one generated scene from one coordinate, ugly but real

### Week 3 — Alignment + SFB dataset
- [ ] Implement M3 with silhouette-based alignment
- [ ] Compare aligned vs raw-GPS poses qualitatively
- [ ] Expand test coordinate set from 5 to 20; ensure each has good Mapillary coverage
- [ ] Designate held-out images per coord (for later M6 evaluation)
- **Deliverable:** SFB-v0 dataset (20 coords, fully packaged, held-out split documented)

### Week 4 — Evaluation + go/no-go
- [ ] Implement M6 metrics 1–3 (silhouette IoU, skyline, landmark)
- [ ] Run paired comparison: WorldGen+SCP vs WorldGen-text-only on all 20 coords
- [ ] Write up Phase 0 results: internal memo with metrics table + 5 side-by-side comparison figures
- [ ] **Go/no-go decision:** does SCP conditioning measurably improve any metric? If yes → proceed to Phase 1. If no → diagnose which module is failing.
- **Deliverable:** Phase 0 retrospective doc, explicit decision on Phase 1 continuation

---

## 7. Known Risks (Technical, Phase 0 Only)

Non-technical risks (schedule, motivation, NIW alignment) are covered in `concept_report_v0.2.docx`. This section only covers what could break the code.

1. **M3 silhouette alignment converges to local minima.** Urban canyons have many similar silhouettes at different poses. Mitigation: initialize from raw GPS, run multiple random restarts, keep top-K poses.

2. **WorldGen's conditioning interface isn't as flexible as its README suggests.** The `i2s` mode takes one image + one prompt; injecting our depth map may require modifying internal code, not just calling a public API. Mitigation: Week 2 has this as its #1 task so it fails early.

3. **Mapillary coverage in some Seattle neighborhoods is too sparse.** Backup plan: narrow the SFB-v0 dataset to downtown + Capitol Hill + SLU, which have dense coverage.

4. **4090 (24 GB VRAM) is insufficient for WorldGen + nvdiffrast concurrent use.** Mitigation: run them in separate processes, serialize SCPBundle to disk between them.

5. **Held-out split leaks through sequence_id.** Mapillary images are captured in sequences; a held-out image at coord C might be 3 meters from a training image in the same sequence. Mitigation: hold out entire sequences, not individual images.

---

## 8. What This Spec Intentionally Does Not Cover

- **Training procedures.** Phase 0 is inference-only. Training (of Geometry Encoder, or of modified Lyra retriever) is a Phase 1–2 concern.
- **Phase 2 Lyra modification details.** Will be specified in a follow-up doc after Phase 0 results are in.
- **Commercial deployment (renderer plugins, API service, etc.).** Out of scope for the research project.
- **Second-city support.** Seattle-only through Phase 1.
- **Multi-query batching / performance optimization.** Phase 0 optimizes for correctness, not throughput.
- **Security / privacy auditing of outputs.** Necessary eventually (e.g., face blurring in retrieved Mapillary imagery) but deferred.

---

## 9. Glossary

- **LoD1, LoD2** — CityGML Levels of Detail. LoD1 is building footprints extruded to flat roofs; LoD2 adds roof shapes and major facade elements.
- **ENU frame** — East-North-Up local Cartesian coordinate frame, origin at a query point. Metric, convenient for rendering and physics.
- **SCP** — Spatial Conditioning Protocol. The canonical bundle format defined in M4, also the name of the broader design philosophy of this project.
- **SFB** — Spatial Fidelity Benchmark. The held-out evaluation dataset + metrics defined in M6.
- **Mapillary** — A crowdsourced street-level imagery platform, CC-BY-SA licensed, owned by Meta. Research/ML use permitted, unlike Google Street View.
- **Anchored generation** — The project's thesis (§1): allow a generative model to fill in unobserved detail, but constrain it so that it cannot move the anchor points (real-world geometry and confirmed visual references) that came from data.
- **Backbone** — The generative model at the core of M5. Phase 0 uses WorldGen; Phase 2 will use a modified Lyra 2.0.

---

## 10. Revision Log

- **v0.1 (April 2026):** Initial draft covering Phase 0 and stable parts of Phase 1. Module interfaces designed to survive the introduction of Backbone B (Lyra 2.0) in Phase 2 without schema changes.

---

*Document maintained by Maxime Zhu. Suggestions, disagreements, and corrections welcome — they are the point of a working spec.*
