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