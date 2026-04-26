# Records of Observation

## 2026-04-23Day 2: OSM Building Retrieval and Visualization
All tests passed.
The shape of the 2D map is pretty accurate, but the height data is not thorough as expected. In "temp_file/47_6059_-122_3392_dlr_group.png", the upper right right sides doesn't have height data, but in Google Map, they are all 3D buildings, including some high-rise. 54% data precision is not good. But it can be a more powerful statement that imperfect data can also create good scene with the power of world model and Mapillary iamges.

### 2026-04-25 Day 3: Fetch Mapillary Data Points
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

Decision: defer all three to Phase 1. Day 3 ships with 15-min runtime;
the research thesis is unaffected by fetch latency.