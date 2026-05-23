"""
SCP (Spatial Conditioning Protocol) — backbone-agnostic encoding of
GIS-anchored geometric priors for 3D reconstruction and generation.

This subpackage is the destination format Atmosphere ships against,
across all phases:

- Phase 0/1: SCP -> WorldMirror 2.0 reconstruction (prior_camera.json
  + prior_depth/*.npy + source images). A/B ablation = SCP-on vs
  SCP-off, scored against held-out Mapillary views (SFB benchmark).
- Phase 2: SCP -> Hunyuan-Pano + WorldMirror generation pipeline.
- Phase 3: SCP -> Lyra 2.0 generation. Same protocol, different
  consumer.

The contribution of Atmosphere is the protocol itself, not the
choice of consumer. SCP is what's portable across world-model
backbones; consumers come and go.

What goes in a bundle
---------------------
- N source images (one per surviving Mapillary observation)
- N Camera priors (c2w + intrinsics, ENU world frame)
- N depth priors (M2-rendered, sky as SKY_DEPTH_M, no inf/nan)
- A manifest documenting which retrieval inputs produced which
  outputs, so SFB benchmark can stratify error by source quality

What stays out
--------------
- Raw OSM building footprints, raw street network, raw DEM tiles.
  These are SCP-internal — the protocol exposes their *effect* on
  the per-image priors, not the raw GIS data. A future Phase 1+
  consumer that wants raw GIS gets it from atmosphere.retrieval
  directly.

Protocol-level constants
------------------------
SKY_DEPTH_M is exported here so M2 (depth renderer) and M4 (bundle
assembler) agree on the same finite faraway value without one
importing the other. WorldMirror's nan_to_num would silently coerce
np.inf to 0 (= depth at camera origin), so we replace non-finite
depth with this constant. See changelog Day 5 Part 1 §1 for the
source-code observation that motivated the choice.
"""

# Protocol constant: depth value used for sky / no-hit pixels.
# Chosen >> any plausible scene scale for street-level views, so the
# downstream model treats those pixels as "very far" rather than
# silently misinterpreting them.
SKY_DEPTH_M: float = 1000.0
