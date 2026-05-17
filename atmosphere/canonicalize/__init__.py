"""
M2 — Canonicalization.

Convert retrieved geometry (M1 outputs) into a form world models can
consume: per-image depth maps rendered from a known camera pose.
"""

from atmosphere.canonicalize.camera import Camera, CameraType
from atmosphere.canonicalize.render import render_depth, buildings_to_mesh

__all__ = [
    "Camera",
    "CameraType",
    "buildings_to_mesh",
    "render_depth",
]