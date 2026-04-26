"""
Visualization primitives for Atmosphere data, in the local ENU frame.

Each function here draws ONE layer. Orchestration (which layers to draw,
in what order) lives in atmosphere.stages. This keeps the responsibilities
clean: this file knows how to paint; stages.py decides the composition.

All functions take a matplotlib Axes and operate on it in-place.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import FancyArrow, Polygon as MplPolygon

from atmosphere.retrieval.buildings import Building, HeightSource
from atmosphere.retrieval.mapillary import MapillaryImage


# Colors
_COLOR_BY_HEIGHT_SOURCE = {
    HeightSource.TAG: "#3b5998",
    HeightSource.LEVELS: "#7ba7d4",
    HeightSource.NONE: "#c8c8c8",
}
_MAPILLARY_COLOR = "#2e9949"
_MAPILLARY_COLOR_NO_COMPASS = "#8ab78f"


# ─────────────────────────────────────────────────────────────────────────────
# Layer primitives
# ─────────────────────────────────────────────────────────────────────────────


def plot_buildings(buildings: list[Building], ax: Axes) -> None:
    """Draw buildings as filled polygons colored by height provenance."""
    for b in buildings:
        color = _COLOR_BY_HEIGHT_SOURCE[b.height_source]
        patch = MplPolygon(
            b.footprint_enu,
            closed=True,
            facecolor=color,
            edgecolor="#1f1f1f",
            linewidth=0.5,
            alpha=0.85,
        )
        ax.add_patch(patch)


def plot_mapillary(
    images: list[MapillaryImage],
    ax: Axes,
    *,
    arrow_length_m: float = 8.0,
) -> None:
    """Draw Mapillary images as camera dots with heading arrows."""
    for img in images:
        x, y = img.position_enu
        if img.has_compass:
            # Compass: 0 = N, clockwise. Matplotlib: 0 = +x (east), CCW.
            theta_rad = np.radians(90.0 - img.compass_angle_deg)
            dx = arrow_length_m * np.cos(theta_rad)
            dy = arrow_length_m * np.sin(theta_rad)
            arrow = FancyArrow(
                x, y, dx, dy,
                width=0.3,
                head_width=2.5,
                head_length=2.5,
                length_includes_head=True,
                color=_MAPILLARY_COLOR,
                alpha=0.85,
                zorder=5,
            )
            ax.add_patch(arrow)
            ax.plot(x, y, "o", color=_MAPILLARY_COLOR, markersize=3, zorder=6)
        else:
            ax.plot(
                x, y, "o",
                color=_MAPILLARY_COLOR_NO_COMPASS,
                markersize=4,
                zorder=5,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Frame helpers (decoration that every stage figure needs)
# ─────────────────────────────────────────────────────────────────────────────


def apply_frame(
    ax: Axes,
    *,
    radius_m: float | None = None,
    title: str | None = None,
    buildings: list[Building] | None = None,
    images: list[MapillaryImage] | None = None,
) -> None:
    """
    Apply common decorations: query center marker, radius circle, axes,
    legend. Called once at the end by the stage runner.
    """
    # Query center
    ax.plot(0, 0, "rx", markersize=14, markeredgewidth=2.5, zorder=10)

    # Radius circle
    if radius_m is not None:
        theta = np.linspace(0, 2 * np.pi, 360)
        ax.plot(
            radius_m * np.cos(theta),
            radius_m * np.sin(theta),
            color="red", linewidth=1.0, linestyle="--", alpha=0.7,
        )

    # Framing: expand axes to cover everything
    extents: list[float] = []
    if buildings:
        all_points = np.concatenate([b.footprint_enu for b in buildings])
        extents.append(float(np.abs(all_points).max()))
    if images:
        img_points = np.array([img.position_enu for img in images])
        extents.append(float(np.abs(img_points).max()))
    if radius_m is not None:
        extents.append(radius_m)
    extent = max(extents) * 1.1 if extents else 100.0
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)

    ax.set_aspect("equal")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.grid(True, linewidth=0.3, alpha=0.4)

    # Legend — assembled from whatever layers are present
    handles = []
    if buildings:
        n_tag = sum(1 for b in buildings if b.height_source == HeightSource.TAG)
        n_lvl = sum(1 for b in buildings if b.height_source == HeightSource.LEVELS)
        n_none = sum(1 for b in buildings if b.height_source == HeightSource.NONE)
        handles.extend([
            MplPolygon([[0, 0]], facecolor=_COLOR_BY_HEIGHT_SOURCE[HeightSource.TAG],
                       edgecolor="black", linewidth=0.5,
                       label=f"height tagged ({n_tag})"),
            MplPolygon([[0, 0]], facecolor=_COLOR_BY_HEIGHT_SOURCE[HeightSource.LEVELS],
                       edgecolor="black", linewidth=0.5,
                       label=f"height estimated ({n_lvl})"),
            MplPolygon([[0, 0]], facecolor=_COLOR_BY_HEIGHT_SOURCE[HeightSource.NONE],
                       edgecolor="black", linewidth=0.5,
                       label=f"height unknown ({n_none})"),
        ])
    if images:
        n_compass = sum(1 for i in images if i.has_compass)
        n_nocompass = len(images) - n_compass
        handles.append(plt.Line2D(
            [0], [0], marker="o", color=_MAPILLARY_COLOR, linestyle="",
            markersize=7, label=f"Mapillary oriented ({n_compass})",
        ))
        if n_nocompass:
            handles.append(plt.Line2D(
                [0], [0], marker="o", color=_MAPILLARY_COLOR_NO_COMPASS,
                linestyle="", markersize=7,
                label=f"Mapillary no compass ({n_nocompass})",
            ))

    if handles:
        ax.legend(handles=handles, loc="upper right", fontsize=9,
                  framealpha=0.9)

    if title:
        ax.set_title(title, fontsize=11)