"""
Visualization utilities for buildings in the local ENU frame.

This module intentionally uses only `Building` dataclass instances — not
GeoPandas. That keeps the `geopandas` dependency scoped to retrieval/.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Polygon as MplPolygon

from atmosphere.retrieval.buildings import Building, HeightSource


# Color palette: color by height provenance. This is a subtle but useful
# communication: the reader can tell at a glance which parts of the city
# we know height for. Choices are colorblind-safe.
_COLOR_BY_SOURCE = {
    HeightSource.TAG: "#3b5998",      # navy — most trusted
    HeightSource.LEVELS: "#7ba7d4",   # mid blue — estimated
    HeightSource.NONE: "#c8c8c8",     # gray — unknown
}


def plot_buildings(
    buildings: list[Building],
    *,
    radius_m: float | None = None,
    title: str | None = None,
    ax: Axes | None = None,
    show: bool = True,
) -> Axes:
    """
    Plot a list of buildings as a top-down map in the ENU frame.

    Args:
        buildings: List of Building objects to plot.
        radius_m: If given, draws a circle of this radius around the origin
            (useful to show the query area).
        title: Optional figure title.
        ax: Existing matplotlib Axes to draw on. If None, a new one is made.
        show: If True, calls plt.show() at the end.

    Returns:
        The matplotlib Axes used. Useful if the caller wants to add more
        elements (e.g., Mapillary camera positions, later).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 9))

    # --- Draw each building as a filled polygon ---
    for b in buildings:
        color = _COLOR_BY_SOURCE[b.height_source]
        # Matplotlib wants an (N, 2) array of (x, y). ENU already is that.
        patch = MplPolygon(
            b.footprint_enu,
            closed=True,
            facecolor=color,
            edgecolor="#1f1f1f",
            linewidth=0.5,
            alpha=0.85,
        )
        ax.add_patch(patch)

    # --- Query center ---
    ax.plot(0, 0, "rx", markersize=14, markeredgewidth=2.5, zorder=10,
            label="query center")

    # --- Query radius circle ---
    if radius_m is not None:
        theta = np.linspace(0, 2 * np.pi, 360)
        ax.plot(
            radius_m * np.cos(theta),
            radius_m * np.sin(theta),
            color="red", linewidth=1.0, linestyle="--", alpha=0.7,
            label=f"query radius ({radius_m:.0f} m)",
        )

    # --- Axis setup ---
    ax.set_aspect("equal")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.grid(True, linewidth=0.3, alpha=0.4)

    # Frame the plot tightly around the data plus a small margin.
    if buildings or radius_m is not None:
        all_points = np.concatenate(
            [b.footprint_enu for b in buildings]
        ) if buildings else np.zeros((0, 2))
        if radius_m is not None:
            extent = max(radius_m * 1.1, 50.0)
        else:
            extent = float(max(np.abs(all_points).max(), 50.0)) * 1.1
        ax.set_xlim(-extent, extent)
        ax.set_ylim(-extent, extent)

    # --- Legend with provenance breakdown ---
    n_tag = sum(1 for b in buildings if b.height_source == HeightSource.TAG)
    n_lvl = sum(1 for b in buildings if b.height_source == HeightSource.LEVELS)
    n_none = sum(1 for b in buildings if b.height_source == HeightSource.NONE)

    legend_handles = [
        MplPolygon([[0, 0]], facecolor=_COLOR_BY_SOURCE[HeightSource.TAG],
                   edgecolor="black", linewidth=0.5,
                   label=f"height tagged ({n_tag})"),
        MplPolygon([[0, 0]], facecolor=_COLOR_BY_SOURCE[HeightSource.LEVELS],
                   edgecolor="black", linewidth=0.5,
                   label=f"height estimated ({n_lvl})"),
        MplPolygon([[0, 0]], facecolor=_COLOR_BY_SOURCE[HeightSource.NONE],
                   edgecolor="black", linewidth=0.5,
                   label=f"height unknown ({n_none})"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9,
              framealpha=0.9)

    if title:
        ax.set_title(title, fontsize=11)

    if show:
        plt.tight_layout()
        plt.show()

    return ax