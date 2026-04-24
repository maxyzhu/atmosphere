"""
Visualize buildings around a given coordinate.

Usage:
    python scripts/visualize_neighborhood.py --lat 47.6097 --lon -122.3331
    python scripts/visualize_neighborhood.py --lat 47.6097 --lon -122.3331 --radius 200 --no-cache
"""

from __future__ import annotations

import argparse
import logging

from atmosphere.retrieval.buildings import fetch_buildings
from atmosphere.viz import plot_buildings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lat", type=float, required=True,
                        help="Query center latitude (degrees)")
    parser.add_argument("--lon", type=float, required=True,
                        help="Query center longitude (degrees)")
    parser.add_argument("--radius", type=float, default=150.0,
                        help="Query radius in meters (default 150)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Skip local OSM cache; refetch from Overpass")
    parser.add_argument("--title", type=str, default=None,
                        help="Figure title (defaults to coordinates)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print detailed retrieval progress")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    buildings = fetch_buildings(
        lat=args.lat,
        lon=args.lon,
        radius_m=args.radius,
        use_cache=not args.no_cache,
    )

    title = args.title or (
        f"Buildings near ({args.lat:.4f}, {args.lon:.4f}), "
        f"radius {args.radius:.0f} m — {len(buildings)} found"
    )
    plot_buildings(buildings, radius_m=args.radius, title=title)


if __name__ == "__main__":
    main()