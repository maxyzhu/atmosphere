"""
Probe the King County DEM TIFF to understand its actual units and
coordinate system. Prints raster metadata + raw sampled values at
DLR before any conversions.
"""
from __future__ import annotations

from pathlib import Path

import rasterio
from pyproj import Transformer


DLR_LAT = 47.6059
DLR_LON = -122.3392


def main() -> None:
    tiff = Path("data/dem_cache/tiles/kingcounty_delivery1_be.tif")
    print(f"DEM file: {tiff}")
    print(f"Size: {tiff.stat().st_size / 1024 / 1024:.1f} MB")

    with rasterio.open(tiff) as src:
        print(f"\n--- Raster metadata ---")
        print(f"  CRS                 : {src.crs}")
        print(f"  CRS to_epsg()       : {src.crs.to_epsg()}")
        print(f"  CRS linear units    : {src.crs.linear_units}")
        print(f"  Shape (H, W)        : {src.height} x {src.width}")
        print(f"  Bounds (in src CRS) : {src.bounds}")
        print(f"  Transform           : {src.transform}")
        print(f"  Pixel size (units)  : {src.transform.a}, {src.transform.e}")
        print(f"  Number of bands     : {src.count}")
        print(f"  Band 1 dtype        : {src.dtypes[0]}")
        print(f"  Band 1 nodata       : {src.nodatavals[0]}")
        if hasattr(src, "units") and src.units:
            print(f"  Band units (gdal)   : {src.units}")
        # Tags may reveal vertical units / datum
        tags = src.tags()
        if tags:
            print(f"  File tags           : {tags}")
        band_tags = src.tags(1)
        if band_tags:
            print(f"  Band 1 tags         : {band_tags}")

        # Project DLR to source CRS
        transformer = Transformer.from_crs(4326, src.crs, always_xy=True)
        x, y = transformer.transform(DLR_LON, DLR_LAT)
        print(f"\n--- DLR ({DLR_LAT}, {DLR_LON}) in source CRS ---")
        print(f"  x = {x:.2f}")
        print(f"  y = {y:.2f}")

        # Read the raw raster value at DLR
        rows, cols = src.index(x, y)
        print(f"\n--- Raw pixel index at DLR ---")
        print(f"  row = {rows}, col = {cols}")
        if 0 <= rows < src.height and 0 <= cols < src.width:
            # Read a tiny 5x5 window for context
            from rasterio.windows import Window
            r0 = max(0, rows - 2)
            c0 = max(0, cols - 2)
            r1 = min(src.height, rows + 3)
            c1 = min(src.width, cols + 3)
            window = Window(c0, r0, c1 - c0, r1 - r0)
            data = src.read(1, window=window)
            print(f"  5x5 window around DLR (raw values, src CRS units):")
            for row in data:
                print("    " + "  ".join(f"{v:8.3f}" for v in row))
            center_raw = float(data[rows - r0, cols - c0])
            print(f"  Center pixel raw value: {center_raw}")
        else:
            print(f"  *** DLR falls outside raster bounds!")

        # Try a few well-known Seattle points for cross-reference
        print("\n--- Sanity points (well-known elevations) ---")
        seattle_points = [
            ("DLR (2nd & Univ)", 47.6059, -122.3392, "~30-40 m"),
            ("Pioneer Square",   47.6017, -122.3327, "~5-15 m (low, near port)"),
            ("Pike Place Market",47.6090, -122.3416, "~5-25 m (cliff edge)"),
            ("Capitol Hill",     47.6210, -122.3211, "~100-140 m (highest)"),
            ("Lake Washington",  47.6131, -122.2580, "~5 m (water level)"),
        ]
        for name, lat, lon, expected in seattle_points:
            x, y = transformer.transform(lon, lat)
            try:
                rows, cols = src.index(x, y)
                if 0 <= rows < src.height and 0 <= cols < src.width:
                    val = float(src.read(1, window=Window(cols, rows, 1, 1))[0, 0])
                    # Convert assuming US Survey Feet
                    val_m_if_ft = val * 1200.0 / 3937.0
                    print(f"  {name:24s}  raw={val:8.2f}  "
                          f"if_ft->m={val_m_if_ft:7.2f}  expected={expected}")
                else:
                    print(f"  {name:24s}  OUTSIDE RASTER")
            except (rasterio.errors.RasterioIOError, IndexError) as e:
                print(f"  {name:24s}  ERROR: {e}")


if __name__ == "__main__":
    from rasterio.windows import Window  # ensure imported in module scope
    main()
