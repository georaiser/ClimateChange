"""
Chapter 5: 15_zonal_statistics.py
====================================
Zonal Statistics -- Per-Watershed Environmental Summary Table

Academic Objective:
  Raw pixel maps are visually powerful but environmental managers need NUMBERS:
  "What is the mean NDVI inside Watershed A? How does glacier proximity vary
  between the northern and southern zones?"

  Zonal Statistics answers this: for each polygon in a vector layer, compute
  summary statistics (mean, std, min, max, percentiles, count) over all raster
  pixels that fall inside that polygon.

  This script demonstrates zonal statistics across ALL major outputs from
  Chapters 2-4, creating a single consolidated summary table:

  Input polygons (3 sources, whichever exist):
    1. Ch03 watershed basins (flow_accumulation.tif basin delineation)
    2. Programmatically generated quadrant zones (NW/NE/SW/SE)
    3. Management zones shapefile (if exists)

  Input rasters:
    NDVI, NDMI, CVI, SDM suitability, DEM, slope

  Output: one row per zone x one column per raster → cross-chapter comparison table.

  Academic note: rasterstats uses ALL-TOUCHED=False by default
  (only pixels whose CENTER falls inside the polygon). This is correct for
  large zones; use ALL_TOUCHED=True only for very thin polygons.

Outputs:
  data/processed/zonal/management_zones.gpkg
  data/processed/zonal/zonal_statistics_summary.csv
  data/processed/zonal/zonal_statistics.png    (multi-panel dark heatmap + maps)

ArcGIS Pro: Spatial Analyst > Zonal Statistics As Table
            Use the watershed polygons + any raster for per-basin stats.
ENVI 5.6:   Vector > Statistics to extract per-polygon raster values.

Run:
  conda activate geocascade_env
  pip install rasterstats   (if not installed)
  python Chapter_05/15_zonal_statistics.py

Dependencies: rasterstats, geopandas, rasterio, numpy, matplotlib, pandas, shapely
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
import geopandas as gpd
from shapely.geometry import box

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
OUT_DIR  = os.path.join(BASE_DIR, "data", "processed", "zonal")
os.makedirs(OUT_DIR, exist_ok=True)

BBOX = [-73.30, -51.10, -72.90, -50.80]

# Paths to outputs from earlier chapters (fallback if not found: STAC download)
RASTER_SOURCES = {
    "NDVI":          os.path.join(ROOT_DIR, "Chapter_02", "data", "processed", "indices", "ndvi.tif"),
    "NDMI":          os.path.join(ROOT_DIR, "Chapter_05", "data", "processed", "moisture", "ndmi.tif"),
    "DEM":           os.path.join(ROOT_DIR, "Chapter_03", "data", "processed", "terrain", "copernicus_dem.tif"),
    "Slope":         os.path.join(ROOT_DIR, "Chapter_03", "data", "processed", "terrain", "slope_degrees.tif"),
    "SDM":           os.path.join(ROOT_DIR, "Chapter_04", "data", "processed", "niche", "ecological_niche_model.tif"),
    "CVI":           os.path.join(ROOT_DIR, "Chapter_04", "data", "processed", "vulnerability", "vulnerability_index.tif"),
    "Glacier_NDSI":  os.path.join(ROOT_DIR, "Chapter_02", "data", "processed", "indices", "glacier_mask_ndsi.tif"),
}

DARK_BG = "#0d1117"
DARK_AX = "#161b22"
C_TEXT  = "#e6edf3"
C_GREY  = "#8b949e"


# ---------------------------------------------------------------------------
# 1. Create or load zone polygons
# ---------------------------------------------------------------------------
def build_zones():
    """
    Build analysis zones in priority order:
      1. Use existing management_zones.gpkg if present
      2. Create 4-quadrant zones over BBOX
      3. Subdivide into 9 cells (3x3 grid) for finer analysis
    Returns a GeoDataFrame in EPSG:4326.
    """
    existing = os.path.join(OUT_DIR, "management_zones.gpkg")
    if os.path.exists(existing):
        gdf = gpd.read_file(existing)
        print(f"  [OK] Loaded existing zones: {len(gdf)} polygons")
        return gdf

    # Build 3x3 grid of analysis cells
    lon_edges = np.linspace(BBOX[0], BBOX[2], 4)
    lat_edges = np.linspace(BBOX[1], BBOX[3], 4)

    zones = []
    for i in range(3):
        for j in range(3):
            geom = box(lon_edges[j], lat_edges[i], lon_edges[j+1], lat_edges[i+1])
            label = f"Zone_{chr(65 + i*3 + j)}"   # A through I
            quad  = ("N" if i == 2 else "C" if i == 1 else "S") + \
                    ("W" if j == 0 else "M" if j == 1 else "E")
            zones.append({"Zone": label, "Quadrant": quad, "geometry": geom})

    gdf = gpd.GeoDataFrame(zones, crs="EPSG:4326")

    # Save as GeoPackage (preferred over Shapefile: single file, no 10-char column limit)
    gpkg_path = os.path.join(OUT_DIR, "management_zones.gpkg")
    gdf.to_file(gpkg_path, driver="GPKG")
    print(f"  [OK] Created 3x3 analysis grid: {len(gdf)} zones -> {gpkg_path}")
    return gdf


# ---------------------------------------------------------------------------
# 2. Discover which rasters exist
# ---------------------------------------------------------------------------
def find_rasters():
    found   = {}
    missing = []
    for name, path in RASTER_SOURCES.items():
        if os.path.exists(path):
            found[name] = path
            print(f"  [OK] {name:<15s}: {os.path.relpath(path, ROOT_DIR)}")
        else:
            missing.append(name)

    if missing:
        print(f"  [NOTE] Not found (will skip): {', '.join(missing)}")
        print(f"         Run earlier chapters to produce these outputs.")

    # Fallback: generate a minimal DEM if nothing found
    if not found:
        print("\n  No local rasters found. Downloading DEM from Planetary Computer...")
        dem_path = _download_dem_fallback()
        if dem_path:
            found["DEM"] = dem_path

    return found


def _download_dem_fallback():
    """Download Copernicus DEM as a last resort so the script still runs."""
    try:
        from pystac_client import Client
        import planetary_computer as pc
        from pyproj import Transformer

        catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                              modifier=pc.sign_inplace)
        search  = catalog.search(collections=["cop-dem-glo-30"], bbox=BBOX)
        items   = list(search.items())
        if not items:
            return None

        with rasterio.open(items[0].assets["data"].href) as src:
            t = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            mnx, mny = t.transform(BBOX[0], BBOX[1])
            mxx, mxy = t.transform(BBOX[2], BBOX[3])
            win  = from_bounds(mnx, mny, mxx, mxy, src.transform)
            dem  = src.read(1, window=win).astype("float32")
            nd   = src.nodata
            h, w = int(round(win.height)), int(round(win.width))
            prof = src.profile.copy()
            prof.update(dtype="float32", count=1, nodata=-9999, compress="lzw",
                        height=h, width=w,
                        transform=rasterio.windows.transform(win, src.transform))

        if nd is not None:
            dem = np.where(dem == nd, np.nan, dem)
        dem = np.where(dem < -500, np.nan, dem)

        out = os.path.join(OUT_DIR, "fallback_dem.tif")
        with rasterio.open(out, "w", **prof) as dst:
            dst.write(np.nan_to_num(dem, nan=-9999).astype("float32"), 1)
        print(f"  [OK] DEM downloaded: {out}")
        return out
    except Exception as e:
        print(f"  [WARN] DEM download failed: {e}")
        return None


# ---------------------------------------------------------------------------
# 3. Run zonal statistics
# ---------------------------------------------------------------------------
def run_zonal(gdf, raster_paths):
    try:
        from rasterstats import zonal_stats
    except ImportError:
        raise ImportError(
            "rasterstats not installed.\n"
            "Run: pip install rasterstats\n"
            "or:  mamba install -n geocascade_env -c conda-forge rasterstats -y"
        )

    all_results = []

    for raster_name, raster_path in raster_paths.items():
        print(f"  Computing zonal stats: {raster_name}...")

        # Reproject zones to match raster CRS
        with rasterio.open(raster_path) as src:
            raster_crs = src.crs

        gdf_proj = gdf.to_crs(raster_crs)

        stats = zonal_stats(
            gdf_proj, raster_path,
            stats=["mean", "std", "min", "max", "median", "count"],
            nodata=-9999,
            all_touched=False
        )

        for i, row in gdf.iterrows():
            s = stats[i]
            if s is None or s.get("count", 0) == 0:
                continue
            all_results.append({
                "Zone":    row["Zone"],
                "Raster":  raster_name,
                "Mean":    round(float(s["mean"]),   4) if s.get("mean")   is not None else None,
                "Std":     round(float(s["std"]),    4) if s.get("std")    is not None else None,
                "Min":     round(float(s["min"]),    4) if s.get("min")    is not None else None,
                "Max":     round(float(s["max"]),    4) if s.get("max")    is not None else None,
                "Median":  round(float(s["median"]), 4) if s.get("median") is not None else None,
                "Count":   int(s["count"]),
            })

    df = pd.DataFrame(all_results)
    return df


# ---------------------------------------------------------------------------
# 4. Pivot to wide format for easy comparison
# ---------------------------------------------------------------------------
def pivot_wide(df):
    """One row per zone, one column per raster statistic."""
    pivot = df.pivot_table(index="Zone", columns="Raster", values="Mean")
    pivot.columns = [f"{c}_mean" for c in pivot.columns]
    pivot = pivot.reset_index()
    return pivot


# ---------------------------------------------------------------------------
# 5. Multi-panel dark figure: heatmap + representative maps
# ---------------------------------------------------------------------------
def plot_zonal(df, gdf, raster_paths):
    print("\n  Building zonal statistics figure...")

    # Pivot to zone x raster matrix for heatmap
    pivot = df.pivot_table(index="Zone", columns="Raster", values="Mean")

    n_rasters = len(pivot.columns)
    n_zones   = len(pivot.index)

    fig = plt.figure(figsize=(max(18, n_rasters * 2.5), 14), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.6, 1],
                            hspace=0.1, wspace=0.35,
                            top=0.92, bottom=0.12, left=0.06, right=0.97)
    fig.text(0.5, 0.97, "Zonal Statistics Summary -- Torres del Paine",
             ha="center", color=C_TEXT, fontsize=13, fontweight="bold")

    # Panel 1: Heatmap
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(DARK_AX)
    mat = pivot.values.astype("float32")
    # Normalize each column to [0,1] for unified color scale
    col_min = np.nanmin(mat, axis=0)
    col_max = np.nanmax(mat, axis=0)
    mat_norm = (mat - col_min) / (col_max - col_min + 1e-9)

    im = ax1.imshow(mat_norm, cmap="plasma", aspect="auto", vmin=0, vmax=1)
    ax1.set_xticks(range(n_rasters))
    ax1.set_xticklabels(pivot.columns, rotation=35, ha="right",
                        color=C_TEXT, fontsize=8)
    ax1.set_yticks(range(n_zones))
    ax1.set_yticklabels(pivot.index, color=C_TEXT, fontsize=8)
    ax1.set_title("Mean Values per Zone (column-normalized)",
                  color=C_TEXT, fontsize=9, fontweight="bold", pad=6)

    # Annotate cells with actual mean values
    for i in range(n_zones):
        for j in range(n_rasters):
            val = mat[i, j]
            if np.isfinite(val):
                ax1.text(j, i, f"{val:.3f}", ha="center", va="center",
                         color=C_TEXT, fontsize=6.5,
                         fontweight="bold" if mat_norm[i, j] > 0.7 else "normal")

    cb = plt.colorbar(im, ax=ax1, fraction=0.025, pad=0.03)
    cb.set_label("Normalized intensity", color=C_TEXT, fontsize=7)
    cb.ax.tick_params(colors=C_TEXT, labelsize=6)

    # Panel 2: Zone map (first available raster as background)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(DARK_AX)
    for sp in ax2.spines.values():
        sp.set_color("#30363d")
    ax2.tick_params(colors=C_TEXT, labelsize=7)

    if raster_paths:
        first_raster = list(raster_paths.values())[0]
        with rasterio.open(first_raster) as src:
            arr = src.read(1).astype("float32")
            nd  = src.nodata
            if nd is not None:
                arr = np.where(arr == nd, np.nan, arr)
            bounds = src.bounds

        valid = arr[np.isfinite(arr)]
        p2, p98 = (np.percentile(valid, 2), np.percentile(valid, 98)) if valid.size > 0 else (0, 1)
        ax2.imshow(arr, cmap="YlGn", vmin=p2, vmax=p98, aspect="auto",
                   extent=[bounds.left, bounds.right, bounds.bottom, bounds.top])

    # Plot zone boundaries
    gdf_plot = gdf.to_crs("EPSG:4326") if gdf.crs and str(gdf.crs) != "EPSG:4326" else gdf
    for _, row in gdf_plot.iterrows():
        geom = row.geometry
        xs, ys = geom.exterior.xy
        ax2.plot(xs, ys, color="#e74c3c", lw=1.5)
        cx, cy = geom.centroid.x, geom.centroid.y
        ax2.text(cx, cy, row["Zone"], ha="center", va="center",
                 color=C_TEXT, fontsize=6.5, fontweight="bold")

    ax2.set_title(f"Analysis Zones ({len(gdf)} polygons)",
                  color=C_TEXT, fontsize=9, fontweight="bold", pad=6)
    ax2.set_xlabel("Longitude", color=C_TEXT, fontsize=8)
    ax2.set_ylabel("Latitude",  color=C_TEXT, fontsize=8)

    out_png = os.path.join(OUT_DIR, "zonal_statistics.png")
    fig.savefig(out_png, dpi=180, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  [OK] Zonal figure: {out_png}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print(" GEOCASCADE - ZONAL STATISTICS (CROSS-CHAPTER SUMMARY)")
    print(f" BBOX: {BBOX}")
    print("=" * 65)

    print("\n[1/5] Building analysis zones...")
    gdf = build_zones()

    print("\n[2/5] Locating raster inputs...")
    raster_paths = find_rasters()
    if not raster_paths:
        print("\n  ERROR: No raster inputs found and fallback download failed.")
        print("         Run Chapters 2-4 first to produce the required TIFs.")
        return

    print(f"\n  Found {len(raster_paths)} rasters to analyze.")

    print("\n[3/5] Running zonal statistics...")
    df = run_zonal(gdf, raster_paths)

    if df.empty:
        print("  WARNING: Zonal statistics returned no data. Check zone/raster CRS alignment.")
        return

    print("\n[4/5] Saving results...")
    # Long format CSV
    csv_long = os.path.join(OUT_DIR, "zonal_statistics_long.csv")
    df.to_csv(csv_long, index=False, encoding="utf-8")

    # Wide format CSV (one row per zone)
    df_wide = pivot_wide(df)
    csv_wide = os.path.join(OUT_DIR, "zonal_statistics_summary.csv")
    df_wide.to_csv(csv_wide, index=False, encoding="utf-8")
    print(f"  [OK] Long format  : {csv_long}")
    print(f"  [OK] Wide format  : {csv_wide}")

    # Print summary table
    print(f"\n  --- Zone Summary (Mean Values) ---")
    print(df_wide.to_string(index=False, float_format="{:.3f}".format))

    print("\n[5/5] Building figure...")
    plot_zonal(df, gdf, raster_paths)

    print("\n" + "=" * 65)
    print(" ZONAL STATISTICS COMPLETE")
    print("=" * 65)
    print(f"  Zones   : {os.path.join(OUT_DIR, 'management_zones.gpkg')}")
    print(f"  CSV     : {csv_wide}")
    print(f"  Figure  : {os.path.join(OUT_DIR, 'zonal_statistics.png')}")
    print()
    print("  ArcGIS Pro: Spatial Analyst > Zonal Statistics As Table")
    print("              Load management_zones.gpkg as zone field.")
    print("  ENVI 5.6  : Vector > Statistics for per-polygon extraction.")
    print("=" * 65)


if __name__ == "__main__":
    main()
