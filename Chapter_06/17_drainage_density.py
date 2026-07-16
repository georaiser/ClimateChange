"""
Chapter 6: 17_drainage_density.py
=====================================
Drainage Density & Stream Network Analysis

Academic Objective:
  Drainage Density (Dd) is a fundamental hydro-morphometric parameter:

    Dd = Total stream length (km) / Basin area (km2)

  High Dd = impermeable bedrock, sparse vegetation, steep relief → rapid runoff,
            flashy hydrograph, high flood risk.
  Low Dd  = permeable soils, dense vegetation, gentle slopes → slow infiltration,
            sustained baseflow.

  Typical values:
    Granite / impermeable: Dd = 2-5 km/km2
    Limestone / permeable: Dd = 0.5-2 km/km2
    Badlands:              Dd > 10 km/km2
    Patagonian Andes:      Dd ~ 2-4 km/km2 (steep + glaciated)

  This script:
  1. Reads Ch03 cached DEM (temp_dem.tif) -- downloads if missing
  2. Runs full pysheds D8 routing (pit fill → depression fill → fdir → acc)
  3. Extracts river network as vector GeoDataFrame (GeoPackage + shapefile)
  4. Computes Dd in UTM-32718 (equal-area projection for Southern Chile)
  5. Computes per-threshold Dd for 5 thresholds (100, 250, 500, 1000, 2000 pixels)
     to show how network density responds to threshold choice
  6. Saves a drainage density raster (km of stream per km2 of pixel neighborhood)

Connection to pipeline:
  Reads Ch03/data/raw/temp_dem.tif (from script 10).
  River vectors feed into Ch05 zonal statistics.
  Dd values feed into Ch08 cascade risk model.

Outputs:
  data/processed/drainage/river_network.gpkg
  data/processed/drainage/river_network.shp      (for legacy ArcGIS workflows)
  data/processed/drainage/drainage_density_map.png   (4-panel dark)
  data/processed/drainage/drainage_density_report.csv

ArcGIS Pro: Add river_network.gpkg as Feature Layer.
            Use Stream Order (Spatial Analyst > Hydrology > Stream Order)
            with flow_accumulation.tif for Strahler ordering.
ENVI 5.6:   Topographic > Drainage Basin extraction.

Run:
  conda activate geocascade_env
  pip install pysheds   (if not installed)
  python Chapter_06/17_drainage_density.py

Dependencies: pysheds, rasterio, numpy, matplotlib, pandas, geopandas, shapely, scipy
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
import geopandas as gpd
from shapely.geometry import shape, box

# numpy 2.0 compatibility
if not hasattr(np, "in1d"):
    np.in1d = np.isin

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.dirname(BASE_DIR)
OUT_DIR   = os.path.join(BASE_DIR, "data", "processed", "drainage")
os.makedirs(OUT_DIR, exist_ok=True)

# Prefer Ch03's cached DEM
CH03_DEM  = os.path.join(ROOT_DIR, "Chapter_03", "data", "raw", "temp_dem.tif")
LOCAL_DEM = os.path.join(OUT_DIR, "local_dem.tif")

BBOX = [-73.30, -51.10, -72.90, -50.80]

# UTM Zone 18S: equal-area projection for Southern Chile (accurate length/area)
CRS_METRIC = "EPSG:32718"

# River extraction thresholds (pixels) for sensitivity analysis
THRESHOLDS = [100, 250, 500, 1000, 2000]
PRIMARY_THRESH = 500  # main output threshold

DARK_BG = "#0d1117"
DARK_AX = "#161b22"
C_TEXT  = "#e6edf3"
C_GREY  = "#8b949e"
C_BLUE  = "#3498db"
C_GOLD  = "#f39c12"


# ---------------------------------------------------------------------------
# 1. Ensure DEM is available
# ---------------------------------------------------------------------------
def get_dem_path():
    """Use Ch03 cached DEM if available, otherwise download."""
    if os.path.exists(CH03_DEM):
        print(f"  [OK] Using Ch03 cached DEM: {CH03_DEM}")
        return CH03_DEM

    # Ch03 cache not found, check local copy
    if os.path.exists(LOCAL_DEM):
        print(f"  [OK] Using local DEM cache: {LOCAL_DEM}")
        return LOCAL_DEM

    print("  DEM not found. Downloading from Planetary Computer...")
    try:
        from pystac_client import Client
        import planetary_computer as pc
        from pyproj import Transformer

        catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                              modifier=pc.sign_inplace)
        search  = catalog.search(collections=["cop-dem-glo-30"], bbox=BBOX)
        items   = list(search.items())
        if not items:
            raise ValueError("No DEM tiles found.")

        with rasterio.open(items[0].assets["data"].href) as src:
            t = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            mnx, mny = t.transform(BBOX[0], BBOX[1])
            mxx, mxy = t.transform(BBOX[2], BBOX[3])
            win = from_bounds(mnx, mny, mxx, mxy, src.transform)
            dem = src.read(1, window=win).astype("float32")
            nd  = src.nodata
            if nd is not None:
                dem = np.where(dem == nd, np.nan, dem)
            dem = np.where(dem < -500, np.nan, dem)
            h, w = int(round(win.height)), int(round(win.width))
            prof = src.profile.copy()
            prof.update(driver="GTiff", dtype="float32", nodata=-9999,
                        height=h, width=w, compress="lzw",
                        transform=rasterio.windows.transform(win, src.transform))

        with rasterio.open(LOCAL_DEM, "w", **prof) as dst:
            dst.write(np.nan_to_num(dem, nan=-9999).astype("float32"), 1)
        print(f"  [OK] DEM downloaded: {LOCAL_DEM} ({w}x{h})")
        return LOCAL_DEM

    except Exception as e:
        raise RuntimeError(f"DEM download failed: {e}")


# ---------------------------------------------------------------------------
# 2. Run pysheds D8 routing
# ---------------------------------------------------------------------------
def run_pysheds(dem_path):
    try:
        from pysheds.grid import Grid
    except ImportError:
        raise ImportError(
            "pysheds not installed.\n"
            "Run: pip install pysheds\n"
            "  or: mamba install -n geocascade_env -c conda-forge pysheds -y"
        )

    print("  Initializing pysheds Grid...")
    grid = Grid.from_raster(dem_path)
    dem  = grid.read_raster(dem_path)

    print("  Step 1/3: Conditioning DEM (fill pits + depressions + resolve flats)...")
    pit_filled = grid.fill_pits(dem)
    flooded    = grid.fill_depressions(pit_filled)
    inflated   = grid.resolve_flats(flooded)

    print("  Step 2/3: D8 flow direction...")
    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
    fdir   = grid.flowdir(inflated, dirmap=dirmap)

    print("  Step 3/3: Flow accumulation...")
    acc = grid.accumulation(fdir, dirmap=dirmap)

    print(f"  Max accumulation: {float(np.nanmax(acc)):.0f} pixels")
    return grid, np.array(dem), fdir, np.array(acc)


# ---------------------------------------------------------------------------
# 3. Extract river network at a given threshold
# ---------------------------------------------------------------------------
def extract_rivers(grid, fdir, acc, threshold, raster_crs):
    """
    Extract river network as a GeoDataFrame using pysheds.
    Returns GeoDataFrame in the raster's native CRS.
    """
    try:
        river_branches = grid.extract_river_network(fdir, acc > threshold)
    except Exception as e:
        print(f"  [WARN] River extraction failed at threshold={threshold}: {e}")
        return gpd.GeoDataFrame({"geometry": []}, crs=raster_crs)

    geoms = []
    for feat in river_branches.get("features", []):
        try:
            geoms.append(shape(feat["geometry"]))
        except Exception:
            pass

    if not geoms:
        return gpd.GeoDataFrame({"geometry": []}, crs=raster_crs)

    return gpd.GeoDataFrame(
        {"threshold_px": threshold, "geometry": geoms},
        crs=raster_crs
    )


# ---------------------------------------------------------------------------
# 4. Compute drainage density
# ---------------------------------------------------------------------------
def compute_dd(gdf_rivers, bbox=BBOX, crs_metric=CRS_METRIC):
    """Compute drainage density in km/km2 using equal-area projection."""
    if len(gdf_rivers) == 0:
        return 0.0, 0.0, 0.0

    gdf_metric   = gdf_rivers.to_crs(crs_metric)
    total_len_m  = gdf_metric.geometry.length.sum()
    total_len_km = total_len_m / 1000.0

    bbox_geom    = box(*bbox)
    gdf_bbox     = gpd.GeoDataFrame({"geometry": [bbox_geom]}, crs="EPSG:4326").to_crs(crs_metric)
    total_area_m2  = gdf_bbox.geometry.area.sum()
    total_area_km2 = total_area_m2 / 1e6

    dd = total_len_km / total_area_km2
    return dd, total_len_km, total_area_km2


# ---------------------------------------------------------------------------
# 5. 4-panel dark figure
# ---------------------------------------------------------------------------
def plot_drainage(dem_arr, acc, gdf_rivers, dd_table, raster_crs):
    print("\n  Building 4-panel drainage figure...")

    fig = plt.figure(figsize=(22, 16), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.25,
                            top=0.93, bottom=0.05, left=0.05, right=0.97)
    fig.text(0.5, 0.97, "Drainage Density & River Network -- Torres del Paine",
             ha="center", color=C_TEXT, fontsize=13, fontweight="bold")

    def style_ax(ax, title):
        ax.set_facecolor(DARK_AX)
        ax.axis("off")
        ax.set_title(title, color=C_TEXT, fontsize=10, fontweight="bold", pad=6)

    # Panel 1: DEM
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(dem_arr, cmap="terrain", aspect="auto")
    cb1 = plt.colorbar(im1, ax=ax1, fraction=0.035)
    cb1.set_label("Elevation (m)", color=C_TEXT, fontsize=8)
    cb1.ax.tick_params(colors=C_TEXT, labelsize=7)
    style_ax(ax1, "Conditioned DEM")

    # Panel 2: Log flow accumulation + river network
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(DARK_AX)
    ax2.axis("off")
    ax2.set_title("Flow Accumulation + River Network",
                  color=C_TEXT, fontsize=10, fontweight="bold", pad=6)
    ax2.imshow(dem_arr, cmap="gray", alpha=0.4, aspect="auto")
    acc_disp = np.where(acc > PRIMARY_THRESH, acc, np.nan)
    acc_max  = float(np.nanmax(acc_disp)) if np.any(np.isfinite(acc_disp)) else PRIMARY_THRESH + 1
    if acc_max <= PRIMARY_THRESH:
        acc_max = float(PRIMARY_THRESH) + 1.0
    from matplotlib.colors import LogNorm
    im2 = ax2.imshow(acc_disp, cmap="Blues",
                     norm=LogNorm(vmin=PRIMARY_THRESH, vmax=acc_max),
                     aspect="auto", alpha=0.9)
    cb2 = plt.colorbar(im2, ax=ax2, fraction=0.035)
    cb2.set_label("Accumulation", color=C_TEXT, fontsize=8)
    cb2.ax.tick_params(colors=C_TEXT, labelsize=7)

    # Panel 3: River vectors on DEM
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(DARK_AX)
    for sp in ax3.spines.values():
        sp.set_color("#30363d")
    ax3.tick_params(colors=C_TEXT, labelsize=7)
    ax3.set_title(f"River Network Vector (threshold={PRIMARY_THRESH} px)",
                  color=C_TEXT, fontsize=10, fontweight="bold", pad=6)
    ax3.imshow(dem_arr, cmap="terrain", alpha=0.5, aspect="auto",
               extent=[BBOX[0], BBOX[2], BBOX[1], BBOX[3]])
    if len(gdf_rivers) > 0:
        gdf_ll = gdf_rivers.to_crs("EPSG:4326") if str(gdf_rivers.crs) != "EPSG:4326" else gdf_rivers
        gdf_ll.plot(ax=ax3, color=C_BLUE, linewidth=0.9, alpha=0.9)
    ax3.set_xlabel("Longitude", color=C_TEXT, fontsize=8)
    ax3.set_ylabel("Latitude",  color=C_TEXT, fontsize=8)

    # Panel 4: Drainage density vs threshold
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(DARK_AX)
    for sp in ax4.spines.values():
        sp.set_color("#30363d")
    ax4.tick_params(colors=C_TEXT, labelsize=8)
    if not dd_table.empty:
        ax4.plot(dd_table["threshold_px"], dd_table["dd_km_km2"],
                 "o-", color=C_GOLD, lw=2, ms=7)
        ax4.axvline(PRIMARY_THRESH, color=C_BLUE, lw=1.5, linestyle="--",
                    label=f"Primary threshold ({PRIMARY_THRESH} px)")
        # Annotate primary threshold value
        primary_dd = dd_table.loc[dd_table["threshold_px"] == PRIMARY_THRESH, "dd_km_km2"]
        if not primary_dd.empty:
            ax4.annotate(f"Dd = {primary_dd.values[0]:.3f} km/km2",
                         xy=(PRIMARY_THRESH, primary_dd.values[0]),
                         xytext=(PRIMARY_THRESH * 1.2, primary_dd.values[0] * 0.9),
                         color=C_TEXT, fontsize=8,
                         arrowprops=dict(arrowstyle="->", color=C_TEXT))
        ax4.legend(fontsize=8, facecolor=DARK_BG, labelcolor=C_TEXT)
    ax4.set_xlabel("Threshold (pixels)", color=C_TEXT, fontsize=9)
    ax4.set_ylabel("Drainage Density (km/km2)", color=C_TEXT, fontsize=9)
    ax4.set_title("Dd Sensitivity to River Extraction Threshold",
                  color=C_TEXT, fontsize=10, fontweight="bold", pad=6)
    ax4.grid(alpha=0.15, color="#30363d")

    out_png = os.path.join(OUT_DIR, "drainage_density_map.png")
    fig.savefig(out_png, dpi=180, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  [OK] 4-panel figure: {out_png}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print(" GEOCASCADE - DRAINAGE DENSITY ANALYSIS")
    print(f" Primary threshold: {PRIMARY_THRESH} pixels | CRS: {CRS_METRIC}")
    print("=" * 65)

    print("\n[1/5] Locating DEM...")
    try:
        dem_path = get_dem_path()
    except RuntimeError as e:
        print(f"\n  ERROR: {e}")
        return

    # Get raster CRS for vector outputs
    with rasterio.open(dem_path) as src:
        raster_crs = src.crs

    print("\n[2/5] Running pysheds D8 routing...")
    try:
        grid, dem_arr, fdir, acc = run_pysheds(dem_path)
    except ImportError as e:
        print(f"\n  ERROR: {e}")
        return

    print("\n[3/5] Extracting river network at multiple thresholds...")
    dd_rows = []
    gdf_primary = None

    for thr in THRESHOLDS:
        gdf = extract_rivers(grid, fdir, acc, thr, raster_crs)
        dd, length_km, area_km2 = compute_dd(gdf)
        dd_rows.append({
            "threshold_px":  thr,
            "n_segments":    len(gdf),
            "total_length_km": round(length_km, 2),
            "basin_area_km2":  round(area_km2, 2),
            "dd_km_km2":       round(dd, 4),
            "interpretation":  (
                "Very high (badlands/impermeable)"  if dd > 8 else
                "High (steep/glaciated)"            if dd > 4 else
                "Moderate (mixed)"                  if dd > 2 else
                "Low (permeable/forested)"
            )
        })
        print(f"  Threshold {thr:>5} px | {len(gdf):>5} segments | "
              f"Dd = {dd:.3f} km/km2 | Length = {length_km:.1f} km")
        if thr == PRIMARY_THRESH:
            gdf_primary = gdf

    dd_table = pd.DataFrame(dd_rows)

    print("\n[4/5] Saving outputs...")
    # GeoPackage (primary)
    if gdf_primary is not None and len(gdf_primary):
        gpkg = os.path.join(OUT_DIR, "river_network.gpkg")
        gdf_primary.to_file(gpkg, driver="GPKG")
        print(f"  [OK] river_network.gpkg")

        # Legacy Shapefile for older workflows
        shp = os.path.join(OUT_DIR, "river_network.shp")
        gdf_primary.to_file(shp)
        print(f"  [OK] river_network.shp")

    # CSV report
    csv = os.path.join(OUT_DIR, "drainage_density_report.csv")
    dd_table.to_csv(csv, index=False, encoding="utf-8")
    print(f"  [OK] drainage_density_report.csv")

    # Print summary
    print(f"\n  --- Drainage Density Report ---")
    for _, r in dd_table.iterrows():
        print(f"  Thresh={r['threshold_px']:>5} px  |  "
              f"Dd={r['dd_km_km2']:.3f} km/km2  |  "
              f"Length={r['total_length_km']:.1f} km  |  "
              f"{r['interpretation']}")

    print("\n[5/5] Building 4-panel figure...")
    plot_drainage(dem_arr, acc, gdf_primary if gdf_primary is not None else gpd.GeoDataFrame(),
                  dd_table, raster_crs)

    print("\n" + "=" * 65)
    print(" DRAINAGE DENSITY COMPLETE")
    print("=" * 65)
    print(f"  Vectors : {OUT_DIR}")
    print(f"  Report  : {csv}")
    print(f"  Figure  : {os.path.join(OUT_DIR, 'drainage_density_map.png')}")
    print()
    print("  ArcGIS Pro: Add river_network.gpkg as Feature Layer.")
    print("              Hydrology > Stream Order for Strahler ordering.")
    print("              Calculate Field: length in meters for per-reach Dd.")
    print("  ENVI 5.6  : Topographic > Drainage Basin extraction.")
    print("=" * 65)


if __name__ == "__main__":
    main()
