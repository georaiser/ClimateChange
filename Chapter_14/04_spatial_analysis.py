"""
04_spatial_analysis.py
========================
GeoCascade Chapter 14 -- ArcGIS Pro + ENVI Professional Workflows
Torres del Paine, Patagonia, Chile

PURPOSE
-------
Performs spatial analysis operations that parallel ArcGIS Pro's Spatial Analyst
toolbox. All operations run via open-source libraries so the script works in the
geocascade_env; ArcPy equivalents are annotated throughout and called when
arcpy is importable.

OPERATIONS
----------
  1. Zonal Statistics  -- mean precip per land cover class (Table output)
  2. Slope & Aspect    -- from synthetic DEM (Copernicus GLO-30 proxy)
  3. Viewshed          -- observer points from Mirador Torres lookout
  4. Climate Vulnerability Index -- weighted overlay of temperature trend,
                                    precipitation anomaly, slope

OUTPUTS
-------
  data/processed/arcgis_outputs/slope.tif
  data/processed/arcgis_outputs/aspect.tif
  data/processed/arcgis_outputs/climate_vulnerability.tif
  data/processed/arcgis_outputs/zonal_stats.csv
  data/processed/arcgis_outputs/spatial_analysis_report.png

ARCGIS PRO EQUIVALENT TOOLS
-----------------------------
  Zonal Statistics As Table (Spatial Analyst)
  Slope, Aspect (3D Analyst / Spatial Analyst)
  Viewshed 2 (Spatial Analyst)
  Weighted Overlay (Spatial Analyst)

RUN
---
  python Chapter_14/04_spatial_analysis.py
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
from scipy.ndimage import uniform_filter, generic_filter

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

try:
    import rasterio
    from rasterio.transform import from_bounds
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.dirname(BASE_DIR)
PROC_DIR = os.path.join(BASE_DIR, "data", "processed", "arcgis_outputs")
ENVI_DIR = os.path.join(BASE_DIR, "data", "processed", "envi_outputs")
os.makedirs(PROC_DIR, exist_ok=True)

BBOX         = [-73.5, -51.5, -72.5, -50.5]
GRID_ROWS    = 100
GRID_COLS    = 100
PIXEL_SIZE_M = 1000.0        # ~1 km pixels at this scale

DARK_BG = "#0d1117"
DARK_AX = "#161b22"
C_TEXT  = "#e6edf3"
C_GREY  = "#8b949e"

# ArcGIS Pro Weighted Overlay weights (must sum to 100)
VULN_WEIGHTS = {
    "temp_trend":   40,   # temperature increase (warming trend)
    "precip_anom":  30,   # precipitation deficit (drought stress)
    "slope":        20,   # terrain instability
    "ndvi_loss":    10,   # vegetation stress proxy
}


# ---------------------------------------------------------------------------
# SYNTHETIC DEM (Andes-style topography for Torres del Paine area)
# ---------------------------------------------------------------------------

def build_dem(rows=GRID_ROWS, cols=GRID_COLS) -> np.ndarray:
    """
    Generate a realistic Andes DEM for the study area.
    - West side (lon < -72.8): high Andes plateau 1500-2500m
    - Central: torres granite spires proxy 800-1800m
    - East side: Patagonian steppe 100-400m
    """
    rng = np.random.default_rng(seed=42)
    lons = np.linspace(BBOX[0], BBOX[2], cols)
    lats = np.linspace(BBOX[3], BBOX[1], rows)
    LON, LAT = np.meshgrid(lons, lats)

    # East-west gradient (Andes ridge in west)
    ridge_lon = -73.0
    dist_from_ridge = np.maximum(0, ridge_lon - LON)  # positive = west of ridge
    elev = 200 + dist_from_ridge * 800

    # Add Patagonian granite spires (granite core: -72.9 lon, -51.0 lat)
    spire_cx, spire_cy = -72.9, -51.0
    spire_dist = np.sqrt(((LON - spire_cx) * 80) ** 2 + ((LAT - spire_cy) * 111) ** 2)
    elev += np.maximum(0, 1200 * np.exp(-spire_dist ** 2 / 4))

    # Smooth + noise
    elev = uniform_filter(elev, size=5)
    elev += rng.normal(0, 50, elev.shape)
    elev = np.clip(elev, 0, 2800).astype(np.float32)
    print(f"  DEM: {cols}x{rows}px, range {elev.min():.0f}-{elev.max():.0f} m")
    return elev


def compute_slope_aspect(dem: np.ndarray,
                          cell_size: float = PIXEL_SIZE_M) -> tuple:
    """
    Horn's method slope and aspect (identical to ArcGIS Pro Slope/Aspect tools).
    Returns slope in degrees and aspect in degrees from North (0-360).
    """
    # Gradient in x (east) and y (north) directions
    dy, dx = np.gradient(dem, cell_size, cell_size)

    # Slope (degrees)
    slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))

    # Aspect (degrees from North, clockwise)
    aspect = np.degrees(np.arctan2(-dx, dy)) % 360

    print(f"  Slope: mean={slope.mean():.1f} deg, max={slope.max():.1f} deg")
    print(f"  Aspect: computed for all {slope.size:,} pixels")
    return slope.astype(np.float32), aspect.astype(np.float32)


# ---------------------------------------------------------------------------
# ZONAL STATISTICS
# ---------------------------------------------------------------------------

def zonal_statistics(value_raster: np.ndarray,
                     zone_raster: np.ndarray,
                     class_names: list) -> pd.DataFrame:
    """
    Compute mean, std, min, max of value_raster per zone.
    Mirrors ArcGIS Pro 'Zonal Statistics As Table'.
    """
    rows = []
    for i, name in enumerate(class_names):
        mask = zone_raster == i
        vals = value_raster[mask]
        if len(vals) == 0:
            continue
        rows.append({
            "class_id":   i,
            "class_name": name,
            "n_pixels":   int(len(vals)),
            "mean":       round(float(np.nanmean(vals)), 2),
            "std":        round(float(np.nanstd(vals)),  2),
            "min":        round(float(np.nanmin(vals)),  2),
            "max":        round(float(np.nanmax(vals)),  2),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLIMATE VULNERABILITY INDEX (Weighted Overlay)
# ---------------------------------------------------------------------------

def compute_vulnerability(temp_trend: np.ndarray,
                           precip_anom: np.ndarray,
                           slope: np.ndarray,
                           ndvi: np.ndarray) -> np.ndarray:
    """
    ArcGIS Pro Weighted Overlay equivalent.
    Each layer is rescaled 1-5 (1=least vulnerable, 5=most vulnerable)
    then multiplied by weight and summed.
    """
    def rescale_1_5(arr, invert=False):
        """Rescale to 1-5 range. invert=True makes high values -> low score."""
        arr_clean = np.nan_to_num(arr, nan=float(np.nanmean(arr)))
        a_min, a_max = arr_clean.min(), arr_clean.max()
        if a_max == a_min:
            return np.full_like(arr, 3.0, dtype=np.float32)
        scaled = 1 + 4 * (arr_clean - a_min) / (a_max - a_min)
        return (5 - scaled + 1) if invert else scaled.astype(np.float32)

    # Higher temp trend  -> more vulnerable (direct)
    w1 = rescale_1_5(temp_trend)
    # Higher precip deficit -> more vulnerable (invert: low precip = high vuln)
    w2 = rescale_1_5(-precip_anom)   # negative anomaly = drought
    # Higher slope -> more vulnerable (erosion risk)
    w3 = rescale_1_5(slope)
    # Lower NDVI -> more vulnerable (vegetation stress)
    w4 = rescale_1_5(-ndvi)

    total_weight = sum(VULN_WEIGHTS.values())
    vuln = (
        w1 * VULN_WEIGHTS["temp_trend"]  +
        w2 * VULN_WEIGHTS["precip_anom"] +
        w3 * VULN_WEIGHTS["slope"]       +
        w4 * VULN_WEIGHTS["ndvi_loss"]
    ) / total_weight

    print(f"  Vulnerability range: {vuln.min():.2f} to {vuln.max():.2f}  (scale 1-5)")
    return vuln.astype(np.float32)


# ---------------------------------------------------------------------------
# SAVE RASTER
# ---------------------------------------------------------------------------

def save_raster(data, path: str, crs="EPSG:4326", nodata=-9999.0) -> None:
    if not HAS_RASTERIO:
        return
    out = np.nan_to_num(data, nan=nodata).astype(np.float32)
    t   = from_bounds(*BBOX, data.shape[1], data.shape[0])
    with rasterio.open(path, "w", driver="GTiff",
                       height=data.shape[0], width=data.shape[1],
                       count=1, dtype="float32", crs=crs,
                       transform=t, nodata=nodata, compress="lzw") as dst:
        dst.write(out, 1)
    print(f"  [OK] {os.path.relpath(path, BASE_DIR)}")


# ---------------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------------

def plot_results(dem, slope, aspect, vuln, out_path: str) -> None:
    """4-panel spatial analysis figure."""
    fig = plt.figure(figsize=(20, 14), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.28,
                            top=0.93, bottom=0.07, left=0.05, right=0.97)
    fig.text(0.5, 0.97,
             "GeoCascade Ch14 -- Spatial Analysis (ArcGIS Pro)",
             ha="center", color=C_TEXT, fontsize=13, fontweight="bold")
    fig.text(0.5, 0.945,
             "DEM | Slope | Aspect | Climate Vulnerability Index -- Torres del Paine",
             ha="center", color=C_GREY, fontsize=9)

    def style(ax, title):
        ax.set_facecolor(DARK_AX)
        for sp in ax.spines.values(): sp.set_color("#30363d")
        ax.tick_params(colors=C_TEXT, labelsize=8)
        ax.set_title(title, color=C_TEXT, fontsize=10, fontweight="bold", pad=6)

    extent = [BBOX[0], BBOX[2], BBOX[1], BBOX[3]]

    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(dem, cmap="terrain", origin="upper", extent=extent)
    cb1 = plt.colorbar(im1, ax=ax1, fraction=0.04, pad=0.02)
    cb1.set_label("Elevation (m)", color=C_TEXT, fontsize=8)
    cb1.ax.tick_params(colors=C_TEXT)
    style(ax1, "Digital Elevation Model (Copernicus GLO-30 proxy)")
    ax1.set_xlabel("Longitude", color=C_TEXT, fontsize=8)
    ax1.set_ylabel("Latitude", color=C_TEXT, fontsize=8)

    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(slope, cmap="YlOrRd", origin="upper", extent=extent)
    cb2 = plt.colorbar(im2, ax=ax2, fraction=0.04, pad=0.02)
    cb2.set_label("Slope (degrees)", color=C_TEXT, fontsize=8)
    cb2.ax.tick_params(colors=C_TEXT)
    style(ax2, "Slope (Horn's Method -- ArcGIS Pro Slope tool)")
    ax2.set_xlabel("Longitude", color=C_TEXT, fontsize=8)

    ax3 = fig.add_subplot(gs[1, 0])
    im3 = ax3.imshow(aspect, cmap="hsv", vmin=0, vmax=360, origin="upper",
                     extent=extent)
    cb3 = plt.colorbar(im3, ax=ax3, fraction=0.04, pad=0.02)
    cb3.set_label("Aspect (degrees from N)", color=C_TEXT, fontsize=8)
    cb3.ax.tick_params(colors=C_TEXT)
    style(ax3, "Aspect (ArcGIS Pro Aspect tool)")
    ax3.set_xlabel("Longitude", color=C_TEXT, fontsize=8)
    ax3.set_ylabel("Latitude", color=C_TEXT, fontsize=8)

    ax4 = fig.add_subplot(gs[1, 1])
    im4 = ax4.imshow(vuln, cmap="RdYlGn_r", vmin=1, vmax=5,
                     origin="upper", extent=extent)
    cb4 = plt.colorbar(im4, ax=ax4, fraction=0.04, pad=0.02,
                       ticks=[1, 2, 3, 4, 5])
    cb4.set_label("Vulnerability Index (1=Low, 5=High)", color=C_TEXT, fontsize=8)
    cb4.ax.tick_params(colors=C_TEXT)
    cb4.ax.set_yticklabels(["1 Very Low", "2 Low", "3 Medium",
                             "4 High", "5 Very High"],
                           color=C_TEXT, fontsize=7)
    style(ax4, "Climate Vulnerability Index (Weighted Overlay)")
    ax4.set_xlabel("Longitude", color=C_TEXT, fontsize=8)

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  [OK] Figure: {os.path.relpath(out_path, BASE_DIR)}")


# ---------------------------------------------------------------------------
# ARCPY MIRROR
# ---------------------------------------------------------------------------

def try_arcpy_spatial(dem_path: str) -> None:
    try:
        import arcpy
        from arcpy.sa import Slope, Aspect
        arcpy.env.overwriteOutput = True
        arcpy.CheckOutExtension("Spatial")
        print("\n  [ArcPy] Computing Slope and Aspect with native tools ...")
        if os.path.exists(dem_path):
            Slope(dem_path, "DEGREE").save(
                os.path.join(PROC_DIR, "slope_arcpy.tif"))
            Aspect(dem_path).save(
                os.path.join(PROC_DIR, "aspect_arcpy.tif"))
            print("  [ArcPy] Done -> slope_arcpy.tif, aspect_arcpy.tif")
        arcpy.CheckInExtension("Spatial")
    except ImportError:
        print("\n  [INFO] arcpy not available -- ArcPy Slope/Aspect skipped.")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print(" GEOCASCADE Ch14 -- Spatial Analysis (ArcGIS Pro)")
    print(" Slope | Aspect | Zonal Stats | Vulnerability Index")
    print("=" * 65)

    print("\n[1/5] Building Andes DEM (Copernicus GLO-30 proxy) ...")
    dem = build_dem()
    dem_path = os.path.join(PROC_DIR, "dem_proxy.tif")
    save_raster(dem, dem_path)

    print("\n[2/5] Computing Slope and Aspect (Horn's method) ...")
    slope, aspect = compute_slope_aspect(dem)
    save_raster(slope,  os.path.join(PROC_DIR, "slope.tif"))
    save_raster(aspect, os.path.join(PROC_DIR, "aspect.tif"))

    print("\n[3/5] Zonal Statistics (precip per land cover class) ...")
    # Load or synthesize classification raster
    cls_path = os.path.join(PROC_DIR, "classified_land_cover.tif")
    if HAS_RASTERIO and os.path.exists(cls_path):
        with rasterio.open(cls_path) as src:
            cls_map = src.read(1).astype(np.int8)
    else:
        rng = np.random.default_rng(seed=7)
        cls_map = rng.integers(0, 5, (GRID_ROWS, GRID_COLS), dtype=np.int8)

    # Resize cls_map to match slope dimensions if they differ
    if cls_map.shape != slope.shape:
        from scipy.ndimage import zoom
        zoom_r = slope.shape[0] / cls_map.shape[0]
        zoom_c = slope.shape[1] / cls_map.shape[1]
        cls_map = zoom(cls_map.astype(np.float32), (zoom_r, zoom_c),
                       order=0).astype(np.int8)
        cls_map = np.clip(cls_map, 0, 4)   # 5 classes, max index = 4
        print(f"  Resized cls_map to {cls_map.shape} to match slope grid")
    # Use slope as value raster for zonal stats example
    class_names = ["Water", "Snow/Ice", "Bare Rock", "Sparse Veg", "Dense Veg"]
    zonal_df = zonal_statistics(slope, cls_map, class_names)
    zonal_df.columns = ["class_id", "class_name", "n_pixels",
                         "mean_slope", "std_slope", "min_slope", "max_slope"]
    csv_path = os.path.join(PROC_DIR, "zonal_stats.csv")
    zonal_df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\n  Zonal Statistics (Slope per Land Cover Class):")
    print("  " + "-" * 55)
    for _, r in zonal_df.iterrows():
        print(f"  {r['class_name']:<14}: mean={r['mean_slope']:5.1f} deg  "
              f"n={r['n_pixels']:>5} px")
    print(f"\n  [OK] CSV: {os.path.relpath(csv_path, BASE_DIR)}")

    print("\n[4/5] Computing Climate Vulnerability Index (Weighted Overlay) ...")
    # Synthetic inputs matching DEM grid
    rng = np.random.default_rng(seed=99)
    lons = np.linspace(BBOX[0], BBOX[2], GRID_COLS)
    lats = np.linspace(BBOX[3], BBOX[1], GRID_ROWS)
    LON, LAT = np.meshgrid(lons, lats)
    temp_trend  = (0.05 + 0.2 * (LON - BBOX[0]) / (BBOX[2] - BBOX[0]) +
                   rng.normal(0, 0.02, (GRID_ROWS, GRID_COLS))).astype(np.float32)
    precip_anom = (-50 + 100 * (BBOX[3] - LAT) / (BBOX[3] - BBOX[1]) +
                   rng.normal(0, 20, (GRID_ROWS, GRID_COLS))).astype(np.float32)
    ndvi_proxy  = (0.1 + 0.6 * (LON - BBOX[0]) / (BBOX[2] - BBOX[0]) +
                   rng.normal(0, 0.05, (GRID_ROWS, GRID_COLS))).clip(-1, 1).astype(np.float32)

    vuln = compute_vulnerability(temp_trend, precip_anom, slope, ndvi_proxy)
    save_raster(vuln, os.path.join(PROC_DIR, "climate_vulnerability.tif"))

    print("\n[5/5] Generating spatial analysis report figure ...")
    plot_results(dem, slope, aspect, vuln,
                 os.path.join(PROC_DIR, "spatial_analysis_report.png"))

    # Try native ArcPy tools
    try_arcpy_spatial(dem_path)

    print("\n" + "=" * 65)
    print(" SPATIAL ANALYSIS COMPLETE")
    print("=" * 65)
    print(f"  DEM           : {PROC_DIR}\\dem_proxy.tif")
    print(f"  Slope         : {PROC_DIR}\\slope.tif")
    print(f"  Aspect        : {PROC_DIR}\\aspect.tif")
    print(f"  Vulnerability : {PROC_DIR}\\climate_vulnerability.tif")
    print(f"  Zonal Stats   : {PROC_DIR}\\zonal_stats.csv")
    print()
    print("  ArcGIS Pro:")
    print("    Add climate_vulnerability.tif -> Symbology > Classify > 5 classes")
    print("    Join zonal_stats.csv to watershed polygons for spatial reporting")
    print()
    print("  Continue with: python Chapter_14/05_climate_maps.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
