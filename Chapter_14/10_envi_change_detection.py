"""
10_envi_change_detection.py
============================
GeoCascade Chapter 14 -- ArcGIS Pro + ENVI Professional Workflows
Torres del Paine, Patagonia, Chile

# NOTE: This script mirrors the ENVI Python API workflow.
# In production with ENVI license:
#   envi = ENVI()
#   task = envi.Task('ChangeDetection')
#   task['INPUT_RASTER1'] = raster_2019
#   task['INPUT_RASTER2'] = raster_2023
#   task.execute()

PURPOSE
-------
Performs multi-date change detection using NDVI time series (2019 vs 2023):
  1. NDVI Difference    -- pixel-wise change magnitude
  2. Binary Change Map  -- threshold-based (|delta| > 0.1)
  3. Directional Change -- Vegetation Gain / Loss / No Change
  4. Change Area Stats  -- area in km² per change class
  5. Time Series Plot   -- NDVI trajectory at representative points

Uses realistic synthetic NDVI arrays reflecting known Patagonian processes:
  - Forest fire scar recovery (south-east study area)
  - Glacial retreat zone (north-west)
  - Stable lenga beech zone (south)

OUTPUTS
-------
  data/processed/envi_outputs/ndvi_2019.tif
  data/processed/envi_outputs/ndvi_2023.tif
  data/processed/envi_outputs/ndvi_change_2019_2023.tif   -- change magnitude
  data/processed/envi_outputs/ndvi_change_class.tif       -- 3-class change map
  data/processed/envi_outputs/change_detection_report.png -- 4-panel figure
  data/processed/envi_outputs/change_area_stats.csv

ENVI 5.6 EQUIVALENT
--------------------
  Toolbox > Change Detection > Image Change Workflow
    Step 1: Select before/after images -> compute indices
    Step 2: Threshold -> binary change map
    Step 3: Attribute -> directional change classes
  Toolbox > Change Detection > Thematic Change Detection

ARCGIS PRO EQUIVALENT
----------------------
  Image Classification Wizard > Classify > Change Detection
  Raster Functions > Band Difference
  Spatial Analyst > Raster Calculator: Con(raster > 0.1, 1, 0)

RUN
---
  python Chapter_14/10_envi_change_detection.py
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
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy.ndimage import uniform_filter, gaussian_filter

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
ENVI_DIR = os.path.join(BASE_DIR, "data", "processed", "envi_outputs")
os.makedirs(ENVI_DIR, exist_ok=True)

BBOX       = [-73.5, -51.5, -72.5, -50.5]
GRID_SHAPE = (120, 120)

# Change thresholds (standard in Patagonian vegetation studies)
LOSS_THRESH  = -0.10   # NDVI delta < -0.10  -> Vegetation Loss
GAIN_THRESH  =  0.10   # NDVI delta > +0.10  -> Vegetation Gain

# Approximate pixel area at 45°S latitude with 0.01-degree grid
LON_DEG  = (BBOX[2] - BBOX[0]) / GRID_SHAPE[1]   # degrees per pixel
LAT_DEG  = (BBOX[3] - BBOX[1]) / GRID_SHAPE[0]
PIXEL_KM2 = (LON_DEG * 111.32 * np.cos(np.radians(-51.0))) * (LAT_DEG * 111.32)

DARK_BG = "#0d1117"
DARK_AX = "#161b22"
C_TEXT  = "#e6edf3"
C_GREY  = "#8b949e"
C_RED   = "#e74c3c"
C_GOLD  = "#f39c12"
C_GREEN = "#2ecc71"

CHANGE_COLORS = [C_RED, "#8b949e", C_GREEN]   # Loss, No Change, Gain
CHANGE_NAMES  = ["Vegetation Loss", "No Change", "Vegetation Gain"]


# ---------------------------------------------------------------------------
# SYNTHETIC NDVI GENERATION (Patagonia 2019 and 2023)
# ---------------------------------------------------------------------------

def build_ndvi_year(year: int, shape=GRID_SHAPE) -> np.ndarray:
    """
    Generate a realistic annual maximum NDVI composite for a given year.
    Changes incorporated:
    - 2023 vs 2019: fire scar recovery (SE area, +0.12 NDVI)
    - Glacial retreat (NW area, -0.08 NDVI as bare rock exposed)
    - Drought stress in steppe (central, -0.06 NDVI)
    - General greening trend (whole scene, +0.01)
    """
    rng  = np.random.default_rng(seed=year)
    rows, cols = shape
    lons = np.linspace(BBOX[0], BBOX[2], cols)
    lats = np.linspace(BBOX[3], BBOX[1], rows)
    LON, LAT = np.meshgrid(lons, lats)

    # Baseline: east -> dense forest, west -> scrub/bare
    base = (-0.1 + 0.75 * (LON - BBOX[0]) / (BBOX[2] - BBOX[0]) +
            rng.normal(0, 0.06, shape))

    # --- Static features ---
    # Lake (Lago Grey / Nordenskjold area)
    base[rows//3:rows//2, cols//3:cols//2] = rng.uniform(-0.5, -0.2,
                                                          (rows//6, cols//6))
    # Glacier (NW)
    base[:rows//5, :cols//4] = rng.uniform(-0.3, 0.0, (rows//5, cols//4))

    # --- Year-specific changes ---
    if year >= 2023:
        # Forest fire scar recovery (SE corner, fire in 2021 -> recovering in 2023)
        fire_mask_r = slice(int(rows*0.65), int(rows*0.85))
        fire_mask_c = slice(int(cols*0.65), int(cols*0.90))
        base[fire_mask_r, fire_mask_c] += rng.uniform(0.08, 0.15,
                                                       (fire_mask_r.stop - fire_mask_r.start,
                                                        fire_mask_c.stop - fire_mask_c.start))
        # Glacial retreat -> bare rock exposed (NDVI decreases)
        base[:rows//5, :cols//4] -= rng.uniform(0.05, 0.12, (rows//5, cols//4))
        # Drought stress in central steppe
        drought_r = slice(int(rows*0.35), int(rows*0.55))
        drought_c = slice(int(cols*0.25), int(cols*0.55))
        base[drought_r, drought_c] -= rng.uniform(0.03, 0.08,
                                                   (drought_r.stop - drought_r.start,
                                                    drought_c.stop - drought_c.start))
        # Global warming greening
        base += 0.012

    ndvi = gaussian_filter(np.clip(base, -1, 1), sigma=1.5).astype(np.float32)
    print(f"  NDVI {year}: mean={ndvi.mean():.3f}  "
          f"std={ndvi.std():.3f}  "
          f"range [{ndvi.min():.3f}, {ndvi.max():.3f}]")
    return ndvi


# ---------------------------------------------------------------------------
# CHANGE DETECTION
# ---------------------------------------------------------------------------

def compute_change(ndvi_a: np.ndarray, ndvi_b: np.ndarray) -> tuple:
    """
    Compute NDVI change map and classify into Loss / No Change / Gain.
    Returns:
      delta       -- continuous change magnitude array
      change_cls  -- integer class map {0=Loss, 1=NoChange, 2=Gain}
    """
    delta = ndvi_b - ndvi_a                          # positive = greening

    change_cls = np.ones_like(delta, dtype=np.int8)  # default: No Change
    change_cls[delta < LOSS_THRESH] = 0              # Loss
    change_cls[delta > GAIN_THRESH] = 2              # Gain

    return delta.astype(np.float32), change_cls


def compute_area_stats(delta: np.ndarray,
                        change_cls: np.ndarray) -> pd.DataFrame:
    """Compute area statistics for each change class."""
    rows_list = []
    for i, name in enumerate(CHANGE_NAMES):
        mask     = change_cls == i
        n_px     = int(mask.sum())
        area_km2 = round(n_px * PIXEL_KM2, 1)
        mean_d   = round(float(delta[mask].mean()) if n_px > 0 else 0.0, 4)
        rows_list.append({
            "class_id":   i,
            "class_name": name,
            "n_pixels":   n_px,
            "area_km2":   area_km2,
            "pct":        round(n_px / delta.size * 100, 1),
            "mean_delta": mean_d,
        })
    return pd.DataFrame(rows_list)


# ---------------------------------------------------------------------------
# SAVE
# ---------------------------------------------------------------------------

def save_raster_f32(data: np.ndarray, path: str,
                    nodata: float = -9999.0) -> None:
    if not HAS_RASTERIO:
        return
    rows, cols = data.shape
    t = from_bounds(*BBOX, cols, rows)
    out = np.nan_to_num(data.astype(np.float32), nan=nodata)
    with rasterio.open(path, "w", driver="GTiff", height=rows, width=cols,
                       count=1, dtype="float32", crs="EPSG:4326",
                       transform=t, nodata=nodata, compress="lzw") as dst:
        dst.write(out, 1)
    print(f"  [OK] {os.path.relpath(path, BASE_DIR)}")


def save_raster_int8(data: np.ndarray, path: str) -> None:
    if not HAS_RASTERIO:
        return
    rows, cols = data.shape
    t = from_bounds(*BBOX, cols, rows)
    with rasterio.open(path, "w", driver="GTiff", height=rows, width=cols,
                       count=1, dtype="int8", crs="EPSG:4326",
                       transform=t, nodata=-1, compress="lzw") as dst:
        dst.write(data, 1)
    print(f"  [OK] {os.path.relpath(path, BASE_DIR)}")


# ---------------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------------

def plot_change_report(ndvi_2019, ndvi_2023, delta, change_cls,
                       stats_df: pd.DataFrame, out_path: str) -> None:
    """4-panel change detection figure."""
    fig = plt.figure(figsize=(22, 14), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.28,
                            top=0.92, bottom=0.07, left=0.05, right=0.97)
    fig.text(0.5, 0.97,
             "GeoCascade Ch14 -- ENVI Change Detection: NDVI 2019 vs 2023",
             ha="center", color=C_TEXT, fontsize=13, fontweight="bold")
    fig.text(0.5, 0.945,
             "Torres del Paine, Patagonia  |  "
             "Threshold: Loss < -0.10, Gain > +0.10",
             ha="center", color=C_GREY, fontsize=9)

    def style(ax, title):
        ax.set_facecolor(DARK_AX)
        for sp in ax.spines.values(): sp.set_color("#30363d")
        ax.tick_params(colors=C_TEXT, labelsize=7)
        ax.set_title(title, color=C_TEXT, fontsize=10, fontweight="bold", pad=6)

    ext = [BBOX[0], BBOX[2], BBOX[1], BBOX[3]]

    # Panel 1: NDVI 2019
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(ndvi_2019, cmap="RdYlGn", vmin=-0.5, vmax=0.9,
                     origin="upper", extent=ext)
    cb1 = plt.colorbar(im1, ax=ax1, fraction=0.04, pad=0.02)
    cb1.set_label("NDVI", color=C_TEXT, fontsize=8)
    cb1.ax.tick_params(colors=C_TEXT)
    style(ax1, f"NDVI 2019 (baseline)  mean={ndvi_2019.mean():.3f}")
    ax1.set_xlabel("Longitude", color=C_TEXT, fontsize=8)
    ax1.set_ylabel("Latitude", color=C_TEXT, fontsize=8)

    # Panel 2: NDVI 2023
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(ndvi_2023, cmap="RdYlGn", vmin=-0.5, vmax=0.9,
                     origin="upper", extent=ext)
    cb2 = plt.colorbar(im2, ax=ax2, fraction=0.04, pad=0.02)
    cb2.set_label("NDVI", color=C_TEXT, fontsize=8)
    cb2.ax.tick_params(colors=C_TEXT)
    style(ax2, f"NDVI 2023 (current)  mean={ndvi_2023.mean():.3f}")
    ax2.set_xlabel("Longitude", color=C_TEXT, fontsize=8)

    # Panel 3: NDVI change magnitude
    ax3 = fig.add_subplot(gs[1, 0])
    vabs = max(abs(float(delta.min())), abs(float(delta.max())))
    im3 = ax3.imshow(delta, cmap="RdYlGn", vmin=-vabs, vmax=vabs,
                     origin="upper", extent=ext)
    cb3 = plt.colorbar(im3, ax=ax3, fraction=0.04, pad=0.02)
    cb3.set_label("NDVI delta (2023-2019)", color=C_TEXT, fontsize=8)
    cb3.ax.tick_params(colors=C_TEXT)
    ax3.axhline(-51.0, color=C_GOLD, lw=0.5, alpha=0.5)
    style(ax3, "NDVI Change Magnitude (2023 - 2019)")
    ax3.set_xlabel("Longitude", color=C_TEXT, fontsize=8)
    ax3.set_ylabel("Latitude", color=C_TEXT, fontsize=8)

    # Panel 4: Classified change map + area bar chart
    ax4 = fig.add_subplot(gs[1, 1])
    chg_cmap = ListedColormap(CHANGE_COLORS)
    ax4.imshow(change_cls, cmap=chg_cmap, vmin=-0.5, vmax=2.5,
               origin="upper", extent=ext)
    legend_patches = [
        Patch(facecolor=CHANGE_COLORS[i],
              label=f"{CHANGE_NAMES[i]}\n{stats_df.iloc[i]['area_km2']:.0f} km² "
                    f"({stats_df.iloc[i]['pct']:.1f}%)")
        for i in range(3)
    ]
    ax4.legend(handles=legend_patches, fontsize=7, facecolor=DARK_BG,
               labelcolor=C_TEXT, loc="lower left", framealpha=0.8)
    style(ax4, "Change Classification (Loss | No Change | Gain)")
    ax4.set_xlabel("Longitude", color=C_TEXT, fontsize=8)

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  [OK] Figure: {os.path.relpath(out_path, BASE_DIR)}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print(" GEOCASCADE Ch14 -- ENVI Change Detection")
    print(" NDVI 2019 vs 2023  |  Loss | No Change | Gain")
    print("=" * 65)

    print("\n[1/5] Generating NDVI composites ...")
    ndvi_2019 = build_ndvi_year(2019)
    ndvi_2023 = build_ndvi_year(2023)

    print("\n[2/5] Saving NDVI rasters ...")
    save_raster_f32(ndvi_2019, os.path.join(ENVI_DIR, "ndvi_2019.tif"))
    save_raster_f32(ndvi_2023, os.path.join(ENVI_DIR, "ndvi_2023.tif"))

    print("\n[3/5] Computing change map ...")
    delta, change_cls = compute_change(ndvi_2019, ndvi_2023)
    save_raster_f32(delta, os.path.join(ENVI_DIR, "ndvi_change_2019_2023.tif"),
                    nodata=-9999.0)
    save_raster_int8(change_cls,
                     os.path.join(ENVI_DIR, "ndvi_change_class.tif"))

    print("\n[4/5] Computing area statistics ...")
    stats_df = compute_area_stats(delta, change_cls)
    print(f"\n  {'Class':<18} {'Pixels':>8} {'Area km2':>10} {'Pct':>7} {'Mean delta':>11}")
    print("  " + "-" * 60)
    for _, r in stats_df.iterrows():
        print(f"  {r['class_name']:<18} {r['n_pixels']:>8,} "
              f"{r['area_km2']:>10.1f} {r['pct']:>6.1f}% "
              f"{r['mean_delta']:>+11.4f}")

    csv_path = os.path.join(ENVI_DIR, "change_area_stats.csv")
    stats_df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\n  [OK] CSV: {os.path.relpath(csv_path, BASE_DIR)}")

    print("\n[5/5] Generating change detection report figure ...")
    plot_change_report(
        ndvi_2019, ndvi_2023, delta, change_cls, stats_df,
        os.path.join(ENVI_DIR, "change_detection_report.png")
    )

    print("\n" + "=" * 65)
    print(" CHANGE DETECTION COMPLETE")
    print("=" * 65)
    print(f"  NDVI 2019  : {ENVI_DIR}\\ndvi_2019.tif")
    print(f"  NDVI 2023  : {ENVI_DIR}\\ndvi_2023.tif")
    print(f"  Change mag : {ENVI_DIR}\\ndvi_change_2019_2023.tif")
    print(f"  Change cls : {ENVI_DIR}\\ndvi_change_class.tif")
    print(f"  Stats CSV  : {ENVI_DIR}\\change_area_stats.csv")
    print(f"  Report PNG : {ENVI_DIR}\\change_detection_report.png")
    print()
    print("  ENVI 5.6:")
    print("    Toolbox > Change Detection > Image Change Workflow")
    print("    Input image 1: ndvi_2019.tif")
    print("    Input image 2: ndvi_2023.tif")
    print("    Threshold method: Standard Deviation (2-sigma)")
    print()
    print("  ArcGIS Pro:")
    print("    Raster Functions > Band Arithmetic: (raster1 - raster2)")
    print("    Spatial Analyst > Raster Calculator:")
    print("      Con(\"ndvi_change\" < -0.1, 0, Con(\"ndvi_change\" > 0.1, 2, 1))")
    print()
    print("  Chapter 14 COMPLETE. See Chapter_14/README.md for full curriculum.")
    print("=" * 65)


if __name__ == "__main__":
    main()
