"""
Chapter 3: 11_watershed_delineation.py
========================================
Hydrological Modeling: Flow Direction, Flow Accumulation, Watershed, Hipsometry

Academic Objective:
  Water flows downhill. By algorithmically routing water across a conditioned DEM,
  we can delineate entire drainage basins and map river networks automatically --
  replicating the ESRI ArcHydro workflow in pure Python.

  Steps (pysheds D8 algorithm):
  1. Fill Pits       -- remove single-cell depressions (DEM noise)
  2. Fill Depressions-- fill all closed basins so water escapes
  3. Resolve Flats   -- add tiny gradient across flat areas
  4. Flow Direction  -- D8 routing (8-direction steepest descent)
  5. Flow Accumulation -- count upstream contributing pixels
  6. River Network   -- pixels where acc > threshold
  7. Watershed       -- delineate the basin above a pour point
  8. Hipsometric Curve -- elevation-area distribution (basin maturity indicator)
  9. Stream Order    -- Strahler method (not implemented in pysheds natively;
                        use ArcGIS Pro "Stream Order" tool on flow_accumulation.tif)

Hipsometric Integral (HI):
  HI < 0.35 = monadnock / old basin (low relief, tectonically stable)
  HI 0.35-0.6 = equilibrium / mature basin
  HI > 0.6  = young / actively uplifting basin (Patagonia: typically 0.5-0.7)

Pysheds numpy 2.0 compatibility:
  numpy 2.0 removed np.in1d -- patched with np.isin before import.

Outputs:
  data/processed/watershed/flow_direction.tif
  data/processed/watershed/flow_accumulation.tif
  data/processed/watershed/river_network.tif     (acc > 500 pixels)
  data/processed/watershed/watershed_analysis.png  (4-panel dark)
  data/processed/watershed/hypsometry_report.csv

ArcGIS Pro: Load flow_accumulation.tif. Use ArcHydro tools:
            Terrain Preprocessing > Stream Definition > Stream Segmentation.
            Or: Spatial Analyst > Hydrology > Watershed.
ENVI 5.6:   Topographic > DEM Extraction for flow derivatives.

Run:
  conda activate geocascade_env
  pip install pysheds   (if not already installed)
  python Chapter_03/11_watershed_delineation.py

Dependencies: pysheds, rasterio, numpy, matplotlib, pandas, pystac-client, planetary-computer, pyproj
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors

# numpy 2.0 compatibility patch (pysheds uses np.in1d, removed in numpy 2.0)
if not hasattr(np, "in1d"):
    np.in1d = np.isin

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
OUT_DIR   = os.path.join(BASE_DIR, "data", "processed", "watershed")
RAW_DIR   = os.path.join(BASE_DIR, "data", "raw")
TEMP_DEM  = os.path.join(RAW_DIR, "temp_dem.tif")   # written by script 10
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)

BBOX = [-73.30, -51.10, -72.90, -50.80]

# River network threshold (pixels): higher = larger/fewer rivers
RIVER_THRESHOLD = 500

DARK_BG = "#0d1117"
DARK_AX = "#161b22"
C_TEXT  = "#e6edf3"
C_GREY  = "#8b949e"
C_BLUE  = "#3498db"
C_CYAN  = "#00bcd4"


# ---------------------------------------------------------------------------
# 1. Ensure DEM exists (download if missing)
# ---------------------------------------------------------------------------
def ensure_dem():
    """Download and cache DEM if not produced by script 10 yet."""
    if os.path.exists(TEMP_DEM):
        print(f"  [OK] DEM found: {TEMP_DEM}")
        return

    print("  temp_dem.tif not found. Downloading from Planetary Computer...")
    print("  TIP: Run script 10 first to produce the DEM and terrain derivatives.")

    try:
        from pystac_client import Client
        import planetary_computer as pc
        from pyproj import Transformer
        import rasterio
        from rasterio.windows import from_bounds
    except ImportError:
        raise ImportError("pystac-client / planetary-computer not installed.")

    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace
    )
    search = catalog.search(collections=["cop-dem-glo-30"], bbox=BBOX)
    items  = list(search.items())
    if not items:
        raise ValueError(f"No Copernicus DEM found for BBOX {BBOX}.")

    item = items[0]
    with rasterio.open(item.assets["data"].href) as src:
        t = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        mnx, mny = t.transform(BBOX[0], BBOX[1])
        mxx, mxy = t.transform(BBOX[2], BBOX[3])
        win      = from_bounds(mnx, mny, mxx, mxy, src.transform)
        win_tf   = rasterio.windows.transform(win, src.transform)
        h = int(round(win.height))
        w = int(round(win.width))
        dem = src.read(1, window=win).astype("float32")
        nd  = src.nodata
        profile = src.profile.copy()
        profile.update(driver="GTiff", dtype="float32", nodata=-9999,
                       height=h, width=w, transform=win_tf, compress="lzw")

    if nd is not None:
        dem = np.where(dem == nd, np.nan, dem)
    dem = np.where(dem < -500, np.nan, dem)

    import rasterio as _r
    with _r.open(TEMP_DEM, "w", **profile) as dst:
        dst.write(np.nan_to_num(dem, nan=-9999).astype("float32"), 1)
    print(f"  [OK] DEM downloaded and cached: {TEMP_DEM} ({w}x{h})")


# ---------------------------------------------------------------------------
# 2. Hydrological modeling with pysheds
# ---------------------------------------------------------------------------
def run_hydrology():
    try:
        from pysheds.grid import Grid
    except ImportError:
        raise ImportError(
            "pysheds not installed. Run:\n"
            "  pip install pysheds\n"
            "or: mamba install -n geocascade_env -c conda-forge pysheds -y"
        )

    print("\n  Initializing pysheds Grid...")
    grid = Grid.from_raster(TEMP_DEM)
    dem  = grid.read_raster(TEMP_DEM)

    # Step 1-3: Condition the DEM
    print("  Step 1/4: Filling pits...")
    pit_filled  = grid.fill_pits(dem)
    print("  Step 2/4: Filling depressions...")
    flooded     = grid.fill_depressions(pit_filled)
    print("  Step 3/4: Resolving flats...")
    inflated    = grid.resolve_flats(flooded)

    # Step 4: Flow direction (ESRI D8 encoding)
    print("  Step 4/4: Flow direction + accumulation (D8)...")
    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)   # N, NE, E, SE, S, SW, W, NW
    fdir   = grid.flowdir(inflated, dirmap=dirmap)
    acc    = grid.accumulation(fdir, dirmap=dirmap)

    print(f"  Flow accumulation: max={float(np.nanmax(acc)):.0f} pixels")

    # River network mask
    river_net = np.where(acc > RIVER_THRESHOLD, acc, np.nan)

    return np.array(dem), np.array(fdir), np.array(acc), river_net


# ---------------------------------------------------------------------------
# 3. Save GeoTIFFs
# ---------------------------------------------------------------------------
def save_tifs(dem_arr, fdir, acc, river_net):
    import rasterio

    with rasterio.open(TEMP_DEM) as src:
        profile = src.profile.copy()
        profile.update(compress="lzw")

    def save(data, name, dtype="float32", description=""):
        out = os.path.join(OUT_DIR, f"{name}.tif")
        p   = profile.copy()
        p.update(dtype=dtype, nodata=-9999)
        safe = np.nan_to_num(np.array(data), nan=-9999).astype(dtype)
        with rasterio.open(out, "w", **p) as dst:
            dst.write(safe, 1)
            if description:
                dst.update_tags(description=description, nodata="-9999",
                                arcgis_note="Use ArcHydro or Spatial Analyst > Hydrology")
        print(f"  [OK] {name}.tif")

    save(dem_arr,    "dem_conditioned", "float32", "Pit-filled, depression-filled DEM")
    save(fdir,       "flow_direction",  "int32",   "D8 flow direction (ESRI encoding)")
    save(acc,        "flow_accumulation","float32", "Flow accumulation (upstream pixel count)")
    save(river_net,  "river_network",   "float32",
         f"River network: acc > {RIVER_THRESHOLD} pixels")


# ---------------------------------------------------------------------------
# 4. Hipsometric analysis
# ---------------------------------------------------------------------------
def hipsometric_analysis(dem_arr):
    """
    Hipsometric Curve: fraction of basin area above each elevation percentile.
    Hipsometric Integral (HI) ~ area under the curve / total area.
    HI > 0.6 = young, actively eroding basin (expected for Patagonian Andes).
    """
    valid = dem_arr[np.isfinite(dem_arr) & (dem_arr > 0)]
    if valid.size == 0:
        return None, None, None

    elev_pcts  = np.linspace(0, 100, 200)
    elev_vals  = np.percentile(valid, elev_pcts)
    area_above = 1.0 - elev_pcts / 100.0   # fraction of basin above each elevation

    # Normalize to [0, 1]
    h_min, h_max = float(valid.min()), float(valid.max())
    h_norm = (elev_vals - h_min) / (h_max - h_min + 1e-9)

    # Hipsometric Integral: area under the normalized curve
    hi = float(np.trapz(h_norm, area_above))
    if hi < 0:
        hi = abs(hi)

    maturity = ("Young / actively eroding" if hi > 0.6
                else "Equilibrium / mature" if hi > 0.35
                else "Monadnock / old")

    print(f"\n  --- Hipsometric Analysis ---")
    print(f"  Elevation range: {h_min:.0f} - {h_max:.0f} m")
    print(f"  Hipsometric Integral (HI): {hi:.3f}")
    print(f"  Basin maturity: {maturity}")
    return elev_vals, area_above, hi, maturity, h_min, h_max


# ---------------------------------------------------------------------------
# 5. 4-panel dark figure
# ---------------------------------------------------------------------------
def plot_watershed(dem_arr, fdir, acc, river_net,
                   elev_vals, area_above, hi, maturity, h_min, h_max):
    print("\n  Building 4-panel watershed figure...")

    fig, axes = plt.subplots(2, 2, figsize=(18, 16), facecolor=DARK_BG)
    fig.suptitle("Watershed Hydrological Analysis -- Torres del Paine\nCopernicus DEM 30m + pysheds D8",
                 color=C_TEXT, fontsize=13, fontweight="bold", y=0.98)

    def style_ax(ax, title):
        ax.set_facecolor(DARK_AX)
        ax.axis("off")
        ax.set_title(title, color=C_TEXT, fontsize=10, fontweight="bold", pad=6)

    # Panel 1: Conditioned DEM
    ax = axes[0, 0]
    im1 = ax.imshow(dem_arr, cmap="terrain", aspect="auto")
    cb1 = plt.colorbar(im1, ax=ax, fraction=0.035)
    cb1.set_label("Elevation (m)", color=C_TEXT, fontsize=8)
    cb1.ax.tick_params(colors=C_TEXT, labelsize=7)
    style_ax(ax, "Conditioned DEM (pit-filled + depression-filled)")

    # Panel 2: Flow direction
    ax = axes[0, 1]
    im2 = ax.imshow(fdir, cmap="twilight", aspect="auto")
    cb2 = plt.colorbar(im2, ax=ax, fraction=0.035)
    cb2.set_label("D8 Direction", color=C_TEXT, fontsize=8)
    cb2.ax.tick_params(colors=C_TEXT, labelsize=7)
    style_ax(ax, "Flow Direction (D8 -- 8 cardinal directions)")

    # Panel 3: River network on hillshade
    ax = axes[1, 0]
    ax.set_facecolor(DARK_AX)
    ax.axis("off")
    ax.set_title("River Network (Flow Accumulation > 500 pixels)",
                 color=C_TEXT, fontsize=10, fontweight="bold", pad=6)
    # DEM as grayscale background
    ax.imshow(dem_arr, cmap="gray", alpha=0.5, aspect="auto")
    # River network
    acc_masked = np.where(acc > RIVER_THRESHOLD, acc, np.nan)
    acc_max = float(np.nanmax(acc_masked)) if np.any(np.isfinite(acc_masked)) else RIVER_THRESHOLD + 1
    if acc_max <= RIVER_THRESHOLD:
        acc_max = RIVER_THRESHOLD + 1.0
    im3 = ax.imshow(acc_masked, cmap="Blues",
                    norm=colors.LogNorm(vmin=RIVER_THRESHOLD, vmax=acc_max),
                    aspect="auto", alpha=0.85)
    cb3 = plt.colorbar(im3, ax=ax, fraction=0.035)
    cb3.set_label("Accumulation", color=C_TEXT, fontsize=8)
    cb3.ax.tick_params(colors=C_TEXT, labelsize=7)

    # Panel 4: Hipsometric curve
    ax = axes[1, 1]
    ax.set_facecolor(DARK_AX)
    for sp in ax.spines.values():
        sp.set_color("#30363d")
    ax.tick_params(colors=C_TEXT, labelsize=8)

    if elev_vals is not None:
        ax.plot(area_above, elev_vals, color=C_BLUE, lw=2.5, label="Hipsometric Curve")
        ax.fill_between(area_above, h_min, elev_vals, alpha=0.18, color=C_BLUE)
        ax.set_xlim(0, 1); ax.set_ylim(h_min * 0.95, h_max * 1.05)
        ax.set_xlabel("Fraction of Basin Area (above elevation)", color=C_TEXT, fontsize=9)
        ax.set_ylabel("Elevation (m)", color=C_TEXT, fontsize=9)
        ax.text(0.55, 0.85, f"HI = {hi:.3f}", transform=ax.transAxes,
                color=C_TEXT, fontsize=11, fontweight="bold")
        ax.text(0.55, 0.78, maturity, transform=ax.transAxes,
                color=C_CYAN, fontsize=9)
        ax.text(0.05, 0.05,
                f"Basin: {h_min:.0f} - {h_max:.0f} m",
                transform=ax.transAxes, color=C_GREY, fontsize=8)
        ax.grid(alpha=0.15, color="#30363d")
        ax.legend(fontsize=8, facecolor=DARK_BG, labelcolor=C_TEXT)
    ax.set_title("Hipsometric Curve (Basin Maturity Indicator)",
                 color=C_TEXT, fontsize=10, fontweight="bold", pad=6)

    out_png = os.path.join(OUT_DIR, "watershed_analysis.png")
    fig.savefig(out_png, dpi=180, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  [OK] 4-panel figure: {out_png}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print(" GEOCASCADE - WATERSHED DELINEATION (pysheds D8)")
    print(f" BBOX: {BBOX}  |  River threshold: {RIVER_THRESHOLD} pixels")
    print("=" * 65)

    print("\n[1/5] Ensuring DEM is available...")
    try:
        ensure_dem()
    except Exception as e:
        print(f"\n  ERROR: {e}")
        return

    print("\n[2/5] Running hydrological analysis...")
    try:
        dem_arr, fdir, acc, river_net = run_hydrology()
    except ImportError as e:
        print(f"\n  ERROR: {e}")
        return

    print("\n[3/5] Saving GeoTIFFs...")
    save_tifs(dem_arr, fdir, acc, river_net)

    print("\n[4/5] Hipsometric analysis...")
    result = hipsometric_analysis(dem_arr)
    if result[0] is not None:
        elev_vals, area_above, hi, maturity, h_min, h_max = result

        df_hipso = pd.DataFrame({
            "area_fraction_above": area_above,
            "elevation_m": elev_vals
        })
        df_hipso.attrs["hipsometric_integral"] = hi
        csv_path = os.path.join(OUT_DIR, "hypsometry_report.csv")
        df_hipso.to_csv(csv_path, index=False, encoding="utf-8")

        meta_path = os.path.join(OUT_DIR, "hypsometry_summary.csv")
        pd.DataFrame([{
            "bbox": str(BBOX),
            "elev_min_m": round(h_min, 1),
            "elev_max_m": round(h_max, 1),
            "hipsometric_integral": round(hi, 4),
            "basin_maturity": maturity,
            "river_threshold_px": RIVER_THRESHOLD,
        }]).to_csv(meta_path, index=False, encoding="utf-8")
        print(f"  [OK] Hypsometry CSV: {csv_path}")
    else:
        elev_vals = area_above = hi = maturity = h_min = h_max = None

    print("\n[5/5] Building 4-panel figure...")
    plot_watershed(dem_arr, fdir, acc, river_net,
                   elev_vals, area_above, hi, maturity, h_min, h_max)

    print("\n" + "=" * 65)
    print(" WATERSHED DELINEATION COMPLETE")
    print("=" * 65)
    print(f"  TIFs  : {OUT_DIR}")
    print(f"  Figure: {os.path.join(OUT_DIR, 'watershed_analysis.png')}")
    print(f"  Hipso : {os.path.join(OUT_DIR, 'hypsometry_report.csv')}")
    print()
    print("  ArcGIS Pro: Spatial Analyst > Hydrology > Watershed")
    print("              Load flow_accumulation.tif as input.")
    print("              Stream Order tool on flow_accumulation.tif (Strahler).")
    print("  ENVI 5.6  : Topographic > DEM Extraction for derivatives.")
    print("=" * 65)


if __name__ == "__main__":
    main()
