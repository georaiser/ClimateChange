"""
Chapter 2: 08_automated_index_batcher.py
==========================================
Automated Temporal Batch Processing -- Full Year of Scenes

Academic Objective:
  Single-date spectral indices (Script 07) give a snapshot. Scientists need
  TEMPORAL STACKS to detect trends, seasonal patterns, and anomalies.
  This script automates index computation across an entire year, building:
    - Time series of mean NDVI, NDSI, NDWI per scene
    - Detecting peak-green season and glacier minimum
    - Identifying drought stress years using NDMI

  Conceptual link to GIS tools:
    ArcGIS Pro ModelBuilder: would build a visual loop with Iterator
    ENVI Classic:            Batch mode (File > Batch Process)
    Python:                  for-loop + rasterio -- portable, faster, free

NEW vs previous version:
  - Progress reporting per scene (scene N of M)
  - Save summary CSV per index + overall time_series.csv
  - matplotlib.use("Agg") + sys.stdout.reconfigure
  - ASCII-safe print statements (no Unicode arrows/checkmarks)
  - Dark-mode 4-panel temporal chart
  - Skip already-processed scenes (incremental mode)
  - Local Ch01 data check before STAC query
  - NDMI added to batch
  - Cloud cover filtering (< 10% threshold)
  - Per-scene GeoTIFF naming: {index}_{date}.tif

Outputs:
  data/processed/batch_indices/ndvi_{date}.tif  (one per cloud-free scene)
  data/processed/batch_indices/ndsi_{date}.tif
  data/processed/batch_indices/ndwi_{date}.tif
  data/processed/batch_indices/ndmi_{date}.tif
  data/processed/batch_indices/batch_time_series.csv
  data/processed/batch_indices/temporal_analysis.png

ArcGIS Pro: In batch_indices folder, use Analysis > Time Series to animate TIFs.
            OR: Load batch_time_series.csv, Insert > Chart > Line for NDVI trend.
ENVI 5.6:   File > Open > select all ndvi_*.tif files at once -> Time Series Analyst.

Run:
  conda activate geocascade_env
  python Chapter_02/08_automated_index_batcher.py

Dependencies: rasterio, numpy, pandas, matplotlib, pystac-client, planetary-computer, pyproj
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import from_bounds
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CH01_DIR   = os.path.join(os.path.dirname(BASE_DIR), "Chapter_01")
OUT_DIR    = os.path.join(BASE_DIR, "data", "processed", "batch_indices")
PLOT_DIR   = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(OUT_DIR,  exist_ok=True)

BBOX         = [-73.30, -51.10, -72.90, -50.80]
DATE_RANGE   = "2023-01-01/2023-12-31"
MAX_CLOUD    = 10    # percent cloud cover threshold
MAX_SCENES   = 24   # safety cap (full year ~ 24-48 scenes at < 10% cloud)
INCREMENTAL  = True  # skip already-processed scenes

DARK_BG = "#0d1117"
DARK_AX = "#161b22"
C_TEXT  = "#e6edf3"
C_GREY  = "#8b949e"
C_RED   = "#e74c3c"
C_BLUE  = "#3498db"
C_GREEN = "#2ecc71"
C_GOLD  = "#f39c12"
C_CYAN  = "#00bcd4"


# ---------------------------------------------------------------------------
# 1. Index functions (NaN-safe)
# ---------------------------------------------------------------------------
def safe_ratio(num, den, eps=1e-6):
    return np.where(np.abs(den) < eps, np.nan, num / den)

def calc_ndvi(nir, red):   return safe_ratio(nir - red, nir + red)
def calc_ndsi(green, swir): return safe_ratio(green - swir, green + swir)
def calc_ndwi(green, nir):  return safe_ratio(green - nir,  green + nir)
def calc_ndmi(nir, swir):   return safe_ratio(nir - swir,   nir + swir)


# ---------------------------------------------------------------------------
# 2. STAC search
# ---------------------------------------------------------------------------
def search_stac():
    try:
        from pystac_client import Client
        import planetary_computer as pc
    except ImportError:
        raise ImportError(
            "pystac-client / planetary-computer not installed.\n"
            "Run: mamba install -n geocascade_env -c conda-forge pystac-client planetary-computer -y"
        )

    print("  Querying Planetary Computer STAC API...")
    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace
    )
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=BBOX,
        datetime=DATE_RANGE,
        query={"eo:cloud_cover": {"lt": MAX_CLOUD}},
        sortby="+properties.datetime"
    )
    items = list(search.items())
    if not items:
        raise ValueError(
            f"No scenes found with cloud < {MAX_CLOUD}%. "
            "Try increasing MAX_CLOUD or widening DATE_RANGE."
        )

    # Sort by cloud cover (cleanest first), cap at MAX_SCENES
    items = sorted(items, key=lambda i: i.properties.get("eo:cloud_cover", 99))
    items = items[:MAX_SCENES]

    # Sort by date for temporal plotting
    items = sorted(items, key=lambda i: i.properties.get("datetime", ""))
    print(f"  [OK] Found {len(items)} cloud-free scenes (<{MAX_CLOUD}%)")
    return items


# ---------------------------------------------------------------------------
# 3. Load one scene's bands (windowed COG read)
# ---------------------------------------------------------------------------
def load_scene_bands(item):
    """
    Load B03(Green), B04(Red), B08(NIR), B11(SWIR1) for one STAC scene.
    B11 is 20m and requires an independent window with out_shape resampling.
    Returns: (bands_dict, rasterio_profile, date_str)
    """
    from pyproj import Transformer

    date_str = item.properties.get("datetime", "")[:10]
    cloud    = item.properties.get("eo:cloud_cover", 0)

    bands    = {}
    profile  = None
    win_10m  = None
    tshape   = None
    transf   = None

    # Establish 10m reference grid from B04 (Red)
    with rasterio.open(item.assets["B04"].href) as src:
        transf = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        minx, miny = transf.transform(BBOX[0], BBOX[1])
        maxx, maxy = transf.transform(BBOX[2], BBOX[3])
        win_10m    = from_bounds(minx, miny, maxx, maxy, src.transform)
        win_tf     = rasterio.windows.transform(win_10m, src.transform)
        tshape     = (int(round(win_10m.height)), int(round(win_10m.width)))
        profile    = src.profile.copy()
        profile.update(
            dtype="float32", count=1, nodata=-9999,
            height=tshape[0], width=tshape[1],
            transform=win_tf, compress="lzw"
        )
        bands["red"] = src.read(1, window=win_10m).astype("float32") / 10000.0

    # 10m bands (use same window)
    for band_key, asset_name in [("green", "B03"), ("nir", "B08")]:
        with rasterio.open(item.assets[asset_name].href) as src:
            bands[band_key] = src.read(1, window=win_10m).astype("float32") / 10000.0

    # B11 (20m) -- independent window with bilinear resampling to 10m shape
    if "B11" in item.assets:
        with rasterio.open(item.assets["B11"].href) as src_b11:
            t11 = Transformer.from_crs("EPSG:4326", src_b11.crs, always_xy=True)
            mnx, mny = t11.transform(BBOX[0], BBOX[1])
            mxx, mxy = t11.transform(BBOX[2], BBOX[3])
            win_b11  = from_bounds(mnx, mny, mxx, mxy, src_b11.transform)
            bands["swir1"] = src_b11.read(
                1, window=win_b11, out_shape=tshape,
                resampling=rasterio.enums.Resampling.bilinear
            ).astype("float32") / 10000.0
    else:
        bands["swir1"] = np.zeros(tshape, dtype="float32")

    return bands, profile, date_str, cloud


# ---------------------------------------------------------------------------
# 4. Process one scene: compute + save 4 indices
# ---------------------------------------------------------------------------
def process_scene(item, scene_num, total):
    date_str = item.properties.get("datetime", "")[:10]
    cloud    = item.properties.get("eo:cloud_cover", 0)

    print(f"\n  Scene {scene_num:2d}/{total}: {date_str}  cloud={cloud:.1f}%")

    # Incremental mode: skip if all 4 TIFs already exist
    if INCREMENTAL:
        existing = [os.path.join(OUT_DIR, f"{idx}_{date_str}.tif")
                    for idx in ["ndvi", "ndsi", "ndwi", "ndmi"]]
        if all(os.path.exists(p) for p in existing):
            print(f"    [SKIP] All 4 index TIFs already exist -- reading stats from disk")
            return _read_stats_from_disk(date_str, cloud)

    try:
        bands, profile, date_str_c, cloud_c = load_scene_bands(item)
    except Exception as e:
        print(f"    [ERROR] {e}")
        return None

    # Compute indices
    idx_ndvi = calc_ndvi(bands["nir"], bands["red"])
    idx_ndsi = calc_ndsi(bands["green"], bands["swir1"])
    idx_ndwi = calc_ndwi(bands["green"], bands["nir"])
    idx_ndmi = calc_ndmi(bands["nir"], bands["swir1"])

    indices = {"ndvi": idx_ndvi, "ndsi": idx_ndsi,
               "ndwi": idx_ndwi, "ndmi": idx_ndmi}

    # Save GeoTIFFs
    for name, arr in indices.items():
        out_path = os.path.join(OUT_DIR, f"{name}_{date_str}.tif")
        safe = np.nan_to_num(arr, nan=-9999).astype("float32")
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(safe, 1)
            dst.update_tags(date=date_str, cloud_cover=str(cloud),
                            index=name.upper(), nodata="-9999")

    # Compute scene-level stats
    stats = {"date": date_str, "cloud_pct": round(float(cloud), 2)}
    for name, arr in indices.items():
        valid = arr[np.isfinite(arr)]
        stats[f"{name}_mean"] = round(float(np.nanmean(valid)), 4) if valid.size > 0 else None
        stats[f"{name}_std"]  = round(float(np.nanstd(valid)),  4) if valid.size > 0 else None
        if name == "ndsi":
            stats["glacier_pct"] = round(float((valid > 0.4).mean() * 100), 2) if valid.size > 0 else None
        if name == "ndwi":
            stats["water_pct"] = round(float((valid > 0.3).mean() * 100), 2) if valid.size > 0 else None

    ndvi_m = stats.get("ndvi_mean", None)
    ndsi_m = stats.get("ndsi_mean", None)
    print(f"    NDVI mean={ndvi_m:+.3f}" if ndvi_m is not None else "    NDVI: N/A", end="")
    print(f"  NDSI mean={ndsi_m:+.3f}" if ndsi_m is not None else "  NDSI: N/A")
    return stats


def _read_stats_from_disk(date_str, cloud):
    """Read stats from existing TIFs (incremental mode)."""
    stats = {"date": date_str, "cloud_pct": round(float(cloud), 2)}
    for name in ["ndvi", "ndsi", "ndwi", "ndmi"]:
        p = os.path.join(OUT_DIR, f"{name}_{date_str}.tif")
        if os.path.exists(p):
            with rasterio.open(p) as src:
                arr = src.read(1).astype("float32")
                arr = np.where(arr == -9999, np.nan, arr)
                valid = arr[np.isfinite(arr)]
                stats[f"{name}_mean"] = round(float(np.nanmean(valid)), 4) if valid.size > 0 else None
                stats[f"{name}_std"]  = round(float(np.nanstd(valid)),  4) if valid.size > 0 else None
                if name == "ndsi":
                    stats["glacier_pct"] = round(float((valid > 0.4).mean() * 100), 2) if valid.size > 0 else None
                if name == "ndwi":
                    stats["water_pct"] = round(float((valid > 0.3).mean() * 100), 2) if valid.size > 0 else None
    return stats


# ---------------------------------------------------------------------------
# 5. Temporal analysis figure
# ---------------------------------------------------------------------------
def plot_temporal(ts_df):
    print("\n  Building temporal analysis figure...")
    ts_df["date"] = pd.to_datetime(ts_df["date"])
    ts_df = ts_df.sort_values("date").reset_index(drop=True)

    fig, axes = plt.subplots(2, 2, figsize=(20, 14), facecolor=DARK_BG)
    fig.suptitle("Temporal Spectral Index Analysis 2023 -- Torres del Paine",
                 color=C_TEXT, fontsize=14, fontweight="bold", y=0.98)

    def style_ax(ax, title, ylabel):
        ax.set_facecolor(DARK_AX)
        for sp in ax.spines.values():
            sp.set_color("#30363d")
        ax.tick_params(colors=C_TEXT, labelsize=9)
        ax.xaxis.label.set_color(C_TEXT)
        ax.yaxis.label.set_color(C_TEXT)
        ax.grid(alpha=0.15, color="#30363d")
        ax.set_title(title, color=C_TEXT, fontsize=10, fontweight="bold", pad=6)
        ax.set_ylabel(ylabel, color=C_TEXT, fontsize=9)
        ax.set_xlabel("Date", color=C_TEXT, fontsize=9)

    # Panel 1: NDVI seasonal cycle
    ax = axes[0, 0]
    col = "ndvi_mean"
    if col in ts_df.columns:
        ax.fill_between(ts_df["date"], ts_df[col].fillna(0), alpha=0.25, color=C_GREEN)
        ax.plot(ts_df["date"], ts_df[col], "o-", color=C_GREEN, lw=2, ms=5)
        if ts_df[col].notna().any():
            pk = ts_df.loc[ts_df[col].idxmax(), "date"]
            ax.axvline(pk, color=C_GOLD, lw=1.2, ls="--", alpha=0.7,
                       label=f"Peak NDVI: {pk.strftime('%b')}")
            ax.legend(fontsize=8, facecolor=DARK_BG, labelcolor=C_TEXT)
    style_ax(ax, "NDVI Seasonal Cycle", "Mean NDVI")

    # Panel 2: NDSI -- glacier tracking
    ax = axes[0, 1]
    col = "ndsi_mean"
    if col in ts_df.columns:
        ax.fill_between(ts_df["date"], ts_df[col].fillna(0), alpha=0.25, color=C_CYAN)
        ax.plot(ts_df["date"], ts_df[col], "s-", color=C_CYAN, lw=2, ms=5)
        ax.axhline(0.4, color=C_BLUE, lw=1, ls=":", label="Ice threshold (0.4)")
        if "glacier_pct" in ts_df.columns:
            ax2 = ax.twinx()
            ax2.bar(ts_df["date"], ts_df["glacier_pct"].fillna(0),
                    color=C_BLUE, alpha=0.2, width=5)
            ax2.set_ylabel("Glacier %", color=C_BLUE, fontsize=8)
            ax2.tick_params(colors=C_BLUE)
            for sp in ax2.spines.values():
                sp.set_color("#30363d")
        ax.legend(fontsize=8, facecolor=DARK_BG, labelcolor=C_TEXT)
    style_ax(ax, "NDSI -- Glacier / Snow Tracking", "Mean NDSI")

    # Panel 3: NDMI -- vegetation water stress
    ax = axes[1, 0]
    col = "ndmi_mean"
    if col in ts_df.columns:
        ax.fill_between(ts_df["date"], ts_df[col].fillna(0), alpha=0.25, color=C_GOLD)
        ax.plot(ts_df["date"], ts_df[col], "^-", color=C_GOLD, lw=2, ms=5)
        ax.axhline(0.0, color=C_RED, lw=0.8, ls=":", label="Stress threshold")
        ax.legend(fontsize=8, facecolor=DARK_BG, labelcolor=C_TEXT)
    style_ax(ax, "NDMI -- Canopy Water Content / Drought Stress", "Mean NDMI")

    # Panel 4: Multi-index comparison
    ax = axes[1, 1]
    colors_map = {"ndvi_mean": C_GREEN, "ndsi_mean": C_CYAN,
                  "ndwi_mean": C_BLUE,  "ndmi_mean": C_GOLD}
    labels_map = {"ndvi_mean": "NDVI", "ndsi_mean": "NDSI",
                  "ndwi_mean": "NDWI", "ndmi_mean": "NDMI"}
    for col_name, c in colors_map.items():
        if col_name in ts_df.columns and ts_df[col_name].notna().any():
            ax.plot(ts_df["date"], ts_df[col_name], "o-", color=c, lw=1.5, ms=4,
                    label=labels_map[col_name], alpha=0.85)
    ax.axhline(0, color=C_GREY, lw=0.6, ls=":")
    ax.legend(fontsize=9, facecolor=DARK_BG, labelcolor=C_TEXT, ncol=2)
    style_ax(ax, "Multi-Index Overlay (Normalised)", "Index value")

    # Format all x-axes
    import matplotlib.dates as mdates
    for ax_row in axes:
        for ax in ax_row:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    out_png = os.path.join(PLOT_DIR, "batch_temporal_analysis.png")
    fig.savefig(out_png, dpi=180, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  [OK] Temporal figure: {out_png}")
    return out_png


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print(" GEOCASCADE - TEMPORAL BATCH INDEX PROCESSOR")
    print(f" Period : {DATE_RANGE}")
    print(f" Max cloud: {MAX_CLOUD}%  |  Max scenes: {MAX_SCENES}")
    print("=" * 65)

    print("\n[1/4] Searching for cloud-free Sentinel-2 scenes...")
    try:
        items = search_stac()
    except Exception as e:
        print(f"\n  ERROR: {e}")
        return

    print(f"\n[2/4] Processing {len(items)} scenes...")
    all_stats = []
    for n, item in enumerate(items, 1):
        stats = process_scene(item, n, len(items))
        if stats:
            all_stats.append(stats)

    if not all_stats:
        print("\n  ERROR: No scenes processed successfully.")
        return

    ts_df = pd.DataFrame(all_stats)
    csv_path = os.path.join(OUT_DIR, "batch_time_series.csv")
    ts_df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\n[3/4] Time series CSV: {csv_path}")

    # Summary
    if "ndvi_mean" in ts_df.columns:
        ndvi_vals = ts_df["ndvi_mean"].dropna()
        if len(ndvi_vals):
            pk_row = ts_df.loc[ts_df["ndvi_mean"].idxmax()]
            lo_row = ts_df.loc[ts_df["ndvi_mean"].idxmin()]
            print(f"\n  NDVI range   : {ndvi_vals.min():+.3f} to {ndvi_vals.max():+.3f}")
            print(f"  Peak green   : {str(pk_row['date'])[:10]} (NDVI={pk_row['ndvi_mean']:+.3f})")
            print(f"  Minimum green: {str(lo_row['date'])[:10]} (NDVI={lo_row['ndvi_mean']:+.3f})")
    if "glacier_pct" in ts_df.columns:
        gp = ts_df["glacier_pct"].dropna()
        if len(gp):
            print(f"  Glacier cover: {gp.min():.1f}% to {gp.max():.1f}% of BBOX")

    print(f"\n[4/4] Building temporal figure...")
    plot_temporal(ts_df)

    n_tifs = len([f for f in os.listdir(OUT_DIR) if f.endswith(".tif")])
    print("\n" + "=" * 65)
    print(" BATCH PROCESSING COMPLETE")
    print("=" * 65)
    print(f"  Scenes processed: {len(all_stats)}")
    print(f"  TIFs saved      : {n_tifs} in {OUT_DIR}")
    print(f"  Time series CSV : {csv_path}")
    print(f"  Figure          : {os.path.join(PLOT_DIR, 'batch_temporal_analysis.png')}")
    print()
    print("  ArcGIS Pro: Load batch_time_series.csv -> Insert > Chart > Line")
    print("              (NDVI mean column shows the seasonal vegetation cycle)")
    print("  ENVI 5.6  : File > Open > select all ndvi_*.tif -> Time Series Analyst")
    print("=" * 65)


if __name__ == "__main__":
    main()
