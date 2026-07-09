"""
Chapter 2: 08_automated_index_batcher.py

Academic Objective:
In commercial GUI software like ArcGIS Pro, you would use ModelBuilder to create
a visual loop to process multiple images. In Python we use a for-loop.

This UPGRADED script processes an entire year of cloud-free Sentinel-2 scenes
and computes THREE indices per scene (NDVI, NDSI, NDWI) — giving a complete
temporal stack usable for change detection.

Improvements over original:
  - safe_ratio() replaces hard-zero divide (water/shadow pixels preserved as NaN)
  - nodata=-9999 (not 0 or np.nan) for ArcGIS/ENVI compatibility
  - int(round()) window dimensions
  - NDSI and NDWI added alongside NDVI
  - B11 (SWIR, 20m) uses its own independent window and transformer
  - Summary statistics table printed at end
  - STAC empty guard
  - Cloud sort (ascending) so cleanest scenes processed first

Dependencies:
mamba install -n geocascade_env -c conda-forge pystac-client planetary-computer rasterio pyproj numpy pandas -y
"""

import os
import rasterio
from rasterio.windows import from_bounds
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pystac_client import Client
import planetary_computer as pc
from pyproj import Transformer

# ==========================================
# 1. Configuration
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(BASE_DIR, "data", "processed", "batch_indices")
os.makedirs(OUT_DIR, exist_ok=True)

BBOX       = [-73.30, -51.10, -72.90, -50.80]
DATE_RANGE = "2023-01-01/2023-12-31"


def safe_ratio(num, den, eps=1e-6):
    """NaN-safe band ratio. Returns NaN where denominator is near-zero (water/shadow)."""
    return np.where(np.abs(den) < eps, np.nan, num / den)


def calculate_ndvi(nir, red):   return safe_ratio(nir - red, nir + red)
def calculate_ndsi(green, swir): return safe_ratio(green - swir, green + swir)
def calculate_ndwi(green, nir):  return safe_ratio(green - nir,  green + nir)

# ==========================================
# 2. Automated Batch Processing Loop
# ==========================================
def run_batch_processor():
    print("\n[INFO] Connecting to Planetary Computer STAC API...")
    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace
    )

    print(f"[INFO] Searching cloud-free (<15%) Sentinel-2 scenes in {DATE_RANGE}...")
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=BBOX,
        datetime=DATE_RANGE,
        query={"eo:cloud_cover": {"lt": 15}}
    )
    items = list(search.items())
    if not items:
        print("       [WARNING] No Sentinel-2 scenes found. Try relaxing cloud cover or date range.")
        return []

    # Sort by cloud cover (cleanest first)
    items = sorted(items, key=lambda i: i.properties["eo:cloud_cover"])
    print(f"       [SUCCESS] Found {len(items)} scenes. Processing NDVI + NDSI + NDWI per scene.")

    summary_rows = []

    # ── MODELBUILDER-equivalent for-loop ──────────────────────────────────────
    for i, item in enumerate(items):
        date_str   = item.datetime.strftime("%Y-%m-%d")
        cloud_pct  = item.properties.get("eo:cloud_cover", "?")
        print(f"\n  [{i+1:02d}/{len(items)}] {date_str}  Cloud: {cloud_pct:.1f}%")

        # Skip if all 3 indices already processed
        ndvi_path  = os.path.join(OUT_DIR, f"NDVI_{date_str}.tif")
        ndsi_path  = os.path.join(OUT_DIR, f"NDSI_{date_str}.tif")
        ndwi_path  = os.path.join(OUT_DIR, f"NDWI_{date_str}.tif")
        if all(os.path.exists(p) for p in [ndvi_path, ndsi_path, ndwi_path]):
            print("         [SKIP] All indices already exist.")
            summary_rows.append({"date": date_str, "cloud": cloud_pct, "status": "skipped"})
            continue

        try:
            transformer = Transformer.from_crs("EPSG:4326", "EPSG:32719", always_xy=True)

            # ── Read 10m bands (B03 Green, B04 Red, B08 NIR) ──
            with rasterio.open(item.assets["B04"].href) as src:
                t10 = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                minx, miny = t10.transform(BBOX[0], BBOX[1])
                maxx, maxy = t10.transform(BBOX[2], BBOX[3])
                win10      = from_bounds(minx, miny, maxx, maxy, src.transform)
                target_h   = int(round(win10.height))
                target_w   = int(round(win10.width))
                win_transform = rasterio.windows.transform(win10, src.transform)
                profile    = src.profile.copy()
                profile.update(
                    driver="GTiff", count=1, dtype=rasterio.float32, nodata=-9999,
                    height=target_h, width=target_w,
                    transform=win_transform, compress="lzw"
                )
                red = src.read(1, window=win10).astype("float32") / 10000.0

            with rasterio.open(item.assets["B03"].href) as src:
                green = src.read(1, window=win10).astype("float32") / 10000.0

            with rasterio.open(item.assets["B08"].href) as src:
                nir = src.read(1, window=win10).astype("float32") / 10000.0

            # ── B11 SWIR1 (20m) — MUST use its own independent window ──
            with rasterio.open(item.assets["B11"].href) as src_b11:
                t20        = Transformer.from_crs("EPSG:4326", src_b11.crs, always_xy=True)
                mn11, my11 = t20.transform(BBOX[0], BBOX[1])
                mx11, my11b = t20.transform(BBOX[2], BBOX[3])
                win20      = from_bounds(mn11, my11, mx11, my11b, src_b11.transform)
                swir1 = src_b11.read(
                    1, window=win20,
                    out_shape=(target_h, target_w),
                    resampling=rasterio.enums.Resampling.bilinear
                ).astype("float32") / 10000.0

            # ── Compute indices ──
            ndvi = calculate_ndvi(nir, red)
            ndsi = calculate_ndsi(green, swir1)
            ndwi = calculate_ndwi(green, nir)

            # ── Save each index as nodata=-9999 GeoTIFF ──
            def save_idx(arr, path):
                data = np.where(np.isnan(arr), -9999, arr).astype("float32")
                with rasterio.open(path, "w", **profile) as dst:
                    dst.write(data, 1)

            save_idx(ndvi, ndvi_path)
            save_idx(ndsi, ndsi_path)
            save_idx(ndwi, ndwi_path)

            valid = ndvi[np.isfinite(ndvi)]
            row_stats = {
                "date":        date_str,
                "cloud":       cloud_pct,
                "ndvi_mean":   round(float(np.nanmean(valid)), 3) if valid.size else None,
                "ndsi_mean":   round(float(np.nanmean(ndsi[np.isfinite(ndsi)])), 3),
                "ndwi_mean":   round(float(np.nanmean(ndwi[np.isfinite(ndwi)])), 3),
                "status":      "ok"
            }
            summary_rows.append(row_stats)
            print(f"         NDVI={row_stats['ndvi_mean']:+.3f}  "
                  f"NDSI={row_stats['ndsi_mean']:+.3f}  "
                  f"NDWI={row_stats['ndwi_mean']:+.3f}")

        except Exception as e:
            print(f"         [ERROR] {e}")
            summary_rows.append({"date": date_str, "cloud": cloud_pct, "status": f"error: {e}"})

    return summary_rows

def plot_temporal_series(df):
    """Plot NDVI/NDSI/NDWI time series for all successfully processed scenes."""
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        return
    ok["date"] = pd.to_datetime(ok["date"])
    ok = ok.sort_values("date")

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle("Sentinel-2 Batch Processing — Annual Index Time Series\n"
                 "Torres del Paine 2023 (all cloud-free scenes)", fontsize=12, fontweight="bold")

    configs = [
        ("ndvi_mean", "RdYlGn",  (-0.1, 0.7),  "NDVI — Vegetation Health",  "#2ecc71"),
        ("ndsi_mean", "cool",    (-0.2, 0.8),  "NDSI — Snow & Ice Extent",  "#3498db"),
        ("ndwi_mean", "Blues",   (-0.5, 0.5),  "NDWI — Open Water Bodies",  "#1f77b4"),
    ]
    for ax, (col, cmap, ylim, title, color) in zip(axes, configs):
        vals = ok[col].astype(float)
        ax.plot(ok["date"], vals, "o-", color=color, linewidth=2, markersize=5)
        ax.fill_between(ok["date"], vals, alpha=0.15, color=color)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_ylim(ylim)
        ax.set_ylabel(col.replace("_mean", "").upper())
        ax.set_title(title, fontsize=9, loc="left")
        ax.grid(axis="y", alpha=0.35)
    axes[-1].set_xlabel("Date")

    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, "batch_index_timeseries.png")
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n       [SUCCESS] Time-series plot: {plot_path}")


def main():
    print("=======================================================")
    print(" GEOCASCADE PIPELINE - AUTOMATED BATCH PROCESSOR       ")
    print(" Computing NDVI + NDSI + NDWI for every 2023 scene     ")
    print("=======================================================")
    summary_rows = run_batch_processor()

    if not summary_rows:
        print("\n[WARNING] No scenes processed.")
        return

    df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(OUT_DIR, "batch_summary.csv")
    df.to_csv(csv_path, index=False)

    ok  = df[df["status"] == "ok"]
    err = df[df["status"].str.startswith("error", na=False)]
    print(f"\n{'='*60}")
    print(f" BATCH SUMMARY: {len(ok)} OK  |  {len(df[df['status']=='skipped'])} skipped  |  {len(err)} errors")
    print(f"{'='*60}")
    if not ok.empty:
        print(f"  NDVI range: {ok['ndvi_mean'].min():+.3f} to {ok['ndvi_mean'].max():+.3f}  "
              f"(mean={ok['ndvi_mean'].mean():+.3f})")
        print(f"  NDSI range: {ok['ndsi_mean'].min():+.3f} to {ok['ndsi_mean'].max():+.3f}")
        print(f"  NDWI range: {ok['ndwi_mean'].min():+.3f} to {ok['ndwi_mean'].max():+.3f}")
    print(f"  Summary CSV: {csv_path}")

    plot_temporal_series(df)
    print("\n[SUCCESS] Batch Processing Complete!")

if __name__ == "__main__":
    main()
