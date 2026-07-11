"""
Chapter 1: 03c_chirps_spatial_precipitation.py

Academic Objective:
CHIRPS (Climate Hazards Group InfraRed Precipitation with Station data) is the
gold-standard for regional precipitation analysis in data-sparse regions like
Patagonia. It fuses satellite thermal infrared estimates with rain gauge records
to produce a 5.5km resolution daily/monthly dataset from 1981 to present.

This script:
  1. Downloads CHIRPS v2.0 monthly GeoTIFFs for the study region (no login)
  2. Clips to the Torres del Paine BBOX
  3. Computes the 30-year precipitation climatology (mean seasonal cycle)
  4. Detects anomaly years using Z-score
  5. Computes the Patagonian precipitation gradient (W-E transect)
  6. Cross-validates with ERA5 point estimates

Key insight: The Andes create one of the world's sharpest precipitation
gradients. The windward (Pacific) side receives >3000mm/yr while the
leeward (Argentine Steppe) side receives <300mm/yr over just 100km.

Dependencies:
mamba install -n geocascade_env -c conda-forge requests rasterio geopandas numpy pandas matplotlib pyproj -y
"""

import os
import requests
import gzip
import shutil
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import rasterio
from rasterio.windows import from_bounds
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pyproj import Transformer

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CHIRPS_DIR = os.path.join(BASE_DIR, "data", "raw", "real_data", "chirps_monthly")
OUT_DIR    = os.path.join(BASE_DIR, "data", "processed", "real_data")
os.makedirs(CHIRPS_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# Extended BBOX to capture the full W-E precipitation gradient
BBOX      = [-75.0, -52.0, -70.0, -49.5]  # wider for gradient analysis
INNER_BOX = [-73.5, -51.5, -72.5, -50.5]  # Torres del Paine focus area

CHIRPS_URL = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/tifs"


def bbox_to_indices(full_bbox, sub_bbox, shape):
    """
    Convert a lon/lat sub-bbox into row/col slices within an array that
    covers full_bbox, assuming standard raster orientation: row 0 = north
    (full_bbox maxlat), increasing row -> south; col 0 = west (full_bbox
    minlon), increasing col -> east.
    """
    minlon, minlat, maxlon, maxlat = full_bbox
    sub_minlon, sub_minlat, sub_maxlon, sub_maxlat = sub_bbox
    nrows, ncols = shape

    lon_res = (maxlon - minlon) / ncols
    lat_res = (maxlat - minlat) / nrows

    col_start = int((sub_minlon - minlon) / lon_res)
    col_end   = int((sub_maxlon - minlon) / lon_res)
    row_start = int((maxlat - sub_maxlat) / lat_res)
    row_end   = int((maxlat - sub_minlat) / lat_res)

    col_start, col_end = sorted((max(col_start, 0), min(col_end, ncols)))
    row_start, row_end = sorted((max(row_start, 0), min(row_end, nrows)))
    return slice(row_start, row_end), slice(col_start, col_end)


def download_chirps_for_year(year):
    """Download 12 monthly CHIRPS GeoTIFFs for one year."""
    downloaded = []
    for month in range(1, 13):
        fname     = f"chirps-v2.0.{year}.{month:02d}.tif.gz"
        tif_path  = os.path.join(CHIRPS_DIR, fname.replace(".gz", ""))
        if os.path.exists(tif_path):
            downloaded.append(tif_path)
            continue
        gz_path = os.path.join(CHIRPS_DIR, fname)
        url     = f"{CHIRPS_URL}/{fname}"
        try:
            r = requests.get(url, timeout=60, stream=True)
            r.raise_for_status()
            with open(gz_path, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            with gzip.open(gz_path, "rb") as gz_in:
                with open(tif_path, "wb") as t_out:
                    shutil.copyfileobj(gz_in, t_out)
            os.remove(gz_path)
            downloaded.append(tif_path)
            time.sleep(0.2)
        except Exception as e:
            print(f"       [WARNING] Failed {year}-{month:02d}: {e}")
    return downloaded


def read_chirps_bbox(tif_path, bbox):
    """Read a CHIRPS TIF clipped to BBOX. Returns (array, transform, nodata)."""
    with rasterio.open(tif_path) as src:
        window = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], src.transform)
        arr = src.read(1, window=window).astype("float32")
        transform = rasterio.windows.transform(window, src.transform)
        nodata = src.nodata if src.nodata else -9999
        arr = np.where(arr == nodata, np.nan, arr)
        arr = np.where(arr < 0, np.nan, arr)  # CHIRPS no-data can be -9999
    return arr, transform


def build_annual_stacks(years):
    """Build dict of annual precipitation rasters from CHIRPS."""
    print(f"\n[1/4] Building annual precipitation stacks ({years[0]}-{years[-1]})...")
    annual_maps = {}
    profile_ref = None

    for year in years:
        tifs = download_chirps_for_year(year)
        if not tifs:
            continue
        monthly_arrs = []
        for tif in sorted(tifs):
            arr, transform = read_chirps_bbox(tif, BBOX)[:2]
            monthly_arrs.append(arr)
        if len(monthly_arrs) == 12:
            annual_sum = np.nansum(np.stack(monthly_arrs, axis=0), axis=0)
            annual_maps[year] = annual_sum
            if profile_ref is None:
                with rasterio.open(tifs[0]) as src:
                    window = from_bounds(BBOX[0], BBOX[1], BBOX[2], BBOX[3], src.transform)
                    profile_ref = src.profile.copy()
                    profile_ref.update(
                        height=int(round(window.height)),
                        width=int(round(window.width)),
                        transform=transform,
                        count=1, dtype=rasterio.float32, nodata=-9999
                    )
                    _, transform = read_chirps_bbox(tifs[0], BBOX)
                    profile_ref["transform"] = transform
        print(f"       {year}: {len(monthly_arrs)} months processed")

    print(f"       Built {len(annual_maps)} annual maps.")
    return annual_maps, profile_ref


def compute_climatology(annual_maps):
    """Compute mean, std, and anomaly Z-scores."""
    print("\n[2/4] Computing 30-year precipitation climatology...")
    stack = np.stack(list(annual_maps.values()), axis=0)
    clim_mean = np.nanmean(stack, axis=0)
    clim_std  = np.nanstd(stack,  axis=0)

    anomalies = {}
    for year, arr in annual_maps.items():
        z = (arr - clim_mean) / (clim_std + 1e-6)
        anomalies[year] = z

    print(f"       Climatological mean range: {np.nanmin(clim_mean):.0f} — {np.nanmax(clim_mean):.0f} mm/yr")
    return clim_mean, clim_std, anomalies


def compute_west_east_transect(annual_maps):
    """Compute W-E precipitation transect at the study latitude."""
    print("\n[3/4] Computing Andes precipitation gradient (W-E transect)...")
    stack = np.stack(list(annual_maps.values()), axis=0)
    mean_map = np.nanmean(stack, axis=0)

    # Average across latitude band (Torres del Paine)
    nrows, ncols = mean_map.shape
    mid_row_start = nrows // 3
    mid_row_end   = 2 * nrows // 3
    transect = np.nanmean(mean_map[mid_row_start:mid_row_end, :], axis=0)

    lon_range = np.linspace(BBOX[0], BBOX[2], ncols)
    return lon_range, transect


def save_climatology_tif(clim_mean, profile_ref):
    """Save mean annual precipitation as GeoTIFF."""
    out_path = os.path.join(OUT_DIR, "chirps_mean_annual_precip.tif")
    data = np.where(np.isnan(clim_mean), -9999, clim_mean).astype("float32")
    with rasterio.open(out_path, "w", **profile_ref) as dst:
        dst.write(data, 1)
    print(f"       [SUCCESS] Climatology TIFF saved: {out_path}")
    return out_path


def plot_chirps_analysis(annual_maps, clim_mean, clim_std, anomalies, lon_range, transect):
    print("\n[4/4] Generating CHIRPS analysis plots...")

    years = sorted(annual_maps.keys())
    row_slice, col_slice = bbox_to_indices(BBOX, INNER_BOX, next(iter(annual_maps.values())).shape)
    point_series = [np.nanmean(annual_maps[y][row_slice, col_slice]) for y in years]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("CHIRPS v2.0 Precipitation Analysis — Patagonia (30-year)\n"
                 "Torres del Paine & Andes Gradient", fontsize=13, fontweight="bold")

    # Panel 1: Mean annual precipitation map
    ax = axes[0, 0]
    im = ax.imshow(clim_mean, cmap="YlGnBu",
                   vmin=np.nanpercentile(clim_mean, 5),
                   vmax=np.nanpercentile(clim_mean, 95),
                   aspect="auto",
                   extent=[BBOX[0], BBOX[2], BBOX[1], BBOX[3]])
    plt.colorbar(im, ax=ax, label="mm/year")
    ax.set_title("Mean Annual Precipitation (CHIRPS)", fontsize=10)
    ax.set_xlabel("Longitude (°W)")
    ax.set_ylabel("Latitude (°S)")
    ax.axvline(-73.0, color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_aspect("equal")

    # Panel 2: Interannual variability (std)
    ax = axes[0, 1]
    im = ax.imshow(clim_std, cmap="Oranges",
                   aspect="auto",
                   extent=[BBOX[0], BBOX[2], BBOX[1], BBOX[3]])
    plt.colorbar(im, ax=ax, label="mm/year (std dev)")
    ax.set_title("Interannual Variability (Std Dev)", fontsize=10)
    ax.set_xlabel("Longitude (°W)")
    ax.set_aspect("equal")

    # Panel 3: Most anomalous year
    if anomalies:
        ax = axes[0, 2]
        # Pick the year with highest absolute area-mean anomaly
        anom_means = {y: np.nanmean(np.abs(z)) for y, z in anomalies.items()}
        worst_year = max(anom_means, key=anom_means.get)
        z_map = anomalies[worst_year]
        norm  = mcolors.TwoSlopeNorm(vmin=-3, vcenter=0, vmax=3)
        im = ax.imshow(z_map, cmap="RdBu_r", norm=norm,
                       aspect="auto",
                       extent=[BBOX[0], BBOX[2], BBOX[1], BBOX[3]])
        plt.colorbar(im, ax=ax, label="Z-score")
        ax.set_title(f"Precipitation Anomaly — {worst_year} (Z-score)", fontsize=10)
        ax.set_xlabel("Longitude (°W)")
        ax.set_aspect("equal")

    # Panel 4: Time series (area-mean)
    ax = axes[1, 0]
    mean_val = np.nanmean(point_series)
    bar_cols  = ["#d62728" if p < mean_val * 0.85 else
                 "#1f77b4" if p > mean_val * 1.15 else "#7f7f7f"
                 for p in point_series]
    ax.bar(years, point_series, color=bar_cols, alpha=0.75)
    ax.axhline(mean_val, color="black", linestyle="--", linewidth=1.5,
               label=f"Mean: {mean_val:.0f} mm")
    ax.set_title("CHIRPS Area-Mean Annual Precipitation", fontsize=10)
    ax.set_ylabel("Precipitation (mm)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.4)

    # Panel 5: W-E precipitation gradient (Andes rain shadow)
    ax = axes[1, 1]
    ax.fill_between(lon_range, transect, alpha=0.4, color="#3498db")
    ax.plot(lon_range, transect, color="#2980b9", linewidth=2)
    ax.axvline(-73.2, color="green",  linestyle="--", label="Torres del Paine")
    ax.axvline(-72.5, color="orange", linestyle="--", label="Park Boundary (E)")
    ax.set_title("W-E Precipitation Transect (Andes Rain Shadow)", fontsize=10)
    ax.set_xlabel("Longitude (°W)")
    ax.set_ylabel("Annual Precipitation (mm)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.4)
    ax.invert_xaxis()
    ax.text(0.02, 0.95, "Pacific\n(Windward)", transform=ax.transAxes,
            fontsize=9, va="top", color="blue")
    ax.text(0.82, 0.95, "Argentina\n(Leeward)", transform=ax.transAxes,
            fontsize=9, va="top", color="brown")

    # Panel 6: Z-score anomaly time series (same INNER_BOX region as Panel 4,
    # so both "area-mean" panels actually describe the same area)
    ax = axes[1, 2]
    anom_years = [y for y in years if y in anomalies]
    anom_ts = [np.nanmean(anomalies[y][row_slice, col_slice]) for y in anom_years]
    bar_anom_cols = ["#d62728" if z < -1 else "#1f77b4" if z > 1 else "#7f7f7f"
                     for z in anom_ts]
    ax.bar(anom_years, anom_ts, color=bar_anom_cols, alpha=0.75)
    ax.axhline(0,  color="black", linewidth=1)
    ax.axhline(1,  color="#1f77b4", linestyle="--", linewidth=1, alpha=0.6, label="|Z|=1")
    ax.axhline(-1, color="#d62728", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_title("Torres del Paine Precipitation Z-Score Anomaly", fontsize=10)
    ax.set_ylabel("Z-score")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.4)

    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, "chirps_precipitation_analysis.png")
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"       [SUCCESS] CHIRPS analysis saved: {plot_path}")


def main():
    print("=" * 65)
    print(" GEOCASCADE — CHIRPS SPATIAL PRECIPITATION ANALYSIS")
    print("=" * 65)
    print("  Data source: CHIRPS v2.0 (UC Santa Barbara, ~5.5km)")
    print("  Coverage: Global, 1981-present, monthly")
    print("  No login required. Direct HTTP download.")
    print()

    YEARS = list(range(1993, 2024))
    annual_maps, profile_ref = build_annual_stacks(YEARS)

    if not annual_maps:
        print("[ERROR] No CHIRPS data downloaded. Check internet connection.")
        return

    clim_mean, clim_std, anomalies = compute_climatology(annual_maps)
    lon_range, transect = compute_west_east_transect(annual_maps)

    if profile_ref:
        save_climatology_tif(clim_mean, profile_ref)

    plot_chirps_analysis(annual_maps, clim_mean, clim_std, anomalies, lon_range, transect)

    print("\n[SUCCESS] CHIRPS analysis complete!")
    print("  Next: Run 23_real_data_convergence.py to combine with satellite data.")


if __name__ == "__main__":
    main()
