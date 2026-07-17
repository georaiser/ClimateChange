"""
Chapter 1: 05_chirps_precipitation.py
======================================
CHIRPS v2.0 Spatial Precipitation Analysis

Academic Objective:
  CHIRPS (Climate Hazards Group InfraRed Precipitation with Station data)
  combines satellite IR with rain-gauge observations at 0.05 deg (~5.5 km)
  resolution from 1981-present. This script processes the 384 monthly TIFs
  already on disk to produce a 30-year climatology, anomaly Z-scores, and a
  West-East precipitation transect showing the Andes rain shadow effect.

NOTE on data:
  The 384 CHIRPS TIFs should already be on disk in:
    data/raw/real_data/chirps_monthly/
  If any are missing, the script downloads them automatically from UCSB.
  Global TIFs are NEVER loaded in full -- only the study BBOX is windowed.

Outputs:
  data/processed/real_data/chirps_mean_annual_precip.tif  (GeoTIFF, ArcGIS/ENVI ready)
  data/processed/real_data/chirps_time_series.csv
  data/processed/real_data/chirps_annual_anomalies.csv
  data/processed/real_data/chirps_precipitation_analysis.png

Run:
  conda activate geocascade_env
  python Chapter_01/05_chirps_precipitation.py

ArcGIS Pro: Add chirps_mean_annual_precip.tif as raster layer.
            Use Symbology > Classify > 5 classes, Yellow-to-Blue palette.
ENVI 5.6:   File > Open > chirps_mean_annual_precip.tif
            Use Band Math for thresholding or Scatter Plot for transect.

Dependencies: rasterio, numpy, pandas, matplotlib, requests
"""

import sys
import os
import gzip
import shutil
import warnings
import glob
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import requests
import rasterio
from rasterio.windows import from_bounds
from rasterio.transform import from_bounds as transform_from_bounds

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
CHIRPS_DIR    = os.path.join(BASE_DIR, "data", "raw",  "real_data", "chirps_monthly")
PROC_DIR      = os.path.join(BASE_DIR, "data", "processed", "real_data")
os.makedirs(CHIRPS_DIR, exist_ok=True)
os.makedirs(PROC_DIR,   exist_ok=True)

# Study BBOX [min_lon, min_lat, max_lon, max_lat]
# NOTE: CHIRPS v2.0 global coverage is 50°S to 50°N only.
# Torres del Paine sits at ~51°S which is just outside CHIRPS.
# We use a 3°×2° window centred on the Patagonian precipitation
# gradient (-73.5 to -70.5°W, -50° to -48°S) to capture the full
# Andes rain shadow signal while staying within CHIRPS coverage.
# The narrow 1°×1° strip at -50° had corrupted fill-values in
# CHIRPS tiles for 2013-2016, producing spuriously low annual sums.
BBOX     = [-73.5, -50.0, -70.5, -48.0]   # 3° lon x 2° lat, within CHIRPS coverage
YEARS    = list(range(2000, 2025))
MONTHS   = list(range(1, 13))

CHIRPS_URL = ("https://data.chc.ucsb.edu/products/CHIRPS-2.0/"
              "global_monthly/tifs/chirps-v2.0.{year}.{month:02d}.tif.gz")

DARK_BG = "#0d1117"
DARK_AX = "#161b22"
C_TEXT  = "#e6edf3"
C_GREY  = "#8b949e"
C_BLUE  = "#3498db"
C_RED   = "#e74c3c"
C_GREEN = "#2ecc71"
C_GOLD  = "#f39c12"
C_CYAN  = "#00bcd4"


# ---------------------------------------------------------------------------
# 1. Download missing CHIRPS files
# ---------------------------------------------------------------------------
def download_chirps(year, month):
    fname  = f"chirps-v2.0.{year}.{month:02d}.tif"
    fpath  = os.path.join(CHIRPS_DIR, fname)
    if os.path.exists(fpath):
        return fpath

    url  = CHIRPS_URL.format(year=year, month=month)
    gz   = fpath + ".gz"
    print(f"  Downloading {fname} ...", end="", flush=True)
    try:
        r = requests.get(url, stream=True, timeout=120)
        r.raise_for_status()
        with open(gz, "wb") as f:
            for chunk in r.iter_content(65536):
                f.write(chunk)
        with gzip.open(gz, "rb") as fin, open(fpath, "wb") as fout:
            shutil.copyfileobj(fin, fout)
        os.remove(gz)
        kb = os.path.getsize(fpath) / 1024
        print(f" OK ({kb:.0f} KB)")
        return fpath
    except Exception as e:
        print(f" FAILED: {e}")
        if os.path.exists(gz):
            os.remove(gz)
        return None


# ---------------------------------------------------------------------------
# 2. Windowed read of BBOX from global CHIRPS TIF
# ---------------------------------------------------------------------------
def read_bbox(fpath):
    """
    Read only the BBOX pixels from the global 0.05-deg CHIRPS grid.
    Avoids loading the full ~200 MB global array into memory.

    Masking layers applied (in order):
      1. Explicit nodata (-9999 or value from metadata)
      2. Any negative value
      3. Values <= MIN_PIXEL_MM -- CHIRPS fill / ocean / ice pixels in some
         versions store tiny positive values (~1.44) instead of -9999.
    """
    with rasterio.open(fpath) as src:
        win = from_bounds(BBOX[0], BBOX[1], BBOX[2], BBOX[3], src.transform)
        win = win.round_offsets().round_lengths()
        data = src.read(1, window=win).astype("float32")
        out_transform = rasterio.windows.transform(win, src.transform)
        nodata = src.nodata if src.nodata is not None else -9999.0
        # Layer 1 + 2: explicit nodata and negatives
        data = np.where((data == nodata) | (data < 0), np.nan, data)
        # Layer 3: pixel floor -- remove suspiciously low fill values
        data = np.where(data <= MIN_PIXEL_MM, np.nan, data)
        profile = src.profile.copy()
        profile.update({
            "height":    win.height,
            "width":     win.width,
            "transform": out_transform,
            "count":     1,
            "dtype":     "float32",
            "nodata":    -9999,
            "compress":  "lzw",
        })
        return data, profile, out_transform


# ---------------------------------------------------------------------------
# 3. Build complete time series
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Quality thresholds
# ---------------------------------------------------------------------------
MIN_MONTHS_COMPLETE = 10    # years with fewer months skipped from grid/stats
MIN_PIXEL_MM        = 1.0   # pixel floor: CHIRPS fill/ocean pixels can be ~1.44mm;
                            #   values at or below this are treated as nodata
MIN_ANNUAL_MEAN_MM  = 150   # year-level outlier guard: Patagonia annual spatial
                            #   mean never legitimately falls below ~150mm/year
                            #   (even Argentine steppe gets >200mm). Years below
                            #   this are flagged as 'suspect' and excluded from stats.

def build_time_series():
    print("[2/5] Building CHIRPS time series...")
    rows         = []
    annual_grids = {}   # year -> annual total array (complete years only)
    yr_months    = {}   # year -> number of months successfully read
    profile_ref  = None

    for yr in YEARS:
        yr_sum   = None
        yr_count = None
        mo_ok    = 0
        for mo in MONTHS:
            fpath = download_chirps(yr, mo)
            if fpath is None:
                continue
            try:
                data, profile, _ = read_bbox(fpath)
                mean_mm = float(np.nanmean(data))
                rows.append({"year": yr, "month": mo, "precip_mm": round(mean_mm, 2)})
                if profile_ref is None:
                    profile_ref = profile
                if yr_sum is None:
                    yr_sum   = np.where(np.isnan(data), 0.0, data)
                    yr_count = np.where(np.isnan(data), 0, 1)
                else:
                    yr_sum   += np.where(np.isnan(data), 0.0, data)
                    yr_count += np.where(np.isnan(data), 0, 1)
                mo_ok += 1
            except Exception as e:
                print(f"  [WARN] {yr}-{mo:02d}: {e}")

        yr_months[yr] = mo_ok

        # Year-level quality check
        if yr_sum is not None and mo_ok > 0:
            yr_mean_annual = float(np.nansum(yr_sum))   # rough spatial sum
            yr_spatial_mean = float(np.nanmean(yr_sum)) # mean mm across pixels
        else:
            yr_spatial_mean = 0.0

        is_complete = mo_ok >= MIN_MONTHS_COMPLETE
        is_quality  = is_complete and (yr_spatial_mean >= MIN_ANNUAL_MEAN_MM)

        if not is_complete:
            flag = f"SKIP — only {mo_ok}/12 months"
        elif not is_quality:
            flag = f"SUSPECT — spatial mean {yr_spatial_mean:.0f} mm < {MIN_ANNUAL_MEAN_MM} mm threshold"
        else:
            flag = "OK"
        print(f"  {yr}: {mo_ok:2d} months  [{flag}]")

        # Only add to spatial grid stack if year passes both checks
        if yr_sum is not None and is_quality:
            annual_grids[yr] = np.where(yr_count > 0, yr_sum, np.nan)

    ts_df = pd.DataFrame(rows)
    print(f"  Time series: {len(ts_df)} monthly records ({ts_df['year'].min()} - {ts_df['year'].max()})")
    incomplete = [yr for yr, n in yr_months.items() if n < MIN_MONTHS_COMPLETE]
    if incomplete:
        print(f"  [NOTE] Skipped (incomplete months): {incomplete}")
    return ts_df, annual_grids, yr_months, profile_ref


# ---------------------------------------------------------------------------
# 4. Compute climatology and anomalies
# ---------------------------------------------------------------------------
def compute_climatology_anomalies(ts_df, yr_months, annual_grids):
    """Compute annual sums and anomaly Z-scores.

    Only years present in annual_grids (i.e. passed both completeness
    and quality checks) contribute to the Z-score baseline.  All other
    years are included in the CSV but marked as 'suspect' or 'incomplete'.
    """
    annual_df = (ts_df.groupby("year")["precip_mm"]
                 .agg(annual_mm="sum", months_available="count")
                 .reset_index())

    annual_df["is_complete"] = annual_df["months_available"] >= MIN_MONTHS_COMPLETE
    annual_df["in_grid"]     = annual_df["year"].isin(annual_grids.keys())

    # Z-scores computed only from years that passed all quality checks
    good = annual_df[annual_df["in_grid"]]
    mean_val = good["annual_mm"].mean()
    std_val  = good["annual_mm"].std()
    annual_df["zscore"] = (annual_df["annual_mm"] - mean_val) / std_val
    annual_df.loc[~annual_df["in_grid"], "zscore"] = np.nan

    def classify(row):
        if not row["is_complete"]:  return "incomplete"
        if not row["in_grid"]:      return "suspect"
        z = row["zscore"]
        if pd.isna(z):              return "suspect"
        if z < -1.0:                return "drought"
        if z >  1.0:                return "wet"
        return "normal"

    annual_df["regime"] = annual_df.apply(classify, axis=1)

    monthly_clim = (ts_df.groupby("month")["precip_mm"]
                    .mean().reset_index()
                    .rename(columns={"precip_mm": "clim_mm"}))

    return annual_df, monthly_clim


# ---------------------------------------------------------------------------
# 5. Save climatology GeoTIFF
# ---------------------------------------------------------------------------
def save_climatology_tif(annual_grids, profile_ref):
    if not annual_grids or profile_ref is None:
        print("  [SKIP] No grid data to save.")
        return None

    stack  = np.stack(list(annual_grids.values()), axis=0)
    clim   = np.nanmean(stack, axis=0)
    nodata = -9999.0
    clim_nd = np.where(np.isnan(clim), nodata, clim).astype("float32")

    out_tif = os.path.join(PROC_DIR, "chirps_mean_annual_precip.tif")
    with rasterio.open(out_tif, "w", **profile_ref) as dst:
        dst.write(clim_nd, 1)
        dst.update_tags(
            description="CHIRPS v2.0 mean annual precipitation (mm/year)",
            source="UCSB CHG - chirps.ucsb.edu",
            period=f"{min(annual_grids.keys())}-{max(annual_grids.keys())}",
            bbox=str(BBOX),
        )
    sz = os.path.getsize(out_tif) / 1e6
    print(f"  [OK] Climatology TIF: {out_tif} ({sz:.1f} MB)")
    valid_clim = clim[np.isfinite(clim)]
    if valid_clim.size > 0:
        print(f"       Range: {np.nanmin(valid_clim):.0f} - {np.nanmax(valid_clim):.0f} mm/year")
    else:
        print("       [WARN] Climatology array is all-NaN. Check BBOX vs CHIRPS coverage.")
    return out_tif, clim


# ---------------------------------------------------------------------------
# 6. Build West-East transect (Andes rain shadow)
# ---------------------------------------------------------------------------
def build_we_transect(annual_grids, profile_ref):
    """Sample mean annual precip along a fixed latitude line."""
    if not annual_grids or profile_ref is None:
        return None, None
    stack = np.stack(list(annual_grids.values()), axis=0)
    clim  = np.nanmean(stack, axis=0)

    # Use the middle row of the clipped grid as the transect latitude
    mid_row = clim.shape[0] // 2
    transect_vals = clim[mid_row, :]

    # Compute longitude array for that row
    transform = profile_ref["transform"]
    n_cols    = clim.shape[1]
    lons      = [transform.c + (col + 0.5) * transform.a for col in range(n_cols)]

    return lons, transect_vals


# ---------------------------------------------------------------------------
# 7. 6-panel analysis figure
# ---------------------------------------------------------------------------
def plot_analysis(ts_df, annual_df, monthly_clim, lons_t, transect_t,
                  annual_grids, profile_ref):
    print("  Building 6-panel analysis figure...")
    fig = plt.figure(figsize=(22, 16), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.32,
                            top=0.93, bottom=0.07, left=0.06, right=0.97)

    def style_ax(ax, title="", xlabel="", ylabel=""):
        ax.set_facecolor(DARK_AX)
        for sp in ax.spines.values():
            sp.set_color("#30363d")
        ax.tick_params(colors=C_TEXT, labelsize=9)
        ax.xaxis.label.set_color(C_TEXT)
        ax.yaxis.label.set_color(C_TEXT)
        ax.grid(alpha=0.18, color="#30363d")
        if title:  ax.set_title(title, color=C_TEXT, fontsize=9, fontweight="bold", pad=6)
        if xlabel: ax.set_xlabel(xlabel, color=C_TEXT, fontsize=9)
        if ylabel: ax.set_ylabel(ylabel, color=C_TEXT, fontsize=9)

    fig.text(0.5, 0.97, "GeoCascade - CHIRPS v2.0 Precipitation Analysis (2000-2024)",
             ha="center", color=C_TEXT, fontsize=13, fontweight="bold")
    fig.text(0.5, 0.945, "Torres del Paine, Patagonia | 0.05 deg resolution | UCSB CHG",
             ha="center", color=C_GREY, fontsize=9)

    # Panel 1: Annual precipitation time series
    ax1 = fig.add_subplot(gs[0, 0])
    colors_p = [C_RED if r == "drought" else C_BLUE if r == "wet" else C_GREY
                for r in annual_df["regime"]]
    ax1.bar(annual_df["year"], annual_df["annual_mm"], color=colors_p, alpha=0.8, width=0.8)
    ax1.axhline(annual_df["annual_mm"].mean(), color=C_GOLD, lw=1.5, ls="--",
                label=f"Mean: {annual_df['annual_mm'].mean():.0f} mm")
    for _, row in annual_df[annual_df["regime"] != "normal"].iterrows():
        ax1.text(row["year"], row["annual_mm"] + 10, str(int(row["year"])),
                 ha="center", fontsize=6, color=C_TEXT)
    ax1.legend(fontsize=8, facecolor=DARK_BG, labelcolor=C_TEXT)
    style_ax(ax1, "Annual Precipitation + Extreme Years", "Year", "mm/year")

    # Panel 2: Seasonal climatology
    ax2 = fig.add_subplot(gs[0, 1])
    mon_labels = ["J","F","M","A","M","J","J","A","S","O","N","D"]
    ax2.bar(monthly_clim["month"], monthly_clim["clim_mm"],
            color=C_BLUE, alpha=0.75, width=0.7)
    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels(mon_labels, color=C_TEXT, fontsize=9)
    style_ax(ax2, "Seasonal Climatology (mean 2000-2024)", "Month", "mm/month")

    # Panel 3: Z-score anomaly chart
    ax3 = fig.add_subplot(gs[0, 2])
    bar_c2 = [C_RED if z < -0.5 else C_BLUE if z > 0.5 else C_GREY
              for z in annual_df["zscore"]]
    ax3.bar(annual_df["year"], annual_df["zscore"], color=bar_c2, alpha=0.8, width=0.8)
    ax3.axhline(0,    color=C_GOLD, lw=1.2, ls="--")
    ax3.axhline(-1.0, color=C_RED,  lw=0.8, ls=":", alpha=0.6)
    ax3.axhline(+1.0, color=C_BLUE, lw=0.8, ls=":", alpha=0.6)
    style_ax(ax3, "Precipitation Anomaly Z-Score", "Year", "Standard deviations")

    # Panel 4: Sen's slope trend
    ax4 = fig.add_subplot(gs[1, 0])
    yrs_ann  = annual_df["year"].values.astype(float)
    prec_ann = annual_df["annual_mm"].values
    z4 = np.polyfit(yrs_ann, prec_ann, 1)
    ax4.fill_between(annual_df["year"], prec_ann, prec_ann.mean(),
                     alpha=0.25, color=C_BLUE)
    ax4.plot(annual_df["year"], prec_ann, "o-", color=C_CYAN, lw=1.8, ms=4)
    ax4.plot(annual_df["year"], np.poly1d(z4)(yrs_ann), "--", color=C_GOLD, lw=2,
             label=f"OLS trend: {z4[0]:+.1f} mm/decade")
    ax4.legend(fontsize=8, facecolor=DARK_BG, labelcolor=C_TEXT)
    style_ax(ax4, "Precipitation Trend", "Year", "mm/year")

    # Panel 5: W-E transect (Andes rain shadow)
    ax5 = fig.add_subplot(gs[1, 1])
    if lons_t is not None and transect_t is not None:
        valid = ~np.isnan(transect_t)
        ax5.fill_between(np.array(lons_t)[valid], transect_t[valid],
                         alpha=0.35, color=C_BLUE)
        ax5.plot(np.array(lons_t)[valid], transect_t[valid],
                 color=C_CYAN, lw=2)
        ax5.axvline(-73.0, color=C_GOLD, lw=1, ls="--", alpha=0.7, label="Andes divide")
        ax5.legend(fontsize=8, facecolor=DARK_BG, labelcolor=C_TEXT)
    style_ax(ax5, "W-E Transect: Andes Rain Shadow (~51 deg S)",
             "Longitude (deg)", "Mean Annual Precip (mm)")

    # Panel 6: Variability map (std dev over years)
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_facecolor(DARK_AX)
    if annual_grids and len(annual_grids) > 1:
        stack = np.stack(list(annual_grids.values()), axis=0)
        std_map = np.nanstd(stack, axis=0)
        transform = profile_ref["transform"]
        h, w = std_map.shape
        extent = [transform.c, transform.c + w * transform.a,
                  transform.f + h * transform.e, transform.f]
        im6 = ax6.imshow(std_map, extent=extent, cmap="YlOrRd",
                         aspect="auto", origin="upper")
        cb6 = plt.colorbar(im6, ax=ax6, fraction=0.04, pad=0.02)
        cb6.set_label("Std Dev (mm/year)", color=C_TEXT, fontsize=8)
        cb6.ax.tick_params(colors=C_TEXT, labelsize=7)
        ax6.set_xlabel("Longitude", color=C_TEXT, fontsize=9)
        ax6.set_ylabel("Latitude",  color=C_TEXT, fontsize=9)
        for sp in ax6.spines.values():
            sp.set_color("#30363d")
        ax6.tick_params(colors=C_TEXT, labelsize=8)
    ax6.set_title("Interannual Variability (Std Dev)", color=C_TEXT,
                  fontsize=9, fontweight="bold", pad=6)

    out_png = os.path.join(PROC_DIR, "chirps_precipitation_analysis.png")
    fig.savefig(out_png, dpi=180, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  [OK] 6-panel figure: {out_png}")
    return out_png


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print(" GEOCASCADE - CHIRPS PRECIPITATION SPATIAL ANALYSIS")
    print(f" BBOX: {BBOX}  |  {len(YEARS)} years x 12 months")
    print("=" * 65)

    # Inventory existing files
    existing = glob.glob(os.path.join(CHIRPS_DIR, "chirps-v2.0.*.tif"))
    print(f"\n[1/5] Found {len(existing)} CHIRPS TIFs on disk.")
    total_needed = len(YEARS) * 12
    if len(existing) < total_needed:
        print(f"  {total_needed - len(existing)} files missing -- will download from UCSB.")
    else:
        print(f"  All {total_needed} files present -- no downloads needed.")

    ts_df, annual_grids, yr_months, profile_ref = build_time_series()

    if ts_df.empty:
        print("\n  ERROR: No CHIRPS data available. Check network connection.")
        return

    print("\n[3/5] Computing climatology and anomalies...")
    annual_df, monthly_clim = compute_climatology_anomalies(ts_df, yr_months, annual_grids)

    # Stats only from years that passed all quality checks
    good_df  = annual_df[annual_df["in_grid"]]
    if good_df.empty:
        print("  [ERROR] No quality years available for statistics.")
        return
    dry_row  = good_df.loc[good_df["annual_mm"].idxmin()]
    wet_row  = good_df.loc[good_df["annual_mm"].idxmax()]
    suspect  = annual_df[annual_df["regime"] == "suspect"]["year"].tolist()
    incompl  = annual_df[annual_df["regime"] == "incomplete"]["year"].tolist()
    print(f"  Mean annual precip : {good_df['annual_mm'].mean():.0f} mm/year  ({len(good_df)} quality years)")
    print(f"  Driest year        : {int(dry_row['year'])} ({dry_row['annual_mm']:.0f} mm)")
    print(f"  Wettest year       : {int(wet_row['year'])} ({wet_row['annual_mm']:.0f} mm)")
    print(f"  Variability (CV)   : {good_df['annual_mm'].std()/good_df['annual_mm'].mean()*100:.1f}%")
    if suspect:  print(f"  [EXCL] Suspect years (low-value artifact): {suspect}")
    if incompl:  print(f"  [EXCL] Incomplete years (< {MIN_MONTHS_COMPLETE} months): {incompl}")

    print("\n[4/5] Saving outputs...")
    # Time series CSVs
    ts_path  = os.path.join(PROC_DIR, "chirps_time_series.csv")
    ann_path = os.path.join(PROC_DIR, "chirps_annual_anomalies.csv")
    ts_df.to_csv(ts_path,  index=False, encoding="utf-8")
    annual_df.to_csv(ann_path, index=False, encoding="utf-8")
    print(f"  [OK] Monthly time series : {ts_path}")
    print(f"  [OK] Annual anomalies    : {ann_path}")

    # GeoTIFF climatology
    result = save_climatology_tif(annual_grids, profile_ref)
    if result:
        out_tif, clim_arr = result
    else:
        out_tif, clim_arr = None, None

    # West-East transect
    lons_t, transect_t = build_we_transect(annual_grids, profile_ref)

    print("\n[5/5] Building figures...")
    plot_analysis(ts_df, annual_df, monthly_clim, lons_t, transect_t,
                  annual_grids, profile_ref)

    print("\n" + "=" * 65)
    print(" CHIRPS ANALYSIS COMPLETE")
    print("=" * 65)
    if out_tif:
        print(f"  GeoTIFF : {out_tif}")
    print(f"  CSVs    : {PROC_DIR}")
    print()
    print("  ArcGIS Pro: Add chirps_mean_annual_precip.tif as raster.")
    print("              Symbology > Classified > Yellow-Blue, 5 classes.")
    print("  ENVI 5.6  : File > Open > chirps_mean_annual_precip.tif")
    print("              Band Math for custom thresholding.")
    print("=" * 65)


if __name__ == "__main__":
    main()
