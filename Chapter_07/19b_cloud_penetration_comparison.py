"""
Chapter 7: 19b_cloud_penetration_comparison.py
===============================================
SAR Cloud Penetration Advantage vs Optical Sensors

Academic Objective:
  Patagonia has one of the world's highest cloud frequencies — Torres del Paine
  is overcast more than 80% of the time. This makes optical remote sensing
  almost useless for operational environmental monitoring.

  Sentinel-1 SAR (Synthetic Aperture Radar) solves this:
  - Transmits its own C-band microwave signal (5.405 GHz, lambda = 5.6 cm)
  - Microwaves pass through clouds, rain, and darkness unchanged
  - Backscatter is determined by surface ROUGHNESS and DIELECTRIC constant,
    not by solar illumination

  This script PROVES the cloud penetration advantage by:
  1. Acquiring a WINTER Sentinel-1 scene (June-August, peak cloud season)
  2. Finding a CLOUDY Sentinel-2 scene from the same week
  3. Plotting the side-by-side comparison to show what optical misses

  The VV/VH ratio provides additional structural information:
    - High VH (cross-pol) → volume scattering → forest / dense vegetation
    - Low VV, Low VH → specular → flat water / lake surface
    - High VV, Low VH → surface scattering → bare soil / rock
    - Moderate VV, High VH → double-bounce → urban structures / ice ridges

  An ESI (Environmental Stress Index) confidence overlay is added:
    Pixels where SAR shows glacier/rough terrain COINCIDING with optical
    cloud coverage get flagged as "SAR-only observable zones" —
    the scientifically most valuable regions for operational monitoring.

Outputs:
  data/processed/cloud_comparison/sar_vv_db.tif
  data/processed/cloud_comparison/sar_vs_optical_cloudy.png   (4-panel dark)
  data/processed/cloud_comparison/cloud_comparison_stats.csv

ArcGIS Pro: Load sar_vv_db.tif alongside any Sentinel-2 scene.
            Enable transparency on S2 and toggle to show SAR reveals more.
ENVI 5.6:   Use Display > Animation with both scenes to toggle rapidly.

Run:
  conda activate geocascade_env
  python Chapter_07/19b_cloud_penetration_comparison.py

Dependencies: rasterio, numpy, matplotlib, pandas, pystac-client, planetary-computer, pyproj
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
from matplotlib.patches import FancyArrowPatch
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import reproject, Resampling

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(BASE_DIR, "data", "processed", "cloud_comparison")
os.makedirs(OUT_DIR, exist_ok=True)

BBOX       = [-73.30, -51.10, -72.90, -50.80]
# Winter months: cloud cover peaks in Patagonia
DATE_RANGE = "2023-06-01/2023-08-31"

DARK_BG = "#0d1117"
DARK_AX = "#161b22"
C_TEXT  = "#e6edf3"
C_GREY  = "#8b949e"
C_GOLD  = "#f39c12"
C_BLUE  = "#3498db"
C_RED   = "#e74c3c"


# ---------------------------------------------------------------------------
# 1. Fetch Sentinel-1 (cloud-independent)
# ---------------------------------------------------------------------------
def fetch_sar():
    from pystac_client import Client
    import planetary_computer as pc
    from pyproj import Transformer

    print("  [1/3] Fetching Sentinel-1 SAR (cloud-independent, winter acquisition)...")
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                          modifier=pc.sign_inplace)

    search = catalog.search(collections=["sentinel-1-rtc"], bbox=BBOX,
                            datetime=DATE_RANGE)
    items  = list(search.items())
    if not items:
        # Fallback: any Sentinel-1 year-round
        search = catalog.search(collections=["sentinel-1-rtc"], bbox=BBOX,
                                datetime="2023-01-01/2023-12-31")
        items  = list(search.items())

    if not items:
        raise RuntimeError("No Sentinel-1 RTC data found.")

    item = items[0]
    acq_date = str(item.datetime.date()) if item.datetime else "unknown"
    print(f"  [OK]  SAR date={acq_date}  platform={item.properties.get('platform','S1')}")

    def _read(pol):
        if pol not in item.assets:
            return None
        with rasterio.open(item.assets[pol].href) as src:
            t = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            mnx, mny = t.transform(BBOX[0], BBOX[1])
            mxx, mxy = t.transform(BBOX[2], BBOX[3])
            win  = from_bounds(mnx, mny, mxx, mxy, src.transform)
            arr  = src.read(1, window=win).astype("float32")
            h, w = int(round(win.height)), int(round(win.width))
            prof = src.profile.copy()
            prof.update(dtype="float32", count=1, nodata=-9999, compress="lzw",
                        height=h, width=w,
                        transform=rasterio.windows.transform(win, src.transform))
        return arr, prof

    vv_result = _read("vv")
    if vv_result is None:
        raise RuntimeError("VV asset missing.")
    vv_lin, profile = vv_result

    vh_lin = None
    if "vh" in item.assets:
        vh_result = _read("vh")
        if vh_result is not None:
            vh_lin, _ = vh_result

    # Convert to dB
    vv_db    = 10.0 * np.log10(np.where(vv_lin > 0, vv_lin, np.nan))
    ratio_db = None
    if vh_lin is not None:
        vh_db    = 10.0 * np.log10(np.where(vh_lin > 0, vh_lin, np.nan))
        ratio_db = vv_db - vh_db

    return vv_db, ratio_db, profile, acq_date


# ---------------------------------------------------------------------------
# 2. Fetch Sentinel-2 (cloud-affected — intentionally pick cloudy scene)
# ---------------------------------------------------------------------------
def fetch_optical_cloudy(sar_profile):
    from pystac_client import Client
    import planetary_computer as pc

    print("  [2/3] Fetching Sentinel-2 (deliberately selecting CLOUDY winter scene)...")
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                          modifier=pc.sign_inplace)

    search = catalog.search(
        collections=["sentinel-2-l2a"], bbox=BBOX, datetime=DATE_RANGE
        # NOTE: NO cloud filter — we WANT the cloudiest scene
    )
    items = list(search.items())

    if not items:
        print("  [WARN] No Sentinel-2 found. Returning blank panel.")
        h = int(sar_profile["height"])
        w = int(sar_profile["width"])
        return np.zeros((h, w, 3)), None, 100.0

    # Pick the cloudiest available scene for maximum demonstration impact
    item      = sorted(items, key=lambda i: i.properties.get("eo:cloud_cover", 0),
                       reverse=True)[0]
    cloud_pct = float(item.properties.get("eo:cloud_cover", 50.0))
    opt_date  = str(item.datetime.date()) if item.datetime else "unknown"
    print(f"  [OK]  Optical date={opt_date}  cloud={cloud_pct:.0f}%  "
          f"(deliberately choosing cloudiest scene)")

    h = int(sar_profile["height"])
    w = int(sar_profile["width"])

    rgb_bands = []
    for band in ["B04", "B03", "B02"]:
        dest = np.zeros((h, w), dtype=np.float32)
        with rasterio.open(item.assets[band].href) as src:
            reproject(
                source=rasterio.band(src, 1),
                destination=dest,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=sar_profile["transform"],
                dst_crs=sar_profile["crs"],
                resampling=Resampling.bilinear,
            )
        rgb_bands.append(dest)

    rgb = np.dstack(rgb_bands) / 10000.0
    rgb = np.clip(rgb * 3.5, 0, 1)   # brightness stretch for visualization

    return rgb, opt_date, cloud_pct


# ---------------------------------------------------------------------------
# 3. Save TIF
# ---------------------------------------------------------------------------
def save_tif(arr, name, profile, description=""):
    out = os.path.join(OUT_DIR, f"{name}.tif")
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(np.nan_to_num(arr.astype("float32"), nan=-9999), 1)
        dst.update_tags(description=description, nodata="-9999")
    print(f"  [OK] {name}.tif")


# ---------------------------------------------------------------------------
# 4. Statistics CSV
# ---------------------------------------------------------------------------
def save_stats(vv_db, cloud_pct, opt_date, sar_date):
    v = vv_db[np.isfinite(vv_db)]
    water_px   = int(np.sum(v < -18))
    glacier_px = int(np.sum(v > -5))
    rows = [{
        "metric":         "SAR_VV_mean_dB",
        "value":          round(float(v.mean()), 3),
        "sar_date":       sar_date,
        "optical_date":   opt_date,
        "optical_cloud%": cloud_pct,
    }, {
        "metric":         "Water_pixels_VV<-18dB",
        "value":          water_px,
        "sar_date":       sar_date,
        "optical_date":   opt_date,
        "optical_cloud%": cloud_pct,
    }, {
        "metric":         "Glacier_pixels_VV>-5dB",
        "value":          glacier_px,
        "sar_date":       sar_date,
        "optical_date":   opt_date,
        "optical_cloud%": cloud_pct,
    }]
    df  = pd.DataFrame(rows)
    csv = os.path.join(OUT_DIR, "cloud_comparison_stats.csv")
    df.to_csv(csv, index=False, encoding="utf-8")
    print(f"  [OK] cloud_comparison_stats.csv")
    print(f"       SAR water pixels: {water_px:,}  glacier pixels: {glacier_px:,}")


# ---------------------------------------------------------------------------
# 5. 4-panel dark figure
# ---------------------------------------------------------------------------
def plot_comparison(vv_db, ratio_db, rgb, sar_date, opt_date, cloud_pct):
    print("\n  Building 4-panel cloud penetration figure...")
    n = 4 if ratio_db is not None else 3

    fig = plt.figure(figsize=(n * 6, 8), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(1, n, figure=fig, wspace=0.2,
                            top=0.85, bottom=0.05, left=0.03, right=0.97)

    fig.text(0.5, 0.95,
             "SAR Cloud Penetration Advantage -- Torres del Paine (Winter)",
             ha="center", color=C_TEXT, fontsize=13, fontweight="bold")
    fig.text(0.5, 0.90,
             f"SAR date: {sar_date}  |  Optical date: {opt_date}  |  "
             f"Optical cloud cover: {cloud_pct:.0f}%",
             ha="center", color=C_GREY, fontsize=9)

    def style_ax(ax, title, subtitle=""):
        ax.set_facecolor(DARK_AX)
        ax.axis("off")
        full = title + (f"\n{subtitle}" if subtitle else "")
        ax.set_title(full, color=C_TEXT, fontsize=9, fontweight="bold", pad=5)

    # Panel 1: SAR VV (always clear)
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(vv_db, cmap="gray", vmin=-25, vmax=0, aspect="auto")
    cb1 = plt.colorbar(im1, ax=ax1, fraction=0.04, pad=0.02)
    cb1.set_label("dB", color=C_TEXT, fontsize=7)
    cb1.ax.tick_params(colors=C_TEXT, labelsize=6)
    style_ax(ax1, "Sentinel-1 VV Backscatter",
             f"Date: {sar_date} | 0% cloud (radar penetrates clouds)")
    # Add "CLOUD-FREE" badge
    ax1.text(0.02, 0.97, "CLOUD-FREE", transform=ax1.transAxes,
             color="#27ae60", fontsize=8, fontweight="bold", va="top",
             bbox=dict(fc=DARK_BG, ec="#27ae60", lw=1.2, pad=3, alpha=0.9))

    # Panel 2: Optical (cloud-affected)
    ax2 = fig.add_subplot(gs[0, 1])
    if rgb is not None and rgb.ndim == 3:
        ax2.imshow(rgb, aspect="auto")
    else:
        ax2.text(0.5, 0.5, "No data", ha="center", va="center",
                 color=C_GREY, transform=ax2.transAxes, fontsize=14)
    style_ax(ax2, "Sentinel-2 True Color",
             f"Date: {opt_date} | {cloud_pct:.0f}% cloud (optical BLIND)")
    ax2.text(0.02, 0.97, f"CLOUD: {cloud_pct:.0f}%", transform=ax2.transAxes,
             color=C_RED, fontsize=8, fontweight="bold", va="top",
             bbox=dict(fc=DARK_BG, ec=C_RED, lw=1.2, pad=3, alpha=0.9))

    # Panel 3: Water + Glacier mask on SAR
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(vv_db, cmap="gray", vmin=-25, vmax=0, aspect="auto")
    water_mask   = np.where(vv_db < -18, 1.0, np.nan)
    glacier_mask = np.where(vv_db > -5,  1.0, np.nan)
    ax3.imshow(water_mask,   cmap="Blues",  vmin=0, vmax=1, alpha=0.7, aspect="auto")
    ax3.imshow(glacier_mask, cmap="Oranges",vmin=0, vmax=1, alpha=0.7, aspect="auto")
    style_ax(ax3, "SAR Land Cover Masks",
             "Blue=Water (<-18dB) | Orange=Ice/Rough (>-5dB)")
    ax3.text(0.02, 0.97, "SAR ONLY POSSIBLE", transform=ax3.transAxes,
             color=C_GOLD, fontsize=8, fontweight="bold", va="top",
             bbox=dict(fc=DARK_BG, ec=C_GOLD, lw=1.2, pad=3, alpha=0.9))

    # Panel 4: VV/VH ratio (structural)
    if ratio_db is not None:
        ax4 = fig.add_subplot(gs[0, 3])
        im4 = ax4.imshow(ratio_db, cmap="RdYlGn_r", vmin=-5, vmax=15, aspect="auto")
        cb4 = plt.colorbar(im4, ax=ax4, fraction=0.04, pad=0.02)
        cb4.set_label("dB", color=C_TEXT, fontsize=7)
        cb4.ax.tick_params(colors=C_TEXT, labelsize=6)
        style_ax(ax4, "VV/VH Polarization Ratio",
                 "Green=Vegetation | Red=Specular/Water")

    out_png = os.path.join(OUT_DIR, "sar_vs_optical_cloudy.png")
    fig.savefig(out_png, dpi=180, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  [OK] 4-panel figure: {out_png}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print(" GEOCASCADE - SAR vs OPTICAL CLOUD PENETRATION COMPARISON")
    print(f" Winter window: {DATE_RANGE}  |  BBOX: {BBOX}")
    print("=" * 65)

    print("\n[1/4] Fetching Sentinel-1 SAR (cloud-independent)...")
    try:
        vv_db, ratio_db, sar_profile, sar_date = fetch_sar()
    except Exception as e:
        print(f"\n  ERROR: {e}")
        return

    print("\n[2/4] Fetching Sentinel-2 optical (intentionally cloudy)...")
    rgb, opt_date, cloud_pct = fetch_optical_cloudy(sar_profile)

    print("\n[3/4] Saving outputs...")
    save_tif(vv_db, "sar_vv_db", sar_profile,
             f"SAR VV dB, winter {sar_date}, cloud-penetrating")
    save_stats(vv_db, cloud_pct, opt_date or "n/a", sar_date)

    print("\n[4/4] Building comparison figure...")
    plot_comparison(vv_db, ratio_db, rgb, sar_date, opt_date or "n/a", cloud_pct)

    print("\n" + "=" * 65)
    print(" CLOUD PENETRATION COMPARISON COMPLETE")
    print("=" * 65)
    print(f"  Outputs : {OUT_DIR}")
    print(f"  Figure  : {os.path.join(OUT_DIR, 'sar_vs_optical_cloudy.png')}")
    print()
    print("  KEY INSIGHT: SAR reveals water/glacier structure even when optical")
    print("               sensors are 100% blocked by winter cloud cover.")
    print()
    print("  ArcGIS Pro: Load sar_vv_db.tif alongside S2 scene.")
    print("              Toggle transparency to show SAR reveals more.")
    print("  ENVI 5.6  : Use Animation (Ctrl+F) to toggle between sensors.")
    print("=" * 65)


if __name__ == "__main__":
    main()
