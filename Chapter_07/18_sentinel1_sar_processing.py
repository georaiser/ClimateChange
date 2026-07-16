"""
Chapter 7: 18_sentinel1_sar_processing.py
==========================================
Sentinel-1 SAR Active Microwave Remote Sensing

Academic Objective:
  Optical satellites (Sentinel-2, Landsat) are PASSIVE sensors — they depend on
  sunlight reflected from the surface. In Patagonia, cloud cover exceeds 80% of days.
  Sentinel-1 is an ACTIVE sensor: it transmits its own microwave pulse and measures
  the echo (backscatter). Microwaves penetrate clouds, rain, and darkness completely.

  This script processes Sentinel-1 RTC (Radiometrically Terrain Corrected) data
  using both polarization channels:

  VV (Vertical transmit / Vertical receive):
    - Sensitive to surface roughness, soil moisture, open water
    - Very low return from specular surfaces (calm water, smooth ice)
    - High return from rough/angular surfaces (ice crevasses, terrain)

  VH (Vertical transmit / Horizontal receive):
    - Sensitive to VOLUME SCATTERING (vegetation canopy, forest structure)
    - Random depolarization by multiple reflections inside vegetation
    - Near-zero for bare soil and water (no volume scattering)

  Derived products:
    dB = 10 * log10(linear amplitude)   [decibel conversion]
    VV/VH ratio = VV_dB - VH_dB         [structural discriminator]
    CR = Cross-Ratio = VH / VV           [vegetation index analog]

  Land cover SAR thresholds (Patagonian landscape):
    VV < -18 dB  → specular return → flat water / calm lake surface
    VV > -5 dB   → high return → rough glacier, rocky terrain, forest
    VH/VV high   → strong depolarization → vegetation canopy
    VH/VV low    → no depolarization → bare surface / ice

Physics note: RTC data is already gamma0 corrected for terrain using the DEM.
Do NOT apply additional terrain correction to Planetary Computer RTC assets.

Connection to pipeline:
  SAR VV dB → Band 2 of cascade_master_stack.tif (Script 20)
  Water mask → validates NDWI from Script 14
  Glacier mask → cross-validates NDSI from Script 07

Outputs:
  data/processed/sar/sar_vv_linear.tif
  data/processed/sar/sar_vv_db.tif
  data/processed/sar/sar_vh_db.tif
  data/processed/sar/sar_vv_vh_ratio_db.tif
  data/processed/sar/sar_cr.tif               (cross-ratio = VH/VV linear)
  data/processed/sar/water_mask_sar.tif
  data/processed/sar/glacier_mask_sar.tif
  data/processed/sar/sar_statistics.csv
  data/processed/sar/sar_dual_analysis.png    (5-panel dark figure)

ArcGIS Pro: Add sar_vv_db.tif. Symbology > Stretched > Gray.
            Use Raster Calculator: Con("sar_vv_db.tif" < -18, 1, 0) for water mask.
ENVI 5.6:   File > Open > sar_vv_db.tif. Apply Density Slice thresholds.
            Band Math: (10 * alog10(b1)) converts linear to dB.

Run:
  conda activate geocascade_env
  python Chapter_07/18_sentinel1_sar_processing.py

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
import rasterio
from rasterio.windows import from_bounds

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(BASE_DIR, "data", "processed", "sar")
os.makedirs(OUT_DIR, exist_ok=True)

BBOX       = [-73.30, -51.10, -72.90, -50.80]
DATE_RANGE = "2023-01-01/2023-03-31"

# SAR land-cover thresholds (VV dB, Patagonian landscape)
WATER_THRESH_DB   = -18.0   # VV < -18 dB = specular/flat water
GLACIER_THRESH_DB = -5.0    # VV > -5 dB  = rough glacier / rocky terrain
# VH/VV cross-ratio threshold for vegetation
VEG_CR_THRESH = 0.35        # CR > 0.35 = strong volume scattering

DARK_BG = "#0d1117"
DARK_AX = "#161b22"
C_TEXT  = "#e6edf3"
C_GREY  = "#8b949e"
C_BLUE  = "#3498db"
C_RED   = "#e74c3c"
C_GREEN = "#27ae60"
C_GOLD  = "#f39c12"


# ---------------------------------------------------------------------------
# 1. Fetch Sentinel-1 RTC from STAC
# ---------------------------------------------------------------------------
def fetch_sentinel1():
    try:
        from pystac_client import Client
        import planetary_computer as pc
        from pyproj import Transformer
    except ImportError:
        raise ImportError("pystac-client / planetary-computer not installed.")

    print("  Connecting to Planetary Computer STAC...")
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                          modifier=pc.sign_inplace)

    search = catalog.search(collections=["sentinel-1-rtc"], bbox=BBOX,
                            datetime=DATE_RANGE)
    items  = list(search.items())

    if not items:
        # Widen date range
        search = catalog.search(collections=["sentinel-1-rtc"], bbox=BBOX,
                                datetime="2023-01-01/2023-06-30")
        items  = list(search.items())

    if not items:
        raise RuntimeError("No Sentinel-1 RTC data found. Check BBOX or date range.")

    item = items[0]
    acq_date  = str(item.datetime.date()) if item.datetime else "unknown"
    platform  = item.properties.get("platform", "S1")
    orbit     = item.properties.get("sat:orbit_state", "?")
    print(f"  [OK] Scene: {item.id}")
    print(f"       Date: {acq_date}  |  Platform: {platform}  |  Orbit: {orbit}")

    from pyproj import Transformer as T
    def _read_pol(pol_key):
        """Read one polarization band with correct windowing."""
        if pol_key not in item.assets:
            return None, None
        with rasterio.open(item.assets[pol_key].href) as src:
            t = T.from_crs("EPSG:4326", src.crs, always_xy=True)
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

    print("  Reading VV polarization...")
    vv_lin, profile = _read_pol("vv")
    if vv_lin is None:
        raise RuntimeError("VV asset not found in Sentinel-1 scene.")

    vh_lin = None
    if "vh" in item.assets:
        print("  Reading VH polarization...")
        vh_lin, _ = _read_pol("vh")
    else:
        print("  [NOTE] No VH asset found. Cross-polarization analysis will be skipped.")

    return vv_lin, vh_lin, profile, acq_date, platform


# ---------------------------------------------------------------------------
# 2. Physics: linear → dB, masks, cross-ratio
# ---------------------------------------------------------------------------
def process_sar_physics(vv_lin, vh_lin, profile):
    # Guard: zero/negative linear values → undefined in log space
    vv_safe = np.where(vv_lin > 0, vv_lin, np.nan)
    vv_db   = 10.0 * np.log10(vv_safe)

    vh_db   = None
    ratio_db = None
    cr       = None   # cross-ratio VH/VV (linear)

    if vh_lin is not None:
        vh_safe  = np.where(vh_lin > 0, vh_lin, np.nan)
        vh_db    = 10.0 * np.log10(vh_safe)
        ratio_db = vv_db - vh_db   # in dB: positive when VV > VH
        # Cross-ratio (linear space): VH/VV — vegetation analog
        cr = np.where(vv_safe > 0, vh_safe / vv_safe, np.nan)

    # Dual-threshold masks
    water_mask   = np.where(vv_db < WATER_THRESH_DB,   1.0, np.nan)
    glacier_mask = np.where(vv_db > GLACIER_THRESH_DB, 1.0, np.nan)
    veg_mask     = (np.where(cr > VEG_CR_THRESH, 1.0, np.nan)
                   if cr is not None else None)

    return vv_db, vh_db, ratio_db, cr, water_mask, glacier_mask, veg_mask


# ---------------------------------------------------------------------------
# 3. Save GeoTIFFs
# ---------------------------------------------------------------------------
def save_tif(data, name, profile, description=""):
    if data is None:
        return None
    out = os.path.join(OUT_DIR, f"{name}.tif")
    safe = np.nan_to_num(data.astype("float32"), nan=-9999)
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(safe, 1)
        dst.update_tags(description=description, nodata="-9999",
                        arcgis_note="Stretched gray symbology for dB products",
                        envi_note="Open as single band, apply density slice")
    print(f"  [OK] {name}.tif")
    return out


# ---------------------------------------------------------------------------
# 4. Statistics CSV
# ---------------------------------------------------------------------------
def save_stats(vv_db, vh_db, water_mask, glacier_mask, profile):
    pix_size = abs(profile.get("transform", [10])[0])
    pix_km2  = (pix_size / 1000.0) ** 2 if pix_size > 1 else (10.0 / 1000.0) ** 2

    rows = []
    for name, arr in [("VV_dB", vv_db), ("VH_dB", vh_db)]:
        if arr is None:
            continue
        v = arr[np.isfinite(arr)]
        rows.append({
            "band":    name,
            "mean_dB": round(float(v.mean()), 3),
            "std_dB":  round(float(v.std()),  3),
            "min_dB":  round(float(v.min()),  3),
            "max_dB":  round(float(v.max()),  3),
        })

    # Mask areas
    water_px   = int(np.nansum(water_mask))   if water_mask is not None   else 0
    glacier_px = int(np.nansum(glacier_mask)) if glacier_mask is not None else 0
    rows.append({"band": "Water_mask",   "mean_dB": water_px,   "std_dB": None,
                 "min_dB": round(water_px * pix_km2, 3), "max_dB": None})
    rows.append({"band": "Glacier_mask", "mean_dB": glacier_px, "std_dB": None,
                 "min_dB": round(glacier_px * pix_km2, 3), "max_dB": None})

    df  = pd.DataFrame(rows)
    csv = os.path.join(OUT_DIR, "sar_statistics.csv")
    df.to_csv(csv, index=False, encoding="utf-8")

    print(f"\n  --- SAR Thresholding Report ---")
    print(f"  Water pixels  (VV < {WATER_THRESH_DB} dB)  : "
          f"{water_px:>7,}  = {water_px*pix_km2:.2f} km2")
    print(f"  Glacier pixels(VV > {GLACIER_THRESH_DB} dB)   : "
          f"{glacier_px:>7,}  = {glacier_px*pix_km2:.2f} km2")
    print(f"  [OK] sar_statistics.csv")


# ---------------------------------------------------------------------------
# 5. 5-panel dark figure
# ---------------------------------------------------------------------------
def plot_sar(vv_db, vh_db, ratio_db, cr, water_mask, glacier_mask):
    print("\n  Building 5-panel SAR figure...")

    n_panels = 5 if (ratio_db is not None and cr is not None) else 3
    fig = plt.figure(figsize=(n_panels * 5, 8), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(1, n_panels, figure=fig, hspace=0.1, wspace=0.18,
                            top=0.88, bottom=0.05, left=0.03, right=0.97)
    fig.text(0.5, 0.95, "Sentinel-1 SAR Processing -- Torres del Paine",
             ha="center", color=C_TEXT, fontsize=13, fontweight="bold")

    def style_ax(ax, title):
        ax.set_facecolor(DARK_AX)
        ax.axis("off")
        ax.set_title(title, color=C_TEXT, fontsize=8.5, fontweight="bold", pad=5)

    # Panel 1: VV dB (gray)
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(vv_db, cmap="gray", vmin=-25, vmax=0, aspect="auto")
    cb1 = plt.colorbar(im1, ax=ax1, fraction=0.04, pad=0.02)
    cb1.set_label("dB", color=C_TEXT, fontsize=7)
    cb1.ax.tick_params(colors=C_TEXT, labelsize=6)
    style_ax(ax1, "VV Backscatter (dB)\nGray = surface roughness proxy")

    # Panel 2: Water mask overlay
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(vv_db, cmap="gray", vmin=-25, vmax=0, aspect="auto")
    if water_mask is not None:
        ax2.imshow(water_mask, cmap="Blues_r", vmin=0, vmax=1, alpha=0.75, aspect="auto")
    style_ax(ax2, f"Water Mask (VV < {WATER_THRESH_DB} dB)\nBlue = specular/flat water")

    # Panel 3: Glacier mask overlay
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(vv_db, cmap="gray", vmin=-25, vmax=0, aspect="auto")
    if glacier_mask is not None:
        ax3.imshow(glacier_mask, cmap="Reds_r", vmin=0, vmax=1, alpha=0.75, aspect="auto")
    style_ax(ax3, f"Glacier/Rough Mask (VV > {GLACIER_THRESH_DB} dB)\nRed = rough surface / ice crevasses")

    if n_panels == 5:
        # Panel 4: VV/VH ratio
        ax4 = fig.add_subplot(gs[0, 3])
        im4 = ax4.imshow(ratio_db, cmap="RdYlGn_r", vmin=-5, vmax=15, aspect="auto")
        cb4 = plt.colorbar(im4, ax=ax4, fraction=0.04, pad=0.02)
        cb4.set_label("dB", color=C_TEXT, fontsize=7)
        cb4.ax.tick_params(colors=C_TEXT, labelsize=6)
        style_ax(ax4, "VV/VH Ratio (dB)\nGreen=Vegetation | Red=Specular")

        # Panel 5: Cross-ratio (VH/VV linear)
        ax5 = fig.add_subplot(gs[0, 4])
        cr_disp = np.clip(cr, 0, 1)
        im5 = ax5.imshow(cr_disp, cmap="YlGn", vmin=0, vmax=0.8, aspect="auto")
        cb5 = plt.colorbar(im5, ax=ax5, fraction=0.04, pad=0.02)
        cb5.set_label("VH/VV", color=C_TEXT, fontsize=7)
        cb5.ax.tick_params(colors=C_TEXT, labelsize=6)
        style_ax(ax5, "Cross-Ratio VH/VV (linear)\nGreen = dense vegetation canopy")

    out_png = os.path.join(OUT_DIR, "sar_dual_analysis.png")
    fig.savefig(out_png, dpi=180, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  [OK] 5-panel figure: {out_png}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print(" GEOCASCADE - SENTINEL-1 SAR PROCESSING")
    print(f" Water: VV < {WATER_THRESH_DB} dB  |  Glacier: VV > {GLACIER_THRESH_DB} dB")
    print("=" * 65)

    print("\n[1/4] Fetching Sentinel-1 RTC from Planetary Computer...")
    try:
        vv_lin, vh_lin, profile, acq_date, platform = fetch_sentinel1()
    except Exception as e:
        print(f"\n  ERROR: {e}")
        return

    print(f"\n[2/4] Computing SAR physics (linear -> dB, masks, cross-ratio)...")
    vv_db, vh_db, ratio_db, cr, water_mask, glacier_mask, veg_mask = \
        process_sar_physics(vv_lin, vh_lin, profile)

    print(f"\n[3/4] Saving GeoTIFFs...")
    save_tif(vv_lin,      "sar_vv_linear",     profile, "SAR VV linear amplitude (gamma0 RTC)")
    save_tif(vv_db,       "sar_vv_db",         profile, "SAR VV backscatter in dB")
    save_tif(vh_db,       "sar_vh_db",         profile, "SAR VH backscatter in dB")
    save_tif(ratio_db,    "sar_vv_vh_ratio_db",profile, "VV/VH ratio in dB (positive=VV>VH)")
    save_tif(cr,          "sar_cr",            profile, "Cross-ratio VH/VV linear (vegetation proxy)")
    save_tif(water_mask,  "water_mask_sar",    profile, f"SAR water mask (VV < {WATER_THRESH_DB} dB)")
    save_tif(glacier_mask,"glacier_mask_sar",  profile, f"SAR glacier/rough mask (VV > {GLACIER_THRESH_DB} dB)")
    save_stats(vv_db, vh_db, water_mask, glacier_mask, profile)

    print("\n[4/4] Building 5-panel SAR figure...")
    plot_sar(vv_db, vh_db, ratio_db, cr, water_mask, glacier_mask)

    print("\n" + "=" * 65)
    print(" SAR PROCESSING COMPLETE")
    print("=" * 65)
    print(f"  Outputs : {OUT_DIR}")
    print(f"  Figure  : {os.path.join(OUT_DIR, 'sar_dual_analysis.png')}")
    print(f"  Stats   : {os.path.join(OUT_DIR, 'sar_statistics.csv')}")
    print()
    print("  ArcGIS Pro: Add sar_vv_db.tif. Stretched > Gray.")
    print("              Raster Calc: Con(sar_vv_db < -18, 1, 0) = water mask.")
    print("  ENVI 5.6  : Open sar_vv_db.tif. Density Slice thresholds.")
    print("              Band Math: 10*alog10(b1) to convert linear to dB.")
    print("=" * 65)


if __name__ == "__main__":
    main()
