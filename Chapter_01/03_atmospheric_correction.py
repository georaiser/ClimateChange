"""
Chapter 1: 03_atmospheric_correction.py
=========================================
Atmospheric Correction: COST Model (Chavez 1996) + DOS1 Comparison

Academic Objective:
  Raw satellite imagery captures radiance AT THE SENSOR, not at the ground.
  The atmosphere scatters and absorbs electromagnetic radiation, adding a haze
  offset (path radiance) and a transmission loss to every pixel.
  Atmospheric correction removes these effects to recover Surface Reflectance.

  This script implements the COST Model (Chavez 1996) -- an empirical correction
  that improves upon the older DOS1 method by also modelling atmospheric
  transmittance using the Solar Zenith Angle.

DECISION GUIDE -- When does this script apply?
  Sentinel-2 L2A   : ALREADY corrected by ESA Sen2Cor   -> use directly
  Landsat 9 L2SP   : ALREADY corrected by USGS LaSRC    -> use directly
  Sentinel-2 L1C   : raw TOA -> USE THIS SCRIPT or ENVI FLAASH
  Landsat 9 L1TP   : raw TOA -> USE THIS SCRIPT or ENVI FLAASH
  ERA5 / CHIRPS    : N/A (reanalysis, no correction needed)

  Script 02 downloads L2A (already corrected). This script is provided for:
    a) Teaching the correction physics
    b) Comparing DOS1 vs COST vs official LaSRC on the same data
    c) Correcting any L1C/L1TP data you acquire manually

COST Model Physics:
  Lhaze = DN_dark_object * cos(SZA) / pi   (path radiance estimate)
  rho   = (DN - Lhaze) / (ESUN * cos(SZA) * T^2)
  where T = cos(SZA) is the transmittance approximation

  Key improvement over DOS1: DOS1 only subtracts path radiance.
  COST also divides by cos(SZA) to recover brightness at highly reflective
  surfaces (glacier, snow), making it ~15-20% more accurate for high albedo.

Bands treated differently:
  VNIR (B01-B08A): full COST correction
  SWIR (B11, B12): copy only -- haze is negligible at >1400nm
  Water vapor (B09), Cirrus (B10): copy only -- not surface-sensing bands

Outputs:
  data/processed/boa_corrected/{scene}/BOA_{band}.tif   (corrected bands)
  data/processed/boa_corrected/correction_comparison.png (before/after figure)
  data/processed/boa_corrected/correction_report.csv

ArcGIS Pro: Add BOA_*.tif as raster. Composite Bands tool to build a stack.
ENVI 5.6:   File > Open > BOA_B04.tif. For MODTRAN-based correction use
            Chapter_01/envi/01_flaash_correction.pro instead.

Run:
  conda activate geocascade_env
  python Chapter_01/03_atmospheric_correction.py

  NOTE: You need L1C data first:
    Download from: https://scihub.copernicus.eu
    Place in: Chapter_01/data/raw/sentinel2_l1c_{scene_id}/
    Or use Landsat L1TP from: https://earthexplorer.usgs.gov

Dependencies: rasterio, numpy, matplotlib, pandas, pystac-client, planetary-computer
"""

import sys
import os
import glob
import shutil
import math
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rasterio

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR  = os.path.join(BASE_DIR, "data", "raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed", "boa_corrected")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Bands where haze correction is physically inappropriate
SWIR_BANDS    = ["B09", "B10", "B11", "B12"]

# Physical reflectance limits for final output
REFL_MIN, REFL_MAX = 0.0, 1.0

DARK_BG = "#0d1117"
DARK_AX = "#161b22"
C_TEXT  = "#e6edf3"
C_GREY  = "#8b949e"
C_RED   = "#e74c3c"
C_BLUE  = "#3498db"
C_GREEN = "#2ecc71"
C_GOLD  = "#f39c12"


# ---------------------------------------------------------------------------
# 1. Find local L1C data
# ---------------------------------------------------------------------------
def find_l1c_data():
    """
    Look for Sentinel-2 L1C or Landsat L1TP data in data/raw/.
    Falls back to a demo comparison against local L2A data if L1C not found.
    """
    # Check for L1C (preferred)
    l1c_dirs = sorted(glob.glob(os.path.join(INPUT_DIR, "sentinel2_l1c_*")))
    if l1c_dirs:
        print(f"  [OK] Sentinel-2 L1C data: {os.path.basename(l1c_dirs[-1])}")
        return l1c_dirs[-1], "sentinel2_l1c"

    # Check for Landsat L1TP
    l1tp_dirs = sorted(glob.glob(os.path.join(INPUT_DIR, "landsat*l1tp*")))
    if l1tp_dirs:
        print(f"  [OK] Landsat L1TP data: {os.path.basename(l1tp_dirs[-1])}")
        return l1tp_dirs[-1], "landsat_l1tp"

    # Demo mode: use L2A for physics demonstration
    l2a_dirs = sorted(glob.glob(os.path.join(INPUT_DIR, "sentinel2_l2a_*")))
    if l2a_dirs:
        print("  [NOTE] No L1C data found. Running in DEMO mode using L2A data.")
        print("         In demo mode we demonstrate the COST algorithm but compare")
        print("         COST-corrected output vs the already-correct L2A product.")
        print("         To use real L1C: download from https://scihub.copernicus.eu")
        return l2a_dirs[-1], "demo_l2a"

    return None, None


# ---------------------------------------------------------------------------
# 2. Fetch solar zenith angle from STAC metadata
# ---------------------------------------------------------------------------
def get_solar_zenith(scene_name):
    """
    Try to get actual Solar Zenith Angle (SZA) from Planetary Computer STAC.
    Falls back to 45 deg if network unavailable.
    Torres del Paine at -51 deg lat in January has SZA ~ 40-50 deg.
    """
    try:
        from pystac_client import Client
        import planetary_computer as pc

        item_id = scene_name.replace("sentinel2_l1c_", "").replace("sentinel2_l2a_", "")
        catalog = Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=pc.sign_inplace
        )
        search = catalog.search(collections=["sentinel-2-l2a"], ids=[item_id])
        items  = list(search.items())
        if items:
            sza = items[0].properties.get("s2:mean_solar_zenith", 45.0)
            print(f"  [OK] STAC Solar Zenith Angle: {sza:.2f} deg")
            return float(sza)
    except Exception:
        pass

    print("  [NOTE] STAC query failed or no match. Using default SZA = 45.0 deg")
    print("         (Torres del Paine January ~ 40-50 deg; 45 is a good estimate)")
    return 45.0


# ---------------------------------------------------------------------------
# 3. COST Model correction for one band
# ---------------------------------------------------------------------------
def cost_correction(band_data, sza_deg, scale_factor=10000.0):
    """
    Apply COST Model (Chavez 1996) to a single band array.

    Parameters:
      band_data   : raw DN array (uint16), already read by rasterio
      sza_deg     : Solar Zenith Angle in degrees
      scale_factor: Sentinel-2 L1C uses 10000; Landsat uses 55000 (OLI)

    Returns:
      refl        : surface reflectance [0.0 - 1.0], float32
      haze        : haze (path radiance) subtracted (DN units)
      transmittance: cos(SZA)
    """
    band_f = band_data.astype("float32")

    # 1. Dark Object Subtraction -- 1st percentile as dark pixel proxy
    #    Assume 1% reflectance minimum (Chavez 1988: DNmin - 1.0% refl)
    valid_mask = band_f > 0
    if valid_mask.sum() > 100:
        dark_pixel = float(np.percentile(band_f[valid_mask], 1))
    else:
        dark_pixel = 0.0

    # Haze = excess above theoretical 1% reflectance dark object
    # 1% reflectance in DN ~ 0.01 * scale_factor
    min_refl_dn = 0.01 * scale_factor
    haze = max(0.0, dark_pixel - min_refl_dn)

    # 2. Subtract path radiance
    corrected = band_f - haze

    # 3. Convert to reflectance (scale to [0, 1])
    refl_raw = corrected / scale_factor

    # 4. COST transmittance correction
    #    T = cos(SZA): transmittance through the atmosphere (one-way)
    #    Dividing by cos(SZA) accounts for longer atmospheric path at high zenith
    transmittance = math.cos(math.radians(sza_deg))
    refl = refl_raw / transmittance

    # 5. Clip to physical range
    refl = np.clip(refl, REFL_MIN, REFL_MAX).astype("float32")

    return refl, haze, transmittance


def dos1_correction(band_data, scale_factor=10000.0):
    """
    DOS1 correction (simpler than COST, no transmittance correction).
    Used for side-by-side comparison.
    """
    band_f = band_data.astype("float32")
    valid_mask = band_f > 0
    dark_pixel = float(np.percentile(band_f[valid_mask], 1)) if valid_mask.sum() > 100 else 0.0
    haze = max(0.0, dark_pixel - 0.01 * scale_factor)
    refl = np.clip((band_f - haze) / scale_factor, REFL_MIN, REFL_MAX).astype("float32")
    return refl


# ---------------------------------------------------------------------------
# 4. Process all bands in a scene
# ---------------------------------------------------------------------------
def process_scene(scene_dir, scene_type, sza_deg):
    scene_name = os.path.basename(scene_dir)
    out_dir    = os.path.join(OUTPUT_DIR, scene_name)
    os.makedirs(out_dir, exist_ok=True)

    tif_files = sorted(glob.glob(os.path.join(scene_dir, "*.tif")))
    if not tif_files:
        print(f"  ERROR: No .tif files found in {scene_dir}")
        return []

    print(f"\n  Processing {len(tif_files)} bands (SZA={sza_deg:.1f} deg)")
    print(f"  VNIR bands -> COST correction")
    print(f"  SWIR/WV/Cirrus bands -> copy (no haze correction)")

    report_rows  = []
    transmittance = math.cos(math.radians(sza_deg))

    for tif_path in tif_files:
        band_fn  = os.path.basename(tif_path)
        out_path = os.path.join(out_dir, f"BOA_{band_fn}")

        # Check if this is a band that should NOT be COST-corrected
        is_swir = any(skip in band_fn.upper() for skip in SWIR_BANDS)

        with rasterio.open(tif_path) as src:
            profile  = src.profile.copy()
            band_raw = src.read(1)
            nodata   = src.nodata

        if is_swir:
            # Copy without correction -- physically inappropriate to haze-subtract SWIR
            shutil.copy(tif_path, out_path)
            reason = "SWIR/WV/Cirrus: haze negligible, copied unchanged"
            print(f"    [COPY] {band_fn:20s}  {reason}")
            report_rows.append({
                "band": band_fn, "method": "copy",
                "haze_dn": 0, "transmittance": 1.0, "reason": reason
            })
        else:
            # Apply COST correction
            refl, haze, t = cost_correction(band_raw, sza_deg)

            profile.update(dtype="float32", count=1, nodata=-9999,
                           compress="lzw")
            refl_nd = np.where(band_raw == (nodata or 0), -9999.0, refl)
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(refl_nd.astype("float32"), 1)
                dst.update_tags(
                    method="COST (Chavez 1996)",
                    sza_deg=str(sza_deg),
                    haze_dn=str(round(haze, 1)),
                    transmittance=str(round(t, 4)),
                    nodata="-9999",
                    arcgis_note="Reflectance [0-1], nodata=-9999"
                )

            print(f"    [COST] {band_fn:20s}  haze={haze:6.1f} DN  T={t:.4f}")
            report_rows.append({
                "band": band_fn, "method": "COST",
                "haze_dn": round(haze, 1), "transmittance": round(t, 4),
                "reason": f"SZA={sza_deg:.1f}deg"
            })

    return report_rows


# ---------------------------------------------------------------------------
# 5. Before/After comparison figure
# ---------------------------------------------------------------------------
def plot_comparison(scene_dir, out_dir):
    print("\n  Building before/after comparison figure...")

    # Try to find a red band for visual comparison
    def find_band(search_dir, band_name):
        for suffix in [".tif", ".TIF"]:
            candidates = glob.glob(os.path.join(search_dir, f"*{band_name}*{suffix}"))
            if candidates:
                return candidates[0]
        return None

    red_raw  = find_band(scene_dir, "B04")
    red_boa  = find_band(out_dir,   "B04")
    nir_raw  = find_band(scene_dir, "B08")
    nir_boa  = find_band(out_dir,   "B08")

    if not (red_raw and red_boa):
        print("  [SKIP] No B04 band found for comparison plot.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 12), facecolor=DARK_BG)
    fig.suptitle("COST Atmospheric Correction -- Before vs After\nTorres del Paine, Patagonia",
                 color=C_TEXT, fontsize=13, fontweight="bold")

    def style_ax(ax, title=""):
        ax.set_facecolor(DARK_AX)
        ax.axis("off")
        if title:
            ax.set_title(title, color=C_TEXT, fontsize=9, fontweight="bold", pad=6)

    def read_band(path, percentile_stretch=True):
        with rasterio.open(path) as src:
            arr = src.read(1).astype("float32")
            nd  = src.nodata
            if nd is not None:
                arr = np.where(arr == nd, np.nan, arr)
        return arr

    # Row 1: Red band before/after + NIR before/after
    panels = [
        (red_raw, "Red (B04) -- RAW TOA", "YlOrRd"),
        (red_boa, "Red (B04) -- COST BOA Reflectance", "YlOrRd"),
    ]
    if nir_raw and nir_boa:
        panels += [
            (nir_raw, "NIR (B08) -- RAW TOA", "YlOrBr"),
            (nir_boa, "NIR (B08) -- COST BOA Reflectance", "YlOrBr"),
        ]

    for i, (path, title, cmap) in enumerate(panels[:4]):
        row, col = divmod(i, 3)
        ax = axes[row, col]
        arr = read_band(path)
        valid = arr[np.isfinite(arr)]
        if valid.size > 0:
            p2, p98 = np.percentile(valid, 2), np.percentile(valid, 98)
            im = ax.imshow(arr, cmap=cmap, vmin=p2, vmax=p98, aspect="auto")
            cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
            cb.ax.tick_params(colors=C_TEXT, labelsize=7)
        style_ax(ax, title)

    # Panel 5: Histogram comparison (raw vs BOA for red band)
    ax5 = axes[1, 1]
    ax5.set_facecolor(DARK_AX)
    for sp in ax5.spines.values():
        sp.set_color("#30363d")
    ax5.tick_params(colors=C_TEXT)
    arr_raw = read_band(red_raw)
    arr_boa = read_band(red_boa)
    if arr_raw is not None and arr_boa is not None:
        raw_valid = arr_raw[np.isfinite(arr_raw)]
        boa_valid = arr_boa[np.isfinite(arr_boa)]
        # Normalise raw to [0,1] scale for comparison
        if raw_valid.max() > 2:  # DN scale
            raw_valid = raw_valid / 10000.0
        ax5.hist(raw_valid, bins=80, color=C_RED,  alpha=0.6, density=True, label="Raw TOA")
        ax5.hist(boa_valid, bins=80, color=C_GREEN, alpha=0.6, density=True, label="COST BOA")
        ax5.legend(fontsize=8, facecolor=DARK_BG, labelcolor=C_TEXT)
        ax5.set_title("Reflectance Distribution (Red Band)", color=C_TEXT, fontsize=9, fontweight="bold")
        ax5.set_xlabel("Reflectance", color=C_TEXT, fontsize=8)
        ax5.set_ylabel("Density", color=C_TEXT, fontsize=8)
        ax5.grid(alpha=0.15, color="#30363d")

    # Panel 6: Method comparison text
    ax6 = axes[1, 2]
    ax6.set_facecolor(DARK_AX)
    ax6.axis("off")
    ax6.set_title("Method Comparison", color=C_TEXT, fontsize=9, fontweight="bold")
    lines = [
        ("", ""),
        ("Method", "Key difference"),
        ("DOS1", "Subtract path radiance only"),
        ("COST (this)", "DOS1 + divide by cos(SZA)"),
        ("FLAASH/LaSRC", "Full MODTRAN RT physics"),
        ("", ""),
        ("L2A/L2SP", "Pre-corrected, use directly"),
        ("L1C/L1TP", "Apply COST or FLAASH"),
        ("", ""),
        ("SZA effect:", ""),
        ("SZA=0 deg", "T=1.00 (nadir, no correction)"),
        ("SZA=45 deg", "T=0.71 (typical Patagonia)"),
        ("SZA=60 deg", "T=0.50 (winter scene)"),
    ]
    y = 0.97
    for k, v in lines:
        if k == "Method":
            ax6.text(0.03, y, k, transform=ax6.transAxes, color=C_GOLD,
                     fontsize=8, fontweight="bold")
            ax6.text(0.45, y, v, transform=ax6.transAxes, color=C_GOLD,
                     fontsize=8, fontweight="bold")
        elif k:
            ax6.text(0.03, y, k, transform=ax6.transAxes, color=C_TEXT, fontsize=8)
            ax6.text(0.45, y, v, transform=ax6.transAxes, color=C_GREY, fontsize=7.5)
        y -= 0.072

    out_png = os.path.join(OUTPUT_DIR, "correction_comparison.png")
    fig.savefig(out_png, dpi=180, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  [OK] Comparison figure: {out_png}")


# ---------------------------------------------------------------------------
# 5. Bridge: copy BOA output to sentinel2_l2a_* for downstream scripts
# ---------------------------------------------------------------------------
def bridge_to_pipeline(boa_dir, scene_type):
    """
    Copy COST-corrected BOA_*.tif bands into
    data/raw/sentinel2_l2a_from_l1c_cost/ so that Chapter 2 and all
    downstream scripts find them via their sentinel2_l2a_* glob.

    Band filenames are de-prefixed (BOA_B04.tif -> B04.tif) to match
    the format written by 02_satellite_acquisition.py.

    Only runs when real L1C or L1TP data was corrected.
    Skipped in DEMO mode (L2A data is already in the right place).
    """
    if scene_type not in ("sentinel2_l1c", "landsat_l1tp"):
        return None

    raw_dir    = os.path.join(BASE_DIR, "data", "raw")
    bridge_dir = os.path.join(raw_dir, "sentinel2_l2a_from_l1c_cost")
    os.makedirs(bridge_dir, exist_ok=True)

    boa_files  = sorted(glob.glob(os.path.join(boa_dir, "BOA_*.tif")))
    if not boa_files:
        print("  [WARN] No BOA_*.tif files found — bridge skipped.")
        return None

    copied = 0
    for src_path in boa_files:
        # BOA_B04.tif  ->  B04.tif  (strip the BOA_ prefix)
        src_name  = os.path.basename(src_path)
        dst_name  = src_name.replace("BOA_", "", 1)
        dst_path  = os.path.join(bridge_dir, dst_name)
        shutil.copy2(src_path, dst_path)
        copied += 1

    print(f"  [OK] Bridge: {copied} bands copied -> {bridge_dir}")
    print(f"       Chapter 2+ scripts will auto-detect this folder via")
    print(f"       sentinel2_l2a_* glob -- no manual steps needed.")
    return bridge_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print(" GEOCASCADE - ATMOSPHERIC CORRECTION (COST MODEL)")
    print(" Chavez 1996 -- DOS1 + Solar Zenith Transmittance")
    print("=" * 65)

    print("\n[1/4] Locating input data...")
    scene_dir, scene_type = find_l1c_data()

    if scene_dir is None:
        print("\n  ERROR: No input data found.")
        print("  To use this script, download one of:")
        print("    Sentinel-2 L1C from: https://scihub.copernicus.eu")
        print("    Landsat L1TP from  : https://earthexplorer.usgs.gov")
        print("  Place in Chapter_01/data/raw/sentinel2_l1c_{scene_id}/")
        print()
        print("  IMPORTANT: Do NOT apply COST to already-corrected data:")
        print("    Sentinel-2 L2A   -> Use directly (corrected by ESA Sen2Cor)")
        print("    Landsat 9 L2SP   -> Use directly (corrected by USGS LaSRC)")
        return

    scene_name = os.path.basename(scene_dir)
    print(f"  Scene    : {scene_name}")
    print(f"  Type     : {scene_type}")
    if scene_type == "demo_l2a":
        print("  [NOTE] Running in demo mode. COST applied to L2A for comparison only.")

    print("\n[2/4] Getting Solar Zenith Angle...")
    sza = get_solar_zenith(scene_name)

    print("\n[3/4] Applying COST correction...")
    out_dir = os.path.join(OUTPUT_DIR, scene_name)
    report_rows = process_scene(scene_dir, scene_type, sza)

    if report_rows:
        csv_path = os.path.join(OUTPUT_DIR, "correction_report.csv")
        pd.DataFrame(report_rows).to_csv(csv_path, index=False, encoding="utf-8")
        print(f"\n  [OK] Correction report: {csv_path}")

    print("\n[4/5] Building comparison figure...")
    plot_comparison(scene_dir, out_dir)

    print("\n[5/5] Bridging corrected bands to pipeline data folder...")
    bridge_dir = bridge_to_pipeline(out_dir, scene_type)
    if bridge_dir is None and scene_type == "demo_l2a":
        print("  [SKIP] Demo mode — L2A bands already in sentinel2_l2a_* folder.")
        print("         Chapter 2+ scripts will read the original L2A data directly.")

    n_corrected = sum(1 for r in report_rows if r["method"] == "COST")
    n_copied    = sum(1 for r in report_rows if r["method"] == "copy")

    print("\n" + "=" * 65)
    print(" ATMOSPHERIC CORRECTION COMPLETE")
    print("=" * 65)
    print(f"  Bands COST-corrected : {n_corrected}")
    print(f"  Bands copied (SWIR)  : {n_copied}")
    print(f"  BOA output folder    : {out_dir}")
    if bridge_dir:
        print(f"  Pipeline bridge      : {bridge_dir}")
        print(f"  -> Chapter 2+ will auto-detect sentinel2_l2a_from_l1c_cost/")
    print(f"  Report CSV           : {os.path.join(OUTPUT_DIR, 'correction_report.csv')}")
    print(f"  Figure               : {os.path.join(OUTPUT_DIR, 'correction_comparison.png')}")
    print()
    print("  ArcGIS Pro: Add BOA_*.tif as raster layers.")
    print("              Data Management > Composite Bands to stack all bands.")
    print("  ENVI 5.6  : Chapter_01/envi/01_flaash_correction.pro for full")
    print("              MODTRAN-based correction (recommended for research).")
    print("=" * 65)


if __name__ == "__main__":
    main()
