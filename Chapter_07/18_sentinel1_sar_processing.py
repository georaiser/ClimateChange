"""
Chapter 7: 18_sentinel1_sar_processing.py

Academic Objective:
Introduce Active Remote Sensing (SAR). This script connects to the Microsoft Planetary 
Computer STAC API to download Sentinel-1 Radiometric Terrain Corrected (RTC) data.
It processes BOTH polarization channels:
  - VV (Vertical-Vertical): sensitive to soil moisture, surface water, ice
  - VH (Vertical-Horizontal): sensitive to volume scattering (vegetation, forest structure)
The VV/VH ratio provides a powerful discriminator between surface types.
Final outputs: dB rasters, dual-threshold masks, and a 4-panel figure.
"""

import os
import rasterio
from rasterio.windows import from_bounds
import numpy as np
import matplotlib.pyplot as plt
from pystac_client import Client
import planetary_computer as pc
from pyproj import Transformer

# ==========================================
# 1. Configuration
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

# Bounding Box: Grey Glacier and Grey Lake (Patagonia)
# This is perfect for both Water (smooth) and Glacier (rough) radar returns!
BBOX = [-73.30, -51.10, -72.90, -50.80]
DATE_RANGE = "2023-01-01/2023-02-28"

# ==========================================
# 2. Fetch Sentinel-1 RTC via STAC
# ==========================================
def fetch_sentinel1_sar(bbox):
    print("\n[INFO] Connecting to Microsoft Planetary Computer STAC API...")
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)

    print(f"[INFO] Searching for Sentinel-1 RTC (Radar) over BBOX: {bbox}...")
    search = catalog.search(
        collections=["sentinel-1-rtc"],
        bbox=bbox,
        datetime=DATE_RANGE
    )

    items = list(search.items())
    if not items:
        raise RuntimeError("No Sentinel-1 data found for this temporal/spatial window.")

    print(f"       [SUCCESS] Found {len(items)} SAR scenes.")
    item = items[0]
    print(f"       Selected Acquisition Date: {item.datetime}")
    print(f"       Platform: {item.properties['platform']} | Orbit: {item.properties['sat:orbit_state']}")

    transformer = Transformer.from_crs("EPSG:4326", None, always_xy=True)  # placeholder

    def read_pol(pol_name):
        """Stream a single polarization band into a float32 array."""
        href = item.assets[pol_name].href
        with rasterio.open(href) as src:
            t = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            minx, miny = t.transform(bbox[0], bbox[1])
            maxx, maxy = t.transform(bbox[2], bbox[3])
            window = from_bounds(minx, miny, maxx, maxy, src.transform)
            arr = src.read(1, window=window).astype('float32')
            profile = src.profile
            profile.update(
                dtype=rasterio.float32, count=1, nodata=-9999,
                height=int(round(window.height)), width=int(round(window.width)),
                transform=rasterio.windows.transform(window, src.transform)
            )
        return arr, profile

    print("[INFO] Downloading VV Polarization...")
    vv_linear, profile = read_pol("vv")

    # Tier 3: also download VH polarization for cross-polarisation ratio
    if "vh" in item.assets:
        print("[INFO] Downloading VH Polarization (Tier 3 — vegetation/volume scattering)...")
        vh_linear, _ = read_pol("vh")
    else:
        print("       [WARNING] No VH asset found. Cross-pol analysis skipped.")
        vh_linear = None

    # Save raw VV linear TIFF
    sar_vv_path = os.path.join(OUT_DIR, "sar_vv_linear.tif")
    with rasterio.open(sar_vv_path, 'w', **profile) as dst:
        dst.write(np.nan_to_num(vv_linear, nan=-9999), 1)

    return vv_linear, vh_linear, profile

# ==========================================
# 3. Radar Physics & Thresholding (Water vs Glacier)
# ==========================================
def process_sar_physics(vv_linear, vh_linear, profile):
    print("\n[INFO] Converting Linear Amplitude to Decibels (dB)...")
    vv_linear = np.where(vv_linear <= 0, np.nan, vv_linear)
    vv_db = 10 * np.log10(vv_linear)

    sar_db_path = os.path.join(OUT_DIR, "sar_vv_db.tif")
    with rasterio.open(sar_db_path, 'w', **profile) as dst:
        dst.write(np.nan_to_num(vv_db, nan=-9999), 1)
    print(f"       [SUCCESS] dB raster exported to: {sar_db_path}")

    # Tier 3: VV/VH ratio (in dB = VV_dB − VH_dB)
    # Ratio ≈ 0 dB  → double-bounce (urban, ice crevasses)
    # Ratio >> 0 dB → volume scattering (forest)
    # Ratio << 0 dB → specular surface (water)
    ratio_db = None
    if vh_linear is not None:
        vh_linear = np.where(vh_linear <= 0, np.nan, vh_linear)
        vh_db = 10 * np.log10(vh_linear)
        ratio_db = vv_db - vh_db
        ratio_path = os.path.join(OUT_DIR, "sar_vv_vh_ratio_db.tif")
        with rasterio.open(ratio_path, 'w', **profile) as dst:
            dst.write(np.nan_to_num(ratio_db, nan=-9999), 1)
        print(f"       [SUCCESS] VV/VH ratio raster exported to: {ratio_path}")

    print("\n[INFO] Running Dual-Thresholding Analysis...")
    water_mask   = np.where(vv_db < -18, 1, np.nan)  # specular return → water/ice lake
    glacier_mask = np.where(vv_db >  -5, 1, np.nan)  # volume/double-bounce → rough glacier

    # --- Quantitative Area Report ---
    pix_m = abs(profile.get('transform', [10])[0])  # pixel size in native units
    pix_km2 = (pix_m / 1000) ** 2
    water_px   = int(np.nansum(water_mask))
    glacier_px = int(np.nansum(glacier_mask))
    print("\n--- SAR THRESHOLDING REPORT ---")
    print(f"  Water pixels    (VV < -18 dB) : {water_px:>6,}  ≈ {water_px * pix_km2:.2f} km²")
    print(f"  Glacier pixels  (VV >  -5 dB) : {glacier_px:>6,}  ≈ {glacier_px * pix_km2:.2f} km²")
    print("-" * 42)

    # --- Visualization ---
    n_panels = 4 if ratio_db is not None else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6))

    im1 = axes[0].imshow(vv_db, cmap='gray', vmin=-25, vmax=0)
    axes[0].set_title("Sentinel-1 VV Backscatter (dB)")
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04, label="dB")
    axes[0].axis('off')

    axes[1].imshow(vv_db, cmap='gray', vmin=-25, vmax=0)
    axes[1].imshow(water_mask, cmap='Blues', vmin=0, vmax=1, alpha=0.7)
    axes[1].set_title("Water Extraction (VV < −18 dB)")
    axes[1].axis('off')

    axes[2].imshow(vv_db, cmap='gray', vmin=-25, vmax=0)
    axes[2].imshow(glacier_mask, cmap='Reds', vmin=0, vmax=1, alpha=0.7)
    axes[2].set_title("Glacier & Rough Terrain (VV > −5 dB)")
    axes[2].axis('off')

    if ratio_db is not None:
        im4 = axes[3].imshow(ratio_db, cmap='RdYlGn', vmin=-5, vmax=15)
        axes[3].set_title("VV/VH Ratio (dB)\nGreen = Vegetation | Red = Specular")
        plt.colorbar(im4, ax=axes[3], fraction=0.046, pad=0.04, label="dB")
        axes[3].axis('off')

    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, "sar_dual_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)  # prevent memory leak
    print(f"       [SUCCESS] Visualization saved to: {plot_path}")

def main():
    print("=======================================================")
    print(" GEOCASCADE PIPELINE - SENTINEL-1 SAR PROCESSING       ")
    print("=======================================================")
    try:
        vv_linear, vh_linear, profile = fetch_sentinel1_sar(BBOX)
        process_sar_physics(vv_linear, vh_linear, profile)
        print("\n[SUCCESS] Chapter 7 (SAR) Pipeline Complete!")
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")

if __name__ == "__main__":
    main()
