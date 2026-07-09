"""
Chapter 7: 18_sentinel1_sar_processing.py

Academic Objective:
Introduce Active Remote Sensing (SAR). This script connects to the Microsoft Planetary 
Computer STAC API to download Sentinel-1 Radiometric Terrain Corrected (RTC) data.
It mathematically converts the linear radar amplitude into Decibels (dB), and performs 
a dual-thresholding analysis to isolate both smooth Water (Flood Mapping) and rough 
Glacial Ice (Glacier Mapping).
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
        raise Exception("No Sentinel-1 data found for this temporal/spatial window.")
        
    print(f"       [SUCCESS] Found {len(items)} SAR scenes.")
    
    # Select the most recent item
    item = items[0]
    print(f"       Selected Acquisition Date: {item.datetime}")
    print(f"       Platform: {item.properties['platform']} | Orbit: {item.properties['sat:orbit_state']}")
    
    # We will process the VV polarization (Vertical transmit, Vertical receive)
    vv_href = item.assets["vv"].href
    
    sar_vv_path = os.path.join(OUT_DIR, "sar_vv_linear.tif")
    
    print(f"[INFO] Downloading VV Polarization...")
    with rasterio.open(vv_href) as src:
        # Reproject BBOX to the SAR's native CRS (usually UTM)
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        minx, miny = transformer.transform(bbox[0], bbox[1])
        maxx, maxy = transformer.transform(bbox[2], bbox[3])
        window = from_bounds(minx, miny, maxx, maxy, src.transform)
        transform = rasterio.windows.transform(window, src.transform)
        
        vv_linear = src.read(1, window=window).astype('float32')
        profile = src.profile
        profile.update(dtype=rasterio.float32, count=1, nodata=np.nan, 
                       height=int(window.height), width=int(window.width), transform=transform)
                       
        with rasterio.open(sar_vv_path, 'w', **profile) as dst:
            dst.write(vv_linear, 1)
            
    return sar_vv_path, vv_linear, profile

# ==========================================
# 3. Radar Physics & Thresholding (Water vs Glacier)
# ==========================================
def process_sar_physics(vv_linear, profile):
    print("\n[INFO] Converting Linear Amplitude to Decibels (dB)...")
    # Physics: dB = 10 * log10(DN)
    # Mask out zeros or negatives to avoid log(0) errors
    vv_linear = np.where(vv_linear <= 0, np.nan, vv_linear)
    vv_db = 10 * np.log10(vv_linear)
    
    sar_db_path = os.path.join(OUT_DIR, "sar_vv_db.tif")
    with rasterio.open(sar_db_path, 'w', **profile) as dst:
        dst.write(vv_db, 1)
    print(f"       [SUCCESS] dB raster exported to: {sar_db_path}")
    
    print("\n[INFO] Running Dual-Thresholding Analysis...")
    # 1. Flood/Water Mapping (Specular Reflection = Low Backscatter)
    # Water usually sits between -25 dB and -18 dB in VV polarization.
    water_mask = np.where(vv_db < -18, 1, np.nan)
    
    # 2. Glacier/Structural Mapping (Volume Scattering = High Backscatter)
    # Crevasses and rough ice bounce a lot of energy back to the sensor (0 to -5 dB)
    glacier_mask = np.where(vv_db > -5, 1, np.nan)
    
    # Generate Output Plot
    print("[INFO] Generating Dual Analysis Visualization...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Raw SAR (dB)
    im1 = axes[0].imshow(vv_db, cmap='gray', vmin=-25, vmax=0)
    axes[0].set_title("Sentinel-1 VV Backscatter (dB)")
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04, label="Decibels (dB)")
    axes[0].axis('off')
    
    # Plot 2: Water Mask
    axes[1].imshow(vv_db, cmap='gray', vmin=-25, vmax=0) # Background
    axes[1].imshow(water_mask, cmap='Blues', vmin=0, vmax=1, alpha=0.7) # Overlay
    axes[1].set_title("Water Extraction (dB < -18)")
    axes[1].axis('off')
    
    # Plot 3: Glacier/Rough Mask
    axes[2].imshow(vv_db, cmap='gray', vmin=-25, vmax=0) # Background
    axes[2].imshow(glacier_mask, cmap='Reds', vmin=0, vmax=1, alpha=0.7) # Overlay
    axes[2].set_title("Glacier & Rough Terrain Extraction (dB > -5)")
    axes[2].axis('off')
    
    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, "sar_dual_analysis.png")
    plt.savefig(plot_path, dpi=300)
    print(f"       [SUCCESS] Visualization saved to: {plot_path}")

def main():
    print("=======================================================")
    print(" GEOCASCADE PIPELINE - SENTINEL-1 SAR PROCESSING       ")
    print("=======================================================")
    try:
        sar_path, vv_linear, profile = fetch_sentinel1_sar(BBOX)
        process_sar_physics(vv_linear, profile)
        print("\n[SUCCESS] Chapter 7 (SAR) Pipeline Complete!")
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")

if __name__ == "__main__":
    main()
