"""
Chapter 7: 19_multisensor_review.py

Academic Objective:
To perform a comparative multi-sensor review utilizing Landsat 9 (Optical 30m), 
MODIS (Thermal 1km), and Sentinel-1 (Radar 10m).
This teaches students how to choose the right spectral region (Optical vs Thermal vs Microwave) 
based on cloud cover and physical analysis requirements.
"""

import os
import rasterio
from rasterio.windows import from_bounds
import numpy as np
import matplotlib.pyplot as plt
from pystac_client import Client
import planetary_computer as pc
from pyproj import Transformer
import urllib.request

# ==========================================
# 1. Configuration
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

# Bounding Box: Punta Arenas Region (City, Ocean, Forests)
BBOX = [-71.05, -53.20, -70.80, -53.10]
DATE_RANGE = "2023-01-01/2023-03-31"

def setup_stac():
    print("[INFO] Connecting to Microsoft Planetary Computer...")
    return Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)

# ==========================================
# 2. Fetch Multi-Sensor Data
# ==========================================
def fetch_landsat_9(catalog):
    print("\n[INFO] Fetching Landsat 9 (Optical 30m)...")
    search = catalog.search(
        collections=["landsat-c2-l2"],
        bbox=BBOX,
        datetime=DATE_RANGE,
        query={"platform": {"in": ["landsat-9"]}, "eo:cloud_cover": {"lt": 30}}
    )
    items = list(search.items())
    if not items:
        print("       [WARNING] No Landsat 9 images found with low cloud cover.")
        return None, None
    item = items[0]
    
    # We'll just fetch the NIR band (Band 5) to show an optical view
    href = item.assets["nir08"].href
    
    with rasterio.open(href) as src:
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        minx, miny = transformer.transform(BBOX[0], BBOX[1])
        maxx, maxy = transformer.transform(BBOX[2], BBOX[3])
        window = from_bounds(minx, miny, maxx, maxy, src.transform)
        nir_array = src.read(1, window=window).astype('float32')
        # Scale Landsat C2-L2 Surface Reflectance (Scale factor: 0.0000275, Offset: -0.2)
        nir_sr = (nir_array * 0.0000275) - 0.2
        nir_sr = np.where(nir_array == 0, np.nan, nir_sr)
        return nir_sr, "Landsat 9 NIR (Optical 30m)"

def fetch_modis_thermal(catalog):
    print("\n[INFO] Fetching MODIS LST (Thermal 1km)...")
    search = catalog.search(collections=["modis-11A1-061"], bbox=BBOX, datetime=DATE_RANGE)
    items = list(search.items())
    if not items:
        return None, None
    item = items[-1]
    
    href = item.assets["LST_Day_1km"].href
    out_tmp = os.path.join(OUT_DIR, "modis_tmp.tif")
    urllib.request.urlretrieve(href, out_tmp)
    
    with rasterio.open(out_tmp) as src:
        lst_raw = src.read(1).astype('float32')
        lst_raw[lst_raw == 0] = np.nan
        # Convert to Celsius
        lst_celsius = (lst_raw * 0.02) - 273.15
        return lst_celsius, "MODIS LST (Thermal 1km)"

def fetch_sentinel_1(catalog):
    print("\n[INFO] Fetching Sentinel-1 RTC (Radar 10m)...")
    search = catalog.search(collections=["sentinel-1-rtc"], bbox=BBOX, datetime=DATE_RANGE)
    items = list(search.items())
    if not items:
        return None, None
    item = items[0]
    
    href = item.assets["vv"].href
    with rasterio.open(href) as src:
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        minx, miny = transformer.transform(BBOX[0], BBOX[1])
        maxx, maxy = transformer.transform(BBOX[2], BBOX[3])
        window = from_bounds(minx, miny, maxx, maxy, src.transform)
        vv_array = src.read(1, window=window).astype('float32')
        # Convert to dB
        vv_db = 10 * np.log10(np.where(vv_array <= 0, np.nan, vv_array))
        return vv_db, "Sentinel-1 VV (Radar 10m)"

# ==========================================
# 3. Main Execution and Plotting
# ==========================================
def main():
    print("=======================================================")
    print(" GEOCASCADE - MULTI-SENSOR REVIEW (L9 vs MODIS vs S1)  ")
    print("=======================================================")
    
    catalog = setup_stac()
    
    l9_data, l9_title = fetch_landsat_9(catalog)
    modis_data, modis_title = fetch_modis_thermal(catalog)
    s1_data, s1_title = fetch_sentinel_1(catalog)
    
    print("\n[INFO] Generating Comparative Visualization...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    if l9_data is not None:
        im0 = axes[0].imshow(l9_data, cmap='RdYlGn', vmin=0, vmax=0.6)
        axes[0].set_title(l9_title)
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="Reflectance")
    axes[0].axis('off')
    
    if modis_data is not None:
        im1 = axes[1].imshow(modis_data, cmap='inferno')
        axes[1].set_title(modis_title)
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Celsius (°C)")
    axes[1].axis('off')
    
    if s1_data is not None:
        im2 = axes[2].imshow(s1_data, cmap='gray', vmin=-25, vmax=0)
        axes[2].set_title(s1_title)
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="Decibels (dB)")
    axes[2].axis('off')
    
    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, "multisensor_comparison.png")
    plt.savefig(plot_path, dpi=300)
    print(f"       [SUCCESS] Multi-sensor review saved to: {plot_path}")
    print("\n[SUCCESS] Chapter 7 Pipeline Complete!")

if __name__ == "__main__":
    main()
