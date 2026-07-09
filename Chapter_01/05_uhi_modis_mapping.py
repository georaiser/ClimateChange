"""
Chapter 1: 05_uhi_modis_mapping.py

Academic Objective:
To map the Urban Heat Island (UHI) effect. Cities absorb and retain heat 
differently than surrounding rural landscapes due to asphalt, concrete, 
and lack of vegetation. 

We will use the STAC API to download NASA's MODIS Land Surface Temperature (LST) 
data over Punta Arenas (the largest city near our ROI). We will mathematically 
scale the thermal radiance into degrees Celsius and plot the thermal gradient.

Dependencies:
mamba install -n geocascade_env -c conda-forge pystac-client planetary-computer rasterio matplotlib numpy -y
"""

import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from pystac_client import Client
import planetary_computer as pc

# ==========================================
# 1. Configuration
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "data", "processed", "uhi_mapping")
os.makedirs(OUT_DIR, exist_ok=True)

# Bounding Box for Punta Arenas (City vs Rural)
PUNTA_ARENAS_BBOX = [-71.05, -53.20, -70.80, -53.10]
DATE_RANGE = "2023-01-01/2023-01-31" # Southern Hemisphere Summer

# ==========================================
# 2. Fetch MODIS LST Data via STAC
# ==========================================
def fetch_modis_lst():
    print("\n[INFO] Connecting to Microsoft Planetary Computer STAC API...")
    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace
    )

    print(f"[INFO] Searching for MODIS LST (modis-11A1-061) over Punta Arenas...")
    search = catalog.search(
        collections=["modis-11A1-061"],
        bbox=PUNTA_ARENAS_BBOX,
        datetime=DATE_RANGE
    )
    
    items = list(search.items())
    if not items:
        print("[ERROR] No MODIS data found for this temporal/spatial window.")
        return None
        
    print(f"       [SUCCESS] Found {len(items)} scenes. Selecting the clearest summer day...")
    
    # We select an item (e.g., the first one, which is usually the most recent in the range)
    item = items[-1] 
    print(f"       Selected Acquisition Date: {item.datetime}")
    
    # Download the LST_Day_1km asset
    asset = item.assets["LST_Day_1km"]
    download_url = asset.href
    
    out_tif = os.path.join(OUT_DIR, "punta_arenas_modis_lst.tif")
    
    print(f"[INFO] Downloading thermal raster to {out_tif}...")
    urllib.request.urlretrieve(download_url, out_tif)
    
    return out_tif

# ==========================================
# 3. Process Thermal Data (Physics Conversion)
# ==========================================
def process_and_plot_lst(tif_path):
    print("\n[INFO] Processing Thermal Radiance into Celsius...")
    
    with rasterio.open(tif_path) as src:
        # Read the thermal band
        lst_raw = src.read(1).astype('float32')
        
        # MODIS LST_Day_1km has a valid range and a fill value (often 0)
        # We mask out the 0s (no data / clouds)
        lst_raw[lst_raw == 0] = np.nan
        
        # --- THE PHYSICS CONVERSION ---
        # 1. Apply MODIS Scale Factor (0.02) to convert raw DN to Kelvin
        lst_kelvin = lst_raw * 0.02
        
        # 2. Convert Kelvin to Celsius
        lst_celsius = lst_kelvin - 273.15
        
        # Calculate statistics
        mean_c = np.nanmean(lst_celsius)
        max_c = np.nanmax(lst_celsius)
        min_c = np.nanmin(lst_celsius)
        print(f"       [STATISTICS] Regional Mean: {mean_c:.1f}°C")
        print(f"       [STATISTICS] Urban Peak:    {max_c:.1f}°C")
        print(f"       [STATISTICS] Rural Min:     {min_c:.1f}°C")
        
        # --- EXPORT GEOCODED TIFF ---
        lst_celsius_tif = os.path.join(OUT_DIR, "uhi_celsius.tif")
        profile = src.profile
        profile.update(dtype=rasterio.float32, count=1, nodata=np.nan)
        with rasterio.open(lst_celsius_tif, 'w', **profile) as dst:
            dst.write(lst_celsius, 1)
        print(f"       [SUCCESS] Geocoded TIFF saved to: {lst_celsius_tif} (Ready for ArcGIS/ENVI)")
        
        print("\n[INFO] Generating Urban Heat Island Heatmap...")
        plt.figure(figsize=(10, 8))
        
        # Use the 'inferno' or 'hot' colormap to clearly visualize heat
        heatmap = plt.imshow(lst_celsius, cmap='inferno', vmin=min_c, vmax=max_c)
        
        cbar = plt.colorbar(heatmap, shrink=0.7)
        cbar.set_label('Land Surface Temperature (°C)', fontsize=12)
        
        plt.title('Urban Heat Island (UHI) Effect - Punta Arenas\nMODIS Terra LST (1km resolution)', fontsize=14)
        plt.axis('off') # Hide coordinate axes for a cleaner map
        
        plot_path = os.path.join(OUT_DIR, "uhi_heatmap.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"       [SUCCESS] Heatmap saved to: {plot_path}")

def main():
    print("=======================================================")
    print(" GEOCASCADE PIPELINE - URBAN HEAT ISLAND (UHI) MAPPING ")
    print("=======================================================")
    
    tif_path = fetch_modis_lst()
    if tif_path:
        process_and_plot_lst(tif_path)
        print("\n[SUCCESS] Chapter 1 complete! You have successfully mastered Optical, Thermal, and Climate Data.")

if __name__ == "__main__":
    main()
