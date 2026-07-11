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
from datetime import datetime, timezone
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
def _item_datetime(item):
    """
    item.datetime is None for some MODIS items, which store their timestamp
    in start_datetime/end_datetime instead (a known pystac behavior for
    items representing a date range rather than a single instant). Fall
    back to those, and finally to the earliest possible datetime so sorting
    never crashes even if none of them are set.
    """
    if item.datetime is not None:
        return item.datetime
    for key in ("start_datetime", "end_datetime"):
        raw = item.properties.get(key)
        if raw:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    return datetime.min.replace(tzinfo=timezone.utc)


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
    
    # Sort by datetime to get the most recent granule (not items[-1] which is oldest)
    items = sorted(items, key=_item_datetime, reverse=True)
    item = items[0]
    print(f"       Selected Acquisition Date: {_item_datetime(item)}")
    
    if "LST_Day_1km" not in item.assets:
        raise KeyError("'LST_Day_1km' asset not found in this MODIS item. Try a different date.")
    download_url = item.assets["LST_Day_1km"].href
    
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
        
        # --- CRITICAL PHYSICS CORRECTION ---
        # MOD11A1 v6.1 fill value is 0 for NO_DATA, and valid range is 7500-65535
        # (0 K would be physically impossible, but 0 is also the actual fill value)
        # Mask fill value (0) AND out-of-range low values (< 7500 = below ~−123°C)
        lst_raw = lst_raw.astype('float32')
        lst_raw[lst_raw < 7500] = np.nan   # masks fill (0) and any sub-range values
        lst_raw[lst_raw > 65535] = np.nan  # guard upper bound
        
        # Apply MODIS LST Scale Factor: 0.02 K per DN (per MOD11A1 User Guide)
        MODIS_LST_SCALE = 0.02  # K / DN
        lst_kelvin  = lst_raw * MODIS_LST_SCALE
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
