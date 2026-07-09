"""
Chapter 8: 20_multisensor_data_fusion.py

Academic Objective:
To perform advanced Multi-Sensor Data Fusion. We will download data from 4 completely 
different sensors (Sentinel-2, Sentinel-1, DEM, MODIS) and mathematically resample 
and reproject them onto a single, perfectly aligned 10m spatial grid.

Output: A 4-band Data Cube (master_cascade_stack.tif) ready for Machine Learning.
"""

import os
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds as transform_from_bounds
import numpy as np
import urllib.request
from pystac_client import Client
import planetary_computer as pc
from pyproj import Transformer
import warnings

# Suppress STAC/Rasterio warnings for cleaner output
warnings.filterwarnings("ignore")

# ==========================================
# 1. Configuration
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "data", "processed")
TMP_DIR = os.path.join(BASE_DIR, "data", "tmp")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

# BBOX: Grey Glacier / Lago Grey (Perfect for ice, water, vegetation variations)
BBOX = [-73.30, -51.10, -72.90, -50.80]
DATE_RANGE = "2023-01-01/2023-03-31"

def setup_stac():
    print("[INFO] Connecting to Microsoft Planetary Computer...")
    return Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)

# ==========================================
# 2. Base Grid Generation (Sentinel-2 10m)
# ==========================================
def fetch_base_grid(catalog):
    print("\n[1/4] Fetching Sentinel-2 (Optical NIR) for Master 10m Grid...")
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=BBOX,
        datetime=DATE_RANGE,
        query={"eo:cloud_cover": {"lt": 10}}
    )
    items = list(search.items())
    if not items:
        raise Exception("No S2 images found. Try a different date/cloud cover.")
    
    # Sort by cloud cover and pick the clearest
    item = sorted(items, key=lambda i: i.properties["eo:cloud_cover"])[0]
    href = item.assets["B08"].href # NIR Band (10m)
    
    with rasterio.open(href) as src:
        # Reproject BBOX to S2 CRS (UTM)
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        minx, miny = transformer.transform(BBOX[0], BBOX[1])
        maxx, maxy = transformer.transform(BBOX[2], BBOX[3])
        window = from_bounds(minx, miny, maxx, maxy, src.transform)
        
        nir_array = src.read(1, window=window).astype('float32')
        transform = rasterio.windows.transform(window, src.transform)
        
        # This profile becomes our MASTER GRID for all other sensors
        master_profile = src.profile.copy()
        master_profile.update(
            height=window.height,
            width=window.width,
            transform=transform,
            count=4, # We will stack 4 bands total
            dtype=rasterio.float32,
            nodata=np.nan
        )
        return nir_array, master_profile

# ==========================================
# 3. Resampling Engine
# ==========================================
def fetch_and_resample(catalog, collection, asset_name, master_profile, layer_name, process_func=None):
    if collection == "cop-dem-glo-30":
        search = catalog.search(collections=[collection], bbox=BBOX)
    else:
        search = catalog.search(collections=[collection], bbox=BBOX, datetime=DATE_RANGE)
    items = list(search.items())
    if not items:
        print(f"       [WARNING] No data found for {layer_name}")
        return np.full((int(master_profile['height']), int(master_profile['width'])), np.nan)
        
    item = items[0]
    href = item.assets[asset_name].href
    
    # Destination array for resampled data
    dest_array = np.zeros((int(master_profile['height']), int(master_profile['width'])), dtype=np.float32)
    
    with rasterio.open(href) as src:
        # Reproject from native CRS/Resolution to Master CRS/Resolution
        reproject(
            source=rasterio.band(src, 1),
            destination=dest_array,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=master_profile['transform'],
            dst_crs=master_profile['crs'],
            resampling=Resampling.bilinear
        )
        
    if process_func:
        dest_array = process_func(dest_array)
        
    print(f"       [SUCCESS] Resampled to match Master 10m Grid.")
    return dest_array

# Physics Processors
def process_sar(arr):
    # Convert SAR linear to dB
    return 10 * np.log10(np.where(arr <= 0, np.nan, arr))

def process_modis(arr):
    # Convert MODIS LST to Celsius
    arr = np.where(arr == 0, np.nan, arr)
    return (arr * 0.02) - 273.15

# ==========================================
# 4. Main Fusion Workflow
# ==========================================
def main():
    print("=======================================================")
    print(" GEOCASCADE - MULTI-SENSOR DATA FUSION ENGINE          ")
    print("=======================================================")
    
    catalog = setup_stac()
    
    # 1. Base Grid (Optical)
    nir_array, master_profile = fetch_base_grid(catalog)
    
    # 2. Sentinel-1 SAR (Structural)
    sar_db = fetch_and_resample(catalog, "sentinel-1-rtc", "vv", master_profile, "Sentinel-1 (SAR VV)", process_sar)
    
    # 3. Copernicus DEM (Elevation)
    dem = fetch_and_resample(catalog, "cop-dem-glo-30", "data", master_profile, "Copernicus DEM (Elevation)")
    
    # 4. MODIS LST (Thermal)
    # MODIS requires downloading first due to HDF container complexities in GDAL
    search = catalog.search(collections=["modis-11A1-061"], bbox=BBOX, datetime=DATE_RANGE)
    modis_item = list(search.items())[-1]
    modis_tmp = os.path.join(TMP_DIR, "modis_tmp.tif")
    urllib.request.urlretrieve(modis_item.assets["LST_Day_1km"].href, modis_tmp)
    
    print("\n[INFO] Fetching and Resampling MODIS LST (Thermal 1km)...")
    dest_modis = np.zeros((int(master_profile['height']), int(master_profile['width'])), dtype=np.float32)
    with rasterio.open(modis_tmp) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dest_modis,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=master_profile['transform'],
            dst_crs=master_profile['crs'],
            resampling=Resampling.bilinear
        )
    lst_celsius = process_modis(dest_modis)
    print("       [SUCCESS] Resampled to match Master 10m Grid.")

    # 5. Build and Export the Data Cube
    print("\n[INFO] Stacking aligned layers into multidimensional Data Cube...")
    out_path = os.path.join(OUT_DIR, "cascade_master_stack.tif")
    
    with rasterio.open(out_path, 'w', **master_profile) as dst:
        dst.set_band_description(1, "S2_NIR")
        dst.set_band_description(2, "S1_SAR_VV_dB")
        dst.set_band_description(3, "DEM_Elevation")
        dst.set_band_description(4, "MODIS_LST_Celsius")
        
        dst.write(nir_array, 1)
        dst.write(sar_db, 2)
        dst.write(dem, 3)
        dst.write(lst_celsius, 4)
        
    print(f"\n[SUCCESS] Fusion Complete! Saved Data Cube: {out_path}")
    print("          The stack is now ready for Machine Learning (Random Forests).")

if __name__ == "__main__":
    main()
