"""
Chapter 2: 08_automated_index_batcher.py

Academic Objective:
In commercial GUI software like ArcGIS Pro, you would use "ModelBuilder" to 
create a visual loop to process multiple images (e.g., an entire year's worth of data).

In Python, we achieve this exact same automation using a simple `for` loop.
This script demonstrates how to query the Planetary Computer for an entire year of
Sentinel-2 imagery over our study area, and automatically calculate NDVI for *every*
cloud-free image found, saving the results out as batch files.

Dependencies:
mamba install -n geocascade_env -c conda-forge pystac-client planetary-computer rasterio pyproj numpy -y
"""

import os
import rasterio
from rasterio.windows import from_bounds
import numpy as np
from pystac_client import Client
import planetary_computer as pc
from pyproj import Transformer

# ==========================================
# 1. Configuration
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "data", "processed", "batch_ndvi")
os.makedirs(OUT_DIR, exist_ok=True)

# Torres del Paine BBOX
BBOX = [-73.30, -51.10, -72.90, -50.80]

# We search an entire year for imagery
DATE_RANGE = "2023-01-01/2023-12-31"

def calculate_ndvi(nir, red):
    # Safely calculate NDVI avoiding divide-by-zero
    return np.where((nir + red) == 0, 0, (nir - red) / (nir + red))

# ==========================================
# 2. Automated Batch Processing Loop
# ==========================================
def run_batch_processor():
    print("\n[INFO] Connecting to Planetary Computer STAC API...")
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)

    print(f"[INFO] Searching for all cloud-free (<5%) images in {DATE_RANGE}...")
    search = catalog.search(
        collections=["sentinel-2-l2a"], 
        bbox=BBOX, 
        datetime=DATE_RANGE, 
        query={"eo:cloud_cover": {"lt": 5}}
    )
    
    items = list(search.items())
    print(f"       [SUCCESS] Found {len(items)} scenes matching criteria.")

    # ----------------------------------------------------
    # THE "MODELBUILDER" LOOP
    # ----------------------------------------------------
    for i, item in enumerate(items):
        date_str = item.datetime.strftime("%Y-%m-%d")
        print(f"\n[Processing {i+1}/{len(items)}] Date: {date_str} | ID: {item.id}")
        
        out_filename = os.path.join(OUT_DIR, f"NDVI_{date_str}_{item.id}.tif")
        
        # Skip if we already processed this one (Resuming capability)
        if os.path.exists(out_filename):
            print("       -> Already processed. Skipping.")
            continue
            
        try:
            # 1. Dynamically read Red (B04) and NIR (B08)
            with rasterio.open(item.assets["B04"].href) as src_red:
                # Calculate window
                transformer = Transformer.from_crs("EPSG:4326", src_red.crs, always_xy=True)
                minx, miny = transformer.transform(BBOX[0], BBOX[1])
                maxx, maxy = transformer.transform(BBOX[2], BBOX[3])
                window = from_bounds(minx, miny, maxx, maxy, src_red.transform)
                
                # Extract meta for saving later
                profile = src_red.profile
                # Update profile to match our cropped window
                profile.update({
                    'driver': 'GTiff',
                    'height': window.height,
                    'width': window.width,
                    'transform': rasterio.windows.transform(window, src_red.transform),
                    'dtype': 'float32',
                    'count': 1,
                    'compress': 'lzw'
                })
                
                red = src_red.read(1, window=window).astype('float32') / 10000.0

            with rasterio.open(item.assets["B08"].href) as src_nir:
                nir = src_nir.read(1, window=window).astype('float32') / 10000.0

            # 2. Math
            print("       -> Calculating NDVI...")
            ndvi = calculate_ndvi(nir, red)

            # 3. Save Output
            print(f"       -> Saving to {out_filename}")
            with rasterio.open(out_filename, 'w', **profile) as dst:
                dst.write(ndvi, 1)
                
        except Exception as e:
            print(f"       [ERROR] Failed to process {item.id}: {e}")

def main():
    print("=======================================================")
    print(" GEOCASCADE PIPELINE - AUTOMATED BATCH PROCESSOR       ")
    print("=======================================================")
    run_batch_processor()
    print("\n[SUCCESS] Batch Processing Complete!")

if __name__ == "__main__":
    main()
