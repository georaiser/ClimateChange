"""
Chapter 1: 01_stac_multisensor_download.py

Academic Objective:
Learn how to programmatically search and download geospatial data using 
Spatiotemporal Asset Catalogs (STAC) APIs. This eliminates manual downloads 
from web portals and ensures a reproducible data pipeline.

Region of Interest (ROI):
Torres del Paine National Park & Grey Glacier, Magallanes Region, Chile.

Sensors Targeted:
1. Sentinel-2 (Optical) - For glacial lakes and vegetation.
2. Copernicus DEM GLO-30 - For terrain modeling and hydrology (Chapter 3).

Dependencies:
conda install -c conda-forge pystac-client planetary-computer requests
"""

import os
import json
import requests
from pystac_client import Client
import planetary_computer as pc

# ==========================================
# 1. Configuration and ROI Definition
# ==========================================

# Torres del Paine / Grey Glacier Bounding Box [min_lon, min_lat, max_lon, max_lat]
# This defines the exact spatial area we want to query.
ROI_BBOX = [-73.30, -51.10, -72.90, -50.80]

# Date range for the query (e.g., peak summer for minimal snow cover)
DATE_RANGE = "2023-01-01/2023-02-28"

# Maximum cloud cover percentage acceptable for optical imagery
MAX_CLOUD_COVER = 10

# Output directory for downloaded raw data
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================
# 2. STAC Client Setup
# ==========================================
def setup_stac_client():
    """
    Connects to the Microsoft Planetary Computer STAC API.
    Planetary Computer is free, open, and hosts Sentinel-2 and Copernicus DEM.
    """
    print("[INFO] Connecting to Planetary Computer STAC API...")
    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace
    )
    return catalog


# ==========================================
# 3. Sentinel-2 Optical Search
# ==========================================
def search_sentinel2(catalog):
    """
    Searches for Sentinel-2 Level-2A (Surface Reflectance) images 
    within the ROI, date range, and cloud cover threshold.
    """
    print(f"\n[INFO] Searching for Sentinel-2 L2A data...")
    print(f"       BBOX: {ROI_BBOX}")
    print(f"       Date: {DATE_RANGE}")
    
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=ROI_BBOX,
        datetime=DATE_RANGE,
        query={"eo:cloud_cover": {"lt": MAX_CLOUD_COVER}}
    )
    
    items = list(search.items())
    print(f"[SUCCESS] Found {len(items)} Sentinel-2 scenes matching criteria.")
    
    # Sort by cloud cover and get the clearest image
    if not items:
        print("[WARNING] No optical images found. Try relaxing date or cloud cover.")
        return None
        
    clearest_item = sorted(items, key=lambda i: i.properties["eo:cloud_cover"])[0]
    print(f"[SELECTED] Scene ID: {clearest_item.id} | Cloud Cover: {clearest_item.properties['eo:cloud_cover']:.2f}%")
    
    return clearest_item


# ==========================================
# 4. Copernicus DEM Search
# ==========================================
def search_dem(catalog):
    """
    Searches for the Copernicus DEM (GLO-30) for the ROI.
    DEMs are static, so we don't need a strict datetime filter, but STAC requires one.
    """
    print(f"\n[INFO] Searching for Copernicus DEM GLO-30...")
    
    search = catalog.search(
        collections=["cop-dem-glo-30"],
        bbox=ROI_BBOX
    )
    
    items = list(search.items())
    print(f"[SUCCESS] Found {len(items)} DEM tiles.")
    
    if items:
        # We just grab the first intersecting tile
        selected_dem = items[0]
        print(f"[SELECTED] DEM ID: {selected_dem.id}")
        return selected_dem
    return None


# ==========================================
# 5. Asset Download Helper
# ==========================================
def download_asset(url, output_path):
    """
    Downloads a file from a URL in chunks to handle large rasters.
    """
    if os.path.exists(output_path):
        print(f"       [SKIP] File already exists: {output_path}")
        return

    print(f"       [DOWNLOAD] Fetching {os.path.basename(output_path)}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(output_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    print(f"       [DONE] Saved to {output_path}")


# ==========================================
# 6. Main Execution Block
# ==========================================
def main():
    print("==================================================")
    print(" GEOCASCADE PIPELINE - STAC ACQUISITION SCRIPT")
    print("==================================================")
    
    catalog = setup_stac_client()
    
    # 1. Find and download Sentinel-2 (Red, Green, Blue, NIR bands)
    s2_item = search_sentinel2(catalog)
    if s2_item:
        s2_dir = os.path.join(OUTPUT_DIR, f"sentinel2_{s2_item.id}")
        os.makedirs(s2_dir, exist_ok=True)
        
        # We download 10m bands (B02, B03, B04, B08) and the 20m SWIR band (B11) for Aerosols/NDSI
        target_bands = ["B02", "B03", "B04", "B08", "B11"]
        for band in target_bands:
            if band in s2_item.assets:
                url = s2_item.assets[band].href
                out_path = os.path.join(s2_dir, f"{band}.tif")
                download_asset(url, out_path)
    
    # 2. Find and download DEM
    dem_item = search_dem(catalog)
    if dem_item:
        dem_dir = os.path.join(OUTPUT_DIR, f"dem_{dem_item.id}")
        os.makedirs(dem_dir, exist_ok=True)
        
        # The primary DEM data is in the 'data' asset
        if "data" in dem_item.assets:
            url = dem_item.assets["data"].href
            out_path = os.path.join(dem_dir, "copernicus_dem_30m.tif")
            download_asset(url, out_path)

    print("\n[SUCCESS] Chapter 1 Acquisition complete. Data is ready for preprocessing.")

if __name__ == "__main__":
    main()
