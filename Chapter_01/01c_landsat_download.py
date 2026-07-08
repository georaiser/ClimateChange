"""
Chapter 1: 01c_landsat_download.py

Academic Objective:
Learn how to adapt STAC queries to acquire data from different satellite constellations.
Here, we query the NASA/USGS Landsat 8/9 constellation (Collection 2, Level-2).
Because this is Level-2 data, it is already Surface Reflectance (atmospherically corrected),
so we can bypass the FLAASH/COST atmospheric correction scripts and proceed directly 
to index calculations.

Region of Interest (ROI):
Torres del Paine National Park & Grey Glacier, Magallanes Region, Chile.

Dependencies:
mamba install -n geocascade_env -c conda-forge pystac-client planetary-computer requests
"""

import os
import requests
from pystac_client import Client
import planetary_computer as pc

# ==========================================
# 1. Configuration and ROI Definition
# ==========================================
ROI_BBOX = [-73.30, -51.10, -72.90, -50.80]
DATE_RANGE = "2023-01-01/2023-02-28"
MAX_CLOUD_COVER = 30  # Increased to 30% because Landsat revisits less frequently than Sentinel-2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "raw")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================
# 2. STAC Client Setup
# ==========================================
def setup_stac_client():
    print("[INFO] Connecting to Planetary Computer STAC API...")
    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace
    )
    return catalog


# ==========================================
# 3. Landsat Optical Search
# ==========================================
def search_landsat(catalog):
    """
    Searches for Landsat 8/9 Collection 2 Level-2 (Surface Reflectance).
    """
    print(f"\n[INFO] Searching for Landsat C2-L2 (Surface Reflectance) data...")
    print(f"       BBOX: {ROI_BBOX}")
    print(f"       Date: {DATE_RANGE}")
    
    search = catalog.search(
        collections=["landsat-c2-l2"],
        bbox=ROI_BBOX,
        datetime=DATE_RANGE,
        query={"eo:cloud_cover": {"lt": MAX_CLOUD_COVER}}
    )
    
    items = list(search.items())
    print(f"[SUCCESS] Found {len(items)} Landsat scenes matching criteria.")
    
    if not items:
        print("[WARNING] No optical images found. Try relaxing date or cloud cover.")
        return None
        
    clearest_item = sorted(items, key=lambda i: i.properties["eo:cloud_cover"])[0]
    print(f"[SELECTED] Scene ID: {clearest_item.id} | Cloud Cover: {clearest_item.properties['eo:cloud_cover']:.2f}%")
    
    return clearest_item


# ==========================================
# 4. Asset Download Helper
# ==========================================
def download_asset(url, output_path):
    if os.path.exists(output_path):
        print(f"       [SKIP] File already exists: {os.path.basename(output_path)}")
        return

    print(f"       [DOWNLOAD] Fetching {os.path.basename(output_path)}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(output_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    print(f"       [DONE] Saved to {output_path}")


# ==========================================
# 5. Main Execution Block
# ==========================================
def main():
    print("==================================================")
    print(" GEOCASCADE PIPELINE - LANDSAT STAC ACQUISITION")
    print("==================================================")
    
    catalog = setup_stac_client()
    
    landsat_item = search_landsat(catalog)
    
    if landsat_item:
        landsat_dir = os.path.join(OUTPUT_DIR, f"landsat_{landsat_item.id}")
        os.makedirs(landsat_dir, exist_ok=True)
        
        # We download the equivalent of S2 (Blue, Green, Red, NIR, SWIR1)
        target_assets = ["blue", "green", "red", "nir08", "swir16"]
        
        for asset in target_assets:
            if asset in landsat_item.assets:
                url = landsat_item.assets[asset].href
                out_path = os.path.join(landsat_dir, f"{asset}.tif")
                download_asset(url, out_path)
                
    print("\n[SUCCESS] Landsat Acquisition complete.")

if __name__ == "__main__":
    main()
