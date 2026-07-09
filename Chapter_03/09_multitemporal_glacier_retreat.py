"""
Chapter 3: 09_multitemporal_glacier_retreat.py

Academic Objective:
One of the most visible impacts of climate change is the rapid retreat of glaciers.
In this script, we perform a "Multi-temporal Analysis" (comparing two different points in time).

We will query the Planetary Computer for Landsat imagery of Grey Glacier from exactly
20 years apart (2003 vs 2023). We will calculate the Normalized Difference Snow Index (NDSI)
for both years to map the glacial ice, and then mathematically subtract them to highlight
the exact areas where the ice has melted into a glacial lake.

Dependencies:
mamba install -n geocascade_env -c conda-forge pystac-client planetary-computer rasterio pyproj matplotlib numpy -y
"""

import os
import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
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

# Tight Bounding Box around the front of Grey Glacier
BBOX = [-73.30, -51.10, -73.15, -50.90]

# ==========================================
# 2. Planetary Computer Queries
# ==========================================
def fetch_landsat_item(catalog, start_year, end_year):
    print(f"[INFO] Searching for the clearest Landsat imagery between {start_year} and {end_year}...")
    search = catalog.search(
        collections=["landsat-c2-l2"],
        bbox=BBOX,
        datetime=f"{start_year}-01-01/{end_year}-12-31", 
        query={"eo:cloud_cover": {"lt": 40}} # Relaxed to 40% because Patagonia is cloudy
    )
    items = list(search.items())
    if not items:
        raise ValueError(f"No images found between {start_year} and {end_year}.")
    
    # Sort by lowest cloud cover and return the best one
    best_item = sorted(items, key=lambda x: x.properties["eo:cloud_cover"])[0]
    print(f"       [SUCCESS] Found Image: {best_item.id} (Cloud Cover: {best_item.properties['eo:cloud_cover']}%)")
    return best_item

def calculate_ndsi(green, swir1):
    # NDSI = (Green - SWIR1) / (Green + SWIR1)
    return np.where((green + swir1) == 0, 0, (green - swir1) / (green + swir1))

# ==========================================
# 3. Main Processing Workflow
# ==========================================
def process_retreat():
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
    
    # 1. Fetch STAC Items (Epochs)
    item_2003 = fetch_landsat_item(catalog, 2000, 2005)
    item_2023 = fetch_landsat_item(catalog, 2019, 2023)
    
    print("\n[INFO] Streaming and aligning pixel data from the Cloud...")
    
    # We use 2023 as our "Master Grid"
    with rasterio.open(item_2023.assets["green"].href) as src2023_green:
        transformer2023 = Transformer.from_crs("EPSG:4326", src2023_green.crs, always_xy=True)
        minx, miny = transformer2023.transform(BBOX[0], BBOX[1])
        maxx, maxy = transformer2023.transform(BBOX[2], BBOX[3])
        win2023 = from_bounds(minx, miny, maxx, maxy, src2023_green.transform)
        
        # --- CRITICAL: Apply Landsat Collection 2 Level-2 Scale Factor ---
        # Landsat C2-L2 Surface Reflectance is stored as scaled integers.
        # Formula: SR = DN * 0.0000275 + (-0.2)
        # WITHOUT this step, raw DNs (~7000-20000) produce NDSI values near 1.0
        # for ALL pixels, making the glacier threshold meaningless.
        green_2023 = green_2023 * 0.0000275 - 0.2
        green_2023 = np.clip(green_2023, 0, 1)  # Clamp to valid [0,1] reflectance range
        
    with rasterio.open(item_2023.assets["swir16"].href) as src:
        swir_2023 = src.read(1, window=win2023).astype('float32')
        swir_2023 = swir_2023 * 0.0000275 - 0.2
        swir_2023 = np.clip(swir_2023, 0, 1)

    # Read 2003 Data, forcing it to resample/align exactly to the 2023 array dimensions
    with rasterio.open(item_2003.assets["green"].href) as src2003_green:
        transformer2003 = Transformer.from_crs("EPSG:4326", src2003_green.crs, always_xy=True)
        minx, miny = transformer2003.transform(BBOX[0], BBOX[1])
        maxx, maxy = transformer2003.transform(BBOX[2], BBOX[3])
        win2003 = from_bounds(minx, miny, maxx, maxy, src2003_green.transform)
        
        green_2003 = src2003_green.read(1, window=win2003, out_shape=target_shape, resampling=Resampling.bilinear).astype('float32')
        green_2003 = green_2003 * 0.0000275 - 0.2
        green_2003 = np.clip(green_2003, 0, 1)
        
    with rasterio.open(item_2003.assets["swir16"].href) as src:
        swir_2003 = src.read(1, window=win2003, out_shape=target_shape, resampling=Resampling.bilinear).astype('float32')
        swir_2003 = swir_2003 * 0.0000275 - 0.2
        swir_2003 = np.clip(swir_2003, 0, 1)

    print("\n[INFO] Calculating Normalized Difference Snow Index (NDSI)...")
    ndsi_2003 = calculate_ndsi(green_2003, swir_2003)
    ndsi_2023 = calculate_ndsi(green_2023, swir_2023)
    
    # Threshold NDSI > 0.4 to isolate pure Snow/Ice
    ice_2003 = (ndsi_2003 > 0.4).astype(int)
    ice_2023 = (ndsi_2023 > 0.4).astype(int)
    
    print("\n[INFO] Calculating Glacial Retreat Map...")
    # Math: 2003 Ice (1) - 2023 Ice (1) = 0 (Unchanged)
    # Math: 2003 Ice (1) - 2023 Ice (0) = 1 (Ice Melted / Retreated)
    # Math: 2003 Ice (0) - 2023 Ice (1) = -1 (Ice Advanced)
    retreat_map = ice_2003 - ice_2023
    
    print("\n[INFO] Exporting Geocoded TIFF for ArcGIS/ENVI...")
    out_tif = os.path.join(OUT_DIR, "glacier_retreat_2003_2023.tif")
    with rasterio.open(out_tif, 'w', **profile) as dst:
        dst.write(retreat_map, 1)
    print(f"       [SUCCESS] Exported TIFF: {out_tif}")

    print("\n[INFO] Generating Multi-temporal Map...")
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    axs[0].imshow(ice_2003, cmap='Blues', vmin=0, vmax=1)
    axs[0].set_title(f'Grey Glacier Extent (2003)\nNDSI > 0.4')
    axs[0].axis('off')
    
    axs[1].imshow(ice_2023, cmap='Blues', vmin=0, vmax=1)
    axs[1].set_title(f'Grey Glacier Extent (2023)\nNDSI > 0.4')
    axs[1].axis('off')
    
    # Custom colormap for retreat: 
    # -1 (Advanced) = Blue, 0 (Stable) = Gray, 1 (Retreated) = Red
    from matplotlib.colors import ListedColormap
    cmap_diff = ListedColormap(['cyan', 'lightgray', 'red'])
    im_diff = axs[2].imshow(retreat_map, cmap=cmap_diff, vmin=-1, vmax=1)
    axs[2].set_title('Glacial Retreat Analysis\nRed = Ice Lost (Melted)')
    axs[2].axis('off')
    
    # Add a custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='Ice Retreated (Melt)'),
        Patch(facecolor='lightgray', label='Stable Ice/Land'),
        Patch(facecolor='cyan', label='Ice Advanced')
    ]
    axs[2].legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.15))
    
    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, "glacier_retreat_2003_2023.png")
    plt.savefig(plot_path, dpi=300)
    print(f"       [SUCCESS] Multi-temporal map saved to: {plot_path}")

def main():
    print("=======================================================")
    print(" GEOCASCADE PIPELINE - GLACIAL RETREAT ANALYSIS        ")
    print("=======================================================")
    try:
        process_retreat()
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")

if __name__ == "__main__":
    main()
