"""
Chapter 3: 11_watershed_delineation.py

Academic Objective:
Water flows downhill. By algorithmically routing water across our Digital Elevation Model (DEM),
we can delineate entire drainage basins and trace river networks automatically. 

This script replicates the ESRI 'ArcHydro' workflow using the `pysheds` library.
1. Fill Sinks (Depressions) so water doesn't get trapped.
2. Calculate Flow Direction (D8 routing).
3. Calculate Flow Accumulation (how many pixels drain into a specific point).
4. Delineate the River Network.

Dependencies:
mamba install -n geocascade_env -c conda-forge pysheds pystac-client planetary-computer rasterio matplotlib numpy -y
"""

import os
import rasterio
from rasterio.windows import from_bounds
import numpy as np

# Monkey-patch for numpy 2.0 compatibility with pysheds
if not hasattr(np, 'in1d'):
    np.in1d = np.isin

import matplotlib.pyplot as plt
from matplotlib import colors
from pystac_client import Client
import planetary_computer as pc
from pyproj import Transformer
from pysheds.grid import Grid

# ==========================================
# 1. Configuration
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
OUT_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

BBOX = [-73.30, -51.10, -72.90, -50.80]
TEMP_DEM = os.path.join(RAW_DIR, "temp_dem.tif")

# ==========================================
# 2. Download a Local Crop for pysheds
# ==========================================
def fetch_local_dem():
    if os.path.exists(TEMP_DEM):
        return
        
    print("\n[INFO] Fetching Copernicus DEM crop from Planetary Computer...")
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
    search = catalog.search(collections=["cop-dem-glo-30"], bbox=BBOX)
    item = list(search.items())[0]
    
    with rasterio.open(item.assets["data"].href) as src:
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        minx, miny = transformer.transform(BBOX[0], BBOX[1])
        maxx, maxy = transformer.transform(BBOX[2], BBOX[3])
        window = from_bounds(minx, miny, maxx, maxy, src.transform)
        
        dem = src.read(1, window=window)
        profile = src.profile
        profile.update({
            'driver': 'GTiff', 'height': window.height, 'width': window.width,
            'transform': rasterio.windows.transform(window, src.transform)
        })
        
        with rasterio.open(TEMP_DEM, 'w', **profile) as dst:
            dst.write(dem, 1)

# ==========================================
# 3. Hydrological Modeling (pysheds)
# ==========================================
def delineate_watershed():
    print("\n[INFO] Initializing PySheds Grid...")
    grid = Grid.from_raster(TEMP_DEM)
    dem = grid.read_raster(TEMP_DEM)
    
    # 1. Condition the DEM (Fill Sinks)
    print("       -> Filling Depressions (Sinks)...")
    pit_filled_dem = grid.fill_pits(dem)
    flooded_dem = grid.fill_depressions(pit_filled_dem)
    inflated_dem = grid.resolve_flats(flooded_dem)
    
    # 2. Flow Direction
    print("       -> Calculating Flow Direction (D8 Algorithm)...")
    # ESRI D8 mapping: 1=E, 2=SE, 4=S, 8=SW, 16=W, 32=NW, 64=N, 128=NE
    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
    fdir = grid.flowdir(inflated_dem, dirmap=dirmap)
    
    # 3. Flow Accumulation
    print("       -> Calculating Flow Accumulation...")
    acc = grid.accumulation(fdir, dirmap=dirmap)
    
    print("\n[INFO] Exporting Geocoded TIFFs for ArcGIS/ENVI...")
    with rasterio.open(TEMP_DEM) as src:
        profile = src.profile
        
    def save_tif(data, name, dtype='float32'):
        out_tif = os.path.join(OUT_DIR, f"{name}.tif")
        prof = profile.copy()
        prof.update(dtype=dtype, nodata=-9999)
        with rasterio.open(out_tif, 'w', **prof) as dst:
            dst.write(np.nan_to_num(np.array(data), nan=-9999).astype(dtype), 1)
        print(f"       [SUCCESS] Exported TIFF: {out_tif}")
        
    save_tif(fdir, "flow_direction", dtype='int32')
    save_tif(acc, "flow_accumulation", dtype='float32')

    print("\n[INFO] Generating Hydrological Maps...")
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # DEM
    im1 = axs[0].imshow(dem, cmap='terrain', zorder=1)
    axs[0].set_title('Raw Digital Elevation Model')
    fig.colorbar(im1, ax=axs[0], shrink=0.7)
    axs[0].axis('off')
    
    # Flow Direction
    im2 = axs[1].imshow(fdir, cmap='viridis')
    axs[1].set_title('Flow Direction (D8)')
    fig.colorbar(im2, ax=axs[1], shrink=0.7)
    axs[1].axis('off')
    
    # River Network (Log Scale of Flow Accumulation)
    axs[2].imshow(dem, cmap='gray', alpha=0.5) # Background DEM
    acc_map = np.where(acc > 1000, acc, np.nan) # Only show major rivers (>1000 pixels draining)
    im3 = axs[2].imshow(acc_map, cmap='Blues', norm=colors.LogNorm(vmin=1000, vmax=acc.max()))
    axs[2].set_title('River Network (Flow > 1000 pixels)')
    fig.colorbar(im3, ax=axs[2], shrink=0.7)
    axs[2].axis('off')
    
    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, "watershed_analysis.png")
    plt.savefig(plot_path, dpi=300)
    print(f"       [SUCCESS] Watershed map saved to: {plot_path}")

def main():
    print("=======================================================")
    print(" GEOCASCADE PIPELINE - WATERSHED DELINEATION           ")
    print("=======================================================")
    try:
        fetch_local_dem()
        delineate_watershed()
        print("\n[SUCCESS] Script 11 Complete!")
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")

if __name__ == "__main__":
    main()
