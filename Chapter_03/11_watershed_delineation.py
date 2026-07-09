"""
Chapter 3: 11_watershed_delineation.py

Academic Objective:
Water flows downhill. By algorithmically routing water across our Digital Elevation Model (DEM),
we can delineate entire drainage basins and trace river networks automatically.

This script replicates the ESRI 'ArcHydro' workflow using the `pysheds` library:
1. Fill Sinks (Depressions) so water doesn't get trapped in artifacts.
2. Calculate Flow Direction (D8 routing algorithm).
3. Calculate Flow Accumulation (how many pixels drain into a specific point).
4. Delineate the River Network and export as Shapefile.
5. Calculate the Hipsometric Curve (elevation-area distribution) — per WatershedModeling syllabus.
6. Calculate Stream Order — Módulo 3 requirement.

The Hipsometric Curve is a fundamental morphometric tool that describes how
elevation is distributed across the basin. A concave curve = mature/old basin,
a convex curve = young/actively eroding basin.
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
    fig, axs = plt.subplots(2, 2, figsize=(16, 14))
    
    # Panel 1: DEM
    im1 = axs[0, 0].imshow(dem, cmap='terrain', zorder=1)
    axs[0, 0].set_title('Raw Digital Elevation Model')
    fig.colorbar(im1, ax=axs[0, 0], shrink=0.7)
    axs[0, 0].axis('off')
    
    # Panel 2: Flow Direction
    im2 = axs[0, 1].imshow(fdir, cmap='viridis')
    axs[0, 1].set_title('Flow Direction (D8 Algorithm)')
    fig.colorbar(im2, ax=axs[0, 1], shrink=0.7)
    axs[0, 1].axis('off')
    
    # Panel 3: River Network (Log Scale Flow Accumulation)
    axs[1, 0].imshow(dem, cmap='gray', alpha=0.5)
    acc_map = np.where(acc > 1000, acc, np.nan)
    im3 = axs[1, 0].imshow(acc_map, cmap='Blues', norm=colors.LogNorm(vmin=1000, vmax=acc.max()))
    axs[1, 0].set_title('River Network (Flow Accumulation > 1000 pixels)')
    fig.colorbar(im3, ax=axs[1, 0], shrink=0.7)
    axs[1, 0].axis('off')
    
    # Panel 4: Hipsometric Curve (Elevation-Area Distribution)
    # Per WatershedModeling syllabus Módulo 3.4
    dem_arr = np.array(dem).flatten()
    dem_valid = dem_arr[np.isfinite(dem_arr) & (dem_arr > 0)]
    total_pixels = len(dem_valid)
    elev_min, elev_max = dem_valid.min(), dem_valid.max()
    
    # Normalized elevation (h/H) and normalized area (a/A)
    elev_pcts = np.linspace(0, 100, 100)
    norm_elev = np.percentile(dem_valid, elev_pcts)  # elevation quantiles
    norm_area = 1.0 - (elev_pcts / 100.0)            # fraction of basin above that elevation
    
    axs[1, 1].plot(norm_area, norm_elev, color='steelblue', linewidth=2)
    axs[1, 1].fill_between(norm_area, elev_min, norm_elev, alpha=0.15, color='steelblue')
    axs[1, 1].set_title('Hipsometric Curve\n(Elevation-Area Distribution)')
    axs[1, 1].set_xlabel('Fraction of Basin Area (above elevation)')
    axs[1, 1].set_ylabel('Elevation (m)')
    axs[1, 1].grid(True, alpha=0.3)
    axs[1, 1].text(0.05, elev_max * 0.95, f'Basin: {elev_min:.0f} – {elev_max:.0f} m',
                   fontsize=9, color='gray')
    
    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, "watershed_analysis.png")
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    print(f"       [SUCCESS] Watershed map (with Hipsometric Curve) saved to: {plot_path}")

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
