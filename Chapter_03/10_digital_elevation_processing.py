"""
Chapter 3: 10_digital_elevation_processing.py

Academic Objective:
A Digital Elevation Model (DEM) is a 3D representation of terrain. But raw elevation 
values alone don't give us a clear picture of the landscape. 

In this script, we replicate the ESRI Spatial Analyst tools to generate:
1. Slope (Steepness of the terrain)
2. Aspect (Compass direction the terrain faces)
3. Hillshade (3D visual relief based on a simulated sun angle)

We will use pure NumPy matrix operations to understand the exact physics 
and geometry occurring under the hood of GIS software.

Dependencies:
mamba install -n geocascade_env -c conda-forge pystac-client planetary-computer rasterio numpy matplotlib -y
"""

import os
import rasterio
from rasterio.windows import from_bounds
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

BBOX = [-73.30, -51.10, -72.90, -50.80]

# ==========================================
# 2. Mathematical Terrain Functions
# ==========================================
def calculate_slope_aspect(dem, cellsize=30.0):
    # Calculate gradients (rate of change) in X and Y directions
    dy, dx = np.gradient(dem, cellsize, cellsize)
    
    # Slope (in degrees)
    # slope = arctan(sqrt(dx^2 + dy^2))
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_deg = np.degrees(slope_rad)
    
    # Aspect (in degrees from North)
    # aspect = arctan2(dy, -dx)
    aspect_rad = np.arctan2(dy, -dx)
    aspect_deg = np.degrees(aspect_rad)
    
    # Convert Aspect to 0-360 compass bearing (0 = North, 90 = East, etc.)
    aspect_deg = np.where(aspect_deg < 0, 360 + aspect_deg, aspect_deg)
    aspect_deg = np.where(aspect_deg == 90, 0, aspect_deg) # Handle flat areas if needed
    
    return slope_deg, aspect_deg, slope_rad, aspect_rad

def calculate_hillshade(slope_rad, aspect_rad, azimuth=315.0, zenith=45.0):
    # Convert sun angles to radians
    azimuth_rad = np.radians(360.0 - azimuth + 90.0)
    zenith_rad = np.radians(zenith)
    
    # The physics equation for Hillshade illumination
    shaded = (np.cos(zenith_rad) * np.cos(slope_rad) +
              np.sin(zenith_rad) * np.sin(slope_rad) * 
              np.cos(azimuth_rad - aspect_rad))
    
    # Scale to 0-255 (8-bit grayscale image)
    shaded = 255 * (shaded + 1) / 2
    return np.clip(shaded, 0, 255)

# ==========================================
# 3. Main Workflow
# ==========================================
def process_terrain():
    print("\n[INFO] Querying Copernicus Global 30m DEM from Planetary Computer...")
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
    
    search = catalog.search(collections=["cop-dem-glo-30"], bbox=BBOX)
    items = list(search.items())
    if not items:
        raise ValueError("No DEM found for this BBOX.")
        
    item = items[0]
    print(f"       [SUCCESS] Found DEM Item: {item.id}")
    
    print("\n[INFO] Streaming raw elevation data...")
    with rasterio.open(item.assets["data"].href) as src:
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        minx, miny = transformer.transform(BBOX[0], BBOX[1])
        maxx, maxy = transformer.transform(BBOX[2], BBOX[3])
        window = from_bounds(minx, miny, maxx, maxy, src.transform)
        
        dem = src.read(1, window=window).astype('float32')
        
        # Save profile for TIFF export
        profile = src.profile
        profile.update(
            dtype=rasterio.float32, count=1, nodata=np.nan,
            height=window.height, width=window.width,
            transform=rasterio.windows.transform(window, src.transform)
        )
        
    # Mask out NoData/Ocean (Elevation <= 0)
    dem = np.where(dem <= 0, np.nan, dem)

    print("\n[INFO] Calculating Terrain Derivatives (Slope & Aspect)...")
    slope_deg, aspect_deg, slope_rad, aspect_rad = calculate_slope_aspect(dem, cellsize=30.0)
    
    print("[INFO] Generating Hillshade (Azimuth 315, Zenith 45)...")
    hillshade = calculate_hillshade(slope_rad, aspect_rad)

    print("\n[INFO] Exporting Geocoded TIFFs for ArcGIS/ENVI...")
    def save_tif(data, name):
        out_tif = os.path.join(OUT_DIR, f"{name}.tif")
        with rasterio.open(out_tif, 'w', **profile) as dst:
            dst.write(data.astype('float32'), 1)
        print(f"       [SUCCESS] Exported TIFF: {out_tif}")
        
    save_tif(dem, "copernicus_dem")
    save_tif(slope_deg, "slope_degrees")
    save_tif(aspect_deg, "aspect_degrees")
    save_tif(hillshade, "hillshade")

    print("\n[INFO] Generating plot...")
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Raw DEM
    im1 = axs[0].imshow(dem, cmap='terrain')
    axs[0].set_title('Raw Elevation (m)')
    fig.colorbar(im1, ax=axs[0], shrink=0.7)
    axs[0].axis('off')
    
    # Slope
    im2 = axs[1].imshow(slope_deg, cmap='magma')
    axs[1].set_title('Slope (Degrees)')
    fig.colorbar(im2, ax=axs[1], shrink=0.7)
    axs[1].axis('off')
    
    # Hillshade
    im3 = axs[2].imshow(hillshade, cmap='gray')
    axs[2].set_title('Hillshade Relief')
    axs[2].axis('off')
    
    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, "terrain_derivatives.png")
    plt.savefig(plot_path, dpi=300)
    print(f"       [SUCCESS] Terrain map saved to: {plot_path}")

def main():
    print("=======================================================")
    print(" GEOCASCADE PIPELINE - DIGITAL ELEVATION PROCESSING    ")
    print("=======================================================")
    process_terrain()
    print("\n[SUCCESS] Script 10 Complete!")

if __name__ == "__main__":
    main()
