"""
Chapter 2: 07_vegetation_soil_indices.py

Academic Objective:
Now that we understand Spectral Signatures (Script 06), we can use mathematical 
ratios between specific bands to isolate and highlight environmental features.

In this script, we calculate 4 critical environmental indices:
1. NDVI (Normalized Difference Vegetation Index) - Standard vegetation health.
2. EVI (Enhanced Vegetation Index) - Corrects for atmospheric noise and canopy background.
3. SAVI (Soil Adjusted Vegetation Index) - Corrects for soil brightness in sparse areas.
4. BSI (Bare Soil Index) - Highlights bare rock/soil to assess erosion risk.

We will use Cloud-Native windowed reading to extract only the pixels covering 
our specific bounding box, preventing the need to download the entire 100x100km tile.

Dependencies:
mamba install -n geocascade_env -c conda-forge pystac-client planetary-computer rasterio pyproj matplotlib numpy -y
"""

import os
import rasterio
from rasterio.windows import from_bounds
import numpy as np
import matplotlib.pyplot as plt
from pystac_client import Client
import planetary_computer as pc

# ==========================================
# 1. Configuration
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

# Torres del Paine BBOX
BBOX = [-73.30, -51.10, -72.90, -50.80]
DATE_RANGE = "2023-01-01/2023-02-28"

# ==========================================
# 2. Mathematical Index Functions
# ==========================================
def calculate_ndvi(nir, red):
    # (NIR - Red) / (NIR + Red)
    return np.where((nir + red) == 0, 0, (nir - red) / (nir + red))

def calculate_evi(nir, red, blue):
    # 2.5 * ((NIR - Red) / (NIR + 6 * Red - 7.5 * Blue + 1))
    denominator = (nir + 6.0 * red - 7.5 * blue + 1.0)
    return np.where(denominator == 0, 0, 2.5 * ((nir - red) / denominator))

def calculate_savi(nir, red, L=0.5):
    # ((NIR - Red) / (NIR + Red + L)) * (1 + L)
    denominator = (nir + red + L)
    return np.where(denominator == 0, 0, ((nir - red) / denominator) * (1.0 + L))

def calculate_bsi(swir1, red, nir, blue):
    # ((SWIR1 + Red) - (NIR + Blue)) / ((SWIR1 + Red) + (NIR + Blue))
    numerator = (swir1 + red) - (nir + blue)
    denominator = (swir1 + red) + (nir + blue)
    return np.where(denominator == 0, 0, numerator / denominator)

# ==========================================
# 3. Cloud-Native Processing
# ==========================================
def process_indices():
    print("\n[INFO] Connecting to Planetary Computer STAC API...")
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)

    search = catalog.search(collections=["sentinel-2-l2a"], bbox=BBOX, datetime=DATE_RANGE, query={"eo:cloud_cover": {"lt": 5}})
    items = list(search.items())
    if not items:
        raise ValueError("No cloud-free images found.")
        
    item = items[0]
    print(f"       [SUCCESS] Found Image: {item.id}")
    
    print("\n[INFO] Streaming BBOX pixel data directly from Microsoft Cloud...")
    
    # We open B02 (Blue) first to calculate the exact pixel window for our BBOX
    b02_url = item.assets["B02"].href
    with rasterio.open(b02_url) as src:
        # Transform our Lat/Lon BBOX into the native UTM projection of the image
        from pyproj import Transformer
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        minx, miny = transformer.transform(BBOX[0], BBOX[1])
        maxx, maxy = transformer.transform(BBOX[2], BBOX[3])
        
        # Calculate the exact pixel Window for our BBOX
        window = from_bounds(minx, miny, maxx, maxy, src.transform)
        transform = rasterio.windows.transform(window, src.transform)
        
        # Save profile for TIFF export
        profile = src.profile
        profile.update(
            dtype=rasterio.float32, count=1, nodata=np.nan,
            height=window.height, width=window.width,
            transform=transform
        )
        
        # We need Blue, Green, Red, NIR, and SWIR1 for these 4 indices
        print("       Reading Blue (B02)...")
        blue = src.read(1, window=window).astype('float32') / 10000.0

    print("       Reading Red (B04)...")
    with rasterio.open(item.assets["B04"].href) as src:
        red = src.read(1, window=window).astype('float32') / 10000.0
        
    print("       Reading NIR (B08)...")
    with rasterio.open(item.assets["B08"].href) as src:
        nir = src.read(1, window=window).astype('float32') / 10000.0
        
    print("       Reading SWIR1 (B11) (Note: Resampling 20m to 10m on the fly)...")
    with rasterio.open(item.assets["B11"].href) as src:
        # B11 is 20m resolution. We use rasterio's out_shape to resample it to match our 10m bands!
        swir1 = src.read(1, window=window, out_shape=blue.shape, resampling=rasterio.enums.Resampling.bilinear).astype('float32') / 10000.0

    print("\n[INFO] Calculating Spectral Indices...")
    ndvi = calculate_ndvi(nir, red)
    evi = calculate_evi(nir, red, blue)
    savi = calculate_savi(nir, red)
    bsi = calculate_bsi(swir1, red, nir, blue)

    print("\n[INFO] Exporting Geocoded TIFFs for ArcGIS/ENVI...")
    def save_tif(data, name):
        out_tif = os.path.join(OUT_DIR, f"{name}.tif")
        with rasterio.open(out_tif, 'w', **profile) as dst:
            dst.write(data, 1)
        print(f"       [SUCCESS] Exported {name.upper()} TIFF: {out_tif}")
        
    save_tif(ndvi, "ndvi")
    save_tif(evi, "evi")
    save_tif(savi, "savi")
    save_tif(bsi, "bsi")

    print("\n[INFO] Generating comparison plots...")
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # NDVI
    im1 = axs[0, 0].imshow(ndvi, cmap='RdYlGn', vmin=-0.2, vmax=0.8)
    axs[0, 0].set_title('NDVI (Vegetation Health)')
    fig.colorbar(im1, ax=axs[0, 0])
    axs[0, 0].axis('off')
    
    # EVI
    im2 = axs[0, 1].imshow(evi, cmap='RdYlGn', vmin=-0.2, vmax=0.8)
    axs[0, 1].set_title('EVI (Enhanced Vegetation)')
    fig.colorbar(im2, ax=axs[0, 1])
    axs[0, 1].axis('off')
    
    # SAVI
    im3 = axs[1, 0].imshow(savi, cmap='RdYlGn', vmin=-0.2, vmax=0.8)
    axs[1, 0].set_title('SAVI (Soil-Adjusted Vegetation)')
    fig.colorbar(im3, ax=axs[1, 0])
    axs[1, 0].axis('off')
    
    # BSI
    im4 = axs[1, 1].imshow(bsi, cmap='copper', vmin=-0.5, vmax=0.5)
    axs[1, 1].set_title('BSI (Bare Soil Erosion Risk)')
    fig.colorbar(im4, ax=axs[1, 1])
    axs[1, 1].axis('off')
    
    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, "vegetation_soil_indices.png")
    plt.savefig(plot_path, dpi=300)
    print(f"       [SUCCESS] Plot saved to: {plot_path}")

def main():
    print("=======================================================")
    print(" GEOCASCADE PIPELINE - SPECTRAL INDICES AUTOMATION     ")
    print("=======================================================")
    process_indices()
    print("\n[SUCCESS] Script 07 Complete!")

if __name__ == "__main__":
    main()
