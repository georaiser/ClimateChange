"""
Chapter 5: 14_moisture_stress_indices.py

Academic Objective:
While NDVI shows the greenness of vegetation, it doesn't tell us directly if the plants 
are suffering from drought. Water strongly absorbs Shortwave Infrared (SWIR) light.
By comparing Near Infrared (NIR, which reflects off healthy leaves) with SWIR (which is 
absorbed by water), we can calculate the exact moisture content of the soil and canopy.

In this script, we calculate:
1. NDMI (Normalized Difference Moisture Index) - Higher = Wetter
2. MSI (Moisture Stress Index) - Higher = Higher Drought Stress

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
from pyproj import Transformer

# ==========================================
# 1. Configuration
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

# Torres del Paine BBOX
BBOX = [-73.30, -51.10, -72.90, -50.80]
DATE_RANGE = "2023-01-01/2023-02-28"

def calculate_ndmi(nir, swir1):
    # NDMI = (NIR - SWIR1) / (NIR + SWIR1)
    return np.where((nir + swir1) == 0, np.nan, (nir - swir1) / (nir + swir1))

def calculate_msi(swir1, nir):
    # MSI = SWIR1 / NIR
    return np.where(nir == 0, np.nan, swir1 / nir)

def process_moisture():
    print("\n[INFO] Connecting to Planetary Computer STAC API...")
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)

    search = catalog.search(collections=["sentinel-2-l2a"], bbox=BBOX, datetime=DATE_RANGE, query={"eo:cloud_cover": {"lt": 5}})
    items = list(search.items())
    if not items:
        raise ValueError("No cloud-free images found.")
        
    item = items[0]
    print(f"       [SUCCESS] Found Image: {item.id}")
    
    print("\n[INFO] Streaming BBOX pixel data directly from Microsoft Cloud...")
    
    # Open NIR (B08) to establish window and profile
    with rasterio.open(item.assets["B08"].href) as src_nir:
        transformer = Transformer.from_crs("EPSG:4326", src_nir.crs, always_xy=True)
        minx, miny = transformer.transform(BBOX[0], BBOX[1])
        maxx, maxy = transformer.transform(BBOX[2], BBOX[3])
        window = from_bounds(minx, miny, maxx, maxy, src_nir.transform)
        transform = rasterio.windows.transform(window, src_nir.transform)
        
        profile = src_nir.profile
        profile.update(
            dtype=rasterio.float32, count=1, nodata=-9999,
            height=int(round(window.height)), width=int(round(window.width)),
            transform=transform
        )
        
        print("       Reading NIR (B08)...")
        nir = src_nir.read(1, window=window).astype('float32') / 10000.0

    print("       Reading SWIR1 (B11) and resampling to 10m...")
    # NOTE: B11 (SWIR) is 20m native resolution. We MUST pass window= and out_shape=
    # to resample it to the 10m NIR grid. The window here uses the 20m transform,
    # but the out_shape forces the output to match nir.shape (10m pixels).
    with rasterio.open(item.assets["B11"].href) as src_swir:
        transformer_swir = Transformer.from_crs("EPSG:4326", src_swir.crs, always_xy=True)
        sminx, sminy = transformer_swir.transform(BBOX[0], BBOX[1])
        smaxx, smaxy = transformer_swir.transform(BBOX[2], BBOX[3])
        win_swir = from_bounds(sminx, sminy, smaxx, smaxy, src_swir.transform)
        swir1 = src_swir.read(
            1, window=win_swir, out_shape=nir.shape,
            resampling=rasterio.enums.Resampling.bilinear
        ).astype('float32') / 10000.0

    print("\n[INFO] Calculating Moisture Indices...")
    ndmi = calculate_ndmi(nir, swir1)
    msi = calculate_msi(swir1, nir)

    print("\n[INFO] Exporting Geocoded TIFFs for ArcGIS/ENVI...")
    def save_tif(data, name):
        out_tif = os.path.join(OUT_DIR, f"{name}.tif")
        with rasterio.open(out_tif, 'w', **profile) as dst:
            dst.write(data, 1)
        print(f"       [SUCCESS] Exported {name.upper()} TIFF: {out_tif}")
        
    save_tif(ndmi, "ndmi")
    save_tif(msi, "msi")

    print("\n[INFO] Generating comparison plots...")
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    
    # NDMI (Moisture)
    im1 = axs[0].imshow(ndmi, cmap='RdYlBu', vmin=-0.2, vmax=0.6)
    axs[0].set_title('NDMI (Normalized Difference Moisture Index)\nBlue = High Moisture')
    fig.colorbar(im1, ax=axs[0], shrink=0.7)
    axs[0].axis('off')
    
    # MSI (Drought Stress)
    # Filter out extreme MSI values for better visualization (MSI > 3 is usually non-vegetated)
    msi_viz = np.where(msi > 3, np.nan, msi)
    im2 = axs[1].imshow(msi_viz, cmap='YlOrRd', vmin=0.4, vmax=2.0)
    axs[1].set_title('MSI (Moisture Stress Index)\nRed = High Drought Stress')
    fig.colorbar(im2, ax=axs[1], shrink=0.7)
    axs[1].axis('off')
    
    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, "moisture_stress_indices.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)  # prevent memory leak in multi-script pipelines
    print(f"       [SUCCESS] Plot saved to: {plot_path}")

def main():
    print("=======================================================")
    print(" GEOCASCADE PIPELINE - MOISTURE & DROUGHT STRESS       ")
    print("=======================================================")
    try:
        process_moisture()
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")

if __name__ == "__main__":
    main()
