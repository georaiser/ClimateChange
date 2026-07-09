"""
Chapter 2: 07_vegetation_soil_indices.py

Academic Objective:
Now that we understand Spectral Signatures (Script 06), we can use mathematical 
ratios between specific bands to isolate and highlight environmental features.

In this script, we calculate 7 critical environmental indices covering the full
curriculum from the original syllabus:
  VEGETATION:   NDVI, EVI, SAVI
  SOIL/BARE:    BSI
  WATER/SNOW:   NDWI (water bodies), NDSI (snow/ice), NDGI (glacier extent)

Key concept: Each index isolates a specific physical signal by exploiting the
different reflectance properties of materials across the electromagnetic spectrum.

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
def safe_ratio(num, den):
    """Generic NaN-safe ratio for all spectral indices."""
    return np.where(np.abs(den) < 1e-6, np.nan, num / den)

def calculate_ndvi(nir, red):
    """NDVI = (NIR - Red) / (NIR + Red)  →  Range [-1, 1], high = dense vegetation."""
    return safe_ratio(nir - red, nir + red)

def calculate_evi(nir, red, blue):
    """EVI corrects for atmospheric effects and canopy background noise.
    Formula: 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)"""
    return safe_ratio(2.5 * (nir - red), nir + 6.0 * red - 7.5 * blue + 1.0)

def calculate_savi(nir, red, L=0.5):
    """SAVI corrects NDVI for soil brightness in sparse-canopy regions.
    L=0.5 is the standard correction factor for intermediate cover."""
    return safe_ratio((nir - red) * (1.0 + L), nir + red + L)

def calculate_bsi(swir1, red, nir, blue):
    """BSI highlights bare soil and rock; high values = erosion risk.
    Formula: ((SWIR1+Red) - (NIR+Blue)) / ((SWIR1+Red) + (NIR+Blue))"""
    return safe_ratio((swir1 + red) - (nir + blue), (swir1 + red) + (nir + blue))

def calculate_ndwi(green, nir):
    """NDWI = (Green - NIR) / (Green + NIR)  →  Positive values = open water bodies.
    Use to detect lakes, rivers, and flooded areas. Threshold > 0.3 for water mask."""
    return safe_ratio(green - nir, green + nir)

def calculate_ndsi(green, swir1):
    """NDSI = (Green - SWIR1) / (Green + SWIR1)  →  Positive values = snow or ice.
    Threshold > 0.4 isolates permanent snow cover and glacier extent."""
    return safe_ratio(green - swir1, green + swir1)

def calculate_ndgi(green, red):
    """NDGI = (Green - Red) / (Green + Red)  →  Glacier/green ice index.
    Discriminates turbid glacial ice from rock and sediment at terminus."""
    return safe_ratio(green - red, green + red)

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
        
    item = sorted(items, key=lambda i: i.properties["eo:cloud_cover"])[0]
    print(f"       [SUCCESS] Found Image: {item.id}  (Cloud: {item.properties['eo:cloud_cover']}%)")
    
    print("\n[INFO] Streaming BBOX pixel data directly from Microsoft Cloud...")
    
    from pyproj import Transformer
    # Open B02 (Blue, 10m) first to establish the master 10m pixel window
    with rasterio.open(item.assets["B02"].href) as src:
        transformer_10m = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        minx, miny = transformer_10m.transform(BBOX[0], BBOX[1])
        maxx, maxy = transformer_10m.transform(BBOX[2], BBOX[3])
        window = from_bounds(minx, miny, maxx, maxy, src.transform)
        transform = rasterio.windows.transform(window, src.transform)
        target_shape = (int(round(window.height)), int(round(window.width)))
        profile = src.profile
        # nodata=-9999 for ArcGIS/ENVI/GDAL compatibility (np.nan is unreliable)
        profile.update(
            dtype=rasterio.float32, count=1, nodata=-9999,
            height=target_shape[0], width=target_shape[1], transform=transform
        )
        print("       Reading Blue (B02, 10m)...")
        blue = src.read(1, window=window).astype('float32') / 10000.0

    print("       Reading Green (B03, 10m)...")
    with rasterio.open(item.assets["B03"].href) as src:
        green = src.read(1, window=window).astype('float32') / 10000.0

    print("       Reading Red (B04, 10m)...")
    with rasterio.open(item.assets["B04"].href) as src:
        red = src.read(1, window=window).astype('float32') / 10000.0

    print("       Reading NIR (B08, 10m)...")
    with rasterio.open(item.assets["B08"].href) as src:
        nir = src.read(1, window=window).astype('float32') / 10000.0

    # IMPORTANT: B11 is 20m native -- compute its OWN window from the 20m transform.
    # Reusing the 10m B02 window here would apply 10m pixel coordinates to a 20m grid,
    # silently reading the wrong geographic area. Use out_shape to resample to 10m.
    print("       Reading SWIR1 (B11, 20m -> resampled to 10m independently)...")
    with rasterio.open(item.assets["B11"].href) as src_b11:
        t_b11 = Transformer.from_crs("EPSG:4326", src_b11.crs, always_xy=True)
        minx_b11, miny_b11 = t_b11.transform(BBOX[0], BBOX[1])
        maxx_b11, maxy_b11 = t_b11.transform(BBOX[2], BBOX[3])
        win_b11 = from_bounds(minx_b11, miny_b11, maxx_b11, maxy_b11, src_b11.transform)
        swir1 = src_b11.read(
            1, window=win_b11, out_shape=target_shape,
            resampling=rasterio.enums.Resampling.bilinear
        ).astype('float32') / 10000.0

    print("\n[INFO] Calculating all 7 Spectral Indices...")
    ndvi = calculate_ndvi(nir, red)
    evi  = calculate_evi(nir, red, blue)
    savi = calculate_savi(nir, red)
    bsi  = calculate_bsi(swir1, red, nir, blue)
    ndwi = calculate_ndwi(green, nir)    # Water bodies
    ndsi = calculate_ndsi(green, swir1)  # Snow & ice
    ndgi = calculate_ndgi(green, red)    # Glacier extent

    print("\n[INFO] Exporting Geocoded TIFFs for ArcGIS/ENVI...")
    def save_tif(data, name):
        out_tif = os.path.join(OUT_DIR, f"{name}.tif")
        with rasterio.open(out_tif, 'w', **profile) as dst:
            dst.write(np.nan_to_num(data, nan=-9999).astype('float32'), 1)
        print(f"       [SUCCESS] Exported {name.upper()} → {out_tif}")
        
    for arr, name in [(ndvi,'ndvi'),(evi,'evi'),(savi,'savi'),(bsi,'bsi'),
                      (ndwi,'ndwi'),(ndsi,'ndsi'),(ndgi,'ndgi')]:
        save_tif(arr, name)

    print("\n[INFO] Generating 7-panel index comparison chart...")
    indices = [
        (ndvi, 'RdYlGn',  (-0.2, 0.8),  'NDVI — Vegetation Health'),
        (evi,  'RdYlGn',  (-0.2, 0.8),  'EVI — Enhanced Vegetation'),
        (savi, 'RdYlGn',  (-0.2, 0.8),  'SAVI — Soil-Adjusted Vegetation'),
        (bsi,  'copper',  (-0.5, 0.5),  'BSI — Bare Soil / Erosion Risk'),
        (ndwi, 'Blues',   (-0.5, 0.6),  'NDWI — Open Water Bodies'),
        (ndsi, 'cool',    (-0.3, 0.8),  'NDSI — Snow & Ice Extent'),
        (ndgi, 'PuBuGn',  (-0.3, 0.5),  'NDGI — Glacier Green Ice'),
    ]

    # Tier 3: print summary statistics for each index
    print("\n--- INDEX SUMMARY STATISTICS ---")
    for (data, _, _, title), name in zip(indices, ['NDVI','EVI','SAVI','BSI','NDWI','NDSI','NDGI']):
        valid = data[np.isfinite(data)]
        if valid.size > 0:
            print(f"  {name:6s}: mean={np.mean(valid):+.3f}  std={np.std(valid):.3f}"
                  f"  min={np.min(valid):+.3f}  max={np.max(valid):+.3f}")
    print("-" * 50)

    fig, axs = plt.subplots(2, 4, figsize=(22, 12))
    axs_flat = axs.flatten()
    for ax, (data, cmap, vrange, title) in zip(axs_flat, indices):
        im = ax.imshow(data, cmap=cmap, vmin=vrange[0], vmax=vrange[1])
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.75)
        ax.axis('off')
    axs_flat[-1].axis('off')  # hide the 8th empty panel
    fig.suptitle('Complete Spectral Index Suite — Torres del Paine', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, "spectral_indices_all.png")
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close(fig)  # prevent memory leak
    print(f"       [SUCCESS] 7-panel chart saved: {plot_path}")

def main():
    print("=======================================================")
    print(" GEOCASCADE PIPELINE - SPECTRAL INDICES AUTOMATION     ")
    print("=======================================================")
    process_indices()
    print("\n[SUCCESS] Script 07 Complete!")

if __name__ == "__main__":
    main()
