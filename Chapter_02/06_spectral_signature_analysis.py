"""
Chapter 2: 06_spectral_signature_analysis.py

Academic Objective:
Every material on Earth absorbs and reflects light differently across the 
electromagnetic spectrum. This unique pattern is called a "Spectral Signature."

In this script, we demonstrate Cloud-Native Geospatial processing. We do NOT 
download the massive satellite image. Instead, we query the Planetary Computer 
STAC API to find the Cloud Optimized GeoTIFF (COG) URLs. We then use Rasterio 
to stream *only the specific pixels* we need directly from the Microsoft cloud!

We will extract the spectral signatures for 3 materials in Patagonia:
1. Glacial Ice
2. Dense Forest
3. Bare Rock

Dependencies:
mamba install -n geocascade_env -c conda-forge pystac-client planetary-computer rasterio pyproj matplotlib -y
"""

import os
import rasterio
from pyproj import Transformer
import matplotlib.pyplot as plt
from pystac_client import Client
import planetary_computer as pc

# ==========================================
# 1. Configuration & Target Materials
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

# Sentinel-2 Core Bands and their Central Wavelengths (in nanometers)
BANDS = {
    "B02": 490,  # Blue
    "B03": 560,  # Green
    "B04": 665,  # Red
    "B08": 842,  # Near-Infrared (NIR)
    "B11": 1610, # Shortwave-Infrared 1 (SWIR1)
    "B12": 2190  # Shortwave-Infrared 2 (SWIR2)
}

# Real-world Coordinates (Lat, Lon) in Torres del Paine
MATERIALS = {
    "Glacial Ice (Grey Glacier)": {"lat": -51.010, "lon": -73.230, "color": "cyan", "reflectance": []},
    "Patagonian Forest":          {"lat": -51.150, "lon": -72.950, "color": "green", "reflectance": []},
    "Bare Rock / Scree":          {"lat": -50.900, "lon": -72.900, "color": "gray", "reflectance": []}
}

DATE_RANGE = "2023-01-01/2023-02-28"

# ==========================================
# 2. STAC API Query (Find the Image)
# ==========================================
def find_sentinel_image():
    print("\n[INFO] Connecting to Microsoft Planetary Computer STAC API...")
    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace
    )

    # Search around the Glacier coordinate
    bbox = [-73.25, -51.20, -72.85, -50.85]
    
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=DATE_RANGE,
        query={"eo:cloud_cover": {"lt": 5}}
    )
    
    items = list(search.items())
    if not items:
        raise ValueError("No cloud-free Sentinel-2 images found in this date range.")
        
    best_item = items[0]
    print(f"       [SUCCESS] Found Cloud-Free Image: {best_item.id}")
    return best_item

# ==========================================
# 3. Cloud-Native Pixel Extraction (No Download!)
# ==========================================
def extract_spectral_signatures(item):
    print("\n[INFO] Streaming pixel data directly from Cloud Optimized GeoTIFFs (COGs)...")
    
    transformer = None

    for band_name, wavelength in BANDS.items():
        print(f"       Extracting Band {band_name} ({wavelength} nm)...")
        # Get the direct Microsoft Cloud URL for this specific band
        cog_url = item.assets[band_name].href
        
        # Open the file over the internet (Cloud-Native!)
        with rasterio.open(cog_url) as src:
            # Dynamically pull the EPSG code from the actual GeoTIFF metadata
            if transformer is None:
                epsg = src.crs.to_epsg()
                transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
                
            for mat_name, mat_data in MATERIALS.items():
                # Convert Lat/Lon to UTM X/Y
                x, y = transformer.transform(mat_data["lon"], mat_data["lat"])
                
                # Sample the exact pixel at this coordinate
                val = next(src.sample([(x, y)]))[0]
                
                # Sentinel-2 L2A Reflectance is scaled by 10000. 
                # Divide by 10000 to get true reflectance (0.0 to 1.0)
                true_reflectance = val / 10000.0
                mat_data["reflectance"].append(true_reflectance)

# ==========================================
# 4. Plot the Signatures
# ==========================================
def plot_signatures():
    print("\n[INFO] Generating Spectral Signature Graph...")
    plt.figure(figsize=(12, 6))
    
    wavelengths = list(BANDS.values())
    
    for mat_name, mat_data in MATERIALS.items():
        plt.plot(wavelengths, mat_data["reflectance"], marker='o', linewidth=3, 
                 markersize=8, color=mat_data["color"], label=mat_name)
        
    plt.title("Spectral Signatures of Earth Materials (Sentinel-2 L2A)", fontsize=16)
    plt.xlabel("Wavelength (nm)", fontsize=14)
    plt.ylabel("Surface Reflectance", fontsize=14)
    
    # Label the X-axis with the Band Names
    plt.xticks(wavelengths, [f"{w}nm\n({b})" for b, w in BANDS.items()])
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.ylim(0, 1.0)
    
    plot_path = os.path.join(OUT_DIR, "spectral_signatures.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"       [SUCCESS] Graph saved to: {plot_path}")

def main():
    print("=======================================================")
    print(" GEOCASCADE PIPELINE - SPECTRAL SIGNATURE EXTRACTION   ")
    print("=======================================================")
    
    try:
        item = find_sentinel_image()
        extract_spectral_signatures(item)
        plot_signatures()
        print("\n[SUCCESS] Spectral Analysis Complete!")
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")

if __name__ == "__main__":
    main()
