"""
Chapter 1: 02_atmospheric_correction.py

Academic Objective:
This script demonstrates the COST Model (Chavez, 1996), a highly advanced empirical 
atmospheric correction algorithm. It improves upon standard Dark Object Subtraction (DOS1)
by not only subtracting atmospheric haze (path radiance), but also estimating 
atmospheric absorption (transmittance) using the cosine of the Solar Zenith Angle.

This makes the Python empirical model behave much closer to rigorous physical 
models like ENVI FLAASH, accurately restoring the brightness of highly reflective surfaces.

Dependencies:
mamba install -n geocascade_env -c conda-forge rasterio numpy pystac-client planetary-computer
"""

import os
import glob
import shutil          # moved from inside loop — top-level import is best practice
import rasterio
import numpy as np
import math
from pystac_client import Client
import planetary_computer as pc

# ==========================================
# 1. Configuration Paths
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "data", "raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed", "boa_corrected")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def find_latest_sentinel_dir():
    # IMPORTANT: This script expects Sentinel-2 L1C (Top-of-Atmosphere) raw bands.
    # Script 01_stac_multisensor_download.py downloads L2A (already atmospherically corrected).
    # NEVER apply COST correction to L2A data — it would double-correct and produce wrong results.
    #
    # To use this script:
    #   1. Download L1C data manually from https://scihub.copernicus.eu
    #   2. Place it in data/raw/ with the naming convention sentinel2_l1c_<scene_id>/
    s2_dirs = glob.glob(os.path.join(INPUT_DIR, "sentinel2_l1c_*"))
    if not s2_dirs:
        raise FileNotFoundError(
            "No Sentinel-2 L1C raw data found in data/raw/sentinel2_l1c_*/\n"
            "Script 01 downloads L2A (already corrected). This script needs L1C.\n"
            "Download L1C manually from: https://scihub.copernicus.eu"
        )
    return max(s2_dirs, key=os.path.getmtime)

def fetch_stac_metadata(item_id):
    print(f"[INFO] Fetching Solar Zenith for {item_id} from STAC API...")
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
    search = catalog.search(collections=["sentinel-2-l2a"], ids=[item_id])
    items = list(search.items())
    if not items:
        return {"s2:mean_solar_zenith": 45.0}
    return items[0].properties

# ==========================================
# 2. COST Model Correction (Improved DOS)
# ==========================================
def apply_cost_correction(input_raster, output_raster, zenith_deg):
    """
    Apply COST Model (Chavez, 1996) atmospheric correction to a single Sentinel-2 band.

    IMPORTANT — Scale Factor:
    Sentinel-2 L1C DN values are scaled by 10,000. This script operates on raw DN
    (0–65535 range) and leaves the output in the same scaled DN units after correction.
    For true physical reflectance [0–1], divide output by 10,000 downstream.
    The COST transmittance division here corrects for atmospheric absorption but does
    NOT normalize to physical reflectance units.
    """
    with rasterio.open(input_raster) as src:
        meta = src.meta.copy()
        band_data = src.read(1, masked=True)

        # 1. Dark Object Subtraction (Chavez 1988): 1st percentile ≈ 1% reflectance = 100 DN
        valid_pixels = band_data.compressed()
        if len(valid_pixels) == 0:
            dark_pixel = 0
        else:
            dark_pixel = np.percentile(valid_pixels, 1)

        haze = max(0, dark_pixel - 100)

        band_name = os.path.basename(input_raster)
        print(f"       [{band_name}] Dark Pixel: {dark_pixel:.0f} | Haze Subtracted: {haze:.0f}")

        # 2. Subtract path radiance
        corrected_data = band_data - haze

        # 3. COST Transmittance: T = cos(Solar Zenith)
        transmittance = math.cos(math.radians(zenith_deg))
        corrected_data = corrected_data / transmittance

        # 4. Clip and write
        corrected_data = np.clip(corrected_data, 0, 65535)

        meta.update(dtype=rasterio.uint16, count=1, compress='lzw')
        with rasterio.open(output_raster, 'w', **meta) as dst:
            dst.write(corrected_data.filled(0).astype(rasterio.uint16), 1)
            
    print(f"       [{band_name}] ✔️ COST Correction Applied (Transmittance: {transmittance:.3f})")


# ==========================================
# 3. Main Execution Block
# ==========================================
def main():
    print("==================================================")
    print(" GEOCASCADE PIPELINE - ATMOSPHERIC CORRECTION (COST)")
    print("==================================================")
    
    try:
        s2_dir = find_latest_sentinel_dir()
        scene_name = os.path.basename(s2_dir)
        item_id = scene_name.replace("sentinel2_", "")
        print(f"[INFO] Found raw Sentinel-2 directory: {scene_name}")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return

    tif_files = glob.glob(os.path.join(s2_dir, "*.tif"))
    if not tif_files:
        print("[ERROR] No .tif files found in the Sentinel-2 directory.")
        return
        
    # Fetch Solar Zenith Angle for the COST model
    props = fetch_stac_metadata(item_id)
    zenith_deg = props.get("s2:mean_solar_zenith", 45.0)
    print(f"       -> Mean Solar Zenith Angle: {zenith_deg:.2f}°")
        
    print(f"\n[INFO] Applying COST Model (Improved DOS) to {len(tif_files)} bands...")
    
    scene_out_dir = os.path.join(OUTPUT_DIR, scene_name)
    os.makedirs(scene_out_dir, exist_ok=True)
    
    for tif in tif_files:
        band_filename = os.path.basename(tif)
        out_tif = os.path.join(scene_out_dir, f"BOA_{band_filename}")
        
        # IMPORTANT PHYSICS NOTE:
        # Haze correction is inappropriate for SWIR (B11, B12), Water Vapor (B09), and Cirrus (B10).
        bands_to_skip = ["B09", "B10", "B11", "B12"]
        skip_band = any(skip_str in band_filename for skip_str in bands_to_skip)
        
        if skip_band:
            print(f"       [{band_filename}] ⏭️ Skipped COST correction. Physically inappropriate for this wavelength.")
            shutil.copy(tif, out_tif)   # shutil imported at top of file
            continue
            
        apply_cost_correction(tif, out_tif, zenith_deg)
        
    print("\n[SUCCESS] Chapter 1 Atmospheric Correction complete.")
    print(f"[OUTPUT] Corrected data saved to: {scene_out_dir}")

if __name__ == "__main__":
    main()
