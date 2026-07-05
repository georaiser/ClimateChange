"""
Chapter 1: 02_atmospheric_correction.py

Academic Objective:
Sensors record Top of Atmosphere (TOA) radiance, which includes atmospheric 
scattering (haze). True surface analysis requires Bottom of Atmosphere (BOA) reflectance.
While modern STAC pipelines (like our Sentinel-2 L2A download) provide pre-corrected 
BOA data via Sen2Cor, understanding the physics of correction is mandatory.

This script demonstrates Dark Object Subtraction (DOS1), a classic, image-based 
atmospheric correction model. DOS1 assumes that within an image, there are pixels 
in complete shadow or deep water that should have zero reflectance. Any recorded 
signal above zero in these pixels is attributed to atmospheric path radiance (haze).
By subtracting this minimum value, we mathematically remove the haze constant.

Dependencies:
mamba install -n geocascade_env -c conda-forge rasterio numpy
"""

import os
import glob
import rasterio
import numpy as np

# ==========================================
# 1. Configuration Paths
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "data", "raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed", "boa_corrected")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def find_latest_sentinel_dir():
    """Finds the Sentinel-2 directory downloaded by script 01."""
    s2_dirs = glob.glob(os.path.join(INPUT_DIR, "sentinel2_*"))
    if not s2_dirs:
        raise FileNotFoundError("No Sentinel-2 raw data found. Run script 01 first.")
    # Return the most recently modified directory
    return max(s2_dirs, key=os.path.getmtime)


# ==========================================
# 2. Dark Object Subtraction (DOS1) Logic
# ==========================================
def apply_dos1_correction(input_raster, output_raster):
    """
    Applies Dark Object Subtraction to a single raster band.
    1. Reads the raster.
    2. Finds the minimum valid value (the "Dark Object").
    3. Subtracts this value from all pixels (clipping at 0 to avoid negatives).
    4. Saves the atmospherically corrected raster.
    """
    with rasterio.open(input_raster) as src:
        meta = src.meta.copy()
        
        # Read the first band as a numpy array
        # Note: We use masked=True to ignore nodata (usually 0 at the image borders)
        band_data = src.read(1, masked=True)
        
        # 1. Identify the Dark Object (Minimum value excluding NoData)
        # In optical remote sensing, deep water often acts as a dark object.
        dark_object_value = band_data.min()
        
        band_name = os.path.basename(input_raster)
        print(f"       [{band_name}] Dark Object Value (Haze): {dark_object_value}")
        
        # 2. Subtract the haze value
        # Subtracting haze mathematically corrects TOA to pseudo-BOA.
        corrected_data = band_data - dark_object_value
        
        # 3. Ensure no negative values (physical impossibility for reflectance)
        corrected_data = np.clip(corrected_data, 0, None)
        
        # 4. Write the corrected data to the new file
        # Update metadata to ensure we maintain the correct datatype and CRS
        meta.update(dtype=rasterio.uint16, count=1, compress='lzw')
        
        with rasterio.open(output_raster, 'w', **meta) as dst:
            # Write the filled array (putting nodata back where the mask is)
            dst.write(corrected_data.filled(0).astype(rasterio.uint16), 1)
            
    print(f"       [{band_name}] ✔️ DOS1 Correction Applied and Saved.")


# ==========================================
# 3. Main Execution Block
# ==========================================
def main():
    print("==================================================")
    print(" GEOCASCADE PIPELINE - ATMOSPHERIC CORRECTION (DOS1)")
    print("==================================================")
    
    try:
        s2_dir = find_latest_sentinel_dir()
        print(f"[INFO] Found raw Sentinel-2 directory: {os.path.basename(s2_dir)}")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return

    # Process all TIF files (B02, B03, B04, B08)
    tif_files = glob.glob(os.path.join(s2_dir, "*.tif"))
    
    if not tif_files:
        print("[ERROR] No .tif files found in the Sentinel-2 directory.")
        return
        
    print(f"\n[INFO] Applying Dark Object Subtraction (DOS1) to {len(tif_files)} bands...")
    
    # Create a specific output folder for this scene
    scene_name = os.path.basename(s2_dir)
    scene_out_dir = os.path.join(OUTPUT_DIR, scene_name)
    os.makedirs(scene_out_dir, exist_ok=True)
    
    for tif in tif_files:
        band_filename = os.path.basename(tif)
        out_tif = os.path.join(scene_out_dir, f"BOA_{band_filename}")
        
        # Execute the DOS1 correction on each band
        apply_dos1_correction(tif, out_tif)
        
    print("\n[SUCCESS] Chapter 1 Atmospheric Correction complete.")
    print(f"[OUTPUT] Corrected data saved to: {scene_out_dir}")

if __name__ == "__main__":
    main()
