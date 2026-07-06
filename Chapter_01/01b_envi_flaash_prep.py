"""
Chapter 1: 01b_envi_flaash_prep.py

Academic Objective:
This utility script automates the ENVI preparation phase entirely using Python.
Critically, it performs **Radiometric Calibration**. Our STAC pipeline downloads 
Top of Atmosphere (TOA) Reflectance. However, ENVI's FLAASH model strictly requires 
TOA Radiance (absolute light energy). 

This script queries the STAC API for the Solar Zenith angle, calculates the Earth-Sun 
distance based on the acquisition date, and uses the Sentinel-2 ESUN constants to 
mathematically convert the arrays into Radiance.

Dependencies:
mamba install -n geocascade_env -c conda-forge rasterio numpy pystac-client planetary-computer
"""

import os
import glob
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
import numpy as np
import math
import datetime
from pystac_client import Client
import planetary_computer as pc

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "data", "raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed_envi")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def find_latest_sentinel_dir():
    s2_dirs = glob.glob(os.path.join(INPUT_DIR, "sentinel2_*"))
    if not s2_dirs:
        raise FileNotFoundError("No Sentinel-2 raw data found. Run script 01 first.")
    return max(s2_dirs, key=os.path.getmtime)

def fetch_stac_metadata(item_id):
    print(f"[INFO] Step 0: Fetching metadata for {item_id} from STAC API...")
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
    search = catalog.search(collections=["sentinel-2-l2a"], ids=[item_id])
    items = list(search.items())
    if not items:
        print("[WARNING] Could not fetch metadata. Using default Patagonian summer values.")
        return {"s2:mean_solar_zenith": 45.0, "datetime": "2023-01-15T14:30:00Z"}
    return items[0].properties

def main():
    print("==================================================")
    print(" GEOCASCADE - ENVI FLAASH PREP (RADIOMETRIC CALIB)")
    print("==================================================")
    
    try:
        s2_dir = find_latest_sentinel_dir()
        scene_name = os.path.basename(s2_dir)
        item_id = scene_name.replace("sentinel2_", "")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return

    # Fetch Solar Zenith and Date for Radiance calculation
    props = fetch_stac_metadata(item_id)
    zenith_deg = props.get("s2:mean_solar_zenith", 45.0)
    zenith_rad = math.radians(zenith_deg)
    
    date_str = props.get("datetime")
    dt = datetime.datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    doy = dt.timetuple().tm_yday
    
    # Earth-Sun distance (d) in AU
    d = 1.0 - 0.01672 * math.cos(math.radians(0.9856 * (doy - 4)))
    d2 = d * d
    
    print(f"       -> Solar Zenith Angle: {zenith_deg:.2f}°")
    print(f"       -> Earth-Sun Distance (d): {d:.4f} AU")

    band_names = ["B02", "B03", "B04", "B08", "B11"]
    
    # ESUN constants for Sentinel-2 in W/(m2*um)
    esun_dict = {"B02": 1959.72, "B03": 1824.93, "B04": 1904.09, "B08": 1042.51, "B11": 245.59}
    
    band_paths = {b: glob.glob(os.path.join(s2_dir, f"{b}.tif"))[0] for b in band_names}
            
    print("\n[INFO] Step 1: Reading 10m Master Profile (B02)...")
    with rasterio.open(band_paths["B02"]) as src_b2:
        meta = src_b2.meta.copy()
        target_transform = src_b2.transform
        target_crs = src_b2.crs
        target_width = src_b2.width
        target_height = src_b2.height
        
    print("[INFO] Step 2: Preparing ENVI BIL Output Stack (Converting to Radiance)...")
    
    # We save as int16, scaled by 10 for FLAASH. (So user enters 10 for Scale Factor)
    meta.update({'driver': 'ENVI', 'count': 5, 'interleave': 'bil', 'dtype': rasterio.int16, 'nodata': 0})
    out_file = os.path.join(OUTPUT_DIR, f"flaash_radiance_stack_{item_id}.dat")
    
    def convert_to_radiance(reflectance_array, esun):
        # 1. Convert L1C digital number to True Reflectance (0.0 - 1.0)
        rho = reflectance_array.astype(np.float32) / 10000.0
        # 2. Convert Reflectance to Radiance in W/(m2*sr*um)
        radiance_W = (rho * esun * math.cos(zenith_rad)) / (math.pi * d2)
        # 3. Convert to FLAASH units: µW/(cm2*sr*nm)
        radiance_flaash = radiance_W * 0.1
        # 4. Scale by 10 and convert to int16 to save disk space
        radiance_scaled = radiance_flaash * 10.0
        mask = reflectance_array == 0
        radiance_scaled[mask] = 0
        return np.clip(radiance_scaled, 0, 32767).astype(np.int16)
        
    with rasterio.open(out_file, 'w', **meta) as dst:
        for i, b in enumerate(["B02", "B03", "B04", "B08"], start=1):
            print(f"       -> Calibrating & Stacking {b} (Native 10m)...")
            with rasterio.open(band_paths[b]) as src:
                rad_data = convert_to_radiance(src.read(1), esun_dict[b])
                dst.write(rad_data, i)
                
        print("       -> Resampling, Calibrating & Stacking B11 (20m -> 10m)...")
        with rasterio.open(band_paths["B11"]) as src_b11:
            b11_resampled = np.empty((target_height, target_width), dtype=rasterio.uint16)
            reproject(
                source=src_b11.read(1), destination=b11_resampled,
                src_transform=src_b11.transform, src_crs=src_b11.crs,
                dst_transform=target_transform, dst_crs=target_crs, resampling=Resampling.bilinear
            )
            dst.write(convert_to_radiance(b11_resampled, esun_dict["B11"]), 5)
            
    print("[INFO] Step 3: Injecting Sensor Metadata into ENVI Header...")
    hdr_file = out_file.replace('.dat', '.hdr')
    with open(hdr_file, 'a') as hdr:
        hdr.write("wavelength units = Nanometers\n")
        hdr.write("sensor type = Sentinel-2\n")
        hdr.write("data ignore value = 0\n")
        hdr.write("wavelength = {490.0, 560.0, 665.0, 842.0, 1610.0}\n")
        hdr.write("fwhm = {65.0, 35.0, 30.0, 115.0, 90.0}\n")
            
    print("\n[SUCCESS] Radiometrically Calibrated ENVI Stack Created!")
    print(f"[OUTPUT] File: {out_file}")
    print("[CRITICAL] When FLAASH asks for the Radiance Scale Factor, enter 10 (NOT 10000)!")

if __name__ == "__main__":
    main()
