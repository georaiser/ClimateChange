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
    """Find the most recently created L1C raw directory."""
    s2_dirs = glob.glob(os.path.join(INPUT_DIR, "sentinel2_l1c_*"))
    l2a_dirs = glob.glob(os.path.join(INPUT_DIR, "sentinel2_l2a_*"))
    if not s2_dirs:
        if l2a_dirs:
            raise FileNotFoundError(
                "Found Sentinel-2 L2A directory, but FLAASH requires L1C (TOA Radiance).\n"
                "Run script 01 with collection='sentinel-2-l1c' to download L1C data."
            )
        raise FileNotFoundError(
            "No Sentinel-2 L1C raw data found. Run script 01_stac_multisensor_download.py first."
        )
    return max(s2_dirs, key=os.path.getmtime)

def fetch_stac_metadata(item_id):
    print(f"[INFO] Step 0: Fetching metadata for {item_id} from STAC API...")
    try:
        catalog = Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=pc.sign_inplace
        )
        search = catalog.search(collections=["sentinel-2-l1c"], ids=[item_id])
        items  = list(search.items())
        if not items:
            print("       [WARNING] Scene not found in STAC. Using default Patagonian summer values.")
            return {"s2:mean_solar_zenith": 45.0, "datetime": "2023-01-15T14:30:00Z"}
        print(f"       [SUCCESS] Metadata fetched from STAC.")
        return items[0].properties
    except Exception as e:
        print(f"       [WARNING] STAC fetch failed ({e}). Using defaults.")
        return {"s2:mean_solar_zenith": 45.0, "datetime": "2023-01-15T14:30:00Z"}

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
    print("       nodata = -9999 (ArcGIS/ENVI compatible)")

    # Float32 output -- FLAASH accepts 32-bit float with scale factor
    # nodata=-9999 replaces 0 to avoid ambiguity with real zero-reflectance pixels
    meta.update({
        'driver': 'ENVI', 'count': 5, 'interleave': 'bil',
        'dtype': rasterio.float32, 'nodata': -9999
    })
    out_file = os.path.join(OUTPUT_DIR, f"flaash_radiance_stack_{item_id}.dat")
    
    def convert_to_radiance(reflectance_array):
        """Convert L1C TOA reflectance DN to FLAASH-compatible radiance (W/m2/sr/um)."""
        # Step 1: DN -> True Reflectance [0.0, 1.0]
        rho = reflectance_array.astype(np.float32) / 10000.0
        # Guard: zero-valued pixels are nodata (missing/cloud) -- set to NaN first
        rho = np.where(reflectance_array == 0, np.nan, rho)
        # Step 2: Reflectance -> Radiance (W/m2/sr/um)
        # L = (rho * ESUN * cos(zenith)) / (pi * d^2)
        # ESUN is band-specific and injected by the caller
        return rho  # caller multiplies by esun and pi factor

    def to_flaash_radiance(refl_dn, esun):
        rho = np.where(refl_dn == 0, np.nan, refl_dn.astype(np.float32) / 10000.0)
        radiance_W     = (rho * esun * math.cos(zenith_rad)) / (math.pi * d2)
        radiance_flaash = radiance_W * 0.1  # -> uW/cm2/sr/nm
        # Replace NaN (cloud/nodata) with -9999
        return np.where(np.isnan(radiance_flaash), -9999, radiance_flaash).astype(np.float32)
        
    with rasterio.open(out_file, 'w', **meta) as dst:
        for i, b in enumerate(["B02", "B03", "B04", "B08"], start=1):
            print(f"       -> Calibrating & Stacking {b} (10m)...")
            with rasterio.open(band_paths[b]) as src:
                dst.write(to_flaash_radiance(src.read(1), esun_dict[b]), i)

        print("       -> Resampling & Calibrating B11 (20m -> 10m)...")
        with rasterio.open(band_paths["B11"]) as src_b11:
            b11_resampled = np.empty((target_height, target_width), dtype=rasterio.uint16)
            reproject(
                source=src_b11.read(1), destination=b11_resampled,
                src_transform=src_b11.transform, src_crs=src_b11.crs,
                dst_transform=target_transform, dst_crs=target_crs,
                resampling=Resampling.bilinear
            )
            dst.write(to_flaash_radiance(b11_resampled, esun_dict["B11"]), 5)

    if not os.path.exists(out_file):
        raise RuntimeError(f"Output file was not created: {out_file}")
            
    print("[INFO] Step 3: Injecting Sensor Metadata into ENVI Header...")
    hdr_file = out_file.replace('.dat', '.hdr')
    with open(hdr_file, 'a') as hdr:
        hdr.write("wavelength units = Nanometers\n")
        hdr.write("sensor type = Sentinel-2\n")
        hdr.write(f"acquisition time = {date_str}\n")
        hdr.write("data ignore value = -9999\n")  # nodata=-9999 not 0
        hdr.write("band names = {B02 (Blue), B03 (Green), B04 (Red), B08 (NIR), B11 (SWIR)}\n")
        hdr.write("wavelength = {490.0, 560.0, 665.0, 842.0, 1610.0}\n")
        hdr.write("fwhm = {65.0, 35.0, 30.0, 115.0, 90.0}\n")

    sz = os.path.getsize(out_file) / 1e6
    print(f"\n[SUCCESS] Radiometrically Calibrated ENVI Stack Created! ({sz:.1f} MB)")
    print(f"[OUTPUT]   {out_file}")
    print("[CRITICAL] In FLAASH: Radiance Scale Factor = 1 (float32, already in uW/cm2/sr/nm)")
    print("[CRITICAL] Data Ignore Value in ENVI header = -9999")

if __name__ == "__main__":
    main()
