"""
envi/01_flaash_correction.py
============================
ENVI 5.6 FLAASH Atmospheric Correction - Python API Wrapper

This script uses the ENVI Python API (available in ENVI 5.3+).

HOW TO RUN:
    Option A - ENVI Python Console (recommended):
        1. Open ENVI 5.6
        2. Tools > Python Console
        3. Run: exec(open(r'D:\\...\\envi\\01_flaash_correction.py').read())

    Option B - ENVI Python Script Runner:
        1. ENVI Toolbox > ENVI Modeler > Python Script
        2. Browse to this file and execute

    Option C - External Python (requires ENVI Runtime license):
        conda activate geocascade_env
        python Chapter_01/envi/01_flaash_correction.py

NOTE: The IDL script (01_flaash_correction.pro) is simpler and more
      reliable for most use cases. Use this script for:
      - Batch automation of many scenes
      - Integration with Python post-processing workflows

DECISION TABLE - when to apply atmospheric correction:
    Sentinel-2 L2A   : ALREADY corrected by ESA Sen2Cor  -> use directly
    Landsat 9 L2SP   : ALREADY corrected by USGS LaSRC   -> use directly
    Sentinel-2 L1C   : raw TOA -> APPLY FLAASH or Sen2Cor
    Landsat 9 L1TP   : raw TOA -> APPLY FLAASH or DOS1

Dependencies: envi (ENVI Python API, bundled with ENVI 5.x)
"""

import os
import sys

# ---------------------------------------------------------------------------
# Try to import ENVI Python API
# ---------------------------------------------------------------------------
try:
    from envi import *
    ENVI_API_AVAILABLE = True
except ImportError:
    ENVI_API_AVAILABLE = False

if not ENVI_API_AVAILABLE:
    print("=" * 60)
    print(" ENVI Python API not found.")
    print()
    print(" Solutions:")
    print("   1. Run this script from ENVI Tools > Python Console")
    print("      (ENVI's built-in Python has 'envi' pre-installed)")
    print()
    print("   2. Use the IDL script instead:")
    print("      ENVI Toolbox > Run IDL Script >")
    print("      Chapter_01/envi/01_flaash_correction.pro")
    print()
    print("   3. If you have ENVI Runtime license, install the envi")
    print("      Python package and run from external Python.")
    print("=" * 60)
    sys.exit(0)


# ---------------------------------------------------------------------------
# Configuration - edit these paths before running
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)

# Input: Landsat L1TP stacked radiance (all bands, ENVI or GeoTIFF format)
INPUT_FILE = os.path.join(
    BASE_DIR, "data", "raw", "landsat_l1tp", "landsat9_l1tp_stack.dat"
)

# Output directory
OUTPUT_DIR = os.path.join(
    BASE_DIR, "data", "processed", "envi_flaash"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Scene parameters
SENSOR_TYPE  = "OLI"         # Landsat 9 OLI; use 'MSI' for Sentinel-2 L1C
SCENE_DATE   = "2023-01-15"  # acquisition date (YYYY-MM-DD)
SCENE_LAT    =  -51.0        # scene centre latitude (negative = South)
SCENE_LON    =  -73.0        # scene centre longitude (negative = West)
ELEVATION_KM =   0.15        # mean terrain elevation in km

# FLAASH atmospheric model
# Sub-Arctic Summer is the best match for Patagonia ~51 deg S in summer
# Options: 'TROPICAL', 'MID_LATITUDE_SUMMER', 'MID_LATITUDE_WINTER',
#          'SUB_ARCTIC_SUMMER', 'SUB_ARCTIC_WINTER', 'US_STANDARD_1962'
ATMOSPHERIC_MODEL = "SUB_ARCTIC_SUMMER"

# Aerosol model
# Maritime recommended for Pacific-coast scenes (dominant onshore winds)
# Options: 'RURAL', 'URBAN', 'MARITIME', 'TROPOSPHERIC'
AEROSOL_MODEL = "MARITIME"

# Visibility in km (clear Patagonia: 40 km, hazy: 20-25 km)
INITIAL_VISIBILITY = 40.0


# ---------------------------------------------------------------------------
# FLAASH Correction
# ---------------------------------------------------------------------------
def run_flaash():
    print("=" * 60)
    print(" ENVI 5.6 FLAASH Atmospheric Correction (Python API)")
    print(f" Input   : {INPUT_FILE}")
    print(f" Output  : {OUTPUT_DIR}")
    print(f" Sensor  : {SENSOR_TYPE}")
    print(f" AtmModel: {ATMOSPHERIC_MODEL}")
    print(f" Aerosol : {AEROSOL_MODEL}")
    print(f" Vis     : {INITIAL_VISIBILITY} km")
    print("=" * 60)

    # Validate input
    if not os.path.exists(INPUT_FILE):
        print(f"\n ERROR: Input file not found:")
        print(f"   {INPUT_FILE}")
        print()
        print(" Stack individual Landsat bands in ENVI:")
        print("   Raster Management > Layer Stacking")
        return None

    # Open ENVI (headless=False to show progress in ENVI UI)
    print("\n Opening ENVI engine...")
    e = ENVI(headless=False)

    # Open input raster
    print(" Opening input raster...")
    raster = e.OpenRaster(INPUT_FILE)
    print(f"   Bands     : {raster.NBANDS}")
    print(f"   Rows      : {raster.NROWS}")
    print(f"   Columns   : {raster.NCOLUMNS}")

    # Output paths
    output_rfl     = os.path.join(OUTPUT_DIR, "flaash_corrected.dat")
    output_rfl_tif = os.path.join(OUTPUT_DIR, "flaash_corrected_rfl.tif")
    output_cloud   = os.path.join(OUTPUT_DIR, "flaash_cloud_mask.dat")
    output_water   = os.path.join(OUTPUT_DIR, "flaash_water_vapor.dat")
    output_report  = os.path.join(OUTPUT_DIR, "flaash_report.txt")

    # Parse date for JULDAY equivalent
    yr, mo, dy = [int(x) for x in SCENE_DATE.split("-")]

    # Get FLAASH task
    print("\n Configuring FLAASH task...")
    try:
        flaash = e.Task("FLAASH")
    except Exception:
        # Some ENVI 5.6 builds use 'SpectralFLAASH'
        try:
            flaash = e.Task("SpectralFLAASH")
        except Exception as err:
            print(f"\n ERROR: Could not find FLAASH task: {err}")
            print(" Ensure ENVI is fully licensed including Atmospheric Correction.")
            return None

    # Set task parameters
    flaash.INPUT_RASTER                  = raster
    flaash.OUTPUT_REFLECTANCE_RASTER_URI = output_rfl
    flaash.OUTPUT_CLOUD_RASTER_URI       = output_cloud
    flaash.OUTPUT_WATER_VAPOR_RASTER_URI = output_water
    flaash.OUTPUT_REPORT_URI             = output_report

    flaash.SCENE_CENTER_LAT    = SCENE_LAT
    flaash.SCENE_CENTER_LON    = SCENE_LON
    flaash.SENSOR_TYPE         = SENSOR_TYPE
    flaash.FLIGHT_DATE         = f"{SCENE_DATE}"
    flaash.SCENE_ELEVATION     = ELEVATION_KM

    flaash.ATMOSPHERIC_MODEL   = ATMOSPHERIC_MODEL
    flaash.AEROSOL_MODEL       = AEROSOL_MODEL
    flaash.INITIAL_VISIBILITY  = INITIAL_VISIBILITY

    flaash.WATER_RETRIEVAL                = True
    flaash.WATER_ABSORPTION_FEATURE_WAVELENGTH = 1135
    flaash.AEROSOL_RETRIEVAL              = True
    flaash.MULTI_SCATTER_MODEL            = 9  # Kaufman-Tanre
    flaash.AEROSOL_SCALE_FACTOR           = 10

    # Execute
    print("\n Running FLAASH (this may take 5-20 minutes)...")
    flaash.Execute()

    # Export as GeoTIFF for ArcGIS Pro / Python post-processing
    print("\n Exporting GeoTIFF...")
    rfl_raster   = e.OpenRaster(output_rfl)
    export_task  = e.Task("ExportRasterToFormat")
    export_task.INPUT_RASTER = rfl_raster
    export_task.OUTPUT_URI   = output_rfl_tif
    export_task.DATA_TYPE    = "FLOAT"
    export_task.Execute()

    if os.path.exists(output_rfl_tif):
        sz = os.path.getsize(output_rfl_tif) / 1e6
        print(f"\n [OK] Reflectance GeoTIFF: {output_rfl_tif} ({sz:.0f} MB)")
        print(f"      ArcGIS Pro: Add as raster layer (Stretched symbology)")
        print(f"      Python    : rasterio.open(r'{output_rfl_tif}')")
    else:
        print("\n WARNING: GeoTIFF export may not have completed.")
        print(f"          Check ENVI format output: {output_rfl}")

    print(f"\n FLAASH report: {output_report}")
    print("\n Done.")
    return output_rfl_tif


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    out = run_flaash()
    if out:
        print(f"\n Next step: open {out} in ArcGIS Pro or ENVI for analysis.")
        print(" See Chapter_01/envi/02_spectral_analysis.pro for spectral profiling.")
