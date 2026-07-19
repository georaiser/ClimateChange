"""
13_envi_api_full.py
====================
GeoCascade Chapter 14 -- ENVI Python API Native Implementation
Torres del Paine, Patagonia, Chile

This script uses 100% native ENVI Python API (envi_py / ENVI Task Engine).
Requires ENVI 5.6+ installed. It does NOT use rasterio, sklearn,
or any open-source GIS library for the core processing.

HOW TO RUN:
  Option A -- ENVI IDL Console (Python Bridge):
    IDL> python_exec, "exec(open('Chapter_14/13_envi_api_full.py').read())"

  Option B -- Python with ENVI installed:
    # Activate ENVI's bundled Python environment:
    cd "C:\\Program Files\\Harris Geospatial\\ENVI 5.6\\classic\\IDL90\\bin\\bin.x86_64"
    python ../../../../python/envi_py/bin/python
    exec(open('Chapter_14/13_envi_api_full.py').read())

  Option C -- System Python if envi_py is on the path:
    pip install envi_py   (if ENVI 5.6 is installed, not a PyPI package)
    python Chapter_14/13_envi_api_full.py

  Option D -- Open-source fallback (no ENVI license):
    python Chapter_14/07_envi_spectral_analysis.py   # NDVI etc.
    python Chapter_14/08_envi_atm_correction.py       # DOS1
    python Chapter_14/09_envi_classification.py       # classify
    python Chapter_14/10_envi_change_detection.py     # change

ENVI TASK ENGINE API REFERENCE:
  https://www.nv5geospatialsoftware.com/docs/python_envi_bridge.html
  All tasks: ENVI().TaskCatalog()

WORKFLOWS COVERED:
  [1] QUAC / FLAASH    -- Atmospheric correction
  [2] Band Math        -- NDVI, NBR, NDWI, NDSI, EVI
  [3] ISODATA          -- Unsupervised classification
  [4] MLC              -- Supervised classification
  [5] Majority Filter  -- Post-classification smoothing
  [6] Class Statistics -- Area summary table
  [7] Change Detection -- NDVI before/after
  [8] Spectral Subset  -- Extract bands from multiband image
  [9] Resampling       -- Resample to common grid
  [10] Export          -- GeoTIFF + ENVI .hdr format

STUDY AREA: Torres del Paine, Patagonia, Chile
BBOX WGS84: [-73.5, -51.5, -72.5, -50.5]

OUTPUTS:
  data/processed/envi_outputs/api_quac.tif
  data/processed/envi_outputs/api_ndvi.tif
  data/processed/envi_outputs/api_nbr.tif
  data/processed/envi_outputs/api_ndwi.tif
  data/processed/envi_outputs/api_ndsi.tif
  data/processed/envi_outputs/api_evi.tif
  data/processed/envi_outputs/api_isodata.tif
  data/processed/envi_outputs/api_mlc.tif
  data/processed/envi_outputs/api_majority.tif
  data/processed/envi_outputs/api_change.tif
  data/processed/envi_outputs/api_class_stats.csv
"""

import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# ENVI Python API import guard
# ---------------------------------------------------------------------------
try:
    from envi import ENVI, ENVITask
    ENVI_OK = True
    print("  [OK] ENVI Python API imported")
except ImportError:
    ENVI_OK = False
    print("=" * 65)
    print(" [ERROR] ENVI Python API (envi_py) not available.")
    print()
    print(" Requires: ENVI 5.6+ installed on this machine.")
    print()
    print(" To use the ENVI Python API:")
    print("   1. Install ENVI 5.6 (Harris Geospatial / NV5)")
    print("   2. Find the ENVI Python bridge:")
    print("      C:\\Program Files\\Harris Geospatial\\ENVI 5.6\\")
    print("           python\\envi_py\\")
    print("   3. Install with:")
    print("      pip install -e \"C:\\...\\ENVI 5.6\\python\\envi_py\"")
    print("   4. Re-run this script.")
    print()
    print(" OPEN-SOURCE ALTERNATIVE (no ENVI required):")
    print("   python Chapter_14/07_envi_spectral_analysis.py")
    print("   python Chapter_14/08_envi_atm_correction.py")
    print("   python Chapter_14/09_envi_classification.py")
    print("   python Chapter_14/10_envi_change_detection.py")
    print()
    print(" IDL ALTERNATIVE (requires ENVI + IDL):")
    print("   .compile Chapter_14/11_idl_reference.pro")
    print("   geocascade_run_all")
    print("=" * 65)
    sys.exit(1)

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
BASE_DIR  = Path(__file__).parent
ROOT      = BASE_DIR.parent
ENVI_DIR  = BASE_DIR / "data" / "processed" / "envi_outputs"
ENVI_DIR.mkdir(parents=True, exist_ok=True)

CHIRPS_TIF = ROOT / "Chapter_01" / "data" / "processed" / "real_data" / "chirps_mean_annual_precip.tif"
TEMP_TIF   = ROOT / "Chapter_01" / "data" / "processed" / "climate_analysis" / "temperature_surface.tif"

# Input from scripts 07/08
NDVI_TIF   = ENVI_DIR / "ndvi.tif"
DOS1_TIF   = ENVI_DIR / "sentinel2_dos1_corrected.tif"
NDVI_2019  = ENVI_DIR / "ndvi_2019.tif"
NDVI_2023  = ENVI_DIR / "ndvi_2023.tif"


def find_tif(preferred: Path, *fallbacks) -> str:
    """Return first existing path."""
    for p in [preferred, *fallbacks]:
        if Path(p).exists():
            return str(p)
    return str(preferred)   # will fail with clear message inside ENVI task


# ---------------------------------------------------------------------------
# ENVI session singleton
# ---------------------------------------------------------------------------
_envi_session = None


def get_envi() -> "ENVI":
    """Return or start a headless ENVI session."""
    global _envi_session
    if _envi_session is None:
        _envi_session = ENVI(headless=True)
        print(f"  ENVI started (headless). Version: {_envi_session.version}")
    return _envi_session


# ---------------------------------------------------------------------------
# [1] ATMOSPHERIC CORRECTION -- QUAC (quick, no metadata) / FLAASH (full)
# ---------------------------------------------------------------------------

def atm_correction_quac() -> str:
    """
    QUAC (Quick Atmospheric Correction) -- no scene metadata required.
    Suitable for Sentinel-2 and Landsat. Faster than FLAASH.
    Output: surface reflectance scaled 0-1.

    ENVI GUI:
      Toolbox > Radiometric Correction > Atmospheric Correction Module > QUAC
      Input  : Sentinel-2 multiband TIF
      Output : api_quac.tif

    FLAASH alternative (requires scene metadata):
      Toolbox > Radiometric Correction > Atmospheric Correction Module > FLAASH
      See 08_envi_atm_correction.py docstring for parameters.
    """
    print("\n[1] Atmospheric Correction (QUAC)")
    e = get_envi()

    in_path  = find_tif(DOS1_TIF)
    out_path = str(ENVI_DIR / "api_quac.tif")

    raster = e.open_raster(in_path)
    print(f"  Input : {os.path.basename(in_path)}  "
          f"({raster.nrows}x{raster.ncols}, {raster.nbands} bands)")

    task = ENVITask("QUACCorrection")
    task["INPUT_RASTER"]     = raster
    task["OUTPUT_RASTER_URI"] = out_path
    task.execute()

    sr_raster = e.open_raster(out_path)
    print(f"  QUAC surface reflectance: {sr_raster.nrows}x{sr_raster.ncols}")
    print(f"  [OK] {os.path.relpath(out_path, BASE_DIR)}")
    return out_path


# ---------------------------------------------------------------------------
# [2] BAND MATH -- Spectral Indices
# ---------------------------------------------------------------------------

def compute_spectral_indices(in_path: str) -> dict:
    """
    Compute NDVI, NBR, NDWI, NDSI, EVI using ENVI Band Math task.

    ENVI GUI:
      Toolbox > Band Math
      Expression: (float(b4)-float(b3))/(float(b4)+float(b3))
      where band indices match Sentinel-2 L2A: b1=B2, b2=B3, b3=B4, b4=B8, b5=B11, b6=B12

    ENVI Task Engine:
      task = ENVITask('BandMath')
      task['EXPRESSION'] = '...'
    """
    print("\n[2] Spectral Indices via ENVI Band Math")
    e = get_envi()

    raster = e.open_raster(in_path)
    print(f"  Input: {os.path.basename(in_path)}  ({raster.nbands} bands)")
    print(f"  Band order: B2(1), B3(2), B4(3), B8(4), B11(5), B12(6)")

    # Sentinel-2 band expressions (1-indexed ENVI convention)
    indices = {
        "ndvi": "(float(b4)-float(b3))/(float(b4)+float(b3))",
        "nbr":  "(float(b4)-float(b6))/(float(b4)+float(b6))",
        "ndwi": "(float(b2)-float(b4))/(float(b2)+float(b4))",
        "ndsi": "(float(b2)-float(b5))/(float(b2)+float(b5))",
        "evi":  "2.5*((float(b4)-float(b3))/(float(b4)+6.0*float(b3)-7.5*float(b1)+1.0))",
    }

    out_paths = {}
    for name, expr in indices.items():
        out_path = str(ENVI_DIR / f"api_{name}.tif")
        task = ENVITask("BandMath")
        task["INPUT_RASTERS"]     = [raster]
        task["EXPRESSION"]        = expr
        task["OUTPUT_RASTER_URI"] = out_path
        task.execute()

        # Get statistics
        r = e.open_raster(out_path)
        stats = ENVITask("StatisticsRaster")
        stats["INPUT_RASTER"] = r
        stats.execute()
        mn  = round(float(stats["MIN_VALUES"][0]), 4)
        mx  = round(float(stats["MAX_VALUES"][0]), 4)
        avg = round(float(stats["MEAN_VALUES"][0]), 4)
        print(f"  {name.upper():<6}: min={mn:+.3f}  max={mx:+.3f}  mean={avg:+.3f}")
        out_paths[name] = out_path

    return out_paths


# ---------------------------------------------------------------------------
# [3] ISODATA -- Unsupervised Classification
# ---------------------------------------------------------------------------

def isodata_classification(in_path: str) -> tuple:
    """
    ISODATA unsupervised classification (iterative self-organising).
    Produces 5 spectral clusters ordered by mean reflectance.

    ENVI GUI:
      Classification > Unsupervised > ISODATA
        Number of classes     : 5
        Maximum iterations    : 20
        Change threshold %    : 5
        Minimum class size    : 10
        Maximum class std dev : 1.0
        Minimum class distance: 5.0
        Maximum merge pairs   : 2
        Maximum split std dev : 1.0
    """
    print("\n[3] ISODATA Unsupervised Classification (5 classes)")
    e = get_envi()

    out_path = str(ENVI_DIR / "api_isodata.tif")

    task = ENVITask("ISODATAClassification")
    task["INPUT_RASTER"]       = e.open_raster(in_path)
    task["NUMBER_OF_CLASSES"]  = 5
    task["ITERATIONS"]         = 20
    task["CHANGE_THRESHOLD"]   = 5.0    # % pixels that change to continue
    task["MINIMUM_CLASS_SIZE"] = 10     # pixels
    task["SPLIT_THRESHOLD"]    = 1.0    # std dev to split a class
    task["MERGE_THRESHOLD"]    = 0.5    # distance to merge two classes
    task["OUTPUT_RASTER_URI"]  = out_path
    task.execute()

    sig_path = out_path.replace(".tif", ".sta")
    print(f"  [OK] ISODATA classified: {os.path.relpath(out_path, BASE_DIR)}")
    print(f"  Signature file: {os.path.relpath(sig_path, BASE_DIR)}")
    return out_path, sig_path


# ---------------------------------------------------------------------------
# [4] MAXIMUM LIKELIHOOD -- Supervised Classification
# ---------------------------------------------------------------------------

def mlc_classification(raster_path: str, sig_path: str) -> str:
    """
    Maximum Likelihood Classification using ISODATA training signatures.

    ENVI GUI:
      Classification > Supervised > Maximum Likelihood
        Input raster    : api_quac.tif (atmospherically corrected)
        Signature file  : api_isodata.sta
        Reject fraction : 0.0
        Output raster   : api_mlc.tif
    """
    print("\n[4] Maximum Likelihood Classification")
    e = get_envi()

    out_path = str(ENVI_DIR / "api_mlc.tif")

    task = ENVITask("MLClassify")
    task["INPUT_RASTER"]       = e.open_raster(raster_path)
    task["INPUT_SIGNATURE"]    = sig_path
    task["REJECT_FRACTION"]    = 0.0    # classify all pixels
    task["A_PRIORI"]           = "EQUAL"   # equal prior probability per class
    task["OUTPUT_RASTER_URI"]  = out_path
    task.execute()

    print(f"  [OK] MLC: {os.path.relpath(out_path, BASE_DIR)}")
    return out_path


# ---------------------------------------------------------------------------
# [5] MAJORITY FILTER -- Post-classification Smoothing
# ---------------------------------------------------------------------------

def majority_filter(cls_path: str) -> str:
    """
    Majority analysis (3x3 window) to remove salt-and-pepper noise.
    Replaces each pixel with the most common class in a 3x3 neighbourhood.

    ENVI GUI:
      Classification > Post Classification > Majority Analysis
        Input raster  : api_mlc.tif
        Kernel size   : 3 x 3
        Center weight : 1
    """
    print("\n[5] Majority Filter (3x3 post-classification)")
    e = get_envi()

    out_path = str(ENVI_DIR / "api_majority.tif")

    task = ENVITask("ClassificationAggregation")
    task["INPUT_RASTER"]     = e.open_raster(cls_path)
    task["KERNEL_SIZE"]      = 3
    task["OUTPUT_RASTER_URI"] = out_path
    task.execute()

    print(f"  [OK] Majority filter: {os.path.relpath(out_path, BASE_DIR)}")
    return out_path


# ---------------------------------------------------------------------------
# [6] CLASS STATISTICS
# ---------------------------------------------------------------------------

def class_statistics(cls_path: str, value_path: str) -> str:
    """
    Compute statistics of a value raster (NDVI) per class from a classified raster.

    ENVI GUI:
      Classification > Post Classification > Class Statistics
        Input raster        : api_majority.tif
        Input value raster  : api_ndvi.tif
        Statistics computed : Count, Min, Max, Mean, Std Dev

    Outputs a text report and a CSV.
    """
    print("\n[6] Class Statistics (NDVI per land cover class)")
    import csv
    e = get_envi()

    out_path     = str(ENVI_DIR / "api_class_stats.csv")
    report_path  = str(ENVI_DIR / "api_class_stats.txt")

    task = ENVITask("ClassificationStatistics")
    task["INPUT_CLASSIFICATION_RASTER"] = e.open_raster(cls_path)
    task["INPUT_RASTER"]                = e.open_raster(value_path)
    task["OUTPUT_REPORT_URI"]           = report_path
    task.execute()

    class_names = ["Water", "Snow/Ice", "Bare Rock", "Sparse Veg", "Dense Veg"]
    rows = []

    stats_arr = task["CLASS_STATISTICS"]    # array: [n_classes, 5] = count,min,max,mean,std
    print(f"  {'Class':<14} {'Count':>8} {'Mean':>8} {'Min':>8} {'Max':>8} {'Std':>8}")
    print("  " + "-" * 58)

    for i, name in enumerate(class_names):
        if i < stats_arr.shape[0]:
            cnt  = int(stats_arr[i, 0])
            mn   = round(float(stats_arr[i, 1]), 4)
            mx   = round(float(stats_arr[i, 2]), 4)
            mean = round(float(stats_arr[i, 3]), 4)
            std  = round(float(stats_arr[i, 4]), 4)
        else:
            cnt = mn = mx = mean = std = 0
        print(f"  {name:<14} {cnt:>8,} {mean:>8.4f} {mn:>8.4f} {mx:>8.4f} {std:>8.4f}")
        rows.append({"class": name, "count": cnt, "mean": mean,
                     "min": mn, "max": mx, "std": std})

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "count", "mean", "min", "max", "std"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"  [OK] Class stats CSV: {os.path.relpath(out_path, BASE_DIR)}")
    return out_path


# ---------------------------------------------------------------------------
# [7] CHANGE DETECTION
# ---------------------------------------------------------------------------

def change_detection(path_before: str, path_after: str) -> str:
    """
    ENVI Image Change Detection using band differencing.
    Compares NDVI 2019 vs NDVI 2023.

    ENVI GUI:
      Toolbox > Change Detection > Image Change Workflow
        Step 1 -- Select images: ndvi_2019.tif, ndvi_2023.tif
        Step 2 -- Method: Image Difference
        Step 3 -- Threshold: Standard Deviation (2 sigma)
        Step 4 -- Export change map, statistics
    """
    print("\n[7] Change Detection (NDVI 2019 vs 2023)")
    e = get_envi()

    diff_path = str(ENVI_DIR / "api_ndvi_diff.tif")
    cls_path  = str(ENVI_DIR / "api_change.tif")

    # Step 1: Image Difference
    r1 = e.open_raster(path_before)
    r2 = e.open_raster(path_after)

    diff_task = ENVITask("ImageBandDifference")
    diff_task["INPUT_RASTER1"]     = r1
    diff_task["INPUT_RASTER2"]     = r2
    diff_task["OUTPUT_RASTER_URI"] = diff_path
    diff_task.execute()

    print(f"  NDVI difference computed: {os.path.relpath(diff_path, BASE_DIR)}")

    # Step 2: Classify change using Band Math thresholds
    # Loss: delta < -0.10  (class 0)
    # Stable: -0.10 to +0.10 (class 1)
    # Gain: delta > +0.10 (class 2)
    cls_task = ENVITask("BandMath")
    cls_task["INPUT_RASTERS"]     = [e.open_raster(diff_path)]
    cls_task["EXPRESSION"]        = ("(b1 lt -0.10) * 0 + "
                                     "(b1 ge -0.10 and b1 le 0.10) * 1 + "
                                     "(b1 gt 0.10) * 2")
    cls_task["OUTPUT_RASTER_URI"] = cls_path
    cls_task.execute()

    # Step 3: Report change areas
    r_cls = e.open_raster(cls_path)
    data  = r_cls.get_data()     # returns numpy array
    pixel_area_km2 = 0.0111 * 0.0111   # ~1.1 km resolution at -51 deg

    labels = {0: "Vegetation Loss", 1: "No Change", 2: "Vegetation Gain"}
    print(f"\n  {'Change Class':<20} {'Pixels':>8} {'Area km2':>10} {'Pct':>7}")
    print("  " + "-" * 50)
    total = data.size
    for cls_id, name in labels.items():
        n = int((data == cls_id).sum())
        area = n * pixel_area_km2
        pct  = n / total * 100
        print(f"  {name:<20} {n:>8,} {area:>10.1f} {pct:>6.1f}%")

    print(f"\n  [OK] Change map: {os.path.relpath(cls_path, BASE_DIR)}")
    return cls_path


# ---------------------------------------------------------------------------
# [8] SPECTRAL SUBSET
# ---------------------------------------------------------------------------

def spectral_subset(in_path: str, bands: list) -> str:
    """
    Extract specific bands from a multiband image.

    ENVI GUI:
      File > Save As > ENVI File -> check specific bands
      OR: Toolbox > Raster Management > Build Raster Series

    Useful for extracting RGB composite (B4, B3, B2) for display.
    """
    print(f"\n[8] Spectral Subset: extracting bands {bands}")
    e = get_envi()

    out_path = str(ENVI_DIR / "api_rgb_subset.tif")
    raster   = e.open_raster(in_path)

    task = ENVITask("SpectralSubset")
    task["INPUT_RASTER"]        = raster
    task["BANDS"]               = bands      # 0-indexed list e.g. [2, 1, 0] for R,G,B
    task["OUTPUT_RASTER_URI"]   = out_path
    task.execute()

    print(f"  Input bands selected: {bands} (0-indexed)")
    print(f"  [OK] {os.path.relpath(out_path, BASE_DIR)}")
    return out_path


# ---------------------------------------------------------------------------
# [9] RESAMPLING -- Align rasters to common grid
# ---------------------------------------------------------------------------

def resample_to_reference(in_path: str, ref_path: str) -> str:
    """
    Resample a raster to match the spatial extent and resolution of a reference.
    Required before Weighted Overlay (all inputs must share the same grid).

    ENVI GUI:
      Toolbox > Raster Management > Reproject Raster
        Input raster  : ndvi.tif (120x120)
        Reference     : dem_proxy.tif (100x100)
        Resampling    : Bilinear
    """
    print(f"\n[9] Resampling: {os.path.basename(in_path)} -> match {os.path.basename(ref_path)}")
    e = get_envi()

    out_path = str(ENVI_DIR / f"api_resampled_{os.path.basename(in_path)}")
    ref_raster = e.open_raster(ref_path)

    task = ENVITask("RegridRaster")
    task["INPUT_RASTER"]     = e.open_raster(in_path)
    task["GRID_RASTER"]      = ref_raster    # defines target grid
    task["RESAMPLING"]       = "Bilinear"
    task["OUTPUT_RASTER_URI"] = out_path
    task.execute()

    r = e.open_raster(out_path)
    print(f"  Resampled to: {r.nrows}x{r.ncols} "
          f"(reference: {ref_raster.nrows}x{ref_raster.ncols})")
    print(f"  [OK] {os.path.relpath(out_path, BASE_DIR)}")
    return out_path


# ---------------------------------------------------------------------------
# [10] EXPORT -- GeoTIFF + ENVI HDR
# ---------------------------------------------------------------------------

def export_envi_format(in_path: str) -> tuple:
    """
    Export a raster in both GeoTIFF and ENVI .hdr format.

    ENVI GUI:
      File > Save As > ENVI File (saves .img + .hdr)
      File > Save As > TIFF/GeoTIFF (saves .tif)

    The ENVI .hdr format stores metadata such as:
      wavelength      = {490.0, 560.0, 665.0, 842.0, 1610.0, 2190.0}
      band names      = {Blue, Green, Red, NIR, SWIR1, SWIR2}
      coordinate info = {WGS-84, ...}
    """
    print(f"\n[10] Export: {os.path.basename(in_path)}")
    e = get_envi()

    base_stem = os.path.splitext(in_path)[0]
    hdr_out   = base_stem + "_envi.hdr"
    tif_out   = base_stem + "_export.tif"

    raster = e.open_raster(in_path)

    # Export as ENVI format
    envi_task = ENVITask("ExportRaster")
    envi_task["INPUT_RASTER"]     = raster
    envi_task["OUTPUT_URI"]        = base_stem + "_envi.dat"
    envi_task["FORMAT"]            = "ENVI"
    envi_task.execute()
    print(f"  [OK] ENVI format: {os.path.basename(base_stem)}_envi.dat + .hdr")

    # Export as GeoTIFF (compress with LZW)
    tif_task = ENVITask("ExportRaster")
    tif_task["INPUT_RASTER"]     = raster
    tif_task["OUTPUT_URI"]        = tif_out
    tif_task["FORMAT"]            = "TIFF"
    tif_task["COMPRESSION"]       = "LZW"
    tif_task.execute()
    print(f"  [OK] GeoTIFF: {os.path.basename(tif_out)}")

    return hdr_out, tif_out


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print(" GEOCASCADE Ch14 -- ENVI Python API Native Pipeline")
    print(" QUAC | Band Math | ISODATA | MLC | Majority | Change")
    print("=" * 65)

    e = get_envi()
    print(f"  ENVI Task catalog: {len(e.task_catalog)} tasks available")
    results = {}

    # [1] Atmospheric Correction
    try:
        results["quac"] = atm_correction_quac()
    except Exception as ex:
        print(f"  [FAIL] QUAC: {ex}")
        results["quac"] = find_tif(DOS1_TIF)    # fall back to DOS1

    # [2] Spectral Indices (on corrected image)
    try:
        in_img = results.get("quac") or find_tif(DOS1_TIF)
        idx = compute_spectral_indices(in_img)
        results.update(idx)
    except Exception as ex:
        print(f"  [FAIL] Band Math: {ex}")
        results["ndvi"] = find_tif(NDVI_TIF)

    # [3] ISODATA Classification
    try:
        in_img = results.get("quac") or find_tif(DOS1_TIF)
        results["isodata"], sig = isodata_classification(in_img)
    except Exception as ex:
        print(f"  [FAIL] ISODATA: {ex}")
        results["isodata"], sig = None, None

    # [4] MLC
    try:
        in_img = results.get("quac") or find_tif(DOS1_TIF)
        if sig:
            results["mlc"] = mlc_classification(in_img, sig)
    except Exception as ex:
        print(f"  [FAIL] MLC: {ex}")
        results["mlc"] = None

    # [5] Majority Filter
    try:
        cls_in = results.get("mlc") or results.get("isodata")
        if cls_in:
            results["majority"] = majority_filter(cls_in)
    except Exception as ex:
        print(f"  [FAIL] Majority Filter: {ex}")
        results["majority"] = None

    # [6] Class Statistics
    try:
        cls_in  = results.get("majority") or results.get("isodata")
        ndvi_in = results.get("ndvi") or find_tif(NDVI_TIF)
        if cls_in and ndvi_in:
            results["stats"] = class_statistics(cls_in, ndvi_in)
    except Exception as ex:
        print(f"  [FAIL] Class Statistics: {ex}")
        results["stats"] = None

    # [7] Change Detection
    try:
        b_path = find_tif(NDVI_2019)
        a_path = find_tif(NDVI_2023)
        results["change"] = change_detection(b_path, a_path)
    except Exception as ex:
        print(f"  [FAIL] Change Detection: {ex}")
        results["change"] = None

    # [8] Spectral Subset (RGB)
    try:
        in_img = results.get("quac") or find_tif(DOS1_TIF)
        results["rgb"] = spectral_subset(in_img, [2, 1, 0])   # R, G, B
    except Exception as ex:
        print(f"  [FAIL] Spectral Subset: {ex}")
        results["rgb"] = None

    # [9-10] Resample + Export
    try:
        ndvi_in = results.get("ndvi")
        if ndvi_in:
            resample_to_reference(ndvi_in, ndvi_in)   # example: resample to self
    except Exception as ex:
        print(f"  [FAIL] Resample: {ex}")

    try:
        ndvi_in = results.get("ndvi")
        if ndvi_in:
            export_envi_format(ndvi_in)
    except Exception as ex:
        print(f"  [FAIL] Export: {ex}")

    # Close ENVI
    e.close()
    print("\n  ENVI session closed.")

    # Summary
    passed = sum(1 for v in results.values() if v is not None)
    total  = len(results)
    print("\n" + "=" * 65)
    print(f" ENVI API PIPELINE COMPLETE  ({passed}/{total} steps succeeded)")
    print("=" * 65)
    for step, path in results.items():
        status = "[OK]  " if path else "[SKIP]"
        name   = os.path.basename(str(path)) if path else "---"
        print(f"  {status} {step:<12}: {name}")
    print()
    print(f"  All outputs: {ENVI_DIR}")
    print()
    print("  ArcGIS Pro: Add any api_*.tif -> Add Data")
    print("  ENVI      : File > Open -> api_*.tif or api_*_envi.dat")
    print("  IDL       : .compile 11_idl_reference.pro -> geocascade_run_all")
    print("=" * 65)


if __name__ == "__main__":
    main()
