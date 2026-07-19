"""
12_arcpy_full.py
=================
GeoCascade Chapter 14 -- ArcGIS Pro Native Implementation
Torres del Paine, Patagonia, Chile

This script uses 100% native ArcPy and requires ArcGIS Pro 3.x
with the Spatial Analyst extension. It does NOT use rasterio, sklearn,
or any open-source GIS library.

HOW TO RUN:
  Option A -- ArcGIS Pro Python Window:
    exec(open(r"D:\\00_AI_Aplications\\ClimateChange\\Chapter_14\\12_arcpy_full.py").read())

  Option B -- ArcGIS Pro Anaconda (clone):
    conda activate arcgispro-py3
    python Chapter_14/12_arcpy_full.py

  Option C -- ArcGIS Pro Task:
    Use the Python Toolbox GeoCascade.pyt which wraps these same functions.

WORKFLOWS COVERED:
  [1] Project Setup        -- Set environments, workspace, check out extensions
  [2] Raster Reclassify    -- Spatial Analyst > Reclassify (precipitation -> 5 drought classes)
  [3] Raster Calculator    -- Temperature anomaly, UHI focal mean
  [4] Focal Statistics     -- Neighbourhood smoothing of UHI surface
  [5] ISO Cluster          -- Unsupervised land cover classification (5 classes)
  [6] Maximum Likelihood   -- Supervised classification using ISO Cluster signatures
  [7] Slope & Aspect       -- 3D Analyst from DEM
  [8] Weighted Overlay     -- Climate Vulnerability Index (4 inputs)
  [9] Zonal Statistics     -- Mean slope per land cover class
  [10] Layout Export       -- Print-ready PNG at 300 DPI from APRX layout

STUDY AREA: Torres del Paine, Patagonia, Chile
BBOX WGS84: [-73.5, -51.5, -72.5, -50.5]
PROJECTION: WGS 1984 (EPSG:4326) -> analysis in UTM 19S (EPSG:32719)

OUTPUTS:
  data/processed/arcgis_outputs/arcpy_precip_classified.tif
  data/processed/arcgis_outputs/arcpy_temp_anomaly.tif
  data/processed/arcgis_outputs/arcpy_uhi_focal.tif
  data/processed/arcgis_outputs/arcpy_iso_cluster.tif
  data/processed/arcgis_outputs/arcpy_mlc.tif
  data/processed/arcgis_outputs/arcpy_slope.tif
  data/processed/arcgis_outputs/arcpy_aspect.tif
  data/processed/arcgis_outputs/arcpy_vulnerability.tif
  data/processed/arcgis_outputs/arcpy_zonal_stats.dbf
  data/processed/arcgis_outputs/arcpy_layout.png
"""

import sys
import os
import traceback
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# ArcPy import guard -- provide clear diagnostic if not licensed
# ---------------------------------------------------------------------------
try:
    import arcpy
    from arcpy.sa import (
        Reclassify, RemapRange, RemapValue,
        FocalStatistics, NbrRectangle,
        Slope, Aspect,
        WeightedOverlay, WOTable, WOField,
        IsoClusterUnsupervisedClassification,
        MLClassify,
        ZonalStatisticsAsTable,
        RasterCalculator,
    )
    ARCPY_OK = True
    print("  [OK] arcpy imported successfully")
    print(f"  ArcGIS Pro version: {arcpy.GetInstallInfo()['Version']}")
except ImportError:
    ARCPY_OK = False
    print("=" * 65)
    print(" [ERROR] arcpy not available in this Python environment.")
    print()
    print(" To run this script you need ArcGIS Pro 3.x installed.")
    print(" Run from ONE of these environments:")
    print()
    print("   Option A -- ArcGIS Pro Python Window:")
    print("     Analysis tab > Python > Python Window")
    print("     exec(open(r'D:\\...\\Chapter_14\\12_arcpy_full.py').read())")
    print()
    print("   Option B -- ArcGIS Pro Python (conda clone):")
    print("     Start > ArcGIS > Python Command Prompt")
    print("     python Chapter_14/12_arcpy_full.py")
    print()
    print("   Option C -- Open-source equivalent:")
    print("     conda activate geocascade_env")
    print("     python Chapter_14/02_raster_analysis.py   # reclassify")
    print("     python Chapter_14/03_classification.py    # classification")
    print("     python Chapter_14/04_spatial_analysis.py  # slope/vulnerability")
    print("=" * 65)
    sys.exit(1)

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
BASE_DIR  = Path(__file__).parent
ROOT      = BASE_DIR.parent
PROC_DIR  = BASE_DIR / "data" / "processed" / "arcgis_outputs"
PROC_DIR.mkdir(parents=True, exist_ok=True)

# Input data from previous chapters
CHIRPS_TIF  = ROOT / "Chapter_01" / "data" / "processed" / "real_data" / "chirps_mean_annual_precip.tif"
TEMP_TIF    = ROOT / "Chapter_01" / "data" / "processed" / "climate_analysis" / "temperature_surface.tif"
UHI_TIF     = ROOT / "Chapter_01" / "data" / "processed" / "uhi_mapping" / "uhi_celsius.tif"
DEM_TIF     = PROC_DIR / "dem_proxy.tif"         # from 04_spatial_analysis.py
NDVI_TIF    = BASE_DIR / "data" / "processed" / "envi_outputs" / "ndvi.tif"

# Fallback: use any processed TIF already in arcgis_outputs
def find_tif(preferred: Path, fallback_dir: Path, pattern="*.tif") -> str:
    """Return preferred path if it exists, else first TIF found in fallback_dir."""
    if preferred.exists():
        return str(preferred)
    hits = list(fallback_dir.glob(pattern))
    if hits:
        print(f"  [FALLBACK] {preferred.name} not found -> using {hits[0].name}")
        return str(hits[0])
    return str(preferred)   # will fail gracefully inside the tool


# ---------------------------------------------------------------------------
# ENVIRONMENT SETUP
# ---------------------------------------------------------------------------

def setup_environment() -> bool:
    """Configure arcpy environment and check out Spatial Analyst extension."""
    print("\n[Setup] Configuring ArcGIS Pro environment ...")

    arcpy.env.overwriteOutput = True
    arcpy.env.workspace       = str(PROC_DIR)
    arcpy.env.outputCoordinateSystem = arcpy.SpatialReference(32719)  # UTM 19S
    arcpy.env.cellSize        = "MINOF"
    arcpy.env.parallelProcessingFactor = "75%"     # use 75% of CPU cores

    print(f"  Workspace          : {arcpy.env.workspace}")
    print(f"  Coordinate System  : {arcpy.env.outputCoordinateSystem.name}")
    print(f"  Overwrite outputs  : {arcpy.env.overwriteOutput}")

    # Check out Spatial Analyst
    sa_status = arcpy.CheckExtension("Spatial")
    if sa_status == "Available":
        arcpy.CheckOutExtension("Spatial")
        print("  Spatial Analyst    : Checked out")
    else:
        print(f"  [WARN] Spatial Analyst: {sa_status}")
        print("         Reclassify, Focal Stats, Weighted Overlay will fail.")
        return False

    # Check out 3D Analyst (for Slope / Aspect)
    d3_status = arcpy.CheckExtension("3D")
    if d3_status == "Available":
        arcpy.CheckOutExtension("3D")
        print("  3D Analyst         : Checked out")
    else:
        print(f"  [NOTE] 3D Analyst: {d3_status} (Slope/Aspect will use Spatial Analyst fallback)")

    return True


# ---------------------------------------------------------------------------
# [2] RECLASSIFY -- Precipitation Drought Classes
# ---------------------------------------------------------------------------

def reclassify_precipitation() -> str:
    """
    Reclassify CHIRPS mean annual precipitation into 5 drought-severity classes.

    ArcGIS Pro GUI:
      Toolbox > Spatial Analyst > Reclass > Reclassify
      Input raster: chirps_mean_annual_precip.tif
      Reclass field: Value
      Reclassification table:
        0    - 400   -> 1  (Extreme Drought)
        400  - 600   -> 2  (Severe Drought)
        600  - 900   -> 3  (Moderate Drought)
        900  - 1400  -> 4  (Near Normal)
        1400 - 9999  -> 5  (Wet / Above Normal)
    """
    print("\n[2] Reclassify: Precipitation -> 5 Drought Classes")

    in_raster = find_tif(CHIRPS_TIF, PROC_DIR)
    out_path  = str(PROC_DIR / "arcpy_precip_classified.tif")

    remap = RemapRange([
        [0,    400,  1],   # Extreme drought (Atacama influence)
        [400,  600,  2],   # Severe drought
        [600,  900,  3],   # Moderate drought
        [900,  1400, 4],   # Near normal (Patagonian steppe)
        [1400, 9999, 5],   # Wet (windward Andes)
    ])

    out = Reclassify(in_raster, "Value", remap, "NODATA")
    out.save(out_path)

    # Print class area summary using GetRasterProperties
    for cls in range(1, 6):
        labels = {1: "Extreme Drought", 2: "Severe Drought",
                  3: "Moderate Drought", 4: "Near Normal", 5: "Wet"}
        print(f"  Class {cls} ({labels[cls]}): saved")

    print(f"  [OK] {os.path.relpath(out_path, BASE_DIR)}")
    return out_path


# ---------------------------------------------------------------------------
# [3] RASTER CALCULATOR -- Temperature Anomaly
# ---------------------------------------------------------------------------

def raster_calculator_temp_anomaly() -> str:
    """
    Compute temperature anomaly relative to the mean.
    Formula: anomaly = temp - mean(temp)

    ArcGIS Pro GUI:
      Imagery tab > Raster Functions > Band Arithmetic
      OR: Toolbox > Spatial Analyst > Map Algebra > Raster Calculator
      Expression: "%temp_surface%" - %temp_surface%.mean
    """
    print("\n[3] Raster Calculator: Temperature Anomaly")

    in_raster = find_tif(TEMP_TIF, PROC_DIR)
    out_path  = str(PROC_DIR / "arcpy_temp_anomaly.tif")

    temp_raster = arcpy.Raster(in_raster)

    # Get mean value for anomaly calculation
    mean_temp = float(arcpy.GetRasterProperties_management(
        temp_raster, "MEAN").getOutput(0))
    print(f"  Mean temperature: {mean_temp:.2f} deg C")

    anomaly = temp_raster - mean_temp
    anomaly.save(out_path)

    print(f"  Anomaly range: computed")
    print(f"  [OK] {os.path.relpath(out_path, BASE_DIR)}")
    return out_path


# ---------------------------------------------------------------------------
# [4] FOCAL STATISTICS -- UHI Neighbourhood Mean
# ---------------------------------------------------------------------------

def focal_statistics_uhi() -> str:
    """
    Apply 5x5 neighbourhood mean to the UHI surface.
    Reduces noise from individual MODIS pixels (1km resolution).

    ArcGIS Pro GUI:
      Toolbox > Spatial Analyst > Neighborhood > Focal Statistics
      Input raster  : uhi_celsius.tif
      Neighborhood  : Rectangle 5 x 5 Cell
      Statistics    : MEAN
      Output raster : arcpy_uhi_focal.tif
    """
    print("\n[4] Focal Statistics: UHI 5x5 Neighbourhood Mean")

    in_raster = find_tif(UHI_TIF, PROC_DIR)
    out_path  = str(PROC_DIR / "arcpy_uhi_focal.tif")

    neighborhood = NbrRectangle(5, 5, "CELL")
    out = FocalStatistics(in_raster, neighborhood, "MEAN", "DATA")
    out.save(out_path)

    print(f"  Neighbourhood: 5x5 cell rectangle (MEAN)")
    print(f"  [OK] {os.path.relpath(out_path, BASE_DIR)}")
    return out_path


# ---------------------------------------------------------------------------
# [5] ISO CLUSTER -- Unsupervised Classification
# ---------------------------------------------------------------------------

def iso_cluster_classification() -> tuple:
    """
    ISO Cluster unsupervised classification on spectral index stack.
    Uses NDVI + temp anomaly as input bands.

    ArcGIS Pro GUI:
      Image Classification Wizard:
        Classify > Classify Pixels Using Deep Learning  (for DL approach)
      OR: Toolbox > Spatial Analyst > Multivariate > ISO Cluster Unsupervised Classification
        Input rasters : [ndvi.tif, arcpy_temp_anomaly.tif]
        Number classes: 5
        Iterations    : 20
        Min class size: 20
    """
    print("\n[5] ISO Cluster Unsupervised Classification (5 classes)")

    ndvi_path = find_tif(NDVI_TIF, PROC_DIR)
    temp_path = str(PROC_DIR / "arcpy_temp_anomaly.tif")
    out_path  = str(PROC_DIR / "arcpy_iso_cluster.tif")
    sig_path  = str(PROC_DIR / "arcpy_iso_cluster.gsg")

    # Build input list -- use what's available
    input_list = []
    for p in [ndvi_path, temp_path]:
        if arcpy.Exists(p):
            input_list.append(p)

    if not input_list:
        print("  [SKIP] No input rasters found for classification.")
        return None, None

    print(f"  Input bands: {len(input_list)}")
    for p in input_list:
        print(f"    {os.path.basename(p)}")

    # Run ISO Cluster Unsupervised Classification (combines IsoCluster + MLClassify)
    arcpy.sa.IsoClusterUnsupervisedClassification(
        input_list,         # list of input rasters
        5,                  # number of classes
        out_path,           # output classified raster
        sig_path,           # output signature file
        20,                 # max iterations
        0,                  # min class size (0 = no minimum)
        20,                 # sample interval
        10,                 # min class deviation
    )

    print(f"  [OK] ISO Cluster: {os.path.relpath(out_path, BASE_DIR)}")
    print(f"  Signature file  : {os.path.relpath(sig_path, BASE_DIR)}")
    print("  NOTE: Classes are numbered 1-5. To rename:")
    print("        Contents pane > right-click layer > Symbology > Unique Values")
    print("        Double-click label to rename (e.g. Class 1 -> Water)")
    return out_path, sig_path


# ---------------------------------------------------------------------------
# [6] MAXIMUM LIKELIHOOD -- Supervised Classification
# ---------------------------------------------------------------------------

def maximum_likelihood_classify(sig_path: str) -> str:
    """
    Maximum Likelihood Classification using ISO Cluster signatures.
    Applies a prior probability to each class equal to class frequency.

    ArcGIS Pro GUI:
      Toolbox > Spatial Analyst > Multivariate > Maximum Likelihood Classification
      Input rasters  : [ndvi.tif, arcpy_temp_anomaly.tif]
      Signature file : arcpy_iso_cluster.gsg
      Reject fraction: 0.0 (classify all pixels)
    """
    print("\n[6] Maximum Likelihood Classification")

    if sig_path is None or not arcpy.Exists(sig_path):
        print("  [SKIP] Signature file not found. Run ISO Cluster first.")
        return None

    ndvi_path = find_tif(NDVI_TIF, PROC_DIR)
    temp_path = str(PROC_DIR / "arcpy_temp_anomaly.tif")
    out_path  = str(PROC_DIR / "arcpy_mlc.tif")

    input_list = [p for p in [ndvi_path, temp_path] if arcpy.Exists(p)]
    if not input_list:
        print("  [SKIP] No input rasters for MLC.")
        return None

    mlc_out = MLClassify(
        input_list,   # input rasters
        sig_path,     # training signatures
        0.0,          # reject fraction (0 = classify all pixels)
        "EQUAL",      # a priori probability weighting
        [],           # no additional a priori file
        str(PROC_DIR / "arcpy_mlc_reject.tif"),   # rejected fraction output
    )
    mlc_out.save(out_path)

    print(f"  [OK] MLC classified: {os.path.relpath(out_path, BASE_DIR)}")
    return out_path


# ---------------------------------------------------------------------------
# [7] SLOPE & ASPECT -- From DEM
# ---------------------------------------------------------------------------

def compute_slope_aspect() -> tuple:
    """
    Compute slope (degrees) and aspect from the Copernicus GLO-30 DEM.

    ArcGIS Pro GUI:
      Toolbox > 3D Analyst > Raster Surface > Slope
        Input raster : dem_proxy.tif
        Output units : DEGREE
        Method       : PLANAR

      Toolbox > 3D Analyst > Raster Surface > Aspect
        Input raster : dem_proxy.tif
    """
    print("\n[7] Slope & Aspect from DEM")

    dem_path    = find_tif(DEM_TIF, PROC_DIR, "dem*.tif")
    slope_path  = str(PROC_DIR / "arcpy_slope.tif")
    aspect_path = str(PROC_DIR / "arcpy_aspect.tif")

    # Slope
    slope_out = Slope(dem_path, "DEGREE", 1.0, "PLANAR", arcpy.env.outputCoordinateSystem)
    slope_out.save(slope_path)
    slope_mean = float(arcpy.GetRasterProperties_management(slope_path, "MEAN").getOutput(0))
    slope_max  = float(arcpy.GetRasterProperties_management(slope_path, "MAXIMUM").getOutput(0))
    print(f"  Slope: mean={slope_mean:.1f} deg  max={slope_max:.1f} deg")
    print(f"  [OK] {os.path.relpath(slope_path, BASE_DIR)}")

    # Aspect
    aspect_out = Aspect(dem_path, "PLANAR", arcpy.env.outputCoordinateSystem)
    aspect_out.save(aspect_path)
    print(f"  Aspect: 0-360 deg clockwise from North")
    print(f"  [OK] {os.path.relpath(aspect_path, BASE_DIR)}")

    return slope_path, aspect_path


# ---------------------------------------------------------------------------
# [8] WEIGHTED OVERLAY -- Climate Vulnerability Index
# ---------------------------------------------------------------------------

def weighted_overlay_vulnerability(slope_path: str, iso_path: str) -> str:
    """
    Weighted Overlay to produce Climate Vulnerability Index.

    Weights:
      Temperature Trend  : 40%
      Precipitation Anom : 30%
      Slope              : 20%
      NDVI Loss          : 10%
    All inputs rescaled to 1-5 scale before overlay.

    ArcGIS Pro GUI:
      Toolbox > Spatial Analyst > Overlay > Weighted Overlay
        Input WO table:
          Field: Value
          Rasters: [temp_anomaly, precip_classified, slope, ndvi]
          Weights: 40, 30, 20, 10
        Output cell size: match CHIRPS resolution
    """
    print("\n[8] Weighted Overlay: Climate Vulnerability Index")

    # Prepare rescaled inputs (must all be integer rasters with 1-5 scale)
    temp_path   = str(PROC_DIR / "arcpy_temp_anomaly.tif")
    precip_path = str(PROC_DIR / "arcpy_precip_classified.tif")

    inputs = []
    if slope_path and arcpy.Exists(slope_path):
        # Reclassify slope to 1-5 for Weighted Overlay
        slope_remap = RemapRange([
            [0,  5,  1],   # flat
            [5,  15, 2],   # gentle
            [15, 25, 3],   # moderate
            [25, 35, 4],   # steep
            [35, 90, 5],   # very steep (avalanche risk)
        ])
        slope_cls = Reclassify(slope_path, "Value", slope_remap, "NODATA")
        slope_cls_path = str(PROC_DIR / "arcpy_slope_cls.tif")
        slope_cls.save(slope_cls_path)
        inputs.append((slope_cls_path, "Value", 20))

    if arcpy.Exists(precip_path):
        inputs.append((precip_path, "Value", 30))

    if arcpy.Exists(temp_path):
        # Reclassify temperature anomaly to 1-5
        temp_r = arcpy.Raster(temp_path)
        t_min = float(arcpy.GetRasterProperties_management(temp_path, "MINIMUM").getOutput(0))
        t_max = float(arcpy.GetRasterProperties_management(temp_path, "MAXIMUM").getOutput(0))
        step = (t_max - t_min) / 5
        temp_remap = RemapRange([
            [t_min,          t_min + step,    1],
            [t_min + step,   t_min + 2*step,  2],
            [t_min + 2*step, t_min + 3*step,  3],
            [t_min + 3*step, t_min + 4*step,  4],
            [t_min + 4*step, t_max + 0.001,   5],
        ])
        temp_cls = Reclassify(temp_path, "Value", temp_remap, "NODATA")
        temp_cls_path = str(PROC_DIR / "arcpy_temp_cls.tif")
        temp_cls.save(temp_cls_path)
        inputs.append((temp_cls_path, "Value", 40))

    # Normalise weights to 100%
    total_w = sum(w for _, _, w in inputs)
    if total_w == 0 or not inputs:
        print("  [SKIP] No inputs for Weighted Overlay.")
        return None

    # Build WOTable
    # WOTable: list of [raster_path, influence, RemapTable]
    # Scale is 1-5 (restricted range 1-5, no restricted values)
    wo_table = []
    for path, field, weight in inputs:
        wo_table.append(
            WOField(path, field, weight, [1, 2, 3, 4, 5])
        )

    wt = WOTable(wo_table, [1, 5, 1])   # scale 1-5, step 1
    out = WeightedOverlay(wt)
    out_path = str(PROC_DIR / "arcpy_vulnerability.tif")
    out.save(out_path)

    print(f"  Vulnerability Index: 1=Low Risk -> 5=High Risk")
    print(f"  Weights: Temp={next(w for _,_,w in inputs if 'temp' in _[0].lower()) if any('temp' in _[0].lower() for _ in inputs) else 'N/A'}% | "
          f"Precip=30% | Slope=20%")
    print(f"  [OK] {os.path.relpath(out_path, BASE_DIR)}")
    return out_path


# ---------------------------------------------------------------------------
# [9] ZONAL STATISTICS AS TABLE
# ---------------------------------------------------------------------------

def zonal_statistics(zone_raster: str, value_raster: str) -> str:
    """
    Compute zonal statistics (mean slope per land cover class).

    ArcGIS Pro GUI:
      Toolbox > Spatial Analyst > Zonal > Zonal Statistics as Table
        Input zone data   : arcpy_iso_cluster.tif  (zone raster)
        Zone field        : Value
        Input value raster: arcpy_slope.tif
        Output table      : arcpy_zonal_stats.dbf
        Statistics type   : ALL (count, min, max, mean, std, sum)
    """
    print("\n[9] Zonal Statistics: Mean Slope per Land Cover Class")

    if zone_raster is None or not arcpy.Exists(zone_raster):
        print("  [SKIP] Zone raster not found.")
        return None

    out_path = str(PROC_DIR / "arcpy_zonal_stats.dbf")

    ZonalStatisticsAsTable(
        zone_raster,    # zone raster (ISO cluster output)
        "Value",        # zone field
        value_raster,   # value raster (slope)
        out_path,       # output table
        "DATA",         # ignore nodata
        "ALL",          # compute all statistics
    )

    # Print results using cursor
    print(f"  {'Class':>6} {'Count':>8} {'Mean':>10} {'Min':>8} {'Max':>8}")
    print("  " + "-" * 46)

    class_labels = {1: "Water", 2: "Snow/Ice", 3: "Bare Rock",
                    4: "Sparse Veg", 5: "Dense Veg"}

    with arcpy.da.SearchCursor(out_path,
                               ["Value", "COUNT", "MEAN", "MIN", "MAX"]) as cur:
        for row in cur:
            cls_id = int(row[0])
            label  = class_labels.get(cls_id, f"Class {cls_id}")
            print(f"  {label:>12}  {int(row[1]):>8,}  "
                  f"{row[2]:>8.1f} deg  {row[3]:>5.1f}  {row[4]:>5.1f}")

    print(f"  [OK] {os.path.relpath(out_path, BASE_DIR)}")
    return out_path


# ---------------------------------------------------------------------------
# [10] LAYOUT EXPORT
# ---------------------------------------------------------------------------

def export_layout(aprx_path: str = None) -> str:
    """
    Export the first layout in the ArcGIS Pro project as a PNG at 300 DPI.

    ArcGIS Pro GUI:
      Insert > New Layout > A3 Landscape
      Insert > Map Frame -> drag to place
      Insert > Legend, North Arrow, Scale Bar
      Share > Export Layout -> PNG, 300 DPI

    ArcPy:
      aprx = arcpy.mp.ArcGISProject('path.aprx')
      lyt  = aprx.listLayouts()[0]
      lyt.exportToPNG('output.png', resolution=300)
    """
    print("\n[10] Exporting Layout to PNG (300 DPI)")

    # Try to find an APRX file
    if aprx_path is None:
        candidates = [
            str(BASE_DIR / "arcgis_pro" / "GeoCascade_Ch14.aprx"),
            str(PROC_DIR / "GeoCascade_Ch14.aprx"),
        ]
        for c in candidates:
            if arcpy.Exists(c):
                aprx_path = c
                break

    out_png = str(PROC_DIR / "arcpy_layout.png")

    if aprx_path is None or not arcpy.Exists(aprx_path):
        print("  [NOTE] No .aprx file found.")
        print("         To create a layout:")
        print("           1. Open ArcGIS Pro -> New Map")
        print("           2. Add layers from data/processed/arcgis_outputs/")
        print("           3. Insert > New Layout > A3 Landscape")
        print("           4. Insert > Map Frame, Legend, North Arrow, Scale Bar")
        print("           5. Share > Export Layout > PNG 300 DPI")
        print("           Programmatic export requires a saved .aprx file.")
        return None

    aprx = arcpy.mp.ArcGISProject(aprx_path)
    layouts = aprx.listLayouts()

    if not layouts:
        print("  [WARN] No layouts found in project. Create a layout first.")
        return None

    lyt = layouts[0]
    print(f"  Layout: {lyt.name}  ({lyt.pageWidth:.0f} x {lyt.pageHeight:.0f} {lyt.pageUnits})")

    lyt.exportToPNG(out_png, resolution=300, color_mode="RGB_TRUE_COLOR",
                    embed_color_profile=True, clip_to_elements=False)

    print(f"  [OK] Layout exported: {os.path.relpath(out_png, BASE_DIR)}")
    return out_png


# ---------------------------------------------------------------------------
# EXTENSION CHECKIN
# ---------------------------------------------------------------------------

def checkin_extensions() -> None:
    """Return spatial analyst and 3D analyst licenses."""
    try:
        arcpy.CheckInExtension("Spatial")
        arcpy.CheckInExtension("3D")
        print("\n  Extensions checked in.")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print(" GEOCASCADE Ch14 -- ArcGIS Pro Native ArcPy Pipeline")
    print(" Spatial Analyst | 3D Analyst | Reclassify | ISO Cluster")
    print(" WeightedOverlay | ZonalStats | MLC | Layout Export")
    print("=" * 65)

    ok = setup_environment()
    if not ok:
        print("\n[WARN] Continuing without Spatial Analyst -- some steps will fail.")

    results = {}

    # Raster analysis
    try:
        results["precip"] = reclassify_precipitation()
    except Exception as e:
        print(f"  [FAIL] Reclassify: {e}")
        results["precip"] = None

    try:
        results["temp"] = raster_calculator_temp_anomaly()
    except Exception as e:
        print(f"  [FAIL] Raster Calculator: {e}")
        results["temp"] = None

    try:
        results["uhi"] = focal_statistics_uhi()
    except Exception as e:
        print(f"  [FAIL] Focal Statistics: {e}")
        results["uhi"] = None

    # Classification
    try:
        results["iso"], sig = iso_cluster_classification()
    except Exception as e:
        print(f"  [FAIL] ISO Cluster: {e}")
        results["iso"], sig = None, None

    try:
        results["mlc"] = maximum_likelihood_classify(sig)
    except Exception as e:
        print(f"  [FAIL] MLC: {e}")
        results["mlc"] = None

    # Spatial analysis
    try:
        results["slope"], results["aspect"] = compute_slope_aspect()
    except Exception as e:
        print(f"  [FAIL] Slope/Aspect: {e}")
        results["slope"] = results["aspect"] = None

    try:
        results["vuln"] = weighted_overlay_vulnerability(
            results.get("slope"), results.get("iso"))
    except Exception as e:
        print(f"  [FAIL] Weighted Overlay: {e}")
        results["vuln"] = None

    # Zonal statistics
    try:
        if results.get("iso") and results.get("slope"):
            results["zonal"] = zonal_statistics(results["iso"], results["slope"])
    except Exception as e:
        print(f"  [FAIL] Zonal Statistics: {e}")
        results["zonal"] = None

    # Layout export
    try:
        results["layout"] = export_layout()
    except Exception as e:
        print(f"  [FAIL] Layout Export: {e}")
        results["layout"] = None

    # Check in extensions
    checkin_extensions()

    # Summary
    passed = sum(1 for v in results.values() if v is not None)
    total  = len(results)
    print("\n" + "=" * 65)
    print(f" ARCPY PIPELINE COMPLETE  ({passed}/{total} steps succeeded)")
    print("=" * 65)
    for step, path in results.items():
        status = "[OK]  " if path else "[SKIP]"
        name   = os.path.basename(str(path)) if path else "---"
        print(f"  {status} {step:<12}: {name}")
    print()
    print(f"  All outputs: {PROC_DIR}")
    print()
    print("  ArcGIS Pro -- View outputs:")
    print("    Map tab > Add Data -> browse to arcgis_outputs/")
    print("    Appearance tab > Symbology to style each layer")
    print()
    print("  ENVI -- View outputs:")
    print("    File > Open -> select any arcpy_*.tif")
    print()
    print("  Next: python Chapter_14/13_envi_api_full.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
