"""
01_arcpy_import_imagery.py
==========================
GeoCascade Chapter 01 - ArcGIS Pro Data Import Script
------------------------------------------------------
Imports and mosaics Chapter_01 geospatial datasets into the active ArcGIS Pro
project geodatabase and adds all layers to the active map frame.

Workflow:
  1. Mosaic 4 Copernicus DEM 30m tiles -> single DEM raster in project GDB
  2. Composite Sentinel-2 bands (B02/B03/B04/B08) -> 4-band raster dataset
  3. Import CHIRPS mean annual precipitation GeoTIFF as raster layer
  4. Import ERA5 trend summary CSV as table view
  5. Import RGI 7.0 glacier outlines GeoPackage as feature class

Usage:
  Run from ArcGIS Pro Notebook (Analysis > Python Notebook)
  OR paste into ArcGIS Pro Python console (Analysis > Python Window).

Requirements:
  ArcGIS Pro 3.x with Spatial Analyst + 3D Analyst extensions.
  Project must be open and saved at least once.

Author : GeoCascade Project
Date   : 2026-07-14
Study  : Torres del Paine, Patagonia, Chile
BBOX   : [-73.5, -51.5, -72.5, -50.5]
"""

import arcpy
# This script runs inside ArcGIS Pro Notebook or Python console
# arcpy is pre-installed in ArcGIS Pro's Python environment
# Do NOT run from geocascade_env -- run from ArcGIS Pro only

import os
import sys
import glob

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

# Resolve Chapter_01 root relative to this script file.
# When running in a Notebook, __file__ may not be defined; fall back to a
# hard-coded path if needed.
try:
    SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
    CHAPTER_ROOT = os.path.dirname(SCRIPT_DIR)          # Chapter_01/
except NameError:
    # Running interactively in ArcGIS Pro Python Window
    CHAPTER_ROOT = r"D:\00_AI_Aplications\ClimateChange\Chapter_01"

RAW_DIR       = os.path.join(CHAPTER_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(CHAPTER_ROOT, "data", "processed")

# Input paths
DEM_PATTERN      = os.path.join(RAW_DIR, "dem_*", "copernicus_dem_30m.tif")
S2_PATTERN_ROOT  = os.path.join(RAW_DIR, "sentinel2_*")
CHIRPS_TIF       = os.path.join(PROCESSED_DIR, "real_data", "chirps_mean_annual_precip.tif")
ERA5_TREND_CSV   = os.path.join(PROCESSED_DIR, "climate_analysis", "trend_summary.csv")
GLACIER_GPKG     = os.path.join(RAW_DIR, "real_data", "rgi70_patagonia_glaciers.gpkg")

# Output names inside project GDB
OUT_DEM_NAME     = "copernicus_dem_mosaic"
OUT_S2_NAME      = "sentinel2_composite_4band"
OUT_CHIRPS_NAME  = "chirps_mean_annual_precip"
OUT_GLACIER_NAME = "rgi70_patagonia_glaciers"
ERA5_TABLE_NAME  = "era5_trend_summary"

# Spatial reference - UTM Zone 19S for metric analysis
SR_UTM19S = arcpy.SpatialReference(32719)

# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def print_step(n, msg):
    """Print a numbered step header to the console."""
    print("\n[STEP {}] {}".format(n, msg))
    print("-" * 60)


def check_extension(ext_name):
    """Check out an ArcGIS extension; warn if unavailable."""
    status = arcpy.CheckExtension(ext_name)
    if status == "Available":
        arcpy.CheckOutExtension(ext_name)
        print("  [OK] Extension checked out: {}".format(ext_name))
    else:
        print("  [WARN] Extension not available: {} (status={})".format(ext_name, status))


def add_layer_to_map(aprx, layer_path, layer_name=None):
    """
    Add a raster or feature layer to the first map in the ArcGIS Pro project.
    Returns the newly added layer object.
    """
    try:
        active_map = aprx.activeMap
        if active_map is None:
            active_map = aprx.listMaps()[0]
        lyr = active_map.addDataFromPath(layer_path)
        if layer_name and lyr:
            lyr.name = layer_name
        print("  [OK] Added to map: {}".format(layer_name or layer_path))
        return lyr
    except Exception as e:
        print("  [WARN] Could not add layer '{}': {}".format(layer_path, e))
        return None


def safe_delete(path):
    """Delete an existing raster/feature class if it exists."""
    if arcpy.Exists(path):
        arcpy.management.Delete(path)
        print("  Deleted existing: {}".format(path))


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("GeoCascade Ch01 - ArcGIS Pro Data Import")
    print("Chapter root: {}".format(CHAPTER_ROOT))
    print("=" * 60)

    # ------------------------------------------------------------------
    # Project / GDB setup
    # ------------------------------------------------------------------
    try:
        aprx = arcpy.mp.ArcGISProject("CURRENT")
    except Exception:
        raise RuntimeError(
            "No active ArcGIS Pro project found. "
            "Open or create a project before running this script."
        )

    gdb = aprx.defaultGeodatabase
    if not gdb:
        raise RuntimeError(
            "Project has no default geodatabase. "
            "Please set one via Project > Properties > Default Geodatabase."
        )
    print("Default GDB: {}".format(gdb))

    arcpy.env.workspace       = gdb
    arcpy.env.overwriteOutput = True

    # Check out extensions
    check_extension("Spatial")
    check_extension("3D")

    # ------------------------------------------------------------------
    # STEP 1: Mosaic Copernicus DEM tiles -> single DEM raster
    # ------------------------------------------------------------------
    print_step(1, "Mosaicking Copernicus DEM tiles")

    dem_tiles = sorted(glob.glob(DEM_PATTERN))
    if not dem_tiles:
        print("  [WARN] No DEM tiles found at pattern: {}".format(DEM_PATTERN))
        print("  Skipping DEM mosaic.")
    else:
        print("  Found {} DEM tile(s):".format(len(dem_tiles)))
        for t in dem_tiles:
            print("    {}".format(t))

        out_dem = os.path.join(gdb, OUT_DEM_NAME)
        safe_delete(out_dem)

        # MosaicToNewRaster: pixel_type=32_BIT_FLOAT, 1 band, LAST mosaic method
        arcpy.management.MosaicToNewRaster(
            input_rasters                      = ";".join(dem_tiles),
            output_location                    = gdb,
            raster_dataset_name_with_extension = OUT_DEM_NAME,
            coordinate_system_for_the_raster   = SR_UTM19S,
            pixel_type                         = "32_BIT_FLOAT",
            cellsize                           = 30,
            number_of_bands                    = 1,
            mosaic_method                      = "LAST",
            mosaic_colormap_mode               = "FIRST"
        )
        print("  [OK] DEM mosaic created: {}".format(out_dem))

        # Ensure projection is set correctly
        arcpy.management.DefineProjection(out_dem, SR_UTM19S)

        add_layer_to_map(aprx, out_dem, "Copernicus DEM 30m Mosaic")

    # ------------------------------------------------------------------
    # STEP 2: Composite Sentinel-2 bands -> 4-band raster
    # ------------------------------------------------------------------
    print_step(2, "Creating Sentinel-2 4-band composite (B02/B03/B04/B08)")

    s2_dirs = sorted(glob.glob(S2_PATTERN_ROOT))
    if not s2_dirs:
        print("  [WARN] No Sentinel-2 directories found at: {}".format(S2_PATTERN_ROOT))
        print("  Skipping Sentinel-2 composite.")
    else:
        # Use the first scene directory found
        s2_dir = s2_dirs[0]
        print("  Using Sentinel-2 scene directory: {}".format(s2_dir))

        band_files = []
        for band in ["B02", "B03", "B04", "B08"]:
            candidate = os.path.join(s2_dir, "{}.tif".format(band))
            if os.path.exists(candidate):
                band_files.append(candidate)
                print("  Found band: {}".format(candidate))
            else:
                print("  [WARN] Band file not found: {}".format(candidate))

        if len(band_files) == 4:
            out_s2 = os.path.join(gdb, OUT_S2_NAME)
            safe_delete(out_s2)

            # CompositeBands stacks in order: Band1=B02, Band2=B03,
            # Band3=B04, Band4=B08
            arcpy.management.CompositeBands(
                in_rasters = ";".join(band_files),
                out_raster = out_s2
            )
            print("  [OK] Sentinel-2 composite created: {}".format(out_s2))
            print("  Band order: B02(Blue) | B03(Green) | B04(Red) | B08(NIR)")

            add_layer_to_map(aprx, out_s2, "Sentinel-2 Composite (B02/B03/B04/B08)")
        else:
            print(
                "  [WARN] Only {}/4 band files found; skipping composite.".format(
                    len(band_files)
                )
            )

    # ------------------------------------------------------------------
    # STEP 3: Import CHIRPS mean annual precipitation TIF
    # ------------------------------------------------------------------
    print_step(3, "Importing CHIRPS mean annual precipitation raster")

    if not os.path.exists(CHIRPS_TIF):
        print("  [WARN] CHIRPS TIF not found: {}".format(CHIRPS_TIF))
        print("  Run the CHIRPS processing script first.")
    else:
        out_chirps = os.path.join(gdb, OUT_CHIRPS_NAME)
        safe_delete(out_chirps)

        arcpy.management.CopyRaster(
            in_raster                   = CHIRPS_TIF,
            out_rasterdataset           = out_chirps,
            config_keyword              = "",
            background_value            = "",
            nodata_value                = "-9999",
            onebit_to_eightbit          = "NONE",
            colormap_to_RGB             = "NONE",
            pixel_type                  = "32_BIT_FLOAT",
            scale_pixel_value           = "NONE",
            RGB_to_Colormap             = "NONE",
            format                      = "GDB"
        )
        print("  [OK] CHIRPS raster imported: {}".format(out_chirps))

        add_layer_to_map(aprx, out_chirps, "CHIRPS Mean Annual Precipitation")

    # ------------------------------------------------------------------
    # STEP 4: Import ERA5 trend summary CSV as geodatabase table
    # ------------------------------------------------------------------
    print_step(4, "Importing ERA5 trend summary CSV as geodatabase table")

    if not os.path.exists(ERA5_TREND_CSV):
        print("  [WARN] ERA5 trend CSV not found: {}".format(ERA5_TREND_CSV))
        print("  Run the ERA5 analysis script first.")
    else:
        out_era5_table = os.path.join(gdb, ERA5_TABLE_NAME)
        safe_delete(out_era5_table)

        arcpy.conversion.TableToTable(
            in_rows        = ERA5_TREND_CSV,
            out_path       = gdb,
            out_name       = ERA5_TABLE_NAME,
            where_clause   = "",
            field_mapping  = "",
            config_keyword = ""
        )
        print("  [OK] ERA5 trend table created: {}".format(out_era5_table))

        # Add standalone table to Contents pane
        try:
            active_map = aprx.activeMap or aprx.listMaps()[0]
            active_map.addDataFromPath(out_era5_table)
            print("  [OK] ERA5 trend table added to Contents pane.")
        except Exception as e:
            print("  [NOTE] Could not add table to map pane: {}".format(e))

    # ------------------------------------------------------------------
    # STEP 5: Import RGI 7.0 Glacier Outlines GeoPackage
    # ------------------------------------------------------------------
    print_step(5, "Importing RGI 7.0 glacier outlines from GeoPackage")

    if not os.path.exists(GLACIER_GPKG):
        print("  [WARN] Glacier GeoPackage not found: {}".format(GLACIER_GPKG))
        print("  Expected: {}".format(GLACIER_GPKG))
    else:
        # List feature classes inside the GeoPackage
        arcpy.env.workspace = GLACIER_GPKG
        gpkg_layers = arcpy.ListFeatureClasses()
        arcpy.env.workspace = gdb          # restore workspace

        if not gpkg_layers:
            print("  [WARN] No feature classes found in GeoPackage.")
        else:
            src_fc_name  = gpkg_layers[0]
            src_path     = os.path.join(GLACIER_GPKG, src_fc_name)
            out_glaciers = os.path.join(gdb, OUT_GLACIER_NAME)
            safe_delete(out_glaciers)

            arcpy.management.CopyFeatures(
                in_features       = src_path,
                out_feature_class = out_glaciers
            )
            feat_count = int(arcpy.management.GetCount(out_glaciers)[0])
            print("  [OK] Glacier FC created: {}".format(out_glaciers))
            print("  Source layer: {} ({} features)".format(src_fc_name, feat_count))

            add_layer_to_map(aprx, out_glaciers, "RGI 7.0 Glacier Outlines")

    # ------------------------------------------------------------------
    # Save project
    # ------------------------------------------------------------------
    print_step(6, "Saving ArcGIS Pro project")
    try:
        aprx.save()
        print("  [OK] Project saved: {}".format(aprx.filePath))
    except Exception as e:
        print("  [WARN] Could not save project: {}".format(e))

    print("\n" + "=" * 60)
    print("Import complete. Check Contents pane for new layers.")
    print("=" * 60)


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except arcpy.ExecuteError:
        msgs = arcpy.GetMessages(2)
        print("\n[ARCPY ERROR]\n{}".format(msgs))
        raise
    except Exception as exc:
        import traceback
        print("\n[ERROR] {}\n{}".format(exc, traceback.format_exc()))
        raise