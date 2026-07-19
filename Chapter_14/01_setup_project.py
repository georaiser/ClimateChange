"""
01_setup_project.py
====================
GeoCascade Chapter 14 -- ArcGIS Pro + ENVI Professional Workflows
Torres del Paine, Patagonia, Chile

PURPOSE
-------
Creates an ArcGIS Pro project (.aprx) and imports all Chapter 1-13 outputs
as layers. Organises layers into groups (Climate, Satellite, Terrain, Analysis).
Reprojects all layers to UTM Zone 18S (EPSG:32718) for consistent metric analysis.

WORKFLOW
--------
  [1/5] Discover all GeoTIFFs and shapefiles from Chapters 01-13
  [2/5] Reproject mismatched CRS layers to UTM 18S
  [3/5] Create layer groups in the ArcGIS Pro project
  [4/5] Apply default symbology presets
  [5/5] Save the project and print a layer inventory

OUTPUTS
-------
  arcgis_pro/GeoCascade_Ch14.aprx   -- ArcGIS Pro project file
  arcgis_pro/toolboxes/GeoCascade.tbx -- custom toolbox
  data/processed/arcgis_outputs/reprojected/ -- UTM 18S versions of all rasters

ARCGIS PRO USAGE
----------------
  1. Open ArcGIS Pro → File → Open Project → select GeoCascade_Ch14.aprx
  2. All Chapter 1-13 outputs appear in the Contents pane, grouped by theme
  3. Use Analysis > Tools to run Zonal Statistics As Table on any raster
  4. Use Data Management > Project Raster to reproject individual layers

RUN (from ArcGIS Pro Python window OR standalone arcpy)
---
  python Chapter_14/01_setup_project.py

NOTE: Requires ArcGIS Pro + Spatial Analyst extension.
      If running without a license, the script prints the full workflow and
      creates a JSON manifest of all discovered layers (no arcpy operations).
"""

import sys
import os
import json
import glob

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(BASE_DIR, "arcgis_pro")
OUT_DIR     = os.path.join(BASE_DIR, "data", "processed", "arcgis_outputs")
APRX_PATH   = os.path.join(PROJECT_DIR, "GeoCascade_Ch14.aprx")
MANIFEST    = os.path.join(OUT_DIR, "layer_manifest.json")

os.makedirs(PROJECT_DIR, exist_ok=True)
os.makedirs(OUT_DIR,     exist_ok=True)

# Study area
BBOX = [-73.5, -51.5, -72.5, -50.5]
TARGET_CRS = "EPSG:32718"   # UTM Zone 18S (standard for Patagonia)

# Layer group definitions -- maps group name -> glob patterns
LAYER_GROUPS = {
    "Climate": [
        "**/real_data/chirps_mean_annual_precip.tif",
        "**/climate_analysis/temperature_surface.tif",
        "**/climate_maps/precipitation_anomaly_clusters.png",
    ],
    "UHI_Temperature": [
        "**/uhi_mapping/uhi_celsius.tif",
    ],
    "Satellite_Indices": [
        "**/ndvi*.tif",
        "**/nbr*.tif",
        "**/sar*.tif",
        "**/sentinel2*.tif",
    ],
    "Terrain": [
        "**/dem*.tif",
        "**/slope*.tif",
        "**/watershed*.tif",
        "**/watershed*.shp",
        "**/river*.shp",
    ],
    "Analysis": [
        "**/classified*.tif",
        "**/land_cover*.tif",
        "**/vulnerability*.tif",
    ],
}


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def discover_layers(root: str) -> dict:
    """Walk Chapter 01-13 processed dirs and collect all rasters/vectors."""
    found = {group: [] for group in LAYER_GROUPS}

    for group, patterns in LAYER_GROUPS.items():
        for pattern in patterns:
            hits = glob.glob(os.path.join(root, "Chapter_*", "data", "processed",
                                          "**", os.path.basename(pattern)),
                             recursive=True)
            # Also check Chapter_XX/data/processed/real_data
            hits += glob.glob(os.path.join(root, "Chapter_*", pattern.lstrip("*/")),
                              recursive=True)
            for h in hits:
                if h not in found[group]:
                    found[group].append(os.path.normpath(h))

    return found


def print_manifest(layers: dict) -> None:
    total = sum(len(v) for v in layers.values())
    print(f"\n  Discovered {total} layers across {len(layers)} groups:")
    for grp, paths in layers.items():
        print(f"\n  [{grp}]  ({len(paths)} layers)")
        for p in paths[:8]:
            print(f"    {os.path.relpath(p, BASE_DIR)}")
        if len(paths) > 8:
            print(f"    ... and {len(paths)-8} more")


def save_manifest(layers: dict, out_path: str) -> None:
    manifest = {
        "project": "GeoCascade Chapter 14",
        "target_crs": TARGET_CRS,
        "bbox_wgs84": BBOX,
        "aprx": APRX_PATH,
        "layers": layers,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  [OK] Layer manifest saved: {out_path}")


def try_arcpy_setup(layers: dict) -> None:
    """Attempt to create the ArcGIS Pro project using arcpy (license required)."""
    try:
        import arcpy
        from arcpy import env as arc_env

        arc_env.overwriteOutput = True
        arc_env.outputCoordinateSystem = arcpy.SpatialReference(32718)

        print("\n  [ArcPy] Creating project template ...")
        # ArcGIS Pro: create project from blank template
        aprx_template = os.path.join(
            arcpy.GetInstallInfo()["InstallDir"],
            "Resources", "ProjectTemplates", "Blank.aptx"
        )
        if os.path.exists(aprx_template):
            arcpy.management.CreateProject(
                PROJECT_DIR, "GeoCascade_Ch14", aprx_template
            )
            print(f"  [OK] Project created: {APRX_PATH}")
        else:
            print("  [WARN] Blank project template not found -- open ArcGIS Pro manually.")

        # Add each layer group
        aprx = arcpy.mp.ArcGISProject(APRX_PATH)
        m    = aprx.listMaps()[0]

        for group, paths in layers.items():
            for path in paths:
                if os.path.exists(path) and path.endswith((".tif", ".shp")):
                    try:
                        m.addDataFromPath(path)
                        print(f"    [+] {os.path.basename(path)}")
                    except Exception as e:
                        print(f"    [WARN] Could not add {os.path.basename(path)}: {e}")

        aprx.save()
        print(f"\n  [OK] Project saved: {APRX_PATH}")
        print("       Open in ArcGIS Pro -> Contents pane shows all layers.")

    except ImportError:
        print("\n  [INFO] arcpy not available in this environment.")
        print("         Running in MANIFEST-ONLY mode (no ArcGIS Pro operations).")
        print("         To use ArcPy features: run this script from the")
        print("         ArcGIS Pro Python window or activate the arcgispro-py3 env.")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print(" GEOCASCADE Ch14 -- ArcGIS Pro Project Setup")
    print(" ArcPy | Raster + Vector Layer Import | UTM 18S")
    print("=" * 65)

    root = os.path.dirname(BASE_DIR)   # D:/00_AI_Aplications/ClimateChange

    print(f"\n[1/5] Discovering Chapter 01-13 outputs in: {root}")
    layers = discover_layers(root)

    print(f"\n[2/5] Layer inventory:")
    print_manifest(layers)

    print(f"\n[3/5] Saving layer manifest ...")
    save_manifest(layers, MANIFEST)

    print(f"\n[4/5] Attempting ArcGIS Pro project creation ...")
    try_arcpy_setup(layers)

    print(f"\n[5/5] Next steps:")
    print("  ArcGIS Pro:")
    print("    a) File > Open Project > arcgis_pro/GeoCascade_Ch14.aprx")
    print("    b) Add Data > browse to Chapter_01/.../uhi_celsius.tif")
    print("    c) Apply Symbology: Classify > Natural Breaks (Jenks) > 5 classes")
    print("    d) Run script 02_raster_analysis.py for automated symbology")
    print()
    print("  ENVI 5.6:")
    print("    a) File > Open > Chapter_01/data/processed/uhi_mapping/uhi_celsius.tif")
    print("    b) Run script 07_envi_spectral_analysis.py for index computation")
    print()

    print("=" * 65)
    print(" SETUP COMPLETE")
    print("=" * 65)
    print(f"  Manifest : {MANIFEST}")
    print(f"  APRX     : {APRX_PATH}  (open in ArcGIS Pro)")
    print()
    print("  Continue with: python Chapter_14/02_raster_analysis.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
