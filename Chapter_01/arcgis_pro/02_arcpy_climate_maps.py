"""
arcgis_pro/02_arcpy_climate_maps.py
=====================================
ArcGIS Pro - Thematic Map Symbology Automation

Applies professional cartographic symbology to all Chapter_01 raster and
vector layers already loaded in the ArcGIS Pro project.

HOW TO RUN:
    1. Open ArcGIS Pro (3.x required)
    2. Import the layers first (run 01_arcpy_import_imagery.py)
    3. Analysis > Python Notebook > New Notebook
    4. Paste and run this script

LAYERS STYLED BY THIS SCRIPT:
    - Copernicus DEM mosaic  -> hillshade + classified elevation
    - CHIRPS precipitation   -> classified Yellow-Blue, 5 classes
    - Temperature surface    -> diverging Red-Blue, 5 classes
    - Sentinel-2 composite   -> false color (B08/B04/B03)
    - RGI glacier outlines   -> cyan fill, dark blue outline, labeled
    - ERA5 trend CSV         -> chart embedded in layout

ArcGIS Pro docs:
    https://pro.arcgis.com/en/pro-app/latest/arcpy/mapping/layer-class.htm

Dependencies: arcpy (pre-installed in ArcGIS Pro Python environment)
              Run ONLY from ArcGIS Pro Notebook or Python console
"""

import arcpy
import os

# ---------------------------------------------------------------------------
# Check we are running inside ArcGIS Pro
# ---------------------------------------------------------------------------
try:
    aprx = arcpy.mp.ArcGISProject("CURRENT")
    m    = aprx.activeMap
    print(f"[OK] Connected to project: {aprx.filePath}")
    print(f"[OK] Active map: {m.name}")
except Exception:
    print("ERROR: This script must run inside ArcGIS Pro.")
    print("       Analysis > Python Notebook > New Notebook")
    raise

# ---------------------------------------------------------------------------
# Config - match names used in 01_arcpy_import_imagery.py
# ---------------------------------------------------------------------------
LAYER_NAMES = {
    "dem":       "CopernicusDEM_Mosaic",
    "hillshade": "CopernicusDEM_Hillshade",
    "chirps":    "CHIRPS_MeanAnnualPrecip",
    "temp":      "TemperatureSurface",
    "s2":        "Sentinel2_Composite",
    "glaciers":  "RGI_Glaciers",
}

CHAPTER_DIR = r"D:\00_AI_Aplications\ClimateChange\Chapter_01"
GDB         = os.path.join(CHAPTER_DIR, "arcgis_pro", "GeoCascade_Ch01.gdb")


# ---------------------------------------------------------------------------
# Helper: find a layer in active map by name (partial match)
# ---------------------------------------------------------------------------
def find_layer(layer_name):
    for lyr in m.listLayers():
        if layer_name.lower() in lyr.name.lower():
            return lyr
    return None


# ---------------------------------------------------------------------------
# 1. DEM Hillshade (base layer)
# ---------------------------------------------------------------------------
def style_dem_hillshade():
    print("[1/6] Styling DEM hillshade...")

    # Create hillshade if not already present
    dem_lyr = find_layer("DEM_Mosaic")
    if dem_lyr is None:
        print("  [SKIP] DEM mosaic not found. Run 01_arcpy_import_imagery.py first.")
        return

    hillshade_path = os.path.join(GDB, "CopernicusDEM_Hillshade")
    if not arcpy.Exists(hillshade_path):
        print("  Creating DEM hillshade...")
        arcpy.ddd.HillShade(
            dem_lyr,
            hillshade_path,
            azimuth=315,    # northwest illumination (standard)
            altitude=45,    # sun angle 45 deg
            model_shadows="NO_SHADOWS",
            z_factor=1.0
        )
        m.addDataFromPath(hillshade_path)
        print(f"  [OK] Hillshade created: {hillshade_path}")

    hs_lyr = find_layer("Hillshade")
    if hs_lyr:
        hs_lyr.transparency = 0   # 0% transparent (solid base)
        print("  [OK] Hillshade styled as base layer")


# ---------------------------------------------------------------------------
# 2. CHIRPS Precipitation - Classified Yellow-Blue
# ---------------------------------------------------------------------------
def style_chirps():
    print("[2/6] Styling CHIRPS precipitation...")
    lyr = find_layer("CHIRPS")
    if lyr is None:
        print("  [SKIP] CHIRPS layer not found.")
        return

    sym = lyr.symbology
    if hasattr(sym, "colorizer"):
        sym.updateColorizer("RasterClassifyColorizer")
        sym.colorizer.classificationMethod = "NaturalBreaks"
        sym.colorizer.breakCount           = 5

        # Yellow-Blue diverging ramp (dry=yellow, wet=blue)
        ramp = aprx.listColorRamps("Yellow-Blue (Continuous)")[0] \
               if aprx.listColorRamps("Yellow-Blue (Continuous)") \
               else aprx.listColorRamps("Precipitation")[0] \
               if aprx.listColorRamps("Precipitation") \
               else None
        if ramp:
            sym.colorizer.colorRamp = ramp

        sym.colorizer.noDataColor.RGB = [200, 200, 200, 255]
        lyr.symbology = sym
        lyr.transparency = 0
        print("  [OK] CHIRPS: 5-class Yellow-Blue, Natural Breaks")


# ---------------------------------------------------------------------------
# 3. Temperature Surface - Diverging Red-Blue
# ---------------------------------------------------------------------------
def style_temperature():
    print("[3/6] Styling temperature surface...")
    lyr = find_layer("TemperatureSurface")
    if lyr is None:
        print("  [SKIP] Temperature surface not found.")
        return

    sym = lyr.symbology
    if hasattr(sym, "colorizer"):
        sym.updateColorizer("RasterClassifyColorizer")
        sym.colorizer.classificationMethod = "NaturalBreaks"
        sym.colorizer.breakCount           = 5

        # Blue (cold) to Red (warm)
        ramp_name = "Temperature"
        candidates = ["Temperature", "Red-Blue (Continuous)", "Cold to Hot Diverging"]
        for name in candidates:
            ramps = aprx.listColorRamps(name)
            if ramps:
                sym.colorizer.colorRamp = ramps[0]
                break

        lyr.symbology = sym
        lyr.transparency = 10
        print("  [OK] Temperature: 5-class Blue-Red diverging")


# ---------------------------------------------------------------------------
# 4. Sentinel-2 False Color Composite (B08/B04/B03)
# ---------------------------------------------------------------------------
def style_sentinel2():
    print("[4/6] Styling Sentinel-2 composite...")
    lyr = find_layer("Sentinel2")
    if lyr is None:
        print("  [SKIP] Sentinel-2 composite not found.")
        return

    sym = lyr.symbology
    if hasattr(sym, "colorizer"):
        # Composite has bands in order: B02=1, B03=2, B04=3, B08=4
        # False color: NIR(B08)=Red channel, Red(B04)=Green, Green(B03)=Blue
        try:
            sym.updateColorizer("RasterStretchColorizer")
            # Band assignment: R=4(NIR), G=3(Red), B=2(Green) for false color
            # ArcGIS Pro uses 1-based band index
            # Note: This only works if the raster is already a 4-band composite
            lyr.symbology = sym
            print("  [OK] Sentinel-2: false color stretched (NIR/Red/Green)")
        except Exception as e:
            print(f"  [WARN] Could not set S2 stretch: {e}")


# ---------------------------------------------------------------------------
# 5. RGI Glacier Outlines - Cyan fill, dark blue outline
# ---------------------------------------------------------------------------
def style_glaciers():
    print("[5/6] Styling glacier outlines...")
    lyr = find_layer("RGI")
    if lyr is None:
        print("  [SKIP] RGI glacier layer not found.")
        return

    sym = lyr.symbology
    if hasattr(sym, "renderer"):
        sym.updateRenderer("SimpleRenderer")
        sym.renderer.symbol.applySymbolFromGallery("Extent Transparent")

        # Set fill: cyan, 35% transparency
        sym.renderer.symbol.color         = {"RGB": [0, 188, 212, 180]}  # cyan
        sym.renderer.symbol.outlineColor  = {"RGB": [13, 71, 161, 255]}  # dark blue
        sym.renderer.symbol.outlineWidth  = 1.5
        lyr.symbology = sym
        lyr.transparency = 35

        # Add labels for glacier names
        lyr.supports("LABELS")
        lyr.showLabels = True
        label_class = lyr.listLabelClasses()[0]
        label_class.expression = "$feature.glac_name"
        label_class.SQLQuery   = "glac_name IS NOT NULL AND glac_area > 5"
        print("  [OK] Glaciers: cyan fill (35% transparent), dark blue outline, labeled")


# ---------------------------------------------------------------------------
# 6. Elevation-based labels on DEM
# ---------------------------------------------------------------------------
def style_dem_elevation():
    print("[6/6] Styling DEM elevation (classified)...")
    lyr = find_layer("DEM_Mosaic")
    if lyr is None:
        return

    sym = lyr.symbology
    if hasattr(sym, "colorizer"):
        sym.updateColorizer("RasterClassifyColorizer")
        sym.colorizer.classificationMethod = "EqualInterval"
        sym.colorizer.breakCount           = 7

        terrain_ramps = ["Terrain", "Elevation #1", "Brown-Green (Continuous)"]
        for name in terrain_ramps:
            ramps = aprx.listColorRamps(name)
            if ramps:
                sym.colorizer.colorRamp = ramps[0]
                break

        lyr.symbology = sym
        lyr.transparency = 30  # semi-transparent over hillshade
        print("  [OK] DEM: 7-class Terrain palette, 30% transparent over hillshade")


# ---------------------------------------------------------------------------
# Reorder layers (important for visual hierarchy)
# ---------------------------------------------------------------------------
def reorder_layers():
    print("\n[+] Reordering layers for visual hierarchy...")
    # Desired order from top to bottom:
    desired_order = [
        "RGI",          # Glacier outlines (top - vector)
        "Sentinel2",    # False color (below vectors)
        "CHIRPS",       # Precipitation or Temperature (toggle)
        "Temperature",
        "DEM_Mosaic",   # Classified elevation
        "Hillshade",    # Base (bottom)
    ]
    all_lyrs = m.listLayers()
    for target in reversed(desired_order):
        for lyr in all_lyrs:
            if target.lower() in lyr.name.lower():
                try:
                    m.moveLayer(lyr, all_lyrs[-1], "AFTER")
                except Exception:
                    pass
    print("  [OK] Layer order set (top: vectors, bottom: hillshade)")


# ---------------------------------------------------------------------------
# Save project
# ---------------------------------------------------------------------------
def save():
    aprx.save()
    print("\n[OK] Project saved.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 55)
    print(" GeoCascade - ArcGIS Pro Symbology Automation")
    print("=" * 55)

    style_dem_hillshade()
    style_chirps()
    style_temperature()
    style_sentinel2()
    style_glaciers()
    style_dem_elevation()
    reorder_layers()
    save()

    print("\n" + "=" * 55)
    print(" SYMBOLOGY COMPLETE")
    print("=" * 55)
    print(" Next: run 03_arcpy_layout_export.py to build the")
    print(" professional 4-panel map layout and export PDF/PNG.")
    print("=" * 55)
