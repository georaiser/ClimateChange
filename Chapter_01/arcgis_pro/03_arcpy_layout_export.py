"""
arcgis_pro/03_arcpy_layout_export.py
======================================
ArcGIS Pro - Professional 4-Panel Map Layout + PDF/PNG Export

Builds a complete professional A3 landscape map layout with:
  - 4 map panels (Sentinel-2, CHIRPS, Temperature, Glaciers+DEM)
  - Title block, north arrow, scale bar, legend
  - Data sources text, date stamp, author field
  - Export to PDF (300 DPI) and PNG (200 DPI)

HOW TO RUN:
    1. Run 01_arcpy_import_imagery.py first (imports all layers)
    2. Run 02_arcpy_climate_maps.py (applies symbology)
    3. Analysis > Python Notebook > New Notebook
    4. Paste and run this script

ArcGIS Pro docs:
    https://pro.arcgis.com/en/pro-app/latest/arcpy/mapping/layout-class.htm

Dependencies: arcpy (pre-installed in ArcGIS Pro Python environment)
              Run ONLY from ArcGIS Pro Notebook or Python console
"""

import arcpy
import os
from datetime import datetime

# ---------------------------------------------------------------------------
# Connect to current ArcGIS Pro project
# ---------------------------------------------------------------------------
try:
    aprx = arcpy.mp.ArcGISProject("CURRENT")
    print(f"[OK] Project: {aprx.filePath}")
except Exception:
    print("ERROR: Run this script inside ArcGIS Pro Notebook.")
    raise

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CHAPTER_DIR = r"D:\00_AI_Aplications\ClimateChange\Chapter_01"
OUTPUT_DIR  = os.path.join(CHAPTER_DIR, "data", "processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LAYOUT_NAME = "GeoCascade_Ch01_Layout"
MAP_NAME    = aprx.activeMap.name if aprx.activeMap else "Map"

TITLE    = "GeoCascade - Torres del Paine Climate Change Analysis"
SUBTITLE = "Patagonia, Chile  |  1993-2024  |  ERA5-Land + Sentinel-2 + CHIRPS + RGI 7.0"
AUTHOR   = "GeoCascade Pipeline | geocascade_env"
DATE_STR = datetime.now().strftime("%Y-%m-%d")

DATA_SOURCES = (
    "Data sources: ERA5-Land (ECMWF/Open-Meteo), CHIRPS v2.0 (UCSB CHG), "
    "Sentinel-2 L2A (ESA/Planetary Computer), Landsat 9 C2L2 (USGS), "
    "Copernicus DEM GLO-30 (ESA), RGI v7.0 (NSIDC)"
)


# ---------------------------------------------------------------------------
# 1. Create or get the layout
# ---------------------------------------------------------------------------
def get_or_create_layout():
    # Check if layout already exists
    for lyt in aprx.listLayouts():
        if LAYOUT_NAME in lyt.name:
            print(f"  [OK] Found existing layout: {lyt.name}")
            return lyt

    # Create new A3 landscape layout (42 x 29.7 cm = 16.54 x 11.69 inches)
    print("  Creating new A3 landscape layout...")
    lyt = aprx.createLayout(16.535, 11.693, "INCH", LAYOUT_NAME)
    print(f"  [OK] Layout created: {lyt.name}")
    return lyt


# ---------------------------------------------------------------------------
# 2. Add 4 map frames
# ---------------------------------------------------------------------------
def add_map_frames(lyt):
    """
    Layout coordinate system: inches from lower-left
    A3 landscape: width=16.535", height=11.693"
    Margins: 0.5" all sides -> usable area: 15.535" x 10.693"
    2x2 grid: each panel ~7.37" x 4.85"
    """
    margin = 0.5
    gutter = 0.25
    w_panel = (16.535 - 2 * margin - gutter) / 2.0   # ~7.64"
    h_panel = (11.693 - 2 * margin - 2.0 - gutter) / 2.0  # ~4.0" (2" for title block)

    panels = [
        # (name, col, row, title_label, layer_to_show)
        ("Panel_TL", 0, 1, "Sentinel-2 L2A - False Color (NIR/Red/Green)",  "Sentinel2"),
        ("Panel_TR", 1, 1, "CHIRPS - Mean Annual Precipitation (2000-2024)", "CHIRPS"),
        ("Panel_BL", 0, 0, "ERA5 Temperature Trend (Mann-Kendall 1993-2024)", "Temperature"),
        ("Panel_BR", 1, 0, "Glacier Extent (RGI v7.0) on DEM Hillshade",    "RGI"),
    ]

    map_frames = {}
    for name, col, row, label, layer_hint in panels:
        # Check if frame already exists
        existing = [e for e in lyt.listElements("MAPFRAME_ELEMENT") if e.name == name]
        if existing:
            mf = existing[0]
            print(f"  [OK] Reusing frame: {name}")
        else:
            x = margin + col * (w_panel + gutter)
            y = margin + row * (h_panel + gutter)  # +1.5 offset for title block

            mf = lyt.createMapFrame(
                arcpy.Point(x, y + 1.5),
                arcpy.Point(x + w_panel, y + 1.5 + h_panel),
                MAP_NAME
            )
            mf.name = name
            print(f"  [OK] Created frame: {name}")

        # Zoom frame to Torres del Paine BBOX
        camera   = mf.camera
        camera.X = -73.0
        camera.Y = -51.0
        camera.scale = 500000  # 1:500,000 -- adjust to scene extent
        mf.camera = camera

        map_frames[name] = (mf, label)

    return map_frames


# ---------------------------------------------------------------------------
# 3. Add cartographic elements
# ---------------------------------------------------------------------------
def add_cartographic_elements(lyt, map_frames):
    """Add title, subtitle, north arrow, scale bar, legend."""

    existing_names = {e.name for e in lyt.listElements()}

    # --- Title block background (white bar at top) ---
    # (Not scriptable directly -- use graphic element instead)

    # --- Title text ---
    if "Title" not in existing_names:
        title_el = lyt.createTextElement(
            arcpy.Point(8.27, 10.9),   # center of page, near top
            "POINT",
            TITLE
        )
        title_el.name              = "Title"
        title_el.elementPositionX  = 0.5
        title_el.elementPositionY  = 10.9
        title_el.elementWidth      = 15.535
        title_el.elementHeight     = 0.5
        print("  [OK] Title added")

    # --- Subtitle ---
    if "Subtitle" not in existing_names:
        sub_el = lyt.createTextElement(
            arcpy.Point(8.27, 10.4), "POINT", SUBTITLE
        )
        sub_el.name = "Subtitle"
        print("  [OK] Subtitle added")

    # --- Data sources (bottom) ---
    if "DataSources" not in existing_names:
        src_el = lyt.createTextElement(
            arcpy.Point(0.5, 0.15), "POINT", DATA_SOURCES
        )
        src_el.name = "DataSources"
        print("  [OK] Data sources text added")

    # --- Date stamp (bottom right) ---
    if "DateStamp" not in existing_names:
        date_el = lyt.createTextElement(
            arcpy.Point(14.5, 0.15), "POINT",
            f"Generated: {DATE_STR}  |  {AUTHOR}"
        )
        date_el.name = "DateStamp"
        print("  [OK] Date stamp added")

    # --- North arrow (bottom-left of first panel) ---
    north_items = [e for e in lyt.listElements("NORTH_ARROW_ELEMENT")]
    if not north_items:
        try:
            north = lyt.createMapSurroundElement(
                arcpy.Point(1.2, 1.8),
                "NORTH_ARROW",
                list(map_frames.values())[0][0],
                "ESRI North 1"  # Classic north arrow style
            )
            north.name = "NorthArrow"
            print("  [OK] North arrow added")
        except Exception as e:
            print(f"  [WARN] North arrow: {e}")

    # --- Scale bar ---
    scale_items = [e for e in lyt.listElements("SCALE_BAR_ELEMENT")]
    if not scale_items:
        try:
            sb = lyt.createMapSurroundElement(
                arcpy.Point(0.5, 1.5),
                "SCALE_BAR",
                list(map_frames.values())[0][0],
                "Scale Line 1"
            )
            sb.name = "ScaleBar"
            sb.unitLabel  = "km"
            sb.divisions  = 4
            sb.subdivision = 2
            print("  [OK] Scale bar added")
        except Exception as e:
            print(f"  [WARN] Scale bar: {e}")

    # --- Legend ---
    legend_items = [e for e in lyt.listElements("LEGEND_ELEMENT")]
    if not legend_items:
        try:
            legend = lyt.createMapSurroundElement(
                arcpy.Point(12.8, 1.5),
                "LEGEND",
                list(map_frames.values())[0][0]
            )
            legend.name = "Legend"
            print("  [OK] Legend added")
        except Exception as e:
            print(f"  [WARN] Legend: {e}")


# ---------------------------------------------------------------------------
# 4. Export
# ---------------------------------------------------------------------------
def export_layout(lyt):
    pdf_path = os.path.join(OUTPUT_DIR, "chapter01_map_layout.pdf")
    png_path = os.path.join(OUTPUT_DIR, "chapter01_map_layout.png")

    print(f"  Exporting PDF (300 DPI)...")
    lyt.exportToPDF(pdf_path, resolution=300)
    print(f"  [OK] PDF: {pdf_path}")

    print(f"  Exporting PNG (200 DPI)...")
    lyt.exportToPNG(png_path, resolution=200)
    print(f"  [OK] PNG: {png_path}")

    return pdf_path, png_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 58)
    print(" GeoCascade - ArcGIS Pro Layout Builder")
    print(f" Layout: {LAYOUT_NAME}")
    print("=" * 58)

    print("\n[1/4] Setting up layout...")
    lyt = get_or_create_layout()

    print("\n[2/4] Adding 4 map frames...")
    map_frames = add_map_frames(lyt)

    print("\n[3/4] Adding cartographic elements...")
    add_cartographic_elements(lyt, map_frames)

    print("\n[4/4] Exporting...")
    pdf_path, png_path = export_layout(lyt)

    aprx.save()

    print("\n" + "=" * 58)
    print(" LAYOUT EXPORT COMPLETE")
    print("=" * 58)
    print(f"  PDF (300 DPI): {pdf_path}")
    print(f"  PNG (200 DPI): {png_path}")
    print()
    print("  To adjust: Insert > Activate Layout, then drag elements.")
    print("  To re-export: re-run this script after making changes.")
    print("=" * 58)
