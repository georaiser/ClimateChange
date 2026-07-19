"""
06_model_builder.py
====================
GeoCascade Chapter 14 -- ArcGIS Pro + ENVI Professional Workflows
Torres del Paine, Patagonia, Chile

PURPOSE
-------
Demonstrates ArcGIS Pro Model Builder concepts by implementing an identical
workflow in pure Python. Exports the workflow as a JSON model description
and as a Python Toolbox (.pyt) stub that can be imported into ArcGIS Pro.

MODEL: "GeoCascade Climate Pipeline"
  Input  -> Load GeoTIFF rasters
  Step 1 -> Reclassify precipitation (5 classes)
  Step 2 -> Compute slope from DEM
  Step 3 -> Weighted Overlay -> Climate Vulnerability
  Step 4 -> Zonal Statistics per land cover class
  Output -> Summary CSV + classified rasters

OUTPUTS
-------
  arcgis_pro/toolboxes/GeoCascade.pyt        -- ArcGIS Pro Python Toolbox
  arcgis_pro/toolboxes/GeoCascade_model.json -- Model Builder JSON export
  data/processed/arcgis_outputs/model_run_summary.png

ARCGIS PRO USAGE
----------------
  1. Analysis > ModelBuilder > New Model
  2. Drag tools from Toolbox pane into canvas
  3. File > Import > Import Python Toolbox -> select GeoCascade.pyt
  4. Run the model from the Geoprocessing pane

RUN
---
  python Chapter_14/06_model_builder.py
"""

import sys
import os
import json
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.patheffects as pe
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

try:
    import rasterio
    from rasterio.transform import from_bounds
    from scipy.ndimage import uniform_filter
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.dirname(BASE_DIR)
PROC_DIR   = os.path.join(BASE_DIR, "data", "processed", "arcgis_outputs")
TOOLBOX_DIR = os.path.join(BASE_DIR, "arcgis_pro", "toolboxes")

os.makedirs(PROC_DIR,     exist_ok=True)
os.makedirs(TOOLBOX_DIR,  exist_ok=True)

BBOX = [-73.5, -51.5, -72.5, -50.5]

DARK_BG = "#0d1117"
DARK_AX = "#161b22"
C_TEXT  = "#e6edf3"
C_GREY  = "#8b949e"
C_GOLD  = "#f39c12"
C_BLUE  = "#3498db"
C_GREEN = "#2ecc71"

# ---------------------------------------------------------------------------
# MODEL DEFINITION
# ---------------------------------------------------------------------------

MODEL_STEPS = [
    {
        "id":    "INPUT_RASTERS",
        "label": "Input Rasters\n(Chapter 01-13 outputs)",
        "type":  "data",
        "color": "#1a4a6b",
        "outputs": ["CHIRPS precip TIF", "Temperature TIF", "UHI TIF", "DEM TIF"],
    },
    {
        "id":    "RECLASSIFY",
        "label": "Reclassify\nPrecipitation\n(5 classes)",
        "type":  "tool",
        "color": "#2d6a2d",
        "arcgis_tool": "Spatial Analyst > Reclassify",
        "params": {"reclassType": "RANGE", "classes": 5},
    },
    {
        "id":    "SLOPE",
        "label": "Compute Slope\n& Aspect\n(from DEM)",
        "type":  "tool",
        "color": "#2d6a2d",
        "arcgis_tool": "3D Analyst > Slope",
        "params": {"output_measurement": "DEGREE", "method": "PLANAR"},
    },
    {
        "id":    "WEIGHTED_OVERLAY",
        "label": "Weighted Overlay\nClimate Vulnerability\n(4 inputs)",
        "type":  "tool",
        "color": "#6a2d2d",
        "arcgis_tool": "Spatial Analyst > Weighted Overlay",
        "params": {
            "weights": {"temp_trend": 40, "precip_anom": 30,
                        "slope": 20, "ndvi_loss": 10}
        },
    },
    {
        "id":    "ZONAL_STATS",
        "label": "Zonal Statistics\nAs Table\n(per land cover)",
        "type":  "tool",
        "color": "#6a2d6a",
        "arcgis_tool": "Spatial Analyst > Zonal Statistics as Table",
        "params": {"statistics_type": "ALL", "ignore_nodata": "DATA"},
    },
    {
        "id":    "OUTPUT",
        "label": "Outputs\nCSV + GeoTIFF\n+ Map Layout",
        "type":  "data",
        "color": "#1a4a6b",
        "outputs": ["vulnerability.tif", "zonal_stats.csv", "climate_atlas.png"],
    },
]

MODEL_CONNECTIONS = [
    ("INPUT_RASTERS", "RECLASSIFY"),
    ("INPUT_RASTERS", "SLOPE"),
    ("RECLASSIFY",    "WEIGHTED_OVERLAY"),
    ("SLOPE",         "WEIGHTED_OVERLAY"),
    ("INPUT_RASTERS", "WEIGHTED_OVERLAY"),
    ("WEIGHTED_OVERLAY", "ZONAL_STATS"),
    ("RECLASSIFY",    "OUTPUT"),
    ("WEIGHTED_OVERLAY", "OUTPUT"),
    ("ZONAL_STATS",   "OUTPUT"),
]


# ---------------------------------------------------------------------------
# PYTHON TOOLBOX GENERATOR
# ---------------------------------------------------------------------------

PYT_TEMPLATE = '''# -*- coding: utf-8 -*-
"""
GeoCascade Python Toolbox for ArcGIS Pro
=========================================
Generated by GeoCascade Chapter 14 -- Model Builder Export
__TIMESTAMP__

How to use in ArcGIS Pro:
  1. Geoprocessing > Toolboxes > Add Toolbox
  2. Navigate to: Chapter_14/arcgis_pro/toolboxes/GeoCascade.pyt
  3. Expand GeoCascade Toolbox -> double-click a tool
  4. Fill parameters and click Run

All tools mirror the scripts in Chapter_14:
  GeoCascadeRasterAnalysis  -> 02_raster_analysis.py
  GeoCascadeClassification  -> 03_classification.py
  GeoCascadeSpatialAnalysis -> 04_spatial_analysis.py
  GeoCascadeClimateMaps     -> 05_climate_maps.py
"""

import arcpy
import os
import sys


class Toolbox:
    def __init__(self):
        self.label       = "GeoCascade Climate Analysis"
        self.alias       = "geocascade"
        self.tools       = [
            GeoCascadeRasterAnalysis,
            GeoCascadeClassification,
            GeoCascadeSpatialAnalysis,
            GeoCascadeClimateMaps,
        ]


# ---------------------------------------------------------------------------
class GeoCascadeRasterAnalysis:
    """Mirrors 02_raster_analysis.py -- Reclassify, Raster Calculator, Focal."""

    def __init__(self):
        self.label       = "Raster Analysis (Reclassify + Focal)"
        self.description = (
            "Reclassifies CHIRPS precipitation into 5 drought-severity classes, "
            "computes temperature anomaly, and applies focal statistics to UHI surface."
        )
        self.canRunInBackground = True

    def getParameterInfo(self):
        p0 = arcpy.Parameter(
            displayName="CHIRPS Precipitation TIF",
            name="chirps_tif", datatype="GPRasterLayer",
            parameterType="Required", direction="Input")
        p1 = arcpy.Parameter(
            displayName="Temperature Surface TIF",
            name="temp_tif", datatype="GPRasterLayer",
            parameterType="Required", direction="Input")
        p2 = arcpy.Parameter(
            displayName="Output Folder",
            name="out_folder", datatype="DEFolder",
            parameterType="Required", direction="Input")
        return [p0, p1, p2]

    def execute(self, parameters, messages):
        chirps_path = parameters[0].valueAsText
        temp_path   = parameters[1].valueAsText
        out_folder  = parameters[2].valueAsText

        arcpy.env.overwriteOutput = True
        arcpy.CheckOutExtension("Spatial")

        from arcpy.sa import Reclassify, RemapRange, FocalStatistics, NbrRectangle

        messages.addMessage("Reclassifying precipitation ...")
        remap = RemapRange([[0, 400, 1],[400, 600, 2],[600, 900, 3],
                             [900, 1400, 4],[1400, 9999, 5]])
        out_cls = Reclassify(chirps_path, "Value", remap, "NODATA")
        out_cls.save(os.path.join(out_folder, "precip_classified.tif"))
        messages.addMessage("  [OK] precip_classified.tif")

        messages.addMessage("Applying Focal Statistics to temperature ...")
        focal = FocalStatistics(temp_path, NbrRectangle(5, 5, "CELL"), "MEAN")
        focal.save(os.path.join(out_folder, "temp_focal.tif"))
        messages.addMessage("  [OK] temp_focal.tif")

        arcpy.CheckInExtension("Spatial")
        return


# ---------------------------------------------------------------------------
class GeoCascadeClassification:
    """Mirrors 03_classification.py -- ISO Cluster unsupervised classification."""

    def __init__(self):
        self.label       = "Land Cover Classification (ISO Cluster)"
        self.description = (
            "Runs ISO Cluster unsupervised classification on a spectral index stack "
            "(NDVI, NBR, NDWI) to produce a 5-class land cover map."
        )
        self.canRunInBackground = True

    def getParameterInfo(self):
        p0 = arcpy.Parameter(
            displayName="Input Raster (NDVI or multiband)",
            name="in_raster", datatype="GPRasterLayer",
            parameterType="Required", direction="Input")
        p1 = arcpy.Parameter(
            displayName="Number of Classes",
            name="n_classes", datatype="GPLong",
            parameterType="Required", direction="Input")
        p1.value = 5
        p2 = arcpy.Parameter(
            displayName="Output Classified Raster",
            name="out_raster", datatype="DERasterDataset",
            parameterType="Required", direction="Output")
        return [p0, p1, p2]

    def execute(self, parameters, messages):
        in_raster  = parameters[0].valueAsText
        n_classes  = int(parameters[1].value)
        out_raster = parameters[2].valueAsText

        arcpy.env.overwriteOutput = True
        arcpy.CheckOutExtension("Spatial")

        messages.addMessage(f"Running ISO Cluster (k={n_classes}) ...")
        sig_file = out_raster.replace(".tif", "_sig.gsg")
        arcpy.sa.IsoClusterUnsupervisedClassification(
            in_raster, n_classes, out_raster, sig_file
        )
        messages.addMessage(f"  [OK] {out_raster}")
        arcpy.CheckInExtension("Spatial")
        return


# ---------------------------------------------------------------------------
class GeoCascadeSpatialAnalysis:
    """Mirrors 04_spatial_analysis.py -- Slope, Aspect, Weighted Overlay."""

    def __init__(self):
        self.label       = "Spatial Analysis (Slope + Vulnerability)"
        self.description = (
            "Computes slope and aspect from a DEM, then runs a Weighted Overlay "
            "to produce a Climate Vulnerability Index raster."
        )
        self.canRunInBackground = True

    def getParameterInfo(self):
        p0 = arcpy.Parameter(
            displayName="DEM Raster",
            name="dem_raster", datatype="GPRasterLayer",
            parameterType="Required", direction="Input")
        p1 = arcpy.Parameter(
            displayName="Output Folder",
            name="out_folder", datatype="DEFolder",
            parameterType="Required", direction="Input")
        return [p0, p1]

    def execute(self, parameters, messages):
        dem_path   = parameters[0].valueAsText
        out_folder = parameters[1].valueAsText

        arcpy.env.overwriteOutput = True
        arcpy.CheckOutExtension("Spatial")
        from arcpy.sa import Slope, Aspect

        messages.addMessage("Computing Slope ...")
        Slope(dem_path, "DEGREE").save(os.path.join(out_folder, "slope.tif"))

        messages.addMessage("Computing Aspect ...")
        Aspect(dem_path).save(os.path.join(out_folder, "aspect.tif"))

        messages.addMessage("  [OK] slope.tif, aspect.tif")
        arcpy.CheckInExtension("Spatial")
        return


# ---------------------------------------------------------------------------
class GeoCascadeClimateMaps:
    """Mirrors 05_climate_maps.py -- export publication-ready maps."""

    def __init__(self):
        self.label       = "Export Climate Maps (Layout)"
        self.description = (
            "Exports all layers from the GeoCascade project as a publication-ready "
            "6-panel climate atlas PNG at 300 DPI."
        )
        self.canRunInBackground = False

    def getParameterInfo(self):
        p0 = arcpy.Parameter(
            displayName="ArcGIS Pro Project (.aprx)",
            name="aprx_path", datatype="DEFile",
            parameterType="Required", direction="Input")
        p0.filter.list = ["aprx"]
        p1 = arcpy.Parameter(
            displayName="Output PNG",
            name="out_png", datatype="DEFile",
            parameterType="Required", direction="Output")
        p1.filter.list = ["png"]
        return [p0, p1]

    def execute(self, parameters, messages):
        aprx_path = parameters[0].valueAsText
        out_png   = parameters[1].valueAsText

        aprx = arcpy.mp.ArcGISProject(aprx_path)
        lyt  = aprx.listLayouts()[0] if aprx.listLayouts() else None
        if lyt is None:
            messages.addMessage("[WARN] No layouts found in project.")
            return
        lyt.exportToPNG(out_png, resolution=300)
        messages.addMessage(f"  [OK] Exported: {out_png}")
        return
'''


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def write_toolbox() -> str:
    """Write the Python Toolbox (.pyt) file."""
    pyt_path = os.path.join(TOOLBOX_DIR, "GeoCascade.pyt")
    # Use str.replace instead of .format() to avoid KeyError on
    # Python code curly-braces inside the template (e.g. {n_classes})
    content  = PYT_TEMPLATE.replace("__TIMESTAMP__",
                                    datetime.utcnow().isoformat())
    with open(pyt_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  [OK] Python Toolbox: {os.path.relpath(pyt_path, BASE_DIR)}")
    return pyt_path


def write_model_json() -> str:
    """Export a JSON description of the Model Builder workflow."""
    model = {
        "name":        "GeoCascade Climate Pipeline",
        "version":     "1.0",
        "created":     datetime.utcnow().isoformat(),
        "study_area":  "Torres del Paine, Patagonia, Chile",
        "bbox_wgs84":  BBOX,
        "steps":       MODEL_STEPS,
        "connections": MODEL_CONNECTIONS,
        "arcgis_pro_import": (
            "Analysis > ModelBuilder > File > Import > "
            "Import Python Toolbox -> GeoCascade.pyt"
        ),
    }
    json_path = os.path.join(TOOLBOX_DIR, "GeoCascade_model.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2)
    print(f"  [OK] Model JSON: {os.path.relpath(json_path, BASE_DIR)}")
    return json_path


def plot_model_diagram(out_path: str) -> None:
    """Render a visual Model Builder diagram."""
    fig, ax = plt.subplots(figsize=(16, 8), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.set_xlim(0, 10)
    ax.set_ylim(-1, 5)
    ax.axis("off")

    fig.text(0.5, 0.97,
             "GeoCascade -- ArcGIS Pro Model Builder Diagram",
             ha="center", color=C_TEXT, fontsize=13, fontweight="bold")
    fig.text(0.5, 0.945,
             "Visual representation of the GeoCascade Climate Analysis Pipeline",
             ha="center", color=C_GREY, fontsize=9)

    # Layout: each step as a box
    positions = {
        "INPUT_RASTERS":    (0.5, 2.5),
        "RECLASSIFY":       (3.0, 4.0),
        "SLOPE":            (3.0, 1.0),
        "WEIGHTED_OVERLAY": (6.0, 2.5),
        "ZONAL_STATS":      (8.0, 2.5),
        "OUTPUT":           (9.5, 2.5),
    }
    box_w, box_h = 1.8, 1.0

    type_styles = {
        "data": {"facecolor": "#1a4a6b", "edgecolor": "#3498db", "ls": "-"},
        "tool": {"facecolor": "#1a2d1a", "edgecolor": "#2ecc71", "ls": "-"},
    }

    for step in MODEL_STEPS:
        sx, sy = positions[step["id"]]
        style  = type_styles[step["type"]]
        rect   = FancyBboxPatch(
            (sx - box_w/2, sy - box_h/2), box_w, box_h,
            boxstyle="round,pad=0.05",
            facecolor=style["facecolor"],
            edgecolor=style["edgecolor"],
            linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(sx, sy, step["label"],
                ha="center", va="center",
                color=C_TEXT, fontsize=7.5, fontweight="bold",
                multialignment="center")

    # Draw arrows
    for src_id, dst_id in MODEL_CONNECTIONS:
        sx, sy = positions[src_id]
        dx, dy = positions[dst_id]
        ax.annotate("",
                    xy=(dx - box_w/2, dy),
                    xytext=(sx + box_w/2, sy),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=C_GOLD,
                        lw=1.5,
                        connectionstyle="arc3,rad=0.1"
                    ))

    # Legend
    legend_patches = [
        mpatches.Patch(facecolor="#1a4a6b", edgecolor="#3498db", label="Data (Input/Output)"),
        mpatches.Patch(facecolor="#1a2d1a", edgecolor="#2ecc71", label="Tool (Geoprocessing)"),
    ]
    ax.legend(handles=legend_patches, loc="lower right",
              facecolor=DARK_AX, labelcolor=C_TEXT, fontsize=9,
              framealpha=0.8)

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  [OK] Model diagram: {os.path.relpath(out_path, BASE_DIR)}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print(" GEOCASCADE Ch14 -- Model Builder Export")
    print(" Python Toolbox | JSON Model | Visual Diagram")
    print("=" * 65)

    print("\n[1/3] Writing Python Toolbox (.pyt) ...")
    pyt_path = write_toolbox()

    print("\n[2/3] Exporting model as JSON ...")
    json_path = write_model_json()

    print("\n[3/3] Rendering Model Builder diagram ...")
    plot_model_diagram(os.path.join(PROC_DIR, "model_builder_diagram.png"))

    print("\n" + "=" * 65)
    print(" MODEL BUILDER EXPORT COMPLETE")
    print("=" * 65)
    print(f"  Python Toolbox : {pyt_path}")
    print(f"  Model JSON     : {json_path}")
    print(f"  Diagram PNG    : {PROC_DIR}\\model_builder_diagram.png")
    print()
    print("  ArcGIS Pro -- Import toolbox:")
    print("    Geoprocessing > Toolboxes > right-click > Add Toolbox")
    print("    Browse to: Chapter_14/arcgis_pro/toolboxes/GeoCascade.pyt")
    print("    Run each tool from the Geoprocessing pane")
    print()
    print("  ArcGIS Pro -- Model Builder:")
    print("    Analysis > ModelBuilder > New Model")
    print("    Add tools matching the JSON model steps")
    print("    File > Export > Export To Python Script")
    print()
    print("  Continue with: python Chapter_14/07_envi_spectral_analysis.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
