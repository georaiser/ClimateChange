# 🗺️ ArcGIS Pro Track — Professional Cartography & Analysis

> **GeoCascade Chapter 1 — ArcGIS Pro Workflow Guide**
> *ArcGIS Pro 3.x + arcpy | Torres del Paine, Patagonia*

---

## Overview

This track uses ArcGIS Pro to create professional cartographic outputs from
the data produced by the Python pipeline in Chapter 1. All Python-generated
GeoTIFFs, CSVs, and GeoPackages are natively compatible with ArcGIS Pro.

| Script | What it does | Run how |
|---|---|---|
| `01_arcpy_import_imagery.py` | Mosaic DEM, composite S2, import all layers | ArcGIS Pro Notebook |
| `02_arcpy_climate_maps.py` | Apply professional symbology to all layers | ArcGIS Pro Notebook |
| `03_arcpy_layout_export.py` | Build 4-panel layout, export PDF + PNG | ArcGIS Pro Notebook |

---

## Step 1: Set Up the ArcGIS Pro Project

### Create a new project

1. Open **ArcGIS Pro 3.x**
2. **New > Map** → name the project **GeoCascade_Ch01**
3. Save to: `D:\00_AI_Aplications\ClimateChange\Chapter_01\arcgis_pro\`
4. This creates `GeoCascade_Ch01.aprx` and a default geodatabase `GeoCascade_Ch01.gdb`

### Set the coordinate system

For display (the Map):
1. **Map > Properties > Coordinate Systems**
2. Set to: **WGS 1984** (EPSG:4326) for geographic overview

For analysis (measuring distances and areas):
1. In geoprocessing tools, always choose **WGS 1984 UTM Zone 19S** (EPSG:32719)
2. This avoids the ~40% distance error of Web Mercator (EPSG:3857) at latitude 51°S

> [!WARNING]
> **NEVER use EPSG:3857 (Web Mercator) for distance or area calculations at 51°S.**
> Web Mercator introduces ~40% linear distortion at Patagonian latitudes.
> Always use a local UTM projection for any measurement-based analysis.

---

## Step 2: Import Chapter 1 Data

### Option A: Run the arcpy script (recommended for repeatability)

1. **Analysis > Python Notebook** → **New Notebook**
2. Paste the content of `01_arcpy_import_imagery.py` into a cell
3. Run (Shift + Enter)

### Option B: Manual import (ArcGIS Pro GUI)

**Import Copernicus DEM tiles:**
1. **Insert > Add Data > Folder Connection** → navigate to `Chapter_01/data/raw/`
2. Expand `dem_*/` folders
3. Drag each `copernicus_dem_30m.tif` to the Map
4. **Data Management > Raster > Mosaic to New Raster**:
   - Input rasters: select all 4 DEM tiles
   - Output: `GeoCascade_Ch01.gdb\CopernicusDEM_Mosaic`
   - Pixel type: **32-bit float**
   - Number of bands: **1**
   - Mosaic operator: **Mean**

**Import Sentinel-2 composite:**
1. Navigate to the Sentinel-2 scene folder: `sentinel2_l2a_*/`
2. Select `B02.tif`, `B03.tif`, `B04.tif`, `B08.tif`
3. **Data Management > Raster > Composite Bands**:
   - Input rasters: B02, B03, B04, B08 (order matters for band assignment)
   - Output: `GeoCascade_Ch01.gdb\Sentinel2_Composite`

**Import CHIRPS climatology:**
1. **Insert > Add Data > Raster Dataset**
2. Navigate to: `Chapter_01/data/processed/real_data/chirps_mean_annual_precip.tif`
3. The TIF opens directly with automatic stretching

**Import RGI glacier outlines:**
1. **Insert > Add Data > File Geodatabase or GeoPackage Layer**
2. Navigate to: `Chapter_01/data/raw/real_data/rgi70_patagonia_glaciers.gpkg`
3. Select the polygon layer and add to Map

> [!TIP]
> All GeoTIFFs produced by the Python pipeline include embedded CRS metadata.
> ArcGIS Pro reads EPSG codes automatically — no manual projection needed.

---

## Step 3: Create Thematic Maps

### Temperature trend map

1. Click the **temperature_surface** layer in the Contents pane
2. **Appearance > Symbology** (or right-click → Symbology)
3. **Primary Symbology** → **Classified**
4. **Method**: Natural Breaks (Jenks), 5 classes
5. **Color scheme**: Search for "Red-Blue Diverging" → select Blue-to-Red
6. **Invert** if needed so Blue = cold, Red = warm
7. **Label** units as `deg C` in the legend

### Precipitation map (CHIRPS)

1. Click the **CHIRPS_MeanAnnualPrecip** layer
2. Symbology → **Classified** → Natural Breaks, 5 classes
3. Color scheme: **Yellow to Blue** (Yellow = dry, Blue = wet)
4. Format labels as integers (e.g., `1200 mm`)

### DEM + Hillshade

1. First, create the hillshade:
   - **Analysis > Tools > Hillshade** (in 3D Analyst toolbox)
   - Input: `CopernicusDEM_Mosaic`
   - Azimuth: 315 (northwest), Altitude: 45
   - Output: `CopernicusDEM_Hillshade`

2. In Contents pane, drag Hillshade **below** the DEM mosaic layer
3. Set DEM mosaic to **30% transparency** (Appearance tab)

### Glacier outlines (RGI)

1. Click **RGI_Glaciers** layer
2. Symbology → **Single Symbol**
3. Symbol type: **Simple Fill**
   - Fill color: **Cyan** (R:0, G:188, B:212), 35% transparency
   - Outline color: **Dark Blue** (R:13, G:71, B:161), width: 1.5 pt

4. Add labels:
   - Right-click layer → **Label Features**
   - Label field: `glac_name`
   - Filter: only label glaciers > 5 km² area

### False color Sentinel-2 composite

1. Click **Sentinel2_Composite** layer
2. Symbology → **Stretch**
3. **Band combination**: R=Band 4 (NIR/B08), G=Band 3 (Red/B04), B=Band 2 (Green/B03)
   - This creates a **False Color** view where vegetation = bright red,
     water = dark, ice/snow = bright cyan
4. **Stretch type**: Percent Clip (2%)

> [!NOTE]
> In standard False Color, vegetation appears **red** because plant leaves
> strongly reflect Near-Infrared (B08) which is mapped to the red channel.
> Healthy dense vegetation = bright red, stressed or sparse = pink/brown.

---

## Step 4: Build the Professional Layout

### Create a new layout

1. **Insert > New Layout > A3 Landscape** (42 × 29.7 cm)
2. The layout canvas opens in a new tab

### Add 4 map frames

1. **Insert > Map Frame** → draw a frame in the top-left quadrant
2. Repeat for top-right, bottom-left, bottom-right
3. Each frame shows the same map — assign different layer combinations

**Suggested panel content:**

| Panel | Position | Layers visible |
|---|---|---|
| **Sentinel-2 False Color** | Top Left | S2 Composite + Glacier outlines |
| **CHIRPS Precipitation** | Top Right | CHIRPS classified + Admin boundaries |
| **Temperature Trend** | Bottom Left | Temperature surface + Hillshade base |
| **Glaciers on DEM** | Bottom Right | RGI glaciers + DEM hillshade + labels |

4. For each frame: right-click → **Activate** to pan/zoom to the study area
5. Set the same extent in all panels: **BBOX −73.5, −51.5 to −72.5, −50.5**

### Add cartographic elements

1. **North arrow**: Insert > North Arrow → place in one panel corner
2. **Scale bar**: Insert > Scale Bar → choose "Scale Line 1", set units to km
3. **Legend**: Insert > Legend → select layers to include
4. **Title text**: Insert > Straight Text →
   ```
   GeoCascade - Torres del Paine Climate Change Analysis
   ```
5. **Subtitle**: Insert > Straight Text →
   ```
   Patagonia, Chile  |  1993-2024  |  ERA5-Land + Sentinel-2 + CHIRPS + RGI 7.0
   ```
6. **Data sources** (small text, bottom of page):
   ```
   ERA5-Land: ECMWF/Open-Meteo | CHIRPS: UCSB CHG | Imagery: Planetary Computer
   Sentinel-2 L2A: ESA | Landsat: USGS | DEM: Copernicus/ESA | Glaciers: RGI v7.0
   ```

---

## Step 5: Export the Layout

### From ArcGIS Pro GUI

1. **Share > Export Layout**
2. **PDF**: resolution 300 DPI, raster quality = Normal, vector quality = Best
3. **PNG**: resolution 200 DPI
4. Save to `Chapter_01/data/processed/`

### From arcpy (automated)

```python
# Run inside ArcGIS Pro Notebook
exec(open(r'D:\00_AI_Aplications\ClimateChange\Chapter_01\arcgis_pro\03_arcpy_layout_export.py').read())
```

---

## Step 6: Running arcpy Scripts from Notebook

### Open the Python Notebook

1. **Analysis > Python Notebook** (in the Analysis ribbon)
2. Click **New** to create a blank notebook

### Run a script

Option A — Run file directly:
```python
exec(open(r'D:\00_AI_Aplications\ClimateChange\Chapter_01\arcgis_pro\01_arcpy_import_imagery.py').read())
```

Option B — Paste script into cell:
- Copy the script content into a notebook cell
- Click Run (Shift + Enter)

> [!IMPORTANT]
> The ArcGIS Pro Python environment already has `arcpy` installed.
> **Do NOT install arcpy into `geocascade_env`** — arcpy is not available
> as a standalone package. It only works inside ArcGIS Pro's Python.

> [!TIP]
> You can call the regular GeoCascade Python scripts from ArcGIS Pro Notebook
> using `subprocess`:
> ```python
> import subprocess
> subprocess.run([r"C:\path\to\geocascade_env\python.exe",
>                r"D:\...\Chapter_01\04_climate_trend_analysis.py"])
> ```

---

## ArcGIS Pro + ENVI Integration

The workflows connect at the GeoTIFF level:

```
Python (downloads + analysis)
    --> GeoTIFFs in data/processed/
          |
          |-- ArcGIS Pro: Add Data > Raster Dataset -> thematic maps
          |
          |-- ENVI 5.6:  File > Open -> spectral analysis
                |
                --> flaash_corrected_rfl.tif (ENVI FLAASH output)
                      |
                      ArcGIS Pro: Add Data -> overlay FLAASH vs L2A
```

---

## Files in This Folder

| File | Purpose |
|---|---|
| `README_ARCGISPRO.md` | This guide |
| `01_arcpy_import_imagery.py` | Mosaic DEM, composite S2, import all layers |
| `02_arcpy_climate_maps.py` | Apply symbology to all Chapter 1 layers |
| `03_arcpy_layout_export.py` | Build A3 layout + export PDF/PNG |

---

*GeoCascade | Chapter 1 | ArcGIS Pro 3.x Track | Study area: Torres del Paine, 51°S*
