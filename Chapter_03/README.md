# Chapter 3: Glaciology & Hydrological Modeling

## 🎯 Academic Objective
This chapter focuses on the physical impacts of climate change on water resources in Torres del Paine. We transition from static, single-image processing (Chapter 2) to **Multi-temporal Analysis** (time-series change detection) and **3D Terrain Modeling** (Hydrology).

We continue our **Hybrid Dual-Track Methodology**: providing automated Python scripts alongside manual GUI workflows for ArcGIS Pro.

---

## 🛠️ Scripts & Modules

### `09_multitemporal_glacier_retreat.py`
- **Concept:** One of the most visible impacts of climate change is glacial retreat. By comparing satellite imagery of a glacier from two different points in time, we can quantify the exact area of ice lost to melting.
- **Action (Python Automation):** We use the Planetary Computer STAC API to dynamically fetch Landsat imagery from 2003 and 2023. We calculate the Normalized Difference Snow Index (NDSI) for both years, threshold the ice extent, and mathematically subtract the two arrays to generate a Glacial Retreat Map. The output is exported as `glacier_retreat_2003_2023.tif` for GIS analysis.
- **Action (ArcGIS Pro Alternative):** 
  1. Download a 2003 and 2023 Landsat image from USGS EarthExplorer and load them into ArcGIS Pro.
  2. Use the **Raster Calculator** to compute the NDSI for both: `(Green - SWIR1) / (Green + SWIR1)`.
  3. Use the **Con** tool (Spatial Analyst) to isolate ice: `Con("NDSI_2003" > 0.4, 1, 0)`.
  4. Use the **Minus** tool to subtract the 2023 Ice raster from the 2003 Ice raster.
  5. The resulting map will have values of `1` where ice melted.

### `10_digital_elevation_processing.py`
- **Concept:** A Digital Elevation Model (DEM) is just a 2D array of heights. By taking the mathematical derivative (the gradient) in the X and Y directions, we can calculate the steepness of the terrain (Slope) and the compass direction it faces (Aspect). We can then simulate lighting (Hillshade) to make the terrain look 3D.
- **Action (Python Automation):** Streams the Copernicus 30m Global DEM from the STAC API. Uses pure `numpy.gradient` to teach the physics of Slope and Aspect, and then calculates the Hillshade illumination equation from scratch. The raw DEM and its derivatives are exported as geocoded TIFFs (`copernicus_dem.tif`, `slope_degrees.tif`, `aspect_degrees.tif`, `hillshade.tif`).
- **Action (ArcGIS Pro Alternative):** 
  1. Load your DEM raster into ArcGIS Pro.
  2. Navigate to **Analysis** -> **Tools** to open the Geoprocessing pane.
  3. Search for the **Surface Parameters** or the classic **Slope**, **Aspect**, and **Hillshade** tools (under *Spatial Analyst -> Surface*).
  4. Run each tool on your DEM to generate the layers. 

### `11_watershed_delineation.py`
- **Concept:** Water flows downhill. By algorithmically evaluating our DEM to find the steepest downhill path from every pixel, we can trace where water will flow (Flow Direction) and sum up how much water accumulates at the bottom (Flow Accumulation) to trace river networks and delineate watershed basins.
- **Action (Python Automation):** Uses the `pysheds` python package to perfectly replicate the ArcHydro toolset. It mathematically fills sinks in the DEM, runs the D8 routing algorithm, and plots the River Network using logarithmic scaling. The hydrological models are exported as `flow_direction.tif` and `flow_accumulation.tif`.
- **Action (ArcGIS Pro Alternative):** 
  1. Open the Geoprocessing pane and search for the **Spatial Analyst -> Hydrology** toolset.
  2. Run the **Fill** tool on your DEM to remove sinks.
  3. Run the **Flow Direction** tool on the Filled DEM.
  4. Run the **Flow Accumulation** tool on the Flow Direction output.
  5. Change the Symbology of the Accumulation layer to a standard deviation stretch to visualize the river network!

---

## 🚀 How to Run

### 1. Install Chapter Dependencies
Activate your environment and ensure the required packages are installed:
```bash
mamba activate geocascade_env
mamba install -c conda-forge pysheds pystac-client planetary-computer rasterio pyproj matplotlib numpy -y
```

### 2. Run Glacial Retreat Analysis (Script 09)
Fetch historical Landsat data and map the melting of Grey Glacier:
```bash
python 09_multitemporal_glacier_retreat.py
```

### 3. Run DEM Terrain Processing (Script 10)
Calculate Slope, Aspect, and Hillshade using physics formulas:
```bash
python 10_digital_elevation_processing.py
```

### 4. Run Hydrological Delineation (Script 11)
Trace water flow and map river networks:
```bash
python 11_watershed_delineation.py
```

---

## 🗺️ GIS Interoperability (ArcGIS Pro & ENVI)
A core academic requirement of this curriculum is **Hybrid Dual-Track Interoperability**. Every Python script in this chapter has been engineered to automatically export **Geocoded TIFF (.tif)** and **Shapefile (.shp)** outputs into the \data/processed/\ directory.

Instead of just looking at matplotlib PNG graphs, students are encouraged to:
1. Run the automated Python pipeline.
2. Open **ArcGIS Pro** or **ENVI**.
3. Drag-and-drop the generated \.tif\ and \.shp\ files directly into the GUI.
4. Verify that the Python outputs align perfectly with the GUI software basemaps.
