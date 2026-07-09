# Chapter 5: Advanced Environmental Monitoring & Zonal Statistics

## 🎯 Academic Objective
This chapter introduces advanced techniques for evaluating the health of an ecosystem beyond basic greenness (NDVI). We will evaluate Moisture Stress and Drought conditions. Furthermore, we will move from raw pixel analysis to concrete quantitative reporting using "Áreas Poligonales" (Zonal Statistics).

---

## 🛠️ Scripts & Modules

### `14_moisture_stress_indices.py`
- **Concept:** Water strongly absorbs Shortwave Infrared (SWIR) light. By comparing Near-Infrared (NIR) with SWIR, we can measure the exact water content inside plant leaves and soil.
- **Indices Calculated:**
  - **NDMI (Normalized Difference Moisture Index):** Measures water stress.
  - **MSI (Moisture Stress Index):** An inverted index where high values equal high drought stress.
- **Action (Python Automation):** Streams Sentinel-2 data to calculate these indices and export geocoded TIFFs (`ndmi.tif`, `msi.tif`).
- **Action (ArcGIS Pro Alternative):** 
  1. Load Sentinel-2 NIR and SWIR1 bands.
  2. Use the **Raster Calculator**.
  3. Formula for NDMI: `("NIR" - "SWIR1") / ("NIR" + "SWIR1")`

### `15_zonal_statistics.py`
- **Concept:** Environmental managers need statistical summaries for specific areas (e.g., "What is the average vegetation health in Zone A?").
- **Action (Python Automation):** Uses `geopandas` to generate synthetic Management Zones (Polygons). Uses `rasterstats` to intersect these polygons with the DEM and NDVI rasters, calculating the mean, min, and max values strictly within the boundaries.
- **Action (ArcGIS Pro Alternative):** 
  1. Have a Polygon feature class (Shapefile) of your zones and a Raster (e.g., NDVI).
  2. Go to **Spatial Analyst Tools -> Zonal -> Zonal Statistics as Table**.
  3. Input your Zones and your Raster. Choose "Mean" as the statistic type.

---

## 🚀 How to Run

### 1. Install Chapter Dependencies
```bash
mamba activate geocascade_env
mamba install -c conda-forge geopandas rasterstats shapely rasterio pyproj pystac-client planetary-computer matplotlib numpy -y
```

### 2. Run Moisture Indices
```bash
python 14_moisture_stress_indices.py
```

### 3. Run Zonal Statistics
```bash
python 15_zonal_statistics.py
```

---

## 🗺️ GIS Interoperability (ArcGIS Pro & ENVI)
A core academic requirement of this curriculum is **Hybrid Dual-Track Interoperability**. Every Python script in this chapter has been engineered to automatically export **Geocoded TIFF (.tif)** and **Shapefile (.shp)** outputs into the \data/processed/\ directory.

Instead of just looking at matplotlib PNG graphs, students are encouraged to:
1. Run the automated Python pipeline.
2. Open **ArcGIS Pro** or **ENVI**.
3. Drag-and-drop the generated \.tif\ and \.shp\ files directly into the GUI.
4. Verify that the Python outputs align perfectly with the GUI software basemaps.
