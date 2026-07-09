# Chapter 6: Advanced Hydrometeorology & Basin Metrics

## 🎯 Academic Objective
This chapter expands upon our hydrological modeling from Chapter 3. We dive deeper into vector-raster integration to create continuous climate contour lines (Isohyets/Isotherms) and evaluate the risk of flooding and erosion by calculating **Drainage Density**.

---

## 🛠️ Scripts & Modules

### `16_isohyets_isotherms.py`
- **Concept:** Climate variables are continuous. To map them cleanly, we generate "isolines" (contours). Isotherms are lines of equal temperature.
- **Action (Python Automation):** Downloads the 30m Copernicus DEM and applies a standard Environmental Lapse Rate (-6.5°C per 1000m) to mathematically simulate a high-resolution temperature raster. It then uses contouring algorithms to extract vector Isotherms and saves them as a Shapefile.
- **Action (ArcGIS Pro Alternative):** 
  1. Have a temperature raster loaded.
  2. Navigate to **Spatial Analyst Tools -> Surface -> Contour**.
  3. Set your input raster, define the contour interval (e.g., every 2 degrees), and run the tool.

### `17_drainage_density.py`
- **Concept:** Drainage Density measures how well a basin drains water. High density = highly impermeable rock/steep slopes = high runoff/flood risk. Formula: Total River Length / Basin Area.
- **Action (Python Automation):** Uses `pysheds` to delineate the flow accumulation and extracts the river branches into a vector `geopandas` dataframe. It reprojects the data to a metric CRS (UTM) to calculate the precise lengths in meters.
- **Action (ArcGIS Pro Alternative):** 
  1. Generate your River Network using Flow Accumulation and the `Con` tool.
  2. Run the **Stream to Feature** tool to convert the raster rivers to vector lines.
  3. Run the **Line Density** tool (Spatial Analyst) to calculate the spatial distribution of the river network, or manually sum the `Shape_Length` column in the Attribute Table.

---

## 🚀 How to Run

### 1. Install Chapter Dependencies
```bash
mamba activate geocascade_env
mamba install -c conda-forge pysheds geopandas shapely rasterio pyproj pystac-client planetary-computer matplotlib numpy -y
```

### 2. Generate Isotherms (Script 16)
```bash
python 16_isohyets_isotherms.py
```

### 3. Calculate Drainage Density (Script 17)
```bash
python 17_drainage_density.py
```
