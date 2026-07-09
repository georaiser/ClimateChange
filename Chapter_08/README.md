# Chapter 8: Data Fusion & The Cascade Effect Modeling

## 🎯 Academic Objective
In the previous chapters, we analyzed physical properties in isolation: Climate Variables (Ch 1), Spectral Signatures (Ch 2), Topography (Ch 3), and Radar Backscatter (Ch 7). 

However, in the real world, earth systems are interconnected. A shift in one system triggers a **Cascade Effect** (e.g., Temperature Rise $\rightarrow$ Glacier Melt $\rightarrow$ Flood Risk $\rightarrow$ Vegetation Shift). 

To model this, we must use **Multi-Sensor Data Fusion** and **Machine Learning**. This chapter teaches you how to geometrically align vastly different satellite datasets into a single multidimensional "Data Cube", and feed that cube into an Artificial Intelligence algorithm (Random Forest) to uncover hidden patterns.

---

## 🛠️ Scripts & Modules

### `20_multisensor_data_fusion.py`
- **Concept:** Satellites capture data at different spatial resolutions (e.g., MODIS is 1km, DEM is 30m, Sentinel-2 is 10m) and in different Coordinate Reference Systems (CRS). You cannot perform pixel-by-pixel mathematical operations unless they are perfectly aligned.
- **Action:** This script hits the STAC API to download:
  1. Sentinel-2 Optical (NIR band)
  2. Sentinel-1 Radar (VV polarization)
  3. Copernicus DEM (Elevation)
  4. MODIS (Thermal LST)
- It uses `rasterio.warp.reproject` to mathematically resample all layers to match the exact 10m grid and CRS of the Sentinel-2 image. It stacks them into a 4-band `cascade_master_stack.tif`.

### `21_cascade_risk_modeling.py`
- **Concept:** With our Data Cube built, we can train an algorithm to recognize complex, multi-variable signatures. For example, a glacier isn't just "white" (Optical); it's also "cold" (Thermal), "rough" (Radar), and "high" (Elevation).
- **Action:** We use `scikit-learn` to train a **Random Forest Classifier**. We provide the algorithm with synthetic training labels for Glaciers, Water, and Vegetation based on logical multi-sensor thresholds. The AI learns these patterns and predicts a complete "Vulnerability / Land Cover Map" for the entire region.
- **Output:** A classified GeoTIFF and a visually stunning `.png` map representing the final synthesized insight.

---

## 🚀 How to Run

### 1. Data Fusion (Stacking)
This script takes several minutes because it must download and dynamically resample gigabytes of data across 4 different satellite constellations.
```bash
python 20_multisensor_data_fusion.py
```

### 2. Machine Learning (Random Forest)
```bash
python 21_cascade_risk_modeling.py
```

---

## 🗺️ GIS Interoperability
Because we exported a massive multi-band stack (`cascade_master_stack.tif`), you can load this into **ENVI**. 
In ENVI, you can assign different bands to RGB (e.g., Red = Radar, Green = Optical, Blue = Thermal) to create bizarre and highly informative false-color composites! 

You can also use ArcGIS Pro's **Image Analyst $\rightarrow$ Classification Wizard** to run Support Vector Machines (SVM) or Random Forests directly on our exported Data Cube via the GUI.
