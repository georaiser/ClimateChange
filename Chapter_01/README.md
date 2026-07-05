# Chapter 1: Climatic Variables & Image Processing

## 🎯 Academic Objective
The goal of this chapter is to establish a strong foundation in programmatic geospatial data acquisition, atmospheric correction, and climate modeling. 

Rather than manually clicking through web portals (like EarthExplorer or Copernicus Browser), we will build scripts that act as automated data pipelines. We will transform raw satellite imagery into **Analysis-Ready Data (ARD)** and use Machine Learning to interpolate local weather station records into continuous spatial maps.

---

## 🛠️ Scripts & Modules

### `01_stac_multisensor_download.py`
- **Concept:** Spatiotemporal Asset Catalogs (STAC) are the modern standard for querying Earth Observation data via APIs. This avoids downloading entire satellite tiles when you only need a specific bounding box (BBOX).
- **Action:** A pure Python script utilizing `pystac-client` and `planetary-computer` (or AWS Earth Search) to search and download Sentinel-2, Landsat, and DEM data for the Torres del Paine Region of Interest (ROI). 
- **Dependencies:** `pystac-client`, `planetary-computer`, `rasterio`

### `02_atmospheric_correction.py`
- **Concept:** Sensors record Top of Atmosphere (TOA) reflectance, which includes atmospheric scattering (haze). True surface analysis requires Bottom of Atmosphere (BOA) reflectance.
- **Action (Python):** A Python implementation utilizing `rasterio` to mathematically correct TOA to BOA using Dark Object Subtraction (DOS1), replicating standard GIS tools.
- **Action (ENVI Alternative):** 
  1. Open ENVI and load the raw optical imagery (L1C/TOA).
  2. Navigate to *Radiometric Correction* -> *Atmospheric Correction Module* -> *FLAASH Atmospheric Correction*.
  3. Input your sensor type, flight date/time, and atmospheric model (e.g., Sub-Arctic Summer for Patagonia).
  4. Run FLAASH to generate the BOA surface reflectance raster.
- **Action (ArcGIS Pro Alternative):** 
  1. Load the raw raster into ArcGIS Pro.
  2. If using Landsat, go to *Imagery* tab -> *Raster Functions* -> *Apparent Reflectance* (this does basic TOA).
  3. For DOS1 (Haze Removal), use the *Raster Calculator*: find the minimum pixel value in the dark areas, and subtract it: `"Band_1" - Min_Value`. Ensure you set negative values to 0 using the `Con()` statement.

### `03_station_ml_interpolation.py`
- **Concept:** In-situ weather stations provide accurate but sparse point data. We use the surrounding topography (elevation, latitude) as covariates to predict the climate in unmonitored areas.
- **Action:** Trains a Random Forest regressor (`scikit-learn`) on known station data to generate a continuous raster surface of temperature/precipitation.

### `04_precipitation_dual_analysis.py`
- **Concept:** Comparing historical anomalies to identify drought/flood cycles.
- **Action:** Analyzes the CHIRPS precipitation dataset and clusters historical climates using unsupervised ML.

### `05_uhi_modis_mapping.py`
- **Concept:** Deriving Land Surface Temperature (LST) from thermal radiance.
- **Action:** Maps the Urban Heat Island effect over Punta Arenas using thermal satellite data.

---

## 🚀 How to Run
### 1. Install Chapter Dependencies
Since we are using a modular approach, activate your base environment and install the required packages for Chapter 1:
```bash
mamba activate geocascade_env
mamba install -c conda-forge pystac-client planetary-computer requests rasterio -y
```

### 2. Execute Scripts
Then, execute the scripts sequentially:
```bash
python 01_stac_multisensor_download.py
python 02_atmospheric_correction.py
```
