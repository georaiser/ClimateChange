# Chapter 2: Spectral Signatures & Environmental Monitoring

## 🎯 Academic Objective
This chapter bridges the gap between physics and environmental science. We will explore how different Earth materials interact with light, and then scale those physical properties into landscape-level spectral indices to monitor vegetation, soil, and water health.

We use a **Hybrid Dual-Track Methodology**: for every concept, we provide the automated Python script for mass processing, alongside the manual workflow for commercial GUI software (ArcGIS Pro / ENVI).

---

## 🛠️ Scripts & Modules

### `06_spectral_signature_analysis.py`
- **Concept:** Every material on Earth (Glacial Ice, Patagonian Forest, Bare Rock) absorbs and reflects light differently across the electromagnetic spectrum. This unique pattern is called a "Spectral Signature" or "Spectral Fingerprint."
- **Action (Python Cloud-Native):** Instead of downloading massive gigabytes of imagery, this script connects to the Microsoft Planetary Computer STAC API. It finds the Cloud Optimized GeoTIFF (COG) URLs for Sentinel-2 bands and uses `rasterio` to stream *only the specific pixels* located at our target coordinates directly from the cloud! It then plots a Spectral Signature graph.
- **Action (ENVI Alternative):** 
  1. Open ENVI and load your stacked Sentinel-2 or Landsat image.
  2. Click the **Spectral Profile** icon on the toolbar.
  3. Click on a pixel of a glacier, a forest, and bare rock in your image. The profile window will plot their signatures.
  4. To identify an unknown material, go to *Options -> Spectral Library*. Open the built-in USGS Spectral Library (e.g., `usgs_min.sli` or `veg.sli`).
  5. Use the *Spectral Angle Mapper (SAM)* tool (*Classification -> Supervised -> Spectral Angle Mapper*) to automatically classify the entire image based on those signatures!

### `07_vegetation_soil_indices.py`
- **Concept:** By dividing and multiplying the physical reflectances of different bands, we can isolate specific environmental phenomena. NDVI, EVI, and SAVI measure vegetation health and biomass. BSI isolates bare rock and exposed soil to assess erosion risk.
- **Action (Python Automation):** Downloads a specific mathematical Window (BBOX) of the Sentinel-2 image directly from the cloud. It resamples the 20m SWIR band to 10m on-the-fly, calculates the four indices, and outputs a 4-panel comparison heatmap. It also natively exports `ndvi.tif`, `evi.tif`, `savi.tif`, and `bsi.tif` for immediate loading into GIS software.
- **Action (ArcGIS Pro Alternative):** 
  1. Load the Sentinel-2 bands into ArcGIS Pro.
  2. To calculate NDVI quickly, navigate to the **Imagery** tab -> **Indices** gallery -> **NDVI**. Select your Red and NIR bands.
  3. To calculate custom indices like BSI or EVI, open the **Raster Calculator** tool.
  4. Type the mathematical formula manually (e.g., `("B11" + "B04" - "B08" - "B02") / ("B11" + "B04" + "B08" + "B02")`).
  5. Apply a color ramp (Symbology) to visualize the result.

### `08_automated_index_batcher.py`
- **Concept:** In industry, environmental monitoring requires calculating indices for dozens or hundreds of images over time to see trends. 
- **Action (Python Automation):** We use a `for` loop to query the STAC API for *every single cloud-free image in 2023*. For every image found, the script automatically crops the image, extracts the B04 and B08 bands, calculates NDVI, and saves a cropped `.tif` file to the hard drive.
- **Action (ArcGIS Pro Alternative):** 
  1. Open ArcGIS Pro and navigate to **Analysis** -> **ModelBuilder**.
  2. Drag and drop the **Iterate Rasters** tool into your model. Point it at a folder full of downloaded Sentinel-2 imagery.
  3. Drag in the **Raster Calculator** and connect the Iterator's output to it.
  4. Set up your mathematical equation, and right-click to save the output dynamically using `%Name%_NDVI.tif`.
  5. Click **Run** to execute the visual loop.

---

## 🚀 How to Run

### 1. Install Chapter Dependencies
Activate your environment and ensure the required packages are installed:
```bash
mamba activate geocascade_env
mamba install -c conda-forge pystac-client planetary-computer rasterio pyproj matplotlib numpy -y
```

### 2. Run Spectral Signature Extraction (Script 06)
Extract and plot the spectral signatures of Ice, Forest, and Rock using Cloud-Native streaming:
```bash
python 06_spectral_signature_analysis.py
```

### 3. Calculate Environmental Indices (Script 07)
Compute NDVI, EVI, SAVI, and BSI dynamically:
```bash
python 07_vegetation_soil_indices.py
```

### 4. Run the Automated Batch Processor (Script 08)
Loop through all of 2023 and batch process NDVI files:
```bash
python 08_automated_index_batcher.py
```

---

## 🗺️ GIS Interoperability (ArcGIS Pro & ENVI)
A core academic requirement of this curriculum is **Hybrid Dual-Track Interoperability**. Every Python script in this chapter has been engineered to automatically export **Geocoded TIFF (.tif)** and **Shapefile (.shp)** outputs into the \data/processed/\ directory.

Instead of just looking at matplotlib PNG graphs, students are encouraged to:
1. Run the automated Python pipeline.
2. Open **ArcGIS Pro** or **ENVI**.
3. Drag-and-drop the generated \.tif\ and \.shp\ files directly into the GUI.
4. Verify that the Python outputs align perfectly with the GUI software basemaps.
