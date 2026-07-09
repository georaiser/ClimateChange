# Chapter 12: Robust Site Analysis Capstone

## 🎯 Academic Objective
This is the **Basic Capstone Project** for the physical sciences block (Chapters 1-6). 
Your objective is to execute a complete, real-world geospatial site analysis from start to finish. You will not only use Raster data (DEM, Satellite Imagery), but you will dynamically source real-world **Vector Data (Shapefiles)** from OpenStreetMap, run advanced Vector-Raster overlays, and automatically generate a final report.

## 🛠️ The Assignment

### Task 1: Choose Your Coordinates
In `capstone_pipeline.py`, locate the `BBOX` variable at the top of the file. Change these coordinates to a bounding box of your choosing (e.g., your hometown, a national park, or a region of interest in Patagonia).

### Task 2: Automated Python Baseline
Run the master script:
```bash
mamba activate geocascade_env
python capstone_pipeline.py
```
This script will:
1. Dynamically query the Overpass API (OpenStreetMap) to download the real-world **Road Network** for your BBOX and export it as a Shapefile.
2. Connect to the Microsoft STAC API to download the **Copernicus DEM** and **Sentinel-2 L2A** imagery.
3. Compute **Slope** and **NDVI**.
4. Perform **Vector-Raster Overlay**: It will create a 1km Buffer around the road network and calculate the exact vegetation health (NDVI) and average steepness strictly within the vicinity of human infrastructure.
5. Generate a comprehensive `site_analysis_report.md` automatically.

### Task 3: Manual ArcGIS Pro Replication
Once the Python script proves the analysis is possible, you must replicate the *exact same workflow* in ArcGIS Pro to prove you understand the GUI tools.
1. Load the generated `dem.tif`, `ndvi.tif`, and `roads.shp` (or download them manually).
2. Use the **Buffer** tool to create a 1km zone around the roads.
3. Use **Zonal Statistics as Table** to calculate the mean NDVI and Slope inside that buffer.

### Task 4: Manual ENVI Replication
Replicate the analysis in ENVI.
1. Load the rasters and the shapefile.
2. Build a vector buffer and convert it to a Region of Interest (ROI).
3. Use the **ROI Statistics** tool to calculate the mean values.

### Task 5: Final Submission
Submit your generated `site_analysis_report.md` along with a 2-page essay comparing the execution speeds, automation capabilities, and ease-of-use of Python vs. ArcGIS Pro vs. ENVI.

---

## 🗺️ GIS Interoperability (ArcGIS Pro & ENVI)
A core academic requirement of this curriculum is **Hybrid Dual-Track Interoperability**. Every Python script in this chapter has been engineered to automatically export **Geocoded TIFF (.tif)** and **Shapefile (.shp)** outputs into the `data/processed/` directory.

Instead of just looking at matplotlib PNG graphs, students are encouraged to:
1. Run the automated Python pipeline.
2. Open **ArcGIS Pro** or **ENVI**.
3. Drag-and-drop the generated `.tif` and `.shp` files directly into the GUI.
4. Verify that the Python outputs align perfectly with the GUI software basemaps.
