# Master Specialization Proposal: Comprehensive Climate, Glacial, and Watershed Analysis

This proposal details a massive, unified academic implementation plan merging the objectives of Climate Change, Glacial Retreat, and Watershed Modeling analysis. It leverages **ArcGIS Pro (ArcHydro), ENVI, RStudio, and pure Python environments** to create a robust analytical pipeline. 

Furthermore, to elevate the project to an enterprise-grade Geo-AI architecture, the specialization progresses into advanced topics including **Agentic Orchestration (LangGraph), Multi-Task PyTorch, Multi-Temporal Change Detection, PostGIS, and Full-Stack web development**. 

The learning process is divided into 11 sequential, academic **Chapters**, ensuring a step-by-step understanding of how to build and scale these complex tools.

---

## Summary of Resources and Techniques
**Data Resources:**
*   **Optical & Thermal Sensors:** Landsat, Sentinel-2, MODIS.
*   **Radar & Elevation:** Sentinel-1 SAR, SRTM / ALOS PALSAR DEMs.
*   **Climatic Datasets:** WorldClim, CHIRPS.
*   **In-Situ Data:** Local meteorological weather stations (e.g., CR2 Chile).

**Advanced Analytical Techniques:**
*   **Agentic AI:** LangGraph for autonomous geospatial orchestration.
*   **Deep Learning (PyTorch):** Multi-task CNNs for simultaneous lake, snow, and change detection segmentation.
*   **Multi-Temporal Change Detection:** Time-series analysis across decades using fused SAR and Optical data.
*   **Hydrological Modeling (ArcHydro / PySheds):** Watershed delineation and water balance.
*   **Spatial Databases:** PostgreSQL + PostGIS for scalable vector storage.
*   **Full-Stack Web GIS:** FastAPI + React + MapLibre GL JS for interactive dashboards.

## Selected Region of Interest (ROI)
**Torres del Paine National Park & Punta Arenas (Grey River Basin), Magallanes Region, Chile.**

---

## Detailed Project Structure and Academic Flow

### CHAPTER 1: Climatic Variables & Image Processing
**Academic Objective:** Foundation in data acquisition, atmospheric correction, and climate modeling.

#### `Chapter_01/`
*   **`01_stac_multisensor_download.py`:** (Concept: Programmatic acquisition). Uses Copernicus API and AWS STAC to download multi-sensor data.
*   **`02_atmospheric_correction.py`:** (Concept: TOA to BOA reflectance physics). Radiometric/Atmospheric correction using Python (`rasterio`/`Py6S`) and ENVI.
*   **`03_station_ml_interpolation.py`:** (Concept: ML for missing data). Training Random Forest on weather station data to predict continuous climate surfaces. **Includes automated Sensor Failure Anomaly Detection (`IsolationForest`) and manual visual QA/QC.**
*   **`04_precipitation_dual_analysis`:** (Concept: R vs Python spatial analysis). CHIRPS historical anomalies and ArcGIS IsoCluster climatic zone classification. **Includes explicit R (`anomalize`) and ArcGIS Pro (Space-Time Pattern Mining) extreme climate event anomaly detection.**
*   **`05_uhi_modis_mapping.py`:** (Concept: Urban microclimates). Urban Heat Island mapping over Punta Arenas using thermal data.

---

### CHAPTER 2: Glacial Retreat, SAR Dynamics, and AI
**Academic Objective:** Tracking ice loss using multi-sensor approaches (Radar/Optical) and foundational Deep Learning.

#### `Chapter_02/`
*   **`01_glacier_area_perimeter.py`:** (Concept: Vector GIS change detection). ArcGIS Pro/Python script to calculate historical Grey Glacier retreat.
*   **`02_sentinel1_sar_analysis.py`:** (Concept: Cloud-penetrating radar physics). Classify dry snow vs. wet snow vs. ice using SAR backscatter.
*   **`03_xarray_ndwi_multitemporal.py`:** (Concept: Spectral indices). Mapping Grey Lake expansion across decades.
*   **`04_pytorch_lake_segmentation.py`:** (Concept: Semantic segmentation). U-Net CNN for automated glacial lake extraction.
*   **`05_arcpy_ndsi_snow_cover.py`:** (Concept: Snow tracking). Automated ArcGIS Pro script for NDSI mapping.
*   **`06_multitemporal_change_detection.py`:** (Concept: Multi-temporal sensor fusion). Fusing Sentinel-1 SAR and Sentinel-2 Optical time-series data to algorithmically detect land-cover changes and glacial calving events between two dates (T1 vs T2). **Includes ENVI RX Anomaly Detection and Python Time-Series analysis for predicting Glacial Lake Outburst Floods (GLOFs).**

---

### CHAPTER 3: Hydrology and Watershed Modeling
**Academic Objective:** Understanding the mechanics of water flow, terrain modeling, and basin management. 

#### `Chapter_03/`
*   **`01_dem_hydro_conditioning.py`:** (Concept: Terrain correction). Filling sinks and applying TIN corrections to DEMs.
*   **`02_archydro_basin_delineation.py`:** (Concept: Automated flow routing). Using ArcHydro and PySheds to delineate the Grey River Watershed.
*   **`03_morphometric_parameters.py`:** (Concept: Basin physical traits). Calculate form factor, mean slope, and plot the Hypsometric Curve.
*   **`04_drainage_and_strahler.py`:** (Concept: Stream hierarchy). Calculating drainage density and Strahler stream order.
*   **`05_water_balance_isohyets.py`:** (Concept: Spatial hydrology inputs/outputs). Generate Isohyets and Isotherms for a spatial Water Balance model.

---

### CHAPTER 4: Vulnerability and Ecosystem Impacts
**Academic Objective:** Understanding the biological and human consequences of the physical changes.

#### `Chapter_04/`
*   **`01_maxent_modeling` (R/Python):** (Concept: Species Distribution Modeling). Map how flora/fauna will migrate as the watershed's climate shifts.
*   **`02_vulnerability_index_mce.py`:** (Concept: Multi-Criteria Evaluation). Combine precipitation, glacial retreat, and flood risks into a "Climate Vulnerability Heatmap". **Includes Vegetation Stress/Burn Scar Anomaly Detection using R (`bfast`), ArcGIS Pro (CCDC), and manual visual interpretation.**

---

### CHAPTER 5: Cartography and Management Plans
**Academic Objective:** Communicating spatial data clearly via standard GIS outputs.

#### `Chapter_05/`
*   **`01_map_automation_layout.py`:** (Concept: Cartographic automation). `arcpy.mp` script generating standardized PDF maps.
*   **`02_watershed_management_report.py`:** (Concept: Reporting). Compile stats, water balance, and vulnerability indices into a final text/PDF management plan.

---

### CHAPTER 6: Capstone Project - The Linear Unified Pipeline
**Academic Objective:** Synthesize previous modules into a single, automated codebase.

#### `Chapter_06/`
*   **`main_linear_pipeline.py`:** (Concept: System integration). A massive Python script that linearly triggers downloads, preprocessing, AI inference, watershed delineation, and report generation in one continuous execution loop.

---

## ADVANCED ARCHITECTURES

### CHAPTER 7: Agentic Orchestration (LangGraph)
**Academic Objective:** Move from rigid, linear scripting to autonomous, state-machine driven AI agents.

#### `Chapter_07/`
*   **Concept:** Instead of a script running top-to-bottom, we build specialized AI agents that "decide" when to act based on the data state.
*   **Action:** Build a LangGraph network with an `AcquisitionAgent`, `GlacierAgent`, `HydrologyAgent`, and `ReportAgent`. They autonomously pass data payloads (like GeoTIFFs) between each other, handling errors and retries intelligently.

### CHAPTER 8: The Cascade Effect Modeling
**Academic Objective:** Move from isolated metrics to modeling a physical chain-reaction.

#### `Chapter_08/`
*   **Concept:** Disasters don't happen in isolation. We will programmatically link the models to prove cause-and-effect.
*   **Action:** A Python framework that specifically tracks the statistical causality: *Temperature Anomaly (Ch1) → Lake Expansion (Ch2) → Altered Water Balance (Ch3) → High Vulnerability Niche Shift (Ch4).* **Includes multi-platform validation against manually detected events.**

### CHAPTER 9: Multi-Task PyTorch Model
**Academic Objective:** Upgrade standard Deep Learning to advanced multi-head architectures.

#### `Chapter_09/`
*   **Concept:** Running multiple separate neural networks is computationally expensive. Multi-task learning predicts several things at once.
*   **Action:** Upgrade the U-Net from Chapter 2 into a **Multi-Head CNN**. A single PyTorch model that ingests Optical, SAR, and DEM tensors simultaneously to output masks in one forward pass: 
    *   (1) Glacial Lake Segmentation
    *   (2) Snow/Ice Cover
    *   (3) Flood Risk Zones
    *   **(4) Multi-Temporal Change Detection Head:** Anomaly scoring that highlights structural changes in the terrain between two different years.

### CHAPTER 10: Spatial Database Integration (PostGIS)
**Academic Objective:** Move from flat files (Shapefiles/GeoTIFFs) to enterprise spatial databases.

#### `Chapter_10/`
*   **Concept:** Flat files are hard to query over time. Databases allow for massive spatial SQL queries.
*   **Action:** Set up a PostgreSQL + PostGIS database via Docker. Write Python (`SQLAlchemy`, `GeoAlchemy2`) scripts to ingest the output polygons (lakes, watersheds, vulnerability zones) and raster metadata directly into the database.

### CHAPTER 11: Interactive Dashboard (Full-Stack GIS)
**Academic Objective:** Move from static PDF maps to real-time, interactive Web GIS.

#### `Chapter_11/`
*   **Concept:** Stakeholders need interactive platforms, not just PDFs.
*   **Action:** Build a **FastAPI** backend that queries the PostGIS database, and a **React + MapLibre GL JS** frontend to visualize the Climate Cascade, Glacial Retreat, and Watershed changes dynamically over a web browser.
