# 🌍 GeoCascade Pipeline — Project Overview

> **Geospatial Climate Change Analysis**
> Torres del Paine & Patagonia, Chile
> Sentinel-2 · Landsat · Copernicus DEM · CHIRPS · NOAA Stations

---

## What Is This Project?

GeoCascade is a full-stack geospatial analysis pipeline that combines satellite remote sensing, terrain analysis, machine learning, and climate data to study the cascading effects of climate change in Patagonia. Starting from raw satellite imagery and weather station data, it progresses through atmospheric correction, vegetation indices, glacier retreat mapping, watershed hydrology, species habitat modeling, and cloud-native geospatial REST APIs.

**Study Area:** Torres del Paine National Park, Patagonia, Chile
**BBOX:** `[-73.30, -51.10, -72.90, -50.80]` (WGS84 / EPSG:4326)
**Tools:** Python · ENVI 5.6 (IDL + Python API) · ArcGIS Pro (ArcPy)
**Environment:** `conda activate geocascade_env`

---

## 📚 Chapter Structure

| Chapter | Theme | Status |
|---------|-------|--------|
| [Chapter 1](#chapter-1--climate-data-acquisition--preprocessing) | Climate Data Acquisition & Preprocessing | ✅ Complete |
| [Chapter 2](#chapter-2--spectral-analysis--vegetation-indices) | Spectral Analysis & Vegetation Indices | ✅ Complete |
| [Chapter 3](#chapter-3--terrain--glacier-analysis) | Terrain & Glacier Analysis | ✅ Complete |
| [Chapter 4](#chapter-4--ecological--vulnerability-analysis) | Ecological & Vulnerability Analysis | ✅ Complete |
| [Chapter 5](#chapter-5--moisture-zonal-statistics) | Moisture & Zonal Statistics | 🔄 In Progress |
| [Chapter 6](#chapter-6--isohyets-drainage-density) | Isohyets & Drainage Density | 🔄 In Progress |
| [Chapter 7](#chapter-7--sar--multisensor-analysis) | SAR & Multisensor Analysis | 🔄 In Progress |
| [Chapter 8](#chapter-8--data-fusion--cascade-risk) | Data Fusion & Cascade Risk Modeling | 🔄 In Progress |
| [Chapter 9](#chapter-9--deep-learning-land-cover) | Deep Learning Land Cover (PyTorch) | 📋 Planned |
| [Chapter 10](#chapter-10--agentic-orchestration) | Agentic Orchestration (LangGraph) | 📋 Planned |
| [Chapter 11](#chapter-11--enterprise-spatial-databases) | Enterprise Spatial Databases (PostGIS) | 📋 Planned |
| [Chapter 12](#chapter-12--capstone) | Capstone — Full Pipeline Integration | 📋 Planned |
| [Chapter 13-14](#chapter-13-14--advanced-capstone--rest-api) | Advanced Capstone — Geospatial REST API | 📋 Planned |

---

## Script Catalog

---

### Chapter 1 — Climate Data Acquisition & Preprocessing

**`Chapter_01/`** | 9 scripts | All updated July 2026

| Script | Description | Key Outputs |
|--------|-------------|-------------|
| `01_data_download.py` | Downloads all raw data: NOAA weather stations via CDO API, CHIRPS precipitation, ERA5 climate variables, and queues STAC metadata from Planetary Computer. Entry point of the pipeline. | `data/raw/real_data/weather_stations.csv`, CHIRPS GeoTIFFs, ERA5 NetCDF |
| `02_satellite_acquisition.py` | Queries Planetary Computer STAC API for Sentinel-2 L2A and Landsat 9 L2SP imagery over the Torres del Paine BBOX. Streams windowed reads of all 10 bands (B02–B12), applies scale factors, writes GeoTIFFs with `nodata=-9999`. Includes ENVI `.hdr` sidecar generation. | `data/raw/sentinel2_l2a_*/B0*.tif`, `data/raw/landsat9_l2sp_*/B*.tif` |
| `03_atmospheric_correction.py` | Implements the COST Model (Chavez 1996): Dark Object Subtraction + Solar Zenith transmittance correction. Applies to Sentinel-2 L1C or Landsat L1TP raw data only — **never** apply to L2A/L2SP (already corrected). Includes DOS1 vs COST comparison plot. | `data/processed/boa_corrected/BOA_*.tif`, `correction_comparison.png`, `correction_report.csv` |
| `04_climate_trend_analysis.py` | Applies Mann-Kendall trend test and Sen's slope estimator to multi-decade temperature and precipitation records. Detects statistically significant climate trends (p < 0.05) without assuming normally distributed residuals. | `data/processed/climate_analysis/trend_analysis.png`, `trend_statistics.csv` |
| `05_chirps_precipitation.py` | Processes CHIRPS 0.05° rainfall data using windowed rasterio reads. Computes monthly climatology, seasonal anomalies (vs 1981–2010 baseline), and inter-annual variability. Uses `Agg` backend for headless execution. | `data/processed/chirps/chirps_seasonal_analysis.png`, `chirps_statistics.csv` |
| `06_station_interpolation.py` | Three-check QA/QC system (modified Z-score, stuck-sensor rule, physical bounds) removes bad weather station readings. Trains a Random Forest regressor with spatial holdout evaluation. Predicts a continuous temperature surface GeoTIFF over the study BBOX. | `data/processed/climate_analysis/temperature_surface.tif`, `anomaly_validation_plot.png`, `station_ml_analysis.png` |
| `07_precipitation_anomaly.py` | Computes standardized precipitation anomalies (Z-scores) vs the long-term mean. Maps drought and excess rainfall events. Computes Standardized Precipitation Index (SPI) at 3- and 12-month timescales. | `data/processed/climate_analysis/precipitation_anomaly.png`, `spi_statistics.csv` |
| `08_uhi_mapping.py` | Urban Heat Island analysis using Landsat 9 Band 10 thermal (TIRS). Converts raw DN to Land Surface Temperature (LST) via split-window algorithm. Computes LST difference: urban vs rural reference zones. | `data/processed/uhi/lst_map.tif`, `uhi_analysis.png`, `uhi_statistics.csv` |
| `09_chapter_report.py` | Automated chapter report generator. Reads all Chapter 1 processed outputs and compiles a 9-panel summary dashboard. Exports a structured CSV of all key metrics. Outputs to `data/processed/reports/`. | `data/processed/reports/chapter_01_report.png`, `chapter_01_metrics.csv` |

**Dependencies:** `rasterio, numpy, matplotlib, pandas, scikit-learn, scipy, pystac-client, planetary-computer, pyproj`

**ENVI integration:** `Chapter_01/envi/` — IDL `.pro` scripts for FLAASH atmospheric correction and spectral library comparison.
**ArcGIS Pro integration:** `Chapter_01/arcgis_pro/` — ArcPy notebooks for LST mapping and station data visualization.

---

### Chapter 2 — Spectral Analysis & Vegetation Indices

**`Chapter_02/`** | 3 scripts | All updated July 2026

| Script | Description | Key Outputs |
|--------|-------------|-------------|
| `06_spectral_signature_analysis.py` | Extracts spectral signatures (mean reflectance per band) for up to 8 land cover classes by sampling pixels inside user-defined polygons. Uses Sentinel-2 B02–B08A + B11–B12. Highlights the Red Edge (B05–B07) for vegetation stress detection. Falls back to Chapter 1 local data before streaming from Planetary Computer. | `data/processed/spectral_signatures.csv`, `spectral_signatures_main.png`, `red_edge_detail.png` |
| `07_vegetation_soil_indices.py` | Computes 9 spectral indices: NDVI, EVI, SAVI, NDWI, NDMI, NBR, BSI, NDSI, NDRE. Each exported as a separate LZW-compressed GeoTIFF (`nodata=-9999`). Produces binary masks for glacier (NDSI > 0.4) and water (NDWI > 0.0). B11 (20m) independently windowed and bilinearly resampled to 10m. | `data/processed/indices/ndvi.tif` … `ndre.tif`, `glacier_mask_ndsi.tif`, `water_mask_ndwi.tif`, `index_statistics.csv`, `vegetation_indices_panel.png` |
| `08_automated_index_batcher.py` | Batch-processes multiple STAC scenes for temporal index stacking. Incremental mode skips already-processed scenes. Produces a temporal animation figure (4-panel grid of NDVI over time). Tracks glacier fraction and water fraction per scene in a time-series CSV. | `data/processed/batch/batch_time_series.csv`, `temporal_ndvi_stack.png`, per-scene index GeoTIFFs |

**Indices formula reference:**

| Index | Formula | Sensitivity |
|-------|---------|------------|
| NDVI | (NIR − Red) / (NIR + Red) | Green biomass density |
| EVI | 2.5 × (NIR − Red) / (NIR + 6×Red − 7.5×Blue + 1) | Biomass in high-density canopy |
| SAVI | (NIR − Red) / (NIR + Red + 0.5) × 1.5 | Sparse vegetation on bare soil |
| NDWI | (Green − NIR) / (Green + NIR) | Open water bodies |
| NDMI | (NIR − SWIR1) / (NIR + SWIR1) | Plant moisture / drought stress |
| NBR | (NIR − SWIR2) / (NIR + SWIR2) | Burn severity |
| BSI | (SWIR1 + Red − NIR − Blue) / (SWIR1 + Red + NIR + Blue) | Bare soil exposure |
| NDSI | (Green − SWIR1) / (Green + SWIR1) | Snow and glacier ice (> 0.4) |
| NDRE | (RedEdge − Red) / (RedEdge + Red) | Chlorophyll stress (early senescence) |

---

### Chapter 3 — Terrain & Glacier Analysis

**`Chapter_03/`** | 3 scripts | All updated July 2026

| Script | Description | Key Outputs |
|--------|-------------|-------------|
| `09_multitemporal_glacier_retreat.py` | 20-year glacier change detection on Grey Glacier using Landsat. Computes NDSI for 2003 and 2023 scenes (after applying Landsat C2 L2 scale factor: `DN × 0.0000275 − 0.2`). Subtracts ice masks to produce a retreat map (+1=melted, 0=stable, −1=advanced). Quantifies area change in km². | `data/processed/glacier_retreat/glacier_retreat_2003_2023.tif`, `glacier_ndsi_2003/2023.tif`, `glacier_retreat_report.csv`, 4-panel dark figure |
| `10_digital_elevation_processing.py` | Downloads Copernicus DEM 30m tile and derives slope (degrees), aspect (0–360° from North), hillshade (ESRI Az=315 Ze=45 formula), and plan curvature (Laplacian). All `np.gradient()` calls use pixel sizes in **metres** (not degrees) to avoid 3300× underestimation of slope. Caches DEM to `data/raw/temp_dem.tif` for script 11. | `data/processed/terrain/slope_degrees.tif`, `aspect_degrees.tif`, `hillshade.tif`, `curvature.tif`, `terrain_statistics.csv`, 5-panel dark figure |
| `11_watershed_delineation.py` | Runs full pysheds D8 hydrological workflow: pit fill → depression fill → resolve flats → flow direction → flow accumulation → river network mask. Computes the Hipsometric Curve and Hipsometric Integral (HI) as a basin maturity indicator. HI > 0.6 = young/actively eroding. | `data/processed/watershed/flow_direction.tif`, `flow_accumulation.tif`, `river_network.tif`, `hypsometry_report.csv`, 4-panel dark figure |

**Run order:** `10 → 11` (script 10 must run first to cache `temp_dem.tif`). Script 09 is independent.

---

### Chapter 4 — Ecological & Vulnerability Analysis

**`Chapter_04/`** | 2 scripts | All updated July 2026

| Script | Description | Key Outputs |
|--------|-------------|-------------|
| `12_ecological_niche_modeling.py` | Species Distribution Model for the Patagonian Huemul Deer (endangered). Runs unsupervised K-Means clustering (k=4) on DEM+Slope+NDVI to discover natural terrain types, then trains a supervised Random Forest SDM on synthetic presence/absence data with spatial holdout. Reports ROC-AUC. Feature importance shows which predictors drive habitat suitability. | `data/processed/niche/kmeans_unsupervised.tif`, `ecological_niche_model.tif`, `feature_importance.csv`, `niche_statistics.csv`, 4-panel dark figure |
| `13_climate_vulnerability_index.py` | Multi-Criteria Decision Analysis (MCDA) climate vulnerability index. Combines 4 layers with fixed weights: Vegetation Exposure (1−NDVI, 50%), Erosion Risk (Slope, 20%), Elevation Exposure (DEM, 15%), Glacier Proximity (distance to NDSI ice, 15%). All standardized to [0,1] before weighting. Outputs 5-class vulnerability map. | `data/processed/vulnerability/vulnerability_index.tif`, `vulnerability_class.tif`, `vulnerability_statistics.csv`, 5-panel dark figure |

**Data sources (Ch04 reads from):**
- `Chapter_02/data/processed/indices/ndvi.tif`
- `Chapter_02/data/processed/indices/glacier_mask_ndsi.tif`
- `Chapter_03/data/processed/terrain/copernicus_dem.tif`
- `Chapter_03/data/processed/terrain/slope_degrees.tif`

All fallback to Planetary Computer STAC streaming if local files not found.

---

### Chapter 5 — Moisture & Zonal Statistics

**`Chapter_05/`** | 2 scripts

| Script | Description | Key Outputs |
|--------|-------------|-------------|
| `14_moisture_stress_indices.py` | Computes multi-scale moisture stress indicators from Sentinel-2 SWIR bands (NDMI, NMDI) and ERA5 evapotranspiration. Combines into a Composite Drought Indicator (CDI). | `moisture_stress.tif`, `drought_composite.png` |
| `15_zonal_statistics.py` | Overlays all raster indices (NDVI, CVI, SDM, LST) against watershed polygons from script 11 to compute per-basin statistics: mean, std, min, max, percentiles. Outputs summary table for cross-chapter comparison. | `zonal_statistics_summary.csv`, `zonal_statistics.png` |

---

### Chapter 6 — Isohyets & Drainage Density

**`Chapter_06/`** | 2 scripts

| Script | Description | Key Outputs |
|--------|-------------|-------------|
| `16_isohyets_isotherms.py` | Generates isohyet (equal rainfall) and isotherm (equal temperature) contour maps by interpolating point measurements with scipy RBF or kriging. Exports contours as GeoJSON for ArcGIS Pro. | `isohyets.geojson`, `isotherms.geojson`, `isohyet_map.png` |
| `17_drainage_density.py` | Computes drainage density (total stream length / basin area) from the river network produced by script 11. Higher drainage density = flashier catchment with rapid storm response. | `drainage_density.tif`, `drainage_density_report.csv` |

---

### Chapter 7 — SAR & Multisensor Analysis

**`Chapter_07/`** | 3 scripts

| Script | Description | Key Outputs |
|--------|-------------|-------------|
| `18_sentinel1_sar_processing.py` | Processes Sentinel-1 GRD SAR imagery (VV/VH polarizations). Converts to backscatter (dB), applies speckle filtering (Lee filter), and computes VH/VV ratio for surface roughness mapping. SAR penetrates cloud cover — critical for Patagonia. | `sar_vv.tif`, `sar_vh.tif`, `sar_ratio.tif`, `sar_processing.png` |
| `19_multisensor_review.py` | Qualitative comparison of Sentinel-2 optical vs Sentinel-1 SAR for the same BBOX and date. Side-by-side visualization of what each sensor "sees" through clouds. | `multisensor_comparison.png` |
| `19b_cloud_penetration_comparison.py` | Quantitative analysis: masks cloudy pixels in Sentinel-2, then shows SAR-derived surface reflectance estimate in the same pixels. Demonstrates SAR's cloud-penetration advantage. | `cloud_penetration_analysis.png`, `cloud_coverage_stats.csv` |

---

### Chapter 8 — Data Fusion & Cascade Risk Modeling

**`Chapter_08/`** | 4 scripts

| Script | Description | Key Outputs |
|--------|-------------|-------------|
| `20_multisensor_data_fusion.py` | Pixel-level fusion of Sentinel-2 optical, Sentinel-1 SAR, and Copernicus DEM into a single multi-band composite for classification. Uses weighted band combination and PCA for dimensionality reduction. | `fused_composite.tif`, `pca_components.tif` |
| `21_cascade_risk_modeling.py` | Combines glacier retreat rate, slope, flow accumulation, and vegetation cover into a Glacial Lake Outburst Flood (GLOF) risk index. Identifies pixels at risk of ice-dammed lake failure. | `glof_risk_index.tif`, `cascade_risk_report.csv` |
| `22_combined_insights_engine.py` | Master insight generator. Reads all processed outputs from Chapters 1–7, computes cross-variable correlations, and generates a comprehensive statistical report. | `combined_insights_report.md`, `correlation_matrix.png` |
| `23_real_data_convergence.py` | Validation script. Cross-checks model predictions against in-situ station measurements and known glacier extents from the GLIMS database. Computes RMSE and bias metrics. | `validation_report.csv`, `convergence_plot.png` |

---

### Chapter 9 — Deep Learning Land Cover

**`Chapter_09/`** | 1 script | 📋 Planned

| Script | Description |
|--------|-------------|
| `24_deep_learning_landcover.py` | Multi-task deep learning with a shared ResNet-50 backbone: (1) semantic land cover segmentation head, (2) change detection head. Trained on 256×256 Sentinel-2 patches. Exported to ONNX for production inference. |

**Install:** `mamba install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia`

---

### Chapter 10 — Agentic Orchestration

**`Chapter_10/`** | 1 script | 📋 Planned

| Script | Description |
|--------|-------------|
| `25_agentic_monitor.py` | LangGraph stateful agent that autonomously runs the GeoCascade pipeline when asked a natural-language question ("analyze glacial flood risk near [location]"). Has STAC query, NDVI calculator, SAR processor, and report generator tools. Human-in-the-loop pause before expensive API calls. |

**Install:** `pip install langgraph langchain langchain-google-genai`

---

### Chapter 11 — Enterprise Spatial Databases

**`Chapter_11/`** | 1 script | 📋 Planned

| Script | Description |
|--------|-------------|
| `26_postgis_integration.py` | Loads all Chapter 8 vector and raster outputs into PostGIS (PostgreSQL spatial extension). Demonstrates spatial SQL: `ST_DWithin` for buffer queries, `ST_Value` for raster sampling, `pg_tileserv` for vector tile serving. |

**Docker:** `docker run -d --name geocascade-postgis -p 5432:5432 postgis/postgis:16-3.4`

---

### Chapter 12 — Capstone

**`Chapter_12/`** | 1 script | 📋 Planned

| Script | Description |
|--------|-------------|
| `capstone_pipeline.py` | Full end-to-end pipeline runner. Executes all chapters in sequence with progress logging, error recovery, and a final 12-panel report dashboard exported as PDF. |

---

### Chapter 13-14 — Advanced Capstone: REST API

**`Chapter_13_14/`** | Planned

| Component | Description |
|-----------|-------------|
| `api.py` | FastAPI async REST API wrapping the full GeoCascade pipeline. Endpoints: `POST /analyze` (accepts GeoJSON bbox + date range), `GET /results/{job_id}`, `GET /health`. GeoJSON response format. Swagger UI at `/docs`. |
| `Dockerfile` | Containerizes the full pipeline for cloud deployment. |

**Run:** `uvicorn api:app --host 0.0.0.0 --port 8000 --reload`

---

## 🛠️ Environment Setup

```bash
# Create environment
mamba create -n geocascade_env python=3.11 -y
conda activate geocascade_env

# Core geospatial stack
mamba install -n geocascade_env -c conda-forge \
    rasterio numpy matplotlib pandas scikit-learn scipy \
    pystac-client planetary-computer pyproj geopandas \
    shapely fiona xarray netcdf4 -y

# Satellite streaming
pip install pysheds

# For Chapter 10 (agentic)
pip install langgraph langchain langchain-google-genai

# For Chapter 13-14 (REST API)
pip install fastapi uvicorn pydantic
```

---

## 🔧 Shared Conventions

| Convention | Value |
|------------|-------|
| nodata value | `-9999` (all GeoTIFFs) |
| CRS | `EPSG:4326` (input/output), projected as needed |
| Compression | `compress="lzw"` (all GeoTIFF writes) |
| Matplotlib backend | `Agg` (headless, no display required) |
| Console encoding | `sys.stdout.reconfigure(encoding="utf-8")` |
| Scale factor | Sentinel-2 L2A: `÷ 10000`, Landsat C2 L2: `× 0.0000275 − 0.2` |
| DEM gradient | Always pass cell size in **metres**, not degrees |

---

## 📊 Complete Output Inventory

| Layer | Source Script | Format | Use in |
|-------|--------------|--------|--------|
| temperature_surface.tif | Ch01/06 | GeoTIFF float32 | Ch04, Ch05 |
| ndvi.tif | Ch02/07 | GeoTIFF float32 | Ch03, Ch04, Ch05 |
| glacier_mask_ndsi.tif | Ch02/07 | GeoTIFF int16 | Ch04 proximity |
| glacier_retreat_2003_2023.tif | Ch03/09 | GeoTIFF float32 | Ch08 |
| slope_degrees.tif | Ch03/10 | GeoTIFF float32 | Ch04, Ch05 |
| aspect_degrees.tif | Ch03/10 | GeoTIFF float32 | Ch06 |
| flow_accumulation.tif | Ch03/11 | GeoTIFF float32 | Ch05, Ch06 |
| ecological_niche_model.tif | Ch04/12 | GeoTIFF float32 | Ch05 zonal |
| vulnerability_index.tif | Ch04/13 | GeoTIFF float32 | Ch08 risk |

---

## 🖥️ Tool Compatibility

| Output Type | ArcGIS Pro | ENVI 5.6 |
|-------------|-----------|----------|
| GeoTIFF float32 (nodata=-9999) | Add as Raster Layer, Stretched symbology | File > Open, Spatial > Resize |
| GeoTIFF int16 / categorical | Unique Values symbology | Density Slice |
| GeoJSON vector | Add as Feature Layer | N/A (use ArcGIS for vectors) |
| CSV (zonal stats) | Join to feature class on station_id | Import via IDL `READ_CSV` |

---

*Last updated: July 2026 | GeoCascade Pipeline v2.0*
