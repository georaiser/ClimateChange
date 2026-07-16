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
| [Chapter 5](#chapter-5--moisture-zonal-statistics) | Moisture & Zonal Statistics | ✅ Complete |
| [Chapter 6](#chapter-6--isohyets-drainage-density) | Isohyets & Drainage Density | ✅ Complete |
| [Chapter 7](#chapter-7--sar--multisensor-analysis) | SAR & Multisensor Analysis | ✅ Complete |
| [Chapter 8](#chapter-8--data-fusion--cascade-risk) | Data Fusion & Cascade Risk Modeling | ✅ Complete |
| [Chapter 9](#chapter-9--deep-learning-land-cover) | Deep Learning Land Cover (PyTorch) | ✅ Complete |
| [Chapter 10](#chapter-10--agentic-orchestration) | Agentic Environmental Monitoring | ✅ Complete |
| [Chapter 11](#chapter-11--enterprise-spatial-databases) | Enterprise Spatial Databases (PostGIS) | ✅ Complete |
| [Chapter 12](#chapter-12--capstone) | Capstone — Full Pipeline Integration | ✅ Complete |
| [Chapter 13](#chapter-13--advanced-techniques) | Advanced — InSAR Velocity & Hyperspectral Unmixing | ✅ Complete |

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

**`Chapter_05/`** | 2 scripts | All updated July 2026

| Script | Description | Key Outputs |
|--------|-------------|-------------|
| `14_moisture_stress_indices.py` | Computes NDMI and MSI from Sentinel-2 NIR (B08) and SWIR1 (B11). **Critical fix:** B11 uses its own independently computed window and transformer — it cannot reuse the NIR window because B11 is natively 20m with a different transform. Higher NDMI = more moisture; higher MSI = more drought stress. | `moisture_stress/ndmi.tif`, `moisture_stress/msi.tif`, `moisture_stress/moisture_comparison.png`, `moisture_statistics.csv` |
| `15_zonal_statistics.py` | Generates 4 vector polygons (NW/NE/SW/SE quadrants) and runs `rasterstats.zonal_stats` with `nodata=-9999`. Computes mean, std, min, max per zone for NDVI, DEM, and slope. None-safe formatter guards zones fully covered by NoData. | `zonal_statistics/quadrant_zones.gpkg`, `zonal_summary.csv`, `zonal_statistics_panel.png` |

---

### Chapter 6 — Isohyets & Drainage Density

**`Chapter_06/`** | 2 scripts | All updated July 2026

| Script | Description | Key Outputs |
|--------|-------------|-------------|
| `16_isohyets_isotherms.py` | Generates temperature isolines using the Environmental Lapse Rate (−6.5°C/1000m). **Critical fix 1:** `src.crs` captured as `raster_crs` inside the `with` block — accessing it outside causes silent errors. **Critical fix 2:** `to_polygons(closed_only=False)` — isotherms are open LineStrings, not closed polygons. Exports isolines as GeoJSON. | `isotherms/isotherms.geojson`, `isotherms/lapse_rate_map.tif`, `isotherm_dashboard.png` |
| `17_drainage_density.py` | Downloads DEM, runs full pysheds D8 workflow, extracts river network as GeoDataFrame, calculates drainage density `Dd = L_total / A_basin` (km/km²). Higher Dd = flashier catchment with rapid storm response. | `drainage/river_network.gpkg`, `drainage_density.tif`, `drainage_density_report.csv`, `drainage_dashboard.png` |

---

### Chapter 7 — SAR & Multisensor Analysis

**`Chapter_07/`** | 3 scripts | All updated July 2026

| Script | Description | Key Outputs |
|--------|-------------|-------------|
| `18_sentinel1_sar_processing.py` | Downloads Sentinel-1 RTC VV and VH polarizations. Converts to dB backscatter (`σ₀_dB = 10 × log₁₀(σ₀_linear)`). Computes cross-polarization ratio (VH/VV). Applies dual thresholds: VV < −18 dB = water; VV > −5 dB = glacier/rough ice. 5-panel dark figure. | `sar/sar_vv_db.tif`, `sar_vh_db.tif`, `sar_cr.tif`, `water_mask_sar.tif`, `glacier_mask_sar.tif`, `sar_statistics.csv` |
| `19_multisensor_review.py` | Side-by-side comparison of Landsat 9 NIR, MODIS LST (°C), and Sentinel-1 VV for the same BBOX. **Critical fix:** MODIS fill = DN==0 (not DN < 7500). Landsat C2-L2 scale factor applied. 4-panel dark figure with sensor matrix table. | `multisensor/landsat9_nir.tif`, `modis_lst_celsius.tif`, `sentinel1_vv_db.tif`, `multisensor_statistics.csv` |
| `19b_cloud_penetration_comparison.py` | Deliberately selects the cloudiest available winter Sentinel-2 scene and the nearest Sentinel-1 acquisition. Proves SAR's all-weather advantage: ice structure visible through 100% cloud cover. | `cloud_comparison/sar_vv_db.tif`, `cloud_comparison_stats.csv`, `sar_vs_optical_cloudy.png` |

---

### Chapter 8 — Data Fusion & Cascade Risk Modeling

**`Chapter_08/`** | 4 scripts | All updated July 2026

| Script | Description | Key Outputs |
|--------|-------------|-------------|
| `20_multisensor_data_fusion.py` | Fuses 4 sensors (S2 NIR, S1 SAR VV, CopDEM, MODIS LST) onto a single 10m master grid via `rasterio.warp.reproject`. Uses local Ch02/Ch03/Ch07 cache before STAC download. Writes a self-documenting 4-band GeoTIFF with band descriptions. | `fusion/cascade_master_stack.tif`, `fusion_statistics.csv`, `data_cube_visualization.png` |
| `21_cascade_risk_modeling.py` | Random Forest (150 trees) trained on 4-band cube using physics-based percentile labels. Outputs 4-class land cover map + continuous glacier probability map. 4-panel dark figure with donut chart and feature importance bars. | `ml/cascade_ml_prediction.tif`, `glacier_probability_map.tif`, `feature_importance.csv`, `classification_report.csv` |
| `22_combined_insights_engine.py` | Convergent Evidence Analysis computing ESI (Ecological Stress Index), CVS (Cryosphere Vulnerability Score), and HECI (Human-Environment Conflict Index). | `combined_insights/esi_ecological_stress.tif`, `cvs_cryosphere_vulnerability.tif`, `heci_human_conflict.tif` |
| `23_real_data_convergence.py` | Full multi-source dashboard integrating ERA5, CHIRPS, 7-station network, Sentinel-2 NDVI, Sentinel-1 SAR, CopDEM, and RGI 7.0 glacier outlines. | `convergence/environmental_stress_composite.tif`, `convergence_dashboard.png`, `convergence_report.csv` |

---

### Chapter 9 — Deep Learning Land Cover

**`Chapter_09/`** | 1 script | ✅ Complete (July 2026)

| Script | Description | Key Outputs |
|--------|-------------|-------------|
| `24_deep_learning_landcover.py` | 3-layer ConvNet trained on 32×32 multi-sensor patches from the Ch08 data cube. Reads `fusion/cascade_master_stack.tif` (fallback: legacy path). Labels generated dynamically from geophysical percentiles (same physics as Ch08 RF). Full sliding-window inference produces spatially-aware classification with confidence map. Exports training history and per-class metrics CSVs. | `dl/cnn_landcover_prediction.tif`, `cnn_confidence_map.tif`, `cnn_training_history.csv`, `cnn_class_metrics.csv`, `cnn_landcover_results.png` |

**Architecture:** Conv(4→32)+BN+ReLU+MaxPool → Conv(32→64)+BN+ReLU+MaxPool → Conv(64→128)+BN+ReLU+MaxPool → FC(2048→256, Dropout 0.3) → FC(256→3)

**Install:** `mamba install -n geocascade_env -c conda-forge pytorch torchvision cpuonly -y`

---

### Chapter 10 — Agentic Environmental Monitoring

**`Chapter_10/`** | 1 script | ✅ Complete (July 2026)

| Script | Description | Key Outputs |
|--------|-------------|-------------|
| `25_agentic_monitor.py` | Rule-based TriggerEngine (no LLM required) that independently evaluates 6 geophysical triggers: temperature anomaly (ERA5 > mean+2σ), drought stress (NDVI < 0.2 for 3+ scenes), precipitation deficit (30-day < 20th pct), SAR glacier change (Δ > 3 dB), wind extreme (> 95th pct), snow melt anomaly. Raises CONVERGENT ALERT when ≥ 3 triggers fire simultaneously. | `monitor_alerts.json`, `trigger_status.csv`, `trigger_dashboard.png` |

**Install:** `mamba install -n geocascade_env -c conda-forge numpy pandas matplotlib -y`

---

### Chapter 11 — Enterprise Spatial Databases

**`Chapter_11/`** | 1 script | ✅ Complete (July 2026)

| Script | Description | Key Outputs |
|--------|-------------|-------------|
| `26_postgis_integration.py` | Loads Ch04 CVI raster, Ch05 zonal stats GeoDataFrame, and Ch01 RGI glacier outlines into PostGIS (or SpatiaLite fallback — always works). Demonstrates 3 spatial SQL queries: vulnerability zone filter, glacier 1km buffer intersection (ST_DWithin with `::geography` cast), and mean CVI per watershed (ST_Within join). Generates FastAPI REST stub. | `vulnerability_zones.geojson`, `glacier_buffer_intersections.geojson`, `watershed_aggregates.csv`, `postgis_integration_dashboard.png` |

**Docker:** `docker run -d --name geocascade-postgis -e POSTGRES_DB=geocascade -e POSTGRES_USER=geo -e POSTGRES_PASSWORD=cascade2024 -p 5432:5432 postgis/postgis:16-3.4`

> Script runs without PostgreSQL — SpatiaLite fallback is auto-detected.

---

### Chapter 12 — Capstone

**`Chapter_12/`** | 1 script | ✅ Complete (July 2026)

| Script | Description | Key Outputs |
|--------|-------------|-------------|
| `capstone_pipeline.py` | Full argparse CLI pipeline. Accepts `--bbox`, `--date_range`, `--buffer_m`. Resolves rasters from local cache chain (Ch02→Ch03→Ch07) before STAC download. Downloads OSMnx road network (named kwargs only), projects to EPSG:32719 for accurate UTM buffers, saves GeoPackage. Runs `rasterstats.zonal_stats` via MemoryFile. Builds 5-panel dark dashboard and auto-generates Markdown site analysis report. | `capstone/ndvi_capstone.tif`, `sar_vv_db_capstone.tif`, `dem_capstone.tif`, `roads_buffer.gpkg`, `capstone_zonal_stats.csv`, `capstone_dashboard.png`, `site_analysis_report.md` |

**Usage:** `python Chapter_12/capstone_pipeline.py --bbox -73.30 -51.10 -72.90 -50.80`

---

### Chapter 13 — Advanced Techniques

**`Chapter_13/`** | 2 scripts | ✅ Complete (July 2026)

| Script | Description | Key Outputs |
|--------|-------------|-------------|
| `27_insar_glacier_velocity.py` | SAR Intensity Cross-Correlation (Offset Tracking) — the Python-native alternative to full InSAR. Correlates two Sentinel-1 VV intensity images separated in time. Peak of correlation window = displacement vector. Velocity = displacement × pixel_size × (365 / time_delta). Valid for Grey Glacier (300–800 m/year). Falls back to synthetic pair if only one SAR scene available. | `insar/vx_map.tif`, `vy_map.tif`, `vmag_map.tif`, `velocity_statistics.csv`, `insar_velocity_dashboard.png` |
| `28_hyperspectral_unmixing.py` | Linear Spectral Unmixing (FCLS) decomposes each Sentinel-2 pixel into fractional abundances of 4 endmembers: Glacier/Snow, Open Water, Dense Vegetation, Bare Rock. Solved with `scipy.optimize.nnls` + sum-to-one constraint. Reveals sub-pixel heterogeneity invisible to classification. | `hyperspectral/abundance_glacier.tif`, `abundance_water.tif`, `abundance_vegetation.tif`, `abundance_rock.tif`, `endmember_spectra.csv`, `abundance_statistics.csv`, `spectral_unmixing_results.png` |

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

*Last updated: July 2026 | GeoCascade Pipeline v3.0 — All 13 chapters complete*
