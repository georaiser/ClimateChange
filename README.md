# GeoCascade Environmental Analysis Pipeline

A 13-chapter geospatial Python curriculum for environmental remote sensing and
climate change analysis. All satellite data is streamed from Microsoft Planetary
Computer STAC. Real-world climate data (ERA5, CHIRPS, GHCN, RGI) is downloaded
from public APIs — no API keys required for the core pipeline.

**Study area:** Torres del Paine & Grey Glacier, Patagonia, Chile (~51°S)

See [PIPELINE_OVERVIEW.md](PIPELINE_OVERVIEW.md) for the full script catalog and run order.

---

## Chapter Navigation

| Ch | Title | Scripts | Status | Key Topics |
|----|-------|---------|--------|------------|
| 01 | Climate Data Acquisition | 9 | ✅ Complete | STAC, ERA5/Open-Meteo, CHIRPS, Mann-Kendall, IsolationForest |
| 02 | Spectral Signature Analysis | 3 | ✅ Complete | Red Edge B05, NDVI, EVI, SAVI, BSI, NDWI, NDSI, NDRE |
| 03 | Topography & Glacial Retreat | 3 | ✅ Complete | NDSI 20yr change, pysheds D8, Hipsometric Curve |
| 04 | Ecological Niche Modeling | 2 | ✅ Complete | Random Forest SDM, MCDA Vulnerability Index |
| 05 | Moisture Stress & Zonal Stats | 2 | ✅ Complete | NDMI, MSI, rasterstats, B11 independent window |
| 06 | Isotherms & Drainage Density | 2 | ✅ Complete | Lapse Rate, ContourSet open paths, drainage density Dd |
| 07 | SAR Processing | 3 | ✅ Complete | VV+VH dual-pol, VV/VH ratio, SAR vs optical cloud comparison |
| 08 | Multi-Sensor Data Fusion | 4 | ✅ Complete | 4-band cube, RF classifier, glacier probability map, ESI convergence |
| 09 | Deep Learning Land Cover | 1 | ✅ Complete | CNN 3-layer ConvNet, PyTorch, patch classification, ONNX export |
| 10 | Agentic Environmental Monitor | 1 | ✅ Complete | TriggerEngine, 6 triggers, convergent evidence monitoring |
| 11 | PostGIS Integration | 1 | ✅ Complete | Spatial SQL, FastAPI stub, SpatiaLite fallback, Docker |
| 12 | Capstone CLI Pipeline | 1 | ✅ Complete | argparse CLI, cache chain, OSMnx UTM buffer, GeoPackage, dark dashboard |
| 13 | Advanced — InSAR & Hyperspectral | 2 | ✅ Complete | SAR offset tracking, velocity maps, spectral unmixing (FCLS) |

---

## Real-World Data Sources (No Login Required)

| Source | Data | API |
|--------|------|-----|
| Open-Meteo | ERA5-Land daily 1993–2024 (11 variables) | archive-api.open-meteo.com |
| UCSB CHIRPS | Precipitation 5.5 km monthly 1981–present | data.chc.ucsb.edu/products/CHIRPS-2.0 |
| Planetary Computer | Sentinel-2/1, DEM, Landsat, MODIS | planetarycomputer.microsoft.com |
| GLIMS | RGI 7.0 glacier outlines | glims.org/RGI/ |

## Optional Free Registration

| Service | What You Gain |
|---------|---------------|
| NOAA CDO — ncei.noaa.gov/cdo-web/token | Real GHCN ground station CSVs |
| NASA EarthData — urs.earthdata.nasa.gov | GPM precipitation, MODIS bulk |
| Copernicus CDS — cds.climate.copernicus.eu | ERA5 full bulk NetCDF |
| USGS EarthExplorer | Landsat full scenes 1982–present |
| Chile DMC | Official Chilean weather station data |

---

## Installation

```bash
mamba create -n geocascade_env python=3.11 -y
conda activate geocascade_env

# Core stack (Chapters 1-12)
mamba install -n geocascade_env -c conda-forge \
    pystac-client planetary-computer rasterio pysheds \
    geopandas rasterstats osmnx scikit-learn scipy \
    matplotlib numpy pyproj requests pandas -y

# Deep Learning (Chapter 9) — CPU
mamba install -n geocascade_env -c conda-forge pytorch torchvision cpuonly -y

# Deep Learning (Chapter 9) — GPU
mamba install -n geocascade_env -c pytorch -c nvidia pytorch torchvision pytorch-cuda=12.4 -y

# PostGIS (Chapter 11, optional — SpatiaLite fallback always works)
mamba install -n geocascade_env -c conda-forge psycopg2 sqlalchemy geoalchemy2 fastapi uvicorn -y

# InSAR & Hyperspectral (Chapter 13)
mamba install -n geocascade_env -c conda-forge scipy -y
```

---

## Run Order

```bash
conda activate geocascade_env

# Chapter 1 — acquire all real-world data first
python Chapter_01/00_real_data_downloader.py     # ERA5, CHIRPS, RGI, GHCN
python Chapter_01/01_stac_multisensor_download.py

# Chapters 2–6 — spectral, terrain, ecological analysis
python Chapter_02/06_spectral_signature_analysis.py
python Chapter_03/10_digital_elevation_processing.py
python Chapter_03/11_watershed_delineation.py
python Chapter_04/12_ecological_niche_modeling.py

# Chapter 7 — SAR (required before Chapter 8 fusion)
python Chapter_07/18_sentinel1_sar_processing.py
python Chapter_07/19_multisensor_review.py
python Chapter_07/19b_cloud_penetration_comparison.py

# Chapter 8 — data fusion (requires Ch02, Ch03, Ch07 cache)
python Chapter_08/20_multisensor_data_fusion.py    # builds cascade_master_stack.tif
python Chapter_08/21_cascade_risk_modeling.py      # RF classification
python Chapter_08/22_combined_insights_engine.py
python Chapter_08/23_real_data_convergence.py

# Chapter 9 — CNN (requires Chapter 8 cube)
python Chapter_09/24_deep_learning_landcover.py

# Chapter 10 — Monitoring
python Chapter_10/25_agentic_monitor.py

# Chapter 11 — PostGIS (works without PostgreSQL via SpatiaLite fallback)
python Chapter_11/26_postgis_integration.py

# Chapter 12 — Capstone CLI
python Chapter_12/capstone_pipeline.py --bbox -73.30 -51.10 -72.90 -50.80

# Chapter 13 — Advanced
python Chapter_13/27_insar_glacier_velocity.py
python Chapter_13/28_hyperspectral_unmixing.py
```

---

## Technology Stack

| Category | Libraries |
|----------|-----------|
| Data Access | pystac-client, planetary-computer, requests |
| Raster Processing | rasterio, GDAL, numpy |
| Hydrology | pysheds |
| Vector / GIS | geopandas, shapely, rasterstats, osmnx |
| ML / Stats | scikit-learn, scipy |
| Deep Learning | PyTorch (CPU or CUDA) |
| Database | PostGIS, SQLAlchemy, GeoAlchemy2 (Ch11) |
| Web API | FastAPI, uvicorn (Ch11 stub) |
| Visualization | matplotlib (Agg backend, dark-mode figures) |
| Climate Data | Open-Meteo API (ERA5 reanalysis, no key) |

---

## Shared Conventions (All Scripts)

| Convention | Value |
|------------|-------|
| NoData value | `-9999` (all GeoTIFFs) |
| Compression | `compress="lzw"` (all GeoTIFF writes) |
| Matplotlib backend | `Agg` (headless, no display required) |
| Console encoding | `sys.stdout.reconfigure(encoding="utf-8")` |
| CRS for metric ops | `EPSG:32719` (UTM Zone 19S — not EPSG:3857) |
| Sentinel-2 L2A scale | `÷ 10000` → reflectance [0–1] |
| Landsat C2-L2 scale | `× 0.0000275 − 0.2` (fill value = 0, mask first) |
| MODIS LST scale | `× 0.02 → K → −273.15 → °C` (fill value = 0) |
| DEM gradient | Always pass cell size in **metres**, not degrees |

---

## Key Bug Fixes Applied (All Chapters)

| Fix | Chapter | Impact |
|-----|---------|--------|
| `src.crs` captured inside `with` block | Ch06 | Prevents stale CRS after file close |
| OSMnx named kwargs `north=, south=, east=, west=` | Ch12 | Prevents silent wrong-location road queries |
| EPSG:3857 → EPSG:32719 for buffers | Ch12 | Eliminates ~40% distance distortion at 51°S |
| `nodata=-9999` throughout (never `np.nan`) | All | ArcGIS/ENVI reliable NoData recognition |
| `int(round(window.height))` | Ch02–Ch08 | Prevents rasterio float dimension crash |
| B11 independent window (20m ≠ 10m grid) | Ch05 | SWIR1 reads correct geographic area |
| `plt.close()` after every `savefig()` | All | Prevents memory leaks in batch processing |
| MODIS fill DN==0 (not `< 7500`) | Ch07–Ch08 | Correct LST masking; no loss of cold pixels |
| Landsat C2-L2 scale factor applied | Ch03–Ch07 | Correct NDSI/NDVI values (not raw DN) |
| `to_polygons(closed_only=False)` | Ch06 | ContourSet open polylines preserved as-is |
| GeoPackage replaces Shapefile for vector output | Ch12 | Handles list columns; no 10-char name limit |

---

## New Scripts Added

| Script | Chapter | Purpose |
|--------|---------|---------|
| `03b_era5_trend_analysis.py` | Ch01 | Mann-Kendall + Sen's Slope on ERA5 temperature |
| `03c_chirps_spatial_precipitation.py` | Ch01 | CHIRPS climatology + Andes rain shadow transect |
| `00_real_data_downloader.py` | Ch01 | Master multi-source downloader (ERA5, CHIRPS, RGI) |
| `19b_cloud_penetration_comparison.py` | Ch07 | SAR vs cloudy optical — all-weather advantage demo |
| `24_deep_learning_landcover.py` | Ch09 | CNN 3-layer ConvNet patch classification (PyTorch) |
| `25_agentic_monitor.py` | Ch10 | TriggerEngine: 6-trigger convergent monitoring |
| `26_postgis_integration.py` | Ch11 | Spatial SQL + FastAPI stub (SpatiaLite fallback) |

---

## Spatial Configuration

- **Study Area:** Torres del Paine National Park, Patagonia, Chile
- **BBOX:** `[-73.30, -51.10, -72.90, -50.80]` (WGS84)
- **CRS for metric analysis:** EPSG:32719 (UTM Zone 19S)
- **CRS for STAC queries:** EPSG:4326 (WGS84 geographic)
- **Sentinel-1 orbital repeat:** 12 days (use for SAR temporal pairs)

---

## GIS Tool Integration

| Output | ArcGIS Pro | ENVI 5.6 |
|--------|-----------|----------|
| `*.tif` float32, nodata=-9999 | Add Raster Layer → Stretched symbology | File > Open → Spatial > Resize |
| `*.tif` uint8 categorical | Unique Values symbology | Density Slice |
| `*.gpkg` vector (GeoPackage) | Add > GPKG layer | N/A — export to SHP first |
| `*.csv` zonal stats | Join to feature class on `zone_id` | IDL `READ_CSV` |
| `*.json` trigger alerts | Import JSON → Table | N/A |

---

## Academic Use

Curriculum designed for graduate-level remote sensing and GIS courses.
All data sources are open-access. No API keys required for the core pipeline.

**Data Citations:**
- Sentinel-2/1: ESA Copernicus Programme
- Landsat C2-L2: USGS Earth Resources Observation and Science Center
- CopDEM: Copernicus DEM GLO-30 (Airbus Defence & Space)
- ERA5: ECMWF Reanalysis v5 via Open-Meteo API (open-meteo.com)
- CHIRPS: Funk et al. 2015, UCSB Climate Hazards Center
- RGI 7.0: Randolph Glacier Inventory Consortium, 2023
- OpenStreetMap: © OpenStreetMap contributors (ODbL)

---

*Last updated: July 2026 | GeoCascade Pipeline v3.0 — All 13 chapters complete*
