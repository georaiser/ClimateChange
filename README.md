# GeoCascade Environmental Analysis Pipeline

A 14-chapter geospatial Python curriculum for environmental remote sensing and
climate change analysis. All satellite data is streamed from Microsoft Planetary
Computer STAC. Real-world climate data (ERA5, CHIRPS, GHCN, RGI) is downloaded
from public APIs -- no API keys required for the core pipeline.

**Study area:** Torres del Paine & Grey Glacier, Patagonia, Chile (~51 deg S)

See [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md) for the recommended run order.

---

## Chapter Navigation

| Chapter | Title | Status | Key Topics |
|---|---|---|---|
| 01 | Climate Data Acquisition | COMPLETE | STAC, ERA5/Open-Meteo, CHIRPS, Mann-Kendall, IsolationForest |
| 02 | Spectral Signature Analysis | COMPLETE | Red Edge B05, NDVI, EVI, SAVI, BSI, NDWI, NDSI |
| 03 | Topography & Glacial Retreat | COMPLETE | NDSI change, pysheds D8, Hipsometric Curve |
| 04 | Ecological Niche Modeling | COMPLETE | Random Forest SDM, MCDA Vulnerability Index |
| 05 | Moisture Stress & Zonal Stats | COMPLETE | NDMI, MSI, rasterstats |
| 06 | Isotherms & Drainage Density | COMPLETE | Lapse Rate, ContourSet, drainage density Dd |
| 07 | SAR Processing | COMPLETE | VV+VH, VV/VH ratio, SAR vs optical cloud comp |
| 08 | Multi-Sensor Data Fusion | COMPLETE | 4-band cube, RF classifier, Real-data ESI convergence |
| 09 | Deep Learning Land Cover | COMPLETE | CNN 3-layer ConvNet, PyTorch, patch classification |
| 10 | Agentic Environmental Monitor | COMPLETE | TriggerEngine, 6 triggers, monitoring dashboard |
| 11 | PostGIS Integration | ROADMAP | Spatial SQL, FastAPI, raster2pgsql |
| 12 | Capstone CLI Pipeline | COMPLETE | argparse, OSMnx UTM buffer, zonal report |
| 13 | InSAR Deformation | ADVANCED | Interferometry, SNAP, glacier velocity |
| 14 | Hyperspectral Analysis | ADVANCED | Spectral unmixing, linear mixing model |

---

## Real-World Data Sources (No Login Required)

| Source | Data | API |
|---|---|---|
| Open-Meteo | ERA5-Land daily 1993-2024 (11 variables) | archive-api.open-meteo.com |
| UCSB CHIRPS | Precipitation 5.5km monthly 1981-present | data.chc.ucsb.edu/products/CHIRPS-2.0 |
| Planetary Computer | Sentinel-2/1, DEM, Landsat, MODIS | planetarycomputer.microsoft.com |
| GLIMS | RGI 7.0 glacier outlines | glims.org/RGI/ |

## Optional Free Registration

| Service | What You Gain |
|---|---|
| NOAA CDO -- ncei.noaa.gov/cdo-web/token | Real GHCN ground station CSVs |
| NASA EarthData -- urs.earthdata.nasa.gov | GPM precipitation, MODIS bulk |
| Copernicus CDS -- cds.climate.copernicus.eu | ERA5 full bulk NetCDF |
| USGS EarthExplorer | Landsat full scenes 1982-present |
| Chile DMC | Official Chilean weather station data |

---

## Real-Data Scripts (Chapter 01)

| Script | Purpose |
|---|---|
| 00_real_data_downloader.py | Master downloader: ERA5, CHIRPS, GHCN, RGI in one command |
| 03a_fetch_real_weather_data.py | 7-station network across Patagonian climate gradient |
| 03b_era5_trend_analysis.py | Mann-Kendall + Sens Slope on 30-year ERA5 temperature |
| 03c_chirps_spatial_precipitation.py | CHIRPS climatology, anomalies, Andes rain shadow transect |

---

## Technology Stack

| Category | Libraries |
|---|---|
| Data Access | pystac-client, planetary-computer, requests |
| Raster Processing | rasterio, GDAL, numpy |
| Hydrology | pysheds |
| Vector / GIS | geopandas, shapely, rasterstats, osmnx |
| ML / Stats | scikit-learn, scipy |
| Deep Learning | PyTorch (CPU or CUDA) |
| Visualization | matplotlib |
| Climate Data | Open-Meteo API (ERA5 reanalysis, no key) |

---

## Installation

`bash
mamba create -n geocascade_env python=3.11 -y
conda activate geocascade_env
mamba install -n geocascade_env -c conda-forge pystac-client planetary-computer rasterio pysheds geopandas rasterstats osmnx scikit-learn scipy matplotlib numpy pyproj requests pandas pytorch torchvision cpuonly -y
`

---

## Key Technical Improvements (All Chapters)

### Critical Bug Fixes Applied
| Fix | Impact |
|---|---|
| src.crs captured inside with-block | Prevents stale CRS after file close (Ch06) |
| OSMnx named kwargs north=, south=, east=, west= | Prevents silent wrong-location queries (Ch12) |
| EPSG:3857 to EPSG:32719 for buffer | Eliminates ~40% distance distortion at 51 deg S |
| nodata=-9999 throughout | ArcGIS/ENVI reliable NoData recognition |
| int(round()) window dimensions | Prevents rasterio float dimension errors |
| B11 independent window (20m vs 10m) | SWIR1 reads correct geographic area |
| plt.close() after every savefig() | Prevents memory leaks in batch processing |
| MODIS fill DN<7500 (not 0) | Correct LST masking and temperature values |
| Landsat C2-L2 scale factor | DN * 0.0000275 - 0.2 (not raw DN) |

### New Scripts Added
| Script | What It Does |
|---|---|
| 03b_era5_trend_analysis.py | Mann-Kendall test + Sens Slope on ERA5 temperature |
| 03c_chirps_spatial_precipitation.py | CHIRPS spatial climatology + rain shadow analysis |
| 00_real_data_downloader.py | Master multi-source downloader (ERA5, CHIRPS, RGI) |
| 19b_cloud_penetration_comparison.py | SAR vs optical through cloud cover |
| 23_real_data_convergence.py | ESI dashboard integrating all real + satellite data |
| 24_deep_learning_landcover.py | CNN land cover classification (PyTorch, 3-class) |
| 25_agentic_monitor.py | Automated trigger-engine environmental alert system |

---

## Spatial Configuration

- **Study Area:** Torres del Paine National Park, Patagonia, Chile
- **BBOX:** [-73.30, -51.10, -72.90, -50.80]
- **CRS for analysis:** EPSG:32719 (UTM Zone 19S) -- distances in metres
- **CRS for STAC queries:** EPSG:4326 (WGS84 geographic)

---

## Academic Use

Curriculum designed for graduate-level remote sensing and GIS courses.
All data sources are open-access. No API keys required for core pipeline.

**Data Citations:**
- Sentinel-2/1: ESA Copernicus Programme
- Landsat C2-L2: USGS Earth Resources Observation and Science Center  
- CopDEM: Copernicus DEM GLO-30 (Airbus Defence & Space)
- ERA5: ECMWF Reanalysis v5 via Open-Meteo API (open-meteo.com)
- CHIRPS: Funk et al. 2015, UCSB Climate Hazards Center
- RGI 7.0: Randolph Glacier Inventory Consortium, 2023
