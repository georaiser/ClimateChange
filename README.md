# GeoCascade Environmental Analysis Pipeline

A 14-chapter geospatial Python curriculum for environmental remote sensing and climate
change analysis. All data is streamed cloud-natively from Microsoft Planetary Computer
STAC — no manual downloads required.

**Study area:** Torres del Paine & Grey Glacier, Patagonia, Chile (~51 deg S)

---

## Chapter Navigation

| Chapter | Title | Status | Key Topics |
|---|---|---|---|
| 01 | Climatic Variables & STAC Acquisition | COMPLETE | STAC, ERA5, IsolationForest, Mann-Kendall |
| 02 | Spectral Signature Analysis | COMPLETE | Red Edge B05, NDVI, EVI, SAVI, BSI, NDWI, NDSI |
| 03 | Topography & Glacial Retreat | COMPLETE | NDSI change, pysheds D8, Hipsometric Curve |
| 04 | Ecological Niche Modeling | COMPLETE | Random Forest SDM, Vulnerability Index |
| 05 | Moisture Stress & Zonal Statistics | COMPLETE | NDMI, MSI, rasterstats |
| 06 | Isotherms & Drainage Density | COMPLETE | Lapse Rate, ContourSet, drainage density |
| 07 | SAR Processing & Cloud Penetration | COMPLETE | VV+VH, VV/VH ratio, SAR vs optical |
| 08 | Multi-Sensor Data Fusion | COMPLETE | 4-band cube, Random Forest classifier |
| 09 | Multi-Task Deep Learning | ROADMAP | MTL, ResNet, uncertainty weighting |
| 10 | Agentic Orchestration | ROADMAP | Trigger engine, sub-agents, evidence fusion |
| 11 | PostGIS Integration | ROADMAP | Spatial SQL, FastAPI, raster2pgsql |
| 12 | Capstone CLI Pipeline | COMPLETE | argparse, OSMnx, UTM buffer, zonal report |
| 13 | InSAR Deformation | ADVANCED | Interferometry, SNAP, glacier velocity |
| 14 | Hyperspectral Analysis | ADVANCED | Spectral unmixing, linear mixing model |

---

## Technology Stack

| Category | Libraries |
|---|---|
| Data Access | pystac-client, planetary-computer |
| Raster Processing | rasterio, GDAL, numpy |
| Hydrology | pysheds |
| Vector / GIS | geopandas, shapely, rasterstats, osmnx |
| ML / Stats | scikit-learn, scipy |
| Visualization | matplotlib |
| Climate Data | Open-Meteo API (ERA5 reanalysis, no key needed) |

---

## Installation

```bash
# Create the environment from the provided YAML
mamba env create -f environment.yml
conda activate geocascade_env

# Or install core packages manually
mamba install -n geocascade_env -c conda-forge \
  pystac-client planetary-computer rasterio pysheds \
  geopandas rasterstats osmnx scikit-learn scipy \
  matplotlib numpy pyproj requests pandas -y
```

---

## Key Technical Improvements (All Chapters)

### Critical Bug Fixes
| Fix | Impact |
|---|---|
| src.crs captured inside with-block | Prevents AttributeError / stale CRS after file close |
| OSMnx named kwargs (north=, south=, east=, west=) | Prevents silent wrong-location queries |
| EPSG:3857 -> EPSG:32719 for buffer | Eliminates ~40% distance distortion at 51 deg S |
| nodata=-9999 throughout | GDAL/ArcGIS/ENVI reliable NoData recognition |
| window int(round()) | Prevents rasterio profile dimension type errors |
| min_max_scale NaN preserved | Normalization no longer treats ocean as minimum value |
| B11 independent window | SWIR1 reads correct geographic area (not the B08 10m window) |
| green_2023 src.read() added | Glacier retreat script no longer crashes on missing read |

### Tier 3 Functional Improvements
| Improvement | Chapter |
|---|---|
| Red Edge B05 (705nm) added to spectral analysis | Ch02 |
| Open Water sample point added | Ch02 |
| Spectral signatures exported to CSV | Ch02 |
| Quantitative km2 area change report (glacier) | Ch03 |
| Expanded zonal stats: mean + std + min + max | Ch05 |
| VH polarization added to SAR processing | Ch07 |
| VV/VH cross-polarization ratio raster | Ch07 |
| SAR quantitative area report (water/glacier) | Ch07 |
| Fusion statistics summary table | Ch08 |
| MODIS STAC guard with NoData fallback | Ch08 |
| Annual precipitation CSV export | Ch01 |

### Tier 4 New Scripts
| Script | What It Does |
|---|---|
| 03b_era5_trend_analysis.py | Mann-Kendall test + Sen's Slope on 30-year ERA5 data |
| 19b_cloud_penetration_comparison.py | SAR vs optical cloud comparison (winter scene) |

---

## Spatial Configuration

- **Study Area:** Torres del Paine National Park, Patagonia, Chile
- **Center:** ~51 deg S, ~73 deg W
- **Coordinate Reference Systems:**
  - Geographic: EPSG:4326 (WGS84) — for STAC queries
  - Projected: EPSG:32719 (UTM Zone 19S) — for distance/area/buffer
  - Web: EPSG:3857 — display only, NOT for analysis

---

## Academic Use

This curriculum is designed for graduate-level remote sensing and GIS courses.
All data sources are open-access (Planetary Computer, Open-Meteo, OpenStreetMap).
No API keys required for any script.

**Citation:** When using this curriculum in academic work, please cite the data sources:
- Sentinel-2/1: ESA Copernicus Programme
- Landsat C2-L2: USGS Earth Resources Observation and Science Center
- CopDEM: Copernicus DEM GLO-30 (Airbus Defence & Space)
- ERA5: ECMWF Reanalysis v5 via Open-Meteo API