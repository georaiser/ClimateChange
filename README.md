# 🌍 GeoCascade: Multi-Sensor Climate & Glacier Analysis Pipeline

> **From raw satellite pixels to actionable environmental insights — a complete academic curriculum in Earth Observation, Geo-AI, and Agentic Geospatial Engineering.**

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![Platform](https://img.shields.io/badge/Data-Planetary%20Computer%20%7C%20STAC-green)](https://planetarycomputer.microsoft.com)
[![License](https://img.shields.io/badge/License-Academic-orange)](LICENSE)

---

## 📖 Project Philosophy

This curriculum answers a single, central question:

> *"What does combining ALL satellite datasets tell us about this landscape that NO SINGLE dataset can reveal alone?"*

We call this approach **Convergent Evidence Analysis** — a professional methodology used in real-world Environmental Impact Assessments (EIAs) and climate science reports. Every chapter builds toward this goal by introducing a new physical layer (optical, radar, thermal, topographic) and teaching how to fuse them.

**Key Principles:**
- **No Google Earth Engine.** All processing uses open cloud-native tools: `rasterio`, `pystac-client`, `planetary-computer`, `numpy`, `geopandas`.
- **Absolute Reproducibility.** Every data source is STAC-queryable. No manual downloads required.
- **ArcGIS/ENVI Interoperability.** Every script exports geocoded `.tif` and `.shp` files ready for drag-and-drop use in industry-standard GIS tools.
- **Real Data.** ERA5 reanalysis, Sentinel-2, Sentinel-1 SAR, MODIS, Copernicus DEM — all real satellite archives.

---

## 🛠️ System Prerequisites & Installation

Because geospatial dependencies (GDAL, PROJ, GEOS) are notoriously difficult to compile from source, this project strictly uses **Miniforge/Mamba**.

### 1. Install Miniforge

Download the [Miniforge installer](https://github.com/conda-forge/miniforge) for your OS. It includes `mamba` — a fast drop-in replacement for `conda`.

### 2. Create the Base Environment

```bash
mamba create -n geocascade_env python=3.12 -y
mamba activate geocascade_env
```

### 3. Install Core Geospatial Stack

```bash
mamba install -n geocascade_env -c conda-forge \
    pystac-client planetary-computer rasterio pyproj \
    geopandas shapely numpy matplotlib scikit-learn \
    pandas requests pysheds rasterstats osmnx -y
```

### 4. PyTorch (Chapter 9 — Deep Learning)

```bash
mamba install -n geocascade_env pytorch torchvision torchaudio \
    pytorch-cuda=12.4 -c pytorch -c nvidia -c conda-forge \
    --channel-priority flexible -y
```

### 5. Docker (Chapter 11 — PostGIS)

Install [Docker Desktop](https://docs.docker.com/desktop/install/windows-install/) for the spatial database chapter.

### 6. Planetary Computer Authentication

```bash
pip install planetary-computer
planetarycomputer login
```

> [!NOTE]
> A free Microsoft Planetary Computer account is required for STAC access. Register at [planetarycomputer.microsoft.com](https://planetarycomputer.microsoft.com).

---

## 📂 Full Curriculum Structure

The project is organized into **3 Phases** covering the complete arc from raw data to advanced AI systems.

---

### 🌱 Phase 1 — Core Physical Sciences (Chapters 1–7)

| Chapter | Title | Key Skills |
|---------|-------|------------|
| `Chapter_01/` | **Climatic Variables & STAC Acquisition** | STAC API, Atmospheric Correction, ML Interpolation, Precipitation Anomaly, UHI Mapping |
| `Chapter_02/` | **Spectral Signature Analysis** | Spectral profiles, NDVI, NDWI, NDSI, NDGI, EVI, SAVI, BSI (7-index suite) |
| `Chapter_03/` | **Topography & Glacial Retreat** | DEM derivatives, 20-year glacier change detection, Watershed Delineation, Hipsometric Curve |
| `Chapter_04/` | **Ecological Niche & Climate Vulnerability** | K-Means unsupervised classification, Random Forest SDM, MCDA Vulnerability Index |
| `Chapter_05/` | **Zonal Statistics & Moisture Stress** | NDMI, MSI, Zonal Statistics by management zone |
| `Chapter_06/` | **Advanced Hydrometeorology** | Isotherms, Drainage Density, Stream Order |
| `Chapter_07/` | **Radar & Multi-Sensor Review (SAR)** | Sentinel-1 SAR processing, VV/VH backscatter, Multi-sensor comparison |

---

### 🤖 Phase 2 — Advanced Modeling & Geo-AI (Chapters 8–11)

| Chapter | Title | Key Skills |
|---------|-------|------------|
| `Chapter_08/` | **Data Fusion & Cascade Effect Modeling** | Multi-sensor data cube, Random Forest land cover, **Convergent Evidence Analysis** (ESI, CVS, WSI) |
| `Chapter_09/` | **Multi-Task Deep Learning** | PyTorch, multi-head CNNs, simultaneous land cover + change detection |
| `Chapter_10/` | **Agentic Orchestration** | LangGraph, autonomous geospatial agents, tool calling |
| `Chapter_11/` | **Enterprise Spatial Databases** | PostGIS, Docker, spatial SQL, vector tile serving |

---

### 🏆 Phase 3 — Capstones (Chapters 12–13)

| Chapter | Title | Key Skills |
|---------|-------|------------|
| `Chapter_12/` | **Automated Site Analysis Capstone** | CLI pipeline, dynamic BBOX, impact zone assessment, zonal stats reporting |
| `Chapter_13_14/` | **Full-Stack REST API Capstone** | FastAPI, GeoJSON endpoints, async processing, cloud deployment |

---

## 🗺️ GIS Interoperability

A core requirement of this curriculum is **Hybrid Dual-Track Interoperability**. Every Python script exports:

- **Geocoded GeoTIFF (`.tif`)** — drag-and-drop into ArcGIS Pro or ENVI
- **Shapefile / GeoPackage (`.shp` / `.gpkg`)** — vector overlays ready for spatial queries

**Workflow:**
1. Run the automated Python pipeline
2. Open **ArcGIS Pro** or **ENVI**
3. Drag-and-drop `.tif` / `.shp` files into the map canvas
4. Verify that Python outputs align perfectly with basemap imagery

---

## 🚀 Quick Start

```bash
# 1. Activate environment
mamba activate geocascade_env

# 2. Start from Chapter 1
cd Chapter_01
python 01_stac_multisensor_download.py

# 3. Follow each chapter's README for the next steps
```

Each chapter folder contains its own `README.md` with:
- 🎯 Academic objective
- 📐 Mathematical concepts
- 🛠️ Script-by-script explanations
- 🚀 Exact run commands
- 🗺️ GIS interoperability notes

---

## 📊 Data Sources

| Dataset | Provider | Access |
|---------|----------|--------|
| Sentinel-2 L2A (10m optical) | ESA / Microsoft | Planetary Computer STAC |
| Sentinel-1 RTC (SAR backscatter) | ESA / Microsoft | Planetary Computer STAC |
| Copernicus DEM GLO-30 (30m elevation) | ESA | Planetary Computer STAC |
| MODIS MOD11A1 (1km thermal LST) | NASA | Planetary Computer STAC |
| Landsat C2-L2 (30m optical) | USGS / Microsoft | Planetary Computer STAC |
| ERA5-Land (hourly reanalysis) | ECMWF | Open-Meteo API (no key needed) |
| OpenStreetMap roads | OSM contributors | osmnx (no key needed) |

---

## 🧪 The Three Composite Insight Scores (Chapter 8)

The curriculum culminates in three composite indices derived from all sensor layers simultaneously:

| Index | Formula | Interpretation |
|-------|---------|----------------|
| **ESI** — Ecological Stress | `0.5×(1-NDVI) + 0.3×LST + 0.2×Slope` | Where is the ecosystem degraded? |
| **CVS** — Cryosphere Vulnerability | `0.4×LST + 0.4×(1-NDSI) + 0.2×(1-SAR)` | Which glacial zones are actively melting? |
| **WSI** — Water Stress Compound | `0.4×NDWI + 0.4×NDSI + 0.2×(1-LST)` | Where is fresh water most available? |

---

*Study Area: Torres del Paine National Park & Grey Glacier, Patagonia, Chile (BBOX: `[-73.30, -51.10, -72.90, -50.80]`)*
