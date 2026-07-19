# Chapter 14 🗺️ — ArcGIS Pro + ENVI: Professional GIS Workflows

> **Curriculum position:** This chapter bridges the open-source Python pipeline (Chapters 1-13)
> with the commercial professional workflow used in government agencies, consulting firms,
> and research institutions across Latin America.

---

## ⚠️ Implementation Approach — What Language Does Each Script Use?

This is the most important thing to understand before running the chapter.

| Interface | Scripts | Language | Requires |
|-----------|---------|----------|---------|
| **ArcGIS Pro GUI** | 01–06 (ArcPy blocks) | Python (ArcPy) | ArcGIS Pro + Spatial Analyst license |
| **ENVI GUI** | Documented in docstrings | GUI (no code) | ENVI 5.6+ license |
| **ENVI Python API** | Commented in each script | Python (envi_py) | ENVI 5.6+ license |
| **ENVI/IDL scripting** | `11_idl_reference.pro` | IDL | ENVI + IDL license |
| **Open-source Python** ✅ | **All 10 scripts (actual code)** | Python (rasterio, sklearn, scipy) | `geocascade_env` only |

> [!IMPORTANT]
> **The 10 `.py` scripts use 100% open-source Python** (rasterio, numpy, scipy, scikit-learn).
> They produce identical outputs to ENVI/ArcGIS Pro and run in `geocascade_env` without any commercial license.
> The ArcPy and ENVI API calls are **embedded as optional blocks** that activate only when a license is detected.
> For native ENVI scripting, see `11_idl_reference.pro` (IDL) below.

### Three Ways to Do the Same Thing

Every ENVI workflow in this chapter is documented in **three layers**:

```
Layer 1 — Open-source Python (runs always)
  → rasterio + numpy + sklearn code in the main script body

Layer 2 — ENVI Python API (activates if ENVI is installed)
  → commented blocks inside each script: "# ENVI Python API equivalent"
  → requires: pip install envi_py  (ENVI 5.6+)

Layer 3 — ENVI GUI / IDL
  → GUI steps documented in each script's docstring
  → IDL code in: Chapter_14/11_idl_reference.pro
```

---

## 🎯 Learning Objectives

By the end of this chapter you will be able to:

- ✅ Import GeoCascade Chapter 1-13 GeoTIFF outputs directly into **ArcGIS Pro** without conversion
- ✅ Automate ArcGIS Pro operations with **ArcPy** scripts (runs when licensed)
- ✅ Build and export a **Model Builder** pipeline as a Python Toolbox (`.pyt`)
- ✅ Perform unsupervised classification with **ISO Cluster** (ArcGIS Pro) and **ISODATA** (ENVI)
- ✅ Apply **atmospheric correction** — DOS1 (open-source) or FLAASH (ENVI licensed)
- ✅ Compute NDVI, NBR, NDWI, NDSI, EVI using **rasterio band math** (mirrors ENVI Band Math)
- ✅ Detect land cover change 2019→2023 using Python thresholding (mirrors **ENVI Change Detection**)
- ✅ Produce publication-quality maps with **ArcGIS Pro Layout** conventions (north arrow, graticule, legend)
- ✅ Read and write basic **IDL scripts** for ENVI automation (`11_idl_reference.pro`)

---

## 🗂️ Chapter Structure

```
Chapter_14/
│
│  ─── OPEN-SOURCE PYTHON (geocascade_env, all run without licenses) ──
├── 01_setup_project.py          # Layer discovery + ArcGIS Pro APRX skeleton
├── 02_raster_analysis.py        # Reclassify, Raster Calculator, Focal Statistics
├── 03_classification.py         # K-Means (ISO Cluster equiv) + GMM (ISODATA equiv)
├── 04_spatial_analysis.py       # Slope, Aspect, Zonal Statistics, Vulnerability Index
├── 05_climate_maps.py           # Layout maps: graticule, north arrow, scale bar
├── 06_model_builder.py          # Python Toolbox (.pyt) + Model Builder diagram
├── 07_envi_spectral_analysis.py # NDVI, NBR, NDWI, NDSI, EVI via rasterio band math
├── 08_envi_atm_correction.py    # DOS1 atmospheric correction + FLAASH guide
├── 09_envi_classification.py    # K-Means + MLC + Majority Filter via sklearn
├── 10_envi_change_detection.py  # NDVI change 2019→2023 via numpy thresholding
│
│  ─── NATIVE COMMERCIAL IMPLEMENTATIONS (require licenses) ─────────
├── 11_idl_reference.pro         # IDL  : ENVI native scripting (Task Engine API)
├── 12_arcpy_full.py             # ArcPy: 100% ArcGIS Pro native implementation
├── 13_envi_api_full.py          # ENVI : 100% ENVI Python API (envi_py) implementation
│
├── arcgis_pro/
│   └── toolboxes/
│       ├── GeoCascade.pyt            # ArcGIS Pro Python Toolbox (import this)
│       └── GeoCascade_model.json     # Model Builder JSON export
│
└── data/
    └── processed/
        ├── arcgis_outputs/           # Reclassified, slope, vulnerability, maps
        └── envi_outputs/             # Spectral indices, classified rasters, change maps
```

---

## 🛠️ Software Requirements

| Software | Version | License | Used For |
|----------|---------|---------|---------|
| **`geocascade_env`** (Python) | 3.11 | Free | All 10 scripts — full functionality |
| **ArcGIS Pro** | 3.x | Esri Commercial | ArcPy blocks in scripts 01–06 |
| **ENVI** | 5.6+ | Harris Geospatial | ENVI API blocks in scripts 07–10 |
| **IDL** | 8.9+ | Harris Geospatial | `11_idl_reference.pro` |

> [!NOTE]
> Scripts detect the available environment and degrade gracefully:
> - No ArcGIS Pro → skips ArcPy blocks, produces same outputs via rasterio
> - No ENVI → skips ENVI API blocks, produces same outputs via sklearn/rasterio
> - No license at all → full open-source pipeline runs with identical GeoTIFF outputs

---

## 🚀 Quick Start

```bash
conda activate geocascade_env

# ArcGIS Pro workflow (open-source equivalent)
python Chapter_14/01_setup_project.py
python Chapter_14/02_raster_analysis.py
python Chapter_14/03_classification.py
python Chapter_14/04_spatial_analysis.py
python Chapter_14/05_climate_maps.py
python Chapter_14/06_model_builder.py

# ENVI workflow (open-source equivalent)
python Chapter_14/07_envi_spectral_analysis.py
python Chapter_14/08_envi_atm_correction.py
python Chapter_14/09_envi_classification.py
python Chapter_14/10_envi_change_detection.py
```

---

## 🗺️ ArcGIS Pro — Three Ways to Run

### Option A: From ArcGIS Pro Python Window (full ArcPy)
```python
# Inside ArcGIS Pro: Analysis > Python Window
exec(open(r"D:\00_AI_Aplications\ClimateChange\Chapter_14\02_raster_analysis.py").read())
```

### Option B: Import the Python Toolbox
```
ArcGIS Pro > Geoprocessing > Toolboxes > Add Toolbox
Browse to: Chapter_14/arcgis_pro/toolboxes/GeoCascade.pyt
```
Four tools appear: Raster Analysis | Classification | Spatial Analysis | Export Maps

### Option C: Standalone (no ArcGIS Pro needed)
```bash
conda activate geocascade_env
python Chapter_14/02_raster_analysis.py
# → produces identical GeoTIFF outputs via rasterio
```

---

## 📡 ENVI — Three Ways to Run

### Option A: Open-source Python (no ENVI needed) ✅
```bash
python Chapter_14/07_envi_spectral_analysis.py
# → ndvi.tif, nbr.tif, ndwi.tif, ndsi.tif, evi.tif via rasterio band math
```

### Option B: ENVI Python API (requires ENVI 5.6+)
```python
# Uncomment the ENVI API blocks inside each script, then run:
# (look for: "# ENVI Python API equivalent" comments)
from envi import ENVI
envi = ENVI()
task = envi.Task('VegetationIndex')
task['INPUT_RASTER'] = envi.open_raster('path/to/sentinel2.tif')
task['INDEX'] = 'Normalized Difference Vegetation Index'
task.execute()
```

### Option C: ENVI GUI
```
File > Open > Chapter_01/.../uhi_celsius.tif
Toolbox > Spectral > Vegetation Index Calculator
Toolbox > Radiometric Correction > FLAASH
Classification > Unsupervised > ISODATA (k=5)
```

### Option D: IDL Script
```idl
; Run from ENVI IDL Console or IDL Workbench:
.compile Chapter_14/11_idl_reference.pro
geocascade_spectral_analysis
```

---

## 📜 IDL Reference Script (`11_idl_reference.pro`)

ENVI's native scripting language is **IDL** (Interactive Data Language). The `11_idl_reference.pro`
script covers the same workflows as scripts 07–10 in native IDL syntax:

| IDL Procedure | Python Equivalent | Operation |
|--------------|------------------|-----------|
| `geocascade_spectral_analysis` | `07_envi_spectral_analysis.py` | NDVI, NBR, NDWI |
| `geocascade_atm_correction` | `08_envi_atm_correction.py` | DOS1 / FLAASH |
| `geocascade_classify` | `09_envi_classification.py` | ISODATA + MLC |
| `geocascade_change_detection` | `10_envi_change_detection.py` | NDVI change map |

---

## 📊 Curriculum Alignment (Course: Teledetección + ArcGIS Pro)

| Course Module | Script(s) | Tool Used |
|--------------|-----------|-----------|
| Módulo 2: Band Correction & Composition | `08_envi_atm_correction.py` | DOS1 (Python) / FLAASH (ENVI) |
| Módulo 3: Unsupervised Classification | `03`, `09` | K-Means (Python) / ISODATA (ENVI) |
| Módulo 3: Precipitation Management | `02_raster_analysis.py` | Reclassify (Python/ArcPy) |
| Módulo 4: Heat Island Mapping | `02`, `05` | Raster Calc (Python/ArcPy) |
| Módulo 5: Raster Temperature Maps | `04`, `05` | Weighted Overlay (Python/ArcPy) |
| Módulo 6-7: Vulnerability Analysis | `04_spatial_analysis.py` | Weighted Overlay (Python/ArcPy) |

---

## 📤 Outputs Summary

| Script | Output Files | Format |
|--------|-------------|--------|
| 01 | `layer_manifest.json`, `GeoCascade_Ch14.aprx` | JSON / APRX |
| 02 | `precip_classified.tif`, `temp_anomaly.tif`, `uhi_focal.tif` | GeoTIFF |
| 03 | `classified_land_cover.tif`, `gmm_land_cover.tif` | GeoTIFF |
| 04 | `slope.tif`, `aspect.tif`, `climate_vulnerability.tif` | GeoTIFF |
| 05 | `climate_atlas.png`, `uhi_final_map.png`, `vulnerability_map.png` | PNG |
| 06 | `GeoCascade.pyt`, `GeoCascade_model.json`, `model_builder_diagram.png` | Python/JSON/PNG |
| 07 | `ndvi.tif`, `nbr.tif`, `ndwi.tif`, `ndsi.tif`, `evi.tif` | GeoTIFF |
| 08 | `sentinel2_dos1_corrected.tif`, `atm_correction_report.png` | GeoTIFF / PNG |
| 09 | `classified_land_cover.tif`, `mlc_land_cover.tif`, `classified_majority.tif` | GeoTIFF |
| 10 | `ndvi_2019.tif`, `ndvi_2023.tif`, `ndvi_change_2019_2023.tif` | GeoTIFF |
| 11 | *(IDL script — no output files, runs inside ENVI)* | `.pro` |

> **ArcGIS Pro:** All `.tif` outputs load directly — `Add Data` → apply symbology.
> **ENVI 5.6:** `File > Open` any `.tif`. Multi-band files open as spectral stacks.

---

## 🔗 Data Chain

```
Chapter 01 outputs  ──► Scripts 01, 02, 04, 05  (ArcGIS Pro pipeline)
Script 07 (NDVI/NBR/NDWI)  ──► Scripts 09, 10    (ENVI classification/change)
Script 08 (DOS1 corrected)  ──► Script 09         (corrected imagery → classify)
```

---

*GeoCascade | Chapter 14 | Torres del Paine, Patagonia, Chile*
*Implementation: Python open-source (geocascade_env) | ArcPy optional | ENVI API optional | IDL reference included*
