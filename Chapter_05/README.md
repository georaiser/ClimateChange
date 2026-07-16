# 💧 Chapter 5: Moisture & Zonal Statistics

> **GeoCascade Pipeline — Stage 5**
> Sentinel-2 SWIR moisture indices, drought stress mapping, and cross-chapter zonal statistics.

---

## 📋 Overview

Chapter 5 answers two complementary questions:

1. **Where is vegetation stressed by drought?** — NDMI, MSI, NDWI, NMDI from Sentinel-2 SWIR bands
2. **How do ALL pipeline outputs vary across the landscape?** — Zonal statistics aggregating every Chapter 2–4 raster across analysis zones

> [!IMPORTANT]
> Script 15 reads outputs from Chapters 2, 3, and 4. Run those chapters first to get the richest analysis. The script will still work with whatever rasters it finds and downloads a fallback DEM if nothing is available.

---

## 📁 Scripts

| # | File | Topic | Key Outputs |
|---|------|--------|-------------|
| 14 | `14_moisture_stress_indices.py` | NDMI, MSI, NDWI, NMDI from Sentinel-2 | 4 moisture GeoTIFFs, `moisture_statistics.csv`, 4-panel dark figure |
| 15 | `15_zonal_statistics.py` | Cross-chapter zonal summary (3×3 zone grid) | `management_zones.gpkg`, `zonal_statistics_summary.csv`, heatmap figure |

---

## 🚀 Setup

```bash
conda activate geocascade_env

mamba install -n geocascade_env -c conda-forge \
    rasterio numpy matplotlib pandas scikit-learn \
    geopandas shapely pystac-client planetary-computer pyproj -y

pip install rasterstats
```

---

## ▶️ Running Scripts

```bash
# Moisture indices (reads Ch02 data, falls back to Planetary Computer)
python Chapter_05/14_moisture_stress_indices.py

# Zonal statistics (reads Ch02-04 outputs, creates 3x3 analysis grid)
python Chapter_05/15_zonal_statistics.py
```

---

## 🔬 Methods Deep-Dive

### Script 14 — Moisture & Drought Indices

**Why SWIR bands for moisture?**
Water molecules strongly absorb radiation in the Shortwave Infrared (1400–2500 nm). Sentinel-2 Band 11 (SWIR1, 1610 nm) and Band 12 (SWIR2, 2190 nm) directly measure how much SWIR is absorbed by leaf water. When plants lose water, SWIR reflectance rises.

**Index formulas:**

| Index | Formula | Bands | Higher = |
|-------|---------|-------|----------|
| NDMI | (NIR − SWIR1) / (NIR + SWIR1) | B08, B11 | Wetter canopy |
| MSI | SWIR1 / NIR | B11, B08 | More stress |
| NDWI | (Green − NIR) / (Green + NIR) | B03, B08 | Surface water |
| NMDI | (NIR − (SWIR1 − SWIR2)) / (NIR + (SWIR1 − SWIR2)) | B08, B11, B12 | Less drought |

**NDMI drought thresholds:**
```
NDMI > 0.2  → No stress (healthy, well-watered)
NDMI 0.0–0.2 → Mild stress
NDMI −0.1–0.0 → Moderate stress
NDMI < −0.1   → Severe drought stress
```

**Critical: independent windows for each band resolution:**
```python
# WRONG: reusing NIR window (10m) for SWIR (20m)
swir = src_swir.read(1, window=win_nir)   # WRONG — wrong pixel grid!

# CORRECT: compute a new window from SWIR's own transform
win_swir = from_bounds(minx, miny, maxx, maxy, src_swir.transform)
swir = src_swir.read(1, window=win_swir, out_shape=nir.shape, resampling=bilinear)
```

---

### Script 15 — Zonal Statistics

**What zonal statistics does:**
```
For each polygon:
  For each raster:
    Extract all pixel values whose CENTER falls inside the polygon
    Compute: mean, std, min, max, median, count
```

**Zone design — 3×3 grid (9 zones):**
The BBOX is divided into a 3×3 grid of equal cells (named A–I from SW to NE). This provides finer spatial granularity than simple NW/NE/SW/SE quadrants, enabling detection of the east-west precipitation gradient and the north-south temperature gradient simultaneously.

**Rasters analyzed:**

| Raster | Source | What it tells us by zone |
|--------|--------|--------------------------|
| NDVI | Ch02/07 | Vegetation density per zone |
| NDMI | Ch05/14 | Moisture stress per zone |
| DEM | Ch03/10 | Elevation range per zone |
| Slope | Ch03/10 | Terrain steepness per zone |
| SDM | Ch04/12 | Habitat suitability per zone |
| CVI | Ch04/13 | Climate vulnerability per zone |
| Glacier NDSI | Ch02/07 | Ice fraction per zone |

**Output table structure (wide format):**
```
Zone | NDVI_mean | NDMI_mean | DEM_mean | Slope_mean | SDM_mean | CVI_mean | ...
A    | 0.412     | 0.183     | 623      | 18.2       | 0.71     | 0.54     | ...
B    | 0.298     | 0.091     | 891      | 24.7       | 0.43     | 0.67     | ...
```

---

## 📂 Output Directory Structure

```
Chapter_05/
└── data/
    └── processed/
        ├── moisture/
        │   ├── ndmi.tif
        │   ├── msi.tif
        │   ├── ndwi.tif
        │   ├── nmdi.tif
        │   ├── moisture_statistics.csv
        │   └── moisture_stress_indices.png
        └── zonal/
            ├── management_zones.gpkg      ← 3×3 grid, 9 zones
            ├── zonal_statistics_long.csv  ← one row per zone-raster pair
            ├── zonal_statistics_summary.csv  ← wide: one row per zone
            └── zonal_statistics.png       ← heatmap + zone map
```

---

## 🖥️ ArcGIS Pro Integration

### Moisture Analysis (Script 14)
```
1. Add ndmi.tif
   Symbology > Stretched > Blue-Red Diverging
   (Blue = moist, Red = stressed)

2. Drought mask:
   Raster Calculator: Con("ndmi.tif" < -0.1, 1, 0)
   → Binary raster: 1 = severe drought stress

3. Combine with SDM:
   Raster Calculator: "ecological_niche_model.tif" * ("ndmi.tif" < 0)
   → Species at risk in drought areas
```

### Zonal Statistics (Script 15)
```
Spatial Analyst > Zonal Statistics As Table
  Input zone raster or zone field: management_zones.gpkg
  Input value raster: any output TIF
  Statistics type: All
  → Validates script 15 output (should match to rounding precision)
```

---

## 🔵 ENVI 5.6 Integration

```
; NDMI via Band Math
; b1=B08 (NIR), b2=B11 (SWIR1)
(b1 - b2) / (b1 + b2)

; Drought mask via density slice on NDMI TIF
File > Open > ndmi.tif
Display > Density Slice: add break at -0.1 for stress threshold
```

---

## ⚠️ Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| `rasterstats` not found | Not installed via pip | `pip install rasterstats` |
| All zones show same value | CRS mismatch between zones and raster | Script auto-reprojects; check console for CRS messages |
| NDMI all NaN | B11 not found or wrong scale factor | Check Sentinel-2 asset key: use `B11`, not `B8A` |
| Zonal stats CSV is empty | No rasters found, STAC fallback failed | Run Ch02-04 first; check Planetary Computer connection |

---

## 📖 Key References

- Gao, B.C. (1996). *NDWI – A normalized difference water index for remote sensing of vegetation liquid water from space.* Remote Sensing of Environment.
- Rock, B.N. et al. (1986). *Remote detection of forest damage.* BioScience.
- Wang, L. & Qu, J.J. (2007). *NMDI: A normalized multi-band drought index for monitoring soil and vegetation moisture.* Geophysical Research Letters.
- Zhu, Z. et al. (2012). *Object-based cloud and cloud shadow detection in Landsat imagery.* Remote Sensing of Environment.