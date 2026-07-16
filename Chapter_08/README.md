# 🔮 Chapter 8: Multi-Sensor Data Fusion & Cascade Risk Modeling

> **GeoCascade Pipeline — Stage 8**
> 4-band data cube fusion, Random Forest classification, convergent evidence analysis,
> and the complete multi-source environmental stress dashboard.

---

## 📋 Overview

Chapter 8 is the **synthesis chapter** — it integrates every previous data stream
into unified multi-sensor analyses that answer questions no single sensor can address.

| Script | Topic | Key Outputs |
|--------|-------|-------------|
| 20 | 4-band data cube: S2 + SAR + DEM + MODIS | `cascade_master_stack.tif`, `fusion_statistics.csv`, 4-panel figure |
| 21 | Random Forest classification + glacier risk | `cascade_ml_prediction.tif`, `glacier_probability_map.tif`, 4-panel figure |
| 22 | Convergent Evidence Analysis (ESI, CVS, HECI) | 3 composite indices, 8-panel dashboard |
| 23 | Real data convergence (ERA5 + CHIRPS + SAR + NDVI) | `environmental_stress_composite.tif`, 8-panel dashboard |

---

## 🚀 Setup

```bash
conda activate geocascade_env

mamba install -n geocascade_env -c conda-forge \
    rasterio numpy matplotlib pandas geopandas shapely \
    scikit-learn pystac-client planetary-computer pyproj osmnx -y
```

---

## ▶️ Run Order

```bash
# Run Chapters 2, 3, 7 first to build cached inputs
# Then run Chapter 8 in order:

python Chapter_08/20_multisensor_data_fusion.py      # Build 4-band cube
python Chapter_08/21_cascade_risk_modeling.py        # RF classification
python Chapter_08/22_combined_insights_engine.py     # ESI / CVS / HECI indices
python Chapter_08/23_real_data_convergence.py        # Full convergent dashboard
```

---

## 🔬 Methods Deep-Dive

### Script 20 — Data Fusion Engine

**The alignment problem:**
Each sensor has its own native CRS, resolution, and spatial footprint:

| Sensor | Native CRS | Native Resolution |
|--------|-----------|-------------------|
| Sentinel-2 (B08) | UTM (varies) | 10m |
| Sentinel-1 RTC | UTM (varies) | 10m |
| Copernicus DEM | EPSG:4326 | ~30m |
| MODIS LST | Sinusoidal | 1000m |

**Solution: single master grid.**
All layers are warped to match Sentinel-2's 10m UTM grid using `rasterio.warp.reproject`:

```python
reproject(
    source=rasterio.band(src, 1),
    destination=dest_array,
    src_transform=src.transform,   # source native geometry
    src_crs=src.crs,               # source native projection
    dst_transform=master_profile["transform"],   # 10m Sentinel-2 grid
    dst_crs=master_profile["crs"],
    resampling=Resampling.bilinear
)
```

**Resampling choices:**
- DEM (30m → 10m): bilinear upsampling — smooth interpolation between DEM posts
- MODIS (1km → 10m): bilinear — upsampling 100x creates a smoothed thermal surface
  (physically acceptable because LST has km-scale spatial autocorrelation)

> [!IMPORTANT]
> Script 20 uses local cached outputs from Ch02/Ch03/Ch07 to avoid redundant downloads.
> Run those chapters first for fastest execution.

**MODIS fill value — critical fix:**
```python
# WRONG: masks valid cold pixels (glaciers, snow!)
arr[arr < 7500] = np.nan

# CORRECT: fill = 0 only
fill_mask = arr == 0
lst_k = arr * 0.02            # scale factor: DN * 0.02 = Kelvin
lst_c = np.where(fill_mask | (lst_k < 150), np.nan, lst_k - 273.15)
```

---

### Script 21 — Random Forest Cascade Risk Modeling

**Why Random Forest for multi-sensor fusion?**
- Handles heterogeneous feature spaces: reflectance [0-1], dB [-25 to 0], metres [0-3000], Celsius [-20 to 20]
- No feature normalization required (unlike k-NN or SVM)
- Out-of-Bag (OOB) accuracy = free cross-validation estimate
- Feature importances reveal which sensor drives classification

**Training label strategy (percentile-based):**
Labels are generated DYNAMICALLY from geophysical percentiles, not hardcoded DN thresholds:

```python
# Water: Very low SAR (p10) AND Very low NIR (p10)
# Glacier: High elevation (p80) AND Cold LST (p20)
# Vegetation: High NIR (p80) AND Warm LST (p70)
```

This ensures labels exist across different dates and scenes where absolute values shift.

**Land cover classification:**

| Class | SAR VV | NIR | Elevation | LST | Physical meaning |
|-------|--------|-----|-----------|-----|-----------------|
| 0 | — | — | — | — | NoData |
| 1 Water | Very low | Very low | Any | — | Lago Grey, lakes |
| 2 Glacier | Low-moderate | Low | High | Very cold | Grey Glacier, snowfields |
| 3 Vegetation | Moderate | High | Low-mid | Warm | Lenga beech forests, grassland |

**Output TIFs:**
- `cascade_ml_prediction.tif`: uint8 class map (0-3), nodata=255
- `glacier_probability_map.tif`: float32 per-pixel glacier probability [0-1]

---

### Scripts 22 & 23 — Convergent Evidence Analysis

**What "convergence" means:**
If only one sensor shows a drought signal, it could be noise. If NDVI (optical), NDMI (SWIR), SAR backscatter (radar), AND ERA5 temperature all show the same degradation signal simultaneously, the evidence is convergent and reliable.

**Script 22 composite indices:**

| Index | Components | Physical Meaning |
|-------|-----------|-----------------|
| ESI (Ecological Stress Index) | NDVI + NDMI + slope | Where is vegetation under stress? |
| CVS (Cryosphere Vulnerability Score) | Elevation + LST + glacier mask | Which ice areas are most at risk? |
| HECI (Human-Environment Conflict Index) | OSMnx roads + CVS + ESI | Where does infrastructure intersect sensitive zones? |

**Script 23 real data sources:**
1. ERA5 climate time series (temperature trend via Open-Meteo API)
2. CHIRPS spatial precipitation (Andes rain shadow gradient)
3. 7-station ground network (climate gradient validation)
4. Sentinel-2 NDVI (vegetation health)
5. Sentinel-1 SAR VV (surface roughness / water)
6. Copernicus DEM (topographic context)
7. RGI 7.0 glacier outlines (official ice extent)

---

## 📂 Output Structure

```
Chapter_08/
└── data/
    ├── processed/
    │   ├── fusion/
    │   │   ├── cascade_master_stack.tif      ← 4-band aligned cube
    │   │   ├── fusion_statistics.csv
    │   │   └── data_cube_visualization.png
    │   ├── ml/
    │   │   ├── cascade_ml_prediction.tif     ← 4-class RF map
    │   │   ├── glacier_probability_map.tif   ← continuous risk
    │   │   ├── feature_importance.csv
    │   │   ├── classification_report.csv
    │   │   └── cascade_ml_prediction.png
    │   ├── combined_insights/
    │   │   ├── esi_ecological_stress.tif
    │   │   ├── cvs_cryosphere_vulnerability.tif
    │   │   ├── heci_human_conflict.tif
    │   │   └── combined_insights_dashboard.png
    │   └── convergence/
    │       ├── environmental_stress_composite.tif
    │       ├── convergence_dashboard.png
    │       └── convergence_report.csv
    └── tmp/
        └── (temporary resampling files, auto-deleted)
```

---

## 🖥️ ArcGIS Pro Integration

```
Script 20 — Data Cube:
  Add cascade_master_stack.tif
  Layer Properties > Symbology > Composite
  Assign Band 1 (NIR) = Red, Band 2 (SAR) = Green, Band 4 (LST) = Blue
  → False-color composite shows vegetation/water/temperature simultaneously

  Use "Individual Bands" to inspect each layer:
    Band 1: NDVI analog (NIR reflectance)
    Band 2: SAR roughness index
    Band 3: Topographic elevation
    Band 4: Thermal heat island / cold glacier zones

Script 21 — Classification:
  Add cascade_ml_prediction.tif
  Symbology > Unique Values
    0 = transparent (NoData)
    1 = #2980b9 (Blue, Water)
    2 = #74b9ff (Light Blue, Glacier)
    3 = #27ae60 (Green, Vegetation)

  Add glacier_probability_map.tif
  Symbology > Stretched > Yellow to Red
  → High probability = yellow-orange-red (high cryosphere risk)

  Spatial Analyst > Zonal Statistics on probability map:
    Zone: RGI 7.0 glacier outlines
    → mean probability per glacier = validation vs official inventory

Script 22 — ESI/CVS/HECI:
  Combine outputs with Raster Calculator:
    "esi.tif" * "cvs.tif"
    → zones where BOTH ecological stress AND cryosphere vulnerability are high
    → top priority for conservation intervention
```

---

## 🔵 ENVI 5.6 Integration

```
; Open 4-band data cube
File > Open > cascade_master_stack.tif (multi-band)
Toolbox > Classification > Supervised > Support Vector Machine
  → compare RF result (Script 21) with ENVI SVM for validation

; RF prediction TIF
Classification > Post Classification > Class Statistics
  → pixel count + area per class validates Script 21 report

; Glacier probability map
Display > Color Table > Yellow-to-Red gradient
Enhancement > Linear 0-1 stretch
  → probability map shows continuous risk gradient
```

---

## ⚠️ Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| `cascade_master_stack.tif` not found | Script 20 not run | `python Chapter_08/20_multisensor_data_fusion.py` |
| RF: < 30 training pixels | Cube mostly NoData | Check Script 20 output; MODIS may be all fill |
| MODIS all NaN in stack | Wrong fill mask | Ensure `fill_mask = arr == 0` in Script 20 |
| Glacier class prob all 0 | Glacier not in RF classes | Scene may lack high-elevation pixels; check DEM layer |
| `osmnx` error in Script 22 | OSM API unavailable | Script gracefully skips HECI if osmnx fails |

---

## 📖 Key References

- Breiman, L. (2001). *Random Forests.* Machine Learning.
- Strozzi, T. et al. (2022). *Glacier area changes in Patagonia from Sentinel-1.* Remote Sensing.
- Immerzeel, W. et al. (2010). *Climate change will affect the Asian water towers.* Science.
- Rott, H. et al. (2018). *Changing pattern of ice flow and mass balance for glaciers discharging into the Larsen A and B embayments.* The Cryosphere.