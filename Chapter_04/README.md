# 🌿 Chapter 4: Ecological & Vulnerability Analysis

> **GeoCascade Pipeline — Stage 4**
> Species distribution modeling, multi-criteria climate vulnerability indexing, and conservation prioritization.

---

## 📋 Overview

Chapter 4 synthesizes all data produced in Chapters 1–3 into two applied analysis workflows:

1. **Ecological Niche Modeling (SDM)** — predicts where the endangered Patagonian Huemul Deer can survive using terrain + vegetation predictors, combining unsupervised clustering (K-Means) and supervised classification (Random Forest).

2. **Climate Vulnerability Index (MCDA)** — combines four environmental layers into a single weighted vulnerability score, replicating the GIS "Weighted Overlay" tool used in disaster risk assessment and conservation planning.

> [!IMPORTANT]
> Chapter 4 reads outputs from Chapters 2 and 3. Run Chapters 2 and 3 first, or let the fallback STAC download run automatically.

---

## 📁 Scripts

| # | File | Topic | Key Outputs |
|---|------|--------|-------------|
| 12 | `12_ecological_niche_modeling.py` | K-Means unsupervised + RF SDM (Huemul Deer) | `kmeans_unsupervised.tif`, `ecological_niche_model.tif`, `feature_importance.csv`, 4-panel dark figure |
| 13 | `13_climate_vulnerability_index.py` | MCDA Weighted Overlay — Climate Vulnerability Index | `vulnerability_index.tif`, `vulnerability_class.tif`, `vulnerability_statistics.csv`, 5-panel dark figure |

---

## 🚀 Setup

```bash
conda activate geocascade_env

mamba install -n geocascade_env -c conda-forge scikit-learn rasterio numpy matplotlib pandas scipy pyproj pystac-client planetary-computer -y
```

---

## ▶️ Running Scripts

```bash
# Ecological niche modeling (reads Ch02 NDVI + Ch03 DEM/slope)
python Chapter_04/12_ecological_niche_modeling.py

# Climate vulnerability index (reads Ch02 NDVI + Ch02 glacier + Ch03 DEM/slope)
python Chapter_04/13_climate_vulnerability_index.py
```

Both scripts auto-download from Planetary Computer if local Chapter 2/3 outputs are not found.

---

## 🔬 Methods Deep-Dive

### Script 12 — Ecological Niche Modeling

**Why two models?**

| Method | Data needed | Output | Use case |
|--------|-------------|--------|----------|
| K-Means | None (unsupervised) | Terrain clusters | Exploratory mapping, no species data |
| RF SDM | Presence/absence | Suitability probability | Habitat mapping, climate change projection |

**K-Means normalization** (critical):
Features must be normalized to [0, 1] before K-Means; otherwise elevation (0–2500 m) dominates over NDVI (−1 to +1) purely because of magnitude:
```python
X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-9)
```

**Random Forest SDM — synthetic training data:**
```python
# Huemul preference rule (literature-derived):
if elevation < 800m AND slope < 20° AND NDVI > 0.4:
    presence probability = 80%   # + 20% noise (real-world uncertainty)
else:
    presence probability = 5%    # stray occurrences outside core habitat
```

**Model evaluation:**
- Uses a 20% hold-out validation split
- Reports ROC-AUC (area under the Receiver Operating Characteristic curve)
- AUC > 0.85 = good discrimination; AUC < 0.7 = model needs improvement

**Why not MaxEnt?**
MaxEnt requires presence-only data + complex regularization tuning. RF SDM is conceptually cleaner: standard binary classifier, directly interpretable via feature importance — better for teaching.

---

### Script 13 — Climate Vulnerability Index (MCDA)

**MCDA weights:**

| Layer | Variable | Weight | Rationale |
|-------|----------|--------|-----------|
| Vegetation exposure | 1 − NDVI (scaled) | **50%** | Dominant driver — sparse vegetation = bare soil exposed to erosion and drought |
| Erosion risk | Slope (scaled) | **20%** | Steep terrain loses soil rapidly under intense precipitation |
| Elevation exposure | DEM (scaled) | **15%** | Higher elevation = harsher climate, longer snow season |
| Glacier proximity | Inverse distance to NDSI ice | **15%** | Retreat changes local hydrology and microclimate |

> [!NOTE]
> Weights must sum to 1.0. To customize, edit the `WEIGHTS` dict at the top of script 13 and re-run.

**Glacier proximity layer** (scipy `distance_transform_edt`):
```python
dist = distance_transform_edt(1 - ice_mask)   # Euclidean distance in pixel space
proximity = exp(-dist / 20.0)                  # exponential decay: 20-pixel half-distance
```

**Vulnerability classes:**

| Class | CVI Range | Label |
|-------|-----------|-------|
| 1 | 0.0 – 0.2 | Very Low |
| 2 | 0.2 – 0.4 | Low |
| 3 | 0.4 – 0.6 | Moderate |
| 4 | 0.6 – 0.8 | High |
| 5 | 0.8 – 1.0 | Very High |

---

## 📂 Output Directory Structure

```
Chapter_04/
└── data/
    └── processed/
        ├── niche/
        │   ├── kmeans_unsupervised.tif         ← 4-class terrain clusters
        │   ├── ecological_niche_model.tif       ← RF habitat suitability [0–1]
        │   ├── feature_importance.csv
        │   ├── niche_statistics.csv
        │   └── ecological_niche_model.png       ← 4-panel dark figure
        └── vulnerability/
            ├── vulnerability_index.tif          ← CVI [0–1], continuous
            ├── vulnerability_class.tif          ← 5-class categorical
            ├── vulnerability_statistics.csv
            └── climate_vulnerability.png        ← 5-panel dark figure
```

---

## 🖥️ ArcGIS Pro Integration

### Script 12 — Habitat Suitability

```
1. Add ecological_niche_model.tif
   Symbology > Stretched > Yellow-Green color ramp
   (0 = unsuitable / dark, 1 = optimal habitat / bright green)

2. Convert to conservation zones:
   Spatial Analyst > Raster Calculator:
     Con("ecological_niche_model.tif" > 0.7, 1, 0)
   → binary "Priority Habitat" raster

3. Combine with vulnerability:
   Raster Calculator:
     "ecological_niche_model.tif" * (1 - "vulnerability_index.tif")
   → conservation urgency index (high habitat + high threat = priority)
```

### Script 13 — Vulnerability Index

```
1. Add vulnerability_index.tif
   Symbology > Classified > 5 Manual Breaks > Inferno color ramp

2. Identify highest-risk areas:
   Select By Attributes (Query Builder):
     CVI_Value > 0.7

3. Weighted Overlay verification:
   Spatial Analyst > Weighted Overlay
   Input the same 4 rasters with the same weights to validate
   against the Python output (should be < 0.01 difference)

4. Combine with Chapter 5 zonal statistics:
   Zonal Statistics As Table (watershed polygons + vulnerability_index.tif)
   → mean CVI per drainage basin for prioritized conservation
```

---

## 🔵 ENVI 5.6 Integration

### Habitat Suitability
```
File > Open > ecological_niche_model.tif
Display > Density Slice → 4 classes: 0–0.25, 0.25–0.5, 0.5–0.75, 0.75–1.0
```

### Vulnerability Index
```
File > Open > vulnerability_index.tif
Display > Density Slice → 5 classes matching vulnerability_class.tif

; Band Math for high-risk mask:
b1 ge 0.7
```

---

## 🔗 Pipeline Data Flow

```
Chapter 1                Chapter 2               Chapter 3
Temperature Surface  →   NDVI.tif           →   slope_degrees.tif
weather_stations.csv     glacier_mask.tif        copernicus_dem.tif
                         ↓                       ↓
                    ┌────────────────────────────────────┐
                    │       Chapter 4 (this chapter)     │
                    │  Script 12: RF SDM = Dem + Slope + NDVI         │
                    │  Script 13: CVI = NDVI + Slope + DEM + Glacier  │
                    └────────────────────────────────────┘
                                    ↓
                         Chapter 5: Zonal Statistics
                         (mean suitability per watershed)
```

---

## ⚠️ Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| `scipy` not found | Not installed | `mamba install -n geocascade_env scipy -y` |
| K-Means produces all one cluster | All features constant (e.g., flat synthetic DEM) | Check that DEM has valid elevation variation |
| SDM AUC very low (~0.5) | Training data rule not matching actual terrain | Check NDVI range; if all NDVI < 0.4, all points will be "absent" class |
| Glacier proximity layer all zeros | No NDSI glacier mask found | Run Ch02 script 07 first, or accept zero-weight glacier layer |
| STAC search returns 0 items | Cloud cover too strict or date range too narrow | Script 12/13 auto-fallback; check console for which fallback ran |

---

## 📖 Key References

- Phillips, S.J., Anderson, R.P., Schapire, R.E. (2006). *Maximum entropy modeling of species geographic distributions.* Ecological Modelling.
- Breiman, L. (2001). *Random Forests.* Machine Learning.
- Malczewski, J. (1999). *GIS and Multicriteria Decision Analysis.* Wiley.
- Patagonian Huemul (*Hippocamelus bisulcus*): [IUCN Red List](https://www.iucnredlist.org/species/10054/21669480) — Endangered