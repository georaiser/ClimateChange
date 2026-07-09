# Chapter 4: Ecological Niche Modeling & Climate Vulnerability

## Academic Objective
Predict species habitat suitability (Patagonian Huemul Deer) using unsupervised K-Means
terrain clustering and supervised Random Forest SDM. Build a composite Climate Vulnerability
Index (ESI + CVS + WSI) from normalized environmental stress indicators.

---

## Scripts

### 12_ecological_niche_modeling.py — Species Distribution Modeling (SDM)

Two complementary approaches:

**1. Unsupervised K-Means (k=4 terrain clusters, no labels)**
Features: [NDVI, Elevation, Slope]. Discovers natural terrain archetypes
(glacier, alpine, forest, bare rock) without any training data.

**2. Supervised Random Forest SDM**
Trains on known presence points. Outputs probability map for species occurrence.
Feature importances rank elevation, slope, and NDVI by contribution.

> [!CAUTION]
> Slope must be computed with metres-per-degree conversion at 51 deg S:
>   pix_lat_m = res * 111000
>   pix_lon_m = res * 111000 * cos(lat_radians)
> Computing slope in geographic degrees gives a physically incorrect gradient.

**Critical bug fixes applied:**
- Slope computed outside with-block with metres/degree conversion
- nodata=-9999 (not np.nan)
- predict_proba uses valid_mask to preserve NaN ocean pixels
- plt.close() after all figures
- STAC empty guard for DEM

Run: `python 12_ecological_niche_modeling.py`

Outputs: sdm_probability_map.tif, sdm_comparison.png

---

### 13_climate_vulnerability_index.py — Composite Vulnerability Index

Builds a 3-component composite index:

  Vulnerability = (ESI + CVS + WSI) / 3

| Component | Meaning | Source |
|---|---|---|
| ESI | Environmental Stress Index | Normalized NDVI deficit |
| CVS | Climate Vulnerability Score | Normalized MODIS LST anomaly |
| WSI | Water Stress Index | Normalized MSI (SWIR1/NIR) |

Each component normalized to [0,1]:
  X_norm = (X - X_min) / (X_max - X_min)

> [!CAUTION]
> min_max_scale() MUST preserve NaN pixels. Setting NaN to 0 before normalization
> treats ocean as minimum elevation/NDVI, corrupting the entire scale.
> Fix: normalize only finite pixels, then restore NaN positions.

> [!WARNING]
> S2 band windows must use the S2 CRS/transform independently. NEVER reuse the
> DEM window (EPSG:4326 geographic) for S2 reads (UTM projected). Wrong area read.

**Fixes applied:** min_max_scale NaN preserved; window=win_s2 correct; slope metres/degree;
nodata=-9999; STAC guards; plt.close()

Run: `python 13_climate_vulnerability_index.py`

Outputs: climate_vulnerability_index.tif, vulnerability_map.png

---

## Key Concepts

| Concept | Explanation |
|---|---|
| SDM | Models probability of species presence from environmental predictors |
| Feature Importance | Random Forest ranks variables by split-purity contribution |
| NaN preservation | Normalization must not treat missing data as valid extremes |
| CRS mismatch | DEM geographic + S2 UTM windows are incompatible — compute separately |
| Composite Index | Averaging normalized sub-indices collapses multi-dim stress |

## Installation

```bash
mamba install -n geocascade_env -c conda-forge pystac-client planetary-computer rasterio scikit-learn numpy matplotlib pyproj -y
```