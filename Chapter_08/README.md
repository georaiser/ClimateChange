# Chapter 8: Data Fusion & Cascade Effect Modeling

## 🎯 Academic Objective

The previous chapters each examined one physical property in isolation: climate, spectral signatures, topography, hydrology, radar. But Earth systems don't operate in isolation — they interact in **cascading chains**:

$$\text{Temperature Rise} \rightarrow \text{Glacier Melt} \rightarrow \text{Runoff Increase} \rightarrow \text{Flood Risk} \rightarrow \text{Vegetation Shift}$$

Modeling this chain requires **Multi-Sensor Data Fusion**: aligning vastly different satellite datasets into a single multidimensional data cube, then applying Machine Learning and **Convergent Evidence Analysis** to extract insights no single sensor can provide.

By the end of this chapter you will be able to:
- Align 4 satellite sensors (Optical + Radar + Thermal + DEM) onto a common spatial grid
- Train a Random Forest classifier on multi-sensor training data
- Compute three composite environmental insight scores (ESI, CVS, WSI)
- Generate a Convergent Risk Map showing where ALL risk factors coincide simultaneously

---

## 🛠️ Scripts & Modules

### `20_multisensor_data_fusion.py`
Builds a 4-band multi-sensor Data Cube by reprojecting all layers to a common 10m Sentinel-2 grid.

**Data sources fused:**
| Band | Sensor | Native Resolution | Property |
|------|--------|-----------------|----------|
| 1 | Sentinel-2 NIR (B08) | 10m | Vegetation / surface reflectance |
| 2 | Sentinel-1 SAR VV | 10m | Surface roughness / water detection |
| 3 | Copernicus DEM | 30m | Elevation (upsampled to 10m) |
| 4 | MODIS LST | 1000m | Land Surface Temperature (downsampled) |

> [!CAUTION]
> **SAR dB Conversion Order — Mathematically Critical**
>
> The log transform (linear → dB) must be applied **before** any resampling or reprojection. Applying log to bilinearly-interpolated linear values produces different results because $\log(\text{mean}) \neq \text{mean}(\log)$:
> ```python
> # ✅ CORRECT — apply log on raw linear values first
> vv_linear = src.read(1)
> vv_db = 10 * np.log10(np.where(vv_linear > 0, vv_linear, np.nan))
> # THEN reproject vv_db to master grid
>
> # ❌ WRONG — reprojecting linear values then taking log
> vv_reprojected = reproject(vv_linear, ...)  # bilinear interpolation on linear
> vv_db = 10 * np.log10(vv_reprojected)       # log of averaged linear values
> ```

**Output:** `cascade_master_stack.tif` — a 4-band GeoTIFF with band descriptions. Load in ENVI for false-color composites (e.g., Red=Radar, Green=Optical, Blue=Thermal).

---

### `21_cascade_risk_modeling.py`
Trains a Random Forest classifier on the 4-band data cube to produce a land cover map.

**Synthetic training data generation (percentile thresholds):**
| Class | NIR | SAR (dB) | Elevation | Label |
|-------|-----|----------|-----------|-------|
| Water | Low (<30th pct) | Very low (<-18dB) | Low | 1 |
| Glacier/Ice | Medium | Low-medium | High (>75th pct) | 2 |
| Land/Vegetation | High (>70th pct) | Medium-high | Medium | 3 |

> [!WARNING]
> **Data Leakage — Academic Teaching Notice**
> The training labels are derived from the same data being classified. In a real research context, this constitutes **circular reasoning** — the classifier learns rules that were already applied to the data. This is a pedagogical device to demonstrate the Random Forest workflow. For real projects, use independent field survey data or manually digitized training polygons.

**Feature Importance — console bar chart:**
```
S2 NIR (Optical)     : ████████████████████ 0.412
S1 SAR dB (Radar)    : ██████████████       0.287
DEM Elevation (m)    : ██████████           0.201
MODIS LST (°C)       : ████                 0.100
```
This answers: *which sensor contributed the most discriminative information for land cover classification?*

**Land Cover Summary table** (printed after classification):
```
Water             :  12,384 px ( 8.3%)
Glacier/Ice       :  23,901 px (16.1%)
Land/Vegetation   : 112,744 px (75.6%)
```

---

### `22_combined_insights_engine.py` ⭐ NEW

The most advanced script in the curriculum. Fuses **5 sensor layers** simultaneously and computes three composite environmental insight scores using **Convergent Evidence Analysis**.

#### The Three Composite Insight Scores

**① ESI — Ecological Stress Index**
$$ESI = 0.5 \cdot (1 - \hat{NDVI}) + 0.3 \cdot \hat{LST} + 0.2 \cdot \hat{Slope}$$

Where $\hat{x}$ denotes normalization to [0,1] using the 2nd/98th percentile. High ESI = degraded, heat-stressed, erosion-prone ecosystem.

**② CVS — Cryosphere Vulnerability Score**
$$CVS = 0.4 \cdot \hat{LST} + 0.4 \cdot (1 - \hat{NDSI}) + 0.2 \cdot (1 - \hat{SAR})$$

Computed **only over pixels where NDSI > 0.2** (actual glaciated surfaces). High CVS = warm + snow-poor + radar-smooth = actively melting ice.

**③ WSI — Water Stress Compound Index**
$$WSI = 0.4 \cdot \hat{NDWI} + 0.4 \cdot \hat{NDSI} + 0.2 \cdot (1 - \hat{LST})$$

High WSI = abundant water (surface water + upstream snow + cool temperatures reducing evaporation).

#### Convergent Risk Map
A pixel-level count of how many risk factors are simultaneously active:
- ESI > 0.6 (ecological stress zone)
- CVS > 0.6 (glacial melt zone)
- WSI < 0.3 (water-scarce zone)

**Pixels where all 3 factors coincide** = the most critical zones for environmental intervention.

#### 12-Panel Dark-Mode Dashboard
| Row | Panels |
|-----|--------|
| Row 1 | ① DEM, ② Slope, ③ SAR backscatter, ④ MODIS LST |
| Row 2 | ⑤ NDVI, ⑥ NDWI, ⑦ NDSI, ⑧ ESI |
| Row 3 | ⑨ CVS, ⑩ WSI, ⑪ Convergent Risk Map, ⑫ Elevation-NDVI Profile |

> [!NOTE]
> **Shape Harmonization:** Slight pixel-count differences between layers (due to rasterio float-precision windows) are automatically corrected using `scipy.ndimage.zoom` before computing composite scores. This ensures all arrays are exactly `(target_h, target_w)`.

---

## 🚀 How to Run

### Install Dependencies
```bash
mamba activate geocascade_env
mamba install -n geocascade_env -c conda-forge \
    pystac-client planetary-computer rasterio geopandas \
    numpy matplotlib scikit-learn rasterstats scipy -y
```

### Execute Scripts
```bash
# Step 1: Build 4-band multi-sensor data cube
python 20_multisensor_data_fusion.py

# Step 2: Random Forest land cover classification
python 21_cascade_risk_modeling.py

# Step 3: Full convergent evidence analysis (all 5 sensors → 3 insight scores)
python 22_combined_insights_engine.py
```

**Script 22 outputs:**
- `data/combined_insights/combined_insights_dashboard.png` — 12-panel dark-mode dashboard
- `data/combined_insights/combined_insights_report.md` — statistical summary with key insights

---

## 🗺️ GIS Interoperability

**ENVI:** Load `cascade_master_stack.tif` (4-band) → assign **Red=SAR, Green=NIR, Blue=Thermal** for a false-color composite that simultaneously shows ice (blue-cold), vegetation (green), and radar-rough terrain (red).

**ArcGIS Pro:** Use the **Image Analyst → Classification Wizard** to run SVM or RF classification directly on `cascade_master_stack.tif` via the GUI — compare against our Python Random Forest output.

> [!TIP]
> The three composite scores (ESI, CVS, WSI) from `22_combined_insights_engine.py` can be used as direct inputs to an Environmental Impact Assessment (EIA) report. High ESI zones require ecological restoration planning; high CVS zones require glacial melt water supply forecasting; low WSI zones require irrigation or water conservation planning.
