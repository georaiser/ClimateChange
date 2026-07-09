# 🛰️ Chapter 8: Multi-Sensor Data Fusion & Machine Learning

> **GeoCascade** | Sentinel-2 · Sentinel-1 SAR · CopDEM · MODIS LST · Random Forest · Convergent Evidence

---

## 📋 Scripts Overview

### 1. `20_multisensor_data_fusion.py` — Multi-Sensor Data Cube

Creates a **4-band master data cube** (`cascade_master_stack.tif`) by reprojecting and co-registering data from four independent sensor sources:

| Band | Source | Type | Description |
|---|---|---|---|
| Band 1 | Sentinel-2 | Optical | NIR reflectance |
| Band 2 | Sentinel-1 SAR | Radar | VV backscatter (dB) |
| Band 3 | Copernicus DEM | Terrain | Elevation (m) |
| Band 4 | MODIS LST | Thermal | Land Surface Temperature |

All bands are resampled to a **common 10 m UTM grid** to ensure pixel-perfect co-registration.

**Outputs:**
- `cascade_master_stack.tif` — 4-band co-registered GeoTIFF
- Fusion stats table (per-band min/max/mean/nodata %)

**Run:**
```bash
python 20_multisensor_data_fusion.py
```

---

### 2. `21_cascade_risk_modeling.py` — Random Forest Classifier *(UPGRADED)*

Trains a **Random Forest classifier** on the 4-band data cube to map land cover and glacier probability.

**Land cover classes:**

| Class | Label |
|---|---|
| 💧 Water | Open water bodies |
| 🧊 Glacier / Ice | Permanent snow and ice |
| 🌿 Land / Vegetation | Vegetated and bare terrain |

**Key improvements:**

| Feature | Description |
|---|---|
| OOB accuracy | Out-of-Bag score — free cross-validation, no held-out set needed |
| Glacier probability TIFF | Continuous probability output — most useful for risk mapping |
| `nodata=255` | Standard nodata for 8-bit classification outputs |
| Flat index reconstruction | Correct valid-pixel reconstruction via `flat_indices` — avoids shape mismatch |
| `plt.close()` | Prevents figure memory leaks in batch runs |

> [!IMPORTANT]
> **OOB Score** — Random Forest trains each tree on a bootstrap sample (≈63% of training pixels). The remaining ≈37% unused pixels ("out-of-bag") are used to test that tree for free. The OOB score is a reliable generalization estimate without requiring a separate validation split. It is enabled via `oob_score=True` in `RandomForestClassifier`.

**Run:**
```bash
python 21_cascade_risk_modeling.py
```

---

### 3. `22_combined_insights_engine.py` — Combined Insights Engine

Full capstone analysis combining all sensor outputs into a **risk summary dashboard**. Integrates classifications, index layers, and terrain data into a unified visualization.

**Run:**
```bash
python 22_combined_insights_engine.py
```

---

### 4. `23_real_data_convergence.py` — Real Data Convergence Dashboard *(NEW CAPSTONE)*

Integrates **ALL real-world data streams** into a single convergence dashboard — the culminating script of the GeoCascade pipeline.

**Input data streams:**

| Source | Data Type |
|---|---|
| ERA5 | 30-year climate reanalysis |
| CHIRPS | Precipitation time series |
| 7-station network | In-situ meteorological observations |
| Sentinel-2 | NDVI vegetation index |
| Sentinel-1 SAR | Radar backscatter anomaly |
| Copernicus DEM | Terrain elevation |
| RGI 7.0 | Official glacier outlines |

**Environmental Stress Index (ESI):**

$$ESI = w_1 \cdot \hat{V}_{stress} + w_2 \cdot \hat{SAR}_{anomaly} + w_3 \cdot \hat{E}_{exposure}$$

Where:
- $\hat{V}_{stress}$ = normalized vegetation stress (inverted NDVI)
- $\hat{SAR}_{anomaly}$ = normalized radar structural anomaly
- $\hat{E}_{exposure}$ = normalized elevation exposure from DEM
- $w_1, w_2, w_3$ = tunable weights (default equal weighting)

**Dashboard:** 8-panel dark-theme convergence visualization showing all sensor layers side-by-side with the composite ESI map.

**Run:**
```bash
python 23_real_data_convergence.py
```

> [!NOTE]
> Script 23 requires ERA5 and CHIRPS data downloaded by Chapter 1 scripts. Run Ch01 acquisition scripts first.

---

## 🛠️ Installation

```bash
mamba install -n geocascade_env -c conda-forge \
    pystac-client planetary-computer rasterio geopandas \
    scikit-learn pandas numpy matplotlib pyproj requests -y
```

---

## 🔄 Recommended Run Order

```
Chapter 01 (ERA5 + CHIRPS acquisition)
         │
         ▼
    20_multisensor_data_fusion.py
    (builds cascade_master_stack.tif)
         │
         ▼
    21_cascade_risk_modeling.py
    (trains RF on data cube)
         │
         ▼
    23_real_data_convergence.py
    (full convergence dashboard)
```

> [!IMPORTANT]
> Run order: **20 → 21 → 23**. Script 22 can be run independently. Script 23 requires both ERA5 + CHIRPS outputs from Chapter 1.

---

## 🧠 Key Concepts

### Data Cube: Co-Registered Multi-Sensor Stack

> [!NOTE]
> A **data cube** is a co-registered multi-sensor raster stack where every pixel contains aligned values from Optical + Radar + Terrain + Thermal sensors. Co-registration to a common grid (10 m UTM) is essential — even sub-pixel misalignment degrades classifier accuracy significantly.

### OOB Score — Free Cross-Validation

Each Random Forest tree is trained on a **bootstrap sample** (~63% of pixels). The remaining ~37% unused pixels test that tree's accuracy for free:

```python
rf = RandomForestClassifier(n_estimators=200, oob_score=True, random_state=42)
rf.fit(X_train, y_train)
print(f"OOB Accuracy: {rf.oob_score_:.3f}")  # No validation split needed
```

### Convergent Evidence

> [!IMPORTANT]
> Environmental stress signals that are **consistent across multiple independent sensors** are far more reliable than any single observation. A glacier retreating signal appearing simultaneously in optical (NDVI), radar (SAR anomaly), thermal (LST rise), and terrain (DEM change) provides high-confidence evidence — this is the core principle of the ESI framework.

### ESI — Environmental Stress Index

The ESI combines three normalized stress dimensions:

| Component | Sensor | Stress Indicator |
|---|---|---|
| Vegetation stress | Sentinel-2 NDVI | Low NDVI → high stress |
| SAR anomaly | Sentinel-1 VV dB | Structural change |
| Elevation exposure | Copernicus DEM | High elevation → high exposure |

### RGI 7.0 — Glacier Validation

The **Randolph Glacier Inventory (RGI) 7.0** provides the official global glacier polygon database. Compare Random Forest–classified ice extents against RGI boundaries to validate classification accuracy and quantify glacier area change.

---

## 📁 Expected Outputs

```
Chapter_08/
├── outputs/
│   ├── cascade_master_stack.tif        # 4-band co-registered data cube
│   ├── rf_classification.tif           # Land cover classification map
│   ├── glacier_probability.tif         # Continuous glacier probability (0–1)
│   ├── fusion_stats.csv                # Per-band statistics table
│   ├── convergence_dashboard.png       # 8-panel dark-theme ESI dashboard
│   └── esi_composite.tif              # Environmental Stress Index raster
```