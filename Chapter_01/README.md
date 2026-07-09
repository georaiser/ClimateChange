# Chapter 1: Climatic Variables & STAC Acquisition

## 🎯 Academic Objective

Build an automated satellite data pipeline that replaces manual web-portal downloads with programmatic STAC API queries. Transform raw satellite imagery into **Analysis-Ready Data (ARD)** and use Machine Learning to fill spatial gaps in sparse weather station networks.

By the end of this chapter you will be able to:
- Query and download Sentinel-2, Landsat, and DEM data for any bounding box on Earth
- Apply atmospheric correction (COST model) to convert TOA to BOA reflectance
- Detect anomalous weather station records using IsolationForest
- Classify 30 years of precipitation into Drought / Normal / Flood regimes
- Map Urban Heat Islands from MODIS thermal data

---

## 🛠️ Scripts & Modules

### `01_stac_multisensor_download.py`
Downloads Sentinel-2 L2A, Landsat C2-L2, and Copernicus DEM via the Planetary Computer STAC API. Sorts results by cloud cover and downloads only the essential bands: **B02 (Blue), B03 (Green), B04 (Red), B08 (NIR), B11 (SWIR)**.

> [!WARNING]
> **L1C vs L2A — The Double-Correction Trap**
> - **Level-1C (TOA):** Raw satellite data including atmospheric haze. Use if you want to practice atmospheric correction yourself.
> - **Level-2A (BOA):** Already corrected by ESA's Sen2Cor algorithm. **Never apply COST or FLAASH to L2A data — you will double-correct the atmosphere and produce scientifically invalid results.**
> - Planetary Computer's STAC archive defaults to L2A. This script downloads L2A.

> [!TIP]
> **For Atmospheric Correction Practice:** Manually download L1C data from the [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/), place bands in `data/raw/sentinel2_l1c_manual/`, then run `01b_envi_flaash_prep.py`.

**Key concepts:**
- STAC (SpatioTemporal Asset Catalog): modern REST standard for querying Earth Observation archives without downloading entire tiles
- Windowed reads: `rasterio` reads only the pixels inside your BBOX — no wasted bandwidth
- Band selection rationale: B02+B03+B04+B08+B11 covers 95% of vegetation, water, glacier, and atmospheric analysis

---

### `01b_envi_flaash_prep.py`
Converts TOA Reflectance to Radiance for ENVI FLAASH/QUAC processing.

> [!NOTE]
> **Why the conversion?** FLAASH runs MODTRAN (a full physical radiative transfer model) and requires absolute Radiance in µW/cm²/sr/nm. If you feed it Reflectance, MODTRAN interprets the Earth as pitch-black and crashes with the infamous `ACC_KTAEROSOL` error.

**Physics pipeline:**
$$L = \frac{\rho \cdot E_{sun} \cdot \cos(\theta_z)}{\pi \cdot d^2}$$

Where:
- $\rho$ = TOA reflectance (DN / 10000)
- $E_{sun}$ = Mean Solar Exoatmospheric Irradiance per band (W/m²/µm)
- $\theta_z$ = Solar Zenith Angle (fetched from STAC metadata)
- $d$ = Earth-Sun distance in Astronomical Units (computed from acquisition date)

Output: BIL-interleaved `.dat` file with ENVI `.hdr` containing wavelengths and FWHM — FLAASH reads this automatically.

**ENVI FLAASH settings for Torres del Paine:**
- Atmospheric Model: **Sub-Arctic Summer**
- Aerosol Retrieval: **2-Band (K-T)** — K-T Upper = Band 5 (SWIR), K-T Lower = Band 3 (Red)
- Radiance Scale Factor: **10** (the script multiplies Radiance × 10 to save as int16)

---

### `01c_landsat_download.py`
Downloads Landsat 8/9 Collection 2 Level-2 Surface Reflectance data.

> [!IMPORTANT]
> **Landsat C2-L2 Scale Factor:** Raw DNs must be converted before any index calculation:
> ```python
> SR = DN * 0.0000275 - 0.2   # Result in [0, 1] reflectance
> SR = np.clip(SR, 0, 1)       # Clamp to valid range
> ```
> Without this step, NDSI values are wrong by ~3 orders of magnitude and glacier thresholds are never reached.

---

### `02_atmospheric_correction.py`
Python implementation of the **COST Model (Chavez, 1996)** — an advanced Dark Object Subtraction method.

**Algorithm:**
1. Find the 1st percentile dark pixel value per band → this represents atmospheric path radiance (haze)
2. Subtract the dark pixel value from all pixels → removes additive haze
3. Query STAC for Solar Zenith Angle → compute transmittance: $T = \cos(\theta_z)$
4. Divide by transmittance → corrects for multiplicative atmospheric absorption

**Band exclusion rules:**
| Band | Action | Reason |
|------|--------|--------|
| B01–B08A | ✅ Apply COST | Visible/NIR strongly affected by Rayleigh/Mie scattering |
| B11, B12 (SWIR) | ⛔ Skip (copy raw) | Wavelength too large; atmospheric path radiance negligible |
| B09 (Water Vapor) | ⛔ Skip | Designed to measure atmospheric absorption — not surface |
| B10 (Cirrus) | ⛔ Skip | Fully absorbed; no surface reflectance to retrieve |

---

### `03a_fetch_real_weather_data.py`
Fetches 25 virtual weather station records from the [Open-Meteo](https://open-meteo.com) ERA5-Land reanalysis API (no API key required). Injects 3 known anomalies at random positions for ML detection training.

> [!NOTE]
> `np.random.seed(42)` must be set before anomaly injection for reproducible results across runs. Without a seed, the corrupted stations change each execution, undermining the pedagogical value.

---

### `03_station_ml_interpolation.py`
Two-stage climate data pipeline:

**Stage 1 — Anomaly Detection (IsolationForest):**
IsolationForest works by randomly partitioning the feature space. Anomalous points (corrupted station records like -9999 or 99°C spikes) are isolated in fewer splits than normal points.

**Stage 2 — Spatial Interpolation (Random Forest):**
Uses `latitude`, `longitude`, and `elevation` as features to predict temperature across a DEM grid — replicating the physical lapse rate relationship without hard-coding it.

> [!TIP]
> Standard k-fold cross-validation overestimates accuracy for spatial data due to spatial autocorrelation. Use leave-one-out or spatial block CV for real research.

---

### `04_precipitation_anomaly.py`
Downloads 30 years (1993–2023) of daily ERA5 precipitation from Open-Meteo. Resamples to annual totals, then applies **two complementary** anomaly detection methods:

| Method | What it detects | Output |
|--------|----------------|--------|
| K-Means (k=3) | Unsupervised clusters: Drought / Normal / Flood | Cluster label per year |
| Z-Score | Statistical distance from mean: $z = (x - \mu) / \sigma$ | Continuous anomaly score |

> [!NOTE]
> K-Means with only 30 data points is sensitive to initialization. `n_init=10` runs 10 random seeds and picks the best. The 3-cluster choice is an assumption — comment it explicitly in teaching contexts.

---

### `05_uhi_modis_mapping.py`
Maps Urban Heat Islands over Punta Arenas using MODIS MOD11A1 Land Surface Temperature.

> [!CAUTION]
> **MODIS Fill Value Bug — Physics Error:**
> The MOD11A1 fill value is **DN = 0**, and the valid DN range is **7500–43200**.
> ```python
> # ❌ WRONG — passes invalid sub-range pixels through
> lst_raw[lst_raw == 0] = np.nan
>
> # ✅ CORRECT — masks fill AND all sub-range DNs
> lst_raw[lst_raw < 7500] = np.nan
> ```
> Without this fix, cloud/shadow pixels with DN 1–7499 pass through the Kelvin conversion, producing temperatures near −123°C that corrupt the heat island gradient.

**Physics conversion:**
```python
MODIS_LST_SCALE = 0.02  # K per DN (from MOD11A1 User Guide)
lst_kelvin  = lst_raw * MODIS_LST_SCALE
lst_celsius = lst_kelvin - 273.15
```

---

## 📐 Key Formulas

| Concept | Formula |
|---------|---------|
| Radiance from Reflectance | $L = \frac{\rho \cdot E_{sun} \cdot \cos\theta_z}{\pi d^2}$ |
| COST transmittance | $T = \cos(\theta_z)$ |
| Landsat C2-L2 SR | $\rho = DN \times 0.0000275 - 0.2$ |
| MODIS LST (K) | $T_K = DN \times 0.02$ |
| Z-Score anomaly | $z = (x - \bar{x}) / \sigma$ |

---

## 🚀 How to Run

### Install Dependencies
```bash
mamba activate geocascade_env
mamba install -n geocascade_env -c conda-forge \
    pystac-client planetary-computer rasterio requests \
    scikit-learn pandas matplotlib numpy pyproj -y
```

### Execute Scripts in Order
```bash
# 1. Download Sentinel-2 L2A + Copernicus DEM
python 01_stac_multisensor_download.py

# 2. Download Landsat 8/9 C2-L2
python 01c_landsat_download.py

# 3. (Optional) Prepare ENVI FLAASH input from L1C data
python 01b_envi_flaash_prep.py

# 4. (Optional) Apply COST atmospheric correction
python 02_atmospheric_correction.py

# 5. Fetch ERA5 weather stations
python 03a_fetch_real_weather_data.py

# 6. Anomaly detection + interpolation
python 03_station_ml_interpolation.py

# 7. 30-year precipitation analysis
python 04_precipitation_anomaly.py

# 8. Urban Heat Island mapping
python 05_uhi_modis_mapping.py
```

---

## 🗺️ GIS Interoperability (ArcGIS Pro & ENVI)

Every script exports geocoded `.tif` outputs to `data/processed/`:

| Script | Output TIF | Load in |
|--------|-----------|---------|
| `01` | `sentinel2_stack.tif` | ArcGIS Pro, ENVI, QGIS |
| `02` | `cost_corrected_*.tif` | ENVI (compare with FLAASH output) |
| `03` | `temp_interpolation.tif` | ArcGIS Pro Spatial Analyst |
| `05` | `uhi_celsius.tif` | ArcGIS Pro → 3D visualization |

**Workflow:** Run script → open ArcGIS Pro → drag `.tif` into map → verify alignment with basemap imagery.
