# 🌍 Chapter 1: Climate Data Acquisition & Preprocessing

> **GeoCascade** · Climate Change Geospatial Analysis Pipeline  
> *From raw satellite archives to analysis-ready geospatial datasets*

---

## 📋 Overview

This chapter establishes the **data foundation** for the entire GeoCascade pipeline. You will learn to acquire multi-source climate datasets — satellite imagery, station records, and glacier inventories — using modern cloud-native APIs (STAC, Open-Meteo, CHIRPS), apply preprocessing and atmospheric correction, detect sensor anomalies with machine learning, and perform rigorous trend analysis using non-parametric statistics.

**Study region:** Patagonia, Chile/Argentina — one of the most climatically extreme and data-sparse regions on Earth, ideal for showcasing real-world geospatial challenges.

---

## 🗂️ Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RAW DATA SOURCES                             │
│                                                                     │
│  Open-Meteo API   CHIRPS UCSB    Planetary Computer    GLIMS/RGI    │
│  (ERA5-Land)      (GeoTIFFs)     (STAC: S2, DEM, L8)  (Glaciers)    │
└────────┬──────────────┬──────────────────┬────────────────┬─────────┘
         │              │                  │                │
         ▼              ▼                  ▼                ▼
┌────────────────┐ ┌──────────┐  ┌─────────────────┐ ┌──────────────┐
│ 00_real_data_  │ │ 03c_     │  │ 01_stac_multi   │ │ 00_real_data │
│ downloader.py  │ │ chirps_  │  │ sensor_download │ │ _downloader  │
│ (MASTER)       │ │ spatial_ │  │ .py             │ │ .py (RGI)    │
└───────┬────────┘ │ precip.  │  └────────┬────────┘ └──────┬───────┘
        │          │ py       │           │                  │
        ▼          └────┬─────┘           ▼                  ▼
┌────────────────┐      │       ┌─────────────────┐ ┌──────────────┐
│ ERA5 time      │      │       │ 02_atmospheric_ │ │ Glacier      │
│ series (11 var)│      │       │ correction.py   │ │ outlines     │
│ Station data   │      │       │ (DOS1 / L2A)    │ │ (GeoJSON)    │
└───────┬────────┘      │       └────────┬────────┘ └──────────────┘
        │               │                │
        ▼               ▼                ▼
┌───────────────────────────────────────────────────────────────────┐
│                   INTERMEDIATE PRODUCTS                           │
│  03_station_ml_interpolation.py  ←  03a_fetch_real_weather_data   │
│  (IsolationForest anomaly + RF temperature surface)               │
└───────────────────────────────────────────────────────────────────┘
        │               │
        ▼               ▼
┌────────────────┐ ┌──────────────────────┐ ┌───────────────────────┐
│ 03b_era5_trend │ │ 04_precipitation_    │ │ 05_uhi_modis_mapping  │
│ _analysis.py   │ │ anomaly.py           │ │ .py                   │
│ (Mann-Kendall) │ │ (K-Means clustering) │ │ (MODIS LST / UHI)     │
└───────┬────────┘ └──────────┬───────────┘ └───────────────────────┘
        │                     │
        ▼                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     data/processed/real_data/                       │
│  climatology GeoTIFF · anomaly maps · trend figures · RMSE metrics  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Installation

```bash
mamba install -n geocascade_env -c conda-forge \
    pystac-client planetary-computer rasterio pyproj \
    numpy pandas matplotlib scikit-learn requests geopandas -y
```

> [!IMPORTANT]
> Always activate the environment before running any script:
> ```bash
> conda activate geocascade_env      / or mamba
> ```

> [!NOTE]
> **No login required** for Open-Meteo or CHIRPS downloads. Planetary Computer access is anonymous by default; an API token is only needed for higher rate limits. NOAA GHCN works without a token but will fall back to Open-Meteo if none is provided.

---

## 🚀 Scripts — Run Order

### `00_real_data_downloader.py` — 🆕 MASTER DOWNLOADER

python Chapter_01/00_real_data_downloader.py 

> [!IMPORTANT]
> **Run this first.** This single script downloads all real-world data required by the downstream scripts in one pass.

Downloads and organises:

| Dataset | Source | Auth |
|---|---|---|
| ERA5-Land daily time series (11 variables, 1993–2024) | Open-Meteo API | ✅ None |
| CHIRPS v2.0 monthly precipitation GeoTIFFs | UCSB server | ✅ None |
| NOAA GHCN station metadata + observations | NOAA / Open-Meteo fallback | ⚠️ Optional token |
| RGI 7.0 glacier outlines — Patagonia (~50 MB zip) | GLIMS server | ✅ None |
| Natural Earth admin boundaries — Chile/Argentina | Natural Earth | ✅ None |

Outputs a summary dashboard to `data/processed/real_data/`.

---

### `01_stac_multisensor_download.py` — STAC API Intro

Downloads **Sentinel-2 L2A** spectral bands and **Copernicus DEM** tiles from Microsoft Planetary Computer using the STAC API.

> [!NOTE]
> **STAC (SpatioTemporal Asset Catalog)** is the modern open standard for cloud-native geospatial data discovery. Instead of manual downloads, you query a catalog with spatial + temporal filters and receive signed asset URLs — the foundation of cloud-native GIS.

---

### `01b_envi_flaash_prep.py` — ENVI FLAASH Metadata Prep

Prepares Landsat metadata files (`.hdr`, solar geometry parameters) for **FLAASH atmospheric correction** in ENVI. Run before `02_atmospheric_correction.py` if working with L1C data destined for FLAASH.

---

### `01c_landsat_download.py` — Landsat C2-L2 Download

Downloads **Landsat Collection 2 Level-2** (surface reflectance) scenes via Planetary Computer STAC for the Patagonia AOI.

---

### `02_atmospheric_correction.py` — Atmospheric Correction

Applies **DOS1 (Dark Object Subtraction)** to L1C TOA data and compares results against pre-corrected L2A BOA reflectance.

> [!WARNING]
> The script includes a mismatch guard — if L1C and L2A scenes are not co-registered (same tile, same date), the comparison is invalid. Check the log for `L1C/L2A MISMATCH` warnings before interpreting spectral difference plots.

---

### `03_station_ml_interpolation.py` — ML Spatial Interpolation

Downloads synthetic weather station data (normally sourced from GHCN-Daily), applies **IsolationForest** anomaly detection, then trains a **Random Forest** regressor to spatially interpolate temperature across Patagonia.

**Outputs:**
- Anomaly map (flagged stations highlighted)
- Temperature surface GeoTIFF
- RMSE metrics (cross-validated)

---

### `03a_fetch_real_weather_data.py` — 🔄 UPDATED Real Station Data

Fetches data from **7 real/virtual stations** covering the full Patagonian climate gradient:

| Station | Type | Elevation | Climate Zone |
|---|---|---|---|
| Punta Arenas | Real | 37 m | Coastal subpolar |
| Puerto Natales | Real | 8 m | Transition |
| Grey Glacier | Virtual | 1 800 m | Alpine |
| Balmaceda | Real | 520 m | Continental (rain shadow) |
| + 3 additional | Mixed | variable | Gradient fill |

Uses **ERA5 via Open-Meteo**. Injects realistic sensor anomalies (freeze, dropout, drift) for ML detection training. Produces a station comparison plot demonstrating the **Andes precipitation gradient**.

> [!TIP]
> The synthetic anomaly injection (freeze → constant value, dropout → NaN runs, drift → slow linear offset) mimics real datalogger failure modes. This makes the IsolationForest training in `03_station_ml_interpolation.py` directly applicable to real QA/QC workflows.

---

### `03b_era5_trend_analysis.py` — 🆕 Mann-Kendall + Sen's Slope

Performs rigorous non-parametric trend analysis on the **30-year ERA5 temperature series**.

> [!NOTE]
> Uses **pure `scipy`** — no external `pymannkendall` dependency required.

#### 📐 Statistical Methods

**Mann-Kendall Monotonic Trend Test**

For a time series $x_1, x_2, \ldots, x_n$, the test statistic $S$ is:

$$S = \sum_{k=1}^{n-1} \sum_{j=k+1}^{n} \text{sgn}(x_j - x_k)$$

where $\text{sgn}(\theta) = +1, 0, -1$ for $\theta > 0, = 0, < 0$ respectively.

Under $H_0$ (no trend), $S$ is approximately normal with:

$$\text{Var}(S) = \frac{n(n-1)(2n+5) - \sum_t t(t-1)(2t+5)}{18}$$

The standardised statistic $Z = S / \sqrt{\text{Var}(S)}$ is compared to $z_{\alpha/2}$.

**Sen's Slope Estimator**

$$\hat{\beta} = \text{median}\left(\frac{x_j - x_k}{j - k}\right) \quad \forall\, j > k$$

Sen's slope is the **median of all $\binom{n}{2}$ pairwise slopes** — resistant to outliers and non-normality, unlike OLS.

**Outputs:** trend direction, slope per decade (°C/10 yr), p-value, and trend visualisation figure.

---

### `03c_chirps_spatial_precipitation.py` — 🆕 CHIRPS Precipitation Climatology

Downloads **CHIRPS v2.0** monthly GeoTIFFs directly from the UCSB server (no login required) and computes:

| Product | Description |
|---|---|
| 30-year climatology grid | Mean monthly precipitation (mm) |
| Interannual variability | Standard deviation per pixel |
| Anomaly Z-scores | Per-year departure from climatology |
| W–E precipitation transect | Patagonian rain shadow cross-section |

> [!NOTE]
> **CHIRPS** (Climate Hazards Group InfraRed Precipitation with Station data) combines satellite thermal IR (RAINS-ANN algorithm) with rain gauge observations to produce a 5.5 km resolution daily/monthly product from 1981–present.

Saves: climatology GeoTIFF + 6-panel analysis figure.

---

### `04_precipitation_anomaly.py` — Anomaly Clustering

Fetches ERA5 precipitation series, computes **monthly anomalies** relative to the 1993–2023 climatological baseline, and applies **K-Means clustering** to identify spatially coherent anomaly regimes.

**Outputs:** anomaly map + cluster membership chart.

---

### `05_uhi_modis_mapping.py` — MODIS Urban Heat Island

Downloads **MODIS MOD11A1** Land Surface Temperature product and maps the **Urban Heat Island** signal.

> [!CAUTION]
> **MODIS LST fill-value handling is critical.** The fill value threshold is DN < 7500 — **not** DN = 0. Pixels with DN values in the range 1–7499 are invalid and must be masked before scaling.
>
> Correct scaling pipeline:
> ```python
> lst_valid = np.where(lst_dn >= 7500, lst_dn, np.nan)
> lst_kelvin = lst_valid * 0.02
> lst_celsius = lst_kelvin - 273.15
> ```

---

## 🧪 Key Academic Concepts

### 🛰️ STAC — Cloud-Native Data Access
**SpatioTemporal Asset Catalog** is the modern open standard for discovering and accessing geospatial data in the cloud. Instead of bulk FTP downloads, STAC lets you query a catalog with spatial + temporal predicates and receive signed, on-demand asset URLs — enabling scalable, reproducible pipelines without local data hoarding.

### 🌡️ ERA5-Land
ECMWF's **9 km global reanalysis** product, spanning 1940–present. ERA5-Land applies a land-surface downscaling to the native ERA5 (~31 km) grid, correcting for orographic effects. Available at any lat/lon via the **Open-Meteo API** — free, no registration, returns any variable as a JSON time series.

### 🌧️ CHIRPS v2.0
**Climate Hazards Group InfraRed Precipitation with Station data.** Combines CCD (Cold Cloud Duration) satellite thermal IR observations with quality-controlled rain gauge records using an interpolation scheme based on the RAINS-ANN (Rainfall Estimation from Remotely Sensed Information using Artificial Neural Networks) algorithm. Resolution: 0.05° (~5.5 km), 1981–present.

### 🗂️ Open-Meteo
A **free, open-access ERA5/CERRA API** that exposes reanalysis variables (temperature, precipitation, wind, soil moisture, snow depth, etc.) at any lat/lon coordinate from 1993–present. No API key or registration required. Returns JSON with optional unit conversion and timezone localisation.

### 🧊 RGI 7.0 — Randolph Glacier Inventory
The **authoritative global glacier polygon dataset**, updated in 2023. RGI 7.0 maps ~275,000 glaciers worldwide with attributes including area, elevation, slope, and aspect. For Patagonia, it covers the Patagonian Ice Fields (HPS/HPN) — among the largest temperate glaciers outside the polar regions.

### 📈 Mann-Kendall Test
A **non-parametric monotonic trend test** that does not assume normality or linearity of the underlying series. Standard in climate science for detecting trends in temperature, precipitation, and streamflow. The null hypothesis $H_0$ is "no monotonic trend"; rejection indicates a statistically significant upward or downward trend.

### 📏 Sen's Slope
A **robust linear trend estimator** computed as the median of all pairwise slopes between data points. Unlike Ordinary Least Squares (OLS), Sen's slope is highly resistant to outliers and asymmetric distributions — particularly important for precipitation series that often include extreme events.

### 🏔️ Patagonian Precipitation Gradient
One of the steepest precipitation gradients on Earth: **>3 000 mm/yr** on the windward (Pacific) western slopes vs. **<300 mm/yr** on the leeward (Argentine) side — across only ~100 km. Driven by persistent westerly airflow forced to rise over the Andes, leading to intense orographic precipitation on the Pacific flank and a pronounced rain shadow to the east.

### 🌲 IsolationForest — Anomaly Detection
A **tree-based unsupervised anomaly detection** algorithm. Data points are isolated by randomly selecting a feature and a split value; anomalies (rare, extreme observations) require fewer splits to isolate and thus receive lower anomaly scores. No labelled training data required — ideal for QA/QC of station sensor records.

---

## 📁 Output Directory Structure

```
data/
└── processed/
    └── real_data/
        ├── era5_timeseries_*.csv          # ERA5-Land daily variables
        ├── chirps_climatology.tif         # 30-year mean precipitation GeoTIFF
        ├── chirps_analysis_6panel.png     # Climatology + variability + transect
        ├── station_anomalies.png          # IsolationForest anomaly map
        ├── temperature_surface.tif        # RF-interpolated temperature grid
        ├── era5_trend_analysis.png        # Mann-Kendall + Sen's slope figure
        ├── precipitation_anomaly_map.png  # Monthly anomaly clustering
        ├── modis_lst_uhi.png              # Urban Heat Island map
        └── summary_dashboard.png          # Master downloader summary
```

---

## 📚 References

- Hersbach, H. et al. (2020). The ERA5 global reanalysis. *QJRMS*, 146(730), 1999–2049.
- Funk, C. et al. (2015). The climate hazards infrared precipitation with stations — a new environmental record for monitoring extremes. *Scientific Data*, 2, 150066.
- Mann, H.B. (1945). Nonparametric tests against trend. *Econometrica*, 13(3), 245–259.
- Sen, P.K. (1968). Estimates of the regression coefficient based on Kendall's tau. *JASA*, 63(324), 1379–1389.
- RGI Consortium (2023). Randolph Glacier Inventory – A Dataset of Global Glacier Outlines, Version 7.0. *NSIDC*.
- Liu, F.T., Ting, K.M., & Zhou, Z-H. (2008). Isolation Forest. *ICDM 2008*.

---

*GeoCascade · Chapter 1 of N · Last updated 2026-07-09*
