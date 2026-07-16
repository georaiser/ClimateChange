# 🌍 Chapter 1: Climate Data Acquisition & Foundation

> **GeoCascade** · Climate Change Geospatial Analysis Pipeline
> *From raw satellite archives to analysis-ready geospatial datasets*
> **Study region:** Torres del Paine National Park, Patagonia, Chile (51°S, 73°W)

---

## 🧭 Three-Track Learning Framework

Chapter 1 is taught three ways simultaneously. Each track uses the same real data but different tools:

| Track | Tool | What you learn | Scripts |
|---|---|---|---|
| 🐍 **Python** | `geocascade_env` | Automation, APIs, ML, statistics | `01_` to `09_` |
| 🔬 **ENVI 5.6** | ENVI + IDL | Atmospheric correction, spectral analysis | `envi/` folder |
| 🗺️ **ArcGIS Pro** | arcpy + GUI | Cartography, spatial analysis, layouts | `arcgis_pro/` folder |

**Data flows between tracks:**
```
Python (downloads + analysis) --> ENVI (spectral refinement) --> ArcGIS Pro (professional maps)
```

---

## 🔑 Atmospheric Correction Decision Table

> [!IMPORTANT]
> Before processing any satellite imagery, check whether it is already corrected.
> **Applying correction twice will corrupt your reflectance values.**

| Dataset | Level | Correction Status | Action |
|---|---|---|---|
| Sentinel-2 L2A | BOA Surface Reflectance | Already corrected by ESA Sen2Cor | **Use directly** |
| Landsat 9 L2SP | SR + ST Products | Already corrected by USGS LaSRC | **Use directly** |
| Sentinel-2 L1C | TOA Radiance | NOT corrected | Apply FLAASH (ENVI) or Sen2Cor |
| Landsat 9 L1TP | TOA Radiance | NOT corrected | Apply FLAASH (ENVI) or DOS1 (Python) |
| Copernicus DEM | Elevation model | Not applicable (not optical) | **Use directly** |
| ERA5-Land | Climate reanalysis | Not applicable | **Use directly** |
| CHIRPS v2.0 | Precipitation product | Not applicable | **Use directly** |

> [!NOTE]
> The data downloaded by `01_data_download.py` and `02_satellite_acquisition.py`
> is **already at Level-2 (surface reflectance)**. The ENVI FLAASH workflow in
> `envi/` teaches the correction process for when you download raw L1C/L1TP data.

---

## 📁 Directory Structure

```
Chapter_01/
|
|-- Python Track (run in order) --------------------
|   |-- 01_data_download.py          ERA5 + CHIRPS + GHCN + RGI + admin boundaries
|   |-- 02_satellite_acquisition.py  Sentinel-2 L2A + Landsat 9 L2SP + CopDEM (STAC)
|   |-- 03_atmospheric_correction.py DOS1 demo + FLAASH vs L2A comparison
|   |-- 04_climate_trend_analysis.py Mann-Kendall + Sen's slope (all ERA5 variables)
|   |-- 05_chirps_precipitation.py   CHIRPS spatial analysis (384 TIFs on disk)
|   |-- 06_station_interpolation.py  IsolationForest + Random Forest interpolation
|   |-- 07_precipitation_anomaly.py  K-Means anomaly regime clustering
|   |-- 08_uhi_mapping.py            MODIS LST Urban Heat Island
|   |-- 09_chapter_report.py         Summary dashboard + narrative report
|
|-- ENVI Track ------------------------------------
|   |-- envi/
|       |-- README_ENVI.md           ENVI 5.6 workflow guide
|       |-- 01_flaash_correction.pro IDL batch FLAASH script (run from ENVI)
|       |-- 01_flaash_correction.py  Python ENVI API script (run from ENVI console)
|       |-- 02_spectral_analysis.pro IDL spectral profiles + ROI export
|       |-- 03_export_arcgis.pro     IDL export to ArcGIS Pro-ready GeoTIFF
|
|-- ArcGIS Pro Track ------------------------------
|   |-- arcgis_pro/
|       |-- README_ARCGISPRO.md      ArcGIS Pro workflow guide
|       |-- 01_arcpy_import_imagery.py  Mosaic DEM + composite S2 + import layers
|       |-- 02_arcpy_climate_maps.py    Thematic map symbology automation
|       |-- 03_arcpy_layout_export.py   Professional 4-panel layout + PDF/PNG export
|
|-- data/ (DO NOT MODIFY - shared across all tracks)
    |-- raw/
    |   |-- real_data/
    |   |   |-- era5_daily_patagonia.csv        (11,688 days x 10 variables)
    |   |   |-- era5_monthly_patagonia.csv       (384 months)
    |   |   |-- ghcn_stations_patagonia.csv      (7 stations x 5,000+ records)
    |   |   |-- chirps_monthly/                  (384 TIFs, 21 GB, 2000-2024)
    |   |   |-- rgi70_patagonia_glaciers.gpkg    (glacier outlines)
    |   |   |-- admin_boundaries.gpkg            (Chile/Argentina)
    |   |-- sentinel2_l2a_*/                     (B02,B03,B04,B08,B11 + metadata)
    |   |-- landsat_*/                           (SR_B2-B7, ST_B10 + metadata)
    |   |-- dem_*/                               (Copernicus DEM 30m, 4 tiles)
    |-- processed/
        |-- climate_analysis/                    (trend CSVs, station outputs)
        |-- climate_maps/                        (anomaly PNGs)
        |-- real_data/                           (CHIRPS TIF, dashboard PNG)
        |-- uhi_mapping/                         (MODIS LST TIFs + PNG)
        |-- report_dashboard.png
        |-- report_climate_story.md
        |-- report_executive_summary.md
```

---

## ⚙️ Installation

### Python track
```bash
mamba install -n geocascade_env -c conda-forge ^
    pystac-client planetary-computer rasterio pyproj ^
    numpy pandas matplotlib scikit-learn requests geopandas -y
conda activate geocascade_env
```

### ENVI track
- Requires ENVI 5.6 (licensed)
- IDL scripts: ENVI Toolbox > Run IDL Script
- Python API: ENVI Tools > Python Console
- See `envi/README_ENVI.md` for full setup

### ArcGIS Pro track
- Requires ArcGIS Pro 3.x (licensed)
- arcpy scripts: Analysis > Python Notebook (inside ArcGIS Pro)
- See `arcgis_pro/README_ARCGISPRO.md` for full setup

---

## 🚀 Python Track: Run Order

```bash
conda activate geocascade_env

# Step 1: Download all real-world data (run once -- resumable)
python Chapter_01/01_data_download.py

# Step 2: Download satellite imagery via STAC (run once -- skip-if-exists)
python Chapter_01/02_satellite_acquisition.py

# Step 3: Atmospheric correction demo (DOS1 vs L2A comparison)
python Chapter_01/03_atmospheric_correction.py

# Step 4: 32-year climate trend analysis (Mann-Kendall, all ERA5 variables)
python Chapter_01/04_climate_trend_analysis.py

# Step 5: CHIRPS precipitation spatial analysis (uses 384 TIFs on disk)
python Chapter_01/05_chirps_precipitation.py

# Step 6: ML weather station anomaly detection + spatial interpolation
python Chapter_01/06_station_interpolation.py

# Step 7: Precipitation anomaly clustering (K-Means, 4 regimes)
python Chapter_01/07_precipitation_anomaly.py

# Step 8: MODIS Urban Heat Island mapping
python Chapter_01/08_uhi_mapping.py

# Step 9: Chapter summary dashboard + narrative reports
python Chapter_01/09_chapter_report.py
```

---

## 🗂️ Data Flow

```
[Open-Meteo API]    [CHIRPS UCSB]   [Planetary Computer STAC]  [GLIMS/RGI]
       |                  |                    |                     |
       v                  v                    v                     v
 01_data_download.py  01_data_download.py  02_satellite_acq.py  01_data_download.py
       |                  |                    |
       v                  v                    v
 ERA5 daily/monthly   CHIRPS 384 TIFs      S2 L2A + Landsat C2L2 + CopDEM
       |                  |                    |
       |                  v                    v
       |          05_chirps_precip.py   03_atmospheric_correction.py
       |                  |                 (comparison only -- data already corrected)
       v                  |                    |
 04_climate_trend.py      |              [ENVI track: FLAASH for L1C/L1TP]
       |                  |
       v                  v
 06_station_interp.py  [Precip TIFs + CSVs]
       |
       v
 07_precip_anomaly.py --> 08_uhi_mapping.py --> 09_chapter_report.py
                                                       |
                                      [ArcGIS Pro track: professional maps]
```

---

## 🧪 Key Academic Concepts

### Atmospheric Correction Methods

**FLAASH** (Fast Line-of-sight Atmospheric Analysis of Spectral Hypercubes)
- Physics-based radiative transfer model (based on MODTRAN)
- Best accuracy, requires scene geometry + atmospheric profile
- Available in ENVI 5.x (see `envi/01_flaash_correction.pro`)

**DOS1** (Dark Object Subtraction)
- Empirical method: minimum DN in each band assumed to be "zero reflectance"
- Fast, no auxiliary data required
- Less accurate than FLAASH, but workable for vegetation indices
- Script `03_atmospheric_correction.py` demonstrates DOS1

**Pre-corrected L2 products**
- ESA Sen2Cor (Sentinel-2 L2A) and USGS LaSRC (Landsat C2 L2SP) apply
  physics-based correction before distribution -- this is what we download by default

### ERA5-Land
ECMWF 9 km global reanalysis, 1940-present. Accessed free via Open-Meteo API (no key required).

### CHIRPS v2.0
Climate Hazards Group InfraRed Precipitation + Station data. 0.05 deg resolution, 1981-present.
Combines satellite thermal IR with rain gauge interpolation (RAINS-ANN algorithm).

### Mann-Kendall Trend Test
Non-parametric test for monotonic trends. Null hypothesis: no trend.
$$S = \sum_{k=1}^{n-1} \sum_{j=k+1}^{n} \text{sgn}(x_j - x_k)$$

Standardised statistic $Z = S / \sqrt{\text{Var}(S)}$ compared to $z_{\alpha/2}$.

### Sen's Slope
Robust trend estimator: median of all pairwise slopes.
$$\hat{\beta} = \text{median}\left(\frac{x_j - x_k}{j - k}\right) \quad \forall\, j > k$$

Resistant to outliers unlike OLS regression.

### Patagonian Precipitation Gradient
>3,000 mm/year on windward Pacific slopes vs <300 mm/year on leeward Argentine side
across only ~100 km -- one of Earth's steepest precipitation gradients.
Driven by persistent westerly airflow forced over the Andes.

### IsolationForest -- Sensor QA/QC
Tree-based unsupervised anomaly detection. Anomalous records (sensor freeze, dropout, drift)
require fewer tree splits to isolate and receive lower anomaly scores.
No labelled data required -- ideal for station QA/QC.

---

## 📁 Expected Outputs After Full Run

| Script | Key Output | ArcGIS Pro | ENVI |
|---|---|---|---|
| `01` | `era5_daily_patagonia.csv` | Table join | N/A |
| `01` | `chirps_monthly/*.tif` | Add Raster | Open as multifile |
| `02` | `sentinel2_l2a_*/B*.tif` | Composite Bands | Open in ENVI |
| `02` | `landsat_*/SR_B*.tif` | Composite Bands | Open in ENVI |
| `03` | `dos1_corrected_ndvi.tif` | Add Raster | Open in ENVI |
| `04` | `trend_summary.csv` | Table > Chart | N/A |
| `05` | `chirps_mean_annual_precip.tif` | Add Raster (Classified) | Open in ENVI |
| `06` | `temperature_surface.tif` | Add Raster (Stretched) | Open in ENVI |
| `07` | `precipitation_clusters.csv` | Table > Chart | N/A |
| `08` | `uhi_celsius.tif` | Add Raster (Red-Yellow) | Open in ENVI |
| `09` | `report_dashboard.png` | N/A (for reports) | N/A |

---

## 📚 References

- Hersbach et al. (2020). ERA5 global reanalysis. *QJRMS*, 146(730), 1999-2049.
- Funk et al. (2015). CHIRPS. *Scientific Data*, 2, 150066.
- Mann (1945). Nonparametric tests against trend. *Econometrica*, 13(3), 245-259.
- Sen (1968). Regression coefficient estimates based on Kendall's tau. *JASA*, 63(324).
- RGI Consortium (2023). Randolph Glacier Inventory v7.0. *NSIDC*.
- Liu, Ting & Zhou (2008). Isolation Forest. *ICDM 2008*.
- Berk et al. (2008). MODTRAN 5: 2006 update. *SPIE Proc.* 6233. [FLAASH basis]
- USGS (2023). Landsat Collection 2 Level-2 Science Product Guide.
- ESA (2021). Sen2Cor Configuration and User Manual. *ESA SNAP*.

---

*GeoCascade | Chapter 1 of 14 | Last updated 2026-07-15*
*Python: geocascade_env | ENVI: 5.6 | ArcGIS Pro: 3.x*
