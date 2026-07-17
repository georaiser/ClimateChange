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
| Sentinel-2 L1C | TOA Radiance | NOT corrected | Run `03_atmospheric_correction.py` (COST) or ENVI FLAASH |
| Landsat 9 L1TP | TOA Radiance | NOT corrected | Run `03_atmospheric_correction.py` (COST) or ENVI FLAASH |
| Copernicus DEM | Elevation model | Not applicable (not optical) | **Use directly** |
| ERA5-Land | Climate reanalysis | Not applicable | **Use directly** |
| CHIRPS v2.0 | Precipitation product | Not applicable | **Use directly** |

> [!NOTE]
> The data downloaded by `01_data_download.py` and `02_satellite_acquisition.py`
> is **already at Level-2 (surface reflectance)**. The COST correction script is
> provided for when you download raw **L1C/L1TP** data manually.
> It auto-bridges its output so all downstream scripts work without changes.

---

## 🛰️ L1C Workflow — If You Have Raw Satellite Images

If you downloaded **Sentinel-2 L1C** from [Copernicus Browser](https://browser.dataspace.copernicus.eu)
or **Landsat L1TP** from [EarthExplorer](https://earthexplorer.usgs.gov), follow this workflow:

### Step 1 — Place the band files

```
Chapter_01/data/raw/
└── sentinel2_l1c_{scene_id}/     ← create this folder
    ├── B01.tif
    ├── B02.tif
    ├── B03.tif
    ├── B04.tif
    ├── B05.tif
    ├── B06.tif
    ├── B07.tif
    ├── B08.tif
    ├── B8A.tif
    ├── B09.tif
    ├── B10.tif
    ├── B11.tif
    └── B12.tif
```

For Landsat L1TP use folder name `landsat_l1tp_{scene_id}/`.

### Step 2 — Run the correction script

```bash
python Chapter_01/03_atmospheric_correction.py
```

The script will:
1. **Auto-detect** the newest `sentinel2_l1c_*` or `landsat*l1tp*` folder
2. **Fetch the Solar Zenith Angle** from Planetary Computer STAC metadata (fallback: 45°)
3. **Apply COST correction** (Chavez 1996) to all VNIR bands — haze subtract + transmittance
4. **Copy SWIR bands** (B11, B12) unchanged — haze is negligible at >1400 nm
5. **Auto-bridge** the corrected bands to the pipeline folder:

```
data/processed/boa_corrected/{scene}/BOA_B04.tif  ← COST output
                                  ↓  (auto-copied, BOA_ prefix stripped)
data/raw/sentinel2_l2a_from_l1c_cost/B04.tif      ← Ch02–Ch13 find this
```

### Step 3 — Continue normally

```bash
# No changes needed in any other script
python Chapter_02/06_spectral_signature_analysis.py   # auto-detects sentinel2_l2a_*
python Chapter_02/07_vegetation_soil_indices.py
# ... all subsequent chapters work unchanged
```

> [!NOTE]
> **COST vs FLAASH accuracy:** COST is ±10–15% reflectance accuracy, adequate for
> vegetation indices and glacier mapping. For publication-grade results use
> ENVI FLAASH (`envi/01_flaash_correction.pro`) which runs full MODTRAN physics.
> FLAASH output should also be copied to `data/raw/sentinel2_l2a_from_l1c_flaash/`.

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
    |   |   |-- ghcn_stations_patagonia.csv      (5 real + 2 virtual stations)
    |   |   |-- chirps_monthly/                  (294+ TIFs, 2000-2024)
    |   |   |-- rgi70_patagonia_glaciers.gpkg    (glacier outlines)
    |   |   |-- admin_boundaries.gpkg            (Chile/Argentina)
    |   |-- sentinel2_l2a_*/                     (default: STAC download, B02-B12)
    |   |-- sentinel2_l1c_*/                     (optional: your L1C raw data)
    |   |-- sentinel2_l2a_from_l1c_cost/         (auto-created by 03_ if L1C found)
    |   |-- landsat_*/                           (SR_B2-B7, ST_B10 + metadata)
    |   |-- dem_*/                               (Copernicus DEM 30m)
    |-- processed/
        |-- boa_corrected/{scene}/BOA_*.tif      (COST output, float32 [0-1])
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
#   - ERA5 climate, CHIRPS precipitation, GHCN stations, RGI glacier outlines
#   - NOAA token from .env used for real GHCND data; Open-Meteo fallback automatic
python Chapter_01/01_data_download.py

# Step 2: Download satellite imagery via STAC (run once -- skip-if-exists)
#   - Sentinel-2 L2A, Landsat 9 L2SP, Copernicus DEM via Planetary Computer
python Chapter_01/02_satellite_acquisition.py

# Step 3: Atmospheric correction
#   - DEFAULT (no L1C): DEMO mode -- shows COST physics vs L2A product
#   - WITH L1C: place bands in data/raw/sentinel2_l1c_{scene}/ first
#     Script auto-corrects and bridges output to sentinel2_l2a_from_l1c_cost/
#     so Chapter 2+ scripts find it automatically -- no further steps needed
python Chapter_01/03_atmospheric_correction.py

# Step 4: 32-year climate trend analysis (Mann-Kendall + Sen's Slope)
python Chapter_01/04_climate_trend_analysis.py

# Step 5: CHIRPS precipitation spatial analysis
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
[Open-Meteo / NOAA CDO]  [CHIRPS UCSB]  [Planetary Computer STAC]  [GLIMS/RGI]
          |                    |                   |                      |
          v                    v                   v                      v
  01_data_download.py    01_data_download.py  02_satellite_acq.py  01_data_download.py
          |                    |                   |
          v                    v                   v
  ERA5 + GHCN stations   CHIRPS 294 TIFs    S2 L2A + Landsat C2L2 + CopDEM
          |                    |                   |
          |                    v                   v
          |            05_chirps_precip.py   03_atmospheric_correction.py
          |                    |               ┌── DEMO mode (L2A exists):
          v                    |               │   comparison figure only
  04_climate_trend.py          |               │
          |                    |               └── L1C mode (your raw data):
          v                    |                   COST correction applied
  06_station_interp.py         |                   → BOA_*.tif produced
          |                    |                   → auto-bridge copies to
          v                    v                     sentinel2_l2a_from_l1c_cost/
  07_precip_anomaly.py --> 08_uhi_mapping.py --> 09_chapter_report.py
                                                          |
                                         [ArcGIS Pro: professional maps]

  sentinel2_l2a_*  (any of: l2a_downloaded / l2a_from_l1c_cost / l2a_from_l1c_flaash)
          |
          v
  Chapter 2+ scripts  (auto-detected via sentinel2_l2a_* glob -- no config needed)
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
| `01` | `ghcn_stations_patagonia.csv` | XY Table to Points | IDL READ_CSV |
| `01` | `chirps_monthly/*.tif` | Add Raster | Open as multifile |
| `02` | `sentinel2_l2a_*/B*.tif` | Composite Bands | Open in ENVI |
| `02` | `landsat_*/SR_B*.tif` | Composite Bands | Open in ENVI |
| `03` | `boa_corrected/BOA_B04.tif` | Add Raster (Stretched) | Open in ENVI |
| `03` | `sentinel2_l2a_from_l1c_cost/` | Composite Bands | Open in ENVI |
| `03` | `correction_comparison.png` | N/A (reference figure) | N/A |
| `03` | `correction_report.csv` | Table (haze + T values) | N/A |
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

*GeoCascade | Chapter 1 of 13 | Last updated 2026-07-16*
*Python: geocascade_env | ENVI: 5.6 | ArcGIS Pro: 3.x*
