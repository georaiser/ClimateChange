# Chapter 8: Multi-Sensor Data Fusion & Risk Modeling

## Academic Objective
Fuse data from 4 completely different sensors (Sentinel-2, Sentinel-1, CopDEM, MODIS)
onto a single 10m spatial grid using rasterio.warp.reproject, then train a Random Forest
classifier on the 4-band data cube for land cover classification.

---

## Scripts

### 20_multisensor_data_fusion.py — 4-Sensor Data Cube

Fuses 4 bands from different sensors onto a single 10m master grid (S2 NIR as base):

| Band | Sensor | Native Res | Physical Layer |
|---|---|---|---|
| 1 | Sentinel-2 B08 | 10m | NIR reflectance |
| 2 | Sentinel-1 RTC | 10m | SAR VV backscatter (dB) |
| 3 | Copernicus DEM | 30m | Elevation (m) |
| 4 | MODIS MOD11A1 | 1km | Land Surface Temperature (Celsius) |

All bands reprojected to S2 UTM grid via rasterio.warp.reproject (bilinear).

> [!WARNING]
> nodata must be -9999, NOT np.nan. Writing np.nan to integer GDAL rasters
> produces silent corruption. The _to_nodata() helper cleans NaN/Inf before writing.

**Improvements applied:**
- nodata=-9999 with _to_nodata() cleanup helper
- MODIS STAC guard: fills thermal band with NoData if no items found
- window dims int(round()) for correct TIFF profile
- Enhanced band descriptions in GeoTIFF metadata (QGIS/ArcGIS readable)
- NEW Tier 3: fusion statistics summary (mean/std/min/max per band)

Run: `python 20_multisensor_data_fusion.py`

Outputs: cascade_master_stack.tif (4-band, ready for ML)

---

### 21_cascade_risk_modeling.py — Random Forest Land Cover Classifier

Trains a Random Forest classifier on the 4-band fusion stack to classify terrain:
Glacier / Water / Vegetation / Rock / Bare Soil

Outputs probability maps showing model confidence per class.

Run: `python 21_cascade_risk_modeling.py`

Outputs: risk_probability_map.tif, risk_classification.png

---

### 22_combined_insights_engine.py — Multi-Sensor Convergence Analysis

Combines optical, SAR, DEM, and climate signals to compute:
- ESI (Environmental Stress Index)
- CVS (Cascade Vulnerability Score)
- WSI (Water Stress Index)

Produces a final convergent evidence map showing areas where multiple sensors
simultaneously indicate high environmental stress.

Run: `python 22_combined_insights_engine.py`

---

## Key Concepts

| Concept | Explanation |
|---|---|
| Data Fusion | Aligning sensors at different resolutions onto one common grid |
| rasterio.warp.reproject | Handles CRS transformation + resampling in one step |
| nodata=-9999 vs np.nan | GDAL reads nodata from profile; np.nan is silently mishandled |
| Band descriptions | set_band_description() makes TIFFs self-documenting in GIS tools |
| Fusion statistics | Validate each band is non-degenerate before passing to ML |
| Random Forest | Pixel vector [NIR, SAR, DEM, LST] -> land cover class probability |

## Installation

```bash
mamba install -n geocascade_env -c conda-forge pystac-client planetary-computer rasterio numpy matplotlib pyproj scikit-learn -y
```