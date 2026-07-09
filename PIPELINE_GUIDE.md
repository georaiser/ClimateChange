# GeoCascade -- Complete Pipeline Run Guide

## Recommended Run Order

### Phase 1 -- Download All Real Data (no satellite, no login)
`bash
conda activate geocascade_env
python Chapter_01/00_real_data_downloader.py
python Chapter_01/03a_fetch_real_weather_data.py
python Chapter_01/03c_chirps_spatial_precipitation.py
python Chapter_01/03b_era5_trend_analysis.py
`
Expected time: 15-30 min (CHIRPS download is the largest step)

### Phase 2 -- Satellite Processing (Planetary Computer, no login)
`bash
python Chapter_02/06_spectral_signature_analysis.py
python Chapter_02/07_vegetation_soil_indices.py
python Chapter_02/08_automated_index_batcher.py
python Chapter_03/09_multitemporal_glacier_retreat.py
python Chapter_03/10_digital_elevation_processing.py
python Chapter_03/11_watershed_delineation.py
python Chapter_04/12_ecological_niche_modeling.py
python Chapter_04/13_climate_vulnerability_index.py
python Chapter_05/14_moisture_stress_indices.py
python Chapter_05/15_zonal_statistics.py
python Chapter_06/16_isohyets_isotherms.py
python Chapter_06/17_drainage_density.py
python Chapter_07/18_sentinel1_sar_processing.py
python Chapter_07/19_multisensor_review.py
python Chapter_07/19b_cloud_penetration_comparison.py
`

### Phase 3 -- Machine Learning & Data Fusion
`bash
python Chapter_08/20_multisensor_data_fusion.py
python Chapter_08/21_cascade_risk_modeling.py
`

### Phase 4 -- Convergence (Final Integration)
`bash
python Chapter_08/23_real_data_convergence.py
python Chapter_08/22_combined_insights_engine.py
python Chapter_12/capstone_pipeline.py
`

---

## Key Outputs per Phase

| Phase | Key Files |
|-------|-----------|
| 1 Real Data | era5_daily_patagonia.csv, chirps_monthly/*.tif, station_data_real.csv, rgi70_patagonia_glaciers.gpkg |
| 2 Satellite | spectral_indices_all.png, batch_index_timeseries.png, batch_indices/NDVI_*.tif |
| 3 ML | cascade_master_stack.tif, cascade_ml_prediction.tif, glacier_probability_map.tif |
| 4 Convergence | convergence_dashboard.png, environmental_stress_index.tif |

---

## Data Sources -- Zero Login Required

| Source | Data | URL |
|--------|------|-----|
| Open-Meteo | ERA5-Land daily 1993-2024, 11 variables | archive-api.open-meteo.com |
| UCSB CHIRPS | Precipitation 5.5km monthly 1981-present | data.chc.ucsb.edu/products/CHIRPS-2.0 |
| Planetary Computer | Sentinel-2/1, DEM, Landsat, MODIS | planetarycomputer.microsoft.com |
| GLIMS | RGI 7.0 glacier outlines | glims.org/RGI/ |

## Optional Free Registration

| Service | What You Gain |
|---------|---------------|
| NOAA CDO -- ncei.noaa.gov/cdo-web/token | Real GHCN ground station CSVs |
| NASA EarthData -- urs.earthdata.nasa.gov | GPM precipitation, MODIS bulk |
| Copernicus CDS -- cds.climate.copernicus.eu | ERA5 full bulk NetCDF |
| USGS EarthExplorer -- earthexplorer.usgs.gov | Landsat scenes 1982-present |
| Chile DMC -- climatologia.meteochile.gob.cl | Official Chilean station data |

---

## Environment Setup (First Time Only)

`bash
mamba create -n geocascade_env python=3.11 -y
conda activate geocascade_env
mamba install -n geocascade_env -c conda-forge \\
  pystac-client planetary-computer rasterio pysheds \\
  geopandas rasterstats shapely numpy pandas matplotlib \\
  scikit-learn pyproj requests osmnx -y
`

---

## Critical Bugs Fixed (All Scripts)

| Bug | Scripts | Fix |
|-----|---------|-----|
| src.crs outside with block | Ch06: 16, 17 | Capture inside rasterio.open() |
| B11 (20m) using 10m window | Ch02: 07, 08; Ch05: 14 | Independent win_b11 from 20m transform |
| OSMnx positional bbox args | Ch12: capstone | Named kwargs: north= south= east= west= |
| nodata=np.nan or 0 | All chapters | Standardized to nodata=-9999 |
| plt.close() missing | All chapters | Added after every savefig() |
| EPSG:3857 distortion at 51S | Ch12 | Changed to EPSG:32719 (UTM 19S) |
| window.height as float | Multiple | int(round(window.height)) |
| MODIS fill DN < 7500 | Ch01, Ch07 | arr[arr < 7500] = nodata |
