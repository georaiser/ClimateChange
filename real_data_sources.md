# 🌍 Real Data Sources for GeoCascade Pipeline
**Study Area: Torres del Paine / Patagonia, Chile (~51°S, ~73°W)**

---

## 1. 🌡️ Real Weather Stations

### Option A — Open-Meteo (Already Integrated, No Registration)
The **easiest starting point** — already used in scripts 03a/04.
Returns ERA5-Land reanalysis at your exact lat/lon as "virtual stations."

```python
import requests, pandas as pd

url = (
    "https://archive-api.open-meteo.com/v1/archive"
    "?latitude=-51.0&longitude=-73.0"
    "&start_date=1993-01-01&end_date=2024-12-31"
    "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,"
    "windspeed_10m_max,relative_humidity_2m_mean"
    "&timezone=America/Santiago"
)
df = pd.DataFrame(requests.get(url).json()["daily"])
df.to_csv("real_era5_daily.csv", index=False)
```

> Best for: 30-year climate series, no gaps, free, no login.

---

### Option B — NOAA GHCN-Daily (Real Physical Stations)
**URL:** https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily

Real ground stations worldwide. Patagonia has sparse coverage but some exist.

**Direct CSV download by station ID:**
```python
import pandas as pd

# Punta Arenas WMO Station: USC00085512 / GHCND:UYM00085512
# Find nearest: https://www.ncei.noaa.gov/cdo-web/datatools/findstation
station_id = "USC00085512"
url = f"https://www.ncdc.noaa.gov/cdo-web/api/v2/data?datasetid=GHCND&stationid=GHCND:{station_id}&startdate=2000-01-01&enddate=2024-12-31&limit=1000&units=metric"
# Requires free token: https://www.ncei.noaa.gov/cdo-web/token
headers = {"token": "YOUR_FREE_TOKEN_HERE"}
r = requests.get(url, headers=headers)
```

**Or download bulk CSV directly (no code needed):**
https://www.ncei.noaa.gov/cdo-web/search → Search "Punta Arenas"

> Best for: ground-truth validation of ERA5 interpolation.

---

### Option C — Chile DMC (Dirección Meteorológica de Chile)
**Official Chilean national weather service**
**URL:** https://climatologia.meteochile.gob.cl/

Variables: Temperature, precipitation, wind, humidity, pressure
Stations: Puerto Natales, Punta Arenas, Balmaceda (nearest to Torres del Paine)

Download: Manual CSV download from the web portal (free, registration required)
```
Puerto Natales station code: 330020
Balmaceda: 310021
Punta Arenas: 300023
```

---

### Option D — Argentina SMN (Nearest cross-border stations)
**URL:** https://www.smn.gob.ar/descarga-de-datos
Station: Río Turbio (closest Argentinian station to the study area)

---

## 2. 🌧️ Precipitation Data

### Option A — CHIRPS (Climate Hazards Group InfraRed Precipitation)
**Best precipitation dataset for South America**
- Resolution: 0.05° (~5.5km), 1981–present, daily
- Combines satellite IR + rain gauge
- **URL:** https://www.chc.ucsb.edu/data/chirps

```python
# Download via UCSB server (no registration)
import requests, os
year, month = 2023, 6
url = f"https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/tifs/p05/{year}/chirps-v2.0.{year}.{month:02d}.01.tif.gz"
r = requests.get(url, stream=True)
with open("chirps_2023_06_01.tif.gz", "wb") as f:
    f.write(r.content)
```

> Best for: 40-year precipitation grids, great for Patagonia.

---

### Option B — GPM IMERG (NASA Global Precipitation Measurement)
- Resolution: 0.1° (~11km), 2000–present, 30-minute to monthly
- **URL:** https://gpm.nasa.gov/data/imerg
- Requires NASA EarthData account (free): https://urs.earthdata.nasa.gov/

```python
# Via earthaccess Python library
import earthaccess
earthaccess.login()
results = earthaccess.search_data(
    short_name="GPM_3IMERGDF",  # Daily final run
    bounding_box=(-73.5, -51.5, -72.5, -50.5),
    temporal=("2023-01-01", "2023-12-31")
)
earthaccess.download(results, "./data/gpm/")
```
Install: `pip install earthaccess`

---

### Option C — ERA5 Precipitation (Already in scripts)
Already integrated in scripts 03a, 04 via Open-Meteo. No extra download needed.

---

## 3. 🛰️ Satellite Imagery

### Option A — Microsoft Planetary Computer (Already Integrated ✅)
All scripts use this. Collections available:
- `sentinel-2-l2a` — 10m optical, 2017–present
- `sentinel-1-rtc` — 10m SAR, 2014–present
- `cop-dem-glo-30` — 30m DEM
- `landsat-c2-l2` — 30m, 1982–present
- `modis-11a1-061` — 1km LST, 2000–present

**No registration needed.** Already working in all your scripts.

---

### Option B — ESA Copernicus Data Space (Official Source)
**URL:** https://dataspace.copernicus.eu/
- Sentinel-2 L1C (TOA) and L2A (BOA, corrected)
- Sentinel-1 GRD and SLC
- Free, registration required

```python
# Using sentinelsat (download full scenes)
# pip install sentinelsat
from sentinelsat import SentinelAPI

api = SentinelAPI("user", "password", "https://apihub.copernicus.eu/apihub")
products = api.query(
    area="POLYGON((-73.5 -51.5,-72.5 -51.5,-72.5 -50.5,-73.5 -50.5,-73.5 -51.5))",
    date=("2023-01-01", "2023-03-31"),
    platformname="Sentinel-2",
    cloudcoverpercentage=(0, 20)
)
api.download_all(products)
```

> Best for: full scenes when you need offline access.

---

### Option C — USGS EarthExplorer (Landsat)
**URL:** https://earthexplorer.usgs.gov/
- Landsat 5, 7, 8, 9 Collection 2 Level-2 (Surface Reflectance, already calibrated)
- Free, registration required

```python
# Using landsatxplore Python library
# pip install landsatxplore
from landsatxplore.api import API
from landsatxplore.earthexplorer import EarthExplorer

api = API("username", "password")
scenes = api.search(
    dataset="landsat_ot_c2_l2",  # Landsat 8/9 C2-L2
    latitude=-51.0, longitude=-73.0,
    start_date="2000-01-01", end_date="2005-12-31",
    max_cloud_cover=20
)
ee = EarthExplorer("username", "password")
for scene in scenes[:3]:
    ee.download(scene["display_id"], output_dir="./data/landsat/")
```

> Best for: long time-series 1982–present for glacier change.

---

## 4. 🏔️ Digital Elevation Models (DEM)

### Option A — Copernicus DEM GLO-30 (Already Integrated ✅)
Already in all your scripts via Planetary Computer.
Native resolution: 30m | CRS: EPSG:4326

### Option B — SRTM 30m (NASA, 2000 snapshot)
**URL:** https://dwtkns.com/srtm30m/ (tile selector map)
Or via OpenTopography:
```python
# No registration needed for SRTM via OpenTopography API
url = (
    "https://portal.opentopography.org/API/globaldem"
    "?demtype=SRTMGL1&south=-51.5&north=-50.5&west=-73.5&east=-72.5"
    "&outputFormat=GTiff&API_Key=YOUR_FREE_KEY"
)
# Free key: https://opentopography.org/developers
```

### Option C — ALOS World 3D (AW3D30, 30m, best accuracy)
**URL:** https://www.eorc.jaxa.jp/ALOS/en/dataset/aw3d30/aw3d30_e.htm
Registration required (free). Best vertical accuracy of the 30m DEMs.

### Option D — TanDEM-X (12m, highest resolution available globally)
**URL:** https://geoservice.dlr.de/web/maps/eoc:tdm90
- 90m free, 12m requires application (scientific use approved quickly)

---

## 5. 🧊 Glaciology Data

### NSIDC — National Snow and Ice Data Center
**URL:** https://nsidc.org/data/explore-data

Key datasets for Patagonia:
| Dataset | Description | URL |
|---|---|---|
| HMA Glacier Inventory | Glacier outlines for South America | https://nsidc.org/data/nsidc-0272 |
| Randolph Glacier Inventory (RGI 7.0) | Global glacier outlines (shapefiles) | https://www.glims.org/RGI/ |
| NSIDC VELMAP | Glacier surface velocity | https://nsidc.org/data/velmap |

```python
# Download RGI glacier outlines for Patagonia (Region 17 = Low Latitudes)
# Free, no registration
import geopandas as gpd
url = "https://www.glims.org/RGI/rgi70_files/17_rgi70_LowLatitudes.zip"
gdf = gpd.read_file(url)
# Filter to Torres del Paine area
patagonia = gdf.cx[-74:-72, -52:-50]
patagonia.to_file("patagonia_glaciers.gpkg")
```

---

## 6. 🌊 Oceanography & Sea Surface Temperature

### NOAA CoastWatch (SST, Chlorophyll)
**URL:** https://coastwatch.pfeg.noaa.gov/erddap/

```python
import requests
# NOAA Optimum Interpolation SST v2.1 (daily, 0.25 deg)
url = (
    "https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21Agg_LonPM180.csv"
    "?sst[(2023-01-01):1:(2023-12-31)][(0.0):1:(0.0)][(-51.5):1:(-50.5)][(-73.5):1:(-72.5)]"
)
df = pd.read_csv(url, skiprows=1)
```

---

## 7. 🌿 Land Cover & Vegetation

### ESA WorldCover 10m (2020, 2021)
**URL:** https://esa-worldcover.org/en
Free 10m global land cover, 11 classes (Tree, Shrub, Grassland, etc.)

```python
# Via Planetary Computer (already integrated)
search = catalog.search(collections=["esa-worldcover"], bbox=BBOX)
```

### MODIS MOD13A1 — NDVI/EVI 500m time series (2000–present)
Already accessible via Planetary Computer:
```python
search = catalog.search(collections=["modis-13a1-061"], bbox=BBOX, datetime="2023-01")
```

---

## 8. 📡 Air Quality & Atmospheric

### Copernicus Atmosphere Monitoring Service (CAMS)
**URL:** https://atmosphere.copernicus.eu/
- Aerosol optical depth, dust, fire radiative power
- Free with ECMWF account

### Sentinel-5P TROPOMI (Air quality, 5.5km)
Via Planetary Computer:
```python
search = catalog.search(
    collections=["sentinel-5p-l2-no2"],  # Also: co, o3, ch4, aerai-ai
    bbox=BBOX, datetime="2023-01"
)
```

---

## 9. 🔑 Free Account Sign-ups Required

| Service | URL | What You Get |
|---|---|---|
| NASA EarthData | https://urs.earthdata.nasa.gov/ | GPM, MODIS bulk, SRTM |
| ESA Copernicus | https://dataspace.copernicus.eu/ | Full Sentinel scenes |
| USGS EarthExplorer | https://earthexplorer.usgs.gov/ | Landsat bulk download |
| NOAA CDO | https://www.ncei.noaa.gov/cdo-web/token | GHCN station data |
| OpenTopography | https://opentopography.org/developers | SRTM/ALOS via API |
| Copernicus CDS | https://cds.climate.copernicus.eu/ | ERA5 bulk GRIB/NetCDF |

---

## 10. 🗂️ Recommended Data for Your GeoCascade Scripts

| Script | Replace Synthetic With | Source |
|---|---|---|
| `03a_fetch_real_weather_data.py` | Real GHCN stations near 51°S | NOAA CDO API |
| `04_precipitation_anomaly.py` | CHIRPS 5km daily grids | UCSB CHIRPS |
| `03b_era5_trend_analysis.py` | ERA5-Land via CDS (bulk NetCDF) | Copernicus CDS |
| `09_glacier_retreat.py` | RGI 7.0 glacier outlines as validation | NSIDC GLIMS |
| `15_zonal_statistics.py` | Real administrative polygons | Natural Earth / GADM |

---

## 11. 🐍 Install All Data Access Libraries

```bash
mamba install -n geocascade_env -c conda-forge \
  earthaccess sentinelsat landsatxplore \
  cdsapi geopandas requests pandas -y

# For CHIRPS / GPM bulk download
pip install earthaccess
```

---

> [!TIP]
> **Best starting point:** Run `03b_era5_trend_analysis.py` as-is — it already pulls
> real 30-year ERA5 data from Open-Meteo with zero registration. Then graduate to
> CHIRPS for spatial precipitation grids and GHCN for ground-truth station validation.
