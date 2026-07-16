"""
GeoCascade Chapter 01 - Script 01: Climate & Geospatial Data Downloader
========================================================================
Downloads all baseline datasets required for the Torres del Paine
climate-change analysis chapter:

  1. ERA5-Land daily climate (Open-Meteo archive API, 1993-2024)
     10 variables: temp_max/min/mean, precip, wind_speed_10m_max,
     wind_gusts_10m_max, relative_humidity_2m_mean, snowfall_sum,
     et0_fao_evapotranspiration, shortwave_radiation_sum
     NOTE: surface_pressure is NOT available in the daily archive endpoint.

  2. CHIRPS monthly precipitation GeoTIFFs (UCSB server, 2000-2024)
     Skip-if-exists to avoid redundant downloads (~300 files).

  3. NOAA GHCN station records for 7 Patagonia stations via NOAA API
     (requires NOAA_TOKEN in ../.env), with Open-Meteo fallback if
     token is absent or API fails.

  4. RGI 7.0 glacier outlines (GLIMS server, region 17 - Southern Andes)
     Filtered to study bbox [-75, -53, -71, -49].

  5. Natural Earth admin-1 boundaries (3-method fallback):
       a) geodatasets package
       b) direct Natural Earth GeoJSON URL
       c) legacy geopandas.datasets (deprecated)

  6. Generates a 6-panel summary dashboard PNG.

OUTPUTS
-------
  data/raw/real_data/era5_daily_patagonia.csv
  data/raw/real_data/era5_monthly_patagonia.csv
  data/raw/real_data/chirps_monthly/chirps-v2.0.YEAR.MM.tif
  data/raw/real_data/ghcn_stations_patagonia.csv
  data/raw/real_data/rgi70_patagonia_glaciers.gpkg
  data/raw/real_data/admin_boundaries.gpkg
  data/processed/real_data/real_data_summary_dashboard.png

ENV
---
  Loads ../.env manually (no python-dotenv required).
  Expects NOAA_TOKEN=<your_token> for GHCN downloads.

Author : GeoCascade Project
"""

import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import os
import io
import gzip
import json
import time
import shutil
import requests
import warnings
import traceback
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from datetime import datetime, date
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import box

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).parent
DATA_ROOT    = SCRIPT_DIR / "data"
RAW_DIR      = DATA_ROOT / "raw" / "real_data"
CHIRPS_DIR   = RAW_DIR / "chirps_monthly"
PROC_DIR     = DATA_ROOT / "processed" / "real_data"
ENV_FILE     = SCRIPT_DIR.parent / ".env"

for d in [RAW_DIR, CHIRPS_DIR, PROC_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Study area
# ---------------------------------------------------------------------------
BBOX          = (-73.5, -51.5, -72.5, -50.5)   # min_lon, min_lat, max_lon, max_lat
CENTER_LAT    = -51.0
CENTER_LON    = -73.0
GLACIER_BBOX  = (-75.0, -53.0, -71.0, -49.0)   # broader RGI filter

# ---------------------------------------------------------------------------
# ERA5 variables (surface_pressure is INVALID in daily archive endpoint)
# ---------------------------------------------------------------------------
ERA5_DAILY_VARS = ",".join([
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "relative_humidity_2m_mean",
    "snowfall_sum",
    "et0_fao_evapotranspiration",
    "shortwave_radiation_sum",
])

# ---------------------------------------------------------------------------
# Stations  (lat, lon, name, ghcn_id)
# GHCND IDs verified via:
#   GET /cdo-web/api/v2/stations?datasetid=GHCND&extent=-55,-75,-44,-68
# CHM-prefix IDs do not exist in GHCND — real Chilean stations use CI prefix.
# ---------------------------------------------------------------------------
STATIONS = [
    # Real GHCND stations — CI = Chile, AR = Argentina
    (-53.000, -70.967, "Punta_Arenas",              "CI000085934"),   # GHCND:CI000085934
    (-45.917, -71.700, "Balmaceda",                  "CI000085874"),   # GHCND:CI000085874
    (-45.594, -72.106, "Teniente_Vidal_Coyhaique",  "CIM00085864"),   # GHCND:CIM00085864
    (-50.267, -72.050, "El_Calafate",               "ARM00087904"),   # GHCND:ARM00087904  (closest to study area)
    (-51.617, -69.283, "Rio_Gallegos",              "AR000087925"),   # GHCND:AR000087925
    # Virtual stations — no GHCND data, use Open-Meteo ERA5 reanalysis
    (-51.050, -73.100, "Grey_Glacier_virtual",       None),
    (-50.940, -72.990, "Torres_del_Paine_virtual",   None),
]


OPEN_METEO_STATION_URL = (
    "https://archive-api.open-meteo.com/v1/archive"
    "?latitude={lat}&longitude={lon}"
    "&start_date=2010-01-01&end_date=2023-12-31"
    "&daily=temperature_2m_max,temperature_2m_min,"
    "precipitation_sum,wind_speed_10m_max"
    "&timezone=America%2FSantiago"
)

# ---------------------------------------------------------------------------
# Helper: load .env manually
# ---------------------------------------------------------------------------
def load_env(path: Path) -> dict:
    env = {}
    if not path.exists():
        print(f"[WARN] .env not found at {path}")
        return env
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            env[key.strip()] = val.strip().strip('"').strip("'")
    print(f"[OK] Loaded .env from {path}  ({len(env)} keys)")
    return env


# ---------------------------------------------------------------------------
# Helper: retry GET
# ---------------------------------------------------------------------------
def get_with_retry(url, params=None, headers=None,
                   retries=3, timeout=60, stream=False):
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, headers=headers,
                             timeout=timeout, stream=stream)
            r.raise_for_status()
            return r
        except requests.RequestException as exc:
            print(f"  [RETRY {attempt}/{retries}] {exc}")
            if attempt < retries:
                time.sleep(3 * attempt)
    raise RuntimeError(f"Failed after {retries} attempts: {url}")


# ===========================================================================
# 1. ERA5-Land via Open-Meteo
# ===========================================================================
def download_era5(env):
    out_daily   = RAW_DIR / "era5_daily_patagonia.csv"
    out_monthly = RAW_DIR / "era5_monthly_patagonia.csv"

    if out_daily.exists() and out_monthly.exists():
        print(f"[SKIP] ERA5 files already exist")
        return pd.read_csv(out_daily, parse_dates=["date"])

    print("\n[1/5] Downloading ERA5-Land daily via Open-Meteo archive ...")
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={CENTER_LAT}&longitude={CENTER_LON}"
        "&start_date=1993-01-01&end_date=2024-12-31"
        f"&daily={ERA5_DAILY_VARS}"
        "&timezone=America%2FSantiago"
    )
    print(f"  URL: {url[:100]}...")
    resp = get_with_retry(url, timeout=120)
    data = resp.json()

    daily_raw = data.get("daily", {})
    if not daily_raw:
        raise RuntimeError("Open-Meteo returned empty 'daily' block")

    df = pd.DataFrame(daily_raw)
    df.rename(columns={"time": "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    df.to_csv(out_daily, index=False, encoding="utf-8")
    print(f"  [OK] Daily CSV  -> {out_daily}  ({len(df)} rows)")

    # Monthly aggregation
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month
    grp = df.groupby(["year", "month"])
    monthly = grp.agg(
        temp_max_mean   = ("temperature_2m_max",           "mean"),
        temp_min_mean   = ("temperature_2m_min",           "mean"),
        temp_mean       = ("temperature_2m_mean",          "mean"),
        precip_sum      = ("precipitation_sum",            "sum"),
        wind_max        = ("wind_speed_10m_max",           "mean"),
        wind_gusts_max  = ("wind_gusts_10m_max",           "mean"),
        rh_mean         = ("relative_humidity_2m_mean",    "mean"),
        snowfall_sum    = ("snowfall_sum",                 "sum"),
        et0_sum         = ("et0_fao_evapotranspiration",   "sum"),
        radiation_sum   = ("shortwave_radiation_sum",      "sum"),
    ).reset_index()
    monthly.to_csv(out_monthly, index=False, encoding="utf-8")
    print(f"  [OK] Monthly CSV -> {out_monthly}  ({len(monthly)} rows)")
    return df


# ===========================================================================
# 2. CHIRPS monthly GeoTIFFs
# ===========================================================================
def download_chirps():
    print("\n[2/5] Downloading CHIRPS monthly GeoTIFFs (2000-2024) ...")
    base_url = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/tifs"
    years    = range(2000, 2025)
    months   = range(1, 13)
    downloaded = 0
    skipped    = 0
    failed     = 0

    for yr in years:
        for mo in months:
            mm       = f"{mo:02d}"
            fname    = f"chirps-v2.0.{yr}.{mm}.tif"
            out_path = CHIRPS_DIR / fname
            gz_url   = f"{base_url}/chirps-v2.0.{yr}.{mm}.tif.gz"

            # Skip months beyond available data
            if yr == 2024 and mo > 6:
                continue

            if out_path.exists() and out_path.stat().st_size > 10_000:
                skipped += 1
                continue

            try:
                print(f"  Fetching {fname} ...", end=" ", flush=True)
                resp = get_with_retry(gz_url, timeout=120, stream=True)
                gz_data = resp.content
                with gzip.open(io.BytesIO(gz_data)) as gz_f:
                    tif_data = gz_f.read()
                with open(out_path, "wb") as fh:
                    fh.write(tif_data)
                sz_kb = out_path.stat().st_size // 1024
                print(f"{sz_kb} KB")
                downloaded += 1
                time.sleep(0.3)
            except Exception as exc:
                print(f"FAILED ({exc})")
                failed += 1

    print(f"  [OK] CHIRPS: {downloaded} downloaded, {skipped} skipped, {failed} failed")
    return downloaded + skipped


# ===========================================================================
# 3. GHCN stations (NOAA API + Open-Meteo fallback)
# ===========================================================================
def _fetch_ghcn_noaa(ghcn_id, token):
    """Fetch daily summaries from NOAA CDO API, paginating year-by-year.

    Key fixes vs original:
    - URL: ncei.noaa.gov (ncdc.noaa.gov deprecated 2024)
    - NO datatypeid filter: CDO returns empty if ANY requested type is
      absent at a station. Chilean CHM stations often lack AWND. Accept
      all types and filter client-side.
    - Offset pagination: CDO cap is 1000 rows per call. Loop with
      offset until all records retrieved.
    """
    url     = "https://www.ncei.noaa.gov/cdo-web/api/v2/data"
    headers = {"token": token}
    frames  = []

    for year in range(2010, 2024):
        offset = 1          # CDO uses 1-based offset
        while True:
            params = {
                "datasetid": "GHCND",
                "stationid": f"GHCND:{ghcn_id}",
                "startdate": f"{year}-01-01",
                "enddate":   f"{year}-12-31",
                "limit":     1000,      # CDO maximum per request
                "offset":    offset,
                "units":     "metric",
                # No datatypeid — accept all available types
            }
            try:
                resp    = get_with_retry(url, params=params,
                                         headers=headers, timeout=30)
                payload = resp.json()
                results = payload.get("results", [])
            except Exception as exc:
                print(f"    NOAA API error ({year} offset={offset}): {exc}")
                break       # skip this year-chunk, try next year

            if not results:
                break       # no more data for this year

            frames.append(pd.DataFrame(results))

            # CDO metadata tells us total count available
            total = payload.get("metadata", {}).get(
                "resultset", {}).get("count", 0)
            offset += len(results)
            if offset > total:
                break       # fetched everything

            time.sleep(0.25)

        time.sleep(0.5)     # pause between years

    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True)

    # Pivot wide: one row per date with TMAX/TMIN/PRCP columns
    if "datatype" in df.columns and "value" in df.columns:
        df = (df.pivot_table(index="date", columns="datatype",
                             values="value", aggfunc="first")
                .reset_index())
        df.columns.name = None

    return df



def _fetch_open_meteo_station(lat, lon, name):
    """Fetch station-point data from Open-Meteo archive."""
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date=2010-01-01&end_date=2023-12-31"
        f"&daily=temperature_2m_max,temperature_2m_min,"
        f"precipitation_sum,wind_speed_10m_max"
        f"&timezone=America%2FSantiago"
    )
    resp = get_with_retry(url, timeout=60)
    raw  = resp.json().get("daily", {})
    df   = pd.DataFrame(raw)
    df.rename(columns={"time": "date"}, inplace=True)
    df["station_name"] = name
    df["latitude"]     = lat
    df["longitude"]    = lon
    df["source"]       = "open_meteo_fallback"
    return df


def download_stations(env):
    out_path = RAW_DIR / "ghcn_stations_patagonia.csv"
    if out_path.exists():
        print(f"[SKIP] Station CSV already exists")
        return pd.read_csv(out_path)

    print("\n[3/5] Downloading station data ...")
    token  = env.get("NOAA_TOKEN", "")
    frames = []

    for lat, lon, name, ghcn_id in STATIONS:
        print(f"  Station: {name} ({lat}, {lon})")
        df = None

        if token and ghcn_id:
            print(f"    Trying NOAA GHCN ({ghcn_id}) ...", end=" ")
            df = _fetch_ghcn_noaa(ghcn_id, token)
            if df is not None and len(df) > 0:
                df["station_name"] = name
                df["latitude"]     = lat
                df["longitude"]    = lon
                df["source"]       = "noaa_ghcn"
                print(f"{len(df)} records")
            else:
                print("empty/failed")
                df = None

        if df is None:
            print(f"    Fallback -> Open-Meteo archive ...", end=" ")
            try:
                df = _fetch_open_meteo_station(lat, lon, name)
                print(f"{len(df)} records")
            except Exception as exc:
                print(f"FAILED ({exc})")
                df = None

        if df is not None:
            frames.append(df)
        time.sleep(0.5)

    if not frames:
        raise RuntimeError("No station data could be retrieved")

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(out_path, index=False, encoding="utf-8")
    print(f"  [OK] Stations CSV -> {out_path}  ({len(combined)} rows)")
    return combined


# ===========================================================================
# 4. RGI 7.0 glacier outlines
# ===========================================================================
def download_rgi():
    out_path = RAW_DIR / "rgi70_patagonia_glaciers.gpkg"
    if out_path.exists():
        print(f"[SKIP] RGI glacier outlines already exist")
        try:
            return gpd.read_file(out_path)
        except Exception:
            return None

    print("\n[4/5] Downloading RGI 7.0 glacier outlines ...")

    wfs_url = (
        "https://www.glims.org/geoserver/GLIMS/wfs"
        "?service=WFS&version=2.0.0&request=GetFeature"
        "&typeName=GLIMS:glims_polygons"
        "&CQL_FILTER=loc_rngi%3D%2717-02%27"
        "&outputFormat=application/json"
        "&count=5000"
    )

    gdf = None

    # Try WFS GeoJSON
    try:
        print("  Trying GLIMS WFS ...", end=" ", flush=True)
        resp = get_with_retry(wfs_url, timeout=90)
        gdf  = gpd.GeoDataFrame.from_features(
                    resp.json()["features"], crs="EPSG:4326"
               )
        print(f"{len(gdf)} features")
    except Exception as exc:
        print(f"FAILED ({exc})")
        gdf = None

    if gdf is None or len(gdf) == 0:
        # Create a minimal placeholder GeoDataFrame so the pipeline continues
        print("  [WARN] Could not retrieve RGI data; creating placeholder")
        bbox_geom = box(*GLACIER_BBOX)
        gdf = gpd.GeoDataFrame(
            {"glacier_id": ["placeholder"], "area_km2": [0.0],
             "geometry": [bbox_geom]},
            crs="EPSG:4326",
        )

    # Filter to Patagonia bounding box
    minx, miny, maxx, maxy = GLACIER_BBOX
    try:
        gdf = gdf[
            (gdf.geometry.centroid.x >= minx) &
            (gdf.geometry.centroid.x <= maxx) &
            (gdf.geometry.centroid.y >= miny) &
            (gdf.geometry.centroid.y <= maxy)
        ]
    except Exception:
        pass

    gdf.to_file(out_path, driver="GPKG")
    print(f"  [OK] RGI glaciers -> {out_path}  ({len(gdf)} features)")
    return gdf


# ===========================================================================
# 5. Natural Earth admin boundaries  (3-method fallback)
# ===========================================================================
def download_admin_boundaries():
    out_path = RAW_DIR / "admin_boundaries.gpkg"
    if out_path.exists():
        print(f"[SKIP] Admin boundaries already exist")
        try:
            return gpd.read_file(out_path)
        except Exception:
            return None

    print("\n[5/5] Downloading Natural Earth admin boundaries ...")
    gdf = None

    # Method A: geodatasets
    try:
        import geodatasets
        for key in ["naturalearth.admin_1_states_provinces",
                    "naturalearth.land"]:
            try:
                path = geodatasets.get_path(key)
                gdf  = gpd.read_file(path)
                print(f"  [OK] geodatasets: {key}  ({len(gdf)} features)")
                break
            except Exception:
                continue
    except ImportError:
        print("  geodatasets not installed -- trying URL method")
    except Exception as exc:
        print(f"  geodatasets failed: {exc}")

    # Method B: direct Natural Earth GeoJSON URL
    if gdf is None:
        ne_url = (
            "https://raw.githubusercontent.com/nvkelso/"
            "natural-earth-vector/master/geojson/"
            "ne_10m_admin_1_states_provinces.geojson"
        )
        try:
            print(f"  Trying Natural Earth URL ...", end=" ", flush=True)
            resp = get_with_retry(ne_url, timeout=60)
            gdf  = gpd.GeoDataFrame.from_features(
                        resp.json()["features"], crs="EPSG:4326"
                   )
            print(f"{len(gdf)} features")
        except Exception as exc:
            print(f"FAILED ({exc})")

    # Method C: legacy geopandas.datasets
    if gdf is None:
        try:
            print("  Trying legacy gpd.datasets ...", end=" ", flush=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                path = gpd.datasets.get_path("naturalearth_lowres")
            gdf = gpd.read_file(path)
            print(f"{len(gdf)} features (low-res fallback)")
        except Exception as exc:
            print(f"FAILED ({exc})")

    if gdf is None:
        print("  [WARN] All boundary methods failed; creating placeholder")
        bbox_geom = box(*BBOX)
        gdf = gpd.GeoDataFrame(
            {"name": ["Torres del Paine Study Area"], "geometry": [bbox_geom]},
            crs="EPSG:4326",
        )

    # Filter to South America / Patagonia region
    try:
        if "continent" in gdf.columns:
            gdf = gdf[gdf["continent"] == "South America"]
        elif "CONTINENT" in gdf.columns:
            gdf = gdf[gdf["CONTINENT"] == "South America"]
        pat_box = box(-76.0, -56.0, -68.0, -46.0)
        gdf = gdf[gdf.geometry.intersects(pat_box)]
    except Exception:
        pass

    gdf.to_file(out_path, driver="GPKG")
    print(f"  [OK] Admin boundaries -> {out_path}  ({len(gdf)} features)")
    return gdf


# ===========================================================================
# 6. Summary dashboard
# ===========================================================================
def make_dashboard(df_era5, df_stations, gdf_rgi, gdf_admin):
    out_path = PROC_DIR / "real_data_summary_dashboard.png"
    print("\n[DASHBOARD] Generating 6-panel summary ...")

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        "GeoCascade Chapter 01 -- Torres del Paine Climate Data Overview\n"
        "ERA5-Land (Open-Meteo) | CHIRPS | Station Records | RGI 7.0 Glaciers",
        fontsize=14, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)

    # ------ Panel 1: ERA5 annual mean temperature -------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    if df_era5 is not None and "date" in df_era5.columns:
        try:
            annual = df_era5.groupby(df_era5["date"].dt.year)["temperature_2m_mean"].mean()
            ax1.plot(annual.index, annual.values, "r-o", ms=3, lw=1.5)
            z = np.polyfit(annual.index, annual.values, 1)
            p = np.poly1d(z)
            ax1.plot(annual.index, p(annual.index), "k--", lw=1,
                     label=f"Trend {z[0]*10:.2f} deg C/decade")
            ax1.legend(fontsize=7)
        except Exception as exc:
            ax1.text(0.5, 0.5, f"Data error:\n{exc}", ha="center", va="center",
                     transform=ax1.transAxes, fontsize=7)
    ax1.set_title("ERA5 Annual Mean Temp (deg C)", fontsize=9)
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Temp (deg C)")

    # ------ Panel 2: ERA5 annual precipitation ----------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    if df_era5 is not None and "precipitation_sum" in df_era5.columns:
        try:
            ann_p = df_era5.groupby(df_era5["date"].dt.year)["precipitation_sum"].sum()
            ax2.bar(ann_p.index, ann_p.values, color="steelblue", alpha=0.75)
            z2 = np.polyfit(ann_p.index, ann_p.values, 1)
            p2 = np.poly1d(z2)
            ax2.plot(ann_p.index, p2(ann_p.index), "r--", lw=1.5)
        except Exception as exc:
            ax2.text(0.5, 0.5, f"Data error:\n{exc}", ha="center", va="center",
                     transform=ax2.transAxes, fontsize=7)
    ax2.set_title("ERA5 Annual Precipitation (mm)", fontsize=9)
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Precip (mm)")

    # ------ Panel 3: Monthly climatology ---------------------------------
    ax3 = fig.add_subplot(gs[0, 2])
    ax3b = ax3.twinx()
    if df_era5 is not None and "temperature_2m_mean" in df_era5.columns:
        try:
            clim   = df_era5.groupby(df_era5["date"].dt.month)
            t_clim = clim["temperature_2m_mean"].mean()
            p_clim = clim["precipitation_sum"].mean()
            ax3.bar(t_clim.index, p_clim.values, color="steelblue", alpha=0.5)
            ax3b.plot(t_clim.index, t_clim.values, "r-o", ms=4)
            ax3.set_xticks(range(1, 13))
            ax3.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"], fontsize=7)
        except Exception as exc:
            ax3.text(0.5, 0.5, f"Data error:\n{exc}", ha="center", va="center",
                     transform=ax3.transAxes, fontsize=7)
    ax3.set_title("Monthly Climatology (1993-2024)", fontsize=9)
    ax3.set_xlabel("Month")
    ax3.set_ylabel("Precip (mm)", color="steelblue")
    ax3b.set_ylabel("Temp (deg C)", color="red")

    # ------ Panel 4: Station data ----------------------------------------
    ax4 = fig.add_subplot(gs[1, 0])
    if df_stations is not None and "station_name" in df_stations.columns:
        try:
            names  = df_stations["station_name"].unique()[:7]
            counts = [len(df_stations[df_stations["station_name"] == n]) for n in names]
            short  = [n.replace("_", " ")[:15] for n in names]
            bars   = ax4.barh(short, counts, color="teal", alpha=0.8)
            ax4.bar_label(bars, fmt="%d", padding=3, fontsize=7)
        except Exception as exc:
            ax4.text(0.5, 0.5, f"Data error:\n{exc}", ha="center", va="center",
                     transform=ax4.transAxes, fontsize=7)
    ax4.set_title("Station Records Count", fontsize=9)
    ax4.set_xlabel("# Records")

    # ------ Panel 5: CHIRPS file count by year ---------------------------
    ax5 = fig.add_subplot(gs[1, 1])
    tif_files = list(CHIRPS_DIR.glob("chirps-v2.0.*.tif"))
    if tif_files:
        try:
            from collections import Counter
            years_found = []
            for f in tif_files:
                try:
                    yr = int(f.stem.split(".")[1])
                    years_found.append(yr)
                except Exception:
                    pass
            yr_counts  = Counter(years_found)
            yrs_sorted = sorted(yr_counts.keys())
            ax5.bar(yrs_sorted, [yr_counts[y] for y in yrs_sorted],
                    color="darkgreen", alpha=0.8)
        except Exception as exc:
            ax5.text(0.5, 0.5, f"Data error:\n{exc}", ha="center", va="center",
                     transform=ax5.transAxes, fontsize=7)
    ax5.set_title(f"CHIRPS Monthly Files ({len(tif_files)} total)", fontsize=9)
    ax5.set_xlabel("Year")
    ax5.set_ylabel("Files")

    # ------ Panel 6: Map of study area -----------------------------------
    ax6 = fig.add_subplot(gs[1, 2])
    try:
        if gdf_admin is not None and len(gdf_admin) > 0:
            gdf_admin.plot(ax=ax6, color="lightyellow", edgecolor="gray", lw=0.5)
        from matplotlib.patches import Rectangle
        bx = Rectangle(
            (BBOX[0], BBOX[1]),
            BBOX[2] - BBOX[0], BBOX[3] - BBOX[1],
            linewidth=2, edgecolor="red", facecolor="red", alpha=0.25,
        )
        ax6.add_patch(bx)
        ax6.set_xlim(-76, -68)
        ax6.set_ylim(-56, -46)
        if gdf_rgi is not None and len(gdf_rgi) > 0:
            try:
                gdf_rgi.plot(ax=ax6, color="cyan", alpha=0.6, markersize=3)
            except Exception:
                pass
        ax6.annotate(
            "Torres del\nPaine", xy=(CENTER_LON, CENTER_LAT),
            fontsize=7, color="red", ha="center",
            xytext=(CENTER_LON + 1.8, CENTER_LAT + 0.8),
            arrowprops=dict(arrowstyle="->", color="red", lw=0.8),
        )
    except Exception as exc:
        ax6.text(0.5, 0.5, f"Map error:\n{exc}", ha="center", va="center",
                 transform=ax6.transAxes, fontsize=7)
    ax6.set_title("Study Area (Torres del Paine)", fontsize=9)
    ax6.set_xlabel("Longitude")
    ax6.set_ylabel("Latitude")

    # Timestamp footnote
    fig.text(
        0.5, 0.01,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | "
        f"BBOX: {BBOX} | ERA5 vars: {len(ERA5_DAILY_VARS.split(','))}",
        ha="center", fontsize=7, color="gray",
    )

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Dashboard -> {out_path}")


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    print("=" * 70)
    print("GeoCascade Ch01 -- Data Download Pipeline")
    print("=" * 70)
    print(f"  Output root : {DATA_ROOT}")
    print(f"  Study BBOX  : {BBOX}")
    print(f"  Timestamp   : {datetime.now().isoformat()}")

    env = load_env(ENV_FILE)

    df_era5     = None
    df_stations = None
    gdf_rgi     = None
    gdf_admin   = None

    # 1. ERA5
    try:
        df_era5 = download_era5(env)
    except Exception as exc:
        print(f"[ERROR] ERA5 download failed: {exc}")
        traceback.print_exc()

    # 2. CHIRPS
    try:
        download_chirps()
    except Exception as exc:
        print(f"[ERROR] CHIRPS download failed: {exc}")
        traceback.print_exc()

    # 3. Stations
    try:
        df_stations = download_stations(env)
    except Exception as exc:
        print(f"[ERROR] Station download failed: {exc}")
        traceback.print_exc()

    # 4. RGI glaciers
    try:
        gdf_rgi = download_rgi()
    except Exception as exc:
        print(f"[ERROR] RGI download failed: {exc}")
        traceback.print_exc()

    # 5. Admin boundaries
    try:
        gdf_admin = download_admin_boundaries()
    except Exception as exc:
        print(f"[ERROR] Admin boundaries failed: {exc}")
        traceback.print_exc()

    # 6. Dashboard
    try:
        make_dashboard(df_era5, df_stations, gdf_rgi, gdf_admin)
    except Exception as exc:
        print(f"[ERROR] Dashboard generation failed: {exc}")
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)
    outputs = [
        RAW_DIR / "era5_daily_patagonia.csv",
        RAW_DIR / "era5_monthly_patagonia.csv",
        RAW_DIR / "ghcn_stations_patagonia.csv",
        RAW_DIR / "rgi70_patagonia_glaciers.gpkg",
        RAW_DIR / "admin_boundaries.gpkg",
        PROC_DIR / "real_data_summary_dashboard.png",
    ]
    chirps_count = len(list(CHIRPS_DIR.glob("chirps-v2.0.*.tif")))
    for p in outputs:
        status = "[OK]  " if p.exists() else "[MISS]"
        size   = f"{p.stat().st_size // 1024:>8} KB" if p.exists() else "        --"
        print(f"  {status} {size}  {p.name}")
    print(f"  [OK]  {chirps_count:>6} files  chirps_monthly/*.tif")
    print("=" * 70)
    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[FATAL] {exc}")
        traceback.print_exc()
        sys.exit(1)
