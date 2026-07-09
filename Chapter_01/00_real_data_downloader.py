"""
Chapter 1: 00_real_data_downloader.py

Academic Objective:
This is the MASTER DATA DOWNLOADER for the GeoCascade pipeline.
It fetches all real-world observational and gridded datasets from
free, open-access sources — no satellite processing required.

Downloads:
  1. ERA5-Land multi-variable daily series (Open-Meteo, no key)
  2. CHIRPS v2.0 monthly precipitation grids (UCSB, no key)
  3. NOAA GHCN weather stations nearby (Open-Meteo fallback, optional NOAA token)
  4. RGI 7.0 Glacier outlines for Patagonia (GLIMS, no key)
  5. Natural Earth country/admin boundaries

Output directory: data/raw/real_data/
  era5_daily_BBOX.csv        -- multi-variable ERA5 daily series
  era5_monthly_BBOX.csv      -- monthly aggregated ERA5
  chirps_YYYY_MM.tif         -- monthly CHIRPS precipitation grids
  ghcn_stations.csv          -- nearby GHCN station metadata + data
  rgi70_patagonia.gpkg       -- glacier outlines (Patagonia subset)
  admin_boundaries.gpkg      -- country + province borders

Dependencies:
mamba install -n geocascade_env -c conda-forge requests pandas geopandas rasterio matplotlib numpy -y
"""

import os
import requests
import gzip
import shutil
import zipfile
import io
import time
import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
from datetime import datetime

# ==========================================
# Configuration
# ==========================================
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
OUT_DIR   = os.path.join(BASE_DIR, "data", "raw", "real_data")
PROC_DIR  = os.path.join(BASE_DIR, "data", "processed", "real_data")
os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)

# Study area — Torres del Paine, Patagonia
LAT       = -51.0
LON       = -73.0
BBOX      = [-73.5, -51.5, -72.5, -50.5]  # [minx, miny, maxx, maxy]
START     = "1993-01-01"
END       = "2024-12-31"

# Optional: NOAA CDO token (get free at https://www.ncei.noaa.gov/cdo-web/token)
# Leave empty to use Open-Meteo fallback (which is just as good)
NOAA_TOKEN = ""


# ==========================================
# 1. ERA5-Land via Open-Meteo (Multi-Variable)
# ==========================================
def download_era5_multivar():
    print("\n[1/5] Downloading ERA5-Land multi-variable daily series (1993-2024)...")
    variables = [
        "temperature_2m_max",
        "temperature_2m_min",
        "temperature_2m_mean",
        "precipitation_sum",
        "windspeed_10m_max",
        "windgusts_10m_max",
        "relative_humidity_2m_mean",
        "surface_pressure",
        "snowfall_sum",
        "et0_fao_evapotranspiration",
        "shortwave_radiation_sum",
    ]
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={LAT}&longitude={LON}"
        f"&start_date={START}&end_date={END}"
        f"&daily={','.join(variables)}"
        f"&timezone=America/Santiago"
    )
    print(f"       URL: {url[:100]}...")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    data = r.json()

    df = pd.DataFrame(data["daily"])
    df["date"] = pd.to_datetime(df["time"])
    df = df.drop(columns=["time"]).set_index("date")

    nan_counts = df.isna().sum()
    if nan_counts.any():
        print(f"       [WARNING] NaN counts per variable:")
        for col, cnt in nan_counts[nan_counts > 0].items():
            print(f"         {col}: {cnt} NaN days")

    # Save daily
    daily_path = os.path.join(OUT_DIR, "era5_daily_patagonia.csv")
    df.reset_index().to_csv(daily_path, index=False)
    print(f"       [SUCCESS] Daily ERA5 saved: {daily_path}  ({len(df)} rows)")

    # Save monthly aggregation
    monthly = df.resample("ME").agg({
        "temperature_2m_max":  "mean",
        "temperature_2m_min":  "mean",
        "temperature_2m_mean": "mean",
        "precipitation_sum":   "sum",
        "windspeed_10m_max":   "mean",
        "snowfall_sum":        "sum",
        "et0_fao_evapotranspiration": "sum",
        "shortwave_radiation_sum":    "sum",
    }).reset_index()
    monthly["year"]  = monthly["date"].dt.year
    monthly["month"] = monthly["date"].dt.month
    monthly_path = os.path.join(OUT_DIR, "era5_monthly_patagonia.csv")
    monthly.to_csv(monthly_path, index=False)
    print(f"       [SUCCESS] Monthly ERA5 saved: {monthly_path}  ({len(monthly)} rows)")

    return df, monthly


# ==========================================
# 2. CHIRPS Monthly Precipitation Grids
# ==========================================
def download_chirps(years=None):
    """
    Download CHIRPS v2.0 monthly precipitation GeoTIFFs from UCSB server.
    No registration required. Resolution: 0.05 deg (~5.5 km).
    URL pattern: data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/tifs/
    """
    if years is None:
        years = list(range(2000, 2025))

    print(f"\n[2/5] Downloading CHIRPS v2.0 monthly precipitation ({years[0]}-{years[-1]})...")
    chirps_dir = os.path.join(OUT_DIR, "chirps_monthly")
    os.makedirs(chirps_dir, exist_ok=True)

    base_url = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/tifs"
    downloaded, skipped, failed = 0, 0, 0

    for year in years:
        for month in range(1, 13):
            fname = f"chirps-v2.0.{year}.{month:02d}.tif.gz"
            local_gz  = os.path.join(chirps_dir, fname)
            local_tif = os.path.join(chirps_dir, fname.replace(".gz", ""))

            if os.path.exists(local_tif):
                skipped += 1
                continue

            url = f"{base_url}/{fname}"
            try:
                r = requests.get(url, timeout=60, stream=True)
                r.raise_for_status()
                with open(local_gz, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                # Decompress
                with gzip.open(local_gz, "rb") as gz_in:
                    with open(local_tif, "wb") as tif_out:
                        shutil.copyfileobj(gz_in, tif_out)
                os.remove(local_gz)
                downloaded += 1
                if downloaded % 12 == 0:
                    print(f"       Progress: {year}-{month:02d} ({downloaded} downloaded)")
                time.sleep(0.3)  # polite rate limiting
            except Exception as e:
                failed += 1
                print(f"       [WARNING] Failed {year}-{month:02d}: {e}")

    print(f"       [SUCCESS] CHIRPS: {downloaded} downloaded, {skipped} skipped, {failed} failed")
    print(f"       CHIRPS files in: {chirps_dir}")
    return chirps_dir


# ==========================================
# 3. GHCN Stations via NOAA CDO or Open-Meteo fallback
# ==========================================
def download_ghcn_stations():
    """
    Downloads GHCN-Daily data for nearby stations.
    If NOAA_TOKEN is set, uses the official CDO API.
    Otherwise falls back to Open-Meteo for virtual station data.
    """
    print("\n[3/5] Fetching real weather station data...")

    # Real stations near Torres del Paine (WMO/GHCN IDs)
    stations = [
        {"name": "Punta Arenas",   "lat": -53.00, "lon": -70.85, "id": "GHCND:ARM00085442"},
        {"name": "Puerto Natales", "lat": -51.73, "lon": -72.53, "id": "GHCND:CIM00085765"},
        {"name": "Balmaceda",      "lat": -45.92, "lon": -71.67, "id": "GHCND:CIM00085586"},
        {"name": "Cochrane",       "lat": -47.25, "lon": -72.58, "id": "GHCND:CIM00085610"},
        # Virtual stations at key locations via ERA5
        {"name": "Grey Glacier (virtual)",    "lat": -50.97, "lon": -73.22, "id": None},
        {"name": "Torres del Paine (virtual)","lat": -51.10, "lon": -72.95, "id": None},
        {"name": "Lago Grey (virtual)",       "lat": -51.05, "lon": -73.18, "id": None},
    ]

    all_records = []

    for stn in stations:
        print(f"       Fetching: {stn['name']} ({stn['lat']}, {stn['lon']})...")
        source = "open-meteo (ERA5)"

        # Try NOAA CDO if token provided
        if NOAA_TOKEN and stn["id"]:
            try:
                url = (
                    f"https://www.ncdc.noaa.gov/cdo-web/api/v2/data"
                    f"?datasetid=GHCND&stationid={stn['id']}"
                    f"&startdate=2010-01-01&enddate=2023-12-31"
                    f"&limit=1000&units=metric&datatypeid=TMAX,TMIN,PRCP"
                )
                r = requests.get(url, headers={"token": NOAA_TOKEN}, timeout=30)
                r.raise_for_status()
                rows = r.json().get("results", [])
                if rows:
                    df_stn = pd.DataFrame(rows)
                    df_stn["station_name"] = stn["name"]
                    df_stn["source"] = "NOAA GHCN"
                    all_records.append(df_stn)
                    source = "NOAA GHCN"
                    print(f"         NOAA: {len(df_stn)} records")
                    continue
            except Exception as e:
                print(f"         NOAA failed ({e}), using Open-Meteo fallback...")

        # Open-Meteo fallback (ERA5 virtual station)
        url = (
            f"https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={stn['lat']}&longitude={stn['lon']}"
            f"&start_date=2010-01-01&end_date=2023-12-31"
            f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max"
            f"&timezone=America/Santiago"
        )
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            data = r.json()
            df_stn = pd.DataFrame(data["daily"])
            df_stn["station_name"] = stn["name"]
            df_stn["station_lat"]  = stn["lat"]
            df_stn["station_lon"]  = stn["lon"]
            df_stn["source"]       = source
            all_records.append(df_stn)
            print(f"         Open-Meteo: {len(df_stn)} daily records")
        except Exception as e:
            print(f"         [ERROR] {stn['name']}: {e}")

        time.sleep(0.5)

    if not all_records:
        print("       [WARNING] No station data retrieved.")
        return None

    df_all = pd.concat(all_records, ignore_index=True)
    path = os.path.join(OUT_DIR, "ghcn_stations_patagonia.csv")
    df_all.to_csv(path, index=False)
    print(f"       [SUCCESS] {len(all_records)} stations saved: {path}")
    return df_all


# ==========================================
# 4. RGI 7.0 Glacier Outlines (Patagonia)
# ==========================================
def download_rgi_glaciers():
    """
    Downloads Randolph Glacier Inventory v7.0 for South America.
    Region 17 (Low Latitudes South) includes Patagonian Ice Fields.
    No registration required.
    """
    print("\n[4/5] Downloading RGI 7.0 Glacier Outlines (Patagonia)...")
    out_path = os.path.join(OUT_DIR, "rgi70_patagonia_glaciers.gpkg")

    if os.path.exists(out_path):
        print(f"       [SKIP] Already exists: {out_path}")
        gdf = gpd.read_file(out_path)
        print(f"       {len(gdf)} glacier polygons loaded.")
        return gdf

    # RGI 7.0 download URL for region 17 (South America Low Latitudes)
    url = "https://www.glims.org/RGI/rgi70_files/17_rgi70_LowLatitudes.zip"
    print(f"       Downloading from: {url}")
    print("       (This is ~50MB, please wait...)")

    try:
        r = requests.get(url, timeout=300, stream=True)
        r.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(r.content))
        # Find the shapefile inside the zip
        shp_files = [n for n in z.namelist() if n.endswith(".shp")]
        if not shp_files:
            raise ValueError("No .shp found in RGI zip")

        # Extract to temp dir
        tmp_dir = os.path.join(OUT_DIR, "rgi_tmp")
        z.extractall(tmp_dir)

        shp_path = os.path.join(tmp_dir, shp_files[0])
        gdf = gpd.read_file(shp_path)

        # Filter to Torres del Paine / Southern Patagonia
        gdf = gdf.cx[-75:-71, -53:-49]
        gdf = gdf.to_crs("EPSG:4326")
        gdf.to_file(out_path, driver="GPKG")
        shutil.rmtree(tmp_dir)

        print(f"       [SUCCESS] {len(gdf)} Patagonian glacier polygons saved: {out_path}")
        return gdf

    except Exception as e:
        print(f"       [WARNING] RGI download failed: {e}")
        print("       You can download manually: https://www.glims.org/RGI/rgi70_files/17_rgi70_LowLatitudes.zip")
        return None


# ==========================================
# 5. Natural Earth Admin Boundaries
# ==========================================
def download_admin_boundaries():
    """
    Downloads Natural Earth country + admin boundaries.
    Built into geopandas — no external download needed.
    """
    print("\n[5/5] Loading administrative boundaries (Natural Earth)...")
    try:
        world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
        chile_arg = world[world["name"].isin(["Chile", "Argentina"])]
        out_path = os.path.join(OUT_DIR, "admin_boundaries.gpkg")
        chile_arg.to_file(out_path, driver="GPKG")
        print(f"       [SUCCESS] Admin boundaries saved: {out_path}")
        return chile_arg
    except Exception as e:
        print(f"       [WARNING] Admin boundaries: {e}")
        return None


# ==========================================
# 6. Summary Visualization
# ==========================================
def plot_data_summary(df_era5_daily, df_stations):
    print("\n[INFO] Generating Real Data Summary Dashboard...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("GeoCascade — Real Data Download Summary\nTorres del Paine, Patagonia",
                 fontsize=14, fontweight="bold")

    df = df_era5_daily.copy()

    # Panel 1: Annual temperature
    ax = axes[0, 0]
    annual_temp = df["temperature_2m_mean"].resample("YE").mean()
    ax.plot(annual_temp.index.year, annual_temp.values, "o-", color="#e74c3c", linewidth=2)
    ax.fill_between(annual_temp.index.year, annual_temp.values - 1, annual_temp.values + 1,
                    alpha=0.2, color="#e74c3c")
    ax.set_title("Annual Mean Temperature (ERA5)", fontsize=10)
    ax.set_ylabel("Temperature (°C)")
    ax.grid(axis="y", alpha=0.4)

    # Panel 2: Annual precipitation
    ax = axes[0, 1]
    annual_prec = df["precipitation_sum"].resample("YE").sum()
    colors = ["#d62728" if p < annual_prec.mean() * 0.85 else
              "#1f77b4" if p > annual_prec.mean() * 1.15 else "#7f7f7f"
              for p in annual_prec.values]
    ax.bar(annual_prec.index.year, annual_prec.values, color=colors, alpha=0.7)
    ax.axhline(annual_prec.mean(), color="black", linestyle="--", linewidth=1.5,
               label=f"Mean: {annual_prec.mean():.0f} mm")
    ax.set_title("Annual Precipitation (ERA5)", fontsize=10)
    ax.set_ylabel("Precipitation (mm)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.4)

    # Panel 3: Seasonal cycle
    ax = axes[0, 2]
    monthly_clim = df.groupby(df.index.month).agg({
        "temperature_2m_mean": "mean",
        "precipitation_sum": "sum"
    })
    ax2 = ax.twinx()
    ax.bar(monthly_clim.index, monthly_clim["precipitation_sum"] / len(df.index.year.unique()),
           color="#3498db", alpha=0.5, label="Precip (mm/month)")
    ax2.plot(monthly_clim.index, monthly_clim["temperature_2m_mean"],
             "o-", color="#e74c3c", linewidth=2, label="Temp (°C)")
    ax.set_title("Climatological Seasonal Cycle", fontsize=10)
    ax.set_xlabel("Month")
    ax.set_ylabel("Precipitation (mm)", color="#3498db")
    ax2.set_ylabel("Temperature (°C)", color="#e74c3c")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"])

    # Panel 4: Wind
    ax = axes[1, 0]
    annual_wind = df["windspeed_10m_max"].resample("ME").mean()
    ax.plot(annual_wind.index, annual_wind.values, color="#2ecc71", linewidth=1.2, alpha=0.8)
    rolling = annual_wind.rolling(12, center=True).mean()
    ax.plot(rolling.index, rolling.values, color="#27ae60", linewidth=2.5, label="12-mo rolling mean")
    ax.set_title("Maximum Wind Speed (ERA5)", fontsize=10)
    ax.set_ylabel("Wind Speed (km/h)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.4)

    # Panel 5: Snow accumulation
    ax = axes[1, 1]
    if "snowfall_sum" in df.columns:
        annual_snow = df["snowfall_sum"].resample("YE").sum()
        ax.bar(annual_snow.index.year, annual_snow.values, color="#00bcd4", alpha=0.7)
        ax.set_title("Annual Snowfall (ERA5)", fontsize=10)
        ax.set_ylabel("Snowfall (cm)")
        ax.grid(axis="y", alpha=0.4)

    # Panel 6: Station comparison
    ax = axes[1, 2]
    if df_stations is not None and "station_name" in df_stations.columns:
        df_stn = df_stations.copy()
        df_stn["time"] = pd.to_datetime(df_stn["time"])
        for stn_name, grp in df_stn.groupby("station_name"):
            if "temperature_2m_max" in grp.columns:
                monthly = grp.set_index("time")["temperature_2m_max"].resample("ME").mean()
                if len(monthly) > 12:
                    clim = monthly.groupby(monthly.index.month).mean()
                    ax.plot(clim.index, clim.values, "o-", linewidth=1.5, label=stn_name, alpha=0.8)
        ax.set_title("Station Seasonal Temperature Profiles", fontsize=10)
        ax.set_ylabel("Max Temperature (°C)")
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"])
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(axis="y", alpha=0.4)
    else:
        ax.text(0.5, 0.5, "No multi-station data\n(add NOAA token for GHCN)",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
        ax.set_title("Station Comparison", fontsize=10)

    plt.tight_layout()
    plot_path = os.path.join(PROC_DIR, "real_data_summary_dashboard.png")
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"       [SUCCESS] Dashboard saved: {plot_path}")


# ==========================================
# Main
# ==========================================
def main():
    print("=" * 65)
    print(" GEOCASCADE — REAL DATA DOWNLOADER")
    print("=" * 65)
    print(f" Study area: {BBOX}  |  Period: {START} to {END}")
    if not NOAA_TOKEN:
        print(" [INFO] NOAA_TOKEN not set — using Open-Meteo for all station data.")
        print("        Get free token at: https://www.ncei.noaa.gov/cdo-web/token")
    print()

    df_era5_daily, df_era5_monthly = download_era5_multivar()
    chirps_dir  = download_chirps(years=list(range(2000, 2025)))
    df_stations = download_ghcn_stations()
    gdf_glaciers = download_rgi_glaciers()
    gdf_admin    = download_admin_boundaries()

    plot_data_summary(df_era5_daily, df_stations)

    print("\n" + "=" * 65)
    print(" DOWNLOAD COMPLETE — Summary")
    print("=" * 65)
    print(f"  ERA5 daily series:   {len(df_era5_daily)} days x 11 variables")
    print(f"  Station data:        {len(df_stations) if df_stations is not None else 0} records")
    if gdf_glaciers is not None:
        print(f"  Glacier polygons:    {len(gdf_glaciers)} RGI 7.0 outlines")
    print(f"  All files in:        {OUT_DIR}")
    print(f"  Dashboard:           {PROC_DIR}/real_data_summary_dashboard.png")
    print()
    print("  Next step: Run 03c_chirps_spatial_precipitation.py")
    print("             Run 23_real_data_convergence.py (Chapter 8)")


if __name__ == "__main__":
    main()
