"""
Chapter 1: 01_prepare_weather_stations.py

Bridges the real data downloaded by 00_real_data_downloader.py into the
schema expected by 03_station_ml_interpolation.py.

00_real_data_downloader.py writes data/raw/real_data/ghcn_stations_patagonia.csv,
which mixes two different shapes:
  - NOAA GHCN rows: long format, one row per station/date/datatype (TMAX/TMIN/PRCP)
  - Open-Meteo fallback rows: wide format, one row per station/date already

Neither has an 'elevation' column, and NOAA rows don't carry lat/lon either
(only the Open-Meteo fallback rows do). This script:
  1. Harmonizes both shapes into a common (station, date, temp_celsius) table
  2. Adds lat/lon from the same station metadata used in 00_real_data_downloader.py
  3. Fetches elevation per station via Open-Meteo's free Elevation API (one
     batched call, no key required)
  4. Writes data/raw/weather_stations.csv for 03_station_ml_interpolation.py

Run this AFTER 00_real_data_downloader.py and BEFORE 03_station_ml_interpolation.py.
"""

import os
import requests
import pandas as pd

# ==========================================
# Configuration
# ==========================================
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
GHCN_PATH = os.path.join(BASE_DIR, "data", "raw", "real_data", "ghcn_stations_patagonia.csv")
OUT_PATH  = os.path.join(BASE_DIR, "data", "raw", "weather_stations.csv")
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

# Same station metadata used in 00_real_data_downloader.py. Needed because the
# NOAA-sourced rows in ghcn_stations_patagonia.csv don't carry coordinates —
# only the Open-Meteo fallback rows do.
STATIONS = {
    "Punta Arenas":                (-53.00, -70.85),
    "Puerto Natales":              (-51.73, -72.53),
    "Balmaceda":                   (-45.92, -71.67),
    "Cochrane":                    (-47.25, -72.58),
    "Grey Glacier (virtual)":      (-50.97, -73.22),
    "Torres del Paine (virtual)":  (-51.10, -72.95),
    "Lago Grey (virtual)":         (-51.05, -73.18),
}


def fetch_elevations(stations):
    """One batched call to Open-Meteo's free elevation API for all stations.
    Response is always {"elevation": [...]} in the same order as input."""
    lats = ",".join(str(lat) for lat, lon in stations.values())
    lons = ",".join(str(lon) for lat, lon in stations.values())
    url = f"https://api.open-meteo.com/v1/elevation?latitude={lats}&longitude={lons}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    elevations = r.json()["elevation"]
    return dict(zip(stations.keys(), elevations))


def harmonize_noaa(df_noaa):
    """NOAA rows are long-format: one row per station/date/datatype."""
    df_noaa = df_noaa.copy()
    df_noaa["date"] = pd.to_datetime(df_noaa["date"]).dt.date
    pivot = df_noaa.pivot_table(
        index=["station_name", "date"], columns="datatype", values="value", aggfunc="mean"
    ).reset_index()
    # TMAX/TMIN are already in Celsius (NOAA request used units=metric)
    pivot["temp_celsius"] = pivot[["TMAX", "TMIN"]].mean(axis=1)
    return pivot[["station_name", "date", "temp_celsius"]]


def harmonize_open_meteo(df_om):
    """Open-Meteo fallback rows are wide-format: one row per station/date already."""
    df_om = df_om.copy()
    df_om["date"] = pd.to_datetime(df_om["time"]).dt.date
    df_om["temp_celsius"] = df_om[["temperature_2m_max", "temperature_2m_min"]].mean(axis=1)
    return df_om[["station_name", "date", "temp_celsius"]]


def main():
    print("=======================================================")
    print(" BUILDING weather_stations.csv FROM REAL DOWNLOADED DATA ")
    print("=======================================================")

    if not os.path.exists(GHCN_PATH):
        print(f"[ERROR] Expected input not found: {GHCN_PATH}")
        print("        Run 00_real_data_downloader.py first.")
        return

    df = pd.read_csv(GHCN_PATH)
    is_noaa = df["source"] == "NOAA GHCN"

    parts = []
    if is_noaa.any():
        print(f"[INFO] Harmonizing {is_noaa.sum()} NOAA GHCN rows...")
        parts.append(harmonize_noaa(df[is_noaa]))
    if (~is_noaa).any():
        print(f"[INFO] Harmonizing {(~is_noaa).sum()} Open-Meteo fallback rows...")
        parts.append(harmonize_open_meteo(df[~is_noaa]))

    if not parts:
        print("[ERROR] No usable rows found in ghcn_stations_patagonia.csv")
        return

    daily = pd.concat(parts, ignore_index=True).dropna(subset=["temp_celsius"])

    unknown = set(daily["station_name"]) - set(STATIONS)
    if unknown:
        print(f"[WARNING] Dropping rows for stations with no known coordinates: {unknown}")
        daily = daily[~daily["station_name"].isin(unknown)]

    print(f"[INFO] Fetching elevations for {len(STATIONS)} stations (Open-Meteo Elevation API)...")
    elevations = fetch_elevations(STATIONS)
    for name, elev in elevations.items():
        print(f"       {name}: {elev:.0f} m")

    daily["lat"]        = daily["station_name"].map(lambda n: STATIONS[n][0])
    daily["lon"]         = daily["station_name"].map(lambda n: STATIONS[n][1])
    daily["elevation"]   = daily["station_name"].map(elevations)
    daily["station_id"]  = daily["station_name"]

    out = daily[["station_id", "date", "lat", "lon", "elevation", "temp_celsius"]]
    out = out.sort_values(["station_id", "date"]).reset_index(drop=True)
    out.to_csv(OUT_PATH, index=False)

    print(f"\n[SUCCESS] {len(out)} rows across {out['station_id'].nunique()} stations saved:")
    print(f"          {OUT_PATH}")
    print(f"\n          Ready for 03_station_ml_interpolation.py")


if __name__ == "__main__":
    main()
