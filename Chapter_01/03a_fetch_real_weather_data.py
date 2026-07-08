"""
Chapter 1: 03a_fetch_real_weather_data.py

Academic Objective:
To perform spatial interpolation (Machine Learning), we need data from multiple 
weather stations. Real in-situ stations in Torres del Paine are extremely sparse.
Instead, we will use the Open-Meteo Historical API to fetch real ERA5-Land 
reanalysis data for a grid of 25 simulated "stations" across our ROI.

This script fetches REAL historical temperature and elevation data for 
Jan 15, 2023. To test our Isolation Forest anomaly detection in Script 3, 
we will intentionally "corrupt" a few of these stations to simulate 
hardware sensor failure.

Dependencies:
requests, pandas, numpy
"""

import os
import requests
import pandas as pd
import numpy as np
from time import sleep

# ==========================================
# 1. Configuration
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "data", "raw")
os.makedirs(OUT_DIR, exist_ok=True)
CSV_PATH = os.path.join(OUT_DIR, "weather_stations.csv")

# Torres del Paine / Grey Glacier Bounding Box
BBOX = [-73.30, -51.10, -72.90, -50.80]
DATE = "2023-01-15"

# ==========================================
# 2. Generate Station Grid
# ==========================================
print("[INFO] Generating a grid of 25 virtual weather stations across Torres del Paine...")
lons = np.linspace(BBOX[0], BBOX[2], 5)
lats = np.linspace(BBOX[1], BBOX[3], 5)

stations = []
station_id = 1
for lat in lats:
    for lon in lons:
        stations.append({
            "station_id": f"TDP_WS_{station_id:03d}",
            "lat": lat,
            "lon": lon
        })
        station_id += 1

# ==========================================
# 3. Fetch Real ERA5 Climate Data
# ==========================================
print(f"[INFO] Fetching REAL ERA5-Land Temperature & Elevation data for {DATE}...")
print("       (Using Open-Meteo Historical API - No API key required)")

for i, st in enumerate(stations):
    # Fetch historical daily mean temperature and elevation
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={st['lat']}&longitude={st['lon']}&start_date={DATE}&end_date={DATE}&daily=temperature_2m_mean&timezone=auto"

    #print(f"       Fetching station {st['station_id']} ({i+1}/{len(stations)}) at ({st['lat']:.4f}, {st['lon']:.4f})...")
    #print(f"       URL: {url}")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        #print(data)  # Print the raw JSON response for debugging
        
        # Open-Meteo automatically returns the DEM elevation of the coordinate
        st['elevation'] = data.get('elevation', 0)
        # Get the daily mean temperature
        st['temp_celsius'] = data['daily']['temperature_2m_mean'][0]
        
    except Exception as e:
        print(f"       [WARNING] Failed to fetch data for station {st['station_id']}: {e}")
        st['elevation'] = np.nan
        st['temp_celsius'] = np.nan
        
    # Sleep briefly to respect API rate limits
    sleep(0.1)

df = pd.DataFrame(stations)
df = df.dropna()

# ==========================================
# 4. Inject Synthetic Hardware Failures
# ==========================================
print("[INFO] Injecting synthetic hardware failures to test our ML Anomaly Detection...")
# We will corrupt 3 random stations to have impossible temperatures for Patagonia in summer
corrupt_indices = np.random.choice(df.index, size=3, replace=False)
df.loc[corrupt_indices[0], 'temp_celsius'] = 85.5   # Impossible heat spike
df.loc[corrupt_indices[1], 'temp_celsius'] = -99.9  # Common sensor dead-value
df.loc[corrupt_indices[2], 'temp_celsius'] = 45.0   # Extreme heat

# ==========================================
# 5. Export Data
# ==========================================
df.to_csv(CSV_PATH, index=False)
print(f"[SUCCESS] Real weather station data (with synthetic anomalies) saved to:")
print(f"          {CSV_PATH}")
print(f"          Total Valid Stations: {len(df)}")
print("\n[NEXT STEP] You can now run `03_station_ml_interpolation.py` to watch the ")
print("            Isolation Forest automatically detect the anomalies we just injected!")
