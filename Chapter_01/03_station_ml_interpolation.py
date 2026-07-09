"""
Chapter 1: 03_station_ml_interpolation.py

Academic Objective:
In-situ weather stations provide accurate point data, but to model the Cascade Effect 
(e.g., how temperature affects a glacier), we need continuous raster maps.

This script demonstrates two advanced concepts:
1. Automated Anomaly Detection: Weather stations often fail in harsh environments like Patagonia. 
   We use an Isolation Forest (Unsupervised ML) to automatically flag extreme outliers, 
   followed by visual/manual validation.
2. Spatial Interpolation: We train a Random Forest Regressor using Elevation, Lat, and Lon 
   as covariates to predict temperature across the entire landscape.

Dependencies:
mamba install -n geocascade_env -c conda-forge scikit-learn pandas geopandas matplotlib rasterio numpy
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# rasterio imported here for the DEM-based prediction step (Step 3 placeholder)
import rasterio

# ==========================================
# 1. Configuration
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "data", "raw", "weather_stations.csv")
OUT_DIR = os.path.join(BASE_DIR, "data", "processed", "climate_maps")
os.makedirs(OUT_DIR, exist_ok=True)


# ==========================================
# 2. Automated Anomaly Detection (Isolation Forest)
# ==========================================
def detect_and_clean_anomalies(df):
    print("\n[INFO] Step 1: Running Isolation Forest Anomaly Detection on Station Data...")
    
    # Guard against NaN temperatures before fitting IsolationForest
    df_clean = df.dropna(subset=['temp_celsius'])
    if len(df_clean) < len(df):
        print(f"       [WARNING] Dropped {len(df) - len(df_clean)} rows with NaN temperatures before anomaly detection.")

    X_temp = df_clean[['temp_celsius']].values

    iso_forest = IsolationForest(contamination=0.15, random_state=42)
    df_clean = df_clean.copy()
    df_clean['anomaly_score'] = iso_forest.fit_predict(X_temp)
    
    normal_data = df_clean[df_clean['anomaly_score'] == 1]
    anomalies   = df_clean[df_clean['anomaly_score'] == -1]

    print(f"       Found {len(anomalies)} anomalous records out of {len(df_clean)} total.")
    for idx, row in anomalies.iterrows():
        print(f"       [FLAGGED] Station {row['station_id']}: {row['temp_celsius']}°C")

    fig = plt.figure(figsize=(8, 5))
    plt.scatter(normal_data['elevation'], normal_data['temp_celsius'], c='blue', label='Normal (Valid)')
    plt.scatter(anomalies['elevation'],   anomalies['temp_celsius'],   c='red',  label='Anomaly (Sensor Error)', marker='x', s=100)
    plt.title("Manual Validation: Elevation vs Temperature Anomalies")
    plt.xlabel("Elevation (m)")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(OUT_DIR, "anomaly_validation_plot.png")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close(fig)  # prevent memory leak in multi-script pipelines
    print(f"       [ACTION] Validation plot saved to: {plot_path}. Please review manually.")

    return normal_data.drop(columns=['anomaly_score'])


# ==========================================
# 3. Machine Learning Interpolation (Random Forest)
# ==========================================
def train_interpolation_model(cleaned_df):
    print("\n[INFO] Step 2: Training Random Forest Interpolator...")
    
    # Features: Latitude, Longitude, Elevation
    X = cleaned_df[['lat', 'lon', 'elevation']]
    # Target: Temperature
    y = cleaned_df['temp_celsius']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    predictions = rf_model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    print(f"       [SUCCESS] Model trained. Test RMSE: {rmse:.2f} °C  (MSE={mse:.2f} °C²)")
    # Note: RMSE is in the same unit as the target (°C). MSE is in °C² — not directly interpretable.
    
    return rf_model


def main():
    print("=======================================================")
    print(" GEOCASCADE PIPELINE - SENSOR QA/QC & ML INTERPOLATION ")
    print("=======================================================")
    
    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] Mock data not found at {CSV_PATH}")
        return
        
    # 1. Load Data
    df = pd.read_csv(CSV_PATH)
    
    # 2. Anomaly Detection & Manual QA
    cleaned_df = detect_and_clean_anomalies(df)
    
    # 3. Train Model
    model = train_interpolation_model(cleaned_df)
    
    print("\n[INFO] Step 3: Applying model to continuous DEM surface (Simulated)...")
    print("       (In a full run, we load the Copernicus DEM array here and pass it to model.predict())")
    print("\n[SUCCESS] Chapter 1 ML Interpolation complete.")

if __name__ == "__main__":
    main()
