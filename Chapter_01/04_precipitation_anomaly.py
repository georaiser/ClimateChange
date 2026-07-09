"""
Chapter 1: 04_precipitation_anomaly.py

Academic Objective:
To analyze climate change, we must look at long-term historical trends. 
This script downloads 30 YEARS of REAL daily precipitation data (ERA5 Reanalysis) 
for the Torres del Paine region.

It then applies Machine Learning (K-Means Clustering) and statistical anomaly 
detection (Z-Scores) to automatically classify historical years into 
Droughts, Normal years, and Extreme Flood years.

Dependencies:
mamba install -n geocascade_env -c conda-forge pandas matplotlib scikit-learn requests -y
"""

import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ==========================================
# 1. Configuration
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "data", "processed", "climate_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

# Torres del Paine central coordinate
LAT = -51.0
LON = -73.0

# 30-Year Climate Baseline
START_DATE = "1993-01-01"
END_DATE = "2023-12-31"

# ==========================================
# 2. Fetch 30 Years of Real Climate Data
# ==========================================
def fetch_historical_precipitation():
    print(f"\n[INFO] Fetching 30 years of real daily precipitation data from {START_DATE} to {END_DATE}...")
    print(f"       Location: Lat {LAT}, Lon {LON} (Torres del Paine)")
    
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={LAT}&longitude={LON}&start_date={START_DATE}&end_date={END_DATE}&daily=precipitation_sum&timezone=auto"
    
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': pd.to_datetime(data['daily']['time']),
        'precip_mm': data['daily']['precipitation_sum']
    })

    # Warn before dropping NaN days (ERA5 gaps can skew annual totals silently)
    nan_count = df['precip_mm'].isna().sum()
    if nan_count > 0:
        print(f"       [WARNING] {nan_count} days have missing precipitation data and will be excluded from annual totals.")
    df = df.dropna(subset=['precip_mm'])
    print(f"       [SUCCESS] Downloaded {len(df)} days of historical climate data!")
    return df

# ==========================================
# 3. Time-Series Aggregation & Anomaly Detection
# ==========================================
def analyze_climate_anomalies(df):
    print("\n[INFO] Aggregating daily data into annual totals...")
    
    # Set date as index for resampling
    df.set_index('date', inplace=True)
    
    # Resample to Annual Sums ('Y' = Year end)
    annual_df = df.resample('YE').sum().reset_index()
    annual_df['year'] = annual_df['date'].dt.year
    
    # Calculate Statistical Anomaly (Z-Score)
    # Z = (Value - Mean) / Standard Deviation
    mean_precip = annual_df['precip_mm'].mean()
    std_precip = annual_df['precip_mm'].std()
    
    annual_df['z_score'] = (annual_df['precip_mm'] - mean_precip) / std_precip
    
    print("\n[INFO] Applying K-Means Clustering to classify Climate Regimes...")
    # We ask K-Means to find 3 distinct clusters: Drought, Normal, Flood
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    
    # Reshape for sklearn
    X = annual_df['precip_mm'].values.reshape(-1, 1)
    annual_df['cluster'] = kmeans.fit_predict(X)
    
    # Sort cluster centers so 0=Drought, 1=Normal, 2=Flood
    centers = kmeans.cluster_centers_.flatten()
    sorted_idx = np.argsort(centers)
    # Guard: assert 3 distinct clusters were found (can merge on low-variance data)
    if len(set(sorted_idx)) != 3:
        print("       [WARNING] K-Means produced degenerate clusters. Regime labels may be inaccurate.")
    mapping = {sorted_idx[0]: 'Drought', sorted_idx[1]: 'Normal', sorted_idx[2]: 'Flood'}
    annual_df['climate_regime'] = annual_df['cluster'].map(mapping)

    # Tier 3: export 30-year annual time series as CSV for downstream analysis
    csv_path = os.path.join(OUT_DIR, "annual_precipitation_30yr.csv")
    annual_df[['year', 'precip_mm', 'z_score', 'climate_regime']].to_csv(csv_path, index=False)
    print(f"       [SUCCESS] Annual series exported: {csv_path}")

    return annual_df, mean_precip

# ==========================================
# 4. Plotting & Export
# ==========================================
def plot_results(annual_df, mean_precip):
    print("\n[INFO] Generating Climate Anomaly Plot...")
    
    plt.figure(figsize=(14, 7))
    
    # Bar colors based on regime
    colors = {'Drought': 'red', 'Normal': 'gray', 'Flood': 'blue'}
    bar_colors = [colors[regime] for regime in annual_df['climate_regime']]
    
    bars = plt.bar(annual_df['year'], annual_df['precip_mm'], color=bar_colors, alpha=0.7)
    
    # Add horizontal line for 30-year average
    plt.axhline(y=mean_precip, color='black', linestyle='--', linewidth=2, label=f'30-Year Average ({mean_precip:.1f} mm)')
    
    plt.title("30-Year Precipitation Anomalies (Torres del Paine)\nMachine Learning Classification: Droughts vs Floods", fontsize=16)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Annual Precipitation (mm)", fontsize=12)
    
    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', alpha=0.7, label='Extreme Flood Year'),
        Patch(facecolor='gray', alpha=0.7, label='Normal Year'),
        Patch(facecolor='red', alpha=0.7, label='Extreme Drought Year'),
        plt.Line2D([0], [0], color='black', linestyle='--', lw=2, label='30-Year Average')
    ]
    plt.legend(handles=legend_elements, loc='upper left')
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save plot
    plot_path = os.path.join(OUT_DIR, "30yr_precipitation_anomalies.png")
    fig = plt.gcf()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)  # prevent memory leak in multi-script pipelines
    print(f"       [SUCCESS] Plot saved to: {plot_path}")

def main():
    print("=======================================================")
    print(" GEOCASCADE PIPELINE - LONG-TERM CLIMATE ANOMALY ML    ")
    print("=======================================================")
    
    df = fetch_historical_precipitation()
    annual_df, mean_precip = analyze_climate_anomalies(df)
    
    # Print the worst drought and worst flood years
    worst_drought = annual_df.loc[annual_df['precip_mm'].idxmin()]
    worst_flood = annual_df.loc[annual_df['precip_mm'].idxmax()]
    
    print("\n--- CLIMATE EXTREMES DISCOVERED ---")
    print(f"Most Severe Drought: {worst_drought['year']} ({worst_drought['precip_mm']:.1f} mm)")
    print(f"Most Severe Flood:   {worst_flood['year']} ({worst_flood['precip_mm']:.1f} mm)")
    
    plot_results(annual_df, mean_precip)
    print("\n[SUCCESS] Chapter 1 Dual Precipitation Analysis complete.")

if __name__ == "__main__":
    main()
