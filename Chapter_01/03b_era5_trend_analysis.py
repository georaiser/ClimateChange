"""
Chapter 1 (Tier 4): 03b_era5_trend_analysis.py

Academic Objective:
Detecting climate change signals requires distinguishing genuine long-term trends from
random inter-annual variability. Simple linear regression is distorted by outliers and
non-normality. This script demonstrates two superior methods:

1. Mann-Kendall Test (Hirsch et al., 1982): Non-parametric H0 test.
   H0: No monotonic trend. H1: A monotonic trend exists.
   Makes NO assumptions about distribution; robust to non-normal precipitation data.

2. Sen's Slope Estimator (Sen, 1968): Non-parametric trend magnitude.
   Computes the MEDIAN slope of all pairwise combinations.
   Completely insensitive to outliers unlike OLS regression.

Outputs:
- Console: Mann-Kendall tau, p-value, trend direction, Sen Slope mm/year
- Chart: Annual bars + Sen trend line vs OLS comparison
- CSV: annual_precipitation_with_trend.csv

Dependencies:
mamba install -n geocascade_env -c conda-forge pandas matplotlib scipy requests -y
"""

import os
import math
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ==========================================
# 1. Configuration
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(BASE_DIR, "data", "processed", "climate_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

LAT, LON   = -51.0, -73.0
START_DATE = "1993-01-01"
END_DATE   = "2023-12-31"


# ==========================================
# 2. Data Acquisition
# ==========================================
def fetch_annual_series():
    print("\n[INFO] Fetching 30-year daily precipitation from ERA5 (Open-Meteo archive)...")
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={LAT}&longitude={LON}"
        f"&start_date={START_DATE}&end_date={END_DATE}"
        f"&daily=precipitation_sum&timezone=auto"
    )
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    data = r.json()

    df = pd.DataFrame({
        'date':      pd.to_datetime(data['daily']['time']),
        'precip_mm': data['daily']['precipitation_sum']
    })
    nan_count = df['precip_mm'].isna().sum()
    if nan_count > 0:
        print(f"       [WARNING] {nan_count} NaN days excluded from annual totals.")
    df = df.dropna(subset=['precip_mm'])
    df.set_index('date', inplace=True)

    annual = df.resample('YE').sum().reset_index()
    annual['year'] = annual['date'].dt.year
    print(f"       [SUCCESS] {len(annual)} annual observations ready.")
    return annual


# ==========================================
# 3. Mann-Kendall Test (Non-Parametric)
# ==========================================
def mann_kendall(series):
    """
    Manual implementation - no external pymannkendall dependency needed.
    Kendall tau in [-1, 1]. p < 0.05 means reject H0 (trend exists).
    """
    n = len(series)
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            diff = series[j] - series[i]
            if diff > 0:
                s += 1
            elif diff < 0:
                s -= 1
    var_s = n * (n - 1) * (2 * n + 5) / 18
    if s > 0:
        z = (s - 1) / math.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / math.sqrt(var_s)
    else:
        z = 0.0
    p_value  = 2 * (1 - stats.norm.cdf(abs(z)))
    tau      = 2 * s / (n * (n - 1))
    direction = "Increasing" if s > 0 else ("Decreasing" if s < 0 else "No trend")
    return tau, p_value, direction


# ==========================================
# 4. Sen's Slope Estimator
# ==========================================
def sens_slope(years, values):
    """
    Median of all pairwise slopes Q_ij = (x_j - x_i) / (j - i).
    Completely robust to outliers (extreme flood/drought years).
    Returns (slope mm/yr, intercept).
    """
    n = len(values)
    slopes = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            dy = values[j] - values[i]
            dx = years[j] - years[i]
            if dx != 0:
                slopes.append(dy / dx)
    slope     = float(np.median(slopes))
    intercept = float(np.median(values - slope * years))
    return slope, intercept


# ==========================================
# 5. Visualization
# ==========================================
def plot_trend(annual, slope_sens, intercept_sens, ols_slope, ols_intercept,
               tau, p_value, direction):
    years  = annual['year'].values
    precip = annual['precip_mm'].values
    mean_p = np.mean(precip)

    fig, ax = plt.subplots(figsize=(14, 7))

    bar_colors = ['#d62728' if p < mean_p * 0.85
                  else ('#1f77b4' if p > mean_p * 1.15 else '#7f7f7f')
                  for p in precip]
    ax.bar(years, precip, color=bar_colors, alpha=0.65, label='Annual Precipitation')
    ax.axhline(mean_p, color='black', linestyle='--', linewidth=1.5,
               label=f'30-yr Mean ({mean_p:.0f} mm)')

    y_sens = slope_sens * years + intercept_sens
    y_ols  = ols_slope  * years + ols_intercept
    ax.plot(years, y_sens, color='darkorange', linewidth=2.5,
            label=f"Sen's Slope: {slope_sens:+.1f} mm/yr  (robust)")
    ax.plot(years, y_ols,  color='red',        linewidth=1.5, linestyle='--',
            label=f"OLS Regression: {ols_slope:+.1f} mm/yr  (comparison)")

    significance = "SIGNIFICANT" if p_value < 0.05 else "NOT significant"
    ax.text(0.02, 0.97,
            f"Mann-Kendall Test\n"
            f"  tau = {tau:+.3f}   p = {p_value:.4f}\n"
            f"  Trend: {direction} ({significance} at alpha=0.05)\n"
            f"  Sen's Slope: {slope_sens:+.2f} mm / year\n"
            f"  Over 30 yr: {slope_sens * 30:+.0f} mm total change",
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8),
            fontsize=10)

    ax.set_title("30-Year Precipitation Trend (ERA5) - Mann-Kendall + Sen's Slope\n"
                 "Torres del Paine, Patagonia", fontsize=14)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Annual Precipitation (mm)", fontsize=12)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()

    plot_path = os.path.join(OUT_DIR, "era5_precipitation_trend.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"       [SUCCESS] Trend chart saved: {plot_path}")


# ==========================================
# 6. Main
# ==========================================
def main():
    print("=======================================================")
    print(" GEOCASCADE - ERA5 TREND ANALYSIS (MANN-KENDALL / SEN) ")
    print("=======================================================")

    annual = fetch_annual_series()
    years  = annual['year'].values.astype(float)
    precip = annual['precip_mm'].values

    print("\n[INFO] Running Mann-Kendall Non-Parametric Trend Test...")
    tau, p_value, direction = mann_kendall(precip)
    print(f"       Kendall tau = {tau:+.3f}   p-value = {p_value:.4f}")
    print(f"       Trend Direction: {direction}")
    if p_value < 0.05:
        print("       [RESULT] Statistically significant at alpha = 0.05")
    else:
        print("       [RESULT] No statistically significant trend at alpha = 0.05")

    print("\n[INFO] Computing Sen's Slope (robust median-based trend rate)...")
    slope_sens, intercept_sens = sens_slope(years, precip)
    print(f"       Sen's Slope : {slope_sens:+.2f} mm / year")
    print(f"       30-yr Change: {slope_sens * 30:+.0f} mm total")

    ols_slope, ols_intercept, *_ = stats.linregress(years, precip)

    annual['sens_trend_mm'] = slope_sens * years + intercept_sens
    annual['ols_trend_mm']  = ols_slope  * years + ols_intercept
    csv_path = os.path.join(OUT_DIR, "annual_precipitation_with_trend.csv")
    annual[['year', 'precip_mm', 'sens_trend_mm', 'ols_trend_mm']].to_csv(csv_path, index=False)
    print(f"\n       [SUCCESS] Trend CSV exported: {csv_path}")

    plot_trend(annual, slope_sens, intercept_sens, ols_slope, ols_intercept,
               tau, p_value, direction)
    print("\n[SUCCESS] Chapter 1 Trend Analysis (Tier 4) Complete!")


if __name__ == "__main__":
    main()
