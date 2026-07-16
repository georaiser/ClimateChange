"""
Chapter 1: 04_climate_trend_analysis.py
========================================
Mann-Kendall Non-Parametric Trend Test + Sen's Slope Estimator
applied to all ERA5-Land variables (1993-2024).

Academic Objective:
  Detect statistically significant monotonic trends in 32 years of daily
  climate data without assuming normality. The Mann-Kendall test is the
  standard method in climate science for trend detection.

Statistical Methods:
  Mann-Kendall S-statistic:
      S = sum_{k<j} sgn(x_j - x_k)
  Under H0 (no trend), Z = S / sqrt(Var(S)) ~ Normal(0,1)
  Sen's Slope = median of all pairwise slopes (x_j-x_k)/(j-k)

Outputs:
  data/processed/climate_analysis/trend_summary.csv
  data/processed/climate_analysis/climate_trends_multivar.png
  data/processed/climate_analysis/temperature_seasonal_heatmap.png

Run:
  conda activate geocascade_env
  python Chapter_01/04_climate_trend_analysis.py

ArcGIS Pro: Load trend_summary.csv via Insert > Add Data > Table, then
            right-click > Charts > Bar Chart to visualise per-variable trends.
ENVI:       Not applicable (tabular analysis only).

Dependencies: pandas, numpy, matplotlib, scipy
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
RAW_DIR   = os.path.join(BASE_DIR, "data", "raw",  "real_data")
PROC_DIR  = os.path.join(BASE_DIR, "data", "processed", "climate_analysis")
os.makedirs(PROC_DIR, exist_ok=True)

ERA5_DAILY   = os.path.join(RAW_DIR, "era5_daily_patagonia.csv")
ERA5_MONTHLY = os.path.join(RAW_DIR, "era5_monthly_patagonia.csv")

DARK_BG  = "#0d1117"
DARK_AX  = "#161b22"
C_TEXT   = "#e6edf3"
C_GREY   = "#8b949e"
C_RED    = "#e74c3c"
C_BLUE   = "#3498db"
C_GREEN  = "#2ecc71"
C_GOLD   = "#f39c12"


# ---------------------------------------------------------------------------
# 1. Mann-Kendall + Sen's Slope (pure scipy, no external package)
# ---------------------------------------------------------------------------
def mann_kendall(x):
    """
    Non-parametric trend test.
    Returns: tau, p_value, sens_slope_per_year, trend_direction
    """
    n = len(x)
    if n < 4:
        return np.nan, np.nan, np.nan, "insufficient"

    # S statistic
    s = 0
    for k in range(n - 1):
        for j in range(k + 1, n):
            diff = x[j] - x[k]
            if diff > 0:
                s += 1
            elif diff < 0:
                s -= 1

    # Variance of S (no tie correction for simplicity)
    var_s = n * (n - 1) * (2 * n + 5) / 18.0

    # Z statistic
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0

    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    tau = s / (0.5 * n * (n - 1))

    # Sen's slope: median of all pairwise slopes
    slopes = []
    for k in range(n - 1):
        for j in range(k + 1, n):
            if j != k:
                slopes.append((x[j] - x[k]) / (j - k))
    sens_slope = float(np.median(slopes)) if slopes else np.nan

    direction = "increasing" if s > 0 else ("decreasing" if s < 0 else "no trend")
    return tau, p_value, sens_slope, direction


# ---------------------------------------------------------------------------
# 2. Load and aggregate ERA5 to annual means
# ---------------------------------------------------------------------------
def load_era5():
    if not os.path.exists(ERA5_DAILY):
        raise FileNotFoundError(
            "ERA5 daily CSV not found: " + ERA5_DAILY +
            "\nRun 01_data_download.py first."
        )
    daily = pd.read_csv(ERA5_DAILY, parse_dates=["date"], index_col="date")
    n_days = len(daily)
    yr_start = daily.index.min().year
    yr_end   = daily.index.max().year
    print(f"  ERA5 loaded : {n_days:,} days  ({yr_start} - {yr_end})")
    print(f"  Columns     : {list(daily.columns)}")
    return daily


def annual_means(daily):
    """Resample daily to annual, use mean for temperature/rates, sum for precip/snow."""
    sum_vars = [c for c in daily.columns if any(k in c for k in
                ["precipitation", "snowfall", "radiation", "et0"])]
    mean_vars = [c for c in daily.columns if c not in sum_vars]

    ann_mean = daily[mean_vars].resample("YE").mean() if mean_vars else pd.DataFrame()
    ann_sum  = daily[sum_vars].resample("YE").sum()  if sum_vars  else pd.DataFrame()

    annual = pd.concat([ann_mean, ann_sum], axis=1)
    annual.index = annual.index.year
    return annual


# ---------------------------------------------------------------------------
# 3. Run trend analysis on all variables
# ---------------------------------------------------------------------------
def compute_all_trends(annual):
    results = []
    for col in annual.columns:
        series = annual[col].dropna()
        if len(series) < 5:
            continue
        x    = series.values.astype(float)
        yrs  = series.index.values.astype(float)

        tau, pval, sens_yr, direction = mann_kendall(x)
        sens_decade = sens_yr * 10 if not np.isnan(sens_yr) else np.nan

        # OLS for comparison
        slope_ols, intercept, r_val, _, _ = stats.linregress(yrs, x)

        significant = bool(pval < 0.05) if not np.isnan(pval) else False

        results.append({
            "variable":           col,
            "n_years":            int(len(series)),
            "mean":               round(float(np.mean(x)), 4),
            "std":                round(float(np.std(x)), 4),
            "sens_slope_per_decade": round(float(sens_decade), 5) if not np.isnan(sens_decade) else None,
            "sens_slope_per_year":   round(float(sens_yr), 6)     if not np.isnan(sens_yr)    else None,
            "mk_tau":             round(float(tau), 4)  if not np.isnan(tau) else None,
            "mk_pvalue":          round(float(pval), 5) if not np.isnan(pval) else None,
            "significant":        significant,
            "trend_direction":    direction,
            "ols_slope_per_year": round(float(slope_ols), 6),
            "r_squared":          round(float(r_val ** 2), 4),
        })

        sig_str = "[SIGNIFICANT]" if significant else ""
        slope_str = f"{sens_decade:+.4f}/decade" if sens_decade and not np.isnan(sens_decade) else "N/A"
        print(f"    {col:<45s}  {slope_str:<22s}  p={pval:.4f}  {sig_str}")

    df = pd.DataFrame(results).sort_values("mk_pvalue", na_position="last")
    return df


# ---------------------------------------------------------------------------
# 4. Multi-variable trend figure (4-panel)
# ---------------------------------------------------------------------------
def plot_trends(annual, trend_df):
    print("\n  Building climate trends figure...")

    fig = plt.figure(figsize=(20, 16), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32,
                            top=0.93, bottom=0.07, left=0.07, right=0.97)

    def style_ax(ax, title=""):
        ax.set_facecolor(DARK_AX)
        for sp in ax.spines.values():
            sp.set_color("#30363d")
        ax.tick_params(colors=C_TEXT, labelsize=9)
        ax.xaxis.label.set_color(C_TEXT)
        ax.yaxis.label.set_color(C_TEXT)
        ax.grid(alpha=0.18, color="#30363d")
        if title:
            ax.set_title(title, color=C_TEXT, fontsize=10, fontweight="bold", pad=6)

    fig.text(0.5, 0.97, "GeoCascade - ERA5 Climate Trend Analysis (1993-2024)",
             ha="center", color=C_TEXT, fontsize=14, fontweight="bold")
    fig.text(0.5, 0.945, "Mann-Kendall + Sen's Slope | Torres del Paine, Patagonia",
             ha="center", color=C_GREY, fontsize=9)

    # ── Panel 1: Temperature ─────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    temp_col = next((c for c in ["temperature_2m_mean", "temperature_2m"] if c in annual.columns), None)
    if temp_col:
        t = annual[temp_col].dropna()
        yrs = t.index.values.astype(float)
        ax1.fill_between(t.index, t.values, t.mean(),
                         where=t.values >= t.mean(), alpha=0.3, color=C_RED)
        ax1.fill_between(t.index, t.values, t.mean(),
                         where=t.values < t.mean(),  alpha=0.3, color=C_BLUE)
        ax1.plot(t.index, t.values, "o-", color=C_RED, lw=1.8, ms=4)
        z   = np.polyfit(yrs, t.values, 1)
        fit = np.poly1d(z)(yrs)
        ax1.plot(t.index, fit, "--", color=C_GOLD, lw=2,
                 label=f"Trend {z[0]*10:+.3f} deg/decade")
        ax1.axhline(t.mean(), color=C_GREY, lw=0.8, ls=":")
        ax1.legend(fontsize=8, facecolor=DARK_BG, labelcolor=C_TEXT)
        ax1.set_ylabel("Temperature (deg C)", color=C_TEXT, fontsize=9)
    style_ax(ax1, "Annual Mean Temperature + Trend")

    # ── Panel 2: Precipitation anomaly bars ─────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    prec_col = next((c for c in ["precipitation_sum", "precipitation"] if c in annual.columns), None)
    if prec_col:
        p = annual[prec_col].dropna()
        z_scores = (p.values - p.mean()) / p.std()
        bar_c = [C_RED if z < -0.5 else C_BLUE if z > 0.5 else C_GREY for z in z_scores]
        ax2.bar(p.index, z_scores, color=bar_c, alpha=0.8, width=0.8)
        ax2.axhline(0,  color=C_GOLD, lw=1.2, ls="--")
        ax2.axhline(-1, color=C_RED,  lw=0.7, ls=":", alpha=0.6)
        ax2.axhline(+1, color=C_BLUE, lw=0.7, ls=":", alpha=0.6)
        ax2.set_ylabel("Precipitation anomaly (z-score)", color=C_TEXT, fontsize=9)
        for yr, z in zip(p.index, z_scores):
            if abs(z) > 1.5:
                ax2.text(yr, z + (0.12 if z > 0 else -0.2), str(yr),
                         ha="center", fontsize=7, color=C_TEXT)
    style_ax(ax2, "Annual Precipitation Anomaly (Z-Score)")

    # ── Panel 3: Wind + Snowfall dual axis ───────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    wind_col = next((c for c in ["wind_speed_10m_max", "windspeed_10m_max"] if c in annual.columns), None)
    if wind_col:
        w = annual[wind_col].dropna()
        ax3.plot(w.index, w.values, color=C_GREEN, lw=1.8, label="Wind max (km/h)")
    snow_col = next((c for c in ["snowfall_sum", "snowfall"] if c in annual.columns), None)
    if snow_col:
        s = annual[snow_col].dropna()
        ax3b = ax3.twinx()
        ax3b.bar(s.index, s.values, color="#00bcd4", alpha=0.35, width=0.8, label="Snowfall (cm)")
        ax3b.set_ylabel("Snowfall (cm/year)", color="#00bcd4", fontsize=9)
        ax3b.tick_params(colors="#00bcd4")
        for sp in ax3b.spines.values():
            sp.set_color("#30363d")
    ax3.set_ylabel("Wind Speed (km/h)", color=C_GREEN, fontsize=9)
    ax3.legend(fontsize=8, facecolor=DARK_BG, labelcolor=C_TEXT, loc="upper left")
    style_ax(ax3, "Wind Speed & Snowfall")

    # ── Panel 4: Trend summary table ─────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(DARK_AX)
    ax4.axis("off")
    ax4.set_title("Mann-Kendall Trend Summary", color=C_TEXT,
                  fontsize=10, fontweight="bold", pad=6)

    sig_rows = trend_df[trend_df["significant"] == True].head(8)
    y = 0.93
    ax4.text(0.02, y, "Variable", color=C_GREY, fontsize=8, transform=ax4.transAxes)
    ax4.text(0.55, y, "Slope/decade", color=C_GREY, fontsize=8, transform=ax4.transAxes)
    ax4.text(0.82, y, "p-value", color=C_GREY, fontsize=8, transform=ax4.transAxes)
    y -= 0.06
    ax4.axhline(y=y + 0.005, xmin=0.02, xmax=0.98, color="#30363d", lw=0.8,
                transform=ax4.transAxes)

    for _, row in sig_rows.iterrows():
        label = str(row["variable"])[:35]
        slope = row["sens_slope_per_decade"]
        pval  = row["mk_pvalue"]
        slope_str = f"{slope:+.4f}" if slope is not None else "N/A"
        pval_str  = f"{pval:.4f}"   if pval  is not None else "N/A"
        c = C_RED if (slope is not None and slope > 0) else C_BLUE
        ax4.text(0.02, y, label,      color=C_TEXT, fontsize=7.5, transform=ax4.transAxes)
        ax4.text(0.55, y, slope_str,  color=c,      fontsize=7.5, transform=ax4.transAxes)
        ax4.text(0.82, y, pval_str,   color=C_GOLD, fontsize=7.5, transform=ax4.transAxes)
        y -= 0.085

    out_png = os.path.join(PROC_DIR, "climate_trends_multivar.png")
    fig.savefig(out_png, dpi=180, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  [OK] Trends figure: {out_png}")
    return out_png


# ---------------------------------------------------------------------------
# 5. Seasonal heatmap (month x year) for temperature
# ---------------------------------------------------------------------------
def plot_seasonal_heatmap(daily):
    print("  Building seasonal temperature heatmap...")
    temp_col = next((c for c in ["temperature_2m_mean", "temperature_2m"] if c in daily.columns), None)
    if temp_col is None:
        print("  [SKIP] No temperature column found for heatmap.")
        return

    monthly = daily[temp_col].resample("ME").mean()
    years   = sorted(monthly.index.year.unique())
    months  = list(range(1, 13))
    grid    = np.full((12, len(years)), np.nan)

    for i, yr in enumerate(years):
        for j, mo in enumerate(months):
            vals = monthly[(monthly.index.year == yr) & (monthly.index.month == mo)]
            if len(vals):
                grid[j, i] = vals.iloc[0]

    # anomaly relative to monthly climatology
    clim = np.nanmean(grid, axis=1, keepdims=True)
    anom = grid - clim

    fig, ax = plt.subplots(figsize=(18, 5), facecolor=DARK_BG)
    ax.set_facecolor(DARK_AX)
    im = ax.imshow(anom, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3,
                   extent=[years[0] - 0.5, years[-1] + 0.5, 12.5, 0.5])
    cb = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("Temperature anomaly (deg C)", color=C_TEXT, fontsize=9)
    cb.ax.tick_params(colors=C_TEXT, labelsize=8)

    mon_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    ax.set_yticks(range(1, 13))
    ax.set_yticklabels(mon_labels, color=C_TEXT, fontsize=9)
    ax.tick_params(colors=C_TEXT, labelsize=9)
    for sp in ax.spines.values():
        sp.set_color("#30363d")

    ax.set_title("ERA5 Temperature Anomaly Heatmap (month x year) - Torres del Paine",
                 color=C_TEXT, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Year", color=C_TEXT, fontsize=9)

    out_png = os.path.join(PROC_DIR, "temperature_seasonal_heatmap.png")
    fig.savefig(out_png, dpi=180, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  [OK] Heatmap: {out_png}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print(" GEOCASCADE - ERA5 TREND ANALYSIS")
    print(" Mann-Kendall + Sen's Slope on all climate variables")
    print("=" * 65)

    print("\n[1/4] Loading ERA5 data...")
    try:
        daily = load_era5()
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        return

    print("\n[2/4] Aggregating to annual values...")
    annual = annual_means(daily)
    print(f"  Annual data: {len(annual)} years x {len(annual.columns)} variables")

    print("\n[3/4] Running Mann-Kendall trend tests...")
    trend_df = compute_all_trends(annual)

    csv_path = os.path.join(PROC_DIR, "trend_summary.csv")
    trend_df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\n  [OK] Trend summary CSV: {csv_path}")

    # Print summary table
    sig = trend_df[trend_df["significant"] == True]
    print(f"\n  Significant trends (p < 0.05): {len(sig)} of {len(trend_df)} variables")
    if not sig.empty:
        print(f"  {'Variable':<45s}  {'Slope/decade':<14s}  p-value")
        print("  " + "-" * 70)
        for _, row in sig.iterrows():
            sl = row['sens_slope_per_decade']
            sl_str = f"{sl:+.4f}" if sl is not None else "N/A"
            pv = row['mk_pvalue']
            pv_str = f"{pv:.5f}" if pv is not None else "N/A"
            print(f"  {str(row['variable']):<45s}  {sl_str:<14s}  {pv_str}")

    print("\n[4/4] Generating figures...")
    plot_trends(annual, trend_df)
    plot_seasonal_heatmap(daily)

    print("\n" + "=" * 65)
    print(" TREND ANALYSIS COMPLETE")
    print("=" * 65)
    print(f"  CSV : {csv_path}")
    print(f"  PNGs: {PROC_DIR}")
    print()
    print("  ArcGIS Pro: Insert > Add Data > Table > trend_summary.csv")
    print("              Right-click table > Charts > Bar Chart")
    print("  ENVI      : N/A (tabular output, load CSV externally)")
    print("=" * 65)


if __name__ == "__main__":
    main()
