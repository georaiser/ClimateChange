"""
07_precipitation_anomaly.py
============================
GeoCascade Chapter 01 -- Torres del Paine, Patagonia Climate Analysis
Improves 04_precipitation_anomaly.py.

PURPOSE
-------
Loads ERA5 monthly CSV and computes monthly precipitation anomalies against a
1993-2023 climatological baseline.  Applies K-Means clustering (k=4) to
identify anomaly regimes (Drought / Dry / Normal / Wet) and evaluates a simple
ENSO correlation using a SOI-style sign proxy derived from the anomaly pattern.

OUTPUTS
-------
1. data/processed/climate_maps/precipitation_anomaly_clusters.png
   4-panel figure:
     Panel 1 -- Monthly anomaly time series coloured by cluster
     Panel 2 -- Seasonal anomaly patterns by cluster (box-plot)
     Panel 3 -- Annual cycle of each cluster mean
     Panel 4 -- Cluster membership heatmap  (year x month)

2. data/processed/climate_maps/anomaly_validation_plot.png
   Classic anomaly validation plot (bar chart + 12-month rolling mean).

3. data/processed/climate_analysis/precipitation_clusters.csv
   Columns: date, anomaly_mm, cluster_id, cluster_name

USAGE
-----
    python 07_precipitation_anomaly.py

NOTE FOR ArcGIS Pro
-------------------
Load precipitation_clusters.csv in ArcGIS Pro to create a temporal anomaly
chart.  Use the cluster_name field as a category field for symbology.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_RAW   = os.path.join(BASE_DIR, "data", "raw", "real_data")
DATA_PROC  = os.path.join(BASE_DIR, "data", "processed")
MAPS_DIR   = os.path.join(DATA_PROC, "climate_maps")
ANALY_DIR  = os.path.join(DATA_PROC, "climate_analysis")

ERA5_MONTHLY_CSV = os.path.join(DATA_RAW, "era5_monthly_patagonia.csv")

OUT_CLUSTER_PNG    = os.path.join(MAPS_DIR,  "precipitation_anomaly_clusters.png")
OUT_VALIDATION_PNG = os.path.join(MAPS_DIR,  "anomaly_validation_plot.png")
OUT_CLUSTER_CSV    = os.path.join(ANALY_DIR, "precipitation_clusters.csv")

os.makedirs(MAPS_DIR,  exist_ok=True)
os.makedirs(ANALY_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# CLUSTER LABELS (ordered Drought -> Wet by mean anomaly)
# ---------------------------------------------------------------------------
CLUSTER_NAMES  = ["Drought", "Dry", "Normal", "Wet"]
CLUSTER_COLORS = ["#8B1A1A", "#E8A838", "#4CAF82", "#1565C0"]
N_CLUSTERS     = 4

BASELINE_START = 1993
BASELINE_END   = 2023


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def load_era5_monthly(path: str) -> pd.DataFrame:
    """Load ERA5 monthly CSV and return a clean DataFrame with a DatetimeIndex.

    Handles two layouts:
      A) Single date column  (date / time / year_month / datetime)
      B) Split year + month  integer columns (as produced by 01_data_download.py)
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    # --- build DatetimeIndex ---
    # Layout B: separate year + month columns (no combined date column)
    if "year" in df.columns and "month" in df.columns and "date" not in df.columns:
        df["date"] = pd.to_datetime(
            df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01"
        )
        df = df.drop(columns=["year", "month"])
        df = df.set_index("date").sort_index()
    else:
        # Layout A: single date column
        date_col = None
        for c in ["date", "time", "year_month", "datetime"]:
            if c in df.columns:
                date_col = c
                break
        if date_col is None:
            date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.set_index(date_col).sort_index()

    # --- detect precipitation column ---
    precip_col = None
    for candidate in ["precip_sum", "tp", "precip", "precipitation", "prcp",
                       "rain", "total_precipitation", "monthly_precip",
                       "precipitation_sum"]:
        if candidate in df.columns:
            precip_col = candidate
            break
    if precip_col is None:
        # Last resort: first numeric column that is not a time component
        skip = {"year", "month", "day", "hour"}
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                        if c not in skip]
        if not numeric_cols:
            raise ValueError(f"No precipitation column found in {path}. "
                             f"Columns: {df.columns.tolist()}")
        precip_col = numeric_cols[0]
        print(f"  [INFO] Precipitation column auto-detected: '{precip_col}'")
    else:
        print(f"  [INFO] Precipitation column: '{precip_col}'")

    # ERA5 delivers tp in metres; convert to mm if max < 5 (heuristic)
    series = df[precip_col].copy()
    if series.dropna().max() < 5:
        print("  [INFO] Values look like metres -> converting to mm (*1000)")
        series = series * 1000.0

    out = pd.DataFrame({"precip_mm": series})
    out.index.name = "date"
    return out


def compute_anomalies(df: pd.DataFrame,
                      start_year: int,
                      end_year: int) -> pd.DataFrame:
    """Compute monthly anomalies against the climatological baseline."""
    baseline = df[(df.index.year >= start_year) & (df.index.year <= end_year)]
    if baseline.empty or baseline["precip_mm"].isna().all():
        raise ValueError(
            f"No valid data in baseline period {start_year}-{end_year}. "
            f"Data spans {df.index.year.min()}-{df.index.year.max()}."
        )
    monthly_clim = baseline.groupby(baseline.index.month)["precip_mm"].mean()

    df = df.copy()
    df["month"]      = df.index.month
    df["clim_mm"]    = df["month"].map(monthly_clim)
    df["anomaly_mm"] = df["precip_mm"] - df["clim_mm"]
    df["year"]       = df.index.year
    return df


def apply_kmeans(df: pd.DataFrame, n_clusters: int = 4):
    """Fit K-Means on scaled anomaly values and return labelled df + model."""
    # Drop rows where anomaly is NaN before clustering
    df_clean = df.dropna(subset=["anomaly_mm"])
    if len(df_clean) < n_clusters:
        raise ValueError(
            f"Only {len(df_clean)} valid anomaly rows -- need at least {n_clusters}."
        )

    scaler = StandardScaler()
    X = scaler.fit_transform(df_clean[["anomaly_mm"]])

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    raw_labels = km.fit_predict(X)

    # Re-order cluster IDs by ascending centroid (Drought=0, Wet=3)
    centroid_order = np.argsort(km.cluster_centers_[:, 0])
    remap  = {old: new for new, old in enumerate(centroid_order)}
    labels = np.array([remap[l] for l in raw_labels])

    df_clean = df_clean.copy()
    df_clean["cluster_id"]   = labels
    df_clean["cluster_name"] = [CLUSTER_NAMES[i] for i in labels]
    return df_clean, km


# ---------------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------------

def plot_4panel(df: pd.DataFrame, out_path: str) -> None:
    """Generate the 4-panel precipitation anomaly cluster figure."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12),
                             facecolor="#0d1117")
    fig.suptitle(
        "Torres del Paine -- Precipitation Anomaly Clusters (1993-2023)\n"
        "K-Means k=4  |  Baseline: 1993-2023 monthly climatology",
        color="#e6edf3", fontsize=14, fontweight="bold", y=1.01
    )

    text_color = "#e6edf3"
    grid_color = "#30363d"
    for ax in axes.flat:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors=text_color)
        ax.xaxis.label.set_color(text_color)
        ax.yaxis.label.set_color(text_color)
        ax.title.set_color(text_color)
        for spine in ax.spines.values():
            spine.set_edgecolor(grid_color)

    months_abbr = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]

    # ---- Panel 1: time series coloured by cluster -------------------------
    ax1 = axes[0, 0]
    for cid, cname in enumerate(CLUSTER_NAMES):
        mask   = df["cluster_id"] == cid
        subset = df[mask]
        ax1.scatter(subset.index, subset["anomaly_mm"],
                    color=CLUSTER_COLORS[cid], s=20, label=cname, zorder=3)
    roll = df["anomaly_mm"].rolling(12, center=True).mean()
    ax1.plot(df.index, roll, color="#e6edf3", lw=1.5,
             linestyle="--", label="12-mo rolling mean", zorder=4)
    ax1.axhline(0, color="#30363d", lw=0.8)
    ax1.set_title("Monthly Anomaly Time Series (coloured by cluster)")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Anomaly (mm)")
    ax1.legend(fontsize=8, framealpha=0.3,
               labelcolor=text_color, facecolor="#161b22")
    ax1.grid(True, color=grid_color, alpha=0.4)

    # ---- Panel 2: Seasonal box-plot by cluster ----------------------------
    ax2 = axes[0, 1]
    seasons = {
        "DJF": [12, 1, 2],
        "MAM": [3, 4, 5],
        "JJA": [6, 7, 8],
        "SON": [9, 10, 11]
    }
    season_labels = list(seasons.keys())
    x_positions = np.arange(len(season_labels))
    width   = 0.18
    offsets = np.linspace(-0.27, 0.27, N_CLUSTERS)

    for cid, cname in enumerate(CLUSTER_NAMES):
        subset = df[df["cluster_id"] == cid]
        season_vals = []
        for s, months in seasons.items():
            vals = subset[subset["month"].isin(months)]["anomaly_mm"].dropna()
            season_vals.append(vals.values if len(vals) > 0 else [0])
        positions = x_positions + offsets[cid]
        ax2.boxplot(season_vals, positions=positions, widths=width,
                    patch_artist=True,
                    boxprops=dict(facecolor=CLUSTER_COLORS[cid], alpha=0.7),
                    medianprops=dict(color="white", lw=1.5),
                    whiskerprops=dict(color=CLUSTER_COLORS[cid]),
                    capprops=dict(color=CLUSTER_COLORS[cid]),
                    flierprops=dict(marker="o", markersize=2,
                                    markerfacecolor=CLUSTER_COLORS[cid]))
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(season_labels)
    ax2.axhline(0, color=grid_color, lw=0.8)
    ax2.set_title("Seasonal Anomaly Patterns by Cluster")
    ax2.set_xlabel("Season")
    ax2.set_ylabel("Anomaly (mm)")
    ax2.grid(True, color=grid_color, alpha=0.4)
    legend_patches = [mpatches.Patch(color=CLUSTER_COLORS[i], label=CLUSTER_NAMES[i])
                      for i in range(N_CLUSTERS)]
    ax2.legend(handles=legend_patches, fontsize=8, framealpha=0.3,
               labelcolor=text_color, facecolor="#161b22")

    # ---- Panel 3: Annual cycle mean by cluster ----------------------------
    ax3 = axes[1, 0]
    for cid, cname in enumerate(CLUSTER_NAMES):
        subset = df[df["cluster_id"] == cid]
        monthly_mean = subset.groupby("month")["anomaly_mm"].mean()
        full = pd.Series(index=range(1, 13), dtype=float)
        full.update(monthly_mean)
        ax3.plot(range(1, 13), full.values,
                 color=CLUSTER_COLORS[cid], lw=2, marker="o",
                 markersize=5, label=cname)
    ax3.axhline(0, color=grid_color, lw=0.8)
    ax3.set_xticks(range(1, 13))
    ax3.set_xticklabels(months_abbr, rotation=45, ha="right")
    ax3.set_title("Annual Cycle of Cluster Mean Anomaly")
    ax3.set_xlabel("Month")
    ax3.set_ylabel("Mean Anomaly (mm)")
    ax3.legend(fontsize=8, framealpha=0.3,
               labelcolor=text_color, facecolor="#161b22")
    ax3.grid(True, color=grid_color, alpha=0.4)

    # ---- Panel 4: Heatmap year x month ------------------------------------
    ax4 = axes[1, 1]
    years   = sorted(df["year"].unique())
    heatmap = np.full((len(years), 12), np.nan)
    for yi, yr in enumerate(years):
        for mi, mo in enumerate(range(1, 13)):
            subset = df[(df["year"] == yr) & (df["month"] == mo)]
            if len(subset) > 0:
                heatmap[yi, mi] = subset["cluster_id"].values[0]

    cmap = ListedColormap(CLUSTER_COLORS)
    im   = ax4.imshow(heatmap, aspect="auto", cmap=cmap,
                      vmin=-0.5, vmax=3.5, origin="upper",
                      interpolation="nearest")
    ax4.set_xticks(range(12))
    ax4.set_xticklabels(months_abbr, rotation=45, ha="right")
    ax4.set_yticks(range(len(years)))
    ax4.set_yticklabels([str(y) if y % 5 == 0 else "" for y in years],
                        fontsize=7)
    ax4.set_title("Cluster Membership Heatmap (Year x Month)")
    ax4.set_xlabel("Month")
    ax4.set_ylabel("Year")
    cbar = plt.colorbar(im, ax=ax4, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(CLUSTER_NAMES, color=text_color)
    cbar.ax.yaxis.set_tick_params(color=text_color)

    plt.tight_layout(pad=2.0)
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [OK] Saved 4-panel cluster figure -> {out_path}")


def plot_validation(df: pd.DataFrame, out_path: str) -> None:
    """Classic anomaly bar chart with 12-month rolling mean overlay."""
    fig, ax = plt.subplots(figsize=(14, 5), facecolor="#0d1117")
    ax.set_facecolor("#161b22")
    text_color = "#e6edf3"
    grid_color = "#30363d"

    pos_mask = df["anomaly_mm"] >= 0
    neg_mask = df["anomaly_mm"] < 0

    ax.bar(df.index[pos_mask], df.loc[pos_mask, "anomaly_mm"],
           color="#1565C0", alpha=0.8, width=25, label="Positive anomaly")
    ax.bar(df.index[neg_mask], df.loc[neg_mask, "anomaly_mm"],
           color="#8B1A1A", alpha=0.8, width=25, label="Negative anomaly")

    roll = df["anomaly_mm"].rolling(12, center=True).mean()
    ax.plot(df.index, roll, color="#FFD700", lw=2,
            label="12-mo rolling mean")
    ax.axhline(0, color=grid_color, lw=0.8)

    ax.set_title("Torres del Paine -- Monthly Precipitation Anomaly\n"
                 "(1993-2023 baseline)",
                 color=text_color, fontsize=13)
    ax.set_xlabel("Date", color=text_color)
    ax.set_ylabel("Anomaly (mm)", color=text_color)
    ax.tick_params(colors=text_color)
    for spine in ax.spines.values():
        spine.set_edgecolor(grid_color)
    ax.grid(True, color=grid_color, alpha=0.4)
    ax.legend(fontsize=9, framealpha=0.3,
              labelcolor=text_color, facecolor="#161b22")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [OK] Saved validation plot -> {out_path}")


# ---------------------------------------------------------------------------
# ENSO PROXY CORRELATION
# ---------------------------------------------------------------------------

def compute_enso_proxy(df: pd.DataFrame) -> None:
    """
    Simple ENSO proxy: define El Nino months as those with anomaly > +1 std,
    La Nina months as anomaly < -1 std.  Count co-occurrences with
    Drought / Wet clusters and report to console.

    NOTE: For a proper ENSO analysis use the actual MEI or SOI index
    downloaded from NOAA PSL.  This proxy is for educational illustration.
    """
    std_val  = df["anomaly_mm"].std()
    el_nino  = df["anomaly_mm"] > std_val
    la_nina  = df["anomaly_mm"] < -std_val
    drought  = df["cluster_name"] == "Drought"
    wet      = df["cluster_name"] == "Wet"

    n_el_nino    = el_nino.sum()
    n_la_nina    = la_nina.sum()
    drought_nina = (drought & la_nina).sum()
    wet_nino     = (wet & el_nino).sum()

    print("\n  [ENSO PROXY CORRELATION]")
    print(f"  El Nino-like months (anomaly > +1 std={std_val:.1f} mm): {n_el_nino}")
    print(f"  La Nina-like months (anomaly < -1 std):                  {n_la_nina}")
    print(f"  Drought months co-occurring with La Nina-like:           {drought_nina}")
    print(f"  Wet months co-occurring with El Nino-like:               {wet_nino}")
    if n_la_nina > 0:
        pct = 100.0 * drought_nina / n_la_nina
        print(f"  -> {pct:.1f}% of La Nina-like months are in Drought cluster")
    if n_el_nino > 0:
        pct2 = 100.0 * wet_nino / n_el_nino
        print(f"  -> {pct2:.1f}% of El Nino-like months are in Wet cluster")
    print("  NOTE: Use actual MEI/SOI index from NOAA PSL for publication-quality analysis.")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Script 07 -- Precipitation Anomaly Clustering")
    print("=" * 60)

    print(f"\n[1/5] Loading ERA5 monthly data from:\n      {ERA5_MONTHLY_CSV}")
    if not os.path.exists(ERA5_MONTHLY_CSV):
        raise FileNotFoundError(
            f"ERA5 monthly CSV not found: {ERA5_MONTHLY_CSV}\n"
            "Run the download script first."
        )
    df_raw = load_era5_monthly(ERA5_MONTHLY_CSV)
    print(f"      Loaded {len(df_raw)} monthly records "
          f"({df_raw.index.min().year} - {df_raw.index.max().year})")

    print(f"\n[2/5] Computing anomalies vs {BASELINE_START}-{BASELINE_END} baseline ...")
    df = compute_anomalies(df_raw, BASELINE_START, BASELINE_END)
    print(f"      Anomaly range: {df['anomaly_mm'].min():.1f} mm to "
          f"{df['anomaly_mm'].max():.1f} mm")

    print(f"\n[3/5] Applying K-Means clustering (k={N_CLUSTERS}) ...")
    df, km_model = apply_kmeans(df, N_CLUSTERS)

    print("\n  Cluster summary:")
    for cid, cname in enumerate(CLUSTER_NAMES):
        subset = df[df["cluster_id"] == cid]
        n      = len(subset)
        mean_a = subset["anomaly_mm"].mean()
        print(f"  Cluster {cid} ({cname}): {n} months, "
              f"mean anomaly = {mean_a:+.1f} mm")

    compute_enso_proxy(df)

    print(f"\n[4/5] Saving precipitation_clusters.csv ...")
    out_df = df[["anomaly_mm", "cluster_id", "cluster_name"]].copy()
    out_df.index.name = "date"
    out_df.to_csv(OUT_CLUSTER_CSV, encoding="utf-8")
    print(f"  [OK] {OUT_CLUSTER_CSV}")

    print(f"\n[5/5] Generating figures ...")
    plot_4panel(df, OUT_CLUSTER_PNG)
    plot_validation(df, OUT_VALIDATION_PNG)

    print("\n" + "=" * 60)
    print("DONE -- Script 07 complete.")
    print(f"  Cluster figure  : {OUT_CLUSTER_PNG}")
    print(f"  Validation plot : {OUT_VALIDATION_PNG}")
    print(f"  Cluster CSV     : {OUT_CLUSTER_CSV}")
    print("\nNOTE: Load precipitation_clusters.csv in ArcGIS Pro to create")
    print("      a temporal anomaly chart.  Use cluster_name as category field.")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[ERROR] Script 07 failed: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
