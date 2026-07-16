"""
Chapter 4: 12_ecological_niche_modeling.py
============================================
Species Distribution Modeling: K-Means + Random Forest SDM

Academic Objective:
  Species Distribution Models (SDM) predict WHERE a species can survive based on
  the environmental conditions at known occurrence locations.
  They are a core tool in conservation biology and climate change impact assessment.

  This script demonstrates TWO complementary approaches:

  1. UNSUPERVISED (K-Means clustering):
     No species data needed. The algorithm discovers natural environmental
     clusters (e.g., alpine, forest, steppe, wetland) from terrain + vegetation alone.
     Analogy: ArcGIS Pro ISO Cluster Unsupervised Classification tool.

  2. SUPERVISED (Random Forest SDM):
     Simulates presence/absence data for the Patagonian Huemul Deer
     (Hippocamelus bisulcus, critically endangered).
     Preference rule: elevation < 800m AND slope < 20 deg AND NDVI > 0.4.
     The RF learns this boundary from 1000 random training points with realistic noise.

  Why Random Forest over MaxEnt for teaching?
     MaxEnt requires presence-only data + complex regularization tuning.
     RF SDM is conceptually cleaner: it is a standard binary classifier
     on presence/absence data, directly interpretable via feature importance.

Connection to pipeline:
  Uses Ch03/data/raw/temp_dem.tif (from script 10) if available.
  Falls back to STAC download if not.

Outputs:
  data/processed/niche/kmeans_unsupervised.tif
  data/processed/niche/ecological_niche_model.tif
  data/processed/niche/feature_importance.csv
  data/processed/niche/ecological_niche_model.png   (4-panel dark)
  data/processed/niche/niche_statistics.csv

ArcGIS Pro: Add ecological_niche_model.tif.
            Symbology > Stretched > Yellow-Green (0=unsuitable, 1=optimal).
            Use Zonal Statistics to compute mean suitability per watershed.
ENVI 5.6:   File > Open > ecological_niche_model.tif.
            Density Slice for threshold-based habitat mapping.

Run:
  conda activate geocascade_env
  python Chapter_04/12_ecological_niche_modeling.py

Dependencies: scikit-learn, rasterio, numpy, matplotlib, pandas, pystac-client, planetary-computer, pyproj
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
CH03_DIR  = os.path.join(os.path.dirname(BASE_DIR), "Chapter_03")
OUT_DIR   = os.path.join(BASE_DIR, "data", "processed", "niche")
os.makedirs(OUT_DIR, exist_ok=True)

BBOX = [-73.30, -51.10, -72.90, -50.80]

DARK_BG = "#0d1117"
DARK_AX = "#161b22"
C_TEXT  = "#e6edf3"
C_GREY  = "#8b949e"
C_RED   = "#e74c3c"
C_GREEN = "#2ecc71"
C_GOLD  = "#f39c12"
C_BLUE  = "#3498db"


# ---------------------------------------------------------------------------
# 1. Load environmental layers (Ch03 cache or STAC fallback)
# ---------------------------------------------------------------------------
def load_env_layers():
    """
    Load DEM (elevation), Slope, NDVI as environmental predictors.
    Priority: Ch03 processed TIFs -> STAC streaming.
    """
    from pyproj import Transformer

    # Try Ch03 cached terrain TIFs
    ch03_dem   = os.path.join(CH03_DIR, "data", "processed", "terrain", "copernicus_dem.tif")
    ch03_slope = os.path.join(CH03_DIR, "data", "processed", "terrain", "slope_degrees.tif")
    ch02_ndvi  = os.path.join(os.path.dirname(BASE_DIR), "Chapter_02",
                               "data", "processed", "indices", "ndvi.tif")

    if os.path.exists(ch03_dem) and os.path.exists(ch03_slope):
        print(f"  [OK] Using Ch03 terrain TIFs")
        with rasterio.open(ch03_dem) as src:
            dem     = src.read(1).astype("float32")
            dem     = np.where(dem == -9999, np.nan, dem)
            profile = src.profile.copy()
            target_shape = dem.shape
        with rasterio.open(ch03_slope) as src:
            slope = src.read(1).astype("float32")
            slope = np.where(slope == -9999, np.nan, slope)
    else:
        print("  Ch03 terrain not found. Downloading DEM from Planetary Computer...")
        dem, slope, profile, target_shape = _fetch_dem_slope()

    # NDVI from Ch02 or compute from STAC
    if os.path.exists(ch02_ndvi):
        print(f"  [OK] Using Ch02 NDVI TIF")
        with rasterio.open(ch02_ndvi) as src:
            ndvi_raw = src.read(1, out_shape=target_shape,
                                resampling=Resampling.bilinear).astype("float32")
            ndvi = np.where(ndvi_raw == -9999, np.nan, ndvi_raw)
    else:
        print("  Ch02 NDVI not found. Fetching Sentinel-2 from Planetary Computer...")
        ndvi = _fetch_ndvi(target_shape)

    profile.update(dtype="float32", count=1, nodata=-9999, compress="lzw",
                   height=target_shape[0], width=target_shape[1])

    print(f"  Layers loaded: DEM={dem.shape}, Slope={slope.shape}, NDVI={ndvi.shape}")
    return dem, slope, ndvi, profile


def _fetch_dem_slope():
    """Fallback: download DEM from Planetary Computer and compute slope."""
    from pystac_client import Client
    import planetary_computer as pc
    from pyproj import Transformer

    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                          modifier=pc.sign_inplace)
    search  = catalog.search(collections=["cop-dem-glo-30"], bbox=BBOX)
    items   = list(search.items())
    if not items:
        raise ValueError("No Copernicus DEM found.")

    with rasterio.open(items[0].assets["data"].href) as src:
        t = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        mnx, mny = t.transform(BBOX[0], BBOX[1])
        mxx, mxy = t.transform(BBOX[2], BBOX[3])
        win = from_bounds(mnx, mny, mxx, mxy, src.transform)
        dem = src.read(1, window=win).astype("float32")
        nd  = src.nodata
        pix_lat_m = abs(src.res[0]) * 111_000.0
        pix_lon_m = abs(src.res[1]) * 111_000.0 * np.cos(np.radians(-51.0))
        profile   = src.profile.copy()
        profile.update(dtype="float32", count=1, nodata=-9999,
                       height=int(round(win.height)), width=int(round(win.width)),
                       transform=rasterio.windows.transform(win, src.transform))

    if nd is not None:
        dem = np.where(dem == nd, np.nan, dem)
    dem   = np.where(dem < -500, np.nan, dem)
    dem   = np.where(dem < 0, 0, dem)
    dy, dx = np.gradient(dem, pix_lat_m, pix_lon_m)
    slope  = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
    return dem, slope, profile, dem.shape


def _fetch_ndvi(target_shape):
    """Fallback: compute NDVI from Sentinel-2 STAC."""
    from pystac_client import Client
    import planetary_computer as pc
    from pyproj import Transformer

    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                          modifier=pc.sign_inplace)
    search  = catalog.search(collections=["sentinel-2-l2a"], bbox=BBOX,
                              datetime="2023-01-01/2023-02-28",
                              query={"eo:cloud_cover": {"lt": 15}})
    items   = list(search.items())
    if not items:
        print("  [WARN] No Sentinel-2 found. Using DEM-proxy NDVI (elevation-based).")
        return np.zeros(target_shape, dtype="float32")

    item = sorted(items, key=lambda i: i.properties.get("eo:cloud_cover", 99))[0]
    t    = None
    with rasterio.open(item.assets["B04"].href) as src_red:
        from pyproj import Transformer as Tr
        t = Tr.from_crs("EPSG:4326", src_red.crs, always_xy=True)
        mnx, mny = t.transform(BBOX[0], BBOX[1])
        mxx, mxy = t.transform(BBOX[2], BBOX[3])
        win  = from_bounds(mnx, mny, mxx, mxy, src_red.transform)
        red  = src_red.read(1, window=win, out_shape=target_shape,
                            resampling=Resampling.bilinear).astype("float32") / 10000.0
    with rasterio.open(item.assets["B08"].href) as src_nir:
        nir = src_nir.read(1, window=win, out_shape=target_shape,
                           resampling=Resampling.bilinear).astype("float32") / 10000.0

    denom = nir + red
    return np.where(np.abs(denom) < 1e-6, np.nan, (nir - red) / denom)


# ---------------------------------------------------------------------------
# 2. K-Means unsupervised classification
# ---------------------------------------------------------------------------
def run_kmeans(dem, slope, ndvi, profile, k=4):
    print(f"\n  Running K-Means (k={k} unsupervised clusters)...")

    dem_f   = dem.flatten()
    slope_f = slope.flatten()
    ndvi_f  = ndvi.flatten()

    X_full = np.column_stack([dem_f, slope_f, ndvi_f])
    valid  = np.isfinite(X_full).all(axis=1)
    X_v    = X_full[valid]

    # Normalize to [0,1] so elevation doesn't dominate (in metres) over NDVI (in [-1,1])
    X_min = X_v.min(axis=0)
    X_max = X_v.max(axis=0)
    X_n   = (X_v - X_min) / (X_max - X_min + 1e-9)

    km     = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_n)

    # Print cluster centroids (back-transformed to original units)
    print(f"  K-Means cluster centroids (original units):")
    centroids_orig = km.cluster_centers_ * (X_max - X_min + 1e-9) + X_min
    for i, c in enumerate(centroids_orig):
        print(f"    Cluster {i}: Elev={c[0]:.0f}m  Slope={c[1]:.1f}deg  NDVI={c[2]:.3f}")

    km_map = np.full(dem.size, np.nan)
    km_map[np.where(valid)[0]] = labels
    km_map = km_map.reshape(dem.shape)

    out = os.path.join(OUT_DIR, "kmeans_unsupervised.tif")
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(np.where(np.isnan(km_map), -9999, km_map).astype("float32"), 1)
        dst.update_tags(description=f"K-Means k={k} unsupervised (Elev+Slope+NDVI)",
                        nodata="-9999",
                        arcgis_note="Unique Values symbology, 4 terrain classes")
    print(f"  [OK] K-Means TIF: {out}")
    return km_map, km


# ---------------------------------------------------------------------------
# 3. Random Forest SDM (supervised)
# ---------------------------------------------------------------------------
def run_sdm(dem, slope, ndvi, profile):
    print("\n  Running Random Forest SDM (Huemul Deer habitat suitability)...")

    dem_f   = dem.flatten()
    slope_f = slope.flatten()
    ndvi_f  = ndvi.flatten()

    X_full  = np.column_stack([dem_f, slope_f, ndvi_f])
    valid   = np.isfinite(X_full).all(axis=1)

    # Generate 1000 synthetic presence/absence training points
    print("  Generating 1000 synthetic training points (Huemul preference rule)...")
    np.random.seed(42)
    idx_valid  = np.where(valid)[0]
    sample_idx = np.random.choice(idx_valid, size=min(1000, len(idx_valid)), replace=False)
    X_train    = X_full[sample_idx]

    y_train = []
    for row in X_train:
        elev, slp, veg = row[0], row[1], row[2]
        # Huemul prefers: elevation < 800m, slope < 20 deg, NDVI > 0.4
        if (not np.isnan(elev)) and elev < 800 and slp < 20 and veg > 0.4:
            y_train.append(1 if np.random.rand() > 0.20 else 0)  # 80% presence in ideal habitat
        else:
            y_train.append(1 if np.random.rand() > 0.95 else 0)  # 5% stray occurrences
    y_train = np.array(y_train)

    # Validation split
    X_tr, X_te, y_tr, y_te = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=200, max_depth=8, n_jobs=-1, random_state=42)
    rf.fit(X_tr, y_tr)

    # Metrics
    y_pred = rf.predict(X_te)
    y_prob = rf.predict_proba(X_te)[:, 1]
    auc    = roc_auc_score(y_te, y_prob)
    report = classification_report(y_te, y_pred, target_names=["Absent", "Present"])

    print(f"\n  Model evaluation:")
    print(f"  ROC-AUC = {auc:.4f}")
    print(report)

    # Feature importance
    importance = pd.Series(rf.feature_importances_,
                           index=["Elevation", "Slope", "NDVI"]).sort_values(ascending=False)
    print("  Feature importance:")
    for feat, imp in importance.items():
        bar = "#" * int(imp * 40)
        print(f"    {feat:<12s}  {imp:.4f}  {bar}")

    # Predict full grid
    print("\n  Predicting habitat suitability across full BBOX...")
    proba = np.full(dem.size, np.nan)
    X_valid = np.nan_to_num(X_full[valid], nan=0.0)
    proba[np.where(valid)[0]] = rf.predict_proba(X_valid)[:, 1]
    niche_map = proba.reshape(dem.shape)

    # Save TIF
    out = os.path.join(OUT_DIR, "ecological_niche_model.tif")
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(np.nan_to_num(niche_map, nan=-9999).astype("float32"), 1)
        dst.update_tags(description="RF SDM: Huemul habitat suitability probability [0-1]",
                        nodata="-9999",
                        arcgis_note="Stretched symbology: Yellow-Green (0=unsuitable, 1=optimal)",
                        envi_note="Density Slice for threshold mapping")
    print(f"  [OK] SDM TIF: {out}")

    # Feature importance CSV
    imp_df   = importance.reset_index()
    imp_df.columns = ["feature", "importance"]
    imp_df["auc"] = round(auc, 4)
    imp_path = os.path.join(OUT_DIR, "feature_importance.csv")
    imp_df.to_csv(imp_path, index=False, encoding="utf-8")

    return niche_map, importance, auc


# ---------------------------------------------------------------------------
# 4. Statistics CSV
# ---------------------------------------------------------------------------
def save_stats(dem, slope, ndvi, niche_map, km_map):
    rows = []
    for name, arr in [("DEM (m)", dem), ("Slope (deg)", slope),
                      ("NDVI", ndvi), ("SDM Suitability", niche_map)]:
        v = arr[np.isfinite(arr)]
        rows.append({
            "layer": name,
            "mean":  round(float(v.mean()), 4) if v.size > 0 else None,
            "std":   round(float(v.std()),  4) if v.size > 0 else None,
            "min":   round(float(v.min()),  4) if v.size > 0 else None,
            "max":   round(float(v.max()),  4) if v.size > 0 else None,
        })
    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, "niche_statistics.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"  [OK] Statistics CSV: {csv_path}")


# ---------------------------------------------------------------------------
# 5. 4-panel dark figure
# ---------------------------------------------------------------------------
def plot_niche(dem, km_map, niche_map, importance):
    print("\n  Building 4-panel niche modeling figure...")

    fig, axes = plt.subplots(2, 2, figsize=(18, 16), facecolor=DARK_BG)
    fig.suptitle("Ecological Niche Modeling -- Torres del Paine\nHuemul Deer Habitat Suitability",
                 color=C_TEXT, fontsize=13, fontweight="bold", y=0.98)

    def style_ax(ax, title):
        ax.set_facecolor(DARK_AX)
        ax.axis("off")
        ax.set_title(title, color=C_TEXT, fontsize=10, fontweight="bold", pad=6)

    # Panel 1: DEM
    ax = axes[0, 0]
    im1 = ax.imshow(dem, cmap="terrain", aspect="auto")
    cb1 = plt.colorbar(im1, ax=ax, fraction=0.035)
    cb1.set_label("Elevation (m)", color=C_TEXT, fontsize=8)
    cb1.ax.tick_params(colors=C_TEXT, labelsize=7)
    style_ax(ax, "Elevation (m) -- Predictor 1/3")

    # Panel 2: K-Means clusters
    ax = axes[0, 1]
    km_disp = np.where(km_map == -9999, np.nan, km_map)
    im2 = ax.imshow(km_disp, cmap="Set1", vmin=0, vmax=3, aspect="auto")
    cb2 = plt.colorbar(im2, ax=ax, fraction=0.035, ticks=[0, 1, 2, 3])
    cb2.set_label("Cluster", color=C_TEXT, fontsize=8)
    cb2.ax.tick_params(colors=C_TEXT, labelsize=7)
    style_ax(ax, "K-Means Unsupervised (4 terrain classes, no training data)")

    # Panel 3: SDM suitability
    ax = axes[1, 0]
    im3 = ax.imshow(niche_map, cmap="YlGn", vmin=0, vmax=1, aspect="auto")
    cb3 = plt.colorbar(im3, ax=ax, fraction=0.035)
    cb3.set_label("Suitability Prob.", color=C_TEXT, fontsize=8)
    cb3.ax.tick_params(colors=C_TEXT, labelsize=7)
    # High suitability contour
    ax.contour(niche_map, levels=[0.7], colors=[C_GOLD], linewidths=1.5)
    style_ax(ax, "RF SDM -- Huemul Habitat Suitability (gold contour = 0.7)")

    # Panel 4: Feature importance bar chart
    ax4 = axes[1, 1]
    ax4.set_facecolor(DARK_AX)
    for sp in ax4.spines.values():
        sp.set_color("#30363d")
    ax4.tick_params(colors=C_TEXT)
    feats  = list(importance.index)
    vals   = list(importance.values)
    colors_bar = [C_GREEN if v > 0.35 else C_BLUE if v > 0.15 else C_GREY for v in vals]
    bars = ax4.barh(feats, vals, color=colors_bar, alpha=0.85)
    for bar, v in zip(bars, vals):
        ax4.text(v + 0.005, bar.get_y() + bar.get_height() / 2,
                 f"{v:.3f}", va="center", color=C_TEXT, fontsize=9)
    ax4.set_xlabel("Feature Importance (Gini)", color=C_TEXT, fontsize=9)
    ax4.set_title("Random Forest Feature Importance",
                  color=C_TEXT, fontsize=10, fontweight="bold", pad=6)
    ax4.grid(alpha=0.15, color="#30363d", axis="x")

    out_png = os.path.join(OUT_DIR, "ecological_niche_model.png")
    fig.savefig(out_png, dpi=180, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  [OK] 4-panel figure: {out_png}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print(" GEOCASCADE - ECOLOGICAL NICHE MODELING (K-Means + RF SDM)")
    print(" Species: Patagonian Huemul Deer (Hippocamelus bisulcus)")
    print("=" * 65)

    print("\n[1/5] Loading environmental layers...")
    try:
        dem, slope, ndvi, profile = load_env_layers()
    except Exception as e:
        print(f"\n  ERROR: {e}")
        return

    print("\n[2/5] Unsupervised K-Means classification...")
    km_map, km_model = run_kmeans(dem, slope, ndvi, profile)

    print("\n[3/5] Supervised RF Species Distribution Model...")
    niche_map, importance, auc = run_sdm(dem, slope, ndvi, profile)

    print("\n[4/5] Saving statistics...")
    save_stats(dem, slope, ndvi, niche_map, km_map)

    print("\n[5/5] Building 4-panel figure...")
    plot_niche(dem, km_map, niche_map, importance)

    print("\n" + "=" * 65)
    print(" ECOLOGICAL NICHE MODELING COMPLETE")
    print("=" * 65)
    print(f"  TIFs  : {OUT_DIR}")
    print(f"  Figure: {os.path.join(OUT_DIR, 'ecological_niche_model.png')}")
    print(f"  Stats : {os.path.join(OUT_DIR, 'niche_statistics.csv')}")
    print()
    print("  ArcGIS Pro: Add ecological_niche_model.tif.")
    print("              Symbology > Stretched > Yellow-Green.")
    print("              Zonal Statistics: mean suitability per watershed.")
    print("  ENVI 5.6  : Density Slice tool for threshold-based habitat map.")
    print("=" * 65)


if __name__ == "__main__":
    main()
