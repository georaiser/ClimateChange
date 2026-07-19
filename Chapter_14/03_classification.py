"""
03_classification.py
=====================
GeoCascade Chapter 14 -- ArcGIS Pro + ENVI Professional Workflows
Torres del Paine, Patagonia, Chile

PURPOSE
-------
Performs unsupervised land cover classification on multi-band satellite data
using two methods:
  1. ISO Cluster / K-Means (mirrors ArcGIS Pro ISO Cluster Unsupervised Classification)
  2. Gaussian Mixture Model (mirrors ENVI ISODATA classification)

Input: Spectral index stack [NDVI, NBR, NDWI] from script 07 or synthetic if unavailable.
Output: Classified land cover map with 5 classes.

CLASSES (Torres del Paine study area)
--------------------------------------
  0  Water / Lakes (Lago Grey, Lago Nordenskjold)
  1  Permanent Snow / Glaciers (Grey Glacier, etc.)
  2  Bare Rock / Scree
  3  Sparse Vegetation / Grassland (Patagonian steppe)
  4  Dense Vegetation / Forest (Lenga beech)

OUTPUTS
-------
  data/processed/arcgis_outputs/classified_land_cover.tif
  data/processed/arcgis_outputs/classification_report.png  -- 3-panel figure
  data/processed/arcgis_outputs/class_statistics.csv

ARCGIS PRO EQUIVALENT
---------------------
  Image Classification Wizard -> Unsupervised Classification
  ISO Cluster Unsupervised Classification (Spatial Analyst)
  Maximum Likelihood Classification (supervised)

ENVI EQUIVALENT
---------------
  Classification -> Unsupervised -> ISODATA
  Classification -> Post Classification -> Majority/Minority Analysis

RUN
---
  python Chapter_14/03_classification.py
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
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

try:
    import rasterio
    from rasterio.transform import from_bounds
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT      = os.path.dirname(BASE_DIR)
PROC_DIR  = os.path.join(BASE_DIR, "data", "processed", "arcgis_outputs")
ENVI_DIR  = os.path.join(BASE_DIR, "data", "processed", "envi_outputs")

os.makedirs(PROC_DIR, exist_ok=True)

BBOX = [-73.5, -51.5, -72.5, -50.5]

# Spectral index rasters (from script 07)
NDVI_TIF = os.path.join(ENVI_DIR, "ndvi.tif")
NBR_TIF  = os.path.join(ENVI_DIR, "nbr.tif")
NDWI_TIF = os.path.join(ENVI_DIR, "ndwi.tif")

# Class definitions
N_CLASSES   = 5
CLASS_NAMES = ["Water", "Snow/Ice", "Bare Rock", "Sparse Veg", "Dense Veg"]
CLASS_COLORS = ["#1565C0", "#E3F2FD", "#795548", "#FFC107", "#2E7D32"]

# ---------------------------------------------------------------------------
# DARK STYLE
# ---------------------------------------------------------------------------
DARK_BG = "#0d1117"
DARK_AX = "#161b22"
C_TEXT  = "#e6edf3"
C_GREY  = "#8b949e"


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def load_index(path: str, label: str, shape=(80, 100)):
    """Load a spectral index GeoTIFF or return realistic synthetic data."""
    if HAS_RASTERIO and os.path.exists(path):
        with rasterio.open(path) as src:
            data = src.read(1).astype(np.float32)
            nodata = src.nodata
            if nodata is not None:
                data = np.where(data == nodata, np.nan, data)
            print(f"  [OK] Loaded {label}: shape={data.shape}")
            return data, src.transform
    else:
        print(f"  [SYNTHETIC] Generating synthetic {label}")
        rng = np.random.default_rng(seed=hash(label) % 1000)
        rows, cols = shape
        lon = np.linspace(BBOX[0], BBOX[2], cols)
        lat = np.linspace(BBOX[3], BBOX[1], rows)
        LON, LAT = np.meshgrid(lon, lat)

        if "ndvi" in label.lower():
            # High NDVI in southeast (forest), low in northwest (rock/glacier)
            base = -0.1 + 0.8 * ((LON - BBOX[0]) / (BBOX[2] - BBOX[0]))
            data = (base + rng.normal(0, 0.1, (rows, cols))).clip(-1, 1)
            # Add glacier patch (north-west corner) -> very low NDVI
            data[:rows//4, :cols//4] = rng.uniform(-0.3, 0.0, (rows//4, cols//4))
            # Add lake patch -> negative NDVI (compute real slice extents)
            r0, r1 = rows//3, rows//2
            c0, c1 = cols//3, cols//2
            data[r0:r1, c0:c1] = rng.uniform(-0.5, -0.2, (r1 - r0, c1 - c0))
        elif "nbr" in label.lower():
            base = 0.1 + 0.5 * ((LON - BBOX[0]) / (BBOX[2] - BBOX[0]))
            data = (base + rng.normal(0, 0.1, (rows, cols))).clip(-1, 1)
        else:  # NDWI
            base = -0.3 + 0.6 * ((BBOX[3] - LAT) / (BBOX[3] - BBOX[1]))
            data = (base + rng.normal(0, 0.1, (rows, cols))).clip(-1, 1)
            # Lake -> high NDWI
            data[rows//3:rows//2, cols//3:cols//2] = rng.uniform(0.3, 0.8,
                                                                  (rows//6, cols//6))

        t = from_bounds(*BBOX, cols, rows) if HAS_RASTERIO else None
        return data.astype(np.float32), t


def build_feature_stack(ndvi, nbr, ndwi):
    """Stack indices into (pixels, features) array for clustering."""
    rows, cols = ndvi.shape
    X = np.stack([ndvi.ravel(), nbr.ravel(), ndwi.ravel()], axis=1)
    # Remove NaN rows
    valid_mask = ~np.any(np.isnan(X), axis=1)
    return X, valid_mask, rows, cols


def kmeans_classify(X: np.ndarray, valid_mask: np.ndarray,
                    rows: int, cols: int) -> tuple:
    """K-Means clustering (mirrors ArcGIS Pro ISO Cluster)."""
    scaler = StandardScaler()
    X_valid = scaler.fit_transform(X[valid_mask])

    print(f"  Running K-Means (k={N_CLASSES}, n_init=20) ...")
    km = KMeans(n_clusters=N_CLASSES, random_state=42, n_init=20, max_iter=300)
    raw_labels = km.fit_predict(X_valid)

    # Re-order by centroid NDVI ascending (Water < Snow < Rock < Sparse < Dense)
    ndvi_centroids = km.cluster_centers_[:, 0]
    order = np.argsort(ndvi_centroids)
    remap = {old: new for new, old in enumerate(order)}
    labels_reordered = np.array([remap[l] for l in raw_labels])

    # Embed back into full grid
    classified = np.full(rows * cols, -1, dtype=np.int8)
    classified[valid_mask] = labels_reordered
    return classified.reshape(rows, cols), km


def gmm_classify(X: np.ndarray, valid_mask: np.ndarray,
                 rows: int, cols: int) -> np.ndarray:
    """Gaussian Mixture Model (mirrors ENVI ISODATA)."""
    scaler = StandardScaler()
    X_valid = scaler.fit_transform(X[valid_mask])

    print(f"  Running GMM (k={N_CLASSES}, covariance_type=full) ...")
    gmm = GaussianMixture(n_components=N_CLASSES, random_state=42,
                          covariance_type="full", max_iter=200)
    raw_labels = gmm.fit_predict(X_valid)

    ndvi_means = np.array([X_valid[raw_labels == i, 0].mean()
                           for i in range(N_CLASSES)])
    order = np.argsort(ndvi_means)
    remap = {old: new for new, old in enumerate(order)}
    labels_reordered = np.array([remap[l] for l in raw_labels])

    classified = np.full(rows * cols, -1, dtype=np.int8)
    classified[valid_mask] = labels_reordered
    return classified.reshape(rows, cols)


def compute_class_stats(km_map: np.ndarray, gmm_map: np.ndarray) -> pd.DataFrame:
    """Per-class area statistics."""
    rows_list = []
    total_px = np.sum(km_map >= 0)
    pixel_area_km2 = (1.0 / 100) ** 2 * 111.32 ** 2   # ~0.01deg pixel at 45S

    for i, name in enumerate(CLASS_NAMES):
        km_n  = int(np.sum(km_map == i))
        gmm_n = int(np.sum(gmm_map == i))
        rows_list.append({
            "class_id":   i,
            "class_name": name,
            "color":      CLASS_COLORS[i],
            "km_pixels":  km_n,
            "km_pct":     km_n / total_px * 100 if total_px > 0 else 0,
            "km_area_km2": km_n * pixel_area_km2,
            "gmm_pixels": gmm_n,
            "gmm_pct":    gmm_n / total_px * 100 if total_px > 0 else 0,
        })
    df = pd.DataFrame(rows_list)
    return df


def save_raster(data: np.ndarray, transform, path: str) -> None:
    if not HAS_RASTERIO or transform is None:
        return
    with rasterio.open(path, "w", driver="GTiff",
                       height=data.shape[0], width=data.shape[1],
                       count=1, dtype="int8", crs="EPSG:4326",
                       transform=transform, nodata=-1, compress="lzw") as dst:
        dst.write(data, 1)
    print(f"  [OK] Raster: {os.path.relpath(path, BASE_DIR)}")


def plot_results(ndvi, km_map, gmm_map, stats_df, out_path: str) -> None:
    """3-panel classification figure."""
    cmap  = ListedColormap(CLASS_COLORS)
    vmin, vmax = -0.5, N_CLASSES - 0.5

    fig = plt.figure(figsize=(20, 7), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(1, 3, figure=fig, hspace=0.15, wspace=0.28,
                            top=0.88, bottom=0.1, left=0.05, right=0.97)
    fig.text(0.5, 0.96, "GeoCascade Ch14 -- Land Cover Classification",
             ha="center", color=C_TEXT, fontsize=13, fontweight="bold")
    fig.text(0.5, 0.935,
             "K-Means (ArcGIS Pro ISO Cluster)  |  GMM (ENVI ISODATA)  |  Torres del Paine",
             ha="center", color=C_GREY, fontsize=9)

    def style(ax, title):
        ax.set_facecolor(DARK_AX)
        for sp in ax.spines.values(): sp.set_color("#30363d")
        ax.tick_params(colors=C_TEXT, labelsize=8)
        ax.set_title(title, color=C_TEXT, fontsize=10, fontweight="bold", pad=6)

    legend_patches = [Patch(facecolor=CLASS_COLORS[i], label=CLASS_NAMES[i])
                      for i in range(N_CLASSES)]

    # Panel 1: K-Means classification
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(km_map, cmap=cmap, vmin=vmin, vmax=vmax,
               origin="upper", extent=[BBOX[0], BBOX[2], BBOX[1], BBOX[3]])
    ax1.legend(handles=legend_patches, loc="lower left", fontsize=7,
               facecolor=DARK_BG, labelcolor=C_TEXT, framealpha=0.7)
    style(ax1, "K-Means Classification (ArcGIS Pro ISO Cluster)")
    ax1.set_xlabel("Longitude", color=C_TEXT, fontsize=8)
    ax1.set_ylabel("Latitude", color=C_TEXT, fontsize=8)

    # Panel 2: GMM classification
    ax2 = fig.add_subplot(gs[1])
    ax2.imshow(gmm_map, cmap=cmap, vmin=vmin, vmax=vmax,
               origin="upper", extent=[BBOX[0], BBOX[2], BBOX[1], BBOX[3]])
    ax2.legend(handles=legend_patches, loc="lower left", fontsize=7,
               facecolor=DARK_BG, labelcolor=C_TEXT, framealpha=0.7)
    style(ax2, "GMM Classification (ENVI ISODATA)")
    ax2.set_xlabel("Longitude", color=C_TEXT, fontsize=8)

    # Panel 3: Area comparison bar chart
    ax3 = fig.add_subplot(gs[2])
    x    = np.arange(N_CLASSES)
    w    = 0.35
    ax3.bar(x - w/2, stats_df["km_pct"],  w, color=CLASS_COLORS, alpha=0.85,
            label="K-Means", edgecolor="#30363d")
    ax3.bar(x + w/2, stats_df["gmm_pct"], w, color=CLASS_COLORS, alpha=0.5,
            label="GMM", edgecolor="#30363d", hatch="//")
    ax3.set_xticks(x)
    ax3.set_xticklabels([n[:8] for n in CLASS_NAMES], rotation=30, ha="right",
                        color=C_TEXT, fontsize=8)
    ax3.set_ylabel("Class Area (%)", color=C_TEXT, fontsize=9)
    ax3.legend(fontsize=8, facecolor=DARK_BG, labelcolor=C_TEXT)
    style(ax3, "Class Area Comparison")

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  [OK] Classification figure: {os.path.relpath(out_path, BASE_DIR)}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print(" GEOCASCADE Ch14 -- Land Cover Classification")
    print(" K-Means (ArcGIS Pro) | GMM (ENVI ISODATA) | 5 Classes")
    print("=" * 65)

    print("\n[1/5] Loading spectral indices ...")
    ndvi, t_ref = load_index(NDVI_TIF, "NDVI")
    nbr,  _     = load_index(NBR_TIF,  "NBR",  shape=ndvi.shape)
    ndwi, _     = load_index(NDWI_TIF, "NDWI", shape=ndvi.shape)

    print("\n[2/5] Building feature stack ...")
    X, valid_mask, rows, cols = build_feature_stack(ndvi, nbr, ndwi)
    print(f"  Feature matrix: {X[valid_mask].shape[0]:,} valid pixels x 3 features")

    print("\n[3/5] K-Means classification (ArcGIS Pro ISO Cluster equivalent) ...")
    km_map, km_model = kmeans_classify(X, valid_mask, rows, cols)
    save_raster(km_map, t_ref, os.path.join(PROC_DIR, "classified_land_cover.tif"))

    print("\n[4/5] GMM classification (ENVI ISODATA equivalent) ...")
    gmm_map = gmm_classify(X, valid_mask, rows, cols)
    save_raster(gmm_map, t_ref, os.path.join(PROC_DIR, "gmm_land_cover.tif"))

    print("\n[5/5] Computing class statistics and generating figure ...")
    stats_df = compute_class_stats(km_map, gmm_map)
    print("\n  Class Area Summary (K-Means):")
    print("  " + "-" * 52)
    for _, row in stats_df.iterrows():
        bar = "#" * int(row["km_pct"] / 2)
        print(f"  {row['class_name']:<12}: {row['km_pct']:5.1f}%  "
              f"{row['km_area_km2']:6.1f} km2  {bar}")

    csv_path = os.path.join(PROC_DIR, "class_statistics.csv")
    stats_df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\n  [OK] Statistics CSV: {os.path.relpath(csv_path, BASE_DIR)}")

    plot_results(ndvi, km_map, gmm_map, stats_df,
                 os.path.join(PROC_DIR, "classification_report.png"))

    print("\n" + "=" * 65)
    print(" CLASSIFICATION COMPLETE")
    print("=" * 65)
    print(f"  K-Means raster  : {PROC_DIR}\\classified_land_cover.tif")
    print(f"  GMM raster      : {PROC_DIR}\\gmm_land_cover.tif")
    print(f"  Statistics CSV  : {PROC_DIR}\\class_statistics.csv")
    print(f"  Report figure   : {PROC_DIR}\\classification_report.png")
    print()
    print("  ArcGIS Pro:")
    print("    classified_land_cover.tif -> Symbology > Unique Values")
    print("    Raster Functions > Reclassify to merge classes")
    print("    Accuracy Assessment -> Create Accuracy Assessment Points")
    print()
    print("  ENVI 5.6:")
    print("    Post-Classification: Clump/Sieve -> minimum class size = 5 pixels")
    print("    Export Classification -> Symbology -> GeoTIFF")
    print()
    print("  Continue with: python Chapter_14/04_spatial_analysis.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
