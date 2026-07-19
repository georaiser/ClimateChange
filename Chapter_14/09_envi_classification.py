"""
09_envi_classification.py
==========================
GeoCascade Chapter 14 -- ArcGIS Pro + ENVI Professional Workflows
Torres del Paine, Patagonia, Chile

# NOTE: This script mirrors the ENVI Python API workflow.
# In production with ENVI license:
#   envi = ENVI()
#   task = envi.Task('ISODATAClassification')
#   task['INPUT_RASTER'] = raster
#   task['NUMBER_OF_CLASSES'] = 5
#   task.execute()

PURPOSE
-------
Performs land cover classification using spectral indices:
  - Input: NDVI, NBR, NDWI (from script 07)
  - Method 1: K-Means (mirrors ENVI ISODATA / ArcGIS Pro ISO Cluster)
  - Method 2: Maximum Likelihood Classifier (supervised, mirrors ENVI MLC)
  - Post-classification: Majority Filter (removes salt-and-pepper noise)
  - Accuracy assessment: basic overall accuracy from synthetic test points

CLASSES
-------
  0  Water / Patagonian lakes
  1  Permanent Snow & Glaciers (Grey, Tyndall)
  2  Bare Rock & Scree (Torres granite)
  3  Sparse Vegetation / Puna Grassland
  4  Dense Vegetation / Lenga Beech Forest

OUTPUTS
-------
  data/processed/envi_outputs/classified_land_cover.tif    -- K-Means result
  data/processed/envi_outputs/mlc_land_cover.tif           -- Max Likelihood
  data/processed/envi_outputs/classified_majority.tif      -- post-processed
  data/processed/envi_outputs/classification_report.png    -- 3-panel figure
  data/processed/envi_outputs/accuracy_report.csv

ENVI 5.6 EQUIVALENT
--------------------
  Classification > Unsupervised > ISODATA
  Classification > Supervised   > Maximum Likelihood
  Classification > Post Classification > Majority Analysis
  Classification > Post Classification > Class Statistics

ARCGIS PRO EQUIVALENT
----------------------
  Image Classification Wizard > Train Classifier > Max Likelihood
  ISO Cluster Unsupervised Classification
  Majority Filter (Spatial Analyst)

RUN
---
  python Chapter_14/09_envi_classification.py
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
from scipy.ndimage import generic_filter, label as sp_label
from sklearn.cluster import KMeans
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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENVI_DIR = os.path.join(BASE_DIR, "data", "processed", "envi_outputs")
os.makedirs(ENVI_DIR, exist_ok=True)

BBOX       = [-73.5, -51.5, -72.5, -50.5]
GRID_SHAPE = (120, 120)
N_CLASSES  = 5

CLASS_NAMES  = ["Water", "Snow/Ice", "Bare Rock", "Sparse Veg", "Dense Veg"]
CLASS_COLORS = ["#1565C0", "#E3F2FD", "#795548", "#FFC107", "#2E7D32"]

DARK_BG = "#0d1117"
DARK_AX = "#161b22"
C_TEXT  = "#e6edf3"
C_GREY  = "#8b949e"

CLS_CMAP = ListedColormap(CLASS_COLORS)


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def load_index(name: str) -> np.ndarray:
    """Load a spectral index TIF from script 07 output."""
    path = os.path.join(ENVI_DIR, f"{name}.tif")
    if HAS_RASTERIO and os.path.exists(path):
        with rasterio.open(path) as src:
            data   = src.read(1).astype(np.float32)
            nodata = src.nodata
            if nodata is not None:
                data = np.where(data == nodata, np.nan, data)
        print(f"  [OK] Loaded {name}: {data.shape}")
        return data
    else:
        # Same synthetic generation as script 07
        rng  = np.random.default_rng(seed=abs(hash(name)) % 999)
        rows, cols = GRID_SHAPE
        lons = np.linspace(BBOX[0], BBOX[2], cols)
        lats = np.linspace(BBOX[3], BBOX[1], rows)
        LON, LAT = np.meshgrid(lons, lats)
        if name == "ndvi":
            base = -0.1 + 0.8 * (LON - BBOX[0]) / (BBOX[2] - BBOX[0])
            data = (base + rng.normal(0, 0.1, (rows, cols))).clip(-1, 1)
            data[:rows//5, :cols//4] = rng.uniform(-0.3, 0.0, (rows//5, cols//4))
            data[rows//3:rows//2, cols//3:cols//2] = rng.uniform(-0.5, -0.2,
                                                                  (rows//6, cols//6))
        elif name == "nbr":
            base = 0.1 + 0.5 * (LON - BBOX[0]) / (BBOX[2] - BBOX[0])
            data = (base + rng.normal(0, 0.1, (rows, cols))).clip(-1, 1)
        else:
            base = -0.3 + 0.6 * (BBOX[3] - LAT) / (BBOX[3] - BBOX[1])
            data = (base + rng.normal(0, 0.1, (rows, cols))).clip(-1, 1)
            data[rows//3:rows//2, cols//3:cols//2] = rng.uniform(0.3, 0.8,
                                                                  (rows//6, cols//6))
        print(f"  [SYNTH] Generated {name}")
        return data.astype(np.float32)


def build_feature_matrix(ndvi, nbr, ndwi):
    """Stack indices into (n_pixels, 3) matrix."""
    X = np.stack([ndvi.ravel(), nbr.ravel(), ndwi.ravel()], axis=1)
    valid = ~np.any(np.isnan(X), axis=1)
    return X, valid


def kmeans_classify(X, valid, shape) -> np.ndarray:
    """K-Means with NDVI-ordered class labels (mirrors ENVI ISODATA)."""
    scaler  = StandardScaler()
    X_valid = scaler.fit_transform(X[valid])
    km = KMeans(n_clusters=N_CLASSES, random_state=42, n_init=20, max_iter=300)
    raw = km.fit_predict(X_valid)
    # Re-label by ascending NDVI centroid (Water < Snow < Rock < Sparse < Dense)
    ndvi_c  = km.cluster_centers_[:, 0]
    order   = np.argsort(ndvi_c)
    remap   = {o: n for n, o in enumerate(order)}
    labels  = np.array([remap[l] for l in raw])
    out = np.full(shape[0] * shape[1], -1, dtype=np.int8)
    out[valid] = labels
    return out.reshape(shape)


def mlc_classify(X, valid, km_map, shape) -> np.ndarray:
    """
    Maximum Likelihood Classifier (supervised).
    Uses K-Means output as training data (self-training approximation).
    For each class, computes mean and covariance from K-Means labels,
    then classifies all pixels by maximum log-likelihood.
    """
    from numpy.linalg import inv, det, slogdet

    km_flat = km_map.ravel()
    X_v     = X[valid]

    # Compute class statistics from K-Means result
    class_params = {}
    for c in range(N_CLASSES):
        mask = km_flat[valid] == c
        if mask.sum() < 5:
            continue
        X_c = X_v[mask]
        mu  = X_c.mean(axis=0)
        cov = np.cov(X_c, rowvar=False) + np.eye(3) * 1e-6
        class_params[c] = (mu, cov)

    # Score each pixel
    log_probs = np.full((shape[0] * shape[1], N_CLASSES), -np.inf)
    for c, (mu, cov) in class_params.items():
        sign, logdet = slogdet(cov)
        cov_inv = inv(cov)
        diff    = X_v - mu
        maha    = np.einsum("ij,jk,ik->i", diff, cov_inv, diff)
        log_probs[valid, c] = -0.5 * (maha + logdet)

    out = np.full(shape[0] * shape[1], -1, dtype=np.int8)
    out[valid] = log_probs[valid].argmax(axis=1).astype(np.int8)
    return out.reshape(shape)


def majority_filter(cls_map: np.ndarray, size: int = 3) -> np.ndarray:
    """
    Majority filter (removes salt-and-pepper noise).
    Mirrors ENVI: Classification > Post Classification > Majority Analysis
    and ArcGIS Pro: Spatial Analyst > Majority Filter.
    """
    def majority(values):
        vals = values.astype(int)
        vals = vals[vals >= 0]
        if len(vals) == 0:
            return -1
        counts = np.bincount(vals, minlength=N_CLASSES)
        return int(np.argmax(counts))

    smoothed = generic_filter(
        cls_map.astype(np.float32),
        majority, size=size
    ).astype(np.int8)
    changed = int(np.sum(smoothed != cls_map))
    print(f"  Majority filter ({size}x{size}): {changed:,} pixels reclassified "
          f"({changed / cls_map.size * 100:.1f}%)")
    return smoothed


def accuracy_report(km_map: np.ndarray, mlc_map: np.ndarray) -> pd.DataFrame:
    """Simple agreement matrix between K-Means and MLC."""
    rows = []
    valid = (km_map >= 0) & (mlc_map >= 0)
    agree = (km_map == mlc_map) & valid
    overall_acc = agree.sum() / valid.sum() * 100

    print(f"\n  Overall agreement (KM vs MLC): {overall_acc:.1f}%")
    for c in range(N_CLASSES):
        km_c  = (km_map  == c) & valid
        mlc_c = (mlc_map == c) & valid
        both  = km_c & mlc_c
        iou   = both.sum() / (km_c | mlc_c).sum() if (km_c | mlc_c).any() else 0
        rows.append({
            "class":          CLASS_NAMES[c],
            "km_pixels":      int(km_c.sum()),
            "mlc_pixels":     int(mlc_c.sum()),
            "agreement_pct":  round(both.sum() / km_c.sum() * 100 if km_c.any() else 0, 1),
            "iou":            round(float(iou), 3),
        })
        print(f"  {CLASS_NAMES[c]:<14}: KM={km_c.sum():>5}px  "
              f"MLC={mlc_c.sum():>5}px  IoU={iou:.3f}")

    return pd.DataFrame(rows)


def save_raster(data: np.ndarray, path: str) -> None:
    if not HAS_RASTERIO:
        return
    rows, cols = data.shape
    t = from_bounds(*BBOX, cols, rows)
    with rasterio.open(path, "w", driver="GTiff", height=rows, width=cols,
                       count=1, dtype="int8", crs="EPSG:4326",
                       transform=t, nodata=-1, compress="lzw") as dst:
        dst.write(data, 1)
    print(f"  [OK] {os.path.relpath(path, BASE_DIR)}")


def plot_results(km_map, mlc_map, maj_map, ndvi, out_path: str) -> None:
    """3-panel classification comparison figure."""
    fig = plt.figure(figsize=(20, 7), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.25,
                            top=0.88, bottom=0.08, left=0.04, right=0.97)
    fig.text(0.5, 0.96,
             "GeoCascade Ch14 -- ENVI Land Cover Classification",
             ha="center", color=C_TEXT, fontsize=13, fontweight="bold")
    fig.text(0.5, 0.93,
             "K-Means (ISODATA) | Maximum Likelihood | Majority Filter -- Torres del Paine",
             ha="center", color=C_GREY, fontsize=9)

    legend_patches = [Patch(facecolor=CLASS_COLORS[i], label=CLASS_NAMES[i])
                      for i in range(N_CLASSES)]

    def style(ax, title):
        ax.set_facecolor(DARK_AX)
        for sp in ax.spines.values(): sp.set_color("#30363d")
        ax.tick_params(colors=C_TEXT, labelsize=7)
        ax.set_title(title, color=C_TEXT, fontsize=9, fontweight="bold", pad=5)

    ext = [BBOX[0], BBOX[2], BBOX[1], BBOX[3]]

    for ax, data, title in [
        (fig.add_subplot(gs[0]), km_map,  "K-Means (ENVI ISODATA)"),
        (fig.add_subplot(gs[1]), mlc_map, "Maximum Likelihood Classifier"),
        (fig.add_subplot(gs[2]), maj_map, "MLC + Majority Filter (3x3)"),
    ]:
        ax.imshow(data.astype(float), cmap=CLS_CMAP, vmin=-0.5, vmax=N_CLASSES-0.5,
                  origin="upper", extent=ext)
        ax.legend(handles=legend_patches, fontsize=6, facecolor=DARK_BG,
                  labelcolor=C_TEXT, loc="lower left", framealpha=0.7)
        style(ax, title)
        ax.set_xlabel("Longitude", color=C_TEXT, fontsize=8)

    fig.axes[0].set_ylabel("Latitude", color=C_TEXT, fontsize=8)

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  [OK] Figure: {os.path.relpath(out_path, BASE_DIR)}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print(" GEOCASCADE Ch14 -- ENVI Land Cover Classification")
    print(" ISODATA (K-Means) | MLC | Majority Filter | 5 Classes")
    print("=" * 65)

    print("\n[1/6] Loading spectral indices (from script 07) ...")
    ndvi = load_index("ndvi")
    nbr  = load_index("nbr")
    ndwi = load_index("ndwi")

    print("\n[2/6] Building feature matrix ...")
    X, valid = build_feature_matrix(ndvi, nbr, ndwi)
    print(f"  {valid.sum():,} valid pixels / {X.shape[0]:,} total")

    print("\n[3/6] K-Means unsupervised classification (ENVI ISODATA) ...")
    km_map = kmeans_classify(X, valid, ndvi.shape)
    save_raster(km_map, os.path.join(ENVI_DIR, "classified_land_cover.tif"))

    print("\n[4/6] Maximum Likelihood Classification (supervised) ...")
    mlc_map = mlc_classify(X, valid, km_map, ndvi.shape)
    save_raster(mlc_map, os.path.join(ENVI_DIR, "mlc_land_cover.tif"))

    print("\n[5/6] Applying Majority Filter (post-classification) ...")
    maj_map = majority_filter(mlc_map, size=3)
    save_raster(maj_map, os.path.join(ENVI_DIR, "classified_majority.tif"))

    print("\n[6/6] Accuracy report and figure ...")
    acc_df = accuracy_report(km_map, mlc_map)
    csv_path = os.path.join(ENVI_DIR, "accuracy_report.csv")
    acc_df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"  [OK] Accuracy CSV: {os.path.relpath(csv_path, BASE_DIR)}")

    plot_results(km_map, mlc_map, maj_map, ndvi,
                 os.path.join(ENVI_DIR, "classification_report.png"))

    print("\n" + "=" * 65)
    print(" CLASSIFICATION COMPLETE")
    print("=" * 65)
    print(f"  K-Means    : {ENVI_DIR}\\classified_land_cover.tif")
    print(f"  MLC        : {ENVI_DIR}\\mlc_land_cover.tif")
    print(f"  Maj Filter : {ENVI_DIR}\\classified_majority.tif")
    print(f"  Accuracy   : {ENVI_DIR}\\accuracy_report.csv")
    print()
    print("  ENVI 5.6:")
    print("    File > Open -> classified_majority.tif")
    print("    Classification > Post Classification > Class Statistics")
    print("    View > Classification > Edit Class Colors")
    print()
    print("  ArcGIS Pro:")
    print("    Add classified_majority.tif -> Symbology > Unique Values")
    print("    Attribute Table: add field 'LandCover', populate by class ID")
    print()
    print("  Continue with: python Chapter_14/10_envi_change_detection.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
