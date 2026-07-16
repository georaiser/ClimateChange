"""
Chapter 8: 21_cascade_risk_modeling.py
==========================================
Random Forest Multi-Sensor Classification & Cascade Risk Mapping

Academic Objective:
  This script applies Machine Learning (Random Forest Classifier) to the
  4-band multi-sensor data cube produced by Script 20. The model learns
  the COMBINED spectral-structural-thermal-topographic signature of four
  land cover classes simultaneously, which no single sensor alone can
  distinguish reliably:

  Class 0: NoData / unclassified
  Class 1: Water / lake surfaces (low VV SAR, low NIR, low elevation)
  Class 2: Glacier / ice / snow (high elevation, cold LST, moderate SAR)
  Class 3: Vegetation / land (high NIR, moderate LST, low-moderate SAR)

  Why Random Forest for SAR+Optical fusion?
  - RF handles heterogeneous feature spaces (reflectance + dB + metres + celsius)
    without requiring normalization of individual bands.
  - Out-of-Bag (OOB) score provides a free cross-validation estimate —
    no train/test split needed for initial assessment.
  - Feature importances reveal which sensor contributes most to each class.
  - Probability maps (predict_proba) give confidence per pixel, enabling
    a "glacier vulnerability" continuous score rather than hard classes.

  Training data generation strategy:
  Labels are generated DYNAMICALLY using geophysical percentile thresholds,
  not hardcoded DN values. This makes the script robust across different dates
  and scenes where absolute values change significantly.

  Cascade Risk interpretation:
  - High glacier probability (>0.7) + high elevation → cryosphere risk
  - Water pixels adjacent to glacier → glacier lake outburst risk (GLOF)
  - Vegetation pixels with declining NIR + high LST → ecosystem stress

Outputs:
  data/processed/ml/cascade_ml_prediction.tif       (uint8, 4 classes)
  data/processed/ml/glacier_probability_map.tif     (float32, [0-1])
  data/processed/ml/feature_importance.csv
  data/processed/ml/classification_report.csv
  data/processed/ml/cascade_ml_prediction.png       (4-panel dark)

ArcGIS Pro: Add cascade_ml_prediction.tif.
            Symbology > Unique Values. Map 0=NoData, 1=Blue, 2=Cyan, 3=Green.
            Add glacier_probability_map.tif with Yellow-Red gradient.
ENVI 5.6:   Classification > Supervised > Read cascade_ml_prediction.tif.
            Display > Classification Image.

Run:
  conda activate geocascade_env
  python Chapter_08/21_cascade_risk_modeling.py

Dependencies: rasterio, numpy, matplotlib, pandas, scikit-learn
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
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import rasterio

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Try fusion/ subdirectory first, fall back to old processed/ path
STACK_CANDIDATES = [
    os.path.join(BASE_DIR, "data", "processed", "fusion", "cascade_master_stack.tif"),
    os.path.join(BASE_DIR, "data", "processed", "cascade_master_stack.tif"),
]
OUT_DIR  = os.path.join(BASE_DIR, "data", "processed", "ml")
os.makedirs(OUT_DIR, exist_ok=True)

DARK_BG  = "#0d1117"
DARK_AX  = "#161b22"
C_TEXT   = "#e6edf3"
C_GREY   = "#8b949e"

CLASS_COLORS = ["#222222", "#2980b9", "#74b9ff", "#27ae60"]
CLASS_NAMES  = ["NoData", "Water / Lake", "Glacier / Ice", "Vegetation / Land"]


# ---------------------------------------------------------------------------
# 1. Load data cube
# ---------------------------------------------------------------------------
def load_data_cube():
    in_cube = None
    for path in STACK_CANDIDATES:
        if os.path.exists(path):
            in_cube = path
            break

    if in_cube is None:
        raise FileNotFoundError(
            "cascade_master_stack.tif not found.\n"
            "Run: python Chapter_08/20_multisensor_data_fusion.py first."
        )

    print(f"  [OK] Loading cube: {os.path.relpath(in_cube, BASE_DIR)}")
    with rasterio.open(in_cube) as src:
        stack   = src.read().astype("float32")   # (4, H, W)
        profile = src.profile.copy()
        nodata  = src.nodata if src.nodata is not None else -9999.0

    n_bands, height, width = stack.shape
    X_raw = stack.reshape(n_bands, -1).T   # (H*W, 4)

    # Valid pixels: no nodata, no nan, no inf
    valid_mask = np.ones(X_raw.shape[0], dtype=bool)
    for b in range(n_bands):
        valid_mask &= np.isfinite(X_raw[:, b])
        valid_mask &= (X_raw[:, b] != nodata)

    X_valid = X_raw[valid_mask]
    print(f"  Grid: {width}x{height}  |  Valid pixels: {X_valid.shape[0]:,}  "
          f"({100.0 * X_valid.shape[0] / X_raw.shape[0]:.1f}%)")
    return X_valid, valid_mask, height, width, profile


# ---------------------------------------------------------------------------
# 2. Generate training labels from geophysical percentiles
# ---------------------------------------------------------------------------
def generate_labels(X_valid):
    """
    Dynamic percentile-based labeling:
      Water:   Low SAR (p10) AND Low NIR (p10)   → specular + little vegetation
      Glacier: High elevation (p80) AND cold LST (p20) → high + cold
      Veg:     High NIR (p80) AND warm LST (p70)  → green + warm

    Using percentiles (not hardcoded thresholds) ensures labels exist
    even when scene-wide absolute values shift between dates.
    """
    NIR = X_valid[:, 0]
    SAR = X_valid[:, 1]
    DEM = X_valid[:, 2]
    LST = X_valid[:, 3]

    # Compute per-band percentiles on finite values
    def pct(arr, p):
        v = arr[np.isfinite(arr)]
        return np.percentile(v, p) if v.size > 0 else 0.0

    p10_nir, p80_nir = pct(NIR, 10), pct(NIR, 80)
    p10_sar          = pct(SAR, 10)
    p20_lst, p70_lst = pct(LST, 20), pct(LST, 70)
    p80_dem          = pct(DEM, 80)

    y = np.zeros(X_valid.shape[0], dtype=np.uint8)
    y[np.where((SAR <= p10_sar) & (NIR <= p10_nir))]              = 1  # Water
    y[np.where((DEM >= p80_dem) & (LST <= p20_lst))]              = 2  # Glacier
    y[np.where((NIR >= p80_nir) & (LST >= p70_lst) & (y == 0))]  = 3  # Vegetation

    labeled = y > 0
    X_train = X_valid[labeled]
    y_train = y[labeled]

    for cls_id, cls_name in enumerate(CLASS_NAMES[1:], 1):
        n = int(np.sum(y_train == cls_id))
        print(f"  Class {cls_id} ({cls_name:<20s}): {n:>6,} training pixels")

    if len(y_train) < 30:
        raise ValueError(
            "Fewer than 30 training pixels generated. "
            "The data cube may be mostly NoData. Check fusion output."
        )
    return X_train, y_train


# ---------------------------------------------------------------------------
# 3. Train Random Forest and predict
# ---------------------------------------------------------------------------
def run_rf(X_train, y_train, X_valid, valid_mask, height, width, profile):
    from sklearn.ensemble import RandomForestClassifier

    print(f"\n  Training Random Forest ({len(X_train):,} samples, 150 trees)...")
    rf = RandomForestClassifier(
        n_estimators=150, max_depth=12, random_state=42,
        n_jobs=-1, oob_score=True, class_weight="balanced"
    )
    rf.fit(X_train, y_train)
    print(f"  [OK] OOB accuracy (free cross-validation): {rf.oob_score_:.3f}")

    print("  Predicting all valid pixels...")
    y_pred  = rf.predict(X_valid)
    y_proba = rf.predict_proba(X_valid)  # (n_valid, n_classes)

    # --- Feature importances ---
    feat_names = ["S2 NIR (Optical)", "S1 SAR VV dB", "DEM Elevation (m)",
                  "MODIS LST (deg C)"]
    imp_df = pd.DataFrame({
        "feature":    feat_names,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)

    print("\n  Feature Importances:")
    for _, r in imp_df.iterrows():
        bar = "|" * int(r["importance"] * 50)
        print(f"    {r['feature']:<25s} {bar} {r['importance']:.3f}")

    csv_imp = os.path.join(OUT_DIR, "feature_importance.csv")
    imp_df.to_csv(csv_imp, index=False, encoding="utf-8")
    print(f"  [OK] feature_importance.csv")

    # --- Reconstruct 2D arrays ---
    pred_img = np.zeros(height * width, dtype=np.uint8)
    flat_idx = np.where(valid_mask)[0]
    pred_img[flat_idx] = y_pred.astype(np.uint8)
    pred_img = pred_img.reshape(height, width)

    # Glacier probability map (class 2)
    glacier_cls_idx = list(rf.classes_).index(2) if 2 in rf.classes_ else None
    prob_img = np.full(height * width, -9999.0, dtype=np.float32)
    if glacier_cls_idx is not None:
        prob_img[flat_idx] = y_proba[:, glacier_cls_idx]
    prob_img = prob_img.reshape(height, width)

    # --- Class summary ---
    print("\n  Land Cover Classification Summary:")
    rows = []
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        count = int(np.sum(pred_img == cls_id))
        pct   = 100.0 * count / pred_img.size
        print(f"    {cls_name:<25s}: {count:>8,} px  ({pct:5.1f}%)")
        rows.append({"class_id": cls_id, "class_name": cls_name,
                     "pixel_count": count, "pct": round(pct, 2)})
    rows.append({"class_id": -1, "class_name": "OOB_accuracy",
                 "pixel_count": -1, "pct": round(rf.oob_score_, 4)})
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "classification_report.csv"),
                              index=False, encoding="utf-8")
    print(f"  [OK] classification_report.csv")

    return pred_img, prob_img, imp_df, rf.oob_score_


# ---------------------------------------------------------------------------
# 4. Save GeoTIFFs
# ---------------------------------------------------------------------------
def save_tifs(pred_img, prob_img, profile):
    # Classification TIF (uint8)
    cls_prof = profile.copy()
    cls_prof.update(count=1, dtype=rasterio.uint8, nodata=255, compress="lzw")
    out_cls = os.path.join(OUT_DIR, "cascade_ml_prediction.tif")
    with rasterio.open(out_cls, "w", **cls_prof) as dst:
        dst.write(pred_img.astype(np.uint8), 1)
        dst.update_tags(classes="0=NoData,1=Water,2=Glacier,3=Vegetation",
                        method="RandomForest_150trees")
    print(f"  [OK] cascade_ml_prediction.tif")

    # Glacier probability TIF (float32)
    prob_prof = profile.copy()
    prob_prof.update(count=1, dtype=rasterio.float32, nodata=-9999, compress="lzw")
    out_prob = os.path.join(OUT_DIR, "glacier_probability_map.tif")
    with rasterio.open(out_prob, "w", **prob_prof) as dst:
        dst.write(prob_img.astype(np.float32), 1)
        dst.update_tags(description="RF probability of Glacier/Ice class (0-1)")
    print(f"  [OK] glacier_probability_map.tif")


# ---------------------------------------------------------------------------
# 5. 4-panel dark figure
# ---------------------------------------------------------------------------
def plot_results(pred_img, prob_img, imp_df, oob_score):
    print("\n  Building 4-panel classification figure...")

    cmap = ListedColormap(CLASS_COLORS)
    norm = BoundaryNorm([0, 1, 2, 3, 4], cmap.N)

    fig = plt.figure(figsize=(22, 8), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.22,
                            top=0.88, bottom=0.05, left=0.04, right=0.97)
    fig.text(0.5, 0.95,
             f"Random Forest Cascade Risk Classification  |  OOB accuracy: {oob_score:.3f}",
             ha="center", color=C_TEXT, fontsize=13, fontweight="bold")

    def style_ax(ax, title):
        ax.set_facecolor(DARK_AX)
        ax.axis("off")
        ax.set_title(title, color=C_TEXT, fontsize=9, fontweight="bold", pad=5)

    # Panel 1: Classification map
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(pred_img, cmap=cmap, norm=norm, aspect="auto")
    legend_patches = [Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i])
                      for i in range(len(CLASS_NAMES))]
    ax1.legend(handles=legend_patches, loc="lower right",
               fontsize=6.5, facecolor=DARK_BG, labelcolor=C_TEXT,
               framealpha=0.9)
    style_ax(ax1, "RF Land Cover Classification\n(4-sensor fusion)")

    # Panel 2: Glacier probability heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    prob_display = np.where(prob_img == -9999, np.nan, prob_img)
    im2 = ax2.imshow(prob_display, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    cb2 = plt.colorbar(im2, ax=ax2, fraction=0.04, pad=0.02)
    cb2.set_label("Probability", color=C_TEXT, fontsize=7)
    cb2.ax.tick_params(colors=C_TEXT, labelsize=6)
    style_ax(ax2, "Glacier / Ice Probability\n(Higher = more cryosphere risk)")

    # Panel 3: Feature importance bar chart
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor(DARK_AX)
    for sp in ax3.spines.values():
        sp.set_color("#30363d")
    ax3.tick_params(colors=C_TEXT, labelsize=8)
    colors_bar = ["#3498db", "#e74c3c", "#f39c12", "#9b59b6"]
    bars = ax3.barh(imp_df["feature"], imp_df["importance"],
                    color=colors_bar[:len(imp_df)], height=0.6)
    ax3.set_xlabel("Importance", color=C_TEXT, fontsize=8)
    ax3.set_title("Feature Importances\n(which sensor drives classification?)",
                  color=C_TEXT, fontsize=9, fontweight="bold", pad=5)
    ax3.set_xlim(0, imp_df["importance"].max() * 1.25)
    for bar, val in zip(bars, imp_df["importance"]):
        ax3.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", va="center", color=C_TEXT, fontsize=8)
    ax3.tick_params(axis="y", colors=C_TEXT, labelsize=7)
    ax3.tick_params(axis="x", colors=C_TEXT, labelsize=6)
    ax3.grid(axis="x", alpha=0.15, color="#30363d")

    # Panel 4: Class area donut
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.set_facecolor(DARK_AX)
    ax4.set_title("Land Cover Area Distribution",
                  color=C_TEXT, fontsize=9, fontweight="bold", pad=5)
    counts = [int(np.sum(pred_img == i)) for i in range(1, 4)]   # skip NoData
    labels = CLASS_NAMES[1:]
    colors_pie = CLASS_COLORS[1:]
    total = sum(counts) or 1
    wedge_props = {"width": 0.5, "edgecolor": DARK_BG, "linewidth": 1.5}
    patches, texts, autotexts = ax4.pie(
        counts, labels=labels, colors=colors_pie,
        autopct=lambda p: f"{p:.1f}%", startangle=90,
        textprops={"color": C_TEXT, "fontsize": 7},
        wedgeprops=wedge_props
    )
    for at in autotexts:
        at.set_color(C_TEXT)
        at.set_fontsize(7)
    ax4.axis("equal")

    out_png = os.path.join(OUT_DIR, "cascade_ml_prediction.png")
    fig.savefig(out_png, dpi=180, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  [OK] 4-panel figure: {out_png}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print(" GEOCASCADE - RANDOM FOREST CASCADE RISK MODELING")
    print(" Input: 4-band data cube  |  Output: classification + risk map")
    print("=" * 65)

    print("\n[1/4] Loading multi-sensor data cube...")
    try:
        X_valid, valid_mask, h, w, profile = load_data_cube()
    except (FileNotFoundError, Exception) as e:
        print(f"\n  ERROR: {e}")
        return

    print("\n[2/4] Generating training labels (percentile-based)...")
    try:
        X_train, y_train = generate_labels(X_valid)
    except ValueError as e:
        print(f"\n  ERROR: {e}")
        return

    print("\n[3/4] Training Random Forest and predicting...")
    pred_img, prob_img, imp_df, oob = run_rf(X_train, y_train, X_valid,
                                              valid_mask, h, w, profile)

    print("\n[4/4] Saving outputs...")
    save_tifs(pred_img, prob_img, profile)
    plot_results(pred_img, prob_img, imp_df, oob)

    print("\n" + "=" * 65)
    print(" CASCADE RISK MODELING COMPLETE")
    print("=" * 65)
    print(f"  Classification : {os.path.join(OUT_DIR, 'cascade_ml_prediction.tif')}")
    print(f"  Glacier risk   : {os.path.join(OUT_DIR, 'glacier_probability_map.tif')}")
    print(f"  Figure         : {os.path.join(OUT_DIR, 'cascade_ml_prediction.png')}")
    print()
    print("  ArcGIS Pro: Add cascade_ml_prediction.tif.")
    print("              Symbology > Unique Values: 1=Blue 2=Cyan 3=Green.")
    print("              Add glacier_probability_map.tif with Yellow-Red gradient.")
    print("  ENVI 5.6  : Classification > Supervised > Load prediction TIF.")
    print("=" * 65)


if __name__ == "__main__":
    main()
