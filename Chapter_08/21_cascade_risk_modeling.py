"""
Chapter 8: 21_cascade_risk_modeling.py

Academic Objective:
To apply Machine Learning (Random Forest) to our fused Multi-Sensor Data Cube. 
The algorithm will learn the multidimensional signature of the landscape (combining 
Optical, Radar, Elevation, and Thermal physics) to predict land cover and cascade vulnerabilities.
"""

import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from matplotlib.colors import ListedColormap

# ==========================================
# 1. Configuration
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IN_CUBE = os.path.join(BASE_DIR, "data", "processed", "cascade_master_stack.tif")
OUT_DIR = os.path.join(BASE_DIR, "data", "processed")

# ==========================================
# 2. Load Data Cube
# ==========================================
def load_data_cube():
    print(f"[INFO] Loading Multi-Sensor Data Cube: {os.path.basename(IN_CUBE)}")
    if not os.path.exists(IN_CUBE):
        raise FileNotFoundError("Master stack not found. Run 20_multisensor_data_fusion.py first.")
        
    with rasterio.open(IN_CUBE) as src:
        # Band 1: NIR, Band 2: SAR dB, Band 3: DEM, Band 4: LST
        img_stack = src.read()
        profile = src.profile
        
    # We must reshape the 3D stack (Bands, Height, Width) into a 2D matrix (Pixels, Features) for scikit-learn
    n_bands, height, width = img_stack.shape
    X_raw = img_stack.reshape(n_bands, -1).T  # Transpose to get Shape: (n_pixels, n_bands)
    
    # Remove NaN values (NoData boundaries or clouds)
    valid_mask = ~np.isnan(X_raw).any(axis=1)
    X_valid = X_raw[valid_mask]
    
    print(f"       [SUCCESS] Flattened array. Valid pixels to classify: {X_valid.shape[0]}")
    return X_valid, valid_mask, height, width, profile

# ==========================================
# 3. Generate Synthetic Training Data (Dynamic)
# ==========================================
def generate_training_data(X_valid):
    print("\n[INFO] Generating Synthetic Training Labels based on Geophysics...")
    
    NIR = X_valid[:, 0]
    SAR = X_valid[:, 1]
    DEM = X_valid[:, 2]
    LST = X_valid[:, 3]
    
    # Calculate percentiles dynamically to ensure we get samples regardless of region
    p10_nir, p90_nir = np.percentile(NIR[~np.isnan(NIR)], [10, 90])
    p10_sar, p90_sar = np.percentile(SAR[~np.isnan(SAR)], [10, 90])
    p10_dem, p90_dem = np.percentile(DEM[~np.isnan(DEM)], [10, 90])
    p10_lst, p90_lst = np.percentile(LST[~np.isnan(LST)], [10, 90])
    
    y_valid = np.zeros(X_valid.shape[0])
    
    # Class 1: Water (Very low radar, very low NIR)
    water_idx = (SAR <= p10_sar) & (NIR <= p10_nir)
    y_valid[water_idx] = 1
    
    # Class 2: Glaciers/Highlands (High elevation, very cold)
    glacier_idx = (DEM >= p90_dem) & (LST <= p10_lst)
    y_valid[glacier_idx] = 2
    
    # Class 3: Vegetation / Land (High NIR, warmer)
    veg_idx = (NIR >= p90_nir) & (LST >= p90_lst)
    y_valid[veg_idx] = 3
    
    # We only train the model on pixels we confidently labeled
    labeled_mask = y_valid > 0
    X_train = X_valid[labeled_mask]
    y_train = y_valid[labeled_mask]
    
    print(f"       Training Samples (Water):    {np.sum(y_train == 1)}")
    print(f"       Training Samples (Glacier):  {np.sum(y_train == 2)}")
    print(f"       Training Samples (Land):     {np.sum(y_train == 3)}")
    
    # Fallback if somehow still 0 (very rare with percentiles)
    if len(y_train) < 10:
        raise ValueError("Dynamic thresholds still yielded no training data. The data cube might be entirely NaN or completely homogeneous.")
        
    return X_train, y_train, y_valid

# ==========================================
# 4. Train Random Forest & Predict
# ==========================================
def run_machine_learning(X_train, y_train, X_valid, valid_mask, height, width, profile):
    print("\n[INFO] Training Random Forest Classifier (100 Trees)...")
    # n_jobs=-1 uses all CPU cores
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    print("[INFO] Predicting Classes for all pixels in the Data Cube...")
    y_pred = rf.predict(X_valid)
    
    # Feature Importance — a key ML teaching moment missing from original version
    print("\n[INFO] Random Forest Feature Importances:")
    feat_names = ['S2 NIR (Optical)', 'S1 SAR dB (Radar)', 'DEM Elevation (m)', 'MODIS LST (°C)']
    for name, imp in zip(feat_names, rf.feature_importances_):
        bar = '█' * int(imp * 50)
        print(f"       {name:25s}: {bar} {imp:.3f}")
    
    # Reconstruct the 2D image from the 1D predictions
    prediction_img = np.zeros((height, width), dtype=np.uint8)
    prediction_img.flat[valid_mask] = y_pred
    
    # 5. Export Geocoded Prediction
    profile.update(count=1, dtype=rasterio.uint8, nodata=0)
    out_tif = os.path.join(OUT_DIR, "cascade_ml_prediction.tif")
    with rasterio.open(out_tif, 'w', **profile) as dst:
        dst.write(prediction_img, 1)
    print(f"       [SUCCESS] Classified Geocoded TIFF saved to: {out_tif}")
    
    # Land cover pixel count summary
    print("\n[INFO] Land Cover Summary:")
    classes = {0: 'NoData', 1: 'Water', 2: 'Glacier/Ice', 3: 'Land/Vegetation'}
    for cls_id, cls_name in classes.items():
        count = np.sum(prediction_img == cls_id)
        pct   = 100.0 * count / prediction_img.size
        print(f"       {cls_name:22s}: {count:7d} px ({pct:.1f}%)")
    
    # 6. Plotting
    print("\n[INFO] Generating Visualization...")
    plt.figure(figsize=(10, 8))
    
    # Define colors: 0=Black(NoData), 1=Blue(Water), 2=Cyan(Glacier), 3=Green(Land)
    cmap = ListedColormap(['black', 'blue', 'cyan', 'forestgreen'])
    
    im = plt.imshow(prediction_img, cmap=cmap, vmin=0, vmax=3)
    cbar = plt.colorbar(im, ticks=[0, 1, 2, 3], shrink=0.7)
    cbar.ax.set_yticklabels(['NoData', 'Water', 'Glaciers/Ice', 'Land/Vegetation'])
    
    plt.title("Multi-Sensor Machine Learning (Random Forest)\nOptical + Radar + Thermal + Elevation", fontsize=14)
    plt.axis('off')
    
    plot_path = os.path.join(OUT_DIR, "cascade_ml_prediction.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"       [SUCCESS] Map saved to: {plot_path}")

def main():
    print("=======================================================")
    print(" GEOCASCADE - RANDOM FOREST CASCADE MODELING           ")
    print("=======================================================")
    
    try:
        X_valid, valid_mask, h, w, prof = load_data_cube()
        X_train, y_train, _ = generate_training_data(X_valid)
        run_machine_learning(X_train, y_train, X_valid, valid_mask, h, w, prof)
        print("\n[SUCCESS] Chapter 8 Pipeline Complete!")
    except Exception as e:
        print(f"\n[ERROR] {e}")

if __name__ == "__main__":
    main()
