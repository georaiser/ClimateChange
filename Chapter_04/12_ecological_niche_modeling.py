"""
Chapter 4: 12_ecological_niche_modeling.py

Academic Objective:
Species Distribution Modeling (SDM) / Ecological Niche Modeling predicts the
geographic distribution of a species based on environmental variables.

This script demonstrates TWO complementary machine learning approaches:

1. UNSUPERVISED CLASSIFICATION (K-Means) — per ClimateChange.txt Módulo 3.1
   No training data needed. The algorithm discovers natural clusters in the data.
   K-Means groups every pixel into N classes based purely on similarity of
   Elevation + Slope + NDVI features.

2. SUPERVISED CLASSIFICATION (Random Forest SDM) — per brainstorm MaxEnt workflow
   We simulate presence/absence training data for the Patagonian Huemul Deer
   and train a classifier to predict habitat suitability probability.

Dependencies:
mamba install -n geocascade_env -c conda-forge scikit-learn rasterio numpy matplotlib pystac-client planetary-computer pyproj -y
"""

import os
import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from pystac_client import Client
import planetary_computer as pc
from pyproj import Transformer

# ==========================================
# 1. Configuration
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

BBOX = [-73.30, -51.10, -72.90, -50.80]

# ==========================================
# 2. Environmental Variables (Predictors)
# ==========================================
def fetch_environmental_layers():
    print("\n[INFO] Connecting to Planetary Computer STAC API...")
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
    
    # 1. Fetch Elevation (DEM)
    print("       Fetching 3D Terrain Data (Copernicus DEM)...")
    search_dem = catalog.search(collections=["cop-dem-glo-30"], bbox=BBOX)
    item_dem = list(search_dem.items())[0]
    
    with rasterio.open(item_dem.assets["data"].href) as src:
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        minx, miny = transformer.transform(BBOX[0], BBOX[1])
        maxx, maxy = transformer.transform(BBOX[2], BBOX[3])
        window_dem = from_bounds(minx, miny, maxx, maxy, src.transform)
        dem = src.read(1, window=window_dem).astype('float32')
        dem = np.where(dem < 0, 0, dem) # Floor sea level
        
        # Calculate Slope
        dy, dx = np.gradient(dem, 30.0, 30.0)
        slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
        
        target_shape = dem.shape
        target_transform = rasterio.windows.transform(window_dem, src.transform)
        target_crs = src.crs
        
        # Save profile for TIFF export
        profile = src.profile
        profile.update(
            dtype=rasterio.float32, count=1, nodata=np.nan,
            height=target_shape[0], width=target_shape[1],
            transform=target_transform
        )

    # 2. Fetch Vegetation (Sentinel-2 NDVI)
    print("       Fetching Vegetation Data (Sentinel-2)...")
    search_s2 = catalog.search(
        collections=["sentinel-2-l2a"], bbox=BBOX, 
        datetime="2023-01-01/2023-02-28", query={"eo:cloud_cover": {"lt": 5}}
    )
    items_s2 = list(search_s2.items())
    if not items_s2:
        raise ValueError("No Sentinel-2 found for the given date and BBOX. Widen cloud cover or date range.")
    item_s2 = sorted(items_s2, key=lambda x: x.properties["eo:cloud_cover"])[0]
    
    with rasterio.open(item_s2.assets["B04"].href) as src_red:
        transformer2 = Transformer.from_crs("EPSG:4326", src_red.crs, always_xy=True)
        minx, miny = transformer2.transform(BBOX[0], BBOX[1])
        maxx, maxy = transformer2.transform(BBOX[2], BBOX[3])
        win_s2 = from_bounds(minx, miny, maxx, maxy, src_red.transform)
        
        # Read and resample to match DEM exactly
        red = src_red.read(1, window=win_s2, out_shape=target_shape, resampling=Resampling.bilinear).astype('float32') / 10000.0
        
    with rasterio.open(item_s2.assets["B08"].href) as src_nir:
        nir = src_nir.read(1, window=win_s2, out_shape=target_shape, resampling=Resampling.bilinear).astype('float32') / 10000.0

    ndvi = np.where((nir + red) == 0, 0, (nir - red) / (nir + red))
    
    return dem, slope, ndvi, profile

# ==========================================
# 3. Machine Learning Species Modeling
# ==========================================
def run_kmeans_classification(dem, slope, ndvi, profile):
    """
    Unsupervised K-Means classification — per ClimateChange.txt Módulo 3.1.
    Discovers natural terrain clusters with no training data whatsoever.
    """
    print("\n[INFO] Running Unsupervised K-Means Classification (k=4 classes)...")
    dem_flat   = dem.flatten()
    slope_flat = slope.flatten()
    ndvi_flat  = ndvi.flatten()
    
    X_full = np.column_stack((dem_flat, slope_flat, ndvi_flat))
    valid_mask = np.isfinite(X_full).all(axis=1)
    X_valid = X_full[valid_mask]
    
    # Normalize each feature to [0,1] so no single variable dominates
    X_norm = (X_valid - X_valid.min(axis=0)) / (X_valid.max(axis=0) - X_valid.min(axis=0) + 1e-9)
    
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = km.fit_predict(X_norm)
    
    km_map = np.full(dem.shape, np.nan)
    km_map.flat[np.where(valid_mask)[0]] = labels
    
    out_tif = os.path.join(OUT_DIR, "kmeans_unsupervised.tif")
    profile.update(dtype=rasterio.float32, nodata=np.nan)
    with rasterio.open(out_tif, 'w', **profile) as dst:
        dst.write(km_map.astype('float32'), 1)
    print(f"       [SUCCESS] K-Means TIFF saved: {out_tif}")
    return km_map


def train_and_predict_niche(dem, slope, ndvi, profile):
    print("\n[INFO] Simulating Patagonian Huemul occurrence data...")
    # Flatten the arrays to 1D for Scikit-Learn
    dem_flat = dem.flatten()
    slope_flat = slope.flatten()
    ndvi_flat = ndvi.flatten()
    
    # Create the Feature Matrix (X)
    X_full = np.column_stack((dem_flat, slope_flat, ndvi_flat))
    
    # We simulate a "Truth" rule: 
    # Huemul prefers Elevation < 800m AND Slope < 20 deg AND NDVI > 0.4
    # We will generate synthetic samples based on this to train the model.
    print("       Generating 1000 synthetic training points...")
    np.random.seed(42)
    sample_indices = np.random.choice(len(X_full), size=1000, replace=False)
    
    X_train = X_full[sample_indices]
    
    # Generate labels (Y) with some natural noise
    y_train = []
    for row in X_train:
        elev, slp, veg = row[0], row[1], row[2]
        if elev < 800 and slp < 20 and veg > 0.4:
            # 80% chance of presence in ideal habitat (adds real-world noise)
            y_train.append(1 if np.random.rand() > 0.2 else 0)
        else:
            # 5% chance of being found outside ideal habitat
            y_train.append(1 if np.random.rand() > 0.95 else 0)
            
    y_train = np.array(y_train)
    
    print("\n[INFO] Training Random Forest Machine Learning Model...")
    # Train the SDM
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    rf.fit(X_train, y_train)
    
    print("       Feature Importances (Elevation / Slope / NDVI):")
    importances = rf.feature_importances_
    feat_names = ['Elevation', 'Slope', 'NDVI']
    for name, imp in zip(feat_names, importances):
        bar = '█' * int(imp * 40)
        print(f"         {name:12s}: {bar} {imp:.3f}")
    
    print("\n[INFO] Predicting Habitat Probability across Torres del Paine...")
    X_full = np.nan_to_num(X_full, 0)
    probabilities = rf.predict_proba(X_full)[:, 1]
    niche_map = probabilities.reshape(dem.shape)
    
    print("\n[INFO] Exporting Geocoded TIFFs for ArcGIS/ENVI...")
    out_tif = os.path.join(OUT_DIR, "ecological_niche_model.tif")
    with rasterio.open(out_tif, 'w', **profile) as dst:
        dst.write(niche_map.astype('float32'), 1)
    print(f"       [SUCCESS] Exported TIFF: {out_tif}")
    
    return niche_map, importances, feat_names


def generate_plots(dem, niche_map, km_map, importances, feat_names):
    print("\n[INFO] Generating Multi-Panel Output Chart...")
    fig, axs = plt.subplots(1, 3, figsize=(21, 7))
    
    # Panel 1: DEM context
    im1 = axs[0].imshow(dem, cmap='terrain')
    axs[0].set_title("Environmental Predictor\nDigital Elevation Model")
    fig.colorbar(im1, ax=axs[0], shrink=0.7)
    axs[0].axis('off')
    
    # Panel 2: K-Means unsupervised clusters
    im2 = axs[1].imshow(km_map, cmap='Set1', vmin=0, vmax=3)
    axs[1].set_title("K-Means Unsupervised Classification\n(4 terrain classes, no training data)")
    fig.colorbar(im2, ax=axs[1], shrink=0.7, ticks=[0,1,2,3], label='Class')
    axs[1].axis('off')
    
    # Panel 3: Supervised Niche Map
    im3 = axs[2].imshow(niche_map, cmap='YlGn', vmin=0, vmax=1)
    axs[2].set_title("Supervised Random Forest\nHuemul Habitat Suitability")
    fig.colorbar(im3, ax=axs[2], shrink=0.7, label='Probability')
    axs[2].axis('off')
    
    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, "ecological_niche_model.png")
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    print(f"       [SUCCESS] Chart saved: {plot_path}")

def main():
    print("=======================================================")
    print(" GEOCASCADE PIPELINE - ECOLOGICAL NICHE MODELING       ")
    print("=======================================================")
    dem, slope, ndvi, profile = fetch_environmental_layers()
    km_map = run_kmeans_classification(dem, slope, ndvi, profile)
    niche_map, importances, feat_names = train_and_predict_niche(dem, slope, ndvi, profile)
    generate_plots(dem, niche_map, km_map, importances, feat_names)
    print("\n[SUCCESS] Script 12 Complete!")

if __name__ == "__main__":
    main()
