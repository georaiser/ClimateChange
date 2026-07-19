"""
07_envi_spectral_analysis.py
==============================
GeoCascade Chapter 14 -- ArcGIS Pro + ENVI Professional Workflows
Torres del Paine, Patagonia, Chile

# NOTE: This script mirrors the ENVI Python API workflow.
# In production, replace open() / rasterio calls with:
#   envi = ENVI()  (from envi_py)
#   raster = envi.open_raster(path)

PURPOSE
-------
Loads Sentinel-2 L2A imagery and computes key spectral indices:
  NDVI  = (B8 - B4) / (B8 + B4)   -- vegetation health
  NBR   = (B8 - B12) / (B8 + B12) -- burn / fire scar detection
  NDWI  = (B3 - B8) / (B3 + B8)   -- water body mapping
  NDSI  = (B3 - B11) / (B3 + B11) -- snow / glacier extent
  EVI   = 2.5 * (B8-B4) / (B8 + 6*B4 - 7.5*B2 + 1)  -- enhanced veg index

Falls back to realistic synthetic 6-band array if no real imagery is found.

OUTPUTS
-------
  data/processed/envi_outputs/ndvi.tif
  data/processed/envi_outputs/nbr.tif
  data/processed/envi_outputs/ndwi.tif
  data/processed/envi_outputs/ndsi.tif
  data/processed/envi_outputs/evi.tif
  data/processed/envi_outputs/spectral_analysis.png  -- 5-panel figure

ENVI 5.6 EQUIVALENT
--------------------
  Toolbox > Spectral > Vegetation Index Calculator
  Toolbox > Band Math (expression: (b4-b3)/(b4+b3) for NDVI)
  File > Save As > select output band
  Display > Cursor Value to read pixel values interactively

ARCGIS PRO EQUIVALENT
----------------------
  Raster Functions > Indices > NDVI / EVI
  Imagery > Raster Function > Band Arithmetic

RUN
---
  python Chapter_14/07_envi_spectral_analysis.py
"""

import sys
import os
import warnings
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

try:
    import rasterio
    from rasterio.transform import from_bounds
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("[WARN] rasterio not available -- outputs will be NumPy only.")

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.dirname(BASE_DIR)
ENVI_DIR = os.path.join(BASE_DIR, "data", "processed", "envi_outputs")
os.makedirs(ENVI_DIR, exist_ok=True)

# Sentinel-2 band mapping (10-20m bands resample to common grid)
# B2=Blue, B3=Green, B4=Red, B8=NIR, B11=SWIR1, B12=SWIR2
BAND_NAMES = ["B2", "B3", "B4", "B8", "B11", "B12"]
BAND_DESC  = {
    "B2":  "Blue     (490nm)",
    "B3":  "Green    (560nm)",
    "B4":  "Red      (665nm)",
    "B8":  "NIR      (842nm)",
    "B11": "SWIR-1   (1610nm)",
    "B12": "SWIR-2   (2190nm)",
}

BBOX = [-73.5, -51.5, -72.5, -50.5]

DARK_BG = "#0d1117"
DARK_AX = "#161b22"
C_TEXT  = "#e6edf3"
C_GREY  = "#8b949e"

GRID_SHAPE = (120, 120)


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

def load_sentinel2_bands(root: str) -> dict:
    """
    Attempt to load Sentinel-2 L2A bands from Chapter 02 outputs.
    Returns dict {band_name: 2D array} or generates synthetic data.
    """
    # Search for processed Sentinel-2 TIFs
    patterns = [
        os.path.join(root, "Chapter_02", "data", "processed", "**", "*B0*.tif"),
        os.path.join(root, "Chapter_02", "data", "raw",       "**", "*B0*.tif"),
    ]
    found_tifs = []
    for pat in patterns:
        found_tifs.extend(glob.glob(pat, recursive=True))

    if found_tifs and HAS_RASTERIO:
        print(f"  Found {len(found_tifs)} Sentinel-2 band files -- loading ...")
        bands = {}
        ref_shape = None
        for tif in sorted(found_tifs)[:6]:
            with rasterio.open(tif) as src:
                data = src.read(1).astype(np.float32)
                if ref_shape is None:
                    ref_shape = data.shape
                if data.shape != ref_shape:
                    from scipy.ndimage import zoom
                    data = zoom(data, (ref_shape[0]/data.shape[0],
                                       ref_shape[1]/data.shape[1]))
                bname = os.path.basename(tif).split("_")[0]
                bands[bname] = data
                print(f"    {bname}: {data.shape}, range "
                      f"{np.nanmin(data):.0f}-{np.nanmax(data):.0f}")
        if len(bands) >= 4:
            return bands

    # Synthetic fallback -- generate realistic 6-band Patagonian image
    print("  [SYNTHETIC] Generating realistic Sentinel-2 6-band array ...")
    rows, cols = GRID_SHAPE
    rng = np.random.default_rng(seed=42)
    lons = np.linspace(BBOX[0], BBOX[2], cols)
    lats = np.linspace(BBOX[3], BBOX[1], rows)
    LON, LAT = np.meshgrid(lons, lats)

    # Fraction of vegetation cover (increases east -> west)
    veg_frac = np.clip(0.1 + 0.7 * (LON - BBOX[0]) / (BBOX[2] - BBOX[0]), 0, 1)
    # Water body (central lake patch)
    water = np.zeros((rows, cols))
    water[rows//3:rows//2, cols//3:cols//2] = 1.0
    # Glacier / snow (northwest corner)
    snow = np.zeros((rows, cols))
    snow[:rows//5, :cols//4] = 1.0

    def band(B_soil, B_veg, B_water, B_snow, noise_std=200):
        b = (B_soil * (1 - veg_frac - water - snow).clip(0, 1) +
             B_veg  * veg_frac +
             B_water * water +
             B_snow  * snow)
        b += rng.normal(0, noise_std, (rows, cols))
        return b.clip(0, 10000).astype(np.float32)

    # Typical Sentinel-2 L2A reflectance values (x10000)
    bands = {
        "B2":  band(800,  400,  200,  9000, 150),
        "B3":  band(1000, 600,  300,  9200, 150),
        "B4":  band(1200, 500,  200,  8800, 200),
        "B8":  band(2500, 7000, 300,  9500, 300),
        "B11": band(2000, 3000, 400,  4000, 250),
        "B12": band(1800, 2200, 350,  3500, 250),
    }
    for b, arr in bands.items():
        print(f"    {b} ({BAND_DESC[b]}): shape={arr.shape}  "
              f"mean={arr.mean():.0f}")
    return bands


# ---------------------------------------------------------------------------
# INDEX COMPUTATION
# ---------------------------------------------------------------------------

def safe_norm_index(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Compute normalised difference index, avoiding division by zero."""
    with np.errstate(invalid="ignore", divide="ignore"):
        idx = np.where(den != 0, num / den, np.nan)
    return np.clip(idx, -1, 1).astype(np.float32)


def compute_indices(bands: dict) -> dict:
    """Compute 5 spectral indices from Sentinel-2 bands."""
    # Normalise bands to 0-1 reflectance
    b = {k: v / 10000.0 for k, v in bands.items()}

    indices = {}

    indices["ndvi"] = safe_norm_index(b["B8"] - b["B4"],
                                       b["B8"] + b["B4"])

    indices["nbr"]  = safe_norm_index(b["B8"] - b["B12"],
                                       b["B8"] + b["B12"])

    indices["ndwi"] = safe_norm_index(b["B3"] - b["B8"],
                                       b["B3"] + b["B8"])

    # NDSI (Normalised Difference Snow Index)
    indices["ndsi"] = safe_norm_index(b["B3"] - b["B11"],
                                       b["B3"] + b["B11"])

    # EVI (Enhanced Vegetation Index)
    denom = b["B8"] + 6 * b["B4"] - 7.5 * b["B2"] + 1
    with np.errstate(invalid="ignore", divide="ignore"):
        evi = np.where(denom != 0,
                       2.5 * (b["B8"] - b["B4"]) / denom,
                       np.nan)
    indices["evi"] = np.clip(evi, -1, 1).astype(np.float32)

    for name, arr in indices.items():
        valid = arr[~np.isnan(arr)]
        print(f"  {name.upper():<6}: min={valid.min():.3f}  "
              f"max={valid.max():.3f}  mean={valid.mean():.3f}  "
              f"std={valid.std():.3f}")
    return indices


# ---------------------------------------------------------------------------
# SAVE & PLOT
# ---------------------------------------------------------------------------

def save_index_tif(data: np.ndarray, name: str) -> None:
    if not HAS_RASTERIO:
        return
    out = np.nan_to_num(data, nan=-9999.0)
    rows, cols = data.shape
    t = from_bounds(*BBOX, cols, rows)
    path = os.path.join(ENVI_DIR, f"{name}.tif")
    with rasterio.open(path, "w", driver="GTiff", height=rows, width=cols,
                       count=1, dtype="float32", crs="EPSG:4326",
                       transform=t, nodata=-9999.0, compress="lzw") as dst:
        dst.write(out, 1)
        dst.update_tags(
            description=name.upper(),
            source="Sentinel-2 L2A -- GeoCascade Ch14",
            envi_note=f"ENVI: Toolbox > Band Math > ({name.upper()} formula)",
            arcgis_note="ArcGIS Pro: Raster Functions > Indices"
        )
    print(f"  [OK] {name}.tif -> {os.path.relpath(path, BASE_DIR)}")


INDEX_META = {
    "ndvi": ("RdYlGn",   "NDVI\n(Vegetation Health)",         "dense veg > 0.6"),
    "nbr":  ("RdYlGn",   "NBR\n(Burn / Fire Scar)",           "healthy = high"),
    "ndwi": ("Blues_r",  "NDWI\n(Water Bodies)",              "water > 0"),
    "ndsi": ("Blues",    "NDSI\n(Snow / Glacier)",            "snow > 0.4"),
    "evi":  ("YlGn",     "EVI\n(Enhanced Vegetation Index)", "dense veg > 0.5"),
}


def plot_indices(indices: dict, out_path: str) -> None:
    """5-panel spectral index figure."""
    names = list(INDEX_META.keys())
    fig   = plt.figure(figsize=(24, 6), facecolor=DARK_BG)
    gs    = gridspec.GridSpec(1, 5, figure=fig, wspace=0.22,
                              left=0.03, right=0.97, top=0.86, bottom=0.06)
    fig.text(0.5, 0.97,
             "GeoCascade Ch14 -- ENVI Spectral Index Analysis",
             ha="center", color=C_TEXT, fontsize=13, fontweight="bold")
    fig.text(0.5, 0.93,
             "Sentinel-2 L2A  |  Torres del Paine, Patagonia  |  "
             "NDVI | NBR | NDWI | NDSI | EVI",
             ha="center", color=C_GREY, fontsize=9)

    ext = [BBOX[0], BBOX[2], BBOX[1], BBOX[3]]

    for i, name in enumerate(names):
        ax = fig.add_subplot(gs[i])
        ax.set_facecolor(DARK_AX)
        for sp in ax.spines.values(): sp.set_color("#30363d")

        cmap, title, note = INDEX_META[name]
        data = indices[name]
        im = ax.imshow(data, cmap=cmap, vmin=-1, vmax=1,
                       origin="upper", extent=ext)
        cb = plt.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
        cb.ax.tick_params(colors=C_TEXT, labelsize=6)

        valid = data[~np.isnan(data)]
        ax.set_title(f"{title}\nmean={valid.mean():.3f}",
                     color=C_TEXT, fontsize=8, fontweight="bold", pad=4)
        ax.tick_params(colors=C_TEXT, labelsize=6)
        ax.set_xlabel("Lon", color=C_TEXT, fontsize=7)
        if i == 0:
            ax.set_ylabel("Lat", color=C_TEXT, fontsize=7)
        ax.text(0.03, 0.04, note, transform=ax.transAxes,
                color=C_GREY, fontsize=6, va="bottom")

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  [OK] Figure: {os.path.relpath(out_path, BASE_DIR)}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print(" GEOCASCADE Ch14 -- ENVI Spectral Analysis")
    print(" NDVI | NBR | NDWI | NDSI | EVI  |  Sentinel-2 L2A")
    print("=" * 65)

    print("\n[1/4] Loading Sentinel-2 bands ...")
    bands = load_sentinel2_bands(ROOT)

    print("\n[2/4] Computing spectral indices ...")
    indices = compute_indices(bands)

    print("\n[3/4] Saving index GeoTIFFs ...")
    for name, arr in indices.items():
        save_index_tif(arr, name)

    print("\n[4/4] Generating spectral analysis figure ...")
    plot_indices(indices,
                 os.path.join(ENVI_DIR, "spectral_analysis.png"))

    print("\n" + "=" * 65)
    print(" SPECTRAL ANALYSIS COMPLETE")
    print("=" * 65)
    for name in indices:
        print(f"  {name.upper():<6}: {ENVI_DIR}\\{name}.tif")
    print(f"  Figure : {ENVI_DIR}\\spectral_analysis.png")
    print()
    print("  ENVI 5.6:")
    print("    Toolbox > Spectral > Vegetation Index Calculator")
    print("    Select Sentinel-2 image > choose NDVI/EVI -> run")
    print("    Or use Band Math: (float(b4)-float(b3))/(float(b4)+float(b3))")
    print()
    print("  ArcGIS Pro:")
    print("    Add ndvi.tif -> Symbology > Stretched > Green-Yellow-Red")
    print("    Imagery tab > Raster Functions > NDVI")
    print()
    print("  Continue with: python Chapter_14/08_envi_atm_correction.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
