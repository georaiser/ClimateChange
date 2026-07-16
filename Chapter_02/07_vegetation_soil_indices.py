"""
Chapter 2: 07_vegetation_soil_indices.py
==========================================
Complete Spectral Index Suite (9 Indices)

Academic Objective:
  Spectral indices are dimensionless mathematical ratios of band reflectances
  that amplify subtle differences between land cover types. This script computes
  9 indices covering vegetation health, soil exposure, water, ice, and moisture.

Indices computed:
  VEGETATION   NDVI  (NIR-Red)/(NIR+Red)          -- density, health
  VEGETATION   EVI   Enhanced, corrects atmosphere  -- dense forest, saturation
  VEGETATION   SAVI  Soil-Adjusted, sparse regions  -- Patagonian steppe
  SOIL/BARE    BSI   Bare Soil Index                -- erosion, overgrazing
  WATER        NDWI  (Green-NIR)/(Green+NIR)        -- open water bodies
  ICE/SNOW     NDSI  (Green-SWIR1)/(Green+SWIR1)   -- glacier extent
  GLACIER      NDGI  (Green-Red)/(Green+Red)        -- turbid glacial ice
  MOISTURE     NDMI  (NIR-SWIR1)/(NIR+SWIR1)       -- canopy water content
  BURN RATIO   NBR   (NIR-SWIR2)/(NIR+SWIR2)       -- fire scars, char

NEW vs previous version:
  - 9 indices (was 7) -- added NDMI and NBR
  - Ch01 local S2 data fallback (no re-download if already on disk)
  - compress="lzw" in all GeoTIFF profiles
  - ENVI/ArcGIS Pro compatibility notes
  - Dark-mode 3x3 panel figure
  - Summary statistics printed to console
  - Threshold masks exported (NDSI>0.4 glacier mask, NDWI>0.3 water mask)
  - matplotlib.use("Agg") + sys.stdout.reconfigure

Outputs:
  data/processed/indices/{ndvi,evi,savi,bsi,ndwi,ndsi,ndgi,ndmi,nbr}.tif
  data/processed/spectral_indices_all.png (3x3 dark panel)
  data/processed/indices/glacier_mask_ndsi.tif
  data/processed/indices/water_mask_ndwi.tif
  data/processed/index_statistics.csv

ArcGIS Pro: Add any .tif as raster. NDVI: Symbology > Stretched > Green.
            NDSI: Symbology > Classified > Blue-White (0.4 threshold).
ENVI 5.6:   File > Open > .tif. Band Math for custom thresholds.
            Use Scatter Plot for NDVI vs NDMI water-stress analysis.

Run:
  conda activate geocascade_env
  python Chapter_02/07_vegetation_soil_indices.py

Dependencies: rasterio, numpy, matplotlib, pandas, pystac-client, planetary-computer, pyproj
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

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
CH01_DIR  = os.path.join(os.path.dirname(BASE_DIR), "Chapter_01")
OUT_DIR   = os.path.join(BASE_DIR, "data", "processed", "indices")
PLOT_DIR  = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

BBOX       = [-73.30, -51.10, -72.90, -50.80]
DATE_RANGE = "2023-01-01/2023-02-28"

DARK_BG = "#0d1117"
DARK_AX = "#161b22"
C_TEXT  = "#e6edf3"
C_GREY  = "#8b949e"


# ---------------------------------------------------------------------------
# 1. Index formulas (NaN-safe)
# ---------------------------------------------------------------------------
def safe_ratio(num, den, eps=1e-6):
    """NaN-safe band ratio. Returns NaN where |denominator| < eps."""
    return np.where(np.abs(den) < eps, np.nan, num / den)

def ndvi(nir, red):
    """NDVI = (NIR - Red) / (NIR + Red)   Range [-1, 1], high = dense veg."""
    return safe_ratio(nir - red, nir + red)

def evi(nir, red, blue):
    """EVI = 2.5 * (NIR-Red) / (NIR + 6*Red - 7.5*Blue + 1)
    Corrects atmospheric path radiance and soil background."""
    return safe_ratio(2.5 * (nir - red), nir + 6.0 * red - 7.5 * blue + 1.0)

def savi(nir, red, L=0.5):
    """SAVI = (NIR - Red)*(1+L) / (NIR + Red + L)
    L=0.5 reduces soil noise in sparse-canopy Patagonian steppe."""
    return safe_ratio((nir - red) * (1.0 + L), nir + red + L)

def bsi(swir1, red, nir, blue):
    """BSI = ((SWIR1+Red) - (NIR+Blue)) / ((SWIR1+Red) + (NIR+Blue))
    High = bare soil or rock. Negative = vegetation cover."""
    return safe_ratio((swir1 + red) - (nir + blue), (swir1 + red) + (nir + blue))

def ndwi(green, nir):
    """NDWI = (Green - NIR) / (Green + NIR)
    Positive values (> 0.3) indicate open water bodies."""
    return safe_ratio(green - nir, green + nir)

def ndsi(green, swir1):
    """NDSI = (Green - SWIR1) / (Green + SWIR1)
    Values > 0.4 indicate snow or ice (glacier extent mask)."""
    return safe_ratio(green - swir1, green + swir1)

def ndgi(green, red):
    """NDGI = (Green - Red) / (Green + Red)
    Discriminates turbid glacial meltwater from rock and sediment."""
    return safe_ratio(green - red, green + red)

def ndmi(nir, swir1):
    """NDMI = (NIR - SWIR1) / (NIR + SWIR1)
    Sensitive to canopy water content and leaf moisture.
    High = high moisture, Low = drought stress.
    Complements NDVI: NDVI shows density, NDMI shows moisture status."""
    return safe_ratio(nir - swir1, nir + swir1)

def nbr(nir, swir2):
    """NBR = (NIR - SWIR2) / (NIR + SWIR2)
    Normalized Burn Ratio. Used to map fire scars.
    Post-fire: NIR drops, SWIR2 rises -> NBR strongly negative.
    Burned area threshold: delta-NBR (pre - post) > 0.1."""
    return safe_ratio(nir - swir2, nir + swir2)


# ---------------------------------------------------------------------------
# 2. Locate Sentinel-2 data (local Ch01 first, STAC fallback)
# ---------------------------------------------------------------------------
def load_s2_bands():
    """
    Strategy:
      1. Use Chapter_01 local S2 L2A bands if available
      2. Fall back to Planetary Computer STAC streaming
    Returns dict of {band_name: numpy_array} + rasterio profile
    """
    import glob

    # Try local first
    s2_dirs = sorted(glob.glob(os.path.join(CH01_DIR, "data", "raw", "sentinel2_l2a_*")))
    if s2_dirs:
        s2_dir = s2_dirs[-1]
        needed = {"B02": None, "B03": None, "B04": None, "B08": None, "B11": None, "B12": None}
        for band in needed:
            p = os.path.join(s2_dir, f"{band}.tif")
            if os.path.exists(p):
                needed[band] = p
        if all(v is not None for k, v in needed.items() if k != "B12"):
            print(f"  [OK] Local S2 data: {os.path.basename(s2_dir)}")
            return _read_local_bands(needed, s2_dir)

    # STAC fallback
    print("  Querying Planetary Computer STAC...")
    return _read_stac_bands()


def _read_local_bands(band_paths, s2_dir):
    """Read all required bands from local TIF files."""
    from pyproj import Transformer
    bands = {}
    profile = None
    target_shape = None
    window_10m = None

    # B02 defines the 10m reference grid
    with rasterio.open(band_paths["B02"]) as src:
        transformer_10m = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        minx, miny = transformer_10m.transform(BBOX[0], BBOX[1])
        maxx, maxy = transformer_10m.transform(BBOX[2], BBOX[3])
        window_10m    = from_bounds(minx, miny, maxx, maxy, src.transform)
        win_transform = rasterio.windows.transform(window_10m, src.transform)
        target_shape  = (int(round(window_10m.height)), int(round(window_10m.width)))
        profile = src.profile.copy()
        profile.update(
            dtype="float32", count=1, nodata=-9999,
            height=target_shape[0], width=target_shape[1],
            transform=win_transform, compress="lzw"
        )
        bands["B02"] = src.read(1, window=window_10m).astype("float32") / 10000.0

    for band in ["B03", "B04", "B08"]:
        if band_paths.get(band):
            with rasterio.open(band_paths[band]) as src:
                bands[band] = src.read(1, window=window_10m).astype("float32") / 10000.0

    # B11 + B12: 20m -- must use own independent window
    for band in ["B11", "B12"]:
        if band_paths.get(band) and os.path.exists(band_paths[band]):
            with rasterio.open(band_paths[band]) as src_b:
                t_b = Transformer.from_crs("EPSG:4326", src_b.crs, always_xy=True)
                mnx, mny = t_b.transform(BBOX[0], BBOX[1])
                mxx, mxy = t_b.transform(BBOX[2], BBOX[3])
                win_b    = from_bounds(mnx, mny, mxx, mxy, src_b.transform)
                bands[band] = src_b.read(
                    1, window=win_b,
                    out_shape=target_shape,
                    resampling=rasterio.enums.Resampling.bilinear
                ).astype("float32") / 10000.0
        else:
            print(f"  [WARN] {band} not found locally -- using zeros (affects BSI/NDMI/NBR/NDSI)")
            bands[band] = np.zeros(target_shape, dtype="float32")

    print(f"  Bands loaded: {list(bands.keys())} | Shape: {target_shape}")
    return bands, profile


def _read_stac_bands():
    """Read bands from Planetary Computer COG URLs (cloud-native)."""
    from pystac_client import Client
    import planetary_computer as pc
    from pyproj import Transformer

    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace
    )
    search = catalog.search(
        collections=["sentinel-2-l2a"], bbox=BBOX,
        datetime=DATE_RANGE, query={"eo:cloud_cover": {"lt": 15}}
    )
    items = list(search.items())
    if not items:
        raise ValueError("No cloud-free S2 images found. Try wider date range or cloud threshold.")

    item = sorted(items, key=lambda i: i.properties.get("eo:cloud_cover", 99))[0]
    print(f"  [OK] STAC scene: {item.id}  (cloud={item.properties.get('eo:cloud_cover',0):.1f}%)")

    bands = {}
    profile = None
    target_shape = None
    window_10m = None

    with rasterio.open(item.assets["B02"].href) as src:
        t = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        minx, miny = t.transform(BBOX[0], BBOX[1])
        maxx, maxy = t.transform(BBOX[2], BBOX[3])
        window_10m    = from_bounds(minx, miny, maxx, maxy, src.transform)
        win_transform = rasterio.windows.transform(window_10m, src.transform)
        target_shape  = (int(round(window_10m.height)), int(round(window_10m.width)))
        profile = src.profile.copy()
        profile.update(
            dtype="float32", count=1, nodata=-9999,
            height=target_shape[0], width=target_shape[1],
            transform=win_transform, compress="lzw"
        )
        bands["B02"] = src.read(1, window=window_10m).astype("float32") / 10000.0

    for band in ["B03", "B04", "B08"]:
        with rasterio.open(item.assets[band].href) as src:
            bands[band] = src.read(1, window=window_10m).astype("float32") / 10000.0

    # B11 (20m) and B12 (20m) -- independent window
    for band in ["B11", "B12"]:
        if band in item.assets:
            with rasterio.open(item.assets[band].href) as src_b:
                t_b = Transformer.from_crs("EPSG:4326", src_b.crs, always_xy=True)
                mnx, mny = t_b.transform(BBOX[0], BBOX[1])
                mxx, mxy = t_b.transform(BBOX[2], BBOX[3])
                win_b    = from_bounds(mnx, mny, mxx, mxy, src_b.transform)
                bands[band] = src_b.read(
                    1, window=win_b, out_shape=target_shape,
                    resampling=rasterio.enums.Resampling.bilinear
                ).astype("float32") / 10000.0

    print(f"  Bands loaded: {list(bands.keys())} | Shape: {target_shape}")
    return bands, profile


# ---------------------------------------------------------------------------
# 3. Compute all 9 indices
# ---------------------------------------------------------------------------
def compute_indices(bands):
    b = bands
    results = {}

    blue  = b.get("B02", None)
    green = b.get("B03", None)
    red   = b.get("B04", None)
    nir   = b.get("B08", None)
    swir1 = b.get("B11", None)
    swir2 = b.get("B12", None)

    if nir is not None and red is not None:
        results["ndvi"] = ndvi(nir, red)
    if nir is not None and red is not None and blue is not None:
        results["evi"]  = evi(nir, red, blue)
    if nir is not None and red is not None:
        results["savi"] = savi(nir, red)
    if swir1 is not None and red is not None and nir is not None and blue is not None:
        results["bsi"]  = bsi(swir1, red, nir, blue)
    if green is not None and nir is not None:
        results["ndwi"] = ndwi(green, nir)
    if green is not None and swir1 is not None:
        results["ndsi"] = ndsi(green, swir1)
    if green is not None and red is not None:
        results["ndgi"] = ndgi(green, red)
    if nir is not None and swir1 is not None:
        results["ndmi"] = ndmi(nir, swir1)
    if nir is not None and swir2 is not None:
        results["nbr"]  = nbr(nir, swir2)

    print(f"\n  Computed {len(results)} indices: {list(results.keys())}")
    return results


# ---------------------------------------------------------------------------
# 4. Save GeoTIFFs + threshold masks
# ---------------------------------------------------------------------------
def save_tifs(indices, profile):
    print("\n  Saving GeoTIFFs...")
    for name, arr in indices.items():
        out = os.path.join(OUT_DIR, f"{name}.tif")
        safe = np.nan_to_num(arr, nan=-9999).astype("float32")
        with rasterio.open(out, "w", **profile) as dst:
            dst.write(safe, 1)
            dst.update_tags(
                index_name=name.upper(),
                nodata_value="-9999",
                scale_note="Sentinel-2 L2A reflectance, scale=1/10000",
                arcgis_note="ArcGIS Pro: Symbology > Stretched or Classified",
                envi_note="ENVI 5.6: File > Open, use Band Math for thresholds"
            )
        print(f"    [OK] {name.upper()}: {out}")

    # Glacier mask (NDSI > 0.4)
    if "ndsi" in indices:
        glacier_mask = np.where(indices["ndsi"] > 0.4, 1.0, -9999.0).astype("float32")
        out = os.path.join(OUT_DIR, "glacier_mask_ndsi.tif")
        with rasterio.open(out, "w", **profile) as dst:
            dst.write(glacier_mask, 1)
            dst.update_tags(description="NDSI > 0.4 glacier extent mask (1=ice, -9999=no ice)")
        print(f"    [OK] Glacier mask: {out}")

    # Water mask (NDWI > 0.3)
    if "ndwi" in indices:
        water_mask = np.where(indices["ndwi"] > 0.3, 1.0, -9999.0).astype("float32")
        out = os.path.join(OUT_DIR, "water_mask_ndwi.tif")
        with rasterio.open(out, "w", **profile) as dst:
            dst.write(water_mask, 1)
            dst.update_tags(description="NDWI > 0.3 open water mask (1=water, -9999=no water)")
        print(f"    [OK] Water mask: {out}")


# ---------------------------------------------------------------------------
# 5. Summary statistics CSV
# ---------------------------------------------------------------------------
def save_statistics(indices):
    rows = []
    for name, arr in indices.items():
        valid = arr[np.isfinite(arr) & (arr != -9999)]
        rows.append({
            "index":    name.upper(),
            "n_pixels": int(valid.size),
            "mean":     round(float(np.nanmean(valid)), 4) if valid.size > 0 else None,
            "std":      round(float(np.nanstd(valid)),  4) if valid.size > 0 else None,
            "min":      round(float(np.nanmin(valid)),  4) if valid.size > 0 else None,
            "max":      round(float(np.nanmax(valid)),  4) if valid.size > 0 else None,
            "pct_positive": round(float((valid > 0).mean() * 100), 1) if valid.size > 0 else None,
        })
    df = pd.DataFrame(rows)
    csv_path = os.path.join(PLOT_DIR, "index_statistics.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\n  [OK] Statistics CSV: {csv_path}")

    # Print summary table
    print(f"\n  {'Index':<6s}  {'Mean':>7s}  {'Std':>6s}  {'Min':>7s}  {'Max':>6s}  {'%>0':>5s}")
    print("  " + "-" * 50)
    for row in rows:
        m = f"{row['mean']:+7.3f}" if row['mean'] is not None else "    N/A"
        s = f"{row['std']:6.3f}"  if row['std']  is not None else "   N/A"
        mn= f"{row['min']:+7.3f}" if row['min']  is not None else "    N/A"
        mx= f"{row['max']:+6.3f}" if row['max']  is not None else "   N/A"
        p = f"{row['pct_positive']:5.1f}%" if row['pct_positive'] is not None else "  N/A"
        print(f"  {row['index']:<6s}  {m}  {s}  {mn}  {mx}  {p}")
    return df


# ---------------------------------------------------------------------------
# 6. 3x3 Dark-mode panel plot
# ---------------------------------------------------------------------------
def plot_panel(indices):
    print("\n  Building 3x3 index panel...")

    panel_cfg = [
        ("ndvi", "RdYlGn",   -0.2, 0.8,  "NDVI -- Vegetation Health"),
        ("evi",  "RdYlGn",   -0.2, 0.8,  "EVI -- Enhanced Vegetation"),
        ("savi", "RdYlGn",   -0.2, 0.8,  "SAVI -- Soil-Adjusted"),
        ("bsi",  "copper",   -0.5, 0.5,  "BSI -- Bare Soil / Rock"),
        ("ndwi", "Blues",    -0.5, 0.6,  "NDWI -- Open Water"),
        ("ndsi", "cool",     -0.3, 0.9,  "NDSI -- Snow & Ice"),
        ("ndgi", "PuBuGn",   -0.3, 0.5,  "NDGI -- Glacier Green Ice"),
        ("ndmi", "BrBG",     -0.5, 0.7,  "NDMI -- Canopy Moisture"),
        ("nbr",  "RdGy",     -0.7, 0.7,  "NBR -- Burn Ratio"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(20, 18), facecolor=DARK_BG)
    fig.suptitle("Spectral Index Suite (9) -- Torres del Paine, Patagonia",
                 color=C_TEXT, fontsize=14, fontweight="bold", y=0.98)

    for ax, (name, cmap, vmin, vmax, title) in zip(axes.flatten(), panel_cfg):
        ax.set_facecolor(DARK_AX)
        if name in indices:
            data = np.where(indices[name] == -9999, np.nan, indices[name])
            im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
            cb.ax.tick_params(colors=C_TEXT, labelsize=7)
            # Print stats on panel
            valid = data[np.isfinite(data)]
            if valid.size > 0:
                ax.text(0.02, 0.04, f"mean={np.nanmean(valid):+.3f}",
                        transform=ax.transAxes, color=C_TEXT, fontsize=7.5,
                        bbox=dict(fc=DARK_BG, ec="none", alpha=0.7))
        else:
            ax.text(0.5, 0.5, f"{name.upper()}\nnot computed",
                    ha="center", va="center", color=C_GREY, transform=ax.transAxes)
        ax.set_title(title, color=C_TEXT, fontsize=9, fontweight="bold")
        ax.axis("off")

    out_png = os.path.join(PLOT_DIR, "spectral_indices_all.png")
    fig.savefig(out_png, dpi=180, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  [OK] 3x3 panel: {out_png}")
    return out_png


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print(" GEOCASCADE - SPECTRAL INDEX SUITE (9 INDICES)")
    print(f" BBOX: {BBOX}")
    print("=" * 65)

    print("\n[1/5] Loading Sentinel-2 bands...")
    try:
        bands, profile = load_s2_bands()
    except Exception as e:
        print(f"\n  ERROR loading bands: {e}")
        print("  Run Chapter_01/02_satellite_acquisition.py first.")
        return

    print("\n[2/5] Computing 9 spectral indices...")
    indices = compute_indices(bands)

    print("\n[3/5] Saving GeoTIFFs...")
    save_tifs(indices, profile)

    print("\n[4/5] Statistics...")
    save_statistics(indices)

    print("\n[5/5] Generating 3x3 panel figure...")
    plot_panel(indices)

    print("\n" + "=" * 65)
    print(" SPECTRAL INDICES COMPLETE")
    print("=" * 65)
    print(f"  TIFs   : {OUT_DIR}")
    print(f"  Plot   : {os.path.join(PLOT_DIR, 'spectral_indices_all.png')}")
    print(f"  Stats  : {os.path.join(PLOT_DIR, 'index_statistics.csv')}")
    print()
    print("  ArcGIS Pro: Add any .tif as raster layer.")
    print("    NDVI -> Symbology > Stretched > Green color ramp")
    print("    NDSI -> Symbology > Classified, threshold 0.4 = glacier")
    print("    NDWI -> Symbology > Classified, threshold 0.3 = water")
    print("  ENVI 5.6  : File > Open > .tif")
    print("    Band Math: b1 gt 0.4 for glacier mask from NDSI TIF")
    print("=" * 65)


if __name__ == "__main__":
    main()
