"""
Chapter 5: 14_moisture_stress_indices.py
==========================================
Moisture & Drought Stress Indices from Sentinel-2 SWIR

Academic Objective:
  NDVI shows how GREEN vegetation is, but not whether plants are STRESSED by drought.
  The key insight: water strongly absorbs Shortwave Infrared (SWIR, 1400-2500 nm).
  By comparing NIR (reflected by healthy leaf structure) with SWIR (absorbed by leaf water),
  we can quantify the moisture content of the plant canopy directly from space.

  This script computes four complementary moisture indicators:

  1. NDMI  (Normalized Difference Moisture Index, Gao 1996)
     Formula: (NIR - SWIR1) / (NIR + SWIR1)   [B08, B11]
     Range:   -1 to +1   Higher = wetter canopy / healthy plant water content
     Threshold: NDMI < 0.0 = moderate drought stress
                NDMI < -0.1 = severe drought stress

  2. MSI   (Moisture Stress Index, Rock et al. 1986)
     Formula: SWIR1 / NIR   [B11, B08]
     Range:   0 to +inf     Higher = more drought stress
     Inverse of NDMI: when SWIR rises (water lost) relative to NIR, MSI rises.

  3. NDWI  (Normalized Difference Water Index, McFeeters 1996)
     Formula: (Green - NIR) / (Green + NIR)   [B03, B08]
     Range:   -1 to +1     Higher = open water / flooded surface
     Distinct from NDMI: NDWI detects SURFACE water, NDMI detects CANOPY water.

  4. NMDI  (Normalized Multi-band Drought Index, Wang & Qu 2007)
     Formula: (NIR - (SWIR1 - SWIR2)) / (NIR + (SWIR1 - SWIR2))   [B08, B11, B12]
     More specific to canopy water than NDMI; less sensitive to bare soil background.

B11/B12 SWIR bands are at 20m native resolution. They are windowed independently
(with their own 20m transform) and bilinearly resampled to the 10m NIR grid.
NEVER use the same window object across bands with different native resolutions.

Connection to pipeline:
  Uses Ch02 local NDVI/indices if available.
  Falls back to Planetary Computer STAC streaming if not found.

Outputs:
  data/processed/moisture/ndmi.tif
  data/processed/moisture/msi.tif
  data/processed/moisture/ndwi.tif
  data/processed/moisture/nmdi.tif
  data/processed/moisture/moisture_statistics.csv
  data/processed/moisture/moisture_stress_indices.png   (4-panel dark)

ArcGIS Pro: Add ndmi.tif. Symbology > Stretched > Blue-Red diverging.
            Raster Calculator: Con("ndmi.tif" < -0.1, 1, 0) for severe drought mask.
ENVI 5.6:   Band Math: (b1-b2)/(b1+b2) where b1=B08, b2=B11 for NDMI.

Run:
  conda activate geocascade_env
  python Chapter_05/14_moisture_stress_indices.py

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
import matplotlib.gridspec as gridspec
import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CH02_DIR = os.path.join(os.path.dirname(BASE_DIR), "Chapter_02")
OUT_DIR  = os.path.join(BASE_DIR, "data", "processed", "moisture")
os.makedirs(OUT_DIR, exist_ok=True)

BBOX       = [-73.30, -51.10, -72.90, -50.80]
DATE_RANGE = "2023-01-01/2023-02-28"
CLOUD_MAX  = 10

DARK_BG = "#0d1117"
DARK_AX = "#161b22"
C_TEXT  = "#e6edf3"
C_GREY  = "#8b949e"
C_GOLD  = "#f39c12"


# ---------------------------------------------------------------------------
# 1. Index formulas
# ---------------------------------------------------------------------------
def ndmi(nir, swir1):
    """NDMI = (NIR - SWIR1) / (NIR + SWIR1). Range -1 to +1."""
    d = nir + swir1
    return np.where(np.abs(d) < 1e-6, np.nan, (nir - swir1) / d)

def msi(swir1, nir):
    """MSI = SWIR1 / NIR. Range 0 to inf. Higher = more stress."""
    return np.where(np.abs(nir) < 1e-6, np.nan, swir1 / nir)

def ndwi(green, nir):
    """NDWI = (Green - NIR) / (Green + NIR). Detects open water."""
    d = green + nir
    return np.where(np.abs(d) < 1e-6, np.nan, (green - nir) / d)

def nmdi(nir, swir1, swir2):
    """NMDI = (NIR - (SWIR1 - SWIR2)) / (NIR + (SWIR1 - SWIR2))."""
    diff = swir1 - swir2
    d    = nir + diff
    return np.where(np.abs(d) < 1e-6, np.nan, (nir - diff) / d)


# ---------------------------------------------------------------------------
# 2. Read band from STAC with proper independent window per resolution
# ---------------------------------------------------------------------------
def read_band(item, asset_key, target_shape, scale=10000.0):
    """
    Read one Sentinel-2 band using its OWN window (correct for its native resolution)
    then resample to target_shape. NEVER reuse a window from another resolution band.
    """
    from pyproj import Transformer
    with rasterio.open(item.assets[asset_key].href) as src:
        t    = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        mnx, mny = t.transform(BBOX[0], BBOX[1])
        mxx, mxy = t.transform(BBOX[2], BBOX[3])
        win  = from_bounds(mnx, mny, mxx, mxy, src.transform)
        arr  = src.read(1, window=win, out_shape=target_shape,
                        resampling=Resampling.bilinear).astype("float32")
    return np.clip(arr / scale, 0.0, 1.0)


# ---------------------------------------------------------------------------
# 3. Fetch Sentinel-2 data (local fallback first)
# ---------------------------------------------------------------------------
def fetch_bands():
    """
    Try to load bands from Ch02 processed TIFs first.
    Falls back to STAC streaming if not found.
    Returns: green, nir, swir1, swir2, profile, target_shape
    """
    ch02_ndvi = os.path.join(CH02_DIR, "data", "processed", "indices", "ndvi.tif")

    if os.path.exists(ch02_ndvi):
        print("  [OK] Found Ch02 processed data. Loading bands from local TIFs...")
        # Load from local Ch02 processed indices
        # We'll reconstruct bands from the raw Ch02 data folder
        ch02_raw = os.path.join(CH02_DIR, "data", "raw")
        scene_dirs = sorted([d for d in os.listdir(ch02_raw)
                             if os.path.isdir(os.path.join(ch02_raw, d))
                             and "sentinel2" in d.lower()]) if os.path.exists(ch02_raw) else []

        if scene_dirs:
            scene = os.path.join(ch02_raw, scene_dirs[-1])
            band_map = {"B03": None, "B08": None, "B11": None, "B12": None}
            for band in band_map:
                for fn in os.listdir(scene):
                    if band in fn and fn.endswith(".tif"):
                        band_map[band] = os.path.join(scene, fn)
                        break
            if all(v is not None for v in band_map.values()):
                print("  [OK] Using Ch02 raw scene bands directly.")
                with rasterio.open(band_map["B08"]) as src:
                    from pyproj import Transformer
                    t = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                    mnx, mny = t.transform(BBOX[0], BBOX[1])
                    mxx, mxy = t.transform(BBOX[2], BBOX[3])
                    win = from_bounds(mnx, mny, mxx, mxy, src.transform)
                    h = int(round(win.height)); w = int(round(win.width))
                    target_shape = (h, w)
                    profile = src.profile.copy()
                    profile.update(dtype="float32", count=1, nodata=-9999, compress="lzw",
                                   height=h, width=w,
                                   transform=rasterio.windows.transform(win, src.transform))
                    nir_arr = src.read(1, window=win).astype("float32")
                    nir_arr = np.clip(nir_arr / 10000.0, 0, 1)

                def _local(path):
                    from pyproj import Transformer as T2
                    with rasterio.open(path) as s:
                        tr = T2.from_crs("EPSG:4326", s.crs, always_xy=True)
                        mn, my_ = tr.transform(BBOX[0], BBOX[1])
                        mx, my2 = tr.transform(BBOX[2], BBOX[3])
                        ww = from_bounds(mn, my_, mx, my2, s.transform)
                        return np.clip(
                            s.read(1, window=ww, out_shape=target_shape,
                                   resampling=Resampling.bilinear).astype("float32") / 10000.0,
                            0, 1)

                return (_local(band_map["B03"]), nir_arr,
                        _local(band_map["B11"]), _local(band_map["B12"]),
                        profile, target_shape)

    # STAC fallback
    print("  Connecting to Planetary Computer...")
    try:
        from pystac_client import Client
        import planetary_computer as pc
        from pyproj import Transformer
    except ImportError:
        raise ImportError("pystac-client / planetary-computer not installed.")

    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                          modifier=pc.sign_inplace)
    search  = catalog.search(collections=["sentinel-2-l2a"], bbox=BBOX,
                             datetime=DATE_RANGE,
                             query={"eo:cloud_cover": {"lt": CLOUD_MAX}})
    items   = list(search.items())
    if not items:
        # Try relaxed cloud threshold
        search = catalog.search(collections=["sentinel-2-l2a"], bbox=BBOX,
                                datetime="2023-01-01/2023-03-31",
                                query={"eo:cloud_cover": {"lt": 30}})
        items  = list(search.items())
    if not items:
        raise ValueError(f"No Sentinel-2 found for BBOX {BBOX} in {DATE_RANGE}.")

    item = sorted(items, key=lambda i: i.properties.get("eo:cloud_cover", 99))[0]
    date = item.properties.get("datetime", "")[:10]
    cloud = item.properties.get("eo:cloud_cover", 0)
    print(f"  [OK] Scene: {item.id}  date={date}  cloud={cloud:.1f}%")

    # NIR (B08, 10m) — establishes master grid
    from pyproj import Transformer
    with rasterio.open(item.assets["B08"].href) as src:
        t = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        mnx, mny = t.transform(BBOX[0], BBOX[1])
        mxx, mxy = t.transform(BBOX[2], BBOX[3])
        win      = from_bounds(mnx, mny, mxx, mxy, src.transform)
        h = int(round(win.height)); w = int(round(win.width))
        target_shape = (h, w)
        profile = src.profile.copy()
        profile.update(dtype="float32", count=1, nodata=-9999, compress="lzw",
                       height=h, width=w,
                       transform=rasterio.windows.transform(win, src.transform))
        nir_arr = np.clip(
            src.read(1, window=win).astype("float32") / 10000.0, 0, 1)

    green_arr = read_band(item, "B03", target_shape)
    swir1_arr = read_band(item, "B11", target_shape)   # 20m → resampled
    swir2_arr = read_band(item, "B12", target_shape)   # 20m → resampled

    return green_arr, nir_arr, swir1_arr, swir2_arr, profile, target_shape


# ---------------------------------------------------------------------------
# 4. Save GeoTIFF
# ---------------------------------------------------------------------------
def save_tif(data, name, profile, description=""):
    out = os.path.join(OUT_DIR, f"{name}.tif")
    safe = np.nan_to_num(data.astype("float32"), nan=-9999)
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(safe, 1)
        dst.update_tags(description=description, nodata="-9999",
                        arcgis_note="Stretched symbology, Blue-Red diverging",
                        envi_note="File > Open, Linear stretch")
    print(f"  [OK] {name}.tif")
    return out


# ---------------------------------------------------------------------------
# 5. Statistics CSV
# ---------------------------------------------------------------------------
def save_stats(indices_dict):
    rows = []
    for name, arr in indices_dict.items():
        v = arr[np.isfinite(arr)]
        if v.size == 0:
            continue
        rows.append({
            "index": name,
            "mean":  round(float(v.mean()), 4),
            "std":   round(float(v.std()),  4),
            "min":   round(float(v.min()),  4),
            "max":   round(float(v.max()),  4),
            "pct_negative": round(float((v < 0).sum() / v.size * 100), 2),
        })
    df = pd.DataFrame(rows)
    csv = os.path.join(OUT_DIR, "moisture_statistics.csv")
    df.to_csv(csv, index=False, encoding="utf-8")
    print(f"\n  --- Moisture Index Statistics ---")
    for _, r in df.iterrows():
        print(f"  {r['index']:<6s}  mean={r['mean']:>7.4f}  std={r['std']:>6.4f}  "
              f"min={r['min']:>7.4f}  max={r['max']:>6.4f}")
    print(f"  [OK] Statistics CSV: {csv}")


# ---------------------------------------------------------------------------
# 6. 4-panel dark figure
# ---------------------------------------------------------------------------
def plot_moisture(ndmi_arr, msi_arr, ndwi_arr, nmdi_arr):
    print("\n  Building 4-panel moisture figure...")

    fig = plt.figure(figsize=(22, 16), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.22,
                            top=0.93, bottom=0.05, left=0.05, right=0.97)
    fig.text(0.5, 0.97, "Moisture & Drought Stress Indices -- Torres del Paine",
             ha="center", color=C_TEXT, fontsize=13, fontweight="bold")

    panels = [
        (ndmi_arr, "RdYlBu",  -0.3, 0.8,  "NDMI -- Canopy Moisture\n(Blue=wet, Red=dry)"),
        (msi_arr,  "YlOrRd",   0.3, 2.5,  "MSI -- Moisture Stress Index\n(Higher = more drought stress)"),
        (ndwi_arr, "Blues",   -0.5, 0.5,  "NDWI -- Surface Water Detection\n(Blue = open water)"),
        (nmdi_arr, "RdYlBu",  -0.5, 0.8,  "NMDI -- Multi-band Drought Index\n(Wang & Qu 2007)"),
    ]

    for i, (arr, cmap, vmin, vmax, title) in enumerate(panels):
        row, col = divmod(i, 2)
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor(DARK_AX)

        # Clip MSI extremes for visualisation
        display = np.where(arr > 5, np.nan, arr) if "msi" in title.lower() else arr

        im = ax.imshow(display, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cb.ax.tick_params(colors=C_TEXT, labelsize=7)

        # Drought stress contour on NDMI panel
        if i == 0:
            ax.contour(ndmi_arr, levels=[-0.1, 0.0], colors=[C_GOLD],
                       linewidths=1.2, linestyles=["--", "-"])
            # Stats overlay
            v = ndmi_arr[np.isfinite(ndmi_arr)]
            if v.size:
                stressed_pct = (v < -0.1).sum() / v.size * 100
                ax.text(0.02, 0.05, f"Severe stress: {stressed_pct:.1f}% of pixels",
                        transform=ax.transAxes, color=C_GOLD, fontsize=8,
                        bbox=dict(fc=DARK_BG, ec="none", alpha=0.8))

        ax.set_title(title, color=C_TEXT, fontsize=9, fontweight="bold", pad=6)
        ax.axis("off")

    out_png = os.path.join(OUT_DIR, "moisture_stress_indices.png")
    fig.savefig(out_png, dpi=180, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  [OK] 4-panel figure: {out_png}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print(" GEOCASCADE - MOISTURE & DROUGHT STRESS INDICES")
    print(" NDMI | MSI | NDWI | NMDI  |  Sentinel-2 B03/B08/B11/B12")
    print("=" * 65)

    print("\n[1/4] Fetching/loading band data...")
    try:
        green, nir, swir1, swir2, profile, shape_ = fetch_bands()
    except Exception as e:
        print(f"\n  ERROR: {e}")
        return

    print(f"  Grid: {shape_[1]} x {shape_[0]} pixels")

    print("\n[2/4] Computing indices...")
    ndmi_arr = ndmi(nir, swir1)
    msi_arr  = msi(swir1, nir)
    ndwi_arr = ndwi(green, nir)
    nmdi_arr = nmdi(nir, swir1, swir2)

    print("\n[3/4] Saving GeoTIFFs...")
    save_tif(ndmi_arr, "ndmi", profile, "NDMI: Canopy moisture (NIR-SWIR1)/(NIR+SWIR1)")
    save_tif(msi_arr,  "msi",  profile, "MSI: Moisture Stress Index SWIR1/NIR")
    save_tif(ndwi_arr, "ndwi", profile, "NDWI: Surface water (Green-NIR)/(Green+NIR)")
    save_tif(nmdi_arr, "nmdi", profile, "NMDI: Multi-band drought index (Wang & Qu 2007)")
    save_stats({"NDMI": ndmi_arr, "MSI": msi_arr, "NDWI": ndwi_arr, "NMDI": nmdi_arr})

    print("\n[4/4] Building figure...")
    plot_moisture(ndmi_arr, msi_arr, ndwi_arr, nmdi_arr)

    print("\n" + "=" * 65)
    print(" MOISTURE INDICES COMPLETE")
    print("=" * 65)
    print(f"  TIFs  : {OUT_DIR}")
    print(f"  Figure: {os.path.join(OUT_DIR, 'moisture_stress_indices.png')}")
    print(f"  Stats : {os.path.join(OUT_DIR, 'moisture_statistics.csv')}")
    print()
    print("  ArcGIS Pro: Add ndmi.tif, Stretched > Blue-Red diverging.")
    print("              Raster Calculator: Con(ndmi < -0.1, 1, 0) = severe drought.")
    print("  ENVI 5.6  : Band Math: (b1-b2)/(b1+b2) with B08, B11.")
    print("=" * 65)


if __name__ == "__main__":
    main()
