"""
Chapter 8: 20_multisensor_data_fusion.py
==========================================
Multi-Sensor Data Fusion Engine -- 4-Band Analysis-Ready Data Cube

Academic Objective:
  The previous chapters collected data from isolated sensors:
  - Ch02: Sentinel-2 optical reflectance (10m)
  - Ch03: Copernicus DEM elevation (30m)
  - Ch07: Sentinel-1 SAR backscatter (10m)

  A Random Forest or CNN model cannot ingest arrays from sensors with
  different resolutions, projections, and coordinate systems simultaneously.
  Data Fusion solves this: all layers are REPROJECTED, RESAMPLED, and
  ALIGNED to a single master grid with consistent spatial resolution.

  Master Grid: Sentinel-2 10m UTM grid (established from Band B08, NIR)
  All other layers are warped (rasterio.warp.reproject) to match this grid.

  Resampling strategy per layer:
    SAR VV (10m → 10m):     bilinear (already same resolution, minor alignment)
    DEM (30m → 10m):        bilinear (upsampling, smooth interpolation)
    MODIS LST (1km → 10m):  bilinear (upsampling 100x, major spatial smoothing)

  Band descriptions in output stack:
    Band 1: S2 NIR (B08)    — Vegetation, land cover
    Band 2: S1 SAR VV dB    — Surface roughness, water, ice
    Band 3: CopDEM elevation — Topographic context
    Band 4: MODIS LST C     — Thermal energy, land surface temperature

  Local data priority: Ch02, Ch03, Ch07 cached outputs are used if available,
  avoiding redundant Planetary Computer downloads.

  Output:
    data/processed/fusion/cascade_master_stack.tif  (4-band, float32, nodata=-9999)
    data/processed/fusion/fusion_statistics.csv
    data/processed/fusion/data_cube_visualization.png  (4-panel dark)

ArcGIS Pro: Add cascade_master_stack.tif.
            Layer > Properties > Symbology > Composite > Assign bands.
            Band 1 = Red, Band 2 = Green (creates 2-band false color).
            Use "Individual bands" mode to inspect each layer separately.
ENVI 5.6:   File > Open > cascade_master_stack.tif as Multi-band file.
            Tools > Band Math for custom combinations.

Run:
  conda activate geocascade_env
  python Chapter_08/20_multisensor_data_fusion.py

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
from rasterio.warp import reproject, Resampling

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
OUT_DIR  = os.path.join(BASE_DIR, "data", "processed", "fusion")
TMP_DIR  = os.path.join(BASE_DIR, "data", "tmp")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

BBOX       = [-73.30, -51.10, -72.90, -50.80]
DATE_RANGE = "2023-01-01/2023-03-31"

# Paths to cached outputs from earlier chapters
CACHE = {
    "nir":  os.path.join(ROOT_DIR, "Chapter_02", "data", "processed", "indices", "ndvi.tif"),
    "sar":  os.path.join(ROOT_DIR, "Chapter_07", "data", "processed", "sar", "sar_vv_db.tif"),
    "dem":  os.path.join(ROOT_DIR, "Chapter_03", "data", "processed", "terrain", "copernicus_dem.tif"),
}

DARK_BG = "#0d1117"
DARK_AX = "#161b22"
C_TEXT  = "#e6edf3"


# ---------------------------------------------------------------------------
# 1. Establish master 10m grid from Sentinel-2 or local cache
# ---------------------------------------------------------------------------
def get_master_grid():
    """
    Returns (nir_array, master_profile) establishing the reference grid.
    Priority: Ch02 local NIR → STAC Sentinel-2
    """
    # Try Ch02 local NDVI TIF (same spatial footprint as NIR)
    if os.path.exists(CACHE["nir"]):
        print(f"  [OK] Using Ch02 cached NIR grid: {os.path.basename(CACHE['nir'])}")
        with rasterio.open(CACHE["nir"]) as src:
            arr   = src.read(1).astype("float32")
            nd    = src.nodata
            if nd is not None:
                arr = np.where(arr == nd, np.nan, arr)
            prof  = src.profile.copy()
            prof.update(dtype="float32", count=4, nodata=-9999, compress="lzw")
        return arr, prof

    # STAC fallback
    print("  Downloading Sentinel-2 NIR from Planetary Computer...")
    from pystac_client import Client
    import planetary_computer as pc
    from pyproj import Transformer

    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                          modifier=pc.sign_inplace)
    search  = catalog.search(collections=["sentinel-2-l2a"], bbox=BBOX,
                             datetime=DATE_RANGE,
                             query={"eo:cloud_cover": {"lt": 15}})
    items   = list(search.items())
    if not items:
        search = catalog.search(collections=["sentinel-2-l2a"], bbox=BBOX,
                                datetime=DATE_RANGE,
                                query={"eo:cloud_cover": {"lt": 50}})
        items  = list(search.items())
    if not items:
        raise RuntimeError("No Sentinel-2 found.")

    item = sorted(items, key=lambda i: i.properties.get("eo:cloud_cover", 99))[0]
    print(f"  [OK] S2 scene: {item.id}")

    with rasterio.open(item.assets["B08"].href) as src:
        t = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        mnx, mny = t.transform(BBOX[0], BBOX[1])
        mxx, mxy = t.transform(BBOX[2], BBOX[3])
        win  = from_bounds(mnx, mny, mxx, mxy, src.transform)
        arr  = src.read(1, window=win).astype("float32")
        h, w = int(round(win.height)), int(round(win.width))
        prof = src.profile.copy()
        prof.update(dtype="float32", count=4, nodata=-9999, compress="lzw",
                    height=h, width=w,
                    transform=rasterio.windows.transform(win, src.transform))
    arr = np.clip(arr / 10000.0, 0, 1)
    return arr, prof


# ---------------------------------------------------------------------------
# 2. Generic resampler: warp any raster to master profile
# ---------------------------------------------------------------------------
def resample_to_master(src_path, master_profile, process_fn=None, label=""):
    """
    Warp src_path raster to match master_profile CRS, transform, and size.
    Optionally apply process_fn(arr) for unit conversion.
    """
    h = int(master_profile["height"])
    w = int(master_profile["width"])
    dest = np.zeros((h, w), dtype=np.float32)

    with rasterio.open(src_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dest,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=master_profile["transform"],
            dst_crs=master_profile["crs"],
            resampling=Resampling.bilinear,
        )

    nd = -9999.0
    dest = np.where(dest == 0, np.nan, dest)   # common fill from reproject

    if process_fn:
        dest = process_fn(dest)

    print(f"  [OK] {label}: resampled to {w}x{h}")
    return dest


# ---------------------------------------------------------------------------
# 3. Per-layer fetch + process functions
# ---------------------------------------------------------------------------
def get_sar_layer(master_profile):
    if os.path.exists(CACHE["sar"]):
        print(f"  Using Ch07 cached SAR VV dB: {os.path.basename(CACHE['sar'])}")
        return resample_to_master(CACHE["sar"], master_profile, label="SAR VV dB")

    print("  Downloading Sentinel-1 SAR from Planetary Computer...")
    from pystac_client import Client
    import planetary_computer as pc
    from pyproj import Transformer

    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                          modifier=pc.sign_inplace)
    search  = catalog.search(collections=["sentinel-1-rtc"], bbox=BBOX,
                             datetime=DATE_RANGE)
    items   = list(search.items())
    if not items:
        print("  [WARN] No SAR found. Filling with NoData.")
        h, w = int(master_profile["height"]), int(master_profile["width"])
        return np.full((h, w), np.nan, dtype=np.float32)

    item    = items[0]
    tmp_sar = os.path.join(TMP_DIR, "tmp_sar.tif")

    # Save to temp TIF then resample
    with rasterio.open(item.assets["vv"].href) as src:
        from pyproj import Transformer as Tr
        t = Tr.from_crs("EPSG:4326", src.crs, always_xy=True)
        mnx, mny = t.transform(BBOX[0], BBOX[1])
        mxx, mxy = t.transform(BBOX[2], BBOX[3])
        win  = from_bounds(mnx, mny, mxx, mxy, src.transform)
        arr  = src.read(1, window=win).astype("float32")
        h, w = int(round(win.height)), int(round(win.width))
        prof = src.profile.copy()
        prof.update(dtype="float32", count=1, nodata=-9999, height=h, width=w,
                    transform=rasterio.windows.transform(win, src.transform))
        with rasterio.open(tmp_sar, "w", **prof) as dst:
            dst.write(np.nan_to_num(arr, nan=-9999), 1)

    def sar_to_db(a):
        return 10.0 * np.log10(np.where(a > 0, a, np.nan))

    return resample_to_master(tmp_sar, master_profile, process_fn=sar_to_db,
                              label="SAR VV dB (STAC)")


def get_dem_layer(master_profile):
    if os.path.exists(CACHE["dem"]):
        print(f"  Using Ch03 cached DEM: {os.path.basename(CACHE['dem'])}")
        return resample_to_master(CACHE["dem"], master_profile, label="DEM elevation")

    print("  Downloading Copernicus DEM from Planetary Computer...")
    from pystac_client import Client
    import planetary_computer as pc
    from pyproj import Transformer

    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                          modifier=pc.sign_inplace)
    search  = catalog.search(collections=["cop-dem-glo-30"], bbox=BBOX)
    items   = list(search.items())
    if not items:
        h, w = int(master_profile["height"]), int(master_profile["width"])
        return np.full((h, w), np.nan, dtype=np.float32)

    item    = items[0]
    tmp_dem = os.path.join(TMP_DIR, "tmp_dem.tif")

    with rasterio.open(item.assets["data"].href) as src:
        from pyproj import Transformer as Tr
        t = Tr.from_crs("EPSG:4326", src.crs, always_xy=True)
        mnx, mny = t.transform(BBOX[0], BBOX[1])
        mxx, mxy = t.transform(BBOX[2], BBOX[3])
        win  = from_bounds(mnx, mny, mxx, mxy, src.transform)
        arr  = src.read(1, window=win).astype("float32")
        h, w = int(round(win.height)), int(round(win.width))
        prof = src.profile.copy()
        prof.update(dtype="float32", count=1, nodata=-9999, height=h, width=w,
                    transform=rasterio.windows.transform(win, src.transform))
        nd = src.nodata
        if nd is not None:
            arr = np.where(arr == nd, np.nan, arr)
        arr = np.where(arr < -500, np.nan, arr)
        with rasterio.open(tmp_dem, "w", **prof) as dst:
            dst.write(np.nan_to_num(arr, nan=-9999), 1)

    return resample_to_master(tmp_dem, master_profile, label="DEM (STAC)")


def get_modis_layer(master_profile):
    """
    MODIS LST from Planetary Computer.
    Fill value = 0; scale = DN * 0.02 → Kelvin; subtract 273.15 for Celsius.
    """
    print("  Fetching MODIS LST from Planetary Computer...")
    try:
        from pystac_client import Client
        import planetary_computer as pc
    except ImportError:
        h, w = int(master_profile["height"]), int(master_profile["width"])
        return np.full((h, w), np.nan, dtype=np.float32)

    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                          modifier=pc.sign_inplace)
    search  = catalog.search(collections=["modis-11A1-061"], bbox=BBOX,
                             datetime=DATE_RANGE)
    items   = list(search.items())
    if not items:
        print("  [WARN] No MODIS LST found. Filling thermal band with NoData.")
        h, w = int(master_profile["height"]), int(master_profile["width"])
        return np.full((h, w), np.nan, dtype=np.float32)

    modis_item  = items[-1]
    tmp_modis   = os.path.join(TMP_DIR, "tmp_modis.tif")

    import urllib.request
    urllib.request.urlretrieve(modis_item.assets["LST_Day_1km"].href, tmp_modis)

    def modis_to_celsius(a):
        a = np.where(a == 0, np.nan, a)           # fill value = 0
        lst_k = a * 0.02
        lst_c = lst_k - 273.15
        return np.where(lst_k < 150, np.nan, lst_c)  # < 150K = invalid

    return resample_to_master(tmp_modis, master_profile, process_fn=modis_to_celsius,
                              label="MODIS LST (Celsius)")


# ---------------------------------------------------------------------------
# 4. Statistics CSV
# ---------------------------------------------------------------------------
def save_stats(bands_dict):
    rows = []
    for name, arr in bands_dict.items():
        v = arr[np.isfinite(arr)]
        rows.append({
            "band":       name,
            "mean":       round(float(v.mean()), 4) if v.size else None,
            "std":        round(float(v.std()),  4) if v.size else None,
            "min":        round(float(v.min()),  4) if v.size else None,
            "max":        round(float(v.max()),  4) if v.size else None,
            "pct_valid":  round(float(v.size / arr.size * 100), 1),
        })
    df  = pd.DataFrame(rows)
    csv = os.path.join(OUT_DIR, "fusion_statistics.csv")
    df.to_csv(csv, index=False, encoding="utf-8")
    print(f"\n  --- Data Cube Band Statistics ---")
    for _, r in df.iterrows():
        print(f"  {r['band']:<20s}  mean={str(r['mean']):>8s}  "
              f"std={str(r['std']):>7s}  valid={r['pct_valid']:.0f}%")
    print(f"  [OK] fusion_statistics.csv")


# ---------------------------------------------------------------------------
# 5. 4-panel dark figure
# ---------------------------------------------------------------------------
def plot_fusion(nir, sar, dem, lst):
    print("\n  Building 4-panel data cube figure...")

    fig = plt.figure(figsize=(22, 8), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.2,
                            top=0.88, bottom=0.05, left=0.04, right=0.97)
    fig.text(0.5, 0.95,
             "Multi-Sensor Data Fusion Cube (4 Bands, 10m Grid) -- Torres del Paine",
             ha="center", color=C_TEXT, fontsize=13, fontweight="bold")

    panels = [
        (nir, "YlGn",    None, None, "Reflectance", "Band 1: S2 NIR B08\nVegetation / land cover"),
        (sar, "gray",    -25,   0,   "dB",          "Band 2: S1 SAR VV dB\nSurface roughness / water / ice"),
        (dem, "terrain", None, None, "Elevation m", "Band 3: Copernicus DEM\nTopographic context"),
        (lst, "inferno", None, None, "deg C",       "Band 4: MODIS LST\nLand surface temperature"),
    ]

    for i, (arr, cmap, vmin, vmax, cbar_lbl, title) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor(DARK_AX)
        ax.axis("off")
        ax.set_title(title, color=C_TEXT, fontsize=8.5, fontweight="bold", pad=5)

        if arr is not None and np.any(np.isfinite(arr)):
            v = arr[np.isfinite(arr)]
            _vmin = vmin if vmin is not None else float(np.percentile(v, 2))
            _vmax = vmax if vmax is not None else float(np.percentile(v, 98))
            im = ax.imshow(arr, cmap=cmap, vmin=_vmin, vmax=_vmax, aspect="auto")
            cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
            cb.set_label(cbar_lbl, color=C_TEXT, fontsize=7)
            cb.ax.tick_params(colors=C_TEXT, labelsize=6)
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    color="#8b949e", fontsize=12, transform=ax.transAxes)

    out_png = os.path.join(OUT_DIR, "data_cube_visualization.png")
    fig.savefig(out_png, dpi=180, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  [OK] 4-panel figure: {out_png}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print(" GEOCASCADE - MULTI-SENSOR DATA FUSION ENGINE")
    print(" S2 NIR | S1 SAR VV | CopDEM | MODIS LST")
    print(f" BBOX: {BBOX}  |  Master grid: 10m")
    print("=" * 65)

    print("\n[1/5] Establishing master 10m grid (Sentinel-2 NIR)...")
    try:
        nir, master_profile = get_master_grid()
    except Exception as e:
        print(f"\n  ERROR establishing master grid: {e}")
        return
    print(f"  Master grid: {master_profile['width']}x{master_profile['height']} pixels  "
          f"CRS={master_profile['crs']}")

    print("\n[2/5] Fetching/resampling SAR layer...")
    sar = get_sar_layer(master_profile)

    print("\n[3/5] Fetching/resampling DEM layer...")
    dem = get_dem_layer(master_profile)

    print("\n[4/5] Fetching/resampling MODIS LST layer...")
    lst = get_modis_layer(master_profile)

    print("\n[5/5] Stacking into 4-band data cube...")
    out_stack = os.path.join(OUT_DIR, "cascade_master_stack.tif")

    def _clean(a):
        return np.nan_to_num(a.astype("float32"), nan=-9999,
                             posinf=-9999, neginf=-9999)

    with rasterio.open(out_stack, "w", **master_profile) as dst:
        dst.set_band_description(1, "S2_NIR_B08")
        dst.set_band_description(2, "S1_SAR_VV_dB")
        dst.set_band_description(3, "CopDEM_Elevation_m")
        dst.set_band_description(4, "MODIS_LST_Celsius")
        dst.write(_clean(nir), 1)
        dst.write(_clean(sar), 2)
        dst.write(_clean(dem), 3)
        dst.write(_clean(lst), 4)

    print(f"  [OK] cascade_master_stack.tif (4 bands, {master_profile['width']}x{master_profile['height']} px)")

    save_stats({"NIR_refl": nir, "SAR_VV_dB": sar, "DEM_m": dem, "LST_Celsius": lst})
    plot_fusion(nir, sar, dem, lst)

    print("\n" + "=" * 65)
    print(" DATA FUSION COMPLETE")
    print("=" * 65)
    print(f"  Stack  : {out_stack}")
    print(f"  Figure : {os.path.join(OUT_DIR, 'data_cube_visualization.png')}")
    print(f"  Stats  : {os.path.join(OUT_DIR, 'fusion_statistics.csv')}")
    print()
    print("  Next step: Run 21_cascade_risk_modeling.py to apply")
    print("             Random Forest classification on this 4-band cube.")
    print()
    print("  ArcGIS Pro: Add cascade_master_stack.tif.")
    print("              Layer Properties > Symbology > Composite.")
    print("  ENVI 5.6  : File > Open as Multi-band. Tools > Band Math.")
    print("=" * 65)


if __name__ == "__main__":
    main()
