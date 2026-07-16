"""
Chapter 7: 19_multisensor_review.py
=====================================
Multi-Sensor Comparative Review: Landsat 9 / MODIS LST / Sentinel-1 SAR

Academic Objective:
  No single sensor sees the complete picture. Each spectral domain reveals
  a different physical property of the surface:

  OPTICAL (Landsat 9, 30m):
    Passive sensor — reflects solar radiation.
    Advantages: high spatial resolution, intuitive RGB, many indices possible.
    Critical limitation: COMPLETELY BLIND under clouds.
    NIR band (B5, 865nm) reflects strongly from green vegetation.

  THERMAL (MODIS LST, 1km):
    Passive sensor — detects emitted thermal radiation (8-12 micron).
    Measures Land Surface Temperature (LST), not air temperature.
    Scale factor: DN * 0.02 → Kelvin. Subtract 273.15 for Celsius.
    Landsat C2-L2 LST scale: DN * 0.00341802 + 149.0 → Kelvin
    Fill value: 0 (DO NOT use DN < 7500; use DN == 0 for MODIS fill)

  MICROWAVE/SAR (Sentinel-1, 10m):
    Active sensor — transmits microwave pulse and measures backscatter.
    Completely cloud-independent. Works at night.
    Reveals surface ROUGHNESS and DIELECTRIC properties, not color.

  By comparing the SAME geographic area across all three sensors simultaneously,
  we learn which questions each sensor can and cannot answer.

Study area: Punta Arenas region (city + coast + mixed land cover)
  — ideal because it contains urban (high thermal), ocean (cold, specular),
  and forested land cover (moderate optical, high SAR VH).

Landsat C2-L2 scale factors (CRITICAL — applied every time):
  Surface Reflectance: DN * 0.0000275 - 0.2   (Band 1-7)
  LST:                 DN * 0.00341802 + 149.0  (Band 10, in Kelvin)
  Fill value:          0 → mask before applying scale factor

MODIS LST scale: DN * 0.02 → Kelvin. Fill = 0.

Outputs:
  data/processed/multisensor/landsat9_nir.tif
  data/processed/multisensor/modis_lst_celsius.tif
  data/processed/multisensor/sentinel1_vv_db.tif
  data/processed/multisensor/multisensor_statistics.csv
  data/processed/multisensor/multisensor_comparison.png  (4-panel dark)

ArcGIS Pro: Add all 3 TIFs. Use Split View to compare sensors side by side.
ENVI 5.6:   File > Open multiple TIFs. Use Geographic Link for pan navigation.

Run:
  conda activate geocascade_env
  python Chapter_07/19_multisensor_review.py

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

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(BASE_DIR, "data", "processed", "multisensor")
os.makedirs(OUT_DIR, exist_ok=True)

# Punta Arenas: city + coast + forest — good multi-cover test area
BBOX       = [-71.05, -53.20, -70.80, -53.10]
DATE_RANGE = "2023-01-01/2023-03-31"

DARK_BG = "#0d1117"
DARK_AX = "#161b22"
C_TEXT  = "#e6edf3"
C_GREY  = "#8b949e"


# ---------------------------------------------------------------------------
# 1. Read any band into float32 array with correct windowing
# ---------------------------------------------------------------------------
def _read_bbox_band(href, bbox):
    """Generic window-read for any STAC asset over a geographic BBOX."""
    from pyproj import Transformer
    with rasterio.open(href) as src:
        t = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        mnx, mny = t.transform(bbox[0], bbox[1])
        mxx, mxy = t.transform(bbox[2], bbox[3])
        win  = from_bounds(mnx, mny, mxx, mxy, src.transform)
        h, w = int(round(win.height)), int(round(win.width))
        arr  = src.read(1, window=win).astype("float32")
        prof = src.profile.copy()
        prof.update(dtype="float32", count=1, nodata=-9999, compress="lzw",
                    height=h, width=w,
                    transform=rasterio.windows.transform(win, src.transform))
    return arr, prof


# ---------------------------------------------------------------------------
# 2. Fetch Landsat 9 NIR (Band 5 / nir08)
# ---------------------------------------------------------------------------
def fetch_landsat9(catalog):
    print("  [1/3] Fetching Landsat 9 NIR (optical 30m)...")
    search = catalog.search(
        collections=["landsat-c2-l2"], bbox=BBOX, datetime=DATE_RANGE,
        query={"platform": {"in": ["landsat-9"]}, "eo:cloud_cover": {"lt": 40}}
    )
    items = list(search.items())
    if not items:
        # Relax cloud filter
        search = catalog.search(
            collections=["landsat-c2-l2"], bbox=BBOX, datetime="2023-01-01/2023-06-30",
            query={"platform": {"in": ["landsat-9"]}, "eo:cloud_cover": {"lt": 70}}
        )
        items = list(search.items())

    if not items:
        print("  [WARN] No Landsat 9 found — returning None.")
        return None, None, "Landsat 9 NIR B5 (no data)"

    item   = sorted(items, key=lambda i: i.properties.get("eo:cloud_cover", 99))[0]
    cloud  = item.properties.get("eo:cloud_cover", "?")
    print(f"  [OK]  L9 date={str(item.datetime.date())} cloud={cloud}%")

    arr, prof = _read_bbox_band(item.assets["nir08"].href, BBOX)

    # CRITICAL: Landsat C2-L2 scale factor for surface reflectance
    # Raw DN 0 = fill value (cloud/no data). Apply mask BEFORE scaling.
    fill_mask = arr == 0
    arr = arr * 0.0000275 - 0.2
    arr = np.where(fill_mask, np.nan, arr)
    arr = np.clip(arr, 0.0, 1.0)   # valid reflectance range

    return arr, prof, f"Landsat 9 NIR B5 (30m, SR)\nCloud={cloud}%"


# ---------------------------------------------------------------------------
# 3. Fetch MODIS LST
# ---------------------------------------------------------------------------
def fetch_modis_lst(catalog):
    print("  [2/3] Fetching MODIS LST (thermal 1km)...")
    search = catalog.search(collections=["modis-11A1-061"], bbox=BBOX,
                            datetime=DATE_RANGE)
    items  = list(search.items())
    if not items:
        print("  [WARN] No MODIS LST found — returning None.")
        return None, None, "MODIS LST 1km (no data)"

    item     = items[-1]
    arr, prof = _read_bbox_band(item.assets["LST_Day_1km"].href, BBOX)

    # MODIS fill value = 0. NEVER threshold on DN < 7500.
    fill_mask = arr == 0
    # MODIS LST scale factor: DN * 0.02 = Kelvin
    lst_k = arr * 0.02
    lst_c = lst_k - 273.15
    lst_c = np.where(fill_mask | (lst_k < 150), np.nan, lst_c)  # < 150K = invalid

    print(f"  [OK]  MODIS date={str(item.datetime.date())}  "
          f"T range: {np.nanmin(lst_c):.1f} to {np.nanmax(lst_c):.1f} C")
    return lst_c, prof, "MODIS LST Day (1km)\nCelsius"


# ---------------------------------------------------------------------------
# 4. Fetch Sentinel-1 VV
# ---------------------------------------------------------------------------
def fetch_sentinel1_vv(catalog):
    print("  [3/3] Fetching Sentinel-1 VV SAR (radar 10m)...")
    search = catalog.search(collections=["sentinel-1-rtc"], bbox=BBOX,
                            datetime=DATE_RANGE)
    items  = list(search.items())
    if not items:
        print("  [WARN] No Sentinel-1 found — returning None.")
        return None, None, "Sentinel-1 VV dB (no data)"

    item     = items[0]
    arr, prof = _read_bbox_band(item.assets["vv"].href, BBOX)

    # Convert linear amplitude to dB
    vv_db = 10.0 * np.log10(np.where(arr > 0, arr, np.nan))

    print(f"  [OK]  SAR date={str(item.datetime.date())}  "
          f"VV range: {np.nanmin(vv_db):.1f} to {np.nanmax(vv_db):.1f} dB")
    return vv_db, prof, "Sentinel-1 VV (10m)\nBackscatter dB"


# ---------------------------------------------------------------------------
# 5. Save TIF
# ---------------------------------------------------------------------------
def save_tif(arr, name, profile, description=""):
    if arr is None:
        return
    out = os.path.join(OUT_DIR, f"{name}.tif")
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(np.nan_to_num(arr.astype("float32"), nan=-9999), 1)
        if description:
            dst.update_tags(description=description, nodata="-9999")
    print(f"  [OK] {name}.tif")


# ---------------------------------------------------------------------------
# 6. Statistics CSV
# ---------------------------------------------------------------------------
def save_stats(sensors):
    rows = []
    for name, arr in sensors.items():
        if arr is None:
            continue
        v = arr[np.isfinite(arr)]
        rows.append({
            "sensor": name,
            "mean": round(float(v.mean()), 4),
            "std":  round(float(v.std()),  4),
            "min":  round(float(v.min()),  4),
            "max":  round(float(v.max()),  4),
            "pct_valid": round(float(v.size / arr.size * 100), 1),
        })
    df  = pd.DataFrame(rows)
    csv = os.path.join(OUT_DIR, "multisensor_statistics.csv")
    df.to_csv(csv, index=False, encoding="utf-8")
    print(f"\n  --- Multi-Sensor Statistics ---")
    for _, r in df.iterrows():
        print(f"  {r['sensor']:<20s}  mean={r['mean']:>8.3f}  "
              f"std={r['std']:>7.3f}  valid={r['pct_valid']:.0f}%")
    print(f"  [OK] multisensor_statistics.csv")


# ---------------------------------------------------------------------------
# 7. 4-panel dark figure
# ---------------------------------------------------------------------------
def plot_comparison(l9, l9_title, modis, modis_title, s1, s1_title):
    print("\n  Building 4-panel multisensor figure...")
    fig = plt.figure(figsize=(22, 8), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.22,
                            top=0.88, bottom=0.05, left=0.04, right=0.97)
    fig.text(0.5, 0.95, "Multi-Sensor Review: Landsat 9 | MODIS LST | Sentinel-1 SAR",
             ha="center", color=C_TEXT, fontsize=13, fontweight="bold")

    panels = [
        (l9,    "RdYlGn",  0.0,  0.6, "Reflectance", l9_title),
        (modis, "inferno", None, None, "deg C",       modis_title),
        (s1,    "gray",   -25,   0,   "dB",           s1_title),
    ]

    for i, (arr, cmap, vmin, vmax, cbar_label, title) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor(DARK_AX)
        ax.axis("off")

        if arr is not None and np.any(np.isfinite(arr)):
            v = arr[np.isfinite(arr)]
            _vmin = vmin if vmin is not None else float(np.percentile(v, 2))
            _vmax = vmax if vmax is not None else float(np.percentile(v, 98))
            im = ax.imshow(arr, cmap=cmap, vmin=_vmin, vmax=_vmax, aspect="auto")
            cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
            cb.set_label(cbar_label, color=C_TEXT, fontsize=7)
            cb.ax.tick_params(colors=C_TEXT, labelsize=6)
        else:
            ax.text(0.5, 0.5, "No data\navailable",
                    ha="center", va="center", color=C_GREY, fontsize=12,
                    transform=ax.transAxes)

        ax.set_title(title, color=C_TEXT, fontsize=8.5, fontweight="bold", pad=5)

    # Panel 4: sensor capability comparison table
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.set_facecolor(DARK_AX)
    ax4.axis("off")
    ax4.set_title("Sensor Capability Matrix",
                  color=C_TEXT, fontsize=8.5, fontweight="bold", pad=5)

    table_data = [
        ["Property",       "L9 Optical", "MODIS Thermal", "SAR"],
        ["Cloud-free",     "No",          "No",            "YES"],
        ["Night capable",  "No",          "Yes",           "Yes"],
        ["Spatial res.",   "30m",         "1km",           "10m"],
        ["Measures",       "Reflectance", "Temperature",   "Roughness"],
        ["Vegetation",     "High NDVI",   "Warm LST",      "High VH"],
        ["Open water",     "Low NIR",     "Cold LST",      "Very low VV"],
        ["Ice/Glacier",    "High NDSI",   "Very cold",     "Moderate VV"],
    ]

    y = 0.95
    for row_i, row in enumerate(table_data):
        x = 0.02
        for col_i, cell in enumerate(row):
            color = C_TEXT if row_i > 0 else C_GREY
            weight = "bold" if row_i == 0 or col_i == 0 else "normal"
            size = 7.5
            if cell == "YES":
                color = "#27ae60"
                weight = "bold"
            elif cell == "No":
                color = "#e74c3c"
            ax4.text(x, y, cell, transform=ax4.transAxes,
                     color=color, fontsize=size, fontweight=weight, va="top")
            x += 0.28 if col_i == 0 else 0.24
        y -= 0.09
        if row_i == 0:
            ax4.axhline(y + 0.005, color="#30363d", lw=0.8, transform=ax4.transAxes)

    out_png = os.path.join(OUT_DIR, "multisensor_comparison.png")
    fig.savefig(out_png, dpi=180, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  [OK] 4-panel figure: {out_png}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print(" GEOCASCADE - MULTI-SENSOR COMPARATIVE REVIEW")
    print(" Landsat 9 (Optical) | MODIS (Thermal) | Sentinel-1 (SAR)")
    print(f" BBOX: {BBOX}  |  Punta Arenas region")
    print("=" * 65)

    try:
        from pystac_client import Client
        import planetary_computer as pc
    except ImportError:
        raise ImportError("pystac-client / planetary-computer not installed.")

    print("\n[1/4] Connecting to Planetary Computer...")
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                          modifier=pc.sign_inplace)

    print("\n[2/4] Fetching all sensors...")
    l9_arr,    l9_prof,    l9_title    = fetch_landsat9(catalog)
    modis_arr, modis_prof, modis_title = fetch_modis_lst(catalog)
    s1_arr,    s1_prof,    s1_title    = fetch_sentinel1_vv(catalog)

    print("\n[3/4] Saving GeoTIFFs and statistics...")
    if l9_arr    is not None: save_tif(l9_arr,    "landsat9_nir",       l9_prof,    l9_title)
    if modis_arr is not None: save_tif(modis_arr, "modis_lst_celsius",  modis_prof, modis_title)
    if s1_arr    is not None: save_tif(s1_arr,    "sentinel1_vv_db",    s1_prof,    s1_title)
    save_stats({
        "Landsat9_NIR_SR":   l9_arr,
        "MODIS_LST_C":       modis_arr,
        "S1_VV_dB":          s1_arr,
    })

    print("\n[4/4] Building comparison figure...")
    plot_comparison(l9_arr, l9_title, modis_arr, modis_title, s1_arr, s1_title)

    print("\n" + "=" * 65)
    print(" MULTI-SENSOR REVIEW COMPLETE")
    print("=" * 65)
    print(f"  Outputs : {OUT_DIR}")
    print(f"  Figure  : {os.path.join(OUT_DIR, 'multisensor_comparison.png')}")
    print()
    print("  ArcGIS Pro: Add all 3 TIFs. Use Split View to compare.")
    print("  ENVI 5.6  : Open each TIF separately. Use Geographic Link.")
    print("=" * 65)


if __name__ == "__main__":
    main()
