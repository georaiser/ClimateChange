"""
Chapter 3: 09_multitemporal_glacier_retreat.py
================================================
Multi-Temporal Glacier Retreat Analysis (2003 vs 2023)

Academic Objective:
  Climate change impact on glaciers is the most visually dramatic signal in
  Patagonia. This script performs a 20-year change detection on Grey Glacier
  using Landsat imagery: 2003 vs 2023.

  Method: NDSI (Normalized Difference Snow Index) thresholding
    NDSI = (Green - SWIR1) / (Green + SWIR1)
    NDSI > 0.4 = snow / ice
    Retreat map = ice_2003 - ice_2023
      +1 = ice melted (red)    0 = stable    -1 = ice advanced (blue)

Key technical notes:
  - Landsat Collection 2 Level-2 scale factor: SR = DN * 0.0000275 - 0.2
    Without this, raw DNs (~7000-20000) produce NDSI near 1.0 for ALL pixels.
  - Cloud cover threshold: 40% (Patagonia is frequently cloudy)
  - 2003 scene resampled to 2023 master grid (bilinear) for pixel-exact subtraction
  - nodata=-9999 for all GeoTIFFs (ArcGIS/ENVI compatible)

Connection to Chapter 1:
  This script re-uses the study BBOX from Chapter 1 and complements the NDSI
  glacier mask produced by Chapter_02/07_vegetation_soil_indices.py.

Outputs:
  data/processed/glacier_retreat/glacier_ice_2003.tif
  data/processed/glacier_retreat/glacier_ice_2023.tif
  data/processed/glacier_retreat/glacier_retreat_2003_2023.tif
  data/processed/glacier_retreat/glacier_retreat_report.csv
  data/processed/glacier_retreat/glacier_retreat_2003_2023.png  (4-panel dark)

ArcGIS Pro: Add retreat TIF. Symbology > Unique Values:
            -9999=NoData, -1=Advanced (blue), 0=Stable (grey), 1=Retreated (red).
ENVI 5.6:   File > Open > glacier_retreat_2003_2023.tif
            Tools > Color Map to assign custom colors to -1, 0, 1.

Run:
  conda activate geocascade_env
  python Chapter_03/09_multitemporal_glacier_retreat.py

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
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(BASE_DIR, "data", "processed", "glacier_retreat")
os.makedirs(OUT_DIR, exist_ok=True)

# Tight BBOX around Grey Glacier front
BBOX = [-73.30, -51.10, -73.15, -50.90]

# Landsat Collection 2 Level-2 scale factor
L2_SCALE  = 0.0000275
L2_OFFSET = -0.2

DARK_BG = "#0d1117"
DARK_AX = "#161b22"
C_TEXT  = "#e6edf3"
C_GREY  = "#8b949e"
C_RED   = "#e74c3c"
C_BLUE  = "#3498db"
C_CYAN  = "#00bcd4"


# ---------------------------------------------------------------------------
# 1. STAC search for best Landsat scene in a year range
# ---------------------------------------------------------------------------
def fetch_landsat(catalog, year_start, year_end, cloud_thresh=40):
    from pystac_client import Client
    print(f"  Searching Landsat {year_start}-{year_end} (cloud < {cloud_thresh}%)...")
    search = catalog.search(
        collections=["landsat-c2-l2"],
        bbox=BBOX,
        datetime=f"{year_start}-01-01/{year_end}-12-31",
        query={"eo:cloud_cover": {"lt": cloud_thresh}}
    )
    items = list(search.items())
    if not items:
        raise ValueError(f"No Landsat scene found for {year_start}-{year_end}. "
                         f"Try increasing cloud_thresh (currently {cloud_thresh}%).")
    best = sorted(items, key=lambda x: x.properties.get("eo:cloud_cover", 99))[0]
    date = best.properties.get("datetime", "")[:10]
    cloud = best.properties.get("eo:cloud_cover", 0)
    print(f"  [OK] {best.id}  date={date}  cloud={cloud:.1f}%")
    return best


# ---------------------------------------------------------------------------
# 2. Read one green + SWIR band pair, apply L2 scale factor
# ---------------------------------------------------------------------------
def read_l2_bands(item, master_shape=None, master_transform=None, master_crs=None):
    """
    Read green + swir16 bands for one Landsat L2 item.
    If master_shape is given, resample output to that shape (for co-registration).
    Returns: green, swir1, profile, pixel_area_km2
    """
    from pyproj import Transformer

    green_asset = item.assets.get("green") or item.assets.get("SR_B3")
    swir_asset  = item.assets.get("swir16") or item.assets.get("SR_B6")
    if not green_asset or not swir_asset:
        raise KeyError(f"Could not find green/swir16 assets in item {item.id}. "
                       f"Available: {list(item.assets.keys())}")

    with rasterio.open(green_asset.href) as src:
        t = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        mnx, mny = t.transform(BBOX[0], BBOX[1])
        mxx, mxy = t.transform(BBOX[2], BBOX[3])
        win      = from_bounds(mnx, mny, mxx, mxy, src.transform)
        shape    = master_shape or (int(round(win.height)), int(round(win.width)))
        win_tf   = rasterio.windows.transform(win, src.transform)

        profile  = src.profile.copy()
        profile.update(
            dtype="float32", count=1, nodata=-9999,
            height=shape[0], width=shape[1],
            transform=win_tf, compress="lzw"
        )

        # Pixel size for area calculation (approximate at 51 deg S)
        pix_m    = abs(src.res[0]) * 111_000.0 * np.cos(np.radians(-51.0))
        area_km2 = (pix_m / 1000.0) ** 2

        green = src.read(1, window=win, out_shape=shape,
                         resampling=Resampling.bilinear).astype("float32")
        green = np.clip(green * L2_SCALE + L2_OFFSET, 0.0, 1.0)

    with rasterio.open(swir_asset.href) as src:
        t = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        mnx, mny = t.transform(BBOX[0], BBOX[1])
        mxx, mxy = t.transform(BBOX[2], BBOX[3])
        win      = from_bounds(mnx, mny, mxx, mxy, src.transform)
        swir1 = src.read(1, window=win, out_shape=shape,
                         resampling=Resampling.bilinear).astype("float32")
        swir1 = np.clip(swir1 * L2_SCALE + L2_OFFSET, 0.0, 1.0)

    return green, swir1, profile, area_km2


# ---------------------------------------------------------------------------
# 3. NDSI and ice mask
# ---------------------------------------------------------------------------
def calc_ndsi(green, swir1):
    """NDSI = (Green - SWIR1) / (Green + SWIR1). NaN-safe."""
    denom = green + swir1
    return np.where(np.abs(denom) < 1e-6, np.nan, (green - swir1) / denom)

def ice_mask(ndsi, threshold=0.4):
    """Binary ice mask: 1 = ice (NDSI > threshold), 0 = no ice."""
    return np.where(np.isfinite(ndsi) & (ndsi > threshold), 1, 0).astype("int16")


# ---------------------------------------------------------------------------
# 4. Save GeoTIFF
# ---------------------------------------------------------------------------
def save_tif(data, path, profile, description=""):
    safe = np.where(np.isnan(data.astype("float32")), -9999, data).astype("float32")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(safe, 1)
        if description:
            dst.update_tags(description=description, nodata="-9999",
                            arcgis_note="Unique Values symbology: -1=advanced, 0=stable, 1=retreated",
                            envi_note="File > Open, Tools > Color Map for categorical display")
    print(f"  [OK] {os.path.basename(path)}")


# ---------------------------------------------------------------------------
# 5. 4-panel dark figure
# ---------------------------------------------------------------------------
def plot_retreat(ndsi_2003, ndsi_2023, ice_2003, ice_2023, retreat_map):
    print("\n  Building 4-panel retreat figure...")

    fig, axes = plt.subplots(2, 2, figsize=(18, 16), facecolor=DARK_BG)
    fig.suptitle("Grey Glacier Retreat Analysis: 2003 vs 2023\nTorres del Paine, Patagonia",
                 color=C_TEXT, fontsize=14, fontweight="bold", y=0.98)

    def style_ax(ax, title):
        ax.set_facecolor(DARK_AX)
        ax.axis("off")
        ax.set_title(title, color=C_TEXT, fontsize=10, fontweight="bold", pad=6)

    # Panel 1: NDSI 2003
    ax = axes[0, 0]
    im1 = ax.imshow(ndsi_2003, cmap="cool", vmin=-0.5, vmax=1.0, aspect="auto")
    cb1 = plt.colorbar(im1, ax=ax, fraction=0.035, pad=0.02)
    cb1.ax.tick_params(colors=C_TEXT, labelsize=7)
    cb1.set_label("NDSI", color=C_TEXT, fontsize=8)
    ax.contour(ice_2003, levels=[0.5], colors=[C_CYAN], linewidths=1.5)
    style_ax(ax, "NDSI 2003 -- Ice extent contour (cyan)")

    # Panel 2: NDSI 2023
    ax = axes[0, 1]
    im2 = ax.imshow(ndsi_2023, cmap="cool", vmin=-0.5, vmax=1.0, aspect="auto")
    cb2 = plt.colorbar(im2, ax=ax, fraction=0.035, pad=0.02)
    cb2.ax.tick_params(colors=C_TEXT, labelsize=7)
    cb2.set_label("NDSI", color=C_TEXT, fontsize=8)
    ax.contour(ice_2023, levels=[0.5], colors=[C_RED], linewidths=1.5)
    style_ax(ax, "NDSI 2023 -- Ice extent contour (red)")

    # Panel 3: Ice masks overlay
    ax = axes[1, 0]
    ax.set_facecolor(DARK_AX)
    ax.axis("off")
    ice_only_2003 = np.ma.masked_where(ice_2003 == 0, ice_2003)
    ice_only_2023 = np.ma.masked_where(ice_2023 == 0, ice_2023)
    ax.imshow(np.zeros_like(ice_2003), cmap="gray", vmin=0, vmax=1, aspect="auto", alpha=0.3)
    ax.imshow(ice_only_2003, cmap="Blues", vmin=0, vmax=1, aspect="auto", alpha=0.6)
    ax.imshow(ice_only_2023, cmap="Reds", vmin=0, vmax=1, aspect="auto", alpha=0.6)
    legend_els = [
        Patch(facecolor="#3498db", alpha=0.7, label="Ice 2003"),
        Patch(facecolor="#e74c3c", alpha=0.7, label="Ice 2023"),
    ]
    ax.legend(handles=legend_els, loc="lower right", fontsize=9,
              facecolor=DARK_BG, labelcolor=C_TEXT)
    ax.set_title("Ice Extent Overlay (Blue=2003, Red=2023)",
                 color=C_TEXT, fontsize=10, fontweight="bold", pad=6)

    # Panel 4: Retreat map
    ax = axes[1, 1]
    cmap_ret = ListedColormap(["#00bcd4", "#2d333b", "#e74c3c"])
    im4 = ax.imshow(retreat_map, cmap=cmap_ret, vmin=-1, vmax=1, aspect="auto")
    legend_ret = [
        Patch(facecolor="#e74c3c", label="Ice Retreated (Melted)"),
        Patch(facecolor="#2d333b", label="Stable / No Ice"),
        Patch(facecolor="#00bcd4", label="Ice Advanced"),
    ]
    ax.legend(handles=legend_ret, loc="lower center",
              bbox_to_anchor=(0.5, -0.12), ncol=3,
              fontsize=8, facecolor=DARK_BG, labelcolor=C_TEXT)
    style_ax(ax, "Glacier Change Map: Red=Retreated, Cyan=Advanced")

    out_png = os.path.join(OUT_DIR, "glacier_retreat_2003_2023.png")
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  [OK] 4-panel figure: {out_png}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print(" GEOCASCADE - GLACIER RETREAT ANALYSIS 2003 vs 2023")
    print(f" Grey Glacier, Torres del Paine | NDSI threshold: 0.4")
    print("=" * 65)

    try:
        from pystac_client import Client
        import planetary_computer as pc
    except ImportError:
        print("\n  ERROR: pystac-client / planetary-computer not installed.")
        print("  Run: mamba install -n geocascade_env -c conda-forge pystac-client planetary-computer -y")
        return

    print("\n[1/5] Connecting to Planetary Computer...")
    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace
    )

    print("\n[2/5] Fetching Landsat scenes...")
    try:
        item_2003 = fetch_landsat(catalog, 2000, 2005, cloud_thresh=40)
        item_2023 = fetch_landsat(catalog, 2019, 2023, cloud_thresh=40)
    except ValueError as e:
        print(f"\n  ERROR: {e}")
        return

    print("\n[3/5] Reading band data with L2 scale factor (DN * 0.0000275 - 0.2)...")
    # 2023 defines the master grid
    green_2023, swir_2023, profile, pix_km2 = read_l2_bands(item_2023)
    master_shape = green_2023.shape
    print(f"  Master grid: {master_shape[1]} x {master_shape[0]} pixels | {pix_km2:.4f} km2/pixel")

    # 2003 resampled to 2023 master shape
    green_2003, swir_2003, _, _ = read_l2_bands(item_2023, master_shape=master_shape)
    # Re-read 2003 using 2003 item but forcing to master shape
    green_2003, swir_2003, _, _ = read_l2_bands(item_2003, master_shape=master_shape)

    print("\n[4/5] Computing NDSI + retreat map...")
    ndsi_2003 = calc_ndsi(green_2003, swir_2003)
    ndsi_2023 = calc_ndsi(green_2023, swir_2023)
    ice_2003  = ice_mask(ndsi_2003)
    ice_2023  = ice_mask(ndsi_2023)

    # Retreat map: +1 melted, 0 stable, -1 advanced
    retreat = (ice_2003 - ice_2023).astype("float32")
    # Mask areas where both epochs are nodata
    nodata_mask = (~np.isfinite(ndsi_2003)) & (~np.isfinite(ndsi_2023))
    retreat = np.where(nodata_mask, np.nan, retreat)

    # Stats
    lost_px    = int(np.sum(retreat == 1))
    gained_px  = int(np.sum(retreat == -1))
    stable_px  = int(np.sum(retreat == 0))
    lost_km2   = lost_px   * pix_km2
    gained_km2 = gained_px * pix_km2
    net_km2    = gained_km2 - lost_km2
    direction  = "ADVANCE" if net_km2 > 0 else "RETREAT"

    print("\n  --- GLACIER CHANGE REPORT (2003 - 2023) ---")
    print(f"  Ice lost (melted) : {lost_km2:7.2f} km2  ({lost_px:,} pixels)")
    print(f"  Ice stable        : {stable_px * pix_km2:7.2f} km2  ({stable_px:,} pixels)")
    print(f"  Ice gained        : {gained_km2:7.2f} km2  ({gained_px:,} pixels)")
    print(f"  Net change        : {abs(net_km2):7.2f} km2 ({direction})")
    print(f"  Pixel area        : {pix_km2:.4f} km2")

    # Save GeoTIFFs
    save_tif(ndsi_2003.astype("float32"), os.path.join(OUT_DIR, "glacier_ndsi_2003.tif"),
             profile, "NDSI 2003 -- Grey Glacier")
    save_tif(ndsi_2023.astype("float32"), os.path.join(OUT_DIR, "glacier_ndsi_2023.tif"),
             profile, "NDSI 2023 -- Grey Glacier")
    save_tif(retreat, os.path.join(OUT_DIR, "glacier_retreat_2003_2023.tif"),
             profile, "Glacier retreat: +1=melted, 0=stable, -1=advanced")

    # Save CSV report
    report = pd.DataFrame([{
        "period": "2003-2023",
        "ice_lost_km2": round(lost_km2, 3),
        "ice_gained_km2": round(gained_km2, 3),
        "net_change_km2": round(net_km2, 3),
        "direction": direction,
        "ice_2003_km2": round((lost_px + stable_px) * pix_km2, 3),
        "ice_2023_km2": round((gained_px + stable_px) * pix_km2, 3),
        "pixel_area_km2": round(pix_km2, 6),
    }])
    csv_path = os.path.join(OUT_DIR, "glacier_retreat_report.csv")
    report.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"  [OK] Report CSV: {csv_path}")

    print("\n[5/5] Building 4-panel figure...")
    plot_retreat(ndsi_2003, ndsi_2023, ice_2003, ice_2023, retreat)

    print("\n" + "=" * 65)
    print(" GLACIER RETREAT ANALYSIS COMPLETE")
    print("=" * 65)
    print(f"  TIFs   : {OUT_DIR}")
    print(f"  Report : {csv_path}")
    print(f"  Figure : {os.path.join(OUT_DIR, 'glacier_retreat_2003_2023.png')}")
    print()
    print("  ArcGIS Pro: Add retreat TIF. Symbology > Unique Values.")
    print("              Value -1=Advanced (blue), 0=Stable, 1=Retreated (red).")
    print("  ENVI 5.6  : File > Open, Tools > Color Map for categorical display.")
    print("=" * 65)


if __name__ == "__main__":
    main()
