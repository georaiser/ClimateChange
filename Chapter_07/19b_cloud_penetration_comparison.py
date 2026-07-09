"""
Chapter 7 (Tier 4): 19b_cloud_penetration_comparison.py

Academic Objective:
One of the most compelling advantages of SAR (Synthetic Aperture Radar) over
optical sensors is its all-weather capability. Optical satellites (Sentinel-2,
Landsat) are completely blind under clouds. SAR penetrates clouds, rain, and
darkness to image the surface regardless of atmospheric conditions.

This script demonstrates the Cloud Penetration Advantage by deliberately
selecting a CLOUDY Sentinel-2 scene and the Sentinel-1 SAR acquisition
from the same week, then generating a side-by-side comparison chart that
shows what optical sensors CANNOT see vs what SAR reveals.

The VV/VH ratio panel additionally highlights:
  - Vegetation canopy structure (high VH cross-pol)
  - Bare soil / rough terrain (moderate VV)
  - Water bodies / ice (low VV, very low VH)

Outputs:
- sar_vs_optical_cloudy.png  (3-panel comparison)
- sar_vs_optical_stats.csv   (per-zone statistics)

Dependencies:
mamba install -n geocascade_env -c conda-forge pystac-client planetary-computer rasterio pyproj numpy matplotlib -y
"""

import os
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pystac_client import Client
import planetary_computer as pc
from pyproj import Transformer

# ==========================================
# 1. Configuration
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

BBOX       = [-73.30, -51.10, -72.90, -50.80]
# Deliberately pick winter month (June-July): high cloud probability in Patagonia
DATE_RANGE = "2023-06-01/2023-07-31"


# ==========================================
# 2. Fetch Sentinel-1 (SAR, cloud-independent)
# ==========================================
def fetch_sar(catalog, bbox):
    print("\n[1/3] Fetching Sentinel-1 SAR (all-weather, cloud-independent)...")
    search = catalog.search(collections=["sentinel-1-rtc"], bbox=bbox, datetime=DATE_RANGE)
    items  = list(search.items())
    if not items:
        raise RuntimeError("No Sentinel-1 data found for this window.")

    item = items[0]
    print(f"       SAR date: {item.datetime.date()}  Platform: {item.properties['platform']}")

    def _read_pol(pol):
        href = item.assets[pol].href
        with rasterio.open(href) as src:
            t = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            minx, miny = t.transform(bbox[0], bbox[1])
            maxx, maxy = t.transform(bbox[2], bbox[3])
            window = from_bounds(minx, miny, maxx, maxy, src.transform)
            arr = src.read(1, window=window).astype("float32")
            profile = src.profile
            profile.update(
                dtype=rasterio.float32, count=1, nodata=-9999,
                height=int(round(window.height)), width=int(round(window.width)),
                transform=rasterio.windows.transform(window, src.transform)
            )
        return arr, profile

    vv_lin, profile = _read_pol("vv")
    vh_lin = None
    if "vh" in item.assets:
        vh_lin, _ = _read_pol("vh")
        print("       VH polarization also acquired.")

    # Convert to dB
    vv_db = 10 * np.log10(np.where(vv_lin <= 0, np.nan, vv_lin))
    vh_db = (10 * np.log10(np.where(vh_lin <= 0, np.nan, vh_lin))
             if vh_lin is not None else None)
    ratio_db = (vv_db - vh_db) if (vh_db is not None) else None

    return vv_db, ratio_db, profile, item.datetime


# ==========================================
# 3. Fetch Sentinel-2 (Optical, cloud-affected)
# ==========================================
def fetch_optical(catalog, bbox, sar_profile):
    print("\n[2/3] Fetching Sentinel-2 (optical, potentially cloud-obscured)...")
    search = catalog.search(
        collections=["sentinel-2-l2a"], bbox=bbox, datetime=DATE_RANGE
        # NOTE: NO cloud filter - we intentionally want a cloudy scene to show limitation
    )
    items = list(search.items())
    if not items:
        print("       [WARNING] No Sentinel-2 data found. Returning blank optical panel.")
        h = int(sar_profile["height"])
        w = int(sar_profile["width"])
        return np.full((h, w), np.nan), None, 100.0

    # Pick the CLOUDIEST scene to maximize the demonstration
    item = sorted(items, key=lambda i: i.properties.get("eo:cloud_cover", 0), reverse=True)[0]
    cloud_pct = item.properties.get("eo:cloud_cover", "?")
    print(f"       Optical date: {item.datetime.date()}  Cloud cover: {cloud_pct}%  "
          f"(deliberately choosing cloudy scene)")

    # Resample RGB to SAR grid
    dest_r = np.zeros((int(sar_profile["height"]), int(sar_profile["width"])), dtype=np.float32)
    dest_g = np.zeros_like(dest_r)
    dest_b = np.zeros_like(dest_r)

    for dest, band_name in zip([dest_r, dest_g, dest_b], ["B04", "B03", "B02"]):
        with rasterio.open(item.assets[band_name].href) as src:
            reproject(
                source=rasterio.band(src, 1),
                destination=dest,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=sar_profile["transform"],
                dst_crs=sar_profile["crs"],
                resampling=Resampling.bilinear
            )

    # Build RGB [0,1] composite
    rgb = np.dstack([dest_r, dest_g, dest_b]) / 10000.0
    rgb = np.clip(rgb, 0, 1)
    return rgb, item.datetime, float(cloud_pct)


# ==========================================
# 4. Generate Comparison Chart
# ==========================================
def generate_comparison(vv_db, ratio_db, rgb, sar_dt, opt_dt, cloud_pct):
    print("\n[3/3] Generating SAR vs Optical Cloud Penetration Comparison...")

    n_panels = 3 if ratio_db is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 7))

    # Panel 1: SAR VV (always clear)
    im0 = axes[0].imshow(vv_db, cmap="gray", vmin=-25, vmax=0)
    axes[0].set_title(f"Sentinel-1 SAR VV (dB)\nDate: {sar_dt.date() if sar_dt else 'N/A'}\n"
                      "Cloud cover: 0% (radar penetrates clouds)", fontsize=10)
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="dB")
    axes[0].axis("off")

    # Panel 2: Optical (cloud-affected)
    if rgb is not None and rgb.ndim == 3:
        axes[1].imshow(rgb)
        opt_label = f"Date: {opt_dt.date() if opt_dt else 'N/A'}"
    else:
        axes[1].imshow(np.zeros((10, 10, 3)))
        opt_label = "No data available"
    axes[1].set_title(f"Sentinel-2 True Color (RGB)\n{opt_label}\n"
                      f"Cloud cover: {cloud_pct:.0f}%  (optical BLIND under clouds)", fontsize=10)
    axes[1].axis("off")

    # Panel 3: VV/VH Ratio (if available)
    if ratio_db is not None and n_panels == 3:
        im2 = axes[2].imshow(ratio_db, cmap="RdYlGn", vmin=-5, vmax=15)
        axes[2].set_title("VV/VH Cross-Polarization Ratio\n"
                          "Green = Vegetation | Yellow = Soil | Red = Water/Ice\n"
                          "(SAR structural discriminator)", fontsize=10)
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="dB")
        axes[2].axis("off")

    plt.suptitle("SAR Cloud Penetration Advantage vs Optical Sensors\n"
                 "Torres del Paine, Patagonia (Winter / High Cloud Season)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    plot_path = os.path.join(OUT_DIR, "sar_vs_optical_cloudy.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"       [SUCCESS] Comparison chart saved: {plot_path}")


# ==========================================
# 5. Main
# ==========================================
def main():
    print("=======================================================")
    print(" GEOCASCADE - SAR vs OPTICAL CLOUD PENETRATION (T4)   ")
    print("=======================================================")
    try:
        catalog = Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=pc.sign_inplace
        )
        vv_db, ratio_db, sar_profile, sar_dt = fetch_sar(catalog, BBOX)
        rgb, opt_dt, cloud_pct = fetch_optical(catalog, BBOX, sar_profile)
        generate_comparison(vv_db, ratio_db, rgb, sar_dt, opt_dt, cloud_pct)
        print("\n[SUCCESS] Chapter 7 Cloud Penetration Comparison (Tier 4) Complete!")
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")


if __name__ == "__main__":
    main()
