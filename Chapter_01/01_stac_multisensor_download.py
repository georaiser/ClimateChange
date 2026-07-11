"""
Chapter 1: 01_stac_multisensor_download.py

Academic Objective:
Learn how to programmatically search and download geospatial data using
SpatioTemporal Asset Catalogs (STAC) APIs. This eliminates manual web-portal
downloads and ensures a reproducible data pipeline.

Upgrades over original:
  - STAC empty guard: prints actionable advice if no scenes are found
  - Available asset list printed before download (educational)
  - File-size progress shown after each download
  - JSON metadata sidecar per scene (date, cloud%, bbox, platform)
  - Summary table at end (scenes found, downloaded, total MB)
  - L1C vs L2A note: this script defaults to L2A (BOA reflectance)
  - DEM multi-tile note: BBOX may span multiple 1-deg tiles; all are downloaded

Region of Interest (ROI):
  Torres del Paine National Park & Grey Glacier, Magallanes Region, Chile.

Dependencies:
mamba install -n geocascade_env -c conda-forge pystac-client planetary-computer requests -y
"""

import os
import json
import requests
from pystac_client import Client
import planetary_computer as pc

# ==========================================
# 1. Configuration
# ==========================================
ROI_BBOX        = [-73.30, -51.10, -72.90, -50.80]   # [min_lon, min_lat, max_lon, max_lat]
DATE_RANGE      = "2023-01-01/2023-02-28"             # Patagonian summer, minimal snow
MAX_CLOUD_COVER = 10

# Sentinel-2 bands to download
# B02-B04 are 10m (RGB), B08 is 10m (NIR), B11 is 20m (SWIR1)
TARGET_BANDS = ["B02", "B03", "B04", "B08", "B11"]

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================
# 2. STAC Client
# ==========================================
def setup_stac_client():
    print("[INFO] Connecting to Planetary Computer STAC API...")
    return Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace
    )


# ==========================================
# 3. Sentinel-2 Search
# ==========================================
def search_sentinel2(catalog, collection="sentinel-2-l2a"):
    """
    Searches for Sentinel-2 images. Defaults to L2A (Surface Reflectance).
    L1C (TOA) is available for ENVI FLAASH workflows -- use script 01b.

    NOTE: Sentinel-2 has a ~5-day revisit at mid-latitudes. A 2-month window
    with <10% cloud cover will typically return 3-8 scenes in Patagonia.
    """
    print(f"\n[INFO] Searching {collection} (cloud < {MAX_CLOUD_COVER}%)...")
    print(f"       BBOX: {ROI_BBOX}")
    print(f"       Date: {DATE_RANGE}")
    search = catalog.search(
        collections=[collection],
        bbox=ROI_BBOX,
        datetime=DATE_RANGE,
        query={"eo:cloud_cover": {"lt": MAX_CLOUD_COVER}}
    )
    items = list(search.items())
    print(f"       Found {len(items)} {collection} scenes.")
    if not items:
        print("[WARNING] No Sentinel-2 scenes found.")
        print("  Try: increasing MAX_CLOUD_COVER, widening DATE_RANGE, or checking BBOX.")
        return None
    item = sorted(items, key=lambda i: i.properties["eo:cloud_cover"])[0]
    print(f"\n[SELECTED] {item.id}")
    print(f"           Date:  {item.datetime.strftime('%Y-%m-%d')}")
    print(f"           Cloud: {item.properties['eo:cloud_cover']:.1f}%")
    print(f"           Available assets: {list(item.assets.keys())}")
    return item


# ==========================================
# 4. Copernicus DEM Search
# ==========================================
def search_dem(catalog):
    """
    Searches for Copernicus DEM GLO-30 (30m) tiles.
    NOTE: DEMs are static (not time-varying). The BBOX may span multiple
    1-degree tiles -- this function returns all intersecting tiles.
    """
    print(f"\n[INFO] Searching for Copernicus DEM GLO-30 tiles...")
    search = catalog.search(
        collections=["cop-dem-glo-30"],
        bbox=ROI_BBOX
    )
    items = list(search.items())
    print(f"       Found {len(items)} DEM tile(s).")
    if not items:
        print("[WARNING] No DEM tiles found. Check BBOX.")
        return []
    for it in items:
        print(f"         Tile: {it.id}")
    return items


# ==========================================
# 5. Asset Download Helper
# ==========================================
def download_asset(url, output_path, description=""):
    """Downloads a single file in chunks, shows size, skips if already on disk."""
    if os.path.exists(output_path):
        sz = os.path.getsize(output_path) / 1e6
        print(f"       [SKIP] {os.path.basename(output_path)} ({sz:.1f} MB already on disk)")
        return True
    label = description or os.path.basename(output_path)
    print(f"       [GET]  {label}...")
    try:
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=65536):
                f.write(chunk)
        sz = os.path.getsize(output_path) / 1e6
        print(f"       [OK]   {os.path.basename(output_path)} ({sz:.1f} MB)")
        return True
    except Exception as e:
        print(f"       [ERROR] {label}: {e}")
        return False


# ==========================================
# 6. Main
# ==========================================
def main():
    print("==================================================")
    print(" GEOCASCADE PIPELINE - STAC ACQUISITION SCRIPT")
    print(" Sentinel-2 L2A (BOA) + Copernicus DEM GLO-30")
    print("==================================================")

    catalog  = setup_stac_client()
    total_mb = 0
    results  = []

    # 1. Sentinel-2 L2A (default: Surface Reflectance, no atmospheric correction needed)
    s2_item = search_sentinel2(catalog, "sentinel-2-l2a")
    if s2_item:
        s2_dir = os.path.join(OUTPUT_DIR, f"sentinel2_l2a_{s2_item.id}")
        os.makedirs(s2_dir, exist_ok=True)
        # Write metadata sidecar
        meta = {
            "scene_id":    s2_item.id,
            "collection":  "sentinel-2-l2a",
            "date":        s2_item.datetime.isoformat(),
            "cloud_cover": s2_item.properties.get("eo:cloud_cover"),
            "bbox":        s2_item.bbox,
            "bands":       TARGET_BANDS,
        }
        with open(os.path.join(s2_dir, "scene_metadata.json"), "w") as f:
            json.dump(meta, f, indent=2, default=str)
        print(f"\n[INFO] Downloading {len(TARGET_BANDS)} Sentinel-2 bands...")
        for band in TARGET_BANDS:
            if band in s2_item.assets:
                desc = f"S2 {band} (20m SWIR)" if band == "B11" else f"S2 {band}"
                ok   = download_asset(s2_item.assets[band].href,
                                      os.path.join(s2_dir, f"{band}.tif"), desc)
                if ok:
                    mb = os.path.getsize(os.path.join(s2_dir, f"{band}.tif")) / 1e6
                    total_mb += mb
            else:
                print(f"       [SKIP] {band} not in scene assets")
        results.append(("Sentinel-2 L2A", s2_item.id, s2_dir))

    # 2. Copernicus DEM (all tiles covering BBOX)
    dem_items = search_dem(catalog)
    for dem_item in dem_items:
        dem_dir = os.path.join(OUTPUT_DIR, f"dem_{dem_item.id}")
        os.makedirs(dem_dir, exist_ok=True)
        if "data" in dem_item.assets:
            out_path = os.path.join(dem_dir, "copernicus_dem_30m.tif")
            ok = download_asset(dem_item.assets["data"].href, out_path, "CopDEM GLO-30")
            if ok:
                total_mb += os.path.getsize(out_path) / 1e6
        results.append(("CopDEM GLO-30", dem_item.id, dem_dir))

    print("\n" + "=" * 55)
    print(" ACQUISITION SUMMARY")
    print("=" * 55)
    for category, scene_id, directory in results:
        print(f"  {category:18s}  {scene_id[:40]}")
        print(f"                     -> {directory}")
    print(f"\n  Total downloaded: {total_mb:.1f} MB")
    print("\n[SUCCESS] Chapter 1 Acquisition complete.")
    print("  Next: python 02_atmospheric_correction.py only if you downloaded Sentinel-2 L1C.")


if __name__ == "__main__":
    main()

