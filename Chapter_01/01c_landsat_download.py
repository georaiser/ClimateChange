"""
Chapter 1: 01c_landsat_download.py

Academic Objective:
Learn how to adapt STAC queries to acquire different satellite constellations.
Here we query Landsat 8/9 Collection 2 Level-2 (Surface Reflectance + Surface Temp).
Because this is Level-2, atmospheric correction is already applied -- we can proceed
directly to index calculations.

Upgrades over original:
  - Downloads 7 bands: Blue, Green, Red, NIR, SWIR1, SWIR2, and ST_B10 (Thermal)
  - Downloads QA_PIXEL cloud/shadow mask
  - Validates each asset before attempting download
  - Writes metadata JSON sidecar (date, cloud %, platform, scene ID)
  - Shows file size after each download
  - Summary report at end
  - STAC empty guard
  - Scale factor reminder: Landsat C2-L2 uses DN * 0.0000275 - 0.2 for SR

Landsat C2-L2 Scale Factors (CRITICAL):
  Surface Reflectance: SR = DN * 0.0000275 - 0.2   (stored as uint16)
  Surface Temperature: ST = DN * 0.00341802 + 149.0  (K, then - 273.15 for Celsius)

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
ROI_BBOX       = [-73.30, -51.10, -72.90, -50.80]
DATE_RANGE     = "2023-01-01/2023-02-28"
MAX_CLOUD_COVER = 30  # Landsat revisits every 16 days so we allow more cloud

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "raw")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Bands to download: SR + thermal + QA mask
# Landsat C2-L2 band names in Planetary Computer
TARGET_ASSETS = {
    "blue":    "Blue (0.45-0.52 um, 30m)",
    "green":   "Green (0.52-0.60 um, 30m)",
    "red":     "Red (0.63-0.69 um, 30m)",
    "nir08":   "NIR (0.77-0.90 um, 30m)",
    "swir16":  "SWIR1 (1.55-1.75 um, 30m)",
    "swir22":  "SWIR2 (2.09-2.35 um, 30m)",
    "lwir11":  "Thermal ST_B10 (10.60-11.19 um, 30m)",
    "qa_pixel": "QA cloud/shadow/snow mask",
}


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
# 3. Search Landsat
# ==========================================
def search_landsat(catalog):
    print(f"\n[INFO] Searching Landsat C2-L2 (Surface Reflectance + Thermal)...")
    print(f"       BBOX:  {ROI_BBOX}")
    print(f"       Date:  {DATE_RANGE}")
    print(f"       Cloud: < {MAX_CLOUD_COVER}%")

    search = catalog.search(
        collections=["landsat-c2-l2"],
        bbox=ROI_BBOX,
        datetime=DATE_RANGE,
        query={"eo:cloud_cover": {"lt": MAX_CLOUD_COVER}}
    )
    items = list(search.items())
    print(f"       Found {len(items)} Landsat scenes.")

    if not items:
        print("[WARNING] No Landsat scenes found. Try relaxing date range or cloud cover.")
        return None

    item = sorted(items, key=lambda i: i.properties["eo:cloud_cover"])[0]
    print(f"\n[SELECTED] {item.id}")
    print(f"           Platform:   {item.properties.get('platform', 'N/A')}")
    print(f"           Date:       {item.datetime.strftime('%Y-%m-%d')}")
    print(f"           Cloud:      {item.properties['eo:cloud_cover']:.1f}%")
    print(f"           Sun Elev:   {item.properties.get('view:sun_elevation', 'N/A'):.1f} deg")

    # List available assets
    print(f"\n       Available assets: {list(item.assets.keys())}")
    return item


# ==========================================
# 4. Download Helper
# ==========================================
def download_asset(url, output_path, description=""):
    if os.path.exists(output_path):
        sz = os.path.getsize(output_path) / 1e6
        print(f"       [SKIP] {os.path.basename(output_path)} ({sz:.1f} MB already on disk)")
        return True

    label = description if description else os.path.basename(output_path)
    print(f"       [GET]  {label}...")
    try:
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=65536):
                f.write(chunk)
        sz = os.path.getsize(output_path) / 1e6
        print(f"       [OK]   {os.path.basename(output_path)}  ({sz:.1f} MB)")
        return True
    except Exception as e:
        print(f"       [ERROR] {label}: {e}")
        return False


# ==========================================
# 5. Main
# ==========================================
def main():
    print("==================================================")
    print(" GEOCASCADE - LANDSAT C2-L2 ACQUISITION (7 Bands)")
    print("==================================================")
    print(" Scale factor reminder:")
    print("   Surface Reflectance: SR = DN * 0.0000275 - 0.2")
    print("   Surface Temperature: ST = DN * 0.00341802 + 149.0 K")
    print()

    catalog    = setup_stac_client()
    item       = search_landsat(catalog)
    if not item:
        return

    landsat_dir = os.path.join(OUTPUT_DIR, f"landsat_{item.id}")
    os.makedirs(landsat_dir, exist_ok=True)

    # Write metadata JSON sidecar
    meta = {
        "scene_id":    item.id,
        "platform":    item.properties.get("platform"),
        "date":        item.datetime.isoformat(),
        "cloud_cover": item.properties.get("eo:cloud_cover"),
        "sun_elevation": item.properties.get("view:sun_elevation"),
        "bbox":        item.bbox,
        "scale_sr":    "DN * 0.0000275 - 0.2",
        "scale_st":    "DN * 0.00341802 + 149.0 K",
    }
    meta_path = os.path.join(landsat_dir, "scene_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"\n[INFO] Metadata sidecar: {meta_path}")

    print("\n[INFO] Downloading bands...")
    results = {}
    for asset_key, description in TARGET_ASSETS.items():
        if asset_key not in item.assets:
            print(f"       [SKIP] {asset_key} not in this scene's assets")
            results[asset_key] = "missing"
            continue
        url      = item.assets[asset_key].href
        out_path = os.path.join(landsat_dir, f"{asset_key}.tif")
        ok       = download_asset(url, out_path, description)
        results[asset_key] = "ok" if ok else "failed"

    print("\n" + "=" * 55)
    print(" DOWNLOAD SUMMARY")
    print("=" * 55)
    for key, status in results.items():
        print(f"  {key:12s}  {TARGET_ASSETS.get(key, ''):<35s}  {status}")
    print(f"\n  Output dir: {landsat_dir}")
    print("\n[SUCCESS] Landsat Acquisition complete.")


if __name__ == "__main__":
    main()

