"""
GeoCascade Chapter 01 - Script 02: Multi-Sensor Satellite Acquisition
=======================================================================
Downloads satellite imagery from Microsoft Planetary Computer STAC for
the Torres del Paine study area.

IMPORTANT -- ATMOSPHERIC CORRECTION STATUS
------------------------------------------
  Sentinel-2 L2A   : ALREADY corrected by ESA Sen2Cor. Use directly.
  Landsat 9 C2 L2  : ALREADY corrected by USGS LaSRC.  Use directly.
  Sentinel-2 L1C   : Needs correction. Use FLAASH (ENVI) or Sen2Cor.
  Landsat 9 L1TP   : Needs correction. Use FLAASH (ENVI) or DOS1.
  --> For ENVI FLAASH correction see: envi/01_flaash_correction.pro

This script downloads only L2-level products (atmospherically corrected).

SENSORS / BANDS
---------------
  Sentinel-2 L2A (sentinel-2-l2a collection)
    B02 (Blue 10m), B03 (Green 10m), B04 (Red 10m),
    B08 (NIR 10m),  B11 (SWIR-1 20m)

  Landsat 9 C2 Level-2 (landsat-c2-l2 collection)
    SR_B2 (Blue),   SR_B3 (Green), SR_B4 (Red),
    SR_B5 (NIR),    SR_B6 (SWIR-1), ST_B10 (Thermal)

  Copernicus DEM GLO-30 (cop-dem-glo-30 collection)
    All tiles covering BBOX -- elevation data, no date filter.

SEARCH PARAMETERS
-----------------
  BBOX        : [-73.30, -51.10, -72.90, -50.80]
  DATE RANGE  : 2023-01-01 to 2023-02-28 (S2 + Landsat)
  MAX CLOUD   : 15% (S2), 20% (Landsat)

FEATURES
--------
  - Prints available asset keys per scene (educational)
  - Skip-if-exists for each band file
  - JSON metadata sidecar per scene
  - File size reported after each download
  - Summary table at end

OUTPUT STRUCTURE
----------------
  data/raw/sentinel2_l2a_{scene_id}/B02.tif, B03.tif, B04.tif, B08.tif, B11.tif
  data/raw/sentinel2_l2a_{scene_id}/metadata.json
  data/raw/landsat_c2l2_{scene_id}/SR_B2.tif, SR_B3.tif, SR_B4.tif,
                                    SR_B5.tif, SR_B6.tif, ST_B10.tif
  data/raw/landsat_c2l2_{scene_id}/metadata.json
  data/raw/dem_{tile_id}/copernicus_dem_30m.tif
  data/raw/dem_{tile_id}/metadata.json

Author : GeoCascade Project
"""

import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
import json
import time
import traceback
import warnings
import requests
import numpy as np
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Optional STAC imports -- graceful if pystac-client not installed
# ---------------------------------------------------------------------------
try:
    import pystac_client
    import planetary_computer as pc
    HAS_STAC = True
except ImportError:
    HAS_STAC = False
    print("[WARN] pystac-client or planetary-computer not installed.")
    print("       Install with: pip install pystac-client planetary-computer")

try:
    import rasterio
    from rasterio.enums import Resampling
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("[WARN] rasterio not installed: pip install rasterio")

try:
    import urllib.request as urllib_request
    HAS_URLLIB = True
except ImportError:
    HAS_URLLIB = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
DATA_ROOT  = SCRIPT_DIR / "data"
RAW_DIR    = DATA_ROOT / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Search parameters
# ---------------------------------------------------------------------------
BBOX       = [-73.30, -51.10, -72.90, -50.80]   # [min_lon, min_lat, max_lon, max_lat]
DATE_RANGE = "2023-01-01/2023-02-28"
MAX_CLOUD_S2      = 15   # percent
MAX_CLOUD_LANDSAT = 20   # percent

# Planetary Computer STAC endpoint (public, no token required for most data)
PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

# ---------------------------------------------------------------------------
# Band configs
# ---------------------------------------------------------------------------
S2_BANDS = ["B02", "B03", "B04", "B08", "B11"]

LANDSAT_BANDS = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "ST_B10"]
LANDSAT_BAND_ALIASES = {
    # some Landsat C2L2 STAC items expose lowercase or alternate keys
    "SR_B2":  ["SR_B2", "blue",   "sr_b2"],
    "SR_B3":  ["SR_B3", "green",  "sr_b3"],
    "SR_B4":  ["SR_B4", "red",    "sr_b4"],
    "SR_B5":  ["SR_B5", "nir08",  "sr_b5"],
    "SR_B6":  ["SR_B6", "swir16", "sr_b6"],
    "ST_B10": ["ST_B10","lwir11", "st_b10"],
}

# ===========================================================================
# Helpers
# ===========================================================================
def _safe_name(s: str, maxlen: int = 40) -> str:
    """Make a filesystem-safe short name."""
    import re
    s = re.sub(r"[^\w\-]", "_", s)
    return s[:maxlen]


def _download_file(url: str, out_path: Path, retries: int = 3) -> bool:
    """Download a file with retry; return True on success."""
    if out_path.exists() and out_path.stat().st_size > 1024:
        return True  # skip-if-exists
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=300, stream=True)
            resp.raise_for_status()
            with open(out_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
            return True
        except Exception as exc:
            print(f"    [RETRY {attempt}/{retries}] {exc}")
            if attempt < retries:
                time.sleep(4 * attempt)
    return False


def _write_metadata(scene_dir: Path, meta: dict) -> None:
    meta_path = scene_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)


def _fmt_mb(path: Path) -> str:
    if path.exists():
        return f"{path.stat().st_size / (1024*1024):.2f} MB"
    return "0.00 MB"


def _print_assets(item, label: str = "") -> None:
    """Print available asset keys for a STAC item (educational)."""
    keys = list(item.assets.keys())
    print(f"  Available assets ({label}): {', '.join(sorted(keys))}")


def _sign_item(item):
    """Sign item for Planetary Computer if pc is available."""
    if HAS_STAC:
        try:
            return pc.sign(item)
        except Exception:
            return item
    return item


def _resolve_asset_url(item, candidates: list) -> tuple:
    """Return (key, url) for the first matching asset key."""
    for key in candidates:
        if key in item.assets:
            asset = item.assets[key]
            return key, asset.href
    return None, None


# ===========================================================================
# 1. Sentinel-2 L2A
# ===========================================================================
def download_sentinel2(catalog) -> dict:
    print("\n[S2] Searching Sentinel-2 L2A ...")
    results = {"found": 0, "downloaded": 0, "skipped": 0, "failed": 0,
               "total_mb": 0.0, "scenes": []}

    try:
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=BBOX,
            datetime=DATE_RANGE,
            query={"eo:cloud_cover": {"lt": MAX_CLOUD_S2}},
            max_items=20,
        )
        items = list(search.items())
    except Exception as exc:
        print(f"  [ERROR] S2 search failed: {exc}")
        return results

    results["found"] = len(items)
    print(f"  Found {len(items)} Sentinel-2 scenes (cloud < {MAX_CLOUD_S2}%)")

    for item in items:
        item   = _sign_item(item)
        sid    = _safe_name(item.id)
        cc     = item.properties.get("eo:cloud_cover", "N/A")
        dt     = item.properties.get("datetime", "N/A")[:10]
        print(f"\n  Scene: {sid[:35]}  date={dt}  cloud={cc}%")
        _print_assets(item, label="S2 L2A")

        scene_dir = RAW_DIR / f"sentinel2_l2a_{sid}"
        scene_dir.mkdir(parents=True, exist_ok=True)

        band_status = {}
        scene_mb    = 0.0

        for band in S2_BANDS:
            out_path = scene_dir / f"{band}.tif"
            if out_path.exists() and out_path.stat().st_size > 1024:
                print(f"    {band}: SKIP (exists {_fmt_mb(out_path)})")
                results["skipped"] += 1
                band_status[band] = "skipped"
                scene_mb += out_path.stat().st_size / (1024 * 1024)
                continue

            key, url = _resolve_asset_url(item, [band, band.lower()])
            if url is None:
                print(f"    {band}: NOT FOUND in assets")
                results["failed"] += 1
                band_status[band] = "not_found"
                continue

            print(f"    {band} ({key}): downloading ...", end=" ", flush=True)
            ok = _download_file(url, out_path)
            if ok:
                mb = out_path.stat().st_size / (1024 * 1024)
                scene_mb += mb
                print(f"{mb:.2f} MB")
                results["downloaded"] += 1
                band_status[band] = "downloaded"
            else:
                print("FAILED")
                results["failed"] += 1
                band_status[band] = "failed"

        # Metadata sidecar
        meta = {
            "scene_id":         item.id,
            "sensor":           "Sentinel-2 L2A",
            "date":             dt,
            "cloud_cover":      cc,
            "bbox":             BBOX,
            "correction_level": "L2A (ESA Sen2Cor -- atmospherically corrected)",
            "bands":            band_status,
            "stac_id":          item.id,
        }
        _write_metadata(scene_dir, meta)
        results["scenes"].append({"id": item.id, "mb": scene_mb})
        results["total_mb"] += scene_mb

    return results


# ===========================================================================
# 2. Landsat 9 C2 Level-2
# ===========================================================================
def download_landsat(catalog) -> dict:
    print("\n[L9] Searching Landsat 9 Collection 2 Level-2 ...")
    results = {"found": 0, "downloaded": 0, "skipped": 0, "failed": 0,
               "total_mb": 0.0, "scenes": []}

    try:
        search = catalog.search(
            collections=["landsat-c2-l2"],
            bbox=BBOX,
            datetime=DATE_RANGE,
            query={
                "eo:cloud_cover":   {"lt": MAX_CLOUD_LANDSAT},
                "platform":         {"in": ["landsat-9"]},
            },
            max_items=10,
        )
        items = list(search.items())
    except Exception as exc:
        print(f"  [ERROR] Landsat search failed: {exc}")
        return results

    results["found"] = len(items)
    print(f"  Found {len(items)} Landsat 9 scenes (cloud < {MAX_CLOUD_LANDSAT}%)")

    for item in items:
        item   = _sign_item(item)
        sid    = _safe_name(item.id)
        cc     = item.properties.get("eo:cloud_cover", "N/A")
        dt     = item.properties.get("datetime", "N/A")[:10]
        print(f"\n  Scene: {sid[:35]}  date={dt}  cloud={cc}%")
        _print_assets(item, label="Landsat C2L2")

        scene_dir = RAW_DIR / f"landsat_c2l2_{sid}"
        scene_dir.mkdir(parents=True, exist_ok=True)

        band_status = {}
        scene_mb    = 0.0

        for band in LANDSAT_BANDS:
            out_path   = scene_dir / f"{band}.tif"
            candidates = LANDSAT_BAND_ALIASES.get(band, [band])

            if out_path.exists() and out_path.stat().st_size > 1024:
                print(f"    {band}: SKIP (exists {_fmt_mb(out_path)})")
                results["skipped"] += 1
                band_status[band] = "skipped"
                scene_mb += out_path.stat().st_size / (1024 * 1024)
                continue

            key, url = _resolve_asset_url(item, candidates)
            if url is None:
                print(f"    {band}: NOT FOUND in assets (tried: {candidates})")
                results["failed"] += 1
                band_status[band] = "not_found"
                continue

            print(f"    {band} ({key}): downloading ...", end=" ", flush=True)
            ok = _download_file(url, out_path)
            if ok:
                mb = out_path.stat().st_size / (1024 * 1024)
                scene_mb += mb
                print(f"{mb:.2f} MB")
                results["downloaded"] += 1
                band_status[band] = "downloaded"
            else:
                print("FAILED")
                results["failed"] += 1
                band_status[band] = "failed"

        # Metadata sidecar
        meta = {
            "scene_id":         item.id,
            "sensor":           "Landsat 9 Collection 2 Level-2",
            "date":             dt,
            "cloud_cover":      cc,
            "bbox":             BBOX,
            "correction_level": "C2 L2SP (USGS LaSRC -- atmospherically corrected)",
            "bands":            band_status,
            "stac_id":          item.id,
        }
        _write_metadata(scene_dir, meta)
        results["scenes"].append({"id": item.id, "mb": scene_mb})
        results["total_mb"] += scene_mb

    return results


# ===========================================================================
# 3. Copernicus DEM GLO-30
# ===========================================================================
def download_dem(catalog) -> dict:
    print("\n[DEM] Searching Copernicus DEM GLO-30 ...")
    results = {"found": 0, "downloaded": 0, "skipped": 0, "failed": 0,
               "total_mb": 0.0, "scenes": []}

    try:
        search = catalog.search(
            collections=["cop-dem-glo-30"],
            bbox=BBOX,
            max_items=20,
        )
        items = list(search.items())
    except Exception as exc:
        print(f"  [ERROR] DEM search failed: {exc}")
        return results

    results["found"] = len(items)
    print(f"  Found {len(items)} DEM tiles")

    for item in items:
        item   = _sign_item(item)
        tid    = _safe_name(item.id)
        print(f"\n  Tile: {tid[:45]}")
        _print_assets(item, label="CopDEM")

        tile_dir = RAW_DIR / f"dem_{tid}"
        tile_dir.mkdir(parents=True, exist_ok=True)
        out_path = tile_dir / "copernicus_dem_30m.tif"

        if out_path.exists() and out_path.stat().st_size > 1024:
            print(f"    SKIP (exists {_fmt_mb(out_path)})")
            results["skipped"] += 1
            results["total_mb"] += out_path.stat().st_size / (1024 * 1024)
            continue

        # CopDEM asset key is typically "data"
        key, url = _resolve_asset_url(item, ["data", "dem", "elevation"])
        if url is None:
            print(f"    DEM asset NOT FOUND")
            results["failed"] += 1
            continue

        print(f"    data ({key}): downloading ...", end=" ", flush=True)
        ok = _download_file(url, out_path)
        if ok:
            mb = out_path.stat().st_size / (1024 * 1024)
            results["total_mb"] += mb
            print(f"{mb:.2f} MB")
            results["downloaded"] += 1
        else:
            print("FAILED")
            results["failed"] += 1

        # Metadata sidecar
        meta = {
            "scene_id":   item.id,
            "sensor":     "Copernicus DEM GLO-30",
            "date":       "2021",
            "cloud_cover": None,
            "bbox":       BBOX,
            "correction_level": "DEM (no atmospheric correction needed)",
            "bands":      {"data": "elevation (m)"},
            "stac_id":    item.id,
        }
        _write_metadata(tile_dir, meta)
        results["scenes"].append({"id": item.id, "mb": mb if ok else 0})

    return results


# ===========================================================================
# Summary table
# ===========================================================================
def print_summary(s2_res, l9_res, dem_res) -> None:
    print("\n" + "=" * 70)
    print("ACQUISITION SUMMARY")
    print("=" * 70)
    header = f"  {'Sensor':<28} {'Found':>6} {'Dld':>5} {'Skip':>5} {'Fail':>5} {'MB':>8}"
    print(header)
    print("  " + "-" * 62)

    rows = [
        ("Sentinel-2 L2A (bands)",       s2_res),
        ("Landsat 9 C2L2 (bands)",       l9_res),
        ("Copernicus DEM GLO-30 (tiles)", dem_res),
    ]
    for label, r in rows:
        print(
            f"  {label:<28} {r['found']:>6} {r['downloaded']:>5} "
            f"{r['skipped']:>5} {r['failed']:>5} {r['total_mb']:>7.1f}"
        )

    total_mb = s2_res["total_mb"] + l9_res["total_mb"] + dem_res["total_mb"]
    print("  " + "-" * 62)
    print(f"  {'TOTAL':>28}                              {total_mb:>7.1f} MB")
    print("=" * 70)

    # List output dirs
    print("\nOutput directories:")
    for d in sorted(RAW_DIR.iterdir()):
        if d.is_dir() and d.name.startswith(("sentinel2_", "landsat_c2", "dem_")):
            files = list(d.glob("*.tif"))
            mb    = sum(f.stat().st_size for f in files) / (1024 * 1024)
            print(f"  {d.name:<45}  {len(files)} TIF  {mb:.1f} MB")


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    print("=" * 70)
    print("GeoCascade Ch01 -- Multi-Sensor Satellite Acquisition")
    print("=" * 70)
    print(f"  BBOX       : {BBOX}")
    print(f"  Date range : {DATE_RANGE}")
    print(f"  Max cloud  : S2={MAX_CLOUD_S2}%  Landsat={MAX_CLOUD_LANDSAT}%")
    print(f"  Output dir : {RAW_DIR}")
    print(f"  Timestamp  : {datetime.now().isoformat()}")
    print()
    print("  NOTE: S2 L2A and Landsat C2 L2 are ALREADY atmospherically")
    print("  corrected. For raw TOA needing FLAASH see:")
    print("  envi/01_flaash_correction.pro")
    print()

    if not HAS_STAC:
        print("[FATAL] pystac-client / planetary-computer not installed.")
        print("        Run: pip install pystac-client planetary-computer")
        sys.exit(1)

    if not HAS_RASTERIO:
        print("[WARN] rasterio not available; files will download but not be validated.")

    # Open catalog
    try:
        catalog = pystac_client.Client.open(
            PC_STAC_URL,
            modifier=pc.sign_inplace,
        )
        print(f"[OK] Connected to Planetary Computer STAC: {PC_STAC_URL}")
    except Exception as exc:
        print(f"[FATAL] Cannot connect to Planetary Computer: {exc}")
        traceback.print_exc()
        sys.exit(1)

    s2_res  = {"found": 0, "downloaded": 0, "skipped": 0, "failed": 0, "total_mb": 0.0, "scenes": []}
    l9_res  = {"found": 0, "downloaded": 0, "skipped": 0, "failed": 0, "total_mb": 0.0, "scenes": []}
    dem_res = {"found": 0, "downloaded": 0, "skipped": 0, "failed": 0, "total_mb": 0.0, "scenes": []}

    try:
        s2_res = download_sentinel2(catalog)
    except Exception as exc:
        print(f"[ERROR] Sentinel-2 download failed: {exc}")
        traceback.print_exc()

    try:
        l9_res = download_landsat(catalog)
    except Exception as exc:
        print(f"[ERROR] Landsat download failed: {exc}")
        traceback.print_exc()

    try:
        dem_res = download_dem(catalog)
    except Exception as exc:
        print(f"[ERROR] DEM download failed: {exc}")
        traceback.print_exc()

    print_summary(s2_res, l9_res, dem_res)
    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[FATAL] {exc}")
        traceback.print_exc()
        sys.exit(1)
