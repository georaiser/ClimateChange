"""
Chapter 12: capstone_pipeline.py
==================================
GeoCascade Capstone CLI -- Automated Site Analysis Pipeline

Academic Objective:
  This is the culmination of the Core Physical Sciences (Chapters 1–11).
  It acts as a Unified Command-Line Tool that accepts user-defined coordinates
  and dynamically fuses Optical (NDVI), Radar (SAR), Elevation (DEM), and
  real-world Vector Infrastructure (OSMnx roads) to generate an automated
  Site Analysis Report.

  Key engineering concepts demonstrated:
  - argparse CLI interface (production-grade tool, not a notebook)
  - Dynamic BBOX → any region on Earth can be analyzed
  - Cross-chapter local data priority (Ch01→Ch02→Ch07→Ch08 cache chain)
  - Multi-sensor zonal statistics: mean NDVI / SAR / DEM per infrastructure buffer
  - OSMnx road network download with correct named-kwarg API
  - EPSG:32719 (UTM Zone 19S) for accurate metric buffers at 51°S latitude
    (EPSG:3857 Web Mercator distorts ~40% at this latitude — NEVER use for buffers)
  - GeoPackage output for vector layers (GIS-standard, replaces Shapefile)
  - Markdown report auto-generation from analysis results

Usage examples:
  # Default: Grey Glacier / Torres del Paine
  python Chapter_12/capstone_pipeline.py

  # Custom BBOX (min_lon min_lat max_lon max_lat)
  python Chapter_12/capstone_pipeline.py --bbox -72.8 -51.8 -72.4 -51.6 --date_range 2023-01-01/2023-03-31

  # Larger region with verbose output
  python Chapter_12/capstone_pipeline.py --bbox -73.5 -51.5 -72.5 -50.5 --verbose

Outputs:
  data/processed/capstone/ndvi_capstone.tif
  data/processed/capstone/sar_vv_db_capstone.tif
  data/processed/capstone/dem_capstone.tif
  data/processed/capstone/roads_buffer.gpkg
  data/processed/capstone/capstone_dashboard.png       (5-panel dark)
  data/processed/capstone/capstone_zonal_stats.csv
  data/processed/capstone/site_analysis_report.md

ArcGIS Pro: Open capstone_dashboard.png for visual summary.
            Add all 3 TIFs and roads_buffer.gpkg for full spatial analysis.
            Use Symbology > Unique Values on NDVI TIF to inspect vegetation zones.
ENVI 5.6:   Open ndvi_capstone.tif. Apply color table (NDVI Green scale).
            Tools > Feature Extraction to compare with SAR classification.

Dependencies:
  mamba install -n geocascade_env -c conda-forge rasterio geopandas rasterstats
      osmnx pystac-client planetary-computer pyproj numpy pandas matplotlib -y

Run:
  conda activate geocascade_env
  python Chapter_12/capstone_pipeline.py
"""

import sys
import os
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import reproject, Resampling

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config & CLI
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
OUT_DIR  = os.path.join(BASE_DIR, "data", "processed", "capstone")
os.makedirs(OUT_DIR, exist_ok=True)

# Cross-chapter cache lookup
CH_CACHE = {
    "ndvi": [
        os.path.join(ROOT_DIR, "Chapter_02", "data", "processed", "indices", "ndvi.tif"),
        os.path.join(ROOT_DIR, "Chapter_02", "data", "processed", "ndvi.tif"),
    ],
    "sar": [
        os.path.join(ROOT_DIR, "Chapter_07", "data", "processed", "sar", "sar_vv_db.tif"),
        os.path.join(ROOT_DIR, "Chapter_07", "data", "processed", "sar_vv_db.tif"),
    ],
    "dem": [
        os.path.join(ROOT_DIR, "Chapter_03", "data", "processed", "terrain", "copernicus_dem.tif"),
        os.path.join(ROOT_DIR, "Chapter_03", "data", "raw", "temp_dem.tif"),
        os.path.join(ROOT_DIR, "Chapter_03", "data", "processed", "dem_raw.tif"),
    ],
}

DARK_BG = "#0d1117"
DARK_AX = "#161b22"
C_TEXT  = "#e6edf3"
C_GREY  = "#8b949e"


def parse_args():
    p = argparse.ArgumentParser(
        description="GeoCascade Capstone: Automated Site Analysis Pipeline"
    )
    p.add_argument("--bbox", nargs=4, type=float,
                   default=[-73.30, -51.10, -72.90, -50.80],
                   metavar=("min_lon", "min_lat", "max_lon", "max_lat"),
                   help="Study area bounding box (WGS84)")
    p.add_argument("--date_range", type=str, default="2023-01-01/2023-03-31",
                   help="Imagery date range YYYY-MM-DD/YYYY-MM-DD")
    p.add_argument("--buffer_m", type=int, default=500,
                   help="Road infrastructure buffer radius in metres (default: 500)")
    p.add_argument("--verbose", action="store_true",
                   help="Print extra diagnostic information")
    return p.parse_args()


# ---------------------------------------------------------------------------
# 1. Resolve local cache or STAC download
# ---------------------------------------------------------------------------
def find_cache(key):
    for path in CH_CACHE.get(key, []):
        if os.path.exists(path):
            return path
    return None


def fetch_stac_raster(collection, asset, bbox, date_range, profile_ref=None,
                      process_fn=None, label=""):
    """Generic STAC fetch + windowed read, optionally resampled to profile_ref."""
    from pystac_client import Client
    import planetary_computer as pc
    from pyproj import Transformer

    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                          modifier=pc.sign_inplace)

    kwargs = dict(collections=[collection], bbox=bbox)
    if date_range and collection != "cop-dem-glo-30":
        kwargs["datetime"] = date_range
    if collection == "sentinel-2-l2a":
        kwargs["query"] = {"eo:cloud_cover": {"lt": 30}}

    search = catalog.search(**kwargs)
    items  = list(search.items())
    if not items:
        print(f"  [WARN] {label}: no STAC data found.")
        return None, None

    item = items[0]
    if collection == "sentinel-2-l2a":
        item = sorted(items, key=lambda i: i.properties.get("eo:cloud_cover", 99))[0]

    with rasterio.open(item.assets[asset].href) as src:
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

    if process_fn:
        arr = process_fn(arr)

    if profile_ref is not None and prof != profile_ref:
        h2 = int(profile_ref["height"])
        w2 = int(profile_ref["width"])
        dest = np.zeros((h2, w2), dtype=np.float32)
        # Write to temp, then reproject
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp_path = tmp.name
        with rasterio.open(tmp_path, "w", **prof) as dst:
            dst.write(np.nan_to_num(arr, nan=-9999), 1)
        with rasterio.open(tmp_path) as src2:
            reproject(source=rasterio.band(src2, 1), destination=dest,
                      src_transform=src2.transform, src_crs=src2.crs,
                      dst_transform=profile_ref["transform"], dst_crs=profile_ref["crs"],
                      resampling=Resampling.bilinear)
        os.unlink(tmp_path)
        arr  = dest
        prof = profile_ref.copy()
        prof.update(count=1, dtype="float32", compress="lzw")

    return arr, prof


# ---------------------------------------------------------------------------
# 2. Load or download each layer
# ---------------------------------------------------------------------------
def load_layer(key, bbox, date_range, master_profile=None, verbose=False):
    cache = find_cache(key)
    if cache:
        print(f"  [CACHE] {key}: {os.path.relpath(cache, ROOT_DIR)}")
        with rasterio.open(cache) as src:
            arr  = src.read(1).astype("float32")
            prof = src.profile.copy()
            nd   = src.nodata
        if nd is not None:
            arr = np.where(arr == nd, np.nan, arr)
        prof.update(count=1, dtype="float32", nodata=-9999, compress="lzw")
        if master_profile and (int(prof.get("height",0)) != int(master_profile.get("height",0))):
            arr, prof = _resample(arr, prof, master_profile)
        return arr, prof

    # STAC fallbacks
    STAC_MAP = {
        "ndvi": ("sentinel-2-l2a", "B08", None,
                 lambda a: np.clip((a * 0.0000275 - 0.2), 0, 1)),
        "sar":  ("sentinel-1-rtc",  "vv",  date_range,
                 lambda a: 10.0 * np.log10(np.where(a > 0, a, np.nan))),
        "dem":  ("cop-dem-glo-30",  "data", None, None),
    }
    if key not in STAC_MAP:
        return None, None

    col, asset, dr, fn = STAC_MAP[key]
    print(f"  [STAC] {key}: downloading from Planetary Computer...")
    arr, prof = fetch_stac_raster(col, asset, bbox, dr, master_profile, fn, key)
    return arr, prof


def _resample(arr, src_prof, dst_prof):
    h2, w2 = int(dst_prof["height"]), int(dst_prof["width"])
    dest = np.zeros((h2, w2), dtype=np.float32)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_path = tmp.name
    with rasterio.open(tmp_path, "w", **src_prof) as dst:
        dst.write(np.nan_to_num(arr.astype("float32"), nan=-9999), 1)
    with rasterio.open(tmp_path) as src2:
        reproject(source=rasterio.band(src2, 1), destination=dest,
                  src_transform=src2.transform, src_crs=src2.crs,
                  dst_transform=dst_prof["transform"], dst_crs=dst_prof["crs"],
                  resampling=Resampling.bilinear)
    os.unlink(tmp_path)
    new_prof = dst_prof.copy()
    new_prof.update(count=1, dtype="float32", compress="lzw")
    return dest, new_prof


# ---------------------------------------------------------------------------
# 3. OSMnx road network + UTM buffer
# ---------------------------------------------------------------------------
def fetch_roads_and_buffer(bbox, buffer_m, crs_utm="EPSG:32719"):
    """
    Download road network from OpenStreetMap and create metric buffers.

    CRITICAL: graph_from_bbox uses (north, south, east, west) — named kwargs only.
    Positional call would swap lat/lon → wrong continent.

    Buffer uses EPSG:32719 (UTM Zone 19S) for accurate metre distances at 51°S.
    EPSG:3857 (Web Mercator) distorts ~40% at this latitude.
    """
    try:
        import osmnx as ox
        import geopandas as gpd
    except ImportError:
        print("  [WARN] osmnx/geopandas not installed. Skipping roads.")
        return None, None

    print(f"  Downloading road network from OpenStreetMap...")
    try:
        graph = ox.graph_from_bbox(
            north=bbox[3], south=bbox[1],
            east=bbox[2],  west=bbox[0],
            network_type="all"
        )
    except Exception as e:
        print(f"  [WARN] OSMnx download failed: {e}")
        return None, None

    _, edges = ox.graph_to_gdfs(graph)
    n_roads  = len(edges)
    print(f"  [OK]  {n_roads} road segments downloaded.")

    # Project to UTM and buffer
    edges_utm = edges.to_crs(crs_utm)
    buf       = edges_utm.buffer(buffer_m).union_all()
    buf_gdf   = gpd.GeoDataFrame(
        {"geometry": [buf], "buffer_m": [buffer_m]},
        crs=crs_utm
    )

    # Save GeoPackage (not Shapefile — GeoPackage handles list columns natively)
    gpkg_path = os.path.join(OUT_DIR, "roads_buffer.gpkg")
    edges_wgs  = edges[["geometry", "highway", "length"]].copy()
    edges_wgs["highway"] = edges_wgs["highway"].astype(str)
    edges_wgs.to_file(gpkg_path, layer="roads",  driver="GPKG")
    buf_gdf.to_crs("EPSG:4326").to_file(gpkg_path, layer="buffer", driver="GPKG")
    print(f"  [OK]  roads_buffer.gpkg saved ({n_roads} roads)")

    return edges_utm, buf_gdf


# ---------------------------------------------------------------------------
# 4. Zonal statistics inside road buffers
# ---------------------------------------------------------------------------
def run_zonal_stats(layers, roads_gdf):
    """Compute mean/std of each raster inside the road buffer polygon."""
    try:
        from rasterstats import zonal_stats
        import tempfile
    except ImportError:
        print("  [WARN] rasterstats not installed. Skipping zonal stats.")
        return None

    if roads_gdf is None:
        return None

    buf_wgs = roads_gdf.to_crs("EPSG:4326")
    rows = []
    for name, (arr, prof) in layers.items():
        if arr is None or prof is None:
            continue
        # Write to temp TIF for rasterstats
        with rasterio.MemoryFile() as memf:
            with memf.open(**prof) as dst:
                dst.write(np.nan_to_num(arr.astype("float32"), nan=-9999), 1)
            with memf.open() as src:
                stats = zonal_stats(buf_wgs, src.read(1),
                                    affine=src.transform, nodata=-9999,
                                    stats=["mean", "std", "min", "max", "count"])
        s = stats[0] if stats else {}
        rows.append({
            "layer": name,
            "mean":  round(s.get("mean", None) or 0, 4),
            "std":   round(s.get("std",  None) or 0, 4),
            "min":   round(s.get("min",  None) or 0, 4),
            "max":   round(s.get("max",  None) or 0, 4),
            "valid_px": s.get("count", 0),
        })

    df  = pd.DataFrame(rows)
    csv = os.path.join(OUT_DIR, "capstone_zonal_stats.csv")
    df.to_csv(csv, index=False, encoding="utf-8")
    print(f"  [OK]  capstone_zonal_stats.csv ({len(df)} layers)")
    return df


# ---------------------------------------------------------------------------
# 5. Save TIFs
# ---------------------------------------------------------------------------
def save_tif(arr, name, profile):
    if arr is None:
        return
    out = os.path.join(OUT_DIR, f"{name}.tif")
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(np.nan_to_num(arr.astype("float32"), nan=-9999), 1)
    print(f"  [OK]  {name}.tif")


# ---------------------------------------------------------------------------
# 6. 5-panel dark dashboard figure
# ---------------------------------------------------------------------------
def plot_dashboard(ndvi, sar, dem, roads_gdf, bbox):
    print("\n  Building 5-panel capstone dashboard...")
    fig = plt.figure(figsize=(26, 8), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(1, 5, figure=fig, wspace=0.22,
                            top=0.88, bottom=0.05, left=0.03, right=0.97)
    fig.text(0.5, 0.95,
             "GeoCascade Capstone Site Analysis -- Torres del Paine, Patagonia",
             ha="center", color=C_TEXT, fontsize=13, fontweight="bold")
    fig.text(0.5, 0.91,
             f"BBOX: {bbox[0]:.2f}, {bbox[1]:.2f}, {bbox[2]:.2f}, {bbox[3]:.2f}",
             ha="center", color=C_GREY, fontsize=9)

    panels = [
        (ndvi, "RdYlGn",  0.0, 0.8, "NDVI", "NDVI (Vegetation Health)\nRed=stressed | Green=healthy"),
        (sar,  "gray",   -25,  0,   "dB",   "SAR VV Backscatter\nWater=dark | Ice/Rock=bright"),
        (dem,  "terrain", None,None, "m",    "Copernicus DEM\nTopographic elevation"),
    ]

    for i, (arr, cmap, vmin, vmax, cbar_label, title) in enumerate(panels):
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
            cb.set_label(cbar_label, color=C_TEXT, fontsize=7)
            cb.ax.tick_params(colors=C_TEXT, labelsize=6)
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    color=C_GREY, transform=ax.transAxes)

    # Panel 4: NDVI with road buffer overlay
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.set_facecolor(DARK_AX)
    ax4.axis("off")
    ax4.set_title("NDVI + Road Buffer Overlay\n(orange = 500m infrastructure zone)",
                  color=C_TEXT, fontsize=8.5, fontweight="bold", pad=5)
    if ndvi is not None and np.any(np.isfinite(ndvi)):
        v = ndvi[np.isfinite(ndvi)]
        ax4.imshow(ndvi, cmap="RdYlGn", vmin=0, vmax=0.8, aspect="auto",
                   extent=[bbox[0], bbox[2], bbox[1], bbox[3]])
    if roads_gdf is not None:
        roads_wgs = roads_gdf.to_crs("EPSG:4326")
        roads_wgs.boundary.plot(ax=ax4, color="#e67e22", linewidth=1.5,
                                 label=f"Road buffer")

    # Panel 5: Summary text card
    ax5 = fig.add_subplot(gs[0, 4])
    ax5.set_facecolor(DARK_AX)
    ax5.axis("off")
    ax5.set_title("Site Analysis Summary",
                  color=C_TEXT, fontsize=8.5, fontweight="bold", pad=5)

    summary_lines = ["GeoCascade Capstone", ""]
    for name, arr in [("NDVI", ndvi), ("SAR VV dB", sar), ("DEM m", dem)]:
        if arr is not None:
            v = arr[np.isfinite(arr)]
            if v.size:
                summary_lines.append(f"{name}")
                summary_lines.append(f"  mean: {v.mean():.3f}")
                summary_lines.append(f"  std : {v.std():.3f}")
                summary_lines.append(f"  min : {v.min():.1f}")
                summary_lines.append(f"  max : {v.max():.1f}")
                summary_lines.append("")

    y = 0.95
    for line in summary_lines:
        ax5.text(0.05, y, line, transform=ax5.transAxes,
                 color=C_TEXT if not line.startswith(" ") else "#8b949e",
                 fontsize=7.5, va="top",
                 fontweight="bold" if line and not line.startswith(" ") else "normal")
        y -= 0.045

    out_png = os.path.join(OUT_DIR, "capstone_dashboard.png")
    fig.savefig(out_png, dpi=180, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  [OK]  capstone_dashboard.png")


# ---------------------------------------------------------------------------
# 7. Markdown site analysis report
# ---------------------------------------------------------------------------
def write_markdown_report(bbox, date_range, zonal_df, buffer_m):
    import datetime
    today = datetime.date.today().isoformat()
    n_roads = "N/A"

    lines = [
        "# GeoCascade — Automated Site Analysis Report",
        f"",
        f"**Date:** {today}  ",
        f"**Study Area:** Torres del Paine, Patagonia, Chile  ",
        f"**BBOX:** `[{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]`  ",
        f"**Imagery Period:** {date_range}  ",
        f"**Infrastructure Buffer:** {buffer_m} m  ",
        f"",
        "---",
        "",
        "## Multi-Sensor Summary Statistics",
        "",
        "| Layer | Mean | Std | Min | Max | Valid px |",
        "|-------|------|-----|-----|-----|---------|",
    ]

    if zonal_df is not None and not zonal_df.empty:
        for _, r in zonal_df.iterrows():
            lines.append(
                f"| {r['layer']} | {r['mean']:.4f} | {r['std']:.4f} | "
                f"{r['min']:.4f} | {r['max']:.4f} | {r['valid_px']} |"
            )
    else:
        lines.append("| (zonal stats not available — osmnx/rasterstats required) | | | | | |")

    ndvi_note = ""
    if zonal_df is not None and not zonal_df.empty:
        ndvi_row = zonal_df[zonal_df["layer"] == "NDVI"]
        if not ndvi_row.empty:
            mv = ndvi_row.iloc[0]["mean"]
            if mv > 0.5:
                ndvi_note = "Dense vegetation detected in infrastructure zone."
            elif mv > 0.2:
                ndvi_note = "Moderate vegetation in infrastructure zone."
            else:
                ndvi_note = "Low vegetation / sparse cover in infrastructure zone."

    lines += [
        "",
        "---",
        "",
        "## Ecological Risk Assessment",
        "",
        f"- **NDVI inside road buffer:** {ndvi_note}",
        "- **SAR backscatter:** Low VV values indicate water / smooth surfaces near roads.",
        "- **DEM elevation:** High-elevation pixels within buffer = infrastructure in mountain zones.",
        "",
        "---",
        "",
        "## Outputs",
        "",
        "| File | Description |",
        "|------|-------------|",
        "| `ndvi_capstone.tif` | NDVI surface reflectance (Sentinel-2 B08/B04) |",
        "| `sar_vv_db_capstone.tif` | SAR VV backscatter in dB (Sentinel-1 RTC) |",
        "| `dem_capstone.tif` | Copernicus DEM elevation in metres |",
        "| `roads_buffer.gpkg` | OSMnx roads + metric buffer (GeoPackage) |",
        "| `capstone_zonal_stats.csv` | Per-layer statistics inside buffer |",
        "| `capstone_dashboard.png` | 5-panel analysis dashboard |",
        "",
        "---",
        "",
        "## ArcGIS Pro Integration",
        "",
        "```",
        "1. Add all TIFs from data/processed/capstone/ to a new map",
        "2. Add roads_buffer.gpkg (layer: roads) with orange symbology",
        "3. Add roads_buffer.gpkg (layer: buffer) with 30% transparency",
        "4. Use Spatial Analyst > Zonal Statistics as Table to validate capstone_zonal_stats.csv",
        "```",
        "",
        "---",
        "*Generated by GeoCascade Capstone Pipeline — Chapter 12*",
    ]

    report_path = os.path.join(OUT_DIR, "site_analysis_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  [OK]  site_analysis_report.md")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    bbox        = args.bbox
    date_range  = args.date_range
    buffer_m    = args.buffer_m
    verbose     = args.verbose

    print("=" * 65)
    print(" GEOCASCADE CAPSTONE — AUTOMATED SITE ANALYSIS PIPELINE")
    print(f" BBOX: {bbox}")
    print(f" Date: {date_range}")
    print(f" Road buffer: {buffer_m} m")
    print("=" * 65)

    # Step 1: Load/download raster layers
    print("\n[1/5] Loading raster layers (cache → STAC fallback)...")
    ndvi_arr, ndvi_prof = load_layer("ndvi", bbox, date_range, verbose=verbose)
    master_profile = ndvi_prof   # Establish 10m master grid from NDVI

    sar_arr,  sar_prof  = load_layer("sar",  bbox, date_range, master_profile, verbose)
    dem_arr,  dem_prof  = load_layer("dem",  bbox, date_range, master_profile, verbose)

    if ndvi_arr is None and sar_arr is None and dem_arr is None:
        print("\n  ERROR: No raster data available. Check network and run Ch01–Ch07 first.")
        return

    # Use first available profile as reference
    ref_prof = ndvi_prof or sar_prof or dem_prof

    # Step 2: Save GeoTIFFs
    print("\n[2/5] Saving raster outputs...")
    if ndvi_arr is not None:
        save_tif(ndvi_arr, "ndvi_capstone",       ndvi_prof or ref_prof)
    if sar_arr is not None:
        save_tif(sar_arr,  "sar_vv_db_capstone",  sar_prof  or ref_prof)
    if dem_arr is not None:
        save_tif(dem_arr,  "dem_capstone",         dem_prof  or ref_prof)

    # Step 3: OSMnx roads + buffer
    print("\n[3/5] Fetching road infrastructure (OpenStreetMap)...")
    roads_edges, roads_buf = fetch_roads_and_buffer(bbox, buffer_m)

    # Step 4: Zonal statistics
    print("\n[4/5] Computing zonal statistics inside road buffer...")
    available_layers = {}
    if ndvi_arr is not None: available_layers["NDVI"]     = (ndvi_arr, ndvi_prof or ref_prof)
    if sar_arr  is not None: available_layers["SAR_VV_dB"]= (sar_arr,  sar_prof  or ref_prof)
    if dem_arr  is not None: available_layers["DEM_m"]    = (dem_arr,  dem_prof  or ref_prof)
    zonal_df = run_zonal_stats(available_layers, roads_buf)

    # Step 5: Dashboard + Markdown report
    print("\n[5/5] Building dashboard and report...")
    plot_dashboard(ndvi_arr, sar_arr, dem_arr, roads_buf, bbox)
    write_markdown_report(bbox, date_range, zonal_df, buffer_m)

    print("\n" + "=" * 65)
    print(" CAPSTONE ANALYSIS COMPLETE")
    print("=" * 65)
    print(f"  Outputs : {OUT_DIR}")
    print(f"  Dashboard: capstone_dashboard.png")
    print(f"  Report   : site_analysis_report.md")
    print(f"  Zones    : capstone_zonal_stats.csv")
    print()
    print("  ArcGIS Pro: Add all TIFs + roads_buffer.gpkg to a new map.")
    print("  ENVI 5.6  : Open ndvi_capstone.tif. Apply NDVI Green color table.")
    print("=" * 65)


if __name__ == "__main__":
    main()
