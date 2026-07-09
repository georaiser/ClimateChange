"""
Chapter 12: Robust Site Analysis Capstone (CLI Version)

This script is the culmination of the Core Physical Sciences (Chapters 1-7).
It acts as a Unified Command-Line Tool, taking user-defined coordinates and 
dynamically fusing Optical (NDVI), Radar (SAR), and Elevation (DEM) data with 
real-world Vector Infrastructure (OSMnx) to generate an automated Site Analysis Report.

Usage:
python capstone_pipeline.py --bbox -72.8 -51.8 -72.4 -51.6 --date_range 2023-01-01/2023-03-31
"""

import os
import argparse
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import reproject, Resampling
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from pystac_client import Client
import planetary_computer as pc
from pyproj import Transformer
from rasterstats import zonal_stats
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. Configuration & CLI
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(description="GeoCascade Automated Site Analysis Capstone")
    parser.add_argument("--bbox", nargs=4, type=float, default=[-72.8, -51.8, -72.4, -51.6],
                        help="Bounding box in format: min_lon min_lat max_lon max_lat")
    parser.add_argument("--date_range", type=str, default="2023-01-01/2023-03-31",
                        help="Date range for optical/radar imagery (YYYY-MM-DD/YYYY-MM-DD)")
    return parser.parse_args()

# ==========================================
# 2. Vector Sourcing (OpenStreetMap)
# ==========================================
def fetch_infrastructure(bbox):
    print("\n[INFO] Fetching Real-World Infrastructure (Roads) from OpenStreetMap...")
    import osmnx as ox
    # CRITICAL: graph_from_bbox uses (north, south, east, west) order — NOT (min_lon, min_lat, max_lon, max_lat).
    # Passing bbox positionally would route bbox[0]=-73.30 as 'north' latitude — silently wrong location.
    try:
        graph = ox.graph_from_bbox(
            north=bbox[3], south=bbox[1],
            east=bbox[2], west=bbox[0],
            network_type='all'
        )
    except Exception as e:
        raise RuntimeError(f"OSMnx download failed: {e}. Check BBOX coordinates and internet access.")

    _, edges = ox.graph_to_gdfs(graph)

    roads_shp = os.path.join(OUT_DIR, "infrastructure_roads.shp")
    for col in edges.columns:
        if edges[col].apply(lambda x: isinstance(x, (list, tuple))).any():
            edges[col] = edges[col].astype(str)

    edges.to_file(roads_shp)
    print(f"       [SUCCESS] Downloaded {len(edges)} road segments.")
    return edges

# ==========================================
# 3. Multi-Sensor Data Fusion (STAC)
# ==========================================
def fetch_and_fuse_rasters(bbox, date_range):
    print("\n[INFO] Connecting to Planetary Computer for Multi-Sensor Fusion...")
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
    
    # 1. Base Grid (DEM)
    print("       Fetching Copernicus DEM...")
    search_dem = catalog.search(collections=["cop-dem-glo-30"], bbox=bbox)
    items_dem = list(search_dem.items())
    if not items_dem:
        raise RuntimeError("No DEM found for BBOX. Check coordinates or Planetary Computer access.")
    item_dem = items_dem[0]
    
    with rasterio.open(item_dem.assets["data"].href) as src:
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        minx, miny = transformer.transform(bbox[0], bbox[1])
        maxx, maxy = transformer.transform(bbox[2], bbox[3])
        window = from_bounds(minx, miny, maxx, maxy, src.transform)
        transform = rasterio.windows.transform(window, src.transform)
        
        dem = src.read(1, window=window).astype('float32')
        dem = np.where(dem < 0, np.nan, dem)
        
        master_profile = src.profile.copy()
        master_profile.update(
            dtype=rasterio.float32, count=1, nodata=-9999,
            height=int(round(window.height)), width=int(round(window.width)),
            transform=transform
        )
        
    # Helper to resample other layers to DEM grid
    def resample_to_master(href, native_process_func=None):
        dest = np.zeros((master_profile['height'], master_profile['width']), dtype=np.float32)
        with rasterio.open(href) as rsrc:
            reproject(
                source=rasterio.band(rsrc, 1),
                destination=dest,
                src_transform=rsrc.transform,
                src_crs=rsrc.crs,
                dst_transform=master_profile['transform'],
                dst_crs=master_profile['crs'],
                resampling=Resampling.bilinear
            )
        if native_process_func:
            dest = native_process_func(dest)
        return dest

    # 2. Optical (NDVI via Sentinel-2)
    print("       Fetching Sentinel-2 (Optical) and computing NDVI...")
    search_s2 = catalog.search(collections=["sentinel-2-l2a"], bbox=bbox, datetime=date_range, query={"eo:cloud_cover": {"lt": 20}})
    items_s2 = list(search_s2.items())
    if not items_s2:
        raise Exception("No Sentinel-2 imagery found. Try a different date range or BBOX.")
    item_s2 = sorted(items_s2, key=lambda i: i.properties["eo:cloud_cover"])[0]
    
    red = resample_to_master(item_s2.assets["B04"].href)
    nir = resample_to_master(item_s2.assets["B08"].href)
    ndvi = np.where((nir + red) == 0, np.nan, (nir - red) / (nir + red))
    
    # 3. Radar (SAR VV via Sentinel-1)
    print("       Fetching Sentinel-1 (Radar) and computing backscatter dB...")
    search_s1 = catalog.search(collections=["sentinel-1-rtc"], bbox=bbox, datetime=date_range)
    items_s1 = list(search_s1.items())
    if not items_s1:
        raise Exception("No Sentinel-1 imagery found.")
    item_s1 = items_s1[0]
    
    def sar_to_db(arr):
        return 10 * np.log10(np.where(arr <= 0, np.nan, arr))
        
    sar_db = resample_to_master(item_s1.assets["vv"].href, native_process_func=sar_to_db)
    
    return dem, ndvi, sar_db, master_profile

# ==========================================
# 4. Vector-Raster Overlay & Statistics
# ==========================================
def export_and_analyze(roads_gdf, dem, ndvi, sar_db, profile):
    print("\n[INFO] Running Zonal Statistics on 1km Human Impact Buffer...")
    
    # Export Rasters temporarily for zonal_stats
    dem_path = os.path.join(OUT_DIR, "capstone_dem.tif")
    ndvi_path = os.path.join(OUT_DIR, "capstone_ndvi.tif")
    sar_path = os.path.join(OUT_DIR, "capstone_sar.tif")
    
    for path, arr in zip([dem_path, ndvi_path, sar_path], [dem, ndvi, sar_db]):
        with rasterio.open(path, 'w', **profile) as dst:
            dst.write(arr, 1)
            
    # CRITICAL: Use a local UTM projection for buffering, NOT EPSG:3857 (Web Mercator).
    # At latitude ~51°S, Web Mercator introduces ~40% distance distortion, making a
    # 1000m buffer actually ~700m on the ground.
    # EPSG:32719 = UTM Zone 19S (Torres del Paine, ~72°W, 51°S) — accurate to <0.1%.
    roads_metric = roads_gdf.to_crs("EPSG:32719")
    road_buffers = roads_metric.copy()
    road_buffers['geometry'] = road_buffers.geometry.buffer(1000)   # exact 1000m
    impact_zone_metric = road_buffers.dissolve()

    impact_zone = impact_zone_metric.to_crs(profile['crs'])
    impact_zone.to_file(os.path.join(OUT_DIR, "impact_zone.shp"))
    
    # Zonal Stats
    print("       Extracting stats from DEM, NDVI, and SAR...")
    stats_dem = zonal_stats(impact_zone, dem_path, stats=['mean', 'max'])[0]
    stats_ndvi = zonal_stats(impact_zone, ndvi_path, stats=['mean', 'min', 'max'])[0]
    stats_sar = zonal_stats(impact_zone, sar_path, stats=['mean'])[0]
    
    return stats_dem, stats_ndvi, stats_sar, road_buffers.to_crs(profile['crs'])

# ==========================================
# 5. Reporting & Visualization
# ==========================================
def generate_outputs(bbox, dem, ndvi, sar, buffers, stats_dem, stats_ndvi, stats_sar):
    print("\n[INFO] Generating Multi-Panel Chart and Markdown Report...")
    
    # 1. Multi-Panel Chart
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # DEM
    im0 = axes[0].imshow(dem, cmap='terrain')
    axes[0].set_title("Copernicus DEM (Topography)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    buffers.plot(ax=axes[0], facecolor='none', edgecolor='red', linewidth=1)
    
    # NDVI
    im1 = axes[1].imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
    axes[1].set_title("Sentinel-2 NDVI (Vegetation)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    buffers.plot(ax=axes[1], facecolor='none', edgecolor='black', linewidth=1)
    
    # SAR
    im2 = axes[2].imshow(sar, cmap='gray')
    axes[2].set_title("Sentinel-1 SAR (Structure/Water)")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    buffers.plot(ax=axes[2], facecolor='none', edgecolor='cyan', linewidth=1)
    
    for ax in axes:
        ax.axis('off')
        
    plot_path = os.path.join(OUT_DIR, "capstone_multi_panel.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)  # prevent memory leak
    
    # 2. Markdown Report
    report_path = os.path.join(BASE_DIR, "site_analysis_report.md")
    def _fmt(val, decimals=3, fallback='N/A'):
        """Format a zonal_stats value safely, guarding against None (all-NoData zone)."""
        return f'{val:.{decimals}f}' if val is not None else fallback

    content = f"""# Capstone Site Analysis Report

## 📍 Territory Overview
**Coordinates (BBOX):** `{bbox}`

This report was generated dynamically via the **GeoCascade CLI Pipeline**. It synthesizes data
across Optical, Radar, and Topographical satellite constellations, overlaid with OpenStreetMap
road networks.

---

## 📊 Analytical Results: Human Impact Zone (1km Buffer)
We buffered all local roads by 1000 meters (using UTM Zone 19S for accurate distance)
to analyze the environmental conditions directly exposed to human activity.

*   **Average Vegetation Health (NDVI):** `{_fmt(stats_ndvi.get('mean'))}` (Max: `{_fmt(stats_ndvi.get('max'))}`)
*   **Terrain Profile (Elevation):** `{_fmt(stats_dem.get('mean'), 1)}m` average, peaking at `{_fmt(stats_dem.get('max'), 1)}m`.
*   **Radar Backscatter (SAR VV):** `{_fmt(stats_sar.get('mean'), 1)} dB`
    *   *Interpretation:* High dB indicates rough terrain or urban structures. Very low dB (< -18) indicates standing water or smooth ice near the roads.

---

## 🗺️ GIS Data Deliverables
The following files are ready for ArcGIS Pro / ENVI in the `data/processed/` directory:
1.  `infrastructure_roads.shp` / `impact_zone.shp`
2.  `capstone_dem.tif`, `capstone_ndvi.tif`, `capstone_sar.tif`

*Report auto-generated by the GeoCascade Agentic Python Pipeline.*
"""
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(content)
        
    print(f"       [SUCCESS] Outputs saved to {OUT_DIR}")

def main():
    print("=======================================================")
    print(" GEOCASCADE PIPELINE - MULTI-SENSOR CAPSTONE CLI       ")
    print("=======================================================")
    
    args = parse_args()
    
    try:
        roads_gdf = fetch_infrastructure(args.bbox)
        dem, ndvi, sar_db, profile = fetch_and_fuse_rasters(args.bbox, args.date_range)
        stats_dem, stats_ndvi, stats_sar, buffers = export_and_analyze(roads_gdf, dem, ndvi, sar_db, profile)
        generate_outputs(args.bbox, dem, ndvi, sar_db, buffers, stats_dem, stats_ndvi, stats_sar)
        print("\n[SUCCESS] Capstone Execution Complete!")
        
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")

if __name__ == "__main__":
    main()
