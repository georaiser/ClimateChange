"""
Chapter 12: Robust Site Analysis Capstone
This consolidated script represents a full-scale real-world GIS project.
It automatically sources:
1. Vector Data (Real-world Roads) via Overpass API / GeoPandas.
2. Raster Data (Copernicus DEM & Sentinel-2 L2A) via STAC.
It then performs Vector-Raster Overlay analysis (Zonal Statistics within buffers)
and generates an automated Site Analysis Report in Markdown.
"""

import os
import requests
import rasterio
from rasterio.windows import from_bounds
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import box
from pystac_client import Client
import planetary_computer as pc
from pyproj import Transformer
from rasterstats import zonal_stats

# ==========================================
# 1. Configuration (STUDENTS: CHANGE YOUR BBOX HERE)
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

# Format: [min_lon, min_lat, max_lon, max_lat]
# Default: A mountainous section near Puerto Natales, Patagonia
BBOX = [-72.8, -51.8, -72.4, -51.6]

# ==========================================
# 2. Vector Sourcing (OpenStreetMap via Overpass API)
# ==========================================
def fetch_infrastructure_vector(bbox):
    print("\n[INFO] Fetching Real-World Infrastructure (Roads) from OpenStreetMap...")
    import osmnx as ox
    try:
        # v2.x syntax
        graph = ox.graph_from_bbox(bbox=(bbox[0], bbox[1], bbox[2], bbox[3]), network_type='all')
    except TypeError:
        # v1.x syntax
        graph = ox.graph_from_bbox(bbox[3], bbox[1], bbox[2], bbox[0], network_type='all')
        
    # Convert graph to GeoDataFrames
    nodes, edges = ox.graph_to_gdfs(graph)
    
    # Export as Shapefile
    roads_shp = os.path.join(OUT_DIR, "infrastructure_roads.shp")
    
    # Clean lists/tuples from columns for Shapefile compatibility
    for col in edges.columns:
        if edges[col].apply(lambda x: isinstance(x, (list, tuple))).any():
            edges[col] = edges[col].astype(str)
            
    edges.to_file(roads_shp)
    print(f"       [SUCCESS] Downloaded {len(edges)} road segments to {roads_shp}")
    return edges, roads_shp

# ==========================================
# 3. Raster Sourcing (STAC API)
# ==========================================
def fetch_rasters(bbox):
    print("\n[INFO] Fetching Copernicus DEM and Sentinel-2 L2A from STAC...")
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
    
    # 1. Fetch DEM
    search_dem = catalog.search(collections=["cop-dem-glo-30"], bbox=bbox)
    item_dem = list(search_dem.items())[0]
    
    dem_path = os.path.join(OUT_DIR, "capstone_dem.tif")
    with rasterio.open(item_dem.assets["data"].href) as src:
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        minx, miny = transformer.transform(bbox[0], bbox[1])
        maxx, maxy = transformer.transform(bbox[2], bbox[3])
        window = from_bounds(minx, miny, maxx, maxy, src.transform)
        transform = rasterio.windows.transform(window, src.transform)
        
        dem = src.read(1, window=window).astype('float32')
        dem = np.where(dem < 0, np.nan, dem)
        profile = src.profile
        profile.update(dtype=rasterio.float32, count=1, nodata=np.nan, 
                       height=int(window.height), width=int(window.width), transform=transform)
        
        with rasterio.open(dem_path, 'w', **profile) as dst:
            dst.write(dem, 1)
            
    # 2. Fetch Sentinel-2 (NDVI)
    search_s2 = catalog.search(collections=["sentinel-2-l2a"], bbox=bbox, 
                               datetime="2022-01-01/2023-12-31", query={"eo:cloud_cover": {"lt": 20}})
    items_s2 = list(search_s2.items())
    if not items_s2:
        raise Exception("No Sentinel-2 imagery found for this BBOX with <20% cloud cover. Try changing the BBOX or the Date Range.")
    item_s2 = items_s2[0]
    
    ndvi_path = os.path.join(OUT_DIR, "capstone_ndvi.tif")
    with rasterio.open(item_s2.assets["B04"].href) as src_red, rasterio.open(item_s2.assets["B08"].href) as src_nir:
        transformer_s2 = Transformer.from_crs("EPSG:4326", src_red.crs, always_xy=True)
        minx_s2, miny_s2 = transformer_s2.transform(bbox[0], bbox[1])
        maxx_s2, maxy_s2 = transformer_s2.transform(bbox[2], bbox[3])
        win_s2 = from_bounds(minx_s2, miny_s2, maxx_s2, maxy_s2, src_red.transform)
        
        red = src_red.read(1, out_shape=dem.shape, window=win_s2, resampling=rasterio.enums.Resampling.bilinear).astype('float32')
        nir = src_nir.read(1, out_shape=dem.shape, window=win_s2, resampling=rasterio.enums.Resampling.bilinear).astype('float32')
        
        ndvi = np.where((nir + red) == 0, np.nan, (nir - red) / (nir + red))
        
        profile_ndvi = src_red.profile
        # Align NDVI exactly to the DEM's CRS and Transform so they overlay perfectly
        profile_ndvi.update(dtype=rasterio.float32, count=1, nodata=np.nan, 
                            height=dem.shape[0], width=dem.shape[1], 
                            crs=profile['crs'], transform=profile['transform'])
        
        with rasterio.open(ndvi_path, 'w', **profile_ndvi) as dst:
            dst.write(ndvi, 1)
            
    print("       [SUCCESS] Exported Raster layers (DEM & NDVI).")
    return dem_path, ndvi_path, dem, ndvi, profile

# ==========================================
# 4. Advanced Vector-Raster GIS Overlay
# ==========================================
def run_gis_overlay(roads_gdf, ndvi_path, dem_path):
    print("\n[INFO] Running Advanced Vector-Raster Overlay Analysis...")
    
    # 1. Reproject Roads to match a Metric CRS for accurate buffering in meters
    roads_metric = roads_gdf.to_crs("EPSG:3857")
    
    # 2. Buffer the roads by 1000 meters (1km Impact Zone)
    print("       Buffering infrastructure by 1000m...")
    road_buffers = roads_metric.copy()
    road_buffers['geometry'] = road_buffers.geometry.buffer(1000)
    
    # Dissolve overlapping buffers into a single massive impact zone
    impact_zone_metric = road_buffers.dissolve()
    
    # 3. Reproject back to the Raster CRS for exact Zonal Statistics overlay
    with rasterio.open(ndvi_path) as src:
        raster_crs = src.crs
    impact_zone = impact_zone_metric.to_crs(raster_crs)
    road_buffers = road_buffers.to_crs(raster_crs)
    impact_zone_path = os.path.join(OUT_DIR, "infrastructure_impact_zone.shp")
    impact_zone.to_file(impact_zone_path)
    
    # 3. Zonal Statistics (What is the NDVI and Elevation strictly inside the 1km human impact zone?)
    print("       Extracting Zonal Statistics within impact zone...")
    ndvi_stats = zonal_stats(impact_zone, ndvi_path, stats=['mean', 'min', 'max'])
    dem_stats = zonal_stats(impact_zone, dem_path, stats=['mean', 'max'])
    
    return ndvi_stats[0], dem_stats[0], impact_zone_path, road_buffers

# ==========================================
# 5. Automated Reporting
# ==========================================
def generate_report(ndvi_stats, dem_stats, bbox):
    print("\n[INFO] Generating Automated Site Analysis Report...")
    report_path = os.path.join(BASE_DIR, "site_analysis_report.md")
    
    content = f"""# Capstone Site Analysis Report
    
## 📍 Territory Overview
**Coordinates (BBOX):** {bbox}

This report was generated automatically by the Python Cloud-Native Geospatial Pipeline. Real-world vector infrastructure was sourced dynamically from OpenStreetMap and overlaid onto Sentinel-2 and Copernicus DEM rasters.

---

## 📊 Analytical Results: Human Impact Zone (1km Buffer)

We buffered all local roads and infrastructure by 1000 meters to analyze the environmental conditions directly exposed to human activity.

*   **Average Vegetation Health (NDVI) in Impact Zone:** `{ndvi_stats['mean']:.3f}`
    *   *Minimum NDVI:* `{ndvi_stats['min']:.3f}` (Indicates paved roads, bare rock, or water)
    *   *Maximum NDVI:* `{ndvi_stats['max']:.3f}` (Indicates dense, healthy forest patches near roads)
*   **Terrain Profile of Impact Zone:**
    *   *Average Elevation:* `{dem_stats['mean']:.1f} meters`
    *   *Maximum Elevation Reached by Roads:* `{dem_stats['max']:.1f} meters`

---

## 🗺️ GIS Data Deliverables
The following files have been generated in the `data/processed/` directory and are ready for **ArcGIS Pro** or **ENVI**:
1.  `infrastructure_roads.shp` (Raw OSM Vector Network)
2.  `infrastructure_impact_zone.shp` (1km Dissolved Buffer)
3.  `capstone_dem.tif` (Digital Elevation Model)
4.  `capstone_ndvi.tif` (Sentinel-2 Vegetation Health)

**Recommended Actions:** Load the `.shp` buffers over the `ndvi.tif` in your GIS software to visually verify these statistical anomalies.
"""
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(content)
        
    print(f"       [SUCCESS] Report saved to {report_path}")

def main():
    print("=======================================================")
    print(" GEOCASCADE PIPELINE - ROBUST CAPSTONE PROJECT         ")
    print("=======================================================")
    try:
        roads_gdf, roads_shp = fetch_infrastructure_vector(BBOX)
        dem_path, ndvi_path, dem, ndvi, profile = fetch_rasters(BBOX)
        ndvi_stats, dem_stats, impact_zone_path, road_buffers = run_gis_overlay(roads_gdf, ndvi_path, dem_path)
        
        # Plotting the visualization
        fig, ax = plt.subplots(figsize=(10, 10))
        import rasterio.plot as rioplot
        with rasterio.open(ndvi_path) as src:
            rioplot.show(src, ax=ax, cmap='RdYlGn', title="Human Impact Zone (1km Buffer) overlay on NDVI")
        
        # Overlay the buffers
        road_buffers.plot(ax=ax, facecolor='none', edgecolor='blue', linewidth=2)
        plt.savefig(os.path.join(OUT_DIR, "capstone_overlay_map.png"), dpi=300)
        
        generate_report(ndvi_stats, dem_stats, BBOX)
        print("\n[SUCCESS] Capstone Pipeline Execution Complete!")
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")

if __name__ == "__main__":
    main()
