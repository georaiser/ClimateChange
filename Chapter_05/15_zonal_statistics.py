"""
Chapter 5: 15_zonal_statistics.py

Academic Objective:
Calculating raw pixels is visually pleasing, but environmental managers need concrete numbers 
(e.g., "What is the average vegetation health inside Management Zone A?"). 
This is called Zonal Statistics ("Áreas Poligonales").

In this script, we will:
1. Programmatically generate 4 vector polygons (simulating Management Zones).
2. Fetch the High-Resolution Copernicus DEM and Sentinel-2 NDVI.
3. Calculate the exact Mean Elevation and Mean Vegetation Health (NDVI) within each polygon.

Dependencies:
mamba install -n geocascade_env -c conda-forge geopandas rasterstats shapely rasterio pyproj pystac-client planetary-computer -y
"""

import os
import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import box
from rasterstats import zonal_stats
from pystac_client import Client
import planetary_computer as pc
from pyproj import Transformer

# ==========================================
# 1. Configuration
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

BBOX = [-73.30, -51.10, -72.90, -50.80]

def create_management_zones():
    print("\n[INFO] Generating Vector Management Zones (Polygons)...")
    # Split the BBOX into 4 quadrants (Zones A, B, C, D)
    mid_lon = (BBOX[0] + BBOX[2]) / 2
    mid_lat = (BBOX[1] + BBOX[3]) / 2
    
    zone_a = box(BBOX[0], mid_lat, mid_lon, BBOX[3]) # NW
    zone_b = box(mid_lon, mid_lat, BBOX[2], BBOX[3]) # NE
    zone_c = box(BBOX[0], BBOX[1], mid_lon, mid_lat) # SW
    zone_d = box(mid_lon, BBOX[1], BBOX[2], mid_lat) # SE
    
    gdf = gpd.GeoDataFrame({
        'Zone': ['NW Sector', 'NE Sector', 'SW Sector', 'SE Sector'],
        'geometry': [zone_a, zone_b, zone_c, zone_d]
    }, crs="EPSG:4326")
    
    # Save shapefile for GIS
    out_shp = os.path.join(OUT_DIR, "management_zones.shp")
    gdf.to_file(out_shp)
    print(f"       [SUCCESS] Exported Shapefile: {out_shp}")
    
    return gdf

def fetch_data():
    print("\n[INFO] Fetching Raster Data from Planetary Computer...")
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
    
    # 1. Fetch DEM
    search_dem = catalog.search(collections=["cop-dem-glo-30"], bbox=BBOX)
    items_dem = list(search_dem.items())
    if not items_dem:
        raise RuntimeError("No DEM found for BBOX. Check coordinates or Planetary Computer access.")
    item_dem = items_dem[0]
    
    # We will save the DEM and NDVI to local temporary TIFFs so rasterstats can read them efficiently
    dem_path = os.path.join(OUT_DIR, "temp_dem.tif")
    ndvi_path = os.path.join(OUT_DIR, "temp_ndvi.tif")
    
    print("       Downloading DEM...")
    with rasterio.open(item_dem.assets["data"].href) as src:
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        minx, miny = transformer.transform(BBOX[0], BBOX[1])
        maxx, maxy = transformer.transform(BBOX[2], BBOX[3])
        window = from_bounds(minx, miny, maxx, maxy, src.transform)
        transform = rasterio.windows.transform(window, src.transform)
        
        dem = src.read(1, window=window).astype('float32')
        dem = np.where(dem < 0, np.nan, dem)
        
        profile_dem = src.profile
        profile_dem.update(
            dtype=rasterio.float32, count=1, nodata=-9999,
            height=int(round(window.height)), width=int(round(window.width)),
            transform=transform
        )

        with rasterio.open(dem_path, 'w', **profile_dem) as dst:
            dst.write(np.nan_to_num(dem, nan=-9999), 1)

    print("       Downloading Sentinel-2 NDVI...")
    search_s2 = catalog.search(
        collections=["sentinel-2-l2a"], bbox=BBOX,
        datetime="2023-01-01/2023-02-28", query={"eo:cloud_cover": {"lt": 5}}
    )
    items_s2 = list(search_s2.items())
    if not items_s2:
        raise RuntimeError("No Sentinel-2 found. Relax cloud cover filter or adjust date range.")
    item_s2 = sorted(items_s2, key=lambda i: i.properties["eo:cloud_cover"])[0]
    
    with rasterio.open(item_s2.assets["B04"].href) as src_red, rasterio.open(item_s2.assets["B08"].href) as src_nir:
        # Align to DEM
        transformer_s2 = Transformer.from_crs("EPSG:4326", src_red.crs, always_xy=True)
        minx_s2, miny_s2 = transformer_s2.transform(BBOX[0], BBOX[1])
        maxx_s2, maxy_s2 = transformer_s2.transform(BBOX[2], BBOX[3])
        win_s2 = from_bounds(minx_s2, miny_s2, maxx_s2, maxy_s2, src_red.transform)
        
        red = src_red.read(1, out_shape=dem.shape, window=win_s2, resampling=Resampling.bilinear).astype('float32') / 10000.0
        nir = src_nir.read(1, out_shape=dem.shape, window=win_s2, resampling=Resampling.bilinear).astype('float32') / 10000.0
        
        ndvi = np.where((nir + red) == 0, np.nan, (nir - red) / (nir + red))
        
        profile_ndvi = profile_dem.copy()
        with rasterio.open(ndvi_path, 'w', **profile_ndvi) as dst:
            dst.write(ndvi, 1)

    return dem_path, ndvi_path, dem, ndvi

def run_zonal_statistics(gdf, dem_path, ndvi_path):
    print("\n[INFO] Calculating Zonal Statistics (Áreas Poligonales)...")
    
    # Rasterstats requires the polygons to be in the same projection as the raster
    with rasterio.open(dem_path) as src:
        raster_crs = src.crs
        
    gdf_proj = gdf.to_crs(raster_crs)
    
    # Calculate stats — include std/min/max for richer analysis (Tier 3 improvement)
    stats_dem  = zonal_stats(gdf_proj, dem_path,  stats="mean max min std", nodata=-9999)
    stats_ndvi = zonal_stats(gdf_proj, ndvi_path, stats="mean min max std", nodata=-9999)
    
    print("\n--- ZONAL STATISTICS REPORT ---")
    for i, row in gdf_proj.iterrows():
        zone_name = row['Zone']
        # Guard against None (zone entirely covered by NoData)
        def _s(v, fmt='.1f'): return format(v, fmt) if v is not None else 'N/A'
        print(f"Zone: {zone_name}")
        print(f"  - Elevation : mean={_s(stats_dem[i].get('mean'))} m  "
              f"max={_s(stats_dem[i].get('max'))} m  "
              f"std=±{_s(stats_dem[i].get('std'))} m")
        print(f"  - NDVI      : mean={_s(stats_ndvi[i].get('mean'), '.3f')}  "
              f"min={_s(stats_ndvi[i].get('min'), '.3f')}  "
              f"max={_s(stats_ndvi[i].get('max'), '.3f')}  "
              f"std=±{_s(stats_ndvi[i].get('std'), '.3f')}")
        print("-" * 45)

    print("\n[INFO] Plotting Management Zones...")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    with rasterio.open(ndvi_path) as src:
        from rasterio.plot import show
        show(src, ax=ax, cmap='YlGn', title='Management Zones & NDVI')
    
    gdf_proj.boundary.plot(ax=ax, color='red', linewidth=2)
    for idx, row in gdf_proj.iterrows():
        plt.annotate(text=row['Zone'], xy=(row.geometry.centroid.x, row.geometry.centroid.y),
                     horizontalalignment='center', color='red', weight='bold', fontsize=12)
                     
    plot_path = os.path.join(OUT_DIR, "zonal_statistics.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)  # prevent memory leak
    print(f"       [SUCCESS] Plot saved to: {plot_path}")

def main():
    print("=======================================================")
    print(" GEOCASCADE PIPELINE - ZONAL STATISTICS (POLYGONS)     ")
    print("=======================================================")
    try:
        gdf = create_management_zones()
        dem_path, ndvi_path, dem, ndvi = fetch_data()
        run_zonal_statistics(gdf, dem_path, ndvi_path)
        print("\n[SUCCESS] Script 15 Complete!")
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")

if __name__ == "__main__":
    main()
