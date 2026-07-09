"""
Chapter 6: 17_drainage_density.py

Academic Objective:
Drainage Density is a fundamental hydro-morphometric parameter. It is defined as the 
total length of all streams and rivers in a drainage basin divided by the total area 
of the drainage basin.
A high drainage density indicates an impermeable subsurface material, sparse vegetation, 
and mountainous relief, which leads to rapid runoff and higher flood risk.

In this script, we will:
1. Re-calculate the river network using pysheds (from Chapter 3).
2. Extract the rivers as Vector LineStrings.
3. Calculate the total river length and divide by the BBOX area to get the density.

Dependencies:
mamba install -n geocascade_env -c conda-forge pysheds geopandas shapely rasterio pystac-client planetary-computer pyproj matplotlib -y
"""

import os
import rasterio
from rasterio.windows import from_bounds
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import LineString, shape
from pysheds.grid import Grid
from pystac_client import Client
import planetary_computer as pc
from pyproj import Transformer
import json

# Monkey-patch pysheds for numpy 2.0 compatibility
if not hasattr(np, 'in1d'):
    np.in1d = np.isin

# ==========================================
# 1. Configuration
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

BBOX = [-73.30, -51.10, -72.90, -50.80]

def process_drainage_density():
    print("\n[INFO] Fetching DEM from Planetary Computer...")
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
    search_dem = catalog.search(collections=["cop-dem-glo-30"], bbox=BBOX)
    item_dem = list(search_dem.items())[0]
    
    temp_dem_path = os.path.join(OUT_DIR, "temp_dem.tif")
    
    with rasterio.open(item_dem.assets["data"].href) as src:
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        minx, miny = transformer.transform(BBOX[0], BBOX[1])
        maxx, maxy = transformer.transform(BBOX[2], BBOX[3])
        window = from_bounds(minx, miny, maxx, maxy, src.transform)
        transform = rasterio.windows.transform(window, src.transform)
        
        dem = src.read(1, window=window).astype('float32')
        dem = np.where(dem < 0, np.nan, dem)
        profile = src.profile
        profile.update(dtype=rasterio.float32, count=1, nodata=np.nan, height=window.height, width=window.width, transform=transform)
        
        with rasterio.open(temp_dem_path, 'w', **profile) as dst:
            dst.write(dem, 1)

    print("\n[INFO] Running ArcHydro Routing Simulation (pysheds)...")
    grid = Grid.from_raster(temp_dem_path)
    dem_grid = grid.read_raster(temp_dem_path)
    
    # Fill sinks and resolve flats
    print("       Filling sinks and resolving flats...")
    dem_filled = grid.fill_pits(dem_grid)
    dem_filled = grid.resolve_flats(dem_filled)
    
    # D8 Routing
    print("       Calculating Flow Direction...")
    fdir = grid.flowdir(dem_filled)
    
    # Accumulation
    print("       Calculating Flow Accumulation...")
    acc = grid.accumulation(fdir)

    # Extract River Network Vector
    print("\n[INFO] Extracting River Network Vectors...")
    # Create a mask for rivers (e.g., accumulation > 1000 pixels)
    threshold = 1000
    river_branches = grid.extract_river_network(fdir, acc > threshold)
    
    # Convert GeoJSON-like features to GeoPandas
    geom_list = []
    for feature in river_branches['features']:
        geom = shape(feature['geometry'])
        geom_list.append(geom)
        
    gdf_rivers = gpd.GeoDataFrame({'geometry': geom_list}, crs=src.crs)
    
    shp_path = os.path.join(OUT_DIR, "river_network.shp")
    gdf_rivers.to_file(shp_path)
    print(f"       [SUCCESS] Exported River Network Vector: {shp_path}")

    print("\n[INFO] Calculating Drainage Density...")
    # Reproject to an equal-area or metric projection to calculate accurate lengths and areas
    # EPSG:32718 is UTM Zone 18S (Southern Chile)
    gdf_rivers_metric = gdf_rivers.to_crs("EPSG:32718")
    
    total_river_length_m = gdf_rivers_metric.geometry.length.sum()
    
    # Calculate Basin Area
    # We will use the BBOX as the "basin" for this example
    from shapely.geometry import box
    bbox_geom = box(BBOX[0], BBOX[1], BBOX[2], BBOX[3])
    gdf_bbox = gpd.GeoDataFrame({'geometry': [bbox_geom]}, crs="EPSG:4326").to_crs("EPSG:32718")
    total_area_m2 = gdf_bbox.geometry.area.sum()
    
    drainage_density = total_river_length_m / total_area_m2
    
    print("--- DRAINAGE DENSITY REPORT ---")
    print(f"Total River Length: {total_river_length_m / 1000:.2f} km")
    print(f"Total Basin Area:   {total_area_m2 / 1_000_000:.2f} km²")
    print(f"Drainage Density:   {drainage_density * 1000:.4f} km/km²")
    print("-------------------------------")

    print("\n[INFO] Plotting...")
    fig, ax = plt.subplots(figsize=(10, 8))
    # Plot DEM background
    im = ax.imshow(dem_grid, cmap='terrain', extent=grid.extent)
    gdf_rivers.plot(ax=ax, color='blue', linewidth=1)
    
    ax.set_title(f"River Network & Drainage Density\nDensity: {drainage_density * 1000:.2f} km/km²")
    fig.colorbar(im, ax=ax, label="Elevation (m)")
    
    plot_path = os.path.join(OUT_DIR, "drainage_density_map.png")
    plt.savefig(plot_path, dpi=300)
    print(f"       [SUCCESS] Plot saved to: {plot_path}")

def main():
    print("=======================================================")
    print(" GEOCASCADE PIPELINE - DRAINAGE DENSITY                ")
    print("=======================================================")
    try:
        process_drainage_density()
        print("\n[SUCCESS] Script 17 Complete!")
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")

if __name__ == "__main__":
    main()
