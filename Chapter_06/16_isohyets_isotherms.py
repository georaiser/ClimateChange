"""
Chapter 6: 16_isohyets_isotherms.py

Academic Objective:
Meteorological variables are continuous surfaces. To visualize them on a map, we use 
isolines: lines connecting points of equal value.
- Isohyets: Lines of equal precipitation.
- Isotherms: Lines of equal temperature.

In this script, we will:
1. Fetch the High-Resolution DEM.
2. Apply the Environmental Lapse Rate (-6.5°C per 1000m) to simulate a high-resolution 
   temperature surface based on elevation.
3. Use contouring algorithms to extract Isotherms and export them as a Vector Shapefile.

Dependencies:
mamba install -n geocascade_env -c conda-forge geopandas shapely rasterio pyproj pystac-client planetary-computer matplotlib numpy -y
"""

import os
import rasterio
from rasterio.windows import from_bounds
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import LineString
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

def process_isotherms():
    print("\n[INFO] Fetching DEM from Planetary Computer...")
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
    
    search_dem = catalog.search(collections=["cop-dem-glo-30"], bbox=BBOX)
    items_dem = list(search_dem.items())
    if not items_dem:
        raise RuntimeError("No DEM tiles found for BBOX. Check coordinates or Planetary Computer access.")
    item_dem = items_dem[0]
    
    with rasterio.open(item_dem.assets["data"].href) as src:
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        minx, miny = transformer.transform(BBOX[0], BBOX[1])
        maxx, maxy = transformer.transform(BBOX[2], BBOX[3])
        window = from_bounds(minx, miny, maxx, maxy, src.transform)
        transform = rasterio.windows.transform(window, src.transform)
        
        dem = src.read(1, window=window).astype('float32')
        dem = np.where(dem < 0, np.nan, dem)
        raster_crs = src.crs        # CRITICAL: capture CRS inside the with-block
        raster_transform = rasterio.windows.transform(window, src.transform)
        profile = src.profile
        profile.update(
            dtype=rasterio.float32, count=1, nodata=-9999,
            height=int(round(window.height)), width=int(round(window.width)),
            transform=raster_transform
        )

    print("\n[INFO] Applying Environmental Lapse Rate (-6.5°C / 1000m)...")
    # Assume base sea level temp in summer is 15°C
    base_temp = 15.0
    lapse_rate = 6.5 / 1000.0
    temperature = base_temp - (dem * lapse_rate)
    
    # Save temperature raster
    temp_tif = os.path.join(OUT_DIR, "simulated_temperature.tif")
    with rasterio.open(temp_tif, 'w', **profile) as dst:
        dst.write(temperature, 1)
    print(f"       [SUCCESS] Exported Temperature Raster: {temp_tif}")

    print("\n[INFO] Extracting Isotherms (Contour Lines)...")
    # We will use matplotlib to calculate the contour lines mathematically
    # First, generate the X and Y coordinates for every pixel
    h, w = temperature.shape
    cols, rows = np.meshgrid(np.arange(w), np.arange(h))
    xs, ys = rasterio.transform.xy(profile['transform'], rows, cols)
    xs = np.array(xs).reshape(h, w)
    ys = np.array(ys).reshape(h, w)
    
    fig, ax = plt.subplots()
    # Create contours every 2 degrees
    levels = np.arange(np.nanmin(temperature), np.nanmax(temperature), 2)
    cs = ax.contour(xs, ys, temperature, levels=levels)
    
    # Convert matplotlib contours to GeoPandas LineStrings
    lines = []
    temps = []
    
    if hasattr(cs, 'collections'):
        for level, collection in zip(cs.levels, cs.collections):
            for path in collection.get_paths():
                # CRITICAL: use closed_only=False — isotherms are OPEN LineStrings,
                # not closed polygons. to_polygons() without this argument forces
                # closure, creating spurious line segments connecting endpoints.
                for v in path.to_polygons(closed_only=False):
                    if len(v) >= 2:
                        lines.append(LineString(v))
                        temps.append(level)
    else:
        for level, segs in zip(cs.levels, cs.allsegs):
            for seg in segs:
                if len(seg) >= 2:
                    lines.append(LineString(seg))
                    temps.append(level)
                    
    plt.close(fig) # We don't need this figure, just the math
    
    # CRITICAL: use raster_crs captured inside the with-block (above).
    # Accessing src.crs here (after the with-block closed) is undefined behaviour.
    if not lines:
        print("       [WARNING] No isoline segments found. Check temperature range.")
    gdf = gpd.GeoDataFrame({'Temperature': temps, 'geometry': lines}, crs=raster_crs)
    
    shp_path = os.path.join(OUT_DIR, "isotherms.shp")
    gdf.to_file(shp_path)
    print(f"       [SUCCESS] Exported Isotherms Shapefile: {shp_path}")

    print("\n[INFO] Plotting Results...")
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(temperature, cmap='coolwarm', extent=[xs.min(), xs.max(), ys.min(), ys.max()])
    gdf.plot(ax=ax, color='black', linewidth=0.5, alpha=0.5)
    
    ax.set_title("Simulated Temperature & Isotherms")
    fig.colorbar(im, ax=ax, label="Temperature (°C)")
    
    plot_path = os.path.join(OUT_DIR, "isotherms_map.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)  # prevent memory leak in multi-script pipelines
    print(f"       [SUCCESS] Plot saved to: {plot_path}")


def main():
    print("=======================================================")
    print(" GEOCASCADE PIPELINE - ISOTHERMS & ISOHYETS            ")
    print("=======================================================")
    try:
        process_isotherms()
        print("\n[SUCCESS] Script 16 Complete!")
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")

if __name__ == "__main__":
    main()
