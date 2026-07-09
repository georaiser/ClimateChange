# Chapter 3: Topography and Glacial Retreat

## Academic Objective
Quantify 20-year glacier retreat using multi-temporal Landsat NDSI analysis, derive
terrain derivatives from the Copernicus DEM, and replicate the ArcHydro watershed
delineation workflow in pure Python (pysheds).

---

## Scripts

### 09_multitemporal_glacier_retreat.py

Downloads Landsat 8 C2-L2 for two epochs (2000-2005 vs 2019-2023) and computes NDSI.

NDSI = (Green - SWIR1) / (Green + SWIR1). Threshold >0.4 = glacier/snow.

Retreat map: ice_2003 - ice_2023. +1=melted, 0=stable, -1=advanced.

> [!WARNING]
> **Landsat C2-L2 Scale Factor is MANDATORY:** SR = DN x 0.0000275 - 0.2
> Without this, raw DNs (~7000-20000) produce NDSI near 1.0 for ALL pixels.

**Critical bug fixes:**
- green_2023 read was missing -- fixed (src.read() call added)
- profile and target_shape captured INSIDE the with-block
- nodata=-9999

**NEW Tier 3 -- Quantitative Area Report:**
Prints ice_lost, ice_stable, ice_gained in km2:
  Area_km2 = N_pixels x (pixel_resolution_m / 1000)^2

Run: python 09_multitemporal_glacier_retreat.py

---

### 10_digital_elevation_processing.py

Downloads Copernicus DEM GLO-30 (30m) and derives slope, aspect, hillshade, curvature.

> [!CAUTION]
> CopDEM is EPSG:4326 (geographic degrees). np.gradient() on degrees gives WRONG slope.
> Must scale: pix_lat_m = res x 111000, pix_lon_m = res x 111000 x cos(lat_rad)
> At 51 deg S: 1 deg lat = 111 km, 1 deg lon = ~69.8 km

Run: python 10_digital_elevation_processing.py

---

### 11_watershed_delineation.py

Replicates ArcHydro workflow: Fill Sinks > Flow Direction D8 > Flow Accumulation > Rivers.

Also computes Hipsometric Curve (elevation-area distribution):
- Concave = mature basin. Convex = actively eroding terrain.

**Fixes:** nodata=-9999 in TEMP_DEM; LogNorm crash guard; plt.close()

Run: python 11_watershed_delineation.py

---

## Key Concepts

| Concept | Explanation |
|---|---|
| NDSI threshold | >0.4 = glacier/snow. <0 = open water or soil |
| D8 algorithm | Water routes to steepest of 8 neighbors |
| Flow Accumulation | High FAcc = channel confluences |
| Hipsometric Curve | Elevation distribution reveals geomorphic maturity |
| CopDEM CRS | EPSG:4326 -- gradient must use metres-per-degree conversion |

## Installation

`ash
mamba install -n geocascade_env -c conda-forge pystac-client planetary-computer rasterio pysheds numpy matplotlib pyproj -y
`
