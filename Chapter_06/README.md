# Chapter 6: Isotherms, Isohyets & Drainage Density

## Academic Objective
Generate temperature isolines (isotherms) from a DEM using the Environmental Lapse Rate,
and quantify watershed drainage density from pysheds D8 flow accumulation data.

---

## Scripts

### 16_isohyets_isotherms.py — Temperature Isolines from DEM

Applies the Environmental Lapse Rate to the Copernicus DEM to model temperature:
  T(z) = T0 - 6.5 * (z / 1000)   [Celsius, z in metres]

Then extracts isotherms as vector LineStrings using matplotlib contours + shapely.

> [!CAUTION]
> THREE critical bugs were fixed:
>
> 1. src.crs SCOPE: rasterio src.crs is only valid INSIDE the with-block.
>    Outside the block the file is closed and src.crs raises AttributeError.
>    Fix: capture as raster_crs = src.crs before exiting the block.
>
> 2. to_polygons(closed_only=False): matplotlib ContourSet paths are OPEN LineStrings.
>    to_polygons() without this flag silently drops all open contour segments.
>    Fix: use closed_only=False to retain all isoline geometry.
>
> 3. nodata=-9999 and int(round()) for window dimensions.

Run: `python 16_isohyets_isotherms.py`

Outputs: temperature_isotherms.gpkg, isotherms_map.png

---

### 17_drainage_density.py — Pysheds Drainage Density

Downloads DEM, computes D8 flow direction and accumulation, extracts river network
(FAcc > 1000 pixels), then calculates drainage density:

  Drainage Density D = L_total / A_basin   [km channel per km2 watershed]

High D = flashy runoff, impermeable bedrock.
Low D = permeable substrate, groundwater recharge dominant.

> [!CAUTION]
> Same src.crs scope fix applied. TEMP_DEM must be written with nodata=-9999
> so pysheds does not misinterpret fill boundaries during sink-filling.

**Fixes applied:** raster_crs captured inside with-block; TEMP_DEM nodata=-9999;
window int(round()); STAC guards; plt.close()

Run: `python 17_drainage_density.py`

Outputs: river_network.gpkg, drainage_density_report.txt

---

## Key Concepts

| Concept | Explanation |
|---|---|
| Environmental Lapse Rate | Temperature decreases ~6.5 degC per 1000m elevation gain |
| Isotherm | Line connecting points of equal temperature across the landscape |
| src.crs scope bug | Only valid inside rasterio with-block — always capture before close |
| ContourSet path type | Matplotlib contours are open LineStrings, not closed polygons |
| Drainage Density | L_total / A_basin — characterizes watershed hydraulic flashiness |

## Installation

```bash
mamba install -n geocascade_env -c conda-forge pystac-client planetary-computer rasterio pysheds geopandas pyproj matplotlib numpy -y
```