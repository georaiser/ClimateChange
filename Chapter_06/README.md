# Chapter 6: Advanced Hydrometeorology

## 🎯 Academic Objective

Temperature and precipitation are not uniform across a landscape — they vary with elevation, aspect, distance from the ocean, and atmospheric circulation. This chapter teaches you to model and visualize these spatial patterns using isoline maps (isolines of equal value) and quantify the drainage network through which water leaves the basin.

By the end of this chapter you will be able to:
- Generate isotherm maps (lines of equal temperature) from DEM-derived lapse rate surfaces
- Export isolines as GIS-ready Shapefiles for overlay analysis
- Calculate drainage density — a fundamental hydrological indicator of basin permeability
- Understand the relationship between terrain, climate, and water routing

---

## 🛠️ Scripts & Modules

### `16_isohyets_isotherms.py`
Generates isotherm maps by applying the **Environmental Lapse Rate** to a DEM, then extracts the isolines as vector geometry.

**Lapse rate temperature model:**
$$T(z) = T_0 - \Gamma \cdot z$$

Where:
- $T_0$ = Base temperature at sea level (≈ 15°C for Torres del Paine in January)
- $\Gamma$ = Environmental Lapse Rate = **6.5°C per 1000m** altitude
- $z$ = Elevation in meters (from Copernicus DEM)

**Isoline extraction pipeline:**
1. Compute temperature surface from DEM using lapse rate formula
2. Use `matplotlib.contourf()` to generate filled contour objects
3. Extract `matplotlib.path.Path` objects from contour collections
4. Convert each Path to a `shapely.geometry.LineString`
5. Assemble into GeoDataFrame and export as Shapefile

> [!CAUTION]
> **Two Critical Bugs to Avoid:**
>
> **Bug 1 — `src.crs` Scope Error:** The rasterio file handle `src` is only valid **inside** its `with` block. Accessing `src.crs` after the block closes is undefined behavior:
> ```python
> # ✅ CORRECT — capture CRS inside the context
> with rasterio.open(dem_path) as src:
>     dem = src.read(1, window=window)
>     raster_crs = src.crs      # capture here
>     raster_transform = src.transform
>
> gdf = gpd.GeoDataFrame(geometry=lines, crs=raster_crs)  # use captured value
>
> # ❌ WRONG — src is closed here; src.crs may raise or return stale value
> gdf = gpd.GeoDataFrame(geometry=lines, crs=src.crs)
> ```
>
> **Bug 2 — `path.to_polygons()` for Open Lines:** Isolines are **open** LineStrings, not closed polygons. Using `path.to_polygons()` forces closure, distorting endpoints:
> ```python
> # ✅ CORRECT — returns vertices without forced closure
> vertices = path.to_polygons(closed_only=False)
>
> # ❌ WRONG — closes open contour paths, creating artificial line segments
> vertices = path.to_polygons()
> ```

**Academic note:** This script currently computes **isotherms** (equal temperature). Isohyets (equal precipitation) require a different input — either ERA5 gridded precipitation or kriged point observations. Adding isohyets is left as an extension exercise.

---

### `17_drainage_density.py`
Computes **Drainage Density** ($D_d$), a fundamental geomorphic indicator that quantifies how much river network exists per unit area.

$$D_d = \frac{\Sigma L}{A}$$

Where:
- $\Sigma L$ = Total length of all stream segments (km)
- $A$ = Basin area (km²)

**Physical interpretation:**
| $D_d$ (km/km²) | Landscape Type | Implication |
|----------------|---------------|-------------|
| < 2 | Arid / Permeable | Water infiltrates; few surface streams |
| 2–5 | Temperate humid | Typical forested watershed |
| 5–10 | Impermeable / High rainfall | Dense network; rapid runoff |
| > 10 | Badlands / Clay-rich | Very dense; extreme runoff |

**River network extraction:**
```
Accumulation Threshold = 1000 pixels
At 30m DEM resolution: 1000 × (30m)² = 900,000 m² = 0.9 km² contributing area
```
Rivers are defined as cells where more than 0.9 km² of upstream area drains through them.

> [!CAUTION]
> **Same `src.crs` scope bug applies here.** Capture `raster_crs = src.crs` inside the `with rasterio.open()` block and use it after the block closes.

**Projection for length measurement:**
```python
# EPSG:32718 = UTM Zone 18S — appropriate for Torres del Paine (~72°W, 51°S)
# NEVER use EPSG:3857 (Web Mercator) for length/area at high latitudes
gdf_rivers_metric = gdf_rivers.to_crs("EPSG:32718")
total_length_km = gdf_rivers_metric.geometry.length.sum() / 1000
```

> [!WARNING]
> The river network accumulation threshold (`1000 pixels`) is a **user-defined parameter** that controls network density. It should be a named constant, not a magic number, and must be documented with its physical meaning (contributing area at the DEM's resolution).

---

## 📐 Key Formulas

| Concept | Formula |
|---------|---------|
| Temperature from elevation | $T(z) = T_0 - 0.0065 \cdot z$ |
| Drainage Density | $D_d = \Sigma L / A$ |
| Contributing area | $A_{contrib} = N_{pixels} \times pixel\_size^2$ |

---

## 🚀 How to Run

### Install Dependencies
```bash
mamba activate geocascade_env
mamba install -n geocascade_env -c conda-forge \
    pystac-client planetary-computer rasterio pysheds \
    geopandas shapely numpy matplotlib pyproj -y
```

### Execute Scripts
```bash
# Generate isotherm map + Shapefile export
python 16_isohyets_isotherms.py

# Compute drainage density + river network extraction
python 17_drainage_density.py
```

---

## 🗺️ GIS Interoperability

**ArcGIS Pro:** Load `isotherms.shp` → symbolize by temperature value → overlay on DEM hillshade → add labels for each isotherm value. This produces a professional-quality climate map identical to those in IPCC reports.

**ENVI:** Load `dem.tif` → use `Band Math` to apply the lapse rate formula directly → use `Contour Lines` tool to extract isolines as vectors.

> [!TIP]
> Drainage density is a core input to the **SCS Curve Number** runoff model used by hydrologists worldwide. A high $D_d$ in the Torres del Paine watershed signals rapid glacial meltwater routing — directly relevant to flood risk modeling in Chapter 8.
