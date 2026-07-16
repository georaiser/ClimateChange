# 🌊 Chapter 6: Isolines & Drainage Density

> **GeoCascade Pipeline — Stage 6**
> Isotherms, isohyets, Patagonian rain shadow modeling, and drainage density analysis.

---

## 📋 Overview

Chapter 6 focuses on two classic geospatial cartographic techniques:

1. **Isolines** (Script 16) — Extract temperature (isotherms) and precipitation (isohyets) contour lines from continuous raster surfaces. Patagonia's extreme east-west precipitation gradient (>3000 mm/yr west of Andes → <300 mm/yr east) makes isohyets one of the most informative layers in this pipeline.

2. **Drainage Density** (Script 17) — Quantify how densely a landscape is dissected by rivers: total stream length / basin area. High drainage density signals flashy flood-prone catchments; low density indicates permeable, vegetation-buffered basins.

> [!NOTE]
> Script 17 reads the Ch03 cached DEM (`temp_dem.tif`) to avoid redundant downloads. Run Chapter 3 first for fastest execution.

---

## 📁 Scripts

| # | File | Topic | Key Outputs |
|---|------|--------|-------------|
| 16 | `16_isohyets_isotherms.py` | Isotherms + Isohyets contour extraction | `isotherms.gpkg`, `isohyets.gpkg`, `temperature_surface.tif`, `precipitation_surface.tif`, 4-panel dark figure |
| 17 | `17_drainage_density.py` | Drainage density + river network vectors | `river_network.gpkg`, `river_network.shp`, `drainage_density_report.csv`, 4-panel dark figure |

---

## 🚀 Setup

```bash
conda activate geocascade_env

mamba install -n geocascade_env -c conda-forge \
    rasterio numpy matplotlib pandas geopandas shapely \
    scipy pystac-client planetary-computer pyproj -y

pip install pysheds
```

---

## ▶️ Running Scripts

```bash
# Run Chapter 3 first (caches DEM)
python Chapter_03/10_digital_elevation_processing.py

# Isotherms and isohyets (reads Ch01 temperature + CHIRPS if available)
python Chapter_06/16_isohyets_isotherms.py

# Drainage density (reads Ch03 DEM cache)
python Chapter_06/17_drainage_density.py
```

---

## 🔬 Methods Deep-Dive

### Script 16 — Isotherms & Isohyets

#### Temperature Surface

**Environmental Lapse Rate model:**
```
T(z) = T_base - Γ × z

where:
  T_base = 4.0°C  (Torres del Paine annual mean at sea level)
  Γ      = 6.5°C / 1000 m  (International Standard Atmosphere)
  z      = elevation from Copernicus DEM
```

This gives a physically realistic temperature surface without weather station data. When Ch01 RF temperature surface is available, it is used instead (more accurate: incorporates station observations).

#### Precipitation Surface (Patagonian Rain Shadow)

Patagonia has one of Earth's steepest precipitation gradients, caused by the Andes blocking westerly moisture:

| Location | Annual Precipitation |
|----------|---------------------|
| Puerto Natales (windward) | ~400 mm/yr |
| Torres del Paine west face | >3000 mm/yr |
| Argentinian Patagonia (lee) | 150–300 mm/yr |

**Model:**
```python
precip = P_west × exp(-dist_east / λ)

where:
  P_west    = 3000 mm/yr  (west-face maximum)
  dist_east = longitude - andes_divide  (degrees east of ridge)
  λ         = 0.3°        (e-folding distance of rainfall decay)
```

When Ch01 CHIRPS raster is available, it replaces this synthetic model.

#### Isoline Extraction

```python
# 1. Build coordinates for contour math
lon_grid, lat_grid = np.meshgrid(lons, lats)

# 2. Compute contours (matplotlib does the math, we throw away the figure)
cs = ax.contour(lon_grid, lat_grid, surface, levels=levels)
plt.close(fig)

# 3. Convert to LineString objects (OPEN lines, not closed polygons)
for level, segs in zip(cs.levels, cs.allsegs):
    for seg in segs:
        if len(seg) >= 2:
            lines.append(LineString(seg))
```

> [!IMPORTANT]
> Isotherms are **open** LineStrings. Using `path.to_polygons(closed_only=True)` (the default) forces closure, creating spurious line segments connecting the endpoints. Always use `closed_only=False`.

**GeoPackage vs Shapefile:**
Script 16 exports as `.gpkg` (GeoPackage) instead of `.shp` (Shapefile):
- Single file (vs 4+ files for shapefile)
- No 10-character column name limit
- Column names preserved exactly: `Temperature` not `TEMPERATUR`
- Fully supported in ArcGIS Pro, QGIS, and ENVI

---

### Script 17 — Drainage Density

**Formula:**
```
Dd = L / A

where:
  L = total river length (km)
  A = basin area (km²)
  Unit: km/km²
```

**Projection note:** Length and area must be computed in an equal-area metric CRS:
```python
# WRONG: compute length in EPSG:4326 (degrees)
gdf.geometry.length.sum()   # gives meaningless degree-units

# CORRECT: reproject to UTM Zone 18S (Southern Chile)
gdf.to_crs("EPSG:32718").geometry.length.sum() / 1000   # km
```

**Typical Dd values:**

| Environment | Dd (km/km²) |
|-------------|-------------|
| Badlands / volcanic | > 10 |
| Steep glaciated mountains | 2 – 5 |
| Temperate forest | 0.5 – 2 |
| Permeable limestone | < 0.5 |

**Threshold sensitivity analysis:**
Script 17 computes Dd for 5 thresholds (100, 250, 500, 1000, 2000 pixels) and plots Dd vs threshold. This is important because:
- Low threshold → many small tributaries → high apparent Dd (may include noise)
- High threshold → only main channels → low Dd (may miss real drainage structure)
- The plateau region of the Dd vs threshold curve is the most physically meaningful

**pysheds D8 conditioning steps (same as Script 11):**
1. `fill_pits` — remove single-cell depressions
2. `fill_depressions` — fill all closed basins
3. `resolve_flats` — add tiny gradient across flat areas
4. `flowdir` — D8 steepest-descent routing
5. `accumulation` — count upstream pixels
6. `extract_river_network` — vectorize pixels above threshold

---

## 📂 Output Directory Structure

```
Chapter_06/
└── data/
    └── processed/
        ├── isolines/
        │   ├── isotherms.gpkg               ← temperature contours (1°C interval)
        │   ├── isohyets.gpkg                ← precipitation contours (200 mm interval)
        │   ├── temperature_surface.tif
        │   ├── precipitation_surface.tif
        │   ├── isoline_statistics.csv
        │   └── isolines_map.png             ← 4-panel dark figure
        └── drainage/
            ├── river_network.gpkg           ← primary output (threshold=500 px)
            ├── river_network.shp            ← legacy format
            ├── drainage_density_report.csv  ← Dd at 5 thresholds
            └── drainage_density_map.png     ← 4-panel dark figure
```

---

## 🖥️ ArcGIS Pro Integration

### Isotherms / Isohyets (Script 16)

```
1. Add isotherms.gpkg as Feature Layer
   Symbology > Graduated Colors on Temperature field
   → Cold colors for low T, warm colors for high T

2. Validate with Surface > Contour tool:
   Spatial Analyst > Surface > Contour
   Input: temperature_surface.tif
   Contour Interval: 1
   → Should match isotherms.gpkg within 0.01°C (numeric precision)

3. Label isotherms:
   Labels > Enable Labels > Field: Temperature
   Expression: f"{$feature.Temperature:.1f}°C"
```

### Drainage Density (Script 17)

```
1. Add river_network.gpkg as Feature Layer
   Symbology > Single Symbol > Blue line, width 1pt

2. Stream Order (Strahler method):
   Spatial Analyst > Hydrology > Stream Order
   Input stream raster: flow_accumulation.tif > 500
   Input flow direction: flow_direction.tif
   → Assigns order 1 to headwaters, highest order to main stem

3. Calculate drainage density manually:
   Add Field "length_km" to river_network.gpkg
   Calculate Geometry: length_km = Shape_Length / 1000  (if in metres)
   Sum all lengths, divide by BBOX area in km²

4. Per-watershed Dd (with Ch03 watershed polygons):
   Spatial Join: river_network.gpkg → watershed_polygons
   Summary Statistics: SUM(length_km) per watershed
   Calculate: Dd = length_km_sum / area_km2
```

---

## 🔵 ENVI 5.6 Integration

```
; Temperature surface from DEM
File > Open > copernicus_dem.tif
Band Math: 4.0 - (b1 * 0.0065)
→ equivalent to temperature_surface.tif

; Isoline from temperature surface
File > Open > temperature_surface.tif
Toolbox > Topographic > Contour Lines
  Contour Interval: 1.0 (degrees C)
  → exports shapefile matching isotherms.gpkg
```

---

## ⚠️ Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| `pysheds` not found | Not pip-installed | `pip install pysheds` |
| No isoline segments extracted | Surface range too narrow (flat terrain) | Check `levels` array vs actual min/max of surface |
| `np.in1d` AttributeError | NumPy 2.0 compatibility | Already patched at top of script 17 |
| Drainage density = 0 | `extract_river_network` returned empty | Try lower threshold (e.g., 100 px); check DEM has valid values |
| GeoPackage opens as empty in ArcGIS | CRS not recognized | Script writes EPSG:4326; ArcGIS should auto-detect; try "Define Projection" if not |
| Isohyets all west of study area | BBOX wrong | Verify BBOX covers the Andes front (`lon ~-73.3`) |

---

## 📖 Key References

- Horton, R.E. (1932). *Drainage basin characteristics.* Transactions of the AGU.
- Strahler, A.N. (1957). *Quantitative analysis of watershed geomorphology.* Transactions of the AGU.
- Garreaud, R.D. et al. (2013). *Large-scale control on the Patagonian climate.* Journal of Climate.
- ISO 19125: *Geographic information — Simple feature access.* (GeoPackage specification)
- pysheds documentation: [mattbartos.com/pysheds](https://mattbartos.com/pysheds/)