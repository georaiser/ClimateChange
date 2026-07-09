# Chapter 12: Automated Site Analysis Capstone

## 🎯 Academic Objective

This capstone synthesizes every technique from Chapters 1–8 into a single, production-quality **Command-Line Interface (CLI) pipeline**. Instead of running individual scripts, you supply a bounding box and date range and receive a complete environmental site assessment — automatically.

This mirrors the output delivered to clients in real Environmental Impact Assessment (EIA) consultancy projects.

By the end of this chapter you will be able to:
- Build a CLI tool with `argparse` for dynamic area-of-interest selection
- Chain multiple geospatial analysis steps in a single automated pipeline
- Create a road infrastructure impact zone and quantify its ecological effect via zonal statistics
- Generate a professional Markdown EIA-style report programmatically

---

## 🛠️ Scripts & Modules

### `capstone_pipeline.py`
A complete end-to-end geospatial site assessment pipeline. Accepts `--bbox` and `--date_range` CLI arguments for full flexibility.

**Analysis pipeline (8 automated steps):**

```
Step 1: Download Copernicus DEM → establishes master spatial grid
Step 2: Download Sentinel-2 → compute NDVI (vegetation health)
Step 3: Download Sentinel-1 SAR → compute backscatter in dB
Step 4: Download OpenStreetMap road network (osmnx)
Step 5: Create 1km road infrastructure impact zone (buffer)
Step 6: Run zonal statistics: NDVI, SAR, DEM inside vs outside buffer
Step 7: Generate 4-panel visualization
Step 8: Write professional Markdown impact report
```

---

## ⚠️ Critical Bug Patterns to Avoid

> [!CAUTION]
> **OSMnx Argument Order — Classic Student Error**
>
> `osmnx.graph_from_bbox()` uses **(north, south, east, west)** order — NOT (min_lon, min_lat, max_lon, max_lat). This is the opposite of BBOX convention everywhere else:
> ```python
> bbox = [-73.30, -51.10, -72.90, -50.80]   # [min_lon, min_lat, max_lon, max_lat]
>
> # ✅ CORRECT — explicitly name the lat/lon arguments
> G = ox.graph_from_bbox(
>     north=bbox[3],   # -50.80 (max lat)
>     south=bbox[1],   # -51.10 (min lat)
>     east=bbox[2],    # -72.90 (max lon)
>     west=bbox[0],    # -73.30 (min lon)
>     network_type='all'
> )
>
> # ❌ WRONG — silently downloads the wrong area
> G = ox.graph_from_bbox(bbox[0], bbox[1], bbox[2], bbox[3])
> # This passes (min_lon=-73.30) as north — an invalid latitude — osmnx clips it
> ```

> [!CAUTION]
> **EPSG:3857 (Web Mercator) — Never Use for Distance/Area at High Latitudes**
>
> Torres del Paine is at ~51°S. Web Mercator (EPSG:3857) introduces **~40% distance distortion** at this latitude. Always use a local UTM zone for buffering:
> ```python
> # ✅ CORRECT — UTM Zone 19S for Torres del Paine (~72°W, 51°S)
> roads_utm = roads_gdf.to_crs("EPSG:32719")
> impact_zone = roads_utm.buffer(1000)   # exact 1000m buffer
>
> # ❌ WRONG — 1000m in EPSG:3857 = ~700m actual distance at 51°S
> roads_webmercator = roads_gdf.to_crs("EPSG:3857")
> impact_zone = roads_webmercator.buffer(1000)
> ```

> [!WARNING]
> **Zonal Stats None Guard:** Always guard against `None` returns from `rasterstats.zonal_stats()` — zones entirely covered by NoData return `None` for all stats:
> ```python
> mean_ndvi = (stats.get('mean') or float('nan'))
> ```

---

## 🚀 How to Run

### Install Dependencies
```bash
mamba activate geocascade_env
mamba install -n geocascade_env -c conda-forge \
    pystac-client planetary-computer rasterio geopandas osmnx \
    rasterstats shapely numpy matplotlib pyproj scipy -y
```

### Default Run (Torres del Paine)
```bash
python capstone_pipeline.py
```

### Custom BBOX and Date Range
```bash
# Torres del Paine summer 2023
python capstone_pipeline.py \
  --bbox -73.30 -51.10 -72.90 -50.80 \
  --date_range 2023-01-01/2023-03-31

# Different region — Atacama Desert
python capstone_pipeline.py \
  --bbox -69.5 -23.5 -68.8 -22.8 \
  --date_range 2022-06-01/2022-08-31

# Different region — Patagonian Ice Field
python capstone_pipeline.py \
  --bbox -73.5 -50.5 -73.0 -50.0 \
  --date_range 2023-02-01/2023-02-28
```

---

## 📊 Sample Report Output

```markdown
# GeoCascade Environmental Impact Assessment
**Region:** [-73.30, -51.10, -72.90, -50.80]
**Date:** 2023-01-15
**Analysis Period:** 2023-01-01 to 2023-03-31

## Road Infrastructure Impact Zone (1 km buffer)

| Metric | Inside Buffer | Outside Buffer | Δ Change |
|--------|--------------|----------------|---------|
| Mean NDVI | 0.187 | 0.421 | **-55.6%** |
| Mean SAR (dB) | -12.3 | -15.7 | +3.4 dB |
| Mean Elevation (m) | 724 | 891 | -167 m |

## Key Findings
- Vegetation health is **55% lower** inside road corridors
- SAR values suggest higher surface roughness (disturbed terrain) within buffer
- Roads preferentially follow valley bottoms (lower elevation average)
```

---

## 📐 UTM Zone Reference for South America

| Region | UTM Zone | EPSG |
|--------|----------|------|
| Torres del Paine (~72°W) | 18S | 32718 |
| Patagonia central (~68°W) | 19S | 32719 |
| Atacama (~69°W) | 19S | 32719 |
| Buenos Aires (~58°W) | 21S | 32721 |

---

## 🗺️ GIS Interoperability

**ArcGIS Pro:** Load `site_assessment.png` and the individual TIFFs → add road buffers as a polygon layer → verify that the 1km buffer geometry aligns correctly with the OSM road centerlines.

**ENVI:** Load `sentinel2_ndvi.tif` and `sar_backscatter_db.tif` → use `Decision Tree` classification to reproduce the road impact zone classification → compare with our Python zonal stats.

> [!NOTE]
> This capstone demonstrates the full **EIA workflow** used by environmental consulting firms. The `capstone_report.md` output is formatted to be directly inserted into an academic report or submitted to an environmental regulatory authority with minimal editing.
