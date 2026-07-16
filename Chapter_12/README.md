# 🏔️ Chapter 12: Capstone CLI Pipeline — Automated Site Analysis

> **GeoCascade Pipeline — Capstone Stage**
> The culmination of Chapters 1–11: a production-grade command-line tool
> that accepts any geographic coordinates and generates a complete multi-sensor
> site analysis report automatically.

---

## 📋 Overview

| Script | Topic | Key Outputs |
|--------|-------|-------------|
| `capstone_pipeline.py` | CLI multi-sensor fusion + zonal stats + report | 3 GeoTIFFs, GeoPackage, 5-panel dashboard, Markdown report |

---

## 🚀 Setup

```bash
conda activate geocascade_env

mamba install -n geocascade_env -c conda-forge \
    rasterio geopandas rasterstats osmnx \
    pystac-client planetary-computer pyproj \
    numpy pandas matplotlib -y
```

---

## ▶️ Usage

```bash
# Default: Torres del Paine (Grey Glacier region)
python Chapter_12/capstone_pipeline.py

# Custom study area
python Chapter_12/capstone_pipeline.py \
    --bbox -72.8 -51.8 -72.4 -51.6 \
    --date_range 2023-01-01/2023-03-31 \
    --buffer_m 1000

# Any region on Earth
python Chapter_12/capstone_pipeline.py \
    --bbox -70.6 -33.5 -70.4 -33.3 \
    --date_range 2022-12-01/2023-02-28
```

---

## 🔬 Methods Deep-Dive

### Local Cache → STAC Fallback Priority Chain

The pipeline is designed to avoid redundant downloads. It searches for cached outputs from earlier chapters in a priority order:

```
NDVI: Ch02/data/processed/indices/ndvi.tif → Ch02/.../ndvi.tif
SAR:  Ch07/data/processed/sar/sar_vv_db.tif → Ch07/.../sar_vv_db.tif
DEM:  Ch03/data/processed/terrain/copernicus_dem.tif → Ch03/data/raw/temp_dem.tif
```

If none found → downloads from Planetary Computer STAC.

> [!IMPORTANT]
> Run Ch01→Ch02→Ch07→Ch08 first for fastest capstone execution.
> Each previous chapter's output becomes a cached input here.

---

### OSMnx Road Network — Correct API

```python
# CRITICAL: graph_from_bbox uses named kwargs (north, south, east, west)
# Positional call would pass bbox[0]=-73.30 as 'north' latitude → wrong continent
graph = ox.graph_from_bbox(
    north=bbox[3], south=bbox[1],
    east=bbox[2],  west=bbox[0],   # named kwargs required
    network_type="all"
)
```

> [!WARNING]
> A common error is calling `ox.graph_from_bbox(bbox[3], bbox[1], bbox[2], bbox[0])`.
> This silently queries the correct values but if the order is wrong it downloads
> a road network in a completely different location. Always use named kwargs.

---

### Why EPSG:32719 for Buffers (Not EPSG:3857)

```python
# Project to UTM Zone 19S BEFORE buffering
edges_utm = edges.to_crs("EPSG:32719")   # metres, accurate at 51°S
buffer    = edges_utm.buffer(500)         # exactly 500m radius

# WRONG: EPSG:3857 (Web Mercator) distorts ~40% at 51°S latitude
# A "500m" buffer in 3857 is actually 700m on the ground at this latitude
```

| CRS | Accuracy at 51°S | Units | Use for |
|-----|-----------------|-------|---------|
| EPSG:32719 (UTM 19S) | < 1% | Metres | Buffers, area calculation |
| EPSG:4326 (WGS84) | Degrees only | Degrees | BBOX queries, STAC searches |
| EPSG:3857 (Web Mercator) | ~40% distortion at 51°S | Metres (distorted) | Web tiles only |

---

### GeoPackage vs Shapefile

```python
# GeoPackage: modern, single-file, handles list columns, no 10-char limit
edges.to_file("roads_buffer.gpkg", layer="roads",  driver="GPKG")
buf.to_file(  "roads_buffer.gpkg", layer="buffer", driver="GPKG")

# Shapefile: legacy format, needs column name truncation, multiple files
# Required: convert list columns to str before saving
for col in edges.columns:
    if isinstance(edges[col].iloc[0], (list, tuple)):
        edges[col] = edges[col].astype(str)
```

---

### Zonal Statistics with MemoryFile

```python
# Uses rasterio.MemoryFile to avoid writing temp files to disk
from rasterstats import zonal_stats

with rasterio.MemoryFile() as memf:
    with memf.open(**profile) as dst:
        dst.write(arr, 1)
    with memf.open() as src:
        stats = zonal_stats(
            buffer_geodataframe,
            src.read(1),
            affine=src.transform,
            nodata=-9999,           # correct nodata — not np.nan
            stats=["mean", "std", "min", "max", "count"]
        )
```

---

## 📂 Output Structure

```
Chapter_12/
└── data/processed/capstone/
    ├── ndvi_capstone.tif              ← NDVI surface reflectance (S2 or cached Ch02)
    ├── sar_vv_db_capstone.tif         ← SAR VV backscatter dB (S1 or cached Ch07)
    ├── dem_capstone.tif               ← Copernicus DEM elevation (or cached Ch03)
    ├── roads_buffer.gpkg
    │   ├── layer: roads               ← OSMnx road edges (highway, length)
    │   └── layer: buffer              ← 500m metric buffer polygon
    ├── capstone_zonal_stats.csv       ← Mean/std/min/max per layer inside buffer
    ├── capstone_dashboard.png         ← 5-panel dark analysis dashboard
    └── site_analysis_report.md        ← Auto-generated Markdown report
```

---

## 🖥️ ArcGIS Pro Integration

```
1. Open ArcGIS Pro > New Map
2. Add Data > ndvi_capstone.tif
   Symbology > Stretched > RdYlGn color ramp (red=stressed, green=healthy)

3. Add Data > sar_vv_db_capstone.tif
   Symbology > Stretched > Gray (dark=water, bright=glacier/rock)

4. Add Data > dem_capstone.tif
   Symbology > Stretched > Terrain color ramp

5. Add Data > roads_buffer.gpkg
   Layer: roads   → Highway orange line symbology, 1.5pt width
   Layer: buffer  → Orange fill, 20% transparency

6. Spatial Analyst > Zonal Statistics as Table
   Input zone: buffer layer
   Value raster: ndvi_capstone.tif
   → Validates capstone_zonal_stats.csv results

7. Layout > Export to PDF for final site analysis report
```

---

## 🔵 ENVI 5.6 Integration

```
; Open NDVI raster
File > Open > ndvi_capstone.tif
Display > Color Table > ndvi_color_table (or NDVI from standard tables)
Tools > Feature Extraction > Object-Based Image Analysis

; Compare with SAR
File > Open > sar_vv_db_capstone.tif
Display > Linked Views
  → Link NDVI and SAR views geographically
  → Pan and zoom together to compare land cover signatures

; Zonal analysis
Tools > Band Math: (b1 > 0.4) AND (b2 < -15)
  → Finds pixels with healthy vegetation AND water-like SAR (potential flooded vegetation)
```

---

## ⚠️ Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| `No raster data available` | Cache + STAC both failed | Run Ch01–Ch07 first; check internet access |
| `OSMnx download failed` | Remote area has no OSM roads | Normal for wilderness — script continues without roads |
| `rasterstats` returns None | Buffer is empty / outside raster | Check BBOX alignment; buffer may be at raster boundary |
| Memory error on large BBOX | Raster too large to load | Reduce BBOX size; use `--bbox` to specify smaller area |

---

## 📖 Key References

- OpenStreetMap contributors, 2023. *openstreetmap.org* (ODbL)
- Boeing, G. (2017). *OSMnx: New Methods for Acquiring, Constructing, Analyzing, and Visualizing Complex Street Networks.* Computers, Environment and Urban Systems.
- rasterstats documentation: https://pythonhosted.org/rasterstats/