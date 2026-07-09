# Chapter 12: Capstone — Automated Site Analysis CLI Pipeline

## Academic Objective
Culmination of Chapters 1-8. A unified command-line tool that accepts a user-defined
BBOX and date range, dynamically fuses Optical (NDVI), Radar (SAR VV), Elevation (DEM),
and Vector Infrastructure (OpenStreetMap roads) to generate an automated Site Analysis Report.

---

## CLI Usage

```bash
python capstone_pipeline.py --bbox -72.8 -51.8 -72.4 -51.6 --date_range 2023-01-01/2023-03-31
```

---

## Critical Bug Fixes

> [!CAUTION]
> **BUG 1 — OSMnx argument order:** graph_from_bbox() uses NAMED kwargs:
>   north, south, east, west (in that order).
> Passing BBOX positionally sends BBOX[0]=-73.30 as 'north' latitude.
> This silently queries a completely wrong geographic location.
> Fix: always use north=..., south=..., east=..., west=... explicitly.

> [!CAUTION]
> **BUG 2 — EPSG:3857 distortion at high latitude:** Web Mercator introduces
> ~40% linear distance distortion at 51 deg S. A 1km buffer becomes ~1.4km.
> Fix: project to EPSG:32719 (UTM Zone 19S) before buffering. Accurate to <0.1%.

> [!WARNING]
> **BUG 3 — None crash in zonal_stats:** rasterstats returns None for stat values
> when the entire zone is covered by NoData pixels. Calling None.format() crashes.
> Fix: _fmt() helper returns 'N/A' for None values instead of crashing.

Additional fixes: nodata=-9999; STAC empty guard for DEM; plt.close().

---

## Pipeline Stages

| Stage | Description |
|---|---|
| 1. Vector Sourcing | OSMnx downloads road network from OpenStreetMap for BBOX |
| 2. Raster Fusion | Planetary Computer STAC: DEM + NDVI (S2) + SAR (S1-RTC) |
| 3. Impact Zone | 1km UTM-accurate buffer around roads (EPSG:32719) |
| 4. Zonal Analysis | rasterstats zonal_stats: mean/std/min/max per impact zone |
| 5. Report | Markdown report + 3-panel chart exported automatically |

---

## Outputs

- capstone_multi_panel.png (3-panel: DEM, NDVI, SAR)
- site_analysis_report.md (auto-generated Markdown report)
- infrastructure_roads.shp + impact_zone.shp
- capstone_dem.tif, capstone_ndvi.tif, capstone_sar.tif

---

## Key Concepts

| Concept | Explanation |
|---|---|
| CLI pipeline | argparse enables reproducible, parameterized analysis |
| OSMnx kwargs | north/south/east/west must be named, never positional |
| UTM vs Web Mercator | EPSG:3857 for display only; EPSG:32719 for distance/area |
| _fmt() pattern | Defensive None-safe formatter for scientific pipelines |
| Convergent evidence | DEM + NDVI + SAR simultaneously answer what is here and how stressed |

## Installation

```bash
mamba install -n geocascade_env -c conda-forge pystac-client planetary-computer rasterio rasterstats osmnx geopandas numpy matplotlib pyproj -y
```