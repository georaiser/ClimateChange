# Chapter 11: PostGIS Spatial Database Integration

## Overview

Demonstrates how to move from flat-file GeoTIFF/Shapefile outputs into a
PostGIS spatial database and expose the results through a lightweight
FastAPI REST endpoint.

The script runs **without a live PostgreSQL/PostGIS server** via a
GeoJSON fallback (auto-detected).

## Scripts

| Script | Purpose |
|---|---|
| `26_postgis_integration.py` | PostGIS ingestion + spatial SQL + FastAPI stub |

## Spatial SQL Examples

```sql
-- Query 1: High-vulnerability zones (CVI > 0.6)
SELECT zone_id, mean_ndvi, cvi_score, ST_AsGeoJSON(geom)
FROM vulnerability_zones WHERE cvi_score > 0.6;

-- Query 2: Zones within 1km of a glacier
SELECT z.zone_id FROM zones z, glaciers g
WHERE ST_Intersects(z.geom, ST_Buffer(g.geom::geography, 1000)::geometry);

-- Query 3: Mean CVI per elevation band
SELECT elev_band, AVG(cvi_score) as mean_cvi
FROM zones GROUP BY elev_band ORDER BY mean_cvi DESC;
```

## FastAPI Endpoints

| Endpoint | Returns |
|---|---|
| `GET /zones` | All management zones as GeoJSON |
| `GET /zones/high-risk` | Zones with CVI > 0.6 |
| `GET /glaciers/adjacent-zones` | Zones within 1km of glacier |
| `GET /health` | Service status |

## Installation

```bash
# Minimum (no PostGIS server):
mamba install -n geocascade_env -c conda-forge geopandas rasterio matplotlib pandas -y

# Full (with PostGIS + FastAPI):
mamba install -n geocascade_env -c conda-forge psycopg2 sqlalchemy geoalchemy2 geopandas fastapi uvicorn rasterio -y
```

## Run

```bash
conda activate geocascade_env
python Chapter_11/26_postgis_integration.py
```
