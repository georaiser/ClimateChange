# 🗄️ Chapter 11: Enterprise Spatial Databases — PostGIS Integration

> **GeoCascade Pipeline — Stage 11**
> Moves from flat-file GIS (GeoTIFFs, Shapefiles) to a PostGIS spatial database
> and exposes results through a FastAPI REST endpoint — bridging satellite analysis
> with enterprise geospatial infrastructure.

---

## 📋 Overview

| Script | Topic | Key Outputs |
|--------|-------|-------------|
| `26_postgis_integration.py` | PostGIS ingestion + 3 spatial SQL queries + FastAPI stub | GeoJSON exports, `spatial_query_results.csv`, `postgis_integration_dashboard.png` |

> [!NOTE]
> The script runs **without a live PostgreSQL server** using an automatic SpatiaLite fallback.
> All demonstrations work out-of-the-box. PostGIS is only required for the optional enterprise path.

---

## 🚀 Setup

### Option A — SpatiaLite Fallback (Always Works)

```bash
conda activate geocascade_env

mamba install -n geocascade_env -c conda-forge \
    geopandas rasterio pandas matplotlib shapely -y
```

### Option B — Full PostGIS Path

```bash
# 1. Python dependencies
mamba install -n geocascade_env -c conda-forge \
    psycopg2 sqlalchemy geoalchemy2 geopandas \
    fastapi uvicorn rasterio pandas matplotlib -y

# 2. PostGIS server via Docker
docker run -d --name geocascade-postgis \
    -e POSTGRES_DB=geocascade \
    -e POSTGRES_USER=geo \
    -e POSTGRES_PASSWORD=cascade2024 \
    -p 5432:5432 \
    postgis/postgis:16-3.4

# 3. Verify connection
psql -h localhost -U geo -d geocascade -c "SELECT PostGIS_Version();"
```

---

## ▶️ Run

```bash
python Chapter_11/26_postgis_integration.py
```

---

## 🔬 Why PostGIS?

### File-Based GIS Limitations (At Scale)

| Problem | Flat Files (GeoTIFF/Shapefile) | PostGIS Solution |
|---------|-------------------------------|-----------------|
| No spatial index | Full scan for every query | GIST index — O(log n) lookup |
| No transactions | Risk of partial writes | ACID transactions |
| No concurrency | One writer at a time | Multi-user concurrent access |
| No SQL | External tools required | 200+ spatial SQL functions |
| File size limit | Shapefile: 2 GB max | PostgreSQL: unlimited |
| Column name length | Shapefile: 10 chars max | PostgreSQL: 63 chars |

### When File-Based GIS Is Still Fine

> [!NOTE]
> For single-user analysis with < 50 rasters and < 1 million features, flat files (GeoTIFF + GeoPackage) are fine. PostGIS is the right choice for multi-user, web-served, or large-scale production environments.

---

## 📐 Spatial SQL Demonstrations

### Query 1: Vulnerability Zone Filtering

```sql
-- Find all zones with high climate vulnerability (CVI > 0.6)
-- and return their geometry as GeoJSON
SELECT
    zone_id,
    mean_ndvi,
    cvi_score,
    ST_AsGeoJSON(geom)    AS geojson,
    ST_Area(geom::geography) / 1e6   AS area_km2
FROM vulnerability_zones
WHERE cvi_score > 0.6
ORDER BY cvi_score DESC;
```

**Returns:** Vulnerable zones for targeted field survey.

### Query 2: Glacier Buffer Intersection

```sql
-- Find all vegetation zones intersecting a 1km glacier buffer
-- (potentially at risk from glacial lake outburst floods - GLOF)
SELECT
    z.zone_id,
    z.mean_ndvi,
    ST_Distance(z.geom::geography, g.geom::geography) AS dist_from_glacier_m
FROM vegetation_zones z, glacier_outlines g
WHERE ST_Intersects(
    z.geom,
    ST_Buffer(g.geom::geography, 1000)::geometry   -- 1km buffer in metres
)
ORDER BY dist_from_glacier_m ASC;
```

### Query 3: Zonal Aggregate by Watershed

```sql
-- Compute mean CVI per watershed for basin-level risk prioritisation
SELECT
    w.basin_id,
    w.basin_name,
    COUNT(z.zone_id)        AS n_zones,
    AVG(z.cvi_score)        AS mean_cvi,
    MAX(z.cvi_score)        AS max_cvi,
    ST_Area(w.geom::geography) / 1e6  AS basin_area_km2
FROM vulnerability_zones z
    JOIN watersheds w ON ST_Within(z.centroid, w.geom)
GROUP BY w.basin_id, w.basin_name, w.geom
ORDER BY mean_cvi DESC;
```

> [!IMPORTANT]
> `::geography` cast converts geometry to the geography type, which uses **metres** for
> `ST_Area` and `ST_Buffer`. Without this cast, `ST_Buffer(geom, 1000)` applies 1000
> **degrees** as the buffer distance — geographically meaningless.

---

## 🌐 FastAPI REST Endpoint Stub

```python
from fastapi import FastAPI
from geoalchemy2 import Geometry
import geopandas as gpd

app = FastAPI(title="GeoCascade Spatial API", version="1.0")

@app.get("/vulnerable-zones", response_model=dict)
async def get_vulnerable_zones(min_cvi: float = 0.6):
    """Return high-vulnerability zones as GeoJSON."""
    gdf = gpd.read_postgis(
        f"SELECT * FROM vulnerability_zones WHERE cvi_score > {min_cvi}",
        con=engine, geom_col="geom"
    )
    return gdf.__geo_interface__

@app.get("/glacier-buffer/{radius_m}", response_model=dict)
async def get_glacier_buffer(radius_m: int = 1000):
    """Return glacier buffer zone as GeoJSON."""
    ...

@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0"}
```

**Start the API:**
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
# API docs: http://localhost:8000/docs
```

---

## 🐳 Docker Compose (Optional)

```yaml
# docker-compose.yml — PostGIS + pgAdmin
version: "3.8"
services:
  postgis:
    image: postgis/postgis:16-3.4
    environment:
      POSTGRES_DB:       geocascade
      POSTGRES_USER:     geo
      POSTGRES_PASSWORD: cascade2024
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

  pgadmin:
    image: dpage/pgadmin4:latest
    environment:
      PGADMIN_DEFAULT_EMAIL:    admin@geocascade.com
      PGADMIN_DEFAULT_PASSWORD: admin
    ports:
      - "5050:80"
    depends_on:
      - postgis

volumes:
  pgdata:
```

```bash
docker compose up -d
# Access pgAdmin: http://localhost:5050
# Access PostGIS: localhost:5432 (user: geo, password: cascade2024)
```

---

## 📂 Output Structure

```
Chapter_11/
└── data/processed/
    ├── vulnerability_zones.geojson        ← Query 1 results (high CVI zones)
    ├── glacier_buffer_intersections.geojson ← Query 2 results
    ├── watershed_aggregates.csv           ← Query 3 results
    ├── spatial_query_results.csv          ← Combined summary
    └── postgis_integration_dashboard.png  ← 4-panel: CVI map, buffer, zonal, schema
```

---

## 🖥️ ArcGIS Pro Integration

```
Connect ArcGIS Pro to a live PostGIS database:
  Catalog > Database Connections > Add Database Connection
    Connection File Name: geocascade_postgis
    Database Platform:    PostgreSQL
    Instance:             localhost:5432
    Database:             geocascade
    User Name/Password:   geo / cascade2024

  After connecting, browse to the connection and add layers directly:
    → vulnerability_zones (polygon layer with cvi_score attribute)
    → glacier_buffer (computed buffer polygon)
    → watersheds (aggregated by basin_id)

  Symbology > Graduated Colors on cvi_score field
  → Compare with capstone_pipeline.py outputs from Chapter 12
```

---

## 🔵 ENVI 5.6 Integration

```
; ENVI primarily works with raster data.
; For PostGIS vector layers, export from pgAdmin as Shapefile or GeoJSON:

pgAdmin > Right-click table > Export > Shapefile
  → Load shapefile in ENVI: File > Open > .shp
  → Overlay on satellite data using Vector > Overlay Vectors

; Raster IngeSTion to PostGIS (from ENVI command line):
;  Use raster2pgsql command-line tool:
raster2pgsql -s 32719 -I -C cascade_ml_prediction.tif public.ml_prediction | \
    psql -h localhost -U geo -d geocascade
;  Then in PostGIS:
SELECT ST_Value(rast, ST_SetSRID(ST_Point(-72.95, -51.02), 4326))
    FROM public.ml_prediction;
;  → Returns land cover class at that lat/lon
```

---

## 📊 PostGIS vs Alternatives Comparison

| Feature | Shapefile | GeoPackage | PostGIS | Cloud-Native (STAC/COG) |
|---------|-----------|------------|---------|------------------------|
| Spatial index | ❌ External | ✅ RTREE | ✅ GIST | ✅ Cloud spatial |
| Multi-user | ❌ | ❌ | ✅ | ✅ |
| Transactions | ❌ | ✅ | ✅ | N/A |
| Spatial SQL | ❌ | Limited | ✅ 200+ functions | N/A |
| Raster support | ❌ | Limited | ✅ PostGIS Raster | ✅ COG |
| Web tile serving | ❌ | ❌ | ✅ pg_tileserv | ✅ Native |
| Best for | Desktop GIS | Single-user | Enterprise | Cloud analysis |

---

## ⚠️ Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| `psycopg2.OperationalError` | PostgreSQL not running | Script auto-switches to SpatiaLite fallback |
| `ST_Buffer returns polygon around globe` | Missing `::geography` cast | Always cast: `ST_Buffer(geom::geography, 1000)::geometry` |
| `GIST index not used` | Table has no ANALYZE | Run `ANALYZE vulnerability_zones;` after ingestion |

---

## 📖 Key References

- Obe, R., Hsu, L. (2021). *PostGIS in Action, 3rd Edition.* Manning.
- ESRI. (2023). *ArcGIS Pro — Connect to a PostgreSQL database.*
- FastAPI documentation: https://fastapi.tiangolo.com
