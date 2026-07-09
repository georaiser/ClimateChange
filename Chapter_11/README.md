# Chapter 11: Enterprise Spatial Databases (PostGIS)

> [!IMPORTANT]
> **Status: 🚧 Planned / In Development**
> This chapter adds a production-grade spatial database layer to the GeoCascade pipeline. The plan below is ready for implementation.

## 🎯 Academic Objective

By Chapter 10, our analysis outputs are files on disk — GeoTIFFs, Shapefiles, CSVs. This works for a single analyst, but fails at scale:
- No concurrent access (two scripts can't write the same file)
- No spatial indexing (full raster scan for every query)
- No transactional integrity (partial writes leave corrupt data)

**PostGIS** extends PostgreSQL with a geometry column type, GIST spatial indexes, and 200+ spatial SQL functions. It is the industry standard for production geospatial backends, used by OpenStreetMap, ESRI, and national mapping agencies worldwide.

By the end of this chapter you will be able to:
- Deploy a PostGIS database with Docker in 3 commands
- Load Chapter 8 vector and raster outputs into the database
- Write spatial SQL queries using `ST_DWithin`, `ST_Intersects`, and `ST_MapAlgebra`
- Serve vector tiles from PostGIS using `pg_tileserv`

---

## 🐳 Docker Setup (3 Commands)

```bash
# 1. Pull the official PostGIS image
docker pull postgis/postgis:16-3.4

# 2. Start the database container
docker run -d \
  --name geocascade-postgis \
  -e POSTGRES_DB=geocascade \
  -e POSTGRES_USER=geo \
  -e POSTGRES_PASSWORD=cascade2024 \
  -p 5432:5432 \
  postgis/postgis:16-3.4

# 3. Enable PostGIS extension
docker exec geocascade-postgis \
  psql -U geo -d geocascade -c "CREATE EXTENSION postgis;"
```

**`docker-compose.yml` (recommended for development):**
```yaml
version: '3.8'
services:
  postgis:
    image: postgis/postgis:16-3.4
    environment:
      POSTGRES_DB: geocascade
      POSTGRES_USER: geo
      POSTGRES_PASSWORD: cascade2024
    ports:
      - "5432:5432"
    volumes:
      - postgis_data:/var/lib/postgresql/data

  pgadmin:
    image: dpage/pgadmin4
    environment:
      PGADMIN_DEFAULT_EMAIL: admin@geocascade.com
      PGADMIN_DEFAULT_PASSWORD: admin
    ports:
      - "5050:80"

volumes:
  postgis_data:
```

---

## 📂 Planned Files

| File | Description |
|------|-------------|
| `30_load_to_postgis.py` | Load Chapter 8 vector outputs using `geopandas.to_postgis()` |
| `31_spatial_queries.py` | Demonstrate spatial SQL: `ST_DWithin`, `ST_Buffer`, `ST_Intersects` |
| `32_raster_postgis.py` | Load ESI/CVS rasters using `raster2pgsql` + query with `ST_Value()` |
| `33_tile_server.py` | Serve vector tiles from PostGIS using `pg_tileserv` |

---

## 🗺️ Example Spatial SQL Queries

```sql
-- Find all ecological stress zones (ESI > 0.6) within 5km of roads
SELECT z.zone_id, ST_Area(z.geom) / 1e6 AS area_km2
FROM esi_high_risk z
JOIN road_network r ON ST_DWithin(z.geom, r.geom, 5000)
WHERE z.esi_score > 0.6;

-- Count glacier pixels per watershed
SELECT w.watershed_id, COUNT(*) AS glacier_pixels
FROM cryosphere_vulnerability c
JOIN watersheds w ON ST_Intersects(c.geom, w.geom)
WHERE c.cvs_score > 0.7
GROUP BY w.watershed_id
ORDER BY glacier_pixels DESC;

-- Water stress compound score within impact buffer
SELECT ST_MapAlgebra(
    ST_Clip(wsi_raster, road_buffer.geom)
) AS wsi_in_buffer
FROM wsi_raster, road_buffer
WHERE road_buffer.buffer_distance_m = 1000;
```

---

## 📊 File-based vs PostGIS Comparison

| Feature | Shapefile / GeoTIFF | PostGIS |
|---------|--------------------|---------| 
| Concurrent writers | ❌ No | ✅ Yes (ACID) |
| Spatial index | ❌ No (full scan) | ✅ GIST R-tree |
| Complex joins | ❌ No | ✅ Full SQL |
| Web serving | ❌ Manual | ✅ pg_tileserv |
| Max file size | ~2 GB (Shapefile) | Unlimited |
| Version control | Limited | ✅ SQL migrations |

---

## 🚀 Planned Installation

```bash
# Python database connector
pip install psycopg2-binary sqlalchemy geoalchemy2

# Load rasters (PostGIS bundled tool)
# raster2pgsql is included in the Docker container
docker exec geocascade-postgis bash -c \
  "raster2pgsql -s 32719 -I -C esi_map.tif public.esi | psql -U geo -d geocascade"
```

---

## 📚 Academic References

- Obe, R. & Hsu, L. (2021). *PostGIS in Action* (3rd ed.). Manning Publications.
- PostGIS Documentation: [postgis.net/docs](https://postgis.net/docs/)
- pg_tileserv: [github.com/CrunchyData/pg_tileserv](https://github.com/CrunchyData/pg_tileserv)
