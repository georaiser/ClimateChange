# Chapter 11: PostGIS Integration & Spatial Database (Roadmap)

> [!NOTE]
> This chapter defines the architecture. Full implementation is planned for a future release.

## Academic Objective
Store all geospatial outputs in a PostGIS spatial database, run spatial SQL queries,
and expose results via a FastAPI REST endpoint for web dashboards.

---

## Planned Stack

| Component | Technology |
|---|---|
| Database | PostgreSQL 15 + PostGIS 3.3 |
| ORM | SQLAlchemy + GeoAlchemy2 |
| REST API | FastAPI + uvicorn |
| Vector ingestion | GDAL ogr2ogr |
| Raster ingestion | raster2pgsql |
| GIS client | QGIS live PostGIS layer |

## Planned Spatial SQL Operations

- ST_Intersects: find roads that cross glacier retreat zones
- ST_Buffer: create impact zones around infrastructure
- ST_Within: identify villages inside flood-risk areas
- ST_Value: extract raster values at point locations
- ST_MapAlgebra: multi-band raster arithmetic in-database

## Key Concepts

- GIST spatial index: dramatically speeds spatial join queries
- GeoJSON: standard format for web-based geospatial data exchange
- REST API design: bbox parameter, CRS negotiation, pagination
- Raster vs vector in PostGIS: different storage backends for different analysis