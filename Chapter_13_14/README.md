# Chapter 13-14: Advanced Capstone — Full-Stack Geospatial REST API

> [!IMPORTANT]
> **Status: 🚧 Planned / In Development**
> This chapter wraps the entire GeoCascade pipeline into a production-grade web service with async job processing, GeoJSON endpoints, and Docker deployment.

## 🎯 Academic Objective

The Chapter 12 capstone runs locally via CLI. This chapter transforms it into a **REST API** that any web application, GIS dashboard, or automated workflow can call over HTTP — without needing Python installed.

This is the architecture used by commercial EO platforms (Sentinel Hub, Planet, Maxar) to expose geospatial analysis to end users.

By the end of this chapter you will be able to:
- Design and implement a FastAPI async REST API for geospatial analysis
- Handle long-running satellite download jobs with background task queues
- Return GeoJSON-formatted spatial results from Python endpoints
- Deploy the full pipeline in a Docker container

---

## 🛠️ API Endpoints (Planned)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/analyze` | Submit new analysis job (BBOX + date range) |
| `GET` | `/results/{job_id}` | Poll job status + download results |
| `GET` | `/health` | Liveness check |
| `GET` | `/docs` | Swagger UI (auto-generated) |

---

## 📋 Request / Response Schemas

**POST `/analyze` — Request:**
```json
{
  "bbox": [-73.30, -51.10, -72.90, -50.80],
  "date_range": "2023-01-01/2023-03-31",
  "indices": ["ndvi", "ndsi", "esi", "cvs", "wsi"],
  "buffer_m": 1000
}
```

**POST `/analyze` — Response (202 Accepted):**
```json
{
  "job_id": "a3f7c2d1-8b44-4e9a-b521-3c5e9f2d7a01",
  "status": "queued",
  "estimated_runtime_s": 180,
  "poll_url": "/results/a3f7c2d1-8b44-4e9a-b521-3c5e9f2d7a01"
}
```

**GET `/results/{job_id}` — Response (200 OK, completed):**
```json
{
  "job_id": "a3f7c2d1-8b44-4e9a-b521-3c5e9f2d7a01",
  "status": "completed",
  "runtime_s": 143,
  "summary": {
    "mean_ndvi_inside_buffer": 0.187,
    "mean_ndvi_outside_buffer": 0.421,
    "esi_score_mean": 0.63,
    "cvs_score_mean": 0.71,
    "wsi_score_mean": 0.38
  },
  "geojson": { "type": "FeatureCollection", "features": [...] },
  "report_url": "/downloads/a3f7c2d1/report.md",
  "dashboard_url": "/downloads/a3f7c2d1/dashboard.png"
}
```

---

## 📂 Planned Files

| File | Description |
|------|-------------|
| `api.py` | FastAPI app definition + endpoint routes |
| `schemas.py` | Pydantic request/response models |
| `pipeline_worker.py` | Async wrapper around `capstone_pipeline.py` |
| `job_store.py` | In-memory (or Redis) job status store |
| `Dockerfile` | Container definition |
| `docker-compose.yml` | API + optional Redis + PostGIS stack |

---

## 🚀 Running Locally

### Install Dependencies
```bash
pip install fastapi uvicorn pydantic aiofiles python-multipart
mamba install -n geocascade_env -c conda-forge \
    pystac-client planetary-computer rasterio geopandas numpy shapely scipy -y
```

### Start the API Server
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

**Interactive API explorer:** Open [http://localhost:8000/docs](http://localhost:8000/docs) in your browser — Swagger UI is auto-generated.

---

## 🐳 Docker Deployment

```bash
# Build the image
docker build -t geocascade-api:latest .

# Run with Planetary Computer credentials
docker run -d \
  -p 8000:8000 \
  -e PC_SDK_SUBSCRIPTION_KEY=your_key_here \
  -v $(pwd)/results:/app/results \
  geocascade-api:latest
```

---

## 📡 Example curl Commands

```bash
# Submit analysis job
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"bbox": [-73.30, -51.10, -72.90, -50.80], "date_range": "2023-01-01/2023-03-31"}'

# Poll job status
curl http://localhost:8000/results/a3f7c2d1-8b44-4e9a-b521-3c5e9f2d7a01

# Health check
curl http://localhost:8000/health
```

---

## 🏗️ Architecture Diagram

```
Client (Browser / GIS Dashboard / Automation Script)
    │
    │  HTTP POST /analyze
    ▼
FastAPI Server (uvicorn)
    │
    ├── Validate request (Pydantic schema)
    ├── Generate job_id (UUID4)
    ├── Launch background task (asyncio)
    └── Return 202 Accepted + job_id
         │
         ▼
Background Worker (pipeline_worker.py)
    │
    ├── Download STAC data (pystac-client)
    ├── Compute indices (rasterio / numpy)
    ├── Run ESI / CVS / WSI scoring
    ├── Generate report + dashboard PNG
    └── Update job store: status = "completed"
         │
         ▼
Client polls GET /results/{job_id}
    └── Receives: summary JSON + GeoJSON + download URLs
```

---

## 📚 Academic References

- FastAPI Documentation: [fastapi.tiangolo.com](https://fastapi.tiangolo.com)
- Ramírez, S. (2018). FastAPI: High performance, easy to learn, fast to code. *GitHub*.
- OGC API — Processes standard: [ogcapi.ogc.org/processes](https://ogcapi.ogc.org/processes/) — the formal geospatial REST API specification this chapter implements informally.
