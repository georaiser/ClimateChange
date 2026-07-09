# Capstone Site Analysis Report
    
## 📍 Territory Overview
**Coordinates (BBOX):** [-72.8, -51.8, -72.4, -51.6]

This report was generated automatically by the Python Cloud-Native Geospatial Pipeline. Real-world vector infrastructure was sourced dynamically from OpenStreetMap and overlaid onto Sentinel-2 and Copernicus DEM rasters.

---

## 📊 Analytical Results: Human Impact Zone (1km Buffer)

We buffered all local roads and infrastructure by 1000 meters to analyze the environmental conditions directly exposed to human activity.

*   **Average Vegetation Health (NDVI) in Impact Zone:** `0.413`
    *   *Minimum NDVI:* `-0.153` (Indicates paved roads, bare rock, or water)
    *   *Maximum NDVI:* `0.721` (Indicates dense, healthy forest patches near roads)
*   **Terrain Profile of Impact Zone:**
    *   *Average Elevation:* `95.7 meters`
    *   *Maximum Elevation Reached by Roads:* `720.2 meters`

---

## 🗺️ GIS Data Deliverables
The following files have been generated in the `data/processed/` directory and are ready for **ArcGIS Pro** or **ENVI**:
1.  `infrastructure_roads.shp` (Raw OSM Vector Network)
2.  `infrastructure_impact_zone.shp` (1km Dissolved Buffer)
3.  `capstone_dem.tif` (Digital Elevation Model)
4.  `capstone_ndvi.tif` (Sentinel-2 Vegetation Health)

**Recommended Actions:** Load the `.shp` buffers over the `ndvi.tif` in your GIS software to visually verify these statistical anomalies.
