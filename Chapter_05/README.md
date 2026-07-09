# Chapter 5: Zonal Statistics & Moisture Stress

## 🎯 Academic Objective

Satellites see the landscape as a continuous raster grid. Environmental managers, however, think in **zones** — watersheds, protected areas, administrative units. This chapter bridges that gap by computing per-zone summary statistics from raster data, and introduces two complementary moisture stress indices that detect vegetation drought stress before it becomes visible to the human eye.

By the end of this chapter you will be able to:
- Compute NDMI and MSI moisture indices from Sentinel-2 NIR and SWIR bands
- Define management zones as vector polygons and run zonal statistics on any raster
- Produce a professional management report table comparing ecological indicators across zones

---

## 🛠️ Scripts & Modules

### `14_moisture_stress_indices.py`
Computes two complementary moisture indices that capture different aspects of vegetation water content.

> [!WARNING]
> **Critical: B11 (SWIR) Requires Its Own Window**
>
> B11 is natively **20m resolution**. Its pixel grid is completely different from B08 NIR (10m). Always compute the rasterio window independently from each band's own transform:
> ```python
> # ✅ CORRECT — independent windows per resolution
> with rasterio.open(item.assets["B08"].href) as src_nir:
>     win_nir = from_bounds(minx, miny, maxx, maxy, src_nir.transform)
>     nir = src_nir.read(1, window=win_nir, out_shape=target_shape,
>                        resampling=Resampling.bilinear).astype('float32')
>
> with rasterio.open(item.assets["B11"].href) as src_swir:
>     win_swir = from_bounds(minx, miny, maxx, maxy, src_swir.transform)
>     swir = src_swir.read(1, window=win_swir, out_shape=target_shape,
>                          resampling=Resampling.bilinear).astype('float32')
>
> # ❌ WRONG — reusing win_nir on a 20m band crops the wrong area
> swir = src_swir.read(1, window=win_nir)
> ```

#### Index Formulas

**① NDMI — Normalized Difference Moisture Index**
$$NDMI = \frac{NIR - SWIR}{NIR + SWIR}$$

| Value | Interpretation |
|-------|---------------|
| +0.4 to +1.0 | High canopy water content — healthy, wet vegetation |
| 0.0 to +0.4 | Moderate moisture — normal vegetation |
| −0.4 to 0.0 | Low moisture — stressed or sparse vegetation |
| < −0.4 | Very dry — bare soil, rock, or severe drought |

**② MSI — Moisture Stress Index**
$$MSI = \frac{SWIR}{NIR}$$

- MSI > 3.0 → severe moisture stress (drought conditions)
- MSI ≈ 0.4 → healthy well-watered canopy
- Inverse of NDMI: high MSI = low moisture = stressed vegetation

> [!NOTE]
> The hard clip `MSI = min(MSI, 3.0)` in the visualization is a **display choice only** — it prevents extreme values from compressed bare rock or ice pixels from dominating the colormap. The clipping does **not** reflect a scientific threshold for the physical MSI value.

**Output:** 2-panel chart (NDMI left, MSI right) + 2 geocoded GeoTIFFs.

---

### `15_zonal_statistics.py`
Divides the study BBOX into 4 management zones (NW, NE, SW, SE quadrants) as synthetic administrative polygons and computes per-zone raster statistics.

**Algorithm:**
1. Create 4 BBOX quadrant polygons as a GeoDataFrame (EPSG:4326)
2. Download Sentinel-2 NDVI and Copernicus DEM rasters
3. **Reproject GeoDataFrame to match raster CRS** — this is the critical step
4. Run `rasterstats.zonal_stats()` for each zone → returns mean, max, min, std per zone
5. Print formatted management report

> [!CAUTION]
> **Two Common Bugs in Zonal Statistics:**
>
> **Bug 1 — CRS Mismatch:** `rasterstats.zonal_stats()` silently produces wrong results if the vector and raster CRS don't match. Always reproject before running stats:
> ```python
> gdf_projected = gdf.to_crs(raster_profile['crs'])
> stats = zonal_stats(gdf_projected, raster_array, ...)
> ```
>
> **Bug 2 — Index Mismatch:** `zonal_stats()` returns a plain Python **list** (not a GeoDataFrame). Use `enumerate()`, not `iterrows()` index:
> ```python
> # ✅ CORRECT
> for i, stats_row in enumerate(stats_list):
>     mean_val = stats_row.get('mean') or float('nan')
>
> # ❌ WRONG — GDF row index may not match list position if GDF was filtered
> for gdf_idx, row in gdf.iterrows():
>     mean_val = stats_list[gdf_idx]['mean']  # IndexError if GDF filtered
> ```

**Example output:**
```
Zone        Mean NDVI   Max NDVI   Mean Elev (m)
NW Sector   0.312       0.741      847.3
NE Sector   0.198       0.652      1204.7
SW Sector   0.421       0.843      623.1
SE Sector   0.356       0.778      741.2
```

---

## 📐 Key Formulas

| Index | Formula | Range | High Value Means |
|-------|---------|-------|-----------------|
| NDMI | $(NIR - SWIR)/(NIR + SWIR)$ | [-1, +1] | High canopy water |
| MSI | $SWIR / NIR$ | [0, ∞] | Drought/stress |

---

## 🚀 How to Run

### Install Dependencies
```bash
mamba activate geocascade_env
mamba install -n geocascade_env -c conda-forge \
    pystac-client planetary-computer rasterio geopandas \
    rasterstats shapely numpy matplotlib pyproj -y
```

### Execute Scripts
```bash
# NDMI and MSI moisture stress indices
python 14_moisture_stress_indices.py

# Zonal statistics report by management zone
python 15_zonal_statistics.py
```

---

## 🗺️ GIS Interoperability

**ArcGIS Pro:** Load `ndmi.tif` → use **Zonal Statistics as Table** (Spatial Analyst) with any polygon layer → join the output table back to the polygon layer for a choropleth map.

**ENVI:** Load NDMI as a single-band raster → use **Statistics** tool to compute per-ROI statistics → export as CSV for reporting.

> [!TIP]
> The 4-quadrant management zones in this script are synthetic. In real projects, replace them with actual administrative boundaries downloaded from national GIS portals or GADM (Global Administrative Areas). Simply swap the `gdf` variable with `geopandas.read_file('your_shapefile.shp')`.
