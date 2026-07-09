# Chapter 5: Moisture Stress & Zonal Statistics

## Academic Objective
Compute vegetation moisture stress from Sentinel-2 SWIR bands and aggregate raster
statistics across spatial zones using GeoDataFrame vector polygons (rasterstats).

---

## Scripts

### 14_moisture_stress_indices.py — NDMI and MSI from Sentinel-2

Calculates two complementary moisture indices from NIR (B08) and SWIR1 (B11):

**NDMI (Normalized Difference Moisture Index):**
  NDMI = (NIR - SWIR1) / (NIR + SWIR1)
  Higher values = more canopy moisture (healthy vegetation).

**MSI (Moisture Stress Index):**
  MSI = SWIR1 / NIR
  Higher values = more drought stress (inverse of NDMI).

> [!CAUTION]
> B11 (SWIR1) is 20m native resolution with a DIFFERENT transform than B08 (10m).
> Must compute an independent window + transformer for B11 from the 20m source CRS.
> Reusing the 10m NIR window on a 20m grid silently reads the wrong geographic area.

**Fixes applied:**
- B11 uses its own independently computed window and Transformer
- nodata=-9999 for all output TIFFs
- plt.close() after figures

Run: `python 14_moisture_stress_indices.py`

Outputs: ndmi.tif, msi.tif, moisture_stress_comparison.png

---

### 15_zonal_statistics.py — Vector-Raster Aggregate Analysis

Generates 4 quadrant polygons (NW/NE/SW/SE of the BBOX) and runs
rasterstats.zonal_stats to extract aggregate raster values per zone.

**Expanded statistics (Tier 3):**
Each zone now reports: mean, std, min, max (was only mean).

**Fixes applied:**
- None-safe formatter: zones entirely NoData return 'N/A' instead of crashing
- nodata=-9999 in all raster exports (required by rasterstats for correct masking)
- STAC guards for DEM and S2 queries
- plt.close()

> [!NOTE]
> rasterstats and GDAL cannot reliably use np.nan as nodata.
> Always pass nodata=-9999 to zonal_stats AND to rasterio write.

Run: `python 15_zonal_statistics.py`

Outputs: moisture_stress_map.tif, zonal_statistics_report.csv

---

## Key Concepts

| Concept | Explanation |
|---|---|
| NDMI | NIR-SWIR1 ratio: positive = moisture, negative = stress |
| MSI | SWIR1/NIR: higher = more drought stress (inverse of NDMI) |
| B11 at 20m | Needs its own window — cannot reuse 10m NIR window |
| Zonal Statistics | Aggregate raster values inside vector polygon per zone |
| nodata=-9999 | GDAL/rasterstats cannot mask np.nan reliably; use integer sentinel |

## Installation

```bash
mamba install -n geocascade_env -c conda-forge pystac-client planetary-computer rasterio geopandas rasterstats shapely pyproj matplotlib numpy -y
```