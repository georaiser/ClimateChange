# 🏔️ Chapter 3: Terrain & Glacier Analysis

> **GeoCascade Pipeline — Stage 3**
> Copernicus DEM terrain derivatives, multi-temporal glacier retreat mapping, and hydrological watershed delineation.

---

## 📋 Overview

Chapter 3 moves from spectral analysis (Chapter 2) into **three-dimensional terrain analysis**. We use the Copernicus Global 30m DEM to derive slope, aspect, hillshade, and curvature — then apply multi-temporal Landsat change detection to quantify glacier retreat over 20 years, and finally delineate river watersheds using the D8 hydrological routing algorithm.

> [!IMPORTANT]
> Run scripts in order: **09 → 10 → 11**. Script 10 writes `data/raw/temp_dem.tif` which is read by script 11. Script 10 must succeed before running script 11.

---

## 📁 Scripts

| # | File | Topic | Key Outputs |
|---|------|--------|-------------|
| 09 | `09_multitemporal_glacier_retreat.py` | Landsat change detection — Grey Glacier 2003 vs 2023 | `glacier_retreat_2003_2023.tif`, `glacier_retreat_report.csv`, 4-panel dark figure |
| 10 | `10_digital_elevation_processing.py` | Copernicus DEM terrain derivatives | `slope_degrees.tif`, `aspect_degrees.tif`, `hillshade.tif`, `curvature.tif`, `terrain_statistics.csv` |
| 11 | `11_watershed_delineation.py` | pysheds D8 watershed + hipsometric curve | `flow_direction.tif`, `flow_accumulation.tif`, `river_network.tif`, `hypsometry_report.csv` |

---

## 🚀 Setup

```bash
conda activate geocascade_env

# Core geospatial
mamba install -n geocascade_env -c conda-forge pystac-client planetary-computer rasterio numpy matplotlib pandas pyproj -y

# For script 11 (watershed)
pip install pysheds
```

> [!NOTE]
> `pysheds` is not on conda-forge. Install via `pip` inside the `geocascade_env`.

---

## ▶️ Running Scripts

```bash
# Step 1: Glacier retreat analysis (independent — downloads its own Landsat)
python Chapter_03/09_multitemporal_glacier_retreat.py

# Step 2: Terrain derivatives (downloads + caches Copernicus DEM)
python Chapter_03/10_digital_elevation_processing.py

# Step 3: Watershed delineation (reads cached DEM from step 2)
python Chapter_03/11_watershed_delineation.py
```

---

## 🔬 Methods Deep-Dive

### Script 09 — Glacier Retreat (Multi-Temporal Change Detection)

**NDSI threshold method:**
```
NDSI = (Green - SWIR1) / (Green + SWIR1)
Ice mask = NDSI > 0.4
Retreat map = ice_2003 - ice_2023
  +1 = ice melted (retreated)
   0 = stable / no change
  -1 = ice advanced
```

**Landsat Collection 2 Level-2 scale factor** (critical — without this NDSI breaks):
```python
surface_reflectance = DN * 0.0000275 + (-0.2)
```

**2003 scene** is bilinearly resampled to match the **2023 master grid** before subtraction. This ensures pixel-exact change detection with no spatial misalignment.

---

### Script 10 — Terrain Derivatives

**Why convert degrees to metres before `np.gradient()`?**

CopDEM is in EPSG:4326 (geographic degrees). At 51°S:
- 1° latitude ≈ 111,000 m
- 1° longitude ≈ 111,000 × cos(51°) ≈ 69,800 m

Passing `cellsize=30` (degrees) instead of metres **underestimates slope by ~3,300×**.

```python
lat_m = abs(res[0]) * 111_000.0
lon_m = abs(res[1]) * 111_000.0 * cos(radians(-51.0))
dy, dx = np.gradient(dem, lat_m, lon_m)
slope  = degrees(arctan(sqrt(dx**2 + dy**2)))
```

**Hillshade formula** (ESRI standard):
```
shaded = cos(zenith) × cos(slope) + sin(zenith) × sin(slope) × cos(azimuth − aspect)
output = clip(255 × shaded, 0, 255)
```
Note: Clamp (not rescale) — negative illumination (back-lit faces) maps to 0, not 128.

**Curvature (Laplacian):**
```python
ddy, _ = gradient(dy, lat_m, lon_m)
_, ddx = gradient(dx, lat_m, lon_m)
curvature = ddy + ddx   # positive=convex ridge, negative=concave valley
```

---

### Script 11 — Watershed Delineation (pysheds D8)

**DEM conditioning steps** (required before flow routing):
1. **Fill Pits** — remove single-cell depressions from DEM noise
2. **Fill Depressions** — fill all closed basins (sinks) so water escapes
3. **Resolve Flats** — add tiny gradient across flat areas to break ties

**D8 encoding** (ESRI convention):
| Direction | Code |
|-----------|------|
| N | 64 |
| NE | 128 |
| E | 1 |
| SE | 2 |
| S | 4 |
| SW | 8 |
| W | 16 |
| NW | 32 |

**Hipsometric Integral (HI)** — basin maturity indicator:
| HI | Interpretation |
|----|---------------|
| > 0.6 | Young, actively eroding basin |
| 0.35–0.6 | Equilibrium / mature basin |
| < 0.35 | Monadnock / old, tectonically stable |

Patagonian Andes basins typically show HI = 0.5–0.7 (young to equilibrium).

---

## 📂 Output Directory Structure

```
Chapter_03/
├── data/
│   ├── raw/
│   │   └── temp_dem.tif              ← cached DEM (shared between scripts 10 & 11)
│   └── processed/
│       ├── glacier_retreat/
│       │   ├── glacier_ndsi_2003.tif
│       │   ├── glacier_ndsi_2023.tif
│       │   ├── glacier_retreat_2003_2023.tif  ← main output
│       │   ├── glacier_retreat_report.csv
│       │   └── glacier_retreat_2003_2023.png
│       ├── terrain/
│       │   ├── copernicus_dem.tif
│       │   ├── slope_degrees.tif
│       │   ├── aspect_degrees.tif
│       │   ├── hillshade.tif
│       │   ├── curvature.tif
│       │   ├── terrain_statistics.csv
│       │   └── terrain_derivatives.png
│       └── watershed/
│           ├── dem_conditioned.tif
│           ├── flow_direction.tif
│           ├── flow_accumulation.tif
│           ├── river_network.tif
│           ├── hypsometry_report.csv
│           └── watershed_analysis.png
```

---

## 🖥️ ArcGIS Pro Integration

### Terrain Analysis (Script 10 validation)
Use **3D Analyst** toolbox to independently compute derivatives and compare with script outputs — differences < 0.5° are numerical noise:

```
Analysis > Tools > Slope          → compare with slope_degrees.tif
Analysis > Tools > Aspect         → compare with aspect_degrees.tif
Analysis > Tools > Hillshade      → compare with hillshade.tif (Az=315, Ze=45)
```

### Watershed Analysis (Script 11)
```
Spatial Analyst > Hydrology > Fill              → condition the DEM
Spatial Analyst > Hydrology > Flow Direction    → D8 routing
Spatial Analyst > Hydrology > Flow Accumulation → upstream count
Spatial Analyst > Hydrology > Stream Order      → Strahler stream ordering
                                                  (not in pysheds natively)
Spatial Analyst > Hydrology > Watershed         → basin delineation from pour point
```

### Glacier Retreat Visualization (Script 09)
```
Add glacier_retreat_2003_2023.tif
Symbology > Unique Values:
  Value -1 → Blue  (Ice Advanced)
  Value  0 → Gray  (Stable)
  Value  1 → Red   (Ice Retreated / Melted)
```

---

## 🔵 ENVI 5.6 Integration

### Terrain
```
File > Open > copernicus_dem.tif
Toolbox > Topographic > Slope
Toolbox > Topographic > Aspect
```

### Glacier
```
File > Open > glacier_retreat_2003_2023.tif
Tools > Color Map → assign Red=1, Gray=0, Blue=-1
```

### IDL workflow (advanced)
```idl
; Chapter_03/envi/03_terrain_derivatives.pro
; Computes slope, aspect, hillshade via ENVI DEM functions
```

---

## ⚠️ Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| `pysheds` import error | Not installed via pip | `pip install pysheds` |
| `np.in1d` AttributeError | NumPy 2.0 removed it | Already patched at top of script 11 |
| No Landsat found | Cloud cover > 40% | Script uses 40% threshold; Patagonia is cloudy — if still failing, try a wider date range |
| `temp_dem.tif` not found | Script 11 before script 10 | Run script 10 first; it creates the cached DEM |
| Flat DEM / zero slope | BBOX in ocean or wrong CRS | Verify BBOX covers land, check that DEM CRS is read correctly |

---

## 📖 Key References

- Chavez, P.S. (1988). *An improved dark-object subtraction technique for atmospheric scattering correction of multispectral data.* Remote Sensing of Environment.
- Strahler, A.N. (1957). *Quantitative analysis of watershed geomorphology.* Transactions of the AGU.
- Mark, D.M. (1988). *Network models in geomorphology.* Modelling Geomorphological Systems.
- Copernicus DEM: [spacedata.copernicus.eu](https://spacedata.copernicus.eu)
- Landsat C2 L2: [USGS Earth Explorer](https://earthexplorer.usgs.gov)
