# Chapter 3: Topography & Glacial Retreat

## 🎯 Academic Objective

The landscape beneath our feet is not static. Glaciers retreat. Rivers carve new paths. Elevation shapes climate, vegetation, and hydrology. This chapter uses two of the most powerful Earth Observation tools — multi-temporal change detection and terrain analysis — to quantify these changes with satellite data.

By the end of this chapter you will be able to:
- Detect 20 years of glacial retreat from Landsat imagery using NDSI
- Compute slope, aspect, hillshade, and curvature from a 30m DEM
- Delineate watersheds and river networks using the ArcHydro D8 algorithm in Python
- Generate a Hipsometric Curve to assess geomorphic maturity of a drainage basin

---

## 🛠️ Scripts & Modules

### `09_multitemporal_glacier_retreat.py`
Downloads Landsat imagery from **2003** and **2023** for the same BBOX and computes NDSI to delineate glacier extent at each epoch. Outputs a 3-panel retreat map.

> [!CAUTION]
> **Critical: Landsat C2-L2 Scale Factor — Without This, All Results Are Wrong**
>
> Landsat Collection 2 Level-2 Surface Reflectance data is stored as scaled integers. Raw DNs (~7000–20000) must be converted before computing any index:
> ```python
> # ✅ CORRECT — applied to BOTH Green and SWIR bands
> band = src.read(1, window=win).astype('float32')
> band = band * 0.0000275 - 0.2   # C2-L2 calibration formula
> band = np.clip(band, 0, 1)       # Clamp to valid reflectance range
> ```
> **Without this step:** Raw Green DNs (~7000) and SWIR DNs (~2000) produce NDSI ≈ (7000-2000)/(7000+2000) ≈ +0.55 for EVERY pixel — including bare rock and ocean. The glacier threshold of `NDSI > 0.4` is exceeded everywhere, making retreat analysis completely meaningless.

**Glacier detection algorithm:**
$$NDSI = \frac{Green - SWIR}{Green + SWIR}$$

- $NDSI > 0.4$ → confirmed glacier / snow pixel
- Retreat Map = `ice_2003 - ice_2023`
  - `+1` = Ice **lost** (melted/retreated) → shown in **red**
  - `0`  = Stable ice extent → shown in **gray**
  - `-1` = Ice **gained** (advanced) → shown in **cyan**

**Visualization:** 3-panel chart using `matplotlib.colors.ListedColormap` (not `plt.cm.colors.ListedColormap` — that raises `AttributeError`).

---

### `10_digital_elevation_processing.py`
Downloads the Copernicus DEM GLO-30 (30m resolution) and computes standard terrain derivatives.

> [!WARNING]
> **Geographic CRS Pitfall — Slope Values Wrong by Factor ~3300**
>
> CopDEM GLO-30 is delivered in **EPSG:4326** (geographic coordinates, units = degrees). Calling `np.gradient(dem, 30.0, 30.0)` tells NumPy that pixels are 30m apart — but they are actually ~0.00028° apart.
>
> **Correct approach:** Account for degrees-to-meters conversion at the study latitude:
> ```python
> lat_m_per_deg = 111_000.0                              # ~constant
> lon_m_per_deg = 111_000.0 * np.cos(np.radians(-51.0)) # ~70,000 at 51°S
> pix_lat_deg = abs(src.res[0])   # pixel height in degrees
> pix_lon_deg = abs(src.res[1])   # pixel width in degrees
>
> dy, dx = np.gradient(dem,
>                       pix_lat_deg * lat_m_per_deg,
>                       pix_lon_deg * lon_m_per_deg)
> slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
> ```

**Terrain derivatives computed:**
| Product | Formula | Use |
|---------|---------|-----|
| Slope | $\arctan(\sqrt{(\partial z/\partial x)^2 + (\partial z/\partial y)^2})$ | Erosion risk, stability |
| Aspect | $\arctan2(-\partial z/\partial x, \partial z/\partial y)$ | Solar radiation, vegetation |
| Hillshade | $\cos(\theta_z) \cdot \cos(slope) + \sin(\theta_z) \cdot \sin(slope) \cdot \cos(\theta_{sun} - aspect)$ | Visualization |
| Curvature | $\partial^2 z / \partial x^2 + \partial^2 z / \partial y^2$ | Flow convergence/divergence |

---

### `11_watershed_delineation.py`
Replicates the ESRI **ArcHydro** workflow in pure Python using the `pysheds` library. Delineates river networks and generates a Hipsometric Curve for basin morphometric analysis.

**D8 Flow Routing Algorithm:**
Each DEM pixel routes water to its single steepest downhill neighbor (8 possible directions). The D8 direction is encoded as powers of 2 (1=E, 2=SE, 4=S, 8=SW, 16=W, 32=NW, 64=N, 128=NE).

**Steps:**
1. **Fill Sinks** — removes DEM depressions that would trap water forever
2. **Flow Direction** — D8 routing to steepest neighbor
3. **Flow Accumulation** — count how many upstream pixels drain to each cell
4. **River Network** — threshold: `accumulation > 1000 pixels` ≈ 0.9 km² contributing area at 30m

**Hipsometric Curve:**
A plot of **normalized elevation** (h/H) vs **fraction of basin area** above that elevation (a/A).

$$H(h) = \frac{A_{above}}{A_{total}}$$

| Curve Shape | Geomorphic Stage | Interpretation |
|-------------|-----------------|----------------|
| Convex (S-shaped) | Young / actively eroding | High mass still in upper basin |
| Straight diagonal | Mature | Balanced erosion across elevations |
| Concave | Old / peneplain | Most mass eroded to low elevations |

**Output:** 2×2 panel chart: DEM, Flow Direction, River Network (log-scale), Hipsometric Curve.

---

## 📐 Key Formulas

| Concept | Formula |
|---------|---------|
| NDSI | $(Green - SWIR) / (Green + SWIR)$ |
| Landsat C2-L2 calibration | $\rho = DN \times 0.0000275 - 0.2$ |
| Slope (radians→degrees) | $s = \arctan\sqrt{(dz/dx)^2 + (dz/dy)^2}$ |
| Drainage Density | $D_d = \Sigma L / A$ |

---

## 🚀 How to Run

### Install Dependencies
```bash
mamba activate geocascade_env
mamba install -n geocascade_env -c conda-forge \
    pystac-client planetary-computer rasterio pysheds \
    numpy matplotlib pyproj -y
```

### Execute Scripts
```bash
# 20-year glacier change detection (Landsat 2003 vs 2023)
python 09_multitemporal_glacier_retreat.py

# DEM derivatives: slope, aspect, hillshade, curvature
python 10_digital_elevation_processing.py

# Watershed delineation + Hipsometric Curve
python 11_watershed_delineation.py
```

---

## 🗺️ GIS Interoperability

**ArcGIS Pro:** Load `glacier_retreat_2003_2023.tif` → Apply classified symbology with custom colormap (cyan=advanced, gray=stable, red=retreated). Use `Calculate Geometry` to quantify area in each class.

**ENVI:** Use `Change Detection` workflow → Load 2003 and 2023 NDSI as separate bands → Apply `Image Differencing` → Classify results to match our retreat map.

> [!TIP]
> The `watershed_analysis.png` Hipsometric Curve can be directly inserted into academic reports. A convex curve over a glaciated Patagonian basin indicates **active glacial erosion** is still reshaping the landscape — the basin has not yet reached geomorphic equilibrium.
