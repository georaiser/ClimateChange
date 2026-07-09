# Chapter 2: Spectral Signature Analysis

## 🎯 Academic Objective

Every material on Earth — glacier ice, healthy forest, bare rock, open water — has a unique **spectral fingerprint**: a characteristic pattern of how much light it reflects at each wavelength. This chapter teaches you to read these fingerprints and convert them into quantitative indices that satellites measure operationally.

By the end of this chapter you will be able to:
- Extract and visualize spectral signature curves for 5 material classes
- Compute a 7-index spectral suite (NDVI, EVI, SAVI, BSI, NDWI, NDSI, NDGI)
- Understand why each index was designed and what physical property it targets
- Batch-process indices across multiple dates to detect temporal change

---

## 🛠️ Scripts & Modules

### `06_spectral_signature_analysis.py`
Extracts mean reflectance per material class across all Sentinel-2 bands (B01–B12) and plots spectral signature curves.

**Material classes sampled:**
| Class | Spectral Behavior |
|-------|-------------------|
| Glacier / Snow | Very high reflectance in Visible, drops sharply in SWIR |
| Healthy Vegetation | Low Red, very high NIR (red edge), moderate SWIR |
| Open Water | Moderate Blue/Green, absorbs NIR and SWIR completely |
| Bare Rock | Moderate and rising reflectance across all bands |
| Bare Soil | Similar to Rock but with higher SWIR |

> [!NOTE]
> **Sentinel-2 L2A Scale Factor:** All band values must be divided by **10,000** before use:
> ```python
> reflectance = src.read(1).astype('float32') / 10000.0
> ```
> Raw DNs are stored as integers in the range 0–10000 to save disk space. Forgetting this step makes all index values nonsensical.

---

### `07_vegetation_soil_indices.py`
Computes a complete 7-index spectral suite with a `safe_ratio()` helper that prevents divide-by-zero errors on water/shadow pixels.

```python
def safe_ratio(num, den, fill=np.nan):
    return np.where(np.abs(den) < 1e-6, fill, num / den)
```

#### Index Formulas & Physical Interpretation

**① NDVI — Normalized Difference Vegetation Index**
$$NDVI = \frac{NIR - Red}{NIR + Red}$$
- Range: [-1, +1]. Dense forest ≈ 0.6–0.9, sparse grass ≈ 0.2–0.4, water/snow ≈ negative
- **Limitation:** Saturates in dense canopies (all values > 0.8 look the same). Use EVI for tropical forests.

**② EVI — Enhanced Vegetation Index**
$$EVI = 2.5 \cdot \frac{NIR - Red}{NIR + 6 \cdot Red - 7.5 \cdot Blue + 1}$$
- Reduces soil and atmospheric noise via the soil correction factor (6) and atmosphere resistance (7.5, Blue)
- Developed by MODIS team; does not saturate in dense canopies

**③ SAVI — Soil-Adjusted Vegetation Index**
$$SAVI = \frac{(NIR - Red)}{(NIR + Red + L)} \cdot (1 + L), \quad L = 0.5$$
- $L = 0.5$ is optimal for intermediate vegetation density; $L = 1$ for very sparse, $L = 0$ → NDVI
- Reduces soil brightness effects in arid and semi-arid environments

**④ BSI — Bare Soil Index**
$$BSI = \frac{(SWIR + Red) - (NIR + Blue)}{(SWIR + Red) + (NIR + Blue)}$$
- Positive BSI → exposed soil or urban surfaces. Negative → vegetated or water surfaces.
- Used in desertification monitoring and urban expansion mapping

**⑤ NDWI — Normalized Difference Water Index**
$$NDWI = \frac{Green - NIR}{Green + NIR}$$
- Positive values indicate open water bodies (lakes, rivers, flooded areas)
- Negative values indicate land. Threshold ≈ 0.0 separates water from land.

**⑥ NDSI — Normalized Difference Snow Index**
$$NDSI = \frac{Green - SWIR}{Green + SWIR}$$
- NDSI > 0.4 → confirmed glacier or snow cover
- Snow/ice has high Green reflectance but absorbs SWIR strongly — unique spectral behavior
- Same Green band as NDWI, but uses SWIR (not NIR) as the reference

**⑦ NDGI — Normalized Difference Greenness Index**
$$NDGI = \frac{Green - Red}{Green + Red}$$
- Sensitive to green biomass, even where NDVI saturates
- Good for urban-vegetation contrast mapping and phenology studies

> [!WARNING]
> **B11 (SWIR) Resolution Mismatch:** B11 is natively **20m** while optical bands (B03, B04, B08) are **10m**. When computing BSI or NDSI you must upsample B11 to 10m using `out_shape`:
> ```python
> # ✅ Correct — separate window computed from B11's own transform
> with rasterio.open(item.assets["B11"].href) as src_swir:
>     win_swir = from_bounds(minx, miny, maxx, maxy, src_swir.transform)
>     swir = src_swir.read(1, window=win_swir,
>                          out_shape=target_shape,
>                          resampling=Resampling.bilinear)
>
> # ❌ Wrong — reusing B08's window on B11 gives wrong spatial extent
> swir = src_swir.read(1, window=win_nir)  # NEVER do this
> ```

**Output:** 7-panel 2×4 matplotlib figure + 7 individually geocoded GeoTIFFs, one per index.

---

### `08_automated_index_batcher.py`
Runs the full 7-index suite across a list of acquisition dates and saves results organized by date. Useful for detecting:
- Seasonal NDVI cycles (growing season onset/offset)
- Post-fire NDVI drops (using dNBR-like change)
- Inter-annual glacier extent change via NDSI

---

## 📐 Index Quick-Reference

| Index | Bands Used | Target Property | Threshold |
|-------|-----------|----------------|-----------|
| NDVI | NIR, Red | Vegetation health | > 0.3 = vegetation |
| EVI | NIR, Red, Blue | Dense canopy | > 0.3 = moderate vegetation |
| SAVI | NIR, Red | Sparse vegetation | > 0.2 = vegetated |
| BSI | SWIR, Red, NIR, Blue | Bare soil / urban | > 0.0 = bare |
| NDWI | Green, NIR | Open water | > 0.0 = water |
| NDSI | Green, SWIR | Snow & glacier | > 0.4 = ice/snow |
| NDGI | Green, Red | Green biomass | > 0.1 = green veg |

---

## 🚀 How to Run

### Install Dependencies
```bash
mamba activate geocascade_env
mamba install -n geocascade_env -c conda-forge \
    pystac-client planetary-computer rasterio numpy matplotlib pyproj -y
```

### Execute Scripts
```bash
# Spectral signature profiles for 5 material classes
python 06_spectral_signature_analysis.py

# Full 7-index spectral suite (7 TIFFs + 7-panel chart)
python 07_vegetation_soil_indices.py

# Batch processor across multiple dates
python 08_automated_index_batcher.py
```

---

## 🗺️ GIS Interoperability

**ArcGIS Pro:** Load individual index TIFFs → use `Raster Calculator` to compare two dates → compute difference raster for change detection.

**ENVI:** Load the 7-band stack → use `Band Math` to recompute any index → apply `Spectral Profile Tool` to compare signatures from field samples vs satellite values.

> [!TIP]
> In ArcGIS Pro's **Image Classification Wizard**, you can use the 7 index TIFFs as input variables for supervised or unsupervised classification — this is the equivalent of what Chapter 4 does programmatically with scikit-learn.
