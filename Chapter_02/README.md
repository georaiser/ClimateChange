# Chapter 2: Spectral Signature Analysis

## Academic Objective
Every material on Earth reflects electromagnetic radiation differently across the spectrum.
This chapter demonstrates cloud-native pixel extraction (streaming directly from COG URLs),
computes 7 vegetation/soil indices, and interprets the physics behind each band ratio.

---

## Scripts

### 06_spectral_signature_analysis.py -- Cloud-Native Spectral Extraction

Extracts per-material reflectance from Sentinel-2 L2A. No full download -- only exact
pixels are streamed via Cloud Optimized GeoTIFF HTTP range requests.

**Materials sampled (4 classes):**
- Glacial Ice (Grey Glacier) -- lat: -51.010, lon: -73.230
- Patagonian Forest -- lat: -51.150, lon: -72.950
- Bare Rock / Scree -- lat: -50.900, lon: -72.900
- **NEW** Open Water (Grey Lake) -- lat: -51.050, lon: -73.180

**Bands (7 total, B05 Red Edge added):**

| Band | Wavelength | Physical Meaning |
|---|---|---|
| B02 | 490 nm | Blue -- atmosphere, deep water |
| B03 | 560 nm | Green -- vegetation peak, NDWI |
| B04 | 665 nm | Red -- chlorophyll absorption |
| **B05** | **705 nm** | **Red Edge -- early vegetation stress** |
| B08 | 842 nm | NIR -- leaf structure, NDVI |
| B11 | 1610 nm | SWIR1 -- soil moisture, NDSI |
| B12 | 2190 nm | SWIR2 -- clay minerals, BSI |

> [!TIP]
> Red Edge (B05, 705nm) is unique to Sentinel-2. Vegetation stress narrows the
> red edge BEFORE NDVI shows any change -- making it an earlier warning indicator.

**Improvements:** B05 added, Water sample point, sorted by cloud cover,
red-edge region shaded on plot, exports spectral_signatures.csv, plt.close().

Run: python 06_spectral_signature_analysis.py

Output: spectral_signatures.png + spectral_signatures.csv

---

### 07_vegetation_soil_indices.py -- 7-Index Spectral Suite

Computes indices with safe_ratio() helper (epsilon denominator guard):

| Index | Formula | Use |
|---|---|---|
| NDVI | (NIR-Red)/(NIR+Red) | Vegetation health |
| EVI | 2.5*(NIR-Red)/(NIR+6*Red-7.5*Blue+1) | Dense canopy |
| SAVI | (NIR-Red)/(NIR+Red+0.5)*1.5 | Sparse vegetation, soil background |
| BSI | ((SWIR+Red)-(NIR+Blue))/((SWIR+Red)+(NIR+Blue)) | Bare soil, urban |
| NDWI | (Green-NIR)/(Green+NIR) | Open water |
| NDSI | (Green-SWIR)/(Green+SWIR) | Snow and ice (>0.4 = glacier) |
| NDGI | (Green-Red)/(Green+Red) | Green vs urban contrast |

> [!WARNING]
> B11 (SWIR1) is 20m native -- must use its OWN independently computed window,
> NOT the 10m NIR window. Reusing B08 window on a 20m grid reads the wrong area.

Run: python 07_vegetation_soil_indices.py

---

### 08_automated_index_batcher.py -- Multi-Date Batch Processor
Runs the full 7-index suite across multiple acquisition dates.

Run: python 08_automated_index_batcher.py

---

## Key Concepts

- Spectral Signature: unique reflectance fingerprint across wavelengths
- Red Edge saturation: NDVI fails in dense canopy (>0.8); EVI corrects this
- Safe ratio: (a-b)/(a+b+epsilon) avoids divide-by-zero
- L2A scale factor: Raw DN / 10000 = Physical Reflectance [0,1]
- COG: Cloud Optimized GeoTIFF -- tiled for HTTP streaming

## Installation

`ash
mamba install -n geocascade_env -c conda-forge pystac-client planetary-computer rasterio numpy matplotlib pyproj pandas -y
`
