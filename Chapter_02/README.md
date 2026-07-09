# 📡 Chapter 2: Spectral Analysis & Index Automation

> **GeoCascade** | Sentinel-2 L2A · Spectral Signatures · Remote Sensing Indices · Batch Automation

---

## 📋 Scripts Overview

### 1. `06_spectral_signature_analysis.py` — Spectral Signature Analysis

Downloads Sentinel-2 L2A imagery and samples spectral signatures across **5 land cover types**:

| Class | Description |
|---|---|
| 🌿 Dense Vegetation | Forest / dense canopy |
| 💧 Open Water | Lakes / rivers |
| 🧊 Glacier Ice | Permanent snow/ice |
| 🪨 Rock / Bare Soil | Exposed geology |
| 🌱 Sparse Vegetation | Shrubland / meadow |

**Bands read:** B02 (Blue), B03 (Green), B04 (Red), B05 (Red Edge), B08 (NIR), B11 (SWIR1), B12 (SWIR2)

> [!IMPORTANT]
> B05 Red Edge is the **7th band** in the band list (index position 6), not the 5th. Confirm band ordering when slicing arrays.

**Outputs:**
- Reflectance curve plot per land cover class
- CSV with mean reflectance per class per band
- Cloud sort applied — cleanest scene selected automatically

**Run:**
```bash
python 06_spectral_signature_analysis.py
```

---

### 2. `07_vegetation_soil_indices.py` — Vegetation & Soil Indices *(UPDATED)*

Computes **7 spectral indices** from Sentinel-2 bands:

| Index | Formula | Sensitivity |
|---|---|---|
| NDVI | $(NIR - Red) / (NIR + Red)$ | Vegetation health |
| EVI | $2.5 \cdot (NIR - Red) / (NIR + 6 \cdot Red - 7.5 \cdot Blue + 1)$ | Canopy structure |
| SAVI | $(NIR - Red) / (NIR + Red + L) \cdot (1 + L)$ | Vegetation over bare soil |
| BSI | $((SWIR1 + Red) - (NIR + Blue)) / ((SWIR1 + Red) + (NIR + Blue))$ | Bare soil exposure |
| NDWI | $(Green - NIR) / (Green + NIR)$ | Open water |
| NDSI | $(Green - SWIR1) / (Green + SWIR1)$ | Snow / glacier ice |
| NDGI | $(Green - Red) / (Green + Red)$ | Greenness |

> [!IMPORTANT]
> **Critical B11 Window Bug Fix** — B11 (SWIR1) is a **20 m** band. Reusing the 10 m B02 window to read B11 silently reads the **wrong geographic area** because the pixel grid is different. B11 must use its **own independent window** computed from the 20 m source transform:
> ```python
> # ✅ CORRECT — independent window from 20m transform
> window_20m = rasterio.windows.from_bounds(*bounds, transform=src_20m.transform)
> b11 = src_20m.read(1, window=window_20m)
>
> # ❌ WRONG — reusing 10m window for a 20m band
> b11 = src_20m.read(1, window=window_10m)
> ```

**Additional fixes:**
- `nodata=-9999` (universally readable by ArcGIS, ENVI, QGIS — avoids NaN bit-pattern issues)
- Summary statistics per index
- `plt.close()` after each figure to prevent memory leaks

**Run:**
```bash
python 07_vegetation_soil_indices.py
```

---

### 3. `08_automated_index_batcher.py` — Automated Index Batcher *(UPGRADED)*

Batch processes **all cloud-free Sentinel-2 scenes in 2023**, computing NDVI + NDSI + NDWI per scene.

**Key upgrades:**

| Feature | Description |
|---|---|
| `safe_ratio()` | NaN (not zero) for water/shadow — never divides by zero |
| B11 independent window | Correct geographic alignment for 20 m SWIR bands |
| `nodata=-9999` | Universally portable nodata value |
| Cloud sort | Cleanest scene always processed first |
| Skip-if-exists | Resumable runs — already processed scenes skipped |
| Summary CSV | Per-scene statistics exported |
| Annual time-series | 3-panel plot across all 2023 scenes |

**`safe_ratio()` implementation:**
```python
def safe_ratio(num, den, eps=1e-6):
    """Returns NaN where denominator is near-zero — never divide directly."""
    return np.where(np.abs(den) < eps, np.nan, num / den)
```

**Run:**
```bash
python 08_automated_index_batcher.py
```

---

## 🛠️ Installation

```bash
mamba install -n geocascade_env -c conda-forge \
    pystac-client planetary-computer rasterio \
    pyproj numpy pandas matplotlib -y
```

---

## 🧠 Key Concepts

### Sentinel-2 Band Resolutions

| Resolution | Bands |
|---|---|
| **10 m** | B02 (Blue), B03 (Green), B04 (Red), B08 (NIR) |
| **20 m** | B05, B06, B07 (Red Edge), B11, B12 (SWIR) |

> [!WARNING]
> **Reading a 20 m band with a 10 m window silently returns wrong pixels.** Always compute the window from the band's own native transform. This is one of the most common silent data corruption bugs in multi-resolution Sentinel-2 workflows.

### `safe_ratio()` — Safe Division

```python
# ✅ Correct: NaN where denominator ≈ 0
result = np.where(np.abs(den) < 1e-6, np.nan, num / den)

# ❌ Wrong: Zero denominator maps to 0 (masks water, shadow, etc.)
result = np.where(den != 0, num / den, 0)
```

### Index Thresholds

| Index | Threshold | Meaning |
|---|---|---|
| NDSI | > 0.4 | Glacier / snow |
| NDWI | > 0.3 | Open water |
| NDVI | > 0.3 | Active vegetation |

### `nodata=-9999` vs `np.nan`

> [!NOTE]
> `rasterio` writes `np.nan` as arbitrary bit patterns on some systems. `-9999` is a universally readable nodata sentinel across **ArcGIS, ENVI, and QGIS** and should always be preferred when writing GeoTIFFs.

### Batch Automation Pattern

- **Cloud sort** → ensures the cleanest scene is always processed first
- **Skip-if-exists** → makes batch runs safely resumable after interruption

---

## 📁 Expected Outputs

```
Chapter_02/
├── outputs/
│   ├── spectral_signatures.csv        # Mean reflectance per class per band
│   ├── spectral_signature_plot.png    # Reflectance curves by land cover
│   ├── index_<scene_id>.tif           # Per-scene index GeoTIFFs
│   ├── batch_summary.csv              # Annual statistics summary
│   └── annual_timeseries.png          # 3-panel NDVI/NDSI/NDWI time series
```
