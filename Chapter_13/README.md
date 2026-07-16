# 🛰️ Chapter 13: Advanced Techniques — InSAR Velocity & Hyperspectral Unmixing

> **GeoCascade Pipeline — Advanced Stage**
> SAR Intensity Cross-Correlation for glacier velocity estimation
> and Linear Spectral Unmixing for sub-pixel land cover fractions.

---

## 📋 Overview

| Script | Topic | Key Outputs |
|--------|-------|-------------|
| `27_insar_glacier_velocity.py` | SAR offset tracking → Grey Glacier velocity (m/year) | `vx_map.tif`, `vy_map.tif`, `vmag_map.tif`, `velocity_statistics.csv`, 4-panel figure |
| `28_hyperspectral_unmixing.py` | Linear Spectral Unmixing (FCLS) of Sentinel-2 | 4 abundance TIFs, `endmember_spectra.csv`, `abundance_statistics.csv`, 5-panel figure |

---

## 🚀 Setup

```bash
conda activate geocascade_env

mamba install -n geocascade_env -c conda-forge \
    rasterio numpy scipy pandas matplotlib -y
```

---

## ▶️ Run Order

```bash
# Step 1: Glacier velocity from SAR cross-correlation
python Chapter_13/27_insar_glacier_velocity.py

# Step 2: Hyperspectral spectral unmixing
python Chapter_13/28_hyperspectral_unmixing.py
```

---

## 🔬 Script 27 — SAR Intensity Cross-Correlation (Offset Tracking)

### Why Not True InSAR?

Full interferometric InSAR for glacier velocity requires:
1. Two Sentinel-1 **SLC** (Single Look Complex) acquisitions separated by 6–12 days
2. SNAP or ISCE software for interferogram formation
3. Phase unwrapping (SNAPHU or py-isce)
4. Atmospheric phase screen correction (ERA5 or PyAPS)

This is a **specialized processing chain** beyond the scope of a Python curriculum.

**Python-native alternative: SAR Intensity Cross-Correlation (Offset Tracking)**
- Correlates two SAR **intensity** images (not complex phase)
- Peak of cross-correlation window = displacement vector
- Displacement / time interval = velocity
- Valid for fast-moving glaciers (>0.5 m/day) like Grey Glacier
- Accuracy: ~1/10 pixel = 1–2 m at 10m Sentinel-1 resolution

### Grey Glacier Velocity Reference Values

Grey Glacier (54°S) is one of the fastest-moving glaciers in Patagonia:
- **Published velocity:** 300–800 m/year (0.8–2.2 m/day)
- At 10m pixel size: 0.08–0.22 pixels/day
- Over a 12-day acquisition interval: 1.0–2.6 pixels shift → **measurable** with sub-pixel correlation

```python
# Cross-correlation workflow
from scipy.signal import correlate2d

def cross_correlate_patch(img1_patch, img2_patch, max_disp):
    """Returns (dx, dy) displacement in pixels."""
    corr = correlate2d(img1_patch, img2_patch, mode="same")
    peak = np.unravel_index(np.argmax(corr), corr.shape)
    cy, cx = corr.shape[0]//2, corr.shape[1]//2
    dy = peak[0] - cy
    dx = peak[1] - cx
    return dx, dy

# Convert pixel displacement → velocity
velocity_m_per_year = displacement_px * PIXEL_SIZE_M * (365.25 / TIME_DELTA_DAYS)
```

### Outputs

| File | Description | Units |
|------|-------------|-------|
| `vx_map.tif` | East-West velocity | m/year |
| `vy_map.tif` | North-South velocity | m/year |
| `vmag_map.tif` | Velocity magnitude | m/year |
| `velocity_statistics.csv` | Mean, max, std of velocity field | m/year |
| `insar_velocity_dashboard.png` | 4-panel: SAR1, SAR2, Vmag, Vx/Vy vector field | — |

> [!NOTE]
> If only one SAR scene is available from Chapter 7, the script synthesizes a second scene
> by applying a small synthetic displacement. This generates a demonstration velocity field
> to illustrate the method even when temporal pairs are unavailable.

> [!IMPORTANT]
> For production glacier velocity: acquire two Sentinel-1 RTC scenes **exactly 12 days apart**
> (one orbital repeat cycle). Greater time gaps reduce correlation quality for fast-moving ice.

---

## 🔬 Script 28 — Linear Spectral Unmixing (FCLS)

### Why Spectral Unmixing?

Every 10m Sentinel-2 pixel in a mountain landscape is typically a **mixture**:
- A glacier edge pixel might be: 40% ice + 35% meltwater + 25% moraine rock
- Standard classification assigns ONE label → ignores sub-pixel heterogeneity
- Spectral unmixing recovers the **fractional abundance** of each land cover type

### Linear Mixing Model (LMM)

$$r = A \cdot x + \epsilon$$

Where:
- $r$ = observed pixel reflectance vector ($B$ bands)
- $A$ = endmember matrix ($B \times N$ pure spectral signatures)
- $x$ = abundance vector ($N$ classes, constraint: $\sum x_i = 1$, $x_i \geq 0$)
- $\epsilon$ = noise residual

**FCLS (Fully Constrained Least Squares):**
```
minimize: ||r - Ax||²
subject to: sum(x) = 1   (partition of unity)
            x ≥ 0        (no negative fractions)
```

Implemented using `scipy.optimize.nnls` (Non-Negative Least Squares) + sum-to-one rescaling.

### Endmembers (Torres del Paine)

| Endmember | Blue | Green | Red | NIR | SWIR | Physical basis |
|-----------|------|-------|-----|-----|------|----------------|
| Glacier/Snow | High | High | High | High | Low | Bright in visible, drops at SWIR |
| Open Water | Low | Low | Low | ~0 | ~0 | Near-zero all bands |
| Dense Vegetation | Low | Moderate | Low | High | Moderate | Red-edge jump |
| Bare Rock/Moraine | Moderate | Moderate | Moderate | Moderate | High | Flat, SWIR reflective |

```python
# Endmember spectra (B02 Blue, B03 Green, B04 Red, B08 NIR, B11 SWIR)
ENDMEMBERS = np.array([
    [0.85, 0.82, 0.78, 0.80, 0.35],   # Glacier/Snow
    [0.04, 0.06, 0.03, 0.02, 0.01],   # Open Water
    [0.03, 0.07, 0.04, 0.45, 0.22],   # Dense Vegetation
    [0.18, 0.20, 0.22, 0.25, 0.38],   # Bare Rock
])
```

### Outputs

| File | Description |
|------|-------------|
| `abundance_glacier.tif` | Fraction of glacier/snow per pixel [0–1] |
| `abundance_water.tif` | Fraction of open water per pixel [0–1] |
| `abundance_vegetation.tif` | Fraction of dense vegetation per pixel [0–1] |
| `abundance_rock.tif` | Fraction of bare rock per pixel [0–1] |
| `endmember_spectra.csv` | Reference endmember reflectance values |
| `abundance_statistics.csv` | Mean/std per endmember across the scene |
| `spectral_unmixing_results.png` | 5-panel: 4 abundance maps + endmember spectra |

> [!NOTE]
> Abundances sum to 1 per pixel. A pixel with `glacier=0.6, water=0.3, rock=0.1` means
> 60% of the 10m pixel area is ice, 30% meltwater pond, 10% exposed moraine.
> This is physically meaningful information that standard classification cannot provide.

---

## 📂 Output Structure

```
Chapter_13/
└── data/processed/
    ├── insar/
    │   ├── vx_map.tif                     ← E-W velocity component (m/year)
    │   ├── vy_map.tif                     ← N-S velocity component (m/year)
    │   ├── vmag_map.tif                   ← Velocity magnitude (m/year)
    │   ├── velocity_statistics.csv
    │   └── insar_velocity_dashboard.png
    └── hyperspectral/
        ├── abundance_glacier.tif
        ├── abundance_water.tif
        ├── abundance_vegetation.tif
        ├── abundance_rock.tif
        ├── endmember_spectra.csv
        ├── abundance_statistics.csv
        └── spectral_unmixing_results.png
```

---

## 🖥️ ArcGIS Pro Integration

```
Script 27 — Glacier Velocity:
  Add vmag_map.tif
  Symbology > Stretched > Yellow-Red (high velocity = red)
  → Compare with RGI 7.0 glacier outlines to validate coverage
  Raster Calculator: Con("vmag_map.tif" > 200, 1, 0)
  → Binary mask of "fast zones" (>200 m/year) for GLOF risk assessment

Script 28 — Spectral Unmixing:
  Add abundance_glacier.tif
  Symbology > Stretched > Blues (high value = more ice)
  Add abundance_vegetation.tif
  Symbology > Stretched > Greens

  Raster Calculator: "abundance_glacier.tif" * "abundance_water.tif"
  → High values = mixed glacier/meltwater pixels = most dynamically active
  Spatial Analyst > Focal Statistics: Mean 3×3 on abundance_glacier.tif
  → Smoothed abundance map reduces pixel noise
```

---

## 🔵 ENVI 5.6 Integration

```
; Script 27 — Velocity
File > Open > vmag_map.tif
Toolbox > SAR > Offset Tracking (compare with script 27 output)
  → Use ENVI's built-in offset tracking to validate Python results

; Script 28 — Unmixing
Toolbox > Spectral > Linear Spectral Unmixing
  Select your 5-band Sentinel-2 image
  Load endmembers from endmember_spectra.csv
  → Compare ENVI FCLS with script 28 output
  Expected: near-identical abundances (both use NNLS)

; Validation
Toolbox > Raster Management > Band Math
  (b1 + b2 + b3 + b4)  ; should equal ~1.0 everywhere (sum-to-one constraint)
```

---

## ⚠️ Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| `SAR file not found` | Ch07 not run yet | `python Chapter_07/18_sentinel1_sar_processing.py` |
| All velocities = 0 | Synthetic scenes identical | Acquire real 12-day temporal pair from Planetary Computer |
| Abundances don't sum to 1 | Constraint not applied | Check NNLS rescaling: `x = x / x.sum()` |
| `scipy` ImportError | scipy not installed | `mamba install -n geocascade_env -c conda-forge scipy -y` |

---

## 📖 Key References

- Mouginot, J. et al. (2012). *Mapping of ice motion in Antarctica using synthetic-aperture radar data.* Remote Sensing.
- Nagler, T. et al. (2015). *The Sentinel-1 Mission: SAR for ice and snow.* IEEE Geoscience.
- Keshava, N., Mustard, J.F. (2002). *Spectral unmixing.* IEEE Signal Processing Magazine.
- Hapke, B. (2012). *Theory of Reflectance and Emittance Spectroscopy.* Cambridge University Press.