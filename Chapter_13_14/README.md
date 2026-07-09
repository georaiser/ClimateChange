# Chapters 13-14: Advanced Topics — InSAR & Hyperspectral Analysis

> [!NOTE]
> These are advanced topics planned for experienced learners.
> Prerequisites: Chapters 1-12 completed.

---

## Chapter 13: InSAR — Measuring Ground Deformation

### Academic Objective
Use two SAR acquisitions from different dates to measure millimetre-scale ground
deformation through interferometric phase difference analysis.

### Applications
- Glacier surface velocity (ice flow rates in m/day)
- Land subsidence (groundwater extraction, mining, permafrost thaw)
- Earthquake coseismic deformation mapping
- Volcanic inflation/deflation monitoring

### Physics
  phi_interferogram = (4 * pi / lambda) * d_los
  d_los = (lambda / 4*pi) * phi_interferogram

For Sentinel-1 C-band: lambda = 5.547 cm
One interferometric fringe = lambda/2 = ~2.77 cm of line-of-sight deformation.

### Planned Tools
- SNAP (ESA Sentinel Application Platform) — free, GUI + command-line
- ISCE2 (JPL InSAR Scientific Computing Environment) — Python-based
- MintPy — time-series InSAR analysis

---

## Chapter 14: Hyperspectral Analysis

### Academic Objective
Analyse hyperspectral imagery (200+ narrow bands at 5-10nm resolution) for
mineral mapping, crop disease detection, and water quality assessment.

### Contrast with Sentinel-2
| Sensor | Bands | Resolution | Use |
|---|---|---|---|
| Sentinel-2 | 13 bands | 10-60m | General land cover, vegetation |
| AVIRIS-NG | 425 bands | 5nm width | Mineralogy, biochemistry |
| DESIS (ISS) | 235 bands | 2.5nm width | Water quality, urban |

### Spectral Unmixing (Linear Mixing Model)
Most pixels contain mixtures of materials. Unmixing decomposes each pixel into
fractional abundances of pure endmembers:

  R_pixel = sum(f_i * R_endmember_i) + noise
  subject to: sum(f_i) = 1, f_i >= 0

### Planned Tools
- ENVI Spectral Analyst
- scikit-learn NMF (Non-negative Matrix Factorization)
- pysptools (Python spectral analysis)
- USGS Spectral Library (reference endmembers)