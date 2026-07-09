# Chapter 7: SAR Processing & Cloud Penetration Analysis

## Academic Objective
Introduce Active Remote Sensing (SAR). Process both VV and VH polarization channels
from Sentinel-1 RTC data, compute the VV/VH cross-polarization ratio as a surface
discriminator, and demonstrate the all-weather advantage of SAR over optical sensors.

---

## Scripts

### 18_sentinel1_sar_processing.py — Dual-Polarization SAR Analysis (MAJOR UPGRADE)

**Previously:** VV only, 3-panel figure.
**Now:** Both VV and VH polarizations + VV/VH ratio, 4-panel figure, area report.

Physics:
  sigma0_dB = 10 * log10(sigma0_linear)
  VV/VH_ratio_dB = VV_dB - VH_dB

| Polarization | Sensitivity |
|---|---|
| VV (Vertical-Vertical) | Surface roughness, soil moisture, water, ice |
| VH (Vertical-Horizontal) | Volume scattering: vegetation canopy, forest structure |

VV/VH Ratio interpretation:
| Ratio | Surface Type |
|---|---|
| ~0 dB | Double-bounce: urban structures, rough ice crevasses |
| >> 0 dB | Volume scattering: dense vegetation, forest |
| << 0 dB | Specular surface: water, smooth ice lake |

Thresholds:
- Water mask: VV < -18 dB (specular reflection off smooth water)
- Glacier mask: VV > -5 dB (volume/double-bounce from rough crevassed ice)

**NEW Tier 3 — Quantitative Area Report:**
Prints water_pixels and glacier_pixels with approximate km2 area.

**Improvements:** read_pol() helper; VH polarization download; VV/VH ratio raster;
nodata=-9999; int(round()) window; plt.close(); STAC guard.

Run: `python 18_sentinel1_sar_processing.py`

Outputs: sar_vv_linear.tif, sar_vv_db.tif, sar_vv_vh_ratio_db.tif, sar_dual_analysis.png

---

### 19_multisensor_review.py — Multi-Temporal SAR Review

Multi-temporal SAR comparison script showing backscatter changes over time.

Run: `python 19_multisensor_review.py`

---

### 19b_cloud_penetration_comparison.py — SAR vs Optical (NEW Tier 4)

Demonstrates the cloud penetration advantage of SAR by deliberately selecting
a CLOUDY winter Sentinel-2 scene alongside the SAR acquisition from the same week.

The 3-panel comparison shows:
1. SAR VV (always clear — cloud-independent)
2. Sentinel-2 true color (cloud-obscured, picked from winter high-cloud season)
3. VV/VH ratio (structural discriminator: vegetation / soil / water)

Strategy: picks the CLOUDIEST available scene (sorted by cloud cover descending)
to maximize the demonstration of optical sensor limitations.

Run: `python 19b_cloud_penetration_comparison.py`

Outputs: sar_vs_optical_cloudy.png

---

## Key Concepts

| Concept | Explanation |
|---|---|
| Active vs Passive | SAR transmits its own microwave pulse; optical sensors use sunlight |
| Cloud Penetration | Microwaves (3-25cm) pass through clouds, rain, and darkness |
| RTC | Radiometric Terrain Correction — removes terrain-induced SAR distortions |
| VV polarization | Sensitive to surface roughness and smooth water surfaces |
| VH polarization | Sensitive to volume scattering from vegetation canopy |
| VV/VH ratio | Surface type discriminator: forest vs water vs ice |
| dB scale | Compresses the dynamic range of radar backscatter for visualization |

## Installation

```bash
mamba install -n geocascade_env -c conda-forge pystac-client planetary-computer rasterio numpy matplotlib pyproj -y
```