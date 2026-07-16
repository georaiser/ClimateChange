# 📡 Chapter 7: SAR Active Microwave Remote Sensing

> **GeoCascade Pipeline — Stage 7**
> Sentinel-1 RTC SAR processing, multi-sensor comparison, and cloud-penetration demonstration.

---

## 📋 Overview

Chapter 7 introduces **Active Remote Sensing** — the most powerful tool for operational environmental monitoring in cloud-dominated regions like Patagonia. Where optical sensors fail, SAR succeeds.

| Script | Topic | Key Outputs |
|--------|-------|-------------|
| 18 | Sentinel-1 SAR: VV, VH, dual thresholding | 5 GeoTIFFs, `sar_statistics.csv`, 5-panel figure |
| 19 | Multi-sensor review: L9 / MODIS / S1 | 3 GeoTIFFs, `multisensor_statistics.csv`, 4-panel figure |
| 19b | Cloud penetration: SAR vs cloudy optical | `sar_vv_db.tif`, `cloud_comparison_stats.csv`, 4-panel figure |

---

## 🚀 Setup

```bash
conda activate geocascade_env

mamba install -n geocascade_env -c conda-forge \
    rasterio numpy matplotlib pandas \
    pystac-client planetary-computer pyproj -y
```

---

## ▶️ Run Order

```bash
# Step 1: Full Sentinel-1 SAR processing (VV, VH, masks, cross-ratio)
python Chapter_07/18_sentinel1_sar_processing.py

# Step 2: Multi-sensor comparison (Landsat 9 / MODIS / Sentinel-1)
python Chapter_07/19_multisensor_review.py

# Step 3: Cloud penetration demonstration (SAR vs winter cloudy optical)
python Chapter_07/19b_cloud_penetration_comparison.py
```

---

## 🔬 Methods Deep-Dive

### SAR Physics

**Why SAR for Patagonia?**
Torres del Paine records cloud cover >80% of the time. Optical satellites (Sentinel-2, Landsat) require cloud-free conditions and daylight. SAR:
- Transmits its own C-band pulse (5.405 GHz, λ = 5.6 cm)
- Completely cloud-independent — images day/night, any weather
- Detects surface ROUGHNESS and DIELECTRIC properties

**Polarization channels:**

| Channel | Physical Mechanism | Sensitive To |
|---------|-------------------|--------------|
| VV | Specular/surface scattering | Water bodies, smooth ice, soil moisture |
| VH | Volume scattering (cross-pol) | Vegetation canopy, forest structure |

**dB conversion (applied every time):**
```python
# Raw RTC product = linear amplitude (gamma0 normalized)
# Negative or zero values = invalid (no physical backscatter)
vv_db = 10 * np.log10(np.where(vv_linear > 0, vv_linear, np.nan))
```

**Dual-threshold classification (Script 18):**

| Threshold | Meaning | Land Cover |
|-----------|---------|-----------|
| VV < -18 dB | Very low backscatter = specular | Flat water / calm lake |
| VV > -5 dB | High backscatter = rough/angular | Glacier crevasses / rocky terrain |
| VH/VV high | Strong depolarization | Dense forest canopy |

**Cross-ratio (CR = VH/VV linear):**
```python
cr = vh_linear / vv_linear   # dimensionless [0-1]
# CR > 0.35 → vegetation canopy (strong volume scattering)
# CR ~ 0.0  → bare surface / water (no depolarization)
```

> [!IMPORTANT]
> The RTC (Radiometrically Terrain Corrected) data on Planetary Computer already has terrain effects removed using the Copernicus DEM. Do NOT apply additional terrain correction.

---

### Multi-Sensor Scale Factors (Script 19)

**Landsat C2-L2 Surface Reflectance (Band 1–7):**
```python
# Fill value = 0 → mask BEFORE applying scale
fill_mask = arr == 0
sr = arr * 0.0000275 - 0.2
sr = np.where(fill_mask, np.nan, sr)
sr = np.clip(sr, 0, 1)   # valid reflectance range
```

**MODIS LST (modis-11A1-061):**
```python
# Fill value = 0 → mask BEFORE scaling (NOT DN < 7500!)
fill_mask = arr == 0
lst_k = arr * 0.02         # → Kelvin
lst_c = lst_k - 273.15     # → Celsius
lst_c = np.where(fill_mask | (lst_k < 150), np.nan, lst_c)
```

> [!WARNING]
> A common error is masking MODIS with `DN < 7500`. The correct fill for `modis-11A1-061` on Planetary Computer is `DN == 0`. Using `< 7500` discards valid cold pixels (glaciers, snow).

---

### Cloud Penetration Comparison (Script 19b)

Script 19b deliberately selects the **cloudiest** available Sentinel-2 scene from June-August (Patagonian winter peak) and the nearest Sentinel-1 acquisition:

```python
# Optical: pick cloudiest for maximum demonstration
item = sorted(items, key=lambda i: i.properties.get("eo:cloud_cover", 0),
              reverse=True)[0]   # HIGHEST cloud cover wins

# SAR: any winter acquisition — always cloud-free by physics
item = items[0]   # any SAR scene is cloud-penetrating
```

The comparison proves the operational advantage: SAR can distinguish water/glacier structure even when optical is 100% obscured.

---

## 📂 Output Structure

```
Chapter_07/
└── data/processed/
    ├── sar/
    │   ├── sar_vv_linear.tif          ← raw gamma0 backscatter
    │   ├── sar_vv_db.tif              ← VV in decibels
    │   ├── sar_vh_db.tif              ← VH in decibels
    │   ├── sar_vv_vh_ratio_db.tif     ← VV-VH ratio (dB)
    │   ├── sar_cr.tif                 ← cross-ratio VH/VV (linear)
    │   ├── water_mask_sar.tif         ← binary water mask
    │   ├── glacier_mask_sar.tif       ← binary glacier/rough mask
    │   └── sar_statistics.csv
    ├── multisensor/
    │   ├── landsat9_nir.tif
    │   ├── modis_lst_celsius.tif
    │   ├── sentinel1_vv_db.tif
    │   ├── multisensor_statistics.csv
    │   └── multisensor_comparison.png
    └── cloud_comparison/
        ├── sar_vv_db.tif
        ├── cloud_comparison_stats.csv
        └── sar_vs_optical_cloudy.png
```

---

## 🖥️ ArcGIS Pro Integration

```
Script 18:
  Add sar_vv_db.tif → Symbology > Stretched > Gray (0 to -25 dB range)
  Raster Calculator: Con("sar_vv_db.tif" < -18, 1, 0) → water binary mask
  Raster Calculator: Con("sar_vv_db.tif" > -5, 1, 0)  → glacier binary mask

Script 19:
  Split View (View > Linked Views): show L9 and SAR side by side
  Use same BBOX to prove cloud cover difference

Script 19b:
  Toggle: add sar_vv_db.tif and Sentinel-2 cloud scene as two layers
  Adjust layer transparency (0-100%) to show SAR reveals ice structure
  that optical cannot see
```

---

## 🔵 ENVI 5.6 Integration

```
; Open SAR VV dB
File > Open > sar_vv_db.tif
Display > Density Slice
  Add break points: -18 dB (water), -5 dB (glacier)

; Cross-polarization ratio
File > Open > sar_cr.tif
Toolbox > Thematic > Threshold Value: 0.35 for vegetation mask

; Compare with optical
Display > Animation
  Load sar_vv_db.tif, then landsat9_nir.tif — toggle to compare
```

---

## ⚠️ Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| No Sentinel-1 found | Winter data gap or STAC outage | Script widens to full-year search as fallback |
| All SAR values NaN | VV linear ≤ 0 before log | Fixed: `np.where(arr > 0, arr, np.nan)` before log |
| MODIS all NaN | Wrong fill threshold | Use `fill_mask = arr == 0`, not `arr < 7500` |
| Landsat very dark | Missing scale factor | Apply `DN * 0.0000275 - 0.2` before visualization |

---

## 📖 Key References

- Ulaby, F.T. et al. (2014). *Microwave Radar and Radiometric Remote Sensing.* University of Michigan Press.
- Torres, R. et al. (2012). *GMES Sentinel-1 mission.* Remote Sensing of Environment.
- Nagler, T. et al. (2016). *The Sentinel-1 Mission: New Opportunities for Ice Sheet Observations.* Remote Sensing.