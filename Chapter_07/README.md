# Chapter 7: Multi-Sensor Remote Sensing & Radar (SAR)

## 🎯 Academic Objective
So far, this curriculum has focused exclusively on **Passive Optical Remote Sensing** (Sentinel-2, Landsat). Passive sensors rely on the sun's illumination to reflect off the Earth and into the camera. The major weakness of optical imagery is that it cannot penetrate clouds, and it cannot see at night. 

In this chapter, we introduce **Active Remote Sensing**, specifically **Synthetic Aperture Radar (SAR)** using the European Space Agency's **Sentinel-1** satellite.

### The Physics of Radar (C-Band SAR)
An active sensor emits its own microwave energy and measures the "backscatter" (the energy that bounces back to the satellite). Because microwaves are much longer than optical light waves, they physically pass right through water vapor, clouds, and smoke!

Radar measures **Surface Roughness** and **Dielectric Constant (Moisture)**:
1.  **Smooth Water (Floods/Lakes):** Acts like a mirror. The radar beam hits the flat water and reflects *away* from the satellite. The backscatter is very low, making water appear pitch black in radar images.
2.  **Rough Surfaces (Glaciers, Urban Areas, Forests):** The radar beam bounces off crevasses, tree branches, and buildings in all directions (volume scattering). Much of the energy returns to the satellite, making these areas appear bright white.

---

## 🛠️ Scripts & Modules

### `18_sentinel1_sar_processing.py`
- **Concept:** Sentinel-1 measures two polarizations: **VV** (Vertical transmit, Vertical receive) and **VH** (Vertical transmit, Horizontal receive). Raw radar amplitude is difficult to interpret, so we mathematically convert the linear amplitude into **Decibels (dB)** using logarithmic scaling: $dB = 10 \cdot \log_{10}(\text{pixel})$.
- **Action (Python Automation):** Downloads Sentinel-1 Radiometric Terrain Corrected (RTC) data via the STAC API. Converts the VV and VH bands to Decibels. 
- **Dual-Thresholding (Flood vs Glacier):** The script applies logical thresholds to mathematically isolate deep water (Flood Mapping, dB < -18) and rough glacial ice (Glacier Mapping, dB > -5) in the same script, proving radar's extreme versatility.

### `19_multisensor_review.py`
- **Concept:** A Geospatial Data Scientist must choose the correct tool for the job. 
- **Action:** This script queries three completely different satellites for the exact same region and timeframe:
  1.  **Landsat 9 (Collection 2 Level-2):** Optical (30m). Excellent for calculating NDVI and true-color mapping, but completely blocked by clouds.
  2.  **MODIS (LST_Day_1km):** Thermal (1km). Low spatial resolution, but perfect for measuring macro-scale Land Surface Temperature (LST) variations.
  3.  **Sentinel-1 (RTC):** Radar (10m). Impervious to clouds and darkness. Excellent for structural mapping (glaciers) and water extraction.
- **Output:** It generates a multi-sensor comparative plot, allowing you to visually analyze the exact same territory through three different regions of the electromagnetic spectrum.

---

## 🚀 How to Run

### 1. Install Dependencies
You can use our standard environment. No extra packages are needed for this chapter!
```bash
mamba activate geocascade_env
```

### 2. Process Sentinel-1 Radar
```bash
python 18_sentinel1_sar_processing.py
```

### 3. Run Multi-Sensor Review (Landsat 9 vs MODIS vs S1)
```bash
python 19_multisensor_review.py
```

---

## 🗺️ GIS Interoperability (ArcGIS Pro & ENVI)
Because Radar analysis is highly complex in GUI software (requiring SNAP or specific ENVI SAR modules to perform radiometric calibration and speckle filtering), our Python pipeline does the heavy lifting for you. 

The scripts automatically output **Geocoded TIFFs** (e.g., `sar_vv_db.tif`) into the `data/processed/` folder. You can drag and drop these directly into ArcGIS Pro or ENVI, apply a standard black-and-white color ramp, and instantly overlay them with your optical datasets!
