# Chapter 1: Climatic Variables & Image Processing

## 🎯 Academic Objective
The goal of this chapter is to establish a strong foundation in programmatic geospatial data acquisition, atmospheric correction, and climate modeling. 

Rather than manually clicking through web portals (like EarthExplorer or Copernicus Browser), we will build scripts that act as automated data pipelines. We will transform raw satellite imagery into **Analysis-Ready Data (ARD)** and use Machine Learning to interpolate local weather station records into continuous spatial maps.

---

## 🛠️ Scripts & Modules

### `01_stac_multisensor_download.py`
- **Concept:** Spatiotemporal Asset Catalogs (STAC) are the modern standard for querying Earth Observation data via APIs. This avoids downloading entire satellite tiles when you only need a specific bounding box (BBOX).
> [!WARNING]
> **Academic Trap: L1C vs L2A Data**
> *   **Level-1C (L1C):** Top of Atmosphere (TOA) Reflectance. This data still contains atmospheric haze. You *must* download L1C if you intend to perform Atmospheric Correction (FLAASH or COST) yourself.
> *   **Level-2A (L2A):** Bottom of Atmosphere (BOA) Surface Reflectance. This data has *already* been atmospherically corrected by the European Space Agency using Sen2Cor. If you download L2A, applying FLAASH or COST will **double-correct** the atmosphere, rendering your data scientifically invalid! If you have L2A data, skip atmospheric correction entirely and jump straight to index calculation.
> *   *Note: Because Microsoft Planetary Computer recently deprecated their global L1C archive, our automated STAC scripts default to downloading **L2A** data. If you want to practice atmospheric correction, you must manually download Level-1 data.*

> [!TIP]
> **Manual Download Instructions for Atmospheric Correction Practice:**
> If you want to run `01b_envi_flaash_prep.py` or `02_atmospheric_correction.py`, you need Top of Atmosphere (Level-1) data. 
> - **Sentinel-2 L1C:** Go to the [Copernicus Data Space Ecosystem (CDSE)](https://dataspace.copernicus.eu/), search for Torres del Paine, download an L1C `.zip` file, extract the `.jp2` bands, and place them in the `data/raw/sentinel2_l1c_manual/` folder.
> - **Landsat 8/9 L1:** Go to [USGS EarthExplorer](https://earthexplorer.usgs.gov/), download a Collection 2 Level-1 `.tar.gz` file, extract the `.TIF` bands, and process them identically to the Sentinel pipeline.

- **Academic Note on Band Selection:** Sentinel-2 captures 13 spectral bands, but this script intentionally downloads only 5 core bands for efficiency: **B02 (Blue), B03 (Green), B04 (Red), B08 (Near-Infrared), and B11 (Shortwave Infrared)**. These bands contain 95% of the information needed for standard optical climate analysis (vegetation, water, and glaciers) and allow for dynamic K-T aerosol retrieval during atmospheric correction, all while reducing storage footprint.
  *   *Spatial Resolution:* The Visible and NIR bands are natively **10-meter** resolution, providing the highest precision. Band 11 is natively **20-meter**, which we programmatically downsample to 10m to match.
  *   *Index Calculation:* Red and NIR are the core bands required to compute spectral indices like NDVI (vegetation) and NDWI (glacial lakes). Band 11 is used for NDSI (snow) and advanced atmospheric modeling.
- **Action:** A pure Python script utilizing `pystac-client` and `planetary-computer` (or AWS Earth Search) to search and download Sentinel-2, Landsat, and DEM data for the Torres del Paine Region of Interest (ROI). 
- **Dependencies:** `pystac-client`, `planetary-computer`, `rasterio`

### `01c_landsat_download.py`
- **Concept:** Adapting our STAC query methodology to a different satellite constellation (NASA/USGS Landsat 8/9).
- **Academic Concept: Pre-Corrected Data:** This script intentionally searches for `landsat-c2-l2` (Collection 2, Level-2) data, which is already Surface Reflectance. By downloading L2 data, we can completely bypass the FLAASH/COST atmospheric correction pipeline and proceed directly to analysis. Microsoft's Planetary Computer does not host Landsat Level-1 data via STAC.
- **The Physics are Identical:** If you were to manually download Landsat Level-1 (Top of Atmosphere) data from the USGS EarthExplorer portal, the atmospheric correction process is exactly the same! Because physics is sensor-agnostic, you could run the exact same `02_atmospheric_correction.py` (COST model) on the Landsat bands, and it would work perfectly. If you wanted to use ENVI FLAASH, you would simply change the FWHM and Wavelength values in the `.hdr` file to match the Landsat 8 sensor specifications instead of Sentinel-2.
- **Action:** Downloads the Landsat equivalents of the Sentinel-2 bands (Blue, Green, Red, NIR, SWIR1) for the same ROI and date range.

### `02_atmospheric_correction.py`
- **Concept:** Sensors record Top of Atmosphere (TOA) reflectance, which includes atmospheric scattering (haze) and absorption. True surface analysis requires Bottom of Atmosphere (BOA) reflectance.
- **Action:** A Python implementation of the **COST Model (Chavez, 1996)**. It mathematically corrects TOA to BOA by calculating the 1st percentile of dark pixels to robustly subtract atmospheric path radiance (haze), and then queries the STAC API for the Solar Zenith angle to estimate atmospheric transmittance, correcting for absorption.
  * **Physics Note on Band Exclusion:**
    * **Applies COST Model:** `B01` through `B08A`. (Visible and Near-Infrared light is heavily affected by Rayleigh/Mie scattering, so we subtract the dark object to remove the haze and divide by transmittance).
    * **Skips DOS1 (Direct Copy):** B09, B10, B11, B12.
        * *Why B11 & B12 (SWIR)?* Wavelengths are too large; atmospheric path radiance is physically negligible.
        * *Why B09 (Water Vapor)?* This band is specifically designed to measure atmospheric water absorption. Normalizing it to the surface corrupts it.
        * *Why B10 (Cirrus)?* This band is completely absorbed by the lower atmosphere and only sees high-altitude clouds. It has no "surface" reflectance to correct!

### `01b_envi_flaash_prep.py`
- **Concept:** Preparing remote sensing data for ENVI FLAASH is incredibly tedious via the GUI. You must manually resample bands to matching pixel sizes, layer stack them, and convert them to BIL interleave format.
- **Action:** A pure Python script that reads the downloaded 10m bands (B02, B03, B04, B08) and the 20m SWIR band (B11), mathematically resamples B11 to 10m using Bilinear interpolation, stacks them, and exports a ready-to-use `.dat` file directly in ENVI's native BIL format.
### Academic Concept: Reflectance vs Radiance
In Remote Sensing, these two properties define how we process imagery:
*   **Radiance** is the absolute amount of light energy hitting the sensor from a specific target, measured in physical units like $\mu W/(cm^2*sr*nm)$. It is heavily influenced by the angle of the sun and the Earth's distance from it.
*   **Reflectance** is a unitless ratio (usually 0.0 to 1.0) of how much light a surface reflects compared to how much it receives.

Our STAC script downloads Top of Atmosphere (TOA) **Reflectance**.
*   **ENVI FLAASH** strictly requires **Radiance** because it runs a physical radiative transfer model (MODTRAN). If you feed it Reflectance, it thinks the Earth is pitch black and crashes (the infamous `ACC_KTAEROSOL` crash).
*   **ENVI QUAC** is an empirical model designed to work seamlessly with either **Reflectance** or **Radiance**.

### Academic Concept: Model-Based vs Empirical Atmospheric Correction
In the academic world of atmospheric correction, algorithms fall into two main branches:
*   **Model-Based (FLAASH):** Uses quantum physics, solar geometry, and atmospheric gas simulations (MODTRAN) to calculate exactly how light scatters. *Requires Radiance.*
*   **Empirical / Image-Based (DOS1, COST, and QUAC):** Uses statistics from the pixels inside the image itself to guess the atmospheric interference. *Works perfectly with Reflectance.*
    *   *Our Python script `02_atmospheric_correction.py` uses the **COST Model (Chavez, 1996)**, which is an advanced version of DOS1. While basic DOS1 only subtracts haze, the COST model dynamically queries the STAC API for the Solar Zenith Angle to estimate atmospheric transmittance (absorption). This division mathematically mimics FLAASH, allowing our Python script to accurately restore the brightness of highly reflective surfaces like snow and rock!* 
    *   *QUAC is an even more advanced empirical model. Instead of looking for one dark pixel, QUAC looks at the histogram of the entire image and forces the average reflectance of the scene to match a known universal baseline of "average Earth materials."*

### `01b_envi_flaash_prep.py`
- **Concept:** Preparing remote sensing data for ENVI FLAASH is incredibly tedious via the GUI. Furthermore, because FLAASH requires Radiance, we cannot feed our downloaded Reflectance directly into it.
- **Action:** A pure Python script that acts as an automated Radiometric Calibration and formatting engine.
- **The Physics Engine in the Script:**
  1. It reads the folder name of your downloaded image, connects back to the STAC API, and fetches the exact **Acquisition Date** and **Mean Solar Zenith Angle** for that specific image.
  2. It uses the Acquisition Date to calculate the precise **Earth-Sun Distance** ($d$) in Astronomical Units for that day.
  3. For every single band, it mathematically divides the Reflectance by 10000, multiplies it by the Sentinel-2 **Mean Solar Exoatmospheric Irradiance ($E_{sun}$)** and the Cosine of the Solar Zenith, and divides by $\pi * d^2$.
  4. It converts the result into $\mu W/(cm^2*sr*nm)$, scales it by 10 to save disk space as `int16`, mathematically resamples B11 to 10m, stacks them, and exports a ready-to-use `.dat` file directly in ENVI's native BIL format.

- **Action (Automated ENVI Prep - FLAASH & QUAC):** 
  1. Run `01b_envi_flaash_prep.py` first. This script physically converts the image to Radiance, scales it by 10, resamples it, and formats it as BIL.
  2. Open ENVI and load the generated `flaash_radiance_stack...dat` file from `data/processed_envi/`. 
  3. Navigate to *Radiometric Correction* -> *Atmospheric Correction Module* -> *FLAASH Atmospheric Correction*. (Or select QUAC, which also accepts Radiance).
  4. **Provide Sensor Metadata:** Because our Python script injected the Wavelengths and FWHM directly into the ENVI `.hdr` file, FLAASH will *automatically* recognize the bands!
  5. When you select your input image, ENVI will ask for **Radiance Scale Factors**. Select **"Use single scale factor for all bands"** and enter **`10`** (NOT 10000. Our Python script scaled the absolute Radiance by 10 to save disk space).
  6. Configure the FLAASH parameters exactly as follows:
     *   **Flight Date & Time:** Extract this from your filename!
     *   **Atmospheric Model:** Sub-Arctic Summer.
     *   **Aerosol Retrieval:** 2-Band (K-T).
  7. **Assign K-T Channels:** Click **Multispectral Settings...**, go to **K-T Aerosol** tab:
      *   **K-T Upper Channel:** `5` (SWIR Band 11)
      *   **K-T Lower Channel:** `3` (Red Band 04)
  8. Run FLAASH. Modtran will not crash because the input is now pure Radiance!

- **Action (Manual ENVI Prep Alternative - Educative):** 
  1. Open ENVI and load the raw optical imagery (B02, B03, B04, B08, and B11).
  2. **Radiometric Calibration:** Because FLAASH requires Radiance, you must manually use ENVI's Radiometric Calibration tool. Without the `.xml` file, you must manually calculate and enter the Solar Zenith, Earth-Sun distance, and ESUN variables to convert the Reflectance to Radiance. 
  3. **Spatial Resampling:** B11 is 20m resolution, while the others are 10m. Navigate to *Toolbox* -> *Raster Management* -> *Resize Data*. Select the calibrated B11 and resize it by a factor of 2 (or explicitly set pixel size to 10m) using Nearest Neighbor or Bilinear interpolation.
  4. Create a Layer Stack of your 10m bands (B02, B03, B04, B08, and the newly resampled B11). By default, ENVI creates this in BSQ (Band Sequential) format.
  5. **Crucial Step:** FLAASH requires BIL (Band Interleaved by Line) or BIP (Band Interleaved by Pixel) format. Navigate to *Toolbox* -> *Raster Management* -> *Convert Interleave* to convert your BSQ stack to BIL.
  6. **Set Data Ignore Value:** Edit the output `.hdr` file in Notepad and add `data ignore value = 0` so ENVI ignores the black borders.
  7. **Provide Sensor Metadata:** Because our Python script downloads bare `.tif` files, ENVI doesn't automatically know this is Sentinel-2 data. FLAASH will ask for an ASCII file containing the Center Wavelength and FWHM (Full Width at Half Maximum) for each band to calculate atmospheric scattering. 
  8. *Solution:* Use the `sentinel2_fwhm.txt` file located in `data/raw/` when prompted. When the "Input ASCII File" dialog appears, configure it exactly like this:
     *   **Wavelength Column:** `1`
     *   **Wavelength Units:** Change dropdown from `micron` to `Nanometers`
     *   **FWHM Column:** `2`
  9. Navigate to *Radiometric Correction* -> *Atmospheric Correction Module* -> *FLAASH Atmospheric Correction*.
  10. When you select your input image, ENVI will ask for **Radiance Scale Factors**. Do *not* use an ASCII file. Select **"Use single scale factor for all bands"** and enter **`10`**. (Because our manual calibration or script scales the radiance by 10).
  11. Configure the FLAASH parameters exactly as follows:
      *   **Flight Date & Time:** Extract this from your filename!
      *   **Pixel Size:** Ensure this says **10.000** (Sentinel-2).
      *   **Atmospheric Model:** Change from Tropical to **Sub-Arctic Summer** (Patagonia is far south, and January is summer).
      *   **Aerosol Retrieval:** Because we now downloaded and stacked the SWIR band (B11), you can leave this on **2-Band (K-T)** for dynamic aerosol calculation!
      *   **Initial Visibility:** 40 km is perfect for pristine Patagonia.
  12. **Assign K-T Channels:** Click the **Multispectral Settings...** button at the bottom, go to the **K-T Aerosol** tab, and set:
      *   **K-T Upper Channel:** `5` (This is our SWIR Band 11)
      *   **K-T Lower Channel:** `3` (This is our Red Band 04)
  13. Run FLAASH to generate the BOA surface reflectance raster.
- **Action (ArcGIS Pro Alternative):** 
  1. Load the raw raster into ArcGIS Pro.
  2. If using Landsat, go to *Imagery* tab -> *Raster Functions* -> *Apparent Reflectance* (this does basic TOA).
  3. For DOS1 (Haze Removal), use the *Raster Calculator*: find the minimum pixel value in the dark areas, and subtract it: `"Band_1" - Min_Value`. Ensure you set negative values to 0 using the `Con()` statement.

### `03_station_ml_interpolation.py`
- **Concept:** In-situ weather stations provide accurate but sparse point data. However, remote stations often fail, producing data anomalies (e.g., -9999 or impossible spikes).
- **Action (Anomaly Detection):** Uses `scikit-learn`'s `IsolationForest` to automatically detect and flag anomalous temperature/precipitation readings. **Manual Validation:** The script outputs plots so the user can visually confirm the dropped anomalies.
- **Action (Interpolation):** Uses the surrounding topography (elevation, latitude) as covariates to train a Random Forest regressor, predicting a continuous climate raster surface using only the cleaned, validated data.

### `04_precipitation_anomaly.py`
- **Concept:** Comparing historical climate trends to identify extreme drought/flood cycles.
- **Action (Python ML):** Connects to the Open-Meteo Historical API to dynamically fetch 30 years (1993-2023) of real daily precipitation data (ERA5-Land) for the region. It resamples this massive dataset into annual totals.
- **Action (Unsupervised Learning):** Uses `scikit-learn`'s K-Means clustering algorithm to mathematically divide the 30-year history into 3 distinct climate regimes: *Extreme Drought*, *Normal*, and *Extreme Flood*.
- **Action (R Alternative):** An R script (`04b_climate_anomaly.R`) using the `anomalize` package to perform rigorous time-series decomposition (STL) to detect extreme climate anomalies over a 30-year period.
- **Action (ArcGIS Pro Alternative):** Documenting the use of the *Space-Time Pattern Mining* toolbox to run Emerging Hot Spot Analysis and Local Outlier Analysis on the precipitation cubes.
- **Output:** Generates a visually classified bar chart showing the 30-year trend and highlighting the exact years that suffered extreme anomalies.

### `05_uhi_modis_mapping.py`
- **Concept:** Deriving Land Surface Temperature (LST) from thermal radiance.
- **Action:** Maps the Urban Heat Island effect over Punta Arenas using thermal satellite data.

---

## 🚀 How to Run

### 1. Install Chapter Dependencies
Activate your environment and install the required packages:
```bash
mamba activate geocascade_env
mamba install -c conda-forge pystac-client planetary-computer requests rasterio scikit-learn pandas geopandas matplotlib -y
```

### 2. Standard Automated Execution (L2A Data)
For most workflows, you will simply download the pre-corrected L2A surface reflectance data and skip the atmospheric correction scripts entirely:
```bash
# Downloads Sentinel-2 L2A and Copernicus DEM to data/raw/
python 01_stac_multisensor_download.py

# Optional: Download Landsat 8/9 L2 data
python 01c_landsat_download.py
```

### 3. Optional: Manual Atmospheric Correction Practice (L1 Data)
If you want to practice physics-based atmospheric correction:
1. Follow the **Manual Download Instructions** (in the Tip box above) to download L1C/L1 data into your `data/raw/` folder.
2. Run the prep script to automatically calibrate the metadata and stack the bands for ENVI FLAASH:
   ```bash
   python 01b_envi_flaash_prep.py
   ```
3. Run the pure-Python COST model correction:
   ```bash
   python 02_atmospheric_correction.py
   ```

### 4. Run ML Interpolation (Script 3)
In this phase, you will fetch real ERA5 climate data to act as "weather stations" and then run an ML pipeline to clean and interpolate them.

1. **Fetch Real Climate Data:**
   ```bash
   python 03a_fetch_real_weather_data.py
   ```
2. **Run Anomaly Detection and Interpolation:**
   ```bash
   python 03_station_ml_interpolation.py
   ```

### 5. Run Long-Term Climate Analysis (Script 4)
Run the 30-year precipitation analysis to classify droughts and floods:
```bash
python 04_precipitation_anomaly.py
```

### 6. Map Urban Heat Islands (Script 5)
Download thermal MODIS data and generate an Urban Heat Island map of Punta Arenas:
```bash
python 05_uhi_modis_mapping.py
```
