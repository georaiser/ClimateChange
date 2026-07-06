# Chapter 1: Climatic Variables & Image Processing

## 🎯 Academic Objective
The goal of this chapter is to establish a strong foundation in programmatic geospatial data acquisition, atmospheric correction, and climate modeling. 

Rather than manually clicking through web portals (like EarthExplorer or Copernicus Browser), we will build scripts that act as automated data pipelines. We will transform raw satellite imagery into **Analysis-Ready Data (ARD)** and use Machine Learning to interpolate local weather station records into continuous spatial maps.

---

## 🛠️ Scripts & Modules

### `01_stac_multisensor_download.py`
- **Concept:** Spatiotemporal Asset Catalogs (STAC) are the modern standard for querying Earth Observation data via APIs. This avoids downloading entire satellite tiles when you only need a specific bounding box (BBOX).
- **Academic Note on Band Selection:** Sentinel-2 captures 13 spectral bands, but this script intentionally downloads only 4: **B02 (Blue), B03 (Green), B04 (Red), and B08 (Near-Infrared)**. 
  *   *Spatial Resolution:* These four bands are the only ones captured at a native **10-meter** resolution, providing the highest precision for mapping glacial fronts.
  *   *Index Calculation:* Red and NIR are the only bands required to compute the core spectral indices for this project (NDVI for vegetation, NDWI for glacial lakes).
  *   *Efficiency:* Excluding the 20m and 60m bands (SWIR, Aerosol) drastically reduces disk storage and API download times without sacrificing the data needed for our specific hydrological pipeline.
- **Action:** A pure Python script utilizing `pystac-client` and `planetary-computer` (or AWS Earth Search) to search and download Sentinel-2, Landsat, and DEM data for the Torres del Paine Region of Interest (ROI). 
- **Dependencies:** `pystac-client`, `planetary-computer`, `rasterio`

### `02_atmospheric_correction.py`
- **Concept:** Sensors record Top of Atmosphere (TOA) reflectance, which includes atmospheric scattering (haze). True surface analysis requires Bottom of Atmosphere (BOA) reflectance.
- **Action (Python):** A Python implementation utilizing `rasterio` to mathematically correct TOA to BOA using Dark Object Subtraction (DOS1), replicating standard GIS tools.
  * **Physics Note on Band Exclusion (DOS1):**
    * **Applies DOS1 (Haze Removal):** B01 through B08A. (Visible and Near-Infrared light is heavily affected by Rayleigh/Mie scattering, so we subtract the dark object to remove the haze).
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
*   **Empirical / Image-Based (DOS1 and QUAC):** Uses statistics from the pixels inside the image itself to guess the atmospheric interference. *Works perfectly with Reflectance.*
    *   *Our Python script `02_atmospheric_correction.py` uses DOS1 (Dark Object Subtraction), which assumes the darkest pixel in the image should be zero.* 
    *   *QUAC is just a more advanced version of DOS1. Instead of looking for one dark pixel, QUAC looks at the histogram of the entire image and forces the average reflectance of the scene to match a known universal baseline of "average Earth materials."*

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

### `04_precipitation_dual_analysis`
- **Concept:** Comparing historical anomalies to identify extreme drought/flood cycles.
- **Action (Python):** Analyzes the CHIRPS precipitation dataset and clusters historical climates using unsupervised ML.
- **Action (R Alternative):** An R script (`04b_climate_anomaly.R`) using the `anomalize` package to perform rigorous time-series decomposition (STL) to detect extreme climate anomalies over a 30-year period.
- **Action (ArcGIS Pro Alternative):** Documenting the use of the *Space-Time Pattern Mining* toolbox to run Emerging Hot Spot Analysis and Local Outlier Analysis on the precipitation cubes.

### `05_uhi_modis_mapping.py`
- **Concept:** Deriving Land Surface Temperature (LST) from thermal radiance.
- **Action:** Maps the Urban Heat Island effect over Punta Arenas using thermal satellite data.

---

## 🚀 How to Run
### 1. Install Chapter Dependencies
Since we are using a modular approach, activate your base environment and install the required packages for Chapter 1:
```bash
mamba activate geocascade_env
mamba install -c conda-forge pystac-client planetary-computer requests rasterio -y
```

### 2. Execute Scripts
Execute the scripts sequentially:
```bash
python 01_stac_multisensor_download.py
python 01b_envi_flaash_prep.py  # Run this if you plan to use ENVI FLAASH
python 02_atmospheric_correction.py
```

### 3. Run ML Interpolation (Script 3)
Before running script 3, install the Machine Learning dependencies into the base environment:
```bash
mamba install -n geocascade_env -c conda-forge scikit-learn pandas geopandas matplotlib -y
```
Then, execute the anomaly detection and interpolation script:
```bash
python 03_station_ml_interpolation.py
```
