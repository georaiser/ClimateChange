# 🔬 ENVI 5.6 Track — Atmospheric Correction & Spectral Analysis

> **GeoCascade Chapter 1 — ENVI Workflow Guide**
> *ENVI 5.6 + IDL | Torres del Paine, Patagonia*

---

## When Do You Need Atmospheric Correction?

> [!IMPORTANT]
> **Never apply FLAASH to data that is already atmospherically corrected.**
> Applying correction twice will corrupt your reflectance values.

| Dataset | Level | Status | Action |
|---|---|---|---|
| **Sentinel-2 L2A** | BOA Surface Reflectance | ✅ Corrected by ESA Sen2Cor | **Use directly** |
| **Landsat 9 L2SP** | Surface Reflectance + ST | ✅ Corrected by USGS LaSRC | **Use directly** |
| **Sentinel-2 L1C** | TOA Radiance | ❌ Raw, uncorrected | Apply FLAASH or Sen2Cor |
| **Landsat 9 L1TP** | TOA Radiance | ❌ Raw, uncorrected | Apply FLAASH or DOS1 |
| **Copernicus DEM** | Elevation model | N/A | Use directly |
| **ERA5 / CHIRPS** | Climate reanalysis | N/A | Use directly |

The Chapter 1 Python pipeline downloads **L2A and L2SP** — already corrected.
The ENVI FLAASH workflow teaches the process for when you acquire raw L1C/L1TP data.

---

## FLAASH Parameters for Torres del Paine

FLAASH uses the **MODTRAN5** radiative transfer model. The key parameters are:

| Parameter | Value | Reason |
|---|---|---|
| **Atmospheric model** | Sub-Arctic Summer | Best match for 51°S summer conditions |
| **Aerosol model** | Maritime | Pacific onshore winds dominate Patagonian aerosol |
| **Initial visibility** | 40 km | Typical clear-sky visibility in Torres del Paine |
| **Scene centre lat** | −51.0° | Southern Hemisphere (negative sign required) |
| **Scene centre lon** | −73.0° | West (negative sign required) |
| **Scene elevation** | 0.15 km | Mean valley floor elevation |
| **Water retrieval** | Enabled | 1135 nm absorption feature used |
| **Aerosol retrieval** | Enabled | 2-band Kaufman-Tanre method |

> [!NOTE]
> **Why Sub-Arctic Summer?** Torres del Paine is at 51°S — equivalent in character
> to sub-Arctic latitudes in the Northern Hemisphere. The Sub-Arctic Summer profile
> in MODTRAN5 has the best-fit temperature lapse rate, humidity profile, and ozone
> column for mid-latitude Southern Hemisphere summer conditions.

> [!NOTE]
> **Why Maritime aerosol?** Persistent westerly winds off the Pacific bring clean
> marine aerosol (sea salt, low loading) to western Patagonia. This differs from
> inland areas where Rural or Tropospheric aerosol models are more appropriate.

**FLAASH radiative transfer equation:**

$$L_{at-sensor} = \frac{\rho_{surface} \cdot E_{down} \cdot \cos\theta}{\pi} \cdot T(\lambda) + L_{path}(\lambda)$$

Where:
- $\rho_{surface}$ = surface reflectance (what we solve for)
- $E_{down}$ = downwelling irradiance (modeled by MODTRAN5)
- $T(\lambda)$ = atmospheric transmittance
- $L_{path}$ = path radiance (atmospheric scattering)

---

## Running the IDL Script

**`01_flaash_correction.pro`** — recommended for most users.

### Step-by-step

1. Open **ENVI 5.6**
2. Prepare your input data (if bands are separate):
   - **Raster Management > Layer Stacking**
   - Add all Landsat L1TP bands (B1–B7)
   - Output format: **ENVI** (BSQ interleave)
   - Save as `landsat9_l1tp_stack.dat`

3. Edit the script — open `01_flaash_correction.pro` in a text editor and update:
   ```idl
   input_file = 'D:\path\to\landsat9_l1tp_stack.dat'
   output_dir = 'D:\path\to\output'
   scene_date = '2023-01-15'
   ```

4. Run the script:
   - **Option A**: ENVI Toolbox → ENVI Modeler → **Run IDL Script** → browse to `.pro`
   - **Option B**: IDL Console (at bottom of ENVI): `.compile 'path\to\01_flaash_correction.pro'` then `flaash_patagonia`

5. Monitor progress in the ENVI status bar (5–20 minutes depending on scene size)

### Expected outputs

```
data/processed/envi_flaash/
├── flaash_corrected.dat        ENVI format reflectance
├── flaash_corrected.hdr        ENVI header
├── flaash_corrected_rfl.tif    GeoTIFF (ArcGIS Pro / QGIS ready)
├── flaash_cloud_mask.dat       Cloud mask
├── flaash_water_vapor.dat      Water vapour column map
└── flaash_report.txt           FLAASH processing report
```

> [!TIP]
> The `.tif` file is directly loadable in **ArcGIS Pro**, **QGIS**, and Python (`rasterio.open()`).
> The `.dat` file is ENVI's native format — open it with **File > Open**.

---

## Running the Python API Script

**`01_flaash_correction.py`** — for automation / batch processing.

ENVI 5.6 includes a built-in Python interpreter with the `envi` module pre-installed.

### From ENVI Python Console

1. ENVI → **Tools > Python Console**
2. In the console:
   ```python
   exec(open(r'D:\00_AI_Aplications\ClimateChange\Chapter_01\envi\01_flaash_correction.py').read())
   ```

### As a Script

1. Edit `INPUT_FILE`, `OUTPUT_DIR`, and `SCENE_DATE` in the script
2. ENVI → **Tools > Python Script** (if available in your ENVI installation)
3. Select and run the script

> [!IMPORTANT]
> The script gracefully handles missing `envi` module: if you run it outside ENVI,
> it prints clear instructions and exits without crashing.

---

## Spectral Analysis After FLAASH

After running FLAASH, perform spectral profiling in ENVI:

### Manual ROI Method (ENVI GUI)

1. **File > Open** → `flaash_corrected_rfl.tif`
2. In the Layer Manager, right-click the raster → **Open in Display**
3. Draw ROIs over representative areas:
   - Grey Glacier (bright white/cyan in NIR)
   - Lenga beech forest (strong NIR reflectance plateau)
   - Grey Lake open water (low reflectance all bands)
   - Rock outcrop (flat, moderate reflectance)
4. **Tools > Region of Interest (ROI)** → **ROI Tool**
5. **Tools > Spectra** → **Spectral Profile** → select your ROIs
6. **File > Export > Spectral Profile to CSV** for use in Chapter 2

### Expected spectral signatures

| Feature | Blue | Green | Red | NIR | SWIR1 |
|---|---|---|---|---|---|
| Glacier/snow | High | High | High | High | **Low** |
| Forest | Low | Moderate | Low | **Very High** | Moderate |
| Open water | Moderate | Moderate | Low | **Very Low** | Very Low |
| Rock/bare | Moderate | Moderate | Moderate | Moderate | Moderate |

> [!TIP]
> The sharp drop in reflectance from NIR to SWIR1 for glaciers is the
> physical basis of NDSI: `(Green − SWIR1) / (Green + SWIR1) > 0.4` = ice/snow.

---

## Exporting for ArcGIS Pro

After FLAASH correction:

1. **File > Save As > Save As New File**
2. Choose format: **GeoTIFF**
3. Set output path: `data/processed/envi_flaash/flaash_corrected_rfl.tif`
4. Ensure **Output CRS** matches input (usually `WGS 84 / UTM Zone 19S`)

**Or use the IDL script** which exports automatically via `ExportRasterToFormat`.

ArcGIS Pro will open it directly:
- **Insert > Add Data > Raster Dataset** → select the `.tif`
- Apply **Stretch** symbology (Minimum-Maximum or 2% Clip)

---

## Method Comparison: FLAASH vs DOS1 vs Pre-corrected L2

| Method | Accuracy | Speed | Requires ENVI? | Best for |
|---|---|---|---|---|
| **Pre-corrected L2A/L2SP** | Highest (physics-based) | Instant (already done) | No | Production analysis |
| **FLAASH** | High (MODTRAN physics) | 5–20 min per scene | Yes (licensed) | L1C/L1TP inputs |
| **DOS1** (script 03) | Moderate (empirical) | Seconds | No | Quick comparison |
| **Sen2Cor** | High (physics-based) | 10–30 min | No (free) | Sentinel-2 L1C only |

> [!NOTE]
> The Python script `03_atmospheric_correction.py` demonstrates **DOS1** on the
> Landsat L2SP data (treating it as pseudo-L1 for teaching). The RMSE comparison
> it prints shows how close DOS1 gets to the official LaSRC product.

---

## Files in This Folder

| File | Format | Run from |
|---|---|---|
| `README_ENVI.md` | Markdown | Any viewer |
| `01_flaash_correction.pro` | IDL script | ENVI 5.6 IDL Console / Toolbox |
| `01_flaash_correction.py` | Python | ENVI Python Console |

---

*GeoCascade | Chapter 1 | ENVI 5.6 Track | Study area: Torres del Paine, 51°S*
