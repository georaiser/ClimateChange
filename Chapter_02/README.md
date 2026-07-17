# 📡 Chapter 2: Spectral Analysis & Index Suite

> **GeoCascade Pipeline — Stage 2**
> *Sentinel-2 L2A | 9 Spectral Indices | Temporal Batch Processing*
> *Torres del Paine, Patagonia, Chile | WGS84 / UTM Zone 19S*

---

## Learning Objectives

| Concept | What you learn |
|---|---|
| **Spectral Signatures** | Every land cover reflects light differently; sampling 7 wavelengths identifies glacier, forest, water, and rock |
| **Red Edge (B05, 705nm)** | Sentinel-2 unique band; detects vegetation stress BEFORE NDVI changes |
| **Cloud-Native COG Reading** | Stream only the BBOX pixels from Microsoft Planetary Computer — no 600MB download |
| **Local Data Priority** | Chapter 1 already downloaded S2 data; use it first, fall back to STAC |
| **9 Spectral Indices** | NDVI, EVI, SAVI, BSI, NDWI, NDSI, NDGI, NDMI, NBR — each isolates a physical signal |
| **Temporal Stack** | Process a full year of scenes to detect seasonal cycles and anomalies |
| **ArcGIS/ENVI Integration** | All GeoTIFFs (nodata=-9999, LZW compressed) open directly in ArcGIS Pro and ENVI 5.6 |

---

## Data Connection: Chapter 1 → Chapter 2

Chapter 2 scripts resolve satellite data using this priority chain:

```
Priority 1: Chapter_01/data/raw/sentinel2_l2a_{date}/
            └─ Standard download from 02_satellite_acquisition.py (L2A, pre-corrected)

Priority 2: Chapter_01/data/raw/sentinel2_l2a_from_l1c_cost/
            └─ Auto-created by 03_atmospheric_correction.py when L1C data present
              (COST-corrected BOA bands, float32 [0-1], nodata=-9999)

Priority 3: Chapter_01/data/raw/sentinel2_l2a_from_l1c_flaash/
            └─ Manually copy FLAASH output here for publication-grade results
              (rename bands to B02.tif, B03.tif ... to match expected format)

Fallback:   Planetary Computer STAC (live stream, no download needed)
            └─ Used when none of the above folders exist
```

All sources produce **float32 reflectance [0–1]** GeoTIFFs with `nodata=-9999` and
`compress=lzw`, so all downstream index calculations work identically regardless
of which source was used.

> [!IMPORTANT]
> Run **Chapter_01/02_satellite_acquisition.py** before Chapter 2 to cache data
> locally. If you have L1C images, run **Chapter_01/03_atmospheric_correction.py**
> first — it will auto-bridge the corrected bands so Chapter 2 finds them via
> the `sentinel2_l2a_*` glob with no manual steps needed.

---

## Run Order

```bash
conda activate geocascade_env

# Step 1: Extract spectral signatures for 4 land cover types
python Chapter_02/06_spectral_signature_analysis.py

# Step 2: Compute 9 spectral indices for the study BBOX
python Chapter_02/07_vegetation_soil_indices.py

# Step 3: Batch process a full year of cloud-free scenes
python Chapter_02/08_automated_index_batcher.py
```

---

## Script Details

### `06_spectral_signature_analysis.py`

Extracts per-pixel reflectance at 7 wavelengths for 4 target materials:

| Material | Coords | Key signature |
|---|---|---|
| Grey Glacier ice | -51.010°, -73.230° | High all bands, drops sharply at SWIR1 |
| Lenga Beech Forest | -51.150°, -72.950° | Low red, sharp rise at NIR (red edge) |
| Bare Rock / Scree | -50.900°, -72.900° | Flat, moderate across all bands |
| Grey Lake (water) | -51.050°, -73.180° | Very low NIR and SWIR |

**Outputs:**
```
data/processed/spectral_signatures.csv
data/processed/spectral_signatures.png
data/processed/spectral_red_edge_detail.png
```

**Red Edge academic note:** Sentinel-2 adds B05 (705nm) between the Red and NIR bands.
Under stress, chlorophyll degrades → the red edge shifts **blueward** and **narrows**
at 705nm. This is detectable before NDVI changes, giving ~2-4 week earlier warning.

---

### `07_vegetation_soil_indices.py`

Computes 9 indices in one pass over the BBOX:

| Index | Formula | Threshold | Use |
|---|---|---|---|
| **NDVI** | (NIR-Red)/(NIR+Red) | >0.5 = dense forest | Vegetation density |
| **EVI** | 2.5*(NIR-Red)/(NIR+6R-7.5B+1) | >0.4 = dense canopy | Forest (corrects saturation) |
| **SAVI** | (NIR-Red)(1+L)/(NIR+Red+L) | L=0.5 | Sparse steppe vegetation |
| **BSI** | (SWIR1+R-NIR-B)/(SWIR1+R+NIR+B) | >0 = bare | Erosion, overgrazing |
| **NDWI** | (Green-NIR)/(Green+NIR) | >0.3 = water | Lakes, rivers |
| **NDSI** | (Green-SWIR1)/(Green+SWIR1) | >0.4 = ice | Glacier extent mask |
| **NDGI** | (Green-Red)/(Green+Red) | >0.1 = turbid ice | Glacier meltwater |
| **NDMI** | (NIR-SWIR1)/(NIR+SWIR1) | <0 = drought stress | Leaf water content |
| **NBR** | (NIR-SWIR2)/(NIR+SWIR2) | dNBR>0.1 = burned | Fire scar mapping |

> [!NOTE]
> **B11 (SWIR1) is 20m native resolution** — all indices using B11 resample it
> to 10m using bilinear resampling. The script uses an **independent window**
> for B11 computed from the 20m grid. Reusing the 10m window directly would
> silently read the wrong geographic area.

**Outputs:**
```
data/processed/indices/ndvi.tif
data/processed/indices/evi.tif
data/processed/indices/savi.tif
data/processed/indices/bsi.tif
data/processed/indices/ndwi.tif
data/processed/indices/ndsi.tif
data/processed/indices/ndgi.tif
data/processed/indices/ndmi.tif
data/processed/indices/nbr.tif
data/processed/indices/glacier_mask_ndsi.tif   (NDSI > 0.4)
data/processed/indices/water_mask_ndwi.tif     (NDWI > 0.3)
data/processed/spectral_indices_all.png        (3x3 dark panel)
data/processed/index_statistics.csv
```

---

### `08_automated_index_batcher.py`

Processes all cloud-free scenes in 2023 (< 10% cloud cover):

```
2023-01-15  NDVI=+0.421  NDSI=+0.512  (winter: glacier max, low vegetation)
2023-03-02  NDVI=+0.456  NDSI=+0.489
...
2023-12-20  NDVI=+0.398  NDSI=+0.531  (southern summer)
```

- Up to 24 scenes per year (configurable)
- **Incremental mode**: skips already-processed scenes on re-run
- Outputs one GeoTIFF per index per date: `ndvi_2023-01-15.tif`

**Outputs:**
```
data/processed/batch_indices/ndvi_{date}.tif   (one per scene)
data/processed/batch_indices/ndsi_{date}.tif
data/processed/batch_indices/ndwi_{date}.tif
data/processed/batch_indices/ndmi_{date}.tif
data/processed/batch_indices/batch_time_series.csv
data/processed/batch_temporal_analysis.png     (4-panel dark figure)
```

---

## ArcGIS Pro Integration

### Import all index TIFs

```python
# Run in ArcGIS Pro Notebook (Analysis > Python Notebook)
import arcpy, glob, os

index_dir = r"D:\00_AI_Aplications\ClimateChange\Chapter_02\data\processed\indices"
aprx = arcpy.mp.ArcGISProject("CURRENT")
m    = aprx.activeMap

for tif in glob.glob(os.path.join(index_dir, "*.tif")):
    m.addDataFromPath(tif)

aprx.save()
```

### Recommended symbology per index

| Layer | Symbology type | Color ramp |
|---|---|---|
| NDVI | Classified, 5 classes | Red-Yellow-Green |
| NDSI | Classified, 2 classes (0.4 threshold) | Blue-White |
| NDWI | Classified, 2 classes (0.3 threshold) | Blue |
| BSI  | Classified, 5 classes | Brown-White |
| NDMI | Diverging, -0.5 to 0.7 | Brown-Blue |
| NBR  | Diverging, -1 to 1 | Red-White-Green |

### Time animation (batch TIFs)

1. Load all `ndvi_*.tif` files
2. **Time** tab in layer properties → enable time
3. Set Time Field = embedded in filename (`ndvi_2023-01-15.tif`)
4. **Map > Time** slider → animate the seasonal NDVI cycle

---

## ENVI 5.6 Integration

### Open index TIFs

```
File > Open > spectral_indices_all.png  (preview)
File > Open > data/processed/indices/ndvi.tif
```

### Threshold glacier mask in ENVI Band Math

```
# Open NDSI TIF, then Tools > Band Math:
(b1 ge 0.4)
# Output: 1 = glacier, 0 = no glacier
```

### Scatter plot (NDVI vs NDMI)

1. Load `ndvi.tif` and `ndmi.tif`
2. **Tools > 2D Scatter Plot**
3. X axis: NDVI, Y axis: NDMI
4. Interpret:
   - High NDVI + High NDMI = healthy wet forest
   - High NDVI + Low NDMI = stressed vegetation (drought signal)
   - Low NDVI + Low NDMI = bare soil or rock

---

## Key Physical Relationships

```
       LOW NDVI                   HIGH NDVI
       Rock, glacier, water       Dense Lenga beech forest
           |                           |
           |    NDMI (moisture axis)   |
           |                           |
       LOW NDMI                   HIGH NDMI
       Drought stress             Saturated canopy

NDSI > 0.4 = glacier / snow
NDWI > 0.3 = open water body
NBR < -0.1 = fire-affected vegetation (post-fire)
BSI > 0.2  = bare soil or rock exposure
```

---

## Install Dependencies

```bash
mamba install -n geocascade_env -c conda-forge \
    pystac-client planetary-computer \
    rasterio pyproj numpy pandas matplotlib -y
```

---

## File Structure

```
Chapter_02/
├── README.md                             (this file)
├── 06_spectral_signature_analysis.py     (signatures, red edge, 4 materials)
├── 07_vegetation_soil_indices.py         (9 indices, masks, statistics)
├── 08_automated_index_batcher.py         (temporal stack, full year)
└── data/
    └── processed/
        ├── spectral_signatures.csv
        ├── spectral_signatures.png
        ├── spectral_red_edge_detail.png
        ├── spectral_indices_all.png
        ├── index_statistics.csv
        ├── batch_temporal_analysis.png
        └── indices/
            ├── ndvi.tif, evi.tif, savi.tif, bsi.tif
            ├── ndwi.tif, ndsi.tif, ndgi.tif, ndmi.tif, nbr.tif
            ├── glacier_mask_ndsi.tif
            ├── water_mask_ndwi.tif
            └── batch_indices/
                ├── ndvi_{date}.tif  ...
                └── batch_time_series.csv
```

---

*GeoCascade | Chapter 2 | Sentinel-2 L2A | Torres del Paine, 51°S*
