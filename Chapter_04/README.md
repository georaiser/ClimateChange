# Chapter 4: Ecological Vulnerability & Prediction

## 🎯 Academic Objective
This is the capstone chapter of our curriculum. Here, we transition from analyzing single variables to performing **Multi-Criteria Decision Analysis (MCDA)** and **Machine Learning Predictive Modeling**.

We will combine all the environmental layers we explored in Chapters 1-3 (Elevation, Slope, Vegetation) to predict the ecological niche of endangered species and map the overall climate vulnerability of Torres del Paine.

---

## 🛠️ Scripts & Modules

### `12_ecological_niche_modeling.py`
- **Concept:** Species Distribution Modeling (SDM) uses environmental data to predict where a species can survive. By training a model on known occurrence locations, it can find similar habitats across the entire landscape.
- **Action (Python Automation):** Uses `scikit-learn` to train a **Random Forest Classifier**. We simulate the habitat preferences of the endangered Patagonian Huemul Deer (low elevation, gentle slopes, high NDVI). The script generates synthetic presence/absence points, trains the Machine Learning model, and predicts the "Probability of Occurrence" across the entire park. The final SDM is exported as `ecological_niche_model.tif`.
- **Action (ArcGIS Pro Alternative):** 
  1. This is equivalent to the **Spatial Analyst -> Suitability Modeler** or the **MaxEnt (Maximum Entropy)** tools in ArcGIS Pro.
  2. You provide your DEM, Slope, and NDVI rasters.
  3. You input a point shapefile of species sightings.
  4. The tool generates a suitability map based on the environmental conditions at those points.

### `13_climate_vulnerability_index.py`
- **Concept:** Multi-Criteria Decision Analysis (MCDA) combines disparate datasets into a single score. You standardize all variables to a common scale (e.g., 0 to 1) and sum them based on their relative importance (weights).
- **Action (Python Automation):** Fetches the DEM, calculates Slope, and fetches NDVI. It standardizes them using Min-Max Scaling (0.0 to 1.0). It then applies a mathematical weighting algorithm: `Vulnerability = (0.5 * Inverted_NDVI) + (0.3 * Slope) + (0.2 * Elevation)`. Areas with low vegetation, steep slopes, and high exposure are flagged as the most vulnerable to climate shocks. The resulting raster is exported as `climate_vulnerability_index.tif`.
- **Action (ArcGIS Pro Alternative):** 
  1. In ArcGIS Pro, this is known as the **Weighted Overlay** tool (under *Spatial Analyst -> Overlay*).
  2. First, use the **Rescale by Function** tool to standardize your DEM, Slope, and NDVI layers to a 0-1 scale.
  3. Input them into the Weighted Overlay tool, assigning 50% influence to NDVI, 30% to Slope, and 20% to Elevation.
  4. The output is your Vulnerability Map.

---

## 🚀 How to Run

### 1. Install Chapter Dependencies
Activate your environment and ensure the required Machine Learning packages are installed:
```bash
mamba activate geocascade_env
mamba install -c conda-forge scikit-learn rasterio numpy matplotlib pystac-client planetary-computer pyproj -y
```

### 2. Run Ecological Niche Modeling (Script 12)
Train a Random Forest to predict species habitat suitability:
```bash
python 12_ecological_niche_modeling.py
```
*(Check `data/processed/ecological_niche_model.png` for the ML probability map).*

### 3. Run Climate Vulnerability MCDA (Script 13)
Perform a Weighted Overlay analysis to map climate risk:
```bash
python 13_climate_vulnerability_index.py
```
*(Check `data/processed/climate_vulnerability_index.png` for the final index).*

---

## 🗺️ GIS Interoperability (ArcGIS Pro & ENVI)
A core academic requirement of this curriculum is **Hybrid Dual-Track Interoperability**. Every Python script in this chapter has been engineered to automatically export **Geocoded TIFF (.tif)** and **Shapefile (.shp)** outputs into the \data/processed/\ directory.

Instead of just looking at matplotlib PNG graphs, students are encouraged to:
1. Run the automated Python pipeline.
2. Open **ArcGIS Pro** or **ENVI**.
3. Drag-and-drop the generated \.tif\ and \.shp\ files directly into the GUI.
4. Verify that the Python outputs align perfectly with the GUI software basemaps.
