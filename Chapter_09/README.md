# 🧠 Chapter 9: Convolutional Neural Network Land Cover Classification

> **GeoCascade Pipeline — Stage 9**
> PyTorch CNN learns spatial texture patterns from the 4-sensor data cube
> to classify land cover with spatial context that Random Forest cannot capture.

---

## 📋 Overview

| Script | Topic | Key Outputs |
|--------|-------|-------------|
| `24_deep_learning_landcover.py` | 3-layer CNN trained on 32×32 multi-sensor patches | `cnn_landcover_prediction.tif`, `cnn_confidence_map.tif`, `cnn_training_history.csv`, `cnn_class_metrics.csv`, 6-panel figure |

---

## 🚀 Setup

```bash
conda activate geocascade_env

# CPU-only (recommended for first run)
mamba install -n geocascade_env -c conda-forge \
    pytorch torchvision cpuonly rasterio numpy \
    matplotlib scikit-learn pandas -y

# GPU (optional, for faster training)
mamba install -n geocascade_env -c pytorch -c nvidia \
    pytorch torchvision pytorch-cuda=12.4 -y
```

---

## ▶️ Run

```bash
# Requires Chapter_08/20_multisensor_data_fusion.py to have run first
python Chapter_09/24_deep_learning_landcover.py
```

**Expected output:**
```
========================================================
 GEOCASCADE -- CHAPTER 9: CNN LAND COVER CLASSIFICATION
 Patch=32px  Epochs=30  LR=0.001
========================================================
[INFO] PyTorch 2.x  device=cpu
[1/6] Loading multi-sensor data cube...
       Shape: 4 bands x 512 x 512
[2/6] Generating training labels...
       [1] Water          :  2,341 pixels
       [2] Glacier        :  8,912 pixels
       [3] Land/Veg       :  5,104 pixels
[3/6] Building patch dataset (max 8,000 patches)...
       Total labeled patches: 7,203
[4/6] Training CNN...
       Parameters: 538,051  Classes: 3  Device: cpu
       Ep  1/30  TL=1.0842  VL=0.9231  VA=0.612
       Ep  5/30  TL=0.6134  VL=0.5812  VA=0.788
       Ep 10/30  TL=0.4217  VL=0.4103  VA=0.851
       Best Val Accuracy: 0.891
[5/6] Full-image sliding-window inference...
[6/6] Evaluating and saving outputs...
```

---

## 🔬 CNN vs Random Forest (Chapter 8)

| Aspect | RF (Ch08) | CNN (Ch09) |
|--------|-----------|------------|
| Spatial context | ❌ Single pixel only | ✅ 32×32 neighbourhood |
| Learns textures | ❌ No | ✅ Yes (lake shorelines, glacier margins, forest edges) |
| Training data | Percentile thresholds | Same physics-based labels + patch context |
| Inference speed | Fast (no GPU needed) | Slower (sliding window, GPU recommended) |
| Interpretability | High (feature importances) | Lower (black-box CNN features) |
| Suitable for | Quick baseline | High-accuracy production |

> [!TIP]
> Run RF (Chapter 8) first to get a baseline accuracy. CNN should outperform RF by 5–15% OA
> due to spatial context, especially at class boundaries (glacier margins, lake shorelines).

---

## 🏗️ Architecture

```
Input: (B, 4, 32, 32)   ← 4 sensor bands, 32×32 pixel patch
         │
         ▼
┌─────────────────────────────────────┐
│  Block 1: Conv2d(4→32, 3×3)        │
│           BatchNorm2d(32)           │  → (B, 32, 32, 32)
│           ReLU                      │
│           MaxPool2d(2×2)            │  → (B, 32, 16, 16)
├─────────────────────────────────────┤
│  Block 2: Conv2d(32→64, 3×3)       │
│           BatchNorm2d(64)           │  → (B, 64, 16, 16)
│           ReLU                      │
│           MaxPool2d(2×2)            │  → (B, 64, 8, 8)
├─────────────────────────────────────┤
│  Block 3: Conv2d(64→128, 3×3)      │
│           BatchNorm2d(128)          │  → (B, 128, 8, 8)
│           ReLU                      │
│           MaxPool2d(2×2)            │  → (B, 128, 4, 4)
├─────────────────────────────────────┤
│  Flatten: 128 × 4 × 4 = 2,048      │
│  Linear(2048 → 256) + ReLU         │
│  Dropout(p=0.3)                     │
│  Linear(256 → 3)                    │
└─────────────────────────────────────┘
         │
         ▼
Output: (B, 3)   ← logits for [Water, Glacier, Land/Veg]
```

**Training hyperparameters:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Patch size | 32×32 px | Captures shoreline/edge context at 10m resolution |
| Max patches | 8,000 | Fits in CPU RAM; GPU can use 50,000+ |
| Epochs | 30 | Convergence observed by epoch 20–25 |
| Optimizer | Adam (lr=1e-3) | Adaptive learning, fast convergence |
| LR schedule | CosineAnnealing | Smooth decay, avoids sharp oscillations |
| Loss | CrossEntropyLoss | Multi-class classification |
| Train/Val | 80/20 split | Standard evaluation |

---

## 📐 Physics-Based Label Generation

Labels are generated dynamically from geophysical percentiles (same approach as Ch08 RF):

```python
# Water:   Low SAR (p10) AND Low NIR (p10)   → specular + barren
# Glacier: High DEM (p90) AND Cold LST (p10) → high elevation + cold
# Land:    High NIR (p90) AND Warm LST (p90) → dense vegetation + warm

# These rules use scene percentiles, NOT hardcoded reflectance values.
# This ensures labels exist even when absolute values shift between dates.
```

> [!NOTE]
> Only labeled pixels are used for training patches. ~60–80% of pixels remain
> unlabeled (ambiguous zones) — the CNN predicts them during full-image inference.

---

## 📂 Output Structure

```
Chapter_09/
└── data/processed/dl/
    ├── cnn_landcover_prediction.tif    ← uint8 class map (1=Water, 2=Glacier, 3=Land/Veg)
    ├── cnn_confidence_map.tif          ← float32 softmax confidence [0-1] per pixel
    ├── cnn_landcover_model.pt          ← PyTorch model weights (reusable)
    ├── cnn_training_history.csv        ← epoch, train_loss, val_loss, val_acc
    ├── cnn_class_metrics.csv           ← precision, recall, f1, support per class
    └── cnn_landcover_results.png       ← 6-panel: prediction, confidence, labels,
                                           training loss, val accuracy, confusion matrix
```

---

## 🖥️ ArcGIS Pro Integration

```
Add cnn_landcover_prediction.tif
  Symbology > Unique Values
    1 = #1f77b4  Blue      (Water / Lake)
    2 = #00d4ff  Cyan      (Glacier / Ice)
    3 = #2ca02c  Green     (Land / Vegetation)

Add cnn_confidence_map.tif
  Symbology > Stretched > Plasma color ramp (low=dark, high=bright)
  → Pixels where CNN is most confident are brightest
  Raster Calculator: Con("cnn_confidence_map.tif" < 0.6, 1, 0)
  → Binary mask of LOW-CONFIDENCE pixels → areas to review manually

Compare with Chapter 8 RF prediction:
  Raster Calculator: Con("cnn_prediction.tif" != "rf_prediction.tif", 1, 0)
  → Disagreement map between CNN and RF → ecotone / mixed-cover zones
```

---

## 🔵 ENVI 5.6 Integration

```
; Open CNN classification result
File > Open > cnn_landcover_prediction.tif
Classification > Post Classification > Class Statistics
  → Pixel count + km² area per class (validate with Python class report)

; Compare with ENVI's own classifier
Toolbox > Classification > Supervised > Support Vector Machine
  Use same training AOI
  → Compare SVM vs CNN classification maps to assess method agreement

; Spatial accuracy
Toolbox > Classification > Post Classification > Confusion Matrix
  Using Existing Statistics from cnn_class_metrics.csv
```

---

## ⚠️ Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| `FileNotFoundError: cascade_master_stack.tif` | Ch08 script 20 not run | `python Chapter_08/20_multisensor_data_fusion.py` |
| `Too few labeled patches (<50)` | Cube mostly NoData | Check Script 20 output; MODIS may be all fill |
| OOM (out of memory) on CPU | Scene too large | Reduce `MAX_PATCHES = 4000` in config |
| Training stuck at 33% accuracy | Class imbalance | Already handled by `class_weight="balanced"` in labels |
| `torch` ImportError | PyTorch not installed | `mamba install -n geocascade_env -c conda-forge pytorch cpuonly -y` |

---

## 📖 Key References

- LeCun, Y. et al. (1998). *Gradient-based learning applied to document recognition.* Proc. IEEE.
- Ioffe, S., Szegedy, C. (2015). *Batch Normalization: Accelerating Deep Network Training.* ICML.
- He, K. et al. (2016). *Deep Residual Learning for Image Recognition.* CVPR.
- Maggiori, E. et al. (2017). *Convolutional Neural Networks for Large-Scale Remote-Sensing Image Classification.* IEEE TGRS.
