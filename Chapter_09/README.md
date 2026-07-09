# Chapter 9: Deep Learning Land Cover Classification

## Overview

Implements a 3-layer Convolutional Neural Network (CNN) for multi-class land cover
classification using the 4-band multi-sensor data cube built in Chapter 8.

**Why CNN vs Random Forest?**
Random Forest classifies each pixel **independently** with no spatial context.
A CNN sees a **32x32 pixel neighbourhood** and learns spatial patterns: edges,
texture gradients, and the surroundings of each pixel.

## Scripts

| Script | Purpose |
|---|---|
| `24_deep_learning_landcover.py` | CNN training + full-image prediction |

## CNN Architecture

```
Input : (B, 4, 32, 32)  -- 4 bands, 32x32 patch
Block1: Conv2d(4->32, 3x3) + BatchNorm + ReLU + MaxPool2x2  -> 16x16
Block2: Conv2d(32->64,3x3) + BatchNorm + ReLU + MaxPool2x2  -> 8x8
Block3: Conv2d(64->128,3x3)+ BatchNorm + ReLU + MaxPool2x2  -> 4x4
FC1   : 128*4*4 -> 256, ReLU, Dropout(0.3)
FC2   : 256 -> 3 classes  [Water | Glacier | Land/Veg]
```

## Classes (Physics-Based Labels)

| Class | Label | Thresholds |
|---|---|---|
| Water | 1 | SAR < 10th pct AND NIR < 10th pct |
| Glacier | 2 | DEM > 90th pct AND LST < 10th pct |
| Land/Veg | 3 | NIR > 90th pct AND LST > 90th pct |

> [!IMPORTANT]
> **Prerequisite:** Run `Chapter_08/20_multisensor_data_fusion.py` first to create
> `cascade_master_stack.tif` (4-band cube: S2 NIR, SAR VV dB, DEM m, MODIS LST C).

## Key Technical Notes

- **Patch normalization:** per-band z-score (mean=0, std=1) -- critical for CNN convergence
- **NoData handling:** `nan_to_num(0.0)` fills missing pixels in patches
- **Inference:** sliding-window batch of 1024 patches per forward pass
- **Outputs:** `cnn_landcover_prediction.tif` (uint8, nodata=0), `cnn_confidence_map.tif` (float32, nodata=-9999)

## Installation

```bash
mamba install -n geocascade_env -c conda-forge pytorch torchvision cpuonly rasterio numpy matplotlib scikit-learn -y
```

## Run

```bash
conda activate geocascade_env
python Chapter_09/24_deep_learning_landcover.py
```
