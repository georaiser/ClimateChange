# Chapter 9: Multi-Task Deep Learning (PyTorch)

> [!IMPORTANT]
> **Status: 🚧 Planned / In Development**
> This chapter is part of Phase 2 and will be implemented in the next iteration. This README documents the architectural plan and learning objectives.

## 🎯 Academic Objective

Standard convolutional neural networks (CNNs) are trained to solve **one task at a time** — classify a pixel OR detect change, never both simultaneously. But in real Earth Observation workflows, both tasks require the same feature representations (edges, texture patterns, spectral gradients). **Multi-Task Learning (MTL)** shares a common backbone between multiple task heads, reducing training data requirements and improving generalization by leveraging cross-task regularization.

By the end of this chapter you will be able to:
- Understand the trade-offs between single-task and multi-task CNN architectures
- Build a shared ResNet-50 encoder with two independent task decoder heads
- Train the model using a combined weighted loss function
- Evaluate each task independently using IoU and F1 metrics

---

## 🏗️ Planned Architecture

```
Input: Sentinel-2 Patch (256×256×5 bands)
            │
    ┌───────┴───────┐
    │ Shared Encoder │  ← ResNet-50 backbone (pre-trained on ImageNet)
    │ (Feature Maps) │
    └───────┬───────┘
            │
    ┌───────┴──────────────┐
    │                      │
┌───┴────┐           ┌─────┴──────┐
│  Head 1 │           │   Head 2   │
│  Land   │           │  Change    │
│  Cover  │           │ Detection  │
│ (Segm.) │           │ (Binary)   │
└─────────┘           └────────────┘
```

**Task 1 — Land Cover Segmentation:**
- Predicts a multi-class label for every pixel
- Classes: Water, Ice/Glacier, Dense Vegetation, Sparse Vegetation, Bare Rock
- Loss: Cross-Entropy with class weights (inversely proportional to class frequency)
- Metric: **IoU (Intersection over Union)** per class

**Task 2 — Change Detection:**
- Binary prediction: has this pixel changed significantly between two dates?
- Input: two co-registered image patches (2023 vs 2003)
- Loss: Binary Cross-Entropy with Focal weighting (upweights rare change pixels)
- Metric: **F1-Score** (precision-recall balance for imbalanced change/no-change)

**Combined Loss:**
$$\mathcal{L}_{total} = \alpha \cdot \mathcal{L}_{segmentation} + (1 - \alpha) \cdot \mathcal{L}_{change}, \quad \alpha = 0.6$$

---

## 📂 Planned Files

| File | Description |
|------|-------------|
| `23_dataset_preparation.py` | Tile Chapter 8 data cube into 256×256 patches + generate pseudo-labels from RF model |
| `24_multitask_model.py` | Define shared ResNet-50 encoder + two decoder heads in PyTorch |
| `25_training_loop.py` | PyTorch Lightning training + validation loop + TensorBoard logging |
| `26_inference_mosaic.py` | Run trained model over full BBOX, stitch patches into full-resolution prediction |

---

## 🚀 Planned Installation

```bash
# GPU-accelerated PyTorch (CUDA 12.4)
mamba install -n geocascade_env pytorch torchvision pytorch-cuda=12.4 \
    -c pytorch -c nvidia -c conda-forge --channel-priority flexible -y

# Training utilities
pip install pytorch-lightning torchmetrics tensorboard onnx
```

---

## 🗺️ Expected Outputs

| Output | Format | Description |
|--------|--------|-------------|
| `land_cover_prediction.tif` | GeoTIFF | 5-class segmentation map |
| `change_detection_map.tif` | GeoTIFF | Binary change map (2003 → 2023) |
| `model_geocascade.onnx` | ONNX | Exportable model for production inference |
| `training_curves.png` | PNG | Loss and IoU curves per epoch |

---

## 📐 Key Metrics

| Metric | Formula | Good Value |
|--------|---------|-----------|
| IoU | $|A \cap B| / |A \cup B|$ | > 0.70 per class |
| F1 | $2PR / (P + R)$ | > 0.75 for change |
| Overall Accuracy | Correct pixels / Total | > 0.85 |

---

## 📚 Academic References

- He, K. et al. (2016). Deep Residual Learning for Image Recognition. *CVPR*.
- Ruder, S. (2017). An Overview of Multi-Task Learning in Deep Neural Networks. *arXiv:1706.05098*.
- Chen, J. et al. (2021). A Spatial-Temporal Attention-Based Method for Remote Sensing Image Change Detection. *IEEE TGRS*.
