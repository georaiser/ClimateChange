"""
Chapter 9: 24_deep_learning_landcover.py

Academic Objective:
Implement a CNN for multi-class land cover classification using the 4-band
multi-sensor data cube built in Chapter 8 (cascade_master_stack.tif).

Why CNN vs Random Forest (Chapter 8)?
  RF classifies each pixel INDEPENDENTLY with no spatial context.
  CNN sees a 32x32 neighbourhood and learns spatial patterns (edges, textures).

Architecture - 3-layer ConvNet:
  Input  : (B, 4, 32, 32) -- 4 bands, 32x32 patch
  Block1 : Conv2d(4->32,3x3) + BatchNorm + ReLU + MaxPool2x2  -> 16x16
  Block2 : Conv2d(32->64,3x3) + BatchNorm + ReLU + MaxPool2x2 -> 8x8
  Block3 : Conv2d(64->128,3x3) + BatchNorm + ReLU + MaxPool2x2 -> 4x4
  FC1    : 128*4*4 -> 256, ReLU, Dropout(0.3)
  FC2    : 256 -> 3 classes  (Water / Glacier / Land-Vegetation)

Classes:
  1 = Water      -- low SAR backscatter + low NIR
  2 = Glacier    -- high elevation + low land surface temperature
  3 = Land/Veg   -- high NIR reflectance + high LST

Dependencies:
mamba install -n geocascade_env -c conda-forge pytorch torchvision cpuonly rasterio numpy matplotlib scikit-learn -y
\"\"\"

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import rasterio

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, random_split
    TORCH_OK = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] PyTorch {torch.__version__}  device={DEVICE}')
except ImportError:
    TORCH_OK = False
    DEVICE = None
    print('[WARNING] PyTorch not installed.')
    print('  mamba install -n geocascade_env -c conda-forge pytorch torchvision cpuonly -y')

try:
    from sklearn.metrics import (accuracy_score, f1_score,
                                 classification_report, confusion_matrix,
                                 ConfusionMatrixDisplay)
    SK_OK = True
except ImportError:
    SK_OK = False

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
# Check fusion/ subdirectory first (new path), then legacy path
_CH08 = os.path.join(BASE_DIR, '..', 'Chapter_08', 'data', 'processed')
IN_CUBE = os.path.join(_CH08, 'fusion', 'cascade_master_stack.tif')
if not os.path.exists(IN_CUBE):
    IN_CUBE = os.path.join(_CH08, 'cascade_master_stack.tif')
OUT_DIR   = os.path.join(BASE_DIR, 'data', 'processed', 'dl')
os.makedirs(OUT_DIR, exist_ok=True)

PATCH_SIZE  = 32
BATCH_SIZE  = 128
NUM_EPOCHS  = 30
LR          = 1e-3
SEED        = 42
MAX_PATCHES = 8000
np.random.seed(SEED)

# ==========================================================
# 1. Load cube
# ==========================================================
def load_cube():
    print(f'\n[1/6] Loading multi-sensor data cube...')
    if not os.path.exists(IN_CUBE):
        raise FileNotFoundError(
            f'cascade_master_stack.tif not found:\n  {IN_CUBE}\n'
            'Run Chapter_08/20_multisensor_data_fusion.py first.'
        )
    with rasterio.open(IN_CUBE) as src:
        cube    = src.read().astype('float32')
        profile = src.profile.copy()
        nd      = src.nodata if src.nodata is not None else -9999
    cube[cube == nd] = np.nan
    print(f'       Shape: {cube.shape[0]} bands x {cube.shape[1]} x {cube.shape[2]}')
    print(f'       Bands: [1] NIR  [2] SAR dB  [3] DEM m  [4] LST C')
    for b, name in enumerate(['NIR', 'SAR dB', 'DEM', 'LST']):
        v = cube[b][np.isfinite(cube[b])]
        if v.size:
            print(f'       Band {b+1} {name:6s}: mean={np.mean(v):+8.2f}  std={np.std(v):.2f}')
    return cube, profile

# ==========================================================
# 2. Physics-based labels
# ==========================================================
def make_labels(cube):
    print('\n[2/6] Generating training labels from physical thresholds...')
    NIR, SAR, DEM, LST = cube[0], cube[1], cube[2], cube[3]
    def pct(a, lo, hi):
        v = a[np.isfinite(a)]
        return np.nanpercentile(v, lo), np.nanpercentile(v, hi)
    lo_nir, hi_nir = pct(NIR, 10, 90)
    lo_sar, _      = pct(SAR, 10, 90)
    _,      hi_dem = pct(DEM, 10, 90)
    lo_lst, hi_lst = pct(LST, 10, 90)
    lbl = np.zeros(cube.shape[1:], dtype=np.uint8)
    lbl[(SAR <= lo_sar) & (NIR <= lo_nir)] = 1   # Water
    lbl[(DEM >= hi_dem) & (LST <= lo_lst)] = 2   # Glacier
    lbl[(NIR >= hi_nir) & (LST >= hi_lst)] = 3   # Land/Veg
    lbl[~np.isfinite(cube).all(axis=0)]    = 0   # NoData
    names = {0: 'NoData', 1: 'Water', 2: 'Glacier', 3: 'Land/Veg'}
    print('       Label distribution:')
    for cls, cnt in zip(*np.unique(lbl, return_counts=True)):
        print(f'         [{cls}] {names[cls]:12s}: {cnt:8,} pixels')
    return lbl

# ==========================================================
# 3. Patch Dataset
# ==========================================================
if TORCH_OK:
    class PatchDS(Dataset):
        def __init__(self, cube, labels):
            pad = PATCH_SIZE // 2
            H, W = labels.shape
            norm = cube.copy()
            for b in range(cube.shape[0]):
                v = cube[b][np.isfinite(cube[b])]
                if v.size:
                    norm[b] = (cube[b] - v.mean()) / (v.std() + 1e-8)
            norm = np.nan_to_num(norm, nan=0.0)
            rng = np.random.default_rng(SEED)
            rows, cols = np.where(labels > 0)
            if len(rows) > MAX_PATCHES:
                idx = rng.choice(len(rows), MAX_PATCHES, replace=False)
                rows, cols = rows[idx], cols[idx]
            self.X = []
            self.y = []
            for r, c in zip(rows, cols):
                r0, r1 = r - pad, r - pad + PATCH_SIZE
                c0, c1 = c - pad, c - pad + PATCH_SIZE
                if r0 < 0 or c0 < 0 or r1 > H or c1 > W:
                    continue
                self.X.append(norm[:, r0:r1, c0:c1].astype(np.float32))
                self.y.append(int(labels[r, c]) - 1)
        def __len__(self):
            return len(self.X)
        def __getitem__(self, i):
            return (torch.tensor(self.X[i]),
                    torch.tensor(self.y[i], dtype=torch.long))

# ==========================================================
# 4. CNN Architecture
# ==========================================================
if TORCH_OK:
    class LandCNN(nn.Module):
        """3-layer ConvNet for multi-spectral patch classification."""
        def __init__(self, n_bands=4, n_cls=3):
            super().__init__()
            def blk(ic, oc):
                return [nn.Conv2d(ic, oc, 3, padding=1),
                        nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
                        nn.MaxPool2d(2)]
            self.features = nn.Sequential(*blk(n_bands, 32), *blk(32, 64), *blk(64, 128))
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 4 * 4, 256), nn.ReLU(inplace=True), nn.Dropout(0.3),
                nn.Linear(256, n_cls))
        def forward(self, x):
            return self.head(self.features(x))

# ==========================================================
# 5. Training
# ==========================================================
def train_model(ds):
    print('\n[4/6] Training CNN...')
    n_val = max(1, int(0.2 * len(ds)))
    n_tr  = len(ds) - n_val
    tr_ds, va_ds = random_split(ds, [n_tr, n_val],
                                generator=torch.Generator().manual_seed(SEED))
    tr_dl = DataLoader(tr_ds, BATCH_SIZE, shuffle=True,  num_workers=0)
    va_dl = DataLoader(va_ds, BATCH_SIZE, shuffle=False, num_workers=0)
    n_cls = len(set(ds.y))
    model = LandCNN(4, n_cls).to(DEVICE)
    n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'       Parameters: {n_p:,}  Classes: {n_cls}  Device: {DEVICE}')
    opt   = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    sch   = optim.lr_scheduler.CosineAnnealingLR(opt, NUM_EPOCHS)
    crit  = nn.CrossEntropyLoss()
    hist  = {'tl': [], 'vl': [], 'va': []}
    for ep in range(NUM_EPOCHS):
        model.train()
        tl = 0.0
        for X, y in tr_dl:
            X, y = X.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(X), y)
            loss.backward()
            opt.step()
            tl += loss.item() * len(y)
        model.eval()
        vl = cor = tot = 0
        with torch.no_grad():
            for X, y in va_dl:
                X, y = X.to(DEVICE), y.to(DEVICE)
                o = model(X)
                vl  += crit(o, y).item() * len(y)
                cor += (o.argmax(1) == y).sum().item()
                tot += len(y)
        sch.step()
        hist['tl'].append(tl / n_tr)
        hist['vl'].append(vl / n_val)
        hist['va'].append(cor / tot)
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f'       Ep {ep+1:2d}/{NUM_EPOCHS}  '
                  f'TL={hist["tl"][-1]:.4f}  '
                  f'VL={hist["vl"][-1]:.4f}  '
                  f'VA={hist["va"][-1]:.3f}')
    print(f'       Best Val Accuracy: {max(hist["va"]):.3f}')
    return model, hist, n_cls

# ==========================================================
# 6. Full-image prediction
# ==========================================================
def predict_full(model, cube, n_cls):
    print('\n[5/6] Full-image sliding-window inference...')
    pad = PATCH_SIZE // 2
    H, W = cube.shape[1], cube.shape[2]
    norm = cube.copy()
    for b in range(cube.shape[0]):
        v = cube[b][np.isfinite(cube[b])]
        if v.size:
            norm[b] = (cube[b] - v.mean()) / (v.std() + 1e-8)
    norm    = np.nan_to_num(norm, nan=0.0)
    nd_mask = ~np.isfinite(cube).all(axis=0)
    pred = np.zeros((H, W), dtype=np.uint8)
    conf = np.zeros((H, W), dtype=np.float32)
    model.eval()
    bufs, coords = [], []
    def flush():
        if not bufs:
            return
        X = torch.tensor(np.stack(bufs)).to(DEVICE)
        with torch.no_grad():
            pb = torch.softmax(model(X), 1).cpu().numpy()
        for (r, c), p in zip(coords, pb):
            pred[r, c] = int(p.argmax()) + 1
            conf[r, c] = float(p.max())
        bufs.clear()
        coords.clear()
    step = max(1, (H - 2*pad) // 10)
    for i, r in enumerate(range(pad, H - pad)):
        for c in range(pad, W - pad):
            if nd_mask[r, c]:
                continue
            bufs.append(norm[:, r-pad:r+pad, c-pad:c+pad].astype(np.float32))
            coords.append((r, c))
            if len(bufs) >= BATCH_SIZE * 8:
                flush()
        if (i + 1) % step == 0:
            print(f'       {100*(i+1)//(H-2*pad)}%...')
    flush()
    return pred, conf

# ==========================================================
# 7. Evaluate and plot
# ==========================================================
def evaluate_and_plot(pred, conf, lbl, hist, profile):
    print('\n[6/6] Evaluating and saving outputs...')
    cls_names = ['Water', 'Glacier', 'Land/Veg']
    mask = lbl > 0
    yt   = (lbl[mask] - 1).astype(int)
    yp   = (pred[mask].astype(int) - 1)
    good = yp >= 0
    yt, yp = yt[good], yp[good]
    oa = f1 = 0
    if SK_OK and len(yt):
        oa = accuracy_score(yt, yp)
        f1 = f1_score(yt, yp, average='macro', zero_division=0)
        print(f'       Overall Accuracy: {oa:.3f}  Macro-F1: {f1:.3f}')
        print(classification_report(yt, yp, target_names=cls_names, zero_division=0))
    cmap = mcolors.ListedColormap(['#111111', '#1f77b4', '#00d4ff', '#2ca02c'])
    fig, axs = plt.subplots(2, 3, figsize=(18, 11))
    fig.patch.set_facecolor('#1a1a2e')
    fig.suptitle('Chapter 9 -- CNN Land Cover Classification\n'
                 'Multi-Sensor: S2 NIR + SAR VV + DEM + MODIS LST',
                 fontsize=13, fontweight='bold', color='white')
    for ax in axs.flat:
        ax.set_facecolor('#16213e')
    axs[0,0].imshow(pred, cmap=cmap, vmin=0, vmax=3, aspect='auto')
    axs[0,0].set_title('CNN Prediction', color='white'); axs[0,0].axis('off')
    im = axs[0,1].imshow(conf, cmap='plasma', vmin=0.4, vmax=1.0, aspect='auto')
    plt.colorbar(im, ax=axs[0,1]).set_label('Confidence', color='white')
    axs[0,1].set_title('Prediction Confidence', color='white'); axs[0,1].axis('off')
    axs[0,2].imshow(lbl, cmap=cmap, vmin=0, vmax=3, aspect='auto')
    axs[0,2].set_title('Physics-Based Labels', color='white'); axs[0,2].axis('off')
    ep = range(1, len(hist['tl']) + 1)
    axs[1,0].plot(ep, hist['tl'], 'b-', lw=1.5, label='Train')
    axs[1,0].plot(ep, hist['vl'], 'r-', lw=1.5, label='Val')
    axs[1,0].set_title('Training Loss', color='white')
    axs[1,0].legend(facecolor='#0f3460', labelcolor='white')
    axs[1,0].set_facecolor('#0f3460'); axs[1,0].tick_params(colors='white')
    axs[1,1].plot(ep, [a*100 for a in hist['va']], 'g-', lw=1.5)
    axs[1,1].set_title('Validation Accuracy (%)', color='white')
    axs[1,1].set_ylim(0, 105); axs[1,1].set_facecolor('#0f3460')
    axs[1,1].tick_params(colors='white')
    if SK_OK and len(yt):
        ConfusionMatrixDisplay(confusion_matrix(yt, yp),
                               display_labels=cls_names).plot(
            ax=axs[1,2], colorbar=False, cmap='Blues')
        axs[1,2].set_title(f'Confusion Matrix (OA={oa:.2%})', color='white')
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, 'cnn_landcover_results.png')
    fig.savefig(fig_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'       [SUCCESS] Figure: {fig_path}')
    pp = profile.copy()
    pp.update(count=1, dtype=rasterio.uint8, nodata=0, compress='lzw')
    with rasterio.open(os.path.join(OUT_DIR, 'cnn_landcover_prediction.tif'), 'w', **pp) as dst:
        dst.write(pred.astype(np.uint8), 1)
        dst.update_tags(classes='1=Water,2=Glacier,3=Land_Veg', method='CNN_3layer')
    cp = profile.copy()
    cp.update(count=1, dtype=rasterio.float32, nodata=-9999, compress='lzw')
    with rasterio.open(os.path.join(OUT_DIR, 'cnn_confidence_map.tif'), 'w', **cp) as dst:
        dst.write(np.where(pred == 0, -9999, conf).astype(np.float32), 1)
        dst.update_tags(description='CNN prediction confidence per pixel [0-1]')
    # Export training history CSV
    if hist['tl']:
        pd.DataFrame({'epoch': range(1, len(hist['tl'])+1),
                      'train_loss': hist['tl'],
                      'val_loss':   hist['vl'],
                      'val_acc':    hist['va']}
                     ).to_csv(os.path.join(OUT_DIR, 'cnn_training_history.csv'),
                              index=False, encoding='utf-8')
    # Export class metrics CSV
    if SK_OK and len(yt):
        from sklearn.metrics import precision_recall_fscore_support
        p, r, f, s = precision_recall_fscore_support(yt, yp, average=None, zero_division=0)
        pd.DataFrame({'class': cls_names, 'precision': p.round(4),
                      'recall': r.round(4), 'f1': f.round(4), 'support': s}
                     ).to_csv(os.path.join(OUT_DIR, 'cnn_class_metrics.csv'),
                              index=False, encoding='utf-8')
    print(f'       [SUCCESS] GeoTIFFs + CSVs saved: {OUT_DIR}')

# ==========================================================
# Main
# ==========================================================
def main():
    print('=' * 60)
    print(' GEOCASCADE -- CHAPTER 9: CNN LAND COVER CLASSIFICATION')
    print(f' Patch={PATCH_SIZE}px  Epochs={NUM_EPOCHS}  LR={LR}')
    print('=' * 60)
    if not TORCH_OK:
        print('[ERROR] PyTorch required. Install with:')
        print('  mamba install -n geocascade_env -c conda-forge pytorch torchvision cpuonly -y')
        return
    try:
        cube, prof = load_cube()
        lbl = make_labels(cube)
        print(f'\n[3/6] Building patch dataset (max {MAX_PATCHES:,} patches)...')
        torch.manual_seed(SEED)
        ds = PatchDS(cube, lbl)
        print(f'       Total labeled patches: {len(ds):,}')
        if len(ds) < 50:
            raise ValueError('Too few labeled patches -- check the cube for valid pixels.')
        model, hist, n_cls = train_model(ds)
        pred, conf = predict_full(model, cube, n_cls)
        evaluate_and_plot(pred, conf, lbl, hist, prof)
        mw = os.path.join(OUT_DIR, 'cnn_landcover_model.pt')
        torch.save(model.state_dict(), mw)
        print(f'       [SUCCESS] Weights saved: {mw}')
        print('\n' + '=' * 60)
        print(' CHAPTER 9 COMPLETE')
        print(f' Results: {OUT_DIR}')
        print(' Next: Chapter_10/25_agentic_monitor.py')
        print('=' * 60)
    except FileNotFoundError as e:
        print(f'\n[ERROR] {e}')
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
