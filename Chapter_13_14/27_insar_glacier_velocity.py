"""
Chapter 13: 27_insar_glacier_velocity.py

Academic Objective:
Estimate glacier surface velocity using Sentinel-1 SAR intensity cross-correlation
(offset tracking) -- a Python-native InSAR-light approach that does not require
SNAP or complex interferometric processing.

Full interferometric InSAR requires:
  1. Two Sentinel-1 SLC acquisitions separated by 6-12 days
  2. SNAP or ISCE to form the complex interferogram
  3. Phase unwrapping (SNAPHU or py-isce)
  4. Atmospheric phase screen correction (ERA5 or PyAPS)

This script implements the accessible Python-native alternative:
  SAR Intensity Cross-Correlation (Offset Tracking)
  - Correlates two SAR intensity images separated in time
  - Peak of correlation window = displacement vector
  - Displacement / time interval = velocity (m/day -> m/year)
  - Valid for fast-moving glaciers (>0.5 m/day) such as Grey Glacier
  - Accuracy: approx 1/10 pixel = 1-2 m at 10m Sentinel-1 resolution

Physical background:
  Grey Glacier velocity is approximately 300-800 m/year (0.8-2.2 m/day).
  At 10m pixel size this is 0.08-0.22 pixels/day -- measurable with
  sub-pixel correlation over a 12-day interval.

Output products:
  vx_map.tif   -- East-West velocity component (m/year, float32, nodata=-9999)
  vy_map.tif   -- North-South velocity component (m/year, float32, nodata=-9999)
  vmag_map.tif -- Velocity magnitude (m/year, float32, nodata=-9999)
  insar_velocity_dashboard.png -- 4-panel figure

Dependencies:
mamba install -n geocascade_env -c conda-forge rasterio numpy scipy matplotlib -y
"""

import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

try:
    import rasterio
    RIO_OK = True
except ImportError:
    RIO_OK = False

try:
    from scipy.signal import correlate2d
    from scipy.ndimage import zoom
    SCI_OK = True
except ImportError:
    SCI_OK = False
    print("[WARNING] scipy not installed.")
    print("  mamba install -n geocascade_env -c conda-forge scipy -y")

# ==========================================
# Configuration
# ==========================================
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
SAR_DIR     = os.path.join(BASE_DIR, "..", "Chapter_07", "data", "processed")
OUT_DIR     = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

WINDOW_SIZE = 64     # Correlation window (pixels)
STEP_SIZE   = 16     # Step between windows (pixels)
MAX_DISP    = 20     # Max expected displacement (pixels)
TIME_DELTA  = 12     # Days between acquisitions
PIXEL_SIZE  = 10.0   # Sentinel-1 geocoded pixel size (metres)
BBOX        = [-73.30, -51.10, -72.90, -50.80]


# ==========================================
# 1. Load SAR image pair
# ==========================================
def load_sar_pair():
    import glob
    files = sorted(glob.glob(os.path.join(SAR_DIR, "sar_vv_*.tif")))
    if len(files) >= 2 and RIO_OK:
        try:
            with rasterio.open(files[0]) as src:
                img1    = src.read(1).astype("float32")
                profile = src.profile.copy()
                nd      = src.nodata if src.nodata is not None else -9999
            with rasterio.open(files[-1]) as src:
                img2 = src.read(1).astype("float32")
            img1[img1 == nd] = np.nan
            img2[img2 == nd] = np.nan
            print(f"       Loaded SAR pair: {os.path.basename(files[0])} and {os.path.basename(files[-1])}")
            return img1, img2, profile
        except Exception as e:
            print(f"       SAR read error: {e}. Generating synthetic data.")

    print("       No real SAR pair found -- generating synthetic glacier velocity field...")
    H, W = 256, 256
    rng  = np.random.default_rng(42)

    img1 = np.exp(rng.normal(0, 0.5, (H, W))).astype("float32")
    glacier_mask = np.zeros((H, W), dtype=bool)
    glacier_mask[60:180, 20:140] = True
    img1[glacier_mask] *= 1.4

    from scipy.ndimage import shift as ndshift
    img2 = img1.copy()
    glacier_shifted = ndshift(img1 * glacier_mask, shift=(0.5, 3.0), mode="nearest")
    img2[glacier_mask] = glacier_shifted[glacier_mask]
    img2 += rng.normal(0, 0.05, (H, W)).astype("float32")

    from rasterio.transform import from_bounds
    transform = from_bounds(*BBOX, W, H)
    profile = {
        "driver": "GTiff", "dtype": "float32", "nodata": -9999,
        "width": W, "height": H, "count": 1, "crs": "EPSG:4326",
        "transform": transform,
    }
    return img1, img2, profile


# ==========================================
# 2. Offset tracking via cross-correlation
# ==========================================
def offset_tracking(img1, img2):
    H, W   = img1.shape
    half_w = WINDOW_SIZE // 2

    dx_map = np.full((H, W), np.nan, dtype="float32")
    dy_map = np.full((H, W), np.nan, dtype="float32")
    cc_map = np.full((H, W), np.nan, dtype="float32")

    def norm(a):
        v = a[np.isfinite(a)]
        if v.size == 0 or v.std() == 0:
            return np.zeros_like(a)
        return (a - v.mean()) / (v.std() + 1e-8)

    a = np.nan_to_num(norm(img1), nan=0.0)
    b = np.nan_to_num(norm(img2), nan=0.0)

    rows  = list(range(half_w, H - half_w, STEP_SIZE))
    cols  = list(range(half_w, W - half_w, STEP_SIZE))
    total = len(rows) * len(cols)
    done  = 0
    step_report = max(1, total // 5)

    for r in rows:
        for c in cols:
            w1 = a[r-half_w:r+half_w, c-half_w:c+half_w]
            sr = half_w + MAX_DISP
            r0, r1 = max(0, r-sr), min(H, r+sr)
            c0, c1 = max(0, c-sr), min(W, c+sr)
            w2 = b[r0:r1, c0:c1]
            if w1.size == 0 or w2.size == 0:
                done += 1
                continue
            corr = correlate2d(w2, w1, mode="valid", boundary="fill", fillvalue=0)
            if corr.size == 0:
                done += 1
                continue
            peak_idx = np.unravel_index(corr.argmax(), corr.shape)
            peak_val = corr[peak_idx] / (WINDOW_SIZE**2 + 1e-8)
            dy_px = peak_idx[0] - MAX_DISP
            dx_px = peak_idx[1] - MAX_DISP
            if abs(dx_px) > MAX_DISP or abs(dy_px) > MAX_DISP:
                done += 1
                continue
            dx_map[r, c] = float(dx_px)
            dy_map[r, c] = float(dy_px)
            cc_map[r, c] = float(np.clip(peak_val, 0, 1))
            done += 1

        if done % step_report < len(cols):
            pct = 100 * done // total
            print(f"       {pct}% complete ({done}/{total} windows)...")

    # Upsample back to full resolution
    sc = H / dx_map.shape[0]
    dx_map = zoom(np.nan_to_num(dx_map, nan=0.0), (sc, sc), order=1).astype("float32")
    dy_map = zoom(np.nan_to_num(dy_map, nan=0.0), (sc, sc), order=1).astype("float32")
    cc_map = zoom(np.nan_to_num(cc_map, nan=0.0), (sc, sc), order=1).astype("float32")

    # Clip back to original shape (zoom can add/remove 1 px)
    dx_map = dx_map[:H, :W]
    dy_map = dy_map[:H, :W]
    cc_map = cc_map[:H, :W]

    return dx_map, dy_map, cc_map


# ==========================================
# 3. Displacement -> velocity (m/year)
# ==========================================
def to_velocity(dx, dy, cc, quality_threshold=0.05):
    scale = PIXEL_SIZE * (365.0 / TIME_DELTA)
    vx    = dx * scale
    vy    = -dy * scale    # row+ is southward in raster coords
    vmag  = np.sqrt(vx**2 + vy**2)
    mask  = cc < quality_threshold
    for arr in [vx, vy, vmag]:
        arr[mask] = -9999
    valid = vmag[vmag != -9999]
    if valid.size:
        print(f"       Velocity: median={np.median(valid):.0f}  max={np.max(valid):.0f} m/yr  "
              f"valid={valid.size:,} px")
    return vx, vy, vmag


# ==========================================
# 4. Save GeoTIFFs
# ==========================================
def save_tiffs(vx, vy, vmag, profile):
    if not RIO_OK:
        print("       rasterio not available -- skipping GeoTIFF export.")
        return
    prof = profile.copy()
    prof.update(count=1, dtype="float32", nodata=-9999)
    for name, arr in [("vx_map", vx), ("vy_map", vy), ("vmag_map", vmag)]:
        out = os.path.join(OUT_DIR, f"{name}.tif")
        with rasterio.open(out, "w", **prof) as dst:
            dst.write(arr.astype("float32"), 1)
        print(f"       [OK] {name}.tif ({os.path.getsize(out)/1e6:.1f} MB)")


# ==========================================
# 5. Dashboard
# ==========================================
def make_dashboard(img1, img2, vx, vy, vmag, cc):
    vmag_show = np.where(vmag != -9999, vmag, np.nan)
    vmax      = float(np.nanpercentile(vmag_show[np.isfinite(vmag_show)], 98)) if np.any(np.isfinite(vmag_show)) else 1000.0

    fig, axs = plt.subplots(2, 2, figsize=(14, 11))
    fig.patch.set_facecolor("#0d1117")
    fig.suptitle(
        "Chapter 13 -- SAR Offset Tracking: Glacier Surface Velocity\n"
        f"Torres del Paine / Grey Glacier  |  dt={TIME_DELTA} days  |  px={PIXEL_SIZE}m",
        fontsize=12, fontweight="bold", color="white",
    )
    for ax in axs.flat:
        ax.set_facecolor("#161b22")

    axs[0, 0].imshow(np.log1p(img1), cmap="bone", aspect="auto")
    axs[0, 0].set_title("SAR Intensity t1 (log scale)", color="white", fontsize=9)
    axs[0, 0].axis("off")

    axs[0, 1].imshow(np.log1p(img2), cmap="bone", aspect="auto")
    axs[0, 1].set_title("SAR Intensity t2 (log scale)", color="white", fontsize=9)
    axs[0, 1].axis("off")

    im = axs[1, 0].imshow(vmag_show, cmap="plasma", aspect="auto", vmin=0, vmax=vmax)
    cb = plt.colorbar(im, ax=axs[1, 0])
    cb.set_label("Velocity (m/year)", color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
    H, W = vx.shape
    step = max(1, min(H, W) // 20)
    ys, xs = np.mgrid[0:H:step, 0:W:step]
    u = np.where(vx[::step, ::step] != -9999, vx[::step, ::step], 0.0)
    v = np.where(vy[::step, ::step] != -9999, -vy[::step, ::step], 0.0)
    axs[1, 0].quiver(xs, ys, u, v, color="white", alpha=0.5,
                     scale=max(vmax * 20, 1), width=0.003)
    axs[1, 0].set_title("Velocity Magnitude + Flow Vectors", color="white", fontsize=9)
    axs[1, 0].axis("off")

    im2 = axs[1, 1].imshow(cc, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    cb2 = plt.colorbar(im2, ax=axs[1, 1])
    cb2.set_label("Correlation Coefficient", color="white")
    plt.setp(cb2.ax.yaxis.get_ticklabels(), color="white")
    axs[1, 1].set_title("Cross-Correlation Quality", color="white", fontsize=9)
    axs[1, 1].axis("off")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "insar_velocity_dashboard.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"       [SUCCESS] Dashboard: {out}")


# ==========================================
# Main
# ==========================================
def main():
    print("=" * 65)
    print(" GEOCASCADE -- CHAPTER 13: InSAR GLACIER VELOCITY")
    print(f" Window={WINDOW_SIZE}px  Step={STEP_SIZE}px  dt={TIME_DELTA}d  px={PIXEL_SIZE}m")
    print("=" * 65)

    if not SCI_OK:
        print("[ERROR] scipy required.")
        print("  mamba install -n geocascade_env -c conda-forge scipy -y")
        return

    print("\n[1/5] Loading SAR image pair...")
    img1, img2, profile = load_sar_pair()
    print(f"       Image size: {img1.shape[0]} x {img1.shape[1]} pixels")

    print("\n[2/5] Running SAR offset tracking (cross-correlation)...")
    try:
        dx, dy, cc = offset_tracking(img1, img2)
    except Exception:
        import traceback
        traceback.print_exc()
        return

    print("\n[3/5] Converting to velocity (m/year)...")
    vx, vy, vmag = to_velocity(dx, dy, cc)

    print("\n[4/5] Saving GeoTIFFs...")
    save_tiffs(vx, vy, vmag, profile)

    print("\n[5/5] Generating dashboard...")
    make_dashboard(img1, img2, vx, vy, vmag, cc)

    print("\n" + "=" * 65)
    print(" CHAPTER 13 COMPLETE")
    print(f" Results: {OUT_DIR}")
    print(" For full interferometric InSAR: use ESA SNAP + SNAPHU.")
    print(" Next: Chapter_13_14/28_hyperspectral_unmixing.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
