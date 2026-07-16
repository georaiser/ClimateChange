"""
Chapter 14: 28_hyperspectral_unmixing.py

Academic Objective:
Implement Linear Spectral Unmixing (LSU) to decompose each Sentinel-2 pixel
into fractional contributions (abundances) of pure spectral signatures (endmembers).

Why spectral unmixing?
  Every pixel in a 10m remote sensing image over a mountain landscape is
  typically a MIXTURE of multiple land cover types at sub-pixel scale.
  A pixel on a glacier edge might contain 40% ice, 35% meltwater, 25% moraine.
  Standard classification assigns ONE label per pixel -- unmixing recovers fractions.

Linear Mixing Model (LMM):
  r = A * x + e
  where:
    r  = observed reflectance (B bands)
    A  = endmember matrix (B x N_endmembers)
    x  = abundance vector (N classes, sums to 1, all >= 0)
    e  = noise residual

  Solved as FCLS (Fully Constrained Least Squares):
    min ||r - Ax||^2   s.t.  sum(x) = 1,  x >= 0

Endmembers (4 classes for Torres del Paine):
  1. Glacier/Snow   -- high reflectance all bands, decreasing toward SWIR
  2. Open Water     -- very low all bands, near-zero NIR/SWIR
  3. Dense Veg      -- low Red, high NIR (red-edge jump), moderate SWIR
  4. Bare Rock      -- moderate, flat spectrum, increasing toward SWIR

Dependencies:
mamba install -n geocascade_env -c conda-forge rasterio numpy scipy matplotlib scikit-learn -y
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
warnings.filterwarnings("ignore")

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    import rasterio
    RIO_OK = True
except ImportError:
    RIO_OK = False

try:
    from scipy.optimize import nnls
    SCI_OK = True
except ImportError:
    SCI_OK = False
    print("[WARNING] scipy not installed.")
    print("  mamba install -n geocascade_env -c conda-forge scipy -y")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(BASE_DIR, "data", "processed", "hyperspectral")
os.makedirs(OUT_DIR, exist_ok=True)

ENDMEMBER_NAMES   = ["Glacier/Snow", "Open Water", "Dense Vegetation", "Bare Rock"]
ENDMEMBER_COLOURS = ["#00d4ff", "#1f77b4", "#2ca02c", "#8c564b"]
BAND_NAMES        = ["B02 Blue", "B03 Green", "B04 Red", "B08 NIR", "B11 SWIR"]
BAND_WL           = [490, 560, 665, 842, 1610]


# ==========================================
# 1. Load image
# ==========================================
def load_image():
    import glob
    s2_dirs = glob.glob(
        os.path.join(BASE_DIR, "..", "Chapter_01", "data", "raw", "sentinel2_l2a_*")
    )
    if s2_dirs and RIO_OK:
        s2_dir = max(s2_dirs, key=os.path.getmtime)
        bands  = []
        profile = None
        for bnd in ["B02", "B03", "B04", "B08", "B11"]:
            path = os.path.join(s2_dir, f"{bnd}.tif")
            if os.path.exists(path):
                with rasterio.open(path) as src:
                    data = src.read(1).astype("float32")
                    nd   = src.nodata if src.nodata is not None else 0
                    data[data == nd] = np.nan
                    bands.append(data / 10000.0)
                    if profile is None:
                        profile = src.profile.copy()
        if len(bands) == 5:
            cube = np.stack(bands, axis=0)
            print(f"       Loaded S2 stack: {cube.shape}")
            return cube, profile

    print("       No real S2 stack found -- generating synthetic 5-band Patagonian scene...")
    H, W = 256, 256
    rng  = np.random.default_rng(42)

    # Physics-based endmember spectra (approximate S2 surface reflectance)
    em = np.array([
        [0.65, 0.60, 0.55, 0.50, 0.20],   # Glacier/Snow
        [0.10, 0.06, 0.03, 0.01, 0.00],   # Open Water
        [0.04, 0.08, 0.04, 0.45, 0.18],   # Dense Vegetation
        [0.20, 0.22, 0.25, 0.28, 0.30],   # Bare Rock
    ], dtype="float32").T   # shape (5, 4)

    abund = np.zeros((4, H, W), dtype="float32")
    abund[0, :H//2, :W//2]             = rng.uniform(0.5, 1.0, (H//2, W//2))
    abund[1, 3*H//4:, W//4:3*W//4]    = rng.uniform(0.6, 1.0, (H//4, W//2))
    abund[2, H//4:3*H//4, W//2:]      = rng.uniform(0.4, 0.9, (H//2, W//2))
    abund[3] = np.clip(1 - abund.sum(axis=0), 0, 1)
    total    = abund.sum(axis=0)
    total    = np.where(total == 0, 1, total)
    abund   /= total

    cube = np.einsum("bk,khw->bhw", em, abund)
    cube += rng.normal(0, 0.01, cube.shape).astype("float32")
    cube  = np.clip(cube, 0, 1).astype("float32")

    from rasterio.transform import from_bounds
    profile = {
        "driver": "GTiff", "dtype": "float32", "nodata": -9999,
        "width": W, "height": H, "count": 5, "crs": "EPSG:4326",
        "transform": from_bounds(-73.30, -51.10, -72.90, -50.80, W, H),
    }
    return cube, profile


# ==========================================
# 2. Extract endmember spectra
# ==========================================
def extract_endmembers(cube):
    B02, B03, B04, B08, B11 = cube

    NDSI = (B03 - B11) / (B03 + B11 + 1e-8)
    NDVI = (B08 - B04) / (B08 + B04 + 1e-8)
    NDWI = (B03 - B08) / (B03 + B08 + 1e-8)
    Rock = (B04 + B11) * 0.5

    defaults = {
        "glacier": np.array([0.65, 0.60, 0.55, 0.50, 0.20], dtype="float32"),
        "water":   np.array([0.10, 0.06, 0.03, 0.01, 0.00], dtype="float32"),
        "veg":     np.array([0.04, 0.08, 0.04, 0.45, 0.18], dtype="float32"),
        "rock":    np.array([0.20, 0.22, 0.25, 0.28, 0.30], dtype="float32"),
    }

    def purest(score, n=50):
        flat  = score.ravel()
        valid = np.isfinite(flat)
        if not valid.any():
            return None
        tmp = flat.copy()
        tmp[~valid] = -999
        idx = np.argsort(tmp)[-n:]
        rs, cs = np.unravel_index(idx, score.shape)
        specs = np.stack([cube[b][rs, cs] for b in range(cube.shape[0])], axis=1)
        return np.median(specs, axis=0).astype("float32")

    endmembers = []
    for em_score, key in [(NDSI, "glacier"), (NDWI, "water"), (NDVI, "veg"), (Rock, "rock")]:
        em = purest(em_score)
        if em is None or not np.isfinite(em).all() or em.max() <= 0:
            print(f"       [INFO] Using physics default for {key}")
            endmembers.append(defaults[key])
        else:
            endmembers.append(em)

    A = np.column_stack(endmembers)   # (B, N)
    print(f"       Endmember matrix: {A.shape}  (bands x classes)")
    for i, name in enumerate(ENDMEMBER_NAMES):
        print(f"         {name:22s}: {A[:, i].round(3)}")
    return A


# ==========================================
# 3. FCLS Unmixing
# ==========================================
def unmix(cube, A):
    """
    Fully Constrained Least Squares (FCLS) unmixing.

    Augment A with a sum-to-one row (weight lambda) and solve with nnls
    (non-negative least squares) -- satisfies both constraints simultaneously.
    """
    B, H, W = cube.shape
    N       = A.shape[1]
    lam     = 10.0
    A_aug   = np.vstack([A, lam * np.ones((1, N))])   # (B+1, N)

    pixels  = cube.reshape(B, -1).T   # (H*W, B)
    ones_v  = np.full((H*W, 1), lam, dtype="float32")
    pix_aug = np.hstack([pixels, ones_v])   # (H*W, B+1)

    abund   = np.zeros((H*W, N), dtype="float32")
    valid   = np.isfinite(pixels).all(axis=1)
    total   = valid.sum()
    step    = max(1, total // 5)
    done    = 0
    print(f"       FCLS: {total:,} valid pixels to unmix...")

    for i in range(H*W):
        if not valid[i]:
            continue
        x, _ = nnls(A_aug.T, pix_aug[i])
        s     = x.sum()
        abund[i] = x / s if s > 0 else x
        done += 1
        if done % step == 0:
            print(f"       {100*done//total}% unmixed...")

    maps = abund.T.reshape(N, H, W)
    for n in range(N):
        maps[n][~valid.reshape(H, W)] = -9999
    return maps


# ==========================================
# 4. Residual
# ==========================================
def compute_residual(cube, A, maps):
    B, H, W = cube.shape
    recon   = np.einsum("bn,nhw->bhw", A,
                        np.where(maps == -9999, 0.0, maps).astype("float32"))
    residual = np.sqrt(np.mean((cube - recon)**2, axis=0))
    residual[maps[0] == -9999] = np.nan
    return residual


# ==========================================
# 5. Save GeoTIFFs
# ==========================================
def save_tiffs(maps, profile):
    if not RIO_OK:
        print("       rasterio not available -- skipping GeoTIFF export.")
        return
    prof = profile.copy()
    prof.update(count=1, dtype="float32", nodata=-9999)
    for i, name in enumerate(ENDMEMBER_NAMES):
        fname = name.lower().replace("/", "_").replace(" ", "_")
        out   = os.path.join(OUT_DIR, f"abundance_{fname}.tif")
        with rasterio.open(out, "w", **prof) as dst:
            dst.write(maps[i].astype("float32"), 1)
        print(f"       [OK] abundance_{fname}.tif  ({os.path.getsize(out)/1e6:.1f} MB)")


# ==========================================
# 6. Dashboard
# ==========================================
def make_dashboard(cube, A, maps, residual):
    N = len(ENDMEMBER_NAMES)
    fig, axs = plt.subplots(2, N + 1, figsize=(4*(N+1), 9))
    fig.patch.set_facecolor("#0d1117")
    fig.suptitle(
        "Chapter 14 -- Linear Spectral Unmixing\n"
        "Sentinel-2: Glacier / Water / Vegetation / Bare Rock Fractions",
        fontsize=12, fontweight="bold", color="white",
    )
    for ax in axs.flat:
        ax.set_facecolor("#161b22")

    # Top-left: Endmember spectra
    ax = axs[0, 0]
    for i in range(N):
        ax.plot(BAND_WL, A[:, i], "o-", color=ENDMEMBER_COLOURS[i],
                lw=2, ms=5, label=ENDMEMBER_NAMES[i])
    ax.set_xlabel("Wavelength (nm)", color="white", fontsize=8)
    ax.set_ylabel("Reflectance", color="white", fontsize=8)
    ax.set_title("Endmember Spectra", color="white", fontsize=9)
    ax.legend(fontsize=7, facecolor="#0d1117", labelcolor="white")
    ax.tick_params(colors="white")
    ax.set_facecolor("#0f3460")

    # Top: Abundance maps
    for i, (name, colour) in enumerate(zip(ENDMEMBER_NAMES, ENDMEMBER_COLOURS)):
        ax = axs[0, i + 1]
        data = np.where(maps[i] == -9999, np.nan, maps[i])
        im   = ax.imshow(data, cmap="hot", aspect="auto", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046).set_label("Fraction", color="white")
        ax.set_title(name, color=colour, fontsize=8)
        ax.axis("off")

    # Bottom-left: False colour composite
    ax = axs[1, 0]

    def norm01(band):
        v = band[np.isfinite(band)]
        if v.size == 0:
            return np.zeros_like(band)
        p2, p98 = np.nanpercentile(v, 2), np.nanpercentile(v, 98)
        return np.clip((band - p2) / (p98 - p2 + 1e-8), 0, 1)

    rgb = np.stack([norm01(cube[3]), norm01(cube[2]), norm01(cube[1])], axis=-1)
    ax.imshow(np.nan_to_num(rgb, nan=0.0), aspect="auto")
    ax.set_title("False Colour NIR-R-G", color="white", fontsize=8)
    ax.axis("off")

    # Bottom: Dominant endmember + residual
    valid   = maps[0] != -9999
    dom_map = np.where(valid,
                       np.argmax(np.where(maps == -9999, -1, maps), axis=0),
                       -1).astype(float)
    dom_map[~valid] = np.nan

    cmap_em = mcolors.ListedColormap(["#111111"] + ENDMEMBER_COLOURS)
    ax2 = axs[1, 1]
    ax2.imshow(dom_map, cmap=cmap_em, vmin=-1, vmax=N - 1, aspect="auto")
    ax2.set_title("Dominant Endmember", color="white", fontsize=8)
    ax2.axis("off")

    ax3 = axs[1, 2]
    if residual is not None:
        im3 = ax3.imshow(residual, cmap="Reds", aspect="auto")
        plt.colorbar(im3, ax=ax3, fraction=0.046).set_label("RMSE", color="white")
    ax3.set_title("Unmixing Residual", color="white", fontsize=8)
    ax3.axis("off")

    # Composite: RGB of first 3 endmembers
    ax4 = axs[1, 3]
    comp = np.stack([
        np.clip(np.where(maps[0] == -9999, 0, maps[0]), 0, 1),
        np.clip(np.where(maps[2] == -9999, 0, maps[2]), 0, 1),
        np.clip(np.where(maps[1] == -9999, 0, maps[1]), 0, 1),
    ], axis=-1)
    ax4.imshow(comp, aspect="auto")
    ax4.set_title("RGB: Glacier(R) Veg(G) Water(B)", color="white", fontsize=8)
    ax4.axis("off")

    # Turn off remaining axes
    for i in range(4, N + 1):
        axs[1, i].axis("off")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "hyperspectral_unmixing_dashboard.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"       [SUCCESS] Dashboard: {out}")


# ==========================================
# Main
# ==========================================
def main():
    print("=" * 65)
    print(" GEOCASCADE -- CHAPTER 14: HYPERSPECTRAL UNMIXING (FCLS)")
    print(f" Endmembers: {', '.join(ENDMEMBER_NAMES)}")
    print("=" * 65)

    if not SCI_OK:
        print("[ERROR] scipy required (scipy.optimize.nnls).")
        print("  mamba install -n geocascade_env -c conda-forge scipy -y")
        return

    print("\n[1/5] Loading Sentinel-2 5-band image...")
    cube, profile = load_image()

    print("\n[2/5] Extracting endmember spectra...")
    A = extract_endmembers(cube)

    print("\n[3/5] Running FCLS unmixing...")
    try:
        maps = unmix(cube, A)
    except Exception:
        import traceback
        traceback.print_exc()
        return

    print("\n       Mean abundance per endmember:")
    for i, name in enumerate(ENDMEMBER_NAMES):
        valid = maps[i][maps[i] != -9999]
        if valid.size:
            print(f"         {name:22s}: {np.mean(valid):.3f}  "
                  f"(max={np.max(valid):.3f})")

    print("\n[4/5] Computing residual and saving outputs...")
    try:
        residual = compute_residual(cube, A, maps)
    except Exception:
        residual = None
    save_tiffs(maps, profile)

    print("\n[5/5] Generating dashboard...")
    make_dashboard(cube, A, maps, residual)

    print("\n" + "=" * 65)
    print(" CHAPTER 14 COMPLETE -- GEOCASCADE PIPELINE FINISHED!")
    print(f" Results: {OUT_DIR}")
    print(" Full pipeline guide: PIPELINE_GUIDE.md")
    print("=" * 65)


if __name__ == "__main__":
    main()
