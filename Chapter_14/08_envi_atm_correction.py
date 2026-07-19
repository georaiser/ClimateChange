"""
08_envi_atm_correction.py
==========================
GeoCascade Chapter 14 -- ArcGIS Pro + ENVI Professional Workflows
Torres del Paine, Patagonia, Chile

# NOTE: This script mirrors the ENVI Python API workflow.
# In production with ENVI + FLAASH license:
#   envi = ENVI()
#   task = envi.Task('FLAASH')
#   task['INPUT_RASTER'] = raster
#   task['SENSOR_TYPE'] = 'Sentinel-2'
#   task.execute()

PURPOSE
-------
Demonstrates atmospheric correction of Sentinel-2 imagery.
Two methods are implemented:

  1. DOS1 (Dark Object Subtraction) -- open-source fallback
     - For each band, find the minimum non-zero DN (dark object = deep shadow)
     - Subtract this minimum from all pixels (removes additive path radiance)
     - Scale to 0-1 surface reflectance
     - Fast but approximate: does not account for multiplicative scattering

  2. FLAASH workflow description (requires ENVI + Atmospheric Correction Module)
     - Converts DN to radiance using sensor-specific gain/offset
     - Runs MODTRAN atmospheric RTM model
     - Outputs surface reflectance with per-pixel water vapour retrieval
     - Recommended for quantitative analysis

OUTPUTS
-------
  data/processed/envi_outputs/sentinel2_dos1_corrected.tif  -- DOS1 corrected
  data/processed/envi_outputs/atm_correction_report.png     -- before/after figure

ENVI 5.6 EQUIVALENT
--------------------
  Toolbox > Radiometric Correction > Atmospheric Correction Module > QUAC
  Toolbox > Radiometric Correction > Atmospheric Correction Module > FLAASH
  (FLAASH requires Atmospheric Correction Module license)

ARCGIS PRO EQUIVALENT
----------------------
  ArcGIS Pro does not include FLAASH. Use ENVI or Sen2Cor for correction,
  then load the corrected TIF into ArcGIS Pro.

RUN
---
  python Chapter_14/08_envi_atm_correction.py
"""

import sys
import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

try:
    import rasterio
    from rasterio.transform import from_bounds
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENVI_DIR = os.path.join(BASE_DIR, "data", "processed", "envi_outputs")
os.makedirs(ENVI_DIR, exist_ok=True)

BBOX       = [-73.5, -51.5, -72.5, -50.5]
GRID_SHAPE = (120, 120)

DARK_BG = "#0d1117"
DARK_AX = "#161b22"
C_TEXT  = "#e6edf3"
C_GREY  = "#8b949e"
C_GOLD  = "#f39c12"
C_RED   = "#e74c3c"
C_BLUE  = "#3498db"
C_GREEN = "#2ecc71"

# Sentinel-2 L2A typical DN scale (0-10000 = 0-100% reflectance)
S2_SCALE = 10000.0

# Band names and central wavelengths (nm) for axis labels
BAND_LABELS = {
    0: "B2\n490nm", 1: "B3\n560nm", 2: "B4\n665nm",
    3: "B8\n842nm", 4: "B11\n1610nm", 5: "B12\n2190nm",
}


# ---------------------------------------------------------------------------
# SYNTHETIC SENTINEL-2 IMAGE (before correction)
# ---------------------------------------------------------------------------

def generate_toa_image(shape=GRID_SHAPE) -> np.ndarray:
    """
    Generate a realistic Top-Of-Atmosphere DN image with atmospheric haze.
    Haze is simulated by adding a spatially uniform additive offset
    (path radiance) to each band -- exactly what DOS1 removes.

    Returns array of shape (6, rows, cols), DN range 0-10000.
    """
    rows, cols = shape
    rng = np.random.default_rng(seed=42)
    lons = np.linspace(BBOX[0], BBOX[2], cols)
    lats = np.linspace(BBOX[3], BBOX[1], rows)
    LON, LAT = np.meshgrid(lons, lats)

    veg_frac = np.clip(0.1 + 0.7 * (LON - BBOX[0]) / (BBOX[2] - BBOX[0]), 0, 1)
    water    = np.zeros(shape)
    water[rows//3:rows//2, cols//3:cols//2] = 1.0
    snow     = np.zeros(shape)
    snow[:rows//5, :cols//4] = 1.0

    def surface_band(B_soil, B_veg, B_water, B_snow, noise=200):
        return np.clip(
            B_soil * (1 - veg_frac - water - snow).clip(0, 1) +
            B_veg * veg_frac + B_water * water + B_snow * snow +
            rng.normal(0, noise, shape),
            0, 10000
        ).astype(np.float32)

    # Surface reflectance (ground truth)
    surface = np.stack([
        surface_band(800,  400,  200,  9000, 150),   # B2 Blue
        surface_band(1000, 600,  300,  9200, 150),   # B3 Green
        surface_band(1200, 500,  200,  8800, 200),   # B4 Red
        surface_band(2500, 7000, 300,  9500, 300),   # B8 NIR
        surface_band(2000, 3000, 400,  4000, 250),   # B11 SWIR1
        surface_band(1800, 2200, 350,  3500, 250),   # B12 SWIR2
    ], axis=0)

    # Path radiance haze: stronger in short wavelengths (Rayleigh scattering)
    # Typical values for low-altitude clear day over Patagonia
    path_radiance = np.array([600, 450, 300, 150, 80, 60], dtype=np.float32)

    toa = np.clip(surface + path_radiance[:, None, None], 0, 10000).astype(np.float32)
    print(f"  Generated TOA image: {toa.shape} (bands, rows, cols)")
    print(f"  Added path radiance: {path_radiance.tolist()} DN per band")
    return toa, path_radiance


# ---------------------------------------------------------------------------
# DOS1 ATMOSPHERIC CORRECTION
# ---------------------------------------------------------------------------

def dos1_correction(toa: np.ndarray) -> tuple:
    """
    Dark Object Subtraction (DOS1) atmospheric correction.

    Algorithm:
      1. For each band, find the minimum non-zero DN value
         (assumes this corresponds to a completely shadowed area with
          zero surface reflectance -> DN = path radiance only)
      2. Subtract this value from all pixels
      3. Divide by S2_SCALE to convert to 0-1 surface reflectance

    Pros: Simple, no metadata required, works on any Sentinel-2 scene.
    Cons: Overestimates correction if the darkest pixel has non-zero reflectance.
    """
    n_bands = toa.shape[0]
    corrected = np.zeros_like(toa, dtype=np.float32)
    dark_objects = []

    for b in range(n_bands):
        band_dn = toa[b].ravel()
        # Dark object: minimum DN > 0 (exclude nodata=0)
        valid = band_dn[band_dn > 0]
        do_dn = float(np.percentile(valid, 0.1)) if len(valid) > 0 else 0.0
        dark_objects.append(do_dn)
        corrected[b] = np.clip((toa[b] - do_dn) / S2_SCALE, 0, 1)

    print(f"  DOS1 Dark Objects (path radiance estimate per band):")
    for b, do in enumerate(dark_objects):
        print(f"    {BAND_LABELS.get(b, f'B{b}').split(chr(10))[0]:<6}: "
              f"DO = {do:6.1f} DN  -> subtract before scaling to reflectance")

    return corrected, np.array(dark_objects)


def print_correction_stats(toa: np.ndarray, corrected: np.ndarray) -> None:
    """Print before/after band statistics."""
    print(f"\n  {'Band':<6} {'TOA mean DN':>12} {'SR mean':>10} {'Improvement':>12}")
    print("  " + "-" * 46)
    for b in range(toa.shape[0]):
        toa_mean = float(toa[b].mean())
        sr_mean  = float(corrected[b].mean())
        label    = BAND_LABELS.get(b, f"B{b}").split("\n")[0]
        print(f"  {label:<6} {toa_mean:>12.0f} {sr_mean:>10.4f}")


# ---------------------------------------------------------------------------
# SAVE OUTPUT
# ---------------------------------------------------------------------------

def save_corrected_tif(corrected: np.ndarray, path: str) -> None:
    """Save corrected multi-band image as GeoTIFF."""
    if not HAS_RASTERIO:
        return
    rows, cols = corrected.shape[1], corrected.shape[2]
    t = from_bounds(*BBOX, cols, rows)
    with rasterio.open(path, "w", driver="GTiff",
                       height=rows, width=cols,
                       count=corrected.shape[0], dtype="float32",
                       crs="EPSG:4326", transform=t,
                       nodata=-9999.0, compress="lzw") as dst:
        for b in range(corrected.shape[0]):
            dst.write(corrected[b], b + 1)
        dst.update_tags(
            correction="DOS1 (Dark Object Subtraction)",
            source="Sentinel-2 L2A simulated -- GeoCascade Ch14",
            envi_equiv="ENVI: Toolbox > Radiometric Correction > QUAC/FLAASH",
            arcgis_note="Load in ArcGIS Pro as multiband raster"
        )
    print(f"  [OK] Saved: {os.path.relpath(path, BASE_DIR)}")


# ---------------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------------

def plot_correction_report(toa, corrected, dark_objects, out_path: str) -> None:
    """3-panel before/after correction figure."""
    fig = plt.figure(figsize=(20, 14), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32,
                            top=0.92, bottom=0.08, left=0.06, right=0.97)
    fig.text(0.5, 0.97,
             "GeoCascade Ch14 -- ENVI Atmospheric Correction (DOS1)",
             ha="center", color=C_TEXT, fontsize=13, fontweight="bold")
    fig.text(0.5, 0.945,
             "Before vs After DOS1  |  Sentinel-2 L2A  |  Torres del Paine",
             ha="center", color=C_GREY, fontsize=9)

    def style(ax, title):
        ax.set_facecolor(DARK_AX)
        for sp in ax.spines.values(): sp.set_color("#30363d")
        ax.tick_params(colors=C_TEXT, labelsize=7)
        ax.set_title(title, color=C_TEXT, fontsize=9, fontweight="bold", pad=5)

    ext = [BBOX[0], BBOX[2], BBOX[1], BBOX[3]]

    # Row 1: RGB before vs after
    # Before: bands B4, B3, B2 scaled to 0-1
    rgb_before = np.stack([toa[2], toa[1], toa[0]], axis=-1) / S2_SCALE
    rgb_after  = np.stack([corrected[2], corrected[1], corrected[0]], axis=-1)
    # Stretch for display
    for rgb in [rgb_before, rgb_after]:
        p2, p98 = np.percentile(rgb, [2, 98])
        rgb[:] = np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0, 1)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(rgb_before, origin="upper", extent=ext)
    style(ax1, "TOA -- Before Correction (RGB)\nHaze visible in blue channel")
    ax1.set_xlabel("Longitude", color=C_TEXT, fontsize=8)
    ax1.set_ylabel("Latitude",  color=C_TEXT, fontsize=8)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(rgb_after, origin="upper", extent=ext)
    style(ax2, "Surface Reflectance -- After DOS1\nPath radiance removed")
    ax2.set_xlabel("Longitude", color=C_TEXT, fontsize=8)

    # NDVI before vs after
    def ndvi(arr):
        num = arr[3] - arr[2]
        den = arr[3] + arr[2]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den != 0, num / den, np.nan).clip(-1, 1)

    ndvi_b = ndvi(toa / S2_SCALE)
    ndvi_a = ndvi(corrected)

    ax3 = fig.add_subplot(gs[0, 2])
    im3  = ax3.imshow(ndvi_a - ndvi_b, cmap="RdYlGn", vmin=-0.1, vmax=0.1,
                      origin="upper", extent=ext)
    cb3  = plt.colorbar(im3, ax=ax3, fraction=0.04, pad=0.02)
    cb3.set_label("NDVI change", color=C_TEXT, fontsize=7)
    cb3.ax.tick_params(colors=C_TEXT)
    style(ax3, "NDVI Improvement (after - before DOS1)")
    ax3.set_xlabel("Longitude", color=C_TEXT, fontsize=8)

    # Row 2: Band histograms before vs after (show all 6 bands)
    n_bands = toa.shape[0]
    ax4 = fig.add_subplot(gs[1, :])
    ax4.set_facecolor(DARK_AX)
    for sp in ax4.spines.values(): sp.set_color("#30363d")

    colors = ["#1565C0", "#2E7D32", "#C62828", "#6A1B9A", "#E65100", "#4E342E"]
    x = np.arange(n_bands)
    w = 0.35
    toa_means = [float(toa[b].mean()) for b in range(n_bands)]
    cor_means = [float(corrected[b].mean() * S2_SCALE) for b in range(n_bands)]
    bars1 = ax4.bar(x - w/2, toa_means, w, color=colors, alpha=0.9,
                    label="TOA DN (before)", edgecolor="#30363d")
    bars2 = ax4.bar(x + w/2, cor_means,  w, color=colors, alpha=0.5,
                    label="Corrected x10000 (after)", edgecolor="#30363d", hatch="//")
    ax4.bar(x - w/2, dark_objects, w, bottom=0,
            color=C_RED, alpha=0.6, label="Path Radiance (DO subtracted)")
    ax4.set_xticks(x)
    ax4.set_xticklabels([BAND_LABELS.get(i, f"B{i}").replace("\n", " ")
                          for i in range(n_bands)],
                         color=C_TEXT, fontsize=9)
    ax4.set_ylabel("Mean DN", color=C_TEXT, fontsize=10)
    ax4.tick_params(colors=C_TEXT)
    ax4.legend(fontsize=9, facecolor=DARK_BG, labelcolor=C_TEXT)
    ax4.set_title("Band-wise Correction: TOA vs Surface Reflectance (scaled x10000)",
                  color=C_TEXT, fontsize=10, fontweight="bold", pad=5)
    ax4.grid(True, color="#30363d", alpha=0.4, axis="y")

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  [OK] Report: {os.path.relpath(out_path, BASE_DIR)}")


# ---------------------------------------------------------------------------
# FLAASH WORKFLOW DESCRIPTION
# ---------------------------------------------------------------------------

def print_flaash_guide() -> None:
    """Print step-by-step FLAASH workflow for ENVI users with license."""
    print("""
  ─────────────────────────────────────────────────────────────
  ENVI FLAASH Workflow (requires Atmospheric Correction Module)
  ─────────────────────────────────────────────────────────────
  1. Open ENVI -> Toolbox > Radiometric Correction >
     Atmospheric Correction Module > FLAASH Atmospheric Correction

  2. Input Radiance Image:
     - Must be in radiance units (mW / (cm² sr nm))
     - For Sentinel-2: use ENVI's Radiometric Calibration first
       Toolbox > Radiometric Correction > Radiometric Calibration
       Set Output Interleave: BIL, Scale Factor: 1.0

  3. FLAASH Parameters (Torres del Paine):
     Sensor Type        : Sentinel-2
     Flight Date        : YYYY-MM-DD of the image
     Flight Time UTC    : HH:MM:SS
     Scene Center Lat   : -51.00
     Scene Center Lon   : -72.90
     Ground Elevation   : 0.500 km  (Patagonian steppe)
     Atmospheric Model  : Sub-Arctic Summer (closest to Patagonia)
     Aerosol Model      : Maritime (Pacific coast influence)
     Initial Visibility : 40 km
     Water Retrieval    : Yes (check box)

  4. Multispectral Settings:
     Filter function file: use Sentinel-2 SRF file from ESA
     (download from: https://sentinels.copernicus.eu/web/sentinel)

  5. Output: saves surface reflectance TIF (0-10000 scale)
     Load result in ArcGIS Pro as multiband raster.
  ─────────────────────────────────────────────────────────────
""")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print(" GEOCASCADE Ch14 -- ENVI Atmospheric Correction")
    print(" DOS1 (open-source) | FLAASH guide (requires license)")
    print("=" * 65)

    print("\n[1/4] Generating/loading TOA Sentinel-2 image ...")
    toa, true_path_rad = generate_toa_image()

    print("\n[2/4] Applying DOS1 atmospheric correction ...")
    corrected, dark_objects = dos1_correction(toa)
    print_correction_stats(toa, corrected)

    print("\n[3/4] Saving corrected image ...")
    save_corrected_tif(
        corrected,
        os.path.join(ENVI_DIR, "sentinel2_dos1_corrected.tif")
    )

    print("\n[4/4] Generating correction report figure ...")
    plot_correction_report(
        toa, corrected, dark_objects,
        os.path.join(ENVI_DIR, "atm_correction_report.png")
    )

    print_flaash_guide()

    print("=" * 65)
    print(" ATMOSPHERIC CORRECTION COMPLETE")
    print("=" * 65)
    print(f"  Corrected TIF : {ENVI_DIR}\\sentinel2_dos1_corrected.tif")
    print(f"  Report figure : {ENVI_DIR}\\atm_correction_report.png")
    print()
    print("  Note: DOS1 is suitable for educational purposes.")
    print("        Use ENVI FLAASH for publication-quality analysis.")
    print()
    print("  Continue with: python Chapter_14/09_envi_classification.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
