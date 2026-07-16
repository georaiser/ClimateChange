"""
Chapter 2: 06_spectral_signature_analysis.py
=============================================
Spectral Signature Extraction & Analysis

Academic Objective:
  Every material on Earth absorbs and reflects electromagnetic radiation
  differently across wavelengths. This unique pattern is called a SPECTRAL
  SIGNATURE. By comparing signatures across bands, we can identify land cover
  types without setting foot in the field.

  Strategy: Cloud-Native + Local Fallback
  1. If Chapter 1 already downloaded Sentinel-2 L2A data -> use local TIFs
  2. Otherwise -> stream pixel data directly from Planetary Computer COGs
     (Cloud Optimized GeoTIFFs) without downloading the full scene (~600 MB)

  This is the most efficient approach: always try local data first.

Key Concept - Red Edge Band (B05, 705nm):
  Sentinel-2 has a unique Red Edge band that standard Landsat does not have.
  The red edge is the sharp reflectance transition from red (chlorophyll
  absorption) to NIR (leaf structure). It NARROWS under stress before NDVI
  changes. This makes B05 an early-warning indicator of vegetation stress.

Materials sampled (Torres del Paine):
  1. Glacial Ice (Grey Glacier)     -- high all bands, drops at SWIR
  2. Patagonian Lenga Beech Forest  -- strong NIR plateau, red edge rise
  3. Bare Rock / Scree              -- flat, moderate reflectance
  4. Open Water (Grey Lake)         -- very low NIR and SWIR

Outputs:
  data/processed/spectral_signatures.csv
  data/processed/spectral_signatures.png
  data/processed/spectral_red_edge_detail.png

ArcGIS Pro: Load spectral_signatures.csv as table. Insert > Chart > Line.
ENVI 5.6:   Open any band TIF, use ROI tool to sample pixels manually.
            Export ROI Statistics for comparison with this Python output.

Run:
  conda activate geocascade_env
  python Chapter_02/06_spectral_signature_analysis.py

Dependencies: rasterio, pyproj, matplotlib, numpy, pandas, pystac-client, planetary-computer
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyproj import Transformer

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
CH01_DIR  = os.path.join(os.path.dirname(BASE_DIR), "Chapter_01")
OUT_DIR   = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

# Sentinel-2 bands and their central wavelengths (nm)
BANDS = {
    "B02": 490,    # Blue         -- atmosphere, deep water
    "B03": 560,    # Green        -- vegetation peak, NDWI
    "B04": 665,    # Red          -- chlorophyll absorption
    "B05": 705,    # Red Edge     -- early stress indicator (Sentinel-2 unique)
    "B08": 842,    # NIR          -- leaf structure, NDVI
    "B11": 1610,   # SWIR1        -- soil moisture, NDMI, NDSI
    "B12": 2190,   # SWIR2        -- clay minerals, BSI
}

# Sample points (Lat, Lon) for each material in Torres del Paine
MATERIALS = {
    "Glacial Ice (Grey Glacier)": {
        "lat": -51.010, "lon": -73.230,
        "color": "#00bcd4", "marker": "o", "reflectance": []
    },
    "Lenga Beech Forest": {
        "lat": -51.150, "lon": -72.950,
        "color": "#2ecc71", "marker": "s", "reflectance": []
    },
    "Bare Rock / Scree": {
        "lat": -50.900, "lon": -72.900,
        "color": "#95a5a6", "marker": "^", "reflectance": []
    },
    "Open Water (Grey Lake)": {
        "lat": -51.050, "lon": -73.180,
        "color": "#3498db", "marker": "D", "reflectance": []
    },
}

DATE_RANGE = "2023-01-01/2023-02-28"
BBOX       = [-73.30, -51.20, -72.85, -50.85]

DARK_BG = "#0d1117"
DARK_AX = "#161b22"
C_TEXT  = "#e6edf3"
C_GREY  = "#8b949e"


# ---------------------------------------------------------------------------
# 1. Local data discovery (Ch01 downloads)
# ---------------------------------------------------------------------------
def find_local_s2():
    """Look for Sentinel-2 L2A bands already downloaded by Ch01 script 02."""
    import glob
    s2_dirs = glob.glob(os.path.join(CH01_DIR, "data", "raw", "sentinel2_l2a_*"))
    if not s2_dirs:
        return None
    # Use the most recent scene
    s2_dirs.sort()
    s2_dir = s2_dirs[-1]
    # Check required bands exist
    required = ["B02", "B03", "B04", "B08", "B11"]
    bands_found = {b: os.path.join(s2_dir, f"{b}.tif") for b in required
                   if os.path.exists(os.path.join(s2_dir, f"{b}.tif"))}
    if len(bands_found) >= len(required):
        print(f"  [OK] Local Sentinel-2 data found: {os.path.basename(s2_dir)}")
        return bands_found
    print(f"  [WARN] Incomplete local S2 data in {s2_dir}. Falling back to STAC.")
    return None


# ---------------------------------------------------------------------------
# 2. STAC API query (fallback if no local data)
# ---------------------------------------------------------------------------
def find_stac_image():
    """Query Planetary Computer STAC for best cloud-free scene."""
    try:
        from pystac_client import Client
        import planetary_computer as pc
    except ImportError:
        raise ImportError("pystac-client / planetary-computer not installed. "
                          "Run: mamba install -n geocascade_env -c conda-forge pystac-client planetary-computer -y")

    print("  Querying Planetary Computer STAC API...")
    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace
    )
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=BBOX,
        datetime=DATE_RANGE,
        query={"eo:cloud_cover": {"lt": 15}}
    )
    items = list(search.items())
    if not items:
        raise ValueError("No Sentinel-2 images found in date range. Widen cloud cover threshold.")

    best = sorted(items, key=lambda i: i.properties.get("eo:cloud_cover", 99))[0]
    print(f"  [OK] Best scene: {best.id} (cloud={best.properties.get('eo:cloud_cover',0):.1f}%)")
    # Return dict of asset hrefs keyed by band name
    return {b: best.assets[b].href for b in BANDS if b in best.assets}


# ---------------------------------------------------------------------------
# 3. Pixel extraction (works for both local files and COG URLs)
# ---------------------------------------------------------------------------
def extract_signatures(band_sources):
    """
    Sample one pixel per material per band.
    band_sources: dict of {band_name: file_path_or_url}
    """
    import rasterio

    transformer = None

    for band_name, wavelength in BANDS.items():
        if band_name not in band_sources:
            print(f"  [SKIP] Band {band_name} not available -- filling with NaN")
            for mat in MATERIALS.values():
                mat["reflectance"].append(np.nan)
            continue

        src_path = band_sources[band_name]
        print(f"  Band {band_name} ({wavelength} nm) <- {os.path.basename(str(src_path)[:60])}")

        with rasterio.open(src_path) as src:
            if transformer is None:
                epsg = src.crs.to_epsg()
                transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)

            for mat_name, mat_data in MATERIALS.items():
                x, y = transformer.transform(mat_data["lon"], mat_data["lat"])
                try:
                    val = next(src.sample([(x, y)]))[0]
                    # L2A scale: divide by 10000 to get reflectance [0, 1]
                    refl = float(val) / 10000.0
                    # Clamp to valid range
                    refl = np.clip(refl, 0.0, 1.5)
                except Exception:
                    refl = np.nan
                mat_data["reflectance"].append(refl)


# ---------------------------------------------------------------------------
# 4. Main spectral signature plot (dark mode)
# ---------------------------------------------------------------------------
def plot_signatures():
    print("\n  Building spectral signature plot...")

    wavelengths = list(BANDS.values())
    band_labels = [f"{w}nm\n({b})" for b, w in BANDS.items()]

    # Export CSV
    rows = {mat: data["reflectance"] for mat, data in MATERIALS.items()}
    df = pd.DataFrame(rows, index=[f"{b}_{w}nm" for b, w in BANDS.items()])
    csv_path = os.path.join(OUT_DIR, "spectral_signatures.csv")
    df.to_csv(csv_path, encoding="utf-8")
    print(f"  [OK] CSV: {csv_path}")

    fig, ax = plt.subplots(figsize=(14, 7), facecolor=DARK_BG)
    ax.set_facecolor(DARK_AX)
    for sp in ax.spines.values():
        sp.set_color("#30363d")
    ax.tick_params(colors=C_TEXT)

    for mat_name, mat_data in MATERIALS.items():
        refl = mat_data["reflectance"]
        valid = [r for r in refl if not np.isnan(r)]
        if not valid:
            continue
        ax.plot(wavelengths, refl, marker=mat_data["marker"], lw=2.5, ms=9,
                color=mat_data["color"], label=mat_name)

    # Shade Red Edge region (Sentinel-2 unique capability)
    ax.axvspan(700, 730, alpha=0.12, color="#e74c3c", label="Red Edge Region (700-730nm)")

    # Add band region annotations
    band_regions = [
        (430, 530, "Blue", "#3498db", 0.08),
        (520, 610, "Green", "#2ecc71", 0.10),
        (630, 690, "Red", "#e74c3c", 0.08),
        (690, 740, "RedEdge", "#e67e22", 0.15),
        (780, 900, "NIR", "#9b59b6", 0.10),
        (1550, 1680, "SWIR1", "#f39c12", 0.08),
        (2100, 2280, "SWIR2", "#e74c3c", 0.06),
    ]
    for x1, x2, label, c, ypos in band_regions:
        if x1 in range(400, 2300) and x2 in range(400, 2400):
            ax.text((x1 + x2) / 2, ypos, label, ha="center", fontsize=7,
                    color=c, alpha=0.7, style="italic")

    ax.set_title("Spectral Signatures of Earth Materials\nSentinel-2 L2A -- Torres del Paine, Patagonia",
                 color=C_TEXT, fontsize=13, fontweight="bold")
    ax.set_xlabel("Wavelength (nm)", color=C_TEXT, fontsize=11)
    ax.set_ylabel("Surface Reflectance [0-1]", color=C_TEXT, fontsize=11)
    ax.set_xticks(wavelengths)
    ax.set_xticklabels(band_labels, fontsize=8, color=C_TEXT)
    ax.set_ylim(-0.02, 0.85)
    ax.grid(True, ls="--", alpha=0.2, color="#30363d")
    ax.legend(fontsize=10, facecolor=DARK_BG, labelcolor=C_TEXT, framealpha=0.8,
              loc="upper right")

    out_png = os.path.join(OUT_DIR, "spectral_signatures.png")
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  [OK] Spectral signature plot: {out_png}")
    return out_png


# ---------------------------------------------------------------------------
# 5. Red Edge detail plot (academic focus)
# ---------------------------------------------------------------------------
def plot_red_edge_detail():
    """Zoom into the 650-900nm region to show the red edge transition."""
    print("  Building Red Edge detail plot...")

    # Bands in the 650-900nm region
    re_bands = {k: v for k, v in BANDS.items() if 600 <= v <= 900}
    wavelengths = list(re_bands.values())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=DARK_BG)

    for ax in axes:
        ax.set_facecolor("#161b22")
        for sp in ax.spines.values():
            sp.set_color("#30363d")
        ax.tick_params(colors=C_TEXT)

    # Left: all materials in red-edge region
    ax1 = axes[0]
    for mat_name, mat_data in MATERIALS.items():
        band_names = list(BANDS.keys())
        re_indices = [i for i, b in enumerate(band_names) if BANDS[b] in re_bands.values()]
        refl_re = [mat_data["reflectance"][i] for i in re_indices]
        ax1.plot(wavelengths, refl_re, marker="o", lw=2.5, ms=8,
                 color=mat_data["color"], label=mat_name)
    ax1.axvspan(700, 730, alpha=0.18, color="#e74c3c")
    ax1.text(715, ax1.get_ylim()[1] * 0.9 if ax1.get_ylim()[1] > 0 else 0.5,
             "Red Edge\n(Sentinel-2 unique)", ha="center", fontsize=8,
             color="#e74c3c")
    ax1.set_title("Red Edge Region (650-900nm)", color=C_TEXT, fontsize=11, fontweight="bold")
    ax1.set_xlabel("Wavelength (nm)", color=C_TEXT)
    ax1.set_ylabel("Reflectance", color=C_TEXT)
    ax1.legend(fontsize=9, facecolor=DARK_BG, labelcolor=C_TEXT)
    ax1.grid(alpha=0.2, color="#30363d")

    # Right: concept explanation
    ax2 = axes[1]
    ax2.axis("off")
    ax2.set_title("Why the Red Edge Matters", color=C_TEXT, fontsize=11, fontweight="bold", pad=10)
    lines = [
        "Standard sensors (Landsat, MODIS):",
        "  RED (665nm) + NIR (842nm) only",
        "  -> NDVI changes AFTER stress occurs",
        "",
        "Sentinel-2 adds Red Edge (B05, 705nm):",
        "  -> Detects stress BEFORE NDVI changes",
        "  -> Canopy narrowing at 700-730nm",
        "  -> Critical for early drought warning",
        "",
        "Physical mechanism:",
        "  Healthy leaf: high chlorophyll = strong",
        "  absorption at red (665nm), sharp rise",
        "  to NIR. Under stress: chlorophyll",
        "  degrades -> red edge SHIFTS blueward",
        "  and NARROWS -> detectable at 705nm",
        "  before NDVI responds.",
        "",
        "Unique to: Sentinel-2, Sentinel-3, MODIS",
        "Not in: Landsat 8/9, SPOT, ASTER",
    ]
    y = 0.97
    for line in lines:
        color = "#e74c3c" if "Sentinel-2" in line or "NDVI" in line else C_TEXT
        ax2.text(0.03, y, line, transform=ax2.transAxes, fontsize=8.5,
                 color=color, va="top",
                 fontweight="bold" if line.startswith("Standard") or line.startswith("Sentinel-2 adds") else "normal")
        y -= 0.052

    out_re = os.path.join(OUT_DIR, "spectral_red_edge_detail.png")
    fig.savefig(out_re, dpi=180, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  [OK] Red Edge detail: {out_re}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print(" GEOCASCADE - SPECTRAL SIGNATURE EXTRACTION")
    print(f" Study area: Torres del Paine, Patagonia")
    print("=" * 65)

    print("\n[1/4] Locating Sentinel-2 data...")
    local_bands = find_local_s2()

    if local_bands:
        band_sources = local_bands
        # Add B05 and B12 from STAC if not in local (they may not have been downloaded)
        missing = [b for b in BANDS if b not in band_sources]
        if missing:
            print(f"  [NOTE] Bands not in local data: {missing}")
            print("         Using local bands for available, STAC for missing.")
            try:
                stac_sources = find_stac_image()
                for b in missing:
                    if b in stac_sources:
                        band_sources[b] = stac_sources[b]
            except Exception as e:
                print(f"  [WARN] STAC fallback failed: {e}")
                print("         Proceeding with available local bands.")
    else:
        print("\n  No local Ch01 data found. Streaming from Planetary Computer...")
        try:
            band_sources = find_stac_image()
        except Exception as e:
            print(f"\n  ERROR: {e}")
            print("  Run Chapter_01/02_satellite_acquisition.py first to download S2 data.")
            return

    print(f"\n[2/4] Extracting spectral signatures ({len(MATERIALS)} materials x {len(BANDS)} bands)...")
    extract_signatures(band_sources)

    # Print raw values
    print("\n  Material reflectance values:")
    print(f"  {'Material':<35s}  " + "  ".join(f"{b:>5s}" for b in BANDS))
    print("  " + "-" * 80)
    for mat_name, mat_data in MATERIALS.items():
        vals = "  ".join(f"{r:5.3f}" if not np.isnan(r) else "  N/A" for r in mat_data["reflectance"])
        print(f"  {mat_name:<35s}  {vals}")

    print("\n[3/4] Generating spectral signature plots...")
    plot_signatures()
    plot_red_edge_detail()

    print("\n[4/4] Summary:")
    print("=" * 65)
    print(" SPECTRAL ANALYSIS COMPLETE")
    print("=" * 65)
    print(f"  CSV : {os.path.join(OUT_DIR, 'spectral_signatures.csv')}")
    print(f"  Plot: {os.path.join(OUT_DIR, 'spectral_signatures.png')}")
    print(f"  Red Edge: {os.path.join(OUT_DIR, 'spectral_red_edge_detail.png')}")
    print()
    print("  ArcGIS Pro: Insert > Add Data > Table > spectral_signatures.csv")
    print("              Right-click table > Charts > Line Chart")
    print("  ENVI 5.6  : Draw ROIs over glacier/forest/water pixels,")
    print("              Tools > Spectral Profile to compare with CSV output")
    print("=" * 65)


if __name__ == "__main__":
    main()
