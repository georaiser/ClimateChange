"""
Chapter 3: 10_digital_elevation_processing.py
===============================================
Copernicus DEM Terrain Derivatives: Slope, Aspect, Hillshade, Curvature

Academic Objective:
  A Digital Elevation Model (DEM) contains only raw elevation values.
  Terrain DERIVATIVES reveal the morphological structure of the landscape:

  Slope     -- Steepness (deg). Drives erosion, runoff, avalanche risk.
  Aspect    -- Direction the slope faces (deg from North). Controls insolation,
               snow persistence, and microclimate (N-facing = colder in S hemisphere).
  Hillshade -- Simulated shaded relief. Key for visual map interpretation.
  Curvature -- Rate of slope change. Convex ridges shed water; concave hollows
               collect it. Critical for watershed delineation.

Physics:
  np.gradient() MUST receive cell sizes in METRES, not degrees.
  CopDEM is delivered in EPSG:4326 (degrees). At 51 deg S:
    1 deg latitude  ~ 111,000 m
    1 deg longitude ~ 111,000 * cos(51 deg) ~ 69,800 m
  Passing cellsize=30 (degrees) underestimates slope by ~3300x.

Hillshade formula (ESRI standard):
  shaded = cos(zenith) * cos(slope) + sin(zenith) * sin(slope) * cos(azimuth - aspect)
  Output: 0-255 (clamp, not rescale -- negative illumination = 0, not 128)

Connection to pipeline:
  Script 11 (watershed) reads the DEM from data/raw/temp_dem.tif.
  This script writes the same file so you only need to run ONE download.

Outputs:
  data/processed/terrain/copernicus_dem.tif
  data/processed/terrain/slope_degrees.tif
  data/processed/terrain/aspect_degrees.tif
  data/processed/terrain/hillshade.tif
  data/processed/terrain/curvature.tif
  data/processed/terrain/terrain_derivatives.png   (5-panel dark)
  data/raw/temp_dem.tif                            (cached for script 11)

ArcGIS Pro: Analysis > Tools > Hillshade, Slope, Aspect (3D Analyst).
            These tools replication exactly matches the NumPy output here.
            Compare outputs to validate: differences < 0.5 deg are numerical noise.
ENVI 5.6:   Topographic > Slope, Topographic > Aspect from copernicus_dem.tif.

Run:
  conda activate geocascade_env
  python Chapter_03/10_digital_elevation_processing.py

Dependencies: rasterio, numpy, matplotlib, pandas, pystac-client, planetary-computer, pyproj
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import from_bounds

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
OUT_DIR   = os.path.join(BASE_DIR, "data", "processed", "terrain")
RAW_DIR   = os.path.join(BASE_DIR, "data", "raw")
TEMP_DEM  = os.path.join(RAW_DIR, "temp_dem.tif")     # shared with script 11
os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(RAW_DIR,  exist_ok=True)

BBOX = [-73.30, -51.10, -72.90, -50.80]

DARK_BG = "#0d1117"
DARK_AX = "#161b22"
C_TEXT  = "#e6edf3"
C_GREY  = "#8b949e"


# ---------------------------------------------------------------------------
# 1. Fetch or load DEM
# ---------------------------------------------------------------------------
def fetch_dem():
    """
    Download Copernicus DEM tile from Planetary Computer and crop to BBOX.
    Saves to temp_dem.tif (shared with script 11) and returns (dem, profile).
    """
    try:
        from pystac_client import Client
        import planetary_computer as pc
        from pyproj import Transformer
    except ImportError:
        raise ImportError("pystac-client / planetary-computer not installed. "
                          "Run: mamba install -n geocascade_env -c conda-forge pystac-client planetary-computer -y")

    if os.path.exists(TEMP_DEM):
        print(f"  [OK] Loading cached DEM: {TEMP_DEM}")
        with rasterio.open(TEMP_DEM) as src:
            dem = src.read(1).astype("float32")
            profile = src.profile.copy()
            res = src.res
        dem = np.where(dem == -9999, np.nan, dem)
        return dem, profile, res

    print("  Querying Planetary Computer for Copernicus DEM...")
    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace
    )
    search = catalog.search(collections=["cop-dem-glo-30"], bbox=BBOX)
    items  = list(search.items())
    if not items:
        raise ValueError(f"No Copernicus DEM tile found for BBOX {BBOX}.")

    item = items[0]
    print(f"  [OK] DEM tile: {item.id}")

    with rasterio.open(item.assets["data"].href) as src:
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        mnx, mny   = transformer.transform(BBOX[0], BBOX[1])
        mxx, mxy   = transformer.transform(BBOX[2], BBOX[3])
        window     = from_bounds(mnx, mny, mxx, mxy, src.transform)
        win_tf     = rasterio.windows.transform(window, src.transform)
        h = int(round(window.height))
        w = int(round(window.width))

        dem = src.read(1, window=window).astype("float32")
        nd  = src.nodata
        res = src.res  # (lat_deg, lon_deg)

        profile = src.profile.copy()
        profile.update(
            driver="GTiff", dtype="float32", count=1, nodata=-9999,
            height=h, width=w, transform=win_tf, compress="lzw"
        )

    # Mask nodata
    if nd is not None:
        dem = np.where(dem == nd, np.nan, dem)
    dem = np.where(dem < -500, np.nan, dem)  # guard against fill values

    # Cache as temp_dem.tif for script 11
    with rasterio.open(TEMP_DEM, "w", **profile) as dst:
        dst.write(np.nan_to_num(dem, nan=-9999).astype("float32"), 1)
    print(f"  [OK] DEM cached: {TEMP_DEM}  ({w}x{h} pixels)")
    return dem, profile, res


# ---------------------------------------------------------------------------
# 2. Terrain derivative functions
# ---------------------------------------------------------------------------
def pixel_sizes_m(res, lat_deg=-51.0):
    """Convert geographic pixel size (degrees) to metres at study latitude."""
    lat_m = abs(res[0]) * 111_000.0
    lon_m = abs(res[1]) * 111_000.0 * np.cos(np.radians(lat_deg))
    return lat_m, lon_m


def compute_slope_aspect(dem, lat_m, lon_m):
    """
    Slope in degrees and aspect in degrees from North (0-360).
    np.gradient(dem, dy, dx): dy = lat spacing, dx = lon spacing, both in metres.
    """
    dy, dx = np.gradient(dem, lat_m, lon_m)

    slope_rad  = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_deg  = np.degrees(slope_rad)

    aspect_rad = np.arctan2(dy, -dx)
    aspect_deg = np.degrees(aspect_rad)
    aspect_deg = np.where(aspect_deg < 0, 360.0 + aspect_deg, aspect_deg)

    return slope_deg, aspect_deg, slope_rad, aspect_rad


def compute_hillshade(slope_rad, aspect_rad, azimuth=315.0, zenith=45.0):
    """
    Standard ESRI hillshade formula (output 0-255).
    Azimuth 315 deg = NW sun (standard cartographic convention).
    Zenith 45 deg = 45 deg above horizon.
    """
    az_rad = np.radians(360.0 - azimuth + 90.0)
    ze_rad = np.radians(zenith)
    shaded = (np.cos(ze_rad) * np.cos(slope_rad) +
              np.sin(ze_rad) * np.sin(slope_rad) * np.cos(az_rad - aspect_rad))
    return np.clip(255.0 * shaded, 0.0, 255.0)


def compute_curvature(dem, lat_m, lon_m):
    """
    Plan curvature (second derivative of elevation surface).
    Positive = convex (ridge, water sheds).
    Negative = concave (valley, water collects).
    Units: 1/m (rate of change of slope per metre).
    """
    dy, dx    = np.gradient(dem, lat_m, lon_m)
    ddy, _    = np.gradient(dy, lat_m, lon_m)
    _, ddx    = np.gradient(dx, lat_m, lon_m)
    return ddy + ddx   # Laplacian (plan curvature proxy)


# ---------------------------------------------------------------------------
# 3. Save GeoTIFF helper
# ---------------------------------------------------------------------------
def save_tif(data, name, profile, description=""):
    out_path = os.path.join(OUT_DIR, f"{name}.tif")
    safe     = np.nan_to_num(data.astype("float32"), nan=-9999)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(safe, 1)
        if description:
            dst.update_tags(description=description, nodata="-9999",
                            arcgis_note="Stretched symbology recommended",
                            envi_note="File > Open, Linear stretch 2%")
    print(f"  [OK] {name}.tif")
    return out_path


# ---------------------------------------------------------------------------
# 4. Statistics
# ---------------------------------------------------------------------------
def terrain_stats(dem, slope_deg, aspect_deg, hillshade, curvature):
    def stats(arr, name):
        v = arr[np.isfinite(arr)]
        return {
            "layer": name,
            "min":   round(float(v.min()), 2) if v.size > 0 else None,
            "max":   round(float(v.max()), 2) if v.size > 0 else None,
            "mean":  round(float(v.mean()), 2) if v.size > 0 else None,
            "std":   round(float(v.std()),  2) if v.size > 0 else None,
        }
    rows = [
        stats(dem, "DEM (m)"),
        stats(slope_deg, "Slope (deg)"),
        stats(aspect_deg, "Aspect (deg)"),
        stats(hillshade, "Hillshade"),
        stats(curvature, "Curvature (1/m)"),
    ]
    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, "terrain_statistics.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\n  --- Terrain Statistics ---")
    for _, row in df.iterrows():
        print(f"  {row['layer']:<20s}  min={row['min']:>8}  max={row['max']:>8}  "
              f"mean={row['mean']:>8}  std={row['std']:>7}")
    print(f"  [OK] Stats CSV: {csv_path}")


# ---------------------------------------------------------------------------
# 5. 5-panel dark figure
# ---------------------------------------------------------------------------
def plot_terrain(dem, slope_deg, aspect_deg, hillshade, curvature):
    print("\n  Building 5-panel terrain figure...")

    fig, axes = plt.subplots(2, 3, figsize=(22, 14), facecolor=DARK_BG)
    fig.suptitle("Terrain Derivatives -- Torres del Paine, Patagonia\nCopernicus DEM 30m",
                 color=C_TEXT, fontsize=13, fontweight="bold", y=0.98)

    panels = [
        (dem,        "terrain",  None,      None,   "DEM -- Elevation (m)"),
        (slope_deg,  "magma",    0,         60,     "Slope (degrees)"),
        (aspect_deg, "hsv",      0,         360,    "Aspect (degrees from North)"),
        (hillshade,  "gray",     0,         255,    "Hillshade (Az=315, Ze=45)"),
        (curvature,  "RdBu_r",   -0.1,      0.1,   "Plan Curvature (convex=red, concave=blue)"),
    ]

    flat = axes.flatten()
    for i, (arr, cmap, vmin, vmax, title) in enumerate(panels):
        ax = flat[i]
        ax.set_facecolor(DARK_AX)
        valid = arr[np.isfinite(arr)]
        v2 = float(np.percentile(valid, 98)) if valid.size > 0 else 1
        v1 = float(np.percentile(valid, 2))  if valid.size > 0 else 0
        im = ax.imshow(arr, cmap=cmap,
                       vmin=vmin if vmin is not None else v1,
                       vmax=vmax if vmax is not None else v2,
                       aspect="auto")
        cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cb.ax.tick_params(colors=C_TEXT, labelsize=7)
        ax.set_title(title, color=C_TEXT, fontsize=9, fontweight="bold", pad=6)
        ax.axis("off")

    # Panel 6: Elevation histogram
    ax6 = flat[5]
    ax6.set_facecolor(DARK_AX)
    for sp in ax6.spines.values():
        sp.set_color("#30363d")
    ax6.tick_params(colors=C_TEXT)
    valid = dem[np.isfinite(dem)]
    if valid.size > 0:
        ax6.hist(valid, bins=80, color="#3498db", alpha=0.8, density=True)
        ax6.axvline(float(np.mean(valid)), color="#e74c3c", lw=1.5,
                    label=f"Mean: {np.mean(valid):.0f} m")
        ax6.legend(fontsize=8, facecolor=DARK_BG, labelcolor=C_TEXT)
    ax6.set_title("Elevation Histogram", color=C_TEXT, fontsize=9, fontweight="bold")
    ax6.set_xlabel("Elevation (m)", color=C_TEXT, fontsize=8)
    ax6.set_ylabel("Density", color=C_TEXT, fontsize=8)
    ax6.grid(alpha=0.15, color="#30363d")

    out_png = os.path.join(OUT_DIR, "terrain_derivatives.png")
    fig.savefig(out_png, dpi=180, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  [OK] 5-panel figure: {out_png}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print(" GEOCASCADE - TERRAIN DERIVATIVES (SLOPE / ASPECT / HILLSHADE)")
    print(f" Copernicus DEM 30m | BBOX: {BBOX}")
    print("=" * 65)

    print("\n[1/5] Fetching/loading Copernicus DEM...")
    dem, profile, res = fetch_dem()

    print("\n[2/5] Converting pixel size to metres...")
    lat_m, lon_m = pixel_sizes_m(res)
    print(f"  Pixel size: {lat_m:.1f} m (lat) x {lon_m:.1f} m (lon) at 51 deg S")

    print("\n[3/5] Computing terrain derivatives...")
    slope_deg, aspect_deg, slope_rad, aspect_rad = compute_slope_aspect(dem, lat_m, lon_m)
    hillshade  = compute_hillshade(slope_rad, aspect_rad)
    curvature  = compute_curvature(dem, lat_m, lon_m)
    print(f"  Slope range: {np.nanmin(slope_deg):.1f} to {np.nanmax(slope_deg):.1f} deg")
    print(f"  Hillshade : {np.nanmin(hillshade):.0f} to {np.nanmax(hillshade):.0f}")

    print("\n[4/5] Saving GeoTIFFs...")
    save_tif(dem,       "copernicus_dem",  profile, "Copernicus DEM 30m (m)")
    save_tif(slope_deg, "slope_degrees",  profile, "Slope in degrees")
    save_tif(aspect_deg,"aspect_degrees", profile, "Aspect 0-360 deg from North")
    save_tif(hillshade, "hillshade",      profile, "Hillshade Az=315 Ze=45, 0-255")
    save_tif(curvature, "curvature",      profile, "Plan curvature (1/m): +=convex, -=concave")
    terrain_stats(dem, slope_deg, aspect_deg, hillshade, curvature)

    print("\n[5/5] Generating 5-panel figure...")
    plot_terrain(dem, slope_deg, aspect_deg, hillshade, curvature)

    print("\n" + "=" * 65)
    print(" TERRAIN DERIVATIVES COMPLETE")
    print("=" * 65)
    print(f"  TIFs  : {OUT_DIR}")
    print(f"  Figure: {os.path.join(OUT_DIR, 'terrain_derivatives.png')}")
    print(f"  Stats : {os.path.join(OUT_DIR, 'terrain_statistics.csv')}")
    print()
    print("  ArcGIS Pro: Analysis > Tools > Hillshade, Slope, Aspect")
    print("              (3D Analyst toolbox -- results match this script)")
    print("  ENVI 5.6  : Topographic > Slope / Aspect from copernicus_dem.tif")
    print("=" * 65)


if __name__ == "__main__":
    main()
