"""
Chapter 6: 16_isohyets_isotherms.py
=====================================
Isohyets (Equal Precipitation) & Isotherms (Equal Temperature) Mapping

Academic Objective:
  Meteorological variables are continuous surfaces. We visualize them with ISOLINES:
  lines connecting points of equal value — the same technique used in every
  weather map, topographic map, and climate atlas.

  This script generates two types of isolines:

  1. ISOTHERMS -- lines of equal temperature
     Method: Apply the Environmental Lapse Rate (-6.5 deg C / 1000 m) to the
     Copernicus DEM to estimate air temperature at each elevation.
     Also uses Ch01 RF temperature surface if available.
     Base temperature: 4.0 deg C at sea level (Torres del Paine annual mean)

  2. ISOHYETS -- lines of equal precipitation
     Method: Interpolate point precipitation data from Ch01 CHIRPS raster
     or ERA5 to a continuous surface, then contour it.
     Patagonia's steep precipitation gradient (>3000 mm/yr west of Andes
     vs <300 mm/yr east) makes this a critical analysis.

  Contour extraction: matplotlib cs.contour() outputs collections of
  LineString vertices. These are converted to a GeoDataFrame and exported
  as a GeoPackage (single file, no column name length limits).

  Critical: isotherms are OPEN LineStrings, not closed polygons.
  Use path.to_polygons(closed_only=False) to prevent spurious closure.

Connection to pipeline:
  Reads Ch01 RF temperature surface + CHIRPS if available.
  Falls back to lapse-rate model + synthetic precipitation for demo.

Outputs:
  data/processed/isolines/isotherms.gpkg
  data/processed/isolines/isohyets.gpkg
  data/processed/isolines/temperature_surface.tif
  data/processed/isolines/precipitation_surface.tif
  data/processed/isolines/isolines_map.png    (4-panel dark)
  data/processed/isolines/isoline_statistics.csv

ArcGIS Pro: Add isotherms.gpkg as Feature Layer.
            Symbology > Graduated Colors on Temperature field.
            Use Contour tool (Spatial Analyst > Surface > Contour) to validate.
ENVI 5.6:   File > Open > temperature_surface.tif.
            Contour Lines tool from Topographic menu.

Run:
  conda activate geocascade_env
  python Chapter_06/16_isohyets_isotherms.py

Dependencies: rasterio, numpy, matplotlib, geopandas, shapely, pandas, scipy, pyproj, pystac-client, planetary-computer
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import rasterio
from rasterio.windows import from_bounds
from rasterio.transform import from_bounds as transform_from_bounds
import geopandas as gpd
from shapely.geometry import LineString

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
OUT_DIR  = os.path.join(BASE_DIR, "data", "processed", "isolines")
os.makedirs(OUT_DIR, exist_ok=True)

BBOX = [-73.30, -51.10, -72.90, -50.80]

# Torres del Paine climate parameters
BASE_TEMP_C       = 4.0      # Annual mean temperature at sea level (~51 deg S)
LAPSE_RATE_C_M    = 6.5e-3   # Environmental lapse rate: -6.5 deg C / 1000 m
PRECIP_WEST_MM    = 3000.0   # Annual precipitation west of Andes
PRECIP_EAST_MM    = 300.0    # Annual precipitation east of Andes (rain shadow)
PRECIP_GRADIENT_DEG = 0.3    # Precipitation halves every 0.3 deg lon east of ridge

DARK_BG = "#0d1117"
DARK_AX = "#161b22"
C_TEXT  = "#e6edf3"
C_GREY  = "#8b949e"
C_BLUE  = "#3498db"
C_RED   = "#e74c3c"


# ---------------------------------------------------------------------------
# 1. Load or build temperature surface
# ---------------------------------------------------------------------------
def get_temperature_surface():
    """
    Priority:
      1. Ch01 RF temperature surface (most accurate)
      2. Lapse-rate model from Ch03 DEM (good approximation)
      3. Lapse-rate model from STAC DEM (fallback)
    Returns: temp_arr (2D float32), lons (1D), lats (1D), crs_str, transform
    """
    ch01_temp = os.path.join(ROOT_DIR, "Chapter_01", "data", "processed",
                             "climate_analysis", "temperature_surface.tif")
    ch03_dem  = os.path.join(ROOT_DIR, "Chapter_03", "data", "processed",
                             "terrain", "copernicus_dem.tif")

    if os.path.exists(ch01_temp):
        print("  [OK] Using Ch01 RF temperature surface.")
        with rasterio.open(ch01_temp) as src:
            arr = src.read(1).astype("float32")
            nd  = src.nodata
            if nd is not None:
                arr = np.where(arr == nd, np.nan, arr)
            b = src.bounds
            h, w = arr.shape
            lons = np.linspace(b.left,  b.right,  w)
            lats = np.linspace(b.bottom, b.top,   h)
            return arr, lons, lats, str(src.crs), src.transform

    if os.path.exists(ch03_dem):
        print("  [OK] Using Ch03 DEM for lapse-rate temperature model.")
        with rasterio.open(ch03_dem) as src:
            dem = src.read(1).astype("float32")
            nd  = src.nodata
            if nd is not None:
                dem = np.where(dem == nd, np.nan, dem)
            b = src.bounds
            h, w = dem.shape
            lons = np.linspace(b.left,  b.right,  w)
            lats = np.linspace(b.bottom, b.top,   h)
            crs_str  = str(src.crs)
            tf       = src.transform
        temp = BASE_TEMP_C - np.where(np.isfinite(dem), dem, 0) * LAPSE_RATE_C_M
        temp = np.where(np.isfinite(dem), temp, np.nan)
        return temp, lons, lats, crs_str, tf

    # STAC fallback
    print("  Downloading Copernicus DEM for lapse-rate model...")
    dem, lons, lats, crs_str, tf = _fetch_dem_stac()
    temp = BASE_TEMP_C - np.where(np.isfinite(dem), dem, 0) * LAPSE_RATE_C_M
    temp = np.where(np.isfinite(dem), temp, np.nan)
    return temp, lons, lats, crs_str, tf


def _fetch_dem_stac():
    from pystac_client import Client
    import planetary_computer as pc
    from pyproj import Transformer

    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                          modifier=pc.sign_inplace)
    search  = catalog.search(collections=["cop-dem-glo-30"], bbox=BBOX)
    items   = list(search.items())
    if not items:
        raise ValueError("No DEM found.")

    with rasterio.open(items[0].assets["data"].href) as src:
        t = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        mnx, mny = t.transform(BBOX[0], BBOX[1])
        mxx, mxy = t.transform(BBOX[2], BBOX[3])
        win = from_bounds(mnx, mny, mxx, mxy, src.transform)
        dem = src.read(1, window=win).astype("float32")
        nd  = src.nodata
        if nd is not None:
            dem = np.where(dem == nd, np.nan, dem)
        dem = np.where(dem < -500, np.nan, dem)
        b   = src.bounds
        h, w = int(round(win.height)), int(round(win.width))
        lons = np.linspace(BBOX[0], BBOX[2], w)
        lats = np.linspace(BBOX[1], BBOX[3], h)
        tf   = rasterio.windows.transform(win, src.transform)
    return dem, lons, lats, "EPSG:4326", tf


# ---------------------------------------------------------------------------
# 2. Build precipitation surface (Patagonian rain shadow model)
# ---------------------------------------------------------------------------
def get_precipitation_surface(lons, lats):
    """
    Patagonia has one of Earth's steepest precipitation gradients:
    - West of Andes divide (~73.1 deg W): >3000 mm/yr (hyperhumid)
    - East of divide (rain shadow): <300 mm/yr

    Model: exponential decay eastward from the Andes ridge.
    Ch01 CHIRPS data is used if available.
    """
    ch01_chirps = os.path.join(ROOT_DIR, "Chapter_01", "data", "processed",
                               "chirps", "chirps_annual_mean.tif")
    if os.path.exists(ch01_chirps):
        print("  [OK] Using Ch01 CHIRPS annual precipitation.")
        with rasterio.open(ch01_chirps) as src:
            arr = src.read(1).astype("float32")
            nd  = src.nodata
            if nd is not None:
                arr = np.where(arr == nd, np.nan, arr)
            b = src.bounds
            h_chirps = arr.shape[0]
            w_chirps = arr.shape[1]
            # Interpolate to our grid
            from scipy.interpolate import RegularGridInterpolator
            chirps_lons = np.linspace(b.left, b.right, w_chirps)
            chirps_lats = np.linspace(b.bottom, b.top, h_chirps)
            lon_grid, lat_grid = np.meshgrid(lons, lats)
            interp = RegularGridInterpolator(
                (chirps_lats, chirps_lons), arr[::-1],
                method="linear", bounds_error=False, fill_value=np.nan
            )
            precip = interp(np.stack([lat_grid.ravel(), lon_grid.ravel()], axis=1))
            return precip.reshape(lat_grid.shape).astype("float32")

    print("  Using Patagonian rain shadow model (no Ch01 CHIRPS found).")
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    andes_lon = -73.1   # approximate Andes divide longitude
    dist_east = lon_grid - andes_lon   # positive = east of Andes

    # Exponential decay with distance from ridge
    precip = np.where(
        dist_east <= 0,
        PRECIP_WEST_MM,                                          # west: constant high
        PRECIP_WEST_MM * np.exp(-dist_east / PRECIP_GRADIENT_DEG)  # east: decay
    )
    # Add latitudinal gradient (more rain at higher latitudes in Patagonia)
    lat_adj = 1.0 + 0.5 * (lat_grid - BBOX[1]) / (BBOX[3] - BBOX[1])
    precip  = precip * lat_adj
    # Add small noise for realism
    np.random.seed(42)
    precip  = precip + np.random.normal(0, 50, precip.shape)
    return np.clip(precip, 50, 5000).astype("float32")


# ---------------------------------------------------------------------------
# 3. Extract contours as GeoDataFrame
# ---------------------------------------------------------------------------
def extract_isolines(surface, lons, lats, levels, crs_str="EPSG:4326"):
    """
    Extract contour lines from a 2D surface array.
    Returns GeoDataFrame with one LineString per contour segment.
    """
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    fig, ax = plt.subplots(figsize=(1, 1))
    # Fill NaN with interpolated values for contour computation only
    surf_filled = surface.copy()
    mask = ~np.isfinite(surf_filled)
    if mask.any():
        from scipy.ndimage import generic_filter
        filled = generic_filter(surf_filled, lambda x: np.nanmean(x) if np.isnan(x[len(x)//2]) else x[len(x)//2],
                                size=5, mode="nearest")
        surf_filled = np.where(mask, filled, surf_filled)

    cs = ax.contour(lon_grid, lat_grid, surf_filled, levels=levels)
    plt.close(fig)

    lines  = []
    values = []

    # matplotlib >= 3.8: cs.collections is deprecated; use cs.allsegs
    if hasattr(cs, "allsegs"):
        for level, segs in zip(cs.levels, cs.allsegs):
            for seg in segs:
                if len(seg) >= 2:
                    lines.append(LineString(seg))
                    values.append(float(level))
    elif hasattr(cs, "collections"):
        for level, col in zip(cs.levels, cs.collections):
            for path in col.get_paths():
                for v in path.to_polygons(closed_only=False):
                    if len(v) >= 2:
                        lines.append(LineString(v))
                        values.append(float(level))

    if not lines:
        print("  [WARN] No isoline segments extracted. Check surface range vs levels.")
        return gpd.GeoDataFrame({"value": [], "geometry": []}, crs=crs_str)

    return gpd.GeoDataFrame({"value": values, "geometry": lines}, crs=crs_str)


# ---------------------------------------------------------------------------
# 4. Save raster surface
# ---------------------------------------------------------------------------
def save_surface_tif(arr, name, lons, lats, description=""):
    out = os.path.join(OUT_DIR, f"{name}.tif")
    h, w = arr.shape
    tf   = transform_from_bounds(lons[0], lats[0], lons[-1], lats[-1], w, h)
    with rasterio.open(out, "w",
                       driver="GTiff", height=h, width=w, count=1,
                       dtype="float32", crs="EPSG:4326", transform=tf,
                       nodata=-9999, compress="lzw") as dst:
        dst.write(np.nan_to_num(arr, nan=-9999).astype("float32"), 1)
        if description:
            dst.update_tags(description=description, nodata="-9999")
    print(f"  [OK] {name}.tif")
    return out


# ---------------------------------------------------------------------------
# 5. Statistics CSV
# ---------------------------------------------------------------------------
def save_stats(temp, precip, temp_gdf, precip_gdf):
    rows = []
    for arr, name in [(temp, "Temperature (deg C)"), (precip, "Precipitation (mm/yr)")]:
        v = arr[np.isfinite(arr)]
        rows.append({
            "variable": name,
            "mean": round(float(v.mean()), 3),
            "std":  round(float(v.std()),  3),
            "min":  round(float(v.min()),  3),
            "max":  round(float(v.max()),  3),
        })
    rows.append({"variable": "Isotherm lines",  "mean": len(temp_gdf),   "std": None, "min": None, "max": None})
    rows.append({"variable": "Isohyet lines",   "mean": len(precip_gdf), "std": None, "min": None, "max": None})
    df = pd.DataFrame(rows)
    csv = os.path.join(OUT_DIR, "isoline_statistics.csv")
    df.to_csv(csv, index=False, encoding="utf-8")
    print(f"  [OK] Statistics CSV: {csv}")
    for _, r in df.iterrows():
        if r["std"] is not None:
            print(f"  {r['variable']:<25s}  mean={r['mean']:>8.2f}  std={r['std']:>7.2f}  "
                  f"min={r['min']:>7.2f}  max={r['max']:>7.2f}")
        else:
            print(f"  {r['variable']:<25s}  count={r['mean']:.0f}")


# ---------------------------------------------------------------------------
# 6. 4-panel dark figure
# ---------------------------------------------------------------------------
def plot_isolines(temp, precip, lons, lats, temp_gdf, precip_gdf):
    print("\n  Building 4-panel isoline figure...")

    lon_grid, lat_grid = np.meshgrid(lons, lats)
    fig = plt.figure(figsize=(22, 16), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25,
                            top=0.93, bottom=0.05, left=0.05, right=0.97)
    fig.text(0.5, 0.97, "Isotherms & Isohyets -- Torres del Paine, Patagonia",
             ha="center", color=C_TEXT, fontsize=13, fontweight="bold")

    def style_ax(ax, title):
        ax.set_facecolor(DARK_AX)
        for sp in ax.spines.values():
            sp.set_color("#30363d")
        ax.tick_params(colors=C_TEXT, labelsize=7)
        ax.set_title(title, color=C_TEXT, fontsize=10, fontweight="bold", pad=6)
        ax.set_xlabel("Longitude", color=C_TEXT, fontsize=8)
        ax.set_ylabel("Latitude",  color=C_TEXT, fontsize=8)

    # Panel 1: Temperature surface + isotherms
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.pcolormesh(lon_grid, lat_grid, temp, cmap="coolwarm",
                          vmin=np.nanmin(temp), vmax=np.nanmax(temp), shading="auto")
    if len(temp_gdf):
        temp_gdf.plot(ax=ax1, color="white", linewidth=0.8, alpha=0.7)
    cb1 = plt.colorbar(im1, ax=ax1, fraction=0.035)
    cb1.set_label("deg C", color=C_TEXT, fontsize=8)
    cb1.ax.tick_params(colors=C_TEXT, labelsize=7)
    style_ax(ax1, "Temperature Surface + Isotherms")

    # Panel 2: Precipitation surface + isohyets
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.pcolormesh(lon_grid, lat_grid, precip, cmap="Blues",
                          vmin=np.nanmin(precip), vmax=np.nanmax(precip), shading="auto")
    if len(precip_gdf):
        precip_gdf.plot(ax=ax2, color=C_RED, linewidth=0.9, alpha=0.8)
    cb2 = plt.colorbar(im2, ax=ax2, fraction=0.035)
    cb2.set_label("mm/yr", color=C_TEXT, fontsize=8)
    cb2.ax.tick_params(colors=C_TEXT, labelsize=7)
    style_ax(ax2, "Precipitation Surface + Isohyets (rain shadow)")

    # Panel 3: Isotherms only on dark background
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(DARK_AX)
    if len(temp_gdf):
        norm = plt.Normalize(vmin=temp_gdf["value"].min(), vmax=temp_gdf["value"].max())
        cmap = plt.cm.RdBu_r
        for _, row in temp_gdf.iterrows():
            if not row.geometry.is_empty:
                xs, ys = row.geometry.xy
                ax3.plot(xs, ys, color=cmap(norm(row["value"])), lw=1.0, alpha=0.8)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cb3 = plt.colorbar(sm, ax=ax3, fraction=0.035)
        cb3.set_label("deg C", color=C_TEXT, fontsize=8)
        cb3.ax.tick_params(colors=C_TEXT, labelsize=7)
    style_ax(ax3, f"Isotherms Only ({len(temp_gdf)} line segments)")
    ax3.set_xlim(BBOX[0], BBOX[2])
    ax3.set_ylim(BBOX[1], BBOX[3])

    # Panel 4: Temperature histogram + precipitation histogram
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(DARK_AX)
    for sp in ax4.spines.values():
        sp.set_color("#30363d")
    ax4.tick_params(colors=C_TEXT, labelsize=8)
    ax4.set_xlabel("deg C", color=C_TEXT, fontsize=9)
    ax4.set_ylabel("Pixel count", color=C_TEXT, fontsize=9)

    t_valid = temp[np.isfinite(temp)]
    if t_valid.size:
        ax4.hist(t_valid, bins=60, color=C_RED, alpha=0.75, label="Temperature (deg C)")
        ax4.axvline(t_valid.mean(), color="#ffffff", lw=1.5, linestyle="--",
                    label=f"Mean: {t_valid.mean():.1f} deg C")

    ax4_r = ax4.twiny()
    p_valid = precip[np.isfinite(precip)]
    if p_valid.size:
        ax4_r.hist(p_valid, bins=60, color=C_BLUE, alpha=0.5, label="Precip (mm/yr)")
        ax4_r.axvline(p_valid.mean(), color=C_BLUE, lw=1.5, linestyle="--",
                      label=f"Mean: {p_valid.mean():.0f} mm/yr")
        ax4_r.set_xlabel("mm/yr", color=C_BLUE, fontsize=8)
        ax4_r.tick_params(colors=C_BLUE, labelsize=7)

    ax4.legend(fontsize=7, facecolor=DARK_BG, labelcolor=C_TEXT, loc="upper left")
    ax4.grid(alpha=0.15, color="#30363d")
    ax4.set_title("Distributions: Temperature & Precipitation",
                  color=C_TEXT, fontsize=10, fontweight="bold", pad=6)

    out_png = os.path.join(OUT_DIR, "isolines_map.png")
    fig.savefig(out_png, dpi=180, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  [OK] 4-panel figure: {out_png}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print(" GEOCASCADE - ISOTHERMS & ISOHYETS")
    print(f" Lapse rate: -{LAPSE_RATE_C_M*1000:.1f} deg C/1000m  |  BBOX: {BBOX}")
    print("=" * 65)

    print("\n[1/5] Building temperature surface...")
    temp, lons, lats, crs_str, tf = get_temperature_surface()
    print(f"  Temperature range: {np.nanmin(temp):.1f} to {np.nanmax(temp):.1f} deg C")

    print("\n[2/5] Building precipitation surface...")
    precip = get_precipitation_surface(lons, lats)
    print(f"  Precipitation range: {np.nanmin(precip):.0f} to {np.nanmax(precip):.0f} mm/yr")

    print("\n[3/5] Extracting isolines...")
    # Isotherms every 1 deg C
    t_levels = np.arange(np.nanmin(temp) + 0.5, np.nanmax(temp), 1.0)
    temp_gdf = extract_isolines(temp, lons, lats, t_levels, crs_str)
    temp_gdf.to_file(os.path.join(OUT_DIR, "isotherms.gpkg"), driver="GPKG")
    print(f"  [OK] isotherms.gpkg  ({len(temp_gdf)} segments)")

    # Isohyets every 200 mm/yr
    p_levels  = np.arange(200, int(np.nanmax(precip)), 200)
    precip_gdf = extract_isolines(precip, lons, lats, p_levels, crs_str)
    precip_gdf.to_file(os.path.join(OUT_DIR, "isohyets.gpkg"), driver="GPKG")
    print(f"  [OK] isohyets.gpkg   ({len(precip_gdf)} segments)")

    print("\n[4/5] Saving raster surfaces and statistics...")
    save_surface_tif(temp,   "temperature_surface",   lons, lats,
                    f"Air temperature (deg C) at {BBOX}, lapse rate model")
    save_surface_tif(precip, "precipitation_surface", lons, lats,
                    f"Annual precipitation (mm/yr), Patagonian rain shadow model")
    save_stats(temp, precip, temp_gdf, precip_gdf)

    print("\n[5/5] Building 4-panel figure...")
    plot_isolines(temp, precip, lons, lats, temp_gdf, precip_gdf)

    print("\n" + "=" * 65)
    print(" ISOLINES COMPLETE")
    print("=" * 65)
    print(f"  TIFs     : {OUT_DIR}")
    print(f"  Vectors  : isotherms.gpkg, isohyets.gpkg")
    print(f"  Figure   : {os.path.join(OUT_DIR, 'isolines_map.png')}")
    print()
    print("  ArcGIS Pro: Add isotherms.gpkg as Feature Layer.")
    print("              Symbology > Graduated Colors on value field.")
    print("              Spatial Analyst > Surface > Contour to validate.")
    print("  ENVI 5.6  : Topographic > Contour Lines from temperature_surface.tif.")
    print("=" * 65)


if __name__ == "__main__":
    main()
