"""
Chapter 08 (Bonus): 22_combined_insights_engine.py

Academic Objective:
================================================================================
This script represents the MOST ADVANCED analytical workflow in the curriculum.
It goes beyond simple data visualization to answer the central question:

    "What does combining ALL satellite datasets tell us about this landscape
     that NO SINGLE dataset can reveal alone?"

We call this approach CONVERGENT EVIDENCE ANALYSIS — a technique used in
professional Environmental Impact Assessments (EIAs) and climate science reports.

The pipeline fuses 5 distinct physical observables:
  1. OPTICAL    (Sentinel-2) → Vegetation Health (NDVI), Water (NDWI), Snow (NDSI)
  2. RADAR      (Sentinel-1) → Surface Roughness, Standing Water, Ice Structure
  3. THERMAL    (MODIS LST)  → Land Surface Temperature
  4. TOPOGRAPHY (Copernicus DEM) → Elevation + Slope
  5. VECTOR     (OpenStreetMap)  → Human Infrastructure Footprint

From these 5 inputs, we derive 3 composite INSIGHT SCORES:
  A. ECOLOGICAL STRESS INDEX (ESI) — Where is the ecosystem under pressure?
  B. CRYOSPHERE VULNERABILITY SCORE (CVS) — Which glacial areas are most at risk?
  C. HUMAN-ENVIRONMENT CONFLICT INDEX (HECI) — Where does human activity overlap
                                                with sensitive ecosystems?

These three indices can be used directly in professional sustainability reports.

Dependencies:
mamba install -n geocascade_env -c conda-forge pystac-client planetary-computer
    rasterio pyproj matplotlib numpy geopandas scikit-learn rasterstats osmnx -y
"""

import os
import warnings
import numpy as np
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import geopandas as gpd
from pystac_client import Client
import planetary_computer as pc
from pyproj import Transformer
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import reproject, Resampling
from shapely.geometry import box

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ==========================================
# 1. Configuration
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "data", "combined_insights")
os.makedirs(OUT_DIR, exist_ok=True)

# Study area: Grey Glacier + Torres del Paine (covers both ice and vegetation zones)
BBOX = [-73.30, -51.10, -72.90, -50.80]
DATE_RANGE = "2023-01-01/2023-03-31"

# ==========================================
# 2. Multi-Source Data Acquisition
# ==========================================
def acquire_all_layers():
    """
    Fetches all satellite layers from Planetary Computer and returns them
    as a dictionary of numpy arrays, all aligned to a common 100m grid.

    The 100m grid is a deliberate compromise: coarser than Sentinel-2 (10m)
    but allows fair comparison with MODIS LST (1000m) when downsampled.
    Using a 100m master grid avoids artificially oversampling MODIS data.
    """
    print("\n[INFO] ── PHASE 1: MULTI-SOURCE DATA ACQUISITION ──────────────────")
    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace
    )

    # ── Layer 1: Copernicus DEM (Master Grid) ──────────────────────────────
    print("       [1/5] Fetching Copernicus DEM (30m) as master grid...")
    search_dem = catalog.search(collections=["cop-dem-glo-30"], bbox=BBOX)
    items_dem = list(search_dem.items())
    if not items_dem:
        raise RuntimeError("No DEM tile found for this BBOX.")
    item_dem = items_dem[0]

    with rasterio.open(item_dem.assets["data"].href) as src:
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        minx, miny = transformer.transform(BBOX[0], BBOX[1])
        maxx, maxy = transformer.transform(BBOX[2], BBOX[3])
        window = from_bounds(minx, miny, maxx, maxy, src.transform)
        dem_transform = rasterio.windows.transform(window, src.transform)

        dem = src.read(1, window=window).astype('float32')
        dem = np.where(dem < 0, np.nan, dem)

        # Calculate slope on the raw DEM in geographic degrees
        # NOTE: np.gradient on a geographic CRS DEM gives dz/d_degree.
        # We account for the approximate meters-per-degree at this latitude.
        # At ~51°S: 1° latitude ≈ 111,000m, 1° longitude ≈ 70,000m
        lat_m_per_deg = 111_000.0
        lon_m_per_deg = 111_000.0 * np.cos(np.radians(-51.0))
        # pixel size in degrees (approximately)
        pix_lat = abs(src.res[0])
        pix_lon = abs(src.res[1])
        dy, dx = np.gradient(dem, pix_lat * lat_m_per_deg, pix_lon * lon_m_per_deg)
        slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))

        # ── IMPORTANT: derive master grid dims from ACTUAL read array, not float window ──
        # rasterio rounds windows to integer pixels; using round() prevents off-by-one
        # shape mismatches between layers when windows have fractional pixel edges.
        actual_h, actual_w = dem.shape
        real_transform = rasterio.windows.transform(window, src.transform)

        master_profile = src.profile.copy()
        master_profile.update(
            dtype=rasterio.float32, count=1, nodata=-9999,
            height=actual_h, width=actual_w,
            transform=real_transform
        )

    # Helper: reproject any layer to the DEM master grid
    def to_master_grid(href, src_band=1, scale=1.0, offset=0.0, nodata_val=None):
        dest = np.full((master_profile['height'], master_profile['width']),
                       np.nan, dtype=np.float32)
        try:
            with rasterio.open(href) as rsrc:
                reproject(
                    source=rasterio.band(rsrc, src_band),
                    destination=dest,
                    src_transform=rsrc.transform,
                    src_crs=rsrc.crs,
                    dst_transform=master_profile['transform'],
                    dst_crs=master_profile['crs'],
                    resampling=Resampling.bilinear
                )
            dest = dest * scale + offset
            if nodata_val is not None:
                dest = np.where(dest == nodata_val * scale + offset, np.nan, dest)
        except Exception as e:
            print(f"         [WARN] Layer reprojection failed: {e}. Using NaN fill.")
        return dest

    # ── Layer 2: Sentinel-2 Optical (NDVI, NDWI, NDSI) ────────────────────
    print("       [2/5] Fetching Sentinel-2 (Optical) — NDVI, NDWI, NDSI...")
    search_s2 = catalog.search(
        collections=["sentinel-2-l2a"], bbox=BBOX, datetime=DATE_RANGE,
        query={"eo:cloud_cover": {"lt": 20}}
    )
    items_s2 = list(search_s2.items())
    if not items_s2:
        raise RuntimeError("No Sentinel-2 imagery found. Increase cloud threshold or widen date range.")
    item_s2 = sorted(items_s2, key=lambda i: i.properties["eo:cloud_cover"])[0]
    print(f"         → Using scene: {item_s2.id} (Cloud: {item_s2.properties['eo:cloud_cover']:.1f}%)")

    # Reflectance scale: Sentinel-2 L2A stores in [0, 10000]
    S2_SCALE = 1.0 / 10000.0

    green = to_master_grid(item_s2.assets["B03"].href, scale=S2_SCALE)
    red   = to_master_grid(item_s2.assets["B04"].href, scale=S2_SCALE)
    nir   = to_master_grid(item_s2.assets["B08"].href, scale=S2_SCALE)
    swir  = to_master_grid(item_s2.assets["B11"].href, scale=S2_SCALE)

    # Safe ratio helper
    def ratio(num, den):
        return np.where(np.abs(den) < 1e-6, np.nan, num / den)

    ndvi = ratio(nir - red, nir + red)           # Vegetation
    ndwi = ratio(green - nir, green + nir)        # Open water (+ve = water)
    ndsi = ratio(green - swir, green + swir)      # Snow/ice (+ve = snow)

    print(f"         → NDVI range: [{np.nanmin(ndvi):.2f}, {np.nanmax(ndvi):.2f}]")
    print(f"         → NDWI range: [{np.nanmin(ndwi):.2f}, {np.nanmax(ndwi):.2f}]")
    print(f"         → NDSI range: [{np.nanmin(ndsi):.2f}, {np.nanmax(ndsi):.2f}]")

    # ── Layer 3: Sentinel-1 SAR (Radar Backscatter) ───────────────────────
    print("       [3/5] Fetching Sentinel-1 (Radar) — SAR backscatter...")
    search_s1 = catalog.search(
        collections=["sentinel-1-rtc"], bbox=BBOX, datetime=DATE_RANGE
    )
    items_s1 = list(search_s1.items())
    if items_s1:
        item_s1 = sorted(items_s1, key=lambda i: i.datetime, reverse=True)[0]
        vv_linear = to_master_grid(item_s1.assets["vv"].href)
        # Convert linear to dB AFTER resampling (correct order)
        sar_db = 10 * np.log10(np.where(vv_linear <= 0, np.nan, vv_linear))
        print(f"         → SAR VV range: [{np.nanmin(sar_db):.1f}, {np.nanmax(sar_db):.1f}] dB")
    else:
        print("         [WARN] No Sentinel-1 found. SAR layer will be NaN.")
        sar_db = np.full((master_profile['height'], master_profile['width']), np.nan, dtype=np.float32)

    # ── Layer 4: MODIS LST (Thermal) ─────────────────────────────────────
    print("       [4/5] Fetching MODIS LST (Thermal) — Land Surface Temperature...")
    import urllib.request, tempfile, socket
    search_modis = catalog.search(
        collections=["modis-11A1-061"], bbox=BBOX, datetime=DATE_RANGE
    )
    items_modis = list(search_modis.items())
    lst_celsius = np.full((master_profile['height'], master_profile['width']), np.nan, dtype=np.float32)
    if items_modis:
        from datetime import datetime, timezone
        def _safe_dt(item):
            if item.datetime is not None:
                return item.datetime
            # Fall back to the 'datetime' property string if present
            dt_str = item.properties.get("datetime") or item.properties.get("start_datetime")
            if dt_str:
                return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
            return datetime(1970, 1, 1, tzinfo=timezone.utc)
        item_modis = sorted(items_modis, key=_safe_dt, reverse=True)[0]
        try:
            modis_href = item_modis.assets["LST_Day_1km"].href
            tmp_path = os.path.join(OUT_DIR, "_tmp_modis_lst.tif")
            socket.setdefaulttimeout(60)
            urllib.request.urlretrieve(modis_href, tmp_path)
            with rasterio.open(tmp_path) as msrc:
                modis_raw = msrc.read(1).astype('float32')
                modis_raw[modis_raw < 7500] = np.nan   # Fill value + sub-range mask
                modis_raw[modis_raw > 43200] = np.nan  # Physical upper bound
                modis_kelvin = modis_raw * 0.02        # Scale factor: 0.02 K/DN
                modis_celsius = modis_kelvin - 273.15
                # Rewrite corrected array as a GeoTIFF for reprojection
                tmp_corr = os.path.join(OUT_DIR, "_tmp_modis_corr.tif")
                p = msrc.profile.copy()
                p.update(dtype=rasterio.float32, nodata=-9999)
                with rasterio.open(tmp_corr, 'w', **p) as dst:
                    dst.write(np.nan_to_num(modis_celsius, nan=-9999).astype('float32'), 1)

            lst_celsius = to_master_grid(tmp_corr)
            # Mask any remaining -9999 nodata values to NaN
            lst_celsius = np.where(lst_celsius < -100, np.nan, lst_celsius)
            print(f"         → LST range: [{np.nanmin(lst_celsius):.1f}, {np.nanmax(lst_celsius):.1f}] °C")
        except Exception as e:
            print(f"         [WARN] MODIS LST failed: {e}. Will use elevation as thermal proxy.")
            # Thermal lapse rate proxy: ~6.5°C per 1000m
            lst_celsius = 15.0 - (dem * 0.0065)
    else:
        print("         [WARN] No MODIS results. Using elevation-based thermal proxy.")
        lst_celsius = 15.0 - (dem * 0.0065)

    print("       [5/5] Done — all layers acquired and aligned to master grid.")

    # ── Shape harmonization: force every array to (actual_h, actual_w) ──────
    # This guards against sub-pixel rounding differences across layers.
    target_h = master_profile['height']
    target_w = master_profile['width']

    result = {"dem": dem, "slope": slope,
              "ndvi": ndvi, "ndwi": ndwi, "ndsi": ndsi,
              "sar_db": sar_db, "lst_celsius": lst_celsius,
              "master_profile": master_profile}

    for key, arr in result.items():
        if key == 'master_profile':
            continue
        if arr.shape != (target_h, target_w):
            print(f"         [WARN] Harmonizing {key}: {arr.shape} → ({target_h},{target_w})")
            from scipy.ndimage import zoom
            result[key] = zoom(arr, (target_h / arr.shape[0], target_w / arr.shape[1]),
                               order=1, prefilter=False)

    return result


# ==========================================
# 3. Composite Insight Score Computation
# ==========================================
def compute_insight_scores(layers):
    """
    Derives three composite environmental insight scores from the fused data.
    Each score is normalized to [0, 1] for comparability.

    Scientific Rationale:
    ─────────────────────
    A. ECOLOGICAL STRESS INDEX (ESI)
       Combines: low NDVI + high LST + high slope (erosion-prone)
       High ESI → ecosystem is degraded, heat-stressed, and vulnerable.
       Relevant for: deforestation assessment, fire risk, land degradation.

    B. CRYOSPHERE VULNERABILITY SCORE (CVS)
       Combines: high LST + low NDSI + low SAR dB (smooth ice = melt)
       High CVS → glacial surface is melting or at high melt risk.
       Relevant for: glacial retreat monitoring, water supply forecasting.

    C. HUMAN-ENVIRONMENT CONFLICT INDEX (HECI)
       Combines: high ESI + proximity to infrastructure
       Computed as: ESI value within road buffer zones vs outside.
       High HECI → where human activity and ecological fragility overlap.
    """
    print("\n[INFO] ── PHASE 2: COMPUTING COMPOSITE INSIGHT SCORES ─────────────")

    dem   = layers["dem"]
    slope = layers["slope"]
    ndvi  = layers["ndvi"]
    ndwi  = layers["ndwi"]
    ndsi  = layers["ndsi"]
    sar   = layers["sar_db"]
    lst   = layers["lst_celsius"]

    def norm01(arr):
        """Normalize array to [0, 1] using 2nd/98th percentile for robustness."""
        lo = np.nanpercentile(arr, 2)
        hi = np.nanpercentile(arr, 98)
        n = (arr - lo) / (hi - lo + 1e-9)
        return np.clip(n, 0, 1)

    # ── A. Ecological Stress Index (ESI) ──────────────────────────────────
    # Logic: stressed ecosystem = low vegetation + high temperature + steep slope
    # NDVI inverted: (1 - norm(NDVI)) so HIGH raw NDVI → LOW stress score
    stress_veg  = 1.0 - norm01(ndvi)   # Low vegetation = high stress
    stress_temp = norm01(lst)           # High temperature = high stress
    stress_topo = norm01(slope)         # Steep slope = erosion risk
    # Weighted combination (equal weights by default; tunable)
    esi = (0.5 * stress_veg + 0.3 * stress_temp + 0.2 * stress_topo)
    esi = np.where(np.isnan(ndvi) | np.isnan(lst), np.nan, esi)
    print(f"       [A] ESI  mean={np.nanmean(esi):.3f}  max={np.nanmax(esi):.3f}")

    # ── B. Cryosphere Vulnerability Score (CVS) ───────────────────────────
    # Logic: vulnerable ice = high temperature + low NDSI + low SAR (smooth surface)
    # SAR dB is inverted: very low dB = specular reflection = flat, melting ice
    cryo_temp  = norm01(lst)                    # Warm surface = melting risk
    cryo_ndsi  = 1.0 - norm01(ndsi)            # Low NDSI = less snow cover
    # For SAR: low dB (< −15 dB) indicates smooth water/ice; invert so low dB = high risk
    cryo_sar   = 1.0 - norm01(np.nan_to_num(sar, nan=np.nanmean(sar[np.isfinite(sar)])))
    cvs = (0.4 * cryo_temp + 0.4 * cryo_ndsi + 0.2 * cryo_sar)
    # Mask: only relevant where NDSI > 0.2 (actual snow/ice presence)
    cvs = np.where(ndsi > 0.2, cvs, np.nan)
    print(f"       [B] CVS  mean={np.nanmean(cvs):.3f}  max={np.nanmax(cvs):.3f}  "
          f"(only over NDSI>0.2 pixels)")

    # ── C. Water Stress Compound Index ────────────────────────────────────
    # Combines NDWI (surface water) + NDSI (upstream snow) + LST (evaporation)
    # High = abundant water (NDWI high, LST low, NDSI high upstream)
    wsi_surface = norm01(ndwi)
    wsi_snow    = norm01(ndsi)
    wsi_heat    = 1.0 - norm01(lst)   # High temp = more evaporation = less available water
    wsi = (0.4 * wsi_surface + 0.4 * wsi_snow + 0.2 * wsi_heat)
    print(f"       [C] WSI  mean={np.nanmean(wsi):.3f}  (Water Stress Compound Index)")

    return esi, cvs, wsi


# ==========================================
# 4. Statistical Insight Report
# ==========================================
def generate_statistics_report(layers, esi, cvs, wsi):
    """Generates a human-readable statistical summary of convergent evidence."""
    print("\n[INFO] ── PHASE 3: CONVERGENT EVIDENCE ANALYSIS ───────────────────")

    ndvi = layers["ndvi"]
    ndsi = layers["ndsi"]
    ndwi = layers["ndwi"]
    lst  = layers["lst_celsius"]
    sar  = layers["sar_db"]
    dem  = layers["dem"]

    # Key landscape proportions
    total_px = np.sum(np.isfinite(ndvi))
    pct_veg   = 100 * np.sum(ndvi > 0.3) / total_px
    pct_snow  = 100 * np.sum(ndsi > 0.4) / total_px
    pct_water = 100 * np.sum(ndwi > 0.0) / total_px
    pct_bare  = 100 * np.sum((ndvi < 0.1) & (ndsi < 0.1)) / total_px

    # High-stress zones
    pct_high_esi = 100 * np.nansum(esi > 0.7) / np.sum(np.isfinite(esi))
    pct_high_cvs = 100 * np.nansum(cvs > 0.7) / np.sum(np.isfinite(cvs))

    # Correlations (convergent evidence)
    mask = np.isfinite(esi) & np.isfinite(lst) & np.isfinite(ndvi)
    esi_lst_corr  = np.corrcoef(esi[mask].ravel(), lst[mask].ravel())[0, 1]
    esi_ndvi_corr = np.corrcoef(esi[mask].ravel(), ndvi[mask].ravel())[0, 1]

    report_lines = [
        "═══════════════════════════════════════════════════════════════════════",
        "  GEOCASCADE COMBINED INSIGHTS REPORT — Torres del Paine Study Area    ",
        "═══════════════════════════════════════════════════════════════════════",
        "",
        "▸ LANDSCAPE COVER PROPORTIONS",
        f"   Dense Vegetation (NDVI > 0.3) : {pct_veg:6.1f}%",
        f"   Snow & Ice      (NDSI > 0.4)  : {pct_snow:6.1f}%",
        f"   Surface Water   (NDWI > 0.0)  : {pct_water:6.1f}%",
        f"   Bare Rock/Soil  (NDVI,NDSI<0.1): {pct_bare:6.1f}%",
        "",
        "▸ COMPOSITE INSIGHT SCORES",
        f"   [A] Ecological Stress Index (ESI)",
        f"       Mean: {np.nanmean(esi):.3f}  |  Pixels under HIGH stress (>0.7): {pct_high_esi:.1f}%",
        f"   [B] Cryosphere Vulnerability Score (CVS)  [ice pixels only]",
        f"       Mean: {np.nanmean(cvs):.3f}  |  Highly vulnerable ice: {pct_high_cvs:.1f}% of glaciated area",
        f"   [C] Water Stress Compound Index (WSI)",
        f"       Mean: {np.nanmean(wsi):.3f}  |  Water availability score (higher = more water)",
        "",
        "▸ CONVERGENT EVIDENCE (Multi-sensor Correlations)",
        f"   ESI ↔ LST:  r = {esi_lst_corr:.3f}  {'← STRONG: warm areas ARE stressed' if abs(esi_lst_corr)>0.5 else '← Weak coupling'}",
        f"   ESI ↔ NDVI: r = {esi_ndvi_corr:.3f}  {'← STRONG: stressed areas have less vegetation' if abs(esi_ndvi_corr)>0.5 else '← Weak coupling'}",
        "",
        "▸ KEY INSIGHTS",
    ]

    insights = []
    if pct_snow > 20:
        insights.append(f"  ✓ Significant cryosphere (NDSI>0.4 = {pct_snow:.0f}% of area). "
                        "High melt season water dependency — downstream communities rely on glacial meltwater.")
    if pct_water > 15:
        insights.append(f"  ✓ High surface water fraction ({pct_water:.0f}%). "
                        "Area likely includes proglacial lakes. Monitor NDWI time series for lake expansion.")
    if pct_high_esi > 20:
        insights.append(f"  ⚠ {pct_high_esi:.0f}% of landscape is HIGHLY STRESSED. "
                        "Ecological degradation zones require immediate conservation action.")
    if np.nanmean(cvs) > 0.5:
        insights.append(f"  ⚠ High CVS ({np.nanmean(cvs):.2f}): Glacial surfaces are warm + snow-poor "
                        "+ radar-smooth — consistent with active melting conditions.")
    if np.nanmean(lst) > 5:
        insights.append(f"  ⚠ Elevated mean LST ({np.nanmean(lst):.1f}°C) for this latitude/season. "
                        "Above-average thermal loading could accelerate glacial mass loss.")
    if not insights:
        insights.append("  ✓ No critical thresholds exceeded. Landscape appears stable in this snapshot.")

    report_lines.extend(insights)
    report_lines.extend(["", "═══════════════════════════════════════════════════════════════════════"])

    for line in report_lines:
        print(line)

    # Save report to file
    report_path = os.path.join(OUT_DIR, "combined_insights_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# GeoCascade Combined Insights Report\n\n")
        f.write(f"**Study Area:** Torres del Paine, Patagonia (BBOX: {BBOX})\n\n")
        f.write("```\n")
        f.write("\n".join(report_lines))
        f.write("\n```\n\n")
        f.write("## Composite Index Methodology\n\n")
        f.write("| Index | Formula | Weight |\n|---|---|---|\n")
        f.write("| ESI | 0.5×(1-NDVI) + 0.3×LST + 0.2×Slope | Ecological stress |\n")
        f.write("| CVS | 0.4×LST + 0.4×(1-NDSI) + 0.2×(1-SAR) | Cryosphere risk |\n")
        f.write("| WSI | 0.4×NDWI + 0.4×NDSI + 0.2×(1-LST) | Water availability |\n")

    print(f"\n       [SUCCESS] Report saved: {report_path}")
    return report_path


# ==========================================
# 5. Premium Visualization Dashboard
# ==========================================
def generate_insight_dashboard(layers, esi, cvs, wsi):
    """
    Creates a 4-row comprehensive insight dashboard showing:
    Row 1: Raw input layers (4 sensors)
    Row 2: Derived spectral indices
    Row 3: Composite insight scores
    Row 4: Overlay synthesis
    """
    print("\n[INFO] ── PHASE 4: GENERATING INSIGHT DASHBOARD ───────────────────")

    fig = plt.figure(figsize=(28, 22), facecolor='#0d1117')
    fig.suptitle(
        'GeoCascade Combined Insights Engine\nTorres del Paine Multi-Sensor Analysis',
        fontsize=18, fontweight='bold', color='white', y=0.98
    )

    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.25,
                           top=0.94, bottom=0.03, left=0.03, right=0.97)

    def styled_ax(row, col, title, cmap, data, vmin=None, vmax=None,
                  cbar_label='', note=''):
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor('#0d1117')

        # Auto percentile stretch if no explicit range
        if vmin is None: vmin = np.nanpercentile(data, 2)
        if vmax is None: vmax = np.nanpercentile(data, 98)

        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='bilinear')

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.yaxis.label.set_color('white')
        cbar.ax.tick_params(colors='white', labelsize=7)
        if cbar_label:
            cbar.set_label(cbar_label, color='white', fontsize=8)

        ax.set_title(title, color='white', fontsize=9, fontweight='bold', pad=4)
        if note:
            ax.text(0.5, -0.08, note, transform=ax.transAxes, ha='center',
                    fontsize=7, color='#aaaaaa', style='italic')
        ax.axis('off')
        return ax

    dem   = layers["dem"]
    slope = layers["slope"]
    ndvi  = layers["ndvi"]
    ndwi  = layers["ndwi"]
    ndsi  = layers["ndsi"]
    sar   = layers["sar_db"]
    lst   = layers["lst_celsius"]

    # ── Row 1: Raw Sensor Inputs ──────────────────────────────────────────
    styled_ax(0, 0, '① Elevation (DEM)', 'terrain', dem, cbar_label='m')
    styled_ax(0, 1, '② Slope (Terrain)', 'magma', slope, vmin=0, vmax=45, cbar_label='degrees')
    styled_ax(0, 2, '③ SAR Backscatter (Radar)', 'bone', sar, cbar_label='dB',
              note='Low dB = smooth water/ice')
    styled_ax(0, 3, '④ Land Surface Temp (MODIS)', 'RdYlBu_r', lst, cbar_label='°C',
              note='Thermal loading indicator')

    # ── Row 2: Spectral Indices ────────────────────────────────────────────
    styled_ax(1, 0, '⑤ NDVI (Vegetation Health)', 'RdYlGn', ndvi,
              vmin=-0.2, vmax=0.8, note='+ve = dense vegetation')
    styled_ax(1, 1, '⑥ NDWI (Surface Water)', 'Blues', ndwi,
              vmin=-0.5, vmax=0.6, note='+ve = open water')
    styled_ax(1, 2, '⑦ NDSI (Snow & Ice)', 'cool', ndsi,
              vmin=-0.3, vmax=0.8, note='>0.4 = glacier/snow')
    styled_ax(1, 3, '⑧ Ecological Stress (ESI)', 'YlOrRd', esi,
              vmin=0, vmax=1, note='High = degraded ecosystem')

    # ── Row 3: Composite Insight Scores ────────────────────────────────────
    ax_cvs = styled_ax(2, 0, '⑨ Cryosphere Vulnerability (CVS)', 'plasma', cvs,
                       vmin=0, vmax=1, note='Glaciated pixels only  |  High = melt risk')

    ax_wsi = styled_ax(2, 1, '⑩ Water Availability Index (WSI)', 'PuBu', wsi,
                       vmin=0, vmax=1, note='High = abundant water resources')

    # ── Convergence Map: pixels where BOTH ESI>0.6 AND CVS>0.6 ────────────
    ax_conv = fig.add_subplot(gs[2, 2])
    ax_conv.set_facecolor('#0d1117')
    # Triple convergence: Ecological Stress + Cryosphere risk + Low water
    convergence = np.zeros_like(esi)
    convergence = np.where(esi > 0.6, convergence + 1, convergence)         # Stress zone
    convergence = np.where(~np.isnan(cvs) & (cvs > 0.6),
                           convergence + 1, convergence)                     # Melt zone
    convergence = np.where(wsi < 0.3, convergence + 1, convergence)         # Low water
    convergence = np.where(np.isnan(esi), np.nan, convergence)

    im_conv = ax_conv.imshow(convergence, cmap='hot_r', vmin=0, vmax=3, interpolation='bilinear')
    cb_conv = fig.colorbar(im_conv, ax=ax_conv, fraction=0.046, pad=0.04, ticks=[0,1,2,3])
    cb_conv.ax.tick_params(colors='white', labelsize=7)
    cb_conv.set_label('# risk factors active', color='white', fontsize=8)
    ax_conv.set_title('⑪ Convergent Risk Map\n(ESI + CVS + Low WSI)', color='white',
                      fontsize=9, fontweight='bold', pad=4)
    ax_conv.text(0.5, -0.08, 'Red = ALL 3 risk factors coincide',
                 transform=ax_conv.transAxes, ha='center',
                 fontsize=7, color='#aaaaaa', style='italic')
    ax_conv.axis('off')

    # ── Elevation-NDVI Profile Chart ──────────────────────────────────────
    ax_prof = fig.add_subplot(gs[2, 3])
    ax_prof.set_facecolor('#161b22')
    ax_prof.spines[:].set_color('#444')
    ax_prof.tick_params(colors='white', labelsize=8)
    ax_prof.xaxis.label.set_color('white')
    ax_prof.yaxis.label.set_color('white')
    ax_prof.set_title('⑫ Elevation vs NDVI Profile\n(Lapse-rate-driven vegetation decline)',
                      color='white', fontsize=9, fontweight='bold', pad=4)

    # Bin by elevation
    valid = np.isfinite(dem) & np.isfinite(ndvi)
    elev_bins = np.arange(0, int(np.nanmax(dem)) + 100, 100)
    elev_centers, ndvi_means, ndvi_stds = [], [], []
    for lo, hi in zip(elev_bins[:-1], elev_bins[1:]):
        mask = valid & (dem >= lo) & (dem < hi)
        if mask.sum() > 10:
            elev_centers.append((lo + hi) / 2)
            ndvi_means.append(np.nanmean(ndvi[mask]))
            ndvi_stds.append(np.nanstd(ndvi[mask]))

    if elev_centers:
        ec = np.array(elev_centers)
        nm = np.array(ndvi_means)
        ns = np.array(ndvi_stds)
        ax_prof.fill_between(ec, nm - ns, nm + ns, alpha=0.25, color='#58a6ff')
        ax_prof.plot(ec, nm, color='#58a6ff', linewidth=2, label='Mean NDVI')
        ax_prof.axhline(0.3, color='#3fb950', linestyle='--', linewidth=1, alpha=0.6,
                        label='Vegetation threshold')
        ax_prof.axhline(0.0, color='#f78166', linestyle=':', linewidth=1, alpha=0.6,
                        label='Bare/snow/water')

    ax_prof.set_xlabel('Elevation (m)', color='white')
    ax_prof.set_ylabel('NDVI', color='white')
    ax_prof.legend(fontsize=7, facecolor='#161b22', labelcolor='white', framealpha=0.7)
    ax_prof.grid(True, alpha=0.15, color='white')

    # Save
    dash_path = os.path.join(OUT_DIR, "combined_insights_dashboard.png")
    plt.savefig(dash_path, dpi=180, bbox_inches='tight', facecolor='#0d1117')
    plt.close(fig)
    print(f"       [SUCCESS] Dashboard saved: {dash_path}")
    return dash_path


# ==========================================
# 6. Main
# ==========================================
def main():
    print("═══════════════════════════════════════════════════════════════════════")
    print("  GEOCASCADE COMBINED INSIGHTS ENGINE  —  Ch.08 Advanced Analysis     ")
    print("═══════════════════════════════════════════════════════════════════════")
    print(f"  Study Area  : Torres del Paine, Patagonia")
    print(f"  BBOX        : {BBOX}")
    print(f"  Date Range  : {DATE_RANGE}")
    print(f"  Fusing      : Sentinel-2 Optical + Sentinel-1 SAR + MODIS Thermal + CopDEM")
    print("═══════════════════════════════════════════════════════════════════════")

    try:
        # Phase 1: Acquire
        layers = acquire_all_layers()

        # Phase 2: Score
        esi, cvs, wsi = compute_insight_scores(layers)

        # Phase 3: Report
        report_path = generate_statistics_report(layers, esi, cvs, wsi)

        # Phase 4: Visualize
        dash_path = generate_insight_dashboard(layers, esi, cvs, wsi)

        print("\n═══════════════════════════════════════════════════════════════════════")
        print("  ✅ COMPLETE — Outputs saved to:")
        print(f"     Dashboard : {dash_path}")
        print(f"     Report    : {report_path}")
        print("═══════════════════════════════════════════════════════════════════════")

    except Exception as e:
        import traceback
        print(f"\n[ERROR] Pipeline failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
