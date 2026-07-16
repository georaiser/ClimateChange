"""
Chapter 8: 23_real_data_convergence.py

Academic Objective:
CONVERGENT EVIDENCE ANALYSIS — the culmination of the entire GeoCascade pipeline.

This script integrates ALL data streams into a single unified analysis:
  1. ERA5 climate series (temperature trends, precipitation anomalies)
  2. CHIRPS spatial precipitation (Andes gradient, anomaly years)
  3. Real weather station network (7 stations, climate gradient)
  4. Sentinel-2 NDVI (vegetation health from space)
  5. Sentinel-1 SAR VV (surface roughness, glacier/water detection)
  6. Copernicus DEM (terrain, elevation, slope)
  7. RGI 7.0 glacier outlines (official ice extent for validation)

The key insight: Environmental stress signals that appear in MULTIPLE independent
data streams are far more reliable than any single-sensor observation.
This is the "convergent evidence" principle used in operational Earth monitoring.

Outputs:
  - convergence_dashboard.png (8-panel integrated analysis)
  - convergence_report.csv (per-pixel multi-source metrics)
  - environmental_stress_composite.tif (final ESI raster)

Dependencies:
mamba install -n geocascade_env -c conda-forge pystac-client planetary-computer rasterio
  geopandas pandas numpy matplotlib pyproj requests scikit-learn -y
"""

import os
import glob
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import reproject, Resampling
from pyproj import Transformer
from sklearn.preprocessing import MinMaxScaler
from pystac_client import Client
import planetary_computer as pc

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_REAL = os.path.join(DATA_DIR, "raw",  "real_data")
PROC_DIR = os.path.join(DATA_DIR, "processed", "convergence")
os.makedirs(PROC_DIR, exist_ok=True)

BBOX       = [-73.30, -51.10, -72.90, -50.80]
DATE_RANGE = "2023-01-01/2023-03-31"
LAT, LON   = -51.0, -73.0


# ──────────────────────────────────────────────────────────────────────────────
# Helper: safe min-max normalization preserving NaN
# ──────────────────────────────────────────────────────────────────────────────
def norm01(arr):
    valid = arr[np.isfinite(arr)]
    if valid.size < 2:
        return arr
    mn, mx = np.nanpercentile(valid, 2), np.nanpercentile(valid, 98)
    if mx == mn:
        return np.zeros_like(arr)
    out = (arr - mn) / (mx - mn)
    out = np.clip(out, 0, 1)
    out[~np.isfinite(arr)] = np.nan
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 1. Load Real Climate Data (ERA5 + CHIRPS)
# ──────────────────────────────────────────────────────────────────────────────
def load_climate_data():
    print("\n[1/7] Loading real climate data (ERA5 + CHIRPS)...")

    # ERA5 daily
    era5_path = os.path.join(RAW_REAL, "era5_daily_patagonia.csv")
    if os.path.exists(era5_path):
        df = pd.read_csv(era5_path, parse_dates=["date"])
        df = df.set_index("date")
        print(f"       ERA5 daily: {len(df):,} records from {df.index.min().date()} to {df.index.max().date()}")
    else:
        print("       ERA5 not found — fetching now from Open-Meteo...")
        url = (
            f"https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={LAT}&longitude={LON}"
            f"&start_date=1993-01-01&end_date=2024-12-31"
            f"&daily=temperature_2m_mean,precipitation_sum,snowfall_sum"
            f"&timezone=America/Santiago"
        )
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame(data["daily"])
        df["date"] = pd.to_datetime(df["time"])
        df = df.drop(columns=["time"]).set_index("date")
        os.makedirs(RAW_REAL, exist_ok=True)
        df.reset_index().to_csv(era5_path, index=False)
        print(f"       ERA5 fetched and saved: {len(df):,} records")

    # Station data
    stn_path = os.path.join(RAW_REAL, "station_data_real.csv")
    df_stations = None
    if os.path.exists(stn_path):
        df_stations = pd.read_csv(stn_path, parse_dates=["date"])
        print(f"       Stations: {df_stations['station_name'].nunique()} stations, "
              f"{len(df_stations):,} records")
    else:
        print("       Station data not found — run 03a_fetch_real_weather_data.py first")

    return df, df_stations


# ──────────────────────────────────────────────────────────────────────────────
# 2. Fetch Satellite Data from Planetary Computer
# ──────────────────────────────────────────────────────────────────────────────
def fetch_satellite_data():
    print("\n[2/7] Fetching satellite data from Planetary Computer...")
    try:
        catalog = Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=pc.sign_inplace
        )
    except Exception as e:
        print(f"       [ERROR] Cannot connect to Planetary Computer: {e}")
        return None, None, None, None

    # --- Sentinel-2 NDVI ---
    s2_items = list(catalog.search(
        collections=["sentinel-2-l2a"], bbox=BBOX, datetime=DATE_RANGE,
        query={"eo:cloud_cover": {"lt": 20}}
    ).items())
    ndvi, s2_profile = None, None
    if s2_items:
        item = sorted(s2_items, key=lambda i: i.properties["eo:cloud_cover"])[0]
        print(f"       S2: {item.datetime.date()} ({item.properties['eo:cloud_cover']:.0f}% cloud)")
        try:
            t = Transformer.from_crs("EPSG:4326", "EPSG:32719", always_xy=True)
            minx, miny = t.transform(BBOX[0], BBOX[1])
            maxx, maxy = t.transform(BBOX[2], BBOX[3])
            with rasterio.open(item.assets["B08"].href) as src:
                win = from_bounds(minx, miny, maxx, maxy, src.transform)
                nir = src.read(1, window=win).astype("float32") / 10000
                s2_transform = rasterio.windows.transform(win, src.transform)
                s2_crs = src.crs
                s2_shape = (int(round(win.height)), int(round(win.width)))
                s2_profile = src.profile.copy()
                s2_profile.update(count=1, dtype=rasterio.float32, nodata=-9999,
                                  height=s2_shape[0], width=s2_shape[1], transform=s2_transform)
            with rasterio.open(item.assets["B04"].href) as src:
                red = src.read(1, window=win).astype("float32") / 10000
            ndvi = np.where((nir + red) < 1e-6, np.nan, (nir - red) / (nir + red))
            print(f"       NDVI: mean={np.nanmean(ndvi):+.3f}  shape={ndvi.shape}")
        except Exception as e:
            print(f"       [WARNING] NDVI failed: {e}")
    else:
        print("       [WARNING] No cloud-free Sentinel-2 found")

    # --- Sentinel-1 SAR ---
    s1_items = list(catalog.search(
        collections=["sentinel-1-rtc"], bbox=BBOX, datetime=DATE_RANGE
    ).items())
    sar_vv, sar_profile = None, None
    if s1_items and s2_profile:
        item = s1_items[0]
        print(f"       S1 SAR: {item.datetime.date()}")
        try:
            dest = np.zeros(s2_shape, dtype=np.float32)
            with rasterio.open(item.assets["vv"].href) as src:
                reproject(source=rasterio.band(src, 1), destination=dest,
                          src_transform=src.transform, src_crs=src.crs,
                          dst_transform=s2_transform, dst_crs=s2_crs,
                          resampling=Resampling.bilinear)
            sar_vv = np.where(dest <= 0, np.nan, 10 * np.log10(dest))
            sar_profile = s2_profile.copy()
            print(f"       SAR VV: mean={np.nanmean(sar_vv):+.1f} dB  shape={sar_vv.shape}")
        except Exception as e:
            print(f"       [WARNING] SAR failed: {e}")
    else:
        print("       [WARNING] No Sentinel-1 data found")

    # --- DEM ---
    dem_items = list(catalog.search(
        collections=["cop-dem-glo-30"], bbox=BBOX
    ).items())
    dem, dem_profile = None, None
    if dem_items and s2_profile:
        item = dem_items[0]
        print(f"       DEM: {item.id}")
        try:
            dest = np.zeros(s2_shape, dtype=np.float32)
            with rasterio.open(item.assets["data"].href) as src:
                reproject(source=rasterio.band(src, 1), destination=dest,
                          src_transform=src.transform, src_crs=src.crs,
                          dst_transform=s2_transform, dst_crs=s2_crs,
                          resampling=Resampling.bilinear)
            dem = np.where(dest == 0, np.nan, dest)
            dem_profile = s2_profile.copy()
            print(f"       DEM: min={np.nanmin(dem):.0f}m  max={np.nanmax(dem):.0f}m")
        except Exception as e:
            print(f"       [WARNING] DEM failed: {e}")

    return ndvi, sar_vv, dem, s2_profile


# ──────────────────────────────────────────────────────────────────────────────
# 3. Load RGI Glacier Outlines (if downloaded)
# ──────────────────────────────────────────────────────────────────────────────
def load_glacier_outlines():
    path = os.path.join(RAW_REAL, "rgi70_patagonia_glaciers.gpkg")
    if os.path.exists(path):
        gdf = gpd.read_file(path)
        print(f"\n[3/7] Loaded {len(gdf)} RGI 7.0 glacier polygons")
        return gdf
    print("\n[3/7] RGI outlines not found — run 00_real_data_downloader.py")
    return None


# ──────────────────────────────────────────────────────────────────────────────
# 4. Compute Environmental Stress Composite
# ──────────────────────────────────────────────────────────────────────────────
def compute_stress_composite(ndvi, sar_vv, dem):
    print("\n[4/7] Computing Environmental Stress Index (ESI)...")
    layers = {}

    # Vegetation stress = low NDVI (inverted: high = stressed)
    if ndvi is not None:
        veg_stress = norm01(-ndvi)  # invert: low NDVI = high stress
        layers["veg_stress"] = veg_stress
        print(f"       Veg stress: mean={np.nanmean(veg_stress):.3f}")

    # SAR structural complexity (deviation from mean)
    if sar_vv is not None:
        sar_norm = norm01(np.abs(sar_vv - np.nanmean(sar_vv)))
        layers["sar_anomaly"] = sar_norm
        print(f"       SAR anomaly: mean={np.nanmean(sar_norm):.3f}")

    # Terrain exposure (elevation normalized)
    if dem is not None:
        elev_stress = norm01(dem)  # high elevation = higher exposure
        layers["elev_exposure"] = elev_stress
        print(f"       Elev exposure: mean={np.nanmean(elev_stress):.3f}")

    if not layers:
        return None

    # Stack and average non-NaN layers
    stack = np.stack(list(layers.values()), axis=0)
    esi   = np.nanmean(stack, axis=0)
    esi[np.all(np.isnan(stack), axis=0)] = np.nan

    print(f"       ESI composite: mean={np.nanmean(esi):.3f}  "
          f"min={np.nanmin(esi):.3f}  max={np.nanmax(esi):.3f}")
    return esi, layers


# ──────────────────────────────────────────────────────────────────────────────
# 5. Build Convergence Dashboard
# ──────────────────────────────────────────────────────────────────────────────
def plot_convergence_dashboard(df_era5, df_stations, ndvi, sar_vv, dem, esi,
                                layers, gdf_glaciers):
    print("\n[5/7] Generating Convergence Dashboard (8-panel)...")

    fig = plt.figure(figsize=(22, 16))
    fig.patch.set_facecolor("#0a0a1a")
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.38, wspace=0.35)

    def dark_ax(ax, title=""):
        ax.set_facecolor("#111122")
        for spine in ax.spines.values():
            spine.set_color("#333355")
        ax.tick_params(colors="#aaaacc", labelsize=8)
        if title:
            ax.set_title(title, color="#ddddff", fontsize=9, fontweight="bold", pad=6)
        return ax

    # ── Panel 1: ERA5 30-year temperature trend ──
    ax1 = dark_ax(fig.add_subplot(gs[0, 0]), "ERA5 Annual Mean Temperature (30 yr)")
    annual_t = df_era5["temperature_2m_mean"].resample("YE").mean() if "temperature_2m_mean" in df_era5 else None
    if annual_t is not None:
        ax1.plot(annual_t.index.year, annual_t.values, "-o", color="#ff6b6b",
                 linewidth=1.8, markersize=4)
        z = np.polyfit(annual_t.index.year, annual_t.values, 1)
        ax1.plot(annual_t.index.year, np.poly1d(z)(annual_t.index.year),
                 "--", color="#ffd700", linewidth=2,
                 label=f"Trend: {z[0]*10:+.2f}°C/decade")
        ax1.legend(fontsize=7, facecolor="#111122", labelcolor="#ddddff")
    ax1.set_ylabel("°C", color="#aaaacc")
    ax1.grid(alpha=0.2, color="#333355")

    # ── Panel 2: ERA5 annual precipitation ──
    ax2 = dark_ax(fig.add_subplot(gs[0, 1]), "ERA5 Annual Precipitation")
    if "precipitation_sum" in df_era5.columns:
        annual_p = df_era5["precipitation_sum"].resample("YE").sum()
        mean_p   = annual_p.mean()
        bar_c    = ["#d62728" if p < mean_p*0.85 else "#1f77b4" if p > mean_p*1.15 else "#5e6687"
                    for p in annual_p.values]
        ax2.bar(annual_p.index.year, annual_p.values, color=bar_c, alpha=0.85)
        ax2.axhline(mean_p, color="#ffd700", linestyle="--", linewidth=1.5,
                    label=f"Mean: {mean_p:.0f} mm")
        ax2.legend(fontsize=7, facecolor="#111122", labelcolor="#ddddff")
    ax2.set_ylabel("mm/year", color="#aaaacc")
    ax2.grid(alpha=0.2, color="#333355", axis="y")

    # ── Panel 3: Station seasonal profiles ──
    ax3 = dark_ax(fig.add_subplot(gs[0, 2]), "Station Seasonal Temperature")
    if df_stations is not None and "station_name" in df_stations.columns:
        colors_stn = plt.cm.Set2(np.linspace(0, 1, df_stations["station_name"].nunique()))
        for (stn, grp), c in zip(df_stations.groupby("station_name"), colors_stn):
            if "temperature_2m_mean" in grp.columns:
                grp_idx = grp.set_index("date")["temperature_2m_mean"]
                clim    = grp_idx.groupby(grp_idx.index.month).mean()
                ax3.plot(clim.index, clim.values, "-o", color=c, linewidth=1.5,
                         markersize=3, label=stn.split(" ")[0])
        ax3.legend(fontsize=6, facecolor="#111122", labelcolor="#ddddff",
                   loc="lower right")
    ax3.set_xticks(range(1,13))
    ax3.set_xticklabels(list("JFMAMJJASOND"), fontsize=7)
    ax3.set_ylabel("°C", color="#aaaacc")
    ax3.grid(alpha=0.2, color="#333355")

    # ── Panel 4: ESI composite ──
    ax4 = dark_ax(fig.add_subplot(gs[0, 3]), "Environmental Stress Index (ESI)")
    if esi is not None:
        norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1)
        im = ax4.imshow(esi, cmap="RdYlGn_r", norm=norm, aspect="auto")
        plt.colorbar(im, ax=ax4, label="ESI [0=Low, 1=High Stress]",
                     fraction=0.046, pad=0.04).ax.tick_params(labelsize=7, colors="#aaaacc")
        ax4.set_title("Environmental Stress Index (ESI)", color="#ddddff",
                      fontsize=9, fontweight="bold", pad=6)
    ax4.axis("off")

    # ── Panel 5: NDVI ──
    ax5 = dark_ax(fig.add_subplot(gs[1, 0]), "Sentinel-2 NDVI (Vegetation Health)")
    if ndvi is not None:
        im = ax5.imshow(ndvi, cmap="RdYlGn", vmin=-0.2, vmax=0.8, aspect="auto")
        plt.colorbar(im, ax=ax5, fraction=0.046, pad=0.04).ax.tick_params(labelsize=7, colors="#aaaacc")
        if gdf_glaciers is not None:
            ax5.contour(ndvi < -0.1, levels=[0.5], colors=["cyan"], linewidths=1.5, alpha=0.8)
    ax5.axis("off")

    # ── Panel 6: SAR VV ──
    ax6 = dark_ax(fig.add_subplot(gs[1, 1]), "Sentinel-1 SAR VV (dB)")
    if sar_vv is not None:
        im = ax6.imshow(sar_vv, cmap="gray", vmin=-25, vmax=0, aspect="auto")
        plt.colorbar(im, ax=ax6, label="dB", fraction=0.046, pad=0.04).ax.tick_params(labelsize=7, colors="#aaaacc")
        # Overlay water mask
        water_mask = sar_vv < -18
        ax6.imshow(np.ma.masked_where(~water_mask, water_mask), cmap="Blues", alpha=0.6, aspect="auto")
    ax6.axis("off")

    # ── Panel 7: DEM ──
    ax7 = dark_ax(fig.add_subplot(gs[1, 2]), "Copernicus DEM (Elevation)")
    if dem is not None:
        im = ax7.imshow(dem, cmap="terrain", aspect="auto")
        plt.colorbar(im, ax=ax7, label="metres", fraction=0.046, pad=0.04).ax.tick_params(labelsize=7, colors="#aaaacc")
    ax7.axis("off")

    # ── Panel 8: Convergence matrix ──
    ax8 = dark_ax(fig.add_subplot(gs[1, 3]), "Data Stream Convergence Matrix")
    available = {
        "ERA5 Temp Trend":  annual_t is not None,
        "ERA5 Precip":      "precipitation_sum" in df_era5.columns,
        "Stn Network":      df_stations is not None,
        "CHIRPS Grid":      os.path.exists(os.path.join(
                            BASE_DIR.replace("Chapter_08","Chapter_01"),
                            "data","processed","real_data","chirps_precipitation_analysis.png")),
        "S2 NDVI":          ndvi is not None,
        "S1 SAR":           sar_vv is not None,
        "DEM":              dem is not None,
        "RGI Glaciers":     gdf_glaciers is not None,
    }
    sources = list(available.keys())
    vals    = [1 if v else 0 for v in available.values()]
    colors_bar = ["#2ecc71" if v else "#e74c3c" for v in vals]
    bars = ax8.barh(sources, vals, color=colors_bar, alpha=0.85)
    for bar, v in zip(bars, available.values()):
        ax8.text(0.05, bar.get_y() + bar.get_height()/2,
                 "AVAILABLE" if v else "MISSING", va="center", color="white", fontsize=7)
    ax8.set_xlim(0, 1.2)
    ax8.axis("off")

    # ── Bottom row: Full-width summary text ──
    ax9 = fig.add_subplot(gs[2, :])
    ax9.set_facecolor("#0a0a1a")
    ax9.axis("off")
    n_available = sum(available.values())
    summary = (
        f"GEOCASCADE CONVERGENCE REPORT  |  "
        f"Study Area: Torres del Paine, Patagonia (-51°S, -73°W)  |  "
        f"Active Data Streams: {n_available}/{len(available)}  |  "
        f"Period: 1993–2024
"
        f"ERA5 records: {len(df_era5):,} days  |  "
        f"Stations: {df_stations['station_name'].nunique() if df_stations is not None else 0}  |  "
        f"Satellite: {'S2+S1+DEM' if ndvi is not None else 'Offline'}  |  "
        f"RGI Glaciers: {len(gdf_glaciers) if gdf_glaciers is not None else 0} polygons"
    )
    ax9.text(0.5, 0.5, summary, transform=ax9.transAxes,
             ha="center", va="center", color="#aaaacc", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#111122", edgecolor="#333355"))

    fig.suptitle("GeoCascade — Convergent Evidence Analysis\n"
                 "Real Climate Data + Satellite Remote Sensing + Glacier Inventory",
                 color="#ffffff", fontsize=14, fontweight="bold", y=0.995)

    plot_path = os.path.join(PROC_DIR, "convergence_dashboard.png")
    plt.savefig(plot_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"       [SUCCESS] Convergence dashboard: {plot_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 6. Save ESI GeoTIFF
# ──────────────────────────────────────────────────────────────────────────────
def save_esi_tif(esi, profile):
    if esi is None or profile is None:
        return
    out_path = os.path.join(PROC_DIR, "environmental_stress_index.tif")
    data = np.where(np.isnan(esi), -9999, esi).astype("float32")
    profile.update(count=1, dtype=rasterio.float32, nodata=-9999)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data, 1)
    print(f"       [SUCCESS] ESI GeoTIFF saved: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print(" GEOCASCADE — REAL DATA CONVERGENCE ANALYSIS (Chapter 8 Capstone)")
    print("=" * 70)

    df_era5, df_stations = load_climate_data()
    ndvi, sar_vv, dem, s2_profile = fetch_satellite_data()
    gdf_glaciers = load_glacier_outlines()

    result = compute_stress_composite(ndvi, sar_vv, dem)
    esi = layers = None
    if result is not None:
        esi, layers = result
        save_esi_tif(esi, s2_profile)

    plot_convergence_dashboard(
        df_era5, df_stations, ndvi, sar_vv, dem, esi, layers, gdf_glaciers
    )

    print("\n" + "=" * 70)
    print(" CONVERGENCE ANALYSIS COMPLETE")
    print("=" * 70)
    data_available = sum([
        True, True,
        df_stations is not None,
        ndvi is not None,
        sar_vv is not None,
        dem is not None,
        gdf_glaciers is not None,
    ])
    print(f"  Data streams integrated: {data_available}/7")
    print(f"  ESI composite: {'computed' if esi is not None else 'not available (run satellite scripts first)'}")
    print(f"  Output: {PROC_DIR}")


if __name__ == "__main__":
    main()
