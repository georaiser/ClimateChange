"""
05_climate_maps.py
==================
GeoCascade Chapter 14 -- ArcGIS Pro + ENVI Professional Workflows
Torres del Paine, Patagonia, Chile

PURPOSE
-------
Generates publication-quality climate maps from all Chapter 14 outputs.
Mirrors ArcGIS Pro Layout view workflow:
  - Map frame with graticule (lat/lon grid lines)
  - Legend, north arrow, scale bar
  - Inset overview map
  - Multi-panel climate atlas figure

OUTPUTS
-------
  data/processed/arcgis_outputs/climate_atlas.png      -- 6-panel atlas
  data/processed/arcgis_outputs/uhi_final_map.png      -- publication UHI map
  data/processed/arcgis_outputs/vulnerability_map.png  -- climate vulnerability

ARCGIS PRO EQUIVALENT
---------------------
  Insert > New Layout (A3 Landscape)
  Insert > Map Frame
  Insert > Legend / North Arrow / Scale Bar
  Share > Export Layout (PDF, PNG 300 DPI)

RUN
---
  python Chapter_14/05_climate_maps.py
"""

import sys
import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import matplotlib.patheffects as pe
from scipy.ndimage import uniform_filter, zoom

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.dirname(BASE_DIR)
PROC_DIR = os.path.join(BASE_DIR, "data", "processed", "arcgis_outputs")
os.makedirs(PROC_DIR, exist_ok=True)

BBOX = [-73.5, -51.5, -72.5, -50.5]

DARK_BG  = "#0d1117"
DARK_AX  = "#161b22"
MAP_BG   = "#1a2332"      # slightly lighter for map backgrounds
C_TEXT   = "#e6edf3"
C_GREY   = "#8b949e"
C_GOLD   = "#f39c12"
C_RED    = "#e74c3c"
C_BLUE   = "#3498db"
C_GREEN  = "#2ecc71"

# Place labels for Torres del Paine landmarks
PLACES = {
    "Torres del Paine": (-72.90, -51.00),
    "Lago Grey":        (-73.10, -51.10),
    "Balmaceda":        (-71.69, -45.91),
    "Punta Arenas":     (-70.90, -53.15),
}


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def load_or_synth(path: str, label: str, shape=(80, 100), dtype=np.float32):
    """Load GeoTIFF or return synthetic array."""
    if HAS_RASTERIO and os.path.exists(path):
        with rasterio.open(path) as src:
            d = src.read(1).astype(dtype)
            nd = src.nodata
            if nd is not None:
                d = np.where(d == nd, np.nan, d)
        print(f"  [OK] {label}: {d.shape}")
        return d
    else:
        rng = np.random.default_rng(seed=abs(hash(label)) % 9999)
        print(f"  [SYNTH] {label}")
        lons = np.linspace(BBOX[0], BBOX[2], shape[1])
        lats = np.linspace(BBOX[3], BBOX[1], shape[0])
        LON, LAT = np.meshgrid(lons, lats)
        base = rng.uniform(0, 1, shape).astype(dtype)
        # Add west-east gradient
        base += 0.4 * (LON - BBOX[0]) / (BBOX[2] - BBOX[0])
        return uniform_filter(base.clip(0, 1), size=5).astype(dtype)


def add_graticule(ax, step=0.2):
    """Add lat/lon grid lines (ArcGIS Pro graticule)."""
    gl_lons = np.arange(np.ceil(BBOX[0] / step) * step,
                        BBOX[2] + step, step)
    gl_lats = np.arange(np.ceil(BBOX[1] / step) * step,
                        BBOX[3] + step, step)
    for lon in gl_lons:
        ax.axvline(lon, color="#30363d", lw=0.5, ls="--", alpha=0.6)
    for lat in gl_lats:
        ax.axhline(lat, color="#30363d", lw=0.5, ls="--", alpha=0.6)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(step))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(step))
    ax.tick_params(colors=C_TEXT, labelsize=7)


def add_north_arrow(ax, x=0.95, y=0.95):
    """Add a north arrow (ArcGIS Pro style)."""
    ax.annotate("N", xy=(x, y), xycoords="axes fraction",
                 color=C_TEXT, fontsize=11, fontweight="bold", ha="center",
                 va="bottom")
    ax.annotate("", xy=(x, y), xytext=(x, y - 0.08),
                 xycoords="axes fraction",
                 arrowprops=dict(arrowstyle="-|>", color=C_GOLD, lw=2))


def add_scalebar(ax, lon_len=0.5, lat_pos=-51.45):
    """Add a simple scale bar."""
    km = lon_len * 111.32 * np.cos(np.radians(-51.0))
    ax.plot([BBOX[0] + 0.05, BBOX[0] + 0.05 + lon_len], [lat_pos, lat_pos],
            "w-", lw=3, solid_capstyle="butt")
    ax.text(BBOX[0] + 0.05 + lon_len / 2, lat_pos + 0.03,
            f"{km:.0f} km", ha="center", va="bottom",
            color=C_TEXT, fontsize=7, fontweight="bold")


def add_place_labels(ax, places=PLACES):
    """Add city/feature labels."""
    for name, (lon, lat) in places.items():
        if BBOX[0] <= lon <= BBOX[2] and BBOX[1] <= lat <= BBOX[3]:
            ax.plot(lon, lat, "o", color=C_GOLD, markersize=4, zorder=5)
            ax.text(lon + 0.03, lat, name, color=C_TEXT, fontsize=6,
                    va="center", zorder=6,
                    path_effects=[pe.withStroke(linewidth=1.5, foreground=DARK_BG)])


def styled_ax(ax, title=""):
    ax.set_facecolor(MAP_BG)
    for sp in ax.spines.values():
        sp.set_color("#30363d")
        sp.set_linewidth(1.2)
    ax.tick_params(colors=C_TEXT, labelsize=7)
    ax.set_xlim(BBOX[0], BBOX[2])
    ax.set_ylim(BBOX[1], BBOX[3])
    if title:
        ax.set_title(title, color=C_TEXT, fontsize=9, fontweight="bold", pad=5)


# ---------------------------------------------------------------------------
# MAP GENERATORS
# ---------------------------------------------------------------------------

def make_climate_atlas(layers: dict, out_path: str) -> None:
    """6-panel climate atlas (ArcGIS Pro A3 Layout equivalent)."""
    fig = plt.figure(figsize=(24, 16), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.25,
                            top=0.92, bottom=0.06, left=0.04, right=0.97)

    fig.text(0.5, 0.97,
             "GeoCascade -- Torres del Paine Climate Atlas",
             ha="center", color=C_TEXT, fontsize=16, fontweight="bold")
    fig.text(0.5, 0.945,
             "Chapter 14: ArcGIS Pro Professional Cartography  |  "
             "Study Area: 73.5W-72.5W / 51.5S-50.5S",
             ha="center", color=C_GREY, fontsize=10)

    panels = [
        ("precip",  "Blues",        "CHIRPS Mean Annual Precipitation (mm/yr)"),
        ("temp",    "RdBu_r",       "RF Temperature Surface (deg C)"),
        ("uhi",     "RdYlBu_r",     "MODIS UHI Celsius (Urban - Rural)"),
        ("slope",   "YlOrRd",       "Terrain Slope (degrees)"),
        ("vuln",    "RdYlGn_r",     "Climate Vulnerability Index (1-5)"),
        ("cls",     None,           "Land Cover Classification (K-Means)"),
    ]
    class_colors = ["#1565C0", "#E3F2FD", "#795548", "#FFC107", "#2E7D32"]
    cls_cmap = ListedColormap(class_colors)

    pos = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

    for (r, c), (key, cmap, title) in zip(pos, panels):
        ax = fig.add_subplot(gs[r, c])
        styled_ax(ax, title)

        data = layers.get(key)
        if data is None:
            ax.text(0.5, 0.5, "Data not found\nRun previous scripts",
                    ha="center", va="center", color=C_GREY, fontsize=9,
                    transform=ax.transAxes)
            continue

        ext = [BBOX[0], BBOX[2], BBOX[1], BBOX[3]]
        if key == "cls":
            im = ax.imshow(data, cmap=cls_cmap, vmin=-0.5, vmax=4.5,
                           origin="upper", extent=ext)
            legend_patches = [
                Patch(facecolor=class_colors[i],
                      label=["Water", "Snow", "Rock", "Sparse Veg", "Dense Veg"][i])
                for i in range(5)
            ]
            ax.legend(handles=legend_patches, fontsize=6, ncol=2,
                      facecolor=DARK_BG, labelcolor=C_TEXT, loc="lower left",
                      framealpha=0.7)
        else:
            vabs = max(abs(float(np.nanmin(data))), abs(float(np.nanmax(data))))
            if "uhi" in key or "temp" in key:
                im = ax.imshow(data, cmap=cmap, vmin=-vabs, vmax=vabs,
                               origin="upper", extent=ext)
            elif "vuln" in key:
                im = ax.imshow(data, cmap=cmap, vmin=1, vmax=5,
                               origin="upper", extent=ext)
            else:
                im = ax.imshow(data, cmap=cmap, origin="upper", extent=ext)
            cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
            cb.ax.tick_params(colors=C_TEXT, labelsize=6)

        add_graticule(ax, step=0.25)
        add_place_labels(ax)

        if r == 0 and c == 0:
            add_north_arrow(ax)
            add_scalebar(ax)

        ax.set_xlabel("Longitude", color=C_GREY, fontsize=7)
        ax.set_ylabel("Latitude",  color=C_GREY, fontsize=7)

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  [OK] Climate atlas: {os.path.relpath(out_path, BASE_DIR)}")


def make_uhi_publication_map(uhi_data: np.ndarray, out_path: str) -> None:
    """Single-panel publication-quality UHI map."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), facecolor=DARK_BG)
    styled_ax(ax, "")

    ext = [BBOX[0], BBOX[2], BBOX[1], BBOX[3]]
    vabs = max(1.0, float(np.nanmax(np.abs(uhi_data))))
    im = ax.imshow(uhi_data, cmap="RdYlBu_r", vmin=-vabs, vmax=vabs,
                   origin="upper", extent=ext, zorder=2)

    cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.03, shrink=0.7)
    cb.set_label("LST Anomaly (Urban - Rural, deg C)", color=C_TEXT, fontsize=10)
    cb.ax.tick_params(colors=C_TEXT, labelsize=9)

    add_graticule(ax, step=0.1)
    add_place_labels(ax)
    add_north_arrow(ax, x=0.96, y=0.96)
    add_scalebar(ax, lon_len=0.2)

    ax.set_title("Punta Arenas Urban Heat Island\nMODIS MOD11A1 Land Surface Temperature (2022)",
                 color=C_TEXT, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Longitude", color=C_TEXT, fontsize=10)
    ax.set_ylabel("Latitude", color=C_TEXT, fontsize=10)

    # Data source note
    fig.text(0.5, 0.02,
             "Source: MODIS MOD11A1-061 | Microsoft Planetary Computer STAC | "
             "GeoCascade Chapter 14",
             ha="center", color=C_GREY, fontsize=8)

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  [OK] UHI map: {os.path.relpath(out_path, BASE_DIR)}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print(" GEOCASCADE Ch14 -- Climate Maps (ArcGIS Pro Layout)")
    print(" Atlas | UHI Map | Vulnerability Map | Cartographic Standards")
    print("=" * 65)

    shape = (80, 100)

    print("\n[1/4] Loading all processed layers ...")
    layers = {
        "precip": load_or_synth(
            os.path.join(PROC_DIR, "precip_classified.tif"), "Precip Classified", shape),
        "temp": load_or_synth(
            os.path.join(PROC_DIR, "temp_anomaly.tif"), "Temperature Anomaly", shape),
        "uhi": load_or_synth(
            os.path.join(BASE_DIR, "..", "Chapter_01", "data", "processed",
                         "uhi_mapping", "uhi_celsius.tif"), "UHI Celsius", shape),
        "slope": load_or_synth(
            os.path.join(PROC_DIR, "slope.tif"), "Slope", shape),
        "vuln": load_or_synth(
            os.path.join(PROC_DIR, "climate_vulnerability.tif"),
            "Vulnerability Index", shape),
        "cls": load_or_synth(
            os.path.join(PROC_DIR, "classified_land_cover.tif"),
            "Land Cover Classification", shape, dtype=np.float32),
    }

    print("\n[2/4] Building 6-panel climate atlas ...")
    make_climate_atlas(layers, os.path.join(PROC_DIR, "climate_atlas.png"))

    print("\n[3/4] Building publication-quality UHI map ...")
    uhi_data = layers["uhi"]
    make_uhi_publication_map(uhi_data,
                             os.path.join(PROC_DIR, "uhi_final_map.png"))

    print("\n[4/4] Building vulnerability map ...")
    vuln = layers["vuln"]
    fig, ax = plt.subplots(figsize=(10, 9), facecolor=DARK_BG)
    styled_ax(ax)
    im = ax.imshow(vuln, cmap="RdYlGn_r", vmin=1, vmax=5,
                   origin="upper", extent=[BBOX[0], BBOX[2], BBOX[1], BBOX[3]])
    cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.03, shrink=0.7,
                      ticks=[1, 2, 3, 4, 5])
    cb.set_label("Vulnerability Index", color=C_TEXT, fontsize=10)
    cb.ax.set_yticklabels(["1 Very Low", "2 Low", "3 Medium", "4 High", "5 Very High"],
                          color=C_TEXT, fontsize=8)
    cb.ax.tick_params(colors=C_TEXT)
    add_graticule(ax, step=0.2)
    add_place_labels(ax)
    add_north_arrow(ax)
    add_scalebar(ax)
    ax.set_title(
        "Climate Vulnerability Index -- Torres del Paine\n"
        "Weighted Overlay: Temperature Trend + Precipitation Deficit + Slope",
        color=C_TEXT, fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Longitude", color=C_TEXT, fontsize=10)
    ax.set_ylabel("Latitude",  color=C_TEXT, fontsize=10)
    fig.text(0.5, 0.01,
             "Weights: Temp Trend 40% | Precip Anomaly 30% | Slope 20% | NDVI Loss 10%",
             ha="center", color=C_GREY, fontsize=8)
    fig.savefig(os.path.join(PROC_DIR, "vulnerability_map.png"),
                dpi=180, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  [OK] Vulnerability map saved.")

    print("\n" + "=" * 65)
    print(" CLIMATE MAPS COMPLETE")
    print("=" * 65)
    print(f"  Climate Atlas     : {PROC_DIR}\\climate_atlas.png")
    print(f"  UHI Final Map     : {PROC_DIR}\\uhi_final_map.png")
    print(f"  Vulnerability Map : {PROC_DIR}\\vulnerability_map.png")
    print()
    print("  ArcGIS Pro:")
    print("    Insert > New Layout (A3) > Map Frame > add atlas layers")
    print("    Insert > Legend (auto-generates from layer symbology)")
    print("    Share > Export Layout > PNG 300 DPI for publication")
    print()
    print("  ENVI 5.6:")
    print("    File > Chip View to Screen > Export as TIFF at 300 DPI")
    print()
    print("  Continue with: python Chapter_14/06_model_builder.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
