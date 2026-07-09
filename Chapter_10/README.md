# Chapter 10: Agentic Environmental Monitoring

## Overview

Implements an automated environmental change monitoring system using a
trigger-engine architecture. Six independent data streams are watched
simultaneously, and alerts fire when evidence of environmental stress
exceeds thresholds.

No LLM required -- pure threshold logic IS an effective monitoring agent.

## Scripts

| Script | Purpose |
|---|---|
| `25_agentic_monitor.py` | TriggerEngine + 6 triggers + monitoring dashboard |

## Trigger Inventory

| Trigger | Data Source | Threshold | Severity |
|---|---|---|---|
| `temp_anomaly` | ERA5 temperature | 90-day mean > baseline +2sigma | HIGH |
| `drought_stress` | Sentinel-2 NDVI | NDVI < 0.2 for 3+ scenes | MEDIUM |
| `precip_deficit` | ERA5 precipitation | 30-day total < 20th percentile | HIGH |
| `sar_glacier_change` | Sentinel-1 SAR VV | Mean dB shift > 3 dB from baseline | CRITICAL |
| `wind_extreme` | ERA5 wind speed | 7-day max > 95th percentile | LOW |
| `snow_melt_anomaly` | ERA5 temperature + snow | T>2C and zero snowfall 15+ days | MEDIUM |

> [!NOTE]
> All data sources fall back to **synthetic Patagonian climate data** if the real
> CSV/TIF files are not found. All 6 triggers run either way.

## Architecture

```
EnvironmentalMonitor
  -> load_era5()           # ERA5-Land CSV from Ch01
  -> load_ndvi_stack()     # NDVI batch TIFFs from Ch02
  -> load_sar()            # SAR TIFs from Ch07
  -> register_triggers()   # 6 TriggerEngine entries
  -> engine.run_all()      # Execute all check_fn()s
  -> generate_dashboard()  # 6-panel dark-mode figure
  -> save_report()         # JSON alert report
```

## Installation

```bash
mamba install -n geocascade_env -c conda-forge rasterio numpy pandas matplotlib -y
```

## Run

```bash
conda activate geocascade_env
python Chapter_10/25_agentic_monitor.py
```
