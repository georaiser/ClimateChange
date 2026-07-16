# 🤖 Chapter 10: Agentic Environmental Change Monitoring

> **GeoCascade Pipeline — Stage 10**
> A threshold-based autonomous TriggerEngine that watches 6 geophysical data streams
> simultaneously and raises convergent alerts when multiple sensors agree on stress.

---

## 📋 Overview

| Script | Topic | Key Outputs |
|--------|-------|-------------|
| `25_agentic_monitor.py` | 6-trigger TriggerEngine, convergence dashboard | `monitor_alerts.json`, `trigger_status.csv`, 6-panel dark dashboard |

---

## 🚀 Setup

```bash
conda activate geocascade_env

# No LLM required — pure trigger logic
mamba install -n geocascade_env -c conda-forge \
    rasterio numpy pandas matplotlib -y
```

---

## ▶️ Run

```bash
python Chapter_10/25_agentic_monitor.py
```

**Expected output:**
```
================================================================
 GEOCASCADE -- CHAPTER 10: AGENTIC ENVIRONMENTAL MONITORING
 Running 6 independent triggers...
================================================================
[Trigger 1] temp_anomaly     ... FIRED  (value=3.21σ) [HIGH]
[Trigger 2] drought_stress   ... OK     (NDVI=0.41)   [MEDIUM]
[Trigger 3] precip_deficit   ... FIRED  (15th pct)    [HIGH]
[Trigger 4] sar_glacier_chng ... OK     (Δ=1.2 dB)    [CRITICAL]
[Trigger 5] wind_extreme     ... OK     (88th pct)    [LOW]
[Trigger 6] snow_melt_anomaly... FIRED  (T=3.1°C)     [MEDIUM]

CONVERGENCE ANALYSIS:
  Triggers fired: 3 / 6
  HIGH or above:  2
  → CONVERGENT ALERT: Multiple independent sensors confirm environmental stress.
```

---

## 🔬 The Agentic Concept

This chapter demonstrates **agentic monitoring WITHOUT an LLM**. The TriggerEngine
is a decision loop where each trigger independently evaluates one data stream,
and the system escalates when **convergent evidence** accumulates.

> [!IMPORTANT]
> Why not just check one variable? A single sensor may have instrument drift,
> cloud contamination, or seasonal bias. **Convergence** — multiple independent
> sensors agreeing simultaneously — is the scientific gold standard for
> detecting real environmental change vs. noise.

### TriggerEngine Class Design

```python
class TriggerEngine:
    def register(self, name, description, check_fn, severity="MEDIUM"):
        """Register a monitoring function that returns (triggered, value, message)."""
        self.triggers[name] = {"check_fn": check_fn, "severity": severity}

    def run_all(self):
        """Run every trigger and aggregate results."""
        return [{"trigger": name, "triggered": triggered, ...}
                for name, cfg in self.triggers.items()
                for triggered, value, message in [cfg["check_fn"]()]]
```

Each `check_fn` must return `(triggered: bool, value: float, message: str)`.
This contract makes triggers interchangeable and testable in isolation.

---

## 🚦 Trigger Inventory

| # | Trigger Name | Data Source | Threshold | Severity | Physical Meaning |
|---|-------------|-------------|-----------|----------|-----------------|
| 1 | `temp_anomaly` | ERA5 temperature | > mean + 2σ | HIGH | Extreme heat event |
| 2 | `drought_stress` | NDVI time series | NDVI < 0.2 for 3+ scenes | MEDIUM | Vegetation collapse |
| 3 | `precip_deficit` | ERA5 precipitation | < 20th percentile (30-day) | HIGH | Drought conditions |
| 4 | `sar_glacier_change` | Sentinel-1 VV dB | Δ > 3 dB from baseline | CRITICAL | Rapid ice mass loss |
| 5 | `wind_extreme` | ERA5 wind speed | > 95th percentile | LOW | Aeolian erosion risk |
| 6 | `snow_melt_anomaly` | ERA5 snowfall + temp | snowfall=0 AND temp > 2°C | MEDIUM | Anomalous melt |

### Severity Color Scheme

| Severity | Color | Action |
|----------|-------|--------|
| OK | 🟢 `#2ecc71` | No action required |
| LOW | 🟡 `#27ae60` | Log only |
| MEDIUM | 🟠 `#f39c12` | Notify analyst |
| HIGH | 🔴 `#e74c3c` | Investigate + report |
| CRITICAL | 🟣 `#8e44ad` | Immediate response |

---

## 📐 Convergence Logic

```python
fired    = [r for r in results if r["triggered"]]
n_fired  = len(fired)
n_high   = sum(1 for r in fired if r["severity"] in ("HIGH", "CRITICAL"))

if n_fired >= 3:
    status = "CONVERGENT ALERT: Multiple independent sensors confirm stress."
elif n_fired >= 1:
    status = "ISOLATED TRIGGER: Monitor closely but not yet convergent."
else:
    status = "NOMINAL: All triggers within baseline."
```

**Why N≥3 convergence?** With 6 independent triggers, random chance of 3+ firing simultaneously is < 3% (assuming 15% base rate per trigger). Convergence at N≥3 dramatically reduces false positives.

---

## 📊 Trigger State Graph

```
  [ERA5 temp] ──────────────────────┐
  [NDVI series] ────────────────────┤
  [ERA5 precip] ────────────────────┤──→ TriggerEngine.run_all()
  [SAR VV dB baseline] ─────────────┤         │
  [ERA5 wind] ──────────────────────┤    [Results list]
  [ERA5 snowfall] ──────────────────┘         │
                                         ┌────▼────────┐
                                         │ Convergence │
                                         │  Analysis   │
                                         └────┬────────┘
                                              │
                        ┌─────────────────────┴──────────────────┐
                        ▼                                        ▼
               monitor_alerts.json                    trigger_dashboard.png
               trigger_status.csv                     (6-panel dark figure)
```

---

## 📂 Output Structure

```
Chapter_10/
└── data/processed/
    ├── monitor_alerts.json           ← Full trigger results with timestamps + values
    ├── trigger_status.csv            ← Summary table: trigger, severity, value, fired
    └── trigger_dashboard.png         ← 6-panel: one panel per trigger, time series + status
```

### monitor_alerts.json structure

```json
{
  "timestamp": "2024-01-15T12:00:00",
  "convergence_level": 3,
  "status": "CONVERGENT ALERT",
  "triggers": [
    {
      "trigger": "temp_anomaly",
      "triggered": true,
      "value": 3.21,
      "message": "ERA5 temperature anomaly: +3.21σ above 30-year baseline",
      "severity": "HIGH"
    }
  ]
}
```

---

## 🖥️ ArcGIS Pro Integration

```
This chapter is attribute-data oriented (time series and trigger thresholds),
not primarily spatial raster outputs. However:

1. The SAR trigger uses Chapter 7 VV dB outputs.
   Add Chapter_07/data/processed/sar/sar_vv_db.tif
   Compare baseline vs current scene using Raster Calculator:
   "current_sar.tif" - "baseline_sar.tif"
   → Pixels where Δ > 3 dB = sar_glacier_change trigger zone

2. The drought trigger uses Chapter 8 NDVI.
   Add cnn_landcover_prediction.tif from Chapter 9
   Use Select By Attributes: class = 3 (Vegetation)
   → Check mean NDVI inside these zones (must be > 0.2 to not trigger)
```

---

## 🔵 ENVI 5.6 Integration

```
; SAR baseline comparison
File > Open > sar_vv_baseline.tif
File > Open > sar_vv_current.tif
Toolbox > Band Math
  b2 - b1    (current - baseline in dB)
  Apply threshold: > 3.0 → glacier change pixels
  Save result and compare with script 25 SAR trigger output
```

---

## ⚠️ Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| All triggers return synthetic data | Missing Ch07 SAR or Ch01 ERA5 cache | Run Ch01 and Ch07 first; script falls back to synthetic data |
| `trigger_status.csv` has all "OK" | All thresholds below baseline | Expected if data is from a quiet season — check with winter date range |
| JSON export fails | Unicode in message strings | Fixed: `sys.stdout.reconfigure(encoding='utf-8')` |

---

## 📖 Key References

- Zhu, Z., et al. (2014). *Continuous change detection and classification of land cover using all available Landsat data.* Remote Sensing of Environment.
- Verbesselt, J., et al. (2010). *Detecting trend and seasonal changes in satellite image time series.* Remote Sensing of Environment.
- Copernicus Emergency Management Service. *Wildfire and flood trigger protocols for operational environmental monitoring.*
