"""
Chapter 10: 25_agentic_monitor.py

Academic Objective:
Implement an Environmental Change Monitoring 'agent' -- a trigger-based
automated system that watches multiple geospatial data streams and fires
alerts when convergent evidence of environmental stress is detected.

This is the 'agentic' principle WITHOUT any LLM: pure threshold logic
orchestrated by a TriggerEngine class. Each trigger independently monitors
one data stream; convergent firing across multiple triggers = high confidence.

Trigger inventory (6 triggers):
  1. temp_anomaly       -- ERA5 monthly temp > mean + 2*std           [HIGH]
  2. drought_stress     -- NDVI < 0.2 for 3+ consecutive scenes       [MEDIUM]
  3. precip_deficit     -- 30-day precip < 20th percentile            [HIGH]
  4. sar_glacier_change -- SAR dB shift > 3 dB from baseline          [CRITICAL]
  5. wind_extreme       -- Wind speed > 95th percentile               [LOW]
  6. snow_melt_anomaly  -- Snowfall drops to 0 while temp > 2C        [MEDIUM]

Dependencies:
mamba install -n geocascade_env -c conda-forge rasterio numpy pandas matplotlib -y
\"\"\"

import os
import sys
import json
import glob
import datetime
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

try:
    import rasterio
    RIO_OK = True
except ImportError:
    RIO_OK = False
    print('[WARNING] rasterio not installed -- SAR/NDVI triggers will use synthetic data.')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(BASE_DIR, 'data', 'processed')
os.makedirs(OUT_DIR, exist_ok=True)

SEVERITY_COLOR = {
    'LOW':      '#27ae60',
    'MEDIUM':   '#f39c12',
    'HIGH':     '#e74c3c',
    'CRITICAL': '#8e44ad',
    'OK':       '#2ecc71',
}

# ==========================================================
# TriggerEngine
# ==========================================================
class TriggerEngine:
    def __init__(self):
        self.triggers = {}

    def register(self, name, description, check_fn, severity='MEDIUM'):
        self.triggers[name] = {
            'description': description,
            'check_fn':    check_fn,
            'severity':    severity,
        }

    def run_all(self):
        results = []
        for name, cfg in self.triggers.items():
            try:
                triggered, value, message = cfg['check_fn']()
            except Exception as e:
                triggered, value, message = False, 0.0, f'ERROR: {e}'
            results.append({
                'trigger':     name,
                'description': cfg['description'],
                'severity':    cfg['severity'] if triggered else 'OK',
                'triggered':   triggered,
                'value':       value,
                'message':     message,
            })
        return results

# ==========================================================
# EnvironmentalMonitor
# ==========================================================
class EnvironmentalMonitor:
    def __init__(self):
        self.engine  = TriggerEngine()
        self.era5_df = None
        self.ndvi_df = None
        self.sar_df  = None

    # ------------------------------------------------------
    # Data loaders
    # ------------------------------------------------------
    def load_era5(self):
        paths = glob.glob(
            os.path.join(BASE_DIR, '..', 'Chapter_01', 'data', 'raw', 'real_data',
                         'era5_daily_patagonia.csv')
        )
        if paths:
            try:
                df = pd.read_csv(paths[0], parse_dates=['date'])
                print(f'       [ERA5] Loaded {len(df)} rows from {os.path.basename(paths[0])}')
                return df
            except Exception as e:
                print(f'       [ERA5] Read error: {e}. Using synthetic.')
        print('       [ERA5] File not found -- generating 30-yr synthetic ERA5...')
        rng  = np.random.default_rng(42)
        dates = pd.date_range('1993-01-01', '2023-12-31', freq='D')
        n    = len(dates)
        doy  = np.array([d.dayofyear for d in dates])
        # Patagonian climate signal: cooler summers (Jan), warmer winters (Jul) due to S hemisphere
        temp = 8 + 4 * np.sin(2 * np.pi * (doy - 196) / 365) + rng.normal(0, 1.5, n)
        temp += np.linspace(0, 0.8, n)  # +0.8C warming trend over 30yr
        precip = np.clip(rng.exponential(3.0, n), 0, 80)
        wind   = np.clip(rng.weibull(2, n) * 12, 0, 80)
        snow   = np.clip(np.where(temp < 2, rng.exponential(2, n), 0), 0, 50)
        return pd.DataFrame({'date': dates, 'temperature_2m_mean': temp,
                             'precipitation_sum': precip,
                             'windspeed_10m_max': wind, 'snowfall_sum': snow})

    def load_ndvi_stack(self):
        pattern = os.path.join(BASE_DIR, '..', 'Chapter_02',
                               'data', 'processed', 'batch_indices', 'NDVI_*.tif')
        files = sorted(glob.glob(pattern))
        if not files or not RIO_OK:
            print('       [NDVI] No batch files found -- generating synthetic NDVI series...')
            rng   = np.random.default_rng(42)
            dates = pd.date_range('2023-01-01', '2023-12-31', freq='ME')
            ndvi  = 0.35 + 0.15 * np.sin(2*np.pi*np.arange(len(dates))/12) + rng.normal(0, 0.05, len(dates))
            return pd.DataFrame({'date': dates, 'ndvi_mean': np.clip(ndvi, 0, 1)})
        rows = []
        for f in files:
            try:
                date_str = os.path.basename(f).replace('NDVI_', '').replace('.tif', '')[:10]
                with rasterio.open(f) as src:
                    arr = src.read(1).astype('float32')
                    nd  = src.nodata if src.nodata is not None else -9999
                arr[arr == nd] = np.nan
                v = arr[np.isfinite(arr)]
                if v.size:
                    rows.append({'date': pd.to_datetime(date_str), 'ndvi_mean': float(np.nanmean(v))})
            except Exception as e:
                print(f'       [NDVI] Skip {f}: {e}')
        if rows:
            print(f'       [NDVI] Loaded {len(rows)} batch scenes.')
            return pd.DataFrame(rows).sort_values('date').reset_index(drop=True)
        return pd.DataFrame(columns=['date', 'ndvi_mean'])

    def load_sar(self):
        pattern = os.path.join(BASE_DIR, '..', 'Chapter_07',
                               'data', 'processed', 'sar_*.tif')
        files = sorted(glob.glob(pattern))
        if not files or not RIO_OK:
            print('       [SAR] No SAR files found -- generating synthetic SAR series...')
            rng   = np.random.default_rng(42)
            dates = pd.date_range('2023-01-01', '2023-12-31', freq='ME')
            db    = -15 + rng.normal(0, 1.5, len(dates))
            return pd.DataFrame({'date': dates, 'sar_db_mean': db})
        rows = []
        for f in files:
            try:
                date_str = os.path.basename(f).replace('sar_', '').replace('.tif', '')[:10]
                with rasterio.open(f) as src:
                    arr = src.read(1).astype('float32')
                    nd  = src.nodata if src.nodata is not None else -9999
                arr[arr == nd] = np.nan
                v = arr[np.isfinite(arr)]
                if v.size:
                    rows.append({'date': pd.to_datetime(date_str), 'sar_db_mean': float(np.nanmean(v))})
            except Exception as e:
                print(f'       [SAR] Skip {f}: {e}')
        if rows:
            return pd.DataFrame(rows).sort_values('date').reset_index(drop=True)
        return pd.DataFrame(columns=['date', 'sar_db_mean'])

    # ------------------------------------------------------
    # Register triggers
    # ------------------------------------------------------
    def register_triggers(self):
        era5 = self.era5_df
        ndvi = self.ndvi_df
        sar  = self.sar_df

        # 1. Temperature anomaly
        def check_temp():
            if era5 is None or era5.empty:
                return False, 0.0, 'No ERA5 data'
            recent = era5.tail(365)['temperature_2m_mean']
            hist   = era5['temperature_2m_mean']
            baseline_mean = hist.mean(); baseline_std = hist.std()
            latest_monthly = recent.resample('ME', on=era5.tail(365)['date']).mean() if False else recent.mean()
            anomaly = latest_monthly - baseline_mean
            threshold = 2 * baseline_std
            triggered = anomaly > threshold
            return (triggered, float(anomaly),
                    f'Temp anomaly {anomaly:+.2f}C (threshold +{threshold:.2f}C)')

        # Simplified version
        def check_temp_v2():
            if era5 is None or era5.empty:
                return False, 0.0, 'No ERA5 data'
            col = 'temperature_2m_mean'
            baseline = era5[col]
            bm, bs   = baseline.mean(), baseline.std()
            recent   = era5.iloc[-90:][col].mean()
            anom     = recent - bm
            return (bool(anom > 2*bs), float(anom),
                    f'90-day mean anomaly={anom:+.2f}C (2-sigma={2*bs:.2f}C)')

        self.engine.register('temp_anomaly', 'Temperature 2-sigma anomaly', check_temp_v2, 'HIGH')

        # 2. Drought / NDVI stress
        def check_drought():
            if ndvi is None or ndvi.empty:
                return False, 0.0, 'No NDVI data'
            low = (ndvi['ndvi_mean'] < 0.2).values
            streak = 0
            for v in low:
                if v: streak += 1
                else: streak = 0
            last = float(ndvi['ndvi_mean'].iloc[-1])
            triggered = streak >= 3
            return (triggered, last,
                    f'NDVI={last:.3f} -- {streak} consecutive scenes below 0.2')

        self.engine.register('drought_stress', 'NDVI drought stress (NDVI<0.2)', check_drought, 'MEDIUM')

        # 3. Precipitation deficit
        def check_precip():
            if era5 is None or era5.empty:
                return False, 0.0, 'No ERA5 data'
            col   = 'precipitation_sum'
            p20   = era5[col].quantile(0.20)
            r30   = era5.iloc[-30:][col].sum()
            clim_30 = era5[col].rolling(30, min_periods=20).sum().quantile(0.20)
            trig  = r30 < clim_30
            return (bool(trig), float(r30),
                    f'30-day precip={r30:.1f}mm (20th pct baseline={clim_30:.1f}mm)')

        self.engine.register('precip_deficit', 'Precipitation 30-day deficit', check_precip, 'HIGH')

        # 4. SAR glacier change
        def check_sar():
            if sar is None or sar.empty or len(sar) < 3:
                return False, 0.0, 'Insufficient SAR data'
            baseline = float(sar.iloc[:max(1,len(sar)//2)]['sar_db_mean'].mean())
            recent   = float(sar.iloc[-3:]['sar_db_mean'].mean())
            delta    = abs(recent - baseline)
            return (bool(delta > 3.0), float(delta),
                    f'SAR change |delta|={delta:.2f} dB (baseline={baseline:.2f}, recent={recent:.2f})')

        self.engine.register('sar_glacier_change', 'SAR backscatter glacier shift', check_sar, 'CRITICAL')

        # 5. Wind extreme
        def check_wind():
            if era5 is None or era5.empty:
                return False, 0.0, 'No ERA5 data'
            col  = 'windspeed_10m_max'
            p95  = era5[col].quantile(0.95)
            last = float(era5.iloc[-7:][col].max())
            return (bool(last > p95), float(last),
                    f'7-day max wind={last:.1f} m/s (95th pct={p95:.1f} m/s)')

        self.engine.register('wind_extreme', 'Extreme wind event', check_wind, 'LOW')

        # 6. Snow melt anomaly
        def check_snow():
            if era5 is None or era5.empty:
                return False, 0.0, 'No ERA5 data'
            recent = era5.iloc[-30:]
            warm_no_snow = ((recent['temperature_2m_mean'] > 2) &
                            (recent['snowfall_sum'] < 0.1))
            days = int(warm_no_snow.sum())
            return (bool(days >= 15), float(days),
                    f'{days} days with T>2C and no snowfall in last 30 days')

        self.engine.register('snow_melt_anomaly', 'Early snowmelt / warm+dry event', check_snow, 'MEDIUM')

    # ------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------
    def generate_dashboard(self, results):
        fig = plt.figure(figsize=(18, 12), facecolor='#0d1117')
        gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)
        fig.suptitle(
            'GeoCascade -- Environmental Monitoring Dashboard\n'
            'Torres del Paine | Automated Trigger Engine',
            fontsize=14, fontweight='bold', color='white', y=0.98)

        style = {'facecolor': '#161b22', 'grid_color': '#30363d', 'tc': 'white'}

        # Panel 1: Temperature
        ax1 = fig.add_subplot(gs[0, :2])
        if self.era5_df is not None and not self.era5_df.empty:
            df = self.era5_df.copy()
            t  = df['temperature_2m_mean']
            bm, bs = t.mean(), t.std()
            ax1.plot(df['date'], t, color='#ff6b6b', lw=0.8, alpha=0.7)
            ax1.axhline(bm + 2*bs, color='#e74c3c', lw=1.5, ls='--', label='+2-sigma')
            ax1.axhline(bm,        color='#3498db', lw=1.0, ls='-',  label='Mean', alpha=0.6)
            ax1.fill_between(df['date'], t, bm,
                             where=t > bm+2*bs, color='#e74c3c', alpha=0.3, label='Anomaly')
        ax1.set_title('ERA5 Temperature (1993-2024)', color='white', fontsize=9)
        ax1.set_ylabel('Temp (C)', color='white'); ax1.tick_params(colors='white')
        ax1.set_facecolor(style['facecolor']); ax1.legend(fontsize=7, facecolor='#0d1117', labelcolor='white')
        ax1.spines[['top','right']].set_visible(False)

        # Panel 2: Precipitation
        ax2 = fig.add_subplot(gs[1, :2])
        if self.era5_df is not None and not self.era5_df.empty:
            df = self.era5_df.tail(365).copy()
            p20 = self.era5_df['precipitation_sum'].quantile(0.20)
            ax2.bar(df['date'], df['precipitation_sum'],
                    color=['#e74c3c' if v < p20 else '#3498db'
                           for v in df['precipitation_sum']], width=1)
            ax2.axhline(p20, color='#e74c3c', lw=1.5, ls='--', label=f'20th pct ({p20:.1f}mm)')
        ax2.set_title('Daily Precipitation -- Last 12 Months (red=deficit)', color='white', fontsize=9)
        ax2.set_ylabel('Precip (mm)', color='white'); ax2.tick_params(colors='white')
        ax2.set_facecolor(style['facecolor']); ax2.legend(fontsize=7, facecolor='#0d1117', labelcolor='white')
        ax2.spines[['top','right']].set_visible(False)

        # Panel 3: NDVI
        ax3 = fig.add_subplot(gs[2, :2])
        if self.ndvi_df is not None and not self.ndvi_df.empty:
            nd = self.ndvi_df
            ax3.plot(nd['date'], nd['ndvi_mean'], 'o-', color='#2ecc71', lw=1.5, ms=4)
            ax3.axhline(0.2, color='#e74c3c', lw=1.5, ls='--', label='Drought threshold (0.2)')
            ax3.fill_between(nd['date'], nd['ndvi_mean'], 0.2,
                             where=nd['ndvi_mean'] < 0.2, color='#e74c3c', alpha=0.25)
        ax3.set_title('Sentinel-2 NDVI Time Series', color='white', fontsize=9)
        ax3.set_ylabel('NDVI', color='white'); ax3.set_ylim(-0.1, 1.0)
        ax3.tick_params(colors='white'); ax3.set_facecolor(style['facecolor'])
        ax3.legend(fontsize=7, facecolor='#0d1117', labelcolor='white')
        ax3.spines[['top','right']].set_visible(False)

        # Panel 4: Active triggers table
        ax4 = fig.add_subplot(gs[0, 2])
        ax4.axis('off')
        tbl_data = [[r['trigger'], r['severity'], 'YES' if r['triggered'] else 'no']
                    for r in results]
        t = ax4.table(cellText=tbl_data, colLabels=['Trigger', 'Severity', 'Fired'],
                      cellLoc='left', loc='center')
        t.auto_set_font_size(False); t.set_fontsize(7)
        for (row, col), cell in t.get_celld().items():
            cell.set_facecolor('#161b22'); cell.set_text_props(color='white')
            if row > 0 and col == 1:
                sev = tbl_data[row-1][1]
                cell.set_facecolor(SEVERITY_COLOR.get(sev, '#444'))
        ax4.set_title('Trigger Status', color='white', fontsize=9)

        # Panel 5: Severity gauge
        ax5 = fig.add_subplot(gs[1, 2])
        sev_counts = {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0, 'CRITICAL': 0}
        for r in results:
            if r['triggered']:
                sev_counts[r['severity']] = sev_counts.get(r['severity'], 0) + 1
        labels_g = [k for k, v in sev_counts.items() if v > 0] or ['OK']
        sizes_g  = [v for v in sev_counts.values() if v > 0] or [1]
        colors_g = [SEVERITY_COLOR[k] for k in labels_g]
        ax5.pie(sizes_g, labels=labels_g, colors=colors_g, autopct='%1.0f%%',
                textprops={'color': 'white', 'fontsize': 8})
        ax5.set_title('Active Alert Severity', color='white', fontsize=9)
        ax5.set_facecolor(style['facecolor'])

        # Panel 6: SAR
        ax6 = fig.add_subplot(gs[2, 2])
        if self.sar_df is not None and not self.sar_df.empty:
            sd = self.sar_df
            ax6.plot(sd['date'], sd['sar_db_mean'], 's-', color='#e67e22', lw=1.5, ms=4)
            if len(sd) >= 3:
                baseline = sd.iloc[:max(1,len(sd)//2)]['sar_db_mean'].mean()
                ax6.axhline(baseline, color='#95a5a6', lw=1, ls='--', label=f'Baseline {baseline:.1f}dB')
                ax6.axhline(baseline + 3, color='#8e44ad', lw=1.5, ls=':', label='+3dB alert')
                ax6.axhline(baseline - 3, color='#8e44ad', lw=1.5, ls=':')
            ax6.legend(fontsize=7, facecolor='#0d1117', labelcolor='white')
        ax6.set_title('SAR Backscatter (VV dB)', color='white', fontsize=9)
        ax6.set_ylabel('dB', color='white'); ax6.tick_params(colors='white')
        ax6.set_facecolor(style['facecolor']); ax6.spines[['top','right']].set_visible(False)

        out = os.path.join(OUT_DIR, 'monitoring_dashboard.png')
        fig.savefig(out, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f'       [SUCCESS] Dashboard: {out}')
        return out

    # ------------------------------------------------------
    # Save report
    # ------------------------------------------------------
    def save_report(self, results):
        ts   = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(OUT_DIR, f'monitoring_report_{ts}.json')
        n_triggered = sum(r['triggered'] for r in results)
        report = {
            'generated_at':   datetime.datetime.now().isoformat(),
            'study_area':     'Torres del Paine, Patagonia, Chile',
            'triggers_total': len(results),
            'triggers_fired': n_triggered,
            'results':        results,
        }
        with open(path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f'       [SUCCESS] Report: {path}')
        return path

    # ------------------------------------------------------
    # Run
    # ------------------------------------------------------
    def run(self):
        print('\n[1/4] Loading ERA5 climate data...')
        self.era5_df = self.load_era5()
        print('\n[2/4] Loading NDVI batch stack...')
        self.ndvi_df = self.load_ndvi_stack()
        print('\n[3/4] Loading SAR time series...')
        self.sar_df  = self.load_sar()
        print('\n[4/4] Running trigger engine...')
        self.register_triggers()
        results = self.engine.run_all()
        print()
        print(f'  {'TRIGGER':<25} {'SEVERITY':<10} {'FIRED':<6} VALUE   MESSAGE')
        print(f'  {'-'*80}')
        for r in results:
            sev = r['severity']; fired = 'YES' if r['triggered'] else 'no'
            print(f'  {r["trigger"]:<25} {sev:<10} {fired:<6} {r["value"]:6.2f}  {r["message"]}')
        print()
        n_fired = sum(r['triggered'] for r in results)
        print(f'  Triggers fired: {n_fired}/{len(results)}')
        self.generate_dashboard(results)
        self.save_report(results)
        return results

# ==========================================================
# Main
# ==========================================================
def main():
    print('=' * 65)
    print(' GEOCASCADE -- CHAPTER 10: AGENTIC ENVIRONMENTAL MONITOR')
    print('=' * 65)
    monitor = EnvironmentalMonitor()
    results = monitor.run()
    n_fired = sum(r['triggered'] for r in results)
    if n_fired == 0:
        print('\n[STATUS] All systems nominal. No environmental alerts.')
    elif n_fired <= 2:
        print(f'\n[STATUS] ADVISORY -- {n_fired} trigger(s) active. Monitor closely.')
    elif n_fired <= 4:
        print(f'\n[STATUS] WARNING -- {n_fired} triggers active. Investigate data streams.')
    else:
        print(f'\n[STATUS] CRITICAL -- {n_fired} triggers active! Immediate review required.')
    print('\n[SUCCESS] Chapter 10 complete.')
    print(' Next: Chapter_11/26_postgis_integration.py')

if __name__ == '__main__':
    main()
