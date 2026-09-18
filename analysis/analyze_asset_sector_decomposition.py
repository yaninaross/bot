# %% ── Asset-by-Asset & Sector Performance Decomposition ─────────────────────
#
# Deep analysis of WHERE our edge exists, WHERE it's gone, and WHERE
# new opportunities might be forming.
#
# Phase 1: Per-ticker pre-market spike rates & MFE over time (GCS data)
# Phase 2: Sector-level aggregation & temporal trends
# Phase 3: Coverage gap scan — find spiking tickers outside deploy universe
# Phase 4: Model-sector specialization analysis (BQ scoring logs)
# Phase 5: Signal migration detection — where are spikes moving?
#
# Data sources: GCS master_inference parquets, BQ live_predictions
# Colab-ready. No GPU needed (data analysis only).

import os, sys, gc, ctypes, subprocess, warnings
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd
import pytz

warnings.filterwarnings('ignore')

try:
    from google.cloud import storage as gcs_storage
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install",
        "google-cloud-storage", "--quiet", "--break-system-packages"])
    from google.cloud import storage as gcs_storage

try:
    from google.cloud import bigquery
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install",
        "google-cloud-bigquery", "--quiet", "--break-system-packages"])
    from google.cloud import bigquery

try:
    from google.colab import auth
    _IN_COLAB = True
    try:
        auth.authenticate_user()
    except Exception:
        pass
except ImportError:
    _IN_COLAB = False

import matplotlib
if not _IN_COLAB:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch

# ==============================================================================
# CONFIG
# ==============================================================================
BUCKET_NAME   = os.environ.get("GCS_BUCKET", "")
INFERENCE_DIR = "Data/Inference_v2"
BQ_LIVE_TABLE = f"{GCP_PROJECT}.scoring_logs.live_predictions"
BQ_WF_TABLE   = f"{GCP_PROJECT}.scoring_logs.wf_optimization_results"
PST           = pytz.timezone('America/Los_Angeles')

ANALYSIS_START = '2026-07-01'
ANALYSIS_END   = '2026-09-07'

TAIL_DAYS = 75

PREMARKET_START_MIN = 360   # 6:00 AM PT
PREMARKET_END_MIN   = 390   # 6:30 AM PT
ENTRY_MINUTE        = 385   # 6:25 AM PT

SPIKE_THRESHOLD_PCT = 3.0   # 3% = spike
SPIKE_THRESHOLD_2   = 2.0   # 2% = moderate move
SPIKE_THRESHOLD_5   = 5.0   # 5% = large move

W = 120

gcs_client = gcs_storage.Client()

# ==============================================================================
# TICKER-TO-SECTOR MAPPING — comprehensive
# ==============================================================================
TICKER_SECTOR = {
    # Crypto Mining
    'CIFR': 'Crypto', 'IREN': 'Crypto', 'KEEL': 'Crypto', 'HUT': 'Crypto',
    'APLD': 'Crypto', 'WULF': 'Crypto', 'RIOT': 'Crypto', 'CLSK': 'Crypto',
    'BTBT': 'Crypto', 'MARA': 'Crypto', 'CORZ': 'Crypto', 'MSTR': 'Crypto',
    'BITF': 'Crypto', 'BGDE': 'Crypto', 'MIGI': 'Crypto',

    # Defense / Space
    'RCAT': 'Defense', 'ASTS': 'Defense', 'LUNR': 'Defense', 'BBAI': 'Defense',
    'RKLB': 'Defense', 'SPIR': 'Defense', 'BKSY': 'Defense', 'MNTS': 'Defense',
    'PL': 'Defense', 'RDW': 'Defense', 'VOYG': 'Defense', 'KTOS': 'Defense',
    'GD': 'Defense', 'GE': 'Defense', 'GEV': 'Defense', 'LHX': 'Defense',
    'LMT': 'Defense', 'NOC': 'Defense', 'RTX': 'Defense', 'JOBY': 'Defense',
    'AVAV': 'Defense', 'ACHR': 'Defense', 'DFNS': 'Defense',

    # Clean Energy
    'PLUG': 'CleanEnergy', 'BE': 'CleanEnergy', 'FCEL': 'CleanEnergy',
    'FLNC': 'CleanEnergy',

    # Nuclear
    'OKLO': 'Nuclear', 'SMR': 'Nuclear', 'NNE': 'Nuclear', 'CCJ': 'Nuclear',
    'CEG': 'Nuclear', 'VST': 'Nuclear',

    # Semiconductors / AI Hardware
    'AMD': 'Semis', 'NVDA': 'Semis', 'SOUN': 'Semis', 'VRT': 'Semis',
    'ALAB': 'Semis', 'MVIS': 'Semis', 'AEHR': 'Semis', 'ACMR': 'Semis',
    'MXL': 'Semis', 'AAOI': 'Semis', 'MRVL': 'Semis', 'WDC': 'Semis',
    'CRDO': 'Semis', 'SMCI': 'Semis', 'WOLF': 'Semis', 'AXTI': 'Semis',
    'MU': 'Semis',

    # Tech / Software
    'PLTR': 'Tech', 'TSLA': 'Tech', 'HOOD': 'Tech', 'SLS': 'Tech',
    'NBIS': 'Tech', 'NTLA': 'Tech',

    # Energy / Resources
    'BHP': 'Energy', 'CAT': 'Energy', 'COP': 'Energy', 'CVX': 'Energy',
    'XOM': 'Energy',

    # Infrastructure / Utilities
    'DUK': 'Utilities', 'EQIX': 'Utilities', 'ETN': 'Utilities',
    'LIN': 'Utilities', 'NEE': 'Utilities',
}

SECTOR_COLORS = {
    'Crypto': '#FF6F00',
    'Defense': '#1565C0',
    'CleanEnergy': '#2E7D32',
    'Nuclear': '#7B1FA2',
    'Semis': '#C62828',
    'Tech': '#00838F',
    'Energy': '#5D4037',
    'Utilities': '#757575',
    'Unknown': '#BDBDBD',
}

DEPLOY_TICKERS_LONG = set()
DEPLOY_TICKERS_SHORT = set()

TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}
EXCLUDE_TICKERS = {'SPY', 'VIX', 'TLT', 'IWM', 'XLE', 'ITA', 'BTCUSD', 'ETHUSD'}

_NEEDED_COLS = {'Open', 'High', 'Low', 'Close', 'Volume',
                'open', 'high', 'low', 'close', 'volume',
                'Datetime', 'Date', 'Time', 'Timestamp', 'Level_0'}


# ==============================================================================
# DATA LOADING
# ==============================================================================
def load_master(ticker, tail_days=None):
    clean = ticker.replace("I:", "").replace("X:", "")
    file_ticker = TICKER_FILE_MAP.get(clean, clean)
    blob_path = f"{INFERENCE_DIR}/{file_ticker}_master_inference.parquet"
    local_path = f"/tmp/_sector_analysis_{file_ticker}.parquet"
    try:
        gcs_client.bucket(BUCKET_NAME).blob(blob_path).download_to_filename(local_path)
        import pyarrow.parquet as pq
        schema = pq.read_schema(local_path)
        all_cols = [f.name for f in schema]
        use_cols = [c for c in all_cols if c in _NEEDED_COLS]
        if not use_cols:
            use_cols = None
        df = pd.read_parquet(local_path, columns=use_cols)
        os.remove(local_path)
        df = df.reset_index()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        if dt_cols:
            df['Datetime'] = pd.to_datetime(df[dt_cols[0]], utc=True)
        else:
            num_col = next(
                (c for c in df.columns
                 if c in ('Datetime', 'Date', 'Time', 'Timestamp', 'Level_0')),
                df.columns[0])
            df['Datetime'] = pd.to_datetime(df[num_col], unit='ms', utc=True)
        df = df.set_index('Datetime')
        df = df[~df.index.duplicated(keep='first')]
        std_map = {
            'open': 'Open', 'high': 'High', 'low': 'Low',
            'close': 'Close', 'volume': 'Volume',
        }
        lower_map = {k.lower(): v for k, v in std_map.items()}
        df.columns = [lower_map.get(str(c).strip().lower(), str(c).strip())
                      for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated(keep='first')].copy()
        df = df[df.index.notna()].sort_index()
        if tail_days is not None:
            cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=tail_days)
            df = df[df.index >= cutoff]
            if df.empty:
                return None
        return df
    except Exception:
        if os.path.exists(local_path):
            os.remove(local_path)
        return None


# ==============================================================================
# SPIKE ANALYSIS FUNCTIONS
# ==============================================================================
def compute_daily_spike_metrics(df, ticker):
    """Compute per-day pre-market spike metrics for a single ticker."""
    try:
        df_pst = df.index.tz_convert(PST)
    except Exception:
        return []
    mins = df_pst.hour * 60 + df_pst.minute
    df = df.copy()
    df['_min'] = mins
    df['_date'] = df_pst.date

    bt_start = pd.Timestamp(ANALYSIS_START).date()
    bt_end = pd.Timestamp(ANALYSIS_END).date()

    results = []
    for date_val, day_df in df.groupby('_date'):
        if date_val < bt_start or date_val > bt_end:
            continue
        if len(day_df) < 30:
            continue

        pm = day_df[(day_df['_min'] >= PREMARKET_START_MIN) &
                     (day_df['_min'] < PREMARKET_END_MIN)]
        if len(pm) < 5:
            continue

        entry_bars = day_df[day_df['_min'] == ENTRY_MINUTE]
        if len(entry_bars) == 0:
            entry_bars = day_df[(day_df['_min'] >= ENTRY_MINUTE - 2) &
                                (day_df['_min'] <= ENTRY_MINUTE + 2)]
        if len(entry_bars) == 0:
            continue

        entry_price = float(entry_bars.iloc[0]['Close'])
        if entry_price <= 0:
            continue

        after_entry = day_df[day_df['_min'] > ENTRY_MINUTE]
        if len(after_entry) < 10:
            continue

        highs_after = after_entry['High'].values
        lows_after = after_entry['Low'].values
        close_eod = float(after_entry.iloc[-1]['Close'])

        # All-day MFE (reference only — inflated by full-day window)
        mfe_long_allday = (max(highs_after) - entry_price) / entry_price * 100
        mfe_short_allday = (entry_price - min(lows_after)) / entry_price * 100

        # 60-minute MFE (matching model prediction horizon: 15/30/60 min)
        after_60m = day_df[(day_df['_min'] > ENTRY_MINUTE) &
                           (day_df['_min'] <= ENTRY_MINUTE + 60)]
        if len(after_60m) >= 3:
            mfe_long = (max(after_60m['High'].values) - entry_price) / entry_price * 100
            mae_long = (min(after_60m['Low'].values) - entry_price) / entry_price * 100
            mfe_short = (entry_price - min(after_60m['Low'].values)) / entry_price * 100
            mae_short = (max(after_60m['High'].values) - entry_price) / entry_price * 100
        else:
            mfe_long = mfe_long_allday
            mae_long = (min(lows_after) - entry_price) / entry_price * 100
            mfe_short = mfe_short_allday
            mae_short = (max(highs_after) - entry_price) / entry_price * 100

        eod_return = (close_eod - entry_price) / entry_price * 100

        pm_volume = float(pm['Volume'].sum()) if 'Volume' in pm.columns else 0
        pm_range = 0
        if len(pm) > 0 and entry_price > 0:
            pm_range = (pm['High'].max() - pm['Low'].min()) / entry_price * 100

        # Pre-market spike magnitude (matching 360 diagnosis methodology)
        pm_open = float(pm.iloc[0]['Open'])
        if pm_open > 0:
            pm_spike_up = (pm['High'].max() - pm_open) / pm_open * 100
            pm_spike_down = (pm_open - pm['Low'].min()) / pm_open * 100
            pm_spike_mag = max(pm_spike_up, pm_spike_down)
        else:
            pm_spike_up = pm_spike_down = pm_spike_mag = 0

        results.append({
            'Ticker': ticker,
            'Date': str(date_val),
            'Date_dt': pd.Timestamp(date_val),
            'Entry_Price': entry_price,
            'MFE_Long': mfe_long,
            'MAE_Long': mae_long,
            'MFE_Short': mfe_short,
            'MAE_Short': mae_short,
            'MFE_Long_AllDay': mfe_long_allday,
            'MFE_Short_AllDay': mfe_short_allday,
            'PM_Spike_Up': pm_spike_up,
            'PM_Spike_Down': pm_spike_down,
            'PM_Spike_Mag': pm_spike_mag,
            'EOD_Return': eod_return,
            'PM_Volume': pm_volume,
            'PM_Range': pm_range,
            'Spike_Long': mfe_long >= SPIKE_THRESHOLD_PCT,
            'Spike_Short': mfe_short >= SPIKE_THRESHOLD_PCT,
            'Spike_Long_2': mfe_long >= SPIKE_THRESHOLD_2,
            'Spike_Short_2': mfe_short >= SPIKE_THRESHOLD_2,
            'Spike_Long_5': mfe_long >= SPIKE_THRESHOLD_5,
            'Spike_Short_5': mfe_short >= SPIKE_THRESHOLD_5,
            'PM_Spike_3pct': pm_spike_mag >= SPIKE_THRESHOLD_PCT,
            'Sector': TICKER_SECTOR.get(ticker, 'Unknown'),
        })
    return results


def list_gcs_inference_tickers():
    """List all tickers with master_inference parquets in GCS."""
    bucket = gcs_client.bucket(BUCKET_NAME)
    prefix = f"{INFERENCE_DIR}/"
    blobs = bucket.list_blobs(prefix=prefix)
    tickers = []
    for blob in blobs:
        name = blob.name.split('/')[-1]
        if name.endswith('_master_inference.parquet'):
            tk = name.replace('_master_inference.parquet', '')
            if tk not in EXCLUDE_TICKERS:
                tickers.append(tk)
    return sorted(set(tickers))


# ==============================================================================
# PHASE 0: Discover full GCS universe + deployed tickers from BQ
# ==============================================================================
print(f"\n{'='*W}")
print(f"  ASSET-BY-ASSET & SECTOR PERFORMANCE DECOMPOSITION")
print(f"  {ANALYSIS_START} → {ANALYSIS_END}")
print(f"  Spike thresholds: {SPIKE_THRESHOLD_2}% (moderate), "
      f"{SPIKE_THRESHOLD_PCT}% (spike), {SPIKE_THRESHOLD_5}% (large)")
print(f"  MFE window: 60 minutes after entry (matching model horizon)")
print(f"  PM spike: pre-market magnitude (matching 360 diagnosis)")
print(f"{'='*W}")

print(f"\n  Phase 0: Discovering ticker universe...")

all_gcs_tickers = list_gcs_inference_tickers()
print(f"    GCS master_inference tickers: {len(all_gcs_tickers)}")

bq_client = bigquery.Client()

# Get deployed LONG tickers
q_long = f"""
SELECT DISTINCT ticker
FROM `{BQ_WF_TABLE}`
WHERE model_version LIKE 'spike_transformer_long_%'
  AND survived_a = TRUE
  AND val_b_avg_pnl >= 0.018
"""
df_long_tk = bq_client.query(q_long).to_dataframe()
DEPLOY_TICKERS_LONG = set(df_long_tk['ticker'].tolist()) - EXCLUDE_TICKERS
print(f"    LONG deployed tickers (BQ survived): {len(DEPLOY_TICKERS_LONG)}")

# Get deployed SHORT tickers
q_short = f"""
SELECT DISTINCT ticker
FROM `{BQ_WF_TABLE}`
WHERE model_version LIKE 'spike_transformer_short_%'
  AND survived_a = TRUE
  AND val_b_avg_pnl >= 0.018
"""
df_short_tk = bq_client.query(q_short).to_dataframe()
DEPLOY_TICKERS_SHORT = set(df_short_tk['ticker'].tolist()) - EXCLUDE_TICKERS
print(f"    SHORT deployed tickers (BQ survived): {len(DEPLOY_TICKERS_SHORT)}")

ALL_DEPLOY = DEPLOY_TICKERS_LONG | DEPLOY_TICKERS_SHORT
NOT_DEPLOYED = set(all_gcs_tickers) - ALL_DEPLOY - EXCLUDE_TICKERS

print(f"    Total deployed (either direction): {len(ALL_DEPLOY)}")
print(f"    In GCS but NOT deployed: {len(NOT_DEPLOYED)}")
for tk in sorted(NOT_DEPLOYED):
    sector = TICKER_SECTOR.get(tk, 'Unknown')
    print(f"      {tk} ({sector})")


# ==============================================================================
# PHASE 1: Per-ticker spike analysis across FULL universe
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  PHASE 1: PER-TICKER SPIKE METRICS ({len(all_gcs_tickers)} tickers)")
print(f"{'='*W}")

all_daily_metrics = []
ticker_status = {}

for i, ticker in enumerate(all_gcs_tickers):
    if ticker in EXCLUDE_TICKERS:
        continue
    sector = TICKER_SECTOR.get(ticker, 'Unknown')
    is_deployed = ticker in ALL_DEPLOY
    tag = 'DEPLOY' if is_deployed else 'SCAN'

    df = load_master(ticker, tail_days=TAIL_DAYS)
    if df is None:
        ticker_status[ticker] = {'status': 'NO_DATA', 'sector': sector}
        continue

    metrics = compute_daily_spike_metrics(df, ticker)
    if not metrics:
        ticker_status[ticker] = {'status': 'NO_BARS', 'sector': sector}
        continue

    n_days = len(metrics)
    n_spike_l = sum(1 for m in metrics if m['Spike_Long'])
    n_spike_s = sum(1 for m in metrics if m['Spike_Short'])
    n_pm_spike = sum(1 for m in metrics if m['PM_Spike_3pct'])
    avg_mfe_l = np.mean([m['MFE_Long'] for m in metrics])
    avg_mfe_s = np.mean([m['MFE_Short'] for m in metrics])
    avg_pm_mag = np.mean([m['PM_Spike_Mag'] for m in metrics])
    spike_rate_l = n_spike_l / n_days * 100 if n_days > 0 else 0
    spike_rate_s = n_spike_s / n_days * 100 if n_days > 0 else 0
    pm_spike_rate = n_pm_spike / n_days * 100 if n_days > 0 else 0

    ticker_status[ticker] = {
        'status': 'OK',
        'sector': sector,
        'deployed': is_deployed,
        'n_days': n_days,
        'spike_rate_long': spike_rate_l,
        'spike_rate_short': spike_rate_s,
        'avg_mfe_long': avg_mfe_l,
        'avg_mfe_short': avg_mfe_s,
        'avg_pm_mag': avg_pm_mag,
        'pm_spike_rate': pm_spike_rate,
        'n_spike_long': n_spike_l,
        'n_spike_short': n_spike_s,
        'n_pm_spike': n_pm_spike,
    }

    all_daily_metrics.extend(metrics)

    if (i + 1) % 20 == 0 or i == len(all_gcs_tickers) - 1:
        print(f"    Processed {i+1}/{len(all_gcs_tickers)} tickers, "
              f"{len(all_daily_metrics)} daily observations so far")

    del df
    gc.collect()

df_daily = pd.DataFrame(all_daily_metrics)
df_daily['Date_dt'] = pd.to_datetime(df_daily['Date'])
df_daily['Week'] = df_daily['Date_dt'].dt.isocalendar().week.astype(int)
df_daily['Year_Week'] = (df_daily['Date_dt'].dt.isocalendar().year.astype(str) + '-W' +
                          df_daily['Date_dt'].dt.isocalendar().week.astype(str).str.zfill(2))
df_daily['Deployed'] = df_daily['Ticker'].isin(ALL_DEPLOY)

print(f"\n    Total daily observations: {len(df_daily)}")
print(f"    Tickers with data: {df_daily['Ticker'].nunique()}")
print(f"    Date range: {df_daily['Date'].min()} → {df_daily['Date'].max()}")


# ==============================================================================
# PHASE 1b: Per-Ticker Summary Table
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  PER-TICKER SPIKE SUMMARY (sorted by total spike rate)")
print(f"{'='*W}\n")

ticker_summary = []
for tk, info in sorted(ticker_status.items()):
    if info['status'] != 'OK':
        continue
    ticker_summary.append({
        'Ticker': tk,
        'Sector': info['sector'],
        'Deployed': 'Y' if info.get('deployed', False) else 'N',
        'Days': info['n_days'],
        'SpikeL': info['n_spike_long'],
        'SpikeS': info['n_spike_short'],
        'RateL': info['spike_rate_long'],
        'RateS': info['spike_rate_short'],
        'AvgMFE_L': info['avg_mfe_long'],
        'AvgMFE_S': info['avg_mfe_short'],
        'PM_Rate': info['pm_spike_rate'],
        'AvgPM': info['avg_pm_mag'],
        'TotalRate': info['spike_rate_long'] + info['spike_rate_short'],
    })

df_summary = pd.DataFrame(ticker_summary).sort_values('TotalRate', ascending=False)

print(f"  NOTE: MFE = 60-min window after entry (not all-day). PM = pre-market spike magnitude.\n")
print(f"  {'Ticker':<7} {'Sector':<12} {'Dep':>3} {'Days':>5} "
      f"{'SpikeL':>7} {'SpikeS':>7} {'RateL%':>7} {'RateS%':>7} "
      f"{'MFE60_L':>8} {'MFE60_S':>8} {'PM3%+':>6} {'AvgPM':>6}")
print(f"  {'─'*100}")

for _, row in df_summary.iterrows():
    print(f"  {row['Ticker']:<7} {row['Sector']:<12} {row['Deployed']:>3} "
          f"{row['Days']:>5} {row['SpikeL']:>7} {row['SpikeS']:>7} "
          f"{row['RateL']:>6.1f}% {row['RateS']:>6.1f}% "
          f"{row['AvgMFE_L']:>7.2f}% {row['AvgMFE_S']:>7.2f}% "
          f"{row['PM_Rate']:>5.1f}% {row['AvgPM']:>5.2f}%")

# Highlight non-deployed tickers with significant spike rates
print(f"\n  ── NON-DEPLOYED TICKERS WITH SPIKE ACTIVITY ──")
non_dep_active = df_summary[(df_summary['Deployed'] == 'N') & (df_summary['TotalRate'] > 1.0)]
if len(non_dep_active) > 0:
    for _, row in non_dep_active.iterrows():
        print(f"    ** {row['Ticker']} ({row['Sector']}): "
              f"LONG {row['RateL']:.1f}% ({row['SpikeL']}d), "
              f"SHORT {row['RateS']:.1f}% ({row['SpikeS']}d), "
              f"avg MFE L={row['AvgMFE_L']:.2f}% S={row['AvgMFE_S']:.2f}%")
else:
    print(f"    None with >1% total spike rate")


# ==============================================================================
# PHASE 2: SECTOR-LEVEL AGGREGATION & TEMPORAL TRENDS
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  PHASE 2: SECTOR-LEVEL ANALYSIS")
print(f"{'='*W}")

# 2a: Sector summary
print(f"\n  ── Sector Summary ──\n")
sectors = sorted(df_daily['Sector'].unique())

sector_stats = []
for sector in sectors:
    s_df = df_daily[df_daily['Sector'] == sector]
    n_tickers = s_df['Ticker'].nunique()
    n_days = len(s_df)
    n_deployed = s_df[s_df['Deployed']]['Ticker'].nunique()
    n_spike_l = s_df['Spike_Long'].sum()
    n_spike_s = s_df['Spike_Short'].sum()
    n_pm_spike = s_df['PM_Spike_3pct'].sum()
    rate_l = n_spike_l / n_days * 100 if n_days > 0 else 0
    rate_s = n_spike_s / n_days * 100 if n_days > 0 else 0
    pm_rate = n_pm_spike / n_days * 100 if n_days > 0 else 0
    avg_mfe_l = s_df['MFE_Long'].mean()
    avg_mfe_s = s_df['MFE_Short'].mean()
    avg_pm_mag = s_df['PM_Spike_Mag'].mean()
    avg_pm_vol = s_df['PM_Volume'].mean()
    sector_stats.append({
        'Sector': sector, 'Tickers': n_tickers, 'Deployed': n_deployed,
        'TkDays': n_days, 'SpikeL': int(n_spike_l), 'SpikeS': int(n_spike_s),
        'RateL': rate_l, 'RateS': rate_s,
        'PM_Rate': pm_rate, 'AvgPM_Mag': avg_pm_mag,
        'AvgMFE_L': avg_mfe_l, 'AvgMFE_S': avg_mfe_s,
        'AvgPM_Vol': avg_pm_vol,
    })

df_sector = pd.DataFrame(sector_stats).sort_values('RateL', ascending=False)

print(f"  NOTE: MFE = 60-min window. PM = pre-market spike (360-comparable).\n")
print(f"  {'Sector':<12} {'Tickers':>8} {'Deploy':>7} {'TkDays':>7} "
      f"{'RateL%':>7} {'RateS%':>7} {'MFE60_L':>8} {'MFE60_S':>8} "
      f"{'PM3%+':>7} {'AvgPM':>7}")
print(f"  {'─'*95}")
for _, row in df_sector.iterrows():
    coverage = f"{row['Deployed']}/{row['Tickers']}"
    print(f"  {row['Sector']:<12} {row['Tickers']:>8} {coverage:>7} "
          f"{row['TkDays']:>7} "
          f"{row['RateL']:>6.1f}% {row['RateS']:>6.1f}% "
          f"{row['AvgMFE_L']:>7.2f}% {row['AvgMFE_S']:>7.2f}% "
          f"{row['PM_Rate']:>6.1f}% {row['AvgPM_Mag']:>6.2f}%")


# 2b: Sector temporal trends — weekly (60m MFE spike rate)
print(f"\n  ── Sector Weekly 60m MFE Spike Rate Trends (3%+ threshold) ──\n")

weeks = sorted(df_daily['Year_Week'].unique())

print(f"  {'Sector':<12}", end="")
for wk in weeks:
    print(f"  {wk:>8}", end="")
print()
print(f"  {'─'*(12 + 10*len(weeks))}")

sector_weekly = {}
for sector in sectors:
    s_df = df_daily[df_daily['Sector'] == sector]
    weekly_rates = []
    for wk in weeks:
        wk_df = s_df[s_df['Year_Week'] == wk]
        if len(wk_df) == 0:
            weekly_rates.append(0)
            continue
        rate = (wk_df['Spike_Long'].sum() + wk_df['Spike_Short'].sum()) / len(wk_df) * 100
        weekly_rates.append(rate)
    sector_weekly[sector] = weekly_rates

    print(f"  {sector:<12}", end="")
    for r in weekly_rates:
        marker = '*' if r >= 5 else ' '
        print(f"  {r:>6.1f}%{marker}", end="")
    print()

# Also show PM spike rate weekly (360-comparable)
print(f"\n  ── Sector Weekly PM Spike Rate (3%+ pre-market, 360-comparable) ──\n")

print(f"  {'Sector':<12}", end="")
for wk in weeks:
    print(f"  {wk:>8}", end="")
print()
print(f"  {'─'*(12 + 10*len(weeks))}")

for sector in sectors:
    s_df = df_daily[df_daily['Sector'] == sector]
    print(f"  {sector:<12}", end="")
    for wk in weeks:
        wk_df = s_df[s_df['Year_Week'] == wk]
        if len(wk_df) == 0:
            print(f"  {'0.0%':>8}", end="")
            continue
        rate = wk_df['PM_Spike_3pct'].sum() / len(wk_df) * 100
        marker = '*' if rate >= 5 else ' '
        print(f"  {rate:>6.1f}%{marker}", end="")
    print()

# 2c: Sector MFE trends — weekly
print(f"\n  ── Sector Weekly Avg MFE (LONG direction) ──\n")

print(f"  {'Sector':<12}", end="")
for wk in weeks:
    print(f"  {wk:>8}", end="")
print()
print(f"  {'─'*(12 + 10*len(weeks))}")

sector_weekly_mfe = {}
for sector in sectors:
    s_df = df_daily[df_daily['Sector'] == sector]
    weekly_mfe = []
    for wk in weeks:
        wk_df = s_df[s_df['Year_Week'] == wk]
        if len(wk_df) == 0:
            weekly_mfe.append(0)
            continue
        weekly_mfe.append(wk_df['MFE_Long'].mean())
    sector_weekly_mfe[sector] = weekly_mfe

    print(f"  {sector:<12}", end="")
    for m in weekly_mfe:
        print(f"  {m:>7.2f}%", end="")
    print()


# 2d: Deployed vs non-deployed spike rates
print(f"\n  ── Deployed vs Non-Deployed Spike Rates ──\n")

for deployed_flag, label in [(True, 'DEPLOYED'), (False, 'NOT DEPLOYED')]:
    sub = df_daily[df_daily['Deployed'] == deployed_flag]
    if len(sub) == 0:
        print(f"  {label}: no data")
        continue
    n_tk = sub['Ticker'].nunique()
    n_days_total = len(sub)
    n_sp_l = sub['Spike_Long'].sum()
    n_sp_s = sub['Spike_Short'].sum()
    n_pm = sub['PM_Spike_3pct'].sum()
    rate_l = n_sp_l / n_days_total * 100
    rate_s = n_sp_s / n_days_total * 100
    pm_rate = n_pm / n_days_total * 100
    avg_mfe_l = sub['MFE_Long'].mean()
    avg_mfe_s = sub['MFE_Short'].mean()
    avg_pm = sub['PM_Spike_Mag'].mean()
    print(f"  {label}: {n_tk} tickers, {n_days_total} ticker-days")
    print(f"    60m MFE LONG:  spike rate {rate_l:.1f}% ({int(n_sp_l)} spikes), avg MFE {avg_mfe_l:.2f}%")
    print(f"    60m MFE SHORT: spike rate {rate_s:.1f}% ({int(n_sp_s)} spikes), avg MFE {avg_mfe_s:.2f}%")
    print(f"    PM spike (360): rate {pm_rate:.1f}% ({int(n_pm)} events), avg magnitude {avg_pm:.2f}%")
    print()


# ==============================================================================
# PHASE 2e: 360 DIAGNOSIS COMPARISON — Pre-market spike frequency
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  PHASE 2e: 360 DIAGNOSIS COMPARISON (Pre-Market Spike Methodology)")
print(f"{'='*W}")
print(f"\n  360 diagnosis (Sep 2) found: 3%+ PM spike frequency 6.6% → 0.9% (-86%)")
print(f"  This section uses the SAME metric: pre-market spike magnitude ≥ 3%\n")

# Per-period PM spike frequency (matching 360 periods)
pm_periods = [
    ('Jul 1-20',  '2026-07-01', '2026-07-20'),
    ('Jul 21-Aug10', '2026-07-21', '2026-08-10'),
    ('Aug 11-24', '2026-08-11', '2026-08-24'),
    ('Aug 25-Sep7', '2026-08-25', '2026-09-07'),
]

print(f"  ── Pre-Market 3%+ Spike Frequency by Period ──\n")
print(f"  {'Period':<15} {'TkDays':>7} {'PM3%+':>6} {'Freq%':>7} "
      f"{'AvgPM_Mag':>10} {'MedPM_Mag':>10}")
print(f"  {'─'*65}")

for label, p_start, p_end in pm_periods:
    p_df = df_daily[(df_daily['Date'] >= p_start) & (df_daily['Date'] <= p_end)]
    if len(p_df) == 0:
        continue
    n_pm3 = p_df['PM_Spike_3pct'].sum()
    freq = n_pm3 / len(p_df) * 100
    avg_mag = p_df['PM_Spike_Mag'].mean()
    med_mag = p_df['PM_Spike_Mag'].median()
    print(f"  {label:<15} {len(p_df):>7} {n_pm3:>6.0f} {freq:>6.1f}% "
          f"{avg_mag:>9.2f}% {med_mag:>9.2f}%")

# By sector × period
print(f"\n  ── PM 3%+ Spike Frequency by Sector × Period ──\n")
print(f"  {'Sector':<12}", end="")
for label, _, _ in pm_periods:
    print(f"  {label:>14}", end="")
print(f"  {'Change':>8}")
print(f"  {'─'*(12 + 16*len(pm_periods) + 10)}")

for sector in sorted(sectors):
    s_df = df_daily[df_daily['Sector'] == sector]
    rates = []
    for label, p_start, p_end in pm_periods:
        p_df = s_df[(s_df['Date'] >= p_start) & (s_df['Date'] <= p_end)]
        if len(p_df) == 0:
            rates.append(0)
        else:
            rates.append(p_df['PM_Spike_3pct'].sum() / len(p_df) * 100)

    change = ''
    if len(rates) >= 2 and rates[0] > 0:
        delta = rates[-1] - rates[0]
        pct_change = delta / rates[0] * 100
        change = f"{pct_change:+.0f}%"
    elif rates[0] == 0 and rates[-1] > 0:
        change = 'NEW'
    elif rates[-1] == 0:
        change = 'DEAD'

    print(f"  {sector:<12}", end="")
    for r in rates:
        print(f"  {r:>12.1f}%", end="")
    print(f"  {change:>8}")

# Weekly PM spike trend
print(f"\n  ── Weekly Pre-Market 3%+ Spike Frequency (all tickers) ──\n")
print(f"  {'Week':<10} {'TkDays':>7} {'PM3%+':>6} {'Freq%':>7} "
      f"{'AvgPM_Mag':>10} {'60m_MFE_L':>10} {'60m_MFE_S':>10}")
print(f"  {'─'*70}")

for wk in weeks:
    wk_df = df_daily[df_daily['Year_Week'] == wk]
    if len(wk_df) == 0:
        continue
    n_pm3 = wk_df['PM_Spike_3pct'].sum()
    freq = n_pm3 / len(wk_df) * 100
    avg_mag = wk_df['PM_Spike_Mag'].mean()
    avg_mfe_l = wk_df['MFE_Long'].mean()
    avg_mfe_s = wk_df['MFE_Short'].mean()
    print(f"  {wk:<10} {len(wk_df):>7} {n_pm3:>6.0f} {freq:>6.1f}% "
          f"{avg_mag:>9.2f}% {avg_mfe_l:>9.2f}% {avg_mfe_s:>9.2f}%")


# ==============================================================================
# PHASE 3: COVERAGE GAP ANALYSIS — temporal
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  PHASE 3: COVERAGE GAP ANALYSIS — WHERE ARE SPIKES WE'RE MISSING?")
print(f"{'='*W}")

# Recent period: last 2 weeks
recent_cutoff = pd.Timestamp(ANALYSIS_END) - pd.Timedelta(days=14)
df_recent = df_daily[df_daily['Date_dt'] >= recent_cutoff]

print(f"\n  Analyzing last 14 days: {recent_cutoff.date()} → {ANALYSIS_END}")
print(f"  Total ticker-days in period: {len(df_recent)}")

# 3a: Recent spike events by deployment status
recent_spikes_l = df_recent[df_recent['Spike_Long']]
recent_spikes_s = df_recent[df_recent['Spike_Short']]

print(f"\n  ── Recent 3%+ LONG Spikes ──")
print(f"  {'Date':<12} {'Ticker':<7} {'Sector':<12} {'Deployed':>8} "
      f"{'MFE':>7} {'EOD':>7} {'PM_Vol':>10}")
print(f"  {'─'*75}")
for _, row in recent_spikes_l.sort_values(['Date', 'MFE_Long'], ascending=[True, False]).iterrows():
    dep = 'Y' if row['Deployed'] else '** N **'
    print(f"  {row['Date']:<12} {row['Ticker']:<7} {row['Sector']:<12} {dep:>8} "
          f"{row['MFE_Long']:>6.1f}% {row['EOD_Return']:>+6.1f}% "
          f"{row['PM_Volume']:>10,.0f}")

print(f"\n  ── Recent 3%+ SHORT Spikes ──")
print(f"  {'Date':<12} {'Ticker':<7} {'Sector':<12} {'Deployed':>8} "
      f"{'MFE':>7} {'EOD':>7} {'PM_Vol':>10}")
print(f"  {'─'*75}")
for _, row in recent_spikes_s.sort_values(['Date', 'MFE_Short'], ascending=[True, False]).iterrows():
    dep = 'Y' if row['Deployed'] else '** N **'
    print(f"  {row['Date']:<12} {row['Ticker']:<7} {row['Sector']:<12} {dep:>8} "
          f"{row['MFE_Short']:>6.1f}% {row['EOD_Return']:>+6.1f}% "
          f"{row['PM_Volume']:>10,.0f}")

# 3b: Missed opportunity summary
missed_l = recent_spikes_l[~recent_spikes_l['Deployed']]
missed_s = recent_spikes_s[~recent_spikes_s['Deployed']]
total_spikes_l = len(recent_spikes_l)
total_spikes_s = len(recent_spikes_s)

print(f"\n  ── Coverage Summary (Last 14 Days) ──")
print(f"  LONG 3%+ spikes:  {total_spikes_l} total, {len(missed_l)} missed "
      f"({len(missed_l)/max(total_spikes_l,1)*100:.0f}% outside coverage)")
print(f"  SHORT 3%+ spikes: {total_spikes_s} total, {len(missed_s)} missed "
      f"({len(missed_s)/max(total_spikes_s,1)*100:.0f}% outside coverage)")

if len(missed_l) > 0:
    print(f"\n  Missed LONG tickers: {', '.join(sorted(missed_l['Ticker'].unique()))}")
    for tk in sorted(missed_l['Ticker'].unique()):
        tk_data = missed_l[missed_l['Ticker'] == tk]
        print(f"    {tk} ({TICKER_SECTOR.get(tk, 'Unknown')}): "
              f"{len(tk_data)} spike days, avg MFE {tk_data['MFE_Long'].mean():.1f}%")

if len(missed_s) > 0:
    print(f"\n  Missed SHORT tickers: {', '.join(sorted(missed_s['Ticker'].unique()))}")
    for tk in sorted(missed_s['Ticker'].unique()):
        tk_data = missed_s[missed_s['Ticker'] == tk]
        print(f"    {tk} ({TICKER_SECTOR.get(tk, 'Unknown')}): "
              f"{len(tk_data)} spike days, avg MFE {tk_data['MFE_Short'].mean():.1f}%")


# 3c: Time-series of coverage gaps
print(f"\n  ── Weekly Coverage Gap Trend ──")
print(f"  {'Week':<10} {'TotalSp':>8} {'Deploy':>7} {'Missed':>7} {'Gap%':>6}")
print(f"  {'─'*45}")

for wk in weeks:
    wk_df = df_daily[df_daily['Year_Week'] == wk]
    wk_spikes = wk_df['Spike_Long'] | wk_df['Spike_Short']
    n_total = wk_spikes.sum()
    n_deploy = (wk_spikes & wk_df['Deployed']).sum()
    n_missed = n_total - n_deploy
    gap = n_missed / max(n_total, 1) * 100
    print(f"  {wk:<10} {n_total:>8.0f} {n_deploy:>7.0f} {n_missed:>7.0f} {gap:>5.0f}%")


# ==============================================================================
# PHASE 4: MODEL-SECTOR SPECIALIZATION ANALYSIS (BQ)
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  PHASE 4: MODEL-SECTOR SPECIALIZATION (BQ SCORING LOGS)")
print(f"{'='*W}")

# Pull live prediction data
q_live = f"""
SELECT
  ticker, score_date, signal, prob, threshold,
  strategy, model_version
FROM `{BQ_LIVE_TABLE}`
WHERE score_date BETWEEN '{ANALYSIS_START}' AND '{ANALYSIS_END}'
  AND strategy LIKE 'ST_%'
ORDER BY score_date, ticker
"""

print(f"\n  Querying BQ live predictions...")
try:
    df_live = bq_client.query(q_live).to_dataframe()
    print(f"    Loaded {len(df_live)} live prediction rows")
    print(f"    Strategies: {sorted(df_live['strategy'].unique())}")
    print(f"    Date range: {df_live['score_date'].min()} → {df_live['score_date'].max()}")

    df_live['Sector'] = df_live['ticker'].map(TICKER_SECTOR).fillna('Unknown')
    df_live['Direction'] = df_live['strategy'].apply(
        lambda s: 'LONG' if 'LONG' in s else 'SHORT' if 'SHORT' in s else 'UNK')
    df_live['prob_margin'] = df_live['prob'] - df_live['threshold']
    df_live['is_go'] = df_live['signal'] == 'GO'
    df_live['score_date'] = pd.to_datetime(df_live['score_date'])

    # 4a: GO rate by sector
    print(f"\n  ── GO Signal Rate by Sector & Direction ──\n")
    print(f"  {'Sector':<12} {'Dir':<6} {'Total':>7} {'GO':>5} {'GO%':>6} "
          f"{'AvgProb':>8} {'AvgMargin':>10}")
    print(f"  {'─'*65}")

    for direction in ['LONG', 'SHORT']:
        dir_df = df_live[df_live['Direction'] == direction]
        for sector in sorted(dir_df['Sector'].unique()):
            sec_df = dir_df[dir_df['Sector'] == sector]
            n_total = len(sec_df)
            n_go = sec_df['is_go'].sum()
            go_rate = n_go / n_total * 100 if n_total > 0 else 0
            avg_prob = sec_df['prob'].mean()
            avg_margin = sec_df['prob_margin'].mean()
            print(f"  {sector:<12} {direction:<6} {n_total:>7} {n_go:>5} "
                  f"{go_rate:>5.1f}% {avg_prob:>7.3f} {avg_margin:>+9.3f}")
        print()

    # 4b: GO rate temporal trend by sector
    print(f"\n  ── Weekly GO Rate by Sector (LONG) ──\n")
    df_live['Year_Week'] = (df_live['score_date'].dt.isocalendar().year.astype(str) + '-W' +
                             df_live['score_date'].dt.isocalendar().week.astype(str).str.zfill(2))
    live_weeks = sorted(df_live['Year_Week'].unique())

    print(f"  {'Sector':<12}", end="")
    for wk in live_weeks:
        print(f"  {wk:>8}", end="")
    print()
    print(f"  {'─'*(12 + 10*len(live_weeks))}")

    for sector in sorted(df_live['Sector'].unique()):
        long_sec = df_live[(df_live['Direction'] == 'LONG') & (df_live['Sector'] == sector)]
        print(f"  {sector:<12}", end="")
        for wk in live_weeks:
            wk_df = long_sec[long_sec['Year_Week'] == wk]
            if len(wk_df) == 0:
                print(f"  {'—':>8}", end="")
            else:
                rate = wk_df['is_go'].mean() * 100
                print(f"  {rate:>6.0f}%", end="")
        print()

    # 4c: Model family performance by sector
    print(f"\n  ── Model Family × Sector — Average Prob Margin for GO Signals ──\n")

    go_signals = df_live[df_live['is_go']]
    go_signals = go_signals.copy()
    go_signals['family'] = go_signals['model_version'].apply(
        lambda v: 'MR-v2-ADV' if 'adv' in str(v).lower() else
                  'MR-v2' if 'multires' in str(v).lower() else
                  'ASL6' if 'asl6' in str(v).lower() or 'l5_2' in str(v).lower() else
                  'L5-1' if 'l5_1' in str(v).lower() else 'Other')

    families = sorted(go_signals['family'].unique())
    print(f"  {'Sector':<12}", end="")
    for fam in families:
        print(f"  {fam:>12}", end="")
    print()
    print(f"  {'─'*(12 + 14*len(families))}")

    for sector in sorted(go_signals['Sector'].unique()):
        print(f"  {sector:<12}", end="")
        for fam in families:
            sub = go_signals[(go_signals['Sector'] == sector) & (go_signals['family'] == fam)]
            if len(sub) < 3:
                print(f"  {'—':>12}", end="")
            else:
                avg_m = sub['prob_margin'].mean()
                print(f"  {avg_m:>+11.3f}", end="")
        print()

except Exception as e:
    print(f"  BQ query failed: {e}")
    print(f"  Skipping Phase 4 — continuing with GCS-only analysis")


# ==============================================================================
# PHASE 5: SIGNAL MIGRATION DETECTION
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  PHASE 5: SIGNAL MIGRATION — WHERE IS THE EDGE MOVING?")
print(f"{'='*W}")

# Split into 3 periods: early July, late July/early Aug, late Aug/Sep
periods = [
    ('Early Jul', '2026-07-01', '2026-07-20'),
    ('Late Jul', '2026-07-21', '2026-08-10'),
    ('Early Aug', '2026-08-11', '2026-08-24'),
    ('Late Aug+', '2026-08-25', '2026-09-07'),
]

print(f"\n  ── Per-Period Spike Rates by Sector ──\n")
print(f"  {'Sector':<12}", end="")
for label, _, _ in periods:
    print(f"  {label:>12}", end="")
print(f"  {'Trend':>8}")
print(f"  {'─'*(12 + 14*len(periods) + 10)}")

sector_period_rates = {}
for sector in sectors:
    s_df = df_daily[df_daily['Sector'] == sector]
    rates = []
    for label, p_start, p_end in periods:
        p_df = s_df[(s_df['Date'] >= p_start) & (s_df['Date'] <= p_end)]
        if len(p_df) == 0:
            rates.append(0)
        else:
            rate = (p_df['Spike_Long'].sum() + p_df['Spike_Short'].sum()) / len(p_df) * 100
            rates.append(rate)
    sector_period_rates[sector] = rates

    if len(rates) >= 2 and rates[0] > 0:
        change = rates[-1] - rates[0]
        if change < -2:
            trend = '↓↓'
        elif change < 0:
            trend = '↓'
        elif change > 2:
            trend = '↑↑'
        elif change > 0:
            trend = '↑'
        else:
            trend = '→'
    else:
        trend = '—'

    print(f"  {sector:<12}", end="")
    for r in rates:
        print(f"  {r:>10.1f}%", end="")
    print(f"  {trend:>8}")


# 5b: Per-ticker MFE trend (early vs late)
print(f"\n  ── Per-Ticker MFE Trend: Early (Jul 1-Aug 10) vs Late (Aug 25-Sep 7) ──\n")
early_end = '2026-08-10'
late_start = '2026-08-25'

print(f"  {'Ticker':<7} {'Sector':<12} {'Dep':>3} "
      f"{'Early MFE_L':>11} {'Late MFE_L':>11} {'Delta':>7} "
      f"{'Early MFE_S':>11} {'Late MFE_S':>11} {'Delta':>7}")
print(f"  {'─'*90}")

ticker_trends = []
for tk in sorted(df_daily['Ticker'].unique()):
    tk_df = df_daily[df_daily['Ticker'] == tk]
    early = tk_df[tk_df['Date'] <= early_end]
    late = tk_df[tk_df['Date'] >= late_start]
    if len(early) < 3 or len(late) < 3:
        continue
    e_mfe_l = early['MFE_Long'].mean()
    l_mfe_l = late['MFE_Long'].mean()
    e_mfe_s = early['MFE_Short'].mean()
    l_mfe_s = late['MFE_Short'].mean()
    delta_l = l_mfe_l - e_mfe_l
    delta_s = l_mfe_s - e_mfe_s
    dep = 'Y' if tk in ALL_DEPLOY else 'N'
    sector = TICKER_SECTOR.get(tk, 'Unknown')

    ticker_trends.append({
        'Ticker': tk, 'Sector': sector, 'Deployed': dep,
        'Early_L': e_mfe_l, 'Late_L': l_mfe_l, 'Delta_L': delta_l,
        'Early_S': e_mfe_s, 'Late_S': l_mfe_s, 'Delta_S': delta_s,
    })

    print(f"  {tk:<7} {sector:<12} {dep:>3} "
          f"{e_mfe_l:>10.2f}% {l_mfe_l:>10.2f}% {delta_l:>+6.2f}% "
          f"{e_mfe_s:>10.2f}% {l_mfe_s:>10.2f}% {delta_s:>+6.2f}%")

df_trends = pd.DataFrame(ticker_trends)

# 5c: Tickers that IMPROVED (signal migrated TO)
print(f"\n  ── Tickers WHERE EDGE GREW (MFE increased late vs early) ──")
improving_l = df_trends[df_trends['Delta_L'] > 0.5].sort_values('Delta_L', ascending=False)
if len(improving_l) > 0:
    print(f"  LONG MFE increased:")
    for _, row in improving_l.iterrows():
        print(f"    {row['Ticker']} ({row['Sector']}, dep={row['Deployed']}): "
              f"{row['Early_L']:.1f}% → {row['Late_L']:.1f}% (+{row['Delta_L']:.1f}%)")
else:
    print(f"  No tickers with LONG MFE improvement > 0.5%")

improving_s = df_trends[df_trends['Delta_S'] > 0.5].sort_values('Delta_S', ascending=False)
if len(improving_s) > 0:
    print(f"\n  SHORT MFE increased:")
    for _, row in improving_s.iterrows():
        print(f"    {row['Ticker']} ({row['Sector']}, dep={row['Deployed']}): "
              f"{row['Early_S']:.1f}% → {row['Late_S']:.1f}% (+{row['Delta_S']:.1f}%)")
else:
    print(f"  No tickers with SHORT MFE improvement > 0.5%")

# 5d: Tickers that DEGRADED the most
print(f"\n  ── Tickers WHERE EDGE COLLAPSED (MFE decreased) ──")
collapsing_l = df_trends[(df_trends['Delta_L'] < -1.0) & (df_trends['Deployed'] == 'Y')].sort_values('Delta_L')
if len(collapsing_l) > 0:
    print(f"  DEPLOYED tickers with LONG MFE collapse > 1%:")
    for _, row in collapsing_l.iterrows():
        print(f"    {row['Ticker']} ({row['Sector']}): "
              f"{row['Early_L']:.1f}% → {row['Late_L']:.1f}% ({row['Delta_L']:+.1f}%)")

collapsing_s = df_trends[(df_trends['Delta_S'] < -1.0) & (df_trends['Deployed'] == 'Y')].sort_values('Delta_S')
if len(collapsing_s) > 0:
    print(f"\n  DEPLOYED tickers with SHORT MFE collapse > 1%:")
    for _, row in collapsing_s.iterrows():
        print(f"    {row['Ticker']} ({row['Sector']}): "
              f"{row['Early_S']:.1f}% → {row['Late_S']:.1f}% ({row['Delta_S']:+.1f}%)")


# ==============================================================================
# PHASE 6: SPECIALIZATION OPPORTUNITY ANALYSIS
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  PHASE 6: SECTOR SPECIALIZATION OPPORTUNITY")
print(f"  Should we train sector-specific models?")
print(f"{'='*W}")

# 6a: Within-sector vs cross-sector signal consistency
print(f"\n  ── Sector Internal Correlation (do tickers within a sector spike together?) ──\n")

for sector in sectors:
    s_tickers = df_daily[df_daily['Sector'] == sector]['Ticker'].unique()
    if len(s_tickers) < 3:
        continue

    pivot_l = df_daily[df_daily['Sector'] == sector].pivot_table(
        index='Date', columns='Ticker', values='MFE_Long', aggfunc='mean')
    if pivot_l.shape[1] < 3:
        continue

    corr_matrix = pivot_l.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    off_diag = corr_matrix.values[mask]
    off_diag = off_diag[~np.isnan(off_diag)]

    if len(off_diag) == 0:
        continue

    avg_corr = np.mean(off_diag)
    med_corr = np.median(off_diag)
    min_corr = np.min(off_diag)
    max_corr = np.max(off_diag)

    verdict = "HIGH" if avg_corr > 0.3 else "MODERATE" if avg_corr > 0.15 else "LOW"
    spec_value = ("YES — shared signal" if avg_corr > 0.3
                  else "MAYBE — partial coherence" if avg_corr > 0.15
                  else "NO — tickers spike independently")

    print(f"  {sector} ({len(s_tickers)} tickers): "
          f"avg ρ={avg_corr:.3f}, med={med_corr:.3f}, "
          f"range=[{min_corr:.3f}, {max_corr:.3f}] → {verdict}")
    print(f"    Specialization value: {spec_value}")

# 6b: Cross-sector correlations
print(f"\n  ── Cross-Sector MFE Correlation ──\n")

sector_daily_mfe = {}
for sector in sectors:
    s_df = df_daily[df_daily['Sector'] == sector]
    daily_avg = s_df.groupby('Date')['MFE_Long'].mean()
    sector_daily_mfe[sector] = daily_avg

sector_mfe_df = pd.DataFrame(sector_daily_mfe)
if sector_mfe_df.shape[1] >= 2:
    sector_corr = sector_mfe_df.corr()
    print(f"  {'':>12}", end="")
    for s in sector_corr.columns:
        print(f"  {s:>10}", end="")
    print()
    for s1 in sector_corr.index:
        print(f"  {s1:<12}", end="")
        for s2 in sector_corr.columns:
            v = sector_corr.loc[s1, s2]
            print(f"  {v:>10.3f}", end="")
        print()

# 6c: Recent spikes — which sectors still have tradeable events?
print(f"\n  ── Sector Health Scorecard (Last 14 Days) ──\n")

print(f"  {'Sector':<12} {'TkDays':>7} {'3%+Sp':>6} {'5%+Sp':>6} "
      f"{'AvgMFE_L':>9} {'AvgMFE_S':>9} {'Deploy%':>8} {'Verdict':>12}")
print(f"  {'─'*85}")

for sector in sorted(sectors):
    r_df = df_recent[df_recent['Sector'] == sector]
    if len(r_df) == 0:
        continue
    n_days = len(r_df)
    n_sp3 = (r_df['Spike_Long'] | r_df['Spike_Short']).sum()
    n_sp5 = (r_df['Spike_Long_5'] | r_df['Spike_Short_5']).sum()
    avg_l = r_df['MFE_Long'].mean()
    avg_s = r_df['MFE_Short'].mean()
    n_dep = r_df[r_df['Deployed']]['Ticker'].nunique()
    n_total = r_df['Ticker'].nunique()
    dep_pct = n_dep / n_total * 100 if n_total > 0 else 0

    if n_sp3 / n_days * 100 >= 5:
        verdict = "ACTIVE"
    elif n_sp3 / n_days * 100 >= 2:
        verdict = "MODERATE"
    elif n_sp3 > 0:
        verdict = "LOW"
    else:
        verdict = "DEAD"

    print(f"  {sector:<12} {n_days:>7} {n_sp3:>6.0f} {n_sp5:>6.0f} "
          f"{avg_l:>8.2f}% {avg_s:>8.2f}% {dep_pct:>7.0f}% {verdict:>12}")


# ==============================================================================
# CHARTS
# ==============================================================================
print(f"\n\n  Generating charts...")

fig, axes = plt.subplots(3, 3, figsize=(28, 24))
fig.suptitle(f'Asset-by-Asset & Sector Performance Decomposition\n'
             f'{ANALYSIS_START} → {ANALYSIS_END} | {df_daily["Ticker"].nunique()} tickers',
             fontsize=16, fontweight='bold')

# Chart 1: Sector spike rates over time (heatmap-style)
ax = axes[0, 0]
sector_list = sorted(sectors)
n_weeks = len(weeks)
n_sectors = len(sector_list)
heatmap_data = np.zeros((n_sectors, n_weeks))
for i, sector in enumerate(sector_list):
    for j, rate in enumerate(sector_weekly.get(sector, [0]*n_weeks)):
        heatmap_data[i, j] = rate
im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto', vmin=0,
               vmax=max(15, heatmap_data.max()))
ax.set_xticks(range(n_weeks))
ax.set_xticklabels([w.split('-W')[1] for w in weeks], rotation=45, fontsize=7)
ax.set_yticks(range(n_sectors))
ax.set_yticklabels(sector_list, fontsize=8)
for i in range(n_sectors):
    for j in range(n_weeks):
        v = heatmap_data[i, j]
        if v > 0.5:
            ax.text(j, i, f"{v:.0f}%", ha='center', va='center', fontsize=6,
                    color='white' if v > 8 else 'black')
plt.colorbar(im, ax=ax, shrink=0.8, label='Spike Rate %')
ax.set_title('Weekly 60m MFE Spike Rate by Sector\n(3%+ combined LONG+SHORT, 60-min window)')
ax.set_xlabel('Week Number')

# Chart 2: Sector MFE trends (line chart)
ax = axes[0, 1]
week_dates = pd.to_datetime([f"2026-W{int(w.split('-W')[1])}-1" for w in weeks],
                             format='%Y-W%W-%w')
for sector in sector_list:
    rates = sector_weekly.get(sector, [0]*n_weeks)
    if max(rates) > 1:
        color = SECTOR_COLORS.get(sector, '#333')
        ax.plot(week_dates, rates, 'o-', color=color, linewidth=2,
                markersize=5, label=sector, alpha=0.8)
ax.axhline(3, color='red', linewidth=1, linestyle='--', alpha=0.5, label='3% threshold')
ax.set_ylabel('Spike Rate (%)')
ax.set_title('Sector Spike Rate Trends')
ax.legend(fontsize=7, ncol=2)
ax.grid(alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Chart 3: Deployed vs Non-deployed spike capture
ax = axes[0, 2]
dep_weekly_rates = []
nondep_weekly_rates = []
for wk in weeks:
    wk_df = df_daily[df_daily['Year_Week'] == wk]
    dep = wk_df[wk_df['Deployed']]
    nondep = wk_df[~wk_df['Deployed']]
    dep_rate = ((dep['Spike_Long'].sum() + dep['Spike_Short'].sum()) /
                max(len(dep), 1) * 100) if len(dep) > 0 else 0
    nondep_rate = ((nondep['Spike_Long'].sum() + nondep['Spike_Short'].sum()) /
                   max(len(nondep), 1) * 100) if len(nondep) > 0 else 0
    dep_weekly_rates.append(dep_rate)
    nondep_weekly_rates.append(nondep_rate)

x_pos = np.arange(n_weeks)
bar_w = 0.35
ax.bar(x_pos - bar_w/2, dep_weekly_rates, bar_w, label='Deployed',
       color='#1565C0', alpha=0.8)
ax.bar(x_pos + bar_w/2, nondep_weekly_rates, bar_w, label='Not Deployed',
       color='#E65100', alpha=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels([w.split('-W')[1] for w in weeks], rotation=45, fontsize=7)
ax.set_ylabel('Spike Rate (%)')
ax.set_title('Deployed vs Non-Deployed Spike Rates')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

# Chart 4: Per-ticker MFE scatter (early vs late)
ax = axes[1, 0]
if len(df_trends) > 0:
    for sector in sorted(df_trends['Sector'].unique()):
        sub = df_trends[df_trends['Sector'] == sector]
        color = SECTOR_COLORS.get(sector, '#333')
        marker = 'o' if sub.iloc[0]['Deployed'] == 'Y' else '^'
        dep = sub[sub['Deployed'] == 'Y']
        ndep = sub[sub['Deployed'] == 'N']
        if len(dep) > 0:
            ax.scatter(dep['Early_L'], dep['Late_L'], color=color,
                      marker='o', s=60, alpha=0.7, label=f'{sector} (dep)')
        if len(ndep) > 0:
            ax.scatter(ndep['Early_L'], ndep['Late_L'], color=color,
                      marker='^', s=60, alpha=0.7, label=f'{sector} (scan)')
    lims = [0, max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, 'k--', alpha=0.3, linewidth=1)
    ax.set_xlabel('Early Period Avg MFE LONG (%)')
    ax.set_ylabel('Late Period Avg MFE LONG (%)')
    ax.set_title('60m MFE Persistence: Early vs Late\n(below diagonal = degraded)')
    ax.legend(fontsize=5, ncol=2, loc='upper left')
    ax.grid(alpha=0.3)

# Chart 5: Sector period comparison (grouped bars)
ax = axes[1, 1]
if len(periods) >= 2:
    n_sectors_plot = min(8, len(sector_list))
    top_sectors = sorted(sector_list,
                         key=lambda s: max(sector_period_rates.get(s, [0])),
                         reverse=True)[:n_sectors_plot]
    x = np.arange(n_sectors_plot)
    width = 0.8 / len(periods)
    period_colors = ['#1565C0', '#2E7D32', '#FF6F00', '#C62828']
    for pi, (label, _, _) in enumerate(periods):
        vals = [sector_period_rates.get(s, [0]*len(periods))[pi] for s in top_sectors]
        ax.bar(x + pi * width - 0.4 + width/2, vals, width,
               label=label, color=period_colors[pi % len(period_colors)], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(top_sectors, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Spike Rate (%)')
    ax.set_title('Sector Spike Rates by Period')
    ax.legend(fontsize=7)
    ax.grid(axis='y', alpha=0.3)

# Chart 6: Coverage gap over time
ax = axes[1, 2]
gap_data = []
for wk in weeks:
    wk_df = df_daily[df_daily['Year_Week'] == wk]
    wk_spikes = wk_df['Spike_Long'] | wk_df['Spike_Short']
    n_total = wk_spikes.sum()
    n_deploy = (wk_spikes & wk_df['Deployed']).sum()
    n_missed = n_total - n_deploy
    gap_data.append({'week': wk, 'total': n_total, 'deployed': n_deploy, 'missed': n_missed})

gap_df = pd.DataFrame(gap_data)
ax.bar(x_pos, gap_df['deployed'], label='Deployed (captured)', color='#2E7D32', alpha=0.8)
ax.bar(x_pos, gap_df['missed'], bottom=gap_df['deployed'],
       label='Missed (not deployed)', color='#C62828', alpha=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels([w.split('-W')[1] for w in weeks], rotation=45, fontsize=7)
ax.set_ylabel('Number of 3%+ Spike Events')
ax.set_title('Spike Coverage: Captured vs Missed')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

# Chart 7: Per-ticker spike frequency heatmap (top tickers)
ax = axes[2, 0]
top_spike_tickers = df_summary.head(20)['Ticker'].tolist()
ticker_week_heatmap = np.zeros((len(top_spike_tickers), n_weeks))
for i, tk in enumerate(top_spike_tickers):
    tk_df = df_daily[df_daily['Ticker'] == tk]
    for j, wk in enumerate(weeks):
        wk_data = tk_df[tk_df['Year_Week'] == wk]
        if len(wk_data) > 0:
            ticker_week_heatmap[i, j] = (
                (wk_data['Spike_Long'].sum() + wk_data['Spike_Short'].sum()) /
                len(wk_data) * 100
            )

im2 = ax.imshow(ticker_week_heatmap, cmap='YlOrRd', aspect='auto', vmin=0)
ax.set_xticks(range(n_weeks))
ax.set_xticklabels([w.split('-W')[1] for w in weeks], rotation=45, fontsize=7)
ax.set_yticks(range(len(top_spike_tickers)))
tick_labels = [f"{tk} ({'D' if tk in ALL_DEPLOY else 'S'})" for tk in top_spike_tickers]
ax.set_yticklabels(tick_labels, fontsize=7)
for i in range(len(top_spike_tickers)):
    for j in range(n_weeks):
        v = ticker_week_heatmap[i, j]
        if v > 0:
            ax.text(j, i, f"{v:.0f}", ha='center', va='center', fontsize=5,
                    color='white' if v > 15 else 'black')
plt.colorbar(im2, ax=ax, shrink=0.8)
ax.set_title('Top 20 Tickers — Weekly Spike Rate %\n(D=deployed, S=scan)')
ax.set_xlabel('Week Number')

# Chart 8: Sector health radar / bar chart
ax = axes[2, 1]
if len(df_recent) > 0:
    sector_health = []
    for sector in sorted(sectors):
        r_df = df_recent[df_recent['Sector'] == sector]
        if len(r_df) == 0:
            continue
        rate = (r_df['Spike_Long'].sum() + r_df['Spike_Short'].sum()) / len(r_df) * 100
        avg_mfe = (r_df['MFE_Long'].mean() + r_df['MFE_Short'].mean()) / 2
        sector_health.append({'Sector': sector, 'Rate': rate, 'AvgMFE': avg_mfe})

    sh_df = pd.DataFrame(sector_health).sort_values('Rate', ascending=True)
    y_pos = range(len(sh_df))
    colors = [SECTOR_COLORS.get(s, '#333') for s in sh_df['Sector']]
    ax.barh(list(y_pos), sh_df['Rate'], color=colors, alpha=0.8)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(sh_df['Sector'], fontsize=9)
    for i, (_, row) in enumerate(sh_df.iterrows()):
        ax.text(row['Rate'] + 0.2, i,
                f"MFE={row['AvgMFE']:.1f}%", va='center', fontsize=7)
    ax.axvline(3, color='red', linewidth=1, linestyle='--', alpha=0.5)
    ax.set_xlabel('Spike Rate (%) — Last 14 Days')
    ax.set_title('Sector Health: Current Spike Activity')
    ax.grid(axis='x', alpha=0.3)

# Chart 9: MFE distribution by sector (violin/box)
ax = axes[2, 2]
sector_mfe_data = []
sector_labels_box = []
for sector in sorted(sectors):
    s_df = df_recent[df_recent['Sector'] == sector]
    if len(s_df) < 5:
        continue
    sector_mfe_data.append(s_df['MFE_Long'].values)
    sector_labels_box.append(sector)

if sector_mfe_data:
    bp = ax.boxplot(sector_mfe_data, labels=sector_labels_box,
                    patch_artist=True, showfliers=False)
    for patch, sector in zip(bp['boxes'], sector_labels_box):
        patch.set_facecolor(SECTOR_COLORS.get(sector, '#333'))
        patch.set_alpha(0.6)
    ax.axhline(3, color='red', linewidth=1, linestyle='--', alpha=0.5, label='3% spike')
    ax.set_ylabel('MFE LONG (%)')
    ax.set_title('LONG 60m MFE Distribution by Sector (Last 14 Days)')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.96])
chart_path = '/tmp/asset_sector_decomposition.png'
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
print(f"\n  Main chart saved: {chart_path}")
if _IN_COLAB:
    plt.show()
plt.close()


# ==============================================================================
# EXECUTIVE SUMMARY
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  EXECUTIVE SUMMARY")
print(f"{'='*W}")

# Count recent spikes by sector (60m MFE)
recent_sector_spikes = {}
for sector in sectors:
    r_df = df_recent[df_recent['Sector'] == sector]
    if len(r_df) == 0:
        continue
    n_sp = (r_df['Spike_Long'] | r_df['Spike_Short']).sum()
    recent_sector_spikes[sector] = n_sp

active_sectors = {s: n for s, n in recent_sector_spikes.items() if n > 0}
dead_sectors = {s for s in sectors if recent_sector_spikes.get(s, 0) == 0}

# Coverage gap rate (60m MFE)
recent_spikes_all = df_recent['Spike_Long'] | df_recent['Spike_Short']
total_recent_spikes = recent_spikes_all.sum()
deployed_recent_spikes = (recent_spikes_all & df_recent['Deployed']).sum()
gap_rate = ((total_recent_spikes - deployed_recent_spikes) /
            max(total_recent_spikes, 1) * 100)

# PM spike comparison to 360
early_df = df_daily[(df_daily['Date'] >= '2026-07-01') & (df_daily['Date'] <= '2026-07-20')]
late_df = df_daily[(df_daily['Date'] >= '2026-08-25') & (df_daily['Date'] <= '2026-09-07')]
pm_freq_early = early_df['PM_Spike_3pct'].mean() * 100 if len(early_df) > 0 else 0
pm_freq_late = late_df['PM_Spike_3pct'].mean() * 100 if len(late_df) > 0 else 0
pm_change = ((pm_freq_late - pm_freq_early) / max(pm_freq_early, 0.01)) * 100

# Improving tickers
n_improving_l = len(df_trends[df_trends['Delta_L'] > 0.5]) if len(df_trends) > 0 else 0
n_collapsing_l = len(df_trends[(df_trends['Delta_L'] < -1.0) &
                                (df_trends['Deployed'] == 'Y')]) if len(df_trends) > 0 else 0

print(f"""
  METHODOLOGY: 60-min MFE (matching model horizon) + PM spike magnitude (matching 360 diagnosis)

  1. PM SPIKE COMPRESSION (360 COMPARISON):
     - Early (Jul 1-20) PM 3%+ freq: {pm_freq_early:.1f}%
     - Late  (Aug 25-Sep 7) PM 3%+ freq: {pm_freq_late:.1f}%
     - Change: {pm_change:+.0f}%
     - 360 diagnosis (Sep 2) found: 6.6% → 0.9% (-86%)

  2. COVERAGE GAP (60m MFE): {gap_rate:.0f}% of recent spikes outside deployed universe
     - {int(total_recent_spikes)} total 60m spikes (last 14d), {int(deployed_recent_spikes)} in deployed tickers
     - {len(missed_l)} missed LONG, {len(missed_s)} missed SHORT

  3. SECTOR HEALTH (last 14 days):
     - Active: {', '.join(f'{s} ({n})' for s, n in sorted(active_sectors.items(), key=lambda x: -x[1]))}
     - Dead: {', '.join(sorted(dead_sectors)) if dead_sectors else 'None'}

  4. SIGNAL MIGRATION:
     - {n_improving_l} tickers with IMPROVING LONG 60m MFE
     - {n_collapsing_l} DEPLOYED tickers with COLLAPSING LONG 60m MFE (>1% drop)

  5. SPECIALIZATION OPPORTUNITY:
     See Phase 6 correlation analysis — sectors with high internal ρ
     would benefit most from sector-specific models.

  6. RECOMMENDATIONS:
""")

if gap_rate > 20:
    print(f"     a) COVERAGE GAP IS SIGNIFICANT — add non-deployed spiking tickers to pipeline")
elif gap_rate > 5:
    print(f"     a) Moderate coverage gap — a few tickers worth adding")
else:
    print(f"     a) Coverage is adequate — spikes are market-wide compression, not migration")

if pm_change < -50:
    print(f"     b) PM SPIKES DOWN {pm_change:.0f}% — CONFIRMS 360 bilateral compression finding")
elif pm_change < -20:
    print(f"     b) PM spikes moderately compressed ({pm_change:.0f}%) — partial confirmation of 360")
else:
    print(f"     b) PM spikes NOT significantly compressed — may differ from 360 window")

if len(active_sectors) <= 2:
    print(f"     c) Only {len(active_sectors)} sectors active — concentrate on those, reduce noise from dead sectors")
else:
    print(f"     c) Multiple sectors still active — broad coverage justified")

if n_collapsing_l > 5:
    print(f"     d) {n_collapsing_l} deployed tickers collapsed — consider removing worst to reduce noise")

print(f"     e) Check Phase 6 sector correlations — high ρ sectors may benefit from dedicated models")
print(f"     f) Re-run after VIX recovers above 18 to confirm regime dependency")


# ==============================================================================
# CLEANUP
# ==============================================================================
gc.collect()
try:
    ctypes.CDLL("libc.so.6").malloc_trim(0)
except Exception:
    pass

print(f"\n{'='*W}")
print(f"  ASSET-SECTOR DECOMPOSITION COMPLETE")
print(f"  Charts: {chart_path}")
print(f"{'='*W}")
