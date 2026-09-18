"""
analyze_regime_history.py — Has the model seen June 2026 conditions before?
===========================================================================
Loads the full historical context features (SPY/VIX/TLT/IWM) and per-ticker
OHLCV across the ENTIRE training + validation + test period (2022-02 → 2026-06).

For every trading day at entry time, computes:
  - The exact context vector the model sees (VIX/SPY/TLT/IWM ROC at multiple horizons)
  - The actual spike outcome (max gain within the session)

Then answers three key questions:
  1. Where does June 2026 sit in the FULL historical distribution of context features?
     (percentile rank — is it truly out-of-distribution, or has the model seen this?)
  2. When context features looked like June 2026 in the PAST, did spikes still happen?
     (conditional spike rate — did the feature→outcome relationship change?)
  3. Which specific context conditions have predictive power for spike suppression,
     and did the model learn them? (stratified analysis by VIX level, IWM trend, etc.)

Run in Colab: paste as cell. No TF needed — pure data analysis, no model inference.
"""

import os, gc, sys, json, subprocess, warnings, ctypes
from datetime import datetime, timedelta
import numpy as np, pandas as pd, pytz

warnings.filterwarnings('ignore')

def _release_memory():
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

try:
    from google.cloud import storage as gcs_storage
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "google-cloud-storage", "pyarrow", "gcsfs",
                           "scipy", "--quiet"])
    from google.cloud import storage as gcs_storage

try:
    import pandas_ta as ta
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "pandas_ta", "--no-deps", "--quiet"])
    import pandas_ta as ta

from scipy import stats as sp_stats

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass

# ==============================================================================
# CONFIG
# ==============================================================================
BUCKET_NAME   = os.environ.get("GCS_BUCKET", "")
INFERENCE_DIR = "Data/Inference_v2"

# Full history: training starts Feb 2022, analysis through current
HISTORY_START = '2022-02-01'
HISTORY_END   = '2026-06-08'

# Model training cutoffs — what each experiment family has seen
FAMILY_CUTOFFS = {
    '160_XSECTOR': '2025-03-01',   # oldest: trained on Feb 2022 → Mar 2025
    '152_XSECTOR': '2025-10-01',   # trained on Feb 2022 → Oct 2025
    '170_XSECTOR': '2026-01-01',   # ASTS, SMR deployed from this (ensemble)
    '185_XSECTOR': '2026-01-01',   # newest: Feb 2022 → Jan 2026
}

# WF validation + test periods (post-training data the model was validated on)
WF_PERIODS = {
    '160_XSECTOR': {'opt': ('2025-04-01', '2025-07-31'), 'val_last': '2026-03-31'},
    '152_XSECTOR': {'opt': ('2025-10-02', '2025-11-30'), 'val_last': '2026-03-31'},
    '170_XSECTOR': {'opt': ('2025-04-01', '2025-07-31'), 'val_last': '2026-03-31'},
    '185_XSECTOR': {'opt': ('2026-01-02', '2026-02-28'), 'val_last': '2026-03-31'},
}

# Tickers to analyze — the full deployed set
DEPLOYED_TICKERS = [
    'ACHR', 'ALAB', 'APLD', 'ASTS', 'BBAI', 'BE', 'BTBT', 'CIFR', 'CLSK',
    'FCEL', 'HUT', 'IREN', 'JOBY', 'KEEL', 'LUNR', 'MARA', 'MSTR', 'MVIS',
    'OKLO', 'PL', 'PLUG', 'RCAT', 'RDW', 'RIOT', 'RKLB', 'SMCI', 'SMR', 'WULF',
]

CONTEXT_ASSETS = ['SPY', 'VIX', 'TLT', 'IWM']
ROC_HORIZONS = [5, 30, 60, 120]
TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

PST = pytz.timezone('America/Los_Angeles')
ENTRY_MINUTES = list(range(360, 391))  # 6:00-6:30 AM PST
SPIKE_THRESHOLDS = [0.03, 0.05, 0.08, 0.10]
EXIT_MINUTE = 780  # 1:00 PM PST

W = 120

gcs_client = gcs_storage.Client()

# ==============================================================================
# STEP 1: LOAD FULL HISTORICAL CONTEXT FEATURES
# ==============================================================================
print(f"{'='*W}")
print(f"  REGIME HISTORY ANALYSIS — Has the model seen June 2026 before?")
print(f"  Full history: {HISTORY_START} → {HISTORY_END}")
print(f"{'='*W}")

print(f"\n  Model training cutoffs:")
for fam, cutoff in sorted(FAMILY_CUTOFFS.items()):
    print(f"    {fam}: trained on {HISTORY_START} → {cutoff}")

# Load context assets for full history
print(f"\n  Loading context assets (full history)...")
context_frames = {}
for ticker in CONTEXT_ASSETS:
    clean = TICKER_FILE_MAP.get(ticker, ticker)
    uri = f"gs://{BUCKET_NAME}/{INFERENCE_DIR}/{clean}_master_inference.parquet"
    try:
        d = pd.read_parquet(uri)
    except Exception:
        print(f"    {clean} FAILED"); continue
    d = d.reset_index()
    dt_cols = [c for c in d.columns if pd.api.types.is_datetime64_any_dtype(d[c])]
    if dt_cols:
        d['Datetime'] = pd.to_datetime(d[dt_cols[0]], utc=True)
    else:
        d['Datetime'] = pd.to_datetime(d[d.columns[0]], unit='ms', utc=True)
    d = d.set_index('Datetime').sort_index()
    d = d[~d.index.duplicated(keep='first')]
    std_map = {'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'}
    lower_map = {k.lower():v for k,v in std_map.items()}
    d.columns = [lower_map.get(str(c).strip().lower(), str(c).strip()) for c in d.columns]
    d = d[d.index >= pd.Timestamp(HISTORY_START, tz='UTC')]
    d = d[['Close']].copy()
    d = d.resample('1min').asfreq()
    d['Close'] = d['Close'].ffill()

    daily_c = d['Close'].resample('D').last().ffill().shift(1).reindex(d.index, method='ffill')
    cols = {f'{clean}_Anchor_ROC': (d['Close'] - daily_c) / daily_c}
    for h in ROC_HORIZONS:
        cols[f'{clean}_ROC_{h}m'] = d['Close'].pct_change(h).fillna(0)
    context_frames[clean] = pd.DataFrame(cols, index=d.index)
    print(f"    {clean}: {len(d):,} rows ({d.index.min().date()} → {d.index.max().date()})")
    del d, daily_c; _release_memory()

market_df = pd.concat(context_frames.values(), axis=1, sort=True).ffill().fillna(0)
del context_frames; _release_memory()
print(f"    Market context: {len(market_df):,} rows")

# ==============================================================================
# STEP 2: BUILD DAILY CONTEXT SNAPSHOTS AT ENTRY TIME
# ==============================================================================
print(f"\n  Building daily context snapshots at entry time...")

# Vectorized: filter to entry-time bars first, then take one per day
SNAPSHOT_MINUTE = 375

market_pst = market_df.copy()
market_pst.index = market_pst.index.tz_convert(PST)
mins = market_pst.index.hour * 60 + market_pst.index.minute
entry_rows = market_pst[(mins >= SNAPSHOT_MINUTE - 2) & (mins <= SNAPSHOT_MINUTE + 2)].copy()
entry_rows['date'] = entry_rows.index.date
df_ctx = entry_rows.groupby('date').apply(lambda g: g.iloc[len(g)//2]).reset_index(drop=True)
df_ctx['date'] = pd.to_datetime(df_ctx['date'])
df_ctx['date_str'] = df_ctx['date'].dt.strftime('%Y-%m-%d')
print(f"    {len(df_ctx)} trading days with context snapshots")

del market_pst, entry_rows; _release_memory()

# ==============================================================================
# STEP 3: COMPUTE DAILY SPIKE RATES ACROSS ALL DEPLOYED TICKERS
# ==============================================================================
print(f"\n  Computing daily spike rates for {len(DEPLOYED_TICKERS)} tickers...")

# For each ticker × day, compute: did a ≥3%/5%/8% spike happen from entry to 1 PM?
ticker_daily_spikes = {}

for ti, ticker in enumerate(DEPLOYED_TICKERS):
    clean = TICKER_FILE_MAP.get(ticker, ticker)
    uri = f"gs://{BUCKET_NAME}/{INFERENCE_DIR}/{clean}_master_inference.parquet"
    try:
        df = pd.read_parquet(uri)
    except Exception:
        print(f"    [{ti+1}/{len(DEPLOYED_TICKERS)}] {ticker} SKIP"); continue

    df = df.reset_index()
    dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    if dt_cols:
        df['Datetime'] = pd.to_datetime(df[dt_cols[0]], utc=True)
    else:
        df['Datetime'] = pd.to_datetime(df[df.columns[0]], unit='ms', utc=True)
    df = df.set_index('Datetime').sort_index()
    df = df[~df.index.duplicated(keep='first')]
    std_map = {'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'}
    lower_map = {k.lower():v for k,v in std_map.items()}
    df.columns = [lower_map.get(str(c).strip().lower(), str(c).strip()) for c in df.columns]
    df = df[df.index >= pd.Timestamp(HISTORY_START, tz='UTC')]

    # Convert to PST, add helper columns
    df.index = df.index.tz_convert(PST)
    df['_min'] = df.index.hour * 60 + df.index.minute
    df['_date'] = df.index.date

    # Entry bars: first bar in 6:00-6:30 window with volume > 0
    entry_mask = (df['_min'] >= 360) & (df['_min'] <= 390)
    if 'Volume' in df.columns:
        entry_mask = entry_mask & (df['Volume'] > 0)
    entries = df[entry_mask].groupby('_date').first()
    entries = entries[entries['Close'] > 0]

    # Session highs/lows: all bars from entry through 1 PM
    session = df[(df['_min'] >= 360) & (df['_min'] <= EXIT_MINUTE)].copy()
    sess_highs = session.groupby('_date')['High'].max()
    sess_lows = session.groupby('_date')['Low'].min()
    sess_close = session.groupby('_date')['Close'].last()

    common_dates = entries.index.intersection(sess_highs.index)
    if len(common_dates) > 0:
        ep = entries.loc[common_dates, 'Close']
        result = pd.DataFrame({
            'date': pd.to_datetime(list(common_dates)),
            'ticker': ticker,
            'entry_price': ep.values,
            'max_gain': (sess_highs.loc[common_dates].values / ep.values) - 1,
            'max_drop': (sess_lows.loc[common_dates].values / ep.values) - 1,
            'eod_ret': (sess_close.loc[common_dates].values / ep.values) - 1,
        })
        ticker_daily_spikes[ticker] = result
        print(f"    [{ti+1}/{len(DEPLOYED_TICKERS)}] {ticker}: {len(result)} days "
              f"({result['date'].min().date()} → {result['date'].max().date()})")
    else:
        print(f"    [{ti+1}/{len(DEPLOYED_TICKERS)}] {ticker}: no data")

    del df; _release_memory()

# Merge all ticker-day spike data
all_spikes = pd.concat(ticker_daily_spikes.values(), ignore_index=True)
print(f"\n    Total: {len(all_spikes):,} ticker-days across {all_spikes['ticker'].nunique()} tickers")
print(f"    Date range: {all_spikes['date'].min().date()} → {all_spikes['date'].max().date()}")

# Compute daily aggregate spike rates (across all tickers)
daily_agg = all_spikes.groupby('date').agg(
    n_tickers=('ticker', 'nunique'),
    spike_3pct=('max_gain', lambda x: (x >= 0.03).mean()),
    spike_5pct=('max_gain', lambda x: (x >= 0.05).mean()),
    spike_8pct=('max_gain', lambda x: (x >= 0.08).mean()),
    spike_10pct=('max_gain', lambda x: (x >= 0.10).mean()),
    avg_max_gain=('max_gain', 'mean'),
    avg_max_drop=('max_drop', 'mean'),
    avg_eod_ret=('eod_ret', 'mean'),
    pct_down=('eod_ret', lambda x: (x < 0).mean()),
).reset_index()

# Merge context + spike data
df_hist = pd.merge(df_ctx, daily_agg, on='date', how='inner')
print(f"    Merged history: {len(df_hist)} days with both context + spike data")

# ==============================================================================
# STEP 4: DEFINE "JUNE-LIKE" STRESS CONDITIONS
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  ANALYSIS 1: Where does June 2026 sit in the historical distribution?")
print(f"{'='*W}")

# June 2026 medians (from the deep analysis)
june_mask = df_hist['date'] >= '2026-06-01'
pre_june = df_hist[~june_mask]
june_only = df_hist[june_mask]

key_context_features = [
    'VIX_Anchor_ROC', 'VIX_ROC_60m', 'VIX_ROC_120m',
    'IWM_Anchor_ROC', 'IWM_ROC_30m', 'IWM_ROC_60m', 'IWM_ROC_120m',
    'SPY_Anchor_ROC', 'SPY_ROC_60m', 'SPY_ROC_120m',
    'TLT_Anchor_ROC', 'TLT_ROC_60m', 'TLT_ROC_120m',
]

print(f"\n  Each June day's percentile rank in the FULL pre-June distribution:")
print(f"  (100th = highest ever seen, 0th = lowest ever seen)")
print(f"\n  {'Feature':<22}", end='')
for _, row in june_only.iterrows():
    print(f" {row['date_str'][:10]:>12}", end='')
print(f" {'Jun Med':>10} {'Full Med':>10} {'Pctile':>8}")
print(f"  {'─'*22}", end='')
for _ in range(len(june_only)):
    print(f" {'─'*12}", end='')
print(f" {'─'*10} {'─'*10} {'─'*8}")

for feat in key_context_features:
    if feat not in pre_june.columns:
        continue
    hist_vals = pre_june[feat].dropna().values
    jun_med = june_only[feat].median()
    full_med = pre_june[feat].median()
    pctile = sp_stats.percentileofscore(hist_vals, jun_med)

    print(f"  {feat:<22}", end='')
    for _, row in june_only.iterrows():
        val = row[feat]
        day_pctile = sp_stats.percentileofscore(hist_vals, val)
        marker = '!!' if day_pctile > 95 or day_pctile < 5 else ''
        print(f" {val:>10.5f}{marker}", end='')
    print(f" {jun_med:>10.5f} {full_med:>10.5f} {pctile:>7.1f}%")

# Also show spike rate + drawdown percentiles
print(f"\n  Daily outcomes:")
print(f"  {'Metric':<22}", end='')
for _, row in june_only.iterrows():
    print(f" {row['date_str'][:10]:>12}", end='')
print()
print(f"  {'─'*22}", end='')
for _ in range(len(june_only)):
    print(f" {'─'*12}", end='')
print()

for metric, fmt in [('spike_3pct', '.0%'), ('spike_5pct', '.0%'),
                     ('spike_8pct', '.0%'), ('spike_10pct', '.0%'),
                     ('avg_max_gain', '.2%'), ('avg_max_drop', '.2%'),
                     ('avg_eod_ret', '.2%'), ('pct_down', '.0%')]:
    if metric not in june_only.columns:
        continue
    print(f"  {metric:<22}", end='')
    for _, row in june_only.iterrows():
        print(f" {row[metric]:>12{fmt}}", end='')
    print()


# ==============================================================================
# ANALYSIS 2: CONDITIONAL SPIKE RATES — When conditions looked like June, what happened?
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  ANALYSIS 2: When historical conditions looked like June, did spikes still happen?")
print(f"{'='*W}")

# Define stress conditions based on June's actual values
# VIX rising intraday: VIX_Anchor_ROC > 0.01
# IWM selling: IWM_ROC_120m < -0.001
# TLT selling: TLT_ROC_60m < -0.0005

conditions = {
    'VIX_rising': ('VIX_Anchor_ROC', '>', 0.01),
    'VIX_surging': ('VIX_Anchor_ROC', '>', 0.02),
    'IWM_selling': ('IWM_ROC_120m', '<', -0.001),
    'IWM_dumping': ('IWM_ROC_120m', '<', -0.003),
    'TLT_selling': ('TLT_ROC_60m', '<', -0.0005),
    'SPY_weak': ('SPY_ROC_120m', '<', -0.001),
}

# First show: how often each condition occurred historically
print(f"\n  Condition frequency across full history:")
print(f"  {'Condition':<20} {'Threshold':<20} {'Days':>6} {'% of History':>13}")
print(f"  {'─'*20} {'─'*20} {'─'*6} {'─'*13}")

condition_masks = {}
for name, (feat, op, thresh) in conditions.items():
    if op == '>':
        mask = df_hist[feat] > thresh
    else:
        mask = df_hist[feat] < thresh
    condition_masks[name] = mask
    n_days = mask.sum()
    pct = n_days / len(df_hist) * 100
    print(f"  {name:<20} {feat} {op} {thresh:<10.4f} {n_days:>6} {pct:>12.1f}%")

# Spike rates under each condition vs normal — at model target thresholds (8%, 10%)
for spike_col, spike_label in [('spike_8pct', '≥8%'), ('spike_10pct', '≥10%'), ('spike_3pct', '≥3% (ref)')]:
    print(f"\n  Spike rates ({spike_label}) under each condition vs normal:")
    print(f"  {'Condition':<20} {'Normal Rate':>12} {'(N)':>6} {'Stress Rate':>12} {'(N)':>6} {'Δ':>8} {'p-value':>8}")
    print(f"  {'─'*20} {'─'*12} {'─'*6} {'─'*12} {'─'*6} {'─'*8} {'─'*8}")

    for name, mask in condition_masks.items():
        normal = df_hist[~mask]
        stress = df_hist[mask]
        if len(stress) < 5 or len(normal) < 5:
            continue
        rate_normal = normal[spike_col].mean()
        rate_stress = stress[spike_col].mean()
        delta = rate_stress - rate_normal
        _, p = sp_stats.mannwhitneyu(normal[spike_col].values, stress[spike_col].values,
                                      alternative='two-sided')
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        print(f"  {name:<20} {rate_normal:>11.1%} {len(normal):>6} {rate_stress:>11.1%} {len(stress):>6} "
              f"{delta:>+7.1%} {p:>7.4f}{sig}")

# Combined stress: VIX rising AND IWM selling
print(f"\n  Combined stress conditions:")
combo_masks = {
    'VIX_rising + IWM_sell': condition_masks['VIX_rising'] & condition_masks['IWM_selling'],
    'VIX_surging + IWM_dump': condition_masks['VIX_surging'] & condition_masks['IWM_dumping'],
    'VIX + IWM + TLT': condition_masks['VIX_rising'] & condition_masks['IWM_selling'] & condition_masks['TLT_selling'],
    'All 4 stress': condition_masks['VIX_rising'] & condition_masks['IWM_selling'] & condition_masks['TLT_selling'] & condition_masks['SPY_weak'],
}

print(f"  {'Combo':<25} {'Days':>5} {'Spike≥3%':>9} {'Spike≥5%':>9} {'Spike≥8%':>9} {'Spike≥10%':>10} {'AvgMaxGain':>11} {'AvgMaxDrop':>11} {'%Down':>6}")
print(f"  {'─'*25} {'─'*5} {'─'*9} {'─'*9} {'─'*9} {'─'*10} {'─'*11} {'─'*11} {'─'*6}")

# Baseline
print(f"  {'ALL DAYS':<25} {len(df_hist):>5} {df_hist['spike_3pct'].mean():>8.1%} "
      f"{df_hist['spike_5pct'].mean():>8.1%} {df_hist['spike_8pct'].mean():>8.1%} "
      f"{df_hist['spike_10pct'].mean():>9.1%} {df_hist['avg_max_gain'].mean():>10.2%} "
      f"{df_hist['avg_max_drop'].mean():>10.2%} {df_hist['pct_down'].mean():>5.0%}")

for name, mask in combo_masks.items():
    subset = df_hist[mask]
    if len(subset) == 0:
        print(f"  {name:<25} {'(never occurred)':>5}")
        continue
    print(f"  {name:<25} {len(subset):>5} {subset['spike_3pct'].mean():>8.1%} "
          f"{subset['spike_5pct'].mean():>8.1%} {subset['spike_8pct'].mean():>8.1%} "
          f"{subset['spike_10pct'].mean():>9.1%} {subset['avg_max_gain'].mean():>10.2%} "
          f"{subset['avg_max_drop'].mean():>10.2%} {subset['pct_down'].mean():>5.0%}")

# June specifically
print(f"  {'JUNE 2026':<25} {len(june_only):>5} {june_only['spike_3pct'].mean():>8.1%} "
      f"{june_only['spike_5pct'].mean():>8.1%} {june_only['spike_8pct'].mean():>8.1%} "
      f"{june_only['spike_10pct'].mean():>9.1%} {june_only['avg_max_gain'].mean():>10.2%} "
      f"{june_only['avg_max_drop'].mean():>10.2%} {june_only['pct_down'].mean():>5.0%}")


# ==============================================================================
# ANALYSIS 3: TIME SERIES — When did similar stress episodes happen?
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  ANALYSIS 3: Historical stress episodes (VIX rising + IWM selling)")
print(f"  Every period where VIX_Anchor_ROC > 0.01 AND IWM_ROC_120m < -0.001")
print(f"{'='*W}")

stress_mask = condition_masks['VIX_rising'] & condition_masks['IWM_selling']
df_hist['is_stress'] = stress_mask

# Find contiguous stress episodes
df_hist['stress_group'] = (df_hist['is_stress'] != df_hist['is_stress'].shift(1)).cumsum()
stress_episodes = df_hist[df_hist['is_stress']].groupby('stress_group')

episodes = []
for group_id, group in stress_episodes:
    if len(group) < 1:
        continue
    episodes.append({
        'start': group['date'].min(),
        'end': group['date'].max(),
        'days': len(group),
        'spike_3pct': group['spike_3pct'].mean(),
        'spike_5pct': group['spike_5pct'].mean(),
        'spike_8pct': group['spike_8pct'].mean(),
        'spike_10pct': group['spike_10pct'].mean(),
        'avg_max_gain': group['avg_max_gain'].mean(),
        'avg_max_drop': group['avg_max_drop'].mean(),
        'avg_vix_roc': group['VIX_Anchor_ROC'].mean(),
        'avg_iwm_roc': group['IWM_ROC_120m'].mean(),
        'pct_down': group['pct_down'].mean(),
    })

ep_df = pd.DataFrame(episodes).sort_values('start')

print(f"\n  Found {len(ep_df)} stress episodes in full history")
print(f"\n  {'Start':<12} {'End':<12} {'Days':>5} {'Spike≥8%':>9} {'Spike≥10%':>10} "
      f"{'AvgGain':>8} {'AvgDrop':>8} {'VIX_ROC':>8} {'IWM_ROC':>8} {'%Down':>6}")
print(f"  {'─'*12} {'─'*12} {'─'*5} {'─'*9} {'─'*10} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*6}")

for _, ep in ep_df.iterrows():
    is_june = ep['start'] >= pd.Timestamp('2026-06-01')
    marker = ' ← JUNE' if is_june else ''
    print(f"  {ep['start'].strftime('%Y-%m-%d'):<12} {ep['end'].strftime('%Y-%m-%d'):<12} {ep['days']:>5} "
          f"{ep['spike_8pct']:>8.1%} {ep['spike_10pct']:>9.1%} "
          f"{ep['avg_max_gain']:>+7.2%} {ep['avg_max_drop']:>+7.2%} "
          f"{ep['avg_vix_roc']:>+7.4f} {ep['avg_iwm_roc']:>+7.4f} "
          f"{ep['pct_down']:>5.0%}{marker}")


# ==============================================================================
# ANALYSIS 4: TRAINING COVERAGE — Did each model family see stress days?
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  ANALYSIS 4: Training coverage — How many stress days did each model see?")
print(f"{'='*W}")

print(f"\n  {'Family':<16} {'Cutoff':<12} {'Train Days':>11} {'Stress Days':>12} "
      f"{'% Stress':>9} {'Spike≥8% (stress)':>18} {'Spike≥8% (normal)':>18} {'Spike≥10% (stress)':>19}")
print(f"  {'─'*16} {'─'*12} {'─'*11} {'─'*12} {'─'*9} {'─'*18} {'─'*18} {'─'*19}")

for fam, cutoff in sorted(FAMILY_CUTOFFS.items(), key=lambda x: x[1]):
    train_data = df_hist[df_hist['date'] < cutoff]
    train_stress = train_data[train_data['is_stress']]
    train_normal = train_data[~train_data['is_stress']]
    spike8_stress = train_stress['spike_8pct'].mean() if len(train_stress) > 0 else 0
    spike8_normal = train_normal['spike_8pct'].mean() if len(train_normal) > 0 else 0
    spike10_stress = train_stress['spike_10pct'].mean() if len(train_stress) > 0 else 0
    print(f"  {fam:<16} {cutoff:<12} {len(train_data):>11} {len(train_stress):>12} "
          f"{len(train_stress)/max(1,len(train_data))*100:>8.1f}% "
          f"{spike8_stress:>17.1%} {spike8_normal:>17.1%} {spike10_stress:>18.1%}")


# ==============================================================================
# ANALYSIS 5: STRATIFIED SPIKE RATES BY VIX LEVEL
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  ANALYSIS 5: Spike rates stratified by VIX regime")
print(f"  Does high VIX always suppress spikes, or is June different?")
print(f"{'='*W}")

# Bucket by VIX_Anchor_ROC level
vix_bins = [(-1, -0.02), (-0.02, -0.01), (-0.01, 0), (0, 0.01), (0.01, 0.02),
            (0.02, 0.03), (0.03, 0.05), (0.05, 1.0)]

print(f"\n  {'VIX_Anchor_ROC':<18} {'Days':>6} {'Spike≥3%':>9} {'Spike≥5%':>9} {'Spike≥8%':>9} {'Spike≥10%':>10} "
      f"{'AvgGain':>8} {'AvgDrop':>8} {'%Down':>6}")
print(f"  {'─'*18} {'─'*6} {'─'*9} {'─'*9} {'─'*9} {'─'*10} {'─'*8} {'─'*8} {'─'*6}")

for lo, hi in vix_bins:
    mask = (df_hist['VIX_Anchor_ROC'] >= lo) & (df_hist['VIX_Anchor_ROC'] < hi)
    subset = df_hist[mask]
    if len(subset) < 3:
        continue
    label = f"[{lo:+.2f}, {hi:+.2f})"
    print(f"  {label:<18} {len(subset):>6} {subset['spike_3pct'].mean():>8.1%} "
          f"{subset['spike_5pct'].mean():>8.1%} {subset['spike_8pct'].mean():>8.1%} "
          f"{subset['spike_10pct'].mean():>9.1%} {subset['avg_max_gain'].mean():>+7.2%} "
          f"{subset['avg_max_drop'].mean():>+7.2%} {subset['pct_down'].mean():>5.0%}")

# Same for IWM_ROC_120m
print(f"\n  {'IWM_ROC_120m':<18} {'Days':>6} {'Spike≥3%':>9} {'Spike≥5%':>9} {'Spike≥8%':>9} {'Spike≥10%':>10} "
      f"{'AvgGain':>8} {'AvgDrop':>8} {'%Down':>6}")
print(f"  {'─'*18} {'─'*6} {'─'*9} {'─'*9} {'─'*9} {'─'*10} {'─'*8} {'─'*8} {'─'*6}")

iwm_bins = [(-1, -0.01), (-0.01, -0.005), (-0.005, -0.002), (-0.002, 0),
            (0, 0.002), (0.002, 0.005), (0.005, 0.01), (0.01, 1.0)]

for lo, hi in iwm_bins:
    mask = (df_hist['IWM_ROC_120m'] >= lo) & (df_hist['IWM_ROC_120m'] < hi)
    subset = df_hist[mask]
    if len(subset) < 3:
        continue
    label = f"[{lo:+.3f}, {hi:+.3f})"
    print(f"  {label:<18} {len(subset):>6} {subset['spike_3pct'].mean():>8.1%} "
          f"{subset['spike_5pct'].mean():>8.1%} {subset['spike_8pct'].mean():>8.1%} "
          f"{subset['spike_10pct'].mean():>9.1%} {subset['avg_max_gain'].mean():>+7.2%} "
          f"{subset['avg_max_drop'].mean():>+7.2%} {subset['pct_down'].mean():>5.0%}")


# ==============================================================================
# ANALYSIS 6: MONTHLY TIME SERIES — Spike rate + context evolution
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  ANALYSIS 6: Monthly evolution — spike rates + context features")
print(f"  Shows whether June is a sudden break or gradual trend")
print(f"{'='*W}")

df_hist['month'] = df_hist['date'].dt.to_period('M')
monthly = df_hist.groupby('month').agg(
    days=('date', 'count'),
    spike_3pct=('spike_3pct', 'mean'),
    spike_5pct=('spike_5pct', 'mean'),
    spike_8pct=('spike_8pct', 'mean'),
    spike_10pct=('spike_10pct', 'mean'),
    avg_max_gain=('avg_max_gain', 'mean'),
    avg_max_drop=('avg_max_drop', 'mean'),
    vix_anchor=('VIX_Anchor_ROC', 'mean'),
    iwm_120=('IWM_ROC_120m', 'mean'),
    spy_120=('SPY_ROC_120m', 'mean'),
    tlt_60=('TLT_ROC_60m', 'mean'),
    stress_pct=('is_stress', 'mean'),
).reset_index()

# Show last 18 months
recent = monthly.tail(18)

print(f"\n  {'Month':<10} {'Days':>5} {'Spike≥8%':>9} {'Spike≥10%':>10} {'Spike≥3%':>9} {'AvgGain':>8} {'AvgDrop':>8} "
      f"{'VIX_ROC':>8} {'IWM_120':>8} {'Stress%':>8}")
print(f"  {'─'*10} {'─'*5} {'─'*9} {'─'*10} {'─'*9} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

for _, r in recent.iterrows():
    marker = ' ←' if str(r['month']).startswith('2026-06') else ''
    print(f"  {str(r['month']):<10} {r['days']:>5} {r['spike_8pct']:>8.1%} {r['spike_10pct']:>9.1%} {r['spike_3pct']:>8.1%} "
          f"{r['avg_max_gain']:>+7.2%} {r['avg_max_drop']:>+7.2%} "
          f"{r['vix_anchor']:>+7.4f} {r['iwm_120']:>+7.4f} {r['stress_pct']*100:>7.1f}%{marker}")


# ==============================================================================
# ANALYSIS 7: PER-TICKER — Has each ticker's spike regime decayed?
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  ANALYSIS 7: Per-ticker spike regime — monthly ≥8% spike rate")
print(f"  Separates: model regime shift vs specific ticker spike decay")
print(f"{'='*W}")

# For each ticker: monthly spike rate for the last 12 months
print(f"\n  {'Ticker':<7}", end='')
months_to_show = monthly.tail(12)['month'].tolist()
for m in months_to_show:
    print(f" {str(m)[-5:]:>7}", end='')
print(f" {'Trend':>7}")
print(f"  {'─'*7}" + " ".join(['─'*7]*len(months_to_show)) + f" {'─'*7}")

for ticker in sorted(DEPLOYED_TICKERS):
    if ticker not in ticker_daily_spikes:
        continue
    tdf = ticker_daily_spikes[ticker].copy()
    tdf['month'] = tdf['date'].dt.to_period('M')

    monthly_rates = {}
    for m in months_to_show:
        m_data = tdf[tdf['month'] == m]
        if len(m_data) >= 3:
            monthly_rates[m] = (m_data['max_gain'] >= 0.08).mean()
        else:
            monthly_rates[m] = None

    print(f"  {ticker:<7}", end='')
    vals = []
    for m in months_to_show:
        rate = monthly_rates.get(m)
        if rate is not None:
            print(f" {rate*100:>6.0f}%", end='')
            vals.append(rate)
        else:
            print(f" {'—':>7}", end='')

    # Trend: compare first half vs second half
    if len(vals) >= 4:
        first = np.mean(vals[:len(vals)//2])
        last = np.mean(vals[len(vals)//2:])
        trend = last - first
        arrow = '↑' if trend > 0.05 else ('↓' if trend < -0.05 else '→')
        print(f" {arrow}{trend*100:>+5.0f}%")
    else:
        print(f" {'?':>7}")


# ==============================================================================
# SYNTHESIS
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  SYNTHESIS: Is this a new regime or a known pattern?")
print(f"{'='*W}")

# Compute key comparisons
stress_days_in_history = df_hist[df_hist['is_stress'] & (df_hist['date'] < '2026-06-01')]
normal_days = df_hist[~df_hist['is_stress'] & (df_hist['date'] < '2026-06-01')]

hist_stress_spike_8 = stress_days_in_history['spike_8pct'].mean() if len(stress_days_in_history) > 0 else 0
hist_normal_spike_8 = normal_days['spike_8pct'].mean() if len(normal_days) > 0 else 0
june_spike_8 = june_only['spike_8pct'].mean() if len(june_only) > 0 else 0
hist_stress_spike_10 = stress_days_in_history['spike_10pct'].mean() if len(stress_days_in_history) > 0 else 0
hist_normal_spike_10 = normal_days['spike_10pct'].mean() if len(normal_days) > 0 else 0
june_spike_10 = june_only['spike_10pct'].mean() if len(june_only) > 0 else 0

# Percentile of June's VIX_Anchor_ROC
vix_vals = pre_june['VIX_Anchor_ROC'].dropna().values
jun_vix_med = june_only['VIX_Anchor_ROC'].median()
vix_pctile = sp_stats.percentileofscore(vix_vals, jun_vix_med)

# How many training days had VIX > June's median?
n_train_similar = len(pre_june[pre_june['VIX_Anchor_ROC'] > jun_vix_med])

print(f"""
  KEY FINDINGS:

  1. HISTORICAL STRESS DAYS: {len(stress_days_in_history)} days in training history had
     VIX_Anchor_ROC > 0.01 AND IWM_ROC_120m < -0.001 (the "June pattern")

  2. SPIKE RATES AT MODEL TARGET THRESHOLDS (≥8% / ≥10%):
     Normal days:  {hist_normal_spike_8:.1%} / {hist_normal_spike_10:.1%} of tickers spiked
     Stress days:  {hist_stress_spike_8:.1%} / {hist_stress_spike_10:.1%} of tickers spiked
     June 2026:    {june_spike_8:.1%} / {june_spike_10:.1%} of tickers spiked

  3. JUNE'S VIX LEVEL IN CONTEXT:
     June median VIX_Anchor_ROC = {jun_vix_med:.4f}
     This is the {vix_pctile:.0f}th percentile of all pre-June history
     {n_train_similar} training days had VIX_Anchor_ROC above this level

  4. INTERPRETATION:
""")

if len(stress_days_in_history) < 10:
    print(f"     → NEVER-BEFORE-SEEN: Only {len(stress_days_in_history)} similar stress days")
    print(f"       in training data. The model is extrapolating into unknown territory.")
    print(f"       This is a genuine out-of-distribution regime.")
elif hist_stress_spike_8 - june_spike_8 > 0.10:
    print(f"     → REGIME CHANGED at model target: Historical stress days had {hist_stress_spike_8:.0%}")
    print(f"       ≥8% spike rate, but June has only {june_spike_8:.0%}. At ≥10%: stress")
    print(f"       had {hist_stress_spike_10:.0%} vs June {june_spike_10:.0%}.")
    print(f"       The BIG spikes the model predicts have genuinely dried up under stress.")
    print(f"       Normal days still had {hist_normal_spike_8:.0%} ≥8% rate — stress suppresses")
    print(f"       large spikes even if small moves (≥3%) persist.")
elif hist_stress_spike_8 < hist_normal_spike_8 * 0.7:
    print(f"     → KNOWN BUT UNLEARNED: Stress historically suppressed ≥8% spikes")
    print(f"       ({hist_stress_spike_8:.0%} vs {hist_normal_spike_8:.0%} normal) but the model")
    print(f"       didn't learn to suppress signals. Architecture/training issue.")
else:
    print(f"     → CONSISTENT: June ≥8% spike rate ({june_spike_8:.0%}) matches historical")
    print(f"       stress rate ({hist_stress_spike_8:.0%}) and normal rate ({hist_normal_spike_8:.0%}).")
    print(f"       Stress conditions don't suppress large spikes. The failure is from")
    print(f"       tail-risk: catastrophic correlated losses on the few days with no spikes.")


# ==============================================================================
# CHARTS
# ==============================================================================
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    fig.suptitle('Regime History Analysis — Has the Model Seen June Before?',
                 fontsize=14, fontweight='bold')

    # 1. VIX_Anchor_ROC histogram with June marked
    ax = axes[0, 0]
    ax.hist(pre_june['VIX_Anchor_ROC'].values, bins=80, color='#1976d2', alpha=0.7, label='Pre-June')
    for _, row in june_only.iterrows():
        ax.axvline(x=row['VIX_Anchor_ROC'], color='red', alpha=0.7, linewidth=1.5)
    ax.axvline(x=jun_vix_med, color='red', linewidth=2, linestyle='--', label=f'June median ({vix_pctile:.0f}th pctile)')
    ax.set_xlabel('VIX_Anchor_ROC at entry')
    ax.set_title('VIX Anchor ROC Distribution')
    ax.legend(fontsize=8)

    # 2. IWM_ROC_120m histogram with June marked
    ax = axes[0, 1]
    ax.hist(pre_june['IWM_ROC_120m'].values, bins=80, color='#1976d2', alpha=0.7, label='Pre-June')
    for _, row in june_only.iterrows():
        ax.axvline(x=row['IWM_ROC_120m'], color='red', alpha=0.7, linewidth=1.5)
    iwm_med = june_only['IWM_ROC_120m'].median()
    iwm_pctile = sp_stats.percentileofscore(pre_june['IWM_ROC_120m'].dropna().values, iwm_med)
    ax.axvline(x=iwm_med, color='red', linewidth=2, linestyle='--', label=f'June median ({iwm_pctile:.0f}th pctile)')
    ax.set_xlabel('IWM_ROC_120m at entry')
    ax.set_title('IWM 120m ROC Distribution')
    ax.legend(fontsize=8)

    # 3. Scatter: VIX_Anchor_ROC vs spike rate
    ax = axes[0, 2]
    pre_scatter = pre_june.copy()
    ax.scatter(pre_scatter['VIX_Anchor_ROC'], pre_scatter['spike_8pct'],
               alpha=0.15, s=8, color='#1976d2', label='Pre-June')
    ax.scatter(june_only['VIX_Anchor_ROC'], june_only['spike_8pct'],
               alpha=0.9, s=40, color='red', marker='X', label='June 2026', zorder=5)
    ax.set_xlabel('VIX_Anchor_ROC')
    ax.set_ylabel('Daily spike rate (≥8%)')
    ax.set_title('VIX vs Spike Rate (≥8%)')
    ax.legend(fontsize=8)

    # 4. Monthly spike rate time series
    ax = axes[1, 0]
    recent_24 = monthly.tail(24)
    x = range(len(recent_24))
    ax.bar(x, recent_24['spike_8pct'].values * 100, color='#1976d2', alpha=0.7)
    if len(recent_24) > 0:
        ax.bar([len(recent_24)-1], [recent_24['spike_8pct'].values[-1]*100], color='#d32f2f', alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([str(m)[-5:] for m in recent_24['month']], rotation=45, fontsize=6)
    ax.set_ylabel('Spike Rate ≥8% (%)')
    ax.set_title('Monthly Spike Rate ≥8% (last 24 months)')

    # 5. Monthly stress percentage
    ax = axes[1, 1]
    ax.bar(x, recent_24['stress_pct'].values * 100, color='#ff9800', alpha=0.7)
    if len(recent_24) > 0:
        ax.bar([len(recent_24)-1], [recent_24['stress_pct'].values[-1]*100], color='#d32f2f', alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([str(m)[-5:] for m in recent_24['month']], rotation=45, fontsize=6)
    ax.set_ylabel('% of Days in Stress')
    ax.set_title('Monthly Stress Frequency')

    # 6. Stress spike rate vs normal spike rate (conditional)
    ax = axes[1, 2]
    normal_monthly = df_hist[~df_hist['is_stress']].groupby('month')['spike_8pct'].mean()
    stress_monthly = df_hist[df_hist['is_stress']].groupby('month')['spike_8pct'].mean()
    all_months = monthly['month'].values
    recent_m = all_months[-24:] if len(all_months) >= 24 else all_months
    x = range(len(recent_m))
    n_vals = [normal_monthly.get(m, np.nan)*100 for m in recent_m]
    s_vals = [stress_monthly.get(m, np.nan)*100 for m in recent_m]
    ax.plot(x, n_vals, 'o-', color='#1976d2', label='Normal days', markersize=3)
    ax.plot(x, s_vals, 's-', color='#d32f2f', label='Stress days', markersize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([str(m)[-5:] for m in recent_m], rotation=45, fontsize=6)
    ax.set_ylabel('Spike Rate ≥8% (%)')
    ax.set_title('Spike Rate ≥8%: Normal vs Stress Days')
    ax.legend(fontsize=8)

    plt.tight_layout()
    chart_path = '/tmp/regime_history_analysis.png'
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    print(f"\n  Charts saved: {chart_path}")
    plt.close()

except ImportError:
    print("\n  (matplotlib not available — skipping charts)")

print(f"\n{'='*W}")
print(f"  REGIME HISTORY ANALYSIS COMPLETE")
print(f"{'='*W}")
