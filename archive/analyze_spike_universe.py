"""
analyze_spike_universe.py — Universal Spike Screener (all T&Q assets)
=====================================================================
Scans ALL assets with Trades & Quotes data in GCS. For each ticker,
computes spike rates at 6%, 8%, and 10% thresholds across TRAIN/OPT/
TEST/LIVE windows. Ranks tickers by spike frequency and stability.

Output: ranked table of all assets to inform teacher/scoring pool
selection for new experiments (EXP-200 SPACE etc.)

Run in Colab or any environment with GCS access.
"""

import gc
import warnings
import numpy as np
import pandas as pd
import pytz
from datetime import datetime
from google.cloud import storage

warnings.filterwarnings('ignore')

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BUCKET_NAME   = os.environ.get("GCS_BUCKET", "")
INFERENCE_DIR = "Data/Inference"
TQ_DIR        = "Data/TradesQuotes"

TICKER_FILE_MAP = {'BITF': 'KEEL'}

DATA_START_DATE   = "2022-01-01"
TRAIN_CUTOFF_DATE = "2025-06-30"
OPT_START         = "2025-07-01"
OPT_END           = "2025-12-31"
TEST_START        = "2026-01-01"
TEST_END          = "2026-03-31"
LIVE_START        = "2026-04-01"
LIVE_END          = datetime.now(pytz.UTC).strftime("%Y-%m-%d")

THRESHOLDS = [0.06, 0.08, 0.10]

PERIODS = [
    ('TRAIN', DATA_START_DATE, TRAIN_CUTOFF_DATE),
    ('OPT',   OPT_START,       OPT_END),
    ('TEST',  TEST_START,      TEST_END),
    ('LIVE',  LIVE_START,      LIVE_END),
]

CONTEXT_ONLY = {'SPY', 'VIX', 'TLT', 'IWM', 'SMH', 'ITA', 'XLE', 'BTCUSD'}

PST = pytz.timezone('America/Los_Angeles')

client = storage.Client()
bucket = client.bucket(BUCKET_NAME)

# ==============================================================================
# DISCOVER ALL T&Q ASSETS
# ==============================================================================
def discover_tq_tickers():
    blobs = bucket.list_blobs(prefix=f"{TQ_DIR}/", delimiter="/")
    list(blobs)
    tickers = []
    for p in sorted(blobs.prefixes):
        t = p.replace(f"{TQ_DIR}/", "").rstrip("/")
        if t and t not in CONTEXT_ONLY:
            tickers.append(t)
    return tickers

# ==============================================================================
# DATA LOADING
# ==============================================================================
def load_master(ticker):
    clean = ticker.replace("I:", "").replace("X:", "")
    file_ticker = TICKER_FILE_MAP.get(clean, clean)
    for suffix in ['_master_inference', '_1m_full_historical_data']:
        blob_path = f"{INFERENCE_DIR}/{file_ticker}{suffix}.parquet"
        blob = bucket.blob(blob_path)
        if blob.exists():
            import io
            df = pd.read_parquet(io.BytesIO(blob.download_as_bytes()))
            df = df.reset_index()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
            if dt_cols:
                df['Datetime'] = pd.to_datetime(df[dt_cols[0]], utc=True)
            else:
                num_col = next((c for c in df.columns
                               if c in ('Datetime', 'Date', 'Time', 'Timestamp', 'Level_0')),
                              df.columns[0])
                df['Datetime'] = pd.to_datetime(df[num_col], unit='ms', utc=True)
            df = df.set_index('Datetime')
            df = df[~df.index.duplicated(keep='first')]
            std_map = {'open': 'Open', 'high': 'High', 'low': 'Low',
                       'close': 'Close', 'volume': 'Volume'}
            lower_map = {k.lower(): v for k, v in std_map.items()}
            df.columns = [lower_map.get(str(c).strip().lower(), str(c).strip())
                          for c in df.columns]
            df = df.loc[:, ~df.columns.duplicated(keep='first')].copy()
            df = df[df.index.notna()].sort_index()
            cutoff = pd.Timestamp(DATA_START_DATE, tz='UTC')
            df = df[df.index >= cutoff]
            keep = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in df.columns]
            return df[keep]
    return None


def heal_data(df):
    df = df.resample('1min').asfreq()
    for c in ['Open', 'High', 'Low', 'Close']:
        if c in df.columns:
            df[c] = df[c].ffill()
    if 'Volume' in df.columns:
        df['Volume'] = df['Volume'].fillna(0)
    return df


# ==============================================================================
# SPIKE COMPUTATION
# ==============================================================================
def compute_entry_bar_magnitudes(df, start_date, end_date):
    mask = (df.index >= pd.Timestamp(start_date, tz='UTC')) & \
           (df.index < pd.Timestamp(end_date, tz='UTC'))
    sub = df[mask]
    if sub.empty:
        return None

    pst_idx = sub.index.tz_convert(PST)
    mins = pst_idx.hour * 60 + pst_idx.minute
    is_entry_window = (mins >= 360) & (mins <= 390)
    is_real = sub['Volume'] > 0
    is_safe = (pst_idx.hour < 13) | ((pst_idx.hour == 13) & (pst_idx.minute == 0))

    safe_high = np.where(is_safe & is_real, sub['High'].values, -1.0)
    safe_high_s = pd.Series(safe_high, index=sub.index)
    future_max = safe_high_s.groupby(safe_high_s.index.date).transform(
        lambda x: x[::-1].cummax()[::-1])

    close_safe = sub['Close'].replace(0, np.nan)
    spike_pct = ((future_max / close_safe) - 1).clip(lower=0).fillna(0)

    entry_mask = is_entry_window & is_real
    return spike_pct[entry_mask]


# ==============================================================================
# MAIN
# ==============================================================================
print("=" * 130)
print("  UNIVERSAL SPIKE SCREENER — All T&Q Assets")
print(f"  Thresholds: {[f'{t:.0%}' for t in THRESHOLDS]}")
print(f"  Periods: TRAIN ({DATA_START_DATE}→{TRAIN_CUTOFF_DATE}), "
      f"OPT ({OPT_START}→{OPT_END}), TEST ({TEST_START}→{TEST_END}), "
      f"LIVE ({LIVE_START}→{LIVE_END})")
print("=" * 130)

# Discover tickers
tickers = discover_tq_tickers()
print(f"\nFound {len(tickers)} tradeable assets with T&Q data (excluding context-only)")

# Phase 1: Load and compute magnitudes
all_magnitudes = {}
ticker_info = {}

print("\n[PHASE 1] Loading tickers and computing spike magnitudes...\n")
for i, ticker in enumerate(tickers, 1):
    print(f"   [{i:>2}/{len(tickers)}] {ticker:<6}...", end=" ")
    df = load_master(ticker)
    if df is None:
        print("NO DATA")
        continue
    df = heal_data(df)
    n_bars = len(df)
    date_range = f"{df.index.min().strftime('%Y-%m-%d')} → {df.index.max().strftime('%Y-%m-%d')}"
    print(f"{n_bars:>10,} bars  {date_range}", end="")

    periods_ok = 0
    for period_name, start, end in PERIODS:
        mags = compute_entry_bar_magnitudes(df, start, end)
        if mags is not None and len(mags) > 0:
            all_magnitudes[(ticker, period_name)] = mags
            periods_ok += 1

    ticker_info[ticker] = {'bars': n_bars, 'range': date_range}
    print(f"  {periods_ok}/4 periods")
    del df
    gc.collect()

# Phase 2: Build summary table
print("\n" + "=" * 130)
print("  TABLE 1: SPIKE RATE BY THRESHOLD (all assets, all periods)")
print("=" * 130)

results = []
for ticker in tickers:
    if ticker not in ticker_info:
        continue
    row = {'ticker': ticker}
    for thresh in THRESHOLDS:
        t_key = f"{int(thresh*100)}pct"
        for period_name in ['TRAIN', 'OPT', 'TEST', 'LIVE']:
            key = (ticker, period_name)
            if key in all_magnitudes:
                mags = all_magnitudes[key]
                row[f'{t_key}_{period_name}'] = (mags >= thresh).mean()
                row[f'{t_key}_{period_name}_n'] = len(mags)
                if period_name == 'LIVE':
                    spikes = mags[mags >= thresh]
                    row[f'{t_key}_LIVE_med_mag'] = float(spikes.median()) if len(spikes) > 0 else 0
                    row[f'{t_key}_LIVE_mean_mag'] = float(spikes.mean()) if len(spikes) > 0 else 0
            else:
                row[f'{t_key}_{period_name}'] = 0
                row[f'{t_key}_{period_name}_n'] = 0

        rates = [row.get(f'{t_key}_{p}', 0) for p in ['TRAIN', 'OPT', 'TEST', 'LIVE']]
        mean_r = np.mean(rates)
        row[f'{t_key}_avg_rate'] = mean_r
        row[f'{t_key}_stability'] = (1.0 - np.std(rates) / mean_r) if mean_r > 0 else 0

    # Overall magnitude stats (LIVE period)
    key = (ticker, 'LIVE')
    if key in all_magnitudes:
        mags = all_magnitudes[key]
        row['live_bars'] = len(mags)
        row['live_med_mag'] = float(mags.median())
        row['live_mean_mag'] = float(mags.mean())
        row['live_p90_mag'] = float(mags.quantile(0.90))
        row['live_max_mag'] = float(mags.max())
    else:
        row['live_bars'] = 0
        row['live_med_mag'] = 0
        row['live_mean_mag'] = 0
        row['live_p90_mag'] = 0
        row['live_max_mag'] = 0

    results.append(row)

df_results = pd.DataFrame(results)

# Print ranked by 10% spike rate (LIVE)
print(f"\n  Ranked by 10% spike rate (LIVE period):\n")
print(f"  {'Ticker':<7} {'LiveBars':>8}  "
      f"{'──── 6% Spike Rate ────':^28}  "
      f"{'──── 8% Spike Rate ────':^28}  "
      f"{'──── 10% Spike Rate ───':^28}  "
      f"{'Live':>5} {'Live':>5}")
print(f"  {'':7} {'':>8}  "
      f"{'TRAIN':>6} {'OPT':>6} {'TEST':>6} {'LIVE':>6}  "
      f"{'TRAIN':>6} {'OPT':>6} {'TEST':>6} {'LIVE':>6}  "
      f"{'TRAIN':>6} {'OPT':>6} {'TEST':>6} {'LIVE':>6}  "
      f"{'MedMg':>5} {'P90':>5}")
print(f"  {'─'*7} {'─'*8}  "
      f"{'─'*6} {'─'*6} {'─'*6} {'─'*6}  "
      f"{'─'*6} {'─'*6} {'─'*6} {'─'*6}  "
      f"{'─'*6} {'─'*6} {'─'*6} {'─'*6}  "
      f"{'─'*5} {'─'*5}")

df_sorted = df_results.sort_values('10pct_LIVE', ascending=False)
for _, r in df_sorted.iterrows():
    print(f"  {r['ticker']:<7} {r.get('live_bars', 0):>8.0f}  "
          f"{r.get('6pct_TRAIN', 0):>5.1%} {r.get('6pct_OPT', 0):>5.1%} "
          f"{r.get('6pct_TEST', 0):>5.1%} {r.get('6pct_LIVE', 0):>5.1%}  "
          f"{r.get('8pct_TRAIN', 0):>5.1%} {r.get('8pct_OPT', 0):>5.1%} "
          f"{r.get('8pct_TEST', 0):>5.1%} {r.get('8pct_LIVE', 0):>5.1%}  "
          f"{r.get('10pct_TRAIN', 0):>5.1%} {r.get('10pct_OPT', 0):>5.1%} "
          f"{r.get('10pct_TEST', 0):>5.1%} {r.get('10pct_LIVE', 0):>5.1%}  "
          f"{r.get('live_med_mag', 0):>4.1%} {r.get('live_p90_mag', 0):>4.1%}")

# Phase 3: Teacher candidate ranking
print("\n" + "=" * 130)
print("  TABLE 2: TEACHER CANDIDATE RANKING (10% threshold)")
print("  Score = avg_spike_rate × stability × min(TEST, LIVE) rate")
print("=" * 130)

df_results['10pct_score'] = (
    df_results['10pct_avg_rate'] *
    df_results['10pct_stability'].clip(lower=0) *
    df_results[['10pct_TEST', '10pct_LIVE']].min(axis=1)
)
df_ranked = df_results.sort_values('10pct_score', ascending=False)

print(f"\n  {'Rank':>4} {'Ticker':<7} {'AvgRate':>8} {'Stab':>6} {'TEST':>6} {'LIVE':>6} "
      f"{'Score':>8}  {'Status'}")
print(f"  {'─'*4} {'─'*7} {'─'*8} {'─'*6} {'─'*6} {'─'*6} {'─'*8}  {'─'*30}")

currently_deployed = {
    'BGDE', 'VRT', 'RKLB', 'NNDM', 'KTOS', 'LUNR', 'JOBY',
    'RIOT', 'APLD', 'KEEL', 'RCAT', 'BBAI', 'ASTS', 'MARA',
    'CIFR', 'NVDA', 'BTBT', 'COIN', 'IREN', 'BE',
    'WULF', 'CLSK', 'HUT', 'ACHR', 'MOD', 'PLUG',
}
exp152_teachers = {
    'BBAI', 'RCAT', 'LUNR', 'ASTS', 'ACHR', 'MVIS',
    'WULF', 'CIFR', 'IREN', 'KEEL', 'HUT', 'CLSK', 'APLD',
}

for rank, (_, r) in enumerate(df_ranked.iterrows(), 1):
    t = r['ticker']
    flags = []
    if t in currently_deployed:
        flags.append('DEPLOYED')
    if t in exp152_teachers:
        flags.append('152-teacher')
    status = ', '.join(flags) if flags else ''
    print(f"  {rank:>4} {t:<7} {r['10pct_avg_rate']:>7.1%} {r['10pct_stability']:>5.3f} "
          f"{r['10pct_TEST']:>5.1%} {r['10pct_LIVE']:>5.1%} "
          f"{r['10pct_score']:>7.5f}  {status}")
    if rank >= 40:
        break

# Phase 4: Same ranking at 6% threshold (for space tickers that may not spike 10%)
print("\n" + "=" * 130)
print("  TABLE 3: TEACHER CANDIDATE RANKING (6% threshold)")
print("  Score = avg_spike_rate × stability × min(TEST, LIVE) rate")
print("=" * 130)

df_results['6pct_score'] = (
    df_results['6pct_avg_rate'] *
    df_results['6pct_stability'].clip(lower=0) *
    df_results[['6pct_TEST', '6pct_LIVE']].min(axis=1)
)
df_ranked6 = df_results.sort_values('6pct_score', ascending=False)

print(f"\n  {'Rank':>4} {'Ticker':<7} {'AvgRate':>8} {'Stab':>6} {'TEST':>6} {'LIVE':>6} "
      f"{'Score':>8}  {'10%Live':>7}  {'Status'}")
print(f"  {'─'*4} {'─'*7} {'─'*8} {'─'*6} {'─'*6} {'─'*6} {'─'*8}  {'─'*7}  {'─'*30}")

for rank, (_, r) in enumerate(df_ranked6.iterrows(), 1):
    t = r['ticker']
    flags = []
    if t in currently_deployed:
        flags.append('DEPLOYED')
    if t in exp152_teachers:
        flags.append('152-teacher')
    status = ', '.join(flags) if flags else ''
    print(f"  {rank:>4} {t:<7} {r['6pct_avg_rate']:>7.1%} {r['6pct_stability']:>5.3f} "
          f"{r['6pct_TEST']:>5.1%} {r['6pct_LIVE']:>5.1%} "
          f"{r['6pct_score']:>7.5f}  {r['10pct_LIVE']:>6.1%}  {status}")
    if rank >= 40:
        break

# Phase 5: Space-specific view
SPACE_TICKERS = {'RKLB', 'ASTS', 'LUNR', 'PL', 'RDW', 'BKSY', 'MNTS', 'VOYG',
                 'JOBY', 'ACHR', 'LMT', 'NOC', 'RTX', 'AVAV'}

print("\n" + "=" * 130)
print("  TABLE 4: SPACE & ADJACENT TICKERS DETAIL")
print("=" * 130)

space_df = df_results[df_results['ticker'].isin(SPACE_TICKERS)].sort_values(
    '10pct_LIVE', ascending=False)

print(f"\n  {'Ticker':<7}  "
      f"{'── 6% ──':^14}  {'── 8% ──':^14}  {'── 10% ──':^14}  "
      f"{'Live':>5} {'Live':>5} {'Live':>5}  {'Verdict'}")
print(f"  {'':7}  "
      f"{'TEST':>6} {'LIVE':>6}  {'TEST':>6} {'LIVE':>6}  {'TEST':>6} {'LIVE':>6}  "
      f"{'Med%':>5} {'P90%':>5} {'Max%':>5}  {'─'*20}")

for _, r in space_df.iterrows():
    t = r['ticker']
    live_10 = r.get('10pct_LIVE', 0)
    live_6 = r.get('6pct_LIVE', 0)
    if live_10 >= 0.10:
        verdict = "TEACHER (10%)"
    elif live_6 >= 0.15:
        verdict = "TEACHER (6%)"
    elif live_6 >= 0.05:
        verdict = "SCORING ONLY"
    else:
        verdict = "TOO QUIET"
    print(f"  {t:<7}  "
          f"{r.get('6pct_TEST', 0):>5.1%} {r.get('6pct_LIVE', 0):>5.1%}  "
          f"{r.get('8pct_TEST', 0):>5.1%} {r.get('8pct_LIVE', 0):>5.1%}  "
          f"{r.get('10pct_TEST', 0):>5.1%} {r.get('10pct_LIVE', 0):>5.1%}  "
          f"{r.get('live_med_mag', 0):>4.1%} {r.get('live_p90_mag', 0):>4.1%} "
          f"{r.get('live_max_mag', 0):>4.1%}  {verdict}")

print(f"\n{'='*130}")
print(f"  ANALYSIS COMPLETE — {len(results)} assets screened across {len(THRESHOLDS)} thresholds")
print(f"{'='*130}")
