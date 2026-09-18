"""
analyze_spike_rates.py — Spike Rate Analysis for v2 Teacher Selection
=====================================================================
Scans ALL tickers with T&Q data, computes spike rates at multiple
thresholds (6/8/10/12/15%), and ranks candidates for teacher selection.

Spike definition (same as exp_170 add_targets):
  Future_Max = max High from next bar through 1:00 PM PST
  Target = 1 if Future_Max > Close * (1 + threshold)
  Valid_Row = bars at 6:00-6:30 AM PST with Volume > 0

Run in Colab. No model needed — pure price analysis.
"""

import numpy as np
import pandas as pd
import pytz
from datetime import datetime
from google.cloud import storage as gcs_storage

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass

# ==============================================================================
# CONFIG
# ==============================================================================
BUCKET_NAME = os.environ.get("GCS_BUCKET", "")
OHLCV_DIR = "Data/Inference"
TQ_DIR = "Data/TradesQuotes"
PST = pytz.timezone('America/Los_Angeles')

SPIKE_THRESHOLDS = [0.06, 0.08, 0.10, 0.12, 0.15]

EXCLUDE_TICKERS = {'SPY', 'VIX', 'TLT', 'IWM', 'SMH', 'ITA', 'BTCUSD', 'XOM',
                    'XLE', 'IWM', 'QQQ', 'HYG', 'USO', 'SLV'}

TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}
TICKER_DISPLAY_MAP = {v: v for v in TICKER_FILE_MAP.values()}

CURRENT_TEACHERS_170 = {
    'BBAI', 'RCAT', 'LUNR', 'ASTS', 'ACHR', 'MVIS',
    'WULF', 'CIFR', 'IREN', 'KEEL', 'HUT', 'CLSK', 'APLD',
}
CURRENT_SCORING_POOL = {
    'MARA', 'RIOT', 'MSTR', 'COIN', 'SMCI', 'RKLB', 'BE',
    'FCEL', 'FLNC', 'PLUG', 'MP', 'POWL', 'MOD', 'ALAB',
    'OKLO', 'SMR',
}
DEPLOYED_170 = {
    'RCAT', 'KEEL', 'PLUG', 'RIOT', 'IREN',
    'ALAB', 'APLD', 'ASTS', 'BBAI', 'BE', 'BTBT', 'CIFR',
    'CLSK', 'FCEL', 'HUT', 'JOBY', 'LUNR',
}

SECTORS = {
    'crypto':      ['RIOT', 'CLSK', 'HUT', 'MARA', 'BTBT', 'CIFR', 'IREN', 'APLD',
                    'CORZ', 'COIN', 'MSTR', 'KEEL', 'WULF', 'HOOD'],
    'defense':     ['LUNR', 'RKLB', 'KTOS', 'AVAV', 'LMT', 'NOC', 'RTX', 'LHX',
                    'GD', 'BKSY', 'PL', 'RDW', 'MNTS', 'NNDM', 'SPIR'],
    'energy':      ['FCEL', 'PLUG', 'BE', 'FLNC', 'SMR', 'OKLO', 'CCJ', 'CEG',
                    'NEE', 'DUK'],
    'space_evtol': ['ASTS', 'ACHR', 'JOBY', 'RCAT', 'BBAI', 'SMLR'],
    'nuclear':     ['SMR', 'OKLO', 'CCJ', 'CEG', 'NXE', 'NBIS'],
    'tech':        ['NVDA', 'AMD', 'PLTR', 'SMCI', 'TSLA', 'VRT', 'VST', 'MU',
                    'GEV', 'POWL', 'MOD', 'MP', 'ALAB', 'MVIS'],
}
SECTOR_LOOKUP = {}
for sector, tickers in SECTORS.items():
    for t in tickers:
        if t not in SECTOR_LOOKUP:
            SECTOR_LOOKUP[t] = sector

RECENT_12M = pd.Timestamp('2025-06-01', tz='UTC')
RECENT_6M = pd.Timestamp('2025-12-01', tz='UTC')

W = 140

# ==============================================================================
# T&Q TICKER UNIVERSE (from gsutil ls gs://{GCS_BUCKET}/Data/TradesQuotes/)
# ==============================================================================
TQ_RAW = [
    'ACHR', 'ALAB', 'AMD', 'APLD', 'ASTS', 'AVAV', 'BBAI', 'BE', 'BGDE',
    'BHP', 'BKSY', 'BTBT', 'CAT', 'CCJ', 'CEG', 'CIFR', 'CLSK', 'COIN',
    'COP', 'CORZ', 'CVX', 'DUK', 'EQIX', 'ETN', 'FCEL', 'FCX', 'FLNC',
    'GD', 'GE', 'GEV', 'HOOD', 'HUT', 'IREN', 'JOBY', 'KEEL', 'KTOS',
    'LHX', 'LIN', 'LMT', 'LUNR', 'MARA', 'MNTS', 'MOD', 'MP', 'MSTR',
    'MU', 'MVIS', 'NBIS', 'NEE', 'NNDM', 'NOC', 'NVDA', 'OKLO', 'PL',
    'PLTR', 'PLUG', 'POWL', 'RCAT', 'RDW', 'RIOT', 'RKLB', 'RTX', 'SMCI',
    'SMLR', 'SMR', 'SOUN', 'SPIR', 'TSLA', 'VOYG', 'VRT', 'VST', 'WULF',
]

tq_tickers = sorted([t for t in TQ_RAW if t not in EXCLUDE_TICKERS])
print(f"T&Q universe: {len(tq_tickers)} tickers (excl. indices/context)")


# ==============================================================================
# SPIKE CALCULATION
# ==============================================================================
def compute_spike_stats(ticker, thresholds=SPIKE_THRESHOLDS):
    file_ticker = ticker
    for orig, mapped in TICKER_FILE_MAP.items():
        if mapped == ticker:
            file_ticker = ticker
            break

    uri = f"gs://{BUCKET_NAME}/{OHLCV_DIR}/{file_ticker}_1m_full_historical_data.parquet"
    try:
        df = pd.read_parquet(uri)
    except Exception:
        return None

    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df = df.sort_index()

    for c in ['Open', 'High', 'Low', 'Close']:
        if c in df.columns:
            df[c] = df[c].ffill()
    if 'Volume' not in df.columns:
        df['Volume'] = 0
    df['Volume'] = df['Volume'].fillna(0)

    df_pst = df.index.tz_convert(PST)
    mins = df_pst.hour * 60 + df_pst.minute
    is_safe = (df_pst.hour < 13) | ((df_pst.hour == 13) & (df_pst.minute == 0))
    is_real = df['Volume'] > 0

    df['Safe_High'] = np.where(is_safe & is_real, df['High'], -1.0)
    df['Future_Max'] = df.groupby(df.index.date)['Safe_High'].transform(
        lambda x: x.shift(-1).fillna(-1.0).iloc[::-1].cummax().iloc[::-1])

    df['Valid_Row'] = ((mins >= 360) & (mins <= 390) & (df['Volume'] > 0)).astype(int)
    valid = df[df['Valid_Row'] == 1].copy()

    if len(valid) == 0:
        return None

    close_safe = valid['Close'].replace(0, np.nan)
    max_gain = ((valid['Future_Max'] / close_safe) - 1).clip(lower=0).fillna(0)

    first_date = valid.index.min()
    last_date = valid.index.max()
    trading_days = valid.index.normalize().nunique()

    result = {
        'ticker': ticker,
        'first_date': first_date,
        'last_date': last_date,
        'trading_days': trading_days,
        'sector': SECTOR_LOOKUP.get(ticker, 'other'),
    }

    for thr in thresholds:
        key = f"spike_{int(thr*100)}"
        spiked = (max_gain >= thr)
        spike_days = spiked.groupby(spiked.index.date).any().sum()
        result[f'{key}_days'] = int(spike_days)
        result[f'{key}_rate'] = spike_days / trading_days if trading_days > 0 else 0

        recent_12 = valid[valid.index >= RECENT_12M]
        if len(recent_12) > 0:
            mg_12 = ((recent_12['Future_Max'] / recent_12['Close'].replace(0, np.nan)) - 1).clip(lower=0).fillna(0)
            s12 = (mg_12 >= thr)
            days_12 = recent_12.index.normalize().nunique()
            spike_days_12 = s12.groupby(s12.index.date).any().sum()
            result[f'{key}_rate_12m'] = spike_days_12 / days_12 if days_12 > 0 else 0
            result[f'{key}_days_12m'] = int(spike_days_12)
        else:
            result[f'{key}_rate_12m'] = 0
            result[f'{key}_days_12m'] = 0

        recent_6 = valid[valid.index >= RECENT_6M]
        if len(recent_6) > 0:
            mg_6 = ((recent_6['Future_Max'] / recent_6['Close'].replace(0, np.nan)) - 1).clip(lower=0).fillna(0)
            s6 = (mg_6 >= thr)
            days_6 = recent_6.index.normalize().nunique()
            spike_days_6 = s6.groupby(s6.index.date).any().sum()
            result[f'{key}_rate_6m'] = spike_days_6 / days_6 if days_6 > 0 else 0
        else:
            result[f'{key}_rate_6m'] = 0

    result['avg_max_gain'] = float(max_gain.mean())
    result['med_max_gain'] = float(max_gain.median())
    result['p90_max_gain'] = float(max_gain.quantile(0.90))
    result['p95_max_gain'] = float(max_gain.quantile(0.95))

    return result


# ==============================================================================
# MAIN ANALYSIS
# ==============================================================================
print(f"\nAnalyzing spike rates for {len(tq_tickers)} tickers...")
print(f"Thresholds: {[f'{t:.0%}' for t in SPIKE_THRESHOLDS]}")
print(f"Recent 12m cutoff: {RECENT_12M.date()}")
print(f"Recent 6m cutoff:  {RECENT_6M.date()}")
print()

results = []
for i, ticker in enumerate(tq_tickers):
    r = compute_spike_stats(ticker)
    if r is not None:
        results.append(r)
        sr8 = r['spike_8_rate']
        sr8r = r['spike_8_rate_12m']
        print(f"  [{i+1:2d}/{len(tq_tickers)}] {ticker:<6}  days={r['trading_days']:>5}  "
              f"8%={sr8:>5.1%}  8%_12m={sr8r:>5.1%}  "
              f"10%={r['spike_10_rate']:>5.1%}  sector={r['sector']}")
    else:
        print(f"  [{i+1:2d}/{len(tq_tickers)}] {ticker:<6}  NO OHLCV DATA")

df = pd.DataFrame(results)
print(f"\nLoaded {len(df)} tickers with OHLCV data")

if df.empty:
    print("No data loaded. Check GCS access and OHLCV file paths.")
    raise SystemExit


# ==============================================================================
# TABLE 1: SPIKE RATE DASHBOARD
# ==============================================================================
df = df.sort_values('spike_8_rate_12m', ascending=False)

print(f"\n{'='*W}")
print(f"  TABLE 1: SPIKE RATE DASHBOARD — All tickers with T&Q data")
print(f"  Sorted by recent 12m spike rate at 8% threshold")
print(f"{'='*W}")
print(f"  {'Ticker':<6} {'Sector':<10} {'History':>10} {'Days':>5} "
      f"{'|':>2} {'6%':>5} {'8%':>5} {'10%':>5} {'12%':>5} {'15%':>5} "
      f"{'|':>2} {'6%r':>5} {'8%r':>5} {'10%r':>5} "
      f"{'|':>2} {'6%6m':>5} {'8%6m':>5} {'10%6m':>5} "
      f"{'|':>2} {'Trnd':>4}  {'Role'}")
print(f"  {'─'*6} {'─'*10} {'─'*10} {'─'*5} "
      f"{'─'*2} {'─'*5} {'─'*5} {'─'*5} {'─'*5} {'─'*5} "
      f"{'─'*2} {'─'*5} {'─'*5} {'─'*5} "
      f"{'─'*2} {'─'*5} {'─'*5} {'─'*5} "
      f"{'─'*2} {'─'*4}  {'─'*8}")

for _, r in df.iterrows():
    ticker = r['ticker']
    first = r['first_date'].strftime('%Y-%m') if pd.notna(r['first_date']) else '??'

    full_8 = r['spike_8_rate']
    recent_8 = r['spike_8_rate_12m']
    if full_8 > 0:
        ratio = recent_8 / full_8
        if ratio > 1.15:
            trend = '↑'
        elif ratio < 0.85:
            trend = '↓'
        else:
            trend = '→'
    else:
        trend = '?'

    if ticker in CURRENT_TEACHERS_170:
        role = 'TEACH'
    elif ticker in DEPLOYED_170:
        role = 'DEPLOY'
    elif ticker in CURRENT_SCORING_POOL:
        role = 'SCORE'
    else:
        role = ''

    print(f"  {ticker:<6} {r['sector']:<10} {first:>10} {r['trading_days']:>5} "
          f"{'|':>2} {r['spike_6_rate']:>4.0%} {r['spike_8_rate']:>4.0%} "
          f"{r['spike_10_rate']:>4.0%} {r['spike_12_rate']:>4.0%} {r['spike_15_rate']:>4.0%} "
          f"{'|':>2} {r['spike_6_rate_12m']:>4.0%} {r['spike_8_rate_12m']:>4.0%} "
          f"{r['spike_10_rate_12m']:>4.0%} "
          f"{'|':>2} {r['spike_6_rate_6m']:>4.0%} {r['spike_8_rate_6m']:>4.0%} "
          f"{r['spike_10_rate_6m']:>4.0%} "
          f"{'|':>2} {trend:>4}  {role}")


# ==============================================================================
# TABLE 2: TEACHER CANDIDATE RANKING
# ==============================================================================
MIN_SPIKE_RATE_8 = 0.05
MIN_TRADING_DAYS = 200

candidates = df[
    (df['spike_8_rate_12m'] >= MIN_SPIKE_RATE_8) &
    (df['trading_days'] >= MIN_TRADING_DAYS)
].copy()
candidates = candidates.sort_values('spike_8_rate_12m', ascending=False)

print(f"\n{'='*W}")
print(f"  TABLE 2: TEACHER CANDIDATES (recent 12m 8% spike rate >= {MIN_SPIKE_RATE_8:.0%}, "
      f"min {MIN_TRADING_DAYS} trading days)")
print(f"  These tickers have sufficient spike frequency for teacher candidacy")
print(f"{'='*W}")
print(f"  {'#':>2} {'Ticker':<6} {'Sector':<10} {'Days':>5} "
      f"{'|':>2} {'8%_all':>6} {'8%_12m':>6} {'8%_6m':>6} "
      f"{'|':>2} {'10%_12m':>7} {'6%_12m':>6} "
      f"{'|':>2} {'AvgGain':>7} {'P95Gain':>7} "
      f"{'|':>2} {'Trend':>5} {'Role':>6} {'Spike_d/yr':>10}")
print(f"  {'─'*2} {'─'*6} {'─'*10} {'─'*5} "
      f"{'─'*2} {'─'*6} {'─'*6} {'─'*6} "
      f"{'─'*2} {'─'*7} {'─'*6} "
      f"{'─'*2} {'─'*7} {'─'*7} "
      f"{'─'*2} {'─'*5} {'─'*6} {'─'*10}")

for rank, (_, r) in enumerate(candidates.iterrows(), 1):
    ticker = r['ticker']
    full_8 = r['spike_8_rate']
    recent_8 = r['spike_8_rate_12m']
    ratio = recent_8 / full_8 if full_8 > 0 else 0
    trend = '↑' if ratio > 1.15 else ('↓' if ratio < 0.85 else '→')

    if ticker in CURRENT_TEACHERS_170:
        role = 'TEACH'
    elif ticker in DEPLOYED_170:
        role = 'DEPLOY'
    elif ticker in CURRENT_SCORING_POOL:
        role = 'SCORE'
    else:
        role = 'NEW'

    spike_days_per_year = r['spike_8_days_12m']

    print(f"  {rank:>2} {ticker:<6} {r['sector']:<10} {r['trading_days']:>5} "
          f"{'|':>2} {r['spike_8_rate']:>5.1%} {r['spike_8_rate_12m']:>5.1%} "
          f"{r['spike_8_rate_6m']:>5.1%} "
          f"{'|':>2} {r['spike_10_rate_12m']:>6.1%} {r['spike_6_rate_12m']:>5.1%} "
          f"{'|':>2} {r['avg_max_gain']:>6.1%} {r['p95_max_gain']:>6.1%} "
          f"{'|':>2} {trend:>5} {role:>6} {spike_days_per_year:>10}")


# ==============================================================================
# TABLE 3: CURRENT 170 TEACHERS vs DATA-DRIVEN SELECTION
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 3: CURRENT 170 TEACHERS vs DATA-DRIVEN ANALYSIS")
print(f"{'='*W}")

print(f"\n  CURRENT TEACHERS (13):")
print(f"  {'Ticker':<6} {'Sector':<10} {'8%_all':>6} {'8%_12m':>6} {'10%_12m':>7} {'Trend':>5} {'Deploy?':>7} {'Verdict'}")
print(f"  {'─'*6} {'─'*10} {'─'*6} {'─'*6} {'─'*7} {'─'*5} {'─'*7} {'─'*12}")

for ticker in sorted(CURRENT_TEACHERS_170):
    row = df[df['ticker'] == ticker]
    if row.empty:
        print(f"  {ticker:<6} {'??':<10} {'N/A':>6} {'N/A':>6} {'N/A':>7} {'?':>5} "
              f"{'Y' if ticker in DEPLOYED_170 else 'N':>7} MISSING DATA")
        continue
    r = row.iloc[0]
    full_8 = r['spike_8_rate']
    recent_8 = r['spike_8_rate_12m']
    ratio = recent_8 / full_8 if full_8 > 0 else 0
    trend = '↑' if ratio > 1.15 else ('↓' if ratio < 0.85 else '→')
    deployed = ticker in DEPLOYED_170

    if recent_8 >= 0.10:
        verdict = 'KEEP'
    elif recent_8 >= 0.05:
        verdict = 'BORDERLINE'
    else:
        verdict = 'DROP'

    print(f"  {ticker:<6} {r['sector']:<10} {full_8:>5.1%} {recent_8:>5.1%} "
          f"{r['spike_10_rate_12m']:>6.1%} {trend:>5} "
          f"{'Y' if deployed else 'N':>7} {verdict}")

print(f"\n  NOT TEACHERS BUT DEPLOYED (scoring pool → deployed):")
deployed_non_teachers = DEPLOYED_170 - CURRENT_TEACHERS_170
print(f"  {'Ticker':<6} {'Sector':<10} {'8%_all':>6} {'8%_12m':>6} {'10%_12m':>7} {'Trend':>5} {'Verdict'}")
print(f"  {'─'*6} {'─'*10} {'─'*6} {'─'*6} {'─'*7} {'─'*5} {'─'*12}")
for ticker in sorted(deployed_non_teachers):
    row = df[df['ticker'] == ticker]
    if row.empty:
        print(f"  {ticker:<6} {'??':<10} {'N/A':>6} {'N/A':>6} {'N/A':>7} {'?':>5} MISSING DATA")
        continue
    r = row.iloc[0]
    full_8 = r['spike_8_rate']
    recent_8 = r['spike_8_rate_12m']
    ratio = recent_8 / full_8 if full_8 > 0 else 0
    trend = '↑' if ratio > 1.15 else ('↓' if ratio < 0.85 else '→')
    verdict = 'PROMOTE' if recent_8 >= 0.05 else 'KEEP POOL'
    print(f"  {ticker:<6} {r['sector']:<10} {full_8:>5.1%} {recent_8:>5.1%} "
          f"{r['spike_10_rate_12m']:>6.1%} {trend:>5} {verdict}")

print(f"\n  NEW CANDIDATES (not in 170 teachers or scoring pool, 8%_12m >= 5%):")
new_candidates = candidates[
    ~candidates['ticker'].isin(CURRENT_TEACHERS_170 | CURRENT_SCORING_POOL)
]
if not new_candidates.empty:
    print(f"  {'Ticker':<6} {'Sector':<10} {'8%_all':>6} {'8%_12m':>6} {'10%_12m':>7} {'Trend':>5}")
    print(f"  {'─'*6} {'─'*10} {'─'*6} {'─'*6} {'─'*7} {'─'*5}")
    for _, r in new_candidates.iterrows():
        full_8 = r['spike_8_rate']
        recent_8 = r['spike_8_rate_12m']
        ratio = recent_8 / full_8 if full_8 > 0 else 0
        trend = '↑' if ratio > 1.15 else ('↓' if ratio < 0.85 else '→')
        print(f"  {r['ticker']:<6} {r['sector']:<10} {full_8:>5.1%} {recent_8:>5.1%} "
              f"{r['spike_10_rate_12m']:>6.1%} {trend:>5}")
else:
    print("  None")


# ==============================================================================
# SECTOR SUMMARY
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 4: SECTOR SUMMARY (median spike rates, recent 12m)")
print(f"{'='*W}")
print(f"  {'Sector':<12} {'Tickers':>7} {'Med 6%':>6} {'Med 8%':>6} {'Med 10%':>7} "
      f"{'Med 12%':>7} {'Top ticker':>12} {'Top 8%':>6}")
print(f"  {'─'*12} {'─'*7} {'─'*6} {'─'*6} {'─'*7} {'─'*7} {'─'*12} {'─'*6}")

for sector in sorted(SECTORS.keys()):
    sec_df = df[df['sector'] == sector]
    if sec_df.empty:
        continue
    top = sec_df.loc[sec_df['spike_8_rate_12m'].idxmax()]
    print(f"  {sector:<12} {len(sec_df):>7} {sec_df['spike_6_rate_12m'].median():>5.1%} "
          f"{sec_df['spike_8_rate_12m'].median():>5.1%} "
          f"{sec_df['spike_10_rate_12m'].median():>6.1%} "
          f"{sec_df['spike_12_rate_12m'].median():>6.1%} "
          f"{top['ticker']:>12} {top['spike_8_rate_12m']:>5.1%}")

other_df = df[df['sector'] == 'other']
if not other_df.empty:
    top = other_df.loc[other_df['spike_8_rate_12m'].idxmax()]
    print(f"  {'other':<12} {len(other_df):>7} {other_df['spike_6_rate_12m'].median():>5.1%} "
          f"{other_df['spike_8_rate_12m'].median():>5.1%} "
          f"{other_df['spike_10_rate_12m'].median():>6.1%} "
          f"{other_df['spike_12_rate_12m'].median():>6.1%} "
          f"{top['ticker']:>12} {top['spike_8_rate_12m']:>5.1%}")

print(f"\n{'='*W}")
print(f"  ANALYSIS COMPLETE — {len(df)} tickers analyzed, {len(candidates)} teacher candidates")
print(f"{'='*W}")
