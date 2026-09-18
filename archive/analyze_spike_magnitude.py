"""
analyze_spike_magnitude.py — Spike magnitude sweep for optimal threshold selection
===================================================================================
Scans thresholds from 4% to 14% in 1% steps across all crypto cluster tickers.
For each threshold and period, computes spike rate, magnitude distribution, and
stability metrics. Helps select the optimal SPIKE_THRESHOLD for next training.

Key outputs:
  1. Per-threshold spike rates across TRAIN/OPT/TEST/LIVE for all tickers
  2. Stability score (how consistent is the spike rate across periods)
  3. Magnitude distribution of actual intraday moves (independent of threshold)
  4. "Sweet spot" analysis: threshold where signal is abundant AND stable
  5. Tradeable PnL estimate (magnitude - entry lag - slippage)

Usage:  python analyze_spike_magnitude.py
"""

import sys
import subprocess
import os
import io
import gc
import warnings

import numpy as np
import pandas as pd
import pytz
from datetime import datetime

warnings.filterwarnings('ignore')

from google.cloud import storage

# ==============================================================================
# CONFIGURATION
# ==============================================================================
TRAIN_TEACHERS = [
    'MARA', 'IREN', 'WULF', 'APLD', 'RIOT', 'CLSK', 'CORZ', 'MSTR',
    'COIN', 'BITF', 'HUT', 'CIFR', 'BTBT', 'HOOD',
]
SCORING_POOL_EXTRAS = ['SMCI', 'NVDA', 'VRT', 'AMD', 'PLTR', 'TSLA']
DEPLOY_TICKERS = ['CLSK', 'IREN', 'RIOT', 'APLD']

ALL_TICKERS = TRAIN_TEACHERS + SCORING_POOL_EXTRAS

TICKER_FILE_MAP = {'BITF': 'KEEL'}
BUCKET_NAME = os.environ.get("GCS_BUCKET", "")
INFERENCE_DIR = "Data/Inference"

DATA_START_DATE = "2022-01-01"
TRAIN_CUTOFF_DATE = "2025-06-30"
OPT_START = "2025-07-01"
OPT_END = "2025-12-31"
TEST_START = "2026-01-01"
TEST_END = "2026-03-31"
LIVE_START = "2026-04-01"
LIVE_END = datetime.now(pytz.UTC).strftime("%Y-%m-%d")

THRESHOLDS = [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14]

PERIODS = [
    ('TRAIN', DATA_START_DATE, TRAIN_CUTOFF_DATE),
    ('OPT',   OPT_START,       OPT_END),
    ('TEST',  TEST_START,      TEST_END),
    ('LIVE',  LIVE_START,      LIVE_END),
]

MICRO_COLS = ['Spread', 'OBI', 'Volume_Delta', 'Trade_Count', 'Delta_Pct',
              'bsize', 'asize', 'Total_Vol']
KEEP_COLS = ['Open', 'High', 'Low', 'Close', 'Volume']
PST = pytz.timezone('America/Los_Angeles')

ENTRY_LAG_PCT = 0.015
SLIPPAGE_PCT = 0.003

client = storage.Client()

# ==============================================================================
# DATA LOADING
# ==============================================================================
def load_master(ticker):
    clean = ticker.replace("I:", "").replace("X:", "")
    file_ticker = TICKER_FILE_MAP.get(clean, clean)
    blob = client.bucket(BUCKET_NAME).blob(
        f"{INFERENCE_DIR}/{file_ticker}_master_inference.parquet")
    if not blob.exists():
        return None
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
    std_map = {
        'open': 'Open', 'high': 'High', 'low': 'Low',
        'close': 'Close', 'volume': 'Volume',
    }
    lower_map = {k.lower(): v for k, v in std_map.items()}
    df.columns = [lower_map.get(str(c).strip().lower(), str(c).strip())
                  for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated(keep='first')].copy()
    df = df[df.index.notna()].sort_index()
    cutoff = pd.Timestamp(DATA_START_DATE, tz='UTC')
    df = df[df.index >= cutoff]
    keep = [c for c in KEEP_COLS if c in df.columns]
    df = df[keep]
    return df


def heal_data(df):
    df = df.resample('1min').asfreq()
    for c in ['Open', 'High', 'Low', 'Close']:
        if c in df.columns:
            df[c] = df[c].ffill()
    if 'Volume' in df.columns:
        df['Volume'] = df['Volume'].fillna(0)
    return df


# ==============================================================================
# COMPUTE FUTURE_MAX ONCE PER TICKER-PERIOD (reuse across thresholds)
# ==============================================================================
def compute_entry_bar_magnitudes(df, start_date, end_date):
    """Returns spike_pct Series for entry-window bars (the max % move available
    from each bar's close to end-of-day safe high). Threshold-independent."""
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
def main():
    print("=" * 90)
    print("  SPIKE MAGNITUDE SWEEP — Optimal Threshold Selection")
    print(f"  Thresholds: {[f'{t:.0%}' for t in THRESHOLDS]}")
    print(f"  Tickers: {len(ALL_TICKERS)} ({len(TRAIN_TEACHERS)} teachers + {len(SCORING_POOL_EXTRAS)} extras)")
    print(f"  LIVE period: {LIVE_START} → {LIVE_END}")
    print("=" * 90)

    # ------------------------------------------------------------------
    # PHASE 1: Compute magnitude distributions per ticker per period
    # ------------------------------------------------------------------
    # Store: {(ticker, period): spike_pct_series}
    all_magnitudes = {}

    print("\n[PHASE 1] Loading tickers and computing magnitude distributions...\n")
    for i, ticker in enumerate(ALL_TICKERS, 1):
        print(f"   [{i}/{len(ALL_TICKERS)}] {ticker}...", end=" ")
        df = load_master(ticker)
        if df is None:
            print("MISSING")
            continue
        df = heal_data(df)
        print(f"{len(df):,} bars", end="")

        for period_name, start, end in PERIODS:
            mags = compute_entry_bar_magnitudes(df, start, end)
            if mags is not None and len(mags) > 0:
                all_magnitudes[(ticker, period_name)] = mags

        print(f"  — {sum(1 for p in ['TRAIN','OPT','TEST','LIVE'] if (ticker,p) in all_magnitudes)}/4 periods")
        del df
        gc.collect()

    # ------------------------------------------------------------------
    # PHASE 2: Raw magnitude distribution (threshold-independent)
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("  [1/5] RAW MAGNITUDE DISTRIBUTION — Entry Window Bars")
    print("=" * 90)

    print(f"\n   What % of entry bars have moves >= X% (cumulative)?")
    print(f"\n   {'Ticker':<8} {'Period':<6} {'Bars':>7} {'>=4%':>6} {'>=5%':>6} {'>=6%':>6} "
          f"{'>=7%':>6} {'>=8%':>6} {'>=9%':>6} {'>=10%':>6} {'>=12%':>6} {'>=14%':>6} "
          f"{'Med%':>6} {'Mean%':>6} {'P75%':>6} {'P90%':>6}")
    print(f"   {'─'*8} {'─'*6} {'─'*7} {'─'*6} {'─'*6} {'─'*6} "
          f"{'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*6} "
          f"{'─'*6} {'─'*6} {'─'*6} {'─'*6}")

    for period_name in ['TRAIN', 'OPT', 'TEST', 'LIVE']:
        for ticker in TRAIN_TEACHERS:
            key = (ticker, period_name)
            if key not in all_magnitudes:
                continue
            mags = all_magnitudes[key]
            n = len(mags)
            pcts = {t: (mags >= t).mean() for t in [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.14]}
            d = '***' if ticker in DEPLOY_TICKERS else ''
            print(f"   {ticker:<8} {period_name:<6} {n:>7,} "
                  f"{pcts[0.04]:>5.1%} {pcts[0.05]:>5.1%} {pcts[0.06]:>5.1%} "
                  f"{pcts[0.07]:>5.1%} {pcts[0.08]:>5.1%} {pcts[0.09]:>5.1%} "
                  f"{pcts[0.10]:>5.1%} {pcts[0.12]:>5.1%} {pcts[0.14]:>5.1%} "
                  f"{mags.median():>5.1%} {mags.mean():>5.1%} "
                  f"{mags.quantile(0.75):>5.1%} {mags.quantile(0.90):>5.1%} {d}")
        print()

    # ------------------------------------------------------------------
    # PHASE 3: Threshold sweep — aggregate by period
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("  [2/5] THRESHOLD SWEEP — Teachers Aggregate by Period")
    print("=" * 90)

    print(f"\n   Average spike rate across all {len(TRAIN_TEACHERS)} teachers:")
    print(f"\n   {'Threshold':>10} {'TRAIN':>8} {'OPT':>8} {'TEST':>8} {'LIVE':>8} "
          f"{'OPT/TRN':>8} {'TST/OPT':>8} {'LIV/TST':>8} {'Stability':>10}")
    print(f"   {'─'*10} {'─'*8} {'─'*8} {'─'*8} {'─'*8} "
          f"{'─'*8} {'─'*8} {'─'*8} {'─'*10}")

    sweep_results = []
    for thresh in THRESHOLDS:
        period_rates = {}
        for period_name in ['TRAIN', 'OPT', 'TEST', 'LIVE']:
            rates = []
            for ticker in TRAIN_TEACHERS:
                key = (ticker, period_name)
                if key not in all_magnitudes:
                    continue
                mags = all_magnitudes[key]
                rate = (mags >= thresh).mean()
                rates.append(rate)
            period_rates[period_name] = np.mean(rates) if rates else 0

        trn = period_rates.get('TRAIN', 0)
        opt = period_rates.get('OPT', 0)
        tst = period_rates.get('TEST', 0)
        liv = period_rates.get('LIVE', 0)

        opt_trn = opt / trn if trn > 0 else np.nan
        tst_opt = tst / opt if opt > 0 else np.nan
        liv_tst = liv / tst if tst > 0 else np.nan

        rates_list = [trn, opt, tst, liv]
        stability = 1.0 - (np.std(rates_list) / np.mean(rates_list)) if np.mean(rates_list) > 0 else 0

        sweep_results.append({
            'threshold': thresh, 'TRAIN': trn, 'OPT': opt, 'TEST': tst, 'LIVE': liv,
            'opt_trn': opt_trn, 'tst_opt': tst_opt, 'liv_tst': liv_tst,
            'stability': stability,
        })

        print(f"   {thresh:>9.0%} {trn:>7.1%} {opt:>7.1%} {tst:>7.1%} {liv:>7.1%} "
              f"{opt_trn:>7.2f} {tst_opt:>7.2f} {liv_tst:>7.2f} "
              f"{stability:>9.3f}")

    # ------------------------------------------------------------------
    # PHASE 4: Per-ticker threshold sweep (deployed tickers only)
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("  [3/5] DEPLOYED TICKER DETAIL — Spike Rate by Threshold")
    print("=" * 90)

    for ticker in DEPLOY_TICKERS:
        print(f"\n   ── {ticker} ──")
        print(f"   {'Threshold':>10} {'TRAIN':>8} {'OPT':>8} {'TEST':>8} {'LIVE':>8} "
              f"{'OPT/TRN':>8} {'LIV/TST':>8} {'Stab':>6}")
        print(f"   {'─'*10} {'─'*8} {'─'*8} {'─'*8} {'─'*8} "
              f"{'─'*8} {'─'*8} {'─'*6}")

        for thresh in THRESHOLDS:
            rates = {}
            for period_name in ['TRAIN', 'OPT', 'TEST', 'LIVE']:
                key = (ticker, period_name)
                if key in all_magnitudes:
                    rates[period_name] = (all_magnitudes[key] >= thresh).mean()
                else:
                    rates[period_name] = 0

            trn, opt, tst, liv = rates['TRAIN'], rates['OPT'], rates['TEST'], rates['LIVE']
            opt_trn = opt / trn if trn > 0 else np.nan
            liv_tst = liv / tst if tst > 0 else np.nan
            rl = [trn, opt, tst, liv]
            stab = 1.0 - (np.std(rl) / np.mean(rl)) if np.mean(rl) > 0 else 0

            marker = ' ◄' if stab > 0.85 and liv > 0.05 else ''
            print(f"   {thresh:>9.0%} {trn:>7.1%} {opt:>7.1%} {tst:>7.1%} {liv:>7.1%} "
                  f"{opt_trn:>7.2f} {liv_tst:>7.2f} {stab:>5.3f}{marker}")

    # ------------------------------------------------------------------
    # PHASE 5: Magnitude distribution of actual spikes
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("  [4/5] ACTUAL MOVE DISTRIBUTION — All Entry Bars (Teachers, All Periods)")
    print("=" * 90)

    for period_name in ['TRAIN', 'OPT', 'TEST', 'LIVE']:
        all_mags = []
        for ticker in TRAIN_TEACHERS:
            key = (ticker, period_name)
            if key in all_magnitudes:
                all_mags.append(all_magnitudes[key])
        if not all_mags:
            continue
        combined = pd.concat(all_mags)
        nonzero = combined[combined > 0.001]

        print(f"\n   --- {period_name} Period ---")
        print(f"   Total entry bars: {len(combined):,}  |  Bars with any move >0.1%: {len(nonzero):,} ({len(nonzero)/len(combined):.1%})")
        print(f"   Percentiles of move magnitude (bars with move > 0):")
        for p in [10, 25, 50, 75, 90, 95, 99]:
            val = nonzero.quantile(p / 100)
            bar = '█' * int(val * 200)
            print(f"      P{p:<2}: {val:>6.2%}  {bar}")
        print(f"      Mean: {nonzero.mean():>5.2%}  |  Std: {nonzero.std():>5.2%}  |  Max: {nonzero.max():>5.2%}")

        print(f"\n   Histogram (1% buckets):")
        for lo in range(0, 20):
            hi = lo + 1
            count = ((nonzero >= lo/100) & (nonzero < hi/100)).sum()
            pct = count / len(nonzero) * 100 if len(nonzero) > 0 else 0
            bar = '█' * int(pct * 2)
            print(f"      {lo:>2}-{hi:>2}%: {count:>7,} ({pct:>5.1f}%)  {bar}")
        above_20 = (nonzero >= 0.20).sum()
        pct_20 = above_20 / len(nonzero) * 100 if len(nonzero) > 0 else 0
        print(f"      >=20%: {above_20:>7,} ({pct_20:>5.1f}%)")

    # ------------------------------------------------------------------
    # PHASE 6: Tradeable PnL estimate
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("  [5/5] ESTIMATED TRADEABLE PnL BY THRESHOLD")
    print("=" * 90)

    print(f"\n   Assumptions: entry lag = {ENTRY_LAG_PCT:.1%} of move lost,")
    print(f"   round-trip slippage+fees = {SLIPPAGE_PCT:.1%}")
    print(f"   Capture ratio = magnitude - entry_lag - slippage")
    print(f"\n   Only counts bars where move >= threshold (i.e., model is 'correct')")

    print(f"\n   {'Threshold':>10} {'Period':<6} {'SpikeRate':>10} {'MedMag':>8} {'MeanMag':>8} "
          f"{'MedCapture':>10} {'MeanCapture':>11} {'TotalEdge':>10}")
    print(f"   {'─'*10} {'─'*6} {'─'*10} {'─'*8} {'─'*8} "
          f"{'─'*10} {'─'*11} {'─'*10}")

    for thresh in [0.06, 0.07, 0.08, 0.09, 0.10]:
        for period_name in ['TRAIN', 'OPT', 'TEST', 'LIVE']:
            all_mags = []
            all_rates = []
            for ticker in TRAIN_TEACHERS:
                key = (ticker, period_name)
                if key not in all_magnitudes:
                    continue
                mags = all_magnitudes[key]
                spike_mags = mags[mags >= thresh]
                all_mags.append(spike_mags)
                all_rates.append((mags >= thresh).mean())

            if not all_mags or not any(len(m) > 0 for m in all_mags):
                continue
            combined = pd.concat([m for m in all_mags if len(m) > 0])
            if combined.empty:
                continue

            avg_rate = np.mean(all_rates)
            med_mag = combined.median()
            mean_mag = combined.mean()
            med_capture = med_mag - ENTRY_LAG_PCT - SLIPPAGE_PCT
            mean_capture = mean_mag - ENTRY_LAG_PCT - SLIPPAGE_PCT
            total_edge = avg_rate * mean_capture

            print(f"   {thresh:>9.0%} {period_name:<6} {avg_rate:>9.1%} "
                  f"{med_mag:>7.1%} {mean_mag:>7.1%} "
                  f"{med_capture:>9.1%} {mean_capture:>10.1%} "
                  f"{total_edge:>9.3%}")
        print()

    # ------------------------------------------------------------------
    # RECOMMENDATION
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("  RECOMMENDATION SUMMARY")
    print("=" * 90)

    best = max(sweep_results, key=lambda x: x['stability'] * min(x['LIVE'], x['TEST']))
    print(f"\n   Best stability × LIVE signal: {best['threshold']:.0%}")
    print(f"      TRAIN={best['TRAIN']:.1%}  OPT={best['OPT']:.1%}  "
          f"TEST={best['TEST']:.1%}  LIVE={best['LIVE']:.1%}")
    print(f"      Stability score: {best['stability']:.3f}")

    print(f"\n   All thresholds ranked by (stability × min(TEST, LIVE) rate):")
    ranked = sorted(sweep_results,
                    key=lambda x: x['stability'] * min(x.get('LIVE', 0), x.get('TEST', 0)),
                    reverse=True)
    for r in ranked:
        score = r['stability'] * min(r.get('LIVE', 0), r.get('TEST', 0))
        marker = ' ◄◄◄' if r == best else ''
        print(f"      {r['threshold']:>4.0%}: stability={r['stability']:.3f}  "
              f"TEST={r['TEST']:.1%}  LIVE={r['LIVE']:.1%}  "
              f"score={score:.4f}{marker}")

    print(f"\n" + "=" * 90)
    print(f"  ANALYSIS COMPLETE")
    print(f"=" * 90)


if __name__ == "__main__":
    main()
