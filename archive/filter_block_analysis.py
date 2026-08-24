"""
filter_block_analysis.py — Regime Trap Filter Historical Analysis
=================================================================
Checks how many trading days over the last year the VIX/TLT/SPY
trap filter would have blocked trades.

Trap logic (AND — all three must be true to block):
  1. VIX  > vix_threshold
  2. TLT gap from prev close < tlt_crash  (user: TLT < 0%)
  3. SPY gap from prev close > spy_ignore

Uses averaged thresholds across all 27 active scoring assets.

Usage:  python filter_block_analysis.py
"""

import pytz
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from google.cloud import storage as gcs_storage

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BUCKET_NAME = "time-series-bucket-ys"
GCS_INFERENCE_DIR = f"gs://{BUCKET_NAME}/Data/Inference"

# Average thresholds across all 27 active assets (Exp-97 + Exp-103)
#   Exp-97  (14): VIX avg=12.1, SPY avg=0.016
#   Exp-103 (13): VIX avg=11.9, SPY avg=0.046
#   Combined (27): VIX≈12, SPY≈0.03
VIX_THRESHOLD = 12.0
TLT_CRASH     = 0.00    # user: "TLT < 0" — any negative TLT gap
SPY_IGNORE    = 0.03    # averaged across 27 assets

# Entry window: trailing 30 min ending at 9:45 AM EST (6:45 AM PST)
# Covers most entry_minutes (360-390 = 6:00-6:30 AM PST)
ENTRY_HOUR_PST   = 6
ENTRY_MINUTE_PST = 45

LOOKBACK_DAYS = 365

PST = pytz.timezone('America/Los_Angeles')

gcs_client = gcs_storage.Client()

# ==============================================================================
# DATA LOADING
# ==============================================================================
def load_master(ticker):
    clean = ticker.replace("I:", "").replace("X:", "")
    uri = f"{GCS_INFERENCE_DIR}/{clean}_master_inference.parquet"
    df = pd.read_parquet(uri)
    df = df.reset_index()
    dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    if dt_cols:
        df['Datetime'] = pd.to_datetime(df[dt_cols[0]], utc=True)
    else:
        df['Datetime'] = pd.to_datetime(df[df.columns[0]], unit='ms', utc=True)
    df = df.set_index('Datetime').sort_index()
    df = df[~df.index.duplicated(keep='first')]
    col_map = {'open': 'Open', 'high': 'High', 'low': 'Low',
               'close': 'Close', 'volume': 'Volume'}
    df.columns = [col_map.get(str(c).strip().lower(), str(c).strip()) for c in df.columns]
    return df

# ==============================================================================
# ANALYSIS
# ==============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("  REGIME TRAP FILTER — HISTORICAL BLOCK ANALYSIS")
    print("=" * 80)
    print(f"  VIX threshold : > {VIX_THRESHOLD}")
    print(f"  TLT crash     : < {TLT_CRASH:+.2f}%  (any negative gap)")
    print(f"  SPY ignore    : > {SPY_IGNORE:+.2f}%")
    print(f"  Entry window  : trailing 30 min ending {ENTRY_HOUR_PST}:{ENTRY_MINUTE_PST:02d} AM PST")
    print(f"  Lookback      : {LOOKBACK_DAYS} days")
    print("=" * 80)

    # Load data
    print("\nLoading VIX, TLT, SPY...")
    vix_df = load_master('VIX')
    tlt_df = load_master('TLT')
    spy_df = load_master('SPY')
    print(f"  VIX: {len(vix_df):,} rows  ({vix_df.index.min().date()} to {vix_df.index.max().date()})")
    print(f"  TLT: {len(tlt_df):,} rows  ({tlt_df.index.min().date()} to {tlt_df.index.max().date()})")
    print(f"  SPY: {len(spy_df):,} rows  ({spy_df.index.min().date()} to {spy_df.index.max().date()})")

    # Compute TLT and SPY anchor ROC (gap from previous day's close)
    tlt_daily_close = tlt_df['Close'].resample('D').last().ffill().shift(1)
    tlt_anchor = tlt_daily_close.reindex(tlt_df.index, method='ffill')
    tlt_df['Anchor_ROC'] = ((tlt_df['Close'] - tlt_anchor) / tlt_anchor) * 100

    spy_daily_close = spy_df['Close'].resample('D').last().ffill().shift(1)
    spy_anchor = spy_daily_close.reindex(spy_df.index, method='ffill')
    spy_df['Anchor_ROC'] = ((spy_df['Close'] - spy_anchor) / spy_anchor) * 100

    # Convert to PST for windowing
    vix_pst = vix_df.copy()
    vix_pst.index = vix_pst.index.tz_convert(PST)
    tlt_pst = tlt_df.copy()
    tlt_pst.index = tlt_pst.index.tz_convert(PST)
    spy_pst = spy_df.copy()
    spy_pst.index = spy_pst.index.tz_convert(PST)

    # Get trading days from SPY
    cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=LOOKBACK_DAYS)
    spy_recent = spy_pst[spy_pst.index >= cutoff.tz_convert(PST)]
    trading_days = sorted(set(spy_recent.index.date))
    print(f"\n  Trading days in range: {len(trading_days)}")

    # Analyze each day
    results = []
    for day in trading_days:
        # Trailing 30-min window ending at entry time
        window_end = PST.localize(datetime(day.year, day.month, day.day,
                                            ENTRY_HOUR_PST, ENTRY_MINUTE_PST))
        window_start = window_end - timedelta(minutes=30)

        # VIX: trailing 30-min median
        vix_bars = vix_pst[(vix_pst.index >= window_start) & (vix_pst.index <= window_end)]
        vix_level = float(vix_bars['Close'].median()) if len(vix_bars) > 0 else None

        # TLT: trailing 30-min median of anchor ROC
        tlt_bars = tlt_pst[(tlt_pst.index >= window_start) & (tlt_pst.index <= window_end)]
        tlt_gap = float(tlt_bars['Anchor_ROC'].median()) if len(tlt_bars) > 0 else None

        # SPY: trailing 30-min median of anchor ROC
        spy_bars = spy_pst[(spy_pst.index >= window_start) & (spy_pst.index <= window_end)]
        spy_gap = float(spy_bars['Anchor_ROC'].median()) if len(spy_bars) > 0 else None

        vix_bad = vix_level is not None and vix_level > VIX_THRESHOLD
        tlt_bad = tlt_gap is not None and tlt_gap < TLT_CRASH
        spy_bad = spy_gap is not None and spy_gap > SPY_IGNORE

        is_trap = vix_bad and tlt_bad and spy_bad

        results.append({
            'date': day,
            'vix': vix_level,
            'tlt_gap': tlt_gap,
            'spy_gap': spy_gap,
            'vix_bad': vix_bad,
            'tlt_bad': tlt_bad,
            'spy_bad': spy_bad,
            'blocked': is_trap,
        })

    df_r = pd.DataFrame(results)
    valid = df_r.dropna(subset=['vix', 'tlt_gap', 'spy_gap'])

    total_days = len(valid)
    blocked_days = valid['blocked'].sum()
    vix_bad_days = valid['vix_bad'].sum()
    tlt_bad_days = valid['tlt_bad'].sum()
    spy_bad_days = valid['spy_bad'].sum()

    # Print summary
    print(f"\n{'=' * 80}")
    print(f"  RESULTS  ({total_days} trading days with complete data)")
    print(f"{'=' * 80}")
    print(f"\n  Individual condition frequency:")
    print(f"    VIX > {VIX_THRESHOLD:.0f}        :  {vix_bad_days:>3} / {total_days}  ({vix_bad_days/total_days*100:.1f}%)")
    print(f"    TLT gap < {TLT_CRASH:+.2f}%  :  {tlt_bad_days:>3} / {total_days}  ({tlt_bad_days/total_days*100:.1f}%)")
    print(f"    SPY gap > {SPY_IGNORE:+.2f}%  :  {spy_bad_days:>3} / {total_days}  ({spy_bad_days/total_days*100:.1f}%)")

    # Pairwise
    vix_tlt = (valid['vix_bad'] & valid['tlt_bad']).sum()
    vix_spy = (valid['vix_bad'] & valid['spy_bad']).sum()
    tlt_spy = (valid['tlt_bad'] & valid['spy_bad']).sum()
    print(f"\n  Pairwise (AND):")
    print(f"    VIX + TLT    :  {vix_tlt:>3} / {total_days}  ({vix_tlt/total_days*100:.1f}%)")
    print(f"    VIX + SPY    :  {vix_spy:>3} / {total_days}  ({vix_spy/total_days*100:.1f}%)")
    print(f"    TLT + SPY    :  {tlt_spy:>3} / {total_days}  ({tlt_spy/total_days*100:.1f}%)")

    # All three = TRAP BLOCKED
    print(f"\n  ╔{'═'*54}╗")
    print(f"  ║  TRAP BLOCKED (all 3):  {blocked_days:>3} / {total_days}  ({blocked_days/total_days*100:.1f}%)  "
          f"{'':>9}║")
    print(f"  ║  TRADE ALLOWED:         {total_days - blocked_days:>3} / {total_days}  "
          f"({(total_days - blocked_days)/total_days*100:.1f}%)  {'':>9}║")
    print(f"  ╚{'═'*54}╝")

    # Monthly breakdown
    print(f"\n  Monthly breakdown:")
    print(f"  {'Month':<10}  {'Days':>5}  {'Blocked':>7}  {'Block%':>7}  {'VIX>':>5}  {'TLT<':>5}  {'SPY>':>5}")
    print(f"  {'─'*55}")
    valid['month'] = pd.to_datetime(valid['date']).dt.to_period('M')
    for month, grp in valid.groupby('month'):
        n = len(grp)
        b = grp['blocked'].sum()
        v = grp['vix_bad'].sum()
        t = grp['tlt_bad'].sum()
        s = grp['spy_bad'].sum()
        print(f"  {str(month):<10}  {n:>5}  {b:>7}  {b/n*100:>6.1f}%  {v:>5}  {t:>5}  {s:>5}")

    # Show blocked dates
    blocked_dates = valid[valid['blocked']]
    if len(blocked_dates) > 0:
        print(f"\n  Blocked dates:")
        print(f"  {'Date':<12}  {'VIX':>6}  {'TLT gap':>8}  {'SPY gap':>8}")
        print(f"  {'─'*38}")
        for _, row in blocked_dates.iterrows():
            print(f"  {row['date']}  {row['vix']:>6.1f}  {row['tlt_gap']:>+7.2f}%  {row['spy_gap']:>+7.2f}%")

    print(f"\n{'=' * 80}")
