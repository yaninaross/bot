# ==============================================================================
# Entry Time Analysis: 6:30 AM vs 7:00 AM PT for SHORT Dump Opportunities
# ==============================================================================
# Loads OHLCV from master_inference parquets (GCS), computes forward returns
# at multiple horizons (3-60 min) from 6:30 and 7:00 AM PT entries, and
# analyzes dump frequency, magnitude, and price shape around market open.
#
# Focus: 6-8% downward dump detection opportunities for SHORT bets.
#
# Colab-ready. No GPU needed.

import os, gc, time, warnings
import numpy as np
import pandas as pd
import pytz
from collections import defaultdict

warnings.filterwarnings('ignore')

import matplotlib
try:
    from google.colab import auth
    _IN_COLAB = True
    auth.authenticate_user()
except (ImportError, Exception):
    matplotlib.use('Agg')
    _IN_COLAB = False
import matplotlib.pyplot as plt

from google.cloud import storage

# -- CONFIG --------------------------------------------------------------------

BUCKET_NAME = 'time-series-bucket-ys'
INFERENCE_DIR = 'Data/Inference_v2'
PST = pytz.timezone('America/Los_Angeles')

STUDY_START = '2026-03-09'
STUDY_END   = '2026-08-08'

ENTRY_MINUTES = {
    '6:30 AM': 390,   # 6*60 + 30 = 390
    '7:00 AM': 420,   # 7*60 + 0  = 420
}

FWD_HORIZONS = [3, 5, 10, 15, 20, 30, 45, 60]

DUMP_THRESHOLDS = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10]

EXCLUDE_TICKERS = {'SPY', 'VIX', 'TLT', 'IWM', 'XLE', 'ITA', 'BITF', 'MIGI', 'SMLR'}
TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

def discover_all_tickers():
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blobs = list(bucket.list_blobs(prefix=f'{INFERENCE_DIR}/', delimiter='/'))
    tickers = sorted([b.name.split('/')[-1].replace('_master_inference.parquet', '')
                      for b in blobs if b.name.endswith('_master_inference.parquet')])
    return [t for t in tickers if t not in EXCLUDE_TICKERS]

FOCUS_TICKERS = discover_all_tickers()

LOCAL_DIR = '/tmp/entry_time_analysis_short'
os.makedirs(LOCAL_DIR, exist_ok=True)


# -- DATA LOADING --------------------------------------------------------------

def load_master_inference(ticker):
    file_ticker = TICKER_FILE_MAP.get(ticker, ticker)
    local_path = f'/tmp/{file_ticker}_master_inference.parquet'
    if not os.path.exists(local_path):
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(f'{INFERENCE_DIR}/{file_ticker}_master_inference.parquet')
        if not blob.exists():
            return None
        blob.download_to_filename(local_path)
    df = pd.read_parquet(local_path)
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
    df.index = df.index.tz_convert(PST)
    return df


# -- FORWARD RETURN COMPUTATION ------------------------------------------------

def compute_forward_stats(df, entry_minute, horizons, study_start, study_end):
    s_start = pd.Timestamp(study_start, tz=PST)
    s_end = pd.Timestamp(study_end, tz=PST) + pd.Timedelta(days=1)
    df = df[(df.index >= s_start) & (df.index < s_end)].copy()

    if len(df) == 0:
        return None

    mins = df.index.hour * 60 + df.index.minute
    dates = df.index.date

    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values

    entry_mask = (mins == entry_minute) & (df['Volume'] > 0)
    entry_indices = np.where(entry_mask)[0]

    if len(entry_indices) == 0:
        return None

    records = []
    max_horizon = max(horizons)

    for idx in entry_indices:
        entry_price = closes[idx]
        entry_date = dates[idx]
        if entry_price <= 0:
            continue

        end_idx = min(idx + max_horizon + 1, len(df))

        same_day_end = idx + 1
        while same_day_end < len(df) and dates[same_day_end] == entry_date:
            same_day_end += 1
        end_idx = min(end_idx, same_day_end)

        if end_idx <= idx + 1:
            continue

        rec = {'date': entry_date, 'entry_price': entry_price, 'entry_minute': entry_minute}

        for h in horizons:
            h_end = min(idx + h + 1, end_idx)
            if h_end <= idx + 1:
                rec[f'max_high_{h}'] = np.nan
                rec[f'max_low_{h}'] = np.nan
                rec[f'close_ret_{h}'] = np.nan
                rec[f'max_high_ret_{h}'] = np.nan
                rec[f'max_low_ret_{h}'] = np.nan
                rec[f'max_dump_ret_{h}'] = np.nan
                continue

            fwd_highs = highs[idx+1:h_end]
            fwd_lows = lows[idx+1:h_end]
            fwd_close = closes[min(idx+h, end_idx-1)]

            max_high = np.max(fwd_highs)
            min_low = np.min(fwd_lows)

            rec[f'max_high_{h}'] = max_high
            rec[f'max_low_{h}'] = min_low
            rec[f'close_ret_{h}'] = (fwd_close - entry_price) / entry_price
            rec[f'max_high_ret_{h}'] = (max_high - entry_price) / entry_price
            rec[f'max_low_ret_{h}'] = (min_low - entry_price) / entry_price
            # max dump = how far price drops below entry (positive = bigger dump = better for SHORT)
            rec[f'max_dump_ret_{h}'] = (entry_price - min_low) / entry_price

        records.append(rec)

    return pd.DataFrame(records) if records else None


# -- INTRADAY PRICE SHAPE ------------------------------------------------------

def compute_avg_intraday_shape(df, study_start, study_end):
    s_start = pd.Timestamp(study_start, tz=PST)
    s_end = pd.Timestamp(study_end, tz=PST) + pd.Timedelta(days=1)
    df = df[(df.index >= s_start) & (df.index < s_end)].copy()

    mins = df.index.hour * 60 + df.index.minute
    dates = df.index.date
    closes = df['Close'].values

    window_start = 360   # 6:00 AM
    window_end = 600     # 10:00 AM
    n_minutes = window_end - window_start

    daily_curves = []
    for d in np.unique(dates):
        day_mask = dates == d
        day_mins = mins[day_mask]
        day_closes = closes[day_mask]

        open_idx = np.where((day_mins >= 389) & (day_mins <= 392))[0]
        if len(open_idx) == 0:
            continue
        open_price = day_closes[open_idx[0]]
        if open_price <= 0:
            continue

        curve = np.full(n_minutes, np.nan)
        for i, (m, c) in enumerate(zip(day_mins, day_closes)):
            if window_start <= m < window_end and c > 0:
                curve[m - window_start] = (c - open_price) / open_price

        if np.sum(~np.isnan(curve)) > 50:
            daily_curves.append(curve)

    if not daily_curves:
        return None, None

    curves = np.array(daily_curves)
    avg_curve = np.nanmean(curves, axis=0)
    std_curve = np.nanstd(curves, axis=0)
    return avg_curve, std_curve


def compute_dump_day_shape(df, study_start, study_end, dump_thresh=0.06):
    """Intraday shape on days where price dropped >= dump_thresh within 60 min of open."""
    s_start = pd.Timestamp(study_start, tz=PST)
    s_end = pd.Timestamp(study_end, tz=PST) + pd.Timedelta(days=1)
    df = df[(df.index >= s_start) & (df.index < s_end)].copy()

    mins = df.index.hour * 60 + df.index.minute
    dates = df.index.date
    lows = df['Low'].values
    closes = df['Close'].values

    window_start = 360
    window_end = 600
    n_minutes = window_end - window_start

    dump_curves = []
    for d in np.unique(dates):
        day_mask = dates == d
        day_mins = mins[day_mask]
        day_lows = lows[day_mask]
        day_closes = closes[day_mask]

        open_idx = np.where((day_mins >= 389) & (day_mins <= 392))[0]
        if len(open_idx) == 0:
            continue
        open_price = day_closes[open_idx[0]]
        if open_price <= 0:
            continue

        dump_window = (day_mins >= 390) & (day_mins <= 450)
        if dump_window.sum() == 0:
            continue
        min_low_in_window = np.min(day_lows[dump_window])
        if (open_price - min_low_in_window) / open_price < dump_thresh:
            continue

        curve = np.full(n_minutes, np.nan)
        for i, (m, c) in enumerate(zip(day_mins, day_closes)):
            if window_start <= m < window_end and c > 0:
                curve[m - window_start] = (c - open_price) / open_price

        if np.sum(~np.isnan(curve)) > 50:
            dump_curves.append(curve)

    if not dump_curves:
        return None, None, 0

    curves = np.array(dump_curves)
    return np.nanmean(curves, axis=0), np.nanstd(curves, axis=0), len(dump_curves)


# -- MAIN ----------------------------------------------------------------------

print(f"{'='*100}")
print(f"  ENTRY TIME ANALYSIS: 6:30 AM vs 7:00 AM PT — SHORT DUMPS")
print(f"  Study period: {STUDY_START} -> {STUDY_END}")
print(f"  Horizons: {FWD_HORIZONS} minutes")
print(f"  Dump thresholds: {[f'{t:.0%}' for t in DUMP_THRESHOLDS]}")
print(f"  Tickers: {len(FOCUS_TICKERS)}")
print(f"{'='*100}")

t_total = time.time()

# -- Step 1: Load data and compute forward stats --------------------------------
print(f"\nStep 1: Loading data and computing forward returns...")

all_stats = {}
all_shapes = {}
all_dump_shapes = {}

for i, ticker in enumerate(FOCUS_TICKERS):
    df = load_master_inference(ticker)
    if df is None:
        print(f"  {ticker}: SKIP (no data)")
        continue

    for entry_label, entry_min in ENTRY_MINUTES.items():
        stats = compute_forward_stats(df, entry_min, FWD_HORIZONS, STUDY_START, STUDY_END)
        if stats is not None and len(stats) > 10:
            key = (ticker, entry_label)
            all_stats[key] = stats

    shape_avg, shape_std = compute_avg_intraday_shape(df, STUDY_START, STUDY_END)
    dump_avg, dump_std, n_dump_days = compute_dump_day_shape(df, STUDY_START, STUDY_END, 0.06)
    if shape_avg is not None:
        all_shapes[ticker] = (shape_avg, shape_std)
    if dump_avg is not None:
        all_dump_shapes[ticker] = (dump_avg, dump_std, n_dump_days)

    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{len(FOCUS_TICKERS)} tickers loaded")

    del df
    gc.collect()

print(f"  Done: {len(all_stats)} (ticker, entry) combos loaded")


# -- Step 2: Dump frequency analysis ---------------------------------------------
print(f"\n{'='*100}")
print(f"  ANALYSIS 1: DUMP FREQUENCY (% of days with max-dump >= threshold)")
print(f"  max-dump = (entry - min_low) / entry  [positive = price fell]")
print(f"{'='*100}")

for dump_t in [0.06, 0.08]:
    print(f"\n  Dump threshold: {dump_t:.0%}")
    print(f"  {'Ticker':<7}", end="")
    for h in FWD_HORIZONS:
        print(f"  {'6:30/'+str(h)+'m':>9}  {'7:00/'+str(h)+'m':>9}", end="")
    print()
    print(f"  {'---'*1:<7}", end="")
    for h in FWD_HORIZONS:
        print(f"  {'-'*9}  {'-'*9}", end="")
    print()

    agg_630 = {h: [] for h in FWD_HORIZONS}
    agg_700 = {h: [] for h in FWD_HORIZONS}

    for ticker in FOCUS_TICKERS:
        k630 = (ticker, '6:30 AM')
        k700 = (ticker, '7:00 AM')
        if k630 not in all_stats and k700 not in all_stats:
            continue

        print(f"  {ticker:<7}", end="")
        for h in FWD_HORIZONS:
            for entry_label, key, agg in [('6:30', k630, agg_630), ('7:00', k700, agg_700)]:
                if key in all_stats:
                    col = f'max_dump_ret_{h}'
                    vals = all_stats[key][col].dropna()
                    if len(vals) > 0:
                        freq = (vals >= dump_t).mean() * 100
                        agg[h].append(freq)
                        print(f"  {freq:>8.1f}%", end="")
                    else:
                        print(f"  {'---':>9}", end="")
                else:
                    print(f"  {'---':>9}", end="")
        print()

    print(f"\n  {'AVERAGE':<7}", end="")
    for h in FWD_HORIZONS:
        for agg in [agg_630, agg_700]:
            if agg[h]:
                print(f"  {np.mean(agg[h]):>8.1f}%", end="")
            else:
                print(f"  {'---':>9}", end="")
    print()


# -- Step 3: Average forward max-dump return ------------------------------------
print(f"\n{'='*100}")
print(f"  ANALYSIS 2: AVG MAX-DUMP RETURN (mean of (entry - min_low) / entry)")
print(f"  Higher = bigger average downward move = better for SHORT")
print(f"{'='*100}")

print(f"\n  {'Ticker':<7}", end="")
for h in FWD_HORIZONS:
    print(f"  {'6:30/'+str(h)+'m':>9}  {'7:00/'+str(h)+'m':>9}", end="")
print()
print(f"  {'---'*1:<7}", end="")
for h in FWD_HORIZONS:
    print(f"  {'-'*9}  {'-'*9}", end="")
print()

agg_dump_630 = {h: [] for h in FWD_HORIZONS}
agg_dump_700 = {h: [] for h in FWD_HORIZONS}

for ticker in FOCUS_TICKERS:
    k630 = (ticker, '6:30 AM')
    k700 = (ticker, '7:00 AM')
    if k630 not in all_stats and k700 not in all_stats:
        continue
    print(f"  {ticker:<7}", end="")
    for h in FWD_HORIZONS:
        for key, agg in [(k630, agg_dump_630), (k700, agg_dump_700)]:
            if key in all_stats:
                col = f'max_dump_ret_{h}'
                vals = all_stats[key][col].dropna()
                if len(vals) > 0:
                    avg = vals.mean() * 100
                    agg[h].append(avg)
                    print(f"  {avg:>+8.2f}%", end="")
                else:
                    print(f"  {'---':>9}", end="")
            else:
                print(f"  {'---':>9}", end="")
    print()

print(f"\n  {'AVERAGE':<7}", end="")
for h in FWD_HORIZONS:
    for agg in [agg_dump_630, agg_dump_700]:
        if agg[h]:
            print(f"  {np.mean(agg[h]):>+8.2f}%", end="")
        else:
            print(f"  {'---':>9}", end="")
print()


# -- Step 4: Conditional analysis -- on dump days only ---------------------------
print(f"\n{'='*100}")
print(f"  ANALYSIS 3: 6:30 vs 7:00 ON DUMP DAYS (days with >=6% max-dump within 60 min of open)")
print(f"{'='*100}")

print(f"\n  Question: On days that DO have a 6%+ dump, how much of the move")
print(f"  is still available at 6:30 vs 7:00?")

print(f"\n  {'Ticker':<7} {'#DmpDays':>8}  "
      f"{'6:30 dump':>10} {'6:30 capt':>9}  "
      f"{'7:00 dump':>10} {'7:00 capt':>9}  "
      f"{'Winner':>7}")
print(f"  {'-'*80}")

win_630 = 0
win_700 = 0
tie = 0

remaining_630_all = []
remaining_700_all = []

for ticker in FOCUS_TICKERS:
    k630 = (ticker, '6:30 AM')
    k700 = (ticker, '7:00 AM')
    if k630 not in all_stats or k700 not in all_stats:
        continue

    df630 = all_stats[k630]
    df700 = all_stats[k700]

    dump_days = set(df630[df630['max_dump_ret_60'] >= 0.06]['date'].values)

    if len(dump_days) < 3:
        continue

    remaining_630 = []
    remaining_700 = []

    for d in dump_days:
        row630 = df630[df630['date'] == d]
        row700 = df700[df700['date'] == d]
        if len(row630) > 0:
            remaining_630.append(row630['max_dump_ret_60'].values[0])
        if len(row700) > 0:
            remaining_700.append(row700['max_dump_ret_60'].values[0])

    if not remaining_630 or not remaining_700:
        continue

    avg_rem_630 = np.mean(remaining_630) * 100
    avg_rem_700 = np.mean(remaining_700) * 100
    remaining_630_all.extend(remaining_630)
    remaining_700_all.extend(remaining_700)

    capture_630 = avg_rem_630
    capture_700 = avg_rem_700

    if capture_630 > capture_700 + 0.3:
        winner = '6:30'
        win_630 += 1
    elif capture_700 > capture_630 + 0.3:
        winner = '7:00'
        win_700 += 1
    else:
        winner = 'TIE'
        tie += 1

    print(f"  {ticker:<7} {len(dump_days):>8}  "
          f"{avg_rem_630:>+9.2f}% {capture_630:>8.2f}%  "
          f"{avg_rem_700:>+9.2f}% {capture_700:>8.2f}%  "
          f"{winner:>7}")

print(f"\n  WINNERS: 6:30={win_630}  7:00={win_700}  TIE={tie}")
if remaining_630_all and remaining_700_all:
    print(f"  Overall avg dump available at 6:30: {np.mean(remaining_630_all)*100:+.2f}%")
    print(f"  Overall avg dump available at 7:00: {np.mean(remaining_700_all)*100:+.2f}%")
    print(f"  Difference (6:30 - 7:00):           {(np.mean(remaining_630_all) - np.mean(remaining_700_all))*100:+.2f}%")


# -- Step 5: Adverse excursion (max rally against SHORT) -------------------------
print(f"\n{'='*100}")
print(f"  ANALYSIS 4: ADVERSE EXCURSION (avg max RALLY from entry — risk for SHORT)")
print(f"  Higher = price rallied MORE against the short = WORSE for SHORT")
print(f"{'='*100}")

print(f"\n  {'Ticker':<7}", end="")
for h in [15, 30, 60]:
    print(f"  {'6:30 AE'+str(h):>10}  {'7:00 AE'+str(h):>10}", end="")
print()
print(f"  {'---'*1:<7}", end="")
for h in [15, 30, 60]:
    print(f"  {'-'*10}  {'-'*10}", end="")
print()

agg_ae_630 = {h: [] for h in [15, 30, 60]}
agg_ae_700 = {h: [] for h in [15, 30, 60]}

for ticker in FOCUS_TICKERS:
    k630 = (ticker, '6:30 AM')
    k700 = (ticker, '7:00 AM')
    if k630 not in all_stats and k700 not in all_stats:
        continue
    print(f"  {ticker:<7}", end="")
    for h in [15, 30, 60]:
        for key, agg in [(k630, agg_ae_630), (k700, agg_ae_700)]:
            if key in all_stats:
                col = f'max_high_ret_{h}'
                vals = all_stats[key][col].dropna()
                if len(vals) > 0:
                    avg_ae = vals.mean() * 100
                    agg[h].append(avg_ae)
                    print(f"  {avg_ae:>+9.2f}%", end="")
                else:
                    print(f"  {'---':>10}", end="")
            else:
                print(f"  {'---':>10}", end="")
    print()

print(f"\n  {'AVERAGE':<7}", end="")
for h in [15, 30, 60]:
    for agg in [agg_ae_630, agg_ae_700]:
        if agg[h]:
            print(f"  {np.mean(agg[h]):>+9.2f}%", end="")
        else:
            print(f"  {'---':>10}", end="")
print()


# -- Step 6: SHORT Reward/Risk ratio --------------------------------------------
print(f"\n{'='*100}")
print(f"  ANALYSIS 5: SHORT REWARD/RISK (avg max-dump / avg max-rally)")
print(f"  Higher = more downside captured vs upside risk = better for SHORT")
print(f"{'='*100}")

print(f"\n  {'Ticker':<7}", end="")
for h in [15, 30, 60]:
    print(f"  {'6:30 RR'+str(h):>10}  {'7:00 RR'+str(h):>10}", end="")
print()
print(f"  {'---'*1:<7}", end="")
for h in [15, 30, 60]:
    print(f"  {'-'*10}  {'-'*10}", end="")
print()

agg_rr_630 = {h: [] for h in [15, 30, 60]}
agg_rr_700 = {h: [] for h in [15, 30, 60]}

for ticker in FOCUS_TICKERS:
    k630 = (ticker, '6:30 AM')
    k700 = (ticker, '7:00 AM')
    if k630 not in all_stats and k700 not in all_stats:
        continue
    print(f"  {ticker:<7}", end="")
    for h in [15, 30, 60]:
        for key, agg in [(k630, agg_rr_630), (k700, agg_rr_700)]:
            if key in all_stats:
                dump_col = f'max_dump_ret_{h}'
                rally_col = f'max_high_ret_{h}'
                dump_vals = all_stats[key][dump_col].dropna()
                rally_vals = all_stats[key][rally_col].dropna()
                if len(dump_vals) > 0 and len(rally_vals) > 0:
                    avg_dump = dump_vals.mean()
                    avg_rally = rally_vals.mean()
                    rr = avg_dump / avg_rally if avg_rally > 1e-6 else 0
                    agg[h].append(rr)
                    print(f"  {rr:>9.2f}x", end="")
                else:
                    print(f"  {'---':>10}", end="")
            else:
                print(f"  {'---':>10}", end="")
    print()

print(f"\n  {'AVERAGE':<7}", end="")
for h in [15, 30, 60]:
    for agg in [agg_rr_630, agg_rr_700]:
        if agg[h]:
            print(f"  {np.mean(agg[h]):>9.2f}x", end="")
        else:
            print(f"  {'---':>10}", end="")
print()


# -- Step 7: Per-ticker optimal entry time for SHORT ----------------------------
print(f"\n{'='*100}")
print(f"  ANALYSIS 6: PER-TICKER OPTIMAL SHORT ENTRY (best RR at 30-min horizon)")
print(f"{'='*100}")

print(f"\n  {'Ticker':<7} {'6:30 RR':>8} {'7:00 RR':>8} {'6:30 dump%':>10} {'7:00 dump%':>10} {'Best':>6}")
print(f"  {'-'*60}")

best_per_ticker = {}
for ticker in FOCUS_TICKERS:
    k630 = (ticker, '6:30 AM')
    k700 = (ticker, '7:00 AM')
    if k630 not in all_stats or k700 not in all_stats:
        continue

    h = 30
    rr_630 = rr_700 = freq_630 = freq_700 = 0

    for key, label in [(k630, '6:30'), (k700, '7:00')]:
        dump_vals = all_stats[key][f'max_dump_ret_{h}'].dropna()
        rally_vals = all_stats[key][f'max_high_ret_{h}'].dropna()
        if len(dump_vals) > 0 and len(rally_vals) > 0:
            rr = dump_vals.mean() / max(rally_vals.mean(), 1e-6)
            freq = (dump_vals >= 0.06).mean() * 100
            if label == '6:30':
                rr_630, freq_630 = rr, freq
            else:
                rr_700, freq_700 = rr, freq

    best = '6:30' if rr_630 > rr_700 else '7:00'
    best_per_ticker[ticker] = best

    print(f"  {ticker:<7} {rr_630:>7.2f}x {rr_700:>7.2f}x {freq_630:>9.1f}% {freq_700:>9.1f}% {best:>6}")

n_630 = sum(1 for v in best_per_ticker.values() if v == '6:30')
n_700 = sum(1 for v in best_per_ticker.values() if v == '7:00')
print(f"\n  OVERALL: {n_630} tickers favor 6:30, {n_700} tickers favor 7:00")


# -- Step 8: Dump threshold sensitivity -----------------------------------------
print(f"\n{'='*100}")
print(f"  ANALYSIS 7: DUMP THRESHOLD SENSITIVITY (at 30-min horizon)")
print(f"{'='*100}")

print(f"\n  What % of trading days see a >= X% dump within 30 min?")
print(f"\n  {'Threshold':<10}", end="")
print(f"  {'6:30 freq':>10}  {'7:00 freq':>10}  {'Delta':>8}")
print(f"  {'-'*45}")

for dump_t in DUMP_THRESHOLDS:
    freq_630_vals = []
    freq_700_vals = []
    for ticker in FOCUS_TICKERS:
        for key, bag in [((ticker, '6:30 AM'), freq_630_vals),
                         ((ticker, '7:00 AM'), freq_700_vals)]:
            if key in all_stats:
                vals = all_stats[key]['max_dump_ret_30'].dropna()
                if len(vals) > 0:
                    bag.append((vals >= dump_t).mean() * 100)

    avg_630 = np.mean(freq_630_vals) if freq_630_vals else 0
    avg_700 = np.mean(freq_700_vals) if freq_700_vals else 0
    delta = avg_630 - avg_700
    print(f"  {dump_t:>8.0%}    {avg_630:>9.2f}%  {avg_700:>9.2f}%  {delta:>+7.2f}%")


# -- Step 9: Cumulative dump probability over time ------------------------------
print(f"\n{'='*100}")
print(f"  ANALYSIS 8: CUMULATIVE DUMP PROBABILITY OVER TIME")
print(f"  (For each horizon 3-60 min, what % of days have had a 6% dump by then?)")
print(f"{'='*100}")

print(f"\n  {'Horizon':<10}", end="")
print(f"  {'6:30 cum%':>10}  {'7:00 cum%':>10}  {'Delta':>8}")
print(f"  {'-'*45}")

for h in FWD_HORIZONS:
    cum_630 = []
    cum_700 = []
    for ticker in FOCUS_TICKERS:
        for key, bag in [((ticker, '6:30 AM'), cum_630),
                         ((ticker, '7:00 AM'), cum_700)]:
            if key in all_stats:
                vals = all_stats[key][f'max_dump_ret_{h}'].dropna()
                if len(vals) > 0:
                    bag.append((vals >= 0.06).mean() * 100)

    avg_630 = np.mean(cum_630) if cum_630 else 0
    avg_700 = np.mean(cum_700) if cum_700 else 0
    delta = avg_630 - avg_700
    print(f"  {h:>4} min    {avg_630:>9.2f}%  {avg_700:>9.2f}%  {delta:>+7.2f}%")


# -- ANALYSIS 9: SHORT-SPECIFIC — Asymmetry (dump vs spike per ticker) -----------
print(f"\n{'='*100}")
print(f"  ANALYSIS 9: DUMP vs SPIKE ASYMMETRY (SHORT signal quality)")
print(f"  Tickers where dumps > spikes = natural short candidates")
print(f"  At 30-min horizon, 6:30 AM entry")
print(f"{'='*100}")

print(f"\n  {'Ticker':<7} {'Avg Dump':>9} {'Avg Spike':>10} {'Dump/Spike':>10} {'6% Dump%':>9} {'6% Spike%':>10} {'Bias':>7}")
print(f"  {'-'*70}")

asymmetry_data = []
for ticker in FOCUS_TICKERS:
    k630 = (ticker, '6:30 AM')
    if k630 not in all_stats:
        continue

    dump_vals = all_stats[k630]['max_dump_ret_30'].dropna()
    spike_vals = all_stats[k630]['max_high_ret_30'].dropna()

    if len(dump_vals) == 0 or len(spike_vals) == 0:
        continue

    avg_dump = dump_vals.mean() * 100
    avg_spike = spike_vals.mean() * 100
    ratio = avg_dump / max(avg_spike, 0.01)
    dump_freq = (dump_vals >= 0.06).mean() * 100
    spike_freq = (spike_vals >= 0.06).mean() * 100

    if ratio > 1.05:
        bias = 'SHORT'
    elif ratio < 0.95:
        bias = 'LONG'
    else:
        bias = 'NEUTRAL'

    asymmetry_data.append({
        'ticker': ticker, 'avg_dump': avg_dump, 'avg_spike': avg_spike,
        'ratio': ratio, 'dump_freq': dump_freq, 'spike_freq': spike_freq, 'bias': bias,
    })

asymmetry_data.sort(key=lambda x: x['ratio'], reverse=True)
for row in asymmetry_data:
    print(f"  {row['ticker']:<7} {row['avg_dump']:>+8.2f}% {row['avg_spike']:>+9.2f}% "
          f"{row['ratio']:>9.2f}x {row['dump_freq']:>8.1f}% {row['spike_freq']:>9.1f}% "
          f"{row['bias']:>7}")

n_short_bias = sum(1 for r in asymmetry_data if r['bias'] == 'SHORT')
n_long_bias = sum(1 for r in asymmetry_data if r['bias'] == 'LONG')
n_neutral = sum(1 for r in asymmetry_data if r['bias'] == 'NEUTRAL')
print(f"\n  BIAS SUMMARY: {n_short_bias} SHORT-biased, {n_long_bias} LONG-biased, {n_neutral} NEUTRAL")

# Top SHORT candidates
top_short = [r for r in asymmetry_data if r['ratio'] > 1.0 and r['dump_freq'] >= 3.0]
if top_short:
    print(f"\n  TOP SHORT CANDIDATES (dump/spike > 1.0 AND dump freq >= 3%):")
    for r in top_short[:15]:
        print(f"    {r['ticker']:<7}  dump/spike={r['ratio']:.2f}x  "
              f"6% dump freq={r['dump_freq']:.1f}%  avg dump={r['avg_dump']:+.2f}%")


# ==============================================================================
# CHARTS
# ==============================================================================

fig, axes = plt.subplots(3, 2, figsize=(20, 22))
fig.suptitle(
    f'Entry Time Analysis: 6:30 AM vs 7:00 AM PT  --  SHORT DUMPS\n'
    f'{STUDY_START} -> {STUDY_END} | {len(FOCUS_TICKERS)} tickers | Focus: 6%+ dumps',
    fontsize=14, fontweight='bold')

# [0,0] Avg max-dump return by horizon
ax = axes[0, 0]
x = np.arange(len(FWD_HORIZONS))
w = 0.35
vals_630 = [np.mean(agg_dump_630[h]) if agg_dump_630[h] else 0 for h in FWD_HORIZONS]
vals_700 = [np.mean(agg_dump_700[h]) if agg_dump_700[h] else 0 for h in FWD_HORIZONS]
bars1 = ax.bar(x - w/2, vals_630, w, label='6:30 AM', color='#c62828', alpha=0.85)
bars2 = ax.bar(x + w/2, vals_700, w, label='7:00 AM', color='#FF6600', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([f'{h}m' for h in FWD_HORIZONS])
ax.set_ylabel('Avg Max-Dump Return (%)')
ax.set_title('Avg Max Downside by Horizon (SHORT opportunity)')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars1, vals_630):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{val:.2f}', ha='center', va='bottom', fontsize=7)
for bar, val in zip(bars2, vals_700):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{val:.2f}', ha='center', va='bottom', fontsize=7)

# [0,1] 6% dump frequency by horizon
ax = axes[0, 1]
freq_630 = []
freq_700 = []
for h in FWD_HORIZONS:
    f630_vals = []
    f700_vals = []
    for ticker in FOCUS_TICKERS:
        for key, bag in [((ticker, '6:30 AM'), f630_vals),
                         ((ticker, '7:00 AM'), f700_vals)]:
            if key in all_stats:
                vals = all_stats[key][f'max_dump_ret_{h}'].dropna()
                if len(vals) > 0:
                    bag.append((vals >= 0.06).mean() * 100)
    freq_630.append(np.mean(f630_vals) if f630_vals else 0)
    freq_700.append(np.mean(f700_vals) if f700_vals else 0)

ax.plot(FWD_HORIZONS, freq_630, 'o-', color='#c62828', linewidth=2,
        markersize=8, label='6:30 AM')
ax.plot(FWD_HORIZONS, freq_700, 's--', color='#FF6600', linewidth=2,
        markersize=8, label='7:00 AM')
ax.set_xlabel('Forward Horizon (min)')
ax.set_ylabel('% of Days with >= 6% Dump')
ax.set_title('6% Dump Frequency by Horizon')
ax.legend()
ax.grid(True, alpha=0.3)

# [1,0] SHORT Reward/Risk ratio by horizon
ax = axes[1, 0]
rr_630 = [np.mean(agg_rr_630[h]) if agg_rr_630.get(h) else 0 for h in [15, 30, 60]]
rr_700 = [np.mean(agg_rr_700[h]) if agg_rr_700.get(h) else 0 for h in [15, 30, 60]]
x_rr = np.arange(3)
ax.bar(x_rr - w/2, rr_630, w, label='6:30 AM', color='#c62828', alpha=0.85)
ax.bar(x_rr + w/2, rr_700, w, label='7:00 AM', color='#FF6600', alpha=0.85)
ax.set_xticks(x_rr)
ax.set_xticklabels(['15 min', '30 min', '60 min'])
ax.set_ylabel('SHORT Reward/Risk (dump / rally)')
ax.set_title('SHORT Reward/Risk Ratio')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

# [1,1] Adverse excursion (max rally against SHORT)
ax = axes[1, 1]
ae_630 = [np.mean(agg_ae_630[h]) if agg_ae_630.get(h) else 0 for h in [15, 30, 60]]
ae_700 = [np.mean(agg_ae_700[h]) if agg_ae_700.get(h) else 0 for h in [15, 30, 60]]
ax.bar(x_rr - w/2, ae_630, w, label='6:30 AM', color='#c62828', alpha=0.85)
ax.bar(x_rr + w/2, ae_700, w, label='7:00 AM', color='#FF6600', alpha=0.85)
ax.set_xticks(x_rr)
ax.set_xticklabels(['15 min', '30 min', '60 min'])
ax.set_ylabel('Avg Max Rally from Entry (%)')
ax.set_title('SHORT Risk: Adverse Excursion (rally against you)')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# [2,0] Average intraday price shape (all tickers combined)
ax = axes[2, 0]
all_avg_shapes = [v[0] for v in all_shapes.values()]
if all_avg_shapes:
    combined_shape = np.nanmean(all_avg_shapes, axis=0) * 100
    combined_std = np.nanstd(all_avg_shapes, axis=0) * 100
    minutes_x = np.arange(len(combined_shape))
    time_labels = [f'{(360+m)//60}:{(360+m)%60:02d}' for m in minutes_x]
    ax.plot(minutes_x, combined_shape, color='steelblue', linewidth=2)
    ax.fill_between(minutes_x,
                    combined_shape - combined_std,
                    combined_shape + combined_std,
                    alpha=0.15, color='steelblue')
    ax.axvline(x=30, color='#c62828', linestyle='--', alpha=0.7, label='6:30 AM')
    ax.axvline(x=60, color='#FF6600', linestyle='--', alpha=0.7, label='7:00 AM')
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5, label='6:00 AM')
    tick_positions = [0, 30, 60, 90, 120, 150, 180, 210, 240]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([time_labels[i] if i < len(time_labels) else '' for i in tick_positions],
                       rotation=45, fontsize=8)
    ax.set_ylabel('Return from Open (%)')
    ax.set_title('Average Intraday Shape (all days, all tickers)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linewidth=0.5)

# [2,1] Dump-day intraday shape
ax = axes[2, 1]
all_dump_avg = [v[0] for v in all_dump_shapes.values()]
if all_dump_avg:
    combined_dump = np.nanmean(all_dump_avg, axis=0) * 100
    combined_dump_std = np.nanstd(all_dump_avg, axis=0) * 100
    total_dump_days = sum(v[2] for v in all_dump_shapes.values())
    ax.plot(minutes_x, combined_dump, color='#c62828', linewidth=2)
    ax.fill_between(minutes_x,
                    combined_dump - combined_dump_std,
                    combined_dump + combined_dump_std,
                    alpha=0.15, color='#c62828')
    ax.axvline(x=30, color='#1565C0', linestyle='--', alpha=0.7, label='6:30 AM')
    ax.axvline(x=60, color='#FF6600', linestyle='--', alpha=0.7, label='7:00 AM')
    ax.axhline(y=-6, color='green', linestyle=':', alpha=0.5, label='-6% target')
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([time_labels[i] if i < len(time_labels) else '' for i in tick_positions],
                       rotation=45, fontsize=8)
    ax.set_ylabel('Return from Open (%)')
    ax.set_title(f'Dump-Day Shape (6%+ dump days, {total_dump_days} ticker-days)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linewidth=0.5)

plt.tight_layout()
chart_path = f'{LOCAL_DIR}/entry_time_630_vs_700_short.png'
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
if _IN_COLAB:
    plt.show()
plt.close()
print(f"\nChart saved: {chart_path}")


# -- Chart 2: Per-ticker dump comparison heatmap --------------------------------
tickers_with_data = [t for t in FOCUS_TICKERS
                     if (t, '6:30 AM') in all_stats and (t, '7:00 AM') in all_stats]

if tickers_with_data:
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(20, max(8, len(tickers_with_data) * 0.35)))

    # Left: dump advantage (6:30 - 7:00)
    delta_vals = []
    for ticker in tickers_with_data:
        dump_630 = all_stats[(ticker, '6:30 AM')]['max_dump_ret_30'].dropna().mean()
        dump_700 = all_stats[(ticker, '7:00 AM')]['max_dump_ret_30'].dropna().mean()
        delta_vals.append((dump_630 - dump_700) * 100)

    sorted_idx = np.argsort(delta_vals)
    sorted_tickers = [tickers_with_data[i] for i in sorted_idx]
    sorted_deltas = [delta_vals[i] for i in sorted_idx]

    colors = ['#c62828' if d > 0 else '#FF6600' for d in sorted_deltas]
    ax2a.barh(range(len(sorted_tickers)), sorted_deltas, color=colors, alpha=0.85)
    ax2a.set_yticks(range(len(sorted_tickers)))
    ax2a.set_yticklabels(sorted_tickers, fontsize=7)
    ax2a.axvline(x=0, color='black', linewidth=0.8)
    ax2a.set_xlabel('Max-Dump Advantage (6:30 minus 7:00) at 30-min (%)')
    ax2a.set_title('Per-Ticker: 6:30 AM Dump Advantage\nRed = 6:30 bigger dumps | Orange = 7:00 bigger',
                   fontsize=10, fontweight='bold')
    ax2a.grid(True, alpha=0.3, axis='x')

    # Right: dump vs spike asymmetry at 6:30
    asym_tickers = [r['ticker'] for r in asymmetry_data if r['ticker'] in tickers_with_data]
    asym_ratios = [r['ratio'] for r in asymmetry_data if r['ticker'] in tickers_with_data]

    asym_colors = ['#c62828' if r > 1.0 else '#1565C0' for r in asym_ratios]
    ax2b.barh(range(len(asym_tickers)), asym_ratios, color=asym_colors, alpha=0.85)
    ax2b.set_yticks(range(len(asym_tickers)))
    ax2b.set_yticklabels(asym_tickers, fontsize=7)
    ax2b.axvline(x=1.0, color='black', linewidth=0.8, linestyle='--')
    ax2b.set_xlabel('Dump/Spike Ratio (>1 = SHORT-biased)')
    ax2b.set_title('Dump vs Spike Asymmetry at 6:30 AM\nRed = SHORT-biased | Blue = LONG-biased',
                   fontsize=10, fontweight='bold')
    ax2b.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    chart2_path = f'{LOCAL_DIR}/entry_time_short_per_ticker.png'
    plt.savefig(chart2_path, dpi=150, bbox_inches='tight')
    if _IN_COLAB:
        plt.show()
    plt.close()
    print(f"Chart 2 saved: {chart2_path}")


# ==============================================================================
# SUMMARY & RECOMMENDATIONS
# ==============================================================================

print(f"\n{'='*100}")
print(f"  SUMMARY & RECOMMENDATIONS — SHORT DUMPS")
print(f"{'='*100}")

print(f"\n  STUDY: {STUDY_START} -> {STUDY_END}, {len(FOCUS_TICKERS)} tickers")
print(f"  Focus: SHORT bets, 3-60 min hold, 6-8% dump targets\n")

for h in [15, 30, 60]:
    avg_dump_630 = np.mean(agg_dump_630[h]) if agg_dump_630[h] else 0
    avg_dump_700 = np.mean(agg_dump_700[h]) if agg_dump_700[h] else 0
    print(f"  {h}-min horizon: 6:30 avg max-dump={avg_dump_630:+.2f}%  vs  "
          f"7:00 avg max-dump={avg_dump_700:+.2f}%  "
          f"(delta={avg_dump_630-avg_dump_700:+.2f}%)")

if remaining_630_all and remaining_700_all:
    print(f"\n  On 6%+ dump days:")
    print(f"    Remaining downside at 6:30: {np.mean(remaining_630_all)*100:+.2f}%")
    print(f"    Remaining downside at 7:00: {np.mean(remaining_700_all)*100:+.2f}%")
    print(f"    6:30 captures {(np.mean(remaining_630_all) - np.mean(remaining_700_all))*100:.2f}% MORE of the move")

print(f"\n  Per-ticker vote: {n_630} favor 6:30, {n_700} favor 7:00")

print(f"\n  SHORT-biased tickers (dump/spike > 1.0 at 6:30, 30-min):")
for r in asymmetry_data[:10]:
    if r['ratio'] > 1.0:
        deployed_marker = ' *DEPLOYED*' if r['ticker'] in {'BE', 'OKLO', 'SMR'} else ''
        print(f"    {r['ticker']:<7}  ratio={r['ratio']:.2f}x  dump_freq={r['dump_freq']:.1f}%{deployed_marker}")

total_elapsed = time.time() - t_total
print(f"\n  Total runtime: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
print(f"{'='*100}")
