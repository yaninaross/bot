# ==============================================================================
# Entry Time Analysis: 6:30 AM vs 7:00 AM PT for LONG Spike Opportunities
# ==============================================================================
# Loads OHLCV from master_inference parquets (GCS), computes forward returns
# at multiple horizons (3–60 min) from 6:30 and 7:00 AM PT entries, and
# analyzes spike frequency, magnitude, and price shape around market open.
#
# Focus: 6–8% spike detection opportunities for LONG bets.
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

# ── CONFIG ────────────────────────────────────────────────────────────────────

BUCKET_NAME = os.environ.get("GCS_BUCKET", "")
INFERENCE_DIR = 'Data/Inference_v2'
PST = pytz.timezone('America/Los_Angeles')

STUDY_START = '2026-03-09'
STUDY_END   = '2026-08-08'

ENTRY_MINUTES = {
    '6:30 AM': 390,   # 6*60 + 30 = 390
    '7:00 AM': 420,   # 7*60 + 0  = 420
}

FWD_HORIZONS = [3, 5, 10, 15, 20, 30, 45, 60]

SPIKE_THRESHOLDS = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10]

EXCLUDE_TICKERS = {'SPY', 'VIX', 'TLT', 'IWM', 'XLE', 'ITA', 'BITF', 'MIGI', 'SMLR'}
TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

# Discover all tradable tickers from GCS (auto-populate)
def discover_all_tickers():
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blobs = list(bucket.list_blobs(prefix=f'{INFERENCE_DIR}/', delimiter='/'))
    tickers = sorted([b.name.split('/')[-1].replace('_master_inference.parquet', '')
                      for b in blobs if b.name.endswith('_master_inference.parquet')])
    return [t for t in tickers if t not in EXCLUDE_TICKERS]

FOCUS_TICKERS = discover_all_tickers()

LOCAL_DIR = '/tmp/entry_time_analysis'
os.makedirs(LOCAL_DIR, exist_ok=True)


# ── DATA LOADING ──────────────────────────────────────────────────────────────

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


# ── FORWARD RETURN COMPUTATION ────────────────────────────────────────────────

def compute_forward_stats(df, entry_minute, horizons, study_start, study_end):
    """For each trading day, find the bar at entry_minute.
    Compute forward max-High return and forward Close return at each horizon."""
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

        records.append(rec)

    return pd.DataFrame(records) if records else None


# ── INTRADAY PRICE SHAPE ─────────────────────────────────────────────────────

def compute_avg_intraday_shape(df, study_start, study_end):
    """Compute average minute-by-minute return from 6:00 AM to 10:00 AM PT."""
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


def compute_spike_day_shape(df, study_start, study_end, spike_thresh=0.06):
    """Same as above but only on days where a >=spike_thresh move occurred within 60 min of open."""
    s_start = pd.Timestamp(study_start, tz=PST)
    s_end = pd.Timestamp(study_end, tz=PST) + pd.Timedelta(days=1)
    df = df[(df.index >= s_start) & (df.index < s_end)].copy()

    mins = df.index.hour * 60 + df.index.minute
    dates = df.index.date
    highs = df['High'].values
    closes = df['Close'].values

    window_start = 360
    window_end = 600
    n_minutes = window_end - window_start

    spike_curves = []
    for d in np.unique(dates):
        day_mask = dates == d
        day_mins = mins[day_mask]
        day_highs = highs[day_mask]
        day_closes = closes[day_mask]

        open_idx = np.where((day_mins >= 389) & (day_mins <= 392))[0]
        if len(open_idx) == 0:
            continue
        open_price = day_closes[open_idx[0]]
        if open_price <= 0:
            continue

        spike_window = (day_mins >= 390) & (day_mins <= 450)
        if spike_window.sum() == 0:
            continue
        max_high_in_window = np.max(day_highs[spike_window])
        if (max_high_in_window - open_price) / open_price < spike_thresh:
            continue

        curve = np.full(n_minutes, np.nan)
        for i, (m, c) in enumerate(zip(day_mins, day_closes)):
            if window_start <= m < window_end and c > 0:
                curve[m - window_start] = (c - open_price) / open_price

        if np.sum(~np.isnan(curve)) > 50:
            spike_curves.append(curve)

    if not spike_curves:
        return None, None, 0

    curves = np.array(spike_curves)
    return np.nanmean(curves, axis=0), np.nanstd(curves, axis=0), len(spike_curves)


# ── MAIN ──────────────────────────────────────────────────────────────────────

print(f"{'='*100}")
print(f"  ENTRY TIME ANALYSIS: 6:30 AM vs 7:00 AM PT")
print(f"  Study period: {STUDY_START} -> {STUDY_END}")
print(f"  Horizons: {FWD_HORIZONS} minutes")
print(f"  Spike thresholds: {[f'{t:.0%}' for t in SPIKE_THRESHOLDS]}")
print(f"  Tickers: {len(FOCUS_TICKERS)}")
print(f"{'='*100}")

t_total = time.time()

# ── Step 1: Load data and compute forward stats ──────────────────────────────
print(f"\nStep 1: Loading data and computing forward returns...")

all_stats = {}
all_shapes = {}
all_spike_shapes = {}

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
    spike_avg, spike_std, n_spike_days = compute_spike_day_shape(df, STUDY_START, STUDY_END, 0.06)
    if shape_avg is not None:
        all_shapes[ticker] = (shape_avg, shape_std)
    if spike_avg is not None:
        all_spike_shapes[ticker] = (spike_avg, spike_std, n_spike_days)

    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{len(FOCUS_TICKERS)} tickers loaded")

    del df
    gc.collect()

print(f"  Done: {len(all_stats)} (ticker, entry) combos loaded")


# ── Step 2: Spike frequency analysis ─────────────────────────────────────────
print(f"\n{'='*100}")
print(f"  ANALYSIS 1: SPIKE FREQUENCY (% of days with max-high >= threshold)")
print(f"{'='*100}")

for spike_t in [0.06, 0.08]:
    print(f"\n  Spike threshold: {spike_t:.0%}")
    print(f"  {'Ticker':<7}", end="")
    for h in FWD_HORIZONS:
        print(f"  {'6:30/'+str(h)+'m':>9}  {'7:00/'+str(h)+'m':>9}", end="")
    print()
    print(f"  {'─'*7}", end="")
    for h in FWD_HORIZONS:
        print(f"  {'─'*9}  {'─'*9}", end="")
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
                    col = f'max_high_ret_{h}'
                    vals = all_stats[key][col].dropna()
                    if len(vals) > 0:
                        freq = (vals >= spike_t).mean() * 100
                        agg[h].append(freq)
                        print(f"  {freq:>8.1f}%", end="")
                    else:
                        print(f"  {'—':>9}", end="")
                else:
                    print(f"  {'—':>9}", end="")
        print()

    print(f"\n  {'AVERAGE':<7}", end="")
    for h in FWD_HORIZONS:
        for agg in [agg_630, agg_700]:
            if agg[h]:
                print(f"  {np.mean(agg[h]):>8.1f}%", end="")
            else:
                print(f"  {'—':>9}", end="")
    print()


# ── Step 3: Average forward max-high return ──────────────────────────────────
print(f"\n{'='*100}")
print(f"  ANALYSIS 2: AVG MAX-HIGH RETURN (mean of max forward High / entry)")
print(f"{'='*100}")

print(f"\n  {'Ticker':<7}", end="")
for h in FWD_HORIZONS:
    print(f"  {'6:30/'+str(h)+'m':>9}  {'7:00/'+str(h)+'m':>9}", end="")
print()
print(f"  {'─'*7}", end="")
for h in FWD_HORIZONS:
    print(f"  {'─'*9}  {'─'*9}", end="")
print()

agg_mhr_630 = {h: [] for h in FWD_HORIZONS}
agg_mhr_700 = {h: [] for h in FWD_HORIZONS}

for ticker in FOCUS_TICKERS:
    k630 = (ticker, '6:30 AM')
    k700 = (ticker, '7:00 AM')
    if k630 not in all_stats and k700 not in all_stats:
        continue
    print(f"  {ticker:<7}", end="")
    for h in FWD_HORIZONS:
        for key, agg in [(k630, agg_mhr_630), (k700, agg_mhr_700)]:
            if key in all_stats:
                col = f'max_high_ret_{h}'
                vals = all_stats[key][col].dropna()
                if len(vals) > 0:
                    avg = vals.mean() * 100
                    agg[h].append(avg)
                    print(f"  {avg:>+8.2f}%", end="")
                else:
                    print(f"  {'—':>9}", end="")
            else:
                print(f"  {'—':>9}", end="")
    print()

print(f"\n  {'AVERAGE':<7}", end="")
for h in FWD_HORIZONS:
    for agg in [agg_mhr_630, agg_mhr_700]:
        if agg[h]:
            print(f"  {np.mean(agg[h]):>+8.2f}%", end="")
        else:
            print(f"  {'—':>9}", end="")
print()


# ── Step 4: Conditional analysis — on spike days only ─────────────────────────
print(f"\n{'='*100}")
print(f"  ANALYSIS 3: 6:30 vs 7:00 ON SPIKE DAYS (days with >=6% max-high within 60 min of open)")
print(f"{'='*100}")

print(f"\n  Question: On days that DO have a 6%+ spike, how much of the move")
print(f"  is still available at 6:30 vs 7:00?")

print(f"\n  {'Ticker':<7} {'#SpkDays':>8}  "
      f"{'6:30 remain':>11} {'6:30 capt':>9}  "
      f"{'7:00 remain':>11} {'7:00 capt':>9}  "
      f"{'Winner':>7}")
print(f"  {'─'*80}")

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

    # Find spike days from 6:30 perspective (max high ret 60 >= 6%)
    spike_days = set(df630[df630['max_high_ret_60'] >= 0.06]['date'].values)

    if len(spike_days) < 3:
        continue

    remaining_630 = []
    remaining_700 = []

    for d in spike_days:
        row630 = df630[df630['date'] == d]
        row700 = df700[df700['date'] == d]
        if len(row630) > 0:
            remaining_630.append(row630['max_high_ret_60'].values[0])
        if len(row700) > 0:
            remaining_700.append(row700['max_high_ret_60'].values[0])

    if not remaining_630 or not remaining_700:
        continue

    avg_rem_630 = np.mean(remaining_630) * 100
    avg_rem_700 = np.mean(remaining_700) * 100
    remaining_630_all.extend(remaining_630)
    remaining_700_all.extend(remaining_700)

    # "captured" = how much of the original spike is still available
    # (higher = better, means the spike hasn't already happened)
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

    print(f"  {ticker:<7} {len(spike_days):>8}  "
          f"{avg_rem_630:>+10.2f}% {capture_630:>8.2f}%  "
          f"{avg_rem_700:>+10.2f}% {capture_700:>8.2f}%  "
          f"{winner:>7}")

print(f"\n  WINNERS: 6:30={win_630}  7:00={win_700}  TIE={tie}")
if remaining_630_all and remaining_700_all:
    print(f"  Overall avg remaining at 6:30: {np.mean(remaining_630_all)*100:+.2f}%")
    print(f"  Overall avg remaining at 7:00: {np.mean(remaining_700_all)*100:+.2f}%")
    print(f"  Difference (6:30 - 7:00):      {(np.mean(remaining_630_all) - np.mean(remaining_700_all))*100:+.2f}%")


# ── Step 5: Risk/Reward — max drawdown from entry ────────────────────────────
print(f"\n{'='*100}")
print(f"  ANALYSIS 4: RISK PROFILE (avg max drawdown from entry)")
print(f"{'='*100}")

print(f"\n  {'Ticker':<7}", end="")
for h in [15, 30, 60]:
    print(f"  {'6:30 DD'+str(h):>10}  {'7:00 DD'+str(h):>10}", end="")
print()
print(f"  {'─'*7}", end="")
for h in [15, 30, 60]:
    print(f"  {'─'*10}  {'─'*10}", end="")
print()

agg_dd_630 = {h: [] for h in [15, 30, 60]}
agg_dd_700 = {h: [] for h in [15, 30, 60]}

for ticker in FOCUS_TICKERS:
    k630 = (ticker, '6:30 AM')
    k700 = (ticker, '7:00 AM')
    if k630 not in all_stats and k700 not in all_stats:
        continue
    print(f"  {ticker:<7}", end="")
    for h in [15, 30, 60]:
        for key, agg in [(k630, agg_dd_630), (k700, agg_dd_700)]:
            if key in all_stats:
                col = f'max_low_ret_{h}'
                vals = all_stats[key][col].dropna()
                if len(vals) > 0:
                    avg_dd = vals.mean() * 100
                    agg[h].append(avg_dd)
                    print(f"  {avg_dd:>+9.2f}%", end="")
                else:
                    print(f"  {'—':>10}", end="")
            else:
                print(f"  {'—':>10}", end="")
    print()

print(f"\n  {'AVERAGE':<7}", end="")
for h in [15, 30, 60]:
    for agg in [agg_dd_630, agg_dd_700]:
        if agg[h]:
            print(f"  {np.mean(agg[h]):>+9.2f}%", end="")
        else:
            print(f"  {'—':>10}", end="")
print()


# ── Step 6: Reward-to-Risk ratio ──────────────────────────────────────────────
print(f"\n{'='*100}")
print(f"  ANALYSIS 5: REWARD/RISK RATIO (avg max-high / abs(avg max-low))")
print(f"{'='*100}")

print(f"\n  {'Ticker':<7}", end="")
for h in [15, 30, 60]:
    print(f"  {'6:30 RR'+str(h):>10}  {'7:00 RR'+str(h):>10}", end="")
print()
print(f"  {'─'*7}", end="")
for h in [15, 30, 60]:
    print(f"  {'─'*10}  {'─'*10}", end="")
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
                hi_col = f'max_high_ret_{h}'
                lo_col = f'max_low_ret_{h}'
                hi_vals = all_stats[key][hi_col].dropna()
                lo_vals = all_stats[key][lo_col].dropna()
                if len(hi_vals) > 0 and len(lo_vals) > 0:
                    avg_hi = hi_vals.mean()
                    avg_lo = abs(lo_vals.mean())
                    rr = avg_hi / avg_lo if avg_lo > 1e-6 else 0
                    agg[h].append(rr)
                    print(f"  {rr:>9.2f}x", end="")
                else:
                    print(f"  {'—':>10}", end="")
            else:
                print(f"  {'—':>10}", end="")
    print()

print(f"\n  {'AVERAGE':<7}", end="")
for h in [15, 30, 60]:
    for agg in [agg_rr_630, agg_rr_700]:
        if agg[h]:
            print(f"  {np.mean(agg[h]):>9.2f}x", end="")
        else:
            print(f"  {'—':>10}", end="")
print()


# ── Step 7: Per-ticker optimal entry time ─────────────────────────────────────
print(f"\n{'='*100}")
print(f"  ANALYSIS 6: PER-TICKER OPTIMAL ENTRY TIME (best RR at 30-min horizon)")
print(f"{'='*100}")

print(f"\n  {'Ticker':<7} {'6:30 RR':>8} {'7:00 RR':>8} {'6:30 spike%':>11} {'7:00 spike%':>11} {'Best':>6}")
print(f"  {'─'*60}")

best_per_ticker = {}
for ticker in FOCUS_TICKERS:
    k630 = (ticker, '6:30 AM')
    k700 = (ticker, '7:00 AM')
    if k630 not in all_stats or k700 not in all_stats:
        continue

    h = 30
    rr_630 = rr_700 = freq_630 = freq_700 = 0

    for key, label in [(k630, '6:30'), (k700, '7:00')]:
        hi = all_stats[key][f'max_high_ret_{h}'].dropna()
        lo = all_stats[key][f'max_low_ret_{h}'].dropna()
        if len(hi) > 0 and len(lo) > 0:
            rr = hi.mean() / max(abs(lo.mean()), 1e-6)
            freq = (hi >= 0.06).mean() * 100
            if label == '6:30':
                rr_630, freq_630 = rr, freq
            else:
                rr_700, freq_700 = rr, freq

    best = '6:30' if rr_630 > rr_700 else '7:00'
    best_per_ticker[ticker] = best

    print(f"  {ticker:<7} {rr_630:>7.2f}x {rr_700:>7.2f}x {freq_630:>10.1f}% {freq_700:>10.1f}% {best:>6}")

n_630 = sum(1 for v in best_per_ticker.values() if v == '6:30')
n_700 = sum(1 for v in best_per_ticker.values() if v == '7:00')
print(f"\n  OVERALL: {n_630} tickers favor 6:30, {n_700} tickers favor 7:00")


# ── Step 8: Spike threshold sensitivity ──────────────────────────────────────
print(f"\n{'='*100}")
print(f"  ANALYSIS 7: SPIKE THRESHOLD SENSITIVITY (at 30-min horizon)")
print(f"{'='*100}")

print(f"\n  What % of trading days achieve >= X% max-high return within 30 min?")
print(f"\n  {'Threshold':<10}", end="")
print(f"  {'6:30 freq':>10}  {'7:00 freq':>10}  {'Delta':>8}")
print(f"  {'─'*45}")

for spike_t in SPIKE_THRESHOLDS:
    freq_630_vals = []
    freq_700_vals = []
    for ticker in FOCUS_TICKERS:
        for key, bag in [((ticker, '6:30 AM'), freq_630_vals),
                         ((ticker, '7:00 AM'), freq_700_vals)]:
            if key in all_stats:
                vals = all_stats[key]['max_high_ret_30'].dropna()
                if len(vals) > 0:
                    bag.append((vals >= spike_t).mean() * 100)

    avg_630 = np.mean(freq_630_vals) if freq_630_vals else 0
    avg_700 = np.mean(freq_700_vals) if freq_700_vals else 0
    delta = avg_630 - avg_700
    print(f"  {spike_t:>8.0%}    {avg_630:>9.2f}%  {avg_700:>9.2f}%  {delta:>+7.2f}%")


# ── Step 9: Time-to-spike analysis ────────────────────────────────────────────
print(f"\n{'='*100}")
print(f"  ANALYSIS 8: CUMULATIVE SPIKE PROBABILITY OVER TIME")
print(f"  (For each horizon 3–60 min, what % of days have reached 6% by then?)")
print(f"{'='*100}")

print(f"\n  {'Horizon':<10}", end="")
print(f"  {'6:30 cum%':>10}  {'7:00 cum%':>10}  {'Delta':>8}")
print(f"  {'─'*45}")

for h in FWD_HORIZONS:
    cum_630 = []
    cum_700 = []
    for ticker in FOCUS_TICKERS:
        for key, bag in [((ticker, '6:30 AM'), cum_630),
                         ((ticker, '7:00 AM'), cum_700)]:
            if key in all_stats:
                vals = all_stats[key][f'max_high_ret_{h}'].dropna()
                if len(vals) > 0:
                    bag.append((vals >= 0.06).mean() * 100)

    avg_630 = np.mean(cum_630) if cum_630 else 0
    avg_700 = np.mean(cum_700) if cum_700 else 0
    delta = avg_630 - avg_700
    print(f"  {h:>4} min    {avg_630:>9.2f}%  {avg_700:>9.2f}%  {delta:>+7.2f}%")


# ══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(3, 2, figsize=(20, 22))
fig.suptitle(
    f'Entry Time Analysis: 6:30 AM vs 7:00 AM PT\n'
    f'{STUDY_START} → {STUDY_END} | {len(FOCUS_TICKERS)} tickers | Focus: 6%+ spikes',
    fontsize=14, fontweight='bold')

# [0,0] Avg max-high return by horizon
ax = axes[0, 0]
x = np.arange(len(FWD_HORIZONS))
w = 0.35
vals_630 = [np.mean(agg_mhr_630[h]) if agg_mhr_630[h] else 0 for h in FWD_HORIZONS]
vals_700 = [np.mean(agg_mhr_700[h]) if agg_mhr_700[h] else 0 for h in FWD_HORIZONS]
bars1 = ax.bar(x - w/2, vals_630, w, label='6:30 AM', color='#1565C0', alpha=0.85)
bars2 = ax.bar(x + w/2, vals_700, w, label='7:00 AM', color='#FF6600', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([f'{h}m' for h in FWD_HORIZONS])
ax.set_ylabel('Avg Max-High Return (%)')
ax.set_title('Avg Max Upside by Horizon')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars1, vals_630):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{val:.2f}', ha='center', va='bottom', fontsize=7)
for bar, val in zip(bars2, vals_700):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{val:.2f}', ha='center', va='bottom', fontsize=7)

# [0,1] 6% spike frequency by horizon
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
                vals = all_stats[key][f'max_high_ret_{h}'].dropna()
                if len(vals) > 0:
                    bag.append((vals >= 0.06).mean() * 100)
    freq_630.append(np.mean(f630_vals) if f630_vals else 0)
    freq_700.append(np.mean(f700_vals) if f700_vals else 0)

ax.plot(FWD_HORIZONS, freq_630, 'o-', color='#1565C0', linewidth=2,
        markersize=8, label='6:30 AM')
ax.plot(FWD_HORIZONS, freq_700, 's--', color='#FF6600', linewidth=2,
        markersize=8, label='7:00 AM')
ax.set_xlabel('Forward Horizon (min)')
ax.set_ylabel('% of Days with ≥6% Spike')
ax.set_title('6% Spike Frequency by Horizon')
ax.legend()
ax.grid(True, alpha=0.3)

# [1,0] Reward/Risk ratio by horizon
ax = axes[1, 0]
rr_630 = [np.mean(agg_rr_630[h]) if agg_rr_630.get(h) else 0 for h in [15, 30, 60]]
rr_700 = [np.mean(agg_rr_700[h]) if agg_rr_700.get(h) else 0 for h in [15, 30, 60]]
x_rr = np.arange(3)
ax.bar(x_rr - w/2, rr_630, w, label='6:30 AM', color='#1565C0', alpha=0.85)
ax.bar(x_rr + w/2, rr_700, w, label='7:00 AM', color='#FF6600', alpha=0.85)
ax.set_xticks(x_rr)
ax.set_xticklabels(['15 min', '30 min', '60 min'])
ax.set_ylabel('Reward/Risk Ratio')
ax.set_title('Reward/Risk (max-high / |max-low|)')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

# [1,1] Risk profile (max drawdown)
ax = axes[1, 1]
dd_630 = [abs(np.mean(agg_dd_630[h])) if agg_dd_630.get(h) else 0 for h in [15, 30, 60]]
dd_700 = [abs(np.mean(agg_dd_700[h])) if agg_dd_700.get(h) else 0 for h in [15, 30, 60]]
ax.bar(x_rr - w/2, dd_630, w, label='6:30 AM', color='#1565C0', alpha=0.85)
ax.bar(x_rr + w/2, dd_700, w, label='7:00 AM', color='#FF6600', alpha=0.85)
ax.set_xticks(x_rr)
ax.set_xticklabels(['15 min', '30 min', '60 min'])
ax.set_ylabel('Avg Max Drawdown from Entry (%)')
ax.set_title('Risk: Avg Max Drawdown')
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
    ax.axvline(x=30, color='#1565C0', linestyle='--', alpha=0.7, label='6:30 AM')
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

# [2,1] Spike-day intraday shape
ax = axes[2, 1]
all_spike_avg = [v[0] for v in all_spike_shapes.values()]
if all_spike_avg:
    combined_spike = np.nanmean(all_spike_avg, axis=0) * 100
    combined_spike_std = np.nanstd(all_spike_avg, axis=0) * 100
    total_spike_days = sum(v[2] for v in all_spike_shapes.values())
    ax.plot(minutes_x, combined_spike, color='#d32f2f', linewidth=2)
    ax.fill_between(minutes_x,
                    combined_spike - combined_spike_std,
                    combined_spike + combined_spike_std,
                    alpha=0.15, color='#d32f2f')
    ax.axvline(x=30, color='#1565C0', linestyle='--', alpha=0.7, label='6:30 AM')
    ax.axvline(x=60, color='#FF6600', linestyle='--', alpha=0.7, label='7:00 AM')
    ax.axhline(y=6, color='green', linestyle=':', alpha=0.5, label='6% target')
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([time_labels[i] if i < len(time_labels) else '' for i in tick_positions],
                       rotation=45, fontsize=8)
    ax.set_ylabel('Return from Open (%)')
    ax.set_title(f'Spike-Day Shape (6%+ days only, {total_spike_days} ticker-days)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linewidth=0.5)

plt.tight_layout()
chart_path = f'{LOCAL_DIR}/entry_time_630_vs_700.png'
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
if _IN_COLAB:
    plt.show()
plt.close()
print(f"\nChart saved: {chart_path}")


# ── Chart 2: Per-ticker comparison heatmap ────────────────────────────────────
tickers_with_data = [t for t in FOCUS_TICKERS
                     if (t, '6:30 AM') in all_stats and (t, '7:00 AM') in all_stats]

if tickers_with_data:
    fig2, ax2 = plt.subplots(figsize=(10, max(8, len(tickers_with_data) * 0.35)))

    delta_vals = []
    for ticker in tickers_with_data:
        hi_630 = all_stats[(ticker, '6:30 AM')]['max_high_ret_30'].dropna().mean()
        hi_700 = all_stats[(ticker, '7:00 AM')]['max_high_ret_30'].dropna().mean()
        delta_vals.append((hi_630 - hi_700) * 100)

    sorted_idx = np.argsort(delta_vals)
    sorted_tickers = [tickers_with_data[i] for i in sorted_idx]
    sorted_deltas = [delta_vals[i] for i in sorted_idx]

    colors = ['#1565C0' if d > 0 else '#FF6600' for d in sorted_deltas]
    ax2.barh(range(len(sorted_tickers)), sorted_deltas, color=colors, alpha=0.85)
    ax2.set_yticks(range(len(sorted_tickers)))
    ax2.set_yticklabels(sorted_tickers, fontsize=8)
    ax2.axvline(x=0, color='black', linewidth=0.8)
    ax2.set_xlabel('Max-High Return Advantage (6:30 minus 7:00) at 30-min horizon (%)')
    ax2.set_title('Per-Ticker: 6:30 AM Advantage over 7:00 AM\n'
                  'Blue = 6:30 better | Orange = 7:00 better',
                  fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    chart2_path = f'{LOCAL_DIR}/entry_time_per_ticker_delta.png'
    plt.savefig(chart2_path, dpi=150, bbox_inches='tight')
    if _IN_COLAB:
        plt.show()
    plt.close()
    print(f"Chart 2 saved: {chart2_path}")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY & RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*100}")
print(f"  SUMMARY & RECOMMENDATIONS")
print(f"{'='*100}")

print(f"\n  STUDY: {STUDY_START} → {STUDY_END}, {len(FOCUS_TICKERS)} tickers")
print(f"  Focus: LONG bets, 3-60 min hold, 6-8% spike targets\n")

# Compute summary stats
for h in [15, 30, 60]:
    avg_hi_630 = np.mean(agg_mhr_630[h]) if agg_mhr_630[h] else 0
    avg_hi_700 = np.mean(agg_mhr_700[h]) if agg_mhr_700[h] else 0
    print(f"  {h}-min horizon: 6:30 avg max-high={avg_hi_630:+.2f}%  vs  "
          f"7:00 avg max-high={avg_hi_700:+.2f}%  "
          f"(delta={avg_hi_630-avg_hi_700:+.2f}%)")

if remaining_630_all and remaining_700_all:
    print(f"\n  On 6%+ spike days:")
    print(f"    Remaining upside at 6:30: {np.mean(remaining_630_all)*100:+.2f}%")
    print(f"    Remaining upside at 7:00: {np.mean(remaining_700_all)*100:+.2f}%")
    print(f"    6:30 captures {(np.mean(remaining_630_all) - np.mean(remaining_700_all))*100:.2f}% MORE of the move")

print(f"\n  Per-ticker vote: {n_630} favor 6:30, {n_700} favor 7:00")

total_elapsed = time.time() - t_total
print(f"\n  Total runtime: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
print(f"{'='*100}")
