"""
V4 Scanner Optimization: Experiments E17 and E18
=================================================
E17: Whipsaw Exhaustion Signal — Dynamic Entry
    Signal A: Volatility Collapse (rolling vol drops below peak fraction)
    Signal B: Range Stabilization (ORB midpoint crossing rate collapse)

E18: Entry at Local Price Extremes
    Rule 1: First local high after open (window=3)
    Rule 2: Highest high in whipsaw window (theoretical ceiling)
    Rule 3: Last local high before commitment

Data: v2_features_full.csv (316 trades), v3_minute_bars.json, v2_daily_data.json
Focus: SHORT trades only with T+2 daily data available.
"""

import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from pathlib import Path
from collections import OrderedDict

warnings.filterwarnings('ignore')

# --- Config ------------------------------------------------------------------
BASE = Path('os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../') + '/sentiment/scanner_optimization')
RESULTS = BASE / 'results'
CSV_PATH = RESULTS / 'v2_features_full.csv'
MINUTE_PATH = RESULTS / 'v3_minute_bars.json'
DAILY_PATH = RESULTS / 'v2_daily_data.json'

LOW_N_THRESHOLD = 20

# --- Load data ---------------------------------------------------------------
print("=" * 80)
print("V4 E17/E18: DYNAMIC ENTRY FOR SHORT TRADES")
print("=" * 80)

df = pd.read_csv(CSV_PATH)
with open(MINUTE_PATH) as f:
    minute_data = json.load(f)
ticker_bars = minute_data['ticker_bars']
with open(DAILY_PATH) as f:
    daily_data = json.load(f)

print(f"CSV: {len(df)} trades")
print(f"Minute bars: {len(ticker_bars)} ticker-dates")
print(f"Daily data: {len(daily_data)} tickers")


# --- Helper functions --------------------------------------------------------

def trimmed_mean(arr, pct=0.05):
    """Trimmed mean at 5-95%."""
    arr = np.array([x for x in arr if np.isfinite(x)])
    if len(arr) < 3:
        return np.nan
    return float(sp_stats.trim_mean(arr, pct))


def sharpe(arr):
    """Sharpe ratio (mean / std)."""
    arr = np.array([x for x in arr if np.isfinite(x)])
    if len(arr) < 3 or np.std(arr) == 0:
        return np.nan
    return float(np.mean(arr) / np.std(arr))


def metrics_dict(arr, label=""):
    """Compute standard metrics for a PnL array."""
    arr = np.array([x for x in arr if np.isfinite(x)])
    n = len(arr)
    if n == 0:
        return {'label': label, 'n': 0, 'avg': np.nan, 'trimmed': np.nan,
                'median': np.nan, 'win_rate': np.nan, 'sharpe': np.nan, 'low_n': True}
    return {
        'label': label,
        'n': n,
        'avg': float(np.mean(arr)),
        'trimmed': trimmed_mean(arr),
        'median': float(np.median(arr)),
        'win_rate': float(np.sum(arr > 0) / n * 100),
        'sharpe': sharpe(arr),
        'low_n': n < LOW_N_THRESHOLD,
    }


def print_metrics_table(metrics_list, title=""):
    """Print a formatted table of metrics."""
    if title:
        print(f"\n{'~' * 70}")
        print(f"  {title}")
        print(f"{'~' * 70}")
    header = f"{'Strategy':<40} {'n':>5} {'Avg%':>8} {'Trim%':>8} {'Med%':>8} {'WR%':>6} {'Sharpe':>7} {'Flag':>5}"
    print(header)
    print("-" * len(header))
    for m in metrics_list:
        flag = " *" if m.get('low_n') else ""
        avg_s = f"{m['avg']:>8.2f}" if np.isfinite(m.get('avg', np.nan)) else "     N/A"
        trm_s = f"{m['trimmed']:>8.2f}" if np.isfinite(m.get('trimmed', np.nan)) else "     N/A"
        med_s = f"{m['median']:>8.2f}" if np.isfinite(m.get('median', np.nan)) else "     N/A"
        wr_s = f"{m['win_rate']:>6.1f}" if np.isfinite(m.get('win_rate', np.nan)) else "   N/A"
        sh_s = f"{m['sharpe']:>7.3f}" if np.isfinite(m.get('sharpe', np.nan)) else "    N/A"
        print(f"{m['label']:<40} {m['n']:>5} {avg_s} {trm_s} {med_s} {wr_s} {sh_s}{flag}")
    if any(m.get('low_n') for m in metrics_list):
        print("  * = low-n (< 20), interpret with caution")


def get_t2_close(ticker, eval_date):
    """Get closing price 2 trading days after eval_date from daily data."""
    if ticker not in daily_data:
        return np.nan
    dates = sorted(daily_data[ticker].keys())
    if eval_date not in dates:
        return np.nan
    idx = dates.index(eval_date)
    if idx + 2 >= len(dates):
        return np.nan
    t2_date = dates[idx + 2]
    return daily_data[ticker][t2_date]['close']


def get_minute_series(ticker, eval_date, start='09:30', end='10:30'):
    """Get sorted minute bar series as list of (time_str, bar_dict).
    Returns only bars in [start, end] inclusive."""
    key = f"{ticker}__{eval_date}"
    if key not in ticker_bars:
        return []
    bars = ticker_bars[key]
    result = []
    for t in sorted(bars.keys()):
        if t < start:
            continue
        if t > end:
            break
        result.append((t, bars[t]))
    return result


def time_to_minutes_after_930(t_str):
    """Convert HH:MM to minutes after 09:30."""
    h, m = map(int, t_str.split(':'))
    return (h - 9) * 60 + m - 30


def short_pnl(entry_price, t2_close):
    """PnL for a SHORT trade: (entry - exit) / entry * 100."""
    if np.isnan(entry_price) or np.isnan(t2_close) or entry_price == 0:
        return np.nan
    return (entry_price - t2_close) / entry_price * 100.0


# --- Build SHORT dataset with T+2 close -------------------------------------
short_df = df[df.direction == 'SHORT'].copy()
print(f"\nSHORT trades total: {len(short_df)}")

# Add T+2 close price
short_df['t2_close_price'] = short_df.apply(
    lambda r: get_t2_close(r['ticker'], r['eval_date']), axis=1)
short_df = short_df.dropna(subset=['t2_close_price']).copy()
print(f"SHORT trades with T+2 data: {len(short_df)}")

# Verify entry price at 09:30
# (entry_price in CSV is the 09:30 open)
# Baseline PnL from the CSV
short_df['baseline_pnl'] = short_df.apply(
    lambda r: short_pnl(r['entry_price'], r['t2_close_price']), axis=1)
print(f"Baseline 09:30 entry avg PnL: {short_df['baseline_pnl'].mean():.2f}%")


# =============================================================================
# BASELINES: 09:30 and 10:00 fixed entries
# =============================================================================
print("\n" + "=" * 80)
print("BASELINES")
print("=" * 80)

baseline_0930_pnls = short_df['baseline_pnl'].values.copy()

# Fixed 10:00 entry
fixed_1000_pnls = []
for _, row in short_df.iterrows():
    bars = get_minute_series(row['ticker'], row['eval_date'], '10:00', '10:00')
    if bars:
        entry_p = bars[0][1]['c']
        fixed_1000_pnls.append(short_pnl(entry_p, row['t2_close_price']))
    else:
        fixed_1000_pnls.append(np.nan)
fixed_1000_pnls = np.array(fixed_1000_pnls)

baselines = [
    metrics_dict(baseline_0930_pnls, "09:30 fixed entry"),
    metrics_dict(fixed_1000_pnls, "10:00 fixed entry"),
]
print_metrics_table(baselines, "SHORT Baselines (T+2 close)")


# =============================================================================
# E17: WHIPSAW EXHAUSTION SIGNAL
# =============================================================================
print("\n" + "=" * 80)
print("E17: WHIPSAW EXHAUSTION SIGNAL -- DYNAMIC ENTRY")
print("=" * 80)


# --- Signal A: Volatility Collapse -------------------------------------------
print("\n--- Signal A: Volatility Collapse ---")

def compute_vol_collapse(ticker, eval_date, threshold_frac, vol_window=5):
    """
    Compute rolling realized volatility from minute closes (09:30-10:30).
    Find peak_vol in 09:30-09:45.
    Find exhaustion_time = first minute after 09:35 where rolling_vol < threshold * peak_vol.
    Returns (exhaustion_time_str, entry_price, delay_minutes) or (None, NaN, NaN).
    """
    bars = get_minute_series(ticker, eval_date, '09:30', '10:30')
    if len(bars) < vol_window + 2:
        return None, np.nan, np.nan

    times = [b[0] for b in bars]
    closes = [b[1]['c'] for b in bars]

    # Compute log returns
    log_rets = [np.nan]  # first bar has no return
    for i in range(1, len(closes)):
        if closes[i - 1] > 0 and closes[i] > 0:
            log_rets.append(np.log(closes[i] / closes[i - 1]))
        else:
            log_rets.append(np.nan)

    # Rolling volatility (std of log returns over vol_window bars)
    rolling_vol = [np.nan] * len(log_rets)
    for i in range(vol_window, len(log_rets)):
        window_rets = log_rets[i - vol_window + 1:i + 1]
        valid = [r for r in window_rets if np.isfinite(r)]
        if len(valid) >= 2:
            rolling_vol[i] = np.std(valid)

    # Find peak_vol in 09:30-09:45
    peak_vol = 0
    for i, t in enumerate(times):
        if t > '09:45':
            break
        if np.isfinite(rolling_vol[i]) and rolling_vol[i] > peak_vol:
            peak_vol = rolling_vol[i]

    if peak_vol == 0:
        return None, np.nan, np.nan

    # Find exhaustion_time: first minute after 09:35 where rolling_vol < threshold * peak_vol
    for i, t in enumerate(times):
        if t <= '09:35':
            continue
        if np.isfinite(rolling_vol[i]) and rolling_vol[i] < threshold_frac * peak_vol:
            delay = time_to_minutes_after_930(t)
            return t, closes[i], delay

    return None, np.nan, np.nan


vol_thresholds = [0.3, 0.5, 0.7]
vol_results = {}

for thresh in vol_thresholds:
    pnls = []
    delays = []
    entry_prices_dyn = []
    for _, row in short_df.iterrows():
        t_exh, p_exh, delay = compute_vol_collapse(
            row['ticker'], row['eval_date'], thresh, vol_window=5)
        if t_exh is not None and np.isfinite(p_exh):
            pnl = short_pnl(p_exh, row['t2_close_price'])
            pnls.append(pnl)
            delays.append(delay)
            entry_prices_dyn.append(p_exh)
        else:
            pnls.append(np.nan)
            delays.append(np.nan)
            entry_prices_dyn.append(np.nan)

    pnls = np.array(pnls)
    delays = np.array(delays)
    valid = np.isfinite(pnls)
    vol_results[thresh] = {
        'pnls': pnls,
        'delays': delays,
        'entry_prices': np.array(entry_prices_dyn),
        'metrics': metrics_dict(pnls[valid], f"Vol-collapse {thresh}x"),
        'avg_delay': float(np.nanmean(delays)) if np.any(np.isfinite(delays)) else np.nan,
    }
    print(f"  {thresh}x threshold: {int(valid.sum())} signals, "
          f"avg delay = {vol_results[thresh]['avg_delay']:.1f} min, "
          f"avg PnL = {np.nanmean(pnls[valid]):.2f}%")

# Print comparison table
vol_metrics_list = [metrics_dict(baseline_0930_pnls, "09:30 baseline")]
for thresh in vol_thresholds:
    vol_metrics_list.append(vol_results[thresh]['metrics'])
print_metrics_table(vol_metrics_list, "E17 Signal A: Volatility Collapse Comparison")


# --- Signal B: Range Stabilization (Crossing Rate Collapse) -------------------
print("\n--- Signal B: Range Stabilization ---")

def compute_range_stabilization(ticker, eval_date, or_high, or_low):
    """
    Track ORB midpoint crossings. Find stabilization_time where crossing_rate=0
    for 3 consecutive minutes after 09:35.
    Returns (stab_time, entry_price, delay, position_status) or (None, NaN, NaN, None).
    position_status: 'confirmed' (price < or_low), 'skip' (price > or_high), 'limbo' (inside range)
    """
    if pd.isna(or_high) or pd.isna(or_low):
        return None, np.nan, np.nan, None

    mid = (or_high + or_low) / 2.0
    bars = get_minute_series(ticker, eval_date, '09:30', '10:30')
    if len(bars) < 10:
        return None, np.nan, np.nan, None

    times = [b[0] for b in bars]
    closes = [b[1]['c'] for b in bars]

    # Track which side of mid each close is on: True = above
    above_mid = [c > mid for c in closes]

    # Compute crossings: a crossing occurs when above_mid changes from bar i-1 to bar i
    crossings = [0] * len(bars)
    for i in range(1, len(bars)):
        if above_mid[i] != above_mid[i - 1]:
            crossings[i] = 1

    # Rolling crossing rate over last 5 minutes
    cross_rate = [np.nan] * len(bars)
    for i in range(4, len(bars)):
        cross_rate[i] = sum(crossings[i - 4:i + 1])

    # Find stabilization: 3 consecutive minutes with cross_rate == 0 after 09:35
    consec_zero = 0
    for i, t in enumerate(times):
        if t <= '09:35':
            consec_zero = 0
            continue
        if np.isfinite(cross_rate[i]) and cross_rate[i] == 0:
            consec_zero += 1
        else:
            consec_zero = 0

        if consec_zero >= 3:
            # Stabilization found at this minute
            stab_price = closes[i]
            delay = time_to_minutes_after_930(t)

            if stab_price < or_low:
                status = 'confirmed'
            elif stab_price > or_high:
                status = 'skip'
            else:
                status = 'limbo'

            return t, stab_price, delay, status

    return None, np.nan, np.nan, None


stab_pnls_confirmed = []
stab_pnls_all = []  # confirmed + limbo
stab_pnls_limbo = []
stab_delays = []
stab_statuses = []

for _, row in short_df.iterrows():
    t_stab, p_stab, delay, status = compute_range_stabilization(
        row['ticker'], row['eval_date'], row['or_high'], row['or_low'])

    if t_stab is not None and np.isfinite(p_stab):
        pnl = short_pnl(p_stab, row['t2_close_price'])
        stab_delays.append(delay)
        stab_statuses.append(status)

        if status == 'confirmed':
            stab_pnls_confirmed.append(pnl)
            stab_pnls_all.append(pnl)
        elif status == 'limbo':
            stab_pnls_limbo.append(pnl)
            stab_pnls_all.append(pnl)
        # skip = don't trade
    else:
        stab_statuses.append(None)

stab_delays = np.array(stab_delays)
status_counts = pd.Series([s for s in stab_statuses if s is not None]).value_counts()
print(f"  Stabilization signals: {len(stab_delays)} total")
print(f"  Status breakdown: {status_counts.to_dict()}")
print(f"  Avg delay: {np.nanmean(stab_delays):.1f} min")

stab_metrics_list = [
    metrics_dict(baseline_0930_pnls, "09:30 baseline"),
    metrics_dict(np.array(stab_pnls_confirmed), "Range-stab: confirmed"),
    metrics_dict(np.array(stab_pnls_limbo), "Range-stab: limbo"),
    metrics_dict(np.array(stab_pnls_all), "Range-stab: confirmed+limbo"),
]
print_metrics_table(stab_metrics_list, "E17 Signal B: Range Stabilization Comparison")


# =============================================================================
# E18: ENTRY AT LOCAL PRICE EXTREMES
# =============================================================================
print("\n" + "=" * 80)
print("E18: ENTRY AT LOCAL PRICE EXTREMES")
print("=" * 80)


def find_local_highs(closes, times, window=3, start_offset=None):
    """Find local highs: points higher than all bars in [i-window, i+window].
    Returns list of (index, time_str, close_price)."""
    if start_offset is None:
        start_offset = window
    highs = []
    for i in range(max(window, start_offset), len(closes) - window):
        is_high = True
        for w in range(-window, window + 1):
            if w == 0:
                continue
            if closes[i + w] >= closes[i]:
                is_high = False
                break
        if is_high:
            highs.append((i, times[i], closes[i]))
    return highs


# --- Rule 1: First Local High After Open ------------------------------------
print("\n--- Rule 1: First Local High After Open ---")

r1_pnls = []
r1_delays = []
r1_improvements = []  # % improvement over 09:30 entry price

for _, row in short_df.iterrows():
    bars = get_minute_series(row['ticker'], row['eval_date'], '09:30', '10:00')
    if len(bars) < 7:
        r1_pnls.append(np.nan)
        r1_delays.append(np.nan)
        r1_improvements.append(np.nan)
        continue

    times = [b[0] for b in bars]
    closes = [b[1]['c'] for b in bars]

    highs = find_local_highs(closes, times, window=3)
    if highs:
        first_high = highs[0]
        entry_p = first_high[2]
        pnl = short_pnl(entry_p, row['t2_close_price'])
        delay = time_to_minutes_after_930(first_high[1])
        # Price improvement: higher entry = better for SHORT
        improvement = (entry_p - row['entry_price']) / row['entry_price'] * 100
        r1_pnls.append(pnl)
        r1_delays.append(delay)
        r1_improvements.append(improvement)
    else:
        r1_pnls.append(np.nan)
        r1_delays.append(np.nan)
        r1_improvements.append(np.nan)

r1_pnls = np.array(r1_pnls)
r1_delays = np.array(r1_delays)
r1_improvements = np.array(r1_improvements)
r1_valid = np.isfinite(r1_pnls)
print(f"  Signals: {int(r1_valid.sum())}, avg delay: {np.nanmean(r1_delays):.1f} min")
print(f"  Avg entry price improvement: {np.nanmean(r1_improvements):.2f}%")


# --- Rule 2: Highest High in Whipsaw Window (Theoretical Ceiling) -----------
print("\n--- Rule 2: Highest High in Whipsaw Window (Ceiling) ---")

r2_pnls = []
r2_delays = []
r2_improvements = []

for _, row in short_df.iterrows():
    bars = get_minute_series(row['ticker'], row['eval_date'], '09:30', '10:00')
    if len(bars) < 2:
        r2_pnls.append(np.nan)
        r2_delays.append(np.nan)
        r2_improvements.append(np.nan)
        continue

    times = [b[0] for b in bars]
    closes = [b[1]['c'] for b in bars]

    max_idx = np.argmax(closes)
    entry_p = closes[max_idx]
    pnl = short_pnl(entry_p, row['t2_close_price'])
    delay = time_to_minutes_after_930(times[max_idx])
    improvement = (entry_p - row['entry_price']) / row['entry_price'] * 100

    r2_pnls.append(pnl)
    r2_delays.append(delay)
    r2_improvements.append(improvement)

r2_pnls = np.array(r2_pnls)
r2_delays = np.array(r2_delays)
r2_improvements = np.array(r2_improvements)
r2_valid = np.isfinite(r2_pnls)
print(f"  Signals: {int(r2_valid.sum())}, avg delay: {np.nanmean(r2_delays):.1f} min")
print(f"  Avg entry price improvement: {np.nanmean(r2_improvements):.2f}%")


# --- Rule 3: Last Local High Before Commitment ------------------------------
print("\n--- Rule 3: Last Local High Before Commitment ---")

def find_commitment_time(closes, times, or_low, consec_needed=3):
    """Find first time price closes below or_low for consec_needed consecutive minutes."""
    consec = 0
    for i, c in enumerate(closes):
        if c < or_low:
            consec += 1
        else:
            consec = 0
        if consec >= consec_needed:
            return i, times[i]
    return None, None


r3_pnls = []
r3_delays = []
r3_improvements = []
r3_commitment_delays = []

for _, row in short_df.iterrows():
    bars = get_minute_series(row['ticker'], row['eval_date'], '09:30', '10:30')
    if len(bars) < 7:
        r3_pnls.append(np.nan)
        r3_delays.append(np.nan)
        r3_improvements.append(np.nan)
        r3_commitment_delays.append(np.nan)
        continue

    times = [b[0] for b in bars]
    closes = [b[1]['c'] for b in bars]

    # Find all local highs
    all_highs = find_local_highs(closes, times, window=3)

    # Find commitment time
    commit_idx, commit_time = find_commitment_time(closes, times, row['or_low'], consec_needed=3)

    if commit_idx is not None and all_highs:
        # Find the last local high BEFORE commitment time
        pre_commit_highs = [h for h in all_highs if h[0] < commit_idx]
        if pre_commit_highs:
            last_high = pre_commit_highs[-1]
            entry_p = last_high[2]
            pnl = short_pnl(entry_p, row['t2_close_price'])
            delay = time_to_minutes_after_930(last_high[1])
            improvement = (entry_p - row['entry_price']) / row['entry_price'] * 100
            r3_pnls.append(pnl)
            r3_delays.append(delay)
            r3_improvements.append(improvement)
            r3_commitment_delays.append(time_to_minutes_after_930(commit_time))
        else:
            r3_pnls.append(np.nan)
            r3_delays.append(np.nan)
            r3_improvements.append(np.nan)
            r3_commitment_delays.append(np.nan)
    else:
        r3_pnls.append(np.nan)
        r3_delays.append(np.nan)
        r3_improvements.append(np.nan)
        r3_commitment_delays.append(np.nan)

r3_pnls = np.array(r3_pnls)
r3_delays = np.array(r3_delays)
r3_improvements = np.array(r3_improvements)
r3_commitment_delays = np.array(r3_commitment_delays)
r3_valid = np.isfinite(r3_pnls)
print(f"  Signals: {int(r3_valid.sum())}, avg delay: {np.nanmean(r3_delays):.1f} min")
print(f"  Avg entry price improvement: {np.nanmean(r3_improvements):.2f}%")
print(f"  Avg commitment delay: {np.nanmean(r3_commitment_delays):.1f} min")


# --- E18 Comparison Table ---------------------------------------------------
e18_metrics = [
    metrics_dict(baseline_0930_pnls, "09:30 baseline"),
    metrics_dict(r1_pnls[r1_valid], "Rule 1: First local high"),
    metrics_dict(r2_pnls[r2_valid], "Rule 2: Highest high (ceiling)"),
    metrics_dict(r3_pnls[r3_valid], "Rule 3: Last high before commit"),
]
print_metrics_table(e18_metrics, "E18: Local Price Extremes Comparison (T+2)")

# --- Key Insight: Theoretical vs Capturable ---------------------------------
print("\n--- Key Insight: Theoretical Improvement ---")
baseline_avg = np.nanmean(baseline_0930_pnls)
r2_avg = np.nanmean(r2_pnls[r2_valid])
r1_avg = np.nanmean(r1_pnls[r1_valid])
r3_avg = np.nanmean(r3_pnls[r3_valid])

theoretical_gap = r2_avg - baseline_avg
r1_capture = (r1_avg - baseline_avg) / theoretical_gap * 100 if theoretical_gap != 0 else np.nan
r3_capture = (r3_avg - baseline_avg) / theoretical_gap * 100 if theoretical_gap != 0 else np.nan

print(f"  Baseline (09:30):       {baseline_avg:.2f}%")
print(f"  Rule 2 (ceiling):       {r2_avg:.2f}%")
print(f"  Theoretical gap:        {theoretical_gap:.2f}% (ceiling - baseline)")
print(f"  Rule 1 avg:             {r1_avg:.2f}%  -> captures {r1_capture:.1f}% of theoretical")
print(f"  Rule 3 avg:             {r3_avg:.2f}%  -> captures {r3_capture:.1f}% of theoretical")

# Entry price improvements
print("\n--- Entry Price Improvement ---")
print(f"  Rule 1 avg improvement: {np.nanmean(r1_improvements):.2f}%")
print(f"  Rule 2 avg improvement: {np.nanmean(r2_improvements):.2f}%")
print(f"  Rule 3 avg improvement: {np.nanmean(r3_improvements):.2f}%")


# =============================================================================
# COMBINED SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("COMBINED SUMMARY: ALL STRATEGIES")
print("=" * 80)

all_metrics = [
    metrics_dict(baseline_0930_pnls, "09:30 baseline"),
    metrics_dict(fixed_1000_pnls, "10:00 fixed entry"),
]
for thresh in vol_thresholds:
    m = vol_results[thresh]['metrics'].copy()
    m['avg_delay'] = vol_results[thresh]['avg_delay']
    all_metrics.append(m)
all_metrics.append(metrics_dict(np.array(stab_pnls_confirmed), "Range-stab confirmed"))
all_metrics.append(metrics_dict(np.array(stab_pnls_all), "Range-stab conf+limbo"))
all_metrics.append(metrics_dict(r1_pnls[r1_valid], "E18 R1: First local high"))
all_metrics.append(metrics_dict(r2_pnls[r2_valid], "E18 R2: Highest (ceiling)"))
all_metrics.append(metrics_dict(r3_pnls[r3_valid], "E18 R3: Last before commit"))

print_metrics_table(all_metrics, "All E17+E18 Strategies vs Baseline")


# =============================================================================
# CHARTS: E17 (v4_e17_dynamic_entry.png)
# =============================================================================
print("\n--- Generating E17 charts ---")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('E17: Whipsaw Exhaustion Signal -- Dynamic Entry (SHORT)', fontsize=14, fontweight='bold')

# Panel 1: Bar chart strategy comparison
ax = axes[0, 0]
labels_e17 = ['09:30\nbaseline', '10:00\nfixed']
avgs_e17 = [np.nanmean(baseline_0930_pnls), np.nanmean(fixed_1000_pnls)]
colors_e17 = ['#555555', '#888888']
for thresh in vol_thresholds:
    labels_e17.append(f'Vol-col\n{thresh}x')
    avgs_e17.append(vol_results[thresh]['metrics']['avg'])
    colors_e17.append('#2196F3')
labels_e17.append('Range-stab\nconfirmed')
avgs_e17.append(np.nanmean(stab_pnls_confirmed) if stab_pnls_confirmed else 0)
colors_e17.append('#FF9800')
labels_e17.append('Range-stab\nconf+limbo')
avgs_e17.append(np.nanmean(stab_pnls_all) if stab_pnls_all else 0)
colors_e17.append('#FFC107')

x_e17 = np.arange(len(labels_e17))
bars_plot = ax.bar(x_e17, avgs_e17, color=colors_e17, edgecolor='black', linewidth=0.5)
ax.set_xticks(x_e17)
ax.set_xticklabels(labels_e17, fontsize=8)
ax.set_ylabel('Avg T+2 PnL (%)')
ax.set_title('Strategy Comparison: Avg T+2 PnL')
ax.axhline(y=0, color='black', linewidth=0.5)
# Add value labels on bars
for bar_obj, v in zip(bars_plot, avgs_e17):
    if np.isfinite(v):
        ax.text(bar_obj.get_x() + bar_obj.get_width()/2, v + 0.1,
                f'{v:.1f}%', ha='center', va='bottom', fontsize=8)

# Panel 2: Histogram of entry delay for vol-collapse (best threshold)
ax = axes[0, 1]
best_thresh = 0.5
valid_delays_vc = vol_results[best_thresh]['delays']
valid_delays_vc = valid_delays_vc[np.isfinite(valid_delays_vc)]
if len(valid_delays_vc) > 0:
    ax.hist(valid_delays_vc, bins=range(0, 65, 2), color='#2196F3', edgecolor='black',
            linewidth=0.5, alpha=0.8)
    ax.axvline(np.nanmean(valid_delays_vc), color='red', linestyle='--',
               label=f'Mean: {np.nanmean(valid_delays_vc):.1f} min')
    ax.legend(fontsize=9)
ax.set_xlabel('Minutes after 09:30')
ax.set_ylabel('Count')
ax.set_title(f'Vol-Collapse {best_thresh}x: Entry Delay Distribution')

# Panel 3: Histogram of entry delay for range stabilization
ax = axes[1, 0]
valid_delays_rs = stab_delays[np.isfinite(stab_delays)]
if len(valid_delays_rs) > 0:
    ax.hist(valid_delays_rs, bins=range(0, 65, 2), color='#FF9800', edgecolor='black',
            linewidth=0.5, alpha=0.8)
    ax.axvline(np.nanmean(valid_delays_rs), color='red', linestyle='--',
               label=f'Mean: {np.nanmean(valid_delays_rs):.1f} min')
    ax.legend(fontsize=9)
ax.set_xlabel('Minutes after 09:30')
ax.set_ylabel('Count')
ax.set_title('Range Stabilization: Entry Delay Distribution')

# Panel 4: Example trade visualizations
ax = axes[1, 1]

# Pick 2-3 example trades where both signals fire
example_trades = []
for idx_row, (_, row) in enumerate(short_df.iterrows()):
    t_vc, p_vc, d_vc = compute_vol_collapse(row['ticker'], row['eval_date'], 0.5, vol_window=5)
    t_rs, p_rs, d_rs, status_rs = compute_range_stabilization(
        row['ticker'], row['eval_date'], row['or_high'], row['or_low'])
    if t_vc is not None and t_rs is not None:
        example_trades.append((row, t_vc, p_vc, t_rs, p_rs, status_rs))
    if len(example_trades) >= 3:
        break

if example_trades:
    # Plot first example trade
    row, t_vc, p_vc, t_rs, p_rs, status_rs = example_trades[0]
    bars = get_minute_series(row['ticker'], row['eval_date'], '09:30', '10:30')
    if bars:
        ptimes = [time_to_minutes_after_930(b[0]) for b in bars]
        pcloses = [b[1]['c'] for b in bars]

        ax.plot(ptimes, pcloses, 'k-', linewidth=1, label='Price')
        ax.axhspan(row['or_low'], row['or_high'], alpha=0.15, color='blue', label='ORB range')
        ax.axhline(y=(row['or_high'] + row['or_low'])/2, color='blue', linestyle=':', alpha=0.5)

        # Mark vol-collapse entry
        vc_delay = time_to_minutes_after_930(t_vc)
        ax.axvline(x=vc_delay, color='#2196F3', linestyle='--', alpha=0.7)
        ax.plot(vc_delay, p_vc, 'v', color='#2196F3', markersize=12,
                label=f'Vol-collapse ({t_vc})')

        # Mark range-stab entry
        rs_delay = time_to_minutes_after_930(t_rs)
        ax.axvline(x=rs_delay, color='#FF9800', linestyle='--', alpha=0.7)
        ax.plot(rs_delay, p_rs, 's', color='#FF9800', markersize=10,
                label=f'Range-stab ({t_rs})')

        ax.set_xlabel('Minutes after 09:30')
        ax.set_ylabel('Price')
        ax.set_title(f'Example: {row["ticker"]} {row["eval_date"]}')
        ax.legend(fontsize=7, loc='best')

plt.tight_layout()
fig.savefig(str(BASE / 'v4_e17_dynamic_entry.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: v4_e17_dynamic_entry.png")


# =============================================================================
# CHARTS: E18 (v4_e18_local_extremes.png)
# =============================================================================
print("--- Generating E18 charts ---")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('E18: Entry at Local Price Extremes (SHORT)', fontsize=14, fontweight='bold')

# Panel 1: Bar chart rule comparison -- avg T+2 PnL
ax = axes[0, 0]
labels_e18 = ['09:30\nbaseline', 'R1: First\nlocal high', 'R2: Highest\n(ceiling)', 'R3: Last before\ncommit']
avgs_e18 = [baseline_avg, r1_avg, r2_avg, r3_avg]
ns_e18 = [len(baseline_0930_pnls), int(r1_valid.sum()), int(r2_valid.sum()), int(r3_valid.sum())]
colors_e18 = ['#555555', '#4CAF50', '#F44336', '#9C27B0']

bars_plot = ax.bar(np.arange(len(labels_e18)), avgs_e18, color=colors_e18,
                    edgecolor='black', linewidth=0.5)
ax.set_xticks(np.arange(len(labels_e18)))
ax.set_xticklabels(labels_e18, fontsize=9)
ax.set_ylabel('Avg T+2 PnL (%)')
ax.set_title('Rule Comparison: Avg T+2 PnL')
ax.axhline(y=0, color='black', linewidth=0.5)
for bar_obj, v, n in zip(bars_plot, avgs_e18, ns_e18):
    if np.isfinite(v):
        ax.text(bar_obj.get_x() + bar_obj.get_width()/2, v + 0.1,
                f'{v:.1f}%\n(n={n})', ha='center', va='bottom', fontsize=8)

# Panel 2: Bar chart -- entry price improvement over 09:30
ax = axes[0, 1]
labels_imp = ['R1: First\nlocal high', 'R2: Highest\n(ceiling)', 'R3: Last before\ncommit']
avgs_imp = [np.nanmean(r1_improvements), np.nanmean(r2_improvements), np.nanmean(r3_improvements)]
colors_imp = ['#4CAF50', '#F44336', '#9C27B0']

bars_plot = ax.bar(np.arange(len(labels_imp)), avgs_imp, color=colors_imp,
                    edgecolor='black', linewidth=0.5)
ax.set_xticks(np.arange(len(labels_imp)))
ax.set_xticklabels(labels_imp, fontsize=9)
ax.set_ylabel('Entry Price Improvement (%)')
ax.set_title('Entry Price Improvement over 09:30 Open')
ax.axhline(y=0, color='black', linewidth=0.5)
for bar_obj, v in zip(bars_plot, avgs_imp):
    if np.isfinite(v):
        ax.text(bar_obj.get_x() + bar_obj.get_width()/2, v + 0.05,
                f'{v:.2f}%', ha='center', va='bottom', fontsize=9)

# Panel 3: Scatter -- entry price improvement vs PnL improvement (Rule 3)
ax = axes[1, 0]
r3_pnl_improvement = r3_pnls - baseline_0930_pnls  # per-trade PnL improvement
valid_both = np.isfinite(r3_improvements) & np.isfinite(r3_pnl_improvement)
if valid_both.sum() > 0:
    ax.scatter(r3_improvements[valid_both], r3_pnl_improvement[valid_both],
               alpha=0.6, color='#9C27B0', s=40, edgecolors='black', linewidth=0.3)
    # Add trend line
    if valid_both.sum() > 2:
        z = np.polyfit(r3_improvements[valid_both], r3_pnl_improvement[valid_both], 1)
        p = np.poly1d(z)
        x_range = np.linspace(np.min(r3_improvements[valid_both]),
                               np.max(r3_improvements[valid_both]), 100)
        ax.plot(x_range, p(x_range), 'r--', alpha=0.5,
                label=f'slope={z[0]:.2f}')
        r_val = np.corrcoef(r3_improvements[valid_both], r3_pnl_improvement[valid_both])[0, 1]
        ax.legend(title=f'r={r_val:.2f}', fontsize=8)
ax.set_xlabel('Entry Price Improvement over 09:30 (%)')
ax.set_ylabel('PnL Improvement over Baseline (%)')
ax.set_title('Rule 3: Price Improvement vs PnL Improvement')
ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='-')
ax.axvline(x=0, color='gray', linewidth=0.5, linestyle='-')

# Panel 4: Example trades with local highs and commitment point
ax = axes[1, 1]

# Find a good example trade for Rule 3 (where commitment exists and highs are found)
example_found = False
for _, row in short_df.iterrows():
    bars = get_minute_series(row['ticker'], row['eval_date'], '09:30', '10:30')
    if len(bars) < 15:
        continue
    times_ex = [b[0] for b in bars]
    closes_ex = [b[1]['c'] for b in bars]

    all_highs = find_local_highs(closes_ex, times_ex, window=3)
    commit_idx, commit_time = find_commitment_time(closes_ex, times_ex, row['or_low'], consec_needed=3)

    if commit_idx is not None and len(all_highs) >= 2:
        pre_commit = [h for h in all_highs if h[0] < commit_idx]
        if len(pre_commit) >= 1:
            ptimes = [time_to_minutes_after_930(t) for t in times_ex]
            ax.plot(ptimes, closes_ex, 'k-', linewidth=1, label='Price')
            ax.axhspan(row['or_low'], row['or_high'], alpha=0.15, color='blue', label='ORB range')

            # Mark all local highs
            for h in all_highs:
                h_t = time_to_minutes_after_930(h[1])
                ax.plot(h_t, h[2], '^', color='#4CAF50', markersize=8)
            # Label first and last pre-commit
            first_h = all_highs[0]
            ax.plot(time_to_minutes_after_930(first_h[1]), first_h[2], '^',
                    color='#4CAF50', markersize=14, label=f'R1 entry ({first_h[1]})')

            last_pre = pre_commit[-1]
            ax.plot(time_to_minutes_after_930(last_pre[1]), last_pre[2], 'D',
                    color='#9C27B0', markersize=12, label=f'R3 entry ({last_pre[1]})')

            # Max close for R2
            max_idx = np.argmax(closes_ex)
            ax.plot(ptimes[max_idx], closes_ex[max_idx], '*',
                    color='#F44336', markersize=14, label=f'R2 ceiling ({times_ex[max_idx]})')

            # Commitment line
            c_t = time_to_minutes_after_930(commit_time)
            ax.axvline(x=c_t, color='red', linestyle='--', alpha=0.5,
                       label=f'Commit ({commit_time})')

            ax.set_xlabel('Minutes after 09:30')
            ax.set_ylabel('Price')
            ax.set_title(f'Example: {row["ticker"]} {row["eval_date"]}')
            ax.legend(fontsize=7, loc='best')
            example_found = True
            break

if not example_found:
    ax.text(0.5, 0.5, 'No suitable example found', ha='center', va='center',
            transform=ax.transAxes)

plt.tight_layout()
fig.savefig(str(BASE / 'v4_e18_local_extremes.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: v4_e18_local_extremes.png")


# =============================================================================
# SAVE RESULTS JSON
# =============================================================================
print("\n--- Saving results JSON ---")

def safe_float(v):
    if isinstance(v, (float, np.floating)):
        return float(v) if np.isfinite(v) else None
    return v


def clean_metrics(m):
    return {k: safe_float(v) for k, v in m.items()}


results_json = {
    'experiment': 'V4 E17+E18: Dynamic Entry for SHORT',
    'n_short_trades': len(short_df),
    'baselines': {
        '0930_fixed': clean_metrics(metrics_dict(baseline_0930_pnls, "09:30 fixed")),
        '1000_fixed': clean_metrics(metrics_dict(fixed_1000_pnls, "10:00 fixed")),
    },
    'E17_vol_collapse': {},
    'E17_range_stabilization': {
        'confirmed': clean_metrics(metrics_dict(np.array(stab_pnls_confirmed), "confirmed")),
        'limbo': clean_metrics(metrics_dict(np.array(stab_pnls_limbo), "limbo")),
        'all': clean_metrics(metrics_dict(np.array(stab_pnls_all), "confirmed+limbo")),
        'avg_delay': safe_float(np.nanmean(stab_delays)),
        'status_counts': {str(k): int(v) for k, v in status_counts.items()},
    },
    'E18_local_extremes': {
        'rule1_first_local_high': {
            **clean_metrics(metrics_dict(r1_pnls[r1_valid], "Rule 1")),
            'avg_delay': safe_float(np.nanmean(r1_delays)),
            'avg_price_improvement_pct': safe_float(np.nanmean(r1_improvements)),
        },
        'rule2_highest_ceiling': {
            **clean_metrics(metrics_dict(r2_pnls[r2_valid], "Rule 2")),
            'avg_delay': safe_float(np.nanmean(r2_delays)),
            'avg_price_improvement_pct': safe_float(np.nanmean(r2_improvements)),
        },
        'rule3_last_before_commit': {
            **clean_metrics(metrics_dict(r3_pnls[r3_valid], "Rule 3")),
            'avg_delay': safe_float(np.nanmean(r3_delays)),
            'avg_price_improvement_pct': safe_float(np.nanmean(r3_improvements)),
        },
    },
    'E18_theoretical_analysis': {
        'baseline_avg': safe_float(baseline_avg),
        'ceiling_avg': safe_float(r2_avg),
        'theoretical_gap': safe_float(theoretical_gap),
        'rule1_capture_pct': safe_float(r1_capture),
        'rule3_capture_pct': safe_float(r3_capture),
    },
}

# Add vol collapse results per threshold
for thresh in vol_thresholds:
    vr = vol_results[thresh]
    results_json['E17_vol_collapse'][f'{thresh}x'] = {
        **clean_metrics(vr['metrics']),
        'avg_delay': safe_float(vr['avg_delay']),
    }

with open(str(RESULTS / 'v4_e17_e18_results.json'), 'w') as f:
    json.dump(results_json, f, indent=2)
print(f"  Saved: results/v4_e17_e18_results.json")


print("\n" + "=" * 80)
print("DONE: V4 E17/E18 experiments complete.")
print("=" * 80)
