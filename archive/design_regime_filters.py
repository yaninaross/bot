"""
design_regime_filters.py -- Design & Backtest New Regime Filters for Spike Strategy
====================================================================================
Designs 5 candidate filters to detect the June 2026 regime shift, backtests each
filter April-June 2026, and ranks them by PnL improvement.

Filters:
  1. VIX Range Filter (intraday range expansion)
  2. SPY Drift Direction Filter (5-day return)
  3. Morning Reversal Detector (fraction of days with AM gains reversing by PM)
  4. Drawdown Expansion Filter (per-ticker recent max drawdown)
  5. Combined Regime Score (composite of best individual filters)

All metrics computed at entry time (6:00-6:30 AM PST) using only backward-looking data.

Run in Colab. Uses google.cloud.storage API (not FUSE).
"""

import numpy as np
import pandas as pd
import pytz
import warnings
from collections import defaultdict

warnings.filterwarnings('ignore')

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass

# ==============================================================================
# CONFIG
# ==============================================================================
BUCKET = os.environ.get("GCS_BUCKET", "")
OHLCV_DIR = "Data/Inference"
INFERENCE_V2_DIR = "Data/Inference_v2"
PST = pytz.timezone('America/Los_Angeles')

# Strategy tickers
TICKERS = ['KEEL', 'CIFR', 'CLSK', 'HUT', 'RIOT', 'RCAT', 'FLNC', 'PL', 'APLD', 'WULF']

# Regime context tickers
REGIME_FILES = {
    'VIX': f'gs://{BUCKET}/{OHLCV_DIR}/VIX_1m_full_historical_data.parquet',
    'SPY': f'gs://{BUCKET}/{OHLCV_DIR}/SPY_1m_full_historical_data.parquet',
}

# Analysis window
ANALYSIS_START = pd.Timestamp('2026-03-15', tz='UTC')  # warm-up for rolling windows
BACKTEST_START = pd.Timestamp('2026-04-01', tz='UTC')
BACKTEST_END   = pd.Timestamp('2026-06-30 23:59:59', tz='UTC')

# Entry window: 6:00-6:30 AM PST = 13:00-13:30 UTC
ENTRY_HOUR_PST = 6
ENTRY_MIN_START = 0
ENTRY_MIN_END = 30

# Regular trading hours PST
MORNING_START_MIN = 390   # 6:30 AM
MORNING_END_MIN   = 600   # 10:00 AM
CLOSE_MIN         = 780   # 1:00 PM

W = 120

# ==============================================================================
# DATA LOADING
# ==============================================================================
def load_ohlcv_from_gcs(uri):
    """Load OHLCV parquet from GCS uri."""
    df = pd.read_parquet(uri)
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df = df.sort_index()
    for c in ['Open', 'High', 'Low', 'Close']:
        if c in df.columns:
            df[c] = df[c].ffill()
    if 'Volume' not in df.columns:
        df['Volume'] = 0
    df['Volume'] = df['Volume'].fillna(0)
    return df


def load_regime_data():
    """Load VIX and SPY 1-minute data."""
    data = {}
    for ticker, uri in REGIME_FILES.items():
        print(f"  Loading {ticker}...", end=' ')
        try:
            df = load_ohlcv_from_gcs(uri)
            df = df[df.index >= ANALYSIS_START]
            data[ticker] = df
            print(f"OK ({len(df):,} bars, {df.index.min().date()} to {df.index.max().date()})")
        except Exception as e:
            print(f"FAILED: {e}")
    return data


def load_ticker_data():
    """Load strategy ticker 1-minute OHLCV data."""
    data = {}
    for ticker in TICKERS:
        # Try Inference first (raw OHLCV), fall back to Inference_v2
        uri1 = f'gs://{BUCKET}/{OHLCV_DIR}/{ticker}_1m_full_historical_data.parquet'
        uri2 = f'gs://{BUCKET}/{INFERENCE_V2_DIR}/{ticker}_master_inference.parquet'
        for uri in [uri1, uri2]:
            try:
                df = load_ohlcv_from_gcs(uri)
                df = df[df.index >= ANALYSIS_START]
                if len(df) > 0:
                    data[ticker] = df
                    print(f"  {ticker}: OK ({len(df):,} bars) from {uri.split('/')[-1]}")
                    break
            except Exception:
                continue
        if ticker not in data:
            print(f"  {ticker}: FAILED to load")
    return data


# ==============================================================================
# HELPER: Extract daily bars from 1-minute data
# ==============================================================================
def make_daily_bars(df_1m, start_min_pst=None, end_min_pst=None, require_volume=False):
    """
    Aggregate 1-minute data into daily bars.
    If start_min_pst/end_min_pst given, only include bars in that PST minute-of-day range.
    Returns DataFrame indexed by date with Open, High, Low, Close, Volume.
    """
    df = df_1m.copy()
    df['pst_time'] = df.index.tz_convert(PST)
    df['date'] = df['pst_time'].dt.date
    df['mins'] = df['pst_time'].dt.hour * 60 + df['pst_time'].dt.minute

    if start_min_pst is not None:
        df = df[df['mins'] >= start_min_pst]
    if end_min_pst is not None:
        df = df[df['mins'] <= end_min_pst]
    if require_volume:
        df = df[df['Volume'] > 0]

    if len(df) == 0:
        return pd.DataFrame()

    daily = df.groupby('date').agg(
        Open=('Open', 'first'),
        High=('High', 'max'),
        Low=('Low', 'min'),
        Close=('Close', 'last'),
        Volume=('Volume', 'sum'),
        n_bars=('Close', 'count'),
    )
    daily.index = pd.DatetimeIndex(daily.index)
    return daily


def make_intraday_sessions(df_1m):
    """
    For each trading day, compute:
    - entry_price: Close at 6:30 AM PST (end of entry window)
    - morning_high: High between 6:30-10:00 AM PST
    - morning_return: (morning_high / entry_price - 1)
    - close_price: Close at 1:00 PM PST (or last bar)
    - close_return: (close_price / entry_price - 1)
    - max_drawdown: worst (Low / entry_price - 1) during the day
    - open_to_high: (day_high / day_open - 1)
    - reversed: 1 if morning was positive but close < entry
    """
    df = df_1m.copy()
    df['pst_time'] = df.index.tz_convert(PST)
    df['date'] = df['pst_time'].dt.date
    df['mins'] = df['pst_time'].dt.hour * 60 + df['pst_time'].dt.minute

    rows = []
    for date, day_df in df.groupby('date'):
        # Entry price: close of bar at 6:30 AM PST (minute 390)
        entry_bars = day_df[(day_df['mins'] >= 360) & (day_df['mins'] <= 390)]
        if len(entry_bars) == 0:
            continue

        # Use bars with volume for reliability
        vol_entry = entry_bars[entry_bars['Volume'] > 0]
        if len(vol_entry) > 0:
            entry_price = vol_entry['Close'].iloc[-1]
        else:
            entry_price = entry_bars['Close'].iloc[-1]

        if entry_price <= 0:
            continue

        # Morning session: 6:30-10:00 AM PST (390-600 minutes)
        morning = day_df[(day_df['mins'] > 390) & (day_df['mins'] <= 600)]
        morning_vol = morning[morning['Volume'] > 0] if len(morning) > 0 else morning

        if len(morning_vol) > 0:
            morning_high = morning_vol['High'].max()
            morning_return = morning_high / entry_price - 1
        else:
            morning_high = entry_price
            morning_return = 0.0

        # Close: last bar before 1:00 PM PST
        close_bars = day_df[(day_df['mins'] >= 390) & (day_df['mins'] <= 780)]
        close_vol = close_bars[close_bars['Volume'] > 0] if len(close_bars) > 0 else close_bars

        if len(close_vol) > 0:
            close_price = close_vol['Close'].iloc[-1]
            close_return = close_price / entry_price - 1
        else:
            close_price = entry_price
            close_return = 0.0

        # Max drawdown from entry through close
        post_entry = day_df[(day_df['mins'] >= 390) & (day_df['mins'] <= 780)]
        post_vol = post_entry[post_entry['Volume'] > 0] if len(post_entry) > 0 else post_entry
        if len(post_vol) > 0:
            max_dd = (post_vol['Low'].min() / entry_price - 1)
        else:
            max_dd = 0.0

        # Open-to-high (full day from first bar)
        all_vol = day_df[day_df['Volume'] > 0]
        if len(all_vol) >= 2:
            day_open = all_vol['Open'].iloc[0]
            day_high = all_vol['High'].max()
            open_to_high = day_high / day_open - 1 if day_open > 0 else 0
        else:
            open_to_high = 0

        # Reversal detection: morning was positive but close is below entry
        reversed_flag = 1 if (morning_return > 0.005 and close_return < 0) else 0

        # "Good day" proxy: entry-to-high > 2% AND close > entry
        good_day = 1 if (morning_return > 0.02 and close_return > 0) else 0
        # "Bad day" proxy: max DD from entry > 3%
        bad_day = 1 if max_dd < -0.03 else 0

        rows.append({
            'date': date,
            'entry_price': entry_price,
            'morning_high': morning_high,
            'morning_return': morning_return,
            'close_price': close_price,
            'close_return': close_return,
            'max_dd': max_dd,
            'open_to_high': open_to_high,
            'reversed': reversed_flag,
            'good_day': good_day,
            'bad_day': bad_day,
        })

    result = pd.DataFrame(rows)
    if len(result) > 0:
        result['date'] = pd.to_datetime(result['date'])
        result = result.set_index('date').sort_index()
    return result


# ==============================================================================
# FILTER 1: VIX RANGE FILTER
# ==============================================================================
def compute_vix_range_filter(vix_1m, lookback_days_list=[3, 5, 10]):
    """
    Compute VIX intraday range as pct of close, rolling over N days.
    Metric = mean((High-Low)/Close) over past N trading days.
    Computed using daily bars ending BEFORE 6:00 AM PST entry time.
    For VIX, use previous day's full session (it trades extended hours).
    """
    # Make daily bars from VIX (full day)
    vix_daily = make_daily_bars(vix_1m)
    if len(vix_daily) == 0:
        return {}

    vix_daily['range_pct'] = (vix_daily['High'] - vix_daily['Low']) / vix_daily['Close']
    vix_daily['range_pct'] = vix_daily['range_pct'].clip(0, 1)  # sanity

    results = {}
    for n in lookback_days_list:
        col = f'vix_range_{n}d'
        # Rolling mean of past N days (shift 1 to avoid look-ahead: today's VIX range
        # isn't known at 6 AM, so use yesterday's and before)
        vix_daily[col] = vix_daily['range_pct'].rolling(n).mean().shift(1)
        results[col] = vix_daily[['range_pct', col]].dropna()

    return vix_daily, results


# ==============================================================================
# FILTER 2: SPY DRIFT DIRECTION FILTER
# ==============================================================================
def compute_spy_drift_filter(spy_1m):
    """
    Compute SPY 5-day cumulative return using yesterday's close.
    At entry time (6:00 AM PST), we know yesterday's close.
    """
    spy_daily = make_daily_bars(spy_1m)
    if len(spy_daily) == 0:
        return None

    # 5-day return as of yesterday's close (shift 1)
    spy_daily['spy_5d_return'] = spy_daily['Close'].pct_change(5).shift(1)
    spy_daily['spy_3d_return'] = spy_daily['Close'].pct_change(3).shift(1)
    spy_daily['spy_10d_return'] = spy_daily['Close'].pct_change(10).shift(1)

    return spy_daily


# ==============================================================================
# FILTER 3: MORNING REVERSAL DETECTOR
# ==============================================================================
def compute_reversal_rate(sessions_df, lookback=5):
    """
    Over the past N days, what fraction had morning gains (6:30-10 AM) reverse by close?
    Uses the 'reversed' column from make_intraday_sessions.
    Shift by 1 to avoid look-ahead.
    """
    if len(sessions_df) == 0:
        return sessions_df
    sessions_df = sessions_df.copy()
    sessions_df['reversal_rate_5d'] = sessions_df['reversed'].rolling(lookback).mean().shift(1)
    sessions_df['reversal_rate_10d'] = sessions_df['reversed'].rolling(10).mean().shift(1)
    return sessions_df


# ==============================================================================
# FILTER 4: DRAWDOWN EXPANSION FILTER
# ==============================================================================
def compute_dd_expansion(sessions_df, lookback=5):
    """
    Average max drawdown from entry over past N trading days.
    max_dd is negative, so avg_dd is also negative.
    """
    if len(sessions_df) == 0:
        return sessions_df
    sessions_df = sessions_df.copy()
    sessions_df['avg_dd_5d'] = sessions_df['max_dd'].rolling(lookback).mean().shift(1)
    sessions_df['avg_dd_10d'] = sessions_df['max_dd'].rolling(10).mean().shift(1)
    return sessions_df


# ==============================================================================
# BACKTEST ENGINE
# ==============================================================================
def evaluate_filter(daily_outcomes, filter_values, filter_col, threshold,
                    direction='below', label=''):
    """
    Evaluate a binary filter.
    direction='below': allow trade when filter_value < threshold (e.g., low VIX range = good)
    direction='above': allow trade when filter_value > threshold (e.g., positive SPY drift = good)

    Returns dict with stats per month and overall.
    """
    merged = daily_outcomes.join(filter_values[[filter_col]], how='inner').dropna(subset=[filter_col])
    if len(merged) == 0:
        return None

    if direction == 'below':
        merged['filter_on'] = (merged[filter_col] < threshold).astype(int)
    else:
        merged['filter_on'] = (merged[filter_col] > threshold).astype(int)

    merged['month'] = merged.index.to_period('M')

    results = {}
    for month, grp in merged.groupby('month'):
        on = grp[grp['filter_on'] == 1]
        off = grp[grp['filter_on'] == 0]
        results[str(month)] = {
            'days_on': len(on),
            'days_off': len(off),
            'total_days': len(grp),
            'avg_return_on': on['close_return'].mean() if len(on) > 0 else np.nan,
            'avg_return_off': off['close_return'].mean() if len(off) > 0 else np.nan,
            'avg_return_all': grp['close_return'].mean(),
            'good_days_on': on['good_day'].sum() if len(on) > 0 else 0,
            'good_days_off': off['good_day'].sum() if len(off) > 0 else 0,
            'bad_days_on': on['bad_day'].sum() if len(on) > 0 else 0,
            'bad_days_off': off['bad_day'].sum() if len(off) > 0 else 0,
            'avg_filter_val': grp[filter_col].mean(),
        }

    # Overall
    on = merged[merged['filter_on'] == 1]
    off = merged[merged['filter_on'] == 0]

    # True positive: good days that filter allows
    total_good = merged['good_day'].sum()
    total_bad = merged['bad_day'].sum()
    tp = on['good_day'].sum() if len(on) > 0 else 0
    tn = off['bad_day'].sum() if len(off) > 0 else 0
    fp = on['bad_day'].sum() if len(on) > 0 else 0
    fn = off['good_day'].sum() if len(off) > 0 else 0

    results['overall'] = {
        'days_on': len(on),
        'days_off': len(off),
        'total_days': len(merged),
        'pct_blocked': len(off) / len(merged) if len(merged) > 0 else 0,
        'avg_return_on': on['close_return'].mean() if len(on) > 0 else np.nan,
        'avg_return_off': off['close_return'].mean() if len(off) > 0 else np.nan,
        'avg_return_all': merged['close_return'].mean(),
        'net_improvement': (on['close_return'].mean() - merged['close_return'].mean()) if len(on) > 0 else 0,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'tp_rate': tp / total_good if total_good > 0 else 0,
        'tn_rate': tn / total_bad if total_bad > 0 else 0,
        'label': label,
        'threshold': threshold,
        'filter_col': filter_col,
        'direction': direction,
    }
    results['_merged'] = merged  # keep for combined filter analysis

    return results


def print_filter_table(results, filter_name):
    """Print monthly breakdown table for a filter."""
    months = sorted([k for k in results.keys() if k not in ('overall', '_merged')])
    overall = results['overall']

    print(f"\n  {'Month':<10} {'Days ON':>8} {'Days OFF':>8} {'Avg Ret (ON)':>14} {'Avg Ret (OFF)':>14} {'Filter Avg':>12}")
    print(f"  {'─'*10} {'─'*8} {'─'*8} {'─'*14} {'─'*14} {'─'*12}")

    for m in months:
        r = results[m]
        on_str = f"{r['avg_return_on']:+.3%}" if not np.isnan(r['avg_return_on']) else "N/A"
        off_str = f"{r['avg_return_off']:+.3%}" if not np.isnan(r['avg_return_off']) else "N/A"
        print(f"  {m:<10} {r['days_on']:>8} {r['days_off']:>8} {on_str:>14} {off_str:>14} {r['avg_filter_val']:>12.4f}")

    print(f"  {'─'*10} {'─'*8} {'─'*8} {'─'*14} {'─'*14} {'─'*12}")
    on_str = f"{overall['avg_return_on']:+.3%}" if not np.isnan(overall['avg_return_on']) else "N/A"
    off_str = f"{overall['avg_return_off']:+.3%}" if not np.isnan(overall['avg_return_off']) else "N/A"
    print(f"  {'TOTAL':<10} {overall['days_on']:>8} {overall['days_off']:>8} {on_str:>14} {off_str:>14}")

    print(f"\n  Filter Quality Metrics:")
    print(f"    True Positive Rate (good days allowed): {overall['tp_rate']:.1%}")
    print(f"    True Negative Rate (bad days blocked):  {overall['tn_rate']:.1%}")
    print(f"    Net PnL Improvement (ON vs ALL):        {overall['net_improvement']:+.3%}")
    print(f"    Trade Reduction (days blocked):         {overall['pct_blocked']:.1%}")
    print(f"    TP={overall['tp']}  FN={overall['fn']}  FP={overall['fp']}  TN={overall['tn']}")


# ==============================================================================
# MAIN ANALYSIS
# ==============================================================================
print(f"{'='*W}")
print(f"  REGIME FILTER DESIGN & BACKTEST")
print(f"  Analysis Period: April 1 - June 30, 2026")
print(f"  Entry Time: 6:00-6:30 AM PST")
print(f"  Tickers: {', '.join(TICKERS)}")
print(f"{'='*W}")

# --- Step 1: Load Data ---
print(f"\n--- Loading regime data ---")
regime_data = load_regime_data()

print(f"\n--- Loading ticker data ---")
ticker_data = load_ticker_data()

# --- Step 2: Compute ticker sessions ---
print(f"\n--- Computing intraday sessions for each ticker ---")
ticker_sessions = {}
for ticker in TICKERS:
    if ticker not in ticker_data:
        continue
    sessions = make_intraday_sessions(ticker_data[ticker])
    # Filter to backtest window
    sessions = sessions[(sessions.index >= BACKTEST_START.tz_localize(None)) &
                        (sessions.index <= BACKTEST_END.tz_localize(None))]
    if len(sessions) > 0:
        ticker_sessions[ticker] = sessions
        good = sessions['good_day'].sum()
        bad = sessions['bad_day'].sum()
        print(f"  {ticker}: {len(sessions)} days, {good} good, {bad} bad, "
              f"avg close_return={sessions['close_return'].mean():+.3%}, "
              f"avg max_dd={sessions['max_dd'].mean():.3%}")

# --- Aggregate all ticker-days into one panel ---
all_sessions = []
for ticker, df in ticker_sessions.items():
    tmp = df.copy()
    tmp['ticker'] = ticker
    all_sessions.append(tmp)
all_sessions_df = pd.concat(all_sessions)
print(f"\n  Total ticker-days: {len(all_sessions_df)}")
print(f"  Good days: {all_sessions_df['good_day'].sum()}")
print(f"  Bad days: {all_sessions_df['bad_day'].sum()}")
print(f"  Avg close return: {all_sessions_df['close_return'].mean():+.4%}")

# Also make a "portfolio day" level: average across tickers per day
port_daily = all_sessions_df.groupby(all_sessions_df.index).agg(
    close_return=('close_return', 'mean'),
    morning_return=('morning_return', 'mean'),
    max_dd=('max_dd', 'mean'),
    reversed=('reversed', 'mean'),
    good_day=('good_day', 'max'),    # any ticker had a good day
    bad_day=('bad_day', 'max'),      # any ticker had a bad day
    n_tickers=('ticker', 'count'),
)
print(f"  Portfolio days: {len(port_daily)}")

# ============================================================================
# FILTER 1: VIX RANGE FILTER
# ============================================================================
print(f"\n\n{'='*W}")
print(f"  FILTER 1: VIX RANGE FILTER")
print(f"  Metric: Rolling avg of VIX (High-Low)/Close over past N days")
print(f"  Hypothesis: High VIX intraday range = whipsawing, spike trades fail")
print(f"{'='*W}")

if 'VIX' in regime_data:
    vix_daily, vix_range_results = compute_vix_range_filter(regime_data['VIX'], [3, 5, 10])

    # Show the time series of VIX range
    vix_bt = vix_daily[(vix_daily.index >= BACKTEST_START.tz_localize(None)) &
                       (vix_daily.index <= BACKTEST_END.tz_localize(None))]
    print(f"\n  VIX Range Statistics (Apr-Jun 2026):")
    print(f"  {'Month':<10} {'Avg Range%':>12} {'Max Range%':>12} {'Avg 5d Roll':>12}")
    print(f"  {'─'*10} {'─'*12} {'─'*12} {'─'*12}")
    for month in ['2026-04', '2026-05', '2026-06']:
        m_mask = vix_bt.index.to_period('M').astype(str) == month
        m = vix_bt[m_mask]
        if len(m) > 0:
            print(f"  {month:<10} {m['range_pct'].mean():>12.4f} {m['range_pct'].max():>12.4f} "
                  f"{m['vix_range_5d'].mean():>12.4f}")

    # Test thresholds
    print(f"\n  Threshold scan (VIX Range 5d rolling avg):")
    print(f"  Filter ON when vix_range_5d < threshold (low range = calm = allow trade)")

    best_f1 = None
    best_f1_score = -999

    for lookback_col in ['vix_range_3d', 'vix_range_5d', 'vix_range_10d']:
        # Get percentiles to guide threshold selection
        vals = vix_bt[lookback_col].dropna()
        if len(vals) == 0:
            continue
        p25, p50, p75, p90 = np.percentile(vals, [25, 50, 75, 90])
        print(f"\n  {lookback_col} distribution: P25={p25:.4f} P50={p50:.4f} P75={p75:.4f} P90={p90:.4f}")

        for threshold in [p50, p75, p90, 0.03, 0.04, 0.05, 0.06, 0.08]:
            r = evaluate_filter(port_daily, vix_daily, lookback_col, threshold,
                               direction='below',
                               label=f"VIX_range {lookback_col} < {threshold:.4f}")
            if r is None:
                continue
            o = r['overall']
            score = o['net_improvement']

            # Print compact line
            print(f"    {lookback_col} < {threshold:.4f}: "
                  f"ON={o['days_on']:>3} OFF={o['days_off']:>3} "
                  f"ret_on={o['avg_return_on']:+.4%} ret_off={o['avg_return_off']:+.4%} "
                  f"net={o['net_improvement']:+.4%} "
                  f"TPR={o['tp_rate']:.0%} TNR={o['tn_rate']:.0%}")

            if score > best_f1_score:
                best_f1_score = score
                best_f1 = r

    if best_f1:
        print(f"\n  >>> BEST VIX Range Filter: {best_f1['overall']['label']}")
        print_filter_table(best_f1, "VIX Range (best)")
else:
    print("  VIX data not available")
    best_f1 = None


# ============================================================================
# FILTER 2: SPY DRIFT DIRECTION FILTER
# ============================================================================
print(f"\n\n{'='*W}")
print(f"  FILTER 2: SPY DRIFT DIRECTION FILTER")
print(f"  Metric: SPY N-day return as of yesterday's close")
print(f"  Hypothesis: Negative SPY drift = bearish regime, spike longs fail")
print(f"{'='*W}")

if 'SPY' in regime_data:
    spy_daily = compute_spy_drift_filter(regime_data['SPY'])

    spy_bt = spy_daily[(spy_daily.index >= BACKTEST_START.tz_localize(None)) &
                       (spy_daily.index <= BACKTEST_END.tz_localize(None))]

    print(f"\n  SPY Drift Statistics (Apr-Jun 2026):")
    print(f"  {'Month':<10} {'Avg 5d Ret':>12} {'Min 5d Ret':>12} {'Max 5d Ret':>12}")
    print(f"  {'─'*10} {'─'*12} {'─'*12} {'─'*12}")
    for month in ['2026-04', '2026-05', '2026-06']:
        m_mask = spy_bt.index.to_period('M').astype(str) == month
        m = spy_bt[m_mask]
        if len(m) > 0:
            print(f"  {month:<10} {m['spy_5d_return'].mean():>12.4%} {m['spy_5d_return'].min():>12.4%} "
                  f"{m['spy_5d_return'].max():>12.4%}")

    # Test thresholds
    print(f"\n  Threshold scan:")
    print(f"  Filter ON when spy_Nd_return > threshold (positive drift = allow trade)")

    best_f2 = None
    best_f2_score = -999

    for drift_col in ['spy_3d_return', 'spy_5d_return', 'spy_10d_return']:
        vals = spy_bt[drift_col].dropna()
        if len(vals) == 0:
            continue
        p10, p25, p50 = np.percentile(vals, [10, 25, 50])
        print(f"\n  {drift_col} distribution: P10={p10:.4f} P25={p25:.4f} P50={p50:.4f}")

        for threshold in [0.0, -0.005, -0.010, -0.015, -0.020, p10, p25]:
            r = evaluate_filter(port_daily, spy_daily, drift_col, threshold,
                               direction='above',
                               label=f"SPY_drift {drift_col} > {threshold:.4f}")
            if r is None:
                continue
            o = r['overall']
            score = o['net_improvement']

            print(f"    {drift_col} > {threshold:+.4f}: "
                  f"ON={o['days_on']:>3} OFF={o['days_off']:>3} "
                  f"ret_on={o['avg_return_on']:+.4%} ret_off={o['avg_return_off']:+.4%} "
                  f"net={o['net_improvement']:+.4%} "
                  f"TPR={o['tp_rate']:.0%} TNR={o['tn_rate']:.0%}")

            if score > best_f2_score:
                best_f2_score = score
                best_f2 = r

    if best_f2:
        print(f"\n  >>> BEST SPY Drift Filter: {best_f2['overall']['label']}")
        print_filter_table(best_f2, "SPY Drift (best)")
else:
    print("  SPY data not available")
    best_f2 = None


# ============================================================================
# FILTER 3: MORNING REVERSAL DETECTOR
# ============================================================================
print(f"\n\n{'='*W}")
print(f"  FILTER 3: MORNING REVERSAL DETECTOR")
print(f"  Metric: Fraction of past N days where morning gains (6:30-10 AM) reversed by close")
print(f"  Hypothesis: High reversal rate = continuation pattern broken, spike trades fail")
print(f"{'='*W}")

# Use SPY for market-wide reversal detection
if 'SPY' in regime_data:
    spy_sessions = make_intraday_sessions(regime_data['SPY'])
    spy_sessions_bt = spy_sessions[(spy_sessions.index >= pd.Timestamp('2026-03-15')) &
                                   (spy_sessions.index <= BACKTEST_END.tz_localize(None))]
    spy_sessions_bt = compute_reversal_rate(spy_sessions_bt, lookback=5)

    # Also compute per-ticker reversal rates and average them
    # Portfolio-level reversal rate
    port_rev = port_daily.copy()
    port_rev['reversal_rate_5d'] = port_rev['reversed'].rolling(5).mean().shift(1)
    port_rev['reversal_rate_10d'] = port_rev['reversed'].rolling(10).mean().shift(1)

    print(f"\n  SPY Reversal Statistics (Apr-Jun 2026):")
    spy_analysis = spy_sessions_bt[(spy_sessions_bt.index >= BACKTEST_START.tz_localize(None))]
    for month in ['2026-04', '2026-05', '2026-06']:
        m_mask = spy_analysis.index.to_period('M').astype(str) == month
        m = spy_analysis[m_mask]
        if len(m) > 0:
            print(f"  {month}: reversal_rate_5d avg={m['reversal_rate_5d'].mean():.2f}, "
                  f"reversal_rate_10d avg={m['reversal_rate_10d'].mean():.2f}, "
                  f"raw reversed days: {m['reversed'].sum()}/{len(m)}")

    print(f"\n  Portfolio Reversal Statistics (Apr-Jun 2026):")
    port_analysis = port_rev[(port_rev.index >= BACKTEST_START.tz_localize(None))]
    for month in ['2026-04', '2026-05', '2026-06']:
        m_mask = port_analysis.index.to_period('M').astype(str) == month
        m = port_analysis[m_mask]
        if len(m) > 0:
            print(f"  {month}: reversal_rate_5d avg={m['reversal_rate_5d'].mean():.2f}, "
                  f"raw reversed frac: {m['reversed'].mean():.2f}")

    # Test thresholds using SPY reversal rate
    print(f"\n  Threshold scan (SPY reversal rate 5d):")
    print(f"  Filter ON when reversal_rate < threshold (low reversal = continuation intact)")

    best_f3 = None
    best_f3_score = -999

    for rev_col, rev_src in [('reversal_rate_5d', spy_sessions_bt), ('reversal_rate_10d', spy_sessions_bt)]:
        for threshold in [0.2, 0.3, 0.4, 0.5, 0.6]:
            r = evaluate_filter(port_daily, rev_src, rev_col, threshold,
                               direction='below',
                               label=f"SPY_reversal {rev_col} < {threshold:.1f}")
            if r is None:
                continue
            o = r['overall']
            score = o['net_improvement']

            print(f"    SPY {rev_col} < {threshold:.1f}: "
                  f"ON={o['days_on']:>3} OFF={o['days_off']:>3} "
                  f"ret_on={o['avg_return_on']:+.4%} ret_off={o['avg_return_off']:+.4%} "
                  f"net={o['net_improvement']:+.4%} "
                  f"TPR={o['tp_rate']:.0%} TNR={o['tn_rate']:.0%}")

            if score > best_f3_score:
                best_f3_score = score
                best_f3 = r

    # Also test portfolio-level reversal rate
    for rev_col in ['reversal_rate_5d', 'reversal_rate_10d']:
        for threshold in [0.2, 0.3, 0.4, 0.5, 0.6]:
            r = evaluate_filter(port_daily, port_rev, rev_col, threshold,
                               direction='below',
                               label=f"Portfolio_reversal {rev_col} < {threshold:.1f}")
            if r is None:
                continue
            o = r['overall']
            score = o['net_improvement']

            print(f"    PORT {rev_col} < {threshold:.1f}: "
                  f"ON={o['days_on']:>3} OFF={o['days_off']:>3} "
                  f"ret_on={o['avg_return_on']:+.4%} ret_off={o['avg_return_off']:+.4%} "
                  f"net={o['net_improvement']:+.4%} "
                  f"TPR={o['tp_rate']:.0%} TNR={o['tn_rate']:.0%}")

            if score > best_f3_score:
                best_f3_score = score
                best_f3 = r

    if best_f3:
        print(f"\n  >>> BEST Reversal Filter: {best_f3['overall']['label']}")
        print_filter_table(best_f3, "Reversal (best)")
else:
    print("  SPY data not available")
    best_f3 = None


# ============================================================================
# FILTER 4: DRAWDOWN EXPANSION FILTER
# ============================================================================
print(f"\n\n{'='*W}")
print(f"  FILTER 4: DRAWDOWN EXPANSION FILTER")
print(f"  Metric: Average max drawdown from entry over past 5 trading days (per-ticker)")
print(f"  Hypothesis: Elevated recent drawdowns = SLs getting hit, risk-off environment")
print(f"{'='*W}")

# Compute per-ticker then average to portfolio level
port_dd = port_daily.copy()
port_dd = compute_dd_expansion(port_dd, lookback=5)

print(f"\n  Portfolio Avg Max DD (Apr-Jun 2026):")
for month in ['2026-04', '2026-05', '2026-06']:
    m_mask = port_dd.index.to_period('M').astype(str) == month
    m = port_dd[m_mask]
    if len(m) > 0:
        print(f"  {month}: avg_dd_5d mean={m['avg_dd_5d'].mean():.4%}, "
              f"today avg max_dd={m['max_dd'].mean():.4%}")

# Test thresholds
print(f"\n  Threshold scan:")
print(f"  Filter ON when avg_dd_Nd > threshold (less negative = smaller drawdowns = allow)")

best_f4 = None
best_f4_score = -999

for dd_col in ['avg_dd_5d', 'avg_dd_10d']:
    vals = port_dd[dd_col].dropna()
    if len(vals) == 0:
        continue
    p10, p25, p50, p75 = np.percentile(vals, [10, 25, 50, 75])
    print(f"\n  {dd_col} distribution: P10={p10:.4%} P25={p25:.4%} P50={p50:.4%} P75={p75:.4%}")

    for threshold in [-0.01, -0.015, -0.02, -0.025, -0.03, -0.035, -0.04, p25, p50]:
        r = evaluate_filter(port_daily, port_dd, dd_col, threshold,
                           direction='above',
                           label=f"DD_expansion {dd_col} > {threshold:.4f}")
        if r is None:
            continue
        o = r['overall']
        score = o['net_improvement']

        print(f"    {dd_col} > {threshold:.4f}: "
              f"ON={o['days_on']:>3} OFF={o['days_off']:>3} "
              f"ret_on={o['avg_return_on']:+.4%} ret_off={o['avg_return_off']:+.4%} "
              f"net={o['net_improvement']:+.4%} "
              f"TPR={o['tp_rate']:.0%} TNR={o['tn_rate']:.0%}")

        if score > best_f4_score:
            best_f4_score = score
            best_f4 = r

if best_f4:
    print(f"\n  >>> BEST DD Expansion Filter: {best_f4['overall']['label']}")
    print_filter_table(best_f4, "DD Expansion (best)")


# ============================================================================
# FILTER 5: COMBINED REGIME SCORE
# ============================================================================
print(f"\n\n{'='*W}")
print(f"  FILTER 5: COMBINED REGIME SCORE")
print(f"  Combining best individual filters: majority vote")
print(f"{'='*W}")

# Collect best filter merged datasets
best_filters = {
    'F1_vix_range': best_f1,
    'F2_spy_drift': best_f2,
    'F3_reversal': best_f3,
    'F4_dd_expand': best_f4,
}

# Build combined daily filter signals
active_filters = {k: v for k, v in best_filters.items() if v is not None}
print(f"\n  Active filters: {len(active_filters)}")
for name, r in active_filters.items():
    o = r['overall']
    print(f"    {name}: {o['label']} (net improvement: {o['net_improvement']:+.4%})")

if len(active_filters) >= 2:
    # Merge filter ON/OFF signals on the same date index
    combined = port_daily[['close_return', 'good_day', 'bad_day']].copy()

    for name, r in active_filters.items():
        merged = r['_merged']
        combined[f'{name}_on'] = merged['filter_on']

    combined = combined.dropna()

    # Composite: count how many filters are ON
    filter_on_cols = [f'{name}_on' for name in active_filters.keys()]
    combined['n_filters_on'] = combined[filter_on_cols].sum(axis=1)
    combined['month'] = combined.index.to_period('M')

    n_active = len(active_filters)

    print(f"\n  Combined filter distribution:")
    for n_on in range(n_active + 1):
        cnt = (combined['n_filters_on'] == n_on).sum()
        if cnt > 0:
            avg_ret = combined[combined['n_filters_on'] == n_on]['close_return'].mean()
            print(f"    {n_on}/{n_active} filters ON: {cnt} days, avg return = {avg_ret:+.4%}")

    # Test composite thresholds (require N out of M filters to pass)
    print(f"\n  Composite threshold scan:")
    print(f"  Filter ON when n_filters_on >= threshold")

    best_f5 = None
    best_f5_score = -999

    for min_on in range(1, n_active + 1):
        # Majority and supermajority
        combined['composite_on'] = (combined['n_filters_on'] >= min_on).astype(int)

        on = combined[combined['composite_on'] == 1]
        off = combined[combined['composite_on'] == 0]

        if len(on) == 0 or len(off) == 0:
            continue

        total_good = combined['good_day'].sum()
        total_bad = combined['bad_day'].sum()
        tp = on['good_day'].sum()
        tn = off['bad_day'].sum()
        fp = on['bad_day'].sum()
        fn = off['good_day'].sum()

        net_imp = on['close_return'].mean() - combined['close_return'].mean()

        print(f"    >= {min_on}/{n_active}: "
              f"ON={len(on):>3} OFF={len(off):>3} "
              f"ret_on={on['close_return'].mean():+.4%} ret_off={off['close_return'].mean():+.4%} "
              f"net={net_imp:+.4%} "
              f"TPR={tp/total_good:.0%} TNR={tn/total_bad:.0%}")

        # Monthly breakdown
        print(f"      Monthly: ", end='')
        for month, grp in combined.groupby('month'):
            m_on = grp[grp['composite_on'] == 1]
            m_off = grp[grp['composite_on'] == 0]
            m_on_ret = m_on['close_return'].mean() if len(m_on) > 0 else np.nan
            m_off_ret = m_off['close_return'].mean() if len(m_off) > 0 else np.nan
            print(f"{month}: ON {len(m_on)} ({m_on_ret:+.3%}) OFF {len(m_off)} ({m_off_ret:+.3%})  ", end='')
        print()

        if net_imp > best_f5_score:
            best_f5_score = net_imp
            best_f5 = {
                'threshold': min_on,
                'n_filters': n_active,
                'net_improvement': net_imp,
                'days_on': len(on),
                'days_off': len(off),
                'avg_return_on': on['close_return'].mean(),
                'avg_return_off': off['close_return'].mean(),
                'tp_rate': tp / total_good if total_good > 0 else 0,
                'tn_rate': tn / total_bad if total_bad > 0 else 0,
                'pct_blocked': len(off) / len(combined),
                'combined_df': combined,
            }

    if best_f5:
        print(f"\n  >>> BEST Combined: >= {best_f5['threshold']}/{best_f5['n_filters']} filters ON")
        print(f"      Net improvement: {best_f5['net_improvement']:+.4%}")
        print(f"      TP rate: {best_f5['tp_rate']:.1%}, TN rate: {best_f5['tn_rate']:.1%}")
        print(f"      Days blocked: {best_f5['pct_blocked']:.1%}")


# ============================================================================
# FINAL RANKING
# ============================================================================
print(f"\n\n{'='*W}")
print(f"  FINAL RANKING: ALL FILTERS")
print(f"{'='*W}")

ranking = []
for name, r in active_filters.items():
    o = r['overall']
    ranking.append({
        'filter': name,
        'label': o['label'],
        'net_improvement': o['net_improvement'],
        'tp_rate': o['tp_rate'],
        'tn_rate': o['tn_rate'],
        'pct_blocked': o['pct_blocked'],
        'avg_return_on': o['avg_return_on'],
        'avg_return_off': o['avg_return_off'],
        'threshold': o['threshold'],
    })

if best_f5:
    ranking.append({
        'filter': 'F5_combined',
        'label': f"Combined >= {best_f5['threshold']}/{best_f5['n_filters']}",
        'net_improvement': best_f5['net_improvement'],
        'tp_rate': best_f5['tp_rate'],
        'tn_rate': best_f5['tn_rate'],
        'pct_blocked': best_f5['pct_blocked'],
        'avg_return_on': best_f5['avg_return_on'],
        'avg_return_off': best_f5['avg_return_off'],
        'threshold': f">={best_f5['threshold']}/{best_f5['n_filters']}",
    })

ranking.sort(key=lambda x: x['net_improvement'], reverse=True)

print(f"\n  {'Rank':<5} {'Filter':<16} {'Net Improve':>12} {'Ret(ON)':>10} {'Ret(OFF)':>10} "
      f"{'TPR':>6} {'TNR':>6} {'Blocked':>8} {'Threshold'}")
print(f"  {'─'*5} {'─'*16} {'─'*12} {'─'*10} {'─'*10} {'─'*6} {'─'*6} {'─'*8} {'─'*25}")

for i, r in enumerate(ranking, 1):
    thr_str = str(r['threshold']) if isinstance(r['threshold'], str) else f"{r['threshold']:.4f}"
    print(f"  {i:<5} {r['filter']:<16} {r['net_improvement']:>+11.4%} "
          f"{r['avg_return_on']:>+9.4%} {r['avg_return_off']:>+9.4%} "
          f"{r['tp_rate']:>5.0%} {r['tn_rate']:>5.0%} {r['pct_blocked']:>7.0%} "
          f"{thr_str}")


# ============================================================================
# RECOMMENDATIONS
# ============================================================================
print(f"\n\n{'='*W}")
print(f"  RECOMMENDATIONS")
print(f"{'='*W}")

if len(ranking) >= 2:
    top2 = ranking[:2]
    print(f"\n  Top 2 Individual Filters for Implementation:")
    for i, r in enumerate(top2, 1):
        print(f"\n  {i}. {r['filter']}: {r['label']}")
        print(f"     Net PnL improvement: {r['net_improvement']:+.4%}")
        print(f"     True Positive Rate:  {r['tp_rate']:.1%} (good days preserved)")
        print(f"     True Negative Rate:  {r['tn_rate']:.1%} (bad days blocked)")
        print(f"     Trade reduction:     {r['pct_blocked']:.1%}")
        print(f"     Avg return ON days:  {r['avg_return_on']:+.4%}")
        print(f"     Avg return OFF days: {r['avg_return_off']:+.4%}")

if best_f5:
    print(f"\n  Combined Regime Score:")
    print(f"     Rule: Require >= {best_f5['threshold']}/{best_f5['n_filters']} individual filters to pass")
    print(f"     Net PnL improvement: {best_f5['net_improvement']:+.4%}")
    print(f"     TP rate: {best_f5['tp_rate']:.1%}, TN rate: {best_f5['tn_rate']:.1%}")
    print(f"     Trade reduction: {best_f5['pct_blocked']:.1%}")


# ============================================================================
# DAY-BY-DAY JUNE DETAIL
# ============================================================================
print(f"\n\n{'='*W}")
print(f"  DAY-BY-DAY: JUNE 2026 FILTER SIGNALS")
print(f"{'='*W}")

if best_f5 and 'combined_df' in best_f5:
    june = best_f5['combined_df']
    june = june[june.index.to_period('M').astype(str) == '2026-06']

    if len(june) > 0:
        print(f"\n  {'Date':<12}", end='')
        for name in active_filters.keys():
            print(f" {name[-8:]:>10}", end='')
        print(f" {'N_ON':>5} {'Action':>8} {'CloseRet':>10} {'Good':>5} {'Bad':>5}")
        print(f"  {'─'*12}", end='')
        for _ in active_filters:
            print(f" {'─'*10}", end='')
        print(f" {'─'*5} {'─'*8} {'─'*10} {'─'*5} {'─'*5}")

        for date, row in june.iterrows():
            print(f"  {str(date.date()):<12}", end='')
            for name in active_filters.keys():
                val = row.get(f'{name}_on', np.nan)
                signal = 'ON' if val == 1 else 'OFF'
                color = signal
                print(f" {color:>10}", end='')
            n_on = row['n_filters_on']
            action = 'TRADE' if n_on >= best_f5['threshold'] else 'BLOCK'
            print(f" {int(n_on):>5} {action:>8} {row['close_return']:>+9.4%} "
                  f"{'Y' if row['good_day'] else ' ':>5} {'Y' if row['bad_day'] else ' ':>5}")


print(f"\n\n{'='*W}")
print(f"  ANALYSIS COMPLETE")
print(f"{'='*W}")
