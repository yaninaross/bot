"""
design_regime_filters_v2.py -- Practical Regime Filter Design
=============================================================
Builds on v1 findings. Focuses on implementable filters with
reasonable trade-through rates (blocking 20-50% of days, not 90%+).

Key insight from v1:
- VIX range 10d < 0.05 was best by net improvement but blocks 98% of days
- SPY drift and reversal rate block 25-44% which is more practical
- The REAL value is in the intermediate thresholds

This version:
1. Focuses on the practical threshold range for each filter
2. Shows per-ticker breakdown (not just portfolio average)
3. Adds a new filter: SPY intraday range (ATR proxy)
4. Tests combinations with 2-of-3 majority vote
"""

import numpy as np
import pandas as pd
import pytz
import warnings

warnings.filterwarnings('ignore')

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass

# ==============================================================================
# CONFIG
# ==============================================================================
BUCKET = "time-series-bucket-ys"
OHLCV_DIR = "Data/Inference"
INFERENCE_V2_DIR = "Data/Inference_v2"
PST = pytz.timezone('America/Los_Angeles')

TICKERS = ['KEEL', 'CIFR', 'CLSK', 'HUT', 'RIOT', 'RCAT', 'FLNC', 'PL', 'APLD', 'WULF']

REGIME_FILES = {
    'VIX': f'gs://{BUCKET}/{OHLCV_DIR}/VIX_1m_full_historical_data.parquet',
    'SPY': f'gs://{BUCKET}/{OHLCV_DIR}/SPY_1m_full_historical_data.parquet',
}

ANALYSIS_START = pd.Timestamp('2026-03-01', tz='UTC')
BACKTEST_START = pd.Timestamp('2026-04-01', tz='UTC')
BACKTEST_END   = pd.Timestamp('2026-06-30 23:59:59', tz='UTC')

W = 120

# ==============================================================================
# DATA LOADING (same as v1)
# ==============================================================================
def load_ohlcv(uri):
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

def make_daily_bars(df_1m):
    df = df_1m.copy()
    df['pst_time'] = df.index.tz_convert(PST)
    df['date'] = df['pst_time'].dt.date
    daily = df.groupby('date').agg(
        Open=('Open', 'first'),
        High=('High', 'max'),
        Low=('Low', 'min'),
        Close=('Close', 'last'),
        Volume=('Volume', 'sum'),
    )
    daily.index = pd.DatetimeIndex(daily.index)
    return daily

def make_sessions(df_1m):
    """Extract daily session stats anchored to entry time."""
    df = df_1m.copy()
    df['pst_time'] = df.index.tz_convert(PST)
    df['date'] = df['pst_time'].dt.date
    df['mins'] = df['pst_time'].dt.hour * 60 + df['pst_time'].dt.minute

    rows = []
    for date, day_df in df.groupby('date'):
        entry_bars = day_df[(day_df['mins'] >= 360) & (day_df['mins'] <= 390)]
        if len(entry_bars) == 0:
            continue
        vol_entry = entry_bars[entry_bars['Volume'] > 0]
        entry_price = vol_entry['Close'].iloc[-1] if len(vol_entry) > 0 else entry_bars['Close'].iloc[-1]
        if entry_price <= 0:
            continue

        # Morning: 6:30-10:00 AM
        morning = day_df[(day_df['mins'] > 390) & (day_df['mins'] <= 600) & (day_df['Volume'] > 0)]
        morning_high = morning['High'].max() if len(morning) > 0 else entry_price
        morning_return = morning_high / entry_price - 1

        # Close: by 1 PM PST
        close_bars = day_df[(day_df['mins'] >= 390) & (day_df['mins'] <= 780) & (day_df['Volume'] > 0)]
        close_price = close_bars['Close'].iloc[-1] if len(close_bars) > 0 else entry_price
        close_return = close_price / entry_price - 1

        # Max drawdown from entry
        post = day_df[(day_df['mins'] >= 390) & (day_df['mins'] <= 780) & (day_df['Volume'] > 0)]
        max_dd = (post['Low'].min() / entry_price - 1) if len(post) > 0 else 0

        reversed_flag = 1 if (morning_return > 0.005 and close_return < 0) else 0
        good_day = 1 if (morning_return > 0.02 and close_return > 0) else 0
        bad_day = 1 if max_dd < -0.03 else 0

        rows.append({
            'date': date, 'entry_price': entry_price,
            'morning_return': morning_return, 'close_return': close_return,
            'max_dd': max_dd, 'reversed': reversed_flag,
            'good_day': good_day, 'bad_day': bad_day,
        })

    result = pd.DataFrame(rows)
    if len(result) > 0:
        result['date'] = pd.to_datetime(result['date'])
        result = result.set_index('date').sort_index()
    return result


# ==============================================================================
# LOAD DATA
# ==============================================================================
print(f"{'='*W}")
print(f"  REGIME FILTER DESIGN v2 -- PRACTICAL THRESHOLDS")
print(f"  April 1 - June 30, 2026")
print(f"{'='*W}")

print(f"\n--- Loading data ---")
regime = {}
for ticker, uri in REGIME_FILES.items():
    print(f"  {ticker}...", end=' ')
    df = load_ohlcv(uri)
    regime[ticker] = df[df.index >= ANALYSIS_START]
    print(f"OK ({len(regime[ticker]):,} bars)")

ticker_data = {}
for ticker in TICKERS:
    for uri in [f'gs://{BUCKET}/{OHLCV_DIR}/{ticker}_1m_full_historical_data.parquet',
                f'gs://{BUCKET}/{INFERENCE_V2_DIR}/{ticker}_master_inference.parquet']:
        try:
            df = load_ohlcv(uri)
            df = df[df.index >= ANALYSIS_START]
            if len(df) > 0:
                ticker_data[ticker] = df
                break
        except:
            continue

print(f"  Loaded {len(ticker_data)} tickers")

# Compute sessions
print(f"\n--- Computing sessions ---")
sessions = {}
for ticker in TICKERS:
    if ticker not in ticker_data:
        continue
    s = make_sessions(ticker_data[ticker])
    s = s[(s.index >= BACKTEST_START.tz_localize(None)) & (s.index <= BACKTEST_END.tz_localize(None))]
    sessions[ticker] = s

# Portfolio daily (average across tickers)
all_sess = []
for ticker, df in sessions.items():
    tmp = df.copy(); tmp['ticker'] = ticker
    all_sess.append(tmp)
all_sess_df = pd.concat(all_sess)

port = all_sess_df.groupby(all_sess_df.index).agg(
    close_return=('close_return', 'mean'),
    morning_return=('morning_return', 'mean'),
    max_dd=('max_dd', 'mean'),
    reversed=('reversed', 'mean'),
    good_day=('good_day', 'max'),
    bad_day=('bad_day', 'max'),
)

# Month labels
port['month'] = port.index.to_period('M').astype(str)

print(f"  Portfolio: {len(port)} days")
for m in ['2026-04', '2026-05', '2026-06']:
    pm = port[port['month'] == m]
    print(f"  {m}: {len(pm)} days, avg return {pm['close_return'].mean():+.3%}, "
          f"good {pm['good_day'].sum():.0f}, bad {pm['bad_day'].sum():.0f}")


# ==============================================================================
# COMPUTE ALL FILTER METRICS
# ==============================================================================
print(f"\n{'='*W}")
print(f"  COMPUTING FILTER METRICS")
print(f"{'='*W}")

# --- VIX Daily ---
vix_daily = make_daily_bars(regime['VIX'])
vix_daily['range_pct'] = (vix_daily['High'] - vix_daily['Low']) / vix_daily['Close']
for n in [3, 5, 10]:
    vix_daily[f'vix_range_{n}d'] = vix_daily['range_pct'].rolling(n).mean().shift(1)

# --- SPY Daily ---
spy_daily = make_daily_bars(regime['SPY'])
spy_daily['spy_range_pct'] = (spy_daily['High'] - spy_daily['Low']) / spy_daily['Close']
for n in [3, 5, 10]:
    spy_daily[f'spy_{n}d_return'] = spy_daily['Close'].pct_change(n).shift(1)
    spy_daily[f'spy_range_{n}d'] = spy_daily['spy_range_pct'].rolling(n).mean().shift(1)

# --- Portfolio Drawdown ---
port['avg_dd_5d'] = port['max_dd'].rolling(5).mean().shift(1)
port['avg_dd_10d'] = port['max_dd'].rolling(10).mean().shift(1)

# --- Portfolio Reversal Rate ---
port['reversal_5d'] = port['reversed'].rolling(5).mean().shift(1)
port['reversal_10d'] = port['reversed'].rolling(10).mean().shift(1)

# --- SPY Morning Reversal ---
spy_sess = make_sessions(regime['SPY'])
spy_sess['spy_reversal_5d'] = spy_sess['reversed'].rolling(5).mean().shift(1)

# --- Merge everything into one frame ---
filters = port.copy()
# Add VIX metrics (join on date)
for col in ['vix_range_3d', 'vix_range_5d', 'vix_range_10d', 'range_pct']:
    filters = filters.join(vix_daily[[col]].rename(columns={col: col if col != 'range_pct' else 'vix_today_range'}), how='left')
# Add SPY metrics
for col in ['spy_5d_return', 'spy_10d_return', 'spy_3d_return',
            'spy_range_3d', 'spy_range_5d', 'spy_range_10d']:
    filters = filters.join(spy_daily[[col]], how='left')
# Add SPY reversal
filters = filters.join(spy_sess[['spy_reversal_5d']], how='left')

filters = filters.dropna(subset=['vix_range_5d', 'spy_5d_return'])

print(f"  Merged filter frame: {len(filters)} days")
print(f"  Columns: {list(filters.columns)}")


# ==============================================================================
# SYSTEMATIC FILTER EVALUATION
# ==============================================================================
def eval_filter(df, col, threshold, direction):
    """Evaluate filter and return stats dict."""
    if direction == 'below':
        mask = df[col] < threshold
    else:
        mask = df[col] > threshold
    on = df[mask]
    off = df[~mask]
    if len(on) == 0 or len(off) == 0:
        return None

    # Per-month breakdown
    months = {}
    for m in ['2026-04', '2026-05', '2026-06']:
        m_df = df[df['month'] == m]
        m_on = m_df[mask.loc[m_df.index] if m_df.index.isin(mask.index).all() else mask.reindex(m_df.index, fill_value=False)]
        m_off = m_df[~mask.loc[m_df.index] if m_df.index.isin(mask.index).all() else ~mask.reindex(m_df.index, fill_value=False)]
        months[m] = {
            'on': len(m_on), 'off': len(m_off),
            'ret_on': m_on['close_return'].mean() if len(m_on) > 0 else np.nan,
            'ret_off': m_off['close_return'].mean() if len(m_off) > 0 else np.nan,
        }

    tg = df['good_day'].sum(); tb = df['bad_day'].sum()
    tp = on['good_day'].sum(); tn = off['bad_day'].sum()

    # June-specific: what fraction of June is blocked?
    jun = df[df['month'] == '2026-06']
    jun_blocked = (~mask).reindex(jun.index, fill_value=False).sum() if len(jun) > 0 else 0
    jun_total = len(jun)

    # April selectivity: what fraction of April is kept?
    apr = df[df['month'] == '2026-04']
    apr_kept = mask.reindex(apr.index, fill_value=False).sum() if len(apr) > 0 else 0
    apr_total = len(apr)

    return {
        'col': col, 'threshold': threshold, 'direction': direction,
        'days_on': len(on), 'days_off': len(off), 'total': len(df),
        'pct_blocked': len(off)/len(df),
        'ret_on': on['close_return'].mean(),
        'ret_off': off['close_return'].mean(),
        'ret_all': df['close_return'].mean(),
        'net_imp': on['close_return'].mean() - df['close_return'].mean(),
        'tp_rate': tp/tg if tg > 0 else 0,
        'tn_rate': tn/tb if tb > 0 else 0,
        'months': months,
        'jun_blocked_pct': jun_blocked/jun_total if jun_total > 0 else 0,
        'apr_kept_pct': apr_kept/apr_total if apr_total > 0 else 0,
        'on_mask': mask,
    }


# ==============================================================================
# RUN ALL FILTER SCANS
# ==============================================================================
all_results = []

# --- FILTER 1: VIX Range ---
print(f"\n\n{'='*W}")
print(f"  FILTER 1: VIX INTRADAY RANGE")
print(f"{'='*W}")

for col in ['vix_range_3d', 'vix_range_5d', 'vix_range_10d']:
    vals = filters[col].dropna()
    p_vals = np.percentile(vals, [10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90])
    print(f"\n  {col} percentiles: " + " ".join([f"P{p}={v:.4f}" for p, v in zip([10,20,25,30,40,50,60,70,75,80,90], p_vals)]))

    for thr in sorted(set([0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12] + [round(p_vals[i],4) for i in [2,4,5,6,7]])):
        r = eval_filter(filters, col, thr, 'below')
        if r and 5 <= r['days_off'] <= 45:  # practical range
            all_results.append(r)
            apr = r['months'].get('2026-04', {})
            jun = r['months'].get('2026-06', {})
            print(f"    < {thr:.4f}: ON={r['days_on']:>3} OFF={r['days_off']:>3} "
                  f"ret_ON={r['ret_on']:+.3%} ret_OFF={r['ret_off']:+.3%} "
                  f"net={r['net_imp']:+.3%} "
                  f"Apr kept={r['apr_kept_pct']:.0%} Jun blocked={r['jun_blocked_pct']:.0%} "
                  f"TPR={r['tp_rate']:.0%} TNR={r['tn_rate']:.0%}")


# --- FILTER 2: SPY Drift ---
print(f"\n\n{'='*W}")
print(f"  FILTER 2: SPY MULTI-DAY DRIFT")
print(f"{'='*W}")

for col in ['spy_3d_return', 'spy_5d_return', 'spy_10d_return']:
    vals = filters[col].dropna()
    p_vals = np.percentile(vals, [10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90])
    print(f"\n  {col} percentiles: " + " ".join([f"P{p}={v:.4f}" for p, v in zip([10,20,25,30,40,50,60,70,75,80,90], p_vals)]))

    for thr in sorted(set([0.0, -0.005, -0.010, -0.015, 0.005, 0.010] + [round(p_vals[i],4) for i in [1,2,3,4]])):
        r = eval_filter(filters, col, thr, 'above')
        if r and 5 <= r['days_off'] <= 45:
            all_results.append(r)
            print(f"    > {thr:+.4f}: ON={r['days_on']:>3} OFF={r['days_off']:>3} "
                  f"ret_ON={r['ret_on']:+.3%} ret_OFF={r['ret_off']:+.3%} "
                  f"net={r['net_imp']:+.3%} "
                  f"Apr kept={r['apr_kept_pct']:.0%} Jun blocked={r['jun_blocked_pct']:.0%} "
                  f"TPR={r['tp_rate']:.0%} TNR={r['tn_rate']:.0%}")


# --- FILTER 3: Morning Reversal Rate ---
print(f"\n\n{'='*W}")
print(f"  FILTER 3: MORNING REVERSAL RATE")
print(f"{'='*W}")

for col in ['reversal_5d', 'reversal_10d']:
    vals = filters[col].dropna()
    p_vals = np.percentile(vals, [10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90])
    print(f"\n  {col} percentiles: " + " ".join([f"P{p}={v:.4f}" for p, v in zip([10,20,25,30,40,50,60,70,75,80,90], p_vals)]))

    for thr in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        r = eval_filter(filters, col, thr, 'below')
        if r and 5 <= r['days_off'] <= 45:
            all_results.append(r)
            print(f"    < {thr:.2f}: ON={r['days_on']:>3} OFF={r['days_off']:>3} "
                  f"ret_ON={r['ret_on']:+.3%} ret_OFF={r['ret_off']:+.3%} "
                  f"net={r['net_imp']:+.3%} "
                  f"Apr kept={r['apr_kept_pct']:.0%} Jun blocked={r['jun_blocked_pct']:.0%} "
                  f"TPR={r['tp_rate']:.0%} TNR={r['tn_rate']:.0%}")


# --- FILTER 4: Drawdown Expansion ---
print(f"\n\n{'='*W}")
print(f"  FILTER 4: DRAWDOWN EXPANSION")
print(f"{'='*W}")

for col in ['avg_dd_5d', 'avg_dd_10d']:
    vals = filters[col].dropna()
    p_vals = np.percentile(vals, [10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90])
    print(f"\n  {col} percentiles: " + " ".join([f"P{p}={v:.4f}" for p, v in zip([10,20,25,30,40,50,60,70,75,80,90], p_vals)]))

    for thr in [-0.03, -0.035, -0.04, -0.045, -0.05, -0.055]:
        r = eval_filter(filters, col, thr, 'above')
        if r and 5 <= r['days_off'] <= 45:
            all_results.append(r)
            print(f"    > {thr:.3f}: ON={r['days_on']:>3} OFF={r['days_off']:>3} "
                  f"ret_ON={r['ret_on']:+.3%} ret_OFF={r['ret_off']:+.3%} "
                  f"net={r['net_imp']:+.3%} "
                  f"Apr kept={r['apr_kept_pct']:.0%} Jun blocked={r['jun_blocked_pct']:.0%} "
                  f"TPR={r['tp_rate']:.0%} TNR={r['tn_rate']:.0%}")


# --- FILTER 5 (NEW): SPY Intraday Range ---
print(f"\n\n{'='*W}")
print(f"  FILTER 5 (NEW): SPY INTRADAY RANGE")
print(f"{'='*W}")

for col in ['spy_range_3d', 'spy_range_5d', 'spy_range_10d']:
    vals = filters[col].dropna()
    p_vals = np.percentile(vals, [10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90])
    print(f"\n  {col} percentiles: " + " ".join([f"P{p}={v:.4f}" for p, v in zip([10,20,25,30,40,50,60,70,75,80,90], p_vals)]))

    for thr in sorted(set([round(p_vals[i],4) for i in [2,4,5,6,7]] + [0.008, 0.010, 0.012, 0.014, 0.016])):
        r = eval_filter(filters, col, thr, 'below')
        if r and 5 <= r['days_off'] <= 45:
            all_results.append(r)
            print(f"    < {thr:.4f}: ON={r['days_on']:>3} OFF={r['days_off']:>3} "
                  f"ret_ON={r['ret_on']:+.3%} ret_OFF={r['ret_off']:+.3%} "
                  f"net={r['net_imp']:+.3%} "
                  f"Apr kept={r['apr_kept_pct']:.0%} Jun blocked={r['jun_blocked_pct']:.0%} "
                  f"TPR={r['tp_rate']:.0%} TNR={r['tn_rate']:.0%}")


# ==============================================================================
# RANK ALL FILTERS
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  RANKING: ALL PRACTICAL FILTERS (blocking 8-73%)")
print(f"{'='*W}")

# Score = net_improvement weighted by practical utility
# Bonus for: high June block rate, high April keep rate, moderate trade reduction
for r in all_results:
    # Composite score: net_improvement is key, but penalize extreme blocking
    blocked = r['pct_blocked']
    # Sweet spot: 20-50% blocked
    if 0.20 <= blocked <= 0.50:
        block_bonus = 1.0
    elif blocked < 0.20:
        block_bonus = 0.8
    else:
        block_bonus = 0.7

    r['score'] = r['net_imp'] * block_bonus + r['jun_blocked_pct'] * 0.002 + r['apr_kept_pct'] * 0.001

all_results.sort(key=lambda x: x['score'], reverse=True)

print(f"\n  {'Rank':<5} {'Filter':<22} {'Threshold':>10} {'Dir':>6} "
      f"{'ON':>4} {'OFF':>4} {'Ret(ON)':>9} {'Ret(OFF)':>9} {'Net':>9} "
      f"{'Apr%':>5} {'Jun%':>5} {'TPR':>5} {'TNR':>5}")
print(f"  {'─'*5} {'─'*22} {'─'*10} {'─'*6} {'─'*4} {'─'*4} {'─'*9} {'─'*9} {'─'*9} {'─'*5} {'─'*5} {'─'*5} {'─'*5}")

top_20 = all_results[:25]
for i, r in enumerate(top_20, 1):
    thr_str = f"{r['threshold']:.4f}" if abs(r['threshold']) < 1 else f"{r['threshold']:.3f}"
    dir_str = '<' if r['direction'] == 'below' else '>'
    print(f"  {i:<5} {r['col']:<22} {dir_str}{thr_str:>9} "
          f"{'':>6} {r['days_on']:>4} {r['days_off']:>4} "
          f"{r['ret_on']:>+8.3%} {r['ret_off']:>+8.3%} {r['net_imp']:>+8.3%} "
          f"{r['apr_kept_pct']:>4.0%} {r['jun_blocked_pct']:>4.0%} "
          f"{r['tp_rate']:>4.0%} {r['tn_rate']:>4.0%}")


# ==============================================================================
# COMBINATION ANALYSIS: BEST 2-of-3 / 3-of-3
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  COMBINATION ANALYSIS")
print(f"{'='*W}")

# Pick the best filter from each family (by practical score)
families = {
    'vix_range': [r for r in all_results if 'vix_range' in r['col']],
    'spy_drift': [r for r in all_results if 'spy_' in r['col'] and 'return' in r['col']],
    'reversal': [r for r in all_results if 'reversal' in r['col']],
    'dd_expand': [r for r in all_results if 'avg_dd' in r['col']],
    'spy_range': [r for r in all_results if 'spy_range' in r['col']],
}

best_per_family = {}
for fam, members in families.items():
    if members:
        best_per_family[fam] = members[0]  # already sorted by score
        o = members[0]
        print(f"\n  Best {fam}: {o['col']} {o['direction']} {o['threshold']:.4f} "
              f"(net={o['net_imp']:+.3%}, blocks {o['pct_blocked']:.0%})")

# Test all pairs and triples
import itertools
family_names = list(best_per_family.keys())
print(f"\n  Testing combinations of {len(family_names)} filter families...")

combo_results = []

for k in [2, 3]:
    for combo in itertools.combinations(family_names, k):
        # "AND" logic: trade only when ALL filters in combo say ON
        masks = [best_per_family[f]['on_mask'] for f in combo]
        combined_on = masks[0]
        for m in masks[1:]:
            combined_on = combined_on & m

        combined_on = combined_on.reindex(filters.index, fill_value=False)
        on_df = filters[combined_on]
        off_df = filters[~combined_on]

        if len(on_df) < 3 or len(off_df) < 3:
            continue

        tg = filters['good_day'].sum(); tb = filters['bad_day'].sum()
        tp = on_df['good_day'].sum(); tn = off_df['bad_day'].sum()

        jun = filters[filters['month'] == '2026-06']
        jun_blocked = (~combined_on).reindex(jun.index, fill_value=False).sum()
        apr = filters[filters['month'] == '2026-04']
        apr_kept = combined_on.reindex(apr.index, fill_value=False).sum()

        net = on_df['close_return'].mean() - filters['close_return'].mean()

        combo_results.append({
            'combo': ' AND '.join(combo),
            'k': k,
            'days_on': len(on_df), 'days_off': len(off_df),
            'ret_on': on_df['close_return'].mean(),
            'ret_off': off_df['close_return'].mean(),
            'net_imp': net,
            'pct_blocked': len(off_df) / len(filters),
            'tp_rate': tp/tg if tg > 0 else 0,
            'tn_rate': tn/tb if tb > 0 else 0,
            'jun_blocked_pct': jun_blocked / len(jun) if len(jun) > 0 else 0,
            'apr_kept_pct': apr_kept / len(apr) if len(apr) > 0 else 0,
            'combined_on': combined_on,
        })

        # Monthly detail
        for m in ['2026-04', '2026-05', '2026-06']:
            m_df = filters[filters['month'] == m]
            m_on = m_df[combined_on.reindex(m_df.index, fill_value=False)]
            m_off = m_df[~combined_on.reindex(m_df.index, fill_value=False)]
            combo_results[-1][f'{m}_on'] = len(m_on)
            combo_results[-1][f'{m}_off'] = len(m_off)
            combo_results[-1][f'{m}_ret_on'] = m_on['close_return'].mean() if len(m_on) > 0 else np.nan
            combo_results[-1][f'{m}_ret_off'] = m_off['close_return'].mean() if len(m_off) > 0 else np.nan

# Also test "OR" / majority vote for triples
for combo in itertools.combinations(family_names, 3):
    masks = [best_per_family[f]['on_mask'] for f in combo]
    # 2-of-3 majority
    vote_sum = masks[0].astype(int) + masks[1].astype(int) + masks[2].astype(int)
    combined_on = vote_sum >= 2
    combined_on = combined_on.reindex(filters.index, fill_value=False)

    on_df = filters[combined_on]
    off_df = filters[~combined_on]

    if len(on_df) < 3 or len(off_df) < 3:
        continue

    tg = filters['good_day'].sum(); tb = filters['bad_day'].sum()
    tp = on_df['good_day'].sum(); tn = off_df['bad_day'].sum()
    jun = filters[filters['month'] == '2026-06']
    jun_blocked = (~combined_on).reindex(jun.index, fill_value=False).sum()
    apr = filters[filters['month'] == '2026-04']
    apr_kept = combined_on.reindex(apr.index, fill_value=False).sum()
    net = on_df['close_return'].mean() - filters['close_return'].mean()

    combo_results.append({
        'combo': f"2-of-3({','.join(combo)})",
        'k': 3,
        'days_on': len(on_df), 'days_off': len(off_df),
        'ret_on': on_df['close_return'].mean(),
        'ret_off': off_df['close_return'].mean(),
        'net_imp': net,
        'pct_blocked': len(off_df) / len(filters),
        'tp_rate': tp/tg if tg > 0 else 0,
        'tn_rate': tn/tb if tb > 0 else 0,
        'jun_blocked_pct': jun_blocked / len(jun) if len(jun) > 0 else 0,
        'apr_kept_pct': apr_kept / len(apr) if len(apr) > 0 else 0,
    })

    for m in ['2026-04', '2026-05', '2026-06']:
        m_df = filters[filters['month'] == m]
        m_on = m_df[combined_on.reindex(m_df.index, fill_value=False)]
        m_off = m_df[~combined_on.reindex(m_df.index, fill_value=False)]
        combo_results[-1][f'{m}_on'] = len(m_on)
        combo_results[-1][f'{m}_off'] = len(m_off)
        combo_results[-1][f'{m}_ret_on'] = m_on['close_return'].mean() if len(m_on) > 0 else np.nan
        combo_results[-1][f'{m}_ret_off'] = m_off['close_return'].mean() if len(m_off) > 0 else np.nan


combo_results.sort(key=lambda x: x['net_imp'], reverse=True)

print(f"\n  {'Combination':<45} {'ON':>4} {'OFF':>4} {'Ret(ON)':>9} {'Ret(OFF)':>9} {'Net':>9} "
      f"{'Apr%':>5} {'Jun%':>5} {'TPR':>5} {'TNR':>5}")
print(f"  {'─'*45} {'─'*4} {'─'*4} {'─'*9} {'─'*9} {'─'*9} {'─'*5} {'─'*5} {'─'*5} {'─'*5}")

for r in combo_results[:20]:
    print(f"  {r['combo']:<45} {r['days_on']:>4} {r['days_off']:>4} "
          f"{r['ret_on']:>+8.3%} {r['ret_off']:>+8.3%} {r['net_imp']:>+8.3%} "
          f"{r['apr_kept_pct']:>4.0%} {r['jun_blocked_pct']:>4.0%} "
          f"{r['tp_rate']:>4.0%} {r['tn_rate']:>4.0%}")

# Monthly detail for top 5
print(f"\n  --- Monthly breakdown for top 5 combinations ---")
for i, r in enumerate(combo_results[:5], 1):
    print(f"\n  #{i}: {r['combo']}")
    for m in ['2026-04', '2026-05', '2026-06']:
        on_n = r.get(f'{m}_on', 0)
        off_n = r.get(f'{m}_off', 0)
        ret_on = r.get(f'{m}_ret_on', np.nan)
        ret_off = r.get(f'{m}_ret_off', np.nan)
        ret_on_s = f"{ret_on:+.3%}" if not np.isnan(ret_on) else "N/A"
        ret_off_s = f"{ret_off:+.3%}" if not np.isnan(ret_off) else "N/A"
        print(f"    {m}: ON {on_n} days ({ret_on_s}), OFF {off_n} days ({ret_off_s})")


# ==============================================================================
# FINAL RECOMMENDATIONS
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  FINAL RECOMMENDATIONS")
print(f"{'='*W}")

print(f"""
  INDIVIDUAL FILTERS (implement these in the regime filter module):
""")

for fam in ['vix_range', 'spy_drift', 'spy_range', 'reversal', 'dd_expand']:
    if fam in best_per_family:
        o = best_per_family[fam]
        dir_sym = '<' if o['direction'] == 'below' else '>'
        print(f"  {fam.upper():}")
        print(f"    Rule: {o['col']} {dir_sym} {o['threshold']:.4f}")
        print(f"    Net improvement: {o['net_imp']:+.3%}")
        print(f"    April kept: {o['apr_kept_pct']:.0%}, June blocked: {o['jun_blocked_pct']:.0%}")
        print(f"    TPR: {o['tp_rate']:.0%}, TNR: {o['tn_rate']:.0%}")
        print(f"    Days blocked: {o['pct_blocked']:.0%}")
        print()

if combo_results:
    best_combo = combo_results[0]
    print(f"  BEST COMBINATION:")
    print(f"    {best_combo['combo']}")
    print(f"    Net improvement: {best_combo['net_imp']:+.3%}")
    print(f"    April kept: {best_combo['apr_kept_pct']:.0%}, June blocked: {best_combo['jun_blocked_pct']:.0%}")
    print(f"    TPR: {best_combo['tp_rate']:.0%}, TNR: {best_combo['tn_rate']:.0%}")
    print(f"    Days blocked: {best_combo['pct_blocked']:.0%}")

    # Day-by-day for June with best combo
    if 'combined_on' in best_combo:
        print(f"\n  Day-by-day June with best combo:")
        jun_df = filters[filters['month'] == '2026-06']
        combined_on = best_combo['combined_on']
        print(f"  {'Date':<12} {'Action':>8} {'CloseRet':>10} {'Good':>5} {'Bad':>5}")
        print(f"  {'─'*12} {'─'*8} {'─'*10} {'─'*5} {'─'*5}")
        for date, row in jun_df.iterrows():
            action = 'TRADE' if combined_on.get(date, False) else 'BLOCK'
            print(f"  {str(date.date()):<12} {action:>8} {row['close_return']:>+9.4%} "
                  f"{'Y' if row['good_day'] else ' ':>5} {'Y' if row['bad_day'] else ' ':>5}")


print(f"\n{'='*W}")
print(f"  ANALYSIS COMPLETE")
print(f"{'='*W}")
