###############################################################################
# Feature Investigation F16-F21: PM Microstructure, Options Proxy, EDGAR
# Gap-trading system feature testing
# Colab-friendly: config vars at top, no if __name__
###############################################################################

import json
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

warnings.filterwarnings('ignore')

# ── Config ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
OUT_DIR    = os.path.join(BASE_DIR, 'features')
CSV_PATH   = os.path.join(BASE_DIR, 'v2_features_full.csv')
BARS_PATH  = os.path.join(BASE_DIR, 'v3_minute_bars.json')
DAILY_PATH = os.path.join(BASE_DIR, 'v2_daily_data.json')
POLYGON_KEY = os.environ["POLYGON_API_KEY"]
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load data ───────────────────────────────────────────────────────────────
print('Loading data...')
df = pd.read_csv(CSV_PATH)
with open(BARS_PATH) as f:
    bars_data = json.load(f)
with open(DAILY_PATH) as f:
    daily_data = json.load(f)

ticker_bars = bars_data['ticker_bars']
print(f'  CSV: {len(df)} rows, {len(df.columns)} cols')
print(f'  Minute bars: {len(ticker_bars)} entries')
print(f'  Daily data: {len(daily_data)} tickers')

# ── Strategy assignment ─────────────────────────────────────────────────────
def assign_strategy(row):
    gp = row['gap_pct']
    agp = row['abs_gap_pct']
    d = row['direction']
    earn = row.get('is_earnings', False)
    if d == 'SHORT' and gp > 0:
        if agp >= 3:
            return 'V1'
        elif agp >= 1.5:
            return 'B1'
        else:
            return 'B_small'
    elif d == 'LONG' and gp < 0:
        if earn:
            return 'C3'
        elif agp >= 3:
            return 'C1'
        else:
            return 'C_small'
    elif gp > 5.0:
        return 'D1'
    elif gp < -5.0:
        return 'D2'
    return 'other'

df['strategy'] = df.apply(assign_strategy, axis=1)
print(f'\nStrategy counts:\n{df["strategy"].value_counts().to_string()}')

# ── PnL target per strategy ────────────────────────────────────────────────
def get_pnl(row):
    s = row['strategy']
    if s in ('D1', 'D2'):
        return row.get('pnl_60m_nosl', np.nan)
    return row.get('pnl_t2_close', np.nan)

df['pnl_target'] = df.apply(get_pnl, axis=1)
print(f'pnl_target non-null: {df["pnl_target"].notna().sum()} / {len(df)}')

# ── Metrics helper ──────────────────────────────────────────────────────────
def compute_metrics(pnl_array):
    pnl = np.array([x for x in pnl_array if x is not None and np.isfinite(x)])
    n = len(pnl)
    if n < 5:
        return {'n': n, 'note': 'insufficient'}
    avg = float(np.mean(pnl))
    std = float(np.std(pnl, ddof=1))
    se = std / np.sqrt(n)
    med = float(np.median(pnl))
    trimmed = float(scipy_stats.trim_mean(pnl, 0.1)) if n >= 10 else avg
    return {
        'n': n, 'avg_pnl': round(avg, 4), 'median_pnl': round(med, 4),
        'trimmed_mean': round(trimmed, 4),
        'std': round(std, 4), 'sharpe': round(avg / std if std > 0 else 0, 4),
        'win_rate': round(float((pnl > 0).mean()), 4),
        'ci_95_low': round(avg - 1.96 * se, 4), 'ci_95_high': round(avg + 1.96 * se, 4),
        'max_loss': round(float(np.min(pnl)), 4), 'max_gain': round(float(np.max(pnl)), 4),
    }

def correlation_test(x, y):
    """Pearson + Spearman correlation with p-values."""
    mask = np.isfinite(x) & np.isfinite(y)
    x2, y2 = x[mask], y[mask]
    n = len(x2)
    if n < 5:
        return {'n': n, 'note': 'insufficient'}
    pr, pp = scipy_stats.pearsonr(x2, y2)
    sr, sp = scipy_stats.spearmanr(x2, y2)
    return {
        'n': n,
        'pearson_r': round(pr, 4), 'pearson_p': round(pp, 4),
        'spearman_r': round(sr, 4), 'spearman_p': round(sp, 4),
    }

def quartile_analysis(df_sub, feature_col, pnl_col='pnl_target'):
    """Split by quartile, report avg PnL."""
    mask = df_sub[feature_col].notna() & df_sub[pnl_col].notna()
    d = df_sub.loc[mask].copy()
    if len(d) < 20:
        return {'note': f'insufficient data (n={len(d)})'}
    try:
        d['q'] = pd.qcut(d[feature_col], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
    except ValueError:
        return {'note': 'cannot create quartiles (too few unique values)'}
    result = {}
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        sub = d.loc[d['q'] == q, pnl_col].values
        result[q] = compute_metrics(sub)
    return result

def threshold_sweep(df_sub, feature_col, thresholds, pnl_col='pnl_target', direction='above'):
    """Split by threshold, report PASS/FAIL."""
    mask = df_sub[feature_col].notna() & df_sub[pnl_col].notna()
    d = df_sub.loc[mask].copy()
    results = {}
    for t in thresholds:
        if direction == 'above':
            pass_mask = d[feature_col] >= t
        else:
            pass_mask = d[feature_col] <= t
        pass_pnl = d.loc[pass_mask, pnl_col].values
        fail_pnl = d.loc[~pass_mask, pnl_col].values
        results[str(t)] = {
            'PASS': compute_metrics(pass_pnl),
            'FAIL': compute_metrics(fail_pnl),
        }
    return results

def cross_strategy_check(df_sub, feature_col, threshold, pnl_col='pnl_target', direction='above'):
    """Check consistency across strategies."""
    mask = df_sub[feature_col].notna() & df_sub[pnl_col].notna()
    d = df_sub.loc[mask].copy()
    strategies = d['strategy'].unique()
    results = {}
    better_count = 0
    tested_count = 0
    for s in strategies:
        sd = d[d['strategy'] == s]
        if direction == 'above':
            pass_mask = sd[feature_col] >= threshold
        else:
            pass_mask = sd[feature_col] <= threshold
        pass_pnl = sd.loc[pass_mask, pnl_col].values
        fail_pnl = sd.loc[~pass_mask, pnl_col].values
        pm = compute_metrics(pass_pnl)
        fm = compute_metrics(fail_pnl)
        results[s] = {'PASS': pm, 'FAIL': fm}
        if pm.get('n', 0) >= 5 and fm.get('n', 0) >= 5:
            tested_count += 1
            if pm.get('avg_pnl', 0) > fm.get('avg_pnl', 0):
                better_count += 1
    results['_summary'] = {
        'tested_strategies': tested_count,
        'better_count': better_count,
        'consistent': better_count >= 6 if tested_count >= 10 else f'{better_count}/{tested_count}',
    }
    return results


###############################################################################
# F16: PM High Retrace
###############################################################################
print('\n' + '='*80)
print('F16: PM High Retrace')
print('='*80)

def compute_pm_retrace_features(bar_map):
    pm_bars = [(t, b) for t, b in sorted(bar_map.items()) if t < '09:30' and b['v'] > 0]
    if len(pm_bars) < 5:
        return None

    pm_opens = [b['o'] for _, b in pm_bars]
    pm_highs = [b['h'] for _, b in pm_bars]
    pm_lows = [b['l'] for _, b in pm_bars]
    pm_closes = [b['c'] for _, b in pm_bars]

    pm_open = pm_opens[0]
    pm_high = max(pm_highs)
    pm_low = min(pm_lows)
    pm_last = pm_closes[-1]

    # How much of PM move has been given back?
    pm_total_move = pm_high - pm_open
    pm_retrace_pct = (pm_high - pm_last) / pm_total_move if pm_total_move > 0 else 0

    # For SHORTs (gap up): also compute retrace from low
    pm_down_move = pm_open - pm_low
    pm_retrace_from_low = (pm_last - pm_low) / pm_down_move if pm_down_move > 0 else 0

    # When did PM high occur?
    pm_high_idx = pm_highs.index(pm_high)
    pm_high_time = pm_bars[pm_high_idx][0]

    # PM VWAP
    total_vol = sum(b['v'] for _, b in pm_bars)
    pm_vwap = sum(b['c'] * b['v'] for _, b in pm_bars) / total_vol if total_vol > 0 else pm_last

    return {
        'pm_retrace_pct': pm_retrace_pct,
        'pm_retrace_from_low': pm_retrace_from_low,
        'pm_high_time': pm_high_time,
        'pm_high_early': pm_high_time < '06:00',
        'pm_high_late': pm_high_time > '08:30',
        'pm_last_vs_vwap': (pm_last - pm_vwap) / pm_vwap if pm_vwap > 0 else 0,
        'pm_bar_count': len(pm_bars),
    }

f16_rows = []
for idx, row in df.iterrows():
    key = f"{row['ticker']}__{row['eval_date']}"
    bars = ticker_bars.get(key, {})
    feats = compute_pm_retrace_features(bars)
    if feats:
        f16_rows.append({'idx': idx, **feats})

f16_df = pd.DataFrame(f16_rows).set_index('idx')
for col in f16_df.columns:
    df.loc[f16_df.index, f'f16_{col}'] = f16_df[col]

print(f'  Computed for {len(f16_rows)}/{len(df)} rows')

# Retrace buckets
def retrace_bucket(v):
    if pd.isna(v): return np.nan
    if v <= 0.2: return 'held'
    if v <= 0.5: return 'partial'
    if v <= 0.8: return 'significant'
    return 'full'

df['f16_retrace_bucket'] = df['f16_pm_retrace_pct'].apply(retrace_bucket)

f16_results = {
    'feature': 'F16_pm_retrace',
    'description': 'PM high retrace - how much of PM move given back before open',
    'validation_required': True,
    'live_trading_weeks_needed': 4,
}

# Correlation
f16_corr = correlation_test(
    df['f16_pm_retrace_pct'].values.astype(float),
    df['pnl_target'].values.astype(float)
)
f16_results['correlation_all'] = f16_corr
print(f'  Correlation (all): r={f16_corr.get("pearson_r","N/A")}, p={f16_corr.get("pearson_p","N/A")}')

# By direction
for d in ['LONG', 'SHORT']:
    mask = df['direction'] == d
    c = correlation_test(
        df.loc[mask, 'f16_pm_retrace_pct'].values.astype(float),
        df.loc[mask, 'pnl_target'].values.astype(float)
    )
    f16_results[f'correlation_{d}'] = c
    print(f'  Correlation ({d}): r={c.get("pearson_r","N/A")}, p={c.get("pearson_p","N/A")}')

# Quartile analysis
f16_results['quartile_all'] = quartile_analysis(df, 'f16_pm_retrace_pct')
for d in ['LONG', 'SHORT']:
    f16_results[f'quartile_{d}'] = quartile_analysis(df[df['direction'] == d], 'f16_pm_retrace_pct')

# Retrace bucket analysis
f16_bucket_results = {}
for bucket in ['held', 'partial', 'significant', 'full']:
    mask = (df['f16_retrace_bucket'] == bucket) & df['pnl_target'].notna()
    pnl = df.loc[mask, 'pnl_target'].values
    f16_bucket_results[bucket] = compute_metrics(pnl)
    print(f'  Bucket {bucket}: n={f16_bucket_results[bucket].get("n",0)}, '
          f'avg={f16_bucket_results[bucket].get("avg_pnl","N/A")}')
f16_results['retrace_buckets'] = f16_bucket_results

# Threshold sweep
f16_results['threshold_sweep'] = threshold_sweep(
    df, 'f16_pm_retrace_pct', [0.2, 0.3, 0.5, 0.7, 0.8])

# PM high timing
f16_timing = {}
for label, col in [('early', 'f16_pm_high_early'), ('late', 'f16_pm_high_late')]:
    mask_t = df[col] == True
    mask_f = df[col] == False
    f16_timing[f'{label}_true'] = compute_metrics(df.loc[mask_t & df['pnl_target'].notna(), 'pnl_target'].values)
    f16_timing[f'{label}_false'] = compute_metrics(df.loc[mask_f & df['pnl_target'].notna(), 'pnl_target'].values)
    print(f'  PM high {label}=True: n={f16_timing[f"{label}_true"].get("n",0)}, '
          f'avg={f16_timing[f"{label}_true"].get("avg_pnl","N/A")}')
f16_results['pm_high_timing'] = f16_timing

# VWAP position
f16_results['vwap_correlation'] = correlation_test(
    df['f16_pm_last_vs_vwap'].values.astype(float),
    df['pnl_target'].values.astype(float)
)

# Cross-strategy
f16_results['cross_strategy'] = cross_strategy_check(df, 'f16_pm_retrace_pct', 0.5)

with open(os.path.join(OUT_DIR, 'F16_pm_retrace_results.json'), 'w') as f:
    json.dump(f16_results, f, indent=2, default=str)
print('  Saved F16 results')


###############################################################################
# F17: PM Volume Profile Shape
###############################################################################
print('\n' + '='*80)
print('F17: PM Volume Profile Shape')
print('='*80)

def compute_pm_volume_profile(bar_map):
    pm_bars = [(t, b) for t, b in sorted(bar_map.items()) if t < '09:30' and b['v'] > 0]
    if len(pm_bars) < 10:
        return None

    total_vol = sum(b['v'] for _, b in pm_bars)
    if total_vol == 0:
        return None

    early = sum(b['v'] for t, b in pm_bars if t < '06:00')
    mid = sum(b['v'] for t, b in pm_bars if '06:00' <= t < '08:30')
    late = sum(b['v'] for t, b in pm_bars if t >= '08:30')

    early_pct = early / total_vol
    mid_pct = mid / total_vol
    late_pct = late / total_vol

    if early_pct > 0.5:
        profile = 'early_institutional'
    elif late_pct > 0.5:
        profile = 'late_retail'
    else:
        profile = 'distributed'

    # Volume acceleration: last hour vs first hour ratio
    first_hour = sum(b['v'] for t, b in pm_bars if t < '05:00')
    last_hour = sum(b['v'] for t, b in pm_bars if t >= '08:30')
    vol_accel = last_hour / first_hour if first_hour > 0 else 0

    return {
        'early_vol_pct': round(early_pct, 4),
        'mid_vol_pct': round(mid_pct, 4),
        'late_vol_pct': round(late_pct, 4),
        'profile_type': profile,
        'vol_accel': round(vol_accel, 4),
        'total_pm_vol': total_vol,
    }

f17_rows = []
for idx, row in df.iterrows():
    key = f"{row['ticker']}__{row['eval_date']}"
    bars = ticker_bars.get(key, {})
    feats = compute_pm_volume_profile(bars)
    if feats:
        f17_rows.append({'idx': idx, **feats})

f17_df = pd.DataFrame(f17_rows).set_index('idx')
for col in f17_df.columns:
    df.loc[f17_df.index, f'f17_{col}'] = f17_df[col]

print(f'  Computed for {len(f17_rows)}/{len(df)} rows')

f17_results = {
    'feature': 'F17_pm_volume_profile',
    'description': 'PM volume distribution shape (early/mid/late)',
    'validation_required': True,
    'live_trading_weeks_needed': 4,
}

# Profile type analysis
f17_profile = {}
for ptype in ['early_institutional', 'late_retail', 'distributed']:
    mask = (df['f17_profile_type'] == ptype) & df['pnl_target'].notna()
    pnl = df.loc[mask, 'pnl_target'].values
    f17_profile[ptype] = compute_metrics(pnl)
    print(f'  Profile {ptype}: n={f17_profile[ptype].get("n",0)}, '
          f'avg={f17_profile[ptype].get("avg_pnl","N/A")}')
f17_results['profile_type_pnl'] = f17_profile

# By direction
for d in ['LONG', 'SHORT']:
    sub = df[df['direction'] == d]
    dir_profile = {}
    for ptype in ['early_institutional', 'late_retail', 'distributed']:
        mask = (sub['f17_profile_type'] == ptype) & sub['pnl_target'].notna()
        pnl = sub.loc[mask, 'pnl_target'].values
        dir_profile[ptype] = compute_metrics(pnl)
    f17_results[f'profile_{d}'] = dir_profile

# Correlation: early_vol_pct
f17_results['correlation_early_vol'] = correlation_test(
    df['f17_early_vol_pct'].values.astype(float),
    df['pnl_target'].values.astype(float)
)
f17_results['correlation_late_vol'] = correlation_test(
    df['f17_late_vol_pct'].values.astype(float),
    df['pnl_target'].values.astype(float)
)
f17_results['correlation_vol_accel'] = correlation_test(
    df['f17_vol_accel'].values.astype(float),
    df['pnl_target'].values.astype(float)
)

print(f'  Early vol corr: r={f17_results["correlation_early_vol"].get("pearson_r","N/A")}')
print(f'  Late vol corr: r={f17_results["correlation_late_vol"].get("pearson_r","N/A")}')

# Threshold: early_vol_pct > 0.4
f17_results['early_gt_0.4'] = threshold_sweep(df, 'f17_early_vol_pct', [0.3, 0.4, 0.5])

# Quartile analysis
f17_results['quartile_early_vol'] = quartile_analysis(df, 'f17_early_vol_pct')
f17_results['quartile_late_vol'] = quartile_analysis(df, 'f17_late_vol_pct')

# Cross-strategy
f17_results['cross_strategy_early'] = cross_strategy_check(df, 'f17_early_vol_pct', 0.4)

with open(os.path.join(OUT_DIR, 'F17_pm_volume_profile_results.json'), 'w') as f:
    json.dump(f17_results, f, indent=2, default=str)
print('  Saved F17 results')


###############################################################################
# F18: PM Price Path Slope and Stability
###############################################################################
print('\n' + '='*80)
print('F18: PM Price Path Slope and Stability')
print('='*80)

def compute_pm_path(bar_map):
    pm_bars = [(t, b) for t, b in sorted(bar_map.items()) if t < '09:30' and b['v'] > 0]
    if len(pm_bars) < 10:
        return None

    closes = [b['c'] for _, b in pm_bars]
    slope, intercept, r_val, p_val, se = scipy_stats.linregress(range(len(closes)), closes)
    normalized_slope = slope / closes[0] if closes[0] > 0 else 0

    # Direction changes
    returns = []
    for i in range(1, len(closes)):
        if closes[i - 1] > 0:
            returns.append((closes[i] - closes[i - 1]) / closes[i - 1])
    dir_changes = sum(1 for i in range(1, len(returns)) if (returns[i] > 0) != (returns[i - 1] > 0))
    dir_changes_pct = dir_changes / len(returns) if returns else 0

    # Late PM consolidation (last 30 bars before 9:30)
    late_pm = closes[-30:] if len(closes) >= 30 else closes
    consol_range = (max(late_pm) - min(late_pm)) / np.mean(late_pm) if np.mean(late_pm) > 0 else 0

    # Acceleration: slope of second half vs first half
    n = len(closes)
    half = n // 2
    if half >= 5:
        s1, _, _, _, _ = scipy_stats.linregress(range(half), closes[:half])
        s2, _, _, _, _ = scipy_stats.linregress(range(n - half), closes[half:])
        slope_accel = (s2 - s1) / closes[0] if closes[0] > 0 else 0
    else:
        slope_accel = 0

    return {
        'pm_slope': round(normalized_slope, 8),
        'pm_slope_r2': round(r_val ** 2, 4),
        'pm_direction_changes': dir_changes,
        'pm_dir_changes_pct': round(dir_changes_pct, 4),
        'pm_consolidation_range': round(consol_range, 6),
        'pm_trending_down': normalized_slope < -0.0001 and r_val ** 2 > 0.3,
        'pm_trending_up': normalized_slope > 0.0001 and r_val ** 2 > 0.3,
        'pm_slope_accel': round(slope_accel, 8),
        'pm_bar_count_f18': len(pm_bars),
    }

f18_rows = []
for idx, row in df.iterrows():
    key = f"{row['ticker']}__{row['eval_date']}"
    bars = ticker_bars.get(key, {})
    feats = compute_pm_path(bars)
    if feats:
        f18_rows.append({'idx': idx, **feats})

f18_df = pd.DataFrame(f18_rows).set_index('idx')
for col in f18_df.columns:
    df.loc[f18_df.index, f'f18_{col}'] = f18_df[col]

print(f'  Computed for {len(f18_rows)}/{len(df)} rows')

f18_results = {
    'feature': 'F18_pm_path_slope',
    'description': 'PM price path linear regression, direction changes, consolidation',
    'validation_required': True,
    'live_trading_weeks_needed': 4,
}

# Correlations
for feat in ['pm_slope', 'pm_slope_r2', 'pm_dir_changes_pct', 'pm_consolidation_range', 'pm_slope_accel']:
    col = f'f18_{feat}'
    c = correlation_test(df[col].values.astype(float), df['pnl_target'].values.astype(float))
    f18_results[f'correlation_{feat}'] = c
    print(f'  {feat}: r={c.get("pearson_r","N/A")}, p={c.get("pearson_p","N/A")}')

# Quartile analysis for key features
f18_results['quartile_slope'] = quartile_analysis(df, 'f18_pm_slope')
f18_results['quartile_r2'] = quartile_analysis(df, 'f18_pm_slope_r2')
f18_results['quartile_consol'] = quartile_analysis(df, 'f18_pm_consolidation_range')

# Trending down for SHORT
short_mask = df['direction'] == 'SHORT'
td_mask = df['f18_pm_trending_down'] == True
f18_results['short_trending_down'] = {
    'trending_down': compute_metrics(df.loc[short_mask & td_mask & df['pnl_target'].notna(), 'pnl_target'].values),
    'not_trending_down': compute_metrics(df.loc[short_mask & ~td_mask & df['pnl_target'].notna(), 'pnl_target'].values),
}
td_m = f18_results['short_trending_down']
print(f'  SHORT + trending_down: n={td_m["trending_down"].get("n",0)}, '
      f'avg={td_m["trending_down"].get("avg_pnl","N/A")}')
print(f'  SHORT + NOT trending_down: n={td_m["not_trending_down"].get("n",0)}, '
      f'avg={td_m["not_trending_down"].get("avg_pnl","N/A")}')

# Trending up for LONG
long_mask = df['direction'] == 'LONG'
tu_mask = df['f18_pm_trending_up'] == True
f18_results['long_trending_up'] = {
    'trending_up': compute_metrics(df.loc[long_mask & tu_mask & df['pnl_target'].notna(), 'pnl_target'].values),
    'not_trending_up': compute_metrics(df.loc[long_mask & ~tu_mask & df['pnl_target'].notna(), 'pnl_target'].values),
}

# Consolidation threshold
f18_results['consolidation_sweep'] = threshold_sweep(
    df, 'f18_pm_consolidation_range', [0.005, 0.01, 0.02, 0.03, 0.05], direction='below')

# Cross-strategy
f18_results['cross_strategy_slope'] = cross_strategy_check(df, 'f18_pm_slope', 0)
f18_results['cross_strategy_consol'] = cross_strategy_check(
    df, 'f18_pm_consolidation_range', 0.02, direction='below')

with open(os.path.join(OUT_DIR, 'F18_pm_path_slope_results.json'), 'w') as f:
    json.dump(f18_results, f, indent=2, default=str)
print('  Saved F18 results')


###############################################################################
# F19: Put/Call Ratio (Options data)
###############################################################################
print('\n' + '='*80)
print('F19: Put/Call Ratio')
print('='*80)

import requests

f19_results = {
    'feature': 'F19_put_call_ratio',
    'validation_required': True,
    'live_trading_weeks_needed': 4,
}

# Try options snapshot API first
unique_tickers = df['ticker'].unique()[:5]  # test with 5 first
options_available = False
api_errors = []

for test_ticker in unique_tickers[:2]:
    try:
        url = f'https://api.polygon.io/v3/snapshot/options/{test_ticker}'
        resp = requests.get(url, params={'apiKey': POLYGON_KEY}, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data.get('results'):
                options_available = True
                print(f'  Options snapshot API works for {test_ticker}')
                break
        else:
            api_errors.append(f'{test_ticker}: status {resp.status_code} - {resp.text[:200]}')
    except Exception as e:
        api_errors.append(f'{test_ticker}: {str(e)}')

if not options_available:
    # Try contracts endpoint
    for test_ticker in unique_tickers[:2]:
        try:
            url = f'https://api.polygon.io/v3/reference/options/contracts'
            resp = requests.get(url, params={
                'underlying_ticker': test_ticker,
                'apiKey': POLYGON_KEY,
                'limit': 10,
            }, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if data.get('results'):
                    options_available = True
                    print(f'  Options contracts API works for {test_ticker}')
                    break
            else:
                api_errors.append(f'contracts {test_ticker}: status {resp.status_code} - {resp.text[:200]}')
        except Exception as e:
            api_errors.append(f'contracts {test_ticker}: {str(e)}')

if not options_available:
    print(f'  OPTIONS API NOT AVAILABLE. Errors:')
    for e in api_errors:
        print(f'    {e}')
    f19_results['status'] = 'SKIPPED'
    f19_results['reason'] = 'Options API not available at current Polygon plan tier'
    f19_results['api_errors'] = api_errors
    with open(os.path.join(OUT_DIR, 'F19_put_call_ratio_results.json'), 'w') as f:
        json.dump(f19_results, f, indent=2, default=str)
    print('  Saved F19 (skipped)')
else:
    # If API works, collect put/call data
    print('  Collecting options data...')
    pc_data = {}
    for ticker in unique_tickers:
        try:
            url = f'https://api.polygon.io/v3/snapshot/options/{ticker}'
            resp = requests.get(url, params={'apiKey': POLYGON_KEY}, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                results = data.get('results', [])
                puts = sum(1 for r in results if r.get('details', {}).get('contract_type') == 'put')
                calls = sum(1 for r in results if r.get('details', {}).get('contract_type') == 'call')
                pc_ratio = puts / calls if calls > 0 else None
                pc_data[ticker] = {'puts': puts, 'calls': calls, 'pc_ratio': pc_ratio}
        except Exception:
            pass

    df['f19_pc_ratio'] = df['ticker'].map(lambda t: pc_data.get(t, {}).get('pc_ratio', np.nan))
    f19_results['data_collected'] = len(pc_data)
    f19_results['correlation'] = correlation_test(
        df['f19_pc_ratio'].values.astype(float),
        df['pnl_target'].values.astype(float)
    )
    f19_results['quartile'] = quartile_analysis(df, 'f19_pc_ratio')
    with open(os.path.join(OUT_DIR, 'F19_put_call_ratio_results.json'), 'w') as f:
        json.dump(f19_results, f, indent=2, default=str)
    print('  Saved F19 results')


###############################################################################
# F20: Implied Volatility Rank (HV proxy)
###############################################################################
print('\n' + '='*80)
print('F20: IV Rank (HV proxy)')
print('='*80)

f20_results = {
    'feature': 'F20_hv_rank',
    'description': '20-day historical volatility rank as IV proxy',
    'validation_required': True,
    'live_trading_weeks_needed': 4,
}

def compute_hv_rank(ticker, eval_date):
    """Compute 20-day historical vol and its rank in available history."""
    if ticker not in daily_data:
        return None
    dates = sorted(daily_data[ticker].keys())
    if len(dates) < 10:
        return None

    # Find eval_date position
    try:
        date_idx = dates.index(eval_date)
    except ValueError:
        # Find closest prior date
        prior = [d for d in dates if d <= eval_date]
        if not prior:
            return None
        date_idx = dates.index(prior[-1])

    if date_idx < 5:
        return None

    # Get closes
    closes = []
    for d in dates[:date_idx + 1]:
        closes.append(daily_data[ticker][d]['close'])

    if len(closes) < 6:
        return None

    closes = np.array(closes)
    log_rets = np.diff(np.log(closes))

    # Current HV: use last N days available (up to 20)
    lookback = min(20, len(log_rets))
    current_hv = np.std(log_rets[-lookback:]) * np.sqrt(252)

    # Rolling HV for rank computation
    window = min(10, lookback)
    hvs = []
    for i in range(window, len(log_rets) + 1):
        hv = np.std(log_rets[max(0, i - lookback):i]) * np.sqrt(252)
        hvs.append(hv)

    if len(hvs) < 3:
        return None

    hv_min = min(hvs)
    hv_max = max(hvs)
    hv_rank = (current_hv - hv_min) / (hv_max - hv_min) if hv_max > hv_min else 0.5

    return {
        'current_hv': round(current_hv, 4),
        'hv_rank': round(hv_rank, 4),
        'hv_min': round(hv_min, 4),
        'hv_max': round(hv_max, 4),
        'hv_history_len': len(hvs),
    }

f20_rows = []
for idx, row in df.iterrows():
    feats = compute_hv_rank(row['ticker'], row['eval_date'])
    if feats:
        f20_rows.append({'idx': idx, **feats})

f20_df = pd.DataFrame(f20_rows).set_index('idx')
for col in f20_df.columns:
    df.loc[f20_df.index, f'f20_{col}'] = f20_df[col]

print(f'  Computed for {len(f20_rows)}/{len(df)} rows')

# Correlation
f20_corr_hv = correlation_test(
    df['f20_current_hv'].values.astype(float),
    df['pnl_target'].values.astype(float)
)
f20_corr_rank = correlation_test(
    df['f20_hv_rank'].values.astype(float),
    df['pnl_target'].values.astype(float)
)
f20_results['correlation_current_hv'] = f20_corr_hv
f20_results['correlation_hv_rank'] = f20_corr_rank
print(f'  Current HV corr: r={f20_corr_hv.get("pearson_r","N/A")}, p={f20_corr_hv.get("pearson_p","N/A")}')
print(f'  HV rank corr: r={f20_corr_rank.get("pearson_r","N/A")}, p={f20_corr_rank.get("pearson_p","N/A")}')

# By direction
for d in ['LONG', 'SHORT']:
    mask = df['direction'] == d
    f20_results[f'correlation_hv_rank_{d}'] = correlation_test(
        df.loc[mask, 'f20_hv_rank'].values.astype(float),
        df.loc[mask, 'pnl_target'].values.astype(float)
    )

# Quartile
f20_results['quartile_hv_rank'] = quartile_analysis(df, 'f20_hv_rank')
f20_results['quartile_current_hv'] = quartile_analysis(df, 'f20_current_hv')

# Threshold: high HV rank predicts stronger fade?
f20_results['hv_rank_sweep'] = threshold_sweep(df, 'f20_hv_rank', [0.5, 0.6, 0.7, 0.8, 0.9])

# Cross-strategy
f20_results['cross_strategy'] = cross_strategy_check(df, 'f20_hv_rank', 0.7)

with open(os.path.join(OUT_DIR, 'F20_iv_rank_results.json'), 'w') as f:
    json.dump(f20_results, f, indent=2, default=str)
print('  Saved F20 results')


###############################################################################
# F21: Offering Detection via EDGAR
###############################################################################
print('\n' + '='*80)
print('F21: EDGAR Offering Detection')
print('='*80)

f21_results = {
    'feature': 'F21_edgar_offering',
    'description': 'SEC EDGAR filing check for offerings near gap date',
    'validation_required': True,
    'live_trading_weeks_needed': 4,
}

def check_edgar_filing(ticker, date):
    """Check EDGAR full-text search for offering filings."""
    start_dt = (pd.Timestamp(date) - pd.Timedelta(days=3)).strftime('%Y-%m-%d')
    # Try the EDGAR full-text search API
    url = 'https://efts.sec.gov/LATEST/search-index'
    params = {
        'q': f'"{ticker}"',
        'dateRange': 'custom',
        'startdt': start_dt,
        'enddt': date,
        'forms': 'S-1,424B4,424B3,S-3,FWP,EFFECT',
    }
    headers = {'User-Agent': 'YS Research research@example.com'}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            hits = data.get('hits', {}).get('hits', [])
            return {'has_offering_filing': len(hits) > 0, 'filing_count': len(hits),
                    'api': 'search-index', 'status': 'ok'}
    except Exception as e:
        pass

    # Try alternative EDGAR EFTS search endpoint
    url2 = 'https://efts.sec.gov/LATEST/search-index'
    params2 = {
        'q': ticker,
        'dateRange': 'custom',
        'startdt': start_dt,
        'enddt': date,
    }
    try:
        resp2 = requests.get(url2, params=params2, headers=headers, timeout=10)
        if resp2.status_code == 200:
            data2 = resp2.json()
            hits = data2.get('hits', {}).get('hits', [])
            # Filter for offering-related forms
            offering_forms = {'S-1', 'S-3', '424B3', '424B4', '424B5', 'FWP', 'EFFECT',
                            'S-1/A', 'S-3/A'}
            offering_hits = [h for h in hits
                           if h.get('_source', {}).get('form_type', '') in offering_forms]
            return {'has_offering_filing': len(offering_hits) > 0,
                    'filing_count': len(offering_hits),
                    'total_filings': len(hits),
                    'api': 'search-index-unfiltered', 'status': 'ok'}
    except Exception:
        pass

    return {'has_offering_filing': False, 'filing_count': 0, 'api': 'failed', 'status': 'error'}

# Test with a few tickers first
test_tickers = df[df['catalyst_type'] == 'offering']['ticker'].unique()[:3]
edgar_api_works = False

print(f'  Testing EDGAR API with offering tickers: {list(test_tickers)}')
for tt in test_tickers:
    test_date = df[df['ticker'] == tt]['eval_date'].iloc[0]
    result = check_edgar_filing(tt, test_date)
    print(f'    {tt} ({test_date}): {result}')
    if result['status'] == 'ok':
        edgar_api_works = True

if not edgar_api_works:
    # Alternative: use EDGAR company search
    print('  Primary EDGAR search failed, trying company filings API...')

    def check_edgar_company(ticker, date):
        """Fallback: check EDGAR company filings."""
        headers = {'User-Agent': 'YS Research research@example.com'}
        # First get CIK for ticker
        try:
            url = f'https://www.sec.gov/cgi-bin/browse-edgar?company=&CIK={ticker}&type=S-3&dateb=&owner=include&count=10&search_text=&action=getcompany&output=atom'
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200 and 'entry' in resp.text:
                return {'has_offering_filing': True, 'filing_count': 1,
                        'api': 'browse-edgar', 'status': 'ok'}
        except Exception:
            pass
        return {'has_offering_filing': False, 'filing_count': 0, 'api': 'fallback', 'status': 'error'}

    for tt in test_tickers:
        test_date = df[df['ticker'] == tt]['eval_date'].iloc[0]
        result = check_edgar_company(tt, test_date)
        print(f'    Fallback {tt}: {result}')
        if result['status'] == 'ok':
            edgar_api_works = True

# Regardless of EDGAR API, cross-validate with catalyst_type field
print('\n  Cross-validating catalyst_type=="offering" with PnL...')

df['f21_is_offering'] = df['catalyst_type'] == 'offering'

# PnL for offering vs non-offering
offering_mask = df['f21_is_offering'] & df['pnl_target'].notna()
non_offering_mask = (~df['f21_is_offering']) & df['pnl_target'].notna()

offering_pnl = compute_metrics(df.loc[offering_mask, 'pnl_target'].values)
non_offering_pnl = compute_metrics(df.loc[non_offering_mask, 'pnl_target'].values)

print(f'  Offering: {offering_pnl}')
print(f'  Non-offering: {non_offering_pnl}')

f21_results['catalyst_offering'] = {
    'offering': offering_pnl,
    'non_offering': non_offering_pnl,
}

# By direction
for d in ['LONG', 'SHORT']:
    mask_d = df['direction'] == d
    f21_results[f'offering_{d}'] = {
        'offering': compute_metrics(df.loc[mask_d & offering_mask, 'pnl_target'].values),
        'non_offering': compute_metrics(df.loc[mask_d & non_offering_mask, 'pnl_target'].values),
    }

# Now run EDGAR for all offering tickers if API works
if edgar_api_works:
    print('\n  Running EDGAR checks for all tickers...')
    edgar_results_detail = {}
    import time
    checked = 0
    for idx, row in df.iterrows():
        if checked > 0 and checked % 20 == 0:
            time.sleep(1)  # Rate limiting
        result = check_edgar_filing(row['ticker'], row['eval_date'])
        edgar_results_detail[f"{row['ticker']}__{row['eval_date']}"] = result
        df.loc[idx, 'f21_edgar_filing'] = result['has_offering_filing']
        checked += 1

    # Cross-validate EDGAR vs catalyst_type
    df['f21_edgar_filing'] = df['f21_edgar_filing'].fillna(False).astype(bool)
    edgar_vs_catalyst = pd.crosstab(df['f21_is_offering'], df['f21_edgar_filing'],
                                     rownames=['catalyst_offering'], colnames=['edgar_filing'])
    f21_results['edgar_vs_catalyst_crosstab'] = edgar_vs_catalyst.to_dict()
    print(f'\n  EDGAR vs catalyst crosstab:\n{edgar_vs_catalyst}')

    # EDGAR-confirmed offerings PnL
    edgar_confirmed = df['f21_edgar_filing'] == True
    f21_results['edgar_confirmed_pnl'] = {
        'confirmed': compute_metrics(df.loc[edgar_confirmed & df['pnl_target'].notna(), 'pnl_target'].values),
        'not_confirmed': compute_metrics(df.loc[~edgar_confirmed & df['pnl_target'].notna(), 'pnl_target'].values),
    }
    f21_results['edgar_api_status'] = 'available'
    f21_results['edgar_checks_run'] = checked
else:
    f21_results['edgar_api_status'] = 'unavailable'
    f21_results['note'] = 'EDGAR API returned errors; analysis uses catalyst_type field only'

# Also check other catalyst types for comparison
print('\n  Catalyst type PnL comparison:')
catalyst_pnl = {}
for ct in df['catalyst_type'].unique():
    mask_ct = (df['catalyst_type'] == ct) & df['pnl_target'].notna()
    pnl = df.loc[mask_ct, 'pnl_target'].values
    m = compute_metrics(pnl)
    catalyst_pnl[ct] = m
    print(f'    {ct}: n={m.get("n",0)}, avg={m.get("avg_pnl","N/A")}')
f21_results['catalyst_type_comparison'] = catalyst_pnl

with open(os.path.join(OUT_DIR, 'F21_edgar_offering_results.json'), 'w') as f:
    json.dump(f21_results, f, indent=2, default=str)
print('  Saved F21 results')


###############################################################################
# Summary: F16-F21
###############################################################################
print('\n' + '='*80)
print('SUMMARY: F16-F21')
print('='*80)

summary = {
    'validation_required': True,
    'live_trading_weeks_needed': 4,
    'dataset': {'rows': len(df), 'dates': sorted(df['eval_date'].unique().tolist()),
                'tickers': int(df['ticker'].nunique())},
    'features': {},
}

# Load individual results
for fname, label in [
    ('F16_pm_retrace_results.json', 'F16_pm_retrace'),
    ('F17_pm_volume_profile_results.json', 'F17_pm_volume_profile'),
    ('F18_pm_path_slope_results.json', 'F18_pm_path_slope'),
    ('F19_put_call_ratio_results.json', 'F19_put_call_ratio'),
    ('F20_iv_rank_results.json', 'F20_hv_rank'),
    ('F21_edgar_offering_results.json', 'F21_edgar_offering'),
]:
    fpath = os.path.join(OUT_DIR, fname)
    if os.path.exists(fpath):
        with open(fpath) as f:
            data = json.load(f)

        # Extract key signal
        corr = data.get('correlation_all', data.get('correlation_hv_rank',
               data.get('correlation_early_vol', {})))
        pr = corr.get('pearson_r', 'N/A')
        pp = corr.get('pearson_p', 'N/A')

        # Determine signal strength
        if isinstance(pr, (int, float)) and isinstance(pp, (int, float)):
            if abs(pr) >= 0.2 and pp < 0.05:
                signal = 'MODERATE'
            elif abs(pr) >= 0.1 and pp < 0.1:
                signal = 'WEAK'
            elif pp < 0.05:
                signal = 'SIGNIFICANT_BUT_SMALL'
            else:
                signal = 'NO_SIGNAL'
        else:
            signal = 'SKIPPED' if data.get('status') == 'SKIPPED' else 'UNKNOWN'

        summary['features'][label] = {
            'signal_strength': signal,
            'pearson_r': pr,
            'pearson_p': pp,
            'status': data.get('status', 'computed'),
        }
        print(f'  {label}: signal={signal}, r={pr}, p={pp}')

with open(os.path.join(OUT_DIR, 'F16_F21_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2, default=str)
print('\n  Saved summary')


###############################################################################
# Charts
###############################################################################
print('\nGenerating charts...')

fig, axes = plt.subplots(3, 3, figsize=(20, 18))
fig.suptitle('F16-F21 Feature Investigation', fontsize=16, fontweight='bold')

# ── F16: Retrace bucket bar chart ──
ax = axes[0, 0]
mask_pnl = df['pnl_target'].notna()
buckets = ['held', 'partial', 'significant', 'full']
bucket_avgs = []
bucket_ns = []
for b in buckets:
    m = (df['f16_retrace_bucket'] == b) & mask_pnl
    vals = df.loc[m, 'pnl_target'].values
    bucket_avgs.append(np.mean(vals) if len(vals) > 0 else 0)
    bucket_ns.append(len(vals))
colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in bucket_avgs]
bars = ax.bar(buckets, bucket_avgs, color=colors, edgecolor='black', linewidth=0.5)
for bar, n in zip(bars, bucket_ns):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
            f'n={n}', ha='center', va='bottom', fontsize=9)
ax.set_title('F16: PM Retrace Bucket vs Avg PnL', fontweight='bold')
ax.set_ylabel('Avg PnL (%)')
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xlabel('Retrace Bucket')

# ── F16: Scatter retrace vs PnL ──
ax = axes[0, 1]
mask = df['f16_pm_retrace_pct'].notna() & mask_pnl
x = df.loc[mask, 'f16_pm_retrace_pct'].values.astype(float)
y = df.loc[mask, 'pnl_target'].values.astype(float)
colors_scatter = ['#3498db' if d == 'LONG' else '#e74c3c' for d in df.loc[mask, 'direction']]
ax.scatter(x, y, c=colors_scatter, alpha=0.4, s=20, edgecolor='none')
if len(x) > 5:
    z = np.polyfit(x[np.isfinite(x) & np.isfinite(y)],
                   y[np.isfinite(x) & np.isfinite(y)], 1)
    p = np.poly1d(z)
    xs = np.linspace(np.nanmin(x), np.nanmax(x), 100)
    ax.plot(xs, p(xs), 'k--', linewidth=1.5, label=f'slope={z[0]:.2f}')
    ax.legend(fontsize=8)
ax.set_title('F16: PM Retrace vs T+2 PnL', fontweight='bold')
ax.set_xlabel('PM Retrace %')
ax.set_ylabel('PnL (%)')
ax.axhline(0, color='black', linewidth=0.5)

# ── F17: Volume profile bar chart ──
ax = axes[0, 2]
profiles = ['early_institutional', 'late_retail', 'distributed']
prof_avgs = []
prof_ns = []
for p in profiles:
    m = (df['f17_profile_type'] == p) & mask_pnl
    vals = df.loc[m, 'pnl_target'].values
    prof_avgs.append(np.mean(vals) if len(vals) > 0 else 0)
    prof_ns.append(len(vals))
colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in prof_avgs]
short_labels = ['Early\nInstitutional', 'Late\nRetail', 'Distributed']
bars = ax.bar(short_labels, prof_avgs, color=colors, edgecolor='black', linewidth=0.5)
for bar, n in zip(bars, prof_ns):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
            f'n={n}', ha='center', va='bottom', fontsize=9)
ax.set_title('F17: PM Vol Profile vs Avg PnL', fontweight='bold')
ax.set_ylabel('Avg PnL (%)')
ax.axhline(0, color='black', linewidth=0.5)

# ── F18: Slope scatter ──
ax = axes[1, 0]
mask18 = df['f18_pm_slope'].notna() & mask_pnl
x18 = df.loc[mask18, 'f18_pm_slope'].values.astype(float)
y18 = df.loc[mask18, 'pnl_target'].values.astype(float)
# Clip extreme slopes for readability
clip_mask = (np.abs(x18) < 0.01)
x18c = x18[clip_mask]
y18c = y18[clip_mask]
colors18 = ['#3498db' if d == 'LONG' else '#e74c3c'
            for d, cm in zip(df.loc[mask18, 'direction'].values, clip_mask) if cm]
ax.scatter(x18c, y18c, c=colors18, alpha=0.4, s=20, edgecolor='none')
ax.set_title('F18: PM Slope vs PnL', fontweight='bold')
ax.set_xlabel('Normalized PM Slope')
ax.set_ylabel('PnL (%)')
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')

# ── F18: Consolidation range quartiles ──
ax = axes[1, 1]
mask_c = df['f18_pm_consolidation_range'].notna() & mask_pnl
d_c = df.loc[mask_c].copy()
if len(d_c) >= 20:
    try:
        d_c['cq'] = pd.qcut(d_c['f18_pm_consolidation_range'], 4,
                             labels=['Q1\n(tight)', 'Q2', 'Q3', 'Q4\n(wide)'], duplicates='drop')
        q_avgs = d_c.groupby('cq')['pnl_target'].mean()
        q_ns = d_c.groupby('cq')['pnl_target'].count()
        colors_q = ['#2ecc71' if v > 0 else '#e74c3c' for v in q_avgs.values]
        bars = ax.bar(q_avgs.index.astype(str), q_avgs.values, color=colors_q,
                      edgecolor='black', linewidth=0.5)
        for bar, n in zip(bars, q_ns.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'n={n}', ha='center', va='bottom', fontsize=9)
    except ValueError:
        ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes,
                ha='center', va='center')
ax.set_title('F18: PM Consolidation Range Quartiles', fontweight='bold')
ax.set_ylabel('Avg PnL (%)')
ax.axhline(0, color='black', linewidth=0.5)

# ── F20: HV rank quartiles ──
ax = axes[1, 2]
mask20 = df['f20_hv_rank'].notna() & mask_pnl
d20 = df.loc[mask20].copy()
if len(d20) >= 20:
    try:
        d20['hq'] = pd.qcut(d20['f20_hv_rank'], 4,
                             labels=['Q1\n(low HV)', 'Q2', 'Q3', 'Q4\n(high HV)'], duplicates='drop')
        q_avgs = d20.groupby('hq')['pnl_target'].mean()
        q_ns = d20.groupby('hq')['pnl_target'].count()
        colors_q = ['#2ecc71' if v > 0 else '#e74c3c' for v in q_avgs.values]
        bars = ax.bar(q_avgs.index.astype(str), q_avgs.values, color=colors_q,
                      edgecolor='black', linewidth=0.5)
        for bar, n in zip(bars, q_ns.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'n={n}', ha='center', va='bottom', fontsize=9)
    except ValueError:
        ax.text(0.5, 0.5, 'Cannot create quartiles', transform=ax.transAxes,
                ha='center', va='center')
ax.set_title('F20: HV Rank Quartiles vs PnL', fontweight='bold')
ax.set_ylabel('Avg PnL (%)')
ax.axhline(0, color='black', linewidth=0.5)

# ── F21: Catalyst type comparison ──
ax = axes[2, 0]
cat_data = []
for ct in df['catalyst_type'].unique():
    m_ct = (df['catalyst_type'] == ct) & mask_pnl
    vals = df.loc[m_ct, 'pnl_target'].values
    if len(vals) >= 5:
        cat_data.append((ct, np.mean(vals), len(vals)))
cat_data.sort(key=lambda x: x[1])
labels = [c[0] for c in cat_data]
avgs = [c[1] for c in cat_data]
ns = [c[2] for c in cat_data]
colors_cat = ['#2ecc71' if v > 0 else '#e74c3c' for v in avgs]
bars = ax.barh(labels, avgs, color=colors_cat, edgecolor='black', linewidth=0.5)
for bar, n in zip(bars, ns):
    ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
            f' n={n}', ha='left', va='center', fontsize=8)
ax.set_title('F21: Catalyst Type vs Avg PnL', fontweight='bold')
ax.set_xlabel('Avg PnL (%)')
ax.axvline(0, color='black', linewidth=0.5)

# ── F16: Retrace by direction ──
ax = axes[2, 1]
for d, color, label in [('LONG', '#3498db', 'LONG'), ('SHORT', '#e74c3c', 'SHORT')]:
    sub = df[(df['direction'] == d) & df['f16_pm_retrace_pct'].notna() & mask_pnl]
    if len(sub) < 10:
        continue
    try:
        sub_sorted = sub.sort_values('f16_pm_retrace_pct')
        sub_sorted['retrace_q'] = pd.qcut(sub_sorted['f16_pm_retrace_pct'], 4,
                                            labels=[1, 2, 3, 4], duplicates='drop')
        q_means = sub_sorted.groupby('retrace_q')['pnl_target'].mean()
        ax.plot(q_means.index.astype(int), q_means.values, 'o-', color=color, label=label,
                linewidth=2, markersize=8)
    except ValueError:
        pass
ax.set_title('F16: Retrace Quartile PnL by Direction', fontweight='bold')
ax.set_xlabel('Retrace Quartile (1=held, 4=full)')
ax.set_ylabel('Avg PnL (%)')
ax.legend()
ax.axhline(0, color='black', linewidth=0.5)

# ── Summary signal table ──
ax = axes[2, 2]
ax.axis('off')
table_data = []
for label, info in summary.get('features', {}).items():
    r = info.get('pearson_r', 'N/A')
    p = info.get('pearson_p', 'N/A')
    sig = info.get('signal_strength', 'N/A')
    r_str = f'{r:.3f}' if isinstance(r, float) else str(r)
    p_str = f'{p:.3f}' if isinstance(p, float) else str(p)
    table_data.append([label.replace('_', ' '), r_str, p_str, sig])

if table_data:
    table = ax.table(cellText=table_data,
                     colLabels=['Feature', 'Pearson r', 'p-value', 'Signal'],
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    # Color code signal column
    for i, row in enumerate(table_data):
        sig = row[3]
        if sig == 'MODERATE':
            table[i + 1, 3].set_facecolor('#a8e6a3')
        elif sig == 'WEAK':
            table[i + 1, 3].set_facecolor('#ffffb3')
        elif sig == 'NO_SIGNAL':
            table[i + 1, 3].set_facecolor('#ffb3b3')
        elif sig == 'SKIPPED':
            table[i + 1, 3].set_facecolor('#d3d3d3')
ax.set_title('Signal Summary', fontweight='bold', pad=20)

plt.tight_layout(rect=[0, 0, 1, 0.96])
chart_path = os.path.join(OUT_DIR, 'F16_F21_analysis.png')
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'  Chart saved: {chart_path}')

print('\n' + '='*80)
print('COMPLETE: All F16-F21 results saved to', OUT_DIR)
print('='*80)
