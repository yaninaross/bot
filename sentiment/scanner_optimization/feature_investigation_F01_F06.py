###############################################################################
# Feature Investigation F01-F06 for Gap-Trading System
# Tests 6 candidate features against T+2 and 60m PnL targets
# Colab-friendly: config vars at top, no if __name__
###############################################################################

import os
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict

warnings.filterwarnings('ignore')

# ── Config ───────────────────────────────────────────────────────────────────
BASE_DIR = '/home/admin_yaninaso_altostrat_com/bot-RAG/sentiment/scanner_optimization'
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FEATURES_DIR = os.path.join(RESULTS_DIR, 'features')
os.makedirs(FEATURES_DIR, exist_ok=True)

CSV_PATH = os.path.join(RESULTS_DIR, 'v2_features_full.csv')
MINUTE_BARS_PATH = os.path.join(RESULTS_DIR, 'v3_minute_bars.json')
DAILY_DATA_PATH = os.path.join(RESULTS_DIR, 'v2_daily_data.json')

CHART_PATH = os.path.join(FEATURES_DIR, 'F01_F06_analysis.png')
SUMMARY_PATH = os.path.join(FEATURES_DIR, 'F01_F06_summary.json')

# Correlation pass thresholds
CORR_THRESHOLD = 0.15
P_THRESHOLD = 0.05
CONSISTENCY_THRESHOLD = 6  # out of 10

# ── Load Data ────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(CSV_PATH)
with open(MINUTE_BARS_PATH) as f:
    minute_data = json.load(f)
with open(DAILY_DATA_PATH) as f:
    daily_data = json.load(f)

ticker_bars = minute_data['ticker_bars']
spy_bars = minute_data['spy_bars']

print(f"  CSV: {df.shape[0]} rows, {df.shape[1]} cols")
print(f"  Minute bars: {len(ticker_bars)} entries")
print(f"  Daily data: {len(daily_data)} tickers")

# ── Utility Functions ────────────────────────────────────────────────────────

def compute_metrics(pnl_array):
    """Standard metrics for a PnL array."""
    pnl = np.array([x for x in pnl_array if x is not None and np.isfinite(x)])
    n = len(pnl)
    if n < 5:
        return {'n': n, 'note': 'insufficient'}
    avg = float(np.mean(pnl))
    med = float(np.median(pnl))
    std = float(np.std(pnl, ddof=1))
    se = std / np.sqrt(n)
    return {
        'n': n,
        'avg_pnl': round(avg, 4),
        'median_pnl': round(med, 4),
        'std': round(std, 4),
        'sharpe': round(avg / std if std > 0 else 0, 4),
        'win_rate': round(float((pnl > 0).mean()), 4),
        'ci_95_low': round(avg - 1.96 * se, 4),
        'ci_95_high': round(avg + 1.96 * se, 4),
        'max_loss': round(float(np.min(pnl)), 4),
        'max_gain': round(float(np.max(pnl)), 4),
    }


def trimmed_mean(arr, trim=0.1):
    """Trimmed mean dropping top/bottom trim fraction."""
    a = np.sort(arr)
    n = len(a)
    lo = int(n * trim)
    hi = n - lo
    if hi <= lo:
        return float(np.mean(a))
    return float(np.mean(a[lo:hi]))


def compute_correlation(feature_vals, pnl_vals):
    """Pearson and Spearman correlation with p-values."""
    mask = np.isfinite(feature_vals) & np.isfinite(pnl_vals)
    f = feature_vals[mask]
    p = pnl_vals[mask]
    if len(f) < 10:
        return {'pearson_r': None, 'pearson_p': None,
                'spearman_r': None, 'spearman_p': None, 'n': len(f)}
    pr, pp = stats.pearsonr(f, p)
    sr, sp = stats.spearmanr(f, p)
    return {
        'pearson_r': round(float(pr), 4),
        'pearson_p': round(float(pp), 4),
        'spearman_r': round(float(sr), 4),
        'spearman_p': round(float(sp), 4),
        'n': int(len(f)),
    }


def quartile_analysis(feature_vals, pnl_vals, feature_name='feature'):
    """Split into quartiles by feature, report PnL metrics per quartile."""
    mask = np.isfinite(feature_vals) & np.isfinite(pnl_vals)
    f = feature_vals[mask]
    p = pnl_vals[mask]
    if len(f) < 20:
        return {'note': 'insufficient data', 'n': len(f)}

    try:
        # First try without labels, then assign labels based on actual number of bins
        quartiles_raw = pd.qcut(f, 4, duplicates='drop')
        n_bins = len(quartiles_raw.cat.categories)
        labels = [f'Q{i+1}' for i in range(n_bins)]
        quartiles = pd.qcut(f, 4, labels=labels, duplicates='drop')
    except Exception:
        return {'note': 'could not create quartiles', 'n': len(f)}

    results = {}
    q_avgs = []
    quartile_arr = np.array(quartiles)
    for q in sorted(quartiles.unique()):
        qmask = quartile_arr == q
        qpnl = p[qmask]
        results[str(q)] = compute_metrics(qpnl)
        if results[str(q)].get('avg_pnl') is not None:
            q_avgs.append(results[str(q)]['avg_pnl'])

    # Check monotonicity
    if len(q_avgs) >= 3:
        diffs = np.diff(q_avgs)
        mono_up = all(d > 0 for d in diffs)
        mono_down = all(d < 0 for d in diffs)
        results['monotonic'] = 'increasing' if mono_up else ('decreasing' if mono_down else 'non-monotonic')
    else:
        results['monotonic'] = 'insufficient_quartiles'

    return results


def threshold_sweep(feature_vals, pnl_vals, thresholds, above_is_pass=True):
    """Test multiple thresholds, splitting PASS/FAIL."""
    mask = np.isfinite(feature_vals) & np.isfinite(pnl_vals)
    f = feature_vals[mask]
    p = pnl_vals[mask]
    results = {}
    for t in thresholds:
        if above_is_pass:
            pass_mask = f >= t
        else:
            pass_mask = f <= t
        pass_pnl = p[pass_mask]
        fail_pnl = p[~pass_mask]
        results[str(round(t, 4))] = {
            'pass': compute_metrics(pass_pnl),
            'fail': compute_metrics(fail_pnl),
        }
    return results


# ── Strategy Assignment ──────────────────────────────────────────────────────

def assign_strategies(row):
    """Return list of strategy names this trade belongs to."""
    strats = []
    d = row['direction']
    agp = row['abs_gap_pct']
    gp = row['gap_pct']
    is_llm = row.get('scanner', '') == 'premarket'  # LLM scanner
    is_earn = row.get('is_earnings', False)
    cat = str(row.get('catalyst_type', '')).lower()
    is_deal = 'deal' in cat or 'merger' in cat or 'acquisition' in cat
    is_macro = 'macro' in cat
    not_excluded = not is_earn and not is_deal and not is_macro
    orb_confirms = row.get('first5_confirms', False)

    # BASELINE_V1: SHORT, abs_gap >= 3, LLM scanner, not earnings/deal/macro
    if d == 'SHORT' and agp >= 3.0 and is_llm and not_excluded and gp > 0:
        strats.append('BASELINE_V1')

    # B1_WIDE_SHORT: SHORT (or could be), abs_gap >= 1.5
    if d == 'SHORT' and agp >= 1.5 and gp > 0:
        strats.append('B1_WIDE_SHORT')

    # B2_NO_ORB: SHORT, abs_gap >= 3, LLM
    if d == 'SHORT' and agp >= 3.0 and is_llm and gp > 0:
        strats.append('B2_NO_ORB')

    # B3_EOD_ENTRY: same as V1 filters
    if d == 'SHORT' and agp >= 3.0 and is_llm and not_excluded and gp > 0:
        strats.append('B3_EOD_ENTRY')

    # B4_T1_ENTRY: same as V1 filters
    if d == 'SHORT' and agp >= 3.0 and is_llm and not_excluded and gp > 0:
        strats.append('B4_T1_ENTRY')

    # C1_LONG_FADE: LONG, abs_gap >= 3, gap_pct < 0 (gap-down fade)
    if d == 'LONG' and agp >= 3.0 and gp < 0:
        strats.append('C1_LONG_FADE')

    # C2_LONG_ORB: LONG, abs_gap >= 3, ORB confirms
    if d == 'LONG' and agp >= 3.0 and gp < 0 and orb_confirms:
        strats.append('C2_LONG_ORB')

    # C3_EARNINGS: LONG, is_earnings, abs_gap >= 3
    if d == 'LONG' and is_earn and agp >= 3.0:
        strats.append('C3_EARNINGS')

    # D1_LONG_MOM: LONG, gap_pct > 5 (momentum on gap-up)
    if d == 'LONG' and gp > 5.0:
        strats.append('D1_LONG_MOM')

    # D2_SHORT_MOM: SHORT, abs_gap >= 5, gap_pct > 0
    if d == 'SHORT' and agp >= 5.0 and gp > 0:
        strats.append('D2_SHORT_MOM')

    return strats


# Assign strategies to each row
df['strategies'] = df.apply(assign_strategies, axis=1)

# Which PnL column to use per strategy
FADE_STRATEGIES = {'BASELINE_V1', 'B1_WIDE_SHORT', 'B2_NO_ORB', 'B3_EOD_ENTRY',
                   'B4_T1_ENTRY', 'C1_LONG_FADE', 'C2_LONG_ORB', 'C3_EARNINGS'}
MOM_STRATEGIES = {'D1_LONG_MOM', 'D2_SHORT_MOM'}
ALL_STRATEGIES = list(FADE_STRATEGIES | MOM_STRATEGIES)


def cross_strategy_consistency(df_with_feature, feature_col, direction='negative'):
    """
    Check if feature shows consistent sign across strategies.
    direction: 'negative' means higher feature -> lower PnL is the hypothesis,
               'positive' means higher feature -> higher PnL.
    Returns (consistent_count, total_tested, details).
    """
    consistent = 0
    total = 0
    details = {}

    for strat in ALL_STRATEGIES:
        # Get trades in this strategy
        mask = df_with_feature['strategies'].apply(lambda s: strat in s)
        sub = df_with_feature[mask].copy()

        # Pick PnL column
        pnl_col = 'pnl_t2_close' if strat in FADE_STRATEGIES else 'pnl_60m_nosl'
        valid = sub[[feature_col, pnl_col]].dropna()

        if len(valid) < 8:
            details[strat] = {'n': len(valid), 'note': 'insufficient', 'consistent': None}
            continue

        total += 1
        sr, sp = stats.spearmanr(valid[feature_col], valid[pnl_col])
        is_consistent = (sr < 0) if direction == 'negative' else (sr > 0)
        if is_consistent:
            consistent += 1

        details[strat] = {
            'n': int(len(valid)),
            'spearman_r': round(float(sr), 4),
            'spearman_p': round(float(sp), 4),
            'consistent': bool(is_consistent),
        }

    return consistent, total, details


# ── Helper: get minute bar for a ticker-date ─────────────────────────────────

def get_bar(ticker, date, time_str):
    """Get a single minute bar from v3_minute_bars."""
    key = f"{ticker}__{date}"
    if key not in ticker_bars:
        return None
    bars = ticker_bars[key]
    if time_str in bars:
        return bars[time_str]
    return None


def get_bars_range(ticker, date, start_time, end_time):
    """Get bars in [start_time, end_time] range as sorted list of (time, bar) tuples."""
    key = f"{ticker}__{date}"
    if key not in ticker_bars:
        return []
    bars = ticker_bars[key]
    result = []
    for t_str, bar in bars.items():
        if start_time <= t_str <= end_time:
            result.append((t_str, bar))
    return sorted(result, key=lambda x: x[0])


def time_to_minutes(t_str):
    """Convert HH:MM to minutes since midnight."""
    parts = t_str.split(':')
    return int(parts[0]) * 60 + int(parts[1])


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE F01: Gap Magnitude Bucket
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("F01: Gap Magnitude Bucket")
print("="*70)

def bucket_gap(agp):
    if agp < 2:
        return 'micro_lt2'
    elif agp < 3:
        return 'small_2_3'
    elif agp < 5:
        return 'medium_3_5'
    elif agp < 8:
        return 'large_5_8'
    else:
        return 'extreme_gt8'

df['f01_gap_bucket'] = df['abs_gap_pct'].apply(bucket_gap)
# Numeric version for correlation
df['f01_abs_gap'] = df['abs_gap_pct']

# --- Correlation ---
f01_corr_t2 = compute_correlation(df['f01_abs_gap'].values, df['pnl_t2_close'].values)
f01_corr_60m = compute_correlation(df['f01_abs_gap'].values, df['pnl_60m_nosl'].values)
print(f"  Corr vs pnl_t2_close: r={f01_corr_t2['pearson_r']}, p={f01_corr_t2['pearson_p']}")
print(f"  Corr vs pnl_60m_nosl: r={f01_corr_60m['pearson_r']}, p={f01_corr_60m['pearson_p']}")

# --- Bucket analysis ---
bucket_order = ['micro_lt2', 'small_2_3', 'medium_3_5', 'large_5_8', 'extreme_gt8']
f01_bucket_results = {}
for b in bucket_order:
    sub = df[df['f01_gap_bucket'] == b]
    pnl_t2 = sub['pnl_t2_close'].dropna().values
    pnl_60m = sub['pnl_60m_nosl'].dropna().values
    f01_bucket_results[b] = {
        'pnl_t2_close': compute_metrics(pnl_t2),
        'pnl_60m_nosl': compute_metrics(pnl_60m),
    }
    avg_t2 = f01_bucket_results[b]['pnl_t2_close'].get('avg_pnl', 'N/A')
    avg_60 = f01_bucket_results[b]['pnl_60m_nosl'].get('avg_pnl', 'N/A')
    n_t2 = f01_bucket_results[b]['pnl_t2_close'].get('n', 0)
    n_60 = f01_bucket_results[b]['pnl_60m_nosl'].get('n', 0)
    print(f"  {b:15s}: T+2 avg={avg_t2} (n={n_t2}), 60m avg={avg_60} (n={n_60})")

# --- Quartile analysis ---
f01_quartile = quartile_analysis(df['f01_abs_gap'].values, df['pnl_t2_close'].values)

# --- Threshold sweep ---
f01_thresholds = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
f01_sweep = threshold_sweep(df['f01_abs_gap'].values, df['pnl_t2_close'].values,
                            f01_thresholds, above_is_pass=True)

# --- Cross-strategy consistency ---
# Higher gap -> potentially higher PnL for fade strategies (more to revert)
f01_cons, f01_total, f01_cons_detail = cross_strategy_consistency(
    df, 'f01_abs_gap', direction='positive')
print(f"  Cross-strategy consistency: {f01_cons}/{f01_total}")

# Identify best bucket
best_bucket = None
best_ci_low = -999
for b in bucket_order:
    m = f01_bucket_results[b]['pnl_t2_close']
    if m.get('ci_95_low') is not None and m['ci_95_low'] > best_ci_low:
        best_ci_low = m['ci_95_low']
        best_bucket = b

# Determine pass/fail
f01_max_corr = max(abs(f01_corr_t2.get('pearson_r', 0) or 0),
                   abs(f01_corr_60m.get('pearson_r', 0) or 0))
f01_min_p = min(f01_corr_t2.get('pearson_p', 1) or 1,
                f01_corr_60m.get('pearson_p', 1) or 1)
f01_pass = (f01_max_corr >= CORR_THRESHOLD and f01_min_p < P_THRESHOLD and
            f01_cons >= CONSISTENCY_THRESHOLD)

f01_result = {
    'feature': 'F01_Gap_Magnitude_Bucket',
    'description': 'Gap sizes bucketized: micro (<2%), small (2-3%), medium (3-5%), large (5-8%), extreme (>8%)',
    'hypothesis': 'Larger gaps provide more reversion potential, higher avg PnL for fade strategies',
    'correlation': {'vs_pnl_t2_close': f01_corr_t2, 'vs_pnl_60m_nosl': f01_corr_60m},
    'bucket_analysis': f01_bucket_results,
    'best_bucket': {'bucket': best_bucket, 'ci_95_low': best_ci_low},
    'quartile_analysis': f01_quartile,
    'threshold_sweep': f01_sweep,
    'cross_strategy_consistency': {
        'score': f01_cons, 'total_tested': f01_total, 'details': f01_cons_detail
    },
    'conclusion': {
        'passes': f01_pass,
        'max_abs_corr': round(f01_max_corr, 4),
        'min_p_value': round(f01_min_p, 4),
        'consistency_score': f'{f01_cons}/{f01_total}',
        'summary': f"Gap magnitude {'PASSES' if f01_pass else 'FAILS'} feature gate. "
                   f"Best bucket: {best_bucket} (ci_low={best_ci_low:.4f})."
    },
    'validation_required': True,
    'live_trading_weeks_needed': 4,
}

with open(os.path.join(FEATURES_DIR, 'F01_gap_magnitude_results.json'), 'w') as f:
    json.dump(f01_result, f, indent=2, default=str)
print(f"  PASS: {f01_pass}")


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE F02: Opening Bar Volatility
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("F02: Opening Bar Volatility")
print("="*70)

f02_vals = []
for _, row in df.iterrows():
    bar = get_bar(row['ticker'], row['eval_date'], '09:30')
    if bar and bar['o'] > 0:
        ob_vol = (bar['h'] - bar['l']) / bar['o']
        f02_vals.append(ob_vol)
    else:
        f02_vals.append(np.nan)

df['f02_ob_volatility'] = f02_vals
n_valid = df['f02_ob_volatility'].notna().sum()
print(f"  Computed for {n_valid}/{len(df)} trades")

# --- Correlation ---
f02_corr_t2 = compute_correlation(df['f02_ob_volatility'].values, df['pnl_t2_close'].values)
f02_corr_60m = compute_correlation(df['f02_ob_volatility'].values, df['pnl_60m_nosl'].values)
print(f"  Corr vs pnl_t2_close: r={f02_corr_t2['pearson_r']}, p={f02_corr_t2['pearson_p']}")
print(f"  Corr vs pnl_60m_nosl: r={f02_corr_60m['pearson_r']}, p={f02_corr_60m['pearson_p']}")

# --- Quartile analysis ---
f02_quartile = quartile_analysis(df['f02_ob_volatility'].values, df['pnl_t2_close'].values)

# --- Threshold sweep ---
f02_pctiles = df['f02_ob_volatility'].dropna().quantile([0.25, 0.5, 0.75]).values.tolist()
f02_sweep_thresholds = sorted(set([round(x, 4) for x in f02_pctiles] +
                                  [0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05]))
f02_sweep = threshold_sweep(df['f02_ob_volatility'].values, df['pnl_t2_close'].values,
                            f02_sweep_thresholds, above_is_pass=False)

# Test excluding top quartile
q75 = df['f02_ob_volatility'].quantile(0.75)
excl_mask = df['f02_ob_volatility'] <= q75
excl_pnl = df.loc[excl_mask, 'pnl_t2_close'].dropna().values
incl_pnl = df.loc[~excl_mask & df['f02_ob_volatility'].notna(), 'pnl_t2_close'].dropna().values
f02_excl_top_q = {
    'included_bottom_75pct': compute_metrics(excl_pnl),
    'excluded_top_25pct': compute_metrics(incl_pnl),
    'q75_threshold': round(float(q75), 4),
}
print(f"  Excl top Q: bottom 75% avg={f02_excl_top_q['included_bottom_75pct'].get('avg_pnl', 'N/A')}, "
      f"top 25% avg={f02_excl_top_q['excluded_top_25pct'].get('avg_pnl', 'N/A')}")

# --- Cross-strategy consistency ---
# High opening vol might hurt PnL (more noise) -> negative
f02_cons, f02_total, f02_cons_detail = cross_strategy_consistency(
    df, 'f02_ob_volatility', direction='negative')
print(f"  Cross-strategy consistency: {f02_cons}/{f02_total}")

f02_max_corr = max(abs(f02_corr_t2.get('pearson_r', 0) or 0),
                   abs(f02_corr_60m.get('pearson_r', 0) or 0))
f02_min_p = min(f02_corr_t2.get('pearson_p', 1) or 1,
                f02_corr_60m.get('pearson_p', 1) or 1)
f02_pass = (f02_max_corr >= CORR_THRESHOLD and f02_min_p < P_THRESHOLD and
            f02_cons >= CONSISTENCY_THRESHOLD)

f02_result = {
    'feature': 'F02_Opening_Bar_Volatility',
    'description': 'First regular-session bar (09:30) range: (high - low) / open',
    'hypothesis': 'High opening bar volatility indicates chaotic price action, hurting fade PnL',
    'correlation': {'vs_pnl_t2_close': f02_corr_t2, 'vs_pnl_60m_nosl': f02_corr_60m},
    'quartile_analysis': f02_quartile,
    'threshold_sweep': f02_sweep,
    'exclude_top_quartile': f02_excl_top_q,
    'cross_strategy_consistency': {
        'score': f02_cons, 'total_tested': f02_total, 'details': f02_cons_detail
    },
    'conclusion': {
        'passes': f02_pass,
        'max_abs_corr': round(f02_max_corr, 4),
        'min_p_value': round(f02_min_p, 4),
        'consistency_score': f'{f02_cons}/{f02_total}',
        'summary': f"Opening bar volatility {'PASSES' if f02_pass else 'FAILS'} feature gate."
    },
    'validation_required': True,
    'live_trading_weeks_needed': 4,
}

with open(os.path.join(FEATURES_DIR, 'F02_opening_bar_volatility_results.json'), 'w') as f:
    json.dump(f02_result, f, indent=2, default=str)
print(f"  PASS: {f02_pass}")


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE F03: Gap Retrace at Entry
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("F03: Gap Retrace at Entry")
print("="*70)

f03_vals = []
for _, row in df.iterrows():
    bar_930 = get_bar(row['ticker'], row['eval_date'], '09:30')
    gp = row['gap_pct']
    ep = row['entry_price']

    if bar_930 is None or gp == 0 or not np.isfinite(ep):
        f03_vals.append(np.nan)
        continue

    open_price = bar_930['o']
    prev_close = open_price / (1 + gp / 100.0)
    gap_size = open_price - prev_close

    if abs(gap_size) < 0.01:
        f03_vals.append(np.nan)
        continue

    retrace = (open_price - ep) / gap_size
    # Retrace > 0 means price moved back toward prev_close
    # Retrace ~ 0 means entry near open, retrace ~ 1 means full retrace
    f03_vals.append(retrace)

df['f03_gap_retrace'] = f03_vals
n_valid = df['f03_gap_retrace'].notna().sum()
print(f"  Computed for {n_valid}/{len(df)} trades")
print(f"  Retrace stats: mean={df['f03_gap_retrace'].mean():.4f}, "
      f"median={df['f03_gap_retrace'].median():.4f}, "
      f"std={df['f03_gap_retrace'].std():.4f}")

# --- Correlation ---
f03_corr_t2 = compute_correlation(df['f03_gap_retrace'].values, df['pnl_t2_close'].values)
f03_corr_60m = compute_correlation(df['f03_gap_retrace'].values, df['pnl_60m_nosl'].values)
print(f"  Corr vs pnl_t2_close: r={f03_corr_t2['pearson_r']}, p={f03_corr_t2['pearson_p']}")
print(f"  Corr vs pnl_60m_nosl: r={f03_corr_60m['pearson_r']}, p={f03_corr_60m['pearson_p']}")

# --- Quartile analysis ---
f03_quartile = quartile_analysis(df['f03_gap_retrace'].values, df['pnl_t2_close'].values)

# --- Threshold sweep ---
f03_sweep = threshold_sweep(df['f03_gap_retrace'].values, df['pnl_t2_close'].values,
                            [-0.5, -0.2, 0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
                            above_is_pass=True)

# --- Cross-strategy consistency ---
# More retrace before entry might mean better entry -> positive correlation
f03_cons, f03_total, f03_cons_detail = cross_strategy_consistency(
    df, 'f03_gap_retrace', direction='positive')
print(f"  Cross-strategy consistency: {f03_cons}/{f03_total}")

f03_max_corr = max(abs(f03_corr_t2.get('pearson_r', 0) or 0),
                   abs(f03_corr_60m.get('pearson_r', 0) or 0))
f03_min_p = min(f03_corr_t2.get('pearson_p', 1) or 1,
                f03_corr_60m.get('pearson_p', 1) or 1)
f03_pass = (f03_max_corr >= CORR_THRESHOLD and f03_min_p < P_THRESHOLD and
            f03_cons >= CONSISTENCY_THRESHOLD)

f03_result = {
    'feature': 'F03_Gap_Retrace_at_Entry',
    'description': 'How much of the gap retraced by entry time: (open - entry) / (open - prev_close)',
    'hypothesis': 'More retrace before entry gives better entry price for fade trades',
    'correlation': {'vs_pnl_t2_close': f03_corr_t2, 'vs_pnl_60m_nosl': f03_corr_60m},
    'quartile_analysis': f03_quartile,
    'threshold_sweep': f03_sweep,
    'cross_strategy_consistency': {
        'score': f03_cons, 'total_tested': f03_total, 'details': f03_cons_detail
    },
    'conclusion': {
        'passes': f03_pass,
        'max_abs_corr': round(f03_max_corr, 4),
        'min_p_value': round(f03_min_p, 4),
        'consistency_score': f'{f03_cons}/{f03_total}',
        'summary': f"Gap retrace {'PASSES' if f03_pass else 'FAILS'} feature gate."
    },
    'validation_required': True,
    'live_trading_weeks_needed': 4,
}

with open(os.path.join(FEATURES_DIR, 'F03_gap_retrace_results.json'), 'w') as f:
    json.dump(f03_result, f, indent=2, default=str)
print(f"  PASS: {f03_pass}")


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE F04: Entry Delay in Minutes
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("F04: Entry Delay in Minutes")
print("="*70)

# Use entry_time_et to compute delay from 09:30
MARKET_OPEN_MINS = 9 * 60 + 30  # 570

f04_vals = []
for _, row in df.iterrows():
    et = str(row['entry_time_et']).strip()
    try:
        entry_mins = time_to_minutes(et)
        delay = entry_mins - MARKET_OPEN_MINS
        if delay < 0:
            delay = 0  # pre-market entries treated as 0 delay
        f04_vals.append(delay)
    except Exception:
        f04_vals.append(np.nan)

df['f04_entry_delay'] = f04_vals
n_valid = df['f04_entry_delay'].notna().sum()
print(f"  Computed for {n_valid}/{len(df)} trades")
print(f"  Delay stats: mean={df['f04_entry_delay'].mean():.1f} min, "
      f"median={df['f04_entry_delay'].median():.1f} min, "
      f"max={df['f04_entry_delay'].max():.0f} min")

# Also compute 3-bar peak detection entry from minute bars (synthetic F04b)
f04b_vals = []
for _, row in df.iterrows():
    bars_list = get_bars_range(row['ticker'], row['eval_date'], '09:30', '10:00')
    if len(bars_list) < 7:
        f04b_vals.append(np.nan)
        continue

    # For SHORT: look for local high (3-bar window, confirmation at i+3)
    # For LONG: look for local low
    is_short = row['direction'] == 'SHORT'
    found_delay = np.nan

    for i in range(3, min(11, len(bars_list))):
        # Check 3-bar window: bars[i-2], bars[i-1], bars[i]
        window_prices = []
        for j in range(max(0, i-2), i+1):
            if is_short:
                window_prices.append(bars_list[j][1]['h'])
            else:
                window_prices.append(bars_list[j][1]['l'])

        if len(window_prices) < 3:
            continue

        if is_short:
            peak_idx = np.argmax(window_prices)
        else:
            peak_idx = np.argmin(window_prices)

        # Peak is at middle of window
        if peak_idx == 1:
            # Confirm at i+3
            if i + 3 < len(bars_list):
                confirm_price = bars_list[i+3][1]['h'] if is_short else bars_list[i+3][1]['l']
                peak_price = window_prices[1]
                if (is_short and confirm_price < peak_price) or \
                   (not is_short and confirm_price > peak_price):
                    peak_time = bars_list[i-1][0]
                    found_delay = time_to_minutes(peak_time) - MARKET_OPEN_MINS
                    break

    f04b_vals.append(found_delay)

df['f04b_peak_delay'] = f04b_vals
n_peak = df['f04b_peak_delay'].notna().sum()
print(f"  Peak detection delay computed for {n_peak}/{len(df)} trades")

# --- Correlation (using actual entry delay) ---
f04_corr_t2 = compute_correlation(df['f04_entry_delay'].values, df['pnl_t2_close'].values)
f04_corr_60m = compute_correlation(df['f04_entry_delay'].values, df['pnl_60m_nosl'].values)
print(f"  Entry delay corr vs pnl_t2_close: r={f04_corr_t2['pearson_r']}, p={f04_corr_t2['pearson_p']}")
print(f"  Entry delay corr vs pnl_60m_nosl: r={f04_corr_60m['pearson_r']}, p={f04_corr_60m['pearson_p']}")

# Also test peak-detection delay
f04b_corr_t2 = compute_correlation(df['f04b_peak_delay'].values, df['pnl_t2_close'].values)
f04b_corr_60m = compute_correlation(df['f04b_peak_delay'].values, df['pnl_60m_nosl'].values)
print(f"  Peak delay corr vs pnl_t2_close: r={f04b_corr_t2['pearson_r']}, p={f04b_corr_t2['pearson_p']}")

# --- Quartile analysis ---
f04_quartile = quartile_analysis(df['f04_entry_delay'].values, df['pnl_t2_close'].values)

# --- Threshold sweep ---
f04_sweep = threshold_sweep(df['f04_entry_delay'].values, df['pnl_t2_close'].values,
                            [0, 2, 5, 10, 15, 30, 60, 120],
                            above_is_pass=True)

# --- Cross-strategy consistency ---
f04_cons, f04_total, f04_cons_detail = cross_strategy_consistency(
    df, 'f04_entry_delay', direction='negative')
print(f"  Cross-strategy consistency: {f04_cons}/{f04_total}")

f04_max_corr = max(abs(f04_corr_t2.get('pearson_r', 0) or 0),
                   abs(f04_corr_60m.get('pearson_r', 0) or 0))
f04_min_p = min(f04_corr_t2.get('pearson_p', 1) or 1,
                f04_corr_60m.get('pearson_p', 1) or 1)
f04_pass = (f04_max_corr >= CORR_THRESHOLD and f04_min_p < P_THRESHOLD and
            f04_cons >= CONSISTENCY_THRESHOLD)

f04_result = {
    'feature': 'F04_Entry_Delay_Minutes',
    'description': 'Minutes from 09:30 to actual entry time, plus peak-detection variant',
    'hypothesis': 'Later entries (more delay) may reduce edge as gap-fill window shrinks',
    'correlation': {
        'entry_delay_vs_pnl_t2_close': f04_corr_t2,
        'entry_delay_vs_pnl_60m_nosl': f04_corr_60m,
        'peak_delay_vs_pnl_t2_close': f04b_corr_t2,
        'peak_delay_vs_pnl_60m_nosl': f04b_corr_60m,
    },
    'quartile_analysis': f04_quartile,
    'threshold_sweep': f04_sweep,
    'peak_detection_stats': {
        'n_detected': int(n_peak),
        'mean_delay': round(float(df['f04b_peak_delay'].mean()), 1) if n_peak > 0 else None,
    },
    'cross_strategy_consistency': {
        'score': f04_cons, 'total_tested': f04_total, 'details': f04_cons_detail
    },
    'conclusion': {
        'passes': f04_pass,
        'max_abs_corr': round(f04_max_corr, 4),
        'min_p_value': round(f04_min_p, 4),
        'consistency_score': f'{f04_cons}/{f04_total}',
        'summary': f"Entry delay {'PASSES' if f04_pass else 'FAILS'} feature gate."
    },
    'validation_required': True,
    'live_trading_weeks_needed': 4,
}

with open(os.path.join(FEATURES_DIR, 'F04_entry_delay_results.json'), 'w') as f:
    json.dump(f04_result, f, indent=2, default=str)
print(f"  PASS: {f04_pass}")


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE F05: Intraday Volatility at Entry
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("F05: Intraday Volatility at Entry")
print("="*70)

f05_vals = []
for _, row in df.iterrows():
    et_str = str(row['entry_time_et']).strip()
    entry_mins = time_to_minutes(et_str)
    # Get 5 bars leading up to entry
    # Compute times for 5 bars before entry
    bars_needed = []
    for offset in range(5, 0, -1):
        bar_min = entry_mins - offset
        h = bar_min // 60
        m = bar_min % 60
        bars_needed.append(f"{h:02d}:{m:02d}")

    key = f"{row['ticker']}__{row['eval_date']}"
    if key not in ticker_bars:
        f05_vals.append(np.nan)
        continue

    bars = ticker_bars[key]
    closes = []
    for t in bars_needed:
        if t in bars:
            closes.append(bars[t]['c'])

    if len(closes) < 3:
        f05_vals.append(np.nan)
        continue

    # Compute 1-min returns
    closes = np.array(closes)
    returns = np.diff(closes) / closes[:-1]
    vol = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    f05_vals.append(vol)

df['f05_entry_volatility'] = f05_vals
n_valid = df['f05_entry_volatility'].notna().sum()
print(f"  Computed for {n_valid}/{len(df)} trades")
if n_valid > 0:
    print(f"  Vol stats: mean={df['f05_entry_volatility'].mean():.6f}, "
          f"median={df['f05_entry_volatility'].median():.6f}")

# --- Correlation ---
f05_corr_t2 = compute_correlation(df['f05_entry_volatility'].values, df['pnl_t2_close'].values)
f05_corr_60m = compute_correlation(df['f05_entry_volatility'].values, df['pnl_60m_nosl'].values)
print(f"  Corr vs pnl_t2_close: r={f05_corr_t2['pearson_r']}, p={f05_corr_t2['pearson_p']}")
print(f"  Corr vs pnl_60m_nosl: r={f05_corr_60m['pearson_r']}, p={f05_corr_60m['pearson_p']}")

# --- Quartile analysis ---
f05_quartile = quartile_analysis(df['f05_entry_volatility'].values, df['pnl_t2_close'].values)

# --- Threshold sweep ---
if n_valid > 10:
    f05_pctiles = df['f05_entry_volatility'].dropna().quantile([0.25, 0.5, 0.75]).values.tolist()
    f05_sweep_thresholds = sorted(set([round(x, 6) for x in f05_pctiles] +
                                      [0.001, 0.002, 0.005, 0.01, 0.02]))
    f05_sweep = threshold_sweep(df['f05_entry_volatility'].values, df['pnl_t2_close'].values,
                                f05_sweep_thresholds, above_is_pass=False)
else:
    f05_sweep = {'note': 'insufficient data'}

# --- Cross-strategy consistency ---
f05_cons, f05_total, f05_cons_detail = cross_strategy_consistency(
    df, 'f05_entry_volatility', direction='negative')
print(f"  Cross-strategy consistency: {f05_cons}/{f05_total}")

f05_max_corr = max(abs(f05_corr_t2.get('pearson_r', 0) or 0),
                   abs(f05_corr_60m.get('pearson_r', 0) or 0))
f05_min_p = min(f05_corr_t2.get('pearson_p', 1) or 1,
                f05_corr_60m.get('pearson_p', 1) or 1)
f05_pass = (f05_max_corr >= CORR_THRESHOLD and f05_min_p < P_THRESHOLD and
            f05_cons >= CONSISTENCY_THRESHOLD)

f05_result = {
    'feature': 'F05_Intraday_Volatility_at_Entry',
    'description': 'Rolling std of 1-min returns from 5 bars before entry',
    'hypothesis': 'High micro-volatility at entry indicates unstable price action, hurting fills and PnL',
    'correlation': {'vs_pnl_t2_close': f05_corr_t2, 'vs_pnl_60m_nosl': f05_corr_60m},
    'quartile_analysis': f05_quartile,
    'threshold_sweep': f05_sweep,
    'cross_strategy_consistency': {
        'score': f05_cons, 'total_tested': f05_total, 'details': f05_cons_detail
    },
    'conclusion': {
        'passes': f05_pass,
        'max_abs_corr': round(f05_max_corr, 4),
        'min_p_value': round(f05_min_p, 4),
        'consistency_score': f'{f05_cons}/{f05_total}',
        'summary': f"Intraday volatility at entry {'PASSES' if f05_pass else 'FAILS'} feature gate."
    },
    'validation_required': True,
    'live_trading_weeks_needed': 4,
}

with open(os.path.join(FEATURES_DIR, 'F05_intraday_volatility_results.json'), 'w') as f:
    json.dump(f05_result, f, indent=2, default=str)
print(f"  PASS: {f05_pass}")


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE F06: Day of Week Effect
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("F06: Day of Week Effect")
print("="*70)

df['f06_dow'] = pd.to_datetime(df['eval_date']).dt.dayofweek  # 0=Mon, 4=Fri
df['f06_dow_name'] = pd.to_datetime(df['eval_date']).dt.day_name()

# --- Per-day analysis ---
dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
f06_day_results = {}
for dow in sorted(df['f06_dow'].unique()):
    name = dow_names[dow]
    sub = df[df['f06_dow'] == dow]
    pnl_t2 = sub['pnl_t2_close'].dropna().values
    pnl_60m = sub['pnl_60m_nosl'].dropna().values
    f06_day_results[name] = {
        'pnl_t2_close': compute_metrics(pnl_t2),
        'pnl_60m_nosl': compute_metrics(pnl_60m),
        'trade_count': int(len(sub)),
    }
    avg_t2 = f06_day_results[name]['pnl_t2_close'].get('avg_pnl', 'N/A')
    avg_60m = f06_day_results[name]['pnl_60m_nosl'].get('avg_pnl', 'N/A')
    print(f"  {name:12s}: T+2 avg={avg_t2} (n={f06_day_results[name]['pnl_t2_close'].get('n', 0)}), "
          f"60m avg={avg_60m} (n={f06_day_results[name]['pnl_60m_nosl'].get('n', 0)})")

# --- Correlation ---
f06_corr_t2 = compute_correlation(df['f06_dow'].values.astype(float), df['pnl_t2_close'].values)
f06_corr_60m = compute_correlation(df['f06_dow'].values.astype(float), df['pnl_60m_nosl'].values)
print(f"  Corr vs pnl_t2_close: r={f06_corr_t2['pearson_r']}, p={f06_corr_t2['pearson_p']}")
print(f"  Corr vs pnl_60m_nosl: r={f06_corr_60m['pearson_r']}, p={f06_corr_60m['pearson_p']}")

# --- ANOVA test (better for categorical day variable) ---
groups_t2 = [df[df['f06_dow'] == d]['pnl_t2_close'].dropna().values
             for d in sorted(df['f06_dow'].unique())]
groups_t2 = [g for g in groups_t2 if len(g) >= 5]
if len(groups_t2) >= 2:
    f_stat, anova_p = stats.f_oneway(*groups_t2)
    f06_anova = {'f_stat': round(float(f_stat), 4), 'p_value': round(float(anova_p), 4)}
    print(f"  ANOVA F={f_stat:.4f}, p={anova_p:.4f}")
else:
    f06_anova = {'note': 'insufficient groups'}

groups_60m = [df[df['f06_dow'] == d]['pnl_60m_nosl'].dropna().values
              for d in sorted(df['f06_dow'].unique())]
groups_60m = [g for g in groups_60m if len(g) >= 5]
if len(groups_60m) >= 2:
    f_stat_60, anova_p_60 = stats.f_oneway(*groups_60m)
    f06_anova_60m = {'f_stat': round(float(f_stat_60), 4), 'p_value': round(float(anova_p_60), 4)}
else:
    f06_anova_60m = {'note': 'insufficient groups'}

# --- Cross-strategy consistency ---
# Day-of-week is categorical; test each day as binary filter
f06_cons_details = {}
consistent_count = 0
tested_count = 0
for dow in sorted(df['f06_dow'].unique()):
    name = dow_names[dow]
    df['_is_this_day'] = (df['f06_dow'] == dow).astype(float)
    cons, total, detail = cross_strategy_consistency(df, '_is_this_day', direction='positive')
    f06_cons_details[name] = detail
    # If any day shows consistent positive or negative effect across strategies, count it
    for strat, d in detail.items():
        if d.get('consistent') is not None:
            tested_count += 1
            if d['consistent']:
                consistent_count += 1

df.drop('_is_this_day', axis=1, inplace=True)

# For DOW, use ANOVA p-value as the significance test instead of correlation
f06_max_corr = max(abs(f06_corr_t2.get('pearson_r', 0) or 0),
                   abs(f06_corr_60m.get('pearson_r', 0) or 0))
f06_min_p = min(f06_corr_t2.get('pearson_p', 1) or 1,
                f06_corr_60m.get('pearson_p', 1) or 1)
# Also consider ANOVA p-value
if f06_anova.get('p_value') is not None:
    f06_min_p = min(f06_min_p, f06_anova['p_value'])
f06_cons_ratio = consistent_count / tested_count if tested_count > 0 else 0
f06_pass = (f06_max_corr >= CORR_THRESHOLD and f06_min_p < P_THRESHOLD and
            f06_cons_ratio >= 0.6)

f06_result = {
    'feature': 'F06_Day_of_Week',
    'description': 'Day of week effect on gap-trade PnL (Mon-Thu in sample)',
    'hypothesis': 'Certain days may systematically produce better/worse gap-trade outcomes',
    'correlation': {'vs_pnl_t2_close': f06_corr_t2, 'vs_pnl_60m_nosl': f06_corr_60m},
    'anova_test': {'pnl_t2_close': f06_anova, 'pnl_60m_nosl': f06_anova_60m},
    'day_analysis': f06_day_results,
    'cross_strategy_consistency': {
        'consistent_pct': round(f06_cons_ratio, 4),
        'details': f06_cons_details,
    },
    'conclusion': {
        'passes': f06_pass,
        'max_abs_corr': round(f06_max_corr, 4),
        'min_p_value': round(f06_min_p, 4),
        'consistency_ratio': round(f06_cons_ratio, 4),
        'summary': f"Day of week {'PASSES' if f06_pass else 'FAILS'} feature gate. "
                   f"Only 4 days in sample -- very limited statistical power.",
        'caveat': 'Only 4 trading days in sample; any finding is unreliable.',
    },
    'validation_required': True,
    'live_trading_weeks_needed': 4,
}

with open(os.path.join(FEATURES_DIR, 'F06_day_of_week_results.json'), 'w') as f:
    json.dump(f06_result, f, indent=2, default=str)
print(f"  PASS: {f06_pass}")


# ══════════════════════════════════════════════════════════════════════════════
# COMBINED SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("COMBINED SUMMARY")
print("="*70)

features_summary = []
for fname, fres in [('F01', f01_result), ('F02', f02_result), ('F03', f03_result),
                     ('F04', f04_result), ('F05', f05_result), ('F06', f06_result)]:
    c = fres['conclusion']
    features_summary.append({
        'feature': fres['feature'],
        'passes': c['passes'],
        'max_abs_corr': c['max_abs_corr'],
        'min_p_value': c['min_p_value'],
        'consistency': c.get('consistency_score', c.get('consistency_ratio', 'N/A')),
        'summary': c['summary'],
    })
    status = "PASS" if c['passes'] else "FAIL"
    print(f"  {fname} {fres['feature']:40s} [{status}] corr={c['max_abs_corr']:.4f} p={c['min_p_value']:.4f}")

# Rank by absolute correlation
features_summary.sort(key=lambda x: x['max_abs_corr'], reverse=True)

combined = {
    'investigation': 'Feature Investigation F01-F06',
    'date': '2026-09-13',
    'dataset': 'v2_features_full.csv (316 trades, 4 days)',
    'targets': ['pnl_t2_close (117 non-null)', 'pnl_60m_nosl (272 non-null)'],
    'pass_criteria': {
        'correlation': f'abs(r) >= {CORR_THRESHOLD}',
        'significance': f'p < {P_THRESHOLD}',
        'cross_strategy_consistency': f'>= {CONSISTENCY_THRESHOLD}/10',
    },
    'rankings': features_summary,
    'passing_features': [f for f in features_summary if f['passes']],
    'failing_features': [f for f in features_summary if not f['passes']],
    'caveat': 'Only 4 trading days and 117 T+2 PnL observations. '
              'All findings require live validation before deployment.',
}

with open(SUMMARY_PATH, 'w') as f:
    json.dump(combined, f, indent=2, default=str)
print(f"\nSummary saved to {SUMMARY_PATH}")


# ══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════════════════════
print("\nGenerating charts...")

fig, axes = plt.subplots(3, 2, figsize=(16, 18))
fig.suptitle('Feature Investigation F01-F06: Gap Trading System', fontsize=16, fontweight='bold')

# --- F01: Gap Magnitude ---
ax = axes[0, 0]
buckets = ['micro\n<2%', 'small\n2-3%', 'medium\n3-5%', 'large\n5-8%', 'extreme\n>8%']
bucket_keys = ['micro_lt2', 'small_2_3', 'medium_3_5', 'large_5_8', 'extreme_gt8']
avgs_t2 = []
ci_lows = []
ci_highs = []
ns = []
for bk in bucket_keys:
    m = f01_bucket_results[bk]['pnl_t2_close']
    avgs_t2.append(m.get('avg_pnl', 0) or 0)
    ci_lows.append(m.get('ci_95_low', 0) or 0)
    ci_highs.append(m.get('ci_95_high', 0) or 0)
    ns.append(m.get('n', 0))

colors = ['#4e79a7' if a >= 0 else '#e15759' for a in avgs_t2]
bars = ax.bar(range(len(buckets)), avgs_t2, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
# Error bars
for i in range(len(buckets)):
    ax.plot([i, i], [ci_lows[i], ci_highs[i]], color='black', linewidth=1.5)
    ax.text(i, avgs_t2[i] + 0.15 * (1 if avgs_t2[i] >= 0 else -1),
            f'n={ns[i]}', ha='center', va='bottom' if avgs_t2[i] >= 0 else 'top', fontsize=8)

ax.set_xticks(range(len(buckets)))
ax.set_xticklabels(buckets, fontsize=8)
ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_ylabel('Avg T+2 PnL (%)')
ax.set_title('F01: Gap Magnitude Bucket', fontweight='bold')

# --- F02: Opening Bar Volatility ---
ax = axes[0, 1]
valid_f02 = df[['f02_ob_volatility', 'pnl_t2_close']].dropna()
if len(valid_f02) > 5:
    ax.scatter(valid_f02['f02_ob_volatility'] * 100, valid_f02['pnl_t2_close'],
               alpha=0.5, s=20, c='#4e79a7', edgecolors='none')
    # Add regression line
    z = np.polyfit(valid_f02['f02_ob_volatility'], valid_f02['pnl_t2_close'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(valid_f02['f02_ob_volatility'].min(), valid_f02['f02_ob_volatility'].max(), 100)
    ax.plot(x_line * 100, p(x_line), color='#e15759', linewidth=2, linestyle='--',
            label=f'r={f02_corr_t2.get("pearson_r", 0):.3f}')
    ax.legend(fontsize=9)
ax.set_xlabel('Opening Bar Range (%)')
ax.set_ylabel('T+2 PnL (%)')
ax.set_title('F02: Opening Bar Volatility vs T+2 PnL', fontweight='bold')
ax.axhline(y=0, color='black', linewidth=0.5)

# --- F03: Gap Retrace ---
ax = axes[1, 0]
valid_f03 = df[['f03_gap_retrace', 'pnl_t2_close']].dropna()
# Clip extreme retrace values for visualization
clip_mask = (valid_f03['f03_gap_retrace'] > -2) & (valid_f03['f03_gap_retrace'] < 3)
plot_f03 = valid_f03[clip_mask]
if len(plot_f03) > 5:
    ax.scatter(plot_f03['f03_gap_retrace'] * 100, plot_f03['pnl_t2_close'],
               alpha=0.5, s=20, c='#59a14f', edgecolors='none')
    z = np.polyfit(plot_f03['f03_gap_retrace'], plot_f03['pnl_t2_close'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(plot_f03['f03_gap_retrace'].min(), plot_f03['f03_gap_retrace'].max(), 100)
    ax.plot(x_line * 100, p(x_line), color='#e15759', linewidth=2, linestyle='--',
            label=f'r={f03_corr_t2.get("pearson_r", 0):.3f}')
    ax.legend(fontsize=9)
ax.set_xlabel('Gap Retrace at Entry (%)')
ax.set_ylabel('T+2 PnL (%)')
ax.set_title('F03: Gap Retrace at Entry vs T+2 PnL', fontweight='bold')
ax.axhline(y=0, color='black', linewidth=0.5)

# --- F04: Entry Delay ---
ax = axes[1, 1]
valid_f04 = df[['f04_entry_delay', 'pnl_t2_close']].dropna()
if len(valid_f04) > 5:
    ax.scatter(valid_f04['f04_entry_delay'], valid_f04['pnl_t2_close'],
               alpha=0.5, s=20, c='#f28e2b', edgecolors='none')
    z = np.polyfit(valid_f04['f04_entry_delay'], valid_f04['pnl_t2_close'], 1)
    p_poly = np.poly1d(z)
    x_line = np.linspace(0, valid_f04['f04_entry_delay'].max(), 100)
    ax.plot(x_line, p_poly(x_line), color='#e15759', linewidth=2, linestyle='--',
            label=f'r={f04_corr_t2.get("pearson_r", 0):.3f}')
    ax.legend(fontsize=9)
ax.set_xlabel('Entry Delay (minutes from 09:30)')
ax.set_ylabel('T+2 PnL (%)')
ax.set_title('F04: Entry Delay vs T+2 PnL', fontweight='bold')
ax.axhline(y=0, color='black', linewidth=0.5)

# --- F05: Intraday Volatility ---
ax = axes[2, 0]
valid_f05 = df[['f05_entry_volatility', 'pnl_t2_close']].dropna()
if len(valid_f05) > 5:
    ax.scatter(valid_f05['f05_entry_volatility'] * 100, valid_f05['pnl_t2_close'],
               alpha=0.5, s=20, c='#76b7b2', edgecolors='none')
    z = np.polyfit(valid_f05['f05_entry_volatility'], valid_f05['pnl_t2_close'], 1)
    p_poly = np.poly1d(z)
    x_line = np.linspace(valid_f05['f05_entry_volatility'].min(),
                         valid_f05['f05_entry_volatility'].max(), 100)
    ax.plot(x_line * 100, p_poly(x_line), color='#e15759', linewidth=2, linestyle='--',
            label=f'r={f05_corr_t2.get("pearson_r", 0):.3f}')
    ax.legend(fontsize=9)
ax.set_xlabel('5-Bar Rolling Volatility (%)')
ax.set_ylabel('T+2 PnL (%)')
ax.set_title('F05: Intraday Volatility at Entry vs T+2 PnL', fontweight='bold')
ax.axhline(y=0, color='black', linewidth=0.5)

# --- F06: Day of Week ---
ax = axes[2, 1]
days_in_data = sorted(df['f06_dow'].unique())
day_labels = [dow_names[d] for d in days_in_data]
day_avgs_t2 = []
day_ci_lo = []
day_ci_hi = []
day_ns = []
for d in days_in_data:
    m = f06_day_results[dow_names[d]]['pnl_t2_close']
    day_avgs_t2.append(m.get('avg_pnl', 0) or 0)
    day_ci_lo.append(m.get('ci_95_low', 0) or 0)
    day_ci_hi.append(m.get('ci_95_high', 0) or 0)
    day_ns.append(m.get('n', 0))

colors = ['#4e79a7' if a >= 0 else '#e15759' for a in day_avgs_t2]
ax.bar(range(len(day_labels)), day_avgs_t2, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
for i in range(len(day_labels)):
    ax.plot([i, i], [day_ci_lo[i], day_ci_hi[i]], color='black', linewidth=1.5)
    ax.text(i, day_avgs_t2[i] + 0.15 * (1 if day_avgs_t2[i] >= 0 else -1),
            f'n={day_ns[i]}', ha='center', va='bottom' if day_avgs_t2[i] >= 0 else 'top', fontsize=8)

ax.set_xticks(range(len(day_labels)))
ax.set_xticklabels(day_labels, fontsize=9)
ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_ylabel('Avg T+2 PnL (%)')
ax.set_title('F06: Day of Week Effect', fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(CHART_PATH, dpi=150, bbox_inches='tight')
print(f"Chart saved to {CHART_PATH}")

# Print final summary table
print("\n" + "="*70)
print("FINAL RANKINGS (sorted by abs correlation)")
print("="*70)
print(f"{'Rank':<5} {'Feature':<40} {'Pass':<6} {'|r|':<8} {'p-value':<10} {'Consistency'}")
print("-" * 85)
for i, f in enumerate(features_summary):
    status = 'PASS' if f['passes'] else 'FAIL'
    print(f"{i+1:<5} {f['feature']:<40} {status:<6} {f['max_abs_corr']:<8.4f} "
          f"{f['min_p_value']:<10.4f} {f['consistency']}")

print(f"\nPassing features: {len([f for f in features_summary if f['passes']])}/6")
print(f"All results saved to {FEATURES_DIR}/")
print("DONE.")
