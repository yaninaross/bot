###############################################################################
# Feature Investigation F07-F10: PM Vol Ratio, ORB Width, Market Cap, Float
# Scanner optimization for gap-trading system
# Colab-friendly: config vars at top, no if __name__
###############################################################################

import pandas as pd
import numpy as np
import json
import os
import time
import requests
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ── Config ──────────────────────────────────────────────────────────────────
BASE_DIR = '/home/admin_yaninaso_altostrat_com/bot-RAG/sentiment/scanner_optimization'
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FEATURES_DIR = os.path.join(RESULTS_DIR, 'features')
CSV_PATH = os.path.join(RESULTS_DIR, 'v2_features_full.csv')
DAILY_DATA_PATH = os.path.join(RESULTS_DIR, 'v2_daily_data.json')
MINUTE_BARS_PATH = os.path.join(RESULTS_DIR, 'v3_minute_bars.json')
POLYGON_API_KEY = '2YBWWKHpvnM6OX5qVhd1MxhsRCorLkFs'
POLYGON_SLEEP = 0.05  # seconds between API calls
TICKER_CACHE_PATH = os.path.join(FEATURES_DIR, 'ticker_reference_cache.json')

os.makedirs(FEATURES_DIR, exist_ok=True)

# ── Shared utilities ────────────────────────────────────────────────────────

def compute_metrics(pnl_array):
    """Standard metrics for a PnL array."""
    pnl = np.array([x for x in pnl_array if x is not None and np.isfinite(x)])
    n = len(pnl)
    if n < 5:
        return {'n': n, 'note': 'insufficient'}
    avg = float(np.mean(pnl))
    std = float(np.std(pnl, ddof=1))
    se = std / np.sqrt(n)
    return {
        'n': n, 'avg_pnl': round(avg, 4), 'median_pnl': round(float(np.median(pnl)), 4),
        'std': round(std, 4), 'sharpe': round(avg / std if std > 0 else 0, 4),
        'win_rate': round(float((pnl > 0).mean()), 4),
        'ci_95_low': round(avg - 1.96 * se, 4), 'ci_95_high': round(avg + 1.96 * se, 4),
        'max_loss': round(float(np.min(pnl)), 4), 'max_gain': round(float(np.max(pnl)), 4),
    }


def assign_strategy(row):
    """Assign trades to strategies based on gap_pct."""
    gp = row['gap_pct']
    direction = row['direction']
    is_earn = row.get('is_earnings', False)

    strategies = []
    # Momentum
    if gp > 5.0 and direction == 'LONG':
        strategies.append('D1_momentum_long')
    if gp < -5.0 and direction == 'SHORT':
        strategies.append('D2_momentum_short')
    # SHORT fade (gap-up being shorted)
    if gp > 0 and direction == 'SHORT':
        if gp >= 3.0:
            strategies.extend(['V1_short_fade', 'B2_short_fade', 'B3_short_fade', 'B4_short_fade'])
        if gp >= 1.5:
            strategies.append('B1_short_fade')
    # LONG fade (gap-down being bought)
    if gp < 0 and direction == 'LONG':
        if gp <= -3.0:
            strategies.extend(['C1_long_fade', 'C2_long_fade'])
            if is_earn:
                strategies.append('C3_long_fade_earnings')
    return strategies


def get_pnl_for_strategy(row, strategy):
    """Return appropriate PnL column for the strategy."""
    if strategy.startswith('D1') or strategy.startswith('D2'):
        return row.get('pnl_60m_nosl')
    else:
        return row.get('pnl_t2_close')


def correlation_analysis(feature_vals, pnl_vals, feature_name):
    """Pearson and Spearman correlation."""
    mask = np.isfinite(feature_vals) & np.isfinite(pnl_vals)
    fv = feature_vals[mask]
    pv = pnl_vals[mask]
    if len(fv) < 10:
        return {'n': len(fv), 'note': 'insufficient_for_correlation'}
    pearson_r, pearson_p = stats.pearsonr(fv, pv)
    spearman_r, spearman_p = stats.spearmanr(fv, pv)
    return {
        'n': int(len(fv)),
        'pearson_r': round(float(pearson_r), 4),
        'pearson_p': round(float(pearson_p), 4),
        'spearman_r': round(float(spearman_r), 4),
        'spearman_p': round(float(spearman_p), 4),
    }


def quartile_analysis(df_sub, feature_col, pnl_col):
    """Split into quartiles, compute metrics for each."""
    valid = df_sub[[feature_col, pnl_col]].dropna()
    if len(valid) < 20:
        return {'note': f'insufficient data ({len(valid)} rows)'}
    try:
        valid['quartile'] = pd.qcut(valid[feature_col], 4, labels=['Q1_low', 'Q2', 'Q3', 'Q4_high'], duplicates='drop')
    except ValueError:
        return {'note': 'cannot form quartiles (too few unique values)'}
    result = {}
    for q in ['Q1_low', 'Q2', 'Q3', 'Q4_high']:
        subset = valid[valid['quartile'] == q][pnl_col].values
        result[q] = compute_metrics(subset)
    return result


def threshold_analysis(df_sub, feature_col, pnl_col, thresholds, direction='above'):
    """Test multiple thresholds: PASS (above/below) vs FAIL."""
    valid = df_sub[[feature_col, pnl_col]].dropna()
    results = {}
    for thresh in thresholds:
        if direction == 'above':
            mask_pass = valid[feature_col] >= thresh
        else:
            mask_pass = valid[feature_col] <= thresh
        pass_pnl = valid[mask_pass][pnl_col].values
        fail_pnl = valid[~mask_pass][pnl_col].values
        results[str(thresh)] = {
            'pass': compute_metrics(pass_pnl),
            'fail': compute_metrics(fail_pnl),
        }
    return results


def cross_strategy_test(df, feature_col, pnl_col_func, median_val=None):
    """Test feature across all strategies. Returns strategy-level results."""
    results = {}
    for _, row in df.iterrows():
        strats = assign_strategy(row)
        for s in strats:
            pnl = pnl_col_func(row, s)
            feat = row.get(feature_col)
            if pnl is None or not np.isfinite(pnl) or feat is None or not np.isfinite(feat):
                continue
            if s not in results:
                results[s] = {'high_feat': [], 'low_feat': []}
            if median_val is not None:
                if feat >= median_val:
                    results[s]['high_feat'].append(pnl)
                else:
                    results[s]['low_feat'].append(pnl)

    summary = {}
    for s, data in results.items():
        summary[s] = {
            'high_feat': compute_metrics(data['high_feat']),
            'low_feat': compute_metrics(data['low_feat']),
        }
    return summary


# ── Load data ───────────────────────────────────────────────────────────────

print("Loading data...")
df = pd.read_csv(CSV_PATH)
with open(DAILY_DATA_PATH) as f:
    daily_data = json.load(f)

print(f"  CSV: {len(df)} rows, {df['ticker'].nunique()} tickers")
print(f"  Daily data: {len(daily_data)} tickers")
print(f"  pnl_t2_close non-null: {df['pnl_t2_close'].notna().sum()}")
print(f"  pnl_60m_nosl non-null: {df['pnl_60m_nosl'].notna().sum()}")

###############################################################################
# F07: PM Volume Ratio (Continuous)
###############################################################################
print("\n" + "="*70)
print("F07: PM Volume Ratio (Continuous)")
print("="*70)

# Compute 20-day avg daily volume from v2_daily_data.json
avg_daily_vol = {}
for ticker, dates_dict in daily_data.items():
    vols = [d['volume'] for d in dates_dict.values() if d.get('volume') and d['volume'] > 0]
    if vols:
        avg_daily_vol[ticker] = np.mean(vols)

print(f"  Avg daily volume computed for {len(avg_daily_vol)} tickers")

# Compute pm_vol_ratio
df['avg_daily_vol'] = df['ticker'].map(avg_daily_vol)
# Use all available dates for the average (the daily data is ~29 days which is close enough to 20-day)
df['pm_vol_ratio'] = df['pm_vol'] / df['avg_daily_vol']

has_pmvr = df['pm_vol_ratio'].notna()
print(f"  pm_vol_ratio computed: {has_pmvr.sum()} rows")
print(f"  pm_vol_ratio range: {df.loc[has_pmvr, 'pm_vol_ratio'].min():.4f} to {df.loc[has_pmvr, 'pm_vol_ratio'].max():.4f}")
print(f"  pm_vol_ratio median: {df.loc[has_pmvr, 'pm_vol_ratio'].median():.4f}")

f07_results = {
    'feature': 'F07_pm_vol_ratio',
    'description': 'Premarket volume / avg daily volume',
    'validation_required': True,
    'live_trading_weeks_needed': 4,
}

# Correlation with pnl_t2_close
pnl_col = 'pnl_t2_close'
mask_f07 = df['pm_vol_ratio'].notna() & df[pnl_col].notna()
if mask_f07.sum() >= 10:
    f07_results['correlation_t2'] = correlation_analysis(
        df.loc[mask_f07, 'pm_vol_ratio'].values,
        df.loc[mask_f07, pnl_col].values,
        'pm_vol_ratio'
    )
    print(f"  Correlation (t2): Pearson r={f07_results['correlation_t2']['pearson_r']}, "
          f"p={f07_results['correlation_t2']['pearson_p']}")

# Also correlate with pnl_60m_nosl
mask_f07_60 = df['pm_vol_ratio'].notna() & df['pnl_60m_nosl'].notna()
if mask_f07_60.sum() >= 10:
    f07_results['correlation_60m'] = correlation_analysis(
        df.loc[mask_f07_60, 'pm_vol_ratio'].values,
        df.loc[mask_f07_60, 'pnl_60m_nosl'].values,
        'pm_vol_ratio'
    )
    print(f"  Correlation (60m): Pearson r={f07_results['correlation_60m']['pearson_r']}, "
          f"p={f07_results['correlation_60m']['pearson_p']}")

# Quartile analysis
f07_results['quartiles_t2'] = quartile_analysis(df, 'pm_vol_ratio', 'pnl_t2_close')
f07_results['quartiles_60m'] = quartile_analysis(df, 'pm_vol_ratio', 'pnl_60m_nosl')

for q_name, q_data in [('quartiles_t2', f07_results['quartiles_t2']),
                        ('quartiles_60m', f07_results['quartiles_60m'])]:
    if 'note' not in q_data:
        print(f"  {q_name}:")
        for q in ['Q1_low', 'Q2', 'Q3', 'Q4_high']:
            if q in q_data and 'avg_pnl' in q_data[q]:
                print(f"    {q}: n={q_data[q]['n']}, avg_pnl={q_data[q]['avg_pnl']:.4f}, "
                      f"sharpe={q_data[q]['sharpe']:.4f}")

# Threshold sweep
thresholds_lower = [1.5, 2.0, 3.0, 5.0, 10.0]
f07_results['threshold_lower_t2'] = threshold_analysis(
    df, 'pm_vol_ratio', 'pnl_t2_close', thresholds_lower, direction='above')
f07_results['threshold_lower_60m'] = threshold_analysis(
    df, 'pm_vol_ratio', 'pnl_60m_nosl', thresholds_lower, direction='above')

print("  Threshold sweep (pnl_t2_close, PASS = pm_vol_ratio >= threshold):")
for t, data in f07_results['threshold_lower_t2'].items():
    p = data['pass']
    f = data['fail']
    if 'avg_pnl' in p and 'avg_pnl' in f:
        print(f"    >= {t}x: PASS n={p['n']} avg={p['avg_pnl']:.4f} | FAIL n={f['n']} avg={f['avg_pnl']:.4f}")

# Upper bound test: exclude pm_vol_ratio > 10x
f07_results['upper_bound_exclude_10x'] = {}
for pnl_name in ['pnl_t2_close', 'pnl_60m_nosl']:
    valid = df[['pm_vol_ratio', pnl_name]].dropna()
    kept = valid[valid['pm_vol_ratio'] <= 10.0][pnl_name].values
    excluded = valid[valid['pm_vol_ratio'] > 10.0][pnl_name].values
    f07_results['upper_bound_exclude_10x'][pnl_name] = {
        'kept': compute_metrics(kept),
        'excluded': compute_metrics(excluded),
    }
    k = f07_results['upper_bound_exclude_10x'][pnl_name]
    if 'avg_pnl' in k['kept'] and 'avg_pnl' in k['excluded']:
        print(f"  Upper bound (exclude >10x, {pnl_name}): "
              f"KEPT n={k['kept']['n']} avg={k['kept']['avg_pnl']:.4f} | "
              f"EXCLUDED n={k['excluded']['n']} avg={k['excluded']['avg_pnl']:.4f}")

# Cross-strategy
med_pmvr = df['pm_vol_ratio'].median()
f07_results['cross_strategy'] = cross_strategy_test(
    df, 'pm_vol_ratio', get_pnl_for_strategy, median_val=med_pmvr)

with open(os.path.join(FEATURES_DIR, 'F07_pm_vol_ratio_results.json'), 'w') as f:
    json.dump(f07_results, f, indent=2, default=str)
print("  Saved F07 results.")

###############################################################################
# F08: ORB Width Continuous Score
###############################################################################
print("\n" + "="*70)
print("F08: ORB Width Continuous Score")
print("="*70)

# Compute ORB width
df['orb_width'] = df['or_high'] - df['or_low']
df['orb_width_pct'] = df['orb_width'] / df['entry_price']
df['orb_width_score'] = df['orb_width_pct'] / df['atr_pct']

has_ows = df['orb_width_score'].notna() & (df['orb_width_score'] > 0) & np.isfinite(df['orb_width_score'])
print(f"  orb_width_score computed: {has_ows.sum()} rows")
print(f"  orb_width_score range: {df.loc[has_ows, 'orb_width_score'].min():.4f} to {df.loc[has_ows, 'orb_width_score'].max():.4f}")
print(f"  orb_width_score median: {df.loc[has_ows, 'orb_width_score'].median():.4f}")

f08_results = {
    'feature': 'F08_orb_width_score',
    'description': 'ORB width / entry_price / ATR%, normalized opening range',
    'validation_required': True,
    'live_trading_weeks_needed': 4,
}

# Correlation
mask_f08 = has_ows & df['pnl_t2_close'].notna()
if mask_f08.sum() >= 10:
    f08_results['correlation_t2'] = correlation_analysis(
        df.loc[mask_f08, 'orb_width_score'].values,
        df.loc[mask_f08, 'pnl_t2_close'].values,
        'orb_width_score'
    )
    print(f"  Correlation (t2): Pearson r={f08_results['correlation_t2']['pearson_r']}, "
          f"p={f08_results['correlation_t2']['pearson_p']}")

mask_f08_60 = has_ows & df['pnl_60m_nosl'].notna()
if mask_f08_60.sum() >= 10:
    f08_results['correlation_60m'] = correlation_analysis(
        df.loc[mask_f08_60, 'orb_width_score'].values,
        df.loc[mask_f08_60, 'pnl_60m_nosl'].values,
        'orb_width_score'
    )
    print(f"  Correlation (60m): Pearson r={f08_results['correlation_60m']['pearson_r']}, "
          f"p={f08_results['correlation_60m']['pearson_p']}")

# Quartile analysis
f08_results['quartiles_t2'] = quartile_analysis(df[has_ows], 'orb_width_score', 'pnl_t2_close')
f08_results['quartiles_60m'] = quartile_analysis(df[has_ows], 'orb_width_score', 'pnl_60m_nosl')

for q_name, q_data in [('quartiles_t2', f08_results['quartiles_t2']),
                        ('quartiles_60m', f08_results['quartiles_60m'])]:
    if 'note' not in q_data:
        print(f"  {q_name}:")
        for q in ['Q1_low', 'Q2', 'Q3', 'Q4_high']:
            if q in q_data and 'avg_pnl' in q_data[q]:
                print(f"    {q}: n={q_data[q]['n']}, avg_pnl={q_data[q]['avg_pnl']:.4f}, "
                      f"sharpe={q_data[q]['sharpe']:.4f}")

# Top 50% vs bottom 50%
med_ows = df.loc[has_ows, 'orb_width_score'].median()
f08_results['median_split'] = {}
for pnl_name in ['pnl_t2_close', 'pnl_60m_nosl']:
    valid = df[has_ows & df[pnl_name].notna()]
    top_half = valid[valid['orb_width_score'] >= med_ows][pnl_name].values
    bot_half = valid[valid['orb_width_score'] < med_ows][pnl_name].values
    f08_results['median_split'][pnl_name] = {
        'top_50pct': compute_metrics(top_half),
        'bottom_50pct': compute_metrics(bot_half),
    }
    t = f08_results['median_split'][pnl_name]
    if 'avg_pnl' in t['top_50pct'] and 'avg_pnl' in t['bottom_50pct']:
        print(f"  Median split ({pnl_name}): "
              f"TOP n={t['top_50pct']['n']} avg={t['top_50pct']['avg_pnl']:.4f} | "
              f"BOT n={t['bottom_50pct']['n']} avg={t['bottom_50pct']['avg_pnl']:.4f}")

# Threshold sweep for orb_width_score
orb_thresholds = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
f08_results['threshold_t2'] = threshold_analysis(
    df[has_ows], 'orb_width_score', 'pnl_t2_close', orb_thresholds, direction='above')
f08_results['threshold_60m'] = threshold_analysis(
    df[has_ows], 'orb_width_score', 'pnl_60m_nosl', orb_thresholds, direction='above')

print("  Threshold sweep (pnl_t2_close, PASS = orb_width_score >= threshold):")
for t, data in f08_results['threshold_t2'].items():
    p = data['pass']
    fa = data['fail']
    if 'avg_pnl' in p and 'avg_pnl' in fa:
        print(f"    >= {t}: PASS n={p['n']} avg={p['avg_pnl']:.4f} | FAIL n={fa['n']} avg={fa['avg_pnl']:.4f}")

# Cross-strategy
f08_results['cross_strategy'] = cross_strategy_test(
    df[has_ows], 'orb_width_score', get_pnl_for_strategy, median_val=med_ows)

# Also test by direction: does wide ORB help LONG vs SHORT differently?
f08_results['by_direction'] = {}
for d in ['LONG', 'SHORT']:
    sub = df[has_ows & (df['direction'] == d)]
    for pnl_name in ['pnl_t2_close', 'pnl_60m_nosl']:
        valid = sub[sub[pnl_name].notna()]
        if len(valid) < 10:
            continue
        top_half = valid[valid['orb_width_score'] >= med_ows][pnl_name].values
        bot_half = valid[valid['orb_width_score'] < med_ows][pnl_name].values
        key = f"{d}_{pnl_name}"
        f08_results['by_direction'][key] = {
            'top_50pct': compute_metrics(top_half),
            'bottom_50pct': compute_metrics(bot_half),
        }

with open(os.path.join(FEATURES_DIR, 'F08_orb_width_results.json'), 'w') as f:
    json.dump(f08_results, f, indent=2, default=str)
print("  Saved F08 results.")

###############################################################################
# F09: Market Cap Filter + F10: Float Size Effect
# (Combined API fetch to minimize calls)
###############################################################################
print("\n" + "="*70)
print("F09/F10: Fetching Polygon reference data...")
print("="*70)

# Load cache if exists
if os.path.exists(TICKER_CACHE_PATH):
    with open(TICKER_CACHE_PATH) as f:
        ticker_ref_cache = json.load(f)
    print(f"  Loaded cache: {len(ticker_ref_cache)} tickers")
else:
    ticker_ref_cache = {}

unique_tickers = sorted(df['ticker'].unique())
tickers_to_fetch = [t for t in unique_tickers if t not in ticker_ref_cache]
print(f"  Need to fetch: {len(tickers_to_fetch)} tickers (of {len(unique_tickers)} total)")

for i, ticker in enumerate(tickers_to_fetch):
    url = f"https://api.polygon.io/v3/reference/tickers/{ticker}?apiKey={POLYGON_API_KEY}"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            results = data.get('results', {})
            ticker_ref_cache[ticker] = {
                'market_cap': results.get('market_cap'),
                'shares_outstanding': results.get('share_class_shares_outstanding'),
                'weighted_shares': results.get('weighted_shares_outstanding'),
                'name': results.get('name'),
                'type': results.get('type'),
                'primary_exchange': results.get('primary_exchange'),
            }
        else:
            ticker_ref_cache[ticker] = {'error': f'HTTP {resp.status_code}'}
    except Exception as e:
        ticker_ref_cache[ticker] = {'error': str(e)}

    if (i + 1) % 50 == 0:
        print(f"    Fetched {i+1}/{len(tickers_to_fetch)}...")
    time.sleep(POLYGON_SLEEP)

# Save cache
with open(TICKER_CACHE_PATH, 'w') as f:
    json.dump(ticker_ref_cache, f, indent=2)
print(f"  Cached {len(ticker_ref_cache)} tickers to {TICKER_CACHE_PATH}")

# Map market cap and float to df
df['market_cap'] = df['ticker'].map(lambda t: ticker_ref_cache.get(t, {}).get('market_cap'))
df['shares_outstanding'] = df['ticker'].map(
    lambda t: ticker_ref_cache.get(t, {}).get('shares_outstanding'))

has_mcap = df['market_cap'].notna()
has_float = df['shares_outstanding'].notna()
print(f"  market_cap available: {has_mcap.sum()} rows")
print(f"  shares_outstanding available: {has_float.sum()} rows")

###############################################################################
# F09: Market Cap Filter
###############################################################################
print("\n" + "="*70)
print("F09: Market Cap Filter")
print("="*70)

# Bucket market cap
def mcap_bucket(mc):
    if mc is None or not np.isfinite(mc):
        return None
    if mc < 300e6:
        return 'micro'
    elif mc < 2e9:
        return 'small'
    elif mc < 10e9:
        return 'mid'
    else:
        return 'large'

df['mcap_bucket'] = df['market_cap'].apply(mcap_bucket)
print(f"  Market cap buckets:")
print(f"    {df['mcap_bucket'].value_counts().to_dict()}")

# Convert market_cap to millions for readability
df['market_cap_m'] = df['market_cap'] / 1e6

f09_results = {
    'feature': 'F09_market_cap',
    'description': 'Market cap from Polygon reference API',
    'validation_required': True,
    'live_trading_weeks_needed': 4,
}

# Correlation (using log market cap)
df['log_market_cap'] = np.log10(df['market_cap'].clip(lower=1))
mask_f09 = df['log_market_cap'].notna() & np.isfinite(df['log_market_cap']) & df['pnl_t2_close'].notna()
if mask_f09.sum() >= 10:
    f09_results['correlation_t2'] = correlation_analysis(
        df.loc[mask_f09, 'log_market_cap'].values,
        df.loc[mask_f09, 'pnl_t2_close'].values,
        'log_market_cap'
    )
    print(f"  Correlation log(mcap) vs t2: r={f09_results['correlation_t2']['pearson_r']}, "
          f"p={f09_results['correlation_t2']['pearson_p']}")

mask_f09_60 = df['log_market_cap'].notna() & np.isfinite(df['log_market_cap']) & df['pnl_60m_nosl'].notna()
if mask_f09_60.sum() >= 10:
    f09_results['correlation_60m'] = correlation_analysis(
        df.loc[mask_f09_60, 'log_market_cap'].values,
        df.loc[mask_f09_60, 'pnl_60m_nosl'].values,
        'log_market_cap'
    )
    print(f"  Correlation log(mcap) vs 60m: r={f09_results['correlation_60m']['pearson_r']}, "
          f"p={f09_results['correlation_60m']['pearson_p']}")

# Bucket analysis
f09_results['bucket_analysis'] = {}
for pnl_name in ['pnl_t2_close', 'pnl_60m_nosl']:
    f09_results['bucket_analysis'][pnl_name] = {}
    for bucket in ['micro', 'small', 'mid', 'large']:
        sub = df[(df['mcap_bucket'] == bucket) & df[pnl_name].notna()][pnl_name].values
        f09_results['bucket_analysis'][pnl_name][bucket] = compute_metrics(sub)

    print(f"  Bucket analysis ({pnl_name}):")
    for bucket in ['micro', 'small', 'mid', 'large']:
        m = f09_results['bucket_analysis'][pnl_name][bucket]
        if 'avg_pnl' in m:
            print(f"    {bucket}: n={m['n']}, avg_pnl={m['avg_pnl']:.4f}, sharpe={m['sharpe']:.4f}, "
                  f"max_loss={m['max_loss']:.4f}")
        else:
            print(f"    {bucket}: {m}")

# Threshold sweep (minimum market cap)
mcap_thresholds = [200e6, 300e6, 500e6, 1e9, 2e9]
f09_results['threshold_min_mcap'] = {}
for pnl_name in ['pnl_t2_close', 'pnl_60m_nosl']:
    f09_results['threshold_min_mcap'][pnl_name] = {}
    valid = df[has_mcap & df[pnl_name].notna()]
    for thresh in mcap_thresholds:
        pass_mask = valid['market_cap'] >= thresh
        pass_pnl = valid[pass_mask][pnl_name].values
        fail_pnl = valid[~pass_mask][pnl_name].values
        label = f"${thresh/1e6:.0f}M"
        f09_results['threshold_min_mcap'][pnl_name][label] = {
            'pass': compute_metrics(pass_pnl),
            'fail': compute_metrics(fail_pnl),
        }

    print(f"  Threshold sweep ({pnl_name}, PASS = mcap >= threshold):")
    for label, data in f09_results['threshold_min_mcap'][pnl_name].items():
        p = data['pass']
        fa = data['fail']
        if 'avg_pnl' in p and 'avg_pnl' in fa:
            print(f"    >= {label}: PASS n={p['n']} avg={p['avg_pnl']:.4f} sharpe={p['sharpe']:.4f} "
                  f"max_loss={p['max_loss']:.4f} | FAIL n={fa['n']} avg={fa['avg_pnl']:.4f}")

# Quartile analysis with log_market_cap
f09_results['quartiles_t2'] = quartile_analysis(df[has_mcap], 'log_market_cap', 'pnl_t2_close')
f09_results['quartiles_60m'] = quartile_analysis(df[has_mcap], 'log_market_cap', 'pnl_60m_nosl')

# Cross-strategy
med_mcap = df.loc[has_mcap, 'log_market_cap'].median()
f09_results['cross_strategy'] = cross_strategy_test(
    df[has_mcap], 'log_market_cap', get_pnl_for_strategy, median_val=med_mcap)

with open(os.path.join(FEATURES_DIR, 'F09_market_cap_results.json'), 'w') as f:
    json.dump(f09_results, f, indent=2, default=str)
print("  Saved F09 results.")

###############################################################################
# F10: Float Size Effect
###############################################################################
print("\n" + "="*70)
print("F10: Float Size Effect")
print("="*70)

# Bucket float (shares outstanding in shares, not millions)
def float_bucket(so):
    if so is None or not np.isfinite(so):
        return None
    so_m = so / 1e6  # convert to millions
    if so_m < 10:
        return 'low'
    elif so_m < 50:
        return 'medium'
    elif so_m < 200:
        return 'high'
    else:
        return 'very_high'

df['float_bucket'] = df['shares_outstanding'].apply(float_bucket)
df['float_millions'] = df['shares_outstanding'] / 1e6
df['log_float'] = np.log10(df['shares_outstanding'].clip(lower=1))

print(f"  Float buckets:")
print(f"    {df['float_bucket'].value_counts().to_dict()}")

f10_results = {
    'feature': 'F10_float_size',
    'description': 'Shares outstanding from Polygon API as float proxy',
    'validation_required': True,
    'live_trading_weeks_needed': 4,
}

# Correlation
mask_f10 = df['log_float'].notna() & np.isfinite(df['log_float']) & df['pnl_t2_close'].notna()
if mask_f10.sum() >= 10:
    f10_results['correlation_t2'] = correlation_analysis(
        df.loc[mask_f10, 'log_float'].values,
        df.loc[mask_f10, 'pnl_t2_close'].values,
        'log_float'
    )
    print(f"  Correlation log(float) vs t2: r={f10_results['correlation_t2']['pearson_r']}, "
          f"p={f10_results['correlation_t2']['pearson_p']}")

mask_f10_60 = df['log_float'].notna() & np.isfinite(df['log_float']) & df['pnl_60m_nosl'].notna()
if mask_f10_60.sum() >= 10:
    f10_results['correlation_60m'] = correlation_analysis(
        df.loc[mask_f10_60, 'log_float'].values,
        df.loc[mask_f10_60, 'pnl_60m_nosl'].values,
        'log_float'
    )
    print(f"  Correlation log(float) vs 60m: r={f10_results['correlation_60m']['pearson_r']}, "
          f"p={f10_results['correlation_60m']['pearson_p']}")

# Bucket analysis
f10_results['bucket_analysis'] = {}
for pnl_name in ['pnl_t2_close', 'pnl_60m_nosl']:
    f10_results['bucket_analysis'][pnl_name] = {}
    for bucket in ['low', 'medium', 'high', 'very_high']:
        sub = df[(df['float_bucket'] == bucket) & df[pnl_name].notna()][pnl_name].values
        f10_results['bucket_analysis'][pnl_name][bucket] = compute_metrics(sub)

    print(f"  Bucket analysis ({pnl_name}):")
    for bucket in ['low', 'medium', 'high', 'very_high']:
        m = f10_results['bucket_analysis'][pnl_name][bucket]
        if 'avg_pnl' in m:
            print(f"    {bucket}: n={m['n']}, avg_pnl={m['avg_pnl']:.4f}, sharpe={m['sharpe']:.4f}")
        else:
            print(f"    {bucket}: {m}")

# Strategy-specific: SHORT strategies vs LONG strategies
f10_results['by_direction'] = {}
for direction in ['LONG', 'SHORT']:
    sub = df[(df['direction'] == direction) & has_float]
    f10_results['by_direction'][direction] = {}
    for pnl_name in ['pnl_t2_close', 'pnl_60m_nosl']:
        f10_results['by_direction'][direction][pnl_name] = {}
        for bucket in ['low', 'medium', 'high', 'very_high']:
            vals = sub[(sub['float_bucket'] == bucket) & sub[pnl_name].notna()][pnl_name].values
            f10_results['by_direction'][direction][pnl_name][bucket] = compute_metrics(vals)

    print(f"  {direction} by float bucket (pnl_t2_close):")
    for bucket in ['low', 'medium', 'high', 'very_high']:
        m = f10_results['by_direction'][direction]['pnl_t2_close'][bucket]
        if 'avg_pnl' in m:
            print(f"    {bucket}: n={m['n']}, avg_pnl={m['avg_pnl']:.4f}")
        else:
            print(f"    {bucket}: {m}")

# Threshold analysis (float in millions)
float_thresholds_m = [10, 20, 50, 100, 200]
f10_results['threshold_min_float'] = {}
for pnl_name in ['pnl_t2_close', 'pnl_60m_nosl']:
    f10_results['threshold_min_float'][pnl_name] = {}
    valid = df[has_float & df[pnl_name].notna()]
    for thresh_m in float_thresholds_m:
        thresh = thresh_m * 1e6
        pass_mask = valid['shares_outstanding'] >= thresh
        pass_pnl = valid[pass_mask][pnl_name].values
        fail_pnl = valid[~pass_mask][pnl_name].values
        label = f"{thresh_m}M_shares"
        f10_results['threshold_min_float'][pnl_name][label] = {
            'pass': compute_metrics(pass_pnl),
            'fail': compute_metrics(fail_pnl),
        }

# Quartile analysis
f10_results['quartiles_t2'] = quartile_analysis(df[has_float], 'log_float', 'pnl_t2_close')
f10_results['quartiles_60m'] = quartile_analysis(df[has_float], 'log_float', 'pnl_60m_nosl')

# Cross-strategy
med_float = df.loc[has_float, 'log_float'].median()
f10_results['cross_strategy'] = cross_strategy_test(
    df[has_float], 'log_float', get_pnl_for_strategy, median_val=med_float)

# Hypothesis test: low float hurts SHORT fade but helps LONG bounce
f10_results['hypothesis_low_float'] = {}
for direction in ['LONG', 'SHORT']:
    sub = df[(df['direction'] == direction) & has_float]
    for pnl_name in ['pnl_t2_close', 'pnl_60m_nosl']:
        valid = sub[sub[pnl_name].notna()]
        low = valid[valid['float_bucket'] == 'low'][pnl_name].values
        others = valid[valid['float_bucket'] != 'low'][pnl_name].values
        key = f"{direction}_{pnl_name}"
        f10_results['hypothesis_low_float'][key] = {
            'low_float': compute_metrics(low),
            'other_float': compute_metrics(others),
        }
        lf = f10_results['hypothesis_low_float'][key]
        if 'avg_pnl' in lf['low_float'] and 'avg_pnl' in lf['other_float']:
            print(f"  Hypothesis ({direction} {pnl_name}): "
                  f"LOW_FLOAT n={lf['low_float']['n']} avg={lf['low_float']['avg_pnl']:.4f} | "
                  f"OTHER n={lf['other_float']['n']} avg={lf['other_float']['avg_pnl']:.4f}")

with open(os.path.join(FEATURES_DIR, 'F10_float_size_results.json'), 'w') as f:
    json.dump(f10_results, f, indent=2, default=str)
print("  Saved F10 results.")

###############################################################################
# Combined Summary
###############################################################################
print("\n" + "="*70)
print("Combined Summary F07-F10")
print("="*70)

summary = {
    'generated': '2026-09-13',
    'dataset': {
        'total_rows': len(df),
        'pnl_t2_close_available': int(df['pnl_t2_close'].notna().sum()),
        'pnl_60m_nosl_available': int(df['pnl_60m_nosl'].notna().sum()),
    },
    'validation_required': True,
    'live_trading_weeks_needed': 4,
    'features': {},
}

# F07 summary
f07_corr_t2 = f07_results.get('correlation_t2', {})
f07_corr_60 = f07_results.get('correlation_60m', {})
f07_q_t2 = f07_results.get('quartiles_t2', {})
f07_q1 = f07_q_t2.get('Q1_low', {})
f07_q4 = f07_q_t2.get('Q4_high', {})

# Count consistent strategies for cross-strategy
def count_consistent_strategies(cross_strat_data, higher_is_better=True):
    """Count how many strategies show consistent direction."""
    consistent = 0
    total = 0
    for strat, data in cross_strat_data.items():
        h = data.get('high_feat', {})
        l = data.get('low_feat', {})
        if 'avg_pnl' not in h or 'avg_pnl' not in l:
            continue
        total += 1
        if higher_is_better and h['avg_pnl'] > l['avg_pnl']:
            consistent += 1
        elif not higher_is_better and h['avg_pnl'] < l['avg_pnl']:
            consistent += 1
    return consistent, total

f07_cons_hi, f07_tot = count_consistent_strategies(f07_results.get('cross_strategy', {}), True)
f07_cons_lo, _ = count_consistent_strategies(f07_results.get('cross_strategy', {}), False)
f07_best_dir = 'higher_better' if f07_cons_hi >= f07_cons_lo else 'lower_better'
f07_cons = max(f07_cons_hi, f07_cons_lo)

summary['features']['F07_pm_vol_ratio'] = {
    'pearson_r_t2': f07_corr_t2.get('pearson_r'),
    'pearson_p_t2': f07_corr_t2.get('pearson_p'),
    'spearman_r_t2': f07_corr_t2.get('spearman_r'),
    'pearson_r_60m': f07_corr_60.get('pearson_r'),
    'Q1_avg_pnl': f07_q1.get('avg_pnl'),
    'Q4_avg_pnl': f07_q4.get('avg_pnl'),
    'cross_strategy_consistent': f07_cons,
    'cross_strategy_total': f07_tot,
    'direction': f07_best_dir,
}

# F08 summary
f08_corr_t2 = f08_results.get('correlation_t2', {})
f08_corr_60 = f08_results.get('correlation_60m', {})
f08_q_t2 = f08_results.get('quartiles_t2', {})
f08_q1 = f08_q_t2.get('Q1_low', {})
f08_q4 = f08_q_t2.get('Q4_high', {})
f08_cons_hi, f08_tot = count_consistent_strategies(f08_results.get('cross_strategy', {}), True)
f08_cons_lo, _ = count_consistent_strategies(f08_results.get('cross_strategy', {}), False)
f08_best_dir = 'higher_better' if f08_cons_hi >= f08_cons_lo else 'lower_better'
f08_cons = max(f08_cons_hi, f08_cons_lo)

summary['features']['F08_orb_width_score'] = {
    'pearson_r_t2': f08_corr_t2.get('pearson_r'),
    'pearson_p_t2': f08_corr_t2.get('pearson_p'),
    'spearman_r_t2': f08_corr_t2.get('spearman_r'),
    'pearson_r_60m': f08_corr_60.get('pearson_r'),
    'Q1_avg_pnl': f08_q1.get('avg_pnl'),
    'Q4_avg_pnl': f08_q4.get('avg_pnl'),
    'cross_strategy_consistent': f08_cons,
    'cross_strategy_total': f08_tot,
    'direction': f08_best_dir,
}

# F09 summary
f09_corr_t2 = f09_results.get('correlation_t2', {})
f09_corr_60 = f09_results.get('correlation_60m', {})
f09_q_t2 = f09_results.get('quartiles_t2', {})
f09_q1 = f09_q_t2.get('Q1_low', {})
f09_q4 = f09_q_t2.get('Q4_high', {})
f09_cons_hi, f09_tot = count_consistent_strategies(f09_results.get('cross_strategy', {}), True)
f09_cons_lo, _ = count_consistent_strategies(f09_results.get('cross_strategy', {}), False)
f09_best_dir = 'higher_better' if f09_cons_hi >= f09_cons_lo else 'lower_better'
f09_cons = max(f09_cons_hi, f09_cons_lo)

summary['features']['F09_market_cap'] = {
    'pearson_r_t2': f09_corr_t2.get('pearson_r'),
    'pearson_p_t2': f09_corr_t2.get('pearson_p'),
    'spearman_r_t2': f09_corr_t2.get('spearman_r'),
    'pearson_r_60m': f09_corr_60.get('pearson_r'),
    'Q1_avg_pnl': f09_q1.get('avg_pnl'),
    'Q4_avg_pnl': f09_q4.get('avg_pnl'),
    'cross_strategy_consistent': f09_cons,
    'cross_strategy_total': f09_tot,
    'direction': f09_best_dir,
}

# F10 summary
f10_corr_t2 = f10_results.get('correlation_t2', {})
f10_corr_60 = f10_results.get('correlation_60m', {})
f10_q_t2 = f10_results.get('quartiles_t2', {})
f10_q1 = f10_q_t2.get('Q1_low', {})
f10_q4 = f10_q_t2.get('Q4_high', {})
f10_cons_hi, f10_tot = count_consistent_strategies(f10_results.get('cross_strategy', {}), True)
f10_cons_lo, _ = count_consistent_strategies(f10_results.get('cross_strategy', {}), False)
f10_best_dir = 'higher_better' if f10_cons_hi >= f10_cons_lo else 'lower_better'
f10_cons = max(f10_cons_hi, f10_cons_lo)

summary['features']['F10_float_size'] = {
    'pearson_r_t2': f10_corr_t2.get('pearson_r'),
    'pearson_p_t2': f10_corr_t2.get('pearson_p'),
    'spearman_r_t2': f10_corr_t2.get('spearman_r'),
    'pearson_r_60m': f10_corr_60.get('pearson_r'),
    'Q1_avg_pnl': f10_q1.get('avg_pnl'),
    'Q4_avg_pnl': f10_q4.get('avg_pnl'),
    'cross_strategy_consistent': f10_cons,
    'cross_strategy_total': f10_tot,
    'direction': f10_best_dir,
}

# Rank features by absolute correlation strength (t2)
ranking = []
for fname, fdata in summary['features'].items():
    r = abs(fdata.get('pearson_r_t2') or 0)
    p = fdata.get('pearson_p_t2')
    cons = fdata.get('cross_strategy_consistent', 0)
    tot = fdata.get('cross_strategy_total', 1)
    ranking.append({
        'feature': fname,
        'abs_pearson_r': round(r, 4),
        'pearson_p': p,
        'cross_strategy_pct': round(cons / max(tot, 1), 4),
        'direction': fdata.get('direction'),
    })
ranking.sort(key=lambda x: x['abs_pearson_r'], reverse=True)
summary['ranking'] = ranking

print("\nFeature Ranking by |Pearson r| (pnl_t2_close):")
for i, r in enumerate(ranking):
    print(f"  {i+1}. {r['feature']}: |r|={r['abs_pearson_r']}, p={r['pearson_p']}, "
          f"cross-strat={r['cross_strategy_pct']*100:.0f}%, dir={r['direction']}")

with open(os.path.join(FEATURES_DIR, 'F07_F10_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2, default=str)
print("  Saved combined summary.")

###############################################################################
# Charts: 2x2 grid
###############################################################################
print("\n" + "="*70)
print("Generating charts...")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('F07-F10 Feature Investigation', fontsize=14, fontweight='bold', y=0.98)

# ── F07: PM Vol Ratio quartile PnL ──
ax = axes[0, 0]
q_data = f07_results.get('quartiles_t2', {})
if 'note' not in q_data:
    labels = ['Q1\n(low)', 'Q2', 'Q3', 'Q4\n(high)']
    avgs = [q_data.get(q, {}).get('avg_pnl', 0) for q in ['Q1_low', 'Q2', 'Q3', 'Q4_high']]
    ns = [q_data.get(q, {}).get('n', 0) for q in ['Q1_low', 'Q2', 'Q3', 'Q4_high']]
    colors = ['#d32f2f' if v < 0 else '#388e3c' for v in avgs]
    bars = ax.bar(labels, avgs, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    for bar, n in zip(bars, ns):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'n={n}', ha='center', va='bottom', fontsize=9)
    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-')
    ax.set_title('F07: PM Vol Ratio Quartiles\n(pnl_t2_close)', fontsize=11)
    ax.set_ylabel('Avg PnL (%)')
    ax.grid(axis='y', alpha=0.3)
else:
    # Fallback: use 60m data
    q_data_60 = f07_results.get('quartiles_60m', {})
    if 'note' not in q_data_60:
        labels = ['Q1\n(low)', 'Q2', 'Q3', 'Q4\n(high)']
        avgs = [q_data_60.get(q, {}).get('avg_pnl', 0) for q in ['Q1_low', 'Q2', 'Q3', 'Q4_high']]
        ns = [q_data_60.get(q, {}).get('n', 0) for q in ['Q1_low', 'Q2', 'Q3', 'Q4_high']]
        colors = ['#d32f2f' if v < 0 else '#388e3c' for v in avgs]
        bars = ax.bar(labels, avgs, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        for bar, n in zip(bars, ns):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'n={n}', ha='center', va='bottom', fontsize=9)
        ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-')
        ax.set_title('F07: PM Vol Ratio Quartiles\n(pnl_60m_nosl)', fontsize=11)
        ax.set_ylabel('Avg PnL (%)')
        ax.grid(axis='y', alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('F07: PM Vol Ratio Quartiles')

# ── F08: ORB Width scatter ──
ax = axes[0, 1]
valid_f08 = df[has_ows & df['pnl_t2_close'].notna()]
if len(valid_f08) >= 10:
    ax.scatter(valid_f08['orb_width_score'], valid_f08['pnl_t2_close'],
               alpha=0.5, s=25, c='#1565c0', edgecolors='none')
    # Add regression line
    z = np.polyfit(valid_f08['orb_width_score'].values, valid_f08['pnl_t2_close'].values, 1)
    p_line = np.poly1d(z)
    x_range = np.linspace(valid_f08['orb_width_score'].min(), valid_f08['orb_width_score'].max(), 100)
    ax.plot(x_range, p_line(x_range), '--', color='#d32f2f', linewidth=2,
            label=f'r={f08_results.get("correlation_t2", {}).get("pearson_r", 0):.3f}')
    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-')
    ax.set_xlabel('ORB Width Score (orb_width / entry_price / ATR)')
    ax.set_ylabel('PnL T+2 Close (%)')
    ax.set_title('F08: ORB Width Score vs T+2 PnL', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
else:
    # Fallback to 60m
    valid_f08_60 = df[has_ows & df['pnl_60m_nosl'].notna()]
    if len(valid_f08_60) >= 10:
        ax.scatter(valid_f08_60['orb_width_score'], valid_f08_60['pnl_60m_nosl'],
                   alpha=0.5, s=25, c='#1565c0', edgecolors='none')
        z = np.polyfit(valid_f08_60['orb_width_score'].values, valid_f08_60['pnl_60m_nosl'].values, 1)
        p_line = np.poly1d(z)
        x_range = np.linspace(valid_f08_60['orb_width_score'].min(), valid_f08_60['orb_width_score'].max(), 100)
        ax.plot(x_range, p_line(x_range), '--', color='#d32f2f', linewidth=2)
        ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-')
        ax.set_xlabel('ORB Width Score')
        ax.set_ylabel('PnL 60m (%)')
        ax.set_title('F08: ORB Width Score vs 60m PnL', fontsize=11)
        ax.grid(alpha=0.3)

# ── F09: Market Cap bucket bars ──
ax = axes[1, 0]
f09_bucket = f09_results.get('bucket_analysis', {})
pnl_key = 'pnl_t2_close' if 'pnl_t2_close' in f09_bucket else 'pnl_60m_nosl'
bucket_data = f09_bucket.get(pnl_key, {})
buckets_ordered = ['micro', 'small', 'mid', 'large']
avgs = [bucket_data.get(b, {}).get('avg_pnl', 0) for b in buckets_ordered]
ns = [bucket_data.get(b, {}).get('n', 0) for b in buckets_ordered]
max_losses = [bucket_data.get(b, {}).get('max_loss', 0) for b in buckets_ordered]

x = np.arange(len(buckets_ordered))
width = 0.35
colors_avg = ['#d32f2f' if v < 0 else '#388e3c' for v in avgs]
bars1 = ax.bar(x - width/2, avgs, width, color=colors_avg, alpha=0.8, label='Avg PnL',
               edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + width/2, max_losses, width, color='#b71c1c', alpha=0.5, label='Max Loss',
               edgecolor='black', linewidth=0.5)
for bar, n in zip(bars1, ns):
    ax.text(bar.get_x() + bar.get_width()/2, max(bar.get_height(), 0) + 0.1,
            f'n={n}', ha='center', va='bottom', fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(['Micro\n<$300M', 'Small\n$300M-2B', 'Mid\n$2B-10B', 'Large\n>$10B'])
ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-')
ax.set_title(f'F09: Market Cap Buckets\n({pnl_key})', fontsize=11)
ax.set_ylabel('PnL (%)')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

# ── F10: Float by direction ──
ax = axes[1, 1]
float_buckets_order = ['low', 'medium', 'high', 'very_high']
float_labels = ['Low\n<10M', 'Med\n10-50M', 'High\n50-200M', 'VHigh\n>200M']

f10_dir = f10_results.get('by_direction', {})
pnl_key_f10 = 'pnl_t2_close'

long_avgs = []
short_avgs = []
for b in float_buckets_order:
    lm = f10_dir.get('LONG', {}).get(pnl_key_f10, {}).get(b, {})
    sm = f10_dir.get('SHORT', {}).get(pnl_key_f10, {}).get(b, {})
    long_avgs.append(lm.get('avg_pnl', 0) if 'avg_pnl' in lm else 0)
    short_avgs.append(sm.get('avg_pnl', 0) if 'avg_pnl' in sm else 0)

x = np.arange(len(float_buckets_order))
width = 0.35
ax.bar(x - width/2, long_avgs, width, color='#1565c0', alpha=0.8, label='LONG', edgecolor='black', linewidth=0.5)
ax.bar(x + width/2, short_avgs, width, color='#d32f2f', alpha=0.8, label='SHORT', edgecolor='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(float_labels)
ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-')
ax.set_title(f'F10: Float Size by Direction\n({pnl_key_f10})', fontsize=11)
ax.set_ylabel('Avg PnL (%)')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
chart_path = os.path.join(FEATURES_DIR, 'F07_F10_analysis.png')
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved chart: {chart_path}")

###############################################################################
# Final Summary to stdout
###############################################################################
print("\n" + "="*70)
print("FINAL RESULTS SUMMARY")
print("="*70)
for fname, fdata in summary['features'].items():
    print(f"\n{fname}:")
    print(f"  Pearson r (t2):  {fdata.get('pearson_r_t2')}")
    print(f"  Pearson p (t2):  {fdata.get('pearson_p_t2')}")
    print(f"  Spearman r (t2): {fdata.get('spearman_r_t2')}")
    print(f"  Pearson r (60m): {fdata.get('pearson_r_60m')}")
    print(f"  Q1 avg PnL:      {fdata.get('Q1_avg_pnl')}")
    print(f"  Q4 avg PnL:      {fdata.get('Q4_avg_pnl')}")
    print(f"  Cross-strategy:  {fdata.get('cross_strategy_consistent')}/{fdata.get('cross_strategy_total')}")
    print(f"  Direction:       {fdata.get('direction')}")

print("\n\nRanking (by |Pearson r| vs pnl_t2_close):")
for i, r in enumerate(ranking):
    print(f"  {i+1}. {r['feature']}: |r|={r['abs_pearson_r']}, p={r['pearson_p']}, "
          f"cross-strat={r['cross_strategy_pct']*100:.0f}%")

print("\nAll results saved to:", FEATURES_DIR)
print("DONE.")
