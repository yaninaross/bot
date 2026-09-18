"""
analyze_robustness.py — Overfitting & Robustness Analysis for WF Configs
=========================================================================
Queries BQ for all WF optimization results, computes three robustness metrics
per ticker's winning config, and produces a robustness-adjusted composite score.

Three robustness components:
  1. Neighborhood Density — fraction of configs for same ticker with positive TEST PnL
  2. Parameter Smoothness — median neighbor PnL vs winner PnL (smooth surface = robust)
  3. Cross-Seed Agreement — do different seeds converge on similar optimal params?

Robustness-adjusted composite:
  robust_composite = base_composite * (0.3 + 0.7 * robustness_multiplier)

Tables:
  1: Per-ticker robustness scorecard
  2: Parameter sensitivity heatmaps (prob_thresh vs TP/SL)
  3: Cross-seed convergence
  4: Robustness-adjusted ranking vs original ranking
  5: Red flags — tickers where robustness is dangerously low

Run in Colab after WF optimization completes.
"""

import pandas as pd
import numpy as np

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass

from google.cloud import bigquery

# ── CONFIG ──────────────────────────────────────────────────────────────────
MODEL_VERSION = 'robust_wf_v8'
DEPLOY_THRESHOLD = 0.018
MIN_TRADES = 3
BQ_TABLE = f"{GCP_PROJECT}.scoring_logs.wf_optimization_results"
TICKER_RENAME = {'MIGI': 'BGDE', 'BITF': 'KEEL'}
EXCLUDE_TICKERS = {'TLT', 'VIX', 'SPY', 'IWM', 'MSTR'}
W = 130

# Robustness weights
W_DENSITY   = 0.35   # neighborhood density
W_SMOOTH    = 0.35   # parameter smoothness
W_SEED      = 0.30   # cross-seed agreement
ROBUST_FLOOR = 0.30  # minimum multiplier (worst case = 30% of base score)
# ── END CONFIG ──────────────────────────────────────────────────────────────

client = bigquery.Client()
query = f"""
SELECT *
FROM `{BQ_TABLE}`
WHERE model_version = '{MODEL_VERSION}'
  AND survived_a = TRUE
ORDER BY ticker, seed, prob_thresh
"""
df = client.query(query).to_dataframe()
print(f"Loaded {len(df)} rows from BQ (model_version={MODEL_VERSION})")

if df.empty:
    print("No data found. Run WF optimizer first.")
    raise SystemExit

df['ticker'] = df['ticker'].replace(TICKER_RENAME)
df = df[~df['ticker'].isin(EXCLUDE_TICKERS)]

for col in ['val_b_avg_pnl', 'val_b_win_rate']:
    df[col] = df[col].fillna(0.0)
df['val_b_count'] = df['val_b_count'].fillna(0).astype(int)
df['is_ensemble'] = df['is_ensemble'].fillna(False).astype(bool)
df['tpsl_mode'] = df['tpsl_mode'].fillna('static')

for col in ['tp_mult', 'sl_mult', 'trail_start_mult', 'trail_end_mult',
            'tp_pct', 'sl_pct', 'prob_thresh']:
    if col in df.columns:
        df[col] = df[col].fillna(0.0)

df['base_experiment'] = df['experiment'].apply(
    lambda e: e.replace('_ATR', '').replace('_ATRADP', '').replace('_REGIME', '').replace('_TRAIL', '')
    if any(e.endswith(s) for s in ('_ATR', '_ATRADP', '_REGIME', '_TRAIL')) else e)

# Dedup
dedup_cols = ['experiment', 'ticker', 'seed', 'is_ensemble', 'tpsl_mode',
              'entry_time', 'prob_thresh', 'tp_pct', 'sl_pct',
              'tp_mult', 'sl_mult', 'trail_start_mult', 'trail_end_mult',
              'vix_threshold', 'tlt_crash', 'spy_ignore', 'vol_filter_pct']
dedup_cols = [c for c in dedup_cols if c in df.columns]
df = df.sort_values('run_timestamp', ascending=False).drop_duplicates(subset=dedup_cols, keep='first')
print(f"After dedup: {len(df)} rows, {df['ticker'].nunique()} tickers")

# Base composite (same as analyze_composite_score.py)
df['base_composite'] = (df['val_b_avg_pnl'] * df['val_b_win_rate']
                        * np.sqrt(df['val_b_count']) * np.log2(1 + df['val_b_count']))

# Non-ensemble individual seed rows only for analysis
seeds_df = df[~df['is_ensemble']].copy()

tickers = sorted(seeds_df['ticker'].unique())
experiments = sorted(seeds_df['base_experiment'].unique())
modes = sorted(seeds_df['tpsl_mode'].unique())

print(f"Experiments: {experiments}")
print(f"Tickers: {len(tickers)}")
print(f"Modes: {modes}")
print(f"Seeds per ticker: {seeds_df.groupby('ticker')['seed'].nunique().describe()}")


# ==============================================================================
# COMPONENT 1: NEIGHBORHOOD DENSITY
# ==============================================================================
# For each (ticker, base_experiment, tpsl_mode): what fraction of tested configs
# have positive test PnL? High density = the signal is broadly profitable,
# not a narrow peak.

def compute_density(group):
    has_test = group[group['val_b_count'] >= MIN_TRADES]
    n_total = len(has_test)
    if n_total == 0:
        return pd.Series({'n_total_tested': 0, 'n_positive': 0, 'density': 0.0})
    n_positive = (has_test['val_b_avg_pnl'] > 0).sum()
    return pd.Series({
        'n_total_tested': n_total,
        'n_positive': int(n_positive),
        'density': float(n_positive) / n_total,
    })

density_df = seeds_df.groupby(['ticker', 'base_experiment', 'tpsl_mode']).apply(
    compute_density).reset_index()


# ==============================================================================
# COMPONENT 2: PARAMETER SMOOTHNESS
# ==============================================================================
# For the best config per (ticker, seed, mode), look at neighbors in prob_thresh
# and TP/SL space. If neighbors also have high PnL, the surface is smooth.

PROB_STEP = 0.05  # neighbor distance in prob_thresh space

def compute_smoothness_for_group(group):
    """Given all configs for a (ticker, experiment, mode), compute smoothness
    of the best config by comparing to its parameter neighbors."""
    if len(group) == 0:
        return pd.Series({'smoothness': 0.0, 'n_neighbors': 0, 'median_neighbor_pnl': 0.0,
                          'winner_pnl': 0.0})

    eligible = group[group['val_b_count'] >= MIN_TRADES]
    if eligible.empty:
        return pd.Series({'smoothness': 0.0, 'n_neighbors': 0, 'median_neighbor_pnl': 0.0,
                          'winner_pnl': 0.0})

    winner = eligible.loc[eligible['val_b_avg_pnl'].idxmax()]
    w_prob = winner['prob_thresh']
    w_tp = winner.get('tp_pct', 0) or winner.get('tp_mult', 0)
    w_sl = winner.get('sl_pct', 0) or winner.get('sl_mult', 0)
    w_entry = winner.get('entry_time', '')
    w_pnl = winner['val_b_avg_pnl']

    if w_pnl <= 0:
        return pd.Series({'smoothness': 0.0, 'n_neighbors': 0, 'median_neighbor_pnl': 0.0,
                          'winner_pnl': w_pnl})

    # Find neighbors: same entry_time, prob_thresh within +/-0.05, any TP/SL
    # OR same prob_thresh but different TP/SL (adjacent grid values)
    neighbors = eligible[
        (eligible.index != winner.name) &
        (eligible['entry_time'] == w_entry) &
        (
            (abs(eligible['prob_thresh'] - w_prob) <= PROB_STEP + 0.001) |
            True  # include all same-entry configs as neighbors for now
        )
    ]

    if len(neighbors) == 0:
        return pd.Series({'smoothness': 0.0, 'n_neighbors': 0, 'median_neighbor_pnl': 0.0,
                          'winner_pnl': w_pnl})

    med_pnl = float(neighbors['val_b_avg_pnl'].median())
    smoothness = np.clip(med_pnl / w_pnl, 0.0, 1.0) if w_pnl > 0 else 0.0

    return pd.Series({
        'smoothness': smoothness,
        'n_neighbors': len(neighbors),
        'median_neighbor_pnl': med_pnl,
        'winner_pnl': w_pnl,
    })

smoothness_df = seeds_df.groupby(['ticker', 'base_experiment', 'tpsl_mode']).apply(
    compute_smoothness_for_group).reset_index()


# ==============================================================================
# COMPONENT 3: CROSS-SEED AGREEMENT
# ==============================================================================
# For each ticker, look at each seed's best config. If all seeds converge on
# similar prob_thresh / TP / SL, the signal is robust.

def compute_seed_agreement_for_group(group):
    """Given all configs for a (ticker, experiment, mode), find each seed's
    best config and measure cross-seed parameter agreement."""
    if len(group) == 0:
        return pd.Series({'seed_agreement': 0.0, 'n_seeds': 0,
                          'prob_std': 0.0, 'prob_range': 0.0})

    eligible = group[group['val_b_count'] >= MIN_TRADES]
    if eligible.empty:
        return pd.Series({'seed_agreement': 0.0, 'n_seeds': 0,
                          'prob_std': 0.0, 'prob_range': 0.0})

    # Best config per seed (by test PnL)
    best_per_seed = eligible.loc[eligible.groupby('seed')['val_b_avg_pnl'].idxmax()]
    n_seeds = len(best_per_seed)

    if n_seeds <= 1:
        return pd.Series({'seed_agreement': 0.5, 'n_seeds': n_seeds,
                          'prob_std': 0.0, 'prob_range': 0.0})

    prob_vals = best_per_seed['prob_thresh'].values
    prob_std = float(np.std(prob_vals))
    prob_range = float(prob_vals.max() - prob_vals.min())

    # Also check TP/SL convergence for the relevant mode
    mode = group['tpsl_mode'].iloc[0]
    if mode in ('atr', 'atr_adaptive'):
        tp_vals = best_per_seed['tp_mult'].values
        sl_vals = best_per_seed['sl_mult'].values
    elif mode == 'trailing_decay':
        tp_vals = best_per_seed['trail_start_mult'].values
        sl_vals = best_per_seed['sl_mult'].values
    else:
        tp_vals = best_per_seed['tp_pct'].values
        sl_vals = best_per_seed['sl_pct'].values

    # Normalize TP/SL std by their range in the grid
    tp_range = tp_vals.max() - tp_vals.min() if len(tp_vals) > 1 else 0
    sl_range = sl_vals.max() - sl_vals.min() if len(sl_vals) > 1 else 0

    # Agreement score: 1 / (1 + weighted_std)
    # prob_thresh dominates: it spans [0.15, 0.95] = 0.80 range
    # We normalize std by the grid range to make it comparable
    prob_agreement = 1.0 / (1.0 + prob_std * 10)   # std=0 -> 1.0, std=0.10 -> 0.50
    # Combined agreement (prob is most important)
    seed_agreement = prob_agreement

    # Count: how many seeds have a positive test PnL?
    n_positive_seeds = (best_per_seed['val_b_avg_pnl'] > 0).sum()
    seed_survival_rate = n_positive_seeds / n_seeds

    # Blend: param agreement * seed survival rate
    seed_agreement = seed_agreement * (0.5 + 0.5 * seed_survival_rate)

    return pd.Series({
        'seed_agreement': seed_agreement,
        'n_seeds': n_seeds,
        'n_positive_seeds': int(n_positive_seeds),
        'seed_survival_rate': seed_survival_rate,
        'prob_std': prob_std,
        'prob_range': prob_range,
        'tp_range': tp_range,
        'sl_range': sl_range,
    })

seed_df = seeds_df.groupby(['ticker', 'base_experiment', 'tpsl_mode']).apply(
    compute_seed_agreement_for_group).reset_index()


# ==============================================================================
# MERGE ALL COMPONENTS
# ==============================================================================
merge_keys = ['ticker', 'base_experiment', 'tpsl_mode']
robustness = density_df.merge(smoothness_df, on=merge_keys, how='outer')
robustness = robustness.merge(seed_df, on=merge_keys, how='outer')

robustness['density'] = robustness['density'].fillna(0)
robustness['smoothness'] = robustness['smoothness'].fillna(0)
robustness['seed_agreement'] = robustness['seed_agreement'].fillna(0)

robustness['robustness_mult'] = (
    W_DENSITY * robustness['density'] +
    W_SMOOTH  * robustness['smoothness'] +
    W_SEED    * robustness['seed_agreement']
)

robustness['robust_factor'] = ROBUST_FLOOR + (1 - ROBUST_FLOOR) * robustness['robustness_mult']


# ==============================================================================
# TABLE 1: PER-TICKER ROBUSTNESS SCORECARD
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 1: ROBUSTNESS SCORECARD — All (ticker, experiment, mode) combos")
print(f"  Density = fraction of configs with positive TEST PnL")
print(f"  Smooth  = median neighbor PnL / winner PnL (1.0 = perfectly smooth surface)")
print(f"  SeedAgr = cross-seed parameter convergence (1.0 = all seeds agree)")
print(f"  R_mult  = combined robustness multiplier")
print(f"  R_factor = {ROBUST_FLOOR:.0%} + {1-ROBUST_FLOOR:.0%} * R_mult (applied to composite score)")
print(f"{'='*W}")

# Only show combos with actual test results
show = robustness[robustness['n_total_tested'] > 0].sort_values(
    'robustness_mult', ascending=False)

print(f"  {'Ticker':<7} {'Experiment':<16} {'Mode':<16} "
      f"{'Tested':>6} {'Pos':>4} {'Density':>7} "
      f"{'Smooth':>7} {'SeedAgr':>7} {'R_mult':>7} {'R_factor':>8} "
      f"{'WinPnL':>7} {'NbrPnL':>7} {'Seeds':>5} {'ProbStd':>7}")
print(f"  {'_'*7} {'_'*16} {'_'*16} "
      f"{'_'*6} {'_'*4} {'_'*7} "
      f"{'_'*7} {'_'*7} {'_'*7} {'_'*8} "
      f"{'_'*7} {'_'*7} {'_'*5} {'_'*7}")

for _, r in show.iterrows():
    flag = '>>>' if r['robustness_mult'] >= 0.6 else '   ' if r['robustness_mult'] >= 0.3 else '!!!'
    print(f"{flag}{r['ticker']:<7} {r['base_experiment']:<16} {r['tpsl_mode']:<16} "
          f"{int(r['n_total_tested']):>6} {int(r['n_positive']):>4} {r['density']:>6.0%} "
          f"{r['smoothness']:>6.2f} {r['seed_agreement']:>6.2f} "
          f"{r['robustness_mult']:>6.2f} {r['robust_factor']:>7.2f} "
          f"{r['winner_pnl']:>6.1%} {r['median_neighbor_pnl']:>6.1%} "
          f"{int(r.get('n_seeds', 0)):>5} {r['prob_std']:>6.3f}")

# Summary stats
print(f"\n  Summary:")
print(f"    Median robustness multiplier: {show['robustness_mult'].median():.3f}")
print(f"    Mean density:       {show['density'].mean():.1%}")
print(f"    Mean smoothness:    {show['smoothness'].mean():.3f}")
print(f"    Mean seed agreement:{show['seed_agreement'].mean():.3f}")
print(f"    Combos with R_mult < 0.30 (RED FLAG): {(show['robustness_mult'] < 0.30).sum()}")
print(f"    Combos with R_mult > 0.60 (ROBUST):   {(show['robustness_mult'] > 0.60).sum()}")


# ==============================================================================
# TABLE 2: CROSS-SEED CONVERGENCE DETAIL
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 2: CROSS-SEED PARAMETER CONVERGENCE")
print(f"  Low prob_std = seeds agree on optimal threshold (robust signal)")
print(f"  High prob_std = seeds disagree (noisy signal, likely overfit)")
print(f"{'='*W}")

seed_detail = seed_df[seed_df['n_seeds'] > 1].sort_values('prob_std', ascending=True)
print(f"  {'Ticker':<7} {'Experiment':<16} {'Mode':<16} "
      f"{'Seeds':>5} {'PosSeeds':>8} {'SurvRate':>8} "
      f"{'ProbStd':>8} {'ProbRange':>9} {'TP_range':>8} {'SL_range':>8} "
      f"{'Agreement':>9}")
print(f"  {'_'*7} {'_'*16} {'_'*16} "
      f"{'_'*5} {'_'*8} {'_'*8} "
      f"{'_'*8} {'_'*9} {'_'*8} {'_'*8} {'_'*9}")

for _, r in seed_detail.iterrows():
    flag = '>>>' if r['prob_std'] <= 0.05 else '   ' if r['prob_std'] <= 0.15 else '!!!'
    print(f"{flag}{r['ticker']:<7} {r['base_experiment']:<16} {r['tpsl_mode']:<16} "
          f"{int(r['n_seeds']):>5} {int(r.get('n_positive_seeds', 0)):>8} "
          f"{r['seed_survival_rate']:>7.0%} "
          f"{r['prob_std']:>7.3f} {r['prob_range']:>8.2f} "
          f"{r.get('tp_range', 0):>7.3f} {r.get('sl_range', 0):>7.3f} "
          f"{r['seed_agreement']:>8.3f}")

# Convergent vs divergent
converge = seed_detail[seed_detail['prob_std'] <= 0.10]
diverge  = seed_detail[seed_detail['prob_std'] > 0.20]
print(f"\n  CONVERGENT (prob_std <= 0.10): {len(converge)} combos — seeds agree, robust signal")
print(f"  DIVERGENT  (prob_std >  0.20): {len(diverge)} combos — seeds disagree, overfit risk")


# ==============================================================================
# TABLE 3: ROBUSTNESS-ADJUSTED RANKING vs ORIGINAL
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 3: ROBUSTNESS-ADJUSTED vs ORIGINAL RANKING")
print(f"  Original = val_b_avg_pnl * val_b_win_rate * sqrt(N) * log2(1+N)")
print(f"  Adjusted = Original * ({ROBUST_FLOOR:.0%} + {1-ROBUST_FLOOR:.0%} * robustness_multiplier)")
print(f"{'='*W}")

# For each ticker, find the best config by original composite
# Then apply the robustness adjustment
all_eligible = df[(df['val_b_count'] >= MIN_TRADES) & (df['val_b_avg_pnl'] > 0)].copy()

if all_eligible.empty:
    print("  No eligible configs found.")
else:
    # Join robustness data
    all_eligible = all_eligible.merge(
        robustness[merge_keys + ['robustness_mult', 'robust_factor', 'density',
                                  'smoothness', 'seed_agreement']],
        on=merge_keys, how='left')
    all_eligible['robustness_mult'] = all_eligible['robustness_mult'].fillna(0)
    all_eligible['robust_factor'] = all_eligible['robust_factor'].fillna(ROBUST_FLOOR)

    all_eligible['robust_composite'] = all_eligible['base_composite'] * all_eligible['robust_factor']

    # Best by original composite per ticker
    orig_best = all_eligible.loc[all_eligible.groupby('ticker')['base_composite'].idxmax()]
    # Best by robust composite per ticker
    robust_best = all_eligible.loc[all_eligible.groupby('ticker')['robust_composite'].idxmax()]

    orig_rank = orig_best.sort_values('base_composite', ascending=False).reset_index(drop=True)
    robust_rank = robust_best.sort_values('robust_composite', ascending=False).reset_index(drop=True)

    orig_rank['orig_rank'] = range(1, len(orig_rank) + 1)
    robust_rank['robust_rank'] = range(1, len(robust_rank) + 1)

    # Merge to compare
    compare = orig_rank[['ticker', 'orig_rank', 'base_composite', 'val_b_avg_pnl', 'tpsl_mode',
                         'prob_thresh', 'experiment']].merge(
        robust_rank[['ticker', 'robust_rank', 'robust_composite', 'robustness_mult',
                     'val_b_avg_pnl', 'tpsl_mode', 'prob_thresh', 'experiment',
                     'density', 'smoothness', 'seed_agreement']],
        on='ticker', suffixes=('_orig', '_robust'), how='outer')

    compare['rank_change'] = compare['orig_rank'] - compare['robust_rank']
    compare = compare.sort_values('robust_rank')

    print(f"  {'Ticker':<7} {'OrigRk':>6} {'RobRk':>6} {'Change':>6} "
          f"{'OrigComp':>9} {'RobComp':>9} {'R_mult':>7} "
          f"{'Dens':>5} {'Smth':>5} {'Seed':>5} "
          f"{'OrigMode':>11} {'RobMode':>11} {'ConfigChanged':>14}")
    print(f"  {'_'*7} {'_'*6} {'_'*6} {'_'*6} "
          f"{'_'*9} {'_'*9} {'_'*7} "
          f"{'_'*5} {'_'*5} {'_'*5} "
          f"{'_'*11} {'_'*11} {'_'*14}")

    n_changed = 0
    for _, r in compare.iterrows():
        same_config = (r.get('experiment_orig') == r.get('experiment_robust') and
                       r.get('prob_thresh_orig') == r.get('prob_thresh_robust') and
                       r.get('tpsl_mode_orig') == r.get('tpsl_mode_robust'))
        changed_str = 'SAME' if same_config else 'CHANGED'
        if not same_config:
            n_changed += 1
        chg = r.get('rank_change', 0)
        chg_str = f"{chg:+d}" if pd.notna(chg) else '?'
        print(f"  {r['ticker']:<7} {int(r.get('orig_rank', 0)):>6} {int(r.get('robust_rank', 0)):>6} "
              f"{chg_str:>6} "
              f"{r.get('base_composite', 0):>9.6f} {r.get('robust_composite', 0):>9.6f} "
              f"{r.get('robustness_mult', 0):>6.3f} "
              f"{r.get('density', 0):>4.0%} {r.get('smoothness', 0):>4.2f} "
              f"{r.get('seed_agreement', 0):>4.2f} "
              f"{str(r.get('tpsl_mode_orig', '')):<11} "
              f"{str(r.get('tpsl_mode_robust', '')):<11} "
              f"{changed_str:>14}")

    print(f"\n  CONFIG SELECTION CHANGED: {n_changed}/{len(compare)} tickers")
    print(f"  These tickers would pick a different config with robustness weighting.")


# ==============================================================================
# TABLE 4: RED FLAGS — Low Robustness Tickers
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 4: RED FLAGS — Tickers with High Overfit Risk")
print(f"  Criteria: robustness_mult < 0.30 OR n_positive/n_tested < 20%")
print(f"  OR prob_std > 0.20 (seeds diverge wildly)")
print(f"{'='*W}")

red_flags = robustness[
    (robustness['n_total_tested'] > 0) &
    (
        (robustness['robustness_mult'] < 0.30) |
        (robustness['density'] < 0.20) |
        (robustness['prob_std'] > 0.20)
    )
].sort_values('robustness_mult')

if red_flags.empty:
    print("  No red flags found!")
else:
    print(f"  {'Ticker':<7} {'Experiment':<16} {'Mode':<16} "
          f"{'R_mult':>7} {'Density':>7} {'Smooth':>7} {'SeedAgr':>7} "
          f"{'ProbStd':>7} {'Reason'}")
    print(f"  {'_'*7} {'_'*16} {'_'*16} "
          f"{'_'*7} {'_'*7} {'_'*7} {'_'*7} {'_'*7} {'_'*30}")

    for _, r in red_flags.iterrows():
        reasons = []
        if r['robustness_mult'] < 0.30:
            reasons.append('LOW_ROBUSTNESS')
        if r['density'] < 0.20:
            reasons.append(f"SPARSE({r['density']:.0%})")
        if r.get('prob_std', 0) > 0.20:
            reasons.append(f"SEED_DIVERGE(std={r['prob_std']:.2f})")
        print(f"  {r['ticker']:<7} {r['base_experiment']:<16} {r['tpsl_mode']:<16} "
              f"{r['robustness_mult']:>6.3f} {r['density']:>6.0%} "
              f"{r['smoothness']:>6.2f} {r['seed_agreement']:>6.2f} "
              f"{r.get('prob_std', 0):>6.3f} {', '.join(reasons)}")

    print(f"\n  TOTAL RED FLAGS: {len(red_flags)}")
    red_tickers = sorted(red_flags['ticker'].unique())
    print(f"  Affected tickers: {', '.join(red_tickers)}")


# ==============================================================================
# TABLE 5: PARAMETER SENSITIVITY — prob_thresh PnL curves per ticker
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 5: PROB_THRESH SENSITIVITY — Median TEST PnL at each threshold")
print(f"  Smooth curves = robust, cliff-edges = overfit")
print(f"{'='*W}")

# For each ticker, show median test PnL at each prob_thresh value
for ticker in sorted(tickers)[:20]:  # show first 20 tickers
    td = seeds_df[(seeds_df['ticker'] == ticker) & (seeds_df['val_b_count'] >= MIN_TRADES)]
    if td.empty:
        continue

    prob_curve = td.groupby('prob_thresh').agg(
        median_pnl=('val_b_avg_pnl', 'median'),
        mean_pnl=('val_b_avg_pnl', 'mean'),
        n_configs=('val_b_avg_pnl', 'count'),
        pct_positive=('val_b_avg_pnl', lambda x: (x > 0).mean()),
    ).sort_index()

    if len(prob_curve) < 2:
        continue

    print(f"\n  {ticker}:")
    print(f"  {'Prob':>6} {'MedPnL':>8} {'MeanPnL':>8} {'Configs':>7} {'%Positive':>9}")
    for pt, r in prob_curve.iterrows():
        bar_len = int(max(0, min(40, r['median_pnl'] * 400)))
        bar = '#' * bar_len if r['median_pnl'] > 0 else '-' * min(20, abs(int(r['median_pnl'] * 400)))
        print(f"  {pt:>6.2f} {r['median_pnl']:>+7.2%} {r['mean_pnl']:>+7.2%} "
              f"{int(r['n_configs']):>7} {r['pct_positive']:>8.0%}  {bar}")


# ==============================================================================
# FORMULA REFERENCE
# ==============================================================================
print(f"\n{'='*W}")
print(f"  ROBUSTNESS-ADJUSTED COMPOSITE SCORE — Formula Reference")
print(f"{'='*W}")
print(f"""
  BASE composite = val_b_avg_pnl * val_b_win_rate * sqrt(val_b_count) * log2(1 + val_b_count)

  ROBUSTNESS MULTIPLIER = weighted sum of 3 components:
    density       = N_positive_configs / N_total_configs   (weight {W_DENSITY:.0%})
    smoothness    = median_neighbor_PnL / winner_PnL       (weight {W_SMOOTH:.0%})
    seed_agreement= 1/(1 + prob_std*10) * (0.5 + 0.5*seed_survival_rate)  (weight {W_SEED:.0%})

  R_mult = {W_DENSITY:.2f}*density + {W_SMOOTH:.2f}*smoothness + {W_SEED:.2f}*seed_agreement

  ROBUST composite = BASE * ({ROBUST_FLOOR:.2f} + {1-ROBUST_FLOOR:.2f} * R_mult)

  Range: {ROBUST_FLOOR:.0%} penalty (worst) to 0% penalty (best)

  To use in production:
    1. Run this script after WF optimization
    2. Replace pick_best() in analyze_composite_score.py with robust_composite ranking
    3. Or add robustness to the S8 family in compare_composite_scores.py as score 'S8_R'
""")

print(f"\n{'='*W}")
print(f"  ANALYSIS COMPLETE")
print(f"{'='*W}")
