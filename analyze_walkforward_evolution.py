"""
analyze_walkforward_evolution.py — Model Development Progress Tracker
======================================================================
Compares walkforward generations and single-fold baselines to evaluate
whether model development is improving results over time.

Families (all share TEST period Apr 1 - Jun 17, 2026):
  152 (late cutoff, Jul 2025) — baseline single-fold transformer
  160 (early cutoff, Apr 2025) — baseline single-fold transformer
  170-175 WF (6 folds, cutoffs Apr-Jan 2026) — standard transformer
  180-185 WF (6 folds, cutoffs Aug-Jan 2026) — standard transformer, later data

Sections:
  1. Data Census
  2. Model Development Timeline — summary progression
  3. 170 vs 180 Direct Comparison — did later cutoffs help?
  4. Walkforward vs Single-Fold — is WF training better?
  5. Per-Ticker Progression — "report card" across generations
  6. Limit Entry Evolution — how limit_entry changed
  7. Deployable Portfolio Comparison
  8. Overfitting Diagnostics — OPT→TEST degradation
  9. Backtest (optional)
  10. Verdict

Colab-ready. BQ-only for sections 1-8.
"""

import gc
import io
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass

from google.cloud import bigquery

# ==============================================================================
# CONFIG
# ==============================================================================
BQ_TABLE       = 'ys-time-series.scoring_logs.wf_optimization_results'
MODEL_VERSION  = 'robust_wf_v12'
DEPLOY_THRESHOLD = 0.018
MIN_TRADES     = 3
RUN_BACKTEST   = False

W = 130

ALL_MODES = ['static', 'atr', 'atr_adaptive', 'regime_scaled', 'trailing_decay',
             'regime_adaptive', 'atr_sl', 'grace', 'limit_entry']

MODE_SUFFIX = {'static': '', 'atr': '_ATR', 'atr_adaptive': '_ATRADP',
               'regime_scaled': '_REGIME', 'trailing_decay': '_TRAIL',
               'regime_adaptive': '_REGADP', 'atr_sl': '_ATRSL',
               'grace': '_GRACE', 'limit_entry': '_LIMIT'}

FAM_ORDER = ['152', '160', '170', '180']
FAM_LABELS = {
    '152': '152 Late',
    '160': '160 Early',
    '170': '170-175 WF',
    '180': '180-185 WF',
}
FAM_CUTOFFS = {
    '152': 'Jul 2025',
    '160': 'Apr 2025',
    '170': 'Apr-Jan 2026 (6 folds)',
    '180': 'Aug-Jan 2026 (6 folds)',
}

client = bigquery.Client()


# ==============================================================================
# LOAD DATA
# ==============================================================================
print(f"\n{'='*W}")
print(f"  WALKFORWARD EVOLUTION — Model Development Progress Tracker")
print(f"  TEST period: Apr 1 - Jun 17, 2026 (shared by all families)")
print(f"  Model version: {MODEL_VERSION}")
print(f"{'='*W}")

query = f"""
SELECT experiment, ticker, seed, tpsl_mode, is_ensemble, n_seeds_avg,
       opt_avg_pnl, opt_win_rate, opt_count,
       val_a_avg_pnl, val_a_win_rate, val_a_count, val_a_freq,
       val_b_avg_pnl, val_b_win_rate, val_b_count, val_b_freq,
       survived_a, survived_b,
       prob_thresh, tp_pct, sl_pct, tp_mult, sl_mult,
       trail_start_mult, trail_end_mult, vol_filter_pct, pre_range_filter_pct,
       grace_minutes, dip_pct, max_wait_minutes,
       opt_start, opt_end, run_timestamp
FROM `{BQ_TABLE}`
WHERE model_version = '{MODEL_VERSION}'
  AND (experiment LIKE '152_%' OR experiment LIKE '160_%'
       OR experiment LIKE '170_%' OR experiment LIKE '171_%'
       OR experiment LIKE '172_%' OR experiment LIKE '173_%'
       OR experiment LIKE '174_%' OR experiment LIKE '175_%'
       OR experiment LIKE '180_%' OR experiment LIKE '181_%'
       OR experiment LIKE '182_%' OR experiment LIKE '183_%'
       OR experiment LIKE '184_%' OR experiment LIKE '185_%')
"""

df_all = client.query(query).to_dataframe()
print(f"  Loaded {len(df_all)} rows from BQ")

df_all['tpsl_mode'] = df_all['tpsl_mode'].fillna('static')
df_all['survived_b'] = df_all['survived_b'].fillna(False).astype(bool)
df_all['survived_a'] = df_all['survived_a'].fillna(False).astype(bool)
df_all['val_b_avg_pnl'] = df_all['val_b_avg_pnl'].fillna(0.0)
df_all['val_b_win_rate'] = df_all['val_b_win_rate'].fillna(0.0)
df_all['val_b_count'] = df_all['val_b_count'].fillna(0).astype(int)
df_all['opt_avg_pnl'] = df_all['opt_avg_pnl'].fillna(0.0)
df_all['opt_count'] = df_all['opt_count'].fillna(0).astype(int)
for col in ['dip_pct', 'grace_minutes', 'max_wait_minutes']:
    if col in df_all.columns:
        df_all[col] = df_all[col].fillna(0)


def assign_family(exp):
    base = exp.split('_')[0]
    if not base.isdigit():
        return 'other'
    num = int(base)
    if num == 152: return '152'
    if num == 160: return '160'
    if 170 <= num <= 175: return '170'
    if 180 <= num <= 185: return '180'
    return 'other'


df_all['family'] = df_all['experiment'].apply(assign_family)
df_all = df_all[df_all['family'] != 'other']

ens = df_all[df_all['is_ensemble'] == True].copy()
seeds = df_all[df_all['is_ensemble'] != True].copy()


# ==============================================================================
# SECTION 1: DATA CENSUS
# ==============================================================================
print(f"\n{'='*W}")
print(f"  SECTION 1: DATA CENSUS")
print(f"{'='*W}")

print(f"\n  {'Family':<14} {'Cutoff':<24} {'Rows':>7} {'Seeds':>7} {'Tickers':>8} "
      f"{'Modes':>6} {'Ensemble':>9}")
print(f"  {'─'*80}")

for fam in FAM_ORDER:
    label = FAM_LABELS[fam]
    cutoff = FAM_CUTOFFS[fam]
    fam_df = df_all[df_all['family'] == fam]
    fam_ind = fam_df[fam_df['is_ensemble'] != True]
    print(f"  {label:<14} {cutoff:<24} {len(fam_df):>7} {fam_ind['seed'].nunique():>7} "
          f"{fam_df['ticker'].nunique():>8} {fam_df['tpsl_mode'].nunique():>6} "
          f"{len(fam_df[fam_df['is_ensemble']==True]):>9}")


# ==============================================================================
# SECTION 2: MODEL DEVELOPMENT TIMELINE
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  SECTION 2: MODEL DEVELOPMENT TIMELINE")
print(f"  Are we getting better over time?")
print(f"{'='*W}")

print(f"\n  {'Family':<14} {'Cutoff':<20} {'Avg Test':>9} {'Lim Test':>9} "
      f"{'Pass%':>7} {'Deploy':>7} {'Best Ticker':>30}")
print(f"  {'─'*100}")

for fam in FAM_ORDER:
    label = FAM_LABELS[fam]
    cutoff = FAM_CUTOFFS[fam][:16]
    fam_ens = ens[ens['family'] == fam]
    if len(fam_ens) == 0:
        print(f"  {label:<14} {cutoff:<20} {'—':>9} {'—':>9} {'—':>7} {'—':>7}")
        continue

    avg_all = fam_ens['val_b_avg_pnl'].mean()
    lim_ens = fam_ens[fam_ens['tpsl_mode'] == 'limit_entry']
    avg_lim = lim_ens['val_b_avg_pnl'].mean() if len(lim_ens) > 0 else 0
    pass_rate = fam_ens['survived_b'].mean()
    deployable = fam_ens[fam_ens['val_b_avg_pnl'] >= DEPLOY_THRESHOLD]['ticker'].nunique()
    total = fam_ens['ticker'].nunique()

    best_ticker_row = fam_ens.loc[fam_ens['val_b_avg_pnl'].idxmax()]
    best_str = f"{best_ticker_row['ticker']} {best_ticker_row['val_b_avg_pnl']:>+.2%} ({best_ticker_row['tpsl_mode']})"

    print(f"  {label:<14} {cutoff:<20} {avg_all:>+8.2%} {avg_lim:>+8.2%} "
          f"{pass_rate:>6.0%} {deployable:>3}/{total:<3} {best_str:>30}")

# Direction arrows
print(f"\n  Progression (all modes → limit_entry):")
prev_all = None
prev_lim = None
for fam in FAM_ORDER:
    fam_ens = ens[ens['family'] == fam]
    if len(fam_ens) == 0: continue
    avg_all = fam_ens['val_b_avg_pnl'].mean()
    lim_ens = fam_ens[fam_ens['tpsl_mode'] == 'limit_entry']
    avg_lim = lim_ens['val_b_avg_pnl'].mean() if len(lim_ens) > 0 else 0

    if prev_all is not None:
        d_all = avg_all - prev_all
        d_lim = avg_lim - prev_lim
        arrow_all = "↑" if d_all > 0.001 else ("↓" if d_all < -0.001 else "→")
        arrow_lim = "↑" if d_lim > 0.001 else ("↓" if d_lim < -0.001 else "→")
        print(f"    {FAM_LABELS[fam]}: all={d_all:>+.2%} {arrow_all}  "
              f"limit={d_lim:>+.2%} {arrow_lim}")
    prev_all = avg_all
    prev_lim = avg_lim


# ==============================================================================
# SECTION 3: 170 vs 180 DIRECT COMPARISON
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  SECTION 3: 170-175 WF vs 180-185 WF — DIRECT COMPARISON")
print(f"  Same architecture, different training cutoffs")
print(f"{'='*W}")

# Per-mode summary
mode_summary = ens[ens['family'].isin(['170', '180'])].groupby(['family', 'tpsl_mode']).agg(
    tickers=('ticker', 'nunique'),
    avg_test_pnl=('val_b_avg_pnl', 'mean'),
    avg_test_wr=('val_b_win_rate', 'mean'),
    pct_pass=('survived_b', 'mean'),
    avg_test_n=('val_b_count', 'mean'),
).reset_index()

print(f"\n  {'Mode':<18}  | {'170-175 WF':^30}  | {'180-185 WF':^30}  | {'Delta':>7}")
print(f"  {'':─<18}──|{'':─^30}──|{'':─^30}──|{'':─^7}")
print(f"  {'':18}  | {'Tix':>4} {'PnL':>7} {'WR':>6} {'Pass%':>6}  | "
      f"{'Tix':>4} {'PnL':>7} {'WR':>6} {'Pass%':>6}  |")
print(f"  {'':─<18}──|{'':─^30}──|{'':─^30}──|{'':─^7}")

for mode in ALL_MODES:
    print(f"  {mode:<18}", end='')
    pnl_170 = None
    pnl_180 = None
    for fam in ['170', '180']:
        row = mode_summary[(mode_summary['family'] == fam) & (mode_summary['tpsl_mode'] == mode)]
        if len(row) == 0:
            print(f"  | {'—':>4} {'—':>7} {'—':>6} {'—':>6}", end='')
        else:
            r = row.iloc[0]
            pnl_val = r['avg_test_pnl']
            if fam == '170': pnl_170 = pnl_val
            else: pnl_180 = pnl_val
            print(f"  | {int(r['tickers']):>4} {pnl_val:>+6.2%} {r['avg_test_wr']:>5.1%} "
                  f"{r['pct_pass']:>5.0%}", end='')
    if pnl_170 is not None and pnl_180 is not None:
        delta = pnl_180 - pnl_170
        print(f"  | {delta:>+6.2%}")
    else:
        print(f"  | {'—':>7}")

print(f"  {'':─<18}──|{'':─^30}──|{'':─^30}──|{'':─^7}")

# Head-to-head per ticker
best_170 = ens[ens['family'] == '170'].copy()
best_180 = ens[ens['family'] == '180'].copy()

if len(best_170) > 0 and len(best_180) > 0:
    best_170_tk = best_170.loc[best_170.groupby('ticker')['val_b_avg_pnl'].idxmax()]
    best_180_tk = best_180.loc[best_180.groupby('ticker')['val_b_avg_pnl'].idxmax()]

    p170 = best_170_tk.set_index('ticker')['val_b_avg_pnl']
    p180 = best_180_tk.set_index('ticker')['val_b_avg_pnl']
    m170 = best_170_tk.set_index('ticker')['tpsl_mode']
    m180 = best_180_tk.set_index('ticker')['tpsl_mode']

    common_wf = sorted(set(p170.index) & set(p180.index))
    print(f"\n  Head-to-head: {len(common_wf)} common tickers")

    if common_wf:
        print(f"\n  {'Ticker':<7}  {'170-175 WF':>20}  {'180-185 WF':>20}  {'Delta':>8} {'Winner':>10}")
        print(f"  {'':7}   {'Mode':>12} {'PnL':>7}  {'Mode':>12} {'PnL':>7}")
        print(f"  {'─'*75}")

        wins_wf = {'170': 0, '180': 0, 'tie': 0}
        for t in common_wf:
            v170 = p170[t]; v180 = p180[t]
            delta = v180 - v170
            winner = '180' if delta > 0.001 else ('170' if delta < -0.001 else 'tie')
            wins_wf[winner] += 1
            w_label = FAM_LABELS.get(winner, 'Tie')[:8]
            print(f"  {t:<7}  {m170[t]:>12} {v170:>+6.2%}  {m180[t]:>12} {v180:>+6.2%}"
                  f"  {delta:>+7.2%} {w_label:>10}")

        print(f"  {'─'*75}")
        d_series = pd.Series({t: p180[t] - p170[t] for t in common_wf})
        print(f"\n  WINS: 180 WF={wins_wf['180']}  170 WF={wins_wf['170']}  Tie={wins_wf['tie']}")
        print(f"  Avg delta (180-170): {d_series.mean():>+.3%}  Med: {d_series.median():>+.3%}")


# ==============================================================================
# SECTION 4: WALKFORWARD vs SINGLE-FOLD
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  SECTION 4: WALKFORWARD vs SINGLE-FOLD BASELINES")
print(f"  Do walkforward models outperform single-fold?")
print(f"{'='*W}")

comparisons = [
    ('170', '152', '170-175 WF vs 152 Late'),
    ('170', '160', '170-175 WF vs 160 Early'),
    ('180', '152', '180-185 WF vs 152 Late'),
    ('180', '160', '180-185 WF vs 160 Early'),
]

for wf_fam, sf_fam, label in comparisons:
    wf_ens_df = ens[ens['family'] == wf_fam]
    sf_ens_df = ens[ens['family'] == sf_fam]
    if len(wf_ens_df) == 0 or len(sf_ens_df) == 0:
        print(f"\n  {label}: insufficient data")
        continue

    wf_best = wf_ens_df.loc[wf_ens_df.groupby('ticker')['val_b_avg_pnl'].idxmax()].set_index('ticker')
    sf_best = sf_ens_df.loc[sf_ens_df.groupby('ticker')['val_b_avg_pnl'].idxmax()].set_index('ticker')
    common = sorted(set(wf_best.index) & set(sf_best.index))

    wf_deploy = wf_ens_df[wf_ens_df['val_b_avg_pnl'] >= DEPLOY_THRESHOLD]['ticker'].nunique()
    sf_deploy = sf_ens_df[sf_ens_df['val_b_avg_pnl'] >= DEPLOY_THRESHOLD]['ticker'].nunique()

    print(f"\n  {label} ({len(common)} common tickers):")
    print(f"    {FAM_LABELS[wf_fam]:>14}: avg={wf_ens_df['val_b_avg_pnl'].mean():>+.2%}  "
          f"deploy={wf_deploy}  pass={wf_ens_df['survived_b'].mean():.0%}")
    print(f"    {FAM_LABELS[sf_fam]:>14}: avg={sf_ens_df['val_b_avg_pnl'].mean():>+.2%}  "
          f"deploy={sf_deploy}  pass={sf_ens_df['survived_b'].mean():.0%}")

    if common:
        wf_vals = wf_best.loc[common, 'val_b_avg_pnl']
        sf_vals = sf_best.loc[common, 'val_b_avg_pnl']
        delta = wf_vals - sf_vals
        n_better = (delta > 0).sum()
        print(f"    Paired ({len(common)} tickers): {FAM_LABELS[wf_fam]} wins {n_better}/{len(common)}  "
              f"delta={delta.mean():>+.3%}")


# ==============================================================================
# SECTION 5: PER-TICKER PROGRESSION
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  SECTION 5: PER-TICKER PROGRESSION — Report Card")
print(f"  Best-mode test PnL across all 4 families")
print(f"{'='*W}")

best_per_fam_tk = ens.loc[ens.groupby(['family', 'ticker'])['val_b_avg_pnl'].idxmax()].copy()
pivot_prog = best_per_fam_tk.pivot(index='ticker', columns='family', values='val_b_avg_pnl')

all_4_tickers = sorted(pivot_prog[FAM_ORDER].dropna().index)
any_tickers = sorted(pivot_prog.index)

print(f"\n  Tickers in ALL 4 families: {len(all_4_tickers)}")
print(f"  Tickers in ANY family: {len(any_tickers)}")

print(f"\n  {'Ticker':<7}", end='')
for fam in FAM_ORDER:
    print(f" {FAM_LABELS[fam]:>12}", end='')
print(f"  {'Trend':>6} {'Best':>12}")
print(f"  {'─'*75}")

n_improving = 0
n_degrading = 0
n_flat = 0

for t in any_tickers:
    print(f"  {t:<7}", end='')
    vals = {}
    for fam in FAM_ORDER:
        v = pivot_prog.loc[t].get(fam)
        if pd.isna(v) or v is None:
            print(f" {'—':>12}", end='')
        else:
            vals[fam] = v
            print(f" {v:>+11.2%}", end='')

    # Determine trend
    if len(vals) >= 2:
        ordered_vals = [vals[f] for f in FAM_ORDER if f in vals]
        first_half = np.mean(ordered_vals[:len(ordered_vals)//2])
        second_half = np.mean(ordered_vals[len(ordered_vals)//2:])
        diff = second_half - first_half
        if diff > 0.005:
            trend = "↑"
            n_improving += 1
        elif diff < -0.005:
            trend = "↓"
            n_degrading += 1
        else:
            trend = "→"
            n_flat += 1
    else:
        trend = "?"

    best_fam = max(vals, key=vals.get) if vals else '?'
    print(f"  {trend:>6} {FAM_LABELS.get(best_fam, best_fam):>12}")

print(f"  {'─'*75}")
print(f"  Improving: {n_improving}  Degrading: {n_degrading}  Flat: {n_flat}")


# ==============================================================================
# SECTION 6: LIMIT_ENTRY EVOLUTION
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  SECTION 6: LIMIT_ENTRY EVOLUTION")
print(f"{'='*W}")

limit_ens = ens[ens['tpsl_mode'] == 'limit_entry'].copy()

print(f"\n  {'Family':<14} {'Tickers':>8} {'Avg PnL':>9} {'Med PnL':>9} "
      f"{'Pass%':>7} {'Deploy':>7} {'Avg N':>6}")
print(f"  {'─'*65}")

deploy_sets = {}
for fam in FAM_ORDER:
    fam_lim = limit_ens[limit_ens['family'] == fam]
    if len(fam_lim) == 0:
        print(f"  {FAM_LABELS[fam]:<14} {'—':>8}")
        deploy_sets[fam] = set()
        continue
    deploy_tickers = set(fam_lim[fam_lim['val_b_avg_pnl'] >= DEPLOY_THRESHOLD]['ticker'].unique())
    deploy_sets[fam] = deploy_tickers
    print(f"  {FAM_LABELS[fam]:<14} {fam_lim['ticker'].nunique():>8} "
          f"{fam_lim['val_b_avg_pnl'].mean():>+8.2%} {fam_lim['val_b_avg_pnl'].median():>+8.2%} "
          f"{fam_lim['survived_b'].mean():>6.0%} "
          f"{len(deploy_tickers):>7} {fam_lim['val_b_count'].mean():>6.1f}")

# New/lost tickers between generations
print(f"\n  Deployable ticker changes:")
for i in range(1, len(FAM_ORDER)):
    prev = FAM_ORDER[i-1]
    curr = FAM_ORDER[i]
    gained = deploy_sets[curr] - deploy_sets[prev]
    lost = deploy_sets[prev] - deploy_sets[curr]
    if gained or lost:
        print(f"    {FAM_LABELS[prev]} → {FAM_LABELS[curr]}:")
        if gained:
            print(f"      GAINED ({len(gained)}): {', '.join(sorted(gained))}")
        if lost:
            print(f"      LOST   ({len(lost)}): {', '.join(sorted(lost))}")

# Dip_pct distribution evolution
limit_seeds_data = seeds[seeds['tpsl_mode'] == 'limit_entry'].copy()
if 'dip_pct' in limit_seeds_data.columns:
    print(f"\n  Dip_pct median by family (survivors with test PnL > 0):")
    for fam in FAM_ORDER:
        fs = limit_seeds_data[(limit_seeds_data['family'] == fam) &
                               (limit_seeds_data['val_b_avg_pnl'] > 0)]
        if len(fs) > 0 and fs['dip_pct'].notna().any():
            med_dip = fs['dip_pct'].median()
            mean_dip = fs['dip_pct'].mean()
            print(f"    {FAM_LABELS[fam]:<14} med={med_dip:.2%}  mean={mean_dip:.2%}  N={len(fs)}")


# ==============================================================================
# SECTION 7: DEPLOYABLE PORTFOLIO COMPARISON
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  SECTION 7: DEPLOYABLE PORTFOLIO COMPARISON")
print(f"  Tickers with test PnL >= {DEPLOY_THRESHOLD:.1%} (ensemble, best mode)")
print(f"{'='*W}")

for fam in FAM_ORDER:
    fam_ens = ens[ens['family'] == fam]
    deployable = fam_ens[fam_ens['val_b_avg_pnl'] >= DEPLOY_THRESHOLD].copy()
    if len(deployable) == 0:
        print(f"\n  {FAM_LABELS[fam]}: (none)")
        continue
    deploy_best = deployable.loc[deployable.groupby('ticker')['val_b_avg_pnl'].idxmax()]
    deploy_best = deploy_best.sort_values('val_b_avg_pnl', ascending=False)
    total = fam_ens['ticker'].nunique()

    unique_to_fam = set(deploy_best['ticker'].unique())
    for other_fam in FAM_ORDER:
        if other_fam == fam:
            continue
        other_deploy = set(ens[(ens['family'] == other_fam) &
                                (ens['val_b_avg_pnl'] >= DEPLOY_THRESHOLD)]['ticker'].unique())
        unique_to_fam -= other_deploy

    print(f"\n  {FAM_LABELS[fam]} — {len(deploy_best)}/{total} tickers deployable "
          f"({len(unique_to_fam)} unique to this family):")
    print(f"    {'Ticker':<7} {'Mode':<15} {'TestPnL':>8} {'TestWR':>7} {'TestN':>6} {'Unique':>7}")
    print(f"    {'─'*55}")
    for _, r in deploy_best.iterrows():
        is_unique = r['ticker'] in unique_to_fam
        print(f"    {r['ticker']:<7} {r['tpsl_mode']:<15} {r['val_b_avg_pnl']:>+7.2%} "
              f"{r['val_b_win_rate']:>6.1%} {int(r['val_b_count']):>6} "
              f"{'UNIQUE' if is_unique else '':>7}")
    if unique_to_fam:
        print(f"    Unique tickers: {', '.join(sorted(unique_to_fam))}")

# Portfolio-level comparison
print(f"\n  Portfolio-level (if deploying best config per ticker from each family):")
print(f"  {'Family':<14} {'Tickers':>8} {'Avg PnL':>9} {'Med PnL':>9} "
      f"{'Avg Trades':>11} {'Avg WR':>7}")
print(f"  {'─'*62}")

for fam in FAM_ORDER:
    fam_ens = ens[ens['family'] == fam]
    deployable = fam_ens[fam_ens['val_b_avg_pnl'] >= DEPLOY_THRESHOLD]
    if len(deployable) == 0:
        print(f"  {FAM_LABELS[fam]:<14} {'—':>8}")
        continue
    deploy_best = deployable.loc[deployable.groupby('ticker')['val_b_avg_pnl'].idxmax()]
    print(f"  {FAM_LABELS[fam]:<14} {len(deploy_best):>8} "
          f"{deploy_best['val_b_avg_pnl'].mean():>+8.2%} "
          f"{deploy_best['val_b_avg_pnl'].median():>+8.2%} "
          f"{deploy_best['val_b_count'].mean():>11.1f} "
          f"{deploy_best['val_b_win_rate'].mean():>6.0%}")


# ==============================================================================
# SECTION 8: OVERFITTING DIAGNOSTICS
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  SECTION 8: OVERFITTING DIAGNOSTICS — OPT → TEST Degradation")
print(f"{'='*W}")

print(f"\n  {'Family':<14} {'Mode':<18} {'Avg OPT':>9} {'Avg TEST':>9} "
      f"{'Degrad':>8} {'K/N':>6} {'Noise':>7}")
print(f"  {'─'*75}")

sigma = 0.10
for fam in FAM_ORDER:
    fam_seeds = seeds[seeds['family'] == fam]
    if len(fam_seeds) == 0:
        continue
    for mode in ALL_MODES:
        mode_data = fam_seeds[fam_seeds['tpsl_mode'] == mode]
        if len(mode_data) == 0:
            continue
        avg_opt = mode_data['opt_avg_pnl'].mean()
        avg_test = mode_data['val_b_avg_pnl'].mean()
        avg_n = mode_data['opt_count'].mean()

        if mode == 'regime_adaptive':
            n_configs = 5
        elif mode == 'grace':
            n_configs = 5
        elif mode == 'limit_entry':
            n_configs = 24
        else:
            n_configs = {'static': 8, 'regime_scaled': 8, 'atr': 5,
                         'atr_adaptive': 5, 'trailing_decay': 5, 'atr_sl': 5}.get(mode, 5)

        k_tested = len([0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]) * n_configs
        kn = k_tested / max(1, avg_n)
        noise = sigma / np.sqrt(max(1, avg_n)) * np.sqrt(2 * np.log(max(2, k_tested)))
        degrad = 1 - avg_test / avg_opt if avg_opt > 0 else 0

        flag = " !" if kn > 3.0 else ""
        print(f"  {FAM_LABELS[fam] if mode == ALL_MODES[0] else '':>14} {mode:<18} "
              f"{avg_opt:>+8.2%} {avg_test:>+8.2%} "
              f"{degrad:>+7.0%} {kn:>5.1f} {noise:>6.1%}{flag}")

# Summary by family
print(f"\n  Summary:")
print(f"  {'Family':<14} {'Avg OPT':>9} {'Avg TEST':>9} {'Degrad%':>8} {'K/N':>6}")
print(f"  {'─'*50}")
for fam in FAM_ORDER:
    fam_seeds = seeds[seeds['family'] == fam]
    if len(fam_seeds) == 0:
        continue
    avg_opt = fam_seeds['opt_avg_pnl'].mean()
    avg_test = fam_seeds['val_b_avg_pnl'].mean()
    avg_n = fam_seeds['opt_count'].mean()
    degrad = 1 - avg_test / avg_opt if avg_opt > 0 else 0
    k_est = 12 * 8
    kn = k_est / max(1, avg_n)
    print(f"  {FAM_LABELS[fam]:<14} {avg_opt:>+8.2%} {avg_test:>+8.2%} "
          f"{degrad:>+7.0%} {kn:>5.1f}")


# ==============================================================================
# SECTION 9: BACKTEST (optional)
# ==============================================================================
if RUN_BACKTEST:
    print(f"\n\n{'='*W}")
    print(f"  SECTION 9: BACKTEST — Top 180-185 WF configs with CSV predictions")
    print(f"{'='*W}")

    try:
        import pytz
        from datetime import timedelta
        from google.cloud import storage as gcs_storage
        from numba import jit

        BUCKET_NAME = "time-series-bucket-ys"
        INFERENCE_DIR = "Data/Inference_v2"
        TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}
        EXIT_TIME_MIN = 780
        SLIPPAGE_PCT = 0.0015
        PST = pytz.timezone('America/Los_Angeles')

        BT_TEST_START = '2026-04-01'
        BT_TEST_END   = '2026-06-17'

        WALKFORWARD_FOLDS_180 = ['180_XSECTOR', '181_XSECTOR', '182_XSECTOR',
                                  '183_XSECTOR', '184_XSECTOR', '185_XSECTOR']
        WALKFORWARD_FOLDS_170 = ['170_XSECTOR', '171_XSECTOR', '172_XSECTOR',
                                  '173_XSECTOR', '174_XSECTOR', '175_XSECTOR']

        gcs_client = gcs_storage.Client()

        def load_master(ticker, start_date=None):
            clean = TICKER_FILE_MAP.get(ticker, ticker)
            uri = f"gs://{BUCKET_NAME}/{INFERENCE_DIR}/{clean}_master_inference.parquet"
            try:
                df = pd.read_parquet(uri).reset_index()
                dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
                if dt_cols:
                    df['Datetime'] = pd.to_datetime(df[dt_cols[0]], utc=True)
                else:
                    nc = next((c for c in df.columns if c in ('Datetime','Date','Time','Timestamp')), df.columns[0])
                    df['Datetime'] = pd.to_datetime(df[nc], unit='ms', utc=True)
                df = df.set_index('Datetime').sort_index()
                df = df[~df.index.duplicated(keep='first')]
                for c in ['Open','High','Low','Close']:
                    if c in df.columns: df[c] = df[c].ffill()
                for c in ['Volume']:
                    if c in df.columns: df[c] = df[c].fillna(0)
                if start_date:
                    cutoff = pd.Timestamp(start_date, tz='UTC') - pd.Timedelta(days=30)
                    df = df[df.index >= cutoff]
                return df
            except Exception as e:
                print(f"    load_master({ticker}): FAILED — {e}")
                return None

        def load_seed_scores_bt(ticker, experiment):
            clean = TICKER_FILE_MAP.get(ticker, ticker)
            results_dir = f"Algos/Results_all_v2/Universal_Exp{experiment}"
            prefix = f"{results_dir}/{clean}/Seed_"
            blobs = list(gcs_client.bucket(BUCKET_NAME).list_blobs(prefix=prefix))
            seed_series = {}
            for blob in blobs:
                name = blob.name.split('/')[-1]
                if not name.startswith('Seed_') or not name.endswith('_Results.csv'):
                    continue
                seed_id = int(name.replace('Seed_', '').replace('_Results.csv', ''))
                d = pd.read_csv(io.BytesIO(blob.download_as_bytes()), index_col=0, parse_dates=True)
                if d.index.tz is None:
                    d.index = d.index.tz_localize('UTC')
                d.index = d.index.tz_convert(PST)
                seed_series[seed_id] = d['Prob_Spike'].sort_index()
            return seed_series

        def load_stitched_scores(ticker, fold_experiments):
            all_fold_seeds = {}
            for fold_exp in fold_experiments:
                all_fold_seeds[fold_exp] = load_seed_scores_bt(ticker, fold_exp)
            stitched = {}
            for pos in range(15):
                series_list = []
                for fold_exp in fold_experiments:
                    fold_seeds = all_fold_seeds[fold_exp]
                    sorted_ids = sorted(fold_seeds.keys())
                    if pos < len(sorted_ids):
                        series_list.append(fold_seeds[sorted_ids[pos]])
                if series_list:
                    combined = pd.concat(series_list).sort_index()
                    combined = combined[~combined.index.duplicated(keep='last')]
                    stitched[pos] = combined
            return stitched

        @jit(nopython=True)
        def run_single_trade_limit_bt(prices, highs, lows, date_ints, time_mins,
                                       entry_idx, tp, sl, exit_time, dip_pct, max_wait):
            n = len(prices)
            if entry_idx >= n - 1: return 0.0, date_ints[entry_idx], False
            sp = prices[entry_idx]; ed = date_ints[entry_idx]; et = time_mins[entry_idx]
            lp = sp * (1.0 - dip_pct)
            fill_idx = -1
            for k in range(entry_idx + 1, n):
                if date_ints[k] != ed: break
                if time_mins[k] - et > max_wait: break
                if lows[k] <= lp: fill_idx = k; break
            if fill_idx < 0: return 0.0, ed, False
            ep = lp; tgt = ep * (1.0 + tp); stp = ep * (1.0 - sl)
            for k in range(fill_idx + 1, n):
                if date_ints[k] != ed: return (prices[k-1] - ep) / ep, ed, True
                if lows[k] <= stp and highs[k] >= tgt:
                    return (-sl if np.random.random() < 0.5 else tp), ed, True
                if lows[k] <= stp: return -sl, ed, True
                if highs[k] >= tgt: return tp, ed, True
                if time_mins[k] >= exit_time: return (prices[k] - ep) / ep, ed, True
            return (prices[n-1] - ep) / ep, ed, True

        @jit(nopython=True)
        def run_single_trade_bt(prices, highs, lows, date_ints, time_mins,
                                entry_idx, tp, sl, exit_time):
            n = len(prices)
            if entry_idx >= n - 1: return 0.0, date_ints[entry_idx], False
            ep = prices[entry_idx]; ed = date_ints[entry_idx]
            tgt = ep * (1 + tp); stp = ep * (1 - sl)
            for k in range(entry_idx + 1, n):
                if date_ints[k] != ed: return (prices[k-1] - ep) / ep, ed, True
                if lows[k] <= stp and highs[k] >= tgt:
                    return (-sl if np.random.random() < 0.5 else tp), ed, True
                if lows[k] <= stp: return -sl, ed, True
                if highs[k] >= tgt: return tp, ed, True
                if time_mins[k] >= exit_time: return (prices[k] - ep) / ep, ed, True
            return (prices[n-1] - ep) / ep, ed, True

        # Backtest both 170 and 180 walkforward for comparison
        for wf_name, wf_folds, wf_fam in [('180-185 WF', WALKFORWARD_FOLDS_180, '180'),
                                            ('170-175 WF', WALKFORWARD_FOLDS_170, '170')]:
            fam_deploy = ens[(ens['family'] == wf_fam) &
                              (ens['val_b_avg_pnl'] >= DEPLOY_THRESHOLD) &
                              (ens['tpsl_mode'] == 'limit_entry')]
            if len(fam_deploy) == 0:
                print(f"\n  {wf_name}: no deployable limit_entry tickers")
                continue

            deploy_best = fam_deploy.loc[fam_deploy.groupby('ticker')['val_b_avg_pnl'].idxmax()]
            deploy_best = deploy_best.sort_values('val_b_avg_pnl', ascending=False).head(10)

            print(f"\n  Backtesting {wf_name} — {len(deploy_best)} tickers...")
            all_bt_trades = []

            for _, r in deploy_best.iterrows():
                ticker = r['ticker']
                df_raw = load_master(ticker, start_date=BT_TEST_START)
                if df_raw is None: continue
                df_pst = df_raw.copy()
                df_pst.index = df_pst.index.tz_convert(PST)

                all_seeds = load_stitched_scores(ticker, wf_folds)
                if not all_seeds:
                    print(f"    {ticker}: no stitched scores"); continue

                idx = df_pst.index
                probs = np.zeros(len(idx), dtype=np.float64)
                n_seeds_used = 0
                for s in sorted(all_seeds.keys()):
                    common = idx.intersection(all_seeds[s].index)
                    if len(common) == 0: continue
                    locs = idx.get_indexer(common)
                    probs[locs] += all_seeds[s].loc[common].values
                    n_seeds_used += 1
                if n_seeds_used == 0: continue
                probs /= n_seeds_used

                times = (df_pst.index.hour * 60 + df_pst.index.minute).values
                s_ts = pd.Timestamp(BT_TEST_START, tz=PST)
                e_ts = pd.Timestamp(BT_TEST_END, tz=PST) + pd.Timedelta(days=1)
                dates = (df_pst.index.year*10000 + df_pst.index.month*100 + df_pst.index.day).values.astype(np.int64)
                prices = df_pst['Close'].values
                hi = df_pst['High'].values
                lo = df_pst['Low'].values

                trades = []; last_d = -1; uf = 0
                em = 375
                prob_thresh = float(r['prob_thresh'])
                tp = float(r['tp_pct']); sl = float(r['sl_pct'])
                dip = float(r.get('dip_pct', 0)); mw = int(r.get('max_wait_minutes', 0))

                for i in range(len(idx)):
                    if idx[i] < s_ts or idx[i] >= e_ts: continue
                    if times[i] != em: continue
                    if probs[i] < prob_thresh: continue
                    d_int = int(dates[i])
                    if d_int == last_d: continue

                    pnl, td, valid = run_single_trade_limit_bt(
                        prices, hi, lo, dates, times.astype(np.int64), i,
                        tp, sl, EXIT_TIME_MIN, dip, mw)
                    if not valid: uf += 1; last_d = d_int; continue

                    pnl -= 2 * SLIPPAGE_PCT
                    trades.append({'ticker': ticker, 'pnl': float(pnl), 'date_int': int(td)})
                    last_d = int(td)

                w = sum(1 for t in trades if t['pnl'] > 0)
                avg = np.mean([t['pnl'] for t in trades]) if trades else 0
                extra = f" unfilled={uf}" if uf > 0 else ""
                print(f"    {ticker:<7} {len(trades):>3} trades ({w}W/{len(trades)-w}L) "
                      f"avg={avg:>+.2%}  dip={dip:.1%}{extra}")
                all_bt_trades.extend(trades)
                del df_raw, df_pst; gc.collect()

            if all_bt_trades:
                bt_df = pd.DataFrame(all_bt_trades)
                n = len(bt_df); w = int((bt_df['pnl'] > 0).sum())
                print(f"\n  {wf_name} PORTFOLIO: {bt_df['ticker'].nunique()} tickers, {n} trades "
                      f"({w}W/{n-w}L), avg={bt_df['pnl'].mean():>+.2%}, WR={w/n:.0%}")

    except Exception as e:
        print(f"\n  Backtest failed: {e}")
        import traceback; traceback.print_exc()

else:
    print(f"\n\n  [Backtest disabled — set RUN_BACKTEST = True to enable]")


# ==============================================================================
# SECTION 10: VERDICT
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  SECTION 10: VERDICT — Is Model Development Improving?")
print(f"{'='*W}")

print(f"\n  Family Rankings (by avg ensemble test PnL, all modes):")
rankings = []
for fam in FAM_ORDER:
    fam_ens = ens[ens['family'] == fam]
    if len(fam_ens) > 0:
        rankings.append((fam, fam_ens['val_b_avg_pnl'].mean()))
rankings.sort(key=lambda x: x[1], reverse=True)

for rank, (fam, avg_pnl) in enumerate(rankings, 1):
    fam_ens = ens[ens['family'] == fam]
    deployable = fam_ens[fam_ens['val_b_avg_pnl'] >= DEPLOY_THRESHOLD]['ticker'].nunique()
    best_mode = fam_ens.groupby('tpsl_mode')['val_b_avg_pnl'].mean().idxmax()
    print(f"  {rank}. {FAM_LABELS[fam]:<14} avg={avg_pnl:>+.2%}  "
          f"deploy={deployable}  best_mode={best_mode}")

print(f"\n  Family Rankings (limit_entry only):")
rankings_lim = []
for fam in FAM_ORDER:
    fam_lim = ens[(ens['family'] == fam) & (ens['tpsl_mode'] == 'limit_entry')]
    if len(fam_lim) > 0:
        rankings_lim.append((fam, fam_lim['val_b_avg_pnl'].mean()))
rankings_lim.sort(key=lambda x: x[1], reverse=True)

for rank, (fam, avg_pnl) in enumerate(rankings_lim, 1):
    fam_lim = ens[(ens['family'] == fam) & (ens['tpsl_mode'] == 'limit_entry')]
    deploy_n = fam_lim[fam_lim['val_b_avg_pnl'] >= DEPLOY_THRESHOLD]['ticker'].nunique()
    print(f"  {rank}. {FAM_LABELS[fam]:<14} limit_avg={avg_pnl:>+.2%}  deploy={deploy_n}")

# Key conclusions
print(f"\n  Key findings:")
if len(rankings) >= 2:
    best_fam = rankings[0][0]
    second_fam = rankings[1][0]
    improvement = rankings[0][1] - rankings[1][1]
    print(f"  - Best overall: {FAM_LABELS[best_fam]} (+{improvement:.2%} over {FAM_LABELS[second_fam]})")

wf_170_avg = ens[ens['family'] == '170']['val_b_avg_pnl'].mean() if len(ens[ens['family'] == '170']) > 0 else 0
wf_180_avg = ens[ens['family'] == '180']['val_b_avg_pnl'].mean() if len(ens[ens['family'] == '180']) > 0 else 0
sf_152_avg = ens[ens['family'] == '152']['val_b_avg_pnl'].mean() if len(ens[ens['family'] == '152']) > 0 else 0
sf_160_avg = ens[ens['family'] == '160']['val_b_avg_pnl'].mean() if len(ens[ens['family'] == '160']) > 0 else 0

wf_avg = (wf_170_avg + wf_180_avg) / 2
sf_avg = (sf_152_avg + sf_160_avg) / 2
if wf_avg > sf_avg:
    print(f"  - Walkforward outperforms single-fold by {wf_avg - sf_avg:.2%} avg")
else:
    print(f"  - Single-fold outperforms walkforward by {sf_avg - wf_avg:.2%} avg")

if wf_180_avg > wf_170_avg:
    print(f"  - Later cutoffs (180) improve over earlier (170) by {wf_180_avg - wf_170_avg:.2%}")
else:
    print(f"  - Earlier cutoffs (170) outperform later (180) by {wf_170_avg - wf_180_avg:.2%}")

print(f"\n{'='*W}")
