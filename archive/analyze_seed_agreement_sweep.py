"""
analyze_seed_agreement_sweep.py — Parametric sweep of seed_agreement composite variants
========================================================================================
Tests 100+ variations of the seed_agreement composite score against val_b (out-of-sample).
Sweeps: min_seeds, min_agree_pct, scoring formula, opt cross-validation, trade count
weighting, agree exponent, limit_entry bonus, and combinations.

Evaluates each variant by: avg val_b PnL, n_tickers, pct positive, total val_b PnL.
No full backtest — uses BQ val_b data directly (fast).

Colab-ready.
"""

from google.cloud import bigquery
import numpy as np
import pandas as pd
from itertools import product

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass

bq = bigquery.Client()

W = 130
DEPLOY_THRESHOLD = 0.018

# ==============================================================================
# LOAD DATA
# ==============================================================================
q = '''
SELECT ticker, experiment, seed, tpsl_mode, entry_minute,
       prob_thresh, tp_pct, sl_pct, tp_mult, sl_mult,
       opt_avg_pnl, opt_win_rate, opt_count,
       val_a_avg_pnl, val_a_win_rate, val_a_count, val_a_freq,
       val_b_avg_pnl, val_b_win_rate, val_b_count, val_b_freq,
       survived_a, survived_b, is_ensemble,
       grace_minutes, dip_pct, max_wait_minutes,
       vol_filter_pct, vix_threshold, tlt_crash, spy_ignore
FROM `ys-time-series.scoring_logs.wf_optimization_results`
WHERE model_version = 'robust_wf_v12'
  AND survived_a = TRUE AND val_a_count >= 3 AND val_a_avg_pnl > 0
  AND opt_avg_pnl > 0
  AND val_a_avg_pnl / opt_avg_pnl <= 3.0
  AND (experiment LIKE '152%' OR experiment LIKE '160%')
'''
df = bq.query(q).to_dataframe()
print(f"Loaded {len(df)} rows, {df['ticker'].nunique()} tickers, {df['experiment'].nunique()} experiments")

for col in ['grace_minutes', 'max_wait_minutes', 'vol_filter_pct']:
    if col in df.columns:
        df[col] = df[col].fillna(0).astype(int)
for col in ['dip_pct', 'tp_mult', 'sl_mult', 'trail_start_mult', 'trail_end_mult']:
    if col in df.columns:
        df[col] = df[col].fillna(0.0)

# ==============================================================================
# GROUPING KEYS (same as backtest)
# ==============================================================================
GRP_KEYS = ['ticker', 'experiment', 'tpsl_mode', 'entry_minute',
            'prob_thresh', 'tp_pct', 'sl_pct',
            'tp_mult', 'sl_mult', 'trail_start_mult', 'trail_end_mult',
            'grace_minutes', 'dip_pct', 'max_wait_minutes',
            'vol_filter_pct']

# ==============================================================================
# PRE-COMPUTE: individual seed data grouped by config
# ==============================================================================
df_indiv = df[df['is_ensemble'] == False].copy()
print(f"Individual seed rows: {len(df_indiv)}")

_extra_agg = {}
for _c in ['vix_threshold', 'tlt_crash', 'spy_ignore']:
    if _c in df_indiv.columns:
        _extra_agg[_c] = (_c, 'first')

sa_base = df_indiv.groupby(GRP_KEYS).agg(
    n_seeds=('seed', 'nunique'),
    n_pos_seeds_18=('val_a_avg_pnl', lambda x: (x >= 0.018).sum()),
    n_pos_seeds_15=('val_a_avg_pnl', lambda x: (x >= 0.015).sum()),
    n_pos_seeds_12=('val_a_avg_pnl', lambda x: (x >= 0.012).sum()),
    n_pos_seeds_10=('val_a_avg_pnl', lambda x: (x >= 0.010).sum()),
    n_pos_seeds_20=('val_a_avg_pnl', lambda x: (x >= 0.020).sum()),
    n_pos_seeds_25=('val_a_avg_pnl', lambda x: (x >= 0.025).sum()),
    mean_va_pnl=('val_a_avg_pnl', 'mean'),
    median_va_pnl=('val_a_avg_pnl', 'median'),
    std_va_pnl=('val_a_avg_pnl', 'std'),
    mean_va_wr=('val_a_win_rate', 'mean'),
    mean_va_count=('val_a_count', 'mean'),
    mean_va_freq=('val_a_freq', 'mean'),
    mean_opt_pnl=('opt_avg_pnl', 'mean'),
    mean_opt_wr=('opt_win_rate', 'mean'),
    mean_opt_count=('opt_count', 'mean'),
    **_extra_agg,
).reset_index()

# merge val_b for evaluation (average across seeds for the config group)
vb_agg = df_indiv.groupby(GRP_KEYS).agg(
    val_b_avg_pnl=('val_b_avg_pnl', 'mean'),
    val_b_win_rate=('val_b_win_rate', 'mean'),
    val_b_count=('val_b_count', 'mean'),
).reset_index()

sa_base = sa_base.merge(vb_agg, on=GRP_KEYS, how='left')

# pre-compute agreement percentages at different thresholds
for thresh_label in ['10', '12', '15', '18', '20', '25']:
    col = f'n_pos_seeds_{thresh_label}'
    sa_base[f'agree_{thresh_label}'] = (sa_base[col] / sa_base['n_seeds']).clip(upper=1.0)

# coefficient of variation (lower = more consistent across seeds)
sa_base['cv'] = (sa_base['std_va_pnl'] / sa_base['mean_va_pnl'].clip(lower=0.001)).fillna(1.0)
sa_base['consistency'] = (1.0 - sa_base['cv']).clip(lower=0.1)

# limit_entry flag
sa_base['is_limit'] = (sa_base['tpsl_mode'] == 'limit_entry').astype(int)

print(f"Config groups: {len(sa_base)}, tickers: {sa_base['ticker'].nunique()}")


# ==============================================================================
# SWEEP FUNCTION
# ==============================================================================
def run_variant(sa, params):
    """Apply params dict to sa_base and return selected configs."""
    d = sa.copy()

    min_seeds = params.get('min_seeds', 3)
    d = d[d['n_seeds'] >= min_seeds]

    agree_thresh_pct = params.get('agree_thresh_pct', '18')
    agree_col = f'agree_{agree_thresh_pct}'
    if agree_col not in d.columns:
        return pd.DataFrame()

    min_agree = params.get('min_agree', 0.0)
    if min_agree > 0:
        d = d[d[agree_col] >= min_agree]

    opt_min_pnl = params.get('opt_min_pnl', 0.0)
    if opt_min_pnl > 0:
        d = d[d['mean_opt_pnl'] >= opt_min_pnl]

    opt_min_wr = params.get('opt_min_wr', 0.0)
    if opt_min_wr > 0:
        d = d[d['mean_opt_wr'] >= opt_min_wr]

    min_va_count = params.get('min_va_count', 0)
    if min_va_count > 0:
        d = d[d['mean_va_count'] >= min_va_count]

    if d.empty:
        return pd.DataFrame()

    # --- SCORING FORMULA ---
    formula = params.get('formula', 'pnl_wr_agree')
    agree_exp = params.get('agree_exp', 1.0)
    agree = d[agree_col] ** agree_exp

    if formula == 'pnl_wr_agree':
        d['score'] = d['mean_va_pnl'] * d['mean_va_wr'] * agree
    elif formula == 'pnl_agree':
        d['score'] = d['mean_va_pnl'] * agree
    elif formula == 'pnl_wr_agree_logN':
        d['score'] = d['mean_va_pnl'] * d['mean_va_wr'] * agree * np.log2(1 + d['mean_va_count'])
    elif formula == 'pnl_agree_logN':
        d['score'] = d['mean_va_pnl'] * agree * np.log2(1 + d['mean_va_count'])
    elif formula == 'pnl_wr_agree_cons':
        d['score'] = d['mean_va_pnl'] * d['mean_va_wr'] * agree * d['consistency']
    elif formula == 'pnl_wr_agree_opt':
        d['score'] = d['mean_va_pnl'] * d['mean_va_wr'] * agree * d['mean_opt_pnl']
    elif formula == 'pnl_sq_agree':
        d['score'] = (d['mean_va_pnl'] ** 2) * agree
    elif formula == 'median_wr_agree':
        d['score'] = d['median_va_pnl'] * d['mean_va_wr'] * agree
    elif formula == 'opt_va_blend_agree':
        blended_pnl = 0.5 * d['mean_opt_pnl'] + 0.5 * d['mean_va_pnl']
        d['score'] = blended_pnl * d['mean_va_wr'] * agree
    elif formula == 'rank_agree':
        d['rank_pnl'] = d.groupby('ticker')['mean_va_pnl'].rank(ascending=False)
        d['rank_wr'] = d.groupby('ticker')['mean_va_wr'].rank(ascending=False)
        d['rank_agree'] = d.groupby('ticker')[agree_col].rank(ascending=False)
        d['score'] = -(d['rank_pnl'] + d['rank_wr'] + d['rank_agree'])
    else:
        d['score'] = d['mean_va_pnl'] * d['mean_va_wr'] * agree

    # limit_entry bonus
    limit_bonus = params.get('limit_bonus', 1.0)
    if limit_bonus != 1.0:
        d['score'] = d['score'] * np.where(d['is_limit'] == 1, limit_bonus, 1.0)

    # select best per ticker
    best = d.loc[d.groupby('ticker')['score'].idxmax()]

    # deploy threshold
    deploy_thresh_col = params.get('deploy_thresh_col', 'mean_va_pnl')
    deploy_thresh_val = params.get('deploy_thresh', DEPLOY_THRESHOLD)
    best = best[best[deploy_thresh_col] >= deploy_thresh_val]

    return best


# ==============================================================================
# DEFINE ALL VARIANTS
# ==============================================================================
variants = {}

# --- 1. BASELINE: current seed_agreement ---
variants['00_BASELINE'] = {
    'min_seeds': 3, 'agree_thresh_pct': '18', 'min_agree': 0.0,
    'formula': 'pnl_wr_agree',
}

# --- 2. MIN SEEDS SWEEP ---
for ms in [3, 4, 5, 7, 10]:
    variants[f'01_min_seeds_{ms}'] = {
        'min_seeds': ms, 'agree_thresh_pct': '18', 'formula': 'pnl_wr_agree',
    }

# --- 3. AGREE THRESHOLD SWEEP (what counts as "positive seed") ---
for at in ['10', '12', '15', '18', '20', '25']:
    variants[f'02_agree_thresh_{at}pct'] = {
        'min_seeds': 3, 'agree_thresh_pct': at, 'formula': 'pnl_wr_agree',
    }

# --- 4. MINIMUM AGREE PCT ---
for ma in [0.0, 0.3, 0.5, 0.6, 0.7, 0.8]:
    pct_str = f'{int(ma*100)}'
    variants[f'03_min_agree_{pct_str}pct'] = {
        'min_seeds': 3, 'agree_thresh_pct': '18', 'min_agree': ma,
        'formula': 'pnl_wr_agree',
    }

# --- 5. SCORING FORMULA SWEEP ---
formulas = [
    'pnl_wr_agree', 'pnl_agree', 'pnl_wr_agree_logN', 'pnl_agree_logN',
    'pnl_wr_agree_cons', 'pnl_wr_agree_opt', 'pnl_sq_agree',
    'median_wr_agree', 'opt_va_blend_agree', 'rank_agree',
]
for f in formulas:
    variants[f'04_formula_{f}'] = {
        'min_seeds': 3, 'agree_thresh_pct': '18', 'formula': f,
    }

# --- 6. AGREE EXPONENT (weight agreement more/less heavily) ---
for ae in [0.5, 1.0, 1.5, 2.0, 3.0]:
    variants[f'05_agree_exp_{ae}'] = {
        'min_seeds': 3, 'agree_thresh_pct': '18', 'formula': 'pnl_wr_agree',
        'agree_exp': ae,
    }

# --- 7. OPT CROSS-VALIDATION ---
for opt_min in [0.005, 0.01, 0.015, 0.018]:
    pct_str = f'{int(opt_min*1000)}'
    variants[f'06_opt_min_{pct_str}bps'] = {
        'min_seeds': 3, 'agree_thresh_pct': '18', 'formula': 'pnl_wr_agree',
        'opt_min_pnl': opt_min,
    }

# --- 8. OPT WR FILTER ---
for opt_wr in [0.50, 0.55, 0.60]:
    variants[f'07_opt_wr_{int(opt_wr*100)}'] = {
        'min_seeds': 3, 'agree_thresh_pct': '18', 'formula': 'pnl_wr_agree',
        'opt_min_wr': opt_wr,
    }

# --- 9. MIN VAL_A COUNT ---
for mc in [3, 5, 8, 10]:
    variants[f'08_min_va_count_{mc}'] = {
        'min_seeds': 3, 'agree_thresh_pct': '18', 'formula': 'pnl_wr_agree',
        'min_va_count': mc,
    }

# --- 10. LIMIT ENTRY BONUS ---
for lb in [1.0, 1.2, 1.5, 2.0, 3.0]:
    variants[f'09_limit_bonus_{lb}x'] = {
        'min_seeds': 3, 'agree_thresh_pct': '18', 'formula': 'pnl_wr_agree',
        'limit_bonus': lb,
    }

# --- 11. COMBINED: best seeds + agree thresh + formula combos ---
for ms, at, f in [
    (5, '15', 'pnl_wr_agree'),
    (5, '18', 'pnl_wr_agree'),
    (5, '15', 'pnl_wr_agree_logN'),
    (5, '18', 'pnl_agree'),
    (7, '15', 'pnl_wr_agree'),
    (7, '18', 'pnl_wr_agree'),
    (3, '15', 'pnl_wr_agree_logN'),
    (3, '12', 'pnl_wr_agree'),
    (5, '12', 'pnl_wr_agree'),
]:
    variants[f'10_combo_s{ms}_t{at}_{f[:12]}'] = {
        'min_seeds': ms, 'agree_thresh_pct': at, 'formula': f,
    }

# --- 12. COMBINED: seeds + agree + opt filter ---
for ms, at, opt_min in [
    (3, '18', 0.01),
    (5, '18', 0.01),
    (5, '15', 0.01),
    (3, '18', 0.015),
    (5, '18', 0.015),
    (3, '15', 0.01),
]:
    variants[f'11_opt_s{ms}_t{at}_opt{int(opt_min*1000)}'] = {
        'min_seeds': ms, 'agree_thresh_pct': at, 'formula': 'pnl_wr_agree',
        'opt_min_pnl': opt_min,
    }

# --- 13. COMBINED: seeds + agree + limit bonus ---
for ms, at, lb in [
    (3, '18', 1.5),
    (5, '18', 1.5),
    (3, '15', 1.5),
    (5, '15', 1.5),
    (3, '18', 2.0),
    (5, '18', 2.0),
]:
    variants[f'12_limit_s{ms}_t{at}_lb{lb}'] = {
        'min_seeds': ms, 'agree_thresh_pct': at, 'formula': 'pnl_wr_agree',
        'limit_bonus': lb,
    }

# --- 14. COMBINED: best of everything combos ---
for ms, at, f, lb, opt_min, ae in [
    (5, '15', 'pnl_wr_agree', 1.5, 0.01, 1.0),
    (5, '15', 'pnl_wr_agree', 1.5, 0.0, 1.5),
    (3, '15', 'pnl_wr_agree', 1.5, 0.01, 1.0),
    (5, '18', 'pnl_wr_agree', 1.5, 0.01, 1.0),
    (5, '15', 'pnl_wr_agree_logN', 1.5, 0.01, 1.0),
    (3, '18', 'pnl_wr_agree', 2.0, 0.01, 1.0),
    (5, '15', 'pnl_agree', 1.5, 0.01, 1.0),
    (3, '15', 'pnl_wr_agree', 2.0, 0.0, 1.0),
    (5, '12', 'pnl_wr_agree', 1.5, 0.01, 1.0),
    (7, '15', 'pnl_wr_agree', 1.5, 0.01, 1.0),
    (3, '18', 'pnl_wr_agree', 1.0, 0.01, 2.0),
    (5, '18', 'pnl_wr_agree', 1.0, 0.01, 2.0),
    (3, '15', 'pnl_wr_agree_logN', 1.5, 0.01, 1.0),
    (5, '15', 'pnl_wr_agree_cons', 1.5, 0.01, 1.0),
]:
    name = f'13_full_s{ms}_t{at}_{f[:8]}_lb{lb}_opt{int(opt_min*1000)}_ae{ae}'
    variants[name] = {
        'min_seeds': ms, 'agree_thresh_pct': at, 'formula': f,
        'limit_bonus': lb, 'opt_min_pnl': opt_min, 'agree_exp': ae,
    }

# --- 15. DEPLOY THRESHOLD ON MEDIAN instead of mean ---
for ms, at in [(3, '18'), (5, '18'), (5, '15')]:
    name = f'14_median_deploy_s{ms}_t{at}'
    variants[name] = {
        'min_seeds': ms, 'agree_thresh_pct': at, 'formula': 'pnl_wr_agree',
        'deploy_thresh_col': 'median_va_pnl',
    }

# --- 16. LIMIT ENTRY ONLY (explore limit_entry performance in isolation) ---
for ms, at in [(3, '18'), (5, '18'), (3, '15')]:
    name = f'15_limit_only_s{ms}_t{at}'
    variants[name] = {
        'min_seeds': ms, 'agree_thresh_pct': at, 'formula': 'pnl_wr_agree',
    }

print(f"\nTotal variants to test: {len(variants)}")


# ==============================================================================
# RUN ALL VARIANTS
# ==============================================================================
results = []

for name, params in sorted(variants.items()):
    # special handling: limit_only filters to limit_entry before grouping
    if 'limit_only' in name:
        sa_input = sa_base[sa_base['tpsl_mode'] == 'limit_entry'].copy()
    else:
        sa_input = sa_base

    deploy = run_variant(sa_input, params)

    if deploy.empty:
        results.append({
            'variant': name, 'n_tickers': 0, 'avg_vb': 0, 'med_vb': 0,
            'pct_positive': 0, 'tot_vb': 0, 'avg_va': 0, 'avg_opt': 0,
            'n_limit': 0, 'pct_limit': 0,
        })
        continue

    n = len(deploy)
    avg_vb = deploy['val_b_avg_pnl'].mean()
    med_vb = deploy['val_b_avg_pnl'].median()
    pos = (deploy['val_b_avg_pnl'] > 0).sum()
    tot_vb = deploy['val_b_avg_pnl'].sum()
    avg_va = deploy['mean_va_pnl'].mean()
    avg_opt = deploy['mean_opt_pnl'].mean()
    n_limit = int(deploy['is_limit'].sum())

    results.append({
        'variant': name, 'n_tickers': n, 'avg_vb': avg_vb, 'med_vb': med_vb,
        'pct_positive': pos / n, 'tot_vb': tot_vb, 'avg_va': avg_va,
        'avg_opt': avg_opt, 'n_limit': n_limit, 'pct_limit': n_limit / n,
    })

res_df = pd.DataFrame(results)


# ==============================================================================
# TABLE 1: ALL VARIANTS SORTED BY AVG VAL_B
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  ALL VARIANTS — sorted by avg val_b PnL")
print(f"{'='*W}")
print(f"\n  {'#':>3} {'Variant':<52} {'N':>3} {'Avg_vb':>8} {'Med_vb':>8} {'%Pos':>6} "
      f"{'Tot_vb':>9} {'Avg_va':>8} {'Avg_opt':>8} {'#Lim':>5} {'%Lim':>5}")
print(f"  {'-'*(W-2)}")

ranked = res_df[res_df['n_tickers'] > 0].sort_values('avg_vb', ascending=False)
for i, (_, r) in enumerate(ranked.iterrows(), 1):
    base_tag = " <<" if r['variant'] == '00_BASELINE' else ""
    print(f"  {i:>3} {r['variant']:<52} {r['n_tickers']:>3.0f} {r['avg_vb']:>+7.2%} "
          f"{r['med_vb']:>+7.2%} {r['pct_positive']:>5.0%} {r['tot_vb']:>+8.2%} "
          f"{r['avg_va']:>+7.2%} {r['avg_opt']:>+7.2%} {r['n_limit']:>5.0f} {r['pct_limit']:>4.0%}{base_tag}")

baseline_row = res_df[res_df['variant'] == '00_BASELINE']
if not baseline_row.empty:
    bl = baseline_row.iloc[0]
    bl_rank = int((ranked['variant'] == '00_BASELINE').values.argmax()) + 1
    print(f"\n  BASELINE rank: #{bl_rank} of {len(ranked)} "
          f"(avg_vb={bl['avg_vb']:+.2%}, {bl['n_tickers']:.0f} tickers)")


# ==============================================================================
# TABLE 2: GROUPED BY SWEEP DIMENSION
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  SWEEP DIMENSION SUMMARIES")
print(f"{'='*W}")

sweep_groups = [
    ('MIN SEEDS', '01_min_seeds'),
    ('AGREE THRESHOLD', '02_agree_thresh'),
    ('MIN AGREE PCT', '03_min_agree'),
    ('FORMULA', '04_formula'),
    ('AGREE EXPONENT', '05_agree_exp'),
    ('OPT CROSS-VALIDATION', '06_opt_min'),
    ('OPT WIN RATE', '07_opt_wr'),
    ('MIN VAL_A COUNT', '08_min_va_count'),
    ('LIMIT BONUS', '09_limit_bonus'),
    ('LIMIT ONLY', '15_limit_only'),
]

for title, prefix in sweep_groups:
    group = ranked[ranked['variant'].str.startswith(prefix)]
    if group.empty:
        continue
    print(f"\n  --- {title} ---")
    print(f"  {'Variant':<52} {'N':>3} {'Avg_vb':>8} {'Med_vb':>8} {'%Pos':>6} {'Tot_vb':>9} {'#Lim':>5}")
    print(f"  {'-'*90}")
    for _, r in group.sort_values('avg_vb', ascending=False).iterrows():
        print(f"  {r['variant']:<52} {r['n_tickers']:>3.0f} {r['avg_vb']:>+7.2%} "
              f"{r['med_vb']:>+7.2%} {r['pct_positive']:>5.0%} {r['tot_vb']:>+8.2%} {r['n_limit']:>5.0f}")


# ==============================================================================
# TABLE 3: TOP 10 OVERALL — per-ticker breakdown
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TOP 10 VARIANTS — per-ticker val_b detail")
print(f"{'='*W}")

top10_names = ranked.head(10)['variant'].tolist()

for vname in top10_names:
    params = variants[vname]
    if 'limit_only' in vname:
        sa_input = sa_base[sa_base['tpsl_mode'] == 'limit_entry'].copy()
    else:
        sa_input = sa_base
    deploy = run_variant(sa_input, params)
    if deploy.empty:
        continue

    r = res_df[res_df['variant'] == vname].iloc[0]
    print(f"\n  {vname}  (avg_vb={r['avg_vb']:+.2%}, {r['n_tickers']:.0f} tickers, "
          f"{r['pct_positive']:.0%} pos, {r['n_limit']:.0f} limit_entry)")
    print(f"  {'Ticker':<8} {'Experiment':<25} {'Mode':<14} {'Seeds':>5} "
          f"{'Agree':>6} {'opt_pnl':>8} {'va_pnl':>8} {'vb_pnl':>8} {'va_wr':>6} {'va_N':>5}")
    print(f"  {'-'*110}")

    for _, row in deploy.sort_values('val_b_avg_pnl', ascending=False).iterrows():
        agree_col = f"agree_{params.get('agree_thresh_pct', '18')}"
        agree_val = row.get(agree_col, 0)
        print(f"  {row['ticker']:<8} {row['experiment']:<25} {row['tpsl_mode']:<14} "
              f"{row['n_seeds']:>5.0f} {agree_val:>5.0%} {row['mean_opt_pnl']:>+7.2%} "
              f"{row['mean_va_pnl']:>+7.2%} {row['val_b_avg_pnl']:>+7.2%} "
              f"{row['mean_va_wr']:>5.0%} {row['mean_va_count']:>5.0f}")


# ==============================================================================
# TABLE 4: BASELINE vs TOP 3 — ticker-level comparison
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  BASELINE vs TOP 3 — per-ticker val_b comparison")
print(f"{'='*W}")

compare_names = ['00_BASELINE'] + ranked.head(3)['variant'].tolist()
compare_names = list(dict.fromkeys(compare_names))

compare_data = {}
all_tickers_set = set()
for vname in compare_names:
    params = variants[vname]
    if 'limit_only' in vname:
        sa_input = sa_base[sa_base['tpsl_mode'] == 'limit_entry'].copy()
    else:
        sa_input = sa_base
    deploy = run_variant(sa_input, params)
    compare_data[vname] = deploy
    if not deploy.empty:
        all_tickers_set.update(deploy['ticker'].tolist())

all_tickers_sorted = sorted(all_tickers_set)

short_names = [n[:25] for n in compare_names]
header = f"  {'Ticker':<8}"
for sn in short_names:
    header += f" {sn:>25}"
print(header)
print(f"  {'-'*(8 + 26 * len(compare_names))}")

for ticker in all_tickers_sorted:
    line = f"  {ticker:<8}"
    for vname in compare_names:
        deploy = compare_data[vname]
        match = deploy[deploy['ticker'] == ticker] if not deploy.empty else pd.DataFrame()
        if not match.empty:
            vb = float(match.iloc[0]['val_b_avg_pnl'])
            line += f" {vb:>+24.2%}"
        else:
            line += f" {'---':>25}"
    print(line)


# ==============================================================================
# TABLE 5: EFFICIENCY FRONTIER
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  EFFICIENCY FRONTIER: avg_vb vs n_tickers")
print(f"{'='*W}")
print(f"\n  {'Variant':<52} {'N':>3} {'Avg_vb':>8} {'Tot_vb':>9} {'Frontier':>9}")
print(f"  {'-'*85}")

frontier = []
for _, r in ranked.sort_values('avg_vb', ascending=False).iterrows():
    is_frontier = True
    for fm in frontier:
        if fm['n_tickers'] >= r['n_tickers'] and fm['avg_vb'] >= r['avg_vb']:
            is_frontier = False
            break
    tag = "YES" if is_frontier else ""
    base_tag = " <<" if r['variant'] == '00_BASELINE' else ""
    print(f"  {r['variant']:<52} {r['n_tickers']:>3.0f} {r['avg_vb']:>+7.2%} "
          f"{r['tot_vb']:>+8.2%} {tag:>9}{base_tag}")
    if is_frontier:
        frontier.append(r)

print(f"\n  FRONTIER POINTS ({len(frontier)}):")
for f in frontier:
    print(f"    {f['variant']}: {f['n_tickers']:.0f} tickers, avg_vb={f['avg_vb']:+.2%}, tot_vb={f['tot_vb']:+.2%}")


# ==============================================================================
# TABLE 6: LIMIT ENTRY ANALYSIS
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  LIMIT ENTRY ANALYSIS")
print(f"{'='*W}")

# how many top-10 variants use limit_entry?
print(f"\n  Top-10 limit_entry usage:")
for vname in top10_names:
    r = res_df[res_df['variant'] == vname].iloc[0]
    print(f"    {vname:<52} {r['n_limit']:.0f}/{r['n_tickers']:.0f} limit ({r['pct_limit']:.0%})")

# limit-only variants
limit_variants = ranked[ranked['variant'].str.startswith('15_limit_only')]
if not limit_variants.empty:
    print(f"\n  Limit-entry only variants:")
    for _, r in limit_variants.iterrows():
        print(f"    {r['variant']:<52} {r['n_tickers']:>3.0f} tickers  avg_vb={r['avg_vb']:>+7.2%}  tot_vb={r['tot_vb']:>+8.2%}")

# compare limit vs non-limit tickers in baseline
bl_deploy = run_variant(sa_base, variants['00_BASELINE'])
if not bl_deploy.empty:
    bl_limit = bl_deploy[bl_deploy['is_limit'] == 1]
    bl_other = bl_deploy[bl_deploy['is_limit'] == 0]
    print(f"\n  BASELINE breakdown:")
    if not bl_limit.empty:
        print(f"    Limit tickers ({len(bl_limit)}): avg_vb={bl_limit['val_b_avg_pnl'].mean():+.2%}, "
              f"med_vb={bl_limit['val_b_avg_pnl'].median():+.2%}, "
              f"pos={int((bl_limit['val_b_avg_pnl']>0).sum())}/{len(bl_limit)}")
    if not bl_other.empty:
        print(f"    Other tickers ({len(bl_other)}): avg_vb={bl_other['val_b_avg_pnl'].mean():+.2%}, "
              f"med_vb={bl_other['val_b_avg_pnl'].median():+.2%}, "
              f"pos={int((bl_other['val_b_avg_pnl']>0).sum())}/{len(bl_other)}")


# ==============================================================================
# FINAL SUMMARY
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  FINAL SUMMARY")
print(f"{'='*W}")

bl = res_df[res_df['variant'] == '00_BASELINE'].iloc[0]
top1 = ranked.iloc[0]
top1_improve = top1['avg_vb'] - bl['avg_vb']

print(f"\n  BASELINE:  {bl['variant']} — {bl['n_tickers']:.0f} tickers, avg_vb={bl['avg_vb']:+.2%}")
print(f"  BEST:      {top1['variant']} — {top1['n_tickers']:.0f} tickers, avg_vb={top1['avg_vb']:+.2%}")
print(f"  IMPROVEMENT: {top1_improve:+.2%} avg_vb")

print(f"\n  Key findings:")
print(f"  - Total variants tested: {len(variants)}")
print(f"  - Variants beating baseline: {int((ranked['avg_vb'] > bl['avg_vb']).sum())}")

# best per sweep dimension
print(f"\n  Best per dimension:")
for title, prefix in sweep_groups:
    group = ranked[ranked['variant'].str.startswith(prefix)]
    if group.empty:
        continue
    best = group.iloc[0]
    delta = best['avg_vb'] - bl['avg_vb']
    print(f"    {title:<25}: {best['variant']:<45} avg_vb={best['avg_vb']:+.2%} ({delta:+.2%} vs baseline)")

print(f"\n{'='*W}")
