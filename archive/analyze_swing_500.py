"""
analyze_swing_500.py — Asset selection for Exp 500 Morning Swing
================================================================
Pulls WF optimization results from BQ, evaluates which tickers pass
the deployment gate using seed stability (no family agreement — 500
is a standalone experiment).

Gate logic:
  1. Per-seed: survived_a=TRUE, val_a_avg_pnl > 0, opt_avg_pnl > 0
  2. Per-ticker: count surviving seeds across all modes
  3. Seed stability: require N+ seeds with positive VAL PnL
  4. Ensemble: check ensemble rows for TEST performance
  5. Rank by score_pnl2 = mean_va_pnl^2 * seed_agreement

No backtesting — BQ analysis only.
"""

from google.cloud import bigquery
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass

bq_client = bigquery.Client()

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
W = 120
MODEL_VERSION = 'robust_wf_v12_swing'
BQ_TABLE = 'ys-time-series.scoring_logs.wf_optimization_results'

DEPLOY_PNL_THRESHOLD = 0.010
MIN_SEEDS_SURVIVING = 5
MIN_SEEDS_FOR_GROUP = 3

TICKER_RENAME = {'MIGI': 'BGDE', 'BITF': 'KEEL'}
EXCLUDE_TICKERS = {'TLT', 'VIX', 'SMLR', 'SPY', 'IWM'}

ALL_MODES = ['static', 'atr', 'atr_adaptive', 'regime_scaled', 'trailing_decay',
             'regime_adaptive', 'atr_sl', 'grace', 'limit_entry']

score_pnl2 = lambda d: (d['mean_va'].clip(lower=0) ** 2) * d['ag_pnl']


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD INDIVIDUAL SEED ROWS
# ══════════════════════════════════════════════════════════════════════════════
q_seeds = f'''
SELECT ticker, experiment, seed, tpsl_mode, entry_minute,
       prob_thresh, tp_pct, sl_pct,
       tp_mult, sl_mult, trail_start_mult, trail_end_mult,
       grace_minutes, dip_pct, max_wait_minutes, vol_filter_pct,
       pre_range_filter_pct,
       opt_avg_pnl, opt_win_rate, opt_count, opt_freq,
       val_a_avg_pnl, val_a_win_rate, val_a_count, val_a_freq,
       val_b_avg_pnl, val_b_win_rate, val_b_count, val_b_freq,
       survived_a, survived_b
FROM `{BQ_TABLE}`
WHERE model_version = '{MODEL_VERSION}'
  AND is_ensemble = FALSE
  AND survived_a = TRUE
  AND val_a_count >= 3
  AND val_a_avg_pnl > 0
  AND opt_avg_pnl > 0
'''
df = bq_client.query(q_seeds).to_dataframe()
df['ticker'] = df['ticker'].replace(TICKER_RENAME)
df = df[~df['ticker'].isin(EXCLUDE_TICKERS)]

for col in ['grace_minutes', 'max_wait_minutes', 'vol_filter_pct']:
    if col in df.columns:
        df[col] = df[col].fillna(0).astype(int)
for col in ['dip_pct', 'tp_mult', 'sl_mult', 'trail_start_mult', 'trail_end_mult',
            'pre_range_filter_pct']:
    if col in df.columns:
        df[col] = df[col].fillna(0.0)

print(f"{'═'*W}")
print(f"  EXP 500 MORNING SWING — ASSET SELECTION")
print(f"  Model version: {MODEL_VERSION}")
print(f"{'═'*W}")
print(f"\n  Loaded {len(df)} qualifying seed rows")
print(f"  Tickers: {df['ticker'].nunique()}")
print(f"  Modes: {sorted(df['tpsl_mode'].unique())}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. LOAD ENSEMBLE ROWS
# ══════════════════════════════════════════════════════════════════════════════
q_ens = f'''
SELECT ticker, experiment, tpsl_mode, n_seeds_avg,
       prob_thresh, tp_pct, sl_pct,
       tp_mult, sl_mult, trail_start_mult, trail_end_mult,
       grace_minutes, dip_pct, max_wait_minutes,
       opt_avg_pnl, opt_win_rate, opt_count, opt_freq,
       val_a_avg_pnl, val_a_win_rate, val_a_count, val_a_freq,
       val_b_avg_pnl, val_b_win_rate, val_b_count, val_b_freq,
       survived_a, survived_b
FROM `{BQ_TABLE}`
WHERE model_version = '{MODEL_VERSION}'
  AND is_ensemble = TRUE
'''
df_ens = bq_client.query(q_ens).to_dataframe()
df_ens['ticker'] = df_ens['ticker'].replace(TICKER_RENAME)
df_ens = df_ens[~df_ens['ticker'].isin(EXCLUDE_TICKERS)]
print(f"  Ensemble rows: {len(df_ens)} ({df_ens['ticker'].nunique()} tickers)")


# ══════════════════════════════════════════════════════════════════════════════
# 3. LOAD OPTIMIZATION COVERAGE (all rows, including 0-survivor tickers)
# ══════════════════════════════════════════════════════════════════════════════
q_coverage = f'''
SELECT ticker, tpsl_mode,
       COUNT(DISTINCT seed) as n_seeds_total,
       SUM(CASE WHEN is_ensemble THEN 1 ELSE 0 END) as n_ensemble
FROM `{BQ_TABLE}`
WHERE model_version = '{MODEL_VERSION}'
GROUP BY ticker, tpsl_mode
'''
df_cov = bq_client.query(q_coverage).to_dataframe()
df_cov['ticker'] = df_cov['ticker'].replace(TICKER_RENAME)
df_cov = df_cov[~df_cov['ticker'].isin(EXCLUDE_TICKERS)]
all_optimized_tickers = sorted(df_cov['ticker'].unique())
print(f"  Total optimized tickers (including 0-survivor): {len(all_optimized_tickers)}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. SEED STABILITY ANALYSIS — per ticker × mode
# ══════════════════════════════════════════════════════════════════════════════
GRP = ['ticker', 'tpsl_mode', 'prob_thresh', 'tp_pct', 'sl_pct',
       'tp_mult', 'sl_mult', 'trail_start_mult', 'trail_end_mult',
       'grace_minutes', 'dip_pct', 'max_wait_minutes', 'vol_filter_pct']

sa = df.groupby(GRP).agg(
    n_seeds  =('seed', 'nunique'),
    n_pos_pnl=('val_a_avg_pnl', lambda x: (x >= DEPLOY_PNL_THRESHOLD).sum()),
    mean_va  =('val_a_avg_pnl', 'mean'),
    med_va   =('val_a_avg_pnl', 'median'),
    std_va   =('val_a_avg_pnl', 'std'),
    wr_va    =('val_a_win_rate', 'mean'),
    cnt_va   =('val_a_count', 'mean'),
    freq_va  =('val_a_freq', 'mean'),
    mean_opt =('opt_avg_pnl', 'mean'),
    wr_opt   =('opt_win_rate', 'mean'),
    cnt_opt  =('opt_count', 'mean'),
    vb_pnl   =('val_b_avg_pnl', 'mean'),
    vb_wr    =('val_b_win_rate', 'mean'),
    vb_cnt   =('val_b_count', 'mean'),
).reset_index()
sa['ag_pnl'] = (sa['n_pos_pnl'] / sa['n_seeds']).clip(upper=1.0)
sa['std_va'] = sa['std_va'].fillna(0)
sa['score'] = score_pnl2(sa)

print(f"\n  Seed groups: {len(sa)}")
print(f"  Tickers with qualifying groups: {sa['ticker'].nunique()}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. PER-TICKER SUMMARY — best config per mode, seed stability
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n\n{'═'*W}")
print(f"  PER-TICKER SEED STABILITY — surviving seeds across modes")
print(f"{'═'*W}")

ticker_summary = []
sa_qual = sa[(sa['n_seeds'] >= MIN_SEEDS_FOR_GROUP) & (sa['mean_va'] >= DEPLOY_PNL_THRESHOLD)]

for tk in sorted(sa['ticker'].unique()):
    tk_seeds = df[df['ticker'] == tk]
    tk_groups = sa_qual[sa_qual['ticker'] == tk]

    n_unique_seeds = tk_seeds['seed'].nunique()
    n_modes_with_survivors = tk_groups['tpsl_mode'].nunique()

    best_per_mode = {}
    for mode in ALL_MODES:
        mode_groups = tk_groups[tk_groups['tpsl_mode'] == mode]
        if mode_groups.empty:
            continue
        best_idx = mode_groups['score'].idxmax()
        best_per_mode[mode] = mode_groups.loc[best_idx]

    best_mode = None
    best_score = -1
    best_row = None
    for mode, row in best_per_mode.items():
        if row['score'] > best_score:
            best_score = row['score']
            best_mode = mode
            best_row = row

    tk_ens = df_ens[df_ens['ticker'] == tk]
    best_ens = None
    if not tk_ens.empty:
        ens_surv = tk_ens[tk_ens['survived_a'] == True]
        if not ens_surv.empty:
            best_ens_idx = ens_surv['val_a_avg_pnl'].idxmax()
            best_ens = ens_surv.loc[best_ens_idx]

    ticker_summary.append({
        'ticker': tk,
        'n_unique_seeds': n_unique_seeds,
        'n_modes': n_modes_with_survivors,
        'best_mode': best_mode,
        'best_score': best_score,
        'best_va_pnl': best_row['mean_va'] if best_row is not None else 0,
        'best_va_wr': best_row['wr_va'] if best_row is not None else 0,
        'best_va_cnt': best_row['cnt_va'] if best_row is not None else 0,
        'best_vb_pnl': best_row['vb_pnl'] if best_row is not None else 0,
        'best_ag': best_row['ag_pnl'] if best_row is not None else 0,
        'ens_va_pnl': float(best_ens['val_a_avg_pnl']) if best_ens is not None else None,
        'ens_vb_pnl': float(best_ens['val_b_avg_pnl']) if best_ens is not None else None,
        'ens_va_cnt': int(best_ens['val_a_count']) if best_ens is not None else 0,
        'ens_vb_cnt': int(best_ens['val_b_count']) if best_ens is not None else 0,
        'ens_mode': best_ens['tpsl_mode'] if best_ens is not None else None,
        'ens_survived_b': bool(best_ens['survived_b']) if best_ens is not None else False,
    })

ts = pd.DataFrame(ticker_summary).sort_values('best_score', ascending=False)

print(f"\n  {'Ticker':<8} {'Seeds':>5} {'Modes':>5} {'BestMode':<16} "
      f"{'VA_PnL':>7} {'VA_WR':>6} {'VA_N':>5} {'Ag%':>5} {'Score':>8} "
      f"│ {'Ens_VA':>7} {'Ens_VB':>7} {'Ens_Mode':<16} {'VB_pass':>7}")
print(f"  {'─'*W}")

for _, r in ts.iterrows():
    ens_va = f"{r['ens_va_pnl']:.2%}" if r['ens_va_pnl'] is not None else '   —'
    ens_vb = f"{r['ens_vb_pnl']:.2%}" if r['ens_vb_pnl'] is not None else '   —'
    ens_mode = r['ens_mode'] or '—'
    vb_pass = '✓' if r['ens_survived_b'] else '✗'
    print(f"  {r['ticker']:<8} {r['n_unique_seeds']:>5} {r['n_modes']:>5} {r['best_mode'] or '—':<16} "
          f"{r['best_va_pnl']:>6.2%} {r['best_va_wr']:>5.1%} {r['best_va_cnt']:>5.0f} "
          f"{r['best_ag']:>4.0%} {r['best_score']:>8.6f} "
          f"│ {ens_va:>7} {ens_vb:>7} {ens_mode:<16} {vb_pass:>7}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. DEPLOYMENT GATE — seed stability
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n\n{'═'*W}")
print(f"  DEPLOYMENT GATE — Seed Stability")
print(f"  Requirements: ≥{MIN_SEEDS_SURVIVING} unique surviving seeds, "
      f"VA PnL ≥ {DEPLOY_PNL_THRESHOLD:.1%}")
print(f"{'═'*W}")

deployed = ts[ts['n_unique_seeds'] >= MIN_SEEDS_SURVIVING].copy()
deployed = deployed[deployed['best_va_pnl'] >= DEPLOY_PNL_THRESHOLD]

if deployed.empty:
    print(f"\n  No tickers pass the deployment gate with ≥{MIN_SEEDS_SURVIVING} seeds.")
    print(f"  Relaxing to ≥3 seeds...")
    deployed = ts[ts['n_unique_seeds'] >= 3].copy()
    deployed = deployed[deployed['best_va_pnl'] >= DEPLOY_PNL_THRESHOLD]

if deployed.empty:
    print(f"  Still no tickers pass. Showing all tickers with any survivors:")
    deployed = ts[ts['n_unique_seeds'] >= 1].copy()

n_pass = len(deployed)
n_with_test = len(deployed[deployed['ens_survived_b'] == True])

print(f"\n  PASS: {n_pass} tickers")
print(f"  Of which {n_with_test} also pass TEST (val_b)")

print(f"\n  {'Ticker':<8} {'Seeds':>5} {'Modes':>5} {'BestMode':<16} "
      f"{'VA_PnL':>7} {'VB_PnL':>7} {'Score':>8} {'TEST':>5}")
print(f"  {'─'*80}")
for _, r in deployed.iterrows():
    vb = f"{r['best_vb_pnl']:.2%}" if r['best_vb_pnl'] != 0 else '   —'
    test = '✓' if r['ens_survived_b'] else '✗'
    print(f"  {r['ticker']:<8} {r['n_unique_seeds']:>5} {r['n_modes']:>5} {r['best_mode'] or '—':<16} "
          f"{r['best_va_pnl']:>6.2%} {vb:>7} {r['best_score']:>8.6f} {test:>5}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. MODE EFFECTIVENESS — which modes produce the most survivors?
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n\n{'═'*W}")
print(f"  MODE EFFECTIVENESS — which modes work for morning swings?")
print(f"{'═'*W}")

mode_stats = []
for mode in ALL_MODES:
    mode_seeds = df[df['tpsl_mode'] == mode]
    mode_groups = sa_qual[sa_qual['tpsl_mode'] == mode]
    n_tickers = mode_groups['ticker'].nunique()
    n_groups = len(mode_groups)
    mean_pnl = mode_groups['mean_va'].mean() if not mode_groups.empty else 0
    mean_vb = mode_groups['vb_pnl'].mean() if not mode_groups.empty else 0
    mean_seeds = mode_groups['n_seeds'].mean() if not mode_groups.empty else 0
    mode_stats.append({
        'mode': mode, 'tickers': n_tickers, 'groups': n_groups,
        'mean_va_pnl': mean_pnl, 'mean_vb_pnl': mean_vb,
        'mean_seeds': mean_seeds,
    })

ms = pd.DataFrame(mode_stats).sort_values('tickers', ascending=False)
print(f"\n  {'Mode':<20} {'Tickers':>8} {'Groups':>8} {'Avg VA':>8} {'Avg VB':>8} {'AvgSeeds':>9}")
print(f"  {'─'*70}")
for _, r in ms.iterrows():
    print(f"  {r['mode']:<20} {r['tickers']:>8} {r['groups']:>8} "
          f"{r['mean_va_pnl']:>7.2%} {r['mean_vb_pnl']:>7.2%} {r['mean_seeds']:>9.1f}")


# ══════════════════════════════════════════════════════════════════════════════
# 8. DETAILED VIEW — per-ticker × per-mode matrix
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n\n{'═'*W}")
print(f"  TICKER × MODE MATRIX — surviving seeds per mode")
print(f"{'═'*W}")

deployed_tickers = sorted(ts['ticker'].unique())
short_modes = ['STAT', 'ATR', 'ATRA', 'REGS', 'TRDCY', 'REGA', 'ATSL', 'GRC', 'LMTE']
mode_map = dict(zip(ALL_MODES, short_modes))

print(f"\n  {'Ticker':<8}", end='')
for sm in short_modes:
    print(f" {sm:>5}", end='')
print(f"  {'Total':>6}")
print(f"  {'─'*(8 + 6*len(short_modes) + 7)}")

for tk in deployed_tickers:
    print(f"  {tk:<8}", end='')
    total = 0
    for mode in ALL_MODES:
        tk_mode = df[(df['ticker'] == tk) & (df['tpsl_mode'] == mode)]
        n = tk_mode['seed'].nunique()
        total += n
        if n == 0:
            print(f" {'·':>5}", end='')
        elif n >= 10:
            print(f" {n:>5}", end='')
        else:
            print(f" {n:>5}", end='')
    print(f"  {total:>6}")


# ══════════════════════════════════════════════════════════════════════════════
# 9. ENSEMBLE DEEP DIVE — for tickers that pass the gate
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n\n{'═'*W}")
print(f"  ENSEMBLE DETAIL — best ensemble config per ticker")
print(f"{'═'*W}")

ens_detail = df_ens[df_ens['survived_a'] == True].copy()
if ens_detail.empty:
    print(f"\n  No ensemble rows with survived_a=TRUE found.")
else:
    ens_detail = ens_detail.sort_values('val_a_avg_pnl', ascending=False)
    ens_best = ens_detail.groupby('ticker').first().reset_index()
    ens_best = ens_best.sort_values('val_a_avg_pnl', ascending=False)

    print(f"\n  {'Ticker':<8} {'Mode':<16} {'Prob':>5} {'TP%':>5} {'SL%':>5} "
          f"{'OptPnL':>7} {'OptN':>5} {'VA_PnL':>7} {'VA_WR':>6} {'VA_N':>5} "
          f"{'VB_PnL':>7} {'VB_WR':>6} {'VB_N':>5} {'Seeds':>5} {'VB':>3}")
    print(f"  {'─'*W}")

    for _, r in ens_best.iterrows():
        tp_str = f"{r['tp_pct']:.1%}" if r['tp_pct'] > 0 else f"x{r.get('tp_mult',0):.1f}"
        sl_str = f"{r['sl_pct']:.1%}" if r['sl_pct'] > 0 else f"x{r.get('sl_mult',0):.1f}"
        vb_flag = '✓' if r['survived_b'] else '✗'
        dip_info = ''
        if r['tpsl_mode'] == 'limit_entry' and r.get('dip_pct', 0) > 0:
            dip_info = f" dip={r['dip_pct']:.1%}"
        print(f"  {r['ticker']:<8} {r['tpsl_mode']:<16} {r['prob_thresh']:>5.2f} "
              f"{tp_str:>5} {sl_str:>5} "
              f"{r['opt_avg_pnl']:>6.2%} {r['opt_count']:>5.0f} "
              f"{r['val_a_avg_pnl']:>6.2%} {r['val_a_win_rate']:>5.1%} {r['val_a_count']:>5.0f} "
              f"{r['val_b_avg_pnl']:>6.2%} {r['val_b_win_rate']:>5.1%} {r['val_b_count']:>5.0f} "
              f"{r['n_seeds_avg']:>5.0f} {vb_flag:>3}{dip_info}")


# ══════════════════════════════════════════════════════════════════════════════
# 10. OPTIMIZATION PROGRESS — which tickers are done?
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n\n{'═'*W}")
print(f"  OPTIMIZATION PROGRESS")
print(f"{'═'*W}")

q_progress = f'''
SELECT ticker, COUNT(DISTINCT tpsl_mode) as n_modes,
       COUNT(DISTINCT CASE WHEN is_ensemble THEN tpsl_mode END) as n_ens_modes,
       COUNT(*) as total_rows
FROM `{BQ_TABLE}`
WHERE model_version = '{MODEL_VERSION}'
GROUP BY ticker
ORDER BY ticker
'''
df_prog = bq_client.query(q_progress).to_dataframe()
df_prog['ticker'] = df_prog['ticker'].replace(TICKER_RENAME)
df_prog = df_prog[~df_prog['ticker'].isin(EXCLUDE_TICKERS)]

complete = df_prog[df_prog['n_modes'] == 9]
partial = df_prog[(df_prog['n_modes'] > 0) & (df_prog['n_modes'] < 9)]
print(f"\n  Complete (9/9 modes): {len(complete)} tickers")
print(f"  Partial (<9 modes):  {len(partial)} tickers")
print(f"  Not started:         estimated {69 - len(df_prog)} tickers")

if not partial.empty:
    print(f"\n  Partial tickers:")
    for _, r in partial.sort_values('ticker').iterrows():
        print(f"    {r['ticker']:<8} {r['n_modes']}/9 modes, {r['total_rows']} rows")


# ══════════════════════════════════════════════════════════════════════════════
# 11. SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n\n{'═'*W}")
print(f"  SUMMARY")
print(f"{'═'*W}")
print(f"  Total tickers optimized:  {len(all_optimized_tickers)}")
print(f"  Tickers with survivors:   {ts['ticker'].nunique()}")
print(f"  Pass gate (≥{MIN_SEEDS_SURVIVING} seeds):    {len(deployed)}")
print(f"  Also pass TEST:           {n_with_test}")

if not deployed.empty:
    top = deployed.head(5)
    print(f"\n  Top candidates:")
    for _, r in top.iterrows():
        test_tag = " [TEST ✓]" if r['ens_survived_b'] else ""
        print(f"    {r['ticker']:<8} — {r['n_unique_seeds']} seeds, "
              f"{r['n_modes']} modes, VA={r['best_va_pnl']:.2%}, "
              f"score={r['best_score']:.6f}{test_tag}")

print(f"\n{'═'*W}")
