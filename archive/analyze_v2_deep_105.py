"""
analyze_v2_deep_105.py — Deep impact analysis of V=0 fix on Exp-105
=====================================================================
Compares old (v1) vs new (v2_bar_is_real) WF optimization results
at the ticker/seed level. Answers:
  - Are we getting fewer trades? Where?
  - Is profit per trade changing? Better or worse?
  - Is total PnL up or down?
  - Which tickers benefit, which suffer?
  - Is the optimizer finding different configs (TP/SL/prob/entry)?

Run in Colab or locally (needs google-cloud-bigquery + pandas).
"""

import pandas as pd
import numpy as np

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass

from google.cloud import bigquery

# ==============================================================================
# CONFIG
# ==============================================================================
BQ_TABLE = f"{GCP_PROJECT}.scoring_logs.wf_optimization_results"
EXPERIMENT = '105_DEFENSE'
W = 130

# ==============================================================================
# LOAD DATA
# ==============================================================================
client = bigquery.Client()
query = f"""
SELECT *
FROM `{BQ_TABLE}`
WHERE experiment = '{EXPERIMENT}'
ORDER BY ticker, seed
"""
df = client.query(query).to_dataframe()

if df.empty:
    print("No data found.")
    raise SystemExit(0)

TICKER_RENAME = {'MIGI': 'BGDE', 'BITF': 'KEEL'}
df['ticker'] = df['ticker'].replace(TICKER_RENAME)
df['survived_a'] = df['survived_a'].fillna(False).astype(bool)
df['survived_b'] = df['survived_b'].fillna(False).astype(bool)
df['val_a_avg_pnl'] = df['val_a_avg_pnl'].fillna(0.0)
df['val_b_avg_pnl'] = df['val_b_avg_pnl'].fillna(0.0)
df['vol_filter_pct'] = df.get('vol_filter_pct', pd.Series(0, index=df.index)).fillna(0).astype(int)
df['model_version'] = df['model_version'].fillna('v1_original')
df['run_timestamp'] = pd.to_datetime(df['run_timestamp'])

# Keep latest run per (ticker, model_version)
_latest = (df.groupby(['ticker', 'model_version', 'run_id'])['run_timestamp']
           .max().reset_index()
           .sort_values('run_timestamp')
           .groupby(['ticker', 'model_version']).last()
           .reset_index()[['ticker', 'model_version', 'run_id']])
df = df.merge(_latest, on=['ticker', 'model_version', 'run_id'], how='inner')

old = df[df['model_version'] == 'v1_original'].copy()
new = df[df['model_version'] == 'v2_bar_is_real'].copy()

# Merge on ticker+seed for matched-pair analysis
m = new.merge(old, on=['ticker', 'seed'], suffixes=('_new', '_old'), how='inner')

n_old_seeds = old.groupby('ticker')['seed'].nunique()
n_new_seeds = new.groupby('ticker')['seed'].nunique()

print(f"\n{'='*W}")
print(f"  EXP-105 DEEP IMPACT ANALYSIS — V=0 FIX")
print(f"  Old (v1): {len(old)} rows across {old['ticker'].nunique()} tickers, "
      f"{old['seed'].nunique()} seeds")
print(f"  New (v2): {len(new)} rows across {new['ticker'].nunique()} tickers, "
      f"{new['seed'].nunique()} seeds")
print(f"  Matched pairs: {len(m)} (same ticker+seed)")
print(f"{'='*W}")

# ==============================================================================
# 1. TRADE COUNT IMPACT
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  1. TRADE COUNT IMPACT — How many trades did V=0 filtering remove?")
print(f"{'='*W}")
print(f"  {'Ticker':<7} "
      f"{'OLD opt_N':>9} {'NEW opt_N':>9} {'Δ N':>6} {'Δ%':>6} "
      f"{'|':>2} {'OLD va_N':>8} {'NEW va_N':>8} {'Δ N':>6} {'Δ%':>6}")
print(f"  {'─'*W}")

tickers = sorted(m['ticker'].unique())
for t in tickers:
    tm = m[m['ticker'] == t]
    old_opt_n = tm['opt_count_old'].mean()
    new_opt_n = tm['opt_count_new'].mean()
    d_opt_n = new_opt_n - old_opt_n
    d_opt_pct = (d_opt_n / old_opt_n * 100) if old_opt_n > 0 else 0

    old_va_n = tm['val_a_count_old'].mean()
    new_va_n = tm['val_a_count_new'].mean()
    d_va_n = new_va_n - old_va_n
    d_va_pct = (d_va_n / old_va_n * 100) if old_va_n > 0 else 0

    print(f"  {t:<7} "
          f"{old_opt_n:>9.1f} {new_opt_n:>9.1f} {d_opt_n:>+6.1f} {d_opt_pct:>+5.0f}% "
          f"{'|':>2} {old_va_n:>8.1f} {new_va_n:>8.1f} {d_va_n:>+6.1f} {d_va_pct:>+5.0f}%")

# Aggregate
print(f"  {'─'*W}")
print(f"  {'TOTAL':<7} "
      f"{m['opt_count_old'].mean():>9.1f} {m['opt_count_new'].mean():>9.1f} "
      f"{m['opt_count_new'].mean() - m['opt_count_old'].mean():>+6.1f} "
      f"{((m['opt_count_new'].mean() - m['opt_count_old'].mean()) / m['opt_count_old'].mean() * 100):>+5.0f}% "
      f"{'|':>2} {m['val_a_count_old'].mean():>8.1f} {m['val_a_count_new'].mean():>8.1f} "
      f"{m['val_a_count_new'].mean() - m['val_a_count_old'].mean():>+6.1f} "
      f"{((m['val_a_count_new'].mean() - m['val_a_count_old'].mean()) / m['val_a_count_old'].mean() * 100):>+5.0f}%")

# ==============================================================================
# 2. PROFIT PER TRADE
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  2. PROFIT PER TRADE — Is avg PnL per trade improving?")
print(f"{'='*W}")
print(f"  {'Ticker':<7} "
      f"{'OLD opt':>8} {'NEW opt':>8} {'Δ':>7} "
      f"{'|':>2} {'OLD va':>8} {'NEW va':>8} {'Δ':>7} "
      f"{'|':>2} {'OLD WR':>7} {'NEW WR':>7} {'Δ WR':>6}")
print(f"  {'─'*W}")

for t in tickers:
    tm = m[m['ticker'] == t]
    o_opt = tm['opt_avg_pnl_old'].mean()
    n_opt = tm['opt_avg_pnl_new'].mean()
    o_va = tm['val_a_avg_pnl_old'].mean()
    n_va = tm['val_a_avg_pnl_new'].mean()
    o_wr = tm['opt_win_rate_old'].mean()
    n_wr = tm['opt_win_rate_new'].mean()

    print(f"  {t:<7} "
          f"{o_opt:>7.2%} {n_opt:>7.2%} {n_opt - o_opt:>+6.2%} "
          f"{'|':>2} {o_va:>7.2%} {n_va:>7.2%} {n_va - o_va:>+6.2%} "
          f"{'|':>2} {o_wr:>6.1%} {n_wr:>6.1%} {n_wr - o_wr:>+5.1%}")

print(f"  {'─'*W}")
print(f"  {'TOTAL':<7} "
      f"{m['opt_avg_pnl_old'].mean():>7.2%} {m['opt_avg_pnl_new'].mean():>7.2%} "
      f"{m['opt_avg_pnl_new'].mean() - m['opt_avg_pnl_old'].mean():>+6.2%} "
      f"{'|':>2} {m['val_a_avg_pnl_old'].mean():>7.2%} {m['val_a_avg_pnl_new'].mean():>7.2%} "
      f"{m['val_a_avg_pnl_new'].mean() - m['val_a_avg_pnl_old'].mean():>+6.2%} "
      f"{'|':>2} {m['opt_win_rate_old'].mean():>6.1%} {m['opt_win_rate_new'].mean():>6.1%} "
      f"{m['opt_win_rate_new'].mean() - m['opt_win_rate_old'].mean():>+5.1%}")

# ==============================================================================
# 3. TOTAL PNL (avg_pnl × count = proxy for total return)
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  3. TOTAL PnL PROXY — avg_pnl × N (total return over window)")
print(f"{'='*W}")

m['opt_total_old'] = m['opt_avg_pnl_old'] * m['opt_count_old']
m['opt_total_new'] = m['opt_avg_pnl_new'] * m['opt_count_new']
m['va_total_old'] = m['val_a_avg_pnl_old'] * m['val_a_count_old']
m['va_total_new'] = m['val_a_avg_pnl_new'] * m['val_a_count_new']

print(f"  {'Ticker':<7} "
      f"{'OLD OPT$':>9} {'NEW OPT$':>9} {'Δ':>8} "
      f"{'|':>2} {'OLD VA$':>9} {'NEW VA$':>9} {'Δ':>8}")
print(f"  {'─'*W}")

for t in tickers:
    tm = m[m['ticker'] == t]
    o_opt_t = tm['opt_total_old'].mean()
    n_opt_t = tm['opt_total_new'].mean()
    o_va_t = tm['va_total_old'].mean()
    n_va_t = tm['va_total_new'].mean()

    print(f"  {t:<7} "
          f"{o_opt_t:>8.2%} {n_opt_t:>8.2%} {n_opt_t - o_opt_t:>+7.2%} "
          f"{'|':>2} {o_va_t:>8.2%} {n_va_t:>8.2%} {n_va_t - o_va_t:>+7.2%}")

print(f"  {'─'*W}")
o_tot_opt = m['opt_total_old'].mean()
n_tot_opt = m['opt_total_new'].mean()
o_tot_va = m['va_total_old'].mean()
n_tot_va = m['va_total_new'].mean()
print(f"  {'TOTAL':<7} "
      f"{o_tot_opt:>8.2%} {n_tot_opt:>8.2%} {n_tot_opt - o_tot_opt:>+7.2%} "
      f"{'|':>2} {o_tot_va:>8.2%} {n_tot_va:>8.2%} {n_tot_va - o_tot_va:>+7.2%}")

# ==============================================================================
# 4. SURVIVAL ANALYSIS
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  4. SURVIVAL ANALYSIS — Did V=0 fix help or hurt OOS survival?")
print(f"{'='*W}")
print(f"  {'Ticker':<7} {'Seeds':>5} "
      f"{'OLD Pass-A':>10} {'NEW Pass-A':>10} {'Δ':>4} "
      f"{'|':>2} {'OLD Pass-B':>10} {'NEW Pass-B':>10} {'Δ':>4} "
      f"{'|':>2} {'Flips':>15}")
print(f"  {'─'*W}")

for t in tickers:
    tm = m[m['ticker'] == t]
    n_seeds = len(tm)
    o_pa = int(tm['survived_a_old'].sum())
    n_pa = int(tm['survived_a_new'].sum())
    o_pb = int(tm['survived_b_old'].sum())
    n_pb = int(tm['survived_b_new'].sum())

    fp = int(((~tm['survived_a_old']) & tm['survived_a_new']).sum())
    ff = int((tm['survived_a_old'] & (~tm['survived_a_new'])).sum())
    flip_str = ""
    if fp > 0:
        flip_str += f"{fp}F→P"
    if ff > 0:
        if flip_str:
            flip_str += " "
        flip_str += f"{ff}P→F"
    if not flip_str:
        flip_str = "—"

    print(f"  {t:<7} {n_seeds:>5} "
          f"{o_pa:>4}/{n_seeds:<5} {n_pa:>4}/{n_seeds:<5} {n_pa - o_pa:>+4d} "
          f"{'|':>2} {o_pb:>4}/{n_seeds:<5} {n_pb:>4}/{n_seeds:<5} {n_pb - o_pb:>+4d} "
          f"{'|':>2} {flip_str:>15}")

print(f"  {'─'*W}")
total_seeds = len(m)
print(f"  {'TOTAL':<7} {total_seeds:>5} "
      f"{int(m['survived_a_old'].sum()):>4}/{total_seeds:<5} "
      f"{int(m['survived_a_new'].sum()):>4}/{total_seeds:<5} "
      f"{int(m['survived_a_new'].sum()) - int(m['survived_a_old'].sum()):>+4d} "
      f"{'|':>2} {int(m['survived_b_old'].sum()):>4}/{total_seeds:<5} "
      f"{int(m['survived_b_new'].sum()):>4}/{total_seeds:<5} "
      f"{int(m['survived_b_new'].sum()) - int(m['survived_b_old'].sum()):>+4d}")

# ==============================================================================
# 5. PnL DECAY (OPT → VAL-A generalization)
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  5. PnL DECAY — Does v2 generalize better from OPT to VAL-A?")
print(f"{'='*W}")
print(f"  {'Ticker':<7} "
      f"{'OLD opt→va':>11} {'OLD decay':>10} "
      f"{'|':>2} {'NEW opt→va':>11} {'NEW decay':>10} "
      f"{'|':>2} {'Better?':>8}")
print(f"  {'─'*W}")

for t in tickers:
    tm = m[m['ticker'] == t]
    o_opt = tm['opt_avg_pnl_old'].mean()
    o_va = tm['val_a_avg_pnl_old'].mean()
    n_opt = tm['opt_avg_pnl_new'].mean()
    n_va = tm['val_a_avg_pnl_new'].mean()

    o_decay = (1 - o_va / o_opt) * 100 if o_opt > 0 else 0
    n_decay = (1 - n_va / n_opt) * 100 if n_opt > 0 else 0

    better = "YES" if n_decay < o_decay else "no"

    print(f"  {t:<7} "
          f"{o_opt:>5.2%}→{o_va:>5.2%} {o_decay:>8.0f}% "
          f"{'|':>2} {n_opt:>5.2%}→{n_va:>5.2%} {n_decay:>8.0f}% "
          f"{'|':>2} {better:>8}")

print(f"  {'─'*W}")
o_opt_all = m['opt_avg_pnl_old'].mean()
o_va_all = m['val_a_avg_pnl_old'].mean()
n_opt_all = m['opt_avg_pnl_new'].mean()
n_va_all = m['val_a_avg_pnl_new'].mean()
o_decay_all = (1 - o_va_all / o_opt_all) * 100 if o_opt_all > 0 else 0
n_decay_all = (1 - n_va_all / n_opt_all) * 100 if n_opt_all > 0 else 0
print(f"  {'TOTAL':<7} "
      f"{o_opt_all:>5.2%}→{o_va_all:>5.2%} {o_decay_all:>8.0f}% "
      f"{'|':>2} {n_opt_all:>5.2%}→{n_va_all:>5.2%} {n_decay_all:>8.0f}% "
      f"{'|':>2} {'YES' if n_decay_all < o_decay_all else 'no':>8}")

# ==============================================================================
# 6. CONFIG SHIFT — Is the optimizer finding different parameters?
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  6. CONFIG SHIFT — What parameters changed between old and new?")
print(f"{'='*W}")

config_cols = ['prob_thresh', 'tp_pct', 'sl_pct', 'entry_minute',
               'vix_threshold', 'tlt_crash', 'spy_ignore', 'vol_filter_pct']

print(f"  {'Param':<16} {'OLD mean':>10} {'NEW mean':>10} {'Δ':>10} {'OLD std':>10} {'NEW std':>10}")
print(f"  {'─'*70}")

for col in config_cols:
    col_old = col + '_old'
    col_new = col + '_new'
    if col_old not in m.columns or col_new not in m.columns:
        continue
    o_mean = m[col_old].mean()
    n_mean = m[col_new].mean()
    o_std = m[col_old].std()
    n_std = m[col_new].std()
    print(f"  {col:<16} {o_mean:>10.4f} {n_mean:>10.4f} {n_mean - o_mean:>+10.4f} "
          f"{o_std:>10.4f} {n_std:>10.4f}")

# Entry time shift detail
print(f"\n  Entry time distribution (minutes after midnight):")
if 'entry_minute_old' in m.columns:
    for label, col in [('OLD', 'entry_minute_old'), ('NEW', 'entry_minute_new')]:
        vals = m[col]
        counts = vals.value_counts().sort_index()
        dist_str = "  ".join(f"{int(v)}:{int(c)}" for v, c in counts.items())
        print(f"    {label}: {dist_str}")

# Prob threshold shift detail
print(f"\n  Prob threshold distribution:")
for label, col in [('OLD', 'prob_thresh_old'), ('NEW', 'prob_thresh_new')]:
    vals = m[col]
    print(f"    {label}: min={vals.min():.3f}  median={vals.median():.3f}  "
          f"max={vals.max():.3f}  std={vals.std():.3f}")

# TP/SL shift detail
print(f"\n  TP/SL distribution:")
for label, tp_col, sl_col in [('OLD', 'tp_pct_old', 'sl_pct_old'),
                                ('NEW', 'tp_pct_new', 'sl_pct_new')]:
    tp = m[tp_col]
    sl = m[sl_col]
    print(f"    {label}: TP min={tp.min():.2%} med={tp.median():.2%} max={tp.max():.2%}  "
          f"SL min={sl.min():.2%} med={sl.median():.2%} max={sl.max():.2%}")

# ==============================================================================
# 7. PER-SEED SCATTER — identify patterns
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  7. WINNERS vs LOSERS — Which ticker/seeds improved most?")
print(f"{'='*W}")

m['va_delta'] = m['val_a_avg_pnl_new'] - m['val_a_avg_pnl_old']
m['opt_n_delta'] = m['opt_count_new'] - m['opt_count_old']
m['va_n_delta'] = m['val_a_count_new'] - m['val_a_count_old']

# Top 10 improvers
print(f"\n  TOP 10 IMPROVERS (by VAL-A PnL delta):")
print(f"  {'Ticker':<7} {'Seed':>5} {'OLD VA':>8} {'NEW VA':>8} {'Δ VA':>7} "
      f"{'OLD N':>6} {'NEW N':>6} {'Δ N':>5}")
print(f"  {'─'*60}")
top = m.nlargest(10, 'va_delta')
for _, r in top.iterrows():
    print(f"  {r['ticker']:<7} {int(r['seed']):>5} "
          f"{r['val_a_avg_pnl_old']:>7.2%} {r['val_a_avg_pnl_new']:>7.2%} "
          f"{r['va_delta']:>+6.2%} "
          f"{int(r['val_a_count_old']):>6} {int(r['val_a_count_new']):>6} "
          f"{int(r['va_n_delta']):>+5d}")

# Top 10 losers
print(f"\n  TOP 10 LOSERS (by VAL-A PnL delta):")
print(f"  {'Ticker':<7} {'Seed':>5} {'OLD VA':>8} {'NEW VA':>8} {'Δ VA':>7} "
      f"{'OLD N':>6} {'NEW N':>6} {'Δ N':>5}")
print(f"  {'─'*60}")
bot = m.nsmallest(10, 'va_delta')
for _, r in bot.iterrows():
    print(f"  {r['ticker']:<7} {int(r['seed']):>5} "
          f"{r['val_a_avg_pnl_old']:>7.2%} {r['val_a_avg_pnl_new']:>7.2%} "
          f"{r['va_delta']:>+6.2%} "
          f"{int(r['val_a_count_old']):>6} {int(r['val_a_count_new']):>6} "
          f"{int(r['va_n_delta']):>+5d}")

# ==============================================================================
# 8. CORRELATION: trade count change vs PnL change
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  8. CORRELATION — Does losing trades (V=0 removal) help or hurt PnL?")
print(f"{'='*W}")

m['opt_n_pct_change'] = (m['opt_count_new'] - m['opt_count_old']) / m['opt_count_old'].clip(lower=1)

corr_opt_n_va = m['opt_n_pct_change'].corr(m['va_delta'])
print(f"  Correlation(Δ% opt_N, Δ va_pnl):  {corr_opt_n_va:>+.3f}")
if corr_opt_n_va < -0.1:
    print(f"  → Losing trades correlates with BETTER OOS PnL (removed trades were hurting)")
elif corr_opt_n_va > 0.1:
    print(f"  → Losing trades correlates with WORSE OOS PnL (removed trades were helping)")
else:
    print(f"  → No clear relationship between trade count change and PnL change")

# Bucket analysis: tickers that lost >20% trades vs those that didn't
big_drop = m[m['opt_n_pct_change'] < -0.20]
small_change = m[m['opt_n_pct_change'].abs() <= 0.20]

if len(big_drop) > 0 and len(small_change) > 0:
    print(f"\n  Tickers that LOST >20% of trades (heavily contaminated):")
    print(f"    N={len(big_drop)}  avg Δ VA PnL = {big_drop['va_delta'].mean():>+.2%}  "
          f"avg old VA PnL = {big_drop['val_a_avg_pnl_old'].mean():.2%}  "
          f"avg new VA PnL = {big_drop['val_a_avg_pnl_new'].mean():.2%}")
    print(f"  Tickers with <20% trade count change:")
    print(f"    N={len(small_change)}  avg Δ VA PnL = {small_change['va_delta'].mean():>+.2%}  "
          f"avg old VA PnL = {small_change['val_a_avg_pnl_old'].mean():.2%}  "
          f"avg new VA PnL = {small_change['val_a_avg_pnl_new'].mean():.2%}")

# ==============================================================================
# 9. DEPLOYED TICKERS — Focus on what matters
# ==============================================================================
DEPLOYED_105 = {'RIOT', 'APLD', 'KEEL'}

deployed_m = m[m['ticker'].isin(DEPLOYED_105)]
if not deployed_m.empty:
    print(f"\n\n{'='*W}")
    print(f"  9. DEPLOYED TICKERS ONLY — {', '.join(sorted(DEPLOYED_105))}")
    print(f"{'='*W}")
    print(f"  {'Ticker':<7} {'Seed':>5} "
          f"{'OLD opt':>8} {'NEW opt':>8} "
          f"{'OLD VA':>8} {'NEW VA':>8} {'Δ VA':>7} "
          f"{'OLD N':>6} {'NEW N':>6} "
          f"{'Surv':>6}")
    print(f"  {'─'*80}")

    for _, r in deployed_m.sort_values(['ticker', 'seed']).iterrows():
        surv_old = 'P' if r['survived_a_old'] else 'F'
        surv_new = 'P' if r['survived_a_new'] else 'F'
        print(f"  {r['ticker']:<7} {int(r['seed']):>5} "
              f"{r['opt_avg_pnl_old']:>7.2%} {r['opt_avg_pnl_new']:>7.2%} "
              f"{r['val_a_avg_pnl_old']:>7.2%} {r['val_a_avg_pnl_new']:>7.2%} "
              f"{r['va_delta']:>+6.2%} "
              f"{int(r['val_a_count_old']):>6} {int(r['val_a_count_new']):>6} "
              f"{surv_old}→{surv_new:>4}")

    print(f"  {'─'*80}")
    print(f"  Deployed avg VAL-A:  old={deployed_m['val_a_avg_pnl_old'].mean():.2%}  "
          f"new={deployed_m['val_a_avg_pnl_new'].mean():.2%}  "
          f"Δ={deployed_m['va_delta'].mean():>+.2%}")

# ==============================================================================
# SUMMARY
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  SUMMARY")
print(f"{'='*W}")

improved = (m['va_delta'] > 0).sum()
degraded = (m['va_delta'] < 0).sum()
flat = (m['va_delta'] == 0).sum()

print(f"  VAL-A PnL:      {improved} improved, {degraded} degraded, {flat} flat")
print(f"  Avg OPT PnL:    old={m['opt_avg_pnl_old'].mean():.2%}  new={m['opt_avg_pnl_new'].mean():.2%}")
print(f"  Avg VAL-A PnL:  old={m['val_a_avg_pnl_old'].mean():.2%}  new={m['val_a_avg_pnl_new'].mean():.2%}")
print(f"  Avg OPT N:      old={m['opt_count_old'].mean():.1f}  new={m['opt_count_new'].mean():.1f}")
print(f"  Avg VAL-A N:    old={m['val_a_count_old'].mean():.1f}  new={m['val_a_count_new'].mean():.1f}")
print(f"  PnL decay:      old={o_decay_all:.0f}%  new={n_decay_all:.0f}%")
print(f"  Survived A:     old={int(m['survived_a_old'].sum())}/{len(m)}  "
      f"new={int(m['survived_a_new'].sum())}/{len(m)}")

print(f"\n  NOTE: v2 has {new['seed'].nunique()} seeds vs v1 has {old['seed'].nunique()} seeds.")
if new['seed'].nunique() < 15:
    print(f"  Results are preliminary — retrain all 15 seeds for conclusive comparison.")

print(f"\n{'='*W}")
