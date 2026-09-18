"""
compare_magnitude_tp.py — Compare Fixed TP vs Magnitude-Based TP (v2)
=====================================================================
Fair comparison: applies S8_mix composite score + quality filters to BOTH
fixed and magnitude results. Minimum N=3 on TEST to exclude noise.

Shows whether magnitude models improve take-profit targeting.
Run in Colab after optimize_magnitude_tp.py completes.
"""

import pandas as pd
import numpy as np
from google.cloud import bigquery

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass

client = bigquery.Client()

TEST_START = '2026-04-01'
FIXED_VERSION = 'robust_wf_v8'
MAG_VERSION = 'mag_tp_v2'
MIN_TEST_N = 3

# ==============================================================================
# COMPOSITE SCORE SQL (same formula for both)
# ==============================================================================
COMPOSITE_SQL = """
    CASE WHEN (val_a_count + val_b_count) > 0
      THEN (val_a_avg_pnl * val_a_count + val_b_avg_pnl * val_b_count) /
           GREATEST(val_a_count + val_b_count, 1)
      ELSE 0 END AS wavg_pnl,
    CASE WHEN (val_a_count + val_b_count) > 0
      THEN (val_a_win_rate * val_a_count + val_b_win_rate * val_b_count) /
           GREATEST(val_a_count + val_b_count, 1)
      ELSE 0 END AS combined_wr
"""

S8_MIX_FILTER = """
    AND survived_a = TRUE
    AND ((val_a_avg_pnl >= 0.015 AND val_b_avg_pnl >= 0.010)
      OR (val_a_avg_pnl >= 0.010 AND val_b_avg_pnl >= 0.015))
    AND val_a_win_rate >= 0.50
    AND val_b_win_rate >= 0.50
"""

# ==============================================================================
# LOAD FIXED TP RESULTS (robust_wf_v8, S8_mix)
# ==============================================================================
print(f"Loading fixed TP results ({FIXED_VERSION}, S8_mix, N>={MIN_TEST_N})...")
q_fixed = f"""
WITH base AS (
  SELECT *,
    {COMPOSITE_SQL}
  FROM `{GCP_PROJECT}.scoring_logs.wf_optimization_results`
  WHERE model_version = '{FIXED_VERSION}'
    {S8_MIX_FILTER}
    AND val_b_start = '{TEST_START}'
    AND val_b_count >= {MIN_TEST_N}
),
ranked AS (
  SELECT *,
    ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY
      POW(wavg_pnl, 2) * POW(combined_wr, 2) * LOG(1 + val_a_count + val_b_count, 2) DESC
    ) as rn
  FROM base
)
SELECT ticker, experiment, seed, tpsl_mode, prob_thresh, entry_minute,
       val_a_avg_pnl, val_a_win_rate, val_a_count,
       val_b_avg_pnl, val_b_win_rate, val_b_count,
       sl_pct, tp_pct, sl_mult, tp_mult,
       wavg_pnl, combined_wr
FROM ranked WHERE rn = 1
"""
df_fixed = client.query(q_fixed).to_dataframe()
print(f"  {len(df_fixed)} tickers (N>={MIN_TEST_N} on TEST)")

# ==============================================================================
# LOAD MAGNITUDE TP RESULTS (mag_tp_v2, same quality filters)
# ==============================================================================
print(f"Loading magnitude TP results ({MAG_VERSION}, same filters, N>={MIN_TEST_N})...")
q_mag = f"""
WITH base AS (
  SELECT *,
    {COMPOSITE_SQL}
  FROM `{GCP_PROJECT}.scoring_logs.wf_optimization_results`
  WHERE model_version = '{MAG_VERSION}'
    {S8_MIX_FILTER}
    AND val_b_start = '{TEST_START}'
    AND val_b_count >= {MIN_TEST_N}
),
ranked AS (
  SELECT *,
    ROW_NUMBER() OVER (PARTITION BY ticker, tpsl_mode ORDER BY
      POW(wavg_pnl, 2) * POW(combined_wr, 2) * LOG(1 + val_a_count + val_b_count, 2) DESC
    ) as rn
  FROM base
)
SELECT ticker, experiment, seed, tpsl_mode, prob_thresh, entry_minute,
       val_a_avg_pnl, val_a_win_rate, val_a_count,
       val_b_avg_pnl, val_b_win_rate, val_b_count,
       sl_pct, tp_pct,
       wavg_pnl, combined_wr
FROM ranked WHERE rn = 1
"""
df_mag = client.query(q_mag).to_dataframe()

mag_cls = df_mag[df_mag['tpsl_mode'].str.contains('cla', case=False, na=False)]
mag_reg = df_mag[df_mag['tpsl_mode'].str.contains('reg', case=False, na=False)]
print(f"  Classification TP: {len(mag_cls)} tickers (after S8_mix + N>={MIN_TEST_N})")
print(f"  Regression TP:     {len(mag_reg)} tickers (after S8_mix + N>={MIN_TEST_N})")

# ==============================================================================
# ALSO LOAD UNFILTERED MAGNITUDE FOR COVERAGE ANALYSIS
# ==============================================================================
q_all_mag = f"""
SELECT ticker, experiment, seed, tpsl_mode, prob_thresh,
       val_a_avg_pnl, val_a_win_rate, val_a_count,
       val_b_avg_pnl, val_b_win_rate, val_b_count,
       sl_pct, opt_avg_pnl, opt_win_rate, opt_count,
       survived_a, survived_b
FROM `{GCP_PROJECT}.scoring_logs.wf_optimization_results`
WHERE model_version = '{MAG_VERSION}'
  AND val_b_start = '{TEST_START}'
"""
df_all_mag = client.query(q_all_mag).to_dataframe()

# ==============================================================================
# HOW MANY TICKERS DROPPED BY QUALITY FILTERS?
# ==============================================================================
q_mag_unfiltered = f"""
SELECT DISTINCT ticker, tpsl_mode
FROM `{GCP_PROJECT}.scoring_logs.wf_optimization_results`
WHERE model_version = '{MAG_VERSION}'
  AND survived_a = TRUE
  AND val_b_start = '{TEST_START}'
"""
df_mag_unfiltered = client.query(q_mag_unfiltered).to_dataframe()
cls_unf = set(df_mag_unfiltered[df_mag_unfiltered['tpsl_mode'].str.contains('cla')]['ticker'])
reg_unf = set(df_mag_unfiltered[df_mag_unfiltered['tpsl_mode'].str.contains('reg')]['ticker'])

print(f"\n  Filter impact on magnitude results:")
print(f"    Classification: {len(cls_unf)} survived_a → {len(mag_cls)} after S8_mix+N>={MIN_TEST_N} "
      f"({len(cls_unf) - len(mag_cls)} dropped)")
print(f"    Regression:     {len(reg_unf)} survived_a → {len(mag_reg)} after S8_mix+N>={MIN_TEST_N} "
      f"({len(reg_unf) - len(mag_reg)} dropped)")

# ==============================================================================
# COMPARISON TABLE — only tickers with data
# ==============================================================================
W = 120
print(f"\n{'='*W}")
print(f"  FIXED TP vs MAGNITUDE TP — Fair Comparison")
print(f"  Both filtered: S8_mix (asym 1.5%/1.0%), WR>=50%, N>={MIN_TEST_N} on TEST")
print(f"  Ranked by: wavg_pnl² × combined_wr² × log2(1+N)")
print(f"  Fixed: {FIXED_VERSION}  |  Magnitude: {MAG_VERSION}")
print(f"{'='*W}")

fixed_tickers = set(df_fixed['ticker'].values)
cls_tickers = set(mag_cls['ticker'].values)
reg_tickers = set(mag_reg['ticker'].values)
all_tickers = sorted(fixed_tickers | cls_tickers | reg_tickers)

print(f"\n  {'Ticker':<7} {'Fixed PnL':>9} {'Fixed WR':>8} {'FxN':>4}"
      f"  {'CLS PnL':>8} {'CLS WR':>7} {'ClN':>4}"
      f"  {'REG PnL':>8} {'REG WR':>7} {'RgN':>4}"
      f"  {'Winner':<8}")
print(f"  " + "-" * (W - 4))

summary = {'fixed_wins': 0, 'cls_wins': 0, 'reg_wins': 0, 'no_data': 0}

for ticker in all_tickers:
    fixed_row = df_fixed[df_fixed['ticker'] == ticker]
    cls_row = mag_cls[mag_cls['ticker'] == ticker]
    reg_row = mag_reg[mag_reg['ticker'] == ticker]

    f_pnl = float(fixed_row['val_b_avg_pnl'].iloc[0]) if len(fixed_row) > 0 else None
    f_wr = float(fixed_row['val_b_win_rate'].iloc[0]) if len(fixed_row) > 0 else None
    f_n = int(fixed_row['val_b_count'].iloc[0]) if len(fixed_row) > 0 else 0

    c_pnl = float(cls_row['val_b_avg_pnl'].iloc[0]) if len(cls_row) > 0 else None
    c_wr = float(cls_row['val_b_win_rate'].iloc[0]) if len(cls_row) > 0 else None
    c_n = int(cls_row['val_b_count'].iloc[0]) if len(cls_row) > 0 else 0

    r_pnl = float(reg_row['val_b_avg_pnl'].iloc[0]) if len(reg_row) > 0 else None
    r_wr = float(reg_row['val_b_win_rate'].iloc[0]) if len(reg_row) > 0 else None
    r_n = int(reg_row['val_b_count'].iloc[0]) if len(reg_row) > 0 else 0

    candidates = {}
    if f_pnl is not None: candidates['FIXED'] = f_pnl
    if c_pnl is not None: candidates['CLS'] = c_pnl
    if r_pnl is not None: candidates['REG'] = r_pnl

    if candidates:
        winner = max(candidates, key=candidates.get)
        if winner == 'FIXED': summary['fixed_wins'] += 1
        elif winner == 'CLS': summary['cls_wins'] += 1
        else: summary['reg_wins'] += 1
    else:
        winner = '—'
        summary['no_data'] += 1

    f_str = f"{f_pnl*100:>+8.2f}%" if f_pnl is not None else f"{'—':>9}"
    f_wr_s = f"{f_wr*100:>7.0f}%" if f_wr is not None else f"{'—':>8}"
    f_n_s = f"{f_n:>4}" if f_n > 0 else f"{'—':>4}"

    c_str = f"{c_pnl*100:>+7.2f}%" if c_pnl is not None else f"{'—':>8}"
    c_wr_s = f"{c_wr*100:>6.0f}%" if c_wr is not None else f"{'—':>7}"
    c_n_s = f"{c_n:>4}" if c_n > 0 else f"{'—':>4}"

    r_str = f"{r_pnl*100:>+7.2f}%" if r_pnl is not None else f"{'—':>8}"
    r_wr_s = f"{r_wr*100:>6.0f}%" if r_wr is not None else f"{'—':>7}"
    r_n_s = f"{r_n:>4}" if r_n > 0 else f"{'—':>4}"

    marker = " <--" if winner != 'FIXED' and winner != '—' else ""
    print(f"  {ticker:<7} {f_str} {f_wr_s} {f_n_s}  {c_str} {c_wr_s} {c_n_s}  {r_str} {r_wr_s} {r_n_s}  {winner:<8}{marker}")

# ==============================================================================
# SUMMARY
# ==============================================================================
print(f"\n  {'='*W}")
print(f"  WINNER COUNT (all tickers with any data)")
print(f"  {'='*W}")
print(f"  Fixed TP wins:          {summary['fixed_wins']} tickers")
print(f"  Classification TP wins: {summary['cls_wins']} tickers")
print(f"  Regression TP wins:     {summary['reg_wins']} tickers")

# Head-to-head: all 3 methods available
both_tickers = fixed_tickers & cls_tickers & reg_tickers
if both_tickers:
    f_avg = df_fixed[df_fixed['ticker'].isin(both_tickers)]['val_b_avg_pnl'].mean()
    c_avg = mag_cls[mag_cls['ticker'].isin(both_tickers)]['val_b_avg_pnl'].mean()
    r_avg = mag_reg[mag_reg['ticker'].isin(both_tickers)]['val_b_avg_pnl'].mean()
    f_wr_avg = df_fixed[df_fixed['ticker'].isin(both_tickers)]['val_b_win_rate'].mean()
    c_wr_avg = mag_cls[mag_cls['ticker'].isin(both_tickers)]['val_b_win_rate'].mean()
    r_wr_avg = mag_reg[mag_reg['ticker'].isin(both_tickers)]['val_b_win_rate'].mean()
    f_n_avg = df_fixed[df_fixed['ticker'].isin(both_tickers)]['val_b_count'].mean()
    c_n_avg = mag_cls[mag_cls['ticker'].isin(both_tickers)]['val_b_count'].mean()
    r_n_avg = mag_reg[mag_reg['ticker'].isin(both_tickers)]['val_b_count'].mean()

    print(f"\n  HEAD-TO-HEAD ({len(both_tickers)} tickers with all 3 methods, all N>={MIN_TEST_N}):")
    print(f"    {'Method':<20} {'Avg PnL':>9} {'Avg WR':>8} {'Avg N':>7}")
    print(f"    {'-'*46}")
    print(f"    {'Fixed TP':<20} {f_avg*100:>+8.2f}% {f_wr_avg*100:>7.0f}% {f_n_avg:>7.1f}")
    print(f"    {'Classification TP':<20} {c_avg*100:>+8.2f}% {c_wr_avg*100:>7.0f}% {c_n_avg:>7.1f}")
    print(f"    {'Regression TP':<20} {r_avg*100:>+8.2f}% {r_wr_avg*100:>7.0f}% {r_n_avg:>7.1f}")

    best = max([('Fixed', f_avg), ('Classification', c_avg), ('Regression', r_avg)], key=lambda x: x[1])
    print(f"    Winner: {best[0]} ({best[1]*100:+.2f}%)")

    # Count wins in head-to-head only
    h2h = {'FIXED': 0, 'CLS': 0, 'REG': 0}
    for t in both_tickers:
        fp = float(df_fixed[df_fixed['ticker']==t]['val_b_avg_pnl'].iloc[0])
        cp = float(mag_cls[mag_cls['ticker']==t]['val_b_avg_pnl'].iloc[0])
        rp = float(mag_reg[mag_reg['ticker']==t]['val_b_avg_pnl'].iloc[0])
        w = max([('FIXED',fp),('CLS',cp),('REG',rp)], key=lambda x:x[1])[0]
        h2h[w] += 1
    print(f"    Per-ticker wins: Fixed={h2h['FIXED']}  CLS={h2h['CLS']}  REG={h2h['REG']}")

# Fixed vs CLS only
fc_tickers = fixed_tickers & cls_tickers
if fc_tickers and (not both_tickers or len(fc_tickers) > len(both_tickers)):
    f_avg = df_fixed[df_fixed['ticker'].isin(fc_tickers)]['val_b_avg_pnl'].mean()
    c_avg = mag_cls[mag_cls['ticker'].isin(fc_tickers)]['val_b_avg_pnl'].mean()
    f_n_avg = df_fixed[df_fixed['ticker'].isin(fc_tickers)]['val_b_count'].mean()
    c_n_avg = mag_cls[mag_cls['ticker'].isin(fc_tickers)]['val_b_count'].mean()
    print(f"\n  Fixed vs Classification ({len(fc_tickers)} tickers, both N>={MIN_TEST_N}):")
    print(f"    Fixed:          {f_avg*100:+.2f}%  (avg N={f_n_avg:.1f})")
    print(f"    Classification: {c_avg*100:+.2f}%  (avg N={c_n_avg:.1f})")
    print(f"    Delta:          {(c_avg - f_avg)*100:+.2f}% per trade")

# Fixed vs REG only
fr_tickers = fixed_tickers & reg_tickers
if fr_tickers and (not both_tickers or len(fr_tickers) > len(both_tickers)):
    f_avg = df_fixed[df_fixed['ticker'].isin(fr_tickers)]['val_b_avg_pnl'].mean()
    r_avg = mag_reg[mag_reg['ticker'].isin(fr_tickers)]['val_b_avg_pnl'].mean()
    f_n_avg = df_fixed[df_fixed['ticker'].isin(fr_tickers)]['val_b_count'].mean()
    r_n_avg = mag_reg[mag_reg['ticker'].isin(fr_tickers)]['val_b_count'].mean()
    print(f"\n  Fixed vs Regression ({len(fr_tickers)} tickers, both N>={MIN_TEST_N}):")
    print(f"    Fixed:      {f_avg*100:+.2f}%  (avg N={f_n_avg:.1f})")
    print(f"    Regression: {r_avg*100:+.2f}%  (avg N={r_n_avg:.1f})")
    print(f"    Delta:      {(r_avg - f_avg)*100:+.2f}% per trade")

# ==============================================================================
# N-WEIGHTED COMPARISON (accounts for trade frequency)
# ==============================================================================
print(f"\n  {'='*W}")
print(f"  N-WEIGHTED COMPARISON — PnL weighted by number of trades")
print(f"  {'='*W}")

if both_tickers:
    f_total_pnl = sum(float(df_fixed[df_fixed['ticker']==t]['val_b_avg_pnl'].iloc[0]) *
                      int(df_fixed[df_fixed['ticker']==t]['val_b_count'].iloc[0])
                      for t in both_tickers)
    f_total_n = sum(int(df_fixed[df_fixed['ticker']==t]['val_b_count'].iloc[0]) for t in both_tickers)

    c_total_pnl = sum(float(mag_cls[mag_cls['ticker']==t]['val_b_avg_pnl'].iloc[0]) *
                      int(mag_cls[mag_cls['ticker']==t]['val_b_count'].iloc[0])
                      for t in both_tickers)
    c_total_n = sum(int(mag_cls[mag_cls['ticker']==t]['val_b_count'].iloc[0]) for t in both_tickers)

    r_total_pnl = sum(float(mag_reg[mag_reg['ticker']==t]['val_b_avg_pnl'].iloc[0]) *
                      int(mag_reg[mag_reg['ticker']==t]['val_b_count'].iloc[0])
                      for t in both_tickers)
    r_total_n = sum(int(mag_reg[mag_reg['ticker']==t]['val_b_count'].iloc[0]) for t in both_tickers)

    print(f"    {'Method':<20} {'Total Trades':>12} {'Total PnL':>10} {'Wt Avg PnL':>11}")
    print(f"    {'-'*55}")
    print(f"    {'Fixed TP':<20} {f_total_n:>12} {f_total_pnl*100:>+9.1f}% {f_total_pnl/max(f_total_n,1)*100:>+10.2f}%")
    print(f"    {'Classification TP':<20} {c_total_n:>12} {c_total_pnl*100:>+9.1f}% {c_total_pnl/max(c_total_n,1)*100:>+10.2f}%")
    print(f"    {'Regression TP':<20} {r_total_n:>12} {r_total_pnl*100:>+9.1f}% {r_total_pnl/max(r_total_n,1)*100:>+10.2f}%")

# ==============================================================================
# SEED COVERAGE
# ==============================================================================
print(f"\n\n  {'='*W}")
print(f"  SEED COVERAGE — magnitude optimization completeness")
print(f"  {'='*W}")

if not df_all_mag.empty:
    seed_summary = df_all_mag.groupby('tpsl_mode').agg(
        tickers=('ticker', 'nunique'),
        seeds_per_ticker=('seed', lambda x: x.groupby(df_all_mag.loc[x.index, 'ticker']).nunique().mean()),
        total_combos=('seed', 'count'),
        survived_pct=('survived_a', 'mean'),
    ).reset_index()

    print(f"\n  {'Mode':<10} {'Tickers':>8} {'Avg Seeds/Tk':>13} {'Total Combos':>13} {'Survived %':>11}")
    print(f"  " + "-" * 60)
    for _, r in seed_summary.iterrows():
        print(f"  {r['tpsl_mode']:<10} {int(r['tickers']):>8} {r['seeds_per_ticker']:>13.1f} "
              f"{int(r['total_combos']):>13} {r['survived_pct']*100:>10.0f}%")

# ==============================================================================
# VERDICT
# ==============================================================================
print(f"\n\n  {'='*W}")
print(f"  VERDICT")
print(f"  {'='*W}")

if both_tickers:
    if f_avg > c_avg and f_avg > r_avg:
        print(f"\n  Fixed TP wins head-to-head ({len(both_tickers)} tickers, all N>={MIN_TEST_N}).")
        print(f"  Magnitude models don't improve TP decisions with quality-filtered configs.")
        print(f"  Keep current S8_mix + fixed TP deployment.")
    else:
        winner_type = 'Classification' if c_avg > r_avg else 'Regression'
        delta = max(c_avg, r_avg) - f_avg
        print(f"\n  {winner_type} magnitude TP wins! +{delta*100:.2f}% per trade vs fixed.")
        print(f"  Consider deploying magnitude-based TP for qualifying tickers.")
else:
    print(f"\n  NO HEAD-TO-HEAD DATA — no tickers have all 3 methods with N>={MIN_TEST_N}.")
    print(f"  Try lowering MIN_TEST_N or wait for more optimization results.")

print(f"\n{'='*W}")
