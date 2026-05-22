"""
analyze_ticker_deep.py — Deep per-ticker WF analysis
======================================================
Drills into a single ticker across all experiments/seeds.
Shows avg PnL, win rate, total PnL, and trade count to help
decide which experiment + seed to deploy.

Set TARGETS below to the tickers you want to analyze.
"""

import pandas as pd
import numpy as np

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass

from google.cloud import bigquery

BQ_TABLE = "ys-time-series.scoring_logs.wf_optimization_results"
TICKER_RENAME = {'MIGI': 'BGDE', 'BITF': 'KEEL'}
EXCLUDE_TICKERS = {'TLT', 'VIX'}

W = 140

# ==============================================================================
# LOAD
# ==============================================================================
client = bigquery.Client()
df = client.query(f"SELECT * FROM `{BQ_TABLE}`").to_dataframe()
df['ticker'] = df['ticker'].replace(TICKER_RENAME)
df = df[~df['ticker'].isin(EXCLUDE_TICKERS)]
df['survived_a'] = df['survived_a'].fillna(False).astype(bool)
df['survived_b'] = df['survived_b'].fillna(False).astype(bool)
df['val_b_avg_pnl'] = df['val_b_avg_pnl'].fillna(0.0)
df['val_b_win_rate'] = df['val_b_win_rate'].fillna(0.0)
df['val_b_count'] = df['val_b_count'].fillna(0).astype(int)
if 'min_vol_bars' not in df.columns:
    df['min_vol_bars'] = 0
df['min_vol_bars'] = df['min_vol_bars'].fillna(0).astype(int)

# Compute total PnL = avg_pnl * count (sum of per-trade returns)
df['val_a_total_pnl'] = df['val_a_avg_pnl'] * df['val_a_count']
df['val_b_total_pnl'] = df['val_b_avg_pnl'] * df['val_b_count']

# ── SET TICKERS TO ANALYZE HERE ──────────────────────────────────────────────
TARGETS = ['APLD']
# TARGETS = ['APLD', 'BBAI', 'RCAT']  # multiple tickers

for TARGET in TARGETS:
    td = df[df['ticker'] == TARGET]
    if td.empty:
        print(f"\n  No data for {TARGET} in BQ table.")
        continue

    print(f"\n{'='*W}")
    print(f"  DEEP ANALYSIS: {TARGET}")
    print(f"{'='*W}")

    # ==========================================================================
    # TABLE A: PER-EXPERIMENT SUMMARY (survivors only)
    # ==========================================================================
    print(f"\n  TABLE A: PER-EXPERIMENT SUMMARY (survivors only)")
    print(f"  {'─'*W}")
    print(f"  {'Experiment':<16} {'Seeds':>5} {'PassA':>7} {'Rate':>6} "
          f"{'|':>2} {'Avg PnL':>8} {'Med PnL':>8} {'Best PnL':>9} "
          f"{'|':>2} {'Avg WR':>7} {'Med WR':>7} {'Best WR':>8} "
          f"{'|':>2} {'Med TotPnL':>11} {'Best TotPnL':>12} {'Med N':>6}")
    print(f"  {'─'*16} {'─'*5} {'─'*7} {'─'*6} "
          f"{'─'*2} {'─'*8} {'─'*8} {'─'*9} "
          f"{'─'*2} {'─'*7} {'─'*7} {'─'*8} "
          f"{'─'*2} {'─'*11} {'─'*12} {'─'*6}")

    for exp in sorted(td['experiment'].unique()):
        ed = td[td['experiment'] == exp]
        n = len(ed)
        surv = ed[ed['survived_a']]
        pa = len(surv)

        if pa > 0:
            avg_pnl  = surv['val_a_avg_pnl'].mean()
            med_pnl  = surv['val_a_avg_pnl'].median()
            best_pnl = surv['val_a_avg_pnl'].max()
            avg_wr   = surv['val_a_win_rate'].mean()
            med_wr   = surv['val_a_win_rate'].median()
            best_wr  = surv['val_a_win_rate'].max()
            med_tot  = surv['val_a_total_pnl'].median()
            best_tot = surv['val_a_total_pnl'].max()
            med_n    = surv['val_a_count'].median()
        else:
            avg_pnl = med_pnl = best_pnl = 0
            avg_wr = med_wr = best_wr = 0
            med_tot = best_tot = 0
            med_n = 0

        print(f"  {exp:<16} {n:>5} {pa:>4}/{n:<2} {pa/n:>5.0%} "
              f"{'|':>2} {avg_pnl:>7.2%} {med_pnl:>7.2%} {best_pnl:>8.2%} "
              f"{'|':>2} {avg_wr:>6.0%} {med_wr:>6.0%} {best_wr:>7.0%} "
              f"{'|':>2} {med_tot:>10.2%} {best_tot:>11.2%} {med_n:>6.0f}")

    # ==========================================================================
    # TABLE B: ALL SURVIVING SEEDS (sorted by avg PnL)
    # ==========================================================================
    surv_all = td[td['survived_a']].sort_values('val_a_avg_pnl', ascending=False)

    print(f"\n  TABLE B: ALL SURVIVING SEEDS — top 25 by Avg PnL")
    print(f"  {'─'*W}")
    print(f"  {'Experiment':<16} {'Seed':>5} {'Avg PnL':>8} {'WinRate':>8} "
          f"{'TotPnL':>8} {'Trades':>6} {'Freq':>5} "
          f"{'|':>2} {'Prob':>6} {'TP':>5} {'SL':>5} {'Entry':>6} "
          f"{'VIX<':>5} {'TLT>':>6} {'SPY<':>5} {'MVB':>4} "
          f"{'|':>2} {'ValB PnL':>9} {'ValB WR':>8} {'ValB N':>7}")
    print(f"  {'─'*16} {'─'*5} {'─'*8} {'─'*8} "
          f"{'─'*8} {'─'*6} {'─'*5} "
          f"{'─'*2} {'─'*6} {'─'*5} {'─'*5} {'─'*6} "
          f"{'─'*5} {'─'*6} {'─'*5} {'─'*4} "
          f"{'─'*2} {'─'*9} {'─'*8} {'─'*7}")

    for _, r in surv_all.head(25).iterrows():
        print(f"  {r['experiment']:<16} {int(r['seed']):>5} "
              f"{r['val_a_avg_pnl']:>7.2%} {r['val_a_win_rate']:>7.0%} "
              f"{r['val_a_total_pnl']:>7.2%} {int(r['val_a_count']):>6} "
              f"{r['val_a_freq']:>5.1f} "
              f"{'|':>2} {r['prob_thresh']:>6.3f} "
              f"{r['tp_pct']:>4.0%} {r['sl_pct']:>4.0%} "
              f"{r['entry_time']:>6} {r['vix_threshold']:>5.0f} "
              f"{r['tlt_crash']:>6.2f} {r['spy_ignore']:>5.2f} "
              f"{int(r['min_vol_bars']):>4} "
              f"{'|':>2} {r['val_b_avg_pnl']:>8.2%} "
              f"{r['val_b_win_rate']:>7.0%} {int(r['val_b_count']):>7}")

    # ==========================================================================
    # TABLE C: RANKING BY DIFFERENT CRITERIA
    # ==========================================================================
    print(f"\n  TABLE C: BEST SEED BY DIFFERENT SELECTION CRITERIA")
    print(f"  {'─'*W}")

    criteria = [
        ('Highest Avg PnL',   'val_a_avg_pnl'),
        ('Highest Win Rate',  'val_a_win_rate'),
        ('Highest Total PnL', 'val_a_total_pnl'),
    ]

    print(f"  {'Criterion':<22} {'Experiment':<16} {'Seed':>5} "
          f"{'Avg PnL':>8} {'WinRate':>8} {'TotPnL':>8} {'Trades':>6} "
          f"{'|':>2} {'Prob':>6} {'TP':>5} {'SL':>5} {'Entry':>6}")
    print(f"  {'─'*22} {'─'*16} {'─'*5} "
          f"{'─'*8} {'─'*8} {'─'*8} {'─'*6} "
          f"{'─'*2} {'─'*6} {'─'*5} {'─'*5} {'─'*6}")

    for label, col in criteria:
        if surv_all.empty:
            print(f"  {label:<22} — no survivors —")
            continue
        best = surv_all.loc[surv_all[col].idxmax()]
        print(f"  {label:<22} {best['experiment']:<16} {int(best['seed']):>5} "
              f"{best['val_a_avg_pnl']:>7.2%} {best['val_a_win_rate']:>7.0%} "
              f"{best['val_a_total_pnl']:>7.2%} {int(best['val_a_count']):>6} "
              f"{'|':>2} {best['prob_thresh']:>6.3f} "
              f"{best['tp_pct']:>4.0%} {best['sl_pct']:>4.0%} "
              f"{best['entry_time']:>6}")

    # ==========================================================================
    # TABLE D: EXPERIMENT-LEVEL RANKING BY EACH METRIC
    # ==========================================================================
    print(f"\n  TABLE D: EXPERIMENT RANKING (median of survivors)")
    print(f"  {'─'*W}")

    exp_list = sorted(td['experiment'].unique())
    exp_agg = []
    for exp in exp_list:
        surv = td[(td['experiment'] == exp) & td['survived_a']]
        if surv.empty:
            continue
        exp_agg.append({
            'experiment': exp,
            'med_avg_pnl': surv['val_a_avg_pnl'].median(),
            'med_wr': surv['val_a_win_rate'].median(),
            'med_total_pnl': surv['val_a_total_pnl'].median(),
            'med_trades': surv['val_a_count'].median(),
            'survivors': len(surv),
        })

    if exp_agg:
        ea = pd.DataFrame(exp_agg)

        for metric, label in [('med_avg_pnl', 'By Median Avg PnL'),
                              ('med_wr', 'By Median Win Rate'),
                              ('med_total_pnl', 'By Median Total PnL')]:
            ranked = ea.sort_values(metric, ascending=False)
            print(f"\n  {label}:")
            print(f"    {'#':>2} {'Experiment':<16} {'Med AvgPnL':>10} {'Med WR':>7} "
                  f"{'Med TotPnL':>11} {'Med Trades':>10} {'Survivors':>9}")
            for i, (_, r) in enumerate(ranked.iterrows()):
                print(f"    {i+1:>2} {r['experiment']:<16} {r['med_avg_pnl']:>9.2%} "
                      f"{r['med_wr']:>6.0%} {r['med_total_pnl']:>10.2%} "
                      f"{r['med_trades']:>10.0f} {int(r['survivors']):>9}")

    print(f"\n{'='*W}")
