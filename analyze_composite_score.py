"""
analyze_composite_score.py — Composite Score Seed Selection Analysis
=====================================================================
Tests the composite scoring formula:
    score = avg_pnl × win_rate × log2(1 + N)

For each ticker, shows:
  TABLE A: Top 10 seeds ranked by composite score (vs old avg-PnL ranking)
  TABLE B: Experiment ranking by median composite score
  TABLE C: Side-by-side comparison — old logic vs new logic pick

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

# Derived metrics
df['val_a_total_pnl'] = df['val_a_avg_pnl'] * df['val_a_count']
df['composite'] = df['val_a_avg_pnl'] * df['val_a_win_rate'] * np.log2(1 + df['val_a_count'])

# ==============================================================================
# OLD LOGIC (from analyze_wf_results.py): pass_rate → med_pnl → best avg_pnl
# ==============================================================================
def old_logic_pick(td):
    exp_stats = td.groupby('experiment').agg(
        pass_a=('survived_a', 'sum'),
        total=('survived_a', 'count'),
    )
    exp_stats['rate'] = exp_stats['pass_a'] / exp_stats['total']
    survived_med = td[td['survived_a']].groupby('experiment')['val_a_avg_pnl'].median()
    exp_stats['med_pnl'] = exp_stats.index.map(lambda e: survived_med.get(e, 0.0))
    best_exp = exp_stats.sort_values(['rate', 'med_pnl'], ascending=False).index[0]
    candidates = td[(td['experiment'] == best_exp) & td['survived_a']]
    if len(candidates) == 0:
        candidates = td[td['experiment'] == best_exp]
    return candidates.loc[candidates['val_a_avg_pnl'].idxmax()]

# ==============================================================================
# NEW LOGIC: best composite seed globally, min 3 trades
# ==============================================================================
MIN_TRADES = 3

def new_logic_pick(td):
    surv = td[td['survived_a']]
    if surv.empty:
        return td.loc[td['composite'].idxmax()]
    eligible = surv[surv['val_a_count'] >= MIN_TRADES]
    if eligible.empty:
        return surv.loc[surv['composite'].idxmax()]
    return eligible.loc[eligible['composite'].idxmax()]

# ── SET TICKERS TO ANALYZE HERE ──────────────────────────────────────────────
TARGETS = [
    'APLD', 'BBAI', 'ACHR', 'RCAT', 'IREN',
    'RIOT', 'BGDE', 'KEEL', 'VRT', 'KTOS',
    'LUNR', 'RKLB', 'NNDM', 'JOBY', 'ASTS',
    'HUT', 'BTBT', 'WULF', 'CIFR', 'CLSK', 'BE',
]

print(f"{'='*W}")
print(f"  COMPOSITE SCORE ANALYSIS: score = avg_pnl × win_rate × log2(1 + N)")
print(f"{'='*W}")

summary_rows = []

for TARGET in TARGETS:
    td = df[df['ticker'] == TARGET]
    if td.empty:
        print(f"\n  No data for {TARGET} in BQ table.")
        continue

    surv = td[td['survived_a']].copy()

    print(f"\n\n{'='*W}")
    print(f"  {TARGET}")
    print(f"{'='*W}")

    # ==========================================================================
    # TABLE A: TOP 10 SEEDS BY COMPOSITE SCORE
    # ==========================================================================
    print(f"\n  TABLE A: TOP 10 SEEDS BY COMPOSITE SCORE")
    print(f"  {'─'*(W-2)}")
    print(f"  {'#':>2} {'Experiment':<16} {'Seed':>5} "
          f"{'AvgPnL':>7} {'WR':>5} {'N':>3} {'TotPnL':>7} "
          f"{'Composite':>10} "
          f"{'|':>2} {'Prob':>6} {'TP':>5} {'SL':>5} {'Entry':>6} "
          f"{'VIX<':>5} {'TLT>':>6} {'SPY<':>5} "
          f"{'|':>2} {'ValB%':>6} {'ValB WR':>7} {'ValB N':>6}")
    print(f"  {'─'*2} {'─'*16} {'─'*5} "
          f"{'─'*7} {'─'*5} {'─'*3} {'─'*7} "
          f"{'─'*10} "
          f"{'─'*2} {'─'*6} {'─'*5} {'─'*5} {'─'*6} "
          f"{'─'*5} {'─'*6} {'─'*5} "
          f"{'─'*2} {'─'*6} {'─'*7} {'─'*6}")

    if not surv.empty:
        top = surv.sort_values('composite', ascending=False).head(10)
        for i, (_, r) in enumerate(top.iterrows()):
            print(f"  {i+1:>2} {r['experiment']:<16} {int(r['seed']):>5} "
                  f"{r['val_a_avg_pnl']:>6.2%} {r['val_a_win_rate']:>4.0%} "
                  f"{int(r['val_a_count']):>3} {r['val_a_total_pnl']:>6.2%} "
                  f"{r['composite']:>10.6f} "
                  f"{'|':>2} {r['prob_thresh']:>6.3f} "
                  f"{r['tp_pct']:>4.0%} {r['sl_pct']:>4.0%} "
                  f"{r['entry_time']:>6} {r['vix_threshold']:>5.0f} "
                  f"{r['tlt_crash']:>6.2f} {r['spy_ignore']:>5.2f} "
                  f"{'|':>2} {r['val_b_avg_pnl']:>5.2%} "
                  f"{r['val_b_win_rate']:>6.0%} {int(r['val_b_count']):>6}")
    else:
        print(f"  — no survivors —")

    # ==========================================================================
    # TABLE B: EXPERIMENT RANKING BY MEDIAN COMPOSITE
    # ==========================================================================
    print(f"\n  TABLE B: EXPERIMENT RANKING (median composite of survivors)")
    print(f"  {'─'*(W-2)}")
    print(f"  {'#':>2} {'Experiment':<16} {'Med Comp':>9} {'Med AvgPnL':>10} "
          f"{'Med WR':>7} {'Med N':>6} {'Med TotPnL':>11} {'Survivors':>9}")
    print(f"  {'─'*2} {'─'*16} {'─'*9} {'─'*10} {'─'*7} {'─'*6} {'─'*11} {'─'*9}")

    exp_agg = []
    for exp in sorted(td['experiment'].unique()):
        s = surv[surv['experiment'] == exp]
        if s.empty:
            continue
        exp_agg.append({
            'experiment': exp,
            'med_composite': s['composite'].median(),
            'med_avg_pnl': s['val_a_avg_pnl'].median(),
            'med_wr': s['val_a_win_rate'].median(),
            'med_n': s['val_a_count'].median(),
            'med_total_pnl': s['val_a_total_pnl'].median(),
            'survivors': len(s),
        })

    if exp_agg:
        ea = pd.DataFrame(exp_agg).sort_values('med_composite', ascending=False)
        for i, (_, r) in enumerate(ea.iterrows()):
            print(f"  {i+1:>2} {r['experiment']:<16} {r['med_composite']:>9.6f} "
                  f"{r['med_avg_pnl']:>9.2%} {r['med_wr']:>6.0%} "
                  f"{r['med_n']:>6.0f} {r['med_total_pnl']:>10.2%} "
                  f"{int(r['survivors']):>9}")

    # ==========================================================================
    # TABLE C: OLD LOGIC vs NEW LOGIC
    # ==========================================================================
    print(f"\n  TABLE C: OLD LOGIC vs NEW LOGIC")
    print(f"  {'─'*(W-2)}")

    old = old_logic_pick(td)
    new = new_logic_pick(td)
    changed = (old['experiment'] != new['experiment']) or (int(old['seed']) != int(new['seed']))

    print(f"  {'Method':<14} {'Experiment':<16} {'Seed':>5} "
          f"{'AvgPnL':>7} {'WR':>5} {'N':>3} {'TotPnL':>7} {'Composite':>10}")
    print(f"  {'─'*14} {'─'*16} {'─'*5} {'─'*7} {'─'*5} {'─'*3} {'─'*7} {'─'*10}")
    print(f"  {'OLD (passrate)':<14} {old['experiment']:<16} {int(old['seed']):>5} "
          f"{old['val_a_avg_pnl']:>6.2%} {old['val_a_win_rate']:>4.0%} "
          f"{int(old['val_a_count']):>3} {old['val_a_total_pnl']:>6.2%} "
          f"{old['composite']:>10.6f}")
    print(f"  {'NEW (composite)':<14} {new['experiment']:<16} {int(new['seed']):>5} "
          f"{new['val_a_avg_pnl']:>6.2%} {new['val_a_win_rate']:>4.0%} "
          f"{int(new['val_a_count']):>3} {new['val_a_total_pnl']:>6.2%} "
          f"{new['composite']:>10.6f}")
    print(f"  → {'CHANGED' if changed else 'SAME'}")

    summary_rows.append({
        'ticker': TARGET,
        'old_exp': old['experiment'], 'old_seed': int(old['seed']),
        'old_avg_pnl': old['val_a_avg_pnl'], 'old_wr': old['val_a_win_rate'],
        'old_n': int(old['val_a_count']), 'old_composite': old['composite'],
        'new_exp': new['experiment'], 'new_seed': int(new['seed']),
        'new_avg_pnl': new['val_a_avg_pnl'], 'new_wr': new['val_a_win_rate'],
        'new_n': int(new['val_a_count']), 'new_composite': new['composite'],
        'changed': changed,
    })

# ==============================================================================
# SUMMARY: ALL TICKERS SIDE-BY-SIDE
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  SUMMARY: OLD vs NEW SELECTION ACROSS ALL {len(summary_rows)} TICKERS")
print(f"{'='*W}")
print(f"  {'Ticker':<7} {'|':>2} {'OLD Experiment':<16} {'Seed':>5} {'AvgPnL':>7} {'WR':>5} {'N':>3} "
      f"{'|':>2} {'NEW Experiment':<16} {'Seed':>5} {'AvgPnL':>7} {'WR':>5} {'N':>3} "
      f"{'|':>2} {'Δ':>7}")
print(f"  {'─'*7} {'─'*2} {'─'*16} {'─'*5} {'─'*7} {'─'*5} {'─'*3} "
      f"{'─'*2} {'─'*16} {'─'*5} {'─'*7} {'─'*5} {'─'*3} "
      f"{'─'*2} {'─'*7}")

n_changed = 0
for r in summary_rows:
    tag = "CHANGED" if r['changed'] else "same"
    if r['changed']:
        n_changed += 1
    print(f"  {r['ticker']:<7} "
          f"{'|':>2} {r['old_exp']:<16} {r['old_seed']:>5} "
          f"{r['old_avg_pnl']:>6.2%} {r['old_wr']:>4.0%} {r['old_n']:>3} "
          f"{'|':>2} {r['new_exp']:<16} {r['new_seed']:>5} "
          f"{r['new_avg_pnl']:>6.2%} {r['new_wr']:>4.0%} {r['new_n']:>3} "
          f"{'|':>2} {tag:>7}")

print(f"\n  {n_changed}/{len(summary_rows)} tickers changed experiment/seed with new logic")
print(f"{'='*W}")
