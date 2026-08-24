"""
pull_v2_deploy_configs.py — Auto-discover best v2 deployment configs
====================================================================
Queries ALL v2_bar_is_real WF results from BQ, filters by:
  - VAL-A PnL ≥ 1.8%
  - VAL-A trade count ≥ 8  (≈1.0/month over 7.6-month validation)
  - survived_a = True
Then picks the best seed per ticker (highest VAL-A PnL) across all
experiments, so each ticker is deployed from exactly one experiment.
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
V2_EXPERIMENTS = ['105_DEFENSE', '106_DEFENSE', '152_XSECTOR']

MIN_VAL_PNL    = 0.018    # 1.8%
MIN_VAL_TRADES = 8        # ≈1.0/month over 7.6 months
VAL_MONTHS     = 7.6      # Oct 2025 → May 2026

client = bigquery.Client()

query = f"""
WITH latest_runs AS (
    SELECT experiment, ticker, seed,
           MAX(run_timestamp) AS max_ts
    FROM `{BQ_TABLE}`
    WHERE model_version = 'v2_bar_is_real'
      AND experiment IN ({', '.join(f"'{e}'" for e in V2_EXPERIMENTS)})
    GROUP BY experiment, ticker, seed
)
SELECT r.*
FROM `{BQ_TABLE}` r
JOIN latest_runs lr
  ON r.experiment = lr.experiment
 AND r.ticker = lr.ticker
 AND r.seed = lr.seed
 AND r.run_timestamp = lr.max_ts
WHERE r.model_version = 'v2_bar_is_real'
ORDER BY r.experiment, r.ticker, r.seed
"""

df = client.query(query).to_dataframe()

if df.empty:
    print("No v2 results found!")
    raise SystemExit(0)

print(f"Total v2 rows (deduplicated): {len(df)}")
print(f"Experiments: {df['experiment'].unique().tolist()}")
print(f"Unique tickers: {df['ticker'].nunique()}")

# ── Apply filters ────────────────────────────────────────────────────
qualified = df[
    (df['val_a_avg_pnl'] >= MIN_VAL_PNL) &
    (df['val_a_count'] >= MIN_VAL_TRADES) &
    (df['survived_a'] == True)
].copy()

print(f"\nAfter filters (PnL≥{MIN_VAL_PNL:.1%}, N≥{MIN_VAL_TRADES}, survived):")
print(f"  {len(qualified)} qualifying seed-ticker-experiment combos")
print(f"  {qualified['ticker'].nunique()} unique tickers")

if qualified.empty:
    print("No qualifying results!")
    raise SystemExit(0)

qualified['n_per_month'] = qualified['val_a_count'] / VAL_MONTHS

# ── Show ALL qualifying combos ───────────────────────────────────────
print(f"\n{'='*100}")
print(f"  ALL QUALIFYING COMBOS (sorted by ticker, then VAL-A PnL desc)")
print(f"{'='*100}")
print(f"  {'Ticker':<7} {'Experiment':<16} {'Seed':>6} {'VAL-A PnL':>10} {'N':>4} {'N/mo':>6} "
      f"{'OPT PnL':>9} {'OPT WR':>7} {'OPT N':>6}")
print(f"  {'─'*90}")

for _, r in qualified.sort_values(['ticker', 'val_a_avg_pnl'], ascending=[True, False]).iterrows():
    print(f"  {r['ticker']:<7} {r['experiment']:<16} {int(r['seed']):>6} "
          f"{r['val_a_avg_pnl']:>9.2%} {int(r['val_a_count']):>4} {r['n_per_month']:>5.1f} "
          f"{r['opt_avg_pnl']:>8.2%} {r['opt_win_rate']:>6.1%} {int(r['opt_count']):>5}")

# ── Pick best per ticker (highest VAL-A PnL) ────────────────────────
best_idx = qualified.groupby('ticker')['val_a_avg_pnl'].idxmax()
best = qualified.loc[best_idx].sort_values('val_a_avg_pnl', ascending=False)

print(f"\n\n{'='*100}")
print(f"  DEPLOYMENT SELECTION — BEST SEED PER TICKER")
print(f"  Criteria: VAL-A PnL ≥ {MIN_VAL_PNL:.1%}, N ≥ {MIN_VAL_TRADES}, survived_a=True")
print(f"  Selection: highest VAL-A PnL per ticker across {', '.join(V2_EXPERIMENTS)}")
print(f"{'='*100}")

for _, r in best.iterrows():
    vfp = int(r['vol_filter_pct']) if pd.notna(r.get('vol_filter_pct')) else 0
    n_mo = r['val_a_count'] / VAL_MONTHS
    print(f"""
  {r['ticker']} — {r['experiment']} seed {int(r['seed'])}
    VAL-A: {r['val_a_avg_pnl']:.2%} PnL, N={int(r['val_a_count'])} ({n_mo:.1f}/mo)
    OPT:   {r['opt_avg_pnl']:.2%} PnL, WR={r['opt_win_rate']:.1%}, N={int(r['opt_count'])}
    ─── config ───
    '{r['ticker']}': {{
        'seed': {int(r['seed'])}, 'prob_threshold': {r['prob_thresh']:.3f},
        'tp_pct': {r['tp_pct']:.2f}, 'sl_pct': {r['sl_pct']:.3f}, 'entry_minute': {int(r['entry_minute'])},
        'vix_threshold': {r['vix_threshold']:.1f}, 'tlt_crash': {r['tlt_crash']:.2f}, 'spy_ignore': {r['spy_ignore']:.2f},
        'vol_filter_pct': {vfp},
        'stats': {{'win_rate': '{r['opt_win_rate']:.0%}', 'avg_pnl': '{r['val_a_avg_pnl']:.1%}', 'freq': '{n_mo:.1f}/mo'}},
    }},""")

# ── Summary by experiment ────────────────────────────────────────────
print(f"\n{'='*100}")
print(f"  SUMMARY BY EXPERIMENT")
print(f"{'='*100}")
for exp in V2_EXPERIMENTS:
    exp_tickers = best[best['experiment'] == exp]['ticker'].tolist()
    if exp_tickers:
        print(f"  {exp}: {', '.join(exp_tickers)}")
    else:
        print(f"  {exp}: (none)")

total_n = best['val_a_count'].sum()
print(f"\n  Total deployed tickers: {len(best)}")
print(f"  Total VAL-A trades: {int(total_n)} ({total_n/VAL_MONTHS:.1f}/month)")
print(f"  Avg VAL-A PnL: {best['val_a_avg_pnl'].mean():.2%}")

# ── Tickers that ALMOST qualify ──────────────────────────────────────
near_miss = df[
    (df['val_a_avg_pnl'] >= MIN_VAL_PNL) &
    (df['val_a_count'] >= 6) &  # relaxed from 8
    (df['val_a_count'] < MIN_VAL_TRADES) &
    (df['survived_a'] == True) &
    (~df['ticker'].isin(best['ticker'].values))
].copy()

if not near_miss.empty:
    near_best = near_miss.loc[near_miss.groupby('ticker')['val_a_avg_pnl'].idxmax()]
    print(f"\n  NEAR MISSES (N=6-7, not already selected):")
    for _, r in near_best.sort_values('val_a_avg_pnl', ascending=False).iterrows():
        print(f"    {r['ticker']:<7} {r['experiment']:<16} S{int(r['seed'])} "
              f"VAL-A={r['val_a_avg_pnl']:.2%} N={int(r['val_a_count'])} "
              f"({r['val_a_count']/VAL_MONTHS:.1f}/mo)")
