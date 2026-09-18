from google.cloud import bigquery
import pandas as pd
import numpy as np

bq = bigquery.Client()

# What does pull_deploy_configs.py currently have deployed?
q1 = '''
SELECT ticker, experiment, seed, tpsl_mode, is_ensemble,
       opt_avg_pnl, opt_win_rate, opt_count,
       val_a_avg_pnl, val_a_win_rate, val_a_count,
       val_b_avg_pnl, val_b_win_rate, val_b_count,
       tp_pct, sl_pct, prob_thresh, entry_minute,
       grace_minutes, dip_pct, max_wait_minutes
FROM `{GCP_PROJECT}.scoring_logs.wf_optimization_results`
WHERE model_version = 'robust_wf_v12'
  AND survived_a = TRUE AND survived_b = TRUE
  AND (experiment LIKE '152%' OR experiment LIKE '160%')
  AND val_a_avg_pnl > 0 AND val_b_avg_pnl > 0
ORDER BY ticker, val_b_avg_pnl DESC
'''
df = bq.query(q1).to_dataframe()

print("="*100)
print("CONFIGS THAT SURVIVED BOTH val_a AND val_b (production deploy pool)")
print("="*100)

for ticker in sorted(df['ticker'].unique()):
    td = df[df['ticker'] == ticker]
    best = td.iloc[0]
    print(f"\n{ticker} ({len(td)} configs survived both)")
    print(f"  Best val_b: {best['tpsl_mode']:<15} vb={best['val_b_avg_pnl']:+.2%} "
          f"va={best['val_a_avg_pnl']:+.2%} opt={best['opt_avg_pnl']:+.2%} "
          f"vb_wr={best['val_b_win_rate']:.0%} tp={best['tp_pct']:.1%} sl={best['sl_pct']:.1%}")

# What would backtest_portfolio_v3 select with current blended composite?
q2 = '''
SELECT ticker, experiment, seed, tpsl_mode, is_ensemble,
       opt_avg_pnl, opt_win_rate, opt_count,
       val_a_avg_pnl, val_a_win_rate, val_a_count,
       val_b_avg_pnl, val_b_win_rate, val_b_count,
       tp_pct, sl_pct, prob_thresh, entry_minute
FROM `{GCP_PROJECT}.scoring_logs.wf_optimization_results`
WHERE model_version = 'robust_wf_v12'
  AND survived_a = TRUE AND val_a_count >= 3 AND val_a_avg_pnl > 0
  AND opt_avg_pnl > 0
  AND (experiment LIKE '152%' OR experiment LIKE '160%')
'''
df2 = bq.query(q2).to_dataframe()

# Apply overfit filter
df2 = df2[df2['val_a_avg_pnl'] / df2['opt_avg_pnl'].clip(lower=0.001) <= 3.0]

# Compute blended composite (current backtest formula)
total_n = df2['opt_count'] + df2['val_a_count']
wavg_pnl = (df2['opt_avg_pnl']*df2['opt_count'] + df2['val_a_avg_pnl']*df2['val_a_count']) / total_n
wavg_wr = (df2['opt_win_rate']*df2['opt_count'] + df2['val_a_win_rate']*df2['val_a_count']) / total_n
consistency = (1.0 - (df2['val_a_avg_pnl'] - df2['opt_avg_pnl']).abs() / df2['opt_avg_pnl'].clip(lower=0.001)).clip(lower=0.2)
df2['composite'] = wavg_pnl * wavg_wr * np.log2(1 + df2['val_a_count']) * consistency
df2['wf_avg_pnl'] = df2['val_a_avg_pnl']

# Select best per ticker
bt_selected = df2.loc[df2.groupby('ticker')['composite'].idxmax()]
bt_deploy = bt_selected[bt_selected['wf_avg_pnl'] >= 0.018]

print("\n\n" + "="*100)
print("BACKTEST WOULD SELECT (blended composite, 1.8% threshold)")
print("="*100)
for _, r in bt_deploy.sort_values('val_b_avg_pnl', ascending=False).iterrows():
    print(f"  {r['ticker']:<7} {r['tpsl_mode']:<15} vb={r['val_b_avg_pnl']:+.2%} "
          f"va={r['val_a_avg_pnl']:+.2%} opt={r['opt_avg_pnl']:+.2%} composite={r['composite']:.4f}")

# Compare
print("\n\n" + "="*100)
print("GAP ANALYSIS: Deploy pool vs Backtest selection")
print("="*100)
deploy_tickers = set(df['ticker'].unique())
bt_tickers = set(bt_deploy['ticker'].unique())
print(f"Deploy pool tickers: {len(deploy_tickers)}")
print(f"Backtest selection tickers: {len(bt_tickers)}")
print(f"Overlap: {len(deploy_tickers & bt_tickers)}")
print(f"In deploy only: {deploy_tickers - bt_tickers}")
print(f"In backtest only: {bt_tickers - deploy_tickers}")

# For overlapping tickers, compare which config each selects
for ticker in sorted(deploy_tickers & bt_tickers):
    deploy_best = df[df['ticker'] == ticker].iloc[0]
    bt_row = bt_deploy[bt_deploy['ticker'] == ticker].iloc[0]
    same = (deploy_best['tpsl_mode'] == bt_row['tpsl_mode'] and
            deploy_best['experiment'] == bt_row['experiment'])
    print(f"  {ticker:<7} Deploy: {deploy_best['tpsl_mode']:<15} vb={deploy_best['val_b_avg_pnl']:+.2%}  |  "
          f"BT: {bt_row['tpsl_mode']:<15} vb={bt_row['val_b_avg_pnl']:+.2%}  "
          f"{'SAME' if same else 'DIFFERENT'}")
