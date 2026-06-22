"""
analyze_wf_results.py — Cross-Experiment WF Optimization Analyzer
===================================================================
Queries BQ table `wf_optimization_results` and finds the best
experiment + config for each ticker across all logged experiments.

Selection logic:
  composite = avg_pnl × win_rate × log2(1 + N)
  Best seed = highest composite among survivors with N >= MIN_TRADES

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
BQ_TABLE = "ys-time-series.scoring_logs.wf_optimization_results"
DEPLOY_PNL_THRESHOLD = 0.018
MIN_TRADES = 3

CLEAN_EXP_BASES = ('105_DEFENSE', '106_DEFENSE', '152_XSECTOR', '160_XSECTOR',
                   '170_XSECTOR', '171_XSECTOR', '172_XSECTOR',
                   '173_XSECTOR', '174_XSECTOR', '175_XSECTOR')

W = 140

# ==============================================================================
# LOAD DATA
# ==============================================================================
print(f"{'='*W}")
print(f"  WF OPTIMIZATION ANALYZER — Cross-Experiment")
print(f"  Selection: composite = avg_pnl × win_rate × log2(1+N),  min N = {MIN_TRADES}")
print(f"{'='*W}")

client = bigquery.Client()
query = f"SELECT * FROM `{BQ_TABLE}` ORDER BY experiment, ticker, seed"
try:
    df = client.query(query).to_dataframe()
except Exception as e:
    if 'Not found' in str(e) or 'notFound' in str(e):
        print(f"\n  Table {BQ_TABLE} does not exist yet.")
        print("  Run a WF optimizer script first — it creates the table on first write.")
        raise SystemExit(0)
    raise

if df.empty:
    print("\n  No data in BQ table yet. Run WF optimizer scripts first.")
    raise SystemExit(0)

TICKER_RENAME = {'MIGI': 'BGDE', 'BITF': 'KEEL'}
EXCLUDE_TICKERS = {'TLT', 'VIX'}
df['ticker'] = df['ticker'].replace(TICKER_RENAME)
df = df[~df['ticker'].isin(EXCLUDE_TICKERS)]
df = df[df['experiment'].apply(lambda e: any(e == b or e.startswith(b + '_') for b in CLEAN_EXP_BASES))]
print(f"  Filtered to V=0-fixed experiments: {sorted(df['experiment'].unique().tolist())}")

df['survived_a'] = df['survived_a'].fillna(False).astype(bool)
df['survived_b'] = df['survived_b'].fillna(False).astype(bool)
df['val_b_avg_pnl'] = df['val_b_avg_pnl'].fillna(0.0)
if 'min_vol_bars' not in df.columns:
    df['min_vol_bars'] = 0
df['min_vol_bars'] = df['min_vol_bars'].fillna(0).astype(int)
if 'vol_filter_pct' not in df.columns:
    df['vol_filter_pct'] = 0
df['vol_filter_pct'] = df['vol_filter_pct'].fillna(0).astype(int)

df['val_a_total_pnl'] = df['val_a_avg_pnl'] * df['val_a_count']
df['composite'] = df['val_a_avg_pnl'] * df['val_a_win_rate'] * np.sqrt(df['val_a_count']) * np.log2(1 + df['val_a_count'])

# Keep only the most recent run per (experiment, ticker) — avoids discarding
# data from complete earlier runs when a newer partial re-run exists.
df['run_timestamp'] = pd.to_datetime(df['run_timestamp'])
_latest_per_et = (df.groupby(['experiment', 'ticker', 'run_id'])['run_timestamp']
                  .max().reset_index()
                  .sort_values('run_timestamp')
                  .groupby(['experiment', 'ticker']).last()
                  .reset_index()[['experiment', 'ticker', 'run_id']])
df = df.merge(_latest_per_et, on=['experiment', 'ticker', 'run_id'], how='inner')

experiments = sorted(df['experiment'].unique())
tickers = sorted(df['ticker'].unique())

# ==============================================================================
# TABLE 1: DATA OVERVIEW
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 1: DATA OVERVIEW")
print(f"{'='*W}")
print(f"  {'Experiment':<20} {'Rows':>6} {'Tickers':>8} {'Seeds':>6} "
      f"{'Opt Window':>28} {'Run Date':>22}")
print(f"  {'─'*20} {'─'*6} {'─'*8} {'─'*6} {'─'*28} {'─'*22}")

for exp in experiments:
    d = df[df['experiment'] == exp]
    n_tickers = d['ticker'].nunique()
    n_seeds = d['seed'].nunique()
    opt_window = f"{d['opt_start'].iloc[0]} → {d['opt_end'].iloc[0]}"
    run_ts = pd.to_datetime(d['run_timestamp'].max())
    run_str = run_ts.strftime('%Y-%m-%d %H:%M') if pd.notna(run_ts) else '?'
    print(f"  {exp:<20} {len(d):>6} {n_tickers:>8} {n_seeds:>6} "
          f"{opt_window:>28} {run_str:>22}")

print(f"\n  Total: {len(df)} rows, {len(tickers)} unique tickers, "
      f"{len(experiments)} experiments")

# ==============================================================================
# TABLE 2: CROSS-EXPERIMENT COMPARISON
# ==============================================================================
ticker_exp_pairs = df.groupby('ticker')['experiment'].nunique()
multi_exp_tickers = sorted(ticker_exp_pairs[ticker_exp_pairs > 1].index.tolist())

print(f"\n\n{'='*W}")
print(f"  TABLE 2: CROSS-EXPERIMENT COMPARISON ({len(multi_exp_tickers)} tickers in 2+ experiments)")
print(f"{'='*W}")

if multi_exp_tickers:
    print(f"  {'Ticker':<7} {'Experiment':<20} {'Seeds':>5} "
          f"{'PassA':>7} {'PassB':>7} {'Both':>6} "
          f"{'Med PnL':>8} {'Best PnL':>9} {'Best N':>6} {'Med Comp':>9}  Winner")
    print(f"  {'─'*7} {'─'*20} {'─'*5} {'─'*7} {'─'*7} {'─'*6} {'─'*8} {'─'*9} {'─'*6} {'─'*9}  {'─'*6}")

    for ticker in multi_exp_tickers:
        td = df[df['ticker'] == ticker]
        surv = td[td['survived_a'] & (td['val_a_count'] >= MIN_TRADES)]
        if not surv.empty:
            best_seed = surv.loc[surv['composite'].idxmax()]
            winner_exp = best_seed['experiment']
        else:
            surv_any = td[td['survived_a']]
            if not surv_any.empty:
                best_seed = surv_any.loc[surv_any['composite'].idxmax()]
                winner_exp = best_seed['experiment']
            else:
                winner_exp = None

        for i, exp in enumerate(sorted(td['experiment'].unique())):
            ed = td[td['experiment'] == exp]
            n = len(ed)
            pass_a = int(ed['survived_a'].sum())
            pass_b = int(ed['survived_b'].sum())
            pass_both = int((ed['survived_a'] & ed['survived_b']).sum())
            survived = ed[ed['survived_a']]
            med_pnl = survived['val_a_avg_pnl'].median() if len(survived) > 0 else 0.0
            med_comp = survived['composite'].median() if len(survived) > 0 else 0.0
            best_row = survived.loc[survived['val_a_avg_pnl'].idxmax()] if len(survived) > 0 else None
            best_pnl = best_row['val_a_avg_pnl'] if best_row is not None else 0.0
            best_n = int(best_row['val_a_count']) if best_row is not None else 0
            tag = " ←" if exp == winner_exp else ""
            print(f"  {ticker if i == 0 else '':<7} {exp:<20} {n:>5} "
                  f"{pass_a:>4}/{n} {pass_b:>4}/{n} "
                  f"{pass_both:>3}/{n} "
                  f"{med_pnl:>7.2%} {best_pnl:>8.2%} {best_n:>6} {med_comp:>9.6f}{tag}")
        print()
else:
    print("  No tickers appear in multiple experiments yet.")

# ==============================================================================
# TABLE 3: RECOMMENDED CONFIG PER TICKER (composite score selection)
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 3: RECOMMENDED CONFIG PER TICKER (composite score, min N={MIN_TRADES})")
print(f"{'='*W}")
print(f"  {'Ticker':<7} {'Experiment':<16} {'Seed':>5} {'Entry':>6} {'Prob':>6} "
      f"{'TP':>4} {'SL':>4} {'VIX<':>5} {'TLT>':>6} {'SPY<':>5} {'VF%':>4} "
      f"{'|':>2} {'VAL-A%':>7} {'WR':>5} {'N':>4} {'TotPnL':>7} {'Comp':>9}")
print(f"  {'─'*7} {'─'*16} {'─'*5} {'─'*6} {'─'*6} {'─'*4} {'─'*4} "
      f"{'─'*5} {'─'*6} {'─'*5} {'─'*4} {'─'*2} {'─'*7} {'─'*5} {'─'*4} {'─'*7} {'─'*9}")

recommendations = []

for ticker in tickers:
    td = df[df['ticker'] == ticker]

    surv = td[td['survived_a']]
    eligible = surv[surv['val_a_count'] >= MIN_TRADES]

    if not eligible.empty:
        best_row = eligible.loc[eligible['composite'].idxmax()]
        tag = "PASS"
    elif not surv.empty:
        best_row = surv.loc[surv['composite'].idxmax()]
        tag = "PASS"
    else:
        best_row = td.loc[td['composite'].idxmax()]
        tag = "FAIL"

    best_exp = best_row['experiment']

    print(f"  {ticker:<7} {best_exp:<16} {int(best_row['seed']):>5} "
          f"{best_row['entry_time']:>6} {best_row['prob_thresh']:>6.3f} "
          f"{best_row['tp_pct']:>3.0%} {best_row['sl_pct']:>3.0%} "
          f"{best_row['vix_threshold']:>5.0f} {best_row['tlt_crash']:>6.2f} "
          f"{best_row['spy_ignore']:>5.2f} {int(best_row['vol_filter_pct']):>4} "
          f"{'|':>2} {best_row['val_a_avg_pnl']:>6.2%} "
          f"{best_row['val_a_win_rate']:>4.0%} "
          f"{int(best_row['val_a_count']):>4} "
          f"{best_row['val_a_total_pnl']:>6.2%} "
          f"{best_row['composite']:>9.6f} {tag}")

    recommendations.append({
        'ticker': ticker,
        'experiment': best_exp,
        'seed': int(best_row['seed']),
        'entry_time': best_row['entry_time'],
        'entry_minute': int(best_row['entry_minute']),
        'prob_thresh': float(best_row['prob_thresh']),
        'tp_pct': float(best_row['tp_pct']),
        'sl_pct': float(best_row['sl_pct']),
        'vix_threshold': float(best_row['vix_threshold']),
        'tlt_crash': float(best_row['tlt_crash']),
        'spy_ignore': float(best_row['spy_ignore']),
        'vol_filter_pct': int(best_row['vol_filter_pct']),
        'val_a_avg_pnl': float(best_row['val_a_avg_pnl']),
        'val_a_win_rate': float(best_row['val_a_win_rate']),
        'val_a_count': int(best_row['val_a_count']),
        'val_a_total_pnl': float(best_row['val_a_total_pnl']),
        'composite': float(best_row['composite']),
        'survived': tag == "PASS",
    })

recs = pd.DataFrame(recommendations)

# ==============================================================================
# TABLE 4: DEPLOY-READY CONFIGS
# ==============================================================================
deploy = recs[
    (recs['survived']) &
    (recs['val_a_avg_pnl'] >= DEPLOY_PNL_THRESHOLD)
].sort_values('composite', ascending=False)

print(f"\n\n{'='*W}")
print(f"  TABLE 4: DEPLOY-READY CONFIGS (PnL >= {DEPLOY_PNL_THRESHOLD:.1%})")
print(f"{'='*W}")

if len(deploy) == 0:
    print("  No tickers meet deploy criteria yet.")
else:
    print(f"  {'Ticker':<7} {'Experiment':<16} {'Seed':>5} {'entry_min':>9} "
          f"{'prob':>6} {'TP':>5} {'SL':>5} "
          f"{'VIX<':>5} {'TLT>':>6} {'SPY<':>5} {'VF%':>4} "
          f"{'VAL-A%':>7} {'WR':>5} {'N':>4} {'TotPnL':>7} {'Comp':>9}")
    print(f"  {'─'*(W-2)}")

    for _, r in deploy.iterrows():
        print(f"  {r['ticker']:<7} {r['experiment']:<16} {r['seed']:>5} "
              f"{r['entry_minute']:>9} {r['prob_thresh']:>6.3f} "
              f"{r['tp_pct']:>4.0%} {r['sl_pct']:>4.0%} "
              f"{r['vix_threshold']:>5.0f} {r['tlt_crash']:>6.2f} {r['spy_ignore']:>5.2f} "
              f"{int(r['vol_filter_pct']):>4} "
              f"{r['val_a_avg_pnl']:>6.2%} {r['val_a_win_rate']:>4.0%} "
              f"{int(r['val_a_count']):>4} "
              f"{r['val_a_total_pnl']:>6.2%} "
              f"{r['composite']:>9.6f}")

    print(f"\n  {len(deploy)} tickers ready for deployment")

    print(f"\n  ── ASSET_CONFIGS snippet ──")
    for _, r in deploy.iterrows():
        print(f"    '{r['ticker']}': {{'seed': {r['seed']}, "
              f"'entry_minute': {r['entry_minute']}, "
              f"'prob_thresh': {r['prob_thresh']:.3f}, "
              f"'tp_pct': {r['tp_pct']:.3f}, 'sl_pct': {r['sl_pct']:.3f}, "
              f"'vix_threshold': {r['vix_threshold']:.1f}, "
              f"'tlt_crash': {r['tlt_crash']:.2f}, "
              f"'spy_ignore': {r['spy_ignore']:.2f}, "
              f"'vol_filter_pct': {int(r['vol_filter_pct'])}}},")

# ==============================================================================
# TABLE 5: DEAD TICKERS
# ==============================================================================
dead = recs[~recs['survived']].sort_values('val_a_avg_pnl', ascending=True)

print(f"\n\n{'='*W}")
print(f"  TABLE 5: DEAD TICKERS (no surviving seeds in best experiment)")
print(f"{'='*W}")

if len(dead) == 0:
    print("  All tickers have at least one surviving seed.")
else:
    print(f"  {'Ticker':<7} {'Experiment':<16} {'Best VAL-A%':>11} "
          f"{'Exps Tried':>10}")
    print(f"  {'─'*7} {'─'*16} {'─'*11} {'─'*10}")
    for _, r in dead.iterrows():
        n_exps = df[df['ticker'] == r['ticker']]['experiment'].nunique()
        print(f"  {r['ticker']:<7} {r['experiment']:<16} "
              f"{r['val_a_avg_pnl']:>10.2%} {n_exps:>10}")

# ==============================================================================
# SUMMARY
# ==============================================================================
n_pass = recs['survived'].sum()
n_deploy = len(deploy)
n_dead = len(dead)

print(f"\n\n{'='*W}")
print(f"  SUMMARY")
print(f"{'='*W}")
print(f"  Total tickers analyzed:  {len(recs)}")
print(f"  Surviving (PnL > 0):     {n_pass}")
print(f"  Deploy-ready:            {n_deploy}")
print(f"  Dead (no survivors):     {n_dead}")
print(f"  Experiments in data:     {', '.join(experiments)}")
print(f"  Selection method:        composite = avg_pnl × win_rate × log2(1+N), min N={MIN_TRADES}")
print(f"{'='*W}")
