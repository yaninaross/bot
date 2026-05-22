"""
analyze_wf_rerun_status.py — WF Rerun Detector
================================================
Queries BQ table `wf_optimization_results` and identifies which
(experiment, ticker) pairs have stale rows from old MIN_TRADES=1.5/mo
runs vs current 2.0/mo runs.

Shows:
  TABLE 1: Run history per experiment (run_ids, timestamps, row counts)
  TABLE 2: Per-ticker rerun status (latest vs stale rows)
  TABLE 3: Action items (which experiments need reruns)
  TABLE 4: Stale rows that can be deleted

Run in Colab:  python analyze_wf_rerun_status.py
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
W = 120

client = bigquery.Client()
query = f"SELECT * FROM `{BQ_TABLE}` ORDER BY experiment, ticker, seed"
df = client.query(query).to_dataframe()

if df.empty:
    print("No data in BQ table.")
    raise SystemExit(0)

df['run_timestamp'] = pd.to_datetime(df['run_timestamp'], utc=True)
df['survived_a'] = df['survived_a'].fillna(False).astype(bool)

experiments = sorted(df['experiment'].unique())

# ==============================================================================
# TABLE 1: RUN HISTORY
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 1: RUN HISTORY PER EXPERIMENT")
print(f"{'='*W}")
print(f"  {'Experiment':<20} {'Run ID':<10} {'Timestamp':<22} {'Rows':>6} "
      f"{'Tickers':>8} {'Min Freq':>9} {'Max Freq':>9}  {'Inferred'}")
print(f"  {'─'*20} {'─'*10} {'─'*22} {'─'*6} {'─'*8} {'─'*9} {'─'*9}  {'─'*12}")

exp_runs = {}  # {exp: [{run_id, ts, rows, tickers, min_freq, is_latest}, ...]}

for exp in experiments:
    ed = df[df['experiment'] == exp]
    runs = []
    for rid, rd in ed.groupby('run_id'):
        ts = rd['run_timestamp'].max()
        runs.append({
            'run_id': rid,
            'ts': ts,
            'rows': len(rd),
            'tickers': rd['ticker'].nunique(),
            'min_freq': rd['opt_freq'].min(),
            'max_freq': rd['opt_freq'].max(),
            'ticker_list': sorted(rd['ticker'].unique()),
        })
    runs.sort(key=lambda x: x['ts'])
    latest_ts = runs[-1]['ts']
    for r in runs:
        r['is_latest'] = (r['ts'] == latest_ts)

    exp_runs[exp] = runs

    for r in runs:
        tag = "CURRENT" if r['is_latest'] else "STALE"
        if r['min_freq'] < 2.0:
            freq_tag = f"{tag} (freq<2.0)"
        else:
            freq_tag = tag
        print(f"  {exp:<20} {r['run_id'][:8]:<10} "
              f"{r['ts'].strftime('%Y-%m-%d %H:%M UTC'):<22} "
              f"{r['rows']:>6} {r['tickers']:>8} "
              f"{r['min_freq']:>9.2f} {r['max_freq']:>9.2f}  {freq_tag}")
    if len(runs) > 1:
        print()

print(f"\n  Total: {len(df)} rows across {len(experiments)} experiments")

# ==============================================================================
# TABLE 2: PER-TICKER RERUN STATUS
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 2: PER-TICKER RERUN STATUS")
print(f"{'='*W}")
print(f"  {'Experiment':<20} {'Ticker':<8} {'Total':>5} {'Latest':>6} "
      f"{'Stale':>5} {'Status':<20} {'Min Freq':>9}")
print(f"  {'─'*20} {'─'*8} {'─'*5} {'─'*6} {'─'*5} {'─'*20} {'─'*9}")

needs_rerun = {}  # {exp: [tickers that need rerun]}

for exp in experiments:
    runs = exp_runs[exp]
    ed = df[df['experiment'] == exp]

    if len(runs) == 1:
        r = runs[0]
        if r['min_freq'] < 2.0:
            status = "NEEDS RERUN (1.5/mo)"
            needs_rerun.setdefault(exp, []).extend(r['ticker_list'])
            for ticker in sorted(r['ticker_list']):
                td = ed[ed['ticker'] == ticker]
                print(f"  {exp:<20} {ticker:<8} {len(td):>5} {'--':>6} "
                      f"{'--':>5} {status:<20} {td['opt_freq'].min():>9.2f}")
        else:
            for ticker in sorted(r['ticker_list']):
                td = ed[ed['ticker'] == ticker]
                print(f"  {exp:<20} {ticker:<8} {len(td):>5} {len(td):>6} "
                      f"{'0':>5} {'OK (2.0/mo)  ':<20} {td['opt_freq'].min():>9.2f}")
    else:
        latest_rid = runs[-1]['run_id']
        for ticker in sorted(ed['ticker'].unique()):
            td = ed[ed['ticker'] == ticker]
            latest_rows = td[td['run_id'] == latest_rid]
            stale_rows = td[td['run_id'] != latest_rid]

            if len(latest_rows) > 0 and len(stale_rows) > 0:
                status = "RE-RUN + STALE"
            elif len(latest_rows) > 0:
                status = "OK (latest only)"
            else:
                status = "STALE ONLY"
                needs_rerun.setdefault(exp, []).append(ticker)

            print(f"  {exp:<20} {ticker:<8} {len(td):>5} "
                  f"{len(latest_rows):>6} {len(stale_rows):>5} "
                  f"{status:<20} {td['opt_freq'].min():>9.2f}")
    print()

# ==============================================================================
# TABLE 3: ACTION ITEMS
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 3: ACTION ITEMS")
print(f"{'='*W}")

for exp in experiments:
    runs = exp_runs[exp]
    n_runs = len(runs)
    total_rows = sum(r['rows'] for r in runs)
    latest = runs[-1]
    stale_rows = total_rows - latest['rows']

    if n_runs == 1 and latest['min_freq'] >= 2.0:
        print(f"  {exp:<20}  OK — single run with 2.0/mo ({latest['rows']} rows)")
    elif n_runs == 1 and latest['min_freq'] < 2.0:
        print(f"  {exp:<20}  RERUN ALL — only has 1.5/mo data "
              f"({latest['rows']} rows, {latest['tickers']} tickers)")
    else:
        rerun_tickers = needs_rerun.get(exp, [])
        if rerun_tickers:
            print(f"  {exp:<20}  PARTIAL — {len(rerun_tickers)} tickers need rerun: "
                  f"{', '.join(rerun_tickers)}")
            print(f"  {'':20}  + {stale_rows} stale rows to clean")
        else:
            print(f"  {exp:<20}  CLEANUP — all tickers re-run, "
                  f"but {stale_rows} stale rows to delete")

# ==============================================================================
# TABLE 4: STALE ROW CLEANUP
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 4: STALE ROWS (deletable)")
print(f"{'='*W}")

stale_run_ids = []
total_stale = 0

for exp in experiments:
    runs = exp_runs[exp]
    if len(runs) <= 1:
        continue
    latest_rid = runs[-1]['run_id']
    for r in runs:
        if not r['is_latest']:
            stale_run_ids.append(r['run_id'])
            total_stale += r['rows']
            print(f"  {exp:<20}  run_id={r['run_id'][:8]}...  "
                  f"{r['rows']:>4} rows  ({r['ts'].strftime('%Y-%m-%d %H:%M')})")

if not stale_run_ids:
    print("  No stale rows found — all experiments have single runs.")
else:
    print(f"\n  Total stale rows: {total_stale}")
    print(f"\n  To delete stale rows, run in Colab:")
    for rid in stale_run_ids:
        print(f"    DELETE FROM `{BQ_TABLE}` WHERE run_id = '{rid}'")

# ==============================================================================
# SUMMARY
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  SUMMARY")
print(f"{'='*W}")

all_need_rerun = []
all_need_cleanup = []
all_ok = []

for exp in experiments:
    runs = exp_runs[exp]
    if len(runs) == 1 and runs[0]['min_freq'] < 2.0:
        all_need_rerun.append(exp)
    elif len(runs) > 1:
        rerun_tickers = needs_rerun.get(exp, [])
        if rerun_tickers:
            all_need_rerun.append(exp)
        else:
            all_need_cleanup.append(exp)
    else:
        all_ok.append(exp)

if all_ok:
    print(f"  OK (no action):     {', '.join(all_ok)}")
if all_need_cleanup:
    print(f"  Cleanup only:       {', '.join(all_need_cleanup)}")
if all_need_rerun:
    print(f"  Need rerun (2/mo):  {', '.join(all_need_rerun)}")
print(f"  Stale rows total:   {total_stale}")
print(f"{'='*W}")
