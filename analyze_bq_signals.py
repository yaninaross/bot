"""
Analyze BigQuery scoring logs for Apr 27-28, 2026.
Pull all GO signals, group by run, identify morning vs afternoon firing.
"""

import sys
try:
    from google.colab import auth
    auth.authenticate_user()
except Exception:
    pass

from google.cloud import bigquery
import pandas as pd

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 200)

client = bigquery.Client()

query = """
SELECT
  CAST(score_date AS STRING) AS score_date,
  experiment,
  ticker,
  seed,
  prob,
  threshold,
  signal,
  entry_minute,
  close,
  bar_timestamp,
  run_timestamp
FROM `ys-time-series.scoring_logs.live_predictions`
WHERE CAST(score_date AS STRING) IN ('2026-04-27', '2026-04-28')
ORDER BY score_date, run_timestamp, experiment, ticker
"""

df = client.query(query).to_dataframe()
print(f"Query returned {len(df)} rows")
if len(df) == 0:
    print("No data! Check table name / score_date format.")
    print("Sampling table to see score_date values:")
    sample = client.query("""
        SELECT DISTINCT score_date, TYPEOF(score_date) as dtype
        FROM `ys-time-series.scoring_logs.live_predictions`
        ORDER BY score_date DESC LIMIT 10
    """).to_dataframe()
    print(sample)
    sys.exit(0)

print(f"Unique signals: {df['signal'].unique()}")
print(f"Signal value counts:\n{df['signal'].value_counts()}")

df['bar_timestamp'] = pd.to_datetime(df['bar_timestamp'], utc=True)
df['run_timestamp'] = pd.to_datetime(df['run_timestamp'], utc=True)

df['bar_hour_pst'] = (df['bar_timestamp'].dt.tz_convert('America/Los_Angeles')
                      .dt.strftime('%H:%M'))
df['run_hour_pst'] = (df['run_timestamp'].dt.tz_convert('America/Los_Angeles')
                      .dt.strftime('%H:%M'))

df['entry_time_pst'] = (df['entry_minute'].apply(
    lambda m: f"{m // 60}:{m % 60:02d}"))

df['bar_min_from_midnight'] = (
    df['bar_timestamp'].dt.tz_convert('America/Los_Angeles').dt.hour * 60 +
    df['bar_timestamp'].dt.tz_convert('America/Los_Angeles').dt.minute
)
df['shift_min'] = df['bar_min_from_midnight'] - df['entry_minute']

# ============================================================
# 1) ALL GO SIGNALS — both days
# ============================================================
go = df[df['signal'] == 'GO'].copy()

print("=" * 140)
print("  ALL GO SIGNALS — Apr 27-28")
print("=" * 140)

for date in ['2026-04-27', '2026-04-28']:
    day_go = go[go['score_date'] == date]
    print(f"\n{'='*140}")
    print(f"  {date}  —  {len(day_go)} total GO rows, {day_go['ticker'].nunique()} unique tickers")
    print(f"{'='*140}")

    # Deduplicate: keep the FIRST GO per ticker per seed (earliest run)
    day_first = day_go.sort_values('run_timestamp').drop_duplicates(
        subset=['ticker', 'seed'], keep='first')

    print(f"\n  First GO per ticker/seed (deduplicated):")
    print(f"  {'Ticker':<6} {'Exp':<14} {'Seed':>6} {'Prob':>6} {'Thr':>6} "
          f"{'Entry(PST)':>10} {'Bar(PST)':>10} {'Run(PST)':>10} {'Shift':>7}")
    print(f"  {'-'*100}")

    for _, r in day_first.sort_values(['experiment', 'ticker']).iterrows():
        shift_str = f"{r['shift_min']:+d}m"
        print(f"  {r['ticker']:<6} {r['experiment']:<14} {r['seed']:>6} {r['prob']:>6.4f} "
              f"{r['threshold']:>6.3f} {r['entry_time_pst']:>10} {r['bar_hour_pst']:>10} "
              f"{r['run_hour_pst']:>10} {shift_str:>7}")

# ============================================================
# 2) MORNING-ONLY GO SIGNALS (bar_timestamp before 7:30 AM PST = 14:30 UTC)
# ============================================================
print(f"\n\n{'='*140}")
print("  MORNING GO SIGNALS ONLY  (bar before 7:30 AM PST)")
print(f"{'='*140}")

morning_go = go[go['bar_min_from_midnight'] <= 450].copy()  # 7:30 AM = 450 min

for date in ['2026-04-27', '2026-04-28']:
    day_go = morning_go[morning_go['score_date'] == date]
    day_first = day_go.sort_values('run_timestamp').drop_duplicates(
        subset=['ticker', 'seed'], keep='first')

    print(f"\n  {date}  —  {len(day_first)} unique ticker/seed GO signals in morning:")
    print(f"  {'Ticker':<6} {'Exp':<14} {'Seed':>6} {'Prob':>6} {'Thr':>6} "
          f"{'Entry(PST)':>10} {'Bar(PST)':>10} {'Run(PST)':>10} {'Shift':>7}")
    print(f"  {'-'*100}")

    for _, r in day_first.sort_values(['experiment', 'ticker']).iterrows():
        shift_str = f"{r['shift_min']:+d}m"
        print(f"  {r['ticker']:<6} {r['experiment']:<14} {r['seed']:>6} {r['prob']:>6.4f} "
              f"{r['threshold']:>6.3f} {r['entry_time_pst']:>10} {r['bar_hour_pst']:>10} "
              f"{r['run_hour_pst']:>10} {shift_str:>7}")

# ============================================================
# 3) GROUP BY RUN — what did each cron execution produce?
# ============================================================
print(f"\n\n{'='*140}")
print("  RUNS THAT PRODUCED GO SIGNALS (grouped by run_timestamp)")
print(f"{'='*140}")

for date in ['2026-04-27', '2026-04-28']:
    day_data = df[df['score_date'] == date]
    day_go = day_data[day_data['signal'] == 'GO']

    runs_with_go = day_go['run_timestamp'].unique()

    print(f"\n  {date}: {len(runs_with_go)} runs produced at least one GO signal\n")

    for run_ts in sorted(runs_with_go):
        run_data = day_data[day_data['run_timestamp'] == run_ts]
        run_go = run_data[run_data['signal'] == 'GO']
        run_no = run_data[run_data['signal'] == 'NO']

        run_pst = run_ts.tz_convert('America/Los_Angeles').strftime('%Y-%m-%d %H:%M:%S')
        go_tickers = ', '.join(
            f"{r['ticker']}({r['seed']})={r['prob']:.3f}"
            for _, r in run_go.sort_values('ticker').iterrows()
        )
        print(f"  Run {run_pst}  |  {len(run_data)} scored  |  "
              f"{len(run_go)} GO: {go_tickers}")

# ============================================================
# 4) DEPLOYED vs OLD configs check
# ============================================================
print(f"\n\n{'='*140}")
print("  CONFIG CHECK: deployed seeds vs what actually scored")
print(f"{'='*140}")

deployed = {
    'RCAT': (10304, 0.525, 365),
    'LUNR': (10311, 0.465, 365),
    'RKLB': (10308, 0.510, 385),
    'SPIR': (10310, 0.310, 360),
    'PLTR': (10311, 0.465, 360),
    'JOBY': (10307, 0.510, 380),
    'ASTS': (10308, 0.510, 390),
    'KTOS': (10312, 0.510, 385),
    'RIOT': (9703,  0.490, 385),
    'IREN': (9715,  0.515, 375),
    'WULF': (9715,  0.515, 360),
    'CIFR': (9714,  0.510, 385),
    'BTBT': (11006, 0.510, 375),
    'APLD': (11204, 0.510, 379),
    'CLSK': (11205, 0.495, 383),
    'HUT':  (11206, 0.510, 370),
}

for date in ['2026-04-27', '2026-04-28']:
    day_data = df[df['score_date'] == date]
    seeds_seen = day_data.groupby('ticker')['seed'].unique()

    wrong_configs = []
    for ticker, seeds in seeds_seen.items():
        if ticker in deployed:
            expected_seed = deployed[ticker][0]
            unexpected = [s for s in seeds if s != expected_seed]
            if unexpected:
                wrong_configs.append((ticker, expected_seed, unexpected))
        elif ticker == 'BBAI':
            wrong_configs.append(('BBAI', 'REMOVED', list(seeds)))

    if wrong_configs:
        print(f"\n  {date} — tickers scored with WRONG/OLD config:")
        for ticker, expected, unexpected in wrong_configs:
            print(f"    {ticker}: expected seed {expected}, also saw {unexpected}")
    else:
        print(f"\n  {date} — all tickers used correct deployed configs")

print(f"\n{'='*140}")
