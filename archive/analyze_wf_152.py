"""
analyze_wf_152.py — Compare EXP-152 WF results against all experiments
======================================================================
Pulls walk-forward optimization results from BQ, computes per-experiment
and per-ticker statistics, and compares 152_XSECTOR against the fleet.

Run in Colab or any environment with BQ access.
"""

import pandas as pd
import numpy as np
from google.cloud import bigquery

BQ_TABLE = "ys-time-series.scoring_logs.wf_optimization_results"

bq_client = bigquery.Client()

# ── Pull all WF results ─────────────────────────────────────────────────────
query = f"""
SELECT
    run_id, experiment, ticker, seed,
    entry_minute, prob_thresh, tp_pct, sl_pct,
    vix_threshold, tlt_crash, spy_ignore, vol_filter_pct,
    opt_avg_pnl, opt_win_rate, opt_freq, opt_count,
    val_a_avg_pnl, val_a_win_rate, val_a_freq, val_a_count, survived_a,
    val_b_avg_pnl, val_b_win_rate, val_b_freq, val_b_count, survived_b,
    run_timestamp
FROM `{BQ_TABLE}`
WHERE experiment IN (
    '97_MICRO', '103_DEFENSE', '104_DEFENSE', '105_DEFENSE',
    '106_DEFENSE', '110_MICRO', '112_MICRO', '113_MICRO',
    '150_ENERGY', '152_XSECTOR'
)
ORDER BY experiment, ticker, seed
"""

print("Pulling WF results from BQ...")
df = bq_client.query(query).to_dataframe()
print(f"Total rows: {len(df)}")

# ── Keep only the latest run_id per experiment ───────────────────────────────
df['run_timestamp'] = pd.to_datetime(df['run_timestamp'])
latest_run = df.sort_values('run_timestamp').groupby('experiment')['run_id'].last().reset_index()
latest_run.columns = ['experiment', 'latest_run_id']
print("\nLatest run per experiment:")
for _, r in latest_run.iterrows():
    subset = df[(df['experiment'] == r['experiment']) &
                (df['run_id'] == r['latest_run_id'])]
    ts = subset['run_timestamp'].max()
    print(f"  {r['experiment']:<15} {ts.strftime('%Y-%m-%d %H:%M')}  run_id={r['latest_run_id'][:8]}...  ({len(subset)} rows)")

df = df.merge(latest_run, on='experiment')
df = df[df['run_id'] == df['latest_run_id']].drop(columns=['latest_run_id'])

# ══════════════════════════════════════════════════════════════════════════════
# TABLE 1: Per-experiment summary
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 120)
print("  TABLE 1: PER-EXPERIMENT SUMMARY (latest run)")
print("=" * 120)

exp_stats = []
for exp, g in df.groupby('experiment'):
    n_rows = len(g)
    n_tickers = g['ticker'].nunique()
    n_seeds = g['seed'].nunique()
    survived_a = g['survived_a'].sum()
    survived_b = g['survived_b'].sum()
    surv_a_pct = survived_a / n_rows * 100 if n_rows else 0
    surv_b_pct = survived_b / n_rows * 100 if n_rows else 0

    # Among survived rows
    sa = g[g['survived_a']]
    sb = g[g['survived_b']]

    opt_pnl_mean = g['opt_avg_pnl'].mean()
    opt_wr_mean = g['opt_win_rate'].mean()
    va_pnl_mean = sa['val_a_avg_pnl'].mean() if len(sa) else 0
    va_wr_mean = sa['val_a_win_rate'].mean() if len(sa) else 0
    vb_pnl_mean = sb['val_b_avg_pnl'].mean() if len(sb) else 0
    vb_wr_mean = sb['val_b_win_rate'].mean() if len(sb) else 0

    exp_stats.append({
        'experiment': exp, 'rows': n_rows,
        'tickers': n_tickers, 'seeds': n_seeds,
        'opt_pnl': opt_pnl_mean, 'opt_wr': opt_wr_mean,
        'surv_a': survived_a, 'surv_a_pct': surv_a_pct,
        'va_pnl': va_pnl_mean, 'va_wr': va_wr_mean,
        'surv_b': survived_b, 'surv_b_pct': surv_b_pct,
        'vb_pnl': vb_pnl_mean, 'vb_wr': vb_wr_mean,
    })

edf = pd.DataFrame(exp_stats).sort_values('opt_pnl', ascending=False)

print(f"  {'Experiment':<15} {'Rows':>5} {'Tick':>4} {'Seed':>4}  "
      f"{'OptPnL':>7} {'OptWR':>6}  "
      f"{'SurvA':>5} {'%':>5} {'VA_PnL':>7} {'VA_WR':>6}  "
      f"{'SurvB':>5} {'%':>5} {'VB_PnL':>7} {'VB_WR':>6}")
print("-" * 120)
for _, r in edf.iterrows():
    marker = " <<<" if r['experiment'] == '152_XSECTOR' else ""
    print(f"  {r['experiment']:<15} {r['rows']:>5} {r['tickers']:>4} {r['seeds']:>4}  "
          f"{r['opt_pnl']:>7.2f} {r['opt_wr']:>5.1f}%  "
          f"{r['surv_a']:>5.0f} {r['surv_a_pct']:>4.1f}% {r['va_pnl']:>7.2f} {r['va_wr']:>5.1f}%  "
          f"{r['surv_b']:>5.0f} {r['surv_b_pct']:>4.1f}% {r['vb_pnl']:>7.2f} {r['vb_wr']:>5.1f}%{marker}")

# ══════════════════════════════════════════════════════════════════════════════
# TABLE 2: EXP-152 per-ticker breakdown
# ══════════════════════════════════════════════════════════════════════════════
exp152 = df[df['experiment'] == '152_XSECTOR']
print("\n" + "=" * 140)
print("  TABLE 2: EXP-152 XSECTOR — PER-TICKER BREAKDOWN")
print("=" * 140)

if exp152.empty:
    print("  No 152_XSECTOR data found in BQ!")
else:
    print(f"  {'Ticker':<6} {'Seeds':>5} {'SurvA':>5} {'SurvB':>5}  "
          f"{'OptPnL':>7} {'OptWR':>6} {'OptFreq':>7} {'OptN':>4}  "
          f"{'VA_PnL':>7} {'VA_WR':>6} {'VA_N':>4}  "
          f"{'VB_PnL':>7} {'VB_WR':>6} {'VB_N':>4}  "
          f"{'BestSeed':>8} {'BestPnL':>7}")
    print("-" * 140)

    for ticker, tg in exp152.groupby('ticker'):
        n_seeds = len(tg)
        sa = tg[tg['survived_a']]
        sb = tg[tg['survived_b']]
        opt_pnl = tg['opt_avg_pnl'].mean()
        opt_wr = tg['opt_win_rate'].mean()
        opt_freq = tg['opt_freq'].mean()
        opt_n = tg['opt_count'].mean()
        va_pnl = sa['val_a_avg_pnl'].mean() if len(sa) else 0
        va_wr = sa['val_a_win_rate'].mean() if len(sa) else 0
        va_n = sa['val_a_count'].mean() if len(sa) else 0
        vb_pnl = sb['val_b_avg_pnl'].mean() if len(sb) else 0
        vb_wr = sb['val_b_win_rate'].mean() if len(sb) else 0
        vb_n = sb['val_b_count'].mean() if len(sb) else 0

        best = tg.loc[tg['opt_avg_pnl'].idxmax()]
        print(f"  {ticker:<6} {n_seeds:>5} {len(sa):>5} {len(sb):>5}  "
              f"{opt_pnl:>7.2f} {opt_wr:>5.1f}% {opt_freq:>7.1f} {opt_n:>4.0f}  "
              f"{va_pnl:>7.2f} {va_wr:>5.1f}% {va_n:>4.0f}  "
              f"{vb_pnl:>7.2f} {vb_wr:>5.1f}% {vb_n:>4.0f}  "
              f"S{best['seed']:>5} {best['opt_avg_pnl']:>7.2f}")

# ══════════════════════════════════════════════════════════════════════════════
# TABLE 3: Cross-experiment ticker comparison
# ══════════════════════════════════════════════════════════════════════════════
# Show tickers that appear in 152 AND at least one other experiment
print("\n" + "=" * 130)
print("  TABLE 3: CROSS-EXPERIMENT TICKER COMPARISON (152 tickers in other experiments)")
print("=" * 130)

tickers_152 = set(exp152['ticker'].unique()) if not exp152.empty else set()
cross = df[df['ticker'].isin(tickers_152)]

if cross.empty:
    print("  No cross-experiment data found.")
else:
    print(f"  {'Ticker':<6} {'Experiment':<15} {'Seeds':>5} {'SurvA':>5} {'SurvB':>5}  "
          f"{'OptPnL':>7} {'OptWR':>6}  "
          f"{'VA_PnL':>7} {'VA_WR':>6}  "
          f"{'VB_PnL':>7} {'VB_WR':>6}  {'Note'}")
    print("-" * 130)

    for ticker in sorted(tickers_152):
        tg = cross[cross['ticker'] == ticker]
        for exp, eg in tg.groupby('experiment'):
            sa = eg[eg['survived_a']]
            sb = eg[eg['survived_b']]
            opt_pnl = eg['opt_avg_pnl'].mean()
            opt_wr = eg['opt_win_rate'].mean()
            va_pnl = sa['val_a_avg_pnl'].mean() if len(sa) else 0
            va_wr = sa['val_a_win_rate'].mean() if len(sa) else 0
            vb_pnl = sb['val_b_avg_pnl'].mean() if len(sb) else 0
            vb_wr = sb['val_b_win_rate'].mean() if len(sb) else 0
            note = " <<< DEPLOYED" if exp == '152_XSECTOR' else ""
            print(f"  {ticker:<6} {exp:<15} {len(eg):>5} {len(sa):>5} {len(sb):>5}  "
                  f"{opt_pnl:>7.2f} {opt_wr:>5.1f}%  "
                  f"{va_pnl:>7.2f} {va_wr:>5.1f}%  "
                  f"{vb_pnl:>7.2f} {vb_wr:>5.1f}%{note}")
        print()

# ══════════════════════════════════════════════════════════════════════════════
# TABLE 4: Survival rate comparison
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 90)
print("  TABLE 4: SURVIVAL RATE + VALIDATION DECAY")
print("=" * 90)
print(f"  {'Experiment':<15} {'Rows':>5}  "
      f"{'Opt→ValA':>10}  {'ValA→ValB':>10}  {'Opt→ValB':>10}  "
      f"{'PnL_Decay_AB':>12}")
print("-" * 90)

for _, r in edf.iterrows():
    exp = r['experiment']
    eg = df[df['experiment'] == exp]
    sa = eg[eg['survived_a']]
    sb = eg[eg['survived_b']]

    opt_to_a = r['surv_a_pct']
    a_to_b = (len(sb) / len(sa) * 100) if len(sa) > 0 else 0
    opt_to_b = r['surv_b_pct']

    pnl_decay = 0
    if len(sa) > 0 and len(sb) > 0:
        pnl_decay = (sb['val_b_avg_pnl'].mean() - sa['val_a_avg_pnl'].mean())

    marker = " <<<" if exp == '152_XSECTOR' else ""
    print(f"  {exp:<15} {r['rows']:>5}  "
          f"{opt_to_a:>9.1f}%  {a_to_b:>9.1f}%  {opt_to_b:>9.1f}%  "
          f"{pnl_decay:>+11.2f}%{marker}")

# ══════════════════════════════════════════════════════════════════════════════
# TABLE 5: EXP-152 vs EXP-150 head-to-head
# ══════════════════════════════════════════════════════════════════════════════
exp150 = df[df['experiment'] == '150_ENERGY']
print("\n" + "=" * 90)
print("  TABLE 5: HEAD-TO-HEAD — 152_XSECTOR vs 150_ENERGY")
print("=" * 90)

for label, edf_sub in [('150_ENERGY', exp150), ('152_XSECTOR', exp152)]:
    if edf_sub.empty:
        print(f"  {label}: no data")
        continue
    sa = edf_sub[edf_sub['survived_a']]
    sb = edf_sub[edf_sub['survived_b']]
    print(f"\n  {label}:")
    print(f"    Total rows:     {len(edf_sub)}")
    print(f"    Unique tickers: {edf_sub['ticker'].nunique()}")
    print(f"    Seeds per tick: {edf_sub.groupby('ticker')['seed'].count().mean():.0f}")
    print(f"    Opt avg PnL:    {edf_sub['opt_avg_pnl'].mean():.2f}%")
    print(f"    Opt avg WR:     {edf_sub['opt_win_rate'].mean():.1f}%")
    print(f"    Opt avg freq:   {edf_sub['opt_freq'].mean():.1f} trades/mo")
    print(f"    Survived A:     {len(sa)}/{len(edf_sub)} ({len(sa)/len(edf_sub)*100:.1f}%)")
    if len(sa):
        print(f"    ValA avg PnL:   {sa['val_a_avg_pnl'].mean():.2f}%")
        print(f"    ValA avg WR:    {sa['val_a_win_rate'].mean():.1f}%")
    print(f"    Survived B:     {len(sb)}/{len(edf_sub)} ({len(sb)/len(edf_sub)*100:.1f}%)")
    if len(sb):
        print(f"    ValB avg PnL:   {sb['val_b_avg_pnl'].mean():.2f}%")
        print(f"    ValB avg WR:    {sb['val_b_win_rate'].mean():.1f}%")

print("\n" + "=" * 90)
print("  DONE — paste output back for analysis")
print("=" * 90)
