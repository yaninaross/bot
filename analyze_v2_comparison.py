"""
analyze_v2_comparison.py — Compare v2_bar_is_real vs old WF results
====================================================================
Pulls old (model_version IS NULL) and new (model_version = 'v2_bar_is_real')
results from BQ for specified experiments. Shows per-ticker/seed deltas
and aggregate improvement metrics.

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
EXPERIMENTS = ['105_DEFENSE', '106_DEFENSE', '152_XSECTOR', '103_DEFENSE']
MIN_TRADES = 3
W = 140

# ==============================================================================
# LOAD DATA
# ==============================================================================
client = bigquery.Client()
query = f"""
SELECT *
FROM `{BQ_TABLE}`
WHERE experiment IN ({', '.join(f"'{e}'" for e in EXPERIMENTS)})
ORDER BY experiment, ticker, seed
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

# Keep latest run per (experiment, ticker, model_version)
_latest = (df.groupby(['experiment', 'ticker', 'model_version', 'run_id'])['run_timestamp']
           .max().reset_index()
           .sort_values('run_timestamp')
           .groupby(['experiment', 'ticker', 'model_version']).last()
           .reset_index()[['experiment', 'ticker', 'model_version', 'run_id']])
df = df.merge(_latest, on=['experiment', 'ticker', 'model_version', 'run_id'], how='inner')

old = df[df['model_version'] == 'v1_original'].copy()
new = df[df['model_version'] == 'v2_bar_is_real'].copy()

# ==============================================================================
# HEADER
# ==============================================================================
print(f"\n{'='*W}")
print(f"  V2_BAR_IS_REAL vs ORIGINAL — WF COMPARISON")
print(f"  Experiments: {', '.join(EXPERIMENTS)}")
print(f"  Old rows: {len(old):,}   New rows: {len(new):,}")
print(f"{'='*W}")

# Check which experiments have new data
for exp in EXPERIMENTS:
    n_old = len(old[old['experiment'] == exp])
    n_new = len(new[new['experiment'] == exp])
    status = "READY" if n_new > 0 else "NO NEW DATA"
    print(f"  {exp:<20}  old: {n_old:>4} rows   new: {n_new:>4} rows   [{status}]")

# ==============================================================================
# PER-EXPERIMENT ANALYSIS
# ==============================================================================
for exp in EXPERIMENTS:
    exp_old = old[old['experiment'] == exp]
    exp_new = new[new['experiment'] == exp]

    if exp_new.empty:
        print(f"\n{'─'*W}")
        print(f"  {exp}: no v2 results yet — skipping")
        continue

    merged = exp_new.merge(
        exp_old, on=['ticker', 'seed'], suffixes=('_new', '_old'), how='inner'
    )

    if merged.empty:
        print(f"\n{'─'*W}")
        print(f"  {exp}: no matching ticker/seed pairs — skipping")
        # Show new-only results
        if not exp_new.empty:
            print(f"  New results (no old baseline):")
            for _, r in exp_new.sort_values(['ticker', 'seed']).iterrows():
                print(f"    {r['ticker']:<7} S{r['seed']}  "
                      f"OPT={r['opt_avg_pnl']:.2%} WR={r['opt_win_rate']:.1%} N={int(r['opt_count'])}  "
                      f"VAL-A={r['val_a_avg_pnl']:.2%} N={int(r['val_a_count'])}")
        continue

    # ── Per-seed comparison table ─────────────────────────────────────
    print(f"\n\n{'='*W}")
    print(f"  {exp} — PER-SEED COMPARISON (old vs new)")
    print(f"  Matched pairs: {len(merged)}")
    print(f"{'='*W}")
    print(f"  {'Ticker':<7} {'Seed':>5} "
          f"{'|':>2} {'OLD OPT':>8} {'NEW OPT':>8} {'Δ OPT':>7} "
          f"{'|':>2} {'OLD N':>5} {'NEW N':>5} {'Δ N':>4} "
          f"{'|':>2} {'OLD VA':>8} {'NEW VA':>8} {'Δ VA':>7} "
          f"{'|':>2} {'OLD VAN':>5} {'NEW VAN':>5} "
          f"{'|':>2} {'SurvA':>6}")
    print(f"  {'─'*W}")

    for _, r in merged.sort_values(['ticker', 'seed']).iterrows():
        d_opt = r['opt_avg_pnl_new'] - r['opt_avg_pnl_old']
        d_n = int(r['opt_count_new']) - int(r['opt_count_old'])
        d_va = r['val_a_avg_pnl_new'] - r['val_a_avg_pnl_old']
        surv_old = 'P' if r['survived_a_old'] else 'F'
        surv_new = 'P' if r['survived_a_new'] else 'F'
        surv_str = f"{surv_old}→{surv_new}"

        d_opt_str = f"{d_opt:>+.2%}" if d_opt != 0 else "  0.00%"
        d_va_str = f"{d_va:>+.2%}" if d_va != 0 else "  0.00%"

        print(f"  {r['ticker']:<7} {int(r['seed']):>5} "
              f"{'|':>2} {r['opt_avg_pnl_old']:>7.2%} {r['opt_avg_pnl_new']:>7.2%} {d_opt_str:>7} "
              f"{'|':>2} {int(r['opt_count_old']):>5} {int(r['opt_count_new']):>5} {d_n:>+4d} "
              f"{'|':>2} {r['val_a_avg_pnl_old']:>7.2%} {r['val_a_avg_pnl_new']:>7.2%} {d_va_str:>7} "
              f"{'|':>2} {int(r['val_a_count_old']):>5} {int(r['val_a_count_new']):>5} "
              f"{'|':>2} {surv_str:>6}")

    # ── Aggregate stats ───────────────────────────────────────────────
    print(f"  {'─'*W}")

    avg_old_opt = merged['opt_avg_pnl_old'].mean()
    avg_new_opt = merged['opt_avg_pnl_new'].mean()
    avg_old_va = merged['val_a_avg_pnl_old'].mean()
    avg_new_va = merged['val_a_avg_pnl_new'].mean()
    avg_old_n = merged['opt_count_old'].mean()
    avg_new_n = merged['opt_count_new'].mean()

    surv_old_a = merged['survived_a_old'].sum()
    surv_new_a = merged['survived_a_new'].sum()
    surv_old_b = merged['survived_b_old'].sum()
    surv_new_b = merged['survived_b_new'].sum()

    print(f"  AVERAGES:")
    print(f"    OPT PnL:     old={avg_old_opt:.2%}  new={avg_new_opt:.2%}  Δ={avg_new_opt - avg_old_opt:>+.2%}")
    print(f"    OPT N:       old={avg_old_n:.1f}     new={avg_new_n:.1f}     Δ={avg_new_n - avg_old_n:>+.1f}")
    print(f"    VAL-A PnL:   old={avg_old_va:.2%}  new={avg_new_va:.2%}  Δ={avg_new_va - avg_old_va:>+.2%}")
    print(f"    Survived A:  old={int(surv_old_a)}/{len(merged)}  new={int(surv_new_a)}/{len(merged)}")
    print(f"    Survived B:  old={int(surv_old_b)}/{len(merged)}  new={int(surv_new_b)}/{len(merged)}")

    # PnL decay comparison
    if avg_old_opt > 0 and avg_new_opt > 0:
        decay_old = (1 - avg_old_va / avg_old_opt) * 100
        decay_new = (1 - avg_new_va / avg_new_opt) * 100
        print(f"    PnL decay:   old={decay_old:.0f}%  new={decay_new:.0f}%  "
              f"({'better' if decay_new < decay_old else 'worse'})")

    # ── Per-ticker summary (best seed) ────────────────────────────────
    tickers_in_exp = sorted(merged['ticker'].unique())
    print(f"\n  PER-TICKER BEST SEED (by VAL-A PnL):")
    print(f"  {'Ticker':<7} "
          f"{'OLD best':>9} {'seed':>5} {'N':>4} "
          f"{'NEW best':>9} {'seed':>5} {'N':>4} "
          f"{'Δ PnL':>8} {'Signal':>12}")
    print(f"  {'─'*70}")

    for t in tickers_in_exp:
        t_old = exp_old[exp_old['ticker'] == t]
        t_new = exp_new[exp_new['ticker'] == t]

        if t_old.empty or t_new.empty:
            continue

        best_old = t_old.loc[t_old['val_a_avg_pnl'].idxmax()]
        best_new = t_new.loc[t_new['val_a_avg_pnl'].idxmax()]

        delta = best_new['val_a_avg_pnl'] - best_old['val_a_avg_pnl']
        if delta > 0.005:
            signal = "IMPROVED"
        elif delta < -0.005:
            signal = "DEGRADED"
        else:
            signal = "FLAT"

        print(f"  {t:<7} "
              f"{best_old['val_a_avg_pnl']:>8.2%} {int(best_old['seed']):>5} {int(best_old['val_a_count']):>4} "
              f"{best_new['val_a_avg_pnl']:>8.2%} {int(best_new['seed']):>5} {int(best_new['val_a_count']):>4} "
              f"{delta:>+7.2%} {signal:>12}")

    # ── Trade count impact (V=0 removal) ──────────────────────────────
    print(f"\n  TRADE COUNT IMPACT (V=0 removal):")
    for t in tickers_in_exp:
        t_merged = merged[merged['ticker'] == t]
        avg_old_n_t = t_merged['opt_count_old'].mean()
        avg_new_n_t = t_merged['opt_count_new'].mean()
        pct_drop = ((avg_new_n_t - avg_old_n_t) / avg_old_n_t * 100) if avg_old_n_t > 0 else 0
        print(f"    {t:<7}  old avg N={avg_old_n_t:.1f}  new avg N={avg_new_n_t:.1f}  "
              f"Δ={pct_drop:>+.0f}% {'← V=0 contamination removed' if pct_drop < -5 else ''}")

# ==============================================================================
# OVERALL SUMMARY
# ==============================================================================
if not new.empty and not old.empty:
    all_merged = new.merge(old, on=['ticker', 'seed', 'experiment'],
                           suffixes=('_new', '_old'), how='inner')
    if not all_merged.empty:
        print(f"\n\n{'='*W}")
        print(f"  OVERALL SUMMARY — ALL EXPERIMENTS")
        print(f"{'='*W}")
        print(f"  Total matched pairs: {len(all_merged)}")
        print(f"  OPT PnL:    old={all_merged['opt_avg_pnl_old'].mean():.2%}  "
              f"new={all_merged['opt_avg_pnl_new'].mean():.2%}  "
              f"Δ={all_merged['opt_avg_pnl_new'].mean() - all_merged['opt_avg_pnl_old'].mean():>+.2%}")
        print(f"  VAL-A PnL:  old={all_merged['val_a_avg_pnl_old'].mean():.2%}  "
              f"new={all_merged['val_a_avg_pnl_new'].mean():.2%}  "
              f"Δ={all_merged['val_a_avg_pnl_new'].mean() - all_merged['val_a_avg_pnl_old'].mean():>+.2%}")
        print(f"  Survived A: old={int(all_merged['survived_a_old'].sum())}/{len(all_merged)}  "
              f"new={int(all_merged['survived_a_new'].sum())}/{len(all_merged)}")

        improved = (all_merged['val_a_avg_pnl_new'] > all_merged['val_a_avg_pnl_old']).sum()
        degraded = (all_merged['val_a_avg_pnl_new'] < all_merged['val_a_avg_pnl_old']).sum()
        flat = len(all_merged) - improved - degraded
        print(f"  Seed-level: {improved} improved, {degraded} degraded, {flat} flat")

        flipped_pass = ((~all_merged['survived_a_old']) & all_merged['survived_a_new']).sum()
        flipped_fail = (all_merged['survived_a_old'] & (~all_merged['survived_a_new'])).sum()
        print(f"  Survival flips: {int(flipped_pass)} FAIL→PASS, {int(flipped_fail)} PASS→FAIL")

print(f"\n{'='*W}")
