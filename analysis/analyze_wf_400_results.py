"""
analyze_wf_400_results.py — Viability Dashboard for Exp 400 SHORT/Dump Model
=============================================================================
Queries BQ for 'robust_wf_v12_short' optimization results and answers:
  1. Does the dump model produce positive PnL at all?
  2. How many tickers survive OPT → VAL → TEST?
  3. Which modes perform best (static, atr_sl, grace, limit_entry)?
  4. What's the degradation profile (overfitting risk)?
  5. Which tickers are deployment candidates?

Run anytime — works with partial results (mid-optimization).
Colab-ready (no if __name__).
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

try:
    from google.colab import auth
    auth.authenticate_user()
except Exception:
    pass

import matplotlib
try:
    from google.colab import output as _
except ImportError:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from google.cloud import bigquery

bq_client = bigquery.Client()

# ==============================================================================
# CONFIG
# ==============================================================================
BQ_TABLE = "ys-time-series.scoring_logs.wf_optimization_results"
MODEL_VERSION = 'robust_wf_v12_short'
DEPLOY_PNL_THRESHOLD = 0.018   # 1.8% avg PnL — same bar as long models
MIN_TEST_TRADES = 3
W = 120


# ==============================================================================
# STEP 1: Load all results from BQ
# ==============================================================================
print("=" * W)
print(f"  EXP 400 SHORT/DUMP — OPTIMIZATION RESULTS DASHBOARD")
print(f"  Source: BQ model_version = '{MODEL_VERSION}'")
print("=" * W)

q = f"""
SELECT
    ticker, experiment, seed, tpsl_mode, entry_time, entry_minute,
    prob_thresh, tp_pct, sl_pct, tp_mult, sl_mult,
    trail_start_mult, trail_end_mult,
    grace_minutes, dip_pct, max_wait_minutes,
    vol_filter_pct, pre_range_filter_pct,
    opt_avg_pnl, opt_win_rate, opt_freq, opt_count,
    val_a_avg_pnl, val_a_win_rate, val_a_freq, val_a_count, survived_a,
    val_b_avg_pnl, val_b_win_rate, val_b_freq, val_b_count, survived_b,
    is_ensemble, n_seeds_avg,
    run_timestamp
FROM `{BQ_TABLE}`
WHERE model_version = '{MODEL_VERSION}'
ORDER BY ticker, tpsl_mode, seed
"""
df = bq_client.query(q).to_dataframe()
print(f"\n  Loaded {len(df)} rows from BQ")

if len(df) == 0:
    print("\n  ⚠ NO RESULTS YET — optimizer hasn't written to BQ.")
    print("  Re-run this script after the optimizer processes at least a few tickers.")
    raise SystemExit

per_seed = df[df['is_ensemble'] != True].copy()
ensemble = df[df['is_ensemble'] == True].copy()
print(f"  Per-seed rows: {len(per_seed)}  |  Ensemble rows: {len(ensemble)}")
print(f"  Tickers: {per_seed['ticker'].nunique()}  |  Modes: {sorted(per_seed['tpsl_mode'].dropna().unique())}")
print(f"  Latest result: {df['run_timestamp'].max()}")


# ==============================================================================
# STEP 2: OVERALL VIABILITY — does it produce positive PnL?
# ==============================================================================
print()
print("=" * W)
print(f"  A. OVERALL VIABILITY — IS THE SHORT MODEL PROFITABLE?")
print("=" * W)

modes = sorted(per_seed['tpsl_mode'].dropna().unique())

print(f"\n  {'Mode':<15} │ {'Seeds':>5} {'Tickers':>7} │ "
      f"{'OPT avg':>8} {'VAL avg':>8} {'TEST avg':>9} │ "
      f"{'VAL+':>4} {'TEST+':>5} │ {'Degrad':>7}")
print(f"  {'─'*15} │ {'─'*5} {'─'*7} │ {'─'*8} {'─'*8} {'─'*9} │ {'─'*4} {'─'*5} │ {'─'*7}")

mode_stats = {}
for mode in modes:
    md = per_seed[per_seed['tpsl_mode'] == mode]
    n_seeds = len(md)
    n_tickers = md['ticker'].nunique()
    opt_avg = md['opt_avg_pnl'].mean()
    val_avg = md['val_a_avg_pnl'].mean()
    test_avg = md['val_b_avg_pnl'].mean()
    val_pos = (md['val_a_avg_pnl'] > 0).sum()
    test_pos = (md['val_b_avg_pnl'] > 0).sum()
    degrad = 1 - test_avg / opt_avg if opt_avg > 0 else 0

    mode_stats[mode] = {
        'n_seeds': n_seeds, 'n_tickers': n_tickers,
        'opt': opt_avg, 'val': val_avg, 'test': test_avg,
        'val_pos': val_pos, 'test_pos': test_pos,
        'degrad': degrad,
    }
    print(f"  {mode:<15} │ {n_seeds:>5} {n_tickers:>7} │ "
          f"{opt_avg:>+7.2%} {val_avg:>+7.2%} {test_avg:>+8.2%} │ "
          f"{val_pos:>4} {test_pos:>5} │ {degrad:>+6.0%}")

# Grand totals
print(f"  {'─'*15} │ {'─'*5} {'─'*7} │ {'─'*8} {'─'*8} {'─'*9} │ {'─'*4} {'─'*5} │ {'─'*7}")
opt_all = per_seed['opt_avg_pnl'].mean()
val_all = per_seed['val_a_avg_pnl'].mean()
test_all = per_seed['val_b_avg_pnl'].mean()
degrad_all = 1 - test_all / opt_all if opt_all > 0 else 0
print(f"  {'ALL':<15} │ {len(per_seed):>5} {per_seed['ticker'].nunique():>7} │ "
      f"{opt_all:>+7.2%} {val_all:>+7.2%} {test_all:>+8.2%} │ "
      f"{(per_seed['val_a_avg_pnl'] > 0).sum():>4} {(per_seed['val_b_avg_pnl'] > 0).sum():>5} │ "
      f"{degrad_all:>+6.0%}")

verdict = "VIABLE ✓" if test_all > 0 else "NOT VIABLE ✗"
print(f"\n  ★ Grand mean TEST PnL = {test_all:>+.3%} → {verdict}")
if test_all > DEPLOY_PNL_THRESHOLD:
    print(f"    Exceeds deploy threshold ({DEPLOY_PNL_THRESHOLD:.1%})")
elif test_all > 0:
    print(f"    Positive but below deploy threshold ({DEPLOY_PNL_THRESHOLD:.1%}) — needs ticker selection")


# ==============================================================================
# STEP 3: PER-TICKER BREAKDOWN — best config per ticker
# ==============================================================================
print()
print("=" * W)
print(f"  B. PER-TICKER BEST CONFIG — OPT / VAL / TEST")
print("=" * W)

best_rows = []
for (ticker, mode), grp in per_seed.groupby(['ticker', 'tpsl_mode']):
    best_seed = grp.loc[grp['opt_avg_pnl'].idxmax()]
    best_rows.append(best_seed)

if not best_rows:
    print("  No per-seed results to analyze.")
else:
    best_df = pd.DataFrame(best_rows)

    # Per-ticker: pick the mode with best TEST PnL
    ticker_best = []
    for ticker, grp in best_df.groupby('ticker'):
        row = grp.loc[grp['val_b_avg_pnl'].idxmax()]
        ticker_best.append(row)
    ticker_df = pd.DataFrame(ticker_best).sort_values('val_b_avg_pnl', ascending=False)

    print(f"\n  {'Ticker':<7} {'Mode':<12} {'Seed':>4} {'Prob':>5} {'TP':>5} {'SL':>5} │ "
          f"{'OPT':>7} {'VAL':>7} {'TEST':>8} {'T_WR':>5} {'T_N':>4} │ "
          f"{'Degrad':>7} {'Deploy':>6}")
    print(f"  {'─'*7} {'─'*12} {'─'*4} {'─'*5} {'─'*5} {'─'*5} │ "
          f"{'─'*7} {'─'*7} {'─'*8} {'─'*5} {'─'*4} │ "
          f"{'─'*7} {'─'*6}")

    n_deploy = 0
    for _, r in ticker_df.iterrows():
        mode = r['tpsl_mode'] or 'static'
        opt_pnl = r['opt_avg_pnl']
        val_pnl = r['val_a_avg_pnl']
        test_pnl = r['val_b_avg_pnl']
        test_wr = r['val_b_win_rate']
        test_n = int(r['val_b_count']) if pd.notna(r['val_b_count']) else 0
        degrad = 1 - test_pnl / opt_pnl if opt_pnl > 0 else 0

        tp_str = f"{r['tp_pct']:.0%}" if r['tp_pct'] > 0 else f"{r.get('tp_mult', 0):.1f}x"
        sl_str = f"{r['sl_pct']:.0%}" if r['sl_pct'] > 0 else f"{r.get('sl_mult', 0):.1f}x"

        deployable = test_pnl >= DEPLOY_PNL_THRESHOLD and test_n >= MIN_TEST_TRADES
        deploy_tag = "YES" if deployable else "—"
        if deployable:
            n_deploy += 1

        print(f"  {r['ticker']:<7} {mode:<12} {int(r['seed']):>4} {r['prob_thresh']:>5.2f} "
              f"{tp_str:>5} {sl_str:>5} │ "
              f"{opt_pnl:>+6.2%} {val_pnl:>+6.2%} {test_pnl:>+7.2%} {test_wr:>4.0%} {test_n:>4} │ "
              f"{degrad:>+6.0%} {deploy_tag:>6}")

    print(f"  {'─'*90}")
    n_pos_test = (ticker_df['val_b_avg_pnl'] > 0).sum()
    print(f"  {n_pos_test}/{len(ticker_df)} tickers positive TEST  |  "
          f"{n_deploy}/{len(ticker_df)} deployable (≥{DEPLOY_PNL_THRESHOLD:.1%} + ≥{MIN_TEST_TRADES} trades)")


# ==============================================================================
# STEP 4: MODE COMPARISON — which TPSL mode works best for shorts?
# ==============================================================================
print()
print("=" * W)
print(f"  C. MODE COMPARISON — BEST MODE FOR SHORT TRADES?")
print("=" * W)

if len(modes) > 1:
    print(f"\n  Per-ticker winner counts (by TEST PnL):")
    mode_wins = {}
    for ticker, grp in best_df.groupby('ticker'):
        winner = grp.loc[grp['val_b_avg_pnl'].idxmax(), 'tpsl_mode']
        mode_wins[winner] = mode_wins.get(winner, 0) + 1
    for mode in modes:
        ct = mode_wins.get(mode, 0)
        pct = ct / len(best_df['ticker'].unique()) * 100
        bar = "█" * int(pct / 2)
        print(f"    {mode:<15} {ct:>3} wins ({pct:>4.0f}%) {bar}")

    print(f"\n  Mode-level TEST stats (across all per-seed results):")
    print(f"  {'Mode':<15} │ {'Med TEST':>9} {'Mean TEST':>10} {'%Positive':>10} │ "
          f"{'Med OPT':>8} {'Degrad':>7}")
    print(f"  {'─'*15} │ {'─'*9} {'─'*10} {'─'*10} │ {'─'*8} {'─'*7}")
    for mode in modes:
        md = per_seed[per_seed['tpsl_mode'] == mode]
        med_test = md['val_b_avg_pnl'].median()
        mean_test = md['val_b_avg_pnl'].mean()
        pct_pos = (md['val_b_avg_pnl'] > 0).mean() * 100
        med_opt = md['opt_avg_pnl'].median()
        degrad = 1 - med_test / med_opt if med_opt > 0 else 0
        print(f"  {mode:<15} │ {med_test:>+8.2%} {mean_test:>+9.2%} {pct_pos:>9.0f}% │ "
              f"{med_opt:>+7.2%} {degrad:>+6.0%}")
else:
    print(f"\n  Only one mode ({modes[0]}) — run more modes for comparison.")


# ==============================================================================
# STEP 5: SEED STABILITY — how consistent are results across seeds?
# ==============================================================================
print()
print("=" * W)
print(f"  D. SEED STABILITY — CROSS-SEED CONSISTENCY")
print("=" * W)

print(f"\n  {'Ticker':<7} {'Mode':<12} │ {'#Seeds':>6} {'#VAL+':>5} {'#TEST+':>6} │ "
      f"{'Agree%':>6} {'Spread':>7} │ {'Stable':>6}")
print(f"  {'─'*7} {'─'*12} │ {'─'*6} {'─'*5} {'─'*6} │ {'─'*6} {'─'*7} │ {'─'*6}")

stability_rows = []
for (ticker, mode), grp in per_seed.groupby(['ticker', 'tpsl_mode']):
    n_seeds = len(grp)
    if n_seeds < 3:
        continue
    n_val_pos = (grp['val_a_avg_pnl'] > 0).sum()
    n_test_pos = (grp['val_b_avg_pnl'] > 0).sum()
    agree = n_test_pos / n_seeds
    spread = grp['val_b_avg_pnl'].std()
    stable = agree >= 0.5 and spread < 0.05
    stability_rows.append({
        'ticker': ticker, 'mode': mode,
        'n_seeds': n_seeds, 'n_val_pos': n_val_pos, 'n_test_pos': n_test_pos,
        'agree': agree, 'spread': spread, 'stable': stable,
    })
    tag = "YES" if stable else "—"
    print(f"  {ticker:<7} {mode:<12} │ {n_seeds:>6} {n_val_pos:>5} {n_test_pos:>6} │ "
          f"{agree:>5.0%} {spread:>6.2%} │ {tag:>6}")

if stability_rows:
    stab_df = pd.DataFrame(stability_rows)
    n_stable = stab_df['stable'].sum()
    print(f"\n  {n_stable}/{len(stab_df)} ticker-mode combos are stable (≥50% seeds positive TEST, spread <5%)")


# ==============================================================================
# STEP 6: ENSEMBLE RESULTS
# ==============================================================================
if len(ensemble) > 0:
    print()
    print("=" * W)
    print(f"  E. ENSEMBLE RESULTS (averaged probabilities across all seeds)")
    print("=" * W)

    print(f"\n  {'Ticker':<7} {'Mode':<12} {'#Seeds':>6} {'Prob':>5} {'TP':>5} {'SL':>5} │ "
          f"{'OPT':>7} │ {'TEST':>8} {'T_WR':>5} {'T_N':>4} │ {'Deploy':>6}")
    print(f"  {'─'*7} {'─'*12} {'─'*6} {'─'*5} {'─'*5} {'─'*5} │ "
          f"{'─'*7} │ {'─'*8} {'─'*5} {'─'*4} │ {'─'*6}")

    ens_sorted = ensemble.sort_values('val_b_avg_pnl', ascending=False)
    n_ens_deploy = 0
    for _, r in ens_sorted.iterrows():
        mode = r['tpsl_mode'] or 'static'
        opt_pnl = r['opt_avg_pnl']
        test_pnl = r['val_b_avg_pnl']
        test_wr = r['val_b_win_rate']
        test_n = int(r['val_b_count']) if pd.notna(r['val_b_count']) else 0
        n_seeds = int(r['n_seeds_avg']) if pd.notna(r['n_seeds_avg']) else 0

        tp_str = f"{r['tp_pct']:.0%}" if r['tp_pct'] > 0 else f"{r.get('tp_mult', 0):.1f}x"
        sl_str = f"{r['sl_pct']:.0%}" if r['sl_pct'] > 0 else f"{r.get('sl_mult', 0):.1f}x"

        deployable = test_pnl >= DEPLOY_PNL_THRESHOLD and test_n >= MIN_TEST_TRADES
        deploy_tag = "YES" if deployable else "—"
        if deployable:
            n_ens_deploy += 1

        print(f"  {r['ticker']:<7} {mode:<12} {n_seeds:>6} {r['prob_thresh']:>5.2f} "
              f"{tp_str:>5} {sl_str:>5} │ "
              f"{opt_pnl:>+6.2%} │ {test_pnl:>+7.2%} {test_wr:>4.0%} {test_n:>4} │ {deploy_tag:>6}")

    n_ens_pos = (ens_sorted['val_b_avg_pnl'] > 0).sum()
    print(f"  {'─'*80}")
    print(f"  {n_ens_pos}/{len(ens_sorted)} positive TEST  |  "
          f"{n_ens_deploy}/{len(ens_sorted)} deployable")

    if n_ens_pos > 0:
        ens_pos = ens_sorted[ens_sorted['val_b_avg_pnl'] > 0]
        print(f"  Positive ensemble avg TEST = {ens_pos['val_b_avg_pnl'].mean():+.2%}  "
              f"WR = {ens_pos['val_b_win_rate'].mean():.0%}")


# ==============================================================================
# STEP 7: TP/SL GRID HEATMAP — which TP/SL combos work best?
# ==============================================================================
print()
print("=" * W)
print(f"  F. TP/SL ANALYSIS — WHICH LEVELS WORK FOR SHORTS?")
print("=" * W)

static_rows = per_seed[per_seed['tpsl_mode'].isin(['static', 'grace', 'limit_entry'])].copy()
if len(static_rows) > 0 and static_rows['tp_pct'].notna().any():
    tp_sl = static_rows.groupby(['tp_pct', 'sl_pct']).agg(
        n=('val_b_avg_pnl', 'count'),
        mean_test=('val_b_avg_pnl', 'mean'),
        med_test=('val_b_avg_pnl', 'median'),
        pct_pos=('val_b_avg_pnl', lambda x: (x > 0).mean() * 100),
        mean_opt=('opt_avg_pnl', 'mean'),
    ).reset_index().sort_values('mean_test', ascending=False)

    print(f"\n  {'TP':>5} {'SL':>5} │ {'N':>5} {'Mean TEST':>10} {'Med TEST':>9} {'%Pos':>5} │ "
          f"{'Mean OPT':>9} {'Degrad':>7}")
    print(f"  {'─'*5} {'─'*5} │ {'─'*5} {'─'*10} {'─'*9} {'─'*5} │ {'─'*9} {'─'*7}")
    for _, r in tp_sl.iterrows():
        degrad = 1 - r['mean_test'] / r['mean_opt'] if r['mean_opt'] > 0 else 0
        print(f"  {r['tp_pct']:>4.0%} {r['sl_pct']:>4.0%} │ {int(r['n']):>5} "
              f"{r['mean_test']:>+9.2%} {r['med_test']:>+8.2%} {r['pct_pos']:>4.0f}% │ "
              f"{r['mean_opt']:>+8.2%} {degrad:>+6.0%}")

    if len(tp_sl) > 0:
        best_tp_sl = tp_sl.iloc[0]
        print(f"\n  ★ Best TP/SL combo: {best_tp_sl['tp_pct']:.0%}/{best_tp_sl['sl_pct']:.0%} "
              f"(mean TEST = {best_tp_sl['mean_test']:+.2%}, "
              f"{best_tp_sl['pct_pos']:.0f}% positive)")
else:
    print("  No static/grace/limit_entry results with TP/SL pct values.")


# ==============================================================================
# STEP 8: PROB THRESHOLD ANALYSIS
# ==============================================================================
print()
print("=" * W)
print(f"  G. PROB THRESHOLD — SIGNAL CONFIDENCE vs TRADE FREQUENCY")
print("=" * W)

prob_stats = per_seed.groupby('prob_thresh').agg(
    n=('val_b_avg_pnl', 'count'),
    mean_test=('val_b_avg_pnl', 'mean'),
    pct_pos=('val_b_avg_pnl', lambda x: (x > 0).mean() * 100),
    mean_count=('val_b_count', 'mean'),
).reset_index().sort_values('prob_thresh')

print(f"\n  {'Prob≥':>6} │ {'N':>5} {'Mean TEST':>10} {'%Pos':>5} │ {'Avg Trades':>10}")
print(f"  {'─'*6} │ {'─'*5} {'─'*10} {'─'*5} │ {'─'*10}")
for _, r in prob_stats.iterrows():
    print(f"  {r['prob_thresh']:>5.2f} │ {int(r['n']):>5} {r['mean_test']:>+9.2%} "
          f"{r['pct_pos']:>4.0f}% │ {r['mean_count']:>9.1f}")

if len(prob_stats) > 0:
    sweet = prob_stats[prob_stats['mean_test'] > 0]
    if len(sweet) > 0:
        best_prob = sweet.loc[sweet['mean_test'].idxmax()]
        print(f"\n  ★ Best prob threshold: ≥{best_prob['prob_thresh']:.2f} "
              f"(mean TEST = {best_prob['mean_test']:+.2%}, "
              f"{best_prob['pct_pos']:.0f}% positive, "
              f"~{best_prob['mean_count']:.0f} trades)")


# ==============================================================================
# STEP 9: OVERFITTING DIAGNOSTIC
# ==============================================================================
print()
print("=" * W)
print(f"  H. OVERFITTING DIAGNOSTIC — OPT → VAL → TEST DEGRADATION")
print("=" * W)

print(f"\n  Distribution of TEST / OPT ratio (1.0 = no degradation, 0.0 = total overfit):")
ratios = per_seed[per_seed['opt_avg_pnl'] > 0].copy()
if len(ratios) > 0:
    ratios['test_opt_ratio'] = ratios['val_b_avg_pnl'] / ratios['opt_avg_pnl']
    ratios['val_opt_ratio'] = ratios['val_a_avg_pnl'] / ratios['opt_avg_pnl']

    for label, col in [('VAL/OPT', 'val_opt_ratio'), ('TEST/OPT', 'test_opt_ratio')]:
        vals = ratios[col].clip(-2, 3)
        p25, p50, p75 = vals.quantile([0.25, 0.50, 0.75])
        neg = (vals < 0).mean() * 100
        print(f"    {label}: median={p50:.2f}  IQR=[{p25:.2f}, {p75:.2f}]  "
              f"negative={neg:.0f}%")

    print(f"\n  Degradation by mode:")
    for mode in modes:
        md = ratios[ratios['tpsl_mode'] == mode]
        if len(md) < 3:
            continue
        med_ratio = md['test_opt_ratio'].median()
        neg_pct = (md['test_opt_ratio'] < 0).mean() * 100
        print(f"    {mode:<15}: median TEST/OPT = {med_ratio:.2f}  "
              f"({neg_pct:.0f}% flipped negative)")


# ==============================================================================
# STEP 10: CHARTS
# ==============================================================================
print(f"\n  Generating charts...")

_all_test_pnl = {}
if 'ticker_df' in dir():
    for _, r in ticker_df.iterrows():
        t = r['ticker']
        _all_test_pnl[t] = max(_all_test_pnl.get(t, -999), r['val_b_avg_pnl'])
if len(ensemble) > 0:
    for _, r in ensemble.iterrows():
        t = r['ticker']
        _all_test_pnl[t] = max(_all_test_pnl.get(t, -999), r['val_b_avg_pnl'])
n_pos_bars = sum(1 for v in _all_test_pnl.values() if v > 0) or 30
fig_h = max(12, 6 + n_pos_bars * 0.3)
fig, axes = plt.subplots(2, 2, figsize=(16, fig_h))

# Chart 1: OPT vs TEST scatter
ax = axes[0, 0]
for mode in modes:
    md = per_seed[per_seed['tpsl_mode'] == mode]
    ax.scatter(md['opt_avg_pnl'] * 100, md['val_b_avg_pnl'] * 100,
               alpha=0.5, s=20, label=mode)
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)
lim = max(abs(per_seed['opt_avg_pnl'].max()), abs(per_seed['val_b_avg_pnl']).max()) * 100 + 1
ax.plot([-lim, lim], [-lim, lim], 'k--', alpha=0.3, linewidth=1)
ax.set_xlabel('OPT Avg PnL (%)')
ax.set_ylabel('TEST Avg PnL (%)')
ax.set_title('OPT vs TEST PnL (below diagonal = degradation)', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Chart 2: TEST PnL distribution
ax = axes[0, 1]
for mode in modes:
    md = per_seed[per_seed['tpsl_mode'] == mode]
    vals = md['val_b_avg_pnl'].clip(-0.10, 0.15) * 100
    ax.hist(vals, bins=30, alpha=0.5, label=mode)
ax.axvline(0, color='black', linewidth=1)
ax.axvline(DEPLOY_PNL_THRESHOLD * 100, color='red', linewidth=1, linestyle='--',
           label=f'Deploy ({DEPLOY_PNL_THRESHOLD:.1%})')
ax.set_xlabel('TEST Avg PnL (%)')
ax.set_ylabel('Count')
ax.set_title('TEST PnL Distribution', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Chart 3: Per-ticker best TEST PnL (bar chart) — merge per-seed + ensemble
ax = axes[1, 0]
_chart_rows = {}
if 'ticker_df' in dir() and len(ticker_df) > 0:
    for _, r in ticker_df.iterrows():
        t = r['ticker']
        pnl = r['val_b_avg_pnl']
        src = f"S{int(r['seed'])}"
        if t not in _chart_rows or pnl > _chart_rows[t][0]:
            _chart_rows[t] = (pnl, src)
if len(ensemble) > 0:
    for _, r in ensemble.iterrows():
        t = r['ticker']
        pnl = r['val_b_avg_pnl']
        if t not in _chart_rows or pnl > _chart_rows[t][0]:
            _chart_rows[t] = (pnl, 'ENS')
if _chart_rows:
    _chart_sorted = sorted(_chart_rows.items(), key=lambda x: x[1][0], reverse=True)
    _chart_pos = [(t, pnl, src) for t, (pnl, src) in _chart_sorted if pnl > 0]
    if len(_chart_pos) == 0:
        _chart_pos = [(t, pnl, src) for t, (pnl, src) in _chart_sorted[:30]]
    tickers_c = [x[0] for x in _chart_pos]
    pnls_c = [x[1] for x in _chart_pos]
    labels_c = [f"{x[0]} ({x[2]})" for x in _chart_pos]
    colors = ['#2ca02c' if v >= DEPLOY_PNL_THRESHOLD else
              '#1f77b4' if v > 0 else '#d62728'
              for v in pnls_c]
    ax.barh(range(len(pnls_c)), [v * 100 for v in pnls_c], color=colors)
    ax.set_yticks(range(len(pnls_c)))
    ax.set_yticklabels(labels_c, fontsize=8)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.axvline(DEPLOY_PNL_THRESHOLD * 100, color='red', linewidth=1, linestyle='--')
    ax.set_xlabel('Best TEST Avg PnL (%)')
    ax.set_title(f'Per-Ticker Best TEST PnL ({len(_chart_pos)} tickers, seed or ensemble)', fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
else:
    ax.text(0.5, 0.5, 'No ticker data', ha='center', va='center', transform=ax.transAxes)

# Chart 4: TP level vs TEST PnL
ax = axes[1, 1]
tp_analysis = static_rows[static_rows['tp_pct'] > 0].copy() if len(static_rows) > 0 else pd.DataFrame()
if len(tp_analysis) > 0:
    tp_grp = tp_analysis.groupby('tp_pct')['val_b_avg_pnl'].agg(['mean', 'median', 'count']).reset_index()
    ax.bar(tp_grp['tp_pct'] * 100, tp_grp['mean'] * 100, width=0.4, alpha=0.7, label='Mean TEST')
    ax.bar(tp_grp['tp_pct'] * 100 + 0.4, tp_grp['median'] * 100, width=0.4, alpha=0.7, label='Median TEST')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('TP Level (%)')
    ax.set_ylabel('TEST Avg PnL (%)')
    ax.set_title('TP Level vs TEST PnL (deeper TP exploration)', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    for _, r in tp_grp.iterrows():
        ax.annotate(f'n={int(r["count"])}', (r['tp_pct'] * 100, r['mean'] * 100),
                    fontsize=7, ha='center', va='bottom')
else:
    ax.text(0.5, 0.5, 'No TP data', ha='center', va='center', transform=ax.transAxes)

plt.suptitle('Exp 400 SHORT/Dump Model — WF Optimization Dashboard', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/tmp/analyze_wf_400_dashboard.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"  Charts saved: /tmp/analyze_wf_400_dashboard.png")


# ==============================================================================
# FINAL SUMMARY
# ==============================================================================
print()
print("=" * W)
print(f"  FINAL SUMMARY — EXP 400 SHORT/DUMP MODEL")
print("=" * W)

print(f"\n  Results so far: {len(per_seed)} per-seed, {len(ensemble)} ensemble")
print(f"  Tickers processed: {per_seed['ticker'].nunique()}")
print(f"  Modes tested: {', '.join(modes)}")

print(f"\n  Grand mean PnL:")
print(f"    OPT:  {opt_all:+.3%}")
print(f"    VAL:  {val_all:+.3%}")
print(f"    TEST: {test_all:+.3%}")

if test_all > DEPLOY_PNL_THRESHOLD:
    print(f"\n  ★ MODEL IS VIABLE — avg TEST exceeds deploy threshold")
    if 'ticker_df' in dir():
        deployable = ticker_df[
            (ticker_df['val_b_avg_pnl'] >= DEPLOY_PNL_THRESHOLD) &
            (ticker_df['val_b_count'] >= MIN_TEST_TRADES)
        ]
        if len(deployable) > 0:
            print(f"  ★ {len(deployable)} DEPLOYABLE TICKERS:")
            for _, r in deployable.iterrows():
                print(f"      {r['ticker']:<7} TEST={r['val_b_avg_pnl']:+.2%}  "
                      f"WR={r['val_b_win_rate']:.0%}  N={int(r['val_b_count'])}")
elif test_all > 0:
    print(f"\n  ★ MODEL IS MARGINALLY VIABLE — positive but below {DEPLOY_PNL_THRESHOLD:.1%}")
    print(f"    Select best tickers individually (see Section B)")
else:
    print(f"\n  ★ MODEL IS NOT VIABLE AT PORTFOLIO LEVEL — avg TEST is negative")
    print(f"    Individual tickers may still work (see Section B)")
    if 'ticker_df' in dir():
        pos = ticker_df[ticker_df['val_b_avg_pnl'] > 0]
        if len(pos) > 0:
            print(f"    {len(pos)} tickers with positive TEST — investigate individually")

print(f"\n  Next steps:")
if per_seed['ticker'].nunique() < 20:
    print(f"    • Optimizer still running — re-run this script when more tickers complete")
if len(modes) < 4:
    print(f"    • Only {len(modes)} modes tested — consider running all 4 default modes")
print(f"    • If viable: build deploy config (same as S3 but SHORT direction)")
print(f"    • If marginal: try training folds 401-405 for more robust parameters")
print("=" * W)
