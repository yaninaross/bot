"""
compare_400_vs_410.py — Exp 400 vs 401 vs 410 — SHORT/Dump Fold Robustness & Alpha Comparison
===============================================================================================
Queries BQ for WF optimization results and splits by EXPERIMENT (not just model_version):
  - 400 (alpha=2.0, cutoff Aug 2025) — model_version='robust_wf_v12_short'
  - 401 (alpha=2.0, cutoff Sep 2025) — model_version='robust_wf_v12_short_a1'
  - 410 (alpha=1.0, cutoff Aug 2025) — model_version='robust_wf_v12_short_a1'

Sections:
  A. Overall viability — side-by-side mode summary (3 experiments)
  B. Fold robustness — 400 vs 401 per-ticker consistency (main analysis)
  C. Alpha comparison — 400 vs 410 head-to-head (if enough 410 data)
  D. Degradation cascade — OPT → VAL → TEST retention
  E. Mode comparison — which modes work best per experiment
  F. Charts (fold scatter, delta bar, deploy candidates, degradation, boxplot, agreement)

Works with partial results. Colab-ready (no if __name__).
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

bq = bigquery.Client()

# ==============================================================================
# CONFIG
# ==============================================================================
W = 130
BQ_TABLE = f"{GCP_PROJECT}.scoring_logs.wf_optimization_results"
MV_400 = 'robust_wf_v12_short'
MV_410 = 'robust_wf_v12_short_a1'
DEPLOY_PNL_THRESHOLD = 0.018
MIN_TEST_TRADES = 3


# ==============================================================================
# LOAD — single query, split by experiment prefix
# ==============================================================================
print(f"{'='*W}")
print(f"  EXP 400 vs 401 vs 410 — SHORT/DUMP FOLD ROBUSTNESS & ALPHA COMPARISON")
print(f"{'='*W}")

q = f"""
SELECT
    model_version,
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
WHERE model_version IN ('{MV_400}', '{MV_410}')
ORDER BY experiment, ticker, tpsl_mode, seed
"""
df_all = bq.query(q).to_dataframe()

df_400 = df_all[df_all['experiment'].str.startswith('400')].copy()
df_401 = df_all[df_all['experiment'].str.startswith('401')].copy()
df_410 = df_all[df_all['experiment'].str.startswith('410')].copy()

EXPS = {'400': df_400, '401': df_401, '410': df_410}
EXP_LABELS = {'400': '400 (a=2.0, Aug)', '401': '401 (a=2.0, Sep)', '410': '410 (a=1.0, Aug)'}

ps, en = {}, {}
for eid, df in EXPS.items():
    ps[eid] = df[df['is_ensemble'] != True].copy()
    en[eid] = df[df['is_ensemble'] == True].copy()

tsets = {eid: set(ps[eid]['ticker'].unique()) for eid in EXPS}
overlap_400_401 = sorted(tsets['400'] & tsets['401'])
overlap_400_410 = sorted(tsets['400'] & tsets['410'])
overlap_all = sorted(tsets['400'] & tsets['401'] & tsets['410'])

print(f"\n  Loaded {len(df_all)} total rows from BQ")
for eid in EXPS:
    print(f"  Exp {EXP_LABELS[eid]}: {len(ps[eid])} per-seed, {len(en[eid])} ensemble, "
          f"{len(tsets[eid])} tickers")
print(f"\n  Overlap 400∩401: {len(overlap_400_401)}  |  400∩410: {len(overlap_400_410)}  |  "
      f"All three: {len(overlap_all)}")

if not overlap_400_401 and not overlap_400_410:
    print(f"\n  No overlapping tickers — cannot compare. Wait for more results.")
    raise SystemExit


# ==============================================================================
# HELPER: best per ticker
# ==============================================================================
def best_per_ticker(per_seed, ensemble):
    result = {}
    if len(per_seed) > 0:
        for (ticker, mode), grp in per_seed.groupby(['ticker', 'tpsl_mode']):
            best = grp.loc[grp['opt_avg_pnl'].idxmax()]
            pnl = best['val_b_avg_pnl']
            src = f"S{int(best['seed'])}"
            if ticker not in result or pnl > result[ticker][0]:
                result[ticker] = (pnl, mode, src,
                                  best['opt_avg_pnl'], best['val_a_avg_pnl'],
                                  best['val_b_win_rate'],
                                  int(best['val_b_count']) if pd.notna(best['val_b_count']) else 0)
    if len(ensemble) > 0:
        for _, r in ensemble.iterrows():
            ticker = r['ticker']
            pnl = r['val_b_avg_pnl']
            mode = r['tpsl_mode'] or 'static'
            if ticker not in result or pnl > result[ticker][0]:
                result[ticker] = (pnl, mode, 'ENS',
                                  r['opt_avg_pnl'], r['val_a_avg_pnl'],
                                  r['val_b_win_rate'],
                                  int(r['val_b_count']) if pd.notna(r['val_b_count']) else 0)
    return result


best = {eid: best_per_ticker(ps[eid], en[eid]) for eid in EXPS}


# ==============================================================================
# A. OVERALL VIABILITY — three-way mode summary
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  A. OVERALL VIABILITY — THREE-WAY MODE SUMMARY")
print(f"{'='*W}")

all_modes = sorted(set().union(*(ps[e]['tpsl_mode'].dropna().unique() for e in EXPS if len(ps[e]) > 0)))

hdr_parts = []
for eid in EXPS:
    hdr_parts.append(f"{'── ' + eid + ' ──────────────────':}")
sep_parts = []
for _ in EXPS:
    sep_parts.append(f"{'─'*7} {'─'*7} {'─'*8} {'─'*4} {'─'*5}")

print(f"\n  {'Mode':<15} │ " + " │ ".join(hdr_parts))
print(f"  {'':15} │ " + " │ ".join(
    f"{'OPT':>7} {'VAL':>7} {'TEST':>8} {'T+':>4} {'Deg':>5}" for _ in EXPS))
print(f"  {'─'*15} │ " + " │ ".join(sep_parts))

for mode in all_modes:
    parts = []
    test_vals = {}
    for eid in EXPS:
        md = ps[eid][ps[eid]['tpsl_mode'] == mode]
        if len(md) >= 3:
            o = md['opt_avg_pnl'].mean()
            v = md['val_a_avg_pnl'].mean()
            t = md['val_b_avg_pnl'].mean()
            tp = (md['val_b_avg_pnl'] > 0).sum()
            d = 1 - t / o if o > 0 else 0
            parts.append(f"{o:>+6.2%} {v:>+6.2%} {t:>+7.2%} {tp:>4} {d:>+4.0%}")
            test_vals[eid] = t
        else:
            parts.append(f"{'—':>7} {'—':>7} {'—':>8} {'—':>4} {'—':>5}")
    print(f"  {mode:<15} │ " + " │ ".join(parts))

print(f"  {'─'*15} │ " + " │ ".join(sep_parts))
for eid in EXPS:
    p = ps[eid]
    if len(p) > 0:
        o = p['opt_avg_pnl'].mean()
        v = p['val_a_avg_pnl'].mean()
        t = p['val_b_avg_pnl'].mean()
        d = 1 - t / o if o > 0 else 0
        tp = (p['val_b_avg_pnl'] > 0).sum()
        print(f"  {'ALL ('+eid+')':<15} │ {o:>+6.2%} {v:>+6.2%} {t:>+7.2%} {tp:>4} {d:>+4.0%} │")

grand = {}
for eid in EXPS:
    if len(ps[eid]) > 0:
        grand[eid] = ps[eid]['val_b_avg_pnl'].mean()
parts = [f"{eid}={v:+.3%}" for eid, v in grand.items()]
print(f"\n  Grand mean TEST:  " + "  ".join(parts))


# ==============================================================================
# B. FOLD ROBUSTNESS — 400 vs 401
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  B. FOLD ROBUSTNESS — 400 vs 401 (same model, different cutoffs)")
print(f"{'='*W}")

if not overlap_400_401:
    print(f"\n  No overlapping tickers between 400 and 401.")
else:
    print(f"\n  {len(overlap_400_401)} tickers scored by BOTH folds.")

    print(f"\n  {'Ticker':<7} │ {'400 TEST':>8} {'WR':>4} {'N':>3} │ "
          f"{'401 TEST':>8} {'WR':>4} {'N':>3} │ "
          f"{'Avg TEST':>8} │ {'Both+':>5} {'Deploy':>6}")
    print(f"  {'─'*7} │ {'─'*8} {'─'*4} {'─'*3} │ "
          f"{'─'*8} {'─'*4} {'─'*3} │ "
          f"{'─'*8} │ {'─'*5} {'─'*6}")

    robustness_rows = []
    for tk in overlap_400_401:
        b4 = best['400'].get(tk)
        b1 = best['401'].get(tk)
        if not b4 or not b1:
            continue

        t4, wr4, n4 = b4[0], b4[5], b4[6]
        t1, wr1, n1 = b1[0], b1[5], b1[6]
        avg_test = (t4 + t1) / 2
        both_pos = t4 > 0 and t1 > 0
        deployable = (both_pos and avg_test >= DEPLOY_PNL_THRESHOLD
                      and n4 >= MIN_TEST_TRADES and n1 >= MIN_TEST_TRADES)

        robustness_rows.append({
            'ticker': tk, 't400': t4, 'wr400': wr4, 'n400': n4,
            't401': t1, 'wr401': wr1, 'n401': n1,
            'avg_test': avg_test, 'both_pos': both_pos, 'deployable': deployable,
        })

    robustness_rows.sort(key=lambda x: x['avg_test'], reverse=True)

    for r in robustness_rows:
        bp_tag = "YES" if r['both_pos'] else "—"
        dep_tag = "YES" if r['deployable'] else "—"
        print(f"  {r['ticker']:<7} │ {r['t400']:>+7.2%} {r['wr400']:>3.0%} {r['n400']:>3} │ "
              f"{r['t401']:>+7.2%} {r['wr401']:>3.0%} {r['n401']:>3} │ "
              f"{r['avg_test']:>+7.2%} │ {bp_tag:>5} {dep_tag:>6}")

    n_both_pos = sum(1 for r in robustness_rows if r['both_pos'])
    n_one_pos = sum(1 for r in robustness_rows if (r['t400'] > 0) != (r['t401'] > 0))
    n_both_neg = sum(1 for r in robustness_rows if r['t400'] <= 0 and r['t401'] <= 0)
    n_deploy = sum(1 for r in robustness_rows if r['deployable'])

    print(f"\n  {'─'*80}")
    print(f"  Both folds positive:  {n_both_pos}/{len(robustness_rows)}  (high confidence)")
    print(f"  One fold only:        {n_one_pos}/{len(robustness_rows)}  (unstable)")
    print(f"  Both folds negative:  {n_both_neg}/{len(robustness_rows)}  (reject)")
    print(f"  Deployable:           {n_deploy}/{len(robustness_rows)}  "
          f"(both+ AND avg>={DEPLOY_PNL_THRESHOLD:.1%} AND >={MIN_TEST_TRADES} trades)")

    if n_both_pos > 0:
        bp_rows = [r for r in robustness_rows if r['both_pos']]
        bp_avg = np.mean([r['avg_test'] for r in bp_rows])
        bp_med = np.median([r['avg_test'] for r in bp_rows])
        print(f"\n  Both-positive tickers:  mean avg TEST = {bp_avg:+.2%}  |  "
              f"median = {bp_med:+.2%}")

    vals_400 = [r['t400'] for r in robustness_rows]
    vals_401 = [r['t401'] for r in robustness_rows]
    if len(vals_400) >= 3:
        corr = np.corrcoef(vals_400, vals_401)[0, 1]
        print(f"  Cross-fold correlation: r = {corr:.2f}")

    if n_deploy > 0:
        print(f"\n  DEPLOY CANDIDATES (both folds positive, avg >= {DEPLOY_PNL_THRESHOLD:.1%}):")
        for r in robustness_rows:
            if r['deployable']:
                print(f"    {r['ticker']:<7}  avg TEST={r['avg_test']:+.2%}  "
                      f"(400={r['t400']:+.2%}, 401={r['t401']:+.2%})")


# ==============================================================================
# C. ALPHA COMPARISON — 400 vs 410
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  C. ALPHA COMPARISON — 400 (alpha=2.0) vs 410 (alpha=1.0)")
print(f"{'='*W}")

if len(overlap_400_410) < 3:
    print(f"\n  Only {len(overlap_400_410)} overlapping tickers — not enough data for comparison.")
    print(f"  410 has {len(tsets['410'])} tickers total. Wait for more 410 WF results.")
    if overlap_400_410:
        print(f"  Available: {', '.join(overlap_400_410)}")
        for tk in overlap_400_410:
            b4 = best['400'].get(tk)
            b1 = best['410'].get(tk)
            if b4 and b1:
                print(f"    {tk}: 400={b4[0]:+.2%}  410={b1[0]:+.2%}  Δ={b1[0]-b4[0]:+.2%}")
else:
    print(f"\n  {'Ticker':<7} │ {'── 400 (alpha=2.0) ─────────────':} │ "
          f"{'── 410 (alpha=1.0) ─────────────':} │ {'Winner':>6} {'Δ':>8}")
    print(f"  {'':7} │ {'Mode':<14} {'Src':>3} {'TEST':>7} {'WR':>4} {'N':>3} │ "
          f"{'Mode':<14} {'Src':>3} {'TEST':>7} {'WR':>4} {'N':>3} │")
    print(f"  {'─'*7} │ {'─'*14} {'─'*3} {'─'*7} {'─'*4} {'─'*3} │ "
          f"{'─'*14} {'─'*3} {'─'*7} {'─'*4} {'─'*3} │ {'─'*6} {'─'*8}")

    wins_400 = 0
    wins_410 = 0
    for tk in overlap_400_410:
        b4 = best['400'].get(tk)
        b1 = best['410'].get(tk)
        s4 = f"{b4[1]:<14} {b4[2]:>3} {b4[0]:>+6.2%} {b4[5]:>3.0%} {b4[6]:>3}" if b4 else f"{'—':<14} {'':>3} {'—':>7} {'—':>4} {'—':>3}"
        s1 = f"{b1[1]:<14} {b1[2]:>3} {b1[0]:>+6.2%} {b1[5]:>3.0%} {b1[6]:>3}" if b1 else f"{'—':<14} {'':>3} {'—':>7} {'—':>4} {'—':>3}"
        v4 = b4[0] if b4 else -999
        v1 = b1[0] if b1 else -999
        if v4 > v1:
            winner = '400'; wins_400 += 1
        elif v1 > v4:
            winner = '410'; wins_410 += 1
        else:
            winner = 'TIE'
        delta_str = f"{v1 - v4:+.2%}" if v4 != -999 and v1 != -999 else '—'
        print(f"  {tk:<7} │ {s4} │ {s1} │ {winner:>6} {delta_str:>8}")

    print(f"\n  Score: 400 wins {wins_400} — 410 wins {wins_410}")


# ==============================================================================
# D. DEGRADATION CASCADE — OPT → VAL → TEST
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  D. DEGRADATION — OPT → VAL → TEST RETENTION")
print(f"{'='*W}")

cascade = {}
for eid in EXPS:
    p = ps[eid]
    if len(p) == 0:
        print(f"\n  ── {EXP_LABELS[eid]}: no data ──")
        continue
    pos = p[p['opt_avg_pnl'] > 0].copy()
    if len(pos) == 0:
        print(f"\n  ── {EXP_LABELS[eid]}: no positive OPT rows ──")
        continue

    o = pos['opt_avg_pnl'].mean()
    v = pos['val_a_avg_pnl'].mean()
    t = pos['val_b_avg_pnl'].mean()
    val_ret = v / o if o != 0 else 0
    test_ret = t / o if o != 0 else 0
    n_val_pos = (pos['val_a_avg_pnl'] > 0).sum()
    n_test_pos = (pos['val_b_avg_pnl'] > 0).sum()
    n_flipped = ((pos['val_a_avg_pnl'] > 0) & (pos['val_b_avg_pnl'] <= 0)).sum()
    flip_rate = n_flipped / n_val_pos if n_val_pos > 0 else 0
    med_ratio = (pos['val_b_avg_pnl'] / pos['opt_avg_pnl']).clip(-2, 3).median()

    cascade[eid] = {'test_ret': test_ret, 'flip_rate': flip_rate, 'med_ratio': med_ratio}

    print(f"\n  ── {EXP_LABELS[eid]} ──")
    print(f"  OPT mean:   {o:>+.3%}")
    print(f"  VAL mean:   {v:>+.3%}  (retention: {val_ret:.0%})")
    print(f"  TEST mean:  {t:>+.3%}  (retention: {test_ret:.0%})")
    print(f"  Median TEST/OPT ratio: {med_ratio:.2f}")
    print(f"  VAL>0: {n_val_pos}  TEST>0: {n_test_pos}  "
          f"Flipped (VAL>0→TEST≤0): {n_flipped} ({flip_rate:.0%})")

if '400' in cascade and '401' in cascade:
    print(f"\n  Fold comparison:")
    print(f"    OPT→TEST retention:  400={cascade['400']['test_ret']:.0%}  "
          f"401={cascade['401']['test_ret']:.0%}")
    print(f"    Flip rate:           400={cascade['400']['flip_rate']:.0%}  "
          f"401={cascade['401']['flip_rate']:.0%}")


# ==============================================================================
# E. MODE COMPARISON
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  E. MODE COMPARISON — BEST MODES PER EXPERIMENT")
print(f"{'='*W}")

hdr = f"  {'Mode':<15}"
for eid in EXPS:
    hdr += f" │ {'─ '+eid+' ────────────────────────':}"
print(f"\n{hdr}")

sub = f"  {'':15}"
for _ in EXPS:
    sub += f" │ {'Mean TEST':>9} {'Med':>7} {'%Pos':>5}"
print(sub)

sep = f"  {'─'*15}"
for _ in EXPS:
    sep += f" │ {'─'*9} {'─'*7} {'─'*5}"
print(sep)

for mode in all_modes:
    line = f"  {mode:<15}"
    for eid in EXPS:
        md = ps[eid][ps[eid]['tpsl_mode'] == mode]
        if len(md) >= 3:
            mean_t = md['val_b_avg_pnl'].mean()
            med_t = md['val_b_avg_pnl'].median()
            pp = (md['val_b_avg_pnl'] > 0).mean() * 100
            line += f" │ {mean_t:>+8.2%} {med_t:>+6.2%} {pp:>4.0f}%"
        else:
            line += f" │ {'—':>9} {'—':>7} {'—':>5}"
    print(line)


# ==============================================================================
# F. CHARTS
# ==============================================================================
print(f"\n  Generating charts...")

fig, axes = plt.subplots(2, 3, figsize=(22, 14))

# Chart 1: Scatter — 400 vs 401 TEST PnL per ticker (fold consistency)
ax = axes[0, 0]
if overlap_400_401:
    x_vals = [best['400'][t][0] * 100 for t in overlap_400_401 if t in best['400'] and t in best['401']]
    y_vals = [best['401'][t][0] * 100 for t in overlap_400_401 if t in best['400'] and t in best['401']]
    tks = [t for t in overlap_400_401 if t in best['400'] and t in best['401']]
    colors_sc = ['#2ca02c' if x > 0 and y > 0 else '#d62728' if x <= 0 and y <= 0
                 else '#ff7f0e' for x, y in zip(x_vals, y_vals)]
    ax.scatter(x_vals, y_vals, c=colors_sc, alpha=0.6, s=40, edgecolors='white', linewidth=0.5)
    lim = max(max(abs(v) for v in x_vals + y_vals) + 1, 5) if x_vals else 5
    ax.plot([-lim, lim], [-lim, lim], 'k--', alpha=0.3, linewidth=1)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    for i, tk in enumerate(tks):
        if abs(x_vals[i]) > 2 or abs(y_vals[i]) > 2:
            ax.annotate(tk, (x_vals[i], y_vals[i]), fontsize=6, alpha=0.7,
                        xytext=(3, 3), textcoords='offset points')
ax.set_xlabel('Exp 400 TEST PnL (%)')
ax.set_ylabel('Exp 401 TEST PnL (%)')
ax.set_title('Fold Consistency: 400 vs 401 TEST PnL\n(green=both+, red=both-, orange=split)',
             fontweight='bold')
ax.grid(True, alpha=0.3)

# Chart 2: Bar — per-ticker Δ(401 - 400) TEST PnL
ax = axes[0, 1]
if robustness_rows:
    deltas = [(r['ticker'], r['t401'] - r['t400']) for r in robustness_rows]
    deltas.sort(key=lambda x: x[1], reverse=True)
    tickers_d = [d[0] for d in deltas]
    vals_d = [d[1] * 100 for d in deltas]
    colors_d = ['#2ca02c' if v > 0 else '#d62728' for v in vals_d]
    ax.barh(range(len(tickers_d)), vals_d, color=colors_d, height=0.7)
    ax.set_yticks(range(len(tickers_d)))
    ax.set_yticklabels(tickers_d, fontsize=7)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.invert_yaxis()
ax.set_xlabel('Δ TEST PnL (%) [401 − 400]')
ax.set_title('Per-Ticker Fold Delta\n(green = 401 better, red = 400 better)', fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

# Chart 3: Bar — deploy candidates (both folds positive), sorted by avg TEST
ax = axes[0, 2]
if robustness_rows:
    bp_rows = [r for r in robustness_rows if r['both_pos']]
    bp_rows.sort(key=lambda x: x['avg_test'], reverse=True)
    if bp_rows:
        tickers_bp = [r['ticker'] for r in bp_rows]
        avg_vals = [r['avg_test'] * 100 for r in bp_rows]
        colors_bp = ['#2ca02c' if v >= DEPLOY_PNL_THRESHOLD * 100 else '#1f77b4' for v in avg_vals]
        ax.barh(range(len(tickers_bp)), avg_vals, color=colors_bp, height=0.7)
        ax.set_yticks(range(len(tickers_bp)))
        ax.set_yticklabels(tickers_bp, fontsize=8)
        ax.axvline(DEPLOY_PNL_THRESHOLD * 100, color='red', linewidth=1, linestyle='--',
                   label=f'Deploy ({DEPLOY_PNL_THRESHOLD:.1%})')
        ax.axvline(0, color='black', linewidth=0.5)
        ax.invert_yaxis()
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, 'No tickers positive\nin both folds',
                ha='center', va='center', transform=ax.transAxes)
else:
    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
ax.set_xlabel('Avg TEST PnL (%)')
ax.set_title('Deploy Candidates\n(positive in BOTH folds)', fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

# Chart 4: Grouped bar — OPT→VAL→TEST for all experiments
ax = axes[1, 0]
labels_c = ['OPT', 'VAL', 'TEST']
exp_with_data = [e for e in EXPS if len(ps[e]) > 0]
n_exp = len(exp_with_data)
width = 0.25
exp_colors = {'400': '#1f77b4', '401': '#ff7f0e', '410': '#2ca02c'}
for i, eid in enumerate(exp_with_data):
    p = ps[eid]
    vals = [p['opt_avg_pnl'].mean() * 100,
            p['val_a_avg_pnl'].mean() * 100,
            p['val_b_avg_pnl'].mean() * 100]
    x_pos = np.arange(len(labels_c))
    offset = (i - (n_exp - 1) / 2) * width
    bars = ax.bar(x_pos + offset, vals, width, label=EXP_LABELS[eid],
                  color=exp_colors[eid], alpha=0.8)
    for j, v in enumerate(vals):
        ax.text(x_pos[j] + offset, v, f'{v:+.2f}%', ha='center',
                va='bottom' if v >= 0 else 'top', fontsize=7)
ax.set_xticks(np.arange(len(labels_c)))
ax.set_xticklabels(labels_c, fontsize=11)
ax.axhline(0, color='black', linewidth=0.5)
ax.legend(fontsize=8)
ax.set_ylabel('Mean PnL (%)')
ax.set_title('OPT → VAL → TEST Pipeline', fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Chart 5: Box plot — TEST PnL distributions by mode, all experiments
ax = axes[1, 1]
box_data = []
box_labels = []
box_colors = []
for mode in all_modes:
    for eid in exp_with_data:
        md = ps[eid][ps[eid]['tpsl_mode'] == mode]
        if len(md) >= 3:
            box_data.append((md['val_b_avg_pnl'].clip(-0.10, 0.15) * 100).values)
            box_labels.append(f'{mode}\n{eid}')
            box_colors.append(exp_colors[eid])
if box_data:
    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True, showfliers=False)
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axhline(DEPLOY_PNL_THRESHOLD * 100, color='red', linewidth=1, linestyle='--', alpha=0.5)
    ax.tick_params(axis='x', labelsize=6, rotation=45)
ax.set_ylabel('TEST Avg PnL (%)')
ax.set_title('TEST PnL Distribution by Mode', fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Chart 6: Fold agreement matrix (2x2 heatmap)
ax = axes[1, 2]
if robustness_rows:
    pp = sum(1 for r in robustness_rows if r['t400'] > 0 and r['t401'] > 0)
    pn = sum(1 for r in robustness_rows if r['t400'] > 0 and r['t401'] <= 0)
    np_ = sum(1 for r in robustness_rows if r['t400'] <= 0 and r['t401'] > 0)
    nn = sum(1 for r in robustness_rows if r['t400'] <= 0 and r['t401'] <= 0)
    matrix = np.array([[pp, pn], [np_, nn]])
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['401+', '401−'])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['400+', '400−'])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha='center', va='center',
                    fontsize=20, fontweight='bold')
    ax.set_title('Fold Agreement Matrix\n(400 vs 401 TEST PnL sign)', fontweight='bold')
else:
    ax.text(0.5, 0.5, 'No overlap data', ha='center', va='center', transform=ax.transAxes)

plt.suptitle('Exp 400 vs 401 vs 410 — SHORT/Dump Fold Robustness & Alpha Comparison',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/tmp/compare_400_vs_410.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"  Charts saved: /tmp/compare_400_vs_410.png")


# ==============================================================================
# SUMMARY
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  SUMMARY")
print(f"{'='*W}")

print(f"\n  Data: {len(df_all)} rows — "
      f"400: {len(tsets['400'])} tickers, 401: {len(tsets['401'])} tickers, "
      f"410: {len(tsets['410'])} tickers")

if grand:
    parts = [f"{eid}={v:+.3%}" for eid, v in grand.items()]
    print(f"  Grand mean TEST:  " + "  ".join(parts))

if overlap_400_401 and robustness_rows:
    n_bp = sum(1 for r in robustness_rows if r['both_pos'])
    n_dep = sum(1 for r in robustness_rows if r['deployable'])
    print(f"\n  FOLD ROBUSTNESS (400 vs 401):")
    print(f"    {n_bp}/{len(robustness_rows)} tickers positive in BOTH folds")
    print(f"    {n_dep}/{len(robustness_rows)} deployable (both+ AND avg>={DEPLOY_PNL_THRESHOLD:.1%})")
    if n_dep > 0:
        dep_list = [r['ticker'] for r in robustness_rows if r['deployable']]
        print(f"    Deploy candidates: {', '.join(dep_list)}")

if len(overlap_400_410) >= 3:
    print(f"\n  ALPHA COMPARISON (400 vs 410):")
    print(f"    400 wins {wins_400} — 410 wins {wins_410} (on {len(overlap_400_410)} tickers)")
else:
    print(f"\n  ALPHA COMPARISON: insufficient data ({len(overlap_400_410)} overlap tickers)")

print(f"\n{'='*W}")
