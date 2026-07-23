"""
compare_400_vs_410.py — Head-to-head: Exp 400 (alpha=2.0) vs Exp 410 (alpha=1.0)
=================================================================================
Both are SHORT/dump models trained with the same architecture but different
ASL alpha_pos values. Compares WF optimization results from BQ:
  A. Overall viability — side-by-side mode summary
  B. Per-ticker head-to-head (best config from per-seed + ensemble)
  C. OPT → VAL → TEST degradation cascade
  D. Mode comparison — which modes benefit from alpha change
  E. Seed stability comparison
  F. Charts (scatter, delta bar, degradation, mode boxplot)

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
BQ_TABLE = 'ys-time-series.scoring_logs.wf_optimization_results'
MV_400 = 'robust_wf_v12_short'
MV_410 = 'robust_wf_v12_short_a1'
DEPLOY_PNL_THRESHOLD = 0.018
MIN_TEST_TRADES = 3


# ==============================================================================
# LOAD — single query, split by model_version
# ==============================================================================
print(f"{'═'*W}")
print(f"  EXP 400 (alpha=2.0) vs EXP 410 (alpha=1.0) — SHORT/DUMP HEAD-TO-HEAD")
print(f"{'═'*W}")

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
ORDER BY model_version, ticker, tpsl_mode, seed
"""
df_all = bq.query(q).to_dataframe()

df_400 = df_all[df_all['model_version'] == MV_400].copy()
df_410 = df_all[df_all['model_version'] == MV_410].copy()

ps_400 = df_400[df_400['is_ensemble'] != True].copy()
en_400 = df_400[df_400['is_ensemble'] == True].copy()
ps_410 = df_410[df_410['is_ensemble'] != True].copy()
en_410 = df_410[df_410['is_ensemble'] == True].copy()

t400 = set(ps_400['ticker'].unique())
t410 = set(ps_410['ticker'].unique())
overlap = sorted(t400 & t410)
only_400 = sorted(t400 - t410)
only_410 = sorted(t410 - t400)

print(f"\n  Exp 400 (alpha=2.0): {len(ps_400)} per-seed, {len(en_400)} ensemble, "
      f"{len(t400)} tickers")
print(f"  Exp 410 (alpha=1.0): {len(ps_410)} per-seed, {len(en_410)} ensemble, "
      f"{len(t410)} tickers")
print(f"  Overlap: {len(overlap)}  |  400-only: {len(only_400)}  |  410-only: {len(only_410)}")
if only_400:
    print(f"  400-only: {', '.join(only_400[:20])}{'...' if len(only_400) > 20 else ''}")
if only_410:
    print(f"  410-only: {', '.join(only_410[:20])}{'...' if len(only_410) > 20 else ''}")

print(f"\n  Scored by BOTH ({len(overlap)}): {', '.join(overlap) if overlap else 'none yet'}")

if not overlap:
    print(f"\n  ⚠ NO OVERLAPPING TICKERS — cannot compare. Wait for more 410 results.")
    raise SystemExit

print(f"\n  ★ All comparisons below use ONLY the {len(overlap)} overlapping tickers.")

ps_400 = ps_400[ps_400['ticker'].isin(overlap)].copy()
en_400 = en_400[en_400['ticker'].isin(overlap)].copy()
ps_410 = ps_410[ps_410['ticker'].isin(overlap)].copy()
en_410 = en_410[en_410['ticker'].isin(overlap)].copy()


# ==============================================================================
# HELPER: best per ticker (merge per-seed + ensemble, same as analyze scripts)
# ==============================================================================
def best_per_ticker(per_seed, ensemble):
    """Returns dict {ticker: (test_pnl, mode, source_label, opt_pnl, val_pnl, test_wr, test_n)}"""
    result = {}

    if len(per_seed) > 0:
        for (ticker, mode), grp in per_seed.groupby(['ticker', 'tpsl_mode']):
            best_seed = grp.loc[grp['opt_avg_pnl'].idxmax()]
            pnl = best_seed['val_b_avg_pnl']
            src = f"S{int(best_seed['seed'])}"
            if ticker not in result or pnl > result[ticker][0]:
                result[ticker] = (pnl, mode, src,
                                  best_seed['opt_avg_pnl'],
                                  best_seed['val_a_avg_pnl'],
                                  best_seed['val_b_win_rate'],
                                  int(best_seed['val_b_count']) if pd.notna(best_seed['val_b_count']) else 0)

    if len(ensemble) > 0:
        for _, r in ensemble.iterrows():
            ticker = r['ticker']
            pnl = r['val_b_avg_pnl']
            mode = r['tpsl_mode'] or 'static'
            if ticker not in result or pnl > result[ticker][0]:
                result[ticker] = (pnl, mode, 'ENS',
                                  r['opt_avg_pnl'],
                                  r['val_a_avg_pnl'],
                                  r['val_b_win_rate'],
                                  int(r['val_b_count']) if pd.notna(r['val_b_count']) else 0)
    return result


best_400 = best_per_ticker(ps_400, en_400)
best_410 = best_per_ticker(ps_410, en_410)


# ==============================================================================
# A. OVERALL VIABILITY — side-by-side mode summary
# ==============================================================================
print(f"\n\n{'═'*W}")
print(f"  A. OVERALL VIABILITY — SIDE-BY-SIDE MODE SUMMARY")
print(f"{'═'*W}")

all_modes = sorted(set(ps_400['tpsl_mode'].dropna().unique()) |
                   set(ps_410['tpsl_mode'].dropna().unique()))

print(f"\n  {'Mode':<15} │ {'── EXP 400 (alpha=2.0) ──────────────':} │ "
      f"{'── EXP 410 (alpha=1.0) ──────────────':} │ {'Winner':}")
print(f"  {'':15} │ {'OPT':>7} {'VAL':>7} {'TEST':>8} {'T+':>4} {'Deg':>5} │ "
      f"{'OPT':>7} {'VAL':>7} {'TEST':>8} {'T+':>4} {'Deg':>5} │")
print(f"  {'─'*15} │ {'─'*7} {'─'*7} {'─'*8} {'─'*4} {'─'*5} │ "
      f"{'─'*7} {'─'*7} {'─'*8} {'─'*4} {'─'*5} │ {'─'*8}")

for mode in all_modes:
    md4 = ps_400[ps_400['tpsl_mode'] == mode]
    md1 = ps_410[ps_410['tpsl_mode'] == mode]

    if len(md4) > 0:
        o4 = md4['opt_avg_pnl'].mean()
        v4 = md4['val_a_avg_pnl'].mean()
        t4 = md4['val_b_avg_pnl'].mean()
        tp4 = (md4['val_b_avg_pnl'] > 0).sum()
        d4 = 1 - t4 / o4 if o4 > 0 else 0
        s4 = f"{o4:>+6.2%} {v4:>+6.2%} {t4:>+7.2%} {tp4:>4} {d4:>+4.0%}"
    else:
        t4 = None
        s4 = f"{'—':>7} {'—':>7} {'—':>8} {'—':>4} {'—':>5}"

    if len(md1) > 0:
        o1 = md1['opt_avg_pnl'].mean()
        v1 = md1['val_a_avg_pnl'].mean()
        t1 = md1['val_b_avg_pnl'].mean()
        tp1 = (md1['val_b_avg_pnl'] > 0).sum()
        d1 = 1 - t1 / o1 if o1 > 0 else 0
        s1 = f"{o1:>+6.2%} {v1:>+6.2%} {t1:>+7.2%} {tp1:>4} {d1:>+4.0%}"
    else:
        t1 = None
        s1 = f"{'—':>7} {'—':>7} {'—':>8} {'—':>4} {'—':>5}"

    if t4 is not None and t1 is not None:
        winner = '400' if t4 > t1 else '410' if t1 > t4 else 'TIE'
    elif t4 is not None:
        winner = '400'
    elif t1 is not None:
        winner = '410'
    else:
        winner = '—'
    print(f"  {mode:<15} │ {s4} │ {s1} │ {winner:>8}")

# Grand totals
print(f"  {'─'*15} │ {'─'*7} {'─'*7} {'─'*8} {'─'*4} {'─'*5} │ "
      f"{'─'*7} {'─'*7} {'─'*8} {'─'*4} {'─'*5} │ {'─'*8}")
for label, ps in [('400', ps_400), ('410', ps_410)]:
    if len(ps) > 0:
        o = ps['opt_avg_pnl'].mean()
        v = ps['val_a_avg_pnl'].mean()
        t = ps['val_b_avg_pnl'].mean()
        d = 1 - t / o if o > 0 else 0
        tp = (ps['val_b_avg_pnl'] > 0).sum()
        print(f"  {'ALL ('+label+')':<15} │ {o:>+6.2%} {v:>+6.2%} {t:>+7.2%} {tp:>4} {d:>+4.0%} │")

grand_400 = ps_400['val_b_avg_pnl'].mean() if len(ps_400) > 0 else None
grand_410 = ps_410['val_b_avg_pnl'].mean() if len(ps_410) > 0 else None
if grand_400 is not None and grand_410 is not None:
    delta = grand_410 - grand_400
    pct = delta / abs(grand_400) * 100 if grand_400 != 0 else 0
    print(f"\n  ★ Grand mean TEST:  400={grand_400:+.3%}  410={grand_410:+.3%}  "
          f"Δ={delta:+.3%} ({pct:+.0f}%)")


# ==============================================================================
# B. PER-TICKER HEAD-TO-HEAD
# ==============================================================================
print(f"\n\n{'═'*W}")
print(f"  B. PER-TICKER HEAD-TO-HEAD — Best TEST PnL (per-seed or ensemble)")
print(f"{'═'*W}")

print(f"\n  {'Ticker':<7} │ {'── 400 (alpha=2.0) ──────────────────':} │ "
      f"{'── 410 (alpha=1.0) ──────────────────':} │ {'Winner':>6} {'Δ':>8}")
print(f"  {'':7} │ {'Mode':<14} {'Src':>3} {'TEST':>7} {'WR':>4} {'N':>3} │ "
      f"{'Mode':<14} {'Src':>3} {'TEST':>7} {'WR':>4} {'N':>3} │")
print(f"  {'─'*7} │ {'─'*14} {'─'*3} {'─'*7} {'─'*4} {'─'*3} │ "
      f"{'─'*14} {'─'*3} {'─'*7} {'─'*4} {'─'*3} │ {'─'*6} {'─'*8}")

wins_400 = 0
wins_410 = 0
ties = 0

for tk in overlap:
    b4 = best_400.get(tk)
    b1 = best_410.get(tk)

    s4 = f"{b4[1]:<14} {b4[2]:>3} {b4[0]:>+6.2%} {b4[5]:>3.0%} {b4[6]:>3}" if b4 else f"{'—':<14} {'':>3} {'—':>7} {'—':>4} {'—':>3}"
    s1 = f"{b1[1]:<14} {b1[2]:>3} {b1[0]:>+6.2%} {b1[5]:>3.0%} {b1[6]:>3}" if b1 else f"{'—':<14} {'':>3} {'—':>7} {'—':>4} {'—':>3}"
    v4 = b4[0] if b4 else -999
    v1 = b1[0] if b1 else -999

    if v4 > v1:
        winner = '400'
        wins_400 += 1
    elif v1 > v4:
        winner = '410'
        wins_410 += 1
    else:
        winner = 'TIE'
        ties += 1
    delta_str = f"{v1 - v4:+.2%}" if v4 != -999 and v1 != -999 else '—'

    print(f"  {tk:<7} │ {s4} │ {s1} │ {winner:>6} {delta_str:>8}")

print(f"\n  Score: 400 wins {wins_400} — 410 wins {wins_410} — TIE/N/A {ties}")

# Deployability comparison
dep_400 = sum(1 for t, v in best_400.items() if t in overlap and v[0] >= DEPLOY_PNL_THRESHOLD and v[6] >= MIN_TEST_TRADES)
dep_410 = sum(1 for t, v in best_410.items() if t in overlap and v[0] >= DEPLOY_PNL_THRESHOLD and v[6] >= MIN_TEST_TRADES)
print(f"  Deployable (≥{DEPLOY_PNL_THRESHOLD:.1%} + ≥{MIN_TEST_TRADES} trades):  "
      f"400={dep_400}  410={dep_410}")


# ==============================================================================
# C. DEGRADATION CASCADE — OPT → VAL → TEST
# ==============================================================================
print(f"\n\n{'═'*W}")
print(f"  C. DEGRADATION — OPT → VAL → TEST RETENTION")
print(f"{'═'*W}")

def cascade_stats(per_seed, label):
    if len(per_seed) == 0:
        print(f"\n  {label}: no data")
        return None
    pos = per_seed[per_seed['opt_avg_pnl'] > 0].copy()
    if len(pos) == 0:
        print(f"\n  {label}: no positive OPT rows")
        return None

    pos['val_ratio'] = pos['val_a_avg_pnl'] / pos['opt_avg_pnl']
    pos['test_ratio'] = pos['val_b_avg_pnl'] / pos['opt_avg_pnl']

    o = pos['opt_avg_pnl'].mean()
    v = pos['val_a_avg_pnl'].mean()
    t = pos['val_b_avg_pnl'].mean()
    val_ret = v / o if o != 0 else 0
    test_ret = t / o if o != 0 else 0

    n_val_pos = (pos['val_a_avg_pnl'] > 0).sum()
    n_test_pos = (pos['val_b_avg_pnl'] > 0).sum()
    n_flipped = ((pos['val_a_avg_pnl'] > 0) & (pos['val_b_avg_pnl'] <= 0)).sum()
    flip_rate = n_flipped / n_val_pos if n_val_pos > 0 else 0

    med_test_ratio = pos['test_ratio'].clip(-2, 3).median()

    print(f"\n  ── {label} ──")
    print(f"  OPT mean:   {o:>+.3%}")
    print(f"  VAL mean:   {v:>+.3%}  (retention: {val_ret:.0%})")
    print(f"  TEST mean:  {t:>+.3%}  (retention: {test_ret:.0%})")
    print(f"  Median TEST/OPT ratio: {med_test_ratio:.2f}")
    print(f"  VAL>0: {n_val_pos}  TEST>0: {n_test_pos}  "
          f"Flipped (VAL>0→TEST≤0): {n_flipped} ({flip_rate:.0%})")

    return {'opt': o, 'val': v, 'test': t, 'val_ret': val_ret, 'test_ret': test_ret,
            'med_ratio': med_test_ratio, 'flip_rate': flip_rate}

c400 = cascade_stats(ps_400, "EXP 400 (alpha=2.0)")
c410 = cascade_stats(ps_410, "EXP 410 (alpha=1.0)")

if c400 and c410:
    print(f"\n  ★ Retention comparison:")
    print(f"    OPT→TEST retention:  400={c400['test_ret']:.0%}  410={c410['test_ret']:.0%}  "
          f"→ {'410 retains more' if c410['test_ret'] > c400['test_ret'] else '400 retains more'}")
    print(f"    Flip rate:           400={c400['flip_rate']:.0%}  410={c410['flip_rate']:.0%}  "
          f"→ {'410 flips less' if c410['flip_rate'] < c400['flip_rate'] else '400 flips less'}")


# ==============================================================================
# D. MODE COMPARISON — which modes benefit from alpha change
# ==============================================================================
print(f"\n\n{'═'*W}")
print(f"  D. MODE COMPARISON — WHICH MODES BENEFIT FROM alpha=1.0?")
print(f"{'═'*W}")

print(f"\n  {'Mode':<15} │ {'400 TEST':>9} {'Med':>7} {'%Pos':>5} │ "
      f"{'410 TEST':>9} {'Med':>7} {'%Pos':>5} │ {'Δ Mean':>8} {'Winner':>6}")
print(f"  {'─'*15} │ {'─'*9} {'─'*7} {'─'*5} │ "
      f"{'─'*9} {'─'*7} {'─'*5} │ {'─'*8} {'─'*6}")

for mode in all_modes:
    md4 = ps_400[ps_400['tpsl_mode'] == mode]
    md1 = ps_410[ps_410['tpsl_mode'] == mode]

    if len(md4) > 0:
        mean4 = md4['val_b_avg_pnl'].mean()
        med4 = md4['val_b_avg_pnl'].median()
        pp4 = (md4['val_b_avg_pnl'] > 0).mean() * 100
        s4 = f"{mean4:>+8.2%} {med4:>+6.2%} {pp4:>4.0f}%"
    else:
        mean4 = None
        s4 = f"{'—':>9} {'—':>7} {'—':>5}"

    if len(md1) > 0:
        mean1 = md1['val_b_avg_pnl'].mean()
        med1 = md1['val_b_avg_pnl'].median()
        pp1 = (md1['val_b_avg_pnl'] > 0).mean() * 100
        s1 = f"{mean1:>+8.2%} {med1:>+6.2%} {pp1:>4.0f}%"
    else:
        mean1 = None
        s1 = f"{'—':>9} {'—':>7} {'—':>5}"

    if mean4 is not None and mean1 is not None:
        delta = mean1 - mean4
        winner = '410' if delta > 0 else '400' if delta < 0 else 'TIE'
        print(f"  {mode:<15} │ {s4} │ {s1} │ {delta:>+7.2%} {winner:>6}")
    else:
        print(f"  {mode:<15} │ {s4} │ {s1} │ {'—':>8} {'—':>6}")


# ==============================================================================
# E. SEED STABILITY COMPARISON
# ==============================================================================
print(f"\n\n{'═'*W}")
print(f"  E. SEED STABILITY — CROSS-SEED CONSISTENCY ON OVERLAPPING TICKERS")
print(f"{'═'*W}")

if overlap:
    print(f"\n  {'Ticker':<7} {'Mode':<12} │ {'── 400 ────────────────':} │ "
          f"{'── 410 ────────────────':} │ {'More stable':}")
    print(f"  {'':7} {'':12} │ {'Seeds':>5} {'T+':>3} {'Agree':>5} {'Spread':>6} │ "
          f"{'Seeds':>5} {'T+':>3} {'Agree':>5} {'Spread':>6} │")
    print(f"  {'─'*7} {'─'*12} │ {'─'*5} {'─'*3} {'─'*5} {'─'*6} │ "
          f"{'─'*5} {'─'*3} {'─'*5} {'─'*6} │ {'─'*11}")

    stable_400 = 0
    stable_410 = 0
    for tk in overlap:
        for mode in all_modes:
            g4 = ps_400[(ps_400['ticker'] == tk) & (ps_400['tpsl_mode'] == mode)]
            g1 = ps_410[(ps_410['ticker'] == tk) & (ps_410['tpsl_mode'] == mode)]
            if len(g4) < 3 and len(g1) < 3:
                continue

            if len(g4) >= 3:
                n4 = len(g4)
                tp4 = (g4['val_b_avg_pnl'] > 0).sum()
                ag4 = tp4 / n4
                sp4 = g4['val_b_avg_pnl'].std()
                s4 = f"{n4:>5} {tp4:>3} {ag4:>4.0%} {sp4:>5.2%}"
            else:
                ag4 = 0
                sp4 = 1
                s4 = f"{'—':>5} {'—':>3} {'—':>5} {'—':>6}"

            if len(g1) >= 3:
                n1 = len(g1)
                tp1 = (g1['val_b_avg_pnl'] > 0).sum()
                ag1 = tp1 / n1
                sp1 = g1['val_b_avg_pnl'].std()
                s1 = f"{n1:>5} {tp1:>3} {ag1:>4.0%} {sp1:>5.2%}"
            else:
                ag1 = 0
                sp1 = 1
                s1 = f"{'—':>5} {'—':>3} {'—':>5} {'—':>6}"

            if ag4 > ag1 or (ag4 == ag1 and sp4 < sp1):
                winner = '400'
                stable_400 += 1
            elif ag1 > ag4 or (ag1 == ag4 and sp1 < sp4):
                winner = '410'
                stable_410 += 1
            else:
                winner = 'TIE'

            print(f"  {tk:<7} {mode:<12} │ {s4} │ {s1} │ {winner:>11}")

    print(f"\n  Stability score: 400={stable_400}  410={stable_410}")
else:
    print(f"\n  No overlapping tickers for stability comparison.")


# ==============================================================================
# F. CHARTS
# ==============================================================================
print(f"\n  Generating charts...")

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Chart 1: Scatter — 400 TEST (x) vs 410 TEST (y) per ticker
ax = axes[0, 0]
common_tickers = [t for t in overlap if t in best_400 and t in best_410]
if common_tickers:
    x_vals = [best_400[t][0] * 100 for t in common_tickers]
    y_vals = [best_410[t][0] * 100 for t in common_tickers]
    colors_sc = ['#2ca02c' if y > x else '#d62728' for x, y in zip(x_vals, y_vals)]
    ax.scatter(x_vals, y_vals, c=colors_sc, alpha=0.6, s=40, edgecolors='white', linewidth=0.5)
    lim = max(max(abs(v) for v in x_vals + y_vals) + 1, 5)
    ax.plot([-lim, lim], [-lim, lim], 'k--', alpha=0.3, linewidth=1)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    for i, tk in enumerate(common_tickers):
        if abs(x_vals[i]) > 2 or abs(y_vals[i]) > 2:
            ax.annotate(tk, (x_vals[i], y_vals[i]), fontsize=6, alpha=0.7,
                        xytext=(3, 3), textcoords='offset points')
ax.set_xlabel('Exp 400 TEST PnL (%)')
ax.set_ylabel('Exp 410 TEST PnL (%)')
ax.set_title('400 vs 410 TEST PnL per ticker\n(green = 410 wins, red = 400 wins)', fontweight='bold')
ax.grid(True, alpha=0.3)

# Chart 2: Bar chart — per-ticker Δ(410 - 400) TEST PnL
ax = axes[0, 1]
if common_tickers and len(common_tickers) > 0:
    deltas = [(t, best_410[t][0] - best_400[t][0]) for t in common_tickers]
    deltas.sort(key=lambda x: x[1], reverse=True)
    tickers_d = [d[0] for d in deltas]
    vals_d = [d[1] * 100 for d in deltas]
    colors_d = ['#2ca02c' if v > 0 else '#d62728' for v in vals_d]
    ax.barh(range(len(tickers_d)), vals_d, color=colors_d, height=0.7)
    ax.set_yticks(range(len(tickers_d)))
    ax.set_yticklabels(tickers_d, fontsize=7)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.invert_yaxis()
ax.set_xlabel('Δ TEST PnL (%) [410 − 400]')
ax.set_title('Per-Ticker Advantage\n(green = 410 better, red = 400 better)', fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

# Chart 3: Grouped bar — OPT→VAL→TEST retention
ax = axes[1, 0]
if c400 and c410:
    labels_c = ['OPT', 'VAL', 'TEST']
    vals_400 = [c400['opt'] * 100, c400['val'] * 100, c400['test'] * 100]
    vals_410 = [c410['opt'] * 100, c410['val'] * 100, c410['test'] * 100]
    x_pos = np.arange(len(labels_c))
    width = 0.35
    ax.bar(x_pos - width/2, vals_400, width, label='400 (alpha=2.0)', color='#1f77b4', alpha=0.8)
    ax.bar(x_pos + width/2, vals_410, width, label='410 (alpha=1.0)', color='#ff7f0e', alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels_c, fontsize=11)
    ax.axhline(0, color='black', linewidth=0.5)
    for i, (v4, v1) in enumerate(zip(vals_400, vals_410)):
        ax.text(i - width/2, v4, f'{v4:+.2f}%', ha='center', va='bottom', fontsize=8)
        ax.text(i + width/2, v1, f'{v1:+.2f}%', ha='center', va='bottom', fontsize=8)
    ax.legend(fontsize=9)
ax.set_ylabel('Mean PnL (%)')
ax.set_title('OPT → VAL → TEST Pipeline', fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Chart 4: Box plot — TEST PnL distributions by mode, 400 vs 410
ax = axes[1, 1]
box_data = []
box_labels = []
box_colors = []
for mode in all_modes:
    md4 = ps_400[ps_400['tpsl_mode'] == mode]['val_b_avg_pnl'].clip(-0.10, 0.15) * 100
    md1 = ps_410[ps_410['tpsl_mode'] == mode]['val_b_avg_pnl'].clip(-0.10, 0.15) * 100
    if len(md4) > 0:
        box_data.append(md4.values)
        box_labels.append(f'{mode}\n400')
        box_colors.append('#1f77b4')
    if len(md1) > 0:
        box_data.append(md1.values)
        box_labels.append(f'{mode}\n410')
        box_colors.append('#ff7f0e')

if box_data:
    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True, showfliers=False)
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axhline(DEPLOY_PNL_THRESHOLD * 100, color='red', linewidth=1, linestyle='--', alpha=0.5)
    ax.tick_params(axis='x', labelsize=7)
ax.set_ylabel('TEST Avg PnL (%)')
ax.set_title('TEST PnL Distribution by Mode', fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('Exp 400 vs 410 — SHORT/Dump alpha_pos Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/tmp/compare_400_vs_410.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"  Charts saved: /tmp/compare_400_vs_410.png")


# ==============================================================================
# SUMMARY
# ==============================================================================
print(f"\n\n{'═'*W}")
print(f"  SUMMARY — EXP 400 (alpha=2.0) vs EXP 410 (alpha=1.0)")
print(f"{'═'*W}")

print(f"""
  Head-to-head TEST PnL:     400 wins {wins_400} — 410 wins {wins_410} — ties {ties}
  Grand mean TEST:           400={grand_400:+.3%}  410={grand_410:+.3%}""" if grand_400 is not None and grand_410 is not None else "")

if c400 and c410:
    print(f"  OPT→TEST retention:       400={c400['test_ret']:.0%}  410={c410['test_ret']:.0%}")
    print(f"  Flip rate (VAL>0→TEST≤0): 400={c400['flip_rate']:.0%}  410={c410['flip_rate']:.0%}")

print(f"  Deployable tickers:       400={dep_400}  410={dep_410}")

if grand_400 is not None and grand_410 is not None:
    overall = '410 (alpha=1.0)' if grand_410 > grand_400 else '400 (alpha=2.0)' if grand_400 > grand_410 else 'TIE'
    print(f"\n  ★ VERDICT: {overall} has higher grand mean TEST PnL")
    if wins_410 != wins_400:
        ticker_winner = '410' if wins_410 > wins_400 else '400'
        print(f"    Per-ticker winner: {ticker_winner} ({max(wins_400, wins_410)} vs {min(wins_400, wins_410)})")

print(f"\n{'═'*W}")
