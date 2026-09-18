"""
compare_500_vs_185_degradation.py — Exp 500 vs 185 only, VAL→TEST degradation
===============================================================================
Focused comparison: only tickers optimized for 500, only fold 185 for spike.
Key question: how much does each model degrade from validation to test?
"""

from google.cloud import bigquery
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass

bq = bigquery.Client()

W = 120
BQ_TABLE = f"{GCP_PROJECT}.scoring_logs.wf_optimization_results"
SWING_MV = 'robust_wf_v12_swing'
SPIKE_MV = 'robust_wf_v12'
TICKER_RENAME = {'MIGI': 'BGDE', 'BITF': 'KEEL'}
EXCLUDE = {'TLT', 'VIX', 'SMLR', 'SPY', 'IWM'}

# ── Load 500 ensembles ──
q_sw = f'''
SELECT ticker, tpsl_mode, experiment,
       prob_thresh, tp_pct, sl_pct,
       opt_avg_pnl, opt_win_rate, opt_count,
       val_a_avg_pnl, val_a_win_rate, val_a_count,
       val_b_avg_pnl, val_b_win_rate, val_b_count,
       survived_a, survived_b, n_seeds_avg
FROM `{BQ_TABLE}`
WHERE model_version = '{SWING_MV}' AND is_ensemble = TRUE
'''
sw = bq.query(q_sw).to_dataframe()
sw['ticker'] = sw['ticker'].replace(TICKER_RENAME)
sw = sw[~sw['ticker'].isin(EXCLUDE)]

# ── Load 185 ensembles only ──
q_sp = f'''
SELECT ticker, tpsl_mode, experiment,
       prob_thresh, tp_pct, sl_pct,
       opt_avg_pnl, opt_win_rate, opt_count,
       val_a_avg_pnl, val_a_win_rate, val_a_count,
       val_b_avg_pnl, val_b_win_rate, val_b_count,
       survived_a, survived_b, n_seeds_avg
FROM `{BQ_TABLE}`
WHERE model_version = '{SPIKE_MV}' AND is_ensemble = TRUE
  AND STARTS_WITH(experiment, '185')
'''
sp = bq.query(q_sp).to_dataframe()
sp['ticker'] = sp['ticker'].replace(TICKER_RENAME)
sp = sp[~sp['ticker'].isin(EXCLUDE)]

# ── Scope: only tickers that 500 has been optimized for ──
sw_tickers = sorted(sw['ticker'].unique())
sp = sp[sp['ticker'].isin(sw_tickers)]

print(f"{'═'*W}")
print(f"  EXP 500 vs EXP 185 — VAL → TEST DEGRADATION")
print(f"  Scope: {len(sw_tickers)} tickers optimized for Exp 500")
print(f"{'═'*W}")
print(f"  Swing 500: {len(sw)} ensemble rows, {sw['ticker'].nunique()} tickers")
print(f"  Spike 185: {len(sp)} ensemble rows, {sp['ticker'].nunique()} tickers in scope")

# ── Best survived ensemble per ticker ──
def best_per_ticker(df):
    surv = df[df['survived_a'] == True].copy()
    if surv.empty:
        return pd.DataFrame()
    return surv.loc[surv.groupby('ticker')['val_a_avg_pnl'].idxmax()].set_index('ticker')

sw_best = best_per_ticker(sw)
sp_best = best_per_ticker(sp)

overlap = sorted(set(sw_best.index) & set(sp_best.index))
sw_only = sorted(set(sw_best.index) - set(sp_best.index))
sp_only = sorted(set(sp_best.index) - set(sw_best.index))

print(f"  Survived ensembles: Swing {len(sw_best)}, Spike {len(sp_best)}")
print(f"  Both survived: {len(overlap)} | Swing-only: {len(sw_only)} | Spike-only: {len(sp_only)}")


# ══════════════════════════════════════════════════════════════════════════════
# A. PER-TICKER: OPT → VAL → TEST pipeline
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'═'*W}")
print(f"  A. PER-TICKER OPT → VAL → TEST  (best survived ensemble)")
print(f"     Degradation = (TEST - VAL) as absolute %, and TEST/VAL ratio")
print(f"{'═'*W}")

header = (f"  {'Ticker':<7} │"
          f" {'── SWING 500 ─────────────────────────────────':} │"
          f" {'── SPIKE 185 ─────────────────────────────────':}")
sub    = (f"  {'':7} │"
          f" {'Mode':<14} {'OPT':>6} {'VAL':>6} {'TEST':>6} {'Δ V→T':>6} {'Ratio':>5} │"
          f" {'Mode':<14} {'OPT':>6} {'VAL':>6} {'TEST':>6} {'Δ V→T':>6} {'Ratio':>5}")
sep    = f"  {'─'*7} │ {'─'*54} │ {'─'*54}"
print(f"\n{header}\n{sub}\n{sep}")

sw_deltas = []
sp_deltas = []

all_tickers = sorted(set(sw_best.index) | set(sp_best.index))
for tk in all_tickers:
    # Swing
    if tk in sw_best.index:
        s = sw_best.loc[tk]
        sw_opt = s['opt_avg_pnl']
        sw_val = s['val_a_avg_pnl']
        sw_tst = s['val_b_avg_pnl'] if s['val_b_count'] > 0 else None
        sw_mode = s['tpsl_mode']
        sw_delta = (sw_tst - sw_val) if sw_tst is not None else None
        sw_ratio = (sw_tst / sw_val) if sw_tst is not None and sw_val != 0 else None
        if sw_delta is not None:
            sw_deltas.append({'ticker': tk, 'val': sw_val, 'test': sw_tst,
                              'delta': sw_delta, 'ratio': sw_ratio,
                              'opt': sw_opt, 'mode': sw_mode,
                              'val_n': int(s['val_a_count']), 'test_n': int(s['val_b_count'])})
        sw_str = (f" {sw_mode:<14} {sw_opt:>+5.1%} {sw_val:>+5.1%}"
                  f" {sw_tst:>+5.1%} {sw_delta:>+5.1%} {sw_ratio:>5.2f}" if sw_tst is not None
                  else f" {sw_mode:<14} {sw_opt:>+5.1%} {sw_val:>+5.1%} {'—':>6} {'—':>6} {'—':>5}")
    else:
        sw_str = f" {'—':<14} {'—':>6} {'—':>6} {'—':>6} {'—':>6} {'—':>5}"

    # Spike
    if tk in sp_best.index:
        s = sp_best.loc[tk]
        sp_opt = s['opt_avg_pnl']
        sp_val = s['val_a_avg_pnl']
        sp_tst = s['val_b_avg_pnl'] if s['val_b_count'] > 0 else None
        sp_mode = s['tpsl_mode']
        sp_delta = (sp_tst - sp_val) if sp_tst is not None else None
        sp_ratio = (sp_tst / sp_val) if sp_tst is not None and sp_val != 0 else None
        if sp_delta is not None:
            sp_deltas.append({'ticker': tk, 'val': sp_val, 'test': sp_tst,
                              'delta': sp_delta, 'ratio': sp_ratio,
                              'opt': sp_opt, 'mode': sp_mode,
                              'val_n': int(s['val_a_count']), 'test_n': int(s['val_b_count'])})
        sp_str = (f" {sp_mode:<14} {sp_opt:>+5.1%} {sp_val:>+5.1%}"
                  f" {sp_tst:>+5.1%} {sp_delta:>+5.1%} {sp_ratio:>5.2f}" if sp_tst is not None
                  else f" {sp_mode:<14} {sp_opt:>+5.1%} {sp_val:>+5.1%} {'—':>6} {'—':>6} {'—':>5}")
    else:
        sp_str = f" {'—':<14} {'—':>6} {'—':>6} {'—':>6} {'—':>6} {'—':>5}"

    print(f"  {tk:<7} │{sw_str} │{sp_str}")


# ══════════════════════════════════════════════════════════════════════════════
# B. AGGREGATE DEGRADATION
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'═'*W}")
print(f"  B. AGGREGATE DEGRADATION — VAL vs TEST")
print(f"{'═'*W}")

def agg_stats(deltas, label):
    if not deltas:
        print(f"\n  {label}: no data")
        return
    df = pd.DataFrame(deltas)
    n = len(df)
    n_pos_val = (df['val'] > 0).sum()
    n_pos_test = (df['test'] > 0).sum()
    n_degraded = (df['test'] < df['val']).sum()
    n_flipped = ((df['val'] > 0) & (df['test'] <= 0)).sum()

    val_mean = df['val'].mean()
    test_mean = df['test'].mean()
    delta_mean = df['delta'].mean()

    val_total_pnl = (df['val'] * df['val_n']).sum()
    test_total_pnl = (df['test'] * df['test_n']).sum()
    val_total_n = df['val_n'].sum()
    test_total_n = df['test_n'].sum()
    val_wavg = val_total_pnl / val_total_n if val_total_n > 0 else 0
    test_wavg = test_total_pnl / test_total_n if test_total_n > 0 else 0

    # Median ratio (exclude extreme outliers)
    valid_ratios = df[df['ratio'].notna() & df['ratio'].between(-10, 10)]['ratio']
    med_ratio = valid_ratios.median() if len(valid_ratios) > 0 else None

    print(f"\n  ── {label} ──")
    print(f"  Tickers with both VAL & TEST:  {n}")
    print(f"  VAL positive / TEST positive:  {n_pos_val} / {n_pos_test}")
    print(f"  Degraded (TEST < VAL):         {n_degraded}/{n} ({n_degraded/n:.0%})")
    print(f"  Flipped (VAL>0 → TEST≤0):      {n_flipped}/{n_pos_val} ({n_flipped/n_pos_val:.0%})" if n_pos_val > 0 else "")
    print(f"  ")
    print(f"  Mean VAL PnL (across tickers): {val_mean:>+.2%}")
    print(f"  Mean TEST PnL:                 {test_mean:>+.2%}")
    print(f"  Mean degradation (T - V):      {delta_mean:>+.2%}")
    print(f"  Median TEST/VAL ratio:         {med_ratio:.2f}" if med_ratio is not None else "")
    print(f"  ")
    print(f"  Trade-weighted VAL PnL:        {val_wavg:>+.2%}  ({int(val_total_n)} trades)")
    print(f"  Trade-weighted TEST PnL:       {test_wavg:>+.2%}  ({int(test_total_n)} trades)")
    print(f"  Trade-weighted degradation:    {test_wavg - val_wavg:>+.2%}")

    return df

sw_df = agg_stats(sw_deltas, "SWING 500 (75 min)")
sp_df = agg_stats(sp_deltas, "SPIKE 185 (6.75 hr)")


# ══════════════════════════════════════════════════════════════════════════════
# C. OVERLAPPING TICKERS — direct comparison
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'═'*W}")
print(f"  C. SAME-TICKER DEGRADATION — only tickers with data in BOTH models")
print(f"{'═'*W}")

if sw_df is not None and sp_df is not None:
    sw_d = pd.DataFrame(sw_deltas).set_index('ticker')
    sp_d = pd.DataFrame(sp_deltas).set_index('ticker')
    common = sorted(set(sw_d.index) & set(sp_d.index))

    if common:
        print(f"\n  {len(common)} tickers with VAL+TEST data in both models\n")
        print(f"  {'Ticker':<7} │ {'─── SWING ──────────────':} │ {'─── SPIKE ──────────────':} │ {'Less degraded':}")
        print(f"  {'':7} │ {'VAL':>6} {'TEST':>6} {'Δ':>7} │ {'VAL':>6} {'TEST':>6} {'Δ':>7} │")
        print(f"  {'─'*7} │ {'─'*6} {'─'*6} {'─'*7} │ {'─'*6} {'─'*6} {'─'*7} │ {'─'*14}")

        sw_better = 0
        sp_better = 0
        for tk in common:
            s = sw_d.loc[tk]
            p = sp_d.loc[tk]
            winner = 'SWING' if s['delta'] > p['delta'] else 'SPIKE'
            if s['delta'] > p['delta']:
                sw_better += 1
            else:
                sp_better += 1
            print(f"  {tk:<7} │ {s['val']:>+5.1%} {s['test']:>+5.1%} {s['delta']:>+6.1%}"
                  f" │ {p['val']:>+5.1%} {p['test']:>+5.1%} {p['delta']:>+6.1%}"
                  f" │ {winner}")

        sw_common = sw_d.loc[common]
        sp_common = sp_d.loc[common]
        print(f"\n  Degradation score: SWING {sw_better} — SPIKE {sp_better}")
        print(f"  Mean Δ on common tickers: SWING {sw_common['delta'].mean():+.2%}  "
              f"SPIKE {sp_common['delta'].mean():+.2%}")
        print(f"  Median Δ on common tickers: SWING {sw_common['delta'].median():+.2%}  "
              f"SPIKE {sp_common['delta'].median():+.2%}")


# ══════════════════════════════════════════════════════════════════════════════
# D. MODE BREAKDOWN — degradation by tpsl_mode
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'═'*W}")
print(f"  D. MODE BREAKDOWN — degradation by optimization mode")
print(f"{'═'*W}")

if sw_deltas:
    sw_mode_df = pd.DataFrame(sw_deltas)
    print(f"\n  SWING 500:")
    print(f"  {'Mode':<18} {'N':>3} {'Mean VAL':>9} {'Mean TEST':>10} {'Mean Δ':>8} {'#Flipped':>8}")
    print(f"  {'─'*18} {'─'*3} {'─'*9} {'─'*10} {'─'*8} {'─'*8}")
    for mode, grp in sw_mode_df.groupby('mode'):
        n_flip = ((grp['val'] > 0) & (grp['test'] <= 0)).sum()
        print(f"  {mode:<18} {len(grp):>3} {grp['val'].mean():>+8.2%} {grp['test'].mean():>+9.2%}"
              f" {grp['delta'].mean():>+7.2%} {n_flip:>8}")

if sp_deltas:
    sp_mode_df = pd.DataFrame(sp_deltas)
    print(f"\n  SPIKE 185:")
    print(f"  {'Mode':<18} {'N':>3} {'Mean VAL':>9} {'Mean TEST':>10} {'Mean Δ':>8} {'#Flipped':>8}")
    print(f"  {'─'*18} {'─'*3} {'─'*9} {'─'*10} {'─'*8} {'─'*8}")
    for mode, grp in sp_mode_df.groupby('mode'):
        n_flip = ((grp['val'] > 0) & (grp['test'] <= 0)).sum()
        print(f"  {mode:<18} {len(grp):>3} {grp['val'].mean():>+8.2%} {grp['test'].mean():>+9.2%}"
              f" {grp['delta'].mean():>+7.2%} {n_flip:>8}")


# ══════════════════════════════════════════════════════════════════════════════
# E. OPT → VAL → TEST cascade
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'═'*W}")
print(f"  E. FULL CASCADE — OPT → VAL → TEST (aggregate)")
print(f"{'═'*W}")

for label, deltas in [("SWING 500", sw_deltas), ("SPIKE 185", sp_deltas)]:
    if not deltas:
        continue
    df = pd.DataFrame(deltas)
    opt_mean = df['opt'].mean()
    val_mean = df['val'].mean()
    test_mean = df['test'].mean()
    opt_to_val = val_mean - opt_mean
    val_to_test = test_mean - val_mean
    opt_to_test = test_mean - opt_mean

    # Retention rates
    val_retention = val_mean / opt_mean if opt_mean != 0 else 0
    test_retention = test_mean / opt_mean if opt_mean != 0 else 0

    print(f"\n  {label}:")
    print(f"    OPT mean:     {opt_mean:>+.2%}")
    print(f"    VAL mean:     {val_mean:>+.2%}  (Δ from OPT: {opt_to_val:>+.2%},"
          f"  retention: {val_retention:.0%})")
    print(f"    TEST mean:    {test_mean:>+.2%}  (Δ from VAL: {val_to_test:>+.2%},"
          f"  retention from OPT: {test_retention:.0%})")
    print(f"    Total decay:  {opt_to_test:>+.2%}  (OPT → TEST)")


# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'═'*W}")
print(f"  SUMMARY")
print(f"{'═'*W}")

if sw_deltas and sp_deltas:
    sw_agg = pd.DataFrame(sw_deltas)
    sp_agg = pd.DataFrame(sp_deltas)
    print(f"""
  Swing 500:  VAL {sw_agg['val'].mean():+.2%} → TEST {sw_agg['test'].mean():+.2%}  (Δ {sw_agg['delta'].mean():+.2%})
  Spike 185:  VAL {sp_agg['val'].mean():+.2%} → TEST {sp_agg['test'].mean():+.2%}  (Δ {sp_agg['delta'].mean():+.2%})

  Swing flipped (VAL>0 → TEST≤0): {((sw_agg['val']>0) & (sw_agg['test']<=0)).sum()}/{(sw_agg['val']>0).sum()}
  Spike flipped (VAL>0 → TEST≤0): {((sp_agg['val']>0) & (sp_agg['test']<=0)).sum()}/{(sp_agg['val']>0).sum()}

  Verdict: {'SWING degrades MORE' if sw_agg['delta'].mean() < sp_agg['delta'].mean() else 'SPIKE degrades MORE'}
""")

print(f"{'═'*W}")
