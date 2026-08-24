# %% ── Fold Degradation Analysis — SHORT Models 400-405 / 410-415 ──────────────
#
# Answers:
#   1. How does ticker survival degrade fold-to-fold?
#   2. Do early folds' configs still perform on TEST (Apr-Jul 2026)?
#   3. Which fold is the best source for deployment configs?
#   4. Per-ticker heatmap: which tickers are robust across folds?
#   5. Fold agreement gate: optimal threshold (4/6? 3/6? 2/6?)

import pandas as pd
import numpy as np
from google.cloud import bigquery

BQ_TABLE = "ys-time-series.scoring_logs.wf_optimization_results"

FOLD_400 = [400, 401, 402, 403, 404, 405]
FOLD_410 = [410, 411, 412, 413, 414, 415]
ALL_FOLDS = FOLD_400 + FOLD_410

MODEL_VERSIONS = [f"robust_wf_v12_short_{f}" for f in ALL_FOLDS]
MV_STR = ", ".join(f"'{mv}'" for mv in MODEL_VERSIONS)

CUTOFF_LABELS = {
    400: 'Aug25', 401: 'Sep25', 402: 'Oct25', 403: 'Nov25', 404: 'Dec25', 405: 'Jan26',
    410: 'Aug25', 411: 'Sep25', 412: 'Oct25', 413: 'Nov25', 414: 'Dec25', 415: 'Jan26',
}

EXCLUDE_MODES = ['limit_entry']

# %% ── Pull all fold data from BQ ─────────────────────────────────────────────

client = bigquery.Client()

query = f"""
SELECT
    model_version,
    experiment,
    ticker,
    tpsl_mode,
    seed,
    is_ensemble,
    prob_thresh,
    tp_pct,
    sl_pct,
    opt_avg_pnl,
    opt_win_rate,
    opt_count,
    val_a_avg_pnl,
    val_a_win_rate,
    val_a_count,
    survived_a,
    val_b_avg_pnl,
    val_b_win_rate,
    val_b_count,
    n_seeds_avg,
    grace_minutes,
    dip_pct,
    max_wait_minutes
FROM `{BQ_TABLE}`
WHERE model_version IN ({MV_STR})
"""

df_all = client.query(query).to_dataframe()
print(f"Loaded {len(df_all):,} rows from BQ")

# Parse fold number from model_version
df_all['fold'] = df_all['model_version'].str.extract(r'_(\d+)$').astype(int)
df_all['alpha'] = df_all['fold'].apply(lambda f: 2.0 if f < 410 else 1.0)
df_all['family'] = df_all['fold'].apply(lambda f: '400' if f < 410 else '410')
df_all['cutoff'] = df_all['fold'].map(CUTOFF_LABELS)

# Exclude limit_entry
df_all = df_all[~df_all['tpsl_mode'].isin(EXCLUDE_MODES)].copy()
print(f"After excluding {EXCLUDE_MODES}: {len(df_all):,} rows")

# Split ensemble vs per-seed
df_ens = df_all[df_all['is_ensemble'] == True].copy()
df_seed = df_all[df_all['is_ensemble'] == False].copy()
print(f"Ensemble rows: {len(df_ens):,}  |  Per-seed rows: {len(df_seed):,}")

# %% ── Section 1: Per-Fold Summary ────────────────────────────────────────────

print("\n" + "="*110)
print("  SECTION 1: PER-FOLD SUMMARY (ensemble rows, excl. limit_entry)")
print("="*110)

for family_name, folds in [('α=2.0 (400-405)', FOLD_400), ('α=1.0 (410-415)', FOLD_410)]:
    print(f"\n  ── {family_name} {'─'*80}")
    print(f"  {'Fold':>6} {'Cutoff':>8} {'Tickers':>8} {'Modes':>6} {'Rows':>6} "
          f"{'VAL_avg':>10} {'VAL_med':>10} {'VAL+%':>8} "
          f"{'TEST_avg':>10} {'TEST_med':>10} {'TEST+%':>8}")
    print(f"  {'─'*6} {'─'*8} {'─'*8} {'─'*6} {'─'*6} "
          f"{'─'*10} {'─'*10} {'─'*8} "
          f"{'─'*10} {'─'*10} {'─'*8}")

    for fold in folds:
        sub = df_ens[df_ens['fold'] == fold]
        if len(sub) == 0:
            print(f"  {fold:>6} {CUTOFF_LABELS[fold]:>8}     (no data)")
            continue

        n_tickers = sub['ticker'].nunique()
        n_modes = sub['tpsl_mode'].nunique()
        val_avg = sub['val_a_avg_pnl'].mean()
        val_med = sub['val_a_avg_pnl'].median()
        val_pos = (sub['val_a_avg_pnl'] > 0).mean()
        test_avg = sub['val_b_avg_pnl'].mean()
        test_med = sub['val_b_avg_pnl'].median()
        test_pos = (sub['val_b_avg_pnl'] > 0).mean()

        print(f"  {fold:>6} {CUTOFF_LABELS[fold]:>8} {n_tickers:>8} {n_modes:>6} {len(sub):>6} "
              f"{val_avg:>+10.2%} {val_med:>+10.2%} {val_pos:>8.0%} "
              f"{test_avg:>+10.2%} {test_med:>+10.2%} {test_pos:>8.0%}")

# %% ── Section 2: Fold-to-Fold Degradation ───────────────────────────────────

print("\n" + "="*110)
print("  SECTION 2: FOLD-TO-FOLD DEGRADATION — Ticker Survival Curve")
print("="*110)

for family_name, folds in [('α=2.0 (400-405)', FOLD_400), ('α=1.0 (410-415)', FOLD_410)]:
    print(f"\n  ── {family_name} {'─'*80}")

    fold_tickers = {}
    for fold in folds:
        sub = df_ens[df_ens['fold'] == fold]
        tickers_val_pos = set(sub[sub['val_a_avg_pnl'] > 0]['ticker'].unique())
        tickers_any = set(sub['ticker'].unique())
        fold_tickers[fold] = {'any': tickers_any, 'val_pos': tickers_val_pos}

    all_tickers = set()
    for ft in fold_tickers.values():
        all_tickers |= ft['any']

    print(f"\n  {'Fold':>6} {'Cutoff':>8} {'Any result':>11} {'VAL+':>6} {'Drop vs F0':>12} {'Cumul. ∩':>10}")

    first_fold = folds[0]
    cumul_intersection = fold_tickers.get(first_fold, {}).get('val_pos', set()).copy()

    for i, fold in enumerate(folds):
        ft = fold_tickers.get(fold, {'any': set(), 'val_pos': set()})
        n_any = len(ft['any'])
        n_val = len(ft['val_pos'])

        if i == 0:
            drop = ""
            cumul_intersection = ft['val_pos'].copy()
        else:
            base = len(fold_tickers[first_fold]['val_pos'])
            drop_pct = (base - n_val) / base * 100 if base > 0 else 0
            drop = f"-{drop_pct:.0f}%"
            cumul_intersection &= ft['val_pos']

        print(f"  {fold:>6} {CUTOFF_LABELS[fold]:>8} {n_any:>11} {n_val:>6} {drop:>12} {len(cumul_intersection):>10}")

    print(f"\n  Tickers in ALL folds (cumulative ∩): {sorted(cumul_intersection)}")

# %% ── Section 3: Ticker × Fold Heatmap ──────────────────────────────────────

print("\n" + "="*110)
print("  SECTION 3: TICKER × FOLD HEATMAP (best ensemble VAL PnL per ticker-fold)")
print("="*110)

for family_name, folds in [('α=2.0 (400-405)', FOLD_400), ('α=1.0 (410-415)', FOLD_410)]:
    print(f"\n  ── {family_name} {'─'*80}")

    sub = df_ens[df_ens['fold'].isin(folds)].copy()

    # Best mode per ticker-fold (highest VAL PnL)
    best = sub.sort_values('val_a_avg_pnl', ascending=False).groupby(['ticker', 'fold']).first().reset_index()

    pivot_val = best.pivot(index='ticker', columns='fold', values='val_a_avg_pnl')
    pivot_test = best.pivot(index='ticker', columns='fold', values='val_b_avg_pnl')

    # Count folds with VAL+ per ticker
    val_pos_count = (pivot_val > 0).sum(axis=1)
    pivot_val['VAL+_folds'] = val_pos_count
    pivot_val = pivot_val.sort_values('VAL+_folds', ascending=False)

    # Align test pivot
    pivot_test = pivot_test.reindex(pivot_val.index)

    # Print header
    fold_headers = "  ".join(f"{f:>8}" for f in folds)
    print(f"\n  {'Ticker':>8}  {fold_headers}  {'VAL+':>5}  │  {fold_headers}")
    print(f"  {'':>8}  {'── VAL PnL ──':^{len(folds)*10}}       │  {'── TEST PnL ──':^{len(folds)*10}}")

    for ticker in pivot_val.index:
        val_cells = []
        test_cells = []
        for fold in folds:
            v = pivot_val.loc[ticker, fold] if fold in pivot_val.columns and pd.notna(pivot_val.loc[ticker].get(fold)) else None
            t = pivot_test.loc[ticker, fold] if fold in pivot_test.columns and pd.notna(pivot_test.loc[ticker].get(fold)) else None

            if v is not None:
                marker = "+" if v > 0 else "-"
                val_cells.append(f"{v:>+7.1%}{marker}")
            else:
                val_cells.append(f"{'·':>8}")

            if t is not None:
                marker = "+" if t > 0 else "-"
                test_cells.append(f"{t:>+7.1%}{marker}")
            else:
                test_cells.append(f"{'·':>8}")

        n_pos = int(pivot_val.loc[ticker, 'VAL+_folds'])
        val_str = "  ".join(val_cells)
        test_str = "  ".join(test_cells)
        print(f"  {ticker:>8}  {val_str}  {n_pos:>3}/6  │  {test_str}")

# %% ── Section 4: Best Deployment Fold Analysis ──────────────────────────────

print("\n" + "="*110)
print("  SECTION 4: WHICH FOLD TO DEPLOY FROM? (TEST performance by source fold)")
print("="*110)
print("  Question: If we deploy configs from fold N, what TEST PnL do we get?")
print("  Early folds have MORE tickers but potentially stale training.")
print("  Late folds have FEWER tickers but fresher training.\n")

for family_name, folds in [('α=2.0 (400-405)', FOLD_400), ('α=1.0 (410-415)', FOLD_410)]:
    print(f"\n  ── {family_name} {'─'*80}")

    print(f"\n  {'Source':>8} {'Cutoff':>8} │ {'N_tick':>7} {'TEST_avg':>10} {'TEST_med':>10} "
          f"{'TEST+%':>8} {'TEST_sum':>10} │ {'VAL_avg':>10} {'VAL→TEST':>10}")
    print(f"  {'─'*8} {'─'*8} │ {'─'*7} {'─'*10} {'─'*10} "
          f"{'─'*8} {'─'*10} │ {'─'*10} {'─'*10}")

    for fold in folds:
        sub = df_ens[(df_ens['fold'] == fold) & (df_ens['val_a_avg_pnl'] > 0)]
        if len(sub) == 0:
            print(f"  {fold:>8} {CUTOFF_LABELS[fold]:>8} │   (no VAL+ configs)")
            continue

        # Best mode per ticker (highest VAL PnL among VAL+ configs)
        best = sub.sort_values('val_a_avg_pnl', ascending=False).groupby('ticker').first().reset_index()

        n_tick = len(best)
        test_avg = best['val_b_avg_pnl'].mean()
        test_med = best['val_b_avg_pnl'].median()
        test_pos = (best['val_b_avg_pnl'] > 0).mean()
        test_sum = best['val_b_avg_pnl'].sum()
        val_avg = best['val_a_avg_pnl'].mean()
        val_test_ratio = test_avg / val_avg if val_avg != 0 else 0

        print(f"  {fold:>8} {CUTOFF_LABELS[fold]:>8} │ {n_tick:>7} {test_avg:>+10.2%} {test_med:>+10.2%} "
              f"{test_pos:>8.0%} {test_sum:>+10.2%} │ {val_avg:>+10.2%} {val_test_ratio:>+10.1%}")

# %% ── Section 5: Cross-Alpha Comparison ─────────────────────────────────────

print("\n" + "="*110)
print("  SECTION 5: α=2.0 vs α=1.0 — MATCHED TICKER COMPARISON")
print("="*110)
print("  For tickers appearing in BOTH families, compare TEST PnL\n")

# Get best ensemble per ticker per family (VAL+ only)
results_by_family = {}
for family_name, folds in [('400', FOLD_400), ('410', FOLD_410)]:
    sub = df_ens[(df_ens['fold'].isin(folds)) & (df_ens['val_a_avg_pnl'] > 0)]
    best = sub.sort_values('val_a_avg_pnl', ascending=False).groupby('ticker').first().reset_index()
    results_by_family[family_name] = best.set_index('ticker')[['fold', 'tpsl_mode', 'val_a_avg_pnl', 'val_b_avg_pnl']]

df_400 = results_by_family['400']
df_410 = results_by_family['410']
common = sorted(set(df_400.index) & set(df_410.index))

if common:
    print(f"  {'Ticker':>8}  │ {'── α=2.0 ──':^30}  │ {'── α=1.0 ──':^30}  │ {'Winner':>8}")
    print(f"  {'':>8}  │ {'Fold':>5} {'Mode':>8} {'VAL':>8} {'TEST':>8}  │ {'Fold':>5} {'Mode':>8} {'VAL':>8} {'TEST':>8}  │")

    a2_wins = 0
    a1_wins = 0

    for t in common:
        r4 = df_400.loc[t]
        r1 = df_410.loc[t]
        winner = "α=2.0" if r4['val_b_avg_pnl'] > r1['val_b_avg_pnl'] else "α=1.0"
        if r4['val_b_avg_pnl'] > r1['val_b_avg_pnl']:
            a2_wins += 1
        else:
            a1_wins += 1

        print(f"  {t:>8}  │ {int(r4['fold']):>5} {r4['tpsl_mode']:>8} {r4['val_a_avg_pnl']:>+7.1%} {r4['val_b_avg_pnl']:>+7.1%}  │"
              f" {int(r1['fold']):>5} {r1['tpsl_mode']:>8} {r1['val_a_avg_pnl']:>+7.1%} {r1['val_b_avg_pnl']:>+7.1%}  │ {winner:>8}")

    print(f"\n  α=2.0 wins: {a2_wins}  |  α=1.0 wins: {a1_wins}  |  matched tickers: {len(common)}")
    print(f"  α=2.0 avg TEST: {df_400.loc[common, 'val_b_avg_pnl'].mean():+.2%}  |  α=1.0 avg TEST: {df_410.loc[common, 'val_b_avg_pnl'].mean():+.2%}")
else:
    print("  No common tickers between families")

# Only-in-one-family
only_400 = sorted(set(df_400.index) - set(df_410.index))
only_410 = sorted(set(df_410.index) - set(df_400.index))
if only_400:
    print(f"\n  Only in α=2.0: {only_400}")
    for t in only_400:
        r = df_400.loc[t]
        print(f"    {t}: fold={int(r['fold'])} {r['tpsl_mode']} VAL={r['val_a_avg_pnl']:+.1%} TEST={r['val_b_avg_pnl']:+.1%}")
if only_410:
    print(f"\n  Only in α=1.0: {only_410}")
    for t in only_410:
        r = df_410.loc[t]
        print(f"    {t}: fold={int(r['fold'])} {r['tpsl_mode']} VAL={r['val_a_avg_pnl']:+.1%} TEST={r['val_b_avg_pnl']:+.1%}")

# %% ── Section 6: Fold Agreement Gate — Sweep Thresholds ─────────────────────

print("\n" + "="*110)
print("  SECTION 6: FOLD AGREEMENT GATE — WHAT THRESHOLD IS OPTIMAL?")
print("="*110)
print("  For each gate (1/6 through 6/6), show: tickers that pass, their avg TEST PnL")
print("  Lower gate = more tickers but less robustness")
print("  Higher gate = fewer tickers but higher confidence\n")

for family_name, folds in [('α=2.0 (400-405)', FOLD_400), ('α=1.0 (410-415)', FOLD_410)]:
    print(f"\n  ── {family_name} {'─'*80}")

    sub = df_ens[df_ens['fold'].isin(folds)].copy()
    all_tickers = sorted(sub['ticker'].unique())

    # Count VAL+ folds per ticker (best mode per ticker-fold)
    best_per_tf = sub.sort_values('val_a_avg_pnl', ascending=False).groupby(['ticker', 'fold']).first().reset_index()
    val_pos_counts = best_per_tf[best_per_tf['val_a_avg_pnl'] > 0].groupby('ticker')['fold'].count()

    # For TEST PnL, use the EARLIEST fold with VAL+ (since user wants to explore early folds)
    best_per_tf_valpos = best_per_tf[best_per_tf['val_a_avg_pnl'] > 0].copy()

    print(f"\n  {'Gate':>6} {'Pass':>6} {'Tickers':>50}  │ {'TEST_avg':>10} {'TEST_med':>10} {'TEST+%':>8}")
    print(f"  {'─'*6} {'─'*6} {'─'*50}  │ {'─'*10} {'─'*10} {'─'*8}")

    for gate in range(1, 7):
        passing = sorted(val_pos_counts[val_pos_counts >= gate].index.tolist())
        if not passing:
            print(f"  {gate:>4}/6 {0:>6} {'(none)':>50}  │")
            continue

        # Get TEST PnL for passing tickers — use best VAL+ config across all folds
        test_pnls = []
        for t in passing:
            t_rows = best_per_tf_valpos[best_per_tf_valpos['ticker'] == t]
            if len(t_rows) > 0:
                best_row = t_rows.sort_values('val_a_avg_pnl', ascending=False).iloc[0]
                test_pnls.append(best_row['val_b_avg_pnl'])

        test_arr = np.array(test_pnls)
        ticker_str = ", ".join(passing[:8])
        if len(passing) > 8:
            ticker_str += f" +{len(passing)-8} more"

        print(f"  {gate:>4}/6 {len(passing):>6} {ticker_str:>50}  │ "
              f"{test_arr.mean():>+10.2%} {np.median(test_arr):>+10.2%} {(test_arr > 0).mean():>8.0%}")

# %% ── Section 7: Early Fold Deployment — Can We Use Fold 400-402 Configs? ───

print("\n" + "="*110)
print("  SECTION 7: EARLY vs LATE FOLD DEPLOYMENT")
print("="*110)
print("  Key question: Do early fold configs (400-402) outperform late folds (403-405)?")
print("  Early folds have MORE tickers. If their TEST PnL holds, they're better deployment sources.\n")

for family_name, folds in [('α=2.0 (400-405)', FOLD_400), ('α=1.0 (410-415)', FOLD_410)]:
    print(f"\n  ── {family_name} {'─'*80}")

    early = folds[:3]
    late = folds[3:]

    sub = df_ens[df_ens['fold'].isin(folds)].copy()
    best_per_tf = sub.sort_values('val_a_avg_pnl', ascending=False).groupby(['ticker', 'fold']).first().reset_index()
    best_valpos = best_per_tf[best_per_tf['val_a_avg_pnl'] > 0].copy()

    # Tickers that appear in early but not late folds
    early_tickers = set(best_valpos[best_valpos['fold'].isin(early)]['ticker'])
    late_tickers = set(best_valpos[best_valpos['fold'].isin(late)]['ticker'])

    early_only = sorted(early_tickers - late_tickers)
    both = sorted(early_tickers & late_tickers)
    late_only = sorted(late_tickers - early_tickers)

    print(f"  Early-only ({early}): {len(early_only)} tickers")
    print(f"  Both groups:          {len(both)} tickers")
    print(f"  Late-only ({late}):  {len(late_only)} tickers")

    # For shared tickers, compare TEST PnL from early vs late fold configs
    if both:
        print(f"\n  {'Ticker':>8}  │ {'── Early best ──':^28}  │ {'── Late best ──':^28}  │ {'Better':>8}")
        print(f"  {'':>8}  │ {'Fold':>5} {'VAL':>8} {'TEST':>8} {'Mode':>8}  │ {'Fold':>5} {'VAL':>8} {'TEST':>8} {'Mode':>8}  │")

        early_wins = 0
        late_wins = 0
        early_tests = []
        late_tests = []

        for t in both:
            e_rows = best_valpos[(best_valpos['ticker'] == t) & (best_valpos['fold'].isin(early))]
            l_rows = best_valpos[(best_valpos['ticker'] == t) & (best_valpos['fold'].isin(late))]

            e_best = e_rows.sort_values('val_a_avg_pnl', ascending=False).iloc[0]
            l_best = l_rows.sort_values('val_a_avg_pnl', ascending=False).iloc[0]

            better = "EARLY" if e_best['val_b_avg_pnl'] > l_best['val_b_avg_pnl'] else "LATE"
            if e_best['val_b_avg_pnl'] > l_best['val_b_avg_pnl']:
                early_wins += 1
            else:
                late_wins += 1

            early_tests.append(e_best['val_b_avg_pnl'])
            late_tests.append(l_best['val_b_avg_pnl'])

            print(f"  {t:>8}  │ {int(e_best['fold']):>5} {e_best['val_a_avg_pnl']:>+7.1%} {e_best['val_b_avg_pnl']:>+7.1%} {e_best['tpsl_mode']:>8}  │"
                  f" {int(l_best['fold']):>5} {l_best['val_a_avg_pnl']:>+7.1%} {l_best['val_b_avg_pnl']:>+7.1%} {l_best['tpsl_mode']:>8}  │ {better:>8}")

        print(f"\n  Early wins: {early_wins}  |  Late wins: {late_wins}")
        print(f"  Early avg TEST: {np.mean(early_tests):+.2%}  |  Late avg TEST: {np.mean(late_tests):+.2%}")

    # Early-only tickers: these are candidates only available from early folds
    if early_only:
        print(f"\n  EARLY-ONLY tickers (deployable ONLY from early folds):")
        for t in early_only:
            e_rows = best_valpos[(best_valpos['ticker'] == t) & (best_valpos['fold'].isin(early))]
            e_best = e_rows.sort_values('val_a_avg_pnl', ascending=False).iloc[0]
            n_folds = len(e_rows)
            print(f"    {t:>8}: fold={int(e_best['fold'])} {e_best['tpsl_mode']:>8} "
                  f"VAL={e_best['val_a_avg_pnl']:+.1%} TEST={e_best['val_b_avg_pnl']:+.1%} "
                  f"(in {n_folds}/3 early folds)")

# %% ── Section 8: Recommended Deploy List ────────────────────────────────────

print("\n" + "="*110)
print("  SECTION 8: DEPLOYMENT CANDIDATES (combining all insights)")
print("="*110)
print("  Strategy: Use configs from the fold with best VAL PnL for that ticker,")
print("  gated by fold agreement (>= 2/6 folds with VAL+).")
print("  This allows early fold configs while still requiring some cross-fold robustness.\n")

for family_name, folds in [('α=2.0 (400-405)', FOLD_400), ('α=1.0 (410-415)', FOLD_410)]:
    print(f"\n  ── {family_name} {'─'*80}")

    sub = df_ens[df_ens['fold'].isin(folds)].copy()
    best_per_tf = sub.sort_values('val_a_avg_pnl', ascending=False).groupby(['ticker', 'fold']).first().reset_index()
    best_valpos = best_per_tf[best_per_tf['val_a_avg_pnl'] > 0].copy()

    val_pos_counts = best_valpos.groupby('ticker')['fold'].count()

    # Gate: >= 2/6
    gate = 2
    passing = sorted(val_pos_counts[val_pos_counts >= gate].index.tolist())

    if not passing:
        print(f"  No tickers pass gate >= {gate}/6")
        continue

    print(f"\n  {'Ticker':>8} {'SrcFold':>8} {'Mode':>8} {'prob':>6} {'TP%':>6} {'SL%':>6} "
          f"{'VAL_PnL':>9} {'TEST_PnL':>9} {'TEST_WR':>8} {'TEST_N':>7} {'Agree':>6}")
    print(f"  {'─'*8} {'─'*8} {'─'*8} {'─'*6} {'─'*6} {'─'*6} "
          f"{'─'*9} {'─'*9} {'─'*8} {'─'*7} {'─'*6}")

    deploy_rows = []
    for t in passing:
        t_rows = best_valpos[best_valpos['ticker'] == t]
        best = t_rows.sort_values('val_a_avg_pnl', ascending=False).iloc[0]
        n_agree = int(val_pos_counts[t])

        print(f"  {t:>8} {int(best['fold']):>8} {best['tpsl_mode']:>8} {best['prob_thresh']:>6.2f} "
              f"{best['tp_pct']:>5.1%} {best['sl_pct']:>5.1%} "
              f"{best['val_a_avg_pnl']:>+8.1%} {best['val_b_avg_pnl']:>+8.1%} "
              f"{best['val_b_win_rate']:>8.0%} {int(best.get('val_b_count', 0)):>7} {n_agree:>4}/6")

        deploy_rows.append({
            'ticker': t, 'fold': int(best['fold']), 'mode': best['tpsl_mode'],
            'prob': best['prob_thresh'], 'tp': best['tp_pct'], 'sl': best['sl_pct'],
            'val_pnl': best['val_a_avg_pnl'], 'test_pnl': best['val_b_avg_pnl'],
            'agree': n_agree
        })

    if deploy_rows:
        test_pnls = [r['test_pnl'] for r in deploy_rows]
        print(f"\n  Total: {len(deploy_rows)} tickers | "
              f"Avg TEST: {np.mean(test_pnls):+.2%} | "
              f"Median TEST: {np.median(test_pnls):+.2%} | "
              f"TEST+ rate: {sum(1 for p in test_pnls if p > 0)/len(test_pnls):.0%}")

print("\n" + "="*110)
print("  NOTE: Fold 415 may still be running — results will update on re-run")
print("="*110)
