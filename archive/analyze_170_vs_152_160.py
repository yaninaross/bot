"""
analyze_170_vs_152_160.py — Head-to-head comparison of WF v12 results
======================================================================
Compares 170-175 (walkforward 6 folds) vs 152 (late cutoff) vs 160 (early cutoff)
across all 9 TPSL modes. Pulls ensemble + per-seed data from BQ.

Sections:
  1. Per-mode summary: avg test PnL, WR, N for each family
  2. Head-to-head per ticker: which family wins on test PnL
  3. Best-mode-per-ticker: for each ticker, which family × mode combo is best
  4. Overlap analysis: tickers scored in all 3 families
"""

import pandas as pd
import numpy as np
from google.cloud import bigquery as bq

try:
    from google.colab import auth
    auth.authenticate_user()
except Exception:
    pass

BQ_TABLE = f"{GCP_PROJECT}.scoring_logs.wf_optimization_results"
MODEL_VERSION = 'robust_wf_v12'
DEPLOY_THRESHOLD = 0.018

TICKER_RENAME   = {'MIGI': 'BGDE', 'BITF': 'KEEL'}
EXCLUDE_TICKERS = {'TLT', 'VIX', 'SMLR', 'SPIR'}

MODE_MAP = {
    '': 'static', '_ATR': 'atr', '_ATRADP': 'atr_adaptive',
    '_REGIME': 'regime_scaled', '_TRAIL': 'trailing_decay',
    '_REGADP': 'regime_adaptive', '_ATRSL': 'atr_sl',
    '_GRACE': 'grace', '_LIMIT': 'limit_entry',
}

ALL_MODES = ['static', 'atr', 'atr_adaptive', 'regime_scaled', 'trailing_decay',
             'regime_adaptive', 'atr_sl', 'grace', 'limit_entry']

FAMILIES = {
    '170': {'label': '170-175 WF', 'prefix': '17'},
    '160': {'label': '160 Early', 'prefix': '160'},
    '152': {'label': '152 Late',  'prefix': '152'},
}

client = bq.Client()

# ═══════════════════════════════════════════════════════════════════════
# 1. LOAD ALL DATA
# ═══════════════════════════════════════════════════════════════════════
print("Loading WF v12 results from BQ...")

query = f"""
SELECT experiment, ticker, seed, tpsl_mode, is_ensemble, n_seeds_avg,
       opt_avg_pnl, opt_win_rate, opt_count,
       val_a_avg_pnl, val_a_win_rate, val_a_count, survived_a,
       val_b_avg_pnl, val_b_win_rate, val_b_count, survived_b,
       prob_thresh, tp_pct, sl_pct, tp_mult, sl_mult,
       trail_start_mult, trail_end_mult,
       grace_minutes, dip_pct, max_wait_minutes,
       vol_filter_pct, pre_range_filter_pct
FROM `{BQ_TABLE}`
WHERE model_version = '{MODEL_VERSION}'
  AND (experiment LIKE '17%' OR experiment LIKE '160%' OR experiment LIKE '152%')
"""

df = client.query(query).to_dataframe()
print(f"  Loaded {len(df)} rows")

df['ticker'] = df['ticker'].replace(TICKER_RENAME)
df = df[~df['ticker'].isin(EXCLUDE_TICKERS)]
for col in ['grace_minutes', 'max_wait_minutes', 'vol_filter_pct']:
    if col in df.columns:
        df[col] = df[col].fillna(0).astype(int)
for col in ['dip_pct']:
    if col in df.columns:
        df[col] = df[col].fillna(0.0)

def assign_family(exp):
    base = exp.split('_')[0]
    if not base.isdigit():
        return 'other'
    num = int(base)
    if num == 152: return '152'
    if num == 160: return '160'
    if 170 <= num <= 175: return '170'
    return 'other'

df['family'] = df['experiment'].apply(assign_family)

if 'tpsl_mode' not in df.columns or df['tpsl_mode'].isna().all():
    def extract_mode(exp):
        for suffix, mode in sorted(MODE_MAP.items(), key=lambda x: -len(x[0])):
            if suffix and exp.endswith(suffix):
                return mode
        return 'static'
    df['tpsl_mode'] = df['experiment'].apply(extract_mode)

df['tpsl_mode'] = df['tpsl_mode'].fillna('static')

ens = df[df['is_ensemble'] == True].copy()
seeds = df[df['is_ensemble'] != True].copy()

print(f"  Ensemble rows: {len(ens)}")
print(f"  Per-seed rows: {len(seeds)}")

for fam in ['170', '160', '152']:
    fam_ens = ens[ens['family'] == fam]
    tickers = fam_ens['ticker'].nunique()
    modes = fam_ens['tpsl_mode'].nunique()
    print(f"  {FAMILIES[fam]['label']}: {tickers} tickers, {modes} modes (ensemble)")

W = 120

# ═══════════════════════════════════════════════════════════════════════
# 2. PER-MODE SUMMARY — ensemble test PnL by family × mode
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*W}")
print(f"  SECTION 1: PER-MODE SUMMARY (Ensemble Test PnL)")
print(f"  Test period: val_b columns (holdout)")
print(f"{'='*W}")

mode_summary = ens.groupby(['family', 'tpsl_mode']).agg(
    tickers=('ticker', 'nunique'),
    avg_test_pnl=('val_b_avg_pnl', 'mean'),
    med_test_pnl=('val_b_avg_pnl', 'median'),
    avg_test_wr=('val_b_win_rate', 'mean'),
    avg_test_n=('val_b_count', 'mean'),
    pct_pass=('survived_b', 'mean'),
).reset_index()

print(f"\n  {'Mode':<18}", end='')
for fam in ['170', '160', '152']:
    print(f"  │ {FAMILIES[fam]['label']:^30}", end='')
print()
print(f"  {'':─<18}", end='')
for _ in ['170', '160', '152']:
    print(f"──┼{'':─^30}", end='')
print()
print(f"  {'':18}", end='')
for _ in ['170', '160', '152']:
    print(f"  │ {'Tix':>4} {'PnL':>7} {'WR':>6} {'Pass%':>6} {'N':>5}", end='')
print()
print(f"  {'':─<18}", end='')
for _ in ['170', '160', '152']:
    print(f"──┼{'':─^30}", end='')
print()

for mode in ALL_MODES:
    print(f"  {mode:<18}", end='')
    for fam in ['170', '160', '152']:
        row = mode_summary[(mode_summary['family'] == fam) & (mode_summary['tpsl_mode'] == mode)]
        if len(row) == 0:
            print(f"  │ {'—':>4} {'—':>7} {'—':>6} {'—':>6} {'—':>5}", end='')
        else:
            r = row.iloc[0]
            print(f"  │ {int(r['tickers']):>4} {r['avg_test_pnl']:>+6.2%} {r['avg_test_wr']:>5.1%} "
                  f"{r['pct_pass']:>5.0%} {r['avg_test_n']:>5.0f}", end='')
    print()

print(f"  {'':─<18}", end='')
for _ in ['170', '160', '152']:
    print(f"──┼{'':─^30}", end='')
print()

# Overall averages
print(f"  {'ALL MODES':<18}", end='')
for fam in ['170', '160', '152']:
    fam_ens = ens[ens['family'] == fam]
    if len(fam_ens) == 0:
        print(f"  │ {'—':>4} {'—':>7} {'—':>6} {'—':>6} {'—':>5}", end='')
    else:
        print(f"  │ {fam_ens['ticker'].nunique():>4} {fam_ens['val_b_avg_pnl'].mean():>+6.2%} "
              f"{fam_ens['val_b_win_rate'].mean():>5.1%} "
              f"{fam_ens['survived_b'].mean():>5.0%} {fam_ens['val_b_count'].mean():>5.0f}", end='')
print()

# ═══════════════════════════════════════════════════════════════════════
# 3. HEAD-TO-HEAD PER TICKER — best ensemble per mode, common tickers
# ═══════════════════════════════════════════════════════════════════════
print(f"\n\n{'='*W}")
print(f"  SECTION 2: HEAD-TO-HEAD PER TICKER (Ensemble, Best Mode Per Family)")
print(f"  For each ticker, picks the mode with highest test PnL per family")
print(f"{'='*W}")

best_per_ticker_fam = ens.loc[ens.groupby(['family', 'ticker'])['val_b_avg_pnl'].idxmax()].copy()
best_per_ticker_fam = best_per_ticker_fam[['family', 'ticker', 'tpsl_mode', 'val_b_avg_pnl',
                                            'val_b_win_rate', 'val_b_count', 'survived_b',
                                            'opt_avg_pnl', 'prob_thresh']].copy()

pivot_pnl = best_per_ticker_fam.pivot(index='ticker', columns='family', values='val_b_avg_pnl')
pivot_mode = best_per_ticker_fam.pivot(index='ticker', columns='family', values='tpsl_mode')
pivot_wr = best_per_ticker_fam.pivot(index='ticker', columns='family', values='val_b_win_rate')
pivot_n = best_per_ticker_fam.pivot(index='ticker', columns='family', values='val_b_count')

all_tickers = sorted(pivot_pnl.index)
common_tickers = sorted(pivot_pnl.dropna().index)

print(f"\n  All tickers with ANY data: {len(all_tickers)}")
print(f"  Tickers in ALL 3 families: {len(common_tickers)}")

# Count which family has only partial coverage
for fam in ['170', '160', '152']:
    fam_tickers = set(pivot_pnl[fam].dropna().index)
    print(f"    {FAMILIES[fam]['label']}: {len(fam_tickers)} tickers")

print(f"\n  {'Ticker':<7}", end='')
for fam in ['170', '160', '152']:
    print(f"  │ {FAMILIES[fam]['label']:^28}", end='')
print(f"  │ {'Winner':>10}")
print(f"  {'':─<7}", end='')
for _ in ['170', '160', '152']:
    print(f"──┼{'':─^28}", end='')
print(f"──┼{'':─^10}")
print(f"  {'':7}", end='')
for _ in ['170', '160', '152']:
    print(f"  │ {'Mode':>12} {'PnL':>7} {'WR':>5} {'N':>3}", end='')
print(f"  │")
print(f"  {'':─<7}", end='')
for _ in ['170', '160', '152']:
    print(f"──┼{'':─^28}", end='')
print(f"──┼{'':─^10}")

wins = {'170': 0, '160': 0, '152': 0, 'tie': 0}

for ticker in all_tickers:
    print(f"  {ticker:<7}", end='')
    best_fam = None
    best_pnl = -999

    for fam in ['170', '160', '152']:
        pnl = pivot_pnl.loc[ticker].get(fam)
        mode = pivot_mode.loc[ticker].get(fam) if fam in pivot_mode.columns else None
        wr = pivot_wr.loc[ticker].get(fam)
        n = pivot_n.loc[ticker].get(fam)

        if pd.isna(pnl):
            print(f"  │ {'—':>12} {'—':>7} {'—':>5} {'—':>3}", end='')
        else:
            mode_short = (mode or 'static')[:12]
            print(f"  │ {mode_short:>12} {pnl:>+6.2%} {wr:>4.0%} {int(n):>3}", end='')
            if pnl > best_pnl:
                best_pnl = pnl
                best_fam = fam

    if best_fam:
        label = FAMILIES[best_fam]['label'][:8]
        wins[best_fam] += 1
    else:
        label = '—'
    print(f"  │ {label:>10}")

print(f"  {'':─<7}", end='')
for _ in ['170', '160', '152']:
    print(f"──┼{'':─^28}", end='')
print(f"──┼{'':─^10}")

print(f"\n  WINS: ", end='')
for fam in ['170', '160', '152']:
    print(f"{FAMILIES[fam]['label']}={wins[fam]}  ", end='')
print()

# ═══════════════════════════════════════════════════════════════════════
# 4. COMMON TICKERS ONLY — paired comparison
# ═══════════════════════════════════════════════════════════════════════
if len(common_tickers) >= 3:
    print(f"\n\n{'='*W}")
    print(f"  SECTION 3: PAIRED COMPARISON — {len(common_tickers)} tickers in ALL 3 families")
    print(f"  Same tickers, best mode per family, ensemble test PnL")
    print(f"{'='*W}")

    common_pivot = pivot_pnl.loc[common_tickers]

    for fam in ['170', '160', '152']:
        vals = common_pivot[fam]
        pos = (vals > 0).sum()
        print(f"  {FAMILIES[fam]['label']:>12}: avg={vals.mean():>+6.2%}  med={vals.median():>+6.2%}  "
              f"pos={pos}/{len(vals)} ({pos/len(vals):.0%})  "
              f"best={vals.max():>+6.2%} ({vals.idxmax()})  "
              f"worst={vals.min():>+6.2%} ({vals.idxmin()})")

    # Paired differences
    print(f"\n  Paired differences (row = row - col):")
    print(f"  {'':>12} {'170-175':>10} {'160':>10} {'152':>10}")
    for fam_a in ['170', '160', '152']:
        print(f"  {FAMILIES[fam_a]['label']:>12}", end='')
        for fam_b in ['170', '160', '152']:
            if fam_a == fam_b:
                print(f" {'—':>10}", end='')
            else:
                diff = common_pivot[fam_a] - common_pivot[fam_b]
                print(f" {diff.mean():>+9.2%}", end='')
        print()

    # Signed rank test
    print(f"\n  170-175 vs 160 (per ticker):")
    diff_170_160 = common_pivot['170'] - common_pivot['160']
    n_better = (diff_170_160 > 0).sum()
    n_worse = (diff_170_160 < 0).sum()
    print(f"    170 better: {n_better}/{len(diff_170_160)}  |  160 better: {n_worse}/{len(diff_170_160)}")
    print(f"    Avg delta: {diff_170_160.mean():>+.2%}  |  Med delta: {diff_170_160.median():>+.2%}")

    print(f"\n  170-175 vs 152 (per ticker):")
    diff_170_152 = common_pivot['170'] - common_pivot['152']
    n_better = (diff_170_152 > 0).sum()
    n_worse = (diff_170_152 < 0).sum()
    print(f"    170 better: {n_better}/{len(diff_170_152)}  |  152 better: {n_worse}/{len(diff_170_152)}")
    print(f"    Avg delta: {diff_170_152.mean():>+.2%}  |  Med delta: {diff_170_152.median():>+.2%}")

# ═══════════════════════════════════════════════════════════════════════
# 5. PER-MODE HEAD-TO-HEAD — same mode, compare families
# ═══════════════════════════════════════════════════════════════════════
print(f"\n\n{'='*W}")
print(f"  SECTION 4: PER-MODE HEAD-TO-HEAD (Ensemble, same mode across families)")
print(f"{'='*W}")

for mode in ALL_MODES:
    mode_ens = ens[ens['tpsl_mode'] == mode].copy()
    if len(mode_ens) == 0:
        continue

    mode_pivot = mode_ens.pivot_table(index='ticker', columns='family',
                                       values='val_b_avg_pnl', aggfunc='mean')
    mode_common = mode_pivot.dropna()

    if len(mode_common) < 2:
        print(f"\n  {mode.upper()}: <2 common tickers, skipping")
        continue

    print(f"\n  {mode.upper()} — {len(mode_common)} common tickers:")
    mode_wins = {f: 0 for f in ['170', '160', '152']}
    for fam in ['170', '160', '152']:
        if fam not in mode_common.columns:
            continue
        vals = mode_common[fam]
        print(f"    {FAMILIES[fam]['label']:>12}: avg={vals.mean():>+6.2%}  "
              f"pos={int((vals>0).sum())}/{len(vals)}")

    if '170' in mode_common.columns and '160' in mode_common.columns:
        diff = mode_common['170'] - mode_common['160']
        n_b = (diff > 0).sum()
        n_w = (diff < 0).sum()
        print(f"    170 vs 160: 170 better={n_b}  160 better={n_w}  delta={diff.mean():>+.2%}")

    if '170' in mode_common.columns and '152' in mode_common.columns:
        diff = mode_common['170'] - mode_common['152']
        n_b = (diff > 0).sum()
        n_w = (diff < 0).sum()
        print(f"    170 vs 152: 170 better={n_b}  152 better={n_w}  delta={diff.mean():>+.2%}")

# ═══════════════════════════════════════════════════════════════════════
# 6. DEPLOYABLE CONFIGS — test PnL >= 1.8% threshold
# ═══════════════════════════════════════════════════════════════════════
DEPLOY_THRESHOLD = 0.018

print(f"\n\n{'='*W}")
print(f"  SECTION 5: DEPLOYABLE CONFIGS (test PnL >= {DEPLOY_THRESHOLD:.1%})")
print(f"{'='*W}")

for fam in ['170', '160', '152']:
    fam_ens = ens[ens['family'] == fam].copy()
    deployable = fam_ens[fam_ens['val_b_avg_pnl'] >= DEPLOY_THRESHOLD].copy()
    deploy_tickers = deployable['ticker'].nunique()
    total_tickers = fam_ens['ticker'].nunique()

    print(f"\n  {FAMILIES[fam]['label']}:")
    print(f"    {deploy_tickers}/{total_tickers} tickers have at least 1 deployable config")

    if len(deployable) > 0:
        deploy_best = deployable.loc[deployable.groupby('ticker')['val_b_avg_pnl'].idxmax()]
        deploy_best = deploy_best.sort_values('val_b_avg_pnl', ascending=False)

        print(f"    {'Ticker':<7} {'Mode':<15} {'TestPnL':>8} {'TestWR':>7} {'TestN':>6} {'Prob':>6}")
        print(f"    {'─'*55}")
        for _, r in deploy_best.iterrows():
            print(f"    {r['ticker']:<7} {r['tpsl_mode']:<15} {r['val_b_avg_pnl']:>+7.2%} "
                  f"{r['val_b_win_rate']:>6.1%} {int(r['val_b_count']):>6} {r['prob_thresh']:>6.2f}")
        print(f"    {'─'*55}")
        print(f"    Avg deployable PnL: {deploy_best['val_b_avg_pnl'].mean():>+.2%}  "
              f"WR: {deploy_best['val_b_win_rate'].mean():.1%}")

# ═══════════════════════════════════════════════════════════════════════
# 7. PER-SEED STABILITY — how many seeds pass test per family
# ═══════════════════════════════════════════════════════════════════════
print(f"\n\n{'='*W}")
print(f"  SECTION 6: SEED STABILITY (Per-seed test pass rate)")
print(f"{'='*W}")

seed_stability = seeds.groupby(['family', 'ticker', 'tpsl_mode']).agg(
    n_seeds=('seed', 'nunique'),
    pass_test=('survived_b', 'sum'),
    avg_test=('val_b_avg_pnl', 'mean'),
    std_test=('val_b_avg_pnl', 'std'),
).reset_index()
seed_stability['pass_rate'] = seed_stability['pass_test'] / seed_stability['n_seeds']

for fam in ['170', '160', '152']:
    fam_ss = seed_stability[seed_stability['family'] == fam]
    if len(fam_ss) == 0:
        continue
    print(f"\n  {FAMILIES[fam]['label']}:")
    print(f"    Avg seeds per ticker-mode: {fam_ss['n_seeds'].mean():.1f}")
    print(f"    Avg pass rate: {fam_ss['pass_rate'].mean():.1%}")
    print(f"    Avg seed test PnL std: {fam_ss['std_test'].mean():.2%}")

    # Best modes by seed agreement
    best_modes = fam_ss.groupby('tpsl_mode').agg(
        avg_pass_rate=('pass_rate', 'mean'),
        avg_test_pnl=('avg_test', 'mean'),
    ).sort_values('avg_pass_rate', ascending=False)
    print(f"    {'Mode':<18} {'Pass Rate':>10} {'Avg Test PnL':>12}")
    for mode, r in best_modes.iterrows():
        print(f"    {mode:<18} {r['avg_pass_rate']:>9.1%} {r['avg_test_pnl']:>+11.2%}")

# ═══════════════════════════════════════════════════════════════════════
# 8. VERDICT
# ═══════════════════════════════════════════════════════════════════════
print(f"\n\n{'='*W}")
print(f"  VERDICT")
print(f"{'='*W}")

for fam in ['170', '160', '152']:
    fam_ens = ens[ens['family'] == fam]
    if len(fam_ens) == 0:
        continue
    best_mode_avg = fam_ens.groupby('tpsl_mode')['val_b_avg_pnl'].mean()
    best_mode = best_mode_avg.idxmax()
    deployable_n = (fam_ens['val_b_avg_pnl'] >= DEPLOY_THRESHOLD).sum()
    print(f"  {FAMILIES[fam]['label']:>12}: avg_test={fam_ens['val_b_avg_pnl'].mean():>+.2%}  "
          f"best_mode={best_mode} ({best_mode_avg.max():>+.2%})  "
          f"deployable={deployable_n}/{len(fam_ens)}")

print(f"\n{'='*W}")
