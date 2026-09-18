"""
audit_wf_v12_completeness.py — Full v12 WF optimization data audit
====================================================================
Pulls ALL robust_wf_v12 results from BQ (no experiment filter) and maps:
  1. Raw census — every experiment, ticker, mode, seed count
  2. Experiment × Mode matrix — which combos exist
  3. Experiment × Ticker matrix — coverage per experiment
  4. Limit Entry deep dive — where it exists, seed counts, configs
  5. Walkforward chain audit — 170-175 and 180-185 fold completeness
  6. Seed coverage — per experiment × ticker × mode, are 15 seeds present?
  7. Gap report — expected vs actual, actionable fix list

Colab-ready (no GPU needed, BQ only).
"""

import os
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from google.cloud import bigquery

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass

# ==============================================================================
# CONFIG
# ==============================================================================
BQ_TABLE = f"{GCP_PROJECT}.scoring_logs.wf_optimization_results"
MODEL_VERSION = 'robust_wf_v12'
W = 130

EXPECTED_SEEDS = 15
EXPECTED_MODES = [
    'static', 'atr', 'atr_adaptive', 'regime_scaled', 'trailing_decay',
    'regime_adaptive', 'atr_sl', 'grace', 'limit_entry',
]

MODE_SUFFIX = {
    'static': '', 'atr': '_ATR', 'atr_adaptive': '_ATRADP',
    'regime_scaled': '_REGIME', 'trailing_decay': '_TRAIL',
    'regime_adaptive': '_REGADP', 'atr_sl': '_ATRSL',
    'grace': '_GRACE', 'limit_entry': '_LIMIT',
}

TICKER_RENAME = {'MIGI': 'BGDE', 'BITF': 'KEEL'}

TRAIN_TEACHERS_13 = [
    'BBAI', 'RCAT', 'LUNR', 'ASTS', 'ACHR', 'MVIS',
    'WULF', 'CIFR', 'IREN', 'KEEL', 'HUT', 'CLSK', 'APLD',
]

TEACHERS_31 = [
    'CIFR', 'IREN', 'KEEL', 'HUT', 'APLD', 'WULF', 'RIOT', 'CLSK', 'BTBT',
    'ALAB', 'MARA', 'RCAT', 'ASTS', 'LUNR', 'BBAI', 'RKLB', 'SPIR', 'BKSY',
    'MNTS', 'PLUG', 'BE', 'FCEL', 'FLNC', 'OKLO', 'SMR', 'MVIS', 'BGDE',
    'RDW', 'PL', 'VOYG', 'SMLR',
]

SCORING_POOL_152_160 = sorted(set(TRAIN_TEACHERS_13 + [
    'RIOT', 'BTBT', 'ALAB', 'MARA', 'RKLB', 'SPIR', 'BKSY', 'MNTS',
    'PLUG', 'BE', 'FCEL', 'FLNC', 'OKLO', 'SMR', 'BGDE', 'RDW', 'PL',
    'VOYG', 'SMLR', 'SMCI', 'CORZ', 'MU', 'FCX',
]))

SCORING_POOL_185 = sorted(set(TEACHERS_31 + [
    'SMCI', 'CORZ', 'MU', 'FCX', 'ACHR',
]))

EXPECTED_EXPERIMENTS = {
    '152_XSECTOR': {'label': '152 Late (single)', 'tickers': SCORING_POOL_152_160, 'seeds': 15, 'seed_base': 152},
    '160_XSECTOR': {'label': '160 Early (single)', 'tickers': SCORING_POOL_152_160, 'seeds': 15, 'seed_base': 160},
    '170_XSECTOR': {'label': '170 WF fold 1', 'tickers': SCORING_POOL_152_160, 'seeds': 15, 'seed_base': 170},
    '171_XSECTOR': {'label': '171 WF fold 2', 'tickers': SCORING_POOL_152_160, 'seeds': 15, 'seed_base': 171},
    '172_XSECTOR': {'label': '172 WF fold 3', 'tickers': SCORING_POOL_152_160, 'seeds': 15, 'seed_base': 172},
    '173_XSECTOR': {'label': '173 WF fold 4', 'tickers': SCORING_POOL_152_160, 'seeds': 15, 'seed_base': 173},
    '174_XSECTOR': {'label': '174 WF fold 5', 'tickers': SCORING_POOL_152_160, 'seeds': 15, 'seed_base': 174},
    '175_XSECTOR': {'label': '175 WF fold 6 (deployed)', 'tickers': SCORING_POOL_152_160, 'seeds': 15, 'seed_base': 175},
    '180_XSECTOR': {'label': '180 WF fold 1', 'tickers': SCORING_POOL_185, 'seeds': 15, 'seed_base': 180},
    '181_XSECTOR': {'label': '181 WF fold 2', 'tickers': SCORING_POOL_185, 'seeds': 15, 'seed_base': 181},
    '182_XSECTOR': {'label': '182 WF fold 3', 'tickers': SCORING_POOL_185, 'seeds': 15, 'seed_base': 182},
    '183_XSECTOR': {'label': '183 WF fold 4', 'tickers': SCORING_POOL_185, 'seeds': 15, 'seed_base': 183},
    '184_XSECTOR': {'label': '184 WF fold 5', 'tickers': SCORING_POOL_185, 'seeds': 15, 'seed_base': 184},
    '185_XSECTOR': {'label': '185 WF fold 6 (deployed)', 'tickers': SCORING_POOL_185, 'seeds': 15, 'seed_base': 185},
    '200_XSECTOR': {'label': '200 CNN-TX (single)', 'tickers': SCORING_POOL_185, 'seeds': 15, 'seed_base': 200},
}

WF_CHAINS = {
    '170-175': ['170_XSECTOR', '171_XSECTOR', '172_XSECTOR',
                '173_XSECTOR', '174_XSECTOR', '175_XSECTOR'],
    '180-185': ['180_XSECTOR', '181_XSECTOR', '182_XSECTOR',
                '183_XSECTOR', '184_XSECTOR', '185_XSECTOR'],
}

bq_client = bigquery.Client()


# ==============================================================================
# 1. LOAD ALL V12 DATA
# ==============================================================================
print(f"\n{'='*W}")
print(f"  WF V12 COMPLETENESS AUDIT")
print(f"  Table: {BQ_TABLE}  |  Model: {MODEL_VERSION}")
print(f"{'='*W}")

print(f"\n  Loading ALL {MODEL_VERSION} rows from BQ (no experiment filter)...")
query = f"""
SELECT experiment, ticker, seed, tpsl_mode, entry_minute, entry_time,
       prob_thresh, tp_pct, sl_pct, tp_mult, sl_mult,
       trail_start_mult, trail_end_mult,
       vix_threshold, tlt_crash, spy_ignore, vol_filter_pct,
       val_a_avg_pnl, val_a_win_rate, val_a_count, val_a_freq,
       val_b_avg_pnl, val_b_win_rate, val_b_count, val_b_freq,
       opt_avg_pnl, opt_win_rate, opt_count,
       is_ensemble, survived_a, survived_b,
       grace_minutes, dip_pct, max_wait_minutes
FROM `{BQ_TABLE}`
WHERE model_version = '{MODEL_VERSION}'
"""
raw = bq_client.query(query).to_dataframe()
print(f"  Total rows: {len(raw):,}")

raw['ticker'] = raw['ticker'].replace(TICKER_RENAME)

all_suffixes = list(MODE_SUFFIX.values())
def strip_mode_suffix(exp_name):
    for sfx in sorted(all_suffixes, key=len, reverse=True):
        if sfx and exp_name.endswith(sfx):
            return exp_name[:-len(sfx)]
    return exp_name

raw['base_experiment'] = raw['experiment'].apply(strip_mode_suffix)


# ==============================================================================
# 2. RAW CENSUS — what experiments exist?
# ==============================================================================
print(f"\n{'='*W}")
print(f"  SECTION 1: RAW DATA CENSUS")
print(f"{'='*W}")

exp_summary = raw.groupby('base_experiment').agg(
    total_rows=('experiment', 'count'),
    n_tickers=('ticker', 'nunique'),
    n_modes=('tpsl_mode', 'nunique'),
    n_seeds_total=('seed', 'nunique'),
    n_ensemble=('is_ensemble', 'sum'),
    n_survived_a=('survived_a', 'sum'),
    modes=('tpsl_mode', lambda x: sorted(x.unique())),
).sort_index()

print(f"\n  {'Base Experiment':<18} {'Rows':>7} {'Tickers':>8} {'Modes':>6} {'Seeds':>6} "
      f"{'Ensemble':>9} {'Surv_A':>7} {'Label':<30}")
print(f"  {'-'*100}")

for exp_id, row in exp_summary.iterrows():
    label = EXPECTED_EXPERIMENTS.get(exp_id, {}).get('label', '??? UNEXPECTED')
    print(f"  {exp_id:<18} {row['total_rows']:>7,} {row['n_tickers']:>8} {row['n_modes']:>6} "
          f"{row['n_seeds_total']:>6} {row['n_ensemble']:>9,} {row['n_survived_a']:>7,} {label:<30}")

observed_exps = set(exp_summary.index)
expected_exps = set(EXPECTED_EXPERIMENTS.keys())
unexpected = observed_exps - expected_exps
missing_entirely = expected_exps - observed_exps

if unexpected:
    print(f"\n  UNEXPECTED experiments in BQ (not in expected list):")
    for e in sorted(unexpected):
        r = exp_summary.loc[e]
        print(f"    {e}: {r['total_rows']:,} rows, {r['n_tickers']} tickers")

if missing_entirely:
    print(f"\n  MISSING experiments (expected but no data in BQ):")
    for e in sorted(missing_entirely):
        info = EXPECTED_EXPERIMENTS[e]
        print(f"    {e}: {info['label']} — {len(info['tickers'])} expected tickers, ZERO rows")


# ==============================================================================
# 3. EXPERIMENT × MODE MATRIX
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  SECTION 2: EXPERIMENT × MODE MATRIX (ticker count per cell)")
print(f"{'='*W}")

mode_matrix = raw.groupby(['base_experiment', 'tpsl_mode'])['ticker'].nunique().unstack(fill_value=0)

mode_order = [m for m in EXPECTED_MODES if m in mode_matrix.columns]
extra_modes = [m for m in mode_matrix.columns if m not in EXPECTED_MODES]
mode_order += extra_modes

print(f"\n  {'Experiment':<18}", end='')
for m in mode_order:
    print(f" {m[:8]:>8}", end='')
print(f" {'TOTAL':>6}")
print(f"  {'-'*(18 + 9*len(mode_order) + 7)}")

for exp_id in sorted(mode_matrix.index):
    print(f"  {exp_id:<18}", end='')
    for m in mode_order:
        v = int(mode_matrix.loc[exp_id, m]) if m in mode_matrix.columns else 0
        cell = f"{v}" if v > 0 else "—"
        print(f" {cell:>8}", end='')
    row_total = int(mode_matrix.loc[exp_id].sum())
    print(f" {row_total:>6}")

print(f"\n  Mode totals (unique tickers):")
for m in mode_order:
    if m in mode_matrix.columns:
        total_tickers = int(raw[raw['tpsl_mode'] == m]['ticker'].nunique())
        total_rows = int(len(raw[raw['tpsl_mode'] == m]))
        print(f"    {m:<18} {total_tickers:>3} tickers, {total_rows:>6,} rows")


# ==============================================================================
# 4. EXPERIMENT × TICKER MATRIX (mode count per cell)
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  SECTION 3: EXPERIMENT × TICKER COVERAGE (modes per ticker)")
print(f"{'='*W}")

all_observed_tickers = sorted(raw['ticker'].unique())
all_observed_exps = sorted(raw['base_experiment'].unique())

ticker_exp_modes = raw.groupby(['ticker', 'base_experiment'])['tpsl_mode'].nunique()

print(f"\n  Total unique tickers across all experiments: {len(all_observed_tickers)}")
print(f"\n  {'Ticker':<7}", end='')
for exp_id in all_observed_exps:
    label = exp_id.split('_')[0]
    print(f" {label:>5}", end='')
print(f" {'Exp#':>5}")
print(f"  {'-'*(7 + 6*len(all_observed_exps) + 6)}")

for ticker in all_observed_tickers:
    print(f"  {ticker:<7}", end='')
    n_exps = 0
    for exp_id in all_observed_exps:
        v = ticker_exp_modes.get((ticker, exp_id), 0)
        cell = f"{v}" if v > 0 else "—"
        print(f" {cell:>5}", end='')
        if v > 0:
            n_exps += 1
    print(f" {n_exps:>5}")


# ==============================================================================
# 5. LIMIT ENTRY DEEP DIVE
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  SECTION 4: LIMIT ENTRY DEEP DIVE")
print(f"{'='*W}")

limit_df = raw[raw['tpsl_mode'] == 'limit_entry'].copy()
print(f"\n  Total limit_entry rows: {len(limit_df):,}")

if len(limit_df) > 0:
    print(f"\n  Per-experiment limit_entry coverage:")
    print(f"  {'Experiment':<18} {'Rows':>7} {'Tickers':>8} {'Ens':>5} {'Indiv':>6} "
          f"{'Surv_A':>7} {'Avg_VA_PnL':>11} {'Tickers>=1.8%':>14}")
    print(f"  {'-'*85}")

    for exp_id in sorted(limit_df['base_experiment'].unique()):
        edf = limit_df[limit_df['base_experiment'] == exp_id]
        n_ens = int(edf['is_ensemble'].sum())
        n_indiv = int((~edf['is_ensemble']).sum())
        n_surv = int(edf['survived_a'].sum())
        avg_va = edf[edf['survived_a'] == True]['val_a_avg_pnl'].mean() if n_surv > 0 else 0
        n_deploy = len(edf[(edf['survived_a'] == True) &
                           (edf['is_ensemble'] == False)].groupby('ticker')
                       .filter(lambda x: x['val_a_avg_pnl'].mean() >= 0.018)
                       ['ticker'].unique()) if n_indiv > 0 else 0
        print(f"  {exp_id:<18} {len(edf):>7,} {edf['ticker'].nunique():>8} {n_ens:>5} "
              f"{n_indiv:>6} {n_surv:>7} {avg_va:>+10.2%} {n_deploy:>14}")

    print(f"\n  Per-ticker limit_entry availability (which experiments have it):")
    limit_tickers = sorted(limit_df['ticker'].unique())
    limit_exps = sorted(limit_df['base_experiment'].unique())

    print(f"  {'Ticker':<7}", end='')
    for exp_id in limit_exps:
        label = exp_id.split('_')[0]
        print(f" {label:>5}", end='')
    print(f" {'Exps':>5} {'Best_VA':>8}")
    print(f"  {'-'*(7 + 6*len(limit_exps) + 14)}")

    for ticker in limit_tickers:
        tdf = limit_df[limit_df['ticker'] == ticker]
        print(f"  {ticker:<7}", end='')
        n_exps = 0
        for exp_id in limit_exps:
            edf = tdf[tdf['base_experiment'] == exp_id]
            if len(edf) > 0:
                n_seeds = edf[edf['is_ensemble'] == False]['seed'].nunique()
                cell = f"{n_seeds}s"
                print(f" {cell:>5}", end='')
                n_exps += 1
            else:
                print(f" {'—':>5}", end='')
        best_va = tdf[tdf['survived_a'] == True]['val_a_avg_pnl'].max() if len(tdf[tdf['survived_a'] == True]) > 0 else 0
        print(f" {n_exps:>5} {best_va:>+7.2%}")

    print(f"\n  Limit entry seed agreement analysis (survived_a, individual seeds):")
    limit_indiv = limit_df[(limit_df['is_ensemble'] == False) & (limit_df['survived_a'] == True)].copy()
    if len(limit_indiv) > 0:
        seed_agree = limit_indiv.groupby(['base_experiment', 'ticker']).agg(
            n_seeds=('seed', 'nunique'),
            n_above_18=('val_a_avg_pnl', lambda x: (x >= 0.018).sum()),
            n_above_15=('val_a_avg_pnl', lambda x: (x >= 0.015).sum()),
            mean_pnl=('val_a_avg_pnl', 'mean'),
        ).reset_index()
        seed_agree['agree_18'] = seed_agree['n_above_18'] / seed_agree['n_seeds']
        seed_agree['agree_15'] = seed_agree['n_above_15'] / seed_agree['n_seeds']

        deployable = seed_agree[(seed_agree['n_seeds'] >= 3) &
                                (seed_agree['mean_pnl'] >= 0.018)]
        print(f"  {'Experiment':<18} {'Ticker':<7} {'Seeds':>6} {'>=1.8%':>7} {'>=1.5%':>7} "
              f"{'Agree18':>8} {'MeanPnL':>8}")
        print(f"  {'-'*68}")
        for _, r in deployable.sort_values(['base_experiment', 'mean_pnl'], ascending=[True, False]).iterrows():
            print(f"  {r['base_experiment']:<18} {r['ticker']:<7} {r['n_seeds']:>6} "
                  f"{r['n_above_18']:>7} {r['n_above_15']:>7} {r['agree_18']:>7.0%} "
                  f"{r['mean_pnl']:>+7.2%}")

        print(f"\n  Deployable limit_entry configs (>=3 seeds, mean PnL >= 1.8%): {len(deployable)}")
        print(f"  Unique tickers with deployable limit_entry: {deployable['ticker'].nunique()}")
        print(f"  Per-experiment:")
        for exp_id in sorted(deployable['base_experiment'].unique()):
            edf = deployable[deployable['base_experiment'] == exp_id]
            tks = sorted(edf['ticker'].tolist())
            print(f"    {exp_id}: {len(edf)} tickers — {tks}")
else:
    print(f"  NO limit_entry data found in BQ!")


# ==============================================================================
# 6. WALKFORWARD CHAIN AUDIT
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  SECTION 5: WALKFORWARD CHAIN COMPLETENESS")
print(f"{'='*W}")

for chain_name, fold_ids in WF_CHAINS.items():
    print(f"\n  --- Chain: {chain_name} ---")
    present = [f for f in fold_ids if f in observed_exps]
    missing = [f for f in fold_ids if f not in observed_exps]

    print(f"  Present folds: {len(present)}/{len(fold_ids)}")
    if missing:
        print(f"  MISSING folds: {missing}")

    if not present:
        print(f"  No data for this chain — skip")
        continue

    print(f"\n  {'Fold':<18} {'Rows':>7} {'Tickers':>8} {'Modes':>6} {'Seeds':>6} "
          f"{'Surv_A':>7} {'Limit?':>7}")
    print(f"  {'-'*65}")

    for fold_id in fold_ids:
        if fold_id in observed_exps:
            fdf = raw[raw['base_experiment'] == fold_id]
            has_limit = 'limit_entry' in fdf['tpsl_mode'].values
            n_surv = int(fdf['survived_a'].sum())
            print(f"  {fold_id:<18} {len(fdf):>7,} {fdf['ticker'].nunique():>8} "
                  f"{fdf['tpsl_mode'].nunique():>6} {fdf['seed'].nunique():>6} "
                  f"{n_surv:>7} {'YES' if has_limit else 'NO':>7}")
        else:
            print(f"  {fold_id:<18} {'— MISSING —':>50}")

    if len(present) >= 2:
        fold_tickers = {}
        for fold_id in present:
            fold_tickers[fold_id] = set(raw[raw['base_experiment'] == fold_id]['ticker'].unique())

        all_chain_tickers = sorted(set().union(*fold_tickers.values()))
        inconsistent = []
        for tk in all_chain_tickers:
            in_folds = [f for f in present if tk in fold_tickers[f]]
            if len(in_folds) != len(present):
                inconsistent.append((tk, in_folds))

        if inconsistent:
            print(f"\n  INCONSISTENT ticker coverage across folds:")
            for tk, folds in sorted(inconsistent):
                missing_folds = [f for f in present if f not in folds]
                print(f"    {tk:<7} present in {len(folds)}/{len(present)} folds, "
                      f"missing from: {[f.split('_')[0] for f in missing_folds]}")
        else:
            print(f"\n  All {len(all_chain_tickers)} tickers present in all {len(present)} folds ✓")

        fold_modes = {}
        for fold_id in present:
            fold_modes[fold_id] = set(raw[raw['base_experiment'] == fold_id]['tpsl_mode'].unique())
        all_chain_modes = sorted(set().union(*fold_modes.values()))
        mode_inconsistent = []
        for m in all_chain_modes:
            in_folds = [f for f in present if m in fold_modes[f]]
            if len(in_folds) != len(present):
                mode_inconsistent.append((m, in_folds))
        if mode_inconsistent:
            print(f"\n  INCONSISTENT mode coverage across folds:")
            for m, folds in mode_inconsistent:
                missing_folds = [f for f in present if f not in folds]
                print(f"    {m:<18} present in {len(folds)}/{len(present)} folds, "
                      f"missing from: {[f.split('_')[0] for f in missing_folds]}")


# ==============================================================================
# 7. SEED COVERAGE PER EXPERIMENT × TICKER × MODE
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  SECTION 6: SEED COVERAGE AUDIT")
print(f"{'='*W}")

indiv = raw[raw['is_ensemble'] == False].copy()

seed_counts = indiv.groupby(['base_experiment', 'ticker', 'tpsl_mode']).agg(
    n_seeds=('seed', 'nunique'),
    seeds_list=('seed', lambda x: sorted(x.unique())),
).reset_index()

incomplete_seeds = seed_counts[seed_counts['n_seeds'] < EXPECTED_SEEDS]
print(f"\n  Total experiment × ticker × mode combos: {len(seed_counts)}")
print(f"  Combos with < {EXPECTED_SEEDS} seeds: {len(incomplete_seeds)} ({len(incomplete_seeds)/len(seed_counts)*100:.1f}%)")

if len(incomplete_seeds) > 0:
    print(f"\n  Seed count distribution:")
    for n in sorted(seed_counts['n_seeds'].unique()):
        count = int((seed_counts['n_seeds'] == n).sum())
        pct = count / len(seed_counts) * 100
        bar = '#' * int(pct)
        print(f"    {n:>2} seeds: {count:>5} combos ({pct:>5.1f}%) {bar}")

    exp_seed_summary = seed_counts.groupby('base_experiment').agg(
        total_combos=('n_seeds', 'count'),
        full_15=('n_seeds', lambda x: (x == 15).sum()),
        partial=('n_seeds', lambda x: ((x > 0) & (x < 15)).sum()),
        min_seeds=('n_seeds', 'min'),
        avg_seeds=('n_seeds', 'mean'),
    ).reset_index()

    print(f"\n  Per-experiment seed completeness:")
    print(f"  {'Experiment':<18} {'Combos':>7} {'Full(15)':>9} {'Partial':>8} "
          f"{'Min':>4} {'Avg':>5} {'Complete%':>10}")
    print(f"  {'-'*65}")
    for _, r in exp_seed_summary.sort_index().iterrows():
        pct = r['full_15'] / r['total_combos'] * 100 if r['total_combos'] > 0 else 0
        print(f"  {r['base_experiment']:<18} {r['total_combos']:>7} {r['full_15']:>9} "
              f"{r['partial']:>8} {r['min_seeds']:>4} {r['avg_seeds']:>5.1f} {pct:>9.1f}%")

    worst = incomplete_seeds.sort_values('n_seeds')
    if len(worst) > 30:
        print(f"\n  Worst 30 incomplete combos (showing {min(30, len(worst))} of {len(worst)}):")
    else:
        print(f"\n  All incomplete combos:")
    print(f"  {'Experiment':<18} {'Ticker':<7} {'Mode':<16} {'Seeds':>6} {'Missing':>8}")
    print(f"  {'-'*60}")
    for _, r in worst.head(30).iterrows():
        missing = EXPECTED_SEEDS - r['n_seeds']
        print(f"  {r['base_experiment']:<18} {r['ticker']:<7} {r['tpsl_mode']:<16} "
              f"{r['n_seeds']:>6} {missing:>8}")


# ==============================================================================
# 8. ENSEMBLE COVERAGE
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  SECTION 7: ENSEMBLE COVERAGE")
print(f"{'='*W}")

ens = raw[raw['is_ensemble'] == True].copy()
print(f"\n  Total ensemble rows: {len(ens):,}")

ens_coverage = ens.groupby(['base_experiment', 'tpsl_mode'])['ticker'].nunique().unstack(fill_value=0)
indiv_coverage = indiv.groupby(['base_experiment', 'tpsl_mode'])['ticker'].nunique().unstack(fill_value=0)

print(f"\n  Experiments with individual seeds but NO ensemble:")
for exp_id in sorted(indiv_coverage.index):
    for mode in indiv_coverage.columns:
        n_indiv = int(indiv_coverage.loc[exp_id, mode]) if mode in indiv_coverage.columns else 0
        n_ens = int(ens_coverage.loc[exp_id, mode]) if (exp_id in ens_coverage.index and
                                                         mode in ens_coverage.columns) else 0
        if n_indiv > 0 and n_ens == 0:
            print(f"    {exp_id:<18} {mode:<16} {n_indiv} tickers with seeds, 0 ensembles")


# ==============================================================================
# 9. EXPECTED vs ACTUAL GAP REPORT
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  SECTION 8: GAP REPORT — EXPECTED vs ACTUAL")
print(f"{'='*W}")

print(f"\n  Checking each expected experiment against actual BQ data...")

gap_items = []
for exp_id, info in sorted(EXPECTED_EXPERIMENTS.items()):
    expected_tickers = set(info['tickers'])
    actual_tickers = set(raw[raw['base_experiment'] == exp_id]['ticker'].unique()) if exp_id in observed_exps else set()
    actual_modes = set(raw[raw['base_experiment'] == exp_id]['tpsl_mode'].unique()) if exp_id in observed_exps else set()

    missing_tickers = expected_tickers - actual_tickers
    extra_tickers = actual_tickers - expected_tickers
    missing_modes = set(EXPECTED_MODES) - actual_modes

    status = 'OK'
    if exp_id not in observed_exps:
        status = 'MISSING'
    elif missing_tickers or missing_modes:
        status = 'GAPS'
    else:
        status = 'COMPLETE'

    print(f"\n  {exp_id} ({info['label']}): {status}")
    print(f"    Expected tickers: {len(expected_tickers)}, Actual: {len(actual_tickers)}, "
          f"Missing: {len(missing_tickers)}")
    print(f"    Expected modes: {len(EXPECTED_MODES)}, Actual: {len(actual_modes)}, "
          f"Missing: {len(missing_modes)}")

    if missing_tickers and exp_id in observed_exps:
        print(f"    Missing tickers: {sorted(missing_tickers)}")
    if missing_modes and exp_id in observed_exps:
        print(f"    Missing modes: {sorted(missing_modes)}")
    if extra_tickers:
        print(f"    Extra tickers (not in expected list): {sorted(extra_tickers)}")

    if status in ('MISSING', 'GAPS'):
        gap_items.append({
            'experiment': exp_id,
            'label': info['label'],
            'status': status,
            'missing_tickers': len(missing_tickers),
            'missing_modes': len(missing_modes),
            'detail': f"tickers: {sorted(missing_tickers)}" if missing_tickers else f"modes: {sorted(missing_modes)}",
        })


# ==============================================================================
# 10. IMPACT ON backtest_tier_comparison.py
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  SECTION 9: IMPACT ON DEPLOYMENT SELECTION (backtest_tier_comparison.py)")
print(f"{'='*W}")

DEPLOY_EXPERIMENTS_CURRENT = {'152_XSECTOR', '160_XSECTOR', '170_XSECTOR', '185_XSECTOR'}

print(f"\n  Current DEPLOY_EXPERIMENTS filter: {sorted(DEPLOY_EXPERIMENTS_CURRENT)}")
print(f"  Rows matching current filter: ", end='')
current_match = raw[raw['base_experiment'].isin(DEPLOY_EXPERIMENTS_CURRENT)]
print(f"{len(current_match):,}")

all_possible = set(observed_exps)
print(f"  All experiments in BQ: {sorted(all_possible)}")
print(f"  Experiments EXCLUDED by current filter: {sorted(all_possible - DEPLOY_EXPERIMENTS_CURRENT)}")

excluded = raw[~raw['base_experiment'].isin(DEPLOY_EXPERIMENTS_CURRENT)]
print(f"  Excluded rows: {len(excluded):,}")

if len(excluded) > 0:
    print(f"\n  What's being excluded:")
    for exp_id in sorted(excluded['base_experiment'].unique()):
        edf = excluded[excluded['base_experiment'] == exp_id]
        has_limit = 'limit_entry' in edf['tpsl_mode'].values
        n_limit = len(edf[edf['tpsl_mode'] == 'limit_entry'])
        n_surv = int(edf['survived_a'].sum())
        surv_limit = edf[(edf['tpsl_mode'] == 'limit_entry') & (edf['survived_a'] == True)]
        n_surv_limit = len(surv_limit)
        n_deploy_limit = 0
        if n_surv_limit > 0:
            indiv_surv = surv_limit[surv_limit['is_ensemble'] == False]
            if len(indiv_surv) > 0:
                by_tk = indiv_surv.groupby('ticker')['val_a_avg_pnl'].mean()
                n_deploy_limit = int((by_tk >= 0.018).sum())
        print(f"    {exp_id:<18} {len(edf):>6,} rows, {edf['ticker'].nunique():>3} tickers, "
              f"limit={'YES' if has_limit else 'NO'} ({n_limit:>4} rows), "
              f"surv_a_limit={n_surv_limit:>4}, deployable_limit={n_deploy_limit:>3}")

    print(f"\n  Proposed expanded DEPLOY_EXPERIMENTS:")
    proposed = DEPLOY_EXPERIMENTS_CURRENT.copy()
    for exp_id in observed_exps:
        edf = raw[raw['base_experiment'] == exp_id]
        if len(edf) >= 100:
            proposed.add(exp_id)
    print(f"    {sorted(proposed)}")
    proposed_match = raw[raw['base_experiment'].isin(proposed)]
    print(f"    Would match: {len(proposed_match):,} rows (was {len(current_match):,})")

    print(f"\n  Limit entry deployable tickers by experiment (>=3 seeds, mean PnL >= 1.8%):")
    limit_surv = raw[(raw['tpsl_mode'] == 'limit_entry') &
                      (raw['is_ensemble'] == False) &
                      (raw['survived_a'] == True)].copy()
    if len(limit_surv) > 0:
        sa = limit_surv.groupby(['base_experiment', 'ticker']).agg(
            n_seeds=('seed', 'nunique'),
            mean_pnl=('val_a_avg_pnl', 'mean'),
            agree_18=('val_a_avg_pnl', lambda x: (x >= 0.018).sum() / len(x)),
        ).reset_index()
        deployable = sa[(sa['n_seeds'] >= 3) & (sa['mean_pnl'] >= 0.018)]

        print(f"\n  {'Experiment':<18} {'Ticker':<7} {'Seeds':>6} {'MeanPnL':>8} {'Agree18':>8} "
              f"{'In_Current?':>12}")
        print(f"  {'-'*65}")
        for _, r in deployable.sort_values(['base_experiment', 'mean_pnl'],
                                            ascending=[True, False]).iterrows():
            in_current = 'YES' if r['base_experiment'] in DEPLOY_EXPERIMENTS_CURRENT else 'EXCLUDED'
            print(f"  {r['base_experiment']:<18} {r['ticker']:<7} {r['n_seeds']:>6} "
                  f"{r['mean_pnl']:>+7.2%} {r['agree_18']:>7.0%} {in_current:>12}")

        print(f"\n  Summary:")
        for exp_id in sorted(deployable['base_experiment'].unique()):
            edf = deployable[deployable['base_experiment'] == exp_id]
            in_filter = 'INCLUDED' if exp_id in DEPLOY_EXPERIMENTS_CURRENT else 'EXCLUDED'
            tks = sorted(edf['ticker'].tolist())
            print(f"    {exp_id:<18} [{in_filter:>8}] {len(edf):>2} deployable tickers: {tks}")

        currently_deployable = set(deployable[deployable['base_experiment'].isin(
            DEPLOY_EXPERIMENTS_CURRENT)]['ticker'])
        all_deployable = set(deployable['ticker'])
        new_tickers = all_deployable - currently_deployable

        print(f"\n  Currently deployable tickers (limit_entry): {len(currently_deployable)} — "
              f"{sorted(currently_deployable)}")
        print(f"  Total deployable if all experiments included: {len(all_deployable)} — "
              f"{sorted(all_deployable)}")
        print(f"  NEW tickers from expanding filter: {len(new_tickers)} — "
              f"{sorted(new_tickers)}")


# ==============================================================================
# 11. MULTI-FAMILY AGREEMENT FOR LIMIT ENTRY
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  SECTION 10: MULTI-FAMILY LIMIT ENTRY AGREEMENT")
print(f"{'='*W}")

limit_surv_all = raw[(raw['tpsl_mode'] == 'limit_entry') &
                      (raw['is_ensemble'] == False) &
                      (raw['survived_a'] == True)].copy()

if len(limit_surv_all) > 0:
    sa_all = limit_surv_all.groupby(['base_experiment', 'ticker']).agg(
        n_seeds=('seed', 'nunique'),
        mean_pnl=('val_a_avg_pnl', 'mean'),
        agree_18=('val_a_avg_pnl', lambda x: (x >= 0.018).sum() / len(x)),
    ).reset_index()
    deployable_all = sa_all[(sa_all['n_seeds'] >= 3) & (sa_all['mean_pnl'] >= 0.018)]

    exp_families = {}
    for _, r in deployable_all.iterrows():
        exp_num = r['base_experiment'].split('_')[0]
        if exp_num.isdigit():
            num = int(exp_num)
            if 170 <= num <= 175:
                family = '170-175 WF'
            elif 180 <= num <= 185:
                family = '180-185 WF'
            elif num == 152:
                family = '152'
            elif num == 160:
                family = '160'
            elif num == 200:
                family = '200'
            else:
                family = exp_num
        else:
            family = exp_num
        exp_families.setdefault(r['ticker'], set()).add(family)

    print(f"\n  Per-ticker family agreement (deployable limit_entry, any experiment):")
    print(f"  {'Ticker':<7} {'Families':>2} {'Which families':<40}")
    print(f"  {'-'*55}")

    for ticker in sorted(exp_families.keys(), key=lambda t: -len(exp_families[t])):
        fams = sorted(exp_families[ticker])
        print(f"  {ticker:<7} {len(fams):>2}   {', '.join(fams):<40}")

    print(f"\n  Family agreement distribution:")
    for n in range(max(len(v) for v in exp_families.values()), 0, -1):
        tickers_n = [t for t, f in exp_families.items() if len(f) >= n]
        print(f"    >= {n} families: {len(tickers_n)} tickers — {sorted(tickers_n)}")

    print(f"\n  Proposed 'Any 2 of N families' pool (limit_entry, >= 2 family agreement):")
    any2 = [t for t, f in exp_families.items() if len(f) >= 2]
    print(f"    {len(any2)} tickers: {sorted(any2)}")

    only_1 = [t for t, f in exp_families.items() if len(f) == 1]
    if only_1:
        print(f"\n  Single-family-only tickers (limit_entry deployable in only 1 family):")
        for tk in sorted(only_1):
            fam = list(exp_families[tk])[0]
            tk_data = deployable_all[deployable_all['ticker'] == tk]
            best_pnl = tk_data['mean_pnl'].max()
            best_agree = tk_data['agree_18'].max()
            print(f"    {tk:<7} {fam:<15} pnl={best_pnl:+.2%} agree={best_agree:.0%}")


# ==============================================================================
# 12. ACTIONABLE FIX LIST
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  SECTION 11: ACTIONABLE FIX LIST")
print(f"{'='*W}")

fix_num = 0

if missing_entirely:
    for exp_id in sorted(missing_entirely):
        fix_num += 1
        info = EXPECTED_EXPERIMENTS[exp_id]
        print(f"\n  FIX {fix_num}: Run WF optimizer for {exp_id} ({info['label']})")
        print(f"    Impact: {len(info['tickers'])} tickers × 9 modes × 15 seeds")
        print(f"    Priority: {'HIGH' if 'deployed' in info['label'] or '185' in exp_id else 'MEDIUM'}")

excluded_with_limit = []
for exp_id in sorted(observed_exps - DEPLOY_EXPERIMENTS_CURRENT):
    edf = raw[raw['base_experiment'] == exp_id]
    if 'limit_entry' in edf['tpsl_mode'].values:
        excluded_with_limit.append(exp_id)

if excluded_with_limit:
    fix_num += 1
    print(f"\n  FIX {fix_num}: Add experiments to DEPLOY_EXPERIMENTS in backtest_tier_comparison.py")
    print(f"    Currently: {sorted(DEPLOY_EXPERIMENTS_CURRENT)}")
    print(f"    Add: {excluded_with_limit}")
    print(f"    These have limit_entry data that is being silently excluded from selection")
    print(f"    Priority: HIGH — zero code change, just config")

has_limit_gaps = []
for exp_id in sorted(observed_exps):
    edf = raw[raw['base_experiment'] == exp_id]
    if 'limit_entry' not in edf['tpsl_mode'].values:
        has_limit_gaps.append(exp_id)
if has_limit_gaps:
    fix_num += 1
    print(f"\n  FIX {fix_num}: Run limit_entry optimizer for experiments missing it")
    print(f"    Experiments without limit_entry: {has_limit_gaps}")
    print(f"    Priority: MEDIUM — these can't participate in limit_entry strategies")

if fix_num == 0:
    print(f"\n  No gaps found — all data is complete!")

print(f"\n{'='*W}")
print(f"  AUDIT COMPLETE — {fix_num} fixes identified")
print(f"{'='*W}")
