"""
analyze_200_comparison.py — Exp 200 (CNN-Transformer) vs 185/175 A/B Analysis
===============================================================================
Head-to-head comparison on IDENTICAL OPT/VAL/TEST windows:
  OPT: Jan 2 - Mar 31, 2026   VAL: Apr 1 - May 15   TEST: May 16 - Jun 17

200 = CNN-Transformer hybrid (new architecture)
185 = Standard Transformer (same cutoff, same data)
175 = Standard Transformer (walkforward last fold, same cutoff)

Sections:
  1. Data Census — what's in BQ per family
  2. Per-Mode Summary — 9 modes, 3 families side by side
  3. Head-to-Head Per Ticker — pivot best-mode test PnL, count wins
  4. Architecture Advantage — paired deltas, sector breakdown
  5. Limit Entry Deep Dive — dip_pct, configs, fill rates
  6. Walkforward vs Single Fold — 180-185 WF vs single 185 (context)
  7. Top Performers — deployable tickers from 200
  8. Backtest (optional) — trade sim with CSV predictions

Colab-ready. BQ-only for sections 1-7.
"""

import gc
import ctypes
import io
import sys
import subprocess
import warnings

import numpy as np
import pandas as pd

_libc = ctypes.CDLL("libc.so.6")
def _free_mem():
    gc.collect()
    _libc.malloc_trim(0)

warnings.filterwarnings('ignore')

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass

from google.cloud import bigquery

# ==============================================================================
# CONFIG
# ==============================================================================
BQ_TABLE       = 'ys-time-series.scoring_logs.wf_optimization_results'
MODEL_VERSION  = 'robust_wf_v12'
DEPLOY_THRESHOLD = 0.018
MIN_TRADES     = 3
RUN_BACKTEST   = False

W = 130

FAMILIES = {
    '200': {'label': '200 CNN-TX', 'match': lambda e: e.startswith('200_XSECTOR')},
    '185': {'label': '185 Single', 'match': lambda e: e.startswith('185_XSECTOR')},
    '175': {'label': '175 Single', 'match': lambda e: e.startswith('175_XSECTOR')},
}

CONTEXT_FAMILIES = {
    '180': {'label': '180-185 WF', 'match': lambda e: e.startswith('18') and e.split('_')[0].isdigit()
                                                        and 180 <= int(e.split('_')[0]) <= 185},
}

ALL_MODES = ['static', 'atr', 'atr_adaptive', 'regime_scaled', 'trailing_decay',
             'regime_adaptive', 'atr_sl', 'grace', 'limit_entry']

MODE_SUFFIX = {'static': '', 'atr': '_ATR', 'atr_adaptive': '_ATRADP',
               'regime_scaled': '_REGIME', 'trailing_decay': '_TRAIL',
               'regime_adaptive': '_REGADP', 'atr_sl': '_ATRSL',
               'grace': '_GRACE', 'limit_entry': '_LIMIT'}

SECTORS = {
    'crypto':   ['CIFR', 'IREN', 'HUT', 'KEEL', 'CLSK', 'APLD', 'WULF', 'MARA',
                 'RIOT', 'MSTR', 'BTBT', 'COIN', 'CORZ'],
    'space':    ['LUNR', 'RCAT', 'ASTS', 'RKLB', 'PL', 'RDW', 'BKSY', 'MNTS'],
    'evtol':    ['ACHR', 'JOBY'],
    'defense':  ['KTOS', 'AVAV', 'GD', 'GE', 'GEV', 'LHX', 'LMT', 'NOC', 'RTX', 'NNDM'],
    'energy':   ['OKLO', 'SMR', 'PLUG', 'FCEL', 'BE', 'FLNC', 'CCJ', 'CEG', 'COP',
                 'CVX', 'XOM', 'BHP', 'CAT'],
    'tech':     ['AMD', 'NVDA', 'TSLA', 'PLTR', 'SOUN', 'VRT', 'VST', 'SMCI', 'HOOD',
                 'MOD', 'POWL', 'ALAB', 'MP', 'NBIS'],
    'other':    ['BBAI', 'MVIS', 'BGDE', 'DUK', 'EQIX', 'ETN', 'LIN', 'NEE', 'VOYG'],
}

TICKER_TO_SECTOR = {}
for sec, tickers in SECTORS.items():
    for t in tickers:
        TICKER_TO_SECTOR[t] = sec

client = bigquery.Client()


# ==============================================================================
# 1. LOAD DATA
# ==============================================================================
print(f"\n{'='*W}")
print(f"  EXP 200 A/B COMPARISON — CNN-Transformer vs Standard Transformer")
print(f"  Windows: OPT Jan-Mar 2026 | VAL Apr-May | TEST May-Jun")
print(f"  Model version: {MODEL_VERSION}")
print(f"{'='*W}")

_EXP_FILTER = """(experiment LIKE '200_%' OR experiment LIKE '185_%'
       OR experiment LIKE '175_%'
       OR experiment LIKE '180_%' OR experiment LIKE '181_%'
       OR experiment LIKE '182_%' OR experiment LIKE '183_%'
       OR experiment LIKE '184_%')"""

TICKER_RENAME   = {'MIGI': 'BGDE', 'BITF': 'KEEL'}
EXCLUDE_TICKERS = {'TLT', 'VIX', 'SMLR', 'SPIR'}


def assign_family(exp):
    base = exp.split('_')[0]
    if not base.isdigit():
        return 'other'
    num = int(base)
    if num == 200: return '200'
    if num == 185: return '185'
    if num == 175: return '175'
    if 180 <= num <= 184: return '180'
    return 'other'


def _clean_df(df):
    df['ticker'] = df['ticker'].replace(TICKER_RENAME)
    df = df[~df['ticker'].isin(EXCLUDE_TICKERS)]
    df['tpsl_mode'] = df['tpsl_mode'].fillna('static')
    df['survived_b'] = df['survived_b'].fillna(False).astype(bool)
    df['survived_a'] = df['survived_a'].fillna(False).astype(bool)
    df['val_b_avg_pnl'] = df['val_b_avg_pnl'].fillna(0.0)
    df['val_b_win_rate'] = df['val_b_win_rate'].fillna(0.0)
    df['val_b_count'] = df['val_b_count'].fillna(0).astype(int)
    for col in ['dip_pct', 'grace_minutes', 'max_wait_minutes']:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    df['family'] = df['experiment'].apply(assign_family)
    df = df[df['family'] != 'other']
    return df


ens_query = f"""
SELECT experiment, ticker, seed, tpsl_mode, is_ensemble, n_seeds_avg,
       opt_avg_pnl, opt_win_rate, opt_count,
       val_a_avg_pnl, val_a_win_rate, val_a_count, val_a_freq,
       val_b_avg_pnl, val_b_win_rate, val_b_count, val_b_freq,
       survived_a, survived_b,
       prob_thresh, tp_pct, sl_pct, tp_mult, sl_mult,
       trail_start_mult, trail_end_mult, vol_filter_pct, pre_range_filter_pct,
       grace_minutes, dip_pct, max_wait_minutes,
       opt_start, opt_end, run_timestamp
FROM `{BQ_TABLE}`
WHERE model_version = '{MODEL_VERSION}'
  AND is_ensemble = TRUE
  AND {_EXP_FILTER}
"""
ens = _clean_df(client.query(ens_query).to_dataframe())
print(f"  Loaded {len(ens)} ensemble rows from BQ")

df_all_query = f"""
SELECT experiment, ticker, seed, tpsl_mode, is_ensemble,
       val_a_avg_pnl, val_b_avg_pnl, val_b_count,
       survived_a, survived_b
FROM `{BQ_TABLE}`
WHERE model_version = '{MODEL_VERSION}'
  AND {_EXP_FILTER}
"""
df_all = _clean_df(client.query(df_all_query).to_dataframe())
print(f"  Loaded {len(df_all)} total rows (lightweight) for census")
_free_mem()

FAM_ORDER = ['200', '185', '175']
FAM_LABELS = {'200': '200 CNN-TX', '185': '185 Std-TX', '175': '175 Std-TX', '180': '180-185 WF'}


# ==============================================================================
# SECTION 1: DATA CENSUS
# ==============================================================================
print(f"\n{'='*W}")
print(f"  SECTION 1: DATA CENSUS")
print(f"{'='*W}")

print(f"\n  {'Family':<14} {'Rows':>7} {'Seeds':>7} {'Tickers':>8} {'Modes':>6} "
      f"{'Ensemble':>9} {'limit_entry':>12}")
print(f"  {'─'*70}")

for fam in FAM_ORDER + ['180']:
    label = FAM_LABELS.get(fam, fam)
    fam_df = df_all[df_all['family'] == fam]
    fam_ind = fam_df[fam_df['is_ensemble'] != True]
    n_rows = len(fam_df)
    n_seeds = fam_ind['seed'].nunique()
    n_tickers = fam_df['ticker'].nunique()
    n_modes = fam_df['tpsl_mode'].nunique()
    n_ens = len(fam_df[fam_df['is_ensemble'] == True])
    n_limit = len(fam_df[fam_df['tpsl_mode'] == 'limit_entry'])
    print(f"  {label:<14} {n_rows:>7} {n_seeds:>7} {n_tickers:>8} {n_modes:>6} "
          f"{n_ens:>9} {n_limit:>12}")

missing_175 = set(ALL_MODES) - set(ens[ens['family'] == '175']['tpsl_mode'].unique())
if missing_175:
    print(f"\n  NOTE: 175 is missing modes: {', '.join(sorted(missing_175))}")
    print(f"        175 vs 200 comparison limited to: "
          f"{', '.join(sorted(set(ALL_MODES) - missing_175))}")

del df_all
_free_mem()


# ==============================================================================
# SECTION 2: PER-MODE SUMMARY (Ensemble Test PnL)
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  SECTION 2: PER-MODE SUMMARY (Ensemble Test PnL)")
print(f"  TEST period: val_b columns (holdout)")
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
for fam in FAM_ORDER:
    print(f"  | {FAM_LABELS[fam]:^30}", end='')
print()
print(f"  {'':─<18}", end='')
for _ in FAM_ORDER:
    print(f"──|{'':─^30}", end='')
print()
print(f"  {'':18}", end='')
for _ in FAM_ORDER:
    print(f"  | {'Tix':>4} {'PnL':>7} {'WR':>6} {'Pass%':>6} {'N':>5}", end='')
print()
print(f"  {'':─<18}", end='')
for _ in FAM_ORDER:
    print(f"──|{'':─^30}", end='')
print()

for mode in ALL_MODES:
    print(f"  {mode:<18}", end='')
    for fam in FAM_ORDER:
        row = mode_summary[(mode_summary['family'] == fam) & (mode_summary['tpsl_mode'] == mode)]
        if len(row) == 0:
            print(f"  | {'—':>4} {'—':>7} {'—':>6} {'—':>6} {'—':>5}", end='')
        else:
            r = row.iloc[0]
            print(f"  | {int(r['tickers']):>4} {r['avg_test_pnl']:>+6.2%} {r['avg_test_wr']:>5.1%} "
                  f"{r['pct_pass']:>5.0%} {r['avg_test_n']:>5.0f}", end='')
    print()

print(f"  {'':─<18}", end='')
for _ in FAM_ORDER:
    print(f"──|{'':─^30}", end='')
print()

print(f"  {'ALL MODES':<18}", end='')
for fam in FAM_ORDER:
    fam_ens = ens[ens['family'] == fam]
    if len(fam_ens) == 0:
        print(f"  | {'—':>4} {'—':>7} {'—':>6} {'—':>6} {'—':>5}", end='')
    else:
        print(f"  | {fam_ens['ticker'].nunique():>4} {fam_ens['val_b_avg_pnl'].mean():>+6.2%} "
              f"{fam_ens['val_b_win_rate'].mean():>5.1%} "
              f"{fam_ens['survived_b'].mean():>5.0%} {fam_ens['val_b_count'].mean():>5.0f}", end='')
print()


# ==============================================================================
# SECTION 3: HEAD-TO-HEAD PER TICKER (200 vs 185)
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  SECTION 3: HEAD-TO-HEAD PER TICKER (Ensemble, Best Mode Per Family)")
print(f"  For each ticker, picks the mode with highest test PnL per family")
print(f"{'='*W}")

best_per = ens.loc[ens.groupby(['family', 'ticker'])['val_b_avg_pnl'].idxmax()].copy()

pivot_pnl = best_per.pivot(index='ticker', columns='family', values='val_b_avg_pnl')
pivot_mode = best_per.pivot(index='ticker', columns='family', values='tpsl_mode')
pivot_wr = best_per.pivot(index='ticker', columns='family', values='val_b_win_rate')
pivot_n = best_per.pivot(index='ticker', columns='family', values='val_b_count')

all_tickers = sorted(pivot_pnl.index)

print(f"\n  Tickers with data: ", end='')
for fam in FAM_ORDER:
    if fam in pivot_pnl.columns:
        n = pivot_pnl[fam].dropna().shape[0]
        print(f"  {FAM_LABELS[fam]}={n}", end='')
print()

common_200_185 = sorted(pivot_pnl[['200', '185']].dropna().index) if '200' in pivot_pnl.columns and '185' in pivot_pnl.columns else []
print(f"  Common 200+185: {len(common_200_185)} tickers")

print(f"\n  {'Ticker':<7} {'Sector':<8}", end='')
for fam in ['200', '185']:
    print(f"  | {FAM_LABELS.get(fam, fam):^28}", end='')
print(f"  | {'Delta':>8} {'Winner':>10}")
print(f"  {'':─<7} {'':─<8}", end='')
for _ in ['200', '185']:
    print(f"──|{'':─^28}", end='')
print(f"──|{'':─^8} {'':─^10}")
print(f"  {'':7} {'':8}", end='')
for _ in ['200', '185']:
    print(f"  | {'Mode':>12} {'PnL':>7} {'WR':>5} {'N':>3}", end='')
print(f"  |")
print(f"  {'':─<7} {'':─<8}", end='')
for _ in ['200', '185']:
    print(f"──|{'':─^28}", end='')
print(f"──|{'':─^8} {'':─^10}")

wins = {'200': 0, '185': 0, 'tie': 0}

for ticker in all_tickers:
    sector = TICKER_TO_SECTOR.get(ticker, '?')[:7]
    print(f"  {ticker:<7} {sector:<8}", end='')
    pnl_vals = {}
    for fam in ['200', '185']:
        pnl = pivot_pnl.loc[ticker].get(fam) if fam in pivot_pnl.columns else None
        mode = pivot_mode.loc[ticker].get(fam) if fam in pivot_mode.columns else None
        wr = pivot_wr.loc[ticker].get(fam) if fam in pivot_wr.columns else None
        n = pivot_n.loc[ticker].get(fam) if fam in pivot_n.columns else None
        if pd.isna(pnl) or pnl is None:
            print(f"  | {'—':>12} {'—':>7} {'—':>5} {'—':>3}", end='')
        else:
            pnl_vals[fam] = pnl
            mode_short = (mode or 'static')[:12]
            print(f"  | {mode_short:>12} {pnl:>+6.2%} {wr:>4.0%} {int(n):>3}", end='')

    if '200' in pnl_vals and '185' in pnl_vals:
        delta = pnl_vals['200'] - pnl_vals['185']
        winner = '200' if delta > 0.001 else ('185' if delta < -0.001 else 'tie')
        wins[winner] = wins.get(winner, 0) + 1
        winner_label = FAM_LABELS.get(winner, winner)[:8]
        print(f"  | {delta:>+7.2%} {winner_label:>10}")
    else:
        print(f"  | {'—':>8} {'—':>10}")

print(f"  {'':─<7} {'':─<8}", end='')
for _ in ['200', '185']:
    print(f"──|{'':─^28}", end='')
print(f"──|{'':─^8} {'':─^10}")

print(f"\n  WINS: 200 CNN-TX={wins.get('200',0)}  185 Std-TX={wins.get('185',0)}  "
      f"Tie={wins.get('tie',0)}")

del pivot_mode, pivot_wr, pivot_n, best_per
_free_mem()


# ==============================================================================
# SECTION 4: ARCHITECTURE ADVANTAGE
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  SECTION 4: ARCHITECTURE ADVANTAGE — 200 vs 185 Paired Analysis")
print(f"  {len(common_200_185)} tickers with data in BOTH families (same windows)")
print(f"{'='*W}")

if common_200_185:
    common_pnl = pivot_pnl.loc[common_200_185]
    delta_series = common_pnl['200'] - common_pnl['185']

    print(f"\n  Overall (best mode per family):")
    print(f"    200 avg test PnL: {common_pnl['200'].mean():>+.3%}  med: {common_pnl['200'].median():>+.3%}")
    print(f"    185 avg test PnL: {common_pnl['185'].mean():>+.3%}  med: {common_pnl['185'].median():>+.3%}")
    print(f"    Delta (200-185):  avg={delta_series.mean():>+.3%}  med={delta_series.median():>+.3%}")
    print(f"    200 better: {(delta_series > 0).sum()}/{len(delta_series)}  "
          f"185 better: {(delta_series < 0).sum()}/{len(delta_series)}")

    # Sector breakdown
    print(f"\n  By sector:")
    print(f"  {'Sector':<10} {'N':>3} {'200 avg':>8} {'185 avg':>8} {'Delta':>8} {'200 wins':>9}")
    print(f"  {'─'*50}")
    for sec in sorted(SECTORS.keys()):
        sec_tickers = [t for t in common_200_185 if TICKER_TO_SECTOR.get(t) == sec]
        if not sec_tickers:
            continue
        sec_200 = common_pnl.loc[sec_tickers, '200']
        sec_185 = common_pnl.loc[sec_tickers, '185']
        sec_delta = sec_200 - sec_185
        n_wins = (sec_delta > 0).sum()
        print(f"  {sec:<10} {len(sec_tickers):>3} {sec_200.mean():>+7.2%} {sec_185.mean():>+7.2%} "
              f"{sec_delta.mean():>+7.2%} {n_wins}/{len(sec_tickers)}")

    # Seed stability comparison (query BQ on-demand to save RAM)
    print(f"\n  Seed Stability (per-seed test PnL std):")
    for fam_prefix in ['200', '185']:
        stab_q = f"""
        SELECT ticker, tpsl_mode, STDDEV(val_b_avg_pnl) as pnl_std
        FROM `{BQ_TABLE}`
        WHERE model_version = '{MODEL_VERSION}' AND is_ensemble = FALSE
          AND experiment LIKE '{fam_prefix}_%'
        GROUP BY ticker, tpsl_mode HAVING COUNT(*) >= 3
        """
        stab_df = client.query(stab_q).to_dataframe()
        if len(stab_df) == 0:
            continue
        print(f"    {FAM_LABELS[fam_prefix]}: avg_std={stab_df['pnl_std'].mean():.3%}  "
              f"med_std={stab_df['pnl_std'].median():.3%}  ({len(stab_df)} ticker-mode combos)")
        del stab_df

    # Top deltas
    print(f"\n  Largest 200 advantages:")
    top_200 = delta_series.nlargest(5)
    for t, d in top_200.items():
        print(f"    {t:<7} 200={common_pnl.loc[t,'200']:>+.2%}  185={common_pnl.loc[t,'185']:>+.2%}  delta={d:>+.2%}")

    print(f"\n  Largest 185 advantages:")
    top_185 = delta_series.nsmallest(5)
    for t, d in top_185.items():
        print(f"    {t:<7} 200={common_pnl.loc[t,'200']:>+.2%}  185={common_pnl.loc[t,'185']:>+.2%}  delta={d:>+.2%}")


# ==============================================================================
# SECTION 5: LIMIT_ENTRY DEEP DIVE
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  SECTION 5: LIMIT_ENTRY DEEP DIVE")
print(f"{'='*W}")

limit_ens = ens[ens['tpsl_mode'] == 'limit_entry'].copy()

for fam in FAM_ORDER:
    fam_limit = limit_ens[limit_ens['family'] == fam]
    if len(fam_limit) == 0:
        print(f"\n  {FAM_LABELS[fam]}: No limit_entry data")
        continue
    pos = fam_limit[fam_limit['val_b_avg_pnl'] > 0]
    deploy = fam_limit[fam_limit['val_b_avg_pnl'] >= DEPLOY_THRESHOLD]
    print(f"\n  {FAM_LABELS[fam]}:")
    print(f"    Tickers: {fam_limit['ticker'].nunique()}  "
          f"Positive test: {pos['ticker'].nunique()}  "
          f"Deployable (>={DEPLOY_THRESHOLD:.1%}): {deploy['ticker'].nunique()}")
    print(f"    Avg test PnL: {fam_limit['val_b_avg_pnl'].mean():>+.2%}  "
          f"Med: {fam_limit['val_b_avg_pnl'].median():>+.2%}")

# Limit dip_pct distribution (query BQ on-demand)
print(f"\n  Dip_pct distribution (per-seed survivors with test PnL > 0):")
dip_q = f"""
SELECT experiment, dip_pct, COUNT(*) as n
FROM `{BQ_TABLE}`
WHERE model_version = '{MODEL_VERSION}' AND is_ensemble = FALSE
  AND tpsl_mode = 'limit_entry' AND val_b_avg_pnl > 0
  AND dip_pct > 0
  AND {_EXP_FILTER}
GROUP BY experiment, dip_pct
"""
dip_df = client.query(dip_q).to_dataframe()
if len(dip_df) > 0:
    dip_df['family'] = dip_df['experiment'].apply(assign_family)
    fam_totals = dip_df.groupby('family')['n'].sum()
    all_dips = sorted(dip_df['dip_pct'].unique())
    print(f"  {'Dip%':<8}", end='')
    for fam in FAM_ORDER:
        print(f"  | {FAM_LABELS[fam]:^12}", end='')
    print()
    print(f"  {'─'*50}")
    for dip in all_dips:
        print(f"  {dip:>6.1%}  ", end='')
        for fam in FAM_ORDER:
            fam_dip = dip_df[(dip_df['family'] == fam) & (dip_df['dip_pct'] == dip)]
            if fam_dip.empty or fam not in fam_totals.index:
                print(f"  | {'—':^12}", end='')
            else:
                cnt = int(fam_dip['n'].sum())
                pct = cnt / fam_totals[fam] * 100
                print(f"  | {cnt:>4} ({pct:>4.0f}%)", end='')
        print()
    del dip_df

# Head-to-head limit_entry configs for common tickers
limit_common = set()
for fam in ['200', '185']:
    fam_tickers = set(limit_ens[limit_ens['family'] == fam]['ticker'].unique())
    if not limit_common:
        limit_common = fam_tickers
    else:
        limit_common &= fam_tickers

if limit_common:
    print(f"\n  Side-by-side limit_entry configs ({len(limit_common)} common tickers):")
    print(f"  {'Ticker':<7}  {'--- 200 CNN-TX ---':>30}  {'--- 185 Std-TX ---':>30}  {'Delta':>7}")
    print(f"  {'':7}   {'PnL':>7} {'WR':>5} {'N':>3} {'dip%':>5} {'prob':>5}  "
          f" {'PnL':>7} {'WR':>5} {'N':>3} {'dip%':>5} {'prob':>5}")
    print(f"  {'─'*80}")

    for t in sorted(limit_common):
        r200 = limit_ens[(limit_ens['family'] == '200') & (limit_ens['ticker'] == t)]
        r185 = limit_ens[(limit_ens['family'] == '185') & (limit_ens['ticker'] == t)]
        if r200.empty or r185.empty:
            continue
        r2 = r200.iloc[0]
        r1 = r185.iloc[0]
        delta = r2['val_b_avg_pnl'] - r1['val_b_avg_pnl']
        print(f"  {t:<7}  {r2['val_b_avg_pnl']:>+6.2%} {r2['val_b_win_rate']:>4.0%} "
              f"{int(r2['val_b_count']):>3} {r2['dip_pct']:>4.1%} {r2['prob_thresh']:>5.2f}  "
              f" {r1['val_b_avg_pnl']:>+6.2%} {r1['val_b_win_rate']:>4.0%} "
              f"{int(r1['val_b_count']):>3} {r1['dip_pct']:>4.1%} {r1['prob_thresh']:>5.2f}"
              f"  {delta:>+6.2%}")

del limit_ens
_free_mem()


# ==============================================================================
# SECTION 6: WALKFORWARD vs SINGLE FOLD (context)
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  SECTION 6: WALKFORWARD vs SINGLE FOLD (context, different TEST windows)")
print(f"  180-185 WF: TEST Apr-Jun | 185 Single: TEST May-Jun")
print(f"{'='*W}")

wf_ens = ens[ens['family'] == '180']
sf_ens = ens[ens['family'] == '185']

if len(wf_ens) > 0 and len(sf_ens) > 0:
    print(f"\n  {'Metric':<25} {'180-185 WF':>14} {'185 Single':>14}")
    print(f"  {'─'*55}")
    print(f"  {'Tickers':<25} {wf_ens['ticker'].nunique():>14} {sf_ens['ticker'].nunique():>14}")
    print(f"  {'Modes':<25} {wf_ens['tpsl_mode'].nunique():>14} {sf_ens['tpsl_mode'].nunique():>14}")
    print(f"  {'Avg test PnL':<25} {wf_ens['val_b_avg_pnl'].mean():>+13.2%} "
          f"{sf_ens['val_b_avg_pnl'].mean():>+13.2%}")
    print(f"  {'Med test PnL':<25} {wf_ens['val_b_avg_pnl'].median():>+13.2%} "
          f"{sf_ens['val_b_avg_pnl'].median():>+13.2%}")
    print(f"  {'Test pass rate':<25} {wf_ens['survived_b'].mean():>13.0%} "
          f"{sf_ens['survived_b'].mean():>13.0%}")

    wf_deploy = wf_ens[wf_ens['val_b_avg_pnl'] >= DEPLOY_THRESHOLD]
    sf_deploy = sf_ens[sf_ens['val_b_avg_pnl'] >= DEPLOY_THRESHOLD]
    print(f"  {'Deployable tickers':<25} {wf_deploy['ticker'].nunique():>14} "
          f"{sf_deploy['ticker'].nunique():>14}")

    # Best mode comparison
    print(f"\n  Best mode by avg test PnL:")
    for fam, fam_ens_df, label in [('180', wf_ens, '180-185 WF'), ('185', sf_ens, '185 Single')]:
        mode_avg = fam_ens_df.groupby('tpsl_mode')['val_b_avg_pnl'].mean().sort_values(ascending=False)
        best = mode_avg.index[0]
        print(f"    {label}: {best} ({mode_avg.iloc[0]:>+.2%})")
else:
    print(f"\n  Insufficient data for WF vs Single comparison")

del wf_ens, sf_ens
_free_mem()


# ==============================================================================
# SECTION 7: TOP PERFORMERS
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  SECTION 7: TOP PERFORMERS — 200 deployable tickers (test PnL >= {DEPLOY_THRESHOLD:.1%})")
print(f"{'='*W}")

deploy_200 = ens[(ens['family'] == '200') & (ens['val_b_avg_pnl'] >= DEPLOY_THRESHOLD)].copy()
if len(deploy_200) > 0:
    deploy_best = deploy_200.loc[deploy_200.groupby('ticker')['val_b_avg_pnl'].idxmax()]
    deploy_best = deploy_best.sort_values('val_b_avg_pnl', ascending=False)

    deploy_185_tickers = set(ens[(ens['family'] == '185') &
                                  (ens['val_b_avg_pnl'] >= DEPLOY_THRESHOLD)]['ticker'].unique())

    print(f"\n  {'Ticker':<7} {'Mode':<15} {'TestPnL':>8} {'TestWR':>7} {'TestN':>6} "
          f"{'Prob':>6} {'dip%':>5} {'New?':>5}")
    print(f"  {'─'*65}")
    n_new = 0
    for _, r in deploy_best.iterrows():
        is_new = r['ticker'] not in deploy_185_tickers
        if is_new:
            n_new += 1
        new_tag = "NEW" if is_new else ""
        dip_str = f"{r['dip_pct']:.1%}" if r['tpsl_mode'] == 'limit_entry' and r['dip_pct'] > 0 else "—"
        print(f"  {r['ticker']:<7} {r['tpsl_mode']:<15} {r['val_b_avg_pnl']:>+7.2%} "
              f"{r['val_b_win_rate']:>6.1%} {int(r['val_b_count']):>6} "
              f"{r['prob_thresh']:>6.2f} {dip_str:>5} {new_tag:>5}")
    print(f"  {'─'*65}")
    print(f"  {len(deploy_best)} deployable tickers ({n_new} NEW vs 185 baseline)")
else:
    print(f"\n  No tickers meet deploy threshold for exp 200")


# ==============================================================================
# SECTION 8: BACKTEST (optional)
# ==============================================================================
if RUN_BACKTEST:
    print(f"\n\n{'='*W}")
    print(f"  SECTION 8: BACKTEST — Top 200 configs with CSV model predictions")
    print(f"{'='*W}")

    try:
        import pytz
        from datetime import timedelta
        from google.cloud import storage as gcs_storage
        from numba import jit

        BUCKET_NAME = "time-series-bucket-ys"
        INFERENCE_DIR = "Data/Inference_v2"
        TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}
        EXIT_TIME_MIN = 780
        SLIPPAGE_PCT = 0.0015
        PST = pytz.timezone('America/Los_Angeles')

        BT_TEST_START = '2026-05-16'
        BT_TEST_END   = '2026-06-17'

        WALKFORWARD_FOLDS = {
            '200_XSECTOR': None,
            '185_XSECTOR': None,
            '175_XSECTOR': None,
        }

        gcs_client = gcs_storage.Client()

        def load_master(ticker, start_date=None):
            clean = TICKER_FILE_MAP.get(ticker, ticker)
            uri = f"gs://{BUCKET_NAME}/{INFERENCE_DIR}/{clean}_master_inference.parquet"
            try:
                df = pd.read_parquet(uri).reset_index()
                dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
                if dt_cols:
                    df['Datetime'] = pd.to_datetime(df[dt_cols[0]], utc=True)
                else:
                    nc = next((c for c in df.columns if c in ('Datetime','Date','Time','Timestamp')), df.columns[0])
                    df['Datetime'] = pd.to_datetime(df[nc], unit='ms', utc=True)
                df = df.set_index('Datetime').sort_index()
                df = df[~df.index.duplicated(keep='first')]
                for c in ['Open','High','Low','Close']:
                    if c in df.columns: df[c] = df[c].ffill()
                for c in ['Volume']:
                    if c in df.columns: df[c] = df[c].fillna(0)
                if start_date:
                    cutoff = pd.Timestamp(start_date, tz='UTC') - pd.Timedelta(days=30)
                    df = df[df.index >= cutoff]
                return df
            except Exception as e:
                print(f"    load_master({ticker}): FAILED — {e}")
                return None

        def load_seed_scores(ticker, experiment):
            clean = TICKER_FILE_MAP.get(ticker, ticker)
            results_dir = f"Algos/Results_all_v2/Universal_Exp{experiment}"
            prefix = f"{results_dir}/{clean}/Seed_"
            blobs = list(gcs_client.bucket(BUCKET_NAME).list_blobs(prefix=prefix))
            seed_series = {}
            for blob in blobs:
                name = blob.name.split('/')[-1]
                if not name.startswith('Seed_') or not name.endswith('_Results.csv'):
                    continue
                seed_id = int(name.replace('Seed_', '').replace('_Results.csv', ''))
                d = pd.read_csv(io.BytesIO(blob.download_as_bytes()), index_col=0, parse_dates=True)
                if d.index.tz is None:
                    d.index = d.index.tz_localize('UTC')
                d.index = d.index.tz_convert(PST)
                seed_series[seed_id] = d['Prob_Spike'].sort_index()
            return seed_series

        @jit(nopython=True)
        def run_single_trade_limit_bt(prices, highs, lows, date_ints, time_mins,
                                       entry_idx, tp, sl, exit_time, dip_pct, max_wait):
            n = len(prices)
            if entry_idx >= n - 1: return 0.0, date_ints[entry_idx], False
            sp = prices[entry_idx]; ed = date_ints[entry_idx]; et = time_mins[entry_idx]
            lp = sp * (1.0 - dip_pct)
            fill_idx = -1
            for k in range(entry_idx + 1, n):
                if date_ints[k] != ed: break
                if time_mins[k] - et > max_wait: break
                if lows[k] <= lp: fill_idx = k; break
            if fill_idx < 0: return 0.0, ed, False
            ep = lp; tgt = ep * (1.0 + tp); stp = ep * (1.0 - sl)
            for k in range(fill_idx + 1, n):
                if date_ints[k] != ed: return (prices[k-1] - ep) / ep, ed, True
                if lows[k] <= stp and highs[k] >= tgt:
                    return (-sl if np.random.random() < 0.5 else tp), ed, True
                if lows[k] <= stp: return -sl, ed, True
                if highs[k] >= tgt: return tp, ed, True
                if time_mins[k] >= exit_time: return (prices[k] - ep) / ep, ed, True
            return (prices[n-1] - ep) / ep, ed, True

        @jit(nopython=True)
        def run_single_trade_bt(prices, highs, lows, date_ints, time_mins,
                                entry_idx, tp, sl, exit_time):
            n = len(prices)
            if entry_idx >= n - 1: return 0.0, date_ints[entry_idx], False
            ep = prices[entry_idx]; ed = date_ints[entry_idx]
            tgt = ep * (1 + tp); stp = ep * (1 - sl)
            for k in range(entry_idx + 1, n):
                if date_ints[k] != ed: return (prices[k-1] - ep) / ep, ed, True
                if lows[k] <= stp and highs[k] >= tgt:
                    return (-sl if np.random.random() < 0.5 else tp), ed, True
                if lows[k] <= stp: return -sl, ed, True
                if highs[k] >= tgt: return tp, ed, True
                if time_mins[k] >= exit_time: return (prices[k] - ep) / ep, ed, True
            return (prices[n-1] - ep) / ep, ed, True

        bt_configs = []
        if len(deploy_best) > 0:
            for _, r in deploy_best.head(10).iterrows():
                base_exp = r['experiment']
                for sfx in sorted(MODE_SUFFIX.values(), key=len, reverse=True):
                    if sfx and base_exp.endswith(sfx):
                        base_exp = base_exp[:-len(sfx)]
                        break
                bt_configs.append({
                    'ticker': r['ticker'], 'experiment': base_exp,
                    'tpsl_mode': r['tpsl_mode'], 'entry_minute': 375,
                    'prob_thresh': float(r['prob_thresh']),
                    'tp_pct': float(r['tp_pct']), 'sl_pct': float(r['sl_pct']),
                    'dip_pct': float(r.get('dip_pct', 0)),
                    'max_wait_minutes': int(r.get('max_wait_minutes', 0)),
                })

        print(f"  Running backtest on {len(bt_configs)} tickers...")
        all_bt_trades = []

        for cfg in bt_configs:
            ticker = cfg['ticker']
            df_raw = load_master(ticker, start_date=BT_TEST_START)
            if df_raw is None:
                continue
            df_pst = df_raw.copy()
            df_pst.index = df_pst.index.tz_convert(PST)

            all_seeds = load_seed_scores(ticker, cfg['experiment'])
            if not all_seeds:
                print(f"    {ticker}: no seed scores")
                continue

            idx = df_pst.index
            probs = np.zeros(len(idx), dtype=np.float64)
            n_seeds_used = 0
            for s in sorted(all_seeds.keys()):
                common = idx.intersection(all_seeds[s].index)
                if len(common) == 0: continue
                locs = idx.get_indexer(common)
                probs[locs] += all_seeds[s].loc[common].values
                n_seeds_used += 1
            if n_seeds_used == 0: continue
            probs /= n_seeds_used

            times = (df_pst.index.hour * 60 + df_pst.index.minute).values
            s_ts = pd.Timestamp(BT_TEST_START, tz=PST)
            e_ts = pd.Timestamp(BT_TEST_END, tz=PST) + pd.Timedelta(days=1)
            dates = (df_pst.index.year*10000 + df_pst.index.month*100 + df_pst.index.day).values.astype(np.int64)
            prices = df_pst['Close'].values
            hi = df_pst['High'].values
            lo = df_pst['Low'].values

            trades = []
            last_d = -1
            uf = 0
            em = cfg['entry_minute']

            for i in range(len(idx)):
                if idx[i] < s_ts or idx[i] >= e_ts: continue
                if times[i] != em: continue
                if probs[i] < cfg['prob_thresh']: continue
                d_int = int(dates[i])
                if d_int == last_d: continue

                mode = cfg['tpsl_mode']
                tp, sl = cfg['tp_pct'], cfg['sl_pct']

                if mode == 'limit_entry':
                    pnl, td, valid = run_single_trade_limit_bt(
                        prices, hi, lo, dates, times.astype(np.int64), i,
                        tp, sl, EXIT_TIME_MIN, cfg['dip_pct'], cfg['max_wait_minutes'])
                    if not valid: uf += 1; last_d = d_int; continue
                else:
                    pnl, td, valid = run_single_trade_bt(
                        prices, hi, lo, dates, times.astype(np.int64), i,
                        tp, sl, EXIT_TIME_MIN)

                if valid:
                    pnl -= 2 * SLIPPAGE_PCT
                    trades.append({'ticker': ticker, 'pnl': float(pnl), 'date_int': int(td),
                                   'mode': mode})
                    last_d = int(td)

            w = sum(1 for t in trades if t['pnl'] > 0)
            l_count = len(trades) - w
            avg = np.mean([t['pnl'] for t in trades]) if trades else 0
            extra = f" unfilled={uf}" if uf > 0 else ""
            print(f"    {ticker:<7} {len(trades):>3} trades ({w}W/{l_count}L) "
                  f"avg={avg:>+.2%}  {cfg['tpsl_mode']}{extra}")
            all_bt_trades.extend(trades)
            del df_raw, df_pst; gc.collect()

        if all_bt_trades:
            bt_df = pd.DataFrame(all_bt_trades)
            bt_df['date'] = pd.to_datetime(bt_df['date_int'].astype(str), format='%Y%m%d')
            n = len(bt_df)
            w = int((bt_df['pnl'] > 0).sum())
            print(f"\n  PORTFOLIO: {bt_df['ticker'].nunique()} tickers, {n} trades "
                  f"({w}W/{n-w}L), avg PnL={bt_df['pnl'].mean():>+.2%}, "
                  f"WR={w/n:.0%}, total={bt_df['pnl'].sum():>+.1%}")
        else:
            print(f"\n  No backtest trades generated")

    except Exception as e:
        print(f"\n  Backtest failed: {e}")
        import traceback; traceback.print_exc()

else:
    print(f"\n\n  [Backtest disabled — set RUN_BACKTEST = True to enable]")

_free_mem()

# ==============================================================================
# VERDICT
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  VERDICT")
print(f"{'='*W}")

for fam in FAM_ORDER:
    fam_ens = ens[ens['family'] == fam]
    if len(fam_ens) == 0:
        print(f"  {FAM_LABELS[fam]:>14}: no data")
        continue
    best_mode_avg = fam_ens.groupby('tpsl_mode')['val_b_avg_pnl'].mean()
    best_mode = best_mode_avg.idxmax()
    deployable_n = fam_ens[fam_ens['val_b_avg_pnl'] >= DEPLOY_THRESHOLD]['ticker'].nunique()
    print(f"  {FAM_LABELS[fam]:>14}: avg_test={fam_ens['val_b_avg_pnl'].mean():>+.2%}  "
          f"best_mode={best_mode} ({best_mode_avg.max():>+.2%})  "
          f"deployable={deployable_n}/{fam_ens['ticker'].nunique()}")

if common_200_185:
    delta_avg = (pivot_pnl.loc[common_200_185, '200'] - pivot_pnl.loc[common_200_185, '185']).mean()
    direction = "outperforms" if delta_avg > 0 else "underperforms"
    print(f"\n  200 CNN-TX {direction} 185 Std-TX by {abs(delta_avg):.2%} avg on {len(common_200_185)} common tickers")
    print(f"  200 wins on {wins.get('200',0)}/{len(common_200_185)} tickers (best-mode comparison)")

print(f"\n{'='*W}")
