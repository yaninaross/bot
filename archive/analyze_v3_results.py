"""
analyze_v3_results.py — V3 Robust WF Optimization Results Analyzer
====================================================================
Pulls v3 (robust_wf_v3) results from BQ for clean experiments only.
Shows:
  TABLE 1: Data overview — tickers, seeds, modes per experiment
  TABLE 2: Best config per ticker (composite score) with TEST performance
  TABLE 3: TPSL mode comparison — which mode wins per ticker
  TABLE 4: Deploy-ready configs (OPT PnL ≥ 1.8% AND TEST PnL > 0)
  TABLE 5: Degradation analysis — OPT vs TEST PnL drop per ticker
  TABLE 6: Head-to-head — v3 vs older per-experiment optimizer results

Run in Colab: paste as cell, runs immediately.
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
BQ_TABLE = f"{GCP_PROJECT}.scoring_logs.wf_optimization_results"
DEPLOY_PNL_THRESHOLD = 0.018
TEST_PNL_THRESHOLD = 0.018
TEST_MONTHS = 1.7  # Apr 1 - May 21
MIN_TEST_TRADES_PER_MONTH = 1.0
MIN_TEST_TRADES = max(2, int(MIN_TEST_TRADES_PER_MONTH * TEST_MONTHS))
MIN_TRADES = 3

CLEAN_EXP_BASES = ('105_DEFENSE', '106_DEFENSE', '152_XSECTOR', '160_XSECTOR',
                   '170_XSECTOR', '171_XSECTOR', '172_XSECTOR',
                   '173_XSECTOR', '174_XSECTOR', '175_XSECTOR')

TICKER_RENAME = {'MIGI': 'BGDE', 'BITF': 'KEEL'}
EXCLUDE_TICKERS = {'TLT', 'VIX', 'SMLR', 'SPIR'}

MODE_LABELS = {
    'static': 'STATIC', 'atr': 'ATR', 'atr_adaptive': 'ATR_ADP',
    'regime_scaled': 'REGIME', 'trailing_decay': 'TRAIL',
}

W = 140

# ==============================================================================
# LOAD DATA
# ==============================================================================
print(f"{'='*W}")
print(f"  V4 ROBUST WF RESULTS ANALYZER — Clean Experiments Only")
print(f"  Deploy threshold: OPT PnL >= {DEPLOY_PNL_THRESHOLD:.1%}, TEST PnL >= {TEST_PNL_THRESHOLD:.1%}, "
      f"min TEST trades = {MIN_TEST_TRADES} ({MIN_TEST_TRADES_PER_MONTH:.0f}/mo × {TEST_MONTHS:.1f} mo)")
print(f"{'='*W}")

client = bigquery.Client()
query = f"SELECT * FROM `{BQ_TABLE}` ORDER BY experiment, ticker, seed"
try:
    df_all = client.query(query).to_dataframe()
except Exception as e:
    if 'Not found' in str(e) or 'notFound' in str(e):
        print(f"\n  Table {BQ_TABLE} does not exist yet.")
        raise SystemExit(0)
    raise

if df_all.empty:
    print("\n  No data in BQ table yet.")
    raise SystemExit(0)

df_all['ticker'] = df_all['ticker'].replace(TICKER_RENAME)
df_all = df_all[~df_all['ticker'].isin(EXCLUDE_TICKERS)]
df_all = df_all[df_all['experiment'].apply(
    lambda e: any(e == b or e.startswith(b + '_') for b in CLEAN_EXP_BASES))]

if 'model_version' not in df_all.columns:
    df_all['model_version'] = 'unknown'
df_all['model_version'] = df_all['model_version'].fillna('unknown')

if 'tpsl_mode' not in df_all.columns:
    df_all['tpsl_mode'] = 'static'
df_all['tpsl_mode'] = df_all['tpsl_mode'].fillna('static').replace('', 'static')

if 'vol_filter_pct' not in df_all.columns:
    df_all['vol_filter_pct'] = 0
df_all['vol_filter_pct'] = df_all['vol_filter_pct'].fillna(0).astype(int)

df_all['survived_a'] = df_all['survived_a'].fillna(False).astype(bool)
df_all['survived_b'] = df_all['survived_b'].fillna(False).astype(bool)
df_all['val_a_avg_pnl'] = df_all['val_a_avg_pnl'].fillna(0.0)
df_all['val_b_avg_pnl'] = df_all['val_b_avg_pnl'].fillna(0.0)
df_all['val_b_win_rate'] = df_all['val_b_win_rate'].fillna(0.0)
df_all['val_b_count'] = df_all['val_b_count'].fillna(0).astype(int)

if 'is_ensemble' not in df_all.columns:
    df_all['is_ensemble'] = False
df_all['is_ensemble'] = df_all['is_ensemble'].fillna(False).astype(bool)
if 'n_seeds_avg' not in df_all.columns:
    df_all['n_seeds_avg'] = 0
df_all['n_seeds_avg'] = df_all['n_seeds_avg'].fillna(0).astype(int)

df_all['composite'] = df_all['val_a_avg_pnl'] * df_all['val_a_win_rate'] * np.sqrt(df_all['val_a_count']) * np.log2(1 + df_all['val_a_count'])
df_all['run_timestamp'] = pd.to_datetime(df_all['run_timestamp'])

# Split by version
df_v3 = df_all[df_all['model_version'] == 'robust_wf_v4'].copy()
df_old = df_all[df_all['model_version'] != 'robust_wf_v4'].copy()

# Dedup v3: keep latest run per (experiment, ticker)
if not df_v3.empty:
    _latest = (df_v3.groupby(['experiment', 'ticker', 'run_id'])['run_timestamp']
               .max().reset_index()
               .sort_values('run_timestamp')
               .groupby(['experiment', 'ticker']).last()
               .reset_index()[['experiment', 'ticker', 'run_id']])
    df_v3 = df_v3.merge(_latest, on=['experiment', 'ticker', 'run_id'], how='inner')

# Dedup old: same logic
if not df_old.empty:
    _latest_old = (df_old.groupby(['experiment', 'ticker', 'run_id'])['run_timestamp']
                   .max().reset_index()
                   .sort_values('run_timestamp')
                   .groupby(['experiment', 'ticker']).last()
                   .reset_index()[['experiment', 'ticker', 'run_id']])
    df_old = df_old.merge(_latest_old, on=['experiment', 'ticker', 'run_id'], how='inner')

# Derive base experiment (strip TPSL mode suffix)
_TPSL_SUFFIXES = ('_ATRADP', '_TRAIL', '_REGIME', '_ATR')
def _base_exp(e):
    for sfx in _TPSL_SUFFIXES:
        if e.endswith(sfx):
            return e[:-len(sfx)]
    return e

df_v3['base_experiment'] = df_v3['experiment'].apply(_base_exp)

# Separate ensemble rows from per-seed rows
df_ens = df_v3[df_v3['is_ensemble']].copy()
df_v3 = df_v3[~df_v3['is_ensemble']].copy()

print(f"\n  Total clean rows: {len(df_all)}")
print(f"  V4 (robust_wf_v4): {len(df_v3)} seed rows + {len(df_ens)} ensemble rows")
print(f"  Older versions:    {len(df_old)} rows")

if df_v3.empty:
    print(f"\n  No v4 results yet. Optimizer may still be running.")
    print(f"  Re-run this script after some tickers complete.")
    if not df_old.empty:
        print(f"\n  Older results available for: {sorted(df_old['experiment'].unique().tolist())}")
    raise SystemExit(0)

# ==============================================================================
# TABLE 1: DATA OVERVIEW
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 1: V3 DATA OVERVIEW")
print(f"{'='*W}")
print(f"  {'Base Experiment':<20} {'Rows':>6} {'Tickers':>8} {'Seeds':>6} {'Modes':>6} "
      f"{'Run Date':>22}")
print(f"  {'─'*20} {'─'*6} {'─'*8} {'─'*6} {'─'*6} {'─'*22}")

for bexp in sorted(df_v3['base_experiment'].unique()):
    d = df_v3[df_v3['base_experiment'] == bexp]
    n_tickers = d['ticker'].nunique()
    n_seeds = d.groupby('ticker')['seed'].nunique().median()
    n_modes = d['tpsl_mode'].nunique()
    run_date = d['run_timestamp'].max().strftime('%Y-%m-%d %H:%M')
    print(f"  {bexp:<20} {len(d):>6} {n_tickers:>8} {n_seeds:>6.0f} {n_modes:>6} "
          f"{run_date:>22}")

print(f"\n  Tickers with v3 results: {sorted(df_v3['ticker'].unique().tolist())}")
print(f"  TPSL modes: {sorted(df_v3['tpsl_mode'].unique().tolist())}")

# ==============================================================================
# TABLE 2: BEST CONFIG PER TICKER (two-stage: composite within exp, TEST across)
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 2: BEST CONFIG PER TICKER — V4 (composite within experiment, TEST across)")
print(f"  Stage 1: best config per experiment (composite)  |  Stage 2: best experiment (TEST)")
print(f"{'='*W}")
print(f"  {'Ticker':<7} {'Experiment':<18} {'Mode':<8} {'Seed':>5} {'Entry':>6} {'Prob':>5} "
      f"{'TP%':>5} {'SL%':>5} {'VPct':>4} "
      f"{'|':>2} {'OPT%':>6} {'WR':>5} {'N':>4} "
      f"{'|':>2} {'TEST%':>7} {'WR':>5} {'N':>4} {'':>2} {'Status'}")
print(f"  {'─'*7} {'─'*18} {'─'*8} {'─'*5} {'─'*6} {'─'*5} "
      f"{'─'*5} {'─'*5} {'─'*4} "
      f"{'─'*2} {'─'*6} {'─'*5} {'─'*4} "
      f"{'─'*2} {'─'*7} {'─'*5} {'─'*4} {'─'*2} {'─'*8}")

deploy_ready = []
all_best = []

for ticker in sorted(df_v3['ticker'].unique()):
    td = df_v3[df_v3['ticker'] == ticker]
    surv = td[td['survived_a']]

    if surv.empty:
        print(f"  {ticker:<7} {'— no survivors —'}")
        continue

    # Stage 1: best config per base_experiment (composite within experiment)
    exp_winners = []
    for bexp in surv['base_experiment'].unique():
        exp_surv = surv[surv['base_experiment'] == bexp]
        eligible = exp_surv[exp_surv['val_a_count'] >= MIN_TRADES]
        if eligible.empty:
            eligible = exp_surv
        exp_best = eligible.loc[eligible['composite'].idxmax()]
        exp_winners.append(exp_best)

    # Stage 2: pick across experiments by TEST (deploy-eligible first, then best TEST)
    ew_df = pd.DataFrame(exp_winners)
    deploy_ew = ew_df[(ew_df['val_a_avg_pnl'] >= DEPLOY_PNL_THRESHOLD)
                      & (ew_df['val_b_avg_pnl'] >= TEST_PNL_THRESHOLD)
                      & (ew_df['val_b_count'] >= MIN_TEST_TRADES)]
    if not deploy_ew.empty:
        best = deploy_ew.loc[deploy_ew['val_b_avg_pnl'].idxmax()]
    else:
        best = ew_df.loc[ew_df['val_b_avg_pnl'].idxmax()]
    mode_label = MODE_LABELS.get(best['tpsl_mode'], best['tpsl_mode'][:6])

    opt_pnl = best['val_a_avg_pnl']
    test_pnl = best['val_b_avg_pnl']
    test_wr = best['val_b_win_rate']
    test_n = int(best['val_b_count'])

    deploy_ok = (opt_pnl >= DEPLOY_PNL_THRESHOLD and test_pnl >= TEST_PNL_THRESHOLD
                 and test_n >= MIN_TEST_TRADES)
    if deploy_ok:
        status = "DEPLOY"
    elif test_pnl <= 0:
        status = "TEST-"
    elif test_pnl < TEST_PNL_THRESHOLD:
        status = "T-LOW"
    elif test_n < MIN_TEST_TRADES:
        status = "T-FEW"
    else:
        status = "OPT-"

    if deploy_ok:
        deploy_ready.append(best)
    all_best.append(best)

    print(f"  {ticker:<7} {best['experiment']:<18} {mode_label:<8} {int(best['seed']):>5} "
          f"{best['entry_time']:>6} {best['prob_thresh']:>5.2f} "
          f"{best['tp_pct']:>5.2f} {best['sl_pct']:>5.2f} {int(best['vol_filter_pct']):>4} "
          f"{'|':>2} {opt_pnl:>5.1%} {best['val_a_win_rate']:>5.1%} {int(best['val_a_count']):>4} "
          f"{'|':>2} {test_pnl:>6.1%} {test_wr:>5.1%} {test_n:>4}  "
          f"{'✓' if deploy_ok else '✗'} {status}")

print(f"\n  Summary: {len(deploy_ready)}/{len(all_best)} tickers deploy-ready "
      f"(OPT≥{DEPLOY_PNL_THRESHOLD:.1%}, TEST≥{TEST_PNL_THRESHOLD:.1%}, N≥{MIN_TEST_TRADES})")

# ==============================================================================
# TABLE 3: TPSL MODE COMPARISON
# ==============================================================================
modes_present = sorted(df_v3['tpsl_mode'].unique())
if len(modes_present) > 1:
    print(f"\n{'='*W}")
    print(f"  TABLE 3: TPSL MODE COMPARISON — Best Seed Per Ticker-Mode")
    print(f"{'='*W}")

    header = f"  {'Ticker':<7}"
    for m in modes_present:
        ml = MODE_LABELS.get(m, m[:6])
        header += f" {'|':>2} {ml + ' OPT':>10} {ml + ' TEST':>10} {'N':>3}"
    header += f" {'|':>2} {'Best Mode':<10}"
    print(header)
    print(f"  {'─'*7}" + (f" {'─'*2} {'─'*10} {'─'*10} {'─'*3}" * len(modes_present))
          + f" {'─'*2} {'─'*10}")

    mode_wins = {m: 0 for m in modes_present}

    for ticker in sorted(df_v3['ticker'].unique()):
        td = df_v3[df_v3['ticker'] == ticker]
        line = f"  {ticker:<7}"
        best_mode = None
        best_composite = -999

        for m in modes_present:
            md = td[td['tpsl_mode'] == m]
            surv = md[md['survived_a']]
            if surv.empty:
                line += f" {'|':>2} {'—':>10} {'—':>10} {'—':>3}"
                continue
            b = surv.loc[surv['composite'].idxmax()]
            opt_s = f"{b['val_a_avg_pnl']:.1%}"
            test_s = f"{b['val_b_avg_pnl']:.1%}"
            line += f" {'|':>2} {opt_s:>10} {test_s:>10} {int(b['val_b_count']):>3}"
            if b['composite'] > best_composite:
                best_composite = b['composite']
                best_mode = m

        ml = MODE_LABELS.get(best_mode, '—') if best_mode else '—'
        line += f" {'|':>2} {ml:<10}"
        if best_mode:
            mode_wins[best_mode] += 1
        print(line)

    print(f"\n  Mode wins: ", end="")
    for m in modes_present:
        ml = MODE_LABELS.get(m, m)
        print(f"  {ml}={mode_wins[m]}", end="")
    print()

# ==============================================================================
# TABLE 4: DEPLOY-READY CONFIGS
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 4: DEPLOY-READY CONFIGS (OPT >= {DEPLOY_PNL_THRESHOLD:.1%}, TEST >= {TEST_PNL_THRESHOLD:.1%}, N >= {MIN_TEST_TRADES})")
print(f"{'='*W}")

if not deploy_ready:
    print(f"  No deploy-ready tickers yet.")
else:
    dr = pd.DataFrame(deploy_ready)
    dr = dr.sort_values('composite', ascending=False)

    print(f"  {'#':>3} {'Ticker':<7} {'Experiment':<18} {'Mode':<8} {'Seed':>5} {'Entry':>6} "
          f"{'Prob':>5} {'TP%':>5} {'SL%':>5} "
          f"{'|':>2} {'OPT%':>6} {'TEST%':>7} {'T-WR':>5} {'T-N':>4} "
          f"{'|':>2} {'Degrad':>7}")
    print(f"  {'─'*3} {'─'*7} {'─'*18} {'─'*8} {'─'*5} {'─'*6} "
          f"{'─'*5} {'─'*5} {'─'*5} "
          f"{'─'*2} {'─'*6} {'─'*7} {'─'*5} {'─'*4} "
          f"{'─'*2} {'─'*7}")

    for i, (_, r) in enumerate(dr.iterrows()):
        ml = MODE_LABELS.get(r['tpsl_mode'], r['tpsl_mode'][:6])
        opt_pnl = r['val_a_avg_pnl']
        test_pnl = r['val_b_avg_pnl']
        degrad = 1 - test_pnl / opt_pnl if opt_pnl > 0 else 0
        print(f"  {i+1:>3} {r['ticker']:<7} {r['experiment']:<18} {ml:<8} {int(r['seed']):>5} "
              f"{r['entry_time']:>6} {r['prob_thresh']:>5.2f} {r['tp_pct']:>5.2f} {r['sl_pct']:>5.2f} "
              f"{'|':>2} {opt_pnl:>5.1%} {test_pnl:>6.1%} {r['val_b_win_rate']:>5.1%} "
              f"{int(r['val_b_count']):>4} "
              f"{'|':>2} {degrad:>6.0%}")

    avg_opt = dr['val_a_avg_pnl'].mean()
    avg_test = dr['val_b_avg_pnl'].mean()
    avg_degrad = 1 - avg_test / avg_opt if avg_opt > 0 else 0
    print(f"\n  Averages: OPT={avg_opt:.1%}  TEST={avg_test:.1%}  "
          f"Degradation={avg_degrad:.0%}  Tickers={len(dr)}")

# ==============================================================================
# TABLE 5: OPT vs TEST DEGRADATION
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 5: OPT → TEST DEGRADATION (all tickers with survivors)")
print(f"{'='*W}")
print(f"  {'Ticker':<7} {'Mode':<8} {'OPT PnL':>8} {'TEST PnL':>9} {'Degrad':>8} "
      f"{'OPT N':>6} {'TEST N':>7} {'TEST WR':>8} {'Status':<10}")
print(f"  {'─'*7} {'─'*8} {'─'*8} {'─'*9} {'─'*8} {'─'*6} {'─'*7} {'─'*8} {'─'*10}")

if all_best:
    ab = pd.DataFrame(all_best).sort_values('val_a_avg_pnl', ascending=False)
    for _, r in ab.iterrows():
        opt_pnl = r['val_a_avg_pnl']
        test_pnl = r['val_b_avg_pnl']
        degrad = 1 - test_pnl / opt_pnl if opt_pnl > 0 else 0
        ml = MODE_LABELS.get(r['tpsl_mode'], r['tpsl_mode'][:6])

        t_n = int(r['val_b_count'])
        if (test_pnl >= TEST_PNL_THRESHOLD and opt_pnl >= DEPLOY_PNL_THRESHOLD
                and t_n >= MIN_TEST_TRADES):
            status = "DEPLOY"
        elif test_pnl >= TEST_PNL_THRESHOLD and t_n >= MIN_TEST_TRADES:
            status = "low-OPT"
        elif test_pnl > 0 and test_pnl < TEST_PNL_THRESHOLD:
            status = "T-LOW"
        elif test_pnl > 0 and t_n < MIN_TEST_TRADES:
            status = "T-FEW"
        elif test_pnl == 0 and t_n == 0:
            status = "no-trades"
        else:
            status = "TEST-FAIL"

        print(f"  {r['ticker']:<7} {ml:<8} {opt_pnl:>7.1%} {test_pnl:>8.1%} {degrad:>7.0%} "
              f"{int(r['val_a_count']):>6} {int(r['val_b_count']):>7} "
              f"{r['val_b_win_rate']:>7.1%} {status:<10}")

# ==============================================================================
# TABLE 6: V3 vs OLDER RESULTS (head-to-head)
# ==============================================================================
if not df_old.empty:
    v3_tickers = set(df_v3['ticker'].unique())
    old_tickers = set(df_old['ticker'].unique())
    overlap = sorted(v3_tickers & old_tickers)

    if overlap:
        print(f"\n{'='*W}")
        print(f"  TABLE 6: V3 vs OLDER OPTIMIZER — Head-to-Head ({len(overlap)} tickers)")
        print(f"{'='*W}")
        print(f"  {'Ticker':<7} "
              f"{'|':>2} {'Old Exp':<16} {'Old OPT':>8} {'Old VAL':>8} {'Old N':>6} "
              f"{'|':>2} {'V3 Exp':<16} {'V3 OPT':>7} {'V3 TEST':>8} {'V3 N':>5} "
              f"{'|':>2} {'Winner':<8}")
        print(f"  {'─'*7} "
              f"{'─'*2} {'─'*16} {'─'*8} {'─'*8} {'─'*6} "
              f"{'─'*2} {'─'*16} {'─'*7} {'─'*8} {'─'*5} "
              f"{'─'*2} {'─'*8}")

        for ticker in overlap:
            # Old best
            otd = df_old[df_old['ticker'] == ticker]
            o_surv = otd[otd['survived_a']]
            if o_surv.empty:
                continue
            o_best = o_surv.loc[(o_surv['val_a_avg_pnl'] * o_surv['val_a_win_rate']
                                  * np.sqrt(o_surv['val_a_count'])).idxmax()]

            # V3 best
            vtd = df_v3[df_v3['ticker'] == ticker]
            v_surv = vtd[vtd['survived_a']]
            if v_surv.empty:
                continue
            v_best = v_surv.loc[v_surv['composite'].idxmax()]

            o_val = o_best['val_a_avg_pnl']
            v_opt = v_best['val_a_avg_pnl']
            v_test = v_best['val_b_avg_pnl']

            v_test_n = int(v_best['val_b_count'])
            v3_deploy = (v_test >= TEST_PNL_THRESHOLD and v_opt >= DEPLOY_PNL_THRESHOLD
                         and v_test_n >= MIN_TEST_TRADES)
            winner = "V3" if v3_deploy else "OLD" if o_val >= DEPLOY_PNL_THRESHOLD else "NEITHER"

            print(f"  {ticker:<7} "
                  f"{'|':>2} {o_best['experiment']:<16} {o_val:>7.1%} "
                  f"{o_best.get('val_b_avg_pnl', 0):>7.1%} {int(o_best['val_a_count']):>6} "
                  f"{'|':>2} {v_best['experiment']:<16} {v_opt:>6.1%} "
                  f"{v_test:>7.1%} {int(v_best['val_b_count']):>5} "
                  f"{'|':>2} {winner:<8}")

# ==============================================================================
# TABLE 7: CROSS-FAMILY COMPARISON
# ==============================================================================
def _exp_family(exp_name):
    base = _base_exp(exp_name)
    if 'XSECTOR' not in base:
        return None
    if base.startswith('17'):
        return 'walkforward'
    if base == '160_XSECTOR':
        return 'early_cutoff'
    if base == '152_XSECTOR':
        return 'late_cutoff'
    return None

_FAMILY_LABELS = {'late_cutoff': 'Late (152)', 'early_cutoff': 'Early (160)', 'walkforward': 'WF (170)'}

df_v3['family'] = df_v3['experiment'].apply(_exp_family)
df_xsector = df_v3[df_v3['family'].notna()].copy()
families_present = sorted(df_xsector['family'].unique())

if len(families_present) > 1:
    print(f"\n{'='*W}")
    print(f"  TABLE 7: CROSS-FAMILY COMPARISON — XSECTOR Only (152 vs 160 vs 170)")
    print(f"{'='*W}")

    header = f"  {'Ticker':<7}"
    for fam in families_present:
        fl = _FAMILY_LABELS.get(fam, fam[:5])
        header += f" {'|':>2} {fl+' TEST':>9} {fl+' WR':>7} {fl+' N':>5}"
    header += f" {'|':>2} {'Best':>6}"
    print(header)

    sep = f"  {'─'*7}"
    for fam in families_present:
        sep += f" {'─'*2} {'─'*9} {'─'*7} {'─'*5}"
    sep += f" {'─'*2} {'─'*6}"
    print(sep)

    family_stats = {f: {'test_pnls': [], 'deploy': 0, 'positive': 0, 'total': 0} for f in families_present}

    for ticker in sorted(df_xsector['ticker'].unique()):
        td = df_xsector[df_xsector['ticker'] == ticker]
        line = f"  {ticker:<7}"
        best_fam = None
        best_test = -999.0

        for fam in families_present:
            fd = td[td['family'] == fam]
            surv = fd[fd['survived_a']]
            if surv.empty:
                line += f" {'|':>2} {'—':>9} {'—':>7} {'—':>5}"
                continue
            b = surv.loc[surv['composite'].idxmax()]
            t_pnl = b['val_b_avg_pnl']
            t_wr = b['val_b_win_rate']
            t_n = int(b['val_b_count'])
            line += f" {'|':>2} {t_pnl:>8.1%} {t_wr:>6.1%} {t_n:>5}"

            family_stats[fam]['test_pnls'].append(t_pnl)
            family_stats[fam]['total'] += 1
            if t_pnl > 0:
                family_stats[fam]['positive'] += 1
            if (t_pnl >= TEST_PNL_THRESHOLD and b['val_a_avg_pnl'] >= DEPLOY_PNL_THRESHOLD
                    and t_n >= MIN_TEST_TRADES):
                family_stats[fam]['deploy'] += 1

            if t_pnl > best_test:
                best_test = t_pnl
                best_fam = fam

        fl = _FAMILY_LABELS.get(best_fam, '—') if best_fam else '—'
        line += f" {'|':>2} {fl:>6}"
        print(line)

    print(f"\n  {'Family':<15} {'Avg TEST':>9} {'Deploy':>7} {'Positive':>9} {'Total':>6}")
    print(f"  {'─'*15} {'─'*9} {'─'*7} {'─'*9} {'─'*6}")
    for fam in families_present:
        s = family_stats[fam]
        avg_t = np.mean(s['test_pnls']) if s['test_pnls'] else 0
        fl = _FAMILY_LABELS.get(fam, fam)
        print(f"  {fl:<15} {avg_t:>8.1%} {s['deploy']:>7} "
              f"{s['positive']:>9} {s['total']:>6}")

# ==============================================================================
# TABLE 8: ENSEMBLE vs BEST SEED
# ==============================================================================
if not df_ens.empty:
    print(f"\n{'='*W}")
    print(f"  TABLE 8: ENSEMBLE vs BEST SEED — Head-to-Head")
    print(f"{'='*W}")
    print(f"  {'Ticker':<7} {'Experiment':<18} {'Mode':<8} "
          f"{'|':>2} {'Seed OPT':>9} {'Seed TEST':>10} {'S-WR':>5} {'S-N':>4} "
          f"{'|':>2} {'Ens OPT':>8} {'Ens TEST':>9} {'E-WR':>5} {'E-N':>4} {'#S':>3} "
          f"{'|':>2} {'Winner':<7}")
    print(f"  {'─'*7} {'─'*18} {'─'*8} "
          f"{'─'*2} {'─'*9} {'─'*10} {'─'*5} {'─'*4} "
          f"{'─'*2} {'─'*8} {'─'*9} {'─'*5} {'─'*4} {'─'*3} "
          f"{'─'*2} {'─'*7}")

    ens_wins = 0
    seed_wins = 0
    ties = 0

    for _, ens_row in df_ens.iterrows():
        ticker = ens_row['ticker']
        exp = ens_row['experiment']
        mode = ens_row['tpsl_mode']

        seed_rows = df_v3[(df_v3['ticker'] == ticker) & (df_v3['experiment'] == exp)
                          & (df_v3['tpsl_mode'] == mode) & df_v3['survived_a']]
        if seed_rows.empty:
            continue
        best_seed = seed_rows.loc[seed_rows['composite'].idxmax()]

        ml = MODE_LABELS.get(mode, mode[:6])
        s_opt = best_seed['val_a_avg_pnl']
        s_test = best_seed['val_b_avg_pnl']
        e_opt = ens_row['val_a_avg_pnl']
        e_test = ens_row['val_b_avg_pnl']
        n_seeds = int(ens_row['n_seeds_avg'])

        if e_test > s_test + 0.001:
            winner = "ENS"
            ens_wins += 1
        elif s_test > e_test + 0.001:
            winner = "SEED"
            seed_wins += 1
        else:
            winner = "TIE"
            ties += 1

        print(f"  {ticker:<7} {exp:<18} {ml:<8} "
              f"{'|':>2} {s_opt:>8.1%} {s_test:>9.1%} {best_seed['val_b_win_rate']:>5.1%} "
              f"{int(best_seed['val_b_count']):>4} "
              f"{'|':>2} {e_opt:>7.1%} {e_test:>8.1%} {ens_row['val_b_win_rate']:>5.1%} "
              f"{int(ens_row['val_b_count']):>4} {n_seeds:>3} "
              f"{'|':>2} {winner:<7}")

    print(f"\n  Ensemble wins: {ens_wins}  |  Seed wins: {seed_wins}  |  Ties: {ties}")

# ==============================================================================
# SUMMARY
# ==============================================================================
print(f"\n{'='*W}")
print(f"  SUMMARY")
print(f"{'='*W}")
print(f"  V3 results:     {df_v3['ticker'].nunique()} tickers, "
      f"{len(df_v3)} rows, {df_v3['tpsl_mode'].nunique()} modes")
print(f"  Deploy-ready:   {len(deploy_ready)} tickers "
      f"(OPT≥{DEPLOY_PNL_THRESHOLD:.1%}, TEST≥{TEST_PNL_THRESHOLD:.1%}, N≥{MIN_TEST_TRADES})")
if all_best:
    ab_df = pd.DataFrame(all_best)
    test_pos = ab_df[ab_df['val_b_avg_pnl'] > 0]
    print(f"  TEST positive:  {len(test_pos)}/{len(ab_df)} tickers")
    if not test_pos.empty:
        print(f"  Avg TEST PnL (positive only): {test_pos['val_b_avg_pnl'].mean():.1%}")
        print(f"  Avg TEST WR  (positive only): {test_pos['val_b_win_rate'].mean():.1%}")
print(f"  Experiments:    {sorted(df_v3['base_experiment'].unique().tolist())}")
print(f"{'='*W}")
