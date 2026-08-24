"""
analyze_wf_v12.py — v12 Walk-Forward Results Analyzer
======================================================
Queries BQ for model_version='robust_wf_v12' results and compares all 9 modes:
  static, atr, atr_adaptive, regime_scaled, trailing_decay,
  regime_adaptive, atr_sl, grace, limit_entry

Focus: TEST period (Apr-Jun 2026) performance under high-vol regime.

Tables:
  1. Mode overview (rows, tickers, OPT/VAL/TEST pass rates)
  2. Per-ticker mode comparison (best seed per mode, TEST PnL)
  3. Mode wins summary
  4. Per-ticker detail with params (all modes side by side)
  5. Aggregate stats by mode (OPT→VAL→TEST decay)
  6. v12 new modes vs existing modes head-to-head
  7. Pre-entry range filter impact
  8. Deploy recommendations (composite-scored)

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
MODEL_VERSION = "robust_wf_v12"
MIN_TRADES = 3
DEPLOY_PNL_THRESHOLD = 0.018

W = 160

TICKER_RENAME = {'MIGI': 'BGDE', 'BITF': 'KEEL'}
EXCLUDE_TICKERS = {'TLT', 'VIX', 'SMLR', 'SPIR'}

CLEAN_EXP_BASES = ('152_XSECTOR', '160_XSECTOR',
                    '170_XSECTOR', '171_XSECTOR', '172_XSECTOR',
                    '173_XSECTOR', '174_XSECTOR', '175_XSECTOR',
                    '180_XSECTOR', '181_XSECTOR', '182_XSECTOR',
                    '183_XSECTOR', '184_XSECTOR', '185_XSECTOR',
                    '200_XSECTOR')

MODE_ORDER = ['static', 'atr', 'atr_adaptive', 'regime_scaled', 'trailing_decay',
              'regime_adaptive', 'atr_sl', 'grace', 'limit_entry']
MODE_LABELS = {
    'static': 'STATIC',
    'atr': 'ATR',
    'atr_adaptive': 'ATR_ADP',
    'regime_scaled': 'REGIME',
    'trailing_decay': 'TRAIL',
    'regime_adaptive': 'REG_ADP',
    'atr_sl': 'ATR_SL',
    'grace': 'GRACE',
    'limit_entry': 'LIMIT',
}

V12_NEW_MODES = {'atr_sl', 'grace', 'limit_entry'}
V12_EXISTING_MODES = {'static', 'atr', 'atr_adaptive', 'regime_scaled', 'trailing_decay', 'regime_adaptive'}

# Set to a specific run_id to analyze a single run, or None for latest per (experiment, ticker)
RUN_ID_FILTER = None

# ==============================================================================
# LOAD DATA
# ==============================================================================
print(f"\n{'='*W}")
print(f"  v12 WALK-FORWARD RESULTS ANALYZER")
print(f"  model_version = '{MODEL_VERSION}'")
print(f"  Selection: composite = avg_pnl × win_rate × log2(1+N),  min N = {MIN_TRADES}")
print(f"{'='*W}")

client = bigquery.Client()

if RUN_ID_FILTER:
    query = f"""
        SELECT * FROM `{BQ_TABLE}`
        WHERE model_version = '{MODEL_VERSION}' AND run_id = '{RUN_ID_FILTER}'
        ORDER BY experiment, ticker, seed
    """
else:
    query = f"""
        SELECT * FROM `{BQ_TABLE}`
        WHERE model_version = '{MODEL_VERSION}'
        ORDER BY experiment, ticker, seed
    """

try:
    df = client.query(query).to_dataframe()
except Exception as e:
    if 'Not found' in str(e) or 'notFound' in str(e):
        print(f"\n  Table {BQ_TABLE} does not exist yet.")
        raise SystemExit(0)
    raise

if df.empty:
    print(f"\n  No data with model_version='{MODEL_VERSION}' in BQ table.")
    raise SystemExit(0)

df['ticker'] = df['ticker'].replace(TICKER_RENAME)
df = df[~df['ticker'].isin(EXCLUDE_TICKERS)]
df = df[df['experiment'].apply(lambda e: any(e == b or e.startswith(b + '_') for b in CLEAN_EXP_BASES))]

for col in ['survived_a', 'survived_b']:
    if col not in df.columns:
        df[col] = False
    df[col] = df[col].fillna(False).astype(bool)

for col in ['val_b_avg_pnl', 'val_b_win_rate', 'val_b_freq']:
    if col not in df.columns:
        df[col] = 0.0
    df[col] = df[col].fillna(0.0)

if 'val_b_count' not in df.columns:
    df['val_b_count'] = 0
df['val_b_count'] = df['val_b_count'].fillna(0).astype(int)

for col in ['vol_filter_pct', 'grace_minutes', 'max_wait_minutes']:
    if col not in df.columns:
        df[col] = 0
    df[col] = df[col].fillna(0).astype(int)

for col in ['tp_mult', 'sl_mult', 'trail_start_mult', 'trail_end_mult',
            'dip_pct', 'pre_range_filter_pct']:
    if col not in df.columns:
        df[col] = 0.0
    df[col] = df[col].fillna(0.0)

if 'tpsl_mode' not in df.columns:
    df['tpsl_mode'] = 'static'
df['tpsl_mode'] = df['tpsl_mode'].fillna('static').replace('', 'static')

df['val_a_total_pnl'] = df['val_a_avg_pnl'] * df['val_a_count']
df['val_b_total_pnl'] = df['val_b_avg_pnl'] * df['val_b_count']
df['composite_val'] = df['val_a_avg_pnl'] * df['val_a_win_rate'] * np.log2(1 + df['val_a_count'])
df['composite_test'] = df['val_b_avg_pnl'] * df['val_b_win_rate'] * np.log2(1 + df['val_b_count'])

df['run_timestamp'] = pd.to_datetime(df['run_timestamp'])

def _base_exp(exp_name):
    for sfx in ['_ATR', '_ATRADP', '_REGIME', '_TRAIL', '_REGADP', '_ATRSL', '_GRACE', '_LIMIT']:
        if exp_name.endswith(sfx):
            return exp_name[:-len(sfx)]
    return exp_name

df['base_experiment'] = df['experiment'].apply(_base_exp)

if not RUN_ID_FILTER:
    _latest = (df.groupby(['experiment', 'ticker', 'run_id'])['run_timestamp']
               .max().reset_index()
               .sort_values('run_timestamp')
               .groupby(['experiment', 'ticker']).last()
               .reset_index()[['experiment', 'ticker', 'run_id']])
    df = df.merge(_latest, on=['experiment', 'ticker', 'run_id'], how='inner')

modes_present = [m for m in MODE_ORDER if m in df['tpsl_mode'].unique()]
tickers = sorted(df['ticker'].unique())
new_modes_present = [m for m in modes_present if m in V12_NEW_MODES]
existing_modes_present = [m for m in modes_present if m in V12_EXISTING_MODES]

_ticker_exp = (df.groupby('ticker')['base_experiment']
               .agg(lambda x: x.value_counts().index[0])
               .to_dict())

run_ids = df['run_id'].unique()
print(f"\n  Run IDs: {len(run_ids)}")
for rid in run_ids[:5]:
    ts = df[df['run_id'] == rid]['run_timestamp'].max()
    n_rows = len(df[df['run_id'] == rid])
    print(f"    {rid[:12]}...  {ts}  ({n_rows} rows)")

print(f"\n  Modes found: {', '.join(MODE_LABELS.get(m, m) for m in modes_present)}")
print(f"  New v12 modes: {', '.join(MODE_LABELS.get(m, m) for m in new_modes_present) or 'none'}")
print(f"  Tickers: {len(tickers)}")
print(f"  Total rows: {len(df):,}")
has_test = df['val_b_count'].sum() > 0
print(f"  Has TEST data: {'YES' if has_test else 'NO'}")

# ==============================================================================
# TABLE 1: MODE OVERVIEW
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 1: MODE OVERVIEW")
print(f"{'='*W}")
print(f"  {'Mode':<10} {'Rows':>6} {'Tkrs':>5} {'Seeds':>6} "
      f"{'PassVAL':>8} {'%':>6} {'PassTST':>8} {'%':>6} "
      f"{'Avg OPT':>8} {'Avg VAL':>8} {'Avg TST':>8} {'OPT→VAL':>8} {'VAL→TST':>8}")
print(f"  {'─'*130}")

for mode in modes_present:
    md = df[df['tpsl_mode'] == mode]
    n = len(md)
    n_tickers = md['ticker'].nunique()
    n_seeds = md['seed'].nunique()
    pass_a = int(md['survived_a'].sum())
    pass_b = int(md['survived_b'].sum())
    pct_a = pass_a / n * 100 if n > 0 else 0
    pct_b = pass_b / n * 100 if n > 0 else 0
    avg_opt = md['opt_avg_pnl'].mean()
    avg_val = md['val_a_avg_pnl'].mean()
    avg_test = md['val_b_avg_pnl'].mean()
    decay_ov = (avg_val - avg_opt) / avg_opt * 100 if avg_opt != 0 else 0
    decay_vt = (avg_test - avg_val) / avg_val * 100 if avg_val != 0 else 0
    label = MODE_LABELS.get(mode, mode)
    print(f"  {label:<10} {n:>6} {n_tickers:>5} {n_seeds:>6} "
          f"{pass_a:>4}/{n:<3} {pct_a:>5.1f}% {pass_b:>4}/{n:<3} {pct_b:>5.1f}% "
          f"{avg_opt:>7.2%} {avg_val:>7.2%} {avg_test:>7.2%} {decay_ov:>7.1f}% {decay_vt:>7.1f}%")

# ==============================================================================
# TABLE 2: PER-TICKER MODE COMPARISON ON TEST PERIOD
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 2: PER-TICKER MODE COMPARISON — TEST PERIOD (best seed per mode)")
print(f"{'='*W}")

use_test = has_test
score_col = 'composite_test' if use_test else 'composite_val'
pnl_col = 'val_b_avg_pnl' if use_test else 'val_a_avg_pnl'
wr_col = 'val_b_win_rate' if use_test else 'val_a_win_rate'
cnt_col = 'val_b_count' if use_test else 'val_a_count'
surv_col = 'survived_b' if use_test else 'survived_a'
period_label = 'TEST' if use_test else 'VAL'

mode_cols = '  '.join(f"{'── ' + MODE_LABELS.get(m, m) + ' ──':>16}" for m in modes_present)
print(f"  {'Ticker':<7} {'Exp':<14} {mode_cols}")
sub_hdr = '  '.join(f"{'PnL':>6} {'WR':>4} {'N':>3}" for _ in modes_present)
print(f"  {'':7} {'':14} {sub_hdr}")
print(f"  {'─'*W}")

ticker_winners = {}
ticker_best_rows = {}

for ticker in tickers:
    td = df[df['ticker'] == ticker]
    parts = []
    best_mode = None
    best_composite = -999.0

    for mode in modes_present:
        md = td[td['tpsl_mode'] == mode]
        if md.empty:
            parts.append(f"{'---':>6} {'--':>4} {'--':>3}")
            continue

        surv = md[md['survived_a']]
        eligible = surv[surv['val_a_count'] >= MIN_TRADES]
        if not eligible.empty:
            best = eligible.loc[eligible[score_col].idxmax()]
            tag = "P"
        elif not surv.empty:
            best = surv.loc[surv[score_col].idxmax()]
            tag = "p"
        else:
            best = md.loc[md[score_col].idxmax()]
            tag = "F"

        pnl = best[pnl_col]
        wr = best[wr_col]
        n_t = int(best[cnt_col])
        comp = best[score_col]

        if tag != "F" and comp > best_composite:
            best_composite = comp
            best_mode = mode

        parts.append(f"{pnl:>5.1%} {wr:>3.0%} {n_t:>3}")

    ticker_winners[ticker] = best_mode
    line = '  '.join(parts)
    winner_mark = f"  ← {MODE_LABELS.get(best_mode, '?')}" if best_mode else ""
    exp_name = _ticker_exp.get(ticker, '?')
    print(f"  {ticker:<7} {exp_name:<14} {line}{winner_mark}")

# ==============================================================================
# TABLE 3: MODE WINS SUMMARY
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 3: MODE WINS ({period_label} period — which mode is best per ticker)")
print(f"{'='*W}")

win_counts = {}
for mode in modes_present:
    win_counts[mode] = sum(1 for v in ticker_winners.values() if v == mode)
n_none = sum(1 for v in ticker_winners.values() if v is None)

print(f"\n  A) ALL MODES ({len(tickers)} tickers)")
print(f"  {'Mode':<10} {'Wins':>6} {'%':>7}  {'Category':>10}")
print(f"  {'─'*40}")
for mode in modes_present:
    cnt = win_counts[mode]
    pct = cnt / len(tickers) * 100 if len(tickers) > 0 else 0
    cat = "NEW" if mode in V12_NEW_MODES else "existing"
    print(f"  {MODE_LABELS.get(mode, mode):<10} {cnt:>6} {pct:>6.1f}%  {cat:>10}")
if n_none > 0:
    print(f"  {'No winner':<10} {n_none:>6} {n_none/len(tickers)*100:>6.1f}%")

new_wins = sum(win_counts.get(m, 0) for m in V12_NEW_MODES)
old_wins = sum(win_counts.get(m, 0) for m in V12_EXISTING_MODES)
total_wins = new_wins + old_wins
print(f"\n  B) NEW vs EXISTING")
print(f"  {'v12 NEW':<10} {new_wins:>6} {new_wins/total_wins*100 if total_wins else 0:>6.1f}%  ({', '.join(MODE_LABELS.get(m,m) for m in new_modes_present)})")
print(f"  {'EXISTING':<10} {old_wins:>6} {old_wins/total_wins*100 if total_wins else 0:>6.1f}%")

# ==============================================================================
# TABLE 4: PER-TICKER DETAIL WITH PARAMS
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 4: PER-TICKER DETAIL (best seed params per mode, {period_label} period)")
print(f"{'='*W}")

for ticker in tickers:
    td = df[df['ticker'] == ticker]
    modes_for_ticker = [m for m in modes_present if m in td['tpsl_mode'].values]
    if not modes_for_ticker:
        continue

    exp_name = _ticker_exp.get(ticker, '?')
    print(f"\n  {ticker}  ({exp_name})")
    print(f"  {'Mode':<8} {'Entry':>5} {'Prob':>5} {'TP/SL Params':<28} {'VolF':>4} {'PreR':>4} "
          f"{'|':>1} {'OPT%':>6} {'VAL%':>6} {'TST%':>6} {'WR_T':>5} {'N_T':>4} {'Comp':>9} {'Tag':>4}")
    print(f"  {'─'*120}")

    for mode in modes_for_ticker:
        md = td[td['tpsl_mode'] == mode]
        surv = md[md['survived_a']]
        eligible = surv[surv['val_a_count'] >= MIN_TRADES]
        if not eligible.empty:
            best = eligible.loc[eligible[score_col].idxmax()]
            tag = "PASS"
        elif not surv.empty:
            best = surv.loc[surv[score_col].idxmax()]
            tag = "pass"
        else:
            best = md.loc[md[score_col].idxmax()]
            tag = "FAIL"

        if mode in ('atr', 'atr_adaptive'):
            tp_sl_str = f"tp_m={best['tp_mult']:.1f}  sl_m={best['sl_mult']:.1f}"
        elif mode == 'trailing_decay':
            tp_sl_str = f"ts={best['trail_start_mult']:.1f} te={best['trail_end_mult']:.1f} sl={best['sl_mult']:.1f}"
        elif mode == 'atr_sl':
            tp_sl_str = f"TP={best['tp_pct']:.0%}  sl_m={best['sl_mult']:.1f}"
        elif mode == 'grace':
            tp_sl_str = f"TP={best['tp_pct']:.0%}  SL={best['sl_pct']:.0%}  grace={int(best['grace_minutes'])}m"
        elif mode == 'limit_entry':
            tp_sl_str = f"TP={best['tp_pct']:.0%}  SL={best['sl_pct']:.0%}  dip={best['dip_pct']:.1%} w={int(best['max_wait_minutes'])}m"
        elif mode == 'regime_adaptive':
            tp_sl_str = f"TP={best.get('regime_tp_pct', 0):.0%}  SL={best.get('regime_sl_pct', 0):.0%}"
        else:
            tp_sl_str = f"TP={best['tp_pct']:.0%}  SL={best['sl_pct']:.0%}"

        vol_f = int(best.get('vol_filter_pct', 0))
        pre_r = int(best.get('pre_range_filter_pct', 0))
        label = MODE_LABELS.get(mode, mode)
        winner = " ←" if ticker_winners.get(ticker) == mode else ""
        test_pnl = best['val_b_avg_pnl']
        test_wr = best['val_b_win_rate']
        test_n = int(best['val_b_count'])
        print(f"  {label:<8} {best['entry_time']:>5} {best['prob_thresh']:>5.3f} {tp_sl_str:<28} {vol_f:>4} {pre_r:>4} "
              f"{'|':>1} {best['opt_avg_pnl']:>5.2%} {best['val_a_avg_pnl']:>5.2%} {test_pnl:>5.2%} "
              f"{test_wr:>4.0%} {test_n:>4} "
              f"{best[score_col]:>9.6f} {tag:>4}{winner}")

# ==============================================================================
# TABLE 5: AGGREGATE STATS BY MODE
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 5: AGGREGATE STATS BY MODE (VAL survivors, {period_label} performance)")
print(f"{'='*W}")
print(f"  {'Mode':<10} {'Cat':>4} {'Seeds':>6} {'Avg OPT':>8} {'Avg VAL':>8} {'Avg TST':>8} "
      f"{'O→V':>6} {'V→T':>6} {'Med TST':>8} {'WR_T':>6} {'Avg N_T':>6} {'Tot PnL':>8}")
print(f"  {'─'*100}")

for mode in modes_present:
    md = df[(df['tpsl_mode'] == mode) & df['survived_a']]
    if md.empty:
        label = MODE_LABELS.get(mode, mode)
        cat = "NEW" if mode in V12_NEW_MODES else "old"
        print(f"  {label:<10} {cat:>4}      0     ---      ---      ---    ---    ---      ---    ---    ---      ---")
        continue

    n_s = len(md)
    avg_opt = md['opt_avg_pnl'].mean()
    avg_val = md['val_a_avg_pnl'].mean()
    avg_test = md['val_b_avg_pnl'].mean()
    decay_ov = (avg_val - avg_opt) / avg_opt * 100 if avg_opt != 0 else 0
    decay_vt = (avg_test - avg_val) / avg_val * 100 if avg_val != 0 else 0
    med_test = md['val_b_avg_pnl'].median()
    avg_wr = md['val_b_win_rate'].mean()
    avg_n = md['val_b_count'].mean()
    tot_pnl = md['val_b_total_pnl'].sum()
    label = MODE_LABELS.get(mode, mode)
    cat = "NEW" if mode in V12_NEW_MODES else "old"
    print(f"  {label:<10} {cat:>4} {n_s:>6} {avg_opt:>7.2%} {avg_val:>7.2%} {avg_test:>7.2%} "
          f"{decay_ov:>5.1f}% {decay_vt:>5.1f}% "
          f"{med_test:>7.2%} {avg_wr:>5.1%} {avg_n:>6.1f} {tot_pnl:>7.2%}")

# ==============================================================================
# TABLE 6: v12 NEW vs EXISTING — HEAD-TO-HEAD PER TICKER
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 6: v12 NEW vs EXISTING — HEAD-TO-HEAD ({period_label} period)")
print(f"{'='*W}")

if new_modes_present and existing_modes_present:
    print(f"  {'Ticker':<7} {'Exp':<14} {'Best NEW':<10} {'PnL_N':>7} {'WR_N':>5} "
          f"{'Best OLD':<10} {'PnL_O':>7} {'WR_O':>5} {'Delta':>7} {'Winner':>7}")
    print(f"  {'─'*100}")

    new_better = 0
    old_better = 0
    for ticker in tickers:
        td = df[df['ticker'] == ticker]
        exp_name = _ticker_exp.get(ticker, '?')

        best_new_comp = -999
        best_new_row = None
        best_new_mode = None
        for mode in new_modes_present:
            md = td[(td['tpsl_mode'] == mode) & td['survived_a'] & (td['val_a_count'] >= MIN_TRADES)]
            if md.empty:
                md = td[(td['tpsl_mode'] == mode) & td['survived_a']]
            if md.empty:
                continue
            best = md.loc[md[score_col].idxmax()]
            if best[score_col] > best_new_comp:
                best_new_comp = best[score_col]
                best_new_row = best
                best_new_mode = mode

        best_old_comp = -999
        best_old_row = None
        best_old_mode = None
        for mode in existing_modes_present:
            md = td[(td['tpsl_mode'] == mode) & td['survived_a'] & (td['val_a_count'] >= MIN_TRADES)]
            if md.empty:
                md = td[(td['tpsl_mode'] == mode) & td['survived_a']]
            if md.empty:
                continue
            best = md.loc[md[score_col].idxmax()]
            if best[score_col] > best_old_comp:
                best_old_comp = best[score_col]
                best_old_row = best
                best_old_mode = mode

        if best_new_row is None and best_old_row is None:
            print(f"  {ticker:<7} {exp_name:<14} {'---':<10} {'---':>7} {'---':>5} {'---':<10} {'---':>7} {'---':>5} {'---':>7} {'---':>7}")
            continue

        n_pnl = best_new_row[pnl_col] if best_new_row is not None else 0
        n_wr = best_new_row[wr_col] if best_new_row is not None else 0
        n_label = MODE_LABELS.get(best_new_mode, '---') if best_new_mode else '---'

        o_pnl = best_old_row[pnl_col] if best_old_row is not None else 0
        o_wr = best_old_row[wr_col] if best_old_row is not None else 0
        o_label = MODE_LABELS.get(best_old_mode, '---') if best_old_mode else '---'

        delta = n_pnl - o_pnl
        if best_new_row is not None and best_old_row is not None:
            winner = "NEW" if delta > 0 else "OLD"
            if delta > 0:
                new_better += 1
            else:
                old_better += 1
        elif best_new_row is not None:
            winner = "NEW"
            new_better += 1
        else:
            winner = "OLD"
            old_better += 1

        print(f"  {ticker:<7} {exp_name:<14} {n_label:<10} {n_pnl:>6.2%} {n_wr:>4.0%} "
              f"{o_label:<10} {o_pnl:>6.2%} {o_wr:>4.0%} {delta:>+6.2%} {winner:>7}")

    print(f"\n  Score: NEW wins {new_better}  |  EXISTING wins {old_better}  "
          f"({new_better/(new_better+old_better)*100:.0f}% / {old_better/(new_better+old_better)*100:.0f}%)" if new_better + old_better > 0 else "")
else:
    print(f"  Need both new and existing modes present to compare.")

# ==============================================================================
# TABLE 7: PRE-ENTRY RANGE FILTER IMPACT
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 7: PRE-ENTRY RANGE FILTER IMPACT")
print(f"{'='*W}")

if 'pre_range_filter_pct' in df.columns and df['pre_range_filter_pct'].max() > 0:
    for mode in modes_present:
        md = df[(df['tpsl_mode'] == mode) & df['survived_a']]
        if md.empty:
            continue

        no_filter = md[md['pre_range_filter_pct'] == 0]
        with_filter = md[md['pre_range_filter_pct'] > 0]

        if no_filter.empty or with_filter.empty:
            continue

        label = MODE_LABELS.get(mode, mode)
        nf_pnl = no_filter[pnl_col].mean()
        wf_pnl = with_filter[pnl_col].mean()
        nf_n = no_filter[cnt_col].mean()
        wf_n = with_filter[cnt_col].mean()
        delta = wf_pnl - nf_pnl
        print(f"  {label:<10}  No filter: {nf_pnl:>6.2%} PnL, {nf_n:.1f} trades  |  "
              f"With filter: {wf_pnl:>6.2%} PnL, {wf_n:.1f} trades  |  Delta: {delta:>+6.2%}")
else:
    print(f"  No pre-entry range filter data found (all pre_range_filter_pct = 0)")

# ==============================================================================
# TABLE 8: DEPLOY RECOMMENDATIONS
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 8: DEPLOY RECOMMENDATIONS ({period_label} period, PnL >= {DEPLOY_PNL_THRESHOLD:.1%})")
print(f"{'='*W}")
print(f"  {'Ticker':<7} {'Exp':<14} {'Mode':<10} {'VAL PnL':>8} {'TST PnL':>8} {'TST WR':>7} {'TST N':>5} "
      f"{'Comp':>9} {'TP/SL':>20} {'Action':>8}")
print(f"  {'─'*110}")

deploy_count = 0
watch_count = 0

for ticker in tickers:
    td = df[df['ticker'] == ticker]
    winner_mode = ticker_winners.get(ticker)
    exp_name = _ticker_exp.get(ticker, '?')

    if winner_mode is None:
        continue

    wd = td[td['tpsl_mode'] == winner_mode]
    surv = wd[wd['survived_a']]
    eligible = surv[surv['val_a_count'] >= MIN_TRADES]
    if not eligible.empty:
        best = eligible.loc[eligible[score_col].idxmax()]
    elif not surv.empty:
        best = surv.loc[surv[score_col].idxmax()]
    else:
        best = wd.loc[wd[score_col].idxmax()]

    test_pnl = best['val_b_avg_pnl']
    test_wr = best['val_b_win_rate']
    test_n = int(best['val_b_count'])
    val_pnl = best['val_a_avg_pnl']

    if winner_mode in ('atr', 'atr_adaptive'):
        tp_sl_str = f"tp_m={best['tp_mult']:.1f} sl_m={best['sl_mult']:.1f}"
    elif winner_mode == 'trailing_decay':
        tp_sl_str = f"ts={best['trail_start_mult']:.1f} te={best['trail_end_mult']:.1f}"
    elif winner_mode == 'atr_sl':
        tp_sl_str = f"TP={best['tp_pct']:.0%} sl_m={best['sl_mult']:.1f}"
    elif winner_mode == 'grace':
        tp_sl_str = f"TP={best['tp_pct']:.0%} SL={best['sl_pct']:.0%} g={int(best['grace_minutes'])}m"
    elif winner_mode == 'limit_entry':
        tp_sl_str = f"TP={best['tp_pct']:.0%} SL={best['sl_pct']:.0%} d={best['dip_pct']:.1%}"
    else:
        tp_sl_str = f"TP={best['tp_pct']:.0%} SL={best['sl_pct']:.0%}"

    ref_pnl = test_pnl if has_test else val_pnl
    if ref_pnl >= DEPLOY_PNL_THRESHOLD and test_n >= MIN_TRADES:
        action = "DEPLOY"
        deploy_count += 1
    elif ref_pnl >= DEPLOY_PNL_THRESHOLD * 0.5:
        action = "WATCH"
        watch_count += 1
    else:
        action = "skip"

    if action == "skip":
        continue

    label = MODE_LABELS.get(winner_mode, winner_mode)
    new_tag = " *" if winner_mode in V12_NEW_MODES else ""
    print(f"  {ticker:<7} {exp_name:<14} {label:<10} {val_pnl:>7.2%} {test_pnl:>7.2%} {test_wr:>6.0%} {test_n:>5} "
          f"{best[score_col]:>9.6f} {tp_sl_str:>20} {action:>8}{new_tag}")

print(f"\n  DEPLOY: {deploy_count}   WATCH: {watch_count}   (* = v12 new mode)")

# ==============================================================================
# SUMMARY
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  SUMMARY")
print(f"{'='*W}")
print(f"  model_version:   {MODEL_VERSION}")
print(f"  Modes analyzed:  {', '.join(MODE_LABELS.get(m, m) for m in modes_present)}")
print(f"  Tickers:         {len(tickers)}")
print(f"  Scoring period:  {period_label}")
print(f"  DEPLOY:          {deploy_count}")
print(f"  WATCH:           {watch_count}")

print(f"\n  Mode pass rates (VAL / TEST):")
for mode in modes_present:
    md = df[df['tpsl_mode'] == mode]
    if len(md) == 0:
        continue
    pct_a = md['survived_a'].mean() * 100
    pct_b = md['survived_b'].mean() * 100
    label = MODE_LABELS.get(mode, mode)
    cat = " (NEW)" if mode in V12_NEW_MODES else ""
    print(f"    {label:<10} VAL: {pct_a:.1f}%  TEST: {pct_b:.1f}%{cat}")

print(f"\n  Winner distribution:")
for mode in modes_present:
    cnt = win_counts.get(mode, 0)
    if cnt > 0:
        label = MODE_LABELS.get(mode, mode)
        tix = [t for t, m in ticker_winners.items() if m == mode]
        print(f"    {label:<10} {cnt} wins: {', '.join(tix)}")

print(f"{'='*W}")
