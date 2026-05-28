"""
analyze_dynamic_tpsl.py — Compare TP/SL Modes Head-to-Head
============================================================
Queries BQ for all tpsl_mode results and compares:
  static vs atr vs atr_adaptive vs regime_scaled vs trailing_decay

Tables:
  1. Mode overview (rows, tickers, pass rates)
  2. Per-ticker mode comparison (best seed per mode, side by side)
  3. Mode wins summary (which mode is best per ticker)
  4. Aggregate stats (N-weighted avg PnL, WR, decay by mode)

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
MIN_TRADES = 3
DEPLOY_PNL_THRESHOLD = 0.018

W = 150

TICKER_RENAME = {'MIGI': 'BGDE', 'BITF': 'KEEL'}
EXCLUDE_TICKERS = {'TLT', 'VIX', 'SMLR', 'SPIR'}

CLEAN_EXP_BASES = ('105_DEFENSE', '106_DEFENSE', '152_XSECTOR', '160_XSECTOR',
                   '170_XSECTOR', '171_XSECTOR', '172_XSECTOR',
                   '173_XSECTOR', '174_XSECTOR', '175_XSECTOR')

MODE_ORDER = ['static', 'atr', 'atr_adaptive', 'regime_scaled', 'trailing_decay']
MODE_LABELS = {
    'static': 'STATIC',
    'atr': 'ATR',
    'atr_adaptive': 'ATR_ADP',
    'regime_scaled': 'REGIME',
    'trailing_decay': 'TRAIL',
}

# ==============================================================================
# LOAD DATA
# ==============================================================================
print(f"\n{'='*W}")
print(f"  DYNAMIC TP/SL MODE ANALYZER")
print(f"  Selection: composite = avg_pnl × win_rate × log2(1+N),  min N = {MIN_TRADES}")
print(f"{'='*W}")

client = bigquery.Client()
query = f"SELECT * FROM `{BQ_TABLE}` ORDER BY experiment, ticker, seed"
try:
    df = client.query(query).to_dataframe()
except Exception as e:
    if 'Not found' in str(e) or 'notFound' in str(e):
        print(f"\n  Table {BQ_TABLE} does not exist yet.")
        raise SystemExit(0)
    raise

if df.empty:
    print("\n  No data in BQ table.")
    raise SystemExit(0)

df['ticker'] = df['ticker'].replace(TICKER_RENAME)
df = df[~df['ticker'].isin(EXCLUDE_TICKERS)]
df = df[df['experiment'].apply(lambda e: any(e == b or e.startswith(b + '_') for b in CLEAN_EXP_BASES))]
df['survived_a'] = df['survived_a'].fillna(False).astype(bool)
df['survived_b'] = df['survived_b'].fillna(False).astype(bool)
df['val_b_avg_pnl'] = df['val_b_avg_pnl'].fillna(0.0)
if 'vol_filter_pct' not in df.columns:
    df['vol_filter_pct'] = 0
df['vol_filter_pct'] = df['vol_filter_pct'].fillna(0).astype(int)
if 'tpsl_mode' not in df.columns:
    df['tpsl_mode'] = 'static'
df['tpsl_mode'] = df['tpsl_mode'].fillna('static').replace('', 'static')
if 'tp_mult' not in df.columns:
    df['tp_mult'] = 0.0
df['tp_mult'] = df['tp_mult'].fillna(0.0)
if 'sl_mult' not in df.columns:
    df['sl_mult'] = 0.0
df['sl_mult'] = df['sl_mult'].fillna(0.0)
if 'trail_start_mult' not in df.columns:
    df['trail_start_mult'] = 0.0
df['trail_start_mult'] = df['trail_start_mult'].fillna(0.0)
if 'trail_end_mult' not in df.columns:
    df['trail_end_mult'] = 0.0
df['trail_end_mult'] = df['trail_end_mult'].fillna(0.0)

df['val_a_total_pnl'] = df['val_a_avg_pnl'] * df['val_a_count']
df['composite'] = df['val_a_avg_pnl'] * df['val_a_win_rate'] * np.log2(1 + df['val_a_count'])

df['run_timestamp'] = pd.to_datetime(df['run_timestamp'])

# Derive base_experiment by stripping mode suffixes from experiment name
def _base_exp(exp_name):
    for sfx in ['_ATR', '_ATRADP', '_REGIME', '_TRAIL']:
        if exp_name.endswith(sfx):
            return exp_name[:-len(sfx)]
    return exp_name

df['base_experiment'] = df['experiment'].apply(_base_exp)

# Dedup: keep latest run per (experiment, ticker)
_latest = (df.groupby(['experiment', 'ticker', 'run_id'])['run_timestamp']
           .max().reset_index()
           .sort_values('run_timestamp')
           .groupby(['experiment', 'ticker']).last()
           .reset_index()[['experiment', 'ticker', 'run_id']])
df = df.merge(_latest, on=['experiment', 'ticker', 'run_id'], how='inner')

modes_present = [m for m in MODE_ORDER if m in df['tpsl_mode'].unique()]
tickers = sorted(df['ticker'].unique())

# Build ticker -> base_experiment lookup (use the static/base experiment for each ticker)
_ticker_exp = (df.groupby('ticker')['base_experiment']
               .agg(lambda x: x.value_counts().index[0])
               .to_dict())
# Track which tickers have dynamic mode data (not just static)
_tickers_with_dynamic = set(
    df[df['tpsl_mode'] != 'static']['ticker'].unique()
)

print(f"\n  Modes found: {', '.join(MODE_LABELS.get(m, m) for m in modes_present)}")
print(f"  Tickers: {len(tickers)}  ({len(_tickers_with_dynamic)} with dynamic modes, "
      f"{len(tickers) - len(_tickers_with_dynamic)} static-only)")
print(f"  Total rows: {len(df):,}")

# ==============================================================================
# TABLE 1: MODE OVERVIEW
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 1: MODE OVERVIEW")
print(f"{'='*W}")
print(f"  {'Mode':<12} {'Rows':>7} {'Tickers':>8} {'Seeds':>6} "
      f"{'PassA':>7} {'PassA%':>7} {'PassB':>7} {'PassB%':>7} "
      f"{'Avg OPT PnL':>12} {'Avg VAL PnL':>12}")
print(f"  {'─'*90}")

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
    label = MODE_LABELS.get(mode, mode)
    print(f"  {label:<12} {n:>7} {n_tickers:>8} {n_seeds:>6} "
          f"{pass_a:>4}/{n} {pct_a:>5.1f}% {pass_b:>4}/{n} {pct_b:>5.1f}% "
          f"{avg_opt:>11.2%} {avg_val:>11.2%}")

# ==============================================================================
# TABLE 2: PER-TICKER MODE COMPARISON (best seed per mode)
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 2: PER-TICKER MODE COMPARISON (best seed per mode)")
print(f"{'='*W}")

mode_cols = '  '.join(f"{'──── ' + MODE_LABELS.get(m, m) + ' ────':>22}" for m in modes_present)
print(f"  {'Ticker':<7} {'Exp':<14} {mode_cols}")
sub_hdr = '  '.join(f"{'PnL':>7} {'WR':>5} {'N':>3} {'Tag':>5}" for _ in modes_present)
print(f"  {'':7} {'':14} {sub_hdr}")
print(f"  {'─'*W}")

ticker_winners = {}

for ticker in tickers:
    td = df[df['ticker'] == ticker]
    parts = []
    best_mode = None
    best_composite = -999.0

    for mode in modes_present:
        md = td[td['tpsl_mode'] == mode]
        if md.empty:
            parts.append(f"{'---':>7} {'---':>5} {'--':>3} {'':>5}")
            continue

        surv = md[md['survived_a']]
        eligible = surv[surv['val_a_count'] >= MIN_TRADES]
        if not eligible.empty:
            best = eligible.loc[eligible['composite'].idxmax()]
            tag = "PASS"
        elif not surv.empty:
            best = surv.loc[surv['composite'].idxmax()]
            tag = "pass"
        else:
            best = md.loc[md['composite'].idxmax()]
            tag = "FAIL"

        pnl = best['val_a_avg_pnl']
        wr = best['val_a_win_rate']
        n_t = int(best['val_a_count'])
        comp = best['composite']

        if tag != "FAIL" and comp > best_composite:
            best_composite = comp
            best_mode = mode

        parts.append(f"{pnl:>6.2%} {wr:>4.0%} {n_t:>3} {tag:>5}")

    ticker_winners[ticker] = best_mode
    line = '  '.join(parts)
    winner_mark = f"  ← {MODE_LABELS.get(best_mode, '?')}" if best_mode else ""
    exp_name = _ticker_exp.get(ticker, '?')
    print(f"  {ticker:<7} {exp_name:<14} {line}{winner_mark}")

# ==============================================================================
# TABLE 3: MODE WINS SUMMARY
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 3: MODE WINS (which mode is best per ticker)")
print(f"{'='*W}")

win_counts = {}
for mode in modes_present:
    win_counts[mode] = sum(1 for v in ticker_winners.values() if v == mode)

n_none = sum(1 for v in ticker_winners.values() if v is None)

# Separate: tickers with dynamic data vs static-only
dyn_winners = {t: m for t, m in ticker_winners.items() if t in _tickers_with_dynamic}
static_only_winners = {t: m for t, m in ticker_winners.items() if t not in _tickers_with_dynamic}

n_dyn = len(dyn_winners)
print(f"  A) TICKERS WITH DYNAMIC DATA ({n_dyn} tickers)")
print(f"  {'Mode':<12} {'Wins':>6} {'%':>7}")
print(f"  {'─'*28}")
for mode in modes_present:
    cnt = sum(1 for v in dyn_winners.values() if v == mode)
    pct = cnt / n_dyn * 100 if n_dyn > 0 else 0
    print(f"  {MODE_LABELS.get(mode, mode):<12} {cnt:>6} {pct:>6.1f}%")
n_none_dyn = sum(1 for v in dyn_winners.values() if v is None)
if n_none_dyn > 0:
    print(f"  {'No winner':<12} {n_none_dyn:>6} {n_none_dyn/n_dyn*100:>6.1f}%")

n_static = len(static_only_winners)
print(f"\n  B) STATIC-ONLY TICKERS ({n_static} tickers — no dynamic modes ran)")
n_static_pass = sum(1 for v in static_only_winners.values() if v == 'static')
n_static_none = sum(1 for v in static_only_winners.values() if v is None)
print(f"  {'STATIC PASS':<12} {n_static_pass:>6}")
if n_static_none > 0:
    print(f"  {'No winner':<12} {n_static_none:>6}")

print(f"\n  C) ALL TICKERS COMBINED ({len(tickers)} tickers)")
print(f"  {'Mode':<12} {'Wins':>6} {'%':>7}")
print(f"  {'─'*28}")
for mode in modes_present:
    pct = win_counts[mode] / len(tickers) * 100 if len(tickers) > 0 else 0
    print(f"  {MODE_LABELS.get(mode, mode):<12} {win_counts[mode]:>6} {pct:>6.1f}%")
if n_none > 0:
    print(f"  {'No winner':<12} {n_none:>6} {n_none/len(tickers)*100:>6.1f}%")

# ==============================================================================
# TABLE 4: PER-TICKER DETAIL — all modes side by side with params
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 4: PER-TICKER DETAIL (best seed params per mode)")
print(f"{'='*W}")

for ticker in tickers:
    td = df[df['ticker'] == ticker]
    modes_for_ticker = [m for m in modes_present if m in td['tpsl_mode'].values]
    if len(modes_for_ticker) <= 1:
        continue

    exp_name = _ticker_exp.get(ticker, '?')
    print(f"\n  {ticker}  ({exp_name})")
    print(f"  {'Mode':<10} {'Entry':>6} {'Prob':>6} {'TP/SL Params':<25} "
          f"{'VIX<':>5} {'TLT>':>6} {'SPY<':>5} "
          f"{'|':>2} {'OPT%':>6} {'VAL%':>6} {'WR':>5} {'N':>4} {'Comp':>9} {'Tag':>5}")
    print(f"  {'─'*110}")

    for mode in modes_for_ticker:
        md = td[td['tpsl_mode'] == mode]
        surv = md[md['survived_a']]
        eligible = surv[surv['val_a_count'] >= MIN_TRADES]
        if not eligible.empty:
            best = eligible.loc[eligible['composite'].idxmax()]
            tag = "PASS"
        elif not surv.empty:
            best = surv.loc[surv['composite'].idxmax()]
            tag = "pass"
        else:
            best = md.loc[md['composite'].idxmax()]
            tag = "FAIL"

        if mode in ('atr', 'atr_adaptive'):
            tp_sl_str = f"tp_m={best['tp_mult']:.1f}  sl_m={best['sl_mult']:.1f}"
        elif mode == 'trailing_decay':
            tp_sl_str = f"ts={best['trail_start_mult']:.1f} te={best['trail_end_mult']:.1f} sl_m={best['sl_mult']:.1f}"
        else:
            tp_sl_str = f"TP={best['tp_pct']:.0%}  SL={best['sl_pct']:.0%}"

        label = MODE_LABELS.get(mode, mode)
        winner = " ←" if ticker_winners.get(ticker) == mode else ""
        print(f"  {label:<10} {best['entry_time']:>6} {best['prob_thresh']:>6.3f} {tp_sl_str:<25} "
              f"{best['vix_threshold']:>5.0f} {best['tlt_crash']:>6.2f} {best['spy_ignore']:>5.2f} "
              f"{'|':>2} {best['opt_avg_pnl']:>5.2%} {best['val_a_avg_pnl']:>5.2%} "
              f"{best['val_a_win_rate']:>4.0%} {int(best['val_a_count']):>4} "
              f"{best['composite']:>9.6f} {tag:>5}{winner}")

# ==============================================================================
# TABLE 5: AGGREGATE STATS BY MODE
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 5: AGGREGATE STATS BY MODE (survivors only)")
print(f"{'='*W}")
print(f"  {'Mode':<12} {'Seeds':>6} {'Avg OPT':>8} {'Avg VAL':>8} {'Decay':>7} "
      f"{'Med VAL':>8} {'Avg WR':>7} {'Avg N':>6} {'Tot PnL':>8}")
print(f"  {'─'*80}")

for mode in modes_present:
    md = df[(df['tpsl_mode'] == mode) & df['survived_a']]
    if md.empty:
        label = MODE_LABELS.get(mode, mode)
        print(f"  {label:<12}      0     ---      ---     ---      ---     ---    ---      ---")
        continue

    n_s = len(md)
    avg_opt = md['opt_avg_pnl'].mean()
    avg_val = md['val_a_avg_pnl'].mean()
    decay = (avg_val - avg_opt) / avg_opt * 100 if avg_opt != 0 else 0
    med_val = md['val_a_avg_pnl'].median()
    avg_wr = md['val_a_win_rate'].mean()
    avg_n = md['val_a_count'].mean()
    tot_pnl = md['val_a_total_pnl'].sum()
    label = MODE_LABELS.get(mode, mode)
    print(f"  {label:<12} {n_s:>6} {avg_opt:>7.2%} {avg_val:>7.2%} {decay:>6.1f}% "
          f"{med_val:>7.2%} {avg_wr:>6.1%} {avg_n:>6.1f} {tot_pnl:>7.2%}")

# ==============================================================================
# TABLE 6: N-WEIGHTED COMPARISON (high-confidence tickers, N >= 10)
# ==============================================================================
HIGH_N_THRESHOLD = 10

print(f"\n\n{'='*W}")
print(f"  TABLE 6: HIGH-N COMPARISON (VAL-A N >= {HIGH_N_THRESHOLD}, best seed per ticker per mode)")
print(f"{'='*W}")

high_n_rows = []
for ticker in tickers:
    td = df[df['ticker'] == ticker]
    for mode in modes_present:
        md = td[(td['tpsl_mode'] == mode) & td['survived_a'] & (td['val_a_count'] >= HIGH_N_THRESHOLD)]
        if md.empty:
            continue
        best = md.loc[md['composite'].idxmax()]
        high_n_rows.append({
            'ticker': ticker,
            'mode': mode,
            'val_a_avg_pnl': best['val_a_avg_pnl'],
            'val_a_win_rate': best['val_a_win_rate'],
            'val_a_count': int(best['val_a_count']),
            'val_a_total_pnl': best['val_a_total_pnl'],
            'opt_avg_pnl': best['opt_avg_pnl'],
            'composite': best['composite'],
        })

if high_n_rows:
    hn = pd.DataFrame(high_n_rows)
    hn_tickers = sorted(hn['ticker'].unique())

    mode_cols = '  '.join(f"{'── ' + MODE_LABELS.get(m, m) + ' ──':>18}" for m in modes_present)
    print(f"  {'Ticker':<7} {mode_cols}")
    sub_hdr = '  '.join(f"{'PnL':>6} {'WR':>4} {'N':>3} {'Tot':>5}" for _ in modes_present)
    print(f"  {'':7} {sub_hdr}")
    print(f"  {'─'*W}")

    for ticker in hn_tickers:
        th = hn[hn['ticker'] == ticker]
        parts = []
        for mode in modes_present:
            mh = th[th['mode'] == mode]
            if mh.empty:
                parts.append(f"{'---':>6} {'--':>4} {'--':>3} {'---':>5}")
            else:
                r = mh.iloc[0]
                parts.append(f"{r['val_a_avg_pnl']:>5.2%} {r['val_a_win_rate']:>3.0%} "
                              f"{int(r['val_a_count']):>3} {r['val_a_total_pnl']:>4.0%}")
        print(f"  {ticker:<7} {'  '.join(parts)}")

    print(f"\n  Mode aggregates (N-weighted):")
    for mode in modes_present:
        mh = hn[hn['mode'] == mode]
        if mh.empty:
            continue
        total_n = mh['val_a_count'].sum()
        wt_pnl = (mh['val_a_avg_pnl'] * mh['val_a_count']).sum() / total_n if total_n > 0 else 0
        wt_wr = (mh['val_a_win_rate'] * mh['val_a_count']).sum() / total_n if total_n > 0 else 0
        tot = mh['val_a_total_pnl'].sum()
        label = MODE_LABELS.get(mode, mode)
        print(f"    {label:<12} N-wt PnL={wt_pnl:.2%}  N-wt WR={wt_wr:.1%}  "
              f"Total PnL={tot:.2%}  Tickers={len(mh)}  Trades={total_n}")
else:
    print(f"  No tickers with N >= {HIGH_N_THRESHOLD} in any mode yet.")

# ==============================================================================
# TABLE 7: OPTIMAL MODE RECOMMENDATION
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 7: RECOMMENDED MODE PER TICKER")
print(f"{'='*W}")
print(f"  {'Ticker':<7} {'Exp':<14} {'Best Mode':<12} {'VAL PnL':>8} {'WR':>5} {'N':>4} {'Comp':>9} "
      f"{'vs Static PnL':>14}")
print(f"  {'─'*80}")

for ticker in tickers:
    td = df[df['ticker'] == ticker]
    winner_mode = ticker_winners.get(ticker)
    exp_name = _ticker_exp.get(ticker, '?')
    if winner_mode is None:
        print(f"  {ticker:<7} {exp_name:<14} {'NONE':<12} {'---':>8}")
        continue

    wd = td[td['tpsl_mode'] == winner_mode]
    surv = wd[wd['survived_a']]
    eligible = surv[surv['val_a_count'] >= MIN_TRADES]
    if not eligible.empty:
        best = eligible.loc[eligible['composite'].idxmax()]
    elif not surv.empty:
        best = surv.loc[surv['composite'].idxmax()]
    else:
        best = wd.loc[wd['composite'].idxmax()]

    static_pnl = 0.0
    sd = td[td['tpsl_mode'] == 'static']
    if not sd.empty:
        ss = sd[sd['survived_a']]
        se = ss[ss['val_a_count'] >= MIN_TRADES]
        if not se.empty:
            static_pnl = se.loc[se['composite'].idxmax()]['val_a_avg_pnl']
        elif not ss.empty:
            static_pnl = ss.loc[ss['composite'].idxmax()]['val_a_avg_pnl']

    delta = best['val_a_avg_pnl'] - static_pnl if static_pnl != 0 else 0
    delta_str = f"{delta:>+7.2%}" if static_pnl != 0 else "    N/A"

    label = MODE_LABELS.get(winner_mode, winner_mode)
    print(f"  {ticker:<7} {exp_name:<14} {label:<12} {best['val_a_avg_pnl']:>7.2%} "
          f"{best['val_a_win_rate']:>4.0%} {int(best['val_a_count']):>4} "
          f"{best['composite']:>9.6f} {delta_str:>14}")

# ==============================================================================
# SUMMARY
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  SUMMARY")
print(f"{'='*W}")
print(f"  Modes analyzed:  {', '.join(MODE_LABELS.get(m, m) for m in modes_present)}")
print(f"  Tickers:         {len(tickers)}")
for mode in modes_present:
    md = df[df['tpsl_mode'] == mode]
    pass_rate = md['survived_a'].mean() * 100 if len(md) > 0 else 0
    label = MODE_LABELS.get(mode, mode)
    print(f"  {label:<15} pass rate: {pass_rate:.1f}%  ({int(md['survived_a'].sum())}/{len(md)})")
print(f"{'='*W}")
