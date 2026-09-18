"""
analyze_composite_score.py — Asset Selection for v5 WF Optimization
====================================================================
Reads walk-forward optimization results from BQ (model_version = robust_wf_v5),
ranks tickers by TEST holdout composite score, and outputs deploy configs
ready to paste into daily_score_combined.py.

Composite score: val_b_avg_pnl × val_b_win_rate × log2(1 + val_b_count)
Deploy threshold: TEST avg_pnl >= 1.8%, min N=3.
Confidence tiers: HIGH (N>=12), MED (N=5-11), LOW (N<=4), WATCH (<1.8%)

Run in Colab after WF optimization completes.
"""

import pandas as pd
import numpy as np

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass

from google.cloud import bigquery

# ── CONFIG ──────────────────────────────────────────────────────────────────
MODEL_VERSION = 'robust_wf_v12'
DEPLOY_THRESHOLD = 0.018
MIN_TRADES = 3
BQ_TABLE = f"{GCP_PROJECT}.scoring_logs.wf_optimization_results"
TICKER_RENAME = {'MIGI': 'BGDE', 'BITF': 'KEEL'}
EXCLUDE_TICKERS = {'TLT', 'VIX', 'SPY', 'IWM', 'MSTR'}
W = 130
# ── END CONFIG ──────────────────────────────────────────────────────────────

DEPLOY_EXPERIMENTS = {'152_XSECTOR', '160_XSECTOR', '170_XSECTOR',
                      '175_XSECTOR', '180_XSECTOR', '185_XSECTOR',
                      '200_XSECTOR'}

client = bigquery.Client()
_exp_prefixes = ' OR '.join(f"experiment LIKE '{e.split('_')[0]}%'" for e in DEPLOY_EXPERIMENTS)
query = f"SELECT * FROM `{BQ_TABLE}` WHERE model_version = '{MODEL_VERSION}' AND ({_exp_prefixes})"
df = client.query(query).to_dataframe()
print(f"Loaded {len(df)} rows from BQ (model_version={MODEL_VERSION})")

if df.empty:
    print("No data found. Run WF optimizer first.")
    raise SystemExit

df['ticker'] = df['ticker'].replace(TICKER_RENAME)
df = df[~df['ticker'].isin(EXCLUDE_TICKERS)]

for col in ['val_b_avg_pnl', 'val_b_win_rate']:
    df[col] = df[col].fillna(0.0)
df['val_b_count'] = df['val_b_count'].fillna(0).astype(int)
df['val_b_freq'] = df['val_b_freq'].fillna(0.0)
df['is_ensemble'] = df['is_ensemble'].fillna(False).astype(bool)
df['survived_b'] = df['survived_b'].fillna(False).astype(bool)
df['tpsl_mode'] = df['tpsl_mode'].fillna('static')

for col in ['tp_mult', 'sl_mult', 'trail_start_mult', 'trail_end_mult']:
    if col in df.columns:
        df[col] = df[col].fillna(0.0)

df['base_experiment'] = df['experiment'].apply(
    lambda e: e.replace('_ATR', '').replace('_ATRADP', '').replace('_REGIME', '').replace('_TRAIL', '')
    if any(e.endswith(s) for s in ('_ATR', '_ATRADP', '_REGIME', '_TRAIL')) else e)

pre_filter = len(df)
df = df[df['base_experiment'].isin(DEPLOY_EXPERIMENTS)]
print(f"Filtered to {DEPLOY_EXPERIMENTS}: {len(df)} rows (dropped {pre_filter - len(df)} from other experiments)")

dedup_cols = ['experiment', 'ticker', 'seed', 'is_ensemble', 'tpsl_mode',
              'entry_time', 'prob_thresh', 'tp_pct', 'sl_pct',
              'tp_mult', 'sl_mult', 'trail_start_mult', 'trail_end_mult',
              'vix_threshold', 'tlt_crash', 'spy_ignore', 'vol_filter_pct']
dedup_cols = [c for c in dedup_cols if c in df.columns]
pre_dedup = len(df)
df = df.sort_values('run_timestamp', ascending=False).drop_duplicates(subset=dedup_cols, keep='first')
print(f"Deduped: {pre_dedup} → {len(df)} rows (removed {pre_dedup - len(df)} duplicate re-runs)")

df['test_composite'] = df['val_b_avg_pnl'] * df['val_b_win_rate'] * np.sqrt(df['val_b_count']) * np.log2(1 + df['val_b_count'])

experiments = sorted(df['base_experiment'].unique())
tickers = sorted(df['ticker'].unique())
modes = sorted(df['tpsl_mode'].unique())

print(f"Experiments: {experiments}")
print(f"Tickers: {len(tickers)}")
print(f"Modes: {modes}")


# ── Confidence tagger ──────────────────────────────────────────────────────
def get_confidence(pnl, n, wr):
    if pnl < DEPLOY_THRESHOLD:
        return 'WATCH', f"PnL {pnl:.1%} below {DEPLOY_THRESHOLD:.1%} threshold, N={n}"
    if n <= 4:
        note = f"N={n} trades in TEST"
        if wr >= 0.95:
            note += ", perfect WR likely noise"
        return 'LOW', note
    if n <= 7:
        return 'MED', f"N={n} trades, moderate sample"
    if n >= 12:
        return 'HIGH', f"N={n} trades, strong sample"
    return 'MED', f"N={n} trades"


# ── Helper: pick best row for a ticker ──────────────────────────────────────
def pick_best(ticker_df):
    eligible = ticker_df[
        (ticker_df['val_b_count'] >= MIN_TRADES) &
        (ticker_df['val_b_avg_pnl'] > 0)
    ]
    if eligible.empty:
        return None
    return eligible.loc[eligible['test_composite'].idxmax()]


# ==============================================================================
# TABLE 1: SUMMARY DASHBOARD
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 1: ASSET SELECTION DASHBOARD (TEST holdout, composite = pnl x wr x log2(1+N))")
print(f"{'='*W}")
print(f"  {'Ticker':<7} {'Status':<8} {'Conf':<6} {'Exp':<16} {'Mode':<16} {'Seed':>6} {'Ens':>4} "
      f"{'TEST PnL':>9} {'TEST WR':>8} {'TEST N':>7} {'Composite':>10} "
      f"{'OPT PnL':>9}")
print(f"  {'─'*7} {'─'*8} {'─'*6} {'─'*16} {'─'*16} {'─'*6} {'─'*4} {'─'*9} {'─'*8} {'─'*7} {'─'*10} {'─'*9}")

deploy_rows = []
watch_rows = []
skip_tickers = []

for ticker in tickers:
    td = df[df['ticker'] == ticker]
    best = pick_best(td)

    if best is None:
        skip_tickers.append(ticker)
        continue

    pnl = best['val_b_avg_pnl']
    wr = best['val_b_win_rate']
    n = int(best['val_b_count'])
    status = 'DEPLOY' if pnl >= DEPLOY_THRESHOLD else 'WATCH'
    conf, conf_note = get_confidence(pnl, n, wr)
    is_ens = best['is_ensemble']
    seed_str = 'ENS' if is_ens else str(int(best['seed']))

    row = {
        'ticker': ticker, 'status': status,
        'confidence': conf, 'confidence_note': conf_note,
        'mode': best['tpsl_mode'], 'seed': seed_str,
        'is_ensemble': is_ens,
        'test_pnl': pnl, 'test_wr': wr,
        'test_n': n,
        'composite': best['test_composite'],
        'opt_pnl': best.get('opt_avg_pnl', 0),
        'entry_time': best.get('entry_time', ''),
        'entry_minute': int(best.get('entry_minute', 0)),
        'prob_thresh': float(best.get('prob_thresh', 0)),
        'tp_pct': float(best.get('tp_pct', 0)),
        'sl_pct': float(best.get('sl_pct', 0)),
        'vix_threshold': float(best.get('vix_threshold', 0)),
        'tlt_crash': float(best.get('tlt_crash', 0)),
        'spy_ignore': float(best.get('spy_ignore', 0)),
        'vol_filter_pct': int(best.get('vol_filter_pct', 0)),
        'tp_mult': float(best.get('tp_mult', 0)),
        'sl_mult': float(best.get('sl_mult', 0)),
        'val_b_freq': float(best.get('val_b_freq', 0)),
        'experiment': best['experiment'],
        'base_experiment': best['base_experiment'],
    }

    if status == 'DEPLOY':
        deploy_rows.append(row)
    else:
        watch_rows.append(row)

all_rows = sorted(deploy_rows + watch_rows, key=lambda r: -r['composite'])

for r in all_rows:
    tag = '>>>' if r['status'] == 'DEPLOY' else '   '
    print(f"{tag}{r['ticker']:<7} {r['status']:<8} {r['confidence']:<6} "
          f"{r['base_experiment']:<16} {r['mode']:<16} {r['seed']:>6} "
          f"{'Y' if r['is_ensemble'] else 'N':>4} "
          f"{r['test_pnl']:>8.2%} {r['test_wr']:>7.0%} {r['test_n']:>7} "
          f"{r['composite']:>10.6f} {r['opt_pnl']:>8.2%}")

print(f"\n  DEPLOY: {len(deploy_rows)}  |  WATCH: {len(watch_rows)}  |  SKIP: {len(skip_tickers)}")
if skip_tickers:
    print(f"  Skipped (no survivors or N<{MIN_TRADES}): {', '.join(skip_tickers)}")


# ==============================================================================
# TABLE 2: MODE COMPARISON PER DEPLOYABLE TICKER
# ==============================================================================
active_tickers = [r['ticker'] for r in deploy_rows + watch_rows]

if active_tickers:
    print(f"\n{'='*W}")
    print(f"  TABLE 2: MODE COMPARISON (DEPLOY + WATCH tickers)")
    print(f"{'='*W}")

    for ticker in sorted(active_tickers):
        td = df[(df['ticker'] == ticker) & (~df['is_ensemble'])]
        print(f"\n  {ticker}:")
        print(f"  {'Mode':<18} {'MedPnL':>8} {'MedWR':>7} {'MedN':>5} "
              f"{'Surv':>6} {'BestSeed':>9} {'BestPnL':>9}")
        print(f"  {'─'*18} {'─'*8} {'─'*7} {'─'*5} {'─'*6} {'─'*9} {'─'*9}")

        for mode in modes:
            md = td[td['tpsl_mode'] == mode]
            survived = md[md['val_b_count'] >= MIN_TRADES]
            if survived.empty:
                print(f"  {mode:<18} {'—':>8} {'—':>7} {'—':>5} "
                      f"{f'0/{len(md)}':>6} {'—':>9} {'—':>9}")
                continue
            med_pnl = survived['val_b_avg_pnl'].median()
            med_wr = survived['val_b_win_rate'].median()
            med_n = survived['val_b_count'].median()
            best_row = survived.loc[survived['val_b_avg_pnl'].idxmax()]
            print(f"  {mode:<18} {med_pnl:>7.2%} {med_wr:>6.0%} {med_n:>5.0f} "
                  f"{f'{len(survived)}/{len(md)}':>6} "
                  f"{int(best_row['seed']):>9} {best_row['val_b_avg_pnl']:>8.2%}")


# ==============================================================================
# TABLE 3: ENSEMBLE vs BEST SEED
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 3: ENSEMBLE vs BEST SEED (TEST performance)")
print(f"{'='*W}")
print(f"  {'Ticker':<7} {'Mode':<16} "
      f"{'ENS PnL':>9} {'ENS WR':>8} {'ENS N':>6} "
      f"{'Best PnL':>9} {'Best WR':>8} {'Best N':>6} "
      f"{'Winner':>8}")
print(f"  {'─'*7} {'─'*16} {'─'*9} {'─'*8} {'─'*6} {'─'*9} {'─'*8} {'─'*6} {'─'*8}")

for ticker in sorted(active_tickers):
    td = df[df['ticker'] == ticker]
    for mode in modes:
        md = td[td['tpsl_mode'] == mode]
        ens = md[md['is_ensemble']]
        seeds = md[~md['is_ensemble']]
        if ens.empty or seeds.empty:
            continue
        ens_row = ens.iloc[0]
        best_seed = seeds.loc[seeds['val_b_avg_pnl'].idxmax()]
        winner = 'ENS' if ens_row['val_b_avg_pnl'] >= best_seed['val_b_avg_pnl'] else 'SEED'
        print(f"  {ticker:<7} {mode:<16} "
              f"{ens_row['val_b_avg_pnl']:>8.2%} {ens_row['val_b_win_rate']:>7.0%} "
              f"{int(ens_row['val_b_count']):>6} "
              f"{best_seed['val_b_avg_pnl']:>8.2%} {best_seed['val_b_win_rate']:>7.0%} "
              f"{int(best_seed['val_b_count']):>6} "
              f"{winner:>8}")


# ==============================================================================
# TABLE 4: DEPLOY CONFIG OUTPUT — GROUPED BY EXPERIMENT
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 4: DEPLOY CONFIG (paste into daily_score_combined.py)")
print(f"  Includes ALL tickers (DEPLOY + WATCH), grouped by experiment")
print(f"{'='*W}")

all_active = sorted(deploy_rows + watch_rows, key=lambda x: x['ticker'])

if all_active:
    by_exp = {}
    for r in all_active:
        exp = r['base_experiment']
        by_exp.setdefault(exp, []).append(r)

    for exp_name in sorted(by_exp.keys()):
        exp_tickers = sorted(by_exp[exp_name], key=lambda x: x['ticker'])
        print(f"\n# --- {exp_name} ({len(exp_tickers)} tickers) ---")
        print(f"# Paste into EXP_{exp_name.split('_')[0]} asset_configs:")

        for r in exp_tickers:
            freq_str = f"{r['val_b_freq']:.1f}/mo" if r['val_b_freq'] > 0 else 'N/A'
            seed_val = 0 if r['is_ensemble'] else int(r['seed'])
            conf = r['confidence']
            conf_note = r['confidence_note']
            print(f"    '{r['ticker']}': {{")
            print(f"        'seed': {seed_val}, 'prob_threshold': {r['prob_thresh']:.3f},")
            print(f"        'tp_pct': {r['tp_pct']:.3f}, 'sl_pct': {r['sl_pct']:.3f}, "
                  f"'entry_minute': {r['entry_minute']},")
            print(f"        'vix_threshold': {r['vix_threshold']:.1f}, "
                  f"'tlt_crash': {r['tlt_crash']:.2f}, 'spy_ignore': {r['spy_ignore']:.2f},")
            print(f"        'vol_filter_pct': {r['vol_filter_pct']},")
            print(f"        'confidence': '{conf}',")
            print(f"        'confidence_note': '{conf_note}',")
            print(f"        'stats': {{'win_rate': '{r['test_wr']:.0%}', "
                  f"'avg_pnl': '{r['test_pnl']:.1%}', 'freq': '{freq_str}'}},")
            print(f"    }},")
else:
    print("\n  No tickers meet minimum criteria.")


# ==============================================================================
# TABLE 5: FULL RESULTS DUMP (all seeds, all modes, sortable)
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 5: ALL RESULTS (every seed x mode with TEST data)")
print(f"{'='*W}")

has_test = df[(df['val_b_count'] >= MIN_TRADES) & (~df['is_ensemble'])].copy()
has_test = has_test.sort_values('test_composite', ascending=False)

print(f"  {'Ticker':<7} {'Experiment':<22} {'Mode':<16} {'Seed':>6} "
      f"{'TEST PnL':>9} {'TEST WR':>8} {'TEST N':>7} {'Composite':>10} "
      f"{'OPT PnL':>9}")
print(f"  {'─'*7} {'─'*22} {'─'*16} {'─'*6} {'─'*9} {'─'*8} {'─'*7} {'─'*10} {'─'*9}")

for _, r in has_test.head(60).iterrows():
    flag = '>>>' if r['val_b_avg_pnl'] >= DEPLOY_THRESHOLD else '   '
    print(f"{flag}{r['ticker']:<7} {r['experiment']:<22} {r['tpsl_mode']:<16} "
          f"{int(r['seed']):>6} "
          f"{r['val_b_avg_pnl']:>8.2%} {r['val_b_win_rate']:>7.0%} "
          f"{int(r['val_b_count']):>7} {r['test_composite']:>10.6f} "
          f"{r['opt_avg_pnl']:>8.2%}")

print(f"\n  Total rows with TEST N>={MIN_TRADES}: {len(has_test)}")
print(f"  Rows with TEST PnL >= {DEPLOY_THRESHOLD:.1%}: "
      f"{len(has_test[has_test['val_b_avg_pnl'] >= DEPLOY_THRESHOLD])}")


# ==============================================================================
# CONFIDENCE SUMMARY
# ==============================================================================
print(f"\n{'='*W}")
print(f"  CONFIDENCE BREAKDOWN")
print(f"{'='*W}")
for tier in ['HIGH', 'MED', 'LOW', 'WATCH']:
    tier_rows = [r for r in all_rows if r['confidence'] == tier]
    if tier_rows:
        tickers_str = ', '.join(f"{r['ticker']}(N={r['test_n']})" for r in tier_rows)
        print(f"  {tier:<6} ({len(tier_rows)}): {tickers_str}")

print(f"\n  HIGH  = N>=12 TEST trades, strong statistical sample")
print(f"  MED   = N=5-11 TEST trades, moderate sample")
print(f"  LOW   = N<=4 TEST trades, may be noise (especially if 100% WR)")
print(f"  WATCH = TEST PnL < 1.8% threshold, monitoring only")

# ==============================================================================
# TABLE 6: HEAD-TO-HEAD EXPERIMENT COMPARISON (152 vs 160 vs 170)
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 6: EXPERIMENT HEAD-TO-HEAD — Best config per ticker per experiment")
print(f"  Composite = pnl x wr x log2(1+N), min N={MIN_TRADES}")
print(f"{'='*W}")

exp_list = sorted(df['base_experiment'].unique())
exp_short = {e: e.split('_')[0] for e in exp_list}

header = f"  {'Ticker':<7}"
for e in exp_list:
    header += f" {'|':>2} {exp_short[e]+' PnL':>8} {exp_short[e]+' WR':>6} {exp_short[e]+' N':>5} {exp_short[e]+' Comp':>9}"
header += f" {'|':>2} {'Winner':<8} {'Δ vs 2nd':>8}"
print(header)
print(f"  {'─'*7}" + (f" {'─'*2} {'─'*8} {'─'*6} {'─'*5} {'─'*9}" * len(exp_list))
      + f" {'─'*2} {'─'*8} {'─'*8}")

exp_win_count = {e: 0 for e in exp_list}
exp_deploy_count = {e: 0 for e in exp_list}

for ticker in tickers:
    td = df[df['ticker'] == ticker]
    line = f"  {ticker:<7}"

    exp_bests = {}
    for e in exp_list:
        ed = td[td['base_experiment'] == e]
        eligible = ed[(ed['val_b_count'] >= MIN_TRADES) & (ed['val_b_avg_pnl'] > 0)]
        if eligible.empty:
            line += f" {'|':>2} {'—':>8} {'—':>6} {'—':>5} {'—':>9}"
            continue
        best = eligible.loc[eligible['test_composite'].idxmax()]
        pnl = best['val_b_avg_pnl']
        wr = best['val_b_win_rate']
        n = int(best['val_b_count'])
        comp = best['test_composite']
        exp_bests[e] = comp
        line += f" {'|':>2} {pnl:>7.1%} {wr:>5.0%} {n:>5} {comp:>9.4f}"

    if exp_bests:
        sorted_exps = sorted(exp_bests.items(), key=lambda x: -x[1])
        winner_exp = sorted_exps[0][0]
        winner_comp = sorted_exps[0][1]
        delta = winner_comp - sorted_exps[1][1] if len(sorted_exps) > 1 else winner_comp
        exp_win_count[winner_exp] += 1

        winner_row = [r for r in all_rows if r['ticker'] == ticker]
        if winner_row and winner_row[0]['status'] == 'DEPLOY':
            exp_deploy_count[winner_row[0]['base_experiment']] += 1

        line += f" {'|':>2} {exp_short[winner_exp]:<8} {delta:>+7.4f}"
    else:
        line += f" {'|':>2} {'—':<8} {'—':>8}"

    print(line)

print(f"\n  EXPERIMENT WINS:")
for e in exp_list:
    print(f"    {exp_short[e]}: {exp_win_count[e]} tickers win (best composite)")
print()

# Per-experiment aggregate stats
print(f"  EXPERIMENT AGGREGATE STATS (across all seeds/modes/tickers):")
print(f"  {'Experiment':<16} {'Rows':>6} {'Tickers':>8} {'MedPnL':>8} {'MedWR':>7} {'MedN':>6} "
      f"{'Deploy':>7} {'Deploy%':>7}")
print(f"  {'─'*16} {'─'*6} {'─'*8} {'─'*8} {'─'*7} {'─'*6} {'─'*7} {'─'*7}")

for e in exp_list:
    ed = df[df['base_experiment'] == e]
    has_test_e = ed[(ed['val_b_count'] >= MIN_TRADES) & (ed['val_b_avg_pnl'] > 0)]
    n_tickers = ed['ticker'].nunique()
    if has_test_e.empty:
        print(f"  {e:<16} {len(ed):>6} {n_tickers:>8} {'—':>8} {'—':>7} {'—':>6} {'—':>7} {'—':>7}")
        continue
    med_pnl = has_test_e['val_b_avg_pnl'].median()
    med_wr = has_test_e['val_b_win_rate'].median()
    med_n = has_test_e['val_b_count'].median()
    n_deploy = len(has_test_e[has_test_e['val_b_avg_pnl'] >= DEPLOY_THRESHOLD])
    deploy_pct = n_deploy / len(has_test_e) * 100
    print(f"  {e:<16} {len(ed):>6} {n_tickers:>8} {med_pnl:>7.1%} {med_wr:>6.0%} {med_n:>6.0f} "
          f"{n_deploy:>7} {deploy_pct:>6.0f}%")

# Ticker-level: which experiment dominates
print(f"\n  PER-TICKER EXPERIMENT SELECTION (from Table 1):")
exp_selected = {}
for r in all_rows:
    be = r['base_experiment']
    exp_selected.setdefault(be, []).append(r['ticker'])
for e in exp_list:
    tks = exp_selected.get(e, [])
    print(f"    {exp_short[e]} ({len(tks)}): {', '.join(sorted(tks)) if tks else '—'}")

print(f"\n{'='*W}")
print(f"  ASSET SELECTION COMPLETE")
print(f"{'='*W}")
