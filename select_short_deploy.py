"""
select_short_deploy.py — SHORT Model Deployment Selection (Exp 400 vs 410)
================================================================================
Selects SHORT/dump model deployment candidates using alpha agreement.

Alpha=2.0: Exp 400 (cutoff Aug).  Alpha=1.0: Exp 410 (cutoff Aug).
Selection based ONLY on VAL_A period (Feb 1 – Mar 31, 2026).
TEST period (Apr 1+) is displayed for reference but never used in selection.

Pipeline:
  1. Mode Discovery     — which modes have positive VAL PnL?
  2. Composite Score     — score_pnl2 = mean_va_pnl² × agree_18
  3. Alpha Agreement     — 400 AND 410 must be positive in VAL
  4. Cross-Alpha Seed Stability — min(agree_400, agree_410)
  5. Degradation Filter  — reject >70% OPT→VAL decay
  6. Final Selection     — ranked deployment candidates

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
MV_STITCHED = 'robust_wf_v12_short_stitched'
_EXP_400_FAMILY = ('400', '401', '402', '403', '404', '405', 'WF400')
DEPLOY_PNL_THRESHOLD = 0.018
MIN_SEEDS = 3
MAX_VA_OPT_RATIO = 3.0
CROSS_AGREE_HIGH = 0.6
CROSS_AGREE_MED = 0.4
MIN_RETENTION = 0.3
EXCLUDE_MODES = {'limit_entry'}

GRP_KEYS = ['ticker', 'experiment', 'tpsl_mode', 'entry_minute',
            'prob_thresh', 'tp_pct', 'sl_pct',
            'tp_mult', 'sl_mult', 'trail_start_mult', 'trail_end_mult',
            'grace_minutes', 'dip_pct', 'max_wait_minutes',
            'vol_filter_pct']


# ==============================================================================
# LOAD — split by experiment prefix (not model_version)
# ==============================================================================
print(f"{'='*W}")
print(f"  SHORT MODEL DEPLOYMENT SELECTION — Exp 400 (α=2.0) vs Exp 410 (α=1.0)")
print(f"  Selection on VAL_A only (Feb–Mar 2026). TEST shown for reference.")
print(f"{'='*W}")

q = f"""
SELECT
    model_version,
    ticker, experiment, seed, tpsl_mode, entry_time, entry_minute,
    prob_thresh, tp_pct, sl_pct, tp_mult, sl_mult,
    trail_start_mult, trail_end_mult,
    grace_minutes, dip_pct, max_wait_minutes,
    vol_filter_pct, pre_range_filter_pct,
    vix_threshold, tlt_crash, spy_ignore,
    opt_avg_pnl, opt_win_rate, opt_freq, opt_count,
    val_a_avg_pnl, val_a_win_rate, val_a_freq, val_a_count, survived_a,
    val_b_avg_pnl, val_b_win_rate, val_b_freq, val_b_count, survived_b,
    is_ensemble, n_seeds_avg,
    run_timestamp
FROM `{BQ_TABLE}`
WHERE model_version IN ('{MV_400}', '{MV_410}', '{MV_STITCHED}')
  AND NOT experiment LIKE '401%'
  AND NOT experiment LIKE '402%'
ORDER BY experiment, ticker, tpsl_mode, seed
"""
df_all = bq.query(q).to_dataframe()

df_400 = df_all[df_all['experiment'].str.startswith(_EXP_400_FAMILY)].copy()
df_410 = df_all[df_all['experiment'].str.startswith('410')].copy()

EXPS = {'400': df_400, '410': df_410}
EXP_LABELS = {'400': '400 (α=2.0, Aug)', '410': '410 (α=1.0, Aug)'}

ps = {}
for eid, df in EXPS.items():
    ps[eid] = df[df['is_ensemble'] != True].copy()

for col in ['grace_minutes', 'max_wait_minutes']:
    for eid in EXPS:
        if col in ps[eid].columns:
            ps[eid][col] = ps[eid][col].fillna(0).astype(int)
for col in ['dip_pct', 'tp_mult', 'sl_mult', 'trail_start_mult', 'trail_end_mult',
            'vol_filter_pct']:
    for eid in EXPS:
        if col in ps[eid].columns:
            ps[eid][col] = ps[eid][col].fillna(0.0)

tsets = {eid: set(ps[eid]['ticker'].unique()) for eid in EXPS}
gate_overlap = sorted(tsets['400'] & tsets['410'])
all_tickers = sorted(tsets['400'] | tsets['410'])

print(f"\n  Loaded {len(df_all)} total rows from BQ")
for eid in EXPS:
    print(f"  Exp {EXP_LABELS[eid]}: {len(ps[eid])} per-seed rows, "
          f"{len(tsets[eid])} tickers")
print(f"\n  400∩410 overlap: {len(gate_overlap)}  |  Total unique: {len(all_tickers)}")

if len(df_all) == 0:
    print(f"\n  No results yet.")
    raise SystemExit


# ==============================================================================
# STEP 1: MODE DISCOVERY (based on VAL_A)
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  STEP 1: MODE DISCOVERY (VAL_A period)")
print(f"{'='*W}")
print(f"\n  Which TPSL modes produce positive VAL PnL?")
if EXCLUDE_MODES:
    print(f"  Hard-excluded: {', '.join(sorted(EXCLUDE_MODES))}")

all_modes = sorted(set().union(*(ps[e]['tpsl_mode'].dropna().unique() for e in EXPS if len(ps[e]) > 0)))

hdr = f"  {'Mode':<15}"
for eid in EXPS:
    hdr += f" │ {'── '+eid+' ──────────────':}"
print(f"\n{hdr}")
sub = f"  {'':15}"
for _ in EXPS:
    sub += f" │ {'VAL':>8} {'V+':>4} {'Tickers':>7}"
print(sub)
sep = f"  {'─'*15}"
for _ in EXPS:
    sep += f" │ {'─'*8} {'─'*4} {'─'*7}"
print(sep)

winning_modes = []
for mode in all_modes:
    parts = []
    val_means = {}
    for eid in EXPS:
        md = ps[eid][ps[eid]['tpsl_mode'] == mode]
        if len(md) >= 3:
            v = md['val_a_avg_pnl'].mean()
            vp = (md['val_a_avg_pnl'] > 0).sum()
            tk = md['ticker'].nunique()
            parts.append(f"{v:>+7.2%} {vp:>4} {tk:>7}")
            val_means[eid] = v
        else:
            parts.append(f"{'—':>8} {'—':>4} {'—':>7}")

    excluded = mode in EXCLUDE_MODES
    n_pos = sum(1 for v in val_means.values() if v > 0)
    if n_pos >= 2 and not excluded:
        winning_modes.append(mode)
    status = ' EXCLUDED' if excluded else (' ✓' if n_pos >= 2 else '')
    print(f"  {mode:<15}" + " │ ".join([''] + parts) + f" │{status}")

if not winning_modes:
    eligible = [m for m in all_modes if m not in EXCLUDE_MODES]
    if eligible:
        print(f"\n  No mode positive in both experiments. Falling back to: {', '.join(eligible)}")
        winning_modes = eligible
    else:
        print(f"\n  No eligible modes. Wait for optimizer results.")
        raise SystemExit

print(f"\n  Winning modes: {', '.join(winning_modes)}")


# ==============================================================================
# STEP 2: COMPOSITE SCORE (VAL_A only)
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  STEP 2: COMPOSITE SCORE — score_pnl2 = mean_va_pnl² × agree_18")
print(f"{'='*W}")
print(f"\n  Mode filter: {', '.join(winning_modes)}")
print(f"  Thresholds: DEPLOY={DEPLOY_PNL_THRESHOLD:.1%}, MIN_SEEDS={MIN_SEEDS}, "
      f"MAX_OPT_RATIO={MAX_VA_OPT_RATIO}")


def compute_seed_groups(per_seed_df, label):
    indiv = per_seed_df.copy()
    indiv = indiv[indiv['opt_avg_pnl'] > 0].copy()
    indiv = indiv[indiv['val_a_avg_pnl'] > 0].copy()
    indiv = indiv[indiv['val_a_avg_pnl'] / indiv['opt_avg_pnl'] <= MAX_VA_OPT_RATIO].copy()

    extra_agg = {}
    for c, fn in [('vix_threshold', 'max'), ('tlt_crash', 'min'), ('spy_ignore', 'max')]:
        if c in indiv.columns and indiv[c].notna().any():
            extra_agg[c] = (c, fn)

    available_grp = [k for k in GRP_KEYS if k in indiv.columns]
    if len(indiv) == 0:
        print(f"\n  {label}: 0 rows after filters")
        return pd.DataFrame()

    sa = indiv.groupby(available_grp).agg(
        n_seeds=('seed', 'nunique'),
        n_pos_seeds_18=('val_a_avg_pnl', lambda x: (x >= DEPLOY_PNL_THRESHOLD).sum()),
        mean_va_pnl=('val_a_avg_pnl', 'mean'),
        mean_va_wr=('val_a_win_rate', 'mean'),
        mean_va_count=('val_a_count', 'mean'),
        vb_pnl=('val_b_avg_pnl', 'mean'),
        mean_vb_wr=('val_b_win_rate', 'mean'),
        mean_vb_count=('val_b_count', 'mean'),
        mean_opt_pnl=('opt_avg_pnl', 'mean'),
        mean_opt_wr=('opt_win_rate', 'mean'),
        mean_opt_count=('opt_count', 'mean'),
        **extra_agg,
    ).reset_index()
    sa['agree_18'] = (sa['n_pos_seeds_18'] / sa['n_seeds']).clip(upper=1.0)

    sa_filt = sa[
        (sa['n_seeds'] >= MIN_SEEDS) &
        (sa['tpsl_mode'].isin(winning_modes))
    ].copy()
    sa_filt['score_pnl2'] = (sa_filt['mean_va_pnl'] ** 2) * sa_filt['agree_18']
    sa_filt = sa_filt[np.isfinite(sa_filt['score_pnl2'])]

    print(f"\n  {label}: {len(indiv)} rows → {len(sa)} groups → "
          f"{len(sa_filt)} after filters ({sa_filt['ticker'].nunique()} tickers)")
    return sa_filt


def best_config_per_ticker(sa_df):
    if sa_df.empty:
        return pd.DataFrame()
    return sa_df.loc[sa_df.groupby('ticker')['score_pnl2'].idxmax()].copy()


# Run pipeline for each experiment (per-seed only, matching LONG strategy)
sa = {}
best = {}
for eid in EXPS:
    sa[eid] = compute_seed_groups(ps[eid], f"Exp {eid}")
    best[eid] = best_config_per_ticker(sa[eid])
    if not best[eid].empty:
        best[eid]['source'] = 'per_seed'

best_400 = best['400']
best_410 = best['410']

if not best_400.empty:
    print(f"\n  Exp 400 best: {len(best_400)} tickers, "
          f"mean VAL PnL: {best_400['mean_va_pnl'].mean():+.3%}, "
          f"mean TEST PnL: {best_400['vb_pnl'].mean():+.3%}")
if not best_410.empty:
    print(f"  Exp 410 best: {len(best_410)} tickers, "
          f"mean VAL PnL: {best_410['mean_va_pnl'].mean():+.3%}, "
          f"mean TEST PnL: {best_410['vb_pnl'].mean():+.3%}")


# ==============================================================================
# STEP 3: ALPHA AGREEMENT GATE — 400 vs 410, VAL_A only
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  STEP 3: ALPHA AGREEMENT GATE — Exp 400 vs 410 (VAL_A only)")
print(f"{'='*W}")

tickers_400 = set(best_400['ticker'].unique()) if not best_400.empty else set()
tickers_410 = set(best_410['ticker'].unique()) if not best_410.empty else set()
gate_tickers = sorted(tickers_400 & tickers_410)
only_400 = sorted(tickers_400 - tickers_410)
only_410 = sorted(tickers_410 - tickers_400)

print(f"\n  Exp 400: {len(tickers_400)} tickers")
print(f"  Exp 410: {len(tickers_410)} tickers")
print(f"  Gate overlap: {len(gate_tickers)}")

gate_pass = []
gate_fail_reasons = {}

for ticker in gate_tickers:
    r400 = best_400[best_400['ticker'] == ticker].iloc[0]
    r410 = best_410[best_410['ticker'] == ticker].iloc[0]

    va_400 = r400['mean_va_pnl']
    va_410 = r410['mean_va_pnl']

    if va_400 > 0 and va_410 > 0:
        selected_alpha = '400' if r400['score_pnl2'] >= r410['score_pnl2'] else '410'
        gate_pass.append({
            'ticker': ticker,
            'selected_alpha': selected_alpha,
            'va_400': va_400, 'va_410': va_410,
            'va_wr_400': r400.get('mean_va_wr', 0), 'va_wr_410': r410.get('mean_va_wr', 0),
            'va_n_400': r400.get('mean_va_count', 0), 'va_n_410': r410.get('mean_va_count', 0),
            'vb_400': r400['vb_pnl'], 'vb_410': r410['vb_pnl'],
            'vb_wr_400': r400.get('mean_vb_wr', 0), 'vb_wr_410': r410.get('mean_vb_wr', 0),
            'vb_n_400': r400.get('mean_vb_count', 0), 'vb_n_410': r410.get('mean_vb_count', 0),
            'opt_pnl_400': r400['mean_opt_pnl'], 'opt_pnl_410': r410['mean_opt_pnl'],
            'opt_wr_400': r400.get('mean_opt_wr', 0), 'opt_wr_410': r410.get('mean_opt_wr', 0),
            'opt_n_400': r400.get('mean_opt_count', 0), 'opt_n_410': r410.get('mean_opt_count', 0),
            'score_400': r400['score_pnl2'], 'score_410': r410['score_pnl2'],
            'agree_400': r400['agree_18'], 'agree_410': r410['agree_18'],
            'mode_400': r400.get('tpsl_mode', ''), 'mode_410': r410.get('tpsl_mode', ''),
            'prob_400': r400.get('prob_thresh', 0), 'prob_410': r410.get('prob_thresh', 0),
            'tp_400': r400.get('tp_pct', 0), 'tp_410': r410.get('tp_pct', 0),
            'sl_400': r400.get('sl_pct', 0), 'sl_410': r410.get('sl_pct', 0),
        })
    else:
        reason = ('400-neg' if va_400 <= 0 else '') + \
                 (' ' if va_400 <= 0 and va_410 <= 0 else '') + \
                 ('410-neg' if va_410 <= 0 else '')
        gate_fail_reasons[ticker] = reason.strip()

for ticker in only_400:
    gate_fail_reasons[ticker] = '410-missing'
for ticker in only_410:
    gate_fail_reasons[ticker] = '400-missing'

print(f"\n  Gate Results:")
print(f"    PASS (both alphas positive VAL): {len(gate_pass)}")
print(f"    FAIL: {len(gate_fail_reasons)}")
if gate_fail_reasons:
    for reason in sorted(set(gate_fail_reasons.values())):
        tks = [t for t, r in gate_fail_reasons.items() if r == reason]
        print(f"      {reason}: {', '.join(tks[:20])}{'...' if len(tks) > 20 else ''}")

if not gate_pass:
    print(f"\n  No tickers pass the alpha agreement gate.")

    for label, best_one in [('400 (α=2.0)', best_400), ('410 (α=1.0)', best_410)]:
        if best_one.empty:
            continue
        top = best_one.nlargest(10, 'score_pnl2')
        print(f"\n  Top Exp {label} (no gate):")
        print(f"  {'Ticker':<8} {'Mode':<13} {'VAL PnL':>8} {'(TEST)':>8} "
              f"{'Agree18':>8} {'Score':>10}")
        print(f"  {'─'*8} {'─'*13} {'─'*8} {'─'*8} {'─'*8} {'─'*10}")
        for _, r in top.iterrows():
            print(f"  {r['ticker']:<8} {r.get('tpsl_mode', '?'):<13} "
                  f"{r['mean_va_pnl']:>+7.2%} {r['vb_pnl']:>+7.2%} "
                  f"{r['agree_18']:>7.0%} {r['score_pnl2']:>10.6f}")

gate_df = pd.DataFrame(gate_pass)


# ==============================================================================
# STEP 4: CROSS-ALPHA SEED STABILITY
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  STEP 4: CROSS-ALPHA SEED STABILITY")
print(f"{'='*W}")

if not gate_df.empty:
    gate_df['cross_agree'] = gate_df[['agree_400', 'agree_410']].min(axis=1)
    gate_df['confidence'] = gate_df['cross_agree'].apply(
        lambda x: 'HIGH' if x >= CROSS_AGREE_HIGH else ('MED' if x >= CROSS_AGREE_MED else 'LOW'))

    print(f"\n  Cross-alpha stability = min(agree_400, agree_410)")
    print(f"  HIGH >= {CROSS_AGREE_HIGH:.0%}  |  MED >= {CROSS_AGREE_MED:.0%}  |  LOW < {CROSS_AGREE_MED:.0%}")
    print(f"\n  {'Ticker':<8} {'Agree_400':>10} {'Agree_410':>10} "
          f"{'CrossAgree':>11} {'Conf':>6}")
    print(f"  {'─'*8} {'─'*10} {'─'*10} {'─'*11} {'─'*6}")
    for _, r in gate_df.sort_values('cross_agree', ascending=False).iterrows():
        print(f"  {r['ticker']:<8} {r['agree_400']:>9.0%} "
              f"{r['agree_410']:>9.0%} {r['cross_agree']:>10.0%} {r['confidence']:>6}")

    conf_counts = gate_df['confidence'].value_counts()
    print(f"\n  Distribution: "
          f"HIGH={conf_counts.get('HIGH', 0)}  "
          f"MED={conf_counts.get('MED', 0)}  "
          f"LOW={conf_counts.get('LOW', 0)}")
else:
    print(f"\n  No tickers passed the gate — skipping.")


# ==============================================================================
# STEP 5: DEGRADATION FILTER — OPT → VAL retention
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  STEP 5: DEGRADATION FILTER — OPT → VAL retention")
print(f"{'='*W}")
print(f"\n  Reject if VAL/OPT retention < {MIN_RETENTION:.0%}")

if not gate_df.empty:
    gate_df['opt_pnl_sel'] = gate_df.apply(
        lambda r: r['opt_pnl_400'] if r['selected_alpha'] == '400' else r['opt_pnl_410'], axis=1)
    gate_df['va_pnl_sel'] = gate_df.apply(
        lambda r: r['va_400'] if r['selected_alpha'] == '400' else r['va_410'], axis=1)
    gate_df['retention'] = np.where(
        gate_df['opt_pnl_sel'] > 0,
        gate_df['va_pnl_sel'] / gate_df['opt_pnl_sel'], 0.0)

    print(f"\n  {'Ticker':<8} {'Alpha':>6} {'OPT':>8} {'VAL':>9} "
          f"{'Retention':>10} {'Status':>8}")
    print(f"  {'─'*8} {'─'*6} {'─'*8} {'─'*9} {'─'*10} {'─'*8}")
    for _, r in gate_df.sort_values('retention', ascending=False).iterrows():
        status = 'PASS' if r['retention'] >= MIN_RETENTION else 'REJECT'
        print(f"  {r['ticker']:<8} {r['selected_alpha']:>6} "
              f"{r['opt_pnl_sel']:>+7.2%} {r['va_pnl_sel']:>+8.2%} "
              f"{r['retention']:>9.0%} {status:>8}")

    n_before = len(gate_df)
    gate_df = gate_df[gate_df['retention'] >= MIN_RETENTION].copy()
    n_rejected = n_before - len(gate_df)
    print(f"\n  {n_rejected} rejected for excessive degradation.")
else:
    print(f"\n  No tickers to filter.")


# ==============================================================================
# STEP 6: FINAL DEPLOYMENT CANDIDATES
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  STEP 6: FINAL DEPLOYMENT CANDIDATES")
print(f"{'='*W}")

if not gate_df.empty:
    gate_df['score_sel'] = gate_df.apply(
        lambda r: r['score_400'] if r['selected_alpha'] == '400' else r['score_410'], axis=1)
    gate_df['mode_sel'] = gate_df.apply(
        lambda r: r['mode_400'] if r['selected_alpha'] == '400' else r['mode_410'], axis=1)
    gate_df['prob_sel'] = gate_df.apply(
        lambda r: r['prob_400'] if r['selected_alpha'] == '400' else r['prob_410'], axis=1)
    gate_df['tp_sel'] = gate_df.apply(
        lambda r: r['tp_400'] if r['selected_alpha'] == '400' else r['tp_410'], axis=1)
    gate_df['sl_sel'] = gate_df.apply(
        lambda r: r['sl_400'] if r['selected_alpha'] == '400' else r['sl_410'], axis=1)
    gate_df['vb_pnl_sel'] = gate_df.apply(
        lambda r: r['vb_400'] if r['selected_alpha'] == '400' else r['vb_410'], axis=1)

    deploy = gate_df[gate_df['va_pnl_sel'] >= DEPLOY_PNL_THRESHOLD].copy()
    deploy = deploy.sort_values('score_sel', ascending=False)

    if len(deploy) == 0:
        print(f"\n  No tickers meet deploy threshold ({DEPLOY_PNL_THRESHOLD:.1%}) after filters.")
        print(f"    {len(gate_df)} passed gate but below VAL PnL threshold.")
    else:
        print(f"\n  {len(deploy)} deployment candidate(s):")
        print(f"  (Selected by VAL only — TEST shown for reference)")

        print(f"\n  {'Ticker':<8} {'α':>4} {'Mode':<13} {'Prob':>5} {'TP':>6} {'SL':>6} │"
              f" {'VAL PnL':>8} {'(TEST)':>8} │"
              f" {'Agree':>6} {'XAgree':>7} {'Retain':>7} {'Conf':>5} │"
              f" {'Score':>10}")
        print(f"  {'─'*8} {'─'*4} {'─'*13} {'─'*5} {'─'*6} {'─'*6} │"
              f" {'─'*8} {'─'*8} │"
              f" {'─'*6} {'─'*7} {'─'*7} {'─'*5} │"
              f" {'─'*10}")

        for _, r in deploy.iterrows():
            agree_sel = r['agree_400'] if r['selected_alpha'] == '400' else r['agree_410']
            print(f"  {r['ticker']:<8} {r['selected_alpha']:>4} "
                  f"{r['mode_sel']:<13} {r['prob_sel']:>4.2f} "
                  f"{r['tp_sel']:>5.1%} {r['sl_sel']:>5.1%} │"
                  f" {r['va_pnl_sel']:>+7.2%} {r['vb_pnl_sel']:>+7.2%} │"
                  f" {agree_sel:>5.0%} {r['cross_agree']:>6.0%} "
                  f"{r['retention']:>6.0%} {r['confidence']:>5} │"
                  f" {r['score_sel']:>10.6f}")

        print(f"\n  ── Summary ──")
        print(f"  Candidates:      {len(deploy)}")
        print(f"  Mean VAL PnL:    {deploy['va_pnl_sel'].mean():+.2%}")
        print(f"  Mean TEST PnL:   {deploy['vb_pnl_sel'].mean():+.2%}  (held-out, not used in selection)")
        print(f"  Mean retention:  {deploy['retention'].mean():.0%}  (VAL/OPT)")
        a400_n = (deploy['selected_alpha'] == '400').sum()
        a410_n = (deploy['selected_alpha'] == '410').sum()
        print(f"  Alpha split:     400(α=2.0)={a400_n}, 410(α=1.0)={a410_n}")

        # ── Detailed stats per candidate ──
        print(f"\n  ── DETAILED STATS PER CANDIDATE ──")
        print(f"  Period dates: OPT=Aug'25–Jan'26  VAL=Feb–Mar'26  TEST=Apr–Jul'26")
        print(f"\n  {'Ticker':<8} {'α':>4} │"
              f" {'── OPT (Aug–Jan) ──────────':} │"
              f" {'── VAL (Feb–Mar) ──────────':} │"
              f" {'── TEST (Apr–Jul) ─────────':}")
        print(f"  {'':8} {'':>4} │"
              f" {'PnL':>8} {'WR':>6} {'#Trd':>5} │"
              f" {'PnL':>8} {'WR':>6} {'#Trd':>5} │"
              f" {'PnL':>8} {'WR':>6} {'#Trd':>5}")
        print(f"  {'─'*8} {'─'*4} │"
              f" {'─'*8} {'─'*6} {'─'*5} │"
              f" {'─'*8} {'─'*6} {'─'*5} │"
              f" {'─'*8} {'─'*6} {'─'*5}")

        for _, r in deploy.iterrows():
            sel = r['selected_alpha']
            opt_p = r[f'opt_pnl_{sel}']
            opt_w = r.get(f'opt_wr_{sel}', 0)
            opt_n = r.get(f'opt_n_{sel}', 0)
            va_p = r[f'va_{sel}']
            va_w = r.get(f'va_wr_{sel}', 0)
            va_n = r.get(f'va_n_{sel}', 0)
            vb_p = r[f'vb_{sel}']
            vb_w = r.get(f'vb_wr_{sel}', 0)
            vb_n = r.get(f'vb_n_{sel}', 0)
            print(f"  {r['ticker']:<8} {sel:>4} │"
                  f" {opt_p:>+7.2%} {opt_w:>5.0%} {opt_n:>5.0f} │"
                  f" {va_p:>+7.2%} {va_w:>5.0%} {va_n:>5.0f} │"
                  f" {vb_p:>+7.2%} {vb_w:>5.0%} {vb_n:>5.0f}")

        print(f"\n  ── Cross-alpha detail (both experiments for each candidate) ──")
        print(f"\n  {'Ticker':<8} │"
              f" {'── Exp 400 (α=2.0) ─────────────────────':} │"
              f" {'── Exp 410 (α=1.0) ─────────────────────':} │ {'Sel':>3}")
        print(f"  {'':8} │"
              f" {'VAL PnL':>8} {'WR':>5} {'#':>4}  {'TST PnL':>8} {'WR':>5} {'#':>4} │"
              f" {'VAL PnL':>8} {'WR':>5} {'#':>4}  {'TST PnL':>8} {'WR':>5} {'#':>4} │")
        print(f"  {'─'*8} │"
              f" {'─'*8} {'─'*5} {'─'*4}  {'─'*8} {'─'*5} {'─'*4} │"
              f" {'─'*8} {'─'*5} {'─'*4}  {'─'*8} {'─'*5} {'─'*4} │ {'─'*3}")

        for _, r in deploy.iterrows():
            print(f"  {r['ticker']:<8} │"
                  f" {r['va_400']:>+7.2%} {r['va_wr_400']:>4.0%} {r['va_n_400']:>4.0f}"
                  f"  {r['vb_400']:>+7.2%} {r['vb_wr_400']:>4.0%} {r['vb_n_400']:>4.0f} │"
                  f" {r['va_410']:>+7.2%} {r['va_wr_410']:>4.0%} {r['va_n_410']:>4.0f}"
                  f"  {r['vb_410']:>+7.2%} {r['vb_wr_410']:>4.0%} {r['vb_n_410']:>4.0f} │"
                  f" {r['selected_alpha']:>3}")
else:
    deploy = pd.DataFrame()
    print(f"\n  No tickers available for deployment selection.")
    print(f"    Re-run after both optimizers process more tickers.")


# ── Watch-list (gate pass, below deploy threshold) ──
print(f"\n  ── WATCH LIST (gate pass, VAL < {DEPLOY_PNL_THRESHOLD:.1%}) ──")
if not gate_df.empty and 'va_pnl_sel' in gate_df.columns:
    watch = gate_df[
        gate_df['va_pnl_sel'] < DEPLOY_PNL_THRESHOLD
    ].sort_values('va_pnl_sel', ascending=False)

    if len(watch) > 0:
        print(f"  {len(watch)} near-deploy ticker(s):")
        print(f"\n  {'Ticker':<8} {'α':>4} {'Mode':<13} {'VAL PnL':>8} {'(TEST)':>8} "
              f"{'XAgree':>7} {'Conf':>5}")
        print(f"  {'─'*8} {'─'*4} {'─'*13} {'─'*8} {'─'*8} {'─'*7} {'─'*5}")
        for _, r in watch.iterrows():
            print(f"  {r['ticker']:<8} {r['selected_alpha']:>4} "
                  f"{r.get('mode_sel', '?'):<13} "
                  f"{r['va_pnl_sel']:>+7.2%} {r['vb_pnl_sel']:>+7.2%} "
                  f"{r.get('cross_agree', 0):>6.0%} {r.get('confidence', '?'):>5}")
    else:
        print(f"  None.")
else:
    print(f"  None — no tickers passed the gate.")


# ── Single-alpha standouts ──
print(f"\n  ── SINGLE-ALPHA STANDOUTS (strong in one alpha, missing from other) ──")
standouts = []
for label, best_one, other_set in [('400 (α=2.0)', best_400, tickers_410),
                                     ('410 (α=1.0)', best_410, tickers_400)]:
    if best_one.empty:
        continue
    solo = best_one[~best_one['ticker'].isin(other_set)].copy()
    if not solo.empty:
        solo = solo.sort_values('score_pnl2', ascending=False)
        standouts.append((label, solo))

if standouts:
    for label, solo in standouts:
        top = solo.head(10)
        print(f"\n  {label} only ({len(solo)} tickers, top 10 by score):")
        print(f"  {'Ticker':<8} {'Mode':<13} {'VAL PnL':>8} {'(TEST)':>8} "
              f"{'Agree18':>8} {'Score':>10}")
        print(f"  {'─'*8} {'─'*13} {'─'*8} {'─'*8} {'─'*8} {'─'*10}")
        for _, r in top.iterrows():
            print(f"  {r['ticker']:<8} {r.get('tpsl_mode', '?'):<13} "
                  f"{r['mean_va_pnl']:>+7.2%} {r['vb_pnl']:>+7.2%} "
                  f"{r.get('agree_18', 0):>7.0%} {r.get('score_pnl2', 0):>10.6f}")
else:
    print(f"  None.")


# ==============================================================================
# STEP 7: CHARTS
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  STEP 7: CHARTS")
print(f"{'='*W}")

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('SHORT Model Deploy Selection — Exp 400 (α=2.0) vs 410 (α=1.0) — VAL_A only',
             fontsize=14, fontweight='bold')

# Panel 1: Alpha agreement scatter (400 vs 410, VAL PnL)
ax = axes[0, 0]
scatter_tickers = []
if not best_400.empty and not best_410.empty:
    for ticker in gate_tickers:
        r4 = best_400[best_400['ticker'] == ticker]
        r41 = best_410[best_410['ticker'] == ticker]
        if not r4.empty and not r41.empty:
            scatter_tickers.append((ticker, r4.iloc[0]['mean_va_pnl'], r41.iloc[0]['mean_va_pnl']))

if scatter_tickers:
    tks, x_vals, y_vals = zip(*scatter_tickers)
    x_vals, y_vals = np.array(x_vals), np.array(y_vals)
    colors = ['green' if (x > 0 and y > 0) else 'red' for x, y in zip(x_vals, y_vals)]
    ax.scatter(x_vals * 100, y_vals * 100, c=colors, s=50, alpha=0.7, edgecolors='k', linewidth=0.5)
    for t, x, y in zip(tks, x_vals * 100, y_vals * 100):
        ax.annotate(t, (x, y), fontsize=6, ha='center', va='bottom')
    lim = max(abs(x_vals).max(), abs(y_vals).max()) * 100 * 1.2
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.fill_between([-lim, 0], -lim, 0, alpha=0.05, color='red')
    ax.fill_between([0, lim], 0, lim, alpha=0.05, color='green')
ax.set_xlabel('Exp 400 (α=2.0) VAL PnL (%)')
ax.set_ylabel('Exp 410 (α=1.0) VAL PnL (%)')
ax.set_title('Alpha Agreement — VAL_A Period')

# Panel 2: Deploy candidates bar (VAL PnL)
ax = axes[0, 1]
if not deploy.empty:
    deploy_sorted = deploy.sort_values('va_pnl_sel', ascending=True)
    colors = ['#1f77b4' if a == '400' else '#ff7f0e' for a in deploy_sorted['selected_alpha']]
    ax.barh(range(len(deploy_sorted)), deploy_sorted['va_pnl_sel'] * 100,
            color=colors, edgecolor='k', linewidth=0.5)
    ax.set_yticks(range(len(deploy_sorted)))
    ax.set_yticklabels(deploy_sorted['ticker'], fontsize=8)
    ax.axvline(DEPLOY_PNL_THRESHOLD * 100, color='red', linestyle='--', linewidth=1)
    ax.axvline(0, color='gray', linewidth=0.5)
    import matplotlib.patches as mpatches
    p_400 = mpatches.Patch(color='#1f77b4', label='400 (α=2.0)')
    p_410 = mpatches.Patch(color='#ff7f0e', label='410 (α=1.0)')
    ax.legend(handles=[p_400, p_410], fontsize=8)
    ax.set_xlabel('VAL PnL (%)')
else:
    ax.text(0.5, 0.5, 'No deploy candidates yet', ha='center', va='center',
            transform=ax.transAxes, fontsize=12, color='gray')
ax.set_title('Deploy Candidates — VAL PnL')

# Panel 3: Cross-agree by ticker
ax = axes[1, 0]
if not gate_df.empty:
    gate_sorted = gate_df.sort_values('cross_agree', ascending=True)
    colors = ['green' if c == 'HIGH' else ('orange' if c == 'MED' else 'red')
              for c in gate_sorted['confidence']]
    ax.barh(range(len(gate_sorted)), gate_sorted['cross_agree'] * 100,
            color=colors, edgecolor='k', linewidth=0.5)
    ax.set_yticks(range(len(gate_sorted)))
    ax.set_yticklabels(gate_sorted['ticker'], fontsize=8)
    ax.axvline(CROSS_AGREE_HIGH * 100, color='green', linestyle='--', linewidth=1,
               label=f'HIGH ({CROSS_AGREE_HIGH:.0%})')
    ax.axvline(CROSS_AGREE_MED * 100, color='orange', linestyle='--', linewidth=1,
               label=f'MED ({CROSS_AGREE_MED:.0%})')
    ax.legend(fontsize=8)
    ax.set_xlabel('Cross-Alpha Seed Agreement (%)')
else:
    ax.text(0.5, 0.5, 'No data yet', ha='center', va='center',
            transform=ax.transAxes, fontsize=12, color='gray')
ax.set_title('Cross-Alpha Seed Stability')

# Panel 4: Score distributions (400 vs 410)
ax = axes[1, 1]
has_data = False
for eid, color, label in [('400', '#1f77b4', '400 (α=2.0)'),
                           ('410', '#2ca02c', '410 (α=1.0)')]:
    if not sa[eid].empty:
        ax.hist(sa[eid]['score_pnl2'] * 10000, bins=30, alpha=0.4, color=color,
                label=label, edgecolor='k', linewidth=0.3)
        has_data = True
if has_data:
    ax.axvline(DEPLOY_PNL_THRESHOLD**2 * 10000, color='red', linestyle='--',
               label=f'Deploy²', linewidth=1)
    ax.legend(fontsize=8)
    ax.set_xlabel('score_pnl2 (x10⁴)')
    ax.set_ylabel('Count')
else:
    ax.text(0.5, 0.5, 'No data yet', ha='center', va='center',
            transform=ax.transAxes, fontsize=12, color='gray')
ax.set_title('Composite Score Distribution')

plt.tight_layout()
SAVE_PATH = '/tmp/select_short_deploy.png'
plt.savefig(SAVE_PATH, dpi=150, bbox_inches='tight')
print(f"\n  Charts saved to {SAVE_PATH}")
plt.show()


# ==============================================================================
# SUMMARY
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  SUMMARY")
print(f"{'='*W}")
print(f"\n  Data:        {len(all_tickers)} tickers "
      f"(400:{len(tsets['400'])}, 410:{len(tsets['410'])})")
print(f"  Win modes:   {', '.join(winning_modes)}")
print(f"  Excluded:    {', '.join(sorted(EXCLUDE_MODES))}")
print(f"  Gate pass:   {len(gate_pass)} / {len(gate_tickers)} overlap tickers")
print(f"  Deploy:      {len(deploy)} candidate(s)")
n_watch = 0
if 'watch' in dir() and isinstance(watch, pd.DataFrame) and not watch.empty:
    n_watch = len(watch)
print(f"  Watch-list:  {n_watch} near-deploy ticker(s)")
if not deploy.empty:
    print(f"  Best ticker: {deploy.iloc[0]['ticker']} "
          f"(α={'2.0' if deploy.iloc[0]['selected_alpha'] == '400' else '1.0'}, "
          f"VAL={deploy.iloc[0]['va_pnl_sel']:+.2%})")
print(f"\n  Selection uses VAL_A only (Feb–Mar 2026).")
print(f"  TEST (Apr+) shown for reference, not used in any gate.")
print(f"{'='*W}")
