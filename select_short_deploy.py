"""
select_short_deploy.py — SHORT Model Deployment Selection (Exp 400/410)
========================================================================
Adapts the S3 composite score + family agreement pipeline for SHORT/dump
models. Two "families" = two alpha_pos variants (400=2.0, 410=1.0).

Pipeline:
  1. Mode Discovery     — which modes work for SHORT?
  2. Composite Score     — score_pnl2 = mean_va_pnl² × agree_18
  3. Alpha Agreement     — both alphas must be positive OOS (family gate analog)
  4. Cross-Alpha Seed Stability — min(agree_18_400, agree_18_410)
  5. Degradation Filter  — reject >70% OPT→TEST decay
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
# CONFIG — mirrors daily_score_combined.py thresholds exactly
# ==============================================================================
W = 130
BQ_TABLE = 'ys-time-series.scoring_logs.wf_optimization_results'
MV_400 = 'robust_wf_v12_short'
MV_410 = 'robust_wf_v12_short_a1'
DEPLOY_PNL_THRESHOLD = 0.018
MIN_TEST_TRADES = 3
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
# LOAD — single query, split by model_version
# ==============================================================================
print(f"{'═'*W}")
print(f"  SHORT MODEL DEPLOYMENT SELECTION — Exp 400 (α=2.0) & Exp 410 (α=1.0)")
print(f"{'═'*W}")

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
WHERE model_version IN ('{MV_400}', '{MV_410}')
ORDER BY model_version, ticker, tpsl_mode, seed
"""
df_all = bq.query(q).to_dataframe()

df_400 = df_all[df_all['model_version'] == MV_400].copy()
df_410 = df_all[df_all['model_version'] == MV_410].copy()

ps_400 = df_400[df_400['is_ensemble'] != True].copy()
en_400 = df_400[df_400['is_ensemble'] == True].copy()
ps_410 = df_410[df_410['is_ensemble'] != True].copy()
en_410 = df_410[df_410['is_ensemble'] == True].copy()

t400 = set(ps_400['ticker'].unique())
t410 = set(ps_410['ticker'].unique())
overlap = sorted(t400 & t410)
all_tickers = sorted(t400 | t410)

print(f"\n  Exp 400 (α=2.0): {len(ps_400)} per-seed, {len(en_400)} ensemble, "
      f"{len(t400)} tickers")
print(f"  Exp 410 (α=1.0): {len(ps_410)} per-seed, {len(en_410)} ensemble, "
      f"{len(t410)} tickers")
print(f"  Overlap: {len(overlap)}  |  400-only: {len(t400 - t410)}  |  "
      f"410-only: {len(t410 - t400)}")
print(f"\n  Scored by BOTH ({len(overlap)}): "
      f"{', '.join(overlap) if overlap else 'none yet'}")

if len(df_all) == 0:
    print(f"\n  ⚠ NO RESULTS YET — optimizers haven't written to BQ.")
    raise SystemExit

for col in ['grace_minutes', 'max_wait_minutes']:
    for d in [ps_400, ps_410, en_400, en_410]:
        if col in d.columns:
            d[col] = d[col].fillna(0).astype(int)
for col in ['dip_pct', 'tp_mult', 'sl_mult', 'trail_start_mult', 'trail_end_mult',
            'vol_filter_pct']:
    for d in [ps_400, ps_410, en_400, en_410]:
        if col in d.columns:
            d[col] = d[col].fillna(0.0)


# ==============================================================================
# STEP 1: MODE DISCOVERY — which modes work for SHORT?
# ==============================================================================
print(f"\n\n{'═'*W}")
print(f"  STEP 1: MODE DISCOVERY")
print(f"{'═'*W}")
print(f"\n  Which TPSL modes produce positive TEST PnL for each alpha?")
if EXCLUDE_MODES:
    print(f"  Hard-excluded modes: {', '.join(sorted(EXCLUDE_MODES))} "
          f"(confirmed toxic for SHORT — 73-81% flipped negative)")

all_modes = sorted(set(ps_400['tpsl_mode'].dropna().unique()) |
                   set(ps_410['tpsl_mode'].dropna().unique()))

print(f"\n  {'Mode':<15} │ {'── EXP 400 (α=2.0) ──────────':} │ "
      f"{'── EXP 410 (α=1.0) ──────────':} │ {'Both+':} │ {'':}")
print(f"  {'':15} │ {'TEST':>8} {'T+':>4} {'WR':>6} {'Tickers':>8} │ "
      f"{'TEST':>8} {'T+':>4} {'WR':>6} {'Tickers':>8} │ {'':>5} │")
print(f"  {'─'*15} │ {'─'*8} {'─'*4} {'─'*6} {'─'*8} │ "
      f"{'─'*8} {'─'*4} {'─'*6} {'─'*8} │ {'─'*5} │ {'─'*8}")

winning_modes = []
for mode in all_modes:
    md4 = ps_400[ps_400['tpsl_mode'] == mode]
    md1 = ps_410[ps_410['tpsl_mode'] == mode]

    t4 = md4['val_b_avg_pnl'].mean() if len(md4) else np.nan
    tp4 = (md4['val_b_avg_pnl'] > 0).sum() if len(md4) else 0
    wr4 = md4['val_b_win_rate'].mean() if len(md4) else np.nan
    tk4 = md4['ticker'].nunique() if len(md4) else 0

    t1 = md1['val_b_avg_pnl'].mean() if len(md1) else np.nan
    tp1 = (md1['val_b_avg_pnl'] > 0).sum() if len(md1) else 0
    wr1 = md1['val_b_win_rate'].mean() if len(md1) else np.nan
    tk1 = md1['ticker'].nunique() if len(md1) else 0

    excluded = mode in EXCLUDE_MODES
    both_pos = '✓' if (pd.notna(t4) and t4 > 0 and pd.notna(t1) and t1 > 0 and not excluded) else ''
    if both_pos:
        winning_modes.append(mode)

    s4 = (f"{t4:>+7.2%} {tp4:>4} {wr4:>5.1%} {tk4:>8}"
          if pd.notna(t4) else f"{'—':>8} {'—':>4} {'—':>6} {'—':>8}")
    s1 = (f"{t1:>+7.2%} {tp1:>4} {wr1:>5.1%} {tk1:>8}"
          if pd.notna(t1) else f"{'—':>8} {'—':>4} {'—':>6} {'—':>8}")
    status = ' EXCLUDED' if excluded else ''
    print(f"  {mode:<15} │ {s4} │ {s1} │ {both_pos:>5} │ {status}")

if winning_modes:
    print(f"\n  ★ WINNING MODES (positive mean TEST in BOTH alphas): {', '.join(winning_modes)}")
    print(f"    → These modes will be used for deployment selection below.")
else:
    eligible_modes = [m for m in all_modes
                      if m not in EXCLUDE_MODES and (
                          len(ps_400[ps_400['tpsl_mode'] == m]) > 0 or
                          len(ps_410[ps_410['tpsl_mode'] == m]) > 0)]
    if eligible_modes:
        print(f"\n  ⚠ No mode has positive mean TEST PnL in BOTH alphas.")
        print(f"    Falling back to eligible modes (excl. {', '.join(sorted(EXCLUDE_MODES))}): "
              f"{', '.join(eligible_modes)}")
        winning_modes = eligible_modes
    else:
        print(f"\n  ⚠ No eligible mode data available. Wait for optimizer results.")
        raise SystemExit


# ==============================================================================
# STEP 2: COMPOSITE SCORE + ALPHA AGREEMENT GATE
# ==============================================================================
print(f"\n\n{'═'*W}")
print(f"  STEP 2: COMPOSITE SCORE + ALPHA AGREEMENT GATE")
print(f"{'═'*W}")
print(f"\n  Replicating S3 pipeline: seed aggregation → score_pnl2 → alpha agreement")
print(f"  Mode filter: {', '.join(winning_modes)}")
print(f"  Thresholds: DEPLOY={DEPLOY_PNL_THRESHOLD:.1%}, MIN_SEEDS={MIN_SEEDS}, "
      f"MAX_OPT_RATIO={MAX_VA_OPT_RATIO}")


def compute_seed_groups(per_seed_df, label):
    """Seed group aggregation — mirrors daily_score_combined.py lines 226-244."""
    indiv = per_seed_df.copy()

    indiv = indiv[indiv['opt_avg_pnl'] > 0].copy()
    n_before = len(indiv)
    indiv = indiv[indiv['val_a_avg_pnl'] > 0].copy()
    ov = indiv[indiv['val_a_avg_pnl'] / indiv['opt_avg_pnl'] <= MAX_VA_OPT_RATIO].copy()
    indiv = ov

    extra_agg = {}
    for c, fn in [('vix_threshold', 'max'), ('tlt_crash', 'min'), ('spy_ignore', 'max')]:
        if c in indiv.columns and indiv[c].notna().any():
            extra_agg[c] = (c, fn)

    available_grp = [k for k in GRP_KEYS if k in indiv.columns]

    if len(indiv) == 0:
        print(f"\n  {label}: 0 rows after filters — no configs to aggregate.")
        return pd.DataFrame()

    sa = indiv.groupby(available_grp).agg(
        n_seeds=('seed', 'nunique'),
        n_pos_seeds_18=('val_a_avg_pnl', lambda x: (x >= DEPLOY_PNL_THRESHOLD).sum()),
        mean_va_pnl=('val_a_avg_pnl', 'mean'),
        mean_va_wr=('val_a_win_rate', 'mean'),
        mean_va_count=('val_a_count', 'mean'),
        vb_pnl=('val_b_avg_pnl', 'mean'),
        mean_opt_pnl=('opt_avg_pnl', 'mean'),
        mean_vb_wr=('val_b_win_rate', 'mean'),
        mean_vb_count=('val_b_count', 'mean'),
        **extra_agg,
    ).reset_index()
    sa['agree_18'] = (sa['n_pos_seeds_18'] / sa['n_seeds']).clip(upper=1.0)

    sa_filt = sa[
        (sa['n_seeds'] >= MIN_SEEDS) &
        (sa['tpsl_mode'].isin(winning_modes))
    ].copy()

    sa_filt['score_pnl2'] = (sa_filt['mean_va_pnl'] ** 2) * sa_filt['agree_18']
    sa_filt = sa_filt[np.isfinite(sa_filt['score_pnl2'])]

    print(f"\n  {label}: {len(indiv)} rows → {len(sa)} seed groups → "
          f"{len(sa_filt)} after mode+seed filters")
    print(f"    Tickers: {sa_filt['ticker'].nunique()}, "
          f"Mean score_pnl2: {sa_filt['score_pnl2'].mean():.6f}")

    return sa_filt


sa_400 = compute_seed_groups(ps_400, "Exp 400 (α=2.0)")
sa_410 = compute_seed_groups(ps_410, "Exp 410 (α=1.0)")


def best_config_per_ticker(sa_df):
    """Pick best config per ticker by score_pnl2 (mirrors S3 best selection)."""
    if sa_df.empty:
        return pd.DataFrame()
    best = sa_df.loc[sa_df.groupby('ticker')['score_pnl2'].idxmax()].copy()
    return best


best_400_ps = best_config_per_ticker(sa_400)
best_410_ps = best_config_per_ticker(sa_410)


def best_ensemble_per_ticker(en_df, label):
    """Pick best ensemble config per ticker — ensemble agree_18 = 1.0 (consensus)."""
    if en_df.empty:
        print(f"\n  {label} ensemble: 0 rows")
        return pd.DataFrame()
    ens = en_df.copy()
    ens = ens[ens['tpsl_mode'].isin(winning_modes)].copy()
    ens = ens[ens['opt_avg_pnl'] > 0].copy()
    ens = ens[ens['val_a_avg_pnl'] > 0].copy()
    ens = ens[ens['val_a_avg_pnl'] / ens['opt_avg_pnl'] <= MAX_VA_OPT_RATIO].copy()
    if ens.empty:
        print(f"\n  {label} ensemble: 0 rows after filters")
        return pd.DataFrame()
    ens['agree_18'] = np.where(ens['val_a_avg_pnl'] >= DEPLOY_PNL_THRESHOLD, 1.0, 0.0)
    ens['n_seeds'] = ens['n_seeds_avg'].fillna(15).astype(int)
    ens['mean_va_pnl'] = ens['val_a_avg_pnl']
    ens['mean_va_wr'] = ens['val_a_win_rate']
    ens['mean_va_count'] = ens['val_a_count']
    ens['vb_pnl'] = ens['val_b_avg_pnl']
    ens['mean_opt_pnl'] = ens['opt_avg_pnl']
    ens['mean_vb_wr'] = ens['val_b_win_rate']
    ens['mean_vb_count'] = ens['val_b_count']
    ens['score_pnl2'] = (ens['mean_va_pnl'] ** 2) * ens['agree_18']
    ens['source'] = 'ensemble'
    best = ens.loc[ens.groupby('ticker')['score_pnl2'].idxmax()].copy()
    print(f"\n  {label} ensemble: {len(en_df)} → {len(ens)} after filters → "
          f"{len(best)} tickers")
    return best


en_best_400 = best_ensemble_per_ticker(en_400, "Exp 400 (α=2.0)")
en_best_410 = best_ensemble_per_ticker(en_410, "Exp 410 (α=1.0)")

if not best_400_ps.empty:
    best_400_ps['source'] = 'per_seed'
if not best_410_ps.empty:
    best_410_ps['source'] = 'per_seed'


def merge_ps_ensemble(ps_best, en_best, label):
    """Take whichever source (per-seed or ensemble) gives better TEST PnL per ticker."""
    if ps_best.empty and en_best.empty:
        return pd.DataFrame()
    if en_best.empty:
        return ps_best
    if ps_best.empty:
        return en_best

    shared_cols = [c for c in ['ticker', 'tpsl_mode', 'prob_thresh', 'tp_pct', 'sl_pct',
                               'mean_va_pnl', 'vb_pnl', 'agree_18', 'score_pnl2',
                               'mean_opt_pnl', 'mean_vb_wr', 'mean_vb_count',
                               'n_seeds', 'source',
                               'tp_mult', 'sl_mult', 'trail_start_mult', 'trail_end_mult',
                               'grace_minutes', 'dip_pct', 'max_wait_minutes',
                               'vol_filter_pct', 'entry_minute', 'experiment']
                  if c in ps_best.columns or c in en_best.columns]
    ps_sub = ps_best[[c for c in shared_cols if c in ps_best.columns]].copy()
    en_sub = en_best[[c for c in shared_cols if c in en_best.columns]].copy()

    combined = pd.concat([ps_sub, en_sub], ignore_index=True)
    best = combined.loc[combined.groupby('ticker')['vb_pnl'].idxmax()].copy()

    ps_tickers = set(ps_sub['ticker'])
    en_tickers = set(en_sub['ticker'])
    n_ps_wins = (best['source'] == 'per_seed').sum()
    n_en_wins = (best['source'] == 'ensemble').sum()
    n_en_only = len(set(best['ticker']) - ps_tickers)
    print(f"\n  {label} merge: per_seed wins {n_ps_wins}, ensemble wins {n_en_wins}, "
          f"ensemble-only {n_en_only}")
    return best


best_400 = merge_ps_ensemble(best_400_ps, en_best_400, "Exp 400")
best_410 = merge_ps_ensemble(best_410_ps, en_best_410, "Exp 410")

tickers_400 = set(best_400['ticker'].unique()) if not best_400.empty else set()
tickers_410 = set(best_410['ticker'].unique()) if not best_410.empty else set()
gate_overlap = sorted(tickers_400 & tickers_410)
gate_only_400 = sorted(tickers_400 - tickers_410)
gate_only_410 = sorted(tickers_410 - tickers_400)

print(f"\n  Alpha Agreement Gate:")
print(f"    Tickers with best config in 400: {len(tickers_400)}")
print(f"    Tickers with best config in 410: {len(tickers_410)}")
print(f"    Overlap (gate candidates):       {len(gate_overlap)}")

gate_pass = []
gate_fail_reasons = {}

for ticker in gate_overlap:
    r400 = best_400[best_400['ticker'] == ticker].iloc[0]
    r410 = best_410[best_410['ticker'] == ticker].iloc[0]

    vb_400 = r400['vb_pnl']
    vb_410 = r410['vb_pnl']

    if vb_400 > 0 and vb_410 > 0:
        selected_alpha = '400' if r400['score_pnl2'] >= r410['score_pnl2'] else '410'
        gate_pass.append({
            'ticker': ticker,
            'selected_alpha': selected_alpha,
            'vb_400': vb_400,
            'vb_410': vb_410,
            'score_400': r400['score_pnl2'],
            'score_410': r410['score_pnl2'],
            'agree_400': r400['agree_18'],
            'agree_410': r410['agree_18'],
            'mode_400': r400['tpsl_mode'],
            'mode_410': r410['tpsl_mode'],
            'va_pnl_400': r400['mean_va_pnl'],
            'va_pnl_410': r410['mean_va_pnl'],
            'opt_pnl_400': r400['mean_opt_pnl'],
            'opt_pnl_410': r410['mean_opt_pnl'],
            'vb_wr_400': r400.get('mean_vb_wr', np.nan),
            'vb_wr_410': r410.get('mean_vb_wr', np.nan),
            'vb_n_400': r400.get('mean_vb_count', 0),
            'vb_n_410': r410.get('mean_vb_count', 0),
            'prob_400': r400['prob_thresh'],
            'prob_410': r410['prob_thresh'],
            'tp_400': r400['tp_pct'],
            'tp_410': r410['tp_pct'],
            'sl_400': r400['sl_pct'],
            'sl_410': r410['sl_pct'],
        })
    else:
        reason = ('400-neg' if vb_400 <= 0 else '') + \
                 (' ' if vb_400 <= 0 and vb_410 <= 0 else '') + \
                 ('410-neg' if vb_410 <= 0 else '')
        gate_fail_reasons[ticker] = reason.strip()

for ticker in gate_only_400:
    gate_fail_reasons[ticker] = '410-missing'
for ticker in gate_only_410:
    gate_fail_reasons[ticker] = '400-missing'

print(f"\n  Gate Results:")
print(f"    PASS (both alphas positive OOS): {len(gate_pass)}")
print(f"    FAIL: {len(gate_fail_reasons)}")
if gate_fail_reasons:
    for reason in sorted(set(gate_fail_reasons.values())):
        tks = [t for t, r in gate_fail_reasons.items() if r == reason]
        print(f"      {reason}: {', '.join(tks[:15])}{'...' if len(tks) > 15 else ''}")

if not gate_pass:
    print(f"\n  ⚠ No tickers pass the alpha agreement gate yet.")
    print(f"    This is expected with partial results. Re-run when both optimizers "
          f"have more overlap.")

    if not best_400.empty or not best_410.empty:
        print(f"\n  ── For reference: top configs per alpha (no gate applied) ──")
        for label, best in [('400', best_400), ('410', best_410)]:
            if best.empty:
                continue
            top = best.nlargest(10, 'score_pnl2')
            print(f"\n  Top Exp {label}:")
            print(f"  {'Ticker':<8} {'Mode':<13} {'VAL PnL':>8} {'TEST PnL':>9} "
                  f"{'Agree18':>8} {'Score':>10}")
            print(f"  {'─'*8} {'─'*13} {'─'*8} {'─'*9} {'─'*8} {'─'*10}")
            for _, r in top.iterrows():
                print(f"  {r['ticker']:<8} {r['tpsl_mode']:<13} "
                      f"{r['mean_va_pnl']:>+7.2%} {r['vb_pnl']:>+8.2%} "
                      f"{r['agree_18']:>7.0%} {r['score_pnl2']:>10.6f}")

gate_df = pd.DataFrame(gate_pass)


# ==============================================================================
# STEP 3: CROSS-ALPHA SEED STABILITY
# ==============================================================================
print(f"\n\n{'═'*W}")
print(f"  STEP 3: CROSS-ALPHA SEED STABILITY")
print(f"{'═'*W}")

if not gate_df.empty:
    gate_df['cross_agree'] = gate_df[['agree_400', 'agree_410']].min(axis=1)
    gate_df['confidence'] = gate_df['cross_agree'].apply(
        lambda x: 'HIGH' if x >= CROSS_AGREE_HIGH else ('MED' if x >= CROSS_AGREE_MED else 'LOW'))

    print(f"\n  Cross-alpha seed stability = min(agree_18_400, agree_18_410)")
    print(f"  HIGH ≥ {CROSS_AGREE_HIGH:.0%}  |  MED ≥ {CROSS_AGREE_MED:.0%}  |  LOW < {CROSS_AGREE_MED:.0%}")
    print(f"\n  {'Ticker':<8} {'Agree400':>9} {'Agree410':>9} {'CrossAgree':>11} {'Confidence':>11}")
    print(f"  {'─'*8} {'─'*9} {'─'*9} {'─'*11} {'─'*11}")
    for _, r in gate_df.sort_values('cross_agree', ascending=False).iterrows():
        print(f"  {r['ticker']:<8} {r['agree_400']:>8.0%} {r['agree_410']:>8.0%} "
              f"{r['cross_agree']:>10.0%} {r['confidence']:>11}")

    conf_counts = gate_df['confidence'].value_counts()
    print(f"\n  Distribution: "
          f"HIGH={conf_counts.get('HIGH', 0)}  "
          f"MED={conf_counts.get('MED', 0)}  "
          f"LOW={conf_counts.get('LOW', 0)}")
else:
    print(f"\n  No tickers passed the alpha agreement gate — skipping.")


# ==============================================================================
# STEP 4: DEGRADATION FILTER
# ==============================================================================
print(f"\n\n{'═'*W}")
print(f"  STEP 4: DEGRADATION FILTER — OPT → TEST retention")
print(f"{'═'*W}")
print(f"\n  Reject if TEST/OPT retention < {MIN_RETENTION:.0%} (>{1-MIN_RETENTION:.0%} signal lost)")

if not gate_df.empty:
    gate_df['opt_pnl_sel'] = gate_df.apply(
        lambda r: r['opt_pnl_400'] if r['selected_alpha'] == '400' else r['opt_pnl_410'], axis=1)
    gate_df['vb_pnl_sel'] = gate_df.apply(
        lambda r: r['vb_400'] if r['selected_alpha'] == '400' else r['vb_410'], axis=1)
    gate_df['retention'] = np.where(
        gate_df['opt_pnl_sel'] > 0,
        gate_df['vb_pnl_sel'] / gate_df['opt_pnl_sel'],
        0.0)

    print(f"\n  {'Ticker':<8} {'Alpha':>6} {'OPT':>8} {'TEST':>9} {'Retention':>10} {'Status':>8}")
    print(f"  {'─'*8} {'─'*6} {'─'*8} {'─'*9} {'─'*10} {'─'*8}")
    for _, r in gate_df.sort_values('retention', ascending=False).iterrows():
        status = '✓' if r['retention'] >= MIN_RETENTION else 'REJECT'
        print(f"  {r['ticker']:<8} {r['selected_alpha']:>6} {r['opt_pnl_sel']:>+7.2%} "
              f"{r['vb_pnl_sel']:>+8.2%} {r['retention']:>9.0%} {status:>8}")

    n_before = len(gate_df)
    gate_df = gate_df[gate_df['retention'] >= MIN_RETENTION].copy()
    n_rejected = n_before - len(gate_df)
    print(f"\n  {n_rejected} ticker(s) rejected for excessive degradation.")
else:
    print(f"\n  No tickers to filter — skipping.")


# ==============================================================================
# STEP 5: FINAL DEPLOYMENT CANDIDATES
# ==============================================================================
print(f"\n\n{'═'*W}")
print(f"  STEP 5: FINAL DEPLOYMENT CANDIDATES")
print(f"{'═'*W}")

if not gate_df.empty:
    gate_df['score_sel'] = gate_df.apply(
        lambda r: r['score_400'] if r['selected_alpha'] == '400' else r['score_410'], axis=1)
    gate_df['mode_sel'] = gate_df.apply(
        lambda r: r['mode_400'] if r['selected_alpha'] == '400' else r['mode_410'], axis=1)
    gate_df['va_pnl_sel'] = gate_df.apply(
        lambda r: r['va_pnl_400'] if r['selected_alpha'] == '400' else r['va_pnl_410'], axis=1)
    gate_df['vb_wr_sel'] = gate_df.apply(
        lambda r: r['vb_wr_400'] if r['selected_alpha'] == '400' else r['vb_wr_410'], axis=1)
    gate_df['vb_n_sel'] = gate_df.apply(
        lambda r: r['vb_n_400'] if r['selected_alpha'] == '400' else r['vb_n_410'], axis=1)
    gate_df['prob_sel'] = gate_df.apply(
        lambda r: r['prob_400'] if r['selected_alpha'] == '400' else r['prob_410'], axis=1)
    gate_df['tp_sel'] = gate_df.apply(
        lambda r: r['tp_400'] if r['selected_alpha'] == '400' else r['tp_410'], axis=1)
    gate_df['sl_sel'] = gate_df.apply(
        lambda r: r['sl_400'] if r['selected_alpha'] == '400' else r['sl_410'], axis=1)

    deploy = gate_df[gate_df['va_pnl_sel'] >= DEPLOY_PNL_THRESHOLD].copy()
    deploy = deploy.sort_values('score_sel', ascending=False)

    if len(deploy) == 0:
        print(f"\n  ⚠ No tickers meet the deploy threshold ({DEPLOY_PNL_THRESHOLD:.1%}) "
              f"after all filters.")
        print(f"    {len(gate_df)} ticker(s) passed gate but below PnL threshold.")
    else:
        print(f"\n  {len(deploy)} deployment candidate(s) "
              f"(from {len(all_tickers)} total tickers, {len(overlap)} overlap):")

        print(f"\n  {'Ticker':<8} {'α':>3} {'Mode':<13} {'Prob':>5} {'TP':>6} {'SL':>6} │"
              f" {'VAL PnL':>8} {'TEST PnL':>9} {'TEST WR':>8} {'TEST N':>7} │"
              f" {'Agree18':>8} {'XAgree':>7} {'Retain':>7} {'Conf':>5} │"
              f" {'Score':>10}")
        print(f"  {'─'*8} {'─'*3} {'─'*13} {'─'*5} {'─'*6} {'─'*6} │"
              f" {'─'*8} {'─'*9} {'─'*8} {'─'*7} │"
              f" {'─'*8} {'─'*7} {'─'*7} {'─'*5} │"
              f" {'─'*10}")

        for _, r in deploy.iterrows():
            agree_sel = r['agree_400'] if r['selected_alpha'] == '400' else r['agree_410']
            print(f"  {r['ticker']:<8} {r['selected_alpha']:>3} {r['mode_sel']:<13} "
                  f"{r['prob_sel']:>4.2f} {r['tp_sel']:>5.1%} {r['sl_sel']:>5.1%} │"
                  f" {r['va_pnl_sel']:>+7.2%} {r['vb_pnl_sel']:>+8.2%} "
                  f"{r['vb_wr_sel']:>7.1%} {r['vb_n_sel']:>7.0f} │"
                  f" {agree_sel:>7.0%} {r['cross_agree']:>6.0%} "
                  f"{r['retention']:>6.0%} {r['confidence']:>5} │"
                  f" {r['score_sel']:>10.6f}")

        print(f"\n  ── Summary ──")
        print(f"  Candidates:      {len(deploy)}")
        print(f"  Mean TEST PnL:   {deploy['vb_pnl_sel'].mean():+.2%}")
        print(f"  Mean TEST WR:    {deploy['vb_wr_sel'].mean():.1%}")
        print(f"  Mean retention:  {deploy['retention'].mean():.0%}")
        a400_n = (deploy['selected_alpha'] == '400').sum()
        a410_n = (deploy['selected_alpha'] == '410').sum()
        print(f"  Alpha split:     400={a400_n}, 410={a410_n}")
        for conf in ['HIGH', 'MED', 'LOW']:
            n = (deploy['confidence'] == conf).sum()
            if n:
                print(f"  {conf} confidence:  {n} ticker(s)")
else:
    deploy = pd.DataFrame()
    print(f"\n  ⚠ No tickers available for deployment selection.")
    print(f"    Gate passed: 0 | Total tickers: {len(all_tickers)} | Overlap: {len(overlap)}")
    print(f"    Re-run after both optimizers process more overlapping tickers.")


# ── Watch-list: gate pass + positive TEST but below deploy threshold ──
print(f"\n  ── WATCH LIST (gate pass, TEST > 0, VAL < {DEPLOY_PNL_THRESHOLD:.1%}) ──")
if not gate_df.empty:
    watch_cols_needed = ['ticker', 'selected_alpha', 'va_pnl_sel', 'vb_pnl_sel',
                         'mode_sel', 'score_sel', 'confidence']
    missing_cols = [c for c in watch_cols_needed if c not in gate_df.columns]
    if missing_cols:
        for c in ['score_sel', 'mode_sel', 'va_pnl_sel', 'vb_pnl_sel', 'vb_wr_sel', 'prob_sel']:
            if c not in gate_df.columns:
                alpha_key = c.replace('_sel', '')
                gate_df[c] = gate_df.apply(
                    lambda r: r.get(f'{alpha_key}_400', np.nan)
                    if r['selected_alpha'] == '400'
                    else r.get(f'{alpha_key}_410', np.nan), axis=1)

    watch = gate_df[
        (gate_df['vb_pnl_sel'] > 0) &
        (gate_df['va_pnl_sel'] < DEPLOY_PNL_THRESHOLD)
    ].sort_values('vb_pnl_sel', ascending=False)

    if len(watch) > 0:
        print(f"  {len(watch)} near-deploy ticker(s) — positive TEST but below VAL threshold:")
        print(f"\n  {'Ticker':<8} {'α':>3} {'Mode':<13} {'VAL PnL':>8} {'TEST PnL':>9} "
              f"{'XAgree':>7} {'Conf':>5}")
        print(f"  {'─'*8} {'─'*3} {'─'*13} {'─'*8} {'─'*9} {'─'*7} {'─'*5}")
        for _, r in watch.iterrows():
            print(f"  {r['ticker']:<8} {r['selected_alpha']:>3} "
                  f"{r.get('mode_sel', '?'):<13} "
                  f"{r['va_pnl_sel']:>+7.2%} {r['vb_pnl_sel']:>+8.2%} "
                  f"{r.get('cross_agree', 0):>6.0%} {r.get('confidence', '?'):>5}")
    else:
        print(f"  None — no tickers with positive TEST below VAL threshold.")
else:
    print(f"  None — no tickers passed the alpha agreement gate.")


# ── Single-alpha standouts: strong in one alpha only ──
print(f"\n  ── SINGLE-ALPHA STANDOUTS (strong in one alpha, missing from other) ──")
standouts = []
for label, best_one, missing_set in [('400', best_400, tickers_410),
                                      ('410', best_410, tickers_400)]:
    if best_one.empty:
        continue
    solo = best_one[~best_one['ticker'].isin(missing_set)].copy()
    solo = solo[solo['vb_pnl'] > 0].copy()
    if not solo.empty:
        solo = solo.sort_values('vb_pnl', ascending=False)
        standouts.append((label, solo))

if standouts:
    for label, solo in standouts:
        top = solo.head(10)
        print(f"\n  Exp {label} only ({len(solo)} with positive TEST, showing top 10):")
        print(f"  {'Ticker':<8} {'Mode':<13} {'Source':<9} {'VAL PnL':>8} {'TEST PnL':>9} "
              f"{'Agree18':>8} {'Score':>10}")
        print(f"  {'─'*8} {'─'*13} {'─'*9} {'─'*8} {'─'*9} {'─'*8} {'─'*10}")
        for _, r in top.iterrows():
            print(f"  {r['ticker']:<8} {r.get('tpsl_mode', '?'):<13} "
                  f"{r.get('source', 'ps'):<9} "
                  f"{r['mean_va_pnl']:>+7.2%} {r['vb_pnl']:>+8.2%} "
                  f"{r.get('agree_18', 0):>7.0%} {r.get('score_pnl2', 0):>10.6f}")
else:
    print(f"  None — all tickers exist in both alphas (or no positive TEST).")


# ==============================================================================
# STEP 6: CHARTS
# ==============================================================================
print(f"\n\n{'═'*W}")
print(f"  STEP 6: CHARTS")
print(f"{'═'*W}")

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('SHORT Model Deployment Selection — Exp 400 vs 410', fontsize=14, fontweight='bold')

# Panel 1: Alpha agreement scatter
ax = axes[0, 0]
if not gate_df.empty or gate_fail_reasons:
    all_candidates = []
    for ticker in gate_overlap:
        if not best_400.empty and ticker in best_400['ticker'].values:
            r4 = best_400[best_400['ticker'] == ticker].iloc[0]
        else:
            continue
        if not best_410.empty and ticker in best_410['ticker'].values:
            r1 = best_410[best_410['ticker'] == ticker].iloc[0]
        else:
            continue
        all_candidates.append((ticker, r4['vb_pnl'], r1['vb_pnl']))

    if all_candidates:
        tks, x_vals, y_vals = zip(*all_candidates)
        x_vals, y_vals = np.array(x_vals), np.array(y_vals)

        colors = ['green' if (x > 0 and y > 0) else 'red' for x, y in zip(x_vals, y_vals)]
        ax.scatter(x_vals * 100, y_vals * 100, c=colors, s=50, alpha=0.7, edgecolors='k', linewidth=0.5)
        for t, x, y in zip(tks, x_vals * 100, y_vals * 100):
            ax.annotate(t, (x, y), fontsize=6, ha='center', va='bottom')

        lim = max(abs(x_vals).max(), abs(y_vals).max()) * 100 * 1.2
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.axvline(0, color='gray', linewidth=0.5)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

        ax.fill_between([-lim, 0], -lim, 0, alpha=0.05, color='red')
        ax.fill_between([0, lim], 0, lim, alpha=0.05, color='green')
        ax.text(lim * 0.5, lim * 0.8, 'BOTH +\n(PASS)', ha='center', fontsize=9, color='green')
        ax.text(-lim * 0.5, -lim * 0.8, 'BOTH -\n(FAIL)', ha='center', fontsize=9, color='red')

ax.set_xlabel('Exp 400 TEST PnL (%)')
ax.set_ylabel('Exp 410 TEST PnL (%)')
ax.set_title('Alpha Agreement: 400 vs 410 TEST PnL')

# Panel 2: Deploy candidates bar
ax = axes[0, 1]
if not deploy.empty:
    deploy_sorted = deploy.sort_values('vb_pnl_sel', ascending=True)
    colors = ['#1f77b4' if a == '400' else '#ff7f0e' for a in deploy_sorted['selected_alpha']]
    bars = ax.barh(range(len(deploy_sorted)), deploy_sorted['vb_pnl_sel'] * 100,
                   color=colors, edgecolor='k', linewidth=0.5)
    ax.set_yticks(range(len(deploy_sorted)))
    ax.set_yticklabels(deploy_sorted['ticker'], fontsize=8)
    ax.axvline(DEPLOY_PNL_THRESHOLD * 100, color='red', linestyle='--', linewidth=1,
               label=f'Deploy threshold ({DEPLOY_PNL_THRESHOLD:.1%})')
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.legend(fontsize=8)
    ax.set_xlabel('TEST PnL (%)')
    import matplotlib.patches as mpatches
    p400 = mpatches.Patch(color='#1f77b4', label='α=2.0 (400)')
    p410 = mpatches.Patch(color='#ff7f0e', label='α=1.0 (410)')
    ax.legend(handles=[p400, p410], fontsize=8)
else:
    ax.text(0.5, 0.5, 'No deploy candidates yet', ha='center', va='center',
            transform=ax.transAxes, fontsize=12, color='gray')
ax.set_title('Deploy Candidates — TEST PnL by Selected Alpha')

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

# Panel 4: Score distribution
ax = axes[1, 1]
if not sa_400.empty or not sa_410.empty:
    if not sa_400.empty:
        ax.hist(sa_400['score_pnl2'] * 10000, bins=30, alpha=0.5, color='#1f77b4',
                label='400 (α=2.0)', edgecolor='k', linewidth=0.3)
    if not sa_410.empty:
        ax.hist(sa_410['score_pnl2'] * 10000, bins=30, alpha=0.5, color='#ff7f0e',
                label='410 (α=1.0)', edgecolor='k', linewidth=0.3)
    ax.axvline(DEPLOY_PNL_THRESHOLD**2 * 10000, color='red', linestyle='--',
               label=f'Deploy² ({DEPLOY_PNL_THRESHOLD**2:.4f})', linewidth=1)
    ax.legend(fontsize=8)
    ax.set_xlabel('score_pnl2 (×10⁴)')
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
print(f"\n\n{'═'*W}")
print(f"  SUMMARY")
print(f"{'═'*W}")
print(f"\n  Data:        {len(all_tickers)} tickers scored ({len(t400)} in 400, "
      f"{len(t410)} in 410, {len(overlap)} overlap)")
print(f"  Win modes:   {', '.join(winning_modes) if winning_modes else 'TBD'}")
print(f"  Excluded:    {', '.join(sorted(EXCLUDE_MODES))}")
print(f"  Gate pass:   {len(gate_pass)} / {len(gate_overlap)} overlap tickers")
print(f"  Deploy:      {len(deploy)} candidate(s)")
n_watch = len(watch) if 'watch' in dir() and not watch.empty else 0
print(f"  Watch-list:  {n_watch} near-deploy ticker(s)")
if not deploy.empty:
    print(f"  Best ticker: {deploy.iloc[0]['ticker']} "
          f"(α={deploy.iloc[0]['selected_alpha']}, "
          f"TEST={deploy.iloc[0]['vb_pnl_sel']:+.2%}, "
          f"score={deploy.iloc[0]['score_sel']:.6f})")
print(f"\n  ★ Re-run this script as more optimization results arrive.")
print(f"{'═'*W}")
