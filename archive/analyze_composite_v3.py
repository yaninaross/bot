"""
analyze_composite_v3.py — Composite scoring across ALL experiment families
==========================================================================
Covers 152, 160, 170-175 WF, 180-185 WF, 200 CNN-TX.

Analyses:
  A. Experiment family coverage map (which tickers span multiple families?)
  B. Filtering cascade (progressive tightening → avg_vb impact)
  C. Config deep-dive for limit-only tickers (dip_pct, TP/SL, seeds)
  D. Dropped ticker analysis (recoverable via other modes?)
  E. Deploy threshold sweep for limit-only
  F. Two-tier deployment portfolio (limit vs stability-scored)
  G. ML with multi-family filter
  H. Dip_pct distribution across all limit survivors
  I. Per-ticker × exp_family selection (Strategy H scoring)

Colab-ready.
"""

from google.cloud import bigquery
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass

bq = bigquery.Client()
W = 140
DEPLOY_THRESHOLD = 0.018

TICKER_RENAME = {'MIGI': 'BGDE', 'BITF': 'KEEL'}
EXCLUDE_TICKERS = {'TLT', 'VIX', 'SMLR', 'SPIR'}

FAM_LABELS = {
    '152': '152 Late', '160': '160 Early',
    '170': '170-175 WF', '180': '180-185 WF', '200': '200 CNN-TX',
}


def assign_family(exp):
    base = exp.split('_')[0]
    if not base.isdigit():
        return 'other'
    num = int(base)
    if num == 152: return '152'
    if num == 160: return '160'
    if 170 <= num <= 175: return '170'
    if 180 <= num <= 185: return '180'
    if num == 200: return '200'
    return 'other'


# ==============================================================================
# 1. LOAD DATA
# ==============================================================================
q = '''
SELECT ticker, experiment, seed, tpsl_mode, entry_minute,
       prob_thresh, tp_pct, sl_pct,
       opt_avg_pnl, opt_win_rate, opt_count,
       val_a_avg_pnl, val_a_win_rate, val_a_count, val_a_freq,
       val_b_avg_pnl, val_b_win_rate, val_b_count, val_b_freq,
       survived_a, survived_b, is_ensemble,
       dip_pct, max_wait_minutes
FROM `ys-time-series.scoring_logs.wf_optimization_results`
WHERE model_version = 'robust_wf_v12'
  AND survived_a = TRUE AND val_a_count >= 3 AND val_a_avg_pnl > 0
  AND opt_avg_pnl > 0
  AND val_a_avg_pnl / opt_avg_pnl <= 3.0
  AND is_ensemble = FALSE
'''
df = bq.query(q).to_dataframe()
df['ticker'] = df['ticker'].replace(TICKER_RENAME)
df = df[~df['ticker'].isin(EXCLUDE_TICKERS)]
df['exp_fam'] = df['experiment'].apply(assign_family)
df = df[df['exp_fam'] != 'other']
print(f"Loaded {len(df)} rows, {df['ticker'].nunique()} tickers, "
      f"{df['tpsl_mode'].nunique()} modes, "
      f"families: {sorted(df['exp_fam'].unique())}")


# ==============================================================================
# 2. EXPERIMENT FAMILY COVERAGE MAP
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  EXPERIMENT FAMILY COVERAGE MAP — which tickers span multiple families?")
print(f"{'='*W}")

exp_map = df.groupby(['ticker', 'exp_fam']).agg(
    n_configs=('seed', 'count'),
    n_above_18=('val_a_avg_pnl', lambda x: (x >= 0.018).sum()),
    n_above_15=('val_a_avg_pnl', lambda x: (x >= 0.015).sum()),
    avg_va=('val_a_avg_pnl', 'mean'),
    avg_vb=('val_b_avg_pnl', 'mean'),
    n_limit=('tpsl_mode', lambda x: (x == 'limit_entry').sum()),
).reset_index()

all_tickers = sorted(df['ticker'].unique())
active_fams = sorted(df['exp_fam'].unique())

hdr = '  '.join(f'{"── " + FAM_LABELS.get(f, f)[:8] + " ──":>18}' for f in active_fams)
print(f"\n  {'Ticker':<8} {hdr}  {'Fams':>4} {'Lim':>4}")
sub = '  '.join(f"{'N':>4} {'>1.8':>5} {'vb':>7}" for _ in active_fams)
print(f"  {'':8} {sub}")
print(f"  {'-'*(W-4)}")

ticker_multi_fam = set()
ticker_both_exp = set()
ticker_has_limit = set()

for tk in all_tickers:
    parts = []
    tk_fams_with_data = set()
    for fam in active_fams:
        r = exp_map[(exp_map['ticker'] == tk) & (exp_map['exp_fam'] == fam)]
        if r.empty:
            parts.append(f"{'--':>4} {'--':>5} {'--':>7}")
        else:
            row = r.iloc[0]
            parts.append(f"{row['n_configs']:>4} {row['n_above_18']:>5} {row['avg_vb']:>+6.2%}")
            if row['n_above_18'] > 0:
                tk_fams_with_data.add(fam)

    n_fams = len(tk_fams_with_data)
    if n_fams >= 2:
        ticker_multi_fam.add(tk)
    if '152' in tk_fams_with_data and '160' in tk_fams_with_data:
        ticker_both_exp.add(tk)

    tk_lim = df[(df['ticker'] == tk) & (df['tpsl_mode'] == 'limit_entry')
                & (df['val_a_avg_pnl'] >= 0.018)]
    if len(tk_lim) > 0:
        ticker_has_limit.add(tk)

    line = '  '.join(parts)
    print(f"  {tk:<8} {line}  {n_fams:>4} {'Y' if tk in ticker_has_limit else '':>3}")

print(f"\n  Multi-family (>= 2 families, >= 1.8%): {len(ticker_multi_fam)} → {sorted(ticker_multi_fam)}")
print(f"  Both 152+160 (>= 1.8%): {len(ticker_both_exp)} → {sorted(ticker_both_exp)}")
print(f"  Has limit_entry:        {len(ticker_has_limit)} → {sorted(ticker_has_limit)}")
both_and_limit = ticker_both_exp & ticker_has_limit
multi_and_limit = ticker_multi_fam & ticker_has_limit
print(f"  Both 152+160 + limit:   {len(both_and_limit)} → {sorted(both_and_limit)}")
print(f"  Multi-fam + limit:      {len(multi_and_limit)} → {sorted(multi_and_limit)}")


# ==============================================================================
# 3. BUILD SEED GROUPS FOR SCORING
# ==============================================================================
GRP = ['ticker', 'experiment', 'tpsl_mode', 'entry_minute',
       'prob_thresh', 'tp_pct', 'sl_pct']

sa = df.groupby(GRP).agg(
    n_seeds  =('seed', 'nunique'),
    n_pos_18 =('val_a_avg_pnl', lambda x: (x >= 0.018).sum()),
    mean_va  =('val_a_avg_pnl', 'mean'),
    med_va   =('val_a_avg_pnl', 'median'),
    wr_va    =('val_a_win_rate', 'mean'),
    cnt_va   =('val_a_count', 'mean'),
    mean_opt =('opt_avg_pnl', 'mean'),
    wr_opt   =('opt_win_rate', 'mean'),
    cnt_opt  =('opt_count', 'mean'),
    vb_pnl   =('val_b_avg_pnl', 'mean'),
    vb_wr    =('val_b_win_rate', 'mean'),
).reset_index()

sa['ag_18'] = (sa['n_pos_18'] / sa['n_seeds']).clip(upper=1.0)
sa['is_lim'] = (sa['tpsl_mode'] == 'limit_entry').astype(int)
sa['exp_fam'] = sa['experiment'].str[:3]
sa['tot_va'] = sa['mean_va'] * sa['cnt_va']
rr = sa['tp_pct'] / sa['sl_pct'].clip(1e-4)
sa['pf'] = (sa['wr_va'] * rr / ((1 - sa['wr_va']).clip(0.01))).clip(upper=20)

# stability score function (used in cascade and tiers)
score_stab = lambda d: (np.sqrt(d['mean_opt'].clip(1e-4) * d['mean_va'].clip(1e-4))
                         * d['wr_va'] * d['ag_18'])
score_pnl2 = lambda d: (d['mean_va'] ** 2) * d['ag_18']

print(f"Seed groups: {len(sa)}, tickers: {sa['ticker'].nunique()}")


# ==============================================================================
# HELPERS
# ==============================================================================
def select_best(df_in, score_fn=None, thresh=DEPLOY_THRESHOLD, thresh_col='mean_va'):
    d = df_in.copy()
    if d.empty:
        return pd.DataFrame()
    if score_fn is None:
        score_fn = lambda d: d['mean_va'] * d['wr_va'] * d['ag_18']
    d['score'] = score_fn(d)
    d = d[np.isfinite(d['score'])]
    if d.empty:
        return pd.DataFrame()
    best = d.loc[d.groupby('ticker')['score'].idxmax()]
    return best[best[thresh_col] >= thresh]


def show_metrics(sel, label=''):
    if sel.empty:
        print(f"  {label}: N=0")
        return 0, 0, 0, 0
    n = len(sel)
    avg = sel['vb_pnl'].mean()
    tot = sel['vb_pnl'].sum()
    pos = (sel['vb_pnl'] > 0).mean()
    if label:
        print(f"  {label}: N={n}, avg_vb={avg:+.2%}, tot_vb={tot:+.2%}, %pos={pos:.0%}")
    return n, avg, tot, pos


# ==============================================================================
# 4. FILTERING CASCADE
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  FILTERING CASCADE — progressive tightening")
print(f"{'='*W}")

base = sa[sa['n_seeds'] >= 3]

steps = []

# S0: all modes
s = select_best(base)
steps.append(('All modes, all tickers', *show_metrics(s, '')))

# S1: limit-only
s = select_best(base[base['tpsl_mode'] == 'limit_entry'])
steps.append(('+ Limit-only', *show_metrics(s, '')))

# S2: + multi-family tickers
s = select_best(base[(base['tpsl_mode'] == 'limit_entry') &
                     (base['ticker'].isin(ticker_multi_fam))])
steps.append(('+ Multi-family (>=2)', *show_metrics(s, '')))

# S3: + both 152+160
s = select_best(base[(base['tpsl_mode'] == 'limit_entry') &
                     (base['ticker'].isin(ticker_both_exp))])
steps.append(('+ Both 152+160', *show_metrics(s, '')))

# S4: + min 5 seeds
s = select_best(sa[(sa['n_seeds'] >= 5) & (sa['tpsl_mode'] == 'limit_entry') &
                    (sa['ticker'].isin(ticker_multi_fam))])
steps.append(('+ Min 5 seeds', *show_metrics(s, '')))

# S5: + win rate >= 55%
s = select_best(sa[(sa['n_seeds'] >= 5) & (sa['tpsl_mode'] == 'limit_entry') &
                    (sa['ticker'].isin(ticker_multi_fam)) & (sa['wr_va'] >= 0.55)])
steps.append(('+ WR >= 55%', *show_metrics(s, '')))

# S6: + opt cross-val
s = select_best(sa[(sa['n_seeds'] >= 5) & (sa['tpsl_mode'] == 'limit_entry') &
                    (sa['ticker'].isin(ticker_multi_fam)) & (sa['wr_va'] >= 0.55) &
                    (sa['mean_opt'] >= 0.01)])
steps.append(('+ OPT >= 1.0%', *show_metrics(s, '')))

# S7: all modes + multi-fam
s = select_best(base[base['ticker'].isin(ticker_multi_fam)])
steps.append(('All modes + multi-family', *show_metrics(s, '')))

# S8: all modes + multi-fam + stability
s = select_best(base[base['ticker'].isin(ticker_multi_fam)], score_fn=score_stab)
steps.append(('All modes + multi-fam + stability', *show_metrics(s, '')))

# S9: all modes + stability (no exp filter)
s = select_best(base, score_fn=score_stab)
steps.append(('All modes + stability score', *show_metrics(s, '')))

print(f"\n  {'Step':<40} {'N':>3} {'Avg_vb':>8} {'Tot_vb':>9} {'%Pos':>6}")
print(f"  {'-'*70}")
for name, n, avg, tot, pos in steps:
    if n > 0:
        print(f"  {name:<40} {n:>3} {avg:>+7.2%} {tot:>+8.2%} {pos:>5.0%}")
    else:
        print(f"  {name:<40}   0")


# ==============================================================================
# 5. CONFIG DEEP-DIVE — LIMIT-ONLY TICKERS
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  CONFIG DEEP-DIVE — limit_entry configs for top tickers (val_a >= 1.8%)")
print(f"{'='*W}")

lim_surv = df[(df['tpsl_mode'] == 'limit_entry') & (df['val_a_avg_pnl'] >= 0.018)].copy()
lim_sel = select_best(base[base['tpsl_mode'] == 'limit_entry'])
top_tickers = sorted(lim_sel['ticker']) if not lim_sel.empty else []

for tk in top_tickers:
    tk_data = lim_surv[lim_surv['ticker'] == tk]
    if tk_data.empty:
        continue

    print(f"\n  {tk} — {len(tk_data)} surviving configs, "
          f"exps: {', '.join(sorted(tk_data['exp_fam'].unique()))}")

    # dip_pct breakdown
    dip = tk_data.groupby('dip_pct').agg(
        n=('seed', 'count'),
        avg_va=('val_a_avg_pnl', 'mean'),
        avg_vb=('val_b_avg_pnl', 'mean'),
        wr=('val_a_win_rate', 'mean'),
    ).reset_index().sort_values('dip_pct')

    print(f"    {'dip%':>6} {'N':>4} {'va':>8} {'vb':>8} {'wr':>6}")
    print(f"    {'-'*36}")
    for _, r in dip.iterrows():
        vb_flag = " **" if r['avg_vb'] > 0.02 else ""
        print(f"    {r['dip_pct']:>5.1%} {r['n']:>4} {r['avg_va']:>+7.2%} "
              f"{r['avg_vb']:>+7.2%} {r['wr']:>5.0%}{vb_flag}")

    # TP/SL breakdown
    tpsl = tk_data.groupby(['tp_pct', 'sl_pct']).agg(
        n=('seed', 'count'),
        avg_vb=('val_b_avg_pnl', 'mean'),
    ).reset_index().sort_values('avg_vb', ascending=False)

    print(f"\n    {'TP':>5} {'SL':>5} {'N':>4} {'vb':>8}")
    print(f"    {'-'*26}")
    for _, r in tpsl.iterrows():
        print(f"    {r['tp_pct']:>4.0%} {r['sl_pct']:>4.0%} {r['n']:>4} {r['avg_vb']:>+7.2%}")

    # best single config (by val_b)
    best_cfg = tk_data.loc[tk_data.groupby(
        ['experiment', 'prob_thresh', 'tp_pct', 'sl_pct', 'dip_pct', 'max_wait_minutes']
    )['val_b_avg_pnl'].transform('mean').idxmax()]
    print(f"    Best: exp={best_cfg['experiment']}, dip={best_cfg['dip_pct']:.1%}, "
          f"tp={best_cfg['tp_pct']:.0%}, sl={best_cfg['sl_pct']:.0%}, "
          f"wait={best_cfg['max_wait_minutes']:.0f}m → "
          f"va={best_cfg['val_a_avg_pnl']:+.2%}, vb={best_cfg['val_b_avg_pnl']:+.2%}")


# ==============================================================================
# 6. DROPPED TICKER ANALYSIS
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  DROPPED TICKERS — in all-mode baseline but NOT in limit-only")
print(f"{'='*W}")

baseline_sel = select_best(base)
baseline_tickers = set(baseline_sel['ticker']) if not baseline_sel.empty else set()
limit_sel_tickers = set(lim_sel['ticker']) if not lim_sel.empty else set()
dropped = sorted(baseline_tickers - limit_sel_tickers)

print(f"\n  Baseline: {sorted(baseline_tickers)}")
print(f"  Limit:    {sorted(limit_sel_tickers)}")
print(f"  Dropped:  {dropped}")

for tk in dropped:
    tk_all = df[df['ticker'] == tk]
    mode_perf = tk_all.groupby('tpsl_mode').agg(
        n=('seed', 'count'),
        avg_va=('val_a_avg_pnl', 'mean'),
        avg_vb=('val_b_avg_pnl', 'mean'),
        wr=('val_a_win_rate', 'mean'),
        best_va=('val_a_avg_pnl', 'max'),
    ).reset_index().sort_values('avg_vb', ascending=False)

    print(f"\n  {tk} — both exp: {'YES' if tk in ticker_both_exp else 'NO'}, "
          f"has limit: {'YES' if tk in ticker_has_limit else 'NO'}")
    print(f"    {'Mode':<18} {'N':>4} {'va':>8} {'vb':>8} {'wr':>6} {'best_va':>8}")
    print(f"    {'-'*56}")
    for _, r in mode_perf.head(5).iterrows():
        print(f"    {r['tpsl_mode']:<18} {r['n']:>4} {r['avg_va']:>+7.2%} "
              f"{r['avg_vb']:>+7.2%} {r['wr']:>5.0%} {r['best_va']:>+7.2%}")

    # verdict: worth adding back?
    best_vb = mode_perf.iloc[0]['avg_vb']
    verdict = "KEEP (vb > 1%)" if best_vb > 0.01 else "DROP (vb weak)"
    print(f"    Verdict: {verdict}")


# ==============================================================================
# 7. DEPLOY THRESHOLD SWEEP — LIMIT-ONLY
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  DEPLOY THRESHOLD SWEEP — limit-only subset")
print(f"{'='*W}")

thresholds = [0.008, 0.010, 0.012, 0.014, 0.015, 0.016, 0.018,
              0.020, 0.022, 0.025, 0.028, 0.030]
limit_sa = base[base['tpsl_mode'] == 'limit_entry'].copy()

print(f"\n  {'Thresh':>8} {'N':>3} {'Avg_vb':>8} {'Med_vb':>8} {'Tot_vb':>9} "
      f"{'%Pos':>6} {'Eff':>8}  {'Tickers'}")
print(f"  {'-'*(W-8)}")

best_eff = 0
best_thresh_eff = 0
for t in thresholds:
    sel = select_best(limit_sa, thresh=t)
    if sel.empty:
        print(f"  {t:>7.1%}   0")
        continue
    n = len(sel)
    avg = sel['vb_pnl'].mean()
    med = sel['vb_pnl'].median()
    tot = sel['vb_pnl'].sum()
    pos = (sel['vb_pnl'] > 0).mean()
    eff = avg * np.sqrt(n)
    if eff > best_eff:
        best_eff = eff
        best_thresh_eff = t
    tks = ', '.join(sorted(sel['ticker']))
    print(f"  {t:>7.1%} {n:>3} {avg:>+7.2%} {med:>+7.2%} {tot:>+8.2%} "
          f"{pos:>5.0%} {eff:>7.4f}  {tks}")

print(f"\n  Best efficiency (avg_vb * sqrt(N)): thresh={best_thresh_eff:.1%}, eff={best_eff:.4f}")


# ==============================================================================
# 8. TWO-TIER DEPLOYMENT PORTFOLIO
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TWO-TIER DEPLOYMENT PORTFOLIO")
print(f"{'='*W}")

# Tier 1+: limit + multi-family tickers
t1plus_data = limit_sa[limit_sa['ticker'].isin(ticker_multi_fam)]
t1plus = select_best(t1plus_data, score_fn=score_pnl2)
t1plus_tks = set(t1plus['ticker']) if not t1plus.empty else set()

# Tier 1: limit-only (all)
t1 = select_best(limit_sa, score_fn=score_pnl2)
t1_tks = set(t1['ticker']) if not t1.empty else set()

# Tier 2: all-mode, stability-scored, excluding Tier 1
t2_data = base[~base['ticker'].isin(t1_tks)]
t2 = select_best(t2_data, score_fn=score_stab)
t2_tks = set(t2['ticker']) if not t2.empty else set()

print(f"\n  TIER 1+ (limit + multi-family) — highest conviction:")
if not t1plus.empty:
    for _, r in t1plus.sort_values('vb_pnl', ascending=False).iterrows():
        print(f"    {r['ticker']:<8} va={r['mean_va']:>+6.2%}  vb={r['vb_pnl']:>+6.2%}")
    show_metrics(t1plus, 'Tier 1+')

print(f"\n  TIER 1 (limit-only) — high conviction:")
if not t1.empty:
    for _, r in t1.sort_values('vb_pnl', ascending=False).iterrows():
        tag = " *1+" if r['ticker'] in t1plus_tks else ""
        print(f"    {r['ticker']:<8} va={r['mean_va']:>+6.2%}  vb={r['vb_pnl']:>+6.2%}{tag}")
    show_metrics(t1, 'Tier 1')

print(f"\n  TIER 2 (all-mode, stability-scored, non-overlap) — moderate conviction:")
if not t2.empty:
    for _, r in t2.sort_values('vb_pnl', ascending=False).iterrows():
        print(f"    {r['ticker']:<8} va={r['mean_va']:>+6.2%}  vb={r['vb_pnl']:>+6.2%}  "
              f"mode={r['tpsl_mode']}")
    show_metrics(t2, 'Tier 2')

print(f"\n  COMBINED (Tier 1 + Tier 2):")
combined = pd.concat([t1, t2])
if not combined.empty:
    show_metrics(combined, 'Combined')


# ==============================================================================
# 9. ML WITH HARD EXP_DIV FILTER
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  ML WITH HARD EXP_DIV FILTER — predict val_b for both-experiment tickers")
print(f"{'='*W}")

try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
    from sklearn.model_selection import GroupKFold, cross_val_predict

    sa_both = sa[sa['ticker'].isin(ticker_multi_fam)].copy()

    FEATS = ['mean_va', 'med_va', 'wr_va', 'cnt_va', 'mean_opt', 'wr_opt', 'cnt_opt',
             'n_seeds', 'ag_18', 'tot_va', 'pf', 'is_lim', 'tp_pct', 'sl_pct']

    ml = sa_both.dropna(subset=['vb_pnl']).copy()
    n_groups = ml['ticker'].nunique()

    if len(ml) > 30 and n_groups >= 3:
        X = ml[FEATS].fillna(0).values
        y = ml['vb_pnl'].values
        groups = ml['ticker'].values

        gkf = GroupKFold(n_splits=min(5, n_groups))
        print(f"\n  Data: {len(ml)} rows, {n_groups} multi-fam tickers, {len(FEATS)} features")

        # Regression: predict val_b PnL
        gbt = GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                         learning_rate=0.05, random_state=42)
        preds = cross_val_predict(gbt, X, y, cv=gkf, groups=groups)
        corr = np.corrcoef(preds, y)[0, 1]
        print(f"  GBT regression: corr={corr:.3f}")

        sa_both['ml_pred'] = np.nan
        sa_both.loc[ml.index, 'ml_pred'] = preds

        ml_sel = select_best(sa_both.dropna(subset=['ml_pred']),
                              score_fn=lambda d: d['ml_pred'])
        show_metrics(ml_sel, 'ML-selected (multi-fam)')
        if not ml_sel.empty:
            print(f"  Tickers: {sorted(ml_sel['ticker'].tolist())}")

        # Classification: predict val_b > 0
        y_cls = (y > 0).astype(int)
        if y_cls.sum() > 5 and (1 - y_cls).sum() > 5:
            rfc = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
            try:
                proba = cross_val_predict(rfc, X, y_cls, cv=gkf, groups=groups,
                                           method='predict_proba')[:, 1]
                cls_sel = select_best(sa_both.assign(ml_prob=np.nan).pipe(
                    lambda d: d.__setitem__('ml_prob', np.nan) or d),
                    score_fn=lambda d: d['ml_pred'])
                auc = np.corrcoef(proba, y_cls)[0, 1]
                print(f"  RF classifier: corr(prob, label)={auc:.3f}")
            except Exception:
                pass

        # Feature importance
        gbt.fit(X, y)
        imp = sorted(zip(FEATS, gbt.feature_importances_), key=lambda x: -x[1])
        print(f"\n  Feature importance (multi-fam filtered):")
        for f, i in imp[:10]:
            print(f"    {f:<20} {i:.3f}")
    else:
        print(f"  Not enough data for ML ({len(ml)} rows, {n_groups} tickers)")

except ImportError:
    print("  sklearn not available")


# ==============================================================================
# 10. DIP_PCT DISTRIBUTION — ALL LIMIT SURVIVORS
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  DIP_PCT DISTRIBUTION — surviving limit_entry configs (val_a >= 1.8%)")
print(f"{'='*W}")

dip_sum = lim_surv.groupby('dip_pct').agg(
    n=('seed', 'count'),
    n_tickers=('ticker', 'nunique'),
    avg_va=('val_a_avg_pnl', 'mean'),
    avg_vb=('val_b_avg_pnl', 'mean'),
    wr=('val_a_win_rate', 'mean'),
    tickers=('ticker', lambda x: ', '.join(sorted(x.unique()))),
).reset_index().sort_values('dip_pct')

print(f"\n  {'dip%':>6} {'N':>5} {'Tkrs':>5} {'avg_va':>8} {'avg_vb':>8} {'wr':>6}  {'Tickers'}")
print(f"  {'-'*(W-8)}")
for _, r in dip_sum.iterrows():
    vb_flag = " **" if r['avg_vb'] > 0.02 else ""
    print(f"  {r['dip_pct']:>5.1%} {r['n']:>5} {r['n_tickers']:>5} "
          f"{r['avg_va']:>+7.2%} {r['avg_vb']:>+7.2%} {r['wr']:>5.0%}  "
          f"{r['tickers'][:55]}{vb_flag}")

# Wait time distribution
wait_sum = lim_surv.groupby('max_wait_minutes').agg(
    n=('seed', 'count'),
    n_tickers=('ticker', 'nunique'),
    avg_vb=('val_b_avg_pnl', 'mean'),
).reset_index().sort_values('max_wait_minutes')

print(f"\n  Max wait distribution:")
print(f"  {'wait':>6} {'N':>5} {'Tkrs':>5} {'avg_vb':>8}")
print(f"  {'-'*30}")
for _, r in wait_sum.iterrows():
    print(f"  {r['max_wait_minutes']:>5.0f}m {r['n']:>5} {r['n_tickers']:>5} {r['avg_vb']:>+7.2%}")


# ==============================================================================
# 11. SEED AGREEMENT WITHIN LIMIT-ONLY — DO WE NEED IT?
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  SEED AGREEMENT — does it still matter within limit-only?")
print(f"{'='*W}")

limit_base = base[base['tpsl_mode'] == 'limit_entry'].copy()

print(f"\n  Testing min_seeds filter on limit-only subset:")
print(f"  {'min_s':>6} {'N':>3} {'Avg_vb':>8} {'Tot_vb':>9}")
print(f"  {'-'*30}")

for ms in [1, 2, 3, 5, 7, 10]:
    s = select_best(sa[(sa['n_seeds'] >= ms) & (sa['tpsl_mode'] == 'limit_entry')])
    if s.empty:
        print(f"  {ms:>6}   0")
        continue
    n = len(s)
    avg = s['vb_pnl'].mean()
    tot = s['vb_pnl'].sum()
    print(f"  {ms:>6} {n:>3} {avg:>+7.2%} {tot:>+8.2%}")

print(f"\n  Testing agree threshold on limit-only:")
print(f"  {'ag_18':>6} {'N':>3} {'Avg_vb':>8} {'Tot_vb':>9}")
print(f"  {'-'*30}")

for ag in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    d = limit_base[limit_base['ag_18'] >= ag]
    s = select_best(d)
    if s.empty:
        print(f"  {ag:>5.0%}   0")
        continue
    print(f"  {ag:>5.0%} {len(s):>3} {s['vb_pnl'].mean():>+7.2%} {s['vb_pnl'].sum():>+8.2%}")


# ==============================================================================
# 12. PER-TICKER × EXP_FAMILY SELECTION (Strategy H scoring)
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  STRATEGY H SELECTION — best config per (ticker × exp_family), limit-only")
print(f"  Score = mean_va_pnl² × agree_18")
print(f"{'='*W}")

lim_by_exp = limit_base.copy()
if not lim_by_exp.empty:
    lim_by_exp['score'] = (lim_by_exp['mean_va'] ** 2) * lim_by_exp['ag_18']
    best_by_exp = lim_by_exp.loc[lim_by_exp.groupby(['ticker', 'exp_fam'])['score'].idxmax()]
    best_by_exp = best_by_exp[best_by_exp['mean_va'] >= DEPLOY_THRESHOLD]

    print(f"\n  {'Ticker':<8} {'Family':>10} {'va':>8} {'vb':>8} {'wr':>6} {'ag':>5} {'score':>9}")
    print(f"  {'-'*60}")
    for _, r in best_by_exp.sort_values(['ticker', 'exp_fam']).iterrows():
        fam_label = FAM_LABELS.get(r['exp_fam'], r['exp_fam'])[:10]
        print(f"  {r['ticker']:<8} {fam_label:>10} {r['mean_va']:>+7.2%} "
              f"{r['vb_pnl']:>+7.2%} {r['wr_va']:>5.0%} {r['ag_18']:>4.0%} {r['score']:>9.6f}")

    print(f"\n  Family agreement on val_b direction:")
    for tk in sorted(best_by_exp['ticker'].unique()):
        tk_exp = best_by_exp[best_by_exp['ticker'] == tk]
        if len(tk_exp) >= 2:
            vals = tk_exp[['exp_fam', 'vb_pnl']].set_index('exp_fam')['vb_pnl']
            pos = (vals > 0).sum()
            neg = (vals <= 0).sum()
            tag = "ALL POS" if neg == 0 else ("ALL NEG" if pos == 0 else f"{pos}/{len(vals)} pos")
            detail = '  '.join(f"{f}={v:+.2%}" for f, v in vals.items())
            print(f"    {tk:<8} {detail}  → {tag}")


# ==============================================================================
# 13. FINAL SUMMARY + RECOMMENDATION
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  FINAL SUMMARY + RECOMMENDATION")
print(f"{'='*W}")

n_fams_seen = len(active_fams)
limit_deploy_tks = sorted(lim_sel['ticker']) if not lim_sel.empty else []
n_limit_deploy = len(limit_deploy_tks)
baseline_deploy_tks = sorted(baseline_sel['ticker']) if not baseline_sel.empty else []
n_baseline_deploy = len(baseline_deploy_tks)

print(f"""
  DYNAMIC SUMMARY:
  ────────────────
  Families analyzed: {n_fams_seen} ({', '.join(FAM_LABELS.get(f, f) for f in active_fams)})
  Total tickers: {len(all_tickers)}
  Multi-family tickers (>= 2 fams): {len(ticker_multi_fam)} → {sorted(ticker_multi_fam)}
  Limit-only deployable: {n_limit_deploy} → {limit_deploy_tks}
  All-mode deployable:   {n_baseline_deploy} → {baseline_deploy_tks}

  SCORING PIPELINE (Strategy H):
  ──────────────────────────────
  Step 1: Filter to tpsl_mode = 'limit_entry'
  Step 2: Require n_seeds >= 3, val_a >= 1.8%
  Step 3: Score by mean_va^2 × agree_18
  Step 4: Select best config per (ticker × exp_family)
  Step 5: Tag tier:
          Tier 1+: multi-family (>= 2 families) → {sorted(multi_and_limit)}
          Tier 1:  limit-only, single family
          Tier 2:  non-limit modes, stability-scored
""")
