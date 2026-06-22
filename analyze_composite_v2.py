"""
analyze_composite_v2.py — Extended composite score exploration
=============================================================
Builds on seed_agreement_sweep with novel scoring dimensions:
  - Power-law exponent sweep (pnl^a * wr^b * agree^g)
  - Total PnL integration (count * avg_pnl)
  - Risk-adjusted (Sharpe-like, profit-factor proxy, Kelly)
  - Nonlinear transforms (sigmoid, log, sqrt, tanh, softplus)
  - Stability (opt->val decay, geometric mean, conservative)
  - Bayesian shrinkage (small-N -> population prior)
  - Rank / z-score / percentile normalization
  - Mode & experiment diversity bonus
  - ML-based scoring (Ridge, RF, GBT predict val_b via GroupKFold)
  - Oracle ceiling (perfect val_b knowledge)
  - Pareto frontier (avg_vb vs n_tickers at varying deploy thresh)

~150 scoring variants + 4 ML models evaluated on val_b (out-of-sample).
Colab-ready.
"""

from google.cloud import bigquery
import numpy as np
import pandas as pd
from itertools import product
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


# ==============================================================================
# 1. LOAD DATA
# ==============================================================================
q = '''
SELECT ticker, experiment, seed, tpsl_mode, entry_minute,
       prob_thresh, tp_pct, sl_pct, tp_mult, sl_mult,
       opt_avg_pnl, opt_win_rate, opt_count,
       val_a_avg_pnl, val_a_win_rate, val_a_count, val_a_freq,
       val_b_avg_pnl, val_b_win_rate, val_b_count, val_b_freq,
       survived_a, survived_b, is_ensemble,
       grace_minutes, dip_pct, max_wait_minutes,
       vol_filter_pct, vix_threshold, tlt_crash, spy_ignore
FROM `ys-time-series.scoring_logs.wf_optimization_results`
WHERE model_version = 'robust_wf_v12'
  AND survived_a = TRUE AND val_a_count >= 3 AND val_a_avg_pnl > 0
  AND opt_avg_pnl > 0
  AND val_a_avg_pnl / opt_avg_pnl <= 3.0
  AND (experiment LIKE '152%' OR experiment LIKE '160%')
'''
df = bq.query(q).to_dataframe()
print(f"Loaded {len(df)} rows, {df['ticker'].nunique()} tickers, "
      f"{df['experiment'].nunique()} exps, {df['tpsl_mode'].nunique()} modes")


# ==============================================================================
# 2. SEED GROUPS + ENHANCED FEATURES
# ==============================================================================
GRP = ['ticker', 'experiment', 'tpsl_mode', 'entry_minute',
       'prob_thresh', 'tp_pct', 'sl_pct']

df_ind = df[df['is_ensemble'] == False].copy()

g = df_ind.groupby(GRP)
sa = g.agg(
    n_seeds   =('seed', 'nunique'),
    n_pos_18  =('val_a_avg_pnl', lambda x: (x >= 0.018).sum()),
    n_pos_15  =('val_a_avg_pnl', lambda x: (x >= 0.015).sum()),
    n_pos_12  =('val_a_avg_pnl', lambda x: (x >= 0.012).sum()),
    n_pos_10  =('val_a_avg_pnl', lambda x: (x >= 0.010).sum()),
    mean_va   =('val_a_avg_pnl', 'mean'),
    med_va    =('val_a_avg_pnl', 'median'),
    std_va    =('val_a_avg_pnl', 'std'),
    min_va    =('val_a_avg_pnl', 'min'),
    max_va    =('val_a_avg_pnl', 'max'),
    wr_va     =('val_a_win_rate', 'mean'),
    std_wr_va =('val_a_win_rate', 'std'),
    cnt_va    =('val_a_count', 'mean'),
    freq_va   =('val_a_freq', 'mean'),
    mean_opt  =('opt_avg_pnl', 'mean'),
    wr_opt    =('opt_win_rate', 'mean'),
    cnt_opt   =('opt_count', 'mean'),
).reset_index()

vb = df_ind.groupby(GRP).agg(
    vb_pnl=('val_b_avg_pnl', 'mean'),
    vb_wr =('val_b_win_rate', 'mean'),
    vb_cnt=('val_b_count', 'mean'),
).reset_index()
sa = sa.merge(vb, on=GRP, how='left')

# ---- derived features ----

for t in ['10', '12', '15', '18']:
    sa[f'ag_{t}'] = (sa[f'n_pos_{t}'] / sa['n_seeds']).clip(upper=1.0)

sa['cv']   = (sa['std_va'] / sa['mean_va'].clip(lower=1e-4)).fillna(1)
sa['cons'] = (1 - sa['cv']).clip(lower=0.01)

sa['tot_va']  = sa['mean_va'] * sa['cnt_va']
sa['tot_opt'] = sa['mean_opt'] * sa['cnt_opt']

sa['stab']    = (sa['mean_va'] / sa['mean_opt'].clip(lower=1e-4)).clip(0.01, 5)
sa['sharpe']  = (sa['mean_va'] / sa['std_va'].clip(lower=1e-4)).fillna(0)

rr = sa['tp_pct'] / sa['sl_pct'].clip(lower=1e-4)
sa['pf']    = (sa['wr_va'] * rr / ((1 - sa['wr_va']).clip(lower=0.01))).clip(upper=20)
sa['kelly'] = ((sa['wr_va'] * rr - (1 - sa['wr_va'])) / rr).clip(lower=0)

pop_mu = sa['mean_va'].mean()
sa['bayes5']  = (sa['n_seeds'] * sa['mean_va'] + 5 * pop_mu)  / (sa['n_seeds'] + 5)
sa['bayes10'] = (sa['n_seeds'] * sa['mean_va'] + 10 * pop_mu) / (sa['n_seeds'] + 10)

sa['min_per'] = np.minimum(sa['mean_opt'], sa['mean_va'])
sa['geo_per'] = np.sqrt(sa['mean_opt'].clip(1e-4) * sa['mean_va'].clip(1e-4))

sa['range_va'] = sa['max_va'] - sa['min_va']
sa['sortino']  = sa['mean_va'] / (sa['range_va'].clip(1e-4) / 2)
sa['wr_stab']  = 1 - (sa['std_wr_va'].fillna(0) / sa['wr_va'].clip(0.01)).clip(upper=1)

sa['is_lim'] = (sa['tpsl_mode'] == 'limit_entry').astype(int)

# mode diversity per ticker
mode_div = (df_ind[df_ind['val_a_avg_pnl'] >= 0.015]
            .groupby('ticker')['tpsl_mode'].nunique().rename('mode_div'))
sa = sa.merge(mode_div, left_on='ticker', right_index=True, how='left')
sa['mode_div'] = sa['mode_div'].fillna(1)

# experiment diversity per ticker
exp_div = (df_ind[df_ind['val_a_avg_pnl'] >= 0.015]
           .groupby('ticker')['experiment'].nunique().rename('exp_div'))
sa = sa.merge(exp_div, left_on='ticker', right_index=True, how='left')
sa['exp_div'] = sa['exp_div'].fillna(1)

print(f"Seed groups: {len(sa)}, tickers: {sa['ticker'].nunique()}, pop mu = {pop_mu:.4f}")


# ==============================================================================
# 3. SCORING ENGINE
# ==============================================================================
def run_score(sa_df, fn, thresh=DEPLOY_THRESHOLD, thresh_col='mean_va',
              lim_only=False, min_s=3):
    d = sa_df.copy()
    if lim_only:
        d = d[d['tpsl_mode'] == 'limit_entry']
    d = d[d['n_seeds'] >= min_s]
    if d.empty:
        return pd.DataFrame()
    d['score'] = fn(d)
    d = d[np.isfinite(d['score'])]
    if d.empty:
        return pd.DataFrame()
    best = d.loc[d.groupby('ticker')['score'].idxmax()]
    return best[best[thresh_col] >= thresh]


def metrics(sel):
    if sel.empty:
        return dict(n=0, avg_vb=0, med_vb=0, pos=0, tot_vb=0, avg_va=0, nlim=0)
    n = len(sel)
    return dict(
        n=n,
        avg_vb=sel['vb_pnl'].mean(),
        med_vb=sel['vb_pnl'].median(),
        pos=(sel['vb_pnl'] > 0).mean(),
        tot_vb=sel['vb_pnl'].sum(),
        avg_va=sel['mean_va'].mean(),
        nlim=int(sel['is_lim'].sum()),
    )


# ==============================================================================
# 4. VARIANT DEFINITIONS
# ==============================================================================
V = {}
IS_LIM_ONLY = set()

# ---- A. BASELINES (from sweep) ----
V['A01_pnl_wr_ag']       = lambda d: d['mean_va'] * d['wr_va'] * d['ag_18']
V['A02_pnl_ag']           = lambda d: d['mean_va'] * d['ag_18']
V['A03_pnl_wr_ag_logN']   = lambda d: d['mean_va'] * d['wr_va'] * d['ag_18'] * np.log2(1 + d['cnt_va'])
V['A04_pnl2_ag']           = lambda d: (d['mean_va'] ** 2) * d['ag_18']
V['A05_med_wr_ag']         = lambda d: d['med_va'] * d['wr_va'] * d['ag_18']
V['A06_pnl_wr_ag_cons']    = lambda d: d['mean_va'] * d['wr_va'] * d['ag_18'] * d['cons']
V['A07_pnl_wr_ag_opt']     = lambda d: d['mean_va'] * d['wr_va'] * d['ag_18'] * d['mean_opt']

# ---- B. POWER-LAW SWEEP ----
for a, b, g in product([0.5, 1.0, 1.5, 2.0, 3.0],
                        [0, 0.5, 1.0, 1.5, 2.0],
                        [0.5, 1.0, 1.5, 2.0]):
    if (a, b, g) == (1.0, 1.0, 1.0):
        continue
    V[f'B_a{a}_b{b}_g{g}'] = (
        lambda a_=a, b_=b, g_=g: lambda d:
            (d['mean_va'] ** a_) * (d['wr_va'] ** b_) * (d['ag_18'] ** g_)
    )()

# ---- C. TOTAL PNL ----
V['C01_tot_wr_ag']         = lambda d: d['tot_va'] * d['wr_va'] * d['ag_18']
V['C02_tot_ag']             = lambda d: d['tot_va'] * d['ag_18']
V['C03_tot2_ag']             = lambda d: (d['tot_va'] ** 2) * d['ag_18']
V['C04_avg_p_1tot_wr_ag']   = lambda d: (d['mean_va'] + 0.1 * d['tot_va']) * d['wr_va'] * d['ag_18']
V['C05_avg_sqrttot_ag']      = lambda d: d['mean_va'] * np.sqrt(d['tot_va'].clip(0)) * d['ag_18']
V['C06_tot_wr_logS']         = lambda d: d['tot_va'] * d['wr_va'] * np.log2(1 + d['n_seeds'])
V['C07_avg_cnt_ag']           = lambda d: d['mean_va'] * d['cnt_va'] * d['ag_18']
V['C08_tot_pnl_wr2']         = lambda d: d['tot_va'] * (d['wr_va'] ** 2) * d['ag_18']

# ---- D. RISK-ADJUSTED ----
V['D01_sharpe_ag']           = lambda d: d['sharpe'] * d['ag_18']
V['D02_sharpe_wr_ag']        = lambda d: d['sharpe'] * d['wr_va'] * d['ag_18']
V['D03_pf_ag']                = lambda d: d['pf'] * d['ag_18']
V['D04_pf_pnl_ag']            = lambda d: d['pf'] * d['mean_va'] * d['ag_18']
V['D05_kelly_ag']              = lambda d: d['kelly'] * d['ag_18']
V['D06_kelly_pnl_ag']          = lambda d: d['kelly'] * d['mean_va'] * d['ag_18']
V['D07_sortino_ag']            = lambda d: d['sortino'] * d['ag_18']
V['D08_pnl_div_range_ag']     = lambda d: (d['mean_va'] / d['range_va'].clip(1e-4)) * d['ag_18']
V['D09_pf_wr_ag']              = lambda d: d['pf'] * d['wr_va'] * d['ag_18']

# ---- E. STABILITY ----
V['E01_stab_wt']              = lambda d: d['mean_va'] * d['stab'].clip(upper=1.5) * d['ag_18']
V['E02_geo_wr_ag']            = lambda d: d['geo_per'] * d['wr_va'] * d['ag_18']
V['E03_min_per_wr_ag']        = lambda d: d['min_per'] * d['wr_va'] * d['ag_18']
V['E04_cons_heavy']           = lambda d: d['mean_va'] * d['cons'] * d['wr_va'] * d['ag_18']
V['E05_wr_stab_ag']           = lambda d: d['mean_va'] * d['wr_stab'] * d['ag_18']
V['E06_geo_cons_ag']          = lambda d: d['geo_per'] * d['cons'] * d['ag_18']
V['E07_harmonic_opt_va']      = lambda d: (2 / (1/d['mean_opt'].clip(1e-4) + 1/d['mean_va'].clip(1e-4))) * d['wr_va'] * d['ag_18']

# ---- F. NONLINEAR TRANSFORMS ----
V['F01_sigm50']       = lambda d: (1 / (1 + np.exp(-50 * d['mean_va']))) * d['wr_va'] * d['ag_18']
V['F02_log']           = lambda d: np.log1p(d['mean_va'] * 100) * d['wr_va'] * d['ag_18']
V['F03_sqrt_pnl']      = lambda d: np.sqrt(d['mean_va'].clip(0)) * d['wr_va'] * d['ag_18']
V['F04_tanh20']        = lambda d: np.tanh(20 * d['mean_va']) * d['wr_va'] * d['ag_18']
V['F05_cbrt_all']      = lambda d: (d['mean_va'] * d['wr_va'] * d['ag_18']) ** (1/3)
V['F06_harmonic_pnl_wr'] = lambda d: 2 / (1/d['mean_va'].clip(1e-4) + 1/d['wr_va'].clip(0.01)) * d['ag_18']
V['F07_sigm_pf']       = lambda d: (1 / (1 + np.exp(-30 * d['mean_va']))) * d['pf'] * d['ag_18']
V['F08_softplus']      = lambda d: np.log1p(np.exp(30 * d['mean_va'] - 0.5)) * d['wr_va'] * d['ag_18']
V['F09_exp_decay']     = lambda d: (1 - np.exp(-40 * d['mean_va'])) * d['wr_va'] * d['ag_18']
V['F10_power_sigm']    = lambda d: ((d['mean_va'] ** 1.5) * (1 / (1 + np.exp(-5 * (d['wr_va'] - 0.55))))) * d['ag_18']

# ---- G. BAYESIAN SHRINKAGE ----
V['G01_bay5_wr_ag']    = lambda d: d['bayes5'] * d['wr_va'] * d['ag_18']
V['G02_bay10_wr_ag']   = lambda d: d['bayes10'] * d['wr_va'] * d['ag_18']
V['G03_bay5_ag']        = lambda d: d['bayes5'] * d['ag_18']
for k in [2, 3, 7, 15, 20]:
    pm = pop_mu
    V[f'G10_k{k}'] = (
        lambda k_=k, m_=pm: lambda d:
            ((d['n_seeds'] * d['mean_va'] + k_ * m_) / (d['n_seeds'] + k_)) * d['wr_va'] * d['ag_18']
    )()

# ---- H. RANK / ZSCORE / PERCENTILE ----
def _rank_equal(d):
    r1 = d.groupby('ticker')['mean_va'].rank(ascending=False, method='min')
    r2 = d.groupby('ticker')['wr_va'].rank(ascending=False, method='min')
    r3 = d.groupby('ticker')['ag_18'].rank(ascending=False, method='min')
    return -(r1 + r2 + r3)
def _rank_pnl2x(d):
    r1 = d.groupby('ticker')['mean_va'].rank(ascending=False, method='min')
    r2 = d.groupby('ticker')['wr_va'].rank(ascending=False, method='min')
    r3 = d.groupby('ticker')['ag_18'].rank(ascending=False, method='min')
    return -(2 * r1 + r2 + r3)
def _rank_pnl_pf(d):
    r1 = d.groupby('ticker')['mean_va'].rank(ascending=False, method='min')
    r2 = d.groupby('ticker')['pf'].rank(ascending=False, method='min')
    r3 = d.groupby('ticker')['ag_18'].rank(ascending=False, method='min')
    return -(r1 + r2 + r3)
def _zscore(d):
    z1 = d.groupby('ticker')['mean_va'].transform(lambda x: (x - x.mean()) / max(x.std(), 1e-4))
    z2 = d.groupby('ticker')['wr_va'].transform(lambda x: (x - x.mean()) / max(x.std(), 1e-4))
    z3 = d.groupby('ticker')['ag_18'].transform(lambda x: (x - x.mean()) / max(x.std(), 1e-4))
    return z1 + z2 + z3
def _pctile(d):
    p1 = d.groupby('ticker')['mean_va'].rank(pct=True)
    p2 = d.groupby('ticker')['wr_va'].rank(pct=True)
    p3 = d.groupby('ticker')['ag_18'].rank(pct=True)
    return p1 * p2 * p3
def _zscore_4feat(d):
    z1 = d.groupby('ticker')['mean_va'].transform(lambda x: (x - x.mean()) / max(x.std(), 1e-4))
    z2 = d.groupby('ticker')['wr_va'].transform(lambda x: (x - x.mean()) / max(x.std(), 1e-4))
    z3 = d.groupby('ticker')['ag_18'].transform(lambda x: (x - x.mean()) / max(x.std(), 1e-4))
    z4 = d.groupby('ticker')['pf'].transform(lambda x: (x - x.mean()) / max(x.std(), 1e-4))
    return z1 + z2 + z3 + z4

V['H01_rank_equal']   = _rank_equal
V['H02_rank_pnl2x']   = _rank_pnl2x
V['H03_rank_pnl_pf']  = _rank_pnl_pf
V['H04_zscore']         = _zscore
V['H05_percentile']     = _pctile
V['H06_zscore_4feat']   = _zscore_4feat

# ---- I. COMBINED BEST IDEAS ----
V['I01_pnl_pf_ag']          = lambda d: d['mean_va'] * d['pf'] * d['ag_18']
V['I02_sharpe_cons_ag']      = lambda d: d['sharpe'] * d['cons'] * d['ag_18']
V['I03_geo_pf_ag']           = lambda d: d['geo_per'] * d['pf'] * d['ag_18']
V['I04_kelly_cons_ag']       = lambda d: d['kelly'] * d['cons'] * d['ag_18']
V['I05_bay_sharpe_ag']       = lambda d: d['bayes5'] * d['sharpe'] * d['ag_18']
V['I06_tot_pf_ag']           = lambda d: d['tot_va'] * d['pf'] * d['ag_18']
V['I07_pnl_wr_cons_ag']      = lambda d: d['mean_va'] * d['wr_va'] * d['cons'] * d['ag_18']
V['I08_min_pf_ag']            = lambda d: d['min_per'] * d['pf'] * d['ag_18']
V['I09_pnl_kelly_ag']         = lambda d: d['mean_va'] * d['kelly'] * d['ag_18']
V['I10_geo_cons_pf']          = lambda d: d['geo_per'] * d['cons'] * d['pf']
V['I11_pnl_wr_moddiv']        = lambda d: d['mean_va'] * d['wr_va'] * d['ag_18'] * np.log2(1 + d['mode_div'])
V['I12_pnl_wr_expdiv']        = lambda d: d['mean_va'] * d['wr_va'] * d['ag_18'] * np.log2(1 + d['exp_div'])
V['I13_pnl_pf_cons_ag']       = lambda d: d['mean_va'] * d['pf'] * d['cons'] * d['ag_18']
V['I14_sharpe_pf_ag']          = lambda d: d['sharpe'] * d['pf'] * d['ag_18']
V['I15_tot_sharpe_ag']         = lambda d: d['tot_va'] * d['sharpe'] * d['ag_18']

# ---- J. LIMIT ENTRY BONUS ----
for bonus in [1.3, 1.5, 2.0]:
    for bk, bfn in [('pnl_wr_ag', lambda d: d['mean_va'] * d['wr_va'] * d['ag_18']),
                     ('pnl_pf_ag', lambda d: d['mean_va'] * d['pf'] * d['ag_18']),
                     ('sharpe_ag', lambda d: d['sharpe'] * d['ag_18']),
                     ('tot_wr_ag', lambda d: d['tot_va'] * d['wr_va'] * d['ag_18'])]:
        V[f'J_{bk}_lb{bonus}'] = (
            lambda fn=bfn, b=bonus: lambda d: fn(d) * np.where(d['is_lim'] == 1, b, 1.0)
        )()

# ---- K. ORACLE ----
V['K_ORACLE'] = lambda d: d['vb_pnl']

# ---- L. LIMIT-ONLY SUBSETS ----
for lname, lfn in [
    ('L01_lim_pnl_wr_ag',   lambda d: d['mean_va'] * d['wr_va'] * d['ag_18']),
    ('L02_lim_pnl_pf_ag',   lambda d: d['mean_va'] * d['pf'] * d['ag_18']),
    ('L03_lim_sharpe_ag',    lambda d: d['sharpe'] * d['ag_18']),
    ('L04_lim_tot_wr_ag',    lambda d: d['tot_va'] * d['wr_va'] * d['ag_18']),
    ('L05_lim_bay5_wr_ag',   lambda d: d['bayes5'] * d['wr_va'] * d['ag_18']),
    ('L06_lim_geo_pf_ag',    lambda d: d['geo_per'] * d['pf'] * d['ag_18']),
    ('L07_lim_pnl_wr_cons',  lambda d: d['mean_va'] * d['wr_va'] * d['cons'] * d['ag_18']),
    ('L08_lim_pnl2_ag',      lambda d: (d['mean_va'] ** 2) * d['ag_18']),
]:
    V[lname] = lfn
    IS_LIM_ONLY.add(lname)


# ==============================================================================
# 5. RUN ALL VARIANTS
# ==============================================================================
n_total = len(V)
print(f"\nTotal variants: {n_total} ({len(IS_LIM_ONLY)} limit-only)")
print("Running...", flush=True)

results = []
for i, (name, fn) in enumerate(sorted(V.items())):
    lim = name in IS_LIM_ONLY
    sel = run_score(sa, fn, lim_only=lim)
    m = metrics(sel)
    m['variant'] = name
    results.append(m)

print(f"Done — {len(results)} variants evaluated")


# ==============================================================================
# 6. ML-BASED SCORING
# ==============================================================================
print(f"\n{'='*W}")
print(f"  ML-BASED SCORING (cross-validated by ticker group)")
print(f"{'='*W}")

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold, cross_val_predict

    FEATS = ['mean_va', 'med_va', 'std_va', 'wr_va', 'cnt_va', 'freq_va',
             'mean_opt', 'wr_opt', 'cnt_opt', 'n_seeds',
             'ag_18', 'ag_15', 'ag_12', 'ag_10',
             'cons', 'tot_va', 'tot_opt', 'stab', 'sharpe',
             'pf', 'kelly', 'sortino', 'geo_per',
             'bayes5', 'min_per', 'is_lim',
             'tp_pct', 'sl_pct', 'wr_stab', 'mode_div', 'exp_div']

    ml = sa.dropna(subset=['vb_pnl']).copy()
    X = ml[FEATS].fillna(0).values
    y = ml['vb_pnl'].values
    groups = ml['ticker'].values
    n_groups = len(set(groups))

    gkf = GroupKFold(n_splits=min(5, n_groups))
    print(f"  Data: {len(ml)} rows, {n_groups} tickers, {len(FEATS)} features")
    print(f"  CV: {gkf.get_n_splits()} folds (GroupKFold by ticker)\n")

    ml_models = [
        ('ML_Ridge', Ridge(alpha=1.0)),
        ('ML_RF',    RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42, n_jobs=-1)),
        ('ML_GBT',   GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)),
    ]

    ml_preds = {}
    for mname, model in ml_models:
        preds = cross_val_predict(model, X, y, cv=gkf, groups=groups)
        ml_preds[mname] = preds
        sa[mname] = np.nan
        sa.loc[ml.index, mname] = preds

        sa_ml = sa.dropna(subset=[mname]).copy()
        sel = run_score(sa_ml, lambda d, c=mname: d[c])
        m = metrics(sel)
        m['variant'] = mname
        results.append(m)

        corr = np.corrcoef(preds, y)[0, 1]
        rmse = np.sqrt(np.mean((preds - y) ** 2))
        print(f"  {mname}: corr={corr:.3f}, RMSE={rmse:.4f}, "
              f"selected={m['n']} tickers, avg_vb={m['avg_vb']:+.2%}")

    # ensemble
    sa['ML_Ens'] = np.nan
    ens_pred = np.mean(list(ml_preds.values()), axis=0)
    sa.loc[ml.index, 'ML_Ens'] = ens_pred
    sa_ml = sa.dropna(subset=['ML_Ens']).copy()
    sel = run_score(sa_ml, lambda d: d['ML_Ens'])
    m = metrics(sel)
    m['variant'] = 'ML_Ensemble'
    results.append(m)
    corr = np.corrcoef(ens_pred, y)[0, 1]
    print(f"  ML_Ensemble: corr={corr:.3f}, "
          f"selected={m['n']} tickers, avg_vb={m['avg_vb']:+.2%}")

    # feature importance
    print(f"\n  {'Feature':<20} {'RF':>8} {'GBT':>8} {'Sum':>8}")
    print(f"  {'-'*48}")
    for mname, model in ml_models[1:]:
        model.fit(X, y)
    rf_imp  = dict(zip(FEATS, ml_models[1][1].feature_importances_))
    gbt_imp = dict(zip(FEATS, ml_models[2][1].feature_importances_))
    for feat in sorted(FEATS, key=lambda f: -(rf_imp.get(f, 0) + gbt_imp.get(f, 0))):
        s = rf_imp[feat] + gbt_imp[feat]
        if s > 0.015:
            print(f"  {feat:<20} {rf_imp[feat]:>7.3f} {gbt_imp[feat]:>7.3f} {s:>7.3f}")

    HAS_ML = True
except ImportError:
    print("  sklearn not available — skipping ML scoring")
    HAS_ML = False


# ==============================================================================
# 7. RESULTS TABLE — TOP 60
# ==============================================================================
res = pd.DataFrame(results)
ranked = res[res['n'] > 0].sort_values('avg_vb', ascending=False).reset_index(drop=True)

print(f"\n\n{'='*W}")
print(f"  ALL VARIANTS — sorted by avg val_b PnL    ({len(ranked)} with N>0 / {len(res)} total)")
print(f"{'='*W}")
print(f"\n  {'#':>3} {'Variant':<35} {'N':>3} {'Avg_vb':>8} {'Med_vb':>8} {'%Pos':>6} "
      f"{'Tot_vb':>9} {'Avg_va':>8} {'#Lim':>4}")
print(f"  {'-'*(W-4)}")

for i, (_, r) in enumerate(ranked.head(60).iterrows(), 1):
    tag = ""
    if r['variant'] == 'K_ORACLE':       tag = " *ORC"
    elif r['variant'].startswith('ML_'): tag = " *ML"
    elif r['variant'] == 'A01_pnl_wr_ag': tag = " <BL"
    print(f"  {i:>3} {r['variant']:<35} {r['n']:>3.0f} {r['avg_vb']:>+7.2%} "
          f"{r['med_vb']:>+7.2%} {r['pos']:>5.0%} {r['tot_vb']:>+8.2%} "
          f"{r['avg_va']:>+7.2%} {r['nlim']:>4.0f}{tag}")

# where is baseline?
bl = ranked[ranked['variant'] == 'A01_pnl_wr_ag']
if not bl.empty:
    bl_rank = bl.index[0] + 1
    b = bl.iloc[0]
    print(f"\n  Baseline rank: #{bl_rank} / {len(ranked)}  "
          f"(avg_vb={b['avg_vb']:+.2%}, N={b['n']:.0f}, tot_vb={b['tot_vb']:+.2%})")


# ==============================================================================
# 8. CATEGORY WINNERS
# ==============================================================================
cats = [
    ('A Baseline',    'A'),
    ('B Power-law',   'B_'),
    ('C Total PnL',   'C'),
    ('D Risk-adj',    'D'),
    ('E Stability',   'E'),
    ('F Nonlinear',   'F'),
    ('G Bayesian',    'G'),
    ('H Rank/Norm',   'H'),
    ('I Combined',    'I'),
    ('J Limit+',      'J_'),
    ('L Limit-only',  'L'),
    ('ML',            'ML'),
]

print(f"\n\n{'='*W}")
print(f"  CATEGORY WINNERS")
print(f"{'='*W}")
print(f"\n  {'Category':<16} {'Best variant':<35} {'N':>3} {'Avg_vb':>8} {'Tot_vb':>9} "
      f"{'%Pos':>6} {'Avg_va':>8} {'#Lim':>4}")
print(f"  {'-'*(W-4)}")

for cat_name, prefix in cats:
    cat_rows = ranked[ranked['variant'].str.startswith(prefix)]
    if cat_rows.empty:
        continue
    c = cat_rows.iloc[0]
    print(f"  {cat_name:<16} {c['variant']:<35} {c['n']:>3.0f} "
          f"{c['avg_vb']:>+7.2%} {c['tot_vb']:>+8.2%} {c['pos']:>5.0%} "
          f"{c['avg_va']:>+7.2%} {c['nlim']:>4.0f}")


# ==============================================================================
# 9. POWER-LAW HEATMAP
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  POWER-LAW HEATMAP — avg_vb for each (alpha_pnl, gamma_agree), best beta_wr per cell")
print(f"{'='*W}")

pow_rows = ranked[ranked['variant'].str.startswith('B_')].copy()
if not pow_rows.empty:
    def _parse_pow(name):
        parts = name.replace('B_', '').split('_')
        return float(parts[0][1:]), float(parts[1][1:]), float(parts[2][1:])
    parsed = pow_rows['variant'].apply(lambda x: pd.Series(_parse_pow(x), index=['alpha', 'beta', 'gamma']))
    pow_rows = pd.concat([pow_rows, parsed], axis=1)

    best_per_ag = pow_rows.loc[pow_rows.groupby(['alpha', 'gamma'])['avg_vb'].idxmax()]
    pivot_vb = best_per_ag.pivot_table(index='alpha', columns='gamma', values='avg_vb')
    pivot_b  = best_per_ag.pivot_table(index='alpha', columns='gamma', values='beta')

    gammas = sorted(pivot_vb.columns)
    print(f"\n  avg_vb (best beta in parentheses)")
    print(f"  {'a \\ g':>8}", end='')
    for g in gammas:
        print(f" {g:>11}", end='')
    print()
    print(f"  {'-'*(8 + 12*len(gammas))}")
    for a in sorted(pivot_vb.index):
        print(f"  {a:>8.1f}", end='')
        for g in gammas:
            v = pivot_vb.loc[a, g] if g in pivot_vb.columns else 0
            b = pivot_b.loc[a, g] if g in pivot_b.columns else 0
            print(f" {v:>+6.2%}({b:.1f})", end='')
        print()

    best_p = pow_rows.iloc[0]
    a_, b_, g_ = _parse_pow(best_p['variant'])
    print(f"\n  Best power-law: pnl^{a_} * wr^{b_} * agree^{g_} "
          f"-> avg_vb={best_p['avg_vb']:+.2%}, N={best_p['n']:.0f}")


# ==============================================================================
# 10. PARETO FRONTIER
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  PARETO FRONTIER — avg_vb vs n_tickers at varying deploy thresholds")
print(f"{'='*W}")

thresholds = [0.010, 0.012, 0.015, 0.018, 0.020, 0.025, 0.030]

show_methods = ranked[~ranked['variant'].str.startswith('K_')].head(12)['variant'].tolist()
if 'A01_pnl_wr_ag' not in show_methods:
    show_methods.append('A01_pnl_wr_ag')

ml_cols = ['ML_Ridge', 'ML_RF', 'ML_GBT', 'ML_Ens']

print(f"\n  {'Variant':<35}", end='')
for t in thresholds:
    print(f" {t:.1%}     ", end='')
print()
print(f"  {' '*35}", end='')
for _ in thresholds:
    print(f"  N   vb   ", end='')
print()
print(f"  {'-'*(35 + len(thresholds)*11)}")

for vname in show_methods:
    print(f"  {vname:<35}", end='')
    for t in thresholds:
        if vname in V:
            lim = vname in IS_LIM_ONLY
            sel = run_score(sa, V[vname], thresh=t, lim_only=lim)
        elif vname.startswith('ML_'):
            col = vname.replace('ML_Ensemble', 'ML_Ens')
            if col in sa.columns:
                sa_ml = sa.dropna(subset=[col]).copy()
                sel = run_score(sa_ml, lambda d, c=col: d[c], thresh=t)
            else:
                sel = pd.DataFrame()
        else:
            sel = pd.DataFrame()
        m = metrics(sel)
        print(f" {m['n']:>2} {m['avg_vb']:>+6.2%}", end='')
    print()


# ==============================================================================
# 11. PER-TICKER BREAKDOWN — TOP 3 + BASELINE + ORACLE
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  PER-TICKER — top 3 methods vs baseline vs oracle")
print(f"{'='*W}")

show_v = ranked[~ranked['variant'].str.startswith('K_') &
                ~ranked['variant'].str.startswith('ML_')].head(3)['variant'].tolist()
if 'A01_pnl_wr_ag' not in show_v:
    show_v.append('A01_pnl_wr_ag')
show_v.append('K_ORACLE')

ticker_data = {}
for vname in show_v:
    if vname in V:
        lim = vname in IS_LIM_ONLY
        sel = run_score(sa, V[vname], lim_only=lim)
    else:
        sel = pd.DataFrame()
    if not sel.empty:
        for _, r in sel.iterrows():
            tk = r['ticker']
            if tk not in ticker_data:
                ticker_data[tk] = {}
            ticker_data[tk][vname] = {'vb': r['vb_pnl'], 'va': r['mean_va'],
                                       'mode': r['tpsl_mode']}

all_tickers = sorted(ticker_data.keys())
col_w = max(18, max(len(v) for v in show_v) + 2) if show_v else 18

print(f"\n  {'Ticker':<8}", end='')
for v in show_v:
    print(f" {v[:col_w-1]:>{col_w-1}}", end='')
print()
print(f"  {'-'*(8 + len(show_v) * col_w)}")

for tk in all_tickers:
    print(f"  {tk:<8}", end='')
    for v in show_v:
        d = ticker_data.get(tk, {}).get(v)
        if d:
            m_short = d['mode'][:5]
            print(f" {d['vb']:>+6.2%} {m_short:>5} ", end=' ' * max(0, col_w - 14))
        else:
            print(f" {'--':>{col_w-1}}", end='')
    print()


# ==============================================================================
# 12. AGREEMENT BETWEEN METHODS (overlap analysis)
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TICKER OVERLAP — top 5 methods")
print(f"{'='*W}")

top5 = ranked[~ranked['variant'].str.startswith('K_')].head(5)['variant'].tolist()
top5_tickers = {}
for vname in top5:
    if vname in V:
        lim = vname in IS_LIM_ONLY
        sel = run_score(sa, V[vname], lim_only=lim)
    elif vname.startswith('ML_'):
        col = vname.replace('ML_Ensemble', 'ML_Ens')
        if col in sa.columns:
            sa_ml = sa.dropna(subset=[col]).copy()
            sel = run_score(sa_ml, lambda d, c=col: d[c])
        else:
            sel = pd.DataFrame()
    else:
        sel = pd.DataFrame()
    top5_tickers[vname] = set(sel['ticker']) if not sel.empty else set()

print(f"\n  Pair-wise overlap:")
for i, v1 in enumerate(top5):
    for v2 in top5[i+1:]:
        s1, s2 = top5_tickers[v1], top5_tickers[v2]
        inter = s1 & s2
        union = s1 | s2
        jaccard = len(inter) / len(union) if union else 0
        print(f"    {v1[:28]:>28} <-> {v2[:28]:<28}  "
              f"overlap={len(inter)}/{len(union)} (J={jaccard:.0%})")

# all methods agree on these tickers
all_agree = set.intersection(*top5_tickers.values()) if top5_tickers else set()
any_agree = set.union(*top5_tickers.values()) if top5_tickers else set()
print(f"\n  All 5 agree: {sorted(all_agree) if all_agree else 'none'}")
print(f"  Any method:  {sorted(any_agree)}")


# ==============================================================================
# 13. FINAL SUMMARY
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  SUMMARY")
print(f"{'='*W}")

if not bl.empty:
    b = bl.iloc[0]
    print(f"\n  Baseline (pnl*wr*ag): avg_vb={b['avg_vb']:+.2%}, "
          f"N={b['n']:.0f}, tot_vb={b['tot_vb']:+.2%}")

oracle_row = ranked[ranked['variant'] == 'K_ORACLE']
if not oracle_row.empty:
    o = oracle_row.iloc[0]
    print(f"  Oracle (cheat):       avg_vb={o['avg_vb']:+.2%}, "
          f"N={o['n']:.0f}, tot_vb={o['tot_vb']:+.2%}")

non_cheat = ranked[~ranked['variant'].str.startswith('K_')]
if not non_cheat.empty:
    top1 = non_cheat.iloc[0]
    print(f"  Best non-oracle:      avg_vb={top1['avg_vb']:+.2%}, "
          f"N={top1['n']:.0f}, tot_vb={top1['tot_vb']:+.2%} -- {top1['variant']}")

if HAS_ML:
    ml_rows = ranked[ranked['variant'].str.startswith('ML_')]
    if not ml_rows.empty:
        bml = ml_rows.iloc[0]
        print(f"  Best ML:              avg_vb={bml['avg_vb']:+.2%}, "
              f"N={bml['n']:.0f}, tot_vb={bml['tot_vb']:+.2%} -- {bml['variant']}")

if not oracle_row.empty and not bl.empty:
    b_vb = bl.iloc[0]['avg_vb']
    o_vb = oracle_row.iloc[0]['avg_vb']
    t_vb = top1['avg_vb'] if not non_cheat.empty else b_vb
    if o_vb != b_vb:
        gap = (t_vb - b_vb) / (o_vb - b_vb) * 100
        print(f"\n  Gap captured (baseline -> oracle): {gap:.0f}%")
    else:
        print(f"\n  No gap (baseline = oracle)")

print(f"\n  Total variants tested: {len(res)}")
print(f"  With results (N>0):   {len(ranked)}")
print(f"  Power-law combos:     {sum(1 for v in V if v.startswith('B_'))}")
if HAS_ML:
    print(f"  ML models:            {len(ml_models) + 1}")
