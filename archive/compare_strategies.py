"""
compare_strategies.py — BQ-only strategy comparison & ranking
=============================================================
Defines 16 portfolio construction strategies, evaluates each using
val_b (out-of-sample PnL), and produces a single ranking table.

Run this first, pick top 3, then update STRATEGIES_TO_RUN in
backtest_strategies.py for live TF scoring backtest.

Colab-ready (no GPU needed).
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

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
W = 140
DEPLOY_THRESHOLD = 0.018
HIGH_THRESHOLD   = 0.030
MIN_SEEDS = 3

TICKER_RENAME   = {'MIGI': 'BGDE', 'BITF': 'KEEL'}
EXCLUDE_TICKERS = {'TLT', 'VIX', 'SMLR', 'SPIR'}

FAM_LABELS = {
    '152': '152 Late', '160': '160 Early',
    '170': '170-175 WF', '180': '180-185 WF', '200': '200 CNN-TX',
}

score_pnl2 = lambda d: (d['mean_va'] ** 2) * d['ag_18']
score_stab = lambda d: (np.sqrt(d['mean_opt'].clip(1e-4) * d['mean_va'].clip(1e-4))
                         * d['wr_va'] * d['ag_18'])


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


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
q = '''
SELECT ticker, experiment, seed, tpsl_mode, entry_minute,
       prob_thresh, tp_pct, sl_pct,
       tp_mult, sl_mult, trail_start_mult, trail_end_mult,
       grace_minutes, dip_pct, max_wait_minutes, vol_filter_pct,
       opt_avg_pnl, opt_win_rate, opt_count,
       val_a_avg_pnl, val_a_win_rate, val_a_count, val_a_freq,
       val_b_avg_pnl, val_b_win_rate, val_b_count, val_b_freq,
       survived_a, survived_b, is_ensemble
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

for col in ['grace_minutes', 'max_wait_minutes', 'vol_filter_pct']:
    if col not in df.columns:
        df[col] = 0
    df[col] = df[col].fillna(0).astype(int)
for col in ['dip_pct', 'tp_mult', 'sl_mult', 'trail_start_mult', 'trail_end_mult']:
    if col not in df.columns:
        df[col] = 0.0
    df[col] = df[col].fillna(0.0)

df['exp_fam'] = df['experiment'].apply(assign_family)
df = df[df['exp_fam'] != 'other']
print(f"Loaded {len(df)} rows, {df['ticker'].nunique()} tickers, "
      f"{df['tpsl_mode'].nunique()} modes, "
      f"families: {sorted(df['exp_fam'].unique())}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. BUILD SEED GROUPS
# ══════════════════════════════════════════════════════════════════════════════
GRP = ['ticker', 'experiment', 'tpsl_mode', 'entry_minute',
       'prob_thresh', 'tp_pct', 'sl_pct',
       'tp_mult', 'sl_mult', 'trail_start_mult', 'trail_end_mult',
       'grace_minutes', 'dip_pct', 'max_wait_minutes',
       'vol_filter_pct']

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
sa['exp_fam'] = sa['experiment'].apply(assign_family)
print(f"Seed groups: {len(sa)}, tickers: {sa['ticker'].nunique()}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def select_best(df_in, score_fn=None, thresh=DEPLOY_THRESHOLD, thresh_col='mean_va',
                group_col='ticker'):
    d = df_in.copy()
    if d.empty:
        return pd.DataFrame()
    if score_fn is None:
        score_fn = score_pnl2
    d['score'] = score_fn(d)
    d = d[np.isfinite(d['score'])]
    if d.empty:
        return pd.DataFrame()
    best = d.loc[d.groupby(group_col)['score'].idxmax()]
    return best[best[thresh_col] >= thresh]


def compute_family_agreement(sa_limit):
    d = sa_limit.copy()
    d['score'] = score_pnl2(d)
    d = d[np.isfinite(d['score'])]
    if d.empty:
        return {}
    best = d.loc[d.groupby(['ticker', 'exp_fam'])['score'].idxmax()]
    best = best[best['mean_va'] >= DEPLOY_THRESHOLD]

    agreement = {}
    for tk in best['ticker'].unique():
        tk_rows = best[best['ticker'] == tk]
        n_fams = len(tk_rows)
        n_pos = int((tk_rows['vb_pnl'] > 0).sum())
        agreement[tk] = {
            'n_fams': n_fams,
            'n_pos': n_pos,
            'frac_pos': n_pos / n_fams if n_fams > 0 else 0,
            'all_pos': n_pos == n_fams and n_fams >= 2,
            'majority_pos': n_pos > n_fams / 2 and n_fams >= 2,
            'families': sorted(tk_rows['exp_fam'].tolist()),
            'vb_by_fam': dict(zip(tk_rows['exp_fam'], tk_rows['vb_pnl'])),
        }
    return agreement


def strategy_seed_family(sa_base, sa_limit, agreement, seed_thresh=None, fam_gate='none'):
    d = sa_limit.copy()
    if seed_thresh is not None:
        d = d[d['ag_18'] >= seed_thresh]
    if d.empty:
        return pd.DataFrame()

    if fam_gate != 'none':
        fam_ag = compute_family_agreement(d)
        if fam_gate == 'all':
            ok = {tk for tk, ag in fam_ag.items() if ag['all_pos']}
        elif fam_gate == 'majority':
            ok = {tk for tk, ag in fam_ag.items() if ag['majority_pos']}
        elif fam_gate == 'half':
            ok = {tk for tk, ag in fam_ag.items()
                  if ag['frac_pos'] >= 0.5 and ag['n_fams'] >= 2}
        else:
            ok = set(d['ticker'].unique())
        d = d[d['ticker'].isin(ok)]

    return select_best(d, score_fn=score_pnl2)


# ══════════════════════════════════════════════════════════════════════════════
# 4. COMPUTE FAMILY AGREEMENT
# ══════════════════════════════════════════════════════════════════════════════
sa_base = sa[sa['n_seeds'] >= MIN_SEEDS].copy()
sa_limit = sa_base[sa_base['tpsl_mode'] == 'limit_entry'].copy()

agreement = compute_family_agreement(sa_limit)
print(f"Family agreement computed for {len(agreement)} tickers")


# ══════════════════════════════════════════════════════════════════════════════
# 4b. OPTIMIZATION COVERAGE — how complete is each family?
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n\n{'═'*W}")
print(f"  OPTIMIZATION COVERAGE — per ticker × family (limit_entry, any seed count)")
print(f"{'═'*W}")

all_fams = sorted(df['exp_fam'].unique())
all_tickers_raw = sorted(df['ticker'].unique())

# Raw coverage: any limit_entry row exists (even 1 seed, even below threshold)
raw_limit = df[df['tpsl_mode'] == 'limit_entry']
# Qualifying: >= MIN_SEEDS and va >= threshold
qual_limit = sa_limit[sa_limit['mean_va'] >= DEPLOY_THRESHOLD]

print(f"\n  Legend: ✓=qualifying (≥{MIN_SEEDS}s, va≥{DEPLOY_THRESHOLD:.1%})  ·=optimized but below thresh  ─=NOT optimized")
print(f"\n  {'Ticker':<8}", end='')
for fam in all_fams:
    print(f" {FAM_LABELS.get(fam, fam)[:8]:>10}", end='')
print(f"  {'Qual':>4} {'Opt':>4} {'Miss':>4}")
print(f"  {'─'*(8 + 11*len(all_fams) + 15)}")

coverage_stats = {'total_tickers': 0, 'fully_optimized': 0, 'partially_optimized': 0,
                  'fam_coverage': {f: {'optimized': 0, 'qualifying': 0} for f in all_fams}}

for tk in all_tickers_raw:
    print(f"  {tk:<8}", end='')
    n_qual = 0
    n_opt = 0
    for fam in all_fams:
        has_raw = len(raw_limit[(raw_limit['ticker'] == tk) & (raw_limit['exp_fam'] == fam)]) > 0
        has_qual = len(qual_limit[(qual_limit['ticker'] == tk) & (qual_limit['exp_fam'] == fam)]) > 0
        if has_qual:
            print(f" {'✓':>10}", end='')
            n_qual += 1
            n_opt += 1
            coverage_stats['fam_coverage'][fam]['qualifying'] += 1
            coverage_stats['fam_coverage'][fam]['optimized'] += 1
        elif has_raw:
            print(f" {'·':>10}", end='')
            n_opt += 1
            coverage_stats['fam_coverage'][fam]['optimized'] += 1
        else:
            print(f" {'─':>10}", end='')
    n_miss = len(all_fams) - n_opt
    print(f"  {n_qual:>4} {n_opt:>4} {n_miss:>4}")
    coverage_stats['total_tickers'] += 1
    if n_miss == 0:
        coverage_stats['fully_optimized'] += 1
    elif n_opt > 0:
        coverage_stats['partially_optimized'] += 1

print(f"\n  COVERAGE SUMMARY:")
print(f"  Total tickers: {coverage_stats['total_tickers']}")
print(f"  Fully optimized (all {len(all_fams)} families): {coverage_stats['fully_optimized']}")
print(f"  Partially optimized: {coverage_stats['partially_optimized']}")
print(f"  Not optimized at all: {coverage_stats['total_tickers'] - coverage_stats['fully_optimized'] - coverage_stats['partially_optimized']}")
print(f"\n  Per-family coverage:")
for fam in all_fams:
    fc = coverage_stats['fam_coverage'][fam]
    label = FAM_LABELS.get(fam, fam)
    print(f"    {label:<14} optimized: {fc['optimized']:>2}/{coverage_stats['total_tickers']}, "
          f"qualifying: {fc['qualifying']:>2}/{coverage_stats['total_tickers']}")

# Which tickers could gain family agreement if optimization were complete?
print(f"\n  TICKERS WITH INCOMPLETE OPTIMIZATION (limit_entry missing in ≥1 family):")
for tk in sorted(agreement.keys()):
    ag = agreement[tk]
    missing = [f for f in all_fams if f not in ag['families']]
    has_raw_missing = []
    no_data_missing = []
    for f in missing:
        if len(raw_limit[(raw_limit['ticker'] == tk) & (raw_limit['exp_fam'] == f)]) > 0:
            has_raw_missing.append(f)
        else:
            no_data_missing.append(f)
    if missing:
        tag = "ALL POS" if ag['all_pos'] else f"{ag['n_pos']}/{ag['n_fams']} pos"
        status_parts = []
        if has_raw_missing:
            status_parts.append(f"below-thresh: {has_raw_missing}")
        if no_data_missing:
            status_parts.append(f"NOT RUN: {no_data_missing}")
        status = ', '.join(status_parts)
        print(f"    {tk:<8} has {ag['n_fams']}/{len(all_fams)} fams ({tag}), missing: {status}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. STRATEGY DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════
def strategy_S1(sa_base, sa_limit, agreement):
    """S1: Limit All — limit_entry, best per ticker, standard threshold."""
    return select_best(sa_limit, score_fn=score_pnl2)


def strategy_S2(sa_base, sa_limit, agreement):
    """S2: Limit + Multi-Fam — limit_entry, tickers with >=2 qualifying families."""
    multi_fam = {tk for tk, ag in agreement.items() if ag['n_fams'] >= 2}
    return select_best(sa_limit[sa_limit['ticker'].isin(multi_fam)], score_fn=score_pnl2)


def strategy_S3(sa_base, sa_limit, agreement):
    """S3: Limit + Agreement — limit_entry, majority-positive vb across families."""
    maj_tickers = {tk for tk, ag in agreement.items() if ag['majority_pos']}
    return select_best(sa_limit[sa_limit['ticker'].isin(maj_tickers)], score_fn=score_pnl2)


def strategy_S4(sa_base, sa_limit, agreement):
    """S4: Limit Core — limit_entry, ALL families positive vb."""
    core_tickers = {tk for tk, ag in agreement.items() if ag['all_pos']}
    return select_best(sa_limit[sa_limit['ticker'].isin(core_tickers)], score_fn=score_pnl2)


def strategy_S5(sa_base, sa_limit, agreement):
    """S5: Limit High-Thresh — limit_entry, va >= 3.0%."""
    return select_best(sa_limit, score_fn=score_pnl2, thresh=HIGH_THRESHOLD)


def strategy_S6(sa_base, sa_limit, agreement):
    """S6: All-Mode Best — any mode, best per ticker."""
    return select_best(sa_base, score_fn=score_pnl2)


def strategy_S7(sa_base, sa_limit, agreement):
    """S7: All-Mode + Stability — any mode, stability score."""
    return select_best(sa_base, score_fn=score_stab)


def strategy_S8(sa_base, sa_limit, agreement):
    """S8: Exploit — limit_entry, best per ticker × family (multiple configs/ticker)."""
    d = sa_limit.copy()
    d['score'] = score_pnl2(d)
    d = d[np.isfinite(d['score'])]
    if d.empty:
        return pd.DataFrame()
    best = d.loc[d.groupby(['ticker', 'exp_fam'])['score'].idxmax()]
    return best[best['mean_va'] >= DEPLOY_THRESHOLD]


def strategy_S9(sa_base, sa_limit, agreement):
    """S9: Best Family — limit_entry, pick best config from the family with highest vb per ticker.
    No cross-family requirement. Tests 'best model for each ticker' vs consensus."""
    d = sa_limit.copy()
    d['score'] = score_pnl2(d)
    d = d[np.isfinite(d['score'])]
    if d.empty:
        return pd.DataFrame()
    best_per_fam = d.loc[d.groupby(['ticker', 'exp_fam'])['score'].idxmax()]
    best_per_fam = best_per_fam[best_per_fam['mean_va'] >= DEPLOY_THRESHOLD]
    if best_per_fam.empty:
        return pd.DataFrame()
    best = best_per_fam.loc[best_per_fam.groupby('ticker')['vb_pnl'].idxmax()]
    return best


def strategy_S10(sa_base, sa_limit, agreement):
    """S10: VB-Positive — limit_entry, best per ticker, but only keep if vb > 0.
    Direct out-of-sample filter instead of family agreement proxy."""
    sel = select_best(sa_limit, score_fn=score_pnl2)
    if sel.empty:
        return pd.DataFrame()
    return sel[sel['vb_pnl'] > 0]


def strategy_S11(sa_base, sa_limit, agreement):
    """S11: 180-185 Only — limit_entry, only 180-185 WF family (widest coverage).
    Tests whether the most complete family alone is sufficient."""
    fam_limit = sa_limit[sa_limit['exp_fam'] == '180'].copy()
    return select_best(fam_limit, score_fn=score_pnl2)


def strategy_S12(sa_base, sa_limit, agreement):
    """S12: Seeds100 + AllFam — 100% seeds ≥1.8%, all families positive vb."""
    return strategy_seed_family(sa_base, sa_limit, agreement, seed_thresh=1.0, fam_gate='all')


def strategy_S13(sa_base, sa_limit, agreement):
    """S13: Seeds80 — 80% seeds ≥1.8% (seed-only gate)."""
    return strategy_seed_family(sa_base, sa_limit, agreement, seed_thresh=0.80, fam_gate='none')


def strategy_S14(sa_base, sa_limit, agreement):
    """S14: Seeds50 — 50% seeds ≥1.8% (seed-only gate)."""
    return strategy_seed_family(sa_base, sa_limit, agreement, seed_thresh=0.50, fam_gate='none')


def strategy_S15(sa_base, sa_limit, agreement):
    """S15: Seeds80 + MajFam — 80% seeds ≥1.8% AND majority-positive families."""
    return strategy_seed_family(sa_base, sa_limit, agreement, seed_thresh=0.80, fam_gate='majority')


def strategy_S16(sa_base, sa_limit, agreement):
    """S16: Seeds50 + HalfFam — 50% seeds ≥1.8% AND ≥50% families positive."""
    return strategy_seed_family(sa_base, sa_limit, agreement, seed_thresh=0.50, fam_gate='half')


def compute_family_agreement_allmode(sa_in):
    """Family agreement computed on ALL modes (not just limit_entry)."""
    d = sa_in.copy()
    d['score'] = score_pnl2(d)
    d = d[np.isfinite(d['score'])]
    if d.empty:
        return {}
    best = d.loc[d.groupby(['ticker', 'exp_fam'])['score'].idxmax()]
    best = best[best['mean_va'] >= DEPLOY_THRESHOLD]

    result = {}
    for tk in best['ticker'].unique():
        tk_rows = best[best['ticker'] == tk]
        n_fams = len(tk_rows)
        n_pos = int((tk_rows['vb_pnl'] > 0).sum())
        result[tk] = {
            'n_fams': n_fams,
            'n_pos': n_pos,
            'frac_pos': n_pos / n_fams if n_fams > 0 else 0,
            'all_pos': n_pos == n_fams and n_fams >= 2,
            'majority_pos': n_pos > n_fams / 2 and n_fams >= 2,
        }
    return result


def strategy_S17(sa_base, sa_limit, agreement):
    """S17: AllMode + MajAgree — any mode, best per ticker, majority-positive families.
    Agreement computed across all modes (mode-agnostic)."""
    ag = compute_family_agreement_allmode(sa_base)
    maj_tickers = {tk for tk, a in ag.items() if a['majority_pos']}
    return select_best(sa_base[sa_base['ticker'].isin(maj_tickers)], score_fn=score_pnl2)


def strategy_S18(sa_base, sa_limit, agreement):
    """S18: AllMode + ModeConsistent — per (ticker, mode), check family agreement.
    Only deploy a ticker+mode combo where that specific mode has majority-positive families."""
    d = sa_base.copy()
    d['score'] = score_pnl2(d)
    d = d[np.isfinite(d['score'])]
    if d.empty:
        return pd.DataFrame()

    best_per_fam_mode = d.loc[d.groupby(['ticker', 'tpsl_mode', 'exp_fam'])['score'].idxmax()]
    best_per_fam_mode = best_per_fam_mode[best_per_fam_mode['mean_va'] >= DEPLOY_THRESHOLD]

    passing = []
    for (tk, mode), grp in best_per_fam_mode.groupby(['ticker', 'tpsl_mode']):
        n_fams = len(grp)
        n_pos = int((grp['vb_pnl'] > 0).sum())
        if n_fams >= 2 and n_pos > n_fams / 2:
            passing.append(grp)
    if not passing:
        return pd.DataFrame()
    pool = pd.concat(passing)

    best = pool.loc[pool.groupby('ticker')['score'].idxmax()]
    return best


def strategy_S19(sa_base, sa_limit, agreement):
    """S19: AllMode + AnyModePasses — per ticker, if ANY mode passes majority-positive
    family agreement, deploy that ticker using whichever mode scored highest overall."""
    d = sa_base.copy()
    d['score'] = score_pnl2(d)
    d = d[np.isfinite(d['score'])]
    if d.empty:
        return pd.DataFrame()

    best_per_fam_mode = d.loc[d.groupby(['ticker', 'tpsl_mode', 'exp_fam'])['score'].idxmax()]
    best_per_fam_mode = best_per_fam_mode[best_per_fam_mode['mean_va'] >= DEPLOY_THRESHOLD]

    passing_tickers = set()
    for (tk, mode), grp in best_per_fam_mode.groupby(['ticker', 'tpsl_mode']):
        n_fams = len(grp)
        n_pos = int((grp['vb_pnl'] > 0).sum())
        if n_fams >= 2 and n_pos > n_fams / 2:
            passing_tickers.add(tk)

    if not passing_tickers:
        return pd.DataFrame()
    return select_best(sa_base[sa_base['ticker'].isin(passing_tickers)], score_fn=score_pnl2)


STRATEGIES = {
    'S1': ('Limit All',              strategy_S1, 'limit_entry only, best per ticker'),
    'S2': ('Limit + Multi-Fam',      strategy_S2, 'limit_entry, ≥2 families'),
    'S3': ('Limit + Agreement',      strategy_S3, 'limit_entry, majority-positive vb'),
    'S4': ('Limit Core',             strategy_S4, 'limit_entry, ALL families positive vb'),
    'S5': ('Limit High-Thresh',      strategy_S5, 'limit_entry, va ≥ 3.0%'),
    'S6': ('All-Mode Best',          strategy_S6, 'any mode, best per ticker'),
    'S7': ('All-Mode + Stability',   strategy_S7, 'any mode, stability score'),
    'S8': ('Exploit',                strategy_S8, 'limit_entry, best per ticker × family'),
    'S9': ('Best Family',            strategy_S9, 'limit_entry, best vb family per ticker'),
    'S10': ('VB-Positive',           strategy_S10, 'limit_entry, best per ticker, vb > 0 only'),
    'S11': ('180-185 Only',          strategy_S11, 'limit_entry, 180-185 WF family only'),
    'S12': ('Seeds100 + AllFam',    strategy_S12, '100% seeds ≥1.8%, all families pos vb'),
    'S13': ('Seeds80',              strategy_S13, '80% seeds ≥1.8% (seed-only gate)'),
    'S14': ('Seeds50',              strategy_S14, '50% seeds ≥1.8% (seed-only gate)'),
    'S15': ('Seeds80 + MajFam',     strategy_S15, '80% seeds ≥1.8% + majority fam pos'),
    'S16': ('Seeds50 + HalfFam',    strategy_S16, '50% seeds ≥1.8% + ≥50% fam pos'),
    'S17': ('AllMode + MajAgree',   strategy_S17, 'any mode, majority-pos families (mode-agnostic)'),
    'S18': ('AllMode + ModeConsist', strategy_S18, 'any mode, per-mode family agreement'),
    'S19': ('AllMode + AnyPass',    strategy_S19, 'any mode if ANY mode passes fam agreement'),
}


# ══════════════════════════════════════════════════════════════════════════════
# 6. RUN ALL STRATEGIES
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n\n{'═'*W}")
print(f"  RUNNING {len(STRATEGIES)} STRATEGIES")
print(f"{'═'*W}")

strategy_results = {}
for key, (name, fn, desc) in STRATEGIES.items():
    sel = fn(sa_base, sa_limit, agreement)
    strategy_results[key] = sel
    n = len(sel) if not sel.empty else 0
    tickers = sorted(sel['ticker'].unique().tolist()) if not sel.empty else []
    n_tickers = len(tickers)
    if not sel.empty:
        avg_vb = sel['vb_pnl'].mean()
        pct_pos = (sel['vb_pnl'] > 0).mean()
        print(f"  {key}: {name:<25} → {n_tickers:>2} tickers, {n:>3} configs, "
              f"avg_vb={avg_vb:>+6.2%}, %pos={pct_pos:.0%}")
    else:
        print(f"  {key}: {name:<25} → 0 tickers")


# ══════════════════════════════════════════════════════════════════════════════
# 7. COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n\n{'═'*W}")
print(f"  STRATEGY COMPARISON TABLE — ranked by avg val_b (out-of-sample PnL)")
print(f"{'═'*W}")

rows = []
for key, (name, _, desc) in STRATEGIES.items():
    sel = strategy_results[key]
    if sel.empty:
        rows.append({'key': key, 'name': name, 'desc': desc,
                     'n_tickers': 0, 'n_configs': 0,
                     'avg_vb': 0, 'tot_vb': 0, 'pct_pos': 0,
                     'avg_va': 0, 'avg_wr': 0, 'tickers': ''})
        continue

    unique_tickers = sorted(sel['ticker'].unique())
    vb_per_ticker = sel.groupby('ticker')['vb_pnl'].mean()

    rows.append({
        'key': key, 'name': name, 'desc': desc,
        'n_tickers': len(unique_tickers),
        'n_configs': len(sel),
        'avg_vb': vb_per_ticker.mean(),
        'tot_vb': vb_per_ticker.sum(),
        'pct_pos': (vb_per_ticker > 0).mean(),
        'avg_va': sel.groupby('ticker')['mean_va'].mean().mean(),
        'avg_wr': sel.groupby('ticker')['wr_va'].mean().mean(),
        'tickers': ', '.join(unique_tickers),
    })

comp = pd.DataFrame(rows).sort_values('avg_vb', ascending=False)

print(f"\n  {'#':>2} {'Key':>3} {'Strategy':<25} {'Tkrs':>4} {'Cfgs':>4} "
      f"{'Avg_vb':>8} {'Tot_vb':>8} {'%Pos':>5} {'Avg_va':>8} {'WR':>5}  Tickers")
print(f"  {'─'*(W-4)}")
for i, (_, r) in enumerate(comp.iterrows(), 1):
    if r['n_tickers'] == 0:
        print(f"  {i:>2} {r['key']:>3} {r['name']:<25}    0")
        continue
    tk_str = r['tickers']
    if len(tk_str) > 45:
        tk_str = tk_str[:42] + '...'
    print(f"  {i:>2} {r['key']:>3} {r['name']:<25} {r['n_tickers']:>4} {r['n_configs']:>4} "
          f"{r['avg_vb']:>+7.2%} {r['tot_vb']:>+7.2%} {r['pct_pos']:>4.0%} "
          f"{r['avg_va']:>+7.2%} {r['avg_wr']:>4.0%}  {tk_str}")


# ══════════════════════════════════════════════════════════════════════════════
# 8. HEAD-TO-HEAD — per-ticker avg_vb across strategies
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n\n{'═'*W}")
print(f"  HEAD-TO-HEAD — which tickers appear in which strategy")
print(f"{'═'*W}")

all_tickers = sorted(set().union(*[
    set(strategy_results[k]['ticker'].unique())
    for k in strategy_results if not strategy_results[k].empty
]))
keys_sorted = list(comp['key'])

print(f"\n  {'Ticker':<8}", end='')
for key in keys_sorted:
    print(f" {key:>6}", end='')
print(f"  {'In':>3}")
print(f"  {'─'*(8 + 7*len(keys_sorted) + 5)}")

for tk in all_tickers:
    print(f"  {tk:<8}", end='')
    count = 0
    for key in keys_sorted:
        sel = strategy_results[key]
        if sel.empty:
            print(f" {'':>6}", end='')
            continue
        tk_data = sel[sel['ticker'] == tk]
        if tk_data.empty:
            print(f" {'·':>6}", end='')
        else:
            vb = tk_data['vb_pnl'].mean()
            print(f" {vb:>+5.1%}", end='')
            count += 1
    print(f"  {count:>3}")


# ══════════════════════════════════════════════════════════════════════════════
# 9. FAMILY AGREEMENT MAP
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n\n{'═'*W}")
print(f"  FAMILY AGREEMENT MAP — val_b direction per ticker across experiment families")
print(f"{'═'*W}")

active_fams = sorted(set().union(*[ag['families'] for ag in agreement.values()]))
print(f"\n  {'Ticker':<8}", end='')
for fam in active_fams:
    print(f" {FAM_LABELS.get(fam, fam)[:8]:>10}", end='')
print(f"  {'Tag':>10}")
print(f"  {'─'*(8 + 11*len(active_fams) + 12)}")

for tk in sorted(agreement.keys()):
    ag = agreement[tk]
    print(f"  {tk:<8}", end='')
    for fam in active_fams:
        vb = ag['vb_by_fam'].get(fam)
        if vb is None:
            print(f" {'─':>10}", end='')
        else:
            print(f" {vb:>+9.2%}", end='')
    if ag['all_pos']:
        tag = "ALL POS"
    elif ag['n_pos'] == 0:
        tag = "ALL NEG"
    else:
        tag = f"{ag['n_pos']}/{ag['n_fams']} pos"
    print(f"  {tag:>10}")

core_tickers = sorted(tk for tk, ag in agreement.items() if ag['all_pos'])
maj_tickers = sorted(tk for tk, ag in agreement.items() if ag['majority_pos'])
print(f"\n  ALL POS ({len(core_tickers)}): {core_tickers}")
print(f"  Majority pos ({len(maj_tickers)}): {maj_tickers}")


# ══════════════════════════════════════════════════════════════════════════════
# 10. PER-STRATEGY DETAIL
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n\n{'═'*W}")
print(f"  PER-STRATEGY DETAIL")
print(f"{'═'*W}")

for key in keys_sorted:
    name, _, desc = STRATEGIES[key]
    sel = strategy_results[key]
    if sel.empty:
        continue
    n_tickers = sel['ticker'].nunique()
    avg_vb = sel.groupby('ticker')['vb_pnl'].mean().mean()

    print(f"\n  ─── {key}: {name} ({desc}) ───")
    print(f"  {n_tickers} tickers, avg_vb = {avg_vb:+.2%}")
    print(f"  {'Ticker':<8} {'Experiment':<22} {'Mode':<14} {'va':>7} {'vb':>7} "
          f"{'wr':>5} {'ag':>4} {'score':>8}")
    print(f"  {'-'*80}")

    for _, r in sel.sort_values('vb_pnl', ascending=False).iterrows():
        print(f"  {r['ticker']:<8} {r['experiment'][:21]:<22} {r['tpsl_mode']:<14} "
              f"{r['mean_va']:>+6.2%} {r['vb_pnl']:>+6.2%} "
              f"{r['wr_va']:>4.0%} {r['ag_18']:>3.0%} {r.get('score', 0):>8.5f}")


# ══════════════════════════════════════════════════════════════════════════════
# 11. OVERLAP ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n\n{'═'*W}")
print(f"  OVERLAP ANALYSIS — which tickers are unique to a strategy")
print(f"{'═'*W}")

strat_tickers = {}
for key in keys_sorted:
    sel = strategy_results[key]
    strat_tickers[key] = set(sel['ticker'].unique()) if not sel.empty else set()

for key in keys_sorted:
    others = set().union(*[strat_tickers[k] for k in keys_sorted if k != key])
    unique = sorted(strat_tickers[key] - others)
    shared = sorted(strat_tickers[key] & others)
    name = STRATEGIES[key][0]
    if strat_tickers[key]:
        print(f"  {key}: {name:<25} total={len(strat_tickers[key]):>2}, "
              f"unique={len(unique):>2} {unique if unique else ''}")
    else:
        print(f"  {key}: {name:<25} total= 0")

print(f"\n  Pairwise overlap:")
for i, k1 in enumerate(keys_sorted):
    for k2 in keys_sorted[i+1:]:
        overlap = strat_tickers[k1] & strat_tickers[k2]
        if overlap:
            print(f"    {k1} ∩ {k2}: {len(overlap)} tickers — {sorted(overlap)}")


# ══════════════════════════════════════════════════════════════════════════════
# 12. RECOMMENDATION
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n\n{'═'*W}")
print(f"  RECOMMENDATION — top 3 strategies for backtesting")
print(f"{'═'*W}")

top3 = comp[comp['n_tickers'] > 0].head(3)
print(f"\n  Ranked by avg out-of-sample PnL (val_b):\n")
for i, (_, r) in enumerate(top3.iterrows(), 1):
    print(f"  #{i}  {r['key']}: {r['name']}")
    print(f"      {r['n_tickers']} tickers, avg_vb = {r['avg_vb']:+.2%}, "
          f"tot_vb = {r['tot_vb']:+.2%}, %pos = {r['pct_pos']:.0%}")
    print(f"      Tickers: {r['tickers']}")
    print()

top3_keys = top3['key'].tolist()
print(f"  → Set STRATEGIES_TO_RUN = {top3_keys} in backtest_strategies.py")
print(f"  → Run backtest_strategies.py for day-by-day equity curves (requires GPU)")
print(f"{'═'*W}")
