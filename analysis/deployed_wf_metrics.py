"""
deployed_wf_metrics.py — WF optimization metrics for deployed configs (NO backtest)
====================================================================================
Pulls val_a / val_b / opt performance directly from BQ for:
  1. Currently live BQ-driven deploy (152+160 limit_entry)
  2. Legacy 20-ticker hardcoded deploy (152+160+175+185, mixed modes)
  3. Candidate configs that qualify but aren't deployed

No trade simulation, no GCS data loading, no regime filter.
Just BQ metrics + seed agreement analysis. Colab-ready.
"""

import os
import sys
import subprocess
import warnings
from datetime import datetime

os.environ['TF_USE_LEGACY_KERAS'] = '1'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from google.cloud import bigquery
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "google-cloud-bigquery", "pyarrow", "--quiet"])
    from google.cloud import bigquery

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass

# ==============================================================================
# CONFIG
# ==============================================================================
MODEL_VERSION = 'robust_wf_v12'
BQ_TABLE = f"{GCP_PROJECT}.scoring_logs.wf_optimization_results"
DEPLOY_THRESHOLD = 0.018
MIN_TRADES = 3
W = 120

MODE_SUFFIX = {
    'static': '', 'atr': '_ATR', 'atr_adaptive': '_ATRADP',
    'regime_scaled': '_REGIME', 'trailing_decay': '_TRAIL',
    'regime_adaptive': '_REGADP', 'atr_sl': '_ATRSL',
    'grace': '_GRACE', 'limit_entry': '_LIMIT',
}

TICKER_RENAME = {'MIGI': 'BGDE', 'BITF': 'KEEL'}

DEPLOYED_HARDCODED = [
    # (ticker, base_experiment, tpsl_mode, seed, is_ensemble)
    # EXP_152
    ('APLD', '152_XSECTOR', 'atr_adaptive', 15203, False),
    ('IREN', '152_XSECTOR', 'static', -1, True),
    # EXP_160
    ('CIFR', '160_XSECTOR', 'static', -1, True),
    ('FCEL', '160_XSECTOR', 'atr_adaptive', 16005, False),
    ('KEEL', '160_XSECTOR', 'regime_scaled', -1, True),
    ('PLUG', '160_XSECTOR', 'trailing_decay', 16015, False),
    ('RCAT', '160_XSECTOR', 'atr', 16013, False),
    # EXP_170 (fold 175)
    ('ALAB', '175_XSECTOR', 'static', 17507, False),
    ('ASTS', '175_XSECTOR', 'atr_adaptive', -1, True),
    ('BE',   '175_XSECTOR', 'static', 17500, False),
    ('BTBT', '175_XSECTOR', 'atr_adaptive', 17510, False),
    ('CLSK', '175_XSECTOR', 'regime_scaled', 17504, False),
    ('OKLO', '175_XSECTOR', 'static', 17501, False),
    ('RKLB', '175_XSECTOR', 'static', 17514, False),
    ('WULF', '175_XSECTOR', 'atr_adaptive', 17511, False),
    # EXP_185
    ('PL',   '185_XSECTOR', 'static', 18507, False),
    ('RDW',  '185_XSECTOR', 'atr', 18503, False),
    ('RIOT', '185_XSECTOR', 'trailing_decay', 18503, False),
    ('SMCI', '185_XSECTOR', 'static', -1, True),
]

BQ_DEPLOY_EXPERIMENTS = {'152_XSECTOR', '160_XSECTOR', '170_XSECTOR',
                         '175_XSECTOR', '180_XSECTOR', '185_XSECTOR'}

bq_client = bigquery.Client()


# ==============================================================================
# LOAD ALL WF DATA
# ==============================================================================
def load_all_wf_data():
    print(f"  Loading ALL WF results from BQ (model_version={MODEL_VERSION})...")
    query = f"""
    SELECT ticker, experiment, seed, tpsl_mode, entry_time, entry_minute,
           prob_thresh, tp_pct, sl_pct, tp_mult, sl_mult,
           trail_start_mult, trail_end_mult,
           vix_threshold, tlt_crash, spy_ignore, vol_filter_pct,
           val_a_avg_pnl, val_a_win_rate, val_a_count, val_a_freq,
           val_b_avg_pnl, val_b_win_rate, val_b_count, val_b_freq,
           opt_avg_pnl, opt_win_rate, opt_count,
           is_ensemble, survived_a, survived_b,
           grace_minutes, dip_pct, max_wait_minutes
    FROM `{BQ_TABLE}`
    WHERE model_version = '{MODEL_VERSION}'
      AND survived_a = TRUE
      AND val_a_count >= {MIN_TRADES}
      AND val_a_avg_pnl > 0
    """
    df = bq_client.query(query).to_dataframe()
    df['ticker'] = df['ticker'].replace(TICKER_RENAME)

    all_suffixes = list(MODE_SUFFIX.values())
    def _base_exp(exp_name):
        for sfx in sorted(all_suffixes, key=len, reverse=True):
            if sfx and exp_name.endswith(sfx):
                return exp_name[:-len(sfx)]
        return exp_name
    df['base_experiment'] = df['experiment'].apply(_base_exp)

    for col in ['grace_minutes', 'max_wait_minutes', 'vol_filter_pct']:
        df[col] = df[col].fillna(0).astype(int)
    for col in ['dip_pct', 'tp_mult', 'sl_mult', 'trail_start_mult', 'trail_end_mult']:
        df[col] = df[col].fillna(0.0)
    for col in ['tp_pct', 'sl_pct', 'prob_thresh']:
        df[col] = df[col].fillna(0.0)
    for col in ['val_b_avg_pnl', 'val_b_win_rate', 'val_b_count', 'val_b_freq']:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = df[col].fillna(np.nan)

    print(f"    {len(df)} rows loaded")
    print(f"    Experiments: {sorted(df['base_experiment'].unique())}")
    print(f"    Tickers: {df['ticker'].nunique()}")
    return df


# ==============================================================================
# SEED GROUP ANALYSIS
# ==============================================================================
GRP_KEYS = ['ticker', 'base_experiment', 'tpsl_mode', 'entry_minute', 'prob_thresh',
            'tp_pct', 'sl_pct',
            'tp_mult', 'sl_mult', 'trail_start_mult', 'trail_end_mult',
            'grace_minutes', 'dip_pct', 'max_wait_minutes',
            'vol_filter_pct']


def build_seed_groups(df):
    df_indiv = df[df['is_ensemble'] == False].copy()
    sa = df_indiv.groupby(GRP_KEYS).agg(
        n_seeds=('seed', 'nunique'),
        seeds=('seed', lambda x: sorted(x.unique().tolist())),
        n_pos_seeds_18=('val_a_avg_pnl', lambda x: (x >= 0.018).sum()),
        n_pos_seeds_15=('val_a_avg_pnl', lambda x: (x >= 0.015).sum()),
        mean_va_pnl=('val_a_avg_pnl', 'mean'),
        std_va_pnl=('val_a_avg_pnl', 'std'),
        min_va_pnl=('val_a_avg_pnl', 'min'),
        max_va_pnl=('val_a_avg_pnl', 'max'),
        mean_va_wr=('val_a_win_rate', 'mean'),
        mean_va_count=('val_a_count', 'mean'),
        mean_va_freq=('val_a_freq', 'mean'),
        mean_vb_pnl=('val_b_avg_pnl', 'mean'),
        mean_vb_wr=('val_b_win_rate', 'mean'),
        mean_vb_count=('val_b_count', 'mean'),
        mean_opt_pnl=('opt_avg_pnl', 'mean'),
        mean_opt_wr=('opt_win_rate', 'mean'),
        mean_opt_count=('opt_count', 'mean'),
        vix_threshold=('vix_threshold', 'first'),
        tlt_crash=('tlt_crash', 'first'),
        spy_ignore=('spy_ignore', 'first'),
    ).reset_index()
    sa['agree_18'] = (sa['n_pos_seeds_18'] / sa['n_seeds']).clip(upper=1.0)
    sa['agree_15'] = (sa['n_pos_seeds_15'] / sa['n_seeds']).clip(upper=1.0)
    return sa


# ==============================================================================
# MATCH DEPLOYED CONFIGS TO BQ
# ==============================================================================
def match_deployed_to_bq(df, sa):
    print(f"\n{'='*W}")
    print(f"  LEGACY HARDCODED DEPLOY — 20 tickers, 4 experiments")
    print(f"{'='*W}")

    matched = []
    unmatched = []

    for ticker, base_exp, mode, seed, is_ens in DEPLOYED_HARDCODED:
        mode_sfx = MODE_SUFFIX.get(mode, '')
        experiment = base_exp + mode_sfx if mode_sfx else base_exp

        if is_ens:
            sg = sa[(sa['ticker'] == ticker) &
                    (sa['base_experiment'] == base_exp) &
                    (sa['tpsl_mode'] == mode)]
            if len(sg) > 0:
                best = sg.loc[sg['mean_va_pnl'].idxmax()]
                matched.append({
                    'ticker': ticker, 'base_exp': base_exp, 'experiment': experiment,
                    'mode': mode, 'seed': 'ENS', 'n_seeds': int(best['n_seeds']),
                    'agree_18': float(best['agree_18']),
                    'va_pnl': float(best['mean_va_pnl']),
                    'va_wr': float(best['mean_va_wr']),
                    'va_count': float(best['mean_va_count']),
                    'va_freq': float(best['mean_va_freq']),
                    'vb_pnl': float(best['mean_vb_pnl']) if not np.isnan(best['mean_vb_pnl']) else None,
                    'vb_wr': float(best['mean_vb_wr']) if not np.isnan(best['mean_vb_wr']) else None,
                    'vb_count': float(best['mean_vb_count']) if not np.isnan(best['mean_vb_count']) else None,
                    'opt_pnl': float(best['mean_opt_pnl']),
                    'opt_wr': float(best['mean_opt_wr']),
                    'opt_count': float(best['mean_opt_count']),
                    'entry_min': int(best['entry_minute']),
                    'prob': float(best['prob_thresh']),
                    'source': 'seed_group',
                })
            else:
                ens_rows = df[(df['ticker'] == ticker) &
                              (df['base_experiment'] == base_exp) &
                              (df['tpsl_mode'] == mode) &
                              (df['is_ensemble'] == True)]
                if len(ens_rows) > 0:
                    best = ens_rows.loc[ens_rows['val_a_avg_pnl'].idxmax()]
                    matched.append({
                        'ticker': ticker, 'base_exp': base_exp, 'experiment': experiment,
                        'mode': mode, 'seed': 'ENS', 'n_seeds': 15,
                        'agree_18': np.nan,
                        'va_pnl': float(best['val_a_avg_pnl']),
                        'va_wr': float(best['val_a_win_rate']),
                        'va_count': float(best['val_a_count']),
                        'va_freq': float(best['val_a_freq']),
                        'vb_pnl': float(best['val_b_avg_pnl']) if not np.isnan(best['val_b_avg_pnl']) else None,
                        'vb_wr': float(best['val_b_win_rate']) if not np.isnan(best['val_b_win_rate']) else None,
                        'vb_count': float(best['val_b_count']) if not np.isnan(best['val_b_count']) else None,
                        'opt_pnl': float(best['opt_avg_pnl']),
                        'opt_wr': float(best['opt_win_rate']),
                        'opt_count': float(best['opt_count']),
                        'entry_min': int(best['entry_minute']),
                        'prob': float(best['prob_thresh']),
                        'source': 'ensemble_row',
                    })
                else:
                    unmatched.append((ticker, base_exp, mode, 'ENS'))
        else:
            row = df[(df['ticker'] == ticker) &
                     (df['base_experiment'] == base_exp) &
                     (df['tpsl_mode'] == mode) &
                     (df['seed'] == seed)]
            if len(row) > 0:
                best = row.iloc[0]
                matched.append({
                    'ticker': ticker, 'base_exp': base_exp, 'experiment': experiment,
                    'mode': mode, 'seed': str(seed), 'n_seeds': 1,
                    'agree_18': np.nan,
                    'va_pnl': float(best['val_a_avg_pnl']),
                    'va_wr': float(best['val_a_win_rate']),
                    'va_count': float(best['val_a_count']),
                    'va_freq': float(best['val_a_freq']),
                    'vb_pnl': float(best['val_b_avg_pnl']) if not np.isnan(best['val_b_avg_pnl']) else None,
                    'vb_wr': float(best['val_b_win_rate']) if not np.isnan(best['val_b_win_rate']) else None,
                    'vb_count': float(best['val_b_count']) if not np.isnan(best['val_b_count']) else None,
                    'opt_pnl': float(best['opt_avg_pnl']),
                    'opt_wr': float(best['opt_win_rate']),
                    'opt_count': float(best['opt_count']),
                    'entry_min': int(best['entry_minute']),
                    'prob': float(best['prob_thresh']),
                    'source': 'seed_row',
                })
            else:
                any_mode = df[(df['ticker'] == ticker) &
                              (df['base_experiment'] == base_exp) &
                              (df['tpsl_mode'] == mode)]
                if len(any_mode) > 0:
                    best = any_mode.loc[any_mode['val_a_avg_pnl'].idxmax()]
                    matched.append({
                        'ticker': ticker, 'base_exp': base_exp, 'experiment': experiment,
                        'mode': mode, 'seed': f"~{int(best['seed'])}", 'n_seeds': 1,
                        'agree_18': np.nan,
                        'va_pnl': float(best['val_a_avg_pnl']),
                        'va_wr': float(best['val_a_win_rate']),
                        'va_count': float(best['val_a_count']),
                        'va_freq': float(best['val_a_freq']),
                        'vb_pnl': float(best['val_b_avg_pnl']) if not np.isnan(best['val_b_avg_pnl']) else None,
                        'vb_wr': float(best['val_b_win_rate']) if not np.isnan(best['val_b_win_rate']) else None,
                        'vb_count': float(best['val_b_count']) if not np.isnan(best['val_b_count']) else None,
                        'opt_pnl': float(best['opt_avg_pnl']),
                        'opt_wr': float(best['opt_win_rate']),
                        'opt_count': float(best['opt_count']),
                        'entry_min': int(best['entry_minute']),
                        'prob': float(best['prob_thresh']),
                        'source': 'nearest_seed',
                    })
                else:
                    unmatched.append((ticker, base_exp, mode, str(seed)))

    return matched, unmatched


# ==============================================================================
# BQ-DRIVEN DEPLOY (current live)
# ==============================================================================
def compute_bq_deploy(sa):
    print(f"\n{'='*W}")
    print(f"  CURRENT BQ-DRIVEN DEPLOY — limit_entry, both 152+160")
    print(f"{'='*W}")

    sa_filt = sa[sa['base_experiment'].isin(BQ_DEPLOY_EXPERIMENTS)].copy()

    indiv_filt = sa_filt.copy()
    indiv_filt['exp_fam'] = indiv_filt['base_experiment'].str[:3]
    qualified = indiv_filt[indiv_filt['mean_va_pnl'] >= DEPLOY_THRESHOLD]

    both_exp = set()
    for tk in qualified['ticker'].unique():
        tk_exps = set(qualified[qualified['ticker'] == tk]['exp_fam'].unique())
        if '152' in tk_exps and '160' in tk_exps:
            both_exp.add(tk)

    sa_limit = sa_filt[(sa_filt['n_seeds'] >= 3) &
                       (sa_filt['tpsl_mode'] == 'limit_entry') &
                       (sa_filt['ticker'].isin(both_exp))]
    sa_limit = sa_limit.copy()
    sa_limit['score'] = (sa_limit['mean_va_pnl'] ** 2) * sa_limit['agree_18']

    if sa_limit.empty:
        print("    No qualifying configs found")
        return []

    best = sa_limit.loc[sa_limit.groupby('ticker')['score'].idxmax()]
    best = best[best['mean_va_pnl'] >= DEPLOY_THRESHOLD].copy()

    configs = []
    for _, row in best.iterrows():
        configs.append({
            'ticker': row['ticker'],
            'base_exp': row['base_experiment'],
            'experiment': row['base_experiment'] + '_LIMIT',
            'mode': 'limit_entry',
            'seed': 'ENS',
            'n_seeds': int(row['n_seeds']),
            'agree_18': float(row['agree_18']),
            'va_pnl': float(row['mean_va_pnl']),
            'va_wr': float(row['mean_va_wr']),
            'va_count': float(row['mean_va_count']),
            'va_freq': float(row['mean_va_freq']),
            'vb_pnl': float(row['mean_vb_pnl']) if not np.isnan(row['mean_vb_pnl']) else None,
            'vb_wr': float(row['mean_vb_wr']) if not np.isnan(row['mean_vb_wr']) else None,
            'vb_count': float(row['mean_vb_count']) if not np.isnan(row['mean_vb_count']) else None,
            'opt_pnl': float(row['mean_opt_pnl']),
            'opt_wr': float(row['mean_opt_wr']),
            'opt_count': float(row['mean_opt_count']),
            'entry_min': int(row['entry_minute']),
            'prob': float(row['prob_thresh']),
            'dip_pct': float(row['dip_pct']),
            'max_wait': int(row['max_wait_minutes']),
            'source': 'bq_deploy',
        })

    return configs


# ==============================================================================
# BEST AVAILABLE PER TICKER (all modes, all experiments)
# ==============================================================================
def compute_best_available(sa):
    print(f"\n{'='*W}")
    print(f"  BEST AVAILABLE — all experiments, all modes, per ticker")
    print(f"{'='*W}")

    sa_filt = sa[sa['n_seeds'] >= 3].copy()
    sa_filt['score'] = (sa_filt['mean_va_pnl'] ** 2) * sa_filt['agree_18']

    if sa_filt.empty:
        return []

    best = sa_filt.loc[sa_filt.groupby('ticker')['score'].idxmax()]
    best = best[best['mean_va_pnl'] >= DEPLOY_THRESHOLD].copy()

    configs = []
    for _, row in best.iterrows():
        mode_sfx = MODE_SUFFIX.get(row['tpsl_mode'], '')
        exp_name = row['base_experiment'] + mode_sfx if mode_sfx else row['base_experiment']
        configs.append({
            'ticker': row['ticker'],
            'base_exp': row['base_experiment'],
            'experiment': exp_name,
            'mode': row['tpsl_mode'],
            'seed': 'ENS',
            'n_seeds': int(row['n_seeds']),
            'agree_18': float(row['agree_18']),
            'va_pnl': float(row['mean_va_pnl']),
            'va_wr': float(row['mean_va_wr']),
            'va_count': float(row['mean_va_count']),
            'va_freq': float(row['mean_va_freq']),
            'vb_pnl': float(row['mean_vb_pnl']) if not np.isnan(row['mean_vb_pnl']) else None,
            'vb_wr': float(row['mean_vb_wr']) if not np.isnan(row['mean_vb_wr']) else None,
            'vb_count': float(row['mean_vb_count']) if not np.isnan(row['mean_vb_count']) else None,
            'opt_pnl': float(row['mean_opt_pnl']),
            'opt_wr': float(row['mean_opt_wr']),
            'opt_count': float(row['mean_opt_count']),
            'entry_min': int(row['entry_minute']),
            'prob': float(row['prob_thresh']),
            'source': 'best_available',
        })

    return configs


# ==============================================================================
# PRINT TABLE
# ==============================================================================
def print_config_table(configs, title):
    if not configs:
        print(f"  No configs for {title}")
        return

    print(f"\n  {'Ticker':<7} {'Experiment':<20} {'Mode':<14} {'Seed':<7} "
          f"{'VA_PnL':>7} {'VA_WR':>5} {'VA_N':>4} {'VA_F':>5} "
          f"{'VB_PnL':>7} {'VB_WR':>5} {'VB_N':>4} "
          f"{'Agr18':>5} {'#S':>3} "
          f"{'OPT_P':>6} {'OPT_W':>5}")
    print(f"  {'-'*(W-2)}")

    sorted_cfgs = sorted(configs, key=lambda c: c['va_pnl'], reverse=True)
    va_pnls = []; va_wrs = []; va_counts = []; vb_pnls = []

    for c in sorted_cfgs:
        vb_pnl_s = f"{c['vb_pnl']:>+6.2%}" if c['vb_pnl'] is not None else f"{'---':>7}"
        vb_wr_s = f"{c['vb_wr']:>4.0%}" if c['vb_wr'] is not None else f"{'---':>5}"
        vb_n_s = f"{c['vb_count']:>4.0f}" if c['vb_count'] is not None else f"{'---':>4}"
        agr_s = f"{c['agree_18']:>4.0%}" if not np.isnan(c.get('agree_18', np.nan)) else f"{'---':>5}"

        deploy_mark = '*' if c['va_pnl'] >= DEPLOY_THRESHOLD else ' '
        print(f" {deploy_mark}{c['ticker']:<7} {c['experiment']:<20} {c['mode']:<14} {c['seed']:<7} "
              f"{c['va_pnl']:>+6.2%} {c['va_wr']:>4.0%} {c['va_count']:>4.0f} {c['va_freq']:>4.1f} "
              f"{vb_pnl_s} {vb_wr_s} {vb_n_s} "
              f"{agr_s} {c['n_seeds']:>3} "
              f"{c['opt_pnl']:>+5.2%} {c['opt_wr']:>4.0%}")

        va_pnls.append(c['va_pnl'])
        va_wrs.append(c['va_wr'])
        va_counts.append(c['va_count'])
        if c['vb_pnl'] is not None:
            vb_pnls.append(c['vb_pnl'])

    print(f"  {'-'*(W-2)}")
    n = len(sorted_cfgs)
    qual = sum(1 for p in va_pnls if p >= DEPLOY_THRESHOLD)
    print(f"  PORTFOLIO ({n} tickers, {qual} above {DEPLOY_THRESHOLD:.1%} threshold):")
    print(f"    val_a:  avg_pnl={np.mean(va_pnls):+.2%}  "
          f"avg_wr={np.mean(va_wrs):.0%}  "
          f"avg_trades={np.mean(va_counts):.1f}  "
          f"total_trades={np.sum(va_counts):.0f}")
    if vb_pnls:
        print(f"    val_b:  avg_pnl={np.mean(vb_pnls):+.2%}  "
              f"({len(vb_pnls)}/{n} tickers with val_b data)")
    pos = sum(1 for p in va_pnls if p > 0)
    print(f"    {pos}/{n} tickers positive  |  "
          f"min={min(va_pnls):+.2%}  max={max(va_pnls):+.2%}  "
          f"std={np.std(va_pnls):.2%}")


# ==============================================================================
# PER-SEED DETAIL FOR A CONFIG SET
# ==============================================================================
def print_seed_detail(configs, df, title):
    print(f"\n{'='*W}")
    print(f"  PER-SEED BREAKDOWN — {title}")
    print(f"{'='*W}")

    for c in sorted(configs, key=lambda x: x['ticker']):
        ticker = c['ticker']
        base_exp = c['base_exp']
        mode = c['mode']

        seeds = df[(df['ticker'] == ticker) &
                   (df['base_experiment'] == base_exp) &
                   (df['tpsl_mode'] == mode) &
                   (df['is_ensemble'] == False)]
        if len(seeds) == 0:
            print(f"\n  {ticker} ({base_exp} {mode}): no per-seed data")
            continue

        seeds = seeds.sort_values('seed')
        pos_18 = (seeds['val_a_avg_pnl'] >= 0.018).sum()
        pos_15 = (seeds['val_a_avg_pnl'] >= 0.015).sum()
        pos_0 = (seeds['val_a_avg_pnl'] > 0).sum()
        n = len(seeds)

        print(f"\n  {ticker} ({base_exp} {mode}) — {n} seeds, "
              f"agree≥1.8%: {pos_18}/{n} ({pos_18/n:.0%}), "
              f"agree≥1.5%: {pos_15}/{n}, "
              f"agree>0%: {pos_0}/{n}")
        print(f"    {'Seed':>7} {'VA_PnL':>8} {'VA_WR':>6} {'VA_N':>5} "
              f"{'VB_PnL':>8} {'VB_WR':>6} "
              f"{'OPT_PnL':>8} {'OPT_WR':>6}")

        for _, s in seeds.iterrows():
            vb_p = f"{s['val_b_avg_pnl']:>+7.2%}" if not np.isnan(s['val_b_avg_pnl']) else f"{'---':>8}"
            vb_w = f"{s['val_b_win_rate']:>5.0%}" if not np.isnan(s['val_b_win_rate']) else f"{'---':>6}"
            mark = '>' if s['val_a_avg_pnl'] >= 0.018 else ' '
            print(f"   {mark}{int(s['seed']):>7} {s['val_a_avg_pnl']:>+7.2%} "
                  f"{s['val_a_win_rate']:>5.0%} {s['val_a_count']:>5.0f} "
                  f"{vb_p} {vb_w} "
                  f"{s['opt_avg_pnl']:>+7.2%} {s['opt_win_rate']:>5.0%}")


# ==============================================================================
# HEAD-TO-HEAD: DEPLOYED vs BEST AVAILABLE
# ==============================================================================
def print_head_to_head(deployed_cfgs, best_cfgs):
    print(f"\n{'='*W}")
    print(f"  HEAD-TO-HEAD: Deployed vs Best Available per Ticker")
    print(f"{'='*W}")

    deployed_map = {c['ticker']: c for c in deployed_cfgs}
    best_map = {c['ticker']: c for c in best_cfgs}

    all_tickers = sorted(set(list(deployed_map.keys()) + list(best_map.keys())))

    print(f"  {'Ticker':<7} {'Deploy Mode':<14} {'D_PnL':>7} {'D_WR':>5} "
          f"{'Best Mode':<14} {'B_PnL':>7} {'B_WR':>5} {'B_Exp':<15} {'Delta':>7} {'Better':>7}")
    print(f"  {'-'*(W-2)}")

    better_deployed = 0
    better_best = 0

    for ticker in all_tickers:
        d = deployed_map.get(ticker)
        b = best_map.get(ticker)

        if d and b:
            delta = b['va_pnl'] - d['va_pnl']
            better = 'BEST' if delta > 0.001 else ('SAME' if abs(delta) <= 0.001 else 'DEPLOY')
            if better == 'BEST': better_best += 1
            elif better == 'DEPLOY': better_deployed += 1
            print(f"  {ticker:<7} {d['mode']:<14} {d['va_pnl']:>+6.2%} {d['va_wr']:>4.0%} "
                  f"{b['mode']:<14} {b['va_pnl']:>+6.2%} {b['va_wr']:>4.0%} {b['base_exp']:<15} "
                  f"{delta:>+6.2%} {better:>7}")
        elif d:
            print(f"  {ticker:<7} {d['mode']:<14} {d['va_pnl']:>+6.2%} {d['va_wr']:>4.0%} "
                  f"{'---':<14} {'---':>7} {'---':>5} {'---':<15} {'---':>7} {'ONLY_D':>7}")
        elif b:
            print(f"  {ticker:<7} {'---':<14} {'---':>7} {'---':>5} "
                  f"{b['mode']:<14} {b['va_pnl']:>+6.2%} {b['va_wr']:>4.0%} {b['base_exp']:<15} "
                  f"{'---':>7} {'NEW':>7}")

    print(f"\n  Deploy wins: {better_deployed}  |  Best available wins: {better_best}  |  "
          f"Only in deploy: {sum(1 for t in all_tickers if t in deployed_map and t not in best_map)}  |  "
          f"New candidates: {sum(1 for t in all_tickers if t not in deployed_map and t in best_map)}")


# ==============================================================================
# MODE DISTRIBUTION
# ==============================================================================
def print_mode_summary(configs, title):
    if not configs:
        return
    print(f"\n  MODE DISTRIBUTION — {title}:")
    mode_counts = {}
    mode_pnls = {}
    for c in configs:
        m = c['mode']
        mode_counts[m] = mode_counts.get(m, 0) + 1
        mode_pnls.setdefault(m, []).append(c['va_pnl'])
    for m in sorted(mode_counts, key=lambda x: mode_counts[x], reverse=True):
        avg = np.mean(mode_pnls[m])
        print(f"    {m:<16} {mode_counts[m]:>3} tickers  avg_va_pnl={avg:+.2%}")

    exp_counts = {}
    for c in configs:
        e = c['base_exp'][:3]
        exp_counts[e] = exp_counts.get(e, 0) + 1
    print(f"\n  EXPERIMENT FAMILY:")
    for e in sorted(exp_counts):
        label = {'152': '152 Late', '160': '160 Early', '170': '170-175 WF',
                 '175': '175 Fold', '180': '180-185 WF', '185': '185 Fold'}.get(e, e)
        print(f"    {label:<16} {exp_counts[e]:>3} tickers")


# ==============================================================================
# CHART
# ==============================================================================
def make_chart(deployed_cfgs, bq_cfgs, best_cfgs):
    fig, axes = plt.subplots(1, 2, figsize=(18, max(8, len(set(
        [c['ticker'] for c in deployed_cfgs] +
        [c['ticker'] for c in best_cfgs]
    )) * 0.35)))
    fig.suptitle(f'WF Optimization Metrics — Deployed vs Best Available\n'
                 f'model_version={MODEL_VERSION}  |  threshold={DEPLOY_THRESHOLD:.1%}',
                 fontsize=13, fontweight='bold')

    # Left: val_a PnL comparison
    ax1 = axes[0]
    d_map = {c['ticker']: c['va_pnl'] for c in deployed_cfgs}
    b_map = {c['ticker']: c['va_pnl'] for c in best_cfgs}
    bq_map = {c['ticker']: c['va_pnl'] for c in bq_cfgs}
    all_t = sorted(set(list(d_map.keys()) + list(b_map.keys())))

    y = np.arange(len(all_t))
    d_vals = [d_map.get(t, 0) * 100 for t in all_t]
    b_vals = [b_map.get(t, 0) * 100 for t in all_t]

    ax1.barh(y + 0.15, d_vals, 0.3, label='Legacy Deploy', color='#9E9E9E', alpha=0.85)
    ax1.barh(y - 0.15, b_vals, 0.3, label='Best Available', color='#FF9800', alpha=0.85)
    ax1.axvline(x=DEPLOY_THRESHOLD * 100, color='red', linewidth=1, linestyle='--',
                label=f'Threshold ({DEPLOY_THRESHOLD:.1%})')
    ax1.set_yticks(y)
    ax1.set_yticklabels(all_t)
    ax1.set_xlabel('Val_A Avg PnL (%)')
    ax1.set_title('WF Val_A Performance')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, axis='x')

    # Right: val_a vs val_b scatter (best available)
    ax2 = axes[1]
    va_vals = []; vb_vals = []; labels = []
    for c in best_cfgs:
        if c['vb_pnl'] is not None:
            va_vals.append(c['va_pnl'] * 100)
            vb_vals.append(c['vb_pnl'] * 100)
            labels.append(c['ticker'])
    if va_vals:
        ax2.scatter(va_vals, vb_vals, c='#2196F3', s=50, alpha=0.7)
        for i, lbl in enumerate(labels):
            ax2.annotate(lbl, (va_vals[i], vb_vals[i]), fontsize=7,
                         ha='left', va='bottom')
        lim = max(max(abs(v) for v in va_vals + vb_vals) * 1.2, 3)
        ax2.plot([-lim, lim], [-lim, lim], 'k--', alpha=0.3, linewidth=0.5)
        ax2.axhline(y=0, color='gray', linewidth=0.5)
        ax2.axvline(x=0, color='gray', linewidth=0.5)
        ax2.set_xlabel('Val_A Avg PnL (%)')
        ax2.set_ylabel('Val_B Avg PnL (%)')
        ax2.set_title('Val_A vs Val_B (out-of-sample)')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No Val_B data', transform=ax2.transAxes,
                 ha='center', fontsize=14, color='gray')

    plt.tight_layout()
    path = '/tmp/deployed_wf_metrics.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"\n  Chart saved: {path}")
    try:
        from IPython.display import display, Image
        display(Image(filename=path))
    except ImportError:
        plt.show()


# ==============================================================================
# RUN
# ==============================================================================
print(f"\n{'='*W}")
print(f"  DEPLOYED WF METRICS (no backtest)")
print(f"  Model: {MODEL_VERSION}  |  Threshold: {DEPLOY_THRESHOLD:.1%}")
print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"{'='*W}")

df = load_all_wf_data()
sa = build_seed_groups(df)

# --- Section 1: Legacy 20-ticker deploy ---
matched, unmatched = match_deployed_to_bq(df, sa)
print_config_table(matched, "LEGACY HARDCODED DEPLOY")
if unmatched:
    print(f"\n  UNMATCHED ({len(unmatched)}):")
    for ticker, exp, mode, seed in unmatched:
        print(f"    {ticker:<7} {exp:<20} {mode:<14} {seed}")
print_mode_summary(matched, "Legacy Deploy")

# --- Section 2: Current BQ-driven deploy ---
bq_configs = compute_bq_deploy(sa)
print_config_table(bq_configs, "CURRENT BQ-DRIVEN DEPLOY")

# --- Section 3: Best available across all experiments ---
best_configs = compute_best_available(sa)
print_config_table(best_configs, "BEST AVAILABLE")
print_mode_summary(best_configs, "Best Available")

# --- Section 4: Head-to-head ---
print_head_to_head(matched, best_configs)

# --- Section 5: Per-seed detail for legacy deploy ---
print_seed_detail(matched, df, "Legacy Deploy")

# --- Section 6: Charts ---
make_chart(matched, bq_configs, best_configs)

# --- Section 7: Val_A vs Val_B decay ---
print(f"\n{'='*W}")
print(f"  VAL_A -> VAL_B DECAY ANALYSIS")
print(f"{'='*W}")
decay_data = []
for c in best_configs:
    if c['vb_pnl'] is not None and c['va_pnl'] > 0:
        decay = (c['vb_pnl'] - c['va_pnl']) / c['va_pnl']
        decay_data.append({
            'ticker': c['ticker'], 'mode': c['mode'], 'base_exp': c['base_exp'],
            'va_pnl': c['va_pnl'], 'vb_pnl': c['vb_pnl'], 'decay_pct': decay,
        })
if decay_data:
    ddf = pd.DataFrame(decay_data).sort_values('decay_pct')
    print(f"  {'Ticker':<7} {'Mode':<14} {'Exp':<15} {'VA_PnL':>7} {'VB_PnL':>7} {'Decay':>7}")
    print(f"  {'-'*62}")
    for _, r in ddf.iterrows():
        print(f"  {r['ticker']:<7} {r['mode']:<14} {r['base_exp']:<15} "
              f"{r['va_pnl']:>+6.2%} {r['vb_pnl']:>+6.2%} {r['decay_pct']:>+6.0%}")
    print(f"\n  Average decay: {ddf['decay_pct'].mean():+.0%}  "
          f"Median: {ddf['decay_pct'].median():+.0%}  "
          f"Survived (VB>0): {(ddf['vb_pnl'] > 0).sum()}/{len(ddf)}")
else:
    print("  No val_b data available for decay analysis")

print(f"\n{'='*W}")
print(f"  DONE")
print(f"{'='*W}")
