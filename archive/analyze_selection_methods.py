"""
analyze_selection_methods.py — per-ticker best-mode selection backtest
=====================================================================
Compares per-ticker best-mode selection against limit-only strategies:
  A: DEPLOYED (exact configs from daily_score_combined.py — 20 tickers, 4 experiments)
  B: Tier 1+ (limit-only + both 152+160 experiments)
  E: Best mode per ticker (unconstrained, all 9 TPSL modes)
  F: Best mode per ticker with June safety (losers fall back to limit_entry)

CSV-scored, val_a period configs. Colab-ready.
"""

import os
import io
import gc
import sys
import subprocess
import warnings
from datetime import datetime, timedelta

os.environ['TF_USE_LEGACY_KERAS'] = '1'

import numpy as np
import pandas as pd
import pytz
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings('ignore')

try:
    import tf_keras
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tf_keras", "--quiet"])

try:
    import tensorflow as tf
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "tensorflow[and-cuda]", "tf_keras", "pandas_ta",
                           "google-cloud-storage", "pyarrow", "gcsfs",
                           "numba", "--quiet"])
    import tensorflow as tf

try:
    import pandas_ta as ta
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "pandas_ta", "--no-deps", "--quiet"])
    import pandas_ta as ta

from numba import jit
from google.cloud import storage as gcs_storage, bigquery

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass

# ==============================================================================
# CONFIG
# ==============================================================================
MODEL_VERSION = 'robust_wf_v12'
MIN_TRADES = 3
DEPLOY_THRESHOLD = 0.018

TEST_START = '2026-04-01'
TEST_END   = '2026-06-15'

STARTING_CAPITAL = 200_000

DEPLOY_EXPERIMENTS = {'152_XSECTOR', '160_XSECTOR', '170_XSECTOR',
                      '175_XSECTOR', '180_XSECTOR', '185_XSECTOR'}
EXCLUDE_TICKERS = set()

BUCKET_NAME    = os.environ.get("GCS_BUCKET", "")
BQ_TABLE       = f"{GCP_PROJECT}.scoring_logs.wf_optimization_results"
INFERENCE_DIR  = "Data/Inference_v2"

EXIT_TIME_MIN = 780
PST = pytz.timezone('America/Los_Angeles')
W = 130

TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}
TICKER_RENAME   = {'MIGI': 'BGDE', 'BITF': 'KEEL'}
SLIPPAGE_PCT = 0.0015
VOL_LOOKBACK_DAYS = 30

MICRO_COLS = ['Spread', 'OBI', 'Volume_Delta', 'Trade_Count', 'Delta_Pct',
              'bsize', 'asize', 'Total_Vol']

REGIME_TP_SCALE = np.array([0.70, 0.85, 1.00, 1.30, 1.60])
REGIME_SL_SCALE = np.array([0.60, 0.80, 1.00, 1.30, 1.80])

MODE_SUFFIX = {'static': '', 'atr': '_ATR', 'atr_adaptive': '_ATRADP',
               'regime_scaled': '_REGIME', 'trailing_decay': '_TRAIL',
               'regime_adaptive': '_REGADP', 'atr_sl': '_ATRSL',
               'grace': '_GRACE', 'limit_entry': '_LIMIT'}

TRAIN_TEACHERS = [
    'BBAI', 'RCAT', 'LUNR', 'ASTS', 'ACHR', 'MVIS',
    'WULF', 'CIFR', 'IREN', 'KEEL', 'HUT', 'CLSK', 'APLD',
]

METHOD_LABELS = {
    'A': 'deployed',
    'B': 'tier1_plus',
    'E': 'best_mode',
    'F': 'best_safe',
}
METHOD_COLORS = {
    'A': '#9E9E9E',
    'B': '#E91E63',
    'E': '#FF9800',
    'F': '#4CAF50',
}

gcs_client = gcs_storage.Client()
bq_client  = bigquery.Client()


# ==============================================================================
# LOAD RAW BQ DATA
# ==============================================================================
def load_raw_wf_data():
    print(f"  Loading WF results from BQ (model_version={MODEL_VERSION})...")
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
    print(f"    {len(df)} rows loaded from BQ")

    df['ticker'] = df['ticker'].replace(TICKER_RENAME)

    all_suffixes = list(MODE_SUFFIX.values())
    def _base_exp(exp_name):
        for sfx in sorted(all_suffixes, key=len, reverse=True):
            if sfx and exp_name.endswith(sfx):
                return exp_name[:-len(sfx)]
        return exp_name

    df['base_experiment'] = df['experiment'].apply(_base_exp)
    df = df[df['base_experiment'].isin(DEPLOY_EXPERIMENTS)]
    df = df[~df['ticker'].isin(EXCLUDE_TICKERS)]
    print(f"    {len(df)} rows after experiment/ticker filter")

    for col in ['grace_minutes', 'max_wait_minutes', 'vol_filter_pct']:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0).astype(int)
    for col in ['dip_pct', 'tp_mult', 'sl_mult', 'trail_start_mult', 'trail_end_mult']:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(0.0)

    MAX_VA_OPT_RATIO = 3.0
    before_n = len(df)
    df = df[df['opt_avg_pnl'] > 0].copy()
    va_opt_ratio = df['val_a_avg_pnl'] / df['opt_avg_pnl']
    df = df[va_opt_ratio <= MAX_VA_OPT_RATIO]
    print(f"    Overfit filter: {before_n} -> {len(df)} configs")

    return df


# ==============================================================================
# ROW TO CONFIG
# ==============================================================================
def _row_to_config(row, wf_avg_pnl_col='val_a_avg_pnl', wf_wr_col='val_a_win_rate',
                   wf_count_col='val_a_count', wf_freq_col='val_a_freq',
                   composite_val=0.0, extra_fields=None):
    exp = row['experiment']
    mode = row['tpsl_mode']
    mode_sfx = MODE_SUFFIX.get(mode, '')
    base_exp = exp[:-len(mode_sfx)] if mode_sfx and exp.endswith(mode_sfx) else exp

    is_ens = bool(row.get('is_ensemble', True))
    seed_val = int(row.get('seed', -1))
    if is_ens or seed_val == -1:
        base_num = int(base_exp.split('_')[0])
        seed_list = list(range(base_num * 100 + 1, base_num * 100 + 16))
        is_ens = True
        seed_val = -1
    else:
        seed_list = [seed_val]

    cfg = {
        'ticker': row['ticker'],
        'base_experiment': base_exp,
        'experiment': exp,
        'tpsl_mode': mode,
        'seed': seed_val,
        'is_ensemble': is_ens,
        'seed_list': seed_list,
        'entry_minute': int(row['entry_minute']),
        'prob_thresh': float(row['prob_thresh']),
        'tp_pct': float(row.get('tp_pct') or 0),
        'sl_pct': float(row.get('sl_pct') or 0),
        'tp_mult': float(row.get('tp_mult') or 3.0),
        'sl_mult': float(row.get('sl_mult') or 1.5),
        'trail_start_mult': float(row.get('trail_start_mult') or 4.0),
        'trail_end_mult': float(row.get('trail_end_mult') or 1.0),
        'vix_threshold': float(row.get('vix_threshold') or 99),
        'tlt_crash': float(row.get('tlt_crash') or -99),
        'spy_ignore': float(row.get('spy_ignore') or 99),
        'vol_filter_pct': int(row.get('vol_filter_pct') or 0),
        'grace_minutes': int(row.get('grace_minutes') or 0),
        'dip_pct': float(row.get('dip_pct') or 0),
        'max_wait_minutes': int(row.get('max_wait_minutes') or 0),
        'model_id': base_exp,
        'stats_id': base_exp,
        'wf_avg_pnl': float(row[wf_avg_pnl_col]),
        'wf_win_rate': float(row[wf_wr_col]),
        'wf_count': int(row[wf_count_col]),
        'wf_freq': float(row.get(wf_freq_col) or 0),
        'wf_composite': composite_val,
    }
    if extra_fields:
        cfg.update(extra_fields)
    return cfg


# ==============================================================================
# SEED GROUP BUILDER
# ==============================================================================
GRP_KEYS = ['ticker', 'experiment', 'tpsl_mode', 'entry_minute',
            'prob_thresh', 'tp_pct', 'sl_pct',
            'tp_mult', 'sl_mult', 'trail_start_mult', 'trail_end_mult',
            'grace_minutes', 'dip_pct', 'max_wait_minutes',
            'vol_filter_pct']


def _build_seed_groups(df):
    df_indiv = df[df['is_ensemble'] == False].copy()
    extra_agg = {}
    for c in ['vix_threshold','tlt_crash','spy_ignore']:
        if c in df_indiv.columns:
            extra_agg[c] = (c, 'first')
    sa = df_indiv.groupby(GRP_KEYS).agg(
        n_seeds=('seed', 'nunique'),
        n_pos_seeds_18=('val_a_avg_pnl', lambda x: (x >= 0.018).sum()),
        n_pos_seeds_15=('val_a_avg_pnl', lambda x: (x >= 0.015).sum()),
        mean_va_pnl=('val_a_avg_pnl', 'mean'),
        mean_opt_pnl=('opt_avg_pnl', 'mean'),
        mean_va_wr=('val_a_win_rate', 'mean'),
        mean_va_count=('val_a_count', 'mean'),
        mean_va_freq=('val_a_freq', 'mean'),
        mean_opt_wr=('opt_win_rate', 'mean'),
        **extra_agg,
    ).reset_index()
    sa['agree_18'] = (sa['n_pos_seeds_18'] / sa['n_seeds']).clip(upper=1.0)
    sa['agree_15'] = (sa['n_pos_seeds_15'] / sa['n_seeds']).clip(upper=1.0)
    return sa


def _sa_to_configs(best_df, agree_col='agree_18'):
    configs = []
    for _, row in best_df.iterrows():
        cfg = _row_to_config(
            row,
            wf_avg_pnl_col='mean_va_pnl',
            wf_wr_col='mean_va_wr',
            wf_count_col='mean_va_count',
            wf_freq_col='mean_va_freq',
            composite_val=float(row.get('score', 0)),
            extra_fields={
                'seed': -1, 'is_ensemble': True,
                'seed_agree_pct': float(row.get(agree_col, 0)),
                'n_seeds': int(row['n_seeds']),
                'tier': row.get('tier', ''),
            }
        )
        configs.append(cfg)
    return configs


# ==============================================================================
# EXPERIMENT DIVERSITY — which tickers have both 152 + 160?
# ==============================================================================
def find_both_exp_tickers(df, threshold=DEPLOY_THRESHOLD):
    indiv = df[df['is_ensemble'] == False].copy()
    indiv['exp_fam'] = indiv['experiment'].str[:3]
    survived = indiv[indiv['val_a_avg_pnl'] >= threshold]

    both_exp = set()
    for tk in survived['ticker'].unique():
        tk_exps = survived[survived['ticker'] == tk]['exp_fam'].unique()
        if '152' in tk_exps and '160' in tk_exps:
            both_exp.add(tk)

    print(f"    Tickers with BOTH 152+160 (>= {threshold:.1%}): {len(both_exp)}")
    print(f"      {sorted(both_exp)}")
    return both_exp


# ==============================================================================
# 4 SELECTION STRATEGIES
# ==============================================================================
def _deployed_cfg(ticker, model_id, seed, is_ensemble, prob_thresh, entry_minute,
                  tpsl_mode, tp_pct=0, sl_pct=0, tp_mult=3.0, sl_mult=1.5,
                  trail_start_mult=4.0, trail_end_mult=1.0,
                  vix_threshold=999, tlt_crash=-999, spy_ignore=999,
                  vol_filter_pct=0, grace_minutes=0, dip_pct=0, max_wait_minutes=0):
    mode_sfx = MODE_SUFFIX.get(tpsl_mode, '')
    exp_name = model_id + mode_sfx if mode_sfx else model_id
    base_num = int(model_id.split('_')[0])
    seed_list = list(range(base_num*100+1, base_num*100+16)) if is_ensemble else [seed]
    return {
        'ticker': ticker, 'base_experiment': model_id, 'experiment': exp_name,
        'tpsl_mode': tpsl_mode, 'seed': -1 if is_ensemble else seed,
        'is_ensemble': is_ensemble, 'seed_list': seed_list,
        'entry_minute': entry_minute, 'prob_thresh': prob_thresh,
        'tp_pct': tp_pct, 'sl_pct': sl_pct, 'tp_mult': tp_mult, 'sl_mult': sl_mult,
        'trail_start_mult': trail_start_mult, 'trail_end_mult': trail_end_mult,
        'vix_threshold': vix_threshold, 'tlt_crash': tlt_crash, 'spy_ignore': spy_ignore,
        'vol_filter_pct': vol_filter_pct, 'grace_minutes': grace_minutes,
        'dip_pct': dip_pct, 'max_wait_minutes': max_wait_minutes,
        'model_id': model_id, 'stats_id': model_id,
        'wf_avg_pnl': 0, 'wf_win_rate': 0, 'wf_count': 0, 'wf_freq': 0,
        'wf_composite': 0, 'tier': 'deployed',
    }


def select_A_deployed():
    """A: Exact deployed configs from daily_score_combined.py (20 tickers, 4 experiments)."""
    print(f"\n  --- A: DEPLOYED (exact production configs) ---")
    configs = [
        # EXP_152 (model_id='152_XSECTOR') — 2 tickers
        _deployed_cfg('APLD', '152_XSECTOR', 15203, False, 0.55, 370,
                      'atr_adaptive', tp_mult=1.5, sl_mult=1.0,
                      vix_threshold=20, tlt_crash=0, spy_ignore=999),
        _deployed_cfg('IREN', '152_XSECTOR', -1, True, 0.60, 380,
                      'static', tp_pct=0.08, sl_pct=0.03,
                      vix_threshold=20, tlt_crash=0, spy_ignore=0),
        # EXP_160 (model_id='160_XSECTOR') — 5 tickers
        _deployed_cfg('CIFR', '160_XSECTOR', -1, True, 0.50, 375,
                      'static', tp_pct=0.12, sl_pct=0.08,
                      vix_threshold=20, tlt_crash=0, spy_ignore=999, vol_filter_pct=25),
        _deployed_cfg('FCEL', '160_XSECTOR', 16005, False, 0.45, 370,
                      'atr_adaptive', tp_mult=2.5, sl_mult=1.25,
                      vix_threshold=20, tlt_crash=0, spy_ignore=0),
        _deployed_cfg('KEEL', '160_XSECTOR', -1, True, 0.60, 375,
                      'regime_scaled', tp_pct=0.16, sl_pct=0.14,
                      vix_threshold=999, tlt_crash=-999, spy_ignore=999),
        _deployed_cfg('PLUG', '160_XSECTOR', 16015, False, 0.50, 370,
                      'trailing_decay', sl_mult=1.5, trail_start_mult=2.0, trail_end_mult=0.5,
                      vix_threshold=20, tlt_crash=0, spy_ignore=0),
        _deployed_cfg('RCAT', '160_XSECTOR', 16013, False, 0.50, 370,
                      'atr', tp_mult=2.0, sl_mult=2.0,
                      vix_threshold=20, tlt_crash=0, spy_ignore=0),
        # EXP_170 (model_id='175_XSECTOR') — 8 tickers
        _deployed_cfg('ALAB', '175_XSECTOR', 17507, False, 0.45, 360,
                      'static', tp_pct=0.12, sl_pct=0.03,
                      vix_threshold=20, tlt_crash=0, spy_ignore=0),
        _deployed_cfg('ASTS', '175_XSECTOR', -1, True, 0.60, 370,
                      'atr_adaptive', tp_mult=2.0, sl_mult=2.0,
                      vix_threshold=20, tlt_crash=0, spy_ignore=999),
        _deployed_cfg('BE', '175_XSECTOR', 17500, False, 0.55, 360,
                      'static', tp_pct=0.25, sl_pct=0.14,
                      vix_threshold=20, tlt_crash=0, spy_ignore=0),
        _deployed_cfg('BTBT', '175_XSECTOR', 17510, False, 0.50, 360,
                      'atr_adaptive', tp_mult=1.5, sl_mult=1.0,
                      vix_threshold=20, tlt_crash=0, spy_ignore=0),
        _deployed_cfg('CLSK', '175_XSECTOR', 17504, False, 0.45, 380,
                      'regime_scaled', tp_pct=0.06, sl_pct=0.05,
                      vix_threshold=20, tlt_crash=0, spy_ignore=999),
        _deployed_cfg('OKLO', '175_XSECTOR', 17501, False, 0.50, 370,
                      'static', tp_pct=0.12, sl_pct=0.05,
                      vix_threshold=20, tlt_crash=0, spy_ignore=999),
        _deployed_cfg('RKLB', '175_XSECTOR', 17514, False, 0.75, 375,
                      'static', tp_pct=0.08, sl_pct=0.08,
                      vix_threshold=20, tlt_crash=0, spy_ignore=0),
        _deployed_cfg('WULF', '175_XSECTOR', 17511, False, 0.50, 365,
                      'atr_adaptive', tp_mult=1.5, sl_mult=1.0,
                      vix_threshold=20, tlt_crash=0, spy_ignore=999),
        # EXP_185 (model_id='185_XSECTOR') — 4 tickers
        _deployed_cfg('PL', '185_XSECTOR', 18507, False, 0.50, 360,
                      'static', tp_pct=0.06, sl_pct=0.03),
        _deployed_cfg('RDW', '185_XSECTOR', 18503, False, 0.60, 370,
                      'atr', tp_mult=2.0, sl_mult=1.0),
        _deployed_cfg('RIOT', '185_XSECTOR', 18503, False, 0.50, 375,
                      'trailing_decay', sl_mult=2.0, trail_start_mult=2.0, trail_end_mult=1.0),
        _deployed_cfg('SMCI', '185_XSECTOR', -1, True, 0.50, 360,
                      'static', tp_pct=0.06, sl_pct=0.06),
    ]
    print(f"    {len(configs)} deployed configs across 4 experiments:")
    by_exp = {}
    for c in configs:
        by_exp.setdefault(c['model_id'], []).append(c)
    for exp_id, cfgs in sorted(by_exp.items()):
        print(f"    {exp_id} ({len(cfgs)} tickers):")
        for c in cfgs:
            ens_tag = 'ENS' if c['is_ensemble'] else f"s{c['seed']}"
            print(f"      {c['ticker']:<7} {c['tpsl_mode']:<16} {ens_tag:<7} "
                  f"prob={c['prob_thresh']:.2f} entry={c['entry_minute']}m")
    return configs


def select_B_tier1plus(df, both_exp_tickers):
    """B: Tier 1+ — limit-only + both experiments."""
    print(f"\n  --- B: Tier 1+ (limit + both 152+160) ---")
    sa = _build_seed_groups(df)
    sa = sa[(sa['n_seeds'] >= 3) & (sa['tpsl_mode'] == 'limit_entry')]
    sa = sa[sa['ticker'].isin(both_exp_tickers)]
    sa['score'] = (sa['mean_va_pnl'] ** 2) * sa['agree_18']
    best = sa.loc[sa.groupby('ticker')['score'].idxmax()] if not sa.empty else sa
    best = best[best['mean_va_pnl'] >= DEPLOY_THRESHOLD].copy()
    best['tier'] = 'tier1+'
    configs = _sa_to_configs(best)
    print(f"    {len(configs)} DEPLOY configs")
    for c in configs:
        print(f"      {c['ticker']:<7} {c['experiment']:<25} "
              f"pnl={c['wf_avg_pnl']:.2%} agree={c['seed_agree_pct']:.0%} ({c['n_seeds']}s) "
              f"dip={c['dip_pct']:.1%} wait={c['max_wait_minutes']}m")
    return configs


def select_E_best_mode(df):
    """E: Per-ticker best mode — pick the highest-scoring config across all 9 TPSL modes."""
    print(f"\n  --- E: Best Mode per Ticker (all 9 modes, unconstrained) ---")
    sa = _build_seed_groups(df)
    sa = sa[sa['n_seeds'] >= 3]
    sa['score'] = (sa['mean_va_pnl'] ** 2) * sa['agree_18']
    best = sa.loc[sa.groupby('ticker')['score'].idxmax()] if not sa.empty else sa
    best = best[best['mean_va_pnl'] >= DEPLOY_THRESHOLD].copy()
    best['tier'] = 'best_mode'
    configs = _sa_to_configs(best)
    print(f"    {len(configs)} DEPLOY configs")
    mode_counts = {}
    for c in configs:
        mode_counts[c['tpsl_mode']] = mode_counts.get(c['tpsl_mode'], 0) + 1
        print(f"      {c['ticker']:<7} {c['experiment']:<25} {c['tpsl_mode']:<16} "
              f"pnl={c['wf_avg_pnl']:.2%} agree={c['seed_agree_pct']:.0%} ({c['n_seeds']}s)")
    print(f"    Mode distribution: {dict(sorted(mode_counts.items(), key=lambda x: -x[1]))}")
    return configs


def _get_best_limit_config(df, ticker):
    """Find the best limit_entry config for a ticker (for fallback)."""
    sa = _build_seed_groups(df)
    sa = sa[(sa['n_seeds'] >= 3) & (sa['tpsl_mode'] == 'limit_entry') & (sa['ticker'] == ticker)]
    if sa.empty:
        return None
    sa['score'] = (sa['mean_va_pnl'] ** 2) * sa['agree_18']
    best = sa.loc[sa['score'].idxmax()]
    if best['mean_va_pnl'] < DEPLOY_THRESHOLD:
        return None
    best_df = sa.loc[[sa['score'].idxmax()]].copy()
    best_df['tier'] = 'safe_fallback'
    cfgs = _sa_to_configs(best_df)
    return cfgs[0] if cfgs else None


def build_F_from_E(e_configs, e_trades_df, df):
    """F: June-safe — swap non-limit tickers that lost in June to limit_entry."""
    print(f"\n  --- F: Best Mode + June Safety ---")

    june_start = 20260601
    june_end = 20260630

    swap_report = []
    f_configs = []

    for cfg in e_configs:
        ticker = cfg['ticker']
        mode = cfg['tpsl_mode']

        if mode == 'limit_entry':
            cfg_copy = dict(cfg)
            cfg_copy['tier'] = 'best_safe'
            f_configs.append(cfg_copy)
            swap_report.append({'ticker': ticker, 'orig_mode': mode,
                                'june_pnl': None, 'action': 'kept (already limit)'})
            continue

        tk_june = e_trades_df[
            (e_trades_df['ticker'] == ticker) &
            (e_trades_df['date_int'] >= june_start) &
            (e_trades_df['date_int'] <= june_end)
        ]

        june_total = tk_june['pnl'].sum() if len(tk_june) > 0 else 0.0
        has_june_loss = len(tk_june) > 0 and june_total < 0

        if not has_june_loss:
            cfg_copy = dict(cfg)
            cfg_copy['tier'] = 'best_safe'
            f_configs.append(cfg_copy)
            tag = f"kept ({mode}, June PnL={june_total:+.2%})" if len(tk_june) > 0 else f"kept ({mode}, no June trades)"
            swap_report.append({'ticker': ticker, 'orig_mode': mode,
                                'june_pnl': june_total if len(tk_june) > 0 else None,
                                'action': tag})
        else:
            limit_cfg = _get_best_limit_config(df, ticker)
            if limit_cfg:
                limit_cfg['tier'] = 'safe_fallback'
                f_configs.append(limit_cfg)
                swap_report.append({'ticker': ticker, 'orig_mode': mode,
                                    'june_pnl': june_total,
                                    'action': f"-> limit_entry (dip={limit_cfg['dip_pct']:.1%} wait={limit_cfg['max_wait_minutes']}m)"})
            else:
                swap_report.append({'ticker': ticker, 'orig_mode': mode,
                                    'june_pnl': june_total,
                                    'action': 'EXCLUDED (no limit config)'})

    print(f"\n  JUNE SAFETY ANALYSIS")
    print(f"  {'Ticker':<8} {'Best Mode':<18} {'June PnL':>10} {'Action'}")
    print(f"  {'-'*80}")
    for r in sorted(swap_report, key=lambda x: x['ticker']):
        jp = f"{r['june_pnl']:>+9.2%}" if r['june_pnl'] is not None else f"{'n/a':>10}"
        print(f"  {r['ticker']:<8} {r['orig_mode']:<18} {jp} {r['action']}")

    n_kept = sum(1 for r in swap_report if r['action'].startswith('kept'))
    n_swapped = sum(1 for r in swap_report if r['action'].startswith('->'))
    n_excluded = sum(1 for r in swap_report if r['action'].startswith('EXCLUDED'))
    print(f"\n    Summary: {n_kept} kept, {n_swapped} swapped to limit, {n_excluded} excluded")
    print(f"    {len(f_configs)} total F configs")

    return f_configs


# ==============================================================================
# DATA LOADING
# ==============================================================================
_master_cache = {}

def load_master(ticker, start_date=None):
    cache_key = (ticker, start_date)
    if cache_key in _master_cache:
        return _master_cache[cache_key]
    clean = TICKER_FILE_MAP.get(ticker, ticker)
    uri = f"gs://{BUCKET_NAME}/{INFERENCE_DIR}/{clean}_master_inference.parquet"
    try:
        df = pd.read_parquet(uri)
        df = df.reset_index()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        if dt_cols:
            df['Datetime'] = pd.to_datetime(df[dt_cols[0]], utc=True)
        else:
            num_col = next((c for c in df.columns
                           if c in ('Datetime', 'Date', 'Time', 'Timestamp', 'Level_0')),
                          df.columns[0])
            df['Datetime'] = pd.to_datetime(df[num_col], unit='ms', utc=True)
        df = df.set_index('Datetime')
        df = df[~df.index.duplicated(keep='first')]
        std_map = {
            'Total_vol': 'Total_Vol', 'Trade_count': 'Trade_Count',
            'Volume_delta': 'Volume_Delta', 'Delta_pct': 'Delta_Pct',
            'Obi': 'OBI', 'Bsize': 'bsize', 'Asize': 'asize',
            'open': 'Open', 'high': 'High', 'low': 'Low',
            'close': 'Close', 'volume': 'Volume',
        }
        lower_map = {k.lower(): v for k, v in std_map.items()}
        df.columns = [lower_map.get(str(c).strip().lower(), str(c).strip())
                      for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated(keep='first')].copy()
        for c in MICRO_COLS:
            if c not in df.columns:
                df[c] = 0.0
        df = df[df.index.notna()].sort_index()
        if start_date:
            cutoff = pd.Timestamp(start_date, tz='UTC') - pd.Timedelta(days=30)
            df = df[df.index >= cutoff]
        _master_cache[cache_key] = df
        return df
    except Exception as e:
        print(f"    load_master({ticker}): FAILED -- {e}")
        return None


def heal_data(df):
    df = df.resample('1min').asfreq()
    for c in ['Open', 'High', 'Low', 'Close']:
        if c in df.columns:
            df[c] = df[c].ffill()
    for c in ['Volume', 'Total_Vol', 'Volume_Delta', 'Trade_Count', 'Delta_Pct',
              'Spread', 'OBI', 'bsize', 'asize']:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    return df


# ==============================================================================
# CSV SCORING
# ==============================================================================
_csv_score_cache = {}

def load_seed_scores(ticker, experiment):
    cache_key = (ticker, experiment)
    if cache_key in _csv_score_cache:
        return _csv_score_cache[cache_key]
    clean = TICKER_FILE_MAP.get(ticker, ticker)
    results_dir = f"Algos/Results_all_v2/Universal_Exp{experiment}"
    prefix = f"{results_dir}/{clean}/Seed_"
    blobs = list(gcs_client.bucket(BUCKET_NAME).list_blobs(prefix=prefix))
    seed_series = {}
    for blob in blobs:
        name = blob.name.split('/')[-1]
        if not name.startswith('Seed_') or not name.endswith('_Results.csv'):
            continue
        seed_id = int(name.replace('Seed_', '').replace('_Results.csv', ''))
        d = pd.read_csv(io.BytesIO(blob.download_as_bytes()), index_col=0, parse_dates=True)
        if d.index.tz is None:
            d.index = d.index.tz_localize('UTC')
        d.index = d.index.tz_convert(PST)
        seed_series[seed_id] = d['Prob_Spike'].sort_index()
    _csv_score_cache[cache_key] = seed_series
    return seed_series


def score_ticker_from_csv(ticker, cfg, df_ohlcv_pst):
    experiment = cfg['model_id']
    entry_minute = cfg['entry_minute']
    prob_thresh = cfg['prob_thresh']

    all_seeds = load_seed_scores(ticker, experiment)
    if not all_seeds:
        return []

    seed_list = sorted(all_seeds.keys()) if cfg.get('is_ensemble', True) else (
        [cfg['seed']] if cfg['seed'] in all_seeds else sorted(all_seeds.keys())
    )

    idx = df_ohlcv_pst.index
    probs = np.zeros(len(idx), dtype=np.float64)
    n_seeds_used = 0

    for s in seed_list:
        if s not in all_seeds:
            continue
        seed_prob = all_seeds[s]
        common = idx.intersection(seed_prob.index)
        if len(common) == 0:
            continue
        common_locs = idx.get_indexer(common)
        probs[common_locs] += seed_prob.loc[common].values
        n_seeds_used += 1

    if n_seeds_used == 0:
        return []
    probs /= n_seeds_used

    times = (df_ohlcv_pst.index.hour * 60 + df_ohlcv_pst.index.minute).values
    test_start_pst = pd.Timestamp(TEST_START, tz=PST)
    test_end_pst = pd.Timestamp(TEST_END, tz=PST) + pd.Timedelta(days=1)

    results = []
    for i in range(len(idx)):
        if idx[i] < test_start_pst or idx[i] >= test_end_pst:
            continue
        if times[i] != entry_minute:
            continue
        if probs[i] < prob_thresh:
            continue
        results.append({'bar_ts': idx[i], 'prob': float(probs[i])})
    return results


# ==============================================================================
# DYNAMIC TP/SL, ATR, VOLUME FILTER, TRADE SIM, REGIME
# ==============================================================================
def compute_dynamic_tpsl(cfg, atr_pct_val, vix_value):
    mode = cfg['tpsl_mode']
    if mode in ('atr', 'atr_adaptive'):
        tp = min(max(atr_pct_val * cfg['tp_mult'], 0.02), 0.60)
        sl = min(max(atr_pct_val * cfg['sl_mult'], 0.01), 0.25)
        return tp, sl
    elif mode == 'regime_scaled':
        regime_idx = 2
        if vix_value is not None:
            if vix_value <= 15: regime_idx = 0
            elif vix_value <= 20: regime_idx = 1
            elif vix_value <= 25: regime_idx = 2
            elif vix_value <= 35: regime_idx = 3
            else: regime_idx = 4
        tp = cfg['tp_pct'] * REGIME_TP_SCALE[regime_idx]
        sl = cfg['sl_pct'] * REGIME_SL_SCALE[regime_idx]
        return tp, sl
    elif mode == 'trailing_decay':
        sl = min(max(atr_pct_val * cfg['sl_mult'], 0.01), 0.25)
        tp = min(max(atr_pct_val * cfg['trail_start_mult'], 0.02), 0.60)
        return tp, sl
    elif mode == 'atr_sl':
        tp = cfg['tp_pct']
        sl = min(max(atr_pct_val * cfg['sl_mult'], 0.01), 0.25)
        return tp, sl
    elif mode in ('grace', 'limit_entry'):
        return cfg['tp_pct'], cfg['sl_pct']
    return cfg['tp_pct'], cfg['sl_pct']


def compute_daily_atr_series(df_ohlcv, period=14):
    daily = df_ohlcv.resample('D').agg({'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
    prev_close = daily['Close'].shift(1)
    tr = pd.concat([daily['High'] - daily['Low'],
                     (daily['High'] - prev_close).abs(),
                     (daily['Low'] - prev_close).abs()], axis=1).max(axis=1)
    atr_pct = tr.rolling(period).mean() / daily['Close']
    return atr_pct.reindex(df_ohlcv.index, method='ffill')


def compute_adaptive_atr_series(df_ohlcv, short=5, long=22, base=14):
    daily = df_ohlcv.resample('D').agg({'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
    prev_close = daily['Close'].shift(1)
    tr = pd.concat([daily['High'] - daily['Low'],
                     (daily['High'] - prev_close).abs(),
                     (daily['Low'] - prev_close).abs()], axis=1).max(axis=1)
    atr_base = tr.rolling(base).mean()
    accel = (tr.rolling(short).mean() / tr.rolling(long).mean()).clip(0.5, 3.0)
    atr_pct = (atr_base * accel) / daily['Close']
    return atr_pct.reindex(df_ohlcv.index, method='ffill')


def precompute_vol_baselines(df_ohlcv, lookback_days=VOL_LOOKBACK_DAYS):
    times = (df_ohlcv.index.hour * 60 + df_ohlcv.index.minute).values
    dates = (df_ohlcv.index.year * 10000 + df_ohlcv.index.month * 100 + df_ohlcv.index.day).values.astype(np.int64)
    volumes = df_ohlcv['Volume'].values
    baselines = {}
    for em in range(358, 393):
        at_em = np.where(times == em)[0]
        for j in range(len(at_em)):
            d_int = int(dates[at_em[j]])
            start_j = max(0, j - lookback_days)
            if j - start_j < 5:
                baselines[(d_int, em)] = 0.0
                continue
            baselines[(d_int, em)] = float(np.median(volumes[at_em[start_j:j]]))
    return baselines


def check_volume_filter(vol_baselines, d_int, entry_minute, volume, vol_filter_pct):
    if vol_filter_pct <= 0:
        return True
    baseline = vol_baselines.get((d_int, entry_minute), 0.0)
    if baseline <= 0:
        return True
    return (volume / baseline) * 100 >= vol_filter_pct


@jit(nopython=True)
def run_single_trade(prices, highs, lows, date_ints, time_mins, entry_idx, tp, sl, exit_time):
    n = len(prices)
    if entry_idx >= n - 1: return 0.0, date_ints[entry_idx], False
    ep = prices[entry_idx]; ed = date_ints[entry_idx]
    tgt = ep * (1 + tp); stp = ep * (1 - sl)
    for k in range(entry_idx + 1, n):
        if date_ints[k] != ed: return (prices[k-1] - ep) / ep, ed, True
        sl_h = lows[k] <= stp; tp_h = highs[k] >= tgt
        if sl_h and tp_h: return (-sl if np.random.random() < 0.5 else tp), ed, True
        if sl_h: return -sl, ed, True
        if tp_h: return tp, ed, True
        if time_mins[k] >= exit_time: return (prices[k] - ep) / ep, ed, True
    return (prices[n-1] - ep) / ep, ed, True


@jit(nopython=True)
def run_single_trade_trailing_decay(prices, highs, lows, date_ints, time_mins,
                                     entry_idx, atr_pct, trail_start_mult,
                                     trail_end_mult, sl_mult, exit_time):
    n = len(prices)
    if entry_idx >= n - 1: return 0.0, date_ints[entry_idx], False
    ep = prices[entry_idx]; ed = date_ints[entry_idx]
    if atr_pct != atr_pct or atr_pct <= 0: atr_pct = 0.05
    sl = min(max(atr_pct * sl_mult, 0.01), 0.25)
    hard_stop = ep * (1 - sl)
    ts = atr_pct * trail_start_mult; te = atr_pct * trail_end_mult
    hwm = ep
    for k in range(entry_idx + 1, n):
        if date_ints[k] != ed: return (prices[k-1] - ep) / ep, ed, True
        if highs[k] > hwm: hwm = highs[k]
        unr = (hwm - ep) / ep; trailing_stop = 0.0
        if unr > 0 and ts > 0:
            r = min(unr / ts, 1.0)
            trailing_stop = hwm * (1.0 - (ts + (te - ts) * r))
        eff = max(hard_stop, trailing_stop)
        if lows[k] <= eff:
            return ((trailing_stop - ep) / ep if trailing_stop > hard_stop else -sl), ed, True
        if time_mins[k] >= exit_time: return (prices[k] - ep) / ep, ed, True
    return (prices[n-1] - ep) / ep, ed, True


@jit(nopython=True)
def run_single_trade_grace(prices, highs, lows, date_ints, time_mins,
                           entry_idx, tp, sl, exit_time, grace_minutes):
    n = len(prices)
    if entry_idx >= n - 1: return 0.0, date_ints[entry_idx], False
    ep = prices[entry_idx]; ed = date_ints[entry_idx]; et = time_mins[entry_idx]
    tgt = ep * (1 + tp); stp = ep * (1 - sl)
    for k in range(entry_idx + 1, n):
        if date_ints[k] != ed: return (prices[k-1] - ep) / ep, ed, True
        elapsed = time_mins[k] - et; sl_active = elapsed >= grace_minutes
        tp_h = highs[k] >= tgt; sl_h = lows[k] <= stp
        if sl_active and sl_h and tp_h: return (-sl if np.random.random() < 0.5 else tp), ed, True
        if sl_active and sl_h: return -sl, ed, True
        if tp_h: return tp, ed, True
        if time_mins[k] >= exit_time: return (prices[k] - ep) / ep, ed, True
    return (prices[n-1] - ep) / ep, ed, True


@jit(nopython=True)
def run_single_trade_limit_entry(prices, highs, lows, date_ints, time_mins,
                                  entry_idx, tp, sl, exit_time, dip_pct, max_wait):
    n = len(prices)
    if entry_idx >= n - 1: return 0.0, date_ints[entry_idx], False
    sp = prices[entry_idx]; ed = date_ints[entry_idx]; et = time_mins[entry_idx]
    lp = sp * (1.0 - dip_pct)
    fill_idx = -1
    for k in range(entry_idx + 1, n):
        if date_ints[k] != ed: break
        if time_mins[k] - et > max_wait: break
        if lows[k] <= lp: fill_idx = k; break
    if fill_idx < 0: return 0.0, ed, False
    ep = lp; tgt = ep * (1.0 + tp); stp = ep * (1.0 - sl)
    for k in range(fill_idx + 1, n):
        if date_ints[k] != ed: return (prices[k-1] - ep) / ep, ed, True
        sl_h = lows[k] <= stp; tp_h = highs[k] >= tgt
        if sl_h and tp_h: return (-sl if np.random.random() < 0.5 else tp), ed, True
        if sl_h: return -sl, ed, True
        if tp_h: return tp, ed, True
        if time_mins[k] >= exit_time: return (prices[k] - ep) / ep, ed, True
    return (prices[n-1] - ep) / ep, ed, True


# ==============================================================================
# REGIME FILTER
# ==============================================================================
def preload_regime_data(eval_start, eval_end):
    print("  Loading regime data (VIX, SPY, TLT)...")
    def _load(ticker):
        clean = TICKER_FILE_MAP.get(ticker, ticker)
        uri = f"gs://{BUCKET_NAME}/{INFERENCE_DIR}/{clean}_master_inference.parquet"
        try:
            d = pd.read_parquet(uri).reset_index()
            dt_cols = [c for c in d.columns if pd.api.types.is_datetime64_any_dtype(d[c])]
            if dt_cols: d['Datetime'] = pd.to_datetime(d[dt_cols[0]], utc=True)
            else:
                nc = next((c for c in d.columns if c in ('Datetime','Date','Time','Timestamp')), d.columns[0])
                d['Datetime'] = pd.to_datetime(d[nc], unit='ms', utc=True)
            d = d.set_index('Datetime').sort_index()
            d = d[~d.index.duplicated(keep='first')]
            d.index = d.index.tz_convert('America/Los_Angeles')
            return d
        except Exception:
            return None

    vix_df = _load('VIX'); spy_df = _load('SPY'); tlt_df = _load('TLT')
    spy_daily = spy_df['Close'].resample('D').last().dropna() if spy_df is not None else pd.Series(dtype=float)
    tlt_daily = tlt_df['Close'].resample('D').last().dropna() if tlt_df is not None else pd.Series(dtype=float)

    start = pd.Timestamp(eval_start, tz=PST) - timedelta(days=10)
    end = pd.Timestamp(eval_end, tz=PST) + timedelta(days=1)

    def _lk(d):
        if d is None: return {}
        sub = d[(d.index >= start) & (d.index <= end)]
        di = (sub.index.year*10000 + sub.index.month*100 + sub.index.day).astype(int)
        mi = (sub.index.hour*60 + sub.index.minute).astype(int)
        return {(int(di[i]), int(mi[i])): float(sub['Close'].values[i]) for i in range(len(sub))}

    vix_lk = _lk(vix_df); spy_lk = _lk(spy_df); tlt_lk = _lk(tlt_df)

    def _med(lk, d_int, em, w=30):
        vals = [lk[(d_int, m)] for m in range(em-w+1, em+1) if (d_int, m) in lk]
        return float(np.median(vals)) if vals else None

    def _pc(daily_s):
        m = {}
        for i in range(1, len(daily_s)):
            m[int(daily_s.index[i].strftime('%Y%m%d'))] = float(daily_s.iloc[i-1])
        return m

    spy_prev = _pc(spy_daily); tlt_prev = _pc(tlt_daily)
    regime_dict = {}
    for d in pd.date_range(start, end, freq='D', tz=PST):
        d_int = int(d.strftime('%Y%m%d'))
        for em in range(360, 391):
            v = _med(vix_lk, d_int, em) or 15.0
            sc = _med(spy_lk, d_int, em); sp = spy_prev.get(d_int)
            s_gap = ((sc - sp) / sp * 100) if sc and sp and sp != 0 else 0.0
            tc = _med(tlt_lk, d_int, em); tp = tlt_prev.get(d_int)
            t_gap = ((tc - tp) / tp * 100) if tc and tp and tp != 0 else 0.0
            regime_dict[(d_int, em)] = np.array([v, t_gap, s_gap])
    print(f"    Regime data: {len(regime_dict)} combos")
    return regime_dict


# ==============================================================================
# OHLCV ARRAYS
# ==============================================================================
def precompute_ohlcv_arrays(df_ohlcv, start_date, end_date):
    s = pd.to_datetime(start_date).tz_localize('America/Los_Angeles')
    e = pd.to_datetime(end_date).tz_localize('America/Los_Angeles') + timedelta(days=1)
    df = df_ohlcv[(df_ohlcv.index >= s) & (df_ohlcv.index < e)]
    if len(df) == 0: return None
    return {
        'index': df.index, 'prices': df['Close'].values.copy(),
        'highs': df['High'].values.copy(), 'lows': df['Low'].values.copy(),
        'volumes': df['Volume'].values.copy(),
        'dates': (df.index.year*10000 + df.index.month*100 + df.index.day).values.astype(np.int64),
        'times': (df.index.hour*60 + df.index.minute).values.astype(np.int64),
    }


# ==============================================================================
# MAIN BACKTEST
# ==============================================================================
def run_backtest(configs, label, regime_dict):
    print(f"\n{'='*W}")
    print(f"  BACKTEST: {label} -- {len(configs)} tickers")
    print(f"{'='*W}")
    if not configs:
        print(f"  No configs -- skipping"); return pd.DataFrame()

    all_trades = []
    for cfg in configs:
        ticker = cfg['ticker']; mode = cfg['tpsl_mode']; em = cfg['entry_minute']
        tier_tag = cfg.get('tier', '')
        df_raw = load_master(ticker, start_date=TEST_START)
        if df_raw is None: print(f"  {ticker}: SKIP"); continue
        df_raw = heal_data(df_raw.sort_index())
        df_pst = df_raw.copy(); df_pst.index = df_pst.index.tz_convert(PST)

        signals = score_ticker_from_csv(ticker, cfg, df_pst)
        if not signals:
            print(f"  {ticker}: No signals"); del df_raw; gc.collect(); continue

        atr_series = (compute_daily_atr_series(df_pst) if mode in ('atr','trailing_decay','atr_sl')
                      else compute_adaptive_atr_series(df_pst) if mode == 'atr_adaptive' else None)
        vol_baselines = precompute_vol_baselines(df_pst)
        ohlcv = precompute_ohlcv_arrays(df_pst, TEST_START, TEST_END)
        if ohlcv is None: del df_raw; gc.collect(); continue

        trades = []; fv = 0; fr = 0; last_d = -1; uf = 0
        for sig in signals:
            bts = sig['bar_ts'].tz_convert(PST) if sig['bar_ts'].tzinfo else sig['bar_ts']
            d_int = bts.year*10000 + bts.month*100 + bts.day
            if d_int == last_d: continue

            oi = None
            for j in range(len(ohlcv['index'])):
                if ohlcv['dates'][j] == d_int and ohlcv['times'][j] == em:
                    oi = j; break
            if oi is not None:
                if not check_volume_filter(vol_baselines, d_int, em,
                                           float(ohlcv['volumes'][oi]), cfg['vol_filter_pct']):
                    fv += 1; continue

            rk = (d_int, em); vix_val = regime_dict[rk][0] if rk in regime_dict else None
            atr_pct = 0.05
            if atr_series is not None:
                ts = pd.Timestamp(bts)
                if ts in atr_series.index:
                    v = float(atr_series.loc[ts])
                    if not np.isnan(v) and v > 0: atr_pct = v
                else:
                    prior = atr_series.index[atr_series.index <= ts]
                    if len(prior) > 0:
                        v = float(atr_series.iloc[atr_series.index.get_loc(prior[-1])])
                        if not np.isnan(v) and v > 0: atr_pct = v

            tp, sl = compute_dynamic_tpsl(cfg, atr_pct, vix_val)
            if oi is None: continue

            if rk in regime_dict:
                rv = regime_dict[rk]
                if rv[0] > cfg.get('vix_threshold', 99) and rv[1] < cfg.get('tlt_crash', -99) and rv[2] > cfg.get('spy_ignore', 99):
                    fr += 1; continue

            if mode == 'trailing_decay':
                pnl, td, valid = run_single_trade_trailing_decay(ohlcv['prices'], ohlcv['highs'], ohlcv['lows'], ohlcv['dates'], ohlcv['times'], oi, atr_pct, cfg['trail_start_mult'], cfg['trail_end_mult'], cfg['sl_mult'], EXIT_TIME_MIN)
            elif mode == 'grace':
                pnl, td, valid = run_single_trade_grace(ohlcv['prices'], ohlcv['highs'], ohlcv['lows'], ohlcv['dates'], ohlcv['times'], oi, tp, sl, EXIT_TIME_MIN, cfg['grace_minutes'])
            elif mode == 'limit_entry':
                pnl, td, valid = run_single_trade_limit_entry(ohlcv['prices'], ohlcv['highs'], ohlcv['lows'], ohlcv['dates'], ohlcv['times'], oi, tp, sl, EXIT_TIME_MIN, cfg['dip_pct'], cfg['max_wait_minutes'])
                if not valid: uf += 1; continue
            else:
                pnl, td, valid = run_single_trade(ohlcv['prices'], ohlcv['highs'], ohlcv['lows'], ohlcv['dates'], ohlcv['times'], oi, tp, sl, EXIT_TIME_MIN)

            if valid:
                pnl -= 2 * SLIPPAGE_PCT
                trades.append({'ticker': ticker, 'experiment': cfg['experiment'],
                               'date_int': int(td), 'pnl': float(pnl), 'prob': sig['prob'],
                               'tp': tp, 'sl': sl, 'mode': mode, 'tier': tier_tag,
                               'dip_pct': cfg.get('dip_pct', 0), 'max_wait': cfg.get('max_wait_minutes', 0)})
                last_d = int(td)

        w = sum(1 for t in trades if t['pnl'] > 0); l = len(trades) - w
        avg = np.mean([t['pnl'] for t in trades]) if trades else 0
        extra = ""
        if fv > 0: extra += f" vol_f={fv}"
        if fr > 0: extra += f" reg_f={fr}"
        if uf > 0: extra += f" unfilled={uf}"
        print(f"  {ticker:<7} {cfg['experiment']:<25} ENS  {len(signals):>3} sig -> {len(trades):>3} trades "
              f"({w}W/{l}L)  avg {avg:>+6.2%}  {mode}  dip={cfg.get('dip_pct',0):.1%}{extra}")
        all_trades.extend(trades)
        del df_raw, df_pst, ohlcv; gc.collect()

    if not all_trades: print(f"  No trades for {label}"); return pd.DataFrame()
    tdf = pd.DataFrame(all_trades)
    tdf['date'] = pd.to_datetime(tdf['date_int'].astype(str), format='%Y%m%d')
    return tdf


# ==============================================================================
# ANALYTICS
# ==============================================================================
def per_asset_stats(trades_df):
    if trades_df.empty: return pd.DataFrame()
    months = max(1.0, (trades_df['date'].max() - trades_df['date'].min()).days / 30.44)
    rows = []
    for ticker in sorted(trades_df['ticker'].unique()):
        td = trades_df[trades_df['ticker'] == ticker]
        n = len(td); w = int((td['pnl'] > 0).sum())
        rows.append({'ticker': ticker, 'experiment': td['experiment'].iloc[0],
                     'mode': td['mode'].iloc[0], 'tier': td['tier'].iloc[0],
                     'trades': n, 'wins': w, 'losses': n-w,
                     'win_rate': w/n, 'avg_pnl': td['pnl'].mean(),
                     'total_pnl': td['pnl'].sum(), 'freq': n/months})
    return pd.DataFrame(rows)


def portfolio_summary(trades_df, label):
    if trades_df.empty: return {}
    n = len(trades_df); w = int((trades_df['pnl'] > 0).sum())
    months = max(1.0, (trades_df['date'].max() - trades_df['date'].min()).days / 30.44)
    return {'label': label, 'tickers': trades_df['ticker'].nunique(), 'trades': n,
            'wins': w, 'losses': n-w, 'win_rate': w/n, 'avg_pnl': trades_df['pnl'].mean(),
            'total_pnl': trades_df['pnl'].sum(), 'freq_per_mo': n/months}


# ==============================================================================
# RUN
# ==============================================================================
KEYS = ['A', 'B', 'E', 'F']

print(f"\n{'='*W}")
print(f"  PER-TICKER BEST-MODE SELECTION BACKTEST")
print(f"  TEST: {TEST_START} -> {TEST_END}  |  Model: {MODEL_VERSION}  |  Threshold: {DEPLOY_THRESHOLD:.1%}")
print(f"{'='*W}")

raw_df = load_raw_wf_data()
both_exp_tickers = find_both_exp_tickers(raw_df)

configs_A = select_A_deployed()
configs_B = select_B_tier1plus(raw_df, both_exp_tickers)
configs_E = select_E_best_mode(raw_df)

regime_dict = preload_regime_data(TEST_START, TEST_END)

results = {}
stats = {}
summaries = {}

for key, cfgs in [('A', configs_A), ('B', configs_B), ('E', configs_E)]:
    label = f"{key}: {METHOD_LABELS[key]}"
    trades = run_backtest(cfgs, label, regime_dict)
    results[key] = trades
    stats[key] = per_asset_stats(trades)
    summaries[key] = portfolio_summary(trades, METHOD_LABELS[key])

configs_F = build_F_from_E(configs_E, results['E'], raw_df)
label_f = f"F: {METHOD_LABELS['F']}"
trades_f = run_backtest(configs_F, label_f, regime_dict)
results['F'] = trades_f
stats['F'] = per_asset_stats(trades_f)
summaries['F'] = portfolio_summary(trades_f, METHOD_LABELS['F'])


# ==============================================================================
# PER-ASSET TABLES
# ==============================================================================
for key in KEYS:
    s = stats[key]
    if s.empty: continue
    print(f"\n\n{'='*W}")
    print(f"  PER-ASSET: {key} ({METHOD_LABELS[key]})")
    print(f"{'='*W}")
    print(f"  {'Ticker':<7} {'Experiment':<25} {'Mode':<14} {'Tier':<6} {'N':>4} {'W/L':>7} "
          f"{'WR':>5} {'AvgPnL':>8} {'TotPnL':>9}")
    print(f"  {'-'*(W-2)}")
    for _, r in s.sort_values('avg_pnl', ascending=False).iterrows():
        print(f"  {r['ticker']:<7} {r['experiment']:<25} {r['mode']:<14} {str(r.get('tier',''))[:6]:<6} {r['trades']:>4} "
              f"{r['wins']:>3}/{r['losses']:<3} {r['win_rate']:>4.0%} "
              f"{r['avg_pnl']:>+7.2%} {r['total_pnl']:>+8.2%}")


# ==============================================================================
# PORTFOLIO SUMMARY
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  PORTFOLIO SUMMARY")
print(f"{'='*W}")
print(f"  {'Metric':<16}", end='')
for key in KEYS:
    print(f" {key+':'+METHOD_LABELS[key]:>18}", end='')
print()
print(f"  {'-'*(16 + 19*4)}")

metrics = ['tickers', 'trades', 'wins', 'win_rate', 'avg_pnl', 'total_pnl', 'freq_per_mo']
for m in metrics:
    print(f"  {m:<16}", end='')
    for key in KEYS:
        s = summaries[key]
        if not s:
            print(f" {'---':>18}", end='')
        elif m in ('win_rate', 'avg_pnl', 'total_pnl'):
            print(f" {s[m]:>17.1%}" if m == 'win_rate' else f" {s[m]:>17.2%}", end='')
        elif m == 'freq_per_mo':
            print(f" {s[m]:>17.1f}", end='')
        else:
            print(f" {s[m]:>18}", end='')
    print()


# ==============================================================================
# HEAD-TO-HEAD
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  HEAD-TO-HEAD: avg PnL per ticker across all methods")
print(f"{'='*W}")
all_tickers = sorted(set().union(*[set(stats[k]['ticker'].tolist()) for k in stats if not stats[k].empty]))
print(f"  {'Ticker':<7}", end='')
for key in KEYS:
    print(f" {key:>10}", end='')
print(f" {'Best':>6}")
print(f"  {'-'*(7 + 11*4 + 7)}")

method_wins = {k: 0 for k in KEYS}
for ticker in all_tickers:
    print(f"  {ticker:<7}", end='')
    best_key = None; best_pnl = -999
    for key in KEYS:
        s = stats[key]
        match = s[s['ticker'] == ticker] if not s.empty else pd.DataFrame()
        if not match.empty:
            v = float(match.iloc[0]['avg_pnl'])
            print(f" {v:>+9.2%}", end='')
            if v > best_pnl: best_pnl = v; best_key = key
        else:
            print(f" {'---':>10}", end='')
    if best_key:
        method_wins[best_key] += 1
    print(f" {best_key or '':>6}")

print(f"\n  Wins: ", end='')
for key in KEYS:
    print(f"  {key}={method_wins[key]}", end='')
print()


# ==============================================================================
# MONTHLY BREAKDOWN
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  MONTHLY BREAKDOWN")
print(f"{'='*W}")

all_months = sorted(set().union(*[
    set(results[k]['date'].dt.to_period('M').unique().tolist())
    for k in results if not results[k].empty
]))

print(f"  {'Month':<10}", end='')
for key in KEYS:
    print(f" {key+'_avg':>8} {key+'_N':>4}", end='')
print()
print(f"  {'-'*(10 + 13*4)}")

for month in all_months:
    print(f"  {str(month):<10}", end='')
    for key in KEYS:
        tdf = results[key]
        if tdf.empty:
            print(f" {'---':>8} {'':>4}", end=''); continue
        mt = tdf[tdf['date'].dt.to_period('M') == month]
        if len(mt) == 0:
            print(f" {'---':>8} {'':>4}", end='')
        else:
            print(f" {mt['pnl'].mean():>+7.2%} {len(mt):>4}", end='')
    print()


# ==============================================================================
# CHARTS
# ==============================================================================
print(f"\n  Generating charts...")
fig, axes = plt.subplots(2, 1, figsize=(16, 13))
fig.suptitle(f'Per-Ticker Best-Mode Selection Backtest\n'
             f'TEST ({TEST_START} -> {TEST_END})', fontsize=14, fontweight='bold')

ax1 = axes[0]
bar_data = []
for ticker in all_tickers:
    row = {'ticker': ticker}
    for key in KEYS:
        s = stats[key]
        m = s[s['ticker'] == ticker] if not s.empty else pd.DataFrame()
        row[key] = float(m.iloc[0]['avg_pnl']) * 100 if not m.empty else 0
    bar_data.append(row)
bdf = pd.DataFrame(bar_data).sort_values('F', ascending=True)
y_pos = np.arange(len(bdf))
bar_h = 0.18
for i, key in enumerate(KEYS):
    ax1.barh(y_pos + (i-1.5)*bar_h, bdf[key].values, bar_h,
             label=f'{key}: {METHOD_LABELS[key]}', color=METHOD_COLORS[key], alpha=0.85)
ax1.set_yticks(y_pos); ax1.set_yticklabels(bdf['ticker'].values)
ax1.axvline(x=0, color='gray', linewidth=0.5, linestyle='--')
ax1.set_title('Avg PnL per Trade (%)', fontsize=12); ax1.set_xlabel('Avg PnL (%)')
ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3, axis='x')

ax2 = axes[1]
for key in KEYS:
    tdf = results[key]
    if tdf.empty: continue
    daily = tdf.groupby('date')['pnl'].mean()
    eq = STARTING_CAPITAL
    pts = [{'date': daily.index[0] - pd.Timedelta(days=1), 'equity': eq}]
    for d, p in daily.items():
        n_t = len(tdf[tdf['date'] == d])
        eq += (eq / max(n_t, 1)) * p * n_t
        pts.append({'date': d, 'equity': eq})
    edf = pd.DataFrame(pts).set_index('date')
    ax2.plot(edf.index, edf['equity']/1000, label=f'{key}: {METHOD_LABELS[key]}',
             color=METHOD_COLORS[key], linewidth=2)
ax2.axhline(y=STARTING_CAPITAL/1000, color='gray', linewidth=0.5, linestyle='--')
ax2.set_title(f'Cumulative Equity (${STARTING_CAPITAL/1000:.0f}K start)', fontsize=12)
ax2.set_ylabel('Equity ($K)')
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
ax2.tick_params(axis='x', rotation=45); ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

plt.tight_layout()
chart_path = '/tmp/backtest_tier_comparison.png'
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
print(f"  Charts saved: {chart_path}")
try:
    from IPython.display import display, Image
    display(Image(filename=chart_path))
except ImportError:
    plt.show()


# ==============================================================================
# FINAL RANKING
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  FINAL RANKING")
print(f"{'='*W}")
ranking = []
for key in KEYS:
    s = summaries[key]
    if s:
        ranking.append({'key': key, 'label': METHOD_LABELS[key], **s})
ranking_df = pd.DataFrame(ranking).sort_values('avg_pnl', ascending=False)
print(f"\n  {'#':>2} {'Method':<20} {'Tickers':>8} {'Trades':>7} {'WR':>6} {'AvgPnL':>8} {'TotPnL':>9} {'Freq/mo':>8}")
print(f"  {'-'*75}")
for i, (_, r) in enumerate(ranking_df.iterrows(), 1):
    print(f"  {i:>2} {r['key']+': '+r['label']:<20} {r['tickers']:>8} {r['trades']:>7} "
          f"{r['win_rate']:>5.0%} {r['avg_pnl']:>+7.2%} {r['total_pnl']:>+8.2%} {r['freq_per_mo']:>7.1f}")

best = ranking_df.iloc[0]
print(f"\n  WINNER: {best['key']}: {best['label']} — {best['tickers']} tickers, "
      f"avg {best['avg_pnl']:+.2%}/trade, {best['win_rate']:.0%} WR, total {best['total_pnl']:+.2%}")
print(f"{'='*W}")
