"""
backtest_tier_comparison.py — Full exploit investigation with live TF scoring
==============================================================================
Live TF inference (no pre-computed CSVs), data through June 18.
Six experiment families: 152, 160, 170 WF (→175 deployed), 175, 180 WF (→185 deployed), 185.
Four strategies:
  B: Tier 1+ (limit-only + both 152+160 experiments)
  E: Best All Families (best config per ticker across all families and modes)
  G: Full Exploit (best config per ticker × experiment family — multiple per ticker)
  H: Exploit No LUNR (G minus LUNR)

BQ data audit + coverage analysis + WF vs actual PnL correlation.
Colab-ready (GPU recommended for inference speed).
"""

import os
import io
import json
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

from tf_keras.layers import (Input, Dense, Dropout, LayerNormalization,
                              MultiHeadAttention, Add, GlobalAveragePooling1D,
                              Embedding, Flatten, Concatenate,
                              Conv1D, MaxPool1D, SpatialDropout1D)
from tf_keras.regularizers import l2
from tf_keras.models import Model

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
TEST_END   = '2026-06-18'

STARTING_CAPITAL = 200_000

DEPLOY_EXPERIMENTS = {'152_XSECTOR', '160_XSECTOR', '170_XSECTOR',
                      '175_XSECTOR', '180_XSECTOR', '185_XSECTOR',
                      '200_XSECTOR'}
EXCLUDE_TICKERS = set()

WALKFORWARD_FOLDS = {
    '175_XSECTOR': ['170_XSECTOR', '171_XSECTOR', '172_XSECTOR',
                     '173_XSECTOR', '174_XSECTOR', '175_XSECTOR'],
    '185_XSECTOR': ['180_XSECTOR', '181_XSECTOR', '182_XSECTOR',
                     '183_XSECTOR', '184_XSECTOR', '185_XSECTOR'],
}
WALKFORWARD_BASE = set(WALKFORWARD_FOLDS.keys())

BUCKET_NAME    = "time-series-bucket-ys"
BQ_TABLE       = "ys-time-series.scoring_logs.wf_optimization_results"
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

GCS_MODEL_PATH = "Algos/to_deploy/Saved_models_v2/UNIVERSAL/Direction"
TIME_STEPS = 240
EMBED_DIM = 16
ROC_HORIZONS = [5, 30, 60, 120]
DATA_START_DATE = "2022-01-01"
CONTEXT_ASSETS = ['SPY', 'VIX', 'TLT', 'IWM']

ASSET_COLS = (['Gap_Pct', 'RSI_14', 'RSI_60', 'RVOL', 'Compression',
               'ATR_14', 'Dist_to_PM_High', 'Bar_Is_Real'] +
              [f'ROC_{h}m' for h in ROC_HORIZONS] + MICRO_COLS)

CONTEXT_COLS = []
for _t in CONTEXT_ASSETS:
    CONTEXT_COLS.append(f'{_t}_Anchor_ROC')
    for _h in ROC_HORIZONS:
        CONTEXT_COLS.append(f'{_t}_ROC_{_h}m')
CONTEXT_COLS += ['Time_Pos']
ALL_FEATURE_COLS = ASSET_COLS + CONTEXT_COLS

TRAIN_TEACHERS = [
    'BBAI', 'RCAT', 'LUNR', 'ASTS', 'ACHR', 'MVIS',
    'WULF', 'CIFR', 'IREN', 'KEEL', 'HUT', 'CLSK', 'APLD',
]

TEACHERS_31 = [
    'CIFR', 'IREN', 'KEEL', 'HUT', 'APLD', 'WULF', 'RIOT', 'CLSK', 'BTBT', 'ALAB', 'MARA',
    'RCAT', 'ASTS', 'LUNR', 'BBAI', 'RKLB', 'SPIR', 'BKSY', 'MNTS', 'PLUG', 'BE', 'FCEL', 'FLNC',
    'OKLO', 'SMR', 'MVIS', 'BGDE', 'RDW', 'PL', 'VOYG', 'SMLR',
]

EXP_CONFIGS = {
    '152_XSECTOR': {
        'train_teachers': TRAIN_TEACHERS,
        'n_assets': len(TRAIN_TEACHERS) + 1,
        'stats_id': '152_XSECTOR',
    },
    '160_XSECTOR': {
        'train_teachers': TRAIN_TEACHERS,
        'n_assets': len(TRAIN_TEACHERS) + 1,
        'stats_id': '160_XSECTOR',
    },
    '170_XSECTOR': {
        'train_teachers': TRAIN_TEACHERS,
        'n_assets': len(TRAIN_TEACHERS) + 1,
        'stats_id': '175_XSECTOR',
        'model_id_override': '175_XSECTOR',
        'seed_base': 175,
    },
    '175_XSECTOR': {
        'train_teachers': TRAIN_TEACHERS,
        'n_assets': len(TRAIN_TEACHERS) + 1,
        'stats_id': '175_XSECTOR',
    },
    '180_XSECTOR': {
        'train_teachers': TEACHERS_31,
        'n_assets': len(TEACHERS_31) + 1,
        'stats_id': '185_XSECTOR',
        'model_id_override': '185_XSECTOR',
        'seed_base': 185,
    },
    '185_XSECTOR': {
        'train_teachers': TEACHERS_31,
        'n_assets': len(TEACHERS_31) + 1,
        'stats_id': '185_XSECTOR',
    },
    '200_XSECTOR': {
        'train_teachers': TEACHERS_31,
        'n_assets': len(TEACHERS_31) + 1,
        'stats_id': '200_XSECTOR',
        'model_builder': 'cnn_transformer',
        'embed_dim': 32,
    },
}
for _eid, _ecfg in EXP_CONFIGS.items():
    _idx = {t: i for i, t in enumerate(_ecfg['train_teachers'])}
    if 'BITF' in _idx: _idx['KEEL'] = _idx['BITF']
    if 'MIGI' in _idx: _idx['BGDE'] = _idx['MIGI']
    _ecfg['asset_index'] = _idx
    _ecfg['other_idx'] = len(_ecfg['train_teachers'])

METHOD_LABELS = {
    'A': 'deployed',
    'B': 'tier1_plus',
    'C': 'tier1_limit',
    'D': 'combined',
    'E': 'best_all_fam',
    'F': 'tier1_wide',
    'G': 'full_exploit',
    'H': 'exploit_no_lunr',
}
METHOD_COLORS = {
    'A': '#9E9E9E',
    'B': '#E91E63',
    'C': '#2196F3',
    'D': '#4CAF50',
    'E': '#FF9800',
    'F': '#9C27B0',
    'G': '#00BCD4',
    'H': '#795548',
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
        _ecfg = EXP_CONFIGS.get(base_exp, {})
        base_num = _ecfg.get('seed_base', int(base_exp.split('_')[0]))
        seed_list = list(range(base_num * 100 + 1, base_num * 100 + 16))
        is_ens = True
        seed_val = -1
    else:
        seed_list = [seed_val]

    model_id = base_exp
    stats_id = base_exp
    for wf_base, wf_folds in WALKFORWARD_FOLDS.items():
        if base_exp in wf_folds:
            model_id = wf_base
            stats_id = wf_base
            break

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
        'model_id': model_id,
        'stats_id': stats_id,
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
    any2_exp = set()
    for tk in survived['ticker'].unique():
        tk_exps = set(survived[survived['ticker'] == tk]['exp_fam'].unique())
        if '152' in tk_exps and '160' in tk_exps:
            both_exp.add(tk)
        if len(tk_exps) >= 2:
            any2_exp.add(tk)

    print(f"    Tickers with BOTH 152+160 (>= {threshold:.1%}): {len(both_exp)}")
    print(f"      {sorted(both_exp)}")
    print(f"    Tickers with ANY 2 families (>= {threshold:.1%}): {len(any2_exp)}")
    print(f"      {sorted(any2_exp)}")

    fam_counts = survived.groupby('exp_fam')['ticker'].nunique()
    for fam in sorted(fam_counts.index):
        label = {'152': '152 Late', '160': '160 Early', '170': '170 WF-f1',
                 '175': '175 WF-deployed', '180': '180 WF-f1',
                 '185': '185 WF-deployed', '200': '200 CNN-TX'}.get(fam, fam)
        print(f"    {label}: {fam_counts[fam]} tickers with qualifying configs")
    return both_exp, any2_exp


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


def select_C_tier1(df):
    """C: Tier 1 — limit-only, all qualifying tickers."""
    print(f"\n  --- C: Tier 1 (limit-only, all tickers) ---")
    sa = _build_seed_groups(df)
    sa = sa[(sa['n_seeds'] >= 3) & (sa['tpsl_mode'] == 'limit_entry')]
    sa['score'] = (sa['mean_va_pnl'] ** 2) * sa['agree_18']
    best = sa.loc[sa.groupby('ticker')['score'].idxmax()] if not sa.empty else sa
    best = best[best['mean_va_pnl'] >= DEPLOY_THRESHOLD].copy()
    best['tier'] = 'tier1'
    configs = _sa_to_configs(best)
    print(f"    {len(configs)} DEPLOY configs")
    for c in configs:
        print(f"      {c['ticker']:<7} {c['experiment']:<25} "
              f"pnl={c['wf_avg_pnl']:.2%} agree={c['seed_agree_pct']:.0%} ({c['n_seeds']}s) "
              f"dip={c['dip_pct']:.1%} wait={c['max_wait_minutes']}m")
    return configs


def select_D_combined(df, both_exp_tickers):
    """D: Tier 1 (limit) + Tier 2 (stability-scored non-limit, non-overlapping)."""
    print(f"\n  --- D: Combined (Tier 1 + Tier 2) ---")

    # Tier 1: limit-only
    sa = _build_seed_groups(df)
    sa_lim = sa[(sa['n_seeds'] >= 3) & (sa['tpsl_mode'] == 'limit_entry')]
    sa_lim['score'] = (sa_lim['mean_va_pnl'] ** 2) * sa_lim['agree_18']
    best_lim = sa_lim.loc[sa_lim.groupby('ticker')['score'].idxmax()] if not sa_lim.empty else sa_lim
    best_lim = best_lim[best_lim['mean_va_pnl'] >= DEPLOY_THRESHOLD].copy()
    best_lim['tier'] = 'tier1'
    tier1_tickers = set(best_lim['ticker'])

    # Tier 2: all modes, stability-scored, non-overlapping
    sa_t2 = sa[(sa['n_seeds'] >= 3) & (~sa['ticker'].isin(tier1_tickers))]
    blended = np.sqrt(sa_t2['mean_opt_pnl'].clip(1e-4) * sa_t2['mean_va_pnl'].clip(1e-4))
    sa_t2 = sa_t2.copy()
    sa_t2['score'] = blended * sa_t2['mean_va_wr'] * sa_t2['agree_18']
    best_t2 = sa_t2.loc[sa_t2.groupby('ticker')['score'].idxmax()] if not sa_t2.empty else sa_t2
    best_t2 = best_t2[best_t2['mean_va_pnl'] >= DEPLOY_THRESHOLD].copy()
    best_t2['tier'] = 'tier2'

    configs_t1 = _sa_to_configs(best_lim)
    configs_t2 = _sa_to_configs(best_t2)
    configs = configs_t1 + configs_t2

    print(f"    Tier 1: {len(configs_t1)} tickers (limit-only)")
    for c in configs_t1:
        print(f"      T1  {c['ticker']:<7} {c['experiment']:<25} "
              f"pnl={c['wf_avg_pnl']:.2%} dip={c['dip_pct']:.1%}")
    print(f"    Tier 2: {len(configs_t2)} tickers (stability-scored)")
    for c in configs_t2:
        print(f"      T2  {c['ticker']:<7} {c['experiment']:<25} {c['tpsl_mode']:<14} "
              f"pnl={c['wf_avg_pnl']:.2%}")
    print(f"    Total: {len(configs)} configs")
    return configs


def select_E_best_all(df):
    """E: Best config per ticker across all families and all 9 modes."""
    print(f"\n  --- E: Best All Families (all modes, best per ticker) ---")
    sa = _build_seed_groups(df)
    sa = sa[sa['n_seeds'] >= 3]
    sa['score'] = (sa['mean_va_pnl'] ** 2) * sa['agree_18']
    best = sa.loc[sa.groupby('ticker')['score'].idxmax()] if not sa.empty else sa
    best = best[best['mean_va_pnl'] >= DEPLOY_THRESHOLD].copy()
    best['tier'] = 'best_all'

    best['exp_fam'] = best['experiment'].str[:3]
    configs = _sa_to_configs(best)
    print(f"    {len(configs)} DEPLOY configs")

    fam_counts = best['exp_fam'].value_counts()
    for fam, n in sorted(fam_counts.items()):
        label = {'152': '152 Late', '160': '160 Early', '170': '170 WF-f1',
                 '175': '175 WF-deployed', '180': '180 WF-f1',
                 '185': '185 WF-deployed', '200': '200 CNN-TX'}.get(fam, fam)
        print(f"      {label}: {n} tickers")

    for c in configs:
        print(f"      {c['ticker']:<7} {c['experiment']:<25} {c['tpsl_mode']:<14} "
              f"pnl={c['wf_avg_pnl']:.2%} agree={c.get('seed_agree_pct',0):.0%} ({c.get('n_seeds',0)}s)")
    return configs


def select_F_tier1_wide(df, any2_exp_tickers):
    """F: Tier 1 wide — limit-only + any 2 experiment families."""
    print(f"\n  --- F: Tier 1 Wide (limit + any 2 families) ---")
    sa = _build_seed_groups(df)
    sa = sa[(sa['n_seeds'] >= 3) & (sa['tpsl_mode'] == 'limit_entry')]
    sa = sa[sa['ticker'].isin(any2_exp_tickers)]
    sa['score'] = (sa['mean_va_pnl'] ** 2) * sa['agree_18']
    best = sa.loc[sa.groupby('ticker')['score'].idxmax()] if not sa.empty else sa
    best = best[best['mean_va_pnl'] >= DEPLOY_THRESHOLD].copy()
    best['tier'] = 'tier1_wide'
    configs = _sa_to_configs(best)
    print(f"    {len(configs)} DEPLOY configs")
    for c in configs:
        print(f"      {c['ticker']:<7} {c['experiment']:<25} "
              f"pnl={c['wf_avg_pnl']:.2%} agree={c['seed_agree_pct']:.0%} ({c['n_seeds']}s) "
              f"dip={c['dip_pct']:.1%} wait={c['max_wait_minutes']}m")
    return configs


def select_G_exploit(df):
    """G: Full Exploit — best config per ticker per experiment family. Multiple configs per ticker."""
    print(f"\n  --- G: Full Exploit (best per ticker × experiment family) ---")
    sa = _build_seed_groups(df)
    sa = sa[sa['n_seeds'] >= 3]
    sa['score'] = (sa['mean_va_pnl'] ** 2) * sa['agree_18']
    sa['exp_fam'] = sa['experiment'].str[:3]

    best = sa.loc[sa.groupby(['ticker', 'exp_fam'])['score'].idxmax()] if not sa.empty else sa
    best = best[best['mean_va_pnl'] >= DEPLOY_THRESHOLD].copy()
    best['tier'] = 'exploit'
    configs = _sa_to_configs(best)

    by_fam = {}
    for c in configs:
        fam = c['experiment'][:3]
        by_fam.setdefault(fam, []).append(c)

    print(f"    {len(configs)} DEPLOY configs across {len(by_fam)} families, "
          f"{len(set(c['ticker'] for c in configs))} unique tickers")
    for fam in sorted(by_fam):
        label = {'152': '152 Late', '160': '160 Early', '170': '170 WF-f1',
                 '175': '175 WF-deployed', '180': '180 WF-f1',
                 '185': '185 WF-deployed', '200': '200 CNN-TX'}.get(fam, fam)
        print(f"    {label}: {len(by_fam[fam])} configs")
    for c in configs:
        print(f"      {c['ticker']:<7} {c['experiment']:<25} {c['tpsl_mode']:<14} "
              f"pnl={c['wf_avg_pnl']:.2%} agree={c.get('seed_agree_pct',0):.0%} ({c.get('n_seeds',0)}s)")
    return configs


def select_H_exploit_no_lunr(df):
    """H: Full Exploit minus LUNR (known June catastrophe)."""
    print(f"\n  --- H: Full Exploit No LUNR ---")
    df_no_lunr = df[df['ticker'] != 'LUNR'].copy()
    sa = _build_seed_groups(df_no_lunr)
    sa = sa[sa['n_seeds'] >= 3]
    sa['score'] = (sa['mean_va_pnl'] ** 2) * sa['agree_18']
    sa['exp_fam'] = sa['experiment'].str[:3]

    best = sa.loc[sa.groupby(['ticker', 'exp_fam'])['score'].idxmax()] if not sa.empty else sa
    best = best[best['mean_va_pnl'] >= DEPLOY_THRESHOLD].copy()
    best['tier'] = 'exploit_nolunr'
    configs = _sa_to_configs(best)

    print(f"    {len(configs)} DEPLOY configs, "
          f"{len(set(c['ticker'] for c in configs))} unique tickers (LUNR excluded)")
    for c in configs:
        print(f"      {c['ticker']:<7} {c['experiment']:<25} {c['tpsl_mode']:<14} "
              f"pnl={c['wf_avg_pnl']:.2%} agree={c.get('seed_agree_pct',0):.0%} ({c.get('n_seeds',0)}s)")
    return configs


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
# LIVE TF SCORING
# ==============================================================================
_model_cache = {}
_stats_cache = {}
_feat_cache = {}

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.4):
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    res = Add()([x, inputs])
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    return Add()([x, res])


def build_universal_model(n_assets, embed_dim=EMBED_DIM):
    in_a  = Input(shape=(TIME_STEPS, len(ASSET_COLS)), name='asset_in')
    in_c  = Input(shape=(len(CONTEXT_COLS),),          name='context_in')
    in_id = Input(shape=(1,), dtype=tf.int32,          name='asset_id')
    x = transformer_encoder(Dense(128)(in_a), 128, 4, 128)
    x = GlobalAveragePooling1D()(transformer_encoder(x, 128, 4, 128))
    y = Dense(128, activation='relu')(LayerNormalization()(in_c))
    e = Embedding(n_assets, embed_dim, name='asset_embed')(in_id)
    e = Flatten()(e)
    e = Dense(32, activation='relu')(e)
    z = Concatenate()([x, Dropout(0.5)(y), e])
    z = Dense(128, activation='relu', kernel_regularizer=l2(5e-3))(z)
    out = Dense(1, activation='sigmoid', dtype='float32')(Dropout(0.5)(z))
    return Model(inputs=[in_a, in_c, in_id], outputs=out)


def build_cnn_transformer_model(n_assets, embed_dim=32):
    CNN_FILTERS = 64
    CNN_KERNELS = [3, 7, 15]
    CNN_POOL_1 = 4
    CNN_POOL_2 = 4

    in_a  = Input(shape=(TIME_STEPS, len(ASSET_COLS)), name='asset_in')
    in_c  = Input(shape=(len(CONTEXT_COLS),),          name='context_in')
    in_id = Input(shape=(1,), dtype=tf.int32,          name='asset_id')

    x = Dense(CNN_FILTERS)(in_a)
    x = SpatialDropout1D(0.2)(x)
    conv_branches = []
    for k in CNN_KERNELS:
        c = Conv1D(CNN_FILTERS, kernel_size=k, padding='same', activation='relu')(x)
        conv_branches.append(c)
    x = Concatenate(axis=-1)(conv_branches)
    x = LayerNormalization()(x)
    x = MaxPool1D(pool_size=CNN_POOL_1)(x)
    x = SpatialDropout1D(0.1)(x)
    x = Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = MaxPool1D(pool_size=CNN_POOL_2)(x)

    x = transformer_encoder(x, 128, 4, 128)
    x = transformer_encoder(x, 128, 4, 128)
    x = GlobalAveragePooling1D()(x)

    y = Dense(128, activation='relu')(LayerNormalization()(in_c))
    e = Embedding(n_assets, embed_dim, name='asset_embed')(in_id)
    e = Flatten()(e)
    e = Dense(32, activation='relu')(e)

    z = Concatenate()([x, Dropout(0.5)(y), e])
    z = Dense(128, activation='relu', kernel_regularizer=l2(5e-3))(z)
    out = Dense(1, activation='sigmoid', dtype='float32')(Dropout(0.5)(z))
    return Model(inputs=[in_a, in_c, in_id], outputs=out)


def load_scaling_stats(stats_id):
    if stats_id in _stats_cache:
        return _stats_cache[stats_id]
    blob = gcs_client.bucket(BUCKET_NAME).blob(
        f"{INFERENCE_DIR}/scaling_stats_experiment_{stats_id}.json")
    stats = json.loads(blob.download_as_text())
    _stats_cache[stats_id] = stats
    print(f"    Loaded scaling stats: {stats_id}")
    return stats


def get_model(model_id, seed, n_assets, ecfg=None):
    key = (model_id, seed)
    if key in _model_cache:
        return _model_cache[key]
    local_path = f"/tmp/UNIVERSAL_S{seed}_M{model_id}.h5"
    if not os.path.exists(local_path):
        blob_path = f"{GCS_MODEL_PATH}/UNIVERSAL_S{seed}_M{model_id}.h5"
        blob = gcs_client.bucket(BUCKET_NAME).blob(blob_path)
        if not blob.exists():
            return None
        blob.download_to_filename(local_path)
    builder = (ecfg or {}).get('model_builder', 'standard')
    embed_dim = (ecfg or {}).get('embed_dim', EMBED_DIM)
    if builder == 'cnn_transformer':
        model = build_cnn_transformer_model(n_assets, embed_dim=embed_dim)
    else:
        model = build_universal_model(n_assets, embed_dim=embed_dim)
    model.load_weights(local_path)
    _model_cache[key] = model
    return model


def build_market_context():
    print("  Building market context for TF scoring...")
    frames = []
    for ticker in CONTEXT_ASSETS:
        df = load_master(ticker)
        if df is None:
            continue
        df = heal_data(df.sort_index())
        daily_c = df['Close'].resample('D').last().ffill().shift(1).reindex(
            df.index, method='ffill')
        cols = {f'{ticker}_Anchor_ROC': (df['Close'] - daily_c) / daily_c}
        for h in ROC_HORIZONS:
            cols[f'{ticker}_ROC_{h}m'] = df['Close'].pct_change(h).fillna(0)
        frames.append(pd.DataFrame(cols, index=df.index))
        del df, daily_c; gc.collect()
    if not frames:
        raise RuntimeError("No context assets loaded")
    return pd.concat(frames, axis=1).ffill().fillna(0)


def engineer_features_scoring(df, market_df):
    daily_c = df['Close'].resample('D').last().ffill().shift(1).reindex(
        df.index, method='ffill')
    df['Gap_Pct'] = (df['Open'] - daily_c) / daily_c
    df['PM_High'] = df.groupby(df.index.date)['High'].cummax()
    df['Dist_to_PM_High'] = (df['Close'] - df['PM_High']) / (df['PM_High'] + 1e-9)
    df['RSI_14'] = ta.rsi(df['Close'], length=14)
    df['RSI_60'] = ta.rsi(df['Close'], length=60)
    vol_real = df['Volume'].where(df['Volume'] > 0)
    df['RVOL'] = df['Volume'] / vol_real.rolling(50, min_periods=5).mean().ffill().clip(lower=1e-9)
    real_mask = df['Volume'] > 0
    atr_raw = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['ATR_14'] = atr_raw.where(real_mask).ffill().fillna(0)
    df['Compression'] = (df['High'] - df['Low']) / (df['ATR_14'] + 1e-9)
    for h in ROC_HORIZONS:
        df[f'ROC_{h}m'] = df['Close'].pct_change(h)
    df = df.join(market_df, how='left').ffill().fillna(0)
    df_pst = df.index.tz_convert(PST) if str(df.index.tz) != 'America/Los_Angeles' else df.index
    mins = df_pst.hour * 60 + df_pst.minute
    df['Time_Pos'] = np.clip((mins - 390) / 390, 0, 1)
    df['Bar_Is_Real'] = (df['Volume'] > 0).astype(float)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


def apply_scaling_scoring(df, stats):
    df_s = df.copy()
    for col in ALL_FEATURE_COLS:
        if col in ('Time_Pos', 'Bar_Is_Real'):
            continue
        if col in stats:
            df_s[col] = ((df_s[col] - stats[col]['med'])
                         / max(stats[col]['iqr'], 1e-9)).clip(-10, 10).astype('float32')
    return df_s


def get_featured_data(ticker, market_df):
    if ticker in _feat_cache:
        return _feat_cache[ticker]
    df = load_master(ticker, start_date=TEST_START)
    if df is None:
        return None
    df = heal_data(df.sort_index())
    df = engineer_features_scoring(df, market_df)
    _feat_cache[ticker] = df
    return df


def score_ticker_live(ticker, cfg, df_scaled):
    entry_minute = cfg['entry_minute']
    prob_thresh = cfg['prob_thresh']
    seed_list = cfg['seed_list']

    base_exp = cfg['base_experiment']
    ecfg = EXP_CONFIGS.get(base_exp)
    if ecfg is None:
        for eid, ec in EXP_CONFIGS.items():
            if ec.get('stats_id') == cfg.get('stats_id'):
                ecfg = ec; break
    if ecfg is None:
        ecfg = EXP_CONFIGS['152_XSECTOR']

    n_assets = ecfg['n_assets']
    ticker_id = ecfg['asset_index'].get(ticker, ecfg['other_idx'])
    actual_model_id = ecfg.get('model_id_override', cfg.get('stats_id', base_exp))

    df_pst = df_scaled.copy()
    if str(df_pst.index.tz) != 'America/Los_Angeles':
        df_pst.index = df_pst.index.tz_convert(PST)
    times = df_pst.index.hour * 60 + df_pst.index.minute
    test_start_pst = pd.Timestamp(TEST_START, tz=PST)
    test_end_pst = pd.Timestamp(TEST_END, tz=PST) + pd.Timedelta(days=1)

    mask = ((df_pst.index >= test_start_pst) & (df_pst.index < test_end_pst) &
            (times == entry_minute) & (df_pst['Volume'] > 0))
    entry_locs = np.where(mask)[0]
    if len(entry_locs) == 0:
        return []

    XA_list, XC_list, valid_locs = [], [], []
    for loc in entry_locs:
        if loc < TIME_STEPS:
            continue
        window = df_pst.iloc[loc - TIME_STEPS:loc][ASSET_COLS].values.astype(np.float32)
        context = df_pst.iloc[loc][CONTEXT_COLS].values.astype(np.float32)
        XA_list.append(window)
        XC_list.append(context)
        valid_locs.append(loc)
    if not XA_list:
        return []

    XA = np.array(XA_list)
    XC = np.array(XC_list)
    XID = np.full((len(XA_list), 1), ticker_id, dtype=np.int32)

    all_probs = np.zeros(len(XA_list), dtype=np.float64)
    n_seeds = 0
    for seed in seed_list:
        model = get_model(actual_model_id, seed, n_assets, ecfg=ecfg)
        if model is None:
            continue
        preds = model.predict([XA, XC, XID], verbose=0, batch_size=256)
        all_probs += preds.flatten()
        n_seeds += 1
    if n_seeds == 0:
        return []
    all_probs /= n_seeds

    results = []
    for i, prob in enumerate(all_probs):
        if prob >= prob_thresh:
            results.append({'bar_ts': df_pst.index[valid_locs[i]], 'prob': float(prob)})
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
def run_backtest(configs, label, regime_dict, market_df):
    print(f"\n{'='*W}")
    print(f"  BACKTEST: {label} -- {len(configs)} tickers")
    print(f"{'='*W}")
    if not configs:
        print(f"  No configs -- skipping"); return pd.DataFrame(), pd.DataFrame()

    all_trades = []
    all_signals_log = []
    for cfg in configs:
        ticker = cfg['ticker']; mode = cfg['tpsl_mode']; em = cfg['entry_minute']
        tier_tag = cfg.get('tier', '')

        df_feat = get_featured_data(ticker, market_df)
        if df_feat is None:
            print(f"  {ticker}: SKIP (no data)"); continue
        stats = load_scaling_stats(cfg['stats_id'])
        df_scaled = apply_scaling_scoring(df_feat, stats)

        df_pst = df_scaled.copy()
        if str(df_pst.index.tz) != 'America/Los_Angeles':
            df_pst.index = df_pst.index.tz_convert(PST)

        signals = score_ticker_live(ticker, cfg, df_scaled)
        if not signals:
            print(f"  {ticker}: No signals"); gc.collect(); continue

        atr_series = (compute_daily_atr_series(df_pst) if mode in ('atr','trailing_decay','atr_sl')
                      else compute_adaptive_atr_series(df_pst) if mode == 'atr_adaptive' else None)
        vol_baselines = precompute_vol_baselines(df_pst)
        ohlcv = precompute_ohlcv_arrays(df_pst, TEST_START, TEST_END)
        if ohlcv is None: gc.collect(); continue

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
                    fv += 1
                    all_signals_log.append({'ticker': ticker, 'date_int': d_int, 'mode': mode,
                                           'prob': sig['prob'], 'status': 'vol_filtered'})
                    continue

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
                    fr += 1
                    all_signals_log.append({'ticker': ticker, 'date_int': d_int, 'mode': mode,
                                           'prob': sig['prob'], 'status': 'regime_filtered'})
                    continue

            if mode == 'trailing_decay':
                pnl, td, valid = run_single_trade_trailing_decay(ohlcv['prices'], ohlcv['highs'], ohlcv['lows'], ohlcv['dates'], ohlcv['times'], oi, atr_pct, cfg['trail_start_mult'], cfg['trail_end_mult'], cfg['sl_mult'], EXIT_TIME_MIN)
            elif mode == 'grace':
                pnl, td, valid = run_single_trade_grace(ohlcv['prices'], ohlcv['highs'], ohlcv['lows'], ohlcv['dates'], ohlcv['times'], oi, tp, sl, EXIT_TIME_MIN, cfg['grace_minutes'])
            elif mode == 'limit_entry':
                pnl, td, valid = run_single_trade_limit_entry(ohlcv['prices'], ohlcv['highs'], ohlcv['lows'], ohlcv['dates'], ohlcv['times'], oi, tp, sl, EXIT_TIME_MIN, cfg['dip_pct'], cfg['max_wait_minutes'])
                if not valid:
                    uf += 1
                    all_signals_log.append({'ticker': ticker, 'date_int': d_int, 'mode': mode,
                                           'prob': sig['prob'], 'status': 'unfilled'})
                    continue
            else:
                pnl, td, valid = run_single_trade(ohlcv['prices'], ohlcv['highs'], ohlcv['lows'], ohlcv['dates'], ohlcv['times'], oi, tp, sl, EXIT_TIME_MIN)

            if valid:
                pnl -= 2 * SLIPPAGE_PCT
                trades.append({'ticker': ticker, 'experiment': cfg['experiment'],
                               'date_int': int(td), 'pnl': float(pnl), 'prob': sig['prob'],
                               'tp': tp, 'sl': sl, 'mode': mode, 'tier': tier_tag,
                               'dip_pct': cfg.get('dip_pct', 0), 'max_wait': cfg.get('max_wait_minutes', 0)})
                all_signals_log.append({'ticker': ticker, 'date_int': int(td), 'mode': mode,
                                       'prob': sig['prob'], 'status': 'filled', 'pnl': float(pnl)})
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
        del ohlcv; gc.collect()

    if not all_trades:
        print(f"  No trades for {label}")
        return pd.DataFrame(), pd.DataFrame(all_signals_log) if all_signals_log else pd.DataFrame()
    tdf = pd.DataFrame(all_trades)
    tdf['date'] = pd.to_datetime(tdf['date_int'].astype(str), format='%Y%m%d')
    sig_df = pd.DataFrame(all_signals_log) if all_signals_log else pd.DataFrame()
    if not sig_df.empty:
        sig_df['date'] = pd.to_datetime(sig_df['date_int'].astype(str), format='%Y%m%d')
    return tdf, sig_df


# ==============================================================================
# ANALYTICS
# ==============================================================================
def per_asset_stats(trades_df):
    if trades_df.empty: return pd.DataFrame()
    months = max(1.0, (trades_df['date'].max() - trades_df['date'].min()).days / 30.44)
    rows = []
    for (ticker, exp), td in trades_df.groupby(['ticker', 'experiment']):
        n = len(td); w = int((td['pnl'] > 0).sum())
        rows.append({'ticker': ticker, 'experiment': exp,
                     'mode': td['mode'].iloc[0], 'tier': td['tier'].iloc[0],
                     'trades': n, 'wins': w, 'losses': n-w,
                     'win_rate': w/n, 'avg_pnl': td['pnl'].mean(),
                     'total_pnl': td['pnl'].sum(), 'freq': n/months})
    return pd.DataFrame(rows)


def portfolio_summary(trades_df, label):
    if trades_df.empty: return {}
    n = len(trades_df); w = int((trades_df['pnl'] > 0).sum())
    months = max(1.0, (trades_df['date'].max() - trades_df['date'].min()).days / 30.44)
    n_configs = trades_df.groupby(['ticker', 'experiment']).ngroups
    return {'label': label, 'tickers': trades_df['ticker'].nunique(), 'configs': n_configs,
            'trades': n, 'wins': w, 'losses': n-w, 'win_rate': w/n,
            'avg_pnl': trades_df['pnl'].mean(), 'total_pnl': trades_df['pnl'].sum(),
            'freq_per_mo': n/months}


# ==============================================================================
# RUN
# ==============================================================================
KEYS = ['B', 'E', 'G', 'H']

print(f"\n{'='*W}")
print(f"  FULL EXPLOIT INVESTIGATION — LIVE TF SCORING")
print(f"  TEST: {TEST_START} -> {TEST_END}  |  Model: {MODEL_VERSION}  |  Threshold: {DEPLOY_THRESHOLD:.1%}")
print(f"{'='*W}")

raw_df = load_raw_wf_data()
both_exp_tickers, any2_exp_tickers = find_both_exp_tickers(raw_df)

# ==============================================================================
# BQ DATA AUDIT — what's available but not being used?
# ==============================================================================
print(f"\n{'='*W}")
print(f"  BQ DATA AUDIT: Full landscape of qualifying configs")
print(f"{'='*W}")

_audit_sa = _build_seed_groups(raw_df)
_audit_sa = _audit_sa[(_audit_sa['n_seeds'] >= 3) & (_audit_sa['mean_va_pnl'] >= DEPLOY_THRESHOLD)]
_audit_sa['exp_fam'] = _audit_sa['experiment'].str[:3]

print(f"\n  Total qualifying configs (>= {DEPLOY_THRESHOLD:.1%}, >= 3 seeds): {len(_audit_sa)}")
print(f"  Unique tickers: {_audit_sa['ticker'].nunique()}")
print(f"  Unique modes: {sorted(_audit_sa['tpsl_mode'].unique())}")

print(f"\n  Per-ticker availability:")
print(f"  {'Ticker':<7} {'Families':<20} {'Modes':<40} {'N_cfg':>5} {'Best_PnL':>9}")
print(f"  {'-'*85}")
for tk in sorted(_audit_sa['ticker'].unique()):
    tk_data = _audit_sa[_audit_sa['ticker'] == tk]
    fams = sorted(tk_data['exp_fam'].unique())
    modes = sorted(tk_data['tpsl_mode'].unique())
    best_pnl = tk_data['mean_va_pnl'].max()
    fam_str = ','.join(fams)
    mode_str = ','.join(m[:6] for m in modes)
    print(f"  {tk:<7} {fam_str:<20} {mode_str:<40} {len(tk_data):>5} {best_pnl:>+8.2%}")

print(f"\n  Per-experiment family:")
for fam in sorted(_audit_sa['exp_fam'].unique()):
    fam_data = _audit_sa[_audit_sa['exp_fam'] == fam]
    label = {'152': '152 Late', '160': '160 Early', '170': '170 WF-f1',
             '175': '175 WF-deployed', '180': '180 WF-f1',
             '185': '185 WF-deployed', '200': '200 CNN-TX'}.get(fam, fam)
    print(f"    {label}: {fam_data['ticker'].nunique()} tickers, {len(fam_data)} configs, "
          f"avg PnL {fam_data['mean_va_pnl'].mean():.2%}")

del _audit_sa

configs_B = select_B_tier1plus(raw_df, both_exp_tickers)
configs_E = select_E_best_all(raw_df)
configs_G = select_G_exploit(raw_df)
configs_H = select_H_exploit_no_lunr(raw_df)

market_df = build_market_context()
regime_dict = preload_regime_data(TEST_START, TEST_END)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print(f"  GPU ACTIVE: {len(gpus)} GPU(s)")
else:
    print(f"  WARNING: No GPU — TF scoring will be slow")

all_methods = [('B', configs_B), ('E', configs_E), ('G', configs_G), ('H', configs_H)]

results = {}
stats = {}
summaries = {}
signal_logs = {}

import ctypes
_libc = ctypes.CDLL("libc.so.6")

for key, cfgs in all_methods:
    label = f"{key}: {METHOD_LABELS[key]}"
    trades, sig_log = run_backtest(cfgs, label, regime_dict, market_df)
    results[key] = trades
    stats[key] = per_asset_stats(trades)
    summaries[key] = portfolio_summary(trades, METHOD_LABELS[key])
    signal_logs[key] = sig_log
    _model_cache.clear()
    _feat_cache.clear()
    _stats_cache.clear()
    gc.collect()
    _libc.malloc_trim(0)


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
print(f"  {'-'*(16 + 19*len(KEYS))}")

metrics = ['tickers', 'configs', 'trades', 'wins', 'win_rate', 'avg_pnl', 'total_pnl', 'freq_per_mo']
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
print(f"  {'-'*(7 + 11*len(KEYS) + 7)}")

method_wins = {k: 0 for k in KEYS}
for ticker in all_tickers:
    print(f"  {ticker:<7}", end='')
    best_key = None; best_pnl = -999
    for key in KEYS:
        s = stats[key]
        match = s[s['ticker'] == ticker] if not s.empty else pd.DataFrame()
        if not match.empty:
            total_trades = match['trades'].sum()
            v = float((match['avg_pnl'] * match['trades']).sum() / total_trades) if total_trades > 0 else 0
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
print(f"  {'-'*(10 + 13*len(KEYS))}")

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
fig.suptitle(f'Full Exploit Comparison (B/E/G/H) — Live TF Scoring\n'
             f'TEST ({TEST_START} -> {TEST_END})', fontsize=14, fontweight='bold')

ax1 = axes[0]
bar_data = []
for ticker in all_tickers:
    row = {'ticker': ticker}
    for key in KEYS:
        s = stats[key]
        m = s[s['ticker'] == ticker] if not s.empty else pd.DataFrame()
        if not m.empty:
            tot_t = m['trades'].sum()
            row[key] = float((m['avg_pnl'] * m['trades']).sum() / tot_t * 100) if tot_t > 0 else 0
        else:
            row[key] = 0
    bar_data.append(row)
sort_key = 'G' if 'G' in KEYS else KEYS[0]
bdf = pd.DataFrame(bar_data).sort_values(sort_key, ascending=True)
y_pos = np.arange(len(bdf))
n_keys = len(KEYS)
bar_h = 0.8 / n_keys
for i, key in enumerate(KEYS):
    ax1.barh(y_pos + (i - (n_keys-1)/2) * bar_h, bdf[key].values, bar_h,
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
print(f"\n  {'#':>2} {'Method':<20} {'Tickers':>8} {'Cfgs':>5} {'Trades':>7} {'WR':>6} {'AvgPnL':>8} {'TotPnL':>9} {'Freq/mo':>8}")
print(f"  {'-'*80}")
for i, (_, r) in enumerate(ranking_df.iterrows(), 1):
    print(f"  {i:>2} {r['key']+': '+r['label']:<20} {r['tickers']:>8} {r.get('configs', r['tickers']):>5} "
          f"{r['trades']:>7} {r['win_rate']:>5.0%} {r['avg_pnl']:>+7.2%} "
          f"{r['total_pnl']:>+8.2%} {r['freq_per_mo']:>7.1f}")

best = ranking_df.iloc[0]
print(f"\n  WINNER: {best['key']}: {best['label']} — {best['tickers']} tickers, "
      f"avg {best['avg_pnl']:+.2%}/trade, {best['win_rate']:.0%} WR, total {best['total_pnl']:+.2%}")
print(f"{'='*W}")


# ==============================================================================
# COVERAGE ANALYSIS: exploit the data
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  COVERAGE ANALYSIS")
print(f"{'='*W}")

# --- 1. Per-experiment family monthly PnL (which models hold up in June?) ---
print(f"\n  --- Per-Experiment Family Monthly PnL (strategy G) ---")
g_trades = results.get('G', pd.DataFrame())
if not g_trades.empty:
    g_trades['month'] = g_trades['date'].dt.to_period('M')
    g_trades['exp_fam'] = g_trades['experiment'].str[:3]
    months_in_data = sorted(g_trades['month'].unique())

    print(f"\n  {'Exp_Family':<12}", end='')
    for m in months_in_data:
        print(f" {str(m):>10} {'N':>4}", end='')
    print(f" {'TOTAL':>8} {'N':>4}")
    print(f"  {'-'*(12 + 15*len(months_in_data) + 13)}")

    for fam in sorted(g_trades['exp_fam'].unique()):
        fam_trades = g_trades[g_trades['exp_fam'] == fam]
        label = {'152': '152 Late', '160': '160 Early', '170': '170 WF-f1',
                 '175': '175 WF-dep', '180': '180 WF-f1',
                 '185': '185 WF-dep', '200': '200 CNN-TX'}.get(fam, fam)
        print(f"  {label:<12}", end='')
        for m in months_in_data:
            mt = fam_trades[fam_trades['month'] == m]
            if len(mt) == 0:
                print(f" {'---':>10} {'':>4}", end='')
            else:
                print(f" {mt['pnl'].mean():>+9.2%} {len(mt):>4}", end='')
        print(f" {fam_trades['pnl'].mean():>+7.2%} {len(fam_trades):>4}")

# --- 2. Per-mode monthly PnL (which modes work?) ---
print(f"\n\n  --- Per-Mode Monthly PnL (strategy G) ---")
if not g_trades.empty:
    print(f"\n  {'Mode':<16}", end='')
    for m in months_in_data:
        print(f" {str(m):>10} {'N':>4}", end='')
    print(f" {'TOTAL':>8} {'N':>4}")
    print(f"  {'-'*(16 + 15*len(months_in_data) + 13)}")

    for mode in sorted(g_trades['mode'].unique()):
        mode_trades = g_trades[g_trades['mode'] == mode]
        print(f"  {mode:<16}", end='')
        for m in months_in_data:
            mt = mode_trades[mode_trades['month'] == m]
            if len(mt) == 0:
                print(f" {'---':>10} {'':>4}", end='')
            else:
                print(f" {mt['pnl'].mean():>+9.2%} {len(mt):>4}", end='')
        print(f" {mode_trades['pnl'].mean():>+7.2%} {len(mode_trades):>4}")

# --- 3. WF PnL vs Actual PnL correlation ---
print(f"\n\n  --- WF Predicted vs Actual PnL (does selection metric predict forward?) ---")
for key in KEYS:
    s = stats.get(key, pd.DataFrame())
    if s.empty or len(s) < 3:
        continue
    all_cfgs = all_methods[[k for k, _ in all_methods].index(key)][1]
    cfg_by_tk = {c['ticker']: c for c in all_cfgs}
    rows = []
    for _, row in s.iterrows():
        tk = row['ticker']
        if tk in cfg_by_tk:
            c = cfg_by_tk[tk]
            rows.append({'ticker': tk, 'wf_pnl': c.get('wf_avg_pnl', 0),
                         'actual_pnl': row['avg_pnl'], 'trades': row['trades'],
                         'mode': row['mode']})
    if rows:
        cdf = pd.DataFrame(rows).sort_values('wf_pnl', ascending=False)
        corr = cdf['wf_pnl'].corr(cdf['actual_pnl'])
        print(f"\n  Strategy {key} ({METHOD_LABELS[key]}): WF↔Actual correlation = {corr:+.2f}")
        print(f"  {'Ticker':<7} {'WF_PnL':>8} {'Actual':>8} {'N':>4} {'Mode':<14} {'Match':>6}")
        print(f"  {'-'*55}")
        for _, r in cdf.iterrows():
            match = 'OK' if r['actual_pnl'] >= DEPLOY_THRESHOLD else 'MISS'
            print(f"  {r['ticker']:<7} {r['wf_pnl']:>+7.2%} {r['actual_pnl']:>+7.2%} {r['trades']:>4} "
                  f"{r['mode']:<14} {match:>6}")

# --- 4. G-only tickers: what did broader coverage find? ---
print(f"\n\n  --- G-Only Tickers: Configs that B/E missed ---")
g_stats = stats.get('G', pd.DataFrame())
e_stats = stats.get('E', pd.DataFrame())
b_stats = stats.get('B', pd.DataFrame())
if not g_stats.empty:
    e_tickers = set(e_stats['ticker']) if not e_stats.empty else set()
    b_tickers = set(b_stats['ticker']) if not b_stats.empty else set()
    be_tickers = b_tickers | e_tickers
    g_only = sorted(set(g_stats['ticker']) - be_tickers)
    g_also = sorted(set(g_stats['ticker']) & be_tickers)

    if g_only:
        print(f"\n  NEW tickers found by G ({len(g_only)}):")
        for tk in g_only:
            tk_row = g_stats[g_stats['ticker'] == tk]
            for _, r in tk_row.iterrows():
                print(f"    {tk:<7} {r['experiment']:<25} {r['mode']:<14} "
                      f"{r['trades']:>3} trades  avg {r['avg_pnl']:>+7.2%}  total {r['total_pnl']:>+7.2%}")

    if g_also:
        print(f"\n  DUPLICATE configs in G (also in B or E) with different experiment:")
        for tk in g_also:
            g_rows = g_stats[g_stats['ticker'] == tk]
            if len(g_rows) > 1:
                for _, r in g_rows.iterrows():
                    in_be = 'B' if tk in b_tickers else ('E' if tk in e_tickers else '')
                    print(f"    {tk:<7} {r['experiment']:<25} {r['mode']:<14} "
                          f"avg {r['avg_pnl']:>+7.2%}  ({in_be})")

# --- 5. Signal fill rate by month ---
print(f"\n\n  --- Signal Fill Rate by Month ---")
for key in KEYS:
    sl = signal_logs.get(key, pd.DataFrame())
    if sl.empty:
        continue
    sl['month'] = sl['date'].dt.to_period('M')
    print(f"\n  Strategy {key} ({METHOD_LABELS[key]}):")
    months_sl = sorted(sl['month'].unique())
    print(f"  {'Month':<10} {'signals':>8} {'filled':>8} {'unfilled':>9} {'vol_f':>6} {'reg_f':>6} {'fill_%':>7}")
    print(f"  {'-'*60}")
    for m in months_sl:
        ms = sl[sl['month'] == m]
        n_total = len(ms)
        n_filled = len(ms[ms['status'] == 'filled'])
        n_unfilled = len(ms[ms['status'] == 'unfilled'])
        n_vol = len(ms[ms['status'] == 'vol_filtered'])
        n_reg = len(ms[ms['status'] == 'regime_filtered'])
        fill_pct = n_filled / n_total * 100 if n_total > 0 else 0
        print(f"  {str(m):<10} {n_total:>8} {n_filled:>8} {n_unfilled:>9} {n_vol:>6} {n_reg:>6} {fill_pct:>6.1f}%")

# --- 6. June drill-down for G ---
print(f"\n\n  --- June Drill-Down: Strategy G ---")
if not g_trades.empty:
    june = g_trades[g_trades['date'].dt.month == 6].sort_values('date')
    if not june.empty:
        jw = int((june['pnl'] > 0).sum()); jl = len(june) - jw
        print(f"\n  G (full_exploit) — {len(june)} June trades ({jw}W/{jl}L), avg {june['pnl'].mean():+.2%}:")
        print(f"  {'Date':<12} {'Ticker':<7} {'Experiment':<25} {'Mode':<14} {'PnL':>8} {'Prob':>6} {'W/L':>4}")
        print(f"  {'-'*82}")
        for _, t in june.iterrows():
            wl = 'W' if t['pnl'] > 0 else 'L'
            print(f"  {t['date'].strftime('%Y-%m-%d'):<12} {t['ticker']:<7} {t['experiment']:<25} "
                  f"{t['mode']:<14} {t['pnl']:>+7.2%} {t['prob']:>5.2f} {wl:>4}")

print(f"\n{'='*W}")
print(f"  ANALYSIS COMPLETE")
print(f"{'='*W}")
