"""
backtest_s3_momentum_live.py — Live TF Backtest: S3 STATIC vs S3 MOMENTUM 10d
================================================================================
Backtests S3 STATIC vs S3 MOMENTUM 10d over TEST_START→TEST_END using live
TF model inference (15-seed ensemble) at every entry bar on every trading day.

Strategy S3 STATIC (benchmark):
  BQ family-agreement gate, one experiment per ticker, limit_entry trade sim.
  Scores via 15-seed ensemble TF inference at entry_minute bar.

Strategy S3 MOMENTUM 10d:
  Same as S3 STATIC but skips ticker if 10-trading-day price return <= 0.

GPU required. Colab-ready (no if __name__).
"""

import os
import sys
import json
import gc
import subprocess
import ctypes
import warnings
import io
from collections import defaultdict

warnings.filterwarnings('ignore')

os.environ['TF_USE_LEGACY_KERAS'] = '1'

import numpy as np
import pandas as pd
import pytz
from datetime import datetime, timedelta

try:
    import tf_keras
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tf_keras", "--quiet"])

try:
    import pandas_ta as ta
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas_ta", "--no-deps", "--quiet"])
    import pandas_ta as ta

import tensorflow as tf

from tensorflow.keras import mixed_precision
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, GlobalAveragePooling1D,
    LayerNormalization, MultiHeadAttention, Add,
    Concatenate, Embedding, Flatten,
    Conv1D, MaxPool1D, SpatialDropout1D,
)
from tensorflow.keras.regularizers import l2
from google.cloud import storage as gcs_storage

import matplotlib
try:
    from google.colab import auth
    _IN_COLAB = True
except ImportError:
    matplotlib.use('Agg')
    _IN_COLAB = False
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

if tf.config.list_physical_devices('GPU'):
    mixed_precision.set_global_policy('mixed_float16')

try:
    from google.cloud import bigquery
except Exception:
    bigquery = None

if _IN_COLAB:
    try:
        auth.authenticate_user()
    except Exception:
        pass

def _release_memory():
    gc.collect()
    try: ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception: pass

# ==============================================================================
# SHARED CONSTANTS
# ==============================================================================
BUCKET_NAME    = os.environ.get("GCS_BUCKET", "")
GCS_MODEL_PATH = "Algos/to_deploy/Saved_models_v2/UNIVERSAL/Direction"
INFERENCE_DIR  = "Data/Inference_v2"
TIME_STEPS     = 240
EMBED_DIM      = 16
PST            = pytz.timezone('America/Los_Angeles')
MIN_PRICE          = 2.00

# ==============================================================================
# BACKTEST CONFIG
# ==============================================================================
TEST_START = '2026-04-01'
TEST_END   = '2026-07-11'
EXIT_TIME_MIN = 780
SLIPPAGE_PCT = 0.0015
DEPLOY_THRESHOLD = 0.018
MOMENTUM_LOOKBACK_DAYS = 10
STARTING_CAPITAL = 200_000

TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

MICRO_COLS = ['Spread', 'OBI', 'Volume_Delta', 'Trade_Count', 'Delta_Pct',
              'bsize', 'asize', 'Total_Vol']

ROC_HORIZONS = [5, 30, 60, 120]

ASSET_COLS = (['Gap_Pct', 'RSI_14', 'RSI_60', 'RVOL', 'Compression',
               'ATR_14', 'Dist_to_PM_High', 'Bar_Is_Real'] +
              [f'ROC_{h}m' for h in ROC_HORIZONS] + MICRO_COLS)

# ==============================================================================
# EXPERIMENT DEFINITIONS
# ==============================================================================
def _build_context_cols(context_assets, has_smh_rsi):
    cols = []
    for t in context_assets:
        cols.append(f'{t}_Anchor_ROC')
        for h in ROC_HORIZONS:
            cols.append(f'{t}_ROC_{h}m')
    if has_smh_rsi:
        cols.append('SMH_RSI')
    cols.append('Time_Pos')
    return cols



EXP_152 = {
    'model_id': '152_XSECTOR',
    'label': 'EXP-152 XSECTOR',
    'train_teachers': [
        'BBAI', 'RCAT', 'LUNR', 'ASTS', 'ACHR', 'MVIS', 'WULF',
        'CIFR', 'IREN', 'KEEL', 'HUT', 'CLSK', 'APLD',
    ],
    'context_assets': ['SPY', 'VIX', 'TLT', 'IWM'],
    'has_smh_rsi': False,
    'asset_configs': {},
    'asset_cols': ASSET_COLS,
}

EXP_160 = {
    'model_id': '160_XSECTOR',
    'label': 'EXP-160 XSECTOR (early cutoff)',
    'train_teachers': [
        'BBAI', 'RCAT', 'LUNR', 'ASTS', 'ACHR', 'MVIS', 'WULF',
        'CIFR', 'IREN', 'KEEL', 'HUT', 'CLSK', 'APLD',
    ],
    'context_assets': ['SPY', 'VIX', 'TLT', 'IWM'],
    'has_smh_rsi': False,
    'asset_configs': {},
    'asset_cols': ASSET_COLS,
}

# ==============================================================================
# BQ CONFIG SELECTION — Strategy S3 (limit_entry, majority-positive val_b)
# Gate: majority of families positive val_b, n_fams >= 2. Mode: limit_entry.
# Deploy: best config by score_pnl2 per ticker across scorable families.
# ==============================================================================
WF_MODEL_VERSION = 'robust_wf_v12'
WF_BQ_TABLE = f"{GCP_PROJECT}.scoring_logs.wf_optimization_results"
WF_MIN_TRADES = 3
WF_DEPLOY_THRESHOLD = 0.018
WF_MIN_SEEDS = 3
WF_MAX_VA_OPT_RATIO = 3.0
WF_DEPLOY_EXPERIMENTS = {'152_XSECTOR', '160_XSECTOR', '170_XSECTOR',
                         '175_XSECTOR',
                         '180_XSECTOR', '181_XSECTOR', '182_XSECTOR',
                         '183_XSECTOR', '184_XSECTOR', '185_XSECTOR',
                         '200_XSECTOR', '201_XSECTOR', '202_XSECTOR',
                         '203_XSECTOR', '204_XSECTOR', '205_XSECTOR',
                         '220_XSECTOR', '221_XSECTOR', '222_XSECTOR',
                         '223_XSECTOR', '224_XSECTOR', '225_XSECTOR'}
WF_EXCLUDE_TICKERS = {'LUNR'}
WF_TICKER_RENAME = {'MIGI': 'BGDE', 'BITF': 'KEEL'}
WF_MODE_SUFFIX = {'static': '', 'atr': '_ATR', 'atr_adaptive': '_ATRADP',
                   'regime_scaled': '_REGIME', 'trailing_decay': '_TRAIL',
                   'regime_adaptive': '_REGADP', 'atr_sl': '_ATRSL',
                   'grace': '_GRACE', 'limit_entry': '_LIMIT'}
WF_GRP_KEYS = ['ticker', 'experiment', 'tpsl_mode', 'entry_minute',
               'prob_thresh', 'tp_pct', 'sl_pct',
               'tp_mult', 'sl_mult', 'trail_start_mult', 'trail_end_mult',
               'grace_minutes', 'dip_pct', 'max_wait_minutes',
               'vol_filter_pct']

def _wf_base_exp(exp_name):
    for sfx in sorted(WF_MODE_SUFFIX.values(), key=len, reverse=True):
        if sfx and exp_name.endswith(sfx):
            return exp_name[:-len(sfx)]
    return exp_name

def _assign_family(exp_name):
    base = exp_name.split('_')[0]
    if not base.isdigit():
        return 'other'
    num = int(base)
    if num == 152: return '152'
    if num == 160: return '160'
    if 170 <= num <= 175: return '170'
    if 180 <= num <= 185: return '180'
    if 200 <= num <= 205: return '200'
    if 220 <= num <= 225: return '220'
    return 'other'

def load_deploy_configs_from_bq():
    """Strategy S3: limit_entry, majority-positive val_b across families, best per ticker."""
    bq_client = bigquery.Client()
    query = f"""
    SELECT ticker, experiment, seed, tpsl_mode, entry_minute, prob_thresh,
           tp_pct, sl_pct, tp_mult, sl_mult,
           trail_start_mult, trail_end_mult,
           vix_threshold, tlt_crash, spy_ignore, vol_filter_pct,
           val_a_avg_pnl, val_a_win_rate, val_a_count, val_a_freq,
           val_b_avg_pnl,
           opt_avg_pnl, is_ensemble,
           grace_minutes, dip_pct, max_wait_minutes
    FROM `{WF_BQ_TABLE}`
    WHERE model_version IN ('robust_wf_v12', 'robust_wf_v12_stitched')
      AND survived_a = TRUE
      AND val_a_count >= {WF_MIN_TRADES}
      AND val_a_avg_pnl > 0
    """
    df = bq_client.query(query).to_dataframe()
    # (BQ loading details suppressed for clean output)

    df['ticker'] = df['ticker'].replace(WF_TICKER_RENAME)
    df['base_experiment'] = df['experiment'].apply(_wf_base_exp)
    df = df[df['base_experiment'].isin(WF_DEPLOY_EXPERIMENTS)]
    df = df[~df['ticker'].isin(WF_EXCLUDE_TICKERS)]

    for col in ['grace_minutes', 'max_wait_minutes', 'vol_filter_pct']:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0).astype(int)
    for col in ['dip_pct', 'tp_mult', 'sl_mult', 'trail_start_mult', 'trail_end_mult']:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(0.0)

    before_n = len(df)
    df = df[df['opt_avg_pnl'] > 0].copy()
    df = df[df['val_a_avg_pnl'] / df['opt_avg_pnl'] <= WF_MAX_VA_OPT_RATIO]
    pass  # overfit filter applied

    # ── seed groups (individual seeds only) ──
    indiv = df[df['is_ensemble'] == False].copy()

    extra_agg = {}
    _regime_agg = {'vix_threshold': 'max', 'tlt_crash': 'min', 'spy_ignore': 'max'}
    for c in ['vix_threshold', 'tlt_crash', 'spy_ignore']:
        if c in indiv.columns:
            extra_agg[c] = (c, _regime_agg[c])

    sa = indiv.groupby(WF_GRP_KEYS).agg(
        n_seeds=('seed', 'nunique'),
        n_pos_seeds_18=('val_a_avg_pnl', lambda x: (x >= 0.018).sum()),
        mean_va_pnl=('val_a_avg_pnl', 'mean'),
        mean_va_wr=('val_a_win_rate', 'mean'),
        mean_va_count=('val_a_count', 'mean'),
        mean_va_freq=('val_a_freq', 'mean'),
        vb_pnl=('val_b_avg_pnl', 'mean'),
        **extra_agg,
    ).reset_index()
    sa['agree_18'] = (sa['n_pos_seeds_18'] / sa['n_seeds']).clip(upper=1.0)

    # ── Strategy S3: majority-positive val_b across families ──
    # (mirrors backtest_strategies.py: sa_base → sa_limit → _compute_family_agreement_bt → _select_best_bt)
    sa_limit = sa[
        (sa['n_seeds'] >= WF_MIN_SEEDS) &
        (sa['tpsl_mode'] == 'limit_entry')
    ].copy()

    # Family agreement gate — threshold applied AFTER best-per-family selection
    d_fam = sa_limit.copy()
    d_fam['score'] = (d_fam['mean_va_pnl'] ** 2) * d_fam['agree_18']
    d_fam = d_fam[np.isfinite(d_fam['score'])]
    d_fam['exp_fam'] = d_fam['experiment'].apply(_assign_family)
    d_fam = d_fam[d_fam['exp_fam'] != 'other']
    best_per_fam = d_fam.loc[d_fam.groupby(['ticker', 'exp_fam'])['score'].idxmax()]
    best_per_fam = best_per_fam[best_per_fam['mean_va_pnl'] >= WF_DEPLOY_THRESHOLD]

    agreement = {}
    for tk in best_per_fam['ticker'].unique():
        tk_rows = best_per_fam[best_per_fam['ticker'] == tk]
        n_fams = len(tk_rows)
        n_pos = int((tk_rows['vb_pnl'] > 0).sum())
        agreement[tk] = {
            'n_fams': n_fams, 'n_pos': n_pos,
            'majority_pos': n_pos > n_fams / 2 and n_fams >= 2,
        }

    maj_tickers = {tk for tk, ag in agreement.items() if ag['majority_pos']}

    # Select best config per ticker — only from scorable experiments
    scorable_exps = set(_EXP_MAP.keys())
    sa_s3 = sa_limit[
        sa_limit['ticker'].isin(maj_tickers) &
        sa_limit['experiment'].apply(_wf_base_exp).isin(scorable_exps)
    ].copy()
    dropped = maj_tickers - set(sa_s3['ticker'].unique())
    for tk in sorted(dropped):
        print(f"  WARNING: {tk} passes S3 gate but has no scorable configs (best may be from unsupported experiment)")
    sa_s3['score'] = (sa_s3['mean_va_pnl'] ** 2) * sa_s3['agree_18']
    best = sa_s3.loc[sa_s3.groupby('ticker')['score'].idxmax()] if not sa_s3.empty else sa_s3
    best = best[best['mean_va_pnl'] >= WF_DEPLOY_THRESHOLD].copy()

    # ── populate experiment asset_configs ──
    n_selected = 0
    for _, row in best.sort_values(['ticker', 'experiment']).iterrows():
        base_exp = _wf_base_exp(row['experiment'])
        exp_dict = _EXP_MAP.get(base_exp)
        ticker = row['ticker']
        tpsl_mode = row['tpsl_mode']
        n_seeds = int(row['n_seeds'])
        cfg = {
            'seed': -1, 'is_ensemble': True,
            'prob_threshold': float(row['prob_thresh']),
            'entry_minute': int(row['entry_minute']),
            'tpsl_mode': tpsl_mode,
            'tp_pct': float(row['tp_pct']),
            'sl_pct': float(row['sl_pct']),
            'vix_threshold': float(row.get('vix_threshold', 999)),
            'tlt_crash': float(row.get('tlt_crash', -999)),
            'spy_ignore': float(row.get('spy_ignore', 999)),
            'vol_filter_pct': int(row.get('vol_filter_pct', 0)),
            'confidence': 'HIGH' if n_seeds >= 7 else ('MED' if n_seeds >= 4 else 'LOW'),
            'confidence_note': f'{n_seeds} seeds, agree={row["agree_18"]:.0%}',
            'stats': {
                'win_rate': f'{row["mean_va_wr"]:.0%}',
                'avg_pnl': f'{row["mean_va_pnl"]:.1%}',
                'freq': f'{row["mean_va_freq"]:.1f}',
            },
        }
        if tpsl_mode == 'limit_entry':
            cfg['dip_pct'] = float(row.get('dip_pct', 0))
            cfg['max_wait_minutes'] = int(row.get('max_wait_minutes', 0))
        if tpsl_mode == 'grace':
            cfg['grace_minutes'] = int(row.get('grace_minutes', 0))
        if tpsl_mode in ('atr', 'atr_adaptive', 'trailing_decay', 'atr_sl'):
            cfg['tp_mult'] = float(row.get('tp_mult', 0))
            cfg['sl_mult'] = float(row.get('sl_mult', 0))
        if tpsl_mode == 'trailing_decay':
            cfg['trail_start_mult'] = float(row.get('trail_start_mult', 0))
            cfg['trail_end_mult'] = float(row.get('trail_end_mult', 0))
        exp_dict['asset_configs'][ticker] = cfg
        n_selected += 1

    pass  # n_selected configs loaded
    return n_selected


EXP_170 = {
    'model_id': '175_XSECTOR',
    'label': 'EXP-170 XSECTOR (walkforward)',
    'train_teachers': [
        'BBAI', 'RCAT', 'LUNR', 'ASTS', 'ACHR', 'MVIS', 'WULF',
        'CIFR', 'IREN', 'KEEL', 'HUT', 'CLSK', 'APLD',
    ],
    'context_assets': ['SPY', 'VIX', 'TLT', 'IWM'],
    'has_smh_rsi': False,
    'asset_configs': {
        # ── Tickers moved to limit_entry in EXP_152/160: ACHR, ASTS, BTBT, CLSK, WULF ──
        'ALAB': {
            'seed': 17507, 'prob_threshold': 0.450, 'entry_minute': 360,
            'tpsl_mode': 'static',
            'tp_pct': 0.12, 'sl_pct': 0.03,
            'vix_threshold': 20.0, 'tlt_crash': 0.0, 'spy_ignore': 0.0,
            'vol_filter_pct': 0,
            'confidence': 'MED', 'confidence_note': 'N=8 trades',
            'stats': {'win_rate': '62%', 'avg_pnl': '3.1%', 'composite': '0.0021'},
        },
        'BE': {
            'seed': 17500, 'prob_threshold': 0.550, 'entry_minute': 360,
            'tpsl_mode': 'static',
            'tp_pct': 0.25, 'sl_pct': 0.14,
            'vix_threshold': 20.0, 'tlt_crash': 0.0, 'spy_ignore': 0.0,
            'vol_filter_pct': 0,
            'confidence': 'HIGH', 'confidence_note': 'N=11 trades',
            'stats': {'win_rate': '64%', 'avg_pnl': '2.0%', 'composite': '0.0005'},
        },
        'OKLO': {
            'seed': 17501, 'prob_threshold': 0.500, 'entry_minute': 370,
            'tpsl_mode': 'static',
            'tp_pct': 0.12, 'sl_pct': 0.05,
            'vix_threshold': 20.0, 'tlt_crash': 0.0, 'spy_ignore': 999.0,
            'vol_filter_pct': 0,
            'confidence': 'LOW', 'confidence_note': 'N=3 trades',
            'stats': {'win_rate': '100%', 'avg_pnl': '5.4%', 'composite': '0.0005'},
        },
        'RKLB': {
            'seed': 17514, 'prob_threshold': 0.750, 'entry_minute': 375,
            'tpsl_mode': 'static',
            'tp_pct': 0.08, 'sl_pct': 0.08,
            'vix_threshold': 20.0, 'tlt_crash': 0.0, 'spy_ignore': 0.0,
            'vol_filter_pct': 0,
            'confidence': 'MED', 'confidence_note': 'N=8 trades',
            'stats': {'win_rate': '75%', 'avg_pnl': '2.5%', 'composite': '0.0006'},
        },
    },
    'asset_cols': ASSET_COLS,
}

TEACHERS_31 = [
    'CIFR', 'IREN', 'KEEL', 'HUT', 'APLD', 'WULF', 'RIOT', 'CLSK', 'BTBT', 'ALAB', 'MARA',
    'RCAT', 'ASTS', 'LUNR', 'BBAI', 'RKLB', 'SPIR', 'BKSY', 'MNTS', 'PLUG', 'BE', 'FCEL', 'FLNC',
    'OKLO', 'SMR', 'MVIS', 'BGDE', 'RDW', 'PL', 'VOYG', 'SMLR',
]
TEACHERS_47 = TEACHERS_31 + [
    'AMD', 'NVDA', 'TSLA', 'PLTR', 'SOUN', 'HOOD', 'SMCI', 'VRT', 'NBIS',
    'KTOS', 'AVAV', 'ACHR', 'JOBY', 'NNDM', 'CCJ', 'CEG',
]

EXP_185 = {
    'model_id': '185_XSECTOR',
    'label': 'EXP-185 XSECTOR (31 teachers)',
    'train_teachers': TEACHERS_31,
    'context_assets': ['SPY', 'VIX', 'TLT', 'IWM'],
    'has_smh_rsi': False,
    'asset_configs': {
        # ── RIOT moved to limit_entry in EXP_160 ──
        'PL': {
            'seed': 18507, 'prob_threshold': 0.500, 'entry_minute': 360,
            'tpsl_mode': 'static',
            'tp_pct': 0.06, 'sl_pct': 0.03,
            'vix_threshold': 999.0, 'tlt_crash': -999.0, 'spy_ignore': 999.0,
            'vol_filter_pct': 0,
            'confidence': 'LOW', 'confidence_note': 'N=3 trades',
            'stats': {'win_rate': '67%', 'avg_pnl': '2.7%', 'composite': '0.0005'},
        },
        'RDW': {
            'seed': 18503, 'prob_threshold': 0.600, 'entry_minute': 370,
            'tpsl_mode': 'atr',
            'tp_pct': 0.0, 'sl_pct': 0.0,
            'tp_mult': 2.0, 'sl_mult': 1.0,
            'vix_threshold': 999.0, 'tlt_crash': -999.0, 'spy_ignore': 999.0,
            'vol_filter_pct': 0,
            'confidence': 'MED', 'confidence_note': 'N=8 trades',
            'stats': {'win_rate': '50%', 'avg_pnl': '2.5%', 'composite': '0.0004'},
        },
        'SMCI': {
            'seed': -1, 'is_ensemble': True, 'prob_threshold': 0.500, 'entry_minute': 360,
            'tpsl_mode': 'static',
            'tp_pct': 0.06, 'sl_pct': 0.06,
            'vix_threshold': 999.0, 'tlt_crash': -999.0, 'spy_ignore': 999.0,
            'vol_filter_pct': 0,
            'confidence': 'MED', 'confidence_note': 'N=5 trades',
            'stats': {'win_rate': '80%', 'avg_pnl': '2.7%', 'composite': '0.0014'},
        },
    },
    'asset_cols': ASSET_COLS,
}

# ── EXP-200 family (same teachers as 185, per-fold models) ──
_EXP_200_FOLDS = {}
for _fold_id in ['200', '201', '202', '203', '204', '205']:
    _EXP_200_FOLDS[_fold_id] = {
        'model_id': f'{_fold_id}_XSECTOR',
        'label': f'EXP-{_fold_id} XSECTOR (200 family)',
        'train_teachers': TEACHERS_31,
        'context_assets': ['SPY', 'VIX', 'TLT', 'IWM'],
        'has_smh_rsi': False,
        'model_builder': 'cnn_transformer',
        'embed_dim': 32,
        'asset_configs': {},
        'asset_cols': ASSET_COLS,
    }

# ── EXP-220 family (47 teachers, stitched) ──
_EXP_220_FOLDS = {}
for _fold_id in ['220', '221', '222', '223', '224', '225']:
    _EXP_220_FOLDS[_fold_id] = {
        'model_id': f'{_fold_id}_XSECTOR',
        'label': f'EXP-{_fold_id} XSECTOR (220 family)',
        'train_teachers': TEACHERS_47,
        'context_assets': ['SPY', 'VIX', 'TLT', 'IWM'],
        'has_smh_rsi': False,
        'asset_configs': {},
        'asset_cols': ASSET_COLS,
    }

_ALL_EXPS = [EXP_152, EXP_160, EXP_170, EXP_185] + list(_EXP_200_FOLDS.values()) + list(_EXP_220_FOLDS.values())

for _exp in _ALL_EXPS:
    _acols = _exp.get('asset_cols', ASSET_COLS)
    _exp['asset_cols'] = _acols
    _exp['context_cols'] = _build_context_cols(
        _exp['context_assets'], _exp['has_smh_rsi'])
    _exp['all_cols'] = _acols + _exp['context_cols']
    _idx = {t: i for i, t in enumerate(_exp['train_teachers'])}
    if 'BITF' in _idx:
        _idx['KEEL'] = _idx['BITF']
    if 'MIGI' in _idx:
        _idx['BGDE'] = _idx['MIGI']
    _exp['asset_index'] = _idx
    _exp['n_assets'] = len(_exp['train_teachers']) + 1
    _exp['other_idx'] = len(_exp['train_teachers'])
    _exp['stats_path'] = f"Data/Inference_v2/scaling_stats_experiment_{_exp['model_id']}.json"

_EXP_MAP = {
    '152_XSECTOR': EXP_152,
    '160_XSECTOR': EXP_160,
    '170_XSECTOR': EXP_170,
    '175_XSECTOR': EXP_170,
    '180_XSECTOR': EXP_185,
    '181_XSECTOR': EXP_185,
    '182_XSECTOR': EXP_185,
    '183_XSECTOR': EXP_185,
    '184_XSECTOR': EXP_185,
    '185_XSECTOR': EXP_185,
}
for _fold_id, _fold_exp in _EXP_200_FOLDS.items():
    _EXP_MAP[f'{_fold_id}_XSECTOR'] = _fold_exp
for _fold_id, _fold_exp in _EXP_220_FOLDS.items():
    _EXP_MAP[f'{_fold_id}_XSECTOR'] = _fold_exp

# ── Clear hardcoded configs — fully BQ-driven ──
for _exp in _ALL_EXPS:
    _exp['asset_configs'] = {}

try:
    _n_bq = load_deploy_configs_from_bq()
    if _n_bq == 0:
        raise RuntimeError("No S3 configs from BQ — check family agreement")
    print(f"  Loaded {_n_bq} S3 configs from BQ")
except Exception as _e:
    raise RuntimeError(f"S3 BQ config loading failed: {_e}")

EXPERIMENTS = [e for e in _ALL_EXPS if e['asset_configs']]
S3_TICKERS = sorted({t for e in EXPERIMENTS for t in e['asset_configs']})
S3_CONFIGS = {}
for _e in EXPERIMENTS:
    for _t, _cfg in _e['asset_configs'].items():
        S3_CONFIGS[_t] = {'exp': _e, 'cfg': _cfg}
print(f"  S3 tickers ({len(S3_TICKERS)}): {', '.join(S3_TICKERS)}")

# ==============================================================================
# GCS CLIENT
# ==============================================================================
gcs_client = gcs_storage.Client()

# ==============================================================================
# DATA LOADING
# ==============================================================================
_master_cache = {}


def load_master(ticker, tail_days=None):
    cache_key = (ticker, tail_days)
    if cache_key in _master_cache:
        return _master_cache[cache_key]
    clean = ticker.replace("I:", "").replace("X:", "")
    file_ticker = TICKER_FILE_MAP.get(clean, clean)
    uri = f"gs://{BUCKET_NAME}/{INFERENCE_DIR}/{file_ticker}_master_inference.parquet"
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
        if tail_days is not None:
            cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=tail_days)
            df = df[df.index >= cutoff]
            if df.empty:
                return None
        _master_cache[cache_key] = df
        return df
    except Exception as e:
        print(f"      load_master({ticker}): FAILED — {e}")
        return None


# ==============================================================================
# DATA HEALING
# ==============================================================================
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
# MARKET CONTEXT (per-experiment)
# ==============================================================================
def build_market_context(exp):
    frames = []
    for ticker in exp['context_assets']:
        df = load_master(ticker)
        if df is None:
            continue
        df = df.sort_index()
        df = heal_data(df)
        daily_c = df['Close'].resample('D').last().ffill().shift(1).reindex(
            df.index, method='ffill')
        cols = {f'{ticker}_Anchor_ROC': (df['Close'] - daily_c) / daily_c}
        for h in ROC_HORIZONS:
            cols[f'{ticker}_ROC_{h}m'] = df['Close'].pct_change(h).fillna(0)
        if exp['has_smh_rsi'] and ticker == 'SMH':
            if len(df) >= 14:
                rsi = ta.rsi(df['Close'], length=14)
                cols['SMH_RSI'] = (rsi.fillna(50) / 100.0
                                   if rsi is not None
                                   else pd.Series(0.5, index=df.index))
            else:
                cols['SMH_RSI'] = pd.Series(0.5, index=df.index)
        frames.append(pd.DataFrame(cols, index=df.index))
        del df, daily_c
        gc.collect()
    if not frames:
        raise RuntimeError(f"No context assets loaded for {exp['label']}.")
    return pd.concat(frames, axis=1, sort=True).ffill().fillna(0)


# ==============================================================================
# FEATURE ENGINEERING
# ==============================================================================
def engineer_features(df, market_df):
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
    df_pst = df.index.tz_convert(PST)
    mins = df_pst.hour * 60 + df_pst.minute
    df['Time_Pos'] = np.clip((mins - 390) / 390, 0, 1)
    df['Bar_Is_Real'] = (df['Volume'] > 0).astype(float)
    return df.replace([np.inf, -np.inf], np.nan).fillna(0)


def engineer_asset(ticker, market_df):
    df = load_master(ticker)
    if df is None:
        return None
    df = df.sort_index()
    df = heal_data(df)
    return engineer_features(df, market_df)


# ==============================================================================
# SCALING
# ==============================================================================
def apply_scaling(df, stats, all_cols):
    df_s = df.copy()
    for col in all_cols:
        if col in ('Time_Pos', 'Bar_Is_Real'):
            continue
        if col in stats:
            df_s[col] = ((df_s[col] - stats[col]['med'])
                         / stats[col]['iqr']).clip(-10, 10).astype('float32')
    return df_s


# ==============================================================================
# TRADE SIMULATION (ported from backtest_optimization_csv.py)
# ==============================================================================
def build_ohlcv_arrays(df_raw):
    idx_pst = df_raw.index.tz_convert(PST)
    return {
        'prices': df_raw['Close'].values.astype(np.float64),
        'highs':  df_raw['High'].values.astype(np.float64),
        'lows':   df_raw['Low'].values.astype(np.float64),
        'dates':  (idx_pst.year * 10000 + idx_pst.month * 100 + idx_pst.day).values.astype(np.int64),
        'times':  (idx_pst.hour * 60 + idx_pst.minute).values.astype(np.int64),
        'index':  df_raw.index,
    }


def find_limit_fill(prices, lows, date_ints, time_mins, entry_idx, dip_pct, max_wait):
    n = len(prices)
    if entry_idx >= n - 1:
        return -1, 0.0
    sp = prices[entry_idx]
    ed = date_ints[entry_idx]
    et = time_mins[entry_idx]
    lp = sp * (1.0 - dip_pct)
    for k in range(entry_idx + 1, n):
        if date_ints[k] != ed:
            break
        if time_mins[k] - et > max_wait:
            break
        if lows[k] <= lp:
            return k, lp
    return -1, 0.0


def compute_exit_pnl(prices, highs, lows, date_ints, time_mins,
                     fill_idx, fill_price, tp, sl, exit_time):
    n = len(prices)
    ed = date_ints[fill_idx]
    tgt = fill_price * (1.0 + tp)
    stp = fill_price * (1.0 - sl)
    for k in range(fill_idx + 1, n):
        if date_ints[k] != ed:
            return (prices[k - 1] - fill_price) / fill_price
        sl_h = lows[k] <= stp
        tp_h = highs[k] >= tgt
        if sl_h and tp_h:
            return -sl if np.random.random() < 0.5 else tp
        if sl_h:
            return -sl
        if tp_h:
            return tp
        if time_mins[k] >= exit_time:
            return (prices[k] - fill_price) / fill_price
    return (prices[n - 1] - fill_price) / fill_price


def summarize(trades_list, label):
    if not trades_list:
        return {'label': label, 'n': 0, 'total': 0, 'avg': 0, 'wr': 0,
                'dd': 0, 'j_n': 0, 'j_total': 0, 'j_avg': 0, 'j_wr': 0,
                'monthly': {}, 'trades_df': pd.DataFrame()}
    df = pd.DataFrame(trades_list)
    df['month'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m')
    cum = df.sort_values('date')['pnl'].cumsum()
    max_dd = (cum - cum.cummax()).min()
    june = df[df['month'] == '2026-06']
    july = df[df['month'] == '2026-07']
    return {
        'label': label,
        'n': len(df), 'total': df['pnl'].sum(), 'avg': df['pnl'].mean(),
        'wr': (df['pnl'] > 0).mean(), 'dd': max_dd,
        'j_n': len(june), 'j_total': june['pnl'].sum() if len(june) else 0,
        'j_avg': june['pnl'].mean() if len(june) else 0,
        'j_wr': (june['pnl'] > 0).mean() if len(june) else 0,
        'jul_n': len(july), 'jul_total': july['pnl'].sum() if len(july) else 0,
        'jul_avg': july['pnl'].mean() if len(july) else 0,
        'monthly': df.groupby('month').agg(
            n=('pnl', 'count'), total=('pnl', 'sum'),
            avg=('pnl', 'mean'), wr=('pnl', lambda x: (x > 0).mean())
        ).to_dict('index'),
        'trades_df': df,
    }


# ==============================================================================
# MODEL ARCHITECTURE (parameterized)
# ==============================================================================
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.4):
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    res = Add()([x, inputs])
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    return Add()([x, res])


def build_model(n_context_cols, n_assets, n_asset_cols=None):
    n_ac = n_asset_cols or len(ASSET_COLS)
    in_a  = Input(shape=(TIME_STEPS, n_ac), name='asset_in')
    in_c  = Input(shape=(n_context_cols,),             name='context_in')
    in_id = Input(shape=(1,), dtype=tf.int32,          name='asset_id')
    x = transformer_encoder(Dense(128)(in_a), 128, 4, 128)
    x = GlobalAveragePooling1D()(transformer_encoder(x, 128, 4, 128))
    y = Dense(128, activation='relu')(LayerNormalization()(in_c))
    e = Embedding(n_assets, EMBED_DIM, name='asset_embed')(in_id)
    e = Flatten()(e)
    e = Dense(32, activation='relu')(e)
    z = Concatenate()([x, Dropout(0.5)(y), e])
    z = Dense(128, activation='relu', kernel_regularizer=l2(5e-3))(z)
    out = Dense(1, activation='sigmoid', dtype='float32')(Dropout(0.5)(z))
    return Model(inputs=[in_a, in_c, in_id], outputs=out)


def build_cnn_transformer_model(n_context_cols, n_assets, n_asset_cols=None, embed_dim=32):
    CNN_FILTERS = 64
    CNN_KERNELS = [3, 7, 15]
    CNN_POOL_1 = 4
    CNN_POOL_2 = 4
    n_ac = n_asset_cols or len(ASSET_COLS)
    in_a  = Input(shape=(TIME_STEPS, n_ac), name='asset_in')
    in_c  = Input(shape=(n_context_cols,),  name='context_in')
    in_id = Input(shape=(1,), dtype=tf.int32, name='asset_id')
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


# ==============================================================================
# MODEL LOADER (keyed by model_id + seed)
# ==============================================================================
_model_cache = {}


def get_model(exp, seed):
    key = (exp['model_id'], seed)
    if key in _model_cache:
        return _model_cache[key]
    model_id = exp['model_id']
    blob_path  = f"{GCS_MODEL_PATH}/UNIVERSAL_S{seed}_M{model_id}.h5"
    local_path = f"/tmp/UNIVERSAL_S{seed}_M{model_id}.h5"
    if not os.path.exists(local_path):
        gcs_client.bucket(BUCKET_NAME).blob(blob_path).download_to_filename(local_path)
    builder = exp.get('model_builder', 'standard')
    embed_dim = exp.get('embed_dim', EMBED_DIM)
    if builder == 'cnn_transformer':
        model = build_cnn_transformer_model(len(exp['context_cols']), exp['n_assets'],
                                            n_asset_cols=len(exp['asset_cols']),
                                            embed_dim=embed_dim)
    else:
        model = build_model(len(exp['context_cols']), exp['n_assets'],
                            n_asset_cols=len(exp['asset_cols']))
    model.load_weights(local_path)
    _model_cache[key] = model
    return model


# SCORING ENGINE
# ==============================================================================
def score_ticker(ticker, cfg, df_scaled, df_raw, exp, target_date):
    if len(df_scaled) == 0:
        return None
    entry_min = cfg['entry_minute']
    target_pst = PST.localize(datetime(target_date.year, target_date.month, target_date.day,
                                       entry_min // 60, entry_min % 60))
    target_utc = target_pst.astimezone(pytz.utc)
    candidates = df_scaled.index[df_scaled.index <= target_utc]
    if len(candidates) == 0:
        return None
    row_idx = df_scaled.index.get_loc(candidates[-1])
    bar_ts_pst = df_scaled.index[row_idx].tz_convert(PST)
    score_date = bar_ts_pst.date()
    if score_date != target_date:
        return None
    bar_minute = bar_ts_pst.hour * 60 + bar_ts_pst.minute
    if bar_minute != entry_min:
        return None
    if row_idx < TIME_STEPS:
        return None
    if 'Volume' in df_scaled.columns and float(df_scaled['Volume'].iloc[row_idx]) == 0:
        return None
    if 'Close' in df_raw.columns and df_raw is not None:
        raw_candidates = df_raw.index[df_raw.index <= df_scaled.index[row_idx]]
        if len(raw_candidates) > 0:
            raw_close = float(df_raw['Close'].iloc[df_raw.index.get_loc(raw_candidates[-1])])
            if raw_close < MIN_PRICE:
                return None
    context_cols = exp['context_cols']
    XA = df_scaled[exp['asset_cols']].values
    XC = df_scaled[context_cols].values
    XA_b  = XA[row_idx - TIME_STEPS:row_idx][np.newaxis]
    XC_b  = XC[row_idx - 1][np.newaxis]
    XID_b = np.array([[exp['asset_index'].get(ticker, exp['other_idx'])]], dtype=np.int32)

    is_ensemble = cfg.get('is_ensemble', False)
    if is_ensemble:
        base_num = int(exp['model_id'].split('_')[0])
        seed_list = list(range(base_num * 100 + 1, base_num * 100 + 16))
        all_probs = []
        for s in seed_list:
            try:
                m = get_model(exp, s)
                p = float(m(
                    {'asset_in': XA_b, 'context_in': XC_b, 'asset_id': XID_b},
                    training=False
                ).numpy().flatten()[0])
                all_probs.append(p)
            except Exception:
                continue
        if not all_probs:
            print(f"      {ticker}: SKIP — no ensemble models loaded")
            return None
        prob = float(np.mean(all_probs))
    else:
        model = get_model(exp, cfg['seed'])
        prob = float(model(
            {'asset_in': XA_b, 'context_in': XC_b, 'asset_id': XID_b},
            training=False
        ).numpy().flatten()[0])

    signal = prob >= cfg['prob_threshold']
    bar_ts = df_scaled.index[row_idx]
    close  = (float(df_scaled['Close'].iloc[row_idx])
              if 'Close' in df_scaled.columns else float('nan'))

    return {
        'ticker': ticker, 'experiment': exp['model_id'],
        'prob': round(prob, 4), 'threshold': cfg['prob_threshold'],
        'signal': signal, 'tp_pct': cfg['tp_pct'], 'sl_pct': cfg['sl_pct'],
        'entry_minute': cfg['entry_minute'], 'score_date': score_date,
        'bar_ts': bar_ts, 'close': round(close, 4),
        'row_idx': row_idx,
        'dip_pct': cfg.get('dip_pct', 0),
        'max_wait_minutes': cfg.get('max_wait_minutes', 30),
    }


# ==============================================================================
# BACKTEST MAIN
# ==============================================================================


def main():
    print(f"\n{'='*100}")
    print(f"  BACKTEST: S3 STATIC vs S3 MOMENTUM 10d   ({TEST_START} → {TEST_END})")
    print(f"  Live TF Inference — {len(S3_TICKERS)} tickers: {', '.join(S3_TICKERS)}")
    print(f"{'='*100}")

    # ── Phase 1: Pre-load data & models per experiment ──
    print(f"\n  Phase 1: Loading data and models...")
    exp_groups = defaultdict(dict)
    for ticker, sc in S3_CONFIGS.items():
        exp = sc['exp']
        exp_groups[exp['model_id']][ticker] = sc

    ticker_data = {}
    for model_id, tickers_sc in exp_groups.items():
        exp = list(tickers_sc.values())[0]['exp']
        print(f"    {exp['label']}...")

        try:
            blob  = gcs_client.bucket(BUCKET_NAME).blob(exp['stats_path'])
            stats = json.loads(blob.download_as_text())
        except Exception as e:
            print(f"      FAILED loading stats: {e}")
            continue

        market_df = build_market_context(exp)

        base_num = int(model_id.split('_')[0])
        seed_list = list(range(base_num * 100 + 1, base_num * 100 + 16))
        n_loaded = 0
        for s in seed_list:
            try:
                get_model(exp, s)
                n_loaded += 1
            except Exception as e:
                print(f"      seed {s} FAILED: {e}")
        if n_loaded == 0:
            print(f"      SKIP — no models loaded for {model_id}")
            continue
        print(f"      {n_loaded}/15 models loaded")

        for ticker, sc in tickers_sc.items():
            df_raw = engineer_asset(ticker, market_df)
            if df_raw is None:
                print(f"      {ticker}: SKIP — no data")
                continue
            df_scaled = apply_scaling(df_raw, stats, exp['all_cols'])
            ohlcv = build_ohlcv_arrays(df_raw)
            ticker_data[ticker] = {
                'df_scaled': df_scaled, 'df_raw': df_raw,
                'ohlcv': ohlcv, 'exp': exp, 'cfg': sc['cfg'],
            }
        del market_df
        gc.collect()

    print(f"  Data loaded for {len(ticker_data)} tickers: {', '.join(sorted(ticker_data.keys()))}")

    # ── Phase 2: Build daily_closes for MOMENTUM filter ──
    daily_closes = {}
    for ticker, td in ticker_data.items():
        o = td['ohlcv']
        closes = {}
        for d_int in np.unique(o['dates']):
            mask = o['dates'] == d_int
            day_prices = o['prices'][mask]
            if len(day_prices) > 0:
                d_str = str(int(d_int))
                d_date = pd.Timestamp(f"{d_str[:4]}-{d_str[4:6]}-{d_str[6:8]}").date()
                closes[d_date] = float(day_prices[-1])
        daily_closes[ticker] = closes

    # ── Phase 3: Get trading days ──
    test_start_dt = pd.Timestamp(TEST_START).date()
    test_end_dt = pd.Timestamp(TEST_END).date()
    all_dates = set()
    for ticker, dc in daily_closes.items():
        all_dates.update(d for d in dc if test_start_dt <= d <= test_end_dt)
    trading_days = sorted(all_dates)
    print(f"  Trading days: {len(trading_days)} ({trading_days[0]} → {trading_days[-1]})")

    # ── Phase 4: Score and simulate ──
    print(f"\n  Phase 2: Scoring and simulating trades...")
    np.random.seed(42)
    static_trades = []
    momentum_trades = []
    n_scored = 0
    n_signals = 0

    for di, day in enumerate(trading_days):
        day_signals = 0
        for ticker, td in ticker_data.items():
            cfg = td['cfg']
            exp = td['exp']
            result = score_ticker(ticker, cfg, td['df_scaled'], td['df_raw'], exp, target_date=day)
            if result is None:
                continue
            n_scored += 1
            if not result['signal']:
                continue
            n_signals += 1

            o = td['ohlcv']
            entry_idx = result['row_idx']
            dip_pct = result.get('dip_pct', 0) or 0
            max_wait = result.get('max_wait_minutes', 30) or 30
            tp = result['tp_pct']
            sl = result['sl_pct']

            fill_idx, fill_price = find_limit_fill(
                o['prices'], o['lows'], o['dates'], o['times'],
                entry_idx, dip_pct, max_wait)
            if fill_idx < 0:
                continue

            pnl = compute_exit_pnl(
                o['prices'], o['highs'], o['lows'], o['dates'], o['times'],
                fill_idx, fill_price, tp, sl, EXIT_TIME_MIN)
            pnl -= SLIPPAGE_PCT

            trade = {'date': day, 'ticker': ticker, 'pnl': pnl,
                     'prob': result['prob'], 'close': result['close'],
                     'fill_price': round(fill_price, 4),
                     'tp': tp, 'sl': sl, 'exp': result['experiment']}

            static_trades.append(trade)
            day_signals += 1

            closes = daily_closes.get(ticker, {})
            prior = sorted(d for d in closes if d < day)
            momentum_ok = True
            if len(prior) >= MOMENTUM_LOOKBACK_DAYS:
                close_now = closes[prior[-1]]
                close_10d = closes[prior[-MOMENTUM_LOOKBACK_DAYS]]
                if close_10d > 0 and (close_now - close_10d) / close_10d <= 0:
                    momentum_ok = False
            if momentum_ok:
                momentum_trades.append(trade)

        if (di + 1) % 10 == 0 or di == len(trading_days) - 1:
            print(f"    Day {di+1}/{len(trading_days)}: {day}  "
                  f"({n_scored} scored, {n_signals} signals, "
                  f"{len(static_trades)} static trades, {len(momentum_trades)} momentum trades)")

    # ── Phase 5: Results ──
    print(f"\n{'='*100}")
    print(f"  RESULTS: S3 STATIC vs S3 MOMENTUM 10d")
    print(f"{'='*100}")

    s_static = summarize(static_trades, 'S3 STATIC')
    s_momentum = summarize(momentum_trades, 'S3 MOMENTUM 10d')

    W = 100
    print(f"\n  {'Strategy':<20}  {'Trades':>6}  {'Total':>8}  {'Avg':>7}  {'WR':>5}  {'MaxDD':>7}  "
          f"{'Jun N':>5}  {'Jun Tot':>8}  {'Jun Avg':>7}  {'Jul N':>5}  {'Jul Tot':>8}")
    print(f"  {'─'*W}")
    for s in [s_static, s_momentum]:
        print(f"  {s['label']:<20}  {s['n']:>6}  {s['total']:>+8.1%}  {s['avg']:>+7.2%}  "
              f"{s['wr']:>5.0%}  {s['dd']:>+7.2%}  "
              f"{s['j_n']:>5}  {s['j_total']:>+8.1%}  {s['j_avg']:>+7.2%}  "
              f"{s.get('jul_n', 0):>5}  {s.get('jul_total', 0):>+8.1%}")

    # Monthly breakdown
    print(f"\n  Monthly Breakdown:")
    print(f"  {'Strategy':<20}  {'Month':<8}  {'N':>4}  {'Total':>8}  {'Avg':>7}  {'WR':>5}")
    print(f"  {'─'*60}")
    for s in [s_static, s_momentum]:
        for month in sorted(s['monthly'].keys()):
            m = s['monthly'][month]
            print(f"  {s['label']:<20}  {month:<8}  {m['n']:>4}  {m['total']:>+8.1%}  "
                  f"{m['avg']:>+7.2%}  {m['wr']:>5.0%}")
        print(f"  {'─'*60}")

    # Per-ticker breakdown
    if static_trades:
        print(f"\n  Per-Ticker (STATIC):")
        df_st = pd.DataFrame(static_trades)
        tk_stats = df_st.groupby('ticker').agg(
            n=('pnl', 'count'), total=('pnl', 'sum'),
            avg=('pnl', 'mean'), wr=('pnl', lambda x: (x > 0).mean())
        ).sort_values('total', ascending=False)
        print(f"  {'Ticker':<8}  {'N':>4}  {'Total':>8}  {'Avg':>7}  {'WR':>5}")
        print(f"  {'─'*40}")
        for tk, row in tk_stats.iterrows():
            print(f"  {tk:<8}  {int(row['n']):>4}  {row['total']:>+8.1%}  "
                  f"{row['avg']:>+7.2%}  {row['wr']:>5.0%}")

    # Momentum-skipped trades
    n_skipped = len(static_trades) - len(momentum_trades)
    if n_skipped > 0:
        skipped = [t for t in static_trades if t not in momentum_trades]
        print(f"\n  MOMENTUM filter skipped {n_skipped} trades:")
        for t in skipped:
            print(f"    {t['date']}  {t['ticker']:<8}  pnl={t['pnl']:+.2%}")

    # ── Charts ──
    if not static_trades:
        print("\n  No trades — skipping charts.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    # Chart 1: Cumulative PnL (simple sum)
    for label, trades, color in [('S3 STATIC', static_trades, '#333333'),
                                  ('S3 MOMENTUM 10d', momentum_trades, '#2196F3')]:
        if not trades:
            continue
        df_t = pd.DataFrame(trades).sort_values('date')
        df_t['cum_pnl'] = df_t['pnl'].cumsum()
        dates = pd.to_datetime(df_t['date'])
        axes[0].plot(dates, df_t['cum_pnl'] * 100, label=label, color=color, linewidth=2)
    axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_ylabel('Cumulative PnL (%)')
    axes[0].set_title('S3 STATIC vs MOMENTUM 10d — Cumulative PnL')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Chart 2: Portfolio equity ($200K starting)
    for label, trades, color in [('S3 STATIC', static_trades, '#333333'),
                                  ('S3 MOMENTUM 10d', momentum_trades, '#2196F3')]:
        if not trades:
            continue
        df_t = pd.DataFrame(trades).sort_values('date')
        dates = pd.to_datetime(df_t['date'])
        equity = [STARTING_CAPITAL]
        n_tickers = len(ticker_data)
        for pnl in df_t['pnl']:
            pos_size = equity[-1] / max(n_tickers, 1)
            equity.append(equity[-1] + pos_size * pnl)
        axes[1].plot(dates, equity[1:], label=label, color=color, linewidth=2)
    axes[1].axhline(y=STARTING_CAPITAL, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_ylabel(f'Portfolio Value ($)')
    axes[1].set_title(f'Portfolio Simulation (${STARTING_CAPITAL:,} start, position = equity / {len(ticker_data)} tickers)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.savefig('/tmp/s3_static_vs_momentum_backtest.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n  Chart saved: /tmp/s3_static_vs_momentum_backtest.png")
    print(f"{'='*100}")


main()
