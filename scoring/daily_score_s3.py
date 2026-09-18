"""
Daily Scoring — Strategy S3
============================
Self-contained. Scores tickers under Strategy S3 and prints dashboard.

Strategy S3 (limit_entry + family agreement):
  Gate: majority of families positive val_b, n_fams >= 2. Mode: limit_entry only.
  Deploy: best limit_entry config per ticker by score_pnl2.
  score_pnl2 = mean_va_pnl^2 * agree_18.
  170 uses fold-175 models; 180 uses fold-185 models.

Confidence tiers:
  HIGH  — N>=7 seeds    MED — N=4-6 seeds    LOW — N<=3 seeds

Run AFTER daily_ohlcv scripts have finished updating GCS.
"""

import os
import sys
import json
import gc
import subprocess

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

if tf.config.list_physical_devices('GPU'):
    mixed_precision.set_global_policy('mixed_float16')

try:
    from google.cloud import bigquery
except Exception:
    bigquery = None

# ==============================================================================
# SHARED CONSTANTS
# ==============================================================================
BUCKET_NAME    = os.environ.get("GCS_BUCKET", "")
GCS_MODEL_PATH = "Algos/to_deploy/Saved_models_v2/UNIVERSAL/Direction"
BQ_TABLE       = f"{GCP_PROJECT}.scoring_logs.live_predictions"
INFERENCE_DIR  = "Data/Inference_v2"
TAIL_DAYS      = 25
VOL_LOOKBACK_DAYS = 30
TIME_STEPS     = 240
EMBED_DIM      = 16
PST            = pytz.timezone('America/Los_Angeles')
MIN_PRICE          = 2.00
MIN_DOLLAR_VOLUME  = 50000
MIN_VOL_FILTER_PCT = 25

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
SPY_3D_RETURN_MIN = 0.005
VIX_RANGE_5D_MAX  = 0.08

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
    if 240 <= num <= 245: return '240'
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
    WHERE (model_version IN ('{WF_MODEL_VERSION}', '{WF_MODEL_VERSION}_stitched')
           OR model_version LIKE 'robust_wf_v12_17%'
           OR model_version LIKE 'robust_wf_v12_18%'
           OR model_version LIKE 'robust_wf_v12_20%'
           OR model_version LIKE 'robust_wf_v12_24%')
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

# ── EXP-200 family (CNN-Transformer, embed_dim=32, all folds share base 200 model) ──
EXP_200 = {
    'model_id': '200_XSECTOR',
    'label': 'EXP-200 XSECTOR (CNN-Transformer)',
    'train_teachers': TEACHERS_31,
    'context_assets': ['SPY', 'VIX', 'TLT', 'IWM'],
    'has_smh_rsi': False,
    'model_builder': 'cnn_transformer',
    'embed_dim': 32,
    'asset_configs': {},
    'asset_cols': ASSET_COLS,
}

TEACHERS_47 = TEACHERS_31 + [
    'AMD', 'NVDA', 'TSLA', 'PLTR', 'SOUN', 'HOOD', 'SMCI', 'VRT', 'NBIS',
    'KTOS', 'AVAV', 'ACHR', 'JOBY', 'NNDM', 'CCJ', 'CEG',
]

EXP_220 = {
    'model_id': '220_XSECTOR',
    'label': 'EXP-220 XSECTOR (47 teachers, stitched)',
    'train_teachers': TEACHERS_47,
    'context_assets': ['SPY', 'VIX', 'TLT', 'IWM'],
    'has_smh_rsi': False,
    'asset_configs': {},
    'asset_cols': ASSET_COLS,
}

_ALL_EXPS = [EXP_152, EXP_160, EXP_170, EXP_185, EXP_200, EXP_220]

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
}
for _n in range(170, 176):
    _EXP_MAP[f'{_n}_XSECTOR'] = EXP_170
for _n in range(180, 186):
    _EXP_MAP[f'{_n}_XSECTOR'] = EXP_185
for _n in range(200, 206):
    _EXP_MAP[f'{_n}_XSECTOR'] = EXP_200
for _n in range(220, 226):
    _EXP_MAP[f'{_n}_XSECTOR'] = EXP_220

# Clear hardcoded configs — S3 is fully BQ-driven
for _exp in _ALL_EXPS:
    _exp['asset_configs'] = {}

try:
    _n_bq = load_deploy_configs_from_bq()
    if _n_bq == 0:
        print("  WARNING: No S3 configs selected from BQ — check family agreement in wf_optimization_results")
except Exception as _e:
    print(f"  ERROR loading S3 BQ configs: {_e}")
    print(f"  S3 scoring will have 0 tickers — check BQ access and WF table")

EXPERIMENTS = [e for e in _ALL_EXPS if e['asset_configs']]

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
        df = load_master(ticker, tail_days=TAIL_DAYS)
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
    df = load_master(ticker, tail_days=TAIL_DAYS)
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
    blob = gcs_client.bucket(BUCKET_NAME).blob(blob_path)
    if os.path.exists(local_path):
        blob.reload()
        if os.path.getsize(local_path) != blob.size:
            os.remove(local_path)
    if not os.path.exists(local_path):
        blob.download_to_filename(local_path)
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


# ==============================================================================
# VOLUME FILTER (percentile-based, anchored to entry_minute)
# ==============================================================================
def _check_vol_filter(ticker, entry_minute, vol_filter_pct):
    if vol_filter_pct <= 0:
        return True, None, None
    df = load_master(ticker, tail_days=VOL_LOOKBACK_DAYS + 15)
    if df is None:
        return True, None, None
    df = heal_data(df)
    df_pst = df.copy()
    df_pst.index = df_pst.index.tz_convert(PST)
    mins = df_pst.index.hour * 60 + df_pst.index.minute
    at_entry = df_pst[mins == entry_minute]
    today = datetime.now(PST).date()
    today_bar = at_entry[at_entry.index.date == today]
    past_real = at_entry[(at_entry.index.date < today) & (at_entry['Volume'] > 0)]
    past_bars = past_real.tail(VOL_LOOKBACK_DAYS)
    if len(today_bar) == 0 or len(past_bars) < 5:
        return True, None, None
    entry_vol = float(today_bar['Volume'].iloc[-1])
    median_vol = float(past_bars['Volume'].median())
    if median_vol <= 0:
        return True, entry_vol, 0.0
    passed = entry_vol >= (vol_filter_pct / 100.0) * median_vol
    return passed, entry_vol, median_vol


# ==============================================================================
# DYNAMIC TP/SL COMPUTATION
# ==============================================================================
REGIME_TP_SCALE = [0.70, 0.85, 1.00, 1.30, 1.60]
REGIME_SL_SCALE = [0.60, 0.80, 1.00, 1.30, 1.80]


def _daily_atr_pct(df_raw, row_idx, period=14):
    daily = df_raw[['High', 'Low', 'Close']].resample('D').agg(
        {'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
    if len(daily) < 2:
        return 0.05
    prev_close = daily['Close'].shift(1)
    tr = pd.concat([
        daily['High'] - daily['Low'],
        (daily['High'] - prev_close).abs(),
        (daily['Low'] - prev_close).abs()
    ], axis=1).max(axis=1)
    atr_series = tr.rolling(period, min_periods=max(period // 2, 3)).mean()
    atr_pct = atr_series / daily['Close']
    atr_pct = atr_pct.reindex(df_raw.index, method='ffill')
    val = float(atr_pct.iloc[row_idx]) if row_idx < len(atr_pct) else np.nan
    if np.isnan(val) or val <= 0:
        return 0.05
    return val


def _adaptive_atr_pct(df_raw, row_idx, short=5, long=22, base=14):
    daily = df_raw[['High', 'Low', 'Close']].resample('D').agg(
        {'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
    if len(daily) < 2:
        return 0.05
    prev_close = daily['Close'].shift(1)
    tr = pd.concat([
        daily['High'] - daily['Low'],
        (daily['High'] - prev_close).abs(),
        (daily['Low'] - prev_close).abs()
    ], axis=1).max(axis=1)
    atr_base  = tr.rolling(base).mean()
    atr_short = tr.rolling(short).mean()
    atr_long  = tr.rolling(long).mean()
    accel = (atr_short / atr_long).clip(0.5, 3.0)
    atr_adaptive = atr_base * accel
    atr_pct = atr_adaptive / daily['Close']
    atr_pct = atr_pct.reindex(df_raw.index, method='ffill')
    val = float(atr_pct.iloc[row_idx]) if row_idx < len(atr_pct) else np.nan
    if np.isnan(val) or val <= 0:
        return 0.05
    return val


def compute_dynamic_tpsl(cfg, df_raw, row_idx, vix_value):
    mode = cfg.get('tpsl_mode', 'static')

    if mode == 'atr':
        atr_pct = _daily_atr_pct(df_raw, row_idx)
        tp = min(max(atr_pct * cfg['tp_mult'], 0.02), 0.60)
        sl = min(max(atr_pct * cfg['sl_mult'], 0.01), 0.25)
        return tp, sl, {'atr_pct': round(atr_pct, 6), 'mode': 'atr'}

    elif mode == 'atr_adaptive':
        atr_pct = _adaptive_atr_pct(df_raw, row_idx)
        tp = min(max(atr_pct * cfg['tp_mult'], 0.02), 0.60)
        sl = min(max(atr_pct * cfg['sl_mult'], 0.01), 0.25)
        return tp, sl, {'atr_pct': round(atr_pct, 6), 'mode': 'atr_adaptive'}

    elif mode in ('regime', 'regime_scaled'):
        regime_idx = 2
        if vix_value is not None:
            if vix_value <= 15: regime_idx = 0
            elif vix_value <= 20: regime_idx = 1
            elif vix_value <= 25: regime_idx = 2
            elif vix_value <= 35: regime_idx = 3
            else: regime_idx = 4
        tp = cfg['tp_pct'] * REGIME_TP_SCALE[regime_idx]
        sl = cfg['sl_pct'] * REGIME_SL_SCALE[regime_idx]
        return tp, sl, {'vix_regime': regime_idx, 'mode': mode}

    elif mode in ('trail', 'trailing_decay'):
        atr_pct = _daily_atr_pct(df_raw, row_idx)
        sl = min(max(atr_pct * cfg['sl_mult'], 0.01), 0.25)
        trail_start = atr_pct * cfg['trail_start_mult']
        trail_end = atr_pct * cfg['trail_end_mult']
        return None, sl, {
            'atr_pct': round(atr_pct, 6),
            'trail_start': round(trail_start, 6),
            'trail_end': round(trail_end, 6),
            'mode': mode,
        }

    elif mode == 'limit_entry':
        return cfg['tp_pct'], cfg['sl_pct'], {
            'mode': 'limit_entry',
            'dip_pct': cfg.get('dip_pct', 0.01),
            'max_wait_minutes': cfg.get('max_wait_minutes', 30),
        }

    return cfg['tp_pct'], cfg['sl_pct'], {'mode': 'static'}


# ==============================================================================
# SCORING ENGINE
# ==============================================================================
def score_ticker(ticker, cfg, df_scaled, df_raw, exp):
    if len(df_scaled) == 0:
        print(f"      {ticker}: SKIP — df_scaled is empty")
        return None
    entry_min = cfg['entry_minute']
    today = datetime.now(PST).date()
    target_pst = PST.localize(datetime(today.year, today.month, today.day,
                                       entry_min // 60, entry_min % 60))
    target_utc = target_pst.astimezone(pytz.utc)
    candidates = df_scaled.index[df_scaled.index <= target_utc]
    if len(candidates) == 0:
        print(f"      {ticker}: SKIP — no bars at or before entry {entry_min} "
              f"(target_utc={target_utc}, index range {df_scaled.index[0]} to {df_scaled.index[-1]})")
        return None
    row_idx = df_scaled.index.get_loc(candidates[-1])
    bar_ts_pst = df_scaled.index[row_idx].tz_convert(PST)
    score_date = bar_ts_pst.date()
    if score_date != today:
        print(f"      {ticker}: SKIP — nearest bar date {score_date} != today {today} "
              f"(bar_ts={df_scaled.index[row_idx]})")
        return None
    bar_minute = bar_ts_pst.hour * 60 + bar_ts_pst.minute
    if bar_minute != entry_min:
        print(f"      {ticker}: SKIP — bar at minute {bar_minute} != entry_minute {entry_min} "
              f"(no real bar at entry time)")
        return None
    if row_idx < TIME_STEPS:
        print(f"      {ticker}: SKIP — not enough history: row_idx={row_idx} < {TIME_STEPS} "
              f"(need {TIME_STEPS} bars before entry bar)")
        return None
    if 'Volume' in df_scaled.columns and float(df_scaled['Volume'].iloc[row_idx]) == 0:
        print(f"      {ticker}: SKIP — Volume=0 at entry bar {df_scaled.index[row_idx]} (synthetic)")
        return None
    if 'Close' in df_raw.columns and df_raw is not None:
        raw_candidates = df_raw.index[df_raw.index <= df_scaled.index[row_idx]]
        if len(raw_candidates) > 0:
            raw_close = float(df_raw['Close'].iloc[df_raw.index.get_loc(raw_candidates[-1])])
            if raw_close < MIN_PRICE:
                print(f"      {ticker}: SKIP — price ${raw_close:.2f} < ${MIN_PRICE:.2f} (penny stock)")
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

    regime = load_regime_data(score_date, cfg['entry_minute'])
    regime_ok = True
    if regime:
        regime_ok, _, _, _ = _filter_pass(cfg, regime)

    vol_pct = cfg.get('vol_filter_pct', 0)
    vol_ok, entry_vol, vol_median = _check_vol_filter(ticker, entry_min, vol_pct)

    vix_value = regime.get('vix') if regime else None
    raw_row_idx = None
    if df_raw is not None and len(df_raw) > 0:
        raw_candidates = df_raw.index[df_raw.index <= bar_ts]
        if len(raw_candidates) > 0:
            raw_row_idx = df_raw.index.get_loc(raw_candidates[-1])

    dollar_vol = None
    dollar_vol_ok = True
    if df_raw is not None and raw_row_idx is not None:
        rv = float(df_raw['Volume'].iloc[raw_row_idx])
        rp = float(df_raw['Close'].iloc[raw_row_idx])
        dollar_vol = rv * rp
        if entry_vol is None:
            entry_vol = rv
        pass

    filter_ok = regime_ok and vol_ok and dollar_vol_ok

    tpsl_mode = cfg.get('tpsl_mode', 'static')
    if tpsl_mode != 'static' and df_raw is not None and raw_row_idx is not None:
        tp_pct, sl_pct, tpsl_info = compute_dynamic_tpsl(cfg, df_raw, raw_row_idx, vix_value)
    else:
        tp_pct, sl_pct, tpsl_info = cfg['tp_pct'], cfg['sl_pct'], {'mode': 'static'}

    return {
        'ticker': ticker, 'experiment': exp['model_id'], 'seed': cfg['seed'],
        'prob': round(prob, 4), 'threshold': cfg['prob_threshold'],
        'signal': signal, 'tp_pct': tp_pct, 'sl_pct': sl_pct,
        'entry_minute': cfg['entry_minute'], 'score_date': score_date,
        'bar_ts': bar_ts, 'close': round(close, 4),
        'filter_ok': filter_ok,
        'filter_status': 'OK' if filter_ok else 'BLOCKED',
        'vix_value': vix_value,
        'tlt_gap_pct': regime.get('tlt_gap') if regime else None,
        'spy_gap_pct': regime.get('spy_gap') if regime else None,
        'vix_range_5d': regime.get('vix_range_5d') if regime else None,
        'spy_3d_return': regime.get('spy_3d_return') if regime else None,
        'vol_filter_pct': vol_pct,
        'vol_filter_ok': vol_ok,
        'dollar_vol_ok': dollar_vol_ok,
        'entry_volume': entry_vol,
        'dollar_volume': dollar_vol,
        'vol_median': vol_median,
        'tpsl_mode': tpsl_info.get('mode', 'static'),
        'atr_pct_entry': tpsl_info.get('atr_pct'),
        'vix_regime_idx': tpsl_info.get('vix_regime'),
        'trail_start': tpsl_info.get('trail_start'),
        'trail_end': tpsl_info.get('trail_end'),
        'dip_pct': tpsl_info.get('dip_pct'),
        'max_wait_minutes': tpsl_info.get('max_wait_minutes'),
    }


# ==============================================================================
# REGIME FILTER (anchored to entry_minute, not run_ts)
# ==============================================================================
_regime_cache = {}


def load_regime_data(score_date, entry_minute):
    cache_key = (score_date, entry_minute)
    if cache_key in _regime_cache:
        return _regime_cache[cache_key]

    window_end   = PST.localize(datetime.combine(score_date, datetime.min.time())) + timedelta(minutes=entry_minute)
    window_start = window_end - timedelta(minutes=30)

    vix_level = None
    df_vix = load_master('VIX', tail_days=TAIL_DAYS)
    if df_vix is not None:
        df_vix = heal_data(df_vix)
        dpst = df_vix.copy()
        dpst.index = dpst.index.tz_convert(PST)
        bars = dpst[(dpst.index >= window_start) & (dpst.index <= window_end)]
        if len(bars):
            vix_level = float(bars['Close'].median())

    tlt_gap = None
    df_tlt = load_master('TLT', tail_days=TAIL_DAYS)
    if df_tlt is not None:
        df_tlt = heal_data(df_tlt)
        dpst = df_tlt.copy()
        dpst.index = dpst.index.tz_convert(PST)
        prev = dpst[dpst.index.date < score_date]
        bars = dpst[(dpst.index >= window_start) & (dpst.index <= window_end)]
        if len(prev) > 0 and len(bars) > 0:
            prev_close = float(prev['Close'].iloc[-1])
            entry_close = float(bars['Close'].median())
            if prev_close != 0:
                tlt_gap = ((entry_close - prev_close) / prev_close) * 100

    spy_gap = None
    df_spy = load_master('SPY', tail_days=TAIL_DAYS)
    if df_spy is not None:
        df_spy = heal_data(df_spy)
        dpst = df_spy.copy()
        dpst.index = dpst.index.tz_convert(PST)
        prev = dpst[dpst.index.date < score_date]
        bars = dpst[(dpst.index >= window_start) & (dpst.index <= window_end)]
        if len(prev) > 0 and len(bars) > 0:
            prev_close  = float(prev['Close'].iloc[-1])
            entry_close = float(bars['Close'].median())
            if prev_close != 0:
                spy_gap = ((entry_close - prev_close) / prev_close) * 100

    vix_range_5d = None
    if df_vix is not None:
        dpst = df_vix.copy()
        dpst.index = dpst.index.tz_convert(PST)
        vix_d = dpst.resample('D').agg({'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
        vix_d = vix_d[vix_d.index.date <= score_date]
        if len(vix_d) >= 3:
            rng = (vix_d['High'] - vix_d['Low']) / vix_d['Close']
            vix_range_5d = float(rng.rolling(5, min_periods=3).mean().iloc[-1])

    spy_3d_return = None
    if df_spy is not None:
        dpst = df_spy.copy()
        dpst.index = dpst.index.tz_convert(PST)
        spy_d = dpst['Close'].resample('D').last().dropna()
        spy_d = spy_d[spy_d.index.date <= score_date]
        if len(spy_d) >= 4:
            spy_3d_return = float(spy_d.pct_change(3).iloc[-1])

    regime = {'vix': vix_level, 'tlt_gap': tlt_gap, 'spy_gap': spy_gap,
              'vix_range_5d': vix_range_5d, 'spy_3d_return': spy_3d_return}
    _regime_cache[cache_key] = regime
    return regime


def _filter_pass(cfg, regime):
    vix_bad = regime['vix']     is not None and regime['vix']     > cfg['vix_threshold']
    tlt_bad = regime['tlt_gap'] is not None and regime['tlt_gap'] < cfg['tlt_crash']
    spy_bad = regime['spy_gap'] is not None and regime['spy_gap'] > cfg['spy_ignore']
    blocked = vix_bad and tlt_bad and spy_bad
    if blocked:
        return False, not vix_bad, not tlt_bad, not spy_bad

    vr = regime.get('vix_range_5d')
    s3d = regime.get('spy_3d_return')
    if vr is not None and vr >= VIX_RANGE_5D_MAX:
        return False, True, True, True
    if s3d is not None and s3d < SPY_3D_RETURN_MIN:
        return False, True, True, True

    return True, not vix_bad, not tlt_bad, not spy_bad


# ==============================================================================
# DASHBOARD
# ==============================================================================
_MODE_LABELS = {
    'static': 'STATIC', 'atr': 'ATR', 'atr_adaptive': 'ATR-ADP',
    'trailing_decay': 'TRAIL', 'trail': 'TRAIL',
    'regime_scaled': 'REGIME', 'regime': 'REGIME',
    'limit_entry': 'LIMIT',
}


def print_experiment_section(exp, results):
    W = 120
    asset_configs = exp['asset_configs']
    print()
    print(f"  {'─'*W}")
    print(f"  {exp['label']}   ({exp['model_id']})   [{len(results)} tickers]")
    print(f"  {'─'*W}")
    print(f"  {'Ticker':<6}  {'Seed':>5}  {'SIG':<7}  {'Prob':>5} / {'Thr':>5}  "
          f"{'Entry':>5}  {'Mode':<8}  {'TP target':>14}  {'SL stop':>14}  "
          f"{'Close':>8}  {'CONF':<4}  {'Filter'}")
    print(f"  {'─'*W}")

    for r in results:
        cfg = asset_configs.get(r['ticker'], {})
        trade_ok = r.get('filter_ok', True)
        conf = cfg.get('confidence', '??')

        if r['signal']:
            sig_str = ("\033[92m GO  \033[0m" if trade_ok
                       else "\033[31mBLOCK\033[0m")
        else:
            sig_str = "  --  "

        close = r['close']
        mode_label = _MODE_LABELS.get(r.get('tpsl_mode', 'static'), '??')

        dip = r.get('dip_pct', 0) or 0
        entry_price = close * (1 - dip) if r.get('tpsl_mode') == 'limit_entry' else close

        if r.get('tpsl_mode') in ('trail', 'trailing_decay') or r.get('tp_pct') is None:
            ts = r.get('trail_start', 0)
            tp_str = f"TRAIL(ts={ts:.1%})"
        else:
            tp_str = f"${entry_price*(1+r['tp_pct']):.2f} (+{r['tp_pct']*100:.0f}%)"
        sl_str = f"${entry_price*(1-r['sl_pct']):.2f} (-{r['sl_pct']*100:.0f}%)"

        if trade_ok:
            filt_str = "OK"
        else:
            parts = []
            if r.get('vix_value') is not None and cfg:
                parts.append(f"VIX:{r['vix_value']:.1f}>{cfg['vix_threshold']:.0f}")
            if r.get('tlt_gap_pct') is not None and cfg:
                parts.append(f"TLT:{r['tlt_gap_pct']:+.2f}<{cfg['tlt_crash']:+.2f}")
            if r.get('spy_gap_pct') is not None and cfg:
                parts.append(f"SPY:{r['spy_gap_pct']:+.2f}>{cfg['spy_ignore']:+.2f}")
            if not r.get('vol_filter_ok', True):
                parts.append("VOL")
            if not r.get('dollar_vol_ok', True):
                dv = r.get('dollar_volume')
                parts.append(f"DOLLAR_VOL:${dv:,.0f}<${MIN_DOLLAR_VOLUME:,.0f}" if dv else "DOLLAR_VOL")
            vr = r.get('vix_range_5d')
            if vr is not None and vr >= VIX_RANGE_5D_MAX:
                parts.append(f"VIX_RNG:{vr:.1%}>={VIX_RANGE_5D_MAX:.0%}")
            s3d = r.get('spy_3d_return')
            if s3d is not None and s3d < SPY_3D_RETURN_MIN:
                parts.append(f"SPY3d:{s3d:+.2%}<{SPY_3D_RETURN_MIN:+.1%}")
            filt_str = f"BLOCKED {' & '.join(parts)}"

        em = r['entry_minute']
        entry_str = f"{em // 60}:{em % 60:02d}"
        seed_str = 'ENS' if cfg.get('is_ensemble') else str(r['seed'])

        print(f"  {r['ticker']:<6}  {seed_str:>5}  {sig_str}  "
              f"{r['prob']:>5.3f} / {r['threshold']:>5.3f}  "
              f"{entry_str:>5}  {mode_label:<8}  {tp_str:>14}  {sl_str:>14}  "
              f"{close:>8.2f}  {conf:<4}  {filt_str}")


def print_dashboard(all_results, run_ts, strategy='S3'):
    W = 120
    pst_str = run_ts.astimezone(PST).strftime('%Y-%m-%d %H:%M PST')

    print()
    print(f"  {'='*W}")
    print(f"  DAILY SIGNALS — {strategy}   {pst_str}")
    print(f"  {'='*W}")

    all_flat_r = [r for _, results in all_results for r in results]
    if all_flat_r:
        last_r = max(all_flat_r, key=lambda r: r['entry_minute'])
        vix_s = f"{last_r['vix_value']:.1f}"       if last_r.get('vix_value')   is not None else "N/A"
        tlt_s = f"{last_r['tlt_gap_pct']:+.2f}%"   if last_r.get('tlt_gap_pct') is not None else "N/A"
        spy_s = f"{last_r['spy_gap_pct']:+.2f}%"   if last_r.get('spy_gap_pct') is not None else "N/A"
        vr_s  = f"{last_r['vix_range_5d']:.1%}"     if last_r.get('vix_range_5d') is not None else "N/A"
        s3d_s = f"{last_r['spy_3d_return']:+.2%}"    if last_r.get('spy_3d_return') is not None else "N/A"
        print(f"  Market:  VIX={vix_s}   TLT={tlt_s}   SPY={spy_s}   VIX_rng5d={vr_s}   SPY_3d={s3d_s}")

    # ── SINGLE FLAT SIGNAL TABLE ────────────────────────────────────────────
    all_rows = [(exp, r) for exp, results in all_results for r in results]
    if all_rows:
        print()
        print(f"  {'─'*W}")
        print(f"  {'Ticker':<6}  {'Exp':<5}  {'SIG':<7}  "
              f"{'Buy@':>10}  {'SL@':>10}  {'TP@':>10}  "
              f"{'Prob':>5} / {'Thr':>5}  {'Entry':>5}  "
              f"{'Close':>8}  {'CONF':<4}  {'Filter'}")
        print(f"  {'─'*W}")

        for exp, r in all_rows:
            cfg = exp.get('asset_configs', {}).get(r['ticker'], {})
            trade_ok = r.get('filter_ok', True)
            conf = cfg.get('confidence', '??')
            exp_short = exp['model_id'].split('_')[0]

            if r['signal']:
                sig_str = ("\033[92m GO  \033[0m" if trade_ok
                           else "\033[31mBLOCK\033[0m")
            else:
                sig_str = "  --  "

            close = r['close']

            dip = r.get('dip_pct', 0) or 0
            is_limit = r.get('tpsl_mode') == 'limit_entry'
            entry_price = close * (1 - dip) if is_limit else close

            buy_str = f"${entry_price:.2f}" if is_limit else f"${close:.2f}"
            sl_price = entry_price * (1 - r['sl_pct'])
            sl_str = f"${sl_price:.2f}"

            if r.get('tpsl_mode') in ('trail', 'trailing_decay') or r.get('tp_pct') is None:
                ts = r.get('trail_start', 0)
                tp_str = f"TRAIL {ts:.0%}"
            else:
                tp_price = entry_price * (1 + r['tp_pct'])
                tp_str = f"${tp_price:.2f}"

            if trade_ok:
                filt_str = "OK"
            else:
                parts = []
                if r.get('vix_value') is not None and cfg:
                    parts.append(f"VIX:{r['vix_value']:.1f}>{cfg['vix_threshold']:.0f}")
                if r.get('tlt_gap_pct') is not None and cfg:
                    parts.append(f"TLT:{r['tlt_gap_pct']:+.2f}<{cfg['tlt_crash']:+.2f}")
                if r.get('spy_gap_pct') is not None and cfg:
                    parts.append(f"SPY:{r['spy_gap_pct']:+.2f}>{cfg['spy_ignore']:+.2f}")
                if not r.get('vol_filter_ok', True):
                    parts.append("VOL")
                if not r.get('dollar_vol_ok', True):
                    dv = r.get('dollar_volume')
                    parts.append(f"DOLLAR_VOL:${dv:,.0f}<${MIN_DOLLAR_VOLUME:,.0f}" if dv else "DOLLAR_VOL")
                vr = r.get('vix_range_5d')
                if vr is not None and vr >= VIX_RANGE_5D_MAX:
                    parts.append(f"VIX_RNG:{vr:.1%}>={VIX_RANGE_5D_MAX:.0%}")
                s3d = r.get('spy_3d_return')
                if s3d is not None and s3d < SPY_3D_RETURN_MIN:
                    parts.append(f"SPY3d:{s3d:+.2%}<{SPY_3D_RETURN_MIN:+.1%}")
                filt_str = f"BLOCKED {' & '.join(parts)}"

            em = r['entry_minute']
            entry_str = f"{em // 60}:{em % 60:02d}"

            print(f"  {r['ticker']:<6}  {exp_short:<5}  {sig_str}  "
                  f"{buy_str:>10}  {sl_str:>10}  {tp_str:>10}  "
                  f"{r['prob']:>5.3f} / {r['threshold']:>5.3f}  {entry_str:>5}  "
                  f"{close:>8.2f}  {conf:<4}  {filt_str}")

    # ── SUMMARY ──────────────────────────────────────────────────────────────
    all_flat = [r for _, results in all_results for r in results]
    signals   = [r for r in all_flat if r['signal']]
    go_trades = [r for r in signals if r.get('filter_ok', True)]
    blocked   = [r for r in signals if not r.get('filter_ok', True)]

    print()
    print(f"  {'='*W}")
    print(f"  SUMMARY")
    print(f"  {'─'*W}")

    total = len(all_flat)
    print(f"  Scored:    {total} tickers")
    print(f"  Signals:   {len(signals)}/{total}" +
          (f"  ({', '.join(r['ticker'] for r in signals)})" if signals else ""))
    if blocked:
        print(f"  Blocked:   {', '.join(r['ticker'] for r in blocked)}")

    if go_trades:
        print()
        print(f"  \033[92mTRADE: {', '.join(r['ticker'] for r in go_trades)}"
              f"  ({len(go_trades)} position{'s' if len(go_trades) != 1 else ''})\033[0m")

        limit_trades = [r for r in go_trades if r.get('tpsl_mode') == 'limit_entry']
        if limit_trades:
            print()
            print(f"  {'─'*W}")
            print(f"  LIMIT ORDERS TO PLACE")
            print(f"  {'─'*W}")
            print(f"  {'Ticker':<6}  {'BUY LIMIT':>12}  {'Dip':>8}  "
                  f"{'TP target':>12}  {'SL stop':>12}  {'Cancel by':>10}  {'Exit by':>10}  {'Signal $':>10}")
            print(f"  {'─'*W}")
            for r in limit_trades:
                c = r['close']
                dip = r.get('dip_pct', 0) or 0
                lp = c * (1 - dip)
                tp_price = lp * (1 + r['tp_pct'])
                sl_price = lp * (1 - r['sl_pct'])
                wait = r.get('max_wait_minutes', 30) or 30
                em = r['entry_minute']
                expire_min = em + wait
                expire_h = expire_min // 60
                expire_ampm = "AM" if expire_h < 12 else "PM"
                expire_h12 = expire_h if expire_h <= 12 else expire_h - 12
                expire_str = f"{expire_h12}:{expire_min % 60:02d}{expire_ampm} PT"
                print(f"  {r['ticker']:<6}  "
                      f"\033[92m${lp:>9.2f}\033[0m  "
                      f"-{dip:.1%}  "
                      f"${tp_price:>9.2f}  ${sl_price:>9.2f}  "
                      f"{expire_str:>10}  {'10:00AM PT':>10}  ${c:>8.2f}")

    # ── ASSET DETAILS ────────────────────────────────────────────────────────
    print()
    print(f"  {'─'*W}")
    print(f"  ASSET DETAILS (WF optimization stats)")
    print(f"  {'─'*W}")
    print(f"  {'Ticker':<6}  {'Exp':<14}  {'CONF':<4}  {'WR':>5}  {'AvgPnL':>7}  "
          f"{'Freq':>8}  {'Mode':<8}  {'Regime':>20}  {'Note'}")
    print(f"  {'─'*W}")

    for exp, results in all_results:
        for r in results:
            cfg = exp.get('asset_configs', {}).get(r['ticker'], {})
            conf = cfg.get('confidence', '??')
            note = cfg.get('confidence_note', '')
            stats = cfg.get('stats', {})
            mode_label = _MODE_LABELS.get(cfg.get('tpsl_mode', 'static'), '??')

            vix_t = cfg.get('vix_threshold', 999)
            tlt_t = cfg.get('tlt_crash', -999)
            spy_t = cfg.get('spy_ignore', 999)
            if vix_t >= 999 and tlt_t <= -999 and spy_t >= 999:
                regime_str = "no filter"
            else:
                parts = []
                if vix_t < 999:
                    parts.append(f"VIX<{vix_t:.0f}")
                if tlt_t > -999:
                    parts.append(f"TLT>{tlt_t:+.2f}")
                if spy_t < 999:
                    parts.append(f"SPY<{spy_t:+.2f}")
                regime_str = ' '.join(parts)

            conf_color = ''
            conf_reset = ''
            if conf == 'LOW':
                conf_color = '\033[33m'
                conf_reset = '\033[0m'
            elif conf == 'MED':
                conf_color = '\033[36m'
                conf_reset = '\033[0m'

            print(f"  {r['ticker']:<6}  {exp['model_id']:<14}  "
                  f"{conf_color}{conf:<4}{conf_reset}  "
                  f"{stats.get('win_rate', '--'):>5}  {stats.get('avg_pnl', '--'):>7}  "
                  f"{stats.get('freq', '--'):>8}  {mode_label:<8}  "
                  f"{regime_str:>20}  {note}")

    print()
    print(f"  Confidence: "
          f"\033[92mHIGH\033[0m = N>=12 reliable   "
          f"\033[36mMED\033[0m = N=5-11 moderate   "
          f"\033[33mLOW\033[0m = N<=4 possible noise")
    print(f"  {'='*W}")


# ==============================================================================
# BIGQUERY LOGGER
# ==============================================================================
_BQ_SCHEMA_DEFS = [
    ("run_timestamp",  "TIMESTAMP"),
    ("bar_timestamp",  "TIMESTAMP"),
    ("score_date",     "DATE"),
    ("experiment",     "STRING"),
    ("ticker",         "STRING"),
    ("seed",           "INT64"),
    ("prob",           "FLOAT64"),
    ("threshold",      "FLOAT64"),
    ("signal",         "STRING"),
    ("tp_pct",         "FLOAT64"),
    ("sl_pct",         "FLOAT64"),
    ("entry_minute",   "INT64"),
    ("close",          "FLOAT64"),
    ("filter_status",  "STRING"),
    ("vix_value",      "FLOAT64"),
    ("tlt_gap_pct",    "FLOAT64"),
    ("spy_gap_pct",    "FLOAT64"),
    ("vol_filter_pct", "INT64"),
    ("vol_filter_ok",  "STRING"),
    ("entry_volume",   "FLOAT64"),
    ("dollar_volume",  "FLOAT64"),
    ("vol_median",     "FLOAT64"),
    ("tpsl_mode",      "STRING"),
    ("atr_pct_entry",  "FLOAT64"),
    ("vix_regime_idx", "INT64"),
    ("trail_start",    "FLOAT64"),
    ("trail_end",      "FLOAT64"),
    ("vix_range_5d",   "FLOAT64"),
    ("spy_3d_return",  "FLOAT64"),
    ("strategy",       "STRING"),
]


def _ensure_bq_schema(bq_client):
    for col_name, col_type in _BQ_SCHEMA_DEFS:
        sql = (f"ALTER TABLE `{BQ_TABLE}` "
               f"ADD COLUMN IF NOT EXISTS `{col_name}` {col_type}")
        bq_client.query(sql).result()


def log_to_bigquery(all_results, run_ts, strategy='S3'):
    if bigquery is None:
        print("   BigQuery not available — skipping")
        return
    rows = []
    for _, results in all_results:
        for r in results:
            rows.append({
                'run_timestamp': run_ts.isoformat(),
                'bar_timestamp': r['bar_ts'].isoformat(),
                'score_date':    str(r['score_date']),
                'experiment':    r['experiment'],
                'ticker':        r['ticker'],
                'seed':          r['seed'],
                'prob':          r['prob'],
                'threshold':     r['threshold'],
                'signal':        'GO' if r['signal'] else 'NO',
                'tp_pct':        r['tp_pct'],
                'sl_pct':        r['sl_pct'],
                'entry_minute':  r['entry_minute'],
                'close':         r['close'],
                'filter_status': r.get('filter_status', 'N/A'),
                'vix_value':     r.get('vix_value'),
                'tlt_gap_pct':   r.get('tlt_gap_pct'),
                'spy_gap_pct':   r.get('spy_gap_pct'),
                'vol_filter_pct': r.get('vol_filter_pct', 0),
                'vol_filter_ok':  'OK' if r.get('vol_filter_ok', True) else 'BLOCKED',
                'entry_volume':   r.get('entry_volume'),
                'dollar_volume':  r.get('dollar_volume'),
                'vol_median':     r.get('vol_median'),
                'tpsl_mode':      r.get('tpsl_mode', 'static'),
                'atr_pct_entry':  r.get('atr_pct_entry'),
                'vix_regime_idx': r.get('vix_regime_idx'),
                'trail_start':    r.get('trail_start'),
                'trail_end':      r.get('trail_end'),
                'vix_range_5d':   r.get('vix_range_5d'),
                'spy_3d_return':  r.get('spy_3d_return'),
                'strategy':       strategy,
            })
    if not rows:
        return
    bq_client = bigquery.Client()
    _ensure_bq_schema(bq_client)
    try:
        errors = bq_client.insert_rows_json(BQ_TABLE, rows)
        if errors:
            print(f"   BigQuery errors: {errors}")
        else:
            print(f"   Logged {len(rows)} rows to {BQ_TABLE}")
    except Exception as e:
        print(f"   BigQuery failed: {e}")


# ==============================================================================
# RUN ONE EXPERIMENT
# ==============================================================================
def run_experiment(exp, run_ts):
    model_id = exp['model_id']
    label    = exp['label']
    n_tickers = len(exp['asset_configs'])

    try:
        blob  = gcs_client.bucket(BUCKET_NAME).blob(exp['stats_path'])
        stats = json.loads(blob.download_as_text())
    except Exception as e:
        print(f"   {label}: FAILED — {e}")
        return []

    market_df = build_market_context(exp)

    needed_seeds = set()
    base_num = int(model_id.split('_')[0])
    for cfg in exp['asset_configs'].values():
        if cfg.get('is_ensemble', False):
            needed_seeds.update(range(base_num * 100 + 1, base_num * 100 + 16))
        else:
            needed_seeds.add(cfg['seed'])
    for seed in sorted(needed_seeds):
        try:
            get_model(exp, seed)
        except Exception as e:
            print(f"   {label}: model seed {seed} FAILED — {e}")

    now_pst = run_ts.astimezone(PST)
    now_minutes = now_pst.hour * 60 + now_pst.minute

    results = []
    for ticker, cfg in exp['asset_configs'].items():
        if now_minutes < cfg['entry_minute']:
            print(f"      {ticker}: SKIP — entry_minute {cfg['entry_minute']} > now {now_minutes} (too early)")
            continue
        df_raw = engineer_asset(ticker, market_df)
        if df_raw is None:
            print(f"      {ticker}: SKIP — engineer_asset returned None (load_master failed)")
            continue
        df_scaled = apply_scaling(df_raw, stats, exp['all_cols'])
        result = score_ticker(ticker, cfg, df_scaled, df_raw, exp)
        if result is not None:
            results.append(result)

    print(f"   {label}: scored {len(results)}/{n_tickers}")
    return results


# ==============================================================================
# MAIN
# ==============================================================================
def _check_stitch_freshness():
    from google.cloud import storage as _gcs
    blob = _gcs.Client().bucket(BUCKET_NAME).blob("Data/Inference/pipeline_status.json")
    try:
        status = json.loads(blob.download_as_text())
    except Exception:
        print("   ⚠️  pipeline_status.json not found — stitch freshness unknown")
        return
    stitch = status.get('stitch', {})
    if stitch.get('status') != 'complete':
        print("   ⚠️  daily_stitch_master has not run — master_inference may be stale")
        return
    ts = pd.Timestamp(stitch.get('timestamp', '2000-01-01'))
    if ts.tz is None:
        ts = ts.tz_localize('UTC')
    today = pd.Timestamp.now(tz='UTC').normalize()
    if ts.normalize() < today:
        print(f"   ⚠️  Stitch last ran {ts.strftime('%Y-%m-%d %H:%M')} — not today")


def _run_strategy(experiments, run_ts, strategy_name):
    """Score all tickers for one strategy. Returns list of (exp, results)."""
    total = sum(len(e['asset_configs']) for e in experiments)
    tickers = sorted({t for e in experiments for t in e['asset_configs']})
    print(f"\n  {strategy_name}: {total} assets — {', '.join(tickers)}")

    all_results = []
    for exp in experiments:
        try:
            results = run_experiment(exp, run_ts)
        except Exception as e:
            print(f"   {exp['label']}: CRASHED — {e}")
            results = []
        all_results.append((exp, results))
        _model_cache.clear()
        _master_cache.clear()
        _regime_cache.clear()
        tf.keras.backend.clear_session()
        gc.collect()

    scored = sorted({r['ticker'] for _, results in all_results for r in results})
    skipped = sorted(set(tickers) - set(scored))
    print(f"  {strategy_name}: scored {len(scored)}/{total}: {', '.join(scored)}")
    if skipped:
        print(f"  {strategy_name}: skipped {len(skipped)}: {', '.join(skipped)}")

    return all_results



# ==============================================================================
# MAIN
# ==============================================================================
def _check_stitch_freshness():
    from google.cloud import storage as _gcs
    blob = _gcs.Client().bucket(BUCKET_NAME).blob("Data/Inference/pipeline_status.json")
    try:
        status = json.loads(blob.download_as_text())
    except Exception:
        print("   ⚠️  pipeline_status.json not found — stitch freshness unknown")
        return
    stitch = status.get('stitch', {})
    if stitch.get('status') != 'complete':
        print("   ⚠️  daily_stitch_master has not run — master_inference may be stale")
        return
    ts = pd.Timestamp(stitch.get('timestamp', '2000-01-01'))
    if ts.tz is None:
        ts = ts.tz_localize('UTC')
    today = pd.Timestamp.now(tz='UTC').normalize()
    if ts.normalize() < today:
        print(f"   ⚠️  Stitch last ran {ts.strftime('%Y-%m-%d %H:%M')} — not today")


run_ts = datetime.now(pytz.utc)
pst_str = run_ts.astimezone(PST).strftime('%Y-%m-%d %I:%M %p PST')
print(f"\n{'='*80}")
print(f"  DAILY SCORING — Strategy S3 — {pst_str}")
print(f"{'='*80}")
_check_stitch_freshness()

s3_results = _run_strategy(EXPERIMENTS, run_ts, 'S3')
print_dashboard(s3_results, run_ts, strategy='S3')
log_to_bigquery(s3_results, run_ts, strategy='S3')
