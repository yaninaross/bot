"""
backtest_june_filter_impact.py — June S3 backtest: with vs without vol/dollar filters
=====================================================================================
Runs full TF model inference backtest for Strategy S3 (limit_entry + agreement),
records every trade, and tags which ones would have been blocked by the two
production-only filters we removed (MIN_VOL_FILTER_PCT=25 floor, MIN_DOLLAR_VOLUME=50K).

Day-by-day progression, per-ticker impact, dual equity curves.

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


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
MODEL_VERSION = 'robust_wf_v12'
MIN_TRADES = 3
DEPLOY_THRESHOLD = 0.018
HIGH_THRESHOLD   = 0.030
MIN_SEEDS = 3

TEST_START = '2026-04-01'
TEST_END   = '2026-06-29'
JUNE_START = '2026-06-01'
JUNE_END   = TEST_END

STARTING_CAPITAL = 200_000
SLIPPAGE_PCT = 0.0015

FILTER_VOL_PCT    = 25
FILTER_DOLLAR_VOL = 50_000

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
W = 110

TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}
TICKER_RENAME   = {'MIGI': 'BGDE', 'BITF': 'KEEL'}
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

gcs_client = gcs_storage.Client()
bq_client  = bigquery.Client()


# ══════════════════════════════════════════════════════════════════════════════
# BQ DATA + STRATEGY BUILDING (from backtest_strategies.py)
# ══════════════════════════════════════════════════════════════════════════════
def assign_family(exp):
    base = exp.split('_')[0]
    if not base.isdigit(): return 'other'
    num = int(base)
    if num == 152: return '152'
    if num == 160: return '160'
    if 170 <= num <= 175: return '170'
    if 180 <= num <= 185: return '180'
    if num == 200: return '200'
    return 'other'


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
            if sfx and exp_name.endswith(sfx): return exp_name[:-len(sfx)]
        return exp_name
    df['base_experiment'] = df['experiment'].apply(_base_exp)
    df = df[df['base_experiment'].isin(DEPLOY_EXPERIMENTS)]
    df = df[~df['ticker'].isin(EXCLUDE_TICKERS)]
    print(f"    {len(df)} rows after experiment/ticker filter")
    for col in ['grace_minutes', 'max_wait_minutes', 'vol_filter_pct']:
        if col not in df.columns: df[col] = 0
        df[col] = df[col].fillna(0).astype(int)
    for col in ['dip_pct', 'tp_mult', 'sl_mult', 'trail_start_mult', 'trail_end_mult']:
        if col not in df.columns: df[col] = 0.0
        df[col] = df[col].fillna(0.0)
    MAX_VA_OPT_RATIO = 3.0
    before_n = len(df)
    df = df[df['opt_avg_pnl'] > 0].copy()
    va_opt_ratio = df['val_a_avg_pnl'] / df['opt_avg_pnl']
    df = df[va_opt_ratio <= MAX_VA_OPT_RATIO]
    print(f"    Overfit filter: {before_n} -> {len(df)} configs")
    return df


GRP_KEYS = ['ticker', 'experiment', 'tpsl_mode', 'entry_minute',
            'prob_thresh', 'tp_pct', 'sl_pct',
            'tp_mult', 'sl_mult', 'trail_start_mult', 'trail_end_mult',
            'grace_minutes', 'dip_pct', 'max_wait_minutes',
            'vol_filter_pct']


def _build_seed_groups(df):
    df_indiv = df[df['is_ensemble'] == False].copy()
    extra_agg = {}
    for c in ['vix_threshold', 'tlt_crash', 'spy_ignore']:
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
        vb_pnl=('val_b_avg_pnl', 'mean'),
        **extra_agg,
    ).reset_index()
    sa['agree_18'] = (sa['n_pos_seeds_18'] / sa['n_seeds']).clip(upper=1.0)
    sa['agree_15'] = (sa['n_pos_seeds_15'] / sa['n_seeds']).clip(upper=1.0)
    return sa


def _row_to_config(row, wf_avg_pnl_col='val_a_avg_pnl', wf_wr_col='val_a_win_rate',
                   wf_count_col='val_a_count', wf_freq_col='val_a_freq',
                   composite_val=0.0, extra_fields=None):
    exp = row['experiment']; mode = row['tpsl_mode']
    mode_sfx = MODE_SUFFIX.get(mode, '')
    base_exp = exp[:-len(mode_sfx)] if mode_sfx and exp.endswith(mode_sfx) else exp
    is_ens = bool(row.get('is_ensemble', True))
    seed_val = int(row.get('seed', -1))
    if is_ens or seed_val == -1:
        _ecfg = EXP_CONFIGS.get(base_exp, {})
        base_num = _ecfg.get('seed_base', int(base_exp.split('_')[0]))
        seed_list = list(range(base_num * 100 + 1, base_num * 100 + 16))
        is_ens = True; seed_val = -1
    else:
        seed_list = [seed_val]
    model_id = base_exp; stats_id = base_exp
    for wf_base, wf_folds in WALKFORWARD_FOLDS.items():
        if base_exp in wf_folds:
            model_id = wf_base; stats_id = wf_base; break
    cfg = {
        'ticker': row['ticker'], 'base_experiment': base_exp,
        'experiment': exp, 'tpsl_mode': mode,
        'seed': seed_val, 'is_ensemble': is_ens, 'seed_list': seed_list,
        'entry_minute': int(row['entry_minute']),
        'prob_thresh': float(row['prob_thresh']),
        'tp_pct': float(row['tp_pct']), 'sl_pct': float(row['sl_pct']),
        'tp_mult': float(row.get('tp_mult', 3.0)),
        'sl_mult': float(row.get('sl_mult', 1.5)),
        'trail_start_mult': float(row.get('trail_start_mult', 4.0)),
        'trail_end_mult': float(row.get('trail_end_mult', 1.0)),
        'vix_threshold': float(row.get('vix_threshold') or 99),
        'tlt_crash': float(row.get('tlt_crash') or -99),
        'spy_ignore': float(row.get('spy_ignore') or 99),
        'vol_filter_pct': int(row.get('vol_filter_pct') or 0),
        'grace_minutes': int(row.get('grace_minutes') or 0),
        'dip_pct': float(row.get('dip_pct') or 0),
        'max_wait_minutes': int(row.get('max_wait_minutes') or 0),
        'model_id': model_id, 'stats_id': stats_id,
        'wf_avg_pnl': float(row.get(wf_avg_pnl_col, 0)),
        'wf_win_rate': float(row.get(wf_wr_col, 0)),
        'wf_count': float(row.get(wf_count_col, 0)),
        'wf_freq': float(row.get(wf_freq_col, 0)),
        'wf_composite': composite_val,
        'tier': '',
    }
    if extra_fields:
        cfg.update(extra_fields)
    return cfg


def _sa_to_configs(best_df, agree_col='agree_18'):
    configs = []
    for _, row in best_df.iterrows():
        cfg = _row_to_config(row, wf_avg_pnl_col='mean_va_pnl', wf_wr_col='mean_va_wr',
                             wf_count_col='mean_va_count', wf_freq_col='mean_va_freq',
                             composite_val=float(row.get('score', 0)),
                             extra_fields={'seed': -1, 'is_ensemble': True,
                                           'seed_agree_pct': float(row.get(agree_col, 0)),
                                           'n_seeds': int(row['n_seeds']), 'tier': row.get('tier', '')})
        configs.append(cfg)
    return configs


score_pnl2 = lambda d: (d['mean_va_pnl'] ** 2) * d['agree_18']


def _select_best_bt(sa, score_fn=None, thresh=DEPLOY_THRESHOLD, group_col='ticker'):
    d = sa.copy()
    if d.empty: return pd.DataFrame()
    if score_fn is None: score_fn = score_pnl2
    d['score'] = score_fn(d)
    d = d[np.isfinite(d['score'])]
    if d.empty: return pd.DataFrame()
    best = d.loc[d.groupby(group_col)['score'].idxmax()]
    return best[best['mean_va_pnl'] >= thresh]


def _compute_family_agreement_bt(sa_limit):
    d = sa_limit.copy()
    d['score'] = score_pnl2(d)
    d = d[np.isfinite(d['score'])]
    if d.empty: return {}
    d['exp_fam'] = d['experiment'].apply(assign_family)
    d = d[d['exp_fam'] != 'other']
    best = d.loc[d.groupby(['ticker', 'exp_fam'])['score'].idxmax()]
    best = best[best['mean_va_pnl'] >= DEPLOY_THRESHOLD]
    agreement = {}
    for tk in best['ticker'].unique():
        tk_rows = best[best['ticker'] == tk]
        n_fams = len(tk_rows); n_pos = int((tk_rows['vb_pnl'] > 0).sum())
        agreement[tk] = {'n_fams': n_fams, 'n_pos': n_pos,
                         'all_pos': n_pos == n_fams and n_fams >= 2,
                         'majority_pos': n_pos > n_fams / 2 and n_fams >= 2}
    return agreement


def build_s3_configs(raw_df):
    sa = _build_seed_groups(raw_df)
    sa_base = sa[sa['n_seeds'] >= MIN_SEEDS].copy()
    sa_limit = sa_base[sa_base['tpsl_mode'] == 'limit_entry'].copy()
    agreement = _compute_family_agreement_bt(sa_limit)
    maj = {tk for tk, ag in agreement.items() if ag['majority_pos']}
    best = _select_best_bt(sa_limit[sa_limit['ticker'].isin(maj)], score_fn=score_pnl2)
    if best.empty: return []
    best['tier'] = 'S3'
    return _sa_to_configs(best)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING + MODEL + SCORING (from backtest_strategies.py)
# ══════════════════════════════════════════════════════════════════════════════
_master_cache = {}
_model_cache = {}
_stats_cache = {}
_feat_cache = {}


def load_master(ticker, start_date=None):
    cache_key = (ticker, start_date)
    if cache_key in _master_cache: return _master_cache[cache_key]
    clean = TICKER_FILE_MAP.get(ticker, ticker)
    uri = f"gs://{BUCKET_NAME}/{INFERENCE_DIR}/{clean}_master_inference.parquet"
    try:
        df = pd.read_parquet(uri).reset_index()
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
        std_map = {'Total_vol': 'Total_Vol', 'Trade_count': 'Trade_Count',
                   'Volume_delta': 'Volume_Delta', 'Delta_pct': 'Delta_Pct',
                   'Obi': 'OBI', 'Bsize': 'bsize', 'Asize': 'asize',
                   'open': 'Open', 'high': 'High', 'low': 'Low',
                   'close': 'Close', 'volume': 'Volume'}
        lower_map = {k.lower(): v for k, v in std_map.items()}
        df.columns = [lower_map.get(str(c).strip().lower(), str(c).strip()) for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated(keep='first')].copy()
        for c in MICRO_COLS:
            if c not in df.columns: df[c] = 0.0
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
        if c in df.columns: df[c] = df[c].ffill()
    for c in ['Volume', 'Total_Vol', 'Volume_Delta', 'Trade_Count', 'Delta_Pct',
              'Spread', 'OBI', 'bsize', 'asize']:
        if c in df.columns: df[c] = df[c].fillna(0)
    return df


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
    in_a  = Input(shape=(TIME_STEPS, len(ASSET_COLS)), name='asset_in')
    in_c  = Input(shape=(len(CONTEXT_COLS),),          name='context_in')
    in_id = Input(shape=(1,), dtype=tf.int32,          name='asset_id')
    x = Dense(64)(in_a)
    x = SpatialDropout1D(0.2)(x)
    conv_branches = [Conv1D(64, kernel_size=k, padding='same', activation='relu')(x)
                     for k in [3, 7, 15]]
    x = Concatenate(axis=-1)(conv_branches)
    x = LayerNormalization()(x)
    x = MaxPool1D(pool_size=4)(x)
    x = SpatialDropout1D(0.1)(x)
    x = Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = MaxPool1D(pool_size=4)(x)
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
    if stats_id in _stats_cache: return _stats_cache[stats_id]
    blob = gcs_client.bucket(BUCKET_NAME).blob(
        f"{INFERENCE_DIR}/scaling_stats_experiment_{stats_id}.json")
    stats = json.loads(blob.download_as_text())
    _stats_cache[stats_id] = stats
    return stats


def get_model(model_id, seed, n_assets, ecfg=None):
    key = (model_id, seed)
    if key in _model_cache: return _model_cache[key]
    local_path = f"/tmp/UNIVERSAL_S{seed}_M{model_id}.h5"
    blob_path = f"{GCS_MODEL_PATH}/UNIVERSAL_S{seed}_M{model_id}.h5"
    if not os.path.exists(local_path):
        blob = gcs_client.bucket(BUCKET_NAME).blob(blob_path)
        if not blob.exists(): return None
        blob.download_to_filename(local_path)
    builder = (ecfg or {}).get('model_builder', 'standard')
    embed_dim = (ecfg or {}).get('embed_dim', EMBED_DIM)
    if builder == 'cnn_transformer':
        model = build_cnn_transformer_model(n_assets, embed_dim=embed_dim)
    else:
        model = build_universal_model(n_assets, embed_dim=embed_dim)
    try:
        model.load_weights(local_path)
    except OSError:
        os.remove(local_path)
        blob = gcs_client.bucket(BUCKET_NAME).blob(blob_path)
        if not blob.exists(): return None
        blob.download_to_filename(local_path)
        model.load_weights(local_path)
    _model_cache[key] = model
    return model


def build_market_context():
    print("  Building market context for TF scoring...")
    frames = []
    for ticker in CONTEXT_ASSETS:
        df = load_master(ticker)
        if df is None: continue
        df = heal_data(df.sort_index())
        daily_c = df['Close'].resample('D').last().ffill().shift(1).reindex(df.index, method='ffill')
        cols = {f'{ticker}_Anchor_ROC': (df['Close'] - daily_c) / daily_c}
        for h in ROC_HORIZONS:
            cols[f'{ticker}_ROC_{h}m'] = df['Close'].pct_change(h).fillna(0)
        frames.append(pd.DataFrame(cols, index=df.index))
        del df, daily_c; gc.collect()
    if not frames: raise RuntimeError("No context assets loaded")
    return pd.concat(frames, axis=1).ffill().fillna(0)


def engineer_features_scoring(df, market_df):
    daily_c = df['Close'].resample('D').last().ffill().shift(1).reindex(df.index, method='ffill')
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
        if col in ('Time_Pos', 'Bar_Is_Real'): continue
        if col in stats:
            df_s[col] = ((df_s[col] - stats[col]['med'])
                         / max(stats[col]['iqr'], 1e-9)).clip(-10, 10).astype('float32')
    return df_s


def get_featured_data(ticker, market_df):
    if ticker in _feat_cache: return _feat_cache[ticker]
    df = load_master(ticker, start_date=TEST_START)
    if df is None: return None
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
            if ec.get('stats_id') == cfg.get('stats_id'): ecfg = ec; break
    if ecfg is None: ecfg = EXP_CONFIGS['152_XSECTOR']
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
    if len(entry_locs) == 0: return []
    XA_list, XC_list, valid_locs = [], [], []
    for loc in entry_locs:
        if loc < TIME_STEPS: continue
        window = df_pst.iloc[loc - TIME_STEPS:loc][ASSET_COLS].values.astype(np.float32)
        context = df_pst.iloc[loc][CONTEXT_COLS].values.astype(np.float32)
        XA_list.append(window); XC_list.append(context); valid_locs.append(loc)
    if not XA_list: return []
    XA = np.array(XA_list); XC = np.array(XC_list)
    XID = np.full((len(XA_list), 1), ticker_id, dtype=np.int32)
    all_probs = np.zeros(len(XA_list), dtype=np.float64)
    n_seeds = 0
    for seed in seed_list:
        model = get_model(actual_model_id, seed, n_assets, ecfg=ecfg)
        if model is None: continue
        preds = model.predict([XA, XC, XID], verbose=0, batch_size=256)
        all_probs += preds.flatten()
        n_seeds += 1
    if n_seeds == 0: return []
    all_probs /= n_seeds
    results = []
    for i, prob in enumerate(all_probs):
        if prob >= prob_thresh:
            results.append({'bar_ts': df_pst.index[valid_locs[i]], 'prob': float(prob)})
    return results


# ══════════════════════════════════════════════════════════════════════════════
# DYNAMIC TP/SL, VOLUME, TRADE SIM, REGIME (from backtest_strategies.py)
# ══════════════════════════════════════════════════════════════════════════════
def compute_dynamic_tpsl(cfg, atr_pct_val, vix_value):
    mode = cfg['tpsl_mode']
    if mode in ('atr', 'atr_adaptive'):
        return (min(max(atr_pct_val * cfg['tp_mult'], 0.02), 0.60),
                min(max(atr_pct_val * cfg['sl_mult'], 0.01), 0.25))
    elif mode == 'regime_scaled':
        regime_idx = 2
        if vix_value is not None:
            if vix_value <= 15: regime_idx = 0
            elif vix_value <= 20: regime_idx = 1
            elif vix_value <= 25: regime_idx = 2
            elif vix_value <= 35: regime_idx = 3
            else: regime_idx = 4
        return cfg['tp_pct'] * REGIME_TP_SCALE[regime_idx], cfg['sl_pct'] * REGIME_SL_SCALE[regime_idx]
    elif mode == 'trailing_decay':
        return (min(max(atr_pct_val * cfg['trail_start_mult'], 0.02), 0.60),
                min(max(atr_pct_val * cfg['sl_mult'], 0.01), 0.25))
    elif mode == 'atr_sl':
        return cfg['tp_pct'], min(max(atr_pct_val * cfg['sl_mult'], 0.01), 0.25)
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
                baselines[(d_int, em)] = 0.0; continue
            baselines[(d_int, em)] = float(np.median(volumes[at_em[start_j:j]]))
    return baselines


def check_volume_filter(vol_baselines, d_int, entry_minute, volume, vol_filter_pct):
    if vol_filter_pct <= 0: return True
    baseline = vol_baselines.get((d_int, entry_minute), 0.0)
    if baseline <= 0: return True
    return (volume / baseline) * 100 >= vol_filter_pct


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


def precompute_ohlcv_arrays(df_ohlcv, start_date, end_date):
    s = pd.to_datetime(start_date).tz_localize('America/Los_Angeles')
    e = pd.to_datetime(end_date).tz_localize('America/Los_Angeles') + timedelta(days=1)
    df = df_ohlcv[(df_ohlcv.index >= s) & (df_ohlcv.index < e)]
    if len(df) == 0: return None
    return {'index': df.index, 'prices': df['Close'].values.copy(),
            'highs': df['High'].values.copy(), 'lows': df['Low'].values.copy(),
            'volumes': df['Volume'].values.copy(),
            'dates': (df.index.year*10000 + df.index.month*100 + df.index.day).values.astype(np.int64),
            'times': (df.index.hour*60 + df.index.minute).values.astype(np.int64)}


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
        except Exception: return None
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


# ══════════════════════════════════════════════════════════════════════════════
# MODIFIED BACKTEST — record all trades, tag filter blocks
# ══════════════════════════════════════════════════════════════════════════════
def run_backtest_with_filter_tags(configs, regime_dict, market_df):
    print(f"\n{'='*W}")
    print(f"  BACKTEST: S3 with filter tagging — {len(configs)} configs")
    print(f"{'='*W}")
    if not configs: return pd.DataFrame()

    all_trades = []
    for cfg in configs:
        ticker = cfg['ticker']; mode = cfg['tpsl_mode']; em = cfg['entry_minute']

        df_feat = get_featured_data(ticker, market_df)
        if df_feat is None: print(f"  {ticker}: SKIP (no data)"); continue
        stats = load_scaling_stats(cfg['stats_id'])
        df_scaled = apply_scaling_scoring(df_feat, stats)

        df_pst = df_scaled.copy()
        if str(df_pst.index.tz) != 'America/Los_Angeles':
            df_pst.index = df_pst.index.tz_convert(PST)

        signals = score_ticker_live(ticker, cfg, df_scaled)
        if not signals: print(f"  {ticker}: No signals"); gc.collect(); continue

        atr_series = (compute_daily_atr_series(df_pst) if mode in ('atr','trailing_decay','atr_sl')
                      else compute_adaptive_atr_series(df_pst) if mode == 'atr_adaptive' else None)
        vol_baselines = precompute_vol_baselines(df_pst)
        ohlcv = precompute_ohlcv_arrays(df_pst, TEST_START, TEST_END)
        if ohlcv is None: gc.collect(); continue

        trades = []; fr = 0; uf = 0; last_d = -1
        for sig in signals:
            bts = sig['bar_ts'].tz_convert(PST) if sig['bar_ts'].tzinfo else sig['bar_ts']
            d_int = bts.year*10000 + bts.month*100 + bts.day
            if d_int == last_d: continue

            oi = None
            for j in range(len(ohlcv['index'])):
                if ohlcv['dates'][j] == d_int and ohlcv['times'][j] == em:
                    oi = j; break
            if oi is None: continue

            entry_vol = float(ohlcv['volumes'][oi])
            entry_price = float(ohlcv['prices'][oi])
            dv = entry_vol * entry_price

            vol_ok_wf = check_volume_filter(vol_baselines, d_int, em, entry_vol, cfg['vol_filter_pct'])
            vol_ok_floor = check_volume_filter(vol_baselines, d_int, em, entry_vol,
                                                max(cfg['vol_filter_pct'], FILTER_VOL_PCT))
            dvol_ok = dv >= FILTER_DOLLAR_VOL

            rk = (d_int, em); vix_val = regime_dict[rk][0] if rk in regime_dict else None
            if rk in regime_dict:
                rv = regime_dict[rk]
                if (rv[0] > cfg.get('vix_threshold', 99) and
                    rv[1] < cfg.get('tlt_crash', -99) and
                    rv[2] > cfg.get('spy_ignore', 99)):
                    fr += 1; continue

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

            pnl, td, valid = run_single_trade_limit_entry(
                ohlcv['prices'], ohlcv['highs'], ohlcv['lows'],
                ohlcv['dates'], ohlcv['times'], oi, tp, sl,
                EXIT_TIME_MIN, cfg['dip_pct'], cfg['max_wait_minutes'])

            if not valid:
                uf += 1; continue

            pnl -= 2 * SLIPPAGE_PCT

            blocked_vol = (not vol_ok_floor) and vol_ok_wf
            blocked_dvol = not dvol_ok
            blocked_any = blocked_vol or blocked_dvol

            trades.append({
                'ticker': ticker, 'experiment': cfg['experiment'],
                'date_int': int(td), 'pnl': float(pnl), 'prob': sig['prob'],
                'tp': tp, 'sl': sl, 'mode': mode,
                'dip_pct': cfg.get('dip_pct', 0), 'max_wait': cfg.get('max_wait_minutes', 0),
                'dollar_vol': dv, 'entry_vol': entry_vol, 'entry_price': entry_price,
                'blocked_vol': blocked_vol, 'blocked_dvol': blocked_dvol,
                'blocked_any': blocked_any,
                'vix': vix_val,
            })
            last_d = int(td)

        w = sum(1 for t in trades if t['pnl'] > 0); l = len(trades) - w
        avg = np.mean([t['pnl'] for t in trades]) if trades else 0
        n_blocked = sum(1 for t in trades if t['blocked_any'])
        print(f"  {ticker:<7} {cfg['experiment']:<20} {len(signals):>3} sig -> {len(trades):>3} trades "
              f"({w}W/{l}L)  avg {avg:>+6.2%}  blocked={n_blocked}  reg_f={fr} unfilled={uf}")
        all_trades.extend(trades)
        del ohlcv; gc.collect()

    if not all_trades: return pd.DataFrame()
    tdf = pd.DataFrame(all_trades)
    tdf['date'] = pd.to_datetime(tdf['date_int'].astype(str), format='%Y%m%d')
    return tdf


# ══════════════════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*W}")
print(f"  JUNE FILTER IMPACT BACKTEST — S3 (Limit + Agreement)")
print(f"  TEST: {TEST_START} → {TEST_END}  |  June analysis: {JUNE_START} → {JUNE_END}")
print(f"  Filters under test: VOL floor={FILTER_VOL_PCT}%  DOLLAR_VOL=${FILTER_DOLLAR_VOL:,}")
print(f"{'='*W}")

raw_df = load_raw_wf_data()
configs = build_s3_configs(raw_df)
tickers = sorted(set(c['ticker'] for c in configs))
print(f"  S3: {len(configs)} configs, {len(tickers)} tickers: {tickers}")

market_df = build_market_context()
regime_dict = preload_regime_data(TEST_START, TEST_END)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print(f"  GPU ACTIVE: {len(gpus)} GPU(s)")
else:
    print(f"  WARNING: No GPU — TF scoring will be slow")

import ctypes
_libc = ctypes.CDLL("libc.so.6")

tdf = run_backtest_with_filter_tags(configs, regime_dict, market_df)
_model_cache.clear(); _feat_cache.clear(); _stats_cache.clear()
gc.collect(); _libc.malloc_trim(0)

if tdf.empty:
    print("  No trades generated — check configs")
    raise SystemExit

# ══════════════════════════════════════════════════════════════════════════════
# SPLIT: filtered vs unfiltered — full period + June
# ══════════════════════════════════════════════════════════════════════════════
june_start_int = int(JUNE_START.replace('-', ''))
june_end_int   = int(JUNE_END.replace('-', ''))

all_trades_full   = tdf.copy()
filtered_out_full = tdf[tdf['blocked_any'] == True].copy()
kept_trades_full  = tdf[tdf['blocked_any'] == False].copy()

june = tdf[(tdf['date_int'] >= june_start_int) & (tdf['date_int'] <= june_end_int)].copy()
all_trades_jun   = june.copy()
filtered_out_jun = june[june['blocked_any'] == True].copy()
kept_trades_jun  = june[june['blocked_any'] == False].copy()

def pstats(label, df):
    if df.empty:
        print(f"  {label}: 0 trades"); return
    n = len(df); w = len(df[df['pnl'] > 0]); wr = w/n*100
    avg = df['pnl'].mean(); tot = df['pnl'].sum()
    alloc = STARTING_CAPITAL / n
    dollar = sum(alloc * p for p in df['pnl'])
    print(f"  {label}:")
    print(f"    Trades: {n}, Wins: {w}, WR: {wr:.0f}%, "
          f"Avg: {avg:+.2%}, Total: {tot:+.2%}, $PnL: ${dollar:+,.0f}")

def print_comparison_table(all_t, kept_t, blocked_t):
    print(f"\n  {'Scenario':<40} {'Trades':>7} {'WR':>5} {'AvgPnL':>9} {'TotalPnL':>10} {'$PnL':>12}")
    print(f"  {'─'*85}")
    for label, df in [('ALL (no vol/dollar filters)', all_t),
                      ('WITH filters (vol+dollar)', kept_t),
                      ('BLOCKED (would be cut)', blocked_t)]:
        if df.empty:
            print(f"  {label:<40} {'0':>7}"); continue
        n = len(df); w = len(df[df['pnl'] > 0])
        alloc = STARTING_CAPITAL / n
        dollar = sum(alloc * p for p in df['pnl'])
        print(f"  {label:<40} {n:>7} {w/n:>4.0%} {df['pnl'].mean():>+8.2%} "
              f"{df['pnl'].sum():>+9.2%} ${dollar:>+11,.0f}")

# ══════════════════════════════════════════════════════════════════════════════
# FULL PERIOD RESULTS (Apr → Jun)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*W}")
print(f"  FULL PERIOD RESULTS — {TEST_START} → {TEST_END} — ALL vs FILTERED")
print(f"{'='*W}")
print()
pstats('ALL TRADES (no vol/dollar filter)', all_trades_full)
pstats('WITH FILTERS (vol 25% floor + $50K dvol)', kept_trades_full)
pstats('BLOCKED by filters (would be removed)', filtered_out_full)
print_comparison_table(all_trades_full, kept_trades_full, filtered_out_full)

# ── Monthly breakdown ──
all_trades_full['month'] = pd.to_datetime(all_trades_full['date']).dt.to_period('M')
kept_full_m = kept_trades_full.copy()
if not kept_full_m.empty:
    kept_full_m['month'] = pd.to_datetime(kept_full_m['date']).dt.to_period('M')

months = sorted(all_trades_full['month'].unique())
print(f"\n  {'Month':<10} {'|':>2} {'All N':>6} {'All WR':>7} {'All Avg':>9} {'All $PnL':>12} "
      f"{'|':>2} {'Flt N':>6} {'Flt WR':>7} {'Flt Avg':>9} {'Flt $PnL':>12} {'|':>2} {'Delta$':>10}")
print(f"  {'─'*105}")

for mo in months:
    aw = all_trades_full[all_trades_full['month'] == mo]
    kw = kept_full_m[kept_full_m['month'] == mo] if not kept_full_m.empty else pd.DataFrame()
    a_n = len(aw)
    a_wr = len(aw[aw['pnl'] > 0]) / a_n * 100 if a_n > 0 else 0
    a_avg = aw['pnl'].mean() if a_n > 0 else 0
    a_alloc = STARTING_CAPITAL / a_n if a_n > 0 else 0
    a_dollar = sum(a_alloc * p for p in aw['pnl']) if a_n > 0 else 0
    k_n = len(kw)
    k_wr = len(kw[kw['pnl'] > 0]) / k_n * 100 if k_n > 0 else 0
    k_avg = kw['pnl'].mean() if k_n > 0 else 0
    k_alloc = STARTING_CAPITAL / k_n if k_n > 0 else 0
    k_dollar = sum(k_alloc * p for p in kw['pnl']) if k_n > 0 else 0
    delta = a_dollar - k_dollar
    print(f"  {str(mo):<10} {'|':>2} {a_n:>6} {a_wr:>6.0f}% {a_avg:>+8.2%} ${a_dollar:>+11,.0f} "
          f"{'|':>2} {k_n:>6} {k_wr:>6.0f}% {k_avg:>+8.2%} ${k_dollar:>+11,.0f} {'|':>2} ${delta:>+9,.0f}")

# ── Per-ticker impact (full period) ──
print(f"\n{'='*W}")
print(f"  PER-TICKER FILTER IMPACT — Full Period {TEST_START} → {TEST_END}")
print(f"{'='*W}")
tickers_full = sorted(all_trades_full['ticker'].unique())
print(f"\n  {'Ticker':<7} {'All':>4} {'Kept':>5} {'Blk':>4} {'All Avg':>9} {'Kept Avg':>9} "
      f"{'All $PnL':>12} {'Kept $PnL':>12} {'Delta$':>10}")
print(f"  {'─'*85}")
for tk in tickers_full:
    tk_all = all_trades_full[all_trades_full['ticker'] == tk]
    tk_kept = kept_trades_full[kept_trades_full['ticker'] == tk] if not kept_trades_full.empty else pd.DataFrame()
    tk_blk = filtered_out_full[filtered_out_full['ticker'] == tk] if not filtered_out_full.empty else pd.DataFrame()
    n_a = len(tk_all); n_k = len(tk_kept); n_b = len(tk_blk)
    a_avg = tk_all['pnl'].mean() if n_a > 0 else 0
    k_avg = tk_kept['pnl'].mean() if n_k > 0 else 0
    a_alloc = STARTING_CAPITAL / max(len(all_trades_full), 1)
    k_alloc = STARTING_CAPITAL / max(len(kept_trades_full), 1)
    a_dollar = tk_all['pnl'].sum() * a_alloc if n_a > 0 else 0
    k_dollar = tk_kept['pnl'].sum() * k_alloc if n_k > 0 else 0
    delta = a_dollar - k_dollar
    marker = ' <<<' if abs(delta) > 2000 else ''
    print(f"  {tk:<7} {n_a:>4} {n_k:>5} {n_b:>4} {a_avg:>+8.2%} {k_avg:>+8.2%} "
          f"${a_dollar:>+11,.0f} ${k_dollar:>+11,.0f} ${delta:>+9,.0f}{marker}")

# ══════════════════════════════════════════════════════════════════════════════
# JUNE RESULTS
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*W}")
print(f"  JUNE RESULTS — {JUNE_START} → {JUNE_END} — ALL vs FILTERED")
print(f"{'='*W}")
print()
pstats('ALL TRADES (no vol/dollar filter)', all_trades_jun)
pstats('WITH FILTERS (vol 25% floor + $50K dvol)', kept_trades_jun)
pstats('BLOCKED by filters (would be removed)', filtered_out_jun)
print_comparison_table(all_trades_jun, kept_trades_jun, filtered_out_jun)

# ══════════════════════════════════════════════════════════════════════════════
# BLOCKED TRADES DETAIL (full period)
# ══════════════════════════════════════════════════════════════════════════════
if not filtered_out_full.empty:
    print(f"\n{'='*W}")
    print(f"  BLOCKED TRADES DETAIL — Full Period")
    print(f"{'='*W}")
    print(f"  {'Date':<12} {'Ticker':<7} {'Prob':>6} {'Entry$':>9} {'TP':>5} {'SL':>5} "
          f"{'PnL':>8} {'$Vol':>10} {'Vol?':>5} {'DVol?':>5} {'W/L':>4}")
    print(f"  {'─'*90}")
    for _, r in filtered_out_full.sort_values('date').iterrows():
        wl = 'W' if r['pnl'] > 0 else 'L'
        vf = 'BLK' if r['blocked_vol'] else 'ok'
        df_ = 'BLK' if r['blocked_dvol'] else 'ok'
        print(f"  {str(r['date'].date()):<12} {r['ticker']:<7} {r['prob']:>5.3f} "
              f"${r['entry_price']:>8.2f} {r['tp']:>4.0%} {r['sl']:>4.0%} "
              f"{r['pnl']:>+7.2%} ${r['dollar_vol']:>9,.0f} {vf:>5} {df_:>5} {wl:>4}")

# ══════════════════════════════════════════════════════════════════════════════
# WEEKLY BREAKDOWN (full period)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*W}")
print(f"  WEEKLY BREAKDOWN — Full Period {TEST_START} → {TEST_END}")
print(f"{'='*W}")

all_trades_full['week_label'] = (pd.to_datetime(all_trades_full['date']).dt.isocalendar().year.astype(str) +
                                  '-W' + pd.to_datetime(all_trades_full['date']).dt.isocalendar().week.astype(str).str.zfill(2))
kept_full_wk = kept_trades_full.copy()
if not kept_full_wk.empty:
    kept_full_wk['week_label'] = (pd.to_datetime(kept_full_wk['date']).dt.isocalendar().year.astype(str) +
                                   '-W' + pd.to_datetime(kept_full_wk['date']).dt.isocalendar().week.astype(str).str.zfill(2))

weeks = sorted(all_trades_full['week_label'].unique())

print(f"\n  {'Week':<10} {'|':>2} {'All N':>6} {'All WR':>7} {'All Avg':>9} {'All $PnL':>12} "
      f"{'|':>2} {'Flt N':>6} {'Flt WR':>7} {'Flt Avg':>9} {'Flt $PnL':>12} {'|':>2} {'Delta$':>10}")
print(f"  {'─'*107}")

for wk in weeks:
    aw = all_trades_full[all_trades_full['week_label'] == wk]
    kw = kept_full_wk[kept_full_wk['week_label'] == wk] if not kept_full_wk.empty else pd.DataFrame()
    a_n = len(aw)
    a_wr = len(aw[aw['pnl'] > 0]) / a_n * 100 if a_n > 0 else 0
    a_avg = aw['pnl'].mean() if a_n > 0 else 0
    a_alloc = STARTING_CAPITAL / a_n if a_n > 0 else 0
    a_dollar = sum(a_alloc * p for p in aw['pnl']) if a_n > 0 else 0
    k_n = len(kw)
    k_wr = len(kw[kw['pnl'] > 0]) / k_n * 100 if k_n > 0 else 0
    k_avg = kw['pnl'].mean() if k_n > 0 else 0
    k_alloc = STARTING_CAPITAL / k_n if k_n > 0 else 0
    k_dollar = sum(k_alloc * p for p in kw['pnl']) if k_n > 0 else 0
    delta = a_dollar - k_dollar
    print(f"  {wk:<10} {'|':>2} {a_n:>6} {a_wr:>6.0f}% {a_avg:>+8.2%} ${a_dollar:>+11,.0f} "
          f"{'|':>2} {k_n:>6} {k_wr:>6.0f}% {k_avg:>+8.2%} ${k_dollar:>+11,.0f} {'|':>2} ${delta:>+9,.0f}")

# ══════════════════════════════════════════════════════════════════════════════
# DAY-BY-DAY PROGRESSION (full period)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*W}")
print(f"  DAY-BY-DAY — Full Period {TEST_START} → {TEST_END}")
print(f"{'='*W}")

dates_full = sorted(all_trades_full['date'].unique())
cum_all = 0.0
cum_flt = 0.0

for d in dates_full:
    day_all = all_trades_full[all_trades_full['date'] == d]
    day_kept = kept_trades_full[kept_trades_full['date'] == d] if not kept_trades_full.empty else pd.DataFrame()
    day_blocked = filtered_out_full[filtered_out_full['date'] == d] if not filtered_out_full.empty else pd.DataFrame()

    n_all = len(day_all); n_kept = len(day_kept); n_blk = len(day_blocked)
    alloc_all = STARTING_CAPITAL / n_all if n_all > 0 else 0
    alloc_kept = STARTING_CAPITAL / n_kept if n_kept > 0 else 0
    dollar_all = sum(alloc_all * p for p in day_all['pnl']) if n_all > 0 else 0
    dollar_kept = sum(alloc_kept * p for p in day_kept['pnl']) if n_kept > 0 else 0
    cum_all += dollar_all
    cum_flt += dollar_kept
    delta = dollar_all - dollar_kept

    is_june = d >= pd.Timestamp(JUNE_START)
    month_tag = ' [JUN]' if is_june else ''

    print(f"\n  ┌─ {d.date()}{month_tag} ───────────────────────────────────────────────────────────────┐")
    print(f"  │  ALL: {n_all} trades ${dollar_all:+,.0f} (cum ${cum_all:+,.0f})  │  "
          f"FILTERED: {n_kept} trades ${dollar_kept:+,.0f} (cum ${cum_flt:+,.0f})  │  "
          f"Δ ${delta:+,.0f}")
    print(f"  └{'─'*(W-3)}┘")

    for _, t in day_all.sort_values('ticker').iterrows():
        wl = 'W' if t['pnl'] > 0 else 'L'
        tag = ''
        if t['blocked_vol']: tag = ' [VOL-BLOCKED]'
        elif t['blocked_dvol']: tag = ' [DVOL-BLOCKED]'
        print(f"    {wl} {t['ticker']:<7} {t['pnl']:>+7.2%}  entry ${t['entry_price']:.2f}  "
              f"prob {t['prob']:.3f}  dv=${t['dollar_vol']:,.0f}{tag}")

# ══════════════════════════════════════════════════════════════════════════════
# PER-TICKER IMPACT — June only
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*W}")
print(f"  PER-TICKER FILTER IMPACT — June Only")
print(f"{'='*W}")

tickers_jun = sorted(all_trades_jun['ticker'].unique())
print(f"\n  {'Ticker':<7} {'All':>4} {'Kept':>5} {'Blk':>4} {'All Avg':>9} {'Kept Avg':>9} "
      f"{'All $PnL':>12} {'Kept $PnL':>12} {'Delta$':>10}")
print(f"  {'─'*85}")

for tk in tickers_jun:
    tk_all = all_trades_jun[all_trades_jun['ticker'] == tk]
    tk_kept = kept_trades_jun[kept_trades_jun['ticker'] == tk] if not kept_trades_jun.empty else pd.DataFrame()
    tk_blk = filtered_out_jun[filtered_out_jun['ticker'] == tk] if not filtered_out_jun.empty else pd.DataFrame()
    n_a = len(tk_all); n_k = len(tk_kept); n_b = len(tk_blk)
    a_avg = tk_all['pnl'].mean() if n_a > 0 else 0
    k_avg = tk_kept['pnl'].mean() if n_k > 0 else 0
    a_alloc = STARTING_CAPITAL / max(len(all_trades_jun), 1)
    k_alloc = STARTING_CAPITAL / max(len(kept_trades_jun), 1)
    a_dollar = tk_all['pnl'].sum() * a_alloc if n_a > 0 else 0
    k_dollar = tk_kept['pnl'].sum() * k_alloc if n_k > 0 else 0
    delta = a_dollar - k_dollar
    marker = ' <<<' if abs(delta) > 2000 else ''
    print(f"  {tk:<7} {n_a:>4} {n_k:>5} {n_b:>4} {a_avg:>+8.2%} {k_avg:>+8.2%} "
          f"${a_dollar:>+11,.0f} ${k_dollar:>+11,.0f} ${delta:>+9,.0f}{marker}")

# ══════════════════════════════════════════════════════════════════════════════
# CHARTS — Full period
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n  Generating charts...")

all_daily_dollar = []
flt_daily_dollar = []
all_daily_n = []
flt_daily_n = []

for d in dates_full:
    da = all_trades_full[all_trades_full['date'] == d]
    dk = kept_trades_full[kept_trades_full['date'] == d] if not kept_trades_full.empty else pd.DataFrame()
    n_a = len(da); n_k = len(dk)
    a_alloc = STARTING_CAPITAL / n_a if n_a > 0 else 0
    k_alloc = STARTING_CAPITAL / n_k if n_k > 0 else 0
    all_daily_dollar.append(sum(a_alloc * p for p in da['pnl']) if n_a > 0 else 0)
    flt_daily_dollar.append(sum(k_alloc * p for p in dk['pnl']) if n_k > 0 else 0)
    all_daily_n.append(n_a)
    flt_daily_n.append(n_k)

date_labels = [d.strftime('%b %d') for d in dates_full]
x = np.arange(len(dates_full))
bar_w = 0.35

june_start_idx = next((i for i, d in enumerate(dates_full) if d >= pd.Timestamp(JUNE_START)), len(dates_full))

fig, axes = plt.subplots(3, 1, figsize=(20, 14))
fig.suptitle(f'S3 Filter Impact — Full Backtest {TEST_START} → {TEST_END}\n'
             f'VOL floor={FILTER_VOL_PCT}% | DOLLAR_VOL=${FILTER_DOLLAR_VOL:,}',
             fontsize=14, fontweight='bold')

# Chart 1: Dual cumulative equity
all_cum = np.cumsum(all_daily_dollar)
flt_cum = np.cumsum(flt_daily_dollar)
axes[0].plot(x, [STARTING_CAPITAL + c for c in all_cum], 'o-',
             label='ALL trades (no vol/dvol filter)', color='#4CAF50', linewidth=2, markersize=3)
axes[0].plot(x, [STARTING_CAPITAL + c for c in flt_cum], 's-',
             label='WITH filters (vol+dvol)', color='#F44336', linewidth=2, markersize=3)
axes[0].axhline(y=STARTING_CAPITAL, color='gray', linewidth=0.5, linestyle='--')
axes[0].axvline(x=june_start_idx - 0.5, color='blue', linewidth=1, linestyle='--', alpha=0.5, label='June start')
axes[0].fill_between(x, [STARTING_CAPITAL + c for c in all_cum],
                     [STARTING_CAPITAL + c for c in flt_cum],
                     alpha=0.15, color='green',
                     where=[a >= f for a, f in zip(all_cum, flt_cum)])
axes[0].fill_between(x, [STARTING_CAPITAL + c for c in all_cum],
                     [STARTING_CAPITAL + c for c in flt_cum],
                     alpha=0.15, color='red',
                     where=[a < f for a, f in zip(all_cum, flt_cum)])
axes[0].set_title(f'Cumulative Equity (${STARTING_CAPITAL/1000:.0f}K start)', fontsize=12)
axes[0].set_ylabel('Equity ($)')
axes[0].set_xticks(x[::max(1, len(x)//30)])
axes[0].set_xticklabels([date_labels[i] for i in range(0, len(date_labels), max(1, len(x)//30))],
                         rotation=45, ha='right', fontsize=7)
axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'${v/1000:.0f}K'))

# Chart 2: Daily $PnL bars
axes[1].bar(x - bar_w/2, all_daily_dollar, bar_w, label='ALL trades', color='#4CAF50', alpha=0.85)
axes[1].bar(x + bar_w/2, flt_daily_dollar, bar_w, label='WITH filters', color='#F44336', alpha=0.85)
axes[1].axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
axes[1].axvline(x=june_start_idx - 0.5, color='blue', linewidth=1, linestyle='--', alpha=0.5)
axes[1].set_title('Daily $PnL', fontsize=12)
axes[1].set_ylabel('$PnL')
axes[1].set_xticks(x[::max(1, len(x)//30)])
axes[1].set_xticklabels([date_labels[i] for i in range(0, len(date_labels), max(1, len(x)//30))],
                         rotation=45, ha='right', fontsize=7)
axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3, axis='y')

# Chart 3: Trade count
axes[2].bar(x - bar_w/2, all_daily_n, bar_w, label='ALL trades', color='#4CAF50', alpha=0.85)
axes[2].bar(x + bar_w/2, flt_daily_n, bar_w, label='WITH filters', color='#F44336', alpha=0.85)
axes[2].axvline(x=june_start_idx - 0.5, color='blue', linewidth=1, linestyle='--', alpha=0.5)
axes[2].set_title('Daily Trade Count', fontsize=12)
axes[2].set_ylabel('Trades')
axes[2].set_xticks(x[::max(1, len(x)//30)])
axes[2].set_xticklabels([date_labels[i] for i in range(0, len(date_labels), max(1, len(x)//30))],
                         rotation=45, ha='right', fontsize=7)
axes[2].legend(fontsize=9); axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
chart_path = '/tmp/filter_impact_backtest_full.png'
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
print(f"  Charts saved: {chart_path}")
try:
    from IPython.display import display, Image
    display(Image(filename=chart_path))
except ImportError:
    plt.show()

# ── Final summary ──
print(f"\n{'='*W}")
print(f"  CONCLUSION")
print(f"{'='*W}")

for period_label, at, kt, fo in [
    (f'Full ({TEST_START} → {TEST_END})', all_trades_full, kept_trades_full, filtered_out_full),
    (f'June ({JUNE_START} → {JUNE_END})', all_trades_jun, kept_trades_jun, filtered_out_jun),
]:
    if at.empty or kt.empty: continue
    a_alloc = STARTING_CAPITAL / len(at)
    k_alloc = STARTING_CAPITAL / len(kt)
    a_dollar = sum(a_alloc * p for p in at['pnl'])
    k_dollar = sum(k_alloc * p for p in kt['pnl'])
    delta = a_dollar - k_dollar
    print(f"\n  {period_label}:")
    print(f"    ALL trades:        {len(at):>3} trades, ${a_dollar:>+10,.0f}")
    print(f"    WITH filters:      {len(kt):>3} trades, ${k_dollar:>+10,.0f}")
    print(f"    Blocked:           {len(fo):>3} trades")
    print(f"    Filter impact:     ${delta:>+10,.0f} ({'filters HURT' if delta > 0 else 'filters HELPED'})")

print(f"\n{'='*W}")
print(f"  DONE")
print(f"{'='*W}")
