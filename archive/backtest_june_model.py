"""
backtest_june_model.py — June 2026 Model-Based Backtest (S3 Strategy)
=====================================================================
Regenerates buy signals from TF model inference (15-seed ensemble)
for every business day in June 2026. No CSV/BQ score dependency.

Pipeline per ticker: BQ config → master_inference → feature engineering
→ scaling → batch TF scoring → regime/volume filter → limit_entry trade sim.

Colab-ready (GPU recommended). Run as a single cell.
"""

import os
import sys
import json
import gc
import subprocess
import ctypes
import warnings
from datetime import datetime, timedelta

os.environ['TF_USE_LEGACY_KERAS'] = '1'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pytz
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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
                              Embedding, Flatten, Concatenate)
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
JUNE_START = '2026-06-01'
JUNE_END   = '2026-06-30'

CAPITAL        = 200_000
SLIPPAGE_PCT   = 0.0015
EXIT_TIME_MIN  = 780

MODEL_VERSION    = 'robust_wf_v12'
MIN_TRADES       = 3
DEPLOY_THRESHOLD = 0.018
MIN_SEEDS        = 3

BUCKET_NAME    = os.environ.get("GCS_BUCKET", "")
BQ_TABLE       = f"{GCP_PROJECT}.scoring_logs.wf_optimization_results"
INFERENCE_DIR  = "Data/Inference_v2"
GCS_MODEL_PATH = "Algos/to_deploy/Saved_models_v2/UNIVERSAL/Direction"

PST          = pytz.timezone('America/Los_Angeles')
TIME_STEPS   = 240
EMBED_DIM    = 16
ROC_HORIZONS = [5, 30, 60, 120]
VOL_LOOKBACK_DAYS = 30

TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}
TICKER_RENAME   = {'MIGI': 'BGDE', 'BITF': 'KEEL'}

MICRO_COLS = ['Spread', 'OBI', 'Volume_Delta', 'Trade_Count', 'Delta_Pct',
              'bsize', 'asize', 'Total_Vol']

ASSET_COLS = (['Gap_Pct', 'RSI_14', 'RSI_60', 'RVOL', 'Compression',
               'ATR_14', 'Dist_to_PM_High', 'Bar_Is_Real'] +
              [f'ROC_{h}m' for h in ROC_HORIZONS] + MICRO_COLS)

CONTEXT_ASSETS = ['SPY', 'VIX', 'TLT', 'IWM']
CONTEXT_COLS = []
for _t in CONTEXT_ASSETS:
    CONTEXT_COLS.append(f'{_t}_Anchor_ROC')
    for _h in ROC_HORIZONS:
        CONTEXT_COLS.append(f'{_t}_ROC_{_h}m')
CONTEXT_COLS += ['Time_Pos']
ALL_FEATURE_COLS = ASSET_COLS + CONTEXT_COLS

MODE_SUFFIX = {'static': '', 'atr': '_ATR', 'atr_adaptive': '_ATRADP',
               'regime_scaled': '_REGIME', 'trailing_decay': '_TRAIL',
               'regime_adaptive': '_REGADP', 'atr_sl': '_ATRSL',
               'grace': '_GRACE', 'limit_entry': '_LIMIT'}

W = 100


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════
TRAIN_TEACHERS = [
    'BBAI', 'RCAT', 'LUNR', 'ASTS', 'ACHR', 'MVIS',
    'WULF', 'CIFR', 'IREN', 'KEEL', 'HUT', 'CLSK', 'APLD',
]

TEACHERS_31 = [
    'CIFR', 'IREN', 'KEEL', 'HUT', 'APLD', 'WULF', 'RIOT', 'CLSK', 'BTBT', 'ALAB', 'MARA',
    'RCAT', 'ASTS', 'LUNR', 'BBAI', 'RKLB', 'SPIR', 'BKSY', 'MNTS', 'PLUG', 'BE', 'FCEL', 'FLNC',
    'OKLO', 'SMR', 'MVIS', 'BGDE', 'RDW', 'PL', 'VOYG', 'SMLR',
]

DEPLOY_EXPERIMENTS = {'152_XSECTOR', '160_XSECTOR', '170_XSECTOR',
                      '175_XSECTOR', '180_XSECTOR', '185_XSECTOR',
                      '200_XSECTOR'}

WALKFORWARD_FOLDS = {
    '175_XSECTOR': ['170_XSECTOR', '171_XSECTOR', '172_XSECTOR',
                     '173_XSECTOR', '174_XSECTOR', '175_XSECTOR'],
    '185_XSECTOR': ['180_XSECTOR', '181_XSECTOR', '182_XSECTOR',
                     '183_XSECTOR', '184_XSECTOR', '185_XSECTOR'],
}

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
# BQ CONFIG LOADING + S3 GATE
# ══════════════════════════════════════════════════════════════════════════════
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


def load_raw_wf_data():
    print(f"  Loading WF results from BQ (model_version={MODEL_VERSION})...")
    query = f"""
    SELECT ticker, experiment, seed, tpsl_mode, entry_time, entry_minute,
           prob_thresh, tp_pct, sl_pct, tp_mult, sl_mult,
           trail_start_mult, trail_end_mult,
           vix_threshold, tlt_crash, spy_ignore, vol_filter_pct,
           val_a_avg_pnl, val_a_win_rate, val_a_count, val_a_freq,
           val_b_avg_pnl,
           opt_avg_pnl, is_ensemble, survived_a,
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
    print(f"    {len(df)} rows after experiment filter")

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
    df = df[df['val_a_avg_pnl'] / df['opt_avg_pnl'] <= MAX_VA_OPT_RATIO]
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
        mean_va_pnl=('val_a_avg_pnl', 'mean'),
        mean_va_wr=('val_a_win_rate', 'mean'),
        mean_va_count=('val_a_count', 'mean'),
        mean_va_freq=('val_a_freq', 'mean'),
        vb_pnl=('val_b_avg_pnl', 'mean'),
        **extra_agg,
    ).reset_index()
    sa['agree_18'] = (sa['n_pos_seeds_18'] / sa['n_seeds']).clip(upper=1.0)
    return sa


def _compute_family_agreement(sa_limit):
    d = sa_limit.copy()
    d['score'] = (d['mean_va_pnl'] ** 2) * d['agree_18']
    d = d[np.isfinite(d['score'])]
    if d.empty:
        return {}
    d['exp_fam'] = d['experiment'].apply(assign_family)
    d = d[d['exp_fam'] != 'other']
    best = d.loc[d.groupby(['ticker', 'exp_fam'])['score'].idxmax()]
    best = best[best['mean_va_pnl'] >= DEPLOY_THRESHOLD]
    agreement = {}
    for tk in best['ticker'].unique():
        tk_rows = best[best['ticker'] == tk]
        n_fams = len(tk_rows)
        n_pos = int((tk_rows['vb_pnl'] > 0).sum())
        agreement[tk] = {
            'n_fams': n_fams, 'n_pos': n_pos,
            'majority_pos': n_pos > n_fams / 2 and n_fams >= 2,
        }
    return agreement


def _select_best(sa, thresh=DEPLOY_THRESHOLD):
    d = sa.copy()
    if d.empty:
        return pd.DataFrame()
    d['score'] = (d['mean_va_pnl'] ** 2) * d['agree_18']
    d = d[np.isfinite(d['score'])]
    if d.empty:
        return pd.DataFrame()
    best = d.loc[d.groupby('ticker')['score'].idxmax()]
    return best[best['mean_va_pnl'] >= thresh]


def _row_to_config(row):
    exp = row['experiment']
    mode = row['tpsl_mode']
    mode_sfx = MODE_SUFFIX.get(mode, '')
    base_exp = exp[:-len(mode_sfx)] if mode_sfx and exp.endswith(mode_sfx) else exp

    _ecfg = EXP_CONFIGS.get(base_exp, {})
    base_num = _ecfg.get('seed_base', int(base_exp.split('_')[0]))
    seed_list = list(range(base_num * 100 + 1, base_num * 100 + 16))

    model_id = base_exp
    stats_id = base_exp
    for wf_base, wf_folds in WALKFORWARD_FOLDS.items():
        if base_exp in wf_folds:
            model_id = wf_base
            stats_id = wf_base
            break

    return {
        'ticker': row['ticker'],
        'base_experiment': base_exp,
        'experiment': exp,
        'tpsl_mode': mode,
        'seed_list': seed_list,
        'entry_minute': int(row['entry_minute']),
        'prob_thresh': float(row['prob_thresh']),
        'tp_pct': float(row.get('tp_pct') or 0),
        'sl_pct': float(row.get('sl_pct') or 0),
        'dip_pct': float(row.get('dip_pct') or 0),
        'max_wait_minutes': int(row.get('max_wait_minutes') or 0),
        'vix_threshold': float(row.get('vix_threshold') or 99),
        'tlt_crash': float(row.get('tlt_crash') or -99),
        'spy_ignore': float(row.get('spy_ignore') or 99),
        'vol_filter_pct': int(row.get('vol_filter_pct') or 0),
        'model_id': model_id,
        'stats_id': stats_id,
        'n_seeds': int(row['n_seeds']),
        'mean_va_pnl': float(row['mean_va_pnl']),
    }


def build_s3_configs(raw_df):
    sa = _build_seed_groups(raw_df)
    sa_base = sa[sa['n_seeds'] >= MIN_SEEDS].copy()
    sa_limit = sa_base[sa_base['tpsl_mode'] == 'limit_entry'].copy()

    agreement = _compute_family_agreement(sa_limit)
    maj = {tk for tk, ag in agreement.items() if ag['majority_pos']}
    best = _select_best(sa_limit[sa_limit['ticker'].isin(maj)])

    if best.empty:
        return []

    configs = []
    for _, row in best.iterrows():
        configs.append(_row_to_config(row))
    return configs


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING + FEATURES + SCALING
# ══════════════════════════════════════════════════════════════════════════════
_master_cache = {}
_stats_cache = {}
_feat_cache = {}


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


def build_market_context():
    print("  Building market context...")
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
    df_pst = df.index.tz_convert(PST) if str(df.index.tz) != 'America/Los_Angeles' else df.index
    mins = df_pst.hour * 60 + df_pst.minute
    df['Time_Pos'] = np.clip((mins - 390) / 390, 0, 1)
    df['Bar_Is_Real'] = (df['Volume'] > 0).astype(float)
    return df.replace([np.inf, -np.inf], np.nan).fillna(0)


def load_scaling_stats(stats_id):
    if stats_id in _stats_cache:
        return _stats_cache[stats_id]
    blob = gcs_client.bucket(BUCKET_NAME).blob(
        f"{INFERENCE_DIR}/scaling_stats_experiment_{stats_id}.json")
    stats = json.loads(blob.download_as_text())
    _stats_cache[stats_id] = stats
    return stats


def apply_scaling(df, stats):
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
    df = load_master(ticker, start_date=JUNE_START)
    if df is None:
        return None
    df = heal_data(df.sort_index())
    df = engineer_features(df, market_df)
    _feat_cache[ticker] = df
    return df


# ══════════════════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE + LOADING
# ══════════════════════════════════════════════════════════════════════════════
_model_cache = {}


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


def get_model(model_id, seed, n_assets, ecfg=None):
    key = (model_id, seed)
    if key in _model_cache:
        return _model_cache[key]
    local_path = f"/tmp/UNIVERSAL_S{seed}_M{model_id}.h5"
    blob_path = f"{GCS_MODEL_PATH}/UNIVERSAL_S{seed}_M{model_id}.h5"
    if not os.path.exists(local_path):
        blob = gcs_client.bucket(BUCKET_NAME).blob(blob_path)
        if not blob.exists():
            return None
        blob.download_to_filename(local_path)
    embed_dim = (ecfg or {}).get('embed_dim', EMBED_DIM)
    model = build_universal_model(n_assets, embed_dim=embed_dim)
    try:
        model.load_weights(local_path)
    except OSError:
        os.remove(local_path)
        blob = gcs_client.bucket(BUCKET_NAME).blob(blob_path)
        if not blob.exists():
            return None
        blob.download_to_filename(local_path)
        model.load_weights(local_path)
    _model_cache[key] = model
    return model


# ══════════════════════════════════════════════════════════════════════════════
# SCORING ENGINE (batch historical)
# ══════════════════════════════════════════════════════════════════════════════
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
    test_start_pst = pd.Timestamp(JUNE_START, tz=PST)
    test_end_pst = pd.Timestamp(JUNE_END, tz=PST) + pd.Timedelta(days=1)

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
        results.append({
            'bar_ts': df_pst.index[valid_locs[i]],
            'prob': float(prob),
            'signal': prob >= prob_thresh,
        })
    return results


# ══════════════════════════════════════════════════════════════════════════════
# REGIME FILTER
# ══════════════════════════════════════════════════════════════════════════════
def preload_regime_data():
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

    start = pd.Timestamp(JUNE_START, tz=PST) - timedelta(days=10)
    end = pd.Timestamp(JUNE_END, tz=PST) + timedelta(days=1)

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
        for em in range(358, 393):
            v = _med(vix_lk, d_int, em) or 15.0
            sc = _med(spy_lk, d_int, em); sp = spy_prev.get(d_int)
            s_gap = ((sc - sp) / sp * 100) if sc and sp and sp != 0 else 0.0
            tc = _med(tlt_lk, d_int, em); tp = tlt_prev.get(d_int)
            t_gap = ((tc - tp) / tp * 100) if tc and tp and tp != 0 else 0.0
            regime_dict[(d_int, em)] = np.array([v, t_gap, s_gap])
    print(f"    Regime data: {len(regime_dict)} combos")
    return regime_dict


# ══════════════════════════════════════════════════════════════════════════════
# VOLUME FILTER
# ══════════════════════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════════════════════
# TRADE SIMULATION (numba-jitted limit entry)
# ══════════════════════════════════════════════════════════════════════════════
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


def precompute_ohlcv_arrays(df_ohlcv):
    s = pd.to_datetime(JUNE_START).tz_localize('America/Los_Angeles')
    e = pd.to_datetime(JUNE_END).tz_localize('America/Los_Angeles') + timedelta(days=1)
    df = df_ohlcv[(df_ohlcv.index >= s) & (df_ohlcv.index < e)]
    if len(df) == 0: return None
    return {
        'index': df.index, 'prices': df['Close'].values.copy(),
        'highs': df['High'].values.copy(), 'lows': df['Low'].values.copy(),
        'volumes': df['Volume'].values.copy(),
        'dates': (df.index.year*10000 + df.index.month*100 + df.index.day).values.astype(np.int64),
        'times': (df.index.hour*60 + df.index.minute).values.astype(np.int64),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN BACKTEST
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*W}")
print(f"  JUNE 2026 MODEL-BASED BACKTEST — Strategy S3")
print(f"  {JUNE_START} -> {JUNE_END}  |  Model: {MODEL_VERSION}  |  Threshold: {DEPLOY_THRESHOLD:.1%}")
print(f"  Capital: ${CAPITAL:,}  |  Slippage: {SLIPPAGE_PCT:.2%}")
print(f"{'='*W}")

raw_df = load_raw_wf_data()
configs = build_s3_configs(raw_df)
tickers = sorted(set(c['ticker'] for c in configs))
print(f"\n  S3 gate: {len(configs)} configs, {len(tickers)} tickers: {tickers}")

market_df = build_market_context()
regime_dict = preload_regime_data()

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print(f"  GPU ACTIVE: {len(gpus)} GPU(s)")
else:
    print(f"  WARNING: No GPU — TF scoring will be slow")

_libc = ctypes.CDLL("libc.so.6")

all_trades = []
all_signals = []

for cfg in configs:
    ticker = cfg['ticker']
    em = cfg['entry_minute']

    df_feat = get_featured_data(ticker, market_df)
    if df_feat is None:
        print(f"  {ticker}: SKIP (no data)"); continue

    stats = load_scaling_stats(cfg['stats_id'])
    df_scaled = apply_scaling(df_feat, stats)

    df_pst = df_scaled.copy()
    if str(df_pst.index.tz) != 'America/Los_Angeles':
        df_pst.index = df_pst.index.tz_convert(PST)

    signals = score_ticker_live(ticker, cfg, df_scaled)
    if not signals:
        print(f"  {ticker}: No entry bars"); gc.collect(); continue

    vol_baselines = precompute_vol_baselines(df_pst)
    ohlcv = precompute_ohlcv_arrays(df_pst)
    if ohlcv is None: gc.collect(); continue

    trades = []; fv = 0; fr = 0; uf = 0; last_d = -1
    n_above = sum(1 for s in signals if s['signal'])
    n_below = len(signals) - n_above

    for sig in signals:
        bts = sig['bar_ts']
        if bts.tzinfo and str(bts.tzinfo) != 'America/Los_Angeles':
            bts = bts.tz_convert(PST)
        d_int = bts.year*10000 + bts.month*100 + bts.day

        sig_record = {
            'ticker': ticker, 'date_int': d_int,
            'date': bts.normalize(), 'prob': sig['prob'],
            'signal': sig['signal'], 'experiment': cfg['experiment'],
            'entry_minute': em,
        }

        if not sig['signal']:
            sig_record['status'] = 'below_threshold'
            all_signals.append(sig_record)
            continue

        if d_int == last_d:
            sig_record['status'] = 'duplicate_day'
            all_signals.append(sig_record)
            continue

        oi = None
        for j in range(len(ohlcv['index'])):
            if ohlcv['dates'][j] == d_int and ohlcv['times'][j] == em:
                oi = j; break

        rk = (d_int, em)
        vix_val = regime_dict[rk][0] if rk in regime_dict else None
        tlt_gap = regime_dict[rk][1] if rk in regime_dict else None
        spy_gap = regime_dict[rk][2] if rk in regime_dict else None

        regime_ok = True
        if rk in regime_dict:
            rv = regime_dict[rk]
            vix_bad = rv[0] > cfg.get('vix_threshold', 99)
            tlt_bad = rv[1] < cfg.get('tlt_crash', -99)
            spy_bad = rv[2] > cfg.get('spy_ignore', 99)
            if vix_bad or tlt_bad or spy_bad:
                regime_ok = False

        vol_ok = True
        if oi is not None:
            vol_ok = check_volume_filter(vol_baselines, d_int, em,
                                          float(ohlcv['volumes'][oi]), cfg['vol_filter_pct'])

        sig_record['vix'] = vix_val
        sig_record['tlt_gap'] = tlt_gap
        sig_record['spy_gap'] = spy_gap
        sig_record['regime_ok'] = regime_ok
        sig_record['vol_ok'] = vol_ok
        sig_record['filter_ok'] = regime_ok and vol_ok

        if not regime_ok:
            fr += 1
            sig_record['status'] = 'regime_filtered'

        if not vol_ok:
            fv += 1
            sig_record['status'] = sig_record.get('status', 'vol_filtered')

        if oi is None:
            sig_record['status'] = 'no_bar'
            all_signals.append(sig_record)
            continue

        pnl, td, valid = run_single_trade_limit_entry(
            ohlcv['prices'], ohlcv['highs'], ohlcv['lows'],
            ohlcv['dates'], ohlcv['times'],
            oi, cfg['tp_pct'], cfg['sl_pct'], EXIT_TIME_MIN,
            cfg['dip_pct'], cfg['max_wait_minutes'])

        if not valid:
            uf += 1
            sig_record['status'] = sig_record.get('status', 'unfilled')
            all_signals.append(sig_record)
            continue

        pnl -= 2 * SLIPPAGE_PCT
        sig_record['pnl'] = float(pnl)
        sig_record['status'] = sig_record.get('status', 'filled')

        if regime_ok and vol_ok:
            trades.append({
                'ticker': ticker, 'experiment': cfg['experiment'],
                'date_int': int(td), 'pnl': float(pnl), 'prob': sig['prob'],
                'tp': cfg['tp_pct'], 'sl': cfg['sl_pct'],
                'dip_pct': cfg['dip_pct'], 'max_wait': cfg['max_wait_minutes'],
                'vix': vix_val, 'tlt_gap': tlt_gap, 'spy_gap': spy_gap,
            })
            last_d = int(td)

        all_signals.append(sig_record)

    w = sum(1 for t in trades if t['pnl'] > 0); l = len(trades) - w
    avg = np.mean([t['pnl'] for t in trades]) if trades else 0
    extra = ""
    if fv > 0: extra += f" vol_f={fv}"
    if fr > 0: extra += f" reg_f={fr}"
    if uf > 0: extra += f" unfilled={uf}"
    print(f"  {ticker:<7} {cfg['experiment']:<22} {n_above:>3} sig -> {len(trades):>3} trades "
          f"({w}W/{l}L)  avg {avg:>+6.2%}{extra}")
    all_trades.extend(trades)
    del ohlcv; gc.collect()
    _libc.malloc_trim(0)

_model_cache.clear()
_feat_cache.clear()
_stats_cache.clear()
gc.collect()
_libc.malloc_trim(0)


# ══════════════════════════════════════════════════════════════════════════════
# BUILD DATAFRAMES
# ══════════════════════════════════════════════════════════════════════════════
if not all_trades:
    print(f"\n  NO TRADES — check configs and data")
else:
    tdf = pd.DataFrame(all_trades)
    tdf['date'] = pd.to_datetime(tdf['date_int'].astype(str), format='%Y%m%d')

    sig_df = pd.DataFrame(all_signals) if all_signals else pd.DataFrame()

    # ══════════════════════════════════════════════════════════════════════════
    # DAILY $PNL HELPER
    # ══════════════════════════════════════════════════════════════════════════
    def _add_daily_dollar(df):
        daily_n = df.groupby('date')['pnl'].transform('count')
        df['dollar_pnl'] = (CAPITAL / daily_n) * df['pnl']
        return df

    tdf = _add_daily_dollar(tdf)

    # Also compute nofilter trades (signals that passed prob but failed filter)
    nofilter_trades = []
    if not sig_df.empty:
        filtered_sigs = sig_df[
            (sig_df['signal'] == True) &
            (sig_df['status'].isin(['regime_filtered', 'vol_filtered'])) &
            (sig_df['pnl'].notna())
        ]
        for _, s in filtered_sigs.iterrows():
            nofilter_trades.append({
                'ticker': s['ticker'], 'experiment': s['experiment'],
                'date_int': s['date_int'], 'date': s['date'],
                'pnl': s['pnl'], 'prob': s['prob'],
                'status': s['status'],
            })

    # ══════════════════════════════════════════════════════════════════════════
    # 1. SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    n = len(tdf); w = int((tdf['pnl'] > 0).sum()); l = n - w
    total_dollar = tdf['dollar_pnl'].sum()

    print(f"\n\n{'='*W}")
    print(f"  SUMMARY — S3 Strategy, June {JUNE_START} → {JUNE_END}")
    print(f"{'='*W}")
    print(f"  Tickers:    {tdf['ticker'].nunique()}")
    print(f"  Trades:     {n}  ({w}W / {l}L)")
    print(f"  Win Rate:   {w/n*100:.0f}%")
    print(f"  Avg PnL:    {tdf['pnl'].mean():+.2%}")
    print(f"  Total PnL:  {tdf['pnl'].sum():+.2%}")
    print(f"  Dollar PnL: ${total_dollar:+,.0f}  (on ${CAPITAL:,})")
    print(f"  Avg $/trade: ${total_dollar/n:+,.0f}")

    SL_TOL = 0.005
    sl_c = len(tdf[(tdf['pnl'] + tdf['sl'] + 2*SLIPPAGE_PCT).abs() < SL_TOL])
    tp_c = len(tdf[(tdf['pnl'] - tdf['tp'] + 2*SLIPPAGE_PCT).abs() < SL_TOL])
    eod_c = n - sl_c - tp_c
    print(f"  Exit types: TP={tp_c}  SL={sl_c}  EOD={eod_c}")

    # ══════════════════════════════════════════════════════════════════════════
    # 2. FILTER IMPACT
    # ══════════════════════════════════════════════════════════════════════════
    if not sig_df.empty:
        go = sig_df[sig_df['signal'] == True]
        n_go = len(go)
        n_filled = len(go[go['status'] == 'filled'])
        n_regime = len(go[go['status'] == 'regime_filtered'])
        n_vol = len(go[go['status'] == 'vol_filtered'])
        n_unfilled = len(go[go['status'] == 'unfilled'])
        n_nobar = len(go[go['status'] == 'no_bar'])

        print(f"\n  ── FILTER IMPACT ──")
        print(f"  Total GO signals (prob >= thresh):  {n_go}")
        print(f"    Filled trades:     {n_filled:>4}  ({n_filled/n_go*100:.0f}%)")
        print(f"    Regime blocked:    {n_regime:>4}  ({n_regime/n_go*100:.0f}%)")
        print(f"    Volume blocked:    {n_vol:>4}  ({n_vol/n_go*100:.0f}%)")
        print(f"    Unfilled (limit):  {n_unfilled:>4}  ({n_unfilled/n_go*100:.0f}%)")
        if n_nobar > 0:
            print(f"    No bar:            {n_nobar:>4}  ({n_nobar/n_go*100:.0f}%)")

        if nofilter_trades:
            nf_df = pd.DataFrame(nofilter_trades)
            nf_w = int((nf_df['pnl'] > 0).sum())
            print(f"\n  Blocked signals that WOULD have filled: {len(nf_df)}")
            print(f"    WR: {nf_w/len(nf_df)*100:.0f}%  avg PnL: {nf_df['pnl'].mean():+.2%}")

    # ══════════════════════════════════════════════════════════════════════════
    # 3. DAILY BREAKDOWN
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n\n{'='*W}")
    print(f"  DAILY BREAKDOWN")
    print(f"{'='*W}")
    print(f"  {'Date':<12} {'Trades':>7} {'Wins':>5} {'Loss':>5} {'WR':>5} "
          f"{'AvgPnL':>8} {'TP':>3} {'SL':>3} {'EOD':>4} {'$PnL':>12} {'Cum$PnL':>12}")
    print(f"  {'─'*80}")

    cum_dollar = 0
    for d in sorted(tdf['date'].unique()):
        day = tdf[tdf['date'] == d]
        nd = len(day); wd = int((day['pnl'] > 0).sum()); ld = nd - wd
        wr = wd / nd * 100 if nd > 0 else 0
        avg = day['pnl'].mean() * 100
        sl_d = len(day[(day['pnl'] + day['sl'] + 2*SLIPPAGE_PCT).abs() < SL_TOL])
        tp_d = len(day[(day['pnl'] - day['tp'] + 2*SLIPPAGE_PCT).abs() < SL_TOL])
        eod_d = nd - sl_d - tp_d
        day_dollar = day['dollar_pnl'].sum()
        cum_dollar += day_dollar
        print(f"  {d.strftime('%Y-%m-%d'):<12} {nd:>7} {wd:>5} {ld:>5} {wr:>4.0f}% "
              f"{avg:>+7.2f}% {tp_d:>3} {sl_d:>3} {eod_d:>4} ${day_dollar:>+11,.0f} ${cum_dollar:>+11,.0f}")

    print(f"\n  TOTAL: {n} trades, {w}W/{l}L ({w/n*100:.0f}% WR), "
          f"avg={tdf['pnl'].mean():+.2%}, ${total_dollar:+,.0f}")

    # ══════════════════════════════════════════════════════════════════════════
    # 4. WEEKLY BREAKDOWN
    # ══════════════════════════════════════════════════════════════════════════
    tdf['week'] = tdf['date'].dt.isocalendar().week.astype(int)
    weeks = sorted(tdf['week'].unique())

    print(f"\n\n{'='*W}")
    print(f"  WEEKLY BREAKDOWN")
    print(f"{'='*W}")
    print(f"  {'Week':>6} {'Dates':<25} {'N':>4} {'W/L':>7} {'WR':>5} "
          f"{'AvgPnL':>8} {'$PnL':>12}")
    print(f"  {'─'*70}")

    for wk in weeks:
        wt = tdf[tdf['week'] == wk]
        nw = len(wt); ww = int((wt['pnl'] > 0).sum()); lw = nw - ww
        wr = ww / nw * 100 if nw > 0 else 0
        d_range = f"{wt['date'].min().strftime('%m/%d')}-{wt['date'].max().strftime('%m/%d')}"
        w_dollar = wt['dollar_pnl'].sum()
        print(f"  {wk:>6} {d_range:<25} {nw:>4} {ww:>3}/{lw:<3} {wr:>4.0f}% "
              f"{wt['pnl'].mean():>+7.2%} ${w_dollar:>+11,.0f}")

    # ══════════════════════════════════════════════════════════════════════════
    # 5. PER-TICKER
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n\n{'='*W}")
    print(f"  PER-TICKER STATS")
    print(f"{'='*W}")
    print(f"  {'Ticker':<7} {'Experiment':<22} {'N':>4} {'W/L':>7} "
          f"{'WR':>5} {'AvgPnL':>8} {'TotPnL':>9} {'$PnL':>12}")
    print(f"  {'─'*80}")

    for tk in sorted(tdf['ticker'].unique()):
        tt = tdf[tdf['ticker'] == tk]
        nt = len(tt); wt = int((tt['pnl'] > 0).sum()); lt = nt - wt
        exp = tt['experiment'].iloc[0]
        tk_dollar = tt['dollar_pnl'].sum()
        print(f"  {tk:<7} {exp:<22} {nt:>4} {wt:>3}/{lt:<3} "
              f"{wt/nt*100:>4.0f}% {tt['pnl'].mean():>+7.2%} "
              f"{tt['pnl'].sum():>+8.2%} ${tk_dollar:>+11,.0f}")

    # ══════════════════════════════════════════════════════════════════════════
    # 6. FULL TRADE LOG
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n\n{'='*W}")
    print(f"  FULL TRADE LOG")
    print(f"{'='*W}")
    print(f"  {'Date':<12} {'Ticker':<7} {'Experiment':<22} {'PnL':>8} "
          f"{'Prob':>6} {'TP':>6} {'SL':>6} {'W/L':>4} {'$PnL':>10}")
    print(f"  {'─'*85}")

    for _, t in tdf.sort_values('date').iterrows():
        wl = 'W' if t['pnl'] > 0 else 'L'
        print(f"  {t['date'].strftime('%Y-%m-%d'):<12} {t['ticker']:<7} {t['experiment']:<22} "
              f"{t['pnl']:>+7.2%} {t['prob']:>5.2f} {t['tp']:>5.1%} {t['sl']:>5.1%} "
              f"{wl:>4} ${t['dollar_pnl']:>+9,.0f}")

    # ══════════════════════════════════════════════════════════════════════════
    # 7. CUMULATIVE EQUITY CHART
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n  Generating cumulative equity chart...")
    fig, ax = plt.subplots(figsize=(14, 6))

    daily_pnl = tdf.groupby('date')['dollar_pnl'].sum().sort_index()
    cum_pnl = daily_pnl.cumsum()
    equity = CAPITAL + cum_pnl

    ax.fill_between(equity.index, CAPITAL/1000, equity.values/1000,
                    where=equity.values >= CAPITAL, alpha=0.15, color='green')
    ax.fill_between(equity.index, CAPITAL/1000, equity.values/1000,
                    where=equity.values < CAPITAL, alpha=0.15, color='red')
    ax.plot(equity.index, equity.values/1000, color='#2196F3', linewidth=2,
            label=f'S3 Equity (${total_dollar:+,.0f})')
    ax.axhline(y=CAPITAL/1000, color='gray', linewidth=0.5, linestyle='--')

    for d in sorted(tdf['date'].unique()):
        day = tdf[tdf['date'] == d]
        day_pnl = day['dollar_pnl'].sum()
        n_trades = len(day)
        if abs(day_pnl) > 3000:
            ax.annotate(f'{n_trades}t', xy=(d, equity.loc[d]/1000),
                        fontsize=7, ha='center', va='bottom' if day_pnl > 0 else 'top',
                        alpha=0.6)

    ax.set_title(f'S3 Model Backtest — June 2026\n'
                 f'{n} trades, {w/n*100:.0f}% WR, ${total_dollar:+,.0f}',
                 fontsize=13, fontweight='bold')
    ax.set_ylabel('Equity ($K)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator())
    ax.tick_params(axis='x', rotation=45)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    chart_path = '/tmp/backtest_june_model.png'
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    print(f"  Chart saved: {chart_path}")
    try:
        from IPython.display import display, Image
        display(Image(filename=chart_path))
    except ImportError:
        plt.show()

    print(f"\n{'='*W}")
    print(f"  BACKTEST COMPLETE")
    print(f"{'='*W}")
