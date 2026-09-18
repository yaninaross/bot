"""
backtest_optimization_live.py — Optimization Strategy Comparison (Live TF Inference)
=====================================================================================
Same 11 strategies × 3 composites = 33 combos as backtest_optimization_csv.py,
but replaces CSV score loading with live TF model inference.

This solves the CSV date-range problem: CSVs end at July 2nd, so July shows
0 trades for most strategies. Live inference scores through TEST_END.

Strategies (11):
  BQ-loaded:  STATIC, GATE lb=7, ADAPTIVE THRESH, MOMENTUM 10d, VOL GATE
  Grid-search: DECAY hl=30d, DECAY hl=60d, HYBRID 30d+gate10, HYBRID 60d+gate10,
               HYBRID 30d+gate21, SELECTIVE DECAY

Composite methods (3):
  S3 Deployed:     BQ best experiment signal, S3 tickers only (benchmark)
  Weighted>50%:    Precision-weighted vote across all experiments
  Adapt Wt>50%:    Same as weighted but with adaptive prob_thresh

Colab-ready (GPU recommended for inference speed).
"""

import os, io, gc, sys, json, warnings, ctypes, math
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np, pandas as pd, pytz

os.environ['TF_USE_LEGACY_KERAS'] = '1'
warnings.filterwarnings('ignore')

import subprocess

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

for _gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(_gpu, True)

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

import matplotlib
try:
    from google.colab import auth
    _IN_COLAB = True
except ImportError:
    matplotlib.use('Agg')
    _IN_COLAB = False
import matplotlib.pyplot as plt

from google.cloud import storage as gcs_storage, bigquery

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
# CONFIG
# ==============================================================================
BUCKET_NAME = os.environ.get("GCS_BUCKET", "")
INFERENCE_DIR = "Data/Inference_v2"

DATA_START = '2025-12-01'
TEST_START = '2026-04-01'
TEST_END   = '2026-07-19'
ENTRY_TIME = '06:15'
ENTRY_MIN  = 375
EXIT_TIME_MIN = 780
SLIPPAGE_PCT = 0.0015
DEPLOY_THRESHOLD = 0.018
REOPT_FREQ = 5

PROB_GRID = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
TP_GRID   = [0.06, 0.10, 0.16, 0.22]
SL_GRID   = [0.03, 0.06, 0.10, 0.14]
MIN_OPT_TRADES = 3
MIN_OPT_PNL = 0.005

PST = pytz.timezone('America/Los_Angeles')
TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}
LN2 = math.log(2)

ALL_TICKERS = []
MIN_SEEDS = 3
MAX_VA_OPT_RATIO = 3.0

GRP_KEYS = ['ticker', 'experiment', 'tpsl_mode', 'entry_minute',
            'prob_thresh', 'tp_pct', 'sl_pct',
            'tp_mult', 'sl_mult', 'trail_start_mult', 'trail_end_mult',
            'grace_minutes', 'dip_pct', 'max_wait_minutes',
            'vol_filter_pct']

# TF scoring config
GCS_MODEL_PATH = "Algos/to_deploy/Saved_models_v2/UNIVERSAL/Direction"
TIME_STEPS = 240
EMBED_DIM = 16
ROC_HORIZONS = [5, 30, 60, 120]
DATA_START_DATE = "2022-01-01"
CONTEXT_ASSETS = ['SPY', 'VIX', 'TLT', 'IWM']

MICRO_COLS = ['Spread', 'OBI', 'Volume_Delta', 'Trade_Count', 'Delta_Pct',
              'bsize', 'asize', 'Total_Vol']

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

gcs_client = gcs_storage.Client()
bq_client = bigquery.Client()
W = 120

# ==============================================================================
# TEACHER LISTS
# ==============================================================================
TRAIN_TEACHERS = [
    'BBAI', 'RCAT', 'LUNR', 'ASTS', 'ACHR', 'MVIS',
    'WULF', 'CIFR', 'IREN', 'KEEL', 'HUT', 'CLSK', 'APLD',
]

TEACHERS_31 = [
    'CIFR', 'IREN', 'KEEL', 'HUT', 'APLD', 'WULF', 'RIOT', 'CLSK', 'BTBT', 'ALAB', 'MARA',
    'RCAT', 'ASTS', 'LUNR', 'BBAI', 'RKLB', 'SPIR', 'BKSY', 'MNTS', 'PLUG', 'BE', 'FCEL', 'FLNC',
    'OKLO', 'SMR', 'MVIS', 'BGDE', 'RDW', 'PL', 'VOYG', 'SMLR',
]

TEACHERS_47 = TEACHERS_31 + [
    'AMD', 'NVDA', 'TSLA', 'PLTR', 'SOUN', 'HOOD', 'SMCI', 'VRT', 'NBIS',
    'KTOS', 'AVAV', 'ACHR', 'JOBY', 'NNDM', 'CCJ', 'CEG',
]

# ==============================================================================
# EXPERIMENT CONFIGS
# ==============================================================================
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
    '220_XSECTOR': {
        'train_teachers': TEACHERS_47,
        'n_assets': len(TEACHERS_47) + 1,
        'stats_id': '220_XSECTOR',
    },
}

for _fold_num in range(201, 206):
    _fold_id = f'{_fold_num}_XSECTOR'
    EXP_CONFIGS[_fold_id] = {
        'train_teachers': TEACHERS_31,
        'n_assets': len(TEACHERS_31) + 1,
        'stats_id': '200_XSECTOR',
        'model_id_override': '200_XSECTOR',
        'seed_base': 200,
        'model_builder': 'cnn_transformer',
        'embed_dim': 32,
    }

for _fold_num in range(171, 176):
    _fold_id = f'{_fold_num}_XSECTOR'
    if _fold_id not in EXP_CONFIGS:
        EXP_CONFIGS[_fold_id] = {
            'train_teachers': TRAIN_TEACHERS,
            'n_assets': len(TRAIN_TEACHERS) + 1,
            'stats_id': '175_XSECTOR',
            'model_id_override': '175_XSECTOR',
            'seed_base': 175,
        }

for _fold_num in range(181, 186):
    _fold_id = f'{_fold_num}_XSECTOR'
    if _fold_id not in EXP_CONFIGS:
        EXP_CONFIGS[_fold_id] = {
            'train_teachers': TEACHERS_31,
            'n_assets': len(TEACHERS_31) + 1,
            'stats_id': '185_XSECTOR',
            'model_id_override': '185_XSECTOR',
            'seed_base': 185,
        }

for _fold_num in range(221, 226):
    _fold_id = f'{_fold_num}_XSECTOR'
    EXP_CONFIGS[_fold_id] = {
        'train_teachers': TEACHERS_47,
        'n_assets': len(TEACHERS_47) + 1,
        'stats_id': '220_XSECTOR',
        'model_id_override': '220_XSECTOR',
        'seed_base': 220,
    }

for _eid, _ecfg in EXP_CONFIGS.items():
    _idx = {t: i for i, t in enumerate(_ecfg['train_teachers'])}
    if 'BITF' in _idx: _idx['KEEL'] = _idx['BITF']
    if 'MIGI' in _idx: _idx['BGDE'] = _idx['MIGI']
    _ecfg['asset_index'] = _idx
    _ecfg['other_idx'] = len(_ecfg['train_teachers'])

WALKFORWARD_FOLDS = {
    '175_XSECTOR': ['170_XSECTOR', '171_XSECTOR', '172_XSECTOR',
                     '173_XSECTOR', '174_XSECTOR', '175_XSECTOR'],
    '185_XSECTOR': ['180_XSECTOR', '181_XSECTOR', '182_XSECTOR',
                     '183_XSECTOR', '184_XSECTOR', '185_XSECTOR'],
    '200_XSECTOR': ['200_XSECTOR', '201_XSECTOR', '202_XSECTOR',
                     '203_XSECTOR', '204_XSECTOR', '205_XSECTOR'],
    '220_XSECTOR': ['220_XSECTOR', '221_XSECTOR', '222_XSECTOR',
                     '223_XSECTOR', '224_XSECTOR', '225_XSECTOR'],
}


# ==============================================================================
# DATA LOADING
# ==============================================================================
_master_cache = {}


def load_master(ticker):
    if ticker in _master_cache:
        return _master_cache[ticker]
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
        cutoff = pd.Timestamp(DATA_START, tz='UTC') - pd.Timedelta(days=5)
        df = df[df.index >= cutoff]
        _master_cache[ticker] = df
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
# TF MODEL BUILDERS
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


# ==============================================================================
# TF SCORING INFRASTRUCTURE
# ==============================================================================
_model_cache = {}
_stats_cache = {}
_feat_cache = {}


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
    blob_path = f"{GCS_MODEL_PATH}/UNIVERSAL_S{seed}_M{model_id}.h5"
    if not os.path.exists(local_path):
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
    df = load_master(ticker)
    if df is None:
        return None
    df = heal_data(df.sort_index())
    df = engineer_features_scoring(df, market_df)
    _feat_cache[ticker] = df
    return df


def score_ticker_for_experiment(ticker, entry_minute, base_exp, df_scaled):
    """Batch-score a ticker for one base experiment. Returns {date: prob}."""
    ecfg = EXP_CONFIGS.get(base_exp)
    if ecfg is None:
        return {}

    n_assets = ecfg['n_assets']
    ticker_id = ecfg['asset_index'].get(ticker, ecfg['other_idx'])
    actual_model_id = ecfg.get('model_id_override', base_exp)
    seed_base_num = ecfg.get('seed_base', int(base_exp.split('_')[0]))
    seed_list = list(range(seed_base_num * 100 + 1, seed_base_num * 100 + 16))

    df_pst = df_scaled.copy()
    if str(df_pst.index.tz) != 'America/Los_Angeles':
        df_pst.index = df_pst.index.tz_convert(PST)
    times = df_pst.index.hour * 60 + df_pst.index.minute

    data_start_pst = pd.Timestamp(DATA_START, tz=PST)
    test_end_pst = pd.Timestamp(TEST_END, tz=PST) + pd.Timedelta(days=1)

    mask = ((df_pst.index >= data_start_pst) & (df_pst.index < test_end_pst) &
            (times == entry_minute) & (df_pst['Volume'] > 0))
    entry_locs = np.where(mask)[0]
    if len(entry_locs) == 0:
        return {}

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
        return {}

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
        return {}
    all_probs /= n_seeds

    result = {}
    for i, prob in enumerate(all_probs):
        dt = df_pst.index[valid_locs[i]]
        d = dt.date()
        result[d] = float(prob)
    return result


# ==============================================================================
# TRADE SIMULATION — limit_entry mode
# ==============================================================================
def precompute_ohlcv_arrays(df, start_date, end_date):
    s = pd.to_datetime(start_date).tz_localize('America/Los_Angeles')
    e = pd.to_datetime(end_date).tz_localize('America/Los_Angeles') + timedelta(days=1)
    idx = df.index
    if str(idx.tz) != 'America/Los_Angeles':
        idx = idx.tz_convert('America/Los_Angeles')
    mask = (idx >= s) & (idx < e)
    df_cut = df[mask]
    if len(df_cut) == 0:
        return None
    idx_cut = df_cut.index if str(df_cut.index.tz) == 'America/Los_Angeles' \
        else df_cut.index.tz_convert('America/Los_Angeles')
    return {
        'prices': df_cut['Close'].values.astype(np.float64).copy(),
        'highs':  df_cut['High'].values.astype(np.float64).copy(),
        'lows':   df_cut['Low'].values.astype(np.float64).copy(),
        'dates':  (idx_cut.year * 10000 + idx_cut.month * 100 + idx_cut.day).values.astype(np.int64),
        'times':  (idx_cut.hour * 60 + idx_cut.minute).values.astype(np.int64),
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


def assign_family(exp):
    base = exp.split('_')[0]
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


# ==============================================================================
# COMPOSITE GATE FUNCTIONS
# ==============================================================================
def apply_s3_signal(config, ticker, test_dates,
                    daily_probs, daily_trades, bq_configs, s3_tickers):
    if config is None or ticker not in s3_tickers:
        return []
    tp, sl = config['tp'], config['sl']
    prob_thresh = config['prob_thresh']
    bq_exp = bq_configs[ticker]['experiment']
    pk = (ticker, bq_exp)
    if pk not in daily_probs:
        return []
    probs = daily_probs[pk]
    trade_key = (ticker, tp, sl)
    if trade_key not in daily_trades:
        return []
    outcomes = daily_trades[trade_key]
    trades = []
    for d in test_dates:
        if d in probs and probs[d] >= prob_thresh and d in outcomes:
            trades.append({'date': d, 'ticker': ticker,
                           'pnl': outcomes[d], 'n_exps': 1})
    return trades


def apply_weighted_vote_opt(config, ticker, test_dates,
                            daily_probs, daily_trades,
                            active_exps, exp_precision):
    if config is None:
        return []
    tp, sl = config['tp'], config['sl']
    prob_thresh = config['prob_thresh']
    trade_key = (ticker, tp, sl)
    if trade_key not in daily_trades:
        return []
    outcomes = daily_trades[trade_key]
    trades = []
    for d in test_dates:
        total_weight = 0.0
        agree_weight = 0.0
        for exp in active_exps:
            pk = (ticker, exp)
            if pk not in daily_probs or d not in daily_probs[pk]:
                continue
            w = exp_precision.get((ticker, exp), 0.5)
            total_weight += w
            if daily_probs[pk][d] >= prob_thresh:
                agree_weight += w
        if total_weight > 0 and agree_weight / total_weight > 0.50:
            if d in outcomes:
                trades.append({'date': d, 'ticker': ticker,
                               'pnl': outcomes[d], 'n_exps': 0})
    return trades


def apply_composite(cfg, ticker, dates, method,
                    daily_probs, daily_trades, active_exps,
                    bq_configs, s3_tickers, exp_precision):
    if method == 'S3 Deployed':
        return apply_s3_signal(cfg, ticker, dates,
                               daily_probs, daily_trades, bq_configs, s3_tickers)
    elif method == 'Weighted>50%':
        return apply_weighted_vote_opt(cfg, ticker, dates,
                                       daily_probs, daily_trades,
                                       active_exps, exp_precision)
    elif method == 'Adapt Wt>50%':
        return apply_weighted_vote_opt(cfg, ticker, dates,
                                       daily_probs, daily_trades,
                                       active_exps, exp_precision)
    return []


# ==============================================================================
# GRID SEARCH FUNCTIONS
# ==============================================================================
def _eval_config(ticker, tp, sl, prob_thresh, opt_dates,
                 daily_probs, daily_trades, active_exps, weights=None):
    trade_key = (ticker, tp, sl)
    if trade_key not in daily_trades:
        return 0.0, 0.0
    outcomes = daily_trades[trade_key]
    w_pnl = 0.0
    w_n = 0.0
    for exp in active_exps:
        prob_key = (ticker, exp)
        if prob_key not in daily_probs:
            continue
        probs = daily_probs[prob_key]
        for d in opt_dates:
            if d in probs and d in outcomes:
                if probs[d] >= prob_thresh:
                    w = weights[d] if weights else 1.0
                    w_pnl += w * outcomes[d]
                    w_n += w
    return w_pnl, w_n


def optimize_configs(opt_dates, ticker, config_list,
                     daily_probs, daily_trades, active_exps, weights=None):
    best_pnl = -999
    best_config = None
    for entry_time_str, tp, sl, prob_thresh in config_list:
        w_pnl, w_n = _eval_config(ticker, tp, sl, prob_thresh, opt_dates,
                                   daily_probs, daily_trades, active_exps, weights)
        if w_n >= MIN_OPT_TRADES:
            avg = w_pnl / w_n
            if avg > best_pnl and avg > MIN_OPT_PNL:
                best_pnl = avg
                best_config = {
                    'entry_time': entry_time_str, 'prob_thresh': prob_thresh,
                    'tp': tp, 'sl': sl, 'avg_pnl': avg, 'n_trades': w_n,
                }
    return best_config


def summarize(trades_list, label):
    if not trades_list:
        return {'label': label, 'n': 0, 'total': 0, 'avg': 0, 'wr': 0,
                'dd': 0,
                'j_n': 0, 'j_total': 0, 'j_avg': 0, 'j_wr': 0,
                'jul_n': 0, 'jul_total': 0, 'jul_avg': 0, 'jul_wr': 0,
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
        'jul_wr': (july['pnl'] > 0).mean() if len(july) else 0,
        'monthly': df.groupby('month').agg(
            n=('pnl', 'count'), total=('pnl', 'sum'),
            avg=('pnl', 'mean'), wr=('pnl', lambda x: (x > 0).mean())
        ).to_dict('index'),
        'trades_df': df,
    }


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
print("=" * W)
print(f"  OPTIMIZATION STRATEGY COMPARISON — 11 strategies × 3 composites (Live TF)")
print("=" * W)


# ── Step 1: Load BQ configs + S3 gate detection ──
print(f"\n  STEP 1: Loading BQ configs (limit_entry, individual seeds)...")
q = """
SELECT ticker, experiment, seed, tpsl_mode, entry_minute,
       prob_thresh, tp_pct, sl_pct, tp_mult, sl_mult,
       trail_start_mult, trail_end_mult, grace_minutes,
       dip_pct, max_wait_minutes, vol_filter_pct,
       val_a_avg_pnl, val_a_win_rate, val_a_count, val_a_freq,
       val_b_avg_pnl, opt_avg_pnl, is_ensemble
FROM `{GCP_PROJECT}.scoring_logs.wf_optimization_results`
WHERE model_version IN ('robust_wf_v12', 'robust_wf_v12_stitched')
  AND survived_a = TRUE
  AND val_a_count >= 3
  AND val_a_avg_pnl > 0
"""
raw_df = bq_client.query(q).to_dataframe()
print(f"    {len(raw_df)} raw BQ rows loaded")

raw_df = raw_df[raw_df['opt_avg_pnl'] > 0].copy()
raw_df = raw_df[raw_df['val_a_avg_pnl'] / raw_df['opt_avg_pnl'] <= MAX_VA_OPT_RATIO]
print(f"    {len(raw_df)} after overfit filter")

indiv = raw_df[raw_df['is_ensemble'] == False].copy()
for k in GRP_KEYS:
    if k not in indiv.columns:
        indiv[k] = 0
    else:
        indiv[k] = indiv[k].fillna(0)

sa = indiv.groupby(GRP_KEYS).agg(
    n_seeds=('seed', 'nunique'),
    n_pos_seeds_18=('val_a_avg_pnl', lambda x: (x >= DEPLOY_THRESHOLD).sum()),
    mean_va_pnl=('val_a_avg_pnl', 'mean'),
    vb_pnl=('val_b_avg_pnl', 'mean'),
).reset_index()
sa['agree_18'] = (sa['n_pos_seeds_18'] / sa['n_seeds']).clip(upper=1.0)

sa = sa[sa['n_seeds'] >= MIN_SEEDS].copy()
sa_limit = sa[sa['tpsl_mode'] == 'limit_entry'].copy()
print(f"    {len(sa_limit)} limit_entry seed groups (n_seeds >= {MIN_SEEDS})")

sa_limit['score'] = (sa_limit['mean_va_pnl'] ** 2) * sa_limit['agree_18']

# S3 family agreement gate
sa_limit['exp_fam'] = sa_limit['experiment'].apply(assign_family)
sa_fam = sa_limit[sa_limit['exp_fam'] != 'other'].copy()
best_per_fam = sa_fam.loc[sa_fam.groupby(['ticker', 'exp_fam'])['score'].idxmax()]
best_per_fam = best_per_fam[best_per_fam['mean_va_pnl'] >= DEPLOY_THRESHOLD]

agreement = {}
for tk in best_per_fam['ticker'].unique():
    tk_rows = best_per_fam[best_per_fam['ticker'] == tk]
    n_fams = len(tk_rows)
    n_pos = int((tk_rows['vb_pnl'] > 0).sum())
    agreement[tk] = {
        'n_fams': n_fams, 'n_pos': n_pos,
        'majority_pos': n_pos > n_fams / 2 and n_fams >= 2,
    }

s3_tickers = sorted([tk for tk, ag in agreement.items() if ag['majority_pos']])
print(f"    S3 family agreement: {len(s3_tickers)} tickers pass gate")
print(f"    S3 tickers: {', '.join(s3_tickers)}")

_qualifying = sa_limit[sa_limit['mean_va_pnl'] >= DEPLOY_THRESHOLD]['ticker'].unique()
ALL_TICKERS = sorted(set(_qualifying) | set(s3_tickers))
print(f"    Dynamic ALL_TICKERS: {len(ALL_TICKERS)} tickers")

bq_configs = {}
for ticker in ALL_TICKERS:
    tk_rows = sa_limit[sa_limit['ticker'] == ticker].copy()
    if tk_rows.empty:
        bq_configs[ticker] = None
        continue
    tk_rows['score'] = (tk_rows['mean_va_pnl'] ** 2) * tk_rows['agree_18']
    best = tk_rows.loc[tk_rows['score'].idxmax()]
    if best['mean_va_pnl'] < DEPLOY_THRESHOLD:
        bq_configs[ticker] = None
        continue
    bq_configs[ticker] = {
        'entry_time': ENTRY_TIME,
        'prob_thresh': float(best['prob_thresh']),
        'tp': float(best['tp_pct']),
        'sl': float(best['sl_pct']),
        'avg_pnl': float(best['mean_va_pnl']),
        'experiment': best['experiment'],
        'dip_pct': float(best['dip_pct']),
        'max_wait_minutes': float(best['max_wait_minutes']),
        'entry_minute': int(best['entry_minute']) if pd.notna(best['entry_minute']) else ENTRY_MIN,
    }

needed_combos = set()
for _, row in sa_limit[sa_limit['ticker'].isin(ALL_TICKERS)].iterrows():
    needed_combos.add((row['ticker'], row['experiment']))
for tk, cfg in bq_configs.items():
    if cfg is not None:
        needed_combos.add((tk, cfg['experiment']))

all_bq_experiments = sorted(set(exp for _, exp in needed_combos))
ACTIVE_EXPS = all_bq_experiments
print(f"    {len(all_bq_experiments)} experiments: {', '.join(all_bq_experiments)}")

for ticker in ALL_TICKERS:
    cfg = bq_configs.get(ticker)
    if cfg is None:
        print(f"    {ticker}: no qualifying config")
        continue
    s3_flag = "S3" if ticker in s3_tickers else "  "
    print(f"    {s3_flag} {ticker}: prob>{cfg['prob_thresh']:.2f} TP={cfg['tp']:.0%} "
          f"SL={cfg['sl']:.0%} dip={cfg['dip_pct']:.1%} wait={cfg['max_wait_minutes']:.0f}m "
          f"avgPnL={cfg['avg_pnl']:.2%}")

all_tickers = [t for t in ALL_TICKERS if bq_configs.get(t) is not None]
print(f"    {len(all_tickers)} tickers with configs")

del raw_df, indiv, sa, sa_limit, sa_fam, best_per_fam
_release_memory()


# ── Step 2: Load OHLCV + Live TF scores ──
print(f"\n  STEP 2a: Loading OHLCV data (from {DATA_START})...")

ohlcv_arrays = {}
for ticker in all_tickers:
    df = load_master(ticker)
    if df is None:
        print(f"    {ticker}: FAILED to load")
        continue
    df = heal_data(df.sort_index())
    if str(df.index.tz) != 'America/Los_Angeles':
        df.index = df.index.tz_convert('America/Los_Angeles')
    ohlcv = precompute_ohlcv_arrays(df, DATA_START, TEST_END)
    if ohlcv is not None:
        ohlcv_arrays[ticker] = ohlcv
    n_days = len(np.unique(ohlcv['dates'])) if ohlcv is not None else 0
    print(f"    {ticker}: {len(df):>8,} bars, {n_days} total days")
    del df; gc.collect()

# Clear master cache — TF scoring reloads what it needs via get_featured_data
_master_cache.clear()
_release_memory()
print(f"    {len(ohlcv_arrays)} tickers loaded, master cache cleared to save memory")


# ── Step 2b: Live TF scoring ──
print(f"\n  STEP 2b: Live TF scoring (replacing CSV scores)...")

market_df = build_market_context()

# Determine which base experiments we need to score
# Strip _LIMIT suffix FIRST, then do walkforward fold lookup
exp_to_base = {}
for bq_exp in all_bq_experiments:
    clean_exp = bq_exp[:-len('_LIMIT')] if bq_exp.endswith('_LIMIT') else bq_exp
    base = clean_exp
    for wf_base, wf_folds in WALKFORWARD_FOLDS.items():
        if clean_exp in wf_folds:
            base = wf_base
            break
    exp_to_base[bq_exp] = base

unique_bases = sorted(set(exp_to_base.values()))
print(f"    Need to score {len(unique_bases)} base experiments: {', '.join(unique_bases)}")

# Determine unique entry minutes per ticker
ticker_entry_minutes = {}
for tk in all_tickers:
    cfg = bq_configs.get(tk)
    if cfg is not None:
        ticker_entry_minutes[tk] = cfg.get('entry_minute', ENTRY_MIN)
    else:
        ticker_entry_minutes[tk] = ENTRY_MIN

# Load scaling stats for all base experiments
for base_exp in unique_bases:
    ecfg = EXP_CONFIGS.get(base_exp)
    if ecfg is None:
        print(f"    WARNING: No config for {base_exp}, skipping")
        continue
    try:
        load_scaling_stats(ecfg['stats_id'])
    except Exception as e:
        print(f"    WARNING: Failed to load stats for {base_exp}: {e}")

# Score each (ticker, base_experiment) combination
daily_probs = {}
n_scored = 0
n_total = 0

for base_exp in unique_bases:
    ecfg = EXP_CONFIGS.get(base_exp)
    if ecfg is None:
        continue

    stats_id = ecfg['stats_id']
    if stats_id not in _stats_cache:
        continue
    stats = _stats_cache[stats_id]

    tickers_for_base = set()
    for bq_exp, mapped_base in exp_to_base.items():
        if mapped_base == base_exp:
            for tk in all_tickers:
                if (tk, bq_exp) in needed_combos:
                    tickers_for_base.add(tk)

    n_base_scored = 0
    for ticker in sorted(tickers_for_base):
        n_total += 1
        entry_min = ticker_entry_minutes.get(ticker, ENTRY_MIN)

        df_feat = get_featured_data(ticker, market_df)
        if df_feat is None:
            continue

        df_scaled = apply_scaling_scoring(df_feat, stats)

        probs_dict = score_ticker_for_experiment(ticker, entry_min, base_exp, df_scaled)

        if probs_dict:
            # Map to all fold experiments that share this base
            for bq_exp, mapped_base in exp_to_base.items():
                if mapped_base == base_exp and (ticker, bq_exp) in needed_combos:
                    daily_probs[(ticker, bq_exp)] = probs_dict
            n_scored += 1
            n_base_scored += 1

        del df_scaled; gc.collect()

    # Clear model+feature caches between experiments to cap memory
    _model_cache.clear()
    _feat_cache.clear()
    tf.keras.backend.clear_session()
    print(f"    {base_exp}: scored {n_base_scored}/{len(tickers_for_base)} tickers")
    _release_memory()

print(f"\n    Total: {n_scored}/{n_total} scored, {len(daily_probs)} daily_probs entries")

# Clear feature cache to save memory
_feat_cache.clear()
_model_cache.clear()
_release_memory()


# ── Step 3: Pre-simulate trades for FULL TP/SL grid (limit_entry) ──
print(f"\n  STEP 3: Pre-simulating trades for TP/SL grid (limit_entry)...")

tpsl_combos = set()
for tp in TP_GRID:
    for sl in SL_GRID:
        if sl <= tp:
            tpsl_combos.add((tp, sl))
grid_count = len(tpsl_combos)
bq_extra = 0
for cfg in bq_configs.values():
    if cfg is None:
        continue
    pair = (cfg['tp'], cfg['sl'])
    if pair not in tpsl_combos:
        tpsl_combos.add(pair)
        bq_extra += 1
tpsl_combos = sorted(tpsl_combos)
print(f"    {grid_count} grid combos + {bq_extra} BQ-specific = {len(tpsl_combos)} total TP/SL combos")

data_start_dt = pd.Timestamp(DATA_START).date()
test_start_dt = pd.Timestamp(TEST_START).date()
test_end_dt = pd.Timestamp(TEST_END).date()

fills = {}
daily_closes = {}

for ticker in all_tickers:
    cfg = bq_configs[ticker]
    ohlcv = ohlcv_arrays.get(ticker)
    if ohlcv is None:
        continue
    dip_pct = cfg['dip_pct']
    max_wait = cfg['max_wait_minutes']
    entry_min = cfg.get('entry_minute', ENTRY_MIN)
    unique_dates = np.unique(ohlcv['dates'])

    closes = {}
    n_fills = 0
    for d_int in unique_dates:
        d_str = str(int(d_int))
        d_date = pd.Timestamp(f"{d_str[:4]}-{d_str[4:6]}-{d_str[6:8]}").date()
        if d_date < data_start_dt or d_date > test_end_dt:
            continue

        mask_day = ohlcv['dates'] == d_int
        day_prices = ohlcv['prices'][mask_day]
        if len(day_prices) > 0:
            closes[d_date] = float(day_prices[-1])

        mask = mask_day & (ohlcv['times'] >= entry_min)
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            continue
        entry_idx = idxs[0]
        if ohlcv['prices'][entry_idx] <= 0:
            continue

        fill_idx, fill_price = find_limit_fill(
            ohlcv['prices'], ohlcv['lows'], ohlcv['dates'], ohlcv['times'],
            entry_idx, dip_pct, max_wait)
        if fill_idx >= 0:
            fills[(ticker, d_date)] = (fill_idx, fill_price)
            n_fills += 1

    daily_closes[ticker] = closes
    print(f"    {ticker}: {n_fills} fills / {len(unique_dates)} days "
          f"(dip={dip_pct:.1%} wait={max_wait:.0f}m)")

daily_trades = {}
all_fill_dates = set()

for ticker in all_tickers:
    ohlcv = ohlcv_arrays.get(ticker)
    if ohlcv is None:
        continue
    for tp, sl in tpsl_combos:
        outcomes = {}
        for d_date in sorted(daily_closes.get(ticker, {}).keys()):
            key = (ticker, d_date)
            if key not in fills:
                continue
            fill_idx, fill_price = fills[key]
            pnl = compute_exit_pnl(
                ohlcv['prices'], ohlcv['highs'], ohlcv['lows'],
                ohlcv['dates'], ohlcv['times'],
                fill_idx, fill_price, tp, sl, EXIT_TIME_MIN)
            outcomes[d_date] = pnl - SLIPPAGE_PCT
        daily_trades[(ticker, tp, sl)] = outcomes
        all_fill_dates.update(outcomes.keys())

all_available_dates = sorted(all_fill_dates)
all_trading_dates = sorted([d for d in all_available_dates
                            if d >= test_start_dt and d <= test_end_dt])

static_opt_start = pd.Timestamp(DATA_START).date()
static_opt_end = (pd.Timestamp(TEST_START) - timedelta(days=1)).date()
static_opt_dates = [d for d in all_available_dates
                    if d >= static_opt_start and d <= static_opt_end]

print(f"\n    {len(daily_trades)} trade series (ticker × TP × SL)")
print(f"    {len(all_trading_dates)} test days, {len(all_available_dates)} total days, "
      f"{len(static_opt_dates)} pre-test days")

del ohlcv_arrays, fills
_release_memory()


# ==============================================================================
# STEP 4: GRID + DECAY + PRECISION WEIGHTS
# ==============================================================================
reopt_indices = list(range(0, len(all_trading_dates), REOPT_FREQ))

CONFIG_LIST = []
for tp in TP_GRID:
    for sl in SL_GRID:
        if sl > tp:
            continue
        for prob_thresh in PROB_GRID:
            CONFIG_LIST.append((ENTRY_TIME, tp, sl, prob_thresh))

print(f"\n  STEP 4: Grid: {len(CONFIG_LIST)} configs "
      f"(1 entry × {len(TP_GRID)} TP × SL × {len(PROB_GRID)} prob)")

print(f"  Pre-computing DECAY configs...")

decay_configs_by_hl = {}
for half_life in [30, 60]:
    decay_rate = LN2 / half_life
    decay_configs = {}
    for ri in reopt_indices:
        reopt_date = all_trading_dates[ri]
        opt_end = reopt_date - timedelta(days=1)
        opt_dates = [d for d in all_available_dates
                     if d >= static_opt_start and d <= opt_end]
        if len(opt_dates) < 10:
            continue
        weights = {d: math.exp(-decay_rate * (reopt_date - d).days) for d in opt_dates}
        week_configs = {}
        for ticker in all_tickers:
            week_configs[ticker] = optimize_configs(
                opt_dates, ticker, CONFIG_LIST,
                daily_probs, daily_trades, ACTIVE_EXPS, weights)
        decay_configs[reopt_date] = week_configs
    decay_configs_by_hl[half_life] = decay_configs
    print(f"    DECAY hl={half_life}d: {len(decay_configs)} windows pre-computed")

# Precision weights for weighted vote composite
print(f"  Computing precision weights for weighted vote...")
exp_precision = {}
for ticker in all_tickers:
    cfg = bq_configs.get(ticker)
    if cfg is None:
        continue
    for exp in ACTIVE_EXPS:
        pk = (ticker, exp)
        if pk not in daily_probs:
            continue
        probs = daily_probs[pk]
        bq_tp, bq_sl = cfg['tp'], cfg['sl']
        trade_key = (ticker, bq_tp, bq_sl)
        outcomes = daily_trades.get(trade_key, {})
        n_sig, n_prof = 0, 0
        for d in all_trading_dates:
            if d in probs and probs[d] >= cfg['prob_thresh'] and d in outcomes:
                n_sig += 1
                if outcomes[d] > 0:
                    n_prof += 1
        exp_precision[(ticker, exp)] = n_prof / n_sig if n_sig >= 3 else 0.5
print(f"    Computed precision for {len(exp_precision)} (ticker, exp) pairs")

_release_memory()


# ==============================================================================
# STEP 5: RUN 11 STRATEGIES × 3 COMPOSITES
# ==============================================================================
COMPOSITE_METHODS = ['S3 Deployed', 'Weighted>50%', 'Adapt Wt>50%']

print(f"\n  STEP 5: Running 11 strategies × {len(COMPOSITE_METHODS)} composites = 33 combos...")

all_results = {}

for comp_method in COMPOSITE_METHODS:
    print(f"\n  ── Composite: {comp_method} ──")

    is_adaptive = (comp_method == 'Adapt Wt>50%')

    def _apply(cfg, ticker, dates, adaptive_state=None):
        if cfg is None:
            return []
        if is_adaptive and adaptive_state is not None:
            state = adaptive_state[ticker]
            if state['thresh'] is None:
                state['thresh'] = cfg['prob_thresh']
            if len(state['results']) >= 3:
                last10 = state['results'][-10:]
                precision = sum(1 for r in last10 if r > 0) / len(last10)
                if precision < 0.40:
                    state['thresh'] = min(0.90, state['thresh'] + 0.10)
                elif precision > 0.60:
                    state['thresh'] = max(0.30, state['thresh'] - 0.10)
            adj_cfg = dict(cfg)
            adj_cfg['prob_thresh'] = state['thresh']
            wt = apply_composite(adj_cfg, ticker, dates, comp_method,
                                 daily_probs, daily_trades, ACTIVE_EXPS,
                                 bq_configs, s3_tickers, exp_precision)
            state['results'].extend([t['pnl'] for t in wt])
            return wt
        return apply_composite(cfg, ticker, dates, comp_method,
                               daily_probs, daily_trades, ACTIVE_EXPS,
                               bq_configs, s3_tickers, exp_precision)

    def _new_adaptive_state():
        return defaultdict(lambda: {'thresh': None, 'results': []})

    # 1. STATIC (BQ)
    adapt_st = _new_adaptive_state()
    static_trades = []
    for ri in reopt_indices:
        test_end_idx = min(ri + REOPT_FREQ, len(all_trading_dates))
        test_dates = all_trading_dates[ri:test_end_idx]
        for ticker in all_tickers:
            static_trades.extend(_apply(bq_configs[ticker], ticker, test_dates, adapt_st))
    all_results[('STATIC (BQ)', comp_method)] = static_trades
    print(f"    [1/11] STATIC (BQ): {len(static_trades)} trades")

    static_by_ticker = defaultdict(list)
    for t in static_trades:
        static_by_ticker[t['ticker']].append(t)

    # 2. GATE lb=7 (BQ)
    adapt_st = _new_adaptive_state()
    gate_trades = []
    for ri in reopt_indices:
        reopt_date = all_trading_dates[ri]
        test_end_idx = min(ri + REOPT_FREQ, len(all_trading_dates))
        test_dates = all_trading_dates[ri:test_end_idx]
        for ticker in all_tickers:
            if bq_configs[ticker] is None:
                continue
            prior = [t for t in static_by_ticker[ticker] if t['date'] < reopt_date]
            recent = prior[-7:]
            if len(recent) >= 2:
                recent_avg = np.mean([t['pnl'] for t in recent])
                if recent_avg < 0:
                    continue
            gate_trades.extend(_apply(bq_configs[ticker], ticker, test_dates, adapt_st))
    all_results[('GATE lb=7 (BQ)', comp_method)] = gate_trades
    print(f"    [2/11] GATE lb=7 (BQ): {len(gate_trades)} trades")

    # 3. DECAY hl=30d
    adapt_st = _new_adaptive_state()
    decay30_trades = []
    for ri in reopt_indices:
        reopt_date = all_trading_dates[ri]
        test_end_idx = min(ri + REOPT_FREQ, len(all_trading_dates))
        test_dates = all_trading_dates[ri:test_end_idx]
        week_configs = decay_configs_by_hl[30].get(reopt_date, {})
        for ticker in all_tickers:
            cfg = week_configs.get(ticker)
            decay30_trades.extend(_apply(cfg, ticker, test_dates, adapt_st))
    all_results[('DECAY hl=30d', comp_method)] = decay30_trades
    print(f"    [3/11] DECAY hl=30d: {len(decay30_trades)} trades")

    # 4. DECAY hl=60d
    adapt_st = _new_adaptive_state()
    decay60_trades = []
    for ri in reopt_indices:
        reopt_date = all_trading_dates[ri]
        test_end_idx = min(ri + REOPT_FREQ, len(all_trading_dates))
        test_dates = all_trading_dates[ri:test_end_idx]
        week_configs = decay_configs_by_hl[60].get(reopt_date, {})
        for ticker in all_tickers:
            cfg = week_configs.get(ticker)
            decay60_trades.extend(_apply(cfg, ticker, test_dates, adapt_st))
    all_results[('DECAY hl=60d', comp_method)] = decay60_trades
    print(f"    [4/11] DECAY hl=60d: {len(decay60_trades)} trades")

    # 5-7. HYBRID Gate+Decay
    for hl, glb, label_suffix in [(30, 10, '30d+gate10'), (60, 10, '60d+gate10'), (30, 21, '30d+gate21')]:
        adapt_st = _new_adaptive_state()
        hybrid_trades = []
        accumulated_trades = defaultdict(list)
        for ri in reopt_indices:
            reopt_date = all_trading_dates[ri]
            test_end_idx = min(ri + REOPT_FREQ, len(all_trading_dates))
            test_dates = all_trading_dates[ri:test_end_idx]
            week_configs = decay_configs_by_hl[hl].get(reopt_date, {})
            for ticker in all_tickers:
                cfg = week_configs.get(ticker)
                if cfg is None:
                    continue
                prior = accumulated_trades[ticker]
                recent = [t for t in prior
                          if t['date'] >= (reopt_date - timedelta(days=int(glb * 1.5)))]
                recent = recent[-glb:]
                if len(recent) >= 2:
                    if np.mean([t['pnl'] for t in recent]) < 0:
                        continue
                wt = _apply(cfg, ticker, test_dates, adapt_st)
                hybrid_trades.extend(wt)
                accumulated_trades[ticker].extend(wt)
        idx = 5 + [(30, 10), (60, 10), (30, 21)].index((hl, glb))
        all_results[(f'HYBRID {label_suffix}', comp_method)] = hybrid_trades
        print(f"    [{idx}/11] HYBRID {label_suffix}: {len(hybrid_trades)} trades")

    # 8. ADAPTIVE THRESH (BQ)
    adapt_st = _new_adaptive_state()
    adaptive_trades = []
    for ticker in all_tickers:
        if bq_configs[ticker] is None:
            continue
        base_thresh = bq_configs[ticker]['prob_thresh']
        current_thresh = base_thresh
        recent_results = []
        for ri in reopt_indices:
            test_end_idx = min(ri + REOPT_FREQ, len(all_trading_dates))
            test_dates = all_trading_dates[ri:test_end_idx]
            if len(recent_results) >= 3:
                last10 = recent_results[-10:]
                precision = sum(1 for r in last10 if r > 0) / len(last10)
                if precision < 0.40:
                    current_thresh = min(0.90, current_thresh + 0.10)
                elif precision > 0.60:
                    current_thresh = max(0.30, current_thresh - 0.10)
            adj_config = dict(bq_configs[ticker])
            adj_config['prob_thresh'] = current_thresh
            wt = _apply(adj_config, ticker, test_dates, adapt_st)
            adaptive_trades.extend(wt)
            recent_results.extend([t['pnl'] for t in wt])
    all_results[('ADAPTIVE THRESH', comp_method)] = adaptive_trades
    print(f"    [8/11] ADAPTIVE THRESH: {len(adaptive_trades)} trades")

    # 9. MOMENTUM 10d (BQ)
    adapt_st = _new_adaptive_state()
    momentum_trades = []
    for ticker in all_tickers:
        if bq_configs[ticker] is None:
            continue
        closes = daily_closes.get(ticker, {})
        for ri in reopt_indices:
            reopt_date = all_trading_dates[ri]
            test_end_idx = min(ri + REOPT_FREQ, len(all_trading_dates))
            test_dates = all_trading_dates[ri:test_end_idx]
            prior_dates = sorted([d for d in closes if d < reopt_date])
            if len(prior_dates) >= 10:
                close_now = closes[prior_dates[-1]]
                close_10d = closes[prior_dates[-10]]
                if close_10d > 0 and (close_now - close_10d) / close_10d <= 0:
                    continue
            momentum_trades.extend(_apply(bq_configs[ticker], ticker, test_dates, adapt_st))
    all_results[('MOMENTUM 10d', comp_method)] = momentum_trades
    print(f"    [9/11] MOMENTUM 10d: {len(momentum_trades)} trades")

    # 10. SELECTIVE DECAY
    adapt_st = _new_adaptive_state()
    selective_trades = []
    for ticker in all_tickers:
        if bq_configs[ticker] is None:
            continue
        accumulated = []
        for ri in reopt_indices:
            reopt_date = all_trading_dates[ri]
            test_end_idx = min(ri + REOPT_FREQ, len(all_trading_dates))
            test_dates = all_trading_dates[ri:test_end_idx]
            recent = [t for t in accumulated
                      if t['date'] >= (reopt_date - timedelta(days=15))][-10:]
            use_decay = len(recent) >= 2 and np.mean([t['pnl'] for t in recent]) < 0
            if use_decay and reopt_date in decay_configs_by_hl[30]:
                cfg = decay_configs_by_hl[30][reopt_date].get(ticker, bq_configs[ticker])
            else:
                cfg = bq_configs[ticker]
            wt = _apply(cfg, ticker, test_dates, adapt_st)
            selective_trades.extend(wt)
            accumulated.extend(wt)
    all_results[('SELECTIVE DECAY', comp_method)] = selective_trades
    print(f"    [10/11] SELECTIVE DECAY: {len(selective_trades)} trades")

    # 11. VOL GATE (BQ)
    adapt_st = _new_adaptive_state()
    vol_gate_trades = []
    for ticker in all_tickers:
        if bq_configs[ticker] is None:
            continue
        closes = daily_closes.get(ticker, {})
        for ri in reopt_indices:
            reopt_date = all_trading_dates[ri]
            test_end_idx = min(ri + REOPT_FREQ, len(all_trading_dates))
            test_dates = all_trading_dates[ri:test_end_idx]
            prior_dates = sorted([d for d in closes if d < reopt_date])
            if len(prior_dates) >= 20:
                prices = [closes[d] for d in prior_dates[-20:]]
                rets = [prices[i] / prices[i - 1] - 1 for i in range(1, len(prices))]
                vol_20d = np.std(rets) if len(rets) >= 10 else 0
                vol_5d = np.std(rets[-5:]) if len(rets) >= 5 else 0
                if vol_20d > 0 and vol_5d / vol_20d > 2.0:
                    continue
            vol_gate_trades.extend(_apply(bq_configs[ticker], ticker, test_dates, adapt_st))
    all_results[('VOL GATE', comp_method)] = vol_gate_trades
    print(f"    [11/11] VOL GATE: {len(vol_gate_trades)} trades")

_release_memory()


# ==============================================================================
# STEP 6: RESULTS TABLES
# ==============================================================================
STRATEGY_ORDER = [
    'STATIC (BQ)', 'GATE lb=7 (BQ)', 'DECAY hl=30d', 'DECAY hl=60d',
    'HYBRID 30d+gate10', 'HYBRID 60d+gate10', 'HYBRID 30d+gate21',
    'ADAPTIVE THRESH', 'MOMENTUM 10d', 'SELECTIVE DECAY', 'VOL GATE',
]

stats = {}
for (strat, comp), trades in all_results.items():
    stats[(strat, comp)] = summarize(trades, f"{strat} ({comp})")

print()
print("=" * W)
print(f"  STEP 6: RESULTS")
print("=" * W)


# ── TABLE A: Strategy × Composite matrix ──
col_w = 28
print(f"\n  TABLE A: STRATEGY × COMPOSITE MATRIX (Full Period)")
print(f"\n  {'Strategy':<22} │ {'S3 Deployed':^{col_w}} │ {'Weighted>50%':^{col_w}} │ {'Adapt Wt>50%':^{col_w}}")
print(f"  {'':<22} │ {'Trd':>4} {'Avg':>7} {'Tot':>8} {'WR':>4} {'DD':>7}"
      f" │ {'Trd':>4} {'Avg':>7} {'Tot':>8} {'WR':>4} {'DD':>7}"
      f" │ {'Trd':>4} {'Avg':>7} {'Tot':>8} {'WR':>4} {'DD':>7}")
print(f"  {'─'*22} │ {'─'*col_w} │ {'─'*col_w} │ {'─'*col_w}")

best_avg = -999
best_key = None
for strat in STRATEGY_ORDER:
    print(f"  {strat:<22}", end="")
    for ci, comp in enumerate(COMPOSITE_METHODS):
        st = stats.get((strat, comp), {'n': 0})
        if st['n'] == 0:
            print(f" │ {'—':>{col_w}}", end="")
        else:
            print(f" │ {st['n']:>4} {st['avg']:>+6.2%} {st['total']:>+7.1%} "
                  f"{st['wr']:>3.0%} {st['dd']:>+6.1%}", end="")
            if st['avg'] > best_avg:
                best_avg = st['avg']
                best_key = (strat, comp)
    print()

if best_key:
    print(f"\n  ★ Best avg PnL: {best_key[0]} ({best_key[1]}) = {best_avg:+.2%}")


# ── TABLE B: June-only matrix ──
jcol_w = 28
print(f"\n  TABLE B: JUNE-ONLY MATRIX")
print(f"\n  {'Strategy':<22} │ {'S3 Deployed':^{jcol_w}} │ {'Weighted>50%':^{jcol_w}} │ {'Adapt Wt>50%':^{jcol_w}}")
print(f"  {'':<22} │ {'Trd':>4} {'Avg':>7} {'Tot':>8} {'WR':>4}"
      f" │ {'Trd':>4} {'Avg':>7} {'Tot':>8} {'WR':>4}"
      f" │ {'Trd':>4} {'Avg':>7} {'Tot':>8} {'WR':>4}")
print(f"  {'─'*22} │ {'─'*jcol_w} │ {'─'*jcol_w} │ {'─'*jcol_w}")

best_june = -999
best_june_key = None
for strat in STRATEGY_ORDER:
    print(f"  {strat:<22}", end="")
    for comp in COMPOSITE_METHODS:
        st = stats.get((strat, comp), {'j_n': 0, 'j_avg': 0, 'j_total': 0, 'j_wr': 0})
        if st['j_n'] == 0:
            print(f" │ {'—':>{jcol_w}}", end="")
        else:
            print(f" │ {st['j_n']:>4} {st['j_avg']:>+6.2%} {st['j_total']:>+7.1%} {st['j_wr']:>3.0%}", end="")
            if st['j_avg'] > best_june:
                best_june = st['j_avg']
                best_june_key = (strat, comp)
    print()

if best_june_key:
    print(f"\n  ★ Best June avg: {best_june_key[0]} ({best_june_key[1]}) = {best_june:+.2%}")


# ── TABLE B2: July-only matrix ──
print(f"\n  TABLE B2: JULY-ONLY MATRIX (1st–19th)")
print(f"\n  {'Strategy':<22} │ {'S3 Deployed':^{jcol_w}} │ {'Weighted>50%':^{jcol_w}} │ {'Adapt Wt>50%':^{jcol_w}}")
print(f"  {'':<22} │ {'Trd':>4} {'Avg':>7} {'Tot':>8} {'WR':>4}"
      f" │ {'Trd':>4} {'Avg':>7} {'Tot':>8} {'WR':>4}"
      f" │ {'Trd':>4} {'Avg':>7} {'Tot':>8} {'WR':>4}")
print(f"  {'─'*22} │ {'─'*jcol_w} │ {'─'*jcol_w} │ {'─'*jcol_w}")

best_july = -999
best_july_key = None
for strat in STRATEGY_ORDER:
    print(f"  {strat:<22}", end="")
    for comp in COMPOSITE_METHODS:
        st = stats.get((strat, comp), {'jul_n': 0, 'jul_avg': 0, 'jul_total': 0, 'jul_wr': 0})
        if st['jul_n'] == 0:
            print(f" │ {'—':>{jcol_w}}", end="")
        else:
            print(f" │ {st['jul_n']:>4} {st['jul_avg']:>+6.2%} {st['jul_total']:>+7.1%} {st['jul_wr']:>3.0%}", end="")
            if st['jul_avg'] > best_july:
                best_july = st['jul_avg']
                best_july_key = (strat, comp)
    print()

if best_july_key:
    print(f"\n  ★ Best July avg: {best_july_key[0]} ({best_july_key[1]}) = {best_july:+.2%}")


# ── TABLE C: Per-ticker breakdown for top 5 combos ──
ranked = sorted(
    [(k, s) for k, s in stats.items() if s['n'] > 0],
    key=lambda x: x[1]['avg'], reverse=True)[:5]

if ranked:
    print(f"\n  TABLE C: PER-TICKER BREAKDOWN — Top 5 combos")
    for (strat, comp), st in ranked:
        label = f"{strat} ({comp})"
        print(f"\n  ── {label}: {st['n']} trades, avg={st['avg']:+.2%}, "
              f"total={st['total']:+.1%}, WR={st['wr']:.0%} ──")
        if st['trades_df'].empty:
            continue
        df = st['trades_df']
        df['month'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m')
        june = df[df['month'] == '2026-06']
        july = df[df['month'] == '2026-07']
        print(f"  {'Ticker':<7} {'Trades':>6} {'AvgPnL':>8} {'TotPnL':>9} {'WR':>6}"
              f" │ {'JunT':>5} {'JunAvg':>8}"
              f" │ {'JulT':>5} {'JulAvg':>8}")
        print(f"  {'─'*7} {'─'*6} {'─'*8} {'─'*9} {'─'*6}"
              f" │ {'─'*5} {'─'*8}"
              f" │ {'─'*5} {'─'*8}")
        for ticker in sorted(df['ticker'].unique()):
            tt = df[df['ticker'] == ticker]
            jt = june[june['ticker'] == ticker]
            jlt = july[july['ticker'] == ticker]
            jun_s = f"{jt['pnl'].mean():>+7.2%}" if len(jt) > 0 else f"{'—':>8}"
            jul_s = f"{jlt['pnl'].mean():>+7.2%}" if len(jlt) > 0 else f"{'—':>8}"
            print(f"  {ticker:<7} {len(tt):>6} {tt['pnl'].mean():>+7.2%} "
                  f"{tt['pnl'].sum():>+8.1%} {(tt['pnl']>0).mean():>5.0%}"
                  f" │ {len(jt):>5} {jun_s}"
                  f" │ {len(jlt):>5} {jul_s}")


# ── TABLE D: Monthly breakdown for top 5 ──
months = ['2026-04', '2026-05', '2026-06', '2026-07']
mlabels = ['Apr', 'May', 'Jun', 'Jul']

if ranked:
    print(f"\n  TABLE D: MONTHLY BREAKDOWN — Top 5 combos")
    print(f"\n  {'Strategy':<36}", end="")
    for ml in mlabels:
        print(f"  {ml:>20}", end="")
    print()
    print(f"  {'─'*36}", end="")
    for _ in mlabels:
        print(f"  {'─'*20}", end="")
    print()

    for (strat, comp), st in ranked:
        label = f"{strat} ({comp})"[:36]
        print(f"  {label:<36}", end="")
        for m in months:
            ms = st.get('monthly', {}).get(m)
            if ms:
                print(f"  {ms['avg']:>+5.2%}({ms['n']:>3}t,{ms['wr']:>4.0%}wr)", end="")
            else:
                print(f"  {'—':>20}", end="")
        print()


# ==============================================================================
# STEP 7: CHARTS
# ==============================================================================
print(f"\n  STEP 7: Generating charts...")

# ── CHART 1: Cumulative PnL — Top 5 combos only ──
fig1, ax1 = plt.subplots(figsize=(14, 7))

line_styles = ['-', '-', '-', '--', '--']
line_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

for i, ((strat, comp), st) in enumerate(ranked[:5]):
    if st['trades_df'].empty:
        continue
    df = st['trades_df'].sort_values('date')
    cum = df['pnl'].cumsum() * 100
    lbl = f"{strat} ({comp})"
    if len(lbl) > 35:
        lbl = lbl[:32] + "..."
    ax1.plot(pd.to_datetime(df['date']), cum.values,
             label=f"{lbl}  [{st['n']}t, {st['avg']:+.2%}]",
             color=line_colors[i], linewidth=2 if i < 3 else 1.5,
             linestyle=line_styles[i], alpha=0.9)

ax1.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
for mo_start, mo_label in [('2026-06-01', 'Jun'), ('2026-07-01', 'Jul')]:
    ax1.axvline(x=pd.Timestamp(mo_start), color='red', linewidth=0.8,
                linestyle=':', alpha=0.5)
    ax1.text(pd.Timestamp(mo_start), ax1.get_ylim()[1] * 0.95 if ax1.get_ylim()[1] > 0 else 0,
             f' {mo_label}', fontsize=9, color='red', alpha=0.7)
ax1.set_title('Cumulative PnL — Top 5 Strategy Combos (Live TF)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Cumulative PnL (%)')
ax1.set_xlabel('Date')
ax1.legend(fontsize=9, loc='best')
ax1.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/optimization_live_cumulative_top5.png', dpi=150, bbox_inches='tight')
print(f"    Chart 1 saved: /tmp/optimization_live_cumulative_top5.png")
plt.show()

# ── CHART 2: Monthly avg PnL bar chart — Top 5 combos, Apr-Jul ──
if ranked:
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    n_combos = min(5, len(ranked))
    bar_width = 0.15
    x = np.arange(len(months))

    for i, ((strat, comp), st) in enumerate(ranked[:n_combos]):
        monthly_avgs = []
        for m in months:
            ms = st.get('monthly', {}).get(m)
            monthly_avgs.append(ms['avg'] * 100 if ms else 0)
        label = f"{strat} ({comp})"
        if len(label) > 30:
            label = label[:27] + "..."
        ax2.bar(x + i * bar_width, monthly_avgs, bar_width,
                label=label, color=line_colors[i], alpha=0.85)

    ax2.axhline(y=DEPLOY_THRESHOLD * 100, color='green', linewidth=1,
                linestyle='--', alpha=0.6, label=f'Deploy threshold ({DEPLOY_THRESHOLD:.1%})')
    ax2.axhline(y=0, color='gray', linewidth=0.5)
    ax2.set_title('Monthly Avg PnL — Top 5 Combos (Live TF)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Avg PnL (%)')
    ax2.set_xticks(x + bar_width * (n_combos - 1) / 2)
    ax2.set_xticklabels(mlabels, fontsize=12)
    ax2.legend(fontsize=8, loc='best')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('/tmp/optimization_live_monthly_comparison.png', dpi=150, bbox_inches='tight')
    print(f"    Chart 2 saved: /tmp/optimization_live_monthly_comparison.png")
    plt.show()


# ── DIAGNOSIS ──
print()
print("=" * W)
print(f"  DIAGNOSIS")
print("=" * W)

if ranked:
    best = ranked[0]
    print(f"\n  BEST OVERALL: {best[0][0]} ({best[0][1]})")
    print(f"    avg={best[1]['avg']:+.2%}, total={best[1]['total']:+.1%}, "
          f"WR={best[1]['wr']:.0%}, MaxDD={best[1]['dd']:+.1%}")

    best_risk = min(
        [(k, s) for k, s in stats.items() if s['n'] > 0],
        key=lambda x: abs(x[1]['dd']))
    print(f"\n  BEST RISK-ADJUSTED: {best_risk[0][0]} ({best_risk[0][1]})")
    print(f"    MaxDD={best_risk[1]['dd']:+.1%}, avg={best_risk[1]['avg']:+.2%}")

    best_j = max(
        [(k, s) for k, s in stats.items() if s['j_n'] > 0],
        key=lambda x: x[1]['j_avg'], default=None)
    if best_j:
        print(f"\n  BEST JUNE: {best_j[0][0]} ({best_j[0][1]})")
        print(f"    June avg={best_j[1]['j_avg']:+.2%}, {best_j[1]['j_n']} trades")

    best_jul = max(
        [(k, s) for k, s in stats.items() if s['jul_n'] > 0],
        key=lambda x: x[1]['jul_avg'], default=None)
    if best_jul:
        print(f"\n  BEST JULY: {best_jul[0][0]} ({best_jul[0][1]})")
        print(f"    July avg={best_jul[1]['jul_avg']:+.2%}, {best_jul[1]['jul_n']} trades")

    print(f"\n  BEST PER COMPOSITE:")
    for comp in COMPOSITE_METHODS:
        comp_ranked = sorted(
            [(k, s) for k, s in stats.items() if k[1] == comp and s['n'] > 0],
            key=lambda x: x[1]['avg'], reverse=True)
        if comp_ranked:
            (strat, _), st = comp_ranked[0]
            jul_s = f", July={st['jul_avg']:+.2%}" if st['jul_n'] > 0 else ""
            print(f"    {comp}: {strat} — avg={st['avg']:+.2%}, "
                  f"total={st['total']:+.1%}, {st['n']} trades, "
                  f"June={st['j_avg']:+.2%}{jul_s}")

print()
print("=" * W)
print(f"  Analysis complete. 11 strategies × 3 composites = 33 combos evaluated.")
print(f"  Scores from: LIVE TF INFERENCE (15-seed ensemble)")
print("=" * W)
