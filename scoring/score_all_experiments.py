"""
score_all_experiments.py — Universal scoring for all 5 experiments
=================================================================
Scores ALL tickers with master_inference using models from experiments
152, 160, 175, 185, and 200. Handles two model architectures:
  - Standard transformer (152, 160, 175, 185)
  - CNN-Transformer (200)

Saves results as CSVs to a dated GCS folder so existing scripts can consume them.
Staleness detection: skips (ticker, seed) pairs where CSV is newer than master file.

Colab-ready (GPU recommended). Run this BEFORE backtest_deep_optimization.py.
"""

import sys
import subprocess
import os
import io
import json
import gc
import ctypes
from datetime import datetime

import numpy as np
import pandas as pd
import pytz

SCRIPT_VERSION = "score_all_1.0"
print(f"Script version: {SCRIPT_VERSION}")

os.environ['TF_USE_LEGACY_KERAS'] = '1'

try:
    import tensorflow as tf
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "tensorflow[and-cuda]", "tf_keras", "pandas_ta",
                           "google-cloud-storage", "pyarrow", "gcsfs", "--quiet"])
    import tensorflow as tf

try:
    import tf_keras
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tf_keras", "--quiet"])
    import tf_keras

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

from google.cloud import storage

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass

client = storage.Client()
bucket = client.bucket(os.environ.get("GCS_BUCKET", ""))

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print(f"GPU ACTIVE: {len(gpus)} GPU(s)")
else:
    print("WARNING: No GPU detected — scoring will be slow")


# ==============================================================================
# CONFIG
# ==============================================================================
BUCKET_NAME = os.environ.get("GCS_BUCKET", "")
INFERENCE_DIR = "Data/Inference_v2"
GCS_MODEL_PATH = "Algos/to_deploy/Saved_models_v2/UNIVERSAL/Direction"

SCORE_DATE_TAG = "20260703"
GCS_RESULTS_BASE = f"Algos/Results_scored_{SCORE_DATE_TAG}"

SCORE_CUTOFF = '2026-04-01'

DATA_START_DATE = "2022-01-01"
SPIKE_THRESHOLD = 0.10

PST = pytz.timezone('America/Los_Angeles')
TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

CONTEXT_ASSETS = ['SPY', 'VIX', 'TLT', 'IWM']
CONTEXT_SKIP = {'SPY', 'VIX', 'TLT', 'IWM', 'SMH', 'BTCUSD', 'ITA', 'XLE'}
EXCLUDE_TICKERS = {'SMLR'}

MICRO_COLS = ['Spread', 'OBI', 'Volume_Delta', 'Trade_Count', 'Delta_Pct',
              'bsize', 'asize', 'Total_Vol']
ROC_HORIZONS = [5, 30, 60, 120]

ASSET_COLS = (['Gap_Pct', 'RSI_14', 'RSI_60', 'RVOL', 'Compression',
               'ATR_14', 'Dist_to_PM_High', 'Bar_Is_Real'] +
              [f'ROC_{h}m' for h in ROC_HORIZONS] + MICRO_COLS)

CONTEXT_COLS = []
for _t in CONTEXT_ASSETS:
    CONTEXT_COLS.append(f'{_t}_Anchor_ROC')
    for _h in ROC_HORIZONS:
        CONTEXT_COLS.append(f'{_t}_ROC_{_h}m')
CONTEXT_COLS += ['Time_Pos']
ALL_COLS = ASSET_COLS + CONTEXT_COLS

TIME_STEPS = 240
EMBED_DIM = 16
MICRO_CHECK_COL = 'Volume_Delta'
MICRO_MIN_PCT = 5.0

LOCAL_MODEL_DIR = "/tmp/score_all_models"
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)


# ==============================================================================
# PER-EXPERIMENT CONFIG
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

EXP_CONFIGS = {
    152: {
        'model_id': '152_XSECTOR',
        'train_teachers': TRAIN_TEACHERS,
        'n_assets': len(TRAIN_TEACHERS) + 1,
        'stats_id': '152_XSECTOR',
        'seed_base': 152,
        'model_builder': 'standard',
        'embed_dim': 16,
        'train_cutoff': '2025-10-01',
    },
    160: {
        'model_id': '160_XSECTOR',
        'train_teachers': TRAIN_TEACHERS,
        'n_assets': len(TRAIN_TEACHERS) + 1,
        'stats_id': '160_XSECTOR',
        'seed_base': 160,
        'model_builder': 'standard',
        'embed_dim': 16,
        'train_cutoff': '2025-03-01',
    },
    175: {
        'model_id': '175_XSECTOR',
        'train_teachers': TRAIN_TEACHERS,
        'n_assets': len(TRAIN_TEACHERS) + 1,
        'stats_id': '175_XSECTOR',
        'seed_base': 175,
        'model_builder': 'standard',
        'embed_dim': 16,
        'train_cutoff': '2026-01-01',
    },
    185: {
        'model_id': '185_XSECTOR',
        'train_teachers': TEACHERS_31,
        'n_assets': len(TEACHERS_31) + 1,
        'stats_id': '185_XSECTOR',
        'seed_base': 185,
        'model_builder': 'standard',
        'embed_dim': 16,
        'train_cutoff': '2026-01-01',
    },
    200: {
        'model_id': '200_XSECTOR',
        'train_teachers': TEACHERS_31,
        'n_assets': len(TEACHERS_31) + 1,
        'stats_id': '200_XSECTOR',
        'seed_base': 200,
        'model_builder': 'cnn_transformer',
        'embed_dim': 32,
        'train_cutoff': '2026-01-01',
    },
}

for _eid, _ecfg in EXP_CONFIGS.items():
    _idx = {t: i for i, t in enumerate(_ecfg['train_teachers'])}
    if 'BITF' in _idx: _idx['KEEL'] = _idx['BITF']
    if 'MIGI' in _idx: _idx['BGDE'] = _idx['MIGI']
    _ecfg['asset_index'] = _idx
    _ecfg['other_idx'] = len(_ecfg['train_teachers'])
    _ecfg['seeds'] = list(range(_ecfg['seed_base'] * 100 + 1,
                                _ecfg['seed_base'] * 100 + 16))

EXPERIMENTS_TO_SCORE = [152, 160, 175, 185, 200]


# ==============================================================================
# MODEL ARCHITECTURES
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
# DATA LOADING
# ==============================================================================
def load_master(ticker):
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
        return df
    except Exception as e:
        print(f"   Failed to load {ticker}: {e}")
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
        df = df.sort_index()
        df = heal_data(df)
        daily_c = df['Close'].resample('D').last().ffill().shift(1).reindex(
            df.index, method='ffill')
        cols = {f'{ticker}_Anchor_ROC': (df['Close'] - daily_c) / daily_c}
        for h in ROC_HORIZONS:
            cols[f'{ticker}_ROC_{h}m'] = df['Close'].pct_change(h).fillna(0)
        frames.append(pd.DataFrame(cols, index=df.index))
        del df, daily_c; gc.collect()
    if not frames:
        raise RuntimeError("No context assets loaded.")
    return pd.concat(frames, axis=1).ffill().fillna(0)


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
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


def apply_scaling(df, stats):
    df_s = df.copy()
    for col in ALL_COLS:
        if col in ('Time_Pos', 'Bar_Is_Real'):
            continue
        if col in stats:
            iqr = stats[col]['iqr']
            if iqr == 0:
                iqr = 1e-9
            df_s[col] = ((df_s[col] - stats[col]['med']) / iqr).clip(-10, 10).astype('float32')
    return df_s


def add_targets(df):
    df_pst = df.index.tz_convert(PST)
    mins = df_pst.hour * 60 + df_pst.minute
    is_safe = (df_pst.hour < 13) | ((df_pst.hour == 13) & (df_pst.minute == 0))
    is_real = df['Volume'] > 0
    df['Safe_High'] = np.where(is_safe & is_real, df['High'], -1.0)
    df['Future_Max'] = df.groupby(df.index.date)['Safe_High'].transform(
        lambda x: x.shift(-1).fillna(-1.0).iloc[::-1].cummax().iloc[::-1])
    df['Target'] = (df['Future_Max'] > (df['Close'] * (1 + SPIKE_THRESHOLD))).astype(int)
    df['Valid_Row'] = ((mins >= 360) & (mins <= 390) & (df['Volume'] > 0)).astype(int)
    keep_cols = ALL_COLS + ['Target', 'Valid_Row',
                            'Open', 'High', 'Low', 'Close', 'Volume']
    drop_cols = [c for c in df.columns if c not in keep_cols]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    for c in df.columns:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    return df


def engineer_asset(asset_name, market_df):
    df = load_master(asset_name)
    if df is None:
        return None
    df = df.sort_index()
    df = df.loc[pd.to_datetime(DATA_START_DATE).tz_localize('UTC'):]
    df = heal_data(df)
    df = engineer_features(df, market_df)
    df = add_targets(df)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


# ==============================================================================
# MODEL LOADING
# ==============================================================================
_model_cache = {}


def get_model(model_id, seed, n_assets, ecfg):
    key = (model_id, seed)
    if key in _model_cache:
        return _model_cache[key]
    local_path = os.path.join(LOCAL_MODEL_DIR, f"UNIVERSAL_S{seed}_M{model_id}.h5")
    blob_path = f"{GCS_MODEL_PATH}/UNIVERSAL_S{seed}_M{model_id}.h5"
    if not os.path.exists(local_path):
        blob = bucket.blob(blob_path)
        if not blob.exists():
            return None
        blob.download_to_filename(local_path)
    builder = ecfg.get('model_builder', 'standard')
    embed_dim = ecfg.get('embed_dim', EMBED_DIM)
    if builder == 'cnn_transformer':
        model = build_cnn_transformer_model(n_assets, embed_dim=embed_dim)
    else:
        model = build_universal_model(n_assets, embed_dim=embed_dim)
    try:
        model.load_weights(local_path)
    except OSError:
        os.remove(local_path)
        blob = bucket.blob(blob_path)
        if not blob.exists():
            return None
        blob.download_to_filename(local_path)
        model.load_weights(local_path)
    _model_cache[key] = model
    return model


def load_scaling_stats(stats_id):
    blob_path = f"{INFERENCE_DIR}/scaling_stats_experiment_{stats_id}.json"
    stats = json.loads(bucket.blob(blob_path).download_as_text())
    return stats


def clear_model_cache():
    global _model_cache
    _model_cache.clear()
    tf.keras.backend.clear_session()
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


# ==============================================================================
# STEP 1: DISCOVER TICKERS WITH MASTER FILES
# ==============================================================================
print(f"\n{'='*80}")
print(f"  UNIVERSAL SCORER — {len(EXPERIMENTS_TO_SCORE)} experiments")
print(f"  Output: gs://{BUCKET_NAME}/{GCS_RESULTS_BASE}/")
print(f"{'='*80}")

print(f"\n  Discovering tickers with master_inference files...")
SCORING_POOL = []
master_blobs = list(bucket.list_blobs(prefix='Data/Inference_v2/'))
for b in master_blobs:
    if b.name.endswith('_master_inference.parquet'):
        ticker = b.name.replace('Data/Inference_v2/', '').replace('_master_inference.parquet', '')
        if ticker and ticker not in CONTEXT_SKIP and ticker not in EXCLUDE_TICKERS:
            SCORING_POOL.append(ticker)
SCORING_POOL = sorted(set(SCORING_POOL))
print(f"  Found {len(SCORING_POOL)} tickers with master files")


# ==============================================================================
# STEP 2: MICRO QUALITY CHECK + STALENESS DETECTION
# ==============================================================================
print(f"\n  Checking micro data quality + staleness...")

master_timestamps = {}
incomplete_tickers = []
complete_tickers = []

for ticker in SCORING_POOL:
    clean = ticker.replace("I:", "").replace("X:", "")
    file_ticker = TICKER_FILE_MAP.get(clean, clean)
    blob_path = f"{INFERENCE_DIR}/{file_ticker}_master_inference.parquet"
    blob = bucket.blob(blob_path)
    if not blob.exists():
        incomplete_tickers.append(ticker)
        continue
    blob.reload()
    master_timestamps[ticker] = blob.updated

    uri = f"gs://{BUCKET_NAME}/{blob_path}"
    try:
        df_check = pd.read_parquet(uri, columns=[MICRO_CHECK_COL])
        if df_check.empty:
            incomplete_tickers.append(ticker)
            continue
        nonzero_pct = (df_check[MICRO_CHECK_COL] != 0).sum() / len(df_check) * 100
        if nonzero_pct >= MICRO_MIN_PCT:
            complete_tickers.append(ticker)
        else:
            incomplete_tickers.append(ticker)
        del df_check
    except Exception:
        incomplete_tickers.append(ticker)

print(f"  Complete (has micro): {len(complete_tickers)}")
print(f"  Incomplete (no micro): {len(incomplete_tickers)}")

work_by_experiment = {}
total_fresh = 0
total_to_score = 0

for exp_id in EXPERIMENTS_TO_SCORE:
    ecfg = EXP_CONFIGS[exp_id]
    model_id = ecfg['model_id']
    seeds = ecfg['seeds']
    results_prefix = f"{GCS_RESULTS_BASE}/Universal_Exp{model_id}"

    exp_work = {}
    exp_fresh = 0

    for ticker in complete_tickers:
        master_ts = master_timestamps.get(ticker)
        seeds_needed = []
        seeds_fresh = 0

        for seed in seeds:
            csv_blob = bucket.blob(f"{results_prefix}/{ticker}/Seed_{seed}_Results.csv")
            if csv_blob.exists():
                csv_blob.reload()
                if master_ts and csv_blob.updated >= master_ts:
                    seeds_fresh += 1
                    continue
            seeds_needed.append(seed)

        if seeds_needed:
            exp_work[ticker] = seeds_needed
        exp_fresh += seeds_fresh

    work_by_experiment[exp_id] = exp_work
    n_work = sum(len(s) for s in exp_work.values())
    total_fresh += exp_fresh
    total_to_score += n_work
    print(f"    Exp {exp_id}: {len(exp_work)} tickers to score, "
          f"{n_work} seed-pairs, {exp_fresh} fresh")

print(f"\n  TOTAL: {total_to_score} to score, {total_fresh} already fresh")

if total_to_score == 0:
    print("\n  All CSVs are fresh! Nothing to score.")
    print(f"  Results at: gs://{BUCKET_NAME}/{GCS_RESULTS_BASE}/")
    raise SystemExit


# ==============================================================================
# STEP 3: LOAD MARKET CONTEXT (one-time)
# ==============================================================================
market_df = build_market_context()
print(f"  Market context: {len(market_df)} rows")


# ==============================================================================
# STEP 4: SCORE — experiment-first, ticker-second
# ==============================================================================
grand_scored = 0
grand_failed = 0

for exp_id in EXPERIMENTS_TO_SCORE:
    ecfg = EXP_CONFIGS[exp_id]
    model_id = ecfg['model_id']
    n_assets = ecfg['n_assets']
    stats_id = ecfg['stats_id']
    seeds = ecfg['seeds']
    aware_cutoff = pd.Timestamp(SCORE_CUTOFF).tz_localize('UTC')
    results_prefix = f"{GCS_RESULTS_BASE}/Universal_Exp{model_id}"

    exp_work = work_by_experiment[exp_id]
    if not exp_work:
        print(f"\n  Exp {exp_id}: nothing to score, skipping")
        continue

    n_tickers = len(exp_work)
    n_pairs = sum(len(s) for s in exp_work.values())
    print(f"\n{'='*80}")
    print(f"  Exp {exp_id} ({model_id}) — {n_tickers} tickers, {n_pairs} seed-pairs")
    print(f"  Architecture: {ecfg['model_builder']}  |  Teachers: {n_assets-1}")
    print(f"  Cutoff: {SCORE_CUTOFF}  |  Stats: {stats_id}")
    print(f"{'='*80}")

    stats = load_scaling_stats(stats_id)
    print(f"  Scaling stats loaded: {len(stats)} features")

    print(f"  Downloading models...")
    available_seeds = []
    for seed in seeds:
        blob_path = f"{GCS_MODEL_PATH}/UNIVERSAL_S{seed}_M{model_id}.h5"
        local_path = os.path.join(LOCAL_MODEL_DIR, f"UNIVERSAL_S{seed}_M{model_id}.h5")
        if bucket.blob(blob_path).exists():
            available_seeds.append(seed)
            if not os.path.exists(local_path):
                bucket.blob(blob_path).download_to_filename(local_path)
        else:
            print(f"    S{seed}: MISSING")
    print(f"  {len(available_seeds)}/{len(seeds)} models available")

    ticker_idx = 0
    for ticker, seeds_needed in exp_work.items():
        ticker_idx += 1
        seeds_to_score = [s for s in seeds_needed if s in available_seeds]
        if not seeds_to_score:
            continue

        print(f"\n  [{ticker_idx}/{n_tickers}] {ticker} — {len(seeds_to_score)} seeds")

        df_target = engineer_asset(ticker, market_df)
        if df_target is None:
            print(f"    SKIP (no data)")
            grand_failed += 1
            continue

        df_scaled = apply_scaling(df_target, stats)
        test_mask = df_scaled.index > aware_cutoff
        df_test = df_scaled[test_mask]

        idx_v = np.where(df_test['Valid_Row'].values)[0]
        idx_v = idx_v[idx_v >= TIME_STEPS]
        if len(idx_v) == 0:
            print(f"    SKIP (0 valid rows post-cutoff)")
            del df_target, df_scaled, df_test; gc.collect()
            continue

        XA_v = df_test[ASSET_COLS].values
        XC_v = df_test[CONTEXT_COLS].values
        asset_id_val = np.int32(ecfg['asset_index'].get(ticker, ecfg['other_idx']))

        max_date = df_test.index[idx_v[-1]].strftime('%Y-%m-%d')

        for seed in seeds_to_score:
            tf.keras.backend.clear_session()
            _model_cache.clear()

            model = get_model(model_id, seed, n_assets, ecfg)
            if model is None:
                print(f"    S{seed}: model load failed, skipping")
                continue

            probs = []
            for i in range(0, len(idx_v), 4096):
                batch_idx = idx_v[i:i+4096]
                XA_b = np.array([XA_v[j - TIME_STEPS:j] for j in batch_idx])
                XC_b = np.array([XC_v[j - 1] for j in batch_idx])
                XID_b = np.full((len(batch_idx), 1), asset_id_val, dtype=np.int32)
                p = model.predict([XA_b, XC_b, XID_b], verbose=0, batch_size=256)
                probs.extend(p.flatten())

            results_df = pd.DataFrame({
                'Target':     df_test['Target'].values[idx_v],
                'Prob_Spike': probs,
                'Prediction': (np.array(probs) > 0.5).astype(int),
                'Close':      df_test['Close'].values[idx_v],
                'Ticker':     ticker
            }, index=df_test.index[idx_v])

            gcs_csv_path = f"{results_prefix}/{ticker}/Seed_{seed}_Results.csv"
            csv_bytes = results_df.to_csv().encode()
            bucket.blob(gcs_csv_path).upload_from_string(csv_bytes, content_type='text/csv')

            grand_scored += 1
            del model, probs, results_df
            gc.collect()

        print(f"    {len(seeds_to_score)} seeds scored → {max_date}")
        del df_target, df_scaled, df_test, XA_v, XC_v
        gc.collect()

    clear_model_cache()
    for seed in available_seeds:
        local_path = os.path.join(LOCAL_MODEL_DIR, f"UNIVERSAL_S{seed}_M{model_id}.h5")
        if os.path.exists(local_path):
            os.remove(local_path)
    print(f"\n  Exp {exp_id} complete — models cleared")


# ==============================================================================
# STEP 5: SUMMARY — verify end dates
# ==============================================================================
print(f"\n\n{'='*80}")
print(f"  SCORING COMPLETE")
print(f"  Scored: {grand_scored}  |  Failed: {grand_failed}  |  Fresh: {total_fresh}")
print(f"  Output: gs://{BUCKET_NAME}/{GCS_RESULTS_BASE}/")
print(f"{'='*80}")

print(f"\n  End date verification (sampling first ticker per experiment):")
for exp_id in EXPERIMENTS_TO_SCORE:
    ecfg = EXP_CONFIGS[exp_id]
    model_id = ecfg['model_id']
    results_prefix = f"{GCS_RESULTS_BASE}/Universal_Exp{model_id}"
    first_seed = ecfg['seeds'][0]

    sample_ticker = complete_tickers[0] if complete_tickers else None
    if sample_ticker is None:
        continue

    csv_path = f"{results_prefix}/{sample_ticker}/Seed_{first_seed}_Results.csv"
    blob = bucket.blob(csv_path)
    if blob.exists():
        df_check = pd.read_csv(io.BytesIO(blob.download_as_bytes()),
                               index_col=0, parse_dates=True)
        print(f"    Exp {exp_id}: {df_check.index.min().date()} → {df_check.index.max().date()}"
              f"  ({len(df_check)} rows, {sample_ticker}/S{first_seed})")
    else:
        print(f"    Exp {exp_id}: no CSV found for {sample_ticker}")

print(f"\n  Next: run backtest_deep_optimization.py with:")
print(f"    RESULTS_DIR = '{GCS_RESULTS_BASE}'")
print(f"    TEST_END = '2026-07-03'")
print(f"{'='*80}")
