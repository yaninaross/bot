"""
Exp 230-235 Walkforward Training (v2 Model, 20 Curated Teachers, 6 Folds)
==========================================================================
Fixes three Exp 220 failure modes:
  1. Signal dilution → curated 20-teacher set (≥2.0 spike/mo recent threshold)
  2. STATS_POOL contamination → stats from 20 teachers only (not 47)
  3. Augmentation noise → removed (matching Exp 180 which outperformed 220)
Plus class-ratio rebalancing: natural per-ticker distribution preserved,
positive class upweighted to ~10:90 effective ratio (penalty for missed spikes).

Folds:
  230: cutoff Aug 2025    231: cutoff Sep 2025    232: cutoff Oct 2025
  233: cutoff Nov 2025    234: cutoff Dec 2025    235: cutoff Jan 2026

Set FOLD_RANGE to control which folds to train in this notebook.
Run multiple notebooks with different FOLD_RANGE for parallel training.
"""

import sys
import subprocess
import os
import json
import random

import numpy as np
import pandas as pd
import pytz
from datetime import datetime
import gc
import ctypes

def _release_memory():
    gc.collect()
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

# ==============================================================================
# ENVIRONMENT SYNC & AUTH
# ==============================================================================
SCRIPT_VERSION = "230WF.1"
print(f"Script version: {SCRIPT_VERSION}")
print("Syncing environment for Experiment 230-235 Walkforward (20 Curated Teachers)...")

try:
    import tensorflow as tf
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "tensorflow[and-cuda]", "pandas_ta",
                           "google-cloud-storage", "pyarrow", "gcsfs", "--quiet"])
    import tensorflow as tf

try:
    import pandas_ta as ta
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas_ta", "--no-deps", "--quiet"])
    import pandas_ta as ta

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, GlobalAveragePooling1D,
    LayerNormalization, MultiHeadAttention, Add,
    BatchNormalization, Concatenate, Embedding, Flatten,
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras import mixed_precision
from google.cloud import storage

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass
client = storage.Client()
bucket = client.bucket(os.environ.get("GCS_BUCKET", ""))

# ==============================================================================
# GPU + MULTI-GPU SETUP
# ==============================================================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    mixed_precision.set_global_policy('mixed_float16')
    print(f"GPU ACTIVE: {len(gpus)} GPU(s) detected, Mixed Precision Enabled")

if len(gpus) > 1:
    strategy = tf.distribute.MirroredStrategy()
    print(f"Multi-GPU: MirroredStrategy with {strategy.num_replicas_in_sync} replicas")
else:
    strategy = tf.distribute.get_strategy()
    print(f"Single-device strategy")

NUM_REPLICAS = strategy.num_replicas_in_sync

# ==============================================================================
# COLUMN DEFINITIONS
# ==============================================================================
MICRO_COLS = ['Spread', 'OBI', 'Volume_Delta', 'Trade_Count', 'Delta_Pct',
              'bsize', 'asize', 'Total_Vol']

ROC_HORIZONS = [5, 30, 60, 120]

ASSET_COLS = (['Gap_Pct', 'RSI_14', 'RSI_60', 'RVOL', 'Compression',
               'ATR_14', 'Dist_to_PM_High', 'Bar_Is_Real'] +
              [f'ROC_{h}m' for h in ROC_HORIZONS] + MICRO_COLS)

TIME_STEPS = 240
EMBED_DIM  = 16
PST = pytz.timezone('America/Los_Angeles')
TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

# ==============================================================================
# CONFIGURATION — 6 WALKFORWARD FOLDS
# ==============================================================================
ALL_FOLDS = [
    {'exp_id': '230_XSECTOR', 'cutoff': '2025-08-01', 'seed_base': 23000},
    {'exp_id': '231_XSECTOR', 'cutoff': '2025-09-01', 'seed_base': 23100},
    {'exp_id': '232_XSECTOR', 'cutoff': '2025-10-01', 'seed_base': 23200},
    {'exp_id': '233_XSECTOR', 'cutoff': '2025-11-01', 'seed_base': 23300},
    {'exp_id': '234_XSECTOR', 'cutoff': '2025-12-01', 'seed_base': 23400},
    {'exp_id': '235_XSECTOR', 'cutoff': '2026-01-01', 'seed_base': 23500},
]

# Set which folds to train in this notebook (e.g. [0,1,2] for folds 230-232)
# Leave empty to train all 6
FOLD_RANGE = []

FOLDS_TO_TRAIN = [ALL_FOLDS[i] for i in FOLD_RANGE] if FOLD_RANGE else ALL_FOLDS
SEEDS_PER_FOLD = 15

TRAIN_TEACHERS = [
    # Semiconductors (4) — all ≥3.35/mo recent
    'AEHR', 'CRDO', 'MXL', 'WOLF',
    # Crypto mining (1)
    'KEEL',
    # Space / defense (3)
    'LUNR', 'MNTS', 'SPIR',
    # Energy / nuclear (4)
    'FCEL', 'NEOV', 'NNE', 'SMR',
    # Small-cap tech (3)
    'NBIS', 'ONDS', 'SLS',
    # Other high-spike (5)
    'BGDE', 'PLUG', 'RDW', 'UCTT', 'VOYG',
]

TICKER_START_DATES = {}

STATS_POOL   = TRAIN_TEACHERS
SCORING_POOL = list(TRAIN_TEACHERS) + [
    # All other downloaded tickers — score broadly, WF optimizer selects
    'ACHR', 'ACMR', 'ALAB', 'AMD', 'APLD', 'ARM', 'ASTS', 'AVAV',
    'BBAI', 'BE', 'BEAM', 'BHP', 'BKSY', 'BTBT',
    'CAT', 'CCJ', 'CEG', 'CIFR', 'CLSK', 'COIN', 'COP', 'CORZ',
    'DUK', 'EQIX', 'ETN', 'FCX', 'FLNC',
    'GD', 'GE', 'GEV', 'HOOD', 'HUT', 'IREN',
    'JOBY', 'KTOS',
    'LHX', 'LIN', 'LMT',
    'MARA', 'MOD', 'MP', 'MRNA', 'MSTR', 'MU', 'MVIS',
    'NEE', 'NNDM', 'NOC', 'NTLA', 'NVDA',
    'OKLO', 'PL', 'PLTR', 'POWL',
    'RCAT', 'RIOT', 'RKLB', 'RTX',
    'SMCI', 'SOUN',
    'TSLA', 'TWST', 'VRT', 'VST', 'WULF', 'XOM',
]
SCORING_POOL = sorted(set(SCORING_POOL))

CONTEXT_ASSETS = ['SPY', 'VIX', 'TLT', 'IWM']

CONTEXT_COLS = []
for _t in CONTEXT_ASSETS:
    CONTEXT_COLS.append(f'{_t}_Anchor_ROC')
    for _h in ROC_HORIZONS:
        CONTEXT_COLS.append(f'{_t}_ROC_{_h}m')
CONTEXT_COLS += ['Time_Pos']
ALL_COLS = ASSET_COLS + CONTEXT_COLS

DATA_START_DATE   = "2022-01-01"
TRAIN_START_DATE  = "2022-02-01"
SPIKE_THRESHOLD   = 0.10
ASL_GAMMA_NEG     = 4.0
ASL_GAMMA_POS     = 0.5
ASL_CLIP          = 0.05
ASL_ALPHA_POS     = 2.0       # positive class cost multiplier (FN costs 2x FP)
SAMPLE_WEIGHT_CAP = 5.0
TARGET_POS_RATIO  = 0.10      # effective positive class ratio (natural ~3-5%, shift to ~10%)
PREC_TOP_K        = 0.02

BUCKET_NAME   = os.environ.get("GCS_BUCKET", "")
INFERENCE_DIR = "Data/Inference_v2"
OHLCV_DIR     = "Data/Inference"

LOCAL_RESULTS_DIR = "Results"
LOCAL_MODEL_DIR   = "Models"
os.makedirs(LOCAL_RESULTS_DIR, exist_ok=True)
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

GCS_MODEL_PATH   = "Algos/to_deploy/Saved_models_v2/UNIVERSAL/Direction"
STATS_DIR        = "Data/Inference_v2"

PER_REPLICA_BATCH = 256
GLOBAL_BATCH_SIZE = PER_REPLICA_BATCH * NUM_REPLICAS

ASSET_INDEX = {ticker: i for i, ticker in enumerate(TRAIN_TEACHERS)}
N_ASSETS    = len(TRAIN_TEACHERS) + 1   # 21
OTHER_IDX   = len(TRAIN_TEACHERS)       # 20

print(f"\nConfig: {len(TRAIN_TEACHERS)} teachers, {N_ASSETS} embeddings, "
      f"batch={PER_REPLICA_BATCH}x{NUM_REPLICAS}={GLOBAL_BATCH_SIZE}")
print(f"No augmentation (matching Exp 180). Class-ratio rebalancing + alpha_pos={ASL_ALPHA_POS}.")
print(f"Folds to train: {[f['exp_id'] for f in FOLDS_TO_TRAIN]}")
print(f"Seeds per fold: {SEEDS_PER_FOLD}")

# ==============================================================================
# DATA LOADING
# ==============================================================================
def load_master(ticker, tail_days=None, add_micro_zeros=True):
    clean = ticker.replace("I:", "").replace("X:", "")
    file_ticker = TICKER_FILE_MAP.get(clean, clean)

    uri = f"gs://{BUCKET_NAME}/{INFERENCE_DIR}/{file_ticker}_master_inference.parquet"
    fallback_uri = f"gs://{BUCKET_NAME}/{OHLCV_DIR}/{file_ticker}_1m_full_historical_data.parquet"

    df = None
    for attempt_uri in [uri, fallback_uri]:
        try:
            df = pd.read_parquet(attempt_uri)
            if attempt_uri == fallback_uri:
                print(f"      {ticker}: loaded from OHLCV fallback (no master_inference)")
            break
        except Exception:
            continue

    if df is None:
        print(f"   Failed to load {ticker}: no master_inference or OHLCV found")
        return None

    try:
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
        if add_micro_zeros:
            for c in MICRO_COLS:
                if c not in df.columns:
                    df[c] = 0.0
        df = df[df.index.notna()].sort_index()
        if tail_days is not None:
            cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=tail_days)
            df = df[df.index >= cutoff]
            if df.empty:
                print(f"      {ticker}: no recent data -- skipping")
                return None
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


# ==============================================================================
# MARKET CONTEXT
# ==============================================================================
def build_market_context():
    print("   Building market context...")
    frames = []
    for ticker in CONTEXT_ASSETS:
        df = load_master(ticker)
        if df is None:
            print(f"      Could not load context asset {ticker}")
            continue
        df = df.sort_index()
        df = heal_data(df)
        daily_c = df['Close'].resample('D').last().ffill().shift(1).reindex(
            df.index, method='ffill')
        cols = {f'{ticker}_Anchor_ROC': (df['Close'] - daily_c) / daily_c}
        for h in ROC_HORIZONS:
            cols[f'{ticker}_ROC_{h}m'] = df['Close'].pct_change(h).fillna(0)
        frames.append(pd.DataFrame(cols, index=df.index))
        del df, daily_c
        _release_memory()
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


def apply_scaling(df, stats, inplace=False):
    df_s = df if inplace else df.copy()
    for col in ALL_COLS:
        if col in ('Time_Pos', 'Bar_Is_Real'):
            continue
        if col in stats:
            df_s[col] = ((df_s[col] - stats[col]['med'])
                         / stats[col]['iqr']).clip(-10, 10).astype('float32')
    return df_s


# ==============================================================================
# TARGET COMPUTATION
# ==============================================================================
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
    close_safe = df['Close'].replace(0, np.nan)
    df['Spike_Mag'] = ((df['Future_Max'] / close_safe) - 1).clip(lower=0).fillna(0)
    df['Sample_Weight'] = np.where(
        df['Target'] == 0, 1.0,
        np.clip(df['Spike_Mag'] / SPIKE_THRESHOLD, 1.0, SAMPLE_WEIGHT_CAP))
    keep_cols = ALL_COLS + ['Target', 'Valid_Row', 'Sample_Weight',
                            'Open', 'High', 'Low', 'Close', 'Volume']
    drop_cols = [c for c in df.columns if c not in keep_cols]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    for c in df.columns:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)

    return df


def engineer_asset(asset_name, market_df):
    df = load_master(asset_name, tail_days=None, add_micro_zeros=True)
    if df is None:
        return None
    df = df.sort_index()
    ticker_start = TICKER_START_DATES.get(asset_name, DATA_START_DATE)
    df = df.loc[pd.to_datetime(ticker_start).tz_localize('UTC'):]
    if asset_name in TICKER_START_DATES:
        print(f"   {asset_name}: using custom start date {ticker_start}")
    df = heal_data(df)
    df = engineer_features(df, market_df)
    df = add_targets(df)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


# ==============================================================================
# MODEL ARCHITECTURE (identical to 180-185)
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


def build_universal_model():
    in_a  = Input(shape=(TIME_STEPS, len(ASSET_COLS)), name='asset_in')
    in_c  = Input(shape=(len(CONTEXT_COLS),),          name='context_in')
    in_id = Input(shape=(1,), dtype=tf.int32,          name='asset_id')
    x = transformer_encoder(Dense(128)(in_a), 128, 4, 128)
    x = GlobalAveragePooling1D()(transformer_encoder(x, 128, 4, 128))
    y = Dense(128, activation='relu')(LayerNormalization()(in_c))
    e = Embedding(N_ASSETS, EMBED_DIM, name='asset_embed')(in_id)
    e = Flatten()(e)
    e = Dense(32, activation='relu')(e)
    z = Concatenate()([x, Dropout(0.5)(y), e])
    z = Dense(128, activation='relu', kernel_regularizer=l2(5e-3))(z)
    out = Dense(1, activation='sigmoid', dtype='float32')(Dropout(0.5)(z))
    return Model(inputs=[in_a, in_c, in_id], outputs=out)


# ==============================================================================
# ASL LOSS
# ==============================================================================
def asymmetric_loss(gamma_neg=ASL_GAMMA_NEG, gamma_pos=ASL_GAMMA_POS,
                    clip=ASL_CLIP, alpha_pos=ASL_ALPHA_POS):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        loss_pos = alpha_pos * y_true * tf.pow(1.0 - y_pred, gamma_pos) * tf.math.log(y_pred)
        y_pred_neg = tf.clip_by_value(y_pred - clip, 0.0, 1.0)
        loss_neg = (1.0 - y_true) * tf.pow(y_pred_neg, gamma_neg) * tf.math.log(
            tf.clip_by_value(1.0 - y_pred_neg, 1e-7, 1.0))
        return -tf.reduce_mean(loss_pos + loss_neg)
    return loss


# ==============================================================================
# CUSTOM METRICS
# ==============================================================================
class PrecisionAtTopK(tf.keras.metrics.Metric):
    def __init__(self, k=PREC_TOP_K, name='prec_top2pct', **kwargs):
        super().__init__(name=name, **kwargs)
        self.k     = k
        self.tp    = self.add_weight(name='tp',    initializer='zeros')
        self.total = self.add_weight(name='total', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_flat = tf.reshape(y_pred,  [-1])
        y_true_flat = tf.reshape(tf.cast(y_true, tf.float32), [-1])
        n = tf.shape(y_pred_flat)[0]
        k = tf.maximum(tf.cast(tf.round(tf.cast(n, tf.float32) * self.k), tf.int32), 1)
        _, top_idx  = tf.math.top_k(y_pred_flat, k=k)
        top_labels  = tf.gather(y_true_flat, top_idx)
        self.tp.assign_add(tf.reduce_sum(top_labels))
        self.total.assign_add(tf.cast(k, tf.float32))

    def result(self):
        return tf.math.divide_no_nan(self.tp, self.total)

    def reset_state(self):
        self.tp.assign(0.)
        self.total.assign(0.)


# ==============================================================================
# STREAMING STATS (per-ticker, no giant concat)
# ==============================================================================
def _compute_stats_streaming(market_df, aware_cutoff):
    print("\nComputing scaling stats (per-ticker streaming)...")
    col_arrays = {col: [] for col in ALL_COLS if col != 'Time_Pos'}
    diag = {}

    for ticker in STATS_POOL:
        print(f"   Loading {ticker}...")
        df = engineer_asset(ticker, market_df)
        if df is None:
            continue
        df = df.loc[TRAIN_START_DATE:]
        print(f"      rows={len(df):,}  range: {df.index.min()} to {df.index.max()}")

        mask_vr = df['Valid_Row'] == 1
        vr_count = int(mask_vr.sum())
        sr = float(df.loc[mask_vr, 'Target'].mean()) if vr_count > 0 else 0
        n_val = int((df.loc[mask_vr].index > aware_cutoff).sum())

        mask_pos = mask_vr & (df['Target'] == 1) & (df.index <= aware_cutoff)
        sw_info = None
        if mask_pos.sum() > 0:
            sw = df.loc[mask_pos, 'Sample_Weight']
            sw_info = {'n_pos': int(mask_pos.sum()), 'avg': float(sw.mean()),
                       'max': float(sw.max()), 'pct_cap': float((sw >= SAMPLE_WEIGHT_CAP).mean())}

        diag[ticker] = {'valid_rows': vr_count, 'spike_rate': sr,
                         'n_val': n_val, 'sw': sw_info}

        stats_mask = mask_vr & (df.index <= aware_cutoff)
        df_stats = df[stats_mask]
        for col in col_arrays:
            if col in df_stats.columns:
                col_arrays[col].append(df_stats[col].values.astype('float32'))

        del df, df_stats
        _release_memory()

    stats = {}
    for col, arrs in col_arrays.items():
        if not arrs:
            continue
        v = np.concatenate(arrs)
        stats[col] = {'med': float(np.median(v)),
                      'iqr': float(max(1e-6, np.percentile(v, 75) - np.percentile(v, 25)))}
        del v
    del col_arrays
    _release_memory()

    return stats, diag


def _print_diagnostics(diag):
    for t, info in diag.items():
        print(f"   {t:<6}: valid_rows={info['valid_rows']:,}  spike_rate={info['spike_rate']:.1%}  val_rows={info['n_val']:,}")

    print(f"\nAsset Embedding Index:")
    for t, idx in ASSET_INDEX.items():
        print(f"   [{idx}] {t}")
    print(f"   [{OTHER_IDX}] OTHER (out-of-training tickers)")

    print(f"\nSample weight distribution (train positives only, cap={SAMPLE_WEIGHT_CAP}x):")
    for t, info in diag.items():
        if info['sw']:
            sw = info['sw']
            print(f"   {t:<6}: n_pos={sw['n_pos']:,}  avg_weight={sw['avg']:.2f}  "
                  f"max_weight={sw['max']:.2f}  pct_at_cap={sw['pct_cap']:.1%}")


# ==============================================================================
# PER-TICKER ARRAY BUILDER (with per-ticker equalized weighting)
# ==============================================================================
def _build_ticker_arrays(market_df, stats, tickers, aware_cutoff, steps):
    print("\nBuilding train+val arrays (per-ticker streaming)...")
    tr_arrays = []
    val_arrays = []

    for ticker in tickers:
        df = engineer_asset(ticker, market_df)
        if df is None:
            continue
        df = df.loc[TRAIN_START_DATE:]
        apply_scaling(df, stats, inplace=True)

        asset_id_val = np.int32(ASSET_INDEX.get(ticker, OTHER_IDX))

        for label, df_slice, arr_list in [
            ('TRAIN', df[df.index <= aware_cutoff], tr_arrays),
            ('VAL',   df[df.index > aware_cutoff],  val_arrays),
        ]:
            if len(df_slice) < steps:
                continue
            XA = df_slice[ASSET_COLS].values.astype('float32')
            XC = df_slice[CONTEXT_COLS].values.astype('float32')
            Y  = df_slice['Target'].values
            SW = df_slice['Sample_Weight'].values.astype('float32') if 'Sample_Weight' in df_slice.columns \
                 else np.ones(len(df_slice), dtype=np.float32)
            vi = np.where(df_slice['Valid_Row'].values)[0]
            vi = vi[vi >= steps]
            if len(vi) > 0:
                arr_list.append((XA, XC, Y, SW, vi.copy(), asset_id_val))
                if label == 'TRAIN':
                    spike_rate = Y[vi].mean()
                    print(f"   [{ticker}] valid={len(vi)}  spike={spike_rate:.1%}  id={asset_id_val}")

        del df
        _release_memory()

    # --- Class-ratio rebalancing (train only) ---
    # Shift effective positive:negative ratio toward TARGET_POS_RATIO (~10:90)
    # while preserving natural per-ticker distribution + magnitude weighting.
    # Positive class gets upweighted (penalty for missing spikes);
    # negative class stays at natural weight.
    total_pos_w = 0.0
    total_neg_w = 0.0
    per_ticker_stats = []
    for XA, XC, Y, SW, vi, aid in tr_arrays:
        pos_mask = Y[vi] == 1
        neg_mask = Y[vi] == 0
        pw = float(SW[vi][pos_mask].sum()) if pos_mask.sum() > 0 else 0.0
        nw = float(SW[vi][neg_mask].sum()) if neg_mask.sum() > 0 else 0.0
        total_pos_w += pw
        total_neg_w += nw
        ticker_name = [t for t, i in ASSET_INDEX.items() if i == aid]
        ticker_name = ticker_name[0] if ticker_name else f"id={aid}"
        per_ticker_stats.append((ticker_name, int(pos_mask.sum()), int(neg_mask.sum()), pw))

    total_w = total_pos_w + total_neg_w
    natural_ratio = total_pos_w / total_w if total_w > 0 else 0.0
    T = TARGET_POS_RATIO

    if natural_ratio > 0 and natural_ratio < T:
        pos_mult = (T * total_neg_w) / ((1.0 - T) * total_pos_w)
        for XA, XC, Y, SW, vi, aid in tr_arrays:
            pos_mask_full = Y == 1
            SW[pos_mask_full] *= np.float32(pos_mult)
    else:
        pos_mult = 1.0

    print(f"\n   Class-ratio rebalancing (train only):")
    print(f"      Natural pos ratio: {natural_ratio:.2%}  ({total_pos_w:.0f} / {total_w:.0f})")
    print(f"      Target  pos ratio: {T:.0%}")
    print(f"      Positive class multiplier: {pos_mult:.3f}")
    print(f"      Per-ticker contribution (natural, no equalization):")
    for tname, npos, nneg, pw in sorted(per_ticker_stats, key=lambda x: -x[3]):
        print(f"         {tname:<6}: pos={npos:5d}  neg={nneg:6d}  pos_weight={pw:8.1f}")

    print(f"   Train: {len(tr_arrays)} tickers   Val: {len(val_arrays)} tickers")
    return tr_arrays, val_arrays


# ==============================================================================
# DATASET FROM PRE-BUILT ARRAYS
# ==============================================================================
def _make_ds_from_arrays(ticker_arrays, steps, batch_size, use_weights=True):
    if not ticker_arrays:
        return None, 0

    combined = np.array(
        [(t, int(i)) for t, (_, _, _, _, idx, _) in enumerate(ticker_arrays) for i in idx],
        dtype=np.int64)

    def gen():
        while True:
            perm = np.random.permutation(len(combined))
            for p in perm:
                t_idx, i = combined[p]
                XA, XC, Y, SW, _, asset_id_val = ticker_arrays[t_idx]
                w = SW[i] if use_weights else np.float32(1.0)
                xa_sample = XA[i - steps:i]

                yield (
                    {'asset_in': xa_sample,
                     'context_in': XC[i - 1],
                     'asset_id': np.array([asset_id_val], dtype=np.int32)},
                    Y[i], w)

    ds = (
        tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                {'asset_in':   tf.TensorSpec((steps, len(ASSET_COLS)), tf.float32),
                 'context_in': tf.TensorSpec((len(CONTEXT_COLS),),     tf.float32),
                 'asset_id':   tf.TensorSpec((1,),                     tf.int32)},
                tf.TensorSpec((), tf.float32),
                tf.TensorSpec((), tf.float32)))
        .repeat()
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE))
    return ds, len(combined) // batch_size


# ==============================================================================
# EXECUTION
# ==============================================================================

# --- Pre-scan: show what needs to be done before loading any data ---
print(f"\n{'='*70}")
print(f"PRE-SCAN: Checking existing models and scored CSVs on GCS...")
print(f"{'='*70}")
any_work = False
for fold_cfg in FOLDS_TO_TRAIN:
    fid = fold_cfg['exp_id']
    seeds = [fold_cfg['seed_base'] + 1 + i for i in range(SEEDS_PER_FOLD)]
    gcs_res = f"Algos/Results_all_v2/Universal_Exp{fid}"
    n_trained = 0
    n_scored = 0
    for s in seeds:
        if bucket.blob(f"{GCS_MODEL_PATH}/UNIVERSAL_S{s}_M{fid}.h5").exists():
            n_trained += 1
            missing = [t for t in SCORING_POOL
                       if not bucket.blob(f"{gcs_res}/{t}/Seed_{s}_Results.csv").exists()]
            if not missing:
                n_scored += 1
    n_need_train = SEEDS_PER_FOLD - n_trained
    n_need_score = n_trained - n_scored
    status = "SKIP" if (n_need_train == 0 and n_need_score == 0) else "WORK"
    print(f"  {fid}: {n_trained}/{SEEDS_PER_FOLD} trained, {n_scored}/{SEEDS_PER_FOLD} fully scored "
          f"→ {n_need_train} to train, {n_need_score} to score  [{status}]")
    if n_need_train > 0 or n_need_score > 0:
        any_work = True

if not any_work:
    print(f"\nAll folds fully trained and scored. Nothing to do.")
    import sys; sys.exit(0)

print(f"\nProceeding with data load and training/scoring...\n")

market_df = build_market_context()


# ==============================================================================
# WALKFORWARD FOLD LOOP
# ==============================================================================
for fold_idx, fold_cfg in enumerate(FOLDS_TO_TRAIN):
    FOLD_EXP_ID = fold_cfg['exp_id']
    FOLD_CUTOFF = fold_cfg['cutoff']
    FOLD_SEEDS  = [fold_cfg['seed_base'] + 1 + i for i in range(SEEDS_PER_FOLD)]

    aware_cutoff = pd.to_datetime(FOLD_CUTOFF).tz_localize('UTC')
    gcs_results_path = f"Algos/Results_all_v2/Universal_Exp{FOLD_EXP_ID}"

    print(f"\n{'='*70}")
    print(f"FOLD {fold_idx+1}/{len(FOLDS_TO_TRAIN)}: Exp {FOLD_EXP_ID}  cutoff={FOLD_CUTOFF}  "
          f"seeds={FOLD_SEEDS[0]}-{FOLD_SEEDS[-1]}")
    print(f"{'='*70}")

    # --- Skip check: separate trained vs fully scored ---
    trained_seeds = []
    fully_scored_seeds = []
    for seed in FOLD_SEEDS:
        model_blob_path = f"{GCS_MODEL_PATH}/UNIVERSAL_S{seed}_M{FOLD_EXP_ID}.h5"
        if bucket.blob(model_blob_path).exists():
            trained_seeds.append(seed)
            missing = [t for t in SCORING_POOL
                       if not bucket.blob(f"{gcs_results_path}/{t}/Seed_{seed}_Results.csv").exists()]
            if not missing:
                fully_scored_seeds.append(seed)
            else:
                print(f"   Seed {seed}: model exists, {len(missing)} tickers unscored")

    needs_training = [s for s in FOLD_SEEDS if s not in trained_seeds]
    needs_scoring  = [s for s in trained_seeds if s not in fully_scored_seeds]
    remaining_seeds = needs_training + needs_scoring

    if not remaining_seeds:
        print(f"   All {len(FOLD_SEEDS)} seeds fully trained and scored. Nothing to do.")
        continue

    print(f"   {len(fully_scored_seeds)} seeds fully done, {len(needs_training)} need training, "
          f"{len(needs_scoring)} need scoring only: {remaining_seeds}")

    try:
        # --- Pass 1: Compute stats per-ticker (no concat, ~75MB peak) ---
        stats, diag = _compute_stats_streaming(market_df, aware_cutoff)

        blob = client.bucket(BUCKET_NAME).blob(
            f"{STATS_DIR}/scaling_stats_experiment_{FOLD_EXP_ID}.json")
        blob.upload_from_string(json.dumps(stats, indent=4), content_type='application/json')
        print(f"SAVED STATS TO: {STATS_DIR}/scaling_stats_experiment_{FOLD_EXP_ID}.json")

        _print_diagnostics(diag)
        del diag
        _release_memory()

        # --- Pass 2: Build train+val arrays per-ticker (~3GB vs ~20GB) ---
        if needs_training:
            tr_arrays, val_arrays = _build_ticker_arrays(
                market_df, stats, TRAIN_TEACHERS, aware_cutoff, TIME_STEPS)

        # ==================================================================
        # PHASE 1: TRAINING — 15 seeds (no augmentation, class-ratio rebalanced)
        # ==================================================================
        for seed in needs_training:
            print(f"\nTRAINING SEED {seed} (Fold {FOLD_EXP_ID})")
            tf.keras.backend.clear_session()
            random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

            local_model_path = os.path.join(LOCAL_MODEL_DIR, f"UNIVERSAL_S{seed}_M{FOLD_EXP_ID}.h5")
            gcs_model_blob = f"{GCS_MODEL_PATH}/UNIVERSAL_S{seed}_M{FOLD_EXP_ID}.h5"

            print(f"   Building datasets (no augmentation, global_batch={GLOBAL_BATCH_SIZE})...")
            ds_tr, st_tr = _make_ds_from_arrays(
                tr_arrays, TIME_STEPS, GLOBAL_BATCH_SIZE, use_weights=True)
            ds_v, st_v = _make_ds_from_arrays(
                val_arrays, TIME_STEPS, GLOBAL_BATCH_SIZE, use_weights=False)

            if ds_tr is None or st_tr == 0:
                print(f"   Seed {seed}: no valid training samples found. Skipping.")
                del ds_tr, ds_v; _release_memory()
                continue

            print(f"   Steps/epoch: {st_tr}  Val steps: {st_v if st_v else 'N/A'}")

            with strategy.scope():
                model = build_universal_model()

                try:
                    optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-4)
                except AttributeError:
                    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

                model.compile(
                    optimizer=optimizer,
                    loss=asymmetric_loss(),
                    metrics=[
                        'accuracy',
                        tf.keras.metrics.AUC(name='auc'),
                        tf.keras.metrics.AUC(curve='PR', name='pr_auc'),
                        PrecisionAtTopK(k=PREC_TOP_K, name='prec_top2pct'),
                    ])

            callbacks = [
                tf.keras.callbacks.ModelCheckpoint(
                    local_model_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1),
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=6, restore_best_weights=True, mode='min', verbose=1),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, mode='min', verbose=1),
            ]

            model.fit(
                ds_tr, validation_data=ds_v,
                epochs=50, steps_per_epoch=st_tr, validation_steps=st_v,
                callbacks=callbacks, verbose=1)

            if os.path.exists(local_model_path):
                bucket.blob(gcs_model_blob).upload_from_filename(local_model_path)
                print(f"   UPLOADED MODEL TO: {gcs_model_blob}")
            else:
                print("   Model file not found locally, upload skipped.")

            del model, ds_tr, ds_v
            tf.keras.backend.clear_session()
            _release_memory()
            print(f"   Seed {seed} training complete — memory released.")

        # Free arrays before scoring phase
        if needs_training:
            del tr_arrays, val_arrays
        _release_memory()
        print(f"\n   Training phase done.")

        # ==================================================================
        # PHASE 2: SCORING — load each ticker ONCE, score all seeds
        # ==================================================================
        all_seeds_to_score = list(range(fold_cfg['seed_base'] + 1,
                                        fold_cfg['seed_base'] + 1 + SEEDS_PER_FOLD))
        seeds_needing_any_scoring = []
        for seed in all_seeds_to_score:
            local_model_path = os.path.join(LOCAL_MODEL_DIR, f"UNIVERSAL_S{seed}_M{FOLD_EXP_ID}.h5")
            gcs_model_blob = f"{GCS_MODEL_PATH}/UNIVERSAL_S{seed}_M{FOLD_EXP_ID}.h5"
            if not os.path.exists(local_model_path):
                if bucket.blob(gcs_model_blob).exists():
                    bucket.blob(gcs_model_blob).download_to_filename(local_model_path)
                else:
                    continue
            missing = [t for t in SCORING_POOL
                       if not bucket.blob(f"{gcs_results_path}/{t}/Seed_{seed}_Results.csv").exists()]
            if missing:
                seeds_needing_any_scoring.append(seed)

        if not seeds_needing_any_scoring:
            print(f"\n   All seeds fully scored. Skipping scoring phase.")
        else:
            print(f"\n   SCORING PHASE: {len(seeds_needing_any_scoring)} seeds need scoring")
            print(f"   Loading scoring ticker data (cached per ticker, not per seed)...")

            for score_ticker in SCORING_POOL:
                seeds_for_this_ticker = []
                for seed in seeds_needing_any_scoring:
                    gcs_csv_check = f"{gcs_results_path}/{score_ticker}/Seed_{seed}_Results.csv"
                    if not bucket.blob(gcs_csv_check).exists():
                        seeds_for_this_ticker.append(seed)

                if not seeds_for_this_ticker:
                    continue

                print(f"\n   Scoring {score_ticker} ({len(seeds_for_this_ticker)} seeds)...")
                df_target = engineer_asset(score_ticker, market_df)
                if df_target is None:
                    continue
                df_target_scaled = apply_scaling(df_target, stats)

                test_mask = df_target_scaled.index > aware_cutoff
                df_test   = df_target_scaled[test_mask]

                idx_v = np.where(df_test['Valid_Row'].values)[0]
                idx_v = idx_v[idx_v >= TIME_STEPS]
                if len(idx_v) == 0:
                    del df_target, df_target_scaled, df_test
                    _release_memory()
                    continue

                XA_v = df_test[ASSET_COLS].values
                XC_v = df_test[CONTEXT_COLS].values
                asset_id_val = np.int32(ASSET_INDEX.get(score_ticker, OTHER_IDX))

                for seed in seeds_for_this_ticker:
                    local_model_path = os.path.join(LOCAL_MODEL_DIR, f"UNIVERSAL_S{seed}_M{FOLD_EXP_ID}.h5")
                    if not os.path.exists(local_model_path):
                        continue

                    tf.keras.backend.clear_session()
                    with strategy.scope():
                        model = build_universal_model()
                        model.compile(optimizer='adam', loss='binary_crossentropy')
                    model.load_weights(local_model_path)

                    probs = []
                    for i in range(0, len(idx_v), 4096):
                        batch_idx = idx_v[i:i+4096]
                        XA_b  = np.array([XA_v[j - TIME_STEPS:j] for j in batch_idx])
                        XC_b  = np.array([XC_v[j - 1] for j in batch_idx])
                        XID_b = np.full((len(batch_idx), 1), asset_id_val, dtype=np.int32)
                        p = model.predict([XA_b, XC_b, XID_b], verbose=0).flatten()
                        probs.extend(p)

                    results_df = pd.DataFrame({
                        'Target':     df_test['Target'].values[idx_v],
                        'Prob_Spike': probs,
                        'Prediction': (np.array(probs) > 0.5).astype(int),
                        'Close':      df_test['Close'].values[idx_v],
                        'Ticker':     score_ticker
                    }, index=df_test.index[idx_v])

                    local_csv_name = f"Seed_{seed}_{score_ticker}_Results.csv"
                    local_csv_path = os.path.join(LOCAL_RESULTS_DIR, local_csv_name)
                    results_df.to_csv(local_csv_path)

                    gcs_csv_path = f"{gcs_results_path}/{score_ticker}/Seed_{seed}_Results.csv"
                    bucket.blob(gcs_csv_path).upload_from_filename(local_csv_path)
                    print(f"      S{seed} → {gcs_csv_path}")

                    del model
                    _release_memory()

                del df_target, df_target_scaled, df_test, XA_v, XC_v
                _release_memory()

        _release_memory()
        tf.keras.backend.clear_session()
        print(f"\n  Fold {FOLD_EXP_ID} complete.")

    except Exception as e:
        print(f"\n  *** FOLD {FOLD_EXP_ID} FAILED: {e}")
        import traceback; traceback.print_exc()
        tf.keras.backend.clear_session()
        _release_memory()
        print(f"  Continuing to next fold...")

print(f"\n{'='*70}")
print(f"ALL FOLDS COMPLETE — {[f['exp_id'] for f in FOLDS_TO_TRAIN]}")
print(f"{'='*70}")
