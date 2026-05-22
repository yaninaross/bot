import sys
import subprocess
import os
import io
import json
import random
import pytz

import numpy as np
import pandas as pd
from datetime import datetime, time, timedelta
import gc

# ==============================================================================
# ☀️ PART 1: ENVIRONMENT SYNC & AUTH
# ==============================================================================
SCRIPT_VERSION = "97.0"
print(f"📋 Script version: {SCRIPT_VERSION}")
print("🧹 Syncing environment for Experiment 97 (Expanded Crypto Cluster)...")

# Auto-fix for Protobuf/TF crash
try:
    import tensorflow as tf
    from google.protobuf import descriptor
    if descriptor._USE_C_DESCRIPTORS == False: raise ImportError
except Exception as e:
    print(f"⚠️ TF/Protobuf env check failed ({e}), reinstalling...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "protobuf<4.24.0", "tensorflow[and-cuda]", "pandas_ta", "google-cloud-storage", "pyarrow", "gcsfs", "--quiet"])

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, GlobalAveragePooling1D,
    LayerNormalization, MultiHeadAttention, Add,
    BatchNormalization, Concatenate, Embedding, Flatten
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras import mixed_precision
from google.cloud import storage

# GCS Auth
try:
    from google.colab import auth
    auth.authenticate_user()
except ImportError:
    pass  # Not running in Colab — GCS credentials handled externally
client = storage.Client()
bucket = client.bucket("time-series-bucket-ys")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    mixed_precision.set_global_policy('mixed_float16')
    print(f"🚀 GPU ACTIVE: {len(gpus)} GPU(s) detected, Mixed Precision Enabled")

if len(gpus) > 1:
    strategy = tf.distribute.MirroredStrategy()
    print(f"Multi-GPU: MirroredStrategy with {strategy.num_replicas_in_sync} replicas")
else:
    strategy = tf.distribute.get_strategy()
    print(f"Single-device strategy")

NUM_REPLICAS = strategy.num_replicas_in_sync

# ==============================================================================
# ⚙️ CONFIGURATION
# ==============================================================================
MODEL_ID = "97_MICRO"
SEEDS_TO_RUN = [9701, 9702, 9703, 9704, 9705, 9706, 9707, 9708, 9709, 9710, 9711, 9712, 9713, 9714, 9715]

# exp_97 expands the teacher pool with 7 additional crypto-cluster assets:
# BITF, HUT, CIFR, BTBT, MIGI, HOOD, SMLR — all crypto-correlated small/micro
# caps that should have had more 10%+ spikes in the Oct 2025–Mar 2026 validation
# period, addressing the validation starvation seen in exp_95.
TRAIN_TEACHERS = [
    'MARA', 'IREN', 'WULF', 'APLD', 'RIOT', 'CLSK', 'CORZ', 'MSTR', 'COIN',
    'BITF', 'HUT', 'CIFR', 'BTBT', 'MIGI', 'HOOD', 'SMLR',
]

TICKER_START_DATES = {
    'CORZ': '2024-01-25',  # Skip old shares + frozen bankruptcy prices + 5414% re-listing jump
}

STATS_POOL   = TRAIN_TEACHERS
SCORING_POOL = TRAIN_TEACHERS + [
    # Assets with microstructure scored but not trained on
    'SMCI', 'NVDA', 'VRT', 'AMD', 'PLTR', 'TSLA',
    # Context assets scored on OHLCV-only (no micro)
    'SMH', 'TLT', 'IWM', 'VIX',
]

# 🌍 CONTEXT ASSETS
CONTEXT_ASSETS = ['SMH', 'MOD', 'TLT', 'IWM', 'VIX', 'BTCUSD']

# 📅 Date Alignment — same as exp_95
DATA_START_DATE   = "2022-01-01"
TRAIN_START_DATE  = "2022-02-01"
TRAIN_CUTOFF_DATE = "2025-10-01"

# 🎯 Spike threshold — 10% (confirmed winner vs 6%)
SPIKE_THRESHOLD = 0.10

# ⚖️ Asymmetric Loss (ASL) — unchanged
ASL_GAMMA_NEG = 4.0
ASL_GAMMA_POS = 0.5
ASL_CLIP      = 0.05

# 📐 Sample weight cap — unchanged
SAMPLE_WEIGHT_CAP = 5.0

# 🎯 Precision@TopK
PREC_TOP_K = 0.02

BUCKET_NAME = "time-series-bucket-ys"

# Directories
LOCAL_RESULTS_DIR = "Results"
LOCAL_MODEL_DIR   = "Models"
os.makedirs(LOCAL_RESULTS_DIR, exist_ok=True)
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

GCS_RESULTS_PATH = f"Algos/Results_all/Universal_Exp{MODEL_ID}"
GCS_MODEL_PATH   = f"Algos/to_deploy/Saved_models/UNIVERSAL/Direction"
STATS_DIR        = f"Data/Inference"
INFERENCE_DIR    = f"Data/Inference"

# 🔑 Architecture — identical to exp_95 (TIME_STEPS=240 is net positive)
PER_REPLICA_BATCH = 1024
GLOBAL_BATCH_SIZE = PER_REPLICA_BATCH * NUM_REPLICAS
TIME_STEPS = 240

# 🔑 Asset Embedding — expanded to 16 training tickers + 1 OTHER
EMBED_DIM   = 16
ASSET_INDEX = {ticker: i for i, ticker in enumerate(TRAIN_TEACHERS)}
N_ASSETS    = len(TRAIN_TEACHERS) + 1   # = 17
OTHER_IDX   = len(TRAIN_TEACHERS)       # = 16: shared embedding for out-of-training tickers

# ==============================================================================
# ☁️ 2. DATA LOADING
# ==============================================================================
MICRO_COLS = ['Spread', 'OBI', 'Volume_Delta', 'Trade_Count', 'Delta_Pct', 'bsize', 'asize', 'Total_Vol']

TICKER_FILE_MAP = {'BITF': 'KEEL'}  # BITF renamed to KEEL on Nasdaq 2026-04-06

def load_aligned_asset(ticker):
    clean_ticker = ticker.replace("I:", "").replace("X:", "")
    file_ticker = TICKER_FILE_MAP.get(clean_ticker, clean_ticker)
    gcs_uri = f"gs://{BUCKET_NAME}/{INFERENCE_DIR}/{file_ticker}_master_inference.parquet"

    try:
        df = pd.read_parquet(gcs_uri)
        df = df.reset_index()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

        dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        if dt_cols:
            df['Datetime'] = pd.to_datetime(df[dt_cols[0]], utc=True)
        else:
            num_col = next((c for c in df.columns if c in ('Datetime', 'Date', 'Time', 'Timestamp', 'Level_0')), df.columns[0])
            df['Datetime'] = pd.to_datetime(df[num_col], unit='ms', utc=True)
        df = df.set_index('Datetime')

        df = df[~df.index.duplicated(keep='first')]

        std_map = {
            'Total_vol': 'Total_Vol', 'Trade_count': 'Trade_Count',
            'Volume_delta': 'Volume_Delta', 'Delta_pct': 'Delta_Pct',
            'Obi': 'OBI', 'Bsize': 'bsize', 'Asize': 'asize',
            'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'
        }
        lower_map = {k.lower(): v for k, v in std_map.items()}
        df.columns = [lower_map.get(str(c).strip().lower(), str(c).strip()) for c in df.columns]

        df = df.loc[:, ~df.columns.duplicated(keep='first')].copy()

        for c in MICRO_COLS:
            if c not in df.columns: df[c] = 0.0

        return df[df.index.notna()].sort_index()

    except Exception as e:
        print(f"   ❌ FAILED TO LOAD MASTER PARQUET {ticker}: {e}")
        return None

def heal_data(df):
    df = df.resample('1min').asfreq()
    for c in ['Open', 'High', 'Low', 'Close', 'Spread', 'OBI', 'bsize', 'asize']:
        if c in df.columns: df[c] = df[c].ffill()
    for c in ['Volume', 'Total_Vol', 'Volume_Delta', 'Trade_Count', 'Delta_Pct']:
        if c in df.columns: df[c] = df[c].fillna(0)
    return df

# ==============================================================================
# 🩹 3. FEATURE ENGINEERING
# ==============================================================================
try:
    import pandas_ta as ta
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas_ta", "--quiet"])
    import pandas_ta as ta

ROC_HORIZONS = [5, 30, 60, 120]
ASSET_COLS = ['Gap_Pct', 'RSI_14', 'RSI_60', 'RVOL', 'Compression', 'ATR_14', 'Dist_to_PM_High', 'Bar_Is_Real'] + \
             [f'ROC_{h}m' for h in ROC_HORIZONS] + MICRO_COLS

CONTEXT_COLS = []
for ticker in CONTEXT_ASSETS:
    CONTEXT_COLS.append(f'{ticker}_Anchor_ROC')
    for h in ROC_HORIZONS: CONTEXT_COLS.append(f'{ticker}_ROC_{h}m')
CONTEXT_COLS += ['SMH_RSI', 'Time_Pos']
ALL_COLS = ASSET_COLS + CONTEXT_COLS

def build_market_context():
    print(f"🌍 Building Global Context...")
    frames = []
    for ticker in CONTEXT_ASSETS:
        df = load_aligned_asset(ticker)
        if df is None:
            print(f"   ⚠️ Could not load context asset {ticker}, skipping.")
            continue
        df = df.sort_index()
        df = heal_data(df)

        daily_c = df['Close'].resample('D').last().ffill().shift(1).reindex(df.index, method='ffill')
        cols = {f'{ticker}_Anchor_ROC': (df['Close'] - daily_c) / daily_c}
        for h in ROC_HORIZONS:
            cols[f'{ticker}_ROC_{h}m'] = df['Close'].pct_change(h).fillna(0)
        if ticker == 'SMH':
            cols['SMH_RSI'] = ta.rsi(df['Close'], length=14).fillna(50) / 100.0

        frames.append(pd.DataFrame(cols, index=df.index))
        del df, daily_c; gc.collect()

    if not frames:
        raise RuntimeError("No context assets loaded — check GCS paths.")
    return pd.concat(frames, axis=1).ffill().fillna(0)

def engineer_asset(asset_name, market_df):
    df = load_aligned_asset(asset_name)
    if df is None: return None

    df = df.sort_index()
    ticker_start = TICKER_START_DATES.get(asset_name, DATA_START_DATE)
    df = df.loc[pd.to_datetime(ticker_start).tz_localize('UTC'):]
    if asset_name in TICKER_START_DATES:
        print(f"   📅 {asset_name}: using custom start date {ticker_start}")
    df = heal_data(df)

    daily_c = df['Close'].resample('D').last().shift(1).reindex(df.index, method='ffill')
    df['Gap_Pct'] = (df['Open'] - daily_c) / daily_c

    df['PM_High'] = df.groupby(df.index.date)['High'].cummax()
    df['Dist_to_PM_High'] = (df['Close'] - df['PM_High']) / (df['PM_High'] + 1e-9)
    df['RSI_14'] = ta.rsi(df['Close'], length=14)
    df['RSI_60'] = ta.rsi(df['Close'], length=60)
    df['RVOL'] = df['Volume'] / df['Volume'].rolling(50).mean().clip(lower=1e-9)
    df['ATR_14'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['Compression'] = (df['High'] - df['Low']) / (df['ATR_14'] + 1e-9)
    for h in ROC_HORIZONS: df[f'ROC_{h}m'] = df['Close'].pct_change(h)

    df = df.join(market_df, how='left').ffill().fillna(0)

    pst = pytz.timezone('America/Los_Angeles')
    df_pst = df.index.tz_convert(pst)
    mins = df_pst.hour * 60 + df_pst.minute
    df['Time_Pos'] = np.clip((mins - 390) / 390, 0, 1)

    is_safe = (df_pst.hour < 13) | ((df_pst.hour == 13) & (df_pst.minute == 0))
    is_real = df['Volume'] > 0
    df['Safe_High'] = np.where(is_safe & is_real, df['High'], -1.0)
    df['Future_Max'] = df.groupby(df.index.date)['Safe_High'].transform(lambda x: x[::-1].cummax()[::-1])
    df['Target'] = (df['Future_Max'] > (df['Close'] * (1 + SPIKE_THRESHOLD))).astype(int)
    df['Valid_Row'] = ((mins >= 360) & (mins <= 390) & (df['Volume'] > 0)).astype(int)

    close_safe = df['Close'].replace(0, np.nan)
    df['Spike_Mag'] = ((df['Future_Max'] / close_safe) - 1).clip(lower=0).fillna(0)
    df['Sample_Weight'] = np.where(
        df['Target'] == 0,
        1.0,
        np.clip(df['Spike_Mag'] / SPIKE_THRESHOLD, 1.0, SAMPLE_WEIGHT_CAP)
    )

    df['Bar_Is_Real'] = (df['Volume'] > 0).astype(float)

    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    keep_cols = ALL_COLS + ['Target', 'Valid_Row', 'Sample_Weight', 'Open', 'High', 'Low', 'Close', 'Volume']
    drop_cols = [c for c in df.columns if c not in keep_cols]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    return df

# ==============================================================================
# 🧠 4. MODEL ARCHITECTURE — identical to exp_95
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
    in_id = Input(shape=(1,), dtype=tf.int32,           name='asset_id')

    # Transformer sequence branch
    x = transformer_encoder(Dense(128)(in_a), 128, 4, 128)
    x = GlobalAveragePooling1D()(transformer_encoder(x, 128, 4, 128))

    # Context branch
    y = Dense(128, activation='relu')(BatchNormalization()(in_c))

    # Asset embedding branch
    e = Embedding(N_ASSETS, EMBED_DIM, name='asset_embed')(in_id)  # (batch, 1, EMBED_DIM)
    e = Flatten()(e)                                                 # (batch, EMBED_DIM)
    e = Dense(32, activation='relu')(e)                             # (batch, 32)

    # Fuse all three branches
    z = Concatenate()([x, Dropout(0.5)(y), e])
    z = Dense(128, activation='relu', kernel_regularizer=l2(5e-3))(z)
    out = Dense(1, activation='sigmoid', dtype='float32')(Dropout(0.5)(z))
    return Model(inputs=[in_a, in_c, in_id], outputs=out)

# ==============================================================================
# 📉 ASYMMETRIC LOSS (ASL) — unchanged
# ==============================================================================
def asymmetric_loss(gamma_neg=ASL_GAMMA_NEG, gamma_pos=ASL_GAMMA_POS, clip=ASL_CLIP):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        loss_pos = y_true * tf.pow(1.0 - y_pred, gamma_pos) * tf.math.log(y_pred)

        y_pred_neg = tf.clip_by_value(y_pred - clip, 0.0, 1.0)
        loss_neg = (1.0 - y_true) * tf.pow(y_pred_neg, gamma_neg) * tf.math.log(
            tf.clip_by_value(1.0 - y_pred_neg, 1e-7, 1.0)
        )

        return -tf.reduce_mean(loss_pos + loss_neg)
    return loss

# ==============================================================================
# 📊 CUSTOM METRICS
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
# 🗂️ DATASET BUILDER
# ==============================================================================
def make_per_ticker_ds(df_scaled, tickers, start_dt, end_dt, steps, batch_size,
                       balanced=True, use_weights=True):
    ticker_arrays = []
    for ticker in tickers:
        mask = df_scaled['Ticker'] == ticker
        if start_dt is not None:
            mask &= df_scaled.index > start_dt
        if end_dt is not None:
            mask &= df_scaled.index <= end_dt
        df_t = df_scaled[mask]
        if len(df_t) < steps:
            continue
        XA = df_t[ASSET_COLS].values.astype('float32')
        XC = df_t[CONTEXT_COLS].values.astype('float32')
        Y  = df_t['Target'].values
        SW = df_t['Sample_Weight'].values.astype('float32') if 'Sample_Weight' in df_t.columns \
             else np.ones(len(df_t), dtype=np.float32)
        vi = np.where(df_t['Valid_Row'].values)[0]
        vi = vi[vi >= steps]
        if len(vi) == 0:
            continue
        if balanced:
            i1, i0 = vi[Y[vi] == 1], vi[Y[vi] == 0]
            if len(i1) == 0:
                idx = i0.copy()
            else:
                n_pos = len(i1) * 3
                n_neg = n_pos * 3
                idx = np.concatenate([
                    np.random.choice(i0, n_neg, replace=(len(i0) < n_neg)),
                    np.random.choice(i1, n_pos, replace=True)
                ])
        else:
            idx = vi.copy()

        asset_id_val = np.int32(ASSET_INDEX.get(ticker, OTHER_IDX))
        ticker_arrays.append((XA, XC, Y, SW, idx, asset_id_val))
        spike_rate = Y[vi].mean()
        print(f"   [{ticker}] valid_rows={len(vi)}  spike_rate={spike_rate:.1%}  "
              f"asset_id={asset_id_val}  "
              f"{'n_pos=' + str(len(vi[Y[vi]==1])) if balanced else 'unbalanced'}")

    if not ticker_arrays:
        return None, 0

    combined = np.array(
        [(t, int(i)) for t, (_, _, _, _, idx, _) in enumerate(ticker_arrays) for i in idx],
        dtype=np.int64
    )

    def gen():
        while True:
            perm = np.random.permutation(len(combined))
            for p in perm:
                t_idx, i = combined[p]
                XA, XC, Y, SW, _, asset_id_val = ticker_arrays[t_idx]
                w = SW[i] if use_weights else np.float32(1.0)
                yield (
                    {
                        'asset_in':  XA[i - steps:i],
                        'context_in': XC[i - 1],
                        'asset_id':  np.array([asset_id_val], dtype=np.int32),
                    },
                    Y[i],
                    w,
                )

    ds = (
        tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                {
                    'asset_in':   tf.TensorSpec((steps, len(ASSET_COLS)), dtype=tf.float32),
                    'context_in': tf.TensorSpec((len(CONTEXT_COLS),),     dtype=tf.float32),
                    'asset_id':   tf.TensorSpec((1,),                     dtype=tf.int32),
                },
                tf.TensorSpec((), dtype=tf.float32),
                tf.TensorSpec((), dtype=tf.float32),
            ),
        )
        .repeat()
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )
    return ds, len(combined) // batch_size


def apply_scaling(df, stats, inplace=False):
    df_s = df if inplace else df.copy()
    for col in ALL_COLS:
        if col in ('Time_Pos', 'Bar_Is_Real'): continue
        if col in stats:
            df_s[col] = ((df_s[col] - stats[col]['med']) / stats[col]['iqr']).clip(-10, 10).astype('float32')
    return df_s

# ==============================================================================
# 🚀 5. EXECUTION PIPELINE
# ==============================================================================
market_df = build_market_context()

print("\n📏 Building Stats Pool...")
stats_dfs = []
for ticker in STATS_POOL:
    print(f"   ⏳ Loading {ticker}...")
    df_temp = engineer_asset(ticker, market_df)
    if df_temp is not None:
        print(f"      rows={len(df_temp)}  tz={df_temp.index.tz}  min={df_temp.index.min()}  max={df_temp.index.max()}")
        df_temp = df_temp.loc[TRAIN_START_DATE:]
        print(f"      after filter: rows={len(df_temp)}")
        df_temp['Ticker'] = ticker
        stats_dfs.append(df_temp)
    gc.collect()

df_all_stats = pd.concat(stats_dfs).sort_index()
aware_cutoff = pd.to_datetime(TRAIN_CUTOFF_DATE).tz_localize('UTC')

print("\n📏 Calculating Stats...")
stats_data_raw = df_all_stats[df_all_stats['Ticker'].isin(TRAIN_TEACHERS) & (df_all_stats.index <= aware_cutoff)]
stats = {col: {'med': float(stats_data_raw[col].median()),
               'iqr': float(max(1e-6, stats_data_raw[col].quantile(0.75) - stats_data_raw[col].quantile(0.25)))}
         for col in ALL_COLS if col != 'Time_Pos'}

blob = client.bucket(BUCKET_NAME).blob(f"{STATS_DIR}/scaling_stats_experiment_{MODEL_ID}.json")
blob.upload_from_string(json.dumps(stats, indent=4), content_type='application/json')
print(f"💾 SAVED STATS TO: {STATS_DIR}")

# Free the stats slice (large subset of df_all_stats) before scaling
del stats_data_raw
gc.collect()

print("\n🎓 Scaling Data (in-place — 16 tickers, large dataset)...")
# Scale in-place to avoid doubling memory with a full copy
df_scaled_all = apply_scaling(df_all_stats, stats, inplace=True)
del stats_dfs  # raw per-ticker list no longer needed
gc.collect()

# Diagnostics
_vr  = df_scaled_all['Valid_Row']
_tgt = df_scaled_all['Target']
print(f"\n🔍 Valid_Row check — 0s: {(_vr==0).sum():,}  1s: {(_vr==1).sum():,}")
print(f"🎯 Target check   — 0s: {(_tgt==0).sum():,}  1s: {(_tgt==1).sum():,}")
print(f"   Overall spike rate on Valid_Row bars: {(_tgt[_vr==1]==1).mean():.1%}  (threshold: {SPIKE_THRESHOLD:.0%})")
print(f"   Date range: {df_scaled_all.index.min()} → {df_scaled_all.index.max()}")
for t in TRAIN_TEACHERS:
    mask_vr = (df_scaled_all['Ticker'] == t) & (df_scaled_all['Valid_Row'] == 1)
    sr = df_scaled_all.loc[mask_vr, 'Target'].mean()
    n_val = (df_scaled_all.loc[mask_vr].index > aware_cutoff).sum()
    print(f"   {t:<6}: valid_rows={mask_vr.sum():,}  spike_rate={sr:.1%}  val_rows={n_val:,}")

print(f"\n🏷️  Asset Embedding Index:")
for t, idx in ASSET_INDEX.items():
    print(f"   [{idx}] {t}")
print(f"   [{OTHER_IDX}] OTHER (out-of-training tickers)")

# Sample weight diagnostics
print(f"\n⚖️  Sample weight distribution (train positives only, cap={SAMPLE_WEIGHT_CAP}x):")
for t in TRAIN_TEACHERS:
    mask_pos = (df_scaled_all['Ticker'] == t) & (df_scaled_all['Valid_Row'] == 1) & \
               (df_scaled_all['Target'] == 1) & (df_scaled_all.index <= aware_cutoff)
    if mask_pos.sum() > 0:
        sw = df_scaled_all.loc[mask_pos, 'Sample_Weight']
        print(f"   {t:<6}: n_pos={mask_pos.sum():,}  avg_weight={sw.mean():.2f}  "
              f"max_weight={sw.max():.2f}  pct_at_cap={(sw >= SAMPLE_WEIGHT_CAP).mean():.1%}")

# ==============================================================================
# 🚀 6. TRAINING
# ==============================================================================
for seed in SEEDS_TO_RUN:
    print(f"\n🧬 STARTING SEED {seed}")
    tf.keras.backend.clear_session()
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

    print(f"   Building training dataset (natural distribution — balanced=False)...")
    ds_tr, st_tr = make_per_ticker_ds(
        df_scaled_all, TRAIN_TEACHERS,
        start_dt=None, end_dt=aware_cutoff,
        steps=TIME_STEPS, batch_size=GLOBAL_BATCH_SIZE,
        balanced=False, use_weights=True
    )
    print(f"   Building validation dataset...")
    ds_v, st_v = make_per_ticker_ds(
        df_scaled_all, TRAIN_TEACHERS,
        start_dt=aware_cutoff, end_dt=None,
        steps=TIME_STEPS, batch_size=GLOBAL_BATCH_SIZE, balanced=False,
        use_weights=False
    )

    if ds_tr is None or st_tr == 0:
        print(f"   ❌ Seed {seed}: no valid training samples found. Skipping.")
        continue

    print(f"   Steps/epoch: {st_tr}  Val steps: {st_v if st_v else 'N/A'}")

    local_model_path = os.path.join(LOCAL_MODEL_DIR, f"UNIVERSAL_S{seed}_M{MODEL_ID}.h5")

    with strategy.scope():
        model = build_universal_model()

        try:
            optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-4)
        except AttributeError:
            print("   ⚠️ AdamW not available, falling back to Adam")
            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

        model.compile(
            optimizer=optimizer,
            loss=asymmetric_loss(),
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.AUC(curve='PR', name='pr_auc'),
                PrecisionAtTopK(k=PREC_TOP_K, name='prec_top2pct'),
            ]
        )
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            local_model_path, monitor='val_pr_auc', save_best_only=True, mode='max', verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_pr_auc', patience=6, restore_best_weights=True, mode='max', verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_pr_auc', factor=0.5, patience=3, min_lr=1e-6, mode='max', verbose=1
        ),
    ]

    model.fit(
        ds_tr, validation_data=ds_v,
        epochs=50, steps_per_epoch=st_tr, validation_steps=st_v,
        callbacks=callbacks, verbose=1
    )

    gcs_model_blob = f"{GCS_MODEL_PATH}/UNIVERSAL_S{seed}_M{MODEL_ID}.h5"
    if os.path.exists(local_model_path):
        bucket.blob(gcs_model_blob).upload_from_filename(local_model_path)
        print(f"   ☁️  UPLOADED MODEL TO: {gcs_model_blob}")
    else:
        print("   ❌ Model file not found locally, upload skipped.")

    print(f"\n🌍 SCORING SEED {seed}...")
    try:
        model.load_weights(local_model_path)
    except Exception as e:
        print(f"   ⚠️ Could not load best weights ({e}), using current weights.")

    for score_ticker in SCORING_POOL:
        print(f"   🎯 Scoring {score_ticker}...")

        if score_ticker in STATS_POOL:
            df_target_scaled = df_scaled_all[df_scaled_all['Ticker'] == score_ticker]
        else:
            df_target = engineer_asset(score_ticker, market_df)
            if df_target is None: continue
            df_target_scaled = apply_scaling(df_target, stats)

        test_mask = df_target_scaled.index > aware_cutoff
        df_test   = df_target_scaled[test_mask]

        idx_v = np.where(df_test['Valid_Row'].values)[0]
        idx_v = idx_v[idx_v >= TIME_STEPS]
        if len(idx_v) == 0: continue

        XA_v = df_test[ASSET_COLS].values
        XC_v = df_test[CONTEXT_COLS].values
        asset_id_val = np.int32(ASSET_INDEX.get(score_ticker, OTHER_IDX))

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

        gcs_csv_path = f"{GCS_RESULTS_PATH}/{score_ticker}/Seed_{seed}_Results.csv"
        bucket.blob(gcs_csv_path).upload_from_filename(local_csv_path)
        print(f"      ✅ Uploaded results to {gcs_csv_path}")

print(f"\n🚀 EXPERIMENT {MODEL_ID} FULLY COMPLETE.")
