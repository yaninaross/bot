"""
Exp-111 — Crypto Spike Model with Regime Awareness (successor to Exp-110)
==========================================================================
Key changes from Exp-110:

  REGIME FEATURES (the model can now "see" multi-day market context):
  - Trailing spike rate (5d, 10d, 20d): how often has this ticker spiked recently?
  - Daily ROC (5d, 20d): multi-day price trend for each ticker
  - Realized volatility (20d): trailing daily return std
  - BTC daily ROC (5d, 20d): crypto market multi-day trend (context)
  - VIX absolute level: raw VIX reading, not just intraday ROC (context)

  LOSS FUNCTION FIXES (let the model learn to say "no"):
  - ASL gamma_neg: 4.0 -> 2.0 (more gradient signal for easy negatives)
  - ASL clip: 0.05 -> 0.0 (model can learn to output near-zero probabilities)
  - Sample weight cap: 5.0 -> 3.0 (less over-emphasis on extreme spikes)

  WHY: Exp-110 produced false positives during Jan-Mar 2026 crypto bear market.
  The model had no features capturing multi-day trends, and ASL loss prevented
  it from learning strong "no spike" convictions. These changes address both.

  Everything else unchanged: same tickers, same architecture, same data split,
  same CONTEXT_ASSETS, same SPIKE_THRESHOLD, same healing/resample pipeline.
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

# ==============================================================================
# ENVIRONMENT SYNC & AUTH
# ==============================================================================
SCRIPT_VERSION = "111.1"
print(f"Script version: {SCRIPT_VERSION}")
print("Syncing environment for Experiment 111 (Crypto — regime-aware)...")

try:
    import tensorflow as tf
    from google.protobuf import descriptor
    if descriptor._USE_C_DESCRIPTORS == False: raise ImportError
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "protobuf<4.24.0", "tensorflow[and-cuda]", "pandas_ta",
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

strategy = tf.distribute.get_strategy()
print(f"Single-device strategy (using GPU:0)")

NUM_REPLICAS = strategy.num_replicas_in_sync

# ==============================================================================
# COLUMN DEFINITIONS
# ==============================================================================
MICRO_COLS = ['Spread', 'OBI', 'Volume_Delta', 'Trade_Count', 'Delta_Pct',
              'bsize', 'asize', 'Total_Vol']

ROC_HORIZONS = [5, 30, 60, 120]

# --- NEW in Exp-111: 6 regime features added to per-ticker columns ---
REGIME_ASSET_COLS = [
    'Spike_Rate_5d', 'Spike_Rate_10d', 'Spike_Rate_20d',
    'Daily_ROC_5d', 'Daily_ROC_20d',
    'Realized_Vol_20d',
]

ASSET_COLS = (['Gap_Pct', 'RSI_14', 'RSI_60', 'RVOL', 'Compression',
               'ATR_14', 'Dist_to_PM_High', 'Bar_Is_Real'] +
              [f'ROC_{h}m' for h in ROC_HORIZONS] + MICRO_COLS +
              REGIME_ASSET_COLS)

TIME_STEPS = 240
EMBED_DIM  = 16
PST = pytz.timezone('America/Los_Angeles')
TICKER_FILE_MAP = {'BITF': 'KEEL'}

# ==============================================================================
# CONFIGURATION
# ==============================================================================
MODEL_ID = "111_MICRO"
SEEDS_TO_RUN = [11101, 11102, 11103, 11104, 11105, 11106, 11107,
                11108, 11109, 11110, 11111, 11112, 11113, 11114, 11115]

TRAIN_TEACHERS = [
    'MARA', 'IREN', 'WULF', 'APLD', 'RIOT', 'CLSK', 'CORZ', 'MSTR', 'COIN',
    'BITF', 'HUT', 'CIFR', 'BTBT', 'HOOD',
]

TICKER_START_DATES = {
    'CORZ': '2024-01-25',
}

STATS_POOL   = TRAIN_TEACHERS
SCORING_POOL = TRAIN_TEACHERS + [
    'SMCI', 'NVDA', 'VRT', 'AMD', 'PLTR', 'TSLA',
    'SMH', 'TLT', 'IWM', 'VIX',
]

CONTEXT_ASSETS = ['SMH', 'MOD', 'TLT', 'IWM', 'VIX', 'BTCUSD']
HAS_SMH_RSI    = True

# --- Base context columns (same as Exp-110) ---
CONTEXT_COLS = []
for _t in CONTEXT_ASSETS:
    CONTEXT_COLS.append(f'{_t}_Anchor_ROC')
    for _h in ROC_HORIZONS:
        CONTEXT_COLS.append(f'{_t}_ROC_{_h}m')
CONTEXT_COLS += ['SMH_RSI', 'Time_Pos']

# --- NEW in Exp-111: 3 regime context features ---
REGIME_CONTEXT_COLS = ['BTCUSD_Daily_ROC_5d', 'BTCUSD_Daily_ROC_20d', 'VIX_Level']
CONTEXT_COLS += REGIME_CONTEXT_COLS

ALL_COLS = ASSET_COLS + CONTEXT_COLS

DATA_START_DATE   = "2022-01-01"
TRAIN_START_DATE  = "2022-02-01"
TRAIN_CUTOFF_DATE = "2025-10-01"

SPIKE_THRESHOLD   = 0.10

# --- CHANGED in Exp-111: ASL and sample weight tuning ---
ASL_GAMMA_NEG     = 2.0     # was 4.0 — more gradient on easy negatives
ASL_GAMMA_POS     = 0.5     # unchanged
ASL_CLIP          = 0.0     # was 0.05 — allow near-zero probability output
SAMPLE_WEIGHT_CAP = 3.0     # was 5.0 — less over-emphasis on extreme spikes

PREC_TOP_K        = 0.02

BUCKET_NAME   = os.environ.get("GCS_BUCKET", "")
INFERENCE_DIR = "Data/Inference_v2"

LOCAL_RESULTS_DIR = "Results"
LOCAL_MODEL_DIR   = "Models"
os.makedirs(LOCAL_RESULTS_DIR, exist_ok=True)
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

GCS_RESULTS_PATH = f"Algos/Results_all_v2/Universal_Exp{MODEL_ID}"
GCS_MODEL_PATH   = f"Algos/to_deploy/Saved_models_v2/UNIVERSAL/Direction"
STATS_DIR        = f"Data/Inference_v2"

PER_REPLICA_BATCH = 256
GLOBAL_BATCH_SIZE = PER_REPLICA_BATCH * NUM_REPLICAS

ASSET_INDEX = {ticker: i for i, ticker in enumerate(TRAIN_TEACHERS)}
N_ASSETS    = len(TRAIN_TEACHERS) + 1   # 15
OTHER_IDX   = len(TRAIN_TEACHERS)       # 14

print(f"\nConfig: {len(TRAIN_TEACHERS)} teachers, {N_ASSETS} embeddings, "
      f"batch={PER_REPLICA_BATCH}x{NUM_REPLICAS}={GLOBAL_BATCH_SIZE}")
print(f"ASSET_COLS:   {len(ASSET_COLS)} features ({len(REGIME_ASSET_COLS)} new regime)")
print(f"CONTEXT_COLS: {len(CONTEXT_COLS)} features ({len(REGIME_CONTEXT_COLS)} new regime)")
print(f"ASL: gamma_neg={ASL_GAMMA_NEG}, gamma_pos={ASL_GAMMA_POS}, clip={ASL_CLIP}")
print(f"Sample weight cap: {SAMPLE_WEIGHT_CAP}")

# ==============================================================================
# DATA LOADING
# ==============================================================================
def load_master(ticker, tail_days=None, add_micro_zeros=True):
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
    for c in ['Spread', 'OBI', 'bsize', 'asize']:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    for c in ['Volume', 'Total_Vol', 'Volume_Delta', 'Trade_Count', 'Delta_Pct']:
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
        if ticker == 'SMH':
            if len(df) >= 14:
                rsi = ta.rsi(df['Close'], length=14)
                cols['SMH_RSI'] = (rsi.fillna(50) / 100.0
                                   if rsi is not None
                                   else pd.Series(0.5, index=df.index))
            else:
                cols['SMH_RSI'] = pd.Series(0.5, index=df.index)

        # --- NEW: BTC multi-day trend ---
        if ticker == 'BTCUSD':
            dc = df['Close'].resample('D').last().ffill()
            btc_5d  = dc.pct_change(5).shift(1).reindex(df.index, method='ffill').fillna(0)
            btc_20d = dc.pct_change(20).shift(1).reindex(df.index, method='ffill').fillna(0)
            cols['BTCUSD_Daily_ROC_5d']  = btc_5d
            cols['BTCUSD_Daily_ROC_20d'] = btc_20d

        # --- NEW: absolute VIX level ---
        if ticker == 'VIX':
            cols['VIX_Level'] = df['Close']

        frames.append(pd.DataFrame(cols, index=df.index))
        del df, daily_c
        gc.collect()
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

    # ------------------------------------------------------------------
    # NEW: Regime features (daily-scale, broadcast to minute bars)
    # ------------------------------------------------------------------
    dc = df['Close'].resample('D').last().ffill()
    dh = df['High'].resample('D').max()
    dl = df['Low'].resample('D').min()

    # Multi-day price trend (shift(1) = use yesterday's close, no look-ahead)
    roc_5d  = dc.pct_change(5).shift(1)
    roc_20d = dc.pct_change(20).shift(1)
    df['Daily_ROC_5d']  = roc_5d.reindex(df.index, method='ffill').fillna(0)
    df['Daily_ROC_20d'] = roc_20d.reindex(df.index, method='ffill').fillna(0)

    # Realized daily volatility (20-day rolling std of daily returns)
    daily_ret = dc.pct_change()
    rvol_20d = daily_ret.rolling(20, min_periods=5).std().shift(1)
    df['Realized_Vol_20d'] = rvol_20d.reindex(df.index, method='ffill').fillna(0)

    # Trailing spike rate: fraction of recent days with 10%+ intraday rally
    # Spike day = (daily high - daily low) / daily low > SPIKE_THRESHOLD
    daily_range = (dh - dl) / (dl + 1e-9)
    is_spike_day = (daily_range > SPIKE_THRESHOLD).astype(float)
    sr_5d  = is_spike_day.rolling(5,  min_periods=1).mean().shift(1)
    sr_10d = is_spike_day.rolling(10, min_periods=1).mean().shift(1)
    sr_20d = is_spike_day.rolling(20, min_periods=1).mean().shift(1)
    df['Spike_Rate_5d']  = sr_5d.reindex(df.index, method='ffill').fillna(0)
    df['Spike_Rate_10d'] = sr_10d.reindex(df.index, method='ffill').fillna(0)
    df['Spike_Rate_20d'] = sr_20d.reindex(df.index, method='ffill').fillna(0)
    # ------------------------------------------------------------------

    df['Bar_Is_Real'] = (df['Volume'] > 0).astype(float)

    df = df.join(market_df, how='left').ffill().fillna(0)
    df_pst = df.index.tz_convert(PST)
    mins = df_pst.hour * 60 + df_pst.minute
    df['Time_Pos'] = np.clip((mins - 390) / 390, 0, 1)
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
# TARGET COMPUTATION (unchanged from Exp-110)
# ==============================================================================
def add_targets(df):
    df_pst = df.index.tz_convert(PST)
    mins = df_pst.hour * 60 + df_pst.minute
    is_safe = (df_pst.hour < 13) | ((df_pst.hour == 13) & (df_pst.minute == 0))
    is_real = df['Volume'] > 0
    df['Safe_High'] = np.where(is_safe & is_real, df['High'], -1.0)
    df['Future_Max'] = df.groupby(df.index.date)['Safe_High'].transform(
        lambda x: x[::-1].cummax()[::-1])
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
# MODEL ARCHITECTURE (unchanged — input dims auto-adapt)
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
# ASL LOSS — CHANGED: gamma_neg=2.0, clip=0.0
# ==============================================================================
def asymmetric_loss(gamma_neg=ASL_GAMMA_NEG, gamma_pos=ASL_GAMMA_POS, clip=ASL_CLIP):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        loss_pos = y_true * tf.pow(1.0 - y_pred, gamma_pos) * tf.math.log(y_pred)
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
# DATASET BUILDER (unchanged from Exp-110)
# ==============================================================================
def make_per_ticker_ds(teacher_data, tickers, start_dt, end_dt, steps, batch_size,
                       balanced=True, use_weights=True):
    ticker_arrays = []
    for ticker in tickers:
        if ticker not in teacher_data:
            continue
        td = teacher_data[ticker]
        dates = td['dates']
        date_mask = np.ones(len(dates), dtype=bool)
        if start_dt is not None:
            date_mask &= dates > start_dt
        if end_dt is not None:
            date_mask &= dates <= end_dt
        if not date_mask.any():
            continue
        first = int(np.argmax(date_mask))
        last = len(date_mask) - int(np.argmax(date_mask[::-1]))
        XA = td['XA'][first:last]
        XC = td['XC'][first:last]
        Y  = td['Y'][first:last]
        SW = td['SW'][first:last]
        VR = td['VR'][first:last]
        if len(XA) < steps:
            continue
        vi = np.where(VR)[0]
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
        dtype=np.int64)

    def gen():
        while True:
            perm = np.random.permutation(len(combined))
            for p in perm:
                t_idx, i = combined[p]
                XA, XC, Y, SW, _, asset_id_val = ticker_arrays[t_idx]
                w = SW[i] if use_weights else np.float32(1.0)
                yield (
                    {'asset_in': XA[i - steps:i],
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
# EXECUTION PIPELINE — Memory-efficient two-pass approach
# ==============================================================================
market_df = build_market_context()
aware_cutoff = pd.to_datetime(TRAIN_CUTOFF_DATE).tz_localize('UTC')

# --- Pass 1: Compute scaling stats (only Valid_Row==1 rows kept in memory) ---
print("\nPass 1: Computing scaling stats...")
valid_rows_for_stats = []
for ticker in STATS_POOL:
    print(f"   Loading {ticker} for stats...")
    df_temp = engineer_asset(ticker, market_df)
    if df_temp is None:
        continue
    df_temp = df_temp.loc[TRAIN_START_DATE:]
    valid_mask = (df_temp['Valid_Row'] == 1) & (df_temp.index <= aware_cutoff)
    stats_cols = [c for c in ALL_COLS if c != 'Time_Pos']
    valid_rows_for_stats.append(df_temp.loc[valid_mask, stats_cols].copy())
    n_valid = int(valid_mask.sum())
    print(f"      rows={len(df_temp):,}  valid_for_stats={n_valid:,}")
    del df_temp; gc.collect()

stats_df = pd.concat(valid_rows_for_stats)
del valid_rows_for_stats; gc.collect()
print(f"   Stats pool: {len(stats_df):,} Valid_Row==1 rows (pre-cutoff)")

stats = {col: {'med': float(stats_df[col].median()),
               'iqr': float(max(1e-6, stats_df[col].quantile(0.75) -
                                       stats_df[col].quantile(0.25)))}
         for col in ALL_COLS if col != 'Time_Pos'}

blob = client.bucket(BUCKET_NAME).blob(
    f"{STATS_DIR}/scaling_stats_experiment_{MODEL_ID}.json")
blob.upload_from_string(json.dumps(stats, indent=4), content_type='application/json')
print(f"SAVED STATS TO: {STATS_DIR}/scaling_stats_experiment_{MODEL_ID}.json")

# --- Regime feature stats sanity check ---
print("\nRegime feature stats (pre-scaling):")
for col in REGIME_ASSET_COLS + REGIME_CONTEXT_COLS:
    if col in stats:
        print(f"   {col:<22}: med={stats[col]['med']:.4f}  iqr={stats[col]['iqr']:.4f}")

del stats_df; gc.collect()

# --- Pass 2: Load, scale, extract numpy arrays per ticker ---
print("\nPass 2: Loading and scaling teacher data...")
teacher_data = {}
total_rows = 0
total_valid = 0
for ticker in TRAIN_TEACHERS:
    print(f"   Loading {ticker}...")
    df_temp = engineer_asset(ticker, market_df)
    if df_temp is None:
        continue
    df_temp = df_temp.loc[TRAIN_START_DATE:]
    df_temp = apply_scaling(df_temp, stats, inplace=True)
    teacher_data[ticker] = {
        'XA': df_temp[ASSET_COLS].values,
        'XC': df_temp[CONTEXT_COLS].values,
        'Y': df_temp['Target'].values,
        'SW': df_temp['Sample_Weight'].values.astype('float32'),
        'VR': df_temp['Valid_Row'].values,
        'dates': df_temp.index,
    }
    n_rows = len(df_temp)
    n_valid = int(df_temp['Valid_Row'].sum())
    vr_mask = np.asarray(df_temp['Valid_Row'] == 1)
    date_mask = np.asarray(df_temp.index > aware_cutoff)
    n_val_rows = int((vr_mask & date_mask).sum()) if n_valid > 0 else 0
    spike_rate = df_temp.loc[df_temp['Valid_Row'] == 1, 'Target'].mean()
    total_rows += n_rows
    total_valid += n_valid
    print(f"      rows={n_rows:,}  valid={n_valid:,}  spike_rate={spike_rate:.1%}  val_rows={n_val_rows:,}")
    del df_temp; gc.collect()

print(f"\nTeacher data loaded: {total_rows:,} total rows, {total_valid:,} valid rows")

# --- Diagnostics ---
print(f"\nAsset Embedding Index:")
for t, idx in ASSET_INDEX.items():
    print(f"   [{idx}] {t}")
print(f"   [{OTHER_IDX}] OTHER (out-of-training tickers)")

# ==============================================================================
# TRAINING — model created inside strategy.scope() for multi-GPU
# ==============================================================================
# ==============================================================================
# PHASE 1: TRAIN ALL SEEDS (teacher_data in memory)
# ==============================================================================
trained_seeds = []
for seed in SEEDS_TO_RUN:
    print(f"\nSTARTING SEED {seed}")
    tf.keras.backend.clear_session()
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

    print(f"   Building training dataset (balanced=False, global_batch={GLOBAL_BATCH_SIZE})...")
    ds_tr, st_tr = make_per_ticker_ds(
        teacher_data, TRAIN_TEACHERS,
        start_dt=None, end_dt=aware_cutoff,
        steps=TIME_STEPS, batch_size=GLOBAL_BATCH_SIZE,
        balanced=False, use_weights=True)
    print(f"   Building validation dataset...")
    ds_v, st_v = make_per_ticker_ds(
        teacher_data, TRAIN_TEACHERS,
        start_dt=aware_cutoff, end_dt=None,
        steps=TIME_STEPS, batch_size=GLOBAL_BATCH_SIZE,
        balanced=False, use_weights=False)

    if ds_tr is None or st_tr == 0:
        print(f"   Seed {seed}: no valid training samples found. Skipping.")
        continue

    print(f"   Steps/epoch: {st_tr}  Val steps: {st_v if st_v else 'N/A'}")

    local_model_path = os.path.join(LOCAL_MODEL_DIR, f"UNIVERSAL_S{seed}_M{MODEL_ID}.h5")

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

    gcs_model_blob = f"{GCS_MODEL_PATH}/UNIVERSAL_S{seed}_M{MODEL_ID}.h5"
    if os.path.exists(local_model_path):
        bucket.blob(gcs_model_blob).upload_from_filename(local_model_path)
        print(f"   UPLOADED MODEL TO: {gcs_model_blob}")
    else:
        print("   Model file not found locally, upload skipped.")

    trained_seeds.append(seed)
    del ds_tr, ds_v, model; gc.collect()

# ==============================================================================
# FREE TEACHER DATA (~13 GB) before scoring
# ==============================================================================
del teacher_data; gc.collect()
print(f"\n{'='*60}")
print(f"TRAINING COMPLETE: {len(trained_seeds)}/{len(SEEDS_TO_RUN)} seeds trained")
print(f"Teacher data freed — starting scoring phase")
print(f"{'='*60}")

# ==============================================================================
# PHASE 2: SCORE ALL SEEDS (loads model + tickers one at a time)
# ==============================================================================
for seed in trained_seeds:
    print(f"\nSCORING SEED {seed}...")
    tf.keras.backend.clear_session()

    local_model_path = os.path.join(LOCAL_MODEL_DIR, f"UNIVERSAL_S{seed}_M{MODEL_ID}.h5")
    with strategy.scope():
        model = build_universal_model()
    try:
        model.load_weights(local_model_path)
    except Exception as e:
        print(f"   Could not load weights ({e}), skipping seed.")
        continue

    for score_ticker in SCORING_POOL:
        print(f"   Scoring {score_ticker}...")

        df_target = engineer_asset(score_ticker, market_df)
        if df_target is None:
            continue
        df_target_scaled = apply_scaling(df_target, stats)

        test_mask = df_target_scaled.index > aware_cutoff
        df_test   = df_target_scaled[test_mask]

        idx_v = np.where(df_test['Valid_Row'].values)[0]
        idx_v = idx_v[idx_v >= TIME_STEPS]
        if len(idx_v) == 0:
            del df_target, df_target_scaled, df_test; gc.collect()
            continue

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
        print(f"      Uploaded results to {gcs_csv_path}")

        del df_target, df_target_scaled, df_test, results_df; gc.collect()


    del model; gc.collect()

print(f"\nEXPERIMENT {MODEL_ID} FULLY COMPLETE.")
