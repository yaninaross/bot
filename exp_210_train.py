"""
Exp 210 — Dip-Then-Rip Daily Classifier (Single Fold)
=====================================================
Predicts whether a stock will dip >= DIP_PCT in the first 60 minutes
after open AND close above the dip entry price by EOD.

Completely different from exp 185/200 (sequence models on minute bars):
  - Daily features (34 flat features per day, not 240-step sequences)
  - MLP architecture (~16K params, not Transformer/CNN)
  - Opening dynamics target (dip-then-rip, not spike probability)

Teachers: 35 tickers with amplitude >= 4%, fill@1% >= 75%, >= 500 days.
Target: binary — dip >= 1% in opening window AND close > entry price by EOD.
"""

import sys
import subprocess
import os
import json
import random
import gc
import ctypes

import numpy as np
import pandas as pd
import pytz

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
SCRIPT_VERSION = "210.1"
print(f"Script version: {SCRIPT_VERSION}")
print("Syncing environment for Experiment 210 (Dip-Then-Rip Daily Classifier)...")

try:
    import tensorflow as tf
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "tensorflow[and-cuda]",
                           "google-cloud-storage", "pyarrow", "gcsfs", "--quiet"])
    import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, BatchNormalization,
    Concatenate, Embedding, Flatten,
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
bucket = client.bucket("time-series-bucket-ys")

# ==============================================================================
# GPU SETUP
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
# CONFIGURATION
# ==============================================================================
FOLD_EXP_ID   = '210_DIPRIP'
FOLD_CUTOFF   = '2026-01-01'
FOLD_SEEDS    = [21001 + i for i in range(15)]

DIP_PCT              = 0.01     # 1% dip threshold for limit entry
SWING_WINDOW_MINUTES = 60       # first 60 min after open
MARKET_OPEN_MINUTE   = 390      # 6:30 AM PST
EOD_MINUTE           = 780      # 1:00 PM PST

PST = pytz.timezone('America/Los_Angeles')
TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

DATA_START_DATE  = "2022-01-01"
TRAIN_START_DATE = "2022-03-01"   # 2-month warmup for rolling features

TRAIN_TEACHERS = [
    # Crypto (10)
    'WULF', 'CIFR', 'IREN', 'APLD', 'CLSK', 'HUT', 'KEEL', 'BTBT', 'MARA', 'RIOT',
    # Defense / space (7)
    'LUNR', 'BBAI', 'BKSY', 'RCAT', 'MNTS', 'SPIR', 'RDW',
    # Energy (3)
    'FCEL', 'PLUG', 'FLNC',
    # Nuclear (2)
    'OKLO', 'SMR',
    # Tech (2)
    'ALAB', 'MVIS',
    # Space / eVTOL (2)
    'ASTS', 'ACHR',
    # Growth / other (9)
    'BGDE', 'RGTI', 'SOUN', 'SIDU', 'LAZR', 'CORZ', 'OUST', 'IONQ', 'UPST',
]

SCORING_EXTRAS = [
    'JOBY', 'MP', 'SMCI', 'COIN', 'MSTR', 'HOOD',
    'POWL', 'NNDM', 'KTOS', 'NBIS', 'AVAV', 'RKLB', 'PL',
]
SCORING_POOL = TRAIN_TEACHERS + [t for t in SCORING_EXTRAS if t not in TRAIN_TEACHERS]

CONTEXT_TICKERS = ['SPY', 'VIX', 'TLT']

FEATURE_COLS = [
    # Group 1: Gap (4)
    'gap_pct', 'gap_abs', 'gap_vs_20d', 'gap_vs_atr',
    # Group 2: Prior day candle (6)
    'prev_return', 'prev_range_pct', 'prev_upper_wick', 'prev_lower_wick',
    'prev_close_position', 'prev_volume_ratio',
    # Group 3: Prior opening dynamics (3)
    'prev_max_dd', 'prev_max_ru', 'prev_dip_first',
    # Group 4: Volatility regime (4)
    'vol_5d', 'vol_20d', 'vol_ratio', 'atr_20d_pct',
    # Group 5: Momentum (4)
    'return_5d', 'return_20d', 'rsi_14d', 'dist_from_20d_high',
    # Group 6: Pattern history (4)
    'fill_rate_20d', 'recovery_rate_20d', 'dip_first_rate_20d', 'consec_dip_days',
    # Group 7: Seasonality (4)
    'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
    # Group 8: Market context (5)
    'spy_prev_return', 'spy_5d_return', 'vix_close_norm', 'vix_change', 'tlt_prev_return',
]
N_FEATURES = len(FEATURE_COLS)   # 34

# Model
EMBED_DIM     = 16
HIDDEN_DIMS   = [128, 64, 32]
DROPOUT_RATES = [0.3, 0.2, 0.1]
LEARNING_RATE = 1e-3
EPOCHS        = 100
PATIENCE      = 15

# ASL (gentler gammas — ~33% positive rate vs 10% for spikes)
ASL_GAMMA_NEG     = 2.0
ASL_GAMMA_POS     = 0.5
ASL_CLIP          = 0.02
LABEL_SMOOTH      = 0.03
SAMPLE_WEIGHT_CAP = 3.0
PREC_TOP_K        = 0.05

# GCS
BUCKET_NAME   = "time-series-bucket-ys"
OHLCV_DIR     = "Data/Inference"
GCS_MODEL_PATH = "Algos/to_deploy/Saved_models_v2/UNIVERSAL/Direction"
STATS_DIR     = "Data/Inference_v2"

LOCAL_RESULTS_DIR = "Results"
LOCAL_MODEL_DIR   = "Models"
os.makedirs(LOCAL_RESULTS_DIR, exist_ok=True)
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

ASSET_INDEX = {ticker: i for i, ticker in enumerate(TRAIN_TEACHERS)}
N_ASSETS    = len(TRAIN_TEACHERS) + 1
OTHER_IDX   = len(TRAIN_TEACHERS)

PER_REPLICA_BATCH = 256
GLOBAL_BATCH_SIZE = PER_REPLICA_BATCH * NUM_REPLICAS

print(f"\nConfig: {len(TRAIN_TEACHERS)} teachers, {N_FEATURES} features, "
      f"DIP_PCT={DIP_PCT:.1%}, MLP hidden={HIDDEN_DIMS}")
print(f"Fold: {FOLD_EXP_ID}  cutoff={FOLD_CUTOFF}  seeds={FOLD_SEEDS[0]}-{FOLD_SEEDS[-1]}")
print(f"ASL: gamma_neg={ASL_GAMMA_NEG}, gamma_pos={ASL_GAMMA_POS}, "
      f"clip={ASL_CLIP}, label_smooth={LABEL_SMOOTH}")

# ==============================================================================
# DATA LOADING
# ==============================================================================
def load_ohlcv(ticker):
    clean = ticker.replace("I:", "").replace("X:", "")
    file_ticker = TICKER_FILE_MAP.get(clean, clean)

    for suffix in ['_1m_full_historical_data', '_master_inference']:
        uri = f"gs://{BUCKET_NAME}/{OHLCV_DIR}/{file_ticker}{suffix}.parquet"
        try:
            df = pd.read_parquet(uri)
            break
        except Exception:
            continue
    else:
        print(f"   Failed to load {ticker}: no data found")
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
        std_map = {'open': 'Open', 'high': 'High', 'low': 'Low',
                   'close': 'Close', 'volume': 'Volume'}
        lower_map = {k.lower(): v for k, v in std_map.items()}
        df.columns = [lower_map.get(str(c).strip().lower(), str(c).strip())
                      for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated(keep='first')].copy()
        df = df[df.index.notna()].sort_index()
        cutoff = pd.Timestamp(DATA_START_DATE, tz='UTC')
        df = df[df.index >= cutoff]
        keep = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in df.columns]
        return df[keep]
    except Exception as e:
        print(f"   Failed to load {ticker}: {e}")
        return None


def heal_data(df):
    df = df.resample('1min').asfreq()
    for c in ['Open', 'High', 'Low', 'Close']:
        if c in df.columns:
            df[c] = df[c].ffill()
    for c in ['Volume']:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    return df


# ==============================================================================
# DAILY RECORDS — 1-min bars → one row per trading day
# ==============================================================================
def compute_daily_records(df_1min):
    df_pst = df_1min.copy()
    df_pst.index = df_pst.index.tz_convert(PST)
    mins = df_pst.index.hour * 60 + df_pst.index.minute

    is_market = (mins >= MARKET_OPEN_MINUTE) & (mins <= EOD_MINUTE)
    is_real   = df_pst['Volume'] > 0
    market    = df_pst[is_market & is_real].copy()

    is_window = (mins >= MARKET_OPEN_MINUTE) & (mins <= MARKET_OPEN_MINUTE + SWING_WINDOW_MINUTES)
    window    = df_pst[is_window & is_real].copy()

    is_eod    = mins >= EOD_MINUTE
    eod_bars  = df_pst[is_eod & is_real].copy()
    eod_bars['date'] = eod_bars.index.date
    eod_closes = eod_bars.groupby('date')['Close'].first()

    market['date'] = market.index.date
    window['date'] = window.index.date

    records = []
    for date, day_market in market.groupby('date'):
        if len(day_market) < 5:
            continue
        day_window = window[window['date'] == date]
        if len(day_window) < 3:
            continue

        open_price = float(day_market.iloc[0]['Close'])
        if open_price <= 0:
            continue

        daily_high   = float(day_market['High'].max())
        daily_low    = float(day_market['Low'].min())
        daily_close  = float(eod_closes.get(date, day_market.iloc[-1]['Close']))
        daily_volume = float(day_market['Volume'].sum())

        window_min_low  = float(day_window['Low'].min())
        window_max_high = float(day_window['High'].max())
        max_dd_pct = (window_min_low - open_price) / open_price
        max_ru_pct = (window_max_high - open_price) / open_price

        trough_idx = day_window['Low'].values.argmin()
        peak_idx   = day_window['High'].values.argmax()
        dip_first  = float(trough_idx < peak_idx)

        entry_price = open_price * (1 - DIP_PCT)
        filled      = window_min_low <= entry_price
        profitable  = daily_close > entry_price
        target      = 1 if (filled and profitable) else 0
        pnl_if_filled = (daily_close - entry_price) / entry_price if filled else 0.0

        records.append({
            'date': pd.Timestamp(date),
            'open': open_price, 'high': daily_high,
            'low': daily_low,   'close': daily_close,
            'volume': daily_volume,
            'max_dd_pct': max_dd_pct, 'max_ru_pct': max_ru_pct,
            'dip_first': dip_first,   'filled': float(filled),
            'target': target,         'pnl_if_filled': pnl_if_filled,
        })

    if not records:
        return None
    df_daily = pd.DataFrame(records).set_index('date')
    df_daily.index = pd.DatetimeIndex(df_daily.index)
    return df_daily


# ==============================================================================
# MARKET CONTEXT — daily SPY / VIX / TLT features
# ==============================================================================
def build_market_context_daily():
    print("   Building daily market context (SPY, VIX, TLT)...")
    ctx = pd.DataFrame()

    for ticker in CONTEXT_TICKERS:
        print(f"      Loading {ticker}...")
        df = load_ohlcv(ticker)
        if df is None:
            print(f"      Could not load {ticker}")
            continue
        df = heal_data(df)

        df_pst = df.copy()
        df_pst.index = df_pst.index.tz_convert(PST)
        mins = df_pst.index.hour * 60 + df_pst.index.minute
        mkt = df_pst[(mins >= MARKET_OPEN_MINUTE) & (mins <= EOD_MINUTE) &
                     (df_pst['Volume'] > 0)].copy()
        mkt['date'] = mkt.index.date
        daily = mkt.groupby('date').agg(
            open=('Close', 'first'), high=('High', 'max'),
            low=('Low', 'min'),     close=('Close', 'last'),
        )
        daily.index = pd.DatetimeIndex(daily.index)
        daily = daily.sort_index()

        if ticker == 'SPY':
            ctx['spy_prev_return'] = daily['close'].pct_change().shift(1)
            ctx['spy_5d_return']   = daily['close'].pct_change(5).shift(1)
        elif ticker == 'VIX':
            ctx['vix_close_norm'] = daily['close'].shift(1) / 20.0
            ctx['vix_change']     = daily['close'].diff().shift(1) / 20.0
        elif ticker == 'TLT':
            ctx['tlt_prev_return'] = daily['close'].pct_change().shift(1)

        del df, df_pst, mkt, daily
        _release_memory()

    ctx = ctx.fillna(0)
    print(f"      Market context: {len(ctx)} days, cols={list(ctx.columns)}")
    return ctx


# ==============================================================================
# FEATURE ENGINEERING — 34 daily features (no lookahead)
# ==============================================================================
def compute_features(df_daily):
    df = df_daily.copy()
    eps = 1e-6

    # --- Group 1: Gap ---
    prev_close = df['close'].shift(1)
    df['gap_pct']    = (df['open'] - prev_close) / prev_close.clip(lower=0.01)
    df['gap_abs']    = df['gap_pct'].abs()
    avg_gap_20       = df['gap_abs'].rolling(20, min_periods=5).mean().shift(1)
    df['gap_vs_20d'] = df['gap_abs'] / avg_gap_20.clip(lower=eps)
    atr_20d          = (df['high'] - df['low']).rolling(20, min_periods=5).mean().shift(1)
    df['gap_vs_atr'] = df['gap_abs'] / (atr_20d / prev_close.clip(lower=0.01)).clip(lower=eps)

    # --- Group 2: Prior day candle ---
    df['prev_return'] = (df['close'].shift(1) - df['open'].shift(1)) / \
                         df['open'].shift(1).clip(lower=0.01)
    prev_range = (df['high'].shift(1) - df['low'].shift(1)).clip(lower=eps)
    df['prev_range_pct'] = prev_range / df['open'].shift(1).clip(lower=0.01)
    body_top = np.maximum(df['open'].shift(1), df['close'].shift(1))
    body_bot = np.minimum(df['open'].shift(1), df['close'].shift(1))
    df['prev_upper_wick']    = (df['high'].shift(1) - body_top) / prev_range
    df['prev_lower_wick']    = (body_bot - df['low'].shift(1)) / prev_range
    df['prev_close_position'] = (df['close'].shift(1) - df['low'].shift(1)) / \
                                 (df['high'].shift(1) - df['low'].shift(1)).clip(lower=eps)
    vol_avg_20 = df['volume'].rolling(20, min_periods=5).mean().shift(1)
    df['prev_volume_ratio'] = df['volume'].shift(1) / vol_avg_20.clip(lower=1)

    # --- Group 3: Prior opening dynamics ---
    df['prev_max_dd']    = df['max_dd_pct'].shift(1)
    df['prev_max_ru']    = df['max_ru_pct'].shift(1)
    df['prev_dip_first'] = df['dip_first'].shift(1)

    # --- Group 4: Volatility regime ---
    daily_ret  = df['close'].pct_change()
    df['vol_5d']      = daily_ret.rolling(5, min_periods=3).std().shift(1)
    df['vol_20d']     = daily_ret.rolling(20, min_periods=5).std().shift(1)
    df['vol_ratio']   = df['vol_5d'] / df['vol_20d'].clip(lower=eps)
    df['atr_20d_pct'] = atr_20d / df['close'].shift(1).clip(lower=0.01)

    # --- Group 5: Momentum ---
    df['return_5d']  = df['close'].pct_change(5).shift(1)
    df['return_20d'] = df['close'].pct_change(20).shift(1)
    delta = df['close'].diff()
    gain  = delta.clip(lower=0).ewm(span=14, min_periods=5).mean()
    loss  = (-delta.clip(upper=0)).ewm(span=14, min_periods=5).mean()
    rs    = gain / loss.clip(lower=eps)
    df['rsi_14d'] = (100 - (100 / (1 + rs))).shift(1) / 100.0
    high_20d = df['high'].rolling(20, min_periods=5).max().shift(1)
    df['dist_from_20d_high'] = (df['close'].shift(1) - high_20d) / high_20d.clip(lower=0.01)

    # --- Group 6: Pattern history ---
    df['fill_rate_20d'] = df['filled'].rolling(20, min_periods=5).mean().shift(1)
    filled_won  = ((df['filled'] == 1) & (df['target'] == 1)).astype(float)
    filled_sum  = df['filled'].rolling(20, min_periods=5).sum().shift(1)
    df['recovery_rate_20d'] = filled_won.rolling(20, min_periods=5).sum().shift(1) / \
                               filled_sum.clip(lower=1)
    df['dip_first_rate_20d'] = df['dip_first'].rolling(20, min_periods=5).mean().shift(1)
    is_filled = (df['filled'] == 1).astype(int)
    groups = is_filled.ne(is_filled.shift()).cumsum()
    consec = is_filled.groupby(groups).cumsum()
    df['consec_dip_days'] = consec.shift(1).clip(upper=10) / 10.0

    # --- Group 7: Seasonality ---
    df['dow_sin']   = np.sin(2 * np.pi * df.index.dayofweek / 5)
    df['dow_cos']   = np.cos(2 * np.pi * df.index.dayofweek / 5)
    df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)

    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    for c in df.columns:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    return df


# ==============================================================================
# SCALING
# ==============================================================================
SKIP_SCALE = {'dow_sin', 'dow_cos', 'month_sin', 'month_cos', 'prev_dip_first'}


def compute_stats(df_train, cols):
    stats = {}
    for col in cols:
        if col in SKIP_SCALE:
            continue
        med = float(df_train[col].median())
        iqr = float(max(1e-6,
                        df_train[col].quantile(0.75) - df_train[col].quantile(0.25)))
        stats[col] = {'med': med, 'iqr': iqr}
    return stats


def apply_scaling(df, stats, cols):
    df_s = df.copy()
    for col in cols:
        if col in stats:
            df_s[col] = ((df_s[col] - stats[col]['med']) /
                         stats[col]['iqr']).clip(-10, 10).astype('float32')
    return df_s


# ==============================================================================
# MODEL — MLP with ticker embedding
# ==============================================================================
def build_model():
    in_feat = Input(shape=(N_FEATURES,), name='features_in')
    in_id   = Input(shape=(1,), dtype=tf.int32, name='asset_id')

    x = BatchNormalization()(in_feat)
    x = Dense(HIDDEN_DIMS[0], activation='relu')(x)
    x = Dropout(DROPOUT_RATES[0])(x)
    x = Dense(HIDDEN_DIMS[1], activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(DROPOUT_RATES[1])(x)

    e = Embedding(N_ASSETS, EMBED_DIM, name='asset_embed')(in_id)
    e = Flatten()(e)

    z = Concatenate()([x, e])
    z = Dense(HIDDEN_DIMS[2], activation='relu', kernel_regularizer=l2(1e-3))(z)
    z = Dropout(DROPOUT_RATES[2])(z)
    out = Dense(1, activation='sigmoid', dtype='float32')(z)

    return Model(inputs=[in_feat, in_id], outputs=out)


# ==============================================================================
# ASL LOSS
# ==============================================================================
def asymmetric_loss(gamma_neg=ASL_GAMMA_NEG, gamma_pos=ASL_GAMMA_POS,
                    clip=ASL_CLIP, label_smooth=LABEL_SMOOTH):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        if label_smooth > 0:
            y_true = y_true * (1.0 - label_smooth) + 0.5 * label_smooth
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
    def __init__(self, k=PREC_TOP_K, name='prec_top5pct', **kwargs):
        super().__init__(name=name, **kwargs)
        self.k     = k
        self.tp    = self.add_weight(name='tp',    initializer='zeros')
        self.total = self.add_weight(name='total', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_flat = tf.reshape(y_pred, [-1])
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
# EXECUTION — BUILD DATA
# ==============================================================================
market_ctx = build_market_context_daily()

print(f"\nLoading all teacher data...")
all_daily = []
for ticker in TRAIN_TEACHERS:
    print(f"   Loading {ticker}...")
    df_1min = load_ohlcv(ticker)
    if df_1min is None:
        continue
    df_1min = heal_data(df_1min)
    df_daily = compute_daily_records(df_1min)
    if df_daily is None or len(df_daily) < 50:
        print(f"      {ticker}: <50 trading days, skip")
        del df_1min
        _release_memory()
        continue
    df_daily = compute_features(df_daily)
    df_daily = df_daily.join(market_ctx, how='left')
    for c in market_ctx.columns:
        if c in df_daily.columns:
            df_daily[c] = df_daily[c].ffill().fillna(0)
    df_daily['ticker']   = ticker
    df_daily['asset_id'] = ASSET_INDEX.get(ticker, OTHER_IDX)
    df_daily = df_daily.loc[TRAIN_START_DATE:]

    tr  = df_daily['target'].mean()
    fr  = df_daily['filled'].mean()
    print(f"      {ticker}: {len(df_daily)} days, fill={fr:.0%}, "
          f"target={tr:.0%}, asset_id={ASSET_INDEX.get(ticker, OTHER_IDX)}")
    all_daily.append(df_daily)
    del df_1min
    _release_memory()

df_all = pd.concat(all_daily)
del all_daily
_release_memory()
df_all.sort_index(inplace=True)

print(f"\nAll teacher data: {len(df_all):,} daily records, "
      f"{df_all.index.min()} → {df_all.index.max()}")
print(f"Overall target rate: {df_all['target'].mean():.1%}")
print(f"Overall fill rate:   {df_all['filled'].mean():.1%}")

df_all['sample_weight'] = np.where(
    df_all['target'] == 0, 1.0,
    np.clip(1.0 + df_all['pnl_if_filled'] / DIP_PCT, 1.0, SAMPLE_WEIGHT_CAP)
).astype('float32')

# ==============================================================================
# TRAIN / VAL SPLIT + SCALING
# ==============================================================================
aware_cutoff     = pd.to_datetime(FOLD_CUTOFF)
gcs_results_path = f"Algos/Results_all_v2/Universal_Exp{FOLD_EXP_ID}"

df_train = df_all[df_all.index <= aware_cutoff].copy()
df_val   = df_all[df_all.index >  aware_cutoff].copy()

print(f"\nTrain: {len(df_train):,} rows  ({df_train.index.min()} → {df_train.index.max()})")
print(f"Val:   {len(df_val):,} rows  ({df_val.index.min()} → {df_val.index.max()})")
print(f"Train target rate: {df_train['target'].mean():.1%}")
print(f"Val target rate:   {df_val['target'].mean():.1%}")

print(f"\nPer-ticker breakdown:")
for t in TRAIN_TEACHERS:
    mask_tr = df_train['ticker'] == t
    mask_v  = df_val['ticker'] == t
    if mask_tr.sum() == 0:
        continue
    tr_rate = df_train.loc[mask_tr, 'target'].mean()
    v_rate  = df_val.loc[mask_v, 'target'].mean() if mask_v.sum() > 0 else 0
    avg_pnl = df_train.loc[mask_tr & (df_train['target'] == 1), 'pnl_if_filled']
    avg_pnl = avg_pnl.mean() if len(avg_pnl) > 0 else 0
    print(f"   {t:<6}: train={mask_tr.sum():>4}  val={mask_v.sum():>3}  "
          f"tgt_tr={tr_rate:.0%}  tgt_v={v_rate:.0%}  avg_pnl_win={avg_pnl:.2%}")

print(f"\nComputing scaling stats from training data...")
stats = compute_stats(df_train, FEATURE_COLS)

blob = client.bucket(BUCKET_NAME).blob(
    f"{STATS_DIR}/scaling_stats_experiment_{FOLD_EXP_ID}.json")
blob.upload_from_string(json.dumps(stats, indent=4), content_type='application/json')
print(f"SAVED STATS TO: {STATS_DIR}/scaling_stats_experiment_{FOLD_EXP_ID}.json")

df_train = apply_scaling(df_train, stats, FEATURE_COLS)
df_val   = apply_scaling(df_val,   stats, FEATURE_COLS)

print(f"\nAsset Embedding Index:")
for t, idx in ASSET_INDEX.items():
    print(f"   [{idx}] {t}")
print(f"   [{OTHER_IDX}] OTHER")

X_train  = df_train[FEATURE_COLS].values.astype('float32')
Y_train  = df_train['target'].values.astype('float32')
ID_train = df_train['asset_id'].values.astype('int32').reshape(-1, 1)
W_train  = df_train['sample_weight'].values.astype('float32')

X_val  = df_val[FEATURE_COLS].values.astype('float32')
Y_val  = df_val['target'].values.astype('float32')
ID_val = df_val['asset_id'].values.astype('int32').reshape(-1, 1)

print(f"\nTraining arrays: X={X_train.shape}, Y pos={Y_train.sum():.0f}/{len(Y_train)}")
print(f"Validation arrays: X={X_val.shape}, Y pos={Y_val.sum():.0f}/{len(Y_val)}")

# ==============================================================================
# TRAINING — 15 seeds
# ==============================================================================
trained_seeds       = []
fully_scored_seeds  = []
for seed in FOLD_SEEDS:
    model_blob_path = f"{GCS_MODEL_PATH}/UNIVERSAL_S{seed}_M{FOLD_EXP_ID}.h5"
    if bucket.blob(model_blob_path).exists():
        trained_seeds.append(seed)
        missing = [t for t in SCORING_POOL
                   if not bucket.blob(
                       f"{gcs_results_path}/{t}/Seed_{seed}_Results.csv").exists()]
        if not missing:
            fully_scored_seeds.append(seed)
        else:
            print(f"   Seed {seed}: model exists, {len(missing)} tickers unscored")

needs_training  = [s for s in FOLD_SEEDS if s not in trained_seeds]
needs_scoring   = [s for s in trained_seeds if s not in fully_scored_seeds]
remaining_seeds = needs_training + needs_scoring

if not remaining_seeds:
    print(f"\n   All {len(FOLD_SEEDS)} seeds fully trained and scored. Nothing to do.")
else:
    print(f"\n   {len(fully_scored_seeds)} done, {len(needs_training)} need training, "
          f"{len(needs_scoring)} need scoring only: {remaining_seeds}")

    for seed in remaining_seeds:
        print(f"\n{'='*60}")
        print(f"SEED {seed}  (Fold {FOLD_EXP_ID})")
        print(f"{'='*60}")
        tf.keras.backend.clear_session()
        random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

        local_model_path = os.path.join(
            LOCAL_MODEL_DIR, f"UNIVERSAL_S{seed}_M{FOLD_EXP_ID}.h5")
        gcs_model_blob   = f"{GCS_MODEL_PATH}/UNIVERSAL_S{seed}_M{FOLD_EXP_ID}.h5"
        skip_training    = seed in trained_seeds

        if skip_training:
            print(f"   Model trained — downloading from GCS for scoring...")
            bucket.blob(gcs_model_blob).download_to_filename(local_model_path)
        else:
            with strategy.scope():
                model = build_model()
                if seed == remaining_seeds[0]:
                    model.summary()

                try:
                    opt = tf.keras.optimizers.AdamW(
                        learning_rate=LEARNING_RATE, weight_decay=1e-4)
                except AttributeError:
                    opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

                model.compile(
                    optimizer=opt,
                    loss=asymmetric_loss(),
                    metrics=[
                        'accuracy',
                        tf.keras.metrics.AUC(name='auc'),
                        tf.keras.metrics.AUC(curve='PR', name='pr_auc'),
                        PrecisionAtTopK(k=PREC_TOP_K, name='prec_top5pct'),
                    ])

            callbacks = [
                tf.keras.callbacks.ModelCheckpoint(
                    local_model_path, monitor='val_loss',
                    save_best_only=True, mode='min', verbose=1),
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=PATIENCE,
                    restore_best_weights=True, mode='min', verbose=1),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', factor=0.5, patience=7,
                    min_lr=1e-6, mode='min', verbose=1),
            ]

            history = model.fit(
                [X_train, ID_train], Y_train,
                sample_weight=W_train,
                validation_data=([X_val, ID_val], Y_val),
                epochs=EPOCHS, batch_size=GLOBAL_BATCH_SIZE,
                callbacks=callbacks, verbose=1)

            h = history.history
            best_epoch   = np.argmin(h['val_loss']) + 1
            total_epochs = len(h['val_loss'])
            print(f"\n   === SEED {seed} TRAINING SUMMARY ===")
            print(f"   Best epoch: {best_epoch} / {total_epochs}")
            print(f"   val_loss:     {h['val_loss'][best_epoch-1]:.4f}")
            print(f"   val_auc:      {h.get('val_auc', [0])[best_epoch-1]:.4f}")
            print(f"   val_prec@5%:  {h.get('val_prec_top5pct', [0])[best_epoch-1]:.4f}")
            print(f"   train_loss:   {h['loss'][best_epoch-1]:.4f}")
            print(f"   train_auc:    {h.get('auc', [0])[best_epoch-1]:.4f}")
            gap = h.get('auc', [0])[best_epoch-1] - h.get('val_auc', [0])[best_epoch-1]
            print(f"   overfit_gap:  {gap:.4f}")
            print(f"   ================================")

            if os.path.exists(local_model_path):
                bucket.blob(gcs_model_blob).upload_from_filename(local_model_path)
                print(f"   UPLOADED MODEL TO: {gcs_model_blob}")

        # ==================================================================
        # SCORING — all tickers, val period
        # ==================================================================
        print(f"\nSCORING SEED {seed}...")

        with strategy.scope():
            model = build_model()
            model.compile(optimizer='adam', loss='binary_crossentropy')
        model.load_weights(local_model_path)

        for score_ticker in SCORING_POOL:
            gcs_csv = f"{gcs_results_path}/{score_ticker}/Seed_{seed}_Results.csv"
            if bucket.blob(gcs_csv).exists():
                continue

            print(f"   Scoring {score_ticker}...")
            df_1min = load_ohlcv(score_ticker)
            if df_1min is None:
                continue
            df_1min  = heal_data(df_1min)
            df_daily = compute_daily_records(df_1min)
            if df_daily is None or len(df_daily) < 50:
                del df_1min
                _release_memory()
                continue

            df_daily = compute_features(df_daily)
            df_daily = df_daily.join(market_ctx, how='left')
            for c in market_ctx.columns:
                if c in df_daily.columns:
                    df_daily[c] = df_daily[c].ffill().fillna(0)
            df_daily = df_daily.loc[TRAIN_START_DATE:]
            df_daily = apply_scaling(df_daily, stats, FEATURE_COLS)

            df_test = df_daily[df_daily.index > aware_cutoff]
            if len(df_test) < 5:
                del df_1min, df_daily
                _release_memory()
                continue

            asset_id_val = np.int32(ASSET_INDEX.get(score_ticker, OTHER_IDX))
            X_test  = df_test[FEATURE_COLS].values.astype('float32')
            ID_test = np.full((len(X_test), 1), asset_id_val, dtype=np.int32)

            probs = model.predict([X_test, ID_test], verbose=0).flatten()

            results_df = pd.DataFrame({
                'Target':        df_test['target'].values,
                'Prob_Spike':    probs,
                'Prediction':    (np.array(probs) > 0.5).astype(int),
                'Close':         df_test['close'].values,
                'Ticker':        score_ticker,
                'Filled':        df_test['filled'].values,
                'PnL_If_Filled': df_test['pnl_if_filled'].values,
            }, index=df_test.index)

            local_csv = os.path.join(
                LOCAL_RESULTS_DIR, f"Seed_{seed}_{score_ticker}_Results.csv")
            results_df.to_csv(local_csv)
            bucket.blob(gcs_csv).upload_from_filename(local_csv)
            print(f"      → {gcs_csv}")

            del df_1min, df_daily, df_test
            _release_memory()

        del model
        _release_memory()
        print(f"   Seed {seed} complete.")

    del df_all
    _release_memory()

print(f"\n{'='*70}")
print(f"EXPERIMENT 210 COMPLETE — {FOLD_EXP_ID}")
print(f"  {len(TRAIN_TEACHERS)} teachers, {N_FEATURES} features, DIP_PCT={DIP_PCT:.1%}")
print(f"  Target: dip >= {DIP_PCT:.1%} in first {SWING_WINDOW_MINUTES}min AND close > entry")
print(f"  Architecture: MLP {HIDDEN_DIMS} + Embed({EMBED_DIM})  ~16K params")
print(f"{'='*70}")
