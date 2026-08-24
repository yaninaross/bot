"""
Exp 500-503 Walkforward Training — Morning Swing Two-Stream Hybrid (3 Folds)
=============================================================================
Predicts whether a stock's peak in the first 30 minutes after market open
will exceed SWING_THRESHOLD (3.5%) above the opening price.

Two-stream architecture (~90K params):
  Stream 1: Lightweight CNN on 55 pre-market bars (5:20-6:15 AM PST)
  Stream 2: MLP on 40 daily features (adapted from Exp 210 + 6 new)
  Fusion:   Concat + Dense head with asset embedding

3 walkforward folds with progressively later training cutoffs.
15 seeds per fold, skip already-trained models.

Folds:
  501: cutoff Sep 2025    502: cutoff Nov 2025    503: cutoff Jan 2026

Set FOLD_RANGE to control which folds to train in this notebook.
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
SCRIPT_VERSION = "500WF.1"
print(f"Script version: {SCRIPT_VERSION}")
print("Syncing environment for Experiment 500 Walkforward (Morning Swing, 3 Folds)...")

try:
    import tensorflow as tf
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "tensorflow[and-cuda]",
                           "google-cloud-storage", "pyarrow", "gcsfs", "--quiet"])
    import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, GlobalAveragePooling1D,
    LayerNormalization, BatchNormalization, Concatenate,
    Embedding, Flatten, Conv1D, MaxPool1D, SpatialDropout1D,
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
# CONFIGURATION — 3 WALKFORWARD FOLDS
# ==============================================================================
ALL_FOLDS = [
    {'exp_id': '501_SWING', 'cutoff': '2025-09-01', 'seed_base': 50100},
    {'exp_id': '502_SWING', 'cutoff': '2025-11-01', 'seed_base': 50200},
    {'exp_id': '503_SWING', 'cutoff': '2026-01-01', 'seed_base': 50300},
]

FOLD_RANGE = []

FOLDS_TO_TRAIN = [ALL_FOLDS[i] for i in FOLD_RANGE] if FOLD_RANGE else ALL_FOLDS
SEEDS_PER_FOLD = 15

# Time constants (PST minutes from midnight)
PST = pytz.timezone('America/Los_Angeles')
PREMARKET_START_MIN = 320   # 5:20 AM PST
PREMARKET_END_MIN   = 375   # 6:15 AM PST
MARKET_OPEN_MIN     = 390   # 6:30 AM PST
SWING_END_MIN       = 420   # 7:00 AM PST (30 min after open)
EOD_MIN             = 780   # 1:00 PM PST

PM_STEPS       = PREMARKET_END_MIN - PREMARKET_START_MIN  # 55 bars
SWING_THRESHOLD = 0.035   # 3.5% up swing target

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

# ==============================================================================
# FEATURE DEFINITIONS
# ==============================================================================
DAILY_FEATURE_COLS = [
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
    'swing_rate_20d', 'big_swing_rate_20d', 'dip_first_rate_20d', 'consec_swing_days',
    # Group 7: Seasonality (4)
    'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
    # Group 8: Market context (5)
    'spy_prev_return', 'spy_5d_return', 'vix_close_norm', 'vix_change', 'tlt_prev_return',
    # Group 9: New features from June investigation (6)
    'vix_daily_range', 'vix_range_5d', 'spy_3d_return',
    'asset_vol_accel', 'prev_swing_up_30m', 'prev_swing_down_30m',
]
N_DAILY_FEATURES = len(DAILY_FEATURE_COLS)   # 40

PM_FEATURE_COLS = [
    'price_vs_prev_close', 'high_vs_prev_close', 'low_vs_prev_close',
    'volume_norm', 'cumul_volume', 'bar_is_real', 'spread', 'obi',
]
N_PM_FEATURES = len(PM_FEATURE_COLS)  # 8

# Model hyperparams
EMBED_DIM     = 16
CNN_FILTERS   = 64
CNN_KERNELS   = [3, 7]
MLP_HIDDEN    = [128, 64]
MLP_DROPOUT   = [0.3, 0.2]
FUSION_DIM    = 64
FUSION_DROPOUT = 0.3

LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
EPOCHS        = 50
PATIENCE      = 10

ASL_GAMMA_NEG     = 2.0
ASL_GAMMA_POS     = 0.5
ASL_CLIP          = 0.02
LABEL_SMOOTH      = 0.03
SAMPLE_WEIGHT_CAP = 3.0
PREC_TOP_K        = 0.05

AUGMENT_SCALE     = (0.95, 1.05)
AUGMENT_BAR_MASK  = 0.05

# GCS paths
BUCKET_NAME    = "time-series-bucket-ys"
OHLCV_DIR      = "Data/Inference"
INFERENCE_DIR  = "Data/Inference_v2"
GCS_MODEL_PATH = "Algos/to_deploy/Saved_models_v2/UNIVERSAL/Direction"
STATS_DIR      = "Data/Inference_v2"

LOCAL_RESULTS_DIR = "Results"
LOCAL_MODEL_DIR   = "Models"
os.makedirs(LOCAL_RESULTS_DIR, exist_ok=True)
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

ASSET_INDEX = {ticker: i for i, ticker in enumerate(TRAIN_TEACHERS)}
N_ASSETS    = len(TRAIN_TEACHERS) + 1
OTHER_IDX   = len(TRAIN_TEACHERS)

PER_REPLICA_BATCH = 256
GLOBAL_BATCH_SIZE = PER_REPLICA_BATCH * NUM_REPLICAS

print(f"\nConfig: {len(TRAIN_TEACHERS)} teachers, {N_DAILY_FEATURES} daily features, "
      f"{PM_STEPS} PM bars × {N_PM_FEATURES} PM features")
print(f"Target: max up swing >= {SWING_THRESHOLD:.1%} in first 30 min after open")
print(f"Pre-market window: {PREMARKET_START_MIN//60}:{PREMARKET_START_MIN%60:02d} - "
      f"{PREMARKET_END_MIN//60}:{PREMARKET_END_MIN%60:02d} PST")
print(f"Folds to train: {[f['exp_id'] for f in FOLDS_TO_TRAIN]}")
print(f"Seeds per fold: {SEEDS_PER_FOLD}")
print(f"Architecture: CNN({CNN_KERNELS}) + MLP({MLP_HIDDEN}) → Fusion({FUSION_DIM}) | "
      f"embed={EMBED_DIM}")
print(f"ASL: gamma_neg={ASL_GAMMA_NEG}, gamma_pos={ASL_GAMMA_POS}, "
      f"clip={ASL_CLIP}, label_smooth={LABEL_SMOOTH}")

# ==============================================================================
# DATA LOADING
# ==============================================================================
MICRO_COLS = ['Spread', 'OBI', 'Volume_Delta', 'Trade_Count', 'Delta_Pct',
              'bsize', 'asize', 'Total_Vol']


def load_master(ticker, add_micro_zeros=True):
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
        cutoff = pd.Timestamp(DATA_START_DATE, tz='UTC')
        df = df[df.index >= cutoff]
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
# DAILY RECORDS — 1-min bars → one row per trading day
# ==============================================================================
def compute_daily_records(df_1min):
    df_pst = df_1min.copy()
    df_pst.index = df_pst.index.tz_convert(PST)
    mins = df_pst.index.hour * 60 + df_pst.index.minute

    is_market = (mins >= MARKET_OPEN_MIN) & (mins <= EOD_MIN)
    is_real   = df_pst['Volume'] > 0
    market    = df_pst[is_market & is_real].copy()

    is_swing_window = (mins >= MARKET_OPEN_MIN) & (mins <= SWING_END_MIN)
    swing_bars      = df_pst[is_swing_window & is_real].copy()

    is_eod    = mins >= EOD_MIN
    eod_bars  = df_pst[is_eod & is_real].copy()
    eod_bars['date'] = eod_bars.index.date
    eod_closes = eod_bars.groupby('date')['Close'].first()

    market['date']     = market.index.date
    swing_bars['date'] = swing_bars.index.date

    records = []
    for date, day_market in market.groupby('date'):
        if len(day_market) < 5:
            continue
        day_swing = swing_bars[swing_bars['date'] == date]
        if len(day_swing) < 3:
            continue

        open_price = float(day_market.iloc[0]['Close'])
        if open_price <= 0:
            continue

        daily_high   = float(day_market['High'].max())
        daily_low    = float(day_market['Low'].min())
        daily_close  = float(eod_closes.get(date, day_market.iloc[-1]['Close']))
        daily_volume = float(day_market['Volume'].sum())

        swing_max_high = float(day_swing['High'].max())
        swing_min_low  = float(day_swing['Low'].min())
        max_up_swing   = (swing_max_high - open_price) / open_price
        max_down_swing = (open_price - swing_min_low) / open_price

        trough_idx = day_swing['Low'].values.argmin()
        peak_idx   = day_swing['High'].values.argmax()
        dip_first  = float(trough_idx < peak_idx)

        target = 1 if max_up_swing >= SWING_THRESHOLD else 0

        records.append({
            'date': pd.Timestamp(date),
            'open': open_price,
            'high': daily_high,
            'low': daily_low,
            'close': daily_close,
            'volume': daily_volume,
            'max_up_swing': max_up_swing,
            'max_down_swing': max_down_swing,
            'dip_first': dip_first,
            'target': target,
        })

    if not records:
        return None
    df_daily = pd.DataFrame(records).set_index('date')
    df_daily.index = pd.DatetimeIndex(df_daily.index)
    return df_daily


# ==============================================================================
# PRE-MARKET SEQUENCE — 55 bars × 8 features per day
# ==============================================================================
def extract_premarket_sequences(df_1min, df_daily):
    df_pst = df_1min.copy()
    df_pst.index = df_pst.index.tz_convert(PST)
    mins = df_pst.index.hour * 60 + df_pst.index.minute

    is_pm = (mins >= PREMARKET_START_MIN) & (mins < PREMARKET_END_MIN)
    pm_bars = df_pst[is_pm].copy()
    pm_bars['date'] = pm_bars.index.date

    pm_vol_avg = pm_bars.groupby('date')['Volume'].sum().rolling(
        20, min_periods=5).mean().shift(1).fillna(1e-6)
    pm_vol_per_bar_avg = pm_bars[pm_bars['Volume'] > 0].groupby('date')['Volume'].mean().rolling(
        20, min_periods=5).mean().shift(1).fillna(1e-6)

    sequences = {}

    for date in df_daily.index:
        date_val = date.date() if hasattr(date, 'date') else date
        prev_close = df_daily.loc[date, 'close'] if 'close' in df_daily.columns else None

        if prev_close is None or prev_close <= 0:
            continue

        prev_day_idx = df_daily.index.get_loc(date)
        if prev_day_idx == 0:
            continue
        prev_day_close = float(df_daily.iloc[prev_day_idx - 1]['close'])
        if prev_day_close <= 0:
            continue

        day_pm = pm_bars[pm_bars['date'] == date_val]

        seq = np.zeros((PM_STEPS, N_PM_FEATURES), dtype=np.float32)

        total_pm_vol_avg = float(pm_vol_avg.get(date_val, 1e-6))
        bar_vol_avg      = float(pm_vol_per_bar_avg.get(date_val, 1e-6))
        if total_pm_vol_avg <= 0:
            total_pm_vol_avg = 1e-6
        if bar_vol_avg <= 0:
            bar_vol_avg = 1e-6

        cumul_vol = 0.0

        for _, row in day_pm.iterrows():
            bar_min = row.name.hour * 60 + row.name.minute
            idx = bar_min - PREMARKET_START_MIN
            if idx < 0 or idx >= PM_STEPS:
                continue

            close_val  = float(row['Close']) if pd.notna(row['Close']) else 0.0
            high_val   = float(row['High']) if pd.notna(row['High']) else 0.0
            low_val    = float(row['Low']) if pd.notna(row['Low']) else 0.0
            vol_val    = float(row['Volume']) if pd.notna(row['Volume']) else 0.0
            spread_val = float(row.get('Spread', 0)) if pd.notna(row.get('Spread', 0)) else 0.0
            obi_val    = float(row.get('OBI', 0)) if pd.notna(row.get('OBI', 0)) else 0.0

            cumul_vol += vol_val
            is_real = 1.0 if vol_val > 0 else 0.0

            seq[idx, 0] = (close_val - prev_day_close) / prev_day_close if close_val > 0 else 0.0
            seq[idx, 1] = (high_val - prev_day_close) / prev_day_close if high_val > 0 else 0.0
            seq[idx, 2] = (low_val - prev_day_close) / prev_day_close if low_val > 0 else 0.0
            seq[idx, 3] = vol_val / bar_vol_avg
            seq[idx, 4] = cumul_vol / total_pm_vol_avg
            seq[idx, 5] = is_real
            seq[idx, 6] = spread_val
            seq[idx, 7] = obi_val

        sequences[date] = seq

    return sequences


# ==============================================================================
# MARKET CONTEXT — daily SPY / VIX / TLT features (with new regime features)
# ==============================================================================
def build_market_context_daily():
    print("   Building daily market context (SPY, VIX, TLT)...")
    ctx = pd.DataFrame()

    for ticker in CONTEXT_TICKERS:
        print(f"      Loading {ticker}...")
        df = load_master(ticker, add_micro_zeros=False)
        if df is None:
            print(f"      Could not load {ticker}")
            continue
        df = heal_data(df)

        df_pst = df.copy()
        df_pst.index = df_pst.index.tz_convert(PST)
        mins = df_pst.index.hour * 60 + df_pst.index.minute
        mkt = df_pst[(mins >= MARKET_OPEN_MIN) & (mins <= EOD_MIN) &
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
            ctx['spy_3d_return']   = daily['close'].pct_change(3).shift(1)
        elif ticker == 'VIX':
            ctx['vix_close_norm'] = daily['close'].shift(1) / 20.0
            ctx['vix_change']     = daily['close'].diff().shift(1) / 20.0
            vix_range = (daily['high'] - daily['low']) / daily['close'].clip(lower=0.01)
            ctx['vix_daily_range'] = vix_range.shift(1)
            ctx['vix_range_5d']    = vix_range.rolling(5, min_periods=2).mean().shift(1)
        elif ticker == 'TLT':
            ctx['tlt_prev_return'] = daily['close'].pct_change().shift(1)

        del df, df_pst, mkt, daily
        _release_memory()

    ctx = ctx.fillna(0)
    print(f"      Market context: {len(ctx)} days, cols={list(ctx.columns)}")
    return ctx


# ==============================================================================
# FEATURE ENGINEERING — 40 daily features (no lookahead)
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
    df['prev_upper_wick']     = (df['high'].shift(1) - body_top) / prev_range
    df['prev_lower_wick']     = (body_bot - df['low'].shift(1)) / prev_range
    df['prev_close_position'] = (df['close'].shift(1) - df['low'].shift(1)) / \
                                 (df['high'].shift(1) - df['low'].shift(1)).clip(lower=eps)
    vol_avg_20 = df['volume'].rolling(20, min_periods=5).mean().shift(1)
    df['prev_volume_ratio'] = df['volume'].shift(1) / vol_avg_20.clip(lower=1)

    # --- Group 3: Prior opening dynamics ---
    df['prev_max_dd']    = df['max_down_swing'].shift(1)
    df['prev_max_ru']    = df['max_up_swing'].shift(1)
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

    # --- Group 6: Pattern history (adapted for swing target) ---
    swing_hit = (df['max_up_swing'] >= SWING_THRESHOLD).astype(float)
    df['swing_rate_20d'] = swing_hit.rolling(20, min_periods=5).mean().shift(1)
    big_swing  = (df['max_up_swing'] >= SWING_THRESHOLD * 1.5).astype(float)
    df['big_swing_rate_20d'] = big_swing.rolling(20, min_periods=5).mean().shift(1)
    df['dip_first_rate_20d'] = df['dip_first'].rolling(20, min_periods=5).mean().shift(1)
    groups = swing_hit.ne(swing_hit.shift()).cumsum()
    consec = swing_hit.groupby(groups).cumsum()
    df['consec_swing_days'] = consec.shift(1).clip(upper=10) / 10.0

    # --- Group 7: Seasonality ---
    df['dow_sin']   = np.sin(2 * np.pi * df.index.dayofweek / 5)
    df['dow_cos']   = np.cos(2 * np.pi * df.index.dayofweek / 5)
    df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)

    # --- Group 9: New features (market context filled by join, asset-level here) ---
    atr_5d  = (df['high'] - df['low']).rolling(5, min_periods=3).mean().shift(1)
    atr_20d_raw = (df['high'] - df['low']).rolling(20, min_periods=5).mean().shift(1)
    df['asset_vol_accel'] = atr_5d / atr_20d_raw.clip(lower=eps)

    df['prev_swing_up_30m']   = df['max_up_swing'].shift(1)
    df['prev_swing_down_30m'] = df['max_down_swing'].shift(1)

    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    for c in df.columns:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    return df


# ==============================================================================
# SCALING
# ==============================================================================
SKIP_SCALE_DAILY = {'dow_sin', 'dow_cos', 'month_sin', 'month_cos', 'prev_dip_first'}


def compute_stats_daily(df_train, cols):
    stats = {}
    for col in cols:
        if col in SKIP_SCALE_DAILY:
            continue
        if col not in df_train.columns:
            continue
        med = float(df_train[col].median())
        iqr = float(max(1e-6,
                        df_train[col].quantile(0.75) - df_train[col].quantile(0.25)))
        stats[col] = {'med': med, 'iqr': iqr}
    return stats


def compute_stats_sequence(all_train_seqs):
    n_features = all_train_seqs.shape[2]
    stats = {}
    for f in range(n_features):
        vals = all_train_seqs[:, :, f].flatten()
        vals = vals[vals != 0]
        if len(vals) < 10:
            stats[f] = {'med': 0.0, 'iqr': 1.0}
            continue
        med = float(np.median(vals))
        iqr = float(max(1e-6, np.percentile(vals, 75) - np.percentile(vals, 25)))
        stats[f] = {'med': med, 'iqr': iqr}
    return stats


def apply_scaling_daily(df, stats, cols):
    df_s = df.copy()
    for col in cols:
        if col in stats:
            df_s[col] = ((df_s[col] - stats[col]['med']) /
                         stats[col]['iqr']).clip(-10, 10).astype('float32')
    return df_s


def apply_scaling_sequence(seqs, stats):
    scaled = seqs.copy()
    for f, s in stats.items():
        mask = scaled[:, :, f] != 0
        scaled[:, :, f] = np.where(
            mask,
            np.clip((scaled[:, :, f] - s['med']) / s['iqr'], -10, 10),
            0.0
        )
    return scaled.astype('float32')


# ==============================================================================
# MODEL — Two-Stream Hybrid
# ==============================================================================
def build_model():
    in_seq  = Input(shape=(PM_STEPS, N_PM_FEATURES), name='sequence_in')
    in_feat = Input(shape=(N_DAILY_FEATURES,), name='features_in')
    in_id   = Input(shape=(1,), dtype=tf.int32, name='asset_id')

    # --- Stream 1: Pre-market CNN ---
    x = Dense(CNN_FILTERS)(in_seq)                              # (55, 64)
    x = SpatialDropout1D(0.1)(x)

    conv_branches = []
    for k in CNN_KERNELS:
        c = Conv1D(CNN_FILTERS, kernel_size=k, padding='same', activation='relu')(x)
        conv_branches.append(c)
    x = Concatenate(axis=-1)(conv_branches)                     # (55, 128)
    x = LayerNormalization()(x)
    x = MaxPool1D(pool_size=5)(x)                               # (11, 128)
    x = Conv1D(CNN_FILTERS, kernel_size=3, padding='same', activation='relu')(x)  # (11, 64)
    x = GlobalAveragePooling1D()(x)                             # (64,)

    # --- Stream 2: Daily features MLP ---
    y = BatchNormalization()(in_feat)
    y = Dense(MLP_HIDDEN[0], activation='relu')(y)              # (128,)
    y = Dropout(MLP_DROPOUT[0])(y)
    y = Dense(MLP_HIDDEN[1], activation='relu')(y)              # (64,)
    y = BatchNormalization()(y)
    y = Dropout(MLP_DROPOUT[1])(y)

    # --- Asset embedding ---
    e = Embedding(N_ASSETS, EMBED_DIM, name='asset_embed')(in_id)
    e = Flatten()(e)
    e = Dense(EMBED_DIM, activation='relu')(e)                  # (16,)

    # --- Fusion ---
    z = Concatenate()([x, y, e])                                # (144,)
    z = Dense(FUSION_DIM, activation='relu',
              kernel_regularizer=l2(1e-3))(z)                   # (64,)
    z = Dropout(FUSION_DROPOUT)(z)
    out = Dense(1, activation='sigmoid', dtype='float32')(z)

    return Model(inputs=[in_seq, in_feat, in_id], outputs=out)


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
# COSINE LR SCHEDULE
# ==============================================================================
def build_cosine_schedule(lr_init, total_epochs, steps_per_epoch, warmup_epochs=3):
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps  = total_epochs * steps_per_epoch

    return tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=lr_init,
        decay_steps=total_steps - warmup_steps,
        alpha=1e-6,
        warmup_target=lr_init,
        warmup_steps=warmup_steps,
    )


# ==============================================================================
# EXECUTION — LOAD ALL DATA
# ==============================================================================
market_ctx = build_market_context_daily()

print(f"\nLoading all teacher data...")
all_daily = []
all_pm_seqs = {}

for ticker in TRAIN_TEACHERS:
    print(f"   Loading {ticker}...")
    df_1min = load_master(ticker)
    if df_1min is None:
        continue
    df_1min = df_1min.sort_index()
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

    pm_seqs = extract_premarket_sequences(df_1min, df_daily)
    for date, seq in pm_seqs.items():
        all_pm_seqs[(ticker, date)] = seq

    df_daily['ticker']   = ticker
    df_daily['asset_id'] = ASSET_INDEX.get(ticker, OTHER_IDX)
    df_daily = df_daily.loc[TRAIN_START_DATE:]

    tr  = df_daily['target'].mean()
    swing_avg = df_daily['max_up_swing'].mean()
    print(f"      {ticker}: {len(df_daily)} days, target={tr:.0%}, "
          f"avg_swing={swing_avg:.2%}, asset_id={ASSET_INDEX.get(ticker, OTHER_IDX)}")
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
print(f"Overall avg up swing: {df_all['max_up_swing'].mean():.2%}")

df_all['sample_weight'] = np.where(
    df_all['target'] == 0, 1.0,
    np.clip(1.0 + df_all['max_up_swing'] / SWING_THRESHOLD, 1.0, SAMPLE_WEIGHT_CAP)
).astype('float32')

# ==============================================================================
# WALKFORWARD TRAINING — loop over folds
# ==============================================================================
for fold_cfg in FOLDS_TO_TRAIN:
    fold_id   = fold_cfg['exp_id']
    cutoff    = fold_cfg['cutoff']
    seed_base = fold_cfg['seed_base']
    fold_seeds = [seed_base + 1 + i for i in range(SEEDS_PER_FOLD)]

    print(f"\n{'='*70}")
    print(f"FOLD: {fold_id}  cutoff={cutoff}  seeds={fold_seeds[0]}-{fold_seeds[-1]}")
    print(f"{'='*70}")

    aware_cutoff     = pd.to_datetime(cutoff)
    gcs_results_path = f"Algos/Results_all_v2/Universal_Exp{fold_id}"

    # --- Train / Val split ---
    df_train = df_all[df_all.index <= aware_cutoff].copy()
    df_val   = df_all[df_all.index >  aware_cutoff].copy()

    print(f"\nTrain: {len(df_train):,} rows  ({df_train.index.min()} → {df_train.index.max()})")
    print(f"Val:   {len(df_val):,} rows  ({df_val.index.min()} → {df_val.index.max()})")
    print(f"Train target rate: {df_train['target'].mean():.1%}")
    print(f"Val target rate:   {df_val['target'].mean():.1%}")

    # --- Compute scaling stats from training data ---
    print(f"\nComputing scaling stats from training data...")
    daily_stats = compute_stats_daily(df_train, DAILY_FEATURE_COLS)

    blob = client.bucket(BUCKET_NAME).blob(
        f"{STATS_DIR}/scaling_stats_experiment_{fold_id}.json")
    blob.upload_from_string(json.dumps(daily_stats, indent=4),
                            content_type='application/json')
    print(f"SAVED DAILY STATS TO: {STATS_DIR}/scaling_stats_experiment_{fold_id}.json")

    df_train_s = apply_scaling_daily(df_train, daily_stats, DAILY_FEATURE_COLS)
    df_val_s   = apply_scaling_daily(df_val,   daily_stats, DAILY_FEATURE_COLS)

    # --- Build arrays ---
    def _build_arrays(df_split, split_name):
        X_feat = df_split[DAILY_FEATURE_COLS].values.astype('float32')
        Y      = df_split['target'].values.astype('float32')
        IDs    = df_split['asset_id'].values.astype('int32').reshape(-1, 1)
        W      = df_split['sample_weight'].values.astype('float32') if 'sample_weight' in df_split.columns \
                 else np.ones(len(df_split), dtype=np.float32)

        X_seq = np.zeros((len(df_split), PM_STEPS, N_PM_FEATURES), dtype=np.float32)
        valid_mask = np.ones(len(df_split), dtype=bool)
        for i, (idx, row) in enumerate(df_split.iterrows()):
            key = (row['ticker'], idx)
            if key in all_pm_seqs:
                X_seq[i] = all_pm_seqs[key]
            else:
                valid_mask[i] = False

        n_missing = (~valid_mask).sum()
        if n_missing > 0:
            print(f"   {split_name}: {n_missing}/{len(df_split)} days missing PM sequence, keeping with zero seq")

        return X_feat, X_seq, Y, IDs, W

    X_feat_train, X_seq_train, Y_train, ID_train, W_train = _build_arrays(df_train_s, 'Train')
    X_feat_val,   X_seq_val,   Y_val,   ID_val,   _       = _build_arrays(df_val_s,   'Val')

    # --- Scale pre-market sequences ---
    seq_stats = compute_stats_sequence(X_seq_train)
    X_seq_train = apply_scaling_sequence(X_seq_train, seq_stats)
    X_seq_val   = apply_scaling_sequence(X_seq_val,   seq_stats)

    seq_stats_blob = client.bucket(BUCKET_NAME).blob(
        f"{STATS_DIR}/seq_scaling_stats_experiment_{fold_id}.json")
    seq_stats_serializable = {str(k): v for k, v in seq_stats.items()}
    seq_stats_blob.upload_from_string(json.dumps(seq_stats_serializable, indent=4),
                                      content_type='application/json')
    print(f"SAVED SEQ STATS TO: {STATS_DIR}/seq_scaling_stats_experiment_{fold_id}.json")

    print(f"\nTraining arrays: feat={X_feat_train.shape}, seq={X_seq_train.shape}, "
          f"Y pos={Y_train.sum():.0f}/{len(Y_train)}")
    print(f"Validation arrays: feat={X_feat_val.shape}, seq={X_seq_val.shape}, "
          f"Y pos={Y_val.sum():.0f}/{len(Y_val)}")

    # --- Per-ticker breakdown ---
    print(f"\nPer-ticker breakdown:")
    for t in TRAIN_TEACHERS:
        mask_tr = df_train_s['ticker'] == t
        mask_v  = df_val_s['ticker'] == t
        if mask_tr.sum() == 0:
            continue
        tr_rate = df_train.loc[mask_tr, 'target'].mean()
        v_rate  = df_val.loc[mask_v, 'target'].mean() if mask_v.sum() > 0 else 0
        avg_swing = df_train.loc[mask_tr, 'max_up_swing'].mean()
        print(f"   {t:<6}: train={mask_tr.sum():>4}  val={mask_v.sum():>3}  "
              f"tgt_tr={tr_rate:.0%}  tgt_v={v_rate:.0%}  avg_swing={avg_swing:.2%}")

    print(f"\nAsset Embedding Index:")
    for t, idx in ASSET_INDEX.items():
        print(f"   [{idx}] {t}")
    print(f"   [{OTHER_IDX}] OTHER")

    # --- Check for already-trained seeds ---
    trained_seeds       = []
    fully_scored_seeds  = []
    for seed in fold_seeds:
        model_blob_path = f"{GCS_MODEL_PATH}/UNIVERSAL_S{seed}_M{fold_id}.h5"
        if bucket.blob(model_blob_path).exists():
            trained_seeds.append(seed)
            missing = [t for t in SCORING_POOL
                       if not bucket.blob(
                           f"{gcs_results_path}/{t}/Seed_{seed}_Results.csv").exists()]
            if not missing:
                fully_scored_seeds.append(seed)
            else:
                print(f"   Seed {seed}: model exists, {len(missing)} tickers unscored")

    needs_training  = [s for s in fold_seeds if s not in trained_seeds]
    needs_scoring   = [s for s in trained_seeds if s not in fully_scored_seeds]
    remaining_seeds = needs_training + needs_scoring

    if not remaining_seeds:
        print(f"\n   All {SEEDS_PER_FOLD} seeds fully trained and scored for fold {fold_id}.")
        continue

    print(f"\n   {len(fully_scored_seeds)} done, {len(needs_training)} need training, "
          f"{len(needs_scoring)} need scoring only: {remaining_seeds}")

    steps_per_epoch = max(1, len(Y_train) // GLOBAL_BATCH_SIZE)

    # --- Train each seed ---
    for seed in remaining_seeds:
        print(f"\n{'='*60}")
        print(f"SEED {seed}  (Fold {fold_id})")
        print(f"{'='*60}")
        tf.keras.backend.clear_session()
        random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

        local_model_path = os.path.join(
            LOCAL_MODEL_DIR, f"UNIVERSAL_S{seed}_M{fold_id}.h5")
        gcs_model_blob   = f"{GCS_MODEL_PATH}/UNIVERSAL_S{seed}_M{fold_id}.h5"
        skip_training    = seed in trained_seeds

        if skip_training:
            print(f"   Model trained — downloading from GCS for scoring...")
            bucket.blob(gcs_model_blob).download_to_filename(local_model_path)
        else:
            with strategy.scope():
                model = build_model()
                if seed == remaining_seeds[0]:
                    model.summary()

                lr_schedule = build_cosine_schedule(
                    LEARNING_RATE, EPOCHS, steps_per_epoch, warmup_epochs=3)

                try:
                    opt = tf.keras.optimizers.AdamW(
                        learning_rate=lr_schedule, weight_decay=WEIGHT_DECAY)
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
            ]

            history = model.fit(
                [X_seq_train, X_feat_train, ID_train], Y_train,
                sample_weight=W_train,
                validation_data=([X_seq_val, X_feat_val, ID_val], Y_val),
                epochs=EPOCHS, batch_size=GLOBAL_BATCH_SIZE,
                callbacks=callbacks, verbose=1)

            h = history.history
            best_epoch   = np.argmin(h['val_loss']) + 1
            total_epochs = len(h['val_loss'])
            print(f"\n   === SEED {seed} TRAINING SUMMARY ===")
            print(f"   Best epoch: {best_epoch} / {total_epochs}")
            print(f"   val_loss:     {h['val_loss'][best_epoch-1]:.4f}")
            print(f"   val_auc:      {h.get('val_auc', [0])[best_epoch-1]:.4f}")
            print(f"   val_pr_auc:   {h.get('val_pr_auc', [0])[best_epoch-1]:.4f}")
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
            df_1min = load_master(score_ticker)
            if df_1min is None:
                continue
            df_1min = df_1min.sort_index()
            df_1min = heal_data(df_1min)

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

            pm_seqs_score = extract_premarket_sequences(df_1min, df_daily)

            df_daily = apply_scaling_daily(df_daily, daily_stats, DAILY_FEATURE_COLS)

            df_test = df_daily[df_daily.index > aware_cutoff]
            if len(df_test) < 5:
                del df_1min, df_daily
                _release_memory()
                continue

            asset_id_val = np.int32(ASSET_INDEX.get(score_ticker, OTHER_IDX))

            X_feat_test = df_test[DAILY_FEATURE_COLS].values.astype('float32')
            ID_test     = np.full((len(df_test), 1), asset_id_val, dtype=np.int32)

            X_seq_test = np.zeros((len(df_test), PM_STEPS, N_PM_FEATURES), dtype=np.float32)
            for i, idx in enumerate(df_test.index):
                if idx in pm_seqs_score:
                    X_seq_test[i] = pm_seqs_score[idx]
            X_seq_test = apply_scaling_sequence(X_seq_test, seq_stats)

            probs = model.predict([X_seq_test, X_feat_test, ID_test], verbose=0).flatten()

            results_df = pd.DataFrame({
                'Target':         df_test['target'].values,
                'Prob_Swing':     probs,
                'Prediction':     (np.array(probs) > 0.5).astype(int),
                'Max_Up_Swing':   df_test['max_up_swing'].values if 'max_up_swing' in df_test.columns
                                  else np.zeros(len(df_test)),
                'Max_Down_Swing': df_test['max_down_swing'].values if 'max_down_swing' in df_test.columns
                                  else np.zeros(len(df_test)),
                'Open':           df_test['open'].values,
                'Close':          df_test['close'].values,
                'Ticker':         score_ticker,
            }, index=df_test.index)

            local_csv = os.path.join(
                LOCAL_RESULTS_DIR, f"Seed_{seed}_{score_ticker}_Results.csv")
            results_df.to_csv(local_csv)
            bucket.blob(gcs_csv).upload_from_filename(local_csv)
            print(f"      → {gcs_csv}")

            del df_1min, df_daily, df_test, pm_seqs_score
            _release_memory()

        del model
        _release_memory()
        print(f"   Seed {seed} complete.")

    del df_train, df_val, df_train_s, df_val_s
    del X_feat_train, X_seq_train, Y_train, ID_train, W_train
    del X_feat_val, X_seq_val, Y_val, ID_val
    _release_memory()
    print(f"\nFold {fold_id} complete.")

del df_all
_release_memory()

print(f"\n{'='*70}")
print(f"EXPERIMENT 500 COMPLETE")
print(f"  {len(TRAIN_TEACHERS)} teachers, {N_DAILY_FEATURES} daily + {PM_STEPS}×{N_PM_FEATURES} sequence features")
print(f"  Target: max up swing >= {SWING_THRESHOLD:.1%} in first 30 min after open")
print(f"  Architecture: CNN({CNN_KERNELS}) + MLP({MLP_HIDDEN}) + Embed({EMBED_DIM}) → Fusion({FUSION_DIM})")
print(f"  Folds: {[f['exp_id'] for f in ALL_FOLDS]}")
print(f"{'='*70}")
