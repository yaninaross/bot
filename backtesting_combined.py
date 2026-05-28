"""
Unified Backtesting — All Deployed Experiments
===============================================
Mirrors daily_score_combined.py's 16 deployed assets across 4 experiments
(Exp-97, Exp-110, Exp-112, Exp-103) with exact deployed configs.

Purpose: determine if backtester reproduces the same GO signals as live
scoring — separates "model issue" from "data pipeline mismatch".

Usage:  python backtesting_combined.py
"""

import sys
import os
import io
import gc
import json
import pytz
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import subprocess
try:
    import pandas_ta as ta
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas_ta", "--quiet"])
    import pandas_ta as ta

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Dropout, GlobalAveragePooling1D,
    LayerNormalization, MultiHeadAttention, Add,
    BatchNormalization, Concatenate, Embedding, Flatten,
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from google.cloud import storage

# ==============================================================================
# SHARED CONSTANTS
# ==============================================================================
BUCKET_NAME   = "time-series-bucket-ys"
INFERENCE_DIR = "Data/Inference"
GCS_MODEL_PATH = "Algos/to_deploy/Saved_models_v2/UNIVERSAL/Direction"
TIME_STEPS    = 240
EMBED_DIM     = 16
BACKTEST_DAYS = 150
PST = pytz.timezone('America/Los_Angeles')

TICKER_FILE_MAP = {'BITF': 'KEEL'}

MICRO_COLS = ['Spread', 'OBI', 'Volume_Delta', 'Trade_Count', 'Delta_Pct',
              'bsize', 'asize', 'Total_Vol']

ROC_HORIZONS = [5, 30, 60, 120]

ASSET_COLS = (['Gap_Pct', 'RSI_14', 'RSI_60', 'RVOL', 'Compression',
               'ATR_14', 'Dist_to_PM_High'] +
              [f'ROC_{h}m' for h in ROC_HORIZONS] + MICRO_COLS)

# ==============================================================================
# EXPERIMENT DEFINITIONS — EXACT COPY from daily_score_combined.py
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


EXP_97 = {
    'model_id': '97_MICRO',
    'label': 'EXP-97 CRYPTO',
    'train_teachers': [
        'MARA', 'IREN', 'WULF', 'APLD', 'RIOT', 'CLSK', 'CORZ', 'MSTR',
        'COIN', 'BITF', 'HUT', 'CIFR', 'BTBT', 'MIGI', 'HOOD', 'SMLR',
    ],
    'context_assets': ['SMH', 'MOD', 'TLT', 'IWM', 'VIX', 'BTCUSD'],
    'has_smh_rsi': True,
    'asset_configs': {
        'RIOT': {
            'seed': 9703, 'prob_threshold': 0.490,
            'tp_pct': 0.06, 'sl_pct': 0.04, 'entry_minute': 385,
            'vix_threshold': 10.0, 'tlt_crash': -0.40, 'spy_ignore': 0.10,
            'stats': {'win_rate': '100.0%', 'avg_pnl': '6.00%', 'freq': '1.1/mo'},
        },
        'IREN': {
            'seed': 9715, 'prob_threshold': 0.515,
            'tp_pct': 0.12, 'sl_pct': 0.10, 'entry_minute': 375,
            'vix_threshold': 10.0, 'tlt_crash': -0.10, 'spy_ignore': 0.04,
            'stats': {'win_rate': '85.7%', 'avg_pnl': '3.23%', 'freq': '2.0/mo'},
        },
        'WULF': {
            'seed': 9715, 'prob_threshold': 0.515,
            'tp_pct': 0.17, 'sl_pct': 0.05, 'entry_minute': 360,
            'vix_threshold': 10.0, 'tlt_crash': -0.10, 'spy_ignore': 0.18,
            'stats': {'win_rate': '100.0%', 'avg_pnl': '6.04%', 'freq': '0.8/mo'},
        },
        'CIFR': {
            'seed': 9714, 'prob_threshold': 0.510,
            'tp_pct': 0.13, 'sl_pct': 0.09, 'entry_minute': 385,
            'vix_threshold': 10.0, 'tlt_crash': -0.10, 'spy_ignore': 0.50,
            'stats': {'win_rate': '83.3%', 'avg_pnl': '3.90%', 'freq': '1.7/mo'},
        },
    },
}

EXP_110 = {
    'model_id': '110_MICRO',
    'label': 'EXP-110 CRYPTO',
    'train_teachers': [
        'MARA', 'IREN', 'WULF', 'APLD', 'RIOT', 'CLSK', 'CORZ', 'MSTR',
        'COIN', 'BITF', 'HUT', 'CIFR', 'BTBT', 'HOOD',
    ],
    'context_assets': ['SMH', 'MOD', 'TLT', 'IWM', 'VIX', 'BTCUSD'],
    'has_smh_rsi': True,
    'asset_configs': {
        'BTBT': {
            'seed': 11006, 'prob_threshold': 0.510,
            'tp_pct': 0.15, 'sl_pct': 0.04, 'entry_minute': 375,
            'vix_threshold': 15.0, 'tlt_crash': 0.00, 'spy_ignore': 0.00,
            'stats': {'win_rate': '100.0%', 'avg_pnl': '2.60%', 'freq': '1.8/mo'},
        },
    },
}

EXP_112 = {
    'model_id': '112_MICRO',
    'label': 'EXP-112 CRYPTO (8% spike)',
    'train_teachers': [
        'MARA', 'IREN', 'WULF', 'APLD', 'RIOT', 'CLSK', 'CORZ', 'MSTR',
        'COIN', 'BITF', 'HUT', 'CIFR', 'BTBT', 'HOOD',
    ],
    'context_assets': ['SMH', 'MOD', 'TLT', 'IWM', 'VIX', 'BTCUSD'],
    'has_smh_rsi': True,
    'asset_configs': {
        'APLD': {
            'seed': 11204, 'prob_threshold': 0.510,
            'tp_pct': 0.08, 'sl_pct': 0.07, 'entry_minute': 379,
            'vix_threshold': 10.0, 'tlt_crash': -0.50, 'spy_ignore': 0.00,
            'stats': {'win_rate': '100.0%', 'avg_pnl': '4.37%', 'freq': '0.6/mo'},
        },
        'CLSK': {
            'seed': 11205, 'prob_threshold': 0.495,
            'tp_pct': 0.19, 'sl_pct': 0.04, 'entry_minute': 383,
            'vix_threshold': 10.0, 'tlt_crash': -0.10, 'spy_ignore': 0.10,
            'stats': {'win_rate': '66.7%', 'avg_pnl': '4.43%', 'freq': '0.8/mo'},
        },
        'HUT': {
            'seed': 11206, 'prob_threshold': 0.510,
            'tp_pct': 0.19, 'sl_pct': 0.09, 'entry_minute': 370,
            'vix_threshold': 10.0, 'tlt_crash': -0.10, 'spy_ignore': 0.00,
            'stats': {'win_rate': '100.0%', 'avg_pnl': '4.09%', 'freq': '0.3/mo'},
        },
    },
}

EXP_103 = {
    'model_id': '103_DEFENSE',
    'label': 'EXP-103 DEFENSE',
    'train_teachers': [
        'AVAV', 'KTOS', 'PLTR', 'BBAI',
        'RKLB', 'ASTS', 'LUNR', 'SPIR',
        'JOBY', 'NNDM', 'MVIS',
    ],
    'context_assets': ['ITA', 'VIX', 'TLT', 'IWM'],
    'has_smh_rsi': False,
    'asset_configs': {
        'RCAT': {
            'seed': 10304, 'prob_threshold': 0.525,
            'tp_pct': 0.16, 'sl_pct': 0.14, 'entry_minute': 365,
            'vix_threshold': 10.0, 'tlt_crash': -0.10, 'spy_ignore': 0.00,
            'stats': {'win_rate': '100.0%', 'avg_pnl': '11.09%', 'freq': '0.8/mo'},
        },
        'LUNR': {
            'seed': 10311, 'prob_threshold': 0.465,
            'tp_pct': 0.21, 'sl_pct': 0.05, 'entry_minute': 365,
            'vix_threshold': 16.0, 'tlt_crash': -0.10, 'spy_ignore': 0.00,
            'stats': {'win_rate': '66.7%', 'avg_pnl': '6.19%', 'freq': '2.4/mo'},
        },
        'RKLB': {
            'seed': 10308, 'prob_threshold': 0.510,
            'tp_pct': 0.08, 'sl_pct': 0.04, 'entry_minute': 385,
            'vix_threshold': 10.0, 'tlt_crash': -0.10, 'spy_ignore': 0.00,
            'stats': {'win_rate': '100.0%', 'avg_pnl': '5.60%', 'freq': '0.5/mo'},
        },
        'SPIR': {
            'seed': 10310, 'prob_threshold': 0.310,
            'tp_pct': 0.12, 'sl_pct': 0.03, 'entry_minute': 360,
            'vix_threshold': 16.0, 'tlt_crash': -0.10, 'spy_ignore': 0.10,
            'stats': {'win_rate': '55.6%', 'avg_pnl': '4.53%', 'freq': '2.4/mo'},
        },
        'PLTR': {
            'seed': 10311, 'prob_threshold': 0.465,
            'tp_pct': 0.07, 'sl_pct': 0.06, 'entry_minute': 360,
            'vix_threshold': 10.0, 'tlt_crash': -0.20, 'spy_ignore': 0.10,
            'stats': {'win_rate': '100.0%', 'avg_pnl': '3.22%', 'freq': '0.9/mo'},
        },
        'JOBY': {
            'seed': 10307, 'prob_threshold': 0.510,
            'tp_pct': 0.12, 'sl_pct': 0.08, 'entry_minute': 380,
            'vix_threshold': 10.0, 'tlt_crash': -0.10, 'spy_ignore': 0.00,
            'stats': {'win_rate': '50.0%', 'avg_pnl': '3.15%', 'freq': '1.1/mo'},
        },
        'ASTS': {
            'seed': 10308, 'prob_threshold': 0.510,
            'tp_pct': 0.10, 'sl_pct': 0.06, 'entry_minute': 390,
            'vix_threshold': 15.0, 'tlt_crash': 0.00, 'spy_ignore': 0.00,
            'stats': {'win_rate': '57.1%', 'avg_pnl': '2.62%', 'freq': '1.9/mo'},
        },
        'KTOS': {
            'seed': 10312, 'prob_threshold': 0.510,
            'tp_pct': 0.05, 'sl_pct': 0.04, 'entry_minute': 385,
            'vix_threshold': 10.0, 'tlt_crash': -0.10, 'spy_ignore': 0.00,
            'stats': {'win_rate': '71.4%', 'avg_pnl': '2.57%', 'freq': '2.0/mo'},
        },
    },
}

# Build derived fields for each experiment
for _exp in (EXP_97, EXP_110, EXP_112, EXP_103):
    _exp['context_cols'] = _build_context_cols(
        _exp['context_assets'], _exp['has_smh_rsi'])
    _exp['all_cols'] = ASSET_COLS + _exp['context_cols']
    _idx = {t: i for i, t in enumerate(_exp['train_teachers'])}
    if 'BITF' in _idx:
        _idx['KEEL'] = _idx['BITF']
    _exp['asset_index'] = _idx
    _exp['n_assets'] = len(_exp['train_teachers']) + 1
    _exp['other_idx'] = len(_exp['train_teachers'])
    _exp['stats_path'] = f"Data/Inference_v2/scaling_stats_experiment_{_exp['model_id']}.json"

EXPERIMENTS = [EXP_97, EXP_110, EXP_112, EXP_103]

# ==============================================================================
# GCS CLIENT
# ==============================================================================
try:
    from google.colab import auth
    auth.authenticate_user()
except Exception:
    pass

client = storage.Client()

# ==============================================================================
# DATA LOADING
# ==============================================================================
_data_cache = {}


def load_historical_base(ticker):
    if ticker in _data_cache:
        return _data_cache[ticker].copy()

    clean = ticker.replace("I:", "").replace("X:", "")
    file_ticker = TICKER_FILE_MAP.get(clean, clean)
    blob = client.bucket(BUCKET_NAME).blob(
        f"{INFERENCE_DIR}/{file_ticker}_master_inference.parquet")
    if not blob.exists():
        print(f"   Missing {file_ticker}_master_inference.parquet in GCS.")
        return None

    df = pd.read_parquet(io.BytesIO(blob.download_as_bytes()))

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

    cutoff_dt = pd.Timestamp.now(tz='UTC') - timedelta(days=BACKTEST_DAYS + 15)
    df = df[df.index >= cutoff_dt]

    _data_cache[ticker] = df
    return df.copy()


# ==============================================================================
# HEAL DATA — matches daily_score_combined.py (WITH resample)
# ==============================================================================
def heal_data(df):
    df = df.resample('1min').asfreq()
    for c in ['Open', 'High', 'Low', 'Close', 'Spread', 'OBI', 'bsize', 'asize']:
        if c in df.columns:
            df[c] = df[c].ffill()
    for c in ['Volume', 'Total_Vol', 'Volume_Delta', 'Trade_Count', 'Delta_Pct']:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    return df


# ==============================================================================
# MARKET CONTEXT (per-experiment)
# ==============================================================================
def build_market_context(exp):
    frames = []
    print(f"   Building context for {exp['label']}...")
    for ticker in exp['context_assets']:
        df = load_historical_base(ticker)
        if df is None or df.empty:
            print(f"      Could not load context asset {ticker}")
            continue

        df = heal_data(df)
        daily_c = df['Close'].resample('D').last().ffill().shift(1).reindex(
            df.index, method='ffill')
        cols = {f'{ticker}_Anchor_ROC': (df['Close'] - daily_c) / daily_c}
        for h in ROC_HORIZONS:
            cols[f'{ticker}_ROC_{h}m'] = df['Close'].pct_change(h).fillna(0)
        if exp['has_smh_rsi'] and ticker == 'SMH':
            rsi = ta.rsi(df['Close'], length=14)
            cols['SMH_RSI'] = (rsi.fillna(50) / 100.0
                               if rsi is not None
                               else pd.Series(0.5, index=df.index))

        frames.append(pd.DataFrame(cols, index=df.index))
        del df, daily_c
        gc.collect()

    if not frames:
        raise RuntimeError(f"No context assets loaded for {exp['label']}.")
    return pd.concat(frames, axis=1).ffill().fillna(0)


# ==============================================================================
# FEATURE ENGINEERING
# ==============================================================================
def engineer_asset(ticker, market_df):
    df = load_historical_base(ticker)
    if df is None or df.empty:
        return None

    df = heal_data(df)
    daily_c = df['Close'].resample('D').last().shift(1).reindex(
        df.index, method='ffill')
    df['Gap_Pct'] = (df['Open'] - daily_c) / daily_c
    df['PM_High'] = df.groupby(df.index.date)['High'].cummax()
    df['Dist_to_PM_High'] = (df['Close'] - df['PM_High']) / (df['PM_High'] + 1e-9)

    df['RSI_14'] = ta.rsi(df['Close'], length=14)
    df['RSI_60'] = ta.rsi(df['Close'], length=60)
    df['RVOL']   = df['Volume'] / df['Volume'].rolling(50).mean().clip(lower=1e-9)
    df['ATR_14'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['Compression'] = (df['High'] - df['Low']) / (df['ATR_14'] + 1e-9)
    for h in ROC_HORIZONS:
        df[f'ROC_{h}m'] = df['Close'].pct_change(h)

    if market_df is not None and not market_df.empty:
        df = df.join(market_df, how='left').ffill().fillna(0)

    df_pst = df.index.tz_convert(PST)
    df['Time_Pos'] = np.clip((df_pst.hour * 60 + df_pst.minute - 390) / 390, 0.0, 1.0)

    return df.replace([np.inf, -np.inf], np.nan).fillna(0)


# ==============================================================================
# SCALING
# ==============================================================================
def apply_scaling(df, stats, all_cols):
    for col in all_cols:
        if col == 'Time_Pos':
            continue
        if col in stats:
            df[col] = ((df[col] - stats[col]['med'])
                       / stats[col]['iqr']).clip(-10, 10).astype('float32')
    return df


# ==============================================================================
# MODEL ARCHITECTURE (parameterized — from daily_score_combined.py)
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


def build_model(n_context_cols, n_assets):
    in_a  = Input(shape=(TIME_STEPS, len(ASSET_COLS)), name='asset_in')
    in_c  = Input(shape=(n_context_cols,),             name='context_in')
    in_id = Input(shape=(1,), dtype=tf.int32,          name='asset_id')
    x = transformer_encoder(Dense(128)(in_a), 128, 4, 128)
    x = GlobalAveragePooling1D()(transformer_encoder(x, 128, 4, 128))
    y = Dense(128, activation='relu')(BatchNormalization()(in_c))
    e = Embedding(n_assets, EMBED_DIM, name='asset_embed')(in_id)
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
    print(f"      Loading model seed {seed} ({model_id})...")
    blob_path  = f"{GCS_MODEL_PATH}/UNIVERSAL_S{seed}_M{model_id}.h5"
    local_path = f"/tmp/bt_combined_{seed}_{model_id}.h5"
    if not os.path.exists(local_path):
        blob = client.bucket(BUCKET_NAME).blob(blob_path)
        if not blob.exists():
            print(f"      Model missing: {blob_path}")
            return None
        blob.download_to_filename(local_path)
    tf.get_logger().setLevel('ERROR')
    model = build_model(len(exp['context_cols']), exp['n_assets'])
    model.load_weights(local_path)
    _model_cache[key] = model
    return model


# ==============================================================================
# THE BACKTESTING ENGINE
# ==============================================================================
def run_backtest(exp, spy_raw, tlt_raw, vix_raw):
    model_id     = exp['model_id']
    label        = exp['label']
    asset_configs = exp['asset_configs']
    context_cols = exp['context_cols']
    all_cols     = exp['all_cols']

    print(f"\n{'='*60}")
    print(f"   {label}  ({model_id})")
    print(f"   Tickers: {list(asset_configs.keys())}")
    print(f"{'='*60}")

    stats_blob = client.bucket(BUCKET_NAME).blob(exp['stats_path'])
    if not stats_blob.exists():
        print(f"   Stats missing: {exp['stats_path']}")
        return [], []
    stats = json.loads(stats_blob.download_as_string())
    print(f"   Loaded {len(stats)} scaling features")

    market_df = build_market_context(exp)

    needed_seeds = sorted(set(cfg['seed'] for cfg in asset_configs.values()))
    for seed in needed_seeds:
        get_model(exp, seed)
    print(f"   Models ready: {needed_seeds}")

    results_summary = []
    trade_log = []

    for ticker, config in asset_configs.items():
        print(f"\n   Testing {ticker}...")

        seed = config['seed']
        model = get_model(exp, seed)
        if model is None:
            continue
        asset_id = exp['asset_index'].get(ticker, exp['other_idx'])

        df = engineer_asset(ticker, market_df)
        if df is None or df.empty:
            print(f"      Could not load data for {ticker}.")
            continue

        df_unscaled = df.copy()
        df = apply_scaling(df, stats, all_cols)

        entry_minute = config['entry_minute']
        entry_h = entry_minute // 60
        entry_m = entry_minute % 60

        tp_pct = config['tp_pct']
        sl_pct = config['sl_pct']
        avg_pnl_pct = float(config['stats']['avg_pnl'].strip('%')) / 100.0

        cm = {'strict': {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0},
              'avg':    {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}}
        trades_taken = 0
        trades_blocked = 0
        skipped_synthetic = 0
        total_pnl = 0.0
        win_trades = 0
        loss_trades = 0

        test_start_dt = pd.Timestamp.now(tz=PST) - timedelta(days=BACKTEST_DAYS)
        df_test = df[df.index.tz_convert(PST) >= test_start_dt]
        unique_dates = np.unique(df_test.index.tz_convert(PST).date)

        for d in unique_dates:
            if d.weekday() >= 5:
                continue

            entry_dt_pst = PST.localize(datetime(d.year, d.month, d.day, entry_h, entry_m))
            entry_dt_utc = entry_dt_pst.astimezone(pytz.utc)

            if entry_dt_utc not in df.index:
                continue

            idx = df.index.get_loc(entry_dt_utc)
            if idx < TIME_STEPS:
                continue

            if pd.isna(df_unscaled.iloc[idx]['Close']):
                continue

            if df_unscaled['Volume'].iloc[idx] == 0:
                skipped_synthetic += 1
                continue

            # --- Regime filter: trailing 30-min median ---
            try:
                spy_prev = spy_raw[(spy_raw.index.date < d)]['Close'].iloc[-1]
                tlt_prev = tlt_raw[(tlt_raw.index.date < d)]['Close'].iloc[-1]

                window_start = entry_dt_pst - timedelta(minutes=30)
                vix_bars = vix_raw[(vix_raw.index >= window_start) & (vix_raw.index <= entry_dt_pst)]
                spy_bars = spy_raw[(spy_raw.index >= window_start) & (spy_raw.index <= entry_dt_pst)]
                tlt_bars = tlt_raw[(tlt_raw.index >= window_start) & (tlt_raw.index <= entry_dt_pst)]

                vix_now = float(vix_bars['Close'].median()) if len(vix_bars) else None
                spy_now = float(spy_bars['Close'].median()) if len(spy_bars) else None
                tlt_now = float(tlt_bars['Close'].median()) if len(tlt_bars) else None

                if vix_now is None or spy_now is None or tlt_now is None:
                    is_trap = False
                else:
                    spy_gap = ((spy_now - spy_prev) / spy_prev) * 100
                    tlt_gap = ((tlt_now - tlt_prev) / tlt_prev) * 100

                    vix_bad = vix_now > config['vix_threshold']
                    tlt_bad = tlt_gap < config['tlt_crash']
                    spy_bad = spy_gap > config['spy_ignore']

                    is_trap = vix_bad and tlt_bad and spy_bad
            except Exception:
                is_trap = False

            # --- Prediction (anchored to exact entry bar) ---
            xa = df[ASSET_COLS].values[idx - TIME_STEPS:idx].astype('float32')
            xc = df[context_cols].values[idx - 1].astype('float32')
            XA_b  = np.expand_dims(xa, axis=0)
            XC_b  = np.expand_dims(xc, axis=0)
            XID_b = np.array([[asset_id]], dtype=np.int32)

            prob = float(model(
                {'asset_in': XA_b, 'context_in': XC_b, 'asset_id': XID_b},
                training=False
            ).numpy().flatten()[0])

            thresh = config['prob_threshold']

            # --- Decision gate ---
            if prob >= thresh:
                if is_trap:
                    trades_blocked += 1
                    signal = 0
                else:
                    signal = 1
            else:
                signal = 0

            entry_price = float(df_unscaled['Close'].iloc[idx])

            # --- Intraday trade simulation: exit at 1:00 PM PST ---
            exit_dt_utc = PST.localize(
                datetime(d.year, d.month, d.day, 13, 0)
            ).astimezone(pytz.utc)
            df_intraday = df_unscaled.loc[
                entry_dt_utc + timedelta(minutes=1):exit_dt_utc
            ]

            if df_intraday.empty:
                continue

            sl_price        = entry_price * (1 - sl_pct)
            tp_price_strict = entry_price * (1 + tp_pct)
            tp_price_avg    = entry_price * (1 + avg_pnl_pct)

            actual_pnl    = 0.0
            hit_strict    = 0
            hit_avg       = 0
            trade_active  = True
            exit_reason   = 'EOD'
            exit_price    = float(df_intraday['Close'].iloc[-1])
            max_favorable = 0.0
            max_adverse   = 0.0

            for _, row in df_intraday.iterrows():
                if not trade_active:
                    break

                bar_pnl_high = (row['High'] - entry_price) / entry_price
                bar_pnl_low  = (row['Low'] - entry_price) / entry_price
                max_favorable = max(max_favorable, bar_pnl_high)
                max_adverse   = min(max_adverse, bar_pnl_low)

                if row['Low'] <= sl_price:
                    actual_pnl   = -sl_pct
                    exit_reason  = 'SL'
                    exit_price   = sl_price
                    trade_active = False
                elif row['High'] >= tp_price_strict:
                    hit_strict   = 1
                    hit_avg      = 1
                    actual_pnl   = tp_pct
                    exit_reason  = 'TP'
                    exit_price   = tp_price_strict
                    trade_active = False
                elif row['High'] >= tp_price_avg:
                    hit_avg = 1

            if trade_active:
                actual_pnl = (df_intraday['Close'].iloc[-1] - entry_price) / entry_price

            trade_record = {
                'ticker': ticker, 'experiment': model_id,
                'date': str(d), 'prob': round(prob, 4),
                'signal': signal, 'blocked': is_trap,
                'entry_price': round(entry_price, 4),
                'exit_price': round(exit_price, 4),
                'exit_reason': exit_reason,
                'pnl': round(actual_pnl * 100, 2),
                'max_favorable': round(max_favorable * 100, 2),
                'max_adverse': round(max_adverse * 100, 2),
            }
            trade_log.append(trade_record)

            if signal == 1:
                trades_taken += 1
                total_pnl += actual_pnl
                if actual_pnl > 0:
                    win_trades += 1
                else:
                    loss_trades += 1
                if hit_strict:
                    cm['strict']['TP'] += 1
                else:
                    cm['strict']['FP'] += 1
                if hit_avg:
                    cm['avg']['TP'] += 1
                else:
                    cm['avg']['FP'] += 1
            else:
                if hit_strict:
                    cm['strict']['FN'] += 1
                else:
                    cm['strict']['TN'] += 1
                if hit_avg:
                    cm['avg']['FN'] += 1
                else:
                    cm['avg']['TN'] += 1

        def calc_metrics(c_dict):
            prec = c_dict['TP'] / (c_dict['TP'] + c_dict['FP']) if (c_dict['TP'] + c_dict['FP']) > 0 else 0
            rec  = c_dict['TP'] / (c_dict['TP'] + c_dict['FN']) if (c_dict['TP'] + c_dict['FN']) > 0 else 0
            return prec, rec

        prec_strict, rec_strict = calc_metrics(cm['strict'])
        prec_avg, rec_avg       = calc_metrics(cm['avg'])
        avg_trade_pnl = total_pnl / trades_taken if trades_taken > 0 else 0
        win_rate      = win_trades / trades_taken if trades_taken > 0 else 0

        results_summary.append({
            'Asset': ticker, 'Experiment': model_id,
            'Trades': trades_taken, 'Blocked': trades_blocked,
            'Synthetic': skipped_synthetic,
            'P_Strict': prec_strict, 'R_Strict': rec_strict,
            'P_Avg': prec_avg, 'R_Avg': rec_avg,
            'WinRate': win_rate, 'TotalPnL': total_pnl, 'AvgPnL': avg_trade_pnl,
            'Wins': win_trades, 'Losses': loss_trades,
        })

    return results_summary, trade_log


# ==============================================================================
# DASHBOARD
# ==============================================================================
def print_dashboard(all_results, all_trades):
    W = 160
    print("\n" + "=" * W)
    print(f"  {BACKTEST_DAYS}-DAY COMBINED BACKTEST RESULTS  (16 deployed tickers)")
    print(f"  EXP-97 (Crypto 10%) + EXP-110 (Crypto regime) + EXP-112 (Crypto 8%) + EXP-103 (Defense)")
    print("=" * W)
    print(f"  {'ASSET':<6} {'EXP':<12} | {'TRADES':<6} | {'W/L':<7} | {'BLK':<4} | {'SYN':<4} | "
          f"{'STRICT TP':<22} | {'AVG PNL':<22} | {'ACTUAL PNL':<30}")
    print(f"  {'':<6} {'':<12} | {'TAKEN':<6} | {'':<7} | {'':<4} | {'':<4} | "
          f"{'Prec.':<10} {'Recall':<10} | {'Prec.':<10} {'Recall':<10} | "
          f"{'WinRate':<8} {'Total':<10} {'Avg':<8}")
    print("  " + "-" * (W - 2))

    portfolio_pnl = 0.0
    portfolio_trades = 0
    portfolio_wins = 0
    portfolio_losses = 0

    current_exp = None
    for r in all_results:
        if r['Experiment'] != current_exp:
            if current_exp is not None:
                print("  " + "-" * (W - 2))
            current_exp = r['Experiment']

        s_prec = f"{r['P_Strict']:.1%}"
        s_rec  = f"{r['R_Strict']:.1%}"
        a_prec = f"{r['P_Avg']:.1%}"
        a_rec  = f"{r['R_Avg']:.1%}"
        wr     = f"{r['WinRate']:.1%}"
        tot    = f"{r['TotalPnL']:.1%}"
        avg    = f"{r['AvgPnL']:.2%}"
        wl     = f"{r['Wins']}/{r['Losses']}"

        portfolio_pnl    += r['TotalPnL']
        portfolio_trades += r['Trades']
        portfolio_wins   += r['Wins']
        portfolio_losses += r['Losses']

        print(f"  {r['Asset']:<6} {r['Experiment']:<12} | {r['Trades']:<6} | {wl:<7} | "
              f"{r['Blocked']:<4} | {r['Synthetic']:<4} | "
              f"{s_prec:<10} {s_rec:<10} | {a_prec:<10} {a_rec:<10} | "
              f"{wr:<8} {tot:<10} {avg:<8}")

    print("  " + "-" * (W - 2))
    port_avg = portfolio_pnl / portfolio_trades if portfolio_trades > 0 else 0
    port_wr  = portfolio_wins / portfolio_trades if portfolio_trades > 0 else 0
    print(f"  {'TOTAL':<6} {'PORTFOLIO':<12} | {portfolio_trades:<6} | "
          f"{portfolio_wins}/{portfolio_losses:<5} | {'':<4} | {'':<4} | "
          f"{'':<10} {'':<10} | {'':<10} {'':<10} | "
          f"{port_wr:.1%}{'':<3} {portfolio_pnl:.1%}{'':>5} {port_avg:.2%}")
    print("=" * W)

    # --- Trade Log ---
    signal_trades = [t for t in all_trades if t['signal'] == 1]
    if signal_trades:
        print(f"\n  TRADE LOG ({len(signal_trades)} executed trades)")
        print("  " + "-" * 140)
        print(f"  {'Date':<12} {'Ticker':<6} {'Exp':<12} {'Prob':>6} {'Entry':>9} {'Exit':>9} "
              f"{'Reason':<6} {'PnL%':>7} {'MFE%':>7} {'MAE%':>7} {'Blocked':<7}")
        print("  " + "-" * 140)
        for t in sorted(signal_trades, key=lambda x: (x['date'], x['ticker'])):
            pnl_color = '\033[92m' if t['pnl'] > 0 else '\033[91m'
            print(f"  {t['date']:<12} {t['ticker']:<6} {t['experiment']:<12} {t['prob']:>6.3f} "
                  f"${t['entry_price']:>8.2f} ${t['exit_price']:>8.2f} "
                  f"{t['exit_reason']:<6} {pnl_color}{t['pnl']:>+6.2f}%\033[0m "
                  f"{t['max_favorable']:>+6.2f}% {t['max_adverse']:>+6.2f}% "
                  f"{'YES' if t['blocked'] else '':<7}")
        print("  " + "-" * 140)

        pnls = [t['pnl'] for t in signal_trades]
        mfes = [t['max_favorable'] for t in signal_trades]
        maes = [t['max_adverse'] for t in signal_trades]
        print(f"\n  Trade Efficiency Stats:")
        print(f"    Avg PnL:  {np.mean(pnls):+.2f}%   Median: {np.median(pnls):+.2f}%")
        print(f"    Avg MFE:  {np.mean(mfes):+.2f}%   (max favorable excursion)")
        print(f"    Avg MAE:  {np.mean(maes):+.2f}%   (max adverse excursion)")
        if np.mean(mfes) > 0:
            capture = np.mean(pnls) / np.mean(mfes) * 100
            print(f"    Capture:  {capture:.1f}%  (PnL / MFE)")

    # --- Blocked Trades ---
    blocked_signals = [t for t in all_trades if t['signal'] == 0 and t['blocked']]
    if blocked_signals:
        print(f"\n  BLOCKED TRADES ANALYSIS ({len(blocked_signals)} blocked)")
        print("  " + "-" * 80)
        blocked_pnls = [t['pnl'] for t in blocked_signals]
        would_have_won = sum(1 for p in blocked_pnls if p > 0)
        print(f"    Blocked trades that would have won:  {would_have_won}/{len(blocked_signals)}")
        print(f"    Avg PnL if taken: {np.mean(blocked_pnls):+.2f}%")
        print(f"    Filter saved:     {-sum(p for p in blocked_pnls if p < 0):.2f}% total loss avoided")


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print(f"COMBINED {BACKTEST_DAYS}-DAY BACKTEST — ALL DEPLOYED EXPERIMENTS")
    print(f"   Experiments: {[e['label'] for e in EXPERIMENTS]}")
    print(f"   Total tickers: {sum(len(e['asset_configs']) for e in EXPERIMENTS)}")

    print("\nPre-loading SPY, TLT, VIX for regime filter...")
    spy_raw = load_historical_base("SPY")
    tlt_raw = load_historical_base("TLT")
    vix_raw = load_historical_base("VIX")
    if spy_raw is not None:
        spy_raw = heal_data(spy_raw)
        spy_raw.index = spy_raw.index.tz_convert(PST)
    if tlt_raw is not None:
        tlt_raw = heal_data(tlt_raw)
        tlt_raw.index = tlt_raw.index.tz_convert(PST)
    if vix_raw is not None:
        vix_raw = heal_data(vix_raw)
        vix_raw.index = vix_raw.index.tz_convert(PST)

    all_results = []
    all_trades = []

    for exp in EXPERIMENTS:
        results, trades = run_backtest(exp, spy_raw, tlt_raw, vix_raw)
        all_results.extend(results)
        all_trades.extend(trades)

    print_dashboard(all_results, all_trades)
    print(f"\nDone. {BACKTEST_DAYS}-day combined backtest complete.")


if __name__ == "__main__":
    main()
