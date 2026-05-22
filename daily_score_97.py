"""
Stage 2 — Daily Scoring Script for Exp-97
==========================================
Loads best-seed exp-97 models, engineers features from updated
master_inference parquets, scores each ticker, prints a dashboard,
and logs predictions to BigQuery.

Tickers: 16 training teachers + extended scoring pool.

Key differences from exp-95 (must match exp_97.py training exactly):
  - 16 training teachers (expanded crypto cluster).
  - Uses pandas_ta for technical indicators to ensure exact alignment.
  - Model has 17 asset embeddings (16 teachers + 1 other).

Run AFTER daily_data.py (or daily_data_97.py) has finished updating GCS.
Usage:  python daily_score_97.py
"""

import os
import sys
import json
import subprocess

try:
    import pandas_ta as ta
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas_ta", "--no-deps", "--quiet"])
    import pandas_ta as ta

import numpy as np
import pandas as pd
import pytz
from datetime import datetime, timedelta

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Dropout, GlobalAveragePooling1D,
    LayerNormalization, MultiHeadAttention, Add,
    BatchNormalization, Concatenate, Embedding, Flatten,
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from google.cloud import storage as gcs_storage

try:
    from google.cloud import bigquery
except Exception:
    bigquery = None
    print("   ⚠️  BigQuery not available — logging will be skipped")

# ==============================================================================
# ⚙️  CONFIGURATION
# ==============================================================================
MODEL_ID    = "97_MICRO"
BUCKET_NAME = "time-series-bucket-ys"

# Context assets — same as training
CONTEXT_ASSETS = ['SMH', 'MOD', 'TLT', 'IWM', 'VIX', 'BTCUSD']

# 4 ROC horizons
ROC_HORIZONS = [5, 30, 60, 120]

MICRO_COLS = ['Spread', 'OBI', 'Volume_Delta', 'Trade_Count', 'Delta_Pct', 'bsize', 'asize', 'Total_Vol']

ASSET_COLS = ['Gap_Pct', 'RSI_14', 'RSI_60', 'RVOL', 'Compression', 'ATR_14', 'Dist_to_PM_High'] + \
             [f'ROC_{h}m' for h in ROC_HORIZONS] + MICRO_COLS

CONTEXT_COLS = []
for _t in CONTEXT_ASSETS:
    CONTEXT_COLS.append(f'{_t}_Anchor_ROC')
    for h in ROC_HORIZONS:
        CONTEXT_COLS.append(f'{_t}_ROC_{h}m')
CONTEXT_COLS += ['SMH_RSI', 'Time_Pos']

TIME_STEPS = 240

# 16 Training Teachers from exp-97
# NOTE: SMLR delisted ~Jan 2026. Kept here ONLY to preserve embedding indices
# (removing it would shift all indices and break saved model weights).
TRAIN_TEACHERS = [
    'MARA', 'IREN', 'WULF', 'APLD', 'RIOT', 'CLSK', 'CORZ', 'MSTR', 'COIN',
    'BITF', 'HUT', 'CIFR', 'BTBT', 'MIGI', 'HOOD', 'SMLR',
]

ASSET_INDEX    = {ticker: i for i, ticker in enumerate(TRAIN_TEACHERS)}
ASSET_INDEX['KEEL'] = ASSET_INDEX['BITF']  # BITF renamed to KEEL on 2026-04-06
N_ASSETS       = len(TRAIN_TEACHERS) + 1   # 17
OTHER_IDX      = len(TRAIN_TEACHERS)       # 16
EMBED_DIM      = 16

TAIL_DAYS = 7  # Need 7 days for feature warmup (RSI, RVOL, ATR) + resample density

GCS_INFERENCE_DIR = f"gs://{BUCKET_NAME}/Data/Inference"
GCS_MODEL_PATH    = f"Algos/to_deploy/Saved_models/UNIVERSAL/Direction"
GCS_STATS_PATH    = f"Data/Inference/scaling_stats_experiment_{MODEL_ID}.json"

BQ_TABLE = "ys-time-series.scoring_logs.live_predictions"

# ==============================================================================
# 🎯  ASSET CONFIGS
# Stats from 180-day backtesting (win_rate=Strict TP WinRate, avg_pnl=BT AvgPnL)
# ==============================================================================
ASSET_CONFIGS = {
    'MIGI': {'seed': 9714, 'prob_threshold': 0.555, 'tp_pct': 0.45, 'sl_pct': 0.12, 'entry_minute': 390, 'vix_threshold': 10.0, 'tlt_crash': -0.10, 'spy_ignore': 0.00, 'stats': {'win_rate': '25.0%', 'avg_pnl': '11.63%', 'freq': '1.3/mo'}},
    'KEEL': {'seed': 9712, 'prob_threshold': 0.540, 'tp_pct': 0.11, 'sl_pct': 0.16, 'entry_minute': 385, 'vix_threshold': 16.0, 'tlt_crash': -0.10, 'spy_ignore': 0.00, 'stats': {'win_rate': '42.9%', 'avg_pnl': '5.32%', 'freq': '1.2/mo'}},
    'HUT':  {'seed': 9703, 'prob_threshold': 0.490, 'tp_pct': 0.19, 'sl_pct': 0.04, 'entry_minute': 385, 'vix_threshold': 10.0, 'tlt_crash': -0.10, 'spy_ignore': 0.00, 'stats': {'win_rate': '12.5%', 'avg_pnl': '5.86%', 'freq': '1.3/mo'}},
    'CIFR': {'seed': 9712, 'prob_threshold': 0.520, 'tp_pct': 0.14, 'sl_pct': 0.08, 'entry_minute': 385, 'vix_threshold': 10.0, 'tlt_crash': -0.10, 'spy_ignore': 0.10, 'stats': {'win_rate': '14.3%', 'avg_pnl': '3.50%', 'freq': '1.2/mo'}},
    'WULF': {'seed': 9710, 'prob_threshold': 0.515, 'tp_pct': 0.17, 'sl_pct': 0.06, 'entry_minute': 360, 'vix_threshold': 10.0, 'tlt_crash': -0.10, 'spy_ignore': 0.02, 'stats': {'win_rate': '16.7%', 'avg_pnl': '6.26%', 'freq': '1.0/mo'}},
    'BTBT': {'seed': 9711, 'prob_threshold': 0.515, 'tp_pct': 0.15, 'sl_pct': 0.08, 'entry_minute': 360, 'vix_threshold': 15.0, 'tlt_crash':  0.00, 'spy_ignore': 0.00, 'stats': {'win_rate': '20.0%', 'avg_pnl': '6.10%', 'freq': '0.8/mo'}},
    'RIOT': {'seed': 9702, 'prob_threshold': 0.500, 'tp_pct': 0.06, 'sl_pct': 0.04, 'entry_minute': 385, 'vix_threshold': 16.0, 'tlt_crash': -0.10, 'spy_ignore': 0.00, 'stats': {'win_rate': '100.0%', 'avg_pnl': '6.00%', 'freq': '1.2/mo'}},
    'IREN': {'seed': 9701, 'prob_threshold': 0.515, 'tp_pct': 0.12, 'sl_pct': 0.11, 'entry_minute': 360, 'vix_threshold': 10.0, 'tlt_crash': -0.10, 'spy_ignore': 0.00, 'stats': {'win_rate': '75.0%', 'avg_pnl': '5.71%', 'freq': '2.0/mo'}},
    # 'SMLR' — REMOVED: delisted ~Jan 2026, no data available
    'CLSK': {'seed': 9707, 'prob_threshold': 0.510, 'tp_pct': 0.19, 'sl_pct': 0.04, 'entry_minute': 365, 'vix_threshold': 10.0, 'tlt_crash': -0.10, 'spy_ignore': 0.00, 'stats': {'win_rate': '0.0%', 'avg_pnl': '3.68%', 'freq': '1.2/mo'}},
    'APLD': {'seed': 9715, 'prob_threshold': 0.550, 'tp_pct': 0.07, 'sl_pct': 0.12, 'entry_minute': 390, 'vix_threshold': 16.0, 'tlt_crash': -0.10, 'spy_ignore': 0.00, 'stats': {'win_rate': '25.0%', 'avg_pnl': '3.93%', 'freq': '1.3/mo'}},
    # 'MARA': {'seed': 9703, 'prob_threshold': 0.470, 'tp_pct': 0.18, 'sl_pct': 0.06, 'entry_minute': 385, 'vix_threshold': 16.0, 'tlt_crash': -0.10, 'spy_ignore': 0.00, 'stats': {'win_rate': '0.0%', 'avg_pnl': '0.58%', 'freq': '1.5/mo'}},  # BT: 0.58%
    'MSTR': {'seed': 9711, 'prob_threshold': 0.510, 'tp_pct': 0.17, 'sl_pct': 0.08, 'entry_minute': 375, 'vix_threshold': 10.0, 'tlt_crash': -0.10, 'spy_ignore': 0.00, 'stats': {'win_rate': '10.0%', 'avg_pnl': '2.19%', 'freq': '1.7/mo'}},
    'SMCI': {'seed': 9706, 'prob_threshold': 0.470, 'tp_pct': 0.08, 'sl_pct': 0.04, 'entry_minute': 390, 'vix_threshold': 16.0, 'tlt_crash': -0.10, 'spy_ignore': 0.00, 'stats': {'win_rate': '30.0%', 'avg_pnl': '2.58%', 'freq': '1.7/mo'}},
    'COIN': {'seed': 9705, 'prob_threshold': 0.460, 'tp_pct': 0.08, 'sl_pct': 0.06, 'entry_minute': 370, 'vix_threshold': 10.0, 'tlt_crash': -0.10, 'spy_ignore': 0.10, 'stats': {'win_rate': '23.1%', 'avg_pnl': '2.86%', 'freq': '2.2/mo'}},
    # 'HOOD': {'seed': 9706, 'prob_threshold': 0.455, 'tp_pct': 0.10, 'sl_pct': 0.06, 'entry_minute': 385, 'vix_threshold': 16.0, 'tlt_crash': -0.10, 'spy_ignore': 0.00, 'stats': {'win_rate': '7.1%', 'avg_pnl': '1.15%', 'freq': '2.3/mo'}},  # BT: 1.15%
    # 'TSLA': {'seed': 9704, 'prob_threshold': 0.510, 'tp_pct': 0.06, 'sl_pct': 0.01, 'entry_minute': 375, 'vix_threshold': 16.0, 'tlt_crash': -0.10, 'spy_ignore': 0.00, 'stats': {'win_rate': '20.0%', 'avg_pnl': '1.26%', 'freq': '2.5/mo'}},  # BT: 1.26%
    'VRT':  {'seed': 9709, 'prob_threshold': 0.460, 'tp_pct': 0.10, 'sl_pct': 0.10, 'entry_minute': 390, 'vix_threshold': 15.0, 'tlt_crash':  0.00, 'spy_ignore': 0.00, 'stats': {'win_rate': '0.0%', 'avg_pnl': '1.92%', 'freq': '1.2/mo'}},
}

# ==============================================================================
# ☁️  GCS HELPERS
# ==============================================================================
gcs_client = gcs_storage.Client()

def _load_master(ticker):
    """Load master_inference parquet for a ticker."""
    clean  = ticker.replace("I:", "").replace("X:", "")
    uri    = f"{GCS_INFERENCE_DIR}/{clean}_master_inference.parquet"
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
            'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume',
        }
        lower_map = {k.lower(): v for k, v in std_map.items()}
        df.columns = [lower_map.get(str(c).strip().lower(), str(c).strip()) for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated(keep='first')].copy()
        if ticker in CONTEXT_ASSETS or ticker == 'SPY':
            for c in MICRO_COLS:
                if c not in df.columns:
                    df[c] = 0.0
        df = df[df.index.notna()].sort_index()
        if len(df) > 0:
            print(f"      🔍 [DEBUG] {ticker}: loaded {len(df)} rows. Range: {df.index.min()} to {df.index.max()}")
        else:
            print(f"      🔍 [DEBUG] {ticker}: loaded file is EMPTY")

        cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=TAIL_DAYS)
        df_filtered = df[df.index >= cutoff]
        print(f"      🔍 [DEBUG] {ticker}: after {TAIL_DAYS}-day cutoff, {len(df_filtered)} rows remain.")
        if df_filtered.empty:
            print(f"      ⚠️  {ticker}: no recent data — skipping")
            return None
        return df_filtered
    except Exception as e:
        print(f"   ❌ Failed to load {ticker}: {e}")
        return None

def _heal(df):
    df = df.resample('1min').asfreq()
    for c in ['Open', 'High', 'Low', 'Close', 'Spread', 'OBI', 'bsize', 'asize']:
        if c in df.columns: df[c] = df[c].ffill()
    for c in ['Volume', 'Total_Vol', 'Volume_Delta', 'Trade_Count', 'Delta_Pct']:
        if c in df.columns: df[c] = df[c].fillna(0)
    return df

# ==============================================================================
# 🌍  CONTEXT BUILDER
# ==============================================================================
def build_market_context():
    """Load CONTEXT_ASSETS + SPY sequentially, return (market_df, spy_df)."""
    print("🌍 Building market context...")
    frames = []
    for ticker in CONTEXT_ASSETS:
        df = _load_master(ticker)
        if df is None:
            print(f"   ⚠️  Could not load context asset {ticker}")
            continue
        df = _heal(df)
        daily_c = df['Close'].resample('D').last().ffill().shift(1).reindex(df.index, method='ffill')
        cols = {f'{ticker}_Anchor_ROC': (df['Close'] - daily_c) / daily_c}
        for h in ROC_HORIZONS:
            cols[f'{ticker}_ROC_{h}m'] = df['Close'].pct_change(h).fillna(0)
        if ticker == 'SMH':
            # Use pandas_ta to match training
            if len(df) >= 14:
                rsi = ta.rsi(df['Close'], length=14)
                cols['SMH_RSI'] = rsi.fillna(50) / 100.0 if rsi is not None else pd.Series(0.5, index=df.index)
            else:
                print(f"      ⚠️  SMH has too few rows ({len(df)}) for RSI — using placeholder 0.5")
                cols['SMH_RSI'] = pd.Series(0.5, index=df.index)
        frames.append(pd.DataFrame(cols, index=df.index))
        del df

    if not frames:
        raise RuntimeError("No context assets loaded.")
    market_df = pd.concat(frames, axis=1).ffill().fillna(0)

    spy_df = _load_master('SPY')
    if spy_df is not None:
        spy_df = _heal(spy_df)

    return market_df, spy_df

# ==============================================================================
# 🔧  FEATURE ENGINEERING
# ==============================================================================
PST = pytz.timezone('America/Los_Angeles')

def engineer_asset(ticker, market_df):
    df = _load_master(ticker)
    if df is None:
        return None
    df = _heal(df)

    daily_c       = df['Close'].resample('D').last().shift(1).reindex(df.index, method='ffill')
    df['Gap_Pct'] = (df['Open'] - daily_c) / daily_c
    df['PM_High'] = df.groupby(df.index.date)['High'].cummax()
    df['Dist_to_PM_High'] = (df['Close'] - df['PM_High']) / (df['PM_High'] + 1e-9)

    # Use pandas_ta for exact alignment with exp_97 training
    df['RSI_14'] = ta.rsi(df['Close'], length=14)
    df['RSI_60'] = ta.rsi(df['Close'], length=60)
    df['RVOL']   = df['Volume'] / df['Volume'].rolling(50).mean().clip(lower=1e-9)
    df['ATR_14'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['Compression'] = (df['High'] - df['Low']) / (df['ATR_14'] + 1e-9)
    for h in ROC_HORIZONS:
        df[f'ROC_{h}m'] = df['Close'].pct_change(h)

    df = df.join(market_df, how='left').ffill().fillna(0)

    df_pst = df.index.tz_convert(PST)
    mins   = df_pst.hour * 60 + df_pst.minute
    df['Time_Pos'] = np.clip((mins - 390) / 390, 0, 1)
    df['_mins']    = mins

    return df.replace([np.inf, -np.inf], np.nan).fillna(0)

# ==============================================================================
# 📏  SCALING
# ==============================================================================
def load_stats():
    blob = gcs_client.bucket(BUCKET_NAME).blob(GCS_STATS_PATH)
    return json.loads(blob.download_as_text())

def apply_scaling(df, stats):
    df_s = df.copy()
    for col in ASSET_COLS + CONTEXT_COLS:
        if col == 'Time_Pos': continue
        if col in stats:
            df_s[col] = ((df_s[col] - stats[col]['med']) / stats[col]['iqr']).clip(-10, 10).astype('float32')
    return df_s

# ==============================================================================
# 🧠  MODEL ARCHITECTURE  (must match exp_97.py training exactly)
# ==============================================================================
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.4):
    x   = LayerNormalization(epsilon=1e-6)(inputs)
    x   = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    res = Add()([x, inputs])
    x   = LayerNormalization(epsilon=1e-6)(res)
    x   = Dense(ff_dim, activation="relu")(x)
    x   = Dropout(dropout)(x)
    x   = Dense(inputs.shape[-1])(x)
    return Add()([x, res])

def build_model():
    in_a  = Input(shape=(TIME_STEPS, len(ASSET_COLS)), name='asset_in')
    in_c  = Input(shape=(len(CONTEXT_COLS),),          name='context_in')
    in_id = Input(shape=(1,), dtype=tf.int32,           name='asset_id')

    # Transformer sequence branch
    x = transformer_encoder(Dense(128)(in_a), 128, 4, 128)
    x = GlobalAveragePooling1D()(transformer_encoder(x, 128, 4, 128))

    # Context branch
    y = Dense(128, activation='relu')(BatchNormalization()(in_c))

    # Asset embedding branch
    e = Embedding(N_ASSETS, EMBED_DIM, name='asset_embed')(in_id)
    e = Flatten()(e)
    e = Dense(32, activation='relu')(e)

    # Fuse all three
    z   = Concatenate()([x, Dropout(0.5)(y), e])
    z   = Dense(128, activation='relu', kernel_regularizer=l2(5e-3))(z)
    out = Dense(1, activation='sigmoid', dtype='float32')(Dropout(0.5)(z))
    return Model(inputs=[in_a, in_c, in_id], outputs=out)

# ==============================================================================
# ☁️  MODEL LOADER
# ==============================================================================
_model_cache = {}

def get_model(seed):
    if seed in _model_cache:
        return _model_cache[seed]
    print(f"   📥 Loading model seed {seed}...")
    blob_path  = f"{GCS_MODEL_PATH}/UNIVERSAL_S{seed}_M{MODEL_ID}.h5"
    local_path = f"/tmp/UNIVERSAL_S{seed}_M{MODEL_ID}.h5"
    gcs_client.bucket(BUCKET_NAME).blob(blob_path).download_to_filename(local_path)
    model = build_model()
    model.load_weights(local_path)
    _model_cache[seed] = model
    return model

# ==============================================================================
# 🎯  SCORING ENGINE
# ==============================================================================
def score_ticker(ticker, cfg, df_scaled):
    entry_minute = cfg['entry_minute']

    if len(df_scaled) == 0:
        print(f"   ❌ {ticker}: no data loaded")
        return None

    # Always score the absolute latest available timestamp
    row_idx = len(df_scaled) - 1
    score_date = df_scaled.index[row_idx].tz_convert(PST).date()
    print(f"   ℹ️  {ticker}: Scoring latest available timestamp: {df_scaled.index[row_idx]}")

    if row_idx < TIME_STEPS:
        print(f"   ⚠️  {ticker}: not enough history ({row_idx} < {TIME_STEPS})")
        return None

    if 'Volume' in df_scaled.columns and float(df_scaled['Volume'].iloc[row_idx]) == 0:
        print(f"   ⚠️  {ticker}: synthetic bar (Volume=0) at entry — skipping")
        return None

    XA = df_scaled[ASSET_COLS].values
    XC = df_scaled[CONTEXT_COLS].values

    XA_b  = XA[row_idx - TIME_STEPS:row_idx][np.newaxis]
    XC_b  = XC[row_idx - 1][np.newaxis]
    XID_b = np.array([[ASSET_INDEX.get(ticker, OTHER_IDX)]], dtype=np.int32)

    model = get_model(cfg['seed'])
    prob  = float(model(
        {'asset_in': XA_b, 'context_in': XC_b, 'asset_id': XID_b},
        training=False
    ).numpy().flatten()[0])
    signal = prob >= cfg['prob_threshold']

    bar_ts = df_scaled.index[row_idx]
    close  = float(df_scaled['Close'].iloc[row_idx]) if 'Close' in df_scaled.columns else float('nan')

    return {
        'ticker':          ticker,
        'seed':            cfg['seed'],
        'prob':            round(prob, 4),
        'threshold':       cfg['prob_threshold'],
        'signal':          signal,
        'tp_pct':          cfg['tp_pct'],
        'sl_pct':          cfg['sl_pct'],
        'entry_minute':    entry_minute,
        'score_date':      score_date,
        'bar_ts':          bar_ts,
        'close':           round(close, 4),
    }

# ==============================================================================
# 📡  REGIME LOADER
# ==============================================================================
def load_regime_data(score_date, results, market_df, spy_df, run_ts):
    """Regime filter using trailing 30-min median ending at run_ts."""
    window_end   = run_ts.astimezone(PST)
    window_start = window_end - timedelta(minutes=30)

    vix_level = None
    df_vix = _load_master('VIX')
    if df_vix is not None:
        df_vix = _heal(df_vix)
        dpst = df_vix.copy(); dpst.index = dpst.index.tz_convert(PST)
        bars = dpst[(dpst.index >= window_start) & (dpst.index <= window_end)]
        if len(bars):
            vix_level = float(bars['Close'].median())

    tlt_gap = None
    if 'TLT_Anchor_ROC' in market_df.columns:
        dpst = market_df.copy(); dpst.index = dpst.index.tz_convert(PST)
        bars = dpst[(dpst.index >= window_start) & (dpst.index <= window_end)]
        if len(bars):
            tlt_gap = float(bars['TLT_Anchor_ROC'].median()) * 100

    spy_gap = None
    if spy_df is not None:
        dpst = spy_df.copy(); dpst.index = dpst.index.tz_convert(PST)
        prev  = dpst[dpst.index.date < score_date]
        bars  = dpst[(dpst.index >= window_start) & (dpst.index <= window_end)]
        if len(prev) > 0 and len(bars) > 0:
            prev_close  = float(prev['Close'].iloc[-1])
            entry_close = float(bars['Close'].median())
            if prev_close != 0:
                spy_gap = ((entry_close - prev_close) / prev_close) * 100

    return {'vix': vix_level, 'tlt_gap': tlt_gap, 'spy_gap': spy_gap}

def _filter_pass(ticker, regime):
    cfg = ASSET_CONFIGS[ticker]

    # Trap Logic from Backtest:
    # Blocks if VIX is high AND TLT drops AND SPY gaps up

    vix_bad = regime['vix']     is not None and regime['vix']     > cfg['vix_threshold']
    tlt_bad = regime['tlt_gap'] is not None and regime['tlt_gap'] < cfg['tlt_crash']
    spy_bad = regime['spy_gap'] is not None and regime['spy_gap'] > cfg['spy_ignore']

    is_trap = vix_bad and tlt_bad and spy_bad
    trade_ok = not is_trap

    return trade_ok, not vix_bad, not tlt_bad, not spy_bad

# ==============================================================================
# 🖥️  DASHBOARD PRINTER
# ==============================================================================
def print_dashboard(results, run_ts, regime=None):
    W       = 145
    pst_str = run_ts.astimezone(PST).strftime('%Y-%m-%d %H:%M PST')
    print()
    print("=" * W)
    print(f"  EXP-97 DAILY SIGNALS   run: {pst_str}")
    if regime:
        vix_val = regime['vix']
        tlt_val = regime['tlt_gap']
        spy_val = regime['spy_gap']
        vix_s = f"{vix_val:.1f}"    if vix_val is not None else "N/A"
        tlt_s = f"{tlt_val:+.2f}%" if tlt_val is not None else "N/A"
        spy_s = f"{spy_val:+.2f}%" if spy_val is not None else "N/A"
        active = list(ASSET_CONFIGS.values())
        vix_thr_ref = min(v['vix_threshold'] for v in active)
        tlt_thr_ref = max(v['tlt_crash']     for v in active)
        spy_thr_ref = min(v['spy_ignore']    for v in active)
        print(f"  Regime:  VIX={vix_s} (need <{vix_thr_ref:.0f})   "
              f"TLT={tlt_s} (need >{tlt_thr_ref:+.2f}%)   "
              f"SPY={spy_s} (need <{spy_thr_ref:+.2f}%)")
    print("=" * W)
    print(f"  {'Ticker':<6}  {'Seed':>5}  {'SIG':>6}  {'Prob':>6}  {'Thr':>5}  "
          f"{'Entry':>7}  {'TP tgt':>13}  {'SL stop':>13}  {'Close':>8}  {'Score bar':<16}  {'VIX<':>5}  {'TLT>':>6}  {'SPY<':>6}  "
          f"{'Win%':>6}  {'AvgPnL':>8}  {'Freq':>8}  FILT")
    print("─" * W)
    for r in results:
        cfg = ASSET_CONFIGS[r['ticker']]

        trade_ok = True
        vix_ok, tlt_ok, spy_ok = True, True, True
        if regime is not None:
            trade_ok, vix_ok, tlt_ok, spy_ok = _filter_pass(r['ticker'], regime)

        if r['signal']:
            if trade_ok:
                sig_str = "\033[92m✅ GO\033[0m "  # Bright green
            else:
                sig_str = "\033[31m❌ GO\033[0m "  # Red (Blocked by filter)
        else:
            sig_str = "──    "

        stats = cfg.get('stats', {})
        win_str = stats.get('win_rate', '--%')
        pnl_str = stats.get('avg_pnl', '--%')
        freq_str = stats.get('freq', '--/mo')

        close  = r['close']
        tp_pct = r['tp_pct']
        sl_pct = r['sl_pct']
        tp_str = f"${close*(1+tp_pct):.2f}(+{tp_pct*100:.0f}%)"
        sl_str = f"${close*(1-sl_pct):.2f}(-{sl_pct*100:.0f}%)"

        vix_t = f"{cfg['vix_threshold']:.0f}"
        tlt_t = f"{cfg['tlt_crash']:+.2f}"
        spy_t = f"{cfg['spy_ignore']:+.2f}"

        if regime is None:
            filt_str = " —"
        else:
            if trade_ok:
                filt_str = " ✅"
            else:
                parts = [
                    f"VIX:{regime['vix']:.1f}>{cfg['vix_threshold']:.0f}",
                    f"TLT:{regime['tlt_gap']:+.2f}<{cfg['tlt_crash']:+.2f}%",
                    f"SPY:{regime['spy_gap']:+.2f}>{cfg['spy_ignore']:+.2f}%",
                ]
                filt_str = f" ❌ {' & '.join(parts)}"

        em = r['entry_minute']
        entry_str = f"{em // 60}:{em % 60:02d}"
        prob_str = f"{r['prob']:.3f}"
        bar_str  = r['bar_ts'].astimezone(PST).strftime('%m-%d %H:%M PST')
        print(
            f"  {r['ticker']:<6}  {r['seed']:>5}  {sig_str}  {prob_str:>6}  {r['threshold']:>5.3f}  "
            f"{entry_str:>7}  {tp_str:>13}  {sl_str:>13}  {close:>8.2f}  {bar_str:<16}  {vix_t:>5}  {tlt_t:>6}  {spy_t:>6}  "
            f"{win_str:>6}  {pnl_str:>8}  {freq_str:>8}{filt_str}"
        )
    print("=" * W)
    signals = [r for r in results if r['signal']]
    if regime:
        go_trades = [r for r in signals if _filter_pass(r['ticker'], regime)[0]]
        blocked   = [r for r in signals if not _filter_pass(r['ticker'], regime)[0]]
    else:
        go_trades, blocked = signals, []
    print(f"  GO signals: {len(signals)}/{len(results)}", end="")
    if signals:
        print(f"  →  {', '.join(r['ticker'] for r in signals)}", end="")
    if blocked:
        print(f"  |  regime-blocked: {', '.join(r['ticker'] for r in blocked)}", end="")
    print()
    if go_trades:
        print(f"  TRADE: {', '.join(r['ticker'] for r in go_trades)}", end="")
        print(f"  ({len(go_trades)} position{'s' if len(go_trades) > 1 else ''})")
    print("=" * W)

# ==============================================================================
# 📊  BIGQUERY LOGGER
# ==============================================================================
_BQ_SCHEMA_DEFS = [
    ("run_timestamp",  "TIMESTAMP"),
    ("bar_timestamp",  "TIMESTAMP"),
    ("score_date",     "DATE"),
    ("experiment",     "STRING"),
    ("ticker",         "STRING"),
    ("seed",           "INT64"),
    ("prob",           "FLOAT64"),
    ("threshold",      "FLOAT64"),
    ("signal",         "STRING"),
    ("tp_pct",         "FLOAT64"),
    ("sl_pct",         "FLOAT64"),
    ("entry_minute",   "INT64"),
    ("close",          "FLOAT64"),
]

def _ensure_bq_schema(bq_client):
    added = []
    for col_name, col_type in _BQ_SCHEMA_DEFS:
        sql = (f"ALTER TABLE `{BQ_TABLE}` "
               f"ADD COLUMN IF NOT EXISTS `{col_name}` {col_type}")
        bq_client.query(sql).result()
        added.append(col_name)
    if added:
        print(f"   ✅ Schema verified ({len(added)} columns)")

def log_to_bigquery(results, run_ts):
    if bigquery is None:
        print("   ⚠️  BigQuery not available — skipping logging")
        return
    if not results:
        print("   ℹ️  No results to log — skipping BigQuery insert")
        return
    bq_client = bigquery.Client()
    _ensure_bq_schema(bq_client)
    rows = []
    for r in results:
        rows.append({
            'run_timestamp':  run_ts.isoformat(),
            'bar_timestamp':  r['bar_ts'].isoformat(),
            'score_date':     str(r['score_date']),
            'experiment':     MODEL_ID,
            'ticker':         r['ticker'],
            'seed':           r['seed'],
            'prob':           r['prob'],
            'threshold':      r['threshold'],
            'signal':         'GO' if r['signal'] else 'NO',
            'tp_pct':         r['tp_pct'],
            'sl_pct':         r['sl_pct'],
            'entry_minute':   r['entry_minute'],
            'close':          r['close'],
        })
    errors = bq_client.insert_rows_json(BQ_TABLE, rows)
    if errors:
        print(f"   ⚠️  BigQuery insert errors: {errors}")
    else:
        print(f"   ✅ Logged {len(rows)} rows to {BQ_TABLE}")

# ==============================================================================
# 🚀  MAIN PIPELINE
# ==============================================================================
print(f"🚀 STAGE 2 — Daily Scoring Pipeline (Exp-{MODEL_ID})")
run_ts = datetime.now(pytz.utc)
print(f"   Run time: {run_ts.strftime('%Y-%m-%d %H:%M UTC')}")
print(f"   TIME_STEPS={TIME_STEPS}  ROC_HORIZONS={ROC_HORIZONS}")

# 1. Load scaling stats
print("\n📏 Loading scaling stats...")
stats = load_stats()
print(f"   ✅ Loaded stats for {len(stats)} features")

# 2. Build market context + SPY regime data
print()
market_df, spy_df = build_market_context()

# 3. Pre-load models grouped by seed
print("\n🧠 Pre-loading models...")
needed_seeds = sorted(set(cfg['seed'] for cfg in ASSET_CONFIGS.values()))
for seed in needed_seeds:
    get_model(seed)
print(f"   ✅ Loaded {len(needed_seeds)} models: {needed_seeds}")

# 4. Score each ticker
print("\n🎯 Scoring tickers...")
results = []
for ticker, cfg in ASSET_CONFIGS.items():
    asset_id = ASSET_INDEX.get(ticker, OTHER_IDX)
    print(f"   ▶  {ticker} (seed {cfg['seed']}, threshold {cfg['prob_threshold']}, asset_id={asset_id})")
    df_raw = engineer_asset(ticker, market_df)
    if df_raw is None:
        print(f"      ❌ Skipping {ticker} — could not engineer features")
        continue
    df_scaled = apply_scaling(df_raw, stats)
    result    = score_ticker(ticker, cfg, df_scaled)
    if result is not None:
        results.append(result)

# 5. Compute regime filters
regime = None
if results:
    score_dates = list(set(r['score_date'] for r in results))
    print(f"\n📡 Computing regime filters for {score_dates[0]}...")
    regime = load_regime_data(score_dates[0], results, market_df, spy_df, run_ts)
    vix_s = f"{regime['vix']:.1f}"         if regime['vix']     is not None else "N/A"
    tlt_s = f"{regime['tlt_gap']:+.2f}%"   if regime['tlt_gap'] is not None else "N/A"
    spy_s = f"{regime['spy_gap']:+.2f}%"   if regime['spy_gap'] is not None else "N/A"
    print(f"   VIX={vix_s}  TLT={tlt_s}  SPY={spy_s}")

# 6. Print dashboard
print_dashboard(results, run_ts, regime)

# 7. Log to BigQuery
print("\n📊 Logging to BigQuery...")
log_to_bigquery(results, run_ts)

print("\n✅ STAGE 2 COMPLETE.")
