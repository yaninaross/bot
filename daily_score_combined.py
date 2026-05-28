"""
Daily Scoring — v5 WF optimized (152 + 160 + 170)
===================================================
Self-contained. Scores 17 tickers across 3 experiments.
Includes data loading, healing, dynamic TP/SL, regime filter, and BQ logging.

Active experiments:
  Exp-152 (Cross-sector, late cutoff):   1 ticker  (RCAT)
  Exp-160 (Cross-sector, early cutoff):  4 tickers (KEEL, PLUG, RIOT, IREN)
  Exp-170 (Cross-sector, walkforward):  12 tickers (ALAB, APLD, ASTS, BBAI, BE, BTBT,
                                                     CIFR, CLSK, FCEL, HUT, JOBY, LUNR)

Selection: best composite score per ticker from robust_wf_v5 optimization.
Dynamic TP/SL modes: static, atr, atr_adaptive, regime_scaled, trailing_decay.
170 uses fold-175 models (latest walkforward cutoff).

Confidence tiers:
  HIGH  — N>=12 TEST trades, reliable signal
  MED   — N=5-11 TEST trades, moderate sample
  LOW   — N<=4 TEST trades, may be noise

Run AFTER daily_ohlcv scripts have finished updating GCS.
Usage:  python daily_score_combined.py
"""

import os
import sys
import json
import gc
import subprocess

import numpy as np
import pandas as pd
import pytz
from datetime import datetime, timedelta

try:
    import pandas_ta as ta
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas_ta", "--no-deps", "--quiet"])
    import pandas_ta as ta

import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, GlobalAveragePooling1D,
    LayerNormalization, MultiHeadAttention, Add,
    Concatenate, Embedding, Flatten,
)
from tensorflow.keras.regularizers import l2
from google.cloud import storage as gcs_storage

if tf.config.list_physical_devices('GPU'):
    mixed_precision.set_global_policy('mixed_float16')

try:
    from google.cloud import bigquery
except Exception:
    bigquery = None

# ==============================================================================
# SHARED CONSTANTS
# ==============================================================================
BUCKET_NAME    = "time-series-bucket-ys"
GCS_MODEL_PATH = "Algos/to_deploy/Saved_models_v2/UNIVERSAL/Direction"
BQ_TABLE       = "ys-time-series.scoring_logs.live_predictions"
INFERENCE_DIR  = "Data/Inference_v2"
TAIL_DAYS      = 25
VOL_LOOKBACK_DAYS = 30
TIME_STEPS     = 240
EMBED_DIM      = 16
PST            = pytz.timezone('America/Los_Angeles')

TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

MICRO_COLS = ['Spread', 'OBI', 'Volume_Delta', 'Trade_Count', 'Delta_Pct',
              'bsize', 'asize', 'Total_Vol']

ROC_HORIZONS = [5, 30, 60, 120]

ASSET_COLS = (['Gap_Pct', 'RSI_14', 'RSI_60', 'RVOL', 'Compression',
               'ATR_14', 'Dist_to_PM_High', 'Bar_Is_Real'] +
              [f'ROC_{h}m' for h in ROC_HORIZONS] + MICRO_COLS)

# ==============================================================================
# EXPERIMENT DEFINITIONS
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



EXP_152 = {
    'model_id': '152_XSECTOR',
    'label': 'EXP-152 XSECTOR',
    'train_teachers': [
        'BBAI', 'RCAT', 'LUNR', 'ASTS', 'ACHR', 'MVIS', 'WULF',
        'CIFR', 'IREN', 'KEEL', 'HUT', 'CLSK', 'APLD',
    ],
    'context_assets': ['SPY', 'VIX', 'TLT', 'IWM'],
    'has_smh_rsi': False,
    'asset_configs': {
        'RCAT': {
            'seed': 15211, 'prob_threshold': 0.500, 'entry_minute': 380,
            'tpsl_mode': 'trailing_decay',
            'tp_pct': 0.0, 'sl_pct': 0.0,
            'sl_mult': 2.0, 'trail_start_mult': 2.0, 'trail_end_mult': 1.0,
            'vix_threshold': 999.0, 'tlt_crash': -999.0, 'spy_ignore': 999.0,
            'vol_filter_pct': 0,
            'confidence': 'MED', 'confidence_note': 'N=5 trades, moderate sample',
            'stats': {'win_rate': '80%', 'avg_pnl': '3.6%', 'freq': '3.0/mo'},
        },
    },
    'asset_cols': ASSET_COLS,
}

EXP_160 = {
    'model_id': '160_XSECTOR',
    'label': 'EXP-160 XSECTOR (early cutoff)',
    'train_teachers': [
        'BBAI', 'RCAT', 'LUNR', 'ASTS', 'ACHR', 'MVIS', 'WULF',
        'CIFR', 'IREN', 'KEEL', 'HUT', 'CLSK', 'APLD',
    ],
    'context_assets': ['SPY', 'VIX', 'TLT', 'IWM'],
    'has_smh_rsi': False,
    'asset_configs': {
        'KEEL': {
            'seed': 16006, 'prob_threshold': 0.500, 'entry_minute': 375,
            'tpsl_mode': 'static',
            'tp_pct': 0.25, 'sl_pct': 0.14,
            'vix_threshold': 999.0, 'tlt_crash': -999.0, 'spy_ignore': 999.0,
            'vol_filter_pct': 0,
            'confidence': 'LOW', 'confidence_note': 'N=4 trades, perfect WR likely noise',
            'stats': {'win_rate': '100%', 'avg_pnl': '6.2%', 'freq': '2.6/mo'},
        },
        'PLUG': {
            'seed': 16011, 'prob_threshold': 0.500, 'entry_minute': 380,
            'tpsl_mode': 'trailing_decay',
            'tp_pct': 0.0, 'sl_pct': 0.0,
            'sl_mult': 2.0, 'trail_start_mult': 2.0, 'trail_end_mult': 1.5,
            'vix_threshold': 999.0, 'tlt_crash': -999.0, 'spy_ignore': 999.0,
            'vol_filter_pct': 0,
            'confidence': 'LOW', 'confidence_note': 'N=4 trades, perfect WR likely noise',
            'stats': {'win_rate': '100%', 'avg_pnl': '5.6%', 'freq': '2.6/mo'},
        },
        'RIOT': {
            'seed': 16008, 'prob_threshold': 0.400, 'entry_minute': 375,
            'tpsl_mode': 'static',
            'tp_pct': 0.08, 'sl_pct': 0.05,
            'vix_threshold': 20.0, 'tlt_crash': 0.00, 'spy_ignore': 0.20,
            'vol_filter_pct': 0,
            'confidence': 'HIGH', 'confidence_note': 'N=12 trades, strong sample',
            'stats': {'win_rate': '83%', 'avg_pnl': '3.5%', 'freq': '7.3/mo'},
        },
        'IREN': {
            'seed': 16006, 'prob_threshold': 0.400, 'entry_minute': 385,
            'tpsl_mode': 'atr_adaptive',
            'tp_pct': 0.0, 'sl_pct': 0.0,
            'tp_mult': 2.0, 'sl_mult': 1.5,
            'vix_threshold': 20.0, 'tlt_crash': 0.00, 'spy_ignore': 0.00,
            'vol_filter_pct': 0,
            'confidence': 'MED', 'confidence_note': 'N=7 trades, moderate sample',
            'stats': {'win_rate': '57%', 'avg_pnl': '3.3%', 'freq': '4.5/mo'},
        },
    },
    'asset_cols': ASSET_COLS,
}

EXP_170 = {
    'model_id': '175_XSECTOR',
    'label': 'EXP-170 XSECTOR (walkforward)',
    'train_teachers': [
        'BBAI', 'RCAT', 'LUNR', 'ASTS', 'ACHR', 'MVIS', 'WULF',
        'CIFR', 'IREN', 'KEEL', 'HUT', 'CLSK', 'APLD',
    ],
    'context_assets': ['SPY', 'VIX', 'TLT', 'IWM'],
    'has_smh_rsi': False,
    'asset_configs': {
        'ALAB': {
            'seed': 17501, 'prob_threshold': 0.500, 'entry_minute': 375,
            'tpsl_mode': 'trailing_decay',
            'tp_pct': 0.0, 'sl_pct': 0.0,
            'sl_mult': 2.0, 'trail_start_mult': 2.0, 'trail_end_mult': 1.0,
            'vix_threshold': 999.0, 'tlt_crash': -999.0, 'spy_ignore': 999.0,
            'vol_filter_pct': 0,
            'confidence': 'MED', 'confidence_note': 'N=8 trades, moderate sample',
            'stats': {'win_rate': '75%', 'avg_pnl': '5.1%', 'freq': '4.9/mo'},
        },
        'APLD': {
            'seed': 17515, 'prob_threshold': 0.400, 'entry_minute': 375,
            'tpsl_mode': 'static',
            'tp_pct': 0.12, 'sl_pct': 0.05,
            'vix_threshold': 20.0, 'tlt_crash': 0.00, 'spy_ignore': 0.00,
            'vol_filter_pct': 25,
            'confidence': 'HIGH', 'confidence_note': 'N=23 trades, strong sample',
            'stats': {'win_rate': '70%', 'avg_pnl': '2.0%', 'freq': '14.0/mo'},
        },
        'ASTS': {
            'seed': 17512, 'prob_threshold': 0.600, 'entry_minute': 385,
            'tpsl_mode': 'regime_scaled',
            'tp_pct': 0.08, 'sl_pct': 0.05,
            'vix_threshold': 20.0, 'tlt_crash': 0.00, 'spy_ignore': 0.00,
            'vol_filter_pct': 0,
            'confidence': 'LOW', 'confidence_note': 'N=3 trades in TEST, perfect WR likely noise',
            'stats': {'win_rate': '100%', 'avg_pnl': '7.7%', 'freq': '1.8/mo'},
        },
        'BBAI': {
            'seed': 17512, 'prob_threshold': 0.600, 'entry_minute': 375,
            'tpsl_mode': 'static',
            'tp_pct': 0.08, 'sl_pct': 0.08,
            'vix_threshold': 20.0, 'tlt_crash': 0.00, 'spy_ignore': 0.00,
            'vol_filter_pct': 0,
            'confidence': 'LOW', 'confidence_note': 'N=3 trades in TEST, perfect WR likely noise',
            'stats': {'win_rate': '100%', 'avg_pnl': '7.7%', 'freq': '1.8/mo'},
        },
        'BE': {
            'seed': 17505, 'prob_threshold': 0.600, 'entry_minute': 385,
            'tpsl_mode': 'atr_adaptive',
            'tp_pct': 0.0, 'sl_pct': 0.0,
            'tp_mult': 1.5, 'sl_mult': 1.0,
            'vix_threshold': 20.0, 'tlt_crash': 0.00, 'spy_ignore': 0.00,
            'vol_filter_pct': 0,
            'confidence': 'LOW', 'confidence_note': 'N=4 trades in TEST',
            'stats': {'win_rate': '75%', 'avg_pnl': '4.8%', 'freq': '2.4/mo'},
        },
        'BTBT': {
            'seed': 17507, 'prob_threshold': 0.500, 'entry_minute': 380,
            'tpsl_mode': 'atr_adaptive',
            'tp_pct': 0.0, 'sl_pct': 0.0,
            'tp_mult': 1.5, 'sl_mult': 1.5,
            'vix_threshold': 999.0, 'tlt_crash': -999.0, 'spy_ignore': 999.0,
            'vol_filter_pct': 0,
            'confidence': 'LOW', 'confidence_note': 'N=3 trades in TEST, perfect WR likely noise',
            'stats': {'win_rate': '100%', 'avg_pnl': '4.0%', 'freq': '1.9/mo'},
        },
        'CIFR': {
            'seed': 17512, 'prob_threshold': 0.500, 'entry_minute': 375,
            'tpsl_mode': 'static',
            'tp_pct': 0.16, 'sl_pct': 0.14,
            'vix_threshold': 20.0, 'tlt_crash': 0.00, 'spy_ignore': 0.00,
            'vol_filter_pct': 0,
            'confidence': 'MED', 'confidence_note': 'N=8 trades, moderate sample',
            'stats': {'win_rate': '88%', 'avg_pnl': '5.4%', 'freq': '4.9/mo'},
        },
        'CLSK': {
            'seed': 17502, 'prob_threshold': 0.400, 'entry_minute': 380,
            'tpsl_mode': 'static',
            'tp_pct': 0.12, 'sl_pct': 0.08,
            'vix_threshold': 20.0, 'tlt_crash': 0.00, 'spy_ignore': 0.20,
            'vol_filter_pct': 25,
            'confidence': 'HIGH', 'confidence_note': 'N=14 trades, strong sample',
            'stats': {'win_rate': '86%', 'avg_pnl': '2.5%', 'freq': '8.5/mo'},
        },
        'FCEL': {
            'seed': 17507, 'prob_threshold': 0.400, 'entry_minute': 375,
            'tpsl_mode': 'atr_adaptive',
            'tp_pct': 0.0, 'sl_pct': 0.0,
            'tp_mult': 3.0, 'sl_mult': 0.5,
            'vix_threshold': 20.0, 'tlt_crash': 0.00, 'spy_ignore': 0.00,
            'vol_filter_pct': 0,
            'confidence': 'HIGH', 'confidence_note': 'N=13 trades, strong sample',
            'stats': {'win_rate': '54%', 'avg_pnl': '5.4%', 'freq': '7.9/mo'},
        },
        'HUT': {
            'seed': 17508, 'prob_threshold': 0.400, 'entry_minute': 385,
            'tpsl_mode': 'atr',
            'tp_pct': 0.0, 'sl_pct': 0.0,
            'tp_mult': 2.0, 'sl_mult': 1.0,
            'vix_threshold': 20.0, 'tlt_crash': -0.50, 'spy_ignore': 0.00,
            'vol_filter_pct': 0,
            'confidence': 'MED', 'confidence_note': 'N=10 trades, moderate sample',
            'stats': {'win_rate': '70%', 'avg_pnl': '3.5%', 'freq': '6.1/mo'},
        },
        'JOBY': {
            'seed': 17512, 'prob_threshold': 0.500, 'entry_minute': 385,
            'tpsl_mode': 'static',
            'tp_pct': 0.16, 'sl_pct': 0.03,
            'vix_threshold': 999.0, 'tlt_crash': -999.0, 'spy_ignore': 999.0,
            'vol_filter_pct': 0,
            'confidence': 'LOW', 'confidence_note': 'N=3 trades in TEST',
            'stats': {'win_rate': '67%', 'avg_pnl': '5.3%', 'freq': '1.9/mo'},
        },
        'LUNR': {
            'seed': 17513, 'prob_threshold': 0.500, 'entry_minute': 380,
            'tpsl_mode': 'atr',
            'tp_pct': 0.0, 'sl_pct': 0.0,
            'tp_mult': 1.5, 'sl_mult': 1.5,
            'vix_threshold': 20.0, 'tlt_crash': 0.00, 'spy_ignore': 0.00,
            'vol_filter_pct': 0,
            'confidence': 'LOW', 'confidence_note': 'N=4 trades in TEST',
            'stats': {'win_rate': '75%', 'avg_pnl': '4.7%', 'freq': '2.4/mo'},
        },
    },
    'asset_cols': ASSET_COLS,
}

for _exp in (EXP_152, EXP_160, EXP_170):
    _acols = _exp.get('asset_cols', ASSET_COLS)
    _exp['asset_cols'] = _acols
    _exp['context_cols'] = _build_context_cols(
        _exp['context_assets'], _exp['has_smh_rsi'])
    _exp['all_cols'] = _acols + _exp['context_cols']
    _idx = {t: i for i, t in enumerate(_exp['train_teachers'])}
    if 'BITF' in _idx:
        _idx['KEEL'] = _idx['BITF']
    if 'MIGI' in _idx:
        _idx['BGDE'] = _idx['MIGI']
    _exp['asset_index'] = _idx
    _exp['n_assets'] = len(_exp['train_teachers']) + 1
    _exp['other_idx'] = len(_exp['train_teachers'])
    _exp['stats_path'] = f"Data/Inference_v2/scaling_stats_experiment_{_exp['model_id']}.json"

EXPERIMENTS = [EXP_152, EXP_160, EXP_170]

# ==============================================================================
# GCS CLIENT
# ==============================================================================
gcs_client = gcs_storage.Client()

# ==============================================================================
# DATA LOADING
# ==============================================================================
_master_cache = {}


def load_master(ticker, tail_days=None):
    cache_key = (ticker, tail_days)
    if cache_key in _master_cache:
        return _master_cache[cache_key]
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
        if tail_days is not None:
            cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=tail_days)
            df = df[df.index >= cutoff]
            if df.empty:
                return None
        _master_cache[cache_key] = df
        return df
    except Exception as e:
        print(f"      load_master({ticker}): FAILED — {e}")
        return None


# ==============================================================================
# DATA HEALING
# ==============================================================================
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
# MARKET CONTEXT (per-experiment)
# ==============================================================================
def build_market_context(exp):
    frames = []
    for ticker in exp['context_assets']:
        df = load_master(ticker, tail_days=TAIL_DAYS)
        if df is None:
            continue
        df = df.sort_index()
        df = heal_data(df)
        daily_c = df['Close'].resample('D').last().ffill().shift(1).reindex(
            df.index, method='ffill')
        cols = {f'{ticker}_Anchor_ROC': (df['Close'] - daily_c) / daily_c}
        for h in ROC_HORIZONS:
            cols[f'{ticker}_ROC_{h}m'] = df['Close'].pct_change(h).fillna(0)
        if exp['has_smh_rsi'] and ticker == 'SMH':
            if len(df) >= 14:
                rsi = ta.rsi(df['Close'], length=14)
                cols['SMH_RSI'] = (rsi.fillna(50) / 100.0
                                   if rsi is not None
                                   else pd.Series(0.5, index=df.index))
            else:
                cols['SMH_RSI'] = pd.Series(0.5, index=df.index)
        frames.append(pd.DataFrame(cols, index=df.index))
        del df, daily_c
        gc.collect()
    if not frames:
        raise RuntimeError(f"No context assets loaded for {exp['label']}.")
    return pd.concat(frames, axis=1, sort=True).ffill().fillna(0)


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
    return df.replace([np.inf, -np.inf], np.nan).fillna(0)


def engineer_asset(ticker, market_df):
    df = load_master(ticker, tail_days=TAIL_DAYS)
    if df is None:
        return None
    df = df.sort_index()
    df = heal_data(df)
    return engineer_features(df, market_df)


# ==============================================================================
# SCALING
# ==============================================================================
def apply_scaling(df, stats, all_cols):
    df_s = df.copy()
    for col in all_cols:
        if col in ('Time_Pos', 'Bar_Is_Real'):
            continue
        if col in stats:
            df_s[col] = ((df_s[col] - stats[col]['med'])
                         / stats[col]['iqr']).clip(-10, 10).astype('float32')
    return df_s


# ==============================================================================
# MODEL ARCHITECTURE (parameterized)
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


def build_model(n_context_cols, n_assets, n_asset_cols=None):
    n_ac = n_asset_cols or len(ASSET_COLS)
    in_a  = Input(shape=(TIME_STEPS, n_ac), name='asset_in')
    in_c  = Input(shape=(n_context_cols,),             name='context_in')
    in_id = Input(shape=(1,), dtype=tf.int32,          name='asset_id')
    x = transformer_encoder(Dense(128)(in_a), 128, 4, 128)
    x = GlobalAveragePooling1D()(transformer_encoder(x, 128, 4, 128))
    y = Dense(128, activation='relu')(LayerNormalization()(in_c))
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
    blob_path  = f"{GCS_MODEL_PATH}/UNIVERSAL_S{seed}_M{model_id}.h5"
    local_path = f"/tmp/UNIVERSAL_S{seed}_M{model_id}.h5"
    if not os.path.exists(local_path):
        gcs_client.bucket(BUCKET_NAME).blob(blob_path).download_to_filename(local_path)
    model = build_model(len(exp['context_cols']), exp['n_assets'],
                        n_asset_cols=len(exp['asset_cols']))
    model.load_weights(local_path)
    _model_cache[key] = model
    return model


# ==============================================================================
# VOLUME FILTER (percentile-based, anchored to entry_minute)
# ==============================================================================
def _check_vol_filter(ticker, entry_minute, vol_filter_pct):
    if vol_filter_pct <= 0:
        return True, None, None
    df = load_master(ticker, tail_days=VOL_LOOKBACK_DAYS + 15)
    if df is None:
        return True, None, None
    df = heal_data(df)
    df_pst = df.copy()
    df_pst.index = df_pst.index.tz_convert(PST)
    mins = df_pst.index.hour * 60 + df_pst.index.minute
    at_entry = df_pst[mins == entry_minute]
    today = datetime.now(PST).date()
    today_bar = at_entry[at_entry.index.date == today]
    past_real = at_entry[(at_entry.index.date < today) & (at_entry['Volume'] > 0)]
    past_bars = past_real.tail(VOL_LOOKBACK_DAYS)
    if len(today_bar) == 0 or len(past_bars) < 5:
        return True, None, None
    entry_vol = float(today_bar['Volume'].iloc[-1])
    median_vol = float(past_bars['Volume'].median())
    if median_vol <= 0:
        return True, entry_vol, 0.0
    passed = entry_vol >= (vol_filter_pct / 100.0) * median_vol
    return passed, entry_vol, median_vol


# ==============================================================================
# DYNAMIC TP/SL COMPUTATION
# ==============================================================================
REGIME_TP_SCALE = [0.70, 0.85, 1.00, 1.30, 1.60]
REGIME_SL_SCALE = [0.60, 0.80, 1.00, 1.30, 1.80]


def _daily_atr_pct(df_raw, row_idx, period=14):
    daily = df_raw[['High', 'Low', 'Close']].resample('D').agg(
        {'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
    if len(daily) < 2:
        return 0.05
    prev_close = daily['Close'].shift(1)
    tr = pd.concat([
        daily['High'] - daily['Low'],
        (daily['High'] - prev_close).abs(),
        (daily['Low'] - prev_close).abs()
    ], axis=1).max(axis=1)
    atr_series = tr.rolling(period, min_periods=max(period // 2, 3)).mean()
    atr_pct = atr_series / daily['Close']
    atr_pct = atr_pct.reindex(df_raw.index, method='ffill')
    val = float(atr_pct.iloc[row_idx]) if row_idx < len(atr_pct) else np.nan
    if np.isnan(val) or val <= 0:
        return 0.05
    return val


def _adaptive_atr_pct(df_raw, row_idx, short=5, long=22, base=14):
    daily = df_raw[['High', 'Low', 'Close']].resample('D').agg(
        {'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
    if len(daily) < 2:
        return 0.05
    prev_close = daily['Close'].shift(1)
    tr = pd.concat([
        daily['High'] - daily['Low'],
        (daily['High'] - prev_close).abs(),
        (daily['Low'] - prev_close).abs()
    ], axis=1).max(axis=1)
    atr_base  = tr.rolling(base).mean()
    atr_short = tr.rolling(short).mean()
    atr_long  = tr.rolling(long).mean()
    accel = (atr_short / atr_long).clip(0.5, 3.0)
    atr_adaptive = atr_base * accel
    atr_pct = atr_adaptive / daily['Close']
    atr_pct = atr_pct.reindex(df_raw.index, method='ffill')
    val = float(atr_pct.iloc[row_idx]) if row_idx < len(atr_pct) else np.nan
    if np.isnan(val) or val <= 0:
        return 0.05
    return val


def compute_dynamic_tpsl(cfg, df_raw, row_idx, vix_value):
    mode = cfg.get('tpsl_mode', 'static')

    if mode == 'atr':
        atr_pct = _daily_atr_pct(df_raw, row_idx)
        tp = min(max(atr_pct * cfg['tp_mult'], 0.02), 0.60)
        sl = min(max(atr_pct * cfg['sl_mult'], 0.01), 0.25)
        return tp, sl, {'atr_pct': round(atr_pct, 6), 'mode': 'atr'}

    elif mode == 'atr_adaptive':
        atr_pct = _adaptive_atr_pct(df_raw, row_idx)
        tp = min(max(atr_pct * cfg['tp_mult'], 0.02), 0.60)
        sl = min(max(atr_pct * cfg['sl_mult'], 0.01), 0.25)
        return tp, sl, {'atr_pct': round(atr_pct, 6), 'mode': 'atr_adaptive'}

    elif mode in ('regime', 'regime_scaled'):
        regime_idx = 2
        if vix_value is not None:
            if vix_value <= 15: regime_idx = 0
            elif vix_value <= 20: regime_idx = 1
            elif vix_value <= 25: regime_idx = 2
            elif vix_value <= 35: regime_idx = 3
            else: regime_idx = 4
        tp = cfg['tp_pct'] * REGIME_TP_SCALE[regime_idx]
        sl = cfg['sl_pct'] * REGIME_SL_SCALE[regime_idx]
        return tp, sl, {'vix_regime': regime_idx, 'mode': mode}

    elif mode in ('trail', 'trailing_decay'):
        atr_pct = _daily_atr_pct(df_raw, row_idx)
        sl = min(max(atr_pct * cfg['sl_mult'], 0.01), 0.25)
        trail_start = atr_pct * cfg['trail_start_mult']
        trail_end = atr_pct * cfg['trail_end_mult']
        return None, sl, {
            'atr_pct': round(atr_pct, 6),
            'trail_start': round(trail_start, 6),
            'trail_end': round(trail_end, 6),
            'mode': mode,
        }

    return cfg['tp_pct'], cfg['sl_pct'], {'mode': 'static'}


# ==============================================================================
# SCORING ENGINE
# ==============================================================================
def score_ticker(ticker, cfg, df_scaled, df_raw, exp):
    if len(df_scaled) == 0:
        print(f"      {ticker}: SKIP — df_scaled is empty")
        return None
    entry_min = cfg['entry_minute']
    today = datetime.now(PST).date()
    target_pst = PST.localize(datetime(today.year, today.month, today.day,
                                       entry_min // 60, entry_min % 60))
    target_utc = target_pst.astimezone(pytz.utc)
    candidates = df_scaled.index[df_scaled.index <= target_utc]
    if len(candidates) == 0:
        print(f"      {ticker}: SKIP — no bars at or before entry {entry_min} "
              f"(target_utc={target_utc}, index range {df_scaled.index[0]} to {df_scaled.index[-1]})")
        return None
    row_idx = df_scaled.index.get_loc(candidates[-1])
    score_date = df_scaled.index[row_idx].tz_convert(PST).date()
    if score_date != today:
        print(f"      {ticker}: SKIP — nearest bar date {score_date} != today {today} "
              f"(bar_ts={df_scaled.index[row_idx]})")
        return None
    if row_idx < TIME_STEPS:
        print(f"      {ticker}: SKIP — not enough history: row_idx={row_idx} < {TIME_STEPS} "
              f"(need {TIME_STEPS} bars before entry bar)")
        return None
    if 'Volume' in df_scaled.columns and float(df_scaled['Volume'].iloc[row_idx]) == 0:
        print(f"      {ticker}: SKIP — Volume=0 at entry bar {df_scaled.index[row_idx]} (synthetic)")
        return None
    context_cols = exp['context_cols']
    XA = df_scaled[exp['asset_cols']].values
    XC = df_scaled[context_cols].values
    XA_b  = XA[row_idx - TIME_STEPS:row_idx][np.newaxis]
    XC_b  = XC[row_idx - 1][np.newaxis]
    XID_b = np.array([[exp['asset_index'].get(ticker, exp['other_idx'])]], dtype=np.int32)

    is_ensemble = cfg.get('is_ensemble', False)
    if is_ensemble:
        base_num = int(exp['model_id'].split('_')[0])
        seed_list = list(range(base_num * 100 + 1, base_num * 100 + 16))
        all_probs = []
        for s in seed_list:
            try:
                m = get_model(exp, s)
                p = float(m(
                    {'asset_in': XA_b, 'context_in': XC_b, 'asset_id': XID_b},
                    training=False
                ).numpy().flatten()[0])
                all_probs.append(p)
            except Exception:
                continue
        if not all_probs:
            print(f"      {ticker}: SKIP — no ensemble models loaded")
            return None
        prob = float(np.mean(all_probs))
    else:
        model = get_model(exp, cfg['seed'])
        prob = float(model(
            {'asset_in': XA_b, 'context_in': XC_b, 'asset_id': XID_b},
            training=False
        ).numpy().flatten()[0])

    signal = prob >= cfg['prob_threshold']
    bar_ts = df_scaled.index[row_idx]
    close  = (float(df_scaled['Close'].iloc[row_idx])
              if 'Close' in df_scaled.columns else float('nan'))

    regime = load_regime_data(score_date, cfg['entry_minute'])
    regime_ok = True
    if regime:
        regime_ok, _, _, _ = _filter_pass(cfg, regime)

    vol_pct = cfg.get('vol_filter_pct', 0)
    vol_ok, entry_vol, vol_median = _check_vol_filter(ticker, entry_min, vol_pct)

    filter_ok = regime_ok and vol_ok

    vix_value = regime.get('vix') if regime else None
    raw_row_idx = None
    if df_raw is not None and len(df_raw) > 0:
        raw_candidates = df_raw.index[df_raw.index <= bar_ts]
        if len(raw_candidates) > 0:
            raw_row_idx = df_raw.index.get_loc(raw_candidates[-1])

    tpsl_mode = cfg.get('tpsl_mode', 'static')
    if tpsl_mode != 'static' and df_raw is not None and raw_row_idx is not None:
        tp_pct, sl_pct, tpsl_info = compute_dynamic_tpsl(cfg, df_raw, raw_row_idx, vix_value)
    else:
        tp_pct, sl_pct, tpsl_info = cfg['tp_pct'], cfg['sl_pct'], {'mode': 'static'}

    return {
        'ticker': ticker, 'experiment': exp['model_id'], 'seed': cfg['seed'],
        'prob': round(prob, 4), 'threshold': cfg['prob_threshold'],
        'signal': signal, 'tp_pct': tp_pct, 'sl_pct': sl_pct,
        'entry_minute': cfg['entry_minute'], 'score_date': score_date,
        'bar_ts': bar_ts, 'close': round(close, 4),
        'filter_ok': filter_ok,
        'filter_status': 'OK' if filter_ok else 'BLOCKED',
        'vix_value': vix_value,
        'tlt_gap_pct': regime.get('tlt_gap') if regime else None,
        'spy_gap_pct': regime.get('spy_gap') if regime else None,
        'vol_filter_pct': vol_pct,
        'vol_filter_ok': vol_ok,
        'entry_volume': entry_vol,
        'vol_median': vol_median,
        'tpsl_mode': tpsl_info.get('mode', 'static'),
        'atr_pct_entry': tpsl_info.get('atr_pct'),
        'vix_regime_idx': tpsl_info.get('vix_regime'),
        'trail_start': tpsl_info.get('trail_start'),
        'trail_end': tpsl_info.get('trail_end'),
    }


# ==============================================================================
# REGIME FILTER (anchored to entry_minute, not run_ts)
# ==============================================================================
_regime_cache = {}


def load_regime_data(score_date, entry_minute):
    cache_key = (score_date, entry_minute)
    if cache_key in _regime_cache:
        return _regime_cache[cache_key]

    window_end   = PST.localize(datetime.combine(score_date, datetime.min.time())) + timedelta(minutes=entry_minute)
    window_start = window_end - timedelta(minutes=30)

    vix_level = None
    df_vix = load_master('VIX', tail_days=TAIL_DAYS)
    if df_vix is not None:
        df_vix = heal_data(df_vix)
        dpst = df_vix.copy()
        dpst.index = dpst.index.tz_convert(PST)
        bars = dpst[(dpst.index >= window_start) & (dpst.index <= window_end)]
        if len(bars):
            vix_level = float(bars['Close'].median())

    tlt_gap = None
    df_tlt = load_master('TLT', tail_days=TAIL_DAYS)
    if df_tlt is not None:
        df_tlt = heal_data(df_tlt)
        dpst = df_tlt.copy()
        dpst.index = dpst.index.tz_convert(PST)
        prev = dpst[dpst.index.date < score_date]
        bars = dpst[(dpst.index >= window_start) & (dpst.index <= window_end)]
        if len(prev) > 0 and len(bars) > 0:
            prev_close = float(prev['Close'].iloc[-1])
            entry_close = float(bars['Close'].median())
            if prev_close != 0:
                tlt_gap = ((entry_close - prev_close) / prev_close) * 100

    spy_gap = None
    df_spy = load_master('SPY', tail_days=TAIL_DAYS)
    if df_spy is not None:
        df_spy = heal_data(df_spy)
        dpst = df_spy.copy()
        dpst.index = dpst.index.tz_convert(PST)
        prev = dpst[dpst.index.date < score_date]
        bars = dpst[(dpst.index >= window_start) & (dpst.index <= window_end)]
        if len(prev) > 0 and len(bars) > 0:
            prev_close  = float(prev['Close'].iloc[-1])
            entry_close = float(bars['Close'].median())
            if prev_close != 0:
                spy_gap = ((entry_close - prev_close) / prev_close) * 100

    regime = {'vix': vix_level, 'tlt_gap': tlt_gap, 'spy_gap': spy_gap}
    _regime_cache[cache_key] = regime
    return regime


def _filter_pass(cfg, regime):
    vix_bad = regime['vix']     is not None and regime['vix']     > cfg['vix_threshold']
    tlt_bad = regime['tlt_gap'] is not None and regime['tlt_gap'] < cfg['tlt_crash']
    spy_bad = regime['spy_gap'] is not None and regime['spy_gap'] > cfg['spy_ignore']
    blocked = vix_bad and tlt_bad and spy_bad
    return not blocked, not vix_bad, not tlt_bad, not spy_bad


# ==============================================================================
# DASHBOARD
# ==============================================================================
_MODE_LABELS = {
    'static': 'STATIC', 'atr': 'ATR', 'atr_adaptive': 'ATR-ADP',
    'trailing_decay': 'TRAIL', 'trail': 'TRAIL',
    'regime_scaled': 'REGIME', 'regime': 'REGIME',
}


def print_experiment_section(exp, results):
    W = 120
    asset_configs = exp['asset_configs']
    print()
    print(f"  {'─'*W}")
    print(f"  {exp['label']}   ({exp['model_id']})   [{len(results)} tickers]")
    print(f"  {'─'*W}")
    print(f"  {'Ticker':<6}  {'Seed':>5}  {'SIG':<7}  {'Prob':>5} / {'Thr':>5}  "
          f"{'Entry':>5}  {'Mode':<8}  {'TP target':>14}  {'SL stop':>14}  "
          f"{'Close':>8}  {'CONF':<4}  {'Filter'}")
    print(f"  {'─'*W}")

    for r in results:
        cfg = asset_configs.get(r['ticker'], {})
        trade_ok = r.get('filter_ok', True)
        conf = cfg.get('confidence', '??')

        if r['signal']:
            sig_str = ("\033[92m GO  \033[0m" if trade_ok
                       else "\033[31mBLOCK\033[0m")
        else:
            sig_str = "  --  "

        close = r['close']
        mode_label = _MODE_LABELS.get(r.get('tpsl_mode', 'static'), '??')

        if r.get('tpsl_mode') in ('trail', 'trailing_decay') or r.get('tp_pct') is None:
            ts = r.get('trail_start', 0)
            tp_str = f"TRAIL(ts={ts:.1%})"
        else:
            tp_str = f"${close*(1+r['tp_pct']):.2f} (+{r['tp_pct']*100:.0f}%)"
        sl_str = f"${close*(1-r['sl_pct']):.2f} (-{r['sl_pct']*100:.0f}%)"

        if trade_ok:
            filt_str = "OK"
        else:
            parts = []
            if r.get('vix_value') is not None and cfg:
                parts.append(f"VIX:{r['vix_value']:.1f}>{cfg['vix_threshold']:.0f}")
            if r.get('tlt_gap_pct') is not None and cfg:
                parts.append(f"TLT:{r['tlt_gap_pct']:+.2f}<{cfg['tlt_crash']:+.2f}")
            if r.get('spy_gap_pct') is not None and cfg:
                parts.append(f"SPY:{r['spy_gap_pct']:+.2f}>{cfg['spy_ignore']:+.2f}")
            if not r.get('vol_filter_ok', True):
                parts.append("VOL")
            filt_str = f"BLOCKED {' & '.join(parts)}"

        em = r['entry_minute']
        entry_str = f"{em // 60}:{em % 60:02d}"
        seed_str = 'ENS' if cfg.get('is_ensemble') else str(r['seed'])

        print(f"  {r['ticker']:<6}  {seed_str:>5}  {sig_str}  "
              f"{r['prob']:>5.3f} / {r['threshold']:>5.3f}  "
              f"{entry_str:>5}  {mode_label:<8}  {tp_str:>14}  {sl_str:>14}  "
              f"{close:>8.2f}  {conf:<4}  {filt_str}")


def print_dashboard(all_results, run_ts):
    W = 120
    pst_str = run_ts.astimezone(PST).strftime('%Y-%m-%d %H:%M PST')

    print()
    print(f"  {'='*W}")
    print(f"  DAILY SIGNALS   {pst_str}")
    print(f"  {'='*W}")

    all_flat_r = [r for _, results in all_results for r in results]
    if all_flat_r:
        last_r = max(all_flat_r, key=lambda r: r['entry_minute'])
        vix_s = f"{last_r['vix_value']:.1f}"       if last_r.get('vix_value')   is not None else "N/A"
        tlt_s = f"{last_r['tlt_gap_pct']:+.2f}%"   if last_r.get('tlt_gap_pct') is not None else "N/A"
        spy_s = f"{last_r['spy_gap_pct']:+.2f}%"   if last_r.get('spy_gap_pct') is not None else "N/A"
        print(f"  Market:  VIX={vix_s}   TLT={tlt_s}   SPY={spy_s}")

    for exp, results in all_results:
        if results:
            print_experiment_section(exp, results)

    # ── SUMMARY ──────────────────────────────────────────────────────────────
    all_flat = [r for _, results in all_results for r in results]
    signals   = [r for r in all_flat if r['signal']]
    go_trades = [r for r in signals if r.get('filter_ok', True)]
    blocked   = [r for r in signals if not r.get('filter_ok', True)]

    print()
    print(f"  {'='*W}")
    print(f"  SUMMARY")
    print(f"  {'─'*W}")

    total = len(all_flat)
    print(f"  Scored:    {total} tickers")
    print(f"  Signals:   {len(signals)}/{total}" +
          (f"  ({', '.join(r['ticker'] for r in signals)})" if signals else ""))
    if blocked:
        print(f"  Blocked:   {', '.join(r['ticker'] for r in blocked)}")

    if go_trades:
        print()
        print(f"  \033[92mTRADE: {', '.join(r['ticker'] for r in go_trades)}"
              f"  ({len(go_trades)} position{'s' if len(go_trades) != 1 else ''})\033[0m")

    # ── ASSET DETAILS ────────────────────────────────────────────────────────
    print()
    print(f"  {'─'*W}")
    print(f"  ASSET DETAILS (WF optimization stats)")
    print(f"  {'─'*W}")
    print(f"  {'Ticker':<6}  {'Exp':<14}  {'CONF':<4}  {'WR':>5}  {'AvgPnL':>7}  "
          f"{'Freq':>8}  {'Mode':<8}  {'Regime':>20}  {'Note'}")
    print(f"  {'─'*W}")

    for exp, results in all_results:
        for r in results:
            cfg = exp.get('asset_configs', {}).get(r['ticker'], {})
            conf = cfg.get('confidence', '??')
            note = cfg.get('confidence_note', '')
            stats = cfg.get('stats', {})
            mode_label = _MODE_LABELS.get(cfg.get('tpsl_mode', 'static'), '??')

            vix_t = cfg.get('vix_threshold', 999)
            tlt_t = cfg.get('tlt_crash', -999)
            spy_t = cfg.get('spy_ignore', 999)
            if vix_t >= 999 and tlt_t <= -999 and spy_t >= 999:
                regime_str = "no filter"
            else:
                parts = []
                if vix_t < 999:
                    parts.append(f"VIX<{vix_t:.0f}")
                if tlt_t > -999:
                    parts.append(f"TLT>{tlt_t:+.2f}")
                if spy_t < 999:
                    parts.append(f"SPY<{spy_t:+.2f}")
                regime_str = ' '.join(parts)

            conf_color = ''
            conf_reset = ''
            if conf == 'LOW':
                conf_color = '\033[33m'
                conf_reset = '\033[0m'
            elif conf == 'MED':
                conf_color = '\033[36m'
                conf_reset = '\033[0m'

            print(f"  {r['ticker']:<6}  {exp['model_id']:<14}  "
                  f"{conf_color}{conf:<4}{conf_reset}  "
                  f"{stats.get('win_rate', '--'):>5}  {stats.get('avg_pnl', '--'):>7}  "
                  f"{stats.get('freq', '--'):>8}  {mode_label:<8}  "
                  f"{regime_str:>20}  {note}")

    print()
    print(f"  Confidence: "
          f"\033[92mHIGH\033[0m = N>=12 reliable   "
          f"\033[36mMED\033[0m = N=5-11 moderate   "
          f"\033[33mLOW\033[0m = N<=4 possible noise")
    print(f"  {'='*W}")


# ==============================================================================
# BIGQUERY LOGGER
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
    ("filter_status",  "STRING"),
    ("vix_value",      "FLOAT64"),
    ("tlt_gap_pct",    "FLOAT64"),
    ("spy_gap_pct",    "FLOAT64"),
    ("vol_filter_pct", "INT64"),
    ("vol_filter_ok",  "STRING"),
    ("entry_volume",   "FLOAT64"),
    ("vol_median",     "FLOAT64"),
    ("tpsl_mode",      "STRING"),
    ("atr_pct_entry",  "FLOAT64"),
    ("vix_regime_idx", "INT64"),
    ("trail_start",    "FLOAT64"),
    ("trail_end",      "FLOAT64"),
]


def _ensure_bq_schema(bq_client):
    for col_name, col_type in _BQ_SCHEMA_DEFS:
        sql = (f"ALTER TABLE `{BQ_TABLE}` "
               f"ADD COLUMN IF NOT EXISTS `{col_name}` {col_type}")
        bq_client.query(sql).result()


def log_to_bigquery(all_results, run_ts):
    if bigquery is None:
        print("   BigQuery not available — skipping")
        return
    rows = []
    for _, results in all_results:
        for r in results:
            rows.append({
                'run_timestamp': run_ts.isoformat(),
                'bar_timestamp': r['bar_ts'].isoformat(),
                'score_date':    str(r['score_date']),
                'experiment':    r['experiment'],
                'ticker':        r['ticker'],
                'seed':          r['seed'],
                'prob':          r['prob'],
                'threshold':     r['threshold'],
                'signal':        'GO' if r['signal'] else 'NO',
                'tp_pct':        r['tp_pct'],
                'sl_pct':        r['sl_pct'],
                'entry_minute':  r['entry_minute'],
                'close':         r['close'],
                'filter_status': r.get('filter_status', 'N/A'),
                'vix_value':     r.get('vix_value'),
                'tlt_gap_pct':   r.get('tlt_gap_pct'),
                'spy_gap_pct':   r.get('spy_gap_pct'),
                'vol_filter_pct': r.get('vol_filter_pct', 0),
                'vol_filter_ok':  'OK' if r.get('vol_filter_ok', True) else 'BLOCKED',
                'entry_volume':   r.get('entry_volume'),
                'vol_median':     r.get('vol_median'),
                'tpsl_mode':      r.get('tpsl_mode', 'static'),
                'atr_pct_entry':  r.get('atr_pct_entry'),
                'vix_regime_idx': r.get('vix_regime_idx'),
                'trail_start':    r.get('trail_start'),
                'trail_end':      r.get('trail_end'),
            })
    if not rows:
        return
    bq_client = bigquery.Client()
    _ensure_bq_schema(bq_client)
    try:
        errors = bq_client.insert_rows_json(BQ_TABLE, rows)
        if errors:
            print(f"   BigQuery errors: {errors}")
        else:
            print(f"   Logged {len(rows)} rows to {BQ_TABLE}")
    except Exception as e:
        print(f"   BigQuery failed: {e}")


# ==============================================================================
# RUN ONE EXPERIMENT
# ==============================================================================
def run_experiment(exp, run_ts):
    model_id = exp['model_id']
    label    = exp['label']
    n_tickers = len(exp['asset_configs'])

    try:
        blob  = gcs_client.bucket(BUCKET_NAME).blob(exp['stats_path'])
        stats = json.loads(blob.download_as_text())
    except Exception as e:
        print(f"   {label}: FAILED — {e}")
        return []

    market_df = build_market_context(exp)

    needed_seeds = set()
    base_num = int(model_id.split('_')[0])
    for cfg in exp['asset_configs'].values():
        if cfg.get('is_ensemble', False):
            needed_seeds.update(range(base_num * 100 + 1, base_num * 100 + 16))
        else:
            needed_seeds.add(cfg['seed'])
    for seed in sorted(needed_seeds):
        try:
            get_model(exp, seed)
        except Exception as e:
            print(f"   {label}: model seed {seed} FAILED — {e}")

    now_pst = run_ts.astimezone(PST)
    now_minutes = now_pst.hour * 60 + now_pst.minute

    results = []
    for ticker, cfg in exp['asset_configs'].items():
        if now_minutes < cfg['entry_minute']:
            print(f"      {ticker}: SKIP — entry_minute {cfg['entry_minute']} > now {now_minutes} (too early)")
            continue
        df_raw = engineer_asset(ticker, market_df)
        if df_raw is None:
            print(f"      {ticker}: SKIP — engineer_asset returned None (load_master failed)")
            continue
        df_scaled = apply_scaling(df_raw, stats, exp['all_cols'])
        result = score_ticker(ticker, cfg, df_scaled, df_raw, exp)
        if result is not None:
            results.append(result)

    print(f"   {label}: scored {len(results)}/{n_tickers}")
    return results


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    run_ts = datetime.now(pytz.utc)
    total_tickers = sum(len(e['asset_configs']) for e in EXPERIMENTS)
    print(f"Scoring {total_tickers} tickers — {run_ts.strftime('%Y-%m-%d %H:%M UTC')}")

    all_results = []
    for exp in EXPERIMENTS:
        try:
            results = run_experiment(exp, run_ts)
        except Exception as e:
            print(f"   {exp['label']}: CRASHED — {e}")
            results = []
        all_results.append((exp, results))
        _model_cache.clear()
        _master_cache.clear()
        _regime_cache.clear()
        tf.keras.backend.clear_session()
        gc.collect()

    print_dashboard(all_results, run_ts)

    log_to_bigquery(all_results, run_ts)


main()
