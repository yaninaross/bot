"""
data_utils_110.py — Shared data processing for Exp-110+
========================================================
Single source of truth for data loading, healing, feature engineering,
scaling, and model architecture. Imported by training (exp_110.py),
backtesting, and scoring scripts.

Critical fix over Exp-97: heal_data() includes resample('1min').asfreq()
to align training with scoring/backtesting pipelines.
"""

import sys
import subprocess
import gc

import numpy as np
import pandas as pd
import pytz

try:
    import pandas_ta as ta
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas_ta", "--no-deps", "--quiet"])
    import pandas_ta as ta

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, GlobalAveragePooling1D,
    LayerNormalization, MultiHeadAttention, Add,
    BatchNormalization, Concatenate, Embedding, Flatten,
)
from tensorflow.keras.regularizers import l2

# ==============================================================================
# COLUMN DEFINITIONS
# ==============================================================================
MICRO_COLS = ['Spread', 'OBI', 'Volume_Delta', 'Trade_Count', 'Delta_Pct',
              'bsize', 'asize', 'Total_Vol']

ROC_HORIZONS = [5, 30, 60, 120]

ASSET_COLS = (['Gap_Pct', 'RSI_14', 'RSI_60', 'RVOL', 'Compression',
               'ATR_14', 'Dist_to_PM_High'] +
              [f'ROC_{h}m' for h in ROC_HORIZONS] + MICRO_COLS)

TIME_STEPS = 240
EMBED_DIM  = 16

PST = pytz.timezone('America/Los_Angeles')

TICKER_FILE_MAP = {'BITF': 'KEEL'}


def build_context_cols(context_assets, has_smh_rsi=False):
    """Build context column list from context asset names."""
    cols = []
    for t in context_assets:
        cols.append(f'{t}_Anchor_ROC')
        for h in ROC_HORIZONS:
            cols.append(f'{t}_ROC_{h}m')
    if has_smh_rsi:
        cols.append('SMH_RSI')
    cols.append('Time_Pos')
    return cols


# ==============================================================================
# DATA LOADING
# ==============================================================================
def load_master(ticker, bucket_name, inference_dir, tail_days=None, add_micro_zeros=True):
    """Load master_inference parquet from GCS.

    Args:
        ticker: Ticker symbol (e.g. 'RIOT', 'BITF').
        bucket_name: GCS bucket name.
        inference_dir: GCS directory path (e.g. 'Data/Inference').
        tail_days: If set, keep only last N days. None = full history.
        add_micro_zeros: Add zero MICRO_COLS if missing in the data.
    """
    clean = ticker.replace("I:", "").replace("X:", "")
    file_ticker = TICKER_FILE_MAP.get(clean, clean)
    uri = f"gs://{bucket_name}/{inference_dir}/{file_ticker}_master_inference.parquet"

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


# ==============================================================================
# DATA HEALING — includes resample('1min') for training-scoring alignment
# ==============================================================================
def heal_data(df):
    """Heal gaps: resample to 1-min grid, ffill prices, zero-fill volumes."""
    df = df.resample('1min').asfreq()
    for c in ['Open', 'High', 'Low', 'Close', 'Spread', 'OBI', 'bsize', 'asize']:
        if c in df.columns:
            df[c] = df[c].ffill()
    for c in ['Volume', 'Total_Vol', 'Volume_Delta', 'Trade_Count', 'Delta_Pct']:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    return df


# ==============================================================================
# MARKET CONTEXT
# ==============================================================================
def build_market_context(context_assets, has_smh_rsi, bucket_name, inference_dir,
                         tail_days=None):
    """Load context assets and compute ROC / anchor features.

    Returns:
        market_df: DataFrame with context columns indexed by datetime.
    """
    print("   Building market context...")
    frames = []
    for ticker in context_assets:
        df = load_master(ticker, bucket_name, inference_dir,
                         tail_days=tail_days, add_micro_zeros=True)
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
        if has_smh_rsi and ticker == 'SMH':
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
        raise RuntimeError("No context assets loaded.")
    return pd.concat(frames, axis=1).ffill().fillna(0)


# ==============================================================================
# FEATURE ENGINEERING
# ==============================================================================
def engineer_features(df, market_df):
    """Engineer asset-level features from healed OHLCV data.

    Computes ASSET_COLS (technical indicators + micro) and joins market context.
    Does NOT compute targets — that is training-specific.
    Preserves OHLCV + Volume columns for downstream use.

    Args:
        df: Healed DataFrame with OHLCV + MICRO_COLS.
        market_df: Context DataFrame from build_market_context().

    Returns:
        DataFrame with all feature columns + OHLCV.
    """
    daily_c = df['Close'].resample('D').last().shift(1).reindex(
        df.index, method='ffill')
    df['Gap_Pct'] = (df['Open'] - daily_c) / daily_c

    df['PM_High'] = df.groupby(df.index.date)['High'].cummax()
    df['Dist_to_PM_High'] = (df['Close'] - df['PM_High']) / (df['PM_High'] + 1e-9)
    df['RSI_14'] = ta.rsi(df['Close'], length=14)
    df['RSI_60'] = ta.rsi(df['Close'], length=60)
    df['RVOL'] = df['Volume'] / df['Volume'].rolling(50).mean().clip(lower=1e-9)
    df['ATR_14'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['Compression'] = (df['High'] - df['Low']) / (df['ATR_14'] + 1e-9)
    for h in ROC_HORIZONS:
        df[f'ROC_{h}m'] = df['Close'].pct_change(h)

    df = df.join(market_df, how='left').ffill().fillna(0)

    df_pst = df.index.tz_convert(PST)
    mins = df_pst.hour * 60 + df_pst.minute
    df['Time_Pos'] = np.clip((mins - 390) / 390, 0, 1)

    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


# ==============================================================================
# SCALING
# ==============================================================================
def apply_scaling(df, stats, all_cols, inplace=False):
    """IQR scaling: (val - median) / IQR, clipped to [-10, 10]. Time_Pos excluded."""
    df_s = df if inplace else df.copy()
    for col in all_cols:
        if col == 'Time_Pos':
            continue
        if col in stats:
            df_s[col] = ((df_s[col] - stats[col]['med'])
                         / stats[col]['iqr']).clip(-10, 10).astype('float32')
    return df_s


# ==============================================================================
# MODEL ARCHITECTURE — 2-layer transformer encoder + context + asset embedding
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


def build_model(n_asset_cols, n_context_cols, n_assets,
                embed_dim=EMBED_DIM, time_steps=TIME_STEPS):
    """Build universal transformer model.

    Args:
        n_asset_cols: Number of asset feature columns (len(ASSET_COLS)).
        n_context_cols: Number of context columns (len(CONTEXT_COLS)).
        n_assets: Total embedding slots (num_teachers + 1 for OTHER).
        embed_dim: Asset embedding dimension (default 16).
        time_steps: Sequence length (default 240).
    """
    in_a  = Input(shape=(time_steps, n_asset_cols), name='asset_in')
    in_c  = Input(shape=(n_context_cols,),          name='context_in')
    in_id = Input(shape=(1,), dtype=tf.int32,       name='asset_id')

    x = transformer_encoder(Dense(128)(in_a), 128, 4, 128)
    x = GlobalAveragePooling1D()(transformer_encoder(x, 128, 4, 128))

    y = Dense(128, activation='relu')(BatchNormalization()(in_c))

    e = Embedding(n_assets, embed_dim, name='asset_embed')(in_id)
    e = Flatten()(e)
    e = Dense(32, activation='relu')(e)

    z = Concatenate()([x, Dropout(0.5)(y), e])
    z = Dense(128, activation='relu', kernel_regularizer=l2(5e-3))(z)
    out = Dense(1, activation='sigmoid', dtype='float32')(Dropout(0.5)(z))
    return Model(inputs=[in_a, in_c, in_id], outputs=out)
