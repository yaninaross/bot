# %% ── Combined ST LONG + SHORT — Deployed Strategy Backtest ──────────────────
#
# Backtests BOTH deployed Spike Transformer strategies over the same test window
# using live model inference from master_inference parquets on GCS.
#
# ST LONG  — M_CONS_6NEWEST (6 newest models for consensus fire counting):
#   MR-v2 F9/F8/F7 + L5-1 F7 + ASL6 F7 + L5-1 F6
#   Hybrid config: best val_b per ticker across all 10 models
#   Newest-first deploy selection for TP/SL
#
# ST SHORT — M_MRV2_NEWEST3 (3 newest MR-v2 folds for consensus fire counting):
#   MR-v2 F8/F9/F10 (L5-1/L5-2 REMOVED: proven negative signal)
#   Hybrid config: best val_b per ticker across all 5 MR-v2 folds (F7 gated >=3.0%)
#
# 3-phase per direction:
#   1. Score all tickers with all consensus models → fire data
#   2. Per-day selection: rank by (n_fires DESC, sum_conviction DESC), top MAX_TRADES
#   3. Trade simulation on selected signals only
#
# Combined portfolio: equal-weight across LONG + SHORT trades each day.
# GPU recommended. Colab-ready.

import os, sys, gc, ctypes, subprocess, warnings
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd
import pytz

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    import pandas_ta as ta
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas_ta", "--no-deps", "--quiet"])
    import pandas_ta as ta

try:
    from numba import jit
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numba", "--quiet"])
    from numba import jit

import tensorflow as tf
import keras
from keras.layers import Dense, Dropout, LayerNormalization
from keras import layers as _kl
import h5py
from google.cloud import storage as gcs_storage

if tf.config.list_physical_devices('GPU'):
    keras.mixed_precision.set_dtype_policy('mixed_float16')

try:
    from google.cloud import bigquery
except Exception:
    bigquery = None

try:
    from google.colab import auth
    _IN_COLAB = True
    try:
        auth.authenticate_user()
    except Exception:
        pass
except (ImportError, Exception):
    _IN_COLAB = False

import matplotlib
if not _IN_COLAB:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ==============================================================================
# CONFIG — SHARED
# ==============================================================================
BUCKET_NAME      = "time-series-bucket-ys"
WF_BQ_TABLE      = "ys-time-series.scoring_logs.wf_optimization_results"
INFERENCE_DIR    = "Data/Inference_v2"
PST              = pytz.timezone('America/Los_Angeles')

BACKTEST_START   = '2026-08-03'
BACKTEST_END     = '2026-08-30'

TAIL_DAYS        = 45
SCORE_MINUTE     = 375
ENTRY_MINUTE     = 385
EXIT_TIME_MIN    = 780
SLIPPAGE_PCT     = 0.0015

BASE_PNL_THRESH  = 0.018
MAX_TRADES       = 3

TIME_STEPS       = 128
D_MODEL          = 64
N_HEADS          = 4
D_FF             = 128
N_BLOCKS         = 2
DROPOUT_RATE     = 0.1
FILM_HIDDEN      = 64
N_HORIZONS       = 3
N_ASSET_F        = 33
N_REGIME_F       = 22

N_BLOCKS_15M     = 1
TIME_STEPS_15M   = 128
N_LOWRES_F       = 12

TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}
MIN_PRICE        = 2.00

EXCLUDE_TICKERS = {'SPY', 'VIX', 'TLT', 'IWM', 'XLE', 'ITA', 'BITF', 'MIGI', 'SMLR'}

LOWRES_COLS = [
    'lr_Close_ret', 'lr_High_ret', 'lr_Low_ret', 'lr_Volume_norm',
    'lr_Range', 'lr_RSI_14', 'lr_RVOL', 'lr_ROC_4', 'lr_ROC_8',
    'lr_ROC_16', 'lr_Bar_Is_Real', 'lr_Time_Pos',
]

MICRO_COLS = ['Spread', 'OBI', 'Volume_Delta', 'Trade_Count', 'Delta_Pct',
              'bsize', 'asize', 'Total_Vol']
ROC_HORIZONS = [5, 30, 60]
ASSET_COLS = (['Gap_Pct', 'RSI_14', 'RVOL', 'ATR_14', 'Bar_Is_Real'] +
              [f'ROC_{h}m' for h in ROC_HORIZONS] + MICRO_COLS)

CONTEXT_ASSETS_LIST = ['SPY', 'VIX', 'TLT', 'IWM']
SECTOR_PROXY_ASSETS = ['MARA', 'MSTR', 'ITA']

CONTEXT_COLS = []
for _t in CONTEXT_ASSETS_LIST:
    CONTEXT_COLS.append(f'{_t}_Anchor_ROC')
    for _h in ROC_HORIZONS:
        CONTEXT_COLS.append(f'{_t}_ROC_{_h}m')
CONTEXT_COLS.append('Time_Pos')

ALL_COLS = ASSET_COLS + CONTEXT_COLS

REGIME_COLS = [
    'regime_rvol_15', 'regime_rvol_60', 'regime_rvol_128',
    'regime_trend_60', 'regime_trend_128', 'regime_autocorr_60',
    'regime_xsect_disp', 'regime_vix_level', 'regime_vix_velocity',
    'regime_time_sin', 'regime_time_cos', 'regime_spy_ret_60',
    'regime_vix_close', 'regime_spy_tlt_corr', 'regime_breadth',
    'regime_skew_60',
    'regime_crypto_ret_60', 'regime_crypto_ret_128', 'regime_crypto_rvol',
    'regime_growth_vs_value', 'regime_defense_rs', 'regime_safe_haven',
]

HORIZON_LABELS = {0: '15m', 1: '30m', 2: '60m'}
HORIZON_FROM_LABEL = {'15m': 0, '30m': 1, '60m': 2}

W = 120
SEED = 42
np.random.seed(SEED)

gcs_client = gcs_storage.Client()

_NEEDED_COLS = {'Open', 'High', 'Low', 'Close', 'Volume',
                'open', 'high', 'low', 'close', 'volume',
                'Spread', 'OBI', 'Volume_Delta', 'Trade_Count', 'Delta_Pct',
                'bsize', 'asize', 'Total_Vol',
                'Total_vol', 'Trade_count', 'Volume_delta', 'Delta_pct',
                'Obi', 'Bsize', 'Asize',
                'Datetime', 'Date', 'Time', 'Timestamp', 'Level_0'}


# ==============================================================================
# CONFIG — PER-DIRECTION
# ==============================================================================
LONG_FAMILIES = {
    'L5-1': {
        'gcs_data':   'Algos/RA_MoE/long_L5_1_data',
        'gcs_models': 'Algos/RA_MoE/long_L5_1_models',
        'arch': 'standard',
        'bq_versions': {
            5: 'spike_transformer_long_L5_1_fold5',
            6: 'spike_transformer_long_L5_1_fold6',
            7: 'spike_transformer_long_L5_1_fold7',
        },
    },
    'ASL6': {
        'gcs_data':   'Algos/RA_MoE/long_L5_2_data',
        'gcs_models': 'Algos/RA_MoE/long_L5_2_asl6_models',
        'arch': 'standard',
        'bq_versions': {
            5: 'spike_transformer_long_L5_2_asl6_fold5',
            6: 'spike_transformer_long_L5_2_asl6_fold6',
            7: 'spike_transformer_long_L5_2_asl6_fold7',
        },
    },
    'MR-v2': {
        'gcs_data':   'Algos/RA_MoE/long_L5_1_multires_v2_data',
        'gcs_models': 'Algos/RA_MoE/long_L5_1_multires_v2_models',
        'arch': 'multires',
        'bq_versions': {
            6: 'spike_transformer_long_L5_1_multires_v2_fold6',
            7: 'spike_transformer_long_L5_1_multires_v2_fold7',
            8: 'spike_transformer_long_L5_1_multires_v2_fold8',
            9: 'spike_transformer_long_L5_1_multires_v2_fold9',
        },
    },
}
LONG_CONSENSUS = {('MR-v2', 9), ('MR-v2', 8), ('MR-v2', 7),
                   ('L5-1', 7), ('ASL6', 7),
                   ('L5-1', 6)}
LONG_NEWEST_FIRST = [
    ('MR-v2', 9), ('MR-v2', 8), ('MR-v2', 7),
    ('L5-1', 7), ('ASL6', 7),
    ('L5-1', 6), ('ASL6', 6),
]
LONG_F7_THRESH = 0.005

SHORT_FAMILIES = {
    'MR-v2': {
        'gcs_data':   'Algos/RA_MoE/short_L5_2_multires_v2_data',
        'gcs_models': 'Algos/RA_MoE/short_L5_2_multires_v2_models',
        'arch': 'multires',
        'bq_versions': {
            6: 'spike_transformer_short_L5_2_multires_v2_fold6',
            7: 'spike_transformer_short_L5_2_multires_v2_fold7',
            8: 'spike_transformer_short_L5_2_multires_v2_fold8',
            9: 'spike_transformer_short_L5_2_multires_v2_fold9',
            10: 'spike_transformer_short_L5_2_multires_v2_fold10',
        },
    },
}
SHORT_CONSENSUS = {('MR-v2', 8), ('MR-v2', 9), ('MR-v2', 10)}
SHORT_F7_THRESH = 0.030


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
    blob_path = f"{INFERENCE_DIR}/{file_ticker}_master_inference.parquet"
    local_path = f"/tmp/_bt_comb_{file_ticker}_master.parquet"
    try:
        gcs_client.bucket(BUCKET_NAME).blob(blob_path).download_to_filename(local_path)
        import pyarrow.parquet as pq
        schema = pq.read_schema(local_path)
        all_cols = [f.name for f in schema]
        use_cols = [c for c in all_cols if c in _NEEDED_COLS]
        if not use_cols:
            use_cols = None
        df = pd.read_parquet(local_path, columns=use_cols)
        os.remove(local_path)
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
        if os.path.exists(local_path):
            os.remove(local_path)
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
# MARKET CONTEXT & FEATURE ENGINEERING
# ==============================================================================
def build_market_context():
    frames = []
    vix_close = None
    for ca in CONTEXT_ASSETS_LIST:
        df = load_master(ca, tail_days=TAIL_DAYS)
        if df is None:
            raise RuntimeError(f"Cannot load context asset {ca}")
        df = heal_data(df.sort_index())
        if ca == 'VIX':
            vix_close = df['Close'].copy()
        cols = {f'{ca}_Anchor_ROC': df['Close'].pct_change(1)}
        for h in ROC_HORIZONS:
            cols[f'{ca}_ROC_{h}m'] = df['Close'].pct_change(h)
        frames.append(pd.DataFrame(cols, index=df.index))
    result = pd.concat(frames, axis=1).ffill().fillna(0)
    if vix_close is not None:
        result['VIX_Close'] = vix_close.reindex(result.index).ffill().fillna(0)
    for sp in SECTOR_PROXY_ASSETS:
        df = load_master(sp, tail_days=TAIL_DAYS)
        if df is not None:
            df = heal_data(df.sort_index())
            result[f'{sp}_ret'] = df['Close'].pct_change(1).reindex(
                result.index).ffill().fillna(0)
    return result


def engineer_asset(ticker, market_df):
    df = load_master(ticker, tail_days=TAIL_DAYS)
    if df is None:
        return None
    df = heal_data(df.sort_index())
    daily_c = df['Close'].resample('D').last().ffill().shift(1).reindex(
        df.index, method='ffill')
    df['Gap_Pct'] = (df['Open'] - daily_c) / daily_c
    df['RSI_14'] = ta.rsi(df['Close'], length=14)
    vol_real = df['Volume'].where(df['Volume'] > 0)
    df['RVOL'] = df['Volume'] / vol_real.rolling(50, min_periods=5).mean().ffill().clip(lower=1e-9)
    real_mask = df['Volume'] > 0
    atr_raw = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['ATR_14'] = atr_raw.where(real_mask).ffill().fillna(0)
    for h in ROC_HORIZONS:
        df[f'ROC_{h}m'] = df['Close'].pct_change(h)
    df = df.join(market_df, how='left').ffill().fillna(0)
    df_pst = df.index.tz_convert(PST)
    mins = df_pst.hour * 60 + df_pst.minute
    df['Time_Pos'] = np.clip((mins - 390) / 390, 0, 1)
    df['Bar_Is_Real'] = (df['Volume'] > 0).astype(float)
    return df.replace([np.inf, -np.inf], np.nan).fillna(0)


def build_15min_features(df_1m):
    ohlcv_15 = df_1m[['Open', 'High', 'Low', 'Close', 'Volume']].resample('15min').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).ffill()
    lr = pd.DataFrame(index=ohlcv_15.index)
    lr['lr_Close_ret'] = ohlcv_15['Close'].pct_change(1)
    lr['lr_High_ret'] = (ohlcv_15['High'] - ohlcv_15['Open']) / ohlcv_15['Open'].clip(lower=1e-8)
    lr['lr_Low_ret'] = (ohlcv_15['Low'] - ohlcv_15['Open']) / ohlcv_15['Open'].clip(lower=1e-8)
    vol_real = ohlcv_15['Volume'].where(ohlcv_15['Volume'] > 0)
    vol_mean = vol_real.rolling(50, min_periods=5).mean().ffill().clip(lower=1e-9)
    lr['lr_Volume_norm'] = (ohlcv_15['Volume'] / vol_mean).fillna(1.0)
    lr['lr_Range'] = (ohlcv_15['High'] - ohlcv_15['Low']) / ohlcv_15['Close'].clip(lower=1e-8)
    lr['lr_RSI_14'] = ta.rsi(ohlcv_15['Close'], length=14)
    lr['lr_RVOL'] = lr['lr_Volume_norm']
    lr['lr_ROC_4'] = ohlcv_15['Close'].pct_change(4)
    lr['lr_ROC_8'] = ohlcv_15['Close'].pct_change(8)
    lr['lr_ROC_16'] = ohlcv_15['Close'].pct_change(16)
    lr['lr_Bar_Is_Real'] = (ohlcv_15['Volume'] > 0).astype(float)
    try:
        idx_pst = ohlcv_15.index.tz_convert(PST)
    except Exception:
        idx_pst = ohlcv_15.index
    mins = idx_pst.hour * 60 + idx_pst.minute
    lr['lr_Time_Pos'] = np.clip((mins - 390) / 390, 0, 1)
    return lr[LOWRES_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)


def compute_regime_features(market_df, universe_returns=None):
    rf = pd.DataFrame(index=market_df.index)
    spy_ret = market_df['SPY_Anchor_ROC'].fillna(0)
    for w, name in [(15, 'regime_rvol_15'), (60, 'regime_rvol_60'),
                     (128, 'regime_rvol_128')]:
        rf[name] = spy_ret.rolling(w, min_periods=max(w // 2, 1)).std().fillna(0)
    for w, name in [(60, 'regime_trend_60'), (128, 'regime_trend_128')]:
        cum_ret = spy_ret.rolling(w, min_periods=max(w // 2, 1)).sum()
        rvol = rf[f'regime_rvol_{w}'].clip(lower=1e-8)
        rf[name] = (cum_ret / rvol).fillna(0).clip(-10, 10)
    spy_vals = spy_ret.values.astype(np.float64)
    n = len(spy_vals)
    autocorr = np.full(n, 0.0)
    for i in range(60, n):
        chunk = spy_vals[i - 60:i]
        if np.count_nonzero(~np.isnan(chunk)) >= 30:
            x, y = chunk[:-1], chunk[1:]
            mx, my = x.mean(), y.mean()
            sx, sy = x.std(), y.std()
            if sx > 1e-12 and sy > 1e-12:
                autocorr[i] = ((x - mx) * (y - my)).mean() / (sx * sy)
    rf['regime_autocorr_60'] = autocorr
    if universe_returns is not None and len(universe_returns.columns) > 1:
        rf['regime_xsect_disp'] = universe_returns.std(axis=1).fillna(0)
    else:
        rf['regime_xsect_disp'] = spy_ret.rolling(60).std().fillna(0)
    vix_col = 'VIX_Anchor_ROC'
    if vix_col in market_df.columns:
        vix_cum = market_df[vix_col].cumsum()
        rf['regime_vix_level'] = vix_cum.rolling(60, min_periods=1).mean().fillna(0)
        rf['regime_vix_velocity'] = vix_cum.diff(15).fillna(0)
    else:
        rf['regime_vix_level'] = 0.0
        rf['regime_vix_velocity'] = 0.0
    try:
        idx_pst = market_df.index.tz_convert(PST)
    except Exception:
        idx_pst = market_df.index.tz_localize('UTC').tz_convert(PST)
    mins_of_day = idx_pst.hour * 60 + idx_pst.minute
    rf['regime_time_sin'] = np.sin(2 * np.pi * mins_of_day / 1440)
    rf['regime_time_cos'] = np.cos(2 * np.pi * mins_of_day / 1440)
    rf['regime_spy_ret_60'] = spy_ret.rolling(60, min_periods=30).sum().fillna(0)
    rf['regime_vix_close'] = (market_df['VIX_Close'].ffill().fillna(0)
                               if 'VIX_Close' in market_df.columns
                               else pd.Series(0.0, index=market_df.index))
    tlt_ret = (market_df['TLT_Anchor_ROC'].fillna(0)
               if 'TLT_Anchor_ROC' in market_df.columns
               else pd.Series(0.0, index=market_df.index))
    rf['regime_spy_tlt_corr'] = spy_ret.rolling(
        60, min_periods=30).corr(tlt_ret).fillna(0).clip(-1, 1)
    if universe_returns is not None and len(universe_returns.columns) > 1:
        cum_rets_60 = universe_returns.rolling(60, min_periods=30).sum()
        rf['regime_breadth'] = (cum_rets_60 > 0).mean(axis=1).fillna(0.5)
    else:
        rf['regime_breadth'] = 0.5
    rolling_mean = spy_ret.rolling(60, min_periods=30).mean()
    rolling_std = spy_ret.rolling(60, min_periods=30).std().clip(lower=1e-12)
    standardized = (spy_ret - rolling_mean) / rolling_std
    rf['regime_skew_60'] = standardized.pow(3).rolling(
        60, min_periods=30).mean().fillna(0).clip(-5, 5)
    crypto_ret = (market_df['MARA_ret'].fillna(0)
                  if 'MARA_ret' in market_df.columns
                  else pd.Series(0.0, index=market_df.index))
    rf['regime_crypto_ret_60'] = crypto_ret.rolling(60, min_periods=30).sum().fillna(0)
    rf['regime_crypto_ret_128'] = crypto_ret.rolling(128, min_periods=60).sum().fillna(0)
    rf['regime_crypto_rvol'] = crypto_ret.rolling(60, min_periods=30).std().fillna(0)
    iwm_ret = (market_df['IWM_Anchor_ROC'].fillna(0)
               if 'IWM_Anchor_ROC' in market_df.columns
               else pd.Series(0.0, index=market_df.index))
    rf['regime_growth_vs_value'] = (
        iwm_ret.rolling(60, min_periods=30).sum() -
        spy_ret.rolling(60, min_periods=30).sum()).fillna(0)
    defense_ret = (market_df['ITA_ret'].fillna(0)
                   if 'ITA_ret' in market_df.columns
                   else pd.Series(0.0, index=market_df.index))
    rf['regime_defense_rs'] = (
        defense_ret.rolling(60, min_periods=30).sum() -
        spy_ret.rolling(60, min_periods=30).sum()).fillna(0)
    rf['regime_safe_haven'] = (
        tlt_ret.rolling(60, min_periods=30).sum() -
        spy_ret.rolling(60, min_periods=30).sum()).fillna(0)
    return rf[REGIME_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)


def apply_scaling(df, stats, cols):
    df_s = df.copy()
    for col in cols:
        if col in ('Time_Pos', 'Bar_Is_Real'):
            continue
        if col in stats:
            df_s[col] = ((df_s[col] - stats[col]['med'])
                         / stats[col]['iqr']).clip(-10, 10).astype('float32')
    return df_s


# ==============================================================================
# MODEL ARCHITECTURE
# ==============================================================================
class _SeriesStationarization(_kl.Layer):
    def call(self, x):
        mean = tf.reduce_mean(x, axis=1, keepdims=True)
        std = tf.math.reduce_std(x, axis=1, keepdims=True) + 1e-5
        return (x - mean) / std, tf.concat([mean[:, 0, :], std[:, 0, :]], axis=-1)

class _DeStationaryAttention(_kl.Layer):
    def __init__(self, d_model, n_heads, dropout=0.1, **kw):
        super().__init__(**kw)
        self.d_model, self.n_heads = d_model, n_heads
        self.d_head = d_model // n_heads
        self.scale = tf.sqrt(tf.cast(self.d_head, tf.float32))
        self.wq = Dense(d_model); self.wk = Dense(d_model)
        self.wv = Dense(d_model); self.wo = Dense(d_model)
        self.attn_drop = Dropout(dropout)
        self.tau_proj = Dense(n_heads)
        self.sigma_proj = Dense(d_model)
    def call(self, x, stats, training=False):
        B, T = tf.shape(x)[0], tf.shape(x)[1]
        H, D = self.n_heads, self.d_head
        Q = tf.transpose(tf.reshape(self.wq(x), (B, T, H, D)), (0, 2, 1, 3))
        K = tf.transpose(tf.reshape(self.wk(x), (B, T, H, D)), (0, 2, 1, 3))
        V = tf.transpose(tf.reshape(self.wv(x), (B, T, H, D)), (0, 2, 1, 3))
        scores = tf.matmul(Q, K, transpose_b=True) / tf.cast(self.scale, Q.dtype)
        tau = tf.reshape(self.tau_proj(stats), (B, H, 1, 1))
        scores = scores + tau
        attn = self.attn_drop(tf.nn.softmax(scores, axis=-1), training=training)
        out = tf.reshape(tf.transpose(tf.matmul(attn, V), (0, 2, 1, 3)),
                         (B, T, self.d_model))
        sigma = tf.nn.softplus(self.sigma_proj(stats))
        return self.wo(out * sigma[:, tf.newaxis, :])

class TransformerBlock(_kl.Layer):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, **kw):
        super().__init__(**kw)
        self.attn = _DeStationaryAttention(d_model, n_heads, dropout, name='destat_attn')
        self.ffn = keras.Sequential([
            Dense(d_ff, activation='gelu'), Dense(d_model), Dropout(dropout)])
        self.norm1 = LayerNormalization()
        self.norm2 = LayerNormalization()
        self.drop = Dropout(dropout)
    def call(self, x, stats, training=False):
        x = x + self.drop(self.attn(self.norm1(x), stats, training=training),
                          training=training)
        return x + self.ffn(self.norm2(x), training=training)

class TransformerBackbone(_kl.Layer):
    def __init__(self, d_model, n_heads, d_ff, n_blocks, seq_len,
                 dropout=0.1, **kw):
        super().__init__(**kw)
        self.stationarize = _SeriesStationarization(name='stationarize')
        self.input_proj = Dense(d_model)
        self.pos_embed = self.add_weight(
            name='pos_embed', shape=(1, seq_len, d_model),
            initializer='glorot_normal', trainable=True)
        self.input_drop = Dropout(dropout)
        self.blocks = [TransformerBlock(d_model, n_heads, d_ff, dropout,
                                        name=f'block_{i}')
                       for i in range(n_blocks)]
        self.final_norm = LayerNormalization()
    def call(self, x, training=False):
        x_norm, stats = self.stationarize(x)
        h = self.input_proj(x_norm) + self.pos_embed
        h = self.input_drop(h, training=training)
        for block in self.blocks:
            h = block(h, stats, training=training)
        return self.final_norm(h)

class _AttentionPooling(_kl.Layer):
    def __init__(self, d_model, **kw):
        super().__init__(**kw)
        self.query = self.add_weight(
            name='pool_query', shape=(1, 1, d_model),
            initializer='glorot_normal', trainable=True)
        self.scale = tf.sqrt(tf.cast(d_model, tf.float32))
    def call(self, x, training=False):
        scores = tf.matmul(self.query, x, transpose_b=True) / tf.cast(self.scale, x.dtype)
        return tf.matmul(tf.nn.softmax(scores, axis=-1), x)[:, 0, :]

class _FiLMConditioner(_kl.Layer):
    def __init__(self, d_model, hidden_dim=64, **kw):
        super().__init__(**kw)
        self.mlp = keras.Sequential([
            Dense(hidden_dim, activation='relu'),
            Dense(hidden_dim, activation='relu')])
        self.gamma_proj = Dense(d_model)
        self.beta_proj = Dense(d_model)
    def call(self, x, regime_features, training=False):
        h = self.mlp(regime_features, training=training)
        return (1.0 + self.gamma_proj(h)) * x + self.beta_proj(h)

class SpikeTransformer(keras.Model):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.backbone = TransformerBackbone(
            D_MODEL, N_HEADS, D_FF, N_BLOCKS,
            TIME_STEPS, DROPOUT_RATE, name='backbone')
        self.attn_pool = _AttentionPooling(D_MODEL, name='attn_pool')
        self.film = _FiLMConditioner(D_MODEL, FILM_HIDDEN, name='film')
        self.drop = Dropout(DROPOUT_RATE)
        self.head = Dense(N_HORIZONS, activation='sigmoid', name='spike_head')
    def call(self, inputs, training=False):
        h = self.backbone(inputs['seq_input'], training=training)
        h = self.attn_pool(h, training=training)
        h = self.film(h, inputs['regime_input'], training=training)
        return self.head(self.drop(h, training=training))

class MultiResSpikeTransformer(keras.Model):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.backbone_1m = TransformerBackbone(
            D_MODEL, N_HEADS, D_FF, N_BLOCKS,
            TIME_STEPS, DROPOUT_RATE, name='backbone_1m')
        self.pool_1m = _AttentionPooling(D_MODEL, name='pool_1m')
        self.backbone_15m = TransformerBackbone(
            D_MODEL, N_HEADS, D_FF, N_BLOCKS_15M,
            TIME_STEPS_15M, DROPOUT_RATE, name='backbone_15m')
        self.pool_15m = _AttentionPooling(D_MODEL, name='pool_15m')
        self.fusion = Dense(D_MODEL, activation='gelu', name='fusion')
        self.film = _FiLMConditioner(D_MODEL, FILM_HIDDEN, name='film')
        self.drop = Dropout(DROPOUT_RATE)
        self.head = Dense(N_HORIZONS, activation='sigmoid', name='spike_head')
    def call(self, inputs, training=False):
        h_1m = self.backbone_1m(inputs['seq_input'], training=training)
        h_1m = self.pool_1m(h_1m, training=training)
        h_15m = self.backbone_15m(inputs['seq_15m_input'], training=training)
        h_15m = self.pool_15m(h_15m, training=training)
        h = self.fusion(tf.concat([h_1m, h_15m], axis=-1))
        h = self.film(h, inputs['regime_input'], training=training)
        return self.head(self.drop(h, training=training))


# ==============================================================================
# H5PY WEIGHT LOADING
# ==============================================================================
def _load_keras3_weights(model, filepath):
    with h5py.File(filepath, 'r') as f:
        _load_layer_recursive(model, f)

def _load_layer_recursive(layer, h5_group):
    if 'vars' in h5_group:
        vg = h5_group['vars']
        own_vars = layer.trainable_weights + layer.non_trainable_weights
        for i, var in enumerate(own_vars):
            key = str(i)
            if key in vg:
                var.assign(np.array(vg[key]))
    for key in sorted(h5_group.keys()):
        if key in ('vars', 'optimizer'):
            continue
        if not isinstance(h5_group[key], h5py.Group):
            continue
        child = getattr(layer, key, None)
        if child is None:
            continue
        if isinstance(child, _kl.Layer):
            _load_layer_recursive(child, h5_group[key])
        elif isinstance(child, (list, tuple)):
            _load_list_recursive(child, h5_group[key])

def _load_list_recursive(items, h5_group):
    used_names = {}
    for item in items:
        if not isinstance(item, _kl.Layer):
            continue
        cname = type(item).__name__
        snake = ''.join(f'_{c.lower()}' if c.isupper() else c
                        for c in cname).lstrip('_')
        if snake in used_names:
            used_names[snake] += 1
            snake = f"{snake}_{used_names[snake]}"
        else:
            used_names[snake] = 0
        if snake in h5_group:
            _load_layer_recursive(item, h5_group[snake])

def _ensure_file(gcs_path, local_path):
    if os.path.exists(local_path):
        return local_path
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    gcs_client.bucket(BUCKET_NAME).blob(gcs_path).download_to_filename(local_path)
    return local_path


# ==============================================================================
# DIRECTION STATE — holds per-direction models, weights, meta
# ==============================================================================
class DirState:
    def __init__(self, label, families):
        self.label = label
        self.families = families
        self.model_std = None
        self.model_mr = None
        self.fold_weights = {}
        self.current_key_std = None
        self.current_key_mr = None
        self.family_meta = {}

    def get_model(self, family):
        arch = self.families[family].get('arch', 'standard')
        return self.model_mr if arch == 'multires' else self.model_std

    def ensure_fold_loaded(self, family, fold_id):
        key = (family, fold_id)
        arch = self.families[family].get('arch', 'standard')
        if arch == 'multires':
            if self.current_key_mr != key:
                _load_keras3_weights(self.model_mr, self.fold_weights[key])
                self.current_key_mr = key
        else:
            if self.current_key_std != key:
                _load_keras3_weights(self.model_std, self.fold_weights[key])
                self.current_key_std = key


def build_models_for_dir(ds, deploy_configs, family_fold_cfgs):
    all_cfgs = list(deploy_configs.values())
    if family_fold_cfgs:
        for cfgs in family_fold_cfgs.values():
            if isinstance(cfgs, dict):
                all_cfgs.extend(cfgs.values())
    needed_families = sorted(set(cfg['family'] for cfg in all_cfgs))
    for family_name in needed_families:
        fam = ds.families[family_name]
        local_meta = f'/tmp/ra_moe_{ds.label}_{family_name}_meta.npz'
        meta_path = _ensure_file(f"{fam['gcs_data']}/meta.npz", local_meta)
        meta = np.load(meta_path, allow_pickle=True)
        meta_dict = {
            'scaling_stats': meta['scaling_stats'].item(),
            'regime_stats': meta['regime_stats'].item(),
        }
        if 'scaling_stats_15m' in meta:
            meta_dict['scaling_stats_15m'] = meta['scaling_stats_15m'].item()
        ds.family_meta[family_name] = meta_dict

    need_std = any(ds.families[f].get('arch', 'standard') == 'standard'
                   for f in needed_families)
    need_mr = any(ds.families[f].get('arch', 'standard') == 'multires'
                  for f in needed_families)

    if need_std:
        ds.model_std = SpikeTransformer(name=f'st_{ds.label}')
        ds.model_std({'seq_input': tf.zeros((1, TIME_STEPS, N_ASSET_F)),
                      'regime_input': tf.zeros((1, N_REGIME_F))})
        print(f"   {ds.label} SpikeTransformer: {ds.model_std.count_params():,} params")

    if need_mr:
        ds.model_mr = MultiResSpikeTransformer(name=f'mr_{ds.label}')
        ds.model_mr({'seq_input': tf.zeros((1, TIME_STEPS, N_ASSET_F)),
                     'seq_15m_input': tf.zeros((1, TIME_STEPS_15M, N_LOWRES_F)),
                     'regime_input': tf.zeros((1, N_REGIME_F))})
        print(f"   {ds.label} MultiResSpikeTransformer: {ds.model_mr.count_params():,} params")

    needed_keys = sorted(set((cfg['family'], cfg['fold']) for cfg in all_cfgs))
    for family_name, fold_id in needed_keys:
        fam = ds.families[family_name]
        local_dir = f'/tmp/ra_moe_{ds.label}_{family_name}_models/fold_{fold_id}'
        weights_path = os.path.join(local_dir, 'model.weights.h5')
        if not os.path.exists(weights_path):
            os.makedirs(local_dir, exist_ok=True)
            prefix = f"{fam['gcs_models']}/fold_{fold_id}"
            bucket = gcs_client.bucket(BUCKET_NAME)
            for blob in bucket.list_blobs(prefix=prefix):
                fname = blob.name.split('/')[-1]
                blob.download_to_filename(os.path.join(local_dir, fname))
        ds.fold_weights[(family_name, fold_id)] = weights_path

    if need_std:
        std_keys = [k for k in needed_keys
                    if ds.families[k[0]].get('arch', 'standard') == 'standard']
        if std_keys:
            _load_keras3_weights(ds.model_std, ds.fold_weights[std_keys[0]])
            ds.current_key_std = std_keys[0]
    if need_mr:
        mr_keys = [k for k in needed_keys
                   if ds.families[k[0]].get('arch', 'standard') == 'multires']
        if mr_keys:
            _load_keras3_weights(ds.model_mr, ds.fold_weights[mr_keys[0]])
            ds.current_key_mr = mr_keys[0]
    print(f"   {ds.label}: {len(needed_keys)} (family,fold) combos ready")


# ==============================================================================
# BQ CONFIG LOADING
# ==============================================================================
def load_bq_configs(families, f7_thresh, label):
    if bigquery is None:
        raise RuntimeError("BigQuery not available")
    bq_client = bigquery.Client()
    family_fold_cfgs = {}
    for family_name, fam in families.items():
        for fold_id, model_version in fam['bq_versions'].items():
            q = f"""
            SELECT ticker, experiment, tpsl_mode, prob_thresh, tp_pct, sl_pct,
                   grace_minutes, dip_pct, max_wait_minutes,
                   val_b_avg_pnl, val_b_win_rate, val_b_count
            FROM `{WF_BQ_TABLE}`
            WHERE model_version = '{model_version}'
              AND survived_a = TRUE
              AND val_b_avg_pnl >= {BASE_PNL_THRESH}
            """
            rows = bq_client.query(q).to_dataframe()
            if len(rows) == 0:
                family_fold_cfgs[(family_name, fold_id)] = {}
                print(f"    {label} {family_name} F{fold_id}: 0 tickers")
                continue
            best = rows.loc[rows.groupby('ticker')['val_b_avg_pnl'].idxmax()]
            cfgs = {}
            for _, row in best.iterrows():
                tk = row['ticker']
                if tk in EXCLUDE_TICKERS:
                    continue
                exp = row['experiment']
                h_part = exp.replace(f'{model_version}_', '').split('_')[0]
                horizon_idx = HORIZON_FROM_LABEL.get(h_part, 1)
                mode = row['tpsl_mode']
                if pd.isna(mode) or mode == '':
                    if '_GRACE' in exp:
                        mode = 'grace'
                    elif '_LIMIT' in exp:
                        mode = 'limit_entry'
                    else:
                        mode = 'static'
                cfgs[tk] = {
                    'family': family_name, 'fold': fold_id,
                    'horizon': horizon_idx,
                    'prob_thresh': float(row['prob_thresh']),
                    'tp_pct': float(row['tp_pct']),
                    'sl_pct': float(row['sl_pct']),
                    'mode': mode,
                    'grace_minutes': int(row['grace_minutes']) if pd.notna(row['grace_minutes']) else 0,
                    'dip_pct': float(row['dip_pct']) if pd.notna(row['dip_pct']) else 0,
                    'max_wait': int(row['max_wait_minutes']) if pd.notna(row['max_wait_minutes']) else 0,
                    'val_b_avg_pnl': float(row['val_b_avg_pnl']),
                }
            family_fold_cfgs[(family_name, fold_id)] = cfgs
            print(f"    {label} {family_name} F{fold_id}: {len(cfgs)} tickers")
    return family_fold_cfgs


def select_long_deploy(family_fold_cfgs):
    deploy = {}
    for family, fold in LONG_NEWEST_FIRST:
        for tk, cfg in family_fold_cfgs.get((family, fold), {}).items():
            if tk not in deploy:
                deploy[tk] = cfg
    print(f"    LONG deploy: {len(deploy)} tickers (Newest First)")
    return deploy


def select_short_deploy(family_fold_cfgs):
    all_tickers = set()
    for cfgs in family_fold_cfgs.values():
        all_tickers |= set(cfgs.keys())
    deploy = {}
    for tk in sorted(all_tickers):
        best_val, best_cfg = -999, None
        for (family_name, fold_id), cfgs in family_fold_cfgs.items():
            if tk not in cfgs:
                continue
            cfg = cfgs[tk]
            v = cfg['val_b_avg_pnl']
            if fold_id >= 7 and v < SHORT_F7_THRESH:
                continue
            if v > best_val:
                best_val = v
                best_cfg = cfg
        if best_cfg is not None:
            deploy[tk] = best_cfg
    print(f"    SHORT deploy: {len(deploy)} tickers (Hybrid)")
    return deploy


# ==============================================================================
# NUMBA TRADE SIMS — LONG DIRECTION
# ==============================================================================
@jit(nopython=True)
def sim_long_static(entry_indices, prices, highs, lows, date_ints, time_mins,
                    tp, sl, exit_time):
    ne = len(entry_indices); np_ = len(prices)
    pnl_a = np.empty(ne, np.float64); date_a = np.empty(ne, np.int64)
    entry_a = np.empty(ne, np.float64); exit_a = np.empty(ne, np.float64)
    out_a = np.empty(ne, np.int64); tc = 0; cp = 0; ltd = -1
    while cp < ne:
        idx = entry_indices[cp]
        if idx >= np_ - 1: break
        ed = date_ints[idx]
        if ed == ltd: cp += 1; continue
        ep = prices[idx]; tgt = ep * (1 + tp); stp = ep * (1 - sl)
        oc = 0.0; xp = ep; xt = 0; done = False; xi = idx
        for k in range(idx + 1, np_):
            if date_ints[k] != ed:
                xp = prices[k-1]; oc = (xp - ep) / ep; xi = k; done = True; break
            sl_h = lows[k] <= stp; tp_h = highs[k] >= tgt
            if sl_h and tp_h:
                if np.random.random() < 0.5: oc = -sl; xp = stp; xt = -1
                else: oc = tp; xp = tgt; xt = 1
                xi = k; done = True; break
            if sl_h: oc = -sl; xp = stp; xt = -1; xi = k; done = True; break
            if tp_h: oc = tp; xp = tgt; xt = 1; xi = k; done = True; break
            if time_mins[k] >= exit_time:
                xp = prices[k]; oc = (xp - ep) / ep; xi = k; done = True; break
        if not done: xp = prices[np_-1]; oc = (xp - ep) / ep; xi = np_
        pnl_a[tc] = oc; date_a[tc] = ed; entry_a[tc] = ep; exit_a[tc] = xp; out_a[tc] = xt
        tc += 1; ltd = ed; cp += 1
    return pnl_a[:tc], date_a[:tc], entry_a[:tc], exit_a[:tc], out_a[:tc]

@jit(nopython=True)
def sim_long_grace(entry_indices, prices, highs, lows, date_ints, time_mins,
                   tp, sl, exit_time, grace_minutes):
    ne = len(entry_indices); np_ = len(prices)
    pnl_a = np.empty(ne, np.float64); date_a = np.empty(ne, np.int64)
    entry_a = np.empty(ne, np.float64); exit_a = np.empty(ne, np.float64)
    out_a = np.empty(ne, np.int64); tc = 0; cp = 0; ltd = -1
    while cp < ne:
        idx = entry_indices[cp]
        if idx >= np_ - 1: break
        ed = date_ints[idx]
        if ed == ltd: cp += 1; continue
        ep = prices[idx]; tgt = ep * (1 + tp); stp = ep * (1 - sl)
        et = time_mins[idx]; oc = 0.0; xp = ep; xt = 0; done = False
        for k in range(idx + 1, np_):
            if date_ints[k] != ed:
                xp = prices[k-1]; oc = (xp - ep) / ep; done = True; break
            elapsed = time_mins[k] - et; sl_act = elapsed >= grace_minutes
            tp_h = highs[k] >= tgt; sl_h = sl_act and (lows[k] <= stp)
            if sl_h and tp_h:
                if np.random.random() < 0.5: oc = -sl; xp = stp; xt = -1
                else: oc = tp; xp = tgt; xt = 1
                done = True; break
            if sl_h: oc = -sl; xp = stp; xt = -1; done = True; break
            if tp_h: oc = tp; xp = tgt; xt = 1; done = True; break
            if time_mins[k] >= exit_time:
                xp = prices[k]; oc = (xp - ep) / ep; done = True; break
        if not done: xp = prices[np_-1]; oc = (xp - ep) / ep
        pnl_a[tc] = oc; date_a[tc] = ed; entry_a[tc] = ep; exit_a[tc] = xp; out_a[tc] = xt
        tc += 1; ltd = ed; cp += 1
    return pnl_a[:tc], date_a[:tc], entry_a[:tc], exit_a[:tc], out_a[:tc]

@jit(nopython=True)
def sim_long_limit(entry_indices, prices, highs, lows, date_ints, time_mins,
                   tp, sl, exit_time, dip_pct, max_wait):
    ne = len(entry_indices); np_ = len(prices)
    pnl_a = np.empty(ne, np.float64); date_a = np.empty(ne, np.int64)
    entry_a = np.empty(ne, np.float64); exit_a = np.empty(ne, np.float64)
    out_a = np.empty(ne, np.int64); tc = 0; cp = 0; ltd = -1
    while cp < ne:
        idx = entry_indices[cp]
        if idx >= np_ - 1: break
        ed = date_ints[idx]
        if ed == ltd: cp += 1; continue
        sp = prices[idx]; lp = sp * (1.0 - dip_pct); st = time_mins[idx]
        fi = -1; fp = 0.0
        for k in range(idx + 1, np_):
            if date_ints[k] != ed: break
            if time_mins[k] - st > max_wait: break
            if lows[k] <= lp: fp = lp; fi = k; break
        if fi < 0: cp += 1; ltd = ed; continue
        tgt = fp * (1 + tp); stp = fp * (1 - sl)
        oc = 0.0; xp = fp; xt = 0; done = False
        for k in range(fi + 1, np_):
            if date_ints[k] != ed:
                xp = prices[k-1]; oc = (xp - fp) / fp; done = True; break
            sl_h = lows[k] <= stp; tp_h = highs[k] >= tgt
            if sl_h and tp_h:
                if np.random.random() < 0.5: oc = -sl; xp = stp; xt = -1
                else: oc = tp; xp = tgt; xt = 1
                done = True; break
            if sl_h: oc = -sl; xp = stp; xt = -1; done = True; break
            if tp_h: oc = tp; xp = tgt; xt = 1; done = True; break
            if time_mins[k] >= exit_time:
                xp = prices[k]; oc = (xp - fp) / fp; done = True; break
        if not done: xp = prices[np_-1]; oc = (xp - fp) / fp
        pnl_a[tc] = oc; date_a[tc] = ed; entry_a[tc] = fp; exit_a[tc] = xp; out_a[tc] = xt
        tc += 1; ltd = ed; cp += 1
    return pnl_a[:tc], date_a[:tc], entry_a[:tc], exit_a[:tc], out_a[:tc]


# ==============================================================================
# NUMBA TRADE SIMS — SHORT DIRECTION
# ==============================================================================
@jit(nopython=True)
def sim_short_static(entry_indices, prices, highs, lows, date_ints, time_mins,
                     tp, sl, exit_time):
    ne = len(entry_indices); np_ = len(prices)
    pnl_a = np.empty(ne, np.float64); date_a = np.empty(ne, np.int64)
    entry_a = np.empty(ne, np.float64); exit_a = np.empty(ne, np.float64)
    out_a = np.empty(ne, np.int64); tc = 0; cp = 0; ltd = -1
    while cp < ne:
        idx = entry_indices[cp]
        if idx >= np_ - 1: break
        ed = date_ints[idx]
        if ed == ltd: cp += 1; continue
        ep = prices[idx]; tgt = ep * (1 - tp); stp = ep * (1 + sl)
        oc = 0.0; xp = ep; xt = 0; done = False; xi = idx
        for k in range(idx + 1, np_):
            if date_ints[k] != ed:
                xp = prices[k-1]; oc = (ep - xp) / ep; xi = k; done = True; break
            tp_h = lows[k] <= tgt; sl_h = highs[k] >= stp
            if sl_h and tp_h:
                if np.random.random() < 0.5: oc = -sl; xp = stp; xt = -1
                else: oc = tp; xp = tgt; xt = 1
                xi = k; done = True; break
            if sl_h: oc = -sl; xp = stp; xt = -1; xi = k; done = True; break
            if tp_h: oc = tp; xp = tgt; xt = 1; xi = k; done = True; break
            if time_mins[k] >= exit_time:
                xp = prices[k]; oc = (ep - xp) / ep; xi = k; done = True; break
        if not done: xp = prices[np_-1]; oc = (ep - xp) / ep; xi = np_
        pnl_a[tc] = oc; date_a[tc] = ed; entry_a[tc] = ep; exit_a[tc] = xp; out_a[tc] = xt
        tc += 1; ltd = ed; cp += 1
    return pnl_a[:tc], date_a[:tc], entry_a[:tc], exit_a[:tc], out_a[:tc]

@jit(nopython=True)
def sim_short_grace(entry_indices, prices, highs, lows, date_ints, time_mins,
                    tp, sl, exit_time, grace_minutes):
    ne = len(entry_indices); np_ = len(prices)
    pnl_a = np.empty(ne, np.float64); date_a = np.empty(ne, np.int64)
    entry_a = np.empty(ne, np.float64); exit_a = np.empty(ne, np.float64)
    out_a = np.empty(ne, np.int64); tc = 0; cp = 0; ltd = -1
    while cp < ne:
        idx = entry_indices[cp]
        if idx >= np_ - 1: break
        ed = date_ints[idx]
        if ed == ltd: cp += 1; continue
        ep = prices[idx]; tgt = ep * (1 - tp); stp = ep * (1 + sl)
        et = time_mins[idx]; oc = 0.0; xp = ep; xt = 0; done = False
        for k in range(idx + 1, np_):
            if date_ints[k] != ed:
                xp = prices[k-1]; oc = (ep - xp) / ep; done = True; break
            elapsed = time_mins[k] - et; sl_act = elapsed >= grace_minutes
            tp_h = lows[k] <= tgt; sl_h = sl_act and (highs[k] >= stp)
            if sl_h and tp_h:
                if np.random.random() < 0.5: oc = -sl; xp = stp; xt = -1
                else: oc = tp; xp = tgt; xt = 1
                done = True; break
            if sl_h: oc = -sl; xp = stp; xt = -1; done = True; break
            if tp_h: oc = tp; xp = tgt; xt = 1; done = True; break
            if time_mins[k] >= exit_time:
                xp = prices[k]; oc = (ep - xp) / ep; done = True; break
        if not done: xp = prices[np_-1]; oc = (ep - xp) / ep
        pnl_a[tc] = oc; date_a[tc] = ed; entry_a[tc] = ep; exit_a[tc] = xp; out_a[tc] = xt
        tc += 1; ltd = ed; cp += 1
    return pnl_a[:tc], date_a[:tc], entry_a[:tc], exit_a[:tc], out_a[:tc]

@jit(nopython=True)
def sim_short_limit(entry_indices, prices, highs, lows, date_ints, time_mins,
                    tp, sl, exit_time, dip_pct, max_wait):
    ne = len(entry_indices); np_ = len(prices)
    pnl_a = np.empty(ne, np.float64); date_a = np.empty(ne, np.int64)
    entry_a = np.empty(ne, np.float64); exit_a = np.empty(ne, np.float64)
    out_a = np.empty(ne, np.int64); tc = 0; cp = 0; ltd = -1
    while cp < ne:
        idx = entry_indices[cp]
        if idx >= np_ - 1: break
        ed = date_ints[idx]
        if ed == ltd: cp += 1; continue
        sp = prices[idx]; lp = sp * (1.0 + dip_pct); st = time_mins[idx]
        fi = -1; fp = 0.0
        for k in range(idx + 1, np_):
            if date_ints[k] != ed: break
            if time_mins[k] - st > max_wait: break
            if highs[k] >= lp: fp = lp; fi = k; break
        if fi < 0: cp += 1; ltd = ed; continue
        tgt = fp * (1 - tp); stp = fp * (1 + sl)
        oc = 0.0; xp = fp; xt = 0; done = False
        for k in range(fi + 1, np_):
            if date_ints[k] != ed:
                xp = prices[k-1]; oc = (fp - xp) / fp; done = True; break
            tp_h = lows[k] <= tgt; sl_h = highs[k] >= stp
            if sl_h and tp_h:
                if np.random.random() < 0.5: oc = -sl; xp = stp; xt = -1
                else: oc = tp; xp = tgt; xt = 1
                done = True; break
            if sl_h: oc = -sl; xp = stp; xt = -1; done = True; break
            if tp_h: oc = tp; xp = tgt; xt = 1; done = True; break
            if time_mins[k] >= exit_time:
                xp = prices[k]; oc = (fp - xp) / fp; done = True; break
        if not done: xp = prices[np_-1]; oc = (fp - xp) / fp
        pnl_a[tc] = oc; date_a[tc] = ed; entry_a[tc] = fp; exit_a[tc] = xp; out_a[tc] = xt
        tc += 1; ltd = ed; cp += 1
    return pnl_a[:tc], date_a[:tc], entry_a[:tc], exit_a[:tc], out_a[:tc]


def _run_sim(indices, ohlcv, cfg, direction):
    p = ohlcv['prices']; h = ohlcv['highs']; l = ohlcv['lows']
    d = ohlcv['dates']; t = ohlcv['times'].astype(np.int64)
    m = cfg['mode']
    if direction == 'LONG':
        if m == 'grace':
            r = sim_long_grace(indices, p, h, l, d, t, cfg['tp_pct'], cfg['sl_pct'],
                               EXIT_TIME_MIN, cfg.get('grace_minutes', 0))
        elif m == 'limit_entry':
            r = sim_long_limit(indices, p, h, l, d, t, cfg['tp_pct'], cfg['sl_pct'],
                               EXIT_TIME_MIN, cfg.get('dip_pct', 0), cfg.get('max_wait', 30))
        else:
            r = sim_long_static(indices, p, h, l, d, t, cfg['tp_pct'], cfg['sl_pct'],
                                EXIT_TIME_MIN)
    else:
        if m == 'grace':
            r = sim_short_grace(indices, p, h, l, d, t, cfg['tp_pct'], cfg['sl_pct'],
                                EXIT_TIME_MIN, cfg.get('grace_minutes', 0))
        elif m == 'limit_entry':
            r = sim_short_limit(indices, p, h, l, d, t, cfg['tp_pct'], cfg['sl_pct'],
                                EXIT_TIME_MIN, cfg.get('dip_pct', 0), cfg.get('max_wait', 30))
        else:
            r = sim_short_static(indices, p, h, l, d, t, cfg['tp_pct'], cfg['sl_pct'],
                                 EXIT_TIME_MIN)
    pnl, dd, ep, xp, oc = r
    if len(pnl) > 0 and SLIPPAGE_PCT > 0:
        pnl = pnl - 2 * SLIPPAGE_PCT
    return pnl, dd, ep, xp, oc


# ==============================================================================
# CONSENSUS SCORING — generic for both directions
# ==============================================================================
def score_ticker_consensus(ticker, df_feat, regime_df, ds, family_fold_cfgs,
                           consensus_set, df_15m=None):
    bt_start = pd.Timestamp(BACKTEST_START, tz='UTC')
    bt_end = pd.Timestamp(BACKTEST_END, tz='UTC') + pd.Timedelta(days=1)

    score_bars = []
    for i in range(TIME_STEPS, len(df_feat)):
        ts = df_feat.index[i]
        if ts < bt_start or ts >= bt_end:
            continue
        bar_pst = ts.tz_convert(PST)
        if bar_pst.hour * 60 + bar_pst.minute != SCORE_MINUTE:
            continue
        if 'Volume' in df_feat.columns and float(df_feat['Volume'].iloc[i]) == 0:
            continue
        score_bars.append(i)

    if not score_bars:
        return {}

    scaled_arrays = {}
    lr_arrays = {}
    for (fam_name, fold_id), cfgs in family_fold_cfgs.items():
        if ticker not in cfgs:
            continue
        if fam_name not in scaled_arrays:
            stats = ds.family_meta[fam_name]['scaling_stats']
            df_s = apply_scaling(df_feat, stats, ALL_COLS)
            scaled_arrays[fam_name] = df_s[ALL_COLS].values.astype(np.float32)
        if ds.families[fam_name].get('arch', 'standard') == 'multires' and df_15m is not None:
            if fam_name not in lr_arrays:
                stats_15m = ds.family_meta[fam_name].get('scaling_stats_15m')
                if stats_15m:
                    df_15m_s = apply_scaling(df_15m, stats_15m, LOWRES_COLS)
                else:
                    df_15m_s = df_15m.copy()
                lr_arrays[fam_name] = df_15m_s[LOWRES_COLS].values.astype(np.float32)

    day_results = {}
    for row_idx in score_bars:
        bar_pst = df_feat.index[row_idx].tz_convert(PST)
        date_int = bar_pst.year * 10000 + bar_pst.month * 100 + bar_pst.day

        regime_at_entry = regime_df.iloc[regime_df.index.get_indexer(
            [df_feat.index[row_idx]], method='pad')[0]]

        all_scores = []
        for (fam_name, fold_id), cfgs in family_fold_cfgs.items():
            if ticker not in cfgs:
                continue
            cfg = cfgs[ticker]
            h_idx = cfg['horizon']

            regime_stats = ds.family_meta[fam_name]['regime_stats']
            regime_scaled = np.zeros(N_REGIME_F, dtype=np.float32)
            for ri, col in enumerate(REGIME_COLS):
                val = float(regime_at_entry[col]) if col in regime_at_entry.index else 0.0
                if col in regime_stats:
                    s = regime_stats[col]
                    val = np.clip((val - s['med']) / s['iqr'], -10, 10)
                regime_scaled[ri] = val

            XA = scaled_arrays[fam_name]
            seq_batch = XA[row_idx - TIME_STEPS:row_idx][np.newaxis]

            ds.ensure_fold_loaded(fam_name, fold_id)
            inputs = {'seq_input': seq_batch, 'regime_input': regime_scaled[np.newaxis]}

            is_mr = ds.families[fam_name].get('arch', 'standard') == 'multires'
            if is_mr and fam_name in lr_arrays:
                XL = lr_arrays[fam_name]
                ts_1m = df_feat.index[row_idx]
                lr_idx = df_15m.index.get_indexer([ts_1m], method='pad')[0]
                lr_end = lr_idx + 1
                lr_start = max(0, lr_end - TIME_STEPS_15M)
                lr_window = XL[lr_start:lr_end]
                if len(lr_window) < TIME_STEPS_15M:
                    pad = np.zeros((TIME_STEPS_15M - len(lr_window), N_LOWRES_F),
                                   dtype=np.float32)
                    lr_window = np.concatenate([pad, lr_window], axis=0)
                inputs['seq_15m_input'] = lr_window[np.newaxis]

            model = ds.get_model(fam_name)
            preds = model(inputs, training=False).numpy().flatten()
            prob = float(preds[h_idx])

            all_scores.append({
                'family': fam_name, 'fold': fold_id,
                'prob': prob, 'thresh': cfg['prob_thresh'],
                'fired': prob >= cfg['prob_thresh'],
            })

        consensus = [s for s in all_scores
                     if (s['family'], s['fold']) in consensus_set]
        fires = [s for s in consensus if s['fired']]

        if fires:
            day_results[date_int] = {
                'n_fires': len(fires),
                'n_available': len(consensus),
                'best_fire_prob': max(s['prob'] for s in fires),
                'sum_conviction': sum(s['prob'] / max(s['thresh'], 1e-6) for s in fires),
                'row_idx': row_idx,
            }

    return day_results


# ==============================================================================
# RUN ONE DIRECTION: score → select → simulate
# ==============================================================================
def run_direction(direction, ds, deploy_configs, family_fold_cfgs,
                  consensus_set, market_df, regime_df):
    tickers = sorted(deploy_configs.keys())
    print(f"\n  --- {direction}: Scoring {len(tickers)} tickers × "
          f"{len(consensus_set)} consensus models ---")

    ticker_day_fires = {}
    for ticker in tickers:
        df_feat = engineer_asset(ticker, market_df)
        if df_feat is None:
            print(f"    {ticker}: SKIP")
            continue
        if direction == 'SHORT' and 'Close' in df_feat.columns:
            last_close = df_feat['Close'].iloc[-1]
            if last_close < MIN_PRICE:
                print(f"    {ticker}: SKIP — price ${last_close:.2f} < ${MIN_PRICE:.2f}")
                continue
        has_mr = any(
            ticker in cfgs
            for (fn, fi), cfgs in family_fold_cfgs.items()
            if ds.families[fn].get('arch', 'standard') == 'multires'
        )
        df_15m = build_15min_features(df_feat) if has_mr else None
        day_fires = score_ticker_consensus(
            ticker, df_feat, regime_df, ds, family_fold_cfgs, consensus_set,
            df_15m)
        if day_fires:
            ticker_day_fires[ticker] = day_fires
            avg_f = np.mean([v['n_fires'] for v in day_fires.values()])
            print(f"    {ticker}: {len(day_fires)} days, avg {avg_f:.1f} fires")
        else:
            print(f"    {ticker}: 0 days")
        _master_cache.pop((ticker, TAIL_DAYS), None)

    all_fire_dates = sorted(set(d for fires in ticker_day_fires.values()
                                for d in fires))
    selected_signals = {}
    for date_int in all_fire_dates:
        candidates = []
        for ticker, day_fires in ticker_day_fires.items():
            if date_int in day_fires:
                info = day_fires[date_int]
                candidates.append((ticker, info['n_fires'], info['sum_conviction']))
        candidates.sort(key=lambda x: (-x[1], -x[2]))
        selected_signals[date_int] = candidates[:MAX_TRADES]

    total_sel = sum(len(v) for v in selected_signals.values())
    print(f"  {direction}: {total_sel} trades selected, {len(selected_signals)} days")

    selected_tickers = set()
    ticker_selected_dates = {}
    for date_int, picks in selected_signals.items():
        for ticker, n_fires, sum_conv in picks:
            selected_tickers.add(ticker)
            ticker_selected_dates.setdefault(ticker, set()).add(date_int)

    all_trades = []
    for ticker in sorted(selected_tickers):
        cfg = deploy_configs[ticker]
        df_feat = engineer_asset(ticker, market_df)
        if df_feat is None:
            continue
        bt_start = pd.Timestamp(BACKTEST_START, tz='UTC')
        bt_end = pd.Timestamp(BACKTEST_END, tz='UTC') + pd.Timedelta(days=1)
        df_bt = df_feat[(df_feat.index >= bt_start) & (df_feat.index < bt_end)].copy()
        df_bt_pst = df_bt.index.tz_convert(PST)

        ohlcv = {
            'prices': df_bt['Close'].values.copy(),
            'highs': df_bt['High'].values.copy(),
            'lows': df_bt['Low'].values.copy(),
            'dates': (df_bt_pst.year * 10000 + df_bt_pst.month * 100 +
                      df_bt_pst.day).values.astype(np.int64),
            'times': (df_bt_pst.hour * 60 + df_bt_pst.minute).values,
        }

        entry_indices = []
        sel_dates = ticker_selected_dates[ticker]
        for j in range(len(df_bt)):
            bt_pst = df_bt_pst[j]
            bt_date_int = bt_pst.year * 10000 + bt_pst.month * 100 + bt_pst.day
            bt_min = bt_pst.hour * 60 + bt_pst.minute
            if bt_date_int in sel_dates and bt_min == ENTRY_MINUTE:
                entry_indices.append(j)

        if not entry_indices:
            continue

        pnl, dates, ep, xp, outcomes = _run_sim(
            np.array(entry_indices, dtype=np.int64), ohlcv, cfg, direction)

        for i in range(len(pnl)):
            d = int(dates[i])
            date_str = f"{d//10000}-{(d%10000)//100:02d}-{d%100:02d}"
            fire_info = ticker_day_fires.get(ticker, {}).get(d, {})
            all_trades.append({
                'Ticker': ticker,
                'Date': date_str,
                'PnL': float(pnl[i]),
                'Entry': float(ep[i]),
                'Exit': float(xp[i]),
                'Outcome': 'TP' if outcomes[i]==1 else 'SL' if outcomes[i]==-1 else 'EOD',
                'Mode': cfg['mode'],
                'Family': f"{cfg['family']} F{cfg['fold']}",
                'N_Fires': fire_info.get('n_fires', 0),
                'Direction': direction,
            })
        n = len(pnl)
        avg = np.mean(pnl) if n > 0 else 0
        print(f"    {ticker} ({cfg['family']} F{cfg['fold']}, {cfg['mode']}): "
              f"{n}t avg={avg:+.2%}")
        _master_cache.pop((ticker, TAIL_DAYS), None)

    return all_trades


# ==============================================================================
# STATS HELPER
# ==============================================================================
def compute_daily_stats(trades_df, label=''):
    if len(trades_df) == 0:
        return [], {}
    dates = sorted(trades_df['Date'].unique())
    daily = []
    for date in dates:
        day = trades_df[trades_df['Date'] == date]
        ew = day['PnL'].mean()
        daily.append({'Date': date, 'EW': ew, 'N': len(day),
                      'Tickers': sorted(day['Ticker'].unique().tolist())})
    ew_vals = [d['EW'] for d in daily]
    cum = np.cumsum(ew_vals)
    avg = np.mean(ew_vals)
    std = np.std(ew_vals) if len(ew_vals) > 1 else 0
    sharpe = (avg / std * np.sqrt(252)) if std > 0 else 0
    peak = 0; max_dd = 0
    for c in cum:
        peak = max(peak, c)
        max_dd = max(max_dd, peak - c)
    stats = {
        'label': label,
        'cum_ew': cum[-1],
        'avg_daily': avg,
        'sharpe': sharpe,
        'total_trades': len(trades_df),
        'n_days': len(daily),
        'win_day_pct': sum(1 for v in ew_vals if v > 0) / len(daily),
        'max_drawdown': max_dd,
        'avg_pnl_trade': trades_df['PnL'].mean(),
        'win_rate': (trades_df['PnL'] > 0).mean(),
    }
    return daily, stats


# ==============================================================================
# MAIN
# ==============================================================================
print(f"\n{'='*W}")
print(f"  COMBINED ST LONG + SHORT — DEPLOYED STRATEGY BACKTEST")
print(f"  {BACKTEST_START} -> {BACKTEST_END}")
print(f"  Score: {SCORE_MINUTE//60}:{SCORE_MINUTE%60:02d} PT | "
      f"Entry: {ENTRY_MINUTE//60}:{ENTRY_MINUTE%60:02d} PT | "
      f"Exit: {EXIT_TIME_MIN//60}:{EXIT_TIME_MIN%60:02d} PT")
print(f"  Slippage: {2*SLIPPAGE_PCT:.2%} RT | MAX_TRADES: {MAX_TRADES} per direction")
print(f"  LONG: M_CONS_6NEWEST ({len(LONG_CONSENSUS)} consensus models)")
print(f"  SHORT: M_MRV2_NEWEST3 ({len(SHORT_CONSENSUS)} consensus models, MR-v2 only)")
print(f"{'='*W}")

# --- Load BQ configs ---
print(f"\n  Loading LONG configs from BQ...")
long_ffc = load_bq_configs(LONG_FAMILIES, LONG_F7_THRESH, 'LONG')
long_deploy = select_long_deploy(long_ffc)

print(f"\n  Loading SHORT configs from BQ...")
short_ffc = load_bq_configs(SHORT_FAMILIES, SHORT_F7_THRESH, 'SHORT')
short_deploy = select_short_deploy(short_ffc)

# --- Build models ---
print(f"\n  Building models...")
long_ds = DirState('long', LONG_FAMILIES)
short_ds = DirState('short', SHORT_FAMILIES)
build_models_for_dir(long_ds, long_deploy, long_ffc)
build_models_for_dir(short_ds, short_deploy, short_ffc)

# --- Market context + regime ---
print(f"\n  Building market context...")
market_df = build_market_context()

_bt_tickers = sorted(set(long_deploy.keys()) | set(short_deploy.keys()))
print(f"  Building universe returns for regime features ({len(_bt_tickers)} tickers)...")
_univ_rets = {}
for _t in _bt_tickers:
    _df = load_master(_t, tail_days=TAIL_DAYS)
    if _df is not None:
        _df = heal_data(_df.sort_index())
        _univ_rets[_t] = _df['Close'].pct_change(1)
        _master_cache.pop((_t, TAIL_DAYS), None)
_univ_ret_df = pd.DataFrame(_univ_rets).reindex(market_df.index).fillna(0)
print(f"    Universe returns: {_univ_ret_df.shape[1]} tickers loaded")
_master_cache.clear()

print(f"  Computing regime features...")
regime_df = compute_regime_features(market_df, _univ_ret_df)
del _univ_ret_df, _univ_rets

# --- Run LONG ---
long_trades = run_direction('LONG', long_ds, long_deploy, long_ffc,
                             LONG_CONSENSUS, market_df, regime_df)
_master_cache.clear()

# --- Run SHORT ---
short_trades = run_direction('SHORT', short_ds, short_deploy, short_ffc,
                              SHORT_CONSENSUS, market_df, regime_df)
_master_cache.clear()

# --- Combine ---
all_trades = long_trades + short_trades

if not all_trades:
    print(f"\n  NO TRADES in backtest window.")
else:
    df_trades = pd.DataFrame(all_trades)
    df_long = df_trades[df_trades['Direction'] == 'LONG']
    df_short = df_trades[df_trades['Direction'] == 'SHORT']

    long_daily, long_stats = compute_daily_stats(df_long, 'ST_LONG')
    short_daily, short_stats = compute_daily_stats(df_short, 'ST_SHORT')

    # Combined: equal-weight across ALL trades each day
    combined_daily, combined_stats = compute_daily_stats(df_trades, 'COMBINED')

    # ==========================================================================
    # RESULTS — PER-DIRECTION SUMMARY
    # ==========================================================================
    for label, stats, daily in [('ST LONG  (M_CONS_6NEWEST)', long_stats, long_daily),
                                 ('ST SHORT (M_MRV2_NEWEST3)', short_stats, short_daily),
                                 ('COMBINED', combined_stats, combined_daily)]:
        if not stats:
            continue
        print(f"\n\n{'='*W}")
        print(f"  {label} — {BACKTEST_START} -> {BACKTEST_END}")
        print(f"{'='*W}")
        print(f"  Trades:       {stats['total_trades']}")
        print(f"  Days:         {stats['n_days']}")
        print(f"  Avg PnL/tr:   {stats['avg_pnl_trade']:+.2%}")
        print(f"  Win rate:     {stats['win_rate']:.0%}")
        print(f"  EW Portfolio: {stats['cum_ew']:+.2%}")
        print(f"  Avg daily:    {stats['avg_daily']:+.2%}")
        print(f"  Sharpe:       {stats['sharpe']:.1f}")
        print(f"  Win day %:    {stats['win_day_pct']:.0%}")
        print(f"  Max DD:       {stats['max_drawdown']:.2%}")

    # ==========================================================================
    # DAILY PNL TABLE
    # ==========================================================================
    all_dates = sorted(df_trades['Date'].unique())

    print(f"\n\n{'='*W}")
    print(f"  DAILY PNL — LONG vs SHORT vs COMBINED")
    print(f"{'='*W}\n")

    long_daily_map = {d['Date']: d for d in long_daily}
    short_daily_map = {d['Date']: d for d in short_daily}
    combined_daily_map = {d['Date']: d for d in combined_daily}

    print(f"  {'Date':<12} {'L_N':>4} {'L_EW':>8} {'L_Cum':>8}  "
          f"{'S_N':>4} {'S_EW':>8} {'S_Cum':>8}  "
          f"{'C_N':>4} {'C_EW':>8} {'C_Cum':>8}")
    print(f"  {'─'*W}")

    l_cum = s_cum = c_cum = 0.0
    for date in all_dates:
        ld = long_daily_map.get(date)
        sd = short_daily_map.get(date)
        cd = combined_daily_map.get(date)
        l_ew = ld['EW'] if ld else 0; l_n = ld['N'] if ld else 0
        s_ew = sd['EW'] if sd else 0; s_n = sd['N'] if sd else 0
        c_ew = cd['EW'] if cd else 0; c_n = cd['N'] if cd else 0
        l_cum += l_ew; s_cum += s_ew; c_cum += c_ew
        print(f"  {date:<12} {l_n:>4} {l_ew:>+7.2%} {l_cum:>+7.2%}  "
              f"{s_n:>4} {s_ew:>+7.2%} {s_cum:>+7.2%}  "
              f"{c_n:>4} {c_ew:>+7.2%} {c_cum:>+7.2%}")

    # ==========================================================================
    # STRATEGY COMPARISON TABLE
    # ==========================================================================
    print(f"\n\n{'='*W}")
    print(f"  STRATEGY COMPARISON SUMMARY")
    print(f"{'='*W}\n")

    print(f"  {'Metric':<20} {'ST LONG':>12} {'ST SHORT':>12} {'COMBINED':>12}")
    print(f"  {'─'*60}")
    for metric, lv, sv, cv in [
        ('Trades', f"{long_stats.get('total_trades', 0)}",
         f"{short_stats.get('total_trades', 0)}",
         f"{combined_stats.get('total_trades', 0)}"),
        ('Avg PnL/trade',
         f"{long_stats.get('avg_pnl_trade', 0):+.2%}" if long_stats else 'N/A',
         f"{short_stats.get('avg_pnl_trade', 0):+.2%}" if short_stats else 'N/A',
         f"{combined_stats.get('avg_pnl_trade', 0):+.2%}" if combined_stats else 'N/A'),
        ('Win rate',
         f"{long_stats.get('win_rate', 0):.0%}" if long_stats else 'N/A',
         f"{short_stats.get('win_rate', 0):.0%}" if short_stats else 'N/A',
         f"{combined_stats.get('win_rate', 0):.0%}" if combined_stats else 'N/A'),
        ('Cum EW PnL',
         f"{long_stats.get('cum_ew', 0):+.2%}" if long_stats else 'N/A',
         f"{short_stats.get('cum_ew', 0):+.2%}" if short_stats else 'N/A',
         f"{combined_stats.get('cum_ew', 0):+.2%}" if combined_stats else 'N/A'),
        ('Sharpe',
         f"{long_stats.get('sharpe', 0):.1f}" if long_stats else 'N/A',
         f"{short_stats.get('sharpe', 0):.1f}" if short_stats else 'N/A',
         f"{combined_stats.get('sharpe', 0):.1f}" if combined_stats else 'N/A'),
        ('Max DD',
         f"{long_stats.get('max_drawdown', 0):.2%}" if long_stats else 'N/A',
         f"{short_stats.get('max_drawdown', 0):.2%}" if short_stats else 'N/A',
         f"{combined_stats.get('max_drawdown', 0):.2%}" if combined_stats else 'N/A'),
        ('Win day %',
         f"{long_stats.get('win_day_pct', 0):.0%}" if long_stats else 'N/A',
         f"{short_stats.get('win_day_pct', 0):.0%}" if short_stats else 'N/A',
         f"{combined_stats.get('win_day_pct', 0):.0%}" if combined_stats else 'N/A'),
    ]:
        print(f"  {metric:<20} {lv:>12} {sv:>12} {cv:>12}")

    # ==========================================================================
    # PER-TICKER SUMMARY
    # ==========================================================================
    print(f"\n\n{'='*W}")
    print(f"  PER-TICKER SUMMARY")
    print(f"{'='*W}\n")
    print(f"  {'Dir':<6} {'Ticker':<7} {'Source':<10} {'Mode':<8} "
          f"{'#Tr':>4} {'Avg':>7} {'WR':>5} {'TP':>3} {'SL':>3} {'EOD':>3}")
    print(f"  {'─'*70}")
    for d_label, sub_df, d_configs in [('LONG', df_long, long_deploy),
                                        ('SHORT', df_short, short_deploy)]:
        for ticker in sorted(sub_df['Ticker'].unique()):
            tk = sub_df[sub_df['Ticker'] == ticker]
            cfg = d_configs.get(ticker, {})
            src = f"{cfg.get('family', '?')} F{cfg.get('fold', '?')}"
            avg = tk['PnL'].mean()
            wr = (tk['PnL'] > 0).mean()
            n_tp = (tk['Outcome'] == 'TP').sum()
            n_sl = (tk['Outcome'] == 'SL').sum()
            n_eod = (tk['Outcome'] == 'EOD').sum()
            print(f"  {d_label:<6} {ticker:<7} {src:<10} {cfg.get('mode', '?'):<8} "
                  f"{len(tk):>4} {avg:>+6.2%} {wr:>4.0%} {n_tp:>3} {n_sl:>3} {n_eod:>3}")

    # ==========================================================================
    # CHARTS
    # ==========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))

    l_cum_ew = long_stats.get('cum_ew', 0) if long_stats else 0
    l_sh = long_stats.get('sharpe', 0) if long_stats else 0
    s_cum_ew = short_stats.get('cum_ew', 0) if short_stats else 0
    s_sh = short_stats.get('sharpe', 0) if short_stats else 0
    c_cum_ew = combined_stats.get('cum_ew', 0) if combined_stats else 0
    c_sh = combined_stats.get('sharpe', 0) if combined_stats else 0

    fig.suptitle(
        f'Combined ST LONG + SHORT — Deployed Strategy Backtest\n'
        f'{BACKTEST_START} -> {BACKTEST_END} | '
        f'LONG {l_cum_ew:+.2%} (Sh {l_sh:.1f}) | '
        f'SHORT {s_cum_ew:+.2%} (Sh {s_sh:.1f}) | '
        f'COMBINED {c_cum_ew:+.2%} (Sh {c_sh:.1f})',
        fontsize=13, fontweight='bold')

    COL_LONG = '#1565C0'
    COL_SHORT = '#D32F2F'
    COL_COMB = '#00897B'

    # [0,0] Equity curves — LONG, SHORT, COMBINED
    ax = axes[0, 0]
    for daily, color, label in [
        (long_daily, COL_LONG, f'LONG ({l_cum_ew:+.2%})'),
        (short_daily, COL_SHORT, f'SHORT ({s_cum_ew:+.2%})'),
        (combined_daily, COL_COMB, f'COMBINED ({c_cum_ew:+.2%})'),
    ]:
        if not daily:
            continue
        dates_dt = pd.to_datetime([d['Date'] for d in daily])
        cum = np.cumsum([d['EW'] for d in daily]) * 100
        lw = 2.5 if label.startswith('COMBINED') else 1.8
        ls = '--' if label.startswith('COMBINED') else '-'
        ax.plot(dates_dt, cum, color=color, linewidth=lw, linestyle=ls,
                marker='o', markersize=4, label=label)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.fill_between(
        pd.to_datetime([d['Date'] for d in combined_daily]),
        0, np.cumsum([d['EW'] for d in combined_daily]) * 100,
        alpha=0.08, color=COL_COMB) if combined_daily else None
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.set_ylabel('Cumulative EW PnL (%)')
    ax.set_title('Equity Curves — LONG vs SHORT vs Combined')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # [0,1] Daily PnL bars — stacked LONG + SHORT
    ax = axes[0, 1]
    dates_dt_all = pd.to_datetime(all_dates)
    long_ew = [long_daily_map.get(d, {}).get('EW', 0) * 100 for d in all_dates]
    short_ew = [short_daily_map.get(d, {}).get('EW', 0) * 100 for d in all_dates]
    bar_w = 0.35
    x_pos = np.arange(len(all_dates))
    ax.bar(x_pos - bar_w/2, long_ew, bar_w, color=COL_LONG, alpha=0.8, label='LONG')
    ax.bar(x_pos + bar_w/2, short_ew, bar_w, color=COL_SHORT, alpha=0.8, label='SHORT')
    combined_ew_arr = [combined_daily_map.get(d, {}).get('EW', 0) * 100 for d in all_dates]
    ax.plot(x_pos, combined_ew_arr, 'ko--', markersize=4, linewidth=1.5, label='Combined')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([d[5:] for d in all_dates], rotation=45, ha='right', fontsize=7)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylabel('Daily EW PnL (%)')
    ax.set_title('Daily PnL — LONG vs SHORT')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # [1,0] Per-ticker avg PnL (grouped by direction)
    ax = axes[1, 0]
    ticker_data = []
    for _, row in df_trades.groupby(['Ticker', 'Direction']).agg(
        avg_pnl=('PnL', 'mean'), count=('PnL', 'count')
    ).reset_index().iterrows():
        ticker_data.append(row)
    if ticker_data:
        td_df = pd.DataFrame(ticker_data).sort_values('avg_pnl')
        y_pos = range(len(td_df))
        bar_colors = [COL_LONG if d == 'LONG' else COL_SHORT
                      for d in td_df['Direction']]
        ax.barh(list(y_pos), td_df['avg_pnl'].values * 100, color=bar_colors,
                alpha=0.8)
        ax.set_yticks(list(y_pos))
        labels = [f"{r['Direction'][0]}: {r['Ticker']} ({int(r['count'])}t)"
                  for _, r in td_df.iterrows()]
        ax.set_yticklabels(labels, fontsize=7)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel('Avg PnL/trade (%)')
        ax.set_title('Per-Ticker Performance (L=LONG, S=SHORT)')
        ax.grid(axis='x', alpha=0.3)

    # [1,1] Daily detail table
    ax = axes[1, 1]
    ax.axis('off')
    table_lines = [f"{'Date':<12} {'L':>3} {'S':>3} {'Tot':>4} "
                   f"{'L_PnL':>7} {'S_PnL':>7} {'Comb':>7} {'Cum':>7}"]
    table_lines.append('─' * 60)
    cum_val = 0.0
    for date in all_dates:
        ld = long_daily_map.get(date)
        sd = short_daily_map.get(date)
        cd = combined_daily_map.get(date)
        l_n = ld['N'] if ld else 0; l_ew = ld['EW'] if ld else 0
        s_n = sd['N'] if sd else 0; s_ew = sd['EW'] if sd else 0
        c_ew = cd['EW'] if cd else 0; c_n = cd['N'] if cd else 0
        cum_val += c_ew
        table_lines.append(
            f"{date:<12} {l_n:>3} {s_n:>3} {c_n:>4} "
            f"{l_ew*100:>+6.2f}% {s_ew*100:>+6.2f}% "
            f"{c_ew*100:>+6.2f}% {cum_val*100:>+6.2f}%")
    table_text = '\n'.join(table_lines)
    ax.text(0.02, 0.98, table_text, transform=ax.transAxes,
            fontsize=7, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    ax.set_title('Daily Detail', fontsize=11)

    plt.tight_layout()
    chart_path = '/tmp/combined_st_deployed_backtest.png'
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    print(f"\n  Chart saved: {chart_path}")
    if _IN_COLAB:
        plt.show()
    plt.close()

    # === Second chart: focused on combined portfolio ===
    fig2, axes2 = plt.subplots(2, 2, figsize=(20, 14))
    fig2.suptitle(
        f'Combined Portfolio Deep Dive\n'
        f'{BACKTEST_START} -> {BACKTEST_END} | '
        f'{combined_stats["total_trades"]} trades | '
        f'EW {c_cum_ew:+.2%} | Sharpe {c_sh:.1f}',
        fontsize=13, fontweight='bold')

    # [0,0] Combined equity with daily annotations
    ax = axes2[0, 0]
    if combined_daily:
        dates_dt_c = pd.to_datetime([d['Date'] for d in combined_daily])
        cum_c = np.cumsum([d['EW'] for d in combined_daily]) * 100
        ax.plot(dates_dt_c, cum_c, 'o-', color=COL_COMB, linewidth=2.5, markersize=6)
        ax.fill_between(dates_dt_c, 0, cum_c, alpha=0.12, color=COL_COMB)
        for i, d in enumerate(combined_daily):
            ax.annotate(f"{d['EW']*100:+.1f}%",
                        (dates_dt_c[i], cum_c[i]),
                        textcoords="offset points", xytext=(0, 12),
                        fontsize=7, ha='center',
                        color='green' if d['EW'] > 0 else 'red')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.set_ylabel('Cumulative EW PnL (%)')
    ax.set_title('Combined Portfolio — Equity Curve')
    ax.grid(alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # [0,1] Daily PnL bars with trade count
    ax = axes2[0, 1]
    if combined_daily:
        ew_vals = [d['EW'] * 100 for d in combined_daily]
        bar_colors = ['#2E7D32' if v > 0 else '#C62828' for v in ew_vals]
        ax.bar(dates_dt_c, ew_vals, color=bar_colors, alpha=0.8, width=0.8)
        for i, d in enumerate(combined_daily):
            l_n = long_daily_map.get(d['Date'], {}).get('N', 0)
            s_n = short_daily_map.get(d['Date'], {}).get('N', 0)
            ax.text(dates_dt_c[i],
                    ew_vals[i] + (0.08 if ew_vals[i] >= 0 else -0.25),
                    f"{l_n}L+{s_n}S", ha='center', fontsize=6)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.set_ylabel('Daily EW PnL (%)')
    ax.set_title('Combined — Daily PnL (count = Long + Short)')
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # [1,0] Outcome distribution by direction
    ax = axes2[1, 0]
    outcomes = ['TP', 'SL', 'EOD']
    x_oc = np.arange(len(outcomes))
    bar_w_oc = 0.35
    for i, (d_label, d_df, color) in enumerate([
        ('LONG', df_long, COL_LONG), ('SHORT', df_short, COL_SHORT)
    ]):
        if len(d_df) == 0:
            continue
        counts = [(d_df['Outcome'] == o).sum() for o in outcomes]
        ax.bar(x_oc + i * bar_w_oc, counts, bar_w_oc, color=color,
               alpha=0.8, label=d_label)
        for j, c in enumerate(counts):
            if c > 0:
                avg_p = d_df[d_df['Outcome'] == outcomes[j]]['PnL'].mean()
                ax.text(x_oc[j] + i * bar_w_oc, c + 0.3,
                        f"{avg_p:+.1%}", ha='center', fontsize=7)
    ax.set_xticks(x_oc + bar_w_oc / 2)
    ax.set_xticklabels(outcomes)
    ax.set_ylabel('Count')
    ax.set_title('Outcomes by Direction (labels = avg PnL)')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # [1,1] Trades per day stacked
    ax = axes2[1, 1]
    long_counts = [long_daily_map.get(d, {}).get('N', 0) for d in all_dates]
    short_counts = [short_daily_map.get(d, {}).get('N', 0) for d in all_dates]
    ax.bar(x_pos, long_counts, 0.6, color=COL_LONG, alpha=0.8, label='LONG')
    ax.bar(x_pos, short_counts, 0.6, bottom=long_counts, color=COL_SHORT,
           alpha=0.8, label='SHORT')
    for i, (lc, sc) in enumerate(zip(long_counts, short_counts)):
        total = lc + sc
        if total > 0:
            ax.text(i, total + 0.1, str(total), ha='center', fontsize=8,
                    fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([d[5:] for d in all_dates], rotation=45, ha='right',
                        fontsize=7)
    ax.set_ylabel('Trades')
    ax.set_title('Trades Per Day (stacked)')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    chart2_path = '/tmp/combined_st_deployed_backtest_detail.png'
    plt.savefig(chart2_path, dpi=150, bbox_inches='tight')
    print(f"  Detail chart saved: {chart2_path}")
    if _IN_COLAB:
        plt.show()
    plt.close()


# ==============================================================================
# CLEANUP
# ==============================================================================
keras.backend.clear_session()
gc.collect()
try:
    ctypes.CDLL("libc.so.6").malloc_trim(0)
except Exception:
    pass

print(f"\n{'='*W}")
print(f"  COMBINED BACKTEST COMPLETE")
print(f"{'='*W}")
