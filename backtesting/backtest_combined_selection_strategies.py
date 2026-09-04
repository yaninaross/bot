# %% ── Combined LONG+SHORT — Selection Strategy Comparison ───────────────────
#
# LONG: M_CONS_6NEWEST (MR-v2 F9/F8/F7, L5-1 F7, ASL6 F7, L5-1 F6)
# SHORT: M_MRV2_NEWEST3 — 3 newest MR-v2 folds (F8/F9/F10), no L5-1/L5-2
#
# Scores ALL tickers with ALL models ONCE (GPU), pre-computes ALL trade outcomes,
# then tests ranking strategies as pure filtering/ranking operations.
#
# KEY EXPERIMENT: margin-based vs conviction-based vs deployed ranking
# Each model has its OWN per-ticker threshold from WF optimizer.
# Current deployed: rank by (-n_fires, -best_fire_prob) — raw prob is flawed.
# Better: use (prob - thresh) margin, avg conviction, or soft voting.
#
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
# CONFIG
# ==============================================================================
BUCKET_NAME      = "time-series-bucket-ys"
WF_BQ_TABLE      = "ys-time-series.scoring_logs.wf_optimization_results"
INFERENCE_DIR    = "Data/Inference_v2"
PST              = pytz.timezone('America/Los_Angeles')

BACKTEST_START   = '2026-08-24'
BACKTEST_END     = '2026-09-01'

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
EXCLUDE_TICKERS = {'SPY', 'VIX', 'TLT', 'IWM', 'XLE', 'ITA', 'BITF', 'MIGI',
                    'SMLR', 'BTCUSD', 'ETHUSD'}

CRYPTO_TICKERS = {'KEEL', 'CIFR', 'CLSK', 'HUT', 'MARA', 'RIOT', 'WULF',
                   'CORZ', 'IREN', 'BTBT', 'COIN'}

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

# --- Per-direction configs ---
LONG_FAMILIES = {
    'L5-1': {'gcs_data': 'Algos/RA_MoE/long_L5_1_data',
             'gcs_models': 'Algos/RA_MoE/long_L5_1_models', 'arch': 'standard',
             'bq_versions': {5: 'spike_transformer_long_L5_1_fold5',
                             6: 'spike_transformer_long_L5_1_fold6',
                             7: 'spike_transformer_long_L5_1_fold7'}},
    'ASL6': {'gcs_data': 'Algos/RA_MoE/long_L5_2_data',
             'gcs_models': 'Algos/RA_MoE/long_L5_2_asl6_models', 'arch': 'standard',
             'bq_versions': {5: 'spike_transformer_long_L5_2_asl6_fold5',
                             6: 'spike_transformer_long_L5_2_asl6_fold6',
                             7: 'spike_transformer_long_L5_2_asl6_fold7'}},
    'MR-v2': {'gcs_data': 'Algos/RA_MoE/long_L5_1_multires_v2_data',
              'gcs_models': 'Algos/RA_MoE/long_L5_1_multires_v2_models', 'arch': 'multires',
              'bq_versions': {6: 'spike_transformer_long_L5_1_multires_v2_fold6',
                              7: 'spike_transformer_long_L5_1_multires_v2_fold7',
                              8: 'spike_transformer_long_L5_1_multires_v2_fold8',
                              9: 'spike_transformer_long_L5_1_multires_v2_fold9'}},
}
LONG_CONSENSUS = {('MR-v2', 9), ('MR-v2', 8), ('MR-v2', 7),
                   ('L5-1', 7), ('ASL6', 7), ('L5-1', 6)}
LONG_NEWEST_FIRST = [('MR-v2', 9), ('MR-v2', 8), ('MR-v2', 7),
                     ('L5-1', 7), ('ASL6', 7),
                     ('L5-1', 6), ('ASL6', 6), ('MR-v2', 6), ('L5-1', 5), ('ASL6', 5)]
LONG_F7_THRESH = 0.005

SHORT_FAMILIES = {
    'MR-v2': {'gcs_data': 'Algos/RA_MoE/short_L5_2_multires_v2_data',
              'gcs_models': 'Algos/RA_MoE/short_L5_2_multires_v2_models', 'arch': 'multires',
              'bq_versions': {6: 'spike_transformer_short_L5_2_multires_v2_fold6',
                              7: 'spike_transformer_short_L5_2_multires_v2_fold7',
                              8: 'spike_transformer_short_L5_2_multires_v2_fold8',
                              9: 'spike_transformer_short_L5_2_multires_v2_fold9',
                              10: 'spike_transformer_short_L5_2_multires_v2_fold10'}},
}
SHORT_CONSENSUS = {('MR-v2', 8), ('MR-v2', 9), ('MR-v2', 10)}
SHORT_F7_THRESH = 0.030


# ==============================================================================
# DATA LOADING + FEATURES (identical to combined backtest)
# ==============================================================================
_master_cache = {}

def load_master(ticker, tail_days=None):
    cache_key = (ticker, tail_days)
    if cache_key in _master_cache:
        return _master_cache[cache_key]
    clean = ticker.replace("I:", "").replace("X:", "")
    file_ticker = TICKER_FILE_MAP.get(clean, clean)
    blob_path = f"{INFERENCE_DIR}/{file_ticker}_master_inference.parquet"
    local_path = f"/tmp/_bt_sel_{file_ticker}_master.parquet"
    try:
        gcs_client.bucket(BUCKET_NAME).blob(blob_path).download_to_filename(local_path)
        import pyarrow.parquet as pq
        schema = pq.read_schema(local_path)
        all_cols = [f.name for f in schema]
        use_cols = [c for c in all_cols if c in _NEEDED_COLS]
        if not use_cols: use_cols = None
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
        std_map = {'Total_vol': 'Total_Vol', 'Trade_count': 'Trade_Count',
                   'Volume_delta': 'Volume_Delta', 'Delta_pct': 'Delta_Pct',
                   'Obi': 'OBI', 'Bsize': 'bsize', 'Asize': 'asize',
                   'open': 'Open', 'high': 'High', 'low': 'Low',
                   'close': 'Close', 'volume': 'Volume'}
        lower_map = {k.lower(): v for k, v in std_map.items()}
        df.columns = [lower_map.get(str(c).strip().lower(), str(c).strip()) for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated(keep='first')].copy()
        for c in MICRO_COLS:
            if c not in df.columns: df[c] = 0.0
        df = df[df.index.notna()].sort_index()
        if tail_days is not None:
            cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=tail_days)
            df = df[df.index >= cutoff]
            if df.empty: return None
        _master_cache[cache_key] = df
        return df
    except Exception:
        if os.path.exists(local_path): os.remove(local_path)
        return None

def heal_data(df):
    df = df.resample('1min').asfreq()
    for c in ['Open', 'High', 'Low', 'Close']:
        if c in df.columns: df[c] = df[c].ffill()
    for c in ['Volume', 'Total_Vol', 'Volume_Delta', 'Trade_Count', 'Delta_Pct',
              'Spread', 'OBI', 'bsize', 'asize']:
        if c in df.columns: df[c] = df[c].fillna(0)
    return df

def build_market_context():
    frames = []; vix_close = None
    for ca in CONTEXT_ASSETS_LIST:
        df = load_master(ca, tail_days=TAIL_DAYS)
        if df is None: raise RuntimeError(f"Cannot load {ca}")
        df = heal_data(df.sort_index())
        if ca == 'VIX': vix_close = df['Close'].copy()
        cols = {f'{ca}_Anchor_ROC': df['Close'].pct_change(1)}
        for h in ROC_HORIZONS: cols[f'{ca}_ROC_{h}m'] = df['Close'].pct_change(h)
        frames.append(pd.DataFrame(cols, index=df.index))
    result = pd.concat(frames, axis=1).ffill().fillna(0)
    if vix_close is not None:
        result['VIX_Close'] = vix_close.reindex(result.index).ffill().fillna(0)
    for sp in SECTOR_PROXY_ASSETS:
        df = load_master(sp, tail_days=TAIL_DAYS)
        if df is not None:
            df = heal_data(df.sort_index())
            result[f'{sp}_ret'] = df['Close'].pct_change(1).reindex(result.index).ffill().fillna(0)
    return result

def engineer_asset(ticker, market_df):
    df = load_master(ticker, tail_days=TAIL_DAYS)
    if df is None: return None
    df = heal_data(df.sort_index())
    daily_c = df['Close'].resample('D').last().ffill().shift(1).reindex(df.index, method='ffill')
    df['Gap_Pct'] = (df['Open'] - daily_c) / daily_c
    df['RSI_14'] = ta.rsi(df['Close'], length=14)
    vol_real = df['Volume'].where(df['Volume'] > 0)
    df['RVOL'] = df['Volume'] / vol_real.rolling(50, min_periods=5).mean().ffill().clip(lower=1e-9)
    real_mask = df['Volume'] > 0
    atr_raw = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['ATR_14'] = atr_raw.where(real_mask).ffill().fillna(0)
    for h in ROC_HORIZONS: df[f'ROC_{h}m'] = df['Close'].pct_change(h)
    df = df.join(market_df, how='left').ffill().fillna(0)
    df_pst = df.index.tz_convert(PST)
    mins = df_pst.hour * 60 + df_pst.minute
    df['Time_Pos'] = np.clip((mins - 390) / 390, 0, 1)
    df['Bar_Is_Real'] = (df['Volume'] > 0).astype(float)
    return df.replace([np.inf, -np.inf], np.nan).fillna(0)

def build_15min_features(df_1m):
    ohlcv_15 = df_1m[['Open', 'High', 'Low', 'Close', 'Volume']].resample('15min').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).ffill()
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
    try: idx_pst = ohlcv_15.index.tz_convert(PST)
    except Exception: idx_pst = ohlcv_15.index
    mins = idx_pst.hour * 60 + idx_pst.minute
    lr['lr_Time_Pos'] = np.clip((mins - 390) / 390, 0, 1)
    return lr[LOWRES_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)

def compute_regime_features(market_df, universe_returns=None):
    rf = pd.DataFrame(index=market_df.index)
    spy_ret = market_df['SPY_Anchor_ROC'].fillna(0)
    for w, name in [(15, 'regime_rvol_15'), (60, 'regime_rvol_60'), (128, 'regime_rvol_128')]:
        rf[name] = spy_ret.rolling(w, min_periods=max(w//2, 1)).std().fillna(0)
    for w, name in [(60, 'regime_trend_60'), (128, 'regime_trend_128')]:
        cum_ret = spy_ret.rolling(w, min_periods=max(w//2, 1)).sum()
        rvol = rf[f'regime_rvol_{w}'].clip(lower=1e-8)
        rf[name] = (cum_ret / rvol).fillna(0).clip(-10, 10)
    spy_vals = spy_ret.values.astype(np.float64); n = len(spy_vals)
    autocorr = np.full(n, 0.0)
    for i in range(60, n):
        chunk = spy_vals[i-60:i]
        if np.count_nonzero(~np.isnan(chunk)) >= 30:
            x, y = chunk[:-1], chunk[1:]
            mx, my = x.mean(), y.mean(); sx, sy = x.std(), y.std()
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
        rf['regime_vix_level'] = 0.0; rf['regime_vix_velocity'] = 0.0
    try: idx_pst = market_df.index.tz_convert(PST)
    except Exception: idx_pst = market_df.index.tz_localize('UTC').tz_convert(PST)
    mins_of_day = idx_pst.hour * 60 + idx_pst.minute
    rf['regime_time_sin'] = np.sin(2 * np.pi * mins_of_day / 1440)
    rf['regime_time_cos'] = np.cos(2 * np.pi * mins_of_day / 1440)
    rf['regime_spy_ret_60'] = spy_ret.rolling(60, min_periods=30).sum().fillna(0)
    rf['regime_vix_close'] = (market_df['VIX_Close'].ffill().fillna(0)
                               if 'VIX_Close' in market_df.columns
                               else pd.Series(0.0, index=market_df.index))
    tlt_ret = (market_df['TLT_Anchor_ROC'].fillna(0) if 'TLT_Anchor_ROC' in market_df.columns
               else pd.Series(0.0, index=market_df.index))
    rf['regime_spy_tlt_corr'] = spy_ret.rolling(60, min_periods=30).corr(tlt_ret).fillna(0).clip(-1, 1)
    if universe_returns is not None and len(universe_returns.columns) > 1:
        cum_rets_60 = universe_returns.rolling(60, min_periods=30).sum()
        rf['regime_breadth'] = (cum_rets_60 > 0).mean(axis=1).fillna(0.5)
    else:
        rf['regime_breadth'] = 0.5
    rolling_mean = spy_ret.rolling(60, min_periods=30).mean()
    rolling_std = spy_ret.rolling(60, min_periods=30).std().clip(lower=1e-12)
    standardized = (spy_ret - rolling_mean) / rolling_std
    rf['regime_skew_60'] = standardized.pow(3).rolling(60, min_periods=30).mean().fillna(0).clip(-5, 5)
    crypto_ret = (market_df['MARA_ret'].fillna(0) if 'MARA_ret' in market_df.columns
                  else pd.Series(0.0, index=market_df.index))
    rf['regime_crypto_ret_60'] = crypto_ret.rolling(60, min_periods=30).sum().fillna(0)
    rf['regime_crypto_ret_128'] = crypto_ret.rolling(128, min_periods=60).sum().fillna(0)
    rf['regime_crypto_rvol'] = crypto_ret.rolling(60, min_periods=30).std().fillna(0)
    iwm_ret = (market_df['IWM_Anchor_ROC'].fillna(0) if 'IWM_Anchor_ROC' in market_df.columns
               else pd.Series(0.0, index=market_df.index))
    rf['regime_growth_vs_value'] = (iwm_ret.rolling(60, min_periods=30).sum() -
                                    spy_ret.rolling(60, min_periods=30).sum()).fillna(0)
    defense_ret = (market_df['ITA_ret'].fillna(0) if 'ITA_ret' in market_df.columns
                   else pd.Series(0.0, index=market_df.index))
    rf['regime_defense_rs'] = (defense_ret.rolling(60, min_periods=30).sum() -
                               spy_ret.rolling(60, min_periods=30).sum()).fillna(0)
    rf['regime_safe_haven'] = (tlt_ret.rolling(60, min_periods=30).sum() -
                               spy_ret.rolling(60, min_periods=30).sum()).fillna(0)
    return rf[REGIME_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)

def apply_scaling(df, stats, cols):
    df_s = df.copy()
    for col in cols:
        if col in ('Time_Pos', 'Bar_Is_Real'): continue
        if col in stats:
            df_s[col] = ((df_s[col] - stats[col]['med']) / stats[col]['iqr']).clip(-10, 10).astype('float32')
    return df_s


# ==============================================================================
# MODEL ARCHITECTURE (SpikeTransformer + MultiResSpikeTransformer)
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
        self.attn_drop = Dropout(dropout); self.tau_proj = Dense(n_heads)
        self.sigma_proj = Dense(d_model)
    def call(self, x, stats, training=False):
        B, T = tf.shape(x)[0], tf.shape(x)[1]; H, D = self.n_heads, self.d_head
        Q = tf.transpose(tf.reshape(self.wq(x), (B, T, H, D)), (0, 2, 1, 3))
        K = tf.transpose(tf.reshape(self.wk(x), (B, T, H, D)), (0, 2, 1, 3))
        V = tf.transpose(tf.reshape(self.wv(x), (B, T, H, D)), (0, 2, 1, 3))
        scores = tf.matmul(Q, K, transpose_b=True) / tf.cast(self.scale, Q.dtype)
        tau = tf.reshape(self.tau_proj(stats), (B, H, 1, 1)); scores = scores + tau
        attn = self.attn_drop(tf.nn.softmax(scores, axis=-1), training=training)
        out = tf.reshape(tf.transpose(tf.matmul(attn, V), (0, 2, 1, 3)), (B, T, self.d_model))
        sigma = tf.nn.softplus(self.sigma_proj(stats))
        return self.wo(out * sigma[:, tf.newaxis, :])

class TransformerBlock(_kl.Layer):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, **kw):
        super().__init__(**kw)
        self.attn = _DeStationaryAttention(d_model, n_heads, dropout, name='destat_attn')
        self.ffn = keras.Sequential([Dense(d_ff, activation='gelu'), Dense(d_model), Dropout(dropout)])
        self.norm1 = LayerNormalization(); self.norm2 = LayerNormalization()
        self.drop = Dropout(dropout)
    def call(self, x, stats, training=False):
        x = x + self.drop(self.attn(self.norm1(x), stats, training=training), training=training)
        return x + self.ffn(self.norm2(x), training=training)

class TransformerBackbone(_kl.Layer):
    def __init__(self, d_model, n_heads, d_ff, n_blocks, seq_len, dropout=0.1, **kw):
        super().__init__(**kw)
        self.stationarize = _SeriesStationarization(name='stationarize')
        self.input_proj = Dense(d_model)
        self.pos_embed = self.add_weight(name='pos_embed', shape=(1, seq_len, d_model),
                                         initializer='glorot_normal', trainable=True)
        self.input_drop = Dropout(dropout)
        self.blocks = [TransformerBlock(d_model, n_heads, d_ff, dropout, name=f'block_{i}')
                       for i in range(n_blocks)]
        self.final_norm = LayerNormalization()
    def call(self, x, training=False):
        x_norm, stats = self.stationarize(x)
        h = self.input_proj(x_norm) + self.pos_embed
        h = self.input_drop(h, training=training)
        for block in self.blocks: h = block(h, stats, training=training)
        return self.final_norm(h)

class _AttentionPooling(_kl.Layer):
    def __init__(self, d_model, **kw):
        super().__init__(**kw)
        self.query = self.add_weight(name='pool_query', shape=(1, 1, d_model),
                                     initializer='glorot_normal', trainable=True)
        self.scale = tf.sqrt(tf.cast(d_model, tf.float32))
    def call(self, x, training=False):
        scores = tf.matmul(self.query, x, transpose_b=True) / tf.cast(self.scale, x.dtype)
        return tf.matmul(tf.nn.softmax(scores, axis=-1), x)[:, 0, :]

class _FiLMConditioner(_kl.Layer):
    def __init__(self, d_model, hidden_dim=64, **kw):
        super().__init__(**kw)
        self.mlp = keras.Sequential([Dense(hidden_dim, activation='relu'),
                                     Dense(hidden_dim, activation='relu')])
        self.gamma_proj = Dense(d_model); self.beta_proj = Dense(d_model)
    def call(self, x, regime_features, training=False):
        h = self.mlp(regime_features, training=training)
        return (1.0 + self.gamma_proj(h)) * x + self.beta_proj(h)

class SpikeTransformer(keras.Model):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.backbone = TransformerBackbone(D_MODEL, N_HEADS, D_FF, N_BLOCKS, TIME_STEPS,
                                            DROPOUT_RATE, name='backbone')
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
        self.backbone_1m = TransformerBackbone(D_MODEL, N_HEADS, D_FF, N_BLOCKS, TIME_STEPS,
                                               DROPOUT_RATE, name='backbone_1m')
        self.pool_1m = _AttentionPooling(D_MODEL, name='pool_1m')
        self.backbone_15m = TransformerBackbone(D_MODEL, N_HEADS, D_FF, N_BLOCKS_15M, TIME_STEPS_15M,
                                                DROPOUT_RATE, name='backbone_15m')
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
# WEIGHT LOADING
# ==============================================================================
def _load_keras3_weights(model, filepath):
    with h5py.File(filepath, 'r') as f: _load_layer_recursive(model, f)

def _load_layer_recursive(layer, h5_group):
    if 'vars' in h5_group:
        vg = h5_group['vars']
        own_vars = layer.trainable_weights + layer.non_trainable_weights
        for i, var in enumerate(own_vars):
            key = str(i)
            if key in vg: var.assign(np.array(vg[key]))
    for key in sorted(h5_group.keys()):
        if key in ('vars', 'optimizer'): continue
        if not isinstance(h5_group[key], h5py.Group): continue
        child = getattr(layer, key, None)
        if child is None: continue
        if isinstance(child, _kl.Layer): _load_layer_recursive(child, h5_group[key])
        elif isinstance(child, (list, tuple)): _load_list_recursive(child, h5_group[key])

def _load_list_recursive(items, h5_group):
    used_names = {}
    for item in items:
        if not isinstance(item, _kl.Layer): continue
        cname = type(item).__name__
        snake = ''.join(f'_{c.lower()}' if c.isupper() else c for c in cname).lstrip('_')
        if snake in used_names: used_names[snake] += 1; snake = f"{snake}_{used_names[snake]}"
        else: used_names[snake] = 0
        if snake in h5_group: _load_layer_recursive(item, h5_group[snake])

def _ensure_file(gcs_path, local_path):
    if os.path.exists(local_path): return local_path
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    gcs_client.bucket(BUCKET_NAME).blob(gcs_path).download_to_filename(local_path)
    return local_path


# ==============================================================================
# DIRECTION STATE
# ==============================================================================
class DirState:
    def __init__(self, label, families):
        self.label = label; self.families = families
        self.model_std = None; self.model_mr = None
        self.fold_weights = {}; self.current_key_std = None; self.current_key_mr = None
        self.family_meta = {}
    def get_model(self, family):
        return self.model_mr if self.families[family].get('arch', 'standard') == 'multires' else self.model_std
    def ensure_fold_loaded(self, family, fold_id):
        key = (family, fold_id)
        arch = self.families[family].get('arch', 'standard')
        if arch == 'multires':
            if self.current_key_mr != key:
                _load_keras3_weights(self.model_mr, self.fold_weights[key]); self.current_key_mr = key
        else:
            if self.current_key_std != key:
                _load_keras3_weights(self.model_std, self.fold_weights[key]); self.current_key_std = key

def build_models_for_dir(ds, deploy_configs, family_fold_cfgs):
    all_cfgs = list(deploy_configs.values())
    if family_fold_cfgs:
        for cfgs in family_fold_cfgs.values():
            if isinstance(cfgs, dict): all_cfgs.extend(cfgs.values())
    needed_families = sorted(set(cfg['family'] for cfg in all_cfgs))
    for family_name in needed_families:
        fam = ds.families[family_name]
        local_meta = f'/tmp/ra_moe_{ds.label}_{family_name}_meta.npz'
        meta_path = _ensure_file(f"{fam['gcs_data']}/meta.npz", local_meta)
        meta = np.load(meta_path, allow_pickle=True)
        meta_dict = {'scaling_stats': meta['scaling_stats'].item(),
                     'regime_stats': meta['regime_stats'].item()}
        if 'scaling_stats_15m' in meta: meta_dict['scaling_stats_15m'] = meta['scaling_stats_15m'].item()
        ds.family_meta[family_name] = meta_dict
    need_std = any(ds.families[f].get('arch', 'standard') == 'standard' for f in needed_families)
    need_mr = any(ds.families[f].get('arch', 'standard') == 'multires' for f in needed_families)
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
        std_keys = [k for k in needed_keys if ds.families[k[0]].get('arch', 'standard') == 'standard']
        if std_keys: _load_keras3_weights(ds.model_std, ds.fold_weights[std_keys[0]]); ds.current_key_std = std_keys[0]
    if need_mr:
        mr_keys = [k for k in needed_keys if ds.families[k[0]].get('arch', 'standard') == 'multires']
        if mr_keys: _load_keras3_weights(ds.model_mr, ds.fold_weights[mr_keys[0]]); ds.current_key_mr = mr_keys[0]
    print(f"   {ds.label}: {len(needed_keys)} (family,fold) combos ready")


# ==============================================================================
# BQ CONFIG LOADING
# ==============================================================================
def load_bq_configs(families, f7_thresh, label):
    bq_client = bigquery.Client()
    family_fold_cfgs = {}
    for family_name, fam in families.items():
        for fold_id, model_version in fam['bq_versions'].items():
            q = f"""SELECT ticker, experiment, tpsl_mode, prob_thresh, tp_pct, sl_pct,
                   grace_minutes, dip_pct, max_wait_minutes, val_b_avg_pnl
            FROM `{WF_BQ_TABLE}`
            WHERE model_version = '{model_version}' AND survived_a = TRUE
              AND val_b_avg_pnl >= {BASE_PNL_THRESH}"""
            rows = bq_client.query(q).to_dataframe()
            if len(rows) == 0:
                family_fold_cfgs[(family_name, fold_id)] = {}
                continue
            best = rows.loc[rows.groupby('ticker')['val_b_avg_pnl'].idxmax()]
            cfgs = {}
            for _, row in best.iterrows():
                tk = row['ticker']
                if tk in EXCLUDE_TICKERS: continue
                exp = row['experiment']
                h_part = exp.replace(f'{model_version}_', '').split('_')[0]
                horizon_idx = HORIZON_FROM_LABEL.get(h_part, 1)
                mode = row['tpsl_mode']
                if pd.isna(mode) or mode == '':
                    if '_GRACE' in exp: mode = 'grace'
                    elif '_LIMIT' in exp: mode = 'limit_entry'
                    else: mode = 'static'
                cfgs[tk] = {
                    'family': family_name, 'fold': fold_id, 'horizon': horizon_idx,
                    'prob_thresh': float(row['prob_thresh']),
                    'tp_pct': float(row['tp_pct']), 'sl_pct': float(row['sl_pct']),
                    'mode': mode,
                    'grace_minutes': int(row['grace_minutes']) if pd.notna(row['grace_minutes']) else 0,
                    'dip_pct': float(row['dip_pct']) if pd.notna(row['dip_pct']) else 0,
                    'max_wait': int(row['max_wait_minutes']) if pd.notna(row['max_wait_minutes']) else 0,
                    'val_b_avg_pnl': float(row['val_b_avg_pnl']),
                }
            family_fold_cfgs[(family_name, fold_id)] = cfgs
            print(f"    {label} {family_name} F{fold_id}: {len(cfgs)} tickers")
    return family_fold_cfgs

def select_long_deploy(ffc):
    deploy = {}
    for family, fold in LONG_NEWEST_FIRST:
        for tk, cfg in ffc.get((family, fold), {}).items():
            if tk not in deploy: deploy[tk] = cfg
    print(f"    LONG deploy: {len(deploy)} tickers")
    return deploy

def select_short_deploy(ffc):
    all_tickers = set()
    for cfgs in ffc.values(): all_tickers |= set(cfgs.keys())
    deploy = {}
    for tk in sorted(all_tickers):
        best_val, best_cfg = -999, None
        for (fn, fi), cfgs in ffc.items():
            if tk not in cfgs: continue
            cfg = cfgs[tk]; v = cfg['val_b_avg_pnl']
            if fi >= 7 and v < SHORT_F7_THRESH: continue
            if v > best_val: best_val = v; best_cfg = cfg
        if best_cfg is not None: deploy[tk] = best_cfg
    print(f"    SHORT deploy: {len(deploy)} tickers")
    return deploy


# ==============================================================================
# NUMBA TRADE SIMS — LONG
# ==============================================================================
@jit(nopython=True)
def sim_long_static(ei, p, h, l, d, t, tp, sl, et):
    ne=len(ei);np_=len(p);pnl=np.empty(ne,np.float64);da=np.empty(ne,np.int64)
    ea=np.empty(ne,np.float64);xa=np.empty(ne,np.float64);oa=np.empty(ne,np.int64);tc=0;cp=0;ltd=-1
    while cp<ne:
        idx=ei[cp]
        if idx>=np_-1:break
        ed=d[idx]
        if ed==ltd:cp+=1;continue
        ep=p[idx];tgt=ep*(1+tp);stp=ep*(1-sl);oc=0.0;xp=ep;xt=0;done=False
        for k in range(idx+1,np_):
            if d[k]!=ed:xp=p[k-1];oc=(xp-ep)/ep;done=True;break
            if l[k]<=stp and h[k]>=tgt:
                if np.random.random()<0.5:oc=-sl;xp=stp;xt=-1
                else:oc=tp;xp=tgt;xt=1
                done=True;break
            if l[k]<=stp:oc=-sl;xp=stp;xt=-1;done=True;break
            if h[k]>=tgt:oc=tp;xp=tgt;xt=1;done=True;break
            if t[k]>=et:xp=p[k];oc=(xp-ep)/ep;done=True;break
        if not done:xp=p[np_-1];oc=(xp-ep)/ep
        pnl[tc]=oc;da[tc]=ed;ea[tc]=ep;xa[tc]=xp;oa[tc]=xt;tc+=1;ltd=ed;cp+=1
    return pnl[:tc],da[:tc],ea[:tc],xa[:tc],oa[:tc]

@jit(nopython=True)
def sim_long_grace(ei,p,h,l,d,t,tp,sl,et,gm):
    ne=len(ei);np_=len(p);pnl=np.empty(ne,np.float64);da=np.empty(ne,np.int64)
    ea=np.empty(ne,np.float64);xa=np.empty(ne,np.float64);oa=np.empty(ne,np.int64);tc=0;cp=0;ltd=-1
    while cp<ne:
        idx=ei[cp]
        if idx>=np_-1:break
        ed=d[idx]
        if ed==ltd:cp+=1;continue
        ep=p[idx];tgt=ep*(1+tp);stp=ep*(1-sl);ent=t[idx];oc=0.0;xp=ep;xt=0;done=False
        for k in range(idx+1,np_):
            if d[k]!=ed:xp=p[k-1];oc=(xp-ep)/ep;done=True;break
            elapsed=t[k]-ent;sl_act=elapsed>=gm;tp_h=h[k]>=tgt;sl_h=sl_act and l[k]<=stp
            if sl_h and tp_h:
                if np.random.random()<0.5:oc=-sl;xp=stp;xt=-1
                else:oc=tp;xp=tgt;xt=1
                done=True;break
            if sl_h:oc=-sl;xp=stp;xt=-1;done=True;break
            if tp_h:oc=tp;xp=tgt;xt=1;done=True;break
            if t[k]>=et:xp=p[k];oc=(xp-ep)/ep;done=True;break
        if not done:xp=p[np_-1];oc=(xp-ep)/ep
        pnl[tc]=oc;da[tc]=ed;ea[tc]=ep;xa[tc]=xp;oa[tc]=xt;tc+=1;ltd=ed;cp+=1
    return pnl[:tc],da[:tc],ea[:tc],xa[:tc],oa[:tc]

@jit(nopython=True)
def sim_long_limit(ei,p,h,l,d,t,tp,sl,et,dp,mw):
    ne=len(ei);np_=len(p);pnl=np.empty(ne,np.float64);da=np.empty(ne,np.int64)
    ea=np.empty(ne,np.float64);xa=np.empty(ne,np.float64);oa=np.empty(ne,np.int64);tc=0;cp=0;ltd=-1
    while cp<ne:
        idx=ei[cp]
        if idx>=np_-1:break
        ed=d[idx]
        if ed==ltd:cp+=1;continue
        sp=p[idx];lp=sp*(1.0-dp);st=t[idx];fi=-1;fp=0.0
        for k in range(idx+1,np_):
            if d[k]!=ed:break
            if t[k]-st>mw:break
            if l[k]<=lp:fp=lp;fi=k;break
        if fi<0:cp+=1;ltd=ed;continue
        tgt=fp*(1+tp);stp=fp*(1-sl);oc=0.0;xp=fp;xt=0;done=False
        for k in range(fi+1,np_):
            if d[k]!=ed:xp=p[k-1];oc=(xp-fp)/fp;done=True;break
            if l[k]<=stp and h[k]>=tgt:
                if np.random.random()<0.5:oc=-sl;xp=stp;xt=-1
                else:oc=tp;xp=tgt;xt=1
                done=True;break
            if l[k]<=stp:oc=-sl;xp=stp;xt=-1;done=True;break
            if h[k]>=tgt:oc=tp;xp=tgt;xt=1;done=True;break
            if t[k]>=et:xp=p[k];oc=(xp-fp)/fp;done=True;break
        if not done:xp=p[np_-1];oc=(xp-fp)/fp
        pnl[tc]=oc;da[tc]=ed;ea[tc]=fp;xa[tc]=xp;oa[tc]=xt;tc+=1;ltd=ed;cp+=1
    return pnl[:tc],da[:tc],ea[:tc],xa[:tc],oa[:tc]


# ==============================================================================
# NUMBA TRADE SIMS — SHORT
# ==============================================================================
@jit(nopython=True)
def sim_short_static(ei,p,h,l,d,t,tp,sl,et):
    ne=len(ei);np_=len(p);pnl=np.empty(ne,np.float64);da=np.empty(ne,np.int64)
    ea=np.empty(ne,np.float64);xa=np.empty(ne,np.float64);oa=np.empty(ne,np.int64);tc=0;cp=0;ltd=-1
    while cp<ne:
        idx=ei[cp]
        if idx>=np_-1:break
        ed=d[idx]
        if ed==ltd:cp+=1;continue
        ep=p[idx];tgt=ep*(1-tp);stp=ep*(1+sl);oc=0.0;xp=ep;xt=0;done=False
        for k in range(idx+1,np_):
            if d[k]!=ed:xp=p[k-1];oc=(ep-xp)/ep;done=True;break
            tp_h=l[k]<=tgt;sl_h=h[k]>=stp
            if sl_h and tp_h:
                if np.random.random()<0.5:oc=-sl;xp=stp;xt=-1
                else:oc=tp;xp=tgt;xt=1
                done=True;break
            if sl_h:oc=-sl;xp=stp;xt=-1;done=True;break
            if tp_h:oc=tp;xp=tgt;xt=1;done=True;break
            if t[k]>=et:xp=p[k];oc=(ep-xp)/ep;done=True;break
        if not done:xp=p[np_-1];oc=(ep-xp)/ep
        pnl[tc]=oc;da[tc]=ed;ea[tc]=ep;xa[tc]=xp;oa[tc]=xt;tc+=1;ltd=ed;cp+=1
    return pnl[:tc],da[:tc],ea[:tc],xa[:tc],oa[:tc]

@jit(nopython=True)
def sim_short_grace(ei,p,h,l,d,t,tp,sl,et,gm):
    ne=len(ei);np_=len(p);pnl=np.empty(ne,np.float64);da=np.empty(ne,np.int64)
    ea=np.empty(ne,np.float64);xa=np.empty(ne,np.float64);oa=np.empty(ne,np.int64);tc=0;cp=0;ltd=-1
    while cp<ne:
        idx=ei[cp]
        if idx>=np_-1:break
        ed=d[idx]
        if ed==ltd:cp+=1;continue
        ep=p[idx];tgt=ep*(1-tp);stp=ep*(1+sl);ent=t[idx];oc=0.0;xp=ep;xt=0;done=False
        for k in range(idx+1,np_):
            if d[k]!=ed:xp=p[k-1];oc=(ep-xp)/ep;done=True;break
            elapsed=t[k]-ent;sl_act=elapsed>=gm;tp_h=l[k]<=tgt;sl_h=sl_act and h[k]>=stp
            if sl_h and tp_h:
                if np.random.random()<0.5:oc=-sl;xp=stp;xt=-1
                else:oc=tp;xp=tgt;xt=1
                done=True;break
            if sl_h:oc=-sl;xp=stp;xt=-1;done=True;break
            if tp_h:oc=tp;xp=tgt;xt=1;done=True;break
            if t[k]>=et:xp=p[k];oc=(ep-xp)/ep;done=True;break
        if not done:xp=p[np_-1];oc=(ep-xp)/ep
        pnl[tc]=oc;da[tc]=ed;ea[tc]=ep;xa[tc]=xp;oa[tc]=xt;tc+=1;ltd=ed;cp+=1
    return pnl[:tc],da[:tc],ea[:tc],xa[:tc],oa[:tc]

@jit(nopython=True)
def sim_short_limit(ei,p,h,l,d,t,tp,sl,et,dp,mw):
    ne=len(ei);np_=len(p);pnl=np.empty(ne,np.float64);da=np.empty(ne,np.int64)
    ea=np.empty(ne,np.float64);xa=np.empty(ne,np.float64);oa=np.empty(ne,np.int64);tc=0;cp=0;ltd=-1
    while cp<ne:
        idx=ei[cp]
        if idx>=np_-1:break
        ed=d[idx]
        if ed==ltd:cp+=1;continue
        sp=p[idx];lp=sp*(1.0+dp);st=t[idx];fi=-1;fp=0.0
        for k in range(idx+1,np_):
            if d[k]!=ed:break
            if t[k]-st>mw:break
            if h[k]>=lp:fp=lp;fi=k;break
        if fi<0:cp+=1;ltd=ed;continue
        tgt=fp*(1-tp);stp=fp*(1+sl);oc=0.0;xp=fp;xt=0;done=False
        for k in range(fi+1,np_):
            if d[k]!=ed:xp=p[k-1];oc=(fp-xp)/fp;done=True;break
            tp_h=l[k]<=tgt;sl_h=h[k]>=stp
            if sl_h and tp_h:
                if np.random.random()<0.5:oc=-sl;xp=stp;xt=-1
                else:oc=tp;xp=tgt;xt=1
                done=True;break
            if sl_h:oc=-sl;xp=stp;xt=-1;done=True;break
            if tp_h:oc=tp;xp=tgt;xt=1;done=True;break
            if t[k]>=et:xp=p[k];oc=(fp-xp)/fp;done=True;break
        if not done:xp=p[np_-1];oc=(fp-xp)/fp
        pnl[tc]=oc;da[tc]=ed;ea[tc]=fp;xa[tc]=xp;oa[tc]=xt;tc+=1;ltd=ed;cp+=1
    return pnl[:tc],da[:tc],ea[:tc],xa[:tc],oa[:tc]


def _run_sim(indices, ohlcv, cfg, direction):
    p=ohlcv['prices'];h=ohlcv['highs'];l=ohlcv['lows'];d=ohlcv['dates'];t=ohlcv['times'].astype(np.int64)
    m=cfg['mode']
    if direction == 'LONG':
        if m=='grace': r=sim_long_grace(indices,p,h,l,d,t,cfg['tp_pct'],cfg['sl_pct'],EXIT_TIME_MIN,cfg.get('grace_minutes',0))
        elif m=='limit_entry': r=sim_long_limit(indices,p,h,l,d,t,cfg['tp_pct'],cfg['sl_pct'],EXIT_TIME_MIN,cfg.get('dip_pct',0),cfg.get('max_wait',30))
        else: r=sim_long_static(indices,p,h,l,d,t,cfg['tp_pct'],cfg['sl_pct'],EXIT_TIME_MIN)
    else:
        if m=='grace': r=sim_short_grace(indices,p,h,l,d,t,cfg['tp_pct'],cfg['sl_pct'],EXIT_TIME_MIN,cfg.get('grace_minutes',0))
        elif m=='limit_entry': r=sim_short_limit(indices,p,h,l,d,t,cfg['tp_pct'],cfg['sl_pct'],EXIT_TIME_MIN,cfg.get('dip_pct',0),cfg.get('max_wait',30))
        else: r=sim_short_static(indices,p,h,l,d,t,cfg['tp_pct'],cfg['sl_pct'],EXIT_TIME_MIN)
    pnl,dd,ep,xp,oc=r
    if len(pnl)>0 and SLIPPAGE_PCT>0: pnl=pnl-2*SLIPPAGE_PCT
    return pnl,dd,ep,xp,oc


# ==============================================================================
# PHASE 1: Score + pre-compute trade outcomes
# ==============================================================================
def score_all_tickers(ds, deploy_configs, family_fold_cfgs, consensus_set,
                      market_df, regime_df, direction):
    tickers = sorted(deploy_configs.keys())
    bt_start = pd.Timestamp(BACKTEST_START, tz='UTC')
    bt_end = pd.Timestamp(BACKTEST_END, tz='UTC') + pd.Timedelta(days=1)

    raw_scores = {}
    trade_pool = {}
    ticker_context = {}

    print(f"\n  Phase 1 [{direction}]: Scoring {len(tickers)} tickers...")
    for ticker in tickers:
        df_feat = engineer_asset(ticker, market_df)
        if df_feat is None: continue

        has_mr = any(ticker in cfgs for (fn, fi), cfgs in family_fold_cfgs.items()
                     if ds.families[fn].get('arch', 'standard') == 'multires')
        df_15m = build_15min_features(df_feat) if has_mr else None

        scaled_arrays = {}; lr_arrays = {}
        for (fam_name, fold_id), cfgs in family_fold_cfgs.items():
            if ticker not in cfgs: continue
            if fam_name not in scaled_arrays:
                stats = ds.family_meta[fam_name]['scaling_stats']
                df_s = apply_scaling(df_feat, stats, ALL_COLS)
                scaled_arrays[fam_name] = df_s[ALL_COLS].values.astype(np.float32)
            if ds.families[fam_name].get('arch', 'standard') == 'multires' and df_15m is not None:
                if fam_name not in lr_arrays:
                    stats_15m = ds.family_meta[fam_name].get('scaling_stats_15m')
                    df_15m_s = apply_scaling(df_15m, stats_15m, LOWRES_COLS) if stats_15m else df_15m.copy()
                    lr_arrays[fam_name] = df_15m_s[LOWRES_COLS].values.astype(np.float32)

        for i in range(TIME_STEPS, len(df_feat)):
            ts = df_feat.index[i]
            if ts < bt_start or ts >= bt_end: continue
            bar_pst = ts.tz_convert(PST)
            if bar_pst.hour * 60 + bar_pst.minute != SCORE_MINUTE: continue
            if 'Volume' in df_feat.columns and float(df_feat['Volume'].iloc[i]) == 0: continue

            date_int = bar_pst.year * 10000 + bar_pst.month * 100 + bar_pst.day
            regime_at_entry = regime_df.iloc[regime_df.index.get_indexer([ts], method='pad')[0]]

            all_model_scores = {}
            for (fam_name, fold_id), cfgs in family_fold_cfgs.items():
                if ticker not in cfgs: continue
                cfg = cfgs[ticker]; h_idx = cfg['horizon']
                regime_stats = ds.family_meta[fam_name]['regime_stats']
                regime_scaled = np.zeros(N_REGIME_F, dtype=np.float32)
                for ri, col in enumerate(REGIME_COLS):
                    val = float(regime_at_entry[col]) if col in regime_at_entry.index else 0.0
                    if col in regime_stats:
                        s = regime_stats[col]; val = np.clip((val - s['med']) / s['iqr'], -10, 10)
                    regime_scaled[ri] = val

                XA = scaled_arrays[fam_name]
                seq_batch = XA[i - TIME_STEPS:i][np.newaxis]
                ds.ensure_fold_loaded(fam_name, fold_id)
                inputs = {'seq_input': seq_batch, 'regime_input': regime_scaled[np.newaxis]}
                is_mr = ds.families[fam_name].get('arch', 'standard') == 'multires'
                if is_mr and fam_name in lr_arrays:
                    XL = lr_arrays[fam_name]
                    lr_idx = df_15m.index.get_indexer([ts], method='pad')[0]
                    lr_end = lr_idx + 1; lr_start = max(0, lr_end - TIME_STEPS_15M)
                    lr_window = XL[lr_start:lr_end]
                    if len(lr_window) < TIME_STEPS_15M:
                        pad = np.zeros((TIME_STEPS_15M - len(lr_window), N_LOWRES_F), dtype=np.float32)
                        lr_window = np.concatenate([pad, lr_window], axis=0)
                    inputs['seq_15m_input'] = lr_window[np.newaxis]

                model = ds.get_model(fam_name)
                preds = model(inputs, training=False).numpy().flatten()
                prob = float(preds[h_idx])
                all_model_scores[(fam_name, fold_id)] = {
                    'prob': prob, 'thresh': cfg['prob_thresh'],
                    'fired': prob >= cfg['prob_thresh'],
                }

            if all_model_scores:
                raw_scores.setdefault(ticker, {})[date_int] = all_model_scores

            daily_close = df_feat['Close'].resample('D').last()
            close_3d_ago = daily_close.shift(3)
            try:
                current_date = bar_pst.normalize().tz_localize(None)
                close_today_prev = daily_close.get(current_date, None)
                close_3d = close_3d_ago.get(current_date, None)
                if close_today_prev is not None and close_3d is not None and close_3d > 0:
                    mom_3d = (close_today_prev - close_3d) / close_3d
                else:
                    mom_3d = 0.0
            except Exception:
                mom_3d = 0.0

            gap = float(df_feat['Gap_Pct'].iloc[i]) if 'Gap_Pct' in df_feat.columns else 0.0
            ticker_context.setdefault(ticker, {})[date_int] = {
                'gap': gap, 'momentum_3d': mom_3d,
                'is_crypto': ticker in CRYPTO_TICKERS,
            }

        cfg = deploy_configs[ticker]
        df_bt = df_feat[(df_feat.index >= bt_start) & (df_feat.index < bt_end)].copy()
        if len(df_bt) > 0:
            df_bt_pst = df_bt.index.tz_convert(PST)
            ohlcv = {
                'prices': df_bt['Close'].values.copy(),
                'highs': df_bt['High'].values.copy(),
                'lows': df_bt['Low'].values.copy(),
                'dates': (df_bt_pst.year*10000+df_bt_pst.month*100+df_bt_pst.day).values.astype(np.int64),
                'times': (df_bt_pst.hour*60+df_bt_pst.minute).values,
            }
            all_entry_indices = []
            all_entry_dates = []
            for j in range(len(df_bt)):
                bpst = df_bt_pst[j]
                bdi = bpst.year*10000+bpst.month*100+bpst.day
                bmin = bpst.hour*60+bpst.minute
                if bmin == ENTRY_MINUTE:
                    all_entry_indices.append(j)
                    all_entry_dates.append(bdi)
            if all_entry_indices:
                pnl, dates, ep, xp, outcomes = _run_sim(
                    np.array(all_entry_indices, dtype=np.int64), ohlcv, cfg, direction)
                for idx_t in range(len(pnl)):
                    d_int = int(dates[idx_t])
                    trade_pool.setdefault(ticker, {})[d_int] = {
                        'pnl': float(pnl[idx_t]),
                        'entry': float(ep[idx_t]),
                        'exit': float(xp[idx_t]),
                        'outcome': 'TP' if outcomes[idx_t]==1 else 'SL' if outcomes[idx_t]==-1 else 'EOD',
                    }

        n_days = len(raw_scores.get(ticker, {}))
        if n_days > 0:
            print(f"    {ticker}: {n_days} scored days")
        _master_cache.pop((ticker, TAIL_DAYS), None)

    _master_cache.clear()
    return raw_scores, trade_pool, ticker_context


# ==============================================================================
# SELECTION STRATEGIES — Ranking experiments
# ==============================================================================

# --- GROUP 1: Baselines (current deployed + simple variants) ---

def select_deployed(candidates, context, direction, history):
    """Current deployed: rank by (n_fires DESC, raw best_prob DESC)."""
    candidates.sort(key=lambda x: (-x['n_fires'], -x['best_prob']))
    return candidates[:MAX_TRADES]

def select_fires_only(candidates, context, direction, history):
    """n_fires only — no tiebreaker (stable sort preserves ticker order)."""
    candidates.sort(key=lambda x: -x['n_fires'])
    return candidates[:MAX_TRADES]


# --- GROUP 2: Margin-based (prob - thresh) — fixes the raw prob flaw ---

def select_max_margin(candidates, context, direction, history):
    """n_fires first, tiebreak by max(prob - thresh) among fired models."""
    candidates.sort(key=lambda x: (-x['n_fires'], -x['max_margin']))
    return candidates[:MAX_TRADES]

def select_avg_margin_fired(candidates, context, direction, history):
    """n_fires first, tiebreak by avg(prob - thresh) among fired models."""
    candidates.sort(key=lambda x: (-x['n_fires'], -x['avg_margin_fired']))
    return candidates[:MAX_TRADES]

def select_sum_margin_all(candidates, context, direction, history):
    """Drop n_fires entirely. Rank by sum(prob - thresh) across ALL consensus models.
    Even non-fired models contribute negative signal."""
    candidates.sort(key=lambda x: -x['sum_margin_all'])
    return candidates[:MAX_TRADES]

def select_sum_excess(candidates, context, direction, history):
    """Drop n_fires. Rank by sum(max(0, prob-thresh)) across ALL models.
    Only positive margins contribute — measures total excess conviction."""
    candidates.sort(key=lambda x: -x['sum_excess'])
    return candidates[:MAX_TRADES]


# --- GROUP 3: Conviction / ratio-based (normalize by threshold magnitude) ---

def select_avg_ratio(candidates, context, direction, history):
    """Drop n_fires. Rank by avg(prob/thresh) across ALL consensus models.
    Normalizes for threshold scale — 0.45/0.30 = 1.50 is stronger than 0.72/0.70 = 1.03."""
    candidates.sort(key=lambda x: -x['avg_ratio'])
    return candidates[:MAX_TRADES]

def select_sum_conviction(candidates, context, direction, history):
    """Drop n_fires. Sum of (prob/thresh) for fired models only.
    Rewards both breadth (n fires) AND depth (how far above thresh)."""
    candidates.sort(key=lambda x: -x['sum_conviction_fired'])
    return candidates[:MAX_TRADES]

def select_weighted_fire(candidates, context, direction, history):
    """n_fires first, tiebreak by sum(prob/thresh) for fired models.
    Like DEPLOYED but tiebreaker accounts for threshold calibration."""
    candidates.sort(key=lambda x: (-x['n_fires'], -x['sum_conviction_fired']))
    return candidates[:MAX_TRADES]


# --- GROUP 4: Soft voting (no binary fire gate) ---

def select_soft_vote(candidates, context, direction, history):
    """No fire requirement at all. Score = sum(max(0, (prob-thresh)/thresh)) across ALL models.
    A model at 0.59 with thresh 0.60 contributes ~0, at 0.80/0.50 contributes 0.60.
    Allows tickers where no model 'fires' but all are close to fire."""
    candidates.sort(key=lambda x: -x['soft_vote'])
    return candidates[:MAX_TRADES]

def select_soft_vote_gated(candidates, context, direction, history):
    """Soft vote, but require at least 1 fire as minimum bar."""
    candidates.sort(key=lambda x: -x['soft_vote'])
    return candidates[:MAX_TRADES]


# --- GROUP 5: Hybrid (combine n_fires with margin) ---

def select_fire_pct_x_margin(candidates, context, direction, history):
    """Single combined score: (n_fires / n_available) × avg_margin_of_fires.
    Naturally balances breadth and depth without arbitrary primary/secondary sort."""
    for c in candidates:
        fire_pct = c['n_fires'] / max(c['n_available'], 1)
        c['combined'] = fire_pct * c['avg_margin_fired']
    candidates.sort(key=lambda x: -x['combined'])
    return candidates[:MAX_TRADES]

def select_fire_x_conviction(candidates, context, direction, history):
    """n_fires × avg(prob/thresh) for fired models.
    More fires + more conviction = higher score."""
    for c in candidates:
        c['combined'] = c['n_fires'] * c['avg_conviction_fired']
    candidates.sort(key=lambda x: -x['combined'])
    return candidates[:MAX_TRADES]


# --- GROUP 6: Filters (kept from original) ---

def select_gap_aligned(candidates, context, direction, history):
    aligned = []
    for c in candidates:
        ctx = context.get(c['ticker'], {})
        gap = ctx.get('gap', 0)
        if direction == 'LONG' and gap >= 0: aligned.append(c)
        elif direction == 'SHORT' and gap <= 0: aligned.append(c)
    if not aligned: aligned = candidates
    aligned.sort(key=lambda x: -x['sum_margin_all'])
    return aligned[:MAX_TRADES]

def select_sector_cap(candidates, context, direction, history):
    candidates.sort(key=lambda x: -x['sum_margin_all'])
    selected = []; n_crypto = 0
    for c in candidates:
        if len(selected) >= MAX_TRADES: break
        ctx = context.get(c['ticker'], {})
        if ctx.get('is_crypto', False):
            if n_crypto >= 1: continue
            n_crypto += 1
        selected.append(c)
    return selected

def select_best_combo(candidates, context, direction, history):
    """Best of margin + gap + sector cap: sum_margin ranking, gap-aligned,
    max 1 crypto."""
    aligned = []
    for c in candidates:
        ctx = context.get(c['ticker'], {})
        gap = ctx.get('gap', 0)
        if direction == 'LONG' and gap >= 0: aligned.append(c)
        elif direction == 'SHORT' and gap <= 0: aligned.append(c)
    if not aligned: aligned = candidates
    aligned.sort(key=lambda x: -x['sum_margin_all'])
    selected = []; n_crypto = 0
    for c in aligned:
        if len(selected) >= MAX_TRADES: break
        ctx = context.get(c['ticker'], {})
        if ctx.get('is_crypto', False):
            if n_crypto >= 1: continue
            n_crypto += 1
        selected.append(c)
    return selected


STRATEGIES = {
    # Baselines
    'DEPLOYED':       {'fn': select_deployed,         'desc': 'Current: n_fires, best_prob (RAW)'},
    'FIRES_ONLY':     {'fn': select_fires_only,       'desc': 'n_fires only, no tiebreaker'},
    # Margin-based (prob - thresh)
    'MAX_MARGIN':     {'fn': select_max_margin,       'desc': 'n_fires + max(prob-thresh) tiebreak'},
    'AVG_MARGIN_F':   {'fn': select_avg_margin_fired, 'desc': 'n_fires + avg margin of fired'},
    'SUM_MARGIN':     {'fn': select_sum_margin_all,   'desc': 'sum(prob-thresh) ALL models'},
    'SUM_EXCESS':     {'fn': select_sum_excess,       'desc': 'sum(max(0,prob-thresh)) ALL models'},
    # Conviction / ratio-based
    'AVG_RATIO':      {'fn': select_avg_ratio,        'desc': 'avg(prob/thresh) ALL models'},
    'SUM_CONVICT':    {'fn': select_sum_conviction,   'desc': 'sum(prob/thresh) fired only'},
    'WFIRE_CONVICT':  {'fn': select_weighted_fire,    'desc': 'n_fires + conviction tiebreak'},
    # Soft voting
    'SOFT_VOTE':      {'fn': select_soft_vote,        'desc': 'sum(max(0,(p-t)/t)) no fire gate'},
    'SOFT_VOTE_G':    {'fn': select_soft_vote_gated,  'desc': 'Soft vote, require ≥1 fire'},
    # Hybrid
    'FIRE%×MARGIN':   {'fn': select_fire_pct_x_margin,'desc': 'fire_rate × avg_margin'},
    'FIRE×CONVICT':   {'fn': select_fire_x_conviction,'desc': 'n_fires × avg(prob/thresh)'},
    # Filters
    'GAP_ALIGNED':    {'fn': select_gap_aligned,      'desc': 'Gap filter + sum_margin rank'},
    'SECTOR_CAP':     {'fn': select_sector_cap,       'desc': 'sum_margin + max 1 crypto'},
    'BEST_COMBO':     {'fn': select_best_combo,       'desc': 'sum_margin + gap + sector cap'},
}


# ==============================================================================
# PHASE 2+3: Evaluate strategies
# ==============================================================================
def evaluate_strategies(raw_scores, trade_pool, ticker_context, consensus_set, direction):
    all_dates = sorted(set(d for tk_days in raw_scores.values() for d in tk_days))
    results = {}

    for strat_name, strat in STRATEGIES.items():
        fn = strat['fn']
        trades = []
        history = defaultdict(list)
        needs_no_fire = strat_name in ('SOFT_VOTE',)

        for date_int in all_dates:
            candidates = []
            for ticker, tk_days in raw_scores.items():
                if date_int not in tk_days: continue
                scores = tk_days[date_int]
                consensus_scores = {k: v for k, v in scores.items() if k in consensus_set}
                if not consensus_scores: continue
                fires = [v for v in consensus_scores.values() if v['fired']]
                all_probs = [v['prob'] for v in consensus_scores.values()]
                all_thresh = [v['thresh'] for v in consensus_scores.values()]
                all_margins = [v['prob'] - v['thresh'] for v in consensus_scores.values()]
                fire_margins = [v['prob'] - v['thresh'] for v in fires] if fires else [0]
                fire_ratios = [v['prob'] / max(v['thresh'], 1e-6) for v in fires] if fires else [0]
                all_ratios = [v['prob'] / max(v['thresh'], 1e-6) for v in consensus_scores.values()]

                if not fires and not needs_no_fire: continue

                candidates.append({
                    'ticker': ticker,
                    'n_fires': len(fires),
                    'n_available': len(consensus_scores),
                    'best_prob': max(s['prob'] for s in fires) if fires else max(all_probs),
                    'avg_prob': np.mean(all_probs),
                    'avg_thresh': np.mean(all_thresh),
                    'agree_pct': len(fires) / len(consensus_scores) if consensus_scores else 0,
                    'max_margin': max(fire_margins),
                    'avg_margin_fired': float(np.mean(fire_margins)),
                    'sum_margin_all': sum(all_margins),
                    'sum_excess': sum(max(0, m) for m in all_margins),
                    'avg_ratio': float(np.mean(all_ratios)),
                    'sum_conviction_fired': sum(fire_ratios),
                    'avg_conviction_fired': float(np.mean(fire_ratios)),
                    'soft_vote': sum(max(0, (v['prob'] - v['thresh']) / max(v['thresh'], 1e-6))
                                     for v in consensus_scores.values()),
                })

            ctx_today = {tk: ticker_context.get(tk, {}).get(date_int, {})
                         for tk in [c['ticker'] for c in candidates]}
            selected = fn(candidates, ctx_today, direction, history)

            for sel in selected:
                tk = sel['ticker']
                if tk in trade_pool and date_int in trade_pool[tk]:
                    t = trade_pool[tk][date_int]
                    d = date_int
                    date_str = f"{d//10000}-{(d%10000)//100:02d}-{d%100:02d}"
                    trades.append({
                        'Ticker': tk, 'Date': date_str, 'PnL': t['pnl'],
                        'Outcome': t['outcome'], 'N_Fires': sel['n_fires'],
                        'Direction': direction,
                    })
                    history[tk].append(t['pnl'])

        results[strat_name] = trades

    return results


# ==============================================================================
# MAIN
# ==============================================================================
print(f"\n{'='*W}")
print(f"  COMBINED LONG+SHORT — SELECTION STRATEGY COMPARISON")
print(f"  {BACKTEST_START} -> {BACKTEST_END}")
print(f"  {len(STRATEGIES)} strategies | MAX_TRADES: {MAX_TRADES} per direction")
print(f"{'='*W}")

print(f"\n  Loading configs from BQ...")
long_ffc = load_bq_configs(LONG_FAMILIES, LONG_F7_THRESH, 'LONG')
long_deploy = select_long_deploy(long_ffc)
short_ffc = load_bq_configs(SHORT_FAMILIES, SHORT_F7_THRESH, 'SHORT')
short_deploy = select_short_deploy(short_ffc)

print(f"\n  Building models...")
long_ds = DirState('long', LONG_FAMILIES)
short_ds = DirState('short', SHORT_FAMILIES)
build_models_for_dir(long_ds, long_deploy, long_ffc)
build_models_for_dir(short_ds, short_deploy, short_ffc)

print(f"\n  Building market context...")
market_df = build_market_context()
_master_cache.clear()
regime_df = compute_regime_features(market_df)

long_raw, long_pool, long_ctx = score_all_tickers(
    long_ds, long_deploy, long_ffc, LONG_CONSENSUS, market_df, regime_df, 'LONG')
short_raw, short_pool, short_ctx = score_all_tickers(
    short_ds, short_deploy, short_ffc, SHORT_CONSENSUS, market_df, regime_df, 'SHORT')

print(f"\n  Phase 2+3: Evaluating {len(STRATEGIES)} strategies...")
long_results = evaluate_strategies(long_raw, long_pool, long_ctx, LONG_CONSENSUS, 'LONG')
short_results = evaluate_strategies(short_raw, short_pool, short_ctx, SHORT_CONSENSUS, 'SHORT')


# ==============================================================================
# RESULTS — STRATEGY COMPARISON (sorted by combined cumulative PnL)
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  STRATEGY COMPARISON — COMBINED LONG + SHORT")
print(f"  {BACKTEST_START} -> {BACKTEST_END}")
print(f"  {len(STRATEGIES)} strategies tested | MAX_TRADES = {MAX_TRADES} per direction")
print(f"{'='*W}\n")

strat_summary = {}
for sn in STRATEGIES:
    lt = long_results.get(sn, [])
    st = short_results.get(sn, [])
    all_t = lt + st

    l_n = len(lt); l_avg = np.mean([t['PnL'] for t in lt]) if lt else 0
    l_wr = np.mean([t['PnL'] > 0 for t in lt]) if lt else 0
    s_n = len(st); s_avg = np.mean([t['PnL'] for t in st]) if st else 0
    s_wr = np.mean([t['PnL'] > 0 for t in st]) if st else 0

    if all_t:
        df_all = pd.DataFrame(all_t)
        daily_ew = df_all.groupby('Date')['PnL'].mean()
        c_cum = daily_ew.sum()
        c_std = daily_ew.std() if len(daily_ew) > 1 else 0
        c_sh = (daily_ew.mean() / c_std * np.sqrt(252)) if c_std > 0 else 0
        l_cum = pd.DataFrame(lt).groupby('Date')['PnL'].mean().sum() if lt else 0
        s_cum = pd.DataFrame(st).groupby('Date')['PnL'].mean().sum() if st else 0
    else:
        c_cum = c_sh = l_cum = s_cum = 0

    strat_summary[sn] = {
        'l_n': l_n, 'l_avg': l_avg, 'l_wr': l_wr, 'l_cum': l_cum,
        's_n': s_n, 's_avg': s_avg, 's_wr': s_wr, 's_cum': s_cum,
        'c_cum': c_cum, 'c_sh': c_sh,
    }

ranked = sorted(strat_summary.keys(), key=lambda x: strat_summary[x]['c_cum'], reverse=True)

print(f"  {'#':>3} {'Strategy':<16} {'L_Tr':>5} {'L_Avg':>7} {'L_WR':>5} {'L_Cum':>8}  "
      f"{'S_Tr':>5} {'S_Avg':>7} {'S_WR':>5} {'S_Cum':>8}  "
      f"{'C_Cum':>8} {'C_Sh':>6}  {'Description'}")
print(f"  {'─'*W}")

for rank, sn in enumerate(ranked, 1):
    s = strat_summary[sn]
    desc = STRATEGIES[sn]['desc']
    marker = ' <-- DEPLOYED' if sn == 'DEPLOYED' else ''
    print(f"  {rank:>3} {sn:<16} {s['l_n']:>5} {s['l_avg']:>+6.2%} {s['l_wr']:>4.0%} {s['l_cum']:>+7.2%}  "
          f"{s['s_n']:>5} {s['s_avg']:>+6.2%} {s['s_wr']:>4.0%} {s['s_cum']:>+7.2%}  "
          f"{s['c_cum']:>+7.2%} {s['c_sh']:>5.1f}  {desc}{marker}")

best_strat = ranked[0]
deployed_cum = strat_summary['DEPLOYED']['c_cum']
best_cum = strat_summary[best_strat]['c_cum']
deployed_rank = ranked.index('DEPLOYED') + 1
print(f"\n  BEST:     {best_strat} ({best_cum:+.2%})")
print(f"  DEPLOYED: #{deployed_rank}/{len(STRATEGIES)} ({deployed_cum:+.2%})")
print(f"  DELTA:    {best_cum - deployed_cum:+.2%} improvement available")


# ==============================================================================
# GROUP ANALYSIS — which category of ranking works best?
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  GROUP ANALYSIS — Which ranking approach works best?")
print(f"{'='*W}\n")

groups = {
    'Baselines (raw prob)': ['DEPLOYED', 'FIRES_ONLY'],
    'Margin-based (prob-thresh)': ['MAX_MARGIN', 'AVG_MARGIN_F', 'SUM_MARGIN', 'SUM_EXCESS'],
    'Conviction (prob/thresh)': ['AVG_RATIO', 'SUM_CONVICT', 'WFIRE_CONVICT'],
    'Soft voting (no binary)': ['SOFT_VOTE', 'SOFT_VOTE_G'],
    'Hybrid (combined)': ['FIRE%×MARGIN', 'FIRE×CONVICT'],
    'Filters': ['GAP_ALIGNED', 'SECTOR_CAP', 'BEST_COMBO'],
}

for group_name, members in groups.items():
    valid = [sn for sn in members if sn in strat_summary]
    if not valid: continue
    best_in_group = max(valid, key=lambda x: strat_summary[x]['c_cum'])
    avg_cum = np.mean([strat_summary[sn]['c_cum'] for sn in valid])
    print(f"  {group_name}:")
    for sn in valid:
        s = strat_summary[sn]
        marker = ' ***' if sn == best_in_group else ''
        print(f"    {sn:<16} C_Cum={s['c_cum']:>+7.2%}  Sharpe={s['c_sh']:>5.1f}  "
              f"L={s['l_cum']:>+6.2%} S={s['s_cum']:>+6.2%}{marker}")
    print(f"    Group avg: {avg_cum:+.2%}")
    print()


# ==============================================================================
# DAILY DETAIL — Best strategy vs Deployed
# ==============================================================================
if best_strat != 'DEPLOYED':
    print(f"\n{'='*W}")
    print(f"  DAILY COMPARISON: {best_strat} vs DEPLOYED")
    print(f"{'='*W}\n")

    all_dates_set = set()
    for sn in [best_strat, 'DEPLOYED']:
        for t in long_results.get(sn, []) + short_results.get(sn, []):
            all_dates_set.add(t['Date'])

    print(f"  {'Date':<12} {'DEPLOYED EW':>12} {'':>4} {best_strat+' EW':>16} {'Delta':>8}  {'Diff tickers'}")
    print(f"  {'─'*90}")

    for date in sorted(all_dates_set):
        dep_t = [t for t in long_results.get('DEPLOYED', []) + short_results.get('DEPLOYED', [])
                 if t['Date'] == date]
        best_t = [t for t in long_results.get(best_strat, []) + short_results.get(best_strat, [])
                  if t['Date'] == date]
        dep_ew = np.mean([t['PnL'] for t in dep_t]) if dep_t else 0
        best_ew = np.mean([t['PnL'] for t in best_t]) if best_t else 0
        delta = best_ew - dep_ew

        dep_tickers = set(t['Ticker'] for t in dep_t)
        best_tickers = set(t['Ticker'] for t in best_t)
        only_dep = dep_tickers - best_tickers
        only_best = best_tickers - dep_tickers
        diff_str = ''
        if only_dep: diff_str += f"-{','.join(sorted(only_dep))} "
        if only_best: diff_str += f"+{','.join(sorted(only_best))}"

        marker = ' <<<' if abs(delta) > 0.005 else ''
        print(f"  {date:<12} {dep_ew:>+11.2%} {'':>4} {best_ew:>+15.2%} {delta:>+7.2%}  {diff_str}{marker}")


# ==============================================================================
# CHARTS
# ==============================================================================
import matplotlib.cm as cm

fig, axes = plt.subplots(2, 2, figsize=(22, 16))
fig.suptitle(f'Combined LONG+SHORT — Selection Strategy Experiment\n'
             f'{BACKTEST_START} -> {BACKTEST_END} | {len(STRATEGIES)} strategies | '
             f'Margin/Conviction vs Deployed',
             fontsize=13, fontweight='bold')

cmap = cm.get_cmap('tab20', len(STRATEGIES))
strat_colors = {sn: cmap(i) for i, sn in enumerate(STRATEGIES)}
strat_colors['DEPLOYED'] = '#1565C0'

# [0,0] Equity curves — top 5 + deployed
ax = axes[0, 0]
top5 = ranked[:5]
show_set = set(top5) | {'DEPLOYED'}
for sn in STRATEGIES:
    all_t = long_results.get(sn, []) + short_results.get(sn, [])
    if not all_t: continue
    df_t = pd.DataFrame(all_t)
    daily = df_t.groupby('Date')['PnL'].mean()
    dates_dt = pd.to_datetime(daily.index)
    cum = daily.cumsum() * 100
    if sn in show_set:
        lw = 3 if sn == best_strat else 2 if sn == 'DEPLOYED' else 1.5
        ls = '--' if sn == 'DEPLOYED' and sn != best_strat else '-'
        ax.plot(dates_dt, cum.values, ls, linewidth=lw, alpha=0.9,
                label=f"{sn} ({cum.values[-1]:+.1f}%)", color=strat_colors.get(sn, '#888'))
    else:
        ax.plot(dates_dt, cum.values, '-', linewidth=0.5, alpha=0.2, color='#999')
ax.axhline(0, color='black', linewidth=0.8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
ax.set_ylabel('Cumulative EW PnL (%)')
ax.set_title(f'Top 5 + DEPLOYED Equity Curves (others in gray)')
ax.legend(fontsize=7, ncol=1); ax.grid(alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# [0,1] Horizontal bar chart — all strategies ranked
ax = axes[0, 1]
y_pos = np.arange(len(ranked))
cum_vals = [strat_summary[sn]['c_cum'] * 100 for sn in ranked]
colors_bar = []
for sn in ranked:
    if sn == 'DEPLOYED':
        colors_bar.append('#1565C0')
    elif strat_summary[sn]['c_cum'] > strat_summary['DEPLOYED']['c_cum']:
        colors_bar.append('#2E7D32')
    elif strat_summary[sn]['c_cum'] > 0:
        colors_bar.append('#66BB6A')
    else:
        colors_bar.append('#C62828')
ax.barh(y_pos, cum_vals, color=colors_bar, edgecolor='white', linewidth=0.5)
ax.set_yticks(y_pos)
ax.set_yticklabels([f"#{i+1} {sn}" for i, sn in enumerate(ranked)], fontsize=7)
ax.axvline(0, color='black', linewidth=0.8)
dep_val = strat_summary['DEPLOYED']['c_cum'] * 100
ax.axvline(dep_val, color='#1565C0', linewidth=1, linestyle='--', alpha=0.7, label=f'DEPLOYED ({dep_val:+.1f}%)')
ax.set_xlabel('Combined Cumulative EW PnL (%)')
ax.set_title('All Strategies Ranked (green = beats DEPLOYED)')
ax.legend(fontsize=8); ax.grid(axis='x', alpha=0.3)
ax.invert_yaxis()

# [1,0] LONG vs SHORT breakdown — top 10
ax = axes[1, 0]
top10 = ranked[:10]
x = np.arange(len(top10))
w = 0.35
l_vals = [strat_summary[sn]['l_cum'] * 100 for sn in top10]
s_vals = [strat_summary[sn]['s_cum'] * 100 for sn in top10]
ax.bar(x - w/2, l_vals, w, label='LONG', color='#1565C0', alpha=0.8)
ax.bar(x + w/2, s_vals, w, label='SHORT', color='#C62828', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels([sn.replace('_', '\n') for sn in top10], fontsize=6, rotation=45, ha='right')
ax.axhline(0, color='black', linewidth=0.8)
ax.set_ylabel('Cumulative EW PnL (%)')
ax.set_title('LONG vs SHORT — Top 10 Strategies')
ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)

# [1,1] Group comparison — avg cumulative by approach category
ax = axes[1, 1]
group_names = []
group_avgs = []
group_bests = []
for gn, members in groups.items():
    valid = [sn for sn in members if sn in strat_summary]
    if not valid: continue
    group_names.append(gn.split('(')[0].strip())
    group_avgs.append(np.mean([strat_summary[sn]['c_cum'] for sn in valid]) * 100)
    group_bests.append(max(strat_summary[sn]['c_cum'] for sn in valid) * 100)
x = np.arange(len(group_names))
ax.bar(x - 0.2, group_avgs, 0.35, label='Group avg', color='#42A5F5', alpha=0.8)
ax.bar(x + 0.2, group_bests, 0.35, label='Group best', color='#2E7D32', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(group_names, fontsize=7, rotation=20, ha='right')
ax.axhline(0, color='black', linewidth=0.8)
ax.set_ylabel('Cumulative EW PnL (%)')
ax.set_title('Approach Category Comparison (avg vs best)')
ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
chart_path = '/tmp/combined_selection_strategies.png'
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
print(f"\n  Chart saved: {chart_path}")
if _IN_COLAB: plt.show()
plt.close()


# ==============================================================================
# FINAL VERDICT
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  FINAL VERDICT")
print(f"{'='*W}")
print(f"  DEPLOYED ({strat_summary['DEPLOYED']['c_cum']:+.2%}) "
      f"vs BEST ({best_strat}: {strat_summary[best_strat]['c_cum']:+.2%})")
n_beat = sum(1 for sn in STRATEGIES if strat_summary[sn]['c_cum'] > strat_summary['DEPLOYED']['c_cum'])
print(f"  {n_beat}/{len(STRATEGIES)} strategies beat DEPLOYED")
print(f"  Improvement range: {strat_summary[ranked[-1]]['c_cum'] - deployed_cum:+.2%} to {best_cum - deployed_cum:+.2%}")
print(f"{'='*W}")


# ==============================================================================
# CLEANUP
# ==============================================================================
keras.backend.clear_session()
gc.collect()
try: ctypes.CDLL("libc.so.6").malloc_trim(0)
except Exception: pass

print(f"\n{'='*W}")
print(f"  SELECTION STRATEGY COMPARISON COMPLETE — {len(STRATEGIES)} strategies")
print(f"  BEST OVERALL: {best_strat}")
print(f"  BEST LAST WEEK: {best_lw}")
print(f"{'='*W}")
