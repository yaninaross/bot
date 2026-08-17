# %% ── SHORT Hybrid Strategy: L5-1 vs L5-2 vs Combined ──────────────────────────
#
# Compares 3 hybrid strategies for SHORT Spike Transformer deployment:
#
#   1. L5-1 HYB (deployed HYB_F7_30): Best fold from L5-1 F5/F6/F7, F7 >= 3.0%
#   2. L5-2 HYB:                      Best fold from L5-2 F5/F6/F7, F7 >= 3.0%
#   3. COMBINED HYB:                  Best (family, fold) from all 6 sources, F7 >= 3.0%
#
# Each model scored with its OWN family's meta.npz scaling stats.
# Models loaded one at a time to manage memory.
#
# Output:
#   - Per-strategy: avg PnL, WR, PF, EqW, total trades, avg trades/day
#   - Per-ticker assignment tables (which source each ticker is assigned to)
#   - Equity curves + comparison charts
#
# SHORT only. Colab-ready.

import os, sys, gc, subprocess, warnings
from datetime import datetime, timedelta
from collections import defaultdict, OrderedDict

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

import tensorflow as tf
import keras
from keras.layers import Dense, Dropout, LayerNormalization
from keras import layers as _kl

from google.cloud import storage as gcs_storage

try:
    from google.cloud import bigquery
except Exception:
    bigquery = None

if tf.config.list_physical_devices('GPU'):
    keras.mixed_precision.set_dtype_policy('mixed_float16')

try:
    import matplotlib
    try:
        import google.colab  # noqa
        _IN_COLAB = True
    except ImportError:
        _IN_COLAB = False
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ==============================================================================
# CONFIG
# ==============================================================================
BUCKET_NAME   = "time-series-bucket-ys"
INFERENCE_DIR = "Data/Inference_v2"
WF_BQ_TABLE   = "ys-time-series.scoring_logs.wf_optimization_results"
PST           = pytz.timezone('America/Los_Angeles')
TAIL_DAYS     = 60

BACKTEST_START = '2026-07-10'
BACKTEST_END   = '2026-08-13'

SLIPPAGE_PCT  = 0.0015
EXIT_TIME_MIN = 780
BASE_PNL_THRESHOLD = 0.018
F7_DEPLOY_THRESHOLD = 0.030

TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}
MICRO_COLS = ['Spread', 'OBI', 'Volume_Delta', 'Trade_Count', 'Delta_Pct',
              'bsize', 'asize', 'Total_Vol']
ROC_HORIZONS = [5, 30, 60]
CONTEXT_ASSETS = ['SPY', 'VIX', 'TLT', 'IWM']
SECTOR_PROXY_ASSETS = ['MARA', 'MSTR', 'ITA']
EXCLUDE_TICKERS = {'SPY', 'VIX', 'TLT', 'IWM', 'XLE', 'ITA', 'BITF', 'MIGI', 'SMLR'}

SCORE_MINUTE = 375
ENTRY_MINUTE = 385

W = 110
gcs_client = gcs_storage.Client()

# ── Shared model params ──
TIME_STEPS = 128
D_MODEL = 64; N_HEADS = 4; D_FF = 128; N_BLOCKS = 2; DROPOUT = 0.1
FILM_HIDDEN = 64; N_HORIZONS = 3; N_ASSET_F = 33; N_REGIME_F = 22

ASSET_COLS = (['Gap_Pct', 'RSI_14', 'RVOL', 'ATR_14', 'Bar_Is_Real'] +
              [f'ROC_{h}m' for h in ROC_HORIZONS] + MICRO_COLS)

CONTEXT_COLS = []
for _t in CONTEXT_ASSETS:
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


# ==============================================================================
# 6 MODEL SOURCES — L5-1 F5/F6/F7 + L5-2 F5/F6/F7
# ==============================================================================
SOURCES = OrderedDict([
    ('L1_F5', {'family': 'L5-1', 'fold_id': 5, 'fold_label': 'F5',
               'bq_mv': 'spike_transformer_short_L5_1_fold5',
               'gcs_data': 'Algos/RA_MoE/short_L5_1_data',
               'gcs_models': 'Algos/RA_MoE/short_L5_1_models'}),
    ('L1_F6', {'family': 'L5-1', 'fold_id': 6, 'fold_label': 'F6',
               'bq_mv': 'spike_transformer_short_L5_1_fold6',
               'gcs_data': 'Algos/RA_MoE/short_L5_1_data',
               'gcs_models': 'Algos/RA_MoE/short_L5_1_models'}),
    ('L1_F7', {'family': 'L5-1', 'fold_id': 7, 'fold_label': 'F7',
               'bq_mv': 'spike_transformer_short_L5_1_fold7',
               'gcs_data': 'Algos/RA_MoE/short_L5_1_data',
               'gcs_models': 'Algos/RA_MoE/short_L5_1_models'}),
    ('L2_F5', {'family': 'L5-2', 'fold_id': 5, 'fold_label': 'F5',
               'bq_mv': 'spike_transformer_short_L5_2_fold5',
               'gcs_data': 'Algos/RA_MoE/short_L5_2_data',
               'gcs_models': 'Algos/RA_MoE/short_L5_2_models'}),
    ('L2_F6', {'family': 'L5-2', 'fold_id': 6, 'fold_label': 'F6',
               'bq_mv': 'spike_transformer_short_L5_2_fold6',
               'gcs_data': 'Algos/RA_MoE/short_L5_2_data',
               'gcs_models': 'Algos/RA_MoE/short_L5_2_models'}),
    ('L2_F7', {'family': 'L5-2', 'fold_id': 7, 'fold_label': 'F7',
               'bq_mv': 'spike_transformer_short_L5_2_fold7',
               'gcs_data': 'Algos/RA_MoE/short_L5_2_data',
               'gcs_models': 'Algos/RA_MoE/short_L5_2_models'}),
])

FAMILIES = OrderedDict([
    ('L5-1', {'gcs_data': 'Algos/RA_MoE/short_L5_1_data',
              'gcs_models': 'Algos/RA_MoE/short_L5_1_models',
              'sources': ['L1_F5', 'L1_F6', 'L1_F7'],
              'color': '#1565C0'}),
    ('L5-2', {'gcs_data': 'Algos/RA_MoE/short_L5_2_data',
              'gcs_models': 'Algos/RA_MoE/short_L5_2_models',
              'sources': ['L2_F5', 'L2_F6', 'L2_F7'],
              'color': '#FF6F00'}),
])

STRATEGIES = OrderedDict([
    ('L5-1 HYB',     {'sources': ['L1_F5', 'L1_F6', 'L1_F7'],
                       'label': 'L5-1 Hybrid (deployed)',
                       'color': '#1565C0', 'marker': 's', 'ls': '-'}),
    ('L5-2 HYB',     {'sources': ['L2_F5', 'L2_F6', 'L2_F7'],
                       'label': 'L5-2 Hybrid',
                       'color': '#FF6F00', 'marker': 'D', 'ls': '-'}),
    ('COMBINED HYB', {'sources': list(SOURCES.keys()),
                       'label': 'Combined L5-1+L5-2',
                       'color': '#2E7D32', 'marker': 'o', 'ls': '-'}),
    ('RECENCY HYB',  {'sources': list(SOURCES.keys()),
                       'label': 'Recency (F7>F6>F5)',
                       'color': '#9C27B0', 'marker': '^', 'ls': '-',
                       'priority': 'recency'}),
])


# ==============================================================================
# MODEL ARCHITECTURE  (SpikeTransformer — same for all sources)
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
            TIME_STEPS, DROPOUT, name='backbone')
        self.attn_pool = _AttentionPooling(D_MODEL, name='attn_pool')
        self.film = _FiLMConditioner(D_MODEL, FILM_HIDDEN, name='film')
        self.drop = Dropout(DROPOUT)
        self.head = Dense(N_HORIZONS, activation='sigmoid', name='spike_head')

    def call(self, inputs, training=False):
        h = self.backbone(inputs['seq_input'], training=training)
        h = self.attn_pool(h, training=training)
        h = self.film(h, inputs['regime_input'], training=training)
        return self.head(self.drop(h, training=training))


# ==============================================================================
# DATA LOADING
# ==============================================================================
_NEEDED_COLS = {'Open', 'High', 'Low', 'Close', 'Volume',
                'open', 'high', 'low', 'close', 'volume',
                'Spread', 'OBI', 'Volume_Delta', 'Trade_Count', 'Delta_Pct',
                'bsize', 'asize', 'Total_Vol',
                'Total_vol', 'Trade_count', 'Volume_delta', 'Delta_pct',
                'Obi', 'Bsize', 'Asize',
                'Datetime', 'Date', 'Time', 'Timestamp', 'Level_0'}


def load_master(ticker, tail_days=None):
    clean = ticker.replace("I:", "").replace("X:", "")
    file_ticker = TICKER_FILE_MAP.get(clean, clean)
    blob_path = f"{INFERENCE_DIR}/{file_ticker}_master_inference.parquet"
    local_path = f"/tmp/_bt_{file_ticker}_master.parquet"
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
        if tail_days is not None:
            cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=tail_days)
            df = df[df.index >= cutoff].copy()
            if df.empty:
                return None
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
        df = df.loc[:, ~df.columns.duplicated(keep='first')]
        for c in MICRO_COLS:
            if c not in df.columns:
                df[c] = 0.0
        df = df[df.index.notna()].sort_index()
        return df
    except Exception as e:
        if os.path.exists(local_path):
            os.remove(local_path)
        print(f"      load_master({ticker}): FAILED - {e}")
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
# FEATURE ENGINEERING
# ==============================================================================
def build_market_context():
    frames = []
    vix_close = None
    for ca in CONTEXT_ASSETS:
        df = load_master(ca, tail_days=TAIL_DAYS)
        if df is None:
            raise RuntimeError(f"Cannot load context asset {ca}")
        df = df.sort_index()
        df = heal_data(df)
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
    df = df.sort_index()
    df = heal_data(df)
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


def compute_regime_features(market_df):
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
    if 'VIX_Close' in market_df.columns:
        rf['regime_vix_close'] = market_df['VIX_Close'].ffill().fillna(0)
    else:
        rf['regime_vix_close'] = 0.0
    tlt_ret = (market_df['TLT_Anchor_ROC'].fillna(0)
               if 'TLT_Anchor_ROC' in market_df.columns
               else pd.Series(0.0, index=market_df.index))
    rf['regime_spy_tlt_corr'] = spy_ret.rolling(
        60, min_periods=30).corr(tlt_ret).fillna(0).clip(-1, 1)
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
        spy_ret.rolling(60, min_periods=30).sum()
    ).fillna(0)
    defense_ret = (market_df['ITA_ret'].fillna(0)
                   if 'ITA_ret' in market_df.columns
                   else pd.Series(0.0, index=market_df.index))
    rf['regime_defense_rs'] = (
        defense_ret.rolling(60, min_periods=30).sum() -
        spy_ret.rolling(60, min_periods=30).sum()
    ).fillna(0)
    rf['regime_safe_haven'] = (
        tlt_ret.rolling(60, min_periods=30).sum() -
        spy_ret.rolling(60, min_periods=30).sum()
    ).fillna(0)
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
# MODEL LOADING — per-family meta.npz, per-source model weights
# ==============================================================================
def _ensure_file(gcs_path, local_path):
    if os.path.exists(local_path):
        return local_path
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    gcs_client.bucket(BUCKET_NAME).blob(gcs_path).download_to_filename(local_path)
    return local_path


def load_family_meta(family_key):
    fam = FAMILIES[family_key]
    tag = family_key.replace('-', '_')
    local_meta = f'/tmp/bt_short_{tag}_hyb_meta.npz'
    meta_path = _ensure_file(f"{fam['gcs_data']}/meta.npz", local_meta)
    meta = np.load(meta_path, allow_pickle=True)
    return meta['scaling_stats'].item(), meta['regime_stats'].item()


def load_source_model(src_key):
    src = SOURCES[src_key]
    tag = src_key
    fold_id = src['fold_id']
    model = SpikeTransformer(name=f'spike_transformer_{tag}')
    dummy_seq = tf.zeros((1, TIME_STEPS, N_ASSET_F))
    dummy_reg = tf.zeros((1, N_REGIME_F))
    model({'seq_input': dummy_seq, 'regime_input': dummy_reg})

    local_dir = f'/tmp/bt_short_hyb_models/{tag}/fold_{fold_id}'
    weights_path = os.path.join(local_dir, 'model.weights.h5')
    if not os.path.exists(weights_path):
        os.makedirs(local_dir, exist_ok=True)
        prefix = f"{src['gcs_models']}/fold_{fold_id}"
        bucket = gcs_client.bucket(BUCKET_NAME)
        for blob in bucket.list_blobs(prefix=prefix):
            fname = blob.name.split('/')[-1]
            blob.download_to_filename(os.path.join(local_dir, fname))
    model.load_weights(weights_path)
    print(f"    {src_key} (fold {fold_id}): {model.count_params():,} params")
    return model


# ==============================================================================
# BQ CONFIG LOADING
# ==============================================================================
def load_source_configs(bq_model_version):
    if bigquery is None:
        print(f"  BigQuery not available")
        return {}
    bq_client = bigquery.Client()
    q = f"""
    SELECT ticker, experiment, tpsl_mode, prob_thresh, tp_pct, sl_pct,
           grace_minutes, dip_pct, max_wait_minutes,
           val_b_avg_pnl, val_b_win_rate, val_b_count
    FROM `{WF_BQ_TABLE}`
    WHERE model_version = '{bq_model_version}'
      AND survived_a = TRUE
      AND val_b_avg_pnl >= {BASE_PNL_THRESHOLD}
    """
    rows = bq_client.query(q).to_dataframe()
    if len(rows) == 0:
        return {}
    best = rows.loc[rows.groupby('ticker')['val_b_avg_pnl'].idxmax()]
    configs = {}
    for _, row in best.iterrows():
        tk = row['ticker']
        if tk in EXCLUDE_TICKERS:
            continue
        exp = row['experiment']
        h_part = exp.replace(f'{bq_model_version}_', '').split('_')[0]
        horizon_idx = HORIZON_FROM_LABEL.get(h_part, 1)
        mode = row['tpsl_mode']
        if pd.isna(mode) or mode == '':
            if '_GRACE' in exp:
                mode = 'grace'
            elif '_LIMIT' in exp:
                mode = 'limit_entry'
            else:
                mode = 'static'
        configs[tk] = {
            'horizon': horizon_idx,
            'horizon_label': h_part,
            'prob_thresh': float(row['prob_thresh']),
            'tp_pct': float(row['tp_pct']),
            'sl_pct': float(row['sl_pct']),
            'mode': mode,
            'grace_minutes': int(row['grace_minutes']) if pd.notna(row['grace_minutes']) else 0,
            'dip_pct': float(row['dip_pct']) if pd.notna(row['dip_pct']) else 0,
            'max_wait': int(row['max_wait_minutes']) if pd.notna(row['max_wait_minutes']) else 0,
            'val_b_avg_pnl': float(row['val_b_avg_pnl']),
            'val_b_win_rate': float(row['val_b_win_rate']) if pd.notna(row['val_b_win_rate']) else 0,
            'val_b_count': int(row['val_b_count']) if pd.notna(row['val_b_count']) else 0,
            'experiment': exp,
        }
    return configs


# ==============================================================================
# SCORING
# ==============================================================================
def score_ticker_day(model, scaling_stats, regime_stats,
                     ticker, cfg, df_feat, regime_df, target_date):
    target_pst = PST.localize(datetime(target_date.year, target_date.month,
                                       target_date.day,
                                       SCORE_MINUTE // 60, SCORE_MINUTE % 60))
    target_utc = target_pst.astimezone(pytz.utc)
    candidates = df_feat.index[df_feat.index <= target_utc]
    if len(candidates) == 0:
        return None
    row_idx = df_feat.index.get_loc(candidates[-1])
    bar_ts_pst = df_feat.index[row_idx].tz_convert(PST)
    score_date = bar_ts_pst.date()
    if score_date != target_date:
        return None
    bar_minute = bar_ts_pst.hour * 60 + bar_ts_pst.minute
    if bar_minute != SCORE_MINUTE:
        return None
    if row_idx < TIME_STEPS:
        return None
    if 'Volume' in df_feat.columns and float(df_feat['Volume'].iloc[row_idx]) == 0:
        return None

    df_scaled = apply_scaling(df_feat, scaling_stats, ALL_COLS)
    XA = df_scaled[ALL_COLS].values.astype(np.float32)
    seq_batch = XA[row_idx - TIME_STEPS:row_idx][np.newaxis]

    regime_at_entry = regime_df.iloc[regime_df.index.get_indexer(
        [df_feat.index[row_idx]], method='pad')[0]]
    regime_scaled = np.zeros(N_REGIME_F, dtype=np.float32)
    for i, col in enumerate(REGIME_COLS):
        val = float(regime_at_entry[col]) if col in regime_at_entry.index else 0.0
        if col in regime_stats:
            s = regime_stats[col]
            val = np.clip((val - s['med']) / s['iqr'], -10, 10)
        regime_scaled[i] = val
    regime_batch = regime_scaled[np.newaxis]

    preds = model(
        {'seq_input': seq_batch, 'regime_input': regime_batch},
        training=False
    ).numpy().flatten()

    h_idx = cfg['horizon']
    prob = float(preds[h_idx])
    close = float(df_feat['Close'].iloc[row_idx])

    return {
        'prob': round(prob, 4),
        'close': round(close, 4),
        'bar_idx': row_idx,
        'all_probs': {HORIZON_LABELS[i]: round(float(preds[i]), 4)
                      for i in range(N_HORIZONS)},
    }


# ==============================================================================
# SHORT TRADE SIMULATION
# ==============================================================================
def simulate_short_trade(df_feat, entry_idx, cfg):
    entry_date_pst = df_feat.index[entry_idx].tz_convert(PST).date()
    close = float(df_feat['Close'].iloc[entry_idx])

    if cfg['mode'] == 'limit_entry':
        entry_price = close * (1 + cfg.get('dip_pct', 0))
        max_wait = cfg.get('max_wait', 30)
        fill_idx = None
        for k in range(entry_idx + 1, len(df_feat)):
            bar_pst = df_feat.index[k].tz_convert(PST)
            if bar_pst.date() != entry_date_pst:
                break
            elapsed = (bar_pst.hour * 60 + bar_pst.minute) - (
                df_feat.index[entry_idx].tz_convert(PST).hour * 60 +
                df_feat.index[entry_idx].tz_convert(PST).minute)
            if elapsed > max_wait:
                break
            if float(df_feat['High'].iloc[k]) >= entry_price:
                fill_idx = k
                break
        if fill_idx is None:
            return None
        start_idx = fill_idx
    else:
        entry_price = close
        start_idx = entry_idx

    tp_price = entry_price * (1 - cfg['tp_pct'])
    sl_price = entry_price * (1 + cfg['sl_pct'])
    grace_minutes = cfg.get('grace_minutes', 0) if cfg['mode'] == 'grace' else 0
    entry_time_min = (df_feat.index[start_idx].tz_convert(PST).hour * 60 +
                      df_feat.index[start_idx].tz_convert(PST).minute)

    for k in range(start_idx + 1, len(df_feat)):
        bar_pst = df_feat.index[k].tz_convert(PST)
        if bar_pst.date() != entry_date_pst:
            exit_price = float(df_feat['Close'].iloc[k - 1])
            pnl = (entry_price - exit_price) / entry_price
            return {'pnl': pnl, 'entry': entry_price, 'exit': exit_price, 'outcome': 'EOD'}

        bar_min = bar_pst.hour * 60 + bar_pst.minute
        high_k = float(df_feat['High'].iloc[k])
        low_k = float(df_feat['Low'].iloc[k])

        elapsed = bar_min - entry_time_min
        sl_active = elapsed >= grace_minutes

        tp_hit = low_k <= tp_price
        sl_hit = sl_active and high_k >= sl_price

        if tp_hit and sl_hit:
            if np.random.random() < 0.5:
                return {'pnl': cfg['tp_pct'], 'entry': entry_price, 'exit': tp_price, 'outcome': 'TP'}
            else:
                return {'pnl': -cfg['sl_pct'], 'entry': entry_price, 'exit': sl_price, 'outcome': 'SL'}
        if tp_hit:
            return {'pnl': cfg['tp_pct'], 'entry': entry_price, 'exit': tp_price, 'outcome': 'TP'}
        if sl_hit:
            return {'pnl': -cfg['sl_pct'], 'entry': entry_price, 'exit': sl_price, 'outcome': 'SL'}
        if bar_min >= EXIT_TIME_MIN:
            exit_price = float(df_feat['Close'].iloc[k])
            pnl = (entry_price - exit_price) / entry_price
            return {'pnl': pnl, 'entry': entry_price, 'exit': exit_price, 'outcome': 'EOD'}

    exit_price = float(df_feat['Close'].iloc[-1])
    pnl = (entry_price - exit_price) / entry_price
    return {'pnl': pnl, 'entry': entry_price, 'exit': exit_price, 'outcome': 'EOD'}


# ==============================================================================
# STRATEGY ASSIGNMENT — best (family, fold) per ticker
# ==============================================================================
def build_hybrid_assignment(source_cfgs, allowed_sources):
    """For each ticker in allowed_sources, pick the source with best val_b_avg_pnl.
    F7 sources are filtered to val_b >= F7_DEPLOY_THRESHOLD.
    Returns: {ticker: source_key}"""
    assignment = {}
    for src_key in allowed_sources:
        if src_key not in source_cfgs:
            continue
        is_f7 = SOURCES[src_key]['fold_label'] == 'F7'
        for tk, cfg in source_cfgs[src_key].items():
            if is_f7 and cfg['val_b_avg_pnl'] < F7_DEPLOY_THRESHOLD:
                continue
            if tk not in assignment or cfg['val_b_avg_pnl'] > source_cfgs[assignment[tk]][tk]['val_b_avg_pnl']:
                assignment[tk] = src_key
    return assignment


def build_recency_assignment(source_cfgs, allowed_sources):
    """Prioritize newer folds: F7 > F6 > F5.
    Within same fold level, pick the source with best val_b_avg_pnl.
    F7 sources still filtered to val_b >= F7_DEPLOY_THRESHOLD."""
    FOLD_PRIORITY = {'F7': 3, 'F6': 2, 'F5': 1}
    assignment = {}
    assignment_prio = {}

    for src_key in sorted(allowed_sources,
                          key=lambda s: FOLD_PRIORITY.get(SOURCES[s]['fold_label'], 0),
                          reverse=True):
        if src_key not in source_cfgs:
            continue
        is_f7 = SOURCES[src_key]['fold_label'] == 'F7'
        fold_prio = FOLD_PRIORITY.get(SOURCES[src_key]['fold_label'], 0)

        for tk, cfg in source_cfgs[src_key].items():
            if is_f7 and cfg['val_b_avg_pnl'] < F7_DEPLOY_THRESHOLD:
                continue
            cur_prio = assignment_prio.get(tk, -1)
            if fold_prio > cur_prio:
                assignment[tk] = src_key
                assignment_prio[tk] = fold_prio
            elif fold_prio == cur_prio:
                if cfg['val_b_avg_pnl'] > source_cfgs[assignment[tk]][tk]['val_b_avg_pnl']:
                    assignment[tk] = src_key
    return assignment


def get_trading_days(start, end):
    spy = load_master('SPY', tail_days=TAIL_DAYS)
    if spy is None:
        raise RuntimeError("Cannot load SPY data")
    spy_pst = spy.index.tz_convert(PST)
    dates = sorted(set(spy_pst.date))
    s = pd.to_datetime(start).date()
    e = pd.to_datetime(end).date()
    return [d for d in dates if s <= d <= e]


# ==============================================================================
# STATS COMPUTATION
# ==============================================================================
def compute_strategy_stats(trades_list, n_trading_days):
    if not trades_list:
        return {'n': 0, 'avg_pnl': 0, 'wr': 0, 'pf': 0,
                'avg_win': 0, 'avg_loss': 0, 'eqw': 0, 'n_tickers': 0,
                'nw': 0, 'nl': 0, 'tp': 0, 'sl': 0, 'eod': 0,
                'trades_per_day': 0, 'n_days': n_trading_days}
    df = pd.DataFrame(trades_list)
    n = len(df)
    nw = (df['PnL'] > 0).sum()
    nl = n - nw
    avg_pnl = df['PnL'].mean()
    wr = nw / n if n > 0 else 0
    avg_win = df.loc[df['PnL'] > 0, 'PnL'].mean() if nw > 0 else 0
    avg_loss = df.loc[df['PnL'] <= 0, 'PnL'].mean() if nl > 0 else 0
    pf = abs(avg_win * nw / (avg_loss * nl)) if nl > 0 and avg_loss != 0 else float('inf')
    eqw = df.groupby('Date')['PnL'].mean().sum()
    n_tickers = df['Ticker'].nunique()
    tp = (df['Outcome'] == 'TP').sum()
    sl = (df['Outcome'] == 'SL').sum()
    eod = (df['Outcome'] == 'EOD').sum()
    trades_per_day = n / n_trading_days if n_trading_days > 0 else 0
    return {'n': n, 'avg_pnl': avg_pnl, 'wr': wr, 'pf': pf,
            'avg_win': avg_win, 'avg_loss': avg_loss, 'eqw': eqw,
            'n_tickers': n_tickers, 'nw': nw, 'nl': nl,
            'tp': tp, 'sl': sl, 'eod': eod,
            'trades_per_day': trades_per_day, 'n_days': n_trading_days}


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
np.random.seed(42)

print(f"\n{'='*W}")
print(f"  SHORT HYBRID STRATEGY COMPARISON: L5-1 vs L5-2 vs COMBINED")
print(f"  Period: {BACKTEST_START} -> {BACKTEST_END}")
print(f"  Slippage: {SLIPPAGE_PCT:.2%} each way ({2*SLIPPAGE_PCT:.2%} RT)")
print(f"  Base threshold: {BASE_PNL_THRESHOLD:.1%} | F7 filter: {F7_DEPLOY_THRESHOLD:.1%}")
print(f"  6 sources: L5-1 F5/F6/F7 + L5-2 F5/F6/F7")
print(f"{'='*W}")


# ── Phase 1: Load BQ configs for all 6 sources ──
print(f"\n{'='*W}")
print(f"  PHASE 1: LOAD BQ CONFIGS (6 sources)")
print(f"{'='*W}")

source_cfgs = {}
for src_key, src in SOURCES.items():
    cfgs = load_source_configs(src['bq_mv'])
    source_cfgs[src_key] = cfgs
    is_f7 = src['fold_label'] == 'F7'
    if is_f7:
        n_f7_pass = sum(1 for c in cfgs.values() if c['val_b_avg_pnl'] >= F7_DEPLOY_THRESHOLD)
        print(f"  {src_key} ({src['bq_mv']}): {len(cfgs)} tickers"
              f" ({n_f7_pass} pass F7 >= {F7_DEPLOY_THRESHOLD:.1%})")
    else:
        print(f"  {src_key} ({src['bq_mv']}): {len(cfgs)} tickers")
    if not cfgs:
        print(f"    >>> NO BQ DATA — optimizer not run yet? Skipping this source.")


# ── Phase 2: Build strategy assignments ──
print(f"\n{'='*W}")
print(f"  PHASE 2: BUILD STRATEGY ASSIGNMENTS")
print(f"{'='*W}")

strategy_assignments = {}
for strat_name, strat_cfg in STRATEGIES.items():
    if strat_cfg.get('priority') == 'recency':
        assignment = build_recency_assignment(source_cfgs, strat_cfg['sources'])
    else:
        assignment = build_hybrid_assignment(source_cfgs, strat_cfg['sources'])
    strategy_assignments[strat_name] = assignment
    print(f"\n  {strat_cfg['label']} ({len(assignment)} tickers):")

    src_counts = defaultdict(list)
    for tk, src_key in sorted(assignment.items()):
        src_counts[src_key].append(tk)
    for src_key in SOURCES:
        if src_key in src_counts:
            tks = src_counts[src_key]
            print(f"    from {src_key}: {len(tks)} — {', '.join(sorted(tks))}")

all_tickers = set()
for cfgs in source_cfgs.values():
    all_tickers |= set(cfgs.keys())
print(f"\n  Total unique tickers across all sources: {len(all_tickers)}")


# ── Phase 3: Market context + regime ──
print(f"\n{'='*W}")
print(f"  PHASE 3: MARKET CONTEXT + REGIME")
print(f"{'='*W}")

market_df = build_market_context()
regime_df = compute_regime_features(market_df)

trading_days = get_trading_days(BACKTEST_START, BACKTEST_END)
n_trading_days = len(trading_days)
print(f"  Trading days: {n_trading_days} ({trading_days[0]} -> {trading_days[-1]})")


# ── Phase 4: Feature engineering ──
print(f"\n{'='*W}")
print(f"  PHASE 4: FEATURE ENGINEERING ({len(all_tickers)} tickers)")
print(f"{'='*W}")

ticker_features = {}
for ticker in sorted(all_tickers):
    df_feat = engineer_asset(ticker, market_df)
    if df_feat is not None:
        ticker_features[ticker] = df_feat
        print(f"    {ticker}: {len(df_feat):,} bars")
    else:
        print(f"    {ticker}: FAILED")


# ── Phase 5: Score all sources — one family at a time, one model at a time ──
print(f"\n{'='*W}")
print(f"  PHASE 5: SCORE ALL SOURCES (models loaded one at a time)")
print(f"{'='*W}")

score_cache = {}

for fam_key, fam_cfg in FAMILIES.items():
    fam_sources = [s for s in fam_cfg['sources'] if source_cfgs.get(s)]
    if not fam_sources:
        print(f"\n  {fam_key}: No sources with BQ data — skipping family")
        continue

    print(f"\n  Loading {fam_key} meta.npz...")
    scaling_stats, regime_stats = load_family_meta(fam_key)

    for src_key in fam_sources:
        cfgs = source_cfgs[src_key]
        if not cfgs:
            continue

        print(f"\n  Loading model {src_key}...")
        model = load_source_model(src_key)

        np.random.seed(42)
        n_scored = 0
        for ticker in sorted(cfgs.keys()):
            if ticker not in ticker_features:
                continue
            cfg = cfgs[ticker]
            for day in trading_days:
                result = score_ticker_day(
                    model, scaling_stats, regime_stats,
                    ticker, cfg, ticker_features[ticker], regime_df, day)
                if result is not None:
                    score_cache[(src_key, ticker, day.strftime('%Y-%m-%d'))] = result
                    n_scored += 1
        print(f"    {src_key}: {n_scored} valid scores across {len(cfgs)} tickers")

        del model
        keras.backend.clear_session()
        gc.collect()

print(f"\n  Total cached scores: {len(score_cache)}")


# ── Phase 6: Run trade simulation for each strategy ──
print(f"\n{'='*W}")
print(f"  PHASE 6: TRADE SIMULATION")
print(f"{'='*W}")

strategy_trades = {}

for strat_name, strat_cfg in STRATEGIES.items():
    assignment = strategy_assignments[strat_name]
    trades = []

    np.random.seed(42)
    for day in trading_days:
        day_str = day.strftime('%Y-%m-%d')
        for ticker, src_key in sorted(assignment.items()):
            cache_key = (src_key, ticker, day_str)
            if cache_key not in score_cache:
                continue
            sc = score_cache[cache_key]
            cfg = source_cfgs[src_key][ticker]
            if sc['prob'] < cfg['prob_thresh']:
                continue
            if ticker not in ticker_features:
                continue
            entry_idx = sc['bar_idx'] + (ENTRY_MINUTE - SCORE_MINUTE)
            trade = simulate_short_trade(ticker_features[ticker], entry_idx, cfg)
            if trade is not None:
                trade['pnl'] -= 2 * SLIPPAGE_PCT
                trades.append({
                    'Date': day_str,
                    'Ticker': ticker,
                    'Source': src_key,
                    'Family': SOURCES[src_key]['family'],
                    'Fold': SOURCES[src_key]['fold_label'],
                    'Prob': sc['prob'],
                    'Thresh': cfg['prob_thresh'],
                    'PnL': trade['pnl'],
                    'Outcome': trade['outcome'],
                    'Entry': trade['entry'],
                    'Exit': trade['exit'],
                    'Mode': cfg['mode'],
                    'Horizon': cfg['horizon_label'],
                    'TP_pct': cfg['tp_pct'],
                    'SL_pct': cfg['sl_pct'],
                    'val_b': cfg['val_b_avg_pnl'],
                })

    strategy_trades[strat_name] = trades
    print(f"  {strat_cfg['label']}: {len(trades)} trades across"
          f" {len(set(t['Ticker'] for t in trades))} tickers")


# ==============================================================================
# RESULTS — STRATEGY COMPARISON
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  STRATEGY COMPARISON")
print(f"  Period: {BACKTEST_START} -> {BACKTEST_END} ({n_trading_days} trading days)")
print(f"{'='*W}")

all_stats = {}
for strat_name in STRATEGIES:
    all_stats[strat_name] = compute_strategy_stats(strategy_trades[strat_name], n_trading_days)

print(f"\n  {'Strategy':<26} {'Tickers':>7} {'Trades':>7} {'Avg PnL':>8} {'WR':>5}"
      f" {'PF':>6} {'AvgWin':>7} {'AvgLoss':>8} {'EqW PnL':>8}"
      f" {'Trd/Day':>8}")
print(f"  {'-'*100}")
for strat_name, strat_cfg in STRATEGIES.items():
    s = all_stats[strat_name]
    pf_str = f"{s['pf']:.2f}" if s['pf'] < 999 else 'inf'
    print(f"  {strat_cfg['label']:<26} {s['n_tickers']:>7} {s['n']:>7} {s['avg_pnl']:>+7.2%}"
          f" {s['wr']:>4.0%} {pf_str:>6} {s['avg_win']:>+6.2%} {s['avg_loss']:>+7.2%}"
          f" {s['eqw']:>+7.2%} {s['trades_per_day']:>7.1f}")

# ── Outcome breakdown ──
print(f"\n  OUTCOME BREAKDOWN:")
print(f"  {'Strategy':<26} {'TP':>5} {'SL':>5} {'EOD':>5} {'TP%':>5} {'SL%':>5} {'EOD%':>5}")
print(f"  {'-'*60}")
for strat_name, strat_cfg in STRATEGIES.items():
    s = all_stats[strat_name]
    if s['n'] == 0:
        continue
    print(f"  {strat_cfg['label']:<26} {s['tp']:>5} {s['sl']:>5} {s['eod']:>5}"
          f" {s['tp']/s['n']:>4.0%} {s['sl']/s['n']:>4.0%} {s['eod']/s['n']:>4.0%}")

# ── Delta analysis ──
ref = all_stats['L5-1 HYB']
print(f"\n  DELTA vs DEPLOYED (L5-1 HYB):")
print(f"  {'Strategy':<26} {'ΔAvgPnL':>8} {'ΔEqW':>8} {'ΔTrades':>8} {'ΔTickers':>8}"
      f" {'ΔTrd/Day':>9}")
print(f"  {'-'*72}")
for strat_name in STRATEGIES:
    if strat_name == 'L5-1 HYB':
        continue
    s = all_stats[strat_name]
    d_avg = s['avg_pnl'] - ref['avg_pnl'] if ref['n'] > 0 else 0
    d_eqw = s['eqw'] - ref['eqw']
    d_n = s['n'] - ref['n']
    d_tk = s['n_tickers'] - ref['n_tickers']
    d_tpd = s['trades_per_day'] - ref['trades_per_day']
    scfg = STRATEGIES[strat_name]
    print(f"  {scfg['label']:<26} {d_avg:>+7.2%} {d_eqw:>+7.2%} {d_n:>+8d} {d_tk:>+8d}"
          f" {d_tpd:>+8.1f}")


# ==============================================================================
# PER-TICKER ASSIGNMENT TABLE — COMBINED HYB
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  COMBINED HYB — PER-TICKER ASSIGNMENT (best family+fold per ticker)")
print(f"{'='*W}")

combined_assign = strategy_assignments['COMBINED HYB']
l1_assign = strategy_assignments['L5-1 HYB']

print(f"\n  {'Ticker':<7} {'Source':>7} {'Family':>6} {'Fold':>4}"
      f" {'val_b':>8} {'N':>4} {'AvgPnL':>8} {'In L5-1?':>9} {'Migration':>12}")
print(f"  {'-'*82}")

combined_trade_df = pd.DataFrame(strategy_trades['COMBINED HYB']) if strategy_trades['COMBINED HYB'] else pd.DataFrame()
l1_trade_df = pd.DataFrame(strategy_trades['L5-1 HYB']) if strategy_trades['L5-1 HYB'] else pd.DataFrame()

n_new = 0; n_upgraded = 0; n_kept = 0
for tk in sorted(combined_assign.keys()):
    src_key = combined_assign[tk]
    src = SOURCES[src_key]
    val_b = source_cfgs[src_key][tk]['val_b_avg_pnl']
    in_l1 = tk in l1_assign
    l1_src = l1_assign.get(tk, '')

    if not combined_trade_df.empty and tk in combined_trade_df['Ticker'].values:
        tk_trades = combined_trade_df[combined_trade_df['Ticker'] == tk]
        n_t = len(tk_trades)
        avg_pnl = tk_trades['PnL'].mean()
    else:
        n_t = 0
        avg_pnl = 0

    if not in_l1:
        migration = 'NEW'
        n_new += 1
    elif src_key == l1_src:
        migration = 'kept'
        n_kept += 1
    else:
        migration = f'{l1_src}->{src_key}'
        n_upgraded += 1

    in_flag = 'Y' if in_l1 else '-'
    print(f"  {tk:<7} {src_key:>7} {src['family']:>6} {src['fold_label']:>4}"
          f" {val_b:>+7.2%} {n_t:>4} {avg_pnl:>+7.2%} {in_flag:>9} {migration:>12}")

print(f"\n  Summary: {n_kept} kept same source | {n_upgraded} upgraded to different source"
      f" | {n_new} NEW from L5-2")


# ==============================================================================
# SOURCE DISTRIBUTION — how many tickers per source per strategy
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  SOURCE DISTRIBUTION")
print(f"{'='*W}")

print(f"\n  {'Source':<8}", end='')
for strat_name in STRATEGIES:
    short = strat_name[:12]
    print(f" {short:>12}", end='')
print()
print(f"  {'-'*44}")

for src_key in SOURCES:
    print(f"  {src_key:<8}", end='')
    for strat_name in STRATEGIES:
        n = sum(1 for v in strategy_assignments[strat_name].values() if v == src_key)
        print(f" {n:>12}", end='')
    print()


# ==============================================================================
# DAILY BREAKDOWN
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  DAILY BREAKDOWN — Trades and EqW PnL")
print(f"{'='*W}")

header = f"  {'Date':<12}"
for strat_name in STRATEGIES:
    short = STRATEGIES[strat_name]['label'][:12]
    header += f" {short+' N':>7} {short+' EqW':>8}"
print(header)
print(f"  {'-'*(12 + (7+8+1)*len(STRATEGIES))}")

all_dates = sorted(set().union(*(
    set(pd.DataFrame(strategy_trades[sk])['Date'].unique())
    for sk in STRATEGIES if strategy_trades[sk]
)))

cum = {sk: 0.0 for sk in STRATEGIES}
for d in all_dates:
    row_str = f"  {d:<12}"
    has_any = False
    for sk in STRATEGIES:
        d_trades = [t for t in strategy_trades[sk] if t['Date'] == d]
        n_d = len(d_trades)
        eqw_d = np.mean([t['PnL'] for t in d_trades]) if n_d > 0 else 0
        cum[sk] += eqw_d
        if n_d > 0:
            has_any = True
        row_str += f" {n_d:>7} {eqw_d:>+7.2%}"
    if has_any:
        print(row_str)

print(f"\n  CUMULATIVE:")
for sk in STRATEGIES:
    label = STRATEGIES[sk]['label']
    s = all_stats[sk]
    print(f"    {label:<26} EqW {cum[sk]:>+7.2%} | {s['n']} trades | {s['trades_per_day']:.1f}/day")


# ==============================================================================
# PER-TICKER PnL COMPARISON (tickers in 2+ strategies)
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  PER-TICKER PnL ACROSS STRATEGIES")
print(f"{'='*W}")

all_strat_tickers = set()
for strat_name in STRATEGIES:
    for t in strategy_trades[strat_name]:
        all_strat_tickers.add(t['Ticker'])

print(f"\n  {'Ticker':<7}", end='')
for strat_name in STRATEGIES:
    short = strat_name[:10]
    print(f" {short+' N':>6} {short+' Avg':>8}", end='')
print(f"  {'Best':>12}")
print(f"  {'-'*(7 + (6+8+1)*len(STRATEGIES) + 14)}")

for tk in sorted(all_strat_tickers):
    print(f"  {tk:<7}", end='')
    avgs = {}
    for strat_name in STRATEGIES:
        tk_trades = [t for t in strategy_trades[strat_name] if t['Ticker'] == tk]
        n_t = len(tk_trades)
        avg = np.mean([t['PnL'] for t in tk_trades]) if tk_trades else 0
        avgs[strat_name] = avg
        if n_t > 0:
            print(f" {n_t:>6} {avg:>+7.2%}", end='')
        else:
            print(f" {'--':>6} {'--':>8}", end='')
    best_strat = max(avgs, key=avgs.get) if any(v != 0 for v in avgs.values()) else '-'
    best_short = best_strat[:12] if best_strat != '-' else '-'
    print(f"  {best_short:>12}")


# ==============================================================================
# TRADE LOG — top strategy details
# ==============================================================================
ranked = sorted(STRATEGIES.keys(), key=lambda sk: all_stats[sk]['eqw'], reverse=True)
top_strat = ranked[0]
top_trades = strategy_trades[top_strat]
top_s = all_stats[top_strat]

if top_trades:
    print(f"\n\n{'='*W}")
    print(f"  TRADE LOG — {STRATEGIES[top_strat]['label'].upper()}"
          f" ({top_s['n']} trades, {top_s['avg_pnl']:+.2%} avg, EqW {top_s['eqw']:+.2%})")
    print(f"{'='*W}")

    tdf = pd.DataFrame(top_trades).sort_values(['Date', 'Ticker'])
    print(f"\n  {'Date':<12} {'Ticker':<7} {'Source':<7} {'H':>4} {'Mode':<10}"
          f" {'Prob':>5}/{'Thr':>5}  {'Entry':>9} {'Exit':>9} {'PnL':>8} {'Type':>4}")
    print(f"  {'-'*95}")
    for _, t in tdf.iterrows():
        print(f"  {t['Date']:<12} {t['Ticker']:<7} {t['Source']:<7} {t['Horizon']:>4}"
              f" {t['Mode']:<10} {t['Prob']:>5.2f}/{t['Thresh']:>5.2f}"
              f"  ${t['Entry']:>8.2f} ${t['Exit']:>8.2f} {t['PnL']:>+7.2%} {t['Outcome']:>4}")


# ==============================================================================
# CHARTS
# ==============================================================================
if HAS_MPL and any(strategy_trades[sk] for sk in STRATEGIES):
    fig, axes = plt.subplots(3, 2, figsize=(24, 24))
    fig.suptitle(f'SHORT Hybrid Strategy Comparison: L5-1 vs L5-2 vs Combined\n'
                 f'{BACKTEST_START} to {BACKTEST_END} | Slippage {2*SLIPPAGE_PCT:.2%} RT'
                 f' | F7 >= {F7_DEPLOY_THRESHOLD:.1%}',
                 fontsize=14, fontweight='bold')

    # ── 1. Strategy stats comparison (grouped bar) ──
    ax = axes[0, 0]
    strat_names = list(STRATEGIES.keys())
    x = np.arange(len(strat_names))
    metrics = ['avg_pnl', 'eqw']
    metric_labels = ['Avg PnL/Trade', 'Cumulative EqW']
    bar_w = 0.35
    for i, (metric, mlabel) in enumerate(zip(metrics, metric_labels)):
        vals = [all_stats[sk][metric] * 100 for sk in strat_names]
        colors = [STRATEGIES[sk]['color'] for sk in strat_names]
        if i == 0:
            bars = ax.bar(x - bar_w/2, vals, bar_w, color=colors, alpha=0.85,
                          edgecolor='black', linewidth=0.5, label=mlabel)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{v:+.2f}%', ha='center', fontsize=8, fontweight='bold')
        else:
            bars2 = ax.bar(x + bar_w/2, vals, bar_w, color=colors, alpha=0.5,
                           edgecolor='black', linewidth=0.5, hatch='//', label=mlabel)
            for bar, v in zip(bars2, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{v:+.1f}%', ha='center', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([STRATEGIES[sk]['label'] for sk in strat_names], fontsize=9)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylabel('PnL (%)')
    ax.set_title('Strategy Performance (Avg PnL + EqW)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='y')

    # ── 2. Equity curves ──
    ax = axes[0, 1]
    for strat_name, strat_cfg in STRATEGIES.items():
        trades = strategy_trades[strat_name]
        if not trades:
            continue
        tdf = pd.DataFrame(trades)
        daily_eqw = tdf.groupby('Date')['PnL'].mean().sort_index()
        cum_val = daily_eqw.cumsum() * 100
        dates_plot = pd.to_datetime(cum_val.index)
        s = all_stats[strat_name]
        ax.plot(dates_plot, cum_val.values, color=strat_cfg['color'],
                linewidth=2.5, marker=strat_cfg['marker'], markersize=4,
                linestyle=strat_cfg['ls'],
                label=f"{strat_cfg['label']} ({s['n']}t, {s['trades_per_day']:.1f}/day,"
                      f" EqW {s['eqw']:+.1%})")
    ax.axhline(0, color='black', linewidth=0.8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax.set_ylabel('Cumulative EqW PnL (%)')
    ax.set_title('Portfolio Equity Curves')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(alpha=0.3)

    # ── 3. Source distribution (stacked bar) ──
    ax = axes[1, 0]
    src_keys = list(SOURCES.keys())
    src_colors = {'L1_F5': '#1565C0', 'L1_F6': '#42A5F5', 'L1_F7': '#90CAF9',
                  'L2_F5': '#FF6F00', 'L2_F6': '#FFA726', 'L2_F7': '#FFCC80'}
    x = np.arange(len(STRATEGIES))
    bottom = np.zeros(len(STRATEGIES))
    for src_key in src_keys:
        vals = []
        for strat_name in STRATEGIES:
            n = sum(1 for v in strategy_assignments[strat_name].values() if v == src_key)
            vals.append(n)
        vals = np.array(vals, dtype=float)
        if vals.sum() > 0:
            ax.bar(x, vals, 0.6, bottom=bottom, label=src_key,
                   color=src_colors.get(src_key, '#999'), alpha=0.85,
                   edgecolor='black', linewidth=0.5)
            for i, v in enumerate(vals):
                if v > 0:
                    ax.text(x[i], bottom[i] + v/2, f'{int(v)}', ha='center',
                            va='center', fontsize=8, fontweight='bold')
            bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([STRATEGIES[sk]['label'] for sk in STRATEGIES], fontsize=9)
    ax.set_ylabel('Number of Tickers')
    ax.set_title('Ticker Assignment by Source')
    ax.legend(fontsize=7, loc='upper right', ncol=2)
    ax.grid(alpha=0.3, axis='y')

    # ── 4. Daily trade count ──
    ax = axes[1, 1]
    for strat_name, strat_cfg in STRATEGIES.items():
        trades = strategy_trades[strat_name]
        if not trades:
            continue
        tdf = pd.DataFrame(trades)
        daily_count = tdf.groupby('Date').size()
        dates_plot = pd.to_datetime(daily_count.index)
        ax.plot(dates_plot, daily_count.values, color=strat_cfg['color'],
                linewidth=2, marker=strat_cfg['marker'], markersize=5,
                label=f"{strat_cfg['label']} (avg {all_stats[strat_name]['trades_per_day']:.1f}/day)")
    ax.axhline(0, color='black', linewidth=0.8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax.set_ylabel('Trades per Day')
    ax.set_title('Daily Trade Count')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── 5. Per-ticker PnL heatmap (combined vs L5-1) ──
    ax = axes[2, 0]
    combined_tickers = sorted(strategy_assignments['COMBINED HYB'].keys())
    if combined_tickers:
        strat_list = list(STRATEGIES.keys())
        data = np.full((len(combined_tickers), len(strat_list)), np.nan)
        for i, tk in enumerate(combined_tickers):
            for j, sk in enumerate(strat_list):
                tk_trades = [t for t in strategy_trades[sk] if t['Ticker'] == tk]
                if tk_trades:
                    data[i, j] = np.mean([t['PnL'] for t in tk_trades]) * 100
        im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=-5, vmax=8)
        ax.set_xticks(range(len(strat_list)))
        ax.set_xticklabels([STRATEGIES[sk]['label'][:14] for sk in strat_list], fontsize=8)
        ax.set_yticks(range(len(combined_tickers)))
        ax.set_yticklabels(combined_tickers, fontsize=7)
        for i in range(len(combined_tickers)):
            for j in range(len(strat_list)):
                val = data[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f'{val:+.1f}', ha='center', va='center',
                            fontsize=6, fontweight='bold',
                            color='white' if abs(val) > 3.5 else 'black')
                else:
                    ax.text(j, i, '--', ha='center', va='center', fontsize=6, color='gray')
        plt.colorbar(im, ax=ax, label='Avg PnL (%)', shrink=0.8)
        ax.set_title('Per-Ticker Avg PnL by Strategy')

    # ── 6. Win rate + trade count scatter ──
    ax = axes[2, 1]
    for strat_name, strat_cfg in STRATEGIES.items():
        s = all_stats[strat_name]
        if s['n'] == 0:
            continue
        ax.scatter(s['n'], s['avg_pnl'] * 100, c=strat_cfg['color'],
                   marker=strat_cfg['marker'], s=200, edgecolors='black',
                   linewidths=1, zorder=3)
        ax.annotate(f"{strat_cfg['label']}\n{s['wr']:.0%} WR | {s['trades_per_day']:.1f}/day",
                    (s['n'], s['avg_pnl'] * 100),
                    fontsize=8, ha='left', va='bottom',
                    xytext=(8, 8), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=strat_cfg['color'],
                              alpha=0.15))
    ax.axhline(BASE_PNL_THRESHOLD * 100, color='green', linewidth=1,
               linestyle='--', alpha=0.5, label=f'{BASE_PNL_THRESHOLD:.1%} threshold')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Total Trades')
    ax.set_ylabel('Avg PnL per Trade (%)')
    ax.set_title('Trade Count vs Avg PnL (Efficiency Frontier)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    save_path = '/tmp/SHORT_hybrid_L5_1_vs_L5_2_combined.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n  Chart saved: {save_path}")
    if _IN_COLAB:
        plt.show()
    plt.close()


# ==============================================================================
# VERDICT
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  VERDICT")
print(f"{'='*W}")

print(f"\n  ── STRATEGY SUMMARY ──")
for strat_name in STRATEGIES:
    s = all_stats[strat_name]
    label = STRATEGIES[strat_name]['label']
    pf_str = f"PF {s['pf']:.2f}" if s['pf'] < 999 else 'PF inf'
    print(f"  {label:<26} {s['avg_pnl']:>+7.2%} avg | {s['n']:>3} trades"
          f" | {s['trades_per_day']:.1f}/day | {s['wr']:.0%} WR | {pf_str}"
          f" | EqW {s['eqw']:>+7.2%} | {s['n_tickers']} tickers")

# ── Best strategy ──
best_by_eqw = max(STRATEGIES.keys(), key=lambda sk: all_stats[sk]['eqw'])
best_by_avg = max(STRATEGIES.keys(), key=lambda sk: all_stats[sk]['avg_pnl'] if all_stats[sk]['n'] >= 5 else -999)

print(f"\n  ── RANKINGS ──")
print(f"  Best by EqW PnL:     {STRATEGIES[best_by_eqw]['label']}"
      f" ({all_stats[best_by_eqw]['eqw']:+.2%})")
print(f"  Best by Avg PnL:     {STRATEGIES[best_by_avg]['label']}"
      f" ({all_stats[best_by_avg]['avg_pnl']:+.2%})")

# ── Combined HYB analysis ──
comb_s = all_stats['COMBINED HYB']
ref_s = all_stats['L5-1 HYB']
l2_s = all_stats['L5-2 HYB']

if comb_s['n'] > 0 and ref_s['n'] > 0:
    print(f"\n  ── COMBINED vs DEPLOYED (L5-1 HYB) ──")
    print(f"  Deployed:  {ref_s['n_tickers']} tickers, {ref_s['n']} trades,"
          f" {ref_s['trades_per_day']:.1f}/day,"
          f" Avg {ref_s['avg_pnl']:+.2%}, EqW {ref_s['eqw']:+.2%}")
    print(f"  Combined:  {comb_s['n_tickers']} tickers, {comb_s['n']} trades,"
          f" {comb_s['trades_per_day']:.1f}/day,"
          f" Avg {comb_s['avg_pnl']:+.2%}, EqW {comb_s['eqw']:+.2%}")
    d_eqw = comb_s['eqw'] - ref_s['eqw']
    d_avg = comb_s['avg_pnl'] - ref_s['avg_pnl']
    d_n = comb_s['n'] - ref_s['n']
    d_tk = comb_s['n_tickers'] - ref_s['n_tickers']
    print(f"  Delta:     {d_tk:+d} tickers, {d_n:+d} trades,"
          f" Avg {d_avg:+.2%}, EqW {d_eqw:+.2%}")

    if d_eqw > 0.02 and d_avg > -0.005:
        print(f"\n  >>> COMBINED WINS: +{d_eqw:.2%} EqW uplift without quality degradation.")
        print(f"  >>> ACTION: Deploy COMBINED HYB — add L5-2 models to scoring pipeline.")
    elif d_avg > 0.005:
        print(f"\n  >>> COMBINED has better avg PnL ({d_avg:+.2%}) but verify on longer window.")
    elif l2_s['n'] > 0 and l2_s['eqw'] > ref_s['eqw']:
        print(f"\n  >>> L5-2 HYB alone beats deployed ({l2_s['eqw']:+.2%} vs {ref_s['eqw']:+.2%}).")
        print(f"  >>> Consider full L5-2 replacement or COMBINED deployment.")
    else:
        print(f"\n  >>> DEPLOYED L5-1 HYB HOLDS: no material improvement from adding L5-2.")

# ── All strategies vs deployed ──
print(f"\n  ── ALL STRATEGIES vs DEPLOYED ──")
for strat_name in STRATEGIES:
    if strat_name == 'L5-1 HYB':
        continue
    s = all_stats[strat_name]
    d_avg = s['avg_pnl'] - ref_s['avg_pnl'] if ref_s['n'] > 0 else 0
    d_eqw = s['eqw'] - ref_s['eqw']
    label = STRATEGIES[strat_name]['label']
    verdict = 'BETTER' if d_eqw > 0.01 else ('WORSE' if d_eqw < -0.01 else 'NEUTRAL')
    print(f"  {label:<26} ΔAvg {d_avg:>+7.2%} | ΔEqW {d_eqw:>+7.2%} | {verdict}")

# ── Assignment summary for COMBINED ──
print(f"\n  ── COMBINED HYB ASSIGNMENT ──")
combined_assign = strategy_assignments['COMBINED HYB']
fam_counts = defaultdict(int)
for tk, src in combined_assign.items():
    fam_counts[SOURCES[src]['family']] += 1
for fam in ['L5-1', 'L5-2']:
    print(f"  From {fam}: {fam_counts.get(fam, 0)} tickers")

src_counts = defaultdict(list)
for tk, src in sorted(combined_assign.items()):
    src_counts[src].append(tk)
for src_key in SOURCES:
    if src_key in src_counts:
        tks = src_counts[src_key]
        print(f"    {src_key}: {len(tks)} — {', '.join(sorted(tks))}")

print(f"\n{'='*W}")
