# %% ── L5-1 vs L5-3 Combination Strategy Comparison ───────────────────────────
#
# L5-3 only has fold 7 trained. To compare fairly across ALL 11 strategies,
# both families share L5-1 folds 5+6. Only fold 7 differs:
#
#   L5-1: L5-1 F5 + L5-1 F6 + L5-1 F7  (current baseline)
#   L5-3: L5-1 F5 + L5-1 F6 + L5-3 F7  (swap in improved F7)
#
# This isolates the L5-3 training hygiene improvement (grad clip,
# warmup+cosine, AdamW) on fold 7 while keeping folds 5+6 identical.
#
# Strategies:
#   Individual:
#     1. F5 Only          2. F6 Only          3. F7 All (>=1.8%)
#     4. F7 Filt (>=2.5%) 5. F7 Filt (>=3.0%)
#   Combinations:
#     6. Hybrid F5+F6+F7          7. Hybrid F5+F6 Only
#     8. Hybrid (F7>=2.5%)        9. Majority (2/3 vote)
#    10. Unanimous (all agree)   11. Recency Weighted
#
# LONG only. Equal-weight portfolio PnL.

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

# ── Model families ──
# Both share L5-1 F5+F6; only F7 differs
MODEL_FAMILIES = OrderedDict([
    ('L5-1', {'label': 'L5-1 F5+F6+F7 (Baseline)', 'color': '#1565C0'}),
    ('L5-3', {'label': 'L5-1 F5+F6 + L5-3 F7',     'color': '#FF6F00'}),
])

GCS_DATA_PATH = 'Algos/RA_MoE/long_L5_1_data'

SHARED_FOLD_SOURCES = OrderedDict([
    ('F5', {'fold_id': 5, 'bq_mv': 'spike_transformer_long_L5_1_fold5',
            'gcs_models': 'Algos/RA_MoE/long_L5_1_models'}),
    ('F6', {'fold_id': 6, 'bq_mv': 'spike_transformer_long_L5_1_fold6',
            'gcs_models': 'Algos/RA_MoE/long_L5_1_models'}),
])

FAMILY_F7_SOURCE = OrderedDict([
    ('L5-1', {'fold_id': 7, 'bq_mv': 'spike_transformer_long_L5_1_fold7',
              'gcs_models': 'Algos/RA_MoE/long_L5_1_models', 'tag': 'L5_1'}),
    ('L5-3', {'fold_id': 7, 'bq_mv': 'spike_transformer_long_L5_3_fold7',
              'gcs_models': 'Algos/RA_MoE/long_L5_3_models', 'tag': 'L5_3'}),
])

FOLD_IDS = [5, 6, 7]
FOLD_KEYS = ['F5', 'F6', 'F7']

# F7 filtering thresholds
F7_THRESHOLDS = [0.018, 0.025, 0.030, 0.035]

# Recency weights
RECENCY_WEIGHTS = {'F5': 0.20, 'F6': 0.30, 'F7': 0.50}

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

# ── Combination strategies ──
COMBO_STRATEGIES = OrderedDict([
    ('F5_ONLY',   {'label': 'F5 Only',           'marker': 's',  'group': 'Individual'}),
    ('F6_ONLY',   {'label': 'F6 Only',           'marker': 'D',  'group': 'Individual'}),
    ('F7_ALL',    {'label': 'F7 All (>=1.8%)',    'marker': '^',  'group': 'Individual'}),
    ('F7_25',     {'label': 'F7 Filt (>=2.5%)',   'marker': 'v',  'group': 'Individual'}),
    ('F7_30',     {'label': 'F7 Filt (>=3.0%)',   'marker': '<',  'group': 'Individual'}),
    ('HYB_ALL',   {'label': 'Hybrid F5+F6+F7',   'marker': 'o',  'group': 'Combination'}),
    ('HYB_F5F6',  {'label': 'Hybrid F5+F6',      'marker': '*',  'group': 'Combination'}),
    ('HYB_F7_25', {'label': 'Hybrid (F7>=2.5%)', 'marker': 'p',  'group': 'Combination'}),
    ('MAJORITY',  {'label': 'Majority (2/3)',     'marker': 'H',  'group': 'Combination'}),
    ('UNANIMOUS', {'label': 'Unanimous',          'marker': 'X',  'group': 'Combination'}),
    ('RECENCY',   {'label': 'Recency Weighted',   'marker': 'd',  'group': 'Combination'}),
])


# ==============================================================================
# MODEL ARCHITECTURE  (v1 SpikeTransformer — same for both families)
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
    if 'VIX_Close' in market_df.columns:
        rf['regime_vix_close'] = market_df['VIX_Close'].ffill().fillna(0)
    else:
        rf['regime_vix_close'] = 0.0
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
# MODEL LOADING
# ==============================================================================
def _ensure_file(gcs_path, local_path):
    if os.path.exists(local_path):
        return local_path
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    gcs_client.bucket(BUCKET_NAME).blob(gcs_path).download_to_filename(local_path)
    return local_path


def load_shared_meta():
    local_meta = '/tmp/bt_combo_meta.npz'
    meta_path = _ensure_file(f'{GCS_DATA_PATH}/meta.npz', local_meta)
    meta = np.load(meta_path, allow_pickle=True)
    return meta['scaling_stats'].item(), meta['regime_stats'].item()


def load_fold_model(gcs_models, tag, fold_key, fold_id):
    model = SpikeTransformer(name=f'spike_transformer_{tag}_{fold_key.lower()}')
    dummy_seq = tf.zeros((1, TIME_STEPS, N_ASSET_F))
    dummy_reg = tf.zeros((1, N_REGIME_F))
    model({'seq_input': dummy_seq, 'regime_input': dummy_reg})

    local_dir = f'/tmp/bt_{tag}_combo_models/{fold_key}/fold_{fold_id}'
    weights_path = os.path.join(local_dir, 'model.weights.h5')
    if not os.path.exists(weights_path):
        os.makedirs(local_dir, exist_ok=True)
        prefix = f'{gcs_models}/fold_{fold_id}'
        bucket = gcs_client.bucket(BUCKET_NAME)
        for blob in bucket.list_blobs(prefix=prefix):
            fname = blob.name.split('/')[-1]
            blob.download_to_filename(os.path.join(local_dir, fname))
    model.load_weights(weights_path)
    print(f"    {tag} {fold_key} (fold {fold_id}): {model.count_params():,} params")
    return model


# ==============================================================================
# BQ CONFIG LOADING
# ==============================================================================
def load_fold_configs(bq_model_version):
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
# LONG TRADE SIMULATION
# ==============================================================================
def simulate_long_trade(df_feat, entry_idx, cfg):
    entry_date_pst = df_feat.index[entry_idx].tz_convert(PST).date()
    close = float(df_feat['Close'].iloc[entry_idx])

    if cfg['mode'] == 'limit_entry':
        entry_price = close * (1 - cfg.get('dip_pct', 0))
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
            if float(df_feat['Low'].iloc[k]) <= entry_price:
                fill_idx = k
                break
        if fill_idx is None:
            return None
        start_idx = fill_idx
    else:
        entry_price = close
        start_idx = entry_idx

    tp_price = entry_price * (1 + cfg['tp_pct'])
    sl_price = entry_price * (1 - cfg['sl_pct'])
    grace_minutes = cfg.get('grace_minutes', 0) if cfg['mode'] == 'grace' else 0
    entry_time_min = (df_feat.index[start_idx].tz_convert(PST).hour * 60 +
                      df_feat.index[start_idx].tz_convert(PST).minute)

    for k in range(start_idx + 1, len(df_feat)):
        bar_pst = df_feat.index[k].tz_convert(PST)
        if bar_pst.date() != entry_date_pst:
            exit_price = float(df_feat['Close'].iloc[k - 1])
            pnl = (exit_price - entry_price) / entry_price
            return {'pnl': pnl, 'entry': entry_price, 'exit': exit_price, 'outcome': 'EOD'}

        bar_min = bar_pst.hour * 60 + bar_pst.minute
        high_k = float(df_feat['High'].iloc[k])
        low_k = float(df_feat['Low'].iloc[k])

        elapsed = bar_min - entry_time_min
        sl_active = elapsed >= grace_minutes

        tp_hit = high_k >= tp_price
        sl_hit = sl_active and low_k <= sl_price

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
            pnl = (exit_price - entry_price) / entry_price
            return {'pnl': pnl, 'entry': entry_price, 'exit': exit_price, 'outcome': 'EOD'}

    exit_price = float(df_feat['Close'].iloc[-1])
    pnl = (exit_price - entry_price) / entry_price
    return {'pnl': pnl, 'entry': entry_price, 'exit': exit_price, 'outcome': 'EOD'}


# ==============================================================================
# TRADING DAYS
# ==============================================================================
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
# STRATEGY HELPERS
# ==============================================================================
def _make_trade_record(day, ticker, fold, prob, cfg, trade_result):
    trade_result['pnl'] -= 2 * SLIPPAGE_PCT
    return {
        'Date': day.strftime('%Y-%m-%d'),
        'Ticker': ticker,
        'Fold': fold,
        'Prob': prob,
        'Thresh': cfg['prob_thresh'],
        'Entry': trade_result['entry'],
        'Exit': trade_result['exit'],
        'PnL': trade_result['pnl'],
        'Outcome': trade_result['outcome'],
        'Mode': cfg['mode'],
        'Horizon': cfg.get('horizon_label', '30m'),
        'TP_pct': cfg['tp_pct'],
        'SL_pct': cfg['sl_pct'],
        'val_b': cfg['val_b_avg_pnl'],
    }


def _try_trade(ticker, day_str, fold, score_cache, fold_cfgs,
               ticker_features, cfg_override=None):
    key = (fold, ticker, day_str)
    if key not in score_cache:
        return None
    sc = score_cache[key]
    cfg = cfg_override if cfg_override else fold_cfgs[fold][ticker]
    if sc['prob'] < cfg['prob_thresh']:
        return None
    if ticker not in ticker_features:
        return None
    entry_idx = sc['bar_idx'] + (ENTRY_MINUTE - SCORE_MINUTE)
    trade = simulate_long_trade(ticker_features[ticker], entry_idx, cfg)
    if trade is None:
        return None
    day_date = datetime.strptime(day_str, '%Y-%m-%d').date()
    return _make_trade_record(day_date, ticker, fold, sc['prob'], cfg, trade)


def best_val_b_fold(ticker, fold_cfgs, allowed_folds=None):
    best_f, best_v, best_c = None, -999, None
    for f in (allowed_folds or fold_cfgs.keys()):
        if ticker in fold_cfgs.get(f, {}):
            v = fold_cfgs[f][ticker]['val_b_avg_pnl']
            if v > best_v:
                best_f, best_v, best_c = f, v, fold_cfgs[f][ticker]
    return best_f, best_c


# ==============================================================================
# APPLY COMBINATION STRATEGIES
# ==============================================================================
def apply_all_strategies(score_cache, fold_cfgs, ticker_features, trading_days):
    day_strs = [d.strftime('%Y-%m-%d') for d in trading_days]

    ticker_folds = defaultdict(list)
    for f in FOLD_KEYS:
        for t in fold_cfgs.get(f, {}):
            ticker_folds[t].append(f)
    multi_fold_tickers = {t: folds for t, folds in ticker_folds.items() if len(folds) >= 2}

    def _hybrid_assignment(allowed_folds):
        assignment = {}
        for t in set().union(*(set(fold_cfgs.get(f, {}).keys()) for f in allowed_folds)):
            bf, bc = best_val_b_fold(t, fold_cfgs, allowed_folds)
            if bf:
                assignment[t] = bf
        return assignment

    hyb_all = _hybrid_assignment(FOLD_KEYS)
    hyb_f5f6 = _hybrid_assignment(['F5', 'F6'])

    f7_25 = {t: c for t, c in fold_cfgs.get('F7', {}).items() if c['val_b_avg_pnl'] >= 0.025}
    f7_30 = {t: c for t, c in fold_cfgs.get('F7', {}).items() if c['val_b_avg_pnl'] >= 0.030}

    f7_filt_cfgs = {'F7_25': f7_25, 'F7_30': f7_30}

    hyb_f7_25_cfgs = {'F5': fold_cfgs.get('F5', {}),
                      'F6': fold_cfgs.get('F6', {}),
                      'F7': f7_25}
    hyb_f7_25_assign = {}
    for t in set().union(*(set(c.keys()) for c in hyb_f7_25_cfgs.values())):
        bf = None
        best_v = -999
        for f in FOLD_KEYS:
            if t in hyb_f7_25_cfgs.get(f, {}):
                v = hyb_f7_25_cfgs[f][t]['val_b_avg_pnl']
                if v > best_v:
                    best_v = v
                    bf = f
        if bf:
            hyb_f7_25_assign[t] = bf

    results = {}

    for skey in COMBO_STRATEGIES:
        trades = []

        for day_str in day_strs:
            if skey == 'F5_ONLY':
                for t in sorted(fold_cfgs.get('F5', {}).keys()):
                    rec = _try_trade(t, day_str, 'F5', score_cache, fold_cfgs, ticker_features)
                    if rec:
                        trades.append(rec)

            elif skey == 'F6_ONLY':
                for t in sorted(fold_cfgs.get('F6', {}).keys()):
                    rec = _try_trade(t, day_str, 'F6', score_cache, fold_cfgs, ticker_features)
                    if rec:
                        trades.append(rec)

            elif skey == 'F7_ALL':
                for t in sorted(fold_cfgs.get('F7', {}).keys()):
                    rec = _try_trade(t, day_str, 'F7', score_cache, fold_cfgs, ticker_features)
                    if rec:
                        trades.append(rec)

            elif skey in ('F7_25', 'F7_30'):
                filtered = f7_filt_cfgs[skey]
                for t in sorted(filtered.keys()):
                    rec = _try_trade(t, day_str, 'F7', score_cache, fold_cfgs, ticker_features)
                    if rec and t in filtered:
                        trades.append(rec)

            elif skey == 'HYB_ALL':
                for t, f in sorted(hyb_all.items()):
                    rec = _try_trade(t, day_str, f, score_cache, fold_cfgs, ticker_features)
                    if rec:
                        trades.append(rec)

            elif skey == 'HYB_F5F6':
                for t, f in sorted(hyb_f5f6.items()):
                    rec = _try_trade(t, day_str, f, score_cache, fold_cfgs, ticker_features)
                    if rec:
                        trades.append(rec)

            elif skey == 'HYB_F7_25':
                for t, f in sorted(hyb_f7_25_assign.items()):
                    cfg_src = hyb_f7_25_cfgs.get(f, {}).get(t)
                    if cfg_src is None:
                        continue
                    rec = _try_trade(t, day_str, f, score_cache, fold_cfgs,
                                     ticker_features, cfg_override=cfg_src)
                    if rec:
                        trades.append(rec)

            elif skey == 'MAJORITY':
                for t, folds in sorted(multi_fold_tickers.items()):
                    votes = []
                    for f in folds:
                        key = (f, t, day_str)
                        if key in score_cache:
                            sc = score_cache[key]
                            cfg = fold_cfgs[f][t]
                            if sc['prob'] >= cfg['prob_thresh']:
                                votes.append((f, sc['prob'], cfg))
                    if len(votes) >= 2:
                        best_vote = max(votes, key=lambda x: x[2]['val_b_avg_pnl'])
                        f_use, prob_use, cfg_use = best_vote
                        rec = _try_trade(t, day_str, f_use, score_cache,
                                         fold_cfgs, ticker_features, cfg_override=cfg_use)
                        if rec:
                            rec['Fold'] = '+'.join(v[0] for v in votes)
                            trades.append(rec)

            elif skey == 'UNANIMOUS':
                for t, folds in sorted(multi_fold_tickers.items()):
                    all_fire = True
                    votes = []
                    for f in folds:
                        key = (f, t, day_str)
                        if key not in score_cache:
                            all_fire = False
                            break
                        sc = score_cache[key]
                        cfg = fold_cfgs[f][t]
                        if sc['prob'] >= cfg['prob_thresh']:
                            votes.append((f, sc['prob'], cfg))
                        else:
                            all_fire = False
                            break
                    if all_fire and len(votes) == len(folds):
                        best_vote = max(votes, key=lambda x: x[2]['val_b_avg_pnl'])
                        f_use, prob_use, cfg_use = best_vote
                        rec = _try_trade(t, day_str, f_use, score_cache,
                                         fold_cfgs, ticker_features, cfg_override=cfg_use)
                        if rec:
                            rec['Fold'] = '+'.join(v[0] for v in votes)
                            trades.append(rec)

            elif skey == 'RECENCY':
                for t, folds in sorted(multi_fold_tickers.items()):
                    probs_by_fold = {}
                    for f in folds:
                        key = (f, t, day_str)
                        if key in score_cache:
                            sc = score_cache[key]
                            probs_by_fold[f] = sc['prob']
                    if len(probs_by_fold) < 2:
                        continue
                    total_w = sum(RECENCY_WEIGHTS[f] for f in probs_by_fold)
                    weighted_prob = sum(RECENCY_WEIGHTS[f] * probs_by_fold[f]
                                       for f in probs_by_fold) / total_w
                    bf, best_cfg = best_val_b_fold(t, fold_cfgs, list(probs_by_fold.keys()))
                    if bf is None or best_cfg is None:
                        continue
                    if weighted_prob >= best_cfg['prob_thresh']:
                        key = (bf, t, day_str)
                        if key in score_cache:
                            sc = score_cache[key]
                            entry_idx = sc['bar_idx'] + (ENTRY_MINUTE - SCORE_MINUTE)
                            if t not in ticker_features:
                                continue
                            trade = simulate_long_trade(ticker_features[t], entry_idx, best_cfg)
                            if trade is not None:
                                day_date = datetime.strptime(day_str, '%Y-%m-%d').date()
                                rec = _make_trade_record(day_date, t, bf, weighted_prob,
                                                         best_cfg, trade)
                                rec['Fold'] = 'RW(' + '+'.join(sorted(probs_by_fold.keys())) + ')'
                                trades.append(rec)

        results[skey] = trades

    return results, {
        'hyb_all': hyb_all, 'hyb_f5f6': hyb_f5f6,
        'hyb_f7_25': hyb_f7_25_assign,
        'multi_fold_tickers': multi_fold_tickers,
        'f7_25_tickers': f7_25, 'f7_30_tickers': f7_30,
    }


# ==============================================================================
# STATS COMPUTATION
# ==============================================================================
def compute_strategy_stats(trades_list):
    if not trades_list:
        return {'n': 0, 'avg_pnl': 0, 'wr': 0, 'pf': 0,
                'avg_win': 0, 'avg_loss': 0, 'eqw': 0, 'n_tickers': 0,
                'nw': 0, 'nl': 0, 'sharpe': 0}
    df = pd.DataFrame(trades_list)
    n = len(df)
    nw = (df['PnL'] > 0).sum()
    nl = n - nw
    avg_pnl = df['PnL'].mean()
    std_pnl = df['PnL'].std()
    wr = nw / n if n > 0 else 0
    avg_win = df.loc[df['PnL'] > 0, 'PnL'].mean() if nw > 0 else 0
    avg_loss = df.loc[df['PnL'] <= 0, 'PnL'].mean() if nl > 0 else 0
    pf = abs(avg_win * nw / (avg_loss * nl)) if nl > 0 and avg_loss != 0 else float('inf')
    eqw = df.groupby('Date')['PnL'].mean().sum()
    n_tickers = df['Ticker'].nunique()
    sharpe = (avg_pnl / std_pnl * np.sqrt(252)) if std_pnl > 0 else 0
    return {'n': n, 'avg_pnl': avg_pnl, 'wr': wr, 'pf': pf,
            'avg_win': avg_win, 'avg_loss': avg_loss, 'eqw': eqw,
            'n_tickers': n_tickers, 'nw': nw, 'nl': nl, 'sharpe': sharpe}


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
np.random.seed(42)

print(f"\n{'='*W}")
print(f"  L5-1 vs L5-3: COMBINATION STRATEGY COMPARISON")
print(f"  L5-3 has only F7 — shared L5-1 F5+F6, swap F7 only")
print(f"  Period: {BACKTEST_START} -> {BACKTEST_END}")
print(f"  Slippage: {SLIPPAGE_PCT:.2%} each way ({2*SLIPPAGE_PCT:.2%} RT)")
print(f"  Base threshold: {BASE_PNL_THRESHOLD:.1%}")
print(f"{'='*W}")
for fam_key, fam_cfg in MODEL_FAMILIES.items():
    f7_src = FAMILY_F7_SOURCE[fam_key]
    print(f"  {fam_key}: {fam_cfg['label']}")
    print(f"    F5+F6: L5-1 (shared)")
    print(f"    F7 BQ: {f7_src['bq_mv']}")

# ── 1. Load BQ configs ──
print(f"\n{'='*W}")
print(f"  PHASE 1: LOAD BQ CONFIGS")
print(f"{'='*W}")

# Shared F5+F6 configs (identical for both families)
shared_fold_cfgs = {}
all_tickers_union = set()

print(f"\n  Shared folds (L5-1 F5+F6):")
for fold_key, src in SHARED_FOLD_SOURCES.items():
    cfgs = load_fold_configs(src['bq_mv'])
    shared_fold_cfgs[fold_key] = cfgs
    tickers = sorted(cfgs.keys())
    all_tickers_union.update(tickers)
    if tickers:
        print(f"    {fold_key}: {len(tickers)} tickers - {', '.join(tickers)}")
    else:
        print(f"    {fold_key}: 0 tickers")

# Per-family F7 configs
family_f7_cfgs = {}
for fam_key in MODEL_FAMILIES:
    f7_src = FAMILY_F7_SOURCE[fam_key]
    cfgs = load_fold_configs(f7_src['bq_mv'])
    family_f7_cfgs[fam_key] = cfgs
    tickers = sorted(cfgs.keys())
    all_tickers_union.update(tickers)
    print(f"\n  {fam_key} F7 ({f7_src['bq_mv']}):")
    if tickers:
        print(f"    F7: {len(tickers)} tickers - {', '.join(tickers)}")
    else:
        print(f"    F7: 0 tickers")

# Assemble per-family fold configs (shared F5+F6 + family-specific F7)
family_fold_cfgs = {}
for fam_key in MODEL_FAMILIES:
    family_fold_cfgs[fam_key] = {
        'F5': shared_fold_cfgs['F5'],
        'F6': shared_fold_cfgs['F6'],
        'F7': family_f7_cfgs[fam_key],
    }

print(f"\n  Total unique tickers across all folds: {len(all_tickers_union)}")

# ── 2. Build market context + regime (shared, once) ──
print(f"\n{'='*W}")
print(f"  PHASE 2: MARKET CONTEXT + REGIME FEATURES")
print(f"{'='*W}")

print(f"  Building market context (incl. sector proxies)...")
market_df = build_market_context()
print(f"  Computing regime features (22 cols)...")
regime_df = compute_regime_features(market_df)

# ── 3. Trading days ──
trading_days = get_trading_days(BACKTEST_START, BACKTEST_END)
print(f"  Trading days: {len(trading_days)} ({trading_days[0]} -> {trading_days[-1]})")

# ── 4. Shared meta ──
print(f"\n  Loading shared meta.npz (scaling + regime stats)...")
scaling_stats, regime_stats = load_shared_meta()

# ── 5. Engineer features for ALL tickers (once) ──
print(f"\n{'='*W}")
print(f"  PHASE 3: FEATURE ENGINEERING ({len(all_tickers_union)} tickers)")
print(f"{'='*W}")

ticker_features = {}
for ticker in sorted(all_tickers_union):
    df_feat = engineer_asset(ticker, market_df)
    if df_feat is not None:
        ticker_features[ticker] = df_feat
        print(f"    {ticker}: {len(df_feat):,} bars")
    else:
        print(f"    {ticker}: FAILED")

# ── 6. Score shared folds (F5+F6) once, then per-family F7 ──
print(f"\n{'='*W}")
print(f"  PHASE 4: SCORE + BACKTEST")
print(f"{'='*W}")

# Score shared folds once (same L5-1 model for both families)
print(f"\n  Scoring shared folds (F5+F6 — identical for both families)...")
shared_score_cache = {}

for fold_key, src in SHARED_FOLD_SOURCES.items():
    f_cfgs = shared_fold_cfgs[fold_key]
    if not f_cfgs:
        print(f"    {fold_key}: skipped (no configs)")
        continue
    model = load_fold_model(src['gcs_models'], 'L5_1', fold_key, src['fold_id'])
    f_count = 0
    for ticker in sorted(f_cfgs.keys()):
        if ticker not in ticker_features:
            continue
        cfg = f_cfgs[ticker]
        for day in trading_days:
            result = score_ticker_day(
                model, scaling_stats, regime_stats,
                ticker, cfg, ticker_features[ticker], regime_df, day)
            if result is not None:
                shared_score_cache[(fold_key, ticker, day.strftime('%Y-%m-%d'))] = result
                f_count += 1
    print(f"    {fold_key}: {f_count} valid scores")
    del model
    keras.backend.clear_session()
    gc.collect()

print(f"  Shared F5+F6 scores: {len(shared_score_cache)}")

# Per-family: score F7 with family-specific model, combine, apply strategies
family_results = {}
family_all_stats = {}
family_strategy_meta = {}

for fam_key, fam_cfg in MODEL_FAMILIES.items():
    print(f"\n  {'─'*W}")
    print(f"  {fam_cfg['label']}: Scoring F7 + applying strategies")
    print(f"  {'─'*W}")

    fold_cfgs = family_fold_cfgs[fam_key]
    np.random.seed(42)

    # Start from shared F5+F6 cache
    score_cache = dict(shared_score_cache)

    # Score F7 with this family's model
    f7_src = FAMILY_F7_SOURCE[fam_key]
    f7_cfgs = family_f7_cfgs[fam_key]
    if f7_cfgs:
        model_f7 = load_fold_model(f7_src['gcs_models'], f7_src['tag'],
                                    'F7', f7_src['fold_id'])
        f7_count = 0
        for ticker in sorted(f7_cfgs.keys()):
            if ticker not in ticker_features:
                continue
            cfg = f7_cfgs[ticker]
            for day in trading_days:
                result = score_ticker_day(
                    model_f7, scaling_stats, regime_stats,
                    ticker, cfg, ticker_features[ticker], regime_df, day)
                if result is not None:
                    score_cache[('F7', ticker, day.strftime('%Y-%m-%d'))] = result
                    f7_count += 1
        print(f"    F7: {f7_count} valid scores")
        del model_f7
        keras.backend.clear_session()
        gc.collect()
    else:
        print(f"    F7: skipped (no configs)")

    print(f"    Total scores (shared F5+F6 + F7): {len(score_cache)}")

    # Apply all combination strategies
    print(f"\n  Applying {len(COMBO_STRATEGIES)} combination strategies...")
    strat_trades, strat_meta = apply_all_strategies(
        score_cache, fold_cfgs, ticker_features, trading_days)

    family_results[fam_key] = strat_trades
    family_strategy_meta[fam_key] = strat_meta

    fam_stats = {}
    for skey in COMBO_STRATEGIES:
        fam_stats[skey] = compute_strategy_stats(strat_trades[skey])
    family_all_stats[fam_key] = fam_stats

    # Print per-family summary
    print(f"\n  {fam_cfg['label']} — Strategy Results:")
    print(f"  {'Strategy':<24} {'Tk':>3} {'N':>4} {'AvgPnL':>8} {'WR':>5}"
          f" {'PF':>6} {'EqW':>8} {'Sharpe':>7}")
    print(f"  {'-'*68}")
    prev_group = None
    for skey, scfg in COMBO_STRATEGIES.items():
        s = fam_stats[skey]
        group = scfg['group']
        if group != prev_group and prev_group is not None:
            print(f"  {'':.<68}")
        prev_group = group
        pf_str = f"{s['pf']:.2f}" if s['pf'] < 999 else 'inf'
        print(f"    {scfg['label']:<22} {s['n_tickers']:>3} {s['n']:>4} {s['avg_pnl']:>+7.2%}"
              f" {s['wr']:>4.0%} {pf_str:>6} {s['eqw']:>+7.2%} {s['sharpe']:>7.1f}")

    del score_cache
    gc.collect()


# ==============================================================================
# HEAD-TO-HEAD COMPARISON
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  HEAD-TO-HEAD COMPARISON: L5-1 F7 vs L5-3 F7 (shared F5+F6)")
print(f"  Period: {BACKTEST_START} -> {BACKTEST_END}")
print(f"{'='*W}")

# ── Side-by-side table ──
fam_keys = list(MODEL_FAMILIES.keys())
f1, f2 = fam_keys[0], fam_keys[1]

print(f"\n  {'Strategy':<22} │{'L5-1 Avg':>9} {'N':>4} {'WR':>5} {'EqW':>8}"
      f" │{'L5-3 Avg':>9} {'N':>4} {'WR':>5} {'EqW':>8}"
      f" │{'Δ Avg':>7} {'ΔEqW':>7} {'Win?':>5}")
print(f"  {'─'*22}─┼{'─'*28}─┼{'─'*28}─┼{'─'*21}")

prev_group = None
for skey, scfg in COMBO_STRATEGIES.items():
    s1 = family_all_stats[f1][skey]
    s2 = family_all_stats[f2][skey]
    group = scfg['group']
    if group != prev_group and prev_group is not None:
        print(f"  {'─'*22}─┼{'─'*28}─┼{'─'*28}─┼{'─'*21}")
    prev_group = group

    delta_avg = s2['avg_pnl'] - s1['avg_pnl'] if (s1['n'] > 0 and s2['n'] > 0) else 0
    delta_eqw = s2['eqw'] - s1['eqw']
    winner = 'L5-3' if delta_avg > 0.001 else ('L5-1' if delta_avg < -0.001 else 'TIE')
    if s1['n'] == 0 and s2['n'] == 0:
        winner = '--'
    elif s1['n'] == 0:
        winner = 'L5-3*'
    elif s2['n'] == 0:
        winner = 'L5-1*'

    print(f"  {scfg['label']:<22} │"
          f"{s1['avg_pnl']:>+8.2%} {s1['n']:>4} {s1['wr']:>4.0%} {s1['eqw']:>+7.2%} │"
          f"{s2['avg_pnl']:>+8.2%} {s2['n']:>4} {s2['wr']:>4.0%} {s2['eqw']:>+7.2%} │"
          f"{delta_avg:>+6.2%} {delta_eqw:>+6.2%} {winner:>5}")

# ── Tally ──
l1_wins = 0; l3_wins = 0; ties = 0
for skey in COMBO_STRATEGIES:
    s1 = family_all_stats[f1][skey]
    s2 = family_all_stats[f2][skey]
    if s1['n'] < 5 and s2['n'] < 5:
        continue
    d = s2['avg_pnl'] - s1['avg_pnl']
    if d > 0.001:
        l3_wins += 1
    elif d < -0.001:
        l1_wins += 1
    else:
        ties += 1

print(f"\n  TALLY (strategies with >= 5 trades):")
print(f"    L5-1 wins: {l1_wins}   L5-3 wins: {l3_wins}   Ties: {ties}")

# ── Rankings ──
print(f"\n  RANKED BY AVG PnL (all strategies, both families):")
combined = []
for fam_key in MODEL_FAMILIES:
    for skey in COMBO_STRATEGIES:
        s = family_all_stats[fam_key][skey]
        if s['n'] >= 5:
            combined.append((fam_key, skey, s))
combined.sort(key=lambda x: -x[2]['avg_pnl'])
for rank, (fam_key, skey, s) in enumerate(combined[:15], 1):
    label = COMBO_STRATEGIES[skey]['label']
    print(f"    #{rank:<2} {fam_key} {label:<22} {s['avg_pnl']:>+7.2%} avg | "
          f"{s['n']:>3}t | {s['wr']:.0%} WR | EqW {s['eqw']:>+7.2%} | Sharpe {s['sharpe']:.1f}")

print(f"\n  RANKED BY EqW PORTFOLIO PnL:")
combined_eqw = sorted(combined, key=lambda x: -x[2]['eqw'])
for rank, (fam_key, skey, s) in enumerate(combined_eqw[:15], 1):
    label = COMBO_STRATEGIES[skey]['label']
    print(f"    #{rank:<2} {fam_key} {label:<22} EqW {s['eqw']:>+7.2%} | "
          f"{s['avg_pnl']:>+7.2%} avg | {s['n']:>3}t")

# ── Per-ticker: which family wins more often ──
print(f"\n  PER-TICKER WINNER (Hybrid F5+F6+F7 strategy):")
skey_compare = 'HYB_ALL'
all_tk = set()
tk_pnl = {}
for fam_key in MODEL_FAMILIES:
    trades = family_results[fam_key][skey_compare]
    for t in trades:
        all_tk.add(t['Ticker'])
        tk_pnl.setdefault(t['Ticker'], {}).setdefault(fam_key, []).append(t['PnL'])

if all_tk:
    print(f"  {'Ticker':<7} {'L5-1 Avg':>9} {'L5-1 N':>6} {'L5-3 Avg':>9} {'L5-3 N':>6} {'Winner':>7}")
    print(f"  {'-'*50}")
    tk_l1_wins = 0; tk_l3_wins = 0
    for tk in sorted(all_tk):
        pnls_1 = tk_pnl.get(tk, {}).get(f1, [])
        pnls_3 = tk_pnl.get(tk, {}).get(f2, [])
        avg_1 = np.mean(pnls_1) if pnls_1 else float('nan')
        avg_3 = np.mean(pnls_3) if pnls_3 else float('nan')
        if pnls_1 and pnls_3:
            w = f2 if avg_3 > avg_1 else f1
            if avg_3 > avg_1:
                tk_l3_wins += 1
            else:
                tk_l1_wins += 1
        elif pnls_3:
            w = f'{f2}*'
            tk_l3_wins += 1
        elif pnls_1:
            w = f'{f1}*'
            tk_l1_wins += 1
        else:
            w = '--'
        a1_str = f"{avg_1:>+8.2%}" if pnls_1 else f"{'---':>8}"
        a3_str = f"{avg_3:>+8.2%}" if pnls_3 else f"{'---':>8}"
        print(f"  {tk:<7} {a1_str} {len(pnls_1):>6} {a3_str} {len(pnls_3):>6} {w:>7}")
    print(f"  Ticker tally: L5-1={tk_l1_wins}  L5-3={tk_l3_wins}")


# ==============================================================================
# CHARTS
# ==============================================================================
if HAS_MPL:
    fig, axes = plt.subplots(3, 2, figsize=(24, 24))
    fig.suptitle(f'L5-1 vs L5-3 F7 Swap: Combination Strategy Comparison\n'
                 f'Shared F5+F6, only F7 differs | {BACKTEST_START} to {BACKTEST_END}',
                 fontsize=14, fontweight='bold', y=1.01)

    FAM_COLORS = {f1: '#1565C0', f2: '#FF6F00'}
    FAM_HATCHES = {f1: '', f2: '//'}

    skeys_with_trades = [sk for sk in COMBO_STRATEGIES
                         if family_all_stats[f1][sk]['n'] >= 1
                         or family_all_stats[f2][sk]['n'] >= 1]
    labels = [COMBO_STRATEGIES[sk]['label'] for sk in skeys_with_trades]
    y_pos = np.arange(len(labels))
    bar_h = 0.35

    # 1. Avg PnL comparison (horizontal bars, paired)
    ax = axes[0, 0]
    for i, fam_key in enumerate(MODEL_FAMILIES):
        vals = [family_all_stats[fam_key][sk]['avg_pnl'] * 100 for sk in skeys_with_trades]
        ax.barh(y_pos + i * bar_h, vals, bar_h, label=MODEL_FAMILIES[fam_key]['label'],
                color=FAM_COLORS[fam_key], alpha=0.85, edgecolor='black', linewidth=0.5,
                hatch=FAM_HATCHES[fam_key])
    ax.set_yticks(y_pos + bar_h / 2)
    ax.set_yticklabels(labels, fontsize=7)
    ax.axvline(0, color='black', linewidth=1)
    ax.axvline(BASE_PNL_THRESHOLD * 100, color='green', linewidth=1,
               linestyle='--', alpha=0.6, label=f'>= {BASE_PNL_THRESHOLD:.1%}')
    ax.set_xlabel('Avg PnL per Trade (%)')
    ax.set_title('Avg PnL per Trade: L5-1 vs L5-3')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='x')
    ax.invert_yaxis()

    # 2. EqW PnL comparison
    ax = axes[0, 1]
    for i, fam_key in enumerate(MODEL_FAMILIES):
        vals = [family_all_stats[fam_key][sk]['eqw'] * 100 for sk in skeys_with_trades]
        ax.barh(y_pos + i * bar_h, vals, bar_h, label=MODEL_FAMILIES[fam_key]['label'],
                color=FAM_COLORS[fam_key], alpha=0.85, edgecolor='black', linewidth=0.5,
                hatch=FAM_HATCHES[fam_key])
    ax.set_yticks(y_pos + bar_h / 2)
    ax.set_yticklabels(labels, fontsize=7)
    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlabel('Cumulative EqW Portfolio PnL (%)')
    ax.set_title('EqW Portfolio PnL: L5-1 vs L5-3')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='x')
    ax.invert_yaxis()

    # 3. Delta chart (L5-3 improvement over L5-1)
    ax = axes[1, 0]
    deltas = []
    for sk in skeys_with_trades:
        s1 = family_all_stats[f1][sk]
        s2 = family_all_stats[f2][sk]
        d = (s2['avg_pnl'] - s1['avg_pnl']) * 100 if (s1['n'] > 0 or s2['n'] > 0) else 0
        deltas.append(d)
    colors_delta = ['#2E7D32' if d > 0 else '#D32F2F' for d in deltas]
    ax.barh(y_pos, deltas, 0.6, color=colors_delta, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.axvline(0, color='black', linewidth=1)
    for i, d in enumerate(deltas):
        ax.text(d + (0.05 if d >= 0 else -0.05), i, f'{d:+.2f}%',
                va='center', ha='left' if d >= 0 else 'right', fontsize=7)
    ax.set_xlabel('L5-3 minus L5-1 Avg PnL (%)')
    ax.set_title('L5-3 Improvement Over L5-1 (green = L5-3 better)')
    ax.grid(alpha=0.3, axis='x')
    ax.invert_yaxis()

    # 4. Equity curves — top 3 strategies, both families
    ax = axes[1, 1]
    top_strats = sorted(skeys_with_trades,
                        key=lambda sk: max(family_all_stats[f1][sk]['eqw'],
                                           family_all_stats[f2][sk]['eqw']),
                        reverse=True)[:4]
    strat_colors = ['#D32F2F', '#2E7D32', '#6A1B9A', '#00ACC1']
    for si, skey in enumerate(top_strats):
        for fam_key in MODEL_FAMILIES:
            trades = family_results[fam_key][skey]
            if not trades:
                continue
            tdf = pd.DataFrame(trades)
            daily_eqw = tdf.groupby('Date')['PnL'].mean().sort_index()
            cum_val = daily_eqw.cumsum() * 100
            dates_plot = pd.to_datetime(cum_val.index)
            ls = '-' if fam_key == f1 else '--'
            lw = 2 if fam_key == f1 else 2.5
            label_str = f"{fam_key} {COMBO_STRATEGIES[skey]['label'][:15]}"
            ax.plot(dates_plot, cum_val.values, color=strat_colors[si],
                    linewidth=lw, linestyle=ls, label=label_str)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax.set_ylabel('Cumulative EqW PnL (%)')
    ax.set_title('Top Strategy Equity Curves (solid=L5-1, dashed=L5-3)')
    ax.legend(fontsize=6, loc='upper left')
    ax.grid(alpha=0.3)

    # 5. Trade count vs Avg PnL scatter (both families)
    ax = axes[2, 0]
    for fam_key in MODEL_FAMILIES:
        for skey in skeys_with_trades:
            s = family_all_stats[fam_key][skey]
            if s['n'] == 0:
                continue
            scfg = COMBO_STRATEGIES[skey]
            marker = scfg['marker']
            ax.scatter(s['n'], s['avg_pnl'] * 100, c=FAM_COLORS[fam_key],
                       marker=marker, s=100, edgecolors='black',
                       linewidths=0.5, zorder=3, alpha=0.8)
            offset_x = 3 if fam_key == f1 else -3
            ha = 'left' if fam_key == f1 else 'right'
            ax.annotate(f"{fam_key[:4]}", (s['n'], s['avg_pnl'] * 100),
                        fontsize=5, ha=ha, va='bottom',
                        xytext=(offset_x, 3), textcoords='offset points',
                        color=FAM_COLORS[fam_key])
    ax.axhline(BASE_PNL_THRESHOLD * 100, color='green', linewidth=1,
               linestyle='--', alpha=0.5, label=f'>= {BASE_PNL_THRESHOLD:.1%}')
    ax.axhline(0, color='black', linewidth=0.8)
    from matplotlib.lines import Line2D
    legend_elems = [Line2D([0], [0], marker='o', color='w', markerfacecolor=FAM_COLORS[fk],
                           markersize=8, label=MODEL_FAMILIES[fk]['label'])
                    for fk in MODEL_FAMILIES]
    ax.legend(handles=legend_elems, fontsize=7)
    ax.set_xlabel('Number of Trades')
    ax.set_ylabel('Avg PnL per Trade (%)')
    ax.set_title('Trade Count vs Avg PnL (both families)')
    ax.grid(alpha=0.3)

    # 6. Win rate comparison
    ax = axes[2, 1]
    x_pos = np.arange(len(skeys_with_trades))
    bar_w = 0.35
    for i, fam_key in enumerate(MODEL_FAMILIES):
        vals = [family_all_stats[fam_key][sk]['wr'] * 100 for sk in skeys_with_trades]
        ax.bar(x_pos + i * bar_w, vals, bar_w, label=MODEL_FAMILIES[fam_key]['label'],
               color=FAM_COLORS[fam_key], alpha=0.85, edgecolor='black', linewidth=0.5,
               hatch=FAM_HATCHES[fam_key])
        for j, v in enumerate(vals):
            if v > 0:
                ax.text(x_pos[j] + i * bar_w, v + 1, f'{v:.0f}%',
                        ha='center', fontsize=6)
    ax.set_xticks(x_pos + bar_w / 2)
    ax.set_xticklabels(labels, fontsize=6, rotation=35, ha='right')
    ax.set_ylabel('Win Rate (%)')
    ax.set_title('Win Rate Comparison')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = '/tmp/L5_1_vs_L5_3_combination_strategies.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n  Chart saved to {save_path}")
    if _IN_COLAB:
        plt.show()
    plt.close()


# ==============================================================================
# VERDICT
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  VERDICT")
print(f"{'='*W}")

qualified = {}
for fam_key in MODEL_FAMILIES:
    qualified[fam_key] = [(sk, family_all_stats[fam_key][sk])
                          for sk in COMBO_STRATEGIES
                          if family_all_stats[fam_key][sk]['n'] >= 5]

for fam_key, fam_cfg in MODEL_FAMILIES.items():
    q = qualified[fam_key]
    if q:
        best = max(q, key=lambda x: x[1]['avg_pnl'])
        best_eqw = max(q, key=lambda x: x[1]['eqw'])
        print(f"\n  {fam_cfg['label']}:")
        print(f"    Best Avg PnL: {COMBO_STRATEGIES[best[0]]['label']}"
              f" ({best[1]['avg_pnl']:+.2%}, {best[1]['n']} trades, Sharpe {best[1]['sharpe']:.1f})")
        print(f"    Best EqW:     {COMBO_STRATEGIES[best_eqw[0]]['label']}"
              f" ({best_eqw[1]['eqw']:+.2%}, {best_eqw[1]['n']} trades)")
    else:
        print(f"\n  {fam_cfg['label']}: No strategies with >= 5 trades")

# Overall winner
print(f"\n  OVERALL:")
all_qualified = []
for fam_key in MODEL_FAMILIES:
    for sk, s in qualified[fam_key]:
        all_qualified.append((fam_key, sk, s))
if all_qualified:
    best_overall = max(all_qualified, key=lambda x: x[2]['avg_pnl'])
    best_eqw_overall = max(all_qualified, key=lambda x: x[2]['eqw'])
    print(f"    Best Avg PnL: {best_overall[0]} {COMBO_STRATEGIES[best_overall[1]]['label']}"
          f" ({best_overall[2]['avg_pnl']:+.2%}, {best_overall[2]['n']}t,"
          f" Sharpe {best_overall[2]['sharpe']:.1f})")
    print(f"    Best EqW PnL: {best_eqw_overall[0]} {COMBO_STRATEGIES[best_eqw_overall[1]]['label']}"
          f" ({best_eqw_overall[2]['eqw']:+.2%}, {best_eqw_overall[2]['n']}t)")

    # Count strategy wins
    print(f"\n  STRATEGY-LEVEL COMPARISON (L5-3 vs L5-1, avg PnL):")
    n_l3_better = 0; n_l1_better = 0; n_both = 0
    for skey in COMBO_STRATEGIES:
        s1 = family_all_stats[f1][skey]
        s2 = family_all_stats[f2][skey]
        label = COMBO_STRATEGIES[skey]['label']
        if s1['n'] < 5 and s2['n'] < 5:
            print(f"    {label:<24} -- too few trades in both")
            continue
        n_both += 1
        d = s2['avg_pnl'] - s1['avg_pnl']
        marker = '>>>' if d > 0.005 else ('<<<' if d < -0.005 else '===')
        winner_str = f"L5-3 +{d:+.2%}" if d > 0.001 else (f"L5-1 {d:+.2%}" if d < -0.001 else "TIE")
        print(f"    {label:<24} {marker} {winner_str}"
              f"  (L5-1: {s1['avg_pnl']:+.2%}/{s1['n']}t  L5-3: {s2['avg_pnl']:+.2%}/{s2['n']}t)")
        if d > 0.001:
            n_l3_better += 1
        elif d < -0.001:
            n_l1_better += 1

    print(f"\n    L5-3 wins {n_l3_better}/{n_both} strategies, "
          f"L5-1 wins {n_l1_better}/{n_both}, "
          f"Ties {n_both - n_l3_better - n_l1_better}/{n_both}")
    if n_l3_better > n_l1_better:
        print(f"\n    >>> L5-3 (Training Hygiene) CONFIRMED as improvement over L5-1 <<<")
    elif n_l1_better > n_l3_better:
        print(f"\n    >>> L5-1 (Baseline) holds — L5-3 does NOT improve <<<")
    else:
        print(f"\n    >>> INCONCLUSIVE — tied across strategies <<<")

print(f"\n{'='*W}")
