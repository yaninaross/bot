# %% ── ST LONG: Deployed F5 vs L5-1 (Fold 5/6/7) Head-to-Head Backtest ────────
#
# Compares:
#   1. DEPLOYED F5 (16 regime features)
#   2. L5-1 Fold 5 (22 regime features, sector-aware FiLM)
#   3. L5-1 Fold 6 (22 regime features)
#   4. L5-1 Fold 7 (22 regime features)
#
# LONG only. Equal-weight portfolio PnL.

import os, sys, gc, subprocess, warnings
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
BACKTEST_END   = '2026-08-12'

SLIPPAGE_PCT  = 0.0015
EXIT_TIME_MIN = 780
DEPLOY_PNL_THRESHOLD = 0.018

TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}
MICRO_COLS = ['Spread', 'OBI', 'Volume_Delta', 'Trade_Count', 'Delta_Pct',
              'bsize', 'asize', 'Total_Vol']
ROC_HORIZONS = [5, 30, 60]
CONTEXT_ASSETS = ['SPY', 'VIX', 'TLT', 'IWM']
SECTOR_PROXY_ASSETS = ['MARA', 'MSTR', 'ITA']
EXCLUDE_TICKERS = {'SPY', 'VIX', 'TLT', 'IWM', 'XLE', 'ITA', 'BITF', 'MIGI', 'SMLR'}

W = 110
gcs_client = gcs_storage.Client()

# ── Strategies ──
STRATEGIES = {
    'DEPLOYED': {
        'label': 'Deployed F5 (16 regime)',
        'bq_model_version': 'spike_transformer_fold5',
        'gcs_data_prefix': 'Algos/RA_MoE/real_data',
        'gcs_model_prefix': 'Algos/RA_MoE/real_models',
        'fold_id': 5,
        'n_regime_f': 16,
        'score_minute': 375,
        'entry_minute': 385,
        'color': '#1565C0',
        'marker': 'o',
    },
    'L51_F5': {
        'label': 'L5-1 Fold 5 (22 regime)',
        'bq_model_version': 'spike_transformer_long_L5_1_fold5',
        'gcs_data_prefix': 'Algos/RA_MoE/long_L5_1_data',
        'gcs_model_prefix': 'Algos/RA_MoE/long_L5_1_models',
        'fold_id': 5,
        'n_regime_f': 22,
        'score_minute': 375,
        'entry_minute': 385,
        'color': '#D32F2F',
        'marker': 's',
    },
    'L51_F6': {
        'label': 'L5-1 Fold 6 (22 regime)',
        'bq_model_version': 'spike_transformer_long_L5_1_fold6',
        'gcs_data_prefix': 'Algos/RA_MoE/long_L5_1_data',
        'gcs_model_prefix': 'Algos/RA_MoE/long_L5_1_models',
        'fold_id': 6,
        'n_regime_f': 22,
        'score_minute': 375,
        'entry_minute': 385,
        'color': '#FF6F00',
        'marker': 'D',
    },
    'L51_F7': {
        'label': 'L5-1 Fold 7 (22 regime)',
        'bq_model_version': 'spike_transformer_long_L5_1_fold7',
        'gcs_data_prefix': 'Algos/RA_MoE/long_L5_1_data',
        'gcs_model_prefix': 'Algos/RA_MoE/long_L5_1_models',
        'fold_id': 7,
        'n_regime_f': 22,
        'score_minute': 375,
        'entry_minute': 385,
        'color': '#2E7D32',
        'marker': '^',
    },
}

# ── Shared model params ──
TIME_STEPS = 128
D_MODEL = 64; N_HEADS = 4; D_FF = 128; N_BLOCKS = 2; DROPOUT = 0.1
FILM_HIDDEN = 64; N_HORIZONS = 3; N_ASSET_F = 33

ASSET_COLS = (['Gap_Pct', 'RSI_14', 'RVOL', 'ATR_14', 'Bar_Is_Real'] +
              [f'ROC_{h}m' for h in ROC_HORIZONS] + MICRO_COLS)

CONTEXT_COLS = []
for _t in CONTEXT_ASSETS:
    CONTEXT_COLS.append(f'{_t}_Anchor_ROC')
    for _h in ROC_HORIZONS:
        CONTEXT_COLS.append(f'{_t}_ROC_{_h}m')
CONTEXT_COLS.append('Time_Pos')

ALL_COLS = ASSET_COLS + CONTEXT_COLS

REGIME_COLS_16 = [
    'regime_rvol_15', 'regime_rvol_60', 'regime_rvol_128',
    'regime_trend_60', 'regime_trend_128', 'regime_autocorr_60',
    'regime_xsect_disp', 'regime_vix_level', 'regime_vix_velocity',
    'regime_time_sin', 'regime_time_cos', 'regime_spy_ret_60',
    'regime_vix_close', 'regime_spy_tlt_corr', 'regime_breadth',
    'regime_skew_60',
]

REGIME_COLS_22 = [
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
    def __init__(self, n_regime_f=16, **kw):
        super().__init__(**kw)
        self._n_regime_f = n_regime_f
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
    # Load sector proxy assets for L5-1 regime features
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
    """Computes all 22 regime features. Callers select 16 or 22 via regime_cols."""
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
    # Sector-aware features (L5-1 additions)
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
    return rf[REGIME_COLS_22].replace([np.inf, -np.inf], np.nan).fillna(0)


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


def load_strategy_model(skey, scfg):
    """Load model + meta for a strategy. Builds SpikeTransformer with correct N_REGIME_F."""
    n_regime_f = scfg['n_regime_f']
    fold_id = scfg['fold_id']
    data_prefix = scfg['gcs_data_prefix']
    model_prefix = scfg['gcs_model_prefix']

    local_meta = f'/tmp/bt_deployed_vs_l51_{skey}_meta.npz'
    meta_path = _ensure_file(f'{data_prefix}/meta.npz', local_meta)
    meta = np.load(meta_path, allow_pickle=True)
    scaling_stats = meta['scaling_stats'].item()
    regime_stats = meta['regime_stats'].item()

    model = SpikeTransformer(n_regime_f=n_regime_f, name=f'spike_transformer_{skey.lower()}')
    dummy_seq = tf.zeros((1, TIME_STEPS, N_ASSET_F))
    dummy_reg = tf.zeros((1, n_regime_f))
    model({'seq_input': dummy_seq, 'regime_input': dummy_reg})

    local_dir = f'/tmp/bt_deployed_vs_l51_models/{skey}/fold_{fold_id}'
    weights_path = os.path.join(local_dir, 'model.weights.h5')
    if not os.path.exists(weights_path):
        os.makedirs(local_dir, exist_ok=True)
        prefix = f'{model_prefix}/fold_{fold_id}'
        bucket = gcs_client.bucket(BUCKET_NAME)
        for blob in bucket.list_blobs(prefix=prefix):
            fname = blob.name.split('/')[-1]
            blob.download_to_filename(os.path.join(local_dir, fname))
    model.load_weights(weights_path)
    print(f"  {scfg['label']} model loaded: {model.count_params():,} params "
          f"(fold {fold_id}, {n_regime_f} regime features)")
    return model, scaling_stats, regime_stats


# ==============================================================================
# BQ CONFIG LOADING
# ==============================================================================
def load_deploy_configs(model_version):
    if bigquery is None:
        print(f"  BigQuery not available")
        return {}
    bq_client = bigquery.Client()
    q = f"""
    SELECT ticker, experiment, tpsl_mode, prob_thresh, tp_pct, sl_pct,
           grace_minutes, dip_pct, max_wait_minutes,
           val_b_avg_pnl, val_b_win_rate, val_b_count
    FROM `{WF_BQ_TABLE}`
    WHERE model_version = '{model_version}'
      AND survived_a = TRUE
      AND val_b_avg_pnl >= {DEPLOY_PNL_THRESHOLD}
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
        # Parse horizon from experiment name after model_version prefix
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
            'experiment': exp,
        }
    return configs


# ==============================================================================
# SCORING
# ==============================================================================
def score_ticker_day(model, scaling_stats, regime_stats,
                     ticker, cfg, df_feat, regime_df,
                     target_date, score_minute,
                     regime_cols, n_regime_f):
    target_pst = PST.localize(datetime(target_date.year, target_date.month,
                                       target_date.day,
                                       score_minute // 60, score_minute % 60))
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
    if bar_minute != score_minute:
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
    regime_scaled = np.zeros(n_regime_f, dtype=np.float32)
    for i, col in enumerate(regime_cols):
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
    signal = prob >= cfg['prob_thresh']
    close = float(df_feat['Close'].iloc[row_idx])

    return {
        'prob': round(prob, 4),
        'signal': signal,
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
            elapsed = (bar_pst.hour * 60 + bar_pst.minute) - (df_feat.index[entry_idx].tz_convert(PST).hour * 60 + df_feat.index[entry_idx].tz_convert(PST).minute)
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
    entry_time_min = df_feat.index[start_idx].tz_convert(PST).hour * 60 + df_feat.index[start_idx].tz_convert(PST).minute

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
# GET TRADING DAYS
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
# REGIME COLS LOOKUP
# ==============================================================================
def regime_cols_for_strategy(skey):
    n = STRATEGIES[skey]['n_regime_f']
    if n == 16:
        return REGIME_COLS_16
    return REGIME_COLS_22


# ==============================================================================
# MAIN BACKTEST
# ==============================================================================
SKEYS = list(STRATEGIES.keys())

print(f"\n{'='*W}")
print(f"  ST LONG: DEPLOYED F5 vs L5-1 (FOLD 5/6/7) HEAD-TO-HEAD")
print(f"  Period: {BACKTEST_START} -> {BACKTEST_END}")
print(f"  Slippage: {SLIPPAGE_PCT:.2%} each way ({2*SLIPPAGE_PCT:.2%} RT)")
print(f"  Deploy threshold: {DEPLOY_PNL_THRESHOLD:.1%}")
print(f"{'='*W}")

# 1. Load configs from BQ for all strategies
strat_configs = {}
for skey, scfg in STRATEGIES.items():
    print(f"\n  Loading {skey} configs (model_version={scfg['bq_model_version']})...")
    cfgs = load_deploy_configs(scfg['bq_model_version'])
    strat_configs[skey] = cfgs
    tickers = sorted(cfgs.keys())
    print(f"  {skey}: {len(tickers)} deployable tickers - {', '.join(tickers)}")

# Compute ticker sets
strat_tickers = {skey: set(cfgs.keys()) for skey, cfgs in strat_configs.items()}
all_strategy_tickers = set()
for s in strat_tickers.values():
    all_strategy_tickers |= s

# Overlap analysis
print(f"\n  TICKER COVERAGE:")
for skey in SKEYS:
    print(f"    {STRATEGIES[skey]['label']}: {len(strat_tickers[skey])} tickers")
all_overlap = set.intersection(*strat_tickers.values()) if len(strat_tickers) > 1 else set()
if all_overlap:
    print(f"    All 4 overlap: {len(all_overlap)} - {', '.join(sorted(all_overlap))}")

# 2. Build market context and regime features
print(f"\n  Building market context (incl. sector proxies)...")
market_df = build_market_context()
print(f"  Computing regime features (22 cols, callers select 16 or 22)...")
regime_df = compute_regime_features(market_df)

# 3. Get trading days
trading_days = get_trading_days(BACKTEST_START, BACKTEST_END)
print(f"  Trading days: {len(trading_days)} ({trading_days[0]} -> {trading_days[-1]})")

# 4. Load all models (separate instance per strategy since N_REGIME_F differs)
print(f"\n  Loading models...")
models = {}
scaling_stats = {}
regime_stats_dict = {}
for skey, scfg in STRATEGIES.items():
    m, ss, rs = load_strategy_model(skey, scfg)
    models[skey] = m
    scaling_stats[skey] = ss
    regime_stats_dict[skey] = rs

# 5. Engineer features for all relevant tickers
print(f"\n  Engineering features for {len(all_strategy_tickers)} tickers...")
ticker_features = {}
for ticker in sorted(all_strategy_tickers):
    df_feat = engineer_asset(ticker, market_df)
    if df_feat is not None:
        ticker_features[ticker] = df_feat
        print(f"    {ticker}: {len(df_feat):,} bars")
    else:
        print(f"    {ticker}: FAILED")

# 6. Run backtest day by day
all_trades = []

print(f"\n{'='*W}")
print(f"  SCORING + SIMULATION")
print(f"{'='*W}")

for day in trading_days:
    day_str = day.strftime('%Y-%m-%d')
    day_signals = {skey: [] for skey in SKEYS}

    for skey, scfg in STRATEGIES.items():
        cfgs = strat_configs[skey]
        score_min = scfg['score_minute']
        entry_min = scfg['entry_minute']
        rcols = regime_cols_for_strategy(skey)
        n_rf = scfg['n_regime_f']

        for ticker in sorted(cfgs.keys()):
            if ticker not in ticker_features:
                continue
            cfg = cfgs[ticker]
            result = score_ticker_day(
                models[skey], scaling_stats[skey], regime_stats_dict[skey],
                ticker, cfg, ticker_features[ticker], regime_df,
                day, score_min,
                regime_cols=rcols, n_regime_f=n_rf)
            if result is None:
                continue

            entry_bar_idx = result['bar_idx']
            if entry_min != score_min:
                entry_bar_idx = result['bar_idx'] + (entry_min - score_min)

            if result['signal']:
                trade = simulate_long_trade(ticker_features[ticker], entry_bar_idx, cfg)
                if trade is not None:
                    trade['pnl'] -= 2 * SLIPPAGE_PCT
                    all_trades.append({
                        'Date': day_str, 'Ticker': ticker, 'Strategy': skey,
                        'Prob': result['prob'], 'Thresh': cfg['prob_thresh'],
                        'Entry': trade['entry'], 'Exit': trade['exit'],
                        'PnL': trade['pnl'], 'Outcome': trade['outcome'],
                        'Mode': cfg['mode'], 'Horizon': cfg.get('horizon_label', '30m'),
                        'TP_pct': cfg['tp_pct'], 'SL_pct': cfg['sl_pct'],
                    })
                    day_signals[skey].append(f"{ticker}({result['prob']:.2f})")
                else:
                    day_signals[skey].append(f"{ticker}(no fill)")

    has_any = any(day_signals[s] for s in SKEYS)
    if has_any:
        parts = []
        for skey in SKEYS:
            sig = ', '.join(day_signals[skey]) if day_signals[skey] else 'none'
            parts.append(f"{skey}: {sig}")
        print(f"\n  {day_str}  {'  |  '.join(parts)}")

keras.backend.clear_session()
gc.collect()


# ==============================================================================
# RESULTS
# ==============================================================================
if not all_trades:
    print(f"\n  NO TRADES in backtest window")
else:
    trades_df = pd.DataFrame(all_trades)
    trades_df['Date_dt'] = pd.to_datetime(trades_df['Date'])
    trades_df = trades_df.sort_values(['Date_dt', 'Strategy', 'Ticker'])

    # Per-strategy detailed log
    for skey in SKEYS:
        sdf = trades_df[trades_df['Strategy'] == skey]
        scfg = STRATEGIES[skey]
        if len(sdf) == 0:
            print(f"\n  {scfg['label']}: No trades")
            continue

        n = len(sdf)
        n_win = (sdf['PnL'] > 0).sum()
        n_loss = (sdf['PnL'] <= 0).sum()
        avg_pnl = sdf['PnL'].mean()
        wr = n_win / n
        avg_win = sdf.loc[sdf['PnL'] > 0, 'PnL'].mean() if n_win > 0 else 0
        avg_loss = sdf.loc[sdf['PnL'] <= 0, 'PnL'].mean() if n_loss > 0 else 0

        print(f"\n{'='*W}")
        print(f"  {scfg['label'].upper()} -- {BACKTEST_START} to {BACKTEST_END}")
        print(f"  Entry: min {scfg['entry_minute']} | Score: min {scfg['score_minute']} | Regime: {scfg['n_regime_f']} features")
        print(f"{'='*W}")
        print(f"  Trades:     {n}")
        print(f"  Avg PnL:    {avg_pnl:+.2%}")
        print(f"  Win rate:   {wr:.0%} ({n_win}W / {n_loss}L)")
        print(f"  Avg win:    {avg_win:+.2%}")
        print(f"  Avg loss:   {avg_loss:+.2%}")
        if n_loss > 0 and avg_loss != 0:
            print(f"  PF:         {abs(avg_win * n_win / (avg_loss * n_loss)):.2f}")

        print(f"\n  TRADE LOG:")
        print(f"  {'Date':<12} {'Ticker':<7} {'H':>4} {'Mode':<13} "
              f"{'Prob':>5}/{' Thr':>5}  "
              f"{'Entry':>9} {'Exit':>9} {'PnL':>8} {'Type':>4}")
        print(f"  {'-'*95}")
        for _, t in sdf.iterrows():
            print(f"  {t['Date']:<12} {t['Ticker']:<7} {t['Horizon']:>4} {t['Mode']:<13} "
                  f"{t['Prob']:>5.2f}/{t['Thresh']:>5.2f}  "
                  f"${t['Entry']:>8.2f} ${t['Exit']:>8.2f} {t['PnL']:>+7.2%} {t['Outcome']:>4}")

        print(f"\n  PER-TICKER:")
        print(f"  {'Ticker':<7} {'N':>3} {'Avg PnL':>8} {'Total':>8} {'WR':>5} {'TP':>3} {'SL':>3} {'EOD':>3}")
        print(f"  {'-'*50}")
        for ticker in sorted(sdf['Ticker'].unique()):
            tdf = sdf[sdf['Ticker'] == ticker]
            t_n = len(tdf)
            t_avg = tdf['PnL'].mean()
            t_tot = tdf['PnL'].sum()
            t_wr = (tdf['PnL'] > 0).mean()
            t_tp = (tdf['Outcome'] == 'TP').sum()
            t_sl = (tdf['Outcome'] == 'SL').sum()
            t_eod = (tdf['Outcome'] == 'EOD').sum()
            print(f"  {ticker:<7} {t_n:>3} {t_avg:>+7.2%} {t_tot:>+7.2%} {t_wr:>4.0%} {t_tp:>3} {t_sl:>3} {t_eod:>3}")

    # ── HEAD-TO-HEAD COMPARISON ──
    print(f"\n{'='*W}")
    print(f"  HEAD-TO-HEAD COMPARISON (4 strategies)")
    print(f"{'='*W}")

    strat_trades = {skey: trades_df[trades_df['Strategy'] == skey] for skey in SKEYS}

    # Build HYBRID: for each ticker, pick strategy with highest val_b_avg_pnl
    hybrid_fold_for_ticker = {}
    for ticker in all_strategy_tickers:
        best_skey = None
        best_val = -999
        for skey in SKEYS:
            if ticker in strat_configs[skey]:
                v = strat_configs[skey][ticker]['val_b_avg_pnl']
                if v > best_val:
                    best_val = v
                    best_skey = skey
        if best_skey is not None:
            hybrid_fold_for_ticker[ticker] = best_skey

    hybrid_trades = trades_df[trades_df.apply(
        lambda r: hybrid_fold_for_ticker.get(r['Ticker']) == r['Strategy'], axis=1)].copy()
    hybrid_trades['Strategy'] = 'HYBRID'

    print(f"\n  HYBRID strategy: {len(hybrid_fold_for_ticker)} tickers")
    for skey in SKEYS:
        skey_tickers = [t for t, s in hybrid_fold_for_ticker.items() if s == skey]
        if skey_tickers:
            print(f"    from {skey}: {len(skey_tickers)} - {', '.join(sorted(skey_tickers))}")

    # Summary table header
    header_parts = [f"{'Metric':<20}"]
    for skey in SKEYS:
        header_parts.append(f"{STRATEGIES[skey]['label'][:18]:>18}")
    header_parts.append(f"{'Hybrid':>18}")
    print(f"\n  {'  '.join(header_parts)}")
    print(f"  {'-'*(20 + 20 * (len(SKEYS) + 1))}")

    # Compute stats for each strategy + hybrid
    all_strat_keys = SKEYS + ['HYBRID']
    all_strat_trades = {skey: strat_trades[skey] for skey in SKEYS}
    all_strat_trades['HYBRID'] = hybrid_trades

    strat_n = {}; strat_avg = {}; strat_wr = {}
    strat_nw = {}; strat_nl = {}; strat_aw = {}; strat_al = {}; strat_pf = {}
    for sk in all_strat_keys:
        sdf = all_strat_trades[sk]
        n = len(sdf)
        strat_n[sk] = n
        strat_avg[sk] = sdf['PnL'].mean() if n > 0 else 0
        strat_wr[sk] = (sdf['PnL'] > 0).mean() if n > 0 else 0
        nw = (sdf['PnL'] > 0).sum() if n > 0 else 0
        nl = n - nw
        strat_nw[sk] = nw; strat_nl[sk] = nl
        strat_aw[sk] = sdf.loc[sdf['PnL'] > 0, 'PnL'].mean() if nw > 0 else 0
        strat_al[sk] = sdf.loc[sdf['PnL'] <= 0, 'PnL'].mean() if nl > 0 else 0
        if nl > 0 and strat_al[sk] != 0:
            strat_pf[sk] = abs(strat_aw[sk] * nw / (strat_al[sk] * nl))
        else:
            strat_pf[sk] = float('inf')

    def _fmt_row(label, vals):
        parts = [f"  {label:<20}"]
        for v in vals:
            parts.append(f"{v:>18}")
        print('  '.join(parts))

    display_keys = SKEYS + ['HYBRID']
    _fmt_row('Tickers', [str(len(strat_tickers.get(sk, hybrid_fold_for_ticker.keys()))) if sk != 'HYBRID' else str(len(hybrid_fold_for_ticker)) for sk in display_keys])
    _fmt_row('Trades', [str(strat_n[sk]) for sk in display_keys])
    _fmt_row('Avg PnL/trade', [f"{strat_avg[sk]:+.2%}" for sk in display_keys])
    _fmt_row('Win Rate', [f"{strat_wr[sk]:.0%}" for sk in display_keys])
    _fmt_row('Avg Win', [f"{strat_aw[sk]:+.2%}" for sk in display_keys])
    _fmt_row('Avg Loss', [f"{strat_al[sk]:+.2%}" for sk in display_keys])
    _fmt_row('Profit Factor', [f"{strat_pf[sk]:.2f}" if strat_pf[sk] < 999 else 'inf' for sk in display_keys])

    # Daily equal-weight PnL
    daily_pnl = {}
    for sk in display_keys:
        sdf = all_strat_trades[sk]
        daily_pnl[sk] = sdf.groupby('Date')['PnL'].mean() if len(sdf) > 0 else pd.Series(dtype=float)

    all_dates = sorted(set().union(*(daily_pnl[sk].index for sk in display_keys if len(daily_pnl[sk]) > 0)))
    eqw = {sk: sum(daily_pnl[sk].get(d, 0) for d in all_dates) for sk in display_keys}
    _fmt_row('EqW Portfolio PnL', [f"{eqw[sk]:+.2%}" for sk in display_keys])

    # Per-ticker head-to-head for tickers in ALL strategies
    shared_all = set.intersection(*[strat_tickers[sk] for sk in SKEYS]) if len(SKEYS) > 1 else set()
    if shared_all:
        print(f"\n  TICKERS IN ALL 4 STRATEGIES ({len(shared_all)}):")
        header = f"  {'Ticker':<7}"
        for skey in SKEYS:
            header += f" {skey[:8]+' Avg':>10} {skey[:8]+' N':>5}"
        header += f" {'Winner':>10}"
        print(header)
        print(f"  {'-'*(7 + (10 + 5 + 2) * len(SKEYS) + 12)}")
        wins_count = {skey: 0 for skey in SKEYS}
        for tk in sorted(shared_all):
            row_str = f"  {tk:<7}"
            avgs = {}
            for skey in SKEYS:
                tdf = strat_trades[skey][strat_trades[skey]['Ticker'] == tk]
                a = tdf['PnL'].mean() if len(tdf) > 0 else 0
                avgs[skey] = a
                row_str += f" {a:>+9.2%} {len(tdf):>5}"
            winner = max(avgs, key=avgs.get)
            wins_count[winner] += 1
            row_str += f" {winner:>10}"
            print(row_str)
        win_str = ', '.join(f"{sk}={wins_count[sk]}" for sk in SKEYS)
        print(f"  Wins: {win_str}")

    # Pairwise comparison: DEPLOYED vs each L5-1 variant
    for l51key in ['L51_F5', 'L51_F6', 'L51_F7']:
        shared = strat_tickers['DEPLOYED'] & strat_tickers[l51key]
        if not shared:
            continue
        print(f"\n  DEPLOYED vs {l51key} -- SHARED TICKERS ({len(shared)}):")
        print(f"  {'Ticker':<7} {'DEP Avg':>8} {'DEP N':>5} {l51key+' Avg':>10} {l51key+' N':>7} {'Winner':>8}")
        print(f"  {'-'*55}")
        dep_w = 0; l51_w = 0
        for tk in sorted(shared):
            dep_tk = strat_trades['DEPLOYED'][strat_trades['DEPLOYED']['Ticker'] == tk]
            l51_tk = strat_trades[l51key][strat_trades[l51key]['Ticker'] == tk]
            dep_a = dep_tk['PnL'].mean() if len(dep_tk) > 0 else 0
            l51_a = l51_tk['PnL'].mean() if len(l51_tk) > 0 else 0
            winner = 'DEPLOYED' if dep_a > l51_a else l51key if l51_a > dep_a else 'TIE'
            if winner == 'DEPLOYED': dep_w += 1
            elif winner == l51key: l51_w += 1
            print(f"  {tk:<7} {dep_a:>+7.2%} {len(dep_tk):>5} {l51_a:>+9.2%} {len(l51_tk):>7} {winner:>8}")
        print(f"  Wins: DEPLOYED={dep_w}, {l51key}={l51_w}")

    # Daily breakdown
    print(f"\n  DAILY EQUAL-WEIGHT PnL:")
    header = f"  {'Date':<12}"
    for skey in SKEYS:
        header += f" {skey[:6]+' N':>6} {skey[:6]+' EqW':>8}"
    header += f" {'Hyb N':>6} {'Hyb EqW':>8}"
    # Cumulative columns
    for skey in SKEYS:
        header += f" {skey[:4]+'Cum':>8}"
    header += f" {'HybCum':>8}"
    print(header)
    print(f"  {'-'*(12 + (6+8+1)*len(SKEYS) + 6+8+1 + (8+1)*(len(SKEYS)+1))}")

    cum = {sk: 0.0 for sk in display_keys}
    for d in sorted(set(trades_df['Date'].unique())):
        row_str = f"  {d:<12}"
        has_any_day = False
        for skey in SKEYS:
            d_s = trades_df[(trades_df['Date'] == d) & (trades_df['Strategy'] == skey)]
            ns = len(d_s)
            eqws = d_s['PnL'].mean() if ns > 0 else 0
            cum[skey] += eqws
            if ns > 0:
                has_any_day = True
            row_str += f" {ns:>6} {eqws:>+7.2%}"
        d_h = hybrid_trades[hybrid_trades['Date'] == d]
        nh = len(d_h)
        eqwh = d_h['PnL'].mean() if nh > 0 else 0
        cum['HYBRID'] += eqwh
        if nh > 0:
            has_any_day = True
        row_str += f" {nh:>6} {eqwh:>+7.2%}"
        for skey in SKEYS:
            row_str += f" {cum[skey]:>+7.2%}"
        row_str += f" {cum['HYBRID']:>+7.2%}"
        if has_any_day:
            print(row_str)

    # ── Charts (6 panels, 3x2) ──
    n_all = len(trades_df)
    if HAS_MPL and n_all > 0:
        fig, axes = plt.subplots(3, 2, figsize=(20, 20))

        # 1. Trade scatter by date (jittered)
        ax = axes[0, 0]
        jitter_map = {}
        for _, row in trades_df.iterrows():
            d = row['Date']
            jitter_map.setdefault(d, []).append(row.name)
        jitter_offsets = {}
        for d, idxs in jitter_map.items():
            n = len(idxs)
            offsets = np.linspace(-0.3, 0.3, n) if n > 1 else [0.0]
            for idx, off in zip(idxs, offsets):
                jitter_offsets[idx] = off

        for skey in SKEYS:
            scfg = STRATEGIES[skey]
            sdf = trades_df[trades_df['Strategy'] == skey]
            if len(sdf) == 0:
                continue
            dates_num = np.array([mdates.date2num(r['Date_dt']) + jitter_offsets.get(i, 0)
                                  for i, r in sdf.iterrows()])
            pnl_pct = sdf['PnL'].values * 100
            winners = pnl_pct >= 0
            ax.scatter(dates_num[winners], pnl_pct[winners],
                       c=scfg['color'], marker=scfg['marker'], s=80, alpha=0.8,
                       edgecolors='black', linewidths=0.5,
                       label=f'{skey} win', zorder=3)
            ax.scatter(dates_num[~winners], pnl_pct[~winners],
                       c=scfg['color'], marker=scfg['marker'], s=80, alpha=0.3,
                       edgecolors='black', linewidths=0.5, zorder=3)
            for idx, (i, row) in enumerate(sdf.iterrows()):
                ax.annotate(row['Ticker'], (dates_num[idx], row['PnL'] * 100),
                            fontsize=5, ha='center', va='bottom', alpha=0.6)
        ax.axhline(0, color='black', linewidth=1)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax.set_ylabel('Trade PnL (%)')
        ax.set_title('Individual Trades by Date')
        ax.legend(loc='upper left', fontsize=7)
        ax.grid(alpha=0.3)

        # 2. Daily equal-weight PnL bars (4 strategies)
        ax = axes[0, 1]
        all_dates_sorted = sorted(trades_df['Date'].unique())
        x_pos = np.arange(len(all_dates_sorted))
        n_bars = len(SKEYS)
        bar_w = 0.8 / n_bars
        for si, skey in enumerate(SKEYS):
            scfg = STRATEGIES[skey]
            daily_vals = []
            for d in all_dates_sorted:
                d_s = trades_df[(trades_df['Date'] == d) & (trades_df['Strategy'] == skey)]
                daily_vals.append(d_s['PnL'].mean() * 100 if len(d_s) > 0 else 0)
            offset = (si - (n_bars - 1) / 2) * bar_w
            ax.bar(x_pos + offset, daily_vals, bar_w, color=scfg['color'],
                   alpha=0.8, label=skey)
        ax.set_xticks(x_pos[::2])
        ax.set_xticklabels([d[5:] for d in all_dates_sorted[::2]], rotation=45, ha='right')
        ax.axhline(0, color='black', linewidth=1)
        ax.set_ylabel('Daily Portfolio PnL (%)')
        ax.set_title('Daily PnL -- Equal Weight')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, axis='y')

        # 3. Cumulative equity curves (4 strategies + hybrid)
        ax = axes[1, 0]
        for skey in SKEYS:
            scfg = STRATEGIES[skey]
            sdf = trades_df[trades_df['Strategy'] == skey]
            if len(sdf) == 0:
                continue
            daily_eqw = sdf.groupby('Date')['PnL'].mean().sort_index()
            cum_val = daily_eqw.cumsum() * 100
            dates_plot = pd.to_datetime(cum_val.index)
            ax.plot(dates_plot, cum_val.values, color=scfg['color'], linewidth=2.5,
                    marker=scfg['marker'], markersize=4,
                    label=f"{skey} ({len(sdf)} trades)")
        if len(hybrid_trades) > 0:
            daily_hyb_eq = hybrid_trades.groupby('Date')['PnL'].mean().sort_index()
            cum_hyb = daily_hyb_eq.cumsum() * 100
            dates_hyb = pd.to_datetime(cum_hyb.index)
            ax.plot(dates_hyb, cum_hyb.values, color='#6A1B9A', linewidth=2.5,
                    marker='*', markersize=5, linestyle='--',
                    label=f"HYBRID ({len(hybrid_trades)} trades)")
        ax.axhline(0, color='black', linewidth=0.8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax.set_ylabel('Cumulative PnL (%)')
        ax.set_title('Portfolio Equity Curve -- Equal Weight')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        # 4. Per-ticker avg PnL comparison (4 strategies side by side)
        ax = axes[1, 1]
        all_tk = sorted(set(trades_df['Ticker'].unique()))
        y_pos = np.arange(len(all_tk))
        bh = 0.8 / len(SKEYS)
        for si, skey in enumerate(SKEYS):
            scfg = STRATEGIES[skey]
            vals = []
            for tk in all_tk:
                tdf = trades_df[(trades_df['Ticker'] == tk) & (trades_df['Strategy'] == skey)]
                vals.append(tdf['PnL'].mean() * 100 if len(tdf) > 0 else 0)
            offset = (si - (len(SKEYS) - 1) / 2) * bh
            ax.barh(y_pos + offset, vals, bh, color=scfg['color'],
                    alpha=0.8, label=skey)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(all_tk, fontsize=7)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.axvline(DEPLOY_PNL_THRESHOLD * 100, color='green', linewidth=0.8,
                   linestyle='--', alpha=0.5, label=f'>= {DEPLOY_PNL_THRESHOLD:.1%}')
        ax.set_xlabel('Avg PnL per trade (%)')
        ax.set_title('Per-Ticker Avg PnL -- 4 Strategies')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, axis='x')

        # 5. Cumulative PnL by trade number
        ax = axes[2, 0]
        for skey in SKEYS:
            scfg = STRATEGIES[skey]
            sdf = trades_df[trades_df['Strategy'] == skey].sort_values('Date_dt')
            if len(sdf) > 0:
                cum_val = sdf['PnL'].cumsum().values * 100
                ax.plot(range(1, len(cum_val) + 1), cum_val, color=scfg['color'],
                        linewidth=2, label=f'{skey} ({len(sdf)} trades)')
        if len(hybrid_trades) > 0:
            hyb_sorted = hybrid_trades.sort_values('Date_dt')
            cum_h = hyb_sorted['PnL'].cumsum().values * 100
            ax.plot(range(1, len(cum_h) + 1), cum_h, color='#6A1B9A',
                    linewidth=2, linestyle='--', label=f'HYBRID ({len(hyb_sorted)} trades)')
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_xlabel('Trade #')
        ax.set_ylabel('Cumulative PnL (%)')
        ax.set_title('Cumulative PnL by Trade Number')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        # 6. Outcome distribution
        ax = axes[2, 1]
        outcomes = ['TP', 'SL', 'EOD']
        x_out = np.arange(len(outcomes))
        bw_out = 0.8 / len(SKEYS)
        for si, skey in enumerate(SKEYS):
            scfg = STRATEGIES[skey]
            sdf = trades_df[trades_df['Strategy'] == skey]
            counts = [(sdf['Outcome'] == o).sum() for o in outcomes]
            offset = (si - (len(SKEYS) - 1) / 2) * bw_out
            ax.bar(x_out + offset, counts, bw_out, color=scfg['color'],
                   alpha=0.8, label=skey)
        ax.set_xticks(x_out)
        ax.set_xticklabels(outcomes)
        ax.set_ylabel('Count')
        ax.set_title('Outcome Distribution')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, axis='y')

        plt.suptitle(f'ST LONG: Deployed F5 vs L5-1 (F5/F6/F7) -- {BACKTEST_START} to {BACKTEST_END}',
                     fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()
        save_path = '/tmp/st_long_deployed_vs_L5_1.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n  Chart saved to {save_path}")
        plt.show()
        plt.close()

    # ── VERDICT ──
    print(f"\n{'='*W}")
    all_options = [(skey, strat_avg[skey]) for skey in SKEYS if strat_n[skey] > 0]
    all_options.append(('HYBRID', strat_avg.get('HYBRID', 0)))
    best = max(all_options, key=lambda x: x[1])
    print(f"  VERDICT: {best[0]} wins with {best[1]:+.2%} avg PnL/trade")
    for skey in SKEYS:
        n = strat_n[skey]
        if n > 0:
            print(f"  {STRATEGIES[skey]['label']}: {strat_avg[skey]:+.2%} avg ({n} trades, {strat_wr[skey]:.0%} WR) | EqW {eqw[skey]:+.2%}")
        else:
            print(f"  {STRATEGIES[skey]['label']}: No trades")
    hn = strat_n.get('HYBRID', 0)
    if hn > 0:
        print(f"  HYBRID: {strat_avg['HYBRID']:+.2%} avg ({hn} trades, {strat_wr['HYBRID']:.0%} WR) | EqW {eqw['HYBRID']:+.2%}")
    print(f"  HYBRID picks best val_b_avg_pnl per ticker across all 4 strategies")
    print(f"{'='*W}")
