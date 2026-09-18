# %% ── SHORT L5-1 vs L5-2: Direct F5 Model Comparison ──────────────────────────
#
# Compares SHORT Spike Transformer fold 5 models head-to-head:
#   L5-1 F5: baseline (64 tickers)
#   L5-2 F5: retrained (99 tickers, +4 bday shift)
#
# Each model uses its OWN optimizer configs from BQ and its OWN meta.npz scaling.
# Backtest window derived from BQ: overlap of both F5 TEST windows (fair comparison).
#
# Output:
#   - Per-ticker: trades, avg PnL, total PnL for each model
#   - Shared tickers: head-to-head delta
#   - L5-2 only tickers: what the expanded universe adds
#   - Aggregate stats + charts
#
# SHORT only. Colab-ready.

import os, sys, gc, subprocess, warnings
from datetime import datetime, timedelta
from collections import OrderedDict

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
BUCKET_NAME   = os.environ.get("GCS_BUCKET", "")
INFERENCE_DIR = "Data/Inference_v2"
WF_BQ_TABLE   = f"{GCP_PROJECT}.scoring_logs.wf_optimization_results"
PST           = pytz.timezone('America/Los_Angeles')
TAIL_DAYS     = 60

SLIPPAGE_PCT  = 0.0015
EXIT_TIME_MIN = 780
BASE_PNL_THRESHOLD = 0.018
SENSITIVITY_THRESHOLDS = [0.018, 0.015, 0.012, 0.010, 0.008]
SENSITIVITY_MIN = min(SENSITIVITY_THRESHOLDS)

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

# ── F5 sources for each family ──
FAMILY_F5 = OrderedDict([
    ('L5-1', {'fold_id': 5, 'bq_mv': 'spike_transformer_short_L5_1_fold5',
              'gcs_data': 'Algos/RA_MoE/short_L5_1_data',
              'gcs_models': 'Algos/RA_MoE/short_L5_1_models', 'tag': 'L5_1',
              'label': 'L5-1 (64 tickers)', 'color': '#1565C0'}),
    ('L5-2', {'fold_id': 5, 'bq_mv': 'spike_transformer_short_L5_2_fold5',
              'gcs_data': 'Algos/RA_MoE/short_L5_2_data',
              'gcs_models': 'Algos/RA_MoE/short_L5_2_models', 'tag': 'L5_2',
              'label': 'L5-2 (99 tickers)', 'color': '#FF6F00'}),
])

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
# MODEL ARCHITECTURE  (SpikeTransformer — same for both families)
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


def load_meta(gcs_data_prefix, tag):
    local_meta = f'/tmp/bt_short_{tag}_f5_meta.npz'
    meta_path = _ensure_file(f'{gcs_data_prefix}/meta.npz', local_meta)
    meta = np.load(meta_path, allow_pickle=True)
    return meta['scaling_stats'].item(), meta['regime_stats'].item()


def load_fold_model(gcs_models, tag, fold_id):
    model = SpikeTransformer(name=f'spike_transformer_{tag}_f5')
    dummy_seq = tf.zeros((1, TIME_STEPS, N_ASSET_F))
    dummy_reg = tf.zeros((1, N_REGIME_F))
    model({'seq_input': dummy_seq, 'regime_input': dummy_reg})

    local_dir = f'/tmp/bt_short_{tag}_f5_models/fold_{fold_id}'
    weights_path = os.path.join(local_dir, 'model.weights.h5')
    if not os.path.exists(weights_path):
        os.makedirs(local_dir, exist_ok=True)
        prefix = f'{gcs_models}/fold_{fold_id}'
        bucket = gcs_client.bucket(BUCKET_NAME)
        for blob in bucket.list_blobs(prefix=prefix):
            fname = blob.name.split('/')[-1]
            blob.download_to_filename(os.path.join(local_dir, fname))
    model.load_weights(weights_path)
    print(f"    {tag} F5 (fold {fold_id}): {model.count_params():,} params")
    return model


# ==============================================================================
# BQ CONFIG LOADING
# ==============================================================================
def load_fold_configs(bq_model_version, threshold=None):
    if bigquery is None:
        print(f"  BigQuery not available")
        return {}
    thresh = threshold if threshold is not None else BASE_PNL_THRESHOLD
    bq_client = bigquery.Client()
    q = f"""
    SELECT ticker, experiment, tpsl_mode, prob_thresh, tp_pct, sl_pct,
           grace_minutes, dip_pct, max_wait_minutes,
           val_b_avg_pnl, val_b_win_rate, val_b_count
    FROM `{WF_BQ_TABLE}`
    WHERE model_version = '{bq_model_version}'
      AND survived_a = TRUE
      AND val_b_avg_pnl >= {thresh}
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


def get_f5_test_windows():
    """Pull val_b_start/val_b_end from BQ for each family's F5.
    Returns {model_version: (test_start, test_end)} and the F5 overlap window."""
    if bigquery is None:
        return {}, None, None
    bq_client = bigquery.Client()
    f5_mvs = [src['bq_mv'] for src in FAMILY_F5.values()]
    mv_list = "', '".join(f5_mvs)
    q = f"""
    SELECT model_version,
           MIN(val_b_start) as test_start,
           MAX(val_b_end) as test_end
    FROM `{WF_BQ_TABLE}`
    WHERE model_version IN ('{mv_list}')
      AND val_b_start IS NOT NULL AND val_b_start != ''
      AND val_b_end IS NOT NULL AND val_b_end != ''
    GROUP BY model_version
    """
    rows = bq_client.query(q).to_dataframe()
    windows = {}
    for _, row in rows.iterrows():
        windows[row['model_version']] = (row['test_start'], row['test_end'])

    f5_found = {mv: windows[mv] for mv in f5_mvs if mv in windows}
    if len(f5_found) < 2:
        return windows, None, None

    overlap_start = max(w[0] for w in f5_found.values())
    overlap_end = min(w[1] for w in f5_found.values())

    if overlap_start >= overlap_end:
        print(f"  WARNING: F5 TEST windows don't overlap — using union")
        overlap_start = min(w[0] for w in f5_found.values())
        overlap_end = max(w[1] for w in f5_found.values())

    return windows, overlap_start, overlap_end


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
# MAIN EXECUTION
# ==============================================================================
np.random.seed(42)

print(f"\n{'='*W}")
print(f"  SHORT L5-1 vs L5-2: DIRECT F5 MODEL COMPARISON")
print(f"  L5-1: 64 tickers  |  L5-2: 99 tickers (+4 bday shift)")
print(f"  Each model uses its OWN optimizer configs + scaling stats")
print(f"  Slippage: {SLIPPAGE_PCT:.2%} each way ({2*SLIPPAGE_PCT:.2%} RT)")
print(f"{'='*W}")

# ── Phase 0: Derive fair backtest window ──
print(f"\n{'='*W}")
print(f"  PHASE 0: DERIVE FAIR BACKTEST WINDOW")
print(f"{'='*W}")

fold_windows, BACKTEST_START, BACKTEST_END = get_f5_test_windows()

for fam_key, src in FAMILY_F5.items():
    mv = src['bq_mv']
    if mv in fold_windows:
        ws, we = fold_windows[mv]
        print(f"  {fam_key} F5 TEST: {ws} -> {we}")
    else:
        print(f"  {fam_key} F5 TEST: NOT IN BQ (optimizer not run yet)")

if BACKTEST_START is None or BACKTEST_END is None:
    raise RuntimeError("Need both F5 optimizers in BQ to derive fair window")

print(f"\n  >>> FAIR WINDOW: {BACKTEST_START} -> {BACKTEST_END} <<<")
print(f"      (overlap of F5 TEST windows — both out-of-sample)")

# ── Phase 1: Load BQ configs ──
print(f"\n{'='*W}")
print(f"  PHASE 1: LOAD BQ CONFIGS")
print(f"{'='*W}")

family_cfgs = {}
for fam_key, src in FAMILY_F5.items():
    thresh = SENSITIVITY_MIN if fam_key == 'L5-2' else BASE_PNL_THRESHOLD
    cfgs = load_fold_configs(src['bq_mv'], threshold=thresh)
    family_cfgs[fam_key] = cfgs
    deployed = {tk for tk, c in cfgs.items() if c['val_b_avg_pnl'] >= BASE_PNL_THRESHOLD}
    borderline = {tk for tk, c in cfgs.items() if c['val_b_avg_pnl'] < BASE_PNL_THRESHOLD}
    print(f"  {fam_key} ({src['bq_mv']}): {len(deployed)} deployed (>={BASE_PNL_THRESHOLD:.1%})"
          f" + {len(borderline)} borderline (>={thresh:.1%})")
    if deployed:
        print(f"    Deployed:   {', '.join(sorted(deployed))}")
    if borderline:
        bl_detail = ', '.join(f'{tk}={cfgs[tk]["val_b_avg_pnl"]:+.2%}' for tk in sorted(borderline))
        print(f"    Borderline: {bl_detail}")

l1_tickers = set(family_cfgs.get('L5-1', {}).keys())
l2_deployed = {tk for tk, c in family_cfgs.get('L5-2', {}).items()
               if c['val_b_avg_pnl'] >= BASE_PNL_THRESHOLD}
l2_borderline = {tk for tk, c in family_cfgs.get('L5-2', {}).items()
                 if c['val_b_avg_pnl'] < BASE_PNL_THRESHOLD}
l2_all = set(family_cfgs.get('L5-2', {}).keys())

l2_tickers = l2_deployed
shared_tickers = l1_tickers & l2_tickers
only_l1 = l1_tickers - l2_tickers
only_l2 = l2_tickers - l1_tickers
all_tickers = l1_tickers | l2_all

print(f"\n  TICKER SETS (main comparison @ {BASE_PNL_THRESHOLD:.1%}):")
print(f"    L5-1 deployable: {len(l1_tickers)}")
print(f"    L5-2 deployed:   {len(l2_deployed)}")
print(f"    L5-2 borderline: {len(l2_borderline)}")
print(f"    Shared:          {len(shared_tickers)}")
if only_l1:
    print(f"    Only L5-1:       {len(only_l1)} — {', '.join(sorted(only_l1))}")
if only_l2:
    print(f"    Only L5-2 (NEW): {len(only_l2)} — {', '.join(sorted(only_l2))}")
print(f"    Total to score:  {len(all_tickers)} (includes borderline for sensitivity)")

# ── Phase 2: Market context + regime ──
print(f"\n{'='*W}")
print(f"  PHASE 2: MARKET CONTEXT + REGIME")
print(f"{'='*W}")

market_df = build_market_context()
regime_df = compute_regime_features(market_df)

trading_days = get_trading_days(BACKTEST_START, BACKTEST_END)
print(f"  Trading days: {len(trading_days)} ({trading_days[0]} -> {trading_days[-1]})")

# ── Phase 3: Feature engineering ──
print(f"\n{'='*W}")
print(f"  PHASE 3: FEATURE ENGINEERING ({len(all_tickers)} tickers)")
print(f"{'='*W}")

ticker_features = {}
for ticker in sorted(all_tickers):
    df_feat = engineer_asset(ticker, market_df)
    if df_feat is not None:
        ticker_features[ticker] = df_feat
        print(f"    {ticker}: {len(df_feat):,} bars")
    else:
        print(f"    {ticker}: FAILED")

# ── Phase 4: Score + trade for each family ──
print(f"\n{'='*W}")
print(f"  PHASE 4: SCORE + TRADE SIMULATION")
print(f"{'='*W}")

family_trades = {}

for fam_key, src in FAMILY_F5.items():
    print(f"\n  {src['label']}:")
    cfgs = family_cfgs[fam_key]
    if not cfgs:
        print(f"    No configs — skipping")
        family_trades[fam_key] = {}
        continue

    scaling_s, regime_s = load_meta(src['gcs_data'], src['tag'])
    model = load_fold_model(src['gcs_models'], src['tag'], src['fold_id'])

    np.random.seed(42)
    fam_trade_dict = {}
    total_scores = 0
    total_trades = 0

    for ticker in sorted(cfgs.keys()):
        if ticker not in ticker_features:
            continue
        cfg = cfgs[ticker]
        trades = []
        for day in trading_days:
            result = score_ticker_day(
                model, scaling_s, regime_s,
                ticker, cfg, ticker_features[ticker], regime_df, day)
            if result is None:
                continue
            total_scores += 1
            if result['prob'] < cfg['prob_thresh']:
                continue
            entry_idx = result['bar_idx'] + (ENTRY_MINUTE - SCORE_MINUTE)
            trade = simulate_short_trade(ticker_features[ticker], entry_idx, cfg)
            if trade is not None:
                trade['pnl'] -= 2 * SLIPPAGE_PCT
                trades.append({
                    'Date': day.strftime('%Y-%m-%d'),
                    'Ticker': ticker,
                    'Prob': result['prob'],
                    'PnL': trade['pnl'],
                    'Outcome': trade['outcome'],
                    'Entry': trade['entry'],
                    'Exit': trade['exit'],
                    'Mode': cfg['mode'],
                    'Horizon': cfg['horizon_label'],
                    'TP': cfg['tp_pct'],
                    'SL': cfg['sl_pct'],
                })
                total_trades += 1
        if trades:
            fam_trade_dict[ticker] = trades

    family_trades[fam_key] = fam_trade_dict
    print(f"    Scores: {total_scores}  Trades: {total_trades}  Tickers with trades: {len(fam_trade_dict)}")

    del model
    keras.backend.clear_session()
    gc.collect()


# ==============================================================================
# RESULTS
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  RESULTS: L5-1 vs L5-2 F5 HEAD-TO-HEAD")
print(f"  Period: {BACKTEST_START} -> {BACKTEST_END} ({len(trading_days)} days)")
print(f"{'='*W}")

# ── Aggregate stats ──
def agg_stats(trade_dict):
    all_trades = [t for trades in trade_dict.values() for t in trades]
    if not all_trades:
        return {'n': 0, 'n_tickers': 0, 'avg_pnl': 0, 'total_pnl': 0,
                'wr': 0, 'avg_win': 0, 'avg_loss': 0, 'pf': 0,
                'tp_count': 0, 'sl_count': 0, 'eod_count': 0}
    df = pd.DataFrame(all_trades)
    n = len(df)
    nw = (df['PnL'] > 0).sum()
    nl = n - nw
    avg_pnl = df['PnL'].mean()
    total_pnl = df.groupby('Date')['PnL'].mean().sum()
    wr = nw / n
    avg_win = df.loc[df['PnL'] > 0, 'PnL'].mean() if nw > 0 else 0
    avg_loss = df.loc[df['PnL'] <= 0, 'PnL'].mean() if nl > 0 else 0
    pf = abs(avg_win * nw / (avg_loss * nl)) if nl > 0 and avg_loss != 0 else float('inf')
    tp_count = (df['Outcome'] == 'TP').sum()
    sl_count = (df['Outcome'] == 'SL').sum()
    eod_count = (df['Outcome'] == 'EOD').sum()
    return {'n': n, 'n_tickers': df['Ticker'].nunique(), 'avg_pnl': avg_pnl,
            'total_pnl': total_pnl, 'wr': wr, 'avg_win': avg_win, 'avg_loss': avg_loss,
            'pf': pf, 'tp_count': tp_count, 'sl_count': sl_count, 'eod_count': eod_count}

l2_deployed_trades = {tk: trades for tk, trades in family_trades.get('L5-2', {}).items()
                      if tk in l2_deployed}
stats = {}
stats['L5-1'] = agg_stats(family_trades.get('L5-1', {}))
stats['L5-2'] = agg_stats(l2_deployed_trades)

shared_stats = {}
for fam_key in FAMILY_F5:
    shared_trade_dict = {tk: family_trades[fam_key].get(tk, [])
                         for tk in shared_tickers if family_trades[fam_key].get(tk)}
    shared_stats[fam_key] = agg_stats(shared_trade_dict)

def _print_comparison_table(label, s1, s2):
    print(f"\n  {label}")
    print(f"  {'Metric':<25} {'L5-1':>12} {'L5-2':>12} {'Delta':>12}")
    print(f"  {'─'*63}")
    metrics = [
        ('Tickers w/ trades', f"{s1['n_tickers']}", f"{s2['n_tickers']}", f"{s2['n_tickers'] - s1['n_tickers']:+d}"),
        ('Total trades', f"{s1['n']}", f"{s2['n']}", f"{s2['n'] - s1['n']:+d}"),
        ('Avg PnL/trade', f"{s1['avg_pnl']:+.2%}", f"{s2['avg_pnl']:+.2%}",
         f"{s2['avg_pnl'] - s1['avg_pnl']:+.2%}"),
        ('Total EqW PnL', f"{s1['total_pnl']:+.2%}", f"{s2['total_pnl']:+.2%}",
         f"{s2['total_pnl'] - s1['total_pnl']:+.2%}"),
        ('Win rate', f"{s1['wr']:.0%}", f"{s2['wr']:.0%}",
         f"{s2['wr'] - s1['wr']:+.0%}"),
        ('Avg win', f"{s1['avg_win']:+.2%}", f"{s2['avg_win']:+.2%}", ''),
        ('Avg loss', f"{s1['avg_loss']:+.2%}", f"{s2['avg_loss']:+.2%}", ''),
        ('Profit factor', f"{s1['pf']:.2f}" if s1['pf'] < 999 else 'inf',
         f"{s2['pf']:.2f}" if s2['pf'] < 999 else 'inf', ''),
        ('TP / SL / EOD', f"{s1['tp_count']}/{s1['sl_count']}/{s1['eod_count']}",
         f"{s2['tp_count']}/{s2['sl_count']}/{s2['eod_count']}", ''),
    ]
    for label_m, v1, v2, delta in metrics:
        print(f"  {label_m:<25} {v1:>12} {v2:>12} {delta:>12}")

s1, s2 = stats['L5-1'], stats['L5-2']
ss1, ss2 = shared_stats['L5-1'], shared_stats['L5-2']

_print_comparison_table(
    f"FULL PORTFOLIO (each model's entire deployable set)",
    s1, s2)

_print_comparison_table(
    f"SHARED TICKERS ONLY ({len(shared_tickers)} tickers — pure model quality comparison)",
    ss1, ss2)

# ── Per-ticker table: SHARED tickers ──
if shared_tickers:
    print(f"\n\n  SHARED TICKERS ({len(shared_tickers)}) — same ticker, different model")
    print(f"  {'Ticker':<7} │{'L5-1 N':>5} {'AvgPnL':>8} {'TotPnL':>8} {'WR':>5} {'Mode':>7}"
          f" │{'L5-2 N':>5} {'AvgPnL':>8} {'TotPnL':>8} {'WR':>5} {'Mode':>7}"
          f" │{'Δ Avg':>7} {'Win':>5}")
    print(f"  {'─'*7}─┼{'─'*36}─┼{'─'*36}─┼{'─'*14}")

    tk_l1_wins = 0; tk_l2_wins = 0
    shared_rows = []
    for tk in sorted(shared_tickers):
        t1 = family_trades['L5-1'].get(tk, [])
        t2 = family_trades['L5-2'].get(tk, [])
        n1 = len(t1); n2 = len(t2)
        avg1 = np.mean([t['PnL'] for t in t1]) if t1 else 0
        avg2 = np.mean([t['PnL'] for t in t2]) if t2 else 0
        tot1 = sum(t['PnL'] for t in t1)
        tot2 = sum(t['PnL'] for t in t2)
        wr1 = sum(1 for t in t1 if t['PnL'] > 0) / n1 if n1 > 0 else 0
        wr2 = sum(1 for t in t2 if t['PnL'] > 0) / n2 if n2 > 0 else 0
        mode1 = family_cfgs['L5-1'][tk]['mode'] if tk in family_cfgs['L5-1'] else ''
        mode2 = family_cfgs['L5-2'][tk]['mode'] if tk in family_cfgs['L5-2'] else ''
        delta = avg2 - avg1
        winner = 'L5-2' if delta > 0.001 else ('L5-1' if delta < -0.001 else 'TIE')
        if n1 == 0 and n2 == 0:
            winner = '--'
        if delta > 0.001:
            tk_l2_wins += 1
        elif delta < -0.001:
            tk_l1_wins += 1
        shared_rows.append((tk, delta))
        a1 = f"{avg1:+.2%}" if n1 > 0 else '---'
        a2 = f"{avg2:+.2%}" if n2 > 0 else '---'
        t1s = f"{tot1:+.2%}" if n1 > 0 else '---'
        t2s = f"{tot2:+.2%}" if n2 > 0 else '---'
        w1s = f"{wr1:.0%}" if n1 > 0 else '---'
        w2s = f"{wr2:.0%}" if n2 > 0 else '---'
        print(f"  {tk:<7} │{n1:>5} {a1:>8} {t1s:>8} {w1s:>5} {mode1:>7}"
              f" │{n2:>5} {a2:>8} {t2s:>8} {w2s:>5} {mode2:>7}"
              f" │{delta:>+6.2%} {winner:>5}")

    # Shared tickers aggregate
    shared_l1_trades = [t for tk in shared_tickers for t in family_trades['L5-1'].get(tk, [])]
    shared_l2_trades = [t for tk in shared_tickers for t in family_trades['L5-2'].get(tk, [])]
    shared_l1_avg = np.mean([t['PnL'] for t in shared_l1_trades]) if shared_l1_trades else 0
    shared_l2_avg = np.mean([t['PnL'] for t in shared_l2_trades]) if shared_l2_trades else 0
    print(f"  {'─'*7}─┼{'─'*36}─┼{'─'*36}─┼{'─'*14}")
    print(f"  {'TOTAL':<7} │{len(shared_l1_trades):>5} {shared_l1_avg:>+7.2%}"
          f"{'':>22} │{len(shared_l2_trades):>5} {shared_l2_avg:>+7.2%}"
          f"{'':>22} │{shared_l2_avg - shared_l1_avg:>+6.2%}")
    print(f"\n  Shared ticker tally: L5-1 wins {tk_l1_wins}  L5-2 wins {tk_l2_wins}"
          f"  Ties {len(shared_tickers) - tk_l1_wins - tk_l2_wins}")

# ── L5-2 ONLY tickers (expanded universe) ──
if only_l2:
    print(f"\n\n  L5-2 ONLY TICKERS ({len(only_l2)}) — new from expanded training")
    print(f"  {'Ticker':<7} {'N':>4} {'AvgPnL':>8} {'TotPnL':>8} {'WR':>5}"
          f" {'Mode':>7} {'Horizon':>8} {'TP':>5} {'SL':>5} {'OptPnL':>8}")
    print(f"  {'─'*75}")
    new_all_trades = []
    for tk in sorted(only_l2):
        trades = family_trades['L5-2'].get(tk, [])
        cfg = family_cfgs['L5-2'].get(tk, {})
        n = len(trades)
        if n > 0:
            avg = np.mean([t['PnL'] for t in trades])
            tot = sum(t['PnL'] for t in trades)
            wr = sum(1 for t in trades if t['PnL'] > 0) / n
            new_all_trades.extend(trades)
        else:
            avg = tot = wr = 0
        mode = cfg.get('mode', '')
        horizon = cfg.get('horizon_label', '')
        tp = cfg.get('tp_pct', 0)
        sl = cfg.get('sl_pct', 0)
        opt_pnl = cfg.get('val_b_avg_pnl', 0)
        print(f"  {tk:<7} {n:>4} {avg:>+7.2%} {tot:>+7.2%} {wr:>4.0%}"
              f" {mode:>7} {horizon:>8} {tp:>4.0%} {sl:>4.0%} {opt_pnl:>+7.2%}")
    if new_all_trades:
        new_avg = np.mean([t['PnL'] for t in new_all_trades])
        new_tot = sum(t['PnL'] for t in new_all_trades)
        print(f"  {'─'*75}")
        print(f"  {'TOTAL':<7} {len(new_all_trades):>4} {new_avg:>+7.2%} {new_tot:>+7.2%}")
        print(f"\n  New tickers contribute {len(new_all_trades)} extra trades"
              f" at {new_avg:+.2%} avg PnL")

# ── L5-1 ONLY tickers (lost in L5-2) ──
if only_l1:
    print(f"\n\n  L5-1 ONLY TICKERS ({len(only_l1)}) — deployable in L5-1 but not L5-2")
    print(f"  {'Ticker':<7} {'N':>4} {'AvgPnL':>8} {'TotPnL':>8} {'WR':>5}")
    print(f"  {'─'*40}")
    for tk in sorted(only_l1):
        trades = family_trades['L5-1'].get(tk, [])
        n = len(trades)
        if n > 0:
            avg = np.mean([t['PnL'] for t in trades])
            tot = sum(t['PnL'] for t in trades)
            wr = sum(1 for t in trades if t['PnL'] > 0) / n
            print(f"  {tk:<7} {n:>4} {avg:>+7.2%} {tot:>+7.2%} {wr:>4.0%}")
        else:
            print(f"  {tk:<7} {0:>4}     ---      ---   ---")

# ── Config comparison for shared tickers ──
if shared_tickers:
    print(f"\n\n  CONFIG COMPARISON (shared tickers — what the optimizer chose differently)")
    print(f"  {'Ticker':<7} │{'L5-1':^25}│{'L5-2':^25}│ {'Changed?'}")
    print(f"  {'':>7} │{'Mode':>7} {'Thr':>5} {'TP':>4} {'SL':>4} {'Hz':>4}"
          f" │{'Mode':>7} {'Thr':>5} {'TP':>4} {'SL':>4} {'Hz':>4} │")
    print(f"  {'─'*7}─┼{'─'*25}┼{'─'*25}┼{'─'*10}")
    n_changed = 0
    for tk in sorted(shared_tickers):
        c1 = family_cfgs['L5-1'][tk]
        c2 = family_cfgs['L5-2'][tk]
        changed = (c1['mode'] != c2['mode'] or
                   abs(c1['prob_thresh'] - c2['prob_thresh']) > 0.001 or
                   abs(c1['tp_pct'] - c2['tp_pct']) > 0.001 or
                   abs(c1['sl_pct'] - c2['sl_pct']) > 0.001 or
                   c1['horizon_label'] != c2['horizon_label'])
        if changed:
            n_changed += 1
        flag = '  ***' if changed else ''
        print(f"  {tk:<7} │{c1['mode']:>7} {c1['prob_thresh']:>.2f} {c1['tp_pct']:>3.0%}"
              f" {c1['sl_pct']:>3.0%} {c1['horizon_label']:>4}"
              f" │{c2['mode']:>7} {c2['prob_thresh']:>.2f} {c2['tp_pct']:>3.0%}"
              f" {c2['sl_pct']:>3.0%} {c2['horizon_label']:>4} │{flag}")
    print(f"\n  {n_changed}/{len(shared_tickers)} tickers changed config between L5-1 and L5-2")


# ==============================================================================
# THRESHOLD SENSITIVITY — L5-2 at different deployment cutoffs
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  THRESHOLD SENSITIVITY — what if we loosen L5-2 deployment?")
print(f"{'='*W}")

print(f"\n  {'Threshold':>10} {'Tickers':>8} {'Trades':>7} {'AvgPnL':>8} {'EqW':>8}"
      f" {'WR':>5} {'PF':>6} {'vs L5-1 Avg':>12} {'vs L5-1 EqW':>12}")
print(f"  {'─'*90}")

l5_1_ref = stats['L5-1']
sensitivity_results = {}

for thresh in SENSITIVITY_THRESHOLDS:
    tks_at = {tk for tk, c in family_cfgs['L5-2'].items()
              if c['val_b_avg_pnl'] >= thresh}
    trades_at = {tk: trades for tk, trades in family_trades.get('L5-2', {}).items()
                 if tk in tks_at}
    st = agg_stats(trades_at)
    sensitivity_results[thresh] = st

    delta_avg = st['avg_pnl'] - l5_1_ref['avg_pnl'] if st['n'] > 0 and l5_1_ref['n'] > 0 else 0
    delta_eqw = st['total_pnl'] - l5_1_ref['total_pnl'] if st['n'] > 0 and l5_1_ref['n'] > 0 else 0
    pf_str = f"{st['pf']:.1f}" if st['pf'] < 999 else 'inf'
    marker = ' ◄── current' if thresh == BASE_PNL_THRESHOLD else ''
    print(f"  {thresh:>9.1%} {st['n_tickers']:>8} {st['n']:>7} {st['avg_pnl']:>+7.2%}"
          f" {st['total_pnl']:>+7.2%} {st['wr']:>4.0%} {pf_str:>6}"
          f" {delta_avg:>+11.2%} {delta_eqw:>+11.2%}{marker}")

    if thresh < BASE_PNL_THRESHOLD:
        added_tks = tks_at - l2_deployed
        added_trades = {tk: trades for tk, trades in family_trades.get('L5-2', {}).items()
                        if tk in added_tks}
        added_all = [t for trades in added_trades.values() for t in trades]
        if added_all:
            added_avg = np.mean([t['PnL'] for t in added_all])
            print(f"  {'':>10} +{len(added_tks)} new tickers add {len(added_all)} trades at {added_avg:+.2%} avg"
                  f" ({', '.join(sorted(added_tks))})")

print(f"\n  L5-1 reference: {l5_1_ref['n_tickers']} tickers, {l5_1_ref['n']} trades,"
      f" Avg {l5_1_ref['avg_pnl']:+.2%}, EqW {l5_1_ref['total_pnl']:+.2%}")


# ==============================================================================
# COMBINED PORTFOLIO — best of both models
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  COMBINED PORTFOLIO — L5-1 + L5-2 together")
print(f"{'='*W}")

def build_combined(l1_trades, l2_trades, l1_cfgs, l2_cfgs, l2_ticker_set, strategy='union'):
    """Build combined trade dict from two models.
    Strategies:
      union: all unique + best-BT model for shared
      consensus: shared tickers only, both must fire same day
    """
    combined = {}
    combined_source = {}
    l1_set = set(l1_cfgs.keys())
    l2_set = l2_ticker_set
    shared = l1_set & l2_set

    if strategy == 'union':
        for tk in l1_set - shared:
            if l1_trades.get(tk):
                combined[tk] = l1_trades[tk]
                combined_source[tk] = 'L5-1'
        for tk in l2_set - shared:
            if l2_trades.get(tk):
                combined[tk] = l2_trades[tk]
                combined_source[tk] = 'L5-2'
        for tk in shared:
            t1 = l1_trades.get(tk, [])
            t2 = l2_trades.get(tk, [])
            avg1 = np.mean([t['PnL'] for t in t1]) if t1 else -999
            avg2 = np.mean([t['PnL'] for t in t2]) if t2 else -999
            if avg2 >= avg1:
                combined[tk] = t2
                combined_source[tk] = 'L5-2'
            else:
                combined[tk] = t1
                combined_source[tk] = 'L5-1'
    elif strategy == 'consensus':
        for tk in shared:
            t1_dates = {t['Date'] for t in l1_trades.get(tk, [])}
            t2_dates = {t['Date'] for t in l2_trades.get(tk, [])}
            both_dates = t1_dates & t2_dates
            if both_dates:
                t1_map = {t['Date']: t for t in l1_trades.get(tk, [])}
                t2_map = {t['Date']: t for t in l2_trades.get(tk, [])}
                avg1 = np.mean([t['PnL'] for t in l1_trades.get(tk, [])]) if l1_trades.get(tk) else -999
                avg2 = np.mean([t['PnL'] for t in l2_trades.get(tk, [])]) if l2_trades.get(tk) else -999
                source = 'L5-2' if avg2 >= avg1 else 'L5-1'
                use_map = t2_map if source == 'L5-2' else t1_map
                combined[tk] = [use_map[d] for d in sorted(both_dates) if d in use_map]
                combined_source[tk] = source

    return combined, combined_source

l1_tr = family_trades.get('L5-1', {})
l2_tr = family_trades.get('L5-2', {})
l1_cf = family_cfgs.get('L5-1', {})
l2_cf = family_cfgs.get('L5-2', {})

strategies = OrderedDict()

strategies['UNION @ 1.8%'] = build_combined(l1_tr, l2_tr, l1_cf, l2_cf, l2_deployed, 'union')

l2_at_12 = {tk for tk, c in l2_cf.items() if c['val_b_avg_pnl'] >= 0.012}
strategies['UNION @ 1.2%'] = build_combined(l1_tr, l2_tr, l1_cf, l2_cf, l2_at_12, 'union')

l2_at_08 = {tk for tk, c in l2_cf.items() if c['val_b_avg_pnl'] >= 0.008}
strategies['UNION @ 0.8%'] = build_combined(l1_tr, l2_tr, l1_cf, l2_cf, l2_at_08, 'union')

strategies['CONSENSUS'] = build_combined(l1_tr, l2_tr, l1_cf, l2_cf, l2_deployed, 'consensus')

print(f"\n  {'Strategy':<20} {'Tickers':>8} {'Trades':>7} {'AvgPnL':>8} {'EqW':>8}"
      f" {'WR':>5} {'PF':>6} {'vs L5-1':>8} {'vs L5-2':>8}")
print(f"  {'─'*88}")

for strat_name, (combined_trades, combined_source) in strategies.items():
    st = agg_stats(combined_trades)
    d1 = st['avg_pnl'] - l5_1_ref['avg_pnl'] if st['n'] > 0 else 0
    d2 = st['avg_pnl'] - stats['L5-2']['avg_pnl'] if st['n'] > 0 else 0
    pf_str = f"{st['pf']:.1f}" if st['pf'] < 999 else 'inf'
    print(f"  {strat_name:<20} {st['n_tickers']:>8} {st['n']:>7} {st['avg_pnl']:>+7.2%}"
          f" {st['total_pnl']:>+7.2%} {st['wr']:>4.0%} {pf_str:>6}"
          f" {d1:>+7.2%} {d2:>+7.2%}")

print(f"\n  Reference:")
print(f"    L5-1 alone: {l5_1_ref['n_tickers']}tk, {l5_1_ref['n']}t,"
      f" Avg {l5_1_ref['avg_pnl']:+.2%}, EqW {l5_1_ref['total_pnl']:+.2%}")
print(f"    L5-2 alone: {stats['L5-2']['n_tickers']}tk, {stats['L5-2']['n']}t,"
      f" Avg {stats['L5-2']['avg_pnl']:+.2%}, EqW {stats['L5-2']['total_pnl']:+.2%}")

best_strat = max(strategies.items(), key=lambda x: agg_stats(x[1][0])['total_pnl'])
best_st = agg_stats(best_strat[1][0])
print(f"\n  >>> Best combined: {best_strat[0]} — {best_st['n_tickers']} tickers,"
      f" {best_st['n']} trades, EqW {best_st['total_pnl']:+.2%} <<<")

best_combined_trades, best_combined_source = best_strat[1]
print(f"\n  Per-ticker assignment ({best_strat[0]}):")
print(f"  {'Ticker':<7} {'Source':>6} {'N':>4} {'AvgPnL':>8} {'OptPnL':>8}")
print(f"  {'─'*40}")
for tk in sorted(best_combined_trades.keys()):
    trades = best_combined_trades[tk]
    src = best_combined_source[tk]
    n = len(trades)
    avg = np.mean([t['PnL'] for t in trades]) if trades else 0
    opt_pnl = l1_cf[tk]['val_b_avg_pnl'] if src == 'L5-1' else l2_cf[tk]['val_b_avg_pnl']
    print(f"  {tk:<7} {src:>6} {n:>4} {avg:>+7.2%} {opt_pnl:>+7.2%}")


# ==============================================================================
# CHARTS
# ==============================================================================
if HAS_MPL:
    n_fam = len(FAMILY_F5)
    fig, axes = plt.subplots(2, 3, figsize=(30, 16))
    fig.suptitle(f'SHORT L5-1 vs L5-2: Model Comparison + Combined Analysis\n'
                 f'{BACKTEST_START} to {BACKTEST_END} | Slippage {2*SLIPPAGE_PCT:.2%} RT',
                 fontsize=14, fontweight='bold')

    # 1. Per-ticker avg PnL (shared tickers only, paired bars)
    ax = axes[0, 0]
    if shared_tickers:
        tk_sorted = sorted(shared_tickers,
                           key=lambda tk: np.mean([t['PnL'] for t in family_trades['L5-2'].get(tk, [])]) -
                                          np.mean([t['PnL'] for t in family_trades['L5-1'].get(tk, [])]),
                           reverse=True)
        y_pos = np.arange(len(tk_sorted))
        bar_h = 0.35
        for i, fam_key in enumerate(FAMILY_F5):
            vals = []
            for tk in tk_sorted:
                trades = family_trades[fam_key].get(tk, [])
                vals.append(np.mean([t['PnL'] for t in trades]) * 100 if trades else 0)
            ax.barh(y_pos + i * bar_h, vals, bar_h,
                    label=FAMILY_F5[fam_key]['label'],
                    color=FAMILY_F5[fam_key]['color'], alpha=0.85,
                    edgecolor='black', linewidth=0.5)
        ax.set_yticks(y_pos + bar_h / 2)
        ax.set_yticklabels(tk_sorted, fontsize=7)
        ax.axvline(0, color='black', linewidth=1)
        ax.axvline(BASE_PNL_THRESHOLD * 100, color='green', linewidth=1,
                   linestyle='--', alpha=0.5, label=f'{BASE_PNL_THRESHOLD:.1%}')
        ax.invert_yaxis()
    ax.set_xlabel('Avg PnL per Trade (%)')
    ax.set_title('Shared Tickers: Avg PnL (sorted by L5-2 advantage)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='x')

    # 2. PnL delta per shared ticker (waterfall)
    ax = axes[0, 1]
    if shared_tickers:
        deltas_sorted = sorted(
            [(tk, (np.mean([t['PnL'] for t in family_trades['L5-2'].get(tk, [])]) if family_trades['L5-2'].get(tk) else 0) -
                  (np.mean([t['PnL'] for t in family_trades['L5-1'].get(tk, [])]) if family_trades['L5-1'].get(tk) else 0))
             for tk in shared_tickers],
            key=lambda x: -x[1])
        tks = [x[0] for x in deltas_sorted]
        dvals = [x[1] * 100 for x in deltas_sorted]
        colors = ['#2E7D32' if d > 0 else '#D32F2F' for d in dvals]
        ax.barh(range(len(tks)), dvals, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(len(tks)))
        ax.set_yticklabels(tks, fontsize=7)
        ax.axvline(0, color='black', linewidth=1)
        ax.invert_yaxis()
    ax.set_xlabel('L5-2 minus L5-1 Avg PnL (%)')
    ax.set_title('Per-Ticker: L5-2 Improvement (green = L5-2 better)')
    ax.grid(alpha=0.3, axis='x')

    # 3. Equity curves (daily EqW)
    ax = axes[1, 0]
    for fam_key, src in FAMILY_F5.items():
        all_trades = [t for trades in family_trades[fam_key].values() for t in trades]
        if not all_trades:
            continue
        tdf = pd.DataFrame(all_trades)
        daily_eqw = tdf.groupby('Date')['PnL'].mean().sort_index()
        cum = daily_eqw.cumsum() * 100
        dates_plot = pd.to_datetime(cum.index)
        ax.plot(dates_plot, cum.values, color=src['color'], linewidth=2.5,
                label=f"{src['label']} ({len(all_trades)}t, EqW {cum.iloc[-1]:+.1f}%)")
    ax.axhline(0, color='black', linewidth=0.8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax.set_ylabel('Cumulative EqW PnL (%)')
    ax.set_title('Equity Curves (daily equal-weight)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # 4. Trade count + new tickers
    ax = axes[1, 1]
    fam_keys = list(FAMILY_F5.keys())
    x = np.arange(3)
    bar_w = 0.35
    cats = ['Shared\nTickers', 'Unique\nTickers', 'Total\nTrades']
    for i, fam_key in enumerate(FAMILY_F5):
        src = FAMILY_F5[fam_key]
        shared_n = sum(len(family_trades[fam_key].get(tk, [])) for tk in shared_tickers)
        unique_tks = only_l1 if fam_key == 'L5-1' else only_l2
        unique_n = sum(len(family_trades[fam_key].get(tk, [])) for tk in unique_tks)
        total_n = sum(len(trades) for trades in family_trades[fam_key].values())
        vals = [shared_n, unique_n, total_n]
        bars = ax.bar(x + i * bar_w, vals, bar_w, label=src['label'],
                      color=src['color'], alpha=0.85, edgecolor='black', linewidth=0.5)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        str(v), ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(x + bar_w / 2)
    ax.set_xticklabels(cats, fontsize=10)
    ax.set_ylabel('Number of Trades')
    ax.set_title('Trade Volume: Shared vs New Tickers')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis='y')

    # 5. Threshold sensitivity — bar chart
    ax = axes[0, 2]
    thresh_labels = [f'{t:.1%}' for t in SENSITIVITY_THRESHOLDS]
    thresh_avg = [sensitivity_results[t]['avg_pnl'] * 100 for t in SENSITIVITY_THRESHOLDS]
    thresh_n = [sensitivity_results[t]['n'] for t in SENSITIVITY_THRESHOLDS]
    colors_t = ['#1565C0' if t == BASE_PNL_THRESHOLD else '#FF6F00' for t in SENSITIVITY_THRESHOLDS]
    bars = ax.bar(thresh_labels, thresh_avg, color=colors_t, alpha=0.85,
                  edgecolor='black', linewidth=0.5)
    ax.axhline(l5_1_ref['avg_pnl'] * 100, color='red', linewidth=2,
               linestyle='--', label=f"L5-1 ref ({l5_1_ref['avg_pnl']:+.2%})")
    for bar, n, v in zip(bars, thresh_n, thresh_avg):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{n}t', ha='center', fontsize=8, fontweight='bold')
    ax.set_xlabel('L5-2 Deploy Threshold')
    ax.set_ylabel('Avg PnL per Trade (%)')
    ax.set_title('L5-2 Threshold Sensitivity')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis='y')

    # 6. Combined portfolio equity curves
    ax = axes[1, 2]
    for fam_key, src in FAMILY_F5.items():
        fam_trades_filtered = family_trades[fam_key] if fam_key == 'L5-1' else l2_deployed_trades
        all_trades = [t for trades in fam_trades_filtered.values() for t in trades]
        if not all_trades:
            continue
        tdf = pd.DataFrame(all_trades)
        daily_eqw = tdf.groupby('Date')['PnL'].mean().sort_index()
        cum = daily_eqw.cumsum() * 100
        dates_plot = pd.to_datetime(cum.index)
        ax.plot(dates_plot, cum.values, color=src['color'], linewidth=1.5,
                linestyle='--', alpha=0.5,
                label=f"{src['label']} alone ({cum.iloc[-1]:+.1f}%)")
    comb_colors = {'UNION @ 1.8%': '#2E7D32', 'UNION @ 1.2%': '#8BC34A',
                   'UNION @ 0.8%': '#CDDC39', 'CONSENSUS': '#9C27B0'}
    for strat_name, (comb_trades, _) in strategies.items():
        all_trades = [t for trades in comb_trades.values() for t in trades]
        if not all_trades:
            continue
        tdf = pd.DataFrame(all_trades)
        daily_eqw = tdf.groupby('Date')['PnL'].mean().sort_index()
        cum = daily_eqw.cumsum() * 100
        dates_plot = pd.to_datetime(cum.index)
        c = comb_colors.get(strat_name, '#333333')
        ax.plot(dates_plot, cum.values, color=c, linewidth=2.5,
                label=f"{strat_name} ({cum.iloc[-1]:+.1f}%)")
    ax.axhline(0, color='black', linewidth=0.8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax.set_ylabel('Cumulative EqW PnL (%)')
    ax.set_title('Combined Portfolio Equity Curves')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    save_path = '/tmp/SHORT_L5_1_vs_L5_2_f5_comparison.png'
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

print(f"\n  ── FULL PORTFOLIO (each model's entire deployable set) ──")
print(f"  L5-1: {s1['n']} trades across {s1['n_tickers']} tickers"
      f" | Avg {s1['avg_pnl']:+.2%} | EqW {s1['total_pnl']:+.2%} | WR {s1['wr']:.0%}")
print(f"  L5-2: {s2['n']} trades across {s2['n_tickers']} tickers"
      f" | Avg {s2['avg_pnl']:+.2%} | EqW {s2['total_pnl']:+.2%} | WR {s2['wr']:.0%}")

if s1['n'] > 0 and s2['n'] > 0:
    avg_delta = s2['avg_pnl'] - s1['avg_pnl']
    eqw_delta = s2['total_pnl'] - s1['total_pnl']
    extra_trades = s2['n'] - s1['n']

    print(f"\n  Impact of retraining (full portfolio):")
    print(f"    Avg PnL change:   {avg_delta:+.2%}")
    print(f"    EqW PnL change:   {eqw_delta:+.2%}")
    print(f"    Trade count change: {extra_trades:+d}")
    if only_l2:
        new_trades = sum(len(family_trades['L5-2'].get(tk, [])) for tk in only_l2)
        print(f"    New tickers:      {len(only_l2)} ({new_trades} extra trades)")
    if only_l1:
        lost_trades = sum(len(family_trades['L5-1'].get(tk, [])) for tk in only_l1)
        print(f"    Lost tickers:     {len(only_l1)} ({lost_trades} trades lost)")

print(f"\n  ── SHARED TICKERS ONLY ({len(shared_tickers)} — pure model quality) ──")
print(f"  L5-1: {ss1['n']} trades across {ss1['n_tickers']} tickers"
      f" | Avg {ss1['avg_pnl']:+.2%} | EqW {ss1['total_pnl']:+.2%} | WR {ss1['wr']:.0%}")
print(f"  L5-2: {ss2['n']} trades across {ss2['n_tickers']} tickers"
      f" | Avg {ss2['avg_pnl']:+.2%} | EqW {ss2['total_pnl']:+.2%} | WR {ss2['wr']:.0%}")

if ss1['n'] > 0 and ss2['n'] > 0:
    shared_avg_delta = ss2['avg_pnl'] - ss1['avg_pnl']
    shared_eqw_delta = ss2['total_pnl'] - ss1['total_pnl']
    print(f"    Avg PnL change:   {shared_avg_delta:+.2%}")
    print(f"    EqW PnL change:   {shared_eqw_delta:+.2%}")

print(f"\n  ── HEAD-TO-HEAD DECISION ──")
full_avg_delta = s2['avg_pnl'] - s1['avg_pnl'] if s1['n'] > 0 and s2['n'] > 0 else 0
full_eqw_delta = s2['total_pnl'] - s1['total_pnl'] if s1['n'] > 0 and s2['n'] > 0 else 0
shared_avg_delta = ss2['avg_pnl'] - ss1['avg_pnl'] if ss1['n'] > 0 and ss2['n'] > 0 else 0

full_better = full_avg_delta > 0.002
shared_better = shared_avg_delta > 0.002
full_worse = full_avg_delta < -0.002
shared_worse = shared_avg_delta < -0.002

if full_better and shared_better:
    print(f"  L5-2 WINS BOTH: full portfolio +{full_avg_delta:.2%}, shared +{shared_avg_delta:.2%}")
elif full_worse and shared_worse:
    print(f"  L5-1 HOLDS BOTH: full portfolio {full_avg_delta:+.2%}, shared {shared_avg_delta:+.2%}")
elif shared_better and not full_better:
    print(f"  MIXED: L5-2 better on shared tickers ({shared_avg_delta:+.2%}) "
          f"but full portfolio {full_avg_delta:+.2%} (EqW {full_eqw_delta:+.2%})")
elif full_better and not shared_better:
    print(f"  MIXED: L5-2 full portfolio better ({full_avg_delta:+.2%}, EqW {full_eqw_delta:+.2%}) "
          f"but shared tickers {shared_avg_delta:+.2%}")
else:
    print(f"  NEUTRAL: full {full_avg_delta:+.2%}, shared {shared_avg_delta:+.2%} — within noise")

print(f"\n  ── THRESHOLD SENSITIVITY ──")
best_thresh = max(sensitivity_results.items(),
                  key=lambda x: x[1]['total_pnl'] if x[1]['n'] > 0 else -999)
bt = best_thresh[1]
print(f"  Best L5-2 threshold: {best_thresh[0]:.1%} → {bt['n_tickers']} tickers,"
      f" {bt['n']} trades, EqW {bt['total_pnl']:+.2%}")
if best_thresh[0] < BASE_PNL_THRESHOLD:
    print(f"  Loosening from {BASE_PNL_THRESHOLD:.1%} to {best_thresh[0]:.1%} adds"
          f" {bt['n'] - stats['L5-2']['n']} trades, EqW {bt['total_pnl'] - stats['L5-2']['total_pnl']:+.2%}")

print(f"\n  ── COMBINED PORTFOLIO ──")
best_st = agg_stats(best_strat[1][0])
print(f"  Best strategy: {best_strat[0]}")
print(f"    {best_st['n_tickers']} tickers, {best_st['n']} trades,"
      f" Avg {best_st['avg_pnl']:+.2%}, EqW {best_st['total_pnl']:+.2%}")
vs_l1 = best_st['total_pnl'] - l5_1_ref['total_pnl']
vs_l2 = best_st['total_pnl'] - stats['L5-2']['total_pnl']
print(f"    vs L5-1 alone: EqW {vs_l1:+.2%}  |  vs L5-2 alone: EqW {vs_l2:+.2%}")

if vs_l1 > 0.02 or vs_l2 > 0.02:
    print(f"\n  >>> RECOMMEND: {best_strat[0]} — combining models beats either alone <<<")
else:
    print(f"\n  >>> Combining models does not materially beat the best individual model <<<")

print(f"\n{'='*W}")
