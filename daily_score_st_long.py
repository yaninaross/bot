"""
Daily Scoring — Spike Transformer LONG F5 (frozen configs)
==========================================================
Self-contained. Scores deployed tickers under ST LONG fold 5 and prints dashboard.

Strategy: ST LONG fold 5 (7 tickers, frozen configs from BQ)
  - 1 model (fold 5), 7 deployed tickers
  - ACHR, CLSK, MARA, ONDS, PL, RIOT, SOUN
  - Score at minute 375 (6:15 AM PT), entry at minute 385 (6:25 AM PT)
  - Modes: static, grace, limit_entry
  - FiLM regime conditioning (no separate regime filter)

Run AFTER daily_stitch_master.py has finished updating GCS.
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
import keras
from keras.layers import Dense, Dropout, LayerNormalization
import h5py
from google.cloud import storage as gcs_storage

if tf.config.list_physical_devices('GPU'):
    keras.mixed_precision.set_dtype_policy('mixed_float16')

try:
    from google.cloud import bigquery
except Exception:
    bigquery = None

# ==============================================================================
# CONSTANTS
# ==============================================================================
BUCKET_NAME    = "time-series-bucket-ys"
BQ_TABLE       = "ys-time-series.scoring_logs.live_predictions"
INFERENCE_DIR  = "Data/Inference_v2"
TAIL_DAYS      = 25
PST            = pytz.timezone('America/Los_Angeles')

TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

MICRO_COLS = ['Spread', 'OBI', 'Volume_Delta', 'Trade_Count', 'Delta_Pct',
              'bsize', 'asize', 'Total_Vol']

gcs_client = gcs_storage.Client()


# ==============================================================================
# DATA LOADING
# ==============================================================================
_master_cache = {}


_NEEDED_COLS = {'Open', 'High', 'Low', 'Close', 'Volume',
                'open', 'high', 'low', 'close', 'volume',
                'Spread', 'OBI', 'Volume_Delta', 'Trade_Count', 'Delta_Pct',
                'bsize', 'asize', 'Total_Vol',
                'Total_vol', 'Trade_count', 'Volume_delta', 'Delta_pct',
                'Obi', 'Bsize', 'Asize',
                'Datetime', 'Date', 'Time', 'Timestamp', 'Level_0'}


def load_master(ticker, tail_days=None):
    cache_key = (ticker, tail_days)
    if cache_key in _master_cache:
        return _master_cache[cache_key]
    clean = ticker.replace("I:", "").replace("X:", "")
    file_ticker = TICKER_FILE_MAP.get(clean, clean)
    blob_path = f"{INFERENCE_DIR}/{file_ticker}_master_inference.parquet"
    local_path = f"/tmp/_st_{file_ticker}_master.parquet"
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
        print(f"      load_master({ticker}): FAILED — {e}")
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
# SPIKE TRANSFORMER — CONFIG
# ==============================================================================
ST_GCS_DATA_PREFIX = 'Algos/RA_MoE/real_data'
ST_GCS_MODEL_PREFIX = 'Algos/RA_MoE/real_models'
ST_FOLD_ID = 5
ST_SCORE_MINUTE = 375
ST_ENTRY_MINUTE = 385
ST_TIME_STEPS = 128
ST_D_MODEL = 64
ST_N_HEADS = 4
ST_D_FF = 128
ST_N_BLOCKS = 2
ST_DROPOUT = 0.1
ST_FILM_HIDDEN = 64
ST_N_HORIZONS = 3
ST_N_ASSET_F = 33
ST_N_REGIME_F = 16

ST_ROC_HORIZONS = [5, 30, 60]
ST_CONTEXT_ASSETS = ['SPY', 'VIX', 'TLT', 'IWM']

ST_ASSET_COLS = (['Gap_Pct', 'RSI_14', 'RVOL', 'ATR_14', 'Bar_Is_Real'] +
                 [f'ROC_{h}m' for h in ST_ROC_HORIZONS] + MICRO_COLS)

ST_CONTEXT_COLS = []
for _t in ST_CONTEXT_ASSETS:
    ST_CONTEXT_COLS.append(f'{_t}_Anchor_ROC')
    for _h in ST_ROC_HORIZONS:
        ST_CONTEXT_COLS.append(f'{_t}_ROC_{_h}m')
ST_CONTEXT_COLS.append('Time_Pos')

ST_ALL_COLS = ST_ASSET_COLS + ST_CONTEXT_COLS

ST_REGIME_COLS = [
    'regime_rvol_15', 'regime_rvol_60', 'regime_rvol_128',
    'regime_trend_60', 'regime_trend_128', 'regime_autocorr_60',
    'regime_xsect_disp', 'regime_vix_level', 'regime_vix_velocity',
    'regime_time_sin', 'regime_time_cos', 'regime_spy_ret_60',
    'regime_vix_close', 'regime_spy_tlt_corr', 'regime_breadth',
    'regime_skew_60',
]

ST_HORIZON_LABELS = {0: '15m', 1: '30m', 2: '60m'}


# ==============================================================================
# FROZEN DEPLOY CONFIGS — ST LONG fold 5
# ==============================================================================
DEPLOY_CONFIGS = {
    'ACHR': {'fold': 5, 'horizon': 0, 'prob_thresh': 0.50, 'tp_pct': 0.12, 'sl_pct': 0.06, 'mode': 'grace', 'grace_minutes': 15},
    'CLSK': {'fold': 5, 'horizon': 2, 'prob_thresh': 0.60, 'tp_pct': 0.08, 'sl_pct': 0.04, 'mode': 'static'},
    'MARA': {'fold': 5, 'horizon': 1, 'prob_thresh': 0.60, 'tp_pct': 0.12, 'sl_pct': 0.06, 'mode': 'static'},
    'ONDS': {'fold': 5, 'horizon': 0, 'prob_thresh': 0.50, 'tp_pct': 0.20, 'sl_pct': 0.10, 'mode': 'limit_entry', 'dip_pct': 0.04, 'max_wait': 90},
    'PL':   {'fold': 5, 'horizon': 1, 'prob_thresh': 0.50, 'tp_pct': 0.16, 'sl_pct': 0.08, 'mode': 'limit_entry', 'dip_pct': 0.025, 'max_wait': 60},
    'RIOT': {'fold': 5, 'horizon': 0, 'prob_thresh': 0.50, 'tp_pct': 0.06, 'sl_pct': 0.03, 'mode': 'static'},
    'SOUN': {'fold': 5, 'horizon': 0, 'prob_thresh': 0.40, 'tp_pct': 0.12, 'sl_pct': 0.06, 'mode': 'grace', 'grace_minutes': 15},
}

DEPLOY_TICKERS = sorted(DEPLOY_CONFIGS.keys())


# ==============================================================================
# SPIKE TRANSFORMER — MODEL ARCHITECTURE
# ==============================================================================
from keras import layers as _kl


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
            ST_D_MODEL, ST_N_HEADS, ST_D_FF, ST_N_BLOCKS,
            ST_TIME_STEPS, ST_DROPOUT, name='backbone')
        self.attn_pool = _AttentionPooling(ST_D_MODEL, name='attn_pool')
        self.film = _FiLMConditioner(ST_D_MODEL, ST_FILM_HIDDEN, name='film')
        self.drop = Dropout(ST_DROPOUT)
        self.head = Dense(ST_N_HORIZONS, activation='sigmoid', name='spike_head')

    def call(self, inputs, training=False):
        h = self.backbone(inputs['seq_input'], training=training)
        h = self.attn_pool(h, training=training)
        h = self.film(h, inputs['regime_input'], training=training)
        return self.head(self.drop(h, training=training))


# ==============================================================================
# MARKET CONTEXT & FEATURES
# ==============================================================================
def build_st_market_context():
    frames = []
    vix_close = None
    for ca in ST_CONTEXT_ASSETS:
        df = load_master(ca, tail_days=TAIL_DAYS)
        if df is None:
            raise RuntimeError(f"Cannot load ST context asset {ca}")
        df = df.sort_index()
        df = heal_data(df)
        if ca == 'VIX':
            vix_close = df['Close'].copy()
        cols = {f'{ca}_Anchor_ROC': df['Close'].pct_change(1)}
        for h in ST_ROC_HORIZONS:
            cols[f'{ca}_ROC_{h}m'] = df['Close'].pct_change(h)
        frames.append(pd.DataFrame(cols, index=df.index))
    result = pd.concat(frames, axis=1).ffill().fillna(0)
    if vix_close is not None:
        result['VIX_Close'] = vix_close.reindex(result.index).ffill().fillna(0)
    return result


def engineer_st_asset(ticker, st_market_df):
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
    for h in ST_ROC_HORIZONS:
        df[f'ROC_{h}m'] = df['Close'].pct_change(h)
    df = df.join(st_market_df, how='left').ffill().fillna(0)
    df_pst = df.index.tz_convert(PST)
    mins = df_pst.hour * 60 + df_pst.minute
    df['Time_Pos'] = np.clip((mins - 390) / 390, 0, 1)
    df['Bar_Is_Real'] = (df['Volume'] > 0).astype(float)
    return df.replace([np.inf, -np.inf], np.nan).fillna(0)


def compute_st_regime_features(market_df):
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
    return rf[ST_REGIME_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)


# ==============================================================================
# WEIGHT LOADING — Keras 3 .weights.h5 via h5py
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


# ==============================================================================
# MODEL LOADING
# ==============================================================================
_st_model = None
_st_scaling_stats = None
_st_regime_stats = None


def _ensure_st_file(gcs_path, local_path):
    if os.path.exists(local_path):
        return local_path
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    gcs_client.bucket(BUCKET_NAME).blob(gcs_path).download_to_filename(local_path)
    return local_path


def load_st_model_and_meta():
    global _st_model, _st_scaling_stats, _st_regime_stats

    meta_path = _ensure_st_file(
        f'{ST_GCS_DATA_PREFIX}/meta.npz', '/tmp/ra_moe_meta.npz')
    meta = np.load(meta_path, allow_pickle=True)
    _st_scaling_stats = meta['scaling_stats'].item()
    _st_regime_stats = meta['regime_stats'].item()

    keras.backend.clear_session()
    gc.collect()
    _st_model = SpikeTransformer(name='spike_transformer')
    dummy_seq = tf.zeros((1, ST_TIME_STEPS, ST_N_ASSET_F))
    dummy_reg = tf.zeros((1, ST_N_REGIME_F))
    _st_model({'seq_input': dummy_seq, 'regime_input': dummy_reg})

    local_dir = f'/tmp/ra_moe_models/fold_{ST_FOLD_ID}'
    weights_path = os.path.join(local_dir, 'model.weights.h5')
    os.makedirs(local_dir, exist_ok=True)
    prefix = f'{ST_GCS_MODEL_PREFIX}/fold_{ST_FOLD_ID}'
    bucket_obj = gcs_client.bucket(BUCKET_NAME)
    for blob in bucket_obj.list_blobs(prefix=prefix):
        fname = blob.name.split('/')[-1]
        blob.download_to_filename(os.path.join(local_dir, fname))
    _load_keras3_weights(_st_model, weights_path)
    print(f"   ST model loaded: {_st_model.count_params():,} params")
    return _st_model


def apply_st_scaling(df, stats, cols):
    df_s = df.copy()
    for col in cols:
        if col in ('Time_Pos', 'Bar_Is_Real'):
            continue
        if col in stats:
            df_s[col] = ((df_s[col] - stats[col]['med'])
                         / stats[col]['iqr']).clip(-10, 10).astype('float32')
    return df_s


# ==============================================================================
# SCORING ENGINE
# ==============================================================================
def score_st_ticker(ticker, cfg, df_feat, regime_df, run_ts):
    today = run_ts.astimezone(PST).date()
    score_min = ST_SCORE_MINUTE
    entry_min = ST_ENTRY_MINUTE
    target_pst = PST.localize(datetime(today.year, today.month, today.day,
                                       score_min // 60, score_min % 60))
    target_utc = target_pst.astimezone(pytz.utc)

    candidates = df_feat.index[df_feat.index <= target_utc]
    if len(candidates) == 0:
        print(f"      {ticker}: SKIP — no data at or before {target_pst.strftime('%H:%M')} PST")
        return None
    row_idx = df_feat.index.get_loc(candidates[-1])
    bar_ts_pst = df_feat.index[row_idx].tz_convert(PST)
    score_date = bar_ts_pst.date()
    if score_date != today:
        print(f"      {ticker}: SKIP — nearest bar is {score_date} (stale master data)")
        return None
    bar_minute = bar_ts_pst.hour * 60 + bar_ts_pst.minute
    if bar_minute != score_min:
        print(f"      {ticker}: SKIP — nearest bar at {bar_ts_pst.strftime('%H:%M')} PST, need {score_min//60}:{score_min%60:02d}")
        return None
    if row_idx < ST_TIME_STEPS:
        print(f"      {ticker}: SKIP — not enough ST history: {row_idx} < {ST_TIME_STEPS}")
        return None
    if 'Volume' in df_feat.columns and float(df_feat['Volume'].iloc[row_idx]) == 0:
        print(f"      {ticker}: SKIP — no liquidity at {bar_ts_pst.strftime('%H:%M')} PST (Volume=0, synthetic bar)")
        return None

    df_scaled = apply_st_scaling(df_feat, _st_scaling_stats, ST_ALL_COLS)

    XA = df_scaled[ST_ALL_COLS].values.astype(np.float32)
    seq_batch = XA[row_idx - ST_TIME_STEPS:row_idx][np.newaxis]

    regime_at_entry = regime_df.iloc[regime_df.index.get_indexer(
        [df_feat.index[row_idx]], method='pad')[0]]
    regime_scaled = np.zeros(ST_N_REGIME_F, dtype=np.float32)
    for i, col in enumerate(ST_REGIME_COLS):
        val = float(regime_at_entry[col]) if col in regime_at_entry.index else 0.0
        if col in _st_regime_stats:
            s = _st_regime_stats[col]
            val = np.clip((val - s['med']) / s['iqr'], -10, 10)
        regime_scaled[i] = val
    regime_batch = regime_scaled[np.newaxis]

    preds = _st_model(
        {'seq_input': seq_batch, 'regime_input': regime_batch},
        training=False
    ).numpy().flatten()

    h_idx = cfg['horizon']
    prob = float(preds[h_idx])
    signal = prob >= cfg['prob_thresh']

    entry_bar_idx = row_idx + (entry_min - score_min)
    if entry_bar_idx < len(df_feat) and 'Close' in df_feat.columns:
        close = float(df_feat['Close'].iloc[entry_bar_idx])
    elif 'Close' in df_feat.columns:
        close = float(df_feat['Close'].iloc[row_idx])
    else:
        close = float('nan')

    entry_price = close
    if cfg['mode'] == 'limit_entry':
        entry_price = close * (1 - cfg.get('dip_pct', 0))

    return {
        'ticker': ticker,
        'horizon': ST_HORIZON_LABELS[cfg['horizon']],
        'prob': round(prob, 4),
        'threshold': cfg['prob_thresh'],
        'signal': signal,
        'tp_pct': cfg['tp_pct'],
        'sl_pct': cfg['sl_pct'],
        'mode': cfg['mode'],
        'dip_pct': cfg.get('dip_pct', 0),
        'max_wait': cfg.get('max_wait', 0),
        'grace_minutes': cfg.get('grace_minutes', 0),
        'entry_minute': entry_min,
        'score_date': score_date,
        'bar_ts': df_feat.index[row_idx],
        'close': round(close, 4),
        'entry_price': round(entry_price, 4),
        'all_horizon_probs': {ST_HORIZON_LABELS[i]: round(float(preds[i]), 4)
                              for i in range(ST_N_HORIZONS)},
    }


def _run_spike_transformer(run_ts):
    now_pst = run_ts.astimezone(PST)
    now_minutes = now_pst.hour * 60 + now_pst.minute
    if now_minutes < ST_SCORE_MINUTE:
        print(f"\n  ST: SKIP — too early ({now_minutes} < {ST_SCORE_MINUTE})")
        return []

    print(f"\n  ST: {len(DEPLOY_CONFIGS)} tickers — {', '.join(DEPLOY_TICKERS)}")

    print(f"  ST: Loading model and scaling stats...")
    load_st_model_and_meta()

    print(f"  ST: Building market context...")
    st_market_df = build_st_market_context()
    _master_cache.clear()

    print(f"  ST: Computing regime features...")
    regime_df = compute_st_regime_features(st_market_df)

    results = []
    for ticker in DEPLOY_TICKERS:
        cfg = DEPLOY_CONFIGS[ticker]
        df_feat = engineer_st_asset(ticker, st_market_df)
        if df_feat is None:
            print(f"      {ticker}: SKIP — load_master failed")
            continue
        result = score_st_ticker(ticker, cfg, df_feat, regime_df, run_ts)
        if result is not None:
            results.append(result)
        del df_feat
        _master_cache.pop((ticker, TAIL_DAYS), None)
        gc.collect()

    scored = [r['ticker'] for r in results]
    print(f"  ST: scored {len(scored)}/{len(DEPLOY_CONFIGS)}: {', '.join(scored)}")

    keras.backend.clear_session()
    gc.collect()
    return results


# ==============================================================================
# DASHBOARD
# ==============================================================================
def print_st_dashboard(st_results, run_ts):
    W = 120
    pst_str = run_ts.astimezone(PST).strftime('%Y-%m-%d %H:%M PST')

    print()
    print(f"  {'='*W}")
    print(f"  SPIKE TRANSFORMER LONG SIGNALS   {pst_str}")
    print(f"  {'='*W}")
    print(f"  Model: fold {ST_FOLD_ID}  |  Score: {ST_SCORE_MINUTE // 60}:"
          f"{ST_SCORE_MINUTE % 60:02d} PST  →  Entry: {ST_ENTRY_MINUTE // 60}:"
          f"{ST_ENTRY_MINUTE % 60:02d} PST  |  {len(DEPLOY_CONFIGS)} tickers (frozen configs)")

    if not st_results:
        print(f"\n  No tickers scored.")
        print(f"  {'='*W}")
        return

    print()
    print(f"  {'Ticker':<6}  {'H':>3}  {'SIG':<7}  "
          f"{'Buy@':>10}  {'SL@':>10}  {'TP@':>10}  "
          f"{'Prob':>5} / {'Thr':>5}  {'Mode':<8}  "
          f"{'Close':>8}  {'P(15m)':>6} {'P(30m)':>6} {'P(60m)':>6}")
    print(f"  {'─'*W}")

    for r in st_results:
        if r['signal']:
            sig_str = "\033[92m GO  \033[0m"
        else:
            sig_str = "  --  "

        entry_price = r['entry_price']
        buy_str = f"${entry_price:.2f}"
        sl_str = f"${entry_price * (1 - r['sl_pct']):.2f}"
        tp_str = f"${entry_price * (1 + r['tp_pct']):.2f}"

        mode_label = r['mode'][:8].upper()
        hp = r['all_horizon_probs']

        print(f"  {r['ticker']:<6}  {r['horizon']:>3}  {sig_str}  "
              f"{buy_str:>10}  {sl_str:>10}  {tp_str:>10}  "
              f"{r['prob']:>5.3f} / {r['threshold']:>5.3f}  {mode_label:<8}  "
              f"{r['close']:>8.2f}  "
              f"{hp.get('15m', 0):>5.3f}  {hp.get('30m', 0):>5.3f}  {hp.get('60m', 0):>5.3f}")

    signals = [r for r in st_results if r['signal']]
    go_trades = signals

    print()
    print(f"  {'='*W}")
    print(f"  ST LONG SUMMARY")
    print(f"  {'─'*W}")
    print(f"  Scored:    {len(st_results)} tickers")
    print(f"  Signals:   {len(signals)}/{len(st_results)}" +
          (f"  ({', '.join(r['ticker'] for r in signals)})" if signals else ""))

    if go_trades:
        print()
        print(f"  \033[92mTRADE: {', '.join(r['ticker'] for r in go_trades)}"
              f"  ({len(go_trades)} position{'s' if len(go_trades) != 1 else ''})\033[0m")

        limit_trades = [r for r in go_trades if r['mode'] == 'limit_entry']
        if limit_trades:
            print()
            print(f"  {'─'*W}")
            print(f"  LIMIT ORDERS TO PLACE")
            print(f"  {'─'*W}")
            print(f"  {'Ticker':<6}  {'BUY LIMIT':>12}  {'Dip':>8}  "
                  f"{'TP target':>12}  {'SL stop':>12}  {'Cancel by':>10}  {'Exit by':>10}  {'Signal $':>10}")
            print(f"  {'─'*W}")
            for r in limit_trades:
                c = r['close']
                dip = r.get('dip_pct', 0) or 0
                lp = r['entry_price']
                tp_price = lp * (1 + r['tp_pct'])
                sl_price = lp * (1 - r['sl_pct'])
                wait = r.get('max_wait', 30) or 30
                em = r['entry_minute']
                expire_min = em + wait
                expire_h = expire_min // 60
                expire_ampm = "AM" if expire_h < 12 else "PM"
                expire_h12 = expire_h if expire_h <= 12 else expire_h - 12
                expire_str = f"{expire_h12}:{expire_min % 60:02d}{expire_ampm} PT"
                print(f"  {r['ticker']:<6}  "
                      f"\033[92m${lp:>9.2f}\033[0m  "
                      f"-{dip:.1%}  "
                      f"${tp_price:>9.2f}  ${sl_price:>9.2f}  "
                      f"{expire_str:>10}  {'10:00AM PT':>10}  ${c:>8.2f}")

        market_trades = [r for r in go_trades if r['mode'] in ('static', 'grace')]
        if market_trades:
            print()
            print(f"  {'─'*W}")
            print(f"  MARKET ORDERS")
            print(f"  {'─'*W}")
            for r in market_trades:
                tp_price = r['entry_price'] * (1 + r['tp_pct'])
                sl_price = r['entry_price'] * (1 - r['sl_pct'])
                grace_str = f"  Grace={r['grace_minutes']}m" if r['mode'] == 'grace' else ""
                print(f"  {r['ticker']:<6}  "
                      f"\033[92mBUY @ ${r['close']:>9.2f}\033[0m  "
                      f"TP ${tp_price:>9.2f}  SL ${sl_price:>9.2f}{grace_str}  "
                      f"[{r['horizon']} spike prob={r['prob']:.3f}]")

    print()
    print(f"  {'─'*W}")
    print(f"  DEPLOY CONFIGS")
    print(f"  {'─'*W}")
    print(f"  {'Ticker':<6}  {'H':>3}  {'Mode':<12}  {'Thr':>5}  "
          f"{'TP':>5}  {'SL':>5}  {'Dip':>5}  {'Wait':>5}  "
          f"{'P(15m)':>6} {'P(30m)':>6} {'P(60m)':>6}")
    print(f"  {'─'*W}")

    for r in st_results:
        mode_label = r['mode'].upper()
        if r['mode'] == 'limit_entry':
            mode_label = f"LIMIT d={r.get('dip_pct', 0):.1%}"
        elif r['mode'] == 'grace':
            mode_label = f"GRACE {r.get('grace_minutes', 0)}m"

        hp = r['all_horizon_probs']
        dip_str = f"{r.get('dip_pct', 0):.1%}" if r['mode'] == 'limit_entry' else "  --"
        wait_str = f"{r.get('max_wait', 0):>3}m" if r['mode'] == 'limit_entry' else "  --"

        print(f"  {r['ticker']:<6}  {r['horizon']:>3}  {mode_label:<12}  {r['threshold']:>5.2f}  "
              f"{r['tp_pct']:>4.0%}  {r['sl_pct']:>4.0%}  {dip_str:>5}  {wait_str:>5}  "
              f"{hp.get('15m', 0):>5.3f}  {hp.get('30m', 0):>5.3f}  {hp.get('60m', 0):>5.3f}")

    print()
    print(f"  Model: spike_transformer fold {ST_FOLD_ID}  |  "
          f"Score {ST_SCORE_MINUTE//60}:{ST_SCORE_MINUTE%60:02d} → Entry {ST_ENTRY_MINUTE//60}:{ST_ENTRY_MINUTE%60:02d} PST  |  "
          f"Scaling: global median/IQR from meta.npz  |  FiLM conditioning")
    print(f"  {'='*W}")


# ==============================================================================
# BQ LOGGING
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
    ("tpsl_mode",      "STRING"),
    ("strategy",       "STRING"),
]


def _ensure_bq_schema(bq_client):
    for col_name, col_type in _BQ_SCHEMA_DEFS:
        sql = (f"ALTER TABLE `{BQ_TABLE}` "
               f"ADD COLUMN IF NOT EXISTS `{col_name}` {col_type}")
        bq_client.query(sql).result()


def log_st_to_bigquery(st_results, run_ts):
    if bigquery is None or not st_results:
        return
    rows = []
    for r in st_results:
        rows.append({
            'run_timestamp': run_ts.isoformat(),
            'bar_timestamp': r['bar_ts'].isoformat(),
            'score_date': str(r['score_date']),
            'experiment': f'spike_transformer_fold5_{r["horizon"]}',
            'ticker': r['ticker'],
            'seed': 0,
            'prob': r['prob'],
            'threshold': r['threshold'],
            'signal': 'GO' if r['signal'] else 'NO',
            'tp_pct': r['tp_pct'],
            'sl_pct': r['sl_pct'],
            'entry_minute': r['entry_minute'],
            'close': r['close'],
            'filter_status': 'OK',
            'tpsl_mode': r['mode'],
            'strategy': 'ST_LONG',
        })
    bq_client = bigquery.Client()
    _ensure_bq_schema(bq_client)
    try:
        errors = bq_client.insert_rows_json(BQ_TABLE, rows)
        if errors:
            print(f"   ST BigQuery errors: {errors}")
        else:
            print(f"   ST logged {len(rows)} rows to {BQ_TABLE}")
    except Exception as e:
        print(f"   ST BigQuery failed: {e}")


# ==============================================================================
# MAIN
# ==============================================================================
def _check_stitch_freshness():
    from google.cloud import storage as _gcs
    blob = _gcs.Client().bucket(BUCKET_NAME).blob("Data/Inference/pipeline_status.json")
    try:
        status = json.loads(blob.download_as_text())
    except Exception:
        print("   ⚠️  pipeline_status.json not found — stitch freshness unknown")
        return
    stitch = status.get('stitch', {})
    if stitch.get('status') != 'complete':
        print("   ⚠️  daily_stitch_master has not run — master_inference may be stale")
        return
    ts = pd.Timestamp(stitch.get('timestamp', '2000-01-01'))
    if ts.tz is None:
        ts = ts.tz_localize('UTC')
    today = pd.Timestamp.now(tz='UTC').normalize()
    if ts.normalize() < today:
        print(f"   ⚠️  Stitch last ran {ts.strftime('%Y-%m-%d %H:%M')} — not today")


run_ts = datetime.now(pytz.utc)
pst_str = run_ts.astimezone(PST).strftime('%Y-%m-%d %I:%M %p PST')
print(f"\n{'='*80}")
print(f"  DAILY SCORING — ST LONG F5 (frozen) — {pst_str}")
print(f"  Tickers: {', '.join(DEPLOY_TICKERS)}")
print(f"{'='*80}")
_check_stitch_freshness()

st_results = _run_spike_transformer(run_ts)
if st_results:
    print_st_dashboard(st_results, run_ts)
    log_st_to_bigquery(st_results, run_ts)
