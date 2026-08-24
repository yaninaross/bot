# %% ── ST LONG + SHORT — Last Week Live-Scoring Backtest ──────────────────────
#
# Backtests BOTH Spike Transformer strategies over the last week using live
# model inference from master_inference parquet files on GCS.
#
# ST LONG:  fold 4, configs from BQ (model_version = spike_transformer_fold4)
# ST SHORT: fold 5, 7 tickers with frozen configs (ALAB, BBAI, BE, CIFR, HUT, MARA, PL)
#
# For each trading day in the window:
#   1. Load master_inference data for all tickers
#   2. Engineer features (same pipeline as daily scoring scripts)
#   3. Score with the deployed model at entry minute
#   4. Simulate trades using 1-min bars (static / grace / limit_entry modes)
#   5. Print per-day, per-ticker, and portfolio results
#
# GPU recommended. Colab-ready.

import os, sys, gc, json, subprocess, warnings
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
BUCKET_NAME    = "time-series-bucket-ys"
INFERENCE_DIR  = "Data/Inference_v2"
WF_BQ_TABLE    = "ys-time-series.scoring_logs.wf_optimization_results"
PST            = pytz.timezone('America/Los_Angeles')
TAIL_DAYS      = 30

BACKTEST_START = '2026-08-04'
BACKTEST_END   = '2026-08-11'

SLIPPAGE_PCT   = 0.0015
EXIT_TIME_MIN  = 780

TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}
MICRO_COLS = ['Spread', 'OBI', 'Volume_Delta', 'Trade_Count', 'Delta_Pct',
              'bsize', 'asize', 'Total_Vol']
ROC_HORIZONS = [5, 30, 60]
CONTEXT_ASSETS = ['SPY', 'VIX', 'TLT', 'IWM']
EXCLUDE_TICKERS = {'SPY', 'VIX', 'TLT', 'IWM', 'XLE', 'ITA', 'BITF', 'MIGI', 'SMLR'}

W = 110

gcs_client = gcs_storage.Client()


# ── ST LONG config ──
LONG_GCS_DATA   = 'Algos/RA_MoE/real_data'
LONG_GCS_MODEL  = 'Algos/RA_MoE/real_models'
LONG_FOLD       = 4
LONG_ENTRY_MIN  = 375
LONG_BQ_MODEL_VERSION = 'spike_transformer_fold4'
LONG_DEPLOY_THRESHOLD = 0.018

# ── ST SHORT config ──
SHORT_GCS_DATA  = 'Algos/RA_MoE/short_data'
SHORT_GCS_MODEL = 'Algos/RA_MoE/short_models'
SHORT_FOLD      = 5
SHORT_SCORE_MIN = 375
SHORT_ENTRY_MIN = 385
SHORT_MIN_PRICE = 2.00

SHORT_DEPLOY_CONFIGS = {
    'ALAB': {'horizon': 0, 'prob_thresh': 0.20, 'tp_pct': 0.12, 'sl_pct': 0.06, 'mode': 'static'},
    'BE':   {'horizon': 1, 'prob_thresh': 0.40, 'tp_pct': 0.12, 'sl_pct': 0.06, 'mode': 'static'},
    'PL':   {'horizon': 0, 'prob_thresh': 0.20, 'tp_pct': 0.08, 'sl_pct': 0.04, 'mode': 'limit_entry', 'dip_pct': 0.012, 'max_wait': 30},
    'HUT':  {'horizon': 2, 'prob_thresh': 0.40, 'tp_pct': 0.08, 'sl_pct': 0.04, 'mode': 'static'},
    'CIFR': {'horizon': 2, 'prob_thresh': 0.40, 'tp_pct': 0.16, 'sl_pct': 0.08, 'mode': 'grace', 'grace_minutes': 15},
    'BBAI': {'horizon': 0, 'prob_thresh': 0.30, 'tp_pct': 0.16, 'sl_pct': 0.08, 'mode': 'static'},
    'MARA': {'horizon': 0, 'prob_thresh': 0.40, 'tp_pct': 0.08, 'sl_pct': 0.04, 'mode': 'static'},
}

# ── Shared model params ──
TIME_STEPS = 128
D_MODEL = 64; N_HEADS = 4; D_FF = 128; N_BLOCKS = 2; DROPOUT = 0.1
FILM_HIDDEN = 64; N_HORIZONS = 3; N_ASSET_F = 33; N_REGIME_F = 16

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
]

HORIZON_LABELS = {0: '15m', 1: '30m', 2: '60m'}
HORIZON_FROM_LABEL = {'15m': 0, '30m': 1, '60m': 2}


# ==============================================================================
# MODEL ARCHITECTURE (shared between LONG and SHORT)
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
        scores = tf.matmul(Q, K, transpose_b=True) / self.scale
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
        scores = tf.matmul(self.query, x, transpose_b=True) / self.scale
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


def load_model(gcs_data_prefix, gcs_model_prefix, fold_id, label):
    meta_path = _ensure_file(
        f'{gcs_data_prefix}/meta.npz', f'/tmp/bt_{label}_meta.npz')
    meta = np.load(meta_path, allow_pickle=True)
    scaling_stats = meta['scaling_stats'].item()
    regime_stats = meta['regime_stats'].item()

    keras.backend.clear_session()
    gc.collect()
    model = SpikeTransformer(name=f'spike_transformer_{label}')
    dummy_seq = tf.zeros((1, TIME_STEPS, N_ASSET_F))
    dummy_reg = tf.zeros((1, N_REGIME_F))
    model({'seq_input': dummy_seq, 'regime_input': dummy_reg})

    local_dir = f'/tmp/bt_{label}_models/fold_{fold_id}'
    weights_path = os.path.join(local_dir, 'model.weights.h5')
    if not os.path.exists(weights_path):
        os.makedirs(local_dir, exist_ok=True)
        prefix = f'{gcs_model_prefix}/fold_{fold_id}'
        bucket = gcs_client.bucket(BUCKET_NAME)
        for blob in bucket.list_blobs(prefix=prefix):
            fname = blob.name.split('/')[-1]
            blob.download_to_filename(os.path.join(local_dir, fname))
    model.load_weights(weights_path)
    print(f"  {label} model loaded: {model.count_params():,} params (fold {fold_id})")
    return model, scaling_stats, regime_stats


# ==============================================================================
# BQ CONFIG LOADING (LONG)
# ==============================================================================
def load_long_deploy_configs():
    if bigquery is None:
        print("  BigQuery not available - no LONG configs")
        return {}
    bq_client = bigquery.Client()
    q = f"""
    SELECT ticker, experiment, tpsl_mode, prob_thresh, tp_pct, sl_pct,
           grace_minutes, dip_pct, max_wait_minutes,
           val_b_avg_pnl, val_b_win_rate, val_b_count
    FROM `{WF_BQ_TABLE}`
    WHERE model_version = '{LONG_BQ_MODEL_VERSION}'
      AND survived_a = TRUE
      AND val_b_avg_pnl >= {LONG_DEPLOY_THRESHOLD}
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
        h_part = exp.replace(f'{LONG_BQ_MODEL_VERSION}_', '').split('_')[0]
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
# SCORING — score one ticker for one day
# ==============================================================================
def score_ticker_day(model, scaling_stats, regime_stats,
                     ticker, cfg, df_feat, regime_df,
                     target_date, score_minute):
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
# TRADE SIMULATION — using 1-min bars from df_feat
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
                pnl = cfg['tp_pct']
                return {'pnl': pnl, 'entry': entry_price, 'exit': tp_price, 'outcome': 'TP'}
            else:
                pnl = -cfg['sl_pct']
                return {'pnl': pnl, 'entry': entry_price, 'exit': sl_price, 'outcome': 'SL'}
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
            elapsed = (bar_pst.hour * 60 + bar_pst.minute) - (df_feat.index[entry_idx].tz_convert(PST).hour * 60 + df_feat.index[entry_idx].tz_convert(PST).minute)
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
    entry_time_min = df_feat.index[start_idx].tz_convert(PST).hour * 60 + df_feat.index[start_idx].tz_convert(PST).minute

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
# GET TRADING DAYS IN BACKTEST WINDOW
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
# MAIN BACKTEST
# ==============================================================================
print(f"\n{'='*W}")
print(f"  ST LONG + SHORT — LAST WEEK LIVE-SCORING BACKTEST")
print(f"  Period: {BACKTEST_START} -> {BACKTEST_END}")
print(f"  Slippage: {SLIPPAGE_PCT:.2%} each way ({2*SLIPPAGE_PCT:.2%} RT)")
print(f"{'='*W}")

# 1. Load LONG configs from BQ
print(f"\n  Loading LONG deploy configs from BQ...")
long_configs = load_long_deploy_configs()
long_tickers = sorted(long_configs.keys())
print(f"  LONG: {len(long_tickers)} tickers - {', '.join(long_tickers)}")

short_tickers = sorted(SHORT_DEPLOY_CONFIGS.keys())
print(f"  SHORT: {len(short_tickers)} tickers - {', '.join(short_tickers)}")

# 2. Build market context and regime features
print(f"\n  Building market context...")
market_df = build_market_context()
_master_cache.clear()
print(f"  Computing regime features...")
regime_df = compute_regime_features(market_df)

# 3. Get trading days
trading_days = get_trading_days(BACKTEST_START, BACKTEST_END)
print(f"  Trading days: {len(trading_days)} ({trading_days[0]} -> {trading_days[-1]})")

# 4. Load both models
print(f"\n  Loading models...")
long_model, long_scaling, long_regime_stats = load_model(
    LONG_GCS_DATA, LONG_GCS_MODEL, LONG_FOLD, 'long')
short_model, short_scaling, short_regime_stats = load_model(
    SHORT_GCS_DATA, SHORT_GCS_MODEL, SHORT_FOLD, 'short')

# 5. Pre-engineer features for all tickers
all_tickers = sorted(set(long_tickers) | set(short_tickers))
print(f"\n  Engineering features for {len(all_tickers)} tickers...")
ticker_features = {}
for ticker in all_tickers:
    df_feat = engineer_asset(ticker, market_df)
    if df_feat is not None:
        ticker_features[ticker] = df_feat
        print(f"    {ticker}: {len(df_feat):,} bars")
    else:
        print(f"    {ticker}: FAILED")
    _master_cache.pop((ticker, TAIL_DAYS), None)

# 6. Run backtest day by day
all_trades = []
daily_signals = defaultdict(list)

print(f"\n{'='*W}")
print(f"  SCORING + SIMULATION")
print(f"{'='*W}")

for day in trading_days:
    day_str = day.strftime('%Y-%m-%d')
    print(f"\n  --- {day_str} ---")

    # LONG scoring
    long_signals = []
    for ticker in long_tickers:
        if ticker not in ticker_features:
            continue
        cfg = long_configs[ticker]
        result = score_ticker_day(
            long_model, long_scaling, long_regime_stats,
            ticker, cfg, ticker_features[ticker], regime_df,
            day, LONG_ENTRY_MIN)
        if result is None:
            continue
        if result['signal']:
            trade = simulate_long_trade(ticker_features[ticker], result['bar_idx'], cfg)
            if trade is not None:
                trade['pnl'] -= 2 * SLIPPAGE_PCT
                all_trades.append({
                    'Date': day_str, 'Ticker': ticker, 'Strategy': 'ST_LONG',
                    'Prob': result['prob'], 'Thresh': cfg['prob_thresh'],
                    'Entry': trade['entry'], 'Exit': trade['exit'],
                    'PnL': trade['pnl'], 'Outcome': trade['outcome'],
                    'Mode': cfg['mode'], 'Horizon': HORIZON_LABELS[cfg['horizon']],
                    'TP_pct': cfg['tp_pct'], 'SL_pct': cfg['sl_pct'],
                })
                long_signals.append(f"{ticker}({result['prob']:.2f})")
            else:
                long_signals.append(f"{ticker}(no fill)")
        daily_signals[day_str].append({
            'ticker': ticker, 'strategy': 'LONG',
            'prob': result['prob'], 'signal': result['signal'],
            'close': result['close'], 'probs': result['all_probs']})

    # SHORT scoring
    short_signals = []
    for ticker in short_tickers:
        if ticker not in ticker_features:
            continue
        cfg = SHORT_DEPLOY_CONFIGS[ticker]
        if 'Close' in ticker_features[ticker].columns:
            target_pst = PST.localize(datetime(day.year, day.month, day.day,
                                               SHORT_SCORE_MIN // 60, SHORT_SCORE_MIN % 60))
            target_utc = target_pst.astimezone(pytz.utc)
            cands = ticker_features[ticker].index[ticker_features[ticker].index <= target_utc]
            if len(cands) > 0:
                check_idx = ticker_features[ticker].index.get_loc(cands[-1])
                raw_close = float(ticker_features[ticker]['Close'].iloc[check_idx])
                if raw_close < SHORT_MIN_PRICE:
                    continue

        result = score_ticker_day(
            short_model, short_scaling, short_regime_stats,
            ticker, cfg, ticker_features[ticker], regime_df,
            day, SHORT_SCORE_MIN)
        if result is None:
            continue

        entry_bar_idx = result['bar_idx']
        if cfg['mode'] != 'limit_entry':
            offset = SHORT_ENTRY_MIN - SHORT_SCORE_MIN
            entry_bar_idx = result['bar_idx'] + offset

        if result['signal']:
            trade = simulate_short_trade(ticker_features[ticker], entry_bar_idx, cfg)
            if trade is not None:
                trade['pnl'] -= 2 * SLIPPAGE_PCT
                all_trades.append({
                    'Date': day_str, 'Ticker': ticker, 'Strategy': 'ST_SHORT',
                    'Prob': result['prob'], 'Thresh': cfg['prob_thresh'],
                    'Entry': trade['entry'], 'Exit': trade['exit'],
                    'PnL': trade['pnl'], 'Outcome': trade['outcome'],
                    'Mode': cfg['mode'], 'Horizon': HORIZON_LABELS[cfg['horizon']],
                    'TP_pct': cfg['tp_pct'], 'SL_pct': cfg['sl_pct'],
                })
                short_signals.append(f"{ticker}({result['prob']:.2f})")
            else:
                short_signals.append(f"{ticker}(no fill)")
        daily_signals[day_str].append({
            'ticker': ticker, 'strategy': 'SHORT',
            'prob': result['prob'], 'signal': result['signal'],
            'close': result['close'], 'probs': result['all_probs']})

    long_str = ', '.join(long_signals) if long_signals else 'none'
    short_str = ', '.join(short_signals) if short_signals else 'none'
    print(f"    LONG trades:  {long_str}")
    print(f"    SHORT trades: {short_str}")

# Free models
keras.backend.clear_session()
gc.collect()


# ==============================================================================
# RESULTS
# ==============================================================================
if not all_trades:
    print(f"\n  NO TRADES in backtest window - nothing to analyze")
else:
    trades_df = pd.DataFrame(all_trades)
    trades_df['Date_dt'] = pd.to_datetime(trades_df['Date'])
    trades_df = trades_df.sort_values(['Date_dt', 'Strategy', 'Ticker'])

    for strat in ['ST_LONG', 'ST_SHORT']:
        sdf = trades_df[trades_df['Strategy'] == strat]
        if len(sdf) == 0:
            print(f"\n  {strat}: No trades")
            continue

        n = len(sdf)
        n_win = (sdf['PnL'] > 0).sum()
        n_loss = (sdf['PnL'] <= 0).sum()
        avg_pnl = sdf['PnL'].mean()
        total_pnl = sdf['PnL'].sum()
        wr = n_win / n if n > 0 else 0
        avg_win = sdf.loc[sdf['PnL'] > 0, 'PnL'].mean() if n_win > 0 else 0
        avg_loss = sdf.loc[sdf['PnL'] <= 0, 'PnL'].mean() if n_loss > 0 else 0

        print(f"\n{'='*W}")
        print(f"  {strat} RESULTS — {BACKTEST_START} to {BACKTEST_END}")
        print(f"{'='*W}")
        print(f"  Trades:     {n}")
        print(f"  Avg PnL:    {avg_pnl:+.2%}")
        print(f"  Total PnL:  {total_pnl:+.2%}")
        print(f"  Win rate:   {wr:.0%} ({n_win}W / {n_loss}L)")
        print(f"  Avg win:    {avg_win:+.2%}")
        print(f"  Avg loss:   {avg_loss:+.2%}")
        if n_loss > 0 and avg_loss != 0:
            print(f"  PF:         {abs(avg_win * n_win / (avg_loss * n_loss)):.2f}")

        print(f"\n  TRADE LOG:")
        print(f"  {'Date':<12} {'Ticker':<7} {'H':>3} {'Mode':<8} "
              f"{'Prob':>5}/{' Thr':>5}  "
              f"{'Entry':>9} {'Exit':>9} {'PnL':>8} {'Type':>4}")
        print(f"  {'-'*90}")
        for _, t in sdf.iterrows():
            sig = '*' if t['Prob'] >= t['Thresh'] else ' '
            print(f"  {t['Date']:<12} {t['Ticker']:<7} {t['Horizon']:>3} {t['Mode']:<8} "
                  f"{t['Prob']:>5.2f}/{t['Thresh']:>5.2f}  "
                  f"${t['Entry']:>8.2f} ${t['Exit']:>8.2f} {t['PnL']:>+7.2%} {t['Outcome']:>4}")

        # Per-ticker breakdown
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

    # ── COMBINED portfolio (equal-weight: split capital equally across day's trades) ──
    print(f"\n{'='*W}")
    print(f"  COMBINED PORTFOLIO — EQUAL WEIGHT (capital split equally per day)")
    print(f"{'='*W}")
    n_all = len(trades_df)
    if n_all > 0:
        n_win_all = (trades_df['PnL'] > 0).sum()
        avg_all = trades_df['PnL'].mean()
        wr_all = n_win_all / n_all

        daily = trades_df.groupby('Date').agg(
            N=('PnL', 'count'),
            EqW_PnL=('PnL', 'mean'),
            Additive_PnL=('PnL', 'sum'),
            Long_N=('Strategy', lambda x: (x == 'ST_LONG').sum()),
            Short_N=('Strategy', lambda x: (x == 'ST_SHORT').sum()),
            Tickers=('Ticker', lambda x: ','.join(sorted(x.unique()))),
        ).sort_index()

        total_eqw = daily['EqW_PnL'].sum()
        total_add = daily['Additive_PnL'].sum()

        print(f"  Total trades:     {n_all}")
        print(f"  Avg PnL/trade:    {avg_all:+.2%}")
        print(f"  Portfolio PnL:    {total_eqw:+.2%}  (equal-weight)")
        print(f"  Additive PnL:     {total_add:+.2%}  (100% per trade, for reference)")
        print(f"  Win rate:         {wr_all:.0%}")

        print(f"\n  DAILY P&L:")
        print(f"  {'Date':<12} {'N':>3} {'L':>2} {'S':>2} {'EqW Day':>8} {'Add Day':>8} {'EqW Cum':>8} {'Add Cum':>8}  Tickers")
        print(f"  {'-'*100}")
        cum_eqw = 0.0
        cum_add = 0.0
        for date, row in daily.iterrows():
            cum_eqw += row['EqW_PnL']
            cum_add += row['Additive_PnL']
            print(f"  {date:<12} {int(row['N']):>3} {int(row['Long_N']):>2} {int(row['Short_N']):>2} "
                  f"{row['EqW_PnL']:>+7.2%} {row['Additive_PnL']:>+7.2%} {cum_eqw:>+7.2%} {cum_add:>+7.2%}  {row['Tickers']}")

    # ── Signal log (all scored tickers, not just trades) ──
    print(f"\n{'='*W}")
    print(f"  FULL SIGNAL LOG (all scored tickers per day)")
    print(f"{'='*W}")
    for day_str in sorted(daily_signals.keys()):
        sigs = daily_signals[day_str]
        print(f"\n  {day_str}:")
        for s in sorted(sigs, key=lambda x: (x['strategy'], x['ticker'])):
            sig_mark = 'GO' if s['signal'] else '--'
            probs_str = ' '.join(f"{k}={v:.3f}" for k, v in s['probs'].items())
            print(f"    {s['strategy']:<5} {s['ticker']:<7} [{sig_mark}] "
                  f"close=${s['close']:>8.2f}  {probs_str}")

    # ── Charts ──
    if HAS_MPL and n_all > 0:
        fig, axes = plt.subplots(3, 2, figsize=(18, 18))

        # 1. Individual trades by date (scatter with jitter so same-day trades don't overlap)
        ax = axes[0, 0]
        jitter_map = {}
        for _, row in trades_df.iterrows():
            d = row['Date']
            jitter_map.setdefault(d, []).append(row.name)
        jitter_offsets = {}
        for d, idxs in jitter_map.items():
            n = len(idxs)
            offsets = np.linspace(-0.25, 0.25, n) if n > 1 else [0.0]
            for idx, off in zip(idxs, offsets):
                jitter_offsets[idx] = off

        for strat, color, marker in [('ST_LONG', '#1565C0', 'o'), ('ST_SHORT', '#D32F2F', 's')]:
            sdf = trades_df[trades_df['Strategy'] == strat]
            if len(sdf) == 0:
                continue
            dates_num = np.array([mdates.date2num(r['Date_dt']) + jitter_offsets.get(i, 0)
                                  for i, r in sdf.iterrows()])
            pnl_pct = sdf['PnL'].values * 100
            winners = pnl_pct >= 0
            ax.scatter(dates_num[winners], pnl_pct[winners],
                       c=color, marker=marker, s=100, alpha=0.8, edgecolors='black',
                       linewidths=0.5, label=f'{strat} win', zorder=3)
            ax.scatter(dates_num[~winners], pnl_pct[~winners],
                       c=color, marker=marker, s=100, alpha=0.4, edgecolors='black',
                       linewidths=0.5, zorder=3)
            for idx, (i, row) in enumerate(sdf.iterrows()):
                x = dates_num[idx]
                y = row['PnL'] * 100
                ax.annotate(row['Ticker'], (x, y),
                            fontsize=7, ha='center', va='bottom', alpha=0.7)
        ax.axhline(0, color='black', linewidth=1)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax.set_ylabel('Trade PnL (%)')
        ax.set_title('Individual Trades by Date')
        ax.legend(loc='upper left')
        ax.grid(alpha=0.3)

        # 2. Daily equal-weight PnL bars (capital split equally across all trades that day)
        ax = axes[0, 1]
        all_dates = sorted(trades_df['Date'].unique())
        x_pos = np.arange(len(all_dates))
        bar_w = 0.35
        eqw_long = []; eqw_short = []; eqw_combined = []
        for d in all_dates:
            day_df = trades_df[trades_df['Date'] == d]
            n_trades = len(day_df)
            l_sum = day_df.loc[day_df['Strategy'] == 'ST_LONG', 'PnL'].sum()
            s_sum = day_df.loc[day_df['Strategy'] == 'ST_SHORT', 'PnL'].sum()
            eqw_long.append(l_sum / n_trades * 100 if n_trades > 0 else 0)
            eqw_short.append(s_sum / n_trades * 100 if n_trades > 0 else 0)
            eqw_combined.append((l_sum + s_sum) / n_trades * 100 if n_trades > 0 else 0)
        ax.bar(x_pos - bar_w/2, eqw_long, bar_w, color='#1565C0', alpha=0.8, label='LONG share')
        ax.bar(x_pos + bar_w/2, eqw_short, bar_w, color='#D32F2F', alpha=0.8, label='SHORT share')
        ax.plot(x_pos, eqw_combined, 'ko--', markersize=5, linewidth=1.5, label='Portfolio')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([d[5:] for d in all_dates], rotation=45, ha='right')
        ax.axhline(0, color='black', linewidth=1)
        ax.set_ylabel('Daily Portfolio PnL (%)')
        ax.set_title('Daily PnL — Equal Weight')
        ax.legend()
        ax.grid(alpha=0.3, axis='y')

        # 3. Cumulative PnL by strategy
        ax = axes[1, 0]
        for strat, color in [('ST_LONG', '#1565C0'), ('ST_SHORT', '#D32F2F')]:
            sdf = trades_df[trades_df['Strategy'] == strat].sort_values('Date_dt')
            if len(sdf) > 0:
                cum = sdf['PnL'].cumsum().values
                ax.plot(range(1, len(cum) + 1), cum * 100, color=color,
                        linewidth=2, label=f'{strat} ({len(sdf)} trades)')
        combined_cum = trades_df.sort_values('Date_dt')['PnL'].cumsum().values
        ax.plot(range(1, len(combined_cum) + 1), combined_cum * 100,
                color='#00897B', linewidth=2, linestyle='--', label='COMBINED')
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_xlabel('Trade #')
        ax.set_ylabel('Cumulative PnL (%)')
        ax.set_title(f'Cumulative PnL — {BACKTEST_START} to {BACKTEST_END}')
        ax.legend()
        ax.grid(alpha=0.3)

        # 4. Per-ticker avg PnL
        ax = axes[1, 1]
        ticker_pnl = trades_df.groupby(['Ticker', 'Strategy'])['PnL'].mean().unstack(fill_value=0)
        tickers_sorted = ticker_pnl.sum(axis=1).sort_values().index
        y_pos = np.arange(len(tickers_sorted))
        for i, strat in enumerate(['ST_LONG', 'ST_SHORT']):
            if strat in ticker_pnl.columns:
                vals = [ticker_pnl.loc[t, strat] * 100 if t in ticker_pnl.index else 0
                        for t in tickers_sorted]
                color = '#1565C0' if strat == 'ST_LONG' else '#D32F2F'
                ax.barh(y_pos + i * 0.35, vals, 0.35, color=color, alpha=0.8, label=strat)
        ax.set_yticks(y_pos + 0.175)
        ax.set_yticklabels(tickers_sorted)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel('Avg PnL per trade (%)')
        ax.set_title('Per-Ticker Avg PnL')
        ax.legend()
        ax.grid(alpha=0.3, axis='x')

        # 5. Cumulative equity curve by date (equal-weight daily PnL)
        ax = axes[2, 0]
        daily_eqw = trades_df.groupby('Date')['PnL'].mean().sort_index()
        daily_cum = daily_eqw.cumsum() * 100
        dates_plot = pd.to_datetime(daily_cum.index)
        ax.fill_between(dates_plot, daily_cum.values, 0,
                        where=daily_cum.values >= 0, color='#00897B', alpha=0.2)
        ax.fill_between(dates_plot, daily_cum.values, 0,
                        where=daily_cum.values < 0, color='crimson', alpha=0.2)
        ax.plot(dates_plot, daily_cum.values, color='#00897B', linewidth=2.5, marker='o', markersize=6)
        for i, (d, v) in enumerate(zip(dates_plot, daily_cum.values)):
            ax.annotate(f'{v:+.1f}%', (d, v), fontsize=8, ha='center',
                        va='bottom' if v >= 0 else 'top')
        ax.axhline(0, color='black', linewidth=1)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax.set_ylabel('Cumulative PnL (%)')
        ax.set_title('Portfolio Equity Curve — Equal Weight')
        ax.grid(alpha=0.3)

        # 6. Outcome distribution
        ax = axes[2, 1]
        for strat, color in [('ST_LONG', '#1565C0'), ('ST_SHORT', '#D32F2F')]:
            sdf = trades_df[trades_df['Strategy'] == strat]
            if len(sdf) == 0:
                continue
            outcomes = ['TP', 'SL', 'EOD']
            counts = [(sdf['Outcome'] == o).sum() for o in outcomes]
            ax.bar([f'{o}\n{strat}' for o in outcomes], counts, color=color, alpha=0.7)
        ax.set_ylabel('Count')
        ax.set_title('Outcome Distribution')
        ax.grid(alpha=0.3, axis='y')

        plt.tight_layout()
        save_path = '/tmp/st_last_week_backtest.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n  Chart saved to {save_path}")
        plt.show()
        plt.close()

print(f"\n{'='*W}")
print(f"  BACKTEST COMPLETE")
print(f"{'='*W}")
