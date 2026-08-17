# %% ── SHORT L5-2 Fold 6 + 7: UNION Expansion Analysis ──────────────────────
#
# Deployed: L5-1 Hybrid F5+F6+F7 (HYB_F7_30, 30 tickers)
# F5 comparison proved UNION @ 1.8% is the best combined strategy.
# This script evaluates L5-2 F6 and F7 to expand the UNION pool.
#
# L5-2: 99 tickers (+4 bday date shift) vs L5-1 (64 tickers).
# Each model uses its OWN optimizer configs from BQ and its OWN meta.npz scaling.
# Backtest window per fold = overlap of both families' TEST windows.
#
# Per fold:
#   - L5-2 standalone performance
#   - L5-1 vs L5-2 head-to-head (which tickers does L5-2 improve?)
#   - UNION / CONSENSUS strategies vs deployed L5-1
# Cross-fold:
#   - Aggregate: does adding L5-2 to L5-1 improve the Hybrid?
#   - Deployment recommendation for UNION expansion
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
BUCKET_NAME   = "time-series-bucket-ys"
INFERENCE_DIR = "Data/Inference_v2"
WF_BQ_TABLE   = "ys-time-series.scoring_logs.wf_optimization_results"
PST           = pytz.timezone('America/Los_Angeles')
TAIL_DAYS     = 60

SLIPPAGE_PCT  = 0.0015
EXIT_TIME_MIN = 780
BASE_PNL_THRESHOLD = 0.018
SENSITIVITY_THRESHOLDS = [0.018, 0.015, 0.012, 0.010, 0.008]
SENSITIVITY_MIN = min(SENSITIVITY_THRESHOLDS)

FOLDS = [6, 7]

FAMILY_META = OrderedDict([
    ('L5-1', {'gcs_data': 'Algos/RA_MoE/short_L5_1_data',
              'gcs_models': 'Algos/RA_MoE/short_L5_1_models',
              'tag': 'L5_1', 'label': 'L5-1 (64 tickers)', 'color': '#1565C0'}),
    ('L5-2', {'gcs_data': 'Algos/RA_MoE/short_L5_2_data',
              'gcs_models': 'Algos/RA_MoE/short_L5_2_models',
              'tag': 'L5_2', 'label': 'L5-2 (99 tickers)', 'color': '#FF6F00'}),
])

def bq_model_version(fam_key, fold_id):
    tag = 'L5_1' if fam_key == 'L5-1' else 'L5_2'
    return f'spike_transformer_short_{tag}_fold{fold_id}'

TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}
MICRO_COLS = ['Spread', 'OBI', 'Volume_Delta', 'Trade_Count', 'Delta_Pct',
              'bsize', 'asize', 'Total_Vol']
ROC_HORIZONS = [5, 30, 60]
CONTEXT_ASSETS = ['SPY', 'VIX', 'TLT', 'IWM']
SECTOR_PROXY_ASSETS = ['MARA', 'MSTR', 'ITA']
EXCLUDE_TICKERS = {'SPY', 'VIX', 'TLT', 'IWM', 'XLE', 'ITA', 'BITF', 'MIGI', 'SMLR'}

SCORE_MINUTE = 375
ENTRY_MINUTE = 385

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

W = 110
gcs_client = gcs_storage.Client()


# ==============================================================================
# MODEL ARCHITECTURE (identical to F5 comparison)
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
# MODEL LOADING
# ==============================================================================
def _ensure_file(gcs_path, local_path):
    if os.path.exists(local_path):
        return local_path
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    gcs_client.bucket(BUCKET_NAME).blob(gcs_path).download_to_filename(local_path)
    return local_path


def load_meta(gcs_data_prefix, tag):
    local_meta = f'/tmp/bt_short_{tag}_meta.npz'
    meta_path = _ensure_file(f'{gcs_data_prefix}/meta.npz', local_meta)
    meta = np.load(meta_path, allow_pickle=True)
    return meta['scaling_stats'].item(), meta['regime_stats'].item()


def load_fold_model(gcs_models, tag, fold_id):
    model = SpikeTransformer(name=f'spike_transformer_{tag}_f{fold_id}')
    dummy_seq = tf.zeros((1, TIME_STEPS, N_ASSET_F))
    dummy_reg = tf.zeros((1, N_REGIME_F))
    model({'seq_input': dummy_seq, 'regime_input': dummy_reg})

    local_dir = f'/tmp/bt_short_{tag}_models/fold_{fold_id}'
    weights_path = os.path.join(local_dir, 'model.weights.h5')
    if not os.path.exists(weights_path):
        os.makedirs(local_dir, exist_ok=True)
        prefix = f'{gcs_models}/fold_{fold_id}'
        bucket = gcs_client.bucket(BUCKET_NAME)
        for blob in bucket.list_blobs(prefix=prefix):
            fname = blob.name.split('/')[-1]
            blob.download_to_filename(os.path.join(local_dir, fname))
    model.load_weights(weights_path)
    print(f"    {tag} F{fold_id}: {model.count_params():,} params")
    return model


# ==============================================================================
# BQ CONFIG LOADING
# ==============================================================================
def load_fold_configs(bq_mv, threshold=None):
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
    WHERE model_version = '{bq_mv}'
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
        h_part = exp.replace(f'{bq_mv}_', '').split('_')[0]
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


def get_fold_test_window(fold_id):
    if bigquery is None:
        return None, None
    bq_client = bigquery.Client()
    mvs = [bq_model_version(fk, fold_id) for fk in FAMILY_META]
    mv_list = "', '".join(mvs)
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

    found = {mv: windows[mv] for mv in mvs if mv in windows}
    if len(found) < 2:
        missing = [mv for mv in mvs if mv not in windows]
        print(f"  WARNING: fold {fold_id} missing BQ data for: {missing}")
        if len(found) == 1:
            single = list(found.values())[0]
            return single[0], single[1]
        return None, None

    overlap_start = max(w[0] for w in found.values())
    overlap_end = min(w[1] for w in found.values())

    if overlap_start >= overlap_end:
        print(f"  WARNING: fold {fold_id} TEST windows don't overlap — using union")
        overlap_start = min(w[0] for w in found.values())
        overlap_end = max(w[1] for w in found.values())

    return overlap_start, overlap_end


# ==============================================================================
# SCORING + TRADE SIMULATION
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
# ANALYSIS HELPERS
# ==============================================================================
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


def print_comparison_table(label, s1, s2):
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


def build_combined(l1_trades, l2_trades, l1_cfgs, l2_cfgs, l2_ticker_set, strategy='union'):
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


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
np.random.seed(42)

print(f"\n{'='*W}")
print(f"  SHORT L5-1 vs L5-2: FOLD 6 + 7 MODEL COMPARISON")
print(f"  L5-1: 64 tickers  |  L5-2: 99 tickers (+4 bday shift)")
print(f"  Folds: {FOLDS}")
print(f"  Each model uses its OWN optimizer configs + scaling stats")
print(f"  Slippage: {SLIPPAGE_PCT:.2%} each way ({2*SLIPPAGE_PCT:.2%} RT)")
print(f"{'='*W}")


# ── Phase 0: BQ test windows per fold ──
print(f"\n{'='*W}")
print(f"  PHASE 0: DERIVE FAIR BACKTEST WINDOWS")
print(f"{'='*W}")

fold_bt_windows = {}
fold_trading_days = {}

for fold_id in FOLDS:
    print(f"\n  Fold {fold_id}:")
    bt_start, bt_end = get_fold_test_window(fold_id)
    if bt_start is None or bt_end is None:
        print(f"    SKIPPED — optimizer results not yet in BQ")
        continue
    fold_bt_windows[fold_id] = (bt_start, bt_end)
    print(f"    Fair window: {bt_start} -> {bt_end}")

    for fk in FAMILY_META:
        mv = bq_model_version(fk, fold_id)
        print(f"    {fk} model_version: {mv}")

if not fold_bt_windows:
    print(f"\n  ERROR: No fold windows available. Run optimizers first.")
    sys.exit(1)

active_folds = sorted(fold_bt_windows.keys())
print(f"\n  Active folds with BQ data: {active_folds}")


# ── Phase 1: Load BQ configs for all folds ──
print(f"\n{'='*W}")
print(f"  PHASE 1: LOAD BQ CONFIGS")
print(f"{'='*W}")

fold_configs = {}
all_tickers_needed = set()

for fold_id in active_folds:
    fold_configs[fold_id] = {}
    for fk in FAMILY_META:
        mv = bq_model_version(fk, fold_id)
        thresh = SENSITIVITY_MIN if fk == 'L5-2' else BASE_PNL_THRESHOLD
        cfgs = load_fold_configs(mv, threshold=thresh)
        fold_configs[fold_id][fk] = cfgs
        deployed = {tk for tk, c in cfgs.items() if c['val_b_avg_pnl'] >= BASE_PNL_THRESHOLD}
        borderline = {tk for tk, c in cfgs.items() if c['val_b_avg_pnl'] < BASE_PNL_THRESHOLD}
        all_tickers_needed |= set(cfgs.keys())
        print(f"  F{fold_id} {fk}: {len(deployed)} deployed + {len(borderline)} borderline")
        if deployed:
            print(f"    Deployed:   {', '.join(sorted(deployed))}")
        if borderline:
            bl_detail = ', '.join(f'{tk}={cfgs[tk]["val_b_avg_pnl"]:+.2%}' for tk in sorted(borderline))
            print(f"    Borderline: {bl_detail}")


# ── Phase 2: Market context + regime (once) ──
print(f"\n{'='*W}")
print(f"  PHASE 2: MARKET CONTEXT + REGIME")
print(f"{'='*W}")

market_df = build_market_context()
regime_df = compute_regime_features(market_df)

for fold_id in active_folds:
    bt_start, bt_end = fold_bt_windows[fold_id]
    td = get_trading_days(bt_start, bt_end)
    fold_trading_days[fold_id] = td
    print(f"  F{fold_id}: {len(td)} trading days ({td[0]} -> {td[-1]})")


# ── Phase 3: Feature engineering (once, all tickers) ──
print(f"\n{'='*W}")
print(f"  PHASE 3: FEATURE ENGINEERING ({len(all_tickers_needed)} tickers)")
print(f"{'='*W}")

ticker_features = {}
for ticker in sorted(all_tickers_needed):
    df_feat = engineer_asset(ticker, market_df)
    if df_feat is not None:
        ticker_features[ticker] = df_feat
        print(f"    {ticker}: {len(df_feat):,} bars")
    else:
        print(f"    {ticker}: FAILED")


# ── Phase 4: Score + trade per fold ──
print(f"\n{'='*W}")
print(f"  PHASE 4: SCORE + TRADE SIMULATION")
print(f"{'='*W}")

fold_trades = {}
family_scaling = {}

for fk, fmeta in FAMILY_META.items():
    scaling_s, regime_s = load_meta(fmeta['gcs_data'], fmeta['tag'])
    family_scaling[fk] = (scaling_s, regime_s)

for fold_id in active_folds:
    fold_trades[fold_id] = {}
    trading_days = fold_trading_days[fold_id]

    for fk, fmeta in FAMILY_META.items():
        print(f"\n  F{fold_id} {fmeta['label']}:")
        cfgs = fold_configs[fold_id][fk]
        if not cfgs:
            print(f"    No configs — skipping")
            fold_trades[fold_id][fk] = {}
            continue

        scaling_s, regime_s = family_scaling[fk]
        model = load_fold_model(fmeta['gcs_models'], fmeta['tag'], fold_id)

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

        fold_trades[fold_id][fk] = fam_trade_dict
        print(f"    Scores: {total_scores}  Trades: {total_trades}  Tickers: {len(fam_trade_dict)}")

        del model
        keras.backend.clear_session()
        gc.collect()


# ==============================================================================
# PER-FOLD RESULTS
# ==============================================================================

fold_stats = {}
fold_combined_best = {}

for fold_id in active_folds:
    bt_start, bt_end = fold_bt_windows[fold_id]
    trading_days = fold_trading_days[fold_id]

    print(f"\n\n{'='*W}")
    print(f"  FOLD {fold_id} RESULTS: L5-1 vs L5-2")
    print(f"  Period: {bt_start} -> {bt_end} ({len(trading_days)} days)")
    print(f"{'='*W}")

    l1_cfgs = fold_configs[fold_id]['L5-1']
    l2_cfgs = fold_configs[fold_id]['L5-2']
    l1_tr = fold_trades[fold_id]['L5-1']
    l2_tr = fold_trades[fold_id]['L5-2']

    l1_tickers = set(l1_cfgs.keys())
    l2_deployed = {tk for tk, c in l2_cfgs.items() if c['val_b_avg_pnl'] >= BASE_PNL_THRESHOLD}
    l2_borderline = {tk for tk, c in l2_cfgs.items() if c['val_b_avg_pnl'] < BASE_PNL_THRESHOLD}
    shared = l1_tickers & l2_deployed
    only_l1 = l1_tickers - l2_deployed
    only_l2 = l2_deployed - l1_tickers

    l2_deployed_trades = {tk: trades for tk, trades in l2_tr.items() if tk in l2_deployed}

    s1 = agg_stats(l1_tr)
    s2 = agg_stats(l2_deployed_trades)
    fold_stats[fold_id] = {'L5-1': s1, 'L5-2': s2}

    print_comparison_table(
        f"FULL PORTFOLIO (each model's deployable set)", s1, s2)

    shared_s1 = agg_stats({tk: l1_tr.get(tk, []) for tk in shared if l1_tr.get(tk)})
    shared_s2 = agg_stats({tk: l2_tr.get(tk, []) for tk in shared if l2_tr.get(tk)})
    print_comparison_table(
        f"SHARED TICKERS ONLY ({len(shared)} — pure model quality)", shared_s1, shared_s2)

    # ── Per-ticker: SHARED tickers ──
    if shared:
        print(f"\n\n  SHARED TICKERS ({len(shared)}) — same ticker, different model")
        print(f"  {'Ticker':<7} │{'L5-1 N':>5} {'AvgPnL':>8} {'WR':>5}"
              f" │{'L5-2 N':>5} {'AvgPnL':>8} {'WR':>5}"
              f" │{'Δ Avg':>7} {'Win':>5}")
        print(f"  {'─'*7}─┼{'─'*20}─┼{'─'*20}─┼{'─'*14}")

        tk_l1_wins = 0; tk_l2_wins = 0
        for tk in sorted(shared):
            t1 = l1_tr.get(tk, [])
            t2 = l2_tr.get(tk, [])
            n1, n2 = len(t1), len(t2)
            avg1 = np.mean([t['PnL'] for t in t1]) if t1 else 0
            avg2 = np.mean([t['PnL'] for t in t2]) if t2 else 0
            wr1 = sum(1 for t in t1 if t['PnL'] > 0) / n1 if n1 > 0 else 0
            wr2 = sum(1 for t in t2 if t['PnL'] > 0) / n2 if n2 > 0 else 0
            delta = avg2 - avg1
            winner = 'L5-2' if delta > 0.001 else ('L5-1' if delta < -0.001 else 'TIE')
            if n1 == 0 and n2 == 0:
                winner = '--'
            if delta > 0.001: tk_l2_wins += 1
            elif delta < -0.001: tk_l1_wins += 1
            a1 = f"{avg1:+.2%}" if n1 > 0 else '---'
            a2 = f"{avg2:+.2%}" if n2 > 0 else '---'
            w1 = f"{wr1:.0%}" if n1 > 0 else '---'
            w2 = f"{wr2:.0%}" if n2 > 0 else '---'
            print(f"  {tk:<7} │{n1:>5} {a1:>8} {w1:>5}"
                  f" │{n2:>5} {a2:>8} {w2:>5}"
                  f" │{delta:>+6.2%} {winner:>5}")
        print(f"\n  Shared tally: L5-1 wins {tk_l1_wins}  L5-2 wins {tk_l2_wins}"
              f"  Ties {len(shared) - tk_l1_wins - tk_l2_wins}")

    # ── L5-2 ONLY tickers ──
    if only_l2:
        print(f"\n  L5-2 ONLY ({len(only_l2)}) — new from expanded training")
        new_all = [t for tk in only_l2 for t in l2_tr.get(tk, [])]
        for tk in sorted(only_l2):
            trades = l2_tr.get(tk, [])
            n = len(trades)
            if n > 0:
                avg = np.mean([t['PnL'] for t in trades])
                print(f"    {tk}: {n} trades, avg {avg:+.2%}")
        if new_all:
            print(f"    Total: {len(new_all)} trades at {np.mean([t['PnL'] for t in new_all]):+.2%} avg")

    # ── L5-1 ONLY tickers ──
    if only_l1:
        print(f"\n  L5-1 ONLY ({len(only_l1)}) — deployable in L5-1 but not L5-2")
        for tk in sorted(only_l1):
            trades = l1_tr.get(tk, [])
            n = len(trades)
            if n > 0:
                avg = np.mean([t['PnL'] for t in trades])
                print(f"    {tk}: {n} trades, avg {avg:+.2%}")

    # ── Threshold sensitivity ──
    print(f"\n\n  THRESHOLD SENSITIVITY — F{fold_id}")
    print(f"  {'Threshold':>10} {'Tickers':>8} {'Trades':>7} {'AvgPnL':>8} {'EqW':>8}"
          f" {'vs L5-1 Avg':>12} {'vs L5-1 EqW':>12}")
    print(f"  {'─'*75}")

    for thresh in SENSITIVITY_THRESHOLDS:
        tks_at = {tk for tk, c in l2_cfgs.items() if c['val_b_avg_pnl'] >= thresh}
        trades_at = {tk: trades for tk, trades in l2_tr.items() if tk in tks_at}
        st = agg_stats(trades_at)
        d_avg = st['avg_pnl'] - s1['avg_pnl'] if st['n'] > 0 and s1['n'] > 0 else 0
        d_eqw = st['total_pnl'] - s1['total_pnl'] if st['n'] > 0 and s1['n'] > 0 else 0
        marker = ' ◄' if thresh == BASE_PNL_THRESHOLD else ''
        print(f"  {thresh:>9.1%} {st['n_tickers']:>8} {st['n']:>7} {st['avg_pnl']:>+7.2%}"
              f" {st['total_pnl']:>+7.2%} {d_avg:>+11.2%} {d_eqw:>+11.2%}{marker}")

    # ── Combined portfolio ──
    print(f"\n  COMBINED PORTFOLIO — F{fold_id}")
    strategies = OrderedDict()
    strategies['UNION @ 1.8%'] = build_combined(l1_tr, l2_tr, l1_cfgs, l2_cfgs, l2_deployed, 'union')
    l2_at_12 = {tk for tk, c in l2_cfgs.items() if c['val_b_avg_pnl'] >= 0.012}
    strategies['UNION @ 1.2%'] = build_combined(l1_tr, l2_tr, l1_cfgs, l2_cfgs, l2_at_12, 'union')
    strategies['CONSENSUS'] = build_combined(l1_tr, l2_tr, l1_cfgs, l2_cfgs, l2_deployed, 'consensus')

    print(f"  {'Strategy':<20} {'Tickers':>8} {'Trades':>7} {'AvgPnL':>8} {'EqW':>8}"
          f" {'vs L5-1':>8} {'vs L5-2':>8}")
    print(f"  {'─'*80}")

    for strat_name, (comb_trades, comb_source) in strategies.items():
        st = agg_stats(comb_trades)
        d1 = st['avg_pnl'] - s1['avg_pnl'] if st['n'] > 0 else 0
        d2 = st['avg_pnl'] - s2['avg_pnl'] if st['n'] > 0 else 0
        pf_str = f"{st['pf']:.1f}" if st['pf'] < 999 else 'inf'
        print(f"  {strat_name:<20} {st['n_tickers']:>8} {st['n']:>7} {st['avg_pnl']:>+7.2%}"
              f" {st['total_pnl']:>+7.2%} {d1:>+7.2%} {d2:>+7.2%}")

    best_strat = max(strategies.items(), key=lambda x: agg_stats(x[1][0])['total_pnl'])
    best_st = agg_stats(best_strat[1][0])
    fold_combined_best[fold_id] = (best_strat[0], best_st)
    print(f"\n  >>> F{fold_id} Best: {best_strat[0]} — {best_st['n_tickers']}tk,"
          f" {best_st['n']}t, EqW {best_st['total_pnl']:+.2%} <<<")

    best_trades, best_source = best_strat[1]
    print(f"\n  Per-ticker assignment ({best_strat[0]}):")
    print(f"  {'Ticker':<7} {'Source':>6} {'N':>4} {'AvgPnL':>8}")
    print(f"  {'─'*30}")
    for tk in sorted(best_trades.keys()):
        trades = best_trades[tk]
        src = best_source[tk]
        n = len(trades)
        avg = np.mean([t['PnL'] for t in trades]) if trades else 0
        print(f"  {tk:<7} {src:>6} {n:>4} {avg:>+7.2%}")


# ==============================================================================
# CROSS-FOLD SUMMARY
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  CROSS-FOLD SUMMARY: L5-2 vs L5-1 across F{'+F'.join(str(f) for f in active_folds)}")
print(f"{'='*W}")

print(f"\n  {'Fold':>6} │{'L5-1':^35}│{'L5-2':^35}│ {'Δ Avg':>7} {'Δ EqW':>7}")
print(f"  {'':>6} │{'Tk':>4} {'N':>4} {'Avg':>7} {'EqW':>8} {'WR':>5}"
      f" │{'Tk':>4} {'N':>4} {'Avg':>7} {'EqW':>8} {'WR':>5} │")
print(f"  {'─'*6}─┼{'─'*35}┼{'─'*35}┼{'─'*16}")

agg_l1_trades = {}
agg_l2_trades = {}

for fold_id in active_folds:
    s1 = fold_stats[fold_id]['L5-1']
    s2 = fold_stats[fold_id]['L5-2']
    d_avg = s2['avg_pnl'] - s1['avg_pnl'] if s1['n'] > 0 and s2['n'] > 0 else 0
    d_eqw = s2['total_pnl'] - s1['total_pnl'] if s1['n'] > 0 and s2['n'] > 0 else 0
    print(f"  F{fold_id:>4} │{s1['n_tickers']:>4} {s1['n']:>4} {s1['avg_pnl']:>+6.2%}"
          f" {s1['total_pnl']:>+7.2%} {s1['wr']:>4.0%}"
          f" │{s2['n_tickers']:>4} {s2['n']:>4} {s2['avg_pnl']:>+6.2%}"
          f" {s2['total_pnl']:>+7.2%} {s2['wr']:>4.0%}"
          f" │{d_avg:>+6.2%} {d_eqw:>+6.2%}")

    for tk, trades in fold_trades[fold_id]['L5-1'].items():
        agg_l1_trades.setdefault(tk, []).extend(trades)
    l2_dep = {tk for tk, c in fold_configs[fold_id]['L5-2'].items()
              if c['val_b_avg_pnl'] >= BASE_PNL_THRESHOLD}
    for tk, trades in fold_trades[fold_id]['L5-2'].items():
        if tk in l2_dep:
            agg_l2_trades.setdefault(tk, []).extend(trades)

agg_s1 = agg_stats(agg_l1_trades)
agg_s2 = agg_stats(agg_l2_trades)
d_avg = agg_s2['avg_pnl'] - agg_s1['avg_pnl'] if agg_s1['n'] > 0 and agg_s2['n'] > 0 else 0
d_eqw = agg_s2['total_pnl'] - agg_s1['total_pnl'] if agg_s1['n'] > 0 and agg_s2['n'] > 0 else 0
print(f"  {'─'*6}─┼{'─'*35}┼{'─'*35}┼{'─'*16}")
print(f"  {'ALL':>6} │{agg_s1['n_tickers']:>4} {agg_s1['n']:>4} {agg_s1['avg_pnl']:>+6.2%}"
      f" {agg_s1['total_pnl']:>+7.2%} {agg_s1['wr']:>4.0%}"
      f" │{agg_s2['n_tickers']:>4} {agg_s2['n']:>4} {agg_s2['avg_pnl']:>+6.2%}"
      f" {agg_s2['total_pnl']:>+7.2%} {agg_s2['wr']:>4.0%}"
      f" │{d_avg:>+6.2%} {d_eqw:>+6.2%}")

print(f"\n  Combined portfolio best per fold:")
for fold_id in active_folds:
    strat_name, st = fold_combined_best[fold_id]
    print(f"    F{fold_id}: {strat_name} — {st['n_tickers']}tk, {st['n']}t,"
          f" Avg {st['avg_pnl']:+.2%}, EqW {st['total_pnl']:+.2%}")


# ==============================================================================
# CHARTS
# ==============================================================================
if HAS_MPL:
    n_active = len(active_folds)
    fig, axes = plt.subplots(2, n_active + 1, figsize=(10 * (n_active + 1), 14))
    if n_active + 1 == 2:
        axes = axes.reshape(2, 2)
    fig.suptitle(f'SHORT L5-1 vs L5-2: Fold {"+".join(str(f) for f in active_folds)} Comparison\n'
                 f'Slippage {2*SLIPPAGE_PCT:.2%} RT',
                 fontsize=14, fontweight='bold')

    for fi, fold_id in enumerate(active_folds):
        bt_start, bt_end = fold_bt_windows[fold_id]

        # Row 0: Equity curves per fold
        ax = axes[0, fi]
        for fk, fmeta in FAMILY_META.items():
            l2_dep = {tk for tk, c in fold_configs[fold_id][fk].items()
                      if c['val_b_avg_pnl'] >= BASE_PNL_THRESHOLD} if fk == 'L5-2' else set(fold_configs[fold_id][fk].keys())
            if fk == 'L5-2':
                trade_dict = {tk: trades for tk, trades in fold_trades[fold_id][fk].items() if tk in l2_dep}
            else:
                trade_dict = fold_trades[fold_id][fk]
            all_t = [t for trades in trade_dict.values() for t in trades]
            if not all_t:
                continue
            tdf = pd.DataFrame(all_t)
            daily_eqw = tdf.groupby('Date')['PnL'].mean().sort_index()
            cum = daily_eqw.cumsum() * 100
            dates_plot = pd.to_datetime(cum.index)
            ax.plot(dates_plot, cum.values, color=fmeta['color'], linewidth=2.5,
                    label=f"{fmeta['label']} ({len(all_t)}t, {cum.iloc[-1]:+.1f}%)")
        ax.axhline(0, color='black', linewidth=0.8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax.set_ylabel('Cumulative EqW PnL (%)')
        ax.set_title(f'Fold {fold_id}: Equity Curves\n{bt_start} to {bt_end}')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        # Row 1: Per-ticker avg PnL (shared tickers)
        ax = axes[1, fi]
        l1_cfgs_f = fold_configs[fold_id]['L5-1']
        l2_cfgs_f = fold_configs[fold_id]['L5-2']
        l2_dep_f = {tk for tk, c in l2_cfgs_f.items() if c['val_b_avg_pnl'] >= BASE_PNL_THRESHOLD}
        shared_f = set(l1_cfgs_f.keys()) & l2_dep_f
        if shared_f:
            tk_sorted = sorted(shared_f,
                               key=lambda tk: (np.mean([t['PnL'] for t in fold_trades[fold_id]['L5-2'].get(tk, [])]) if fold_trades[fold_id]['L5-2'].get(tk) else 0) -
                                              (np.mean([t['PnL'] for t in fold_trades[fold_id]['L5-1'].get(tk, [])]) if fold_trades[fold_id]['L5-1'].get(tk) else 0),
                               reverse=True)
            y_pos = np.arange(len(tk_sorted))
            bar_h = 0.35
            for i, fk in enumerate(FAMILY_META):
                vals = []
                for tk in tk_sorted:
                    trades = fold_trades[fold_id][fk].get(tk, [])
                    vals.append(np.mean([t['PnL'] for t in trades]) * 100 if trades else 0)
                ax.barh(y_pos + i * bar_h, vals, bar_h,
                        label=FAMILY_META[fk]['label'],
                        color=FAMILY_META[fk]['color'], alpha=0.85,
                        edgecolor='black', linewidth=0.5)
            ax.set_yticks(y_pos + bar_h / 2)
            ax.set_yticklabels(tk_sorted, fontsize=7)
            ax.axvline(0, color='black', linewidth=1)
            ax.invert_yaxis()
        ax.set_xlabel('Avg PnL per Trade (%)')
        ax.set_title(f'Fold {fold_id}: Shared Tickers Avg PnL')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis='x')

    # Cross-fold summary column
    ax = axes[0, n_active]
    for fk, fmeta in FAMILY_META.items():
        agg_tr = agg_l1_trades if fk == 'L5-1' else agg_l2_trades
        all_t = [t for trades in agg_tr.values() for t in trades]
        if not all_t:
            continue
        tdf = pd.DataFrame(all_t)
        daily_eqw = tdf.groupby('Date')['PnL'].mean().sort_index()
        cum = daily_eqw.cumsum() * 100
        dates_plot = pd.to_datetime(cum.index)
        ax.plot(dates_plot, cum.values, color=fmeta['color'], linewidth=2.5,
                label=f"{fmeta['label']} ({len(all_t)}t, {cum.iloc[-1]:+.1f}%)")
    ax.axhline(0, color='black', linewidth=0.8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax.set_ylabel('Cumulative EqW PnL (%)')
    ax.set_title(f'ALL FOLDS: Combined Equity')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Cross-fold bar chart
    ax = axes[1, n_active]
    fold_labels = [f'F{f}' for f in active_folds] + ['ALL']
    x = np.arange(len(fold_labels))
    bar_w = 0.35
    for i, fk in enumerate(FAMILY_META):
        fmeta = FAMILY_META[fk]
        vals = []
        for fold_id in active_folds:
            vals.append(fold_stats[fold_id][fk]['avg_pnl'] * 100)
        agg_s = agg_s1 if fk == 'L5-1' else agg_s2
        vals.append(agg_s['avg_pnl'] * 100)
        bars = ax.bar(x + i * bar_w, vals, bar_w, label=fmeta['label'],
                      color=fmeta['color'], alpha=0.85, edgecolor='black', linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f'{v:+.1f}%', ha='center', fontsize=8, fontweight='bold')
    ax.set_xticks(x + bar_w / 2)
    ax.set_xticklabels(fold_labels, fontsize=10)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylabel('Avg PnL per Trade (%)')
    ax.set_title('Avg PnL by Fold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = '/tmp/SHORT_L5_1_vs_L5_2_fold67_comparison.png'
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

l2_fold_wins = 0
l1_fold_wins = 0
neutral_folds = 0

for fold_id in active_folds:
    s1 = fold_stats[fold_id]['L5-1']
    s2 = fold_stats[fold_id]['L5-2']
    avg_d = s2['avg_pnl'] - s1['avg_pnl'] if s1['n'] > 0 and s2['n'] > 0 else 0
    eqw_d = s2['total_pnl'] - s1['total_pnl'] if s1['n'] > 0 and s2['n'] > 0 else 0

    print(f"\n  ── FOLD {fold_id} ──")
    print(f"  L5-1: {s1['n']}t, {s1['n_tickers']}tk | Avg {s1['avg_pnl']:+.2%} | EqW {s1['total_pnl']:+.2%}")
    print(f"  L5-2: {s2['n']}t, {s2['n_tickers']}tk | Avg {s2['avg_pnl']:+.2%} | EqW {s2['total_pnl']:+.2%}")
    print(f"  Delta: Avg {avg_d:+.2%}, EqW {eqw_d:+.2%}")

    strat_name, best_st = fold_combined_best[fold_id]
    vs_l1 = best_st['total_pnl'] - s1['total_pnl']
    vs_l2 = best_st['total_pnl'] - s2['total_pnl']
    print(f"  Best combined: {strat_name} — EqW {best_st['total_pnl']:+.2%}"
          f" (vs L5-1: {vs_l1:+.2%}, vs L5-2: {vs_l2:+.2%})")

    if avg_d > 0.002:
        print(f"  Decision: L5-2 WINS F{fold_id}")
        l2_fold_wins += 1
    elif avg_d < -0.002:
        print(f"  Decision: L5-1 HOLDS F{fold_id}")
        l1_fold_wins += 1
    else:
        print(f"  Decision: NEUTRAL F{fold_id} (within noise)")
        neutral_folds += 1

print(f"\n  ── AGGREGATE ({'+'.join(f'F{f}' for f in active_folds)}) ──")
print(f"  L5-1: {agg_s1['n']}t, {agg_s1['n_tickers']}tk | Avg {agg_s1['avg_pnl']:+.2%} | EqW {agg_s1['total_pnl']:+.2%}")
print(f"  L5-2: {agg_s2['n']}t, {agg_s2['n_tickers']}tk | Avg {agg_s2['avg_pnl']:+.2%} | EqW {agg_s2['total_pnl']:+.2%}")
agg_avg_d = agg_s2['avg_pnl'] - agg_s1['avg_pnl'] if agg_s1['n'] > 0 and agg_s2['n'] > 0 else 0
agg_eqw_d = agg_s2['total_pnl'] - agg_s1['total_pnl'] if agg_s1['n'] > 0 and agg_s2['n'] > 0 else 0
print(f"  Delta: Avg {agg_avg_d:+.2%}, EqW {agg_eqw_d:+.2%}")

print(f"\n  ── DEPLOYMENT RECOMMENDATION ──")
print(f"  Deployed: L5-1 Hybrid F5+F6+F7 (HYB_F7_30)")
print(f"  L5-2 is ADDITIVE — UNION proven best on F5 (see backtest_short_L5_1_vs_L5_2_combinations.py)")
print(f"  Per-fold: L5-2 wins {l2_fold_wins}, L5-1 wins {l1_fold_wins}, Neutral {neutral_folds}")

print(f"\n  Per-fold best combination strategy:")
for fold_id in active_folds:
    strat_name, st = fold_combined_best[fold_id]
    s1 = fold_stats[fold_id]['L5-1']
    uplift = st['total_pnl'] - s1['total_pnl']
    print(f"    F{fold_id}: {strat_name} — EqW {st['total_pnl']:+.2%}"
          f" ({uplift:+.2%} vs deployed L5-1)")

any_combined_beats = any(
    fold_combined_best[f][1]['total_pnl'] > fold_stats[f]['L5-1']['total_pnl']
    for f in active_folds
)
l2_stronger = agg_s2['avg_pnl'] > agg_s1['avg_pnl']

if l2_stronger:
    print(f"\n  >>> L5-2 is STRONGER than L5-1 on F6+F7 (Avg {agg_avg_d:+.2%}) <<<")
    print(f"  ACTION: Add L5-2 F6+F7 models to scoring pipeline.")
    print(f"  Use UNION per-ticker assignment (best BT PnL wins shared tickers).")
    print(f"  Consistent with F5 finding: UNION @ 1.8% was the winner.")
elif any_combined_beats:
    print(f"\n  >>> L5-2 adds value via combination strategies <<<")
    print(f"  ACTION: Add L5-2 to scoring pipeline, use per-fold best strategy.")
else:
    print(f"\n  >>> L5-2 does NOT improve F6+F7 — keep deployed L5-1 Hybrid <<<")

print(f"\n  Full deployment: L5-1 HYB_F7_30 + L5-2 UNION across F5+F6+F7")

print(f"\n{'='*W}")
