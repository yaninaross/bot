# %% ── ST vs S3 LONG-Only Backtest (Live Scoring) ────────────────────────────
#
# Three strategies compared head-to-head:
#   1. S3_LONG       — UNIVERSAL model, live inference, limit_entry LONG
#   2. SPIKE_TRANS   — Spike Transformer fold 4, live inference, LONG
#   3. COMBINED      — S3_LONG + SPIKE_TRANS combined portfolio
#
# Both models score LIVE (no CSV scoring).
# GPU RECOMMENDED. Colab-ready.

import os, io, gc, sys, json, time, ctypes, warnings
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np, pandas as pd, pytz

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import (Input, Dense, Dropout, Embedding, Flatten,
                                     Concatenate, MultiHeadAttention, Add,
                                     LayerNormalization, GlobalAveragePooling1D)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

try:
    import pandas_ta as ta
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas_ta",
                           "--no-deps", "--quiet"])
    import pandas_ta as ta

try:
    from numba import jit
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numba", "--quiet"])
    from numba import jit

import matplotlib
try:
    from google.colab import auth
    _IN_COLAB = True
    auth.authenticate_user()
except (ImportError, Exception):
    matplotlib.use('Agg')
    _IN_COLAB = False
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from google.cloud import storage as gcs_storage, bigquery


# ==============================================================================
# 1. CONFIG
# ==============================================================================
BUCKET_NAME   = os.environ.get("GCS_BUCKET", "")
GCS_MODEL_DIR = "Algos/to_deploy/Saved_models_v2/UNIVERSAL/Direction"
INFERENCE_DIR = "Data/Inference_v2"

TEST_START  = '2026-04-01'
TEST_END    = '2026-08-08'
HALF2_START = '2026-06-01'

DEPLOY_FLOOR    = 0.010
LONG_SLIPPAGE_RT = 0.003   # 0.15% one-way for LONG
EXIT_TIME_MIN   = 780
MIN_H2_TRADES   = 3

TIME_STEPS     = 240
EMBED_DIM      = 16
SEEDS_PER_FOLD = 15

PST = pytz.timezone('America/Los_Angeles')
TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

BQ_TABLE = f"{GCP_PROJECT}.scoring_logs.wf_optimization_results"

MICRO_COLS = ['Spread', 'OBI', 'Volume_Delta', 'Trade_Count', 'Delta_Pct',
              'bsize', 'asize', 'Total_Vol']
ROC_HORIZONS = [5, 30, 60, 120]
CONTEXT_ASSETS = ['SPY', 'VIX', 'TLT', 'IWM']

ASSET_COLS_10AM = (['Gap_Pct', 'RSI_14', 'RSI_60', 'RVOL', 'Compression',
                    'ATR_14', 'Dist_to_PM_High', 'Bar_Is_Real',
                    'Morning_Ext', 'Gap_VIX'] +
                   [f'ROC_{h}m' for h in ROC_HORIZONS] + MICRO_COLS)

CONTEXT_COLS_10AM = []
for _t in CONTEXT_ASSETS:
    CONTEXT_COLS_10AM.append(f'{_t}_Anchor_ROC')
    for _h in ROC_HORIZONS:
        CONTEXT_COLS_10AM.append(f'{_t}_ROC_{_h}m')
CONTEXT_COLS_10AM += ['Time_Pos', 'VIX_Level']

ALL_COLS_10AM = ASSET_COLS_10AM + CONTEXT_COLS_10AM

# ── S3 LONG model feature columns (must match training scripts) ──
S3_ASSET_COLS = (['Gap_Pct', 'RSI_14', 'RSI_60', 'RVOL', 'Compression',
                  'ATR_14', 'Dist_to_PM_High', 'Bar_Is_Real'] +
                 [f'ROC_{h}m' for h in ROC_HORIZONS] + MICRO_COLS)
S3_CONTEXT_COLS = []
for _t in CONTEXT_ASSETS:
    S3_CONTEXT_COLS.append(f'{_t}_Anchor_ROC')
    for _h in ROC_HORIZONS:
        S3_CONTEXT_COLS.append(f'{_t}_ROC_{_h}m')
S3_CONTEXT_COLS += ['Time_Pos']
S3_ALL_COLS = S3_ASSET_COLS + S3_CONTEXT_COLS
S3_SKIP_EXPS = {'200_XSECTOR'}

# ── S3 LONG gate constants ──
S3_MODEL_VERSIONS = ('robust_wf_v12', 'robust_wf_v12_stitched')
S3_MIN_SEEDS     = 3
S3_MAX_VA_OPT    = 3.0
S3_DEPLOY_THRESHOLD = 0.018
S3_EXCLUDE       = {'LUNR'}
S3_GRP_KEYS = ['ticker', 'experiment', 'tpsl_mode', 'entry_minute',
               'prob_thresh', 'tp_pct', 'sl_pct',
               'tp_mult', 'sl_mult', 'trail_start_mult', 'trail_end_mult',
               'grace_minutes', 'dip_pct', 'max_wait_minutes', 'vol_filter_pct']
S3_ENTRY_MINUTE = 375
EXCLUDE_TICKERS = {'SPY', 'VIX', 'TLT', 'IWM', 'XLE', 'ITA', 'BITF', 'MIGI', 'SMLR'}

# ── Spike Transformer config ──
GCS_DATA_PREFIX    = 'Algos/RA_MoE/real_data'
GCS_STMODEL_PREFIX = 'Algos/RA_MoE/real_models'
ST_DATA_DIR  = '/tmp/ra_moe_real_data'
ST_MODEL_DIR = '/tmp/ra_moe_models'
os.makedirs(ST_DATA_DIR, exist_ok=True)
os.makedirs(ST_MODEL_DIR, exist_ok=True)

ST_D_MODEL = 64; ST_N_HEADS = 4; ST_D_FF = 128; ST_N_BLOCKS = 2
ST_DROPOUT = 0.1; ST_FILM_HIDDEN = 64; ST_TIME_STEPS = 128
ST_N_HORIZONS = 3
ST_ASL_GAMMA_NEG = 5.0; ST_ASL_GAMMA_POS = 0.1; ST_ASL_CLIP = 0.10
ST_FOLD_ID = 4
ST_ENTRY_MINUTE = 375
ST_SCORE_HORIZONS = [0, 1, 2]
ST_HORIZON_LABELS = {0: '15m', 1: '30m', 2: '60m'}
ST_HORIZON_FROM_LABEL = {'15m': 0, '30m': 1, '60m': 2}
ST_DEPLOY_THRESHOLD = 0.018

# ── Strategy colors ──
STRATEGY_COLORS = {
    'S3_LONG':     '#1565C0',
    'SPIKE_TRANS': '#FF6600',
    'COMBINED':    '#00897B',
}
STRAT_ORDER = ['S3_LONG', 'SPIKE_TRANS', 'COMBINED']

gcs_client = gcs_storage.Client()
bq_client  = bigquery.Client()
W = 120


# ==============================================================================
# 2. HELPERS
# ==============================================================================
def _release_memory():
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


_master_cache = {}


def load_master(ticker):
    if ticker in _master_cache:
        return _master_cache[ticker]
    clean = ticker.replace("I:", "").replace("X:", "")
    file_ticker = TICKER_FILE_MAP.get(clean, clean)
    uri = f"gs://{BUCKET_NAME}/{INFERENCE_DIR}/{file_ticker}_master_inference.parquet"
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
        for c in MICRO_COLS:
            if c not in df.columns:
                df[c] = 0.0
        df = df[df.index.notna()].sort_index()
        _master_cache[ticker] = df
        return df
    except Exception as e:
        print(f"    load_master({ticker}): FAILED -- {e}")
        return None


def heal_data(df):
    df = df.resample('1min').asfreq()
    for c in ['Open', 'High', 'Low', 'Close']:
        if c in df.columns:
            df[c] = df[c].ffill()
    for c in ['Volume'] + MICRO_COLS:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    return df


# ==============================================================================
# 3. UNIVERSAL MODEL (for S3 live scoring)
# ==============================================================================
def detect_n_assets(exp_id, seed_base):
    seed = seed_base + 1
    blob_path = f"{GCS_MODEL_DIR}/UNIVERSAL_S{seed}_M{exp_id}.h5"
    local_path = f"/tmp/UNIVERSAL_S{seed}_M{exp_id}.h5"
    bucket = gcs_client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_path)
    if not os.path.exists(local_path):
        blob.download_to_filename(local_path)
    elif os.path.getsize(local_path) != blob.size:
        os.remove(local_path)
        blob.download_to_filename(local_path)
    import h5py
    with h5py.File(local_path, 'r') as f:
        def _find_embedding(group):
            for key in group.keys():
                item = group[key]
                if isinstance(item, h5py.Dataset) and 'embedding' in key.lower():
                    return item.shape[0]
                if isinstance(item, h5py.Group):
                    result = _find_embedding(item)
                    if result is not None:
                        return result
            return None
        n = _find_embedding(f)
    if n is None:
        raise RuntimeError(f"Cannot detect N_ASSETS from {blob_path}")
    return n


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.4):
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    res = Add()([x, inputs])
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    return Add()([x, res])


def build_model(n_asset_features, n_context_features, n_assets=31):
    in_a  = Input(shape=(TIME_STEPS, n_asset_features), name='asset_in')
    in_c  = Input(shape=(n_context_features,),          name='context_in')
    in_id = Input(shape=(1,), dtype=tf.int32,           name='asset_id')
    x = transformer_encoder(Dense(128)(in_a), 128, 4, 128)
    x = GlobalAveragePooling1D()(transformer_encoder(x, 128, 4, 128))
    y = Dense(128, activation='relu')(LayerNormalization()(in_c))
    e = Embedding(n_assets, EMBED_DIM, name='asset_embed')(in_id)
    e = Flatten()(e)
    e = Dense(32, activation='relu')(e)
    z = Concatenate()([x, Dropout(0.5)(y), e])
    z = Dense(128, activation='relu', kernel_regularizer=l2(5e-3))(z)
    out = Dense(1, activation='sigmoid', dtype='float32')(Dropout(0.5)(z))
    return Model(inputs=[in_a, in_c, in_id], outputs=out)


def determine_features(scaling_stats):
    """Determine asset and context columns from scaling stats keys."""
    all_10am = ASSET_COLS_10AM + CONTEXT_COLS_10AM
    if all(c in scaling_stats for c in all_10am):
        return ASSET_COLS_10AM, CONTEXT_COLS_10AM
    stat_cols = sorted(scaling_stats.keys())
    context = [c for c in stat_cols
               if any(c.startswith(a + '_') for a in CONTEXT_ASSETS)
               or c in ('Time_Pos', 'VIX_Level')]
    asset = [c for c in stat_cols if c not in context]
    return asset, context


def load_s3_seed_models(exp_id, seed_base, n_ac, n_cc, n_assets):
    """Load 15 seed models for a single S3 experiment."""
    seeds = list(range(seed_base + 1, seed_base + SEEDS_PER_FOLD + 1))
    bucket = gcs_client.bucket(BUCKET_NAME)
    models = []
    for seed in seeds:
        blob_path = f"{GCS_MODEL_DIR}/UNIVERSAL_S{seed}_M{exp_id}.h5"
        local_path = f"/tmp/UNIVERSAL_S{seed}_M{exp_id}.h5"
        try:
            blob = bucket.blob(blob_path)
            if not os.path.exists(local_path):
                blob.download_to_filename(local_path)
            elif os.path.getsize(local_path) != blob.size:
                os.remove(local_path)
                blob.download_to_filename(local_path)
            tf.keras.backend.clear_session()
            model = build_model(n_ac, n_cc, n_assets=n_assets)
            model.compile(optimizer='adam', loss='binary_crossentropy')
            model.load_weights(local_path)
            models.append(model)
        except Exception as e:
            print(f"      Seed {seed}: FAILED -- {e}")
    return models


# ==============================================================================
# 3B. SPIKE TRANSFORMER MODEL LAYERS
# ==============================================================================

class SeriesStationarization(layers.Layer):
    def call(self, x):
        mean = tf.reduce_mean(x, axis=1, keepdims=True)
        std  = tf.math.reduce_std(x, axis=1, keepdims=True) + 1e-5
        x_norm = (x - mean) / std
        stats = tf.concat([mean[:, 0, :], std[:, 0, :]], axis=-1)
        return x_norm, stats

class DeStationaryAttention(layers.Layer):
    def __init__(self, d_model, n_heads, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model; self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = tf.sqrt(tf.cast(self.d_head, tf.float32))
        self.wq = Dense(d_model); self.wk = Dense(d_model)
        self.wv = Dense(d_model); self.wo = Dense(d_model)
        self.attn_drop = Dropout(dropout)
        self.tau_proj = Dense(n_heads)
        self.sigma_proj = Dense(d_model)

    def call(self, x, stats, training=False):
        B = tf.shape(x)[0]; T = tf.shape(x)[1]
        H, D = self.n_heads, self.d_head
        Q = tf.transpose(tf.reshape(self.wq(x), (B, T, H, D)), (0, 2, 1, 3))
        K = tf.transpose(tf.reshape(self.wk(x), (B, T, H, D)), (0, 2, 1, 3))
        V = tf.transpose(tf.reshape(self.wv(x), (B, T, H, D)), (0, 2, 1, 3))
        scores = tf.matmul(Q, K, transpose_b=True) / self.scale
        tau = tf.reshape(self.tau_proj(stats), (B, H, 1, 1))
        scores = scores + tau
        attn = self.attn_drop(tf.nn.softmax(scores, axis=-1), training=training)
        out = tf.matmul(attn, V)
        out = tf.reshape(tf.transpose(out, (0, 2, 1, 3)), (B, T, self.d_model))
        sigma = tf.nn.softplus(self.sigma_proj(stats))
        out = out * sigma[:, tf.newaxis, :]
        return self.wo(out)

class TransformerBlock(layers.Layer):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.attn = DeStationaryAttention(d_model, n_heads, dropout)
        self.ffn = tf.keras.Sequential([
            Dense(d_ff, activation='gelu'), Dense(d_model), Dropout(dropout)])
        self.norm1 = LayerNormalization()
        self.norm2 = LayerNormalization()
        self.drop = Dropout(dropout)

    def call(self, x, stats, training=False):
        x = x + self.drop(self.attn(self.norm1(x), stats, training=training),
                          training=training)
        x = x + self.ffn(self.norm2(x), training=training)
        return x

class TransformerBackbone(layers.Layer):
    def __init__(self, d_model, n_heads, d_ff, n_blocks, seq_len,
                 dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.stationarize = SeriesStationarization()
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

class AttentionPooling(layers.Layer):
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.query = self.add_weight(
            name='pool_query', shape=(1, 1, d_model),
            initializer='glorot_normal', trainable=True)
        self.scale = tf.sqrt(tf.cast(d_model, tf.float32))

    def call(self, x, training=False):
        scores = tf.matmul(self.query, x, transpose_b=True) / self.scale
        weights = tf.nn.softmax(scores, axis=-1)
        pooled = tf.matmul(weights, x)
        return pooled[:, 0, :]

class FiLMConditioner(layers.Layer):
    def __init__(self, d_model, hidden_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.mlp = tf.keras.Sequential([
            Dense(hidden_dim, activation='relu'),
            Dense(hidden_dim, activation='relu')])
        self.gamma_proj = Dense(d_model)
        self.beta_proj = Dense(d_model)

    def call(self, x, regime_features, training=False):
        h = self.mlp(regime_features, training=training)
        gamma = 1.0 + self.gamma_proj(h)
        beta = self.beta_proj(h)
        return gamma * x + beta

class SpikeTransformer(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.backbone = TransformerBackbone(
            ST_D_MODEL, ST_N_HEADS, ST_D_FF, ST_N_BLOCKS,
            ST_TIME_STEPS, ST_DROPOUT, name='backbone')
        self.attn_pool = AttentionPooling(ST_D_MODEL, name='attn_pool')
        self.film = FiLMConditioner(ST_D_MODEL, ST_FILM_HIDDEN, name='film')
        self.drop = Dropout(ST_DROPOUT)
        self.head = Dense(ST_N_HORIZONS, activation='sigmoid', name='spike_head')

    def call(self, inputs, training=False):
        seq = inputs['seq_input']
        regime = inputs['regime_input']
        h = self.backbone(seq, training=training)
        h = self.attn_pool(h, training=training)
        h = self.film(h, regime, training=training)
        h = self.drop(h, training=training)
        return self.head(h)

    def train_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            preds = self(x, training=True)
            loss = st_asymmetric_focal_loss(y, preds, sample_weight)
        self.optimizer.apply_gradients(
            zip(tape.gradient(loss, self.trainable_variables),
                self.trainable_variables))
        return {'loss': loss}

    def test_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        preds = self(x, training=False)
        loss = st_asymmetric_focal_loss(y, preds, sample_weight)
        return {'loss': loss}

def st_asymmetric_focal_loss(y_true, y_pred, sample_weight=None):
    p = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    pos_loss = -y_true * tf.pow(1.0 - p, ST_ASL_GAMMA_POS) * tf.math.log(p)
    p_m = tf.maximum(p - ST_ASL_CLIP, 0.0)
    neg_loss = -(1.0 - y_true) * tf.pow(p_m, ST_ASL_GAMMA_NEG) * tf.math.log(
        tf.clip_by_value(1.0 - p_m, 1e-7, 1.0))
    per_sample = tf.reduce_mean(pos_loss + neg_loss, axis=-1)
    if sample_weight is not None:
        per_sample = per_sample * sample_weight
    return tf.reduce_mean(per_sample)


def ensure_st_data(filename):
    local_path = os.path.join(ST_DATA_DIR, filename)
    if os.path.exists(local_path):
        return local_path
    gcs_path = f'{GCS_DATA_PREFIX}/{filename}'
    print(f"    Downloading {filename} from GCS...")
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    gcs_client.bucket(BUCKET_NAME).blob(gcs_path).download_to_filename(local_path)
    return local_path

def ensure_st_model(fold_id):
    local_dir = os.path.join(ST_MODEL_DIR, f'fold_{fold_id}')
    weights_path = os.path.join(local_dir, 'model.weights.h5')
    if os.path.exists(weights_path):
        return weights_path
    os.makedirs(local_dir, exist_ok=True)
    gcs_prefix = f'{GCS_STMODEL_PREFIX}/fold_{fold_id}'
    print(f"    Downloading fold {fold_id} model from GCS...")
    bucket = gcs_client.bucket(BUCKET_NAME)
    for blob in bucket.list_blobs(prefix=gcs_prefix):
        fname = blob.name.split('/')[-1]
        blob.download_to_filename(os.path.join(local_dir, fname))
    return weights_path


# ==============================================================================
# 4. FEATURE ENGINEERING
# ==============================================================================
def build_market_context(include_vix_level=False):
    frames = []
    vix_close = None
    for ticker in CONTEXT_ASSETS:
        df = load_master(ticker)
        if df is None:
            raise RuntimeError(f"Cannot load context asset {ticker}")
        df = df.sort_index()
        df = heal_data(df)
        if ticker == 'VIX':
            vix_close = df['Close'].copy()
        daily_c = df['Close'].resample('D').last().ffill().shift(1).reindex(
            df.index, method='ffill')
        cols = {f'{ticker}_Anchor_ROC': (df['Close'] - daily_c) / daily_c}
        for h in ROC_HORIZONS:
            cols[f'{ticker}_ROC_{h}m'] = df['Close'].pct_change(h).fillna(0)
        frames.append(pd.DataFrame(cols, index=df.index))
        del df, daily_c
    _master_cache.clear()
    result = pd.concat(frames, axis=1).ffill().fillna(0)
    if include_vix_level and vix_close is not None:
        result['VIX_Level'] = vix_close.reindex(result.index).ffill().fillna(0)
    return result


def engineer_features_10am(df, market_df):
    df_pst = df.index.tz_convert(PST)
    mins = df_pst.hour * 60 + df_pst.minute
    dates = df_pst.date

    daily_c = df['Close'].resample('D').last().ffill().shift(1).reindex(
        df.index, method='ffill')
    df['Gap_Pct'] = (df['Open'] - daily_c) / daily_c
    df['PM_High'] = df.groupby(df.index.date)['High'].cummax()
    df['Dist_to_PM_High'] = (df['Close'] - df['PM_High']) / (df['PM_High'] + 1e-9)
    df['RSI_14'] = ta.rsi(df['Close'], length=14)
    df['RSI_60'] = ta.rsi(df['Close'], length=60)
    vol_real = df['Volume'].where(df['Volume'] > 0)
    df['RVOL'] = df['Volume'] / vol_real.rolling(50, min_periods=5).mean().ffill().clip(lower=1e-9)
    real_mask = df['Volume'] > 0
    atr_raw = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['ATR_14'] = atr_raw.where(real_mask).ffill().fillna(0)
    df['Compression'] = (df['High'] - df['Low']) / (df['ATR_14'] + 1e-9)
    for h in ROC_HORIZONS:
        df[f'ROC_{h}m'] = df['Close'].pct_change(h)

    open_window = (mins >= 388) & (mins <= 400) & (df['Volume'] > 0)
    temp_dates = pd.Series(dates, index=df.index)
    if open_window.any():
        daily_open_930 = df.loc[open_window].groupby(temp_dates[open_window])['Open'].first()
        mapped_open = temp_dates.map(daily_open_930)
    else:
        mapped_open = pd.Series(np.nan, index=df.index)

    after_open = mins >= 390
    df['Morning_Ext'] = np.where(
        after_open & (mapped_open.fillna(0) > 0),
        (df['Close'] / mapped_open.fillna(1).clip(lower=1e-8)) - 1,
        0.0
    )

    df = df.join(market_df, how='left').ffill().fillna(0)
    vix_level = df['VIX_Level'] if 'VIX_Level' in df.columns else pd.Series(0.0, index=df.index)
    df['Gap_VIX'] = df['Gap_Pct'] * vix_level

    df_pst2 = df.index.tz_convert(PST)
    mins2 = df_pst2.hour * 60 + df_pst2.minute
    df['Time_Pos'] = np.clip((mins2 - 390) / 390, 0, 1)
    df['Bar_Is_Real'] = (df['Volume'] > 0).astype(float)
    return df.replace([np.inf, -np.inf], np.nan).fillna(0)


# ==============================================================================
# 5. SCALING
# ==============================================================================
def load_scaling_stats(gcs_path):
    blob = gcs_client.bucket(BUCKET_NAME).blob(gcs_path)
    return json.loads(blob.download_as_text())


def apply_scaling(df, stats, all_cols):
    df_s = df.copy()
    for col in all_cols:
        if col in ('Time_Pos', 'Bar_Is_Real'):
            continue
        if col in stats:
            df_s[col] = ((df_s[col] - stats[col]['med'])
                         / stats[col]['iqr']).clip(-10, 10).astype('float32')
    return df_s


# ==============================================================================
# 6. SCORING
# ==============================================================================
def batch_score_ticker(df_scaled, entry_indices, models, asset_cols, context_cols, other_idx):
    XA = df_scaled[asset_cols].values.astype(np.float32)
    XC = df_scaled[context_cols].values.astype(np.float32)
    valid = [i for i in entry_indices if i >= TIME_STEPS]
    if not valid:
        return {}
    valid = np.array(valid)
    XA_batch = np.array([XA[i - TIME_STEPS:i] for i in valid])
    XC_batch = np.array([XC[i - 1] for i in valid])
    XID_batch = np.full((len(valid), 1), other_idx, dtype=np.int32)

    all_seed_probs = []
    for model in models:
        preds = model.predict(
            {'asset_in': XA_batch, 'context_in': XC_batch, 'asset_id': XID_batch},
            batch_size=min(len(valid), 4096), verbose=0
        ).flatten()
        all_seed_probs.append(preds)

    ensemble = np.mean(all_seed_probs, axis=0)
    idx_pst = df_scaled.index.tz_convert(PST)
    result = {}
    for j, i in enumerate(valid):
        d = idx_pst[i].date()
        result[d] = float(ensemble[j])
    return result


# ==============================================================================
# 7. LONG TRADE SIMULATION (Numba)
# ==============================================================================

@jit(nopython=True)
def run_trade_sim_static(entry_indices, prices, highs, lows, date_ints, time_mins,
                         tp, sl, exit_time):
    n_entries = len(entry_indices)
    n_prices  = len(prices)
    trades      = np.empty(n_entries, dtype=np.float64)
    entry_dates = np.empty(n_entries, dtype=np.int64)
    trade_count       = 0
    current_entry_ptr = 0
    last_traded_date  = -1
    while current_entry_ptr < n_entries:
        idx = entry_indices[current_entry_ptr]
        if idx >= n_prices - 1: break
        entry_date = date_ints[idx]
        if entry_date == last_traded_date:
            current_entry_ptr += 1; continue
        entry_price  = prices[idx]
        target_price = entry_price * (1 + tp)
        stop_price   = entry_price * (1 - sl)
        outcome    = 0.0
        trade_done = False
        exit_idx   = idx
        for k in range(idx + 1, n_prices):
            if date_ints[k] != entry_date:
                outcome = (prices[k-1] - entry_price) / entry_price
                exit_idx = k; trade_done = True; break
            sl_hit = lows[k] <= stop_price
            tp_hit = highs[k] >= target_price
            if sl_hit and tp_hit:
                outcome = -sl if np.random.random() < 0.5 else tp
                exit_idx = k; trade_done = True; break
            if sl_hit:
                outcome = -sl; exit_idx = k; trade_done = True; break
            if tp_hit:
                outcome = tp; exit_idx = k; trade_done = True; break
            if time_mins[k] >= exit_time:
                outcome = (prices[k] - entry_price) / entry_price
                exit_idx = k; trade_done = True; break
        if not trade_done:
            outcome = (prices[n_prices-1] - entry_price) / entry_price
            exit_idx = n_prices
        trades[trade_count] = outcome
        entry_dates[trade_count] = entry_date
        trade_count += 1
        last_traded_date = entry_date
        next_ptr = current_entry_ptr + 1
        while next_ptr < n_entries:
            if entry_indices[next_ptr] > exit_idx: break
            next_ptr += 1
        current_entry_ptr = next_ptr
    return trades[:trade_count], entry_dates[:trade_count]


@jit(nopython=True)
def run_trade_sim_grace_long(entry_indices, prices, highs, lows, date_ints,
                             time_mins, tp, sl, exit_time, grace_minutes):
    n_entries = len(entry_indices)
    n_prices  = len(prices)
    trades      = np.empty(n_entries, dtype=np.float64)
    entry_dates = np.empty(n_entries, dtype=np.int64)
    trade_count       = 0
    current_entry_ptr = 0
    last_traded_date  = -1
    while current_entry_ptr < n_entries:
        idx = entry_indices[current_entry_ptr]
        if idx >= n_prices - 1: break
        entry_date = date_ints[idx]
        if entry_date == last_traded_date:
            current_entry_ptr += 1; continue
        entry_price  = prices[idx]
        target_price = entry_price * (1 + tp)
        stop_price   = entry_price * (1 - sl)
        entry_time   = time_mins[idx]
        outcome    = 0.0
        trade_done = False
        exit_idx   = idx
        for k in range(idx + 1, n_prices):
            if date_ints[k] != entry_date:
                outcome = (prices[k-1] - entry_price) / entry_price
                exit_idx = k; trade_done = True; break
            minutes_elapsed = time_mins[k] - entry_time
            sl_active = minutes_elapsed >= grace_minutes
            tp_hit = highs[k] >= target_price
            sl_hit = sl_active and (lows[k] <= stop_price)
            if sl_hit and tp_hit:
                outcome = -sl if np.random.random() < 0.5 else tp
                exit_idx = k; trade_done = True; break
            if sl_hit:
                outcome = -sl; exit_idx = k; trade_done = True; break
            if tp_hit:
                outcome = tp; exit_idx = k; trade_done = True; break
            if time_mins[k] >= exit_time:
                outcome = (prices[k] - entry_price) / entry_price
                exit_idx = k; trade_done = True; break
        if not trade_done:
            outcome = (prices[n_prices-1] - entry_price) / entry_price
            exit_idx = n_prices
        trades[trade_count] = outcome
        entry_dates[trade_count] = entry_date
        trade_count += 1
        last_traded_date = entry_date
        next_ptr = current_entry_ptr + 1
        while next_ptr < n_entries:
            if entry_indices[next_ptr] > exit_idx: break
            next_ptr += 1
        current_entry_ptr = next_ptr
    return trades[:trade_count], entry_dates[:trade_count]


@jit(nopython=True)
def run_trade_sim_limit_entry(entry_indices, prices, highs, lows, date_ints,
                              time_mins, tp, sl, exit_time, dip_pct,
                              max_wait_minutes):
    n_entries = len(entry_indices)
    n_prices  = len(prices)
    trades      = np.empty(n_entries, dtype=np.float64)
    entry_dates = np.empty(n_entries, dtype=np.int64)
    trade_count       = 0
    current_entry_ptr = 0
    last_traded_date  = -1

    while current_entry_ptr < n_entries:
        idx = entry_indices[current_entry_ptr]
        if idx >= n_prices - 1: break
        entry_date = date_ints[idx]
        if entry_date == last_traded_date:
            current_entry_ptr += 1; continue
        signal_price = prices[idx]
        limit_price  = signal_price * (1.0 - dip_pct)
        signal_time  = time_mins[idx]

        fill_idx   = -1
        fill_price = 0.0
        for k in range(idx + 1, n_prices):
            if date_ints[k] != entry_date: break
            if time_mins[k] - signal_time > max_wait_minutes: break
            if lows[k] <= limit_price:
                fill_price = limit_price; fill_idx = k; break

        if fill_idx < 0:
            current_entry_ptr += 1
            last_traded_date = entry_date; continue

        target_price = fill_price * (1 + tp)
        stop_price   = fill_price * (1 - sl)
        outcome    = 0.0
        trade_done = False
        exit_idx   = fill_idx

        for k in range(fill_idx + 1, n_prices):
            if date_ints[k] != entry_date:
                outcome = (prices[k-1] - fill_price) / fill_price
                exit_idx = k; trade_done = True; break
            sl_hit = lows[k] <= stop_price
            tp_hit = highs[k] >= target_price
            if sl_hit and tp_hit:
                outcome = -sl if np.random.random() < 0.5 else tp
                exit_idx = k; trade_done = True; break
            if sl_hit:
                outcome = -sl; exit_idx = k; trade_done = True; break
            if tp_hit:
                outcome = tp; exit_idx = k; trade_done = True; break
            if time_mins[k] >= exit_time:
                outcome = (prices[k] - fill_price) / fill_price
                exit_idx = k; trade_done = True; break

        if not trade_done:
            outcome = (prices[n_prices-1] - fill_price) / fill_price
            exit_idx = n_prices
        trades[trade_count] = outcome
        entry_dates[trade_count] = entry_date
        trade_count += 1
        last_traded_date = entry_date
        next_ptr = current_entry_ptr + 1
        while next_ptr < n_entries:
            if entry_indices[next_ptr] > exit_idx: break
            next_ptr += 1
        current_entry_ptr = next_ptr

    return trades[:trade_count], entry_dates[:trade_count]


def dispatch_long_trade_sim(indices, ohlcv, cfg):
    prices = ohlcv['prices']; highs = ohlcv['highs']; lows = ohlcv['lows']
    dates = ohlcv['dates']; times = ohlcv['times'].astype(np.int64)
    mode = cfg['mode']
    if mode == 'grace':
        pnl, d = run_trade_sim_grace_long(
            indices, prices, highs, lows, dates, times,
            cfg['tp_pct'], cfg['sl_pct'], EXIT_TIME_MIN, cfg['grace_minutes'])
    elif mode == 'limit_entry':
        pnl, d = run_trade_sim_limit_entry(
            indices, prices, highs, lows, dates, times,
            cfg['tp_pct'], cfg['sl_pct'], EXIT_TIME_MIN,
            cfg['dip_pct'], cfg['max_wait_minutes'])
    else:
        pnl, d = run_trade_sim_static(
            indices, prices, highs, lows, dates, times,
            cfg['tp_pct'], cfg['sl_pct'], EXIT_TIME_MIN)
    if len(pnl) > 0:
        pnl = pnl - LONG_SLIPPAGE_RT
    return pnl, d


# ==============================================================================
# 8. OHLCV HELPERS
# ==============================================================================
def precompute_ohlcv_arrays(df, start_date, end_date):
    s = pd.to_datetime(start_date).tz_localize(PST)
    e = pd.to_datetime(end_date).tz_localize(PST) + timedelta(days=1)
    idx = df.index
    if str(idx.tz) != 'America/Los_Angeles':
        idx = idx.tz_convert(PST)
    mask = (idx >= s) & (idx < e)
    df_cut = df[mask]
    if len(df_cut) == 0:
        return None
    idx_cut = df_cut.index if str(df_cut.index.tz) == 'America/Los_Angeles' \
        else df_cut.index.tz_convert(PST)
    return {
        'prices': df_cut['Close'].values.astype(np.float64).copy(),
        'highs':  df_cut['High'].values.astype(np.float64).copy(),
        'lows':   df_cut['Low'].values.astype(np.float64).copy(),
        'dates':  (idx_cut.year * 10000 + idx_cut.month * 100 + idx_cut.day).values.astype(np.int64),
        'times':  (idx_cut.hour * 60 + idx_cut.minute).values.astype(np.int64),
    }


def load_ticker_ohlcv(ticker):
    if ticker in _master_cache:
        return _master_cache[ticker]
    clean = TICKER_FILE_MAP.get(ticker, ticker)
    for suffix in ['_master_inference', '_1m_full_historical_data']:
        uri = f"gs://{BUCKET_NAME}/{INFERENCE_DIR}/{clean}{suffix}.parquet"
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
            df = df[df.index.notna()].sort_index()
            df = df[~df.index.duplicated(keep='first')]
            df.index = df.index.tz_convert(PST)
            std_map = {'open': 'Open', 'high': 'High', 'low': 'Low',
                       'close': 'Close', 'volume': 'Volume'}
            lower_map = {k.lower(): v for k, v in std_map.items()}
            df.columns = [lower_map.get(str(c).strip().lower(), str(c).strip())
                          for c in df.columns]
            df = df.loc[:, ~df.columns.duplicated(keep='first')].copy()
            _master_cache[ticker] = df
            return df
        except Exception:
            continue
    return None


def precompute_ohlcv_arrays_pst(df_ohlcv, start_date, end_date):
    eval_start = pd.to_datetime(start_date).tz_localize('America/Los_Angeles')
    eval_end   = pd.to_datetime(end_date).tz_localize('America/Los_Angeles') + timedelta(days=1)
    df = df_ohlcv[(df_ohlcv.index >= eval_start) & (df_ohlcv.index < eval_end)]
    if len(df) == 0:
        return None
    idx = df.index
    if str(idx.tz) != 'America/Los_Angeles':
        idx = idx.tz_convert(PST)
    dates = (idx.year * 10000 + idx.month * 100 + idx.day).values.astype(np.int64)
    times = (idx.hour * 60 + idx.minute).values.astype(np.int64)
    return {
        'prices': df['Close'].values.astype(np.float64).copy(),
        'highs':  df['High'].values.astype(np.float64).copy(),
        'lows':   df['Low'].values.astype(np.float64).copy(),
        'dates':  dates,
        'times':  times,
    }


# ==============================================================================
# 9. BQ CONFIG LOADING
# ==============================================================================
def assign_family(exp):
    base = exp.split('_')[0]
    if not base.isdigit():
        return 'other'
    num = int(base)
    if num == 152: return '152'
    if num == 160: return '160'
    if 170 <= num <= 175: return '170'
    if 180 <= num <= 185: return '180'
    if 200 <= num <= 205: return '200'
    if 220 <= num <= 225: return '220'
    return 'other'


def load_bq_configs_s3():
    print(f"    Loading S3 LONG configs from BQ (family-agreement gate)...")
    s3_q = f"""
    SELECT ticker, experiment, seed, tpsl_mode, entry_minute,
           prob_thresh, tp_pct, sl_pct, tp_mult, sl_mult,
           trail_start_mult, trail_end_mult, grace_minutes,
           dip_pct, max_wait_minutes, vol_filter_pct,
           val_a_avg_pnl, val_a_win_rate, val_a_count, val_a_freq,
           val_b_avg_pnl, opt_avg_pnl, is_ensemble
    FROM `{BQ_TABLE}`
    WHERE model_version IN {str(S3_MODEL_VERSIONS)}
      AND survived_a = TRUE
      AND val_a_count >= 3
      AND val_a_avg_pnl > 0
    """
    raw_s3 = bq_client.query(s3_q).to_dataframe()
    print(f"    {len(raw_s3)} raw BQ rows")

    raw_s3 = raw_s3[raw_s3['opt_avg_pnl'] > 0].copy()
    raw_s3 = raw_s3[raw_s3['val_a_avg_pnl'] / raw_s3['opt_avg_pnl'] <= S3_MAX_VA_OPT]
    indiv = raw_s3[raw_s3['is_ensemble'] == False].copy()
    for k in S3_GRP_KEYS:
        if k not in indiv.columns: indiv[k] = 0
        else: indiv[k] = indiv[k].fillna(0)

    sa = indiv.groupby(S3_GRP_KEYS).agg(
        n_seeds=('seed', 'nunique'),
        n_pos_seeds_18=('val_a_avg_pnl', lambda x: (x >= S3_DEPLOY_THRESHOLD).sum()),
        mean_va_pnl=('val_a_avg_pnl', 'mean'),
        mean_va_wr=('val_a_win_rate', 'mean'),
        vb_pnl=('val_b_avg_pnl', 'mean'),
    ).reset_index()
    sa['agree_18'] = (sa['n_pos_seeds_18'] / sa['n_seeds']).clip(upper=1.0)

    sa = sa[sa['n_seeds'] >= S3_MIN_SEEDS].copy()
    sa_limit = sa[sa['tpsl_mode'] == 'limit_entry'].copy()
    sa_limit['score'] = (sa_limit['mean_va_pnl'] ** 2) * sa_limit['agree_18']
    sa_limit['exp_fam'] = sa_limit['experiment'].apply(assign_family)
    sa_fam = sa_limit[sa_limit['exp_fam'] != 'other'].copy()

    best_per_fam = sa_fam.loc[sa_fam.groupby(['ticker', 'exp_fam'])['score'].idxmax()]
    best_per_fam = best_per_fam[best_per_fam['mean_va_pnl'] >= S3_DEPLOY_THRESHOLD]

    agreement = {}
    for tk in best_per_fam['ticker'].unique():
        if tk in S3_EXCLUDE: continue
        tk_rows = best_per_fam[best_per_fam['ticker'] == tk]
        n_fams = len(tk_rows)
        n_pos = int((tk_rows['vb_pnl'] > 0).sum())
        agreement[tk] = {'n_fams': n_fams, 'n_pos': n_pos,
                         'majority_pos': n_pos > n_fams / 2 and n_fams >= 2}

    s3_gate_tickers = sorted([tk for tk, ag in agreement.items() if ag['majority_pos']])
    print(f"    Family agreement: {len(s3_gate_tickers)} tickers pass S3 gate")

    sa_s3 = sa_limit[sa_limit['ticker'].isin(s3_gate_tickers)].copy()
    sa_s3['score'] = (sa_s3['mean_va_pnl'] ** 2) * sa_s3['agree_18']
    best_s3 = sa_s3.loc[sa_s3.groupby('ticker')['score'].idxmax()]
    best_s3 = best_s3[best_s3['mean_va_pnl'] >= S3_DEPLOY_THRESHOLD]

    s3_configs = {}
    for _, row in best_s3.iterrows():
        tk = row['ticker']
        s3_configs[tk] = {
            'prob_thresh': float(row['prob_thresh']),
            'tp_pct': float(row['tp_pct']),
            'sl_pct': float(row['sl_pct']),
            'mode': 'limit_entry',
            'dip_pct': float(row['dip_pct']),
            'max_wait_minutes': float(row['max_wait_minutes']),
            'grace_minutes': 0,
            'experiment': row['experiment'],
            'entry_minute': int(row['entry_minute']) if pd.notna(row['entry_minute']) else S3_ENTRY_MINUTE,
            'val_a_pnl': float(row['mean_va_pnl']),
        }

    del raw_s3, indiv, sa, sa_limit, sa_fam, best_per_fam, sa_s3, best_s3
    _release_memory()
    return s3_configs


# ==============================================================================
# 10. PORTFOLIO ANALYSIS HELPERS
# ==============================================================================
def summarize_trades(trades_list, label):
    if not trades_list:
        return {'label': label, 'n': 0, 'total': 0, 'avg': 0, 'wr': 0,
                'dd': 0, 'sharpe': 0, 'trades_df': pd.DataFrame(),
                'h1_n': 0, 'h1_avg': 0, 'h1_total': 0,
                'h2_n': 0, 'h2_avg': 0, 'h2_total': 0, 'h2_wr': 0,
                'monthly': {}, 'n_tickers': 0}
    df = pd.DataFrame(trades_list)
    df['month'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m')
    cum = df.sort_values('date')['pnl'].cumsum()
    max_dd = (cum - cum.cummax()).min()
    h1 = df[df['half'] == 'H1']
    h2 = df[df['half'] == 'H2']
    daily_pnl = df.groupby('date')['pnl'].sum()
    sharpe = (daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)
              if daily_pnl.std() > 0 else 0)
    return {
        'label': label, 'n': len(df),
        'total': df['pnl'].sum(), 'avg': df['pnl'].mean(),
        'wr': (df['pnl'] > 0).mean(), 'dd': max_dd, 'sharpe': float(sharpe),
        'trades_df': df,
        'n_tickers': df['ticker'].nunique(),
        'h1_n': len(h1), 'h1_avg': h1['pnl'].mean() if len(h1) > 0 else 0,
        'h1_total': h1['pnl'].sum() if len(h1) > 0 else 0,
        'h2_n': len(h2), 'h2_avg': h2['pnl'].mean() if len(h2) > 0 else 0,
        'h2_total': h2['pnl'].sum() if len(h2) > 0 else 0,
        'h2_wr': (h2['pnl'] > 0).mean() if len(h2) > 0 else 0,
        'monthly': df.groupby('month').agg(
            n=('pnl', 'count'), total=('pnl', 'sum'),
            avg=('pnl', 'mean'), wr=('pnl', lambda x: (x > 0).mean())
        ).to_dict('index'),
    }


def compute_capital_adjusted(trades_list, deploy_floor=None):
    if not trades_list:
        return {'n_days': 0, 'daily_avg': 0, 'daily_total': 0, 'daily_wr': 0,
                'daily_dd': 0, 'avg_positions': 0, 'daily_series': pd.Series(dtype=float),
                'h2_daily_avg': 0, 'h2_daily_total': 0, 'h2_n_days': 0}
    df = pd.DataFrame(trades_list)
    if deploy_floor is not None:
        tk_avg = df.groupby('ticker')['pnl'].mean()
        good_tickers = tk_avg[tk_avg >= deploy_floor].index
        df = df[df['ticker'].isin(good_tickers)]
    if len(df) == 0:
        return {'n_days': 0, 'daily_avg': 0, 'daily_total': 0, 'daily_wr': 0,
                'daily_dd': 0, 'avg_positions': 0, 'daily_series': pd.Series(dtype=float),
                'h2_daily_avg': 0, 'h2_daily_total': 0, 'h2_n_days': 0}
    daily = df.groupby('date').agg(
        day_pnl=('pnl', 'mean'),
        n_positions=('pnl', 'count'),
    )
    daily.index = pd.to_datetime(daily.index)
    daily = daily.sort_index()
    cum = daily['day_pnl'].cumsum()
    max_dd = (cum - cum.cummax()).min()
    h2_mask = daily.index >= pd.Timestamp(HALF2_START)
    h2_daily = daily[h2_mask]
    return {
        'n_days': len(daily),
        'daily_avg': daily['day_pnl'].mean(),
        'daily_total': daily['day_pnl'].sum(),
        'daily_wr': (daily['day_pnl'] > 0).mean(),
        'daily_dd': max_dd,
        'avg_positions': daily['n_positions'].mean(),
        'daily_series': daily['day_pnl'],
        'h2_daily_avg': h2_daily['day_pnl'].mean() if len(h2_daily) > 0 else 0,
        'h2_daily_total': h2_daily['day_pnl'].sum() if len(h2_daily) > 0 else 0,
        'h2_n_days': len(h2_daily),
    }


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
np.random.seed(42)
t_total = time.time()

print(f"\n{'='*W}")
print(f"  ST vs S3 LONG-ONLY BACKTEST (LIVE SCORING)")
print(f"  TEST: {TEST_START} -> {TEST_END}  |  H2: {HALF2_START}+")
print(f"{'='*W}")
print(f"  LEG 1: S3 LONG -- LIVE model inference, limit_entry")
print(f"  LEG 2: Spike Transformer -- LONG live inference (fold {ST_FOLD_ID})")
print(f"  Deploy threshold: {DEPLOY_FLOOR:.1%}")
print(f"  Slippage LONG: {LONG_SLIPPAGE_RT/2:.2%} one-way ({LONG_SLIPPAGE_RT:.3%} RT)")
print(f"  GPU: {tf.config.list_physical_devices('GPU')}")


# ── Step 1: Load BQ configs ─────────────────────────────────────────────────
print(f"\n  STEP 1: Loading BQ configs...")

s3_configs = load_bq_configs_s3()
s3_tickers = sorted(s3_configs.keys())
print(f"    {len(s3_tickers)} S3 tickers: {', '.join(s3_tickers)}")
for tk in s3_tickers[:5]:
    c = s3_configs[tk]
    print(f"      {tk:<7} {c['experiment']:<20} Prob>{c['prob_thresh']:.2f}  "
          f"TP={c['tp_pct']:.0%}  SL={c['sl_pct']:.0%}  "
          f"dip={c['dip_pct']:.1%}  wait={c['max_wait_minutes']:.0f}m  "
          f"valA={c['val_a_pnl']:+.2%}")
if len(s3_tickers) > 5:
    print(f"      ... +{len(s3_tickers)-5} more")

# Group S3 tickers by experiment for live scoring
s3_by_exp = defaultdict(list)
for tk in s3_tickers:
    exp_raw = s3_configs[tk]['experiment']
    exp_base = exp_raw.replace('_LIMIT', '').replace('_GRACE', '')
    s3_by_exp[exp_base].append(tk)

print(f"\n    S3 experiments to load: {len(s3_by_exp)}")
for exp_id, tks in sorted(s3_by_exp.items()):
    print(f"      {exp_id}: {', '.join(tks)}")

# Spike Transformer BQ configs
print(f"\n    Loading Spike Transformer configs from BQ...")
st_q = f"""
SELECT ticker, experiment, tpsl_mode, prob_thresh, tp_pct, sl_pct,
       grace_minutes, dip_pct, max_wait_minutes,
       opt_avg_pnl, val_a_avg_pnl, val_b_avg_pnl, val_b_win_rate, val_b_count
FROM `{BQ_TABLE}`
WHERE model_version = 'spike_transformer_fold4'
  AND survived_a = TRUE
"""
st_bq = bq_client.query(st_q).to_dataframe()
print(f"    {len(st_bq)} BQ rows for spike_transformer_fold4")

st_deploy = st_bq[st_bq['val_b_avg_pnl'] >= ST_DEPLOY_THRESHOLD].copy()
print(f"    {len(st_deploy)} deployable (val_b >= {ST_DEPLOY_THRESHOLD:.1%})")

st_configs = {}
if len(st_deploy) > 0:
    best_st = st_deploy.loc[st_deploy.groupby('ticker')['val_b_avg_pnl'].idxmax()]
    for _, row in best_st.iterrows():
        exp = row['experiment']
        h_part = exp.replace('spike_transformer_fold4_', '').split('_')[0]
        horizon_idx = ST_HORIZON_FROM_LABEL.get(h_part, 1)
        mode = row['tpsl_mode']
        if pd.isna(mode) or mode == '':
            if '_GRACE' in exp: mode = 'grace'
            elif '_LIMIT' in exp: mode = 'limit_entry'
            else: mode = 'static'
        st_configs[row['ticker']] = {
            'horizon_idx': horizon_idx,
            'horizon_label': h_part,
            'prob_thresh': float(row['prob_thresh']),
            'tp_pct': float(row['tp_pct']),
            'sl_pct': float(row['sl_pct']),
            'mode': mode,
            'grace_minutes': int(row['grace_minutes']) if pd.notna(row['grace_minutes']) else 0,
            'dip_pct': float(row['dip_pct']) if pd.notna(row['dip_pct']) else 0,
            'max_wait_minutes': int(row['max_wait_minutes']) if pd.notna(row['max_wait_minutes']) else 0,
            'val_b_avg_pnl': float(row['val_b_avg_pnl']),
            'experiment': exp,
        }

st_tickers_list = sorted(st_configs.keys())
print(f"    {len(st_tickers_list)} ST tickers selected:")
for tk in st_tickers_list:
    c = st_configs[tk]
    print(f"      {tk:<7} {c['mode']:<13} H={c['horizon_label']}  "
          f"Prob>{c['prob_thresh']:.2f}  TP={c['tp_pct']:.0%}  SL={c['sl_pct']:.0%}")
del st_bq, st_deploy; _release_memory()


# ── Step 2: Build market context ────────────────────────────────────────────
print(f"\n  STEP 2: Building market context...")
t0 = time.time()
market_df_10am = build_market_context(include_vix_level=True)
print(f"    Market context: {market_df_10am.shape} ({time.time()-t0:.1f}s)")
_master_cache.clear()


# ── Step 3: S3 LONG live scoring ────────────────────────────────────────────
print(f"\n  STEP 3: S3 LONG -- LIVE model inference...")

s3_trades = []
s3_per_tk = defaultdict(list)
test_start_dt = pd.Timestamp(TEST_START).date()
test_end_dt   = pd.Timestamp(TEST_END).date()
h2_start_dt   = pd.Timestamp(HALF2_START).date()
lookback_start = pd.Timestamp(TEST_START, tz='UTC') - pd.Timedelta(days=10)

for exp_id, tickers_in_exp in sorted(s3_by_exp.items()):
    if exp_id in S3_SKIP_EXPS:
        print(f"\n  ── Experiment {exp_id} -- SKIPPED (different architecture) ──")
        continue
    exp_num = int(exp_id.split('_')[0])
    seed_base = exp_num * 100
    print(f"\n  ── Experiment {exp_id} (seed_base={seed_base}) ──")
    print(f"     Tickers: {', '.join(tickers_in_exp)}")

    # Load scaling stats
    scaling_path = f'{INFERENCE_DIR}/scaling_stats_experiment_{exp_id}.json'
    try:
        stats = load_scaling_stats(scaling_path)
        print(f"     Scaling stats: {len(stats)} features")
    except Exception as e:
        print(f"     ERROR loading scaling stats: {e}")
        print(f"     Skipping experiment {exp_id}")
        continue

    asset_cols, context_cols = S3_ASSET_COLS, S3_CONTEXT_COLS
    all_cols = S3_ALL_COLS
    print(f"     Features: {len(asset_cols)} asset + {len(context_cols)} context = {len(all_cols)}")

    # Detect n_assets and load models
    try:
        n_assets = detect_n_assets(exp_id, seed_base)
        print(f"     N_ASSETS={n_assets}")
    except Exception as e:
        print(f"     ERROR detecting N_ASSETS: {e}")
        continue

    t0 = time.time()
    models = load_s3_seed_models(exp_id, seed_base, len(asset_cols), len(context_cols), n_assets)
    print(f"     Models loaded: {len(models)}/{SEEDS_PER_FOLD} ({time.time()-t0:.0f}s)")

    if not models:
        print(f"     ERROR: No models loaded -- skipping")
        continue

    other_idx = n_assets - 1

    # Score each ticker in this experiment group
    for ticker in sorted(tickers_in_exp):
        cfg = s3_configs[ticker]
        entry_min = cfg.get('entry_minute', S3_ENTRY_MINUTE)

        df = load_master(ticker)
        if df is None:
            print(f"     {ticker}: SKIP -- load_master failed")
            continue
        df = df[df.index >= lookback_start]
        df = heal_data(df)
        df = engineer_features_10am(df, market_df_10am)

        for col in all_cols:
            if col not in df.columns:
                df[col] = 0.0

        df_scaled = apply_scaling(df, stats, all_cols)

        # Find entry bars in test period
        df_test = df_scaled[df_scaled.index >= pd.Timestamp(TEST_START, tz='UTC')]
        df_test = df_test[df_test.index <= pd.Timestamp(TEST_END, tz='UTC') + pd.Timedelta(days=1)]
        idx_pst = df_test.index.tz_convert(PST)
        mins = idx_pst.hour * 60 + idx_pst.minute
        entry_mask = (mins == entry_min)
        entry_positions = np.where(entry_mask)[0]

        global_entry_indices = []
        for pos in entry_positions:
            global_idx = df_scaled.index.get_loc(df_test.index[pos])
            global_entry_indices.append(global_idx)

        if not global_entry_indices:
            print(f"     {ticker}: SKIP -- 0 entry bars at min={entry_min} "
                  f"(df_scaled={len(df_scaled)}, df_test={len(df_test)}, "
                  f"unique_mins={sorted(set(mins.values))[:5]}...)")
            _master_cache.pop(ticker, None)
            continue

        daily_probs = batch_score_ticker(
            df_scaled, global_entry_indices, models, asset_cols, context_cols,
            other_idx=other_idx)

        if not daily_probs:
            print(f"     {ticker}: SKIP -- batch_score empty "
                  f"(entries={len(global_entry_indices)}, "
                  f"min_idx={min(global_entry_indices)}, TIME_STEPS={TIME_STEPS})")
            _master_cache.pop(ticker, None)
            continue

        # Load OHLCV and run trade sim
        df_ohlcv = load_ticker_ohlcv(ticker)
        if df_ohlcv is None:
            print(f"     {ticker}: SKIP -- load_ticker_ohlcv returned None")
            _master_cache.pop(ticker, None)
            continue
        df_ohlcv = heal_data(df_ohlcv.sort_index())
        ohlcv = precompute_ohlcv_arrays_pst(df_ohlcv, TEST_START, TEST_END)
        if ohlcv is None:
            print(f"     {ticker}: SKIP -- precompute_ohlcv returned None (tz={df_ohlcv.index.tz})")
            _master_cache.pop(ticker, None)
            continue

        dates_arr = ohlcv['dates']; times_arr = ohlcv['times']
        probs = np.zeros(len(dates_arr), dtype=np.float64)
        for i in range(len(dates_arr)):
            if times_arr[i] == entry_min:
                d_int = int(dates_arr[i])
                d_str = f"{d_int // 10000}-{(d_int % 10000) // 100:02d}-{d_int % 100:02d}"
                d_date = pd.Timestamp(d_str).date()
                probs[i] = daily_probs.get(d_date, 0.0)

        mask = (probs >= cfg['prob_thresh']) & (times_arr == entry_min)
        indices = np.where(mask)[0].astype(np.int64)

        if len(indices) == 0:
            n_scored = len(daily_probs)
            prob_vals = list(daily_probs.values())
            above = sum(1 for p in prob_vals if p >= cfg['prob_thresh'])
            p_min, p_max, p_mean = min(prob_vals), max(prob_vals), np.mean(prob_vals)
            print(f"     {ticker}: {n_scored} scored, {above} above thresh={cfg['prob_thresh']:.2f}, "
                  f"prob range=[{p_min:.4f}, {p_max:.4f}], mean={p_mean:.4f}")
            _master_cache.pop(ticker, None)
            continue

        np.random.seed(42)
        pnl_arr, trade_dates = run_trade_sim_limit_entry(
            indices, ohlcv['prices'], ohlcv['highs'], ohlcv['lows'],
            ohlcv['dates'], ohlcv['times'].astype(np.int64),
            cfg['tp_pct'], cfg['sl_pct'], EXIT_TIME_MIN,
            cfg['dip_pct'], cfg['max_wait_minutes'])

        if len(pnl_arr) > 0:
            pnl_arr = pnl_arr - LONG_SLIPPAGE_RT

        for j in range(len(pnl_arr)):
            d_int = int(trade_dates[j])
            d_str = f"{d_int // 10000}-{(d_int % 10000) // 100:02d}-{d_int % 100:02d}"
            d_date = pd.Timestamp(d_str).date()
            half = 'H2' if d_date >= h2_start_dt else 'H1'
            trade = {
                'date': d_date, 'ticker': ticker, 'pnl': float(pnl_arr[j]),
                'half': half, 'leg': 'S3_LONG', 'slot': 'S3',
                'direction': 'LONG', 'mode': 'limit_entry',
            }
            s3_trades.append(trade)
            s3_per_tk[ticker].append(trade)

        if len(pnl_arr) > 0:
            print(f"     {ticker}: {len(daily_probs)} scored, {len(pnl_arr)} trades, "
                  f"avg={np.mean(pnl_arr):+.2%}, WR={(pnl_arr > 0).mean():.0%}")
        else:
            print(f"     {ticker}: {len(daily_probs)} scored, {len(indices)} signals, "
                  f"0 trades (limit orders unfilled)")

        _master_cache.pop(ticker, None)
        _release_memory()

    # Free models for this experiment
    del models
    tf.keras.backend.clear_session()
    _release_memory()

s3_per_tk = dict(s3_per_tk)
print(f"\n  S3 LONG TOTAL: {len(s3_trades)} trades across {len(s3_per_tk)} tickers")


# ── Step 4: Spike Transformer LONG ──────────────────────────────────────────
print(f"\n  STEP 4: Spike Transformer LONG -- model inference...")

meta_path = ensure_st_data('meta.npz')
meta = np.load(meta_path, allow_pickle=True)
st_ticker_names = meta['ticker_names']

probe_path = ensure_st_data(f'fold_{ST_FOLD_ID}_val.npz')
probe = np.load(probe_path)
_N_ASSET_F = probe['X_seq'].shape[-1]
_N_REGIME_F = probe['X_regime'].shape[-1]
probe.close()

print(f"  Building Spike Transformer model...")
tf.keras.backend.clear_session()
gc.collect()
st_model = SpikeTransformer(name='spike_transformer')
dummy_seq = tf.zeros((1, ST_TIME_STEPS, _N_ASSET_F))
dummy_reg = tf.zeros((1, _N_REGIME_F))
st_model({'seq_input': dummy_seq, 'regime_input': dummy_reg})
weights_path = ensure_st_model(ST_FOLD_ID)
st_model.load_weights(weights_path)
print(f"  Model loaded: {st_model.count_params():,} params")

path = ensure_st_data(f'fold_{ST_FOLD_ID}_test.npz')
data = np.load(path)
X_seq     = data['X_seq']
X_reg     = data['X_regime']
X_id      = data['X_id'].flatten()
st_date_ints = data['date_ints']
st_minutes   = data['minutes']
data.close()

entry_mask = (st_minutes == ST_ENTRY_MINUTE)
X_seq     = X_seq[entry_mask]
X_reg     = X_reg[entry_mask]
X_id      = X_id[entry_mask]
st_date_ints_filt = st_date_ints[entry_mask]
del st_minutes, st_date_ints; gc.collect()
print(f"  {len(X_id):,} entry-minute samples")

preds = st_model.predict(
    {'seq_input': X_seq.astype(np.float32),
     'regime_input': X_reg.astype(np.float32)},
    batch_size=512, verbose=0)
del X_seq, X_reg; gc.collect()

st_daily_scores = {}
for i in range(len(X_id)):
    tid = int(X_id[i])
    d_int = int(st_date_ints_filt[i])
    if tid >= len(st_ticker_names): continue
    tname = str(st_ticker_names[tid])
    if tname in EXCLUDE_TICKERS: continue
    if tname not in st_daily_scores:
        st_daily_scores[tname] = {h: {} for h in ST_SCORE_HORIZONS}
    for h in ST_SCORE_HORIZONS:
        st_daily_scores[tname][h][d_int] = float(preds[i, h])

del preds, X_id, st_date_ints_filt, st_model
tf.keras.backend.clear_session(); gc.collect()

print(f"\n  Running Spike Transformer trade simulation...")
spike_trades = []
spike_per_tk = defaultdict(list)

for ticker in st_tickers_list:
    cfg = st_configs[ticker]
    df = load_ticker_ohlcv(ticker)
    if df is None:
        continue
    df = heal_data(df.sort_index())
    ohlcv = precompute_ohlcv_arrays_pst(df, TEST_START, TEST_END)
    if ohlcv is None:
        _master_cache.pop(ticker, None)
        continue

    t_scores = st_daily_scores.get(ticker, {})
    h_scores = t_scores.get(cfg['horizon_idx'], {})
    if not h_scores:
        _master_cache.pop(ticker, None)
        continue

    dates_arr = ohlcv['dates']; times_arr = ohlcv['times']
    probs = np.zeros(len(dates_arr), dtype=np.float64)
    for i in range(len(dates_arr)):
        if times_arr[i] == ST_ENTRY_MINUTE:
            probs[i] = h_scores.get(int(dates_arr[i]), 0.0)

    mask = (probs >= cfg['prob_thresh']) & (times_arr == ST_ENTRY_MINUTE)
    indices = np.where(mask)[0].astype(np.int64)
    if len(indices) == 0:
        _master_cache.pop(ticker, None)
        continue

    np.random.seed(42)
    pnl, trade_dates = dispatch_long_trade_sim(indices, ohlcv, cfg)

    for j in range(len(pnl)):
        d_int = int(trade_dates[j])
        d_str = f"{d_int // 10000}-{(d_int % 10000) // 100:02d}-{d_int % 100:02d}"
        d_date = pd.Timestamp(d_str).date()
        half = 'H2' if d_date >= h2_start_dt else 'H1'
        trade = {
            'date': d_date, 'ticker': ticker, 'pnl': float(pnl[j]),
            'half': half, 'leg': 'SPIKE_TRANS', 'slot': 'ST',
            'direction': 'LONG', 'mode': cfg['mode'],
        }
        spike_trades.append(trade)
        spike_per_tk[ticker].append(trade)

    if len(pnl) > 0:
        print(f"    {ticker}: {len(pnl)} trades, avg={np.mean(pnl):+.2%}, "
              f"WR={(pnl > 0).mean():.0%} ({cfg['mode']}, H={cfg['horizon_label']})")

    _master_cache.pop(ticker, None)
    _release_memory()

spike_per_tk = dict(spike_per_tk)
print(f"\n  SPIKE_TRANS TOTAL: {len(spike_trades)} trades across {len(spike_per_tk)} tickers")
del st_daily_scores; _release_memory()


# ==============================================================================
# STEP 5: BUILD STRATEGIES
# ==============================================================================
print(f"\n  STEP 5: Building 3 strategies...")

# COMBINED = S3 + ST
combined_trades = s3_trades + spike_trades
combined_per_tk = defaultdict(list)
for tk, tl in s3_per_tk.items():
    combined_per_tk[tk].extend(tl)
for tk, tl in spike_per_tk.items():
    combined_per_tk[tk].extend(tl)
combined_per_tk = dict(combined_per_tk)

port_s3    = summarize_trades(s3_trades, 'S3_LONG')
port_spike = summarize_trades(spike_trades, 'SPIKE_TRANS')
port_combo = summarize_trades(combined_trades, 'COMBINED')

strategies = {
    'S3_LONG':     {'summary': port_s3,    'per_ticker': s3_per_tk,      'trades': s3_trades},
    'SPIKE_TRANS': {'summary': port_spike, 'per_ticker': spike_per_tk,   'trades': spike_trades},
    'COMBINED':    {'summary': port_combo, 'per_ticker': combined_per_tk, 'trades': combined_trades},
}

elapsed = time.time() - t_total
print(f"\n  Total scoring time: {elapsed:.0f}s ({elapsed/60:.1f} min)")


# ==============================================================================
# TABLE 1: HEAD-TO-HEAD
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 1: HEAD-TO-HEAD COMPARISON ({TEST_START} -> {TEST_END})")
print(f"{'='*W}")

print(f"\n  {'Strategy':<14} {'#Tk':>4} {'Trades':>6} {'Avg PnL':>8} {'Total':>9} "
      f"{'WR':>5} {'MaxDD':>7} {'Sharpe':>7}")
print(f"  {'─'*65}")

for sname in STRAT_ORDER:
    p = strategies[sname]['summary']
    if p['n'] == 0:
        print(f"  {sname:<14} -- no trades")
        continue
    print(f"  {sname:<14} {p['n_tickers']:>4} {p['n']:>6} {p['avg']:>+7.2%} "
          f"{p['total']:>+8.1%} {p['wr']:>5.0%} {p['dd']:>+7.2%} {p['sharpe']:>6.1f}")


# ==============================================================================
# TABLE 2: MONTH-BY-MONTH DEEP DIVE
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 2: MONTH-BY-MONTH DEEP DIVE")
print(f"{'='*W}")

all_months = sorted(set(m for sname in STRAT_ORDER
                        for m in strategies[sname]['summary'].get('monthly', {}).keys()))

for sname in STRAT_ORDER:
    p = strategies[sname]['summary']
    if p['n'] == 0:
        continue
    tdf = p['trades_df']

    print(f"\n  {sname}:")
    print(f"    {'Month':<10} {'Trades':>6} {'Avg':>7} {'Total':>8} {'WR':>5} "
          f"{'Sharpe':>7} {'Best':>7} {'Worst':>7} {'#Tk':>4} {'Tr/Day':>6}")
    print(f"    {'─'*75}")

    for m in all_months:
        m_trades = tdf[tdf['month'] == m]
        if len(m_trades) == 0:
            print(f"    {m:<10}     --")
            continue
        avg_pnl = m_trades['pnl'].mean()
        total_pnl = m_trades['pnl'].sum()
        wr = (m_trades['pnl'] > 0).mean()
        best = m_trades['pnl'].max()
        worst = m_trades['pnl'].min()
        n_tickers = m_trades['ticker'].nunique()
        daily = m_trades.groupby('date')['pnl'].sum()
        sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0
        n_trading_days = m_trades['date'].nunique()
        trades_per_day = len(m_trades) / max(1, n_trading_days)
        print(f"    {m:<10} {len(m_trades):>6} {avg_pnl:>+6.2%} {total_pnl:>+7.1%} "
              f"{wr:>4.0%} {sharpe:>6.1f} {best:>+6.2%} {worst:>+6.2%} "
              f"{n_tickers:>4} {trades_per_day:>5.1f}")

# Monthly comparison table (side by side)
print(f"\n  MONTHLY COMPARISON (Avg PnL per trade):")
print(f"    {'Month':<10}", end="")
for sname in STRAT_ORDER:
    print(f"  {sname:>14}", end="")
print()
print(f"    {'─'*10}", end="")
for _ in STRAT_ORDER:
    print(f"  {'─'*14}", end="")
print()

for m in all_months:
    print(f"    {m:<10}", end="")
    for sname in STRAT_ORDER:
        monthly = strategies[sname]['summary'].get('monthly', {})
        if m in monthly:
            d = monthly[m]
            print(f"  {d['avg']:>+6.2%} ({d['n']:>3})", end="")
        else:
            print(f"  {'--':>14}", end="")
    print()


# ==============================================================================
# TABLE 3: PER-TICKER BREAKDOWN
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 3: PER-TICKER BREAKDOWN")
print(f"{'='*W}")

all_tickers_union = sorted(set(
    tk for sname in STRAT_ORDER
    for tk in strategies[sname]['per_ticker'].keys()
))

col_w = 22
print(f"\n  {'Ticker':<7}", end="")
for sname in STRAT_ORDER:
    print(f"  {sname:<{col_w}}", end="")
print()
print(f"  {'─'*7}", end="")
for _ in STRAT_ORDER:
    print(f"  {'─'*col_w}", end="")
print()

for ticker in all_tickers_union:
    print(f"  {ticker:<7}", end="")
    for sname in STRAT_ORDER:
        per_tk = strategies[sname]['per_ticker']
        trades = per_tk.get(ticker, [])
        if trades:
            avg = np.mean([t['pnl'] for t in trades])
            n = len(trades)
            wr = sum(1 for t in trades if t['pnl'] > 0) / n * 100
            cell = f"{avg:>+5.2%} {n:>3}t {wr:>3.0f}%w"
            print(f"  {cell:<{col_w}}", end="")
        else:
            print(f"  {'--':<{col_w}}", end="")
    print()


# ==============================================================================
# TABLE 4: CAPITAL-ADJUSTED PnL (deployed only)
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 4: CAPITAL-ADJUSTED PnL (deployed only, floor={DEPLOY_FLOOR:.1%})")
print(f"{'='*W}")
print(f"\n  {'Strategy':<14} {'Days':>5} {'Avg/Day':>8} {'Total':>8} {'WR':>5} "
      f"{'MaxDD':>7} {'Pos/Day':>7}  {'H2 Avg':>7} {'H2 Days':>7}")
print(f"  {'─'*85}")

for sname in STRAT_ORDER:
    per_tk = strategies[sname]['per_ticker']
    deploy_tickers = [tk for tk, trades in per_tk.items()
                      if trades and np.mean([t['pnl'] for t in trades]) >= DEPLOY_FLOOR]
    if not deploy_tickers:
        print(f"  {sname:<14} (no tickers above threshold)")
        continue
    deploy_trades = []
    for tk in deploy_tickers:
        deploy_trades.extend(per_tk[tk])
    ca = compute_capital_adjusted(deploy_trades)
    print(f"  {sname:<14} {ca['n_days']:>5} {ca['daily_avg']:>+7.2%} "
          f"{ca['daily_total']:>+7.1%} {ca['daily_wr']:>4.0%} "
          f"{ca['daily_dd']:>+7.2%} {ca['avg_positions']:>6.1f}  "
          f"{ca['h2_daily_avg']:>+7.2%} {ca['h2_n_days']:>7}")
    print(f"  {'':14} Deployed ({len(deploy_tickers)}): {', '.join(sorted(deploy_tickers))}")


# ==============================================================================
# TABLE 5: OVERLAP & COMPLEMENTARITY ANALYSIS
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 5: OVERLAP & COMPLEMENTARITY ANALYSIS")
print(f"{'='*W}")

st_set = set(spike_per_tk.keys())
s3_set = set(s3_per_tk.keys())
overlap = st_set & s3_set
only_st = st_set - s3_set
only_s3 = s3_set - st_set

print(f"\n  TICKER OVERLAP:")
print(f"    ST only:     {len(only_st)}  {', '.join(sorted(only_st)) or '—'}")
print(f"    S3 only:     {len(only_s3)}  {', '.join(sorted(only_s3)) or '—'}")
print(f"    Both:        {len(overlap)}  {', '.join(sorted(overlap)) or '—'}")
print(f"    Union:       {len(st_set | s3_set)} unique tickers")
print(f"    Jaccard:     {len(overlap)/max(1,len(st_set|s3_set)):.0%}")

# Daily PnL correlation
print(f"\n  DAILY PnL CORRELATION:")
strat_daily = {}
for sname, trades_list in [('S3_LONG', s3_trades), ('SPIKE_TRANS', spike_trades)]:
    if trades_list:
        df_t = pd.DataFrame(trades_list)
        strat_daily[sname] = df_t.groupby('date')['pnl'].sum()

if len(strat_daily) == 2:
    all_dates_corr = sorted(set().union(*(set(v.index) for v in strat_daily.values())))
    corr_df = pd.DataFrame({k: v.reindex(all_dates_corr, fill_value=0) for k, v in strat_daily.items()})
    corr_val = corr_df['S3_LONG'].corr(corr_df['SPIKE_TRANS'])
    print(f"    Correlation: {corr_val:.3f}")
    common_dates = set(strat_daily['S3_LONG'].index) & set(strat_daily['SPIKE_TRANS'].index)
    print(f"    Overlapping trading days: {len(common_dates)}")
    if common_dates:
        s3_on_common = strat_daily['S3_LONG'].loc[strat_daily['S3_LONG'].index.isin(common_dates)]
        st_on_common = strat_daily['SPIKE_TRANS'].loc[strat_daily['SPIKE_TRANS'].index.isin(common_dates)]
        print(f"    On overlapping days: S3 avg={s3_on_common.mean():+.2%}, ST avg={st_on_common.mean():+.2%}")

# Same-day trades on overlapping tickers
if overlap and s3_trades and spike_trades:
    s3_df_temp = pd.DataFrame(s3_trades)
    st_df_temp = pd.DataFrame(spike_trades)
    s3_dates = set(zip(s3_df_temp['ticker'], s3_df_temp['date']))
    st_dates = set(zip(st_df_temp['ticker'], st_df_temp['date']))
    same_day = s3_dates & st_dates
    print(f"\n  SAME-DAY SAME-TICKER TRADES:")
    print(f"    {len(same_day)} instances")
    if same_day:
        for tk, dt in sorted(same_day)[:10]:
            s3_pnl = [t['pnl'] for t in s3_trades if t['ticker'] == tk and t['date'] == dt]
            st_pnl = [t['pnl'] for t in spike_trades if t['ticker'] == tk and t['date'] == dt]
            print(f"      {tk} {dt}: S3={sum(s3_pnl):+.2%}, ST={sum(st_pnl):+.2%}, "
                  f"NET={sum(s3_pnl)+sum(st_pnl):+.2%}")

# Monthly overlap analysis
print(f"\n  MONTHLY TRADING COVERAGE:")
print(f"    {'Month':<10} {'S3 days':>8} {'ST days':>8} {'Both':>6} {'Combined':>10}")
print(f"    {'─'*50}")
for m in all_months:
    s3_days_m = set()
    st_days_m = set()
    if s3_trades:
        for t in s3_trades:
            if pd.Timestamp(t['date']).strftime('%Y-%m') == m:
                s3_days_m.add(t['date'])
    if spike_trades:
        for t in spike_trades:
            if pd.Timestamp(t['date']).strftime('%Y-%m') == m:
                st_days_m.add(t['date'])
    both_m = s3_days_m & st_days_m
    combined_m = s3_days_m | st_days_m
    print(f"    {m:<10} {len(s3_days_m):>8} {len(st_days_m):>8} {len(both_m):>6} {len(combined_m):>10}")


# ==============================================================================
# TABLE 6: OVERLAPPING PERIOD COMPARISON (Jun 1 - Jul 3 or actual overlap)
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 6: OVERLAPPING PERIOD ANALYSIS")
print(f"{'='*W}")

# Find the actual overlapping date range
s3_dates_all = set(t['date'] for t in s3_trades) if s3_trades else set()
st_dates_all = set(t['date'] for t in spike_trades) if spike_trades else set()

if s3_dates_all and st_dates_all:
    overlap_start = max(min(s3_dates_all), min(st_dates_all))
    overlap_end = min(max(s3_dates_all), max(st_dates_all))
    print(f"\n  Overlapping period: {overlap_start} to {overlap_end}")

    s3_overlap = [t for t in s3_trades if overlap_start <= t['date'] <= overlap_end]
    st_overlap = [t for t in spike_trades if overlap_start <= t['date'] <= overlap_end]
    combo_overlap = s3_overlap + st_overlap

    p_s3_o = summarize_trades(s3_overlap, 'S3 (overlap)')
    p_st_o = summarize_trades(st_overlap, 'ST (overlap)')
    p_co_o = summarize_trades(combo_overlap, 'COMBINED (overlap)')

    print(f"\n  {'Strategy':<20} {'Trades':>6} {'Avg':>7} {'Total':>8} {'WR':>5} {'Sharpe':>7}")
    print(f"  {'─'*60}")
    for p in [p_s3_o, p_st_o, p_co_o]:
        if p['n'] > 0:
            print(f"  {p['label']:<20} {p['n']:>6} {p['avg']:>+6.2%} {p['total']:>+7.1%} "
                  f"{p['wr']:>4.0%} {p['sharpe']:>6.1f}")


# ==============================================================================
# 15. CHARTS
# ==============================================================================
print(f"\n  Generating charts...")

fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# Chart [0,0]: Equity curves (3 strategies)
ax1 = axes[0, 0]
for sname in STRAT_ORDER:
    p = strategies[sname]['summary']
    if p['n'] == 0:
        continue
    df_t = p['trades_df'].sort_values('date')
    cum = df_t['pnl'].cumsum() * 100
    lw = 2.5 if sname == 'COMBINED' else 1.8
    ls = '--' if sname == 'COMBINED' else '-'
    ax1.plot(pd.to_datetime(df_t['date'].values), cum.values,
             color=STRATEGY_COLORS[sname], linewidth=lw, linestyle=ls,
             label=f"{sname} ({p['n']}tr, {p['avg']:+.2%})", alpha=0.9)
ax1.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
ax1.axvline(x=pd.Timestamp(HALF2_START), color='red', linewidth=1,
            linestyle=':', alpha=0.5, label='H2 start')
ax1.set_ylabel('Cumulative PnL (%)')
ax1.set_title('Equity Curves -- ST vs S3 vs COMBINED', fontsize=13, fontweight='bold')
ax1.legend(fontsize=9, loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

# Chart [0,1]: Monthly avg PnL bars (S3 vs ST vs COMBINED)
ax2 = axes[0, 1]
if all_months:
    x = np.arange(len(all_months))
    bar_width = 0.25
    for i, sname in enumerate(STRAT_ORDER):
        p = strategies[sname]['summary']
        if p['n'] == 0: continue
        monthly = p.get('monthly', {})
        monthly_avgs = []
        for m in all_months:
            ms = monthly.get(m)
            monthly_avgs.append(ms['avg'] * 100 if ms else 0)
        bars = ax2.bar(x + i * bar_width, monthly_avgs, bar_width,
                       label=sname, color=STRATEGY_COLORS[sname], alpha=0.85)
        for bar, val in zip(bars, monthly_avgs):
            if abs(val) > 0.01:
                ax2.text(bar.get_x() + bar.get_width()/2,
                         bar.get_height() + (0.05 if val >= 0 else -0.15),
                         f'{val:.1f}%', ha='center', va='bottom', fontsize=7)
    ax2.set_xticks(x + bar_width)
    month_labels = [pd.Timestamp(m + '-01').strftime('%b') for m in all_months]
    ax2.set_xticklabels(month_labels, fontsize=10)
    ax2.set_ylabel('Avg PnL per Trade (%)')
    ax2.axhline(y=0, color='gray', linewidth=0.5)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
ax2.set_title('Monthly Avg PnL -- S3 vs ST vs COMBINED', fontsize=13, fontweight='bold')

# Chart [1,0]: Per-ticker heatmap (S3 vs ST side by side)
ax3 = axes[1, 0]
if all_tickers_union:
    heatmap_data = np.full((len(all_tickers_union), 2), np.nan)
    for j, (sname, per_tk) in enumerate([('S3_LONG', s3_per_tk), ('SPIKE_TRANS', spike_per_tk)]):
        for i, ticker in enumerate(all_tickers_union):
            trades = per_tk.get(ticker, [])
            if trades:
                heatmap_data[i, j] = np.mean([t['pnl'] for t in trades]) * 100
    vmax = max(abs(np.nanmin(heatmap_data)), abs(np.nanmax(heatmap_data)))
    if vmax == 0 or np.isnan(vmax):
        vmax = 1
    im = ax3.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=-vmax, vmax=vmax)
    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(['S3 LONG', 'Spike Trans'], fontsize=10)
    ax3.set_yticks(range(len(all_tickers_union)))
    ax3.set_yticklabels(all_tickers_union, fontsize=7)
    for i in range(len(all_tickers_union)):
        for j in range(2):
            val = heatmap_data[i, j]
            if not np.isnan(val):
                ax3.text(j, i, f'{val:+.1f}%', ha='center', va='center',
                         fontsize=7, fontweight='bold',
                         color='white' if abs(val) > vmax*0.5 else 'black')
    fig.colorbar(im, ax=ax3, label='Avg PnL (%)', shrink=0.8)
ax3.set_title('Per-Ticker Heatmap -- S3 vs ST', fontsize=13, fontweight='bold')

# Chart [1,1]: Daily PnL scatter (S3 vs ST)
ax4 = axes[1, 1]
if len(strat_daily) == 2:
    common = strat_daily['S3_LONG'].index.intersection(strat_daily['SPIKE_TRANS'].index)
    if len(common) >= 2:
        x_vals = strat_daily['S3_LONG'].loc[common].values * 100
        y_vals = strat_daily['SPIKE_TRANS'].loc[common].values * 100
        ax4.scatter(x_vals, y_vals, alpha=0.6, s=40, c='steelblue')
        ax4.axhline(y=0, color='gray', linewidth=0.5)
        ax4.axvline(x=0, color='gray', linewidth=0.5)
        corr = np.corrcoef(x_vals, y_vals)[0, 1]
        ax4.set_title(f'Daily PnL Correlation (r={corr:.2f})', fontsize=13, fontweight='bold')
        ax4.set_xlabel('S3 LONG daily PnL (%)')
        ax4.set_ylabel('Spike Transformer daily PnL (%)')
    ax4.grid(True, alpha=0.3)
else:
    ax4.text(0.5, 0.5, 'Insufficient data for correlation',
             ha='center', va='center', transform=ax4.transAxes)
    ax4.set_title('Daily PnL Correlation', fontsize=13, fontweight='bold')

fig.suptitle(f'ST vs S3 LONG-Only Backtest (Live Scoring)\n'
             f'{TEST_START} -> {TEST_END}  |  Deploy: {DEPLOY_FLOOR:.1%}  |  '
             f'Slippage: {LONG_SLIPPAGE_RT/2:.2%} one-way',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('/tmp/backtest_st_vs_s3_live.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"\n  Chart 1 saved: /tmp/backtest_st_vs_s3_live.png")


# ── Chart 2: Monthly Equity Panels ──
if all_months:
    n_months = len(all_months)
    fig2, axes2 = plt.subplots(1, n_months, figsize=(5 * n_months, 5), sharey=True)
    if n_months == 1:
        axes2 = [axes2]

    for mi, m in enumerate(all_months):
        ax = axes2[mi]
        for sname in ['S3_LONG', 'SPIKE_TRANS', 'COMBINED']:
            p = strategies[sname]['summary']
            if p['n'] == 0: continue
            m_trades = p['trades_df'][p['trades_df']['month'] == m].sort_values('date')
            if len(m_trades) == 0: continue
            cum = m_trades['pnl'].cumsum() * 100
            lw = 2.5 if sname == 'COMBINED' else 1.5
            ls = '--' if sname == 'COMBINED' else '-'
            ax.plot(range(len(cum)), cum.values,
                    color=STRATEGY_COLORS[sname], linewidth=lw, linestyle=ls,
                    label=sname if mi == 0 else '')
        month_label = pd.Timestamp(m + '-01').strftime('%B %Y')
        ax.set_title(month_label, fontsize=11, fontweight='bold')
        ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
        ax.set_xlabel('Trade #')
        if mi == 0:
            ax.set_ylabel('Cumulative PnL (%)')
        ax.grid(True, alpha=0.3)

    axes2[0].legend(fontsize=8, loc='upper left')
    fig2.suptitle('Monthly Equity Curves', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/tmp/backtest_st_vs_s3_monthly.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Chart 2 saved: /tmp/backtest_st_vs_s3_monthly.png")


# ── Chart 3: Portfolio Equity Simulation ──
STARTING_CAPITAL = 200_000
combo_equity = {}
for sname in STRAT_ORDER:
    s = strategies[sname]['summary']
    if s['n'] == 0: continue
    df_t = s['trades_df'].sort_values('date')
    equity = float(STARTING_CAPITAL)
    curve = [{'date': pd.Timestamp(TEST_START).date(), 'equity': equity}]
    peak = equity; max_dd_pct = 0.0
    for _, row in df_t.iterrows():
        alloc = equity / max(1, s['n_tickers'])
        equity += alloc * row['pnl']
        curve.append({'date': row['date'], 'equity': equity})
        peak = max(peak, equity)
        dd = (equity - peak) / peak
        if dd < max_dd_pct: max_dd_pct = dd
    combo_equity[sname] = pd.DataFrame(curve)

if combo_equity:
    fig3, ax_c = plt.subplots(figsize=(16, 7))
    for sname in STRAT_ORDER:
        if sname not in combo_equity: continue
        cdf = combo_equity[sname]
        dates_c = pd.to_datetime(cdf['date'])
        eq = cdf['equity'] / 1000
        is_main = sname == 'COMBINED'
        ax_c.plot(dates_c, eq, label=f"{sname} ({strategies[sname]['summary']['n']}tr, "
                  f"{strategies[sname]['summary']['avg']:+.2%})",
                  color=STRATEGY_COLORS.get(sname, 'gray'),
                  linewidth=2.5 if is_main else 1.5,
                  linestyle='--' if is_main else '-')
    ax_c.axhline(y=STARTING_CAPITAL/1000, color='gray', linestyle='--', alpha=0.4)
    ax_c.axvline(x=pd.Timestamp(HALF2_START), color='red', linewidth=1,
                 linestyle=':', alpha=0.5, label='H2 start')
    ax_c.set_ylabel('Equity ($K)', fontsize=12)
    ax_c.set_title(f'LONG Portfolio: Spike Transformer + S3\n'
                   f'{TEST_START} -> {TEST_END}', fontsize=14, fontweight='bold')
    ax_c.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax_c.grid(True, alpha=0.3)
    ax_c.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.tight_layout()
    plt.savefig('/tmp/backtest_st_vs_s3_equity.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Chart 3 saved: /tmp/backtest_st_vs_s3_equity.png")


# ==============================================================================
# VERDICT
# ==============================================================================
print(f"\n{'='*W}")
print(f"  VERDICT")
print(f"{'='*W}")

for sname in STRAT_ORDER:
    p = strategies[sname]['summary']
    per_tk = strategies[sname]['per_ticker']
    deploy = [tk for tk, trades in per_tk.items()
              if trades and np.mean([t['pnl'] for t in trades]) >= DEPLOY_FLOOR]
    deploy_trades = []
    for tk in deploy:
        deploy_trades.extend(per_tk[tk])
    ca = compute_capital_adjusted(deploy_trades) if deploy_trades else None

    print(f"\n  {sname}:")
    print(f"    All:      {p['n']:>4} trades ({p['n_tickers']} tickers), avg={p['avg']:>+.2%}, "
          f"total={p['total']:>+.1%}, Sharpe={p['sharpe']:.1f}")
    print(f"    H1:       {p['h1_n']:>4} trades, avg={p['h1_avg']:>+.2%}")
    print(f"    H2:       {p['h2_n']:>4} trades, avg={p['h2_avg']:>+.2%}, WR={p['h2_wr']:.0%}")
    if deploy:
        dp = summarize_trades(deploy_trades, '')
        print(f"    Deployed: {dp['n']:>4} trades ({len(deploy)} tickers), "
              f"avg={dp['avg']:>+.2%}, total={dp['total']:>+.1%}")
        if ca and ca['n_days'] > 0:
            print(f"    Cap-adj:  {ca['daily_avg']:>+.2%}/day, {ca['n_days']} days, "
                  f"WR={ca['daily_wr']:.0%}, DD={ca['daily_dd']:>+.2%}, "
                  f"H2={ca['h2_daily_avg']:>+.2%}/day")
        print(f"    Tickers:  {', '.join(sorted(deploy))}")
    else:
        print(f"    Deployed: none above {DEPLOY_FLOOR:.1%}")

# Portfolio simulation summary
print(f"\n  PORTFOLIO SIMULATION (${STARTING_CAPITAL:,} start):")
for sname in STRAT_ORDER:
    if sname in combo_equity:
        final = combo_equity[sname]['equity'].iloc[-1]
        total_ret = (final - STARTING_CAPITAL) / STARTING_CAPITAL
        print(f"    {sname:<14} -> ${final:,.0f} ({total_ret:>+.1%})")

# Best strategy summary
best_avg = max(STRAT_ORDER, key=lambda s: strategies[s]['summary']['avg'] if strategies[s]['summary']['n'] > 0 else -999)
best_sharpe = max(STRAT_ORDER, key=lambda s: strategies[s]['summary']['sharpe'] if strategies[s]['summary']['n'] > 0 else -999)
print(f"\n  BEST BY AVG PnL:  {best_avg} ({strategies[best_avg]['summary']['avg']:+.2%})")
print(f"  BEST BY SHARPE:   {best_sharpe} ({strategies[best_sharpe]['summary']['sharpe']:.1f})")

print(f"\n  Deploy threshold:  {DEPLOY_FLOOR:.1%}")
print(f"  Slippage LONG:     {LONG_SLIPPAGE_RT/2:.2%} one-way ({LONG_SLIPPAGE_RT:.3%} RT)")
print(f"  Total runtime:     {time.time()-t_total:.0f}s")
print(f"{'='*W}")
