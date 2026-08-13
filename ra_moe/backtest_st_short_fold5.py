# %% ── Spike Transformer SHORT Fold 5 — Portfolio Backtest ────────────────────
#
# Unbiased portfolio backtest on TEST window (Jul 10 - Aug 10, 2026).
# Uses all tickers with >1% avg PnL from the fold 5 optimizer, each with its
# best config. No re-optimization — configs are frozen from the optimizer.
#
# Tickers (7): ALAB, BE, PL, HUT, CIFR, BBAI, MARA
# Score: 6:15 AM (min 375), Entry: 6:25 AM (min 385)
# Model: spike_transformer_short_fold5 (Non-Stationary Transformer + FiLM)
#
# GPU recommended for scoring phase. Colab-ready.

import os, time, gc, ctypes, warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from google.cloud import storage
from numba import jit
import pytz
from datetime import timedelta
from collections import defaultdict

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    from google.colab import auth
    _IN_COLAB = True
    auth.authenticate_user()
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
FOLD_ID        = 5
SCORE_HORIZONS = [0, 1, 2]
HORIZON_LABELS = {0: '15m', 1: '30m', 2: '60m'}

TEST_START     = '2026-07-10'
TEST_END       = '2026-08-10'

SCORE_MINUTE   = 375
ENTRY_MINUTE   = 385
ENTRY_TIME     = '06:25'
EXIT_TIME_MIN  = 780
SLIPPAGE_PCT   = 0.0015

GCS_BUCKET       = 'time-series-bucket-ys'
GCS_DATA_PREFIX  = 'Algos/RA_MoE/short_data'
GCS_MODEL_PREFIX = 'Algos/RA_MoE/short_models'
GCS_INFER_DIR    = 'Data/Inference_v2'

DATA_DIR   = '/tmp/ra_moe_short_data'
MODEL_DIR  = '/tmp/ra_moe_short_models'
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

D_MODEL = 64; N_HEADS = 4; D_FF = 128; N_TRANSFORMER_BLOCKS = 2
DROPOUT = 0.1; FILM_HIDDEN = 64; TIME_STEPS = 128
TARGET_HORIZONS = [15, 30, 60]; N_HORIZONS = 3
ASL_GAMMA_NEG = 6.0; ASL_GAMMA_POS = 0.1; ASL_CLIP = 0.08

N_ASSET_FEATURES  = None
N_REGIME_FEATURES = None

TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}
EXCLUDE_TICKERS = {'SPY', 'VIX', 'TLT', 'IWM', 'XLE', 'ITA', 'BITF', 'MIGI', 'SMLR'}

PST = pytz.timezone('America/Los_Angeles')

# ── Frozen configs from optimizer (>1% TEST PnL) ──
DEPLOY_CONFIGS = {
    'ALAB': {'horizon': 0, 'prob_thresh': 0.20, 'tp_pct': 0.12, 'sl_pct': 0.06,
             'mode': 'static'},
    'BE':   {'horizon': 1, 'prob_thresh': 0.40, 'tp_pct': 0.12, 'sl_pct': 0.06,
             'mode': 'static'},
    'PL':   {'horizon': 0, 'prob_thresh': 0.20, 'tp_pct': 0.08, 'sl_pct': 0.04,
             'mode': 'limit_entry', 'dip_pct': 0.012, 'max_wait': 30},
    'HUT':  {'horizon': 2, 'prob_thresh': 0.40, 'tp_pct': 0.08, 'sl_pct': 0.04,
             'mode': 'static'},
    'CIFR': {'horizon': 2, 'prob_thresh': 0.40, 'tp_pct': 0.16, 'sl_pct': 0.08,
             'mode': 'grace', 'grace_minutes': 15},
    'BBAI': {'horizon': 0, 'prob_thresh': 0.30, 'tp_pct': 0.16, 'sl_pct': 0.08,
             'mode': 'static'},
    'MARA': {'horizon': 0, 'prob_thresh': 0.40, 'tp_pct': 0.08, 'sl_pct': 0.04,
             'mode': 'static'},
}

DEPLOY_TICKERS = sorted(DEPLOY_CONFIGS.keys())

SEED = 42
np.random.seed(SEED)
W = 100


# ==============================================================================
# GCS HELPERS
# ==============================================================================
def _gcs_bucket():
    return storage.Client().bucket(GCS_BUCKET)

def download_from_gcs(gcs_blob_path, local_path):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    _gcs_bucket().blob(gcs_blob_path).download_to_filename(local_path)

def ensure_data(filename):
    local_path = os.path.join(DATA_DIR, filename)
    if os.path.exists(local_path):
        return local_path
    gcs_path = f'{GCS_DATA_PREFIX}/{filename}'
    print(f"    Downloading {filename} from GCS...")
    download_from_gcs(gcs_path, local_path)
    return local_path

def ensure_model(fold_id):
    local_dir = os.path.join(MODEL_DIR, f'fold_{fold_id}')
    weights_path = os.path.join(local_dir, 'model.weights.h5')
    if os.path.exists(weights_path):
        return weights_path
    os.makedirs(local_dir, exist_ok=True)
    gcs_prefix = f'{GCS_MODEL_PREFIX}/fold_{fold_id}'
    print(f"    Downloading fold {fold_id} model from GCS...")
    bucket = _gcs_bucket()
    blobs = list(bucket.list_blobs(prefix=gcs_prefix))
    for blob in blobs:
        fname = blob.name.split('/')[-1]
        blob.download_to_filename(os.path.join(local_dir, fname))
    return weights_path


# ==============================================================================
# CUSTOM LAYERS (must match training exactly)
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
        self.d_model   = d_model
        self.n_heads   = n_heads
        self.d_head    = d_model // n_heads
        self.scale     = tf.sqrt(tf.cast(self.d_head, tf.float32))
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        self.wo = layers.Dense(d_model)
        self.attn_drop = layers.Dropout(dropout)
        self.tau_proj   = layers.Dense(n_heads)
        self.sigma_proj = layers.Dense(d_model)

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
        out  = tf.matmul(attn, V)
        out  = tf.reshape(tf.transpose(out, (0, 2, 1, 3)), (B, T, self.d_model))
        sigma = tf.nn.softplus(self.sigma_proj(stats))
        out   = out * sigma[:, tf.newaxis, :]
        return self.wo(out)


class TransformerBlock(layers.Layer):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.attn  = DeStationaryAttention(d_model, n_heads, dropout)
        self.ffn   = tf.keras.Sequential([
            layers.Dense(d_ff, activation='gelu'),
            layers.Dense(d_model),
            layers.Dropout(dropout)])
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.drop  = layers.Dropout(dropout)

    def call(self, x, stats, training=False):
        x = x + self.drop(
            self.attn(self.norm1(x), stats, training=training), training=training)
        x = x + self.ffn(self.norm2(x), training=training)
        return x


class TransformerBackbone(layers.Layer):
    def __init__(self, d_model, n_heads, d_ff, n_blocks, seq_len,
                 dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.stationarize = SeriesStationarization()
        self.input_proj   = layers.Dense(d_model)
        self.pos_embed    = self.add_weight(
            name='pos_embed', shape=(1, seq_len, d_model),
            initializer='glorot_normal', trainable=True)
        self.input_drop   = layers.Dropout(dropout)
        self.blocks       = [TransformerBlock(d_model, n_heads, d_ff, dropout,
                                              name=f'block_{i}')
                             for i in range(n_blocks)]
        self.final_norm   = layers.LayerNormalization()

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
            layers.Dense(hidden_dim, activation='relu'),
            layers.Dense(hidden_dim, activation='relu'),
        ])
        self.gamma_proj = layers.Dense(d_model)
        self.beta_proj  = layers.Dense(d_model)

    def call(self, x, regime_features, training=False):
        h = self.mlp(regime_features, training=training)
        gamma = 1.0 + self.gamma_proj(h)
        beta  = self.beta_proj(h)
        return gamma * x + beta


class SpikeTransformer(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.backbone  = TransformerBackbone(
            D_MODEL, N_HEADS, D_FF, N_TRANSFORMER_BLOCKS,
            TIME_STEPS, DROPOUT, name='backbone')
        self.attn_pool = AttentionPooling(D_MODEL, name='attn_pool')
        self.film      = FiLMConditioner(D_MODEL, FILM_HIDDEN, name='film')
        self.drop      = layers.Dropout(DROPOUT)
        self.head      = layers.Dense(N_HORIZONS, activation='sigmoid',
                                       name='spike_head')

    def call(self, inputs, training=False):
        seq    = inputs['seq_input']
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
            loss = asymmetric_focal_loss(y, preds, sample_weight)
        self.optimizer.apply_gradients(
            zip(tape.gradient(loss, self.trainable_variables),
                self.trainable_variables))
        return {'loss': loss}

    def test_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        preds = self(x, training=False)
        loss = asymmetric_focal_loss(y, preds, sample_weight)
        return {'loss': loss}


def asymmetric_focal_loss(y_true, y_pred, sample_weight=None):
    p = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    pos_loss = -y_true * tf.pow(1.0 - p, ASL_GAMMA_POS) * tf.math.log(p)
    p_m = tf.maximum(p - ASL_CLIP, 0.0)
    neg_loss = -(1.0 - y_true) * tf.pow(p_m, ASL_GAMMA_NEG) * tf.math.log(
        tf.clip_by_value(1.0 - p_m, 1e-7, 1.0))
    per_sample = tf.reduce_mean(pos_loss + neg_loss, axis=-1)
    if sample_weight is not None:
        per_sample = per_sample * sample_weight
    return tf.reduce_mean(per_sample)


# ==============================================================================
# SHORT SIMULATION ENGINES (Numba)
# ==============================================================================

@jit(nopython=True)
def run_short_trade_sim(entry_indices, prices, highs, lows, date_ints, time_mins,
                        tp, sl, exit_time):
    n_entries = len(entry_indices)
    n_prices  = len(prices)
    pnl_arr     = np.empty(n_entries, dtype=np.float64)
    date_arr    = np.empty(n_entries, dtype=np.int64)
    entry_arr   = np.empty(n_entries, dtype=np.float64)
    exit_arr    = np.empty(n_entries, dtype=np.float64)
    outcome_arr = np.empty(n_entries, dtype=np.int64)  # 1=TP, -1=SL, 0=EOD

    trade_count       = 0
    current_entry_ptr = 0
    last_traded_date  = -1

    while current_entry_ptr < n_entries:
        idx = entry_indices[current_entry_ptr]
        if idx >= n_prices - 1: break

        entry_date = date_ints[idx]
        if entry_date == last_traded_date:
            current_entry_ptr += 1
            continue

        entry_price  = prices[idx]
        target_price = entry_price * (1 - tp)
        stop_price   = entry_price * (1 + sl)

        outcome    = 0.0
        exit_price = entry_price
        exit_type  = 0
        trade_done = False
        exit_idx   = idx

        for k in range(idx + 1, n_prices):
            if date_ints[k] != entry_date:
                exit_price = prices[k-1]
                outcome = (entry_price - exit_price) / entry_price
                exit_idx = k; trade_done = True; break
            tp_hit = lows[k] <= target_price
            sl_hit = highs[k] >= stop_price
            if sl_hit and tp_hit:
                if np.random.random() < 0.5:
                    outcome = -sl; exit_price = stop_price; exit_type = -1
                else:
                    outcome = tp; exit_price = target_price; exit_type = 1
                exit_idx = k; trade_done = True; break
            if sl_hit:
                outcome = -sl; exit_price = stop_price; exit_type = -1
                exit_idx = k; trade_done = True; break
            if tp_hit:
                outcome = tp; exit_price = target_price; exit_type = 1
                exit_idx = k; trade_done = True; break
            if time_mins[k] >= exit_time:
                exit_price = prices[k]
                outcome = (entry_price - exit_price) / entry_price
                exit_idx = k; trade_done = True; break

        if not trade_done:
            exit_price = prices[n_prices-1]
            outcome  = (entry_price - exit_price) / entry_price
            exit_idx = n_prices

        pnl_arr[trade_count]     = outcome
        date_arr[trade_count]    = entry_date
        entry_arr[trade_count]   = entry_price
        exit_arr[trade_count]    = exit_price
        outcome_arr[trade_count] = exit_type
        trade_count += 1
        last_traded_date = entry_date

        next_ptr = current_entry_ptr + 1
        while next_ptr < n_entries:
            if entry_indices[next_ptr] > exit_idx: break
            next_ptr += 1
        current_entry_ptr = next_ptr

    return (pnl_arr[:trade_count], date_arr[:trade_count],
            entry_arr[:trade_count], exit_arr[:trade_count],
            outcome_arr[:trade_count])


@jit(nopython=True)
def run_short_trade_sim_grace(entry_indices, prices, highs, lows, date_ints, time_mins,
                              tp, sl, exit_time, grace_minutes):
    n_entries = len(entry_indices)
    n_prices  = len(prices)
    pnl_arr     = np.empty(n_entries, dtype=np.float64)
    date_arr    = np.empty(n_entries, dtype=np.int64)
    entry_arr   = np.empty(n_entries, dtype=np.float64)
    exit_arr    = np.empty(n_entries, dtype=np.float64)
    outcome_arr = np.empty(n_entries, dtype=np.int64)

    trade_count       = 0
    current_entry_ptr = 0
    last_traded_date  = -1

    while current_entry_ptr < n_entries:
        idx = entry_indices[current_entry_ptr]
        if idx >= n_prices - 1: break

        entry_date = date_ints[idx]
        if entry_date == last_traded_date:
            current_entry_ptr += 1
            continue

        entry_price  = prices[idx]
        target_price = entry_price * (1 - tp)
        stop_price   = entry_price * (1 + sl)
        entry_time   = time_mins[idx]

        outcome    = 0.0
        exit_price = entry_price
        exit_type  = 0
        trade_done = False
        exit_idx   = idx

        for k in range(idx + 1, n_prices):
            if date_ints[k] != entry_date:
                exit_price = prices[k-1]
                outcome = (entry_price - exit_price) / entry_price
                exit_idx = k; trade_done = True; break
            minutes_elapsed = time_mins[k] - entry_time
            sl_active = minutes_elapsed >= grace_minutes
            tp_hit = lows[k] <= target_price
            sl_hit = sl_active and (highs[k] >= stop_price)
            if sl_hit and tp_hit:
                if np.random.random() < 0.5:
                    outcome = -sl; exit_price = stop_price; exit_type = -1
                else:
                    outcome = tp; exit_price = target_price; exit_type = 1
                exit_idx = k; trade_done = True; break
            if sl_hit:
                outcome = -sl; exit_price = stop_price; exit_type = -1
                exit_idx = k; trade_done = True; break
            if tp_hit:
                outcome = tp; exit_price = target_price; exit_type = 1
                exit_idx = k; trade_done = True; break
            if time_mins[k] >= exit_time:
                exit_price = prices[k]
                outcome = (entry_price - exit_price) / entry_price
                exit_idx = k; trade_done = True; break

        if not trade_done:
            exit_price = prices[n_prices-1]
            outcome  = (entry_price - exit_price) / entry_price
            exit_idx = n_prices

        pnl_arr[trade_count]     = outcome
        date_arr[trade_count]    = entry_date
        entry_arr[trade_count]   = entry_price
        exit_arr[trade_count]    = exit_price
        outcome_arr[trade_count] = exit_type
        trade_count += 1
        last_traded_date = entry_date

        next_ptr = current_entry_ptr + 1
        while next_ptr < n_entries:
            if entry_indices[next_ptr] > exit_idx: break
            next_ptr += 1
        current_entry_ptr = next_ptr

    return (pnl_arr[:trade_count], date_arr[:trade_count],
            entry_arr[:trade_count], exit_arr[:trade_count],
            outcome_arr[:trade_count])


@jit(nopython=True)
def run_short_trade_sim_limit_entry(entry_indices, prices, highs, lows, date_ints, time_mins,
                                    tp, sl, exit_time, rally_pct, max_wait_minutes):
    n_entries = len(entry_indices)
    n_prices  = len(prices)
    pnl_arr     = np.empty(n_entries, dtype=np.float64)
    date_arr    = np.empty(n_entries, dtype=np.int64)
    entry_arr   = np.empty(n_entries, dtype=np.float64)
    exit_arr    = np.empty(n_entries, dtype=np.float64)
    outcome_arr = np.empty(n_entries, dtype=np.int64)

    trade_count       = 0
    current_entry_ptr = 0
    last_traded_date  = -1

    while current_entry_ptr < n_entries:
        idx = entry_indices[current_entry_ptr]
        if idx >= n_prices - 1: break

        entry_date = date_ints[idx]
        if entry_date == last_traded_date:
            current_entry_ptr += 1
            continue

        signal_price = prices[idx]
        limit_price  = signal_price * (1.0 + rally_pct)
        signal_time  = time_mins[idx]

        fill_idx   = -1
        fill_price = 0.0
        for k in range(idx + 1, n_prices):
            if date_ints[k] != entry_date:
                break
            if time_mins[k] - signal_time > max_wait_minutes:
                break
            if highs[k] >= limit_price:
                fill_price = limit_price
                fill_idx = k
                break

        if fill_idx < 0:
            current_entry_ptr += 1
            last_traded_date = entry_date
            continue

        target_price = fill_price * (1 - tp)
        stop_price   = fill_price * (1 + sl)

        outcome    = 0.0
        exit_price = fill_price
        exit_type  = 0
        trade_done = False
        exit_idx   = fill_idx

        for k in range(fill_idx + 1, n_prices):
            if date_ints[k] != entry_date:
                exit_price = prices[k-1]
                outcome = (fill_price - exit_price) / fill_price
                exit_idx = k; trade_done = True; break
            tp_hit = lows[k] <= target_price
            sl_hit = highs[k] >= stop_price
            if sl_hit and tp_hit:
                if np.random.random() < 0.5:
                    outcome = -sl; exit_price = stop_price; exit_type = -1
                else:
                    outcome = tp; exit_price = target_price; exit_type = 1
                exit_idx = k; trade_done = True; break
            if sl_hit:
                outcome = -sl; exit_price = stop_price; exit_type = -1
                exit_idx = k; trade_done = True; break
            if tp_hit:
                outcome = tp; exit_price = target_price; exit_type = 1
                exit_idx = k; trade_done = True; break
            if time_mins[k] >= exit_time:
                exit_price = prices[k]
                outcome = (fill_price - exit_price) / fill_price
                exit_idx = k; trade_done = True; break

        if not trade_done:
            exit_price = prices[n_prices-1]
            outcome  = (fill_price - exit_price) / fill_price
            exit_idx = n_prices

        pnl_arr[trade_count]     = outcome
        date_arr[trade_count]    = entry_date
        entry_arr[trade_count]   = fill_price
        exit_arr[trade_count]    = exit_price
        outcome_arr[trade_count] = exit_type
        trade_count += 1
        last_traded_date = entry_date

        next_ptr = current_entry_ptr + 1
        while next_ptr < n_entries:
            if entry_indices[next_ptr] > exit_idx: break
            next_ptr += 1
        current_entry_ptr = next_ptr

    return (pnl_arr[:trade_count], date_arr[:trade_count],
            entry_arr[:trade_count], exit_arr[:trade_count],
            outcome_arr[:trade_count])


def _run_sim(indices, ohlcv_arrays, cfg):
    prices = ohlcv_arrays['prices']
    highs  = ohlcv_arrays['highs']
    lows   = ohlcv_arrays['lows']
    dates  = ohlcv_arrays['dates']
    times  = ohlcv_arrays['times'].astype(np.int64)

    mode = cfg['mode']
    if mode == 'grace':
        result = run_short_trade_sim_grace(
            indices, prices, highs, lows, dates, times,
            cfg['tp_pct'], cfg['sl_pct'], EXIT_TIME_MIN, cfg['grace_minutes'])
    elif mode == 'limit_entry':
        result = run_short_trade_sim_limit_entry(
            indices, prices, highs, lows, dates, times,
            cfg['tp_pct'], cfg['sl_pct'], EXIT_TIME_MIN,
            cfg['dip_pct'], cfg['max_wait'])
    else:
        result = run_short_trade_sim(
            indices, prices, highs, lows, dates, times,
            cfg['tp_pct'], cfg['sl_pct'], EXIT_TIME_MIN)

    pnl, d, entry_p, exit_p, outcome = result
    if len(pnl) > 0 and SLIPPAGE_PCT > 0:
        pnl = pnl - 2 * SLIPPAGE_PCT
    return pnl, d, entry_p, exit_p, outcome


# ==============================================================================
# DATA LOADING
# ==============================================================================

def _load_parquet_with_dt_index(gcs_uri, tz_out='America/Los_Angeles'):
    df = pd.read_parquet(gcs_uri)
    df = df.reset_index()
    dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    if dt_cols:
        df['Datetime'] = pd.to_datetime(df[dt_cols[0]], utc=True)
    else:
        num_col = next((c for c in df.columns if c in ('Timestamp', 'Date', 'Time')), df.columns[0])
        df['Datetime'] = pd.to_datetime(df[num_col], unit='ms', utc=True)
    df = df.set_index('Datetime')
    df = df[df.index.notna()].sort_index()
    df = df[~df.index.duplicated(keep='first')]
    df.index = df.index.tz_convert(tz_out)
    return df


def load_ticker_ohlcv(ticker):
    clean = TICKER_FILE_MAP.get(ticker, ticker)
    for suffix in ['_master_inference', '_1m_full_historical_data']:
        uri = f"gs://{GCS_BUCKET}/{GCS_INFER_DIR}/{clean}{suffix}.parquet"
        try:
            return _load_parquet_with_dt_index(uri)
        except Exception:
            continue
    raise FileNotFoundError(f"No OHLCV data found for {ticker}")


def precompute_ohlcv_arrays(df_ohlcv, start_date, end_date):
    eval_start = pd.to_datetime(start_date).tz_localize('America/Los_Angeles')
    eval_end   = pd.to_datetime(end_date).tz_localize('America/Los_Angeles') + timedelta(days=1)
    df = df_ohlcv[(df_ohlcv.index >= eval_start) & (df_ohlcv.index < eval_end)]
    if len(df) == 0:
        return None
    dates = (df.index.year * 10000 + df.index.month * 100 + df.index.day).values.astype(np.int64)
    times = (df.index.hour * 60 + df.index.minute).values
    return {
        'index':  df.index,
        'prices': df['Close'].values.copy(),
        'highs':  df['High'].values.copy(),
        'lows':   df['Low'].values.copy(),
        'dates':  dates,
        'times':  times,
    }


# ==============================================================================
# SCORING
# ==============================================================================

def score_npz_split(model, split, ticker_names):
    path = ensure_data(f'fold_{FOLD_ID}_{split}.npz')
    print(f"    Loading {split} npz...")
    data = np.load(path)
    X_seq     = data['X_seq']
    X_reg     = data['X_regime']
    X_id      = data['X_id'].flatten()
    date_ints = data['date_ints']
    minutes   = data['minutes']

    entry_mask = (minutes == SCORE_MINUTE)
    X_seq     = X_seq[entry_mask]
    X_reg     = X_reg[entry_mask]
    X_id      = X_id[entry_mask]
    date_ints = date_ints[entry_mask]
    n = len(X_id)
    del minutes, entry_mask; gc.collect()

    print(f"    Scoring {n:,} windows...")
    preds = model.predict(
        {'seq_input': X_seq.astype(np.float32),
         'regime_input': X_reg.astype(np.float32)},
        batch_size=512, verbose=0)
    del X_seq, X_reg; gc.collect()

    ticker_scores = {}
    for i in range(n):
        tid   = int(X_id[i])
        d_int = int(date_ints[i])
        if tid >= len(ticker_names):
            continue
        tname = str(ticker_names[tid])
        if tname in EXCLUDE_TICKERS or tname not in DEPLOY_TICKERS:
            continue
        if tname not in ticker_scores:
            ticker_scores[tname] = {h: {} for h in SCORE_HORIZONS}
        for h in SCORE_HORIZONS:
            ticker_scores[tname][h][d_int] = float(preds[i, h])
    del preds, X_id, date_ints; gc.collect()
    return ticker_scores


def run_scoring():
    print(f"\n{'='*W}")
    print(f"  SCORING — Spike Transformer SHORT fold {FOLD_ID}")
    print(f"{'='*W}")

    meta_path = ensure_data('meta.npz')
    meta = np.load(meta_path, allow_pickle=True)
    ticker_names = meta['ticker_names']

    global N_ASSET_FEATURES, N_REGIME_FEATURES
    probe_path = ensure_data(f'fold_{FOLD_ID}_val.npz')
    probe = np.load(probe_path)
    N_ASSET_FEATURES = probe['X_seq'].shape[-1]
    N_REGIME_FEATURES = probe['X_regime'].shape[-1]
    probe.close()

    model = SpikeTransformer(name='spike_transformer')
    dummy_seq = tf.zeros((1, TIME_STEPS, N_ASSET_FEATURES))
    dummy_reg = tf.zeros((1, N_REGIME_FEATURES))
    model({'seq_input': dummy_seq, 'regime_input': dummy_reg})
    weights_path = ensure_model(FOLD_ID)
    model.load_weights(weights_path)
    print(f"  Model: {model.count_params():,} params")

    val_scores  = score_npz_split(model, 'val', ticker_names)
    test_scores = score_npz_split(model, 'test', ticker_names)

    all_scores = {}
    for tname in set(list(val_scores.keys()) + list(test_scores.keys())):
        all_scores[tname] = {}
        for h in SCORE_HORIZONS:
            merged = {}
            if tname in val_scores and h in val_scores[tname]:
                merged.update(val_scores[tname][h])
            if tname in test_scores and h in test_scores[tname]:
                merged.update(test_scores[tname][h])
            all_scores[tname][h] = merged

    del model, val_scores, test_scores
    tf.keras.backend.clear_session()
    gc.collect()
    return all_scores


# ==============================================================================
# BACKTEST
# ==============================================================================

print(f"\n{'='*W}")
print(f"  ST SHORT FOLD 5 — PORTFOLIO BACKTEST")
print(f"  TEST: {TEST_START} -> {TEST_END}")
print(f"  Tickers: {len(DEPLOY_TICKERS)} ({', '.join(DEPLOY_TICKERS)})")
print(f"  Slippage: {SLIPPAGE_PCT:.2%} each way ({2*SLIPPAGE_PCT:.2%} RT)")
print(f"{'='*W}")

all_scores = run_scoring()

all_trades = []

for ticker in DEPLOY_TICKERS:
    cfg = DEPLOY_CONFIGS[ticker]
    h_idx = cfg['horizon']
    h_label = HORIZON_LABELS[h_idx]
    mode = cfg['mode']

    print(f"\n  {ticker} — H={h_label} mode={mode} prob>{cfg['prob_thresh']:.2f} "
          f"TP={cfg['tp_pct']:.0%} SL={cfg['sl_pct']:.0%}", end="")
    if mode == 'grace':
        print(f" grace={cfg['grace_minutes']}m", end="")
    elif mode == 'limit_entry':
        print(f" rally={cfg['dip_pct']:.1%} wait={cfg['max_wait']}m", end="")
    print()

    scores = all_scores.get(ticker, {})
    h_scores = scores.get(h_idx, {})
    if not h_scores:
        print(f"    No scores for H={h_label}")
        continue

    try:
        df_ohlcv = load_ticker_ohlcv(ticker)
    except Exception as e:
        print(f"    OHLCV load failed: {e}")
        continue

    ohlcv = precompute_ohlcv_arrays(df_ohlcv, TEST_START, TEST_END)
    if ohlcv is None:
        print(f"    No OHLCV in test window")
        continue

    score_by_date = {int(d): s for d, s in h_scores.items()}
    entry_list = []
    for i in range(len(ohlcv['dates'])):
        if ohlcv['times'][i] == ENTRY_MINUTE:
            day_score = score_by_date.get(int(ohlcv['dates'][i]), 0.0)
            if day_score >= cfg['prob_thresh']:
                entry_list.append(i)

    if not entry_list:
        print(f"    No signals in test window")
        continue

    indices = np.array(entry_list, dtype=np.int64)
    pnl, dates, entry_p, exit_p, outcomes = _run_sim(indices, ohlcv, cfg)

    if len(pnl) == 0:
        print(f"    No trades executed")
        continue

    avg_pnl = float(np.mean(pnl))
    wr = float(np.mean(pnl > 0))
    n = len(pnl)
    print(f"    {n} trades | Avg PnL: {avg_pnl:+.2%} | WR: {wr:.0%} | "
          f"Total: {float(np.sum(pnl)):+.2%}")

    for i in range(n):
        d = int(dates[i])
        date_str = f"{d // 10000}-{(d % 10000) // 100:02d}-{d % 100:02d}"
        otype = 'TP' if outcomes[i] == 1 else 'SL' if outcomes[i] == -1 else 'EOD'
        all_trades.append({
            'Ticker': ticker,
            'Date': date_str,
            'Date_int': d,
            'Entry': float(entry_p[i]),
            'Exit': float(exit_p[i]),
            'PnL': float(pnl[i]),
            'Outcome': otype,
            'Mode': mode,
            'Horizon': h_label,
        })

    del df_ohlcv, ohlcv
    gc.collect()


# ==============================================================================
# PORTFOLIO ANALYSIS
# ==============================================================================

if not all_trades:
    print("\n  NO TRADES — nothing to analyze")
else:
    trades_df = pd.DataFrame(all_trades)
    trades_df['Date_dt'] = pd.to_datetime(trades_df['Date'])
    trades_df = trades_df.sort_values('Date_dt')

    n_total   = len(trades_df)
    n_winners = (trades_df['PnL'] > 0).sum()
    n_losers  = (trades_df['PnL'] <= 0).sum()
    avg_pnl   = trades_df['PnL'].mean()
    total_pnl = trades_df['PnL'].sum()
    win_rate  = n_winners / n_total if n_total > 0 else 0
    avg_win   = trades_df.loc[trades_df['PnL'] > 0, 'PnL'].mean() if n_winners > 0 else 0
    avg_loss  = trades_df.loc[trades_df['PnL'] <= 0, 'PnL'].mean() if n_losers > 0 else 0

    n_days    = trades_df['Date_dt'].nunique()
    test_days = (pd.to_datetime(TEST_END) - pd.to_datetime(TEST_START)).days
    n_tickers_traded = trades_df['Ticker'].nunique()

    print(f"\n\n{'='*W}")
    print(f"  PORTFOLIO SUMMARY — ST SHORT Fold 5")
    print(f"  TEST: {TEST_START} -> {TEST_END} ({test_days} calendar days)")
    print(f"{'='*W}")
    print(f"  Total trades:     {n_total}")
    print(f"  Trading days:     {n_days}")
    print(f"  Tickers traded:   {n_tickers_traded}/{len(DEPLOY_TICKERS)}")
    print(f"  Trades/day:       {n_total/n_days:.1f}" if n_days > 0 else "")
    print(f"  Avg PnL/trade:    {avg_pnl:+.2%}")
    print(f"  Total PnL:        {total_pnl:+.2%}")
    print(f"  Win rate:         {win_rate:.1%} ({n_winners}W / {n_losers}L)")
    print(f"  Avg win:          {avg_win:+.2%}")
    print(f"  Avg loss:         {avg_loss:+.2%}")
    if avg_loss != 0:
        print(f"  Profit factor:    {abs(avg_win * n_winners / (avg_loss * n_losers)):.2f}")

    # ── Per-ticker breakdown ──
    print(f"\n  {'Ticker':<7} {'N':>4} {'Avg PnL':>9} {'Total':>9} {'WR':>6} {'TP':>4} {'SL':>4} {'EOD':>4}")
    print(f"  {'─'*52}")
    for ticker in DEPLOY_TICKERS:
        t_df = trades_df[trades_df['Ticker'] == ticker]
        if len(t_df) == 0:
            print(f"  {ticker:<7} {'—':>4}")
            continue
        t_n = len(t_df)
        t_avg = t_df['PnL'].mean()
        t_tot = t_df['PnL'].sum()
        t_wr = (t_df['PnL'] > 0).mean()
        t_tp = (t_df['Outcome'] == 'TP').sum()
        t_sl = (t_df['Outcome'] == 'SL').sum()
        t_eod = (t_df['Outcome'] == 'EOD').sum()
        print(f"  {ticker:<7} {t_n:>4} {t_avg:>+8.2%} {t_tot:>+8.2%} {t_wr:>5.0%} {t_tp:>4} {t_sl:>4} {t_eod:>4}")

    # ── Daily PnL ──
    daily = trades_df.groupby('Date').agg(
        N=('PnL', 'count'),
        Avg_PnL=('PnL', 'mean'),
        Total_PnL=('PnL', 'sum'),
        Tickers=('Ticker', lambda x: ','.join(sorted(x.unique()))),
    ).sort_index()

    print(f"\n  DAILY BREAKDOWN:")
    print(f"  {'Date':<12} {'N':>3} {'Avg PnL':>9} {'Total':>9} {'Cum':>9}  Tickers")
    print(f"  {'─'*75}")
    cum = 0.0
    for date, row in daily.iterrows():
        cum += row['Total_PnL']
        print(f"  {date:<12} {int(row['N']):>3} {row['Avg_PnL']:>+8.2%} "
              f"{row['Total_PnL']:>+8.2%} {cum:>+8.2%}  {row['Tickers']}")

    # ── Trade log ──
    print(f"\n  TRADE LOG:")
    print(f"  {'Date':<12} {'Ticker':<7} {'Entry':>8} {'Exit':>8} {'PnL':>8} {'Type':>5}")
    print(f"  {'─'*52}")
    for _, t in trades_df.iterrows():
        print(f"  {t['Date']:<12} {t['Ticker']:<7} ${t['Entry']:>7.2f} ${t['Exit']:>7.2f} "
              f"{t['PnL']:>+7.2%} {t['Outcome']:>5}")

    # ── Outcome distribution ──
    print(f"\n  OUTCOME DISTRIBUTION:")
    for otype in ['TP', 'SL', 'EOD']:
        o_df = trades_df[trades_df['Outcome'] == otype]
        if len(o_df) > 0:
            print(f"    {otype:>3}: {len(o_df):>3} trades ({len(o_df)/n_total:.0%})  "
                  f"Avg PnL: {o_df['PnL'].mean():+.2%}")

    # ==============================================================================
    # CHARTS
    # ==============================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 1. Cumulative PnL
    ax = axes[0, 0]
    cum_pnl = trades_df['PnL'].cumsum().values
    ax.plot(range(1, n_total + 1), cum_pnl * 100, color='#1565C0', linewidth=2)
    ax.fill_between(range(1, n_total + 1), cum_pnl * 100, 0,
                    where=cum_pnl >= 0, color='#1565C0', alpha=0.15)
    ax.fill_between(range(1, n_total + 1), cum_pnl * 100, 0,
                    where=cum_pnl < 0, color='crimson', alpha=0.15)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Trade #')
    ax.set_ylabel('Cumulative PnL (%)')
    ax.set_title(f'Cumulative PnL — {n_total} trades, {avg_pnl:+.2%}/trade')
    ax.grid(alpha=0.3)

    # 2. Daily PnL bars
    ax = axes[0, 1]
    daily_dates = pd.to_datetime(daily.index)
    colors = ['#2E7D32' if v >= 0 else '#C62828' for v in daily['Total_PnL']]
    ax.bar(daily_dates, daily['Total_PnL'].values * 100, color=colors,
           edgecolor='white', linewidth=0.5, width=0.8)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.set_ylabel('Daily Total PnL (%)')
    ax.set_title(f'Daily PnL — {n_days} trading days')
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 3. Per-ticker PnL
    ax = axes[1, 0]
    ticker_stats = trades_df.groupby('Ticker')['PnL'].agg(['mean', 'count'])
    ticker_stats = ticker_stats.sort_values('mean', ascending=True)
    colors = ['#2E7D32' if v >= 0 else '#C62828' for v in ticker_stats['mean']]
    bars = ax.barh(range(len(ticker_stats)), ticker_stats['mean'].values * 100,
                   color=colors, edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(ticker_stats)))
    ax.set_yticklabels([f"{t} (N={int(ticker_stats.loc[t, 'count'])})"
                        for t in ticker_stats.index], fontsize=9)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Avg PnL per Trade (%)')
    ax.set_title('Per-Ticker Performance')
    ax.grid(axis='x', alpha=0.3)

    # 4. Outcome pie
    ax = axes[1, 1]
    outcome_counts = trades_df['Outcome'].value_counts()
    labels = [f"{k}\n({v})" for k, v in outcome_counts.items()]
    pie_colors = {'TP': '#2E7D32', 'SL': '#C62828', 'EOD': '#FFA000'}
    ax.pie(outcome_counts.values,
           labels=labels,
           colors=[pie_colors.get(k, '#999') for k in outcome_counts.index],
           autopct='%1.0f%%', startangle=90, textprops={'fontsize': 10})
    ax.set_title('Trade Outcomes')

    fig.suptitle(f'ST SHORT Fold 5 Portfolio — TEST {TEST_START} to {TEST_END}\n'
                 f'{n_total} trades | {avg_pnl:+.2%}/trade | {win_rate:.0%} WR | '
                 f'{n_tickers_traded} tickers',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    chart_path = os.path.join(MODEL_DIR, 'backtest_st_short_fold5.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    print(f"\n  Chart saved: {chart_path}")
    if _IN_COLAB:
        plt.show()
    plt.close()

    print(f"\n{'='*W}")
    print(f"  BACKTEST COMPLETE")
    print(f"{'='*W}")
