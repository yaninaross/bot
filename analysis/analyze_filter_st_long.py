# %% ── Regime Filter Analysis for Spike Transformer LONG ─────────────────────
#
# Evaluates whether per-ticker regime filter (VIX/TLT/SPY trap day detection)
# improves Spike Transformer LONG PnL.
#
# TP/SL/prob_thresh/mode are FROZEN from existing BQ configs.
# Only the 3 regime filter params are swept on OPT, validated on TEST.
# Results saved to BQ as model_version = 'spike_transformer_fold4_v13'.
#
# GPU recommended for scoring phase. Colab-ready.

import os, time, gc, ctypes, uuid, warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from google.cloud import storage, bigquery as bq
from numba import jit
import pytz
from datetime import timedelta
from itertools import product

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


# ==============================================================================
# CONFIG
# ==============================================================================
FOLD_ID        = 4
SCORE_HORIZONS = [0, 1, 2]
HORIZON_LABELS = {0: '15m', 1: '30m', 2: '60m'}
DATA_START     = '2024-08-01'
OPT_START      = '2026-04-01'
OPT_END        = '2026-05-31'
TEST_START     = '2026-06-01'
TEST_END       = '2026-08-08'

ENTRY_TIME     = '06:15'
ENTRY_MINUTE   = 375
EXIT_TIME_MIN  = 780
SLIPPAGE_PCT   = 0.0015

MODEL_VERSION_IN   = 'spike_transformer_fold4'
MODEL_VERSION_OUT  = 'spike_transformer_fold4_v13'
EXPERIMENT_OUT     = 'spike_transformer_fold4_v13'
DEPLOY_PNL_THRESHOLD = 0.018
MIN_TRADES_ABSOLUTE  = 3

GCS_BUCKET       = os.environ.get("GCS_BUCKET", "")
GCS_DATA_PREFIX  = 'Algos/RA_MoE/real_data'
GCS_MODEL_PREFIX = 'Algos/RA_MoE/real_models'
GCS_INFER_DIR    = 'Data/Inference_v2'

DATA_DIR   = '/tmp/ra_moe_real_data'
MODEL_DIR  = '/tmp/ra_moe_models'
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

D_MODEL = 64; N_HEADS = 4; D_FF = 128; N_TRANSFORMER_BLOCKS = 2
DROPOUT = 0.1; FILM_HIDDEN = 64; TIME_STEPS = 128
TARGET_HORIZONS = [15, 30, 60]; N_HORIZONS = 3
ASL_GAMMA_NEG = 5.0; ASL_GAMMA_POS = 0.1; ASL_CLIP = 0.10

N_ASSET_FEATURES  = None
N_REGIME_FEATURES = None

NO_FILTER_DEFAULTS = {'vix_threshold': 999.0, 'tlt_crash': -999.0, 'spy_ignore': 999.0}

TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}
EXCLUDE_TICKERS = {'SPY', 'VIX', 'TLT', 'IWM', 'XLE', 'ITA', 'BITF', 'MIGI', 'SMLR'}

PST = pytz.timezone('America/Los_Angeles')

REGIME_GRID = {
    'vix_threshold': [15, 20, 25, 30, 35, 40, 50, 75, 100],
    'tlt_crash':     [0.0, -0.1, -0.2, -0.3, -0.4, -0.5],
    'spy_ignore':    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
}

WRITE_TO_BQ = False
WIPE_EXISTING_V13 = True

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


# ==============================================================================
# MODEL
# ==============================================================================

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
# BIGQUERY
# ==============================================================================
BQ_TABLE_ID = f"{GCP_PROJECT}.scoring_logs.wf_optimization_results"

def _ensure_wf_table(bq_client, table_id):
    from google.cloud.bigquery import SchemaField, Table
    schema = [
        SchemaField('run_id', 'STRING'),
        SchemaField('run_timestamp', 'TIMESTAMP'),
        SchemaField('experiment', 'STRING'),
        SchemaField('ticker', 'STRING'),
        SchemaField('seed', 'INT64'),
        SchemaField('entry_time', 'STRING'),
        SchemaField('entry_minute', 'INT64'),
        SchemaField('prob_thresh', 'FLOAT64'),
        SchemaField('tp_pct', 'FLOAT64'),
        SchemaField('sl_pct', 'FLOAT64'),
        SchemaField('vix_threshold', 'FLOAT64'),
        SchemaField('tlt_crash', 'FLOAT64'),
        SchemaField('spy_ignore', 'FLOAT64'),
        SchemaField('min_vol_bars', 'INT64'),
        SchemaField('vol_filter_pct', 'FLOAT64'),
        SchemaField('opt_avg_pnl', 'FLOAT64'),
        SchemaField('opt_win_rate', 'FLOAT64'),
        SchemaField('opt_freq', 'FLOAT64'),
        SchemaField('opt_count', 'INT64'),
        SchemaField('val_a_avg_pnl', 'FLOAT64'),
        SchemaField('val_a_win_rate', 'FLOAT64'),
        SchemaField('val_a_freq', 'FLOAT64'),
        SchemaField('val_a_count', 'INT64'),
        SchemaField('survived_a', 'BOOL'),
        SchemaField('val_b_avg_pnl', 'FLOAT64'),
        SchemaField('val_b_win_rate', 'FLOAT64'),
        SchemaField('val_b_freq', 'FLOAT64'),
        SchemaField('val_b_count', 'INT64'),
        SchemaField('survived_b', 'BOOL'),
        SchemaField('opt_start', 'STRING'),
        SchemaField('opt_end', 'STRING'),
        SchemaField('val_a_start', 'STRING'),
        SchemaField('val_a_end', 'STRING'),
        SchemaField('val_b_start', 'STRING'),
        SchemaField('val_b_end', 'STRING'),
        SchemaField('tpsl_mode', 'STRING'),
        SchemaField('tp_mult', 'FLOAT64'),
        SchemaField('sl_mult', 'FLOAT64'),
        SchemaField('trail_start_mult', 'FLOAT64'),
        SchemaField('trail_end_mult', 'FLOAT64'),
        SchemaField('model_version', 'STRING'),
        SchemaField('v0_fixed', 'BOOL'),
        SchemaField('is_ensemble', 'BOOL'),
        SchemaField('n_seeds_avg', 'INT64'),
        SchemaField('regime_tp_pct', 'FLOAT64'),
        SchemaField('regime_sl_pct', 'FLOAT64'),
        SchemaField('regime_lookback', 'INT64'),
        SchemaField('regime_spike_thresh', 'FLOAT64'),
        SchemaField('grace_minutes', 'INT64'),
        SchemaField('dip_pct', 'FLOAT64'),
        SchemaField('max_wait_minutes', 'INT64'),
        SchemaField('pre_range_filter_pct', 'FLOAT64'),
    ]
    bq_client.create_table(Table(table_id, schema=schema), exists_ok=True)
    table = bq_client.get_table(table_id)
    existing = {f.name for f in table.schema}
    new_fields = [f for f in schema if f.name not in existing]
    if new_fields:
        for f in new_fields:
            bq_client.query(f"ALTER TABLE `{table_id}` ADD COLUMN IF NOT EXISTS {f.name} {f.field_type}").result()

_bq_client = bq.Client()
_ensure_wf_table(_bq_client, BQ_TABLE_ID)


# ==============================================================================
# LONG SIMULATION ENGINES (Numba)
# ==============================================================================

@jit(nopython=True)
def run_trade_sim(entry_indices, prices, highs, lows, date_ints, time_mins,
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
            current_entry_ptr += 1
            continue

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
            outcome  = (prices[n_prices-1] - entry_price) / entry_price
            exit_idx = n_prices

        trades[trade_count]      = outcome
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
def run_trade_sim_grace(entry_indices, prices, highs, lows, date_ints, time_mins,
                        tp, sl, exit_time, grace_minutes):
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
            current_entry_ptr += 1
            continue

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
            outcome  = (prices[n_prices-1] - entry_price) / entry_price
            exit_idx = n_prices

        trades[trade_count]      = outcome
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
def run_trade_sim_limit_entry(entry_indices, prices, highs, lows, date_ints, time_mins,
                              tp, sl, exit_time, dip_pct, max_wait_minutes):
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
            current_entry_ptr += 1
            continue

        signal_price = prices[idx]
        limit_price  = signal_price * (1.0 - dip_pct)
        signal_time  = time_mins[idx]

        fill_idx   = -1
        fill_price = 0.0
        for k in range(idx + 1, n_prices):
            if date_ints[k] != entry_date:
                break
            if time_mins[k] - signal_time > max_wait_minutes:
                break
            if lows[k] <= limit_price:
                fill_price = limit_price
                fill_idx = k
                break

        if fill_idx < 0:
            current_entry_ptr += 1
            last_traded_date = entry_date
            continue

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
            outcome  = (prices[n_prices-1] - fill_price) / fill_price
            exit_idx = n_prices

        trades[trade_count]      = outcome
        entry_dates[trade_count] = entry_date
        trade_count += 1
        last_traded_date = entry_date

        next_ptr = current_entry_ptr + 1
        while next_ptr < n_entries:
            if entry_indices[next_ptr] > exit_idx: break
            next_ptr += 1
        current_entry_ptr = next_ptr

    return trades[:trade_count], entry_dates[:trade_count]


def _run_sim(indices, ohlcv_arrays, s_cfg, mode):
    prices = ohlcv_arrays['prices']
    highs  = ohlcv_arrays['highs']
    lows   = ohlcv_arrays['lows']
    dates  = ohlcv_arrays['dates']
    times  = ohlcv_arrays['times'].astype(np.int64)

    if mode == 'grace':
        pnl, d = run_trade_sim_grace(
            indices, prices, highs, lows, dates, times,
            s_cfg['tp_pct'], s_cfg['sl_pct'], EXIT_TIME_MIN, s_cfg['grace_minutes'])
    elif mode == 'limit_entry':
        pnl, d = run_trade_sim_limit_entry(
            indices, prices, highs, lows, dates, times,
            s_cfg['tp_pct'], s_cfg['sl_pct'], EXIT_TIME_MIN,
            s_cfg['dip_pct'], s_cfg['max_wait'])
    else:
        pnl, d = run_trade_sim(
            indices, prices, highs, lows, dates, times,
            s_cfg['tp_pct'], s_cfg['sl_pct'], EXIT_TIME_MIN)
    if len(pnl) > 0 and SLIPPAGE_PCT > 0:
        pnl = pnl - 2 * SLIPPAGE_PCT
    return pnl, d


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
        'months': max(1.0, (df.index[-1] - df.index[0]).days / 30.44),
    }


# ==============================================================================
# REGIME DATA LOADING (from optimize_wf_v12.py)
# ==============================================================================

def preload_regime_data(eval_start, eval_end):
    print("Loading regime data (VIX, SPY, TLT)...")
    pst = pytz.timezone('US/Pacific')

    def _load_regime_asset(ticker):
        clean = TICKER_FILE_MAP.get(ticker, ticker)
        for suffix in ['_master_inference', '_1m_full_historical_data']:
            uri = f"gs://{GCS_BUCKET}/{GCS_INFER_DIR}/{clean}{suffix}.parquet"
            try:
                return _load_parquet_with_dt_index(uri)
            except Exception:
                continue
        return None

    vix_df = _load_regime_asset('VIX')
    spy_df = _load_regime_asset('SPY')
    tlt_df = _load_regime_asset('TLT')

    spy_daily = spy_df['Close'].resample('D').last().dropna() if spy_df is not None else pd.Series(dtype=float)
    tlt_daily = tlt_df['Close'].resample('D').last().dropna() if tlt_df is not None else pd.Series(dtype=float)

    entry_minutes = list(range(360, 391))

    start = pd.Timestamp(eval_start, tz=pst) - timedelta(days=10)
    end   = pd.Timestamp(eval_end, tz=pst) + timedelta(days=1)

    def _build_lookup(df):
        if df is None: return {}
        sub = df[(df.index >= start) & (df.index <= end)]
        d_ints = (sub.index.year * 10000 + sub.index.month * 100 + sub.index.day).astype(int)
        m_ints = (sub.index.hour * 60 + sub.index.minute).astype(int)
        closes = sub['Close'].values
        return {(int(d_ints[i]), int(m_ints[i])): float(closes[i]) for i in range(len(sub))}

    vix_lk = _build_lookup(vix_df)
    spy_lk = _build_lookup(spy_df)
    tlt_lk = _build_lookup(tlt_df)

    def _trailing_median(lookup, d_int, entry_min, window=30):
        vals = []
        for m in range(entry_min - window + 1, entry_min + 1):
            v = lookup.get((d_int, m))
            if v is not None:
                vals.append(v)
        return float(np.median(vals)) if vals else None

    def _prev_close_map(daily_s):
        m = {}
        for i in range(1, len(daily_s)):
            d_int = int(daily_s.index[i].strftime('%Y%m%d'))
            m[d_int] = float(daily_s.iloc[i - 1])
        return m

    spy_prev = _prev_close_map(spy_daily)
    tlt_prev = _prev_close_map(tlt_daily)

    all_dates = pd.date_range(start, end, freq='D', tz=pst)
    regime_dict = {}
    for d in all_dates:
        d_int = int(d.strftime('%Y%m%d'))
        for em in entry_minutes:
            v_val = _trailing_median(vix_lk, d_int, em) or 15.0
            s_gap = 0.0
            sc = _trailing_median(spy_lk, d_int, em)
            sp = spy_prev.get(d_int)
            if sc is not None and sp is not None and sp != 0:
                s_gap = ((sc - sp) / sp) * 100
            t_gap = 0.0
            tc = _trailing_median(tlt_lk, d_int, em)
            tp = tlt_prev.get(d_int)
            if tc is not None and tp is not None and tp != 0:
                t_gap = ((tc - tp) / tp) * 100
            regime_dict[(d_int, em)] = np.array([v_val, t_gap, s_gap])

    print(f"   Regime data: {len(regime_dict)} (date, minute) combos")
    return regime_dict


def _attach_regime(raw_dates, regime_dict, n, entry_minute):
    trade_vix = np.zeros(n)
    trade_tlt = np.zeros(n)
    trade_spy = np.zeros(n)
    for i in range(n):
        key = (raw_dates[i], entry_minute)
        if key in regime_dict:
            v = regime_dict[key]
            trade_vix[i] = v[0]
            trade_tlt[i] = v[1]
            trade_spy[i] = v[2]
    return trade_vix, trade_tlt, trade_spy


def _apply_filter(raw_pnl, trade_vix, trade_tlt, trade_spy, vix_t, tlt_c, spy_i):
    trap = (trade_vix > vix_t) & (trade_tlt < tlt_c) & (trade_spy > spy_i)
    return raw_pnl[~trap], trap


# ==============================================================================
# SCORING PHASE
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
    n_total   = len(X_id)
    data.close()

    entry_mask = (minutes == ENTRY_MINUTE)
    n_entry = entry_mask.sum()
    print(f"    Filtering {n_total:,} -> {n_entry:,} entry-minute windows (min={ENTRY_MINUTE})")

    X_seq     = X_seq[entry_mask]
    X_reg     = X_reg[entry_mask]
    X_id      = X_id[entry_mask]
    date_ints = date_ints[entry_mask]
    del minutes, entry_mask; gc.collect()
    n = n_entry

    print(f"    Running inference on {n:,} samples...")
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
        if tname in EXCLUDE_TICKERS:
            continue
        if tname not in ticker_scores:
            ticker_scores[tname] = {h: {} for h in SCORE_HORIZONS}
        for h in SCORE_HORIZONS:
            ticker_scores[tname][h][d_int] = float(preds[i, h])
    del preds, X_id, date_ints; gc.collect()

    n_tickers = len(ticker_scores)
    n_scores = sum(len(v) for h_scores in ticker_scores.values() for v in h_scores.values())
    print(f"    {split}: {n_tickers} tickers, {n_scores} daily scores ({len(SCORE_HORIZONS)} horizons)")
    return ticker_scores


def run_scoring_phase():
    print(f"\n{'='*W}")
    print(f"  PHASE 1: SCORING — Spike Transformer fold {FOLD_ID}")
    print(f"{'='*W}")

    meta_path = ensure_data('meta.npz')
    meta = np.load(meta_path, allow_pickle=True)
    ticker_names = meta['ticker_names']
    print(f"  {len(ticker_names)} tickers in meta")

    print("  Building model...")
    global N_ASSET_FEATURES, N_REGIME_FEATURES
    probe_path = ensure_data(f'fold_{FOLD_ID}_val.npz')
    probe = np.load(probe_path)
    N_ASSET_FEATURES = probe['X_seq'].shape[-1]
    N_REGIME_FEATURES = probe['X_regime'].shape[-1]
    probe.close()
    print(f"    N_ASSET_FEATURES={N_ASSET_FEATURES}, N_REGIME_FEATURES={N_REGIME_FEATURES}")

    model = SpikeTransformer(name='spike_transformer')
    dummy_seq = tf.zeros((1, TIME_STEPS, N_ASSET_FEATURES))
    dummy_reg = tf.zeros((1, N_REGIME_FEATURES))
    model({'seq_input': dummy_seq, 'regime_input': dummy_reg})

    weights_path = ensure_model(FOLD_ID)
    model.load_weights(weights_path)
    print(f"  Model loaded: {model.count_params():,} params")

    val_scores = score_npz_split(model, 'val', ticker_names)
    test_scores = score_npz_split(model, 'test', ticker_names)

    all_scores = {}
    all_tickers = set(val_scores.keys()) | set(test_scores.keys())
    for tname in all_tickers:
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
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

    print(f"\n  Scoring complete: {len(all_scores)} tickers")
    return all_scores


# ==============================================================================
# PHASE 2: LOAD FROZEN CONFIGS FROM BQ
# ==============================================================================

def load_frozen_configs():
    print(f"\n{'='*W}")
    print(f"  PHASE 2: LOAD FROZEN CONFIGS from BQ (model_version='{MODEL_VERSION_IN}')")
    print(f"{'='*W}")

    query = f"""
    SELECT ticker, experiment, prob_thresh, tp_pct, sl_pct, tpsl_mode,
           grace_minutes, dip_pct, max_wait_minutes,
           val_a_avg_pnl, val_b_avg_pnl, val_b_win_rate, val_b_count,
           opt_avg_pnl, opt_count
    FROM `{BQ_TABLE_ID}`
    WHERE model_version = '{MODEL_VERSION_IN}'
      AND survived_a = TRUE
      AND val_b_avg_pnl >= {DEPLOY_PNL_THRESHOLD}
    ORDER BY val_b_avg_pnl DESC
    """
    df = _bq_client.query(query).to_dataframe()

    if len(df) == 0:
        print(f"  WARNING: No configs found. Trying without val_b threshold...")
        query_all = f"""
        SELECT ticker, experiment, prob_thresh, tp_pct, sl_pct, tpsl_mode,
               grace_minutes, dip_pct, max_wait_minutes,
               val_a_avg_pnl, val_b_avg_pnl, val_b_win_rate, val_b_count,
               opt_avg_pnl, opt_count
        FROM `{BQ_TABLE_ID}`
        WHERE model_version = '{MODEL_VERSION_IN}'
          AND survived_a = TRUE
        ORDER BY val_b_avg_pnl DESC
        """
        df = _bq_client.query(query_all).to_dataframe()

    if len(df) == 0:
        print("  ERROR: No configs found at all!")
        return {}

    HORIZON_LABEL_TO_IDX = {'15m': 0, '30m': 1, '60m': 2}

    def _parse_horizon_from_experiment(exp_str):
        for label, idx in HORIZON_LABEL_TO_IDX.items():
            if f'_{label}' in str(exp_str):
                return idx, label
        return 1, '30m'

    configs = {}
    for ticker in df['ticker'].unique():
        tdf = df[df['ticker'] == ticker]
        best = tdf.iloc[0]
        mode = best['tpsl_mode'] if pd.notna(best['tpsl_mode']) else 'static'
        h_idx, h_label = _parse_horizon_from_experiment(best['experiment'])

        cfg = {
            'ticker': ticker,
            'experiment': best['experiment'],
            'prob_thresh': float(best['prob_thresh']),
            'tp_pct': float(best['tp_pct']),
            'sl_pct': float(best['sl_pct']),
            'tpsl_mode': mode,
            'grace_minutes': int(best['grace_minutes']) if pd.notna(best.get('grace_minutes')) else 0,
            'dip_pct': float(best['dip_pct']) if pd.notna(best.get('dip_pct')) else 0.0,
            'max_wait': int(best['max_wait_minutes']) if pd.notna(best.get('max_wait_minutes')) else 0,
            'score_horizon': h_idx,
            'horizon_label': h_label,
            'val_b_avg_pnl': float(best['val_b_avg_pnl']) if pd.notna(best.get('val_b_avg_pnl')) else 0.0,
            'val_b_win_rate': float(best['val_b_win_rate']) if pd.notna(best.get('val_b_win_rate')) else 0.0,
            'val_b_count': int(best['val_b_count']) if pd.notna(best.get('val_b_count')) else 0,
            'opt_avg_pnl': float(best['opt_avg_pnl']) if pd.notna(best.get('opt_avg_pnl')) else 0.0,
        }
        configs[ticker] = cfg

    print(f"  Loaded {len(configs)} ticker configs from BQ")
    for t, c in sorted(configs.items()):
        print(f"    {t:<7} {c['tpsl_mode']:<14} H={c['horizon_label']}  "
              f"Prob>{c['prob_thresh']:.2f}  TP={c['tp_pct']:.0%}  SL={c['sl_pct']:.0%}  "
              f"TEST={c['val_b_avg_pnl']:+.2%}")
    return configs


# ==============================================================================
# PHASE 3: REGIME FILTER SWEEP
# ==============================================================================

def run_regime_filter_analysis(all_daily_scores, frozen_configs, regime_dict):
    print(f"\n{'='*W}")
    print(f"  PHASE 3: REGIME FILTER SWEEP")
    print(f"  Grid: VIX {REGIME_GRID['vix_threshold']}")
    print(f"         TLT {REGIME_GRID['tlt_crash']}")
    print(f"         SPY {REGIME_GRID['spy_ignore']}")
    n_combos = (len(REGIME_GRID['vix_threshold']) *
                len(REGIME_GRID['tlt_crash']) *
                len(REGIME_GRID['spy_ignore']))
    print(f"  Total: {n_combos} combos per ticker")
    print(f"{'='*W}\n")

    filter_combos = list(product(
        REGIME_GRID['vix_threshold'],
        REGIME_GRID['tlt_crash'],
        REGIME_GRID['spy_ignore'],
    ))

    results = []
    tickers = sorted(frozen_configs.keys())

    for t_idx, ticker in enumerate(tickers):
        cfg = frozen_configs[ticker]
        mode = cfg['tpsl_mode']
        h_idx = cfg['score_horizon']
        daily_scores = all_daily_scores.get(ticker, {})

        print(f"  [{t_idx+1}/{len(tickers)}] {ticker}  mode={mode}  H={cfg['horizon_label']}  "
              f"Prob>{cfg['prob_thresh']:.2f}  TP={cfg['tp_pct']:.0%}  SL={cfg['sl_pct']:.0%}")

        if not daily_scores:
            print(f"    SKIP: no daily scores")
            continue

        try:
            df_ohlcv = load_ticker_ohlcv(ticker)
        except Exception as e:
            print(f"    SKIP: {e}")
            continue

        ohlcv_opt  = precompute_ohlcv_arrays(df_ohlcv, OPT_START, OPT_END)
        ohlcv_test = precompute_ohlcv_arrays(df_ohlcv, TEST_START, TEST_END)
        del df_ohlcv

        if ohlcv_opt is None:
            print(f"    SKIP: no OHLCV in OPT window")
            continue

        # ── Simulate trades ONCE on OPT (no filter) ──
        h_scores = daily_scores.get(h_idx, {})
        opt_dates = ohlcv_opt['dates']
        opt_times = ohlcv_opt['times']

        probs_opt = np.zeros(len(opt_dates), dtype=np.float64)
        for i in range(len(opt_dates)):
            if opt_times[i] == ENTRY_MINUTE:
                probs_opt[i] = h_scores.get(int(opt_dates[i]), 0.0)

        mask_opt = (probs_opt >= cfg['prob_thresh']) & (opt_times == ENTRY_MINUTE)
        indices_opt = np.where(mask_opt)[0]

        if len(indices_opt) == 0:
            print(f"    SKIP: no signals in OPT")
            continue

        s_cfg = {
            'tp_pct': cfg['tp_pct'], 'sl_pct': cfg['sl_pct'],
            'grace_minutes': cfg['grace_minutes'],
            'dip_pct': cfg['dip_pct'], 'max_wait': cfg['max_wait'],
        }
        raw_pnl_opt, raw_dates_opt = _run_sim(indices_opt, ohlcv_opt, s_cfg, mode)

        if len(raw_pnl_opt) < MIN_TRADES_ABSOLUTE:
            print(f"    SKIP: only {len(raw_pnl_opt)} OPT trades (need {MIN_TRADES_ABSOLUTE})")
            continue

        # Baseline OPT
        baseline_opt_pnl = float(np.mean(raw_pnl_opt))
        baseline_opt_wr  = float(np.mean(raw_pnl_opt > 0))
        baseline_opt_n   = len(raw_pnl_opt)

        # Attach regime values to OPT trades
        n_opt = len(raw_pnl_opt)
        trade_vix_opt, trade_tlt_opt, trade_spy_opt = _attach_regime(
            raw_dates_opt, regime_dict, n_opt, ENTRY_MINUTE)

        # ── Sweep 324 combos on OPT ──
        best_combo = None
        best_opt_avg = baseline_opt_pnl

        for vix_t, tlt_c, spy_i in filter_combos:
            filtered_pnl, trap_mask = _apply_filter(
                raw_pnl_opt, trade_vix_opt, trade_tlt_opt, trade_spy_opt,
                vix_t, tlt_c, spy_i)
            if len(filtered_pnl) < MIN_TRADES_ABSOLUTE:
                continue
            avg = float(np.mean(filtered_pnl))
            if avg > best_opt_avg:
                best_opt_avg = avg
                best_combo = (vix_t, tlt_c, spy_i)

        # ── Simulate trades ONCE on TEST (no filter) ──
        baseline_test_pnl = 0.0
        baseline_test_wr  = 0.0
        baseline_test_n   = 0
        filtered_test_pnl = 0.0
        filtered_test_wr  = 0.0
        filtered_test_n   = 0
        test_blocked_pct  = 0.0

        if ohlcv_test is not None:
            test_dates = ohlcv_test['dates']
            test_times = ohlcv_test['times']

            probs_test = np.zeros(len(test_dates), dtype=np.float64)
            for i in range(len(test_dates)):
                if test_times[i] == ENTRY_MINUTE:
                    probs_test[i] = h_scores.get(int(test_dates[i]), 0.0)

            mask_test = (probs_test >= cfg['prob_thresh']) & (test_times == ENTRY_MINUTE)
            indices_test = np.where(mask_test)[0]

            if len(indices_test) > 0:
                raw_pnl_test, raw_dates_test = _run_sim(indices_test, ohlcv_test, s_cfg, mode)

                if len(raw_pnl_test) > 0:
                    baseline_test_pnl = float(np.mean(raw_pnl_test))
                    baseline_test_wr  = float(np.mean(raw_pnl_test > 0))
                    baseline_test_n   = len(raw_pnl_test)

                    if best_combo is not None:
                        n_test = len(raw_pnl_test)
                        trade_vix_test, trade_tlt_test, trade_spy_test = _attach_regime(
                            raw_dates_test, regime_dict, n_test, ENTRY_MINUTE)
                        filt_pnl, trap_mask = _apply_filter(
                            raw_pnl_test, trade_vix_test, trade_tlt_test, trade_spy_test,
                            best_combo[0], best_combo[1], best_combo[2])
                        if len(filt_pnl) > 0:
                            filtered_test_pnl = float(np.mean(filt_pnl))
                            filtered_test_wr  = float(np.mean(filt_pnl > 0))
                            filtered_test_n   = len(filt_pnl)
                            test_blocked_pct  = float(trap_mask.sum()) / n_test * 100

        # ── Result ──
        if best_combo is not None:
            opt_blocked = 0
            _, trap_mask_opt = _apply_filter(
                raw_pnl_opt, trade_vix_opt, trade_tlt_opt, trade_spy_opt,
                best_combo[0], best_combo[1], best_combo[2])
            opt_blocked = float(trap_mask_opt.sum()) / n_opt * 100

        delta = filtered_test_pnl - baseline_test_pnl if best_combo else 0.0
        filter_helps = best_combo is not None and delta > 0

        result = {
            'ticker': ticker,
            'mode': mode,
            'horizon_label': cfg['horizon_label'],
            'score_horizon': h_idx,
            'prob_thresh': cfg['prob_thresh'],
            'tp_pct': cfg['tp_pct'],
            'sl_pct': cfg['sl_pct'],
            'grace_minutes': cfg['grace_minutes'],
            'dip_pct': cfg['dip_pct'],
            'max_wait': cfg['max_wait'],
            'baseline_opt_pnl': baseline_opt_pnl,
            'baseline_opt_wr': baseline_opt_wr,
            'baseline_opt_n': baseline_opt_n,
            'baseline_test_pnl': baseline_test_pnl,
            'baseline_test_wr': baseline_test_wr,
            'baseline_test_n': baseline_test_n,
            'best_vix': best_combo[0] if best_combo else NO_FILTER_DEFAULTS['vix_threshold'],
            'best_tlt': best_combo[1] if best_combo else NO_FILTER_DEFAULTS['tlt_crash'],
            'best_spy': best_combo[2] if best_combo else NO_FILTER_DEFAULTS['spy_ignore'],
            'filtered_opt_pnl': best_opt_avg if best_combo else baseline_opt_pnl,
            'filtered_test_pnl': filtered_test_pnl if best_combo else baseline_test_pnl,
            'filtered_test_wr': filtered_test_wr if best_combo else baseline_test_wr,
            'filtered_test_n': filtered_test_n if best_combo else baseline_test_n,
            'opt_blocked_pct': opt_blocked if best_combo else 0.0,
            'test_blocked_pct': test_blocked_pct,
            'delta': delta,
            'filter_helps': filter_helps,
            'filter_found': best_combo is not None,
        }
        results.append(result)

        # Print
        if best_combo:
            verdict = "BETTER" if filter_helps else "WORSE"
            print(f"    Baseline OPT: {baseline_opt_pnl:+.2%}  ({baseline_opt_n} trades)")
            print(f"    Baseline TEST: {baseline_test_pnl:+.2%}  ({baseline_test_n} trades)")
            print(f"    Filter: VIX>{best_combo[0]}  TLT<{best_combo[1]}  SPY>{best_combo[2]}")
            print(f"    Filtered OPT:  {best_opt_avg:+.2%}  (blocked {opt_blocked:.0f}%)")
            print(f"    Filtered TEST: {filtered_test_pnl:+.2%}  ({filtered_test_n} trades, blocked {test_blocked_pct:.0f}%)  [{verdict}]")
            print(f"    Delta: {delta:+.2%}")
        else:
            print(f"    Baseline OPT: {baseline_opt_pnl:+.2%}  ({baseline_opt_n} trades)")
            print(f"    Baseline TEST: {baseline_test_pnl:+.2%}  ({baseline_test_n} trades)")
            print(f"    No filter combo improved OPT avg PnL")

        del ohlcv_opt, ohlcv_test
        gc.collect()

    return results


# ==============================================================================
# PHASE 4: CHARTS + SUMMARY
# ==============================================================================

def print_summary_and_charts(results):
    print(f"\n\n{'='*W}")
    print(f"  REGIME FILTER ANALYSIS — SUMMARY")
    print(f"{'='*W}")

    if not results:
        print("  No results to display.")
        return

    df = pd.DataFrame(results)

    # ── Summary table ──
    print(f"\n  {'Ticker':<7} {'Mode':<14} {'BL_OPT':>7} {'BL_TEST':>8} {'FILT_TEST':>9} "
          f"{'Delta':>7} {'VIX':>5} {'TLT':>6} {'SPY':>5} {'Blk%':>5} {'Verdict':>8}")
    print(f"  {'─'*90}")

    for _, r in df.sort_values('delta', ascending=False).iterrows():
        if r['filter_found']:
            verdict = "BETTER" if r['filter_helps'] else "WORSE"
            print(f"  {r['ticker']:<7} {r['mode']:<14} "
                  f"{r['baseline_opt_pnl']:>+6.2%} {r['baseline_test_pnl']:>+7.2%} "
                  f"{r['filtered_test_pnl']:>+8.2%} {r['delta']:>+6.2%} "
                  f"{r['best_vix']:>5.0f} {r['best_tlt']:>+5.1f} {r['best_spy']:>+4.1f} "
                  f"{r['test_blocked_pct']:>4.0f}% {verdict:>8}")
        else:
            print(f"  {r['ticker']:<7} {r['mode']:<14} "
                  f"{r['baseline_opt_pnl']:>+6.2%} {r['baseline_test_pnl']:>+7.2%} "
                  f"{'N/A':>9} {'0.00%':>7} "
                  f"{'---':>5} {'---':>6} {'---':>5} "
                  f"{'0%':>5} {'NO_FILT':>8}")

    # ── Aggregate stats ──
    n_total = len(df)
    n_filter_found = df['filter_found'].sum()
    n_helps = df['filter_helps'].sum()
    n_hurts = n_filter_found - n_helps
    n_no_filter = n_total - n_filter_found

    avg_baseline = df['baseline_test_pnl'].mean()
    filt_mask = df['filter_found']
    if filt_mask.any():
        avg_filtered = df.loc[filt_mask, 'filtered_test_pnl'].mean()
        avg_delta = df.loc[filt_mask, 'delta'].mean()
    else:
        avg_filtered = avg_baseline
        avg_delta = 0.0

    print(f"\n  AGGREGATE:")
    print(f"    {n_total} tickers analyzed")
    print(f"    {n_filter_found} found a filter combo that beat OPT baseline")
    print(f"    {n_helps} filter HELPS on TEST (delta > 0)")
    print(f"    {n_hurts} filter HURTS on TEST (delta <= 0)")
    print(f"    {n_no_filter} no filter improved OPT")
    print(f"    Avg baseline TEST PnL: {avg_baseline:+.2%}")
    print(f"    Avg filtered TEST PnL: {avg_filtered:+.2%}")
    print(f"    Avg delta: {avg_delta:+.2%}")

    # ── Charts ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1: Per-ticker PnL bars (baseline vs filtered)
    ax1 = axes[0, 0]
    df_sorted = df.sort_values('delta', ascending=True)
    y_pos = range(len(df_sorted))
    baseline_vals = df_sorted['baseline_test_pnl'].values * 100
    filtered_vals = df_sorted.apply(
        lambda r: r['filtered_test_pnl'] * 100 if r['filter_found'] else r['baseline_test_pnl'] * 100,
        axis=1).values
    ax1.barh(y_pos, baseline_vals, height=0.4, align='edge', label='Baseline', color='steelblue', alpha=0.7)
    ax1.barh([y + 0.4 for y in y_pos], filtered_vals, height=0.4, align='edge', label='Filtered', color='green', alpha=0.7)
    ax1.set_yticks([y + 0.4 for y in y_pos])
    ax1.set_yticklabels(df_sorted['ticker'].values, fontsize=8)
    ax1.set_xlabel('TEST Avg PnL (%)')
    ax1.set_title('Baseline vs Filtered TEST PnL')
    ax1.axvline(0, color='black', linewidth=0.5)
    ax1.legend(fontsize=8)
    ax1.grid(axis='x', alpha=0.3)

    # Panel 2: Selected filter values per ticker
    ax2 = axes[0, 1]
    df_with_filter = df[df['filter_found']].sort_values('ticker')
    if len(df_with_filter) > 0:
        x_pos = range(len(df_with_filter))
        width = 0.25
        ax2.bar([x - width for x in x_pos], df_with_filter['best_vix'].values,
                width=width, label='VIX thresh', color='red', alpha=0.7)
        ax2_tlt = ax2.twinx()
        ax2_tlt.bar(list(x_pos), df_with_filter['best_tlt'].values,
                     width=width, label='TLT crash', color='blue', alpha=0.7)
        ax2_tlt.bar([x + width for x in x_pos], df_with_filter['best_spy'].values,
                     width=width, label='SPY ignore', color='orange', alpha=0.7)
        ax2.set_xticks(list(x_pos))
        ax2.set_xticklabels(df_with_filter['ticker'].values, rotation=45, fontsize=8)
        ax2.set_ylabel('VIX threshold', color='red')
        ax2_tlt.set_ylabel('TLT crash / SPY ignore (%)', color='blue')
        ax2.set_title('Selected Filter Values per Ticker')
        ax2.legend(loc='upper left', fontsize=7)
        ax2_tlt.legend(loc='upper right', fontsize=7)
    else:
        ax2.text(0.5, 0.5, 'No filters found', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Selected Filter Values per Ticker')

    # Panel 3: Scatter — % trades blocked vs PnL delta
    ax3 = axes[1, 0]
    if len(df_with_filter) > 0:
        colors = ['green' if d > 0 else 'red' for d in df_with_filter['delta']]
        ax3.scatter(df_with_filter['test_blocked_pct'], df_with_filter['delta'] * 100,
                    c=colors, s=60, edgecolors='black', linewidth=0.5, alpha=0.8)
        for _, r in df_with_filter.iterrows():
            ax3.annotate(r['ticker'], (r['test_blocked_pct'], r['delta'] * 100),
                        fontsize=7, ha='left', va='bottom')
        ax3.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax3.set_xlabel('Trades Blocked (%)')
        ax3.set_ylabel('PnL Delta (%)')
        ax3.set_title('Filter Impact: Blocked % vs PnL Change')
        ax3.grid(alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No filters found', ha='center', va='center', transform=ax3.transAxes)

    # Panel 4: Portfolio bar chart
    ax4 = axes[1, 1]
    categories = ['Baseline\n(all tickers)', 'Filtered\n(filter found)', 'Helps\n(delta>0)', 'Hurts\n(delta<=0)']
    values = [avg_baseline * 100]
    if filt_mask.any():
        values.append(avg_filtered * 100)
        helps_df = df[df['filter_helps']]
        hurts_df = df[df['filter_found'] & ~df['filter_helps']]
        values.append(helps_df['filtered_test_pnl'].mean() * 100 if len(helps_df) > 0 else 0)
        values.append(hurts_df['filtered_test_pnl'].mean() * 100 if len(hurts_df) > 0 else 0)
    else:
        values.extend([0, 0, 0])

    bar_colors = ['steelblue', 'green', 'darkgreen', 'crimson']
    ax4.bar(categories, values, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax4.set_ylabel('Avg TEST PnL (%)')
    ax4.set_title('Portfolio Summary')
    ax4.axhline(0, color='black', linewidth=0.5)
    ax4.grid(axis='y', alpha=0.3)
    for i, v in enumerate(values):
        ax4.text(i, v + 0.05, f'{v:.2f}%', ha='center', fontsize=9, fontweight='bold')

    plt.suptitle(f'Regime Filter Analysis — Spike Transformer LONG (Fold {FOLD_ID})\n'
                 f'OPT: {OPT_START} to {OPT_END}  |  TEST: {TEST_START} to {TEST_END}  |  '
                 f'{n_total} tickers, {len(filter_combos)} filter combos',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()

    chart_path = '/tmp/ra_moe_models/regime_filter_analysis_st_long.png'
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    print(f"\n  Chart saved: {chart_path}")
    if _IN_COLAB:
        plt.show()
    plt.close()

    return df


# ==============================================================================
# PHASE 5: OPTIONAL BQ WRITE
# ==============================================================================

def write_results_to_bq(results):
    if not WRITE_TO_BQ:
        print(f"\n  WRITE_TO_BQ = False — skipping BQ write.")
        print(f"  Set WRITE_TO_BQ = True and re-run to save as '{MODEL_VERSION_OUT}'")
        return

    print(f"\n{'='*W}")
    print(f"  PHASE 5: WRITING TO BQ as model_version='{MODEL_VERSION_OUT}'")
    print(f"{'='*W}")

    if WIPE_EXISTING_V13:
        try:
            del_q = (f"DELETE FROM `{BQ_TABLE_ID}` "
                     f"WHERE model_version = '{MODEL_VERSION_OUT}'")
            _bq_client.query(del_q).result()
            print(f"  WIPE: deleted existing model_version='{MODEL_VERSION_OUT}'")
        except Exception as e:
            if 'streaming buffer' in str(e).lower():
                copy_q = (f"CREATE OR REPLACE TABLE `{BQ_TABLE_ID}` AS "
                          f"SELECT * FROM `{BQ_TABLE_ID}` "
                          f"WHERE model_version != '{MODEL_VERSION_OUT}'")
                _bq_client.query(copy_q).result()
                print(f"  WIPE: rebuilt without '{MODEL_VERSION_OUT}'")
            else:
                raise

    run_id = str(uuid.uuid4())
    n_written = 0

    for r in results:
        vix_t = r['best_vix'] if r['filter_helps'] else NO_FILTER_DEFAULTS['vix_threshold']
        tlt_c = r['best_tlt'] if r['filter_helps'] else NO_FILTER_DEFAULTS['tlt_crash']
        spy_i = r['best_spy'] if r['filter_helps'] else NO_FILTER_DEFAULTS['spy_ignore']

        test_pnl = r['filtered_test_pnl'] if r['filter_helps'] else r['baseline_test_pnl']
        test_wr  = r['filtered_test_wr'] if r['filter_helps'] else r['baseline_test_wr']
        test_n   = r['filtered_test_n'] if r['filter_helps'] else r['baseline_test_n']

        mode = r['mode']
        h_label = r['horizon_label']
        mode_sfx = {'static': '', 'grace': '_GRACE', 'limit_entry': '_LIMIT'}.get(mode, '')
        bq_experiment = f"{EXPERIMENT_OUT}_{h_label}{mode_sfx}"

        row = {
            'run_id': run_id,
            'run_timestamp': pd.Timestamp.utcnow().isoformat(),
            'experiment': bq_experiment,
            'ticker': r['ticker'],
            'seed': 0,
            'entry_time': ENTRY_TIME,
            'entry_minute': ENTRY_MINUTE,
            'prob_thresh': r['prob_thresh'],
            'tp_pct': r['tp_pct'],
            'sl_pct': r['sl_pct'],
            'vix_threshold': vix_t,
            'tlt_crash': tlt_c,
            'spy_ignore': spy_i,
            'min_vol_bars': 0,
            'vol_filter_pct': 0.0,
            'opt_avg_pnl': r['baseline_opt_pnl'],
            'opt_win_rate': r['baseline_opt_wr'],
            'opt_freq': 0.0,
            'opt_count': r['baseline_opt_n'],
            'val_a_avg_pnl': r['filtered_opt_pnl'] if r['filter_helps'] else r['baseline_opt_pnl'],
            'val_a_win_rate': r['baseline_opt_wr'],
            'val_a_freq': 0.0,
            'val_a_count': r['baseline_opt_n'],
            'survived_a': True,
            'val_b_avg_pnl': test_pnl,
            'val_b_win_rate': test_wr,
            'val_b_freq': 0.0,
            'val_b_count': test_n,
            'survived_b': test_pnl > 0,
            'opt_start': OPT_START,
            'opt_end': OPT_END,
            'val_a_start': OPT_START,
            'val_a_end': OPT_END,
            'val_b_start': TEST_START,
            'val_b_end': TEST_END,
            'tpsl_mode': mode,
            'tp_mult': None, 'sl_mult': None,
            'trail_start_mult': None, 'trail_end_mult': None,
            'model_version': MODEL_VERSION_OUT,
            'v0_fixed': True,
            'is_ensemble': False, 'n_seeds_avg': 1,
            'regime_tp_pct': None, 'regime_sl_pct': None,
            'regime_lookback': None, 'regime_spike_thresh': None,
            'grace_minutes': r['grace_minutes'] if mode == 'grace' else None,
            'dip_pct': r['dip_pct'] if mode == 'limit_entry' else None,
            'max_wait_minutes': r['max_wait'] if mode == 'limit_entry' else None,
            'pre_range_filter_pct': 0.0,
        }

        try:
            errors = _bq_client.insert_rows_json(BQ_TABLE_ID, [row])
            if errors:
                print(f"    BQ error {r['ticker']}: {errors[0]}")
            else:
                filter_tag = f"VIX>{vix_t:.0f} TLT<{tlt_c:.1f} SPY>{spy_i:.1f}" if r['filter_helps'] else "NO_FILTER"
                print(f"    BQ: {r['ticker']}/{mode}  {filter_tag}  TEST={test_pnl:+.2%}")
                n_written += 1
        except Exception as e:
            print(f"    BQ save failed {r['ticker']}: {e}")

    print(f"\n  Written {n_written}/{len(results)} rows to BQ as '{MODEL_VERSION_OUT}'")


# ==============================================================================
# RUN ALL PHASES
# ==============================================================================

# Generate filter combos count for chart title reference
filter_combos = list(product(
    REGIME_GRID['vix_threshold'],
    REGIME_GRID['tlt_crash'],
    REGIME_GRID['spy_ignore'],
))

# Phase 1: Score
all_daily_scores = run_scoring_phase()

# Phase 2: Load frozen configs
frozen_configs = load_frozen_configs()

if not frozen_configs:
    print("\nERROR: No frozen configs loaded. Cannot proceed.")
else:
    # Phase 3: Load regime data + sweep
    regime_dict = preload_regime_data(OPT_START, TEST_END)

    results = run_regime_filter_analysis(all_daily_scores, frozen_configs, regime_dict)

    # Phase 4: Summary + charts
    summary_df = print_summary_and_charts(results)

    # Phase 5: Optional BQ write
    write_results_to_bq(results)

    # Grand total
    n_helps = sum(1 for r in results if r['filter_helps'])
    n_total = len(results)
    print(f"\n{'='*W}")
    print(f"  COMPLETE — {n_total} tickers, {n_helps} improved by regime filter")
    print(f"  model_version_in:  '{MODEL_VERSION_IN}'")
    print(f"  model_version_out: '{MODEL_VERSION_OUT}' (WRITE_TO_BQ={WRITE_TO_BQ})")
    print(f"{'='*W}")
