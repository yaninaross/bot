# %% ── Volume Filter Analysis for Spike Transformer LONG ─────────────────────
#
# Evaluates whether per-ticker volume filter (entry-bar volume percentile)
# improves Spike Transformer LONG PnL.
#
# TP/SL/prob_thresh/mode are FROZEN from existing BQ configs.
# Only the volume filter threshold is swept on OPT, validated on TEST.
# Results saved to BQ as model_version = 'spike_transformer_fold4_v14'.
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
MODEL_VERSION_OUT  = 'spike_transformer_fold4_v14'
EXPERIMENT_OUT     = 'spike_transformer_fold4_v14'
DEPLOY_PNL_THRESHOLD = 0.018
MIN_TRADES_ABSOLUTE  = 3

GCS_BUCKET       = 'time-series-bucket-ys'
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

VOL_PCT_GRID = [0, 10, 15, 20, 25, 30, 35, 40, 50, 60, 75]
VOL_LOOKBACK_DAYS = 30

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
BQ_TABLE_ID = "ys-time-series.scoring_logs.wf_optimization_results"

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


def precompute_ohlcv_arrays(df_ohlcv, start_date, end_date, vol_baselines=None):
    eval_start = pd.to_datetime(start_date).tz_localize('America/Los_Angeles')
    eval_end   = pd.to_datetime(end_date).tz_localize('America/Los_Angeles') + timedelta(days=1)
    df = df_ohlcv[(df_ohlcv.index >= eval_start) & (df_ohlcv.index < eval_end)]
    if len(df) == 0:
        return None
    dates = (df.index.year * 10000 + df.index.month * 100 + df.index.day).values.astype(np.int64)
    times = (df.index.hour * 60 + df.index.minute).values
    result = {
        'index':  df.index,
        'prices': df['Close'].values.copy(),
        'highs':  df['High'].values.copy(),
        'lows':   df['Low'].values.copy(),
        'volumes': df['Volume'].values.copy(),
        'dates':  dates,
        'times':  times,
        'months': max(1.0, (df.index[-1] - df.index[0]).days / 30.44),
    }
    if vol_baselines is not None:
        result['vol_baselines'] = vol_baselines
    return result


# ==============================================================================
# VOLUME FILTER (from optimize_wf_v12.py)
# ==============================================================================

def precompute_vol_baselines(df_ohlcv, lookback_days=VOL_LOOKBACK_DAYS):
    times = (df_ohlcv.index.hour * 60 + df_ohlcv.index.minute).values
    dates = (df_ohlcv.index.year * 10000 + df_ohlcv.index.month * 100
             + df_ohlcv.index.day).values.astype(np.int64)
    volumes = df_ohlcv['Volume'].values

    baselines = {}
    for em in range(358, 393):
        at_em = np.where(times == em)[0]
        for j in range(len(at_em)):
            d_int = int(dates[at_em[j]])
            start_j = max(0, j - lookback_days)
            if j - start_j < 5:
                baselines[(d_int, em)] = 0.0
                continue
            prev_vols = volumes[at_em[start_j:j]]
            baselines[(d_int, em)] = float(np.median(prev_vols))
    return baselines


def _compute_vol_pct_ok(volumes, times, dates, vol_baselines, entry_minute, threshold_pct):
    n = len(volumes)
    ok = np.ones(n, dtype=np.bool_)
    if threshold_pct <= 0:
        return ok
    at_entry = np.where(times == entry_minute)[0]
    for idx in at_entry:
        baseline = vol_baselines.get((int(dates[idx]), entry_minute), 0.0)
        if baseline > 0:
            pct = (volumes[idx] / baseline) * 100
            if pct < threshold_pct:
                ok[idx] = False
    return ok


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
# PHASE 3: VOLUME FILTER SWEEP
# ==============================================================================

def run_volume_filter_analysis(all_daily_scores, frozen_configs):
    print(f"\n{'='*W}")
    print(f"  PHASE 3: VOLUME FILTER SWEEP")
    print(f"  Grid: vol_filter_pct = {VOL_PCT_GRID}")
    print(f"  Lookback: {VOL_LOOKBACK_DAYS} days trailing median")
    print(f"  {len(VOL_PCT_GRID)} thresholds per ticker")
    print(f"{'='*W}\n")

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

        vol_baselines = precompute_vol_baselines(df_ohlcv)
        ohlcv_opt  = precompute_ohlcv_arrays(df_ohlcv, OPT_START, OPT_END, vol_baselines)
        ohlcv_test = precompute_ohlcv_arrays(df_ohlcv, TEST_START, TEST_END, vol_baselines)
        del df_ohlcv

        if ohlcv_opt is None:
            print(f"    SKIP: no OHLCV in OPT window")
            continue

        h_scores = daily_scores.get(h_idx, {})
        s_cfg = {
            'tp_pct': cfg['tp_pct'], 'sl_pct': cfg['sl_pct'],
            'grace_minutes': cfg['grace_minutes'],
            'dip_pct': cfg['dip_pct'], 'max_wait': cfg['max_wait'],
        }

        # ── Baseline on OPT (vol_filter_pct=0, no filter) ──
        opt_dates = ohlcv_opt['dates']
        opt_times = ohlcv_opt['times']

        probs_opt = np.zeros(len(opt_dates), dtype=np.float64)
        for i in range(len(opt_dates)):
            if opt_times[i] == ENTRY_MINUTE:
                probs_opt[i] = h_scores.get(int(opt_dates[i]), 0.0)

        base_mask_opt = (probs_opt >= cfg['prob_thresh']) & (opt_times == ENTRY_MINUTE)
        base_indices_opt = np.where(base_mask_opt)[0]

        if len(base_indices_opt) == 0:
            print(f"    SKIP: no signals in OPT")
            continue

        raw_pnl_opt_base, raw_dates_opt_base = _run_sim(base_indices_opt, ohlcv_opt, s_cfg, mode)
        if len(raw_pnl_opt_base) < MIN_TRADES_ABSOLUTE:
            print(f"    SKIP: only {len(raw_pnl_opt_base)} OPT trades (need {MIN_TRADES_ABSOLUTE})")
            continue

        baseline_opt_pnl = float(np.mean(raw_pnl_opt_base))
        baseline_opt_wr  = float(np.mean(raw_pnl_opt_base > 0))
        baseline_opt_n   = len(raw_pnl_opt_base)
        n_signals_opt    = int(base_mask_opt.sum())

        # ── Baseline on TEST (vol_filter_pct=0) ──
        baseline_test_pnl = 0.0
        baseline_test_wr  = 0.0
        baseline_test_n   = 0

        if ohlcv_test is not None:
            test_dates = ohlcv_test['dates']
            test_times = ohlcv_test['times']

            probs_test = np.zeros(len(test_dates), dtype=np.float64)
            for i in range(len(test_dates)):
                if test_times[i] == ENTRY_MINUTE:
                    probs_test[i] = h_scores.get(int(test_dates[i]), 0.0)

            base_mask_test = (probs_test >= cfg['prob_thresh']) & (test_times == ENTRY_MINUTE)
            base_indices_test = np.where(base_mask_test)[0]

            if len(base_indices_test) > 0:
                raw_pnl_test_base, _ = _run_sim(base_indices_test, ohlcv_test, s_cfg, mode)
                if len(raw_pnl_test_base) > 0:
                    baseline_test_pnl = float(np.mean(raw_pnl_test_base))
                    baseline_test_wr  = float(np.mean(raw_pnl_test_base > 0))
                    baseline_test_n   = len(raw_pnl_test_base)

        # ── Sweep vol_filter_pct on OPT ──
        best_vpct = 0
        best_opt_avg = baseline_opt_pnl

        for vpct in VOL_PCT_GRID:
            if vpct == 0:
                continue
            vol_ok = _compute_vol_pct_ok(
                ohlcv_opt['volumes'], ohlcv_opt['times'], ohlcv_opt['dates'],
                vol_baselines, ENTRY_MINUTE, vpct)
            filt_mask = base_mask_opt & vol_ok
            filt_indices = np.where(filt_mask)[0]
            if len(filt_indices) == 0:
                continue

            pnl, _ = _run_sim(filt_indices, ohlcv_opt, s_cfg, mode)
            if len(pnl) < MIN_TRADES_ABSOLUTE:
                continue
            avg = float(np.mean(pnl))
            if avg > best_opt_avg:
                best_opt_avg = avg
                best_vpct = vpct

        # ── Validate best vpct on TEST ──
        filtered_test_pnl = baseline_test_pnl
        filtered_test_wr  = baseline_test_wr
        filtered_test_n   = baseline_test_n
        test_blocked_pct  = 0.0
        opt_blocked_pct   = 0.0

        if best_vpct > 0:
            vol_ok_opt = _compute_vol_pct_ok(
                ohlcv_opt['volumes'], ohlcv_opt['times'], ohlcv_opt['dates'],
                vol_baselines, ENTRY_MINUTE, best_vpct)
            n_signals_blocked_opt = int((base_mask_opt & ~vol_ok_opt).sum())
            opt_blocked_pct = n_signals_blocked_opt / max(1, n_signals_opt) * 100

            if ohlcv_test is not None and len(base_indices_test) > 0:
                vol_ok_test = _compute_vol_pct_ok(
                    ohlcv_test['volumes'], ohlcv_test['times'], ohlcv_test['dates'],
                    vol_baselines, ENTRY_MINUTE, best_vpct)
                filt_mask_test = base_mask_test & vol_ok_test
                filt_indices_test = np.where(filt_mask_test)[0]
                n_signals_test = int(base_mask_test.sum())
                n_blocked_test = int((base_mask_test & ~vol_ok_test).sum())
                test_blocked_pct = n_blocked_test / max(1, n_signals_test) * 100

                if len(filt_indices_test) > 0:
                    pnl_test, _ = _run_sim(filt_indices_test, ohlcv_test, s_cfg, mode)
                    if len(pnl_test) > 0:
                        filtered_test_pnl = float(np.mean(pnl_test))
                        filtered_test_wr  = float(np.mean(pnl_test > 0))
                        filtered_test_n   = len(pnl_test)
                    else:
                        filtered_test_pnl = 0.0
                        filtered_test_wr  = 0.0
                        filtered_test_n   = 0
                else:
                    filtered_test_pnl = 0.0
                    filtered_test_wr  = 0.0
                    filtered_test_n   = 0

        delta = filtered_test_pnl - baseline_test_pnl if best_vpct > 0 else 0.0
        filter_found = best_vpct > 0
        filter_helps = filter_found and delta > 0

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
            'best_vpct': best_vpct,
            'filtered_opt_pnl': best_opt_avg,
            'filtered_test_pnl': filtered_test_pnl,
            'filtered_test_wr': filtered_test_wr,
            'filtered_test_n': filtered_test_n,
            'opt_blocked_pct': opt_blocked_pct,
            'test_blocked_pct': test_blocked_pct,
            'delta': delta,
            'filter_helps': filter_helps,
            'filter_found': filter_found,
        }
        results.append(result)

        if filter_found:
            verdict = "BETTER" if filter_helps else "WORSE"
            print(f"    Baseline OPT: {baseline_opt_pnl:+.2%}  ({baseline_opt_n} trades)")
            print(f"    Baseline TEST: {baseline_test_pnl:+.2%}  ({baseline_test_n} trades)")
            print(f"    Filter: vol_filter_pct >= {best_vpct}%")
            print(f"    Filtered OPT:  {best_opt_avg:+.2%}  (blocked {opt_blocked_pct:.0f}% signals)")
            print(f"    Filtered TEST: {filtered_test_pnl:+.2%}  ({filtered_test_n} trades, blocked {test_blocked_pct:.0f}%)  [{verdict}]")
            print(f"    Delta: {delta:+.2%}")
        else:
            print(f"    Baseline OPT: {baseline_opt_pnl:+.2%}  ({baseline_opt_n} trades)")
            print(f"    Baseline TEST: {baseline_test_pnl:+.2%}  ({baseline_test_n} trades)")
            print(f"    No vol_filter_pct improved OPT avg PnL")

        del ohlcv_opt, ohlcv_test
        gc.collect()

    return results


# ==============================================================================
# PHASE 4: CHARTS + SUMMARY
# ==============================================================================

def print_summary_and_charts(results):
    print(f"\n\n{'='*W}")
    print(f"  VOLUME FILTER ANALYSIS — SUMMARY")
    print(f"{'='*W}")

    if not results:
        print("  No results to display.")
        return

    df = pd.DataFrame(results)

    # ── Summary table ──
    print(f"\n  {'Ticker':<7} {'Mode':<14} {'BL_OPT':>7} {'BL_TEST':>8} {'FILT_TEST':>9} "
          f"{'Delta':>7} {'VolPct':>6} {'Blk%':>5} {'BL_N':>5} {'FLT_N':>5} {'Verdict':>8}")
    print(f"  {'─'*90}")

    for _, r in df.sort_values('delta', ascending=False).iterrows():
        if r['filter_found']:
            verdict = "BETTER" if r['filter_helps'] else "WORSE"
            print(f"  {r['ticker']:<7} {r['mode']:<14} "
                  f"{r['baseline_opt_pnl']:>+6.2%} {r['baseline_test_pnl']:>+7.2%} "
                  f"{r['filtered_test_pnl']:>+8.2%} {r['delta']:>+6.2%} "
                  f"{r['best_vpct']:>5.0f}% {r['test_blocked_pct']:>4.0f}% "
                  f"{r['baseline_test_n']:>5.0f} {r['filtered_test_n']:>5.0f} {verdict:>8}")
        else:
            print(f"  {r['ticker']:<7} {r['mode']:<14} "
                  f"{r['baseline_opt_pnl']:>+6.2%} {r['baseline_test_pnl']:>+7.2%} "
                  f"{'N/A':>9} {'0.00%':>7} "
                  f"{'---':>6} {'0%':>5} "
                  f"{r['baseline_test_n']:>5.0f} {'---':>5} {'NO_FILT':>8}")

    # ── Aggregate stats ──
    n_total = len(df)
    n_filter_found = int(df['filter_found'].sum())
    n_helps = int(df['filter_helps'].sum())
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

    # Total PnL
    total_baseline = (df['baseline_test_pnl'] * df['baseline_test_n']).sum()
    total_baseline_n = int(df['baseline_test_n'].sum())
    selective_pnl = 0.0
    selective_n = 0
    for _, r in df.iterrows():
        if r['filter_helps']:
            selective_pnl += r['filtered_test_pnl'] * r['filtered_test_n']
            selective_n += int(r['filtered_test_n'])
        else:
            selective_pnl += r['baseline_test_pnl'] * r['baseline_test_n']
            selective_n += int(r['baseline_test_n'])

    print(f"\n  AGGREGATE:")
    print(f"    {n_total} tickers analyzed")
    print(f"    {n_filter_found} found a vol threshold that beat OPT baseline")
    print(f"    {n_helps} filter HELPS on TEST (delta > 0)")
    print(f"    {n_hurts} filter HURTS on TEST (delta <= 0)")
    print(f"    {n_no_filter} no filter improved OPT")
    print(f"    Avg baseline TEST PnL: {avg_baseline:+.2%}")
    print(f"    Avg filtered TEST PnL: {avg_filtered:+.2%}")
    print(f"    Avg delta: {avg_delta:+.2%}")
    print(f"\n  TOTAL PnL:")
    print(f"    Baseline:  {total_baseline:+.2%} across {total_baseline_n} trades")
    print(f"    Selective: {selective_pnl:+.2%} across {selective_n} trades")
    print(f"    Delta:     {selective_pnl - total_baseline:+.2%}")

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
    ax1.barh(y_pos, baseline_vals, height=0.4, align='edge', label='Baseline (no filter)', color='steelblue', alpha=0.7)
    ax1.barh([y + 0.4 for y in y_pos], filtered_vals, height=0.4, align='edge', label='Vol filtered', color='green', alpha=0.7)
    ax1.set_yticks([y + 0.4 for y in y_pos])
    ax1.set_yticklabels(df_sorted['ticker'].values, fontsize=8)
    ax1.set_xlabel('TEST Avg PnL (%)')
    ax1.set_title('Baseline vs Volume Filtered TEST PnL')
    ax1.axvline(0, color='black', linewidth=0.5)
    ax1.legend(fontsize=8)
    ax1.grid(axis='x', alpha=0.3)

    # Panel 2: Selected vol_filter_pct per ticker
    ax2 = axes[0, 1]
    df_with_filter = df[df['filter_found']].sort_values('ticker')
    if len(df_with_filter) > 0:
        x_pos = range(len(df_with_filter))
        colors = ['green' if d > 0 else 'red' for d in df_with_filter['delta']]
        ax2.bar(list(x_pos), df_with_filter['best_vpct'].values, color=colors, alpha=0.7,
                edgecolor='black', linewidth=0.5)
        ax2.set_xticks(list(x_pos))
        ax2.set_xticklabels(df_with_filter['ticker'].values, rotation=45, fontsize=8)
        ax2.set_ylabel('vol_filter_pct (%)')
        ax2.set_title('Selected Volume Threshold per Ticker\n(green=helps, red=hurts)')
        ax2.grid(axis='y', alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No filters found', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Selected Volume Threshold per Ticker')

    # Panel 3: Scatter — % signals blocked vs PnL delta
    ax3 = axes[1, 0]
    if len(df_with_filter) > 0:
        colors = ['green' if d > 0 else 'red' for d in df_with_filter['delta']]
        ax3.scatter(df_with_filter['test_blocked_pct'], df_with_filter['delta'] * 100,
                    c=colors, s=60, edgecolors='black', linewidth=0.5, alpha=0.8)
        for _, r in df_with_filter.iterrows():
            ax3.annotate(r['ticker'], (r['test_blocked_pct'], r['delta'] * 100),
                        fontsize=7, ha='left', va='bottom')
        ax3.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax3.set_xlabel('Signals Blocked (%)')
        ax3.set_ylabel('PnL Delta (%)')
        ax3.set_title('Volume Filter Impact: Blocked % vs PnL Change')
        ax3.grid(alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No filters found', ha='center', va='center', transform=ax3.transAxes)

    # Panel 4: Total PnL comparison
    ax4 = axes[1, 1]
    categories = ['Baseline\n(no filter)', 'Selective\n(filter where helps)']
    values = [total_baseline, selective_pnl]
    bar_colors = ['steelblue', 'green']
    bars = ax4.bar(categories, values, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax4.set_ylabel('Total TEST PnL (%)')
    ax4.set_title(f'Total PnL Comparison\nBaseline: {total_baseline_n} trades  |  Selective: {selective_n} trades')
    ax4.axhline(0, color='black', linewidth=0.5)
    ax4.grid(axis='y', alpha=0.3)
    for i, v in enumerate(values):
        ax4.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=11, fontweight='bold')

    plt.suptitle(f'Volume Filter Analysis — Spike Transformer LONG (Fold {FOLD_ID})\n'
                 f'OPT: {OPT_START} to {OPT_END}  |  TEST: {TEST_START} to {TEST_END}  |  '
                 f'{n_total} tickers, {len(VOL_PCT_GRID)} vol thresholds',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()

    chart_path = '/tmp/ra_moe_models/vol_filter_analysis_st_long.png'
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
        vpct = r['best_vpct'] if r['filter_helps'] else 0.0
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
            'vix_threshold': NO_FILTER_DEFAULTS['vix_threshold'],
            'tlt_crash': NO_FILTER_DEFAULTS['tlt_crash'],
            'spy_ignore': NO_FILTER_DEFAULTS['spy_ignore'],
            'min_vol_bars': 0,
            'vol_filter_pct': float(vpct),
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
                filter_tag = f"vol>={vpct:.0f}%" if r['filter_helps'] else "NO_FILTER"
                print(f"    BQ: {r['ticker']}/{mode}  {filter_tag}  TEST={test_pnl:+.2%}")
                n_written += 1
        except Exception as e:
            print(f"    BQ save failed {r['ticker']}: {e}")

    print(f"\n  Written {n_written}/{len(results)} rows to BQ as '{MODEL_VERSION_OUT}'")


# ==============================================================================
# RUN ALL PHASES
# ==============================================================================

# Phase 1: Score
all_daily_scores = run_scoring_phase()

# Phase 2: Load frozen configs
frozen_configs = load_frozen_configs()

if not frozen_configs:
    print("\nERROR: No frozen configs loaded. Cannot proceed.")
else:
    # Phase 3: Volume filter sweep
    results = run_volume_filter_analysis(all_daily_scores, frozen_configs)

    # Phase 4: Summary + charts
    summary_df = print_summary_and_charts(results)

    # Phase 5: Optional BQ write
    write_results_to_bq(results)

    # Grand total
    n_helps = sum(1 for r in results if r['filter_helps'])
    n_total = len(results)
    print(f"\n{'='*W}")
    print(f"  COMPLETE — {n_total} tickers, {n_helps} improved by volume filter")
    print(f"  model_version_in:  '{MODEL_VERSION_IN}'")
    print(f"  model_version_out: '{MODEL_VERSION_OUT}' (WRITE_TO_BQ={WRITE_TO_BQ})")
    print(f"{'='*W}")
