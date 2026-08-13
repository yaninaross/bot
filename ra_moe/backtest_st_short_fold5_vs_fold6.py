# %% ── SHORT Backtest: Fold 5 vs Fold 6 vs Fold 7 vs Hybrid ──────────────────
#
# Multi-fold comparison of the SAME L5-1 Spike Transformer architecture
# (sector-aware FiLM, 22 regime features, standard DeStationaryAttention)
# trained on THREE DIFFERENT FOLDS:
#
#   Fold 5 — train → Jun 9, val Jun 10 → Jul 9, test Jul 10 → Aug 10 (~4.4 weeks)
#            BQ model_version = 'spike_transformer_short_L5_1_fold5'
#
#   Fold 6 — train → Jun 23, val Jun 24 → Jul 21, test Jul 22 → Aug 10 (~2.7 weeks)
#            BQ model_version = 'spike_transformer_short_L5_1_fold6'
#
#   Fold 7 — train → Jun 28, val Jun 29 → Jul 28, test Jul 29 → Aug 12 (~2 weeks)
#            BQ model_version = 'spike_transformer_short_L5_1_fold7'
#
# CRITICAL: All models backtested on the OVERLAP window only:
#   Jul 29 → Aug 12 (F7's full test window — Aug 11-12 also unseen for F5/F6).
#
# HYBRID STRATEGY: F5 configs for F5 tickers + F6-only tickers (F6 model) +
#   F7-only tickers (F7 model). Cherry-picks best configs per ticker.
#
# PRORATED N: Trade count thresholds prorated by test window length:
#   Fold 5: 31 cal days test → val_b_count >= 2 (standard)
#   Fold 6: 19 cal days test → val_b_count >= 1
#   Fold 7: 14 cal days test → val_b_count >= 1
#
# Primary normalized metric: trades/week (over 14-day F7 test window)
#
# Same architecture for all — one set of model classes, clear_session() between.
#
# GPU recommended for scoring. Colab-ready.

import os, time, gc, ctypes, warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from google.cloud import storage, bigquery
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
SCORE_HORIZONS = [0, 1, 2]
HORIZON_LABELS = {0: '15m', 1: '30m', 2: '60m'}

# Comparison window = F7's full test window (Jul 29 → Aug 12)
# Aug 11-12 also unseen for F5/F6 (beyond their test end but never in train/val)
OVERLAP_START  = '2026-07-29'
OVERLAP_END    = '2026-08-12'

SCORE_MINUTE   = 375
ENTRY_MINUTE   = 385
EXIT_TIME_MIN  = 780
SLIPPAGE_PCT   = 0.0015

DEPLOY_PNL_THRESHOLD = 0.01

GCS_BUCKET    = 'time-series-bucket-ys'
GCS_INFER_DIR = 'Data/Inference_v2'

# Architecture (shared — same L5-1 for both folds)
D_MODEL = 64; N_HEADS = 4; D_FF = 128; N_TRANSFORMER_BLOCKS = 2
DROPOUT = 0.1; FILM_HIDDEN = 64; TIME_STEPS = 128
N_HORIZONS = 3

TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}
EXCLUDE_TICKERS = {'SPY', 'VIX', 'TLT', 'IWM', 'XLE', 'ITA', 'BITF', 'MIGI', 'SMLR'}
PST = pytz.timezone('America/Los_Angeles')

BQ_TABLE = "ys-time-series.scoring_logs.wf_optimization_results"
bq_client = bigquery.Client()

SEED = 42
np.random.seed(SEED)
W = 100

# -- Prorated trade counts --
F5_TEST_DAYS = 31   # Jul 10 → Aug 10
F6_TEST_DAYS = 19   # Jul 22 → Aug 10
F7_TEST_DAYS = 14   # Jul 29 → Aug 12
OVERLAP_DAYS = 14   # Jul 29 → Aug 12 (F7's full test window)
MIN_TRADES_PER_MONTH = 2.0

F5_MIN_COUNT = 2
F6_MIN_COUNT = max(1, round(MIN_TRADES_PER_MONTH * F6_TEST_DAYS / 30))
F7_MIN_COUNT = max(1, round(MIN_TRADES_PER_MONTH * F7_TEST_DAYS / 30))

# -- Fold 5 config --
F5_GCS_DATA   = 'Algos/RA_MoE/short_L5_1_data'
F5_GCS_MODELS = 'Algos/RA_MoE/short_L5_1_models'
F5_DATA_DIR   = '/tmp/ra_moe_short_L5_1_data'
F5_MODEL_DIR  = '/tmp/ra_moe_short_L5_1_models'
F5_MODEL_VERSION = 'spike_transformer_short_L5_1_fold5'
F5_FOLD_ID = 5

# -- Fold 6 config --
F6_GCS_DATA   = 'Algos/RA_MoE/short_L5_1_data'
F6_GCS_MODELS = 'Algos/RA_MoE/short_L5_1_models'
F6_DATA_DIR   = '/tmp/ra_moe_short_L5_1_data'
F6_MODEL_DIR  = '/tmp/ra_moe_short_L5_1_models'
F6_MODEL_VERSION = 'spike_transformer_short_L5_1_fold6'
F6_FOLD_ID = 6

# -- Fold 7 config --
F7_GCS_DATA   = 'Algos/RA_MoE/short_L5_1_data'
F7_GCS_MODELS = 'Algos/RA_MoE/short_L5_1_models'
F7_DATA_DIR   = '/tmp/ra_moe_short_L5_1_data'
F7_MODEL_DIR  = '/tmp/ra_moe_short_L5_1_models'
F7_MODEL_VERSION = 'spike_transformer_short_L5_1_fold7'
F7_FOLD_ID = 7

for d in [F5_DATA_DIR, F5_MODEL_DIR, F6_DATA_DIR, F6_MODEL_DIR,
          F7_DATA_DIR, F7_MODEL_DIR]:
    os.makedirs(d, exist_ok=True)


# ==============================================================================
# GCS HELPERS
# ==============================================================================
def _gcs_bucket():
    return storage.Client().bucket(GCS_BUCKET)

def download_from_gcs(gcs_path, local_path):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    _gcs_bucket().blob(gcs_path).download_to_filename(local_path)

def ensure_data(filename, data_dir, gcs_prefix):
    local_path = os.path.join(data_dir, filename)
    if os.path.exists(local_path):
        return local_path
    gcs_path = f'{gcs_prefix}/{filename}'
    print(f"    Downloading {filename} from {gcs_prefix}...")
    download_from_gcs(gcs_path, local_path)
    return local_path

def ensure_model(fold_id, model_dir, gcs_prefix):
    local_dir = os.path.join(model_dir, f'fold_{fold_id}')
    weights_path = os.path.join(local_dir, 'model.weights.h5')
    if os.path.exists(weights_path):
        return weights_path
    os.makedirs(local_dir, exist_ok=True)
    gcs_p = f'{gcs_prefix}/fold_{fold_id}'
    print(f"    Downloading fold {fold_id} model from {gcs_prefix}...")
    bucket = _gcs_bucket()
    blobs = list(bucket.list_blobs(prefix=gcs_p))
    for blob in blobs:
        fname = blob.name.split('/')[-1]
        blob.download_to_filename(os.path.join(local_dir, fname))
    return weights_path


# ==============================================================================
# MODEL ARCHITECTURE — L5-1 (standard DeStationaryAttention)
# Same architecture for both folds — only weights differ
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


def build_spike_transformer_L5_1(n_asset_features, n_regime_features,
                                  name='spike_transformer_L5_1'):
    """L5-1: Standard DeStationaryAttention (no temporal decay)."""
    class _ST(Model):
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

    model = _ST(name=name)
    dummy_seq = tf.zeros((1, TIME_STEPS, n_asset_features))
    dummy_reg = tf.zeros((1, n_regime_features))
    model({'seq_input': dummy_seq, 'regime_input': dummy_reg})
    return model


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
# OHLCV DATA LOADING
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
        'prices': df['Close'].values.copy(),
        'highs':  df['High'].values.copy(),
        'lows':   df['Low'].values.copy(),
        'dates':  dates,
        'times':  times,
    }


# ==============================================================================
# SCORING PHASE — generic for any fold
# ==============================================================================

def score_fold(fold_id, data_dir, gcs_data, model_dir, gcs_models,
               label, ticker_filter=None):
    print(f"\n{'='*W}")
    print(f"  SCORING -- {label} (fold {fold_id})")
    print(f"{'='*W}")

    meta_path = ensure_data('meta.npz', data_dir, gcs_data)
    meta = np.load(meta_path, allow_pickle=True)
    ticker_names = meta['ticker_names']
    print(f"  {len(ticker_names)} tickers in meta")

    probe_path = ensure_data(f'fold_{fold_id}_val.npz', data_dir, gcs_data)
    probe = np.load(probe_path)
    n_asset = probe['X_seq'].shape[-1]
    n_regime = probe['X_regime'].shape[-1]
    probe.close()
    print(f"  N_ASSET_FEATURES={n_asset}, N_REGIME_FEATURES={n_regime}")

    model = build_spike_transformer_L5_1(n_asset, n_regime)
    weights_path = ensure_model(fold_id, model_dir, gcs_models)
    model.load_weights(weights_path)
    print(f"  Model loaded: {model.count_params():,} params")

    all_scores = {}
    for split in ['val', 'test']:
        path = ensure_data(f'fold_{fold_id}_{split}.npz', data_dir, gcs_data)
        data = np.load(path)
        X_seq     = data['X_seq']
        X_reg     = data['X_regime']
        X_id      = data['X_id'].flatten()
        date_ints = data['date_ints']
        minutes   = data['minutes']
        data.close()

        entry_mask = (minutes == SCORE_MINUTE)
        X_seq     = X_seq[entry_mask]
        X_reg     = X_reg[entry_mask]
        X_id      = X_id[entry_mask]
        date_ints = date_ints[entry_mask]
        n = len(X_id)
        del minutes, entry_mask; gc.collect()

        print(f"  {split}: scoring {n:,} windows...")
        preds = model.predict(
            {'seq_input': X_seq.astype(np.float32),
             'regime_input': X_reg.astype(np.float32)},
            batch_size=512, verbose=0)
        del X_seq, X_reg; gc.collect()

        for i in range(n):
            tid = int(X_id[i])
            d_int = int(date_ints[i])
            if tid >= len(ticker_names):
                continue
            tname = str(ticker_names[tid])
            if tname in EXCLUDE_TICKERS:
                continue
            if ticker_filter and tname not in ticker_filter:
                continue
            if tname not in all_scores:
                all_scores[tname] = {h: {} for h in SCORE_HORIZONS}
            for h in SCORE_HORIZONS:
                all_scores[tname][h][d_int] = float(preds[i, h])
        del preds, X_id, date_ints; gc.collect()

    del model
    tf.keras.backend.clear_session()
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

    n_t = len(all_scores)
    n_s = sum(len(v) for h_scores in all_scores.values() for v in h_scores.values())
    print(f"  Done: {n_t} tickers, {n_s} daily scores")
    return all_scores


# ==============================================================================
# BQ CONFIG LOADER — with prorated min count
# ==============================================================================

def load_configs_from_bq(model_version, min_count, label):
    query = f"""
    SELECT ticker, prob_thresh, tp_pct, sl_pct, tpsl_mode,
           grace_minutes, dip_pct, max_wait_minutes,
           val_b_avg_pnl, val_b_win_rate, val_b_count,
           experiment
    FROM `{BQ_TABLE}`
    WHERE model_version = '{model_version}'
      AND val_b_avg_pnl >= {DEPLOY_PNL_THRESHOLD}
      AND val_b_count >= {min_count}
    ORDER BY val_b_avg_pnl DESC
    """
    print(f"\n  Querying BQ for {label} configs (model_version='{model_version}', "
          f"min_count={min_count})...")
    df = bq_client.query(query).to_dataframe()
    if len(df) == 0:
        print(f"  WARNING: No {label} configs found in BQ. Has the optimizer been run?")
        return {}

    configs = {}
    for _, row in df.iterrows():
        ticker = row['ticker']
        if ticker in configs:
            continue
        mode = row['tpsl_mode']
        exp = row.get('experiment', '')
        h_idx = 1
        for h_label_key, h_label_val in HORIZON_LABELS.items():
            if h_label_val in exp:
                h_idx = h_label_key
                break
        cfg = {
            'horizon': h_idx,
            'prob_thresh': float(row['prob_thresh']),
            'tp_pct': float(row['tp_pct']),
            'sl_pct': float(row['sl_pct']),
            'mode': mode,
            'val_b_avg_pnl': float(row['val_b_avg_pnl']),
            'val_b_win_rate': float(row['val_b_win_rate']),
            'val_b_count': int(row['val_b_count']),
        }
        if mode == 'grace' and pd.notna(row.get('grace_minutes')):
            cfg['grace_minutes'] = int(row['grace_minutes'])
        if mode == 'limit_entry':
            if pd.notna(row.get('dip_pct')):
                cfg['dip_pct'] = float(row['dip_pct'])
            if pd.notna(row.get('max_wait_minutes')):
                cfg['max_wait'] = int(row['max_wait_minutes'])
        configs[ticker] = cfg

    print(f"  {len(configs)} deployable {label} tickers "
          f"(val_b_avg_pnl >= {DEPLOY_PNL_THRESHOLD:.0%}, val_b_count >= {min_count})")
    for t, c in sorted(configs.items(), key=lambda x: -x[1]['val_b_avg_pnl']):
        h = HORIZON_LABELS[c['horizon']]
        print(f"    {t:<7} H={h} mode={c['mode']:<13} "
              f"val_b={c['val_b_avg_pnl']:+.2%} WR={c['val_b_win_rate']:.0%} N={c['val_b_count']}")
    return configs


# ==============================================================================
# RUN BACKTEST FOR ONE MODEL — on OVERLAP window only
# ==============================================================================

def run_backtest(all_scores, deploy_configs, label):
    tickers = sorted(deploy_configs.keys())
    if not tickers:
        print(f"\n  {label}: no tickers to backtest")
        return []

    print(f"\n{'─'*W}")
    print(f"  BACKTEST -- {label} ({len(tickers)} tickers)")
    print(f"  OVERLAP WINDOW: {OVERLAP_START} -> {OVERLAP_END}")
    print(f"{'─'*W}")

    all_trades = []
    for ticker in tickers:
        cfg = deploy_configs[ticker]
        h_idx = cfg['horizon']
        h_label = HORIZON_LABELS[h_idx]
        mode = cfg['mode']

        print(f"  {ticker} -- H={h_label} mode={mode} prob>{cfg['prob_thresh']:.2f} "
              f"TP={cfg['tp_pct']:.0%} SL={cfg['sl_pct']:.0%}", end="")
        if mode == 'grace':
            print(f" grace={cfg.get('grace_minutes', 0)}m", end="")
        elif mode == 'limit_entry':
            print(f" rally={cfg.get('dip_pct', 0):.1%} wait={cfg.get('max_wait', 0)}m", end="")
        print()

        scores = all_scores.get(ticker, {})
        h_scores = scores.get(h_idx, {})
        if not h_scores:
            print(f"    No scores")
            continue

        try:
            df_ohlcv = load_ticker_ohlcv(ticker)
        except Exception as e:
            print(f"    OHLCV failed: {e}")
            continue

        # Use OVERLAP window for backtesting — fair comparison zone
        ohlcv = precompute_ohlcv_arrays(df_ohlcv, OVERLAP_START, OVERLAP_END)
        if ohlcv is None:
            print(f"    No OHLCV in overlap window")
            del df_ohlcv; continue

        score_by_date = {int(d): s for d, s in h_scores.items()}
        entry_list = []
        for i in range(len(ohlcv['dates'])):
            if ohlcv['times'][i] == ENTRY_MINUTE:
                day_score = score_by_date.get(int(ohlcv['dates'][i]), 0.0)
                if day_score >= cfg['prob_thresh']:
                    entry_list.append(i)

        if not entry_list:
            print(f"    No signals in overlap window")
            del df_ohlcv; continue

        indices = np.array(entry_list, dtype=np.int64)
        pnl, dates, entry_p, exit_p, outcomes = _run_sim(indices, ohlcv, cfg)

        if len(pnl) == 0:
            print(f"    No trades")
            del df_ohlcv; continue

        avg_pnl = float(np.mean(pnl))
        wr = float(np.mean(pnl > 0))
        n = len(pnl)
        trades_per_week = n / (OVERLAP_DAYS / 7.0)
        print(f"    {n} trades ({trades_per_week:.1f}/wk) | Avg: {avg_pnl:+.2%} | "
              f"WR: {wr:.0%} | Total: {float(np.sum(pnl)):+.2%}")

        for i in range(n):
            d = int(dates[i])
            date_str = f"{d // 10000}-{(d % 10000) // 100:02d}-{d % 100:02d}"
            otype = 'TP' if outcomes[i] == 1 else 'SL' if outcomes[i] == -1 else 'EOD'
            all_trades.append({
                'Ticker': ticker, 'Date': date_str, 'Date_int': d,
                'Entry': float(entry_p[i]), 'Exit': float(exit_p[i]),
                'PnL': float(pnl[i]), 'Outcome': otype,
                'Mode': mode, 'Horizon': h_label, 'Strategy': label,
            })

        del df_ohlcv, ohlcv; gc.collect()

    return all_trades


# ==============================================================================
# COMPARISON ANALYSIS — with trades/week as primary normalized metric
# ==============================================================================

def analyze_strategy(trades, label):
    if not trades:
        return {'label': label, 'n': 0, 'avg_pnl': 0, 'total_pnl': 0,
                'win_rate': 0, 'n_tickers': 0, 'sharpe': 0,
                'trades_per_week': 0, 'n_days': 0,
                'avg_win': 0, 'avg_loss': 0}
    df = pd.DataFrame(trades)
    n = len(df)
    avg = df['PnL'].mean()
    std = df['PnL'].std()
    trades_per_week = n / (OVERLAP_DAYS / 7.0)
    return {
        'label': label,
        'n': n,
        'avg_pnl': avg,
        'total_pnl': df['PnL'].sum(),
        'win_rate': (df['PnL'] > 0).mean(),
        'n_tickers': df['Ticker'].nunique(),
        'n_days': df['Date'].nunique(),
        'trades_per_week': trades_per_week,
        'avg_win': df.loc[df['PnL'] > 0, 'PnL'].mean() if (df['PnL'] > 0).any() else 0,
        'avg_loss': df.loc[df['PnL'] <= 0, 'PnL'].mean() if (df['PnL'] <= 0).any() else 0,
        'sharpe': (avg / std * np.sqrt(252)) if std > 0 else 0,
    }


# ==============================================================================
# MAIN
# ==============================================================================

print(f"\n{'='*W}")
print(f"  SHORT BACKTEST: Fold 5 vs Fold 6 vs Fold 7 vs Hybrid (L5-1)")
print(f"  OVERLAP WINDOW: {OVERLAP_START} -> {OVERLAP_END} ({OVERLAP_DAYS} cal days)")
print(f"  Fold 5: train->Jun 9, test Jul 10->Aug 10 ({F5_TEST_DAYS}d) | min: {F5_MIN_COUNT}")
print(f"  Fold 6: train->Jun 23, test Jul 22->Aug 10 ({F6_TEST_DAYS}d) | min: {F6_MIN_COUNT}")
print(f"  Fold 7: train->Jun 28, test Jul 29->Aug 12 ({F7_TEST_DAYS}d) | min: {F7_MIN_COUNT}")
print(f"  Deploy threshold: >{DEPLOY_PNL_THRESHOLD:.0%} val_b avg PnL")
print(f"  Slippage: {SLIPPAGE_PCT:.2%} each way ({2*SLIPPAGE_PCT:.2%} RT)")
print(f"{'='*W}")


# -- Phase 1: Load configs from BQ for all folds (prorated min counts) --
f5_configs = load_configs_from_bq(F5_MODEL_VERSION, F5_MIN_COUNT, 'Fold 5')
f6_configs = load_configs_from_bq(F6_MODEL_VERSION, F6_MIN_COUNT, 'Fold 6')
f7_configs = load_configs_from_bq(F7_MODEL_VERSION, F7_MIN_COUNT, 'Fold 7')

# -- Phase 2: Score fold 5 model --
f5_tickers = set(f5_configs.keys()) if f5_configs else None
f5_scores = score_fold(F5_FOLD_ID, F5_DATA_DIR, F5_GCS_DATA,
                       F5_MODEL_DIR, F5_GCS_MODELS,
                       'L5-1 Fold 5', ticker_filter=f5_tickers)

# -- Phase 3: Clear session, then score fold 6 model --
tf.keras.backend.clear_session()
gc.collect()

f6_tickers = set(f6_configs.keys()) if f6_configs else None
f6_scores = score_fold(F6_FOLD_ID, F6_DATA_DIR, F6_GCS_DATA,
                       F6_MODEL_DIR, F6_GCS_MODELS,
                       'L5-1 Fold 6', ticker_filter=f6_tickers)

# -- Phase 3b: Clear session, then score fold 7 model --
tf.keras.backend.clear_session()
gc.collect()

f7_tickers = set(f7_configs.keys()) if f7_configs else None
f7_scores = score_fold(F7_FOLD_ID, F7_DATA_DIR, F7_GCS_DATA,
                       F7_MODEL_DIR, F7_GCS_MODELS,
                       'L5-1 Fold 7', ticker_filter=f7_tickers)

# -- Phase 4: Run backtests on OVERLAP window --
f5_trades = run_backtest(f5_scores, f5_configs, 'Fold 5') if f5_configs else []
f6_trades = run_backtest(f6_scores, f6_configs, 'Fold 6') if f6_configs else []
f7_trades = run_backtest(f7_scores, f7_configs, 'Fold 7') if f7_configs else []


# ==============================================================================
# Phase 5: Build HYBRID — F5 tickers + F6-only + F7-only
# ==============================================================================
f5_ticker_set = set(f5_configs.keys()) if f5_configs else set()
f6_ticker_set = set(f6_configs.keys()) if f6_configs else set()
f7_ticker_set = set(f7_configs.keys()) if f7_configs else set()
f6_only_tickers = f6_ticker_set - f5_ticker_set
f7_only_tickers = f7_ticker_set - f5_ticker_set - f6_ticker_set

print(f"\n{'='*W}")
print(f"  BUILDING HYBRID STRATEGY")
print(f"{'='*W}")
print(f"  Rule: Fold 5 configs for all F5 tickers ({len(f5_ticker_set)})")
print(f"        + Fold 6 configs for F6-only tickers ({len(f6_only_tickers)})")
print(f"        + Fold 7 configs for F7-only tickers ({len(f7_only_tickers)})")
print(f"  F6-only: {', '.join(sorted(f6_only_tickers)) or 'none'}")
print(f"  F7-only: {', '.join(sorted(f7_only_tickers)) or 'none'}")

hybrid_configs = dict(f5_configs) if f5_configs else {}
for t in f6_only_tickers:
    hybrid_configs[t] = f6_configs[t]
for t in f7_only_tickers:
    hybrid_configs[t] = f7_configs[t]

hybrid_scores = dict(f5_scores) if f5_scores else {}
for t in f6_only_tickers:
    if t in f6_scores:
        hybrid_scores[t] = f6_scores[t]
for t in f7_only_tickers:
    if t in f7_scores:
        hybrid_scores[t] = f7_scores[t]

print(f"  Hybrid total: {len(hybrid_configs)} tickers")

hybrid_trades = run_backtest(hybrid_scores, hybrid_configs, 'Hybrid') if hybrid_configs else []


# ==============================================================================
# Phase 5b: Build HYBRID v2 — F5 > F7 > F6 (F7 prioritized over F6)
# ==============================================================================
f7_not_f5 = f7_ticker_set - f5_ticker_set
f6_after_f7 = f6_ticker_set - f5_ticker_set - f7_ticker_set
f6f7_shared = (f6_ticker_set & f7_ticker_set) - f5_ticker_set

print(f"\n{'='*W}")
print(f"  BUILDING HYBRID v2 (F5 > F7 > F6)")
print(f"{'='*W}")
print(f"  Rule: F5 configs ({len(f5_ticker_set)}) + F7 non-F5 ({len(f7_not_f5)}) "
      f"+ F6-only ({len(f6_after_f7)})")
if f6f7_shared:
    print(f"  F6->F7 switches: {', '.join(sorted(f6f7_shared))}")
    for t in sorted(f6f7_shared):
        f6v = f6_configs[t]['val_b_avg_pnl'] if t in f6_configs else 0
        f7v = f7_configs[t]['val_b_avg_pnl'] if t in f7_configs else 0
        print(f"    {t}: F6 val_b={f6v:+.2%} -> F7 val_b={f7v:+.2%}")

hybv2_configs = dict(f5_configs) if f5_configs else {}
for t in f7_not_f5:
    hybv2_configs[t] = f7_configs[t]
for t in f6_after_f7:
    hybv2_configs[t] = f6_configs[t]

hybv2_scores = dict(f5_scores) if f5_scores else {}
for t in f7_not_f5:
    if t in f7_scores:
        hybv2_scores[t] = f7_scores[t]
for t in f6_after_f7:
    if t in f6_scores:
        hybv2_scores[t] = f6_scores[t]

print(f"  F5: {len(f5_ticker_set)} | F7 non-F5: {len(f7_not_f5)} | F6-only: {len(f6_after_f7)}")
print(f"  Hybrid v2 total: {len(hybv2_configs)} tickers")

hybv2_trades = run_backtest(hybv2_scores, hybv2_configs, 'Hyb-v2') if hybv2_configs else []


# ==============================================================================
# MULTI-FOLD COMPARISON: F5 vs F6 vs F7 vs Hybrid vs Hyb-v2
# ==============================================================================
f5_stats = analyze_strategy(f5_trades, 'Fold 5')
f6_stats = analyze_strategy(f6_trades, 'Fold 6')
f7_stats = analyze_strategy(f7_trades, 'Fold 7')
hy_stats = analyze_strategy(hybrid_trades, 'Hybrid')
hv2_stats = analyze_strategy(hybv2_trades, 'Hyb-v2')

all_stats = [f5_stats, f6_stats, f7_stats, hy_stats, hv2_stats]

print(f"\n\n{'='*W}")
print(f"  MULTI-FOLD COMPARISON -- F5 vs F6 vs F7 vs Hybrid vs Hyb-v2")
print(f"  OVERLAP WINDOW: {OVERLAP_START} -> {OVERLAP_END} ({OVERLAP_DAYS} cal days)")
print(f"  Hybrid   = F5 + F6-only + F7-only  (F5>F6>F7)")
print(f"  Hyb-v2   = F5 + F7 non-F5 + F6-only  (F5>F7>F6)")
print(f"{'='*W}\n")

header = f"  {'Metric':<18} {'F5':>8} {'F6':>8} {'F7':>8} {'Hybrid':>8} {'Hyb-v2':>8} {'Best':>6}"
print(header)
print(f"  {'─'*68}")

def _fmt_pct(v): return f"{v:+.2%}" if v != 0 else "---"
def _fmt_n(v): return f"{v}" if v != 0 else "---"

def _best_of(vals, labels, higher_better=True):
    valid = [(v, l) for v, l in zip(vals, labels) if v != 0]
    if not valid: return "---"
    if higher_better:
        return max(valid, key=lambda x: x[0])[1]
    return min(valid, key=lambda x: x[0])[1]

_labels = ['F5', 'F6', 'F7', 'Hyb', 'HV2']

metrics = [
    ('Tickers',
     [_fmt_n(s['n_tickers']) for s in all_stats],
     _best_of([s['n_tickers'] for s in all_stats], _labels)),
    ('Total trades',
     [_fmt_n(s['n']) for s in all_stats],
     _best_of([s['n'] for s in all_stats], _labels)),
    ('Trades/week',
     [f"{s['trades_per_week']:.1f}" for s in all_stats], ""),
    ('Avg PnL/trade',
     [_fmt_pct(s['avg_pnl']) for s in all_stats],
     _best_of([s['avg_pnl'] for s in all_stats], _labels)),
    ('Total PnL',
     [_fmt_pct(s['total_pnl']) for s in all_stats],
     _best_of([s['total_pnl'] for s in all_stats], _labels)),
    ('Win rate',
     [f"{s['win_rate']:.1%}" for s in all_stats],
     _best_of([s['win_rate'] for s in all_stats], _labels)),
    ('Avg win',
     [_fmt_pct(s['avg_win']) for s in all_stats], ""),
    ('Avg loss',
     [_fmt_pct(s['avg_loss']) for s in all_stats], ""),
    ('Sharpe (ann.)',
     [f"{s['sharpe']:.1f}" for s in all_stats],
     _best_of([s['sharpe'] for s in all_stats], _labels)),
]

for label, vals, best in metrics:
    print(f"  {label:<18} {vals[0]:>8} {vals[1]:>8} {vals[2]:>8} {vals[3]:>8} {vals[4]:>8} {best:>6}")


# -- Per-ticker breakdown (4-way) --
all_bt_tickers = sorted(set(
    [t['Ticker'] for t in f5_trades] + [t['Ticker'] for t in f6_trades]
    + [t['Ticker'] for t in f7_trades] + [t['Ticker'] for t in hybrid_trades]
    + [t['Ticker'] for t in hybv2_trades]
))

if all_bt_tickers:
    f5_df = pd.DataFrame(f5_trades) if f5_trades else pd.DataFrame(columns=['Ticker','PnL','Outcome'])
    f6_df = pd.DataFrame(f6_trades) if f6_trades else pd.DataFrame(columns=['Ticker','PnL','Outcome'])
    f7_df = pd.DataFrame(f7_trades) if f7_trades else pd.DataFrame(columns=['Ticker','PnL','Outcome'])
    hy_df = pd.DataFrame(hybrid_trades) if hybrid_trades else pd.DataFrame(columns=['Ticker','PnL','Outcome'])
    hv2_df = pd.DataFrame(hybv2_trades) if hybv2_trades else pd.DataFrame(columns=['Ticker','PnL','Outcome'])

    print(f"\n\n  PER-TICKER COMPARISON ({OVERLAP_START} -> {OVERLAP_END}):")
    print(f"  {'Ticker':<6} {'F5':>6} {'F6':>6} {'F7':>6} "
          f"{'Hyb':>6} {'HV2':>6} {'Hyb':>4} {'HV2':>4}")
    print(f"  {'':6} {'Avg':>6} {'Avg':>6} {'Avg':>6} "
          f"{'Avg':>6} {'Avg':>6} {'src':>4} {'src':>4}")
    print(f"  {'─'*52}")

    ticker_comparison = []
    for ticker in all_bt_tickers:
        t5 = f5_df[f5_df['Ticker'] == ticker] if len(f5_df) > 0 else pd.DataFrame()
        t6 = f6_df[f6_df['Ticker'] == ticker] if len(f6_df) > 0 else pd.DataFrame()
        t7 = f7_df[f7_df['Ticker'] == ticker] if len(f7_df) > 0 else pd.DataFrame()
        th = hy_df[hy_df['Ticker'] == ticker] if len(hy_df) > 0 else pd.DataFrame()
        tv = hv2_df[hv2_df['Ticker'] == ticker] if len(hv2_df) > 0 else pd.DataFrame()
        n5 = len(t5); avg5 = t5['PnL'].mean() if n5 > 0 else 0
        n6 = len(t6); avg6 = t6['PnL'].mean() if n6 > 0 else 0
        n7 = len(t7); avg7 = t7['PnL'].mean() if n7 > 0 else 0
        nh = len(th); avgh = th['PnL'].mean() if nh > 0 else 0
        nv = len(tv); avgv = tv['PnL'].mean() if nv > 0 else 0

        if ticker in f5_ticker_set:
            hyb_src = "F5"
        elif ticker in f6_only_tickers:
            hyb_src = "F6"
        elif ticker in f7_only_tickers:
            hyb_src = "F7"
        else:
            hyb_src = "---"

        if ticker in f5_ticker_set:
            hv2_src = "F5"
        elif ticker in f7_not_f5:
            hv2_src = "F7"
        elif ticker in f6_after_f7:
            hv2_src = "F6"
        else:
            hv2_src = "---"

        changed = " *" if hyb_src != hv2_src else ""
        print(f"  {ticker:<6} {avg5:>+5.1%} {avg6:>+5.1%} {avg7:>+5.1%} "
              f"{avgh:>+5.1%} {avgv:>+5.1%} {hyb_src:>4} {hv2_src:>4}{changed}")
        ticker_comparison.append({
            'Ticker': ticker, 'F5_n': n5, 'F5_avg': avg5,
            'F6_n': n6, 'F6_avg': avg6,
            'F7_n': n7, 'F7_avg': avg7,
            'Hy_n': nh, 'Hy_avg': avgh,
            'HV2_n': nv, 'HV2_avg': avgv,
            'hyb_src': hyb_src, 'hv2_src': hv2_src,
        })

    print(f"\n  HYBRID (F5>F6>F7):")
    print(f"    F5: {len(f5_ticker_set)} | F6-only: {len(f6_only_tickers)} "
          f"({', '.join(sorted(f6_only_tickers)) or 'none'}) | "
          f"F7-only: {len(f7_only_tickers)} ({', '.join(sorted(f7_only_tickers)) or 'none'})")
    print(f"  HYBRID v2 (F5>F7>F6):")
    print(f"    F5: {len(f5_ticker_set)} | F7 non-F5: {len(f7_not_f5)} "
          f"({', '.join(sorted(f7_not_f5)) or 'none'}) | "
          f"F6-only: {len(f6_after_f7)} ({', '.join(sorted(f6_after_f7)) or 'none'})")
    if f6f7_shared:
        print(f"  SWITCHES (F6->F7): {', '.join(sorted(f6f7_shared))} (* in table above)")


# ==============================================================================
# CHARTS — 6-panel, 4-way
# ==============================================================================

COLORS = {'Fold 5': '#1565C0', 'Fold 6': '#E65100', 'Fold 7': '#7B1FA2',
          'Hybrid': '#2E7D32', 'Hyb-v2': '#00838F'}
ALL_STRATS = [('Fold 5', f5_trades), ('Fold 6', f6_trades),
              ('Fold 7', f7_trades), ('Hybrid', hybrid_trades),
              ('Hyb-v2', hybv2_trades)]

fig, axes = plt.subplots(2, 3, figsize=(24, 13))
fig.suptitle(f'SHORT Backtest: F5 vs F6 vs F7 vs Hybrid vs Hyb-v2 (L5-1)\n'
             f'{OVERLAP_START} -> {OVERLAP_END} ({OVERLAP_DAYS}d) | '
             f'Hyb=F5>F6>F7 | HV2=F5>F7>F6',
             fontsize=13, fontweight='bold')

# [0,0] Cumulative PnL
ax = axes[0, 0]
for label, trades in ALL_STRATS:
    if not trades: continue
    df = pd.DataFrame(trades).sort_values('Date')
    cum = df['PnL'].cumsum().values
    ax.plot(range(1, len(cum)+1), cum * 100, color=COLORS[label], linewidth=2, label=label)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_xlabel('Trade #'); ax.set_ylabel('Cumulative PnL (%)')
ax.set_title('Cumulative PnL'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

# [0,1] Daily PnL comparison
ax = axes[0, 1]
offsets = {'Fold 5': -0.4, 'Fold 6': -0.2, 'Fold 7': 0.0, 'Hybrid': 0.2, 'Hyb-v2': 0.4}
for label, trades in ALL_STRATS:
    if not trades: continue
    df = pd.DataFrame(trades)
    daily = df.groupby('Date')['PnL'].sum()
    dates = pd.to_datetime(daily.index)
    ax.bar(dates + timedelta(hours=int(offsets[label] * 24)), daily.values * 100,
           width=0.2, color=COLORS[label], alpha=0.8, label=label,
           edgecolor='white', linewidth=0.5)
ax.axhline(0, color='black', linewidth=0.8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
ax.set_ylabel('Daily Total PnL (%)'); ax.set_title('Daily PnL')
ax.legend(fontsize=7); ax.grid(axis='y', alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# [0,2] Per-ticker Hyb-v2 avg PnL (colored by source)
ax = axes[0, 2]
if all_bt_tickers and ticker_comparison:
    tc_df = pd.DataFrame(ticker_comparison).sort_values('HV2_avg', ascending=True)
    y_pos = np.arange(len(tc_df))
    src_colors = [COLORS.get(f'Fold {s}', '#888888') if s in ('5','6','7')
                  else '#888888' for s in tc_df['hv2_src'].str.replace('F','')]
    ax.barh(y_pos, tc_df['HV2_avg'].values * 100, color=src_colors, alpha=0.8,
            edgecolor='white', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{t} ({s})" for t, s in
                        zip(tc_df['Ticker'].values, tc_df['hv2_src'].values)], fontsize=7)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Avg PnL per Trade (%)')
    ax.set_title('Hyb-v2: Per-Ticker (F5>F7>F6)')
    ax.grid(axis='x', alpha=0.3)

# [1,0] Hyb-v2 source breakdown
ax = axes[1, 0]
if hybv2_trades:
    hv2_df_chart = pd.DataFrame(hybv2_trades)
    parts = [
        ('F5 tickers', hv2_df_chart[hv2_df_chart['Ticker'].isin(f5_ticker_set)], COLORS['Fold 5']),
        ('F7 non-F5', hv2_df_chart[hv2_df_chart['Ticker'].isin(f7_not_f5)], COLORS['Fold 7']),
        ('F6-only', hv2_df_chart[hv2_df_chart['Ticker'].isin(f6_after_f7)], COLORS['Fold 6']),
    ]
    x_pos = np.arange(len(parts))
    for i, (lbl, part_df, clr) in enumerate(parts):
        avg_v = part_df['PnL'].mean() * 100 if len(part_df) > 0 else 0
        total_v = part_df['PnL'].sum() * 100 if len(part_df) > 0 else 0
        bar = ax.bar(i, avg_v, color=clr, alpha=0.8, edgecolor='white', linewidth=0.5)
        ax.text(i, max(avg_v, 0) + 0.1, f"N={len(part_df)}\nTot={total_v:+.0f}%",
                ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([p[0] for p in parts], fontsize=9)
    ax.set_ylabel('Avg PnL/trade (%)')
    ax.set_title('Hyb-v2: Contribution by Source (F5>F7>F6)')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.grid(axis='y', alpha=0.3)

# [1,1] Portfolio summary table (4-way)
ax = axes[1, 1]
ax.axis('off')
table_data = [
    ['Metric', 'F5', 'F6', 'F7', 'Hybrid', 'HV2'],
    ['Tickers'] + [str(s['n_tickers']) for s in all_stats],
    ['Trades'] + [str(s['n']) for s in all_stats],
    ['Trades/wk'] + [f"{s['trades_per_week']:.1f}" for s in all_stats],
    ['Avg PnL'] + [f"{s['avg_pnl']:+.2%}" for s in all_stats],
    ['Total PnL'] + [f"{s['total_pnl']:+.2%}" for s in all_stats],
    ['Win rate'] + [f"{s['win_rate']:.1%}" for s in all_stats],
    ['Sharpe'] + [f"{s['sharpe']:.1f}" for s in all_stats],
]
table = ax.table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(7)
table.scale(1, 1.6)
col_colors = {1: '#E3F2FD', 2: '#FFF3E0', 3: '#F3E5F5', 4: '#E8F5E9', 5: '#E0F2F1'}
for (r, c), cell in table.get_celld().items():
    if r == 0:
        cell.set_facecolor('#BBDEFB')
        cell.set_text_props(fontweight='bold')
    elif c in col_colors:
        cell.set_facecolor(col_colors[c])
ax.set_title('Portfolio Summary', pad=20)

# [1,2] Outcome distribution (4-way)
ax = axes[1, 2]
outcomes_data = {}
for label, trades in ALL_STRATS:
    if not trades:
        outcomes_data[label] = {'TP': 0, 'SL': 0, 'EOD': 0}
        continue
    df = pd.DataFrame(trades)
    vc = df['Outcome'].value_counts()
    outcomes_data[label] = {k: vc.get(k, 0) for k in ['TP', 'SL', 'EOD']}

x = np.arange(3)
bar_w = 0.15
for i, (label, _) in enumerate(ALL_STRATS):
    vals = [outcomes_data[label].get(k, 0) for k in ['TP', 'SL', 'EOD']]
    ax.bar(x + i * bar_w, vals, bar_w, color=COLORS[label], alpha=0.8, label=label,
           edgecolor='white', linewidth=0.5)
ax.set_xticks(x + 2.0 * bar_w)
ax.set_xticklabels(['TP', 'SL', 'EOD'])
ax.set_ylabel('Count'); ax.set_title('Outcome Distribution')
ax.legend(fontsize=7); ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
chart_path = '/tmp/backtest_fold567_hybrid_short.png'
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
print(f"\n  Chart saved: {chart_path}")
if _IN_COLAB:
    plt.show()
plt.close()


# ==============================================================================
# VERDICT
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  VERDICT — F5 vs F6 vs F7 vs Hybrid vs Hyb-v2")
print(f"{'='*W}")

strats_with_trades = [(n, s) for n, s in
    [('Fold 5', f5_stats), ('Fold 6', f6_stats), ('Fold 7', f7_stats),
     ('Hybrid', hy_stats), ('Hyb-v2', hv2_stats)]
    if s['n'] > 0]
strats_missing = [(n, s) for n, s in
    [('Fold 5', f5_stats), ('Fold 6', f6_stats), ('Fold 7', f7_stats)]
    if s['n'] == 0]

if strats_missing:
    for name, _ in strats_missing:
        mv = {'Fold 5': F5_MODEL_VERSION, 'Fold 6': F6_MODEL_VERSION,
              'Fold 7': F7_MODEL_VERSION}.get(name, '?')
        print(f"  {name}: no trades — optimizer may not have run (model_version='{mv}')")

if len(strats_with_trades) >= 2:
    print(f"\n  Window: {OVERLAP_START} -> {OVERLAP_END} ({OVERLAP_DAYS} cal days, "
          f"{OVERLAP_DAYS/7.0:.1f} weeks)")
    print(f"  Fold 5: train->Jun 9 ({F5_TEST_DAYS}d test)")
    print(f"  Fold 6: train->Jun 23 ({F6_TEST_DAYS}d test)")
    print(f"  Fold 7: train->Jun 28 ({F7_TEST_DAYS}d test)")
    print(f"  Hybrid:  F5({len(f5_ticker_set)}) + F6-only({len(f6_only_tickers)}) + "
          f"F7-only({len(f7_only_tickers)})  [F5>F6>F7]")
    print(f"  Hyb-v2:  F5({len(f5_ticker_set)}) + F7 non-F5({len(f7_not_f5)}) + "
          f"F6-only({len(f6_after_f7)})  [F5>F7>F6]")
    if f6f7_shared:
        print(f"  Switches: {', '.join(sorted(f6f7_shared))} moved F6->F7 in v2")
    print()

    ranked = sorted(strats_with_trades, key=lambda x: x[1]['avg_pnl'], reverse=True)
    print(f"  RANKING BY AVG PNL/TRADE:")
    for i, (name, s) in enumerate(ranked, 1):
        print(f"    #{i} {name:<10}: {s['avg_pnl']:+.2%}/trade | {s['n']} trades "
              f"({s['trades_per_week']:.1f}/wk) | {s['n_tickers']} tickers | "
              f"Total {s['total_pnl']:+.2%} | WR {s['win_rate']:.0%} | Sharpe {s['sharpe']:.1f}")

    ranked_total = sorted(strats_with_trades, key=lambda x: x[1]['total_pnl'], reverse=True)
    print(f"\n  RANKING BY TOTAL PNL:")
    for i, (name, s) in enumerate(ranked_total, 1):
        print(f"    #{i} {name:<10}: Total {s['total_pnl']:+.2%} | "
              f"{s['avg_pnl']:+.2%}/trade x {s['n']} trades | {s['n_tickers']} tickers")

    winner_pnl = ranked[0][0]
    winner_total = ranked_total[0][0]

    print(f"\n  ANALYSIS:")
    print(f"  -> Best per-trade: {winner_pnl} ({ranked[0][1]['avg_pnl']:+.2%})")
    print(f"  -> Best total PnL: {winner_total} ({ranked_total[0][1]['total_pnl']:+.2%})")

    # Hybrid decomposition
    if hy_stats['n'] > 0:
        hy_f5_trades = [t for t in hybrid_trades if t['Ticker'] in f5_ticker_set]
        hy_f6_trades = [t for t in hybrid_trades if t['Ticker'] in f6_only_tickers]
        hy_f7_trades = [t for t in hybrid_trades if t['Ticker'] in f7_only_tickers]
        print(f"\n  HYBRID DECOMPOSITION (F5>F6>F7):")
        for src_label, src_trades in [('F5-sourced', hy_f5_trades),
                                       ('F6-sourced', hy_f6_trades),
                                       ('F7-sourced', hy_f7_trades)]:
            if src_trades:
                avg_src = np.mean([t['PnL'] for t in src_trades])
                n_tickers_src = len(set(t['Ticker'] for t in src_trades))
                print(f"    {src_label}: {n_tickers_src} tickers, {len(src_trades)} trades, "
                      f"avg {avg_src:+.2%}/trade")

    # Hyb-v2 decomposition
    if hv2_stats['n'] > 0:
        hv2_f5_trades = [t for t in hybv2_trades if t['Ticker'] in f5_ticker_set]
        hv2_f7_trades = [t for t in hybv2_trades if t['Ticker'] in f7_not_f5]
        hv2_f6_trades = [t for t in hybv2_trades if t['Ticker'] in f6_after_f7]
        print(f"\n  HYB-V2 DECOMPOSITION (F5>F7>F6):")
        for src_label, src_trades in [('F5-sourced', hv2_f5_trades),
                                       ('F7-sourced', hv2_f7_trades),
                                       ('F6-sourced', hv2_f6_trades)]:
            if src_trades:
                avg_src = np.mean([t['PnL'] for t in src_trades])
                n_tickers_src = len(set(t['Ticker'] for t in src_trades))
                print(f"    {src_label}: {n_tickers_src} tickers, {len(src_trades)} trades, "
                      f"avg {avg_src:+.2%}/trade")

    # Head-to-head: Hybrid vs Hyb-v2
    if hy_stats['n'] > 0 and hv2_stats['n'] > 0:
        print(f"\n  HYBRID vs HYB-V2 HEAD-TO-HEAD:")
        delta_avg = hv2_stats['avg_pnl'] - hy_stats['avg_pnl']
        delta_total = hv2_stats['total_pnl'] - hy_stats['total_pnl']
        delta_wr = hv2_stats['win_rate'] - hy_stats['win_rate']
        print(f"    Avg PnL:   Hybrid {hy_stats['avg_pnl']:+.2%} vs Hyb-v2 {hv2_stats['avg_pnl']:+.2%}  "
              f"(delta {delta_avg:+.2%})")
        print(f"    Total PnL: Hybrid {hy_stats['total_pnl']:+.2%} vs Hyb-v2 {hv2_stats['total_pnl']:+.2%}  "
              f"(delta {delta_total:+.2%})")
        print(f"    Win rate:  Hybrid {hy_stats['win_rate']:.1%} vs Hyb-v2 {hv2_stats['win_rate']:.1%}  "
              f"(delta {delta_wr:+.1%})")
        if f6f7_shared:
            print(f"\n    SWITCHED TICKERS (F6->F7 in v2):")
            hy_df_h2h = pd.DataFrame(hybrid_trades)
            hv2_df_h2h = pd.DataFrame(hybv2_trades)
            for t in sorted(f6f7_shared):
                th = hy_df_h2h[hy_df_h2h['Ticker'] == t] if len(hy_df_h2h) > 0 else pd.DataFrame()
                tv = hv2_df_h2h[hv2_df_h2h['Ticker'] == t] if len(hv2_df_h2h) > 0 else pd.DataFrame()
                nh = len(th); avgh = th['PnL'].mean() if nh > 0 else 0
                nv = len(tv); avgv = tv['PnL'].mean() if nv > 0 else 0
                delta = avgv - avgh
                winner = "F7 wins" if delta > 0 else "F6 wins" if delta < 0 else "tie"
                print(f"      {t}: Hyb(F6) {nh}t {avgh:+.2%} vs HV2(F7) {nv}t {avgv:+.2%}  "
                      f"delta {delta:+.2%}  [{winner}]")

    best_name = winner_pnl if winner_pnl == winner_total else winner_total
    if winner_pnl != winner_total:
        print(f"\n  TRADEOFF: {winner_pnl} wins per-trade, {winner_total} wins total PnL")

    print(f"\n  RECOMMENDATION: Deploy {best_name}")

    print(f"\n  TRADE FREQUENCY:")
    for name, s, mc, td in [('Fold 5', f5_stats, F5_MIN_COUNT, F5_TEST_DAYS),
                             ('Fold 6', f6_stats, F6_MIN_COUNT, F6_TEST_DAYS),
                             ('Fold 7', f7_stats, F7_MIN_COUNT, F7_TEST_DAYS)]:
        if s['n'] > 0:
            print(f"    {name}:  {s['trades_per_week']:.1f}/wk "
                  f"(BQ min: {mc}, {td}d test)")
    if hy_stats['n'] > 0:
        print(f"    Hybrid: {hy_stats['trades_per_week']:.1f}/wk")
    if hv2_stats['n'] > 0:
        print(f"    Hyb-v2: {hv2_stats['trades_per_week']:.1f}/wk")

elif len(strats_with_trades) == 1:
    name, s = strats_with_trades[0]
    print(f"\n  Only {name} has trades: {s['avg_pnl']:+.2%}/trade, {s['n']} trades")
    print(f"  Run remaining pipelines to enable comparison.")
else:
    print(f"\n  No folds have trades. Run the pipelines first.")

print(f"\n{'='*W}")
