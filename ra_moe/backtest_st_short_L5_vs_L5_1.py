# %% ── SHORT Backtest: L5 (baseline) vs L5-1 (sector-aware FiLM) ───────────────
#
# Head-to-head comparison of two SHORT Spike Transformer models on the
# SAME unseen TEST window (Jul 10 - Aug 10, 2026):
#
#   L5  — Baseline fold 5 (16 regime features)
#         GCS: short_data/ + short_models/
#         Deploy threshold: >1% TEST avg PnL
#         7 deployed tickers (ALAB, BE, PL, HUT, CIFR, BBAI, MARA)
#
#   L5-1 — Sector-aware FiLM (22 regime features, +6 sector features)
#          GCS: short_L5_1_data/ + short_L5_1_models/
#          Deploy configs from BQ: model_version = 'spike_transformer_short_L5_1_fold5'
#          Threshold: >1% TEST avg PnL (same as L5 for fair comparison)
#
# Both models share the same architecture (Non-Stationary Transformer + FiLM +
# AttentionPooling) but L5-1 has 6 extra sector-health regime features
# feeding into the FiLM conditioner.
#
# Comparison axes:
#   - Portfolio-level: avg PnL, total PnL, win rate, Sharpe proxy, trade count
#   - Per-ticker: which model wins per ticker, new tickers added/lost
#   - Overlap: tickers in both → same-ticker comparison
#   - Signal quality: score distributions, calibration
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
FOLD_ID        = 5
SCORE_HORIZONS = [0, 1, 2]
HORIZON_LABELS = {0: '15m', 1: '30m', 2: '60m'}

TEST_START     = '2026-07-10'
TEST_END       = '2026-08-10'

SCORE_MINUTE   = 375
ENTRY_MINUTE   = 385
EXIT_TIME_MIN  = 780
SLIPPAGE_PCT   = 0.0015

DEPLOY_PNL_THRESHOLD = 0.01

GCS_BUCKET    = os.environ.get("GCS_BUCKET", "")
GCS_INFER_DIR = 'Data/Inference_v2'

# Architecture (identical for both)
D_MODEL = 64; N_HEADS = 4; D_FF = 128; N_TRANSFORMER_BLOCKS = 2
DROPOUT = 0.1; FILM_HIDDEN = 64; TIME_STEPS = 128
N_HORIZONS = 3
ASL_GAMMA_NEG = 6.0; ASL_GAMMA_POS = 0.1; ASL_CLIP = 0.08

TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}
EXCLUDE_TICKERS = {'SPY', 'VIX', 'TLT', 'IWM', 'XLE', 'ITA', 'BITF', 'MIGI', 'SMLR'}
PST = pytz.timezone('America/Los_Angeles')

BQ_TABLE = f"{GCP_PROJECT}.scoring_logs.wf_optimization_results"
bq_client = bigquery.Client()

SEED = 42
np.random.seed(SEED)
W = 100

# ── L5 (baseline) config ──
L5_GCS_DATA   = 'Algos/RA_MoE/short_data'
L5_GCS_MODELS = 'Algos/RA_MoE/short_models'
L5_DATA_DIR   = '/tmp/ra_moe_short_data'
L5_MODEL_DIR  = '/tmp/ra_moe_short_models'
L5_MV         = 'spike_transformer_short_fold5'

L5_DEPLOY_CONFIGS = {
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

# ── L5-1 (sector-aware) config ──
L51_GCS_DATA   = 'Algos/RA_MoE/short_L5_1_data'
L51_GCS_MODELS = 'Algos/RA_MoE/short_L5_1_models'
L51_DATA_DIR   = '/tmp/ra_moe_short_L5_1_data'
L51_MODEL_DIR  = '/tmp/ra_moe_short_L5_1_models'
L51_MV         = 'spike_transformer_short_L5_1_fold5'

for d in [L5_DATA_DIR, L5_MODEL_DIR, L51_DATA_DIR, L51_MODEL_DIR]:
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
# MODEL ARCHITECTURE (shared — same class for both L5 and L5-1)
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


def build_spike_transformer(n_asset_features, n_regime_features, name='spike_transformer'):
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
# SCORING PHASE
# ==============================================================================

def score_model(data_dir, gcs_data, model_dir, gcs_models, label,
                ticker_filter=None):
    print(f"\n{'='*W}")
    print(f"  SCORING — {label}")
    print(f"{'='*W}")

    meta_path = ensure_data('meta.npz', data_dir, gcs_data)
    meta = np.load(meta_path, allow_pickle=True)
    ticker_names = meta['ticker_names']
    print(f"  {len(ticker_names)} tickers in meta")

    probe_path = ensure_data(f'fold_{FOLD_ID}_val.npz', data_dir, gcs_data)
    probe = np.load(probe_path)
    n_asset = probe['X_seq'].shape[-1]
    n_regime = probe['X_regime'].shape[-1]
    probe.close()
    print(f"  N_ASSET_FEATURES={n_asset}, N_REGIME_FEATURES={n_regime}")

    model = build_spike_transformer(n_asset, n_regime, name='spike_transformer')
    weights_path = ensure_model(FOLD_ID, model_dir, gcs_models)
    model.load_weights(weights_path)
    print(f"  Model loaded: {model.count_params():,} params")

    all_scores = {}
    for split in ['val', 'test']:
        path = ensure_data(f'fold_{FOLD_ID}_{split}.npz', data_dir, gcs_data)
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
# BQ CONFIG LOADER (for L5-1)
# ==============================================================================

def load_l51_configs_from_bq():
    query = f"""
    SELECT ticker, prob_thresh, tp_pct, sl_pct, tpsl_mode,
           grace_minutes, dip_pct, max_wait_minutes,
           val_b_avg_pnl, val_b_win_rate, val_b_count,
           experiment
    FROM `{BQ_TABLE}`
    WHERE model_version = '{L51_MV}'
      AND val_b_avg_pnl >= {DEPLOY_PNL_THRESHOLD}
      AND val_b_count >= 2
    ORDER BY val_b_avg_pnl DESC
    """
    print(f"\n  Querying BQ for L5-1 configs (model_version='{L51_MV}')...")
    df = bq_client.query(query).to_dataframe()
    if len(df) == 0:
        print("  WARNING: No L5-1 configs found in BQ. Has the optimizer been run?")
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

    print(f"  {len(configs)} deployable L5-1 tickers (TEST avg PnL > {DEPLOY_PNL_THRESHOLD:.0%})")
    for t, c in sorted(configs.items(), key=lambda x: -x[1]['val_b_avg_pnl']):
        h = HORIZON_LABELS[c['horizon']]
        print(f"    {t:<7} H={h} mode={c['mode']:<13} "
              f"TEST={c['val_b_avg_pnl']:+.2%} WR={c['val_b_win_rate']:.0%} N={c['val_b_count']}")
    return configs


# ==============================================================================
# RUN BACKTEST FOR ONE MODEL
# ==============================================================================

def run_backtest(all_scores, deploy_configs, label):
    tickers = sorted(deploy_configs.keys())
    if not tickers:
        print(f"\n  {label}: no tickers to backtest")
        return []

    print(f"\n{'─'*W}")
    print(f"  BACKTEST — {label} ({len(tickers)} tickers)")
    print(f"{'─'*W}")

    all_trades = []
    for ticker in tickers:
        cfg = deploy_configs[ticker]
        h_idx = cfg['horizon']
        h_label = HORIZON_LABELS[h_idx]
        mode = cfg['mode']

        print(f"  {ticker} — H={h_label} mode={mode} prob>{cfg['prob_thresh']:.2f} "
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

        ohlcv = precompute_ohlcv_arrays(df_ohlcv, TEST_START, TEST_END)
        if ohlcv is None:
            print(f"    No OHLCV in test window")
            del df_ohlcv; continue

        score_by_date = {int(d): s for d, s in h_scores.items()}
        entry_list = []
        for i in range(len(ohlcv['dates'])):
            if ohlcv['times'][i] == ENTRY_MINUTE:
                day_score = score_by_date.get(int(ohlcv['dates'][i]), 0.0)
                if day_score >= cfg['prob_thresh']:
                    entry_list.append(i)

        if not entry_list:
            print(f"    No signals")
            del df_ohlcv; continue

        indices = np.array(entry_list, dtype=np.int64)
        pnl, dates, entry_p, exit_p, outcomes = _run_sim(indices, ohlcv, cfg)

        if len(pnl) == 0:
            print(f"    No trades")
            del df_ohlcv; continue

        avg_pnl = float(np.mean(pnl))
        wr = float(np.mean(pnl > 0))
        n = len(pnl)
        print(f"    {n} trades | Avg: {avg_pnl:+.2%} | WR: {wr:.0%} | Total: {float(np.sum(pnl)):+.2%}")

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
# COMPARISON ANALYSIS
# ==============================================================================

def analyze_strategy(trades, label):
    if not trades:
        return {'label': label, 'n': 0, 'avg_pnl': 0, 'total_pnl': 0,
                'win_rate': 0, 'n_tickers': 0, 'sharpe': 0}
    df = pd.DataFrame(trades)
    n = len(df)
    avg = df['PnL'].mean()
    std = df['PnL'].std()
    return {
        'label': label,
        'n': n,
        'avg_pnl': avg,
        'total_pnl': df['PnL'].sum(),
        'win_rate': (df['PnL'] > 0).mean(),
        'n_tickers': df['Ticker'].nunique(),
        'n_days': df['Date'].nunique(),
        'avg_win': df.loc[df['PnL'] > 0, 'PnL'].mean() if (df['PnL'] > 0).any() else 0,
        'avg_loss': df.loc[df['PnL'] <= 0, 'PnL'].mean() if (df['PnL'] <= 0).any() else 0,
        'sharpe': (avg / std * np.sqrt(252)) if std > 0 else 0,
    }


# ==============================================================================
# MAIN
# ==============================================================================

print(f"\n{'='*W}")
print(f"  SHORT BACKTEST: L5 (baseline) vs L5-1 (sector-aware FiLM)")
print(f"  TEST: {TEST_START} -> {TEST_END}")
print(f"  Deploy threshold: >{DEPLOY_PNL_THRESHOLD:.0%} TEST avg PnL")
print(f"  Slippage: {SLIPPAGE_PCT:.2%} each way ({2*SLIPPAGE_PCT:.2%} RT)")
print(f"{'='*W}")


# ── Phase 1: Load L5-1 configs from BQ ──
l51_configs = load_l51_configs_from_bq()

# ── Phase 2: Score L5 model ──
l5_tickers = set(L5_DEPLOY_CONFIGS.keys())
l5_scores = score_model(
    L5_DATA_DIR, L5_GCS_DATA, L5_MODEL_DIR, L5_GCS_MODELS,
    label='L5 (baseline, 16 regime features)',
    ticker_filter=l5_tickers)

# ── Phase 3: Score L5-1 model ──
l51_tickers = set(l51_configs.keys()) if l51_configs else None
l51_scores = score_model(
    L51_DATA_DIR, L51_GCS_DATA, L51_MODEL_DIR, L51_GCS_MODELS,
    label='L5-1 (sector-aware, 22 regime features)',
    ticker_filter=l51_tickers)

# ── Phase 4: Run backtests ──
l5_trades = run_backtest(l5_scores, L5_DEPLOY_CONFIGS, 'L5')
l51_trades = run_backtest(l51_scores, l51_configs, 'L5-1') if l51_configs else []


# ==============================================================================
# HEAD-TO-HEAD COMPARISON
# ==============================================================================
l5_stats = analyze_strategy(l5_trades, 'L5')
l51_stats = analyze_strategy(l51_trades, 'L5-1')

print(f"\n\n{'='*W}")
print(f"  HEAD-TO-HEAD COMPARISON — L5 vs L5-1 (Sector-Aware FiLM)")
print(f"  TEST: {TEST_START} -> {TEST_END}")
print(f"{'='*W}\n")

header = f"  {'Metric':<25} {'L5 (baseline)':>15} {'L5-1 (sector)':>15} {'Delta':>12}"
print(header)
print(f"  {'─'*67}")

def _fmt_pct(v): return f"{v:+.2%}" if v != 0 else "—"
def _fmt_n(v): return f"{v}" if v != 0 else "—"

metrics = [
    ('Tickers',      _fmt_n(l5_stats['n_tickers']),   _fmt_n(l51_stats['n_tickers']),
     f"{l51_stats['n_tickers'] - l5_stats['n_tickers']:+d}" if l51_stats['n'] > 0 else "—"),
    ('Total trades',  _fmt_n(l5_stats['n']),            _fmt_n(l51_stats['n']),
     f"{l51_stats['n'] - l5_stats['n']:+d}" if l51_stats['n'] > 0 else "—"),
    ('Avg PnL/trade', _fmt_pct(l5_stats['avg_pnl']),   _fmt_pct(l51_stats['avg_pnl']),
     _fmt_pct(l51_stats['avg_pnl'] - l5_stats['avg_pnl'])),
    ('Total PnL',     _fmt_pct(l5_stats['total_pnl']), _fmt_pct(l51_stats['total_pnl']),
     _fmt_pct(l51_stats['total_pnl'] - l5_stats['total_pnl'])),
    ('Win rate',      f"{l5_stats['win_rate']:.1%}",   f"{l51_stats['win_rate']:.1%}" if l51_stats['n'] else "—",
     _fmt_pct(l51_stats['win_rate'] - l5_stats['win_rate'])),
    ('Avg win',       _fmt_pct(l5_stats.get('avg_win', 0)),  _fmt_pct(l51_stats.get('avg_win', 0)), ""),
    ('Avg loss',      _fmt_pct(l5_stats.get('avg_loss', 0)), _fmt_pct(l51_stats.get('avg_loss', 0)), ""),
    ('Sharpe (ann.)', f"{l5_stats['sharpe']:.1f}",     f"{l51_stats['sharpe']:.1f}" if l51_stats['n'] else "—",
     f"{l51_stats['sharpe'] - l5_stats['sharpe']:+.1f}" if l51_stats['n'] else "—"),
]

for label, v5, v51, delta in metrics:
    print(f"  {label:<25} {v5:>15} {v51:>15} {delta:>12}")


# ── Per-ticker breakdown ──
all_bt_tickers = sorted(set(
    [t['Ticker'] for t in l5_trades] + [t['Ticker'] for t in l51_trades]
))

if all_bt_tickers:
    l5_df  = pd.DataFrame(l5_trades)  if l5_trades  else pd.DataFrame(columns=['Ticker','PnL','Outcome'])
    l51_df = pd.DataFrame(l51_trades) if l51_trades else pd.DataFrame(columns=['Ticker','PnL','Outcome'])

    print(f"\n\n  PER-TICKER COMPARISON:")
    print(f"  {'Ticker':<7} {'L5 N':>5} {'L5 Avg':>8} {'L51 N':>6} {'L51 Avg':>8} {'Delta':>8} {'Winner':>8}")
    print(f"  {'─'*58}")

    ticker_comparison = []
    for ticker in all_bt_tickers:
        t5  = l5_df[l5_df['Ticker'] == ticker]  if len(l5_df) > 0  else pd.DataFrame()
        t51 = l51_df[l51_df['Ticker'] == ticker] if len(l51_df) > 0 else pd.DataFrame()
        n5  = len(t5);  avg5  = t5['PnL'].mean()  if n5 > 0  else 0
        n51 = len(t51); avg51 = t51['PnL'].mean() if n51 > 0 else 0
        delta = avg51 - avg5

        winner = "L5-1" if avg51 > avg5 else "L5" if avg5 > avg51 else "TIE"
        if n5 == 0: winner = "L5-1 NEW"
        if n51 == 0: winner = "L5 ONLY"

        tag = ""
        if n5 == 0:
            tag = " ← NEW"
        elif n51 == 0:
            tag = " ← LOST"

        print(f"  {ticker:<7} {n5:>5} {avg5:>+7.2%} {n51:>6} {avg51:>+7.2%} "
              f"{delta:>+7.2%} {winner:>8}{tag}")
        ticker_comparison.append({
            'Ticker': ticker, 'L5_n': n5, 'L5_avg': avg5,
            'L51_n': n51, 'L51_avg': avg51, 'delta': delta, 'winner': winner,
        })

    overlap_tickers = [tc for tc in ticker_comparison if tc['L5_n'] > 0 and tc['L51_n'] > 0]
    l5_only   = [tc for tc in ticker_comparison if tc['L51_n'] == 0]
    l51_new   = [tc for tc in ticker_comparison if tc['L5_n'] == 0]
    l51_wins  = [tc for tc in overlap_tickers if tc['delta'] > 0]

    print(f"\n  OVERLAP ANALYSIS:")
    print(f"    Tickers in both:      {len(overlap_tickers)}")
    print(f"    L5-1 wins (overlap):  {len(l51_wins)}/{len(overlap_tickers)}")
    print(f"    L5-only (lost):       {len(l5_only)} ({', '.join(t['Ticker'] for t in l5_only) or 'none'})")
    print(f"    L5-1 new (gained):    {len(l51_new)} ({', '.join(t['Ticker'] for t in l51_new) or 'none'})")

    if overlap_tickers:
        avg_delta = np.mean([t['delta'] for t in overlap_tickers])
        print(f"    Avg delta (overlap):  {avg_delta:+.2%}")


# ==============================================================================
# CHARTS
# ==============================================================================

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle(f'SHORT Backtest: L5 (baseline) vs L5-1 (sector-aware FiLM)\n'
             f'TEST: {TEST_START} → {TEST_END} | Deploy threshold: >{DEPLOY_PNL_THRESHOLD:.0%}',
             fontsize=14, fontweight='bold')

COLORS = {'L5': '#1565C0', 'L5-1': '#E65100'}

# [0,0] Cumulative PnL
ax = axes[0, 0]
for trades, label, color in [(l5_trades, 'L5', COLORS['L5']),
                               (l51_trades, 'L5-1', COLORS['L5-1'])]:
    if not trades:
        continue
    df = pd.DataFrame(trades).sort_values('Date')
    cum = df['PnL'].cumsum().values
    ax.plot(range(1, len(cum)+1), cum * 100, color=color, linewidth=2, label=label)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_xlabel('Trade #')
ax.set_ylabel('Cumulative PnL (%)')
ax.set_title('Cumulative PnL')
ax.legend()
ax.grid(alpha=0.3)

# [0,1] Daily PnL comparison
ax = axes[0, 1]
for trades, label, color, offset in [(l5_trades, 'L5', COLORS['L5'], -0.2),
                                      (l51_trades, 'L5-1', COLORS['L5-1'], 0.2)]:
    if not trades:
        continue
    df = pd.DataFrame(trades)
    daily = df.groupby('Date')['PnL'].sum()
    dates = pd.to_datetime(daily.index)
    ax.bar(dates + timedelta(hours=int(offset * 24)), daily.values * 100,
           width=0.4, color=color, alpha=0.8, label=label, edgecolor='white', linewidth=0.5)
ax.axhline(0, color='black', linewidth=0.8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
ax.set_ylabel('Daily Total PnL (%)')
ax.set_title('Daily PnL')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# [0,2] Per-ticker comparison bars (paired)
ax = axes[0, 2]
if all_bt_tickers:
    tc_df = pd.DataFrame(ticker_comparison).sort_values('delta', ascending=True)
    y_pos = np.arange(len(tc_df))
    bar_h = 0.35
    ax.barh(y_pos - bar_h/2, tc_df['L5_avg'].values * 100,
            height=bar_h, color=COLORS['L5'], alpha=0.8, label='L5')
    ax.barh(y_pos + bar_h/2, tc_df['L51_avg'].values * 100,
            height=bar_h, color=COLORS['L5-1'], alpha=0.8, label='L5-1')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tc_df['Ticker'].values, fontsize=8)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Avg PnL per Trade (%)')
    ax.set_title('Per-Ticker Head-to-Head')
    ax.legend(fontsize=8)
    ax.grid(axis='x', alpha=0.3)

# [1,0] Delta chart (L5-1 minus L5)
ax = axes[1, 0]
if all_bt_tickers and ticker_comparison:
    tc_df = pd.DataFrame(ticker_comparison).sort_values('delta', ascending=True)
    colors = ['#2E7D32' if d > 0 else '#C62828' for d in tc_df['delta']]
    ax.barh(range(len(tc_df)), tc_df['delta'].values * 100,
            color=colors, edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(tc_df)))
    ax.set_yticklabels(tc_df['Ticker'].values, fontsize=8)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('L5-1 minus L5 (pp)')
    ax.set_title('Per-Ticker Delta (green = L5-1 better)')
    ax.grid(axis='x', alpha=0.3)

# [1,1] Portfolio summary table
ax = axes[1, 1]
ax.axis('off')
table_data = [
    ['Metric', 'L5', 'L5-1'],
    ['Tickers', str(l5_stats['n_tickers']), str(l51_stats['n_tickers'])],
    ['Total trades', str(l5_stats['n']), str(l51_stats['n'])],
    ['Avg PnL/trade', f"{l5_stats['avg_pnl']:+.2%}", f"{l51_stats['avg_pnl']:+.2%}"],
    ['Total PnL', f"{l5_stats['total_pnl']:+.2%}", f"{l51_stats['total_pnl']:+.2%}"],
    ['Win rate', f"{l5_stats['win_rate']:.1%}", f"{l51_stats['win_rate']:.1%}"],
    ['Sharpe', f"{l5_stats['sharpe']:.1f}", f"{l51_stats['sharpe']:.1f}"],
]
table = ax.table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.6)
for (r, c), cell in table.get_celld().items():
    if r == 0:
        cell.set_facecolor('#BBDEFB')
        cell.set_text_props(fontweight='bold')
    elif c == 1:
        cell.set_facecolor('#E3F2FD')
    elif c == 2:
        cell.set_facecolor('#FFF3E0')
ax.set_title('Portfolio Summary', pad=20)

# [1,2] Outcome distribution comparison
ax = axes[1, 2]
outcomes_data = {}
for trades, label in [(l5_trades, 'L5'), (l51_trades, 'L5-1')]:
    if not trades:
        outcomes_data[label] = {'TP': 0, 'SL': 0, 'EOD': 0}
        continue
    df = pd.DataFrame(trades)
    vc = df['Outcome'].value_counts()
    outcomes_data[label] = {k: vc.get(k, 0) for k in ['TP', 'SL', 'EOD']}

x = np.arange(3)
bar_w = 0.35
for i, (label, color) in enumerate([('L5', COLORS['L5']), ('L5-1', COLORS['L5-1'])]):
    vals = [outcomes_data[label].get(k, 0) for k in ['TP', 'SL', 'EOD']]
    ax.bar(x + i * bar_w, vals, bar_w, color=color, alpha=0.8, label=label,
           edgecolor='white', linewidth=0.5)
ax.set_xticks(x + bar_w / 2)
ax.set_xticklabels(['TP', 'SL', 'EOD'])
ax.set_ylabel('Count')
ax.set_title('Outcome Distribution')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
chart_path = '/tmp/backtest_L5_vs_L5_1_short.png'
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
print(f"\n  Chart saved: {chart_path}")
if _IN_COLAB:
    plt.show()
plt.close()


# ==============================================================================
# VERDICT
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  VERDICT")
print(f"{'='*W}")

if l51_stats['n'] == 0:
    print(f"  L5-1 has no trades — optimizer may not have run yet.")
    print(f"  Run the L5-1 pipeline first:")
    print(f"    1. ra_moe_short_L5_1_data_prep.py")
    print(f"    2. ra_moe_short_L5_1_train.py")
    print(f"    3. ra_moe_short_L5_1_optimizer.py")
    print(f"  Then re-run this backtest.")
elif l51_stats['avg_pnl'] > l5_stats['avg_pnl']:
    delta = l51_stats['avg_pnl'] - l5_stats['avg_pnl']
    print(f"  L5-1 (sector-aware FiLM) WINS by {delta:+.2%}/trade")
    print(f"  L5:   {l5_stats['avg_pnl']:+.2%}/trade ({l5_stats['n']} trades, {l5_stats['n_tickers']} tickers)")
    print(f"  L5-1: {l51_stats['avg_pnl']:+.2%}/trade ({l51_stats['n']} trades, {l51_stats['n_tickers']} tickers)")
    if l51_stats['total_pnl'] > l5_stats['total_pnl']:
        print(f"  Total PnL also higher: {l51_stats['total_pnl']:+.2%} vs {l5_stats['total_pnl']:+.2%}")
    if l51_stats['n_tickers'] > l5_stats['n_tickers']:
        print(f"  Wider coverage: {l51_stats['n_tickers']} vs {l5_stats['n_tickers']} tickers")
    print(f"\n  RECOMMENDATION: Deploy L5-1 to replace L5")
elif l51_stats['avg_pnl'] == l5_stats['avg_pnl']:
    print(f"  TIE — sector features had no measurable impact")
else:
    delta = l51_stats['avg_pnl'] - l5_stats['avg_pnl']
    print(f"  L5 (baseline) WINS by {-delta:+.2%}/trade")
    print(f"  L5:   {l5_stats['avg_pnl']:+.2%}/trade ({l5_stats['n']} trades, {l5_stats['n_tickers']} tickers)")
    print(f"  L5-1: {l51_stats['avg_pnl']:+.2%}/trade ({l51_stats['n']} trades, {l51_stats['n_tickers']} tickers)")
    print(f"\n  Sector-aware features did not help. Consider:")
    print(f"    - Check if sector features are informative (data_diagnostics.png)")
    print(f"    - Try #5 (vol-scaled TP/SL) or #7 (multi-horizon ensemble) instead")

print(f"\n{'='*W}")
