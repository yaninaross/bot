# %% ── LONG Backtest: L5-1 vs L5-2 vs L5-2 ASL6 ─────────────────────────────
#
# 3-way head-to-head comparison of LONG Spike Transformer models on
# the COMMON UNSEEN window (Jul 15 – Aug 12, 2026):
#
#   L5-1     — 89 tickers, ASL gamma_neg=5.0 clip=0.10
#              GCS: long_L5_1_data/ + long_L5_1_models/
#              BQ: model_version = 'spike_transformer_long_L5_1_fold5'
#
#   L5-2     — ~100 tickers, dates pushed +5 bdays, same ASL as L5-1
#              GCS: long_L5_2_data/ + long_L5_2_models/
#              BQ: model_version = 'spike_transformer_long_L5_2_fold5'
#
#   L5-2 ASL6 — same as L5-2 but ASL gamma_neg=6.0 clip=0.08 (SHORT-style loss)
#              GCS: long_L5_2_data/ + long_L5_2_asl6_models/
#              BQ: model_version = 'spike_transformer_long_L5_2_asl6_fold5'
#
# Unseen window derivation:
#   L5-1 F5 opt window: val Jun 10 – Jul 9  → test Jul 10 – Aug 12
#   L5-2 F5 opt window: val Jun 15 – Jul 14 → test Jul 15 – Aug 17
#   Common unseen: Jul 15 – Aug 12 (after ALL fold-5 opt windows, within ALL data)
#
# All 3 models share IDENTICAL architecture (DeStationaryAttention + FiLM),
# only training data scope and ASL loss params differ.
#
# LONG trade mechanics:
#   Score at 6:15 AM (min 375), enter at 6:25 AM (min 385)
#   TP: HIGH >= entry*(1+tp)  |  SL: LOW <= entry*(1-sl)
#   PnL = (exit - entry) / entry
#   limit_entry = dip-buy: fill when LOW <= limit_price
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

# Common unseen window — after ALL fold-5 optimization windows, within ALL data
TEST_START     = '2026-07-15'
TEST_END       = '2026-08-12'

SCORE_MINUTE   = 375
ENTRY_MINUTE   = 385
EXIT_TIME_MIN  = 780
SLIPPAGE_PCT   = 0.0015

DEPLOY_PNL_THRESHOLD = 0.018

GCS_BUCKET    = os.environ.get("GCS_BUCKET", "")
GCS_INFER_DIR = 'Data/Inference_v2'

# Architecture (identical for all 3 models)
D_MODEL = 64; N_HEADS = 4; D_FF = 128; N_TRANSFORMER_BLOCKS = 2
DROPOUT = 0.1; FILM_HIDDEN = 64; TIME_STEPS = 128
N_HORIZONS = 3

TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}
EXCLUDE_TICKERS = {'SPY', 'VIX', 'TLT', 'IWM', 'XLE', 'ITA', 'BITF', 'MIGI', 'SMLR'}
PST = pytz.timezone('America/Los_Angeles')

BQ_TABLE = f"{GCP_PROJECT}.scoring_logs.wf_optimization_results"
bq_client = bigquery.Client()

SEED = 42
np.random.seed(SEED)
W = 100

# -- Model configs --
MODELS = {
    'L5-1': {
        'gcs_data':   'Algos/RA_MoE/long_L5_1_data',
        'gcs_models': 'Algos/RA_MoE/long_L5_1_models',
        'data_dir':   '/tmp/ra_moe_long_L5_1_data',
        'model_dir':  '/tmp/ra_moe_long_L5_1_models',
        'mv':         f'spike_transformer_long_L5_1_fold{FOLD_ID}',
        'color':      '#1565C0',
        'desc':       '89 tickers, ASL γ=5.0/clip=0.10',
    },
    'L5-2': {
        'gcs_data':   'Algos/RA_MoE/long_L5_2_data',
        'gcs_models': 'Algos/RA_MoE/long_L5_2_models',
        'data_dir':   '/tmp/ra_moe_long_L5_2_data',
        'model_dir':  '/tmp/ra_moe_long_L5_2_models',
        'mv':         f'spike_transformer_long_L5_2_fold{FOLD_ID}',
        'color':      '#E65100',
        'desc':       '~100 tickers, ASL γ=5.0/clip=0.10',
    },
    'L5-2 ASL6': {
        'gcs_data':   'Algos/RA_MoE/long_L5_2_data',
        'gcs_models': 'Algos/RA_MoE/long_L5_2_asl6_models',
        'data_dir':   '/tmp/ra_moe_long_L5_2_data',
        'model_dir':  '/tmp/ra_moe_long_L5_2_asl6_models',
        'mv':         f'spike_transformer_long_L5_2_asl6_fold{FOLD_ID}',
        'color':      '#2E7D32',
        'desc':       '~100 tickers, ASL γ=6.0/clip=0.08',
    },
}

for cfg in MODELS.values():
    os.makedirs(cfg['data_dir'], exist_ok=True)
    os.makedirs(cfg['model_dir'], exist_ok=True)


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
# MODEL ARCHITECTURE (shared — all 3 models are identical architecture)
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
# LONG SIMULATION ENGINES (Numba) — TP above entry, SL below entry
# ==============================================================================

@jit(nopython=True)
def run_trade_sim(entry_indices, prices, highs, lows, date_ints, time_mins,
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
        target_price = entry_price * (1 + tp)
        stop_price   = entry_price * (1 - sl)
        outcome    = 0.0
        exit_price = entry_price
        exit_type  = 0
        trade_done = False
        exit_idx   = idx
        for k in range(idx + 1, n_prices):
            if date_ints[k] != entry_date:
                exit_price = prices[k-1]
                outcome = (exit_price - entry_price) / entry_price
                exit_idx = k; trade_done = True; break
            sl_hit = lows[k] <= stop_price
            tp_hit = highs[k] >= target_price
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
                outcome = (exit_price - entry_price) / entry_price
                exit_idx = k; trade_done = True; break
        if not trade_done:
            exit_price = prices[n_prices-1]
            outcome  = (exit_price - entry_price) / entry_price
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
def run_trade_sim_grace(entry_indices, prices, highs, lows, date_ints, time_mins,
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
        target_price = entry_price * (1 + tp)
        stop_price   = entry_price * (1 - sl)
        entry_time   = time_mins[idx]
        outcome    = 0.0
        exit_price = entry_price
        exit_type  = 0
        trade_done = False
        exit_idx   = idx
        for k in range(idx + 1, n_prices):
            if date_ints[k] != entry_date:
                exit_price = prices[k-1]
                outcome = (exit_price - entry_price) / entry_price
                exit_idx = k; trade_done = True; break
            minutes_elapsed = time_mins[k] - entry_time
            sl_active = minutes_elapsed >= grace_minutes
            tp_hit = highs[k] >= target_price
            sl_hit = sl_active and (lows[k] <= stop_price)
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
                outcome = (exit_price - entry_price) / entry_price
                exit_idx = k; trade_done = True; break
        if not trade_done:
            exit_price = prices[n_prices-1]
            outcome  = (exit_price - entry_price) / entry_price
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
def run_trade_sim_limit_entry(entry_indices, prices, highs, lows, date_ints, time_mins,
                              tp, sl, exit_time, dip_pct, max_wait_minutes):
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
        exit_price = fill_price
        exit_type  = 0
        trade_done = False
        exit_idx   = fill_idx
        for k in range(fill_idx + 1, n_prices):
            if date_ints[k] != entry_date:
                exit_price = prices[k-1]
                outcome = (exit_price - fill_price) / fill_price
                exit_idx = k; trade_done = True; break
            sl_hit = lows[k] <= stop_price
            tp_hit = highs[k] >= target_price
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
                outcome = (exit_price - fill_price) / fill_price
                exit_idx = k; trade_done = True; break
        if not trade_done:
            exit_price = prices[n_prices-1]
            outcome  = (exit_price - fill_price) / fill_price
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
        result = run_trade_sim_grace(
            indices, prices, highs, lows, dates, times,
            cfg['tp_pct'], cfg['sl_pct'], EXIT_TIME_MIN, cfg['grace_minutes'])
    elif mode == 'limit_entry':
        result = run_trade_sim_limit_entry(
            indices, prices, highs, lows, dates, times,
            cfg['tp_pct'], cfg['sl_pct'], EXIT_TIME_MIN,
            cfg['dip_pct'], cfg['max_wait'])
    else:
        result = run_trade_sim(
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
# GENERIC SCORING — works for any model
# ==============================================================================

def score_model(model_key, ticker_filter=None):
    mcfg = MODELS[model_key]
    print(f"\n{'='*W}")
    print(f"  SCORING -- {model_key} ({mcfg['desc']})")
    print(f"{'='*W}")

    meta_path = ensure_data('meta.npz', mcfg['data_dir'], mcfg['gcs_data'])
    meta = np.load(meta_path, allow_pickle=True)
    ticker_names = meta['ticker_names']
    print(f"  {len(ticker_names)} tickers in meta")

    probe_path = ensure_data(f'fold_{FOLD_ID}_val.npz', mcfg['data_dir'], mcfg['gcs_data'])
    probe = np.load(probe_path)
    n_asset = probe['X_seq'].shape[-1]
    n_regime = probe['X_regime'].shape[-1]
    probe.close()
    print(f"  N_ASSET_FEATURES={n_asset}, N_REGIME_FEATURES={n_regime}")

    safe_name = model_key.replace(' ', '_').replace('-', '_')
    model = build_spike_transformer(n_asset, n_regime, name=f'st_{safe_name}')
    weights_path = ensure_model(FOLD_ID, mcfg['model_dir'], mcfg['gcs_models'])
    model.load_weights(weights_path)
    print(f"  Model loaded: {model.count_params():,} params")

    all_scores = {}
    for split in ['val', 'test']:
        path = ensure_data(f'fold_{FOLD_ID}_{split}.npz', mcfg['data_dir'], mcfg['gcs_data'])
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
# BQ CONFIG LOADER
# ==============================================================================

def load_configs_from_bq(model_version, label):
    query = f"""
    SELECT ticker, prob_thresh, tp_pct, sl_pct, tpsl_mode,
           grace_minutes, dip_pct, max_wait_minutes,
           val_b_avg_pnl, val_b_win_rate, val_b_count,
           experiment
    FROM `{BQ_TABLE}`
    WHERE model_version = '{model_version}'
      AND val_b_avg_pnl >= {DEPLOY_PNL_THRESHOLD}
      AND val_b_count >= 2
    ORDER BY val_b_avg_pnl DESC
    """
    print(f"\n  Querying BQ for {label} configs (model_version='{model_version}')...")
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
            if h_label_val in str(exp):
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

    print(f"  {len(configs)} deployable {label} tickers (val_b >= {DEPLOY_PNL_THRESHOLD:.1%})")
    for t, c in sorted(configs.items(), key=lambda x: -x[1]['val_b_avg_pnl']):
        h = HORIZON_LABELS[c['horizon']]
        print(f"    {t:<7} H={h} mode={c['mode']:<13} "
              f"val_b={c['val_b_avg_pnl']:+.2%} WR={c['val_b_win_rate']:.0%} N={c['val_b_count']}")
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
    print(f"  BACKTEST -- {label} ({len(tickers)} tickers)")
    print(f"  Window: {TEST_START} -> {TEST_END} (unseen by ALL models)")
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
            print(f" dip={cfg.get('dip_pct', 0):.1%} wait={cfg.get('max_wait', 0)}m", end="")
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
                'ew_total_pnl': 0, 'ew_avg_daily': 0, 'avg_trades_per_day': 0,
                'win_rate': 0, 'n_tickers': 0, 'sharpe': 0,
                'avg_win': 0, 'avg_loss': 0, 'n_days': 0}
    df = pd.DataFrame(trades)
    n = len(df)
    avg = df['PnL'].mean()
    std = df['PnL'].std()
    daily_ew = df.groupby('Date')['PnL'].mean()
    n_days = len(daily_ew)
    ew_std = daily_ew.std()
    return {
        'label': label,
        'n': n,
        'avg_pnl': avg,
        'total_pnl': df['PnL'].sum(),
        'ew_total_pnl': daily_ew.sum(),
        'ew_avg_daily': daily_ew.mean() if n_days > 0 else 0,
        'avg_trades_per_day': n / n_days if n_days > 0 else 0,
        'win_rate': (df['PnL'] > 0).mean(),
        'n_tickers': df['Ticker'].nunique(),
        'n_days': n_days,
        'avg_win': df.loc[df['PnL'] > 0, 'PnL'].mean() if (df['PnL'] > 0).any() else 0,
        'avg_loss': df.loc[df['PnL'] <= 0, 'PnL'].mean() if (df['PnL'] <= 0).any() else 0,
        'sharpe': (avg / std * np.sqrt(252)) if std > 0 else 0,
        'ew_sharpe': (daily_ew.mean() / ew_std * np.sqrt(252)) if n_days > 1 and ew_std > 0 else 0,
    }


# ==============================================================================
# MAIN
# ==============================================================================

print(f"\n{'='*W}")
print(f"  LONG BACKTEST: L5-1 vs L5-2 vs L5-2 ASL6")
print(f"  COMMON UNSEEN TEST: {TEST_START} -> {TEST_END}")
print(f"  Fold: {FOLD_ID} | Deploy threshold: >{DEPLOY_PNL_THRESHOLD:.1%}")
print(f"  Slippage: {SLIPPAGE_PCT:.2%} each way ({2*SLIPPAGE_PCT:.2%} RT)")
print(f"{'='*W}")
print(f"\n  Unseen window logic:")
print(f"    L5-1 F5 opt: Jun 10 – Jul 9   | test ends Aug 12")
print(f"    L5-2 F5 opt: Jun 15 – Jul 14  | test ends Aug 17")
print(f"    Common unseen: Jul 15 – Aug 12 (~4 weeks)")

# -- Phase 1: Load configs from BQ for all 3 models --
all_configs = {}
for key, mcfg in MODELS.items():
    all_configs[key] = load_configs_from_bq(mcfg['mv'], key)

# -- Phase 2: Score each model (skip if no configs in BQ) --
all_scores = {}
for key in MODELS:
    if not all_configs[key]:
        print(f"\n  Skipping {key} scoring — no configs in BQ (optimizer not run yet)")
        all_scores[key] = {}
        continue
    ticker_filter = set(all_configs[key].keys())
    all_scores[key] = score_model(key, ticker_filter=ticker_filter)
    tf.keras.backend.clear_session()
    gc.collect()

# -- Phase 3: Run backtests --
all_trades = {}
for key in MODELS:
    all_trades[key] = run_backtest(all_scores[key], all_configs[key], key) if all_configs[key] else []

# -- Phase 4: Analyze --
all_stats = {key: analyze_strategy(all_trades[key], key) for key in MODELS}


# ==============================================================================
# HEAD-TO-HEAD COMPARISON TABLE
# ==============================================================================

model_keys = list(MODELS.keys())

print(f"\n\n{'='*W}")
print(f"  3-WAY COMPARISON — LONG L5-1 vs L5-2 vs L5-2 ASL6")
print(f"  UNSEEN TEST: {TEST_START} -> {TEST_END}")
print(f"{'='*W}\n")

col_w = 17
header_parts = [f"  {'Metric':<22}"]
for key in model_keys:
    header_parts.append(f"{key:>{col_w}}")
header_parts.append(f"{'Best':>8}")
print("".join(header_parts))
print(f"  {'─'*(22 + col_w * len(model_keys) + 8)}")

def _fmt_pct(v): return f"{v:+.2%}" if v != 0 else "---"
def _fmt_n(v): return f"{v}" if v != 0 else "---"
def _best(metric, higher_better=True):
    vals = {k: all_stats[k][metric] for k in model_keys if all_stats[k]['n'] > 0}
    if not vals: return "---"
    return max(vals, key=vals.get) if higher_better else min(vals, key=vals.get)

metrics = [
    ('Tickers',         'n_tickers',         True,  _fmt_n),
    ('Total trades',    'n',                 True,  _fmt_n),
    ('Active days',     'n_days',            True,  _fmt_n),
    ('Avg trades/day',  'avg_trades_per_day',True,  lambda v: f"{v:.1f}" if v != 0 else "---"),
    ('Avg PnL/trade',   'avg_pnl',          True,  _fmt_pct),
    ('EW Portfolio PnL', 'ew_total_pnl',    True,  _fmt_pct),
    ('EW Avg daily',    'ew_avg_daily',      True,  _fmt_pct),
    ('Win rate',        'win_rate',          True,  lambda v: f"{v:.1%}" if v != 0 else "---"),
    ('Avg win',         'avg_win',           True,  _fmt_pct),
    ('Avg loss',        'avg_loss',          False, _fmt_pct),
    ('Sharpe (trade)',  'sharpe',            True,  lambda v: f"{v:.1f}" if v != 0 else "---"),
    ('Sharpe (EW)',     'ew_sharpe',         True,  lambda v: f"{v:.1f}" if v != 0 else "---"),
]

for label, metric, higher_better, fmt in metrics:
    parts = [f"  {label:<22}"]
    for key in model_keys:
        parts.append(f"{fmt(all_stats[key][metric]):>{col_w}}")
    parts.append(f"{_best(metric, higher_better):>8}")
    print("".join(parts))


# ==============================================================================
# PER-TICKER COMPARISON
# ==============================================================================

all_bt_tickers = sorted(set(
    t['Ticker'] for trades in all_trades.values() for t in trades
))

if all_bt_tickers:
    dfs = {key: pd.DataFrame(all_trades[key]) if all_trades[key]
           else pd.DataFrame(columns=['Ticker','PnL','Outcome'])
           for key in model_keys}

    print(f"\n\n  PER-TICKER COMPARISON (unseen window only):")
    tc_header = f"  {'Ticker':<7}"
    for key in model_keys:
        tc_header += f" {key+' N':>7} {key+' Avg':>9}"
    tc_header += f" {'Best':>10}"
    print(tc_header)
    print(f"  {'─'*(7 + (7+9) * len(model_keys) + 10)}")

    ticker_comparison = []
    for ticker in all_bt_tickers:
        row = {'Ticker': ticker}
        best_key = None
        best_avg = -999.0
        for key in model_keys:
            tdf = dfs[key]
            t = tdf[tdf['Ticker'] == ticker] if len(tdf) > 0 else pd.DataFrame()
            n = len(t)
            avg = float(t['PnL'].mean()) if n > 0 else 0.0
            row[f'{key}_n'] = n
            row[f'{key}_avg'] = avg
            if n > 0 and avg > best_avg:
                best_avg = avg
                best_key = key
        row['best'] = best_key or "---"
        ticker_comparison.append(row)

        parts = [f"  {ticker:<7}"]
        for key in model_keys:
            n = row[f'{key}_n']
            avg = row[f'{key}_avg']
            parts.append(f" {n:>7} {avg:>+8.2%}")
        parts.append(f" {row['best']:>10}")
        print("".join(parts))

    # Overlap analysis
    print(f"\n  COVERAGE ANALYSIS:")
    for key in model_keys:
        tickers_with_trades = [tc for tc in ticker_comparison if tc[f'{key}_n'] > 0]
        print(f"    {key}: {len(tickers_with_trades)} tickers with trades")

    # Pairwise wins
    print(f"\n  PAIRWISE WINS (per-ticker avg PnL, tickers present in BOTH):")
    for i, k1 in enumerate(model_keys):
        for k2 in model_keys[i+1:]:
            overlap = [tc for tc in ticker_comparison
                       if tc[f'{k1}_n'] > 0 and tc[f'{k2}_n'] > 0]
            if not overlap:
                print(f"    {k1} vs {k2}: no overlap")
                continue
            k1_wins = sum(1 for tc in overlap if tc[f'{k1}_avg'] > tc[f'{k2}_avg'])
            k2_wins = sum(1 for tc in overlap if tc[f'{k2}_avg'] > tc[f'{k1}_avg'])
            ties = len(overlap) - k1_wins - k2_wins
            avg_delta = np.mean([tc[f'{k2}_avg'] - tc[f'{k1}_avg'] for tc in overlap])
            print(f"    {k1} vs {k2}: {len(overlap)} overlap | "
                  f"{k1}={k1_wins} {k2}={k2_wins} ties={ties} | "
                  f"avg delta={avg_delta:+.2%}")


# ==============================================================================
# CHARTS
# ==============================================================================

fig, axes = plt.subplots(2, 3, figsize=(22, 13))
fig.suptitle(f'LONG Backtest: L5-1 vs L5-2 vs L5-2 ASL6\n'
             f'UNSEEN TEST: {TEST_START} -> {TEST_END} | '
             f'Fold {FOLD_ID} | Deploy > {DEPLOY_PNL_THRESHOLD:.1%}',
             fontsize=14, fontweight='bold')

COLORS = {key: MODELS[key]['color'] for key in model_keys}

# [0,0] Equal-Weight Portfolio Cumulative PnL
ax = axes[0, 0]
for key in model_keys:
    trades = all_trades[key]
    if not trades:
        continue
    df = pd.DataFrame(trades)
    daily_ew = df.groupby('Date')['PnL'].mean().sort_index()
    dates = pd.to_datetime(daily_ew.index)
    cum = daily_ew.cumsum().values
    ax.plot(dates, cum * 100, color=COLORS[key], linewidth=2,
            label=f"{key} ({cum[-1]*100:+.1f}%)", marker='o', markersize=3)
ax.axhline(0, color='black', linewidth=0.8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
ax.set_ylabel('Cumulative EW PnL (%)')
ax.set_title('Equal-Weight Portfolio (cumulative)\ncash split equally across all trades each day')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# [0,1] Equal-Weight Daily PnL (cash split equally among that day's trades)
ax = axes[0, 1]
offsets = {model_keys[0]: -0.25, model_keys[1]: 0.0, model_keys[2]: 0.25}
for key in model_keys:
    trades = all_trades[key]
    if not trades:
        continue
    df = pd.DataFrame(trades)
    daily_ew = df.groupby('Date').agg(ew_pnl=('PnL', 'mean'), n=('PnL', 'count'))
    dates = pd.to_datetime(daily_ew.index)
    ax.bar(dates + timedelta(hours=int(offsets[key] * 24)), daily_ew['ew_pnl'].values * 100,
           width=0.28, color=COLORS[key], alpha=0.8, label=key,
           edgecolor='white', linewidth=0.5)
ax.axhline(0, color='black', linewidth=0.8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
ax.set_ylabel('EW Daily PnL (%)')
ax.set_title('Equal-Weight Daily PnL\n(avg of all trades that day)')
ax.legend(fontsize=8)
ax.grid(axis='y', alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# [0,2] Per-ticker comparison bars (grouped)
ax = axes[0, 2]
if all_bt_tickers and ticker_comparison:
    tc_df = pd.DataFrame(ticker_comparison)
    tc_df['max_avg'] = tc_df[[f'{k}_avg' for k in model_keys]].max(axis=1)
    tc_df = tc_df.sort_values('max_avg', ascending=True)
    y_pos = np.arange(len(tc_df))
    n_models = len(model_keys)
    bar_h = 0.8 / n_models
    for i, key in enumerate(model_keys):
        offset = (i - n_models/2 + 0.5) * bar_h
        ax.barh(y_pos + offset, tc_df[f'{key}_avg'].values * 100,
                height=bar_h, color=COLORS[key], alpha=0.8, label=key)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tc_df['Ticker'].values, fontsize=7)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Avg PnL per Trade (%)')
    ax.set_title('Per-Ticker Head-to-Head')
    ax.legend(fontsize=7)
    ax.grid(axis='x', alpha=0.3)

# [1,0] L5-2 vs L5-1 delta
ax = axes[1, 0]
if ticker_comparison:
    tc_df = pd.DataFrame(ticker_comparison)
    overlap = tc_df[(tc_df['L5-1_n'] > 0) & (tc_df['L5-2_n'] > 0)].copy()
    if len(overlap) > 0:
        overlap['delta'] = overlap['L5-2_avg'] - overlap['L5-1_avg']
        overlap = overlap.sort_values('delta', ascending=True)
        colors_delta = ['#2E7D32' if d > 0 else '#C62828' for d in overlap['delta']]
        ax.barh(range(len(overlap)), overlap['delta'].values * 100,
                color=colors_delta, edgecolor='white', linewidth=0.5)
        ax.set_yticks(range(len(overlap)))
        ax.set_yticklabels(overlap['Ticker'].values, fontsize=8)
        ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('L5-2 minus L5-1 (pp)')
    ax.set_title('Delta: L5-2 vs L5-1 (green = L5-2 better)')
    ax.grid(axis='x', alpha=0.3)

# [1,1] Portfolio summary table
ax = axes[1, 1]
ax.axis('off')
table_data = [['Metric'] + model_keys]
table_rows = [
    ('Tickers',       'n_tickers',         str),
    ('Trades',        'n',                 str),
    ('Active days',   'n_days',            str),
    ('Trades/day',    'avg_trades_per_day', lambda v: f"{v:.1f}"),
    ('Avg PnL/trade', 'avg_pnl',           lambda v: f"{v:+.2%}"),
    ('EW Total PnL',  'ew_total_pnl',      lambda v: f"{v:+.2%}"),
    ('Win rate',      'win_rate',           lambda v: f"{v:.1%}"),
    ('Sharpe (EW)',   'ew_sharpe',          lambda v: f"{v:.1f}"),
]
for label, metric, fmt in table_rows:
    row = [label]
    for key in model_keys:
        row.append(fmt(all_stats[key][metric]))
    table_data.append(row)

table = ax.table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.5)
for (r, c), cell in table.get_celld().items():
    if r == 0:
        cell.set_facecolor('#BBDEFB')
        cell.set_text_props(fontweight='bold')
    elif c == 1:
        cell.set_facecolor('#E3F2FD')
    elif c == 2:
        cell.set_facecolor('#FFF3E0')
    elif c == 3:
        cell.set_facecolor('#E8F5E9')
ax.set_title('Portfolio Summary', pad=20)

# [1,2] Outcome distribution comparison
ax = axes[1, 2]
outcomes_data = {}
for key in model_keys:
    trades = all_trades[key]
    if not trades:
        outcomes_data[key] = {'TP': 0, 'SL': 0, 'EOD': 0}
        continue
    df = pd.DataFrame(trades)
    vc = df['Outcome'].value_counts()
    outcomes_data[key] = {k: vc.get(k, 0) for k in ['TP', 'SL', 'EOD']}

x = np.arange(3)
n_models = len(model_keys)
bar_w = 0.8 / n_models
for i, key in enumerate(model_keys):
    vals = [outcomes_data[key].get(k, 0) for k in ['TP', 'SL', 'EOD']]
    ax.bar(x + i * bar_w, vals, bar_w, color=COLORS[key], alpha=0.8,
           label=key, edgecolor='white', linewidth=0.5)
ax.set_xticks(x + bar_w)
ax.set_xticklabels(['TP', 'SL', 'EOD'])
ax.set_ylabel('Count')
ax.set_title('Outcome Distribution')
ax.legend(fontsize=8)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
chart_path = '/tmp/backtest_long_L5_1_vs_L5_2_vs_asl6.png'
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

active_models = [k for k in model_keys if all_stats[k]['n'] > 0]
inactive_models = [k for k in model_keys if all_stats[k]['n'] == 0]

if not active_models:
    print(f"  No models have trades -- optimizers may not have run yet.")
    print(f"  Run pipeline for each model:")
    print(f"    1. ra_moe_long_L5_2_data_prep.py")
    print(f"    2. ra_moe_long_L5_2_train.py + ra_moe_long_L5_2_asl6_train.py")
    print(f"    3. ra_moe_long_L5_2_optimizer.py + ra_moe_long_L5_2_asl6_optimizer.py")
    print(f"  Then re-run this comparison.")
elif len(active_models) == 1:
    k = active_models[0]
    print(f"  Only {k} has trades ({all_stats[k]['n']})")
    print(f"  Waiting for optimizers: {', '.join(inactive_models)}")
else:
    best_key = max(active_models, key=lambda k: all_stats[k]['avg_pnl'])
    best_avg = all_stats[best_key]['avg_pnl']

    print(f"\n  RANKING (by avg PnL/trade on unseen window {TEST_START} -> {TEST_END}):")
    ranked = sorted(active_models, key=lambda k: all_stats[k]['avg_pnl'], reverse=True)
    for rank, key in enumerate(ranked, 1):
        s = all_stats[key]
        delta = s['avg_pnl'] - best_avg if key != best_key else 0
        tag = " <-- WINNER" if key == best_key else ""
        print(f"    #{rank} {key:<12} {s['avg_pnl']:+.2%}/trade | "
              f"{s['n']} trades ({s['avg_trades_per_day']:.1f}/day) | "
              f"EW total {s['ew_total_pnl']:+.2%} | WR {s['win_rate']:.0%} | "
              f"Sharpe {s['ew_sharpe']:.1f}{tag}")
        if delta != 0:
            print(f"       delta vs winner: {delta:+.2%}/trade")

    if inactive_models:
        print(f"\n  Not yet optimized: {', '.join(inactive_models)}")

    print(f"\n  Key questions:")
    if 'L5-2' in active_models and 'L5-1' in active_models:
        d = all_stats['L5-2']['avg_pnl'] - all_stats['L5-1']['avg_pnl']
        print(f"    More tickers help?  L5-2 vs L5-1 delta = {d:+.2%}/trade "
              f"({'YES' if d > 0 else 'NO'})")
    if 'L5-2 ASL6' in active_models and 'L5-2' in active_models:
        d = all_stats['L5-2 ASL6']['avg_pnl'] - all_stats['L5-2']['avg_pnl']
        print(f"    ASL6 loss helps?    ASL6 vs L5-2 delta = {d:+.2%}/trade "
              f"({'YES' if d > 0 else 'NO'})")
    if 'L5-2 ASL6' in active_models and 'L5-1' in active_models:
        d = all_stats['L5-2 ASL6']['avg_pnl'] - all_stats['L5-1']['avg_pnl']
        print(f"    ASL6 vs baseline?   ASL6 vs L5-1 delta = {d:+.2%}/trade "
              f"({'YES' if d > 0 else 'NO'})")

    if best_avg >= DEPLOY_PNL_THRESHOLD:
        print(f"\n  RECOMMENDATION: Deploy {best_key} ({best_avg:+.2%}/trade >= {DEPLOY_PNL_THRESHOLD:.1%} threshold)")
    else:
        print(f"\n  WARNING: Best model ({best_key}) avg PnL {best_avg:+.2%} < {DEPLOY_PNL_THRESHOLD:.1%} threshold")
        print(f"  None of the models meet deploy threshold on unseen data.")

print(f"\n{'='*W}")
