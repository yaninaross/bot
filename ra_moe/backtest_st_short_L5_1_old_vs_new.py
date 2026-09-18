# %% ── SHORT Backtest: L5-1 Old Optimization vs New Optimization ────────────────
#
# Same L5-1 model (sector-aware FiLM, 22 regime features), different optimization:
#
#   OLD  — static / grace / limit_entry configs (initial optimizer run)
#          BQ: tpsl_mode IN ('static', 'grace', 'limit_entry')
#
#   NEW  — volscaled / ensemble configs (vol-scaled TP/SL + multi-horizon blend)
#          BQ: tpsl_mode IN ('volscaled', 'ensemble')
#
#   COMBINED — best config across ALL 5 modes per ticker
#
# All configs share model_version = 'spike_transformer_short_L5_1_fold5'.
# Answers: "Did vol-scaled TP/SL or multi-horizon ensemble find better configs?"
#
# TEST: Jul 10 - Aug 10, 2026  |  Deploy threshold: >1% TEST avg PnL
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

# L5-1 model paths (same model for both OLD and NEW)
GCS_DATA   = 'Algos/RA_MoE/short_L5_1_data'
GCS_MODELS = 'Algos/RA_MoE/short_L5_1_models'
DATA_DIR   = '/tmp/ra_moe_short_L5_1_data'
MODEL_DIR  = '/tmp/ra_moe_short_L5_1_models'
MODEL_VERSION = 'spike_transformer_short_L5_1_fold5'

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Architecture
D_MODEL = 64; N_HEADS = 4; D_FF = 128; N_TRANSFORMER_BLOCKS = 2
DROPOUT = 0.1; FILM_HIDDEN = 64; TIME_STEPS = 128
N_HORIZONS = 3
ASL_GAMMA_NEG = 6.0; ASL_GAMMA_POS = 0.1; ASL_CLIP = 0.08

# Vol-scaled params (must match optimizer)
VOL_LOOKBACK  = 20
VOL_SCALE_MIN = 0.5
VOL_SCALE_MAX = 2.0

# Ensemble presets (must match optimizer)
ENSEMBLE_PRESETS = [
    {'name': 'equal',  'weights': [1/3, 1/3, 1/3]},
    {'name': 'short',  'weights': [0.5, 0.3, 0.2]},
    {'name': 'medium', 'weights': [0.2, 0.5, 0.3]},
    {'name': 'max',    'weights': None},
]

OLD_MODES = {'static', 'grace', 'limit_entry'}
NEW_MODES = {'volscaled', 'ensemble'}

TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}
EXCLUDE_TICKERS = {'SPY', 'VIX', 'TLT', 'IWM', 'XLE', 'ITA', 'BITF', 'MIGI', 'SMLR'}
PST = pytz.timezone('America/Los_Angeles')

BQ_TABLE = f"{GCP_PROJECT}.scoring_logs.wf_optimization_results"
bq_client = bigquery.Client()

SEED = 42
np.random.seed(SEED)
W = 100


# ==============================================================================
# GCS HELPERS
# ==============================================================================
def _gcs_bucket():
    return storage.Client().bucket(GCS_BUCKET)

def download_from_gcs(gcs_path, local_path):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    _gcs_bucket().blob(gcs_path).download_to_filename(local_path)

def ensure_data(filename):
    local_path = os.path.join(DATA_DIR, filename)
    if os.path.exists(local_path):
        return local_path
    gcs_path = f'{GCS_DATA}/{filename}'
    print(f"    Downloading {filename}...")
    download_from_gcs(gcs_path, local_path)
    return local_path

def ensure_model(fold_id):
    local_dir = os.path.join(MODEL_DIR, f'fold_{fold_id}')
    weights_path = os.path.join(local_dir, 'model.weights.h5')
    if os.path.exists(weights_path):
        return weights_path
    os.makedirs(local_dir, exist_ok=True)
    gcs_p = f'{GCS_MODELS}/fold_{fold_id}'
    print(f"    Downloading fold {fold_id} model...")
    bucket = _gcs_bucket()
    for blob in bucket.list_blobs(prefix=gcs_p):
        fname = blob.name.split('/')[-1]
        blob.download_to_filename(os.path.join(local_dir, fname))
    return weights_path


# ==============================================================================
# MODEL ARCHITECTURE
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
        self.wq = layers.Dense(d_model); self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model); self.wo = layers.Dense(d_model)
        self.attn_drop = layers.Dropout(dropout)
        self.tau_proj = layers.Dense(n_heads)
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
        out = tf.matmul(attn, V)
        out = tf.reshape(tf.transpose(out, (0, 2, 1, 3)), (B, T, self.d_model))
        sigma = tf.nn.softplus(self.sigma_proj(stats))
        return self.wo(out * sigma[:, tf.newaxis, :])

class TransformerBlock(layers.Layer):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.attn = DeStationaryAttention(d_model, n_heads, dropout)
        self.ffn = tf.keras.Sequential([
            layers.Dense(d_ff, activation='gelu'), layers.Dense(d_model),
            layers.Dropout(dropout)])
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.drop = layers.Dropout(dropout)

    def call(self, x, stats, training=False):
        x = x + self.drop(self.attn(self.norm1(x), stats, training=training), training=training)
        x = x + self.ffn(self.norm2(x), training=training)
        return x

class TransformerBackbone(layers.Layer):
    def __init__(self, d_model, n_heads, d_ff, n_blocks, seq_len, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.stationarize = SeriesStationarization()
        self.input_proj = layers.Dense(d_model)
        self.pos_embed = self.add_weight(
            name='pos_embed', shape=(1, seq_len, d_model),
            initializer='glorot_normal', trainable=True)
        self.input_drop = layers.Dropout(dropout)
        self.blocks = [TransformerBlock(d_model, n_heads, d_ff, dropout, name=f'block_{i}')
                       for i in range(n_blocks)]
        self.final_norm = layers.LayerNormalization()

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
        self.query = self.add_weight(name='pool_query', shape=(1, 1, d_model),
                                     initializer='glorot_normal', trainable=True)
        self.scale = tf.sqrt(tf.cast(d_model, tf.float32))

    def call(self, x, training=False):
        scores = tf.matmul(self.query, x, transpose_b=True) / self.scale
        weights = tf.nn.softmax(scores, axis=-1)
        return tf.matmul(weights, x)[:, 0, :]

class FiLMConditioner(layers.Layer):
    def __init__(self, d_model, hidden_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.mlp = tf.keras.Sequential([
            layers.Dense(hidden_dim, activation='relu'),
            layers.Dense(hidden_dim, activation='relu')])
        self.gamma_proj = layers.Dense(d_model)
        self.beta_proj = layers.Dense(d_model)

    def call(self, x, regime_features, training=False):
        h = self.mlp(regime_features, training=training)
        return (1.0 + self.gamma_proj(h)) * x + self.beta_proj(h)


def build_spike_transformer(n_asset_features, n_regime_features):
    class _ST(Model):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.backbone = TransformerBackbone(
                D_MODEL, N_HEADS, D_FF, N_TRANSFORMER_BLOCKS, TIME_STEPS, DROPOUT, name='backbone')
            self.attn_pool = AttentionPooling(D_MODEL, name='attn_pool')
            self.film = FiLMConditioner(D_MODEL, FILM_HIDDEN, name='film')
            self.drop = layers.Dropout(DROPOUT)
            self.head = layers.Dense(N_HORIZONS, activation='sigmoid', name='spike_head')

        def call(self, inputs, training=False):
            h = self.backbone(inputs['seq_input'], training=training)
            h = self.attn_pool(h, training=training)
            h = self.film(h, inputs['regime_input'], training=training)
            h = self.drop(h, training=training)
            return self.head(h)

    model = _ST(name='spike_transformer')
    model({'seq_input': tf.zeros((1, TIME_STEPS, n_asset_features)),
           'regime_input': tf.zeros((1, n_regime_features))})
    return model


# ==============================================================================
# SHORT SIMULATION ENGINES (Numba)
# ==============================================================================

@jit(nopython=True)
def run_short_trade_sim(entry_indices, prices, highs, lows, date_ints, time_mins,
                        tp, sl, exit_time):
    n_entries = len(entry_indices)
    n_prices  = len(prices)
    pnl_arr = np.empty(n_entries, dtype=np.float64)
    date_arr = np.empty(n_entries, dtype=np.int64)
    outcome_arr = np.empty(n_entries, dtype=np.int64)
    trade_count = 0; current_entry_ptr = 0; last_traded_date = -1
    while current_entry_ptr < n_entries:
        idx = entry_indices[current_entry_ptr]
        if idx >= n_prices - 1: break
        entry_date = date_ints[idx]
        if entry_date == last_traded_date:
            current_entry_ptr += 1; continue
        entry_price = prices[idx]
        target_price = entry_price * (1 - tp)
        stop_price = entry_price * (1 + sl)
        outcome = 0.0; exit_type = 0; trade_done = False; exit_idx = idx
        for k in range(idx + 1, n_prices):
            if date_ints[k] != entry_date:
                outcome = (entry_price - prices[k-1]) / entry_price
                exit_idx = k; trade_done = True; break
            tp_hit = lows[k] <= target_price
            sl_hit = highs[k] >= stop_price
            if sl_hit and tp_hit:
                if np.random.random() < 0.5:
                    outcome = -sl; exit_type = -1
                else:
                    outcome = tp; exit_type = 1
                exit_idx = k; trade_done = True; break
            if sl_hit:
                outcome = -sl; exit_type = -1; exit_idx = k; trade_done = True; break
            if tp_hit:
                outcome = tp; exit_type = 1; exit_idx = k; trade_done = True; break
            if time_mins[k] >= exit_time:
                outcome = (entry_price - prices[k]) / entry_price
                exit_idx = k; trade_done = True; break
        if not trade_done:
            outcome = (entry_price - prices[n_prices-1]) / entry_price; exit_idx = n_prices
        pnl_arr[trade_count] = outcome
        date_arr[trade_count] = entry_date
        outcome_arr[trade_count] = exit_type
        trade_count += 1; last_traded_date = entry_date
        next_ptr = current_entry_ptr + 1
        while next_ptr < n_entries:
            if entry_indices[next_ptr] > exit_idx: break
            next_ptr += 1
        current_entry_ptr = next_ptr
    return pnl_arr[:trade_count], date_arr[:trade_count], outcome_arr[:trade_count]


@jit(nopython=True)
def run_short_trade_sim_grace(entry_indices, prices, highs, lows, date_ints, time_mins,
                              tp, sl, exit_time, grace_minutes):
    n_entries = len(entry_indices)
    n_prices  = len(prices)
    pnl_arr = np.empty(n_entries, dtype=np.float64)
    date_arr = np.empty(n_entries, dtype=np.int64)
    outcome_arr = np.empty(n_entries, dtype=np.int64)
    trade_count = 0; current_entry_ptr = 0; last_traded_date = -1
    while current_entry_ptr < n_entries:
        idx = entry_indices[current_entry_ptr]
        if idx >= n_prices - 1: break
        entry_date = date_ints[idx]
        if entry_date == last_traded_date:
            current_entry_ptr += 1; continue
        entry_price = prices[idx]
        target_price = entry_price * (1 - tp)
        stop_price = entry_price * (1 + sl)
        entry_time = time_mins[idx]
        outcome = 0.0; exit_type = 0; trade_done = False; exit_idx = idx
        for k in range(idx + 1, n_prices):
            if date_ints[k] != entry_date:
                outcome = (entry_price - prices[k-1]) / entry_price
                exit_idx = k; trade_done = True; break
            minutes_elapsed = time_mins[k] - entry_time
            sl_active = minutes_elapsed >= grace_minutes
            tp_hit = lows[k] <= target_price
            sl_hit = sl_active and (highs[k] >= stop_price)
            if sl_hit and tp_hit:
                if np.random.random() < 0.5:
                    outcome = -sl; exit_type = -1
                else:
                    outcome = tp; exit_type = 1
                exit_idx = k; trade_done = True; break
            if sl_hit:
                outcome = -sl; exit_type = -1; exit_idx = k; trade_done = True; break
            if tp_hit:
                outcome = tp; exit_type = 1; exit_idx = k; trade_done = True; break
            if time_mins[k] >= exit_time:
                outcome = (entry_price - prices[k]) / entry_price
                exit_idx = k; trade_done = True; break
        if not trade_done:
            outcome = (entry_price - prices[n_prices-1]) / entry_price; exit_idx = n_prices
        pnl_arr[trade_count] = outcome
        date_arr[trade_count] = entry_date
        outcome_arr[trade_count] = exit_type
        trade_count += 1; last_traded_date = entry_date
        next_ptr = current_entry_ptr + 1
        while next_ptr < n_entries:
            if entry_indices[next_ptr] > exit_idx: break
            next_ptr += 1
        current_entry_ptr = next_ptr
    return pnl_arr[:trade_count], date_arr[:trade_count], outcome_arr[:trade_count]


@jit(nopython=True)
def run_short_trade_sim_limit_entry(entry_indices, prices, highs, lows, date_ints, time_mins,
                                    tp, sl, exit_time, rally_pct, max_wait_minutes):
    n_entries = len(entry_indices)
    n_prices  = len(prices)
    pnl_arr = np.empty(n_entries, dtype=np.float64)
    date_arr = np.empty(n_entries, dtype=np.int64)
    outcome_arr = np.empty(n_entries, dtype=np.int64)
    trade_count = 0; current_entry_ptr = 0; last_traded_date = -1
    while current_entry_ptr < n_entries:
        idx = entry_indices[current_entry_ptr]
        if idx >= n_prices - 1: break
        entry_date = date_ints[idx]
        if entry_date == last_traded_date:
            current_entry_ptr += 1; continue
        signal_price = prices[idx]
        limit_price = signal_price * (1.0 + rally_pct)
        signal_time = time_mins[idx]
        fill_idx = -1; fill_price = 0.0
        for k in range(idx + 1, n_prices):
            if date_ints[k] != entry_date: break
            if time_mins[k] - signal_time > max_wait_minutes: break
            if highs[k] >= limit_price:
                fill_price = limit_price; fill_idx = k; break
        if fill_idx < 0:
            current_entry_ptr += 1; last_traded_date = entry_date; continue
        target_price = fill_price * (1 - tp)
        stop_price = fill_price * (1 + sl)
        outcome = 0.0; exit_type = 0; trade_done = False; exit_idx = fill_idx
        for k in range(fill_idx + 1, n_prices):
            if date_ints[k] != entry_date:
                outcome = (fill_price - prices[k-1]) / fill_price
                exit_idx = k; trade_done = True; break
            tp_hit = lows[k] <= target_price
            sl_hit = highs[k] >= stop_price
            if sl_hit and tp_hit:
                if np.random.random() < 0.5:
                    outcome = -sl; exit_type = -1
                else:
                    outcome = tp; exit_type = 1
                exit_idx = k; trade_done = True; break
            if sl_hit:
                outcome = -sl; exit_type = -1; exit_idx = k; trade_done = True; break
            if tp_hit:
                outcome = tp; exit_type = 1; exit_idx = k; trade_done = True; break
            if time_mins[k] >= exit_time:
                outcome = (fill_price - prices[k]) / fill_price
                exit_idx = k; trade_done = True; break
        if not trade_done:
            outcome = (fill_price - prices[n_prices-1]) / fill_price; exit_idx = n_prices
        pnl_arr[trade_count] = outcome
        date_arr[trade_count] = entry_date
        outcome_arr[trade_count] = exit_type
        trade_count += 1; last_traded_date = entry_date
        next_ptr = current_entry_ptr + 1
        while next_ptr < n_entries:
            if entry_indices[next_ptr] > exit_idx: break
            next_ptr += 1
        current_entry_ptr = next_ptr
    return pnl_arr[:trade_count], date_arr[:trade_count], outcome_arr[:trade_count]


@jit(nopython=True)
def run_short_trade_sim_volscaled(entry_indices, prices, highs, lows, date_ints, time_mins,
                                  base_tp, base_sl, exit_time, vol_scales):
    n_entries = len(entry_indices)
    n_prices  = len(prices)
    pnl_arr = np.empty(n_entries, dtype=np.float64)
    date_arr = np.empty(n_entries, dtype=np.int64)
    outcome_arr = np.empty(n_entries, dtype=np.int64)
    trade_count = 0; current_entry_ptr = 0; last_traded_date = -1
    while current_entry_ptr < n_entries:
        idx = entry_indices[current_entry_ptr]
        if idx >= n_prices - 1: break
        entry_date = date_ints[idx]
        if entry_date == last_traded_date:
            current_entry_ptr += 1; continue
        entry_price = prices[idx]
        vs = vol_scales[idx]
        tp = base_tp * vs
        sl = base_sl * vs
        target_price = entry_price * (1 - tp)
        stop_price = entry_price * (1 + sl)
        outcome = 0.0; exit_type = 0; trade_done = False; exit_idx = idx
        for k in range(idx + 1, n_prices):
            if date_ints[k] != entry_date:
                outcome = (entry_price - prices[k-1]) / entry_price
                exit_idx = k; trade_done = True; break
            tp_hit = lows[k] <= target_price
            sl_hit = highs[k] >= stop_price
            if sl_hit and tp_hit:
                if np.random.random() < 0.5:
                    outcome = -sl; exit_type = -1
                else:
                    outcome = tp; exit_type = 1
                exit_idx = k; trade_done = True; break
            if sl_hit:
                outcome = -sl; exit_type = -1; exit_idx = k; trade_done = True; break
            if tp_hit:
                outcome = tp; exit_type = 1; exit_idx = k; trade_done = True; break
            if time_mins[k] >= exit_time:
                outcome = (entry_price - prices[k]) / entry_price
                exit_idx = k; trade_done = True; break
        if not trade_done:
            outcome = (entry_price - prices[n_prices-1]) / entry_price; exit_idx = n_prices
        pnl_arr[trade_count] = outcome
        date_arr[trade_count] = entry_date
        outcome_arr[trade_count] = exit_type
        trade_count += 1; last_traded_date = entry_date
        next_ptr = current_entry_ptr + 1
        while next_ptr < n_entries:
            if entry_indices[next_ptr] > exit_idx: break
            next_ptr += 1
        current_entry_ptr = next_ptr
    return pnl_arr[:trade_count], date_arr[:trade_count], outcome_arr[:trade_count]


def _run_sim(indices, ohlcv_arrays, cfg):
    prices = ohlcv_arrays['prices']; highs = ohlcv_arrays['highs']
    lows = ohlcv_arrays['lows']; dates = ohlcv_arrays['dates']
    times = ohlcv_arrays['times'].astype(np.int64)
    mode = cfg['mode']
    if mode == 'grace':
        pnl, d, out = run_short_trade_sim_grace(
            indices, prices, highs, lows, dates, times,
            cfg['tp_pct'], cfg['sl_pct'], EXIT_TIME_MIN, cfg['grace_minutes'])
    elif mode == 'limit_entry':
        pnl, d, out = run_short_trade_sim_limit_entry(
            indices, prices, highs, lows, dates, times,
            cfg['tp_pct'], cfg['sl_pct'], EXIT_TIME_MIN, cfg['dip_pct'], cfg['max_wait'])
    elif mode == 'volscaled':
        vol_scales = ohlcv_arrays.get('vol_scales')
        if vol_scales is None:
            vol_scales = np.ones(len(prices), dtype=np.float64)
        pnl, d, out = run_short_trade_sim_volscaled(
            indices, prices, highs, lows, dates, times,
            cfg['tp_pct'], cfg['sl_pct'], EXIT_TIME_MIN, vol_scales)
    else:
        pnl, d, out = run_short_trade_sim(
            indices, prices, highs, lows, dates, times,
            cfg['tp_pct'], cfg['sl_pct'], EXIT_TIME_MIN)
    if len(pnl) > 0 and SLIPPAGE_PCT > 0:
        pnl = pnl - 2 * SLIPPAGE_PCT
    return pnl, d, out


# ==============================================================================
# VOL SCALING + ENSEMBLE SCORING
# ==============================================================================

def compute_vol_scales(df_ohlcv, ohlcv_dates):
    df = df_ohlcv.copy()
    df['date_int'] = (df.index.year * 10000 + df.index.month * 100 + df.index.day).astype(np.int64)
    daily_close = df.groupby('date_int')['Close'].last().sort_index()
    returns = daily_close.pct_change().dropna()
    rolling_vol = returns.rolling(VOL_LOOKBACK, min_periods=max(5, VOL_LOOKBACK // 2)).std()
    median_vol = rolling_vol.median()
    if median_vol == 0 or pd.isna(median_vol):
        return np.ones(len(ohlcv_dates), dtype=np.float64)
    vol_ratio = (rolling_vol / median_vol).clip(VOL_SCALE_MIN, VOL_SCALE_MAX)
    vol_by_date = vol_ratio.to_dict()
    vol_scales = np.ones(len(ohlcv_dates), dtype=np.float64)
    for i in range(len(ohlcv_dates)):
        d = int(ohlcv_dates[i])
        if d in vol_by_date:
            v = vol_by_date[d]
            if not np.isnan(v):
                vol_scales[i] = v
    return vol_scales


def compute_ensemble_scores(daily_scores, preset_name):
    preset = next((p for p in ENSEMBLE_PRESETS if p['name'] == preset_name), ENSEMBLE_PRESETS[0])
    weights = preset['weights']
    use_max = weights is None
    all_dates = set()
    for h in SCORE_HORIZONS:
        all_dates.update(daily_scores.get(h, {}).keys())
    blended = {}
    for d in all_dates:
        scores = [daily_scores.get(h, {}).get(d, 0.0) for h in SCORE_HORIZONS]
        blended[d] = max(scores) if use_max else sum(w * s for w, s in zip(weights, scores))
    return blended


def extract_ensemble_preset(experiment_str):
    for p in ENSEMBLE_PRESETS:
        if f"ens_{p['name']}" in experiment_str:
            return p['name']
    return 'equal'


# ==============================================================================
# OHLCV DATA LOADING
# ==============================================================================

def _load_parquet_with_dt_index(gcs_uri, tz_out='America/Los_Angeles'):
    df = pd.read_parquet(gcs_uri).reset_index()
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
        'prices': df['Close'].values.copy(), 'highs': df['High'].values.copy(),
        'lows': df['Low'].values.copy(), 'dates': dates, 'times': times,
    }


# ==============================================================================
# SCORING PHASE (one pass for both OLD and NEW)
# ==============================================================================

def score_model(all_tickers):
    print(f"\n{'='*W}")
    print(f"  SCORING — L5-1 (sector-aware FiLM, 22 regime features)")
    print(f"{'='*W}")

    probe_path = ensure_data(f'fold_{FOLD_ID}_val.npz')
    probe = np.load(probe_path)
    n_asset = probe['X_seq'].shape[-1]
    n_regime = probe['X_regime'].shape[-1]
    probe.close()
    print(f"  N_ASSET={n_asset}, N_REGIME={n_regime}")

    model = build_spike_transformer(n_asset, n_regime)
    weights_path = ensure_model(FOLD_ID)
    model.load_weights(weights_path)
    print(f"  Model loaded: {model.count_params():,} params")

    meta_path = ensure_data('meta.npz')
    meta = np.load(meta_path, allow_pickle=True)
    ticker_names = meta['ticker_names']

    all_scores = {}
    for split in ['val', 'test']:
        path = ensure_data(f'fold_{FOLD_ID}_{split}.npz')
        data = np.load(path)
        X_seq = data['X_seq']; X_reg = data['X_regime']
        X_id = data['X_id'].flatten(); date_ints = data['date_ints']
        minutes = data['minutes']; data.close()

        entry_mask = (minutes == SCORE_MINUTE)
        X_seq = X_seq[entry_mask]; X_reg = X_reg[entry_mask]
        X_id = X_id[entry_mask]; date_ints = date_ints[entry_mask]
        del minutes, entry_mask; gc.collect()

        print(f"  {split}: scoring {len(X_id):,} windows...")
        preds = model.predict(
            {'seq_input': X_seq.astype(np.float32),
             'regime_input': X_reg.astype(np.float32)},
            batch_size=512, verbose=0)
        del X_seq, X_reg; gc.collect()

        for i in range(len(X_id)):
            tid = int(X_id[i]); d_int = int(date_ints[i])
            if tid >= len(ticker_names): continue
            tname = str(ticker_names[tid])
            if tname in EXCLUDE_TICKERS or tname not in all_tickers: continue
            if tname not in all_scores:
                all_scores[tname] = {h: {} for h in SCORE_HORIZONS}
            for h in SCORE_HORIZONS:
                all_scores[tname][h][d_int] = float(preds[i, h])
        del preds, X_id, date_ints; gc.collect()

    del model; tf.keras.backend.clear_session(); gc.collect()
    try: ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception: pass
    print(f"  Done: {len(all_scores)} tickers scored")
    return all_scores


# ==============================================================================
# BQ CONFIG LOADER
# ==============================================================================

def load_configs_from_bq():
    query = f"""
    SELECT ticker, prob_thresh, tp_pct, sl_pct, tpsl_mode, experiment,
           grace_minutes, dip_pct, max_wait_minutes,
           val_b_avg_pnl, val_b_win_rate, val_b_count
    FROM `{BQ_TABLE}`
    WHERE model_version = '{MODEL_VERSION}'
      AND val_b_avg_pnl >= {DEPLOY_PNL_THRESHOLD}
      AND val_b_count >= 2
    ORDER BY val_b_avg_pnl DESC
    """
    print(f"\n  Querying BQ for ALL L5-1 configs (model_version='{MODEL_VERSION}')...")
    df = bq_client.query(query).to_dataframe()
    if len(df) == 0:
        print("  WARNING: No configs found in BQ!")
        return {}, {}, {}

    old_configs = {}
    new_configs = {}
    combined_configs = {}

    for _, row in df.iterrows():
        ticker = row['ticker']
        mode = row['tpsl_mode']
        experiment = row.get('experiment', '')
        h_idx = 1
        if mode == 'ensemble':
            pass
        else:
            for h_key, h_val in HORIZON_LABELS.items():
                if h_val in experiment:
                    h_idx = h_key; break

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
            if pd.notna(row.get('dip_pct')): cfg['dip_pct'] = float(row['dip_pct'])
            if pd.notna(row.get('max_wait_minutes')): cfg['max_wait'] = int(row['max_wait_minutes'])
        if mode == 'ensemble':
            cfg['ensemble_preset'] = extract_ensemble_preset(experiment)

        if mode in OLD_MODES and ticker not in old_configs:
            old_configs[ticker] = cfg
        if mode in NEW_MODES and ticker not in new_configs:
            new_configs[ticker] = cfg
        if ticker not in combined_configs:
            combined_configs[ticker] = cfg

    print(f"  OLD (static/grace/limit_entry): {len(old_configs)} tickers")
    print(f"  NEW (volscaled/ensemble):       {len(new_configs)} tickers")
    print(f"  COMBINED (best across all):     {len(combined_configs)} tickers")

    for label, configs in [('OLD', old_configs), ('NEW', new_configs)]:
        print(f"\n  {label} configs:")
        for t, c in sorted(configs.items(), key=lambda x: -x[1]['val_b_avg_pnl']):
            h = HORIZON_LABELS.get(c['horizon'], c.get('ensemble_preset', '?'))
            print(f"    {t:<7} mode={c['mode']:<13} H={h}  "
                  f"TEST={c['val_b_avg_pnl']:+.2%} WR={c['val_b_win_rate']:.0%} N={c['val_b_count']}")

    return old_configs, new_configs, combined_configs


# ==============================================================================
# BACKTEST RUNNER
# ==============================================================================

def run_backtest(all_scores, deploy_configs, label):
    tickers = sorted(deploy_configs.keys())
    if not tickers:
        print(f"\n  {label}: no tickers")
        return []

    print(f"\n{'─'*W}")
    print(f"  BACKTEST — {label} ({len(tickers)} tickers)")
    print(f"{'─'*W}")

    all_trades = []
    for ticker in tickers:
        cfg = deploy_configs[ticker]
        mode = cfg['mode']

        scores = all_scores.get(ticker, {})
        if not scores:
            print(f"  {ticker}: no scores"); continue

        if mode == 'ensemble':
            preset = cfg.get('ensemble_preset', 'equal')
            blended = compute_ensemble_scores(scores, preset)
            score_by_date = {int(d): s for d, s in blended.items()}
            h_label = f"ens_{preset}"
        else:
            h_idx = cfg['horizon']
            h_scores = scores.get(h_idx, {})
            score_by_date = {int(d): s for d, s in h_scores.items()}
            h_label = HORIZON_LABELS.get(h_idx, '?')

        print(f"  {ticker} — H={h_label} mode={mode} prob>{cfg['prob_thresh']:.2f} "
              f"TP={cfg['tp_pct']:.0%} SL={cfg['sl_pct']:.0%}", end="")
        if mode == 'grace': print(f" grace={cfg.get('grace_minutes', 0)}m", end="")
        elif mode == 'limit_entry': print(f" rally={cfg.get('dip_pct',0):.1%} wait={cfg.get('max_wait',0)}m", end="")
        elif mode == 'volscaled': print(f" vol[{VOL_SCALE_MIN:.1f}-{VOL_SCALE_MAX:.1f}]", end="")
        elif mode == 'ensemble': print(f" preset={cfg.get('ensemble_preset','?')}", end="")
        print()

        try:
            df_ohlcv = load_ticker_ohlcv(ticker)
        except Exception as e:
            print(f"    OHLCV failed: {e}"); continue

        ohlcv = precompute_ohlcv_arrays(df_ohlcv, TEST_START, TEST_END)
        if ohlcv is None:
            print(f"    No OHLCV in test window"); del df_ohlcv; continue

        if mode == 'volscaled':
            ohlcv['vol_scales'] = compute_vol_scales(df_ohlcv, ohlcv['dates'])

        entry_list = []
        for i in range(len(ohlcv['dates'])):
            if ohlcv['times'][i] == ENTRY_MINUTE:
                day_score = score_by_date.get(int(ohlcv['dates'][i]), 0.0)
                if day_score >= cfg['prob_thresh']:
                    entry_list.append(i)

        if not entry_list:
            print(f"    No signals"); del df_ohlcv; continue

        indices = np.array(entry_list, dtype=np.int64)
        pnl, dates, outcomes = _run_sim(indices, ohlcv, cfg)

        if len(pnl) == 0:
            print(f"    No trades"); del df_ohlcv; continue

        avg_pnl = float(np.mean(pnl)); wr = float(np.mean(pnl > 0)); n = len(pnl)
        print(f"    {n} trades | Avg: {avg_pnl:+.2%} | WR: {wr:.0%} | Total: {float(np.sum(pnl)):+.2%}")

        for i in range(n):
            d = int(dates[i])
            date_str = f"{d // 10000}-{(d % 10000) // 100:02d}-{d % 100:02d}"
            otype = 'TP' if outcomes[i] == 1 else 'SL' if outcomes[i] == -1 else 'EOD'
            all_trades.append({
                'Ticker': ticker, 'Date': date_str, 'PnL': float(pnl[i]),
                'Outcome': otype, 'Mode': mode, 'Horizon': h_label, 'Strategy': label,
            })

        del df_ohlcv, ohlcv; gc.collect()

    return all_trades


# ==============================================================================
# ANALYSIS
# ==============================================================================

def analyze_strategy(trades, label):
    if not trades:
        return {'label': label, 'n': 0, 'avg_pnl': 0, 'total_pnl': 0,
                'win_rate': 0, 'n_tickers': 0, 'sharpe': 0}
    df = pd.DataFrame(trades)
    avg = df['PnL'].mean(); std = df['PnL'].std()
    return {
        'label': label, 'n': len(df), 'avg_pnl': avg,
        'total_pnl': df['PnL'].sum(),
        'win_rate': (df['PnL'] > 0).mean(),
        'n_tickers': df['Ticker'].nunique(),
        'avg_win': df.loc[df['PnL'] > 0, 'PnL'].mean() if (df['PnL'] > 0).any() else 0,
        'avg_loss': df.loc[df['PnL'] <= 0, 'PnL'].mean() if (df['PnL'] <= 0).any() else 0,
        'sharpe': (avg / std * np.sqrt(252)) if std > 0 else 0,
    }


# ==============================================================================
# MAIN
# ==============================================================================

print(f"\n{'='*W}")
print(f"  SHORT BACKTEST: L5-1 Old Optimization vs New Optimization")
print(f"  TEST: {TEST_START} -> {TEST_END}")
print(f"  Same model (sector-aware FiLM, 22 regime features)")
print(f"  OLD = static/grace/limit_entry  |  NEW = volscaled/ensemble")
print(f"  Deploy threshold: >{DEPLOY_PNL_THRESHOLD:.0%} TEST avg PnL")
print(f"  Slippage: {SLIPPAGE_PCT:.2%} each way ({2*SLIPPAGE_PCT:.2%} RT)")
print(f"{'='*W}")

old_configs, new_configs, combined_configs = load_configs_from_bq()

if not old_configs and not new_configs:
    print("\n  ERROR: No configs found. Run both optimizer passes first.")
    print("    1. ra_moe_short_L5_1_optimizer.py (SKIP_EXISTING_MODES=False) for static/grace/limit_entry")
    print("    2. ra_moe_short_L5_1_optimizer.py (SKIP_EXISTING_MODES=True) for volscaled/ensemble")
else:
    all_tickers = set(old_configs.keys()) | set(new_configs.keys())
    all_scores = score_model(all_tickers)

    old_trades = run_backtest(all_scores, old_configs, 'OLD')
    new_trades = run_backtest(all_scores, new_configs, 'NEW')
    combined_trades = run_backtest(all_scores, combined_configs, 'COMBINED')

    old_stats = analyze_strategy(old_trades, 'OLD')
    new_stats = analyze_strategy(new_trades, 'NEW')
    combined_stats = analyze_strategy(combined_trades, 'COMBINED')

    # ── Head-to-Head ──
    print(f"\n\n{'='*W}")
    print(f"  HEAD-TO-HEAD: OLD vs NEW vs COMBINED")
    print(f"  TEST: {TEST_START} -> {TEST_END}")
    print(f"{'='*W}\n")

    header = f"  {'Metric':<25} {'OLD':>12} {'NEW':>12} {'COMBINED':>12} {'NEW-OLD':>12}"
    print(header)
    print(f"  {'─'*73}")

    def _fp(v): return f"{v:+.2%}" if v != 0 else "—"

    delta_pnl = new_stats['avg_pnl'] - old_stats['avg_pnl'] if new_stats['n'] and old_stats['n'] else 0
    delta_wr = new_stats['win_rate'] - old_stats['win_rate'] if new_stats['n'] and old_stats['n'] else 0

    rows = [
        ('Tickers',     str(old_stats['n_tickers']),  str(new_stats['n_tickers']),
         str(combined_stats['n_tickers']), f"{new_stats['n_tickers'] - old_stats['n_tickers']:+d}"),
        ('Total trades', str(old_stats['n']),          str(new_stats['n']),
         str(combined_stats['n']),         f"{new_stats['n'] - old_stats['n']:+d}"),
        ('Avg PnL/trade', _fp(old_stats['avg_pnl']),  _fp(new_stats['avg_pnl']),
         _fp(combined_stats['avg_pnl']),   _fp(delta_pnl)),
        ('Total PnL',    _fp(old_stats['total_pnl']), _fp(new_stats['total_pnl']),
         _fp(combined_stats['total_pnl']), _fp(new_stats['total_pnl'] - old_stats['total_pnl'])),
        ('Win rate',     f"{old_stats['win_rate']:.1%}" if old_stats['n'] else "—",
         f"{new_stats['win_rate']:.1%}" if new_stats['n'] else "—",
         f"{combined_stats['win_rate']:.1%}" if combined_stats['n'] else "—",
         _fp(delta_wr)),
        ('Sharpe (ann.)', f"{old_stats['sharpe']:.1f}" if old_stats['n'] else "—",
         f"{new_stats['sharpe']:.1f}" if new_stats['n'] else "—",
         f"{combined_stats['sharpe']:.1f}" if combined_stats['n'] else "—", ""),
    ]
    for label, v_old, v_new, v_comb, v_delta in rows:
        print(f"  {label:<25} {v_old:>12} {v_new:>12} {v_comb:>12} {v_delta:>12}")

    # ── Per-ticker comparison ──
    all_bt_tickers = sorted(set(
        [t['Ticker'] for t in old_trades] + [t['Ticker'] for t in new_trades]))

    if all_bt_tickers:
        old_df  = pd.DataFrame(old_trades) if old_trades else pd.DataFrame(columns=['Ticker','PnL','Mode'])
        new_df  = pd.DataFrame(new_trades) if new_trades else pd.DataFrame(columns=['Ticker','PnL','Mode'])
        comb_df = pd.DataFrame(combined_trades) if combined_trades else pd.DataFrame(columns=['Ticker','PnL','Mode'])

        print(f"\n\n  PER-TICKER COMPARISON:")
        print(f"  {'Ticker':<7} {'OLD Mode':>12} {'OLD Avg':>8} {'NEW Mode':>12} {'NEW Avg':>8} "
              f"{'Delta':>8} {'Winner':>10} {'COMB Mode':>12}")
        print(f"  {'─'*90}")

        ticker_comparison = []
        for ticker in all_bt_tickers:
            t_old = old_df[old_df['Ticker'] == ticker] if len(old_df) > 0 else pd.DataFrame()
            t_new = new_df[new_df['Ticker'] == ticker] if len(new_df) > 0 else pd.DataFrame()
            t_comb = comb_df[comb_df['Ticker'] == ticker] if len(comb_df) > 0 else pd.DataFrame()
            n_old = len(t_old); avg_old = t_old['PnL'].mean() if n_old > 0 else 0
            n_new = len(t_new); avg_new = t_new['PnL'].mean() if n_new > 0 else 0
            delta = avg_new - avg_old

            old_mode = t_old['Mode'].iloc[0] if n_old > 0 else "—"
            new_mode = t_new['Mode'].iloc[0] if n_new > 0 else "—"
            comb_mode = t_comb['Mode'].iloc[0] if len(t_comb) > 0 else "—"

            if n_old == 0:
                winner = "NEW ONLY"
            elif n_new == 0:
                winner = "OLD ONLY"
            elif avg_new > avg_old:
                winner = "NEW"
            elif avg_old > avg_new:
                winner = "OLD"
            else:
                winner = "TIE"

            tag = ""
            if n_old == 0: tag = " ← NEW"
            elif n_new == 0: tag = " ← OLD"

            print(f"  {ticker:<7} {old_mode:>12} {avg_old:>+7.2%} {new_mode:>12} {avg_new:>+7.2%} "
                  f"{delta:>+7.2%} {winner:>10}{tag} {comb_mode:>12}")

            ticker_comparison.append({
                'Ticker': ticker, 'old_n': n_old, 'old_avg': avg_old,
                'new_n': n_new, 'new_avg': avg_new, 'delta': delta,
                'winner': winner, 'old_mode': old_mode, 'new_mode': new_mode,
                'comb_mode': comb_mode,
            })

        overlap = [tc for tc in ticker_comparison if tc['old_n'] > 0 and tc['new_n'] > 0]
        old_only = [tc for tc in ticker_comparison if tc['new_n'] == 0]
        new_only = [tc for tc in ticker_comparison if tc['old_n'] == 0]
        new_wins = [tc for tc in overlap if tc['delta'] > 0]

        print(f"\n  OVERLAP ANALYSIS:")
        print(f"    Tickers in both:      {len(overlap)}")
        print(f"    NEW wins (overlap):   {len(new_wins)}/{len(overlap)}")
        print(f"    OLD-only (not in NEW): {len(old_only)} ({', '.join(t['Ticker'] for t in old_only) or 'none'})")
        print(f"    NEW-only (not in OLD): {len(new_only)} ({', '.join(t['Ticker'] for t in new_only) or 'none'})")
        if overlap:
            avg_delta = np.mean([t['delta'] for t in overlap])
            print(f"    Avg delta (overlap):  {avg_delta:+.2%}")

        # ── Mode distribution in COMBINED ──
        if combined_trades:
            mode_counts = comb_df.groupby('Mode')['Ticker'].nunique()
            print(f"\n  COMBINED MODE DISTRIBUTION (which mode won per ticker):")
            for mode, count in mode_counts.items():
                print(f"    {mode:<13}: {count} tickers")

    # ── Charts ──
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'SHORT L5-1: Old Optimization vs New Optimization\n'
                 f'TEST: {TEST_START} → {TEST_END} | Same model, different optimizer modes',
                 fontsize=14, fontweight='bold')

    COLORS = {'OLD': '#1565C0', 'NEW': '#E65100', 'COMBINED': '#2E7D32'}

    # [0,0] Cumulative PnL
    ax = axes[0, 0]
    for trades, label, color in [(old_trades, 'OLD', COLORS['OLD']),
                                  (new_trades, 'NEW', COLORS['NEW']),
                                  (combined_trades, 'COMBINED', COLORS['COMBINED'])]:
        if not trades: continue
        df = pd.DataFrame(trades).sort_values('Date')
        cum = df['PnL'].cumsum().values
        ax.plot(range(1, len(cum)+1), cum * 100, color=color, linewidth=2, label=label)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Trade #'); ax.set_ylabel('Cumulative PnL (%)')
    ax.set_title('Cumulative PnL'); ax.legend(); ax.grid(alpha=0.3)

    # [0,1] Daily PnL
    ax = axes[0, 1]
    for trades, label, color, offset in [(old_trades, 'OLD', COLORS['OLD'], -0.3),
                                          (new_trades, 'NEW', COLORS['NEW'], 0.0),
                                          (combined_trades, 'COMBINED', COLORS['COMBINED'], 0.3)]:
        if not trades: continue
        df = pd.DataFrame(trades)
        daily = df.groupby('Date')['PnL'].sum()
        dates = pd.to_datetime(daily.index)
        ax.bar(dates + timedelta(hours=int(offset * 12)), daily.values * 100,
               width=0.3, color=color, alpha=0.7, label=label, edgecolor='white', linewidth=0.3)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.set_ylabel('Daily Total PnL (%)'); ax.set_title('Daily PnL')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # [0,2] Per-ticker paired bars (OLD vs NEW)
    ax = axes[0, 2]
    if all_bt_tickers and ticker_comparison:
        tc_df = pd.DataFrame(ticker_comparison).sort_values('delta', ascending=True)
        y_pos = np.arange(len(tc_df)); bar_h = 0.35
        ax.barh(y_pos - bar_h/2, tc_df['old_avg'].values * 100,
                height=bar_h, color=COLORS['OLD'], alpha=0.8, label='OLD')
        ax.barh(y_pos + bar_h/2, tc_df['new_avg'].values * 100,
                height=bar_h, color=COLORS['NEW'], alpha=0.8, label='NEW')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(tc_df['Ticker'].values, fontsize=8)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel('Avg PnL per Trade (%)'); ax.set_title('Per-Ticker Head-to-Head')
        ax.legend(fontsize=8); ax.grid(axis='x', alpha=0.3)

    # [1,0] Delta chart
    ax = axes[1, 0]
    if all_bt_tickers and ticker_comparison:
        tc_df = pd.DataFrame(ticker_comparison).sort_values('delta', ascending=True)
        colors_bar = ['#2E7D32' if d > 0 else '#C62828' for d in tc_df['delta']]
        ax.barh(range(len(tc_df)), tc_df['delta'].values * 100,
                color=colors_bar, edgecolor='white', linewidth=0.5)
        ax.set_yticks(range(len(tc_df)))
        ax.set_yticklabels(tc_df['Ticker'].values, fontsize=8)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel('NEW minus OLD (pp)')
        ax.set_title('Per-Ticker Delta (green = NEW better)')
        ax.grid(axis='x', alpha=0.3)

    # [1,1] Summary table
    ax = axes[1, 1]; ax.axis('off')
    table_data = [
        ['Metric', 'OLD', 'NEW', 'COMBINED'],
        ['Tickers', str(old_stats['n_tickers']), str(new_stats['n_tickers']),
         str(combined_stats['n_tickers'])],
        ['Trades', str(old_stats['n']), str(new_stats['n']), str(combined_stats['n'])],
        ['Avg PnL', f"{old_stats['avg_pnl']:+.2%}", f"{new_stats['avg_pnl']:+.2%}",
         f"{combined_stats['avg_pnl']:+.2%}"],
        ['Total PnL', f"{old_stats['total_pnl']:+.2%}", f"{new_stats['total_pnl']:+.2%}",
         f"{combined_stats['total_pnl']:+.2%}"],
        ['Win Rate', f"{old_stats['win_rate']:.1%}", f"{new_stats['win_rate']:.1%}",
         f"{combined_stats['win_rate']:.1%}"],
        ['Sharpe', f"{old_stats['sharpe']:.1f}", f"{new_stats['sharpe']:.1f}",
         f"{combined_stats['sharpe']:.1f}"],
    ]
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 1.5)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor('#BBDEFB'); cell.set_text_props(fontweight='bold')
        elif c == 1: cell.set_facecolor('#E3F2FD')
        elif c == 2: cell.set_facecolor('#FFF3E0')
        elif c == 3: cell.set_facecolor('#E8F5E9')
    ax.set_title('Portfolio Summary', pad=20)

    # [1,2] Mode distribution in COMBINED
    ax = axes[1, 2]
    if combined_trades:
        comb_df = pd.DataFrame(combined_trades)
        mode_ticker_counts = comb_df.groupby('Mode')['Ticker'].nunique()
        colors_pie = {'static': '#1565C0', 'grace': '#42A5F5', 'limit_entry': '#90CAF9',
                      'volscaled': '#E65100', 'ensemble': '#FF9800'}
        pie_colors = [colors_pie.get(m, '#999') for m in mode_ticker_counts.index]
        ax.pie(mode_ticker_counts.values, labels=mode_ticker_counts.index,
               colors=pie_colors, autopct='%1.0f%%', startangle=90)
        ax.set_title('COMBINED: Which mode won\n(tickers per mode)')
    else:
        ax.axis('off')

    plt.tight_layout()
    chart_path = '/tmp/backtest_L5_1_old_vs_new.png'
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    print(f"\n  Chart saved: {chart_path}")
    if _IN_COLAB: plt.show()
    plt.close()


    # ── Verdict ──
    print(f"\n\n{'='*W}")
    print(f"  VERDICT")
    print(f"{'='*W}")

    if not new_trades:
        print(f"  NEW has no trades — run the optimizer with SKIP_EXISTING_MODES=True first.")
    elif not old_trades:
        print(f"  OLD has no trades — run the optimizer with SKIP_EXISTING_MODES=False first.")
    else:
        if new_stats['avg_pnl'] > old_stats['avg_pnl']:
            delta = new_stats['avg_pnl'] - old_stats['avg_pnl']
            print(f"  NEW optimization WINS by {delta:+.2%}/trade")
        elif old_stats['avg_pnl'] > new_stats['avg_pnl']:
            delta = old_stats['avg_pnl'] - new_stats['avg_pnl']
            print(f"  OLD optimization WINS by {delta:+.2%}/trade")
        else:
            print(f"  TIE — no measurable difference")

        print(f"  OLD:      {old_stats['avg_pnl']:+.2%}/trade ({old_stats['n']} trades, {old_stats['n_tickers']} tickers)")
        print(f"  NEW:      {new_stats['avg_pnl']:+.2%}/trade ({new_stats['n']} trades, {new_stats['n_tickers']} tickers)")
        print(f"  COMBINED: {combined_stats['avg_pnl']:+.2%}/trade ({combined_stats['n']} trades, {combined_stats['n_tickers']} tickers)")

        if combined_stats['avg_pnl'] > max(old_stats['avg_pnl'], new_stats['avg_pnl']):
            print(f"\n  COMBINED is best — new modes add value on top of existing ones.")
            print(f"  RECOMMENDATION: Use COMBINED configs for deployment")
        elif combined_stats['n_tickers'] > max(old_stats['n_tickers'], new_stats['n_tickers']):
            print(f"\n  COMBINED has wider coverage — new modes capture additional tickers.")
            print(f"  RECOMMENDATION: Use COMBINED configs for deployment")
        else:
            winner = 'NEW' if new_stats['avg_pnl'] > old_stats['avg_pnl'] else 'OLD'
            print(f"\n  {winner} is sufficient — COMBINED doesn't improve further.")

    print(f"\n{'='*W}")
