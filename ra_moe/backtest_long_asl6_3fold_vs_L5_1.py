# %% ── LONG Hybrid vs Hybrid: L5-1 (3 folds) vs ASL6 (3 folds) ───────────────
#
# Both families have 3 folds each, built as F5>F6>F7 hybrids.
# This script compares the two hybrids head-to-head, plus combination strategies.
#
# L5-1 FAMILY (16 regime, ASL γ=5.0/clip=0.10):
#   L5-1 F5: test Jul 10-Aug 12, BQ 'spike_transformer_long_L5_1_fold5'
#   L5-1 F6: test Jul 22-Aug 12, BQ 'spike_transformer_long_L5_1_fold6'
#   L5-1 F7: test Jul 29-Aug 12, BQ 'spike_transformer_long_L5_1_fold7'
#
# ASL6 FAMILY (22 regime, ASL γ=6.0/clip=0.08):
#   ASL6 F5: test Jul 15-Aug 17, BQ 'spike_transformer_long_L5_2_asl6_fold5'
#   ASL6 F6: test Jul 29-Aug 17, BQ 'spike_transformer_long_L5_2_asl6_fold6'
#   ASL6 F7: test Aug  3-Aug 17, BQ 'spike_transformer_long_L5_2_asl6_fold7'
#
# STRATEGIES TESTED:
#   1. L5-1 Hybrid   — L5-1 F5>F6>F7 [DEPLOYED]
#   2. ASL6 Hybrid   — ASL6 F5>F6>F7
#   3. Union L5-1>   — L5-1 Hybrid for shared, + ASL6-unique
#   4. Union ASL6>   — ASL6 Hybrid for shared, + L5-1-unique
#   5. Best Val-B    — per-ticker: pick model with highest val_b_avg_pnl
#   6. Newest First  — all F7>all F6>all F5 (recency)
#   7. Oracle        — per-ticker best ACTUAL backtest PnL (upper bound)
#
# OVERLAP WINDOW: Aug 3 → Aug 12 (common unseen across all 6 models)
# EW Portfolio: daily_ew = df.groupby('Date')['PnL'].mean()
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

OVERLAP_START  = '2026-08-03'
OVERLAP_END    = '2026-08-12'
OVERLAP_DAYS   = 9

SCORE_MINUTE   = 375
ENTRY_MINUTE   = 385
EXIT_TIME_MIN  = 780
SLIPPAGE_PCT   = 0.0015

DEPLOY_PNL_THRESHOLD = 0.018

GCS_BUCKET    = os.environ.get("GCS_BUCKET", "")
GCS_INFER_DIR = 'Data/Inference_v2'

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

MIN_TRADES_PER_MONTH = 2.0

# -- All 6 models: 2 families × 3 folds --
MODELS = {
    'L5-1 F5': {
        'gcs_data':    'Algos/RA_MoE/long_L5_1_data',
        'gcs_models':  'Algos/RA_MoE/long_L5_1_models',
        'data_dir':    '/tmp/ra_moe_long_L5_1_data',
        'model_dir':   '/tmp/ra_moe_long_L5_1_models',
        'mv':          'spike_transformer_long_L5_1_fold5',
        'fold_id':     5,
        'family':      'L5-1',
        'test_days':   33,
        'color':       '#C62828',
        'desc':        '16 regime, ASL γ=5.0, test Jul10-Aug12',
    },
    'L5-1 F6': {
        'gcs_data':    'Algos/RA_MoE/long_L5_1_data',
        'gcs_models':  'Algos/RA_MoE/long_L5_1_models',
        'data_dir':    '/tmp/ra_moe_long_L5_1_data',
        'model_dir':   '/tmp/ra_moe_long_L5_1_models',
        'mv':          'spike_transformer_long_L5_1_fold6',
        'fold_id':     6,
        'family':      'L5-1',
        'test_days':   21,
        'color':       '#E53935',
        'desc':        '16 regime, ASL γ=5.0, test Jul22-Aug12',
    },
    'L5-1 F7': {
        'gcs_data':    'Algos/RA_MoE/long_L5_1_data',
        'gcs_models':  'Algos/RA_MoE/long_L5_1_models',
        'data_dir':    '/tmp/ra_moe_long_L5_1_data',
        'model_dir':   '/tmp/ra_moe_long_L5_1_models',
        'mv':          'spike_transformer_long_L5_1_fold7',
        'fold_id':     7,
        'family':      'L5-1',
        'test_days':   14,
        'color':       '#EF5350',
        'desc':        '16 regime, ASL γ=5.0, test Jul29-Aug12',
    },
    'ASL6 F5': {
        'gcs_data':    'Algos/RA_MoE/long_L5_2_data',
        'gcs_models':  'Algos/RA_MoE/long_L5_2_asl6_models',
        'data_dir':    '/tmp/ra_moe_long_L5_2_data',
        'model_dir':   '/tmp/ra_moe_long_L5_2_asl6_models',
        'mv':          'spike_transformer_long_L5_2_asl6_fold5',
        'fold_id':     5,
        'family':      'ASL6',
        'test_days':   33,
        'color':       '#1565C0',
        'desc':        '22 regime, ASL γ=6.0, test Jul15-Aug17',
    },
    'ASL6 F6': {
        'gcs_data':    'Algos/RA_MoE/long_L5_2_data',
        'gcs_models':  'Algos/RA_MoE/long_L5_2_asl6_models',
        'data_dir':    '/tmp/ra_moe_long_L5_2_data',
        'model_dir':   '/tmp/ra_moe_long_L5_2_asl6_models',
        'mv':          'spike_transformer_long_L5_2_asl6_fold6',
        'fold_id':     6,
        'family':      'ASL6',
        'test_days':   19,
        'color':       '#E65100',
        'desc':        '22 regime, ASL γ=6.0, test Jul29-Aug17',
    },
    'ASL6 F7': {
        'gcs_data':    'Algos/RA_MoE/long_L5_2_data',
        'gcs_models':  'Algos/RA_MoE/long_L5_2_asl6_models',
        'data_dir':    '/tmp/ra_moe_long_L5_2_data',
        'model_dir':   '/tmp/ra_moe_long_L5_2_asl6_models',
        'mv':          'spike_transformer_long_L5_2_asl6_fold7',
        'fold_id':     7,
        'family':      'ASL6',
        'test_days':   14,
        'color':       '#7B1FA2',
        'desc':        '22 regime, ASL γ=6.0, test Aug3-Aug17',
    },
}

L5_1_KEYS = ['L5-1 F5', 'L5-1 F6', 'L5-1 F7']
ASL6_KEYS = ['ASL6 F5', 'ASL6 F6', 'ASL6 F7']

MODEL_KEYS = list(MODELS.keys())

for mcfg in MODELS.values():
    os.makedirs(mcfg['data_dir'], exist_ok=True)
    os.makedirs(mcfg['model_dir'], exist_ok=True)


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
        self.tau_proj = layers.Dense(n_heads); self.sigma_proj = layers.Dense(d_model)

    def call(self, x, stats, training=False):
        B = tf.shape(x)[0]; T = tf.shape(x)[1]; H, D = self.n_heads, self.d_head
        Q = tf.transpose(tf.reshape(self.wq(x), (B, T, H, D)), (0, 2, 1, 3))
        K = tf.transpose(tf.reshape(self.wk(x), (B, T, H, D)), (0, 2, 1, 3))
        V = tf.transpose(tf.reshape(self.wv(x), (B, T, H, D)), (0, 2, 1, 3))
        scores = tf.matmul(Q, K, transpose_b=True) / self.scale
        scores = scores + tf.reshape(self.tau_proj(stats), (B, H, 1, 1))
        attn = self.attn_drop(tf.nn.softmax(scores, axis=-1), training=training)
        out = tf.reshape(tf.transpose(tf.matmul(attn, V), (0, 2, 1, 3)), (B, T, self.d_model))
        out = out * tf.nn.softplus(self.sigma_proj(stats))[:, tf.newaxis, :]
        return self.wo(out)

class TransformerBlock(layers.Layer):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.attn = DeStationaryAttention(d_model, n_heads, dropout)
        self.ffn = tf.keras.Sequential([layers.Dense(d_ff, activation='gelu'),
                                         layers.Dense(d_model), layers.Dropout(dropout)])
        self.norm1 = layers.LayerNormalization(); self.norm2 = layers.LayerNormalization()
        self.drop = layers.Dropout(dropout)
    def call(self, x, stats, training=False):
        x = x + self.drop(self.attn(self.norm1(x), stats, training=training), training=training)
        return x + self.ffn(self.norm2(x), training=training)

class TransformerBackbone(layers.Layer):
    def __init__(self, d_model, n_heads, d_ff, n_blocks, seq_len, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.stationarize = SeriesStationarization()
        self.input_proj = layers.Dense(d_model)
        self.pos_embed = self.add_weight(name='pos_embed', shape=(1, seq_len, d_model),
                                          initializer='glorot_normal', trainable=True)
        self.input_drop = layers.Dropout(dropout)
        self.blocks = [TransformerBlock(d_model, n_heads, d_ff, dropout, name=f'block_{i}')
                       for i in range(n_blocks)]
        self.final_norm = layers.LayerNormalization()
    def call(self, x, training=False):
        x_norm, stats = self.stationarize(x)
        h = self.input_drop(self.input_proj(x_norm) + self.pos_embed, training=training)
        for block in self.blocks: h = block(h, stats, training=training)
        return self.final_norm(h)

class AttentionPooling(layers.Layer):
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.query = self.add_weight(name='pool_query', shape=(1, 1, d_model),
                                      initializer='glorot_normal', trainable=True)
        self.scale = tf.sqrt(tf.cast(d_model, tf.float32))
    def call(self, x, training=False):
        w = tf.nn.softmax(tf.matmul(self.query, x, transpose_b=True) / self.scale, axis=-1)
        return tf.matmul(w, x)[:, 0, :]

class FiLMConditioner(layers.Layer):
    def __init__(self, d_model, hidden_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.mlp = tf.keras.Sequential([layers.Dense(hidden_dim, activation='relu'),
                                         layers.Dense(hidden_dim, activation='relu')])
        self.gamma_proj = layers.Dense(d_model); self.beta_proj = layers.Dense(d_model)
    def call(self, x, regime_features, training=False):
        h = self.mlp(regime_features, training=training)
        return (1.0 + self.gamma_proj(h)) * x + self.beta_proj(h)

def build_spike_transformer(n_asset, n_regime, name='st'):
    class _ST(Model):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.backbone = TransformerBackbone(D_MODEL, N_HEADS, D_FF, N_TRANSFORMER_BLOCKS,
                                                 TIME_STEPS, DROPOUT, name='backbone')
            self.attn_pool = AttentionPooling(D_MODEL, name='attn_pool')
            self.film = FiLMConditioner(D_MODEL, FILM_HIDDEN, name='film')
            self.drop = layers.Dropout(DROPOUT)
            self.head = layers.Dense(N_HORIZONS, activation='sigmoid', name='spike_head')
        def call(self, inputs, training=False):
            h = self.backbone(inputs['seq_input'], training=training)
            h = self.film(self.attn_pool(h, training=training),
                          inputs['regime_input'], training=training)
            return self.head(self.drop(h, training=training))
    m = _ST(name=name)
    m({'seq_input': tf.zeros((1, TIME_STEPS, n_asset)),
       'regime_input': tf.zeros((1, n_regime))})
    return m


# ==============================================================================
# LONG TRADE SIMS (Numba) — TP above, SL below, PnL = (exit-entry)/entry
# ==============================================================================

@jit(nopython=True)
def run_trade_sim(entry_indices, prices, highs, lows, date_ints, time_mins,
                  tp, sl, exit_time):
    ne = len(entry_indices); np_ = len(prices)
    pnl_a = np.empty(ne, np.float64); date_a = np.empty(ne, np.int64)
    entry_a = np.empty(ne, np.float64); exit_a = np.empty(ne, np.float64)
    out_a = np.empty(ne, np.int64); tc = 0; cp = 0; ltd = -1
    while cp < ne:
        idx = entry_indices[cp]
        if idx >= np_ - 1: break
        ed = date_ints[idx]
        if ed == ltd: cp += 1; continue
        ep = prices[idx]; tgt = ep * (1 + tp); stp = ep * (1 - sl)
        oc = 0.0; xp = ep; xt = 0; done = False; xi = idx
        for k in range(idx + 1, np_):
            if date_ints[k] != ed:
                xp = prices[k-1]; oc = (xp - ep) / ep; xi = k; done = True; break
            sl_h = lows[k] <= stp; tp_h = highs[k] >= tgt
            if sl_h and tp_h:
                if np.random.random() < 0.5: oc = -sl; xp = stp; xt = -1
                else: oc = tp; xp = tgt; xt = 1
                xi = k; done = True; break
            if sl_h: oc = -sl; xp = stp; xt = -1; xi = k; done = True; break
            if tp_h: oc = tp; xp = tgt; xt = 1; xi = k; done = True; break
            if time_mins[k] >= exit_time:
                xp = prices[k]; oc = (xp - ep) / ep; xi = k; done = True; break
        if not done: xp = prices[np_-1]; oc = (xp - ep) / ep; xi = np_
        pnl_a[tc] = oc; date_a[tc] = ed; entry_a[tc] = ep; exit_a[tc] = xp; out_a[tc] = xt
        tc += 1; ltd = ed; nxt = cp + 1
        while nxt < ne:
            if entry_indices[nxt] > xi: break
            nxt += 1
        cp = nxt
    return pnl_a[:tc], date_a[:tc], entry_a[:tc], exit_a[:tc], out_a[:tc]

@jit(nopython=True)
def run_trade_sim_grace(entry_indices, prices, highs, lows, date_ints, time_mins,
                        tp, sl, exit_time, grace_minutes):
    ne = len(entry_indices); np_ = len(prices)
    pnl_a = np.empty(ne, np.float64); date_a = np.empty(ne, np.int64)
    entry_a = np.empty(ne, np.float64); exit_a = np.empty(ne, np.float64)
    out_a = np.empty(ne, np.int64); tc = 0; cp = 0; ltd = -1
    while cp < ne:
        idx = entry_indices[cp]
        if idx >= np_ - 1: break
        ed = date_ints[idx]
        if ed == ltd: cp += 1; continue
        ep = prices[idx]; tgt = ep * (1 + tp); stp = ep * (1 - sl)
        et = time_mins[idx]; oc = 0.0; xp = ep; xt = 0; done = False; xi = idx
        for k in range(idx + 1, np_):
            if date_ints[k] != ed:
                xp = prices[k-1]; oc = (xp - ep) / ep; xi = k; done = True; break
            elapsed = time_mins[k] - et; sl_act = elapsed >= grace_minutes
            tp_h = highs[k] >= tgt; sl_h = sl_act and (lows[k] <= stp)
            if sl_h and tp_h:
                if np.random.random() < 0.5: oc = -sl; xp = stp; xt = -1
                else: oc = tp; xp = tgt; xt = 1
                xi = k; done = True; break
            if sl_h: oc = -sl; xp = stp; xt = -1; xi = k; done = True; break
            if tp_h: oc = tp; xp = tgt; xt = 1; xi = k; done = True; break
            if time_mins[k] >= exit_time:
                xp = prices[k]; oc = (xp - ep) / ep; xi = k; done = True; break
        if not done: xp = prices[np_-1]; oc = (xp - ep) / ep; xi = np_
        pnl_a[tc] = oc; date_a[tc] = ed; entry_a[tc] = ep; exit_a[tc] = xp; out_a[tc] = xt
        tc += 1; ltd = ed; nxt = cp + 1
        while nxt < ne:
            if entry_indices[nxt] > xi: break
            nxt += 1
        cp = nxt
    return pnl_a[:tc], date_a[:tc], entry_a[:tc], exit_a[:tc], out_a[:tc]

@jit(nopython=True)
def run_trade_sim_limit_entry(entry_indices, prices, highs, lows, date_ints, time_mins,
                              tp, sl, exit_time, dip_pct, max_wait_minutes):
    ne = len(entry_indices); np_ = len(prices)
    pnl_a = np.empty(ne, np.float64); date_a = np.empty(ne, np.int64)
    entry_a = np.empty(ne, np.float64); exit_a = np.empty(ne, np.float64)
    out_a = np.empty(ne, np.int64); tc = 0; cp = 0; ltd = -1
    while cp < ne:
        idx = entry_indices[cp]
        if idx >= np_ - 1: break
        ed = date_ints[idx]
        if ed == ltd: cp += 1; continue
        sp = prices[idx]; lp = sp * (1.0 - dip_pct); st = time_mins[idx]
        fi = -1; fp = 0.0
        for k in range(idx + 1, np_):
            if date_ints[k] != ed: break
            if time_mins[k] - st > max_wait_minutes: break
            if lows[k] <= lp: fp = lp; fi = k; break
        if fi < 0: cp += 1; ltd = ed; continue
        tgt = fp * (1 + tp); stp = fp * (1 - sl)
        oc = 0.0; xp = fp; xt = 0; done = False; xi = fi
        for k in range(fi + 1, np_):
            if date_ints[k] != ed:
                xp = prices[k-1]; oc = (xp - fp) / fp; xi = k; done = True; break
            sl_h = lows[k] <= stp; tp_h = highs[k] >= tgt
            if sl_h and tp_h:
                if np.random.random() < 0.5: oc = -sl; xp = stp; xt = -1
                else: oc = tp; xp = tgt; xt = 1
                xi = k; done = True; break
            if sl_h: oc = -sl; xp = stp; xt = -1; xi = k; done = True; break
            if tp_h: oc = tp; xp = tgt; xt = 1; xi = k; done = True; break
            if time_mins[k] >= exit_time:
                xp = prices[k]; oc = (xp - fp) / fp; xi = k; done = True; break
        if not done: xp = prices[np_-1]; oc = (xp - fp) / fp; xi = np_
        pnl_a[tc] = oc; date_a[tc] = ed; entry_a[tc] = fp; exit_a[tc] = xp; out_a[tc] = xt
        tc += 1; ltd = ed; nxt = cp + 1
        while nxt < ne:
            if entry_indices[nxt] > xi: break
            nxt += 1
        cp = nxt
    return pnl_a[:tc], date_a[:tc], entry_a[:tc], exit_a[:tc], out_a[:tc]

def _run_sim(indices, ohlcv, cfg):
    p, h, l, d, t = ohlcv['prices'], ohlcv['highs'], ohlcv['lows'], ohlcv['dates'], ohlcv['times'].astype(np.int64)
    m = cfg['mode']
    if m == 'grace':
        r = run_trade_sim_grace(indices, p, h, l, d, t, cfg['tp_pct'], cfg['sl_pct'], EXIT_TIME_MIN, cfg['grace_minutes'])
    elif m == 'limit_entry':
        r = run_trade_sim_limit_entry(indices, p, h, l, d, t, cfg['tp_pct'], cfg['sl_pct'], EXIT_TIME_MIN, cfg['dip_pct'], cfg['max_wait'])
    else:
        r = run_trade_sim(indices, p, h, l, d, t, cfg['tp_pct'], cfg['sl_pct'], EXIT_TIME_MIN)
    pnl, dd, ep, xp, oc = r
    if len(pnl) > 0 and SLIPPAGE_PCT > 0: pnl = pnl - 2 * SLIPPAGE_PCT
    return pnl, dd, ep, xp, oc


# ==============================================================================
# OHLCV DATA LOADING
# ==============================================================================

def _load_parquet_with_dt_index(gcs_uri, tz_out='America/Los_Angeles'):
    df = pd.read_parquet(gcs_uri).reset_index()
    dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    if dt_cols: df['Datetime'] = pd.to_datetime(df[dt_cols[0]], utc=True)
    else:
        nc = next((c for c in df.columns if c in ('Timestamp','Date','Time')), df.columns[0])
        df['Datetime'] = pd.to_datetime(df[nc], unit='ms', utc=True)
    df = df.set_index('Datetime')
    df = df[df.index.notna()].sort_index()
    df = df[~df.index.duplicated(keep='first')]
    df.index = df.index.tz_convert(tz_out)
    return df

def load_ticker_ohlcv(ticker):
    clean = TICKER_FILE_MAP.get(ticker, ticker)
    for sfx in ['_master_inference', '_1m_full_historical_data']:
        try: return _load_parquet_with_dt_index(f"gs://{GCS_BUCKET}/{GCS_INFER_DIR}/{clean}{sfx}.parquet")
        except Exception: continue
    raise FileNotFoundError(f"No OHLCV for {ticker}")

def precompute_ohlcv_arrays(df_ohlcv, start, end):
    s = pd.to_datetime(start).tz_localize('America/Los_Angeles')
    e = pd.to_datetime(end).tz_localize('America/Los_Angeles') + timedelta(days=1)
    df = df_ohlcv[(df_ohlcv.index >= s) & (df_ohlcv.index < e)]
    if len(df) == 0: return None
    dates = (df.index.year * 10000 + df.index.month * 100 + df.index.day).values.astype(np.int64)
    return {'prices': df['Close'].values.copy(), 'highs': df['High'].values.copy(),
            'lows': df['Low'].values.copy(), 'dates': dates,
            'times': (df.index.hour * 60 + df.index.minute).values}


# ==============================================================================
# SCORING — generic per model
# ==============================================================================

def score_model(model_key, ticker_filter=None):
    mcfg = MODELS[model_key]
    fold_id = mcfg['fold_id']
    print(f"\n{'='*W}\n  SCORING -- {model_key} ({mcfg['desc']})\n{'='*W}")

    meta = np.load(ensure_data('meta.npz', mcfg['data_dir'], mcfg['gcs_data']), allow_pickle=True)
    ticker_names = meta['ticker_names']
    print(f"  {len(ticker_names)} tickers in meta")

    probe = np.load(ensure_data(f'fold_{fold_id}_val.npz', mcfg['data_dir'], mcfg['gcs_data']))
    n_asset, n_regime = probe['X_seq'].shape[-1], probe['X_regime'].shape[-1]
    probe.close()
    print(f"  N_ASSET={n_asset}, N_REGIME={n_regime}")

    safe = model_key.replace(' ', '_').replace('-', '_')
    model = build_spike_transformer(n_asset, n_regime, name=f'st_{safe}')
    model.load_weights(ensure_model(fold_id, mcfg['model_dir'], mcfg['gcs_models']))
    print(f"  Model: {model.count_params():,} params")

    all_scores = {}
    for split in ['val', 'test']:
        data = np.load(ensure_data(f'fold_{fold_id}_{split}.npz', mcfg['data_dir'], mcfg['gcs_data']))
        X_seq, X_reg = data['X_seq'], data['X_regime']
        X_id, date_ints, minutes = data['X_id'].flatten(), data['date_ints'], data['minutes']
        data.close()
        mask = minutes == SCORE_MINUTE
        X_seq, X_reg, X_id, date_ints = X_seq[mask], X_reg[mask], X_id[mask], date_ints[mask]
        n = len(X_id); del minutes, mask; gc.collect()
        print(f"  {split}: {n:,} windows...")
        preds = model.predict({'seq_input': X_seq.astype(np.float32),
                                'regime_input': X_reg.astype(np.float32)},
                               batch_size=512, verbose=0)
        del X_seq, X_reg; gc.collect()
        for i in range(n):
            tid, d_int = int(X_id[i]), int(date_ints[i])
            if tid >= len(ticker_names): continue
            tname = str(ticker_names[tid])
            if tname in EXCLUDE_TICKERS: continue
            if ticker_filter and tname not in ticker_filter: continue
            if tname not in all_scores: all_scores[tname] = {h: {} for h in SCORE_HORIZONS}
            for h in SCORE_HORIZONS: all_scores[tname][h][d_int] = float(preds[i, h])
        del preds, X_id, date_ints; gc.collect()
    del model; tf.keras.backend.clear_session(); gc.collect()
    try: ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception: pass
    print(f"  Done: {len(all_scores)} tickers")
    return all_scores


# ==============================================================================
# BQ CONFIG LOADER
# ==============================================================================

def load_configs_from_bq(model_version, min_count, label):
    query = f"""
    SELECT ticker, prob_thresh, tp_pct, sl_pct, tpsl_mode,
           grace_minutes, dip_pct, max_wait_minutes,
           val_b_avg_pnl, val_b_win_rate, val_b_count, experiment
    FROM `{BQ_TABLE}`
    WHERE model_version = '{model_version}'
      AND val_b_avg_pnl >= {DEPLOY_PNL_THRESHOLD}
      AND val_b_count >= {min_count}
    ORDER BY val_b_avg_pnl DESC
    """
    print(f"\n  BQ: {label} (mv='{model_version}', min_count={min_count})...")
    df = bq_client.query(query).to_dataframe()
    if len(df) == 0:
        print(f"  WARNING: No configs — optimizer not run?")
        return {}
    configs = {}
    for _, row in df.iterrows():
        t = row['ticker']
        if t in configs: continue
        mode = row['tpsl_mode']; exp = str(row.get('experiment', ''))
        h_idx = 1
        for hk, hv in HORIZON_LABELS.items():
            if hv in exp: h_idx = hk; break
        cfg = {'horizon': h_idx, 'prob_thresh': float(row['prob_thresh']),
               'tp_pct': float(row['tp_pct']), 'sl_pct': float(row['sl_pct']),
               'mode': mode, 'val_b_avg_pnl': float(row['val_b_avg_pnl']),
               'val_b_win_rate': float(row['val_b_win_rate']),
               'val_b_count': int(row['val_b_count'])}
        if mode == 'grace' and pd.notna(row.get('grace_minutes')):
            cfg['grace_minutes'] = int(row['grace_minutes'])
        if mode == 'limit_entry':
            if pd.notna(row.get('dip_pct')): cfg['dip_pct'] = float(row['dip_pct'])
            if pd.notna(row.get('max_wait_minutes')): cfg['max_wait'] = int(row['max_wait_minutes'])
        configs[t] = cfg
    print(f"  {len(configs)} deployable tickers (>= {DEPLOY_PNL_THRESHOLD:.1%})")
    for t, c in sorted(configs.items(), key=lambda x: -x[1]['val_b_avg_pnl']):
        print(f"    {t:<7} H={HORIZON_LABELS[c['horizon']]} {c['mode']:<13} "
              f"val_b={c['val_b_avg_pnl']:+.2%} N={c['val_b_count']}")
    return configs


# ==============================================================================
# BACKTEST — per model, returns trade list with Ticker field
# ==============================================================================

def run_backtest(all_scores, deploy_configs, label):
    tickers = sorted(deploy_configs.keys())
    if not tickers: return []
    print(f"\n{'─'*W}\n  BACKTEST -- {label} ({len(tickers)} tickers) | "
          f"{OVERLAP_START} -> {OVERLAP_END}\n{'─'*W}")
    all_trades = []
    for ticker in tickers:
        cfg = deploy_configs[ticker]
        h_idx, h_label, mode = cfg['horizon'], HORIZON_LABELS[cfg['horizon']], cfg['mode']
        h_scores = all_scores.get(ticker, {}).get(h_idx, {})
        if not h_scores: continue
        try: df_ohlcv = load_ticker_ohlcv(ticker)
        except Exception: continue
        ohlcv = precompute_ohlcv_arrays(df_ohlcv, OVERLAP_START, OVERLAP_END)
        if ohlcv is None: del df_ohlcv; continue
        score_by_date = {int(d): s for d, s in h_scores.items()}
        entry_list = [i for i in range(len(ohlcv['dates']))
                      if ohlcv['times'][i] == ENTRY_MINUTE
                      and score_by_date.get(int(ohlcv['dates'][i]), 0.0) >= cfg['prob_thresh']]
        if not entry_list: del df_ohlcv; continue
        pnl, dates, ep, xp, outcomes = _run_sim(np.array(entry_list, dtype=np.int64), ohlcv, cfg)
        if len(pnl) == 0: del df_ohlcv; continue
        print(f"  {ticker:<7} {len(pnl)}t avg={np.mean(pnl):+.2%} WR={np.mean(pnl>0):.0%}")
        for i in range(len(pnl)):
            d = int(dates[i])
            all_trades.append({
                'Ticker': ticker,
                'Date': f"{d//10000}-{(d%10000)//100:02d}-{d%100:02d}",
                'PnL': float(pnl[i]),
                'Outcome': 'TP' if outcomes[i]==1 else 'SL' if outcomes[i]==-1 else 'EOD',
                'Mode': mode, 'Horizon': h_label,
            })
        del df_ohlcv, ohlcv; gc.collect()
    return all_trades


# ==============================================================================
# STRATEGY BUILDERS
# ==============================================================================

def build_priority(model_order, all_cfgs):
    assignments = {}
    for mk in model_order:
        for t in all_cfgs.get(mk, {}):
            if t not in assignments:
                assignments[t] = mk
    return assignments

def build_best_pnl(all_cfgs):
    best = {}
    for mk, cfgs in all_cfgs.items():
        for t, cfg in cfgs.items():
            if t not in best or cfg['val_b_avg_pnl'] > all_cfgs[best[t]][t]['val_b_avg_pnl']:
                best[t] = mk
    return best

def build_consensus(all_cfgs, min_models=2):
    ticker_models = defaultdict(list)
    for mk, cfgs in all_cfgs.items():
        for t, cfg in cfgs.items():
            ticker_models[t].append((mk, cfg['val_b_avg_pnl']))
    assignments = {}
    for t, models in ticker_models.items():
        if len(models) >= min_models:
            assignments[t] = max(models, key=lambda x: x[1])[0]
    return assignments

def assemble_trades(assignments, model_trades):
    trades = []
    for ticker, mk in assignments.items():
        for t in model_trades.get(mk, []):
            if t['Ticker'] == ticker:
                t2 = dict(t)
                t2['Source'] = mk
                trades.append(t2)
    return trades


# ==============================================================================
# EW ANALYSIS
# ==============================================================================

def analyze(trades, label):
    if not trades:
        return {'label': label, 'n': 0, 'avg_pnl': 0, 'ew_total': 0, 'ew_avg': 0,
                'ew_sharpe': 0, 'win_rate': 0, 'n_tickers': 0, 'n_days': 0,
                'avg_trades_day': 0, 'sharpe': 0, 'total_pnl': 0}
    df = pd.DataFrame(trades)
    n = len(df); avg = df['PnL'].mean(); std = df['PnL'].std()
    daily_ew = df.groupby('Date')['PnL'].mean()
    nd = len(daily_ew); ew_std = daily_ew.std()
    return {
        'label': label, 'n': n, 'avg_pnl': avg, 'total_pnl': df['PnL'].sum(),
        'ew_total': daily_ew.sum(),
        'ew_avg': daily_ew.mean() if nd > 0 else 0,
        'win_rate': (df['PnL'] > 0).mean(),
        'n_tickers': df['Ticker'].nunique(), 'n_days': nd,
        'avg_trades_day': n / nd if nd > 0 else 0,
        'sharpe': (avg / std * np.sqrt(252)) if std > 0 else 0,
        'ew_sharpe': (daily_ew.mean() / ew_std * np.sqrt(252)) if nd > 1 and ew_std > 0 else 0,
    }


# ==============================================================================
# MAIN
# ==============================================================================

print(f"\n{'='*W}")
print(f"  LONG HYBRID vs HYBRID COMPARISON")
print(f"  L5-1 Hybrid (3 folds) [DEPLOYED] vs ASL6 Hybrid (3 folds)")
print(f"  6 models × 7 strategies → optimal portfolio")
print(f"  OVERLAP: {OVERLAP_START} -> {OVERLAP_END} ({OVERLAP_DAYS} cal days)")
print(f"  Deploy threshold: >{DEPLOY_PNL_THRESHOLD:.1%} | Slippage: {2*SLIPPAGE_PCT:.2%} RT")
print(f"{'='*W}")

# -- Phase 1: Load BQ configs for all 6 models --
all_cfgs = {}
for mk in MODEL_KEYS:
    mcfg = MODELS[mk]
    min_count = max(1, round(MIN_TRADES_PER_MONTH * mcfg['test_days'] / 30))
    all_cfgs[mk] = load_configs_from_bq(mcfg['mv'], min_count, mk)

active_models = [mk for mk in MODEL_KEYS if all_cfgs[mk]]
missing_models = [mk for mk in MODEL_KEYS if not all_cfgs[mk]]
if missing_models:
    print(f"\n  MISSING MODELS (no BQ configs): {', '.join(missing_models)}")
print(f"\n  L5-1 family: {sum(len(all_cfgs[mk]) for mk in L5_1_KEYS)} total configs")
print(f"  ASL6 family: {sum(len(all_cfgs[mk]) for mk in ASL6_KEYS)} total configs")
print(f"  ACTIVE: {', '.join(active_models)}")

# -- Phase 2: Score each active model (share data across folds in same family) --
all_scores = {}
for mk in MODEL_KEYS:
    if not all_cfgs[mk]:
        all_scores[mk] = {}
        continue
    all_scores[mk] = score_model(mk, ticker_filter=set(all_cfgs[mk].keys()))
    tf.keras.backend.clear_session(); gc.collect()

# -- Phase 3: Backtest each model independently --
model_trades = {}
for mk in MODEL_KEYS:
    model_trades[mk] = run_backtest(all_scores[mk], all_cfgs[mk], mk) if all_cfgs[mk] else []

# -- Phase 4: Build combination strategies --
print(f"\n{'='*W}\n  BUILDING COMBINATION STRATEGIES\n{'='*W}")

STRATEGIES = {}

STRATEGIES['L5-1 Hybrid'] = build_priority(L5_1_KEYS, all_cfgs)
STRATEGIES['ASL6 Hybrid'] = build_priority(ASL6_KEYS, all_cfgs)
STRATEGIES['Union L5-1>'] = build_priority(L5_1_KEYS + ASL6_KEYS, all_cfgs)
STRATEGIES['Union ASL6>'] = build_priority(ASL6_KEYS + L5_1_KEYS, all_cfgs)
STRATEGIES['Best Val-B'] = build_best_pnl(all_cfgs)
STRATEGIES['Newest First'] = build_priority(
    ['L5-1 F7', 'ASL6 F7', 'L5-1 F6', 'ASL6 F6', 'L5-1 F5', 'ASL6 F5'], all_cfgs)

STRAT_KEYS = list(STRATEGIES.keys())

for sname, assigns in STRATEGIES.items():
    src_counts = defaultdict(int)
    for mk in assigns.values(): src_counts[mk] += 1
    src_str = ', '.join(f"{mk}:{c}" for mk, c in sorted(src_counts.items()))
    print(f"  {sname:<16}: {len(assigns)} tickers [{src_str}]")

# -- Phase 5: Assemble strategy trades --
strat_trades = {}
for sname in STRAT_KEYS:
    strat_trades[sname] = assemble_trades(STRATEGIES[sname], model_trades)

# Build Oracle (best actual PnL per ticker, computed after backtests)
all_tickers = sorted(set(t for assigns in STRATEGIES.values() for t in assigns))
oracle_assigns = {}
for ticker in all_tickers:
    best_mk, best_avg = None, -999.0
    for mk in MODEL_KEYS:
        mt = [t for t in model_trades.get(mk, []) if t['Ticker'] == ticker]
        if mt:
            avg = np.mean([t['PnL'] for t in mt])
            if avg > best_avg:
                best_avg = avg; best_mk = mk
    if best_mk:
        oracle_assigns[ticker] = best_mk
STRATEGIES['Oracle'] = oracle_assigns
STRAT_KEYS.append('Oracle')
strat_trades['Oracle'] = assemble_trades(oracle_assigns, model_trades)
print(f"  {'Oracle':<16}: {len(oracle_assigns)} tickers [best ACTUAL PnL per ticker — upper bound]")

# -- Phase 6: Analyze all strategies --
strat_stats = {sname: analyze(strat_trades[sname], sname) for sname in STRAT_KEYS}


# ==============================================================================
# STRATEGY COMPARISON TABLE
# ==============================================================================

print(f"\n\n{'='*W}")
print(f"  STRATEGY COMPARISON ({OVERLAP_START} -> {OVERLAP_END})")
print(f"{'='*W}\n")

col_w = 12
hdr = f"  {'Metric':<18}" + "".join(f"{sn:>{col_w}}" for sn in STRAT_KEYS)
print(hdr)
print(f"  {'─'*(18 + col_w * len(STRAT_KEYS))}")

def _fp(v): return f"{v:+.2%}" if v != 0 else "---"
def _fn(v): return str(v) if v != 0 else "---"

rows = [
    ('Tickers',       'n_tickers',      _fn),
    ('Trades',        'n',              _fn),
    ('Active days',   'n_days',         _fn),
    ('Trades/day',    'avg_trades_day', lambda v: f"{v:.1f}" if v > 0 else "---"),
    ('Avg PnL/trade', 'avg_pnl',       _fp),
    ('EW Portfolio',  'ew_total',       _fp),
    ('EW Avg daily',  'ew_avg',         _fp),
    ('Win rate',      'win_rate',       lambda v: f"{v:.1%}" if v > 0 else "---"),
    ('Sharpe (trade)','sharpe',         lambda v: f"{v:.1f}" if v > 0 else "---"),
    ('Sharpe (EW)',   'ew_sharpe',      lambda v: f"{v:.1f}" if v > 0 else "---"),
]

for label, metric, fmt in rows:
    line = f"  {label:<18}"
    for sn in STRAT_KEYS:
        line += f"{fmt(strat_stats[sn][metric]):>{col_w}}"
    print(line)


# ==============================================================================
# DAILY SUMMARY TABLE — trades per day, EW PnL, for each strategy
# ==============================================================================

print(f"\n\n{'='*W}")
print(f"  DAILY SUMMARY — Trades and EW PnL by Date")
print(f"{'='*W}\n")

core_strats = ['L5-1 Hybrid', 'ASL6 Hybrid', 'Union L5-1>', 'Best Val-B', 'Oracle']

all_dates = sorted(set(t['Date'] for st in strat_trades.values() for t in st))
cw = 22
print(f"  {'Date':<12}", end="")
for sn in core_strats:
    print(f" {sn:>{cw}}", end="")
print()
print(f"  {'─'*(12 + (cw+1) * len(core_strats))}")

for date in all_dates:
    line = f"  {date:<12}"
    for sn in core_strats:
        day_trades = [t for t in strat_trades[sn] if t['Date'] == date]
        nt = len(day_trades)
        if nt > 0:
            avg_pnl = np.mean([t['PnL'] for t in day_trades])
            n_win = sum(1 for t in day_trades if t['PnL'] > 0)
            line += f" {nt:>2}t {avg_pnl:>+5.1%} ({n_win}W/{nt-n_win}L)"
        else:
            line += f" {'---':>{cw}}"
    print(line)

print(f"  {'─'*(12 + (cw+1) * len(core_strats))}")
line = f"  {'TOTAL':<12}"
for sn in core_strats:
    s = strat_stats[sn]
    if s['n'] > 0:
        line += f" {s['n']:>2}t {s['ew_total']:>+5.1%}  EW total  "
    else:
        line += f" {'---':>{cw}}"
print(line)


# ==============================================================================
# L5-1 HYBRID vs ASL6 HYBRID HEAD-TO-HEAD
# ==============================================================================

print(f"\n\n{'='*W}")
print(f"  HEAD-TO-HEAD: L5-1 Hybrid [DEPLOYED] vs ASL6 Hybrid")
print(f"{'='*W}")

l51_s = strat_stats['L5-1 Hybrid']
asl6_s = strat_stats['ASL6 Hybrid']

l51_tickers = set(STRATEGIES['L5-1 Hybrid'].keys())
asl6_tickers = set(STRATEGIES['ASL6 Hybrid'].keys())
shared_t = l51_tickers & asl6_tickers
l51_only = l51_tickers - asl6_tickers
asl6_only = asl6_tickers - l51_tickers

print(f"\n  TICKER OVERLAP:")
print(f"    L5-1 Hybrid tickers:   {len(l51_tickers)}")
print(f"    ASL6 Hybrid tickers:   {len(asl6_tickers)}")
print(f"    Shared:                {len(shared_t)}")
print(f"    L5-1 exclusive:        {len(l51_only)}  {', '.join(sorted(l51_only)) if l51_only else '(none)'}")
print(f"    ASL6 exclusive:        {len(asl6_only)}  {', '.join(sorted(asl6_only)) if asl6_only else '(none)'}")

l51_traded = set(t['Ticker'] for t in strat_trades['L5-1 Hybrid'])
asl6_traded = set(t['Ticker'] for t in strat_trades['ASL6 Hybrid'])
print(f"\n  TICKERS THAT ACTUALLY TRADED:")
print(f"    L5-1 Hybrid:  {len(l51_traded)}  {', '.join(sorted(l51_traded))}")
print(f"    ASL6 Hybrid:  {len(asl6_traded)}  {', '.join(sorted(asl6_traded))}")

metrics = [
    ('Tickers traded', l51_s['n_tickers'], asl6_s['n_tickers'], False),
    ('Total trades', l51_s['n'], asl6_s['n'], False),
    ('Active days', l51_s['n_days'], asl6_s['n_days'], False),
    ('Trades/day', l51_s['avg_trades_day'], asl6_s['avg_trades_day'], False),
    ('Avg PnL/trade', l51_s['avg_pnl'], asl6_s['avg_pnl'], True),
    ('Win rate', l51_s['win_rate'], asl6_s['win_rate'], True),
    ('EW Total PnL', l51_s['ew_total'], asl6_s['ew_total'], True),
    ('EW Avg daily', l51_s['ew_avg'], asl6_s['ew_avg'], True),
    ('Sharpe (EW)', l51_s['ew_sharpe'], asl6_s['ew_sharpe'], False),
]

print(f"\n  {'Metric':<18} {'L5-1 Hybrid':>14} {'ASL6 Hybrid':>14} {'Delta':>10} {'Winner':>8}")
print(f"  {'─'*70}")
for label, v1, v2, is_pct in metrics:
    delta = v2 - v1
    winner = 'ASL6' if delta > 0.001 else 'L5-1' if delta < -0.001 else 'TIE'
    if is_pct:
        print(f"  {label:<18} {v1:>+13.2%} {v2:>+13.2%} {delta:>+9.2%} {winner:>8}")
    else:
        print(f"  {label:<18} {v1:>14.1f} {v2:>14.1f} {delta:>+10.1f} {winner:>8}")

# Per-ticker head-to-head on shared tickers
if shared_t:
    print(f"\n  SHARED TICKERS HEAD-TO-HEAD ({len(shared_t)} tickers):")
    l51_wins, asl6_wins, ties = 0, 0, 0
    for ticker in sorted(shared_t):
        l51_t = [t for t in strat_trades['L5-1 Hybrid'] if t['Ticker'] == ticker]
        asl6_t = [t for t in strat_trades['ASL6 Hybrid'] if t['Ticker'] == ticker]
        l51_avg = np.mean([t['PnL'] for t in l51_t]) if l51_t else None
        asl6_avg = np.mean([t['PnL'] for t in asl6_t]) if asl6_t else None
        l51_str = f"{len(l51_t)}t {l51_avg:+.2%}" if l51_t else "no trades"
        asl6_str = f"{len(asl6_t)}t {asl6_avg:+.2%}" if asl6_t else "no trades"
        if l51_avg is not None and asl6_avg is not None:
            if l51_avg > asl6_avg + 0.001: l51_wins += 1; w = "<- L5-1"
            elif asl6_avg > l51_avg + 0.001: asl6_wins += 1; w = "ASL6 ->"
            else: ties += 1; w = "TIE"
        elif l51_avg is not None: l51_wins += 1; w = "<- L5-1"
        elif asl6_avg is not None: asl6_wins += 1; w = "ASL6 ->"
        else: ties += 1; w = "SKIP"
        l51_src = STRATEGIES['L5-1 Hybrid'].get(ticker, '?')
        asl6_src = STRATEGIES['ASL6 Hybrid'].get(ticker, '?')
        print(f"    {ticker:<7} L5-1({l51_src[-2:]}): {l51_str:<18} ASL6({asl6_src[-2:]}): {asl6_str:<18} {w}")
    print(f"    SCORE: L5-1 {l51_wins} — ASL6 {asl6_wins} — TIE {ties}")


# ==============================================================================
# PER-TICKER MODEL SELECTION ANALYSIS
# ==============================================================================

print(f"\n\n{'='*W}")
print(f"  PER-TICKER MODEL SELECTION (all 6 models)")
print(f"{'='*W}")

ticker_analysis = []
for ticker in all_tickers:
    row = {'Ticker': ticker}
    for mk in MODEL_KEYS:
        cfg = all_cfgs.get(mk, {}).get(ticker)
        mt = [t for t in model_trades.get(mk, []) if t['Ticker'] == ticker]
        row[f'{mk}_val_b'] = cfg['val_b_avg_pnl'] if cfg else None
        row[f'{mk}_actual'] = np.mean([t['PnL'] for t in mt]) if mt else None
        row[f'{mk}_n'] = len(mt)
    row['n_models'] = sum(1 for mk in MODEL_KEYS if row[f'{mk}_val_b'] is not None)
    row['n_families'] = sum(1 for fam_keys in [L5_1_KEYS, ASL6_KEYS]
                            if any(row[f'{mk}_val_b'] is not None for mk in fam_keys))
    row['oracle_pick'] = oracle_assigns.get(ticker, '---')
    row['best_valb_pick'] = STRATEGIES['Best Val-B'].get(ticker, '---')
    ticker_analysis.append(row)

ticker_analysis.sort(key=lambda x: (-x['n_families'], -x['n_models'], x['Ticker']))

print(f"\n  {'Ticker':<7} {'Fam':>3} {'#M':>2}", end="")
for mk in MODEL_KEYS:
    short = mk.replace('ASL6 ', 'A').replace('L5-1 ', '1')
    print(f" {short:>5}v {short:>5}a", end="")
print(f" {'BestV':>7} {'Oracle':>7} {'Match':>5}")
print(f"  {'─'*110}")

correct_picks = 0; total_picks = 0
for row in ticker_analysis:
    line = f"  {row['Ticker']:<7} {row['n_families']:>3} {row['n_models']:>2}"
    for mk in MODEL_KEYS:
        vb = row[f'{mk}_val_b']
        ac = row[f'{mk}_actual']
        line += f" {vb:>+4.1%}" if vb is not None else " " * 6
        line += f" {ac:>+4.1%}" if ac is not None else " " * 6
    bv = row['best_valb_pick']; orc = row['oracle_pick']
    match = "Y" if bv == orc else "N"
    if row['n_models'] > 1:
        total_picks += 1
        if bv == orc: correct_picks += 1
    bv_short = bv.replace('ASL6 ', 'A').replace('L5-1 ', '1') if bv != '---' else '---'
    orc_short = orc.replace('ASL6 ', 'A').replace('L5-1 ', '1') if orc != '---' else '---'
    line += f" {bv_short:>7} {orc_short:>7} {match:>5}"
    print(line)

print(f"\n  BOTH FAMILIES: {sum(1 for r in ticker_analysis if r['n_families'] == 2)}")
print(f"  L5-1 ONLY:     {sum(1 for r in ticker_analysis if r['n_families'] == 1 and any(r[f'{mk}_val_b'] is not None for mk in L5_1_KEYS))}")
print(f"  ASL6 ONLY:     {sum(1 for r in ticker_analysis if r['n_families'] == 1 and any(r[f'{mk}_val_b'] is not None for mk in ASL6_KEYS))}")
if total_picks > 0:
    print(f"  Val-B predicts Oracle: {correct_picks}/{total_picks} ({correct_picks/total_picks:.0%})")

# Family-exclusive analysis
print(f"\n  FAMILY-EXCLUSIVE TICKERS:")
for fam_name, fam_keys, other_keys in [('L5-1', L5_1_KEYS, ASL6_KEYS), ('ASL6', ASL6_KEYS, L5_1_KEYS)]:
    exclusive = [r['Ticker'] for r in ticker_analysis
                 if any(r[f'{mk}_val_b'] is not None for mk in fam_keys)
                 and not any(r[f'{mk}_val_b'] is not None for mk in other_keys)]
    if exclusive:
        avgs = [r[f'{mk}_actual'] for r in ticker_analysis
                for mk in fam_keys
                if r['Ticker'] in exclusive and r[f'{mk}_actual'] is not None]
        avg_str = f" avg={np.mean(avgs):+.2%}" if avgs else ""
        print(f"    {fam_name}: {len(exclusive)} tickers{avg_str} — {', '.join(exclusive)}")


# ==============================================================================
# CHARTS
# ==============================================================================

n_strats = len(STRAT_KEYS)
STRAT_COLORS = {'L5-1 Hybrid': '#C62828', 'ASL6 Hybrid': '#1565C0',
                'Union L5-1>': '#E65100', 'Union ASL6>': '#2E7D32',
                'Best Val-B': '#6A1B9A', 'Newest First': '#00838F',
                'Oracle': '#FF6F00'}

fig, axes = plt.subplots(2, 3, figsize=(26, 13))
fig.suptitle(f'LONG Hybrid vs Hybrid: L5-1 (3 folds) vs ASL6 (3 folds)\n'
             f'{OVERLAP_START} -> {OVERLAP_END} ({OVERLAP_DAYS}d) | '
             f'Deploy >{DEPLOY_PNL_THRESHOLD:.1%}',
             fontsize=13, fontweight='bold')

# [0,0] EW Cumulative by strategy
ax = axes[0, 0]
for sn in STRAT_KEYS:
    trades = strat_trades[sn]
    if not trades: continue
    df = pd.DataFrame(trades)
    ew = df.groupby('Date')['PnL'].mean().sort_index()
    cum = ew.cumsum().values
    lw = 3 if sn in ('L5-1 Hybrid', 'ASL6 Hybrid') else 1.5
    ls = '--' if sn == 'Oracle' else '-'
    ax.plot(pd.to_datetime(ew.index), cum * 100,
            color=STRAT_COLORS.get(sn, '#888'), linewidth=lw, linestyle=ls,
            label=f"{sn} ({cum[-1]*100:+.1f}%)", marker='o', markersize=3)
ax.axhline(0, color='black', linewidth=0.8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
ax.set_ylabel('Cumulative EW PnL (%)')
ax.set_title('EW Portfolio: All Strategies')
ax.legend(fontsize=6); ax.grid(alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# [0,1] L5-1 Hybrid vs ASL6 Hybrid bar chart (head-to-head metrics)
ax = axes[0, 1]
h2h_metrics = ['EW Total', 'Avg PnL', 'Win Rate', 'Sharpe']
h2h_l51 = [l51_s['ew_total']*100, l51_s['avg_pnl']*100, l51_s['win_rate']*100, l51_s['ew_sharpe']]
h2h_asl6 = [asl6_s['ew_total']*100, asl6_s['avg_pnl']*100, asl6_s['win_rate']*100, asl6_s['ew_sharpe']]
x = np.arange(len(h2h_metrics))
ax.bar(x - 0.2, h2h_l51, 0.35, color='#C62828', alpha=0.8, label='L5-1 Hybrid')
ax.bar(x + 0.2, h2h_asl6, 0.35, color='#1565C0', alpha=0.8, label='ASL6 Hybrid')
for i in range(len(h2h_metrics)):
    ax.text(i - 0.2, h2h_l51[i] + 0.3, f"{h2h_l51[i]:.1f}", ha='center', fontsize=7)
    ax.text(i + 0.2, h2h_asl6[i] + 0.3, f"{h2h_asl6[i]:.1f}", ha='center', fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(h2h_metrics)
ax.set_title('L5-1 Hybrid vs ASL6 Hybrid'); ax.legend(); ax.grid(axis='y', alpha=0.3)

# [0,2] Model contribution per strategy (stacked)
ax = axes[0, 2]
model_colors = {mk: MODELS[mk]['color'] for mk in MODEL_KEYS}
strat_x = np.arange(n_strats)
bottom = np.zeros(n_strats)
for mk in MODEL_KEYS:
    counts = []
    for sn in STRAT_KEYS:
        c = sum(1 for v in STRATEGIES[sn].values() if v == mk)
        counts.append(c)
    ax.bar(strat_x, counts, bottom=bottom, color=model_colors[mk],
           alpha=0.8, label=mk, edgecolor='white', linewidth=0.5)
    bottom += counts
ax.set_xticks(strat_x)
ax.set_xticklabels([sn.replace(' ', '\n') for sn in STRAT_KEYS], fontsize=6)
ax.set_ylabel('# Tickers'); ax.set_title('Model Source per Strategy')
ax.legend(fontsize=5, loc='upper right', ncol=2); ax.grid(axis='y', alpha=0.3)

# [1,0] Val-B vs Actual scatter
ax = axes[1, 0]
scatter_data = []
for row in ticker_analysis:
    for mk in MODEL_KEYS:
        vb, ac = row[f'{mk}_val_b'], row[f'{mk}_actual']
        if vb is not None and ac is not None:
            scatter_data.append((vb, ac, mk, row['Ticker']))
if scatter_data:
    for mk in MODEL_KEYS:
        pts = [(vb, ac) for vb, ac, m, _ in scatter_data if m == mk]
        if pts:
            vbs, acs = zip(*pts)
            ax.scatter([v*100 for v in vbs], [a*100 for a in acs],
                      color=model_colors[mk], alpha=0.7, s=40, label=mk, edgecolors='white')
    all_vb = [d[0]*100 for d in scatter_data]
    all_ac = [d[1]*100 for d in scatter_data]
    lo = min(min(all_vb), min(all_ac)) - 1
    hi = max(max(all_vb), max(all_ac)) + 1
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.3, linewidth=1)
    ax.axhline(0, color='black', linewidth=0.5, alpha=0.3)
    ax.axvline(0, color='black', linewidth=0.5, alpha=0.3)
    if len(scatter_data) > 2:
        corr = np.corrcoef([d[0] for d in scatter_data], [d[1] for d in scatter_data])[0, 1]
        ax.set_title(f'Val-B vs Actual PnL (r={corr:.2f})')
    else:
        ax.set_title('Val-B vs Actual PnL')
ax.set_xlabel('Val-B Avg PnL (%)'); ax.set_ylabel('Actual BT PnL (%)')
ax.legend(fontsize=5, ncol=2); ax.grid(alpha=0.3)

# [1,1] Summary table
ax = axes[1, 1]; ax.axis('off')
non_oracle = [sn for sn in STRAT_KEYS if sn != 'Oracle']
tdata = [[''] + [sn.replace(' ', '\n') for sn in non_oracle]]
for label, metric, fmt in [
    ('Tickers', 'n_tickers', str),
    ('Trades', 'n', str),
    ('Trades/day', 'avg_trades_day', lambda v: f"{v:.1f}"),
    ('Avg PnL', 'avg_pnl', lambda v: f"{v:+.2%}"),
    ('EW Total', 'ew_total', lambda v: f"{v:+.2%}"),
    ('WR', 'win_rate', lambda v: f"{v:.0%}"),
    ('Sharpe', 'ew_sharpe', lambda v: f"{v:.1f}"),
]:
    tdata.append([label] + [fmt(strat_stats[sn][metric]) for sn in non_oracle])

table = ax.table(cellText=tdata, loc='center', cellLoc='center')
table.auto_set_font_size(False); table.set_fontsize(6); table.scale(1, 1.6)
for (r, c), cell in table.get_celld().items():
    if r == 0:
        cell.set_facecolor('#BBDEFB'); cell.set_text_props(fontweight='bold')
ax.set_title('Strategy Summary (excl. Oracle)', pad=20)

# [1,2] Per-model (all 6) standalone performance
ax = axes[1, 2]
active_mk = [mk for mk in MODEL_KEYS if model_trades[mk]]
if active_mk:
    mx = np.arange(len(active_mk))
    avgs = [np.mean([t['PnL'] for t in model_trades[mk]]) * 100 if model_trades[mk] else 0
            for mk in active_mk]
    ax.bar(mx, avgs, color=[model_colors[mk] for mk in active_mk],
           alpha=0.8, edgecolor='white', linewidth=0.5)
    for i, mk in enumerate(active_mk):
        nt = len(set(t['Ticker'] for t in model_trades[mk]))
        ax.text(i, max(avgs[i], 0) + 0.1, f"N={len(model_trades[mk])}\n{nt}t",
                ha='center', va='bottom', fontsize=7)
    ax.set_xticks(mx)
    ax.set_xticklabels([mk.replace('ASL6 ', 'A').replace('L5-1 ', '1') for mk in active_mk], fontsize=7)
ax.set_ylabel('Avg PnL/trade (%)'); ax.set_title('Per-Model Performance (6 models)')
ax.axhline(0, color='black', linewidth=0.8); ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
chart_path = '/tmp/backtest_long_hybrid_vs_hybrid.png'
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
print(f"\n  Chart saved: {chart_path}")
if _IN_COLAB: plt.show()
plt.close()


# ==============================================================================
# VERDICT
# ==============================================================================

print(f"\n\n{'='*W}")
print(f"  VERDICT — L5-1 Hybrid vs ASL6 Hybrid + Combination Strategies")
print(f"{'='*W}")

real_strats = [(sn, strat_stats[sn]) for sn in STRAT_KEYS
               if sn != 'Oracle' and strat_stats[sn]['n'] > 0]
oracle_s = strat_stats.get('Oracle', {})

if not real_strats:
    print(f"\n  No strategies have trades. Run the optimizers first.")
else:
    ranked_ew = sorted(real_strats, key=lambda x: x[1]['ew_total'], reverse=True)

    print(f"\n  RANKING BY EW PORTFOLIO PNL (the metric that matters):")
    for i, (sn, s) in enumerate(ranked_ew, 1):
        tag = " [DEPLOYED]" if sn == 'L5-1 Hybrid' else ""
        print(f"    #{i} {sn:<16}: EW {s['ew_total']:+.2%} | "
              f"{s['avg_pnl']:+.2%}/trade | {s['n_tickers']}t {s['n']} trades | "
              f"t/d {s['avg_trades_day']:.1f} | "
              f"Sharpe(EW) {s['ew_sharpe']:.1f}{tag}")

    if oracle_s.get('n', 0) > 0:
        print(f"\n  ORACLE (upper bound): EW {oracle_s['ew_total']:+.2%} | "
              f"{oracle_s['avg_pnl']:+.2%}/trade | {oracle_s['n_tickers']}t")
        best_real_ew = ranked_ew[0][1]['ew_total']
        gap = oracle_s['ew_total'] - best_real_ew
        print(f"  Gap to Oracle: {gap:+.2%} EW PnL "
              f"({'small — selection barely matters' if abs(gap) < 0.02 else 'significant — model selection matters'})")

    winner = ranked_ew[0][0]
    baseline = strat_stats.get('L5-1 Hybrid', {})

    print(f"\n  HEAD-TO-HEAD CORE COMPARISON:")
    print(f"    L5-1 Hybrid [DEPLOYED]: EW {l51_s['ew_total']:+.2%} | {l51_s['n_tickers']}t | {l51_s['n']} trades | t/d {l51_s['avg_trades_day']:.1f}")
    print(f"    ASL6 Hybrid:            EW {asl6_s['ew_total']:+.2%} | {asl6_s['n_tickers']}t | {asl6_s['n']} trades | t/d {asl6_s['avg_trades_day']:.1f}")
    delta_ew = asl6_s['ew_total'] - l51_s['ew_total']
    print(f"    Delta:                  {delta_ew:+.2%} EW | {'ASL6 wins' if delta_ew > 0 else 'L5-1 wins'}")

    if winner != 'L5-1 Hybrid' and baseline.get('n', 0) > 0:
        best_s = ranked_ew[0][1]
        delta_ew = best_s['ew_total'] - baseline['ew_total']
        delta_tickers = best_s['n_tickers'] - baseline['n_tickers']
        print(f"\n  BEST STRATEGY vs DEPLOYED:")
        print(f"    {winner}: EW {best_s['ew_total']:+.2%} vs L5-1 Hybrid {baseline['ew_total']:+.2%} (delta {delta_ew:+.2%})")
        print(f"    Tickers: {best_s['n_tickers']} vs {baseline['n_tickers']} ({delta_tickers:+d})")
        print(f"    Trades:  {best_s['n']} vs {baseline['n']}")

    print(f"\n  RECOMMENDATION: Deploy '{winner}'")
    assigns = STRATEGIES[winner]
    src_counts = defaultdict(list)
    for t, mk in assigns.items(): src_counts[mk].append(t)
    print(f"  DEPLOYMENT CONFIG ({len(assigns)} tickers):")
    for mk in MODEL_KEYS:
        if mk in src_counts:
            tickers = sorted(src_counts[mk])
            print(f"    {mk} ({len(tickers)}): {', '.join(tickers)}")

print(f"\n{'='*W}")
