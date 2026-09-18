# %% ── SHORT L5-2: 5-WAY MODEL COMPARISON ────────────────────────────────────
#
# Head-to-head evaluation of 5 SHORT L5-2 models on fold 5 TEST set:
#   1. BASELINE      — standard SpikeTransformer (33 seq, 22 regime)
#   2. X-ASSET       — BTC/ETH lead-lag features (47 seq, 26 regime)
#   3. MULTI-RES     — dual Transformer backbone 1-min + 15-min, 50k samples
#   4. MULTI-RES v2  — same architecture, 150k samples (3x data to reduce overfitting)
#   5. V2+XASSET     — Multi-Res v2 + BTC/ETH lead-lag (47 seq, 12 lowres, 26 regime, 150k)
#
# Phase 1: Load all 5 models, score test npz at SCORE_MINUTE
# Phase 2: ML Metrics (AUC-ROC, Avg Precision, Brier, calibration)
# Phase 3: Per-Ticker AUC comparison (30m horizon)
# Phase 4: Trade Simulation (standard TP/SL grid, per-ticker PnL)
# Phase 5: PnL Performance Over TEST Window (weekly cumulative, drawdown)
# Phase 6: Verdict
#
# GPU recommended. Colab-ready.

import os, sys, gc, ctypes, warnings, time
import numpy as np
import pandas as pd
import pytz

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    from numba import jit
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numba", "--quiet"])
    from numba import jit

import tensorflow as tf
from tensorflow.keras import layers, Model
from google.cloud import storage

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

from sklearn.metrics import (roc_auc_score, precision_recall_curve,
                             average_precision_score, brier_score_loss,
                             precision_score, recall_score, f1_score)


# ==============================================================================
# CONFIG
# ==============================================================================
GCS_BUCKET = os.environ.get("GCS_BUCKET", "")
GCS_INFER_DIR = 'Data/Inference_v2'

MODELS = {
    'BASELINE': {
        'label': 'Baseline L5-2',
        'arch': 'standard',
        'gcs_data':   'Algos/RA_MoE/short_L5_2_data',
        'gcs_models': 'Algos/RA_MoE/short_L5_2_models',
        'local_data':  '/tmp/_bt_3way_baseline_data',
        'local_model': '/tmp/_bt_3way_baseline_models',
    },
    'XASSET': {
        'label': 'X-ASSET (BTC/ETH)',
        'arch': 'standard',
        'gcs_data':   'Algos/RA_MoE/short_L5_2_xasset_data',
        'gcs_models': 'Algos/RA_MoE/short_L5_2_xasset_models',
        'local_data':  '/tmp/_bt_3way_xasset_data',
        'local_model': '/tmp/_bt_3way_xasset_models',
    },
    'MULTIRES': {
        'label': 'Multi-Res v1 (50k)',
        'arch': 'multires',
        'gcs_data':   'Algos/RA_MoE/short_L5_2_multires_data',
        'gcs_models': 'Algos/RA_MoE/short_L5_2_multires_models',
        'local_data':  '/tmp/_bt_3way_multires_data',
        'local_model': '/tmp/_bt_3way_multires_models',
    },
    'MULTIRES_V2': {
        'label': 'Multi-Res v2 (150k)',
        'arch': 'multires',
        'gcs_data':   'Algos/RA_MoE/short_L5_2_multires_v2_data',
        'gcs_models': 'Algos/RA_MoE/short_L5_2_multires_v2_models',
        'local_data':  '/tmp/_bt_5way_multires_v2_data',
        'local_model': '/tmp/_bt_5way_multires_v2_models',
    },
    'MULTIRES_V2_XASSET': {
        'label': 'v2+X-ASSET (150k)',
        'arch': 'multires',
        'gcs_data':   'Algos/RA_MoE/short_L5_2_multires_v2_xasset_data',
        'gcs_models': 'Algos/RA_MoE/short_L5_2_multires_v2_xasset_models',
        'local_data':  '/tmp/_bt_5way_v2_xasset_data',
        'local_model': '/tmp/_bt_5way_v2_xasset_models',
    },
}
MODEL_KEYS = list(MODELS.keys())

FOLD_ID = 5
SCORE_MINUTE = 375
ENTRY_MINUTE = 385
EXIT_TIME_MIN = 780
SLIPPAGE_PCT = 0.0015

D_MODEL = 64
N_HEADS = 4
D_FF = 128
N_BLOCKS = 2
N_BLOCKS_15M = 1
DROPOUT = 0.1
FILM_HIDDEN = 64
TIME_STEPS = 128
TIME_STEPS_15M = 128
N_HORIZONS = 3
HORIZON_LABELS = {0: '15m', 1: '30m', 2: '60m'}
TARGET_HORIZONS = [15, 30, 60]

PROB_THRESHOLDS = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
TPSL_PAIRS = [
    {'tp_pct': 0.06, 'sl_pct': 0.03},
    {'tp_pct': 0.08, 'sl_pct': 0.04},
    {'tp_pct': 0.12, 'sl_pct': 0.06},
    {'tp_pct': 0.16, 'sl_pct': 0.08},
]

DEPLOY_PNL_THRESHOLD = 0.018
MIN_TRADES_PER_MONTH = 2.0

CRYPTO_TICKERS = {
    'MARA', 'RIOT', 'CLSK', 'COIN', 'CORZ', 'CIFR', 'HUT', 'IREN',
    'WULF', 'BTBT', 'MSTR', 'HOOD',
}

TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}
EXCLUDE_TICKERS = {'SPY', 'VIX', 'TLT', 'IWM', 'XLE', 'ITA', 'BITF', 'MIGI', 'SMLR'}

PST = pytz.timezone('America/Los_Angeles')
SEED = 42
np.random.seed(SEED)
W = 110

OUTPUT_DIR = '/tmp/_bt_xasset_comparison'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================================================================
# GCS HELPERS
# ==============================================================================
_gcs_client = storage.Client()

def _bucket():
    return _gcs_client.bucket(GCS_BUCKET)

def download_from_gcs(gcs_path, local_path):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    _bucket().blob(gcs_path).download_to_filename(local_path)

def ensure_file(filename, gcs_prefix, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, filename)
    if os.path.exists(local_path):
        return local_path
    gcs_path = f'{gcs_prefix}/{filename}'
    print(f"    Downloading {filename}...")
    download_from_gcs(gcs_path, local_path)
    return local_path

def ensure_model_weights(fold_id, gcs_prefix, local_dir):
    local_fold = os.path.join(local_dir, f'fold_{fold_id}')
    weights_path = os.path.join(local_fold, 'model.weights.h5')
    if os.path.exists(weights_path):
        return weights_path
    os.makedirs(local_fold, exist_ok=True)
    gcs_fold = f'{gcs_prefix}/fold_{fold_id}'
    print(f"    Downloading model fold {fold_id}...")
    blobs = list(_bucket().list_blobs(prefix=gcs_fold))
    for blob in blobs:
        fname = blob.name.split('/')[-1]
        blob.download_to_filename(os.path.join(local_fold, fname))
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
    def __init__(self, d_model, n_heads, dropout=0.1, **kw):
        super().__init__(**kw)
        self.d_model, self.n_heads = d_model, n_heads
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
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, **kw):
        super().__init__(**kw)
        self.attn = DeStationaryAttention(d_model, n_heads, dropout)
        self.ffn = tf.keras.Sequential([
            layers.Dense(d_ff, activation='gelu'),
            layers.Dense(d_model),
            layers.Dropout(dropout)])
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.drop = layers.Dropout(dropout)
    def call(self, x, stats, training=False):
        x = x + self.drop(
            self.attn(self.norm1(x), stats, training=training), training=training)
        x = x + self.ffn(self.norm2(x), training=training)
        return x

class TransformerBackbone(layers.Layer):
    def __init__(self, d_model, n_heads, d_ff, n_blocks, seq_len,
                 dropout=0.1, **kw):
        super().__init__(**kw)
        self.stationarize = SeriesStationarization()
        self.input_proj = layers.Dense(d_model)
        self.pos_embed = self.add_weight(
            name='pos_embed', shape=(1, seq_len, d_model),
            initializer='glorot_normal', trainable=True)
        self.input_drop = layers.Dropout(dropout)
        self.blocks = [TransformerBlock(d_model, n_heads, d_ff, dropout,
                                        name=f'block_{i}')
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
    def __init__(self, d_model, **kw):
        super().__init__(**kw)
        self.query = self.add_weight(
            name='pool_query', shape=(1, 1, d_model),
            initializer='glorot_normal', trainable=True)
        self.scale = tf.sqrt(tf.cast(d_model, tf.float32))
    def call(self, x, training=False):
        scores = tf.matmul(self.query, x, transpose_b=True) / self.scale
        weights = tf.nn.softmax(scores, axis=-1)
        return tf.matmul(weights, x)[:, 0, :]

class FiLMConditioner(layers.Layer):
    def __init__(self, d_model, hidden_dim=64, **kw):
        super().__init__(**kw)
        self.mlp = tf.keras.Sequential([
            layers.Dense(hidden_dim, activation='relu'),
            layers.Dense(hidden_dim, activation='relu')])
        self.gamma_proj = layers.Dense(d_model)
        self.beta_proj = layers.Dense(d_model)
    def call(self, x, regime_features, training=False):
        h = self.mlp(regime_features, training=training)
        return (1.0 + self.gamma_proj(h)) * x + self.beta_proj(h)

class SpikeTransformer(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.backbone  = TransformerBackbone(
            D_MODEL, N_HEADS, D_FF, N_BLOCKS, TIME_STEPS, DROPOUT, name='backbone')
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

class MultiResSpikeTransformer(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.backbone_1m  = TransformerBackbone(
            D_MODEL, N_HEADS, D_FF, N_BLOCKS, TIME_STEPS, DROPOUT, name='backbone_1m')
        self.pool_1m      = AttentionPooling(D_MODEL, name='pool_1m')
        self.backbone_15m = TransformerBackbone(
            D_MODEL, N_HEADS, D_FF, N_BLOCKS_15M, TIME_STEPS_15M, DROPOUT, name='backbone_15m')
        self.pool_15m     = AttentionPooling(D_MODEL, name='pool_15m')
        self.fusion = layers.Dense(D_MODEL, activation='gelu', name='fusion')
        self.film      = FiLMConditioner(D_MODEL, FILM_HIDDEN, name='film')
        self.drop      = layers.Dropout(DROPOUT)
        self.head      = layers.Dense(N_HORIZONS, activation='sigmoid',
                                       name='spike_head')

    def call(self, inputs, training=False):
        h_1m  = self.backbone_1m(inputs['seq_input'], training=training)
        h_1m  = self.pool_1m(h_1m, training=training)
        h_15m = self.backbone_15m(inputs['seq_15m_input'], training=training)
        h_15m = self.pool_15m(h_15m, training=training)
        h = self.fusion(tf.concat([h_1m, h_15m], axis=-1))
        h = self.film(h, inputs['regime_input'], training=training)
        h = self.drop(h, training=training)
        return self.head(h)

def build_model(n_asset_features, n_regime_features, arch='standard', n_lowres_features=None):
    if arch == 'multires':
        model = MultiResSpikeTransformer(name='multires_spike_transformer')
        dummy_seq = tf.zeros((1, TIME_STEPS, n_asset_features))
        dummy_15m = tf.zeros((1, TIME_STEPS_15M, n_lowres_features))
        dummy_reg = tf.zeros((1, n_regime_features))
        model({'seq_input': dummy_seq, 'seq_15m_input': dummy_15m,
               'regime_input': dummy_reg}, training=False)
    else:
        model = SpikeTransformer(name='spike_transformer')
        dummy_seq = tf.zeros((1, TIME_STEPS, n_asset_features))
        dummy_reg = tf.zeros((1, n_regime_features))
        model({'seq_input': dummy_seq, 'regime_input': dummy_reg}, training=False)
    return model


# ==============================================================================
# SHORT TRADE SIM (Numba)
# ==============================================================================
@jit(nopython=True)
def run_short_trade_sim(entry_indices, prices, highs, lows, date_ints, time_mins,
                        tp, sl, exit_time):
    n_entries = len(entry_indices)
    n_prices = len(prices)
    trades = np.empty(n_entries, dtype=np.float64)
    entry_dates = np.empty(n_entries, dtype=np.int64)
    trade_count = 0
    current_entry_ptr = 0
    last_traded_date = -1
    while current_entry_ptr < n_entries:
        idx = entry_indices[current_entry_ptr]
        if idx >= n_prices - 1:
            break
        entry_date = date_ints[idx]
        if entry_date == last_traded_date:
            current_entry_ptr += 1
            continue
        entry_price = prices[idx]
        target_price = entry_price * (1 - tp)
        stop_price = entry_price * (1 + sl)
        outcome = 0.0
        trade_done = False
        exit_idx = idx
        for k in range(idx + 1, n_prices):
            if date_ints[k] != entry_date:
                outcome = (entry_price - prices[k-1]) / entry_price
                exit_idx = k; trade_done = True; break
            tp_hit = lows[k] <= target_price
            sl_hit = highs[k] >= stop_price
            if sl_hit and tp_hit:
                outcome = -sl if np.random.random() < 0.5 else tp
                exit_idx = k; trade_done = True; break
            if sl_hit:
                outcome = -sl; exit_idx = k; trade_done = True; break
            if tp_hit:
                outcome = tp; exit_idx = k; trade_done = True; break
            if time_mins[k] >= exit_time:
                outcome = (entry_price - prices[k]) / entry_price
                exit_idx = k; trade_done = True; break
        if not trade_done:
            outcome = (entry_price - prices[n_prices-1]) / entry_price
            exit_idx = n_prices
        trades[trade_count] = outcome
        entry_dates[trade_count] = entry_date
        trade_count += 1
        last_traded_date = entry_date
        next_ptr = current_entry_ptr + 1
        while next_ptr < n_entries:
            if entry_indices[next_ptr] > exit_idx:
                break
            next_ptr += 1
        current_entry_ptr = next_ptr
    return trades[:trade_count], entry_dates[:trade_count]


# ==============================================================================
# OHLCV LOADING (for trade sim)
# ==============================================================================
def load_ticker_ohlcv(ticker):
    clean = TICKER_FILE_MAP.get(ticker, ticker)
    for suffix in ['_master_inference', '_1m_full_historical_data']:
        blob_path = f"{GCS_INFER_DIR}/{clean}{suffix}.parquet"
        local_path = f"/tmp/_bt_xasset_{clean}{suffix}.parquet"
        try:
            _bucket().blob(blob_path).download_to_filename(local_path)
            df = pd.read_parquet(local_path)
            os.remove(local_path)
            df = df.reset_index()
            dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
            if dt_cols:
                df['Datetime'] = pd.to_datetime(df[dt_cols[0]], utc=True)
            else:
                num_col = next((c for c in df.columns
                               if c in ('Datetime', 'Date', 'Time', 'Timestamp', 'Level_0')),
                              df.columns[0])
                df['Datetime'] = pd.to_datetime(df[num_col], unit='ms', utc=True)
            df = df.set_index('Datetime').sort_index()
            df = df[~df.index.duplicated(keep='first')]
            std_map = {'open': 'Open', 'high': 'High', 'low': 'Low',
                       'close': 'Close', 'volume': 'Volume'}
            lower_map = {k.lower(): v for k, v in std_map.items()}
            df.columns = [lower_map.get(str(c).strip().lower(), str(c).strip())
                          for c in df.columns]
            df.index = df.index.tz_convert(PST)
            return df
        except Exception:
            if os.path.exists(local_path):
                os.remove(local_path)
            continue
    return None


def precompute_ohlcv(df, start_date, end_date):
    from datetime import timedelta
    s = pd.to_datetime(start_date).tz_localize('America/Los_Angeles')
    e = pd.to_datetime(end_date).tz_localize('America/Los_Angeles') + timedelta(days=1)
    df = df[(df.index >= s) & (df.index < e)]
    if len(df) == 0:
        return None
    dates = (df.index.year * 10000 + df.index.month * 100 + df.index.day).values.astype(np.int64)
    times = (df.index.hour * 60 + df.index.minute).values
    return {
        'prices': df['Close'].values.copy(),
        'highs': df['High'].values.copy(),
        'lows': df['Low'].values.copy(),
        'dates': dates,
        'times': times,
        'months': max(1.0, (df.index[-1] - df.index[0]).days / 30.44),
    }


# ==============================================================================
# PHASE 1: LOAD MODELS & SCORE TEST SET
# ==============================================================================
t_total = time.time()

print(f"\n{'=' * W}")
print(f"  SHORT L5-2: {len(MODEL_KEYS)}-WAY MODEL COMPARISON")
print(f"  Fold {FOLD_ID} TEST set (Jul 16 - Aug 21, 2026)")
print(f"  Models: {', '.join(MODELS[mk]['label'] for mk in MODEL_KEYS)}")
print(f"{'=' * W}")

model_preds = {}
model_meta = {}

for model_key, mcfg in MODELS.items():
    print(f"\n{'─' * W}")
    print(f"  Loading {mcfg['label']}...")
    print(f"{'─' * W}")

    meta_path = ensure_file('meta.npz', mcfg['gcs_data'], mcfg['local_data'])
    meta = np.load(meta_path, allow_pickle=True)
    ticker_names = meta['ticker_names']

    arch = mcfg.get('arch', 'standard')
    test_path = ensure_file(f'fold_{FOLD_ID}_test.npz', mcfg['gcs_data'], mcfg['local_data'])
    test_data = np.load(test_path)
    X_seq = test_data['X_seq']
    X_reg = test_data['X_regime']
    X_id = test_data['X_id'].flatten()
    Y = test_data['Y']
    Y_cont = test_data['Y_continuous']
    date_ints = test_data['date_ints']
    minutes = test_data['minutes']
    X_seq_15m = test_data['X_seq_15m'] if arch == 'multires' else None
    test_data.close()

    n_asset_f = X_seq.shape[-1]
    n_regime_f = X_reg.shape[-1]
    n_lowres_f = X_seq_15m.shape[-1] if X_seq_15m is not None else 0
    n_total = len(X_id)

    extra = f", {n_lowres_f} lowres features" if n_lowres_f > 0 else ""
    print(f"    Test set: {n_total:,} samples, {n_asset_f} seq features, {n_regime_f} regime features{extra}")
    print(f"    Tickers in meta: {len(ticker_names)}")

    entry_mask = (minutes == SCORE_MINUTE)
    n_entry = entry_mask.sum()
    print(f"    Score-minute ({SCORE_MINUTE}) samples: {n_entry:,}")

    X_seq_e = X_seq[entry_mask]
    X_reg_e = X_reg[entry_mask]
    X_seq_15m_e = X_seq_15m[entry_mask] if X_seq_15m is not None else None
    X_id_e = X_id[entry_mask]
    Y_e = Y[entry_mask]
    Y_cont_e = Y_cont[entry_mask]
    date_ints_e = date_ints[entry_mask]
    del X_seq, X_reg, X_id, Y, Y_cont, date_ints, minutes, entry_mask, X_seq_15m
    gc.collect()

    print(f"    Building model (arch={arch}, n_asset={n_asset_f}, n_regime={n_regime_f})...")
    model = build_model(n_asset_f, n_regime_f, arch=arch,
                        n_lowres_features=n_lowres_f if n_lowres_f > 0 else None)
    weights_path = ensure_model_weights(FOLD_ID, mcfg['gcs_models'], mcfg['local_model'])
    model.load_weights(weights_path)
    print(f"    Model loaded: {model.count_params():,} params")

    print(f"    Running inference on {n_entry:,} samples...")
    if arch == 'multires':
        predict_inputs = {
            'seq_input': X_seq_e.astype(np.float32),
            'seq_15m_input': X_seq_15m_e.astype(np.float32),
            'regime_input': X_reg_e.astype(np.float32)}
    else:
        predict_inputs = {
            'seq_input': X_seq_e.astype(np.float32),
            'regime_input': X_reg_e.astype(np.float32)}
    preds = model.predict(predict_inputs, batch_size=512, verbose=0)
    del X_seq_15m_e

    model_preds[model_key] = {
        'preds': preds,
        'Y': Y_e,
        'Y_cont': Y_cont_e,
        'X_id': X_id_e,
        'date_ints': date_ints_e,
        'ticker_names': ticker_names,
        'n_asset_f': n_asset_f,
        'n_regime_f': n_regime_f,
    }

    del model, X_seq_e, X_reg_e
    tf.keras.backend.clear_session()
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

    dump_rates = Y_e.mean(axis=0)
    pred_means = preds.mean(axis=0)
    print(f"    Dump rates:  " + "  ".join(
        f"{HORIZON_LABELS[i]}={dump_rates[i]*100:.1f}%" for i in range(N_HORIZONS)))
    print(f"    Avg preds:   " + "  ".join(
        f"{HORIZON_LABELS[i]}={pred_means[i]*100:.1f}%" for i in range(N_HORIZONS)))

    if np.isnan(preds).any():
        print(f"    *** WARNING: {np.isnan(preds).sum()} NaN predictions — "
              f"model trained on corrupted data, SKIPPING ***")
        model_preds[model_key]['_skip'] = True


# ==============================================================================
# PHASE 2: ML METRICS COMPARISON
# ==============================================================================
print(f"\n\n{'=' * W}")
print(f"  PHASE 2: ML METRICS — {len(MODEL_KEYS)}-WAY COMPARISON")
print(f"{'=' * W}")

ml_results = {}

for model_key in MODELS:
    d = model_preds[model_key]
    if d.get('_skip'):
        print(f"\n  {MODELS[model_key]['label']}: SKIPPED (NaN predictions)")
        ml_results[model_key] = {HORIZON_LABELS[h]: {
            'auc': 0, 'ap': 0, 'brier': 1, 'n_pos': 0, 'n_neg': 0,
            'thresh_metrics': {t: {'precision': 0, 'recall': 0, 'f1': 0, 'n_predicted': 0}
                               for t in PROB_THRESHOLDS},
            'cal_pred': [], 'cal_actual': [], 'cal_count': [],
        } for h in range(N_HORIZONS)}
        continue
    preds = d['preds']
    Y = d['Y']
    ml_results[model_key] = {}

    for h_idx in range(N_HORIZONS):
        h_label = HORIZON_LABELS[h_idx]
        y_true = Y[:, h_idx]
        y_pred = preds[:, h_idx]

        n_pos = int(y_true.sum())
        n_neg = len(y_true) - n_pos

        auc = roc_auc_score(y_true, y_pred) if n_pos > 0 and n_neg > 0 else 0.0
        ap = average_precision_score(y_true, y_pred) if n_pos > 0 else 0.0
        brier = brier_score_loss(y_true, y_pred)

        thresh_metrics = {}
        for t in PROB_THRESHOLDS:
            y_hat = (y_pred >= t).astype(float)
            n_predicted = y_hat.sum()
            if n_predicted > 0:
                prec = precision_score(y_true, y_hat, zero_division=0)
                rec = recall_score(y_true, y_hat, zero_division=0)
                f1 = f1_score(y_true, y_hat, zero_division=0)
            else:
                prec, rec, f1 = 0, 0, 0
            thresh_metrics[t] = {
                'precision': prec, 'recall': rec, 'f1': f1,
                'n_predicted': int(n_predicted),
            }

        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        cal_pred, cal_actual, cal_count = [], [], []
        for i in range(n_bins):
            mask = (y_pred >= bin_edges[i]) & (y_pred < bin_edges[i+1])
            if mask.sum() > 0:
                cal_pred.append(float(y_pred[mask].mean()))
                cal_actual.append(float(y_true[mask].mean()))
                cal_count.append(int(mask.sum()))

        ml_results[model_key][h_label] = {
            'auc': auc, 'ap': ap, 'brier': brier,
            'n_pos': n_pos, 'n_neg': n_neg,
            'thresh_metrics': thresh_metrics,
            'cal_pred': cal_pred, 'cal_actual': cal_actual, 'cal_count': cal_count,
        }

exp_keys = [mk for mk in MODEL_KEYS if mk != 'BASELINE']
print(f"\n  {'Metric':<20} {'Horizon':>8}", end='')
for mk in MODEL_KEYS:
    print(f"  {MODELS[mk]['label']:>18}", end='')
for ek in exp_keys:
    print(f"  {'Δ'+MODELS[ek]['label'][:6]:>10}", end='')
print()
print(f"  {'─' * (30 + 20 * len(MODEL_KEYS) + 12 * len(exp_keys))}")

for h_idx in range(N_HORIZONS):
    h_label = HORIZON_LABELS[h_idx]
    for metric_name, metric_key, fmt, higher_better in [
        ('AUC-ROC', 'auc', '.4f', True),
        ('Avg Precision', 'ap', '.4f', True),
        ('Brier Score', 'brier', '.4f', False),
    ]:
        vals = {mk: ml_results[mk][h_label][metric_key] for mk in MODEL_KEYS}
        print(f"  {metric_name:<20} {h_label:>8}", end='')
        for mk in MODEL_KEYS:
            print(f"  {vals[mk]:>18{fmt}}", end='')
        for ek in exp_keys:
            delta = vals[ek] - vals['BASELINE']
            sign = '+' if delta > 0 else ''
            better = (delta > 0) == higher_better
            marker = ' *' if better and abs(delta) > 0.005 else ''
            print(f"  {sign}{delta:>8{fmt}}{marker}", end='')
        print()
    print()

for metric_label, metric_sub in [('Precision', 'precision'), ('Recall', 'recall')]:
    print(f"\n  {metric_label} @ Threshold:")
    print(f"  {'Thresh':>8} {'Horizon':>8}", end='')
    for mk in MODEL_KEYS:
        print(f"  {MODELS[mk]['label']:>18}", end='')
    print()
    print(f"  {'─' * (18 + 20 * len(MODEL_KEYS))}")
    for h_idx in range(N_HORIZONS):
        h_label = HORIZON_LABELS[h_idx]
        for t in [0.30, 0.50, 0.70]:
            print(f"  {t:>8.2f} {h_label:>8}", end='')
            for mk in MODEL_KEYS:
                v = ml_results[mk][h_label]['thresh_metrics'][t][metric_sub]
                print(f"  {v:>17.1%}", end='')
            print()


# ==============================================================================
# PHASE 3: PER-TICKER AUC COMPARISON
# ==============================================================================
print(f"\n\n{'=' * W}")
print(f"  PHASE 3: PER-TICKER AUC COMPARISON ({len(MODEL_KEYS)} models)")
print(f"{'=' * W}")

all_tickers_phase3 = set()
for mk in MODEL_KEYS:
    for tn in model_preds[mk]['ticker_names']:
        if str(tn) not in EXCLUDE_TICKERS:
            all_tickers_phase3.add(str(tn))

h_compare = 1
h_label = HORIZON_LABELS[h_compare]
print(f"\n  Comparing on {h_label} horizon (primary SHORT horizon)")

ticker_auc = {}
for tk in sorted(all_tickers_phase3):
    row = {}
    for model_key in MODEL_KEYS:
        d = model_preds[model_key]
        if d.get('_skip'):
            continue
        tnames = d['ticker_names']
        tid = None
        for i, tn in enumerate(tnames):
            if str(tn) == tk:
                tid = i
                break
        if tid is None:
            continue
        mask = d['X_id'] == tid
        if mask.sum() < 10:
            continue
        y_true = d['Y'][mask, h_compare]
        y_pred = d['preds'][mask, h_compare]
        if y_true.sum() > 0 and (1 - y_true).sum() > 0:
            row[model_key] = roc_auc_score(y_true, y_pred)
        else:
            row[model_key] = None
    if 'BASELINE' in row and row['BASELINE'] is not None:
        has_any_exp = any(ek in row and row[ek] is not None for ek in exp_keys)
        if has_any_exp:
            ticker_auc[tk] = row

if ticker_auc:
    auc_df = pd.DataFrame(ticker_auc).T
    for ek in exp_keys:
        col = f'delta_{ek}'
        if ek in auc_df.columns:
            auc_df[col] = auc_df[ek] - auc_df['BASELINE']
        else:
            auc_df[col] = np.nan
    auc_df['is_crypto'] = auc_df.index.isin(CRYPTO_TICKERS)

    for ek in exp_keys:
        col = f'delta_{ek}'
        valid = auc_df[col].dropna()
        n_imp = (valid > 0).sum()
        n_deg = (valid < 0).sum()
        avg_l = valid.mean()
        crypto_l = valid[auc_df.loc[valid.index, 'is_crypto']].mean() if auc_df.loc[valid.index, 'is_crypto'].any() else 0
        noncrypto_l = valid[~auc_df.loc[valid.index, 'is_crypto']].mean() if (~auc_df.loc[valid.index, 'is_crypto']).any() else 0
        print(f"\n  {MODELS[ek]['label']} vs Baseline:")
        print(f"    {n_imp} improved / {n_deg} degraded / {len(valid)} total")
        print(f"    Avg AUC lift:         {avg_l:+.4f}")
        print(f"    Crypto tickers lift:  {crypto_l:+.4f}")
        print(f"    Non-crypto lift:      {noncrypto_l:+.4f}")

    print(f"\n  {'Ticker':<8} {'Baseline':>10}", end='')
    for ek in exp_keys:
        print(f" {MODELS[ek]['label'][:10]:>10} {'Δ':>6}", end='')
    print(f" {'Type':>8}")
    print(f"  {'─' * (20 + 18 * len(exp_keys) + 10)}")
    sort_col = f'delta_{exp_keys[0]}'
    auc_df_sorted = auc_df.sort_values(sort_col, ascending=False, na_position='last')
    for tk, r in auc_df_sorted.iterrows():
        tag = 'CRYPTO' if r['is_crypto'] else ''
        print(f"  {tk:<8} {r['BASELINE']:>10.4f}", end='')
        for ek in exp_keys:
            v = r.get(ek, np.nan)
            d = r.get(f'delta_{ek}', np.nan)
            if pd.notna(v):
                print(f" {v:>10.4f} {d:>+6.4f}", end='')
            else:
                print(f" {'n/a':>10} {'':>6}", end='')
        print(f" {tag:>8}")


# ==============================================================================
# PHASE 4: TRADE SIMULATION COMPARISON
# ==============================================================================
print(f"\n\n{'=' * W}")
print(f"  PHASE 4: TRADE SIMULATION — {len(MODEL_KEYS)}-WAY COMPARISON")
print(f"  Standard grid: {len(PROB_THRESHOLDS)} thresholds x {len(TPSL_PAIRS)} TP/SL")
print(f"  Test period: 2026-07-16 to 2026-08-21")
print(f"{'=' * W}")

TEST_START = '2026-07-16'
TEST_END = '2026-08-21'

trade_results = {mk: [] for mk in MODELS}

scored_tickers = set()
for mk in MODELS:
    d = model_preds[mk]
    for i in range(len(d['X_id'])):
        tid = int(d['X_id'][i])
        if tid < len(d['ticker_names']):
            tk = str(d['ticker_names'][tid])
            if tk not in EXCLUDE_TICKERS:
                scored_tickers.add(tk)

scored_tickers = sorted(scored_tickers)
print(f"  {len(scored_tickers)} tickers to simulate")

for t_idx, ticker in enumerate(scored_tickers):
    df_ohlcv = load_ticker_ohlcv(ticker)
    if df_ohlcv is None:
        continue
    ohlcv = precompute_ohlcv(df_ohlcv, TEST_START, TEST_END)
    if ohlcv is None:
        del df_ohlcv
        continue

    for model_key in MODELS:
        d = model_preds[model_key]
        if d.get('_skip'):
            continue
        tnames = d['ticker_names']
        tid = None
        for i, tn in enumerate(tnames):
            if str(tn) == ticker:
                tid = i
                break
        if tid is None:
            continue

        mask = d['X_id'] == tid
        if mask.sum() == 0:
            continue

        preds_tk = d['preds'][mask]
        dates_tk = d['date_ints'][mask]

        for h_idx in range(N_HORIZONS):
            h_label = HORIZON_LABELS[h_idx]
            score_by_date = {}
            for j in range(len(preds_tk)):
                d_int = int(dates_tk[j])
                score_by_date[d_int] = float(preds_tk[j, h_idx])

            for prob_thresh in PROB_THRESHOLDS:
                entry_list = []
                for i in range(len(ohlcv['dates'])):
                    if ohlcv['times'][i] == ENTRY_MINUTE:
                        day_score = score_by_date.get(int(ohlcv['dates'][i]), 0.0)
                        if day_score >= prob_thresh:
                            entry_list.append(i)
                if not entry_list:
                    continue
                indices = np.array(entry_list, dtype=np.int64)

                for pair in TPSL_PAIRS:
                    pnl, _ = run_short_trade_sim(
                        indices, ohlcv['prices'], ohlcv['highs'], ohlcv['lows'],
                        ohlcv['dates'], ohlcv['times'].astype(np.int64),
                        pair['tp_pct'], pair['sl_pct'], EXIT_TIME_MIN)
                    if len(pnl) > 0 and SLIPPAGE_PCT > 0:
                        pnl = pnl - 2 * SLIPPAGE_PCT
                    if len(pnl) < 2:
                        continue
                    trade_results[model_key].append({
                        'ticker': ticker,
                        'horizon': h_label,
                        'prob_thresh': prob_thresh,
                        'tp_pct': pair['tp_pct'],
                        'sl_pct': pair['sl_pct'],
                        'avg_pnl': float(np.mean(pnl)),
                        'total_pnl': float(np.sum(pnl)),
                        'win_rate': float(np.mean(pnl > 0)),
                        'n_trades': len(pnl),
                        'freq': len(pnl) / ohlcv['months'],
                        'is_crypto': ticker in CRYPTO_TICKERS,
                    })

    del df_ohlcv
    if (t_idx + 1) % 15 == 0:
        gc.collect()
        print(f"    {t_idx+1}/{len(scored_tickers)} tickers simulated")

print(f"    Simulation complete")

for mk in MODELS:
    df_r = pd.DataFrame(trade_results[mk])
    if len(df_r) == 0:
        print(f"\n  {MODELS[mk]['label']}: No trades")
        continue
    pos = df_r[df_r['avg_pnl'] > 0]
    deploy = df_r[df_r['avg_pnl'] >= DEPLOY_PNL_THRESHOLD]
    print(f"\n  {MODELS[mk]['label']}:")
    print(f"    Total grid combos:  {len(df_r):,}")
    print(f"    Positive PnL:       {len(pos):,} ({len(pos)/len(df_r)*100:.0f}%)")
    print(f"    Deployable (>={DEPLOY_PNL_THRESHOLD:.1%}): {len(deploy):,}")
    if len(deploy) > 0:
        print(f"    Deploy tickers:     {deploy['ticker'].nunique()}")
        print(f"    Deploy avg PnL:     {deploy['avg_pnl'].mean():.2%}")

best_per_ticker = {}
for mk in MODEL_KEYS:
    df_r = pd.DataFrame(trade_results[mk])
    if len(df_r) == 0:
        continue
    best = df_r.loc[df_r.groupby('ticker')['avg_pnl'].idxmax()]
    best_per_ticker[mk] = best.set_index('ticker')['avg_pnl'].to_dict()

baseline_tickers = set(best_per_ticker.get('BASELINE', {}).keys())
pnl_summary = {}
for ek in exp_keys:
    common_trade_tickers = baseline_tickers & set(best_per_ticker.get(ek, {}).keys())
    if not common_trade_tickers:
        continue
    pnl_comparison = []
    for tk in sorted(common_trade_tickers):
        b = best_per_ticker['BASELINE'].get(tk, 0)
        e = best_per_ticker[ek].get(tk, 0)
        pnl_comparison.append({'ticker': tk, 'baseline': b, 'exp': e,
                               'delta': e - b, 'is_crypto': tk in CRYPTO_TICKERS})
    pnl_df = pd.DataFrame(pnl_comparison).sort_values('delta', ascending=False)
    pnl_summary[ek] = pnl_df

    n_better = (pnl_df['delta'] > 0).sum()
    avg_delta = pnl_df['delta'].mean()
    crypto_delta = pnl_df[pnl_df['is_crypto']]['delta'].mean() if pnl_df['is_crypto'].any() else 0

    print(f"\n  {MODELS[ek]['label']} vs Baseline ({len(common_trade_tickers)} common tickers):")
    print(f"  {'Ticker':<8} {'Baseline':>10} {MODELS[ek]['label'][:10]:>10} {'Delta':>10} {'Type':>8}")
    print(f"  {'─' * 55}")
    for _, r in pnl_df.iterrows():
        tag = 'CRYPTO' if r['is_crypto'] else ''
        print(f"  {r['ticker']:<8} {r['baseline']:>+9.2%} {r['exp']:>+9.2%} "
              f"{r['delta']:>+9.2%} {tag:>8}")
    print(f"\n  {n_better}/{len(pnl_df)} tickers improved")
    print(f"  Avg PnL lift:        {avg_delta:+.2%}")
    print(f"  Crypto PnL lift:     {crypto_delta:+.2%}")


# ==============================================================================
# CHARTS
# ==============================================================================
print(f"\n\n{'=' * W}")
print(f"  GENERATING CHARTS")
print(f"{'=' * W}")

n_models = len(MODEL_KEYS)
colors = {}
color_palette = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63', '#9C27B0']
for i, mk in enumerate(MODEL_KEYS):
    colors[mk] = color_palette[i % len(color_palette)]

fig, axes = plt.subplots(2, 3, figsize=(24, 13))
fig.suptitle(f'SHORT L5-2: {n_models}-WAY MODEL COMPARISON',
             fontsize=14, fontweight='bold')

# [0,0] AUC-ROC by horizon
ax = axes[0, 0]
x_pos = np.arange(N_HORIZONS)
width = 0.8 / n_models
for i, mk in enumerate(MODEL_KEYS):
    aucs = [ml_results[mk][HORIZON_LABELS[h]]['auc'] for h in range(N_HORIZONS)]
    offset = (i - (n_models - 1) / 2) * width
    ax.bar(x_pos + offset, aucs, width, label=MODELS[mk]['label'],
           color=colors[mk], alpha=0.85)
    for j, v in enumerate(aucs):
        ax.text(x_pos[j] + offset, v + 0.002, f'{v:.3f}',
                ha='center', fontsize=7, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels([HORIZON_LABELS[h] for h in range(N_HORIZONS)])
ax.set_ylabel('AUC-ROC')
ax.set_title('AUC-ROC by Horizon')
ax.legend(fontsize=7)
ax.grid(axis='y', alpha=0.3)

# [0,1] Precision-Recall curve (30m horizon)
ax = axes[0, 1]
for mk in MODEL_KEYS:
    d = model_preds[mk]
    if d.get('_skip'):
        continue
    y_true = d['Y'][:, 1]
    y_pred = d['preds'][:, 1]
    prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_pred)
    ap = average_precision_score(y_true, y_pred)
    ax.plot(rec_curve, prec_curve, label=f"{MODELS[mk]['label']} (AP={ap:.4f})",
            color=colors[mk], linewidth=2)
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve (30m)')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# [0,2] Calibration curve (30m horizon)
ax = axes[0, 2]
for mk in MODEL_KEYS:
    cal = ml_results[mk]['30m']
    ax.plot(cal['cal_pred'], cal['cal_actual'],
            'o-', label=MODELS[mk]['label'], color=colors[mk],
            markersize=5, linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect')
ax.set_xlabel('Mean Predicted Probability')
ax.set_ylabel('Actual Dump Rate')
ax.set_title('Calibration Curve (30m)')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# [1,0] Score distribution (30m horizon)
ax = axes[1, 0]
for mk in MODEL_KEYS:
    d = model_preds[mk]
    if d.get('_skip'):
        continue
    scores = d['preds'][:, 1]
    ax.hist(scores, bins=50, alpha=0.4, label=MODELS[mk]['label'],
            color=colors[mk], density=True)
ax.set_xlabel('Predicted Probability')
ax.set_ylabel('Density')
ax.set_title('Score Distribution (30m)')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# [1,1] Per-ticker AUC lift (first exp model vs baseline)
ax = axes[1, 1]
if ticker_auc:
    first_ek = exp_keys[0]
    dcol = f'delta_{first_ek}'
    valid_auc = auc_df[auc_df[dcol].notna()].sort_values(dcol, ascending=True)
    bar_colors = [colors[first_ek] if valid_auc.loc[t, 'is_crypto'] else '#90CAF9'
                  for t in valid_auc.index]
    ax.barh(range(len(valid_auc)), valid_auc[dcol].values * 100,
            color=bar_colors, edgecolor='white', linewidth=0.3)
    ax.set_yticks(range(len(valid_auc)))
    ax.set_yticklabels(valid_auc.index.tolist(), fontsize=5)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('AUC Lift (%)')
    ax.set_title(f'Per-Ticker AUC Lift ({MODELS[first_ek]["label"]} vs Baseline)')
    ax.grid(axis='x', alpha=0.3)

# [1,2] Best PnL per ticker comparison (first exp model)
ax = axes[1, 2]
first_ek = exp_keys[0]
if first_ek in pnl_summary:
    pdf = pnl_summary[first_ek].sort_values('delta', ascending=True)
    bar_colors = [colors[first_ek] if r['is_crypto'] else '#90CAF9'
                  for _, r in pdf.iterrows()]
    ax.barh(range(len(pdf)), pdf['delta'].values * 100,
            color=bar_colors, edgecolor='white', linewidth=0.3)
    ax.set_yticks(range(len(pdf)))
    ax.set_yticklabels(pdf['ticker'].values, fontsize=5)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Best PnL Lift (%)')
    ax.set_title(f'Per-Ticker PnL Lift ({MODELS[first_ek]["label"]} vs Baseline)')
    ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
chart_path = os.path.join(OUTPUT_DIR, '5way_model_comparison.png')
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
print(f"  Chart saved: {chart_path}")
if _IN_COLAB:
    plt.show()
plt.close()


# ==============================================================================
# PHASE 5: PnL PERFORMANCE OVER TEST WINDOW
# ==============================================================================
print(f"\n\n{'=' * W}")
print(f"  PHASE 5: PnL PERFORMANCE OVER TEST WINDOW")
print(f"  Best config per ticker, daily cumulative PnL, weekly breakdown")
print(f"{'=' * W}")

daily_pnl_by_model = {}

for mk in MODEL_KEYS:
    d = model_preds[mk]
    tnames = d['ticker_names']
    arch = MODELS[mk].get('arch', 'standard')

    df_r = pd.DataFrame(trade_results[mk])
    if len(df_r) == 0:
        daily_pnl_by_model[mk] = pd.Series(dtype=float)
        continue

    deployable = df_r[df_r['avg_pnl'] >= DEPLOY_PNL_THRESHOLD]
    if len(deployable) == 0:
        daily_pnl_by_model[mk] = pd.Series(dtype=float)
        continue

    best_cfg = deployable.loc[deployable.groupby('ticker')['avg_pnl'].idxmax()]
    deploy_tickers = best_cfg['ticker'].unique()

    all_daily = []
    for _, cfg_row in best_cfg.iterrows():
        ticker = cfg_row['ticker']
        h_label = cfg_row['horizon']
        prob_thresh = cfg_row['prob_thresh']
        tp_pct = cfg_row['tp_pct']
        sl_pct = cfg_row['sl_pct']
        h_idx = {v: k for k, v in HORIZON_LABELS.items()}[h_label]

        tid = None
        for i, tn in enumerate(tnames):
            if str(tn) == ticker:
                tid = i
                break
        if tid is None:
            continue

        mask = d['X_id'] == tid
        if mask.sum() == 0:
            continue

        preds_tk = d['preds'][mask]
        dates_tk = d['date_ints'][mask]

        score_by_date = {}
        for j in range(len(preds_tk)):
            d_int = int(dates_tk[j])
            score_by_date[d_int] = float(preds_tk[j, h_idx])

        df_ohlcv = load_ticker_ohlcv(ticker)
        if df_ohlcv is None:
            continue
        ohlcv = precompute_ohlcv(df_ohlcv, TEST_START, TEST_END)
        del df_ohlcv
        if ohlcv is None:
            continue

        entry_list = []
        for i in range(len(ohlcv['dates'])):
            if ohlcv['times'][i] == ENTRY_MINUTE:
                day_score = score_by_date.get(int(ohlcv['dates'][i]), 0.0)
                if day_score >= prob_thresh:
                    entry_list.append(i)

        if not entry_list:
            continue
        indices = np.array(entry_list, dtype=np.int64)
        pnl_arr, entry_dates = run_short_trade_sim(
            indices, ohlcv['prices'], ohlcv['highs'], ohlcv['lows'],
            ohlcv['dates'], ohlcv['times'].astype(np.int64),
            tp_pct, sl_pct, EXIT_TIME_MIN)
        if len(pnl_arr) > 0 and SLIPPAGE_PCT > 0:
            pnl_arr = pnl_arr - 2 * SLIPPAGE_PCT

        for j in range(len(pnl_arr)):
            d_int = int(entry_dates[j])
            dt = pd.Timestamp(year=d_int // 10000,
                              month=(d_int % 10000) // 100,
                              day=d_int % 100)
            all_daily.append({'date': dt, 'ticker': ticker, 'pnl': float(pnl_arr[j])})

    if all_daily:
        daily_df = pd.DataFrame(all_daily)
        daily_agg = daily_df.groupby('date')['pnl'].mean()
        daily_pnl_by_model[mk] = daily_agg.sort_index()
    else:
        daily_pnl_by_model[mk] = pd.Series(dtype=float)

    gc.collect()

for mk in MODEL_KEYS:
    series = daily_pnl_by_model[mk]
    if len(series) == 0:
        print(f"\n  {MODELS[mk]['label']}: No deployable trades for daily PnL")
        continue
    cum = series.cumsum()
    n_days = len(series)
    n_trades_total = len(series)
    cum_max = cum.cummax()
    drawdown = cum - cum_max

    print(f"\n  {MODELS[mk]['label']} — Daily PnL over TEST window:")
    print(f"    Trading days:    {n_days}")
    print(f"    Total cumul PnL: {cum.iloc[-1]:+.2%}")
    print(f"    Avg daily PnL:   {series.mean():+.2%}")
    print(f"    Best day:        {series.max():+.2%}")
    print(f"    Worst day:       {series.min():+.2%}")
    print(f"    Max drawdown:    {drawdown.min():+.2%}")
    print(f"    Win rate:        {(series > 0).mean():.0%}")

print(f"\n  Weekly breakdown:")
for mk in MODEL_KEYS:
    series = daily_pnl_by_model[mk]
    if len(series) == 0:
        continue
    weekly = series.groupby(pd.Grouper(freq='W')).agg(['sum', 'count', 'mean'])
    print(f"\n  {MODELS[mk]['label']}:")
    print(f"  {'Week':>12} {'PnL':>8} {'Trades':>7} {'Avg':>8}")
    print(f"  {'─' * 40}")
    for wk, row in weekly.iterrows():
        if row['count'] > 0:
            print(f"  {wk.strftime('%Y-%m-%d'):>12} {row['sum']:>+7.2%} {int(row['count']):>7} {row['mean']:>+7.2%}")

fig2, axes2 = plt.subplots(1, 2, figsize=(18, 6))
fig2.suptitle(f'SHORT L5-2: PnL Performance Over TEST Window (Jul 16 - Aug 21)',
              fontsize=13, fontweight='bold')

ax = axes2[0]
for mk in MODEL_KEYS:
    series = daily_pnl_by_model[mk]
    if len(series) == 0:
        continue
    cum = series.cumsum() * 100
    ax.plot(cum.index, cum.values, label=MODELS[mk]['label'],
            color=colors[mk], linewidth=2)
ax.axhline(0, color='black', linewidth=0.8, alpha=0.5)
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative PnL (%)')
ax.set_title('Cumulative PnL (Best Config Per Ticker)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.tick_params(axis='x', rotation=45)

ax = axes2[1]
for mk in MODEL_KEYS:
    series = daily_pnl_by_model[mk]
    if len(series) == 0:
        continue
    cum = series.cumsum()
    dd = (cum - cum.cummax()) * 100
    ax.fill_between(dd.index, dd.values, 0, alpha=0.3, color=colors[mk],
                    label=MODELS[mk]['label'])
    ax.plot(dd.index, dd.values, color=colors[mk], linewidth=1)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_xlabel('Date')
ax.set_ylabel('Drawdown (%)')
ax.set_title('Drawdown from Peak')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
pnl_chart_path = os.path.join(OUTPUT_DIR, '5way_pnl_over_test.png')
plt.savefig(pnl_chart_path, dpi=150, bbox_inches='tight')
print(f"\n  PnL chart saved: {pnl_chart_path}")
if _IN_COLAB:
    plt.show()
plt.close()


# ==============================================================================
# VERDICT
# ==============================================================================
total_elapsed = time.time() - t_total
print(f"\n\n{'=' * W}")
print(f"  VERDICT — {len(MODEL_KEYS)}-WAY MODEL COMPARISON")
print(f"{'=' * W}")

b_auc_30 = ml_results['BASELINE']['30m']['auc']
print(f"\n  Baseline 30m AUC: {b_auc_30:.4f}\n")

for ek in exp_keys:
    print(f"  --- {MODELS[ek]['label']} ---")
    for h_idx in range(N_HORIZONS):
        h_label = HORIZON_LABELS[h_idx]
        b_auc = ml_results['BASELINE'][h_label]['auc']
        e_auc = ml_results[ek][h_label]['auc']
        delta = e_auc - b_auc
        tag = "BETTER" if delta > 0.005 else "WORSE" if delta < -0.005 else "SAME"
        print(f"    {h_label}: AUC {b_auc:.4f} -> {e_auc:.4f} ({delta:+.4f}) [{tag}]")

    if ticker_auc:
        dcol = f'delta_{ek}'
        valid = auc_df[dcol].dropna()
        if len(valid) > 0:
            n_imp = (valid > 0).sum()
            print(f"    Per-ticker AUC: {n_imp}/{len(valid)} improved, avg lift {valid.mean():+.4f}")

    if ek in pnl_summary:
        pdf = pnl_summary[ek]
        n_b = (pdf['delta'] > 0).sum()
        avg_d = pdf['delta'].mean()
        print(f"    Trade PnL: {n_b}/{len(pdf)} improved, avg lift {avg_d:+.2%}")

    e_auc_30 = ml_results[ek]['30m']['auc']
    auc_delta = e_auc_30 - b_auc_30
    if auc_delta > 0.01:
        verdict = "PROMISING — run optimizer next"
    elif auc_delta > 0.005:
        verdict = "MARGINAL — may not survive walkforward"
    elif auc_delta > -0.005:
        if ek in pnl_summary and pnl_summary[ek]['delta'].mean() > 0.002:
            verdict = "AUC FLAT, PnL POSITIVE — worth testing optimizer"
        else:
            verdict = "NO LIFT — abandon"
    else:
        verdict = "DEGRADED — abandon"
    print(f"    30m AUC delta: {auc_delta:+.4f} → {verdict}")
    print()

print(f"  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
print(f"{'=' * W}")
