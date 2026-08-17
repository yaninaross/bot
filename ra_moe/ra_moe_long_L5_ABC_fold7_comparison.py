# %% ── Spike Transformer LONG: A/B/C Experiment Comparison (Fold 7) ────────────
#
# Compares 4 experiments head-to-head on fold 7:
#   L5-1  Baseline (sector-aware FiLM, shared head, gamma_pos=0.1)
#   L5-3  Exp A: training hygiene (grad clip, warmup+cosine, AdamW)
#   L5-4  Exp B: A + architecture (FiLM-backbone, per-horizon heads, gamma_pos=2.0)
#   L5-5  Exp C: A + B + features (37 asset features vs 33)
#
# ML metrics:  AUC-ROC, PR-AUC, Brier, calibration, precision/recall at thresholds
# Trading metrics:  from BQ wf_optimization_results (opt/val/test PnL, WR, freq)
#
# All metrics computed on TEST split at SCORE_MINUTE=375.
# GPU recommended for inference. Colab-ready.

import os, gc, ctypes, warnings
import numpy as np
import h5py
import tensorflow as tf
from tensorflow.keras import layers, Model
from google.cloud import storage, bigquery as bq
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, precision_recall_curve,
                             roc_curve, log_loss)

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib
try:
    from google.colab import auth
    _IN_COLAB = True
    auth.authenticate_user()
except (ImportError, Exception):
    _IN_COLAB = False
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ==============================================================================
# CONFIG
# ==============================================================================
FOLD_ID = 7
SCORE_MINUTE = 375
TIME_STEPS = 128
TARGET_HORIZONS = [15, 30, 60]
N_HORIZONS = 3
HORIZON_LABELS = {0: '15m', 1: '30m', 2: '60m'}

D_MODEL = 64; N_HEADS = 4; D_FF = 128; N_TRANSFORMER_BLOCKS = 2
DROPOUT = 0.1; FILM_HIDDEN = 64

GCS_BUCKET = 'time-series-bucket-ys'
BQ_TABLE_ID = 'ys-time-series.scoring_logs.wf_optimization_results'

EXPERIMENTS = {
    'L5-1': {
        'label': 'L5-1 (Baseline)',
        'color': '#1f77b4',
        'gcs_data': 'Algos/RA_MoE/long_L5_1_data',
        'gcs_model': 'Algos/RA_MoE/long_L5_1_models',
        'data_dir': '/tmp/ra_moe_long_L5_1_data',
        'model_dir': '/tmp/ra_moe_long_L5_1_models',
        'model_version': f'spike_transformer_long_L5_1_fold{FOLD_ID}',
        'arch': 'v1',   # shared head, no FiLM-backbone
        'gamma_pos': 0.1,
    },
    'L5-3': {
        'label': 'L5-3 (Exp A: training)',
        'color': '#ff7f0e',
        'gcs_data': 'Algos/RA_MoE/long_L5_1_data',
        'gcs_model': 'Algos/RA_MoE/long_L5_3_models',
        'data_dir': '/tmp/ra_moe_long_L5_1_data',
        'model_dir': '/tmp/ra_moe_long_L5_3_models',
        'model_version': f'spike_transformer_long_L5_3_fold{FOLD_ID}',
        'arch': 'v1',
        'gamma_pos': 0.1,
    },
    'L5-4': {
        'label': 'L5-4 (Exp B: arch+loss)',
        'color': '#2ca02c',
        'gcs_data': 'Algos/RA_MoE/long_L5_1_data',
        'gcs_model': 'Algos/RA_MoE/long_L5_4_models',
        'data_dir': '/tmp/ra_moe_long_L5_1_data',
        'model_dir': '/tmp/ra_moe_long_L5_4_models',
        'model_version': f'spike_transformer_long_L5_4_fold{FOLD_ID}',
        'arch': 'v2',   # FiLM-backbone, per-horizon heads
        'gamma_pos': 2.0,
    },
    'L5-5': {
        'label': 'L5-5 (Exp C: +features)',
        'color': '#d62728',
        'gcs_data': 'Algos/RA_MoE/long_L5_5_data',
        'gcs_model': 'Algos/RA_MoE/long_L5_5_models',
        'data_dir': '/tmp/ra_moe_long_L5_5_data',
        'model_dir': '/tmp/ra_moe_long_L5_5_models',
        'model_version': f'spike_transformer_long_L5_5_fold{FOLD_ID}',
        'arch': 'v2',
        'gamma_pos': 2.0,
    },
}

OUTPUT_DIR = '/tmp/ra_moe_L5_comparison'
os.makedirs(OUTPUT_DIR, exist_ok=True)

PROB_THRESHOLDS_EVAL = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

# ── Backtest config ──
FOLD_CONFIGS = {
    5: {'test_start': '2026-07-10', 'test_end': '2026-08-12'},
    6: {'test_start': '2026-07-22', 'test_end': '2026-08-12'},
    7: {'test_start': '2026-07-29', 'test_end': '2026-08-12'},
}
TEST_START = FOLD_CONFIGS[FOLD_ID]['test_start']
TEST_END   = FOLD_CONFIGS[FOLD_ID]['test_end']

ENTRY_MINUTE   = 385       # 6:25 AM PST
EXIT_TIME_MIN  = 780       # 1:00 PM PST
SLIPPAGE_PCT   = 0.0015
GCS_INFER_DIR  = 'Data/Inference_v2'

TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}
EXCLUDE_TICKERS = {'SPY', 'VIX', 'TLT', 'IWM', 'XLE', 'ITA', 'BITF', 'MIGI', 'SMLR'}

DEPLOY_PNL_THRESHOLD = 0.018

SEED = 42
np.random.seed(SEED)
W = 100


# ==============================================================================
# GCS HELPERS
# ==============================================================================
def _gcs_bucket():
    return storage.Client().bucket(GCS_BUCKET)

def ensure_data(filename, data_dir, gcs_prefix):
    local_path = os.path.join(data_dir, filename)
    if os.path.exists(local_path):
        return local_path
    os.makedirs(data_dir, exist_ok=True)
    gcs_path = f'{gcs_prefix}/{filename}'
    print(f"      Downloading {filename}...")
    _gcs_bucket().blob(gcs_path).download_to_filename(local_path)
    return local_path

def ensure_model(fold_id, model_dir, gcs_prefix):
    local_dir = os.path.join(model_dir, f'fold_{fold_id}')
    weights_path = os.path.join(local_dir, 'model.weights.h5')
    if os.path.exists(weights_path):
        return weights_path
    os.makedirs(local_dir, exist_ok=True)
    gcs_pfx = f'{gcs_prefix}/fold_{fold_id}'
    print(f"      Downloading fold {fold_id} model...")
    bucket = _gcs_bucket()
    for blob in bucket.list_blobs(prefix=gcs_pfx):
        fname = blob.name.split('/')[-1]
        blob.download_to_filename(os.path.join(local_dir, fname))
    return weights_path


# ==============================================================================
# ARCHITECTURE v1 — L5-1/L5-3 (shared head, no FiLM-in-backbone)
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
        out = out * sigma[:, tf.newaxis, :]
        return self.wo(out)


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
        if len(x.shape) == 3:
            gamma = gamma[:, tf.newaxis, :]
            beta  = beta[:, tf.newaxis, :]
        return gamma * x + beta


# ── v1 blocks (no FiLM-in-backbone) ──

class TransformerBlockV1(layers.Layer):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.attn  = DeStationaryAttention(d_model, n_heads, dropout)
        self.ffn   = tf.keras.Sequential([
            layers.Dense(d_ff, activation='gelu'),
            layers.Dense(d_model), layers.Dropout(dropout)])
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.drop  = layers.Dropout(dropout)

    def call(self, x, stats, training=False):
        x = x + self.drop(
            self.attn(self.norm1(x), stats, training=training), training=training)
        x = x + self.ffn(self.norm2(x), training=training)
        return x


class TransformerBackboneV1(layers.Layer):
    def __init__(self, d_model, n_heads, d_ff, n_blocks, seq_len,
                 dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.stationarize = SeriesStationarization()
        self.input_proj = layers.Dense(d_model)
        self.pos_embed = self.add_weight(
            name='pos_embed', shape=(1, seq_len, d_model),
            initializer='glorot_normal', trainable=True)
        self.input_drop = layers.Dropout(dropout)
        self.blocks = [TransformerBlockV1(d_model, n_heads, d_ff, dropout,
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


class SpikeTransformerV1(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.backbone = TransformerBackboneV1(
            D_MODEL, N_HEADS, D_FF, N_TRANSFORMER_BLOCKS,
            TIME_STEPS, DROPOUT, name='backbone')
        self.attn_pool = AttentionPooling(D_MODEL, name='attn_pool')
        self.film = FiLMConditioner(D_MODEL, FILM_HIDDEN, name='film')
        self.drop = layers.Dropout(DROPOUT)
        self.head = layers.Dense(N_HORIZONS, activation='sigmoid', name='spike_head')

    def call(self, inputs, training=False):
        seq = inputs['seq_input']
        regime = inputs['regime_input']
        h = self.backbone(seq, training=training)
        h = self.attn_pool(h, training=training)
        h = self.film(h, regime, training=training)
        h = self.drop(h, training=training)
        return self.head(h)


# ── v2 blocks (FiLM-in-backbone + per-horizon heads) ──

class TransformerBlockV2(layers.Layer):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.attn  = DeStationaryAttention(d_model, n_heads, dropout)
        self.ffn   = tf.keras.Sequential([
            layers.Dense(d_ff, activation='gelu'),
            layers.Dense(d_model), layers.Dropout(dropout)])
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.drop  = layers.Dropout(dropout)
        self.block_film = FiLMConditioner(d_model, FILM_HIDDEN)

    def call(self, x, stats, regime_features=None, training=False):
        x = x + self.drop(
            self.attn(self.norm1(x), stats, training=training), training=training)
        x = x + self.ffn(self.norm2(x), training=training)
        if regime_features is not None:
            x = self.block_film(x, regime_features, training=training)
        return x


class TransformerBackboneV2(layers.Layer):
    def __init__(self, d_model, n_heads, d_ff, n_blocks, seq_len,
                 dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.stationarize = SeriesStationarization()
        self.input_proj = layers.Dense(d_model)
        self.pos_embed = self.add_weight(
            name='pos_embed', shape=(1, seq_len, d_model),
            initializer='glorot_normal', trainable=True)
        self.input_drop = layers.Dropout(dropout)
        self.blocks = [TransformerBlockV2(d_model, n_heads, d_ff, dropout,
                                          name=f'block_{i}')
                       for i in range(n_blocks)]
        self.final_norm = layers.LayerNormalization()

    def call(self, x, regime_features=None, training=False):
        x_norm, stats = self.stationarize(x)
        h = self.input_proj(x_norm) + self.pos_embed
        h = self.input_drop(h, training=training)
        for block in self.blocks:
            h = block(h, stats, regime_features=regime_features, training=training)
        return self.final_norm(h)


class SpikeTransformerV2(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.backbone = TransformerBackboneV2(
            D_MODEL, N_HEADS, D_FF, N_TRANSFORMER_BLOCKS,
            TIME_STEPS, DROPOUT, name='backbone')
        self.attn_pool = AttentionPooling(D_MODEL, name='attn_pool')
        self.film = FiLMConditioner(D_MODEL, FILM_HIDDEN, name='film')
        self.drop = layers.Dropout(DROPOUT)
        self.heads = [layers.Dense(1, activation='sigmoid', name=f'head_{h}')
                      for h in TARGET_HORIZONS]

    def call(self, inputs, training=False):
        seq = inputs['seq_input']
        regime = inputs['regime_input']
        h = self.backbone(seq, regime_features=regime, training=training)
        h = self.attn_pool(h, training=training)
        h = self.film(h, regime, training=training)
        h = self.drop(h, training=training)
        return tf.concat([head(h) for head in self.heads], axis=-1)


# ==============================================================================
# MODEL BUILDER
# ==============================================================================

def build_model(arch, n_asset_features, n_regime_features):
    if arch == 'v1':
        model = SpikeTransformerV1(name='spike_transformer')
    else:
        model = SpikeTransformerV2(name='spike_transformer')
    dummy_seq = tf.zeros((1, TIME_STEPS, n_asset_features))
    dummy_reg = tf.zeros((1, n_regime_features))
    model({'seq_input': dummy_seq, 'regime_input': dummy_reg})
    return model


# ==============================================================================
# KERAS 3 WEIGHT LOADER (h5py)
# ==============================================================================

def _load_keras3_weights(model, filepath):
    with h5py.File(filepath, 'r') as f:
        _load_layer_recursive(model, f)

def _load_layer_recursive(layer, h5_group):
    if 'vars' in h5_group:
        vg = h5_group['vars']
        own_vars = layer.trainable_weights + layer.non_trainable_weights
        for i, var in enumerate(own_vars):
            key = str(i)
            if key in vg:
                var.assign(np.array(vg[key]))
    for key in sorted(h5_group.keys()):
        if key in ('vars', 'optimizer'):
            continue
        if not isinstance(h5_group[key], h5py.Group):
            continue
        child = getattr(layer, key, None)
        if child is None:
            continue
        if isinstance(child, layers.Layer):
            _load_layer_recursive(child, h5_group[key])
        elif isinstance(child, (list, tuple)):
            _load_list_recursive(child, h5_group[key])

def _load_list_recursive(items, h5_group):
    used_names = {}
    for item in items:
        if not isinstance(item, layers.Layer):
            continue
        cname = type(item).__name__
        snake = ''.join(f'_{c.lower()}' if c.isupper() else c
                        for c in cname).lstrip('_')
        if snake in used_names:
            used_names[snake] += 1
            snake = f"{snake}_{used_names[snake]}"
        else:
            used_names[snake] = 0
        if snake in h5_group:
            _load_layer_recursive(item, h5_group[snake])


# ==============================================================================
# OHLCV + TRADE SIMULATION
# ==============================================================================

import pytz
from datetime import timedelta
import pandas as pd

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
    raise FileNotFoundError(f"No OHLCV for {ticker}")

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


def simulate_long_trade(ohlcv, idx, cfg):
    prices = ohlcv['prices']; highs = ohlcv['highs']; lows = ohlcv['lows']
    dates = ohlcv['dates']; times = ohlcv['times']
    n = len(prices)
    entry_date = dates[idx]
    mode = cfg.get('mode', 'static')

    if mode == 'limit_entry':
        signal_price = prices[idx]
        limit_price = signal_price * (1.0 - cfg.get('dip_pct', 0))
        max_wait = cfg.get('max_wait', 30)
        fill_idx = -1
        for k in range(idx + 1, n):
            if dates[k] != entry_date:
                break
            if times[k] - times[idx] > max_wait:
                break
            if lows[k] <= limit_price:
                fill_idx = k
                break
        if fill_idx < 0:
            return None
        entry_price = limit_price
        start_idx = fill_idx
    else:
        entry_price = prices[idx]
        start_idx = idx

    tp_price = entry_price * (1 + cfg['tp_pct'])
    sl_price = entry_price * (1 - cfg['sl_pct'])
    grace_minutes = cfg.get('grace_minutes', 0) if mode == 'grace' else 0
    entry_time = times[start_idx]

    for k in range(start_idx + 1, n):
        if dates[k] != entry_date:
            exit_price = prices[k - 1]
            pnl = (exit_price - entry_price) / entry_price
            return {'pnl': pnl, 'entry': entry_price, 'exit': exit_price, 'outcome': 'EOD'}
        elapsed = times[k] - entry_time
        sl_active = elapsed >= grace_minutes
        tp_hit = highs[k] >= tp_price
        sl_hit = sl_active and lows[k] <= sl_price
        if tp_hit and sl_hit:
            if np.random.random() < 0.5:
                return {'pnl': cfg['tp_pct'], 'entry': entry_price, 'exit': tp_price, 'outcome': 'TP'}
            else:
                return {'pnl': -cfg['sl_pct'], 'entry': entry_price, 'exit': sl_price, 'outcome': 'SL'}
        if tp_hit:
            return {'pnl': cfg['tp_pct'], 'entry': entry_price, 'exit': tp_price, 'outcome': 'TP'}
        if sl_hit:
            return {'pnl': -cfg['sl_pct'], 'entry': entry_price, 'exit': sl_price, 'outcome': 'SL'}
        if times[k] >= EXIT_TIME_MIN:
            exit_price = prices[k]
            pnl = (exit_price - entry_price) / entry_price
            return {'pnl': pnl, 'entry': entry_price, 'exit': exit_price, 'outcome': 'EOD'}

    exit_price = prices[-1]
    pnl = (exit_price - entry_price) / entry_price
    return {'pnl': pnl, 'entry': entry_price, 'exit': exit_price, 'outcome': 'EOD'}


def build_deploy_configs(bq_df, model_version):
    edf = bq_df[(bq_df['model_version'] == model_version) &
                (bq_df['survived_b'] == True) &
                (bq_df['val_b_avg_pnl'] >= DEPLOY_PNL_THRESHOLD)]
    if len(edf) == 0:
        return {}
    best = edf.loc[edf.groupby('ticker')['val_b_avg_pnl'].idxmax()]
    configs = {}
    for _, row in best.iterrows():
        ticker = row['ticker']
        if ticker in EXCLUDE_TICKERS:
            continue
        mode = row['tpsl_mode']
        exp = str(row.get('experiment', ''))
        h_idx = 1
        for h_key, h_label in HORIZON_LABELS.items():
            if h_label in exp:
                h_idx = h_key
                break
        cfg = {
            'horizon': h_idx,
            'prob_thresh': float(row['prob_thresh']),
            'tp_pct': float(row['tp_pct']),
            'sl_pct': float(row['sl_pct']),
            'mode': mode if pd.notna(mode) and mode else 'static',
            'val_b_avg_pnl': float(row['val_b_avg_pnl']),
            'val_b_win_rate': float(row['val_b_win_rate']) if pd.notna(row.get('val_b_win_rate')) else 0,
            'val_b_count': int(row['val_b_count']) if pd.notna(row.get('val_b_count')) else 0,
        }
        if cfg['mode'] == 'grace' and pd.notna(row.get('grace_minutes')):
            cfg['grace_minutes'] = int(row['grace_minutes'])
        if cfg['mode'] == 'limit_entry':
            if pd.notna(row.get('dip_pct')):
                cfg['dip_pct'] = float(row['dip_pct'])
            if pd.notna(row.get('max_wait_minutes')):
                cfg['max_wait'] = int(row['max_wait_minutes'])
        configs[ticker] = cfg
    return configs


def run_backtest(all_scores, deploy_configs, ohlcv_cache, label):
    tickers = sorted(deploy_configs.keys())
    if not tickers:
        print(f"\n  {label}: no tickers to backtest")
        return []

    print(f"\n{'─'*W}")
    print(f"  BACKTEST -- {label} ({len(tickers)} tickers)")
    print(f"  TEST WINDOW: {TEST_START} -> {TEST_END}")
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
        h_scores = scores.get(h_idx, {}) if isinstance(next(iter(scores.values()), None), dict) else scores
        if not h_scores:
            print(f"    No scores")
            continue

        if ticker not in ohlcv_cache:
            print(f"    No OHLCV")
            continue
        ohlcv = precompute_ohlcv_arrays(ohlcv_cache[ticker], TEST_START, TEST_END)
        if ohlcv is None:
            print(f"    No OHLCV in test window")
            continue

        score_by_date = {}
        for d, s in h_scores.items():
            score_by_date[int(d)] = float(s) if not isinstance(s, dict) else float(s.get(h_idx, 0))

        n_trades = 0
        traded_dates = set()
        for i in range(len(ohlcv['dates'])):
            if ohlcv['times'][i] != ENTRY_MINUTE:
                continue
            d = int(ohlcv['dates'][i])
            if d in traded_dates:
                continue
            day_score = score_by_date.get(d, 0.0)
            if day_score < cfg['prob_thresh']:
                continue

            result = simulate_long_trade(ohlcv, i, cfg)
            if result is None:
                traded_dates.add(d)
                continue

            result['pnl'] -= 2 * SLIPPAGE_PCT
            d_str = f"{d // 10000}-{(d % 10000) // 100:02d}-{d % 100:02d}"
            all_trades.append({
                'Ticker': ticker, 'Date': d_str, 'Date_int': d,
                'Entry': result['entry'], 'Exit': result['exit'],
                'PnL': result['pnl'], 'Outcome': result['outcome'],
                'Mode': mode, 'Horizon': h_label, 'Strategy': label,
                'Prob': day_score,
            })
            traded_dates.add(d)
            n_trades += 1

        if n_trades > 0:
            t_pnls = [t['PnL'] for t in all_trades if t['Ticker'] == ticker]
            avg = np.mean(t_pnls)
            wr = np.mean(np.array(t_pnls) > 0)
            print(f"    {n_trades} trades | Avg: {avg:+.2%} | WR: {wr:.0%} | "
                  f"Total: {sum(t_pnls):+.2%}")
        else:
            print(f"    No trades")

    return all_trades

def extract_daily_scores(all_results, exp_key, ticker_names):
    scores = {}
    for split in ['val', 'test']:
        r = all_results[exp_key][split]
        preds = r['preds']
        x_id = r['X_id']
        d_ints = r['date_ints']
        for i in range(len(x_id)):
            tid = int(x_id[i])
            if tid >= len(ticker_names):
                continue
            tname = str(ticker_names[tid])
            if tname in EXCLUDE_TICKERS:
                continue
            if tname not in scores:
                scores[tname] = {}
            d = int(d_ints[i])
            scores[tname][d] = {h: float(preds[i, h]) for h in range(N_HORIZONS)}
    return scores


# ==============================================================================
# ML METRICS
# ==============================================================================

def compute_ml_metrics(y_true, y_pred, horizon_idx):
    """Compute ML metrics for a single horizon."""
    y = y_true[:, horizon_idx]
    p = y_pred[:, horizon_idx]

    n_pos = y.sum()
    n_neg = len(y) - n_pos
    spike_rate = float(n_pos / len(y))

    metrics = {
        'n_samples': len(y),
        'n_pos': int(n_pos),
        'spike_rate': spike_rate,
    }

    if n_pos == 0 or n_neg == 0:
        metrics.update({
            'auc_roc': float('nan'), 'pr_auc': float('nan'),
            'brier': float('nan'), 'log_loss': float('nan'),
            'avg_pred': float(p.mean()), 'avg_pred_pos': float('nan'),
            'avg_pred_neg': float('nan'),
        })
        return metrics

    metrics['auc_roc'] = float(roc_auc_score(y, p))
    metrics['pr_auc'] = float(average_precision_score(y, p))
    metrics['brier'] = float(brier_score_loss(y, p))
    metrics['log_loss'] = float(log_loss(y, np.clip(p, 1e-7, 1-1e-7)))
    metrics['avg_pred'] = float(p.mean())
    metrics['avg_pred_pos'] = float(p[y == 1].mean())
    metrics['avg_pred_neg'] = float(p[y == 0].mean())
    metrics['pred_separation'] = metrics['avg_pred_pos'] - metrics['avg_pred_neg']

    for thresh in PROB_THRESHOLDS_EVAL:
        pred_pos = p >= thresh
        tp = ((pred_pos) & (y == 1)).sum()
        fp = ((pred_pos) & (y == 0)).sum()
        fn = ((~pred_pos) & (y == 1)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        n_signals = pred_pos.sum()
        metrics[f'prec_{thresh:.0%}'] = float(prec)
        metrics[f'recall_{thresh:.0%}'] = float(rec)
        metrics[f'n_signals_{thresh:.0%}'] = int(n_signals)

    return metrics


def compute_calibration(y_true, y_pred, horizon_idx, n_bins=10):
    """Compute calibration curve for a single horizon."""
    y = y_true[:, horizon_idx]
    p = y_pred[:, horizon_idx]
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_actuals = []
    bin_counts = []
    for i in range(n_bins):
        mask = (p >= bins[i]) & (p < bins[i+1])
        if mask.sum() > 0:
            bin_centers.append(float(p[mask].mean()))
            bin_actuals.append(float(y[mask].mean()))
            bin_counts.append(int(mask.sum()))
    return bin_centers, bin_actuals, bin_counts


# ==============================================================================
# SCORING
# ==============================================================================

def score_experiment(exp_key, exp_cfg):
    """Load model, score test NPZ, return predictions and labels."""
    print(f"\n  [{exp_key}] {exp_cfg['label']}")

    data_dir = exp_cfg['data_dir']
    gcs_data = exp_cfg['gcs_data']
    gcs_model = exp_cfg['gcs_model']
    model_dir = exp_cfg['model_dir']
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    probe_path = ensure_data(f'fold_{FOLD_ID}_val.npz', data_dir, gcs_data)
    probe = np.load(probe_path)
    n_asset = probe['X_seq'].shape[-1]
    n_regime = probe['X_regime'].shape[-1]
    probe.close()
    print(f"    Features: {n_asset} asset, {n_regime} regime")

    model = build_model(exp_cfg['arch'], n_asset, n_regime)
    weights_path = ensure_model(FOLD_ID, model_dir, gcs_model)
    _load_keras3_weights(model, weights_path)
    print(f"    Model: {model.count_params():,} params")

    results = {}
    for split in ['val', 'test']:
        path = ensure_data(f'fold_{FOLD_ID}_{split}.npz', data_dir, gcs_data)
        data = np.load(path)
        X_seq = data['X_seq']
        X_reg = data['X_regime']
        Y = data['Y']
        X_id = data['X_id'].flatten()
        minutes = data['minutes']
        date_ints = data['date_ints']
        data.close()

        mask = (minutes == SCORE_MINUTE)
        X_seq = X_seq[mask]
        X_reg = X_reg[mask]
        Y = Y[mask]
        X_id = X_id[mask]
        date_ints = date_ints[mask]

        preds = model.predict(
            {'seq_input': X_seq.astype(np.float32),
             'regime_input': X_reg.astype(np.float32)},
            batch_size=512, verbose=0)

        results[split] = {
            'Y': Y, 'preds': preds, 'X_id': X_id, 'date_ints': date_ints,
            'n_total': len(Y),
        }
        n_pos = Y.sum(axis=0)
        print(f"    {split}: {len(Y):,} samples @ min {SCORE_MINUTE}, "
              f"spikes: {'/'.join(str(int(x)) for x in n_pos)}")
        del X_seq, X_reg

    del model
    tf.keras.backend.clear_session()
    gc.collect()
    return results


# ==============================================================================
# BQ TRADING METRICS
# ==============================================================================

def fetch_bq_trading_metrics():
    """Query BQ for WF optimization results for all 4 experiments."""
    client = bq.Client()
    versions = [cfg['model_version'] for cfg in EXPERIMENTS.values()]
    version_str = ', '.join(f"'{v}'" for v in versions)

    query = f"""
    SELECT
        model_version,
        ticker,
        tpsl_mode,
        experiment,
        prob_thresh,
        tp_pct,
        sl_pct,
        opt_avg_pnl,
        opt_win_rate,
        opt_count,
        val_a_avg_pnl,
        val_a_win_rate,
        val_a_count,
        val_b_avg_pnl,
        val_b_win_rate,
        val_b_count,
        survived_a,
        survived_b,
        grace_minutes,
        dip_pct,
        max_wait_minutes
    FROM `{BQ_TABLE_ID}`
    WHERE model_version IN ({version_str})
    ORDER BY model_version, ticker, tpsl_mode
    """
    print(f"\n  Querying BQ for {len(versions)} model versions...")
    df = client.query(query).to_dataframe()
    print(f"  Got {len(df)} rows")
    return df


# ==============================================================================
# MAIN
# ==============================================================================

print(f"{'='*W}")
print(f"  SPIKE TRANSFORMER LONG — A/B/C EXPERIMENT COMPARISON (FOLD {FOLD_ID})")
print(f"{'='*W}")
print(f"  Experiments:")
for key, cfg in EXPERIMENTS.items():
    print(f"    {key}: {cfg['label']} (arch={cfg['arch']}, gamma_pos={cfg['gamma_pos']})")
print(f"  Score minute: {SCORE_MINUTE} | Horizons: {TARGET_HORIZONS}")
print(f"{'='*W}")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"  GPU: {gpus[0].name}")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass
else:
    print(f"  GPU: NONE")


# ── Phase 1: Score all models ─────────────────────────────────────────────────
print(f"\n{'='*W}")
print(f"  PHASE 1: ML SCORING")
print(f"{'='*W}")

all_results = {}
for exp_key, exp_cfg in EXPERIMENTS.items():
    all_results[exp_key] = score_experiment(exp_key, exp_cfg)
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


# ── Phase 2: Compute ML metrics ──────────────────────────────────────────────
print(f"\n{'='*W}")
print(f"  PHASE 2: ML METRICS (TEST split @ minute {SCORE_MINUTE})")
print(f"{'='*W}")

ml_metrics = {}
for exp_key in EXPERIMENTS:
    ml_metrics[exp_key] = {}
    for h_idx in range(N_HORIZONS):
        Y = all_results[exp_key]['test']['Y']
        preds = all_results[exp_key]['test']['preds']
        ml_metrics[exp_key][h_idx] = compute_ml_metrics(Y, preds, h_idx)

# ── ML metrics table ──
for h_idx in range(N_HORIZONS):
    h_label = HORIZON_LABELS[h_idx]
    print(f"\n  ── {h_label} Horizon ──")
    header = (f"  {'Experiment':<28} {'AUC-ROC':>8} {'PR-AUC':>8} {'Brier':>7} "
              f"{'LogLoss':>8} {'AvgP+':>7} {'AvgP-':>7} {'Sep':>7} "
              f"{'P@30%':>6} {'P@50%':>6} {'P@70%':>6}")
    print(header)
    print(f"  {'─' * (len(header) - 2)}")

    for exp_key, exp_cfg in EXPERIMENTS.items():
        m = ml_metrics[exp_key][h_idx]
        print(f"  {exp_cfg['label']:<28} "
              f"{m.get('auc_roc', float('nan')):>8.4f} "
              f"{m.get('pr_auc', float('nan')):>8.4f} "
              f"{m.get('brier', float('nan')):>7.4f} "
              f"{m.get('log_loss', float('nan')):>8.4f} "
              f"{m.get('avg_pred_pos', float('nan')):>7.4f} "
              f"{m.get('avg_pred_neg', float('nan')):>7.4f} "
              f"{m.get('pred_separation', float('nan')):>7.4f} "
              f"{m.get('prec_30%', 0):>6.1%} "
              f"{m.get('prec_50%', 0):>6.1%} "
              f"{m.get('prec_70%', 0):>6.1%}")

# ── Best experiment per metric ──
print(f"\n  ── Winners (higher is better, except Brier/LogLoss) ──")
for h_idx in range(N_HORIZONS):
    h_label = HORIZON_LABELS[h_idx]
    vals = {k: ml_metrics[k][h_idx] for k in EXPERIMENTS}
    best_auc = max(vals, key=lambda k: vals[k].get('auc_roc', -1))
    best_pr = max(vals, key=lambda k: vals[k].get('pr_auc', -1))
    best_brier = min(vals, key=lambda k: vals[k].get('brier', 999))
    best_sep = max(vals, key=lambda k: vals[k].get('pred_separation', -1))
    print(f"  {h_label}:  AUC={best_auc}({vals[best_auc]['auc_roc']:.4f})  "
          f"PR={best_pr}({vals[best_pr]['pr_auc']:.4f})  "
          f"Brier={best_brier}({vals[best_brier]['brier']:.4f})  "
          f"Sep={best_sep}({vals[best_sep]['pred_separation']:.4f})")


# ── Phase 3: BQ trading metrics ──────────────────────────────────────────────
print(f"\n{'='*W}")
print(f"  PHASE 3: TRADING METRICS (from BQ WF optimization)")
print(f"{'='*W}")

bq_df = fetch_bq_trading_metrics()

if len(bq_df) > 0:
    for mode in ['static', 'grace', 'limit_entry']:
        mode_df = bq_df[bq_df['tpsl_mode'] == mode]
        if len(mode_df) == 0:
            continue

        print(f"\n  ── {mode.upper()} mode ──")
        header = (f"  {'Experiment':<28} {'Tickers':>7} {'PassTest':>8} "
                  f"{'AvgOPT':>8} {'AvgTEST':>9} {'AvgWR':>6} "
                  f"{'Deploy':>6} {'AvgDplyPnL':>11}")
        print(header)
        print(f"  {'─' * (len(header) - 2)}")

        for exp_key, exp_cfg in EXPERIMENTS.items():
            mv = exp_cfg['model_version']
            edf = mode_df[mode_df['model_version'] == mv]
            if len(edf) == 0:
                print(f"  {exp_cfg['label']:<28} {'(no data)':>7}")
                continue

            n_tickers = len(edf)
            n_pass = edf['survived_b'].sum()
            avg_opt = edf['opt_avg_pnl'].mean()
            avg_test = edf['val_b_avg_pnl'].mean()
            avg_wr = edf['val_b_win_rate'].mean()

            deploy = edf[edf['survived_b'] & (edf['val_b_avg_pnl'] >= 0.018)]
            n_deploy = len(deploy)
            avg_deploy_pnl = deploy['val_b_avg_pnl'].mean() if n_deploy > 0 else 0

            print(f"  {exp_cfg['label']:<28} "
                  f"{n_tickers:>7} {n_pass:>8} "
                  f"{avg_opt:>+8.2%} {avg_test:>+9.2%} {avg_wr:>6.1%} "
                  f"{n_deploy:>6} {avg_deploy_pnl:>+11.2%}")

    # ── Per-ticker comparison (best mode per ticker) ──
    print(f"\n  ── Per-ticker TEST PnL comparison (best mode) ──")
    print(f"  {'Ticker':<7}", end='')
    for exp_key, exp_cfg in EXPERIMENTS.items():
        print(f" {exp_key:>10}", end='')
    print(f" {'Winner':>10}")
    print(f"  {'─' * (7 + 11 * (len(EXPERIMENTS) + 1))}")

    all_tickers = sorted(bq_df['ticker'].unique())
    for ticker in all_tickers:
        tdf = bq_df[bq_df['ticker'] == ticker]
        print(f"  {ticker:<7}", end='')
        best_pnl = -999
        best_exp = ''
        for exp_key, exp_cfg in EXPERIMENTS.items():
            mv = exp_cfg['model_version']
            edf = tdf[tdf['model_version'] == mv]
            if len(edf) == 0:
                print(f" {'---':>10}", end='')
            else:
                best_row = edf.loc[edf['val_b_avg_pnl'].idxmax()]
                pnl = best_row['val_b_avg_pnl']
                print(f" {pnl:>+10.2%}", end='')
                if pnl > best_pnl:
                    best_pnl = pnl
                    best_exp = exp_key
        print(f" {best_exp:>10}")

    # ── Aggregate comparison ──
    print(f"\n  ── Aggregate Trading Summary ──")
    header = (f"  {'Experiment':<28} {'Tickers':>7} {'PassRate':>8} "
              f"{'MeanTest':>9} {'MedTest':>8} {'Deploy':>6} "
              f"{'Overfit':>8}")
    print(header)
    print(f"  {'─' * (len(header) - 2)}")

    for exp_key, exp_cfg in EXPERIMENTS.items():
        mv = exp_cfg['model_version']
        edf = bq_df[bq_df['model_version'] == mv]
        if len(edf) == 0:
            print(f"  {exp_cfg['label']:<28} {'(no data)':>7}")
            continue
        best_per_ticker = edf.loc[edf.groupby('ticker')['val_b_avg_pnl'].idxmax()]
        n = len(best_per_ticker)
        n_pass = (best_per_ticker['val_b_avg_pnl'] > 0).sum()
        mean_test = best_per_ticker['val_b_avg_pnl'].mean()
        med_test = best_per_ticker['val_b_avg_pnl'].median()
        n_deploy = (best_per_ticker['val_b_avg_pnl'] >= 0.018).sum()
        avg_opt = best_per_ticker['opt_avg_pnl'].mean()
        overfit = 1.0 - (mean_test / avg_opt) if avg_opt > 0 else float('nan')
        print(f"  {exp_cfg['label']:<28} "
              f"{n:>7} {n_pass/n:>8.0%} "
              f"{mean_test:>+9.2%} {med_test:>+8.2%} {n_deploy:>6} "
              f"{overfit:>8.2f}")

else:
    print("  No BQ data found. Run the optimizer scripts first.")


# ── Phase 4: TRADING BACKTEST (per-ticker BQ configs) ────────────────────────
print(f"\n{'='*W}")
print(f"  PHASE 4: TRADING BACKTEST (TEST window: {TEST_START} -> {TEST_END})")
print(f"  Per-ticker BQ configs (survived_b, val_b >= {DEPLOY_PNL_THRESHOLD:.1%})")
print(f"  Entry: {ENTRY_MINUTE} (6:25 AM)  Exit: {EXIT_TIME_MIN} (1:00 PM)  Slippage: {SLIPPAGE_PCT:.2%} RT")
print(f"{'='*W}")

meta_path = ensure_data('meta.npz',
                        EXPERIMENTS['L5-1']['data_dir'],
                        EXPERIMENTS['L5-1']['gcs_data'])
meta = np.load(meta_path, allow_pickle=True)
ticker_names = meta['ticker_names']
print(f"  {len(ticker_names)} tickers in meta")

print("  Building deploy configs from BQ for each experiment...")
exp_deploy_configs = {}
for exp_key, exp_cfg in EXPERIMENTS.items():
    cfgs = build_deploy_configs(bq_df, exp_cfg['model_version'])
    exp_deploy_configs[exp_key] = cfgs
    tickers = sorted(cfgs.keys())
    print(f"    {exp_key}: {len(tickers)} deployable - {', '.join(tickers)}")

print("\n  Extracting per-ticker daily scores for all experiments...")
exp_daily_scores = {}
for exp_key in EXPERIMENTS:
    exp_daily_scores[exp_key] = extract_daily_scores(all_results, exp_key, ticker_names)
    n_tickers = len(exp_daily_scores[exp_key])
    print(f"    {exp_key}: {n_tickers} tickers with scores")

all_bt_tickers = set()
for exp_key in EXPERIMENTS:
    all_bt_tickers.update(exp_deploy_configs[exp_key].keys())
all_bt_tickers = sorted(all_bt_tickers)
print(f"  {len(all_bt_tickers)} unique tickers across all experiments")

print("  Loading OHLCV for backtest tickers...")
ohlcv_cache = {}
for ticker in all_bt_tickers:
    try:
        ohlcv_cache[ticker] = load_ticker_ohlcv(ticker)
    except Exception:
        pass
print(f"  Loaded OHLCV for {len(ohlcv_cache)} tickers")

bt_results = {}
for exp_key in EXPERIMENTS:
    scores_for_exp = {}
    raw = exp_daily_scores[exp_key]
    for ticker, day_scores in raw.items():
        scores_for_exp[ticker] = {}
        for d, horizon_dict in day_scores.items():
            for h_idx, prob in horizon_dict.items():
                if h_idx not in scores_for_exp[ticker]:
                    scores_for_exp[ticker][h_idx] = {}
                scores_for_exp[ticker][h_idx][d] = prob
    bt_results[exp_key] = run_backtest(
        scores_for_exp, exp_deploy_configs[exp_key], ohlcv_cache,
        EXPERIMENTS[exp_key]['label'])


# ── Portfolio comparison ──
print(f"\n\n{'='*W}")
print(f"  BACKTEST COMPARISON — Per-Ticker BQ Configs")
print(f"  TEST WINDOW: {TEST_START} -> {TEST_END}")
print(f"{'='*W}")

def _analyze(trades, label):
    if not trades:
        return {'label': label, 'n': 0, 'avg_pnl': 0, 'total_pnl': 0,
                'win_rate': 0, 'n_tickers': 0, 'sharpe': 0,
                'avg_win': 0, 'avg_loss': 0}
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
        'avg_win': df.loc[df['PnL'] > 0, 'PnL'].mean() if (df['PnL'] > 0).any() else 0,
        'avg_loss': df.loc[df['PnL'] <= 0, 'PnL'].mean() if (df['PnL'] <= 0).any() else 0,
        'sharpe': (avg / std * np.sqrt(252)) if std > 0 else 0,
    }

bt_stats = {}
for exp_key, exp_cfg in EXPERIMENTS.items():
    bt_stats[exp_key] = _analyze(bt_results[exp_key], exp_cfg['label'])

header = (f"  {'Experiment':<28} {'Tickers':>7} {'Trades':>6} {'AvgPnL':>8} "
          f"{'TotalPnL':>9} {'WR':>5} {'AvgWin':>7} {'AvgLoss':>8} {'Sharpe':>7}")
print(f"\n{header}")
print(f"  {'─' * (len(header) - 2)}")
for exp_key, exp_cfg in EXPERIMENTS.items():
    s = bt_stats[exp_key]
    if s['n'] == 0:
        print(f"  {exp_cfg['label']:<28} {'(no trades)':>7}")
        continue
    print(f"  {exp_cfg['label']:<28} {s['n_tickers']:>7} {s['n']:>6} "
          f"{s['avg_pnl']:>+8.2%} {s['total_pnl']*100:>+8.1f}% "
          f"{s['win_rate']:>4.0%} {s['avg_win']:>+6.2%} {s['avg_loss']:>+7.2%} "
          f"{s['sharpe']:>7.1f}")

# ── Per-ticker comparison ──
print(f"\n  ── Per-Ticker Backtest Breakdown ──")
ticker_pnl = {}
for exp_key in EXPERIMENTS:
    for t in bt_results.get(exp_key, []):
        ticker_pnl.setdefault(t['Ticker'], {})[exp_key] = \
            ticker_pnl.get(t['Ticker'], {}).get(exp_key, [])
        if not isinstance(ticker_pnl[t['Ticker']][exp_key], list):
            ticker_pnl[t['Ticker']][exp_key] = []
        ticker_pnl[t['Ticker']][exp_key].append(t['PnL'])

if ticker_pnl:
    print(f"  {'Ticker':<7}", end='')
    for exp_key in EXPERIMENTS:
        print(f" {exp_key:>10}", end='')
    print(f" {'Winner':>10}  {'N':>3}")
    print(f"  {'─' * (7 + 11 * (len(EXPERIMENTS) + 1) + 5)}")

    for ticker in sorted(ticker_pnl.keys()):
        print(f"  {ticker:<7}", end='')
        best_pnl = -999
        best_exp = ''
        max_n = 0
        for exp_key in EXPERIMENTS:
            pnls = ticker_pnl[ticker].get(exp_key, [])
            if not pnls:
                print(f" {'---':>10}", end='')
            else:
                avg = np.mean(pnls)
                print(f" {avg:>+10.2%}", end='')
                if avg > best_pnl:
                    best_pnl = avg
                    best_exp = exp_key
                max_n = max(max_n, len(pnls))
        print(f" {best_exp:>10}  {max_n:>3}")


# ==============================================================================
# CHARTS
# ==============================================================================
print(f"\n{'='*W}")
print(f"  CHARTS")
print(f"{'='*W}")

# ── Chart 1: ROC curves (3 horizons x 4 experiments) ──
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
fig.suptitle(f'ROC Curves — A/B/C Comparison (Fold {FOLD_ID}, TEST split)',
             fontsize=14, fontweight='bold')

for h_idx in range(N_HORIZONS):
    ax = axes[h_idx]
    for exp_key, exp_cfg in EXPERIMENTS.items():
        Y = all_results[exp_key]['test']['Y'][:, h_idx]
        P = all_results[exp_key]['test']['preds'][:, h_idx]
        if Y.sum() > 0 and (1-Y).sum() > 0:
            fpr, tpr, _ = roc_curve(Y, P)
            auc = ml_metrics[exp_key][h_idx]['auc_roc']
            ax.plot(fpr, tpr, color=exp_cfg['color'], linewidth=1.5,
                    label=f"{exp_key} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.5, alpha=0.5)
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title(f'{HORIZON_LABELS[h_idx]} Horizon')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

plt.tight_layout()
path = os.path.join(OUTPUT_DIR, 'roc_curves.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
print(f"  ROC curves -> {path}")
if _IN_COLAB: plt.show()
plt.close()

# ── Chart 2: Precision-Recall curves ──
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
fig.suptitle(f'Precision-Recall Curves — A/B/C Comparison (Fold {FOLD_ID}, TEST)',
             fontsize=14, fontweight='bold')

for h_idx in range(N_HORIZONS):
    ax = axes[h_idx]
    for exp_key, exp_cfg in EXPERIMENTS.items():
        Y = all_results[exp_key]['test']['Y'][:, h_idx]
        P = all_results[exp_key]['test']['preds'][:, h_idx]
        if Y.sum() > 0:
            prec, rec, _ = precision_recall_curve(Y, P)
            ap = ml_metrics[exp_key][h_idx]['pr_auc']
            ax.plot(rec, prec, color=exp_cfg['color'], linewidth=1.5,
                    label=f"{exp_key} (AP={ap:.3f})")
            spike_rate = Y.mean()
            ax.axhline(spike_rate, color='gray', linestyle=':', linewidth=0.5)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'{HORIZON_LABELS[h_idx]} Horizon')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

plt.tight_layout()
path = os.path.join(OUTPUT_DIR, 'pr_curves.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
print(f"  PR curves -> {path}")
if _IN_COLAB: plt.show()
plt.close()

# ── Chart 3: Calibration plots ──
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
fig.suptitle(f'Calibration Diagrams — A/B/C Comparison (Fold {FOLD_ID}, TEST)',
             fontsize=14, fontweight='bold')

for h_idx in range(N_HORIZONS):
    ax = axes[h_idx]
    for exp_key, exp_cfg in EXPERIMENTS.items():
        Y = all_results[exp_key]['test']['Y']
        P = all_results[exp_key]['test']['preds']
        centers, actuals, counts = compute_calibration(Y, P, h_idx, n_bins=10)
        if centers:
            ax.plot(centers, actuals, 'o-', color=exp_cfg['color'],
                    linewidth=1.5, markersize=4, label=exp_key)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.5, alpha=0.5)
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Observed Frequency')
    ax.set_title(f'{HORIZON_LABELS[h_idx]} Horizon')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 0.5)
    ax.set_ylim(-0.02, 0.5)

plt.tight_layout()
path = os.path.join(OUTPUT_DIR, 'calibration.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
print(f"  Calibration -> {path}")
if _IN_COLAB: plt.show()
plt.close()

# ── Chart 4: ML metrics comparison bar chart ──
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle(f'ML Metrics Comparison — Fold {FOLD_ID} TEST\n'
             f'L5-1 (baseline) vs L5-3 (training) vs L5-4 (arch) vs L5-5 (features)',
             fontsize=13, fontweight='bold')

metric_sets = [
    ('AUC-ROC', 'auc_roc', True),
    ('PR-AUC', 'pr_auc', True),
    ('Brier Score', 'brier', False),
    ('Pred Separation', 'pred_separation', True),
]

x_pos = np.arange(N_HORIZONS)
bar_w = 0.18
for ax_idx, (metric_name, metric_key, higher_better) in enumerate(metric_sets):
    ax = axes[ax_idx // 2, ax_idx % 2]
    for i, (exp_key, exp_cfg) in enumerate(EXPERIMENTS.items()):
        vals = [ml_metrics[exp_key][h].get(metric_key, 0) for h in range(N_HORIZONS)]
        bars = ax.bar(x_pos + i * bar_w, vals, bar_w, label=exp_key,
                      color=exp_cfg['color'], alpha=0.85, edgecolor='white')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{v:.3f}', ha='center', va='bottom', fontsize=7)
    ax.set_xticks(x_pos + bar_w * 1.5)
    ax.set_xticklabels([HORIZON_LABELS[h] for h in range(N_HORIZONS)])
    ax.set_title(metric_name + (' (higher=better)' if higher_better else ' (lower=better)'))
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
path = os.path.join(OUTPUT_DIR, 'ml_metrics_bars.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
print(f"  ML metrics bars -> {path}")
if _IN_COLAB: plt.show()
plt.close()

# ── Chart 5: Trading metrics comparison (if BQ data exists) ──
if len(bq_df) > 0:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Trading Metrics — A/B/C Comparison (Fold {FOLD_ID})\n'
                 f'Best config per ticker across all modes',
                 fontsize=13, fontweight='bold')

    for exp_i, (exp_key, exp_cfg) in enumerate(EXPERIMENTS.items()):
        mv = exp_cfg['model_version']
        edf = bq_df[bq_df['model_version'] == mv]
        if len(edf) == 0:
            continue
        best = edf.loc[edf.groupby('ticker')['val_b_avg_pnl'].idxmax()]

        ax = axes[0]
        test_pnls = best['val_b_avg_pnl'].values * 100
        ax.hist(test_pnls, bins=20, alpha=0.5, color=exp_cfg['color'],
                label=f"{exp_key} (med={np.median(test_pnls):.1f}%)", edgecolor='white')
    axes[0].axvline(1.8, color='green', linestyle='--', linewidth=1, alpha=0.7, label='Deploy (1.8%)')
    axes[0].axvline(0, color='black', linestyle='-', linewidth=0.8)
    axes[0].set_xlabel('TEST Avg PnL (%)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('TEST PnL Distribution')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    summary_data = []
    for exp_key, exp_cfg in EXPERIMENTS.items():
        mv = exp_cfg['model_version']
        edf = bq_df[bq_df['model_version'] == mv]
        if len(edf) == 0:
            continue
        best = edf.loc[edf.groupby('ticker')['val_b_avg_pnl'].idxmax()]
        summary_data.append({
            'exp': exp_key,
            'color': exp_cfg['color'],
            'n_tickers': len(best),
            'pass_rate': (best['val_b_avg_pnl'] > 0).mean(),
            'deploy': (best['val_b_avg_pnl'] >= 0.018).sum(),
            'mean_test': best['val_b_avg_pnl'].mean(),
            'mean_opt': best['opt_avg_pnl'].mean(),
        })

    if summary_data:
        ax = axes[1]
        x_pos = np.arange(len(summary_data))
        ax.bar(x_pos, [d['pass_rate'] * 100 for d in summary_data],
               color=[d['color'] for d in summary_data], alpha=0.85, edgecolor='white')
        for i, d in enumerate(summary_data):
            ax.text(i, d['pass_rate'] * 100 + 1, f"{d['pass_rate']:.0%}",
                    ha='center', fontsize=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([d['exp'] for d in summary_data])
        ax.set_ylabel('Pass Rate (%)')
        ax.set_title('TEST Pass Rate (PnL > 0)')
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)

        ax = axes[2]
        opt_vals = [d['mean_opt'] * 100 for d in summary_data]
        test_vals = [d['mean_test'] * 100 for d in summary_data]
        x = np.arange(len(summary_data))
        bw = 0.35
        ax.bar(x - bw/2, opt_vals, bw, label='OPT', alpha=0.7,
               color=[d['color'] for d in summary_data], edgecolor='white')
        ax.bar(x + bw/2, test_vals, bw, label='TEST', alpha=0.9,
               color=[d['color'] for d in summary_data], edgecolor='black', linewidth=0.5)
        for i in range(len(summary_data)):
            ax.text(x[i] + bw/2, test_vals[i] + 0.1,
                    f"{test_vals[i]:+.1f}%", ha='center', fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([d['exp'] for d in summary_data])
        ax.set_ylabel('Avg PnL (%)')
        ax.set_title('OPT vs TEST Avg PnL (overfit check)')
        ax.axhline(0, color='black', linewidth=0.8)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'trading_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Trading comparison -> {path}")
    if _IN_COLAB: plt.show()
    plt.close()


# ── Chart 6: Prediction histograms (spike vs non-spike) ──
fig, axes = plt.subplots(N_HORIZONS, len(EXPERIMENTS), figsize=(20, 12))
fig.suptitle(f'Score Distributions — Spike vs Non-Spike (Fold {FOLD_ID}, TEST)',
             fontsize=14, fontweight='bold')

for h_idx in range(N_HORIZONS):
    for exp_i, (exp_key, exp_cfg) in enumerate(EXPERIMENTS.items()):
        ax = axes[h_idx, exp_i]
        Y = all_results[exp_key]['test']['Y'][:, h_idx]
        P = all_results[exp_key]['test']['preds'][:, h_idx]
        ax.hist(P[Y == 0], bins=50, alpha=0.6, color='steelblue',
                label='Non-spike', density=True)
        ax.hist(P[Y == 1], bins=50, alpha=0.6, color='crimson',
                label='Spike', density=True)
        sep = ml_metrics[exp_key][h_idx].get('pred_separation', 0)
        ax.set_title(f'{exp_key} — {HORIZON_LABELS[h_idx]} (sep={sep:.4f})', fontsize=9)
        ax.legend(fontsize=7)
        ax.set_xlabel('Predicted Prob')
        ax.grid(True, alpha=0.2)

plt.tight_layout()
path = os.path.join(OUTPUT_DIR, 'score_distributions.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
print(f"  Score distributions -> {path}")
if _IN_COLAB: plt.show()
plt.close()

# ── Chart 7: Cumulative equity curves (per-ticker BQ configs) ──
fig, axes = plt.subplots(1, 2, figsize=(20, 7))
fig.suptitle(f'Trading Backtest — A/B/C Comparison (Fold {FOLD_ID}, TEST {TEST_START} to {TEST_END})\n'
             f'Per-ticker BQ configs (survived_b, val_b >= {DEPLOY_PNL_THRESHOLD:.1%})',
             fontsize=13, fontweight='bold')

exp_colors = {k: v['color'] for k, v in EXPERIMENTS.items()}

ax = axes[0]
for exp_key, exp_cfg in EXPERIMENTS.items():
    trades = bt_results.get(exp_key, [])
    if not trades:
        continue
    tdf = pd.DataFrame(trades).sort_values('Date')
    cum = tdf['PnL'].cumsum().values * 100
    ax.plot(range(1, len(cum) + 1), cum, color=exp_cfg['color'], linewidth=2,
            label=f"{exp_cfg['label']} ({len(trades)}t, {bt_stats[exp_key]['n_tickers']}tk)")
ax.axhline(0, color='black', linewidth=0.8)
ax.set_xlabel('Trade #')
ax.set_ylabel('Cumulative PnL (%)')
ax.set_title('Cumulative PnL')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[1]
import matplotlib.dates as mdates
for exp_key, exp_cfg in EXPERIMENTS.items():
    trades = bt_results.get(exp_key, [])
    if not trades:
        continue
    tdf = pd.DataFrame(trades)
    daily_eqw = tdf.groupby('Date')['PnL'].mean().sort_index()
    cum_val = daily_eqw.cumsum() * 100
    dates_plot = pd.to_datetime(cum_val.index)
    ax.plot(dates_plot, cum_val.values, color=exp_cfg['color'], linewidth=2,
            marker='o', markersize=4,
            label=f"{exp_cfg['label']} (EqW {cum_val.iloc[-1]:+.1f}%)" if len(cum_val) > 0 else exp_cfg['label'])
ax.axhline(0, color='black', linewidth=0.8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
ax.set_ylabel('Cumulative EqW PnL (%)')
ax.set_title('Equal-Weight Portfolio Equity')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
path = os.path.join(OUTPUT_DIR, 'equity_curves.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
print(f"  Equity curves -> {path}")
if _IN_COLAB: plt.show()
plt.close()

# ── Chart 8: Backtest summary bars + outcome distribution ──
fig, axes = plt.subplots(1, 3, figsize=(22, 7))
fig.suptitle(f'Backtest Results — Per-Ticker BQ Configs (Fold {FOLD_ID} TEST)',
             fontsize=13, fontweight='bold')

ax = axes[0]
exp_keys = list(EXPERIMENTS.keys())
x_pos = np.arange(len(exp_keys))
avg_vals = [bt_stats[k]['avg_pnl'] * 100 for k in exp_keys]
colors = [EXPERIMENTS[k]['color'] for k in exp_keys]
bars = ax.bar(x_pos, avg_vals, color=colors, alpha=0.85, edgecolor='white')
for i, (v, k) in enumerate(zip(avg_vals, exp_keys)):
    s = bt_stats[k]
    ax.text(i, max(v, 0) + 0.1, f"{v:+.2f}%\n{s['n']}t/{s['n_tickers']}tk",
            ha='center', va='bottom', fontsize=8)
ax.set_xticks(x_pos)
ax.set_xticklabels([EXPERIMENTS[k]['label'] for k in exp_keys], fontsize=8, rotation=15, ha='right')
ax.set_ylabel('Avg PnL per Trade (%)')
ax.set_title('Average PnL per Trade')
ax.axhline(0, color='black', linewidth=0.8)
ax.axhline(DEPLOY_PNL_THRESHOLD * 100, color='green', linestyle='--', linewidth=1, alpha=0.5)
ax.grid(axis='y', alpha=0.3)

ax = axes[1]
total_vals = [bt_stats[k]['total_pnl'] * 100 for k in exp_keys]
bars = ax.bar(x_pos, total_vals, color=colors, alpha=0.85, edgecolor='white')
for i, v in enumerate(total_vals):
    ax.text(i, max(v, 0) + 0.5, f"{v:+.1f}%", ha='center', va='bottom', fontsize=9)
ax.set_xticks(x_pos)
ax.set_xticklabels([EXPERIMENTS[k]['label'] for k in exp_keys], fontsize=8, rotation=15, ha='right')
ax.set_ylabel('Total PnL (%)')
ax.set_title('Total Portfolio PnL')
ax.axhline(0, color='black', linewidth=0.8)
ax.grid(axis='y', alpha=0.3)

ax = axes[2]
bar_w = 0.18
for i, exp_key in enumerate(exp_keys):
    trades = bt_results.get(exp_key, [])
    if not trades:
        continue
    tdf = pd.DataFrame(trades)
    vc = tdf['Outcome'].value_counts()
    vals = [vc.get(k, 0) for k in ['TP', 'SL', 'EOD']]
    ax.bar(np.arange(3) + i * bar_w, vals, bar_w, color=EXPERIMENTS[exp_key]['color'],
           alpha=0.8, label=exp_key, edgecolor='white')
ax.set_xticks(np.arange(3) + bar_w * 1.5)
ax.set_xticklabels(['TP', 'SL', 'EOD'])
ax.set_ylabel('Count')
ax.set_title('Outcome Distribution')
ax.legend(fontsize=8)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
path = os.path.join(OUTPUT_DIR, 'backtest_summary.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
print(f"  Backtest summary -> {path}")
if _IN_COLAB: plt.show()
plt.close()

# ── Chart 9: Per-ticker PnL heatmap ──
if ticker_pnl:
    ticker_avg = {}
    for tk, exp_pnls in ticker_pnl.items():
        for exp_key, pnls in exp_pnls.items():
            if isinstance(pnls, list) and pnls:
                ticker_avg.setdefault(tk, {})[exp_key] = np.mean(pnls)

    if ticker_avg:
        tickers_sorted = sorted(ticker_avg.keys(),
                                key=lambda tk: max(ticker_avg[tk].values(), default=0),
                                reverse=True)
        tickers_sorted = tickers_sorted[:40]

        matrix = np.full((len(tickers_sorted), len(EXPERIMENTS)), np.nan)
        for i, tk in enumerate(tickers_sorted):
            for j, exp_key in enumerate(EXPERIMENTS):
                if exp_key in ticker_avg.get(tk, {}):
                    matrix[i, j] = ticker_avg[tk][exp_key] * 100

        fig, ax = plt.subplots(figsize=(8, max(6, len(tickers_sorted) * 0.35)))
        vmax = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)), 3)
        im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn', vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(EXPERIMENTS)))
        ax.set_xticklabels([EXPERIMENTS[k]['label'] for k in EXPERIMENTS], fontsize=8, rotation=30, ha='right')
        ax.set_yticks(range(len(tickers_sorted)))
        ax.set_yticklabels(tickers_sorted, fontsize=7)
        for i in range(len(tickers_sorted)):
            for j in range(len(EXPERIMENTS)):
                v = matrix[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f'{v:+.1f}', ha='center', va='center', fontsize=6,
                            color='white' if abs(v) > vmax * 0.6 else 'black')
        plt.colorbar(im, ax=ax, label='Avg PnL (%)', shrink=0.8)
        ax.set_title(f'Per-Ticker Avg PnL (%) — BQ Configs\n'
                     f'Fold {FOLD_ID} TEST ({TEST_START} to {TEST_END})',
                     fontsize=11, fontweight='bold')
        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, 'ticker_heatmap.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Ticker heatmap -> {path}")
        if _IN_COLAB: plt.show()
        plt.close()


# ==============================================================================
# FINAL VERDICT
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  FINAL COMPARISON SUMMARY")
print(f"{'='*W}")

print(f"\n  ML METRICS (TEST, 30m horizon — primary trading horizon):")
h = 1
for exp_key, exp_cfg in EXPERIMENTS.items():
    m = ml_metrics[exp_key][h]
    print(f"    {exp_cfg['label']:<28}  AUC={m['auc_roc']:.4f}  "
          f"PR={m['pr_auc']:.4f}  Brier={m['brier']:.4f}  "
          f"Sep={m['pred_separation']:.4f}")

if len(bq_df) > 0:
    print(f"\n  TRADING METRICS (best config per ticker, all modes):")
    for exp_key, exp_cfg in EXPERIMENTS.items():
        mv = exp_cfg['model_version']
        edf = bq_df[bq_df['model_version'] == mv]
        if len(edf) == 0:
            print(f"    {exp_cfg['label']:<28}  (no optimizer data)")
            continue
        best = edf.loc[edf.groupby('ticker')['val_b_avg_pnl'].idxmax()]
        n = len(best)
        mean_test = best['val_b_avg_pnl'].mean()
        n_deploy = (best['val_b_avg_pnl'] >= 0.018).sum()
        pass_rate = (best['val_b_avg_pnl'] > 0).mean()
        print(f"    {exp_cfg['label']:<28}  "
              f"TestPnL={mean_test:+.2%}  Pass={pass_rate:.0%}  "
              f"Deploy={n_deploy}/{n}")

print(f"\n  BACKTEST (per-ticker BQ configs, TEST window {TEST_START}->{TEST_END}):")
ranked_bt = sorted(
    [(k, bt_stats[k]) for k in EXPERIMENTS if bt_stats[k]['n'] > 0],
    key=lambda x: x[1]['avg_pnl'], reverse=True)
for exp_key, s in ranked_bt:
    exp_cfg = EXPERIMENTS[exp_key]
    print(f"    {exp_cfg['label']:<28}  "
          f"AvgPnL={s['avg_pnl']:+.2%}  Total={s['total_pnl']*100:+.1f}%  "
          f"N={s['n']}  WR={s['win_rate']:.0%}  "
          f"{s['n_tickers']}tk  Sharpe={s['sharpe']:.1f}")
if ranked_bt:
    winner = ranked_bt[0]
    print(f"\n  BACKTEST WINNER: {EXPERIMENTS[winner[0]]['label']} "
          f"({winner[1]['avg_pnl']:+.2%}/trade, {winner[1]['n']} trades)")

print(f"\n  Output:  {OUTPUT_DIR}/")
for f in sorted(os.listdir(OUTPUT_DIR)):
    sz = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1e6
    print(f"    {f}  ({sz:.1f} MB)")
print(f"{'='*W}")
