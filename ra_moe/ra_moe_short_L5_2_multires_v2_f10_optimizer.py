# %% ── Spike Transformer SHORT L5-2 MULTI-RES v2: WF Optimizer (Fold 10) ───────
#
# Walk-forward optimizer for the SHORT L5-2 MULTI-RES v2 model
# (dual Transformer backbone: 1-min + 15-min, trained on ALL GCS tickers).
#
# Fold 10 — rolling 1-week offset from F9, extending the F6/F7/F8/F9 series.
# ALL tickers discovered from GCS master_inference.
#
# Phase 1: Load model weights, score val+test per fold
# Phase 2: Per-ticker grid search on OPT window, validate on TEST window, save to BQ
#
# FILL COMPENSATION: limit_entry mode uses ROAC = total_pnl / n_signals
# (unfilled signals count as 0% PnL, penalizing low fill rates).
#
# Fold 10 timeline:
#   F10: OPT Jul 22–Aug 21, TEST Aug 22–Aug 30
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
FOLD_CONFIGS = [
    {'fold': 10, 'opt_start': '2026-07-22', 'opt_end': '2026-08-21',
     'test_start': '2026-08-22', 'test_end': '2026-08-30'},
]
FOLDS_TO_RUN = [10]

SCORE_HORIZONS = [0, 1, 2]
HORIZON_LABELS = {0: '15m', 1: '30m', 2: '60m'}
DATA_START     = '2024-08-01'

FOLD_ID    = None
OPT_START  = None
OPT_END    = None
TEST_START = None
TEST_END   = None

SCORE_MINUTE   = 375
ENTRY_MINUTE   = 385
ENTRY_TIME     = '06:25'
EXIT_TIME_MIN  = 780
SLIPPAGE_PCT   = 0.0015

MODEL_VERSION  = None
EXPERIMENT     = None
DEPLOY_PNL_THRESHOLD = 0.018
MIN_TRADES_PER_MONTH = 2.0
MIN_TRADES_ABSOLUTE  = 2

GCS_BUCKET       = os.environ.get("GCS_BUCKET", "")
GCS_DATA_PREFIX  = 'Algos/RA_MoE/short_L5_2_multires_v2_data'
GCS_MODEL_PREFIX = 'Algos/RA_MoE/short_L5_2_multires_v2_models'
GCS_INFER_DIR    = 'Data/Inference_v2'

DATA_DIR   = '/tmp/ra_moe_short_L5_2_multires_v2_data'
MODEL_DIR  = '/tmp/ra_moe_short_L5_2_multires_v2_models'
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Architecture (must match training — dual backbone)
D_MODEL = 64; N_HEADS = 4; D_FF = 128; N_TRANSFORMER_BLOCKS = 2
N_BLOCKS_15M = 1
DROPOUT = 0.1; FILM_HIDDEN = 64; TIME_STEPS = 128; TIME_STEPS_15M = 128
TARGET_HORIZONS = [15, 30, 60]; N_HORIZONS = 3

# Loss constants (needed for model build)
ASL_GAMMA_NEG = 6.0; ASL_GAMMA_POS = 0.1; ASL_CLIP = 0.08

N_ASSET_FEATURES  = None
N_REGIME_FEATURES = None
N_LOWRES_FEATURES = None   # auto-detected from npz

# Grid
PROB_THRESHOLDS = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

TPSL_PAIRS = {
    'static': [
        {'tp_pct': 0.06, 'sl_pct': 0.03},
        {'tp_pct': 0.08, 'sl_pct': 0.04},
        {'tp_pct': 0.12, 'sl_pct': 0.06},
        {'tp_pct': 0.16, 'sl_pct': 0.08},
        {'tp_pct': 0.16, 'sl_pct': 0.12},
    ],
}

GRACE_CONFIGS = [
    {'tp_pct': 0.08, 'sl_pct': 0.04, 'grace_minutes': 15},
    {'tp_pct': 0.12, 'sl_pct': 0.06, 'grace_minutes': 15},
    {'tp_pct': 0.16, 'sl_pct': 0.08, 'grace_minutes': 15},
]

LIMIT_ENTRY_CONFIGS = [
    {'tp_pct': 0.08, 'sl_pct': 0.04, 'dip_pct': 0.005, 'max_wait': 30},
    {'tp_pct': 0.08, 'sl_pct': 0.04, 'dip_pct': 0.01, 'max_wait': 30},
    {'tp_pct': 0.08, 'sl_pct': 0.04, 'dip_pct': 0.0125, 'max_wait': 30},
    {'tp_pct': 0.16, 'sl_pct': 0.08, 'dip_pct': 0.01, 'max_wait': 30},
    {'tp_pct': 0.16, 'sl_pct': 0.08, 'dip_pct': 0.015, 'max_wait': 30},
    {'tp_pct': 0.12, 'sl_pct': 0.06, 'dip_pct': 0.02, 'max_wait': 30},
    {'tp_pct': 0.16, 'sl_pct': 0.08, 'dip_pct': 0.025, 'max_wait': 60},
    {'tp_pct': 0.20, 'sl_pct': 0.10, 'dip_pct': 0.04, 'max_wait': 90},
]

# ── Vol-scaled TP/SL (#5) ──
# Base TP/SL scaled by trailing realized vol ratio at entry
VOLSCALED_BASE_PAIRS = [
    {'tp_pct': 0.06, 'sl_pct': 0.03},
    {'tp_pct': 0.08, 'sl_pct': 0.04},
    {'tp_pct': 0.12, 'sl_pct': 0.06},
    {'tp_pct': 0.16, 'sl_pct': 0.08},
    {'tp_pct': 0.16, 'sl_pct': 0.12},
]
VOL_LOOKBACK  = 20
VOL_SCALE_MIN = 0.5
VOL_SCALE_MAX = 2.0

# ── Multi-horizon ensemble scoring (#7) ──
# Blend 15m/30m/60m scores instead of picking one horizon
ENSEMBLE_PRESETS = [
    {'name': 'equal',  'weights': [1/3, 1/3, 1/3]},
    {'name': 'short',  'weights': [0.5, 0.3, 0.2]},
    {'name': 'medium', 'weights': [0.2, 0.5, 0.3]},
    {'name': 'max',    'weights': None},
]
ENSEMBLE_TPSL_PAIRS = [
    {'tp_pct': 0.06, 'sl_pct': 0.03},
    {'tp_pct': 0.08, 'sl_pct': 0.04},
    {'tp_pct': 0.12, 'sl_pct': 0.06},
    {'tp_pct': 0.16, 'sl_pct': 0.08},
    {'tp_pct': 0.16, 'sl_pct': 0.12},
]

RUN_MODES = ['static', 'grace', 'limit_entry', 'volscaled', 'ensemble']
MODE_SUFFIX = {'static': '', 'grace': '_GRACE', 'limit_entry': '_LIMIT',
               'volscaled': '_VOLSCALED', 'ensemble': '_ENSEMBLE'}

SKIP_EXISTING_MODES = False

TPSL_MODE = 'static'

NO_FILTER_DEFAULTS = {'vix_threshold': 999.0, 'tlt_crash': -999.0, 'spy_ignore': 999.0}

TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}
EXCLUDE_TICKERS = {
    'SPY', 'VIX', 'TLT', 'IWM', 'XLE', 'ITA',
    'BITF', 'MIGI', 'SMLR', 'BTCUSD', 'ETHUSD',
}

PST = pytz.timezone('America/Los_Angeles')

WIPE_EXISTING  = True
TICKER_SUBSET  = []

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
# MODEL — DUAL BACKBONE (MultiResSpikeTransformer)
# ==============================================================================

class MultiResSpikeTransformer(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 1-min backbone: 2 blocks (same as baseline)
        self.backbone_1m  = TransformerBackbone(
            D_MODEL, N_HEADS, D_FF, N_TRANSFORMER_BLOCKS,
            TIME_STEPS, DROPOUT, name='backbone_1m')
        self.pool_1m      = AttentionPooling(D_MODEL, name='pool_1m')

        # 15-min backbone: 1 block (lighter — fewer features, smoother signal)
        self.backbone_15m = TransformerBackbone(
            D_MODEL, N_HEADS, D_FF, N_BLOCKS_15M,
            TIME_STEPS_15M, DROPOUT, name='backbone_15m')
        self.pool_15m     = AttentionPooling(D_MODEL, name='pool_15m')

        # Late fusion: concat(64, 64) -> 64
        self.fusion = layers.Dense(D_MODEL, activation='gelu', name='fusion')

        # FiLM + head (unchanged)
        self.film = FiLMConditioner(D_MODEL, FILM_HIDDEN, name='film')
        self.drop = layers.Dropout(DROPOUT)
        self.head = layers.Dense(N_HORIZONS, activation='sigmoid',
                                  name='spike_head')

    def call(self, inputs, training=False):
        seq_1m  = inputs['seq_input']
        seq_15m = inputs['seq_15m_input']
        regime  = inputs['regime_input']

        # Process both resolutions independently
        h_1m  = self.backbone_1m(seq_1m, training=training)
        h_1m  = self.pool_1m(h_1m, training=training)

        h_15m = self.backbone_15m(seq_15m, training=training)
        h_15m = self.pool_15m(h_15m, training=training)

        # Late fusion
        h_cat = tf.concat([h_1m, h_15m], axis=-1)  # (B, 2*d_model)
        h     = self.fusion(h_cat)                   # (B, d_model)

        # FiLM conditioning + prediction
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
        SchemaField('val_b_fill_rate', 'FLOAT64'),
        SchemaField('val_b_n_signals', 'INT64'),
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


def save_row_to_bq(row_dict, run_id, experiment, mode,
                    opt_s, opt_e, va_s, va_e, vb_s='', vb_e=''):
    try:
        row = {
            'run_id': run_id, 'run_timestamp': pd.Timestamp.utcnow().isoformat(),
            'experiment': experiment,
            'ticker': row_dict['Ticker'], 'seed': 0,
            'entry_time': ENTRY_TIME, 'entry_minute': ENTRY_MINUTE,
            'prob_thresh': float(row_dict['prob_thresh']),
            'tp_pct': float(row_dict['tp_pct']),
            'sl_pct': float(row_dict['sl_pct']),
            'vix_threshold': NO_FILTER_DEFAULTS['vix_threshold'],
            'tlt_crash': NO_FILTER_DEFAULTS['tlt_crash'],
            'spy_ignore': NO_FILTER_DEFAULTS['spy_ignore'],
            'min_vol_bars': 0,
            'vol_filter_pct': 0.0,
            'opt_avg_pnl': float(row_dict['Avg_PnL']),
            'opt_win_rate': float(row_dict['Win_Rate']),
            'opt_freq': float(row_dict['Freq']),
            'opt_count': int(row_dict['Count']),
            'val_a_avg_pnl': float(row_dict.get('val_a_avg_pnl', 0)),
            'val_a_win_rate': float(row_dict.get('val_a_win_rate', 0)),
            'val_a_freq': float(row_dict.get('val_a_freq', 0)),
            'val_a_count': int(row_dict.get('val_a_count', 0)),
            'survived_a': bool(row_dict.get('survived_a', False)),
            'val_b_avg_pnl': float(row_dict.get('val_b_avg_pnl', 0)),
            'val_b_win_rate': float(row_dict.get('val_b_win_rate', 0)),
            'val_b_freq': float(row_dict.get('val_b_freq', 0)),
            'val_b_count': int(row_dict.get('val_b_count', 0)),
            'survived_b': bool(row_dict.get('survived_b', False)),
            'opt_start': opt_s, 'opt_end': opt_e,
            'val_a_start': va_s, 'val_a_end': va_e,
            'val_b_start': vb_s or '', 'val_b_end': vb_e or '',
            'tpsl_mode': mode,
            'tp_mult': None, 'sl_mult': None,
            'trail_start_mult': None, 'trail_end_mult': None,
            'model_version': MODEL_VERSION,
            'v0_fixed': True,
            'is_ensemble': mode == 'ensemble', 'n_seeds_avg': 1,
            'regime_tp_pct': None, 'regime_sl_pct': None,
            'regime_lookback': None, 'regime_spike_thresh': None,
            'grace_minutes': int(row_dict.get('grace_minutes', 0)) if mode == 'grace' else None,
            'dip_pct': float(row_dict.get('dip_pct', 0)) if mode == 'limit_entry' else None,
            'max_wait_minutes': int(row_dict.get('max_wait', 0)) if mode == 'limit_entry' else None,
            'pre_range_filter_pct': 0.0,
            'val_b_fill_rate': float(row_dict.get('val_b_fill_rate', 1.0)),
            'val_b_n_signals': int(row_dict.get('val_b_n_signals', row_dict.get('val_b_count', 0))),
        }
        errors = _bq_client.insert_rows_json(BQ_TABLE_ID, [row])
        if errors:
            print(f"      BQ error: {errors[0]}")
        else:
            print(f"      BQ: saved {row_dict['Ticker']}/{mode}")
    except Exception as e:
        print(f"      BQ save failed: {e}")


# ==============================================================================
# SHORT SIMULATION ENGINES (Numba)
# ==============================================================================

@jit(nopython=True)
def run_short_trade_sim(entry_indices, prices, highs, lows, date_ints, time_mins,
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
        target_price = entry_price * (1 - tp)
        stop_price   = entry_price * (1 + sl)
        outcome    = 0.0
        trade_done = False
        exit_idx   = idx
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
            outcome  = (entry_price - prices[n_prices-1]) / entry_price
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
def run_short_trade_sim_grace(entry_indices, prices, highs, lows, date_ints, time_mins,
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
        target_price = entry_price * (1 - tp)
        stop_price   = entry_price * (1 + sl)
        entry_time   = time_mins[idx]
        outcome    = 0.0
        trade_done = False
        exit_idx   = idx
        for k in range(idx + 1, n_prices):
            if date_ints[k] != entry_date:
                outcome = (entry_price - prices[k-1]) / entry_price
                exit_idx = k; trade_done = True; break
            minutes_elapsed = time_mins[k] - entry_time
            sl_active = minutes_elapsed >= grace_minutes
            tp_hit = lows[k] <= target_price
            sl_hit = sl_active and (highs[k] >= stop_price)
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
            outcome  = (entry_price - prices[n_prices-1]) / entry_price
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
def run_short_trade_sim_limit_entry(entry_indices, prices, highs, lows, date_ints, time_mins,
                                    tp, sl, exit_time, rally_pct, max_wait_minutes):
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
        trade_done = False
        exit_idx   = fill_idx
        for k in range(fill_idx + 1, n_prices):
            if date_ints[k] != entry_date:
                outcome = (fill_price - prices[k-1]) / fill_price
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
                outcome = (fill_price - prices[k]) / fill_price
                exit_idx = k; trade_done = True; break
        if not trade_done:
            outcome  = (fill_price - prices[n_prices-1]) / fill_price
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
def run_short_trade_sim_volscaled(entry_indices, prices, highs, lows, date_ints, time_mins,
                                  base_tp, base_sl, exit_time, vol_scales):
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
        vs = vol_scales[idx]
        tp = base_tp * vs
        sl = base_sl * vs
        target_price = entry_price * (1 - tp)
        stop_price   = entry_price * (1 + sl)
        outcome    = 0.0
        trade_done = False
        exit_idx   = idx
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
            outcome  = (entry_price - prices[n_prices-1]) / entry_price
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
        pnl, d = run_short_trade_sim_grace(
            indices, prices, highs, lows, dates, times,
            s_cfg['tp_pct'], s_cfg['sl_pct'], EXIT_TIME_MIN, s_cfg['grace_minutes'])
    elif mode == 'limit_entry':
        pnl, d = run_short_trade_sim_limit_entry(
            indices, prices, highs, lows, dates, times,
            s_cfg['tp_pct'], s_cfg['sl_pct'], EXIT_TIME_MIN,
            s_cfg['dip_pct'], s_cfg['max_wait'])
    elif mode == 'volscaled':
        vol_scales = ohlcv_arrays.get('vol_scales')
        if vol_scales is None:
            vol_scales = np.ones(len(prices), dtype=np.float64)
        pnl, d = run_short_trade_sim_volscaled(
            indices, prices, highs, lows, dates, times,
            s_cfg['tp_pct'], s_cfg['sl_pct'], EXIT_TIME_MIN, vol_scales)
    else:
        pnl, d = run_short_trade_sim(
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


def compute_vol_scales(df_ohlcv, ohlcv_dates, lookback=VOL_LOOKBACK):
    df = df_ohlcv.copy()
    df['date_int'] = (df.index.year * 10000 + df.index.month * 100 + df.index.day).astype(np.int64)
    daily_close = df.groupby('date_int')['Close'].last().sort_index()
    returns = daily_close.pct_change().dropna()
    rolling_vol = returns.rolling(lookback, min_periods=max(5, lookback // 2)).std()
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


def compute_ensemble_scores(daily_scores, preset):
    weights = preset['weights']
    use_max = weights is None
    all_dates = set()
    for h in SCORE_HORIZONS:
        all_dates.update(daily_scores.get(h, {}).keys())
    blended = {}
    for d in all_dates:
        scores = [daily_scores.get(h, {}).get(d, 0.0) for h in SCORE_HORIZONS]
        if use_max:
            blended[d] = max(scores)
        else:
            blended[d] = sum(w * s for w, s in zip(weights, scores))
    return blended


# ==============================================================================
# SCORING PHASE (multi-resolution: 3 inputs)
# ==============================================================================

def score_npz_split(model, split, ticker_names):
    path = ensure_data(f'fold_{FOLD_ID}_{split}.npz')
    print(f"    Loading {split} npz...")
    data = np.load(path)
    X_seq     = data['X_seq']
    X_seq_15m = data['X_seq_15m']
    X_reg     = data['X_regime']
    X_id      = data['X_id'].flatten()
    date_ints = data['date_ints']
    minutes   = data['minutes']
    n_total   = len(X_id)
    data.close()
    entry_mask = (minutes == SCORE_MINUTE)
    n_entry = entry_mask.sum()
    print(f"    Filtering {n_total:,} -> {n_entry:,} score-minute windows (min={SCORE_MINUTE})")
    X_seq     = X_seq[entry_mask]
    X_seq_15m = X_seq_15m[entry_mask]
    X_reg     = X_reg[entry_mask]
    X_id      = X_id[entry_mask]
    date_ints = date_ints[entry_mask]
    del minutes, entry_mask; gc.collect()
    n = n_entry
    print(f"    Running inference on {n:,} samples...")
    preds = model.predict(
        {'seq_input': X_seq.astype(np.float32),
         'seq_15m_input': X_seq_15m.astype(np.float32),
         'regime_input': X_reg.astype(np.float32)},
        batch_size=512, verbose=0)
    del X_seq, X_seq_15m, X_reg; gc.collect()
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
    print(f"  PHASE 1: SCORING — Spike Transformer SHORT L5-2 MULTI-RES v2 fold {FOLD_ID}")
    print(f"  Dual Transformer Backbone (1-min + 15-min, 22 regime features)")
    print(f"  Score at minute {SCORE_MINUTE} (6:15 AM), enter at minute {ENTRY_MINUTE} (6:25 AM)")
    print(f"{'='*W}")
    meta_path = ensure_data('meta.npz')
    meta = np.load(meta_path, allow_pickle=True)
    ticker_names = meta['ticker_names']
    print(f"  {len(ticker_names)} tickers in meta")
    print("  Building model...")
    global N_ASSET_FEATURES, N_REGIME_FEATURES, N_LOWRES_FEATURES
    probe_path = ensure_data(f'fold_{FOLD_ID}_val.npz')
    probe = np.load(probe_path)
    N_ASSET_FEATURES  = probe['X_seq'].shape[-1]
    N_LOWRES_FEATURES = probe['X_seq_15m'].shape[-1]
    N_REGIME_FEATURES = probe['X_regime'].shape[-1]
    probe.close()
    print(f"    N_ASSET_FEATURES={N_ASSET_FEATURES}, N_LOWRES_FEATURES={N_LOWRES_FEATURES}, "
          f"N_REGIME_FEATURES={N_REGIME_FEATURES}")
    model = MultiResSpikeTransformer(name='multires_spike_transformer')
    dummy_seq  = tf.zeros((1, TIME_STEPS, N_ASSET_FEATURES))
    dummy_15m  = tf.zeros((1, TIME_STEPS_15M, N_LOWRES_FEATURES))
    dummy_reg  = tf.zeros((1, N_REGIME_FEATURES))
    model({'seq_input': dummy_seq, 'seq_15m_input': dummy_15m, 'regime_input': dummy_reg})
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
# GRID SEARCH + OPTIMIZATION
# ==============================================================================

def grid_search(ohlcv_arrays, daily_scores, ticker, mode):
    dates = ohlcv_arrays['dates']
    times = ohlcv_arrays['times']
    months = ohlcv_arrays['months']
    if mode == 'grace':
        configs = GRACE_CONFIGS
    elif mode == 'limit_entry':
        configs = LIMIT_ENTRY_CONFIGS
    elif mode == 'volscaled':
        configs = VOLSCALED_BASE_PAIRS
    else:
        configs = TPSL_PAIRS['static']
    results = []
    for h_idx in SCORE_HORIZONS:
        h_scores = daily_scores.get(h_idx, {})
        if not h_scores:
            continue
        score_by_date = {}
        for d_int, score in h_scores.items():
            score_by_date[int(d_int)] = score
        for prob_thresh in PROB_THRESHOLDS:
            entry_list = []
            for i in range(len(dates)):
                if times[i] == ENTRY_MINUTE:
                    day_score = score_by_date.get(int(dates[i]), 0.0)
                    if day_score >= prob_thresh:
                        entry_list.append(i)
            if len(entry_list) == 0:
                continue
            indices = np.array(entry_list, dtype=np.int64)
            n_signals = len(indices)
            for pair in configs:
                s_cfg = {'entry_time': ENTRY_TIME, 'prob_thresh': prob_thresh, **pair}
                raw_pnl, raw_dates = _run_sim(indices, ohlcv_arrays, s_cfg, mode)
                if len(raw_pnl) == 0:
                    continue
                n = len(raw_pnl)
                if n < MIN_TRADES_ABSOLUTE:
                    continue
                freq = n / months
                if freq < MIN_TRADES_PER_MONTH:
                    continue
                avg = float(np.mean(raw_pnl))
                total = float(np.sum(raw_pnl))
                fill_rate = n / n_signals if n_signals > 0 else 1.0
                roac = total / n_signals if n_signals > 0 else 0.0
                result_row = {
                    'Ticker': ticker,
                    'score_horizon': h_idx,
                    'horizon_label': HORIZON_LABELS[h_idx],
                    **s_cfg,
                    'Avg_PnL': avg,
                    'Total_PnL': total,
                    'Win_Rate': float(np.mean(raw_pnl > 0)),
                    'Freq': freq,
                    'Count': n,
                    'N_Signals': n_signals,
                    'Fill_Rate': fill_rate,
                    'ROAC': roac,
                    '_raw_pnl': raw_pnl.copy(),
                }
                results.append(result_row)
    return results


def grid_search_ensemble(ohlcv_arrays, daily_scores, ticker):
    dates = ohlcv_arrays['dates']
    times = ohlcv_arrays['times']
    months = ohlcv_arrays['months']
    results = []
    for preset in ENSEMBLE_PRESETS:
        blended = compute_ensemble_scores(daily_scores, preset)
        if not blended:
            continue
        score_by_date = {int(d): s for d, s in blended.items()}
        for prob_thresh in PROB_THRESHOLDS:
            entry_list = []
            for i in range(len(dates)):
                if times[i] == ENTRY_MINUTE:
                    day_score = score_by_date.get(int(dates[i]), 0.0)
                    if day_score >= prob_thresh:
                        entry_list.append(i)
            if not entry_list:
                continue
            indices = np.array(entry_list, dtype=np.int64)
            for pair in ENSEMBLE_TPSL_PAIRS:
                s_cfg = {'entry_time': ENTRY_TIME, 'prob_thresh': prob_thresh, **pair}
                raw_pnl, raw_dates = _run_sim(indices, ohlcv_arrays, s_cfg, 'static')
                if len(raw_pnl) == 0:
                    continue
                n = len(raw_pnl)
                if n < MIN_TRADES_ABSOLUTE:
                    continue
                freq = n / months
                if freq < MIN_TRADES_PER_MONTH:
                    continue
                avg = float(np.mean(raw_pnl))
                result_row = {
                    'Ticker': ticker,
                    'ensemble_preset': preset['name'],
                    'score_horizon': -1,
                    'horizon_label': f"ens_{preset['name']}",
                    **s_cfg,
                    'Avg_PnL': avg,
                    'Total_PnL': float(np.sum(raw_pnl)),
                    'Win_Rate': float(np.mean(raw_pnl > 0)),
                    'Freq': freq,
                    'Count': n,
                    '_raw_pnl': raw_pnl.copy(),
                }
                results.append(result_row)
    return results


def eval_combo(ohlcv_arrays, daily_scores, params, mode):
    _empty = {'Avg_PnL': 0, 'Count': 0, 'Win_Rate': 0, 'Freq': 0,
              'N_Signals': 0, 'Fill_Rate': 1.0, 'ROAC': 0}
    if ohlcv_arrays is None:
        return _empty
    dates = ohlcv_arrays['dates']
    times = ohlcv_arrays['times']
    months = ohlcv_arrays['months']
    if mode == 'ensemble':
        preset_name = params.get('ensemble_preset', 'equal')
        preset = next((p for p in ENSEMBLE_PRESETS if p['name'] == preset_name), ENSEMBLE_PRESETS[0])
        blended = compute_ensemble_scores(daily_scores, preset)
        score_by_date = {int(d): s for d, s in blended.items()}
    else:
        h_idx = params.get('score_horizon', 1)
        h_scores = daily_scores.get(h_idx, {})
        score_by_date = {int(d_int): score for d_int, score in h_scores.items()}
    entry_list = []
    for i in range(len(dates)):
        if times[i] == ENTRY_MINUTE:
            day_score = score_by_date.get(int(dates[i]), 0.0)
            if day_score >= params['prob_thresh']:
                entry_list.append(i)
    if len(entry_list) == 0:
        return _empty
    indices = np.array(entry_list, dtype=np.int64)
    n_signals = len(indices)
    sim_mode = 'static' if mode == 'ensemble' else mode
    raw_pnl, raw_dates = _run_sim(indices, ohlcv_arrays, params, sim_mode)
    if len(raw_pnl) == 0:
        return {**_empty, 'N_Signals': n_signals, 'Fill_Rate': 0.0}
    n_filled = len(raw_pnl)
    fill_rate = n_filled / n_signals if n_signals > 0 else 1.0
    total_pnl = float(np.sum(raw_pnl))
    roac = total_pnl / n_signals if n_signals > 0 else 0.0
    return {
        'Avg_PnL':   float(np.mean(raw_pnl)),
        'Total_PnL': total_pnl,
        'Win_Rate':  float(np.mean(raw_pnl > 0)),
        'Freq':      n_filled / months,
        'Count':     n_filled,
        'N_Signals': n_signals,
        'Fill_Rate': fill_rate,
        'ROAC':      roac,
    }


def select_robust_params(all_results, mode):
    if not all_results:
        return None
    strategy_groups = {}
    for r in all_results:
        h = r.get('score_horizon', 1)
        if mode == 'grace':
            key = (h, r['prob_thresh'], r.get('tp_pct', 0), r.get('sl_pct', 0),
                   r.get('grace_minutes', 0))
        elif mode == 'limit_entry':
            key = (h, r['prob_thresh'], r.get('tp_pct', 0), r.get('sl_pct', 0),
                   r.get('dip_pct', 0), r.get('max_wait', 0))
        elif mode == 'ensemble':
            key = (r.get('ensemble_preset', ''), r['prob_thresh'],
                   r.get('tp_pct', 0), r.get('sl_pct', 0))
        else:
            key = (h, r['prob_thresh'], r.get('tp_pct', 0), r.get('sl_pct', 0))
        strategy_groups.setdefault(key, []).append(r)
    rank_key = 'ROAC' if mode == 'limit_entry' else 'Avg_PnL'
    best = None
    best_robust_score = -999.0
    for key, results in strategy_groups.items():
        pnls = np.array([r.get(rank_key, r['Avg_PnL']) for r in results])
        mean_pnl = float(np.mean(pnls))
        if mean_pnl <= 0:
            continue
        best_in_group = max(results, key=lambda r: r.get(rank_key, r['Avg_PnL']))
        raw = best_in_group.get('_raw_pnl')
        consistency = 1.0
        if raw is not None and len(raw) >= 4:
            std_pnl = float(np.std(raw))
            consistency = max(0.0, 1.0 - std_pnl / abs(mean_pnl))
        robust_score = mean_pnl * (0.5 + 0.5 * consistency)
        if robust_score > best_robust_score:
            best_robust_score = robust_score
            best = best_in_group
    return best


# ==============================================================================
# MAIN — LOOP OVER FOLDS
# ==============================================================================

active_folds = [fc for fc in FOLD_CONFIGS if fc['fold'] in FOLDS_TO_RUN]
active_modes = [m for m in RUN_MODES if not SKIP_EXISTING_MODES or m in ('volscaled', 'ensemble')]

print(f"\n{'='*W}")
print(f"  SHORT L5-2 MULTI-RES v2 WF OPTIMIZER — FOLD 10")
print(f"  Running {len(active_folds)} fold(s), {len(active_modes)} mode(s) each")
print(f"  Fill compensation: limit_entry ranked by ROAC = total_pnl / n_signals")
print(f"  ALL tickers from GCS master_inference")
print(f"{'='*W}\n")

grand_deploy_rows = []

for fold_cfg in active_folds:
    FOLD_ID    = fold_cfg['fold']
    OPT_START  = fold_cfg['opt_start']
    OPT_END    = fold_cfg['opt_end']
    TEST_START = fold_cfg['test_start']
    TEST_END   = fold_cfg['test_end']
    MODEL_VERSION = f'spike_transformer_short_L5_2_multires_v2_fold{FOLD_ID}'
    EXPERIMENT    = f'spike_transformer_short_L5_2_multires_v2_fold{FOLD_ID}'

    print(f"\n\n{'#'*W}")
    print(f"  FOLD {FOLD_ID} — OPT: {OPT_START} -> {OPT_END}  TEST: {TEST_START} -> {TEST_END}")
    print(f"  model_version: {MODEL_VERSION}")
    print(f"{'#'*W}")

    # ── Phase 1: Scoring ──
    all_daily_scores = run_scoring_phase()

    # ── Phase 2: Grid search ──
    print(f"\n{'='*W}")
    print(f"  PHASE 2: WALK-FORWARD OPTIMIZATION (FOLD {FOLD_ID}, DUAL TRANSFORMER BACKBONE)")
    if SKIP_EXISTING_MODES:
        print(f"  *** NEW MODES ONLY (SKIP_EXISTING_MODES=True) ***")
    print(f"  ACTIVE MODES: {', '.join(m.upper() for m in active_modes)}")
    print(f"  OPT:   {OPT_START} -> {OPT_END}")
    print(f"  TEST:  {TEST_START} -> {TEST_END}")
    print(f"  SCORE: 6:15 AM (min {SCORE_MINUTE})  ENTRY: 6:25 AM (min {ENTRY_MINUTE})")
    print(f"  SLIPPAGE: {SLIPPAGE_PCT:.2%} each way ({2*SLIPPAGE_PCT:.2%} RT)")
    print(f"  PROB THRESHOLDS: {PROB_THRESHOLDS}")
    for m in active_modes:
        if m == 'grace':
            nc = len(GRACE_CONFIGS)
        elif m == 'limit_entry':
            nc = len(LIMIT_ENTRY_CONFIGS)
        elif m == 'volscaled':
            nc = len(VOLSCALED_BASE_PAIRS)
        elif m == 'ensemble':
            nc = len(ENSEMBLE_TPSL_PAIRS)
        else:
            nc = len(TPSL_PAIRS['static'])
        if m == 'ensemble':
            nh = len(ENSEMBLE_PRESETS)
            print(f"    {m.upper():>18}: {nh} presets x {len(PROB_THRESHOLDS)} prob x {nc} tpsl = {nh*len(PROB_THRESHOLDS)*nc} grid pts")
        else:
            nh = len(SCORE_HORIZONS)
            print(f"    {m.upper():>18}: {nh} horizons x {len(PROB_THRESHOLDS)} prob x {nc} tpsl = {nh*len(PROB_THRESHOLDS)*nc} grid pts")
    if 'volscaled' in active_modes:
        print(f"  VOL-SCALED: lookback={VOL_LOOKBACK}d, scale=[{VOL_SCALE_MIN:.1f}, {VOL_SCALE_MAX:.1f}]")
    if 'ensemble' in active_modes:
        print(f"  ENSEMBLE PRESETS: {', '.join(p['name'] for p in ENSEMBLE_PRESETS)}")
    print(f"{'='*W}\n")

    if WIPE_EXISTING:
        if SKIP_EXISTING_MODES:
            wipe_suffixes = [MODE_SUFFIX[m] for m in active_modes if MODE_SUFFIX.get(m)]
            for sfx in wipe_suffixes:
                try:
                    del_q = (f"DELETE FROM `{BQ_TABLE_ID}` "
                             f"WHERE model_version = '{MODEL_VERSION}' "
                             f"AND experiment LIKE '%{sfx}'")
                    _bq_client.query(del_q).result()
                except Exception as e:
                    if 'streaming buffer' in str(e).lower():
                        copy_q = (f"CREATE OR REPLACE TABLE `{BQ_TABLE_ID}` AS "
                                  f"SELECT * FROM `{BQ_TABLE_ID}` "
                                  f"WHERE NOT (model_version = '{MODEL_VERSION}' "
                                  f"AND experiment LIKE '%{sfx}')")
                        _bq_client.query(copy_q).result()
                    else:
                        raise
            print(f"  WIPE: deleted experiments matching {wipe_suffixes} for '{MODEL_VERSION}'")
        else:
            try:
                del_q = (f"DELETE FROM `{BQ_TABLE_ID}` "
                         f"WHERE model_version = '{MODEL_VERSION}'")
                _bq_client.query(del_q).result()
                print(f"  WIPE: deleted model_version='{MODEL_VERSION}'")
            except Exception as e:
                if 'streaming buffer' in str(e).lower():
                    copy_q = (f"CREATE OR REPLACE TABLE `{BQ_TABLE_ID}` AS "
                              f"SELECT * FROM `{BQ_TABLE_ID}` "
                              f"WHERE model_version != '{MODEL_VERSION}'")
                    _bq_client.query(copy_q).result()
                    print(f"  WIPE: rebuilt without '{MODEL_VERSION}'")
                else:
                    raise

    RUN_ID = str(uuid.uuid4())
    print(f"  BQ run_id: {RUN_ID[:8]}...")

    tickers = sorted([t for t in all_daily_scores.keys() if t not in EXCLUDE_TICKERS])
    if TICKER_SUBSET:
        tickers = [t for t in tickers if t in TICKER_SUBSET]
    print(f"  {len(tickers)} tickers with scores\n")

    all_results = {m: [] for m in active_modes}
    n_tickers = len(tickers)

    for t_idx, ticker in enumerate(tickers):
        print(f"\n{'='*70}")
        print(f"  [{t_idx+1}/{n_tickers}] {ticker} (SHORT L5-2 MULTI-RES v2, Fold {FOLD_ID})")
        print(f"{'='*70}")
        try:
            df_ohlcv = load_ticker_ohlcv(ticker)
        except Exception as e:
            print(f"  Could not load OHLCV: {e}")
            continue
        daily_scores = all_daily_scores.get(ticker, {})
        if not daily_scores:
            print(f"  No daily scores")
            continue
        opt_start_int = int(OPT_START.replace('-', ''))
        opt_end_int   = int(OPT_END.replace('-', ''))
        test_start_int = int(TEST_START.replace('-', ''))
        test_end_int   = int(TEST_END.replace('-', ''))
        all_date_ints = set()
        for h_scores in daily_scores.values():
            all_date_ints.update(h_scores.keys())
        n_opt_scores  = sum(1 for d in all_date_ints if opt_start_int <= d <= opt_end_int)
        n_test_scores = sum(1 for d in all_date_ints if test_start_int <= d <= test_end_int)
        print(f"  Daily scores: {len(all_date_ints)} dates, {n_opt_scores} in OPT, {n_test_scores} in TEST")
        ohlcv_opt  = precompute_ohlcv_arrays(df_ohlcv, OPT_START, OPT_END)
        ohlcv_test = precompute_ohlcv_arrays(df_ohlcv, TEST_START, TEST_END)
        if ohlcv_opt is None:
            print(f"  No OHLCV data in OPT window")
            del df_ohlcv; gc.collect()
            continue

        if 'volscaled' in active_modes:
            vs_opt = compute_vol_scales(df_ohlcv, ohlcv_opt['dates'])
            ohlcv_opt['vol_scales'] = vs_opt
            if ohlcv_test is not None:
                vs_test = compute_vol_scales(df_ohlcv, ohlcv_test['dates'])
                ohlcv_test['vol_scales'] = vs_test

        for mode in active_modes:
            TPSL_MODE = mode
            mode_sfx = MODE_SUFFIX.get(mode, '')
            print(f"\n  [{mode.upper()}] Grid search...")

            if mode == 'ensemble':
                opt_results = grid_search_ensemble(ohlcv_opt, daily_scores, ticker)
            else:
                opt_results = grid_search(ohlcv_opt, daily_scores, ticker, mode)

            survivors = [r for r in opt_results if r['Avg_PnL'] > 0]

            if mode == 'ensemble':
                nc = len(ENSEMBLE_TPSL_PAIRS)
                k_tested = len(ENSEMBLE_PRESETS) * len(PROB_THRESHOLDS) * nc
            elif mode == 'grace':
                nc = len(GRACE_CONFIGS)
                k_tested = len(SCORE_HORIZONS) * len(PROB_THRESHOLDS) * nc
            elif mode == 'limit_entry':
                nc = len(LIMIT_ENTRY_CONFIGS)
                k_tested = len(SCORE_HORIZONS) * len(PROB_THRESHOLDS) * nc
            elif mode == 'volscaled':
                nc = len(VOLSCALED_BASE_PAIRS)
                k_tested = len(SCORE_HORIZONS) * len(PROB_THRESHOLDS) * nc
            else:
                nc = len(TPSL_PAIRS['static'])
                k_tested = len(SCORE_HORIZONS) * len(PROB_THRESHOLDS) * nc

            print(f"    K={k_tested} grid pts -> {len(opt_results)} with trades -> {len(survivors)} positive")
            if not survivors:
                print(f"    No positive configs")
                continue
            best = select_robust_params(survivors, mode)
            if best is None:
                _rk = 'ROAC' if mode == 'limit_entry' else 'Avg_PnL'
                best = max(survivors, key=lambda r: r.get(_rk, r['Avg_PnL']))
            best.pop('_raw_pnl', None)
            for r in survivors:
                r.pop('_raw_pnl', None)
            val_a = eval_combo(ohlcv_opt, daily_scores, best, mode)
            val_b = eval_combo(ohlcv_test, daily_scores, best, mode)
            if mode == 'grace':
                cfg_str = (f"TP={best['tp_pct']:.0%} SL={best['sl_pct']:.0%} "
                           f"grace={best.get('grace_minutes',0)}m")
            elif mode == 'limit_entry':
                cfg_str = (f"TP={best['tp_pct']:.0%} SL={best['sl_pct']:.0%} "
                           f"rally={best.get('dip_pct',0):.1%} wait={best.get('max_wait',0)}m")
            elif mode == 'volscaled':
                cfg_str = (f"baseTP={best['tp_pct']:.0%} baseSL={best['sl_pct']:.0%} "
                           f"vol[{VOL_SCALE_MIN:.1f}-{VOL_SCALE_MAX:.1f}]")
            elif mode == 'ensemble':
                cfg_str = (f"TP={best['tp_pct']:.0%} SL={best['sl_pct']:.0%} "
                           f"preset={best.get('ensemble_preset','?')}")
            else:
                cfg_str = f"TP={best['tp_pct']:.0%} SL={best['sl_pct']:.0%}"
            h_label = best.get('horizon_label', '30m')
            print(f"    OPT:  AvgPnL={best['Avg_PnL']:.2%}  WR={best['Win_Rate']:.1%}  "
                  f"N={best['Count']}  H={h_label}  Prob>{best['prob_thresh']:.2f}  {cfg_str}")
            tag_a = "PASS" if val_a['Avg_PnL'] > 0 else "FAIL"
            print(f"    VAL:  AvgPnL={val_a['Avg_PnL']:.2%}  WR={val_a['Win_Rate']:.1%}  "
                  f"N={val_a['Count']}  [{tag_a}]")
            tag_b = "PASS" if val_b['ROAC'] > 0 else "FAIL"
            fill_str = f"  Fill={val_b['Fill_Rate']:.0%}" if val_b['Fill_Rate'] < 1.0 else ""
            print(f"    TEST: ROAC={val_b['ROAC']:.2%}  AvgPnL={val_b['Avg_PnL']:.2%}  WR={val_b['Win_Rate']:.1%}  "
                  f"N={val_b['Count']}/{val_b['N_Signals']}{fill_str}  [{tag_b}]")
            result_row = {
                'Ticker': ticker,
                'score_horizon': best.get('score_horizon', 1),
                'horizon_label': best.get('horizon_label', '30m'),
                'entry_time': ENTRY_TIME,
                'prob_thresh': best['prob_thresh'],
                'tp_pct': best.get('tp_pct', 0),
                'sl_pct': best.get('sl_pct', 0),
                'grace_minutes': best.get('grace_minutes', 0),
                'dip_pct': best.get('dip_pct', 0),
                'max_wait': best.get('max_wait', 0),
                'ensemble_preset': best.get('ensemble_preset', ''),
                'Avg_PnL': best['Avg_PnL'],
                'Win_Rate': best['Win_Rate'],
                'Freq': best.get('Freq', 0),
                'Count': best['Count'],
                'val_a_avg_pnl': val_a['Avg_PnL'],
                'val_a_win_rate': val_a['Win_Rate'],
                'val_a_freq': val_a.get('Freq', 0),
                'val_a_count': val_a['Count'],
                'survived_a': val_a['Avg_PnL'] > 0,
                'val_b_avg_pnl': val_b['ROAC'],
                'val_b_avg_pnl_filled': val_b['Avg_PnL'],
                'val_b_win_rate': val_b['Win_Rate'],
                'val_b_freq': val_b.get('Freq', 0),
                'val_b_count': val_b['Count'],
                'val_b_n_signals': val_b['N_Signals'],
                'val_b_fill_rate': val_b['Fill_Rate'],
                'survived_b': val_b['ROAC'] > 0,
            }
            all_results[mode].append(result_row)
            bq_experiment = f"{EXPERIMENT}_{h_label}{mode_sfx}"
            save_row_to_bq(result_row, RUN_ID, bq_experiment, mode,
                            OPT_START, OPT_END,
                            OPT_START, OPT_END,
                            TEST_START, TEST_END)
        del df_ohlcv, ohlcv_opt, ohlcv_test
        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

    # ── Per-fold summary ──
    print(f"\n\n{'='*W}")
    print(f"  SUMMARY — Fold {FOLD_ID} (Dual Transformer Backbone)")
    print(f"  model_version: {MODEL_VERSION}")
    print(f"  Active modes: {', '.join(m.upper() for m in active_modes)}")
    print(f"  Score: min {SCORE_MINUTE} (6:15 AM)  Entry: min {ENTRY_MINUTE} (6:25 AM)")
    print(f"{'='*W}")

    for mode in active_modes:
        results = all_results[mode]
        if not results:
            print(f"\n  {mode.upper()}: No results")
            continue
        df = pd.DataFrame(results)
        n_total = len(df)
        n_pass_a = df['survived_a'].sum()
        n_pass_b = df['survived_b'].sum()
        print(f"\n  {mode.upper()}: {n_total} tickers -> {n_pass_a} PASS VAL -> {n_pass_b} PASS TEST")
        if n_total > 0:
            extra_hdr = ""
            if mode == 'ensemble':
                extra_hdr = " Preset"
            print(f"\n  {'Ticker':<7} {'H':>10} {'Prob':>5} {'TP':>5} {'SL':>5} "
                  f"{'OPT PnL':>8} {'VAL PnL':>8} {'TEST ROAC':>10} {'TEST PnL':>9} {'Fill':>5} "
                  f"{'TEST WR':>8} {'TEST N':>7} {'Status':>8}{extra_hdr}")
            print(f"  {'─'*105}")
            for _, r in df.sort_values('val_b_avg_pnl', ascending=False).iterrows():
                status = "DEPLOY" if r['survived_b'] and r['val_b_avg_pnl'] >= DEPLOY_PNL_THRESHOLD else \
                         "pass" if r['survived_b'] else "fail"
                extra = ""
                if mode == 'grace':
                    extra = f"  g={int(r.get('grace_minutes', 0))}m"
                elif mode == 'limit_entry':
                    extra = f"  r={r.get('dip_pct', 0):.1%} w={int(r.get('max_wait', 0))}m"
                elif mode == 'volscaled':
                    extra = f"  vol[{VOL_SCALE_MIN:.1f}-{VOL_SCALE_MAX:.1f}]"
                elif mode == 'ensemble':
                    extra = f"  {r.get('ensemble_preset', '?')}"
                h_lbl = r.get('horizon_label', '30m')
                fill_pct = r.get('val_b_fill_rate', 1.0)
                fill_str = f"{fill_pct:>4.0%}" if fill_pct < 1.0 else f"{'100%':>5}"
                print(f"  {r['Ticker']:<7} {h_lbl:>10} {r['prob_thresh']:>5.2f} "
                      f"{r['tp_pct']:>4.0%} {r['sl_pct']:>4.0%} "
                      f"{r['Avg_PnL']:>+7.2%} {r['val_a_avg_pnl']:>+7.2%} "
                      f"{r['val_b_avg_pnl']:>+9.2%} {r.get('val_b_avg_pnl_filled', r['val_b_avg_pnl']):>+8.2%} {fill_str} "
                      f"{r['val_b_win_rate']:>7.1%} "
                      f"{int(r['val_b_count']):>7} {status:>8}{extra}")

    deploy_rows = []
    for mode in active_modes:
        for r in all_results[mode]:
            if r['survived_b'] and r['val_b_avg_pnl'] >= DEPLOY_PNL_THRESHOLD:
                deploy_rows.append({**r, 'mode': mode})

    print(f"\n\n{'='*W}")
    print(f"  DEPLOYABLE TICKERS — Fold {FOLD_ID} (TEST ROAC >= {DEPLOY_PNL_THRESHOLD:.1%})")
    print(f"{'='*W}")
    if deploy_rows:
        deploy_df = pd.DataFrame(deploy_rows).sort_values('val_b_avg_pnl', ascending=False)
        print(f"  {len(deploy_df)} configs across {deploy_df['Ticker'].nunique()} tickers:\n")
        for _, r in deploy_df.iterrows():
            extra = ""
            if r['mode'] == 'ensemble':
                extra = f"  preset={r.get('ensemble_preset', '?')}"
            elif r['mode'] == 'volscaled':
                extra = f"  vol[{VOL_SCALE_MIN:.1f}-{VOL_SCALE_MAX:.1f}]"
            fill_info = f"  Fill={r.get('val_b_fill_rate', 1.0):.0%}" if r.get('val_b_fill_rate', 1.0) < 1.0 else ""
            print(f"    {r['Ticker']:<7} {r['mode']:<13} H={r.get('horizon_label','30m')}  "
                  f"ROAC={r['val_b_avg_pnl']:>+.2%}  "
                  f"WR={r['val_b_win_rate']:.0%}  N={int(r['val_b_count'])}/{int(r.get('val_b_n_signals', r['val_b_count']))}{fill_info}  "
                  f"Prob>{r['prob_thresh']:.2f}  TP={r['tp_pct']:.0%}  SL={r['sl_pct']:.0%}{extra}")
    else:
        print("  None")
    grand_deploy_rows.extend([{**r, 'fold': FOLD_ID} for r in deploy_rows])

    all_test_data = []
    for mode in active_modes:
        for r in all_results[mode]:
            all_test_data.append({'Ticker': r['Ticker'], 'mode': mode,
                                  'test_pnl': r['val_b_avg_pnl']})

    if all_test_data:
        test_df = pd.DataFrame(all_test_data)
        best_per_ticker = test_df.loc[test_df.groupby('Ticker')['test_pnl'].idxmax()]
        best_per_ticker = best_per_ticker.sort_values('test_pnl', ascending=True)
        fig, ax = plt.subplots(figsize=(12, max(6, len(best_per_ticker) * 0.35)))
        colors = ['green' if v >= DEPLOY_PNL_THRESHOLD else 'steelblue' if v >= 0 else 'crimson'
                  for v in best_per_ticker['test_pnl']]
        bars = ax.barh(range(len(best_per_ticker)), best_per_ticker['test_pnl'].values * 100,
                       color=colors, edgecolor='white', linewidth=0.5)
        ax.set_yticks(range(len(best_per_ticker)))
        ax.set_yticklabels([f"{r['Ticker']} ({r['mode'][:4]})" for _, r in best_per_ticker.iterrows()],
                           fontsize=8)
        ax.set_xlabel('TEST ROAC (%)')
        modes_str = '+'.join(m.upper() for m in active_modes)
        ax.set_title(f'SHORT L5-2 MULTI-RES v2 Fold {FOLD_ID} — TEST Performance ({modes_str})\n'
                     f'Green: >= {DEPLOY_PNL_THRESHOLD:.1%} (deployable)  |  Blue: positive  |  Red: negative')
        ax.axvline(0, color='black', linewidth=0.8)
        ax.axvline(DEPLOY_PNL_THRESHOLD * 100, color='green', linewidth=0.8, linestyle='--', alpha=0.5)
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        chart_path = os.path.join(MODEL_DIR, f'short_L5_2_multires_v2_fold{FOLD_ID}_optimizer.png')
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        print(f"\n  Chart saved: {chart_path}")
        if _IN_COLAB:
            plt.show()
        plt.close()

    grand_total = sum(len(v) for v in all_results.values())
    print(f"\n{'='*W}")
    print(f"  FOLD {FOLD_ID} COMPLETE — {grand_total} results, {len(deploy_rows)} deployable")
    print(f"  BQ: model_version='{MODEL_VERSION}', run_id={RUN_ID[:8]}...")
    print(f"{'='*W}")

    del all_daily_scores, all_results
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


# ==============================================================================
# GRAND SUMMARY (all folds)
# ==============================================================================
print(f"\n\n{'#'*W}")
print(f"  GRAND SUMMARY — SHORT L5-2 MULTI-RES v2 Fold 10")
print(f"{'#'*W}")
if grand_deploy_rows:
    gdf = pd.DataFrame(grand_deploy_rows).sort_values(['fold', 'val_b_avg_pnl'], ascending=[True, False])
    for fid in sorted(gdf['fold'].unique()):
        fold_df = gdf[gdf['fold'] == fid]
        print(f"\n  Fold {fid}: {len(fold_df)} deployable configs across {fold_df['Ticker'].nunique()} tickers")
        for _, r in fold_df.iterrows():
            print(f"    {r['Ticker']:<7} {r['mode']:<13} H={r.get('horizon_label','30m')}  "
                  f"ROAC={r['val_b_avg_pnl']:>+.2%}  "
                  f"Prob>{r['prob_thresh']:.2f}  TP={r['tp_pct']:.0%}  SL={r['sl_pct']:.0%}")
    print(f"\n  Total: {len(gdf)} deployable configs across {gdf['Ticker'].nunique()} unique tickers")
else:
    print("  No deployable configs across any fold")
print(f"{'#'*W}")
