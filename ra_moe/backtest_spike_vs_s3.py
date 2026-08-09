# %% ── Spike Transformer vs S3 LONG — Portfolio Backtest Comparison ──────────
#
# Head-to-head comparison of Spike Transformer (fold 4) vs S3 LONG on the
# same test period (Jun 1 - Aug 8, 2026). Both strategies scored and
# simulated with identical OHLCV data, slippage, and trade mechanics.
#
# Spike Transformer: model inference on fold_4_test.npz at entry_minute (375).
#   Configs from BQ (model_version = 'spike_transformer_fold4').
#   Modes: static, grace, limit_entry (best per ticker from optimizer).
#
# S3 LONG: CSV-scored (GCS Prob_Spike CSVs). Configs from BQ with full S3
#   family-agreement gate. All limit_entry mode.
#
# NOTE: CSV scores may have partial date coverage (esp. post Jul 3), so S3
#   trade count will be lower. Spike Transformer scores every test day.
#
# GPU recommended (TF model inference). Colab-ready.

import os, io, gc, ctypes, warnings
import numpy as np, pandas as pd, pytz
import tensorflow as tf
from tensorflow.keras import layers, Model
from google.cloud import storage, bigquery
from numba import jit
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

def _release_memory():
    gc.collect()
    try: ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception: pass


# ==============================================================================
# CONFIG
# ==============================================================================
TEST_START     = '2026-06-01'
TEST_END       = '2026-08-08'
ENTRY_MINUTE   = 375
EXIT_TIME_MIN  = 780
SLIPPAGE_PCT   = 0.0015
SLIPPAGE_RT    = 2 * SLIPPAGE_PCT
STARTING_CAPITAL = 200_000
DEPLOY_THRESHOLD = 0.018

GCS_BUCKET       = 'time-series-bucket-ys'
GCS_INFER_DIR    = 'Data/Inference_v2'
GCS_DATA_PREFIX  = 'Algos/RA_MoE/real_data'
GCS_MODEL_PREFIX = 'Algos/RA_MoE/real_models'
RESULTS_DIR      = 'Algos/Results_scored_20260703'
RESULTS_FALLBACK = 'Algos/Results_all_v2'
BQ_TABLE_ID      = 'ys-time-series.scoring_logs.wf_optimization_results'

DATA_DIR  = '/tmp/ra_moe_real_data'
MODEL_DIR = '/tmp/ra_moe_models'
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

PST = pytz.timezone('America/Los_Angeles')
TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}
EXCLUDE_TICKERS = {'SPY', 'VIX', 'TLT', 'IWM', 'XLE', 'ITA', 'BITF', 'MIGI', 'SMLR'}

# Spike Transformer architecture (must match training)
D_MODEL = 64; N_HEADS = 4; D_FF = 128; N_TRANSFORMER_BLOCKS = 2
DROPOUT = 0.1; FILM_HIDDEN = 64; TIME_STEPS = 128
TARGET_HORIZONS = [15, 30, 60]; N_HORIZONS = 3
ASL_GAMMA_NEG = 5.0; ASL_GAMMA_POS = 0.1; ASL_CLIP = 0.10
FOLD_ID = 4
SCORE_HORIZONS = [0, 1, 2]
HORIZON_LABELS = {0: '15m', 1: '30m', 2: '60m'}
HORIZON_FROM_LABEL = {'15m': 0, '30m': 1, '60m': 2}

# S3 gate constants
S3_MODEL_VERSIONS = ('robust_wf_v12', 'robust_wf_v12_stitched')
S3_MIN_SEEDS     = 3
S3_MAX_VA_OPT    = 3.0
S3_EXCLUDE       = {'LUNR'}
S3_GRP_KEYS = ['ticker', 'experiment', 'tpsl_mode', 'entry_minute',
               'prob_thresh', 'tp_pct', 'sl_pct',
               'tp_mult', 'sl_mult', 'trail_start_mult', 'trail_end_mult',
               'grace_minutes', 'dip_pct', 'max_wait_minutes', 'vol_filter_pct']

gcs_client = storage.Client()
bq_client  = bigquery.Client()
W = 120

N_ASSET_FEATURES  = None
N_REGIME_FEATURES = None


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
    for blob in bucket.list_blobs(prefix=gcs_prefix):
        fname = blob.name.split('/')[-1]
        blob.download_to_filename(os.path.join(local_dir, fname))
    return weights_path


# ==============================================================================
# OHLCV LOADING
# ==============================================================================
_master_cache = {}

def _load_parquet_with_dt_index(gcs_uri, tz_out='America/Los_Angeles'):
    df = pd.read_parquet(gcs_uri)
    df = df.reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    if dt_cols:
        df['Datetime'] = pd.to_datetime(df[dt_cols[0]], utc=True)
    else:
        num_col = next((c for c in df.columns
                       if c in ('Datetime', 'Date', 'Time', 'Timestamp', 'Level_0')),
                      df.columns[0])
        df['Datetime'] = pd.to_datetime(df[num_col], unit='ms', utc=True)
    df = df.set_index('Datetime')
    df = df[df.index.notna()].sort_index()
    df = df[~df.index.duplicated(keep='first')]
    df.index = df.index.tz_convert(tz_out)
    std_map = {'open': 'Open', 'high': 'High', 'low': 'Low',
               'close': 'Close', 'volume': 'Volume'}
    lower_map = {k.lower(): v for k, v in std_map.items()}
    df.columns = [lower_map.get(str(c).strip().lower(), str(c).strip())
                  for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated(keep='first')].copy()
    return df

def load_ticker_ohlcv(ticker):
    if ticker in _master_cache:
        return _master_cache[ticker]
    clean = TICKER_FILE_MAP.get(ticker, ticker)
    for suffix in ['_master_inference', '_1m_full_historical_data']:
        uri = f"gs://{GCS_BUCKET}/{GCS_INFER_DIR}/{clean}{suffix}.parquet"
        try:
            df = _load_parquet_with_dt_index(uri)
            _master_cache[ticker] = df
            return df
        except Exception:
            continue
    return None

def heal_data(df):
    df = df.resample('1min').asfreq()
    for c in ['Open', 'High', 'Low', 'Close']:
        if c in df.columns:
            df[c] = df[c].ffill()
    for c in ['Volume']:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    return df

def precompute_ohlcv_arrays(df_ohlcv, start_date, end_date):
    eval_start = pd.to_datetime(start_date).tz_localize('America/Los_Angeles')
    eval_end   = pd.to_datetime(end_date).tz_localize('America/Los_Angeles') + timedelta(days=1)
    df = df_ohlcv[(df_ohlcv.index >= eval_start) & (df_ohlcv.index < eval_end)]
    if len(df) == 0:
        return None
    dates = (df.index.year * 10000 + df.index.month * 100 + df.index.day).values.astype(np.int64)
    times = (df.index.hour * 60 + df.index.minute).values.astype(np.int64)
    return {
        'prices': df['Close'].values.astype(np.float64).copy(),
        'highs':  df['High'].values.astype(np.float64).copy(),
        'lows':   df['Low'].values.astype(np.float64).copy(),
        'dates':  dates,
        'times':  times,
    }


# ==============================================================================
# SPIKE TRANSFORMER — MODEL LAYERS (must match training)
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
        x = x + self.drop(self.attn(self.norm1(x), stats, training=training),
                          training=training)
        x = x + self.ffn(self.norm2(x), training=training)
        return x

class TransformerBackbone(layers.Layer):
    def __init__(self, d_model, n_heads, d_ff, n_blocks, seq_len,
                 dropout=0.1, **kwargs):
        super().__init__(**kwargs)
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
            layers.Dense(hidden_dim, activation='relu')])
        self.gamma_proj = layers.Dense(d_model)
        self.beta_proj = layers.Dense(d_model)

    def call(self, x, regime_features, training=False):
        h = self.mlp(regime_features, training=training)
        gamma = 1.0 + self.gamma_proj(h)
        beta = self.beta_proj(h)
        return gamma * x + beta

class SpikeTransformer(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.backbone = TransformerBackbone(
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
# NUMBA TRADE SIMULATION ENGINES (LONG)
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
            current_entry_ptr += 1; continue
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
            outcome = (prices[n_prices-1] - entry_price) / entry_price
            exit_idx = n_prices
        trades[trade_count] = outcome
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
            current_entry_ptr += 1; continue
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
            outcome = (prices[n_prices-1] - entry_price) / entry_price
            exit_idx = n_prices
        trades[trade_count] = outcome
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
def run_trade_sim_limit_entry(entry_indices, prices, highs, lows, date_ints,
                              time_mins, tp, sl, exit_time, dip_pct,
                              max_wait_minutes):
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
            current_entry_ptr += 1; continue
        signal_price = prices[idx]
        limit_price  = signal_price * (1.0 - dip_pct)
        signal_time  = time_mins[idx]

        fill_idx   = -1
        fill_price = 0.0
        for k in range(idx + 1, n_prices):
            if date_ints[k] != entry_date: break
            if time_mins[k] - signal_time > max_wait_minutes: break
            if lows[k] <= limit_price:
                fill_price = limit_price; fill_idx = k; break

        if fill_idx < 0:
            current_entry_ptr += 1
            last_traded_date = entry_date; continue

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
            outcome = (prices[n_prices-1] - fill_price) / fill_price
            exit_idx = n_prices
        trades[trade_count] = outcome
        entry_dates[trade_count] = entry_date
        trade_count += 1
        last_traded_date = entry_date
        next_ptr = current_entry_ptr + 1
        while next_ptr < n_entries:
            if entry_indices[next_ptr] > exit_idx: break
            next_ptr += 1
        current_entry_ptr = next_ptr

    return trades[:trade_count], entry_dates[:trade_count]


def dispatch_trade_sim(indices, ohlcv, cfg):
    prices = ohlcv['prices']; highs = ohlcv['highs']; lows = ohlcv['lows']
    dates = ohlcv['dates']; times = ohlcv['times'].astype(np.int64)
    mode = cfg['mode']
    if mode == 'grace':
        pnl, d = run_trade_sim_grace(
            indices, prices, highs, lows, dates, times,
            cfg['tp_pct'], cfg['sl_pct'], EXIT_TIME_MIN, cfg['grace_minutes'])
    elif mode == 'limit_entry':
        pnl, d = run_trade_sim_limit_entry(
            indices, prices, highs, lows, dates, times,
            cfg['tp_pct'], cfg['sl_pct'], EXIT_TIME_MIN,
            cfg['dip_pct'], cfg['max_wait_minutes'])
    else:
        pnl, d = run_trade_sim(
            indices, prices, highs, lows, dates, times,
            cfg['tp_pct'], cfg['sl_pct'], EXIT_TIME_MIN)
    if len(pnl) > 0:
        pnl = pnl - SLIPPAGE_RT
    return pnl, d


# ==============================================================================
# S3 CSV SCORE HELPERS
# ==============================================================================

def assign_family(exp):
    base = exp.split('_')[0]
    if not base.isdigit():
        return 'other'
    num = int(base)
    if num == 152: return '152'
    if num == 160: return '160'
    if 170 <= num <= 175: return '170'
    if 180 <= num <= 185: return '180'
    if 200 <= num <= 205: return '200'
    if 220 <= num <= 225: return '220'
    return 'other'


def load_seed_scores(ticker, experiment_base):
    clean = TICKER_FILE_MAP.get(ticker, ticker)
    bucket = gcs_client.bucket(GCS_BUCKET)
    all_results = {}
    for results_dir in [RESULTS_FALLBACK, RESULTS_DIR]:
        prefix = f"{results_dir}/Universal_Exp{experiment_base}/{clean}/Seed_"
        blobs = list(bucket.list_blobs(prefix=prefix))
        for blob in blobs:
            name = blob.name.split('/')[-1]
            if not name.startswith('Seed_') or not name.endswith('_Results.csv'):
                continue
            seed_id = int(name.replace('Seed_', '').replace('_Results.csv', ''))
            d = pd.read_csv(io.BytesIO(blob.download_as_bytes()),
                            index_col=0, parse_dates=True)
            if d.index.tz is None:
                d.index = d.index.tz_localize('UTC')
            d.index = d.index.tz_convert(PST)
            d = d.sort_index()
            if seed_id in all_results:
                old = all_results[seed_id]
                combined = pd.concat([old, d])
                combined = combined[~combined.index.duplicated(keep='last')]
                all_results[seed_id] = combined.sort_index()
            else:
                all_results[seed_id] = d
    return all_results


# ==============================================================================
# ██ STEP 1: BQ CONFIG LOADING
# ==============================================================================
print("=" * W)
print(f"  SPIKE TRANSFORMER vs S3 LONG — PORTFOLIO BACKTEST COMPARISON")
print(f"  Test period: {TEST_START} → {TEST_END}")
print(f"  Entry: minute {ENTRY_MINUTE} (6:15 AM PST), Exit: {EXIT_TIME_MIN}")
print(f"  Slippage: {SLIPPAGE_RT:.2%} RT")
print("=" * W)

# ── 1A: Spike Transformer configs from BQ ──
print(f"\n  STEP 1A: Loading Spike Transformer configs from BQ...")
st_q = f"""
SELECT ticker, experiment, tpsl_mode, prob_thresh, tp_pct, sl_pct,
       grace_minutes, dip_pct, max_wait_minutes,
       opt_avg_pnl, val_a_avg_pnl, val_b_avg_pnl, val_b_win_rate, val_b_count
FROM `{BQ_TABLE_ID}`
WHERE model_version = 'spike_transformer_fold4'
  AND survived_a = TRUE
"""
st_bq = bq_client.query(st_q).to_dataframe()
print(f"    {len(st_bq)} BQ rows for spike_transformer_fold4")

st_deploy = st_bq[st_bq['val_b_avg_pnl'] >= DEPLOY_THRESHOLD].copy()
print(f"    {len(st_deploy)} deployable (val_b >= {DEPLOY_THRESHOLD:.1%})")

st_configs = {}
if len(st_deploy) > 0:
    best_st = st_deploy.loc[st_deploy.groupby('ticker')['val_b_avg_pnl'].idxmax()]
    for _, row in best_st.iterrows():
        exp = row['experiment']
        h_part = exp.replace('spike_transformer_fold4_', '').split('_')[0]
        horizon_idx = HORIZON_FROM_LABEL.get(h_part, 1)
        mode = row['tpsl_mode']
        if pd.isna(mode) or mode == '':
            if '_GRACE' in exp: mode = 'grace'
            elif '_LIMIT' in exp: mode = 'limit_entry'
            else: mode = 'static'
        st_configs[row['ticker']] = {
            'horizon_idx': horizon_idx,
            'horizon_label': h_part,
            'prob_thresh': float(row['prob_thresh']),
            'tp_pct': float(row['tp_pct']),
            'sl_pct': float(row['sl_pct']),
            'mode': mode,
            'grace_minutes': int(row['grace_minutes']) if pd.notna(row['grace_minutes']) else 0,
            'dip_pct': float(row['dip_pct']) if pd.notna(row['dip_pct']) else 0,
            'max_wait_minutes': int(row['max_wait_minutes']) if pd.notna(row['max_wait_minutes']) else 0,
            'val_b_avg_pnl': float(row['val_b_avg_pnl']),
            'experiment': exp,
        }

st_tickers = sorted(st_configs.keys())
print(f"    {len(st_tickers)} tickers selected (best config per ticker):")
for tk in st_tickers:
    c = st_configs[tk]
    print(f"      {tk:<7} {c['mode']:<13} H={c['horizon_label']}  "
          f"Prob>{c['prob_thresh']:.2f}  TP={c['tp_pct']:.0%}  SL={c['sl_pct']:.0%}  "
          f"TEST={c['val_b_avg_pnl']:+.2%}")


# ── 1B: S3 LONG configs from BQ (full gate) ──
print(f"\n  STEP 1B: Loading S3 LONG configs from BQ (family-agreement gate)...")
s3_q = f"""
SELECT ticker, experiment, seed, tpsl_mode, entry_minute,
       prob_thresh, tp_pct, sl_pct, tp_mult, sl_mult,
       trail_start_mult, trail_end_mult, grace_minutes,
       dip_pct, max_wait_minutes, vol_filter_pct,
       val_a_avg_pnl, val_a_win_rate, val_a_count, val_a_freq,
       val_b_avg_pnl, opt_avg_pnl, is_ensemble
FROM `{BQ_TABLE_ID}`
WHERE model_version IN {str(S3_MODEL_VERSIONS)}
  AND survived_a = TRUE
  AND val_a_count >= 3
  AND val_a_avg_pnl > 0
"""
raw_s3 = bq_client.query(s3_q).to_dataframe()
print(f"    {len(raw_s3)} raw BQ rows")

raw_s3 = raw_s3[raw_s3['opt_avg_pnl'] > 0].copy()
raw_s3 = raw_s3[raw_s3['val_a_avg_pnl'] / raw_s3['opt_avg_pnl'] <= S3_MAX_VA_OPT]
indiv = raw_s3[raw_s3['is_ensemble'] == False].copy()
for k in S3_GRP_KEYS:
    if k not in indiv.columns: indiv[k] = 0
    else: indiv[k] = indiv[k].fillna(0)

sa = indiv.groupby(S3_GRP_KEYS).agg(
    n_seeds=('seed', 'nunique'),
    n_pos_seeds_18=('val_a_avg_pnl', lambda x: (x >= DEPLOY_THRESHOLD).sum()),
    mean_va_pnl=('val_a_avg_pnl', 'mean'),
    mean_va_wr=('val_a_win_rate', 'mean'),
    vb_pnl=('val_b_avg_pnl', 'mean'),
).reset_index()
sa['agree_18'] = (sa['n_pos_seeds_18'] / sa['n_seeds']).clip(upper=1.0)

sa = sa[sa['n_seeds'] >= S3_MIN_SEEDS].copy()
sa_limit = sa[sa['tpsl_mode'] == 'limit_entry'].copy()
sa_limit['score'] = (sa_limit['mean_va_pnl'] ** 2) * sa_limit['agree_18']
sa_limit['exp_fam'] = sa_limit['experiment'].apply(assign_family)
sa_fam = sa_limit[sa_limit['exp_fam'] != 'other'].copy()

best_per_fam = sa_fam.loc[sa_fam.groupby(['ticker', 'exp_fam'])['score'].idxmax()]
best_per_fam = best_per_fam[best_per_fam['mean_va_pnl'] >= DEPLOY_THRESHOLD]

agreement = {}
for tk in best_per_fam['ticker'].unique():
    if tk in S3_EXCLUDE: continue
    tk_rows = best_per_fam[best_per_fam['ticker'] == tk]
    n_fams = len(tk_rows)
    n_pos = int((tk_rows['vb_pnl'] > 0).sum())
    agreement[tk] = {'n_fams': n_fams, 'n_pos': n_pos,
                     'majority_pos': n_pos > n_fams / 2 and n_fams >= 2}

s3_gate_tickers = sorted([tk for tk, ag in agreement.items() if ag['majority_pos']])
print(f"    Family agreement: {len(s3_gate_tickers)} tickers pass S3 gate")

sa_s3 = sa_limit[sa_limit['ticker'].isin(s3_gate_tickers)].copy()
sa_s3['score'] = (sa_s3['mean_va_pnl'] ** 2) * sa_s3['agree_18']
best_s3 = sa_s3.loc[sa_s3.groupby('ticker')['score'].idxmax()]
best_s3 = best_s3[best_s3['mean_va_pnl'] >= DEPLOY_THRESHOLD]

s3_configs = {}
for _, row in best_s3.iterrows():
    tk = row['ticker']
    s3_configs[tk] = {
        'prob_thresh': float(row['prob_thresh']),
        'tp_pct': float(row['tp_pct']),
        'sl_pct': float(row['sl_pct']),
        'mode': 'limit_entry',
        'dip_pct': float(row['dip_pct']),
        'max_wait_minutes': float(row['max_wait_minutes']),
        'grace_minutes': 0,
        'experiment': row['experiment'],
        'entry_minute': int(row['entry_minute']) if pd.notna(row['entry_minute']) else ENTRY_MINUTE,
        'val_a_pnl': float(row['mean_va_pnl']),
    }

s3_tickers = sorted(s3_configs.keys())
print(f"    {len(s3_tickers)} S3 tickers: {', '.join(s3_tickers)}")
for tk in s3_tickers:
    c = s3_configs[tk]
    print(f"      {tk:<7} {c['experiment']:<16} Prob>{c['prob_thresh']:.2f}  "
          f"TP={c['tp_pct']:.0%}  SL={c['sl_pct']:.0%}  "
          f"dip={c['dip_pct']:.1%}  wait={c['max_wait_minutes']:.0f}m  "
          f"valA={c['val_a_pnl']:+.2%}")

del raw_s3, indiv, sa, sa_limit, sa_fam, best_per_fam, sa_s3, best_s3, st_bq, st_deploy
_release_memory()


# ==============================================================================
# ██ STEP 2: SPIKE TRANSFORMER SCORING (model inference)
# ==============================================================================
print(f"\n{'='*W}")
print(f"  STEP 2: SCORING — Spike Transformer fold {FOLD_ID}")
print(f"{'='*W}")

meta_path = ensure_data('meta.npz')
meta = np.load(meta_path, allow_pickle=True)
ticker_names = meta['ticker_names']
print(f"  {len(ticker_names)} tickers in meta")

probe_path = ensure_data(f'fold_{FOLD_ID}_val.npz')
probe = np.load(probe_path)
N_ASSET_FEATURES = probe['X_seq'].shape[-1]
N_REGIME_FEATURES = probe['X_regime'].shape[-1]
probe.close()
print(f"  N_ASSET_FEATURES={N_ASSET_FEATURES}, N_REGIME_FEATURES={N_REGIME_FEATURES}")

print("  Building model...")
model = SpikeTransformer(name='spike_transformer')
dummy_seq = tf.zeros((1, TIME_STEPS, N_ASSET_FEATURES))
dummy_reg = tf.zeros((1, N_REGIME_FEATURES))
model({'seq_input': dummy_seq, 'regime_input': dummy_reg})
weights_path = ensure_model(FOLD_ID)
model.load_weights(weights_path)
print(f"  Model loaded: {model.count_params():,} params")

# Score test npz at entry_minute only
path = ensure_data(f'fold_{FOLD_ID}_test.npz')
print(f"  Loading test npz...")
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
print(f"  Filtering {n_total:,} → {n_entry:,} entry-minute windows")

X_seq     = X_seq[entry_mask]
X_reg     = X_reg[entry_mask]
X_id      = X_id[entry_mask]
date_ints_filt = date_ints[entry_mask]
del minutes, entry_mask, date_ints; gc.collect()

print(f"  Running inference on {n_entry:,} samples...")
preds = model.predict(
    {'seq_input': X_seq.astype(np.float32),
     'regime_input': X_reg.astype(np.float32)},
    batch_size=512, verbose=0)
del X_seq, X_reg; gc.collect()

st_daily_scores = {}
for i in range(n_entry):
    tid = int(X_id[i])
    d_int = int(date_ints_filt[i])
    if tid >= len(ticker_names): continue
    tname = str(ticker_names[tid])
    if tname in EXCLUDE_TICKERS: continue
    if tname not in st_daily_scores:
        st_daily_scores[tname] = {h: {} for h in SCORE_HORIZONS}
    for h in SCORE_HORIZONS:
        st_daily_scores[tname][h][d_int] = float(preds[i, h])

del preds, X_id, date_ints_filt; gc.collect()
n_scores = sum(len(v) for t_scores in st_daily_scores.values() for v in t_scores.values())
print(f"  Scoring complete: {len(st_daily_scores)} tickers, {n_scores} daily scores")

del model
tf.keras.backend.clear_session()
gc.collect()
try: ctypes.CDLL("libc.so.6").malloc_trim(0)
except Exception: pass


# ==============================================================================
# ██ STEP 3: S3 CSV SCORES
# ==============================================================================
print(f"\n{'='*W}")
print(f"  STEP 3: Loading S3 CSV scores from GCS")
print(f"{'='*W}")

s3_daily_probs = {}
for tk in s3_tickers:
    cfg = s3_configs[tk]
    exp = cfg['experiment']
    exp_base = exp.replace('_LIMIT', '').replace('_GRACE', '')
    entry_min = cfg.get('entry_minute', ENTRY_MINUTE)

    seeds_dict = load_seed_scores(tk, exp_base)
    if not seeds_dict:
        print(f"    {tk}: no CSV scores for {exp_base}")
        continue

    avg_probs = None
    n_seeds = 0
    for seed_id, sdf in seeds_dict.items():
        if 'Prob_Spike' not in sdf.columns:
            col = [c for c in sdf.columns if 'prob' in c.lower() or 'spike' in c.lower()]
            if not col: continue
            sdf = sdf.rename(columns={col[0]: 'Prob_Spike'})
        bar_mins = sdf.index.hour * 60 + sdf.index.minute
        entry_bars = sdf[bar_mins == entry_min]
        if entry_bars.empty: continue
        daily_prob = entry_bars.groupby(entry_bars.index.date)['Prob_Spike'].first()
        if avg_probs is None:
            avg_probs = daily_prob.copy()
        else:
            avg_probs = avg_probs.add(daily_prob, fill_value=0)
        n_seeds += 1

    if n_seeds > 0 and avg_probs is not None:
        prob_dict = {}
        for dt, prob in (avg_probs / n_seeds).items():
            d_int = dt.year * 10000 + dt.month * 100 + dt.day
            prob_dict[d_int] = float(prob)
        s3_daily_probs[tk] = prob_dict
        test_start_int = int(TEST_START.replace('-', ''))
        test_end_int = int(TEST_END.replace('-', ''))
        n_test = sum(1 for d in prob_dict if test_start_int <= d <= test_end_int)
        print(f"    {tk}: {len(prob_dict)} total dates, {n_test} in test, {n_seeds} seeds ({exp_base})")
    else:
        print(f"    {tk}: no valid seed scores")

_release_memory()


# ==============================================================================
# ██ STEP 4: OHLCV LOADING + TRADE SIMULATION
# ==============================================================================
print(f"\n{'='*W}")
print(f"  STEP 4: Loading OHLCV + running trade simulations")
print(f"{'='*W}")

all_tickers = sorted(set(st_tickers) | set(s3_tickers))
print(f"  {len(all_tickers)} unique tickers ({len(st_tickers)} ST, {len(s3_tickers)} S3)")

ohlcv_cache = {}
for ticker in all_tickers:
    df = load_ticker_ohlcv(ticker)
    if df is None:
        print(f"    {ticker}: FAILED to load OHLCV")
        continue
    df = heal_data(df.sort_index())
    ohlcv = precompute_ohlcv_arrays(df, TEST_START, TEST_END)
    if ohlcv is not None:
        ohlcv_cache[ticker] = ohlcv
        n_days = len(np.unique(ohlcv['dates']))
        print(f"    {ticker}: {len(df):>8,} bars, {n_days} test days")
    del df; gc.collect()

_release_memory()

# ── Run Spike Transformer trades ──
print(f"\n  Running Spike Transformer trade simulation...")
st_trades = []
for ticker in st_tickers:
    cfg = st_configs[ticker]
    ohlcv = ohlcv_cache.get(ticker)
    if ohlcv is None: continue

    t_scores = st_daily_scores.get(ticker, {})
    h_scores = t_scores.get(cfg['horizon_idx'], {})
    if not h_scores:
        print(f"    {ticker}: no scores for horizon {cfg['horizon_label']}")
        continue

    dates = ohlcv['dates']; times = ohlcv['times']
    probs = np.zeros(len(dates), dtype=np.float64)
    for i in range(len(dates)):
        if times[i] == ENTRY_MINUTE:
            probs[i] = h_scores.get(int(dates[i]), 0.0)

    mask = (probs >= cfg['prob_thresh']) & (times == ENTRY_MINUTE)
    indices = np.where(mask)[0].astype(np.int64)
    if len(indices) == 0:
        print(f"    {ticker}: 0 signals (prob>={cfg['prob_thresh']:.2f})")
        continue

    pnl, trade_dates = dispatch_trade_sim(indices, ohlcv, cfg)
    for j in range(len(pnl)):
        d_int = int(trade_dates[j])
        d_str = f"{d_int // 10000}-{(d_int % 10000) // 100:02d}-{d_int % 100:02d}"
        st_trades.append({'date': pd.Timestamp(d_str).date(), 'ticker': ticker,
                          'pnl': float(pnl[j]), 'mode': cfg['mode']})

    print(f"    {ticker}: {len(pnl)} trades, avg={np.mean(pnl):+.2%}, "
          f"WR={(pnl > 0).mean():.0%} ({cfg['mode']}, H={cfg['horizon_label']})")

# ── Run S3 LONG trades ──
print(f"\n  Running S3 LONG trade simulation...")
s3_trades = []
for ticker in s3_tickers:
    cfg = s3_configs[ticker]
    ohlcv = ohlcv_cache.get(ticker)
    if ohlcv is None: continue

    probs_dict = s3_daily_probs.get(ticker, {})
    if not probs_dict:
        print(f"    {ticker}: no daily probs")
        continue

    dates = ohlcv['dates']; times = ohlcv['times']
    probs = np.zeros(len(dates), dtype=np.float64)
    for i in range(len(dates)):
        if times[i] == ENTRY_MINUTE:
            probs[i] = probs_dict.get(int(dates[i]), 0.0)

    mask = (probs >= cfg['prob_thresh']) & (times == ENTRY_MINUTE)
    indices = np.where(mask)[0].astype(np.int64)
    if len(indices) == 0:
        print(f"    {ticker}: 0 signals (prob>={cfg['prob_thresh']:.2f})")
        continue

    pnl, trade_dates = dispatch_trade_sim(indices, ohlcv, cfg)
    for j in range(len(pnl)):
        d_int = int(trade_dates[j])
        d_str = f"{d_int // 10000}-{(d_int % 10000) // 100:02d}-{d_int % 100:02d}"
        s3_trades.append({'date': pd.Timestamp(d_str).date(), 'ticker': ticker,
                          'pnl': float(pnl[j])})

    print(f"    {ticker}: {len(pnl)} trades, avg={np.mean(pnl):+.2%}, "
          f"WR={(pnl > 0).mean():.0%}")

del ohlcv_cache, st_daily_scores, s3_daily_probs
_release_memory()


# ==============================================================================
# ██ STEP 5: COMPARISON RESULTS
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  STEP 5: PORTFOLIO COMPARISON — {TEST_START} → {TEST_END}")
print(f"{'='*W}")

def compute_stats(trades_list, label):
    if not trades_list:
        return {'label': label, 'n': 0, 'total': 0, 'avg': 0, 'wr': 0,
                'dd': 0, 'sharpe': 0, 'per_mo': 0, 'n_tickers': 0, 'monthly': {}}
    df = pd.DataFrame(trades_list)
    df['month'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m')
    cum = df.sort_values('date')['pnl'].cumsum()
    max_dd = float((cum - cum.cummax()).min())
    n_months = max(1, df['month'].nunique())
    daily_pnl = df.groupby('date')['pnl'].sum()
    sharpe = (daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)
              if daily_pnl.std() > 0 else 0)
    return {
        'label': label,
        'n': len(df), 'total': float(df['pnl'].sum()),
        'avg': float(df['pnl'].mean()),
        'wr': float((df['pnl'] > 0).mean()),
        'dd': max_dd, 'sharpe': float(sharpe),
        'per_mo': len(df) / n_months,
        'n_tickers': df['ticker'].nunique(),
        'monthly': df.groupby('month').agg(
            n=('pnl', 'count'), total=('pnl', 'sum'),
            avg=('pnl', 'mean'), wr=('pnl', lambda x: (x > 0).mean())
        ).to_dict('index'),
        'trades_df': df,
    }


STRATEGIES = {'Spike Transformer': st_trades, 'S3 LONG': s3_trades}
stats = {s: compute_stats(trades, s) for s, trades in STRATEGIES.items()}
STRAT_ORDER = ['Spike Transformer', 'S3 LONG']
COLORS = {'Spike Transformer': '#FF6600', 'S3 LONG': '#2196F3'}
LINESTYLES = {'Spike Transformer': '-', 'S3 LONG': '-'}


# ── TABLE 1: HEAD-TO-HEAD ──
print(f"\n  TABLE 1: HEAD-TO-HEAD COMPARISON")
print(f"\n  {'Strategy':<22} {'#Tk':>3} {'Trd':>4} {'AvgPnL':>7} {'TotPnL':>8} "
      f"{'WR':>4} {'MaxDD':>7} {'Sharpe':>6} {'T/mo':>5}")
print(f"  {'─'*22} {'─'*3} {'─'*4} {'─'*7} {'─'*8} "
      f"{'─'*4} {'─'*7} {'─'*6} {'─'*5}")
for s in STRAT_ORDER:
    st = stats[s]
    if st['n'] == 0:
        print(f"  {s:<22} — no trades")
        continue
    print(f"  {s:<22} {st['n_tickers']:>3} {st['n']:>4} {st['avg']:>+6.2%} "
          f"{st['total']:>+7.1%} {st['wr']:>3.0%} {st['dd']:>+6.1%} "
          f"{st['sharpe']:>5.1f} {st['per_mo']:>5.1f}")


# ── TABLE 2: MONTHLY BREAKDOWN ──
all_months = sorted(set(m for s in STRAT_ORDER for m in stats[s].get('monthly', {}).keys()))
month_labels = {m: pd.Timestamp(m + '-01').strftime('%b') for m in all_months}

print(f"\n  TABLE 2: MONTHLY BREAKDOWN")
print(f"\n  {'Strategy':<22}", end="")
for m in all_months:
    print(f"  {month_labels[m]:>22}", end="")
print()
print(f"  {'─'*22}", end="")
for _ in all_months:
    print(f"  {'─'*22}", end="")
print()

for s in STRAT_ORDER:
    st = stats[s]
    if st['n'] == 0: continue
    print(f"  {s:<22}", end="")
    for m in all_months:
        ms = st.get('monthly', {}).get(m)
        if ms:
            print(f"  {ms['avg']:>+5.2%}({ms['n']:>3}t,{ms['wr']:>4.0%}wr)", end="")
        else:
            print(f"  {'—':>22}", end="")
    print()


# ── TABLE 3: PER-TICKER BREAKDOWN ──
print(f"\n  TABLE 3: PER-TICKER BREAKDOWN")
col_w = 22

all_bt_tickers = sorted(set(
    t for s in STRAT_ORDER
    for t in (stats[s]['trades_df']['ticker'].unique() if stats[s]['n'] > 0 else [])
))

print(f"\n  {'Ticker':<7}", end="")
for s in STRAT_ORDER:
    print(f"  {s:<{col_w}}", end="")
print()
print(f"  {'─'*7}", end="")
for _ in STRAT_ORDER:
    print(f"  {'─'*col_w}", end="")
print()

for ticker in all_bt_tickers:
    print(f"  {ticker:<7}", end="")
    for s in STRAT_ORDER:
        st = stats[s]
        if st['n'] == 0 or 'trades_df' not in st:
            print(f"  {'—':<{col_w}}", end=""); continue
        tt = st['trades_df'][st['trades_df']['ticker'] == ticker]
        if len(tt) > 0:
            cell = f"{tt['pnl'].mean():>+5.2%} {len(tt):>3}t {(tt['pnl']>0).mean():>3.0%}w"
            print(f"  {cell:<{col_w}}", end="")
        else:
            print(f"  {'—':<{col_w}}", end="")
    print()


# ── TABLE 4: PORTFOLIO SIMULATION ──
print(f"\n  TABLE 4: PORTFOLIO SIMULATION (${STARTING_CAPITAL:,} starting capital)")

equity_curves = {}
for s in STRAT_ORDER:
    st = stats[s]
    if st['n'] == 0: continue
    df_t = st['trades_df'].sort_values('date')
    equity = STARTING_CAPITAL
    curve = [{'date': pd.Timestamp(TEST_START).date(), 'equity': equity}]
    peak = equity
    max_dd_pct = 0; max_dd_dollar = 0

    for _, row in df_t.iterrows():
        alloc = equity / max(1, st['n_tickers'])
        pnl_dollar = alloc * row['pnl']
        equity += pnl_dollar
        curve.append({'date': row['date'], 'equity': equity})
        peak = max(peak, equity)
        dd = (equity - peak) / peak
        if dd < max_dd_pct:
            max_dd_pct = dd
            max_dd_dollar = equity - peak

    total_return = (equity - STARTING_CAPITAL) / STARTING_CAPITAL
    daily_pnl = df_t.groupby('date')['pnl'].sum()
    sharpe = (daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)
              if daily_pnl.std() > 0 else 0)

    print(f"\n  {s}")
    print(f"    Final equity: ${equity:,.0f} ({total_return:+.1%})")
    print(f"    Max drawdown: ${max_dd_dollar:,.0f} ({max_dd_pct:+.1%})")
    print(f"    Sharpe (approx): {sharpe:.2f}")
    print(f"    Trades/month: {st['per_mo']:.1f}")

    equity_curves[s] = pd.DataFrame(curve)


# ── OVERLAP & COMBINATION ANALYSIS ──
st_set = set(st_tickers)
s3_set = set(s3_tickers)
shared_tickers = sorted(st_set & s3_set)
st_only_set = sorted(st_set - s3_set)
s3_only_set = sorted(s3_set - st_set)
union_set = sorted(st_set | s3_set)

print(f"\n  TICKER OVERLAP:")
print(f"    Spike Transformer only ({len(st_only_set)}): {', '.join(st_only_set) or '—'}")
print(f"    S3 LONG only ({len(s3_only_set)}):           {', '.join(s3_only_set) or '—'}")
print(f"    Shared ({len(shared_tickers)}):               {', '.join(shared_tickers) or '—'}")
print(f"    Union: {len(union_set)} unique tickers")

st_df = pd.DataFrame(st_trades) if st_trades else pd.DataFrame(columns=['date','ticker','pnl'])
s3_df = pd.DataFrame(s3_trades) if s3_trades else pd.DataFrame(columns=['date','ticker','pnl'])

if shared_tickers and len(st_df) > 0 and len(s3_df) > 0:
    print(f"\n  SIGNAL OVERLAP ON SHARED TICKERS:")
    print(f"    {'Ticker':<7} {'ST#':>3} {'S3#':>3} {'Both':>4} {'ST-only':>7} {'S3-only':>7} {'Jaccard':>7}")
    print(f"    {'—'*7} {'—'*3} {'—'*3} {'—'*4} {'—'*7} {'—'*7} {'—'*7}")

    total_both = total_st_o = total_s3_o = 0
    for tk in shared_tickers:
        st_dates = set(st_df[st_df['ticker'] == tk]['date'])
        s3_dates = set(s3_df[s3_df['ticker'] == tk]['date'])
        both_d = st_dates & s3_dates
        st_o = st_dates - s3_dates
        s3_o = s3_dates - st_dates
        union_d = st_dates | s3_dates
        jaccard = len(both_d) / max(1, len(union_d))
        total_both += len(both_d); total_st_o += len(st_o); total_s3_o += len(s3_o)
        print(f"    {tk:<7} {len(st_dates):>3} {len(s3_dates):>3} {len(both_d):>4} "
              f"{len(st_o):>7} {len(s3_o):>7} {jaccard:>6.0%}")

    total_union = total_both + total_st_o + total_s3_o
    print(f"    {'TOTAL':<7} {'':>3} {'':>3} {total_both:>4} "
          f"{total_st_o:>7} {total_s3_o:>7} "
          f"{total_both / max(1, total_union):>6.0%}")

if len(st_df) > 0 and len(s3_df) > 0:
    st_daily = st_df.groupby('date')['pnl'].sum()
    s3_daily = s3_df.groupby('date')['pnl'].sum()
    common_dates = st_daily.index.intersection(s3_daily.index)

    print(f"\n  DAILY PnL CORRELATION:")
    st_only_days = len(set(st_daily.index) - set(s3_daily.index))
    s3_only_days = len(set(s3_daily.index) - set(st_daily.index))
    print(f"    Trading days: ST={len(st_daily)}, S3={len(s3_daily)}, "
          f"common={len(common_dates)}, ST-only={st_only_days}, S3-only={s3_only_days}")

    if len(common_dates) >= 3:
        corr = st_daily.loc[common_dates].corr(s3_daily.loc[common_dates])
        print(f"    Correlation on {len(common_dates)} common days: {corr:.3f}")
        if corr < 0.3:
            print(f"    → LOW correlation — strong diversification potential")
        elif corr < 0.6:
            print(f"    → MODERATE correlation — some diversification benefit")
        else:
            print(f"    → HIGH correlation — limited diversification benefit")

# ── COMBINED STRATEGY SIMULATIONS ──
print(f"\n  COMBINED STRATEGIES:")

st_trade_keys = {(t['ticker'], t['date']): t for t in st_trades}
s3_trade_keys = {(t['ticker'], t['date']): t for t in s3_trades}

# UNION: all unique trades, ST config wins conflicts
union_trades = list(st_trade_keys.values())
for key, trade in s3_trade_keys.items():
    if key not in st_trade_keys:
        union_trades.append(trade)

# ADDITIVE: all ST + S3 trades on S3-only tickers only
s3_only_set_s = set(s3_only_set)
additive_trades = list(st_trades)
for trade in s3_trades:
    if trade['ticker'] in s3_only_set_s:
        additive_trades.append(trade)

# INTERSECTION: only where both fire same (ticker, date)
intersection_trades = [st_trade_keys[k] for k in st_trade_keys if k in s3_trade_keys]

combo_names = ['Spike Transformer', 'S3 LONG',
               'UNION (ST priority)', 'ADDITIVE (ST+S3-only)', 'INTERSECTION']
combo_trades = {
    'Spike Transformer': st_trades,
    'S3 LONG': s3_trades,
    'UNION (ST priority)': union_trades,
    'ADDITIVE (ST+S3-only)': additive_trades,
    'INTERSECTION': intersection_trades,
}
combo_colors = {
    'Spike Transformer': '#FF6600', 'S3 LONG': '#2196F3',
    'UNION (ST priority)': '#4CAF50', 'ADDITIVE (ST+S3-only)': '#9C27B0',
    'INTERSECTION': '#FF9800',
}

print(f"\n  {'Strategy':<28} {'#Tk':>3} {'Trd':>4} {'AvgPnL':>7} {'TotPnL':>8} "
      f"{'WR':>4} {'MaxDD':>7} {'Sharpe':>6}")
print(f"  {'—'*28} {'—'*3} {'—'*4} {'—'*7} {'—'*8} {'—'*4} {'—'*7} {'—'*6}")

combo_stats = {}
for name in combo_names:
    s = compute_stats(combo_trades[name], name)
    combo_stats[name] = s
    if s['n'] == 0:
        print(f"  {name:<28} — no trades")
        continue
    print(f"  {name:<28} {s['n_tickers']:>3} {s['n']:>4} {s['avg']:>+6.2%} "
          f"{s['total']:>+7.1%} {s['wr']:>3.0%} {s['dd']:>+6.1%} {s['sharpe']:>5.1f}")

# PnL by signal source
st_only_trades = [st_trade_keys[k] for k in st_trade_keys if k not in s3_trade_keys]
s3_only_trades_list = [s3_trade_keys[k] for k in s3_trade_keys if k not in st_trade_keys]
both_trades_st = [st_trade_keys[k] for k in st_trade_keys if k in s3_trade_keys]
both_trades_s3 = [s3_trade_keys[k] for k in s3_trade_keys if k in st_trade_keys]

print(f"\n  PnL BY SIGNAL SOURCE:")
for label, trades_list in [('ST-only signals', st_only_trades),
                            ('S3-only signals', s3_only_trades_list),
                            ('Both-fire (ST cfg)', both_trades_st),
                            ('Both-fire (S3 cfg)', both_trades_s3)]:
    if not trades_list:
        print(f"    {label:<22} — no trades")
        continue
    pnls = [t['pnl'] for t in trades_list]
    avg = np.mean(pnls)
    wr = sum(1 for p in pnls if p > 0) / len(pnls)
    print(f"    {label:<22} {len(pnls):>4} trades  avg={avg:>+.2%}  WR={wr:.0%}")

# Portfolio simulation for combined strategies
print(f"\n  PORTFOLIO SIMULATION (${STARTING_CAPITAL:,} start):")
combo_equity_curves = {}
for name in combo_names:
    s = combo_stats[name]
    if s['n'] == 0: continue
    df_t = s['trades_df'].sort_values('date')
    equity = float(STARTING_CAPITAL)
    curve = [{'date': pd.Timestamp(TEST_START).date(), 'equity': equity}]
    peak = equity
    max_dd_pct = 0.0
    for _, row in df_t.iterrows():
        alloc = equity / max(1, s['n_tickers'])
        equity += alloc * row['pnl']
        curve.append({'date': row['date'], 'equity': equity})
        peak = max(peak, equity)
        dd = (equity - peak) / peak
        if dd < max_dd_pct: max_dd_pct = dd
    total_ret = (equity - STARTING_CAPITAL) / STARTING_CAPITAL
    print(f"    {name:<28} → ${equity:,.0f} ({total_ret:>+.1%})  MaxDD={max_dd_pct:>+.1%}")
    combo_equity_curves[name] = pd.DataFrame(curve)

# 50/50 capital split
half_cap = STARTING_CAPITAL / 2
combined_50 = 0.0
print(f"\n  50/50 CAPITAL SPLIT (${half_cap:,.0f} each):")
for name in ['Spike Transformer', 'S3 LONG']:
    s = combo_stats[name]
    if s['n'] == 0: continue
    final_eq = half_cap * (1 + s['total'])
    combined_50 += final_eq
    print(f"    {name}: ${final_eq:,.0f}")
if combined_50 > 0:
    print(f"    Combined: ${combined_50:,.0f} ({(combined_50 - STARTING_CAPITAL) / STARTING_CAPITAL:+.1%})")

# Recommendation
print(f"\n  RECOMMENDATION:")
best_avg = max(combo_names, key=lambda n: combo_stats[n].get('avg', -999))
best_sharpe = max(combo_names, key=lambda n: combo_stats[n].get('sharpe', -999))
print(f"    Best by avg PnL: {best_avg} ({combo_stats[best_avg]['avg']:+.2%})")
print(f"    Best by Sharpe:  {best_sharpe} ({combo_stats[best_sharpe]['sharpe']:.1f})")
add_s = combo_stats.get('ADDITIVE (ST+S3-only)', {})
st_s = combo_stats.get('Spike Transformer', {})
if add_s.get('n', 0) > st_s.get('n', 0):
    extra = add_s['n'] - st_s['n']
    print(f"    ADDITIVE adds {extra} S3-only-ticker trades to ST portfolio")
    if add_s['avg'] >= st_s['avg']:
        print(f"    → S3-only tickers ADD value ({add_s['avg']:+.2%} >= {st_s['avg']:+.2%})")
    else:
        print(f"    → S3-only tickers DILUTE ({add_s['avg']:+.2%} < {st_s['avg']:+.2%})")

print(f"\n  CAVEATS:")
print(f"    • S3 uses CSV scores with partial date coverage (~10% of test days)")
print(f"    • Spike Transformer uses live model inference (full coverage)")
print(f"    • Overlap analysis limited by S3's sparse CSV availability")
print(f"    • For production, S3 needs live TF scoring for fair comparison")
print(f"    • Slippage: {SLIPPAGE_RT:.2%} RT applied to both strategies")


# ==============================================================================
# ██ STEP 6: CHARTS
# ==============================================================================
print(f"\n  Generating charts...")

# ── Chart 1: Equity Curves ──
fig, ax1 = plt.subplots(figsize=(14, 6))
for s in STRAT_ORDER:
    if s not in equity_curves: continue
    curve_df = equity_curves[s]
    dates = pd.to_datetime(curve_df['date'])
    equity = curve_df['equity'] / 1000
    ax1.plot(dates, equity, label=s, color=COLORS[s], linewidth=2.5,
             linestyle=LINESTYLES[s])

ax1.axhline(y=STARTING_CAPITAL/1000, color='gray', linestyle='--', alpha=0.4)
ax1.set_ylabel('Equity ($K)', fontsize=12)
ax1.set_title(f'Spike Transformer vs S3 LONG — Equity Curves (${STARTING_CAPITAL//1000}K start)\n'
              f'Test: {TEST_START} → {TEST_END}', fontsize=14, fontweight='bold')
ax1.legend(loc='upper left', fontsize=11, framealpha=0.9)
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.tight_layout()
plt.savefig('/tmp/chart1_spike_vs_s3_equity.png', dpi=150, bbox_inches='tight')
if _IN_COLAB: plt.show()
plt.close()


# ── Chart 2: Cumulative PnL (%) ──
fig, ax2 = plt.subplots(figsize=(14, 6))
for s in STRAT_ORDER:
    st = stats[s]
    if st['n'] == 0 or 'trades_df' not in st: continue
    df_sorted = st['trades_df'].sort_values('date')
    cum_pnl = df_sorted['pnl'].cumsum() * 100
    dates = pd.to_datetime(df_sorted['date'].values)
    ax2.plot(dates, cum_pnl.values, label=s, color=COLORS[s], linewidth=2)

ax2.axhline(y=0, color='gray', linewidth=0.5)
ax2.set_ylabel('Cumulative PnL (%)', fontsize=12)
ax2.set_title('Cumulative PnL — Trade by Trade', fontsize=14, fontweight='bold')
ax2.legend(loc='upper left', fontsize=11, framealpha=0.9)
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.tight_layout()
plt.savefig('/tmp/chart2_spike_vs_s3_cumulative.png', dpi=150, bbox_inches='tight')
if _IN_COLAB: plt.show()
plt.close()


# ── Chart 3: Drawdowns ──
fig, ax3 = plt.subplots(figsize=(14, 4))
for s in STRAT_ORDER:
    st = stats[s]
    if st['n'] == 0 or 'trades_df' not in st: continue
    df_sorted = st['trades_df'].sort_values('date')
    cum_pnl = df_sorted['pnl'].cumsum()
    dd = (cum_pnl - cum_pnl.cummax()) * 100
    dates = pd.to_datetime(df_sorted['date'].values)
    ax3.fill_between(dates, dd.values, 0, alpha=0.15, color=COLORS[s])
    ax3.plot(dates, dd.values, label=s, color=COLORS[s], linewidth=1.5)

ax3.set_ylabel('Drawdown (%)', fontsize=12)
ax3.set_title('Strategy Drawdowns', fontsize=14, fontweight='bold')
ax3.legend(loc='lower left', fontsize=11, framealpha=0.9)
ax3.grid(True, alpha=0.3)
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.tight_layout()
plt.savefig('/tmp/chart3_spike_vs_s3_drawdowns.png', dpi=150, bbox_inches='tight')
if _IN_COLAB: plt.show()
plt.close()


# ── Chart 4: Monthly Avg PnL Bars ──
if all_months:
    fig, ax4 = plt.subplots(figsize=(10, 5))
    x = np.arange(len(all_months))
    bar_width = 0.35

    for i, s in enumerate(STRAT_ORDER):
        st = stats[s]
        if st['n'] == 0: continue
        monthly_avgs = []
        for m in all_months:
            ms = st.get('monthly', {}).get(m)
            monthly_avgs.append(ms['avg'] * 100 if ms else 0)
        bars = ax4.bar(x + i * bar_width, monthly_avgs, bar_width,
                       label=s, color=COLORS[s], alpha=0.85)
        for bar, val in zip(bars, monthly_avgs):
            ax4.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.05,
                     f'{val:.2f}%', ha='center', va='bottom', fontsize=9)

    ax4.set_xticks(x + bar_width / 2)
    ax4.set_xticklabels([month_labels[m] for m in all_months], fontsize=12)
    ax4.set_ylabel('Avg PnL per Trade (%)', fontsize=12)
    ax4.set_title('Monthly Average PnL Comparison', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=11, framealpha=0.9)
    ax4.axhline(y=0, color='gray', linewidth=0.5)
    ax4.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('/tmp/chart4_spike_vs_s3_monthly.png', dpi=150, bbox_inches='tight')
    if _IN_COLAB: plt.show()
    plt.close()


# ── Chart 5: Per-Ticker Heatmap ──
if all_bt_tickers:
    heatmap_data = np.full((len(all_bt_tickers), len(STRAT_ORDER)), np.nan)
    for j, s in enumerate(STRAT_ORDER):
        st = stats[s]
        if st['n'] == 0 or 'trades_df' not in st: continue
        for i, ticker in enumerate(all_bt_tickers):
            tt = st['trades_df'][st['trades_df']['ticker'] == ticker]
            if len(tt) > 0:
                heatmap_data[i, j] = tt['pnl'].mean() * 100

    fig, ax5 = plt.subplots(figsize=(8, max(6, len(all_bt_tickers) * 0.45)))
    im = ax5.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=-5, vmax=5)
    ax5.set_xticks(range(len(STRAT_ORDER)))
    ax5.set_xticklabels(['Spike Trans.', 'S3 LONG'], fontsize=11)
    ax5.set_yticks(range(len(all_bt_tickers)))
    ax5.set_yticklabels(all_bt_tickers, fontsize=10)
    for i in range(len(all_bt_tickers)):
        for j in range(len(STRAT_ORDER)):
            val = heatmap_data[i, j]
            if not np.isnan(val):
                ax5.text(j, i, f'{val:+.1f}%', ha='center', va='center',
                         fontsize=9, fontweight='bold',
                         color='white' if abs(val) > 3 else 'black')
    plt.colorbar(im, label='Avg PnL (%)', shrink=0.8)
    ax5.set_title('Per-Ticker Avg PnL — Spike Transformer vs S3 LONG',
                  fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/tmp/chart5_spike_vs_s3_heatmap.png', dpi=150, bbox_inches='tight')
    if _IN_COLAB: plt.show()
    plt.close()


# ── Chart 6: Summary Metrics Bars ──
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
active = [s for s in STRAT_ORDER if stats[s]['n'] > 0]
short_names = ['Spike Trans.', 'S3 LONG'][:len(active)]

metrics = [
    ('Total PnL (%)', [stats[s]['total'] * 100 for s in active]),
    ('Avg PnL (%)', [stats[s]['avg'] * 100 for s in active]),
    ('Max DD (%)', [stats[s]['dd'] * 100 for s in active]),
    ('Sharpe', [stats[s]['sharpe'] for s in active]),
]

for ax, (name, vals) in zip(axes, metrics):
    colors_list = [COLORS[s] for s in active]
    bars = ax.bar(range(len(vals)), vals, color=colors_list, alpha=0.85)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(short_names, fontsize=9, rotation=15, ha='right')
    ax.set_title(name, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    if 'DD' in name:
        ax.axhline(y=0, color='gray', linewidth=0.5)
    for bar, val in zip(bars, vals):
        y_pos = bar.get_height() + (0.2 if bar.get_height() >= 0 else -1.0)
        ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)

fig.suptitle(f'Spike Transformer vs S3 LONG — Summary ({TEST_START} → {TEST_END})',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/tmp/chart6_spike_vs_s3_summary.png', dpi=150, bbox_inches='tight')
if _IN_COLAB: plt.show()
plt.close()


# ── Chart 7: Combined Strategy Equity Curves ──
if combo_equity_curves:
    fig, ax7 = plt.subplots(figsize=(14, 6))
    for name in combo_names:
        if name not in combo_equity_curves: continue
        curve_df = combo_equity_curves[name]
        dates = pd.to_datetime(curve_df['date'])
        equity = curve_df['equity'] / 1000
        is_base = name in ('Spike Transformer', 'S3 LONG')
        ax7.plot(dates, equity, label=name, color=combo_colors.get(name, 'gray'),
                 linewidth=2.5 if is_base else 1.8,
                 linestyle='-' if is_base else '--')

    ax7.axhline(y=STARTING_CAPITAL/1000, color='gray', linestyle='--', alpha=0.4)
    ax7.set_ylabel('Equity ($K)', fontsize=12)
    ax7.set_title(f'Combined Strategy Equity Curves (${STARTING_CAPITAL//1000}K start)\n'
                  f'Test: {TEST_START} → {TEST_END}', fontsize=14, fontweight='bold')
    ax7.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax7.grid(True, alpha=0.3)
    ax7.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.tight_layout()
    plt.savefig('/tmp/chart7_combined_equity.png', dpi=150, bbox_inches='tight')
    if _IN_COLAB: plt.show()
    plt.close()


# ── Chart 8: PnL by Signal Source ──
source_data = []
for label, trades_list in [('ST-only', st_only_trades),
                            ('S3-only', s3_only_trades_list),
                            ('Both→ST', both_trades_st),
                            ('Both→S3', both_trades_s3)]:
    if trades_list:
        pnls = [t['pnl'] for t in trades_list]
        source_data.append({
            'source': label, 'avg_pnl': np.mean(pnls) * 100,
            'count': len(pnls), 'wr': sum(1 for p in pnls if p > 0) / len(pnls) * 100
        })

if source_data:
    fig, (ax8a, ax8b) = plt.subplots(1, 2, figsize=(12, 5))
    names_src = [d['source'] for d in source_data]
    avg_pnls = [d['avg_pnl'] for d in source_data]
    counts_src = [d['count'] for d in source_data]
    wrs_src = [d['wr'] for d in source_data]
    colors_src = ['#FF6600', '#2196F3', '#4CAF50', '#9C27B0'][:len(source_data)]

    bars1 = ax8a.bar(names_src, avg_pnls, color=colors_src, alpha=0.85)
    ax8a.set_ylabel('Avg PnL (%)', fontsize=12)
    ax8a.set_title('Avg PnL by Signal Source', fontsize=13, fontweight='bold')
    ax8a.axhline(y=0, color='gray', linewidth=0.5)
    ax8a.grid(True, alpha=0.3, axis='y')
    for bar, val, cnt in zip(bars1, avg_pnls, counts_src):
        ax8a.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                  f'{val:.2f}%\n({cnt}t)', ha='center', va='bottom', fontsize=9)

    bars2 = ax8b.bar(names_src, wrs_src, color=colors_src, alpha=0.85)
    ax8b.set_ylabel('Win Rate (%)', fontsize=12)
    ax8b.set_title('Win Rate by Signal Source', fontsize=13, fontweight='bold')
    ax8b.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50%')
    ax8b.grid(True, alpha=0.3, axis='y')
    ax8b.legend()
    for bar, val in zip(bars2, wrs_src):
        ax8b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                  f'{val:.0f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('/tmp/chart8_signal_source_pnl.png', dpi=150, bbox_inches='tight')
    if _IN_COLAB: plt.show()
    plt.close()


# ── Chart 9: Ticker Overlap Visualization ──
all_union_tickers = sorted(st_set | s3_set)
if all_union_tickers:
    fig, ax9 = plt.subplots(figsize=(10, max(5, len(all_union_tickers) * 0.4)))
    y_pos = list(range(len(all_union_tickers)))
    in_st = [1 if tk in st_set else 0 for tk in all_union_tickers]
    in_s3 = [1 if tk in s3_set else 0 for tk in all_union_tickers]

    bh = 0.35
    ax9.barh([y - bh/2 for y in y_pos], in_st,
             height=bh, color='#FF6600', alpha=0.85, label='Spike Transformer')
    ax9.barh([y + bh/2 for y in y_pos], in_s3,
             height=bh, color='#2196F3', alpha=0.85, label='S3 LONG')

    ax9.set_yticks(y_pos)
    ax9.set_yticklabels(all_union_tickers, fontsize=10)
    ax9.set_xticks([0, 1])
    ax9.set_xticklabels(['Not Selected', 'Selected'], fontsize=10)
    ax9.set_title('Ticker Selection Overlap — Spike Transformer vs S3 LONG',
                  fontsize=13, fontweight='bold')
    ax9.legend(loc='lower right', fontsize=11)
    ax9.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('/tmp/chart9_ticker_overlap.png', dpi=150, bbox_inches='tight')
    if _IN_COLAB: plt.show()
    plt.close()


# ── DONE ──
print()
print("=" * W)
print(f"  BACKTEST COMPLETE")
print(f"  Charts saved to /tmp/chart[1-9]_spike_vs_s3_*.png")
print(f"  Spike Transformer: {stats['Spike Transformer']['n']} trades across "
      f"{stats['Spike Transformer']['n_tickers']} tickers")
print(f"  S3 LONG:           {stats['S3 LONG']['n']} trades across "
      f"{stats['S3 LONG']['n_tickers']} tickers")
print("=" * W)
