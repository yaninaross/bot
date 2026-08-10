# %% ── ST SHORT Post-July Analysis ─────────────────────────────────────────────
#
# Diagnose which ST SHORT tickers lose money after July 15.
# Splits performance at BREAK_DATE and shows:
#   - Per-ticker PnL before vs after the break
#   - Weekly PnL timeline per ticker
#   - Per-ticker equity curves
#   - Trade-level detail for losing tickers
#
# GPU RECOMMENDED (ST model scoring). Colab-ready.

import os, gc, sys, time, warnings, ctypes
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np, pandas as pd, pytz

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization

try:
    import pandas_ta as ta
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas_ta",
                           "--no-deps", "--quiet"])
    import pandas_ta as ta

from numba import jit

import matplotlib
try:
    from google.colab import auth
    _IN_COLAB = True
    auth.authenticate_user()
except (ImportError, Exception):
    matplotlib.use('Agg')
    _IN_COLAB = False
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from google.cloud import storage as gcs_storage

# ==============================================================================
# CONFIG
# ==============================================================================
BUCKET_NAME = "time-series-bucket-ys"
INFERENCE_DIR = "Data/Inference_v2"

TEST_START = '2026-04-01'
TEST_END   = '2026-08-08'
BREAK_DATE = '2026-07-15'

SLIPPAGE_PCT = 0.0015
SLIPPAGE_RT  = 2 * SLIPPAGE_PCT
EXIT_TIME_MIN = 780

PST = pytz.timezone('America/Los_Angeles')
TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

MICRO_COLS = ['Spread', 'OBI', 'Volume_Delta', 'Trade_Count', 'Delta_Pct',
              'bsize', 'asize', 'Total_Vol']

# ── ST SHORT model config ──
ST_SCORE_MINUTE = 375
ST_ENTRY_MINUTE = 385
ST_D_MODEL = 64; ST_N_HEADS = 4; ST_D_FF = 128; ST_N_BLOCKS = 2
ST_DROPOUT = 0.1; ST_FILM_HIDDEN = 64; ST_TIME_STEPS = 128
ST_N_HORIZONS = 3
ST_ASL_GAMMA_NEG = 6.0; ST_ASL_GAMMA_POS = 0.1; ST_ASL_CLIP = 0.08

GCS_ST_DATA   = 'Algos/RA_MoE/short_data'
GCS_ST_MODELS = 'Algos/RA_MoE/short_models'
ST_FOLD_ID = 4
EXCLUDE_TICKERS = {'SPY', 'VIX', 'TLT', 'IWM', 'XLE', 'ITA', 'BITF', 'MIGI', 'SMLR'}

ST_CONFIGS = {
    'IREN':  {'horizon': 0, 'prob_thresh': 0.60, 'tp_pct': 0.08, 'sl_pct': 0.04,
              'mode': 'static'},
    'WOLF':  {'horizon': 2, 'prob_thresh': 0.70, 'tp_pct': 0.12, 'sl_pct': 0.06,
              'mode': 'static'},
    'AEHR':  {'horizon': 2, 'prob_thresh': 0.50, 'tp_pct': 0.20, 'sl_pct': 0.10,
              'mode': 'limit_entry', 'dip_pct': 0.04, 'max_wait': 90},
    'SOUN':  {'horizon': 0, 'prob_thresh': 0.60, 'tp_pct': 0.08, 'sl_pct': 0.04,
              'mode': 'grace', 'grace_minutes': 15},
    'OKLO':  {'horizon': 0, 'prob_thresh': 0.60, 'tp_pct': 0.12, 'sl_pct': 0.06,
              'mode': 'static'},
    'MVIS':  {'horizon': 2, 'prob_thresh': 0.60, 'tp_pct': 0.16, 'sl_pct': 0.08,
              'mode': 'grace', 'grace_minutes': 15},
    'VOYG':  {'horizon': 1, 'prob_thresh': 0.40, 'tp_pct': 0.06, 'sl_pct': 0.03,
              'mode': 'static'},
    'AVAV':  {'horizon': 0, 'prob_thresh': 0.30, 'tp_pct': 0.08, 'sl_pct': 0.04,
              'mode': 'static'},
    'CCJ':   {'horizon': 0, 'prob_thresh': 0.20, 'tp_pct': 0.08, 'sl_pct': 0.04,
              'mode': 'limit_entry', 'dip_pct': 0.01, 'max_wait': 30},
    'JOBY':  {'horizon': 0, 'prob_thresh': 0.60, 'tp_pct': 0.16, 'sl_pct': 0.08,
              'mode': 'limit_entry', 'dip_pct': 0.025, 'max_wait': 60},
    'SMR':   {'horizon': 0, 'prob_thresh': 0.50, 'tp_pct': 0.16, 'sl_pct': 0.08,
              'mode': 'limit_entry', 'dip_pct': 0.025, 'max_wait': 60},
}
ST_TICKERS = sorted(ST_CONFIGS.keys())

gcs_client = gcs_storage.Client()
W = 100


# ==============================================================================
# HELPERS
# ==============================================================================
def _release_memory():
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


_master_cache = {}


def load_master(ticker):
    if ticker in _master_cache:
        return _master_cache[ticker]
    clean = ticker.replace("I:", "").replace("X:", "")
    file_ticker = TICKER_FILE_MAP.get(clean, clean)
    uri = f"gs://{BUCKET_NAME}/{INFERENCE_DIR}/{file_ticker}_master_inference.parquet"
    try:
        df = pd.read_parquet(uri)
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
        df = df[~df.index.duplicated(keep='first')]
        std_map = {
            'Total_vol': 'Total_Vol', 'Trade_count': 'Trade_Count',
            'Volume_delta': 'Volume_Delta', 'Delta_pct': 'Delta_Pct',
            'Obi': 'OBI', 'Bsize': 'bsize', 'Asize': 'asize',
            'open': 'Open', 'high': 'High', 'low': 'Low',
            'close': 'Close', 'volume': 'Volume',
        }
        lower_map = {k.lower(): v for k, v in std_map.items()}
        df.columns = [lower_map.get(str(c).strip().lower(), str(c).strip())
                      for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated(keep='first')].copy()
        for c in MICRO_COLS:
            if c not in df.columns:
                df[c] = 0.0
        df = df[df.index.notna()].sort_index()
        _master_cache[ticker] = df
        return df
    except Exception as e:
        print(f"    load_master({ticker}): FAILED -- {e}")
        return None


def heal_data(df):
    df = df.resample('1min').asfreq()
    for c in ['Open', 'High', 'Low', 'Close']:
        if c in df.columns:
            df[c] = df[c].ffill()
    for c in ['Volume'] + MICRO_COLS:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    return df


def precompute_ohlcv_arrays(df, start_date, end_date):
    s = pd.to_datetime(start_date).tz_localize(PST)
    e = pd.to_datetime(end_date).tz_localize(PST) + timedelta(days=1)
    idx = df.index
    if str(idx.tz) != 'America/Los_Angeles':
        idx = idx.tz_convert(PST)
    mask = (idx >= s) & (idx < e)
    df_cut = df[mask]
    if len(df_cut) == 0:
        return None
    idx_cut = df_cut.index if str(df_cut.index.tz) == 'America/Los_Angeles' \
        else df_cut.index.tz_convert(PST)
    return {
        'prices': df_cut['Close'].values.astype(np.float64).copy(),
        'highs':  df_cut['High'].values.astype(np.float64).copy(),
        'lows':   df_cut['Low'].values.astype(np.float64).copy(),
        'dates':  (idx_cut.year * 10000 + idx_cut.month * 100 + idx_cut.day).values.astype(np.int64),
        'times':  (idx_cut.hour * 60 + idx_cut.minute).values.astype(np.int64),
    }


# ==============================================================================
# ST MODEL ARCHITECTURE
# ==============================================================================
class SeriesStationarization(layers.Layer):
    def call(self, x):
        mean = tf.reduce_mean(x, axis=1, keepdims=True)
        std  = tf.math.reduce_std(x, axis=1, keepdims=True) + 1e-5
        return (x - mean) / std, tf.concat([mean[:, 0, :], std[:, 0, :]], axis=-1)


class DeStationaryAttention(layers.Layer):
    def __init__(self, d_model, n_heads, dropout=0.1, **kw):
        super().__init__(**kw)
        self.d_model = d_model; self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = tf.sqrt(tf.cast(self.d_head, tf.float32))
        self.wq = Dense(d_model); self.wk = Dense(d_model)
        self.wv = Dense(d_model); self.wo = Dense(d_model)
        self.attn_drop = layers.Dropout(dropout)
        self.tau_proj = Dense(n_heads)
        self.sigma_proj = Dense(d_model)

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
        out = tf.reshape(tf.transpose(tf.matmul(attn, V), (0, 2, 1, 3)),
                         (B, T, self.d_model))
        sigma = tf.nn.softplus(self.sigma_proj(stats))
        return self.wo(out * sigma[:, tf.newaxis, :])


class TransformerBlock(layers.Layer):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, **kw):
        super().__init__(**kw)
        self.attn = DeStationaryAttention(d_model, n_heads, dropout)
        self.ffn = tf.keras.Sequential([
            Dense(d_ff, activation='gelu'), Dense(d_model), layers.Dropout(dropout)])
        self.norm1 = LayerNormalization()
        self.norm2 = LayerNormalization()
        self.drop = layers.Dropout(dropout)

    def call(self, x, stats, training=False):
        x = x + self.drop(
            self.attn(self.norm1(x), stats, training=training), training=training)
        return x + self.ffn(self.norm2(x), training=training)


class TransformerBackbone(layers.Layer):
    def __init__(self, d_model, n_heads, d_ff, n_blocks, seq_len,
                 dropout=0.1, **kw):
        super().__init__(**kw)
        self.stationarize = SeriesStationarization()
        self.input_proj = Dense(d_model)
        self.pos_embed = self.add_weight(
            name='pos_embed', shape=(1, seq_len, d_model),
            initializer='glorot_normal', trainable=True)
        self.input_drop = layers.Dropout(dropout)
        self.blocks = [TransformerBlock(d_model, n_heads, d_ff, dropout,
                                        name=f'block_{i}')
                       for i in range(n_blocks)]
        self.final_norm = LayerNormalization()

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
        return tf.matmul(tf.nn.softmax(scores, axis=-1), x)[:, 0, :]


class FiLMConditioner(layers.Layer):
    def __init__(self, d_model, hidden_dim=64, **kw):
        super().__init__(**kw)
        self.mlp = tf.keras.Sequential([
            Dense(hidden_dim, activation='relu'),
            Dense(hidden_dim, activation='relu')])
        self.gamma_proj = Dense(d_model)
        self.beta_proj = Dense(d_model)

    def call(self, x, regime_features, training=False):
        h = self.mlp(regime_features, training=training)
        return (1.0 + self.gamma_proj(h)) * x + self.beta_proj(h)


class SpikeTransformer(Model):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.backbone = TransformerBackbone(
            ST_D_MODEL, ST_N_HEADS, ST_D_FF, ST_N_BLOCKS,
            ST_TIME_STEPS, ST_DROPOUT, name='backbone')
        self.attn_pool = AttentionPooling(ST_D_MODEL, name='attn_pool')
        self.film = FiLMConditioner(ST_D_MODEL, ST_FILM_HIDDEN, name='film')
        self.drop = layers.Dropout(ST_DROPOUT)
        self.head = Dense(ST_N_HORIZONS, activation='sigmoid', name='spike_head')

    def call(self, inputs, training=False):
        h = self.backbone(inputs['seq_input'], training=training)
        h = self.attn_pool(h, training=training)
        h = self.film(h, inputs['regime_input'], training=training)
        return self.head(self.drop(h, training=training))

    def train_step(self, data):
        x, y, sw = tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            preds = self(x, training=True)
            loss = _asl_loss(y, preds, sw)
        self.optimizer.apply_gradients(
            zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))
        return {'loss': loss}

    def test_step(self, data):
        x, y, sw = tf.keras.utils.unpack_x_y_sample_weight(data)
        return {'loss': _asl_loss(y, self(x, training=False), sw)}


def _asl_loss(y_true, y_pred, sample_weight=None):
    p = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    pos_loss = -y_true * tf.pow(1.0 - p, ST_ASL_GAMMA_POS) * tf.math.log(p)
    p_m = tf.maximum(p - ST_ASL_CLIP, 0.0)
    neg_loss = -(1.0 - y_true) * tf.pow(p_m, ST_ASL_GAMMA_NEG) * tf.math.log(
        tf.clip_by_value(1.0 - p_m, 1e-7, 1.0))
    per_sample = tf.reduce_mean(pos_loss + neg_loss, axis=-1)
    if sample_weight is not None:
        per_sample = per_sample * sample_weight
    return tf.reduce_mean(per_sample)


# ==============================================================================
# ST SCORING — load model, score npz, build daily scores
# ==============================================================================
ST_DATA_DIR  = '/tmp/ra_moe_short_data'
ST_MODEL_DIR = '/tmp/ra_moe_short_models'
os.makedirs(ST_DATA_DIR, exist_ok=True)
os.makedirs(ST_MODEL_DIR, exist_ok=True)


def _ensure_st_data(filename):
    local_path = os.path.join(ST_DATA_DIR, filename)
    if os.path.exists(local_path):
        return local_path
    gcs_path = f'{GCS_ST_DATA}/{filename}'
    print(f"    Downloading {filename} from GCS...")
    gcs_client.bucket(BUCKET_NAME).blob(gcs_path).download_to_filename(local_path)
    return local_path


def _ensure_st_model(fold_id):
    local_dir = os.path.join(ST_MODEL_DIR, f'fold_{fold_id}')
    weights_path = os.path.join(local_dir, 'model.weights.h5')
    if os.path.exists(weights_path):
        return weights_path
    os.makedirs(local_dir, exist_ok=True)
    bucket = gcs_client.bucket(BUCKET_NAME)
    for blob in bucket.list_blobs(prefix=f'{GCS_ST_MODELS}/fold_{fold_id}'):
        fname = blob.name.split('/')[-1]
        blob.download_to_filename(os.path.join(local_dir, fname))
    return weights_path


def score_st_npz(model, split, ticker_names):
    path = _ensure_st_data(f'fold_{ST_FOLD_ID}_{split}.npz')
    data = np.load(path)
    X_seq = data['X_seq']; X_reg = data['X_regime']
    X_id = data['X_id'].flatten()
    date_ints = data['date_ints']; minutes = data['minutes']
    data.close()

    entry_mask = (minutes == ST_SCORE_MINUTE)
    X_seq = X_seq[entry_mask]; X_reg = X_reg[entry_mask]
    X_id = X_id[entry_mask]; date_ints = date_ints[entry_mask]
    n = len(X_id)
    print(f"    {split}: {n} score-minute samples")

    preds = model.predict(
        {'seq_input': X_seq.astype(np.float32),
         'regime_input': X_reg.astype(np.float32)},
        batch_size=512, verbose=0)
    del X_seq, X_reg; gc.collect()

    ticker_scores = {}
    for i in range(n):
        tid = int(X_id[i])
        d_int = int(date_ints[i])
        if tid >= len(ticker_names):
            continue
        tname = str(ticker_names[tid])
        if tname in EXCLUDE_TICKERS:
            continue
        if tname not in ticker_scores:
            ticker_scores[tname] = {h: {} for h in range(ST_N_HORIZONS)}
        for h in range(ST_N_HORIZONS):
            ticker_scores[tname][h][d_int] = float(preds[i, h])
    del preds, X_id, date_ints; gc.collect()
    return ticker_scores


def run_st_scoring():
    print(f"\n  Loading ST SHORT model...")
    meta_path = _ensure_st_data('meta.npz')
    meta = np.load(meta_path, allow_pickle=True)
    ticker_names = meta['ticker_names']

    probe_path = _ensure_st_data(f'fold_{ST_FOLD_ID}_val.npz')
    probe = np.load(probe_path)
    n_af = probe['X_seq'].shape[-1]
    n_rf = probe['X_regime'].shape[-1]
    probe.close()

    model = SpikeTransformer(name='spike_transformer')
    dummy_seq = tf.zeros((1, ST_TIME_STEPS, n_af))
    dummy_reg = tf.zeros((1, n_rf))
    model({'seq_input': dummy_seq, 'regime_input': dummy_reg})

    weights_path = _ensure_st_model(ST_FOLD_ID)
    model.load_weights(weights_path)
    print(f"    ST model loaded: {model.count_params():,} params")

    val_scores = score_st_npz(model, 'val', ticker_names)
    test_scores = score_st_npz(model, 'test', ticker_names)

    all_scores = {}
    for tname in set(val_scores.keys()) | set(test_scores.keys()):
        all_scores[tname] = {}
        for h in range(ST_N_HORIZONS):
            merged = {}
            if tname in val_scores and h in val_scores[tname]:
                merged.update(val_scores[tname][h])
            if tname in test_scores and h in test_scores[tname]:
                merged.update(test_scores[tname][h])
            all_scores[tname][h] = merged

    del model, val_scores, test_scores
    tf.keras.backend.clear_session(); gc.collect()
    return all_scores


# ==============================================================================
# TRADE SIMS (Numba)
# ==============================================================================
@jit(nopython=True)
def st_trade_sim_static(entry_indices, prices, highs, lows, date_ints, time_mins,
                        tp, sl, exit_time):
    n_entries = len(entry_indices); n_prices = len(prices)
    trades = np.empty(n_entries, dtype=np.float64)
    entry_dates = np.empty(n_entries, dtype=np.int64)
    entry_prices = np.empty(n_entries, dtype=np.float64)
    tc = 0; ptr = 0; last_date = -1
    while ptr < n_entries:
        idx = entry_indices[ptr]
        if idx >= n_prices - 1: break
        ed = date_ints[idx]
        if ed == last_date: ptr += 1; continue
        ep = prices[idx]; tp_px = ep * (1 - tp); sl_px = ep * (1 + sl)
        outcome = 0.0; done = False; exit_idx = idx
        for k in range(idx + 1, n_prices):
            if date_ints[k] != ed:
                outcome = (ep - prices[k-1]) / ep; exit_idx = k; done = True; break
            if highs[k] >= sl_px and lows[k] <= tp_px:
                outcome = -sl if np.random.random() < 0.5 else tp; exit_idx = k; done = True; break
            if highs[k] >= sl_px: outcome = -sl; exit_idx = k; done = True; break
            if lows[k] <= tp_px: outcome = tp; exit_idx = k; done = True; break
            if time_mins[k] >= exit_time:
                outcome = (ep - prices[k]) / ep; exit_idx = k; done = True; break
        if not done: outcome = (ep - prices[n_prices-1]) / ep; exit_idx = n_prices
        trades[tc] = outcome; entry_dates[tc] = ed; entry_prices[tc] = ep; tc += 1; last_date = ed
        nxt = ptr + 1
        while nxt < n_entries:
            if entry_indices[nxt] > exit_idx: break
            nxt += 1
        ptr = nxt
    return trades[:tc], entry_dates[:tc], entry_prices[:tc]


@jit(nopython=True)
def st_trade_sim_grace(entry_indices, prices, highs, lows, date_ints, time_mins,
                       tp, sl, exit_time, grace_minutes):
    n_entries = len(entry_indices); n_prices = len(prices)
    trades = np.empty(n_entries, dtype=np.float64)
    entry_dates = np.empty(n_entries, dtype=np.int64)
    entry_prices = np.empty(n_entries, dtype=np.float64)
    tc = 0; ptr = 0; last_date = -1
    while ptr < n_entries:
        idx = entry_indices[ptr]
        if idx >= n_prices - 1: break
        ed = date_ints[idx]
        if ed == last_date: ptr += 1; continue
        ep = prices[idx]; tp_px = ep * (1 - tp); sl_px = ep * (1 + sl)
        et = time_mins[idx]; outcome = 0.0; done = False; exit_idx = idx
        for k in range(idx + 1, n_prices):
            if date_ints[k] != ed:
                outcome = (ep - prices[k-1]) / ep; exit_idx = k; done = True; break
            elapsed = time_mins[k] - et; sl_on = elapsed >= grace_minutes
            tp_hit = lows[k] <= tp_px; sl_hit = sl_on and (highs[k] >= sl_px)
            if sl_hit and tp_hit:
                outcome = -sl if np.random.random() < 0.5 else tp; exit_idx = k; done = True; break
            if sl_hit: outcome = -sl; exit_idx = k; done = True; break
            if tp_hit: outcome = tp; exit_idx = k; done = True; break
            if time_mins[k] >= exit_time:
                outcome = (ep - prices[k]) / ep; exit_idx = k; done = True; break
        if not done: outcome = (ep - prices[n_prices-1]) / ep; exit_idx = n_prices
        trades[tc] = outcome; entry_dates[tc] = ed; entry_prices[tc] = ep; tc += 1; last_date = ed
        nxt = ptr + 1
        while nxt < n_entries:
            if entry_indices[nxt] > exit_idx: break
            nxt += 1
        ptr = nxt
    return trades[:tc], entry_dates[:tc], entry_prices[:tc]


@jit(nopython=True)
def st_trade_sim_limit_entry(entry_indices, prices, highs, lows, date_ints, time_mins,
                             tp, sl, exit_time, rally_pct, max_wait_minutes):
    n_entries = len(entry_indices); n_prices = len(prices)
    trades = np.empty(n_entries, dtype=np.float64)
    entry_dates = np.empty(n_entries, dtype=np.int64)
    entry_prices = np.empty(n_entries, dtype=np.float64)
    tc = 0; ptr = 0; last_date = -1
    while ptr < n_entries:
        idx = entry_indices[ptr]
        if idx >= n_prices - 1: break
        ed = date_ints[idx]
        if ed == last_date: ptr += 1; continue
        sig_px = prices[idx]; limit_px = sig_px * (1.0 + rally_pct)
        sig_t = time_mins[idx]
        fill_idx = -1; fill_px = 0.0
        for k in range(idx + 1, n_prices):
            if date_ints[k] != ed: break
            if time_mins[k] - sig_t > max_wait_minutes: break
            if highs[k] >= limit_px: fill_px = limit_px; fill_idx = k; break
        if fill_idx < 0: ptr += 1; last_date = ed; continue
        tp_px = fill_px * (1 - tp); sl_px = fill_px * (1 + sl)
        outcome = 0.0; done = False; exit_idx = fill_idx
        for k in range(fill_idx + 1, n_prices):
            if date_ints[k] != ed:
                outcome = (fill_px - prices[k-1]) / fill_px; exit_idx = k; done = True; break
            if highs[k] >= sl_px and lows[k] <= tp_px:
                outcome = -sl if np.random.random() < 0.5 else tp; exit_idx = k; done = True; break
            if highs[k] >= sl_px: outcome = -sl; exit_idx = k; done = True; break
            if lows[k] <= tp_px: outcome = tp; exit_idx = k; done = True; break
            if time_mins[k] >= exit_time:
                outcome = (fill_px - prices[k]) / fill_px; exit_idx = k; done = True; break
        if not done: outcome = (fill_px - prices[n_prices-1]) / fill_px; exit_idx = n_prices
        trades[tc] = outcome; entry_dates[tc] = ed; entry_prices[tc] = fill_px; tc += 1; last_date = ed
        nxt = ptr + 1
        while nxt < n_entries:
            if entry_indices[nxt] > exit_idx: break
            nxt += 1
        ptr = nxt
    return trades[:tc], entry_dates[:tc], entry_prices[:tc]


def _run_st_sim(indices, ohlcv, cfg):
    prices = ohlcv['prices']; highs = ohlcv['highs']; lows = ohlcv['lows']
    dates = ohlcv['dates']; times = ohlcv['times'].astype(np.int64)
    mode = cfg['mode']
    if mode == 'grace':
        pnl, d, ep = st_trade_sim_grace(indices, prices, highs, lows, dates, times,
                                         cfg['tp_pct'], cfg['sl_pct'], EXIT_TIME_MIN,
                                         cfg['grace_minutes'])
    elif mode == 'limit_entry':
        pnl, d, ep = st_trade_sim_limit_entry(indices, prices, highs, lows, dates, times,
                                               cfg['tp_pct'], cfg['sl_pct'], EXIT_TIME_MIN,
                                               cfg['dip_pct'], cfg['max_wait'])
    else:
        pnl, d, ep = st_trade_sim_static(indices, prices, highs, lows, dates, times,
                                          cfg['tp_pct'], cfg['sl_pct'], EXIT_TIME_MIN)
    if len(pnl) > 0:
        pnl = pnl - SLIPPAGE_RT
    return pnl, d, ep


# ==============================================================================
# RUN ST STRATEGY — collect all trades with detail
# ==============================================================================
def run_st_strategy(all_daily_scores):
    test_start_dt = pd.Timestamp(TEST_START).date()
    test_end_dt   = pd.Timestamp(TEST_END).date()
    break_dt      = pd.Timestamp(BREAK_DATE).date()

    all_trades = []

    for ticker in sorted(ST_TICKERS):
        cfg = ST_CONFIGS[ticker]
        daily_scores = all_daily_scores.get(ticker, {})
        if not daily_scores:
            print(f"    {ticker}: no ST scores")
            continue

        h_idx = cfg['horizon']
        h_scores = daily_scores.get(h_idx, {})
        if not h_scores:
            print(f"    {ticker}: no scores for horizon {h_idx}")
            continue

        df_ohlcv = load_master(ticker)
        if df_ohlcv is None:
            continue

        ohlcv = precompute_ohlcv_arrays(df_ohlcv, TEST_START, TEST_END)
        if ohlcv is None:
            _master_cache.pop(ticker, None)
            continue

        dates_arr = ohlcv['dates']; times_arr = ohlcv['times']
        score_by_date = {int(d): s for d, s in h_scores.items()}

        entry_list = []
        for i in range(len(dates_arr)):
            if times_arr[i] == ST_ENTRY_MINUTE:
                day_score = score_by_date.get(int(dates_arr[i]), 0.0)
                if day_score >= cfg['prob_thresh']:
                    entry_list.append(i)
        if not entry_list:
            _master_cache.pop(ticker, None)
            continue
        indices = np.array(entry_list, dtype=np.int64)

        raw_pnl, raw_dates, raw_entry_prices = _run_st_sim(indices, ohlcv, cfg)

        for j in range(len(raw_pnl)):
            d_int = int(raw_dates[j])
            d_str = str(d_int)
            d_date = pd.Timestamp(f"{d_str[:4]}-{d_str[4:6]}-{d_str[6:8]}").date()
            if d_date < test_start_dt or d_date > test_end_dt:
                continue
            period = 'POST' if d_date >= break_dt else 'PRE'
            prob = score_by_date.get(d_int, 0.0)
            all_trades.append({
                'date': d_date, 'ticker': ticker, 'pnl': float(raw_pnl[j]),
                'prob': prob, 'entry_price': float(raw_entry_prices[j]),
                'mode': cfg['mode'], 'tp': cfg['tp_pct'], 'sl': cfg['sl_pct'],
                'horizon': cfg['horizon'], 'period': period,
            })

        _master_cache.pop(ticker, None)
        n = len([t for t in all_trades if t['ticker'] == ticker])
        avg = np.mean([t['pnl'] for t in all_trades if t['ticker'] == ticker])
        print(f"    {ticker}: {n} trades, avg={avg:+.2%}")
        _release_memory()

    return all_trades


# ==============================================================================
# MAIN
# ==============================================================================
t_total = time.time()

print(f"\n{'='*W}")
print(f"  ST SHORT POST-JULY ANALYSIS")
print(f"  Period: {TEST_START} -> {TEST_END}  |  Break: {BREAK_DATE}")
print(f"  Slippage: {SLIPPAGE_RT:.2%} RT")
print(f"  GPU: {tf.config.list_physical_devices('GPU')}")
print(f"{'='*W}")

# ── Score ──
np.random.seed(42)
st_daily_scores = run_st_scoring()

# ── Trade ──
print(f"\n  Running ST SHORT trades...")
np.random.seed(42)
all_trades = run_st_strategy(st_daily_scores)

if not all_trades:
    print("  NO TRADES — cannot analyze")
    sys.exit(0)

df = pd.DataFrame(all_trades)
df['date'] = pd.to_datetime(df['date'])
df['week'] = df['date'].dt.isocalendar().week.astype(int)
df['month'] = df['date'].dt.strftime('%Y-%m')
break_dt = pd.Timestamp(BREAK_DATE).date()

pre  = df[df['period'] == 'PRE']
post = df[df['period'] == 'POST']

elapsed = time.time() - t_total
print(f"\n  Scoring + trading done ({elapsed:.0f}s)")


# ==============================================================================
# TABLE 1: PRE vs POST aggregate
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 1: PRE vs POST {BREAK_DATE} — AGGREGATE")
print(f"{'='*W}")

print(f"\n  {'Period':<8} {'Trades':>7} {'Avg PnL':>8} {'Total':>9} {'WR':>5} "
      f"{'Winners':>8} {'Losers':>7} {'Avg Win':>8} {'Avg Loss':>9}")
print(f"  {'─'*78}")

for label, subset in [('PRE', pre), ('POST', post), ('ALL', df)]:
    if len(subset) == 0:
        print(f"  {label:<8} -- no trades")
        continue
    wins = subset[subset['pnl'] > 0]
    losses = subset[subset['pnl'] <= 0]
    print(f"  {label:<8} {len(subset):>7} {subset['pnl'].mean():>+7.2%} "
          f"{subset['pnl'].sum():>+8.1%} {(subset['pnl'] > 0).mean():>5.0%} "
          f"{len(wins):>8} {len(losses):>7} "
          f"{wins['pnl'].mean() if len(wins) > 0 else 0:>+7.2%} "
          f"{losses['pnl'].mean() if len(losses) > 0 else 0:>+8.2%}")


# ==============================================================================
# TABLE 2: PRE vs POST per-ticker
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 2: PRE vs POST {BREAK_DATE} — PER TICKER")
print(f"{'='*W}")

print(f"\n  {'Ticker':<7} {'Mode':<8} {'H':>2} "
      f"{'PRE N':>6} {'PRE Avg':>8} {'PRE WR':>7} "
      f"{'POST N':>7} {'POST Avg':>9} {'POST WR':>8} "
      f"{'Delta':>7}")
print(f"  {'─'*82}")

ticker_summary = []
for ticker in ST_TICKERS:
    cfg = ST_CONFIGS[ticker]
    t_pre  = pre[pre['ticker'] == ticker]
    t_post = post[post['ticker'] == ticker]
    pre_avg  = t_pre['pnl'].mean()  if len(t_pre) > 0  else 0
    post_avg = t_post['pnl'].mean() if len(t_post) > 0 else 0
    pre_wr   = (t_pre['pnl'] > 0).mean()  if len(t_pre) > 0  else 0
    post_wr  = (t_post['pnl'] > 0).mean() if len(t_post) > 0 else 0
    delta = post_avg - pre_avg

    ticker_summary.append({
        'ticker': ticker, 'pre_n': len(t_pre), 'post_n': len(t_post),
        'pre_avg': pre_avg, 'post_avg': post_avg,
        'pre_wr': pre_wr, 'post_wr': post_wr, 'delta': delta,
    })

    print(f"  {ticker:<7} {cfg['mode']:<8} {cfg['horizon']:>2} "
          f"{len(t_pre):>6} {pre_avg:>+7.2%} {pre_wr:>6.0%} "
          f"{len(t_post):>7} {post_avg:>+8.2%} {post_wr:>7.0%} "
          f"{delta:>+6.2%}")

# Sort by POST avg PnL to highlight worst
print(f"\n  WORST POST tickers (sorted by POST avg PnL):")
for s in sorted(ticker_summary, key=lambda x: x['post_avg']):
    flag = " *** LOSING" if s['post_avg'] < 0 else ""
    print(f"    {s['ticker']:<7} POST: {s['post_avg']:>+7.2%} ({s['post_n']} trades){flag}")


# ==============================================================================
# TABLE 3: WEEKLY TIMELINE
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 3: WEEKLY TIMELINE — ALL TICKERS COMBINED")
print(f"{'='*W}")

weekly = df.groupby(df['date'].dt.to_period('W')).agg(
    n=('pnl', 'count'),
    avg=('pnl', 'mean'),
    total=('pnl', 'sum'),
    wr=('pnl', lambda x: (x > 0).mean()),
    tickers=('ticker', 'nunique'),
)
print(f"\n  {'Week':<12} {'N':>4} {'Avg PnL':>8} {'Total':>8} {'WR':>5} {'#Tk':>4}")
print(f"  {'─'*45}")
for idx, row in weekly.iterrows():
    week_start = idx.start_time.date()
    marker = " <<< BREAK" if week_start >= break_dt and (week_start - timedelta(days=7)) < break_dt else ""
    print(f"  {str(idx):<12} {row['n']:>4} {row['avg']:>+7.2%} "
          f"{row['total']:>+7.2%} {row['wr']:>5.0%} {row['tickers']:>4}{marker}")


# ==============================================================================
# TABLE 4: PER-TICKER WEEKLY (losers only, post-break)
# ==============================================================================
losing_tickers = [s['ticker'] for s in ticker_summary if s['post_avg'] < 0]

if losing_tickers:
    print(f"\n{'='*W}")
    print(f"  TABLE 4: WEEKLY DETAIL — LOSING POST-BREAK TICKERS")
    print(f"  Tickers: {', '.join(losing_tickers)}")
    print(f"{'='*W}")

    for ticker in losing_tickers:
        t_df = df[df['ticker'] == ticker].copy()
        t_weekly = t_df.groupby(t_df['date'].dt.to_period('W')).agg(
            n=('pnl', 'count'), avg=('pnl', 'mean'), total=('pnl', 'sum'))
        print(f"\n  {ticker} ({ST_CONFIGS[ticker]['mode']}, "
              f"TP={ST_CONFIGS[ticker]['tp_pct']:.0%} SL={ST_CONFIGS[ticker]['sl_pct']:.0%}):")
        print(f"    {'Week':<12} {'N':>4} {'Avg PnL':>8} {'Total':>8}")
        print(f"    {'─'*35}")
        for idx, row in t_weekly.iterrows():
            print(f"    {str(idx):<12} {row['n']:>4} {row['avg']:>+7.2%} {row['total']:>+7.2%}")


# ==============================================================================
# TABLE 5: TRADE-LEVEL DETAIL (post-break losing trades)
# ==============================================================================
if losing_tickers:
    print(f"\n{'='*W}")
    print(f"  TABLE 5: TRADE-LEVEL DETAIL — POST-BREAK LOSSES")
    print(f"{'='*W}")

    for ticker in losing_tickers:
        t_post_trades = post[post['ticker'] == ticker].sort_values('date')
        if len(t_post_trades) == 0:
            continue
        print(f"\n  {ticker}:")
        print(f"    {'Date':<12} {'PnL':>8} {'Prob':>6} {'Entry$':>8} {'W/L':<4}")
        print(f"    {'─'*42}")
        for _, t in t_post_trades.iterrows():
            wl = 'WIN' if t['pnl'] > 0 else 'LOSS'
            print(f"    {t['date'].strftime('%Y-%m-%d'):<12} {t['pnl']:>+7.2%} "
                  f"{t['prob']:>5.3f} ${t['entry_price']:>7.2f} {wl:<4}")


# ==============================================================================
# TABLE 6: SIGNAL FREQUENCY — did prediction rate change?
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 6: SIGNAL FREQUENCY — PRE vs POST")
print(f"{'='*W}")

print(f"\n  {'Ticker':<7} {'PRE signals':>12} {'PRE days':>9} {'PRE rate':>9} "
      f"{'POST signals':>13} {'POST days':>10} {'POST rate':>10}")
print(f"  {'─'*75}")

for ticker in ST_TICKERS:
    cfg = ST_CONFIGS[ticker]
    h_scores = st_daily_scores.get(ticker, {}).get(cfg['horizon'], {})
    if not h_scores:
        continue

    test_start_int = int(pd.Timestamp(TEST_START).strftime('%Y%m%d'))
    test_end_int   = int(pd.Timestamp(TEST_END).strftime('%Y%m%d'))
    break_int      = int(pd.Timestamp(BREAK_DATE).strftime('%Y%m%d'))

    pre_days = pre_signals = post_days = post_signals = 0
    for d_int, score in h_scores.items():
        d_int = int(d_int)
        if d_int < test_start_int or d_int > test_end_int:
            continue
        if d_int < break_int:
            pre_days += 1
            if score >= cfg['prob_thresh']:
                pre_signals += 1
        else:
            post_days += 1
            if score >= cfg['prob_thresh']:
                post_signals += 1

    pre_rate  = pre_signals / pre_days  if pre_days > 0  else 0
    post_rate = post_signals / post_days if post_days > 0 else 0
    print(f"  {ticker:<7} {pre_signals:>12} {pre_days:>9} {pre_rate:>8.1%} "
          f"{post_signals:>13} {post_days:>10} {post_rate:>9.1%}")


# ==============================================================================
# TABLE 7: AVG PROB — did model confidence change?
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 7: MODEL CONFIDENCE — AVG PROB PRE vs POST")
print(f"{'='*W}")

print(f"\n  {'Ticker':<7} {'H':>2} {'Thr':>5} "
      f"{'PRE avg prob':>13} {'POST avg prob':>14} {'Delta':>7}")
print(f"  {'─'*55}")

for ticker in ST_TICKERS:
    cfg = ST_CONFIGS[ticker]
    h_scores = st_daily_scores.get(ticker, {}).get(cfg['horizon'], {})
    if not h_scores:
        continue

    test_start_int = int(pd.Timestamp(TEST_START).strftime('%Y%m%d'))
    test_end_int   = int(pd.Timestamp(TEST_END).strftime('%Y%m%d'))
    break_int      = int(pd.Timestamp(BREAK_DATE).strftime('%Y%m%d'))

    pre_probs = []
    post_probs = []
    for d_int, score in h_scores.items():
        d_int = int(d_int)
        if d_int < test_start_int or d_int > test_end_int:
            continue
        if d_int < break_int:
            pre_probs.append(score)
        else:
            post_probs.append(score)

    pre_avg  = np.mean(pre_probs)  if pre_probs  else 0
    post_avg = np.mean(post_probs) if post_probs else 0
    delta = post_avg - pre_avg
    print(f"  {ticker:<7} {cfg['horizon']:>2} {cfg['prob_thresh']:>4.2f} "
          f"{pre_avg:>12.4f} {post_avg:>13.4f} {delta:>+6.4f}")


# ==============================================================================
# CHARTS
# ==============================================================================
fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# ── [0,0]: Per-ticker equity curves ──
ax = axes[0, 0]
colors = plt.cm.tab20(np.linspace(0, 1, len(ST_TICKERS)))
for i, ticker in enumerate(ST_TICKERS):
    t_df = df[df['ticker'] == ticker].sort_values('date')
    if len(t_df) == 0:
        continue
    cum = t_df['pnl'].cumsum() * 100
    ax.plot(t_df['date'], cum.values, label=f"{ticker} ({len(t_df)}t)",
            color=colors[i], linewidth=1.5)
ax.axvline(pd.Timestamp(BREAK_DATE), color='red', linestyle='--', linewidth=2,
           label=f'Break: {BREAK_DATE}')
ax.set_title('Per-Ticker Cumulative PnL (%)')
ax.set_ylabel('Cumulative PnL (%)')
ax.legend(fontsize=7, ncol=2, loc='upper left')
ax.grid(alpha=0.3)

# ── [0,1]: PRE vs POST bar chart ──
ax = axes[0, 1]
tickers_sorted = sorted(ST_TICKERS, key=lambda t: next(
    (s['post_avg'] for s in ticker_summary if s['ticker'] == t), 0))
x = np.arange(len(tickers_sorted))
pre_vals  = [next((s['pre_avg'] for s in ticker_summary if s['ticker'] == t), 0) * 100
             for t in tickers_sorted]
post_vals = [next((s['post_avg'] for s in ticker_summary if s['ticker'] == t), 0) * 100
             for t in tickers_sorted]
ax.barh(x - 0.2, pre_vals, 0.35, label=f'PRE ({TEST_START} to {BREAK_DATE})',
        color='#2E7D32', alpha=0.8)
ax.barh(x + 0.2, post_vals, 0.35, label=f'POST ({BREAK_DATE} to {TEST_END})',
        color='#C62828', alpha=0.8)
ax.set_yticks(x)
ax.set_yticklabels(tickers_sorted, fontsize=9)
ax.set_xlabel('Avg PnL (%)')
ax.set_title(f'Per-Ticker Avg PnL: PRE vs POST {BREAK_DATE}')
ax.axvline(0, color='black', linewidth=0.5)
ax.legend(fontsize=8)
ax.grid(axis='x', alpha=0.3)

# ── [1,0]: Weekly PnL timeline (portfolio) ──
ax = axes[1, 0]
weekly_dates = [p.start_time for p in weekly.index]
weekly_colors = ['#2E7D32' if p.start_time.date() < break_dt else '#C62828'
                 for p in weekly.index]
ax.bar(weekly_dates, weekly['avg'].values * 100, width=5, color=weekly_colors, alpha=0.8)
ax.axvline(pd.Timestamp(BREAK_DATE), color='red', linestyle='--', linewidth=2)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_title('Weekly Avg PnL (%) — All Tickers')
ax.set_ylabel('Avg PnL (%)')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
ax.grid(axis='y', alpha=0.3)

# ── [1,1]: Signal rate + confidence shift ──
ax = axes[1, 1]
tickers_for_conf = []
pre_conf = []
post_conf = []
for ticker in ST_TICKERS:
    cfg = ST_CONFIGS[ticker]
    h_scores = st_daily_scores.get(ticker, {}).get(cfg['horizon'], {})
    if not h_scores:
        continue
    test_start_int = int(pd.Timestamp(TEST_START).strftime('%Y%m%d'))
    test_end_int   = int(pd.Timestamp(TEST_END).strftime('%Y%m%d'))
    break_int      = int(pd.Timestamp(BREAK_DATE).strftime('%Y%m%d'))
    pp = [s for d, s in h_scores.items() if test_start_int <= int(d) < break_int]
    po = [s for d, s in h_scores.items() if break_int <= int(d) <= test_end_int]
    if pp and po:
        tickers_for_conf.append(ticker)
        pre_conf.append(np.mean(pp))
        post_conf.append(np.mean(po))

if tickers_for_conf:
    x = np.arange(len(tickers_for_conf))
    ax.bar(x - 0.2, pre_conf, 0.35, label='PRE avg prob', color='#1565C0', alpha=0.8)
    ax.bar(x + 0.2, post_conf, 0.35, label='POST avg prob', color='#E65100', alpha=0.8)
    for i, ticker in enumerate(tickers_for_conf):
        cfg = ST_CONFIGS[ticker]
        ax.axhline(cfg['prob_thresh'], xmin=(i - 0.4) / len(tickers_for_conf),
                    xmax=(i + 0.4) / len(tickers_for_conf),
                    color='gray', linestyle=':', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(tickers_for_conf, rotation=45, fontsize=8)
    ax.set_ylabel('Avg Probability')
    ax.set_title('Model Confidence: PRE vs POST')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

fig.suptitle(f'ST SHORT Post-July Analysis\n'
             f'{TEST_START} to {TEST_END} | Break: {BREAK_DATE} | '
             f'Slippage: {SLIPPAGE_RT:.2%} RT',
             fontsize=13, fontweight='bold')
plt.tight_layout()

chart_path = '/tmp/analysis_st_short_post_july.png'
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
print(f"\n  Chart saved: {chart_path}")
if _IN_COLAB:
    plt.show()
plt.close()


# ==============================================================================
# RECOMMENDATIONS
# ==============================================================================
print(f"\n{'='*W}")
print(f"  RECOMMENDATIONS")
print(f"{'='*W}")

if losing_tickers:
    print(f"\n  LOSING TICKERS POST {BREAK_DATE}: {', '.join(losing_tickers)}")
    print(f"  Options:")
    print(f"    1. REMOVE from deploy (quickest fix, no retraining)")
    print(f"    2. TIGHTEN thresholds (raise prob_thresh to reduce false signals)")
    print(f"    3. RETRAIN fold 5 (train through Jun 30, val Jul, test Aug)")
    print(f"    4. ATR-normalize target (regime-agnostic dump definition)")
else:
    print(f"\n  No tickers losing money post {BREAK_DATE}.")
    print(f"  The degradation may be aggregate (smaller wins, not losses).")

# Check if it's fewer/smaller wins vs actual losses
post_wins = post[post['pnl'] > 0]
post_losses = post[post['pnl'] <= 0]
if len(post) > 0:
    print(f"\n  POST breakdown:")
    print(f"    Trades: {len(post)}, Wins: {len(post_wins)}, Losses: {len(post_losses)}")
    if len(post_wins) > 0:
        print(f"    Avg win:  {post_wins['pnl'].mean():+.2%}")
    if len(post_losses) > 0:
        print(f"    Avg loss: {post_losses['pnl'].mean():+.2%}")
    print(f"    Win rate: {(post['pnl'] > 0).mean():.0%}")

print(f"\n{'='*W}")
print(f"  DONE ({time.time() - t_total:.0f}s)")
print(f"{'='*W}")
