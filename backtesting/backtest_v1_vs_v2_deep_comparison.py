# %% ── SHORT L5-2: Multi-Res v1 (50k) vs v2 (150k) — Deep Per-Asset Comparison ──
#
# Head-to-head analysis answering: what did 3x more training data change?
#
# Phase 1: Load both models, score test set at SCORE_MINUTE
# Phase 2: Deploy universe comparison (v1-only, v2-only, overlap)
# Phase 3: Per-ticker PnL comparison (best config each)
# Phase 4: Signal agreement — how often do models agree on trade days?
# Phase 5: Per-ticker daily PnL curves (top tickers, divergent tickers)
# Phase 6: Summary verdict
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

from sklearn.metrics import roc_auc_score


# ==============================================================================
# CONFIG
# ==============================================================================
GCS_BUCKET = 'time-series-bucket-ys'
GCS_INFER_DIR = 'Data/Inference_v2'

MODELS = {
    'V1': {
        'label': 'Multi-Res v1 (50k)',
        'gcs_data':   'Algos/RA_MoE/short_L5_2_multires_data',
        'gcs_models': 'Algos/RA_MoE/short_L5_2_multires_models',
        'local_data':  '/tmp/_bt_v1v2_v1_data',
        'local_model': '/tmp/_bt_v1v2_v1_models',
        'train_samples': 50_000,
    },
    'V2': {
        'label': 'Multi-Res v2 (150k)',
        'gcs_data':   'Algos/RA_MoE/short_L5_2_multires_v2_data',
        'gcs_models': 'Algos/RA_MoE/short_L5_2_multires_v2_models',
        'local_data':  '/tmp/_bt_v1v2_v2_data',
        'local_model': '/tmp/_bt_v1v2_v2_models',
        'train_samples': 150_000,
    },
}
MODEL_KEYS = ['V1', 'V2']

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
W = 100

TEST_START = '2026-07-16'
TEST_END = '2026-08-21'

OUTPUT_DIR = '/tmp/_bt_v1v2_comparison'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================================================================
# GCS HELPERS
# ==============================================================================
_gcs_client = storage.Client()

def _bucket():
    return _gcs_client.bucket(GCS_BUCKET)

def ensure_file(filename, gcs_prefix, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, filename)
    if os.path.exists(local_path):
        return local_path
    print(f"    Downloading {filename}...")
    _bucket().blob(f'{gcs_prefix}/{filename}').download_to_filename(local_path)
    return local_path

def ensure_model_weights(fold_id, gcs_prefix, local_dir):
    local_fold = os.path.join(local_dir, f'fold_{fold_id}')
    weights_path = os.path.join(local_fold, 'model.weights.h5')
    if os.path.exists(weights_path):
        return weights_path
    os.makedirs(local_fold, exist_ok=True)
    print(f"    Downloading model fold {fold_id}...")
    for blob in _bucket().list_blobs(prefix=f'{gcs_prefix}/fold_{fold_id}'):
        blob.download_to_filename(os.path.join(local_fold, blob.name.split('/')[-1]))
    return weights_path


# ==============================================================================
# MODEL ARCHITECTURE (Multi-Res only)
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
    def __init__(self, d_model, n_heads, d_ff, n_blocks, seq_len, dropout=0.1, **kw):
        super().__init__(**kw)
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
            layers.Dense(hidden_dim, activation='relu'),
            layers.Dense(hidden_dim, activation='relu')])
        self.gamma_proj = layers.Dense(d_model)
        self.beta_proj = layers.Dense(d_model)
    def call(self, x, regime_features, training=False):
        h = self.mlp(regime_features, training=training)
        return (1.0 + self.gamma_proj(h)) * x + self.beta_proj(h)

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
        self.film   = FiLMConditioner(D_MODEL, FILM_HIDDEN, name='film')
        self.drop   = layers.Dropout(DROPOUT)
        self.head   = layers.Dense(N_HORIZONS, activation='sigmoid', name='spike_head')
    def call(self, inputs, training=False):
        h_1m  = self.pool_1m(self.backbone_1m(inputs['seq_input'], training=training), training=training)
        h_15m = self.pool_15m(self.backbone_15m(inputs['seq_15m_input'], training=training), training=training)
        h = self.fusion(tf.concat([h_1m, h_15m], axis=-1))
        h = self.film(h, inputs['regime_input'], training=training)
        return self.head(self.drop(h, training=training))

def build_model(n_asset, n_regime, n_lowres):
    model = MultiResSpikeTransformer(name='multires_spike_transformer')
    model({'seq_input': tf.zeros((1, TIME_STEPS, n_asset)),
           'seq_15m_input': tf.zeros((1, TIME_STEPS_15M, n_lowres)),
           'regime_input': tf.zeros((1, n_regime))}, training=False)
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
# OHLCV LOADING
# ==============================================================================
def load_ticker_ohlcv(ticker):
    clean = TICKER_FILE_MAP.get(ticker, ticker)
    for suffix in ['_master_inference', '_1m_full_historical_data']:
        blob_path = f"{GCS_INFER_DIR}/{clean}{suffix}.parquet"
        local_path = f"/tmp/_bt_v1v2_{clean}{suffix}.parquet"
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
# PHASE 1: LOAD MODELS & SCORE
# ==============================================================================
t_total = time.time()

print(f"\n{'=' * W}")
print(f"  Multi-Res v1 (50k) vs v2 (150k) — DEEP COMPARISON")
print(f"  Test: {TEST_START} to {TEST_END}")
print(f"{'=' * W}")

model_preds = {}

for mk, mcfg in MODELS.items():
    print(f"\n  Loading {mcfg['label']}...")
    meta = np.load(ensure_file('meta.npz', mcfg['gcs_data'], mcfg['local_data']),
                   allow_pickle=True)
    ticker_names = meta['ticker_names']
    test_data = np.load(ensure_file(f'fold_{FOLD_ID}_test.npz', mcfg['gcs_data'],
                                     mcfg['local_data']))
    X_seq = test_data['X_seq']
    X_reg = test_data['X_regime']
    X_seq_15m = test_data['X_seq_15m']
    X_id = test_data['X_id'].flatten()
    Y = test_data['Y']
    date_ints = test_data['date_ints']
    minutes = test_data['minutes']
    test_data.close()

    entry_mask = (minutes == SCORE_MINUTE)
    X_seq_e = X_seq[entry_mask]
    X_reg_e = X_reg[entry_mask]
    X_15m_e = X_seq_15m[entry_mask]
    X_id_e = X_id[entry_mask]
    Y_e = Y[entry_mask]
    date_ints_e = date_ints[entry_mask]
    del X_seq, X_reg, X_seq_15m, X_id, Y, date_ints, minutes
    gc.collect()

    n_asset = X_seq_e.shape[-1]
    n_regime = X_reg_e.shape[-1]
    n_lowres = X_15m_e.shape[-1]

    model = build_model(n_asset, n_regime, n_lowres)
    model.load_weights(ensure_model_weights(FOLD_ID, mcfg['gcs_models'], mcfg['local_model']))
    preds = model.predict({
        'seq_input': X_seq_e.astype(np.float32),
        'seq_15m_input': X_15m_e.astype(np.float32),
        'regime_input': X_reg_e.astype(np.float32),
    }, batch_size=512, verbose=0)

    model_preds[mk] = {
        'preds': preds, 'Y': Y_e, 'X_id': X_id_e,
        'date_ints': date_ints_e, 'ticker_names': ticker_names,
    }
    print(f"    {entry_mask.sum():,} score-minute samples, {model.count_params():,} params")
    print(f"    Avg preds: " + "  ".join(
        f"{HORIZON_LABELS[i]}={preds[:, i].mean()*100:.1f}%" for i in range(N_HORIZONS)))

    del model, X_seq_e, X_reg_e, X_15m_e
    tf.keras.backend.clear_session()
    gc.collect()


# ==============================================================================
# PHASE 2: TRADE SIMULATION — FULL GRID BOTH MODELS
# ==============================================================================
print(f"\n{'=' * W}")
print(f"  PHASE 2: TRADE SIMULATION")
print(f"{'=' * W}")

scored_tickers = set()
for mk in MODEL_KEYS:
    d = model_preds[mk]
    for i in range(len(d['X_id'])):
        tid = int(d['X_id'][i])
        if tid < len(d['ticker_names']):
            tk = str(d['ticker_names'][tid])
            if tk not in EXCLUDE_TICKERS:
                scored_tickers.add(tk)
scored_tickers = sorted(scored_tickers)

trade_results = {mk: [] for mk in MODEL_KEYS}
daily_trades = {mk: [] for mk in MODEL_KEYS}

for t_idx, ticker in enumerate(scored_tickers):
    df_ohlcv = load_ticker_ohlcv(ticker)
    if df_ohlcv is None:
        continue
    ohlcv = precompute_ohlcv(df_ohlcv, TEST_START, TEST_END)
    del df_ohlcv
    if ohlcv is None:
        continue

    for mk in MODEL_KEYS:
        d = model_preds[mk]
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
                score_by_date[int(dates_tk[j])] = float(preds_tk[j, h_idx])

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
                    pnl, e_dates = run_short_trade_sim(
                        indices, ohlcv['prices'], ohlcv['highs'], ohlcv['lows'],
                        ohlcv['dates'], ohlcv['times'].astype(np.int64),
                        pair['tp_pct'], pair['sl_pct'], EXIT_TIME_MIN)
                    if len(pnl) > 0 and SLIPPAGE_PCT > 0:
                        pnl = pnl - 2 * SLIPPAGE_PCT
                    if len(pnl) < 2:
                        continue
                    trade_results[mk].append({
                        'ticker': ticker, 'horizon': h_label,
                        'prob_thresh': prob_thresh,
                        'tp_pct': pair['tp_pct'], 'sl_pct': pair['sl_pct'],
                        'avg_pnl': float(np.mean(pnl)),
                        'total_pnl': float(np.sum(pnl)),
                        'win_rate': float(np.mean(pnl > 0)),
                        'n_trades': len(pnl),
                        'freq': len(pnl) / ohlcv['months'],
                        'is_crypto': ticker in CRYPTO_TICKERS,
                    })
                    for j in range(len(pnl)):
                        daily_trades[mk].append({
                            'ticker': ticker, 'horizon': h_label,
                            'prob_thresh': prob_thresh,
                            'tp_pct': pair['tp_pct'], 'sl_pct': pair['sl_pct'],
                            'date_int': int(e_dates[j]),
                            'pnl': float(pnl[j]),
                        })

    if (t_idx + 1) % 20 == 0:
        gc.collect()
        print(f"    {t_idx+1}/{len(scored_tickers)} tickers")
print(f"    Done")


# ==============================================================================
# PHASE 3: DEPLOY UNIVERSE COMPARISON
# ==============================================================================
print(f"\n{'=' * W}")
print(f"  PHASE 3: DEPLOY UNIVERSE — v1 vs v2")
print(f"{'=' * W}")

deploy_info = {}
for mk in MODEL_KEYS:
    df_r = pd.DataFrame(trade_results[mk])
    deployable = df_r[df_r['avg_pnl'] >= DEPLOY_PNL_THRESHOLD]
    best = deployable.loc[deployable.groupby('ticker')['avg_pnl'].idxmax()] if len(deployable) > 0 else pd.DataFrame()
    deploy_info[mk] = best.set_index('ticker') if len(best) > 0 else pd.DataFrame()

v1_tickers = set(deploy_info['V1'].index) if len(deploy_info['V1']) > 0 else set()
v2_tickers = set(deploy_info['V2'].index) if len(deploy_info['V2']) > 0 else set()
both = v1_tickers & v2_tickers
v1_only = v1_tickers - v2_tickers
v2_only = v2_tickers - v1_tickers

print(f"\n  v1 deployable: {len(v1_tickers)} tickers")
print(f"  v2 deployable: {len(v2_tickers)} tickers")
print(f"  Both:          {len(both)} tickers")
print(f"  v1-only:       {len(v1_only)} tickers")
print(f"  v2-only:       {len(v2_only)} tickers")

if v1_only:
    print(f"\n  v1-ONLY deploy tickers (v2 lost these):")
    for tk in sorted(v1_only):
        r = deploy_info['V1'].loc[tk]
        tag = ' CRYPTO' if tk in CRYPTO_TICKERS else ''
        print(f"    {tk:<8} avg={r['avg_pnl']:+.2%}  n={int(r['n_trades'])}  "
              f"WR={r['win_rate']:.0%}  h={r['horizon']}  "
              f"tp={r['tp_pct']:.0%}/sl={r['sl_pct']:.0%}{tag}")

if v2_only:
    print(f"\n  v2-ONLY deploy tickers (v2 gained these):")
    for tk in sorted(v2_only):
        r = deploy_info['V2'].loc[tk]
        tag = ' CRYPTO' if tk in CRYPTO_TICKERS else ''
        print(f"    {tk:<8} avg={r['avg_pnl']:+.2%}  n={int(r['n_trades'])}  "
              f"WR={r['win_rate']:.0%}  h={r['horizon']}  "
              f"tp={r['tp_pct']:.0%}/sl={r['sl_pct']:.0%}{tag}")

if v2_only:
    v2_only_avg = deploy_info['V2'].loc[list(v2_only)]['avg_pnl'].mean()
    print(f"\n  v2-only tickers avg PnL: {v2_only_avg:+.2%}")
if v1_only:
    v1_only_avg = deploy_info['V1'].loc[list(v1_only)]['avg_pnl'].mean()
    print(f"  v1-only tickers avg PnL: {v1_only_avg:+.2%}")
if both:
    v1_shared_avg = deploy_info['V1'].loc[list(both)]['avg_pnl'].mean()
    v2_shared_avg = deploy_info['V2'].loc[list(both)]['avg_pnl'].mean()
    print(f"\n  Shared tickers avg PnL:  v1={v1_shared_avg:+.2%}  v2={v2_shared_avg:+.2%}  "
          f"delta={v2_shared_avg - v1_shared_avg:+.2%}")


# ==============================================================================
# PHASE 4: PER-TICKER HEAD-TO-HEAD (best config each)
# ==============================================================================
print(f"\n{'=' * W}")
print(f"  PHASE 4: PER-TICKER HEAD-TO-HEAD")
print(f"{'=' * W}")

all_deploy = v1_tickers | v2_tickers
comparison_rows = []
for tk in sorted(all_deploy):
    v1_pnl = deploy_info['V1'].loc[tk, 'avg_pnl'] if tk in v1_tickers else None
    v2_pnl = deploy_info['V2'].loc[tk, 'avg_pnl'] if tk in v2_tickers else None
    v1_n = int(deploy_info['V1'].loc[tk, 'n_trades']) if tk in v1_tickers else 0
    v2_n = int(deploy_info['V2'].loc[tk, 'n_trades']) if tk in v2_tickers else 0
    v1_wr = deploy_info['V1'].loc[tk, 'win_rate'] if tk in v1_tickers else 0
    v2_wr = deploy_info['V2'].loc[tk, 'win_rate'] if tk in v2_tickers else 0
    comparison_rows.append({
        'ticker': tk, 'v1_pnl': v1_pnl, 'v2_pnl': v2_pnl,
        'v1_n': v1_n, 'v2_n': v2_n, 'v1_wr': v1_wr, 'v2_wr': v2_wr,
        'is_crypto': tk in CRYPTO_TICKERS,
        'category': 'both' if tk in both else ('v1_only' if tk in v1_only else 'v2_only'),
    })

comp_df = pd.DataFrame(comparison_rows).set_index('ticker')
comp_df['delta'] = comp_df['v2_pnl'].fillna(0) - comp_df['v1_pnl'].fillna(0)
comp_df = comp_df.sort_values('delta', ascending=False)

print(f"\n  {'Ticker':<8} {'v1 PnL':>8} {'v1 N':>5} {'v1 WR':>6}  "
      f"{'v2 PnL':>8} {'v2 N':>5} {'v2 WR':>6}  {'Delta':>8} {'Cat':>8} {'Type':>7}")
print(f"  {'─' * 85}")
for tk, r in comp_df.iterrows():
    v1_s = f"{r['v1_pnl']:+.2%}" if r['v1_pnl'] is not None else '  n/a  '
    v2_s = f"{r['v2_pnl']:+.2%}" if r['v2_pnl'] is not None else '  n/a  '
    tag = 'CRYPTO' if r['is_crypto'] else ''
    print(f"  {tk:<8} {v1_s:>8} {r['v1_n']:>5} {r['v1_wr']:>5.0%}  "
          f"{v2_s:>8} {r['v2_n']:>5} {r['v2_wr']:>5.0%}  "
          f"{r['delta']:>+7.2%} {r['category']:>8} {tag:>7}")

v2_better = (comp_df['delta'] > 0).sum()
v1_better = (comp_df['delta'] < 0).sum()
print(f"\n  v2 better: {v2_better}/{len(comp_df)}  |  v1 better: {v1_better}/{len(comp_df)}")

shared_mask = comp_df['category'] == 'both'
if shared_mask.any():
    shared = comp_df[shared_mask]
    v2_wins_shared = (shared['delta'] > 0).sum()
    print(f"  On shared tickers: v2 wins {v2_wins_shared}/{len(shared)}")
    print(f"  Shared avg delta:  {shared['delta'].mean():+.2%}")


# ==============================================================================
# PHASE 5: SIGNAL AGREEMENT ANALYSIS
# ==============================================================================
print(f"\n{'=' * W}")
print(f"  PHASE 5: SIGNAL AGREEMENT (30m horizon)")
print(f"{'=' * W}")

h_compare = 1
common_tickers_for_signal = set()
for mk in MODEL_KEYS:
    d = model_preds[mk]
    for i, tn in enumerate(d['ticker_names']):
        if str(tn) not in EXCLUDE_TICKERS:
            common_tickers_for_signal.add(str(tn))

for thresh in [0.30, 0.50, 0.70]:
    agree_both_fire = 0
    agree_neither = 0
    v1_only_fire = 0
    v2_only_fire = 0
    total_days = 0

    for tk in sorted(common_tickers_for_signal):
        v1_scores = {}
        v2_scores = {}
        for mk in MODEL_KEYS:
            d = model_preds[mk]
            tid = None
            for i, tn in enumerate(d['ticker_names']):
                if str(tn) == tk:
                    tid = i
                    break
            if tid is None:
                continue
            mask = d['X_id'] == tid
            preds_tk = d['preds'][mask]
            dates_tk = d['date_ints'][mask]
            scores = {}
            for j in range(len(preds_tk)):
                scores[int(dates_tk[j])] = float(preds_tk[j, h_compare])
            if mk == 'V1':
                v1_scores = scores
            else:
                v2_scores = scores

        all_dates = set(v1_scores.keys()) | set(v2_scores.keys())
        for d_int in all_dates:
            s1 = v1_scores.get(d_int, 0)
            s2 = v2_scores.get(d_int, 0)
            fire1 = s1 >= thresh
            fire2 = s2 >= thresh
            if fire1 and fire2:
                agree_both_fire += 1
            elif not fire1 and not fire2:
                agree_neither += 1
            elif fire1:
                v1_only_fire += 1
            else:
                v2_only_fire += 1
            total_days += 1

    agree_pct = (agree_both_fire + agree_neither) / max(total_days, 1)
    print(f"\n  Threshold >= {thresh:.0%}:")
    print(f"    Total ticker-days:  {total_days:,}")
    print(f"    Both fire:          {agree_both_fire:,} ({agree_both_fire/max(total_days,1)*100:.1f}%)")
    print(f"    Neither fire:       {agree_neither:,} ({agree_neither/max(total_days,1)*100:.1f}%)")
    print(f"    v1-only fire:       {v1_only_fire:,} ({v1_only_fire/max(total_days,1)*100:.1f}%)")
    print(f"    v2-only fire:       {v2_only_fire:,} ({v2_only_fire/max(total_days,1)*100:.1f}%)")
    print(f"    Agreement:          {agree_pct*100:.1f}%")


# ==============================================================================
# PHASE 6: DAILY PnL CURVES + CHARTS
# ==============================================================================
print(f"\n{'=' * W}")
print(f"  PHASE 6: PnL OVER TIME — BEST CONFIG PER TICKER")
print(f"{'=' * W}")

daily_pnl_by_model = {}
for mk in MODEL_KEYS:
    d = model_preds[mk]
    tnames = d['ticker_names']
    df_r = pd.DataFrame(trade_results[mk])
    if len(df_r) == 0:
        daily_pnl_by_model[mk] = pd.Series(dtype=float)
        continue
    deployable = df_r[df_r['avg_pnl'] >= DEPLOY_PNL_THRESHOLD]
    if len(deployable) == 0:
        daily_pnl_by_model[mk] = pd.Series(dtype=float)
        continue
    best_cfg = deployable.loc[deployable.groupby('ticker')['avg_pnl'].idxmax()]

    all_daily = []
    for _, cfg_row in best_cfg.iterrows():
        ticker = cfg_row['ticker']
        h_label = cfg_row['horizon']
        prob_thresh = cfg_row['prob_thresh']
        h_idx = {v: k for k, v in HORIZON_LABELS.items()}[h_label]

        tid = None
        for i, tn in enumerate(tnames):
            if str(tn) == ticker:
                tid = i
                break
        if tid is None:
            continue
        mask = d['X_id'] == tid
        preds_tk = d['preds'][mask]
        dates_tk = d['date_ints'][mask]
        score_by_date = {int(dates_tk[j]): float(preds_tk[j, h_idx]) for j in range(len(preds_tk))}

        df_ohlcv = load_ticker_ohlcv(ticker)
        if df_ohlcv is None:
            continue
        ohlcv = precompute_ohlcv(df_ohlcv, TEST_START, TEST_END)
        del df_ohlcv
        if ohlcv is None:
            continue

        entry_list = [i for i in range(len(ohlcv['dates']))
                      if ohlcv['times'][i] == ENTRY_MINUTE
                      and score_by_date.get(int(ohlcv['dates'][i]), 0) >= prob_thresh]
        if not entry_list:
            continue
        pnl_arr, entry_dates = run_short_trade_sim(
            np.array(entry_list, dtype=np.int64),
            ohlcv['prices'], ohlcv['highs'], ohlcv['lows'],
            ohlcv['dates'], ohlcv['times'].astype(np.int64),
            cfg_row['tp_pct'], cfg_row['sl_pct'], EXIT_TIME_MIN)
        if len(pnl_arr) > 0 and SLIPPAGE_PCT > 0:
            pnl_arr = pnl_arr - 2 * SLIPPAGE_PCT
        for j in range(len(pnl_arr)):
            d_int = int(entry_dates[j])
            dt = pd.Timestamp(year=d_int // 10000, month=(d_int % 10000) // 100, day=d_int % 100)
            all_daily.append({'date': dt, 'ticker': ticker, 'pnl': float(pnl_arr[j])})

    if all_daily:
        daily_df = pd.DataFrame(all_daily)
        daily_pnl_by_model[mk] = daily_df.groupby('date')['pnl'].mean().sort_index()
    else:
        daily_pnl_by_model[mk] = pd.Series(dtype=float)

for mk in MODEL_KEYS:
    s = daily_pnl_by_model[mk]
    if len(s) == 0:
        continue
    cum = s.cumsum()
    dd = cum - cum.cummax()
    print(f"\n  {MODELS[mk]['label']}:")
    print(f"    Trading days:    {len(s)}")
    print(f"    Cumul PnL:       {cum.iloc[-1]:+.2%}")
    print(f"    Avg daily PnL:   {s.mean():+.2%}")
    print(f"    Win rate:        {(s > 0).mean():.0%}")
    print(f"    Max drawdown:    {dd.min():+.2%}")
    print(f"    Best/worst day:  {s.max():+.2%} / {s.min():+.2%}")

# -- Charts --
colors = {'V1': '#4CAF50', 'V2': '#E91E63'}

fig, axes = plt.subplots(2, 3, figsize=(22, 12))
fig.suptitle('Multi-Res v1 (50k) vs v2 (150k) — Deep Comparison',
             fontsize=14, fontweight='bold')

# [0,0] Cumulative PnL
ax = axes[0, 0]
for mk in MODEL_KEYS:
    s = daily_pnl_by_model[mk]
    if len(s) > 0:
        cum = s.cumsum() * 100
        ax.plot(cum.index, cum.values, label=MODELS[mk]['label'],
                color=colors[mk], linewidth=2.5)
ax.axhline(0, color='black', linewidth=0.8, alpha=0.5)
ax.set_ylabel('Cumulative PnL (%)')
ax.set_title('Cumulative PnL — Best Config Per Ticker')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.tick_params(axis='x', rotation=45)

# [0,1] Drawdown
ax = axes[0, 1]
for mk in MODEL_KEYS:
    s = daily_pnl_by_model[mk]
    if len(s) > 0:
        dd = (s.cumsum() - s.cumsum().cummax()) * 100
        ax.fill_between(dd.index, dd.values, 0, alpha=0.3, color=colors[mk])
        ax.plot(dd.index, dd.values, color=colors[mk], linewidth=1.5,
                label=MODELS[mk]['label'])
ax.axhline(0, color='black', linewidth=0.8)
ax.set_ylabel('Drawdown (%)')
ax.set_title('Drawdown from Peak')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.tick_params(axis='x', rotation=45)

# [0,2] Deploy universe Venn
ax = axes[0, 2]
categories = ['v1-only', 'Both', 'v2-only']
counts = [len(v1_only), len(both), len(v2_only)]
bar_colors = [colors['V1'], '#9C27B0', colors['V2']]
bars = ax.bar(categories, counts, color=bar_colors, alpha=0.85, edgecolor='white')
for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            str(count), ha='center', fontweight='bold')
if v1_only:
    ax.text(0, -2, f"avg={deploy_info['V1'].loc[list(v1_only)]['avg_pnl'].mean():+.2%}",
            ha='center', fontsize=8, color=colors['V1'])
if both:
    v1s = deploy_info['V1'].loc[list(both)]['avg_pnl'].mean()
    v2s = deploy_info['V2'].loc[list(both)]['avg_pnl'].mean()
    ax.text(1, -2, f"v1={v1s:+.2%}\nv2={v2s:+.2%}", ha='center', fontsize=7)
if v2_only:
    ax.text(2, -2, f"avg={deploy_info['V2'].loc[list(v2_only)]['avg_pnl'].mean():+.2%}",
            ha='center', fontsize=8, color=colors['V2'])
ax.set_ylabel('# Tickers')
ax.set_title('Deploy Universe Overlap')
ax.grid(axis='y', alpha=0.3)

# [1,0] Per-ticker delta bar chart
ax = axes[1, 0]
comp_sorted = comp_df.sort_values('delta')
bar_colors_list = [colors['V2'] if d > 0 else colors['V1'] for d in comp_sorted['delta']]
ax.barh(range(len(comp_sorted)), comp_sorted['delta'].values * 100,
        color=bar_colors_list, edgecolor='white', linewidth=0.3)
ax.set_yticks(range(len(comp_sorted)))
ax.set_yticklabels(comp_sorted.index.tolist(), fontsize=5)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('v2 - v1 Best PnL Delta (%)')
ax.set_title('Per-Ticker PnL: v2 minus v1')
ax.grid(axis='x', alpha=0.3)

# [1,1] Score distribution comparison (30m)
ax = axes[1, 1]
for mk in MODEL_KEYS:
    d = model_preds[mk]
    scores = d['preds'][:, 1]
    ax.hist(scores, bins=50, alpha=0.5, label=MODELS[mk]['label'],
            color=colors[mk], density=True)
ax.set_xlabel('Predicted Dump Probability (30m)')
ax.set_ylabel('Density')
ax.set_title('Score Distribution (30m horizon)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# [1,2] Scatter: v1 score vs v2 score per ticker-day
ax = axes[1, 2]
scatter_v1 = []
scatter_v2 = []
for tk in sorted(common_tickers_for_signal):
    v1_scores = {}
    v2_scores = {}
    for mk in MODEL_KEYS:
        d = model_preds[mk]
        tid = None
        for i, tn in enumerate(d['ticker_names']):
            if str(tn) == tk:
                tid = i
                break
        if tid is None:
            continue
        mask = d['X_id'] == tid
        for j in range(mask.sum()):
            idx = np.where(mask)[0][j]
            d_int = int(d['date_ints'][idx])
            if mk == 'V1':
                v1_scores[d_int] = float(d['preds'][idx, 1])
            else:
                v2_scores[d_int] = float(d['preds'][idx, 1])
    for d_int in set(v1_scores) & set(v2_scores):
        scatter_v1.append(v1_scores[d_int])
        scatter_v2.append(v2_scores[d_int])

ax.scatter(scatter_v1, scatter_v2, alpha=0.15, s=8, color='#333333')
ax.plot([0, 1], [0, 1], 'r--', alpha=0.5, linewidth=1)
ax.set_xlabel('v1 score (30m)')
ax.set_ylabel('v2 score (30m)')
ax.set_title(f'Score Agreement ({len(scatter_v1):,} ticker-days)')
ax.grid(True, alpha=0.3)
if scatter_v1:
    corr = np.corrcoef(scatter_v1, scatter_v2)[0, 1]
    ax.text(0.05, 0.92, f'r = {corr:.3f}', transform=ax.transAxes,
            fontsize=11, fontweight='bold')

plt.tight_layout()
chart_path = os.path.join(OUTPUT_DIR, 'v1_vs_v2_deep_comparison.png')
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
print(f"\n  Chart saved: {chart_path}")
if _IN_COLAB:
    plt.show()
plt.close()


# ==============================================================================
# VERDICT
# ==============================================================================
total_elapsed = time.time() - t_total
print(f"\n{'=' * W}")
print(f"  VERDICT")
print(f"{'=' * W}")

print(f"\n  Architecture: IDENTICAL (153k params, dual backbone 1m+15m)")
print(f"  Only difference: training samples (v1=50k, v2=150k)")

print(f"\n  Deploy coverage:")
print(f"    v1: {len(v1_tickers)} tickers  |  v2: {len(v2_tickers)} tickers  |  "
      f"v2 added {len(v2_only)}, lost {len(v1_only)}")

if both:
    shared = comp_df[comp_df['category'] == 'both']
    v2_wins = (shared['delta'] > 0).sum()
    print(f"\n  On {len(shared)} shared tickers:")
    print(f"    v2 better: {v2_wins}  |  v1 better: {len(shared) - v2_wins}")
    print(f"    Avg PnL delta: {shared['delta'].mean():+.2%}")

for mk in MODEL_KEYS:
    s = daily_pnl_by_model[mk]
    if len(s) > 0:
        print(f"\n  {MODELS[mk]['label']} portfolio:")
        print(f"    Cumul: {s.cumsum().iloc[-1]:+.2%}  |  "
              f"Avg daily: {s.mean():+.2%}  |  "
              f"WR: {(s > 0).mean():.0%}  |  "
              f"DD: {(s.cumsum() - s.cumsum().cummax()).min():+.2%}")

print(f"\n  Recommendation:")
if len(v2_tickers) > len(v1_tickers) * 1.1:
    v2_cum = daily_pnl_by_model['V2'].cumsum().iloc[-1] if len(daily_pnl_by_model['V2']) > 0 else 0
    v1_cum = daily_pnl_by_model['V1'].cumsum().iloc[-1] if len(daily_pnl_by_model['V1']) > 0 else 0
    if v2_cum > v1_cum:
        print(f"    v2 wins on BOTH coverage and cumulative PnL → USE v2")
    else:
        print(f"    v2 has more coverage but v1 has higher cumul PnL")
        print(f"    → CONSIDER: v1 for concentrated alpha, v2 for diversified portfolio")
        print(f"    → OR: deploy v1 tickers + v2-only tickers (union)")
else:
    print(f"    Coverage similar — pick the model with better per-ticker PnL")

print(f"\n  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
print(f"{'=' * W}")
